r"""``occlusion`` -- port; owner E1-O (``notes/EVAL_PLAN.md`` §2, decision D5).

**Question.** What is the forecast worth without the UP at a band of lags? There are three arms, each
scored against an untouched reference at the same anchor and under the same latent noise:

* ``baseline`` (**primary**): every raw UP sample of the band's tokens is set to the segment's
  resting tone (:func:`raw.resting_tone`), then the stream is re-patchified. This is the
  "no contraction here" intervention. Delta is 0 inside the band and consistent at its edges.
* ``zero``: the same fill at 0.0, the loader's z-mean. This is CFS-comparable ("flat at mean UP"),
  and it is not "no contraction": the z-mean sits above resting tone.
* ``missing``: the band's samples become NaN, so ``patchify`` marks the tokens invalid and the model
  reads its learned ``missing`` embedding. It is **unreliable**: under ``source_validity: finite``
  that embedding is rarely trained.

In every arm an invalid sample stays invalid, so an edit never turns a gap into signal.

Everything representation-agnostic is the shared analysis's, imported:
* the band mask (lags relative to each segment's own scored anchor);
* the one seeded anchor per segment and its seeds;
* the paired re-encode/re-pose/re-decode (``controls.occluded_forward_outputs`` with
  ``occlusion=None`` on the edited stream);
* the per-horizon block score;
* the frames, summary, precision, cost record, clinical-clock join and figures.

So a band delta is computed exactly as in CFS.

Two things are this port's own:
* the edit is made on **raw** UP and re-patchified;
* the **live fraction** is read off the validity channel, as the share of band tokens (out of rows
  × band width, counting tokens before the segment start as absent) that held valid UP. The shared
  analysis counted non-zero channels, which reads 32/33 on a fully valid patch.

Column naming: the primary arm keeps the shared name ``occlusion_delta_<band>_nats``, so
``lag_high_kl``'s occlusion-consistency join reads the ``baseline`` arm. The other two arms are
``occlusion_delta_<band>__zero_nats`` and ``occlusion_delta_<band>__missing_nats``.

There is no availability announcement on a patch model, so ``announcement_invariance`` is NaN by
construction.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.analyses import occlusion as _shared
from teb_vae.lag_attn_cfs.eval.attributions import symlog_legend
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry, per_recording_means
from teb_vae.lag_attn_cfs.eval.metrics import batch_guids, batch_size_of
from teb_vae.lag_attn_rws.nets.controls import occluded_forward_outputs
from teb_vae.lag_attn_transformer_patch.eval import figures as patch_figures
from teb_vae.lag_attn_transformer_patch.eval import raw

#: ``(headline_name, key)`` pairs registered as ``("occlusion", "headline", key)``. The four shared
#: ``occlusion_peak_band*`` entries already resolve on this module's ``headline`` (primary arm);
#: these add the CFS-comparable arm beside it.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("occlusion_zero_peak_band", "zero_band"),
    ("occlusion_zero_peak_band_delta_nats", "zero_delta_total_nats"),
)

#: The D5 arms, primary first.
ARMS: Tuple[str, ...] = ("baseline", "zero", "missing")
PRIMARY_ARM = "baseline"

#: What each arm's fill means, carried into the record.
ARM_MEANING: Dict[str, str] = {
    "baseline": "raw UP in the band set to the segment's resting tone (10th percentile of its valid "
                "samples), re-patchified: no contraction at these lags (PRIMARY)",
    "zero": "raw UP in the band set to 0.0, the loader z-mean, re-patchified: flat at mean UP "
            "(CFS-comparable; above resting tone, so not 'no contraction')",
    "missing": "raw UP in the band set to NaN, so patchify marks it invalid and the model reads its "
               "learned missing token. UNRELIABLE: missing is rarely trained under source_validity "
               "finite",
}

PATCH_CAVEAT = (
    "the occlusion edits raw UP inside the band and re-patchifies; an invalid sample stays invalid in "
    "every arm. It is measured at one anchor per segment, because a band occluded relative to one "
    "anchor is read at other lags by every other anchor. A band with a small live fraction held "
    "little valid UP (or lay before the segment start), so its delta says little about the source. "
    "The 'missing' arm is unreliable: the learned missing token is rarely trained."
)


def arm_band_name(band: str, arm: str) -> str:
    """The flat name one (band, arm) pair travels under; the primary arm keeps the band's own."""
    return band if arm == PRIMARY_ARM else f"{band}__{arm}"


@torch.no_grad()
def collect_batch(
    task: Any,
    batch: Any,
    *,
    bands: Dict[str, Tuple[int, int]],
    seed: int,
    batch_index: int = 0,
    arms: Tuple[str, ...] = ARMS,
) -> Dict[str, Any]:
    """Score the reference and every (band, arm) at one seeded anchor per segment.

    The anchor draw and the noise seeds are the shared analysis's, to the offset. The record has the
    shared ``collect_batch``'s shape, with one delta per ``arm_band_name``, so the shared frame
    builders reduce it unchanged.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    fhr, up, weight = raw.batch_signals(batch)
    steps_per_token = int(model.raw_per_step)
    outputs = raw.forward_raw(model, fhr, up, weight)
    _, u_patch = raw.patch_streams(model, fhr, up, weight)
    summaries = model.summary_target(fhr, weight)

    device = fhr.device
    batch_seed = int(seed) + _shared._BATCH_SEED_STRIDE * int(batch_index)
    anchor_generator = torch.Generator(device=device)
    anchor_generator.manual_seed(batch_seed + _shared._SEED_OFFSET_ANCHOR)
    columns = _shared.choose_anchors(outputs["anchor_valid"], anchor_generator)
    scored, anchors, anchor_valid = raw.at_columns(outputs, columns[:, None])

    def score(source: torch.Tensor) -> torch.Tensor:
        noise = torch.Generator(device=device)
        noise.manual_seed(batch_seed + _shared._SEED_OFFSET_NOISE)
        arm = occluded_forward_outputs(
            model, scored, source, anchors=anchors, generator=noise
        )
        return _shared._horizon_scores(
            model, arm, target_features=summaries, weight=weight, anchors=anchors,
            anchor_valid=anchor_valid, likelihood=likelihood,
        )

    reference = score(u_patch)
    tone = raw.resting_tone(
        up, weight, raw_per_step=steps_per_token, validity=model.source_validity
    )
    validity = u_patch[..., -1] + 1.0  # (B, T): 1 valid, 0 invalid
    deltas: Dict[str, torch.Tensor] = {}
    live: Dict[str, float] = {}
    for band, span in bands.items():
        tokens = _shared.band_mask(anchors[:, 0], span, int(u_patch.shape[1]))
        slots = float(tokens.shape[0] * (int(span[1]) - int(span[0]) + 1))
        live_fraction = float((validity * tokens).sum().item()) / slots
        for arm in arms:
            edited = raw.replace_tokens(
                up, tokens, raw.occlusion_fill(arm, up, tone), raw_per_step=steps_per_token
            )
            name = arm_band_name(band, arm)
            deltas[name] = score(raw.patch_streams(model, fhr, edited, weight)[1]) - reference
            live[name] = live_fraction

    return {
        "guids": batch_guids(batch, batch_size_of(batch)),
        "epochs": _shared._epochs(batch),
        "reference": reference.cpu().numpy(),
        "deltas": {name: value.cpu().numpy() for name, value in deltas.items()},
        "live_fraction": live,
        "announcement_max_abs_change": float("nan"),
        "anchors": anchors[:, 0].detach().cpu().numpy(),
        "tone": tone.detach().cpu().to(torch.float64).numpy(),
    }


def build_arm_figure(per_horizon: pd.DataFrame, bands: Dict[str, Tuple[int, int]]) -> Any:
    """One panel per arm: each band's delta against the horizon step, symmetric-log, zero ruled."""
    figure, axes = figures.new_figure(len(ARMS), height_per_row=2.0)
    for row, arm in enumerate(ARMS):
        axis = axes[row, 0]
        curves: List[np.ndarray] = []
        for band in bands:
            name = arm_band_name(band, arm)
            subset = per_horizon[per_horizon["band"] == name].sort_values("horizon_step") \
                if not per_horizon.empty else per_horizon
            if subset.empty:
                continue
            values = np.asarray(subset["delta_nats"], dtype=np.float64)
            curves.append(values)
            axis.plot(
                np.asarray(subset["horizon_step"]), values, linewidth=figures.LINE_REGULAR,
                label=f"{band} [{bands[band][0]}-{bands[band][1]}] "
                      f"(live {float(subset['live_fraction'].iloc[0]):.2f})",
            )
        axis.axhline(0.0, linestyle="--", linewidth=figures.LINE_THIN, color="0.4")
        flag = " (unreliable)" if arm == "missing" else (" (primary)" if arm == PRIMARY_ARM else "")
        axis.set_title(f"{arm}{flag}: forecast cost of band removal")
        axis.set_ylabel("nats per anchor per step")
        if curves and patch_figures.spans_decades(*curves):
            symlog_legend(axis, *curves, ncol=2)
        elif curves:
            axis.legend(loc="upper left", ncol=2, fontsize=figures.FONT_TINY)
    axes[-1, 0].set_xlabel("horizon step τ (4 s)")
    figures.caveat_note(figure, "Lag l = UP patch l tokens before the anchor patch; no filter delay.")
    return figure


def run_occlusion_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score every configured band under the three D5 arms, resolved by horizon step.

    ``eval_config.occlusion_bands`` names the bands; ``caps.occlusion`` bounds the segments (absent
    means every segment, as in CFS). Needs ``context.task`` and ``context.loader`` and records a skip
    without them.
    """
    del probe
    configured = eval_config.get("occlusion_bands") or {}
    bands = {str(name): (int(span[0]), int(span[1])) for name, span in configured.items()}
    if not bands:
        return _shared._skip("eval_config.occlusion_bands names no band, so there is no intervention "
                             "to score")
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return _shared._skip("the occlusion arms re-encode an edited source stream, so they need a "
                             "model and a loader; an offline re-run has neither")

    directory = Path(output_dir) / _shared.ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    cap = (eval_config.get("caps") or {}).get(_shared.CAP_NAME)
    seed = int(eval_config.get("seed", 0))
    resamples = int(eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES))

    records: List[Dict[str, Any]] = []
    n_scored = n_batches = 0
    started = time.perf_counter()
    for batch in loader:
        if cap is not None and n_scored >= int(cap):
            break
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        record = collect_batch(task, moved, bands=bands, seed=seed, batch_index=n_batches)
        records.append(record)
        n_scored += len(record["guids"])
        n_batches += 1
    elapsed_s = time.perf_counter() - started

    flat = {arm_band_name(band, arm): span for band, span in bands.items() for arm in ARMS}
    primary = {band: span for band, span in bands.items()}  # arm_band_name(band, PRIMARY_ARM) == band
    scored, per_horizon = _shared.build_frames(records, flat)
    tones = np.concatenate([record["tone"] for record in records]) if records else np.zeros(0)
    if not scored.empty:
        scored["resting_tone_z"] = tones
    per_sample, clock_census = _shared.join_collected(scored, getattr(context, "collection", None))
    per_sample.to_csv(directory / _shared.PER_SEGMENT_FILENAME, index=False)
    columns = [_shared._band_column(name) for name in flat]
    per_guid = (
        per_recording_means(per_sample, ["reference_block_nats", *columns])
        if not per_sample.empty else pd.DataFrame()
    )
    summary = _shared.build_summary(
        per_horizon, flat, per_recording=per_guid, n_segments=len(per_sample),
        resamples=resamples, seed=seed,
    )
    if not summary.empty:
        summary.insert(1, "arm", [name.split("__")[1] if "__" in name else PRIMARY_ARM
                                  for name in summary["band"]])
        summary.insert(2, "lag_band", [name.split("__")[0] for name in summary["band"]])
        summary["reliable"] = summary["arm"] != "missing"
    per_horizon.to_csv(directory / _shared.PER_HORIZON_FILENAME, index=False)
    summary.to_csv(directory / _shared.SUMMARY_FILENAME, index=False)
    per_guid.to_csv(directory / _shared.PER_RECORDING_FILENAME)

    figure_name = str(figures.render_figure(
        build_arm_figure(per_horizon, bands), directory / _shared.HORIZON_FIGURE
    ).name)
    clocks = _shared.clock_frames(per_sample if clock_census["joined"] else pd.DataFrame(), primary)
    clock_census["n_rows"] = int(len(clocks))
    clocks.to_csv(directory / _shared.CLOCK_FILENAME, index=False)
    clock_figure = str(figures.render_figure(
        _shared.build_clock_figure(clocks, primary, delay_steps=0), directory / _shared.CLOCK_FIGURE
    ).name)

    def arm_summary(arm: str) -> pd.DataFrame:
        if summary.empty:
            return summary
        cut = summary[summary["arm"] == arm].copy()
        cut["band"] = cut["lag_band"]
        return cut

    headline = _shared.headline_record(arm_summary(PRIMARY_ARM))
    zero = _shared.headline_record(arm_summary("zero"))
    headline.update(zero_band=zero["band"], zero_delta_total_nats=zero["delta_total_nats"],
                    arm=PRIMARY_ARM)
    return {
        "n_samples": int(len(per_sample)),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {
            "capped": cap is not None and n_scored >= int(cap),
            "cap": None if cap is None else int(cap),
            "seed": seed,
            "bands": {name: [int(span[0]), int(span[1])] for name, span in bands.items()},
            "arms": list(ARMS),
            "primary_arm": PRIMARY_ARM,
            "anchors_per_segment": 1,
            "implementation": "patch port (E1-O): raw-UP arms over the shared band machinery",
        },
        "cost": _shared.cost_record(
            elapsed_s=elapsed_s, n_batches=n_batches, n_samples=n_scored,
            n_arms=1 + len(flat), device=getattr(task, "device", None),
        ),
        "unit": _shared.NATS_PER_ANCHOR_STEP,
        "arms": ARM_MEANING,
        "bands": summary.to_dict(orient="records"),
        "headline": headline,
        "announcement_invariance": {
            "max_abs_change": float("nan"), "n_batches_checked": 0,
            "meaning": "the patch source adapter builds no availability announcement, so there is "
                       "nothing to hold fixed",
        },
        "clocks": clock_census,
        "caveat": PATCH_CAVEAT,
        "grouped_frames": [grouped_frame_entry(
            _shared.ANALYSIS_DIRNAME, _shared.PER_RECORDING_FILENAME,
            tuple(_shared._band_column(name) for name in primary),
        )],
        "files": [
            _shared.PER_SEGMENT_FILENAME, _shared.PER_RECORDING_FILENAME,
            _shared.PER_HORIZON_FILENAME, _shared.SUMMARY_FILENAME, _shared.CLOCK_FILENAME,
            figure_name, clock_figure,
        ],
    }
