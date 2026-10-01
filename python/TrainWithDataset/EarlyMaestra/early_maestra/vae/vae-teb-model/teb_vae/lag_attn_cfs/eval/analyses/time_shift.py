r"""Is the coupling about the source at the right moment, or about this recording's source at all?

The two controls the pass already runs leave one alternative standing. The permutation control
hands a segment **another recording's** source, so it removes the recording and the moment together;
the source-null arm zeroes the source's values, so it removes everything but the availability clock.
Neither can tell a model that reads *what the uterus is doing now* from one that reads *what kind of
uterine activity this patient has* -- a recording-level state that a slowly varying source carries
at any moment. Both would pass the permutation control, and both would clear the clock.

This analysis separates them. Every segment is paired with the **nearest segment of the same
recording that shares no stored sample with it**, and the posterior is posed again on the partner's
source values while everything target-side -- the encoder state, the prior, the anchors, the
persistence input -- stays the segment's own. The source is then the same patient's, from the same
stretch of labour, with the same warm-up geometry (each stored segment carries its own), and wrong in
time by at least one segment's length. Three readings of the same anchors follow:

* $K^{\mathrm{aligned}}$, the source-conditioned KL with the segment's own source (the pass's
  ``source_conditioned_kl_raw``, re-read here on the same support);
* $K^{\mathrm{shifted}}$, the same KL with the partner's source;
* $\Delta g = g^{\mathrm{aligned}} - g^{\mathrm{shifted}}$, the mean-decoded forecast gap under the two
  sources, which is the forecast cost of the misalignment in nats per anchor (the base branch is the
  same in both, so it cancels).

$\Delta K = K^{\mathrm{aligned}} - K^{\mathrm{shifted}}$ is reduced per recording and bootstrapped
over recordings, and the reported verdict ``coupling_is_time_specific`` passes when the lower end of
its interval is above zero. It is reported beside the pre-registered acceptance verdicts, not among
them: the gate is a pre-registered list and this control was added after it.

Mean-decoded, never sampled: the question is whether the source moved the belief, and a draw would
add noise that differs between the two arms for no reason. A segment whose recording has no
non-overlapping partner is counted and excluded, never paired with an overlapping one -- a partner
sharing samples would carry part of the aligned source and understate the difference.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval import traces
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.dataset_rows import check_batch_identity, dataset_index_map, resolve_rows
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry, per_recording_means
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    anchor_support,
    forecast_likelihood_terms,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "time_shift"
PER_SEGMENT_FILENAME = "time_shift_per_segment.csv"
PER_RECORDING_FILENAME = "time_shift_per_recording.csv"
SUMMARY_FILENAME = "time_shift_summary.csv"
FIGURE = "time_shift_control"

#: ``eval_config.caps`` entry bounding how many segments are paired and scored. Absent means
#: :data:`DEFAULT_SEGMENTS`, not every segment: the arm costs one forward per segment.
CAP_NAME = "time_shift"
DEFAULT_SEGMENTS = 512

#: Offset on ``eval_config.seed`` for the capped draw, distinct from every other analysis's.
SEED_OFFSET = 14

#: The reported verdict's name.
VERDICT_NAME = "coupling_is_time_specific"

#: The per-segment readouts, and the two differences the verdict and the figure read.
SEGMENT_COLUMNS = ("kld_aligned", "kld_shifted", "gap_aligned", "gap_shifted")
DIFFERENCES = (
    ("delta_kld", "kld_aligned", "kld_shifted", "nats per step"),
    ("delta_gap", "gap_aligned", "gap_shifted", "nats per anchor"),
)

#: The collected source-null KL, joined in so the figure can show the clock floor beside the arms.
NULL_COLUMN = "kld_source_null"


def stored_segment_seconds(model: Any) -> float:
    """Length of one STORED segment in seconds: the loaded steps plus the trim cut from each end."""
    return float(model.geometry.t) * float(SECONDS_PER_STEP) + 2.0 * float(traces.LOADER_TRIM_S)


def choose_partners(rows: pd.DataFrame, *, min_separation_s: float) -> pd.DataFrame:
    """Pair every segment with the nearest segment of its recording that shares no stored sample.

    Nearest, so the partner is from the same stretch of labour and the arms differ by the timing
    rather than by the patient's state hours apart; ties go to the earlier segment, so the pairing
    is a function of the table alone.

    Args:
        rows: One row per segment with ``guid``, ``epoch`` and ``dataset_index``.
        min_separation_s: The smallest ``|epoch difference|`` at which two stored segments share no
            sample (:func:`stored_segment_seconds`).

    Returns:
        ``rows`` restricted to segments with a partner, plus ``partner_dataset_index``,
        ``partner_epoch`` and ``separation_s``.
    """
    pieces: List[pd.DataFrame] = []
    for _, cell in rows.groupby("guid", sort=True):
        epochs = np.asarray(cell["epoch"], dtype=np.float64)
        distance = np.abs(epochs[:, None] - epochs[None, :])
        eligible = distance >= float(min_separation_s)
        # Ineligible pairs pushed past every eligible one, then the earlier epoch breaks a tie.
        order = np.where(eligible, distance, np.inf) + 1e-9 * (epochs[None, :] - epochs.min())
        choice = order.argmin(axis=1)
        has_partner = eligible.any(axis=1)
        if not has_partner.any():
            continue
        paired = cell[has_partner].copy()
        partner = cell.iloc[choice[has_partner]]
        paired["partner_dataset_index"] = partner["dataset_index"].to_numpy()
        paired["partner_epoch"] = partner["epoch"].to_numpy(dtype=np.float64)
        paired["separation_s"] = np.abs(paired["partner_epoch"] - paired["epoch"]).to_numpy()
        pieces.append(paired)
    if not pieces:
        return rows.head(0).assign(partner_dataset_index=[], partner_epoch=[], separation_s=[])
    return pd.concat(pieces, ignore_index=True)


def _load(loader: Any, indices: Sequence[int]) -> Any:
    """Collate the dataset rows ``indices`` into one batch, in the order given (repeats allowed)."""
    dataset = loader.dataset
    return loader.collate_fn([dataset[int(index)] for index in indices])


@torch.no_grad()
def score_pairs(task: Any, own: Any, partner: Any) -> Dict[str, np.ndarray]:
    """Score one batch of segments under their own source and under their partners' source.

    Args:
        task: The loaded task.
        own: The segments' batch, on the task's device.
        partner: Their partners' batch, positionally paired, on the same device.

    Returns:
        Per-segment ``n_anchors`` and :data:`SEGMENT_COLUMNS`, as float arrays.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, own)
    partner_source = model_inputs(task, partner)[2]
    phase, stride = DENSE_ANCHOR_GEOMETRY
    outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    anchors = outputs["anchor_index"]
    target = model._build_forecast_target(target_features, anchors)
    mask, _coverage, kl_support = anchor_support(model, weight, outputs)
    density = forecast_likelihood_terms(model)

    gated = partner_source if model.source_gate is None else model.source_gate(partner_source)
    # A seeded generator only because the helper also decodes one draw, which nothing here reads;
    # it keeps the global stream untouched.
    generator = torch.Generator(device=gated.device).manual_seed(0)
    shifted = controls.occluded_forward_outputs(model, outputs, gated, anchors=anchors, generator=generator)

    gather = anchors.to(torch.long)[:, :, None].expand(-1, -1, outputs["mu_prior"].shape[-1])
    persistence = outputs.get("persistence")

    def scored(latent: torch.Tensor) -> torch.Tensor:
        mu, logvar = model.decoder(latent.gather(1, gather), persistence=persistence)
        block, _ = masked_raw_block_per_anchor(
            mu, target, mask, likelihood=likelihood, logvar=logvar, **density
        )
        return block

    contributing = (mask.amax(dim=-1) > 0).to(target.dtype)

    def per_segment(values: torch.Tensor, weights: torch.Tensor) -> np.ndarray:
        mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return mean.detach().cpu().to(torch.float64).numpy()

    base = scored(outputs["mu_prior"])
    kld_shifted = model.kld_tensor(
        mu_prior=outputs["mu_prior"], logvar_prior=outputs["logvar_prior"],
        mu_post=shifted["mu_post"], logvar_post=shifted["logvar_post"],
    ).sum(dim=-1)
    return {
        "n_anchors": contributing.sum(dim=1).detach().cpu().to(torch.float64).numpy(),
        "kld_aligned": per_segment(outputs["kld_per_t"], kl_support),
        "kld_shifted": per_segment(kld_shifted, kl_support),
        "gap_aligned": per_segment(base - scored(outputs["mu_post"]), contributing),
        "gap_shifted": per_segment(base - scored(shifted["mu_post"]), contributing),
    }


def summarise(per_recording: pd.DataFrame, *, resamples: int, seed: int) -> pd.DataFrame:
    """Each arm's and each difference's mean over recordings, with a bootstrap interval.

    Args:
        per_recording: One row per recording carrying the arms and the differences.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per quantity: ``quantity``, ``unit``, ``value``, ``ci_lo``, ``ci_hi``, ``n``.
    """
    units = {left: unit for _name, left, _right, unit in DIFFERENCES}
    units.update({right: unit for _name, _left, right, unit in DIFFERENCES})
    units.update({name: unit for name, _left, _right, unit in DIFFERENCES})
    units[NULL_COLUMN] = "nats per step"
    rows = []
    for quantity in (*SEGMENT_COLUMNS, NULL_COLUMN, *(name for name, *_ in DIFFERENCES)):
        if quantity not in per_recording.columns:
            continue
        interval = shared_stats.bootstrap_ci(
            np.asarray(per_recording[quantity], dtype=np.float64), resamples=resamples, seed=seed
        )
        rows.append({
            "quantity": quantity, "unit": units.get(quantity, ""),
            "value": interval.get("point"), "ci_lo": interval.get("lo"),
            "ci_hi": interval.get("hi"), "n": interval.get("n"),
        })
    return pd.DataFrame(rows, columns=["quantity", "unit", "value", "ci_lo", "ci_hi", "n"])


def verdict_record(summary: pd.DataFrame) -> Dict[str, Any]:
    r"""``coupling_is_time_specific``: does the aligned source out-couple the time-shifted one?

    PASS when the lower end of the bootstrap interval of $\Delta K$ is above zero, FAIL when it is
    not, INCONCLUSIVE when there are too few recordings for an interval. The forecast-gap difference
    is carried beside it as evidence, not decided on: the KL is the coupling readout.

    Args:
        summary: :func:`summarise`'s table.

    Returns:
        ``{'name', 'status', 'reason', 'values'}``, the shape of the acceptance verdicts.
    """
    by_quantity = {str(row["quantity"]): row for _, row in summary.iterrows()}
    kld = by_quantity.get("delta_kld")
    gap = by_quantity.get("delta_gap")
    values: Dict[str, Any] = {}
    for prefix, row in (("delta_kld", kld), ("delta_gap", gap)):
        if row is not None:
            values.update({
                f"{prefix}_nats": row["value"], f"{prefix}_ci_lo": row["ci_lo"],
                f"{prefix}_ci_hi": row["ci_hi"], "n_recordings": row["n"],
            })
    lower = None if kld is None else kld["ci_lo"]
    if lower is None or not np.isfinite(float(lower)):
        status, reason = "INCONCLUSIVE", "too few recordings with a non-overlapping partner segment for an interval"
    elif float(lower) > 0.0:
        status, reason = "PASS", "the aligned source moves the posterior more than the same recording's time-shifted source"
    else:
        status, reason = "FAIL", (
            "the same recording's source from another time moves the posterior as much as the aligned "
            "one: the coupling reads a recording-level source state, not the moment"
        )
    return {"name": VERDICT_NAME, "status": status, "reason": reason, "values": values}


def build_figure(per_recording: pd.DataFrame, summary: pd.DataFrame) -> Any:
    """Per recording, the KL under the three sources; and both differences with their intervals.

    Args:
        per_recording: One row per recording.
        summary: :func:`summarise`'s table.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 3, height_per_row=2.8)
    left, middle, right = axes[0, 0], axes[0, 1], axes[0, 2]
    arms = [(NULL_COLUMN, "source null"), ("kld_shifted", "shifted"), ("kld_aligned", "aligned")]
    present = [(column, label) for column, label in arms if column in per_recording.columns]
    classes = per_recording.get(labels.CLASS_COLUMN, pd.Series(index=per_recording.index, dtype=object))
    colours = figures.group_colors(
        cohort.ordered_groups(sorted({str(value) for value in classes.dropna()}), labels.CLASS_COLUMN)
    )
    for guid, row in per_recording.iterrows():
        values = [float(row[column]) for column, _ in present]
        colour = colours.get(str(classes.get(guid)), figures.COLOR_GRAY)
        left.plot(range(len(present)), values, color=colour, linewidth=figures.LINE_THIN,
                  marker="o", markersize=figures.MARKER_SMALL, alpha=0.8)
    left.set_xticks(range(len(present)), [label for _, label in present])
    left.set_ylabel("KL (nats per step)")
    left.set_title("KL by source")
    for name, colour in colours.items():
        left.plot([], [], color=colour, label=name)
    if colours:
        left.legend(loc="upper left", fontsize=figures.FONT_SMALL)
    figures.style_axes(left)

    rows = summary.set_index("quantity") if len(summary) else summary
    # One panel per difference: the two are in different units and three orders apart.
    for ax, (name, _left, _right, unit) in zip((middle, right), DIFFERENCES):
        symbol = r"$\Delta K$" if name == "delta_kld" else r"$\Delta g$"
        ax.set_title(f"{symbol}: aligned minus shifted")
        ax.set_ylabel(unit)
        ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
        if name in per_recording.columns:
            spread = np.asarray(per_recording[name], dtype=np.float64)
            jitter = 0.25 + 0.04 * np.random.default_rng(0).standard_normal(spread.size)
            ax.scatter(jitter, spread, s=6, color=figures.COLOR_GRAY, alpha=0.6, linewidths=0,
                       label="recordings")
        if name in getattr(rows, "index", []):
            record = rows.loc[name]
            ax.errorbar(
                [0.0], [record["value"]],
                yerr=[[record["value"] - record["ci_lo"]], [record["ci_hi"] - record["value"]]],
                fmt="o", color=figures.COLOR_BLUE, capsize=3, linewidth=figures.LINE_REGULAR,
                label="mean, 95% CI",
            )
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.6)
        # The two panels share their keys, so the legend is drawn once, in the first.
        if ax is middle:
            figures.legend_with_headroom(ax, ncol=2)
        figures.style_axes(ax)
    return figure


def _skip(reason: str) -> Dict[str, Any]:
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {"n_samples": None, "composition": {}, "plan": {"capped": True}, "skipped": True,
            "reason": reason, "files": []}


def run_time_shift_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pair, score, reduce over recordings, and report the time-specificity verdict.

    One of the analyses that reads ``context.task`` and ``context.loader``: the arm poses the
    posterior on a source the pass never fed the model, which no table can serve.

    Args:
        context: The analysis context: the collection (for the segments and the collected
            source-null KL), the task and the loader.
        eval_config: The validated block, for the cap, the seed, the bootstrap resamples and the
            delivery-window bound.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the summary rows, the verdict, the pairing census and the files.
    """
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return _skip(
            "the time-shifted arm poses the posterior on another segment's source, so it needs a "
            "model and a loader; this pass built neither, which is what an offline re-run is"
        )
    collection = context.collection
    per_sample = collection.per_sample
    scored = per_sample[per_sample["n_anchors"] > 0] if "n_anchors" in per_sample.columns else per_sample
    scored = cohort.within_horizon(scored, eval_config.get("max_hours_before_delivery"))
    rows = resolve_rows(scored, dataset_index_map(loader))
    model = task.orig_model
    min_separation_s = stored_segment_seconds(model)
    paired = choose_partners(rows, min_separation_s=min_separation_s)
    census = {
        "n_segments_scored": int(len(scored)),
        "n_segments_locatable": int(len(rows)),
        "n_segments_with_partner": int(len(paired)),
        "n_recordings_with_partner": int(paired["guid"].nunique()) if len(paired) else 0,
        "min_separation_s": float(min_separation_s),
    }
    if paired.empty:
        return {**_skip("no recording has two segments that share no stored sample"), "pairing": census}

    cap = int((eval_config.get("caps") or {}).get(CAP_NAME) or DEFAULT_SEGMENTS)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    if len(paired) > cap:
        chosen = np.random.default_rng(seed).choice(len(paired), size=cap, replace=False)
        paired = paired.iloc[np.sort(chosen)].reset_index(drop=True)
    census["n_segments_scored_here"] = int(len(paired))

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, int(getattr(loader, "batch_size", None) or 1))
    records: List[Dict[str, np.ndarray]] = []
    started = time.perf_counter()
    for start in range(0, len(paired), batch_size):
        chunk = paired.iloc[start:start + batch_size]
        own = _load(loader, chunk["dataset_index"].tolist())
        partner = _load(loader, chunk["partner_dataset_index"].tolist())
        check_batch_identity(own, chunk)
        check_batch_identity(partner, chunk.assign(epoch=chunk["partner_epoch"]))
        device = task.device
        records.append(score_pairs(
            task,
            task.transfer_batch_to_device(own, device, dataloader_idx=0),
            task.transfer_batch_to_device(partner, device, dataloader_idx=0),
        ))
    elapsed_s = time.perf_counter() - started

    values = {name: np.concatenate([record[name] for record in records]) for name in records[0]}
    per_segment = paired.reset_index(drop=True).assign(**values)
    for name, left, right, _unit in DIFFERENCES:
        per_segment[name] = per_segment[left] - per_segment[right]
    if NULL_COLUMN in per_sample.columns and "sample_index" in per_segment.columns:
        null = per_sample.set_index("sample_index")[NULL_COLUMN]
        per_segment[NULL_COLUMN] = per_segment["sample_index"].map(null).to_numpy(dtype=np.float64)
    identity = [name for name in ("sample_index", "guid", labels.SUBGROUP_COLUMN, labels.CLASS_COLUMN,
                                  "epoch", "partner_epoch", "separation_s", "n_anchors")
                if name in per_segment.columns]
    measured = [*SEGMENT_COLUMNS, *(name for name, *_ in DIFFERENCES)]
    if NULL_COLUMN in per_segment.columns:
        measured.append(NULL_COLUMN)
    per_segment = per_segment[identity + measured]
    per_segment.to_csv(directory / PER_SEGMENT_FILENAME, index=False)

    per_recording = per_recording_means(per_segment, measured)
    per_recording.to_csv(directory / PER_RECORDING_FILENAME)
    summary = summarise(
        per_recording,
        resamples=int(eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES)),
        seed=int(eval_config.get("seed", 0)),
    )
    summary.to_csv(directory / SUMMARY_FILENAME, index=False)
    figure_name = str(figures.render_figure(build_figure(per_recording, summary), directory / FIGURE).name)
    verdict = verdict_record(summary)
    logger.info(
        f"{ANALYSIS_DIRNAME}: {verdict['status']} -- {verdict['reason']} "
        f"({len(per_segment)} segment(s), {len(per_recording)} recording(s), {elapsed_s:.1f} s)"
    )
    return {
        "n_samples": int(len(per_segment)),
        "composition": {"n_recordings": int(len(per_recording))},
        "plan": {"capped": True, "cap": cap, "seed": seed, "decode": "latent mean, both arms"},
        "pairing": census,
        "cost": {"elapsed_s": float(elapsed_s), "n_segments": int(len(per_segment)),
                 "seconds_per_segment": float(elapsed_s / max(1, len(per_segment)))},
        "summary": summary.to_dict(orient="records"),
        "verdict": verdict,
        "headline": {
            key: verdict["values"].get(key)
            for key in ("delta_kld_nats", "delta_kld_ci_lo", "delta_kld_ci_hi", "delta_gap_nats")
        },
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, ("delta_kld", "delta_gap"))
        ],
        "files": [PER_SEGMENT_FILENAME, PER_RECORDING_FILENAME, SUMMARY_FILENAME, figure_name],
    }
