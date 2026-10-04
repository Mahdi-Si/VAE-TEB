r"""``delay_map`` -- new; owner E1-O (``notes/EVAL_PLAN.md`` §2, ``EVAL_MAP_LAG.md`` N4).

**Question.** At which UP→FHR delay does removing UP cost the forecast? The lag map cannot answer
this. A delay of ``d`` tokens is informative at every lag in ``[d − H, d − 1]``, one horizon step
each, so the lag axis alone smears a delay over up to ``H`` lags. The **interventional** answer
lives on the diagonals of the lag × horizon plane.

**Computation.**
* For every single lag ``ℓ ∈ [0, L − 1]`` (bands of width one), apply the ``baseline`` arm of
  ``occlusion`` (D5). The raw UP of token ``a − ℓ`` is set to the segment's resting tone and the
  stream is re-patchified.
* Score the block at the scored anchor ``a``, resolved by horizon step ``τ``, against an untouched
  reference under the same latent noise. These are the shared occlusion's anchor seeds, pairing and
  ``_horizon_scores``.
* This gives the field ``Δ(ℓ, τ)``, in nats per anchor per horizon step; positive means the forecast
  needed that UP.
* Several anchors per segment (``caps.delay_map_anchors``), each in its own row, so no anchor reads
  another's edit. Anchors are drawn among those whose whole lag window lies inside the segment
  (``a ≥ L − 1``), so every cell of the field has the same support.
* Rows are averaged within a recording, then across recordings (the house aggregation chain).
* The **delay curve** is ``Δ(d)``, the mean of the field over the diagonal ``ℓ + 1 + τ = d``, in
  tokens of 4 s. The sum over the diagonal travels beside it with its cell count, because diagonals
  near the corners hold fewer cells. Intervals are percentile bootstraps over recordings.

**Reading.** The curve's argmax ``d*`` is the delay the forecast uses. On ``planted.yaml`` the
shard couples UP to FHR at 45 tokens (180 s), so ``d* ≈ 45`` means the model found the plant. A
curve that is flat (or flat after its first few delays) means the forecast does not use UP timing.
This analysis has no verdict; the reading is the user's.

**Costs.**
* Segments: ``caps.delay_map_segments`` (default :data:`DEFAULT_SEGMENTS`).
* Anchors per segment: ``caps.delay_map_anchors`` (default :data:`DEFAULT_ANCHORS`).
* One forward per batch, then ``1 + L`` arms over ``rows = B · anchors``. Each arm is re-patchify,
  adapter, lag attention and posterior over the segment, then one single-anchor decode.
* The ``cost`` block records ``seconds_per_arm_row``.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.analyses import occlusion as _occlusion
from teb_vae.lag_attn_cfs.eval.attributions import signed_log_norm, symlog_legend
from teb_vae.lag_attn_cfs.eval.frames import per_recording_labels
from teb_vae.lag_attn_cfs.eval.metrics import batch_guids, batch_size_of
from teb_vae.lag_attn_rws.nets.controls import occluded_forward_outputs
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_transformer_patch.eval import figures as patch_figures
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

#: ``(headline_name, key)`` pairs registered as ``("delay_map", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("delay_map_argmax_steps", "argmax_delay_steps"),
    ("delay_map_argmax_s", "argmax_delay_s"),
    ("delay_map_peak_nats", "peak_delta_nats"),
)

ANALYSIS_DIRNAME = "delay_map"
FIELD_FILENAME = "delay_map_field.csv"
CURVE_FILENAME = "delay_map_curve.csv"
PER_RECORDING_FILENAME = "delay_map_per_recording.csv"
FIGURE = "delay_map"

#: ``eval_config.caps`` names and their defaults.
CAP_SEGMENTS, DEFAULT_SEGMENTS = "delay_map_segments", 128
CAP_ANCHORS, DEFAULT_ANCHORS = "delay_map_anchors", 16

#: Its own seed offset (the shared analyses use 11-14), so its draw is its own stream.
_SEED_OFFSET = 21

UNIT = "nats per anchor per horizon step"


def delay_curve(field: np.ndarray) -> Dict[str, np.ndarray]:
    """Reduce an ``(L, H)`` field onto the delay ``d = ℓ + 1 + τ`` (tokens), ``d = 1 … L + H − 1``.

    Returns:
        ``delay``, ``mean`` (over the diagonal's finite cells), ``sum`` and ``n_cells``; NaN where a
        diagonal has no finite cell.
    """
    n_lags, horizon = field.shape
    delay = (np.arange(n_lags)[:, None] + 1 + np.arange(horizon)[None, :]).ravel()
    values = np.asarray(field, dtype=np.float64).ravel()
    finite = np.isfinite(values)
    delays = np.arange(1, n_lags + horizon)
    total = np.bincount(delay[finite] - 1, weights=values[finite], minlength=delays.size)
    count = np.bincount(delay[finite] - 1, minlength=delays.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return {"delay": delays, "mean": mean, "sum": np.where(count > 0, total, np.nan),
            "n_cells": count}


def _draw_columns(valid: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    """``count`` anchor columns per row among the ``valid`` ones, without replacement where possible."""
    weights = valid.to(torch.float32)
    weights = torch.where(weights.sum(1, keepdim=True) > 0, weights, torch.ones_like(weights))
    replacement = bool((weights > 0).sum(1).min().item() < count)
    return torch.multinomial(weights, count, replacement=replacement, generator=generator)


@torch.no_grad()
def collect_batch(
    task: Any, batch: Any, *, anchors_per_segment: int, seed: int, batch_index: int
) -> Dict[str, Any]:
    """The ``(rows, L, H)`` single-lag ``baseline`` occlusion field for one batch."""
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    fhr, up, weight = raw.batch_signals(batch)
    steps_per_token = int(model.raw_per_step)
    n_lags = int(model.lag_attn.L)
    repeat = int(anchors_per_segment)
    outputs = raw.forward_raw(model, fhr, up, weight)

    device = fhr.device
    batch_seed = int(seed) + _SEED_OFFSET + _occlusion._BATCH_SEED_STRIDE * int(batch_index)
    generator = torch.Generator(device=device)
    generator.manual_seed(batch_seed + _occlusion._SEED_OFFSET_ANCHOR)
    # The whole lag window inside the segment, so every cell of the field has the same support.
    full_window = outputs["anchor_valid"].to(torch.bool) & (outputs["anchor_index"] >= n_lags - 1)
    columns = _draw_columns(full_window, repeat, generator).reshape(-1, 1)  # (B·K, 1)

    scored, anchors, anchor_valid = raw.at_columns(outputs, columns, repeat=repeat)
    fhr, up, weight = (x.repeat_interleave(repeat, dim=0) for x in (fhr, up, weight))
    summaries = model.summary_target(fhr, weight)
    _, u_patch = raw.patch_streams(model, fhr, up, weight)
    tone = raw.resting_tone(up, weight, raw_per_step=steps_per_token, validity=model.source_validity)
    fill = raw.occlusion_fill("baseline", up, tone)

    def score(source: torch.Tensor) -> torch.Tensor:
        noise = torch.Generator(device=device)
        noise.manual_seed(batch_seed + _occlusion._SEED_OFFSET_NOISE)
        arm = occluded_forward_outputs(model, scored, source, anchors=anchors, generator=noise)
        return _occlusion._horizon_scores(
            model, arm, target_features=summaries, weight=weight, anchors=anchors,
            anchor_valid=anchor_valid, likelihood=likelihood,
        )

    reference = score(u_patch)
    mask, _ = forecast_mask(model.scored_weight(weight), model.geometry,
                            coverage_floor=model.coverage_floor, anchors=anchors,
                            anchor_valid=anchor_valid)
    scored_step = mask.reshape(mask.shape[0], mask.shape[2], -1).amax(-1) > 0  # (rows, H)
    field = torch.full((reference.shape[0], n_lags, reference.shape[1]), float("nan"),
                       dtype=torch.float64, device=device)
    for lag in range(n_lags):
        tokens = _occlusion.band_mask(anchors[:, 0], (lag, lag), int(u_patch.shape[1]))
        edited = raw.replace_tokens(up, tokens, fill, raw_per_step=steps_per_token)
        delta = score(raw.patch_streams(model, fhr, edited, weight)[1]) - reference
        # A step the forecast mask dropped scored nothing on either side: blank, not a zero.
        field[:, lag] = torch.where(scored_step, delta, torch.full_like(delta, float("nan")))
    guids = batch_guids(batch, batch_size_of(batch))
    return {
        "guids": [guid for guid in guids for _ in range(repeat)],
        "anchors": anchors[:, 0].cpu().numpy(),
        "field": field.cpu().numpy(),
    }


def _bootstrap_curve(curves: np.ndarray, *, resamples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-delay percentile interval of the mean over recordings (``curves``: recordings × delays)."""
    low, high = np.full(curves.shape[1], np.nan), np.full(curves.shape[1], np.nan)
    for column in range(curves.shape[1]):
        values = curves[:, column]
        interval = shared_stats.bootstrap_ci(values[np.isfinite(values)], resamples=resamples, seed=seed)
        low[column], high[column] = interval["lo"], interval["hi"]
    return low, high


def build_figure(field: np.ndarray, curve: pd.DataFrame, argmax: Optional[int]) -> Any:
    """Two panels: the lag × horizon field (symmetric-log, blank where unscored) and the delay curve."""
    figure, axes = figures.new_figure(2, height_per_row=2.6)
    ax = axes[0, 0]
    finite = np.abs(field[np.isfinite(field)])
    limit = float(np.percentile(finite, 99)) if finite.size and finite.max() > 0 else 1.0
    norm = (signed_log_norm(field) if patch_figures.spans_decades(field) else None) or \
        mcolors.Normalize(vmin=-limit, vmax=limit)
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("white")  # unscored cells are blank, never a colour
    image = ax.imshow(field, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                      interpolation="none")
    if argmax is not None:  # the iso-delay diagonal of the curve's argmax
        taus = np.arange(field.shape[1])
        ax.plot(taus, argmax - 1 - taus, color="0.1", linewidth=figures.LINE_REGULAR, linestyle="--",
                label=f"ℓ + 1 + τ = {argmax}")
        ax.set_ylim(-0.5, field.shape[0] - 0.5)
        ax.legend(loc="upper right", fontsize=figures.FONT_TINY)
    ax.set_xlabel("horizon step τ (4 s)")
    ax.set_ylabel("lag ℓ (4 s tokens)")
    ax.set_title("Forecast cost of removing one UP token (baseline fill), by lag and horizon")
    colorbar = figure.colorbar(image, ax=ax, fraction=0.03, pad=0.015)
    colorbar.set_label(UNIT, fontsize=figures.FONT_TINY)

    ax = axes[1, 0]
    seconds = np.asarray(curve["delay_s"], dtype=np.float64)
    mean = np.asarray(curve["delta_mean_nats"], dtype=np.float64)
    ax.fill_between(seconds, curve["ci_lo"], curve["ci_hi"], color=figures.COLOR_LIGHT_GRAY,
                    linewidth=0, label="95% bootstrap over recordings")
    ax.plot(seconds, mean, color=figures.COLOR_BLUE, linewidth=figures.LINE_REGULAR,
            label="mean over the diagonal")
    ax.axhline(0.0, linestyle="--", linewidth=figures.LINE_THIN, color="0.4")
    if argmax is not None:
        ax.axvline(argmax * SECONDS_PER_STEP, color=figures.COLOR_VERMILLION,
                   linewidth=figures.LINE_THIN, label=f"argmax d = {argmax} ({argmax * SECONDS_PER_STEP:g} s)")
    ax.set_xlabel("UP→FHR delay d = 4(ℓ + 1 + τ) s")
    ax.set_ylabel(UNIT)
    ax.set_title("Interventional delay curve")
    if patch_figures.spans_decades(mean):
        symlog_legend(ax, mean, np.asarray(curve["ci_lo"]), np.asarray(curve["ci_hi"]), ncol=3)
    else:
        ax.legend(loc="upper left", ncol=3, fontsize=figures.FONT_TINY)
    figures.caveat_note(figure, "Lag l = UP patch l tokens before the anchor patch; no filter delay.")
    return figure


def run_delay_map_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the single-lag ``baseline`` occlusion field and reduce it onto the delay axis."""
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return skip_record(ANALYSIS_DIRNAME, "the field re-encodes an edited UP stream, so it needs a "
                                             "model and a loader; an offline re-run has neither")
    caps = eval_config.get("caps") or {}
    cap = int(caps.get(CAP_SEGMENTS) or DEFAULT_SEGMENTS)
    repeat = int(caps.get(CAP_ANCHORS) or DEFAULT_ANCHORS)
    seed = int(eval_config.get("seed", 0))
    resamples = int(eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES))
    model = task.orig_model
    n_lags, horizon = int(model.lag_attn.L), int(model.horizon)

    records: List[Dict[str, Any]] = []
    n_segments = n_batches = 0
    started = time.perf_counter()
    for batch in loader:
        if n_segments >= cap:
            break
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        records.append(collect_batch(task, moved, anchors_per_segment=repeat, seed=seed,
                                     batch_index=n_batches))
        n_segments += batch_size_of(batch)
        n_batches += 1
    elapsed_s = time.perf_counter() - started
    if not records:
        return skip_record(ANALYSIS_DIRNAME, "the loader yielded no batch")

    guids = np.asarray([guid for record in records for guid in record["guids"]])
    fields = np.concatenate([record["field"] for record in records], axis=0)  # (rows, L, H)
    recordings = sorted(set(guids.tolist()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN cell stays NaN
        per_recording = np.stack([np.nanmean(fields[guids == guid], axis=0) for guid in recordings])
        population = np.nanmean(per_recording, axis=0)  # (L, H)
    curve = delay_curve(population)
    curves = np.stack([delay_curve(item)["mean"] for item in per_recording])
    low, high = _bootstrap_curve(curves, resamples=resamples, seed=seed)
    finite = np.isfinite(curve["mean"])
    argmax = int(curve["delay"][np.nanargmax(curve["mean"])]) if finite.any() else None

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    lag_grid, tau_grid = np.meshgrid(np.arange(n_lags), np.arange(horizon), indexing="ij")
    pd.DataFrame({
        "lag": lag_grid.ravel(), "horizon_step": tau_grid.ravel(),
        "delay_steps": (lag_grid + 1 + tau_grid).ravel(),
        "delay_s": ((lag_grid + 1 + tau_grid) * SECONDS_PER_STEP).ravel(),
        "delta_nats": population.ravel(),
        "n_recordings": np.isfinite(per_recording).sum(axis=0).ravel(),
    }).to_csv(directory / FIELD_FILENAME, index=False)
    curve_frame = pd.DataFrame({
        "delay_steps": curve["delay"], "delay_s": curve["delay"] * SECONDS_PER_STEP,
        "delta_mean_nats": curve["mean"], "ci_lo": low, "ci_hi": high,
        "delta_sum_nats": curve["sum"], "n_cells": curve["n_cells"],
    })
    curve_frame.to_csv(directory / CURVE_FILENAME, index=False)
    labels = per_recording_labels(context.collection.per_sample)
    recording_rows = []
    for guid, item, own in zip(recordings, per_recording, curves):
        usable = np.isfinite(own)
        recording_rows.append({
            "guid": guid,
            "n_rows": int((guids == guid).sum()),
            "argmax_delay_steps": int(curve["delay"][np.nanargmax(own)]) if usable.any() else None,
            "peak_delta_nats": float(np.nanmax(own)) if usable.any() else float("nan"),
            "total_delta_nats": float(np.nansum(item)),
            **({k: labels.at[guid, k] for k in labels.columns} if guid in getattr(labels, "index", []) else {}),
        })
    pd.DataFrame(recording_rows).to_csv(directory / PER_RECORDING_FILENAME, index=False)
    figure_name = str(figures.render_figure(
        build_figure(population, curve_frame, argmax), directory / FIGURE
    ).name)

    n_rows, n_arms = int(fields.shape[0]), 1 + n_lags
    lag_profile = np.nansum(population, axis=1)
    return {
        "n_samples": int(n_segments),
        "composition": {"n_recordings": len(recordings), "n_rows": n_rows},
        "plan": {
            "capped": n_segments >= cap, "cap": cap, "anchors_per_segment": repeat, "seed": seed,
            "arm": "baseline (resting-tone fill of one UP token, re-patchified)",
            "lags": [0, n_lags - 1], "horizon": horizon,
            "anchor_rule": "anchors with the whole lag window inside the segment (a >= L - 1)",
        },
        "cost": {
            "elapsed_s": float(elapsed_s), "n_batches": n_batches, "n_rows": n_rows, "n_arms": n_arms,
            "seconds_per_arm_row": float(elapsed_s / max(1, n_rows * n_arms)),
            "note": "seconds = segments * anchors * (1 + L) * seconds_per_arm_row",
        },
        "unit": UNIT,
        "delay_unit": "tokens of 4 s; d = l + 1 + tau",
        "headline": {
            "argmax_delay_steps": argmax,
            "argmax_delay_s": None if argmax is None else float(argmax * SECONDS_PER_STEP),
            "peak_delta_nats": float(np.nanmax(curve["mean"])) if finite.any() else float("nan"),
        },
        "lag_profile_argmax": int(np.nanargmax(lag_profile)) if np.isfinite(lag_profile).any() else None,
        "files": [FIELD_FILENAME, CURVE_FILENAME, PER_RECORDING_FILENAME, figure_name],
    }
