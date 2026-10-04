r"""``channel_skill`` -- the forecast resolved by target channel and by lead time; new, owner E1-F.

It replaces ``spectral_skill``, which needs a filter-frequency channel axis. The patch target has two
named channels, ``level`` and ``variability`` (``notes/EVAL_PLAN.md`` §2).

**Question.** How much does each channel carry of the source gain, how good is each in clinical
units, and up to what lead time does the level forecast beat persistence?

Two readings:

* **Per channel, whole split.** The collection's per-channel vectors ``gap_per_channel`` and
  ``sq_error_per_channel_{base,full}``, reduced per recording and bootstrapped over recordings:
  - the per-channel gap ``pred_gap_level`` / ``pred_gap_variability`` (nats). Asserted to
    recompose to ``pred_gap``.
  - the error-space skill of full against base.
  - the level RMSE in bpm. The variability RMSE is a log-ratio, so it is reported as a factor on
    rms Δ.

  The column names keep ``spectral_skill``'s ``pred_gap_<x>`` pattern, so ``pred_gap_variability``
  is the per-recording metric ``cross_subgroup`` registers.
* **Per channel and lead, retained segments** (``caps.waveforms``). The RMSE of base, full and the
  shared baselines at every lead ``4(τ+1)`` s. The baselines are persistence, the segment's running
  mean and climatology (``metrics.baseline_forecasts`` on the patch summary target rebuilt from the
  retained raw FHR, under the same forecast mask). From these comes the skill against persistence,
  ``1 − MSE/MSE_persist``, and the **crossover lead**: the first lead at which the level forecast's
  skill against persistence changes sign. Usually it starts below persistence and overtakes it, or
  starts above it and stops beating it; the record names which. The crossover is bootstrapped over
  recordings and broken down by class.

Reductions follow the pipeline's chain: per segment (anchor-mean per lead), then per recording,
then across recordings. The bootstrap resamples recordings. The forecast pair is the retained
**mean-decoded** one.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import (
    RECOMPOSITION_SCALE_COLUMN,
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    recomposition_check,
    scored_sample_count,
    skill_against,
)
from teb_vae.lag_attn_cfs.eval.metrics import baseline_forecasts
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.raw import SummaryUnits

#: ``(headline_name, key)`` pairs registered as ``("channel_skill", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("channel_skill_pred_gap_level_nats", "pred_gap_level"),
    ("channel_skill_pred_gap_variability_nats", "pred_gap_variability"),
    ("channel_skill_level_rmse_full_bpm", "level_rmse_full_bpm"),
    ("channel_skill_level_rmse_persistence_bpm", "level_rmse_persistence_bpm"),
    ("channel_skill_level_crossover_s", "level_crossover_s"),
    ("channel_skill_level_skill_vs_persistence_4s", "level_skill_first_lead"),
    ("channel_skill_level_frac_leads_beating_persistence", "level_frac_leads_beating"),
)

ANALYSIS_DIRNAME = "channel_skill"
CHANNELS: Tuple[str, ...] = ("level", "variability")
MODEL_BRANCHES: Tuple[str, ...] = ("base", "full")
BASELINES: Tuple[str, ...] = ("persistence", "segment_mean", "climatology")
TOTAL_COLUMN = "pred_gap"

PER_RECORDING_FILENAME = "channel_skill_per_recording.csv"
CHANNEL_FILENAME = "channel_skill_channels.csv"
HORIZON_FILENAME = "channel_skill_horizon.csv"
CROSSOVER_FILENAME = "channel_skill_crossover.csv"
CHANNEL_FIGURE = "channel_skill"
HORIZON_FIGURE = "channel_horizon_skill"

_GAP_VECTOR = "gap_per_channel"
_ERROR_VECTORS: Tuple[Tuple[str, str], ...] = (
    ("base", "sq_error_per_channel_base"), ("full", "sq_error_per_channel_full"),
)
#: The retained arrays the horizon curves read (``raw.retained_block`` adds weight and anchors).
_RETAINED = ("target", "mu_base", "mu_full", "fhr_raw")
_COLOURS = {
    "base": figures.COLOR_GRAY, "full": figures.COLOR_VERMILLION, "persistence": figures.COLOR_BLUE,
    "segment_mean": figures.COLOR_GREEN, "climatology": figures.COLOR_PURPLE,
}


# =============================================================================
# Shared inputs
# =============================================================================
def _vector(collection: Any, name: str, n_rows: int) -> Optional[np.ndarray]:
    """One per-sample ``(N, 2)`` vector readout in the per-sample table's order, or ``None``."""
    values = dict(getattr(collection, "vectors", None) or {}).get(name)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    return array if array.ndim == 2 and array.shape == (int(n_rows), len(CHANNELS)) else None


# =============================================================================
# Reading 1: per channel, whole split
# =============================================================================
def channel_frame(per_sample: pd.DataFrame, gap: np.ndarray, errors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """The per-channel columns on the per-sample rows: ``pred_gap_<ch>`` and ``sq_error_<branch>_<ch>``."""
    carried = [c for c in ("guid", "epoch", "clinical_class", "subgroup", "sample_index") if c in per_sample.columns]
    frame = pd.DataFrame(per_sample[carried]).copy()
    for c, channel in enumerate(CHANNELS):
        frame[f"pred_gap_{channel}"] = gap[:, c]
        for branch, values in errors.items():
            frame[f"sq_error_{branch}_{channel}"] = values[:, c]
    for name in (TOTAL_COLUMN, RECOMPOSITION_SCALE_COLUMN):
        frame[name] = finite_column(per_sample, name)
    return frame


def channel_rows(per_guid: pd.DataFrame, units: SummaryUnits, *, resamples: int, seed: int) -> List[Dict[str, Any]]:
    """One row per channel: the gap with its interval, full-vs-base skill, RMSE in clinical units."""
    rows: List[Dict[str, Any]] = []
    for c, channel in enumerate(CHANNELS):
        gap = shared_stats.bootstrap_ci(finite_column(per_guid, f"pred_gap_{channel}"), resamples=resamples, seed=seed)
        base = finite_column(per_guid, f"sq_error_base_{channel}")
        full = finite_column(per_guid, f"sq_error_full_{channel}")
        skill = shared_stats.bootstrap_ci(skill_against(full, base), resamples=resamples, seed=seed)
        row: Dict[str, Any] = {
            "channel": channel,
            "pred_gap_nats": gap["point"], "pred_gap_ci_lo": gap["lo"], "pred_gap_ci_hi": gap["hi"],
            "n_recordings": int(gap["n"]),
            "mse_skill_full_vs_base": skill["point"], "mse_skill_ci_lo": skill["lo"], "mse_skill_ci_hi": skill["hi"],
        }
        for branch, values in (("base", base), ("full", full)):
            rmse = math.sqrt(float(np.nanmean(values))) if np.isfinite(values).any() else float("nan")
            row[f"rmse_{branch}_standardized"] = rmse
            row[f"rmse_{branch}_bpm"] = float(units.level_delta_bpm(rmse)) if channel == "level" else float("nan")
            row[f"rmse_{branch}_factor"] = float(units.variability_factor(rmse)) if channel == "variability" else float("nan")
        rows.append(row)
    return rows


def build_channel_figure(per_guid: pd.DataFrame) -> Any:
    """Per-recording gap per channel (nats) and full-vs-base error skill per channel."""
    figure, axes = figures.new_figure(2)
    figures.violin_panel(
        axes[0, 0], {ch: finite_column(per_guid, f"pred_gap_{ch}") for ch in CHANNELS},
        title="Forecast gap by channel", ylabel="nats per anchor", reference=0.0, reference_label="no improvement",
    )
    figures.violin_panel(
        axes[1, 0],
        {ch: skill_against(finite_column(per_guid, f"sq_error_full_{ch}"), finite_column(per_guid, f"sq_error_base_{ch}"))
         for ch in CHANNELS},
        title="Error-space skill by channel", ylabel="1 - MSE_full / MSE_base", reference=0.0,
        reference_label="no improvement",
    )
    return figure


# =============================================================================
# Reading 2: per channel and lead, retained segments
# =============================================================================
def horizon_errors(context: Any, units: SummaryUnits) -> Optional[Dict[str, Any]]:
    """Masked squared error per ``(segment, τ, channel)``, anchor-averaged, for both branches and the baselines.

    Returns ``None`` without retained waveforms.
    """
    block = raw.retained_block(context, _RETAINED)
    if block is None:
        return None
    target, fhr = block["arrays"]["target"], block["arrays"]["fhr_raw"]
    weight, anchors, mask, r = block["weight"], block["anchors"], block["mask"], block["r"]
    horizon = int(target.shape[2])
    features = raw.summaries_from_raw(fhr, weight, units, r)
    # The rebuilt target must be the scored one: gather it at a + 1 + τ and compare on scored cells.
    steps = anchors[:, :, None] + 1 + torch.arange(horizon)
    rebuilt = features[torch.arange(len(features))[:, None, None], steps]
    agreement = float(((rebuilt - target).abs() * mask[..., None]).max()) if mask.any() else float("nan")
    # The shared baselines read only these three attributes; a patch model has none of them.
    stub = SimpleNamespace(target_gate=None, target_forecast_shift=None, target_warmup_steps=None)
    forecasts = {
        "base": block["arrays"]["mu_base"],
        "full": block["arrays"]["mu_full"],
        **baseline_forecasts(features, weight, stub, anchors),
    }
    if units.is_identity:
        # Zero is the population mean of the summaries only once summary_stats.py's constants are set.
        forecasts.pop("climatology")
    # Reduced over anchors here (anchor-mean per segment and lead), so nothing (N, A, H, 2) is held.
    m = mask[..., None].double()
    counts = m.sum(dim=1)                                                    # (N, H, 1)
    segment = {
        name: torch.where(counts > 0, (((prediction - target).double() ** 2) * m).sum(dim=1) / counts.clamp_min(1.0),
                          torch.full_like(counts, float("nan"))).numpy()
        for name, prediction in forecasts.items()
    }                                                                        # (N, H, 2)
    return {
        "segment_mse": segment, "horizon": horizon,
        "sample_index": block["sample_index"],
        "target_agreement_max_abs": agreement,
    }


def per_recording_curves(errors: Dict[str, Any], ids: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """``name -> (R, H, 2)`` per-recording MSE per lead (segment anchor-mean, then segment-mean).

    Returns:
        The curves and the recordings' identity rows, in the curves' order.
    """
    guids = ids["guid"].astype(str).to_numpy()
    order = list(dict.fromkeys(guids))
    curves: Dict[str, np.ndarray] = {
        name: np.stack([np.nanmean(segment[guids == guid], axis=0) for guid in order])
        for name, segment in errors["segment_mse"].items()
    }
    labels = ids.drop_duplicates("guid").set_index("guid").reindex(order).reset_index()
    return curves, labels


def crossover(skill: np.ndarray) -> Tuple[Optional[int], str]:
    """The first lead index at which the skill against persistence changes sign, and the direction.

    Returns:
        ``(index, kind)``: ``kind`` is ``'stops_beating'`` (positive at the first lead, then ≤ 0),
        ``'starts_beating'`` (≤ 0 first, then positive), ``'never_beats'`` or ``'always_beats'``
        (no sign change; ``index`` is ``None``).
    """
    finite = np.isfinite(skill)
    if not finite.any():
        return None, "undefined"
    values = skill[finite]
    first_positive = bool(values[0] > 0.0)
    for index in np.flatnonzero(finite)[1:]:
        if bool(skill[index] > 0.0) != first_positive:
            return int(index), ("stops_beating" if first_positive else "starts_beating")
    return None, ("always_beats" if first_positive else "never_beats")


def _mean_curves(curves: Mapping[str, np.ndarray], rows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    return {name: np.nanmean(values if rows is None else values[rows], axis=0) for name, values in curves.items()}


def horizon_table(
    curves: Mapping[str, np.ndarray], units: SummaryUnits, *, resamples: int, seed: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Per lead and channel: RMSE of every predictor and the skill against persistence, with intervals.

    The bootstrap resamples recordings; each resample gives a full set of curves, so the intervals
    and the crossover distribution come from one draw.
    """
    n_rec = next(iter(curves.values())).shape[0]
    point = _mean_curves(curves)
    rng = np.random.default_rng(int(seed))
    draws = [_mean_curves(curves, rng.integers(0, n_rec, n_rec)) for _ in range(int(resamples))] if n_rec >= 2 else []
    horizon = point["full"].shape[0]
    lead = (np.arange(horizon) + 1.0) * float(SECONDS_PER_STEP)
    frames = []
    crossing: Dict[str, Any] = {}
    for c, channel in enumerate(CHANNELS):
        frame = pd.DataFrame({"channel": channel, "horizon_step": np.arange(horizon), "lead_seconds": lead})
        for name, mse in point.items():
            rmse = np.sqrt(np.clip(mse[:, c], 0.0, None))
            frame[f"rmse_{name}_standardized"] = rmse
            if channel == "level":
                frame[f"rmse_{name}_bpm"] = units.level_delta_bpm(rmse)
            else:
                frame[f"rmse_{name}_factor"] = units.variability_factor(rmse)
            if draws:
                boot = np.sqrt(np.clip(np.stack([d[name][:, c] for d in draws]), 0.0, None))
                frame[f"rmse_{name}_lo"], frame[f"rmse_{name}_hi"] = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        for branch in MODEL_BRANCHES:
            skill = 1.0 - point[branch][:, c] / point["persistence"][:, c]
            frame[f"skill_{branch}_vs_persistence"] = skill
            boot_cross: List[float] = []
            if draws:
                boot = np.stack([1.0 - d[branch][:, c] / d["persistence"][:, c] for d in draws])
                frame[f"skill_{branch}_lo"], frame[f"skill_{branch}_hi"] = np.nanpercentile(boot, [2.5, 97.5], axis=0)
                for row in boot:
                    index, _ = crossover(row)
                    boot_cross.append(float("nan") if index is None else float(lead[index]))
            index, kind = crossover(skill)
            crossing[f"{channel}_{branch}"] = {
                "kind": kind,
                "crossover_lead_s": None if index is None else float(lead[index]),
                "bootstrap_crossover_lead_s": describe(np.asarray(boot_cross), name="crossover_lead_s") if boot_cross else None,
                "bootstrap_frac_no_crossing": float(np.mean(~np.isfinite(boot_cross))) if boot_cross else None,
                "skill_first_lead": raw.finite_or_none(skill[0]),
                "skill_last_lead": raw.finite_or_none(skill[-1]),
                # Robust beside the first sign change, which a noisy curve can trip early.
                "frac_leads_beating": raw.finite_or_none(np.mean(skill[np.isfinite(skill)] > 0.0))
                if np.isfinite(skill).any() else None,
            }
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), crossing


def class_crossover(curves: Mapping[str, np.ndarray], labels: pd.DataFrame) -> pd.DataFrame:
    """The level crossover lead per clinical class (point estimate, per-recording curves of that class)."""
    rows = []
    if "clinical_class" not in labels.columns:
        return pd.DataFrame()
    classes = labels["clinical_class"].astype(str).to_numpy()
    for name in dict.fromkeys(classes):
        select = np.flatnonzero(classes == name)
        mean = _mean_curves(curves, select)
        for branch in MODEL_BRANCHES:
            skill = 1.0 - mean[branch][:, 0] / mean["persistence"][:, 0]
            index, kind = crossover(skill)
            rows.append({
                "clinical_class": name, "branch": branch, "n_recordings": int(select.size), "kind": kind,
                "crossover_lead_s": float("nan") if index is None else float((index + 1) * SECONDS_PER_STEP),
                "skill_first_lead": float(skill[0]), "skill_last_lead": float(skill[-1]),
            })
    return pd.DataFrame(rows)


def build_horizon_figure(table: pd.DataFrame, crossing: Mapping[str, Any]) -> Any:
    """RMSE per lead (level in bpm, variability as a factor) and the skill against persistence."""
    figure, axes = figures.new_figure(2, 2, height_per_row=2.6)
    for c, channel in enumerate(CHANNELS):
        part = table[table["channel"] == channel]
        lead = part["lead_seconds"].to_numpy()
        unit = "bpm" if channel == "level" else "factor"
        ax = axes[0, c]
        for name, colour in _COLOURS.items():
            column = f"rmse_{name}_{unit}"
            if column not in part.columns:
                continue
            style = "-" if name in MODEL_BRANCHES else "--"
            ax.plot(lead, part[column], color=colour, linewidth=figures.LINE_EMPHASIS, linestyle=style, label=name)
            if f"rmse_{name}_lo" in part.columns and name in ("full", "persistence"):
                lo, hi = part[f"rmse_{name}_lo"].to_numpy(), part[f"rmse_{name}_hi"].to_numpy()
                if channel == "level":
                    lo, hi = lo * (part[column] / part[f"rmse_{name}_standardized"]), hi * (part[column] / part[f"rmse_{name}_standardized"])
                else:
                    scale = np.log(part[column].to_numpy()) / part[f"rmse_{name}_standardized"].to_numpy()
                    lo, hi = np.exp(lo * scale), np.exp(hi * scale)
                ax.fill_between(lead, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.set_title(f"{channel.capitalize()}: RMSE by lead")
        ax.set_xlabel("lead (s)")
        ax.set_ylabel("RMSE (bpm)" if channel == "level" else "RMSE as a factor on rms Δ")
        ax.legend(loc="upper left", fontsize=figures.FONT_TINY, ncol=3)
        figures.style_axes(ax)

        ax = axes[1, c]
        for branch in MODEL_BRANCHES:
            colour = _COLOURS[branch]
            ax.plot(lead, part[f"skill_{branch}_vs_persistence"], color=colour, linewidth=figures.LINE_EMPHASIS,
                    label=branch)
            if f"skill_{branch}_lo" in part.columns:
                ax.fill_between(lead, part[f"skill_{branch}_lo"], part[f"skill_{branch}_hi"], color=colour, alpha=0.15,
                                linewidth=0)
            lead_s = (crossing.get(f"{channel}_{branch}") or {}).get("crossover_lead_s")
            if lead_s is not None:
                ax.axvline(lead_s, color=colour, linewidth=figures.LINE_THIN, linestyle=":",
                           label=f"{branch} crossover {lead_s:.0f} s")
        ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
        ax.set_title(f"{channel.capitalize()}: skill against persistence")
        ax.set_xlabel("lead (s)")
        ax.set_ylabel("1 - MSE / MSE_persistence")
        ax.legend(loc="lower right", fontsize=figures.FONT_TINY, ncol=2)
        figures.style_axes(ax)
    return figure


# =============================================================================
# Entry point
# =============================================================================
def run_channel_skill_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-channel gap and skill on the whole split, then per-lead skill against persistence on the retained segments."""
    collection = context.collection
    per_sample = collection.per_sample
    gap = _vector(collection, _GAP_VECTOR, len(per_sample))
    if gap is None:
        return {"n_samples": None, "composition": {}, "plan": {"capped": False}, "skipped": True,
                "reason": f"the collection carries no usable (N, 2) {_GAP_VECTOR!r} vector"}
    errors = {branch: v for branch, name in _ERROR_VECTORS
              if (v := _vector(collection, name, len(per_sample))) is not None}
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    units = raw.SummaryUnits.from_context(context)
    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))

    frame = channel_frame(per_sample, gap, errors)
    value_columns = [c for c in frame.columns if c.startswith(("pred_gap", "sq_error_"))] + [RECOMPOSITION_SCALE_COLUMN]
    per_guid = per_recording_means(frame, value_columns)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)
    rows = channel_rows(per_guid, units, resamples=resamples, seed=seed)
    pd.DataFrame(rows).to_csv(directory / CHANNEL_FILENAME, index=False)
    files = [PER_RECORDING_FILENAME, CHANNEL_FILENAME,
             figures.render_figure(build_channel_figure(per_guid), directory / CHANNEL_FIGURE).name]
    recomposition = recomposition_check(
        per_guid, [f"pred_gap_{ch}" for ch in CHANNELS], TOTAL_COLUMN,
        identity="pred_gap_level + pred_gap_variability == pred_gap",
    )

    horizon: Dict[str, Any] = {"skipped": True, "reason": "no retained waveforms with fhr_raw (eval_config.caps.waveforms)"}
    headline_extra: Dict[str, Optional[float]] = {}
    found = horizon_errors(context, units)
    if found is not None:
        columns = [c for c in ("sample_index", "guid", "clinical_class", "subgroup") if c in per_sample.columns]
        ids = per_sample[columns].drop_duplicates("sample_index").set_index("sample_index").reindex(found["sample_index"]).reset_index()
        curves, labels = per_recording_curves(found, ids)
        table, crossing = horizon_table(curves, units, resamples=resamples, seed=seed)
        table.to_csv(directory / HORIZON_FILENAME, index=False)
        by_class = class_crossover(curves, labels)
        by_class.to_csv(directory / CROSSOVER_FILENAME, index=False)
        files += [HORIZON_FILENAME, CROSSOVER_FILENAME,
                  figures.render_figure(build_horizon_figure(table, crossing), directory / HORIZON_FIGURE).name]
        level = table[table["channel"] == "level"]
        horizon = {
            "skipped": False,
            "n_segments": int(len(found["sample_index"])),
            "n_recordings": int(len(labels)),
            "baselines": [name for name in BASELINES if name in found["segment_mse"]],
            "crossover": crossing,
            "target_rebuild_max_abs": found["target_agreement_max_abs"],
            "decode": "mean-decoded retained pair",
            "mean_over_leads_bpm": {
                name: raw.finite_or_none(level[f"rmse_{name}_bpm"].mean())
                for name in (*MODEL_BRANCHES, *BASELINES) if f"rmse_{name}_bpm" in level.columns
            },
        }
        headline_extra = {
            "level_rmse_full_bpm": horizon["mean_over_leads_bpm"].get("full"),
            "level_rmse_persistence_bpm": horizon["mean_over_leads_bpm"].get("persistence"),
            "level_crossover_s": (crossing.get("level_full") or {}).get("crossover_lead_s"),
            "level_skill_first_lead": (crossing.get("level_full") or {}).get("skill_first_lead"),
            "level_frac_leads_beating": (crossing.get("level_full") or {}).get("frac_leads_beating"),
        }

    by_channel = {row["channel"]: row for row in rows}
    return {
        "n_samples": scored_sample_count(per_sample, TOTAL_COLUMN),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed,
                 "horizon_capped_by": "eval_config.caps.waveforms",
                 "horizon_segments": None if found is None else int(len(found["sample_index"]))},
        "skipped": False,
        "channels": rows,
        "recomposition": recomposition,
        "horizon": horizon,
        "units": {"level": "bpm (RMSE, gap in nats)", "variability": "factor on rms Δ (a log-ratio)",
                  "summary_constants_identity": units.is_identity},
        "headline": {
            "pred_gap_level": raw.finite_or_none(by_channel["level"]["pred_gap_nats"]),
            "pred_gap_variability": raw.finite_or_none(by_channel["variability"]["pred_gap_nats"]),
            "level_rmse_full_bpm": None, "level_rmse_persistence_bpm": None, "level_crossover_s": None,
            "level_skill_first_lead": None, "level_frac_leads_beating": None,
            **{k: raw.finite_or_none(v) for k, v in headline_extra.items()},
        },
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, [f"pred_gap_{ch}" for ch in CHANNELS])
        ],
        "files": [str(name) for name in files],
    }
