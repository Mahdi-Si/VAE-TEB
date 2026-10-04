r"""``calibration`` -- per-channel calibration and variability against raw STV; port, owner E1-F.

**Question** (``notes/EVAL_PLAN.md`` §2). Is the decoder's predicted uncertainty right **per
channel**? Level and log-variability have different error shapes, and one pooled PIT hides which
of the two is miscalibrated. Does the variability channel track a clinical variability measured on
the raw trace?

Three readings:

* **Pooled** (whole split): the shared CFS analysis, unchanged. It computes PIT, coverage, CRPS
  and the gain over the homoscedastic MLE over every scored cell of both channels, from the
  collection pass's sums. It also gives the clamp recommendation and the per-recording table that
  ``cross_subgroup`` reads.
* **Per channel** (retained segments, ``caps.waveforms``): the shared ``calibration_sums`` /
  ``calibration_report`` arithmetic on each channel alone. Same AR(1) innovation (that channel's
  ``φ_c``), same forecast mask (``forecast_mask`` with the coverage floor), both branches. The
  level CRPS is also given in bpm. The retained forecast pair is the **mean-decoded** one, which
  differs from the pooled census (one posterior draw); the record says so.
* **Variability against raw STV** (retained segments). Per decoded anchor, over its 120 s horizon:
  - the predicted mean rms-Δ (bpm per 0.25 s, full branch), against
  - the realised rms-Δ of the same patches (the target itself), and
  - a raw short-term variability: the mean absolute difference of successive 3.75 s epoch means
    of the raw FHR in bpm (Dawes–Redman style).

  A reliability curve over deciles of the prediction and a per-recording Spearman ρ. Both floors
  are marked: the 0.25 bpm monitor resolution and ``variability_eps`` in bpm. Below them the
  target measures quantisation or ``eps``, not physiology.

Anchors overlap in ``H - 1`` of ``H`` steps, so the reliability bins describe the anchors. The
per-recording ρ is the statistic.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import ticker as mticker

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval import traces
from teb_vae.lag_attn_cfs.eval.analyses import calibration as _shared
from teb_vae.lag_attn_cfs.eval.frames import describe
from teb_vae.lag_attn_cfs.eval.metrics import COVERAGE_NOMINALS, calibration_report, calibration_sums
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.raw import SummaryUnits

#: ``(headline_name, key)`` pairs registered as ``("calibration", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("calibration_level_coverage_2sigma", "level_coverage_2sigma"),
    ("calibration_variability_coverage_2sigma", "variability_coverage_2sigma"),
    ("calibration_level_crps_bpm", "level_crps_bpm"),
    ("calibration_variability_stv_spearman", "variability_stv_spearman"),
)

ANALYSIS_DIRNAME = "calibration"
CHANNELS: Tuple[str, ...] = ("level", "variability")
BRANCHES: Tuple[str, ...] = ("base", "full")

CHANNEL_FILENAME = "calibration_channels.csv"
CHANNEL_PIT_FILENAME = "calibration_channel_pit.csv"
RELIABILITY_FILENAME = "variability_reliability.csv"
VARIABILITY_RECORDINGS_FILENAME = "variability_per_recording.csv"
CHANNEL_FIGURE = "calibration_channels"
RELIABILITY_FIGURE = "variability_reliability"

#: The FHR monitor's resolution, bpm.
FHR_RESOLUTION_BPM = 0.25
#: Samples per raw STV epoch: 3.75 s at 4 Hz, the Dawes–Redman epoch.
STV_EPOCH_SAMPLES = 15
#: Reliability bins over the predicted variability (deciles).
RELIABILITY_BINS = 10
#: Fewest anchors a recording needs for its Spearman ρ.
MIN_ANCHORS_PER_RECORDING = 10

#: The retained arrays this analysis reads.
#: The retained arrays the per-channel calibration reads (``raw.retained_block`` adds weight and anchors).
_RETAINED = ("target", "mu_base", "mu_full", "logvar_base", "logvar_full")


# =============================================================================
# Inputs
# =============================================================================
def identities(per_sample: pd.DataFrame, sample_index: np.ndarray) -> pd.DataFrame:
    """``guid`` and the cohort columns of each retained row, in retained order."""
    columns = [c for c in ("sample_index", "guid", "clinical_class", "subgroup") if c in per_sample.columns]
    table = per_sample[columns].drop_duplicates("sample_index").set_index("sample_index")
    return table.reindex(sample_index).reset_index()


# =============================================================================
# Per-channel calibration
# =============================================================================
def channel_reports(block: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``(branch, channel) -> calibration_report`` on one channel at a time, under that channel's φ."""
    arrays, mask = block["arrays"], block["mask"]
    reports: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for branch in BRANCHES:
        for c, channel in enumerate(CHANNELS):
            phi = None if block["phi"] is None else torch.tensor([block["phi"][c]], dtype=torch.float32)
            sums = calibration_sums(
                arrays[f"mu_{branch}"][..., c:c + 1], arrays[f"logvar_{branch}"][..., c:c + 1],
                arrays["target"][..., c:c + 1], mask, logvar_clamp=block["clamp"], ar_coef=phi,
            )
            reports[(branch, channel)] = calibration_report(
                {k: v.cpu().numpy() for k, v in sums.items()}, logvar_clamp=block["clamp"]
            )
    return reports


def channel_rows(reports: Mapping[Tuple[str, str], Dict[str, Any]], units: SummaryUnits) -> List[Dict[str, Any]]:
    """One row per ``(branch, channel)``: coverage at 1/2/3σ, PIT departure, CRPS (bpm for level)."""
    rows: List[Dict[str, Any]] = []
    for (branch, channel), report in reports.items():
        if not report:
            continue
        crps = float(report["crps_normalised"])
        row: Dict[str, Any] = {
            "branch": branch, "channel": channel, "n_cells": int(report["n_coefficients"]),
            "pit_max_cdf_deviation": float(report["pit"]["max_cdf_deviation"]),
            "mean_standardised_sq": float(report["mean_standardised_sq"]),
            "mean_logvar": float(report["mean_logvar_full"]),
            "nll_gain_per_cell": float(report["nll"]["gain_per_coefficient"]),
            "crps_standardized": crps,
            # Level: a σ in bpm. Variability: a log-ratio, reported as the factor on rms Δ.
            "crps_bpm": float(units.level_delta_bpm(crps)) if channel == "level" else float("nan"),
            "crps_variability_factor": float(units.variability_factor(crps)) if channel == "variability" else float("nan"),
        }
        for entry in report["coverage"]:
            row[f"coverage_{entry['level_sigma']}sigma"] = float(entry["observed"])
            row[f"nominal_{entry['level_sigma']}sigma"] = float(entry["nominal"])
        rows.append(row)
    return rows


def pit_frame(reports: Mapping[Tuple[str, str], Dict[str, Any]]) -> pd.DataFrame:
    """The PIT density per ``(branch, channel)``, long form."""
    parts = []
    for (branch, channel), report in reports.items():
        if not report:
            continue
        edges = np.asarray(report["pit"]["bin_edges"], dtype=np.float64)
        parts.append(pd.DataFrame({
            "branch": branch, "channel": channel, "bin_left": edges[:-1], "bin_right": edges[1:],
            "density": np.asarray(report["pit"]["density"], dtype=np.float64),
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# =============================================================================
# Variability against raw STV
# =============================================================================
def raw_stv_windows(fhr_bpm: np.ndarray, valid: np.ndarray, anchors: np.ndarray, *, r: int, horizon: int) -> np.ndarray:
    r"""Raw STV over each anchor's horizon: mean ``|Δ|`` of successive 3.75 s epoch means, bpm.

    The window is raw samples ``[R(a + 1), R(a + 1 + H))``. An epoch counts only when every one of
    its samples is valid. A difference counts only between two consecutive valid epochs.

    Returns:
        ``(N, A)``, ``NaN`` where fewer than two consecutive valid epochs exist.
    """
    n_rows, n_anchors = anchors.shape
    width = r * horizon
    n_epochs = width // STV_EPOCH_SAMPLES
    offsets = np.arange(n_epochs * STV_EPOCH_SAMPLES)
    out = np.full((n_rows, n_anchors), np.nan)
    for row in range(n_rows):
        start = r * (anchors[row] + 1)                                       # (A,)
        index = start[:, None] + offsets[None, :]                           # (A, E*15)
        inside = index < fhr_bpm.shape[1]
        index = np.minimum(index, fhr_bpm.shape[1] - 1)
        values = fhr_bpm[row][index].reshape(n_anchors, n_epochs, STV_EPOCH_SAMPLES)
        ok = (valid[row][index] & inside).reshape(n_anchors, n_epochs, STV_EPOCH_SAMPLES).all(axis=-1)
        means = np.where(ok, values.mean(axis=-1), np.nan)
        diffs = np.abs(np.diff(means, axis=-1))
        count = np.isfinite(diffs).sum(axis=-1)
        total = np.nansum(diffs, axis=-1)
        out[row] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def variability_frame(block: Dict[str, Any], units: SummaryUnits, ids: pd.DataFrame) -> Optional[pd.DataFrame]:
    """One row per contributing retained anchor: predicted, realised and raw-STV variability, bpm."""
    arrays, mask = block["arrays"], block["mask"].numpy()
    if block["fhr_raw"] is None:
        return None
    m = mask.astype(bool)                                                   # (N, A, H)
    counts = m.sum(axis=-1)
    contributing = counts > 0

    def _horizon_mean(values: np.ndarray) -> np.ndarray:
        return np.where(contributing, np.where(m, values, 0.0).sum(axis=-1) / np.maximum(counts, 1), np.nan)

    predicted = _horizon_mean(units.variability_bpm(arrays["mu_full"][..., 1].numpy().astype(np.float64)))
    realised = _horizon_mean(units.variability_bpm(arrays["target"][..., 1].numpy().astype(np.float64)))
    weight = block["weight"].numpy()
    fhr = block["fhr_raw"]
    raw_valid = np.stack([traces.raw_validity_of(row, fhr.shape[1]) for row in weight]).astype(bool)
    scales = {"fhr": (units.fhr_mean, units.fhr_std)}
    fhr_bpm = np.stack([traces.physical_raw(row, "fhr", scales, valid)[0] for row, valid in zip(fhr, raw_valid)])
    raw_valid &= np.isfinite(fhr_bpm)
    stv = raw_stv_windows(fhr_bpm, raw_valid, block["anchors"].numpy(), r=block["r"], horizon=block["geometry"].horizon)
    rows, cols = np.nonzero(contributing)
    frame = pd.DataFrame({
        "row": rows, "anchor": block["anchors"].numpy()[rows, cols],
        "predicted_rms_delta_bpm": predicted[rows, cols],
        "realised_rms_delta_bpm": realised[rows, cols],
        "raw_stv_bpm": stv[rows, cols],
    })
    return frame.join(ids.reset_index(drop=True), on="row")


def reliability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Deciles of the prediction: per bin the median and quartiles of the realised and the raw STV."""
    usable = frame[np.isfinite(frame["predicted_rms_delta_bpm"]) & (frame["predicted_rms_delta_bpm"] > 0)]
    if len(usable) < RELIABILITY_BINS:
        return pd.DataFrame()
    bins = pd.qcut(usable["predicted_rms_delta_bpm"], RELIABILITY_BINS, labels=False, duplicates="drop")
    out = []
    for b, part in usable.groupby(bins):
        row = {"bin": int(b), "n_anchors": int(len(part)),
               "predicted_median_bpm": float(part["predicted_rms_delta_bpm"].median())}
        for name in ("realised_rms_delta_bpm", "raw_stv_bpm"):
            values = part[name].dropna()
            q = values.quantile([0.25, 0.5, 0.75]) if len(values) else pd.Series([np.nan] * 3, index=[0.25, 0.5, 0.75])
            row.update({f"{name}_q1": float(q[0.25]), f"{name}_median": float(q[0.5]), f"{name}_q3": float(q[0.75])})
        out.append(row)
    return pd.DataFrame(out)


def per_recording_spearman(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ of predicted against realised rms-Δ and against raw STV, within each recording."""
    rows = []
    for guid, part in frame.groupby("guid"):
        row: Dict[str, Any] = {"guid": guid, "n_anchors": int(len(part))}
        for c in ("clinical_class", "subgroup"):
            if c in part.columns:
                row[c] = part[c].iloc[0]
        for name, column in (("realised", "realised_rms_delta_bpm"), ("raw_stv", "raw_stv_bpm")):
            pair = part[["predicted_rms_delta_bpm", column]].dropna()
            row[f"spearman_{name}"] = (
                float(pair.corr(method="spearman").iloc[0, 1])
                if len(pair) >= MIN_ANCHORS_PER_RECORDING and pair.nunique().min() > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================
def build_channel_figure(pit: pd.DataFrame, rows: List[Dict[str, Any]]) -> Any:
    """PIT density per channel (full solid, base dashed) and the observed central coverage."""
    figure, axes = figures.new_figure(2, 2, height_per_row=2.4)
    colours = {"base": figures.COLOR_GRAY, "full": figures.COLOR_VERMILLION}
    for c, channel in enumerate(CHANNELS):
        ax = axes[0, c]
        for branch in BRANCHES:
            part = pit[(pit["branch"] == branch) & (pit["channel"] == channel)] if len(pit) else pit
            if len(part):
                ax.stairs(part["density"].to_numpy(), np.append(part["bin_left"].to_numpy(), part["bin_right"].iloc[-1]),
                          color=colours[branch], linestyle="-" if branch == "full" else "--",
                          linewidth=figures.LINE_EMPHASIS, label=branch)
        ax.axhline(1.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle=":", label="uniform")
        ax.set_title(f"PIT, {channel}")
        ax.set_xlabel("PIT")
        ax.set_ylabel("density")
        ax.legend(loc="upper center", fontsize=figures.FONT_TINY, ncol=3)
        figures.style_axes(ax)

        ax = axes[1, c]
        levels = np.arange(1, len(COVERAGE_NOMINALS) + 1)
        for k, branch in enumerate(BRANCHES):
            row = next((r for r in rows if r["branch"] == branch and r["channel"] == channel), None)
            if row is None:
                continue
            observed = [row.get(f"coverage_{level}sigma", np.nan) for level in levels]
            ax.bar(levels + (k - 0.5) * 0.35, np.asarray(observed) - np.asarray(COVERAGE_NOMINALS), width=0.35,
                   color=colours[branch], label=branch)
        ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{level}σ ({nominal:.4f})" for level, nominal in zip(levels, COVERAGE_NOMINALS)])
        ax.set_title(f"Central coverage minus nominal, {channel}")
        ax.set_ylabel("observed - nominal")
        ax.legend(loc="lower right", fontsize=figures.FONT_TINY)
        figures.style_axes(ax)
    return figure


#: Most anchors drawn in the reliability scatter (seeded subsample), so a production pass stays light.
SCATTER_MAX_POINTS = 20000


def _floors(units: SummaryUnits) -> List[Tuple[float, str, str]]:
    """The two floors, merged when ``variability_eps`` was derived from the monitor resolution."""
    eps_bpm = units.eps * units.fhr_std
    if abs(eps_bpm - FHR_RESOLUTION_BPM) <= 0.05 * FHR_RESOLUTION_BPM:
        return [(FHR_RESOLUTION_BPM, f"resolution {FHR_RESOLUTION_BPM:g} bpm = variability_eps", figures.COLOR_BLUE)]
    return [(FHR_RESOLUTION_BPM, f"resolution {FHR_RESOLUTION_BPM:g} bpm", figures.COLOR_BLUE),
            (eps_bpm, f"variability_eps = {eps_bpm:.2g} bpm", figures.COLOR_PURPLE)]


def build_reliability_figure(
    frame: pd.DataFrame, reliability: pd.DataFrame, per_recording: pd.DataFrame, units: SummaryUnits, *, seed: int = 0
) -> Any:
    """Predicted rms Δ per anchor against the realised rms Δ and the raw STV, log-log, floors marked.

    Every anchor is a faint point (a seeded subsample beyond :data:`SCATTER_MAX_POINTS`); the decile
    medians with their IQR are drawn over them when the prediction spans at least two deciles. A
    prediction with no spread (a variability forecast at climatology) collapses to one column of
    points, and the record's ``predicted_spread`` says so.
    """
    figure, axes = figures.new_figure(1, 3, height_per_row=3.2)
    floors = _floors(units)
    usable = frame[np.isfinite(frame["predicted_rms_delta_bpm"]) & (frame["predicted_rms_delta_bpm"] > 0)]
    if len(usable) > SCATTER_MAX_POINTS:
        usable = usable.sample(SCATTER_MAX_POINTS, random_state=int(seed))
    for ax, name, title, identity in (
        (axes[0, 0], "realised_rms_delta_bpm", "Realised rms Δ of the same patches", True),
        (axes[0, 1], "raw_stv_bpm", "Raw STV (3.75 s epochs; not the same quantity)", False),
    ):
        x = usable["predicted_rms_delta_bpm"].to_numpy()
        y = usable[name].to_numpy()
        keep = np.isfinite(y) & (y > 0)
        if not keep.any():
            ax.text(0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        ax.scatter(x[keep], y[keep], s=1.5, color=figures.COLOR_GRAY, alpha=0.2, linewidths=0, rasterized=True,
                   label="anchors")
        if len(reliability) >= 2:
            mx = reliability["predicted_median_bpm"].to_numpy()
            my = reliability[f"{name}_median"].to_numpy()
            lo, hi = reliability[f"{name}_q1"].to_numpy(), reliability[f"{name}_q3"].to_numpy()
            ax.errorbar(mx, my, yerr=np.vstack([np.clip(my - lo, 0, None), np.clip(hi - my, 0, None)]),
                        fmt="o-", color=figures.COLOR_VERMILLION, markersize=figures.MARKER_SMALL,
                        linewidth=figures.LINE_REGULAR, elinewidth=figures.LINE_THIN, label="decile median, IQR")
        data = np.concatenate([x[keep], y[keep]])
        low = min(float(np.percentile(data, 0.5)), min(value for value, _, _ in floors)) / 1.5
        high = float(np.percentile(data, 99.5)) * 1.5
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(mticker.NullFormatter())
        if identity:
            ax.plot([low, high], [low, high], color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle=":",
                    label="y = x")
        for value, label, colour in floors:
            ax.axvline(value, color=colour, linewidth=figures.LINE_THIN, linestyle="--", label=label)
            ax.axhline(value, color=colour, linewidth=figures.LINE_THIN, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("predicted rms Δ, full (bpm per 0.25 s)")
        ax.set_ylabel("rms Δ (bpm)" if identity else "STV (bpm)")
        ax.legend(loc="upper left", fontsize=figures.FONT_TINY)
        figures.style_axes(ax)
    samples = {label: per_recording[column].dropna().to_numpy() for label, column in
               (("vs realised", "spearman_realised"), ("vs raw STV", "spearman_raw_stv"))
               if column in per_recording.columns}
    figures.violin_panel(axes[0, 2], samples, title="Within-recording Spearman ρ", ylabel="ρ", reference=0.0)
    return figure


# =============================================================================
# Entry point
# =============================================================================
def run_calibration_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The pooled shared census, then the per-channel and the STV readings on the retained segments."""
    result = _shared.run_calibration_analysis(context, eval_config=eval_config, output_dir=output_dir, probe=probe)
    if result.get("skipped"):
        return result
    plan = result.setdefault("plan", {})
    plan["implementation"] = "patch port: pooled (shared) + per-channel + variability vs raw STV (retained)"
    block = raw.retained_block(context, _RETAINED)
    if block is None:
        result["per_channel"] = {"skipped": True, "reason": "no retained waveforms (eval_config.caps.waveforms)"}
        return result

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    units = raw.SummaryUnits.from_context(context)
    reports = channel_reports(block)
    rows = channel_rows(reports, units)
    pd.DataFrame(rows).to_csv(directory / CHANNEL_FILENAME, index=False)
    pit = pit_frame(reports)
    pit.to_csv(directory / CHANNEL_PIT_FILENAME, index=False)
    files = [CHANNEL_FILENAME, CHANNEL_PIT_FILENAME,
             figures.render_figure(build_channel_figure(pit, rows), directory / CHANNEL_FIGURE).name]

    ids = identities(context.collection.per_sample, block["sample_index"])
    frame = variability_frame(block, units, ids)
    variability: Dict[str, Any] = {"skipped": True, "reason": "fhr_raw not retained"}
    if frame is not None and len(frame):
        reliability = reliability_frame(frame)
        reliability.to_csv(directory / RELIABILITY_FILENAME, index=False)
        per_recording = per_recording_spearman(frame)
        per_recording.to_csv(directory / VARIABILITY_RECORDINGS_FILENAME, index=False)
        files += [RELIABILITY_FILENAME, VARIABILITY_RECORDINGS_FILENAME,
                  figures.render_figure(build_reliability_figure(frame, reliability, per_recording, units,
                                                                       seed=int(eval_config.get("seed", 0))),
                                        directory / RELIABILITY_FIGURE).name]
        variability = {
            "skipped": False,
            "n_anchors": int(len(frame)),
            "n_recordings": int(per_recording["guid"].nunique()) if len(per_recording) else 0,
            "spearman_realised": describe(per_recording.get("spearman_realised", pd.Series(dtype=float)).dropna().to_numpy(), name="spearman_realised"),
            "spearman_raw_stv": describe(per_recording.get("spearman_raw_stv", pd.Series(dtype=float)).dropna().to_numpy(), name="spearman_raw_stv"),
            "floors_bpm": {"monitor_resolution": FHR_RESOLUTION_BPM, "variability_eps": units.eps * units.fhr_std},
            # How much the prediction moves across anchors: a forecast at climatology has no spread,
            # and then no rank correlation with anything can be read.
            "predicted_spread": describe(frame["predicted_rms_delta_bpm"].to_numpy(), name="predicted_rms_delta_bpm"),
            "predicted_n_deciles": int(len(reliability)),
            "frac_predicted_below_eps": float(np.mean(frame["predicted_rms_delta_bpm"] < units.eps * units.fhr_std)),
            "frac_realised_below_resolution": float(np.mean(frame["realised_rms_delta_bpm"] < FHR_RESOLUTION_BPM)),
            "stv_definition": "mean |diff| of successive 3.75 s epoch means of raw FHR (bpm) over the anchor's 120 s horizon",
        }

    def _get(channel: str, key: str) -> Optional[float]:
        row = next((r for r in rows if r["branch"] == "full" and r["channel"] == channel), None)
        value = None if row is None else row.get(key)
        return None if value is None or not math.isfinite(float(value)) else float(value)

    rho = (variability.get("spearman_raw_stv") or {}).get("q50") if not variability.get("skipped") else None
    result["headline"] = {
        "level_coverage_2sigma": _get("level", "coverage_2sigma"),
        "variability_coverage_2sigma": _get("variability", "coverage_2sigma"),
        "level_crps_bpm": _get("level", "crps_bpm"),
        "variability_stv_spearman": None if rho is None or not math.isfinite(float(rho)) else float(rho),
    }
    result["per_channel"] = {
        "rows": rows,
        "n_segments": int(len(block["sample_index"])),
        "decode": "mean-decoded retained pair (the pooled census scores the forward's own draw)",
        "weighting": "pooled over scored cells of the retained segments",
        "capped_by": "eval_config.caps.waveforms",
    }
    result["variability"] = variability
    plan["per_channel_capped"] = True
    plan["per_channel_segments"] = int(len(block["sample_index"]))
    result.setdefault("files", [])
    result["files"] = list(result["files"]) + [str(name) for name in files]
    return result
