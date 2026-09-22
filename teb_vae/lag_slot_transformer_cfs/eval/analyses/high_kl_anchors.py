r"""The anchors that carry the divergence: where their proposals sit, and whether they forecast.

This cell's reading of the question the lag-attentive cells ask through ``lag_high_kl``. Averaging
over every anchor can hide what happens where the latent actually moves, so anchors are selected
by their own per-anchor divergence and read on their own: the lag structure of the selection off
the per-anchor proposal map, the forecast gain of the selection against the rest, and the
selection followed through labour on both clocks.

**One set of thresholds for the whole run.** The quantiles of the pooled per-anchor divergence
over every scored anchor within the horizon, applied identically to every segment, window and
class. Three fixed bands -- ``high``, the upper share :data:`HIGH_QUANTILE` cuts; ``rest``, its
complement; ``top``, the narrower share :data:`TOP_QUANTILE` cuts -- and a fourth, ``gain``,
selected on the forecast gain rather than on the
divergence, whose overlap with ``high`` is the usefulness reading: two selections naming the same
anchors is one finding, two that do not is the other.

**Usefulness is assessed through prediction.** A large divergence says the latent moved, not that
the forecast improved. The high band's mean forecast gain against the rest band's is tested once,
paired within recording, and the gain is resolved by divergence decile and by the anchor's
largest-proposal lag.

Everything here is a per-lag readout of one fitted parameterisation and not an allocation; the
proposal map is the head's output at each lag and says nothing about which lag was necessary.
The hot-lag set is a top-share selection taken deliberately and with its circularity stated on
every artifact: the set is chosen from the same map it then summarises.

Reads the per-anchor table and the per-anchor sidecar the collection pass writes, and nothing
else; a directory collected before the sidecar existed records a skip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry
from teb_vae.lag_slot_transformer_cfs.eval import lag_structure

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "high_kl_anchors"

#: What it writes.
THRESHOLDS_FILENAME = "high_kl_thresholds.csv"
SELECTION_FILENAME = "high_kl_selection.csv"
RECORDINGS_FILENAME = "high_kl_recordings.csv"
GAIN_BY_QUANTILE_FILENAME = "high_kl_gain_by_kl_quantile.csv"
ARGMAX_BY_QUANTILE_FILENAME = "high_kl_argmax_by_quantile.csv"
CONTRACTION_FILENAME = "high_kl_contraction.csv"
PER_RECORDING_FILENAME = "high_kl_per_recording.csv"
TRAJECTORY_FILENAME = "high_kl_trajectory.csv"
SIGNIFICANCE_FILENAME = "high_kl_significance.csv"
PAIRWISE_FILENAME = "high_kl_pairwise.csv"
SELECTION_FIGURE = "high_kl_selection"
USEFULNESS_FIGURE = "high_kl_usefulness"
CLOCK_STEM = "high_kl"
WINDOWS_SUFFIX = "windows"

#: The per-anchor columns read, as the collection pass names them.
KL_COLUMN = "kld_per_t"
GAIN_COLUMNS: tuple = ("mean_pred_gap", "mc_pred_gap", "pred_gap")
ARGMAX_COLUMN = "proposal_argmax_lag"
CONTRACTION_COLUMN = "seconds_since_contraction"
PROPOSAL_MAP = "proposal_lag_map"

#: The selection bands and the quantile each is cut at, on the pooled per-anchor divergence.
HIGH_QUANTILE = 0.70
TOP_QUANTILE = 0.90
GAIN_QUANTILE = 0.70

#: How many deciles the gain is resolved over.
N_QUANTILES = 10

#: The quantile of the pooled all-anchor proposal profile above which a lag counts as hot.
HOT_LAG_QUANTILE = 0.70

#: Fewest anchors a recording needs in EACH arm of the contraction enrichment for the difference
#: to be reported.
MIN_CONTRACTION_ANCHORS = 5

#: The per-recording columns the runner fans out by cohort.
GROUPED_COLUMNS: tuple = (
    "high_anchor_frac", "high_pred_gap_nats", "rest_pred_gap_nats", "high_centroid_s",
)

#: The two readouts tested on each clock.
TESTED_COLUMNS: tuple = ("high_anchor_frac", "high_centroid_s")


def figure_stem(clock: lag_structure.Clock, suffix: str = "") -> str:
    """The stem one clock's figure is written under.

    Args:
        clock: The clock.
        suffix: ``''`` or :data:`WINDOWS_SUFFIX`.

    Returns:
        ``high_kl_<clock>`` or ``high_kl_<clock>_<suffix>``.
    """
    stem = f"{CLOCK_STEM}_{clock.name}"
    return f"{stem}_{suffix}" if suffix else stem


# =============================================================================
# The population and its selection
# =============================================================================
def anchor_frame(collection: Any, eval_config: Mapping[str, Any]) -> Tuple[pd.DataFrame, Optional[str]]:
    """The per-anchor table joined to its segment's identity, within the run's horizon.

    Args:
        collection: The collection, read for both tables.
        eval_config: The validated block, for ``max_hours_before_delivery``.

    Returns:
        ``(frame, reason)`` -- the joined rows carrying the sidecar row position, or an empty
        frame with the reason.
    """
    per_anchor = collection.per_anchor
    per_sample = collection.per_sample
    if per_anchor is None or len(per_anchor) == 0:
        return pd.DataFrame(), "the per-anchor table is empty"
    if KL_COLUMN not in per_anchor.columns:
        return pd.DataFrame(), f"the per-anchor table carries no {KL_COLUMN!r} column"
    gain = next((name for name in GAIN_COLUMNS if name in per_anchor.columns), None)
    if gain is None:
        return pd.DataFrame(), "the per-anchor table carries no forecast gain column"
    identity = [
        name for name in lag_structure.IDENTITY_COLUMNS
        if name in per_sample.columns and name not in ("guid", "epoch")
    ]
    kept = cohort.within_horizon(per_sample, eval_config.get("max_hours_before_delivery"))
    frame = per_anchor.reset_index(drop=True)
    frame["sidecar_row"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.merge(
        kept[identity].drop_duplicates("sample_index"), on="sample_index", how="inner"
    )
    frame["gain"] = frame[gain].astype(np.float64)
    frame = frame[np.isfinite(frame[KL_COLUMN].to_numpy(dtype=np.float64))]
    return frame.reset_index(drop=True), None


def thresholds_of(frame: pd.DataFrame) -> Dict[str, float]:
    """The run-level thresholds, on the pooled per-anchor divergence and gain.

    Args:
        frame: The anchor frame.

    Returns:
        ``high_nats``, ``top_nats`` and ``gain_nats``.
    """
    kl = frame[KL_COLUMN].to_numpy(dtype=np.float64)
    gain = frame["gain"].to_numpy(dtype=np.float64)
    finite_gain = gain[np.isfinite(gain)]
    return {
        "high_nats": float(np.quantile(kl, HIGH_QUANTILE)),
        "top_nats": float(np.quantile(kl, TOP_QUANTILE)),
        "gain_nats": float(np.quantile(finite_gain, GAIN_QUANTILE)) if finite_gain.size else float("nan"),
    }


def select_bands(frame: pd.DataFrame, thresholds: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """The four anchor selections as boolean masks over the frame's rows.

    Args:
        frame: The anchor frame.
        thresholds: The run-level thresholds.

    Returns:
        ``{'high', 'rest', 'top', 'gain'}``.
    """
    kl = frame[KL_COLUMN].to_numpy(dtype=np.float64)
    gain = frame["gain"].to_numpy(dtype=np.float64)
    high = kl >= thresholds["high_nats"]
    return {
        "high": high,
        "rest": ~high,
        "top": kl >= thresholds["top_nats"],
        "gain": np.isfinite(gain) & (gain >= thresholds["gain_nats"]),
    }


def proposal_map(collection: Any, frame: pd.DataFrame) -> Optional[np.ndarray]:
    """The per-anchor proposal map at the frame's rows, or ``None`` when the sidecar lacks it.

    Args:
        collection: The collection, read for its per-anchor vectors.
        frame: The anchor frame, carrying the sidecar row positions.

    Returns:
        The $(n, L)$ float64 map.
    """
    vectors = dict(getattr(collection, "anchor_vectors", None) or {})
    values = vectors.get(PROPOSAL_MAP)
    if values is None or np.asarray(values).ndim != 2:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] != len(collection.per_anchor):
        return None
    return array[frame["sidecar_row"].to_numpy(dtype=np.int64)]


def pooled_band_profiles(
    lag_map: np.ndarray, bands: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """The mean proposal map over the anchors of each band, plus over every anchor.

    Args:
        lag_map: The $(n, L)$ map.
        bands: The selections.

    Returns:
        ``{'all', 'high', 'rest', 'top', 'gain'}``, each $(L,)$.
    """
    profiles = {"all": np.nanmean(lag_map, axis=0) if len(lag_map) else np.array([])}
    for name, mask in bands.items():
        profiles[name] = (
            np.nanmean(lag_map[mask], axis=0) if mask.any() else np.full(lag_map.shape[1], np.nan)
        )
    return profiles


def hot_lags(profile: np.ndarray) -> np.ndarray:
    """The lags whose pooled proposal norm sits in the upper share of the profile.

    Args:
        profile: The pooled all-anchor profile, $(L,)$.

    Returns:
        A boolean mask over the lags; all-``False`` on an empty or flat profile.
    """
    values = np.asarray(profile, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.ptp(finite)) == 0.0:
        return np.zeros(values.shape, dtype=bool)
    return np.isfinite(values) & (values >= float(np.quantile(finite, HOT_LAG_QUANTILE)))


def segment_readouts(
    frame: pd.DataFrame,
    bands: Mapping[str, np.ndarray],
    lag_map: Optional[np.ndarray],
    hot: np.ndarray,
    axis: lag_structure.LagAxis,
) -> pd.DataFrame:
    """One row per segment: the selected fractions, the gains and the shape of the selections.

    Args:
        frame: The anchor frame.
        bands: The selections.
        lag_map: The per-anchor proposal map, or ``None``.
        hot: The hot-lag mask.
        axis: The lag axis.

    Returns:
        The segment table with the identity columns and every readout.
    """
    keys = ["sample_index"]
    identity = [
        name for name in lag_structure.IDENTITY_COLUMNS if name in frame.columns and name != "sample_index"
    ]
    groups = frame.groupby("sample_index", sort=True)
    rows: List[Dict[str, Any]] = []
    for sample_index, cell in groups:
        positions = cell.index.to_numpy()
        gain = cell["gain"].to_numpy(dtype=np.float64)
        row: Dict[str, Any] = {"sample_index": int(sample_index)}
        row.update({name: cell[name].iloc[0] for name in identity})
        row["n_anchors"] = int(len(cell))
        for name, mask in bands.items():
            selected = mask[positions]
            row[f"{name}_anchor_frac"] = float(selected.mean()) if len(selected) else float("nan")
            chosen = gain[selected]
            chosen = chosen[np.isfinite(chosen)]
            row[f"{name}_pred_gap_nats"] = float(chosen.mean()) if chosen.size else float("nan")
            if lag_map is not None:
                profile = (
                    np.nanmean(lag_map[positions][selected], axis=0)
                    if selected.any() else np.full(axis.n_lags, np.nan)
                )
                statistics, _census = lag_structure.lag_shape.profile_statistics(
                    profile[None, :], axis.seconds
                )
                row[f"{name}_centroid_s"] = float(statistics["centroid"][0])
                row[f"{name}_spread_s"] = float(statistics["spread"][0])
                row[f"{name}_entropy_nats"] = float(statistics["entropy"][0])
                total = float(np.nansum(np.where(np.isfinite(profile), profile, 0.0)))
                row[f"{name}_hot_share"] = (
                    float(np.nansum(np.where(np.isfinite(profile) & hot, profile, 0.0)) / total)
                    if total > 0.0 else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows, columns=None) if rows else pd.DataFrame(columns=keys + identity)


def gain_by_quantile(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
    """The mean forecast gain per divergence decile, pooled and per class.

    Args:
        frame: The anchor frame.

    Returns:
        ``(rows, edges)`` -- one row per (cohort, decile), and the pooled decile edges in nats.
    """
    kl = frame[KL_COLUMN].to_numpy(dtype=np.float64)
    edges = np.quantile(kl, np.linspace(0.0, 1.0, N_QUANTILES + 1)).tolist() if kl.size else []
    if not edges:
        return pd.DataFrame(columns=["group", "quantile", "kl_lo_nats", "kl_hi_nats", "n_anchors", "mean_gain_nats"]), []
    decile = np.clip(np.searchsorted(edges[1:-1], kl, side="right"), 0, N_QUANTILES - 1)
    rows: List[Dict[str, Any]] = []
    cuts: List[Tuple[str, np.ndarray]] = [("all", np.ones(len(frame), dtype=bool))]
    if labels.CLASS_COLUMN in frame.columns:
        present = cohort.ordered_groups(
            sorted(set(frame[labels.CLASS_COLUMN].dropna().astype(str))), labels.CLASS_COLUMN
        )
        cuts.extend(
            (group, frame[labels.CLASS_COLUMN].astype(str).to_numpy() == group) for group in present
        )
    gain = frame["gain"].to_numpy(dtype=np.float64)
    for group, selector in cuts:
        for index in range(N_QUANTILES):
            chosen = selector & (decile == index) & np.isfinite(gain)
            rows.append({
                "group": group, "quantile": index,
                "kl_lo_nats": float(edges[index]), "kl_hi_nats": float(edges[index + 1]),
                "n_anchors": int(chosen.sum()),
                "mean_gain_nats": float(gain[chosen].mean()) if chosen.any() else float("nan"),
            })
    return pd.DataFrame(rows), [float(edge) for edge in edges]


def argmax_by_quantile(frame: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """The share of anchors in each divergence decile whose largest proposal sits at each lag.

    Args:
        frame: The anchor frame.
        n_lags: The lag count.

    Returns:
        One row per (decile, lag), with the share and the decile's anchor count; empty when the
        table carries no argmax column.
    """
    columns = ["quantile", "lag_step", "n_anchors", "share"]
    if ARGMAX_COLUMN not in frame.columns or len(frame) == 0:
        return pd.DataFrame(columns=columns)
    kl = frame[KL_COLUMN].to_numpy(dtype=np.float64)
    argmax = frame[ARGMAX_COLUMN].to_numpy(dtype=np.int64)
    edges = np.quantile(kl, np.linspace(0.0, 1.0, N_QUANTILES + 1))
    decile = np.clip(np.searchsorted(edges[1:-1], kl, side="right"), 0, N_QUANTILES - 1)
    rows: List[Dict[str, Any]] = []
    for index in range(N_QUANTILES):
        chosen = (decile == index) & (argmax >= 0)
        count = int(chosen.sum())
        for lag in range(n_lags):
            rows.append({
                "quantile": index, "lag_step": lag, "n_anchors": count,
                "share": float((argmax[chosen] == lag).mean()) if count else float("nan"),
            })
    return pd.DataFrame(rows, columns=columns)


def contraction_enrichment(
    frame: pd.DataFrame, high: np.ndarray, window_s: float
) -> pd.DataFrame:
    """Per recording, the high-anchor share inside a contraction window against outside it.

    Args:
        frame: The anchor frame.
        high: The high-band mask.
        window_s: Seconds after a detected contraction within which an anchor counts as inside.

    Returns:
        One row per recording; ``reportable`` is ``False`` below the per-arm anchor minimum.
    """
    columns = [
        "guid", labels.CLASS_COLUMN, "n_inside", "n_outside", "high_share_inside",
        "high_share_outside", "difference", "reportable",
    ]
    if CONTRACTION_COLUMN not in frame.columns or len(frame) == 0:
        return pd.DataFrame(columns=columns)
    age = frame[CONTRACTION_COLUMN].to_numpy(dtype=np.float64)
    inside = np.isfinite(age) & (age >= 0.0) & (age <= float(window_s))
    outside = np.isfinite(age) & ~inside
    rows: List[Dict[str, Any]] = []
    for guid, cell in frame.groupby(frame["guid"].astype(str), sort=True):
        positions = cell.index.to_numpy()
        n_in, n_out = int(inside[positions].sum()), int(outside[positions].sum())
        share_in = float(high[positions][inside[positions]].mean()) if n_in else float("nan")
        share_out = float(high[positions][outside[positions]].mean()) if n_out else float("nan")
        reportable = n_in >= MIN_CONTRACTION_ANCHORS and n_out >= MIN_CONTRACTION_ANCHORS
        rows.append({
            "guid": guid,
            labels.CLASS_COLUMN: (
                cell[labels.CLASS_COLUMN].iloc[0] if labels.CLASS_COLUMN in cell.columns else None
            ),
            "n_inside": n_in, "n_outside": n_out,
            "high_share_inside": share_in, "high_share_outside": share_out,
            "difference": share_in - share_out if reportable else float("nan"),
            "reportable": reportable,
        })
    return pd.DataFrame(rows, columns=columns)


def usefulness_record(
    per_recording: pd.DataFrame, bands: Mapping[str, np.ndarray], *, resamples: int, seed: int
) -> Dict[str, Any]:
    """The high band's gain against the rest band's, paired within recording, and the overlap.

    Args:
        per_recording: The per-recording table.
        bands: The selections, for the overlap of ``high`` and ``gain``.
        resamples: Bootstrap resamples for the interval on the mean paired difference.
        seed: Its seed.

    Returns:
        The paired test, the interval, and the overlap statistics.
    """
    high = per_recording["high_pred_gap_nats"].to_numpy(dtype=np.float64)
    rest = per_recording["rest_pred_gap_nats"].to_numpy(dtype=np.float64)
    usable = np.isfinite(high) & np.isfinite(rest)
    paired = shared_stats.wilcoxon_paired(
        high[usable], rest[usable], label_left="high band gain", label_right="rest band gain"
    )
    interval = shared_stats.bootstrap_ci(high[usable] - rest[usable], resamples=resamples, seed=seed)
    both = bands["high"] & bands["gain"]
    n_high, n_gain = int(bands["high"].sum()), int(bands["gain"].sum())
    return {
        "high_minus_rest_paired": paired,
        "high_minus_rest_mean_interval": interval,
        "overlap": {
            "n_high": n_high, "n_gain": n_gain, "n_both": int(both.sum()),
            "share_of_high_in_gain": float(both.sum() / n_high) if n_high else float("nan"),
            "jaccard": (
                float(both.sum() / (bands["high"] | bands["gain"]).sum())
                if (bands["high"] | bands["gain"]).any() else float("nan")
            ),
            "expected_share_if_independent": 1.0 - GAIN_QUANTILE,
        },
        "reading": (
            "Positive means the anchors carrying the divergence are the anchors where the source "
            "bought forecast; zero or negative means the divergence is not where the usefulness is."
        ),
    }


# =============================================================================
# The figures
# =============================================================================
def build_selection_figure(
    frame: pd.DataFrame,
    thresholds: Mapping[str, float],
    profiles: Mapping[str, np.ndarray],
    hot: np.ndarray,
    argmax_shares: pd.DataFrame,
    contraction: pd.DataFrame,
    axis: lag_structure.LagAxis,
) -> Any:
    """The selection page: the divergence distribution, the restricted profiles, the argmax
    shares by decile and the contraction enrichment.

    Args:
        frame: The anchor frame.
        thresholds: The run-level thresholds.
        profiles: The pooled profiles per band.
        hot: The hot-lag mask.
        argmax_shares: The argmax-by-decile table.
        contraction: The contraction enrichment table.
        axis: The lag axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2, 2, height_per_row=2.6)

    # a. The pooled per-anchor divergence per class, log10, with the two thresholds.
    ax = axes[0, 0]
    kl = frame[KL_COLUMN].to_numpy(dtype=np.float64)
    positive = kl[kl > 0.0]
    if positive.size:
        edges = np.linspace(np.log10(positive.min()), np.log10(positive.max()), 40)
        groups = (
            cohort.ordered_groups(
                sorted(set(frame[labels.CLASS_COLUMN].dropna().astype(str))), labels.CLASS_COLUMN
            )
            if labels.CLASS_COLUMN in frame.columns else []
        )
        colours = figures.group_colors(groups)
        for group in groups:
            values = kl[(frame[labels.CLASS_COLUMN].astype(str).to_numpy() == group) & (kl > 0.0)]
            ax.hist(
                np.log10(values), bins=edges, histtype="stepfilled", alpha=0.35,
                color=colours.get(group), edgecolor=colours.get(group),
                linewidth=figures.LINE_HAIRLINE, label=f"{group} ({values.size} anchors)",
            )
        for name, style in (("high_nats", "--"), ("top_nats", ":")):
            if thresholds[name] > 0.0:
                ax.axvline(
                    np.log10(thresholds[name]), color=figures.COLOR_BLACK, linestyle=style,
                    linewidth=figures.LINE_THIN, label=f"{name[:-5]} threshold",
                )
        ax.set_xlabel("$\\log_{10}$ per-anchor divergence (nats)")
        ax.set_ylabel("anchors")
        ax.legend(fontsize=figures.FONT_SMALL, loc="best")
    else:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
    ax.set_title("Per-anchor divergence and the selection thresholds")
    figures.style_axes(ax)

    # b. The pooled proposal profile of each band, hot lags shaded.
    ax = axes[0, 1]
    drawn = False
    for name, colour in (
        ("all", figures.COLOR_GRAY), ("high", figures.COLOR_VERMILLION),
        ("rest", figures.COLOR_BLUE), ("top", figures.COLOR_PURPLE), ("gain", figures.COLOR_GREEN),
    ):
        profile = np.asarray(profiles.get(name, []), dtype=np.float64)
        if profile.size == axis.n_lags and np.isfinite(profile).any():
            ax.plot(axis.seconds, profile, color=colour, linewidth=figures.LINE_REGULAR, label=name)
            drawn = True
    if drawn:
        half = float(axis.seconds[1] - axis.seconds[0]) / 2.0 if axis.n_lags > 1 else 0.5
        for lag in np.nonzero(hot)[0]:
            ax.axvspan(
                axis.seconds[lag] - half, axis.seconds[lag] + half, color=figures.COLOR_ORANGE,
                alpha=0.12, linewidth=0, zorder=0,
            )
        ax.set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("mean proposal norm (latent units)")
        figures.legend_with_headroom(ax, ncol=5)
    else:
        ax.text(
            0.5, 0.5, "no per-anchor proposal map in the sidecar", transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
            fontstyle="italic",
        )
    ax.set_title("Mean proposal profile of each selection (hot lags shaded)")
    figures.style_axes(ax)

    # c. The share of anchors per decile whose largest proposal sits at each lag.
    ax = axes[1, 0]
    if not argmax_shares.empty:
        field = (
            argmax_shares.pivot(index="lag_step", columns="quantile", values="share")
            .reindex(range(axis.n_lags)).to_numpy(dtype=np.float64)
        )
        figures.heatmap_with_colorbar(
            figure, ax, field[::-1], title="Largest-proposal lag by divergence decile",
            xlabel="divergence decile (low to high)", ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
            symmetric=False, colorbar_label="share of the decile's anchors",
            extent=(-0.5, N_QUANTILES - 0.5, float(axis.seconds[0]) - 0.5, float(axis.seconds[-1]) + 0.5),
            interpolation="none",
        )
    else:
        ax.set_title("Largest-proposal lag by divergence decile")
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
        figures.style_axes(ax)

    # d. The contraction enrichment per class.
    ax = axes[1, 1]
    reportable = contraction[contraction["reportable"].astype(bool)] if len(contraction) else contraction
    samples: Dict[str, Any] = {}
    if len(reportable) and labels.CLASS_COLUMN in reportable.columns:
        for group in cohort.ordered_groups(
            sorted(set(reportable[labels.CLASS_COLUMN].dropna().astype(str))), labels.CLASS_COLUMN
        ):
            samples[group] = reportable.loc[
                reportable[labels.CLASS_COLUMN].astype(str) == group, "difference"
            ].to_numpy(dtype=np.float64)
    figures.violin_panel(
        ax, samples, title="High-anchor share inside minus outside a contraction window",
        ylabel="difference in share, per recording",
        colors=figures.group_colors(list(samples)), reference=0.0, reference_label="no enrichment",
    )
    figures.caveat_note(
        figure,
        "The hot-lag set is chosen from the same proposal map it summarises, so a share on it "
        "describes the run's own selection. " + lag_structure.QUALIFICATION,
    )
    return figure


def build_usefulness_figure(
    gain_rows: pd.DataFrame,
    per_recording: pd.DataFrame,
    usefulness: Mapping[str, Any],
    frame: pd.DataFrame,
    axis: lag_structure.LagAxis,
) -> Any:
    """The usefulness page: gain by decile, high against rest per recording, gain by argmax lag,
    and the overlap of the two selections.

    Args:
        gain_rows: The gain-by-decile table.
        per_recording: The per-recording table.
        usefulness: The usefulness record.
        frame: The anchor frame, for the gain by argmax lag.
        axis: The lag axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2, 2, height_per_row=2.6)

    ax = axes[0, 0]
    groups = [group for group in gain_rows["group"].unique()] if len(gain_rows) else []
    colours = figures.group_colors([group for group in groups if group != "all"])
    colours["all"] = figures.COLOR_BLACK
    for group in groups:
        cell = gain_rows[gain_rows["group"] == group].sort_values("quantile")
        ax.plot(
            cell["quantile"], cell["mean_gain_nats"], marker="o", markersize=figures.MARKER_SMALL,
            color=colours.get(group), linewidth=figures.LINE_REGULAR, label=group,
        )
    if groups:
        ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
        ax.set_xlabel("divergence decile (low to high)")
        ax.set_ylabel("mean forecast gain (nats per anchor)")
        ax.legend(fontsize=figures.FONT_SMALL, loc="best")
    else:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
    ax.set_title("Forecast gain by divergence decile")
    figures.style_axes(ax)

    ax = axes[0, 1]
    high = per_recording["high_pred_gap_nats"].to_numpy(dtype=np.float64) if len(per_recording) else np.array([])
    rest = per_recording["rest_pred_gap_nats"].to_numpy(dtype=np.float64) if len(per_recording) else np.array([])
    usable = np.isfinite(high) & np.isfinite(rest) if high.size else np.array([], dtype=bool)
    if usable.any():
        for left, right in zip(high[usable], rest[usable]):
            ax.plot([0, 1], [right, left], color=figures.COLOR_LIGHT_GRAY, linewidth=figures.LINE_HAIRLINE)
        ax.plot(
            np.zeros(int(usable.sum())), rest[usable], marker="o", linestyle="none",
            markersize=figures.MARKER_SMALL, color=figures.COLOR_BLUE, label="rest band",
        )
        ax.plot(
            np.ones(int(usable.sum())), high[usable], marker="o", linestyle="none",
            markersize=figures.MARKER_SMALL, color=figures.COLOR_VERMILLION, label="high band",
        )
        paired = usefulness["high_minus_rest_paired"]
        interval = usefulness["high_minus_rest_mean_interval"]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["rest", "high"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("mean forecast gain per recording (nats per anchor)")
        ax.text(
            0.02, 0.98,
            f"paired difference {float(interval.get('point', float('nan'))):.3g} "
            f"[{float(interval.get('lo', float('nan'))):.3g}, "
            f"{float(interval.get('hi', float('nan'))):.3g}]; "
            f"Wilcoxon p = {float(paired.get('p_value', float('nan'))):.3g} "
            f"(n = {int(paired.get('n_pairs', 0))})",
            transform=ax.transAxes, ha="left", va="top", fontsize=figures.FONT_TINY,
            color=figures.COLOR_GRAY,
        )
        ax.legend(fontsize=figures.FONT_SMALL, loc="lower right")
    else:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
    ax.set_title("High against rest, paired within recording")
    figures.style_axes(ax)

    ax = axes[1, 0]
    if ARGMAX_COLUMN in frame.columns and len(frame):
        argmax = frame[ARGMAX_COLUMN].to_numpy(dtype=np.int64)
        gain = frame["gain"].to_numpy(dtype=np.float64)
        means = np.array([
            gain[(argmax == lag) & np.isfinite(gain)].mean()
            if ((argmax == lag) & np.isfinite(gain)).any() else np.nan
            for lag in range(axis.n_lags)
        ])
        counts = np.array([int((argmax == lag).sum()) for lag in range(axis.n_lags)])
        ax.bar(
            axis.seconds, means, width=(axis.seconds[1] - axis.seconds[0]) * 0.8 if axis.n_lags > 1 else 1.0,
            color=figures.COLOR_BLUE, alpha=0.8, edgecolor=figures.COLOR_BLACK,
            linewidth=figures.LINE_HAIRLINE,
        )
        for second, mean, count in zip(axis.seconds, means, counts):
            if np.isfinite(mean):
                ax.annotate(
                    str(count), (second, mean), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=figures.FONT_TINY, color=figures.COLOR_GRAY,
                )
        ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
        ax.set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("mean forecast gain (nats per anchor)")
    else:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
    ax.set_title("Forecast gain by the anchor's largest-proposal lag (anchor counts)")
    figures.style_axes(ax)

    ax = axes[1, 1]
    overlap = usefulness["overlap"]
    values = [
        overlap.get("share_of_high_in_gain", float("nan")),
        overlap.get("expected_share_if_independent", float("nan")),
        overlap.get("jaccard", float("nan")),
    ]
    ax.bar(
        [0, 1, 2], values, color=[figures.COLOR_VERMILLION, figures.COLOR_GRAY, figures.COLOR_BLUE],
        alpha=0.8, edgecolor=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE,
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["share of high\nin gain", "expected if\nindependent", "Jaccard"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("share")
    ax.set_title(
        f"Overlap of the high and gain selections "
        f"({overlap.get('n_high', 0)} high, {overlap.get('n_gain', 0)} gain anchors)"
    )
    figures.style_axes(ax)
    figures.caveat_note(figure, lag_structure.QUALIFICATION)
    return figure


def build_clock_figure(
    clock: lag_structure.Clock, rows: Sequence[Mapping[str, Any]]
) -> Any:
    """The tested readouts of the selection against one clock, per class.

    Args:
        clock: The clock.
        rows: This clock's trajectory rows on the class axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(len(TESTED_COLUMNS), height_per_row=2.6)
    for index, column in enumerate(TESTED_COLUMNS):
        lag_structure.draw_trajectory_panel(
            axes[index, 0], [row for row in rows if row["metric"] == column],
            axis=labels.CLASS_COLUMN, clock=clock,
            title=f"{column}: median over recordings per window (tested)",
            ylabel="share of the segment's anchors" if column.endswith("frac") else "stored-coefficient seconds",
        )
    figures.caveat_note(figure, lag_structure.QUALIFICATION)
    return figure


def _skip(reason: str) -> Dict[str, Any]:
    """The recorded skip, logged.

    Args:
        reason: Why nothing was written.

    Returns:
        The protocol's keys with ``n_samples`` ``None``.
    """
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None, "composition": {}, "plan": {"capped": False},
        "skipped": True, "reason": reason, "files": [],
    }


def run_high_kl_anchors_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select the anchors carrying the divergence and read their lag structure and their gain.

    Args:
        context: The analysis context, read for both tables, the per-anchor sidecar and the
            results block.
        eval_config: The validated block, for the horizon, the contraction window, the
            bootstrap resamples and the seed.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the thresholds, the hot lags, the usefulness reading, the clock
        tests and the files written. A recorded skip when the per-anchor table or the sidecar
        is unusable.
    """
    del probe
    collection = context.collection
    results = dict(getattr(collection, "results", None) or {})
    axis = lag_structure.lag_axis_of(results)
    if axis is None:
        return _skip("the run reported no lag axis: this arm has no source pathway")
    frame, reason = anchor_frame(collection, eval_config)
    if reason is not None or frame.empty:
        return _skip(reason or "no anchor lies within the configured horizon")
    lag_map = proposal_map(collection, frame)
    if lag_map is not None and lag_map.shape[1] != axis.n_lags:
        lag_map = None

    thresholds = thresholds_of(frame)
    bands = select_bands(frame, thresholds)
    profiles = (
        pooled_band_profiles(lag_map, bands) if lag_map is not None
        else {name: np.full(axis.n_lags, np.nan) for name in ("all", *bands)}
    )
    hot = hot_lags(profiles["all"])
    segments = segment_readouts(frame, bands, lag_map, hot, axis)
    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    readout_columns = [
        name for name in segments.columns
        if name not in lag_structure.IDENTITY_COLUMNS and name != "n_anchors"
    ]
    per_recording = lag_structure.per_recording_table(segments, readout_columns)
    usefulness = usefulness_record(per_recording, bands, resamples=resamples, seed=seed)
    gain_rows, edges = gain_by_quantile(frame)
    argmax_rows = argmax_by_quantile(frame, axis.n_lags)
    contraction = contraction_enrichment(
        frame, bands["high"], float(eval_config.get("event_lag_window_s", 120.0))
    )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"band": "high", "quantile": HIGH_QUANTILE, "threshold_nats": thresholds["high_nats"],
         "n_anchors": int(bands["high"].sum())},
        {"band": "top", "quantile": TOP_QUANTILE, "threshold_nats": thresholds["top_nats"],
         "n_anchors": int(bands["top"].sum())},
        {"band": "gain", "quantile": GAIN_QUANTILE, "threshold_nats": thresholds["gain_nats"],
         "n_anchors": int(bands["gain"].sum())},
    ]).to_csv(directory / THRESHOLDS_FILENAME, index=False)
    pd.DataFrame({
        "lag_step": np.arange(axis.n_lags), "seconds": axis.seconds, "hot": hot,
        **{f"pooled_{name}_proposal_norm": np.asarray(profile) for name, profile in profiles.items()},
    }).to_csv(directory / SELECTION_FILENAME, index=False)
    per_recording.to_csv(directory / RECORDINGS_FILENAME)
    gain_rows.to_csv(directory / GAIN_BY_QUANTILE_FILENAME, index=False)
    argmax_rows.to_csv(directory / ARGMAX_BY_QUANTILE_FILENAME, index=False)
    contraction.to_csv(directory / CONTRACTION_FILENAME, index=False)

    written = [
        str(figures.render_figure(
            build_selection_figure(frame, thresholds, profiles, hot, argmax_rows, contraction, axis),
            directory / SELECTION_FIGURE,
        ).name),
        str(figures.render_figure(
            build_usefulness_figure(gain_rows, per_recording, usefulness, frame, axis),
            directory / USEFULNESS_FIGURE,
        ).name),
    ]

    # The selection through labour, on both clocks.
    tall: List[pd.DataFrame] = []
    rows: List[Dict[str, Any]] = []
    tests: List[Dict[str, Any]] = []
    clocks: Dict[str, Any] = {}
    tested = [column for column in TESTED_COLUMNS if column in segments.columns]
    for clock in lag_structure.CLOCKS:
        binned, population = lag_structure.clock_rows(segments, clock)
        per_axis = lag_structure.per_recording_by_axis(binned, readout_columns, clock)
        clock_rows = lag_structure.trajectory_rows(per_axis, readout_columns, clock)
        rows.extend(clock_rows)
        for group_axis, part in per_axis.items():
            if len(part):
                tall.append(part.assign(clock=clock.name, group_column=group_axis))
        class_frame = per_axis.get(labels.CLASS_COLUMN, pd.DataFrame())
        records = [lag_structure.windowed_tests(class_frame, column, clock) for column in tested]
        tests.extend(records)
        written.append(str(figures.render_figure(
            build_clock_figure(
                clock, [row for row in clock_rows if row["group_column"] == labels.CLASS_COLUMN]
            ),
            directory / figure_stem(clock),
        ).name))
        written.append(str(figures.render_figure(
            lag_structure.windows_page(
                class_frame, [(column, record, column) for column, record in zip(tested, records)],
                clock,
            ),
            directory / figure_stem(clock, WINDOWS_SUFFIX),
        ).name))
        clocks[clock.name] = {
            **population,
            "n_windows": int(binned[clock.bin_column].nunique()) if len(binned) else 0,
        }
    pd.DataFrame(rows).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    (pd.concat(tall, ignore_index=True) if tall else pd.DataFrame()).to_csv(
        directory / PER_RECORDING_FILENAME, index=False
    )
    lag_structure.significance_frame(tests).to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    lag_structure.pairwise_frame(tests).to_csv(directory / PAIRWISE_FILENAME, index=False)

    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(frame)} anchor(s) of {len(segments)} segment(s); high band "
        f"{int(bands['high'].sum())} anchor(s) above {thresholds['high_nats']:.3g} nats; "
        f"{int(hot.sum())} hot lag(s)"
    )
    return {
        "n_samples": int(len(segments)),
        "composition": {"n_recordings": int(len(per_recording)), "n_anchors": int(len(frame)), **clocks},
        "plan": {"capped": True, "reason": "the second-stage clock admits eligible recordings only"},
        "thresholds": {**thresholds, "high_quantile": HIGH_QUANTILE, "top_quantile": TOP_QUANTILE,
                       "gain_quantile": GAIN_QUANTILE},
        "bands": {name: int(mask.sum()) for name, mask in bands.items()},
        "hot_lags": {
            "lag_steps": [int(lag) for lag in np.nonzero(hot)[0]],
            "quantile": HOT_LAG_QUANTILE,
            "circularity": (
                "chosen from the pooled all-anchor proposal profile and then summarised on it; "
                "a share on the hot set describes the run's own selection"
            ),
        },
        "proposal_map_present": lag_map is not None,
        "usefulness": {**usefulness, "gain_by_kl_quantile_edges_nats": edges},
        "contraction_enrichment": {
            "n_reportable": int(contraction["reportable"].astype(bool).sum()) if len(contraction) else 0,
            "min_anchors_per_arm": MIN_CONTRACTION_ANCHORS,
        },
        "tested": tested,
        "significance": [lag_structure.summary_of(record) for record in tests],
        "qualification": lag_structure.QUALIFICATION,
        "grouped_frames": [
            grouped_frame_entry(
                ANALYSIS_DIRNAME, RECORDINGS_FILENAME,
                [name for name in GROUPED_COLUMNS if name in per_recording.columns],
            ),
        ],
        "files": [
            THRESHOLDS_FILENAME, SELECTION_FILENAME, RECORDINGS_FILENAME,
            GAIN_BY_QUANTILE_FILENAME, ARGMAX_BY_QUANTILE_FILENAME, CONTRACTION_FILENAME,
            TRAJECTORY_FILENAME, PER_RECORDING_FILENAME, SIGNIFICANCE_FILENAME,
            PAIRWISE_FILENAME, *written,
        ],
    }
