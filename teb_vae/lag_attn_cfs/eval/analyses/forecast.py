r"""Is the forecast any good, against predictors that know nothing, and where in the horizon.

Every other readout in this pipeline is about the *coupling* -- what the source adds. This one is
about the forecast itself, and it exists because a block score alone cannot answer the question. A
block score of several hundred nats per anchor is not a number anybody can judge: it is a negative
log density summed over $H \cdot C_{\mathrm{keep}}$ coefficients, so it is large under every
predictor and its scale is set by the block size rather than by the model. Two things make it
readable, and a third says where in the forecast window the answer holds.

**Baselines.** The same loss function, the same mask, the same anchors, applied to three predictors
that know nothing: persistence, climatology, and the segment's own mean
(:func:`~teb_vae.lag_attn_cfs.eval.metrics.baseline_forecasts`). Two skill columns come out of
them and they answer different questions:

* $\mathrm{skill} = 1 - \mathrm{MSE}_{\mathrm{model}} / \mathrm{MSE}_{\mathrm{baseline}}$, where
  the observation variance cancels. A forecast equal to the truth scores $1$; one equal to the
  baseline scores $0$.
* $\mathrm{advantage} = D_{\mathrm{baseline}} - D_{\mathrm{model}}$ in nats per anchor. A
  *difference*, not $1 - $ a ratio, because a log score has no natural zero: the ratio of two
  negative log densities is not bounded above by $1$ and changes sign with the baseline's.

Both are reported because a learned-variance model otherwise beats a fixed-variance baseline
partly on variance modelling alone, and no single number separates the two effects. The baselines'
$\sigma$ is fixed, stated and recorded for the same reason.

**Units, and the conversion that is deliberately absent.** Everything stays in the loader's $z$
units, labelled ``normalised``. A scattering or phase-harmonic coefficient has no clinical unit,
and inverting the per-channel statistics would put the $C_{\mathrm{keep}}$ scored channels on scales spanning
orders of magnitude -- which destroys every pooled statistic here: the mean squared error, the
skill ratio and the shared axis of every figure below. So there is no second unit and no column
carrying one; the ``normalised`` label exists to say that out loud rather than to leave a bare
number to be read as whatever the reader assumes.

**The horizon axis.** $D(\tau)$ answers whether the forecast -- and the source's contribution to
it -- holds up at a minute or only at four seconds. Two properties are load-bearing. It is built on
the **single-draw** path, because the Monte Carlo marginalisation does not commute with the sum
over $\tau$: by Jensen,
$\sum_\tau -\log \frac{1}{K}\sum_r e^{-D_r(\tau)} \neq -\log \frac{1}{K}\sum_r e^{-\sum_\tau
D_r(\tau)}$, so a marginalised curve would not sum back to the marginalised headline. And its
denominator is the per-$\tau$ masked anchor count rather than the per-anchor contributing
indicator, which is an ``amax`` over $\tau$ and would count masked late-horizon steps as scored
zeros -- flattering exactly the horizons that fall in gaps.

**Everything here is per recording**, with one stated exception. Anchors overlap in $H - 1$ of their
$H$ horizon steps and one recording contributes tens of segments, so every statistic is averaged
within a recording first and the bootstrap resamples recordings, never anchors. The exception is
the anchor profile, which averages the retained segments' per-anchor scores directly -- it is a
picture of where in the segment the score sits, not a cohort statistic, and `FIGURE_GUIDE.md` says
so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from teb_vae.lag_attn_cfs.eval import cohort, events, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import band_partition as shared_bands
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import finite_column as _finite_column
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry
from teb_vae.lag_attn_cfs.eval.frames import per_recording_means
from teb_vae.lag_attn_cfs.eval.frames import skill_against
from teb_vae.lag_attn_cfs.eval.metrics import (
    BASELINE_LOGVAR,
    BASELINE_NAMES,
    FORECAST_BRANCHES,
    NORMALISED_UNIT,
)
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "forecast"

#: What it writes. One per-recording table, one long-form skill table, the horizon curve and the
#: anchor profile -- each a CSV a reader can open with ``pandas`` and no import from this package.
SCORES_FILENAME = "forecast_scores.csv"
SKILL_FILENAME = "forecast_skill.csv"
HORIZON_FILENAME = "forecast_horizon.csv"
ANCHOR_FILENAME = "forecast_anchor_profile.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` will name them.
BASELINE_FIGURE = "baseline_comparison"
ANCHOR_FIGURE = "anchor_profile"
OVERLAY_FIGURE = "forecast_overlay"
HORIZON_FIGURE = "horizon_skill"

#: The model branches whose skill is reported. The two negative controls are the coupling
#: analysis's subject, not the forecast's: a stranger's source is not a forecast of anything.
MODEL_BRANCHES: Tuple[str, ...] = ("base", "full")

#: The baseline $R^2$ is measured against, and the closed set it must be one of.
#:
#: $R^2$ is a skill score whose reference is usually left implicit -- "explained variance" against
#: an unstated null. Here the reference is a key, checked against this enum, because against the
#: segment mean and against climatology it is a different number and the difference is the whole
#: content of the claim.
R2_REFERENCES: Tuple[str, ...] = BASELINE_NAMES
R2_REFERENCE = "climatology"

#: The per-recording columns the skill arithmetic reads, by branch.
_BLOCK_COLUMN = "nll_{branch}_block"
_SQUARED_ERROR_COLUMN = "sq_error_{branch}"

#: The forecast overlay's layout: at most this many target channels under the raw rows (one per
#: clinical band plus one phase-harmonic channel, see :func:`overlay_channels`), for this many
#: retained samples side by side, each with this many horizons of target history before its
#: anchor -- as long a past as the future it is asked to extend.
OVERLAY_CHANNELS = 5
OVERLAY_SAMPLES = 3
OVERLAY_HISTORY_HORIZONS = 1

#: The overlay's panel size in inches: one column per sample, one row per signal or channel.
OVERLAY_COLUMN_WIDTH_IN = 3.4
OVERLAY_ROW_HEIGHT_IN = 1.15

#: Offset added to the run's seed for the overlay's sample draw, so it is not another draw's rows.
OVERLAY_SEED_OFFSET = 11

#: The table naming every sample the overlay draws -- the identity a figure lifted out of the run
#: must still be traceable to -- and the kept-axis channel map it names the channels from, written
#: into the results root by the channel-map step.
OVERLAY_SAMPLES_FILENAME = "forecast_overlay_samples.csv"
KEPT_CHANNEL_MAP_FILENAME = "band_channel_map_kept.csv"

#: The metrics resolved by cohort: the two model branches' squared error and the full branch's
#: block score. Not the baselines' columns, which describe how forecastable a cohort's recordings
#: are rather than how the model did on them -- and which the skill scores already divide out.
GROUPED_METRICS: Tuple[str, ...] = tuple(
    [_SQUARED_ERROR_COLUMN.format(branch=name) for name in MODEL_BRANCHES]
    + [_BLOCK_COLUMN.format(branch="full"), "signed_error_full"]
)


# =============================================================================
# Skill
# =============================================================================
def build_skill_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Score every model branch against every baseline, in both spaces, with uncertainty.

    Args:
        per_guid: Per-recording means, as :func:`per_recording_means` returns them.
        resamples: Bootstrap resamples, from ``eval_config.bootstrap_resamples``.
        seed: Bootstrap seed, from ``eval_config.seed``, so the intervals are reproducible from
            the summary alone.

    Returns:
        One row per ``(branch, baseline)`` pair, each carrying the two skill statistics, their
        confidence intervals, the paired signed-rank test on the log-score difference, and the
        honest $n$ behind all three.
    """
    rows: List[Dict[str, Any]] = []
    for branch in MODEL_BRANCHES:
        model_sq = _finite_column(per_guid, _SQUARED_ERROR_COLUMN.format(branch=branch))
        model_block = _finite_column(per_guid, _BLOCK_COLUMN.format(branch=branch))
        for baseline in BASELINE_NAMES:
            baseline_sq = _finite_column(per_guid, _SQUARED_ERROR_COLUMN.format(branch=baseline))
            baseline_block = _finite_column(per_guid, _BLOCK_COLUMN.format(branch=baseline))

            skill = skill_against(model_sq, baseline_sq)
            # Positive means the model scores fewer nats per anchor than the baseline does.
            advantage = baseline_block - model_block
            skill_ci = shared_stats.bootstrap_ci(skill, resamples=resamples, seed=seed)
            advantage_ci = shared_stats.bootstrap_ci(advantage, resamples=resamples, seed=seed)
            paired = shared_stats.wilcoxon_paired(
                baseline_block, model_block,
                label_left=f"{baseline} block score", label_right=f"{branch} block score",
            )
            rows.append(
                {
                    "branch": branch,
                    "baseline": baseline,
                    "n_recordings": int(skill_ci["n"]),
                    "mse_skill": skill_ci["point"],
                    "mse_skill_lo": skill_ci["lo"],
                    "mse_skill_hi": skill_ci["hi"],
                    "advantage_nats_per_anchor": advantage_ci["point"],
                    "advantage_lo": advantage_ci["lo"],
                    "advantage_hi": advantage_ci["hi"],
                    "model_mean_squared_error": float(np.nanmean(model_sq))
                    if np.isfinite(model_sq).any() else float("nan"),
                    "baseline_mean_squared_error": float(np.nanmean(baseline_sq))
                    if np.isfinite(baseline_sq).any() else float("nan"),
                    "wilcoxon_p_value": paired["p_value"],
                    "wilcoxon_n_pairs": paired["n_pairs"],
                    "is_r2_reference": baseline == R2_REFERENCE,
                }
            )
    return rows


def build_error_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    r"""Report each model branch's point-forecast error, per scored coefficient, in $z$ units.

    The RMSE roots **once**, after the per-recording mean of the unrooted squares. Rooting per
    segment and averaging the roots is biased low by Jensen -- in the direction that flatters the
    model -- which is why the collection pass accumulates the squares rather than the roots.

    Args:
        per_guid: Per-recording means.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per branch: MAE, RMSE and bias with their intervals, and the unit label that says
        what scale they are on. Every column is ``normalised``; there is no second unit here and
        the reason is in the module docstring.
    """
    rows: List[Dict[str, Any]] = []
    for branch in MODEL_BRANCHES:
        squares = _finite_column(per_guid, _SQUARED_ERROR_COLUMN.format(branch=branch))
        absolute = _finite_column(per_guid, f"abs_error_{branch}")
        signed = _finite_column(per_guid, f"signed_error_{branch}")

        squares_ci = shared_stats.bootstrap_ci(squares, resamples=resamples, seed=seed)
        absolute_ci = shared_stats.bootstrap_ci(absolute, resamples=resamples, seed=seed)
        signed_ci = shared_stats.bootstrap_ci(signed, resamples=resamples, seed=seed)
        rmse = float(np.sqrt(squares_ci["point"])) if squares_ci["point"] >= 0.0 else float("nan")
        rows.append(
            {
                "branch": branch,
                "n_recordings": int(squares_ci["n"]),
                "unit": NORMALISED_UNIT,
                "rmse_normalised": rmse,
                # The interval is on the mean square, so its bounds are rooted rather than the
                # interval being rebuilt: a monotone transform of a percentile interval is the
                # percentile interval of the transform.
                "rmse_lo_normalised": float(np.sqrt(max(squares_ci["lo"], 0.0))),
                "rmse_hi_normalised": float(np.sqrt(max(squares_ci["hi"], 0.0))),
                "mae_normalised": absolute_ci["point"],
                # Positive means the forecast runs above the truth.
                "bias_normalised": signed_ci["point"],
            }
        )
    return rows


# =============================================================================
# The horizon axis
# =============================================================================
def horizon_curves(horizon: Dict[str, Any]) -> pd.DataFrame:
    r"""Turn the streamed per-$\tau$ accumulators into the horizon-resolved curves.

    $$D_{\mathrm{branch}}(\tau) = \frac{\sum_{b,a} D_{b,a,\tau}}{\sum_{b,a} m_{b,a,\tau}},
    \qquad \mathrm{gap}(\tau) = D_{\mathrm{base}}(\tau) - D_{\mathrm{full}}(\tau).$$

    Args:
        horizon: The collection record's ``horizon`` block, one list per branch and statistic.

    Where the record carries the scored-coefficient counts $n_\tau$, each branch's score is also
    reported per scored coefficient, $S^{D}_\tau / n_\tau$, beside its standardised innovation
    variance and mean log-variance, and the channel count $n_\tau / n^{a}_\tau$ the scored-cell mask
    leaves at each lead.

    Returns:
        One row per horizon step, carrying the lead time in seconds, both branches' scores, the
        gap, and each branch's RMSE. Empty when the record carries no horizon block -- a pass at
        a likelihood or a geometry that produced none must record that rather than invent a
        curve.
    """
    required = ("base_sum_block", "base_n_anchors", "full_sum_block", "full_n_anchors")
    if not horizon or any(name not in horizon for name in required):
        return pd.DataFrame()

    def _mean(numerator: str, denominator: str) -> np.ndarray:
        top = np.asarray(horizon[numerator], dtype=np.float64)
        bottom = np.asarray(horizon[denominator], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(bottom > 0.0, top / bottom, np.nan)

    d_base = _mean("base_sum_block", "base_n_anchors")
    d_full = _mean("full_sum_block", "full_n_anchors")
    steps = np.arange(d_base.size, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "horizon_step": steps.astype(int),
            # Horizon step tau covers decimated step t + 1 + tau, so its lead time from the
            # anchor's causal endpoint ends at 4(tau + 1) seconds. Step 0 is therefore 4 s ahead,
            # not 0 s: the anchor's own block is the past, not the forecast.
            "lead_seconds": (steps + 1.0) * SECONDS_PER_STEP,
            "d_base_nats": d_base,
            "d_full_nats": d_full,
            "gap_nats": d_base - d_full,
            "n_anchors": np.asarray(horizon["base_n_anchors"], dtype=np.float64),
            "score_path": "single-draw (training path)",
        }
    )
    for branch in MODEL_BRANCHES:
        squares = f"{branch}_sum_sq"
        counts = f"{branch}_count"
        if counts not in horizon:
            continue
        # ``count`` is the scored coefficients behind each step -- under the scored-cell mask a far
        # step holds fewer channels than a near one -- so a per-coefficient score is the one that
        # compares across the horizon, where the per-step score also moves with the channel count.
        frame[f"d_{branch}_nats_per_coefficient"] = _mean(f"{branch}_sum_block", counts)
        if squares in horizon:
            # Per coefficient too, matching ``sq_error_*`` on the sample table.
            frame[f"rmse_{branch}_normalised"] = np.sqrt(
                np.clip(_mean(squares, counts), 0.0, None)
            )
            frame["rmse_unit"] = NORMALISED_UNIT
        if f"{branch}_sum_standardised_sq" in horizon:
            # $e^2 / \sigma^2$ on the AR(1) innovation, which a calibrated variance puts at 1.
            frame[f"standardised_sq_{branch}"] = _mean(f"{branch}_sum_standardised_sq", counts)
        if f"{branch}_sum_logvar" in horizon:
            frame[f"mean_logvar_{branch}"] = _mean(f"{branch}_sum_logvar", counts)
    if "full_count" in horizon:
        frame["n_scored_coefficients"] = np.asarray(horizon["full_count"], dtype=np.float64)
        # $\sum_c m_{\tau,c}$: the channels the density holds at this lead. It steps down where
        # the fast phase channels' scored horizon $H_c$ ends.
        frame["scored_channels_per_anchor"] = _mean("full_count", "full_n_anchors")
    return frame


def anchor_profile(per_anchor: pd.DataFrame) -> pd.DataFrame:
    """Average each per-anchor score across every segment that scored that anchor.

    Args:
        per_anchor: The per-anchor table, keyed on the forward's own ``anchor_index`` -- the
            decimated step -- rather than on a position in the decoded set.

    Returns:
        One row per anchor index, with the two block scores, the gap and the contributing count.
        Empty when the table is.
    """
    if per_anchor is None or len(per_anchor) == 0 or "anchor" not in per_anchor.columns:
        return pd.DataFrame()
    columns = [
        name for name in ("nll_base_block", "nll_full_block", "pred_gap")
        if name in per_anchor.columns
    ]
    grouped = per_anchor.groupby("anchor")
    profile = grouped[columns].mean() if columns else pd.DataFrame(index=grouped.size().index)
    profile["n_segments"] = grouped.size()
    return profile.reset_index()


# =============================================================================
# Figures
# =============================================================================
def build_baseline_figure(
    per_guid: pd.DataFrame, skill_rows: Sequence[Dict[str, Any]], *, unit: str
) -> Any:
    """Draw the per-recording score distributions and the skill scores that compare them.

    Args:
        per_guid: Per-recording means.
        skill_rows: The skill table, as :func:`build_skill_rows` returns it.
        unit: The unit the errors are in. Unused by the drawing, because the skill is a unitless
            ratio; kept so the builder's signature matches the record it is called with.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    figures.violin_panel(
        axes[0, 0],
        {
            name: _finite_column(per_guid, _BLOCK_COLUMN.format(branch=name))
            for name in FORECAST_BRANCHES
        },
        title="Block score",
        ylabel="nats per anchor",
    )

    axis = axes[1, 0]
    labels = [f"{row['branch']} vs {row['baseline']}" for row in skill_rows]
    values = np.asarray([float(row["mse_skill"]) for row in skill_rows], dtype=np.float64)
    positions = np.arange(len(skill_rows), dtype=np.float64)
    if values.size and np.isfinite(values).any():
        # Asymmetric whiskers, clipped at zero: a percentile interval is not symmetric about its
        # point estimate and drawing it as if it were would misstate which side the uncertainty
        # is on.
        bounds = np.asarray(
            [[float(row["mse_skill_lo"]), float(row["mse_skill_hi"])] for row in skill_rows],
            dtype=np.float64,
        )
        low = np.clip(values - bounds[:, 0], 0.0, None)
        high = np.clip(bounds[:, 1] - values, 0.0, None)
        axis.barh(positions, values, color=figures.COLOR_BLUE, alpha=0.85, height=0.6)
        axis.errorbar(
            values, positions, xerr=np.vstack([low, high]),
            fmt="none", ecolor=figures.COLOR_BLACK, elinewidth=figures.LINE_REGULAR, capsize=3.0,
        )
        axis.axvline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=figures.FONT_LABEL)
    else:
        axis.text(0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes)
    axis.set_title("Squared-error skill")
    axis.set_xlabel("1 - MSE(model) / MSE(baseline)")
    figures.style_axes(axis)
    return figure


def build_anchor_profile_figure(
    profile: pd.DataFrame, geometry: Dict[str, Any]
) -> Tuple[Any, Dict[str, Tuple[float, float]]]:
    r"""Draw the block scores and the gap against time in segment, structural regions shaded.

    Two spans are shaded and neither is a finding. The prefix $[0, F)$ below the anchor floor holds
    no decoded anchor **at all** -- unlike the raw cells, where the warm-up anchors exist and carry
    no loss term, here the forecast simply starts at $F$ -- and the tail $[T_{\mathrm{valid}}, T)$
    holds the anchors whose forecast window would run past the end of the segment. An unshaded
    profile reads as a model that produces nothing for the first half of every recording.

    Args:
        profile: The per-anchor profile.
        geometry: The collection record's geometry block.

    Returns:
        ``(figure, spans)``, where ``spans`` names the two shaded intervals in decimated steps --
        returned rather than only drawn, so the bounds are checkable without reading the artist.
    """
    floor = int(geometry.get("anchor_floor", 0))
    t_valid = int(geometry.get("t_valid", 0))
    total = int(geometry.get("t", t_valid))
    spans = {
        "below_anchor_floor": (0.0, float(floor)),
        "untrained_tail": (float(t_valid), float(total)),
    }

    figure, axes = figures.new_figure(2)
    anchors = _finite_column(profile, "anchor")
    figures.multi_line_panel(
        axes[0, 0],
        anchors,
        np.vstack(
            [
                _finite_column(profile, "nll_base_block"),
                _finite_column(profile, "nll_full_block"),
            ]
        ),
        ["target-only", "source-conditioned"],
        title="Block score",
        xlabel="anchor (decimated steps)",
        ylabel="nats per anchor",
    )
    figures.multi_line_panel(
        axes[1, 0],
        anchors,
        _finite_column(profile, "pred_gap")[None, :],
        ["pred_gap"],
        title="Forecast gap",
        xlabel="anchor (decimated steps)",
        ylabel="nats per anchor",
    )
    for axis in (axes[0, 0], axes[1, 0]):
        for low, high in spans.values():
            if high > low:
                axis.axvspan(low, high, color=figures.COLOR_LIGHT_GRAY, alpha=0.6, zorder=0)
        axis.set_xlim(0.0, float(total) if total else None)
    return figure, spans


def build_horizon_figure(curves: pd.DataFrame, *, horizon_steps: int) -> Any:
    r"""Draw $D(\tau)$, the gap, and the RMSE against lead time in **seconds**.

    Seconds rather than horizon steps, on every panel: the question the curve answers -- "does the
    source still help a minute out?" -- is asked in seconds, and a reader who has to multiply by
    four is a reader who will eventually forget to.

    Args:
        curves: The horizon table.
        horizon_steps: $H$, so the axis spans the whole forecast window even where the curve is
            shorter or empty.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(4)
    lead = _finite_column(curves, "lead_seconds")
    # Per scored coefficient where the record carries the counts: the per-step score also falls
    # wherever the fast channels' scored horizon ends, which is fewer terms and not a better
    # forecast. Panel d draws that channel count.
    per_coefficient = "d_base_nats_per_coefficient" in getattr(curves, "columns", [])
    suffix = "_per_coefficient" if per_coefficient else ""
    figures.multi_line_panel(
        axes[0, 0], lead,
        np.vstack(
            [_finite_column(curves, f"d_base_nats{suffix}"), _finite_column(curves, f"d_full_nats{suffix}")]
        ),
        ["target-only", "source-conditioned"],
        title="Forecast score",
        xlabel="lead time (s)",
        ylabel="nats per scored coefficient" if per_coefficient else "nats per horizon step",
    )
    figures.multi_line_panel(
        axes[1, 0], lead, _finite_column(curves, "gap_nats")[None, :], ["pred_gap"],
        title="Forecast gap",
        xlabel="lead time (s)", ylabel="nats per horizon step",
    )
    has_unit = len(curves) and "rmse_unit" in getattr(curves, "columns", [])
    unit = str(curves["rmse_unit"].iloc[0]) if has_unit else NORMALISED_UNIT
    figures.multi_line_panel(
        axes[2, 0], lead,
        np.vstack(
            [_finite_column(curves, f"rmse_{branch}_normalised") for branch in MODEL_BRANCHES]
        ),
        list(MODEL_BRANCHES),
        title="Forecast error",
        xlabel="lead time (s)", ylabel=f"RMSE ({unit})",
    )
    figures.multi_line_panel(
        axes[3, 0], lead, _finite_column(curves, "scored_channels_per_anchor")[None, :],
        ["scored channels"],
        title="Scored channels",
        xlabel="lead time (s)", ylabel="channels per anchor",
    )
    for axis in axes[:, 0]:
        axis.set_xlim(0.0, float(horizon_steps) * SECONDS_PER_STEP)
    return figure


def overlay_channels(
    width: int,
    count: int = OVERLAY_CHANNELS,
    kept_map: Optional[pd.DataFrame] = None,
    scored_horizon: Optional[np.ndarray] = None,
) -> List[int]:
    r"""Choose which kept target channels the overlay draws, deterministically.

    With the kept-axis channel map: the middle scattering channel of every clinical band, in
    ascending frequency, and one phase-harmonic channel -- the one scored over the fewest horizon
    steps when $H_c$ is known, since its unscored cells are what the figure has to show. Without
    the map, evenly spaced positions. Fixed either way, so two runs of one checkpoint draw the same
    channels and a figure can be compared across arms rather than only read.

    Args:
        width: $C_{\mathrm{keep}}$, the retained block's channel axis.
        count: How many to draw at most.
        kept_map: The persisted kept-axis channel map, or ``None``.
        scored_horizon: $H_c$ per kept channel, or ``None``.

    Returns:
        Ascending channel positions on the kept axis, without duplicates. Empty when the block
        has no channels at all.
    """
    if width <= 0 or count <= 0:
        return []
    usable = (
        kept_map is not None and len(kept_map) == int(width)
        and {"kept_channel", "band"} <= set(kept_map.columns)
    )
    if not usable:
        if width <= count:
            return list(range(width))
        positions = np.linspace(0, width - 1, num=count)
        return sorted({int(round(float(value))) for value in positions})
    ordered = kept_map.sort_values("kept_channel")
    position = ordered["kept_channel"].to_numpy(dtype=np.int64)
    band = ordered["band"].astype(str).to_numpy()
    scattering = (
        ordered["block"].astype(str).to_numpy() == "scattering"
        if "block" in ordered.columns else np.ones(position.size, dtype=bool)
    )
    picks: List[int] = []
    for name in shared_bands.CLINICAL_BANDS:
        members = position[(band == name) & scattering]
        if members.size:
            picks.append(int(members[members.size // 2]))
    phase = position[~scattering]
    if phase.size:
        fastest = (
            int(phase[int(np.argmin(np.asarray(scored_horizon)[phase]))])
            if scored_horizon is not None else int(phase[phase.size // 2])
        )
        picks.append(fastest)
    return sorted(dict.fromkeys(picks[:count]))


def channel_label(kept_map: Optional[pd.DataFrame], channel: int) -> str:
    """Name a kept channel by what it is: its declared index, its kind, its frequency and band."""
    if kept_map is None or "kept_channel" not in kept_map.columns:
        return f"kept channel {channel}"
    rows = kept_map[kept_map["kept_channel"] == channel]
    if rows.empty:
        return f"kept channel {channel}"
    row = rows.iloc[0]
    frequencies = [
        float(row[name]) for name in ("freq_hz_primary", "freq_hz_secondary")
        if name in row.index and pd.notna(row[name])
    ]
    hertz = "/".join(f"{value:.3g}" for value in frequencies) + " Hz" if frequencies else "no centre frequency"
    return f"declared ch {int(row.get('channel', channel))} {row.get('kind', '')}, {hertz} ({row.get('band', '')})"


def overlay_density_terms(
    context: Any, record: Dict[str, Any], width: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    r"""The scored horizon $H_c$ and the AR(1) coefficient $\phi_c$ per kept channel.

    Read off the collection record's ``likelihood_structure`` -- written from the model's own
    ``forecast_likelihood_kwargs``, the terms every score in this run was computed under -- so the
    offline re-run draws the same boundary and band as the pass did, and this analysis stays one
    that never reaches for the model. A record from before the per-channel vectors existed falls
    back to $H_c$ re-resolved from the run's configuration and its shards, with no $\phi_c$.

    Args:
        context: The analysis context, read for the merged config on the fallback path.
        record: The collection record, for the likelihood structure and the kept-channel index.
        width: $C_{\mathrm{keep}}$.

    Returns:
        ``(scored_horizon, ar_coef)``, each $(C_{\mathrm{keep}},)$ or ``None`` where unknown.
        ``None`` for $H_c$ on a model without a scored-cell mask means every channel is scored
        over the whole horizon.
    """
    structure = dict(record.get("likelihood_structure") or {})

    def _vector(name: str, dtype: Any) -> Optional[np.ndarray]:
        values = structure.get(name)
        if values is None:
            return None
        array = np.asarray(values, dtype=dtype).reshape(-1)
        return array if array.size == int(width) else None

    if "scored_horizon_per_channel" in structure or "ar_coef_per_channel" in structure:
        return (
            _vector("scored_horizon_per_channel", np.int64),
            _vector("ar_coef_per_channel", np.float64),
        )
    try:
        from teb_vae.lag_attn_cfs.scored_horizon import resolve_target_scored_horizon

        declared = resolve_target_scored_horizon(dict(getattr(context, "config", None) or {}))
    except Exception:  # noqa: BLE001 - an unreadable shard leaves the boundary undrawn, not wrong
        declared = None
    if declared is None:
        return None, None
    declared = np.asarray(declared, dtype=np.int64)
    keep = (record.get("geometry") or {}).get("target_keep_index")
    scored = declared if keep is None else declared[np.asarray(keep, dtype=np.int64)]
    return (scored if scored.size == int(width) else None), None


def marginal_variance(innovation_variance: np.ndarray, phi: Optional[float]) -> np.ndarray:
    r"""The forecast variance of each horizon step under the AR(1) residual.

    $$v_\tau = \sigma^2_\tau + \phi_c^2\, v_{\tau-1}, \qquad v_{-1} = 0,$$

    because the density scores the innovation $e_\tau = r_\tau - \phi_c r_{\tau-1}$, so the
    residual $r_\tau$ itself accumulates the earlier steps' innovations. ``phi=None`` returns the
    innovation variance, which is the marginal one only without the AR term.

    Args:
        innovation_variance: $\sigma^2_\tau$ along the horizon, $(H,)$.
        phi: $\phi_c$, or ``None``.

    Returns:
        $v_\tau$, $(H,)$.
    """
    variance = np.asarray(innovation_variance, dtype=np.float64).reshape(-1)
    if phi is None:
        return variance
    out = np.empty_like(variance)
    carried = 0.0
    for step, value in enumerate(variance):
        carried = float(value) + float(phi) ** 2 * carried
        out[step] = carried
    return out


def step_series(block: np.ndarray, *, first: int, stride: int) -> np.ndarray:
    r"""Lay one sample's gathered $(A, H)$ block out on the decimated step axis it came from.

    Horizon step $\tau$ of the anchor at step $t$ is step $t + 1 + \tau$, so every stored step the
    decoded anchors look ahead to is recovered -- including the history before any one anchor,
    which is an earlier anchor's future.

    Args:
        block: $(A, H)$, one channel of the retained target.
        first: The first decoded anchor's step.
        stride: The decoded anchors' stride.

    Returns:
        The coefficient at every step, ``NaN`` at steps no anchor looks ahead to.
    """
    anchors, horizon = block.shape
    steps = first + stride * np.arange(anchors)[:, None] + 1 + np.arange(horizon)[None, :]
    series = np.full(int(steps.max()) + 1, np.nan)
    series[steps.ravel()] = np.asarray(block, dtype=np.float64).ravel()
    return series


def overlay_samples(
    retained: Dict[str, np.ndarray],
    per_sample: pd.DataFrame,
    per_anchor: Optional[pd.DataFrame],
    *,
    geometry: Dict[str, Any],
    count: int = OVERLAY_SAMPLES,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Choose the retained samples the overlay draws, one per clinical class first, and their anchor.

    The rows are a seeded draw over the retained set taken class by class in the evaluation's
    cohort order, so the three classes sit side by side where the retention holds them. The anchor
    is each sample's median **scored** anchor from the per-anchor table -- the same relative place
    in every segment -- or the middle decoded position where the table cannot say.

    Args:
        retained: The collection's retained arrays.
        per_sample: The per-sample table, for each retained row's identity.
        per_anchor: The per-anchor table, for the scored anchors, or ``None``.
        geometry: The collection record's geometry.
        count: Samples to draw at most.
        seed: The draw's seed.

    Returns:
        One record per sample: the retained row, the anchor position and step, and the identity
        (``guid``, ``subgroup``, ``clinical_class``, ``epoch``, ``sample_index``).
    """
    target = retained["target"]
    n_rows, n_anchors = int(target.shape[0]), int(target.shape[1])
    first = int(geometry.get("anchor_first", geometry.get("anchor_floor", 0)) or 0)
    stride = int(geometry.get("anchor_stride") or 1)
    index = np.asarray(retained.get("waveforms_sample_index", np.arange(n_rows)), dtype=np.int64)
    table = (
        per_sample.set_index("sample_index", drop=False)
        if "sample_index" in per_sample.columns else per_sample
    )

    def _identity(row: int) -> Dict[str, Any]:
        key = int(index[row]) if row < index.size else row
        found = table.loc[key] if key in table.index else None
        found = found.iloc[0] if isinstance(found, pd.DataFrame) else found
        record: Dict[str, Any] = {
            name: None if found is None or name not in found.index else found[name]
            for name in ("guid", labels.SUBGROUP_COLUMN, labels.CLASS_COLUMN, "epoch")
        }
        record["sample_index"] = key
        return record

    identities = [_identity(row) for row in range(n_rows)]
    order = np.random.default_rng(int(seed)).permutation(n_rows)
    classes = [str(item[labels.CLASS_COLUMN]) for item in identities if item[labels.CLASS_COLUMN] is not None]
    chosen: List[int] = []
    for name in cohort.ordered_groups(list(dict.fromkeys(classes)), labels.CLASS_COLUMN):
        match = [int(row) for row in order if str(identities[row][labels.CLASS_COLUMN]) == name]
        if match:
            chosen.append(match[0])
    chosen += [int(row) for row in order if int(row) not in chosen]

    samples: List[Dict[str, Any]] = []
    for row in chosen[: int(count)]:
        position = n_anchors // 2
        if per_anchor is not None and {"sample_index", "anchor"} <= set(per_anchor.columns):
            scored = np.sort(np.asarray(
                per_anchor.loc[per_anchor["sample_index"] == identities[row]["sample_index"], "anchor"],
                dtype=np.int64,
            ))
            if scored.size:
                position = int(np.clip((scored[scored.size // 2] - first) // stride, 0, n_anchors - 1))
        step = first + stride * position
        epoch = identities[row]["epoch"]
        samples.append(
            {
                **identities[row], "row": row, "anchor_position": position, "anchor": step,
                # The trace convention: the absolute time of the anchor's step, before delivery.
                "hours_before_delivery": (
                    -float(traces.absolute_seconds(epoch, step)) / cohort.SECONDS_PER_HOUR
                    if epoch is not None and pd.notna(epoch) else float("nan")
                ),
            }
        )
    return samples


def build_overlay_figure(
    retained: Dict[str, np.ndarray],
    samples: Sequence[Dict[str, Any]],
    channels: Sequence[int],
    *,
    geometry: Dict[str, Any],
    channel_labels: Optional[Sequence[str]] = None,
    scored_horizon: Optional[np.ndarray] = None,
    ar_coef: Optional[np.ndarray] = None,
    raw_scales: Optional[Dict[str, Tuple[float, float]]] = None,
    predictive_band: bool = True,
) -> Any:
    r"""Draw retained samples side by side: the raw signals, then each chosen channel's forecast.

    One column per sample, on one time axis in seconds from the anchor's causal endpoint, so
    $x < 0$ is what the model had read and $x > 0$ the forecast window of $H$ steps. The top rows
    are the raw FHR and UP on the fixed CTG scale, the contractions the event detector found shaded
    on UP; under them one row per chosen channel carries the target's history, the truth over the
    horizon, both branches' means and the predictive $\pm 1\sigma$ band. Cells past a channel's
    scored horizon $H_c$ are shaded: the decoder emits them but the density does not contain them.
    Every row shares its y-axis across the samples, so a column is compared with its neighbours
    rather than only read.

    Args:
        retained: The collection's retained arrays: ``target``, ``mu_*`` and ``logvar_*`` as
            $(N, A_{\max}, H, C_{\mathrm{keep}})$, ``up_raw`` / ``fhr_raw`` as $(N, R)$ and
            ``weight`` as $(N, T)$. Whatever is absent is left out of the drawing, not invented.
        samples: From :func:`overlay_samples`.
        channels: Positions on the kept channel axis.
        geometry: The collection record's geometry, for $H$ and the decoded anchor grid.
        channel_labels: One label per channel, or ``None`` for the kept index.
        scored_horizon: $H_c$ per kept channel, or ``None`` when every cell is scored.
        ar_coef: $\phi_c$ per kept channel, or ``None``; widens the band to the marginal variance.
        raw_scales: From :func:`~teb_vae.lag_attn_cfs.eval.traces.raw_signal_scales`.
        predictive_band: Whether the log-variance head was trained, so a band means anything.

    Returns:
        The figure, already laid out; the caller renders and closes it.
    """
    target = np.asarray(retained["target"])
    horizon = int(geometry.get("horizon") or target.shape[2])
    first = int(geometry.get("anchor_first", geometry.get("anchor_floor", 0)) or 0)
    stride = int(geometry.get("anchor_stride") or 1)
    history = OVERLAY_HISTORY_HORIZONS * horizon
    x_limits = (-history * SECONDS_PER_STEP, (horizon + 0.5) * SECONDS_PER_STEP)
    labels_of = list(channel_labels or [f"kept channel {channel}" for channel in channels])
    rows = [*traces.RAW_SIGNAL_FIELDS, *channels]
    columns = max(len(samples), 1)
    figure, axes = figures.new_figure(
        len(rows), columns, height_per_row=OVERLAY_ROW_HEIGHT_IN,
        width=OVERLAY_COLUMN_WIDTH_IN * columns + 0.9,
    )

    for column, sample in enumerate(samples):
        row, position, step = int(sample["row"]), int(sample["anchor_position"]), int(sample["anchor"])
        weight = retained.get("weight")
        for index, name in enumerate(traces.RAW_SIGNAL_FIELDS):
            axis = axes[index, column]
            values = retained.get(f"{name}_raw")
            if values is None or row >= len(values):
                axis.text(0.5, 0.5, "not retained", transform=axis.transAxes,
                          ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY)
                axis.set_ylabel(name.upper())
                axis.set_yticks([])
                continue
            raw = np.asarray(values[row], dtype=np.float64).reshape(-1)
            valid = None if weight is None else traces.raw_validity_of(weight[row], raw.size)
            physical, unit = traces.physical_raw(raw, name, raw_scales or {}, valid)
            # Raw sample i ends at (i + 1) / f_s into the segment; the anchor's step ends at 4(t + 1).
            x = (np.arange(raw.size) + 1.0) / events.FS_RAW - (step + 1) * SECONDS_PER_STEP
            window = (x >= x_limits[0]) & (x <= x_limits[1])
            traces.draw_raw_signal(axis, x[window], physical[window], name, unit)
            if name == "up":
                found = events.detect_contractions(raw, valid=valid)
                for onset, end in zip(found["onset_raw"], found["end_raw"]):
                    axis.axvspan(x[onset], x[end], color=figures.COLOR_GREEN, alpha=0.12, linewidth=0, zorder=0)

        for offset, channel in enumerate(channels):
            axis = axes[len(traces.RAW_SIGNAL_FIELDS) + offset, column]
            series = step_series(target[row, :, :, channel], first=first, stride=stride)
            past = np.arange(step - history + 1, step + 1)
            known = (past >= 0) & (past < series.size)
            axis.plot((past[known] - step) * SECONDS_PER_STEP, series[past[known]],
                      color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle="--")
            lead = (np.arange(horizon) + 1.0) * SECONDS_PER_STEP
            axis.plot(lead, target[row, position, :, channel], color=figures.COLOR_BLACK,
                      linewidth=figures.LINE_REGULAR)
            for branch, colour in (("base", figures.COLOR_BLUE), ("full", figures.COLOR_VERMILLION)):
                mean = retained.get(f"mu_{branch}")
                if mean is None:
                    continue
                mean = np.asarray(mean[row, position, :, channel], dtype=np.float64)
                axis.plot(lead, mean, color=colour, linewidth=figures.LINE_REGULAR)
                logvar = retained.get(f"logvar_{branch}")
                if predictive_band and logvar is not None:
                    spread = np.sqrt(marginal_variance(
                        np.exp(np.asarray(logvar[row, position, :, channel], dtype=np.float64)),
                        None if ar_coef is None else float(ar_coef[channel]),
                    ))
                    axis.fill_between(lead, mean - spread, mean + spread, color=colour, alpha=0.15, linewidth=0)
            if scored_horizon is not None and int(scored_horizon[channel]) < horizon:
                axis.axvspan((int(scored_horizon[channel]) + 0.5) * SECONDS_PER_STEP, x_limits[1],
                             color=figures.COLOR_LIGHT_GRAY, alpha=0.7, linewidth=0, zorder=0)
            if column == 0:
                axis.set_title(labels_of[offset])
                axis.set_ylabel("coefficient ($z$)")

        subgroup = sample.get(labels.SUBGROUP_COLUMN)
        axes[0, column].set_title(
            f"guid {sample.get('guid')}, subgroup {'n/a' if pd.isna(subgroup) else subgroup}"
        )

    for grid_row in range(len(rows)):
        drawn = [axis for axis in axes[grid_row] if axis.has_data()]
        if drawn:
            low = min(axis.get_ylim()[0] for axis in drawn)
            high = max(axis.get_ylim()[1] for axis in drawn)
            for axis in axes[grid_row]:
                axis.set_ylim(low, high)
        for axis in axes[grid_row]:
            axis.set_xlim(*x_limits)
            axis.axvline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
            figures.style_axes(axis)
            if grid_row < len(rows) - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("time from anchor (s)")

    handles = [
        Line2D([0], [0], color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle="--", label="target, history"),
        Line2D([0], [0], color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="target, forecast"),
        Line2D([0], [0], color=figures.COLOR_BLUE, linewidth=figures.LINE_REGULAR, label="target-only"),
        Line2D([0], [0], color=figures.COLOR_VERMILLION, linewidth=figures.LINE_REGULAR, label="source-conditioned"),
        Patch(color=figures.COLOR_LIGHT_GRAY, label="unscored cells"),
        Patch(color=figures.COLOR_GREEN, alpha=0.3, label="contraction"),
    ]
    height_in = float(figure.get_size_inches()[1])
    # One short key for the band, and only where a band is drawn; the rest of what a reader may ask
    # (the anchor each column is drawn at, the shared scales, the AR(1) widening) is in
    # ``FIGURE_GUIDE.md``.
    bottom = (
        figures.caveat_note(figure, "Band: predictive $\\pm 1\\sigma$.") if predictive_band else 0.0
    )
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                  fontsize=figures.FONT_SMALL, bbox_to_anchor=(0.5, 1.0))
    figure.subplots_adjust(
        left=0.9 / float(figure.get_size_inches()[0]), right=0.99,
        top=1.0 - 0.75 / height_in, bottom=bottom + 0.45 / height_in, hspace=0.55, wspace=0.12,
    )
    figures.mark_laid_out(figure)
    return figure


# =============================================================================
# The analysis
# =============================================================================
def run_forecast_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the forecast against the trivial baselines and resolve it by horizon step.

    Args:
        context: The analysis context, read for the collected tables and the pass's own record.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: every population fact this analysis needs is on
            the per-sample table, which is the table the probe's own counts were checked against.

    Returns:
        The protocol's keys plus the skill table, the error table, the horizon curve's headline
        points and the paths written.
    """
    collection = context.collection
    record = dict(getattr(collection, "record", None) or {})
    geometry = dict(record.get("geometry") or {})
    per_sample = collection.per_sample

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    value_columns = [_BLOCK_COLUMN.format(branch=name) for name in FORECAST_BRANCHES]
    value_columns += [_SQUARED_ERROR_COLUMN.format(branch=name) for name in FORECAST_BRANCHES]
    value_columns += [f"abs_error_{name}" for name in MODEL_BRANCHES]
    value_columns += [f"signed_error_{name}" for name in MODEL_BRANCHES]
    per_guid = per_recording_means(per_sample, value_columns)
    per_guid.to_csv(directory / SCORES_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    skill_rows = build_skill_rows(per_guid, resamples=resamples, seed=seed)
    error_rows = build_error_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(skill_rows).to_csv(directory / SKILL_FILENAME, index=False)

    curves = horizon_curves(record.get("horizon") or {})
    curves.to_csv(directory / HORIZON_FILENAME, index=False)
    profile = anchor_profile(collection.per_anchor)
    profile.to_csv(directory / ANCHOR_FILENAME, index=False)

    written: List[str] = [
        str(figures.render_figure(
            build_baseline_figure(per_guid, skill_rows, unit=NORMALISED_UNIT),
            directory / BASELINE_FIGURE,
        ).name),
        str(figures.render_figure(
            build_anchor_profile_figure(profile, geometry)[0], directory / ANCHOR_FIGURE
        ).name),
        str(figures.render_figure(
            build_horizon_figure(curves, horizon_steps=int(geometry.get("horizon", 0))),
            directory / HORIZON_FIGURE,
        ).name),
    ]
    written += _emit_overlay(context, directory, output_dir=output_dir, seed=seed)

    r2_rows = [row for row in skill_rows if row["is_r2_reference"]]
    return {
        "n_samples": int(per_sample[_BLOCK_COLUMN.format(branch="full")].notna().sum())
        if _BLOCK_COLUMN.format(branch="full") in per_sample.columns else 0,
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        # One unit throughout, stated rather than implied. There is no conversion out of it and
        # the module docstring says why.
        "unit": NORMALISED_UNIT,
        # Fixed and recorded, not fitted: under a Gaussian likelihood the entire NLL-space skill
        # of a point predictor is decided by the sigma it is handed.
        "baseline_logvar": float(BASELINE_LOGVAR),
        "baselines": list(BASELINE_NAMES),
        "skill": skill_rows,
        "error": error_rows,
        "r2": {
            "reference": R2_REFERENCE,
            "references_available": list(R2_REFERENCES),
            **{str(row["branch"]): row["mse_skill"] for row in r2_rows},
        },
        "horizon": {
            "n_steps": int(len(curves)),
            "score_path": "single-draw (training path)",
            "gap_first_step_nats": float(curves["gap_nats"].iloc[0]) if len(curves) else None,
            "gap_last_step_nats": float(curves["gap_nats"].iloc[-1]) if len(curves) else None,
        },
        "grouped_frames": [grouped_frame_entry(ANALYSIS_DIRNAME, SCORES_FILENAME, GROUPED_METRICS)],
        "files": [SCORES_FILENAME, SKILL_FILENAME, HORIZON_FILENAME, ANCHOR_FILENAME] + written,
    }


def _emit_overlay(context: Any, directory: Path, *, output_dir: Any, seed: int) -> List[str]:
    """Draw the forecast overlay when a block was retained, and say nothing when none was.

    Retention is opt-in -- ``eval_config.caps.waveforms`` -- because the tensors this figure needs
    are megabytes per sample. A run that did not ask for them has not failed, so the absent figure
    is silence rather than an empty page.

    Args:
        context: The analysis context: the collection, and the model and loader when present.
        directory: This analysis's output directory.
        output_dir: The results directory, where the kept-axis channel map is read from.
        seed: The run's seed, offset by :data:`OVERLAY_SEED_OFFSET` for the sample draw.

    Returns:
        The files written -- the figure and its sample table -- or nothing.
    """
    collection = context.collection
    retained = dict(getattr(collection, "retained", None) or {})
    if not all(name in retained for name in ("target", "mu_base", "mu_full")):
        return []
    target = retained["target"]
    if len(target) == 0 or target.ndim != 4:
        return []
    record = dict(getattr(collection, "record", None) or {})
    geometry = dict(record.get("geometry") or {})
    width = int(target.shape[3])
    map_path = Path(output_dir) / KEPT_CHANNEL_MAP_FILENAME
    kept_map = pd.read_csv(map_path) if map_path.is_file() else None
    scored_horizon, ar_coef = overlay_density_terms(context, record, width)
    channels = overlay_channels(width, kept_map=kept_map, scored_horizon=scored_horizon)
    samples = overlay_samples(
        retained, collection.per_sample, getattr(collection, "per_anchor", None),
        geometry=geometry, seed=seed + OVERLAY_SEED_OFFSET,
    )
    figure = build_overlay_figure(
        retained, samples, channels,
        geometry=geometry,
        channel_labels=[channel_label(kept_map, channel) for channel in channels],
        scored_horizon=scored_horizon,
        ar_coef=ar_coef,
        raw_scales=traces.raw_signal_scales(getattr(context, "config", None)),
        predictive_band=str(record.get("likelihood") or "") == "gaussian_nll",
    )
    written = figures.render_figure(figure, directory / OVERLAY_FIGURE)
    table = pd.DataFrame(samples)
    table["kept_channels"] = ";".join(str(channel) for channel in channels)
    table["channel_labels"] = " ; ".join(channel_label(kept_map, channel) for channel in channels)
    table["figure_file"] = written.name
    table.to_csv(directory / OVERLAY_SAMPLES_FILENAME, index=False)
    return [str(written.name), OVERLAY_SAMPLES_FILENAME]
