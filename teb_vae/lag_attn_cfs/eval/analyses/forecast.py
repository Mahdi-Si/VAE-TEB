r"""Is the forecast any good, against predictors that know nothing, and where in the horizon.

Every other readout in this pipeline is about the *coupling* -- what the source adds. This one is
about the forecast itself, and it exists because a block score alone cannot answer the question. A
block score of several hundred nats per anchor is not a number anybody can judge: it is a negative
log density summed over $H \cdot C_{\mathrm{keep}} = 1470$ coefficients, so it is large under every
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
and inverting the per-channel statistics would put the $98$ scored channels on scales spanning
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

**Everything here is per recording.** Anchors overlap in $14$ of their $15$ horizon steps and one
recording contributes tens of segments, so every statistic is averaged within a recording first
and the bootstrap resamples recordings, never anchors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
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
BASELINE_FIGURE = "baseline_comparison.pdf"
ANCHOR_FIGURE = "anchor_profile.pdf"
OVERLAY_FIGURE = "forecast_overlay.pdf"
HORIZON_FIGURE = "horizon_skill.pdf"

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

#: How many target channels the forecast overlay draws, and where they are taken from. Evenly
#: spaced across the kept channel axis and fixed, so two runs of one checkpoint draw the same
#: channels and a figure can be compared across arms rather than only read.
OVERLAY_CHANNELS = 3

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
        if squares in horizon and counts in horizon:
            # ``count`` already carries the channel factor, so this is the per-coefficient mean
            # square rather than a per-anchor one, matching ``sq_error_*`` on the sample table.
            frame[f"rmse_{branch}_normalised"] = np.sqrt(
                np.clip(_mean(squares, counts), 0.0, None)
            )
            frame["rmse_unit"] = NORMALISED_UNIT
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
        unit: The unit label the error panel is in.

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
        title="Block score per recording (lower is better)",
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
    axis.set_title(f"Squared-error skill, bootstrap CI over recordings (errors in {unit})")
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
        ["target-only (base)", "source-conditioned (full)"],
        title="Block score against time in segment",
        xlabel="anchor (decimated steps from the start of the trimmed segment)",
        ylabel="nats per anchor",
    )
    figures.multi_line_panel(
        axes[1, 0],
        anchors,
        _finite_column(profile, "pred_gap")[None, :],
        ["pred_gap"],
        title="pred_gap against time in segment",
        xlabel="anchor (decimated steps from the start of the trimmed segment)",
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
    figure, axes = figures.new_figure(3)
    lead = _finite_column(curves, "lead_seconds")
    figures.multi_line_panel(
        axes[0, 0], lead,
        np.vstack([_finite_column(curves, "d_base_nats"), _finite_column(curves, "d_full_nats")]),
        ["target-only (base)", "source-conditioned (full)"],
        title="Forecast score by lead time (single-draw path)",
        xlabel="lead time (s)", ylabel="nats per horizon step",
    )
    figures.multi_line_panel(
        axes[1, 0], lead, _finite_column(curves, "gap_nats")[None, :], ["pred_gap"],
        title="Source contribution by lead time",
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
        title="Forecast error by lead time",
        xlabel="lead time (s)", ylabel=f"RMSE per coefficient ({unit})",
    )
    for axis in (axes[0, 0], axes[1, 0], axes[2, 0]):
        axis.set_xlim(0.0, float(horizon_steps) * SECONDS_PER_STEP)
    return figure


def overlay_channels(width: int, count: int = OVERLAY_CHANNELS) -> List[int]:
    """Choose which kept target channels the overlay draws, evenly spaced and deterministic.

    Args:
        width: $C_{\\mathrm{keep}}$, the retained block's channel axis.
        count: How many to draw.

    Returns:
        Ascending channel positions on the kept axis, without duplicates. Empty when the block
        has no channels at all.
    """
    if width <= 0 or count <= 0:
        return []
    if width <= count:
        return list(range(width))
    positions = np.linspace(0, width - 1, num=count)
    return sorted({int(round(float(value))) for value in positions})


def build_overlay_figure(
    retained: Dict[str, np.ndarray],
    *,
    row: int,
    anchor: int,
    channels: Sequence[int],
) -> Any:
    r"""Draw one anchor's truth and both forecasts against lead time, for a few channels.

    Not the raw-grid overlay the raw cells draw, and the difference is the target domain rather
    than a simplification: what this model forecasts is an $H \times C_{\mathrm{keep}}$ block of
    wavelet coefficients, so there is no single waveform to overlay. Three channels are drawn as
    separate panels rather than the block as one heatmap, because the comparison being made is
    *between the three curves within a channel*, and three heatmaps on three independent colour
    scales cannot support it.

    Args:
        retained: The collection's retained arrays, each $(N, A_{\max}, H, C_{\mathrm{keep}})$.
        row: Which retained sample to draw.
        anchor: Which **position** in the decoded anchor set, not which decimated step.
        channels: Positions on the kept channel axis.

    Returns:
        The figure.
    """
    panels = list(channels) or [0]
    figure, axes = figures.new_figure(len(panels))
    for panel, channel in enumerate(panels):
        axis = axes[panel, 0]
        curves: List[np.ndarray] = []
        labels: List[str] = []
        for name, label in (
            ("target", "truth"),
            ("mu_base", "target-only (base)"),
            ("mu_full", "source-conditioned (full)"),
        ):
            block = retained.get(name)
            if block is None or row >= len(block):
                continue
            values = np.asarray(block[row, anchor], dtype=np.float64)
            if values.ndim != 2 or channel >= values.shape[1]:
                continue
            curves.append(values[:, channel])
            labels.append(label)

        if curves:
            # Horizon step tau covers decimated step t + 1 + tau, so the first point is one
            # decimated step ahead of the anchor rather than at it.
            lead = (np.arange(curves[0].size, dtype=np.float64) + 1.0) * SECONDS_PER_STEP
            figures.multi_line_panel(
                axis, lead, np.vstack(curves), labels,
                title=f"Kept channel {channel}, retained row {row}, anchor position {anchor}",
                xlabel="lead time (s)", ylabel=f"coefficient ({NORMALISED_UNIT})",
            )
        else:
            figures.multi_line_panel(
                axis, np.zeros(0), np.zeros((0, 0)), [],
                title=f"Kept channel {channel}",
                xlabel="lead time (s)", ylabel=f"coefficient ({NORMALISED_UNIT})",
            )
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
        str(figures.render_to_pdf(
            build_baseline_figure(per_guid, skill_rows, unit=NORMALISED_UNIT),
            directory / BASELINE_FIGURE,
        ).name),
        str(figures.render_to_pdf(
            build_anchor_profile_figure(profile, geometry)[0], directory / ANCHOR_FIGURE
        ).name),
        str(figures.render_to_pdf(
            build_horizon_figure(curves, horizon_steps=int(geometry.get("horizon", 0))),
            directory / HORIZON_FIGURE,
        ).name),
    ]
    overlay = _emit_overlay(collection, directory)
    if overlay is not None:
        written.append(overlay)

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


def _emit_overlay(collection: Any, directory: Path) -> Optional[str]:
    """Draw the forecast overlay when a block was retained, and say nothing when none was.

    Retention is opt-in -- ``eval_config.caps.waveforms`` -- because the tensors this figure needs
    are megabytes per sample. A run that did not ask for them has not failed, so the absent figure
    is silence rather than an empty page.

    Args:
        collection: What the pass produced.
        directory: This analysis's output directory.

    Returns:
        The filename written, or ``None``.
    """
    retained = dict(getattr(collection, "retained", None) or {})
    if not all(name in retained for name in ("target", "mu_base", "mu_full")):
        return None
    target = retained["target"]
    if len(target) == 0 or target.ndim != 4:
        return None
    # The first retained row, and the middle **position** of the decoded anchor set. The row is
    # arbitrary because the retention draw is already a seeded stratified sample over the whole
    # split, so position 0 is a uniform draw rather than a prefix. The anchor is a position rather
    # than a decimated step: this model gathers its anchors, so a step index would be out of range
    # on the retained axis and silently draw a different anchor than the one named.
    anchor = int(target.shape[1] // 2)
    figure = build_overlay_figure(
        retained,
        row=0,
        anchor=anchor,
        channels=overlay_channels(int(target.shape[3])),
    )
    return str(figures.render_to_pdf(figure, directory / OVERLAY_FIGURE).name)
