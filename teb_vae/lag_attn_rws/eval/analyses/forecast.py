r"""Is the forecast any good, in units a clinician reads, and where in the horizon.

Every other readout in this pipeline is about the *coupling* -- what the source adds. This one is
about the forecast itself, and it exists because a block score alone cannot answer the question.
$D_{\mathrm{full}} = 674$ nats per anchor is not a number anybody can judge: it is a negative log
density summed over $H \cdot R = 480$ raw samples, so it is large under every predictor and its
scale is set by the block size rather than by the model. Three things make it readable.

**Baselines.** The same loss function, the same mask, the same anchors, applied to three
predictors that know nothing: persistence, climatology, and the segment's own mean
(:func:`~teb_vae.lag_attn_rws.eval.metrics.baseline_forecasts`). Two skill columns come out of
them and they answer different questions:

* $\mathrm{skill} = 1 - \mathrm{MSE}_{\mathrm{model}} / \mathrm{MSE}_{\mathrm{baseline}}$, where
  the observation variance cancels. A forecast equal to the truth scores $1$; one equal to the
  baseline scores $0$.
* $\mathrm{advantage} = D_{\mathrm{baseline}} - D_{\mathrm{model}}$ in nats per anchor. A
  *difference*, not $1 - $ a ratio, because a log score has no natural zero: the ratio of two
  negative log densities is not bounded above by $1$ and changes sign with the baseline's.

Both are reported because a learned-variance model otherwise beats a fixed-variance baseline
partly on variance modelling alone, and no single number separates the two effects. The
baselines' $\sigma$ is fixed, stated and recorded for the same reason.

**bpm.** A root-mean-square error of $0.1$ in loader units is about $1$ bpm. Getting there is a
*scale* conversion and not the affine one that inverts a level: put an RMSE through the level
map and it comes back as $141$ bpm, which is physiologically plausible and therefore never
questioned. :func:`~teb_vae.lag_attn_rws.eval.metrics.sigma_to_bpm` is that conversion, and
without the loader's statistics the numbers stay labelled ``normalised``.

**The horizon axis.** $D(\tau)$ answers whether the forecast -- and the source's contribution to
it -- holds up at two minutes or only at four seconds, and neither predecessor pipeline computes
it. Two properties are load-bearing. It is built on the **single-draw** path, because the Monte
Carlo marginalisation does not commute with the sum over $\tau$: by Jensen,
$\sum_\tau -\log \frac{1}{K}\sum_r e^{-D_r(\tau)} \neq -\log \frac{1}{K}\sum_r e^{-\sum_\tau
D_r(\tau)}$, so a marginalised curve would not sum back to the marginalised headline. And its
denominator is the per-$\tau$ masked anchor count rather than the per-anchor contributing
indicator, which is an ``amax`` over $\tau$ and would count masked late-horizon steps as scored
zeros -- flattering exactly the horizons that fall in gaps.

**Everything here is per recording.** Anchors overlap in $29$ of their $30$ horizon steps and one
recording contributes tens of segments, so every statistic is averaged within a recording first
and the bootstrap resamples recordings, never anchors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_rws.eval.frames import finite_column as _finite_column
from teb_vae.lag_attn_rws.eval.frames import grouped_frame_entry
from teb_vae.lag_attn_rws.eval.frames import per_recording_means
from teb_vae.lag_attn_rws.eval.metrics import (
    BASELINE_LOGVAR,
    BASELINE_NAMES,
    FORECAST_BRANCHES,
    NORMALISED_UNIT,
    sigma_to_bpm,
    to_bpm,
)
from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP

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
def skill_against(model: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    r"""Per-recording squared-error skill, $1 - \mathrm{MSE}_{\rm model}/\mathrm{MSE}_{\rm ref}$.

    Computed per recording and then averaged, rather than as a ratio of the two averages. The two
    differ, and this is the form the acceptance criteria are stated in: a forecast equal to the
    truth scores exactly $1$ on **every** recording and a forecast equal to the baseline exactly
    $0$ on every recording, so the mean carries those answers unchanged -- and a bootstrap over
    recordings then has a per-recording quantity to resample.

    Args:
        model: Per-recording mean squared error of the model branch.
        baseline: Per-recording mean squared error of the baseline.

    Returns:
        The per-recording skill, ``NaN`` wherever the baseline's error is zero or either value is
        missing. A zero-error baseline is a degenerate recording -- a constant signal the baseline
        reproduces exactly -- and dividing by it would report an infinite skill as evidence.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(baseline > 0.0, 1.0 - model / baseline, np.nan)


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
    per_guid: pd.DataFrame, normalization: Optional[Dict[str, Any]], *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    r"""Report each model branch's point-forecast error, in $z$ units and in bpm.

    The RMSE roots **once**, after the per-recording mean of the unrooted squares. Rooting per
    segment and averaging the roots is biased low by Jensen -- in the direction that flatters the
    model -- which is why the collection pass accumulates the squares rather than the roots.

    All three quantities are differences of levels, so all three convert by the *scale* alone.

    Args:
        per_guid: Per-recording means.
        normalization: The loader's FHR statistics, or ``None``.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per branch: MAE, RMSE and bias with their intervals, in both units, and the unit
        label that says which the bpm columns really are.
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
        converted, unit = sigma_to_bpm(
            [rmse, absolute_ci["point"], signed_ci["point"]], normalization
        )
        rows.append(
            {
                "branch": branch,
                "n_recordings": int(squares_ci["n"]),
                "unit": unit,
                "rmse_normalised": rmse,
                # The interval is on the mean square, so its bounds are rooted rather than the
                # interval being rebuilt: a monotone transform of a percentile interval is the
                # percentile interval of the transform.
                "rmse_lo_normalised": float(np.sqrt(max(squares_ci["lo"], 0.0))),
                "rmse_hi_normalised": float(np.sqrt(max(squares_ci["hi"], 0.0))),
                "mae_normalised": absolute_ci["point"],
                "bias_normalised": signed_ci["point"],
                "rmse": float(converted[0]),
                "mae": float(converted[1]),
                # Positive means the forecast runs above the truth.
                "bias": float(converted[2]),
            }
        )
    return rows


# =============================================================================
# The horizon axis
# =============================================================================
def horizon_curves(
    horizon: Dict[str, Any],
    *,
    normalization: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    r"""Turn the streamed per-$\tau$ accumulators into the horizon-resolved curves.

    $$D_{\mathrm{branch}}(\tau) = \frac{\sum_{b,t} D_{b,t,\tau}}{\sum_{b,t} m_{b,t,\tau}},
    \qquad \mathrm{gap}(\tau) = D_{\mathrm{base}}(\tau) - D_{\mathrm{full}}(\tau).$$

    Args:
        horizon: The collection record's ``horizon`` block, one list per branch and statistic.
        normalization: The loader's FHR statistics, for the bpm columns.

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
            rmse = np.sqrt(np.clip(_mean(squares, counts), 0.0, None))
            converted, unit = sigma_to_bpm(rmse, normalization)
            frame[f"rmse_{branch}_normalised"] = rmse
            frame[f"rmse_{branch}"] = converted
            frame["rmse_unit"] = unit
    return frame


def anchor_profile(per_anchor: pd.DataFrame) -> pd.DataFrame:
    """Average each per-anchor score across every segment that scored that anchor.

    Args:
        per_anchor: The per-anchor table.

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
            fmt="none", ecolor=figures.COLOR_BLACK, elinewidth=1.0, capsize=3.0,
        )
        axis.axvline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=1.2)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=7)
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

    Two spans are shaded and neither is a finding: the warm-up prefix $[0, w)$ carries no loss
    term at all, and the tail $[T - H, T)$ holds the anchors whose forecast window runs past the
    end of the segment, so no anchor there is ever scored. An unshaded profile reads as a model
    that fails at both ends of every recording.

    Args:
        profile: The per-anchor profile.
        geometry: The collection record's geometry block.

    Returns:
        ``(figure, spans)``, where ``spans`` names the two shaded intervals in anchor steps --
        returned rather than only drawn, so the bounds are checkable without reading the artist.
    """
    warmup = int(geometry.get("warmup", 0))
    t_valid = int(geometry.get("t_valid", 0))
    total = int(geometry.get("t", t_valid))
    spans = {"warmup": (0.0, float(warmup)), "untrained_tail": (float(t_valid), float(total))}

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
    source still help two minutes out?" -- is asked in seconds, and a reader who has to multiply
    by four is a reader who will eventually forget to.

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
        np.vstack([_finite_column(curves, f"rmse_{branch}") for branch in MODEL_BRANCHES]),
        list(MODEL_BRANCHES),
        title="Forecast error by lead time",
        xlabel="lead time (s)", ylabel=f"RMSE ({unit})",
    )
    for axis in (axes[0, 0], axes[1, 0], axes[2, 0]):
        axis.set_xlim(0.0, float(horizon_steps) * SECONDS_PER_STEP)
    return figure


def build_overlay_figure(
    retained: Dict[str, np.ndarray],
    *,
    row: int,
    anchor: int,
    geometry: Dict[str, Any],
    normalization: Optional[Dict[str, Any]] = None,
) -> Any:
    """Draw one anchor's truth, base forecast and full forecast on the raw grid.

    Args:
        retained: The collection's retained arrays.
        row: Which retained sample to draw.
        anchor: Which anchor of that sample.
        geometry: The collection record's geometry block, for the raw sampling rate.
        normalization: The loader's FHR statistics, for the bpm axis.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1)
    axis = axes[0, 0]
    decimation = int(geometry.get("decimation", 1)) or 1
    curves: List[np.ndarray] = []
    labels: List[str] = []
    unit = NORMALISED_UNIT
    for name, label in (
        ("target", "truth"),
        ("mu_base", "target-only (base)"),
        ("mu_full", "source-conditioned (full)"),
    ):
        block = retained.get(name)
        if block is None or row >= len(block):
            continue
        converted, unit = to_bpm(np.asarray(block[row, anchor]).reshape(-1), normalization)
        curves.append(converted)
        labels.append(label)

    if curves:
        # Raw sample index to seconds: the block starts one raw sample past the anchor's causal
        # endpoint, and there are ``decimation`` raw samples per 4-second decimated step.
        seconds = (np.arange(curves[0].size, dtype=np.float64) + 1.0) * (
            SECONDS_PER_STEP / float(decimation)
        )
        figures.multi_line_panel(
            axis, seconds, np.vstack(curves), labels,
            title=f"Forecast block, retained sample row {row}, anchor {anchor}",
            xlabel="lead time (s)", ylabel=f"FHR ({unit})",
        )
    else:
        figures.multi_line_panel(
            axis, np.zeros(0), np.zeros((0, 0)), [],
            title="Forecast block", xlabel="lead time (s)", ylabel=f"FHR ({unit})",
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
    normalization = dict(record.get("normalization") or {})
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
    error_rows = build_error_rows(per_guid, normalization, resamples=resamples, seed=seed)
    pd.DataFrame(skill_rows).to_csv(directory / SKILL_FILENAME, index=False)

    curves = horizon_curves(record.get("horizon") or {}, normalization=normalization)
    curves.to_csv(directory / HORIZON_FILENAME, index=False)
    profile = anchor_profile(collection.per_anchor)
    profile.to_csv(directory / ANCHOR_FILENAME, index=False)

    unit = str(error_rows[0]["unit"]) if error_rows else NORMALISED_UNIT
    written: List[str] = [
        str(figures.render_to_pdf(
            build_baseline_figure(per_guid, skill_rows, unit=unit), directory / BASELINE_FIGURE
        ).name),
        str(figures.render_to_pdf(
            build_anchor_profile_figure(profile, geometry)[0], directory / ANCHOR_FIGURE
        ).name),
        str(figures.render_to_pdf(
            build_horizon_figure(curves, horizon_steps=int(geometry.get("horizon", 0))),
            directory / HORIZON_FIGURE,
        ).name),
    ]
    overlay = _emit_overlay(collection, directory, geometry, normalization)
    if overlay is not None:
        written.append(overlay)

    r2_rows = [row for row in skill_rows if row["is_r2_reference"]]
    return {
        "n_samples": int(per_sample[_BLOCK_COLUMN.format(branch="full")].notna().sum())
        if _BLOCK_COLUMN.format(branch="full") in per_sample.columns else 0,
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "unit": unit,
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


def _emit_overlay(
    collection: Any,
    directory: Path,
    geometry: Dict[str, Any],
    normalization: Dict[str, Any],
) -> Optional[str]:
    """Draw the forecast overlay when a waveform was retained, and say nothing when none was.

    Retention is opt-in -- ``eval_config.caps.waveforms`` -- because the three tensors this figure
    needs are two megabytes per sample. A run that did not ask for them has not failed, so the
    absent figure is silence rather than an empty page.

    Args:
        collection: What the pass produced.
        directory: This analysis's output directory.
        geometry: The geometry block.
        normalization: The loader's FHR statistics.

    Returns:
        The filename written, or ``None``.
    """
    retained = dict(getattr(collection, "retained", None) or {})
    if not all(name in retained for name in ("target", "mu_base", "mu_full")):
        return None
    if len(retained["target"]) == 0:
        return None
    # The first retained row, and the middle of the trained anchor range. The row is arbitrary
    # because the retention draw is already a seeded stratified sample over the whole split, so
    # position 0 is a uniform draw rather than a prefix; the anchor is not, because a warm-up
    # anchor carries no loss term and a tail anchor is never scored, so either would draw a
    # forecast nothing was fitted to.
    warmup = int(geometry.get("warmup", 0))
    t_valid = int(geometry.get("t_valid", retained["target"].shape[1]))
    anchor = int((warmup + t_valid) // 2)
    figure = build_overlay_figure(
        retained, row=0, anchor=anchor, geometry=geometry, normalization=normalization
    )
    return str(figures.render_to_pdf(figure, directory / OVERLAY_FIGURE).name)
