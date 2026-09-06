r"""What the causal front end's warm-up cost this run, per recording and with intervals.

Three readings, and they are this cell's own because no other cell in the encoder-by-target grid
has a channel axis whose members become honest at different times:

* **The forecast gap split by warm-up tertile.** ``pred_gap`` is a sum over the $C_{\mathrm{keep}}$ surviving
  target channels, and those channels differ enormously in how long their one-sided filter needs
  before its output is a function of the recording rather than of assumed pre-recording history.
  Cutting the gap by that speed says *where* the model's advantage lives: on the fast channels the
  front end warms in seconds, or on the slow ones it warms in minutes. The three tertiles
  **recompose to** ``pred_gap`` over the same denominator, and that recomposition is checked here
  rather than described -- it is the only property that makes them a decomposition rather than
  three unrelated numbers.

* **The source-lag warmth fractions.** How much of the lag attention mass lands on lags where each
  stored source block is warm. **A small value here is the expected finding, not a fault**, and
  the emitted record says so: ``lag_attn_cfs/DESIGN.md`` section 8 records that the source blocks'
  warm-ups are long against the lag window, so most of the searched lag range reads source steps
  that are still settling. The number exists to be *measured* across arms, not to be passed.

* **The two geometry guards.** ``anchors_per_sample`` must be $T_{\mathrm{valid}} - F$ and
  ``target_warm_frac`` exactly $1.0$. Both are exact structural numbers rather than statistics, and
  both decide what population every other number in the run was computed over: a decoded anchor
  count off by one means the forward ran at the training tiling, and a warm fraction below one
  means some scored coefficient lay inside its channel's warm-up and was scored as signal.

**The guards' expectations are computed from the run's own geometry, never from a literal.** The
collection record carries $T$, $T_{\mathrm{valid}}$ and the anchor floor the pass decoded at, so
$A_{\max} = T_{\mathrm{valid}} - F$ is arithmetic on numbers the checkpoint itself produced -- and
an arm that legitimately moves the horizon or the floor moves the expectation with it instead of
failing a guard written against the shipped geometry.

**Two figures come from the modules that already draw them**, and neither is reimplemented: the
warm-up staircase from :mod:`teb_vae.lag_attn_cfs.warmup_budget` and the budget tradeoff curve
from the same module's own curve builder. The tradeoff curve is a constant of the *shard* rather
than of the run, and it is drawn into the run directory anyway because a reader opening one
finished run has to be able to see why the budget that decided its channel axis was chosen without
going back to the dataset for it. Both are drawn from the budget re-resolved against the
configured shards; a shard that cannot be read is a **recorded skip**, exactly as the channel map
treats one, because a run whose readouts all succeeded must not be marked failed for a figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval._reuse import figures, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import (
    RECOMPOSITION_SCALE_COLUMN,
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    recomposition_check,
    scored_sample_count,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "warmup"

#: What it writes. The per-recording frame's name is the one ``cross_subgroup`` reads the three
#: tertile columns out of, so it is a contract rather than a filename.
PER_RECORDING_FILENAME = "warmup_per_recording.csv"
SUMMARY_FILENAME = "warmup_summary.csv"
TERTILE_FIGURE = "warmup_tertiles"

#: The three tertile columns, in warm-up order, and the ``pred_gap`` they must sum to. Written out
#: rather than derived from a prefix match: the recomposition below is only a check if the set of
#: parts is declared independently of what happens to be on the table.
TERTILE_COLUMNS: Tuple[str, ...] = (
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
)
TOTAL_COLUMN = "pred_gap"

#: Every metric this analysis reports per recording: the column, its unit, and what it means. The
#: two guards are on the list as well as in their own block, so the per-recording CSV carries them
#: and a reader can see *which* recording disagreed rather than only that one did.
METRICS: Tuple[Tuple[str, str, str], ...] = (
    (
        TOTAL_COLUMN,
        "nats per anchor",
        "the forecast gap the three tertiles below decompose, on the training path",
    ),
    (
        "pred_gap_warm_lo",
        "nats per anchor",
        "the part of the gap on the fastest third of the kept target channels -- the ones whose "
        "one-sided filter settles soonest",
    ),
    (
        "pred_gap_warm_mid",
        "nats per anchor",
        "the part of the gap on the middle third by warm-up length",
    ),
    (
        "pred_gap_warm_hi",
        "nats per anchor",
        "the part of the gap on the slowest third -- the channels the budget came closest to "
        "dropping",
    ),
    (
        "source_lag_warmth_frac_st",
        "fraction of attention mass",
        "attention landing on lags where the source scattering block is warm; a small value is "
        "the expected finding here, not a fault",
    ),
    (
        "source_lag_warmth_frac_ph",
        "fraction of attention mass",
        "attention landing on lags where the source phase-harmonic block is warm; its warm-up is "
        "the longer of the two, so this is expected to be the smaller of the pair",
    ),
    (
        "anchors_per_sample",
        "anchors",
        "the anchor count the forward actually decoded, which must be T_valid - F at the dense "
        "geometry this pipeline evaluates at",
    ),
    (
        "target_warm_frac",
        "fraction of scored coefficients",
        "the share of scored target coefficients past their own channel's warm-up, which must be "
        "exactly 1.0",
    ),
)

#: The columns the per-recording chain reduces, in emission order. The block score is on the end
#: and is not a reported metric: it is the magnitude the recomposition tolerance is scaled by, and
#: it travels on the per-recording frame so a reader can see what that tolerance actually was.
VALUE_COLUMNS: Tuple[str, ...] = tuple(column for column, _, _ in METRICS) + (
    RECOMPOSITION_SCALE_COLUMN,
)

#: The columns worth a by-class and by-subgroup variant. The three tertiles and the two warmth
#: fractions: a cohort that differs in *which* channels its gap came from, or in how much of the
#: lag window its source was warm over, is a finding. The two guards are deliberately absent --
#: they are structural constants of the run, identical on every recording, and a grouped figure of
#: a constant is a figure of nothing.
GROUPED_METRICS: Tuple[str, ...] = TERTILE_COLUMNS + (
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
)

#: The statement that travels with the warmth fractions, so a reader does not treat a small value
#: as a defect. It cites the design record rather than restating its argument.
WARMTH_EXPECTATION = (
    "a small warmth fraction is the expected finding on this cell rather than a fault: the source "
    "blocks' causal warm-ups are long against the 91-step lag window, so most of the searched lag "
    "range reads source steps that are still settling. See lag_attn_cfs/DESIGN.md section 8. The "
    "number is here to be compared across arms, not to be passed."
)

#: The value ``target_warm_frac`` must take, and the only one it can take on a checkpoint built by
#: a constructor that refuses the pairing which would produce anything else.
EXPECTED_TARGET_WARM_FRAC = 1.0

#: How far either guard may drift from its exact expectation. Both are means of values identical on
#: every sample, so the exact answer is reachable; this guards the float accumulation across
#: recordings rather than admitting a real difference.
GEOMETRY_TOLERANCE = 1e-6


def expected_geometry(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r"""What the run's own geometry says the two guards must read.

    $A_{\max} = T_{\mathrm{valid}} - F$ is recomputed here from the two numbers it is made of
    rather than read straight out of ``anchors_per_sample``, so the expectation and the collection
    pass's own record of it are two derivations that can disagree -- and a disagreement between
    them is reported instead of one number checking itself.

    Args:
        record: The collection record. An offline re-run against a directory whose record carries
            no geometry block passes ``None``.

    Returns:
        The two expectations and the geometry they were derived from, each ``None`` where the
        record does not say. ``None`` makes that half of the guard **unevaluated**, which is not
        the same as satisfied.
    """
    geometry = dict((record or {}).get("geometry") or {})
    t_valid = geometry.get("t_valid")
    floor = geometry.get("anchor_floor")
    derived = (
        None if t_valid is None or floor is None else int(t_valid) - int(floor)
    )
    recorded = geometry.get("anchors_per_sample")
    return {
        "t_valid": None if t_valid is None else int(t_valid),
        "anchor_floor": None if floor is None else int(floor),
        "anchor_stride": geometry.get("anchor_stride"),
        "expected_anchors_per_sample": derived,
        # The collection pass's own copy of the same number. Two derivations rather than one, for
        # the reason above.
        "recorded_anchors_per_sample": None if recorded is None else int(recorded),
        "geometry_self_consistent": (
            None if derived is None or recorded is None else int(recorded) == derived
        ),
        "expected_target_warm_frac": EXPECTED_TARGET_WARM_FRAC,
    }


def recomposition(per_guid: pd.DataFrame) -> Dict[str, Any]:
    """Whether the three tertiles sum back to ``pred_gap``, on the worst recording.

    Args:
        per_guid: Per-recording means carrying the three tertiles, their total and the block score
            the tolerance is scaled by.

    Returns:
        What :func:`~teb_vae.lag_attn_cfs.eval.frames.recomposition_check` reports.
    """
    return recomposition_check(
        per_guid, TERTILE_COLUMNS, TOTAL_COLUMN,
        identity="pred_gap_warm_lo + _mid + _hi == pred_gap",
    )


def build_rows(per_guid: pd.DataFrame, *, resamples: int, seed: int) -> List[Dict[str, Any]]:
    """Summarise every metric over the recordings, with a bootstrap interval on each.

    Args:
        per_guid: Per-recording means.
        resamples: Bootstrap resamples, from ``eval_config.bootstrap_resamples``.
        seed: Bootstrap seed, from ``eval_config.seed``, so the interval is reproducible from the
            summary alone.

    Returns:
        One row per metric, carrying the mean and its interval, the quartiles and the honest
        denominator behind them.
    """
    rows: List[Dict[str, Any]] = []
    for column, unit, meaning in METRICS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        rows.append(
            {
                "metric": column,
                "unit": unit,
                "meaning": meaning,
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "ci_method": interval["method"],
                "bootstrap_resamples": int(interval["resamples"]),
            }
        )
    return rows


def headline_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten the five comparable readouts into the block the headline registry digs into.

    A flat block rather than a path into :data:`METRICS` by position: an arm table reads these by
    name, and a path whose last step is a list index resolves to the wrong row the day a metric is
    added above it, with nothing in the artifact saying so.

    Args:
        rows: What :func:`build_rows` produced.

    Returns:
        ``{name: mean}`` for the three tertiles and the two warmth fractions. The two geometry
        guards are deliberately not here -- they have their own keyed block, where they travel with
        the expectation they are judged against.
    """
    means = {str(row["metric"]): row.get("mean") for row in rows}
    headline: Dict[str, Any] = {}
    for column in TERTILE_COLUMNS:
        headline[f"{column}_nats"] = means.get(column)
    for column in ("source_lag_warmth_frac_st", "source_lag_warmth_frac_ph"):
        headline[column] = means.get(column)
    return headline


def guard_record(per_guid: pd.DataFrame, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Report both geometry guards against the expectations the run's own geometry produced.

    The extremes travel beside the mean deliberately. Both quantities are identical on every
    sample of a healthy run, so a mean that sits on its expectation while the minimum does not is
    the signature of a subset of the pass having decoded a different anchor set -- which a mean
    alone would average away.

    Args:
        per_guid: Per-recording means carrying the two guard columns.
        expected: What :func:`expected_geometry` derived.

    Returns:
        Both measurements with their extremes, the expectations, and a verdict-shaped ``ok`` per
        guard that is ``None`` where the expectation is unknown rather than ``False``.
    """
    record: Dict[str, Any] = dict(expected)
    anchors = finite_column(per_guid, "anchors_per_sample")
    warm = finite_column(per_guid, "target_warm_frac")
    for name, values in (("anchors_per_sample", anchors), ("target_warm_frac", warm)):
        finite = values[np.isfinite(values)]
        record[name] = float(finite.mean()) if finite.size else float("nan")
        record[f"{name}_min"] = float(finite.min()) if finite.size else float("nan")
        record[f"{name}_max"] = float(finite.max()) if finite.size else float("nan")
        record[f"{name}_n_recordings"] = int(finite.size)

    target = expected.get("expected_anchors_per_sample")
    record["anchors_per_sample_ok"] = (
        None
        if target is None or not np.isfinite(record["anchors_per_sample"])
        else bool(abs(record["anchors_per_sample"] - float(target)) <= GEOMETRY_TOLERANCE)
    )
    record["target_warm_frac_ok"] = (
        None
        if not np.isfinite(record["target_warm_frac"])
        else bool(
            abs(record["target_warm_frac"] - EXPECTED_TARGET_WARM_FRAC) <= GEOMETRY_TOLERANCE
        )
    )
    record["tolerance"] = GEOMETRY_TOLERANCE
    return record


def write_budget_figures(config: Dict[str, Any], directory: Any) -> Dict[str, Any]:
    """Draw the warm-up staircase and the budget tradeoff curve, or record why neither was drawn.

    Both come from :mod:`teb_vae.lag_attn_cfs.warmup_budget`, which owns the panel the training
    diagnostics already draw, and both are reached through the resolver the training driver itself
    calls -- so the figure in a run directory describes the budget the checkpoint was built under
    rather than a second resolution of it. That module is imported **inside** this function
    because importing it costs a filter-bank module at module scope, and an evaluation that skipped
    this analysis must not pay for it.

    Args:
        config: The merged run configuration, carrying the budget, the geometry and the shards.
        directory: This analysis's own subdirectory.

    Returns:
        The resolved budget's summary and the files written, or ``skipped`` with the reason. A
        shard that cannot be read is a skip rather than a raise: a run whose every readout
        succeeded must not be reported as failed because a figure's input was unavailable.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
    from teb_vae.lag_attn_cfs.warmup_budget import (
        budget_tradeoff,
        write_tradeoff_figure,
        write_warmup_budget_figure,
    )

    vae_config = dict((config.get("model_config") or {}).get("VAE_model") or {})
    try:
        budget = resolve_warmup_budget(config)
    except Exception as exc:  # noqa: BLE001 - recorded and reported; see the docstring
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
    if budget is None:
        return {
            "skipped": True,
            "reason": (
                "the configuration sets no causal_warmup_budget_steps, so there is no warm-up "
                "guard to draw"
            ),
        }

    horizon = int(vae_config["horizon"])
    written = [
        write_warmup_budget_figure(budget, Path(directory), horizon=horizon).name,
        write_tradeoff_figure(
            budget_tradeoff(
                budget.target.declared_warmup_steps,
                sequence_length=int(vae_config["sequence_length"]),
                horizon=horizon,
                anchor_stride=int(vae_config["anchor_stride"]),
            ),
            Path(directory),
            shipped_budget=budget.budget_steps,
        ).name,
    ]
    return {
        "skipped": False,
        "budget_steps": int(budget.budget_steps),
        "quantile": None if budget.quantile is None else float(budget.quantile),
        "trim_minutes": budget.trim_minutes,
        "summary": budget.summary(),
        "target_kept_width": int(budget.target.kept_width),
        "target_declared_width": int(budget.target.declared_width),
        "target_dropped_index": [int(index) for index in budget.target.dropped_index],
        "target_dropped_warmup_steps": [
            int(budget.target.declared_warmup_steps[index])
            for index in budget.target.dropped_index
        ],
        "source_kept_width": int(budget.source.kept_width),
        "source_declared_width": int(budget.source.declared_width),
        # The survivors' own maximum, which is what the anchor floor must clear -- not the
        # configured threshold, which sits above it wherever the staircase has a gap.
        "realised_max_warmup_steps": int(budget.target.max_warmup),
        "files": [str(name) for name in written],
    }


def build_tertile_figure(
    per_guid: pd.DataFrame, rows: Sequence[Dict[str, Any]]
) -> Any:
    """Draw the tertile decomposition and the two warmth fractions.

    Two panels, and the split is the content. The three tertiles share an axis because they are
    three parts of one number in one unit and are meant to be compared against each other; the two
    warmth fractions get their own axis because they are a fraction of attention mass and putting
    them on a nats axis would flatten whichever is smaller into a line at zero.

    Args:
        per_guid: Per-recording means.
        rows: The summary rows, read for the total's denominator.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    total = next((row for row in rows if row.get("metric") == TOTAL_COLUMN), {})
    figures.violin_panel(
        axes[0, 0],
        {name: finite_column(per_guid, name) for name in TERTILE_COLUMNS},
        title=(
            f"pred_gap per recording by warm-up tertile, n = {int(total.get('n') or 0)} "
            f"(the three sum to pred_gap)"
        ),
        ylabel="nats per anchor",
        reference=0.0,
        reference_label="no improvement",
    )
    figures.violin_panel(
        axes[1, 0],
        {
            name: finite_column(per_guid, name)
            for name in ("source_lag_warmth_frac_st", "source_lag_warmth_frac_ph")
        },
        title="attention mass on lags where the source block is warm (a small value is expected)",
        ylabel="fraction of attention mass",
    )
    return figure


def run_warmup_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the tertile decomposition, the warmth fractions and the two geometry guards.

    Args:
        context: The analysis context, read for the per-sample table, the collection record's
            geometry block and the merged configuration the budget is re-resolved from.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population here is the set of recordings that
            scored an anchor, which only the table knows.

    Returns:
        The protocol's keys, the per-metric rows, the recomposition check, the two guards against
        their geometry-derived expectations, the budget record, and the paths written.
    """
    collection = context.collection
    per_sample = collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    per_guid = per_recording_means(per_sample, VALUE_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    rows = build_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    figure_name = str(
        figures.render_figure(
            build_tertile_figure(per_guid, rows), directory / TERTILE_FIGURE
        ).name
    )
    budget = write_budget_figures(
        dict(getattr(context, "config", None) or {}), directory
    )

    return {
        "n_samples": scored_sample_count(per_sample, TOTAL_COLUMN),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "metrics": rows,
        # Flat, finite scalars only: this is what the binding's headline registry digs into.
        "headline": headline_block(rows),
        # The property that makes the three tertiles a decomposition rather than three readouts.
        "recomposition": recomposition(per_guid),
        "geometry_guards": guard_record(
            per_guid, expected_geometry(getattr(collection, "record", None))
        ),
        "source_lag_warmth_note": WARMTH_EXPECTATION,
        "budget": budget,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            PER_RECORDING_FILENAME,
            SUMMARY_FILENAME,
            figure_name,
            *(budget.get("files") or []),
        ],
    }
