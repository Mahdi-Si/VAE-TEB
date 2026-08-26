r"""The context-sufficiency gap: what the latent bottleneck costs the forecast.

Every other analysis compares the two branches with each other. This one compares the
target-only branch with what a decoder could have done had the bottleneck not been there:

$$\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}},$$

where $D_{\mathrm{oracle}}$ comes from a decoder of the same capacity reading the target
encoder's own state instead of the $d_z$-wide latent, fitted on one half of the evaluation
recordings and scored on the other. Until that number exists, ``pred_gap`` is a difference
between two models rather than an information rate: a small gap is equally consistent with "the
source adds little" and with "the bottleneck discards so much that neither branch could have used
what the source offered".

**The number is an estimate and the run says so in those words.** Conditioning on
``target_state`` omits the encoder's own information loss and biases the gap **down**; fitting the
probe on the evaluation population while $D_{\mathrm{base}}$ comes from a model trained on the
disjoint, healthier pretraining cohort biases it **up**. The two oppose, neither is measured, and
both sentences are written into the emitted record rather than left in a document beside it.

**Three quantities, one set of recordings.** $D_{\mathrm{oracle}}$, $D_{\mathrm{base}}$ and
$D_{\mathrm{full}}$ are all resolved on the *held-out* recordings, by joining the probe's
per-segment scores onto the per-sample table -- so the two gaps drawn on ``sufficiency.pdf`` are
differences over the same denominator, not three numbers from three populations.

**This is the second analysis that touches the model**, and for a reason retention cannot serve: a
probe fit is thousands of passes over the same segments, and what it needs -- the encoder state of
every segment -- is not on either durable table. A pass with no checkpoint records a skip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval import oracle
from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "sufficiency"

#: What it writes.
PER_RECORDING_FILENAME = "sufficiency_per_recording.csv"
SUMMARY_FILENAME = "sufficiency_summary.csv"
CURVE_FILENAME = "oracle_training_curve.csv"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
SUFFICIENCY_FIGURE = "sufficiency"

#: The three block scores compared, as ``(column, label)``. All three are Monte Carlo marginalised
#: where the model produced them, so the oracle -- which has no latent to marginalise over -- is
#: compared against the same estimator the headline quotes rather than against the training-path
#: single-draw column.
SCORE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("nll_oracle_block", "oracle (target_state)"),
    ("mc_nll_base_block", "target-only (base)"),
    ("mc_nll_full_block", "source-conditioned (full)"),
)

#: The two differences, as ``(name, left column, right column, meaning)``. Both are left minus
#: right, both in nats per anchor, and both are computed **after** the per-recording chain so that
#: a recording contributes one number to each.
GAP_METRICS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "delta_suff_nats",
        "mc_nll_base_block",
        "nll_oracle_block",
        "context-sufficiency gap D_base - D_oracle: what the latent bottleneck costs the "
        "forecast, as an estimate biased in two unmeasured directions",
    ),
    (
        "pred_gap_mc_nats",
        "mc_nll_base_block",
        "mc_nll_full_block",
        "the coupling readout D_base - D_full on the same held-out recordings, so the two gaps "
        "are drawn against one denominator",
    ),
)

#: The metrics resolved by cohort.
GROUPED_METRICS: Tuple[str, ...] = (
    "delta_suff_nats", "nll_oracle_block", "mc_nll_base_block",
)


def _epoch_key(value: Any) -> Optional[int]:
    """Return an ``epoch`` as whole seconds, or ``None`` when it is not a finite number.

    The join key between the probe's scores and the per-sample table. Rounded rather than compared
    as a float: both sides read the same field off the same batch, but one of them has been
    through a CSV round trip on an offline re-run, and a join that silently matched nothing would
    report an empty analysis rather than a broken key.

    Args:
        value: The segment's ``epoch``.

    Returns:
        The rounded value, or ``None``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(round(number)) if np.isfinite(number) else None


def join_oracle_scores(per_sample: pd.DataFrame, per_segment: Dict[str, Any]) -> pd.DataFrame:
    """Attach the probe's per-segment scores to the rows of the per-sample table they describe.

    Args:
        per_sample: The collection pass's per-sample table.
        per_segment: The oracle's ``per_segment`` block -- ``guid``, ``epoch`` and the score.

    Returns:
        The held-out rows of ``per_sample`` with ``nll_oracle_block`` beside their own readouts.
        Empty with the column present when nothing joined, which a caller reports rather than
        divides by.
    """
    scores = pd.DataFrame(
        {
            "guid": [str(value) for value in per_segment["guid"]],
            "_epoch_key": [_epoch_key(value) for value in per_segment["epoch"]],
            "nll_oracle_block": np.asarray(per_segment["nll_oracle_block"], dtype=np.float64),
            "oracle_n_anchors": np.asarray(per_segment["oracle_n_anchors"], dtype=np.float64),
        }
    ).dropna(subset=["_epoch_key"])
    if per_sample.empty or "guid" not in per_sample.columns or "epoch" not in per_sample.columns:
        return per_sample.head(0).assign(
            nll_oracle_block=pd.Series(dtype=np.float64),
            oracle_n_anchors=pd.Series(dtype=np.float64),
        )

    left = per_sample.copy()
    left["guid"] = left["guid"].astype(str)
    left["_epoch_key"] = [_epoch_key(value) for value in left["epoch"]]
    merged = left.merge(scores, on=["guid", "_epoch_key"], how="inner")
    return merged.drop(columns=["_epoch_key"])


def build_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise the three scores and the two gaps, each with an interval over recordings.

    The gaps carry a paired test as well as an interval: every held-out recording contributes both
    sides, so the paired form removes the between-recording variance that dominates every readout
    here and the unpaired one would throw that away.

    Args:
        per_guid: Per-recording means of the three score columns.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per reported quantity.
    """
    rows: List[Dict[str, Any]] = []
    for column, label in SCORE_COLUMNS:
        interval = shared_stats.bootstrap_ci(
            finite_column(per_guid, column), resamples=resamples, seed=seed
        )
        rows.append({
            "metric": column,
            "label": label,
            "meaning": "block score in nats per anchor, lower is better",
            "n": int(interval["n"]),
            "value": interval["point"],
            "lo": interval["lo"],
            "hi": interval["hi"],
        })

    for name, left_column, right_column, meaning in GAP_METRICS:
        left = finite_column(per_guid, left_column)
        right = finite_column(per_guid, right_column)
        interval = shared_stats.bootstrap_ci(left - right, resamples=resamples, seed=seed)
        paired = shared_stats.wilcoxon_paired(
            left, right, label_left=left_column, label_right=right_column
        )
        rows.append({
            "metric": name,
            "label": f"{left_column} - {right_column}",
            "meaning": meaning,
            "n": int(interval["n"]),
            "value": interval["point"],
            "lo": interval["lo"],
            "hi": interval["hi"],
            "p_value": paired.get("p_value"),
            "median_paired_difference": paired.get("median_difference"),
            "test_note": paired.get("note", ""),
        })
    return rows


def curve_frame(record: Dict[str, Any]) -> pd.DataFrame:
    """Flatten both probes' held-out curves into one long frame.

    Args:
        record: The oracle's record.

    Returns:
        ``width_multiplier``, ``step``, ``held_out_nats`` and ``fit_nats``, one row per evaluation
        point. Empty with those columns when no fit ran.
    """
    columns = ["width_multiplier", "step", "held_out_nats", "fit_nats"]
    fits = [record.get("fit")] + [(record.get("capacity") or {}).get("wide_fit")]
    rows = [
        {
            "width_multiplier": int(fit.get("width_multiplier", 1)),
            "step": float(point.get("step", np.nan)),
            "held_out_nats": float(point.get("held_out_nats", np.nan)),
            "fit_nats": float(point.get("fit_nats", np.nan)),
        }
        for fit in fits
        if isinstance(fit, dict)
        for point in (fit.get("curve") or [])
    ]
    return pd.DataFrame(rows, columns=columns)


def build_sufficiency_figure(
    per_guid: pd.DataFrame, curve: pd.DataFrame, rows: Sequence[Dict[str, Any]]
) -> Any:
    r"""Draw the three scores side by side and the probe's own convergence beneath them.

    The upper panel is where the two gaps are read, so both are annotated on it rather than left
    for a reader to subtract: $\Delta_{\mathrm{suff}}$ between the oracle and the base violin, and
    ``pred_gap`` between the base and full ones. The lower panel is the evidence for whether the
    upper one may be believed -- a held-out curve still descending at its right-hand edge is a
    probe that understates the gap it was fitted to measure.

    Args:
        per_guid: Per-recording means of the three score columns.
        curve: The long-format training curve.
        rows: The summary rows, read for the two gap point estimates.

    Returns:
        The figure; the caller renders and closes it.
    """
    by_metric = {str(row["metric"]): row for row in rows}
    gaps = " | ".join(
        f"{name} = {float(by_metric[name]['value']):.4g} "
        f"[{float(by_metric[name]['lo']):.4g}, {float(by_metric[name]['hi']):.4g}]"
        for name, _left, _right, _meaning in GAP_METRICS
        if name in by_metric and np.isfinite(float(by_metric[name]["value"]))
    )

    figure, axes = figures.new_figure(2)
    figures.violin_panel(
        axes[0, 0],
        {label: finite_column(per_guid, column) for column, label in SCORE_COLUMNS},
        title=(
            "Block score per held-out recording (lower is better)"
            + (f" -- {gaps}" if gaps else "")
        ),
        ylabel="nats per anchor",
    )

    axis = axes[1, 0]
    if len(curve):
        widths = sorted({int(value) for value in curve["width_multiplier"]})
        # One x axis for every series, so the two probes' curves are read against the same steps
        # even where one of them was evaluated at a different interval.
        steps = np.unique(np.asarray(curve["step"], dtype=np.float64))
        series, labels = [], []
        for width in widths:
            block = curve[curve["width_multiplier"] == width]
            for column, kind in (("held_out_nats", "held out"), ("fit_nats", "fit")):
                lookup = dict(zip(np.asarray(block["step"], dtype=np.float64),
                                  np.asarray(block[column], dtype=np.float64)))
                series.append(np.asarray([lookup.get(step, np.nan) for step in steps]))
                labels.append(f"width x{width}, {kind}")
        figures.multi_line_panel(
            axis, steps, np.vstack(series), labels,
            title="Oracle probe fit (curve points are a fixed subsample of each side)",
            xlabel="optimizer step", ylabel="nats per anchor",
        )
    else:
        axis.text(0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center",
                  transform=axis.transAxes)
        figures.style_axes(axis)
    return figure


def _skip(reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the protocol's keys for a pass that measured nothing, and say why.

    Args:
        reason: What stopped the measurement.
        extra: Anything already computed before the guard fired.

    Returns:
        The recorded skip.
    """
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": True},
        "skipped": True,
        "reason": reason,
        "files": [],
        **(extra or {}),
    }


def run_sufficiency_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit the oracle probe, report the sufficiency gap, and say how to read it.

    Args:
        context: The analysis context, read for the per-sample table and -- with the per-sample
            pages, uniquely -- for the task and the loader the probe is fitted through.
        eval_config: The validated block, for the seed, ``caps.oracle`` and the bootstrap
            settings.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused, and deliberately not renamed: the argument is
            the protocol's, and the *oracle* probe is a different thing entirely.

    Returns:
        The protocol's keys plus the gap, the probe's convergence and capacity records, and both
        bias directions. A pass with no model records a skip.
    """
    collection = context.collection
    per_sample = getattr(collection, "per_sample", None)
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None or per_sample is None or per_sample.empty:
        return _skip(
            "the oracle is a decoder fitted on this split's encoder states, so it needs a model "
            "and a loader rather than a table; this pass built neither, which is what an offline "
            "re-run against a finished directory is"
        )

    record = oracle.run_oracle(task, loader, eval_config=eval_config)
    if record.get("skipped"):
        return _skip(str(record.get("reason", "the oracle reported no measurement")),
                     {"oracle": record})

    merged = join_oracle_scores(per_sample, record["per_segment"])
    if merged.empty:
        return _skip(
            "no held-out segment the probe scored could be joined back onto the per-sample table "
            "by (guid, epoch), so D_oracle and D_base would describe different recordings",
            {"oracle": {name: value for name, value in record.items() if name != "per_segment"}},
        )

    per_guid = per_recording_means(merged, [column for column, _label in SCORE_COLUMNS])
    for name, left_column, right_column, _meaning in GAP_METRICS:
        per_guid[name] = finite_column(per_guid, left_column) - finite_column(
            per_guid, right_column
        )
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    rows = build_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    curve = curve_frame(record)
    curve.to_csv(directory / CURVE_FILENAME, index=False)
    figures.render_figure(
        build_sufficiency_figure(per_guid, curve, rows), directory / SUFFICIENCY_FIGURE
    )

    by_metric = {str(row["metric"]): row for row in rows}
    fit = record["fit"]
    logger.info(
        f"{ANALYSIS_DIRNAME}: delta_suff = "
        f"{float(by_metric['delta_suff_nats']['value']):.4g} nats/anchor over "
        f"{int(by_metric['delta_suff_nats']['n'])} held-out recording(s); probe "
        f"{'converged' if fit['converged'] else 'did NOT converge'}"
    )
    return {
        "n_samples": int(merged["nll_oracle_block"].notna().sum()),
        "composition": {
            "n_held_out_recordings": int(len(per_guid)),
            "n_fit_recordings": int(record["split"]["n_fit_recordings"]),
        },
        "plan": {
            "capped": True,
            "cap": record["cache"].get("cap"),
            "seed": seed,
            "bootstrap_resamples": resamples,
            # The amendment to the one-model-touching-pass rule, as a number rather than a
            # surprise in a profile: this analysis runs the encoder over the split a second time
            # and holds the result for the duration of the fit.
            "extra_encoder_pass": {
                "reason": (
                    "a probe fit is thousands of passes over the same segments, and the encoder "
                    "state they read is on neither durable table"
                ),
                "n_segments": int(record["cache"]["n_segments"]),
                "n_bytes": int(record["cache"]["n_bytes"]),
            },
            "fit": record["settings"],
        },
        "metrics": rows,
        "gap": {
            "delta_suff_nats": by_metric["delta_suff_nats"]["value"],
            "d_oracle_nats": by_metric["nll_oracle_block"]["value"],
            "d_base_mc_nats": by_metric["mc_nll_base_block"]["value"],
            "d_full_mc_nats": by_metric["mc_nll_full_block"]["value"],
        },
        "split": record["split"],
        "convergence": {
            "converged": bool(fit["converged"]),
            "detail": fit["convergence_detail"],
            "final_held_out_nats": fit["final_held_out_nats"],
            "best_held_out_nats": fit["best_held_out_nats"],
            "best_step": fit["best_step"],
        },
        "capacity": record["capacity"],
        # In the emitted record, not only in the documentation: a reader meets the number here.
        "bias_directions": record["bias_directions"],
        "estimate_not_a_bound": record["estimate_not_a_bound"],
        "score_distribution": [
            describe(finite_column(per_guid, column), name=column)
            for column, _label in SCORE_COLUMNS
        ],
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            PER_RECORDING_FILENAME, SUMMARY_FILENAME, CURVE_FILENAME, SUFFICIENCY_FIGURE
        ],
    }
