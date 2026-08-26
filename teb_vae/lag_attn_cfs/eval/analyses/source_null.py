r"""How much of the coupling readout is source *variation*, and how much is a clock.

$$\Delta_{\mathrm{clock}} \;=\;
\texttt{source\_conditioned\_kl\_raw} \;-\; \texttt{kld\_source\_null}$$

The hazard this sizes is the one no other control in this pipeline can see, and it exists here and
in no other cell of the grid. The source availability pattern $m^u_{t,c}$ is a **deterministic
function of $t$**, identical in every row of a batch, and it enters $q(z \mid Y, U)$ but not
$p(z \mid Y)$ -- so the posterior can be pushed off the prior by the availability clock alone, with
no source information in it at all, and the coupling readout would report that as coupling. The
permutation control deranges *rows*, and no permutation of rows can remove something every row
shares; ``perm_control`` is therefore structurally blind to this and is not a weaker version of it.

The null arm is a **zeroed** source stream rather than a permuted one, re-encoded through the
source gate, the input adapter and the source encoder. Both of those are nonlinear, so a zeroed
stream is not a rearrangement of a real one and the re-encode is where the arm's content lives. It
costs one source encode and no decode, and draws no ``randn_like``, so it does not move the
reparameterisation stream for the rest of the run.

**One thing weakens the claim in the model's favour, and it is emitted rather than left to be
noticed.** Zeroing floors the source's *variation*; it is not literally the availability pattern
acting alone, because the encoder's response to a flat trajectory is not the pattern's response.
So $\Delta_{\mathrm{clock}}$ is a slightly weaker statement than "the coupling exceeds the clock".
``lag_attn_cfs/DESIGN.md`` section 8 is the record and this analysis cites it rather than
restating the argument in a second place that could drift from it.

**Both readouts are reduced on one support**, the dense $(B, T)$ anchor mask the matched KL itself
was reduced over, summed over $d_z$ and divided by the same contributing-anchor count. That is
what makes their difference a difference rather than a comparison of two averaging conventions,
and it is a property of the collection pass rather than of this module -- ``tests/
test_eval_source_null.py`` proves it by driving a batch whose source stream is already zero, where
the two columns must come out bit-identical.

What is emitted is the difference **per recording**, with a bootstrap interval over recordings,
the fraction of recordings on which it is positive *with its denominator*, and a paired
signed-rank test over the per-recording pairs. The interval is the one the acceptance verdict is
decided on: a difference measured over fourteen recordings can clear any margin on its mean while
its interval crosses zero, so the mean is precisely the statistic that cannot decide this.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from teb_vae.lag_attn_cfs.eval._reuse import figures, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    positive_fraction,
    scored_sample_count,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "source_null"

#: What it writes. The per-recording frame's name and its ``coupling_minus_clock`` column are what
#: ``cross_subgroup`` reads, so both are a contract rather than a filename and a label.
PER_RECORDING_FILENAME = "source_null_per_recording.csv"
SUMMARY_FILENAME = "source_null_summary.csv"
DISTRIBUTION_FIGURE = "source_null_difference"

#: The matched coupling readout and the null beside it, in the order the difference is taken.
COUPLING_COLUMN = "source_conditioned_kl_raw"
NULL_COLUMN = "kld_source_null"

#: The difference itself. Computed in the collection pass per sample rather than differenced here
#: from two per-recording means, so the per-recording value is the mean of a per-sample difference
#: and not the difference of two means -- the two coincide only where every recording contributed
#: the same segments to both, which is true here and would stop being true the moment either
#: column acquired its own exclusion rule.
DIFFERENCE_COLUMN = "coupling_minus_clock"

#: Every column the per-recording chain reduces, and what each means.
METRICS: Tuple[Tuple[str, str], ...] = (
    (
        COUPLING_COLUMN,
        "the matched coupling readout: KL(q(z | Y, U) || p(z | Y)) per anchor, in nats",
    ),
    (
        NULL_COLUMN,
        "the same KL with the source stream zeroed and re-encoded, on the identical support -- "
        "what the availability pattern alone can account for, up to the encoder's nonlinearity",
    ),
    (
        DIFFERENCE_COLUMN,
        "the part of the coupling readout attributable to source variation; the acceptance "
        "criterion is stated on the lower end of this quantity's interval over recordings",
    ),
)

#: The columns the by-class and by-subgroup fan-out resolves. All three: a cohort whose coupling
#: differs may differ in the clock rather than in the source, and only carrying both halves beside
#: the difference lets a reader tell which.
GROUPED_METRICS: Tuple[str, ...] = tuple(column for column, _ in METRICS)

#: The unit every number here is in.
NATS_PER_ANCHOR = "nats per anchor"

#: The statement that must travel in the output because it weakens the claim in the model's favour
#: and nothing else in a run would surface it.
NULL_CAVEAT = (
    "the null zeroes the source stream, which floors its variation but is not literally the "
    "availability pattern acting alone -- both the input adapter and the source encoder are "
    "nonlinear, so the encoder's response to a flat trajectory is not the pattern's response. "
    "This difference is therefore a slightly weaker statement than 'the coupling exceeds the "
    "clock'. See lag_attn_cfs/DESIGN.md section 8 for the argument; it is cited rather than "
    "restated so the two cannot drift apart."
)

#: Why the permutation control cannot answer this, recorded beside the number so a reader holding
#: both does not read them as two attempts at one question.
PERM_CONTROL_NOTE = (
    "perm_control is blind to this hazard by construction rather than by weakness: it deranges "
    "rows, and the availability pattern is identical in every row of a batch, so no permutation of "
    "rows can remove it. The two controls answer different questions and both are reported."
)


def build_rows(per_guid: pd.DataFrame, *, resamples: int, seed: int) -> List[Dict[str, Any]]:
    """Summarise the two readouts and their difference over the recordings.

    Args:
        per_guid: Per-recording means.
        resamples: Bootstrap resamples, from ``eval_config.bootstrap_resamples``.
        seed: Bootstrap seed, from ``eval_config.seed``, so the interval is reproducible from the
            summary alone.

    Returns:
        One row per metric, carrying the mean and its interval, the quartiles, and -- on the
        difference row alone -- the positive fraction with its denominator and the paired
        signed-rank test over the two halves it is the difference of.
    """
    rows: List[Dict[str, Any]] = []
    for column, meaning in METRICS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        row: Dict[str, Any] = {
            "metric": column,
            "meaning": meaning,
            "unit": NATS_PER_ANCHOR,
            **{key: value for key, value in describe(values).items() if key != "metric"},
            "ci_lo": interval["lo"],
            "ci_hi": interval["hi"],
            "ci_method": interval["method"],
            "bootstrap_resamples": int(interval["resamples"]),
        }
        if column == DIFFERENCE_COLUMN:
            positive = positive_fraction(values)
            paired = shared_stats.wilcoxon_paired(
                finite_column(per_guid, NULL_COLUMN),
                finite_column(per_guid, COUPLING_COLUMN),
                label_left="source-null KL",
                label_right="matched coupling KL",
            )
            row.update(
                {
                    # The fraction and the count it came from, never one without the other: a
                    # recording that scored no anchor measured nothing, and counting it silently
                    # as evidence against would make a coverage collapse read as a falling
                    # positive fraction.
                    "positive_fraction": positive["fraction"],
                    "n_positive": positive["n_positive"],
                    "n_recordings_scored": positive["n"],
                    "n_recordings_dropped_not_finite": positive["n_dropped_not_finite"],
                    "wilcoxon_p_value": paired["p_value"],
                    "wilcoxon_n_pairs": paired["n_pairs"],
                    "wilcoxon_median_difference": paired["median_difference"],
                }
            )
        rows.append(row)
    return rows


def difference_record(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten the difference row into the block the acceptance verdict is decided from.

    Promoted out of the row list rather than left in it because the verdict resolves a *path* into
    this analysis's block, and a path into a list positioned by a filter is a path that silently
    resolves to the wrong row the day a metric is added above it.

    Args:
        rows: What :func:`build_rows` produced.

    Returns:
        The measured difference, both interval ends, the denominators and the caveat -- or the same
        keys at ``NaN`` when the run produced no difference at all, which the verdict reads as
        unmeasured rather than as zero.
    """
    by_metric = {str(entry.get("metric")): entry for entry in rows}
    row = by_metric.get(DIFFERENCE_COLUMN, {})
    return {
        "metric": DIFFERENCE_COLUMN,
        "unit": NATS_PER_ANCHOR,
        # Both halves beside the difference, because the headline registry reads this block and a
        # difference quoted without the two numbers it came from cannot be sanity-checked: 0.1 nats
        # out of 0.2 and 0.1 out of 20 are the same difference and opposite findings.
        "source_conditioned_kl_raw_nats": by_metric.get(COUPLING_COLUMN, {}).get(
            "mean", float("nan")
        ),
        "kld_source_null_nats": by_metric.get(NULL_COLUMN, {}).get("mean", float("nan")),
        "coupling_minus_clock_nats": row.get("mean", float("nan")),
        "ci_lo": row.get("ci_lo", float("nan")),
        "ci_hi": row.get("ci_hi", float("nan")),
        "n_recordings": int(row.get("n") or 0),
        "positive_fraction": row.get("positive_fraction", float("nan")),
        "n_positive": row.get("n_positive"),
        "n_recordings_scored": row.get("n_recordings_scored"),
        "wilcoxon_p_value": row.get("wilcoxon_p_value"),
        "wilcoxon_n_pairs": row.get("wilcoxon_n_pairs"),
        "caveat": NULL_CAVEAT,
    }


def build_difference_figure(
    per_guid: pd.DataFrame, rows: List[Dict[str, Any]]
) -> Any:
    """Draw the difference's distribution and both halves it came from.

    Two panels. The histogram answers "on how many recordings did the coupling exceed the clock,
    and by how much", with zero marked because the sign is the finding and a distribution
    straddling it is a different result from one sitting above it. The violin puts the matched
    readout beside the null under their own names, so a large difference between two large numbers
    is visibly that rather than being inferred from one subtraction.

    Args:
        per_guid: Per-recording means.
        rows: The summary rows, read for the difference's denominator.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    difference = next(
        (row for row in rows if row.get("metric") == DIFFERENCE_COLUMN), {}
    )
    figures.histogram_panel(
        axes[0, 0],
        finite_column(per_guid, DIFFERENCE_COLUMN),
        title=(
            f"coupling minus availability clock per recording, "
            f"n = {int(difference.get('n_recordings_scored') or 0)}"
        ),
        xlabel="nats per anchor",
        reference=0.0,
        reference_label="the clock accounts for all of it",
    )
    figures.violin_panel(
        axes[1, 0],
        {
            COUPLING_COLUMN: finite_column(per_guid, COUPLING_COLUMN),
            NULL_COLUMN: finite_column(per_guid, NULL_COLUMN),
        },
        title="the matched coupling readout and the source-null arm, on one support",
        ylabel="nats per anchor",
    )
    return figure


def run_source_null_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report $\\Delta_{\\mathrm{clock}}$ per recording, with the interval the verdict reads.

    Args:
        context: The analysis context, read for the per-sample table.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the three metric rows, the flat difference block the acceptance
        verdict resolves, the two caveats, and the paths written.
    """
    per_sample = context.collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    per_guid = per_recording_means(per_sample, GROUPED_METRICS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    rows = build_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    figure_name = str(
        figures.render_figure(
            build_difference_figure(per_guid, rows), directory / DISTRIBUTION_FIGURE
        ).name
    )
    return {
        "n_samples": scored_sample_count(per_sample, DIFFERENCE_COLUMN),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "metrics": rows,
        # The block the availability-clock verdict is decided from; see difference_record for why
        # it is promoted out of the list rather than filtered back out of it.
        "difference": difference_record(rows),
        "unit": NATS_PER_ANCHOR,
        "caveat": NULL_CAVEAT,
        "perm_control_note": PERM_CONTROL_NOTE,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [PER_RECORDING_FILENAME, SUMMARY_FILENAME, figure_name],
    }
