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

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import figures_seam
from teb_vae.lag_attn_cfs.eval._reuse import figures, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    COEFFICIENT_LAG_AXIS_LABEL,
    GROUP_DELAY_CAVEAT,
    compensated_seconds_axis,
)
from teb_vae.lag_attn_cfs.eval.lag_shape import (
    PEAK_FRACTION,
    band_mass,
    degeneracy,
    peak_width,
    rectified_profile,
)
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

#: The lag-resolved half. One row per lag, and the run's durable record of which lags the
#: clock-excess selection kept -- a mask reconstructed later from a re-run is not the mask the
#: numbers beside it were chosen with.
LAG_PROFILE_FILENAME = "source_null_lag_profile.csv"
LAG_PROFILE_FIGURE = "source_null_lag_profile"

#: The statement every rectified total carries. It is the one thing a reader can get wrong here
#: and the arithmetic does not announce: rectification only ever adds, so the positive total is an
#: upper bound on the signed one, and the signed one is the gated scalar.
RECTIFICATION_CAVEAT = (
    "the clock-excess profile is SIGNED -- the null arm re-poses the posterior against a zeroed "
    "source, so its attention is its own and can exceed the matched arm's at a lag. Only the "
    "signed sum equals coupling_minus_clock; the rectified (positive) total is an UPPER BOUND on "
    "it, larger by exactly the negative mass. A band share is taken of the positive part, because "
    "a share of a signed vector is not a share -- so the shares below partition the positive mass "
    "and not the gated scalar. rectified_frac is how much was discarded relative to what survived"
)

#: Why the mask is guarded rather than always emitted, in the record beside the mask itself.
MASK_GUARD_NOTE = (
    "the delta mask is the contiguous run of bins around the clock-excess peak at or above "
    f"{PEAK_FRACTION:g} of it -- lag_shape.peak_width's own definition, so no threshold is "
    "invented here. It is withheld entirely when the clock-excess profile is degenerate, because "
    "entmax15 assigns lags exactly zero and a flat or nearly empty profile still has a perfectly "
    "confident argmax: a mask cut from one would name a band the run has no evidence for. A "
    "withheld mask is a measurement, not a failure -- the geometry-fixed bands remain the "
    "selection that needs no estimate"
)

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
            # Matched first, null second: the shared test reports the median of left minus
            # right, and this row IS coupling minus clock, so the sign must read the same way.
            paired = shared_stats.wilcoxon_paired(
                finite_column(per_guid, COUPLING_COLUMN),
                finite_column(per_guid, NULL_COLUMN),
                label_left="matched coupling KL",
                label_right="source-null KL",
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


# =================================================================================================
# The lag-resolved half: where in the past the coupling exceeded the availability clock
# =================================================================================================
def _finite_or_none(value: Any) -> Optional[float]:
    """A float for the headline, or ``None`` where there is no number.

    ``None`` and not ``NaN``, and the distinction is enforced rather than stylistic: the headline
    finiteness check treats a non-finite *number* as a failed run and ``None`` as an analysis that
    did not report. A ``NaN`` here would fail every run whose clock-excess profile was not
    measured, which is a configuration state rather than a defect.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def delta_mask(positive: Sequence[float], shape: Dict[str, Any]) -> Tuple[
    Optional[Tuple[int, int]], str
]:
    r"""The lag band the clock-excess profile itself nominates, or a stated refusal.

    The contiguous run around the peak at or above :data:`PEAK_FRACTION` of it -- which is exactly
    :func:`~teb_vae.lag_attn_cfs.eval.lag_shape.peak_width`, reused rather than reimplemented so
    that "the peak" means one thing across this package and no threshold is invented here.

    **Withheld entirely when the profile is degenerate**, and that is the load-bearing half. This
    family's diagnosed run put $67.5\%$ of its KL in an availability clock and $0.160$ nats of
    source content across $91$ lags; a mask cut from a profile that flat would name a band on
    arithmetic accident, and it would look exactly like a finding. The guard firing is a result to
    report, not a failure to route around.

    Args:
        positive: The rectified clock-excess profile.
        shape: What :func:`~teb_vae.lag_attn_cfs.eval.lag_shape.degeneracy` said about it.

    Returns:
        ``(band, reason)`` -- the inclusive band and an empty reason, or ``None`` and the sentence
        saying why there is none.
    """
    if not len(positive):
        return None, "the run carries no clock-excess profile to cut a mask from"
    if shape.get("degenerate"):
        reasons = "; ".join(shape.get("reasons") or []) or "the profile has no readable shape"
        return None, (
            f"the clock-excess profile is degenerate, so no mask is emitted: {reasons}. "
            f"The geometry-fixed bands remain the selection that needs no estimate"
        )
    peak = peak_width(positive)
    if peak.get("lo") is None or peak.get("hi") is None:
        return None, "the clock-excess profile has no readable peak"
    return (int(peak["lo"]), int(peak["hi"])), ""


def lag_profile_frame(
    lag: Dict[str, Any],
    seconds: np.ndarray,
    positive: np.ndarray,
    mask: Optional[Tuple[int, int]],
    bands: Dict[str, Tuple[int, int]],
) -> pd.DataFrame:
    """Lay the matched arm, the null arm and their difference out as one table, keyed by lag.

    One table rather than three, and keyed by lag rather than by arm, because the question a reader
    brings to it is always about a *lag*: what did the coupling say here, what did the clock
    account for here, and what was left.

    Args:
        lag: The collection pass's lag block.
        seconds: The compensated axis.
        positive: The rectified clock-excess profile.
        mask: The band the profile nominated, or ``None``.
        bands: The geometry-fixed partition, for the band each lag falls in.

    Returns:
        One row per lag.
    """
    matched = list(lag.get("kl_lag_profile") or [])
    null = list(lag.get("kl_lag_profile_null") or [])
    excess = list(lag.get("kl_lag_profile_clock_excess") or [])
    rows: List[Dict[str, Any]] = []
    for index, second in enumerate(seconds):
        # The band a lag belongs to, or empty. Written per row rather than left to a join, so the
        # observational table and the interventional one can be read against each other directly.
        member = next(
            (name for name, span in bands.items() if span[0] <= index <= span[1]), ""
        )
        rows.append(
            {
                "lag_step": int(index),
                "compensated_seconds": float(second),
                "kl_nats": float(matched[index]) if index < len(matched) else float("nan"),
                "kl_null_nats": float(null[index]) if index < len(null) else float("nan"),
                "clock_excess_nats": (
                    float(excess[index]) if index < len(excess) else float("nan")
                ),
                "clock_excess_positive_nats": (
                    float(positive[index]) if index < len(positive) else float("nan")
                ),
                "in_delta_mask": bool(mask is not None and mask[0] <= index <= mask[1]),
                "band": member,
            }
        )
    return pd.DataFrame(rows)


def lag_record(
    lag: Dict[str, Any], bands: Dict[str, Tuple[int, int]]
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    """Reduce the clock-excess profile to the block the headline resolves.

    Args:
        lag: The collection pass's lag block, read off ``collection.results``.
        bands: The geometry-fixed lag partition.

    Returns:
        ``(record, seconds, positive, mask)``. The record carries every key on every run --
        ``None`` where nothing was measured, never ``NaN``, because the headline finiteness check
        reads a non-finite number as a broken run and a ``None`` as an analysis that did not
        report.
    """
    excess = [float(value) for value in (lag.get("kl_lag_profile_clock_excess") or [])]
    n_lags = int(lag.get("n_lags") or len(lag.get("kl_lag_profile") or []) or 0)
    delay_steps = int(lag.get("delay_steps") or 0)
    seconds = compensated_seconds_axis(n_lags, delay_steps) if n_lags else np.zeros(0)

    if not excess:
        return (
            {
                "measured": False,
                "reason": (
                    "the collection pass reported no clock-excess profile, so this run predates "
                    "the lag-resolved source-null arm or scored no anchors"
                ),
                "n_lags": n_lags,
                "delay_steps": delay_steps,
                "clock_excess_argmax_lag_step": None,
                "clock_excess_compensated_seconds": None,
                "clock_excess_peak_share": None,
                "clock_excess_degenerate": None,
                "clock_excess_rectified_frac": None,
                "net_nats": None,
                "positive_nats": None,
                "negative_nats": None,
                "band_shares": {},
                "delta_mask": None,
                "delta_mask_reason": "there is no clock-excess profile to cut a mask from",
                "mask_guard": MASK_GUARD_NOTE,
                "rectification_caveat": RECTIFICATION_CAVEAT,
                "axis_caveat": GROUP_DELAY_CAVEAT,
            },
            seconds,
            np.zeros(0),
            None,
        )

    positive, census = rectified_profile(np.asarray(excess, dtype=np.float64))
    shape = degeneracy(positive)
    mask, mask_reason = delta_mask(positive, shape)
    above = peak_width(positive)
    argmax = above.get("argmax")
    # The share of the POSITIVE mass in the peak, which is the concentration a positional claim
    # rests on. Taken through the shared reducer rather than recomputed.
    from teb_vae.lag_attn_cfs.eval.lag_shape import mass_above

    peak_share = mass_above(positive).get("share")
    return (
        {
            "measured": True,
            "n_lags": n_lags,
            "delay_steps": delay_steps,
            "clock_excess_argmax_lag_step": None if argmax is None else int(argmax),
            "clock_excess_compensated_seconds": (
                None if argmax is None or argmax >= seconds.size else float(seconds[argmax])
            ),
            "clock_excess_peak_share": _finite_or_none(peak_share),
            "clock_excess_degenerate": bool(shape["degenerate"]),
            "clock_excess_rectified_frac": _finite_or_none(census["rectified_frac"]),
            # The signed sum, which is the gated scalar, beside the two halves it is made of.
            "net_nats": _finite_or_none(census["net_nats"]),
            "positive_nats": _finite_or_none(census["positive_nats"]),
            "negative_nats": _finite_or_none(census["negative_nats"]),
            # Shares of the POSITIVE mass, on the same partition the interventional readout
            # removes source from -- so the two instruments' answers are stated on one axis.
            "band_shares": {
                name: _finite_or_none(band_mass(positive, span)["share"])
                for name, span in bands.items()
            },
            "delta_mask": None if mask is None else [int(mask[0]), int(mask[1])],
            "delta_mask_reason": mask_reason,
            "mask_guard": MASK_GUARD_NOTE,
            "rectification_caveat": RECTIFICATION_CAVEAT,
            "axis_caveat": GROUP_DELAY_CAVEAT,
        },
        seconds,
        positive,
        mask,
    )


def build_lag_figure(
    frame: pd.DataFrame,
    record: Dict[str, Any],
    bands: Dict[str, Tuple[int, int]],
    seconds: np.ndarray,
) -> Any:
    """Draw the two arms in nats and their signed difference beneath, on the compensated axis.

    Two panels rather than one, because the two quantities are on different scales and share only
    an axis: the arms are totals in nats and the difference is a residual around zero, and drawing
    them together would make a $0.16$-nat excess a flat line under a $0.49$-nat total.

    Args:
        frame: The per-lag table.
        record: The lag record, for the mask and the degeneracy verdict.
        bands: The geometry-fixed partition, shaded and labelled.
        seconds: The compensated axis.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axis = np.asarray(frame["compensated_seconds"], dtype=np.float64)
    top.plot(axis, frame["kl_nats"], label="matched: KL(q(z|Y,U) || p(z|Y))")
    top.plot(axis, frame["kl_null_nats"], label="source-null: the availability clock")
    top.set_ylabel("nats per anchor")
    top.legend(loc="upper right", fontsize="small")
    top.set_title("Where the coupling exceeded the availability clock")

    bottom.axhline(0.0, linewidth=0.8, color="0.4")
    bottom.plot(axis, frame["clock_excess_nats"], color="C3", label="clock-excess (signed)")
    bottom.set_ylabel("nats per anchor")
    bottom.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)

    # The geometry-fixed partition, shaded on both panels so a reader can place a feature in the
    # same vocabulary the interventional readout reports its deltas in.
    # Bin EDGES: the bands are inclusive lag ranges, so a shade from centre to centre would be
    # one bin short at each end, adjacent bands would leave a gap, and a one-lag band would vanish.
    half_lag = SECONDS_PER_STEP / 2.0
    for index, (name, span) in enumerate(bands.items()):
        lo = float(seconds[max(span[0], 0)]) - half_lag if seconds.size else 0.0
        hi = float(seconds[min(span[1], seconds.size - 1)]) + half_lag if seconds.size else 0.0
        for panel in (top, bottom):
            panel.axvspan(lo, hi, color="0.9" if index % 2 else "0.95", zorder=0)
        top.annotate(
            name, xy=((lo + hi) / 2.0, top.get_ylim()[1]), ha="center", va="top",
            fontsize="x-small", color="0.35",
        )

    mask = record.get("delta_mask")
    if mask and seconds.size:
        bottom.axvspan(
            float(seconds[mask[0]]) - half_lag, float(seconds[mask[1]]) + half_lag,
            color="C3", alpha=0.15, zorder=1, label="delta mask",
        )
    bottom.legend(loc="upper right", fontsize="small")
    if record.get("clock_excess_degenerate"):
        bottom.annotate(
            "clock-excess profile is degenerate: no mask emitted",
            xy=(0.02, 0.05), xycoords="axes fraction", fontsize="small", color="C3",
        )
    figures_seam.caveat_note(figure)
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
        verdict resolves, the lag-resolved decomposition of that same difference, the two caveats,
        and the paths written.
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

    # The lag-resolved half. Read off the collection pass's own lag block, which round-trips
    # through ``collection.json`` -- so this works on an offline re-run against a finished
    # directory, exactly as the scalar half above does.
    lag = dict(dict(getattr(context.collection, "results", None) or {}).get("lag") or {})
    # The geometry-fixed partition, read from the same key the interventional readout removes
    # source from. Not a second setting: one partition per run means ``near`` names one lag range
    # in the observational table and in the interventional one.
    bands = {
        str(name): (int(span[0]), int(span[1]))
        for name, span in (eval_config.get("occlusion_bands") or {}).items()
        if isinstance(span, (list, tuple)) and len(span) == 2
    }
    lag_block, seconds, positive, mask = lag_record(lag, bands)
    written = [PER_RECORDING_FILENAME, SUMMARY_FILENAME, figure_name]
    if lag_block["measured"]:
        frame = lag_profile_frame(lag, seconds, positive, mask, bands)
        frame.to_csv(directory / LAG_PROFILE_FILENAME, index=False)
        written.append(LAG_PROFILE_FILENAME)
        written.append(
            str(
                figures.render_figure(
                    build_lag_figure(frame, lag_block, bands, seconds),
                    directory / LAG_PROFILE_FIGURE,
                ).name
            )
        )

    return {
        "n_samples": scored_sample_count(per_sample, DIFFERENCE_COLUMN),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "metrics": rows,
        # Where in the past the coupling exceeded the availability clock, and which lags a
        # selection built from it would keep. Every key is present on every run so the headline
        # paths resolve; see ``lag_record`` for why the unmeasured ones are None and not NaN.
        "lag": lag_block,
        # The block the availability-clock verdict is decided from; see difference_record for why
        # it is promoted out of the list rather than filtered back out of it.
        "difference": difference_record(rows),
        "unit": NATS_PER_ANCHOR,
        "caveat": NULL_CAVEAT,
        "perm_control_note": PERM_CONTROL_NOTE,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": written,
    }
