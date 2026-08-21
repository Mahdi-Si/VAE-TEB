r"""Who was evaluated, in which cohorts, and the time axis those cohorts are cut on.

Two things live here, and they are one subject: the *population* an evaluation describes, and the
one axis every clinical reading resolves it against.

**The time axis.** ``epoch`` is the segment's start in seconds relative to delivery and is
**negative before it**, so $h = -\mathrm{epoch}/3600$ is hours before delivery and is
non-negative for a segment recorded before delivery. Segments are binned on a fixed $0.5$ h grid.
:data:`TRAJECTORY_BIN_HOURS` is a module constant and deliberately not an ``eval_config`` key --
an operator who could widen it could merge two windows until a difference appeared or disappeared,
which is the same argument that keeps the significance level out of the configuration.

The binning lives here rather than in the analysis that needed it first because two analyses cut
on it -- the trajectory of the two coupling readouts, and the lag structure resolved by time
window -- and an analysis may not import another.

**The unit is the recording, inside the window as well as across it.** A window's value for a
recording is the mean over that recording's segments falling in it, and every statistic then runs
over one value per recording. Skipping that step would let a recording contributing eleven
segments to a window outvote one contributing two, which is the pseudo-replication the whole
aggregation chain exists to keep out.

**The provenance.** Which shards were evaluated, how the segments and the recordings split across
the subgroups and the classes, and the two statements a reader must have before quoting any class
contrast:

* The evaluation cohort is **disjoint** from the training cohort, or it is not. Computed by
  comparing the two resolved dataset lists rather than asserted, because a constant outlives the
  configuration that made it true.
* When it is disjoint, every class contrast is **out of distribution** rather than held-out
  discrimination -- and the scope is wider than the class axis suggests. The pretraining split is
  built from the healthy *with-background* recordings only
  (``hdf5_dataset/create_hdf5_dataset.py``'s pre-training-only mode draws exactly
  ``healthy_bg_cs`` and ``healthy_bg_no_cs``), so the two healthy **no-background** subgroups are
  unseen as well, not just acidosis and HIE.

A forecast score from an evaluation run is also not comparable with a ``test_*`` metric logged
during training: the populations differ, and nothing in either number says so. That sentence
travels in the record rather than in a document beside it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval._reuse import labels

#: Width of a time-before-delivery window, in hours. See the module docstring for why this is a
#: constant rather than a setting.
TRAJECTORY_BIN_HOURS = 0.5

#: Seconds per hour, for the ``epoch``-to-hours conversion.
SECONDS_PER_HOUR = 3600.0

#: The columns :func:`add_time_bins` adds, named here so a consumer asks for them by name rather
#: than by spelling them again.
HOURS_COLUMN = "hours_before_delivery"
BIN_COLUMN = "time_bin"
BIN_CENTER_COLUMN = "time_bin_center_h"

#: The per-segment field the **second** clinical clock is read off, and the columns
#: :func:`add_second_stage_bins` derives from it. The shard stores
#: $\texttt{second\_stage\_onset} = \texttt{domain\_start} - t_{\mathrm{SSO}}$ -- already signed the
#: way an axis wants it, negative before second-stage onset and positive after -- so unlike
#: ``epoch`` it is **not** negated. That asymmetry between the two clocks is the single most likely
#: thing to get wrong here, and it is pinned by a known-answer test rather than by this comment.
SECOND_STAGE_COLUMN = "second_stage_onset"
SECOND_STAGE_HOURS_COLUMN = "hours_from_second_stage"
SECOND_STAGE_BIN_COLUMN = "second_stage_bin"
SECOND_STAGE_BIN_CENTER_COLUMN = "second_stage_bin_center_h"

#: The two coupling readouts a clinical clock resolves, as
#: ``(reported name, per-sample column, what it is)``. **Both** travel rather than the KL alone,
#: because the two fail differently: ``pred_gap`` is in the decoder's own units and is immune to
#: the prior-variance inflation, while ``source_conditioned_kl_raw`` is multiplied by an arbitrary
#: factor whenever the prior variance sits on its clamp. A trajectory visible in one and absent
#: from the other is a finding about which of the two is being read, and a run tracking only the
#: KL cannot see it.
#:
#: They live here, beside the axes, rather than in the analysis that needed them first, and for the
#: same reason :data:`TRAJECTORY_BIN_HOURS` does: **both** clocks resolve exactly these two
#: quantities, an analysis may never import another, and a second copy of this tuple would be a
#: second answer to "what is drawn against a clinical clock" with nothing keeping the two equal.
CLOCK_READOUTS: Tuple[Tuple[str, str, str], ...] = (
    (
        "pred_gap_mc_nats",
        "mc_pred_gap",
        "Monte Carlo marginalised D_base - D_full in nats per anchor; in the decoder's own units "
        "and immune to the prior-variance inflation",
    ),
    (
        "source_conditioned_kl_raw_nats",
        "source_conditioned_kl_raw",
        "the unfloored KL between the two latents; inflated by an arbitrary factor whenever the "
        "prior variance sits on its clamp, which is why it is not read alone",
    ),
)

#: The per-sample columns those readouts are reduced from, in the order the tables carry them.
CLOCK_VALUE_COLUMNS: Tuple[str, ...] = tuple(column for _, column, _ in CLOCK_READOUTS)

#: How far a recording's implied onset may vary across its own segments before it is reported
#: inconsistent, in seconds. Both operands are stored ``float32`` at magnitudes around
#: $4 \times 10^{4}$ s, whose representable spacing is about $4 \times 10^{-3}$ s, so an exact
#: comparison would report inconsistency on arithmetic alone.
ONSET_CONSISTENCY_TOLERANCE_S = 1.0

#: The subgroups the pretraining split is drawn from -- see the module docstring for the source.
#: Everything else among the canonical eight is unseen by a checkpoint trained on it, which is the
#: scope of the out-of-distribution statement below and is wider than "acidosis and HIE".
PRETRAINING_SUBGROUPS: Sequence[str] = ("healthy_bg_cs", "healthy_bg_no_cs")

#: Written into every summary. The two populations are different and neither number says so, so a
#: run states it rather than leaving a reader to discover the comparison is meaningless.
NON_COMPARABILITY_SENTENCE = (
    "a forecast score from this evaluation is not comparable with a test_* metric logged during "
    "training: they are computed over different populations, and nothing in either number says so"
)

#: Emitted only when the two cohorts are disjoint, because it is a consequence of that and of
#: nothing else. Its subgroup scope is filled in per run from what the pretraining split actually
#: covers.
OUT_OF_DISTRIBUTION_SENTENCE = (
    "the evaluation cohort is disjoint from the training cohort, so this is a leakage-free "
    "evaluation -- and the checkpoint has never seen the unseen subgroups listed beside this "
    "sentence, so every class contrast is an out-of-distribution comparison rather than held-out "
    "clinical discrimination"
)


# =============================================================================
# The time axis
# =============================================================================
def add_time_bins(
    frame: pd.DataFrame, *, width: float = TRAJECTORY_BIN_HOURS
) -> pd.DataFrame:
    r"""Add the time-before-delivery coordinate and its bin, dropping rows that have neither.

    Args:
        frame: A per-sample table carrying ``epoch``.
        width: Bin width in hours.

    Returns:
        A copy holding only the rows with a finite ``epoch``, with :data:`HOURS_COLUMN`,
        :data:`BIN_COLUMN` and :data:`BIN_CENTER_COLUMN` added. Empty -- with those columns
        present, so a caller can still group on them -- when the frame carries no usable ``epoch``.
    """
    def _empty() -> pd.DataFrame:
        """The frame's own schema with the three added columns and no rows."""
        return pd.DataFrame(
            {
                **{name: frame[name].iloc[:0] for name in getattr(frame, "columns", [])},
                HOURS_COLUMN: pd.Series(dtype=np.float64),
                BIN_COLUMN: pd.Series(dtype=np.int64),
                BIN_CENTER_COLUMN: pd.Series(dtype=np.float64),
            }
        )

    if frame.empty or "epoch" not in frame.columns:
        return _empty()

    epochs = np.asarray(frame["epoch"], dtype=np.float64)
    usable = frame[np.isfinite(epochs)].copy()
    if usable.empty:
        return _empty()

    # Negative seconds before delivery, so the sign flips to give a non-negative coordinate. A
    # segment at or after delivery lands at or below zero; the clip keeps it in the first bin
    # rather than producing a negative index that would sort before every real window.
    hours = -np.asarray(usable["epoch"], dtype=np.float64) / SECONDS_PER_HOUR
    index = np.clip(np.floor(hours / float(width)).astype(np.int64), 0, None)
    usable[HOURS_COLUMN] = hours
    usable[BIN_COLUMN] = index
    usable[BIN_CENTER_COLUMN] = (index + 0.5) * float(width)
    return usable


def add_second_stage_bins(
    frame: pd.DataFrame, *, width: float = TRAJECTORY_BIN_HOURS
) -> pd.DataFrame:
    r"""Add the second clinical clock's coordinate and its bin, dropping rows that have neither.

    The mirror of :func:`add_time_bins`, and it differs from it in exactly two places -- both of
    which are the difference between the two clocks rather than an inconsistency:

    * **No sign flip.** :data:`SECOND_STAGE_COLUMN` is stored as
      $\texttt{domain\_start} - t_{\mathrm{SSO}}$, so $h = \texttt{second\_stage\_onset} / 3600$ is
      already negative before onset and positive after it. ``epoch`` is negated because it is
      stored as time *before* delivery; negating this one as well would run every trajectory
      backwards through the second stage with nothing raising.
    * **No clip at zero.** The delivery axis clips because a segment at or after delivery has no
      meaningful window; here the positive side *is* half the content, so the bin index is signed
      and a window before onset sorts before one after it.

    Args:
        frame: A per-sample table carrying :data:`SECOND_STAGE_COLUMN`.
        width: Bin width in hours -- the same grid the delivery clock uses, so a window on one
            figure is the same duration as a window on the other.

    Returns:
        A copy holding only the rows with a finite offset, with :data:`SECOND_STAGE_HOURS_COLUMN`,
        :data:`SECOND_STAGE_BIN_COLUMN` and :data:`SECOND_STAGE_BIN_CENTER_COLUMN` added. Empty --
        with those columns present, so a caller can still group on them -- when the frame carries
        no usable offset, which is the ordinary outcome on a cohort with no onset table.
    """
    def _empty() -> pd.DataFrame:
        """The frame's own schema with the three added columns and no rows."""
        return pd.DataFrame(
            {
                **{name: frame[name].iloc[:0] for name in getattr(frame, "columns", [])},
                SECOND_STAGE_HOURS_COLUMN: pd.Series(dtype=np.float64),
                SECOND_STAGE_BIN_COLUMN: pd.Series(dtype=np.int64),
                SECOND_STAGE_BIN_CENTER_COLUMN: pd.Series(dtype=np.float64),
            }
        )

    if frame.empty or SECOND_STAGE_COLUMN not in frame.columns:
        return _empty()

    offsets = np.asarray(frame[SECOND_STAGE_COLUMN], dtype=np.float64)
    usable = frame[np.isfinite(offsets)].copy()
    if usable.empty:
        return _empty()

    hours = np.asarray(usable[SECOND_STAGE_COLUMN], dtype=np.float64) / SECONDS_PER_HOUR
    index = np.floor(hours / float(width)).astype(np.int64)
    usable[SECOND_STAGE_HOURS_COLUMN] = hours
    usable[SECOND_STAGE_BIN_COLUMN] = index
    usable[SECOND_STAGE_BIN_CENTER_COLUMN] = (index + 0.5) * float(width)
    return usable


def per_recording_in_bins(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    group_column: str,
    bin_column: str = BIN_COLUMN,
    center_column: str = BIN_CENTER_COLUMN,
) -> pd.DataFrame:
    """Reduce a binned per-sample table to one row per (cohort, window, recording).

    This is the aggregation chain's middle step applied *inside* a window: a recording's value in
    a window is the mean over its own segments in that window, and everything downstream then has
    one value per recording rather than one per segment.

    Args:
        frame: A binned per-sample table, as :func:`add_time_bins` or
            :func:`add_second_stage_bins` returns it.
        columns: The value columns to reduce. Names absent from the frame are skipped.
        group_column: The cohort axis. Rows with no label on it are dropped -- a segment with no
            cohort belongs to no trajectory.
        bin_column: The window index to group on, defaulting to the delivery clock's. The second
            clock passes its own; the reduction itself is the same arithmetic on either, and one
            implementation of it is what keeps the two clocks' tables comparable.
        center_column: The window centre travelling with it.

    Returns:
        One row per ``(group, window, guid)`` with the reduced columns and the window's centre.
        Empty with the key columns present when nothing is usable.
    """
    present = [name for name in columns if name in getattr(frame, "columns", [])]
    keys = ["group", bin_column, center_column, "guid"]
    if frame.empty or not present or group_column not in frame.columns or "guid" not in frame:
        return pd.DataFrame(columns=keys + present)

    labelled = frame[frame[group_column].notna()].copy()
    if labelled.empty:
        return pd.DataFrame(columns=keys + present)
    labelled["group"] = labelled[group_column].astype(str)
    reduced = (
        labelled.groupby(["group", bin_column, center_column, "guid"], sort=True)[present]
        .mean()
        .reset_index()
    )
    return reduced


def trajectory_rows(
    per_recording: pd.DataFrame,
    column: str,
    *,
    metric: str,
    bin_column: str = BIN_COLUMN,
    center_column: str = BIN_CENTER_COLUMN,
) -> List[Dict[str, Any]]:
    """Summarise one metric within each (cohort, window) cell, over its recordings.

    Args:
        per_recording: The per-recording-per-window frame from :func:`per_recording_in_bins`.
        column: The value column to summarise.
        metric: The name the rows are reported under.
        bin_column: The window index to group on, defaulting to the delivery clock's.
        center_column: The window centre travelling with it. The emitted keys stay ``time_bin``
            and ``bin_center_h`` on either clock: they name a window and its centre, which is what
            they are on both, and renaming them per clock would fork every consumer of this table.

    Returns:
        One row per non-empty cell: the cohort, the window and its centre, the count of
        **recordings** behind it, the mean and the quartiles. Quartiles rather than a standard
        deviation because these distributions are skewed, matching the grouped-variant convention.
    """
    rows: List[Dict[str, Any]] = []
    if per_recording.empty or column not in per_recording.columns:
        return rows
    for (group, bin_index, centre), cell in per_recording.groupby(
        ["group", bin_column, center_column], sort=True
    ):
        values = np.asarray(cell[column], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "group": str(group),
                "time_bin": int(bin_index),
                "bin_center_h": float(centre),
                # Recordings, not segments: the unit every statistic on this table is computed on.
                "n_recordings": int(values.size),
                "mean": float(values.mean()),
                "q25": float(np.percentile(values, 25)),
                "median": float(np.percentile(values, 50)),
                "q75": float(np.percentile(values, 75)),
            }
        )
    return rows


def ordered_groups(groups: Sequence[str], axis: str) -> List[str]:
    """Return cohort labels in a stable, human-meaningful order.

    **This is the evaluation's one cohort order**, and every table and figure that resolves a
    quantity by cohort reads it: the grouped violin fan-out, the trajectories, the significance
    tables, the stratified lag profiles and the conditioned-coupling violins alike. Alphabetical
    is the default everywhere it is not called, and alphabetical is wrong in a specific way -- it
    puts ``acidosis`` before ``healthy`` on every class figure, and on the subgroup axis it
    interleaves the three classes (``acidosis_cs``, ``acidosis_no_cs``, ``healthy_bg_cs``, ...) so
    that neither the severity ordering nor the background/caesarean structure is visible.

    The order is severity-ascending on both axes, which is also the order the two source tables
    already carry: :data:`~teb_vae.lag_attn.eval.labels.CLASS_NAMES` is keyed by the dataset's own
    class codes $1, 2, 3$, and :data:`~teb_vae.lag_attn.eval.labels.CANONICAL_SUBGROUPS` is
    written in the intended order. Neither is restated here, so a subgroup added to the dataset
    appears in the figures without an edit to this module.

    Args:
        groups: The labels present.
        axis: The cohort axis, choosing the canonical order. An axis that is neither the class nor
            the subgroup column -- ``lag_kl``'s time-window axis is the one such caller -- matches
            nothing and falls through to the alphabetical order, which is the previous behaviour
            rather than a new one.

    Returns:
        Classes in healthy / acidosis / HIE order, subgroups in canonical order, with anything
        unrecognised appended alphabetically so nothing is silently dropped.
    """
    present = {str(group) for group in groups}
    preferred = (
        [labels.CLASS_NAMES[code] for code in sorted(labels.CLASS_NAMES)]
        if axis == labels.CLASS_COLUMN
        else list(labels.CANONICAL_SUBGROUPS)
    )
    ordered = [name for name in preferred if name in present]
    return ordered + sorted(present - set(ordered))


# =============================================================================
# The population and its provenance
# =============================================================================
def _resolved_paths(values: Optional[Sequence[Any]]) -> List[str]:
    """Return dataset paths in a form two lists can be compared on.

    Args:
        values: A configured dataset list, or ``None``.

    Returns:
        The absolute, case-normalised paths. Normalised because a train list written with forward
        slashes and a test list written with backslashes name the same file on this platform, and
        a string comparison would call them disjoint.
    """
    return [os.path.normcase(os.path.abspath(str(value))) for value in (values or [])]


def cohort_counts(frame: pd.DataFrame, column: str) -> Dict[str, Dict[str, int]]:
    """Count segments and recordings per cohort on one axis.

    Both levels, because they answer different questions and routinely disagree: a subgroup with
    many segments and three recordings is one whose statistics have $n = 3$, and only the second
    count says so.

    Args:
        frame: The per-sample table.
        column: The cohort axis.

    Returns:
        ``{'segments': {...}, 'recordings': {...}}``, skipping unlabelled rows. Both are keyed in
        the canonical cohort order rather than by descending count, so the population block of a
        summary reads in the same order as the figures beside it.
    """
    if frame.empty or column not in frame.columns:
        return {"segments": {}, "recordings": {}}
    labelled = frame[frame[column].notna()]
    segments = labelled[column].astype(str).value_counts()
    recordings = (
        labelled.groupby(labelled[column].astype(str))["guid"].nunique()
        if "guid" in labelled.columns
        else segments.iloc[:0]
    )
    counted = {
        "segments": {str(name): int(count) for name, count in segments.items()},
        "recordings": {str(name): int(count) for name, count in recordings.items()},
    }
    return {
        level: {name: table[name] for name in ordered_groups(list(table), column)}
        for level, table in counted.items()
    }


def labor_onset_readout(frame: pd.DataFrame) -> Dict[str, Any]:
    """Report ``time_from_labor_onset`` with the rows it is missing on counted, not dropped.

    The field is NaN wherever the recording is absent from the labour-onset table, and that is a
    fact about the cohort rather than a defect in a row: a reader needs the denominator before
    reading any labour-onset number, and a summary that quietly dropped those rows would report a
    mean over a population it does not name.

    Args:
        frame: The per-sample table.

    Returns:
        The counts, the fraction missing, and the range in hours over the rows that have one.
        ``present: False`` when the column is absent entirely.
    """
    if "time_from_labor_onset" not in getattr(frame, "columns", []):
        return {"present": False, "n_rows": int(len(frame))}
    values = np.asarray(frame["time_from_labor_onset"], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "present": True,
        "n_rows": int(values.size),
        "n_finite": int(finite.size),
        # Counted rather than dropped: it is the denominator of every labour-onset statement.
        "n_nan": int(values.size - finite.size),
        "nan_fraction": float((values.size - finite.size) / values.size) if values.size else float("nan"),
        "min_hours": float(finite.min() / SECONDS_PER_HOUR) if finite.size else float("nan"),
        "max_hours": float(finite.max() / SECONDS_PER_HOUR) if finite.size else float("nan"),
        "mean_hours": float(finite.mean() / SECONDS_PER_HOUR) if finite.size else float("nan"),
    }


def second_stage_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    r"""Say, recording by recording, whether it can be placed on the second-stage clock.

    **One recording is dropped for one reason only: it has no onset.** That is the policy, and the
    two further ways a stored onset can be wrong are *counted here and filtered nowhere*, because
    excluding a recording changes the population every number is computed over and a count does
    not:

    * ``onset_at_delivery`` -- the implied onset falls at delivery itself, which is what a
      pipeline writes when it substitutes zero for a missing time. The sibling classifier
      pipeline was burned by exactly this: such recordings pass a NaN filter and then place a
      whole labour at twelve hours before a second stage that never happened.
    * ``inconsistent_onset`` -- the implied onset moves across a recording's own segments by more
      than :data:`ONSET_CONSISTENCY_TOLERANCE_S`. It is a property of the *recording*, so it
      cannot legitimately move at all; when it does, the field was written wrong.

    Both come free from an identity the stored field satisfies by construction:

    $$t^{\mathrm{onset}}_{\mathrm{epoch}} = \texttt{epoch} - \texttt{second\_stage\_onset}$$

    Args:
        frame: A per-sample table carrying ``guid``, ``epoch`` and :data:`SECOND_STAGE_COLUMN`.

    Returns:
        One row per recording: how many segments it contributed, how many carried a finite offset,
        the implied onset and its spread, whether it is eligible, and the reason when it is not.
        Empty with those columns present when the table carries none of what it needs.
    """
    columns = [
        "guid", "n_segments", "n_finite", "implied_onset_epoch_s", "onset_spread_s",
        "eligible", "reason", "onset_at_delivery", "inconsistent_onset",
    ]
    if (
        frame.empty
        or SECOND_STAGE_COLUMN not in getattr(frame, "columns", [])
        or "guid" not in getattr(frame, "columns", [])
    ):
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    for guid, cell in frame.groupby(frame["guid"].astype(str), sort=True):
        offsets = np.asarray(cell[SECOND_STAGE_COLUMN], dtype=np.float64)
        finite = offsets[np.isfinite(offsets)]
        epochs = (
            np.asarray(cell["epoch"], dtype=np.float64)
            if "epoch" in cell.columns
            else np.full(offsets.shape, np.nan)
        )
        implied = epochs - offsets
        implied = implied[np.isfinite(implied)]
        eligible = bool(offsets.size) and finite.size == offsets.size
        spread = float(implied.max() - implied.min()) if implied.size else float("nan")
        onset = float(implied.mean()) if implied.size else float("nan")
        rows.append(
            {
                "guid": str(guid),
                "n_segments": int(offsets.size),
                "n_finite": int(finite.size),
                "implied_onset_epoch_s": onset,
                "onset_spread_s": spread,
                "eligible": eligible,
                "reason": "" if eligible else "no second-stage onset recorded",
                # Counted, never filtered -- see the docstring.
                "onset_at_delivery": bool(implied.size and np.all(implied == 0.0)),
                "inconsistent_onset": bool(
                    np.isfinite(spread) and spread > ONSET_CONSISTENCY_TOLERANCE_S
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def second_stage_readout(frame: pd.DataFrame) -> Dict[str, Any]:
    """Report the second clock's coverage with the rows it is missing on counted, not dropped.

    The mirror of :func:`labor_onset_readout`, and it exists for the same reason: a reader needs
    the denominator before reading any second-stage number, and a summary that quietly dropped the
    recordings without an onset would report a trajectory over a population it does not name. It
    carries the two diagnostics beside it, so a cohort whose onsets were written wrong is visible
    in the summary rather than only in a figure that looks odd.

    Args:
        frame: The per-sample table.

    Returns:
        The counts, the fraction missing, the range in signed hours over the rows that have one,
        and the per-recording eligibility tally. ``present: False`` when the column is absent
        entirely.
    """
    if SECOND_STAGE_COLUMN not in getattr(frame, "columns", []):
        return {"present": False, "n_rows": int(len(frame))}

    values = np.asarray(frame[SECOND_STAGE_COLUMN], dtype=np.float64)
    finite = values[np.isfinite(values)]
    eligibility = second_stage_eligibility(frame)
    return {
        "present": True,
        "n_rows": int(values.size),
        "n_finite": int(finite.size),
        # Counted rather than dropped: it is the denominator of every second-stage statement.
        "n_nan": int(values.size - finite.size),
        "nan_fraction": (
            float((values.size - finite.size) / values.size) if values.size else float("nan")
        ),
        # Signed, and reported as such: negative is before onset and positive after it.
        "min_hours": float(finite.min() / SECONDS_PER_HOUR) if finite.size else float("nan"),
        "max_hours": float(finite.max() / SECONDS_PER_HOUR) if finite.size else float("nan"),
        "median_hours": (
            float(np.median(finite) / SECONDS_PER_HOUR) if finite.size else float("nan")
        ),
        "n_recordings": int(len(eligibility)),
        "n_recordings_eligible": int(eligibility["eligible"].sum()) if len(eligibility) else 0,
        "n_recordings_missing": (
            int((~eligibility["eligible"]).sum()) if len(eligibility) else 0
        ),
        # Reported, not excluded. See second_stage_eligibility for why.
        "n_recordings_onset_at_delivery": (
            int(eligibility["onset_at_delivery"].sum()) if len(eligibility) else 0
        ),
        "n_recordings_inconsistent_onset": (
            int(eligibility["inconsistent_onset"].sum()) if len(eligibility) else 0
        ),
        "onset_consistency_tolerance_s": float(ONSET_CONSISTENCY_TOLERANCE_S),
    }


def build_cohort_block(
    per_sample: pd.DataFrame, config: Dict[str, Any], probe: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Assemble the cohort record every summary carries.

    Args:
        per_sample: The per-sample table, for the composition and the labour-onset readout.
        config: The merged run configuration, read for both resolved dataset lists -- the training
            one survives the override merge from the checkpoint's own resolved config, and the
            evaluation one is what the delta repointed.
        probe: The loader probe's record, for its per-file counts. Optional: the composition below
            comes from the table, and the probe adds only the loader's own view of it.

    Returns:
        The shard lists, the per-cohort counts at both levels, the computed disjointness flag with
        the unseen subgroups behind it, the two statements, and the labour-onset readout.
    """
    dataset = config.get("dataset_config") or {}
    train = _resolved_paths(dataset.get("vae_train_datasets"))
    test = _resolved_paths(dataset.get("vae_test_datasets"))
    overlap = sorted(set(train) & set(test))
    # Only when both lists are known: a run whose config named no training set cannot claim
    # disjointness, and False there would read as "they overlap".
    disjoint = None if not train or not test else not overlap

    evaluated_subgroups = sorted(cohort_counts(per_sample, labels.SUBGROUP_COLUMN)["segments"])
    unseen = [
        name for name in labels.CANONICAL_SUBGROUPS if name not in PRETRAINING_SUBGROUPS
    ]
    block: Dict[str, Any] = {
        "vae_train_datasets": list(dataset.get("vae_train_datasets") or []),
        "vae_test_datasets": list(dataset.get("vae_test_datasets") or []),
        "n_segments": int(len(per_sample)),
        "n_recordings": (
            int(per_sample["guid"].nunique()) if "guid" in getattr(per_sample, "columns", []) else 0
        ),
        "by_clinical_class": cohort_counts(per_sample, labels.CLASS_COLUMN),
        "by_subgroup": cohort_counts(per_sample, labels.SUBGROUP_COLUMN),
        "per_file": dict((probe or {}).get("per_file") or {}),
        "training_cohort_disjoint": disjoint,
        "training_cohort_overlap": overlap,
        # Which of the canonical eight the pretraining split never covered, and which of those
        # this run actually evaluated. Both, because the first is a property of the training data
        # and the second is what the out-of-distribution statement applies to here.
        "pretraining_subgroups": list(PRETRAINING_SUBGROUPS),
        "unseen_subgroups": unseen,
        "unseen_subgroups_evaluated": [
            name for name in evaluated_subgroups if name in unseen
        ],
        "non_comparability": NON_COMPARABILITY_SENTENCE,
    }
    if disjoint:
        block["out_of_distribution"] = OUT_OF_DISTRIBUTION_SENTENCE
    block["time_from_labor_onset"] = labor_onset_readout(per_sample)
    block[SECOND_STAGE_COLUMN] = second_stage_readout(per_sample)
    return block


__all__ = [
    "BIN_CENTER_COLUMN",
    "BIN_COLUMN",
    "CLOCK_READOUTS",
    "CLOCK_VALUE_COLUMNS",
    "HOURS_COLUMN",
    "NON_COMPARABILITY_SENTENCE",
    "ONSET_CONSISTENCY_TOLERANCE_S",
    "OUT_OF_DISTRIBUTION_SENTENCE",
    "PRETRAINING_SUBGROUPS",
    "SECONDS_PER_HOUR",
    "SECOND_STAGE_BIN_CENTER_COLUMN",
    "SECOND_STAGE_BIN_COLUMN",
    "SECOND_STAGE_COLUMN",
    "SECOND_STAGE_HOURS_COLUMN",
    "TRAJECTORY_BIN_HOURS",
    "add_second_stage_bins",
    "add_time_bins",
    "build_cohort_block",
    "cohort_counts",
    "labor_onset_readout",
    "ordered_groups",
    "per_recording_in_bins",
    "second_stage_eligibility",
    "second_stage_readout",
    "trajectory_rows",
]
