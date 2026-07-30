r"""The aggregation chain, written once, for every analysis that reads the per-sample table.

$$
\text{per anchor} \;\longrightarrow\;
\underbrace{\text{support-weighted mean}}_{\text{within a segment}} \;\longrightarrow\;
\underbrace{\text{unweighted mean}}_{\text{over a recording's segments}} \;\longrightarrow\;
\underbrace{\text{across recordings}}_{\text{the headline}}
$$

The first arrow happens in the collection pass, where the anchors are. This module owns the
second: turning ``per_sample.csv`` into one row per recording, which is the unit every statistic
downstream is computed on. It is a layer below ``analyses/`` because five of them need it and an
analysis may not import another -- and because the chain is a rule of this pipeline rather than a
detail of whichever analysis happened to need it first.

Two properties are worth stating, because both are choices:

* **The per-recording step is unweighted.** A recording contributing thirty-seven segments and one
  contributing two count the same. That is the unit the clinical question is asked in; the
  anchor-count weighted variant is recoverable offline from ``n_anchors``, which every row carries
  for exactly that reason.
* **``NaN`` is skipped, not imputed.** A segment that scored no anchors measured nothing, and the
  collection pass writes that as ``NaN`` rather than as the ``0.0`` an empty numerator over a
  clamped denominator produces. ``mean`` skips it, so the denominator every statistic reports is
  the count of segments that actually measured something -- which is why that count travels beside
  every number here rather than being inferred from the frame's length.

**The cohort a recording belongs to travels with its row.** A clinical class and a subgroup are
properties of the *recording*, not of the question being asked, so every per-recording frame this
module builds carries both -- which is what lets a by-class or by-subgroup variant be a ``groupby``
on an existing column rather than a second reduction with its own chance of using a different
unit. A recording whose segments disagree about either label carries **no** label rather than the
first or the commonest one: it means the recording appears in two shards, which is a fault the
loader probe raises on, and inventing a cohort for it here would hide that.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.eval._reuse import labels

#: Quantiles reported beside a mean. A mean and an interval describe a symmetric distribution;
#: these readouts are routinely skewed -- a handful of recordings sit far from the rest -- and the
#: quartiles are what say so.
QUANTILES = (0.25, 0.5, 0.75)

#: The cohort axes a grouped variant is cut on, in the order they are emitted. Bound from the
#: shared labelling module rather than restated, so the column a frame carries and the column the
#: grouped emitter looks for are one name.
GROUP_COLUMNS: Sequence[str] = tuple(labels.GROUP_COLUMNS)


def per_recording_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Return each recording's cohort labels, one row per ``guid``.

    A clinical class and a subgroup belong to the recording rather than to the segment, so a
    recording resolves to one value of each. Where its segments disagree the recording carries
    ``None`` on that axis: two segments of one recording in two different shards is the duplicated
    GUID the loader probe raises on, and choosing the first or the commonest label here would
    replace that fault with a plausible answer.

    Args:
        frame: The per-sample table.

    Returns:
        One row per recording, indexed by ``guid``, with one column per axis of
        :data:`GROUP_COLUMNS` the frame carries. Empty with those columns present when the frame
        carries no ``guid``.
    """
    present = [name for name in GROUP_COLUMNS if name in frame.columns]
    if frame.empty or "guid" not in frame.columns or not present:
        return pd.DataFrame(columns=list(present))
    resolved = frame.groupby("guid")[present].agg(_single_label)
    return resolved


def _single_label(values: Any) -> Optional[str]:
    """Return the one label a recording's segments agree on, or ``None``.

    Args:
        values: One recording's values on one axis, possibly holding ``None`` or ``NaN``.

    Returns:
        The single distinct non-null value, or ``None`` when there is none or more than one.
    """
    distinct = {
        str(value)
        for value in values
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    }
    return distinct.pop() if len(distinct) == 1 else None


def per_recording_means(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Average each column within a recording -- the middle step of the aggregation chain.

    The cohort labels come along, because they are what every by-class and by-subgroup variant
    resolves on and they are a property of the recording rather than of the reduction; see
    :func:`per_recording_labels` for what a recording whose segments disagree carries.

    Args:
        frame: The per-sample table.
        columns: The value columns to reduce. Names absent from the frame are skipped, so a caller
            may name a column the pass did not produce.

    Returns:
        One row per recording, indexed by ``guid``, carrying the reduced columns, the cohort
        labels and an ``n_segments`` count. Empty with the requested columns present when the
        frame carries no ``guid`` or none of the columns -- an empty frame a caller can still
        index is friendlier than one with no schema.
    """
    present = [name for name in columns if name in frame.columns]
    label_columns = [name for name in GROUP_COLUMNS if name in getattr(frame, "columns", [])]
    if frame.empty or "guid" not in frame.columns or not present:
        return pd.DataFrame(columns=list(present) + label_columns + ["n_segments"])
    grouped = frame.groupby("guid")
    means = grouped[present].mean()
    for name, column in per_recording_labels(frame).items():
        means[name] = column
    means["n_segments"] = grouped.size()
    return means


def grouped_frame_entry(
    analysis_dirname: str,
    filename: str,
    value_columns: Sequence[str],
    *,
    stem: Optional[str] = None,
) -> Dict[str, Any]:
    """Declare one written frame for the runner's by-class and by-subgroup fan-out.

    An analysis *names* a frame it has already written and the columns worth resolving by cohort;
    the emission itself is the runner's, so an analysis added later gets both variants without
    anyone remembering to add them. The frame is named rather than returned because the return
    value is serialised into ``summary.json``.

    The path is **relative to the results directory** for the same reason every ``files`` entry is
    a bare filename: an absolute path in the summary is a machine-specific string in a block two
    runs of one checkpoint must compare equal, and it stops resolving the moment the directory is
    copied anywhere.

    Args:
        analysis_dirname: The analysis's own subdirectory inside the results directory.
        filename: The CSV written into it.
        value_columns: The metrics to resolve by cohort. A name absent from the frame is skipped
            by the emitter rather than raising.
        stem: Filename stem for the variants, yielding ``<stem>_by_<axis>.{csv,pdf}``. Defaults to
            the frame's own stem.

    Returns:
        The declaration, as the runner's fan-out reads it.
    """
    return {
        "directory": str(analysis_dirname),
        "path": f"{analysis_dirname}/{filename}",
        "stem": str(stem or Path(filename).stem),
        "value_columns": [str(name) for name in value_columns],
    }


def finite_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Return one column as a float array, or all-``NaN`` when the frame does not carry it.

    Args:
        frame: Any frame.
        name: The column to read.

    Returns:
        The values as ``float64``. A missing column yields NaNs of the right length rather than
        raising, so a readout absent from an older run's tables reports as unmeasured rather than
        taking down the analysis that wanted it.
    """
    if name not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return np.asarray(frame[name], dtype=np.float64)


def describe(values: Any, *, name: str = "") -> Dict[str, Any]:
    """Summarise a per-recording vector by its mean, its quartiles and its honest $n$.

    Args:
        values: One value per recording.
        name: Optional label carried into the record, so a table of these reads without a key.

    Returns:
        The count of finite values, the mean, the quartiles, and the extremes. Every statistic is
        ``NaN`` when nothing is finite -- never ``0.0``, which reads as a measurement.
    """
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                       dtype=np.float64)
    finite = array[np.isfinite(array)]
    record: Dict[str, Any] = {
        "n": int(finite.size),
        "n_dropped": int(array.size - finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
    }
    if name:
        record = {"metric": name, **record}
    for quantile in QUANTILES:
        record[f"q{int(quantile * 100):02d}"] = (
            float(np.quantile(finite, quantile)) if finite.size else float("nan")
        )
    return record


def positive_fraction(values: Any) -> Dict[str, Any]:
    r"""The fraction of recordings on which a quantity is positive, with its denominator.

    The denominator is the point. ``np.nan > 0`` is ``False``, so a recording that scored no
    anchors -- and therefore measured nothing -- would otherwise be counted silently as evidence
    *against*, and a run whose coverage collapsed would report a falling positive fraction rather
    than a falling $n$.

    Args:
        values: One value per recording.

    Returns:
        The fraction, the numerator, the denominator, and how many recordings were dropped for
        carrying no finite value. The fraction is ``NaN`` when nothing is finite.
    """
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                       dtype=np.float64)
    finite = array[np.isfinite(array)]
    n_positive = int((finite > 0.0).sum())
    return {
        "fraction": (n_positive / finite.size) if finite.size else float("nan"),
        "n_positive": n_positive,
        "n": int(finite.size),
        "n_dropped_not_finite": int(array.size - finite.size),
    }


def scored_sample_count(frame: pd.DataFrame, column: str) -> Optional[int]:
    """How many segments actually contributed to a column.

    Args:
        frame: The per-sample table.
        column: The column whose finite rows define the population.

    Returns:
        The count, or ``None`` when the table does not carry the column at all -- which the
        coverage block reads as "this analysis describes something other than a population"
        rather than as a disagreement with every analysis that does.
    """
    if column not in frame.columns:
        return None
    return int(frame[column].notna().sum())
