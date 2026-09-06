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

**Every value on this chain is in the loader's $z$ units, and there is no conversion anywhere.**
The forecast target here is $98$ wavelet-modulus and phase-harmonic coefficients, which have no
clinical unit to convert to, so the raw pipeline's ``to_bpm`` has no analogue here and is removed
rather than repointed. Inverting the per-channel statistics instead would put the $C_{\mathrm{keep}}$ channels on
scales spanning orders of magnitude, which destroys every pooled statistic this module computes --
the mean, the quartiles and the positive fraction alike.

**The chain reduces *unrooted* quantities, and that is load-bearing rather than incidental.** An
RMS is the square root of a mean, and by Jensen the average of per-segment roots is biased
**low** -- in the direction that flatters the model. So the collection pass accumulates squares,
:func:`per_recording_means` averages squares, and the root is taken once at the end by the
analysis that reports it. A reduction here that rooted first would be invisible in every shape and
wrong in every number.

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

from teb_vae.lag_attn_cfs.eval._reuse import labels

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


#: The per-sample column whose magnitude every ``pred_gap`` recomposition is checked against, and
#: the tolerance it is checked at.
#:
#: **The denominator is the block score, not the gap, and that is the whole design of the check.**
#: ``pred_gap`` is a *difference* of two block scores of order $10^{3}$, so the float32 accumulation
#: error it inherits is a property of those scores rather than of the small number between them.
#: Dividing the residual by the gap would make the tolerance tighten without limit exactly as a
#: model improves -- a run whose gap approached zero could not satisfy any relative tolerance at
#: all -- and would report a healthy decomposition as broken on the very runs that matter most.
#: This is the same rule ``tests/test_eval_metrics.py`` states the per-sample identities at, so the
#: per-sample and the per-recording forms of one property are checked the same way.
RECOMPOSITION_SCALE_COLUMN = "nll_base_block"
RECOMPOSITION_RTOL = 1e-6

#: Absolute floor under that tolerance, for a run whose block score is legitimately near zero.
RECOMPOSITION_ATOL = 1e-9


def recomposition_check(
    per_guid: pd.DataFrame,
    parts: Sequence[str],
    total: str,
    *,
    identity: str,
    rtol: float = RECOMPOSITION_RTOL,
    atol: float = RECOMPOSITION_ATOL,
) -> Dict[str, Any]:
    r"""Whether a channel-axis split sums back to the quantity it decomposes, on the worst recording.

    Every split of the forecast gap over the kept target channels -- by warm-up tertile, by stored
    block, by frequency band -- is only a *decomposition* if its parts recompose. This is the one
    check that says so, and it lives here rather than in whichever analysis needed it first because
    two of them need it now and an analysis may not import another.

    **The worst recording rather than the mean of the residuals.** The mechanism that would break
    any of these splits -- a partition that stopped tiling the channel axis -- moves gap *between*
    the parts, so the per-recording error is zero-mean by construction and a mean would report
    nothing.

    Args:
        per_guid: Per-recording means carrying the parts, the total and, ideally, the block score
            of :data:`RECOMPOSITION_SCALE_COLUMN`.
        parts: The columns that must sum to ``total``.
        total: The column they decompose.
        identity: The identity in words, carried into the record so it reads without this source.
        rtol: Relative tolerance, applied to the **block score** magnitude; see
            :data:`RECOMPOSITION_SCALE_COLUMN` for why not to the total.
        atol: Absolute floor under it.

    Returns:
        The largest absolute residual over recordings, the largest relative to the scale, the count
        behind them, and whether the identity holds. ``holds`` is ``None`` when nothing finite was
        available -- never ``True``, which would read as a checked identity.
    """
    recomposed = np.zeros(len(per_guid), dtype=np.float64)
    for column in parts:
        recomposed = recomposed + finite_column(per_guid, column)
    expected = finite_column(per_guid, total)
    residual = recomposed - expected

    # The block score where the frame carries it, the total itself where it does not: an older
    # run's tables may not have the column, and the total is the only other magnitude available.
    scale = np.abs(finite_column(per_guid, RECOMPOSITION_SCALE_COLUMN))
    scale = np.where(np.isfinite(scale), scale, np.abs(expected))
    usable = np.isfinite(residual) & np.isfinite(expected)
    record: Dict[str, Any] = {
        "identity": str(identity),
        "n_recordings": int(usable.sum()),
        "rtol": float(rtol),
        "atol": float(atol),
        "scale_column": RECOMPOSITION_SCALE_COLUMN,
        "max_abs_residual": float("nan"),
        "max_rel_residual": float("nan"),
        "max_scale": float("nan"),
        "holds": None,
    }
    if not usable.any():
        return record

    absolute = np.abs(residual[usable])
    bound = atol + rtol * np.maximum(scale[usable], np.abs(expected[usable]))
    record["max_abs_residual"] = float(absolute.max())
    record["max_rel_residual"] = float(
        (absolute / np.maximum(scale[usable], atol)).max()
    )
    record["max_scale"] = float(np.nanmax(scale[usable])) if scale[usable].size else float("nan")
    record["holds"] = bool(np.all(absolute <= bound))
    return record


def skill_against(model: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    r"""Per-recording squared-error skill, $1 - \mathrm{MSE}_{\rm model}/\mathrm{MSE}_{\rm ref}$.

    Computed per recording and then averaged, rather than as a ratio of the two averages. The two
    differ, and this is the form the acceptance criteria are stated in: a forecast equal to the
    truth scores exactly $1$ on **every** recording and a forecast equal to the baseline exactly
    $0$ on every recording, so the mean carries those answers unchanged -- and a bootstrap over
    recordings then has a per-recording quantity to resample.

    It lives here rather than in the analysis that needed it first because two of them need it
    now: ``forecast`` scores each model branch against the three trivial baselines, and
    ``coupling`` scores the source-conditioned branch against the target-only one to say what
    percentage of the forecast error the source removed. An analysis may not import another, and
    a second copy of this arithmetic is exactly the drift the layering rule exists to prevent.

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
