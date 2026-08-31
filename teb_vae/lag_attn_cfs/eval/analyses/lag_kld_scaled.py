r"""The lag structure read on the lags that carry the coupling, and with its magnitude kept.

``lag_clocks`` already resolves the lag profile against both clinical clocks. It does so over
**all** $L$ lags and through statistics that are functions of $p_\ell = w_\ell / \sum_k w_k$ alone,
and on this family both of those are load-bearing limitations rather than conventions:

* **Most of the profile is not source content.** The diagnosed conv-Transformer run put
  $\texttt{kld\_source\_null} = 0.333$ of a $\texttt{kl\_total} = 0.494$ -- $67.5\%$ of the
  coupling readout survives zeroing the source -- leaving $0.160$ nats spread over $91$ lags. The
  availability staircase behind that share is a deterministic function of $t$ and is readable from
  the source state at *any* lag, so it enters the attribution wherever the attention happens to sit
  and no renormalisation of the matched profile removes it.
* **The scale is divided out.** Two windows whose profile has the same shape and a tenfold
  different coupling reduce identically, so "the informative past moved closer" and "there is less
  of it" are the same number.
* **The heads are averaged.** ``kl_top_dimension_share`` was $0.954$ on that run: the pooled
  profile is one latent group wearing four heads' name.

This analysis is the three answers, against the same two clocks and the same $0.5$ h grid, so its
rows sit beside ``lag_clocks``' rather than replacing them. **Nothing here re-bases anything
there** -- that analysis's twenty-eight columns are untouched, which is asserted by the fact that
its own suite runs unchanged.

**Four families of source, and the first two are the selection.**

* **The geometry-fixed bands**, ``eval_config.occlusion_bands``. Nothing about them is estimated
  from the KL, so restricting to one is free of the circularity that makes a top-$K$ selection
  test its own selector -- and they are the *same* partition the interventional readout removes
  source from, so a band means one lag range in this run's observational table and in its
  interventional one.
* **The soft weight**, $\omega_\ell = \Delta^+_\ell / \max_k \Delta^+_k$, from the pooled
  clock-excess profile. Computed **once at run level** and applied identically to every segment,
  window and class: a per-segment weight would let each segment choose its own lag axis, and a
  comparison across segments would then be comparing different axes. Withheld entirely when that
  profile is degenerate.
* **The full support**, carrying the two nats-scale statistics only -- the twelve scale-free ones
  are ``lag_clocks``' and restating them here would put one quantity on two pages.
* **The heads**, each head's own KL attribution $K^{(m)}\alpha^{(m)}_\ell$, which sums over $m$ to
  the pooled profile exactly.

**Two statistics are absent from every banded source, and their absence is a measurement.**
``near_mass`` and ``far_mass`` are measured from the axis's own start
(:mod:`~teb_vae.lag_attn_cfs.eval.lag_shape`), so on a band they would silently mean "within
:data:`~teb_vae.lag_attn_cfs.eval.lag_shape.NEAR_SECONDS` of *the band's* start" -- and
``far_mass`` would be identically zero on every band narrower than
:data:`~teb_vae.lag_attn_cfs.eval.lag_shape.FAR_SECONDS`, which is three of the four shipped ones.
Four columns of structural zeros presented as measurements is worse than four absent ones.

**Nothing here is tested, and that is a decision rather than an omission.** Every feature ships
untested, so this analysis adds **no** Holm family to the four ``lag_clocks`` already carries and
writes no significance table at all. At $0.160$ nats of clock-exceeding coupling across $91$ lags,
per-segment restricted centroids are very likely noise, and eight new corrected families over
noise is how a family-wise correction stops being believed. Promoting one is a single flag; the
record says so outright rather than leaving a reader to infer that a $p$-value was withheld.

**The emission is long-form** -- ``source`` and ``statistic`` are row keys rather than columns.
That is what lets ``num_heads`` be a *run* property (it is a ``binding.GEOMETRY_KEYS`` entry, so an
arm can change it) and a band be added without widening a table, and it is the shape
``lag_clocks``' own trajectory table and ``lag_kl``'s stratified profile already use.

**The axis is stored-coefficient time.** Every row and every figure carries
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT`.

.. note::

    lean-limit: :data:`CLOCKS` is a second copy of ``lag_clocks.CLOCKS`` -- an analysis may not
    import another, and everything expensive underneath (the binning, the per-recording reduction,
    the trajectory rows, the figure seam) is already shared, so only the declaration duplicates.
    Promote the pair into ``cohort.py`` when a third clock consumer appears; it is a legal move
    today and is deliberately not made as part of the change that introduced the second copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    GROUP_DELAY_CAVEAT,
    compensated_seconds_axis,
)
from teb_vae.lag_attn_cfs.eval.lag_shape import (
    STATISTIC_KEYS,
    degeneracy,
    profile_statistics,
    rectified_profile,
    restrict_to_band,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "lag_kld_scaled"

#: What it writes. Three tables and two figures, and **no significance or pairwise table** -- see
#: the module docstring for why that absence is the decision it looks like.
PER_RECORDING_FILENAME = "lag_kld_scaled_per_recording.csv"
TRAJECTORY_FILENAME = "lag_kld_scaled_trajectory.csv"
SELECTION_FILENAME = "lag_kld_scaled_selection.csv"

#: Window width in hours, bound from the layer below rather than restated, so a window here is the
#: same duration as one on ``lag_clocks`` and on either coupling clock.
TRAJECTORY_BIN_HOURS = cohort.TRAJECTORY_BIN_HOURS

#: A positional index onto the per-sample vector sidecar, attached so a binned subset can still
#: find its own rows in a matrix aligned with the table's *original* row order. Never emitted.
ROW_COLUMN = "_source_row"

#: The separator between a statistic and its source inside a working column name. Two underscores
#: because both halves contain single ones, and the pair is split back apart on emission -- the
#: emitted table keys on ``source`` and ``statistic`` rather than on a name a reader has to parse.
COLUMN_SEPARATOR = "__"

#: What each statistic is in, for the ``unit`` column every emitted row carries. A number whose
#: unit a reader has to infer is one they will infer wrongly, and this table has four kinds of
#: number in one ``value`` column.
UNITS: Dict[str, str] = {
    "centroid": "s (stored-coefficient time)",
    "spread": "s",
    "median": "s (stored-coefficient time)",
    "iqr": "s",
    "effective_support": "s",
    "peak": "s (stored-coefficient time)",
    "peak_width": "s",
    "entropy": "nats (of the lag distribution)",
    "skewness": "dimensionless",
    "near_mass": "share of the mass",
    "far_mass": "share of the mass",
    "peak_mass": "share of the mass",
    "peak_degenerate": "share of segments",
    "zero_fraction": "share of bins",
    "total_nats": "nats per anchor",
    "peak_nats": "nats per anchor",
}

#: The two statistics a **restricted** source may not carry, and the reason is arithmetic rather
#: than editorial: both are measured from ``seconds[0]``, so on a band they silently re-base onto
#: the band's own start, and ``far_mass`` is identically zero on any band narrower than
#: ``FAR_SECONDS``. Emitting them would be emitting structural zeros as measurements.
NON_RESTRICTABLE: Tuple[str, ...] = ("near_mass", "far_mass")

#: What a **full-support** source carries: the two that keep the scale, and nothing else. The
#: twelve scale-free statistics on the full support are ``lag_clocks``' own columns, and a second
#: copy here would put one quantity on two pages under two names.
FULL_SUPPORT_STATISTICS: Tuple[str, ...] = ("total_nats", "peak_nats")

#: The per-sample vector each base profile is read from. The **untruncated** attribution and the
#: **support-corrected** attention, which is exactly the pair ``lag_clocks`` reduces -- so a
#: nats-scale total here and a centroid there describe the same vector.
BASE_PROFILES: Tuple[Tuple[str, str, str], ...] = (
    (
        "kl",
        "lag_profile_untruncated",
        "the per-lag KL attribution, over the anchors whose lag support is complete",
    ),
    (
        "attn",
        "attention_profile_support_corrected",
        "the attention over the lags, each on its own contributing-anchor count",
    ),
)

#: The per-head KL attribution's vector, $M \cdot L$ wide and head-major.
PER_HEAD_ATTRIBUTE = "lag_profile_per_head"

#: The pooled clock-excess profile the soft weight is built from, on the collection's lag block.
CLOCK_EXCESS_KEY = "kl_lag_profile_clock_excess"

#: The statement every row of this analysis carries about what it did *not* do.
NO_INFERENCE_NOTE = (
    "this analysis runs no significance test and writes no significance or pairwise table. Every "
    "feature here is untested, so it adds NO Holm family to the four lag_clocks carries and a "
    "reader quoting a trajectory from this page is quoting a description, not a claim. That is a "
    "decision and not an omission: at the clock-exceeding coupling this family has measured -- "
    "0.160 nats spread over 91 lags on the diagnosed run -- per-segment restricted centroids are "
    "very likely noise, and correcting eight new families over noise is how a family-wise "
    "correction stops being believed. Promoting a feature is the Feature.tested flag and nothing "
    "else"
)

#: Why the selection is what it is, in the record rather than left to be reconstructed.
SELECTION_NOTE = (
    "the band sources restrict to eval_config.occlusion_bands, which is a GEOMETRY-fixed "
    "partition: nothing about it is estimated from the KL, so a statistic on a band is free of "
    "the circularity that makes a top-K-by-KL selection test its own selector. It is also the "
    "same partition the occlusion analysis removes source from, so a band names one lag range "
    "across this run. The weighted sources use a run-level weight from the pooled clock-excess "
    "profile, applied identically to every segment, window and class -- a per-segment weight "
    "would let each segment choose its own lag axis"
)


# =================================================================================================
# The clocks
# =================================================================================================
@dataclass(frozen=True)
class Clock:
    """One clinical landmark and everything that differs about reading a quantity against it.

    A second copy of ``lag_clocks.Clock``; see this module's ``lean-limit`` note for why it is a
    copy and what would earn its promotion into ``cohort.py``.

    Attributes:
        name: Carried on every row, so the two clocks are one file set.
        binner: The shared binning function for this landmark.
        bin_column: The window index column it adds.
        center_column: The window centre column travelling with it.
        axis_label: The x-axis label, naming the sign convention outright.
        inverted: Whether the axis is drawn with the landmark at the right.
        figure: This clock's figure filename.
        eligible_only: Whether the second-stage eligibility rule applies before binning.
    """

    name: str
    binner: Callable[[pd.DataFrame], pd.DataFrame]
    bin_column: str
    center_column: str
    axis_label: str
    inverted: bool
    figure: str
    eligible_only: bool


#: The two clocks, in the order their pages are written.
CLOCKS: Tuple[Clock, ...] = (
    Clock(
        name="time_to_delivery",
        binner=cohort.add_time_bins,
        bin_column=cohort.BIN_COLUMN,
        center_column=cohort.BIN_CENTER_COLUMN,
        axis_label="Time before delivery (hours)",
        inverted=True,
        figure="lag_kld_scaled_time_to_delivery",
        eligible_only=False,
    ),
    Clock(
        name="second_stage",
        binner=cohort.add_second_stage_bins,
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
        axis_label="Hours from second-stage onset (negative = before onset, positive = after)",
        inverted=False,
        figure="lag_kld_scaled_second_stage",
        eligible_only=True,
    ),
)


# =================================================================================================
# The sources, built at run time
# =================================================================================================
@dataclass(frozen=True)
class Source:
    """One per-lag vector this analysis reduces, and which statistics it may carry.

    Built at **run time** rather than declared as a module constant, because two of the four
    families are run-dimensioned: the bands come from ``eval_config`` and the heads from the
    checkpoint's ``num_heads``, which is a ``binding.GEOMETRY_KEYS`` entry an arm can change. A
    module-level cross product could carry neither, which is the structural reason this analysis
    is not more columns on ``lag_clocks``.

    Attributes:
        key: The short name every row is keyed by.
        family: ``"full"``, ``"band"``, ``"weighted"`` or ``"head"``.
        statistics: Which of :data:`~teb_vae.lag_attn_cfs.eval.lag_shape.STATISTIC_KEYS` it
            carries. A band omits :data:`NON_RESTRICTABLE`; a full-support source carries only
            :data:`FULL_SUPPORT_STATISTICS`.
        meaning: What the profile is, written into every record derived from it.
        band: The inclusive lag band, for a banded source; ``None`` otherwise.
    """

    key: str
    family: str
    statistics: Tuple[str, ...]
    meaning: str
    band: Optional[Tuple[int, int]] = None


def feature_column(source_key: str, statistic: str) -> str:
    """The working column one statistic of one source is carried under.

    Assembled in one place rather than formatted at each use, for the reason the lag axis is: a
    name built its own way in a second file is how a writer and a reader come to disagree about
    which column holds which number. Split back into its two halves on emission, so the durable
    table keys on ``source`` and ``statistic`` rather than on a parsed string.

    Args:
        source_key: The source's short name.
        statistic: The reduction's key.

    Returns:
        ``<statistic>__<source>``.
    """
    return f"{statistic}{COLUMN_SEPARATOR}{source_key}"


def soft_weight(lag: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    r"""The run-level lag weight, $\omega_\ell = \Delta^+_\ell / \max_k \Delta^+_k$.

    Built from the **pooled** clock-excess profile and applied identically to every segment,
    window and class. A per-segment weight is deliberately refused: each segment's own $\Delta$ is
    estimated from the same forward its statistics are, so a per-segment weight would let every
    segment choose its own lag axis and a comparison across segments would compare different axes.

    Peak-normalised rather than sum-normalised, so multiplying a profile by it leaves the result in
    nats and the nats-scale statistics stay readable as nats.

    **Withheld when the clock-excess profile is degenerate**, which is the guard that matters: a
    weight built from a flat profile is a near-uniform vector dressed as a selection, and it would
    make the weighted sources look like independent evidence while being the unweighted ones.

    Args:
        lag: The collection pass's lag block.

    Returns:
        ``(weight, record)`` -- the $(L,)$ weight or ``None``, and why, with the degeneracy
        statistics that decided it.
    """
    excess = [float(value) for value in (lag.get(CLOCK_EXCESS_KEY) or [])]
    if not excess:
        return None, {
            "available": False,
            "reason": (
                "the collection pass reported no clock-excess profile, so this run predates the "
                "lag-resolved source-null arm; the geometry-fixed bands are unaffected"
            ),
        }
    positive, census = rectified_profile(np.asarray(excess, dtype=np.float64))
    shape = degeneracy(positive)
    peak = float(np.nanmax(positive)) if positive.size else 0.0
    if shape["degenerate"] or not np.isfinite(peak) or peak <= 0.0:
        return None, {
            "available": False,
            "reason": (
                "the clock-excess profile has no readable shape, so a weight built from it would "
                "be a near-uniform vector dressed as a selection: "
                + ("; ".join(shape.get("reasons") or []) or "its peak is not positive")
            ),
            "degenerate": bool(shape["degenerate"]),
            "peak_to_median": shape["peak_to_median"],
            "zero_fraction": shape["zero_fraction"],
            "rectified_frac": census["rectified_frac"],
        }
    return np.asarray(positive, dtype=np.float64) / peak, {
        "available": True,
        "definition": "omega_l = max(clock_excess_l, 0) / max_k max(clock_excess_k, 0)",
        "level": "run",
        "degenerate": False,
        "peak_to_median": shape["peak_to_median"],
        "rectified_frac": census["rectified_frac"],
        "net_nats": census["net_nats"],
    }


def build_sources(
    vectors: Dict[str, np.ndarray],
    lag: Dict[str, Any],
    bands: Dict[str, Tuple[int, int]],
    n_lags: int,
    seconds: np.ndarray,
) -> Tuple[List[Source], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Resolve every source this run can carry, with its matrix and its axis.

    Args:
        vectors: The per-sample vector sidecar.
        lag: The collection pass's lag block, for the soft weight.
        bands: The geometry-fixed partition.
        n_lags: The lag axis width.
        seconds: The compensated axis.

    Returns:
        ``(sources, matrices, axes, record)``. A source whose vector the sidecar does not carry is
        absent from all three rather than present and empty, and the record says which and why --
        a run directory collected before a readout existed is a partial input, not a broken one.
    """
    sources: List[Source] = []
    matrices: Dict[str, np.ndarray] = {}
    axes: Dict[str, np.ndarray] = {}
    skipped: List[Dict[str, str]] = []

    def _matrix(attribute: str) -> Optional[np.ndarray]:
        """The sidecar's matrix at the axis width, or ``None`` when it is unusable."""
        raw = vectors.get(attribute)
        if raw is None:
            return None
        array = np.asarray(raw, dtype=np.float64)
        return array if array.ndim == 2 and array.shape[1] == n_lags else None

    weight, weight_record = soft_weight(lag)

    for key, attribute, meaning in BASE_PROFILES:
        matrix = _matrix(attribute)
        if matrix is None:
            skipped.append(
                {"source": key, "reason": f"the sidecar carries no {n_lags}-wide {attribute!r}"}
            )
            continue

        # The full support, carrying the scale the twelve scale-free statistics divide out.
        sources.append(
            Source(key, "full", FULL_SUPPORT_STATISTICS, f"{meaning}, over every lag")
        )
        matrices[key] = matrix
        axes[key] = seconds

        # One source per geometry-fixed band.
        for name, span in bands.items():
            banded_key = f"{key}_{name}"
            restricted, restricted_axis = restrict_to_band(matrix, seconds, span)
            sources.append(
                Source(
                    banded_key,
                    "band",
                    tuple(k for k in STATISTIC_KEYS if k not in NON_RESTRICTABLE),
                    f"{meaning}, restricted to the {name!r} band, lags {span[0]}-{span[1]}",
                    band=(int(span[0]), int(span[1])),
                )
            )
            matrices[banded_key] = restricted
            axes[banded_key] = restricted_axis

        # The soft-weighted source, when the clock-excess profile earned one.
        if weight is not None:
            weighted_key = f"{key}_dw"
            sources.append(
                Source(
                    weighted_key,
                    "weighted",
                    tuple(STATISTIC_KEYS),
                    f"{meaning}, weighted by the run-level clock-excess weight",
                )
            )
            matrices[weighted_key] = matrix * weight[None, :]
            axes[weighted_key] = seconds

    # The heads, from the per-head KL attribution. Reshaped here, head-major, with the same
    # drop-whole guard the other two sites carry -- see ``metrics.lag_summary`` and
    # ``lag_kl._per_head_rows``, which are the other two. A flat vector whose length does not
    # factor is a mis-assembled profile rather than a short one.
    flat = vectors.get(PER_HEAD_ATTRIBUTE)
    if flat is not None:
        array = np.asarray(flat, dtype=np.float64)
        num_heads = int(lag.get("num_heads") or 0)
        if array.ndim == 2 and num_heads and array.shape[1] == num_heads * n_lags:
            for head in range(num_heads):
                head_key = f"kl_h{head}"
                sources.append(
                    Source(
                        head_key,
                        "head",
                        tuple(STATISTIC_KEYS),
                        (
                            f"head {head}'s own KL attribution, K^({head}) times its attention "
                            f"over the lags; the heads sum to the pooled attribution exactly"
                        ),
                    )
                )
                matrices[head_key] = array[:, head * n_lags : (head + 1) * n_lags]
                axes[head_key] = seconds
        else:
            skipped.append(
                {
                    "source": "kl_h*",
                    "reason": (
                        f"the per-head vector is {array.shape[-1] if array.ndim == 2 else '?'} "
                        f"wide against num_heads={num_heads} and n_lags={n_lags}, so it does not "
                        f"factor and is dropped whole rather than reshaped"
                    ),
                }
            )

    return sources, matrices, axes, {
        "n_sources": len(sources),
        "families": sorted({source.family for source in sources}),
        "skipped": skipped,
        "weight": weight_record,
        "selection_note": SELECTION_NOTE,
    }


# =================================================================================================
# The scalars, per segment
# =================================================================================================
def add_feature_columns(
    per_sample: pd.DataFrame,
    sources: Sequence[Source],
    matrices: Dict[str, np.ndarray],
    axes: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Attach every source's statistics to a copy of the per-sample table.

    Args:
        per_sample: The collected per-sample table.
        sources: The resolved sources.
        matrices: Each source's $(n, L')$ matrix.
        axes: Each source's own seconds axis, which for a band is the band's.

    Returns:
        ``(frame, columns, record)`` -- the copy carrying the working columns and
        :data:`ROW_COLUMN`, the column names in source order, and one entry per source saying how
        many segments it reduced.
    """
    frame = per_sample.copy()
    frame[ROW_COLUMN] = np.arange(len(frame), dtype=np.int64)
    columns: List[str] = []
    record: Dict[str, Any] = {}
    # Accumulated and attached in ONE concat at the end. Assigned column by column this builds
    # several hundred of them on a run with four bands and four heads, and pandas fragments the
    # block manager once per assignment -- which it warns about, loudly, once per column.
    built: Dict[str, np.ndarray] = {}
    for source in sources:
        rows = matrices[source.key][: len(frame)]
        statistics, counts = profile_statistics(rows, axes[source.key])
        for statistic in source.statistics:
            column = feature_column(source.key, statistic)
            values = np.asarray(statistics[statistic], dtype=np.float64)
            if values.size != len(frame):
                padded = np.full(len(frame), np.nan, dtype=np.float64)
                padded[: min(values.size, len(frame))] = values[: len(frame)]
                values = padded
            built[column] = values
            columns.append(column)
        record[source.key] = {
            "family": source.family,
            "meaning": source.meaning,
            "band": None if source.band is None else [source.band[0], source.band[1]],
            "n_statistics": len(source.statistics),
            **counts,
        }
    if built:
        frame = pd.concat(
            [frame, pd.DataFrame(built, index=frame.index)], axis=1, copy=False
        )
    return frame, columns, record


def clock_rows(clock: Clock, featured: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bin the featured table on one clock, applying that clock's population rule first.

    Args:
        clock: The clock to place the segments on.
        featured: The per-sample table with the working columns attached.

    Returns:
        ``(binned, population)``. The second clock's population is a **subset** -- a recording with
        no recorded onset cannot be placed on that axis at all -- and the rule is
        :func:`~teb_vae.lag_attn_cfs.eval.cohort.second_stage_eligibility`'s, the shared one, so
        the two analyses that use it cannot disagree about who is eligible.
    """
    frame = featured
    population: Dict[str, Any] = {"clock": clock.name, "n_segments_before": int(len(frame))}
    if clock.eligible_only:
        eligibility = cohort.second_stage_eligibility(frame)
        keep = set(eligibility.loc[eligibility["eligible"], "guid"]) if len(eligibility) else set()
        frame = frame[frame["guid"].isin(sorted(keep))] if "guid" in frame.columns else frame.iloc[:0]
        population["n_recordings_eligible"] = int(len(keep))
    binned = clock.binner(frame)
    population["n_segments_binned"] = int(len(binned))
    population["drawn"] = bool(len(binned))
    return binned, population


# =================================================================================================
# Emission
# =================================================================================================
def _split(column: str) -> Tuple[str, str]:
    """Split a working column back into ``(statistic, source)``."""
    statistic, _, source = column.partition(COLUMN_SEPARATOR)
    return statistic, source


def per_recording_frame(
    clock: Clock, binned: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """One row per (cohort, window, recording, source, statistic), long-form.

    Reduced wide -- :func:`~teb_vae.lag_attn_cfs.eval.cohort.per_recording_in_bins` takes a column
    list -- and melted immediately, so ``source`` and ``statistic`` become row keys. That is what
    lets ``num_heads`` be a run property and a band be added without widening a table.

    Args:
        clock: The clock whose windows to group on.
        binned: The binned per-sample table.
        columns: The working columns.

    Returns:
        The long-form frame, empty with its key columns present when nothing is usable.
    """
    keys = ["clock", "group_column", "group", "guid", "time_bin", "bin_center_h",
            "source", "statistic", "unit", "value"]
    frames: List[pd.DataFrame] = []
    for axis in labels.GROUP_COLUMNS:
        wide = cohort.per_recording_in_bins(
            binned,
            columns,
            group_column=axis,
            bin_column=clock.bin_column,
            center_column=clock.center_column,
        )
        present = [name for name in columns if name in getattr(wide, "columns", [])]
        if wide.empty or not present:
            continue
        melted = wide.melt(
            id_vars=["group", clock.bin_column, clock.center_column, "guid"],
            value_vars=present,
            var_name="_column",
            value_name="value",
        )
        melted["clock"] = clock.name
        melted["group_column"] = axis
        melted["statistic"] = [_split(name)[0] for name in melted["_column"]]
        melted["source"] = [_split(name)[1] for name in melted["_column"]]
        melted["unit"] = [UNITS.get(name, "") for name in melted["statistic"]]
        melted = melted.rename(
            columns={clock.bin_column: "time_bin", clock.center_column: "bin_center_h"}
        )
        frames.append(melted[keys])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keys)


def trajectory_frame(
    clock: Clock, binned: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """One row per (cohort, window, source, statistic), over that cell's recordings.

    Args:
        clock: The clock the windows belong to.
        binned: The binned per-sample table.
        columns: The working columns.

    Returns:
        The long-form trajectory, carrying the count of **recordings** behind each cell rather
        than of segments: a recording contributing eleven segments to a window must not outvote
        one contributing two.
    """
    rows: List[Dict[str, Any]] = []
    for axis in labels.GROUP_COLUMNS:
        wide = cohort.per_recording_in_bins(
            binned,
            columns,
            group_column=axis,
            bin_column=clock.bin_column,
            center_column=clock.center_column,
        )
        if wide.empty:
            continue
        for column in columns:
            if column not in wide.columns:
                continue
            statistic, source = _split(column)
            for row in cohort.trajectory_rows(
                wide,
                column,
                metric=column,
                bin_column=clock.bin_column,
                center_column=clock.center_column,
            ):
                entry = {
                    "clock": clock.name,
                    "group_column": axis,
                    "source": source,
                    "statistic": statistic,
                    "unit": UNITS.get(statistic, ""),
                    **row,
                }
                entry.pop("metric", None)
                rows.append(entry)
    return pd.DataFrame(rows)


def selection_frame(
    lag: Dict[str, Any],
    seconds: np.ndarray,
    bands: Dict[str, Tuple[int, int]],
    weight: Optional[np.ndarray],
) -> pd.DataFrame:
    """The run's durable record of which lags this analysis selected and how.

    One row per lag. Written because a selection reconstructed later from a re-run is not the
    selection the numbers beside it were chosen with -- the bands come from a config that can be
    edited and the weight from a profile that a different checkpoint would move.

    Args:
        lag: The collection pass's lag block.
        seconds: The compensated axis.
        bands: The geometry-fixed partition.
        weight: The run-level soft weight, or ``None``.

    Returns:
        One row per lag: its band, its clock-excess value, and the weight it received.
    """
    excess = list(lag.get(CLOCK_EXCESS_KEY) or [])
    rows: List[Dict[str, Any]] = []
    for index, second in enumerate(seconds):
        member = next(
            (name for name, span in bands.items() if span[0] <= index <= span[1]), ""
        )
        rows.append(
            {
                "lag_step": int(index),
                "compensated_seconds": float(second),
                "band": member,
                "clock_excess_nats": (
                    float(excess[index]) if index < len(excess) else float("nan")
                ),
                "soft_weight": (
                    float(weight[index])
                    if weight is not None and index < weight.size
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def build_band_figure(
    clock: Clock,
    trajectory: pd.DataFrame,
    bands: Dict[str, Tuple[int, int]],
    seconds: np.ndarray,
) -> Any:
    """One panel per band: how many nats of KL attribution that band held, window by window.

    This is the page the analysis exists for. The bands are geometry-fixed, so a line moving
    between panels is the informative past moving; a line moving *within* a panel while the others
    hold is that band's coupling changing magnitude. Neither is visible on a scale-free statistic
    over the whole window, which is what ``lag_clocks`` draws.

    Args:
        clock: The clock, for its axis label and orientation.
        trajectory: The long-form trajectory frame.
        bands: The partition, in panel order.
        seconds: The compensated axis, for each band's span in the panel title.

    Returns:
        The figure.
    """
    names = list(bands)
    figure, axes = figures.new_figure(max(len(names), 1), 1)
    subset = (
        trajectory[
            (trajectory["clock"] == clock.name)
            & (trajectory["statistic"] == "total_nats")
            & (trajectory["group_column"] == labels.CLASS_COLUMN)
        ]
        if len(trajectory)
        else trajectory
    )
    groups = sorted({str(value) for value in subset["group"]}) if len(subset) else []
    colors = figures.group_colors(groups) if groups else {}
    for index, name in enumerate(names):
        panel = axes[index][0]
        span = bands[name]
        lo = float(seconds[max(span[0], 0)]) if seconds.size else 0.0
        hi = float(seconds[min(span[1], seconds.size - 1)]) if seconds.size else 0.0
        band_rows = subset[subset["source"] == f"kl_{name}"] if len(subset) else subset
        if not len(band_rows):
            panel.text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=panel.transAxes,
                ha="center", va="center", color=figures.COLOR_GRAY, fontsize=figures.FONT_NOTE,
            )
        for group in groups:
            cell = band_rows[band_rows["group"] == group].sort_values("bin_center_h")
            if not len(cell):
                continue
            panel.plot(
                cell["bin_center_h"], cell["mean"],
                color=colors.get(group, figures.COLOR_GRAY),
                linewidth=figures.LINE_REGULAR, label=group,
            )
        panel.set_title(
            f"{name}: lags {span[0]}-{span[1]} ({lo:g}-{hi:g} s)", fontsize=figures.FONT_SMALL
        )
        panel.set_ylabel("nats per anchor")
        if clock.inverted:
            panel.invert_xaxis()
        figures.style_axes(panel)
    axes[-1][0].set_xlabel(clock.axis_label)
    if groups:
        axes[0][0].legend(loc="best", fontsize=figures.FONT_TINY)
    figures.caveat_note(figure)
    return figure


# =================================================================================================
# Entry point
# =================================================================================================
def _skip(reason: str, n_segments: int) -> Dict[str, Any]:
    """A recorded skip carrying the protocol's keys and naming its cause."""
    return {
        "n_samples": int(n_segments),
        "composition": {},
        "plan": {"capped": True, "skipped": True, "reason": reason},
        "skipped": True,
        "reason": reason,
        "tested": False,
        "no_inference_note": NO_INFERENCE_NOTE,
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "files": [],
    }


def run_lag_kld_scaled_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the KLD-scaled, band-restricted and per-head lag structure against both clocks.

    Args:
        context: The analysis context, read for the per-sample table, the vector sidecar and the
            collection's lag block. No model is touched, so this runs against a finished run
            directory with no checkpoint and no GPU.
        eval_config: The validated block, read for ``occlusion_bands`` -- the geometry-fixed
            partition, shared with the interventional readout -- and the run's horizon.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population is the set of segments carrying a
            finite clock coordinate and a lag profile, which only the tables know.

    Returns:
        The protocol's keys, the resolved sources, the selection record, the per-clock populations
        and the paths written. A recorded skip -- naming its cause -- when the table is empty, when
        no band is configured, or when the sidecar carries no profile.
    """
    collection = context.collection
    per_sample = cohort.within_horizon(
        collection.per_sample, eval_config.get("max_hours_before_delivery")
    )
    if per_sample.empty:
        return _skip("the collected per-sample table was empty", 0)

    bands = {
        str(name): (int(span[0]), int(span[1]))
        for name, span in (eval_config.get("occlusion_bands") or {}).items()
        if isinstance(span, (list, tuple)) and len(span) == 2
    }
    if not bands:
        return _skip(
            "eval_config.occlusion_bands names no band, so there is no geometry-fixed partition "
            "to restrict to. That key is read by two analyses -- this one and occlusion -- so "
            "emptying it to skip the interventional pass also removes the restricted statistics; "
            "the full-support and per-head sources are withheld with them rather than emitted "
            "under a page whose selection half is missing",
            len(per_sample),
        )

    lag = dict(dict(getattr(collection, "results", None) or {}).get("lag") or {})
    n_lags = int(lag.get("n_lags") or 0)
    if n_lags <= 0:
        return _skip(
            "the collection record carries no lag geometry, so there is no axis to resolve a "
            "profile against",
            len(per_sample),
        )
    seconds = compensated_seconds_axis(n_lags, int(lag.get("delay_steps") or 0))
    vectors = dict(getattr(collection, "vectors", None) or {})

    sources, matrices, axes, source_record = build_sources(vectors, lag, bands, n_lags, seconds)
    if not sources:
        return _skip(
            "no per-lag vector the sidecar carries could be reduced at this run's lag width, so "
            f"there is nothing to resolve against a clock (skipped: {source_record['skipped']})",
            len(per_sample),
        )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    featured, columns, feature_record = add_feature_columns(per_sample, sources, matrices, axes)
    weight, _ = soft_weight(lag)

    per_recording: List[pd.DataFrame] = []
    trajectory: List[pd.DataFrame] = []
    populations: List[Dict[str, Any]] = []
    written: List[str] = []
    for clock in CLOCKS:
        binned, population = clock_rows(clock, featured)
        populations.append(population)
        if binned.empty:
            population["reason"] = f"no segment could be placed on the {clock.name} clock"
            continue
        per_recording.append(per_recording_frame(clock, binned, columns))
        clock_trajectory = trajectory_frame(clock, binned, columns)
        trajectory.append(clock_trajectory)
        written.append(
            str(
                figures.render_figure(
                    build_band_figure(clock, clock_trajectory, bands, seconds),
                    directory / clock.figure,
                ).name
            )
        )

    # Written even when empty, with their headers: a table a reader can open and find nothing in
    # is a different statement from a table that was never written.
    combined_recording = (
        pd.concat(per_recording, ignore_index=True) if per_recording else pd.DataFrame()
    )
    combined_trajectory = (
        pd.concat(trajectory, ignore_index=True) if trajectory else pd.DataFrame()
    )
    combined_recording.to_csv(directory / PER_RECORDING_FILENAME, index=False)
    combined_trajectory.to_csv(directory / TRAJECTORY_FILENAME, index=False)
    selection_frame(lag, seconds, bands, weight).to_csv(
        directory / SELECTION_FILENAME, index=False
    )
    written = [PER_RECORDING_FILENAME, TRAJECTORY_FILENAME, SELECTION_FILENAME] + written

    return {
        "n_samples": scored_sample_count(per_sample, "source_conditioned_kl_raw"),
        "composition": {
            "n_recordings": int(featured["guid"].nunique()) if "guid" in featured else 0,
            "n_lags": n_lags,
            "n_bands": len(bands),
        },
        # Capped because the second clock's population is a strict subset of the first's -- a
        # recording with no recorded onset cannot be placed on that axis -- so the coverage block
        # must read this as a different population by design rather than as a disagreement.
        "plan": {
            "capped": True,
            "bands": {name: [span[0], span[1]] for name, span in bands.items()},
            "tested_features": 0,
        },
        "sources": source_record,
        "features": feature_record,
        "clocks": populations,
        # Stated outright rather than left to be inferred from the absence of a file.
        "tested": False,
        "no_inference_note": NO_INFERENCE_NOTE,
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "files": written,
    }
