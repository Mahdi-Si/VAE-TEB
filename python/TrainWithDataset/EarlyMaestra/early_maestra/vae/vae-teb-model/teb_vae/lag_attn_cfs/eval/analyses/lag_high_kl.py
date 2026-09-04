r"""The lag structure of the anchors that carry the coupling, selected by their own $K_t$.

Every lag readout in this pipeline averages a segment's anchors together before it reads a lag
position -- ``lag_kl`` pools them over a recording, ``lag_clocks`` and ``lag_kld_scaled`` per
segment before placing it on a clock. On this family that average is dominated by anchors whose
KL is the availability clock and little else: the diagnosed runs put $67$--$92\%$ of the coupling
readout in a term that survives zeroing the source, spread over every anchor of every segment. The
anchors at which the source actually informed the future are the minority with a large $K_t$, and
their lag profile is what the pooled one dilutes.

**This analysis selects those anchors and reads their lag structure.** The selection is a
**quantile band of the pooled per-anchor KL** -- pooled over every scored anchor of every class in
the evaluated population, so one run has one threshold in nats and every segment, window and class
is cut by the same number. The shipped bands are

* ``high`` -- the anchors in the upper $30\%$ of the pooled $K_t$ distribution ($q \in [0.7, 1]$);
* ``rest`` -- its complement, so the two recompose to every anchor and a contrast is a contrast
  against everything that was not selected rather than against a second selection;
* ``top`` -- the upper $10\%$, the same question asked more sharply.

For every band, per segment: the **share of the segment's anchors** that fall in it, and the
**lag profile restricted to those anchors** -- the mean of the per-anchor KL attribution
$\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$ over the selected anchors, and
the head-averaged attention over the same anchors -- reduced through the shared shape vocabulary
of :mod:`~teb_vae.lag_attn_cfs.eval.lag_shape`. Both are then resolved against the two clinical
clocks on the same $0.5$ h grid every other clock analysis uses, by class, and the two readouts
that state the hypothesis -- the high band's KL centroid and the high-anchor share -- are tested
per window with Holm across windows.

**Whether any of it is *useful* is asked directly, in forecast space.** A large $K_t$ says the
source moved the belief; it does not say the forecast got better. The per-anchor table carries the
Monte Carlo forecast gain of the same anchor, ``mc_pred_gap`` $= D_{\mathrm{base}} -
D_{\mathrm{full}}$ in nats, so every band's anchors are also scored by what the source bought
there: ``<band>_pred_gap_nats`` per segment, the paired within-recording difference of the high
band's gain against the rest band's with a Wilcoxon signed-rank test over recordings, the gain
resolved by KL decile and by the anchor's argmax lag, a fourth band ``gain`` -- the anchors in the
upper $30\%$ of the pooled forecast gain -- with its own lag profile and its overlap with the high
band against the $30\%$ independence would give, a **gain-weighted** attention profile
$\sum_t \max(g_t, 0)\,\alpha_{t\ell} / \sum_t \max(g_t, 0)$ (where does the source look when it
helps), and -- when the interventional ``occlusion`` analysis has run -- a per-recording join of
each geometry band's attribution share against the forecast cost of removing that band. A model
whose high-KL anchors carry no forecast gain, whose gain anchors are not its high-KL anchors and
whose attribution share in a band does not track the cost of occluding it is a model whose lag
readout is not useful, however sharp its profile.

**Three further readings come from the same selection**, each cheap once the per-anchor table
and the sidecar are open:

* **Hot lags.** The lags whose *pooled* attribution -- each segment's mean over all its anchors,
  averaged over every segment of the population -- sits in the upper $30\%$ across the $L$ lags. A run-level set, recorded lag by lag, and the
  per-segment share of attribution landing on it is placed on both clocks beside the band
  readouts. This is the top-$K$-by-KL selection ``lag_kld_scaled`` declines for its own bands,
  and it is taken here deliberately, with the circularity stated on every artifact: the set is
  chosen from the same attribution it then summarises, so a share on it is a description of the
  run's own selection and must not be read as an independent test of it. The selection is pooled
  over every class, which is what keeps a *class contrast* on it honest -- no class chose its own
  lags.
* **Where the KL sits on the lag axis, by KL magnitude.** The per-anchor ``argmax_lag`` against
  the KL decile of the same anchor: whether the anchors with the most coupling peak at the same
  lags as the anchors with the least. A flat picture across deciles says the argmax is a property
  of the geometry rather than of the coupling; a picture that moves says which lags the coupling
  actually lives at.
* **Contraction enrichment.** Whether high-KL anchors are more common within
  ``event_lag_window_s`` of a detected contraction than outside it, per recording and then by
  class -- the coupling-magnitude counterpart of the ``events`` analysis, on the same per-anchor
  contraction age and with no extra pass.

**What is tested and what is not, stated rather than left to inference.** Two readouts on two
clocks: four Holm families, each within its own clock and readout, none joint -- the rule
``lag_clocks`` states and this analysis adopts -- plus **one** run-level paired test, the high
band's forecast gain against the rest band's within recording, which is its own family of one.
Everything else here is tabled and drawn, with no $p$-value: the ``rest``, ``top`` and ``gain``
bands' clock trajectories, the attention profiles, the hot-lag shares, the decile and argmax
tables, the occlusion join and the contraction enrichment. The record says so in ``method`` and
``untested_note``.

**The population is the recording, inside a window as well as across it**, on both clocks; the
second clock's is the subset that carries a second-stage onset, by the shared eligibility rule.

**The axis is stored-coefficient time.** Every lag-resolved artifact carries
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT`.

.. note::

    lean-limit: :data:`CLOCKS` is the third copy of the two-clock declaration (``lag_clocks`` and
    ``lag_kld_scaled`` carry the other two). The promotion those modules name -- into
    ``cohort.py`` -- is blocked by ``test_eval_sibling_agreement``, which pins that file as one
    import line from the sibling; replace the three copies with a layer-0 ``clocks`` module when
    the layering allow-list admits one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    GROUP_DELAY_CAVEAT,
    compensated_seconds_axis,
    read_lag_support,
)
from teb_vae.lag_attn_cfs.eval.lag_shape import (
    DEGENERATE_PEAK_TO_MEDIAN,
    DEGENERATE_ZERO_FRACTION,
    SECONDS_PER_LAG_STEP,
    STATISTIC_KEYS,
    profile_statistics,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "lag_high_kl"

#: What it writes. Every table names the clock on its rows where a clock applies, so the two
#: clocks are one file set and a comparison across them is a filter rather than a join.
THRESHOLDS_FILENAME = "lag_high_kl_thresholds.csv"
SELECTION_FILENAME = "lag_high_kl_selection.csv"
RECORDINGS_FILENAME = "lag_high_kl_recordings.csv"
PER_RECORDING_FILENAME = "lag_high_kl_per_recording.csv"
TRAJECTORY_FILENAME = "lag_high_kl_trajectory.csv"
PROFILE_FILENAME = "lag_high_kl_profile.csv"
SIGNIFICANCE_FILENAME = "lag_high_kl_significance.csv"
PAIRWISE_FILENAME = "lag_high_kl_pairwise.csv"
ARGMAX_FILENAME = "lag_high_kl_argmax_by_quantile.csv"
CONTRACTION_FILENAME = "lag_high_kl_contraction.csv"
GAIN_BY_QUANTILE_FILENAME = "lag_high_kl_gain_by_kl_quantile.csv"
GAIN_BY_ARGMAX_FILENAME = "lag_high_kl_gain_by_argmax.csv"
OCCLUSION_CONSISTENCY_FILENAME = "lag_high_kl_occlusion_consistency.csv"
SELECTION_FIGURE = "lag_high_kl_selection"
USEFULNESS_FIGURE = "lag_high_kl_usefulness"

#: The interventional analysis's per-recording table, read off disk when it exists. Named here
#: rather than imported: an analysis may not import another, and the dependency is on a FILE
#: existing -- which is what keeps ``--only lag_high_kl`` working against a finished directory.
OCCLUSION_PER_RECORDING = ("occlusion", "occlusion_per_recording.csv")
OCCLUSION_DELTA_PREFIX = "occlusion_delta_"
OCCLUSION_DELTA_SUFFIX = "_nats"

#: The per-anchor vector sidecar's two maps and the per-anchor table's columns this analysis reads.
#: Named here rather than imported from ``collect``: an analysis may not import the collection
#: module, and the names are part of the sidecar's on-disk contract either way.
KL_MAP_KEY = "kl_lag_map"
ATTENTION_MAP_KEY = "attention_lag_map"
KL_COLUMN = "kld_per_t"
#: The per-anchor forecast gain, $D_{\mathrm{base}} - D_{\mathrm{full}}$ in nats: the Monte Carlo
#: marginalised one where the pass produced it, else the single-draw training-path one.
GAIN_COLUMNS: Tuple[str, ...] = ("mc_pred_gap", "pred_gap")
ARGMAX_COLUMN = "argmax_lag"
SAMPLE_INDEX_COLUMN = "sample_index"
CONTRACTION_AGE_COLUMN = "seconds_since_contraction"

#: Family-wise error rate each (clock, readout) family's Holm correction controls. Not an
#: ``eval_config`` key, for the reason no significance level in this pipeline is one.
DEFAULT_ALPHA = 0.05

#: Window width in hours, bound from the layer below so a window here is the same duration as a
#: window on every other clock page of the run.
TRAJECTORY_BIN_HOURS = cohort.TRAJECTORY_BIN_HOURS

#: A positional index onto the per-sample row order, attached to the frame this analysis works on
#: so a binned subset can still find its rows in the restricted-profile matrices. Never emitted.
ROW_COLUMN = "_source_row"

#: Lags whose pooled attribution sits at or above this quantile *across the lags* form the hot set.
#: A module constant rather than a setting: an operator who could move it could move which lags a
#: run is said to have concentrated on.
HOT_LAG_QUANTILE = 0.7

#: How many equal-count bins of the pooled per-anchor KL the argmax table is resolved over.
N_KL_QUANTILE_BINS = 10

#: Fewest anchors a recording must hold in **each** arm -- within the contraction window and
#: outside it -- for its enrichment difference to be reported. Below this a share is a coin toss.
MIN_ENRICHMENT_ANCHORS = 5


@dataclass(frozen=True)
class AnchorBand:
    """One quantile band of the pooled per-anchor KL, and whether its readouts are tested.

    Attributes:
        key: The short name every column of the band is prefixed with.
        q_lo: Lower quantile, inclusive.
        q_hi: Upper quantile. Inclusive when it is $1$ (the maximum belongs to the top band),
            exclusive otherwise, so adjacent bands partition rather than overlap.
        tested: Whether this band's centroid and anchor share enter the inference. One band is,
            which keeps each clock at two families rather than eight.
        meaning: What the band is, written into every record derived from it.
        on: The per-anchor quantity the band is cut on -- ``"kl"`` for $K_t$, ``"gain"`` for the
            forecast gain. The ``gain`` band is the usefulness counterpart of the ``high`` one:
            the anchors where the source bought the most forecast, whatever their KL.
    """

    key: str
    q_lo: float
    q_hi: float
    tested: bool
    meaning: str
    on: str = "kl"


#: The three bands. ``high`` is the hypothesis; ``rest`` is its complement, so the two recompose to
#: every anchor and the contrast is against everything unselected; ``top`` is the same question
#: asked of the sharpest tenth.
ANCHOR_BANDS: Tuple[AnchorBand, ...] = (
    AnchorBand(
        "high", 0.7, 1.0, tested=True,
        meaning="anchors whose KL lies in the upper 30% of the pooled per-anchor distribution",
    ),
    AnchorBand(
        "rest", 0.0, 0.7, tested=False,
        meaning="the complement of the high band: every anchor below its pooled 0.7 quantile",
    ),
    AnchorBand(
        "top", 0.9, 1.0, tested=False,
        meaning="anchors whose KL lies in the upper 10% of the pooled per-anchor distribution",
    ),
    AnchorBand(
        "gain", 0.7, 1.0, tested=False,
        meaning=(
            "anchors whose forecast gain (base minus full block NLL, nats) lies in the upper 30% "
            "of the pooled per-anchor distribution -- selected on usefulness rather than on KL"
        ),
        on="gain",
    ),
)

#: The band the usefulness readouts are stated against, and the one it is contrasted with.
HIGH_BAND_KEY = "high"
REST_BAND_KEY = "rest"
GAIN_BAND_KEY = "gain"

#: The two per-anchor maps a band's profile is read from, and why both: the attribution is $K_t$
#: times the attention and inherits the prior-variance inflation the attention is immune to, so a
#: shift visible in one and absent from the other is a finding about which is being read.
PROFILE_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("kl", KL_MAP_KEY, "the per-anchor KL attribution over the lags, at the selected anchors"),
    ("attn", ATTENTION_MAP_KEY, "the head-averaged attention over the lags, at the selected anchors"),
)

#: The unit each statistic is spelled into its column name with. Seconds for the positional ones,
#: nats for the entropy; the shares, the moment and the two nats-scale totals carry their unit in
#: the key already.
STATISTIC_SUFFIX: Dict[str, str] = {
    "centroid": "_s",
    "spread": "_s",
    "median": "_s",
    "iqr": "_s",
    "effective_support": "_s",
    "peak": "_s",
    "peak_width": "_s",
    "entropy": "_nats",
}

#: The clock-independent columns every band adds beside its profile statistics: its share of the
#: segment's anchors, their count, and the mean forecast gain over them.
ANCHOR_FRAC_SUFFIX = "anchor_frac"
N_ANCHORS_SUFFIX = "n_anchors"
PRED_GAP_SUFFIX = "pred_gap_nats"

#: The hot-lag readouts, one per profile source.
HOT_SHARE_COLUMNS: Dict[str, str] = {"kl": "hot_lag_share_kl", "attn": "hot_lag_share_attn"}


def band_column(band_key: str, suffix: str) -> str:
    """Build a band-prefixed column name -- ``<band>_<suffix>`` -- in one place."""
    return f"{band_key}_{suffix}"


def feature_column(band_key: str, source_key: str, statistic: str) -> str:
    """The column one statistic of one profile of one band is carried under.

    Args:
        band_key: The anchor band's short name.
        source_key: ``"kl"`` or ``"attn"``.
        statistic: One of :data:`~teb_vae.lag_attn_cfs.eval.lag_shape.STATISTIC_KEYS`.

    Returns:
        ``<band>_lag_<statistic>_<source><unit suffix>`` -- for example
        ``high_lag_centroid_kl_s``, the tested one.
    """
    return f"{band_key}_lag_{statistic}_{source_key}{STATISTIC_SUFFIX.get(statistic, '')}"


#: Every per-segment column this analysis attaches, in table order: per band, its anchor share and
#: count, then every statistic of both profiles; then the two hot-lag shares.
FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    [
        column
        for band in ANCHOR_BANDS
        for column in (
            band_column(band.key, ANCHOR_FRAC_SUFFIX),
            band_column(band.key, N_ANCHORS_SUFFIX),
            band_column(band.key, PRED_GAP_SUFFIX),
            *(
                feature_column(band.key, source_key, statistic)
                for statistic in STATISTIC_KEYS
                for source_key, _, _ in PROFILE_SOURCES
            ),
        )
    ]
    + list(HOT_SHARE_COLUMNS.values())
)

#: The tested readouts: the high band's KL centroid and its anchor share. Two, so each clock is two
#: Holm families; promoting another is a change to this tuple and nothing else.
READOUTS: Tuple[str, ...] = tuple(
    column
    for band in ANCHOR_BANDS
    if band.tested
    for column in (
        feature_column(band.key, "kl", "centroid"),
        band_column(band.key, ANCHOR_FRAC_SUFFIX),
    )
)

#: The method sentence written into every record, so a $p$-value here is readable without this
#: module.
METHOD = (
    "Anchors are selected by a quantile band of the POOLED per-anchor KL -- pooled over every "
    "scored anchor of every clinical class in the evaluated population -- so one run has one "
    "threshold in nats applied identically to every segment, window and class. Per window of the "
    "named clock: Kruskal-Wallis across clinical classes over one value per recording, Holm "
    "step-down across that clock's windows as one family, pairwise two-sided Mann-Whitney U with "
    "Cliff's delta for the windows significant after Holm only, every pair oriented from the more "
    "severe class to the less severe one. Two readouts are tested -- the high band's KL centroid "
    "and the high-anchor share -- on two clocks, so four Holm families, none joint. One further "
    "run-level test, its own family of one: a Wilcoxon signed-rank test over recordings of the "
    "high band's mean forecast gain against the rest band's, paired within recording. Classes with "
    f"fewer than {shared_stats.MIN_GROUP_SIZE} recordings in a window are excluded and recorded."
)

#: What ships untested, said outright.
UNTESTED_NOTE = (
    "the rest, top and gain bands' clock trajectories, every attention-profile statistic, the "
    "hot-lag shares, the argmax-by-KL-decile table, the gain-by-decile and gain-by-argmax tables, "
    "the occlusion-consistency join and the contraction enrichment are tabled and drawn with no "
    "p-value. They add no Holm family; a trajectory quoted from them is a description, not a claim."
)

#: What the usefulness block answers and how, in the record.
USEFULNESS_NOTE = (
    "a large K_t says the source moved the belief, not that the forecast improved. Each band's "
    "anchors are therefore also scored by the forecast gain of the same anchor, "
    "D_base - D_full in nats, and the high band's mean gain is compared with the rest band's "
    "WITHIN each recording. Positive means the anchors carrying the coupling are the anchors "
    "where the source bought forecast; zero or negative means the KL is not where the usefulness "
    "is. The gain band is the same question from the other side: its lag profile is where the "
    "source looked when it helped, and its overlap with the high band against the 30% that "
    "independence would give says whether the two selections name the same anchors."
)

#: The circularity, stated on the artifact rather than left for a reader to notice.
SELECTION_NOTE = (
    "the anchor bands and the hot-lag set are selected FROM the KL they then summarise. A share of "
    "attribution on the hot lags, or a centroid over the high anchors, describes the run's own "
    "selection and is not an independent test of it. Two things keep a cohort contrast on them "
    "honest: the thresholds are pooled over every class, so no class chose its own anchors or its "
    "own lags, and every per-recording value is compared across classes under the same threshold. "
    "The geometry-fixed bands of lag_kld_scaled and the interventional occlusion readout remain "
    "the selections that need no estimate."
)


# =================================================================================================
# The clocks
# =================================================================================================
@dataclass(frozen=True)
class Clock:
    """One clinical landmark and everything that differs about reading a quantity against it.

    Attributes:
        name: Carried on every row, so the two clocks are one file set.
        binner: The shared binning function for this landmark.
        bin_column: The window index column it adds.
        center_column: The window centre column travelling with it.
        axis_label: The x-axis label, naming the sign convention outright.
        inverted: Whether the axis is drawn with the landmark at the right.
        figure: This clock's profile-and-trajectory page.
        windows_figure: This clock's tested page.
        eligible_only: Whether the second-stage eligibility rule applies before binning.
    """

    name: str
    binner: Callable[[pd.DataFrame], pd.DataFrame]
    bin_column: str
    center_column: str
    axis_label: str
    inverted: bool
    figure: str
    windows_figure: str
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
        figure="lag_high_kl_time_to_delivery",
        windows_figure="lag_high_kl_time_to_delivery_windows",
        eligible_only=False,
    ),
    Clock(
        name="second_stage",
        binner=cohort.add_second_stage_bins,
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
        axis_label="Hours from second-stage onset (negative = before onset, positive = after)",
        inverted=False,
        figure="lag_high_kl_second_stage",
        windows_figure="lag_high_kl_second_stage_windows",
        eligible_only=True,
    ),
)


# =================================================================================================
# The anchor population and its thresholds
# =================================================================================================
@dataclass
class AnchorPopulation:
    """The scored anchors of the evaluated population, joined to their segments.

    Attributes:
        rows: Positions into the per-anchor table (and therefore into the sidecar), one per
            anchor that belongs to a segment of the population and carries a finite KL.
        sample_rows: For each of those anchors, the position of its segment in the per-sample
            table's row order.
        kl: Each anchor's $K_t$, in nats.
        argmax_lag: Each anchor's ``argmax_lag``, or ``-1`` where the column is absent.
        contraction_age: Each anchor's seconds since the most recent contraction, NaN where none.
        gain: Each anchor's forecast gain in nats, NaN where the table carries none.
        gain_column: Which of :data:`GAIN_COLUMNS` supplied it, or ``None``.
    """

    rows: np.ndarray
    sample_rows: np.ndarray
    kl: np.ndarray
    argmax_lag: np.ndarray
    contraction_age: np.ndarray
    gain: np.ndarray
    gain_column: Optional[str] = None


def anchor_population(per_sample: pd.DataFrame, per_anchor: pd.DataFrame) -> AnchorPopulation:
    """Join the per-anchor table onto the (horizon-bounded) per-sample table.

    Through ``sample_index``, which both tables carry and which the collection pass assigns in
    one place -- rather than through ``(guid, epoch)``, whose float equality is exact here but
    whose join is a second definition of the same identity.

    Args:
        per_sample: The per-sample table after the run's horizon has been applied.
        per_anchor: The per-anchor table.

    Returns:
        The population. Empty when either table lacks the columns the join needs.
    """
    empty = AnchorPopulation(
        rows=np.zeros(0, dtype=np.int64),
        sample_rows=np.zeros(0, dtype=np.int64),
        kl=np.zeros(0, dtype=np.float64),
        argmax_lag=np.zeros(0, dtype=np.int64),
        contraction_age=np.zeros(0, dtype=np.float64),
        gain=np.zeros(0, dtype=np.float64),
    )
    needed = {SAMPLE_INDEX_COLUMN, KL_COLUMN}
    if (
        per_sample.empty
        or per_anchor.empty
        or SAMPLE_INDEX_COLUMN not in per_sample.columns
        or not needed <= set(per_anchor.columns)
    ):
        return empty
    position = pd.Series(
        np.arange(len(per_sample), dtype=np.int64),
        index=np.asarray(per_sample[SAMPLE_INDEX_COLUMN], dtype=np.int64),
    )
    # A duplicated sample index would map one anchor onto two rows; the table is built so it
    # cannot happen, and the first is taken rather than raising over a table another guard owns.
    position = position[~position.index.duplicated(keep="first")]
    anchor_sample = np.asarray(per_anchor[SAMPLE_INDEX_COLUMN], dtype=np.int64)
    kl = np.asarray(per_anchor[KL_COLUMN], dtype=np.float64)
    mapped = position.reindex(anchor_sample).to_numpy()
    keep = np.isfinite(mapped) & np.isfinite(kl)
    rows = np.nonzero(keep)[0].astype(np.int64)
    argmax = (
        np.asarray(per_anchor[ARGMAX_COLUMN], dtype=np.int64)[rows]
        if ARGMAX_COLUMN in per_anchor.columns
        else np.full(rows.size, -1, dtype=np.int64)
    )
    ages = (
        np.asarray(per_anchor[CONTRACTION_AGE_COLUMN], dtype=np.float64)[rows]
        if CONTRACTION_AGE_COLUMN in per_anchor.columns
        else np.full(rows.size, np.nan, dtype=np.float64)
    )
    gain_column = next((name for name in GAIN_COLUMNS if name in per_anchor.columns), None)
    gain = (
        np.asarray(per_anchor[gain_column], dtype=np.float64)[rows]
        if gain_column is not None
        else np.full(rows.size, np.nan, dtype=np.float64)
    )
    return AnchorPopulation(
        rows=rows,
        sample_rows=mapped[rows].astype(np.int64),
        kl=kl[rows],
        argmax_lag=argmax,
        contraction_age=ages,
        gain=gain,
        gain_column=gain_column,
    )


def band_values(population: AnchorPopulation, band: AnchorBand) -> np.ndarray:
    """The per-anchor quantity one band is cut on."""
    return population.gain if band.on == "gain" else population.kl


def band_thresholds(
    population: AnchorPopulation, bands: Sequence[AnchorBand] = ANCHOR_BANDS
) -> Dict[str, Dict[str, Any]]:
    """Turn each band's quantile pair into a pair of thresholds, in nats, on the pooled sample.

    Args:
        population: The anchor population; each band is cut on the quantity :attr:`AnchorBand.on`
            names, pooled over every anchor that carries a finite value of it.
        bands: The bands.

    Returns:
        Band key to ``{on, q_lo, q_hi, lo_nats, hi_nats, n_pooled}``. NaN thresholds on an
        empty sample -- a table with no forecast-gain column leaves the ``gain`` band empty.
    """
    thresholds: Dict[str, Dict[str, Any]] = {}
    for band in bands:
        values = band_values(population, band)
        finite = values[np.isfinite(values)]
        lo = float(np.quantile(finite, band.q_lo)) if finite.size else float("nan")
        hi = float(np.quantile(finite, band.q_hi)) if finite.size else float("nan")
        thresholds[band.key] = {
            "on": band.on,
            "q_lo": float(band.q_lo),
            "q_hi": float(band.q_hi),
            "lo_nats": lo,
            "hi_nats": hi,
            "n_pooled": int(finite.size),
        }
    return thresholds


def band_mask(values: np.ndarray, band: AnchorBand, threshold: Dict[str, Any]) -> np.ndarray:
    """Which anchors fall in a band: ``lo <= v < hi``, upper edge inclusive at $q = 1$."""
    lo, hi = threshold["lo_nats"], threshold["hi_nats"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    upper = values <= hi if band.q_hi >= 1.0 else values < hi
    return finite & (values >= lo) & upper


# =================================================================================================
# Restricted profiles: the per-anchor map averaged over a selection, per segment
# =================================================================================================
#: Rows of the sidecar read per chunk. The maps are stored in half precision and widened to
#: float64 as they are read, so a whole real split at once would be several gigabytes.
_CHUNK_ROWS = 250_000


def restricted_profiles(
    lag_map: Any,
    population: AnchorPopulation,
    selected: np.ndarray,
    n_samples: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Average a per-anchor lag map over the selected anchors of each segment.

    $$\bar w_{i\ell} = \frac{\sum_{t \in S_i} u_t\, w_{t\ell}}{\sum_{t \in S_i} u_t},$$

    with $S_i$ the selected anchors of segment $i$ and $u_t \ge 0$ the per-anchor weight, which
    is $1$ unless ``weights`` is given. A segment with no selected anchor (or no positive weight)
    is a NaN row rather than a zero one -- "nothing selected here" and "the selection attends
    nowhere" are different statements, and the shape reducer drops NaN rows from every statistic.

    Args:
        lag_map: The sidecar's $(N, L)$ map, in any float dtype; read in chunks.
        population: The anchor population.
        selected: A boolean mask over the population's anchors.
        n_samples: Rows of the per-sample table, which is the first axis of the result.
        weights: Optional non-negative per-anchor weights over the population, for a weighted
            mean -- the gain-weighted attention profile is the one consumer. Non-finite weights
            count as zero.

    Returns:
        ``(profiles, counts)`` -- the $(n_{\mathrm{samples}}, L)$ mean map and the $(n,)$ total
        weight (the anchor count when unweighted) each row averaged over.
    """
    n_lags = int(np.asarray(lag_map).shape[1]) if np.asarray(lag_map).ndim == 2 else 0
    sums = np.zeros((int(n_samples), n_lags), dtype=np.float64)
    counts = np.zeros(int(n_samples), dtype=np.float64)
    if n_lags == 0 or not selected.any():
        return np.full_like(sums, np.nan), counts
    rows = population.rows[selected]
    sample_rows = population.sample_rows[selected]
    anchor_weight = (
        np.ones(rows.size, dtype=np.float64)
        if weights is None
        else np.where(np.isfinite(weights[selected]), np.maximum(weights[selected], 0.0), 0.0)
    )
    for start in range(0, rows.size, _CHUNK_ROWS):
        stop = start + _CHUNK_ROWS
        block = np.asarray(lag_map[rows[start:stop]], dtype=np.float64)
        # Non-finite bins are dropped from the mean rather than propagated: one unmeasured lag
        # must not blank a whole anchor's row.
        block = np.where(np.isfinite(block), block, 0.0) * anchor_weight[start:stop, None]
        targets = sample_rows[start:stop]
        # ``bincount`` per lag rather than ``np.add.at`` on the whole block: the latter is
        # unbuffered and at a real split's million-plus rows is minutes per call, while ninety-one
        # C-speed bincounts over the same rows are well under a second.
        for lag in range(n_lags):
            sums[:, lag] += np.bincount(targets, weights=block[:, lag], minlength=int(n_samples))
        counts += np.bincount(targets, weights=anchor_weight[start:stop], minlength=int(n_samples))
    profiles = np.divide(
        sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0.0
    )
    return profiles, counts


def hot_lag_set(pooled: np.ndarray, quantile: float = HOT_LAG_QUANTILE) -> Tuple[np.ndarray, float]:
    """The lags whose pooled attribution sits at or above ``quantile`` across the lag axis.

    Args:
        pooled: The pooled $(L,)$ attribution.
        quantile: The quantile across lags.

    Returns:
        ``(mask, threshold)`` -- the boolean lag mask and the nats threshold it was cut at. An
        all-``False`` mask with a NaN threshold when the profile carries no finite value.
    """
    finite = pooled[np.isfinite(pooled)]
    if finite.size == 0:
        return np.zeros(pooled.shape, dtype=bool), float("nan")
    threshold = float(np.quantile(finite, quantile))
    return np.isfinite(pooled) & (pooled >= threshold), threshold


def share_on(profiles: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Share of each row's finite mass that sits on the masked lags; NaN where a row has none."""
    finite = np.where(np.isfinite(profiles), profiles, 0.0)
    total = finite.sum(axis=1)
    inside = finite[:, mask].sum(axis=1) if mask.any() else np.zeros(total.shape)
    return np.divide(inside, total, out=np.full(total.shape, np.nan), where=total > 0.0)


# =================================================================================================
# The per-segment features
# =================================================================================================
def add_feature_columns(
    per_sample: pd.DataFrame,
    population: AnchorPopulation,
    maps: Dict[str, Any],
    thresholds: Dict[str, Dict[str, float]],
    seconds: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """Attach every per-segment feature to a copy of the per-sample table.

    Args:
        per_sample: The horizon-bounded per-sample table.
        population: The anchor population joined onto it.
        maps: The sidecar's maps by key; the attention map may be absent.
        thresholds: The bands' KL thresholds.
        seconds: The compensated lag axis.

    Returns:
        ``(frame, profiles, record)`` -- the copy carrying :data:`FEATURE_COLUMNS` and
        :data:`ROW_COLUMN`; the restricted profile matrices keyed ``<band>_<source>`` plus the
        all-anchor ``all_kl`` / ``all_attn`` ones the hot-lag readouts are built from; and the
        record of what each band selected, including the hot-lag set.
    """
    frame = per_sample.copy()
    frame[ROW_COLUMN] = np.arange(len(frame), dtype=np.int64)
    n_samples = len(frame)
    built: Dict[str, np.ndarray] = {}
    profiles: Dict[str, np.ndarray] = {}
    record: Dict[str, Any] = {"bands": {}, "sources": {}}

    # Per-segment anchor counts over the whole population, the denominator of every band share.
    total = np.zeros(n_samples, dtype=np.float64)
    np.add.at(total, population.sample_rows, 1.0)

    classes = (
        np.asarray(frame[labels.CLASS_COLUMN].astype(object))
        if labels.CLASS_COLUMN in frame.columns
        else np.full(n_samples, None, dtype=object)
    )
    anchor_classes = classes[population.sample_rows] if population.rows.size else classes[:0]
    # One string array over the anchors, compared once per class, rather than a Python
    # comprehension per band per class over a million anchors.
    anchor_class_names = np.asarray(
        [str(value) if value is not None else "" for value in anchor_classes], dtype=object
    )
    present_classes = cohort.ordered_groups(
        sorted({name for name in anchor_class_names if name}), labels.CLASS_COLUMN
    )

    available = [(key, attribute, meaning) for key, attribute, meaning in PROFILE_SOURCES
                 if attribute in maps]
    available_keys = {key for key, _, _ in available}
    for key, attribute, meaning in PROFILE_SOURCES:
        record["sources"][key] = {
            "attribute": attribute,
            "meaning": meaning,
            "available": attribute in maps,
        }

    masks: Dict[str, np.ndarray] = {}
    for band in ANCHOR_BANDS:
        selected = band_mask(band_values(population, band), band, thresholds[band.key])
        masks[band.key] = selected
        counts = np.bincount(
            population.sample_rows[selected], minlength=n_samples
        ).astype(np.float64)
        built[band_column(band.key, N_ANCHORS_SUFFIX)] = np.where(total > 0.0, counts, np.nan)
        built[band_column(band.key, ANCHOR_FRAC_SUFFIX)] = np.divide(
            counts, total, out=np.full(n_samples, np.nan), where=total > 0.0
        )
        # The forecast gain the source bought at the selected anchors, per segment: the mean of
        # the per-anchor gain over the band's anchors that carry one.
        with_gain = selected & np.isfinite(population.gain)
        gain_sum = np.bincount(
            population.sample_rows[with_gain], weights=population.gain[with_gain],
            minlength=n_samples,
        )
        gain_count = np.bincount(population.sample_rows[with_gain], minlength=n_samples)
        built[band_column(band.key, PRED_GAP_SUFFIX)] = np.divide(
            gain_sum, gain_count, out=np.full(n_samples, np.nan), where=gain_count > 0
        )
        by_class: Dict[str, Dict[str, Any]] = {}
        for group in present_classes:
            member = anchor_class_names == group
            n_group = int(member.sum())
            by_class[group] = {
                "n_anchors": n_group,
                "n_selected": int((member & selected).sum()),
                "share": float((member & selected).sum() / n_group) if n_group else float("nan"),
            }
        record["bands"][band.key] = {
            **thresholds[band.key],
            "meaning": band.meaning,
            "tested": band.tested,
            "n_anchors_selected": int(selected.sum()),
            "n_anchors_population": int(selected.size),
            "n_segments_with_selected_anchor": int((counts > 0.0).sum()),
            "share_by_class": by_class,
        }
        for key, attribute, _ in available:
            matrix, _ = restricted_profiles(maps[attribute], population, selected, n_samples)
            profiles[f"{band.key}_{key}"] = matrix
            statistics, census = profile_statistics(matrix, seconds)
            for statistic in STATISTIC_KEYS:
                built[feature_column(band.key, key, statistic)] = statistics[statistic]
            record["bands"][band.key][f"profile_{key}"] = census
        # A source the sidecar does not carry yields NaN columns rather than absent ones, so the
        # table's schema is the same on every run and a reader meets a blank rather than a
        # missing column.
        for key, _, _ in PROFILE_SOURCES:
            if key not in available_keys:
                for statistic in STATISTIC_KEYS:
                    built[feature_column(band.key, key, statistic)] = np.full(n_samples, np.nan)

    # The hot-lag set, from the pooled attribution: each segment's mean over ALL its anchors,
    # then the mean over every segment of the population, every class -- one selection for the
    # run, on the same per-segment unit every other pooled profile in the pipeline uses.
    everything = np.ones(population.kl.shape, dtype=bool)
    hot = np.zeros(int(seconds.size), dtype=bool)
    hot_threshold = float("nan")
    pooled = np.full(int(seconds.size), np.nan)
    for key, attribute, _ in available:
        matrix, _ = restricted_profiles(maps[attribute], population, everything, n_samples)
        profiles[f"all_{key}"] = matrix
    if "all_kl" in profiles and np.isfinite(profiles["all_kl"]).any():
        pooled = np.nanmean(profiles["all_kl"], axis=0)
        hot, hot_threshold = hot_lag_set(pooled)
    for key, _, _ in PROFILE_SOURCES:
        column = HOT_SHARE_COLUMNS[key]
        built[column] = (
            share_on(profiles[f"all_{key}"], hot)
            if f"all_{key}" in profiles and hot.any()
            else np.full(n_samples, np.nan)
        )
    # The gain-weighted attention profile: where the source looked, weighted by what it bought.
    # Positive gain only -- an anchor the source made worse is not evidence about where useful
    # information sat -- and over every anchor, so a segment whose source never helped is NaN.
    if ATTENTION_MAP_KEY in maps and np.isfinite(population.gain).any():
        profiles["gainw_attn"], _ = restricted_profiles(
            maps[ATTENTION_MAP_KEY], population, everything, n_samples, weights=population.gain
        )
    record["masks"] = masks
    record["hot_lags"] = {
        "quantile_across_lags": float(HOT_LAG_QUANTILE),
        "threshold_nats": hot_threshold,
        "n_lags": int(hot.sum()),
        "lag_steps": [int(index) for index in np.nonzero(hot)[0]],
        "compensated_seconds": [float(seconds[index]) for index in np.nonzero(hot)[0]],
        "pooled_attribution_nats": [float(value) for value in pooled],
        "selection_note": SELECTION_NOTE,
    }
    record["hot_mask"] = hot

    ordered = {column: built[column] for column in FEATURE_COLUMNS if column in built}
    frame = pd.concat([frame, pd.DataFrame(ordered, index=frame.index)], axis=1, copy=False)
    return frame, profiles, record


# =================================================================================================
# The argmax against the KL decile, and the contraction enrichment
# =================================================================================================
def argmax_by_quantile(
    population: AnchorPopulation,
    classes: np.ndarray,
    n_lags: int,
    seconds: np.ndarray,
    n_bins: int = N_KL_QUANTILE_BINS,
) -> Tuple[pd.DataFrame, np.ndarray, List[float]]:
    r"""Resolve where the per-anchor argmax sits, by equal-count bin of the pooled KL.

    Args:
        population: The anchor population.
        classes: Each anchor's clinical class, or ``None``.
        n_lags: The lag axis width.
        seconds: The compensated axis.
        n_bins: How many equal-count KL bins.

    Returns:
        ``(table, pooled_field, edges)`` -- the long-form table (one row per bin, lag and cohort,
        the pooled cohort named ``all``), the pooled $(n_{\mathrm{bins}}, L)$ share field the
        selection page draws, and the bin edges in nats.
    """
    columns = [
        "group", "kl_bin", "kl_lo_nats", "kl_hi_nats", "mean_kl_nats", "n_anchors",
        "lag_step", "compensated_seconds", "argmax_share",
    ]
    valid = population.argmax_lag >= 0
    if not valid.any() or population.kl.size == 0:
        return pd.DataFrame(columns=columns), np.zeros((0, n_lags)), []
    kl = population.kl[valid]
    argmax = np.clip(population.argmax_lag[valid], 0, n_lags - 1)
    group = np.asarray(
        [str(value) if value is not None else "" for value in classes[valid]], dtype=object
    )
    edges = [float(value) for value in np.quantile(kl, np.linspace(0.0, 1.0, int(n_bins) + 1))]
    bins = np.clip(np.searchsorted(edges, kl, side="right") - 1, 0, int(n_bins) - 1)
    rows: List[Dict[str, Any]] = []
    field = np.zeros((int(n_bins), n_lags), dtype=np.float64)
    cohorts = ["all"] + cohort.ordered_groups(
        sorted({name for name in group if name}), labels.CLASS_COLUMN
    )
    for name in cohorts:
        member = np.ones(kl.shape, dtype=bool) if name == "all" else (group == name)
        for index in range(int(n_bins)):
            cell = member & (bins == index)
            count = int(cell.sum())
            shares = (
                np.bincount(argmax[cell], minlength=n_lags).astype(np.float64) / count
                if count else np.full(n_lags, np.nan)
            )
            if name == "all":
                field[index] = shares
            for lag in range(n_lags):
                rows.append(
                    {
                        "group": name,
                        "kl_bin": index,
                        "kl_lo_nats": edges[index],
                        "kl_hi_nats": edges[index + 1],
                        "mean_kl_nats": float(kl[cell].mean()) if count else float("nan"),
                        "n_anchors": count,
                        "lag_step": lag,
                        "compensated_seconds": float(seconds[lag]),
                        "argmax_share": float(shares[lag]),
                    }
                )
    return pd.DataFrame(rows, columns=columns), field, edges


def gain_by_kl_quantile(
    population: AnchorPopulation,
    per_sample: pd.DataFrame,
    n_bins: int = N_KL_QUANTILE_BINS,
) -> Tuple[pd.DataFrame, List[float]]:
    """The forecast gain against the KL of the same anchor, by equal-count KL bin, per class.

    Per recording first -- each recording's mean gain in each bin -- and then the median and
    quartiles over recordings per (class, bin), so a recording contributing many anchors to a bin
    cannot outvote one contributing few. The pooled cohort is named ``all``.

    Args:
        population: The anchor population.
        per_sample: The horizon-bounded per-sample table, for each anchor's recording and class.
        n_bins: How many equal-count KL bins.

    Returns:
        ``(table, edges)`` -- one row per (cohort, bin) with the recording count, the mean and
        the quartiles of the per-recording mean gain and the pooled share of anchors with a
        positive gain; and the bin edges in nats. Empty when the table carries no gain.
    """
    columns = [
        "group", "kl_bin", "kl_lo_nats", "kl_hi_nats", "n_recordings", "n_recordings_total",
        "n_anchors", "mean_gain_nats", "q25", "median", "q75", "positive_anchor_share",
    ]
    finite = np.isfinite(population.gain)
    if not finite.any():
        return pd.DataFrame(columns=columns), []
    kl = population.kl[finite]
    gain = population.gain[finite]
    edges = [float(v) for v in np.quantile(kl, np.linspace(0.0, 1.0, int(n_bins) + 1))]
    bins = np.clip(np.searchsorted(edges, kl, side="right") - 1, 0, int(n_bins) - 1)
    guids = np.asarray(per_sample["guid"].astype(str))[population.sample_rows[finite]]
    classes = (
        np.asarray(per_sample[labels.CLASS_COLUMN].astype(object))[population.sample_rows[finite]]
        if labels.CLASS_COLUMN in per_sample.columns
        else np.full(guids.shape, None, dtype=object)
    )
    table = pd.DataFrame({"guid": guids, "group": [str(c) if c is not None else "" for c in classes],
                          "bin": bins, "gain": gain})
    rows: List[Dict[str, Any]] = []
    cohorts = ["all"] + cohort.ordered_groups(
        sorted({g for g in table["group"] if g}), labels.CLASS_COLUMN
    )
    for name in cohorts:
        subset = table if name == "all" else table[table["group"] == name]
        per_guid = subset.groupby(["bin", "guid"])["gain"].mean()
        n_total = int(subset["guid"].nunique())
        for index in range(int(n_bins)):
            values = (
                per_guid.loc[index].to_numpy(dtype=np.float64)
                if index in per_guid.index.get_level_values(0) else np.zeros(0)
            )
            anchors = subset[subset["bin"] == index]["gain"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "group": name,
                    "kl_bin": index,
                    "kl_lo_nats": edges[index],
                    "kl_hi_nats": edges[index + 1],
                    "n_recordings": int(values.size),
                    "n_recordings_total": n_total,
                    "n_anchors": int(anchors.size),
                    "mean_gain_nats": float(values.mean()) if values.size else float("nan"),
                    "q25": float(np.percentile(values, 25)) if values.size else float("nan"),
                    "median": float(np.percentile(values, 50)) if values.size else float("nan"),
                    "q75": float(np.percentile(values, 75)) if values.size else float("nan"),
                    "positive_anchor_share": (
                        float((anchors > 0.0).mean()) if anchors.size else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns), edges


def gain_by_argmax(
    population: AnchorPopulation,
    masks: Dict[str, np.ndarray],
    n_lags: int,
    seconds: np.ndarray,
) -> pd.DataFrame:
    """The forecast gain against the lag the anchor's attribution peaks at, pooled and per band.

    Args:
        population: The anchor population.
        masks: The band masks, keyed by band; the pooled selection is named ``all``.
        n_lags: The lag axis width.
        seconds: The compensated axis.

    Returns:
        One row per (selection, lag): the anchor count, the mean gain and the share of anchors
        with a positive gain. Empty when there is no gain or no argmax.
    """
    columns = ["selection", "lag_step", "compensated_seconds", "n_anchors", "mean_gain_nats",
               "positive_anchor_share"]
    usable = np.isfinite(population.gain) & (population.argmax_lag >= 0)
    if not usable.any():
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    selections = {"all": np.ones(usable.shape, dtype=bool), **masks}
    for name, mask in selections.items():
        member = usable & mask
        argmax = np.clip(population.argmax_lag[member], 0, n_lags - 1)
        gain = population.gain[member]
        counts = np.bincount(argmax, minlength=n_lags).astype(np.float64)
        sums = np.bincount(argmax, weights=gain, minlength=n_lags)
        positive = np.bincount(argmax, weights=(gain > 0.0).astype(np.float64), minlength=n_lags)
        for lag in range(n_lags):
            rows.append(
                {
                    "selection": name,
                    "lag_step": lag,
                    "compensated_seconds": float(seconds[lag]),
                    "n_anchors": int(counts[lag]),
                    "mean_gain_nats": float(sums[lag] / counts[lag]) if counts[lag] else float("nan"),
                    "positive_anchor_share": (
                        float(positive[lag] / counts[lag]) if counts[lag] else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def band_overlap(masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """How far the high-KL anchors and the high-gain anchors are the same anchors.

    Returns:
        The two counts, their intersection, the share of high anchors that are gain anchors, the
        share independence would give (the gain band's share of all anchors), and the Jaccard
        index. ``available`` is ``False`` when the gain band is empty.
    """
    high = masks.get(HIGH_BAND_KEY)
    gain = masks.get(GAIN_BAND_KEY)
    if high is None or gain is None or not gain.any() or not high.any():
        return {"available": False, "reason": "the gain band selected no anchor"}
    both = int((high & gain).sum())
    return {
        "available": True,
        "n_high": int(high.sum()),
        "n_gain": int(gain.sum()),
        "n_both": both,
        "share_of_high_in_gain": both / int(high.sum()),
        "share_expected_if_independent": float(gain.mean()),
        "jaccard": both / int((high | gain).sum()),
    }


def usefulness_test(recordings: pd.DataFrame, *, resamples: int, seed: int) -> Dict[str, Any]:
    """The high band's forecast gain against the rest band's, paired within recording.

    Args:
        recordings: The whole-recording table.
        resamples: Bootstrap resamples for the interval on the mean difference.
        seed: Seed for that bootstrap.

    Returns:
        The paired Wilcoxon record, the mean difference with its bootstrap interval over
        recordings, the positive fraction with its denominator, and ``tested``.
    """
    high_column = band_column(HIGH_BAND_KEY, PRED_GAP_SUFFIX)
    rest_column = band_column(REST_BAND_KEY, PRED_GAP_SUFFIX)
    base = {
        "left": high_column,
        "right": rest_column,
        "note": USEFULNESS_NOTE,
        "family": "its own family of one; not corrected with the four clock families",
    }
    if recordings.empty or high_column not in recordings.columns or rest_column not in recordings.columns:
        return {**base, "tested": False, "reason": "no per-recording gain columns"}
    high = recordings[high_column].to_numpy(dtype=np.float64)
    rest = recordings[rest_column].to_numpy(dtype=np.float64)
    paired = np.isfinite(high) & np.isfinite(rest)
    difference = high[paired] - rest[paired]
    if difference.size < shared_stats.MIN_GROUP_SIZE:
        return {
            **base, "tested": False, "n_pairs": int(difference.size),
            "reason": f"fewer than {shared_stats.MIN_GROUP_SIZE} recordings carry both bands' gain",
        }
    wilcoxon = shared_stats.wilcoxon_paired(
        high[paired], rest[paired], label_left=high_column, label_right=rest_column
    )
    interval = shared_stats.bootstrap_ci(difference, resamples=int(resamples), seed=int(seed))
    return {
        **base,
        "tested": True,
        "n_pairs": int(difference.size),
        "mean_difference_nats": float(difference.mean()),
        "median_difference_nats": float(np.median(difference)),
        "ci_lo": interval.get("lo"),
        "ci_hi": interval.get("hi"),
        "positive_fraction": float((difference > 0.0).mean()),
        "wilcoxon": wilcoxon,
    }


def occlusion_consistency(
    output_dir: Any,
    featured: pd.DataFrame,
    profiles: Dict[str, np.ndarray],
    bands: Dict[str, Tuple[int, int]],
    seconds: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Join each geometry band's attribution share against the forecast cost of occluding it.

    The observational reading and the interventional one on the same partition, per recording:
    the share of a recording's all-anchor KL attribution inside a band (and of its high band's),
    beside the ``occlusion`` analysis's per-recording total delta for that band. Read off disk,
    so ``--only lag_high_kl`` against a directory whose interventional pass never ran records a
    skip rather than raising.

    Args:
        output_dir: The results directory.
        featured: The featured per-sample table, carrying :data:`ROW_COLUMN`.
        profiles: The restricted profile matrices, in per-sample row order.
        bands: ``eval_config.occlusion_bands``.
        seconds: The compensated axis, for the row count.

    Returns:
        ``(table, record)`` -- one row per (recording, band) with both shares and the delta, and
        the record: per band, the recording count and Spearman's rho between the all-anchor share
        and the delta, descriptive only.
    """
    columns = ["guid", labels.CLASS_COLUMN, "band", "lag_lo", "lag_hi",
               "attribution_share_all", "attribution_share_high", "occlusion_delta_nats"]
    path = Path(output_dir) / OCCLUSION_PER_RECORDING[0] / OCCLUSION_PER_RECORDING[1]
    if not bands:
        return pd.DataFrame(columns=columns), {
            "available": False, "reason": "eval_config.occlusion_bands names no band"}
    if not path.is_file():
        return pd.DataFrame(columns=columns), {
            "available": False,
            "reason": f"{OCCLUSION_PER_RECORDING[0]}/{OCCLUSION_PER_RECORDING[1]} was not written; "
                      "the interventional pass needs a checkpoint and did not run in this directory",
        }
    occluded = pd.read_csv(path)
    if "guid" not in occluded.columns:
        occluded = occluded.rename(columns={occluded.columns[0]: "guid"})
    occluded["guid"] = occluded["guid"].astype(str)
    all_kl = profiles.get("all_kl")
    high_kl = profiles.get(f"{HIGH_BAND_KEY}_kl")
    if all_kl is None or "guid" not in featured.columns:
        return pd.DataFrame(columns=columns), {
            "available": False, "reason": "no all-anchor attribution profile to take a share from"}

    def _per_recording_share(matrix: Optional[np.ndarray], span: Tuple[int, int]) -> pd.Series:
        """Share of the attribution inside ``span``, per segment then averaged per recording."""
        if matrix is None:
            return pd.Series(dtype=np.float64)
        mask = np.zeros(int(seconds.size), dtype=bool)
        mask[max(span[0], 0): min(span[1], seconds.size - 1) + 1] = True
        share = share_on(matrix, mask)
        return pd.Series(share, index=featured["guid"].astype(str).to_numpy()).groupby(level=0).mean()

    identity = featured.drop_duplicates("guid")
    classes = pd.Series(
        identity[labels.CLASS_COLUMN].to_numpy() if labels.CLASS_COLUMN in identity.columns
        else [None] * len(identity),
        index=identity["guid"].astype(str).to_numpy(),
    )
    rows: List[Dict[str, Any]] = []
    record: Dict[str, Any] = {"available": True, "bands": {}, "tested": False}
    for name, span in bands.items():
        column = f"{OCCLUSION_DELTA_PREFIX}{name}{OCCLUSION_DELTA_SUFFIX}"
        if column not in occluded.columns:
            record["bands"][name] = {"available": False, "reason": f"{column} absent"}
            continue
        delta = occluded.set_index("guid")[column]
        share_all = _per_recording_share(all_kl, span)
        share_high = _per_recording_share(high_kl, span)
        joined = pd.concat(
            [share_all.rename("all"), share_high.rename("high"), delta.rename("delta")],
            axis=1, join="inner",
        )
        for guid, row in joined.iterrows():
            rows.append(
                {
                    "guid": str(guid),
                    labels.CLASS_COLUMN: classes.get(str(guid)),
                    "band": name,
                    "lag_lo": int(span[0]),
                    "lag_hi": int(span[1]),
                    "attribution_share_all": float(row["all"]),
                    "attribution_share_high": float(row["high"]),
                    "occlusion_delta_nats": float(row["delta"]),
                }
            )
        usable = joined[np.isfinite(joined["all"]) & np.isfinite(joined["delta"])]
        rho = float("nan")
        if len(usable) >= shared_stats.MIN_GROUP_SIZE:
            from scipy import stats as scipy_stats

            rho = float(scipy_stats.spearmanr(usable["all"], usable["delta"]).statistic)
        record["bands"][name] = {
            "available": True,
            "n_recordings": int(len(usable)),
            "spearman_rho_share_vs_delta": rho if np.isfinite(rho) else None,
            "mean_share_all": float(usable["all"].mean()) if len(usable) else None,
            "mean_delta_nats": float(usable["delta"].mean()) if len(usable) else None,
        }
    record["note"] = (
        "descriptive: Spearman's rho between a recording's share of KL attribution inside a "
        "geometry band and the forecast cost of occluding that band. Positive means the lags the "
        "attribution names are the lags the forecast used; near zero means the attribution and "
        "the intervention disagree about where the source mattered"
    )
    return pd.DataFrame(rows, columns=columns), record


def contraction_enrichment(
    population: AnchorPopulation,
    high: np.ndarray,
    per_sample: pd.DataFrame,
    window_s: float,
) -> pd.DataFrame:
    """Per recording: the high-anchor share within the contraction window against outside it.

    Args:
        population: The anchor population.
        high: The high band's membership over the population's anchors.
        per_sample: The horizon-bounded per-sample table, for the recording and cohort of each
            segment.
        window_s: ``eval_config.event_lag_window_s``.

    Returns:
        One row per recording: the two shares, the two anchor counts, their difference and
        whether both arms clear :data:`MIN_ENRICHMENT_ANCHORS`. Empty when the age column was
        never measured.
    """
    columns = [
        "guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "n_event_anchors",
        "n_control_anchors", "high_share_event", "high_share_control", "enrichment",
        "reportable",
    ]
    if population.rows.size == 0 or not np.isfinite(population.contraction_age).any():
        return pd.DataFrame(columns=columns)
    event = np.isfinite(population.contraction_age) & (population.contraction_age <= float(window_s))
    guids = np.asarray(per_sample["guid"].astype(str))[population.sample_rows]
    table = pd.DataFrame({"guid": guids, "event": event, "high": high})
    grouped = table.groupby("guid", sort=True)
    n_event = grouped["event"].sum()
    n_total = grouped["event"].size()
    high_event = table[table["event"]].groupby("guid")["high"].mean()
    high_control = table[~table["event"]].groupby("guid")["high"].mean()
    identity = (
        per_sample.drop_duplicates("guid").set_index(per_sample.drop_duplicates("guid")["guid"].astype(str))
        if "guid" in per_sample.columns else pd.DataFrame()
    )
    rows: List[Dict[str, Any]] = []
    for guid in n_total.index:
        n_e = int(n_event.loc[guid])
        n_c = int(n_total.loc[guid] - n_e)
        share_e = float(high_event.get(guid, np.nan))
        share_c = float(high_control.get(guid, np.nan))
        reportable = n_e >= MIN_ENRICHMENT_ANCHORS and n_c >= MIN_ENRICHMENT_ANCHORS
        rows.append(
            {
                "guid": str(guid),
                labels.CLASS_COLUMN: (
                    identity.loc[str(guid), labels.CLASS_COLUMN]
                    if labels.CLASS_COLUMN in identity.columns and str(guid) in identity.index
                    else None
                ),
                labels.SUBGROUP_COLUMN: (
                    identity.loc[str(guid), labels.SUBGROUP_COLUMN]
                    if labels.SUBGROUP_COLUMN in identity.columns and str(guid) in identity.index
                    else None
                ),
                "n_event_anchors": n_e,
                "n_control_anchors": n_c,
                "high_share_event": share_e,
                "high_share_control": share_c,
                "enrichment": share_e - share_c if reportable else float("nan"),
                "reportable": bool(reportable),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def enrichment_summary(table: pd.DataFrame) -> Dict[str, Any]:
    """Median and quartiles of the per-recording enrichment, by class, over reportable rows."""
    summary: Dict[str, Any] = {
        "min_anchors_per_arm": int(MIN_ENRICHMENT_ANCHORS),
        "n_recordings": int(len(table)),
        "n_reportable": int(table["reportable"].sum()) if len(table) else 0,
        "by_class": {},
        "tested": False,
    }
    if not len(table):
        return summary
    usable = table[table["reportable"].astype(bool) & table[labels.CLASS_COLUMN].notna()]
    for group in cohort.ordered_groups(
        sorted({str(value) for value in usable[labels.CLASS_COLUMN]}), labels.CLASS_COLUMN
    ):
        values = np.asarray(
            usable.loc[usable[labels.CLASS_COLUMN].astype(str) == group, "enrichment"],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        summary["by_class"][group] = {
            "n_recordings": int(values.size),
            "median": float(np.median(values)) if values.size else None,
            "q25": float(np.percentile(values, 25)) if values.size else None,
            "q75": float(np.percentile(values, 75)) if values.size else None,
            "positive_fraction": float((values > 0.0).mean()) if values.size else None,
        }
    return summary


# =================================================================================================
# The clocks
# =================================================================================================
def clock_rows(clock: Clock, featured: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bin the featured table on one clock, applying that clock's population rule first."""
    population: Dict[str, Any] = {"clock": clock.name, "n_segments_before": int(len(featured))}
    frame = featured
    if clock.eligible_only:
        eligibility = cohort.second_stage_eligibility(featured)
        keep = (
            set(eligibility.loc[eligibility["eligible"].astype(bool), "guid"].astype(str))
            if len(eligibility) else set()
        )
        frame = (
            featured[featured["guid"].astype(str).isin(sorted(keep))]
            if "guid" in featured.columns else featured.iloc[:0]
        )
        population["n_recordings_eligible"] = int(len(keep))
        population["eligibility_table"] = "second_stage/second_stage_eligibility.csv"
    binned = clock.binner(frame)
    population["n_segments"] = int(len(binned))
    population["n_windows"] = int(binned[clock.bin_column].nunique()) if len(binned) else 0
    return binned, population


def _class_values(frame: pd.DataFrame, column: str) -> Dict[str, np.ndarray]:
    """One window's rows split into per-class finite vectors, in the canonical severity order."""
    values_by_class: Dict[str, np.ndarray] = {}
    if frame.empty or "group" not in frame.columns or column not in frame.columns:
        return values_by_class
    for group in cohort.ordered_groups(
        sorted(set(frame["group"].astype(str))), labels.CLASS_COLUMN
    ):
        values = np.asarray(frame.loc[frame["group"].astype(str) == group, column], dtype=np.float64)
        values_by_class[group] = values[np.isfinite(values)]
    return values_by_class


def window_samples(
    clock: Clock, per_recording: pd.DataFrame, column: str
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, Any]]]:
    """The cells one readout is tested and drawn from -- one function, so the two agree."""
    samples: Dict[int, Dict[str, np.ndarray]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    if per_recording.empty or clock.bin_column not in getattr(per_recording, "columns", []):
        return samples, meta
    for window in sorted(int(value) for value in per_recording[clock.bin_column].unique()):
        cell = per_recording[per_recording[clock.bin_column] == window]
        values_by_class = _class_values(cell, column)
        samples[window] = values_by_class
        meta[window] = {
            "bin_center_h": float(cell[clock.center_column].iloc[0]),
            "groups_excluded_as_too_small": {
                group: int(values.size)
                for group, values in values_by_class.items()
                if values.size < shared_stats.MIN_GROUP_SIZE
            },
            "min_group_size": shared_stats.MIN_GROUP_SIZE,
        }
    return samples, meta


def analyse_windows(
    clock: Clock, per_recording: pd.DataFrame, column: str, *, alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """Test one readout's class trajectories window by window; Holm across the clock's windows."""
    groups = (
        cohort.ordered_groups(sorted(set(per_recording["group"].astype(str))), labels.CLASS_COLUMN)
        if not per_recording.empty and "group" in per_recording.columns else []
    )
    base: Dict[str, Any] = {
        "clock": clock.name,
        "metric_column": column,
        "group_column": labels.CLASS_COLUMN,
        "alpha": float(alpha),
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "method": METHOD,
    }
    if len(groups) < 2:
        return {
            **base,
            "tested": False,
            "reason": f"fewer than two clinical classes present ({groups or 'none'})",
            "per_window": [],
            "pairwise": {},
        }
    samples, meta = window_samples(clock, per_recording, column)
    testable = {
        window: {
            group: values for group, values in cell.items()
            if values.size >= shared_stats.MIN_GROUP_SIZE
        }
        for window, cell in samples.items()
    }
    outcome = shared_stats.windowed_group_comparisons(testable, meta_by_window=meta, alpha=alpha)
    significant = [record for record in outcome["per_window"] if record["significant"]]
    return {
        **base,
        "tested": True,
        "classes": groups,
        "n_windows": outcome["n_windows"],
        "n_windows_tested": outcome["n_windows_tested"],
        "n_significant_windows": len(significant),
        "significant_bin_centers_h": [record["bin_center_h"] for record in significant],
        "per_window": outcome["per_window"],
        "pairwise": outcome["pairwise"],
    }


def window_profiles(
    clock: Clock, binned: pd.DataFrame, matrix: Optional[np.ndarray], n_lags: int
) -> Tuple[List[int], List[float], List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]]:
    """Average a per-segment profile within each recording, then within each (class, window).

    Returns:
        ``(windows, centres, fields)`` with one ``(group, mean, share, counts, n_recordings)`` per
        class in the canonical order -- the last being the cohort's distinct recordings over the
        whole axis -- every class on the same window axis with NaN where it has no recording.
    """
    if matrix is None:
        return [], [], []
    values = np.asarray(matrix, dtype=np.float64)
    needed = {labels.CLASS_COLUMN, "guid", clock.bin_column, clock.center_column, ROW_COLUMN}
    if binned.empty or values.ndim != 2 or values.shape[1] != int(n_lags) or not needed <= set(binned.columns):
        return [], [], []
    frame = binned[binned[labels.CLASS_COLUMN].notna()]
    if frame.empty:
        return [], [], []
    positions = np.asarray(frame[ROW_COLUMN], dtype=np.int64)
    columns = list(range(int(n_lags)))
    table = pd.DataFrame(values[positions], columns=columns)
    table["group"] = [str(value) for value in frame[labels.CLASS_COLUMN]]
    table["window"] = [int(value) for value in frame[clock.bin_column]]
    table["centre"] = [float(value) for value in frame[clock.center_column]]
    table["guid"] = list(frame["guid"].astype(str))
    per_guid = table.groupby(["group", "window", "guid"], sort=True)[columns].mean()
    per_cell = per_guid.groupby(["group", "window"], sort=True)[columns].mean()
    counted = per_guid.notna().any(axis=1).groupby(["group", "window"], sort=True).sum()
    # Every window between the first and the last, so an empty interior window is a blank
    # column rather than a shift of every later column onto the wrong hour.
    lowest, highest = int(table["window"].min()), int(table["window"].max())
    windows = list(range(lowest, highest + 1))
    centres = [(window + 0.5) * float(TRAJECTORY_BIN_HOURS) for window in windows]
    fields: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for group in cohort.ordered_groups(sorted(set(table["group"])), labels.CLASS_COLUMN):
        mean = np.full((int(n_lags), len(windows)), np.nan)
        counts = np.zeros(len(windows), dtype=np.int64)
        for index, window in enumerate(windows):
            if (group, window) in per_cell.index:
                mean[:, index] = np.asarray(per_cell.loc[(group, window)], dtype=np.float64)
                counts[index] = int(counted.loc[(group, window)])
        totals = np.nansum(mean, axis=0)
        share = np.divide(
            mean, np.where(totals > 0.0, totals, np.nan)[None, :],
            out=np.full_like(mean, np.nan), where=np.isfinite(mean),
        )
        fields.append(
            (
                group, mean, share, counts,
                int(per_guid.loc[group].index.get_level_values("guid").nunique()),
            )
        )
    return windows, centres, fields


def profile_frame(
    clock: Clock,
    band_key: str,
    windows: Sequence[int],
    centres: Sequence[float],
    fields: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]],
    seconds: np.ndarray,
) -> pd.DataFrame:
    """Lay the per-cell mean restricted profiles out long-form: one row per (class, window, lag)."""
    rows: List[Dict[str, Any]] = []
    for group, mean, share, counts, _ in fields:
        for index, window in enumerate(windows):
            for lag in range(int(seconds.size)):
                rows.append(
                    {
                        "clock": clock.name,
                        "band": band_key,
                        "group_column": labels.CLASS_COLUMN,
                        "group": group,
                        "time_bin": int(window),
                        "bin_center_h": float(centres[index]),
                        "n_recordings": int(counts[index]),
                        "lag_step": int(lag),
                        "compensated_seconds": float(seconds[lag]),
                        "mean_nats": float(mean[lag, index]),
                        "share": float(share[lag, index]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "clock", "band", "group_column", "group", "time_bin", "bin_center_h", "n_recordings",
            "lag_step", "compensated_seconds", "mean_nats", "share",
        ],
    )


def significance_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the per-window omnibus results of every (clock, readout) into one table."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        for window in record.get("per_window") or []:
            rows.append(
                {
                    "clock": record["clock"],
                    "metric_column": record["metric_column"],
                    "time_bin": window["time_bin"],
                    "bin_center_h": window["bin_center_h"],
                    "n_classes": window.get("n_groups"),
                    "n_recordings": sum((window.get("n_per_group") or {}).values()),
                    "statistic": window.get("statistic"),
                    "p_value": window.get("p_value"),
                    "p_holm": window.get("p_holm", float("nan")),
                    "significant": window.get("significant", False),
                    "alpha": window.get("alpha", float("nan")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "clock", "metric_column", "time_bin", "bin_center_h", "n_classes", "n_recordings",
            "statistic", "p_value", "p_holm", "significant", "alpha",
        ],
    )


def pairwise_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the surviving windows' pairwise comparisons into one table."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        centres = {int(w["time_bin"]): w["bin_center_h"] for w in record.get("per_window") or []}
        for key, comparisons in (record.get("pairwise") or {}).items():
            for item in comparisons:
                rows.append(
                    {
                        "clock": record["clock"],
                        "metric_column": record["metric_column"],
                        "time_bin": int(key),
                        "bin_center_h": centres.get(int(key), float("nan")),
                        "left": item["left"],
                        "right": item["right"],
                        "n_left": item["n_left"],
                        "n_right": item["n_right"],
                        "p_value": item["p_value"],
                        "cliffs_delta": item["cliffs_delta"],
                        "magnitude": item["magnitude"],
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "clock", "metric_column", "time_bin", "bin_center_h", "left", "right", "n_left",
            "n_right", "p_value", "cliffs_delta", "magnitude",
        ],
    )


# =================================================================================================
# The figures
# =================================================================================================
def _empty_panel(ax: Any, title: str) -> None:
    """Mark a panel that has nothing to draw, rather than leaving blank axes."""
    ax.text(
        0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
        ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
    )
    ax.set_title(title)
    figures.style_axes(ax)


def build_selection_figure(
    population: AnchorPopulation,
    classes: np.ndarray,
    guids: np.ndarray,
    thresholds: Dict[str, Dict[str, float]],
    profiles: Dict[str, np.ndarray],
    hot: np.ndarray,
    field: np.ndarray,
    edges: Sequence[float],
    enrichment: pd.DataFrame,
    seconds: np.ndarray,
) -> Any:
    r"""Draw the run-level selection: what was chosen, and what it looks like pooled.

    Four panels. The pooled per-anchor KL distribution per class on a $\log_{10}$ axis with the
    band thresholds marked -- the histogram the bands were cut on; the pooled restricted lag
    profiles of the three bands with the hot-lag set shaded; the argmax share by KL decile and
    lag; and the per-recording contraction enrichment of high anchors, by class.

    Args:
        population: The anchor population.
        classes: Each anchor's class, or ``None``.
        guids: Each per-sample row's recording, in per-sample row order, so a count on this page
            can be stated in deliveries rather than in anchors or segments.
        thresholds: The bands' thresholds.
        profiles: The restricted profile matrices.
        hot: The hot-lag mask.
        field: The pooled argmax-share field, $(n_{\mathrm{bins}}, L)$.
        edges: The KL bin edges.
        enrichment: The per-recording enrichment table.
        seconds: The compensated axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(4, height_per_row=3.0)

    # --- Panel 1: the pooled KL, by class, and the thresholds -----------------------------------
    ax = axes[0, 0]
    positive = population.kl > 0.0
    if positive.any():
        log_kl = np.log10(population.kl[positive])
        groups = cohort.ordered_groups(
            sorted({str(value) for value in classes[positive] if value is not None}),
            labels.CLASS_COLUMN,
        )
        colours = figures.group_colors(groups)
        bins = np.linspace(float(log_kl.min()), float(log_kl.max()) + 1e-9, 60)
        anchor_guids = np.asarray(guids, dtype=object)[population.sample_rows[positive]]
        for group in groups:
            member = np.asarray([str(value) == group for value in classes[positive]], dtype=bool)
            ax.hist(
                log_kl[member], bins=bins, density=True, histtype="step",
                color=colours.get(group, figures.COLOR_BLUE), linewidth=figures.LINE_REGULAR,
                label=(
                    f"{group} (n={len(set(anchor_guids[member]))} deliveries, "
                    f"{int(member.sum())} anchors)"
                ),
            )
        for band in ANCHOR_BANDS:
            # Only the bands cut on the KL belong on a KL histogram.
            lo = thresholds[band.key]["lo_nats"]
            if band.on == "kl" and np.isfinite(lo) and lo > 0.0 and band.q_lo > 0.0:
                ax.axvline(
                    np.log10(lo), color=figures.COLOR_GRAY, linestyle="--",
                    linewidth=figures.LINE_THIN,
                )
                ax.annotate(
                    f"{band.key}: q{band.q_lo:g} = {lo:.3g} nats", (np.log10(lo), 0.0),
                    xycoords=("data", "axes fraction"), xytext=(2, 4), textcoords="offset points",
                    fontsize=figures.FONT_TINY, color=figures.COLOR_GRAY, rotation=90,
                    va="bottom",
                )
        ax.set_title("pooled per-anchor KL, by clinical class, with the band thresholds")
        ax.set_xlabel("log10 K_t (nats per anchor)")
        ax.set_ylabel("density of anchors")
        ax.legend(fontsize=figures.FONT_LABEL, loc="best")
        figures.style_axes(ax)
    else:
        _empty_panel(ax, "pooled per-anchor KL")

    # --- Panel 2: the pooled restricted profiles and the hot lags -------------------------------
    ax = axes[1, 0]
    drawn = 0
    if hot.any():
        for index in np.nonzero(hot)[0]:
            ax.axvspan(
                float(seconds[index]) - SECONDS_PER_LAG_STEP / 2.0,
                float(seconds[index]) + SECONDS_PER_LAG_STEP / 2.0,
                color=figures.COLOR_LIGHT_GRAY, alpha=0.5, linewidth=0,
            )
    styles = {"high": (figures.COLOR_VERMILLION, "-"), "top": (figures.COLOR_ORANGE, "-"),
              "rest": (figures.COLOR_BLUE, "--")}
    for band in ANCHOR_BANDS:
        matrix = profiles.get(f"{band.key}_kl")
        if matrix is None or not np.isfinite(matrix).any():
            continue
        colour, style = styles.get(band.key, (figures.COLOR_GRAY, "-"))
        with_rows = np.isfinite(matrix).any(axis=1)
        ax.plot(
            seconds, np.nanmean(matrix, axis=0), color=colour, linestyle=style,
            linewidth=figures.LINE_EMPHASIS,
            label=(
                f"{band.key} anchors (n={len(set(np.asarray(guids, dtype=object)[with_rows]))} "
                f"deliveries)"
            ),
        )
        drawn += 1
    if drawn:
        ax.set_title(
            "pooled KL attribution per lag, by anchor band; shaded = hot lags "
            f"(upper {100 * (1 - HOT_LAG_QUANTILE):.0f}% of pooled attribution across lags)"
        )
        ax.set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("nats per anchor")
        ax.legend(fontsize=figures.FONT_LABEL, loc="best")
        figures.style_axes(ax)
    else:
        _empty_panel(ax, "pooled KL attribution per lag, by anchor band")

    # --- Panel 3: the argmax by KL decile -------------------------------------------------------
    ax = axes[2, 0]
    if field.size and np.isfinite(field).any():
        extent = (
            float(seconds[0]) - SECONDS_PER_LAG_STEP / 2.0,
            float(seconds[-1]) + SECONDS_PER_LAG_STEP / 2.0,
            -0.5, field.shape[0] - 0.5,
        )
        figures.heatmap_with_colorbar(
            figure, ax, field[::-1], title=(
                "share of anchors whose KL attribution peaks at each lag, by KL decile "
                "(bottom row = lowest KL)"
            ),
            xlabel=figures.COEFFICIENT_LAG_AXIS_LABEL, ylabel="KL decile (0 = lowest)",
            symmetric=False, colorbar_label="share of the decile's anchors", extent=extent,
            interpolation="none",
        )
        # ``field[::-1]`` under ``origin='upper'`` already puts decile 0 at the bottom, where
        # tick 0 sits; the labels are therefore in natural order, not reversed a second time.
        ax.set_yticks(range(field.shape[0]))
        ax.set_yticklabels([
            f"{index}: {edges[index]:.2g}-{edges[index + 1]:.2g}"
            for index in range(field.shape[0])
        ], fontsize=figures.FONT_TINY)
    else:
        _empty_panel(ax, "argmax lag by KL decile")

    # --- Panel 4: the contraction enrichment ----------------------------------------------------
    ax = axes[3, 0]
    usable = (
        enrichment[enrichment["reportable"].astype(bool) & enrichment[labels.CLASS_COLUMN].notna()]
        if len(enrichment) else enrichment
    )
    if len(usable):
        groups = cohort.ordered_groups(
            sorted({str(value) for value in usable[labels.CLASS_COLUMN]}), labels.CLASS_COLUMN
        )
        samples = {
            group: np.asarray(
                usable.loc[usable[labels.CLASS_COLUMN].astype(str) == group, "enrichment"],
                dtype=np.float64,
            )
            for group in groups
        }
        figures.violin_panel(
            ax, samples, title=(
                "high-anchor share within the contraction window minus outside it, per recording "
                f"(>= {MIN_ENRICHMENT_ANCHORS} anchors in each arm)"
            ),
            ylabel="enrichment (share difference)", colors=figures.group_colors(groups),
            reference=0.0, reference_label="no enrichment",
        )
        figures.style_axes(ax)
    else:
        _empty_panel(ax, "contraction enrichment of high-KL anchors")

    figures.caveat_note(figure)
    return figure


def build_usefulness_figure(
    gain_table: pd.DataFrame,
    argmax_gain: pd.DataFrame,
    profiles: Dict[str, np.ndarray],
    overlap: Dict[str, Any],
    usefulness: Dict[str, Any],
    consistency: pd.DataFrame,
    seconds: np.ndarray,
) -> Any:
    r"""Draw whether the lags the model attends to are lags it forecasts better from.

    Four panels. The forecast gain by KL decile, per class, as the median over recordings with
    its inter-quartile ribbon and zero marked; the mean gain by the anchor's argmax lag for every
    anchor and for the high and gain bands; the pooled lag profiles as shares -- the all-anchor
    KL attribution, the gain-weighted attention and the gain band's attribution -- with the
    overlap statement in the legend; and the observational-against-interventional scatter per
    geometry band, or the note saying the interventional pass did not run.

    Args:
        gain_table: :func:`gain_by_kl_quantile`'s table.
        argmax_gain: :func:`gain_by_argmax`'s table.
        profiles: The restricted profile matrices.
        overlap: :func:`band_overlap`'s record.
        usefulness: :func:`usefulness_test`'s record.
        consistency: :func:`occlusion_consistency`'s table.
        seconds: The compensated axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(4, height_per_row=3.0)

    # --- Panel 1: gain by KL decile, per class ---------------------------------------------------
    ax = axes[0, 0]
    populated = gain_table[gain_table["n_recordings"] > 0] if len(gain_table) else gain_table
    if len(populated):
        groups = cohort.ordered_groups(
            sorted({g for g in populated["group"] if g != "all"}), labels.CLASS_COLUMN
        )
        colours = figures.group_colors(groups)
        for group in ["all"] + groups:
            cell = populated[populated["group"] == group].sort_values("kl_bin")
            if not len(cell):
                continue
            colour = figures.COLOR_BLACK if group == "all" else colours.get(group, figures.COLOR_BLUE)
            x = cell["kl_bin"].to_numpy(dtype=np.float64)
            ax.fill_between(x, cell["q25"], cell["q75"], color=colour, alpha=0.12, linewidth=0)
            ax.plot(
                x, cell["median"], marker="o", markersize=3, color=colour,
                linewidth=figures.LINE_EMPHASIS if group == "all" else figures.LINE_REGULAR,
                label=f"{group} (n={int(cell['n_recordings_total'].iloc[0])} deliveries)",
            )
        ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR)
        headline = usefulness if usefulness.get("tested") else {}
        title = "forecast gain (D_base - D_full) by KL decile of the same anchor: median per recording"
        if headline:
            title += (
                f"; high - rest = {headline['mean_difference_nats']:+.3g} nats, "
                f"p = {headline['wilcoxon'].get('p_value', float('nan')):.2g}, "
                f"positive in {headline['positive_fraction']:.0%} of {headline['n_pairs']} recordings"
            )
        ax.set_title(title)
        ax.set_xlabel("KL decile (0 = lowest K_t)")
        ax.set_ylabel("nats per anchor")
        ax.legend(fontsize=figures.FONT_LABEL, loc="best", ncol=2)
        figures.style_axes(ax)
    else:
        _empty_panel(ax, "forecast gain by KL decile (the per-anchor table carries no gain)")

    # --- Panel 2: gain by argmax lag -------------------------------------------------------------
    ax = axes[1, 0]
    drawn = 0
    styles = {"all": (figures.COLOR_BLACK, "-"), HIGH_BAND_KEY: (figures.COLOR_VERMILLION, "-"),
              GAIN_BAND_KEY: (figures.COLOR_GREEN, "-"), REST_BAND_KEY: (figures.COLOR_BLUE, "--")}
    for name, (colour, style) in styles.items():
        cell = argmax_gain[argmax_gain["selection"] == name] if len(argmax_gain) else argmax_gain
        if not len(cell) or not np.isfinite(cell["mean_gain_nats"]).any():
            continue
        ax.plot(
            cell["compensated_seconds"], cell["mean_gain_nats"], color=colour, linestyle=style,
            linewidth=figures.LINE_REGULAR, label=f"{name} anchors (n={int(cell['n_anchors'].sum())})",
        )
        drawn += 1
    if drawn:
        ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR)
        ax.set_title("mean forecast gain by the lag the anchor's KL attribution peaks at")
        ax.set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("nats per anchor")
        ax.legend(fontsize=figures.FONT_LABEL, loc="best", ncol=2)
        figures.style_axes(ax)
    else:
        _empty_panel(ax, "forecast gain by argmax lag")

    # --- Panel 3: where the source looks when it helps -------------------------------------------
    ax = axes[2, 0]
    drawn = 0
    for key, label, colour, style in (
        ("all_kl", "KL attribution, all anchors", figures.COLOR_BLACK, "-"),
        ("gainw_attn", "attention weighted by positive forecast gain", figures.COLOR_GREEN, "-"),
        (f"{GAIN_BAND_KEY}_kl", "KL attribution, gain band", figures.COLOR_GREEN, "--"),
        (f"{HIGH_BAND_KEY}_kl", "KL attribution, high band", figures.COLOR_VERMILLION, "--"),
    ):
        matrix = profiles.get(key)
        if matrix is None or not np.isfinite(matrix).any():
            continue
        pooled = np.nanmean(matrix, axis=0)
        total = np.nansum(pooled)
        if not np.isfinite(total) or total <= 0.0:
            continue
        ax.plot(seconds, pooled / total, color=colour, linestyle=style,
                linewidth=figures.LINE_REGULAR, label=label)
        drawn += 1
    if drawn:
        title = "pooled lag profiles as shares: where the source looks, and where it looks when it helps"
        if overlap.get("available"):
            title += (
                f"; {overlap['share_of_high_in_gain']:.0%} of high anchors are gain anchors "
                f"(independence: {overlap['share_expected_if_independent']:.0%})"
            )
        ax.set_title(title)
        ax.set_xlabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share of the profile")
        ax.legend(fontsize=figures.FONT_LABEL, loc="best")
        figures.style_axes(ax)
    else:
        _empty_panel(ax, "pooled lag profiles")

    # --- Panel 4: observational against interventional ------------------------------------------
    ax = axes[3, 0]
    usable = (
        consistency[np.isfinite(consistency["attribution_share_all"])
                    & np.isfinite(consistency["occlusion_delta_nats"])]
        if len(consistency) else consistency
    )
    if len(usable):
        bands = list(dict.fromkeys(usable["band"]))
        palette = [figures.COLOR_BLUE, figures.COLOR_ORANGE, figures.COLOR_PURPLE,
                   figures.COLOR_GREEN, figures.COLOR_VERMILLION, figures.COLOR_GRAY]
        for index, name in enumerate(bands):
            cell = usable[usable["band"] == name]
            ax.scatter(
                cell["attribution_share_all"], cell["occlusion_delta_nats"], s=9,
                color=palette[index % len(palette)], alpha=0.7,
                label=f"{name} (lags {int(cell['lag_lo'].iloc[0])}-{int(cell['lag_hi'].iloc[0])}, "
                      f"n={len(cell)} deliveries)",
            )
        ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR)
        ax.set_title(
            "per recording: share of KL attribution inside a geometry band against the forecast "
            "cost of occluding that band"
        )
        ax.set_xlabel("share of the recording's KL attribution in the band")
        ax.set_ylabel("occlusion delta (nats per anchor)")
        ax.legend(fontsize=figures.FONT_LABEL, loc="best")
        figures.style_axes(ax)
    else:
        _empty_panel(
            ax, "attribution share against occlusion cost (the interventional pass did not run here)"
        )

    figures.caveat_note(figure)
    return figure


def _draw_trajectory_panel(
    ax: Any,
    clock: Clock,
    rows: Sequence[Dict[str, Any]],
    column: str,
    *,
    companion: Optional[str],
    ylabel: str,
    title: str,
) -> int:
    """Draw one readout's class trajectories -- median with its inter-quartile ribbon -- and,
    dashed, a companion column's median on the same axis."""
    selected = [
        row for row in rows
        if row["group_column"] == labels.CLASS_COLUMN and row["clock"] == clock.name
    ]
    primary = [row for row in selected if row["metric"] == column]
    secondary = [row for row in selected if companion and row["metric"] == companion]
    groups = cohort.ordered_groups([row["group"] for row in primary], labels.CLASS_COLUMN) if primary else []
    if not groups:
        _empty_panel(ax, title)
        return 0
    colours = figures.group_colors(groups)
    labelled = False
    for group in groups:
        cell = sorted((row for row in primary if row["group"] == group), key=lambda r: r["bin_center_h"])
        if not cell:
            continue
        x = np.array([row["bin_center_h"] for row in cell], dtype=np.float64)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.fill_between(
            x, np.array([row["q25"] for row in cell]), np.array([row["q75"] for row in cell]),
            color=colour, alpha=0.15, linewidth=0,
        )
        ax.plot(
            x, np.array([row["median"] for row in cell]), marker="o", markersize=3, color=colour,
            linewidth=figures.LINE_EMPHASIS,
            label=f"{group} (n={int(cell[0].get('n_recordings_total', 0))} deliveries)",
        )
        for row in cell:
            ax.annotate(
                str(int(row["n_recordings"])), (float(row["bin_center_h"]), float(row["median"])),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=figures.FONT_TINY, color=colour,
            )
        other = sorted((row for row in secondary if row["group"] == group), key=lambda r: r["bin_center_h"])
        if other:
            ax.plot(
                np.array([row["bin_center_h"] for row in other]),
                np.array([row["median"] for row in other]),
                linestyle="--", color=colour, linewidth=figures.LINE_THIN,
                label=f"{companion} (dashed)" if not labelled else "_nolegend_",
            )
            labelled = True
    ax.set_title(title)
    ax.set_xlabel(clock.axis_label)
    ax.set_ylabel(ylabel)
    if clock.inverted:
        ax.invert_xaxis()
    else:
        ax.axvline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR, zorder=0)
    ax.legend(fontsize=figures.FONT_LABEL, loc="best", ncol=2)
    figures.style_axes(ax)
    return len(groups)


def build_clock_figure(
    clock: Clock,
    windows: Sequence[int],
    centres: Sequence[float],
    fields: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]],
    rows: Sequence[Dict[str, Any]],
    seconds: np.ndarray,
) -> Any:
    """One clock's page: the high band's lag share by window per class, then four trajectories.

    The heatmaps share one colour scale across the classes, for the reason ``lag_clocks`` gives:
    three panels each scaled to their own extremes paint one colour for three different shares.
    The trajectories are the high-anchor share, the high band's KL centroid with the rest band's
    dashed beside it, and the hot-lag share of the attribution.
    """
    n_trajectories = 4
    figure, axes = figures.new_figure(max(len(fields), 1) + n_trajectories, height_per_row=3.0)
    limit = max(
        (float(np.nanmax(share)) for _, _, share, _, _ in fields if np.isfinite(share).any()),
        default=0.0,
    )
    half = float(TRAJECTORY_BIN_HOURS) / 2.0
    half_lag = SECONDS_PER_LAG_STEP / 2.0
    extent = (
        (float(centres[0]) - half, float(centres[-1]) + half,
         float(seconds[0]) - half_lag, float(seconds[-1]) + half_lag)
        if centres and seconds.size else None
    )
    for index, (group, _, share, _, n_recordings) in enumerate(fields):
        ax = axes[index, 0]
        figures.heatmap_with_colorbar(
            figure, ax, share[::-1],
            title=(
                f"{group}: share of the HIGH-band KL attribution by lag and window "
                f"(n={int(n_recordings)} deliveries)"
            ),
            ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL, symmetric=False,
            vlimits=(0.0, limit) if limit > 0.0 else None,
            colorbar_label="share of the attribution", extent=extent, interpolation="none",
        )
        if clock.inverted:
            ax.invert_xaxis()
        else:
            ax.axvline(0.0, color=figures.COLOR_LIGHT_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR)
        ax.set_xlabel(clock.axis_label)
    if not fields:
        _empty_panel(axes[0, 0], "share of the high-band KL attribution by lag and window")

    base = max(len(fields), 1)
    high_frac = band_column("high", ANCHOR_FRAC_SUFFIX)
    high_centroid = feature_column("high", "kl", "centroid")
    rest_centroid = feature_column("rest", "kl", "centroid")
    _draw_trajectory_panel(
        axes[base, 0], clock, rows, high_frac, companion=band_column("top", ANCHOR_FRAC_SUFFIX),
        ylabel="share of the segment's anchors",
        title=f"{high_frac} against the clock, by {labels.CLASS_COLUMN} (tested)",
    )
    _draw_trajectory_panel(
        axes[base + 1, 0], clock, rows, high_centroid, companion=rest_centroid,
        ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        title=f"{high_centroid} against the clock, by {labels.CLASS_COLUMN} (tested)",
    )
    _draw_trajectory_panel(
        axes[base + 2, 0], clock, rows, HOT_SHARE_COLUMNS["kl"], companion=HOT_SHARE_COLUMNS["attn"],
        ylabel="share of the attribution on the hot lags",
        title=f"{HOT_SHARE_COLUMNS['kl']} against the clock, by {labels.CLASS_COLUMN} (untested)",
    )
    high_gain = band_column(HIGH_BAND_KEY, PRED_GAP_SUFFIX)
    _draw_trajectory_panel(
        axes[base + 3, 0], clock, rows, high_gain, companion=band_column(REST_BAND_KEY, PRED_GAP_SUFFIX),
        ylabel="forecast gain (nats per anchor)",
        title=f"{high_gain} against the clock, by {labels.CLASS_COLUMN} (untested per window)",
    )
    figures.caveat_note(figure)
    return figure


def build_windows_figure(
    clock: Clock, class_frame: pd.DataFrame, records: Sequence[Dict[str, Any]]
) -> Any:
    """Draw what the two tested trajectories are made of: the cells, their $p$ and the effects."""
    present = (
        sorted(set(class_frame["group"].astype(str)))
        if len(class_frame) and "group" in class_frame.columns else []
    )
    readouts = []
    for column, record in zip(READOUTS, records):
        samples, _ = window_samples(clock, class_frame, column)
        order = [int(row["time_bin"]) for row in record.get("per_window") or []]
        readouts.append((column, [samples.get(key, {}) for key in order], record))
    figure = figures.windowed_comparison_figure(
        readouts,
        groups=cohort.ordered_groups(present, labels.CLASS_COLUMN),
        bin_width=TRAJECTORY_BIN_HOURS,
        min_body_size=shared_stats.MIN_GROUP_SIZE,
        xlabel=clock.axis_label,
        ylabel="value (seconds for the centroid; share of anchors for the fraction)",
        delivery_orientation=clock.inverted,
    )
    figures.caveat_note(figure)
    return figure


# =================================================================================================
# The whole-recording table and the headline
# =================================================================================================
#: The whole-recording columns: the high and top shares, the high band's centroid and total, and
#: the hot-lag share. ``cross_subgroup`` reads the first of them.
RECORDING_COLUMNS: Tuple[str, ...] = (
    band_column("high", ANCHOR_FRAC_SUFFIX),
    band_column("top", ANCHOR_FRAC_SUFFIX),
    band_column("gain", ANCHOR_FRAC_SUFFIX),
    feature_column("high", "kl", "centroid"),
    feature_column("high", "kl", "total_nats"),
    feature_column("rest", "kl", "centroid"),
    feature_column("gain", "kl", "centroid"),
    band_column("high", PRED_GAP_SUFFIX),
    band_column("rest", PRED_GAP_SUFFIX),
    band_column("gain", PRED_GAP_SUFFIX),
    HOT_SHARE_COLUMNS["kl"],
)


def recordings_frame(featured: pd.DataFrame) -> pd.DataFrame:
    """One row per recording over the whole evaluated population, no clock.

    The unweighted mean over a recording's segments, which is the chain's middle step and the unit
    every cohort test in the run consumes.
    """
    present = [column for column in RECORDING_COLUMNS if column in featured.columns]
    keys = ["guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN]
    if featured.empty or "guid" not in featured.columns or not present:
        return pd.DataFrame(columns=keys + list(RECORDING_COLUMNS) + ["n_segments"])
    identity = featured.groupby("guid", sort=True)[[c for c in keys[1:] if c in featured.columns]].first()
    means = featured.groupby("guid", sort=True)[present].mean()
    counts = featured.groupby("guid", sort=True).size().rename("n_segments")
    table = pd.concat([identity, means, counts], axis=1).reset_index()
    return table.reindex(columns=keys + list(RECORDING_COLUMNS) + ["n_segments"])


def _finite_or_none(value: Any) -> Optional[float]:
    """A finite float, or ``None`` -- never NaN, which the headline's finiteness check refuses."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def headline_block(
    thresholds: Dict[str, Dict[str, Any]],
    recordings: pd.DataFrame,
    hot: Dict[str, Any],
    usefulness: Dict[str, Any],
    overlap: Dict[str, Any],
) -> Dict[str, Any]:
    """The flat block of scalars the binding's headline entries resolve into."""
    def _mean(column: str) -> Optional[float]:
        if column not in recordings.columns or not len(recordings):
            return None
        values = np.asarray(recordings[column], dtype=np.float64)
        values = values[np.isfinite(values)]
        return _finite_or_none(values.mean()) if values.size else None

    return {
        "high_kl_threshold_nats": _finite_or_none(thresholds["high"]["lo_nats"]),
        "high_kl_centroid_kl_s": _mean(feature_column("high", "kl", "centroid")),
        "high_kl_total_nats": _mean(feature_column("high", "kl", "total_nats")),
        "hot_lag_count": int(hot.get("n_lags") or 0),
        "hot_lag_share_kl": _mean(HOT_SHARE_COLUMNS["kl"]),
        # The usefulness three: the high band's forecast gain, its paired difference against the
        # rest band's, and how far the high-KL anchors are the high-gain anchors.
        "high_kl_pred_gap_nats": _mean(band_column(HIGH_BAND_KEY, PRED_GAP_SUFFIX)),
        "high_minus_rest_pred_gap_nats": _finite_or_none(usefulness.get("mean_difference_nats")),
        "high_gain_overlap_share": _finite_or_none(overlap.get("share_of_high_in_gain")),
    }


# =================================================================================================
# Entry point
# =================================================================================================
def _skip(reason: str, n_segments: int) -> Dict[str, Any]:
    """The recorded skip, carrying the protocol's keys and naming its cause."""
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": True, "skipped": True, "reason": reason},
        "skipped": True,
        "reason": reason,
        "n_segments": int(n_segments),
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "files": [],
    }


def run_lag_high_kl_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select the anchors by pooled KL quantile band and resolve their lag structure on both clocks.

    Args:
        context: The analysis context, read for the per-sample table, the per-anchor table, the
            per-anchor vector sidecar and the collection's lag block. No model is touched, so this
            runs against a finished run directory with no checkpoint and no GPU.
        eval_config: The validated block, read for ``max_hours_before_delivery`` and
            ``event_lag_window_s``. The quantile bands are module constants, for the reason the
            significance level is.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population is what the tables know.

    Returns:
        The protocol's keys, the thresholds and what each band selected, the hot-lag set, the
        headline block, the per-clock populations and significance, and the paths written. A
        recorded skip -- naming its cause -- when the table is empty, when the run predates the
        per-anchor vector sidecar, or when no anchor carries a finite KL.
    """
    collection = context.collection
    per_sample = cohort.within_horizon(
        collection.per_sample, eval_config.get("max_hours_before_delivery")
    )
    if per_sample.empty:
        return _skip("the collected per-sample table was empty", 0)

    lag = dict(dict(getattr(collection, "results", None) or {}).get("lag") or {})
    n_lags = int(lag.get("n_lags") or 0)
    if n_lags <= 0:
        return _skip(
            "the collection record carries no lag geometry, so there is no axis to resolve a "
            "profile against",
            len(per_sample),
        )
    seconds = compensated_seconds_axis(n_lags, int(lag.get("delay_steps") or 0))

    maps = {
        name: array
        for name, array in dict(getattr(collection, "anchor_vectors", None) or {}).items()
        if np.asarray(array).ndim == 2 and int(np.asarray(array).shape[1]) == n_lags
    }
    per_anchor = getattr(collection, "per_anchor", pd.DataFrame())
    if KL_MAP_KEY not in maps or per_anchor is None or len(per_anchor) == 0:
        return _skip(
            f"the run carries no {n_lags}-wide {KL_MAP_KEY!r} per-anchor vector sidecar beside "
            "its per-anchor table, so no anchor's lag map can be read; a directory collected "
            "before the sidecar existed is a partial input, not a broken one -- re-collect to "
            "produce it",
            len(per_sample),
        )
    if int(np.asarray(maps[KL_MAP_KEY]).shape[0]) != len(per_anchor):
        return _skip(
            f"the {KL_MAP_KEY!r} sidecar holds {int(np.asarray(maps[KL_MAP_KEY]).shape[0])} row(s) "
            f"against {len(per_anchor)} per-anchor rows, so it cannot be aligned with the table",
            len(per_sample),
        )

    population = anchor_population(per_sample, per_anchor)
    if population.rows.size == 0:
        return _skip(
            "no anchor of the evaluated population carries a finite KL, or the per-anchor table "
            f"lacks {SAMPLE_INDEX_COLUMN!r} / {KL_COLUMN!r}",
            len(per_sample),
        )

    thresholds = band_thresholds(population)
    featured, profiles, selection = add_feature_columns(
        per_sample, population, maps, thresholds, seconds
    )
    hot = selection.pop("hot_mask")
    masks = selection.pop("masks")

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    classes = (
        np.asarray(per_sample[labels.CLASS_COLUMN].astype(object))[population.sample_rows]
        if labels.CLASS_COLUMN in per_sample.columns
        else np.full(population.rows.size, None, dtype=object)
    )
    high = masks[HIGH_BAND_KEY]
    argmax_table, field, edges = argmax_by_quantile(population, classes, n_lags, seconds)
    enrichment = contraction_enrichment(
        population, high, per_sample, float(eval_config.get("event_lag_window_s", 120.0))
    )
    recordings = recordings_frame(featured)
    # The usefulness half: is the coupling where the forecast gain is?
    gain_table, gain_edges = gain_by_kl_quantile(population, per_sample)
    argmax_gain = gain_by_argmax(population, {k: masks[k] for k in (HIGH_BAND_KEY, REST_BAND_KEY, GAIN_BAND_KEY) if k in masks}, n_lags, seconds)
    overlap = band_overlap(masks)
    usefulness = usefulness_test(
        recordings,
        resamples=int(eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES)),
        seed=int(eval_config.get("seed", 0)),
    )
    occlusion_bands = {
        str(name): (int(span[0]), int(span[1]))
        for name, span in (eval_config.get("occlusion_bands") or {}).items()
        if isinstance(span, (list, tuple)) and len(span) == 2
    }
    consistency, consistency_record = occlusion_consistency(
        output_dir, featured, profiles, occlusion_bands, seconds
    )

    # --- The clocks ------------------------------------------------------------------------------
    clocks: List[Dict[str, Any]] = []
    significance: List[Dict[str, Any]] = []
    per_recording_tables: List[pd.DataFrame] = []
    trajectory: List[Dict[str, Any]] = []
    profile_tables: List[pd.DataFrame] = []
    written: List[str] = []
    for clock in CLOCKS:
        binned, record = clock_rows(clock, featured)
        if binned.empty:
            record.update({"drawn": False, "reason": f"no segment could be placed on the {clock.name} clock"})
            clocks.append(record)
            continue
        frames = {
            axis: cohort.per_recording_in_bins(
                binned, FEATURE_COLUMNS, group_column=axis,
                bin_column=clock.bin_column, center_column=clock.center_column,
            )
            for axis in labels.GROUP_COLUMNS
        }
        for axis, frame in frames.items():
            if len(frame):
                per_recording_tables.append(frame.assign(clock=clock.name, group_column=axis))
        rows: List[Dict[str, Any]] = []
        for axis, frame in frames.items():
            for column in FEATURE_COLUMNS:
                for row in cohort.trajectory_rows(
                    frame, column, metric=column,
                    bin_column=clock.bin_column, center_column=clock.center_column,
                ):
                    rows.append({"clock": clock.name, "group_column": axis, **row})
        trajectory.extend(rows)
        class_frame = frames.get(labels.CLASS_COLUMN, pd.DataFrame())
        records = [analyse_windows(clock, class_frame, column) for column in READOUTS]
        significance.extend(records)
        for band in ANCHOR_BANDS:
            windows, centres, fields = window_profiles(clock, binned, profiles.get(f"{band.key}_kl"), n_lags)
            profile_tables.append(profile_frame(clock, band.key, windows, centres, fields, seconds))
            if band.key == "high":
                written.append(str(figures.render_figure(
                    build_clock_figure(clock, windows, centres, fields, rows, seconds),
                    directory / clock.figure,
                ).name))
        written.append(str(figures.render_figure(
            build_windows_figure(clock, class_frame, records), directory / clock.windows_figure
        ).name))
        record.update(
            {
                "drawn": True,
                "n_recordings": record.get(
                    "n_recordings_eligible",
                    int(class_frame["guid"].nunique()) if len(class_frame) else 0,
                ),
                "n_significant_windows": {
                    r["metric_column"]: r.get("n_significant_windows", 0) for r in records
                },
            }
        )
        clocks.append(record)

    # --- The run-level pages and the tables ------------------------------------------------------
    written.append(str(figures.render_figure(
        build_selection_figure(
            population, classes, np.asarray(per_sample["guid"].astype(str)), thresholds,
            profiles, hot, field, edges, enrichment, seconds,
        ),
        directory / SELECTION_FIGURE,
    ).name))
    written.append(str(figures.render_figure(
        build_usefulness_figure(
            gain_table, argmax_gain, profiles, overlap, usefulness, consistency, seconds
        ),
        directory / USEFULNESS_FIGURE,
    ).name))

    pd.DataFrame(
        [
            {"band": key, **{k: v for k, v in value.items() if not isinstance(v, dict)}}
            for key, value in selection["bands"].items()
        ]
    ).to_csv(directory / THRESHOLDS_FILENAME, index=False)
    pd.DataFrame(
        {
            "lag_step": np.arange(n_lags, dtype=np.int64),
            "compensated_seconds": seconds,
            "pooled_attribution_nats": selection["hot_lags"]["pooled_attribution_nats"],
            "hot": hot.astype(bool),
            **{
                f"pooled_{band.key}_kl_nats": (
                    np.nanmean(profiles[f"{band.key}_kl"], axis=0)
                    if f"{band.key}_kl" in profiles and np.isfinite(profiles[f"{band.key}_kl"]).any()
                    else np.full(n_lags, np.nan)
                )
                for band in ANCHOR_BANDS
            },
            "pooled_gain_weighted_attention": (
                np.nanmean(profiles["gainw_attn"], axis=0)
                if "gainw_attn" in profiles and np.isfinite(profiles["gainw_attn"]).any()
                else np.full(n_lags, np.nan)
            ),
        }
    ).to_csv(directory / SELECTION_FILENAME, index=False)
    gain_table.to_csv(directory / GAIN_BY_QUANTILE_FILENAME, index=False)
    argmax_gain.to_csv(directory / GAIN_BY_ARGMAX_FILENAME, index=False)
    consistency.to_csv(directory / OCCLUSION_CONSISTENCY_FILENAME, index=False)
    recordings.to_csv(directory / RECORDINGS_FILENAME, index=False)
    (
        pd.concat(per_recording_tables, ignore_index=True)
        if per_recording_tables
        else pd.DataFrame(columns=["clock", "group_column", "group", "guid", *FEATURE_COLUMNS])
    ).to_csv(directory / PER_RECORDING_FILENAME, index=False)
    pd.DataFrame(
        trajectory,
        columns=[
            "clock", "group_column", "metric", "group", "time_bin", "bin_center_h",
            "n_recordings", "mean", "q25", "median", "q75",
        ],
    ).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    (
        pd.concat(profile_tables, ignore_index=True)
        if profile_tables else profile_frame(CLOCKS[0], "high", [], [], [], seconds)
    ).to_csv(directory / PROFILE_FILENAME, index=False)
    significance_frame(significance).to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    pairwise_frame(significance).to_csv(directory / PAIRWISE_FILENAME, index=False)
    argmax_table.to_csv(directory / ARGMAX_FILENAME, index=False)
    enrichment.to_csv(directory / CONTRACTION_FILENAME, index=False)

    files = [
        THRESHOLDS_FILENAME, SELECTION_FILENAME, RECORDINGS_FILENAME, PER_RECORDING_FILENAME,
        TRAJECTORY_FILENAME, PROFILE_FILENAME, SIGNIFICANCE_FILENAME, PAIRWISE_FILENAME,
        ARGMAX_FILENAME, CONTRACTION_FILENAME, GAIN_BY_QUANTILE_FILENAME,
        GAIN_BY_ARGMAX_FILENAME, OCCLUSION_CONSISTENCY_FILENAME, *written,
    ]
    drawn = [record for record in clocks if record.get("drawn")]
    logger.info(
        f"{ANALYSIS_DIRNAME}: high band >= {thresholds['high']['lo_nats']:.4g} nats selected "
        f"{selection['bands']['high']['n_anchors_selected']} of "
        f"{selection['bands']['high']['n_anchors_population']} anchors; "
        f"{selection['hot_lags']['n_lags']} hot lag(s); {len(drawn)} of {len(CLOCKS)} clock(s) drawn; "
        f"{sum(r.get('n_significant_windows', 0) for r in significance)} significant window(s) "
        f"across {len(significance)} Holm family(ies)"
    )
    return {
        "n_samples": scored_sample_count(featured, band_column("high", ANCHOR_FRAC_SUFFIX)),
        "composition": {
            record["clock"]: {
                "n_segments": record.get("n_segments", 0),
                "n_recordings": record.get("n_recordings", 0),
                "n_windows": record.get("n_windows", 0),
            }
            for record in clocks
        },
        # Capped: the second clock scores the recordings that carry an onset only, a strict
        # subset of the evaluated cohort, so the coverage block must read this as a different
        # population by design rather than as a disagreement.
        "plan": {
            "capped": True,
            "reason": (
                "the second-stage half is scored over the recordings that carry an onset only, "
                "which is a subset of the evaluated cohort"
            ),
            "tested_features": len(READOUTS),
        },
        "population": {
            "n_anchors": int(population.rows.size),
            "n_segments": int(len(per_sample)),
            "n_recordings": int(per_sample["guid"].nunique()) if "guid" in per_sample else 0,
            "pooled_over": "every scored anchor of every clinical class within the run's horizon",
        },
        "thresholds": thresholds,
        "bands": selection["bands"],
        "hot_lags": selection["hot_lags"],
        "sources": selection["sources"],
        "headline": headline_block(
            thresholds, recordings, selection["hot_lags"], usefulness, overlap
        ),
        "usefulness": {
            **usefulness,
            "gain_column": population.gain_column,
            "gain_by_kl_quantile_edges_nats": gain_edges,
            "overlap": overlap,
            "occlusion_consistency": consistency_record,
        },
        "argmax_by_quantile": {
            "n_bins": int(N_KL_QUANTILE_BINS),
            "edges_nats": edges,
            "n_anchors": int((population.argmax_lag >= 0).sum()),
        },
        "contraction_enrichment": {
            **enrichment_summary(enrichment),
            "event_lag_window_s": float(eval_config.get("event_lag_window_s", 120.0)),
        },
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "n_lags": n_lags,
        "lag_support": read_lag_support(output_dir),
        "statistic_thresholds": {
            "degenerate_peak_to_median": float(DEGENERATE_PEAK_TO_MEDIAN),
            "degenerate_zero_fraction": float(DEGENERATE_ZERO_FRACTION),
        },
        "features": list(FEATURE_COLUMNS),
        "readouts": list(READOUTS),
        "method": METHOD,
        "untested_note": UNTESTED_NOTE,
        "selection_note": SELECTION_NOTE,
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "clocks": clocks,
        "significance": [
            {key: value for key, value in record.items() if key not in ("per_window", "pairwise")}
            for record in significance
        ],
        "files": files,
    }
