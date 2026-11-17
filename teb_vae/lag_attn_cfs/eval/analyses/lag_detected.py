r"""The anchors at which the model detects a lag beyond the immediate one.

The pooled lag readouts (``lag_kl``, ``lag_clocks``, ``lag_kld_scaled``) average every scored
anchor, and ``gate_open`` splits the anchors by the gate only. Two effects then hide the anchors
this analysis is about:

* **A closed gate still has a lag map.** While the effective gate $g^{\mathrm{eff}}_t$ is shut, the
  posterior equals the prior and $K_t \approx 0$, but the attention row still counts in full in an
  attention profile.
* **A tie resolves to lag 0.** The per-anchor ``argmax_lag`` takes the lowest index of equal
  values, so an anchor whose KL map is all zeros reads as a peak at the immediate lag.

**The selection.** An anchor is *active* when the gate is on and the source moved the belief:

$$g^{\mathrm{eff}}_t \ge g_{\mathrm{shut}} \quad\text{and}\quad K_t \ge \varepsilon_K,$$

with $g_{\mathrm{shut}}$ = :data:`~teb_vae.lag_attn_cfs.gate_states.GATE_SHUT_BELOW` and
$\varepsilon_K$ = :data:`MIN_KL_NATS`. An active anchor is *detected* when its KL attribution
$\widetilde K_{t,\ell} = \sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$ peaks at a lag index
$\hat\ell_t = \arg\max_\ell \widetilde K_{t,\ell} \ge 1$ (:data:`MIN_DETECTED_LAG_INDEX`), and
*immediate* when it peaks at $\hat\ell_t = 0$. Every scored anchor falls in exactly one of four
populations: ``gate_off``, ``inactive`` (gate on, $K_t < \varepsilon_K$), ``immediate`` and
``detected``.

**What is reported.**

* The detected anchors themselves, one row each (``lag_detected_anchors.parquet``), and the runs of
  consecutive detected anchors inside a segment (``lag_detected_episodes.csv``). These are the
  step indices a sample page or a later analysis can look up.
* The share of each population per cohort, on anchors and as a mean over recordings with a
  bootstrap interval.
* Where the detected peaks sit on the lag axis, and the KL and attention profiles of the detected
  and the immediate anchors.
* Whether a detected anchor is *useful*: the paired, within-recording difference of the forecast
  gain $D_{\mathrm{base}} - D_{\mathrm{full}}$ and of $K_t$ between the detected and the immediate
  anchors, with a Wilcoxon signed-rank test over recordings.
* The detected share and the median detected lag against the time before delivery.

**Aggregation.** A summary, a profile and a contrast use one value per recording, so a long
recording does not outvote a short one. The peak histogram and the anchor list are anchor-level.

**The axis is stored-coefficient time.** A lag position here is an attribution over the axis the
coefficients are stored on, so every record carries
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT` and every figure the one-line
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_NOTE`.

The analysis reads the per-anchor table and the per-anchor vector sidecar only, so it runs against
a finished run directory with no checkpoint. A model with neither a gate nor a UA mask records a
skip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.collect import CONTRACTION_AGE_COLUMN
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry, per_recording_labels
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    COEFFICIENT_LAG_AXIS_LABEL,
    GROUP_DELAY_CAVEAT,
    compensated_seconds_axis,
)
from teb_vae.lag_attn_cfs.gate_states import GATE_SHUT_BELOW
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS

ANALYSIS_DIRNAME = "lag_detected"

ANCHORS_FILENAME = "lag_detected_anchors.parquet"
EPISODES_FILENAME = "lag_detected_episodes.csv"
SUMMARY_FILENAME = "lag_detected_summary.csv"
PER_RECORDING_FILENAME = "lag_detected_per_recording.csv"
PEAKS_FILENAME = "lag_detected_peaks.csv"
PROFILE_FILENAME = "lag_detected_profiles.csv"
CONTRAST_FILENAME = "lag_detected_contrast.csv"
BY_TIME_FILENAME = "lag_detected_by_time_to_delivery.csv"
OVERVIEW_FIGURE_STEM = "lag_detected"
TIME_FIGURE_STEM = "lag_detected_time"

APPLIED_COLUMN = "event_gate"
OPEN_COLUMN = "source_open"
KL_COLUMN = "kld_per_t"
ARGMAX_COLUMN = "argmax_lag"
SAMPLE_INDEX_COLUMN = "sample_index"
KL_MAP_KEY = "kl_lag_map"
ATTENTION_MAP_KEY = "attention_lag_map"

#: The forecast gain of an anchor, $D_{\mathrm{base}} - D_{\mathrm{full}}$ in nats, in order of
#: preference: the mean-decoded gap, then the Monte Carlo gap, then the training-path gap.
GAIN_COLUMNS: Tuple[str, ...] = ("mean_pred_gap", "mc_pred_gap", "pred_gap")

#: $\varepsilon_K$: the KL in nats below which the source did not move the belief. The objective's
#: own activity threshold, so "active" means here what it means in the loss.
MIN_KL_NATS = float(KLD_ACTIVE_EPS)

#: The smallest peak lag index that counts as a detected lag. Index $0$ is the most recent source
#: step the model can attend to.
MIN_DETECTED_LAG_INDEX = 1

#: The four populations, as ``(key, label)``. They partition the anchors with a finite gate.
POPULATIONS: Tuple[Tuple[str, str], ...] = (
    ("gate_off", f"gate off ($g < {GATE_SHUT_BELOW:g}$)"),
    ("inactive", f"gate on, $K_t < {MIN_KL_NATS:g}$ nats"),
    ("immediate", "gate on, peak at lag 0"),
    ("detected", f"gate on, peak at lag $\\geq {MIN_DETECTED_LAG_INDEX}$"),
)
POPULATION_KEYS: Tuple[str, ...] = tuple(key for key, _ in POPULATIONS)
POPULATION_LABELS: Dict[str, str] = dict(POPULATIONS)

#: The palette's secondary blue: the immediate anchors are the detected anchors' closest relative.
IMMEDIATE_COLOR = "#3775BA"
#: The palette's lightest red: active neither in the gate nor in the KL.
INACTIVE_COLOR = "#F6CFCB"

#: The cohort of every recording.
ALL_COHORT = "all"

#: The clock of the time table: half-hour windows up to twelve hours before delivery.
TIME_BIN_HOURS = 0.5
MAX_HOURS = 12.0
#: A recording enters a window's recording-level share only with this many active anchors there.
MIN_WINDOW_ANCHORS = 10

#: The lag panels show the lags up to the one where this share of the pooled mass is reached, so a
#: tail of empty lags does not take most of the axis.
VISIBLE_MASS = 0.995
#: The lag panels always show at least this many lags.
MIN_VISIBLE_LAGS = 6

#: The per-recording columns resolved by cohort by the runner's grouped variants.
GROUPED_METRICS: Tuple[str, ...] = (
    "detected_share_of_active",
    "detected_median_lag_s",
    "detected_gain_nats",
    "immediate_gain_nats",
)


# =================================================================================================
# The selection
# =================================================================================================
def applied_gate(per_anchor: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    r"""Return the effective gate $g^{\mathrm{eff}}_t$ of every per-anchor row.

    Args:
        per_anchor: The collection pass's per-anchor table.

    Returns:
        The applied gate, or the UA-open flag $o_t$ on a model that masks the UA but learns no
        gate. ``None`` when the table carries neither column.
    """
    if per_anchor is None or len(per_anchor) == 0:
        return None
    for column in (APPLIED_COLUMN, OPEN_COLUMN):
        if column in per_anchor.columns:
            return per_anchor[column].to_numpy(dtype=np.float64)
    return None


def classify_anchors(
    gate: np.ndarray,
    kl: np.ndarray,
    argmax: np.ndarray,
    *,
    min_kl: float = MIN_KL_NATS,
    min_lag: int = MIN_DETECTED_LAG_INDEX,
) -> np.ndarray:
    r"""Assign every anchor to one of the four populations.

    Args:
        gate: The effective gate $g^{\mathrm{eff}}_t$, $(N,)$.
        kl: The source-conditioned KL $K_t$ in nats, $(N,)$. A non-finite value counts as inactive.
        argmax: The peak lag index $\hat\ell_t$ of the KL attribution, $(N,)$.
        min_kl: $\varepsilon_K$.
        min_lag: The smallest peak index that counts as detected.

    Returns:
        The index into :data:`POPULATION_KEYS` of every anchor, $(N,)$ ``int8``.
    """
    gate = np.asarray(gate, dtype=np.float64)
    kl = np.asarray(kl, dtype=np.float64)
    argmax = np.asarray(argmax, dtype=np.int64)
    on = gate >= GATE_SHUT_BELOW
    active = on & np.isfinite(kl) & (kl >= float(min_kl))
    codes = np.zeros(gate.shape, dtype=np.int8)
    codes[on & ~active] = POPULATION_KEYS.index("inactive")
    codes[active & (argmax < int(min_lag))] = POPULATION_KEYS.index("immediate")
    codes[active & (argmax >= int(min_lag))] = POPULATION_KEYS.index("detected")
    return codes


def gain_column(frame: pd.DataFrame) -> Optional[str]:
    """Return the first column of :data:`GAIN_COLUMNS` that holds a finite value, or ``None``."""
    for column in GAIN_COLUMNS:
        if column in frame.columns and np.isfinite(frame[column].to_numpy(dtype=np.float64)).any():
            return column
    return None


def group_sums(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of ``values`` per group with one sparse product.

    Args:
        values: $(N,)$ or $(N, L)$.
        codes: The group of each row, in $[0, n_{\\mathrm{groups}})$.
        n_groups: How many groups there are.

    Returns:
        $(G,)$ or $(G, L)$.
    """
    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    matrix = sparse.csr_matrix(
        (np.ones(codes.size), (codes, np.arange(codes.size))), shape=(int(n_groups), codes.size)
    )
    return np.asarray(matrix @ values)


class AnchorData:
    """The per-anchor arrays every table of this analysis is reduced from.

    Attributes:
        frame: The per-anchor rows with a finite gate, re-indexed from zero.
        gate: The effective gate of each row.
        kl: $K_t$ of each row.
        argmax: The peak lag index of each row.
        codes: The population index of each row (:func:`classify_anchors`).
        gain: The forecast gain of each row in nats, ``NaN`` where absent.
        gain_column: The column ``gain`` was read from, or ``None``.
        kl_map: The KL attribution per lag, $(N, L)$.
        attention_map: The attention weight per lag, $(N, L)$, zeros when the sidecar lacks it.
        seconds: The lag axis in seconds, $(L,)$.
        hours: The time before delivery of each row in hours, never negative.
        recording_codes: The recording of each row as an index into ``recordings``.
        recordings: The recording identifiers, in code order.
        cohort_of_recording: The clinical class of each recording, ``None`` where unresolved.
    """

    def __init__(
        self,
        per_anchor: pd.DataFrame,
        gate: np.ndarray,
        maps: Dict[str, np.ndarray],
        seconds: np.ndarray,
        labelled: pd.DataFrame,
    ) -> None:
        finite = np.isfinite(gate)
        self.frame = per_anchor.loc[finite].reset_index(drop=True)
        self.gate = gate[finite]
        self.kl = (
            self.frame[KL_COLUMN].to_numpy(dtype=np.float64) if KL_COLUMN in self.frame.columns
            else np.full(len(self.frame), np.nan)
        )
        self.kl_map = np.asarray(maps[KL_MAP_KEY], dtype=np.float32)[finite]
        self.attention_map = (
            np.asarray(maps[ATTENTION_MAP_KEY], dtype=np.float32)[finite]
            if ATTENTION_MAP_KEY in maps else np.zeros_like(self.kl_map)
        )
        self.argmax = (
            self.frame[ARGMAX_COLUMN].to_numpy(dtype=np.int64)
            if ARGMAX_COLUMN in self.frame.columns else self.kl_map.argmax(axis=1)
        )
        self.codes = classify_anchors(self.gate, self.kl, self.argmax)
        self.gain_column = gain_column(self.frame)
        self.gain = (
            self.frame[self.gain_column].to_numpy(dtype=np.float64) if self.gain_column
            else np.full(len(self.frame), np.nan)
        )
        self.seconds = np.asarray(seconds, dtype=np.float64)
        if {"epoch", "anchor"} <= set(self.frame.columns):
            to_delivery = (
                self.frame["epoch"].to_numpy(dtype=float)
                + self.frame["anchor"].to_numpy(dtype=float) * SECONDS_PER_STEP
            )
            self.hours = np.clip(-to_delivery / 3600.0, 0.0, None)
        else:
            self.hours = np.full(len(self.frame), np.nan)
        codes, recordings = pd.factorize(self.frame["guid"], sort=True)
        self.recording_codes = codes.astype(np.int64)
        self.recordings = [str(name) for name in recordings]
        if labels.CLASS_COLUMN in getattr(labelled, "columns", []):
            classes = labelled[labels.CLASS_COLUMN]
        elif labels.CLASS_COLUMN in self.frame.columns:
            classes = self.frame.groupby("guid")[labels.CLASS_COLUMN].first()
        else:
            classes = pd.Series(dtype=object)
        self.cohort_of_recording = [
            None if pd.isna(classes.get(name, None)) else str(classes.get(name))
            for name in self.recordings
        ]

    @property
    def n_lags(self) -> int:
        """The lag window width $L$."""
        return int(self.kl_map.shape[1])

    def mask(self, population: str) -> np.ndarray:
        """Select the rows of one population, or of ``'active'`` (``immediate`` and ``detected``)."""
        if population == "active":
            return np.isin(self.codes, [POPULATION_KEYS.index("immediate"), POPULATION_KEYS.index("detected")])
        return self.codes == POPULATION_KEYS.index(population)

    def cohorts(self) -> List[str]:
        """The pooled cohort first, then each clinical class the recordings carry, by severity.

        The classes follow ``labels.CLASS_NAMES`` (healthy, acidosis, hie); a class outside it
        comes after them, sorted.
        """
        present = {name for name in self.cohort_of_recording if name is not None}
        known = [name for name in labels.CLASS_NAMES.values() if name in present]
        return [ALL_COHORT, *known, *sorted(present - set(known))]

    def rows_in(self, cohort: str) -> np.ndarray:
        """Select the rows whose recording belongs to one cohort."""
        if cohort == ALL_COHORT:
            return np.ones(len(self.frame), dtype=bool)
        members = np.asarray([name == cohort for name in self.cohort_of_recording], dtype=bool)
        return members[self.recording_codes]

    def recordings_in(self, cohort: str) -> np.ndarray:
        """Select the recordings of one cohort, as a mask over ``recordings``."""
        if cohort == ALL_COHORT:
            return np.ones(len(self.recordings), dtype=bool)
        return np.asarray([name == cohort for name in self.cohort_of_recording], dtype=bool)


def _interval(values: np.ndarray, *, resamples: int, seed: int) -> Tuple[float, float, float, int]:
    """The mean of a per-recording vector with its percentile bootstrap interval.

    Args:
        values: One value per recording. Non-finite entries are dropped.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        ``(mean, lo, hi, n)``. The bounds are ``NaN`` below three recordings.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if finite.size < 3:
        return float(finite.mean()), float("nan"), float("nan"), int(finite.size)
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, finite.size, size=(int(resamples), finite.size))
    means = finite[draws].mean(axis=1)
    return (
        float(finite.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)),
        int(finite.size),
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide elementwise, ``NaN`` where the denominator is zero."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


# =================================================================================================
# The detected anchors and their episodes
# =================================================================================================
def anchor_table(data: AnchorData) -> pd.DataFrame:
    """List every detected anchor, one row each.

    Args:
        data: The anchors.

    Returns:
        One row per detected anchor: its identity (``guid``, ``epoch``, ``sample_index``,
        ``anchor``), the time before delivery, the peak lag index and seconds, $K_t$, the gate, the
        forecast gain, the share of the anchor's KL at its peak and at lag 0, the contraction age
        and the clinical class.
    """
    rows = np.flatnonzero(data.mask("detected"))
    frame = data.frame.iloc[rows]
    kl_map = data.kl_map[rows].astype(np.float64)
    total = kl_map.sum(axis=1)
    peak = np.clip(data.argmax[rows], 0, data.n_lags - 1)
    table = pd.DataFrame({
        "guid": frame["guid"].astype(str).to_numpy(),
        "epoch": frame["epoch"].to_numpy(dtype=float) if "epoch" in frame.columns else np.nan,
        SAMPLE_INDEX_COLUMN: (
            frame[SAMPLE_INDEX_COLUMN].to_numpy() if SAMPLE_INDEX_COLUMN in frame.columns else -1
        ),
        "anchor": frame["anchor"].to_numpy() if "anchor" in frame.columns else -1,
        "hours_before_delivery": data.hours[rows],
        "lag_index": peak,
        "lag_s": data.seconds[peak],
        "kl_nats": data.kl[rows],
        "gate": data.gate[rows],
        "gain_nats": data.gain[rows],
        "peak_kl_share": _safe_ratio(kl_map[np.arange(rows.size), peak], total),
        "lag0_kl_share": _safe_ratio(kl_map[:, 0], total),
        CONTRACTION_AGE_COLUMN: (
            frame[CONTRACTION_AGE_COLUMN].to_numpy(dtype=float)
            if CONTRACTION_AGE_COLUMN in frame.columns else np.nan
        ),
        labels.CLASS_COLUMN: [data.cohort_of_recording[code] for code in data.recording_codes[rows]],
    })
    return table


def anchor_stride(frame: pd.DataFrame) -> int:
    """The anchor stride of the table: the commonest positive step between anchors of one segment.

    Args:
        frame: The per-anchor rows, with ``guid``, ``sample_index`` and ``anchor``.

    Returns:
        The stride in stored steps; ``1`` when it cannot be read.
    """
    if not {"guid", SAMPLE_INDEX_COLUMN, "anchor"} <= set(frame.columns) or len(frame) < 2:
        return 1
    ordered = frame.sort_values(["guid", SAMPLE_INDEX_COLUMN, "anchor"])
    same = (
        (ordered["guid"].to_numpy()[1:] == ordered["guid"].to_numpy()[:-1])
        & (ordered[SAMPLE_INDEX_COLUMN].to_numpy()[1:] == ordered[SAMPLE_INDEX_COLUMN].to_numpy()[:-1])
    )
    steps = np.diff(ordered["anchor"].to_numpy(dtype=np.int64))[same]
    steps = steps[steps > 0]
    if steps.size == 0:
        return 1
    values, counts = np.unique(steps, return_counts=True)
    return int(values[np.argmax(counts)])


def episode_table(anchors: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Join consecutive detected anchors of one segment into episodes.

    Two detected anchors of one segment belong to one episode when their anchors are one
    ``stride`` apart, so an episode is a run of detection with no anchor between its members.

    Args:
        anchors: From :func:`anchor_table`.
        stride: From :func:`anchor_stride`.

    Returns:
        One row per episode: the segment, the first and last anchor, the anchor count, the
        duration in seconds, the commonest peak lag and its seconds, the mean $K_t$ and gain, the
        time before delivery at its start and the clinical class.
    """
    columns = [
        "guid", SAMPLE_INDEX_COLUMN, "epoch", "start_anchor", "end_anchor", "n_anchors",
        "duration_s", "modal_lag_index", "modal_lag_s", "mean_kl_nats", "mean_gain_nats",
        "hours_before_delivery", labels.CLASS_COLUMN,
    ]
    if anchors.empty:
        return pd.DataFrame(columns=columns)
    ordered = anchors.sort_values(["guid", SAMPLE_INDEX_COLUMN, "anchor"]).reset_index(drop=True)
    anchor = ordered["anchor"].to_numpy(dtype=np.int64)
    same_segment = np.r_[False, (
        (ordered["guid"].to_numpy()[1:] == ordered["guid"].to_numpy()[:-1])
        & (ordered[SAMPLE_INDEX_COLUMN].to_numpy()[1:] == ordered[SAMPLE_INDEX_COLUMN].to_numpy()[:-1])
    )]
    joined = same_segment & (np.r_[0, np.diff(anchor)] == int(stride))
    ordered["episode"] = np.cumsum(~joined)

    def modal(values: pd.Series) -> int:
        """The commonest value; the smallest on a tie."""
        counts = values.value_counts()
        return int(counts.index[counts.to_numpy() == counts.max()].min())

    grouped = ordered.groupby("episode", sort=True)
    table = grouped.agg(
        guid=("guid", "first"),
        sample_index=(SAMPLE_INDEX_COLUMN, "first"),
        epoch=("epoch", "first"),
        start_anchor=("anchor", "min"),
        end_anchor=("anchor", "max"),
        n_anchors=("anchor", "size"),
        modal_lag_index=("lag_index", modal),
        mean_kl_nats=("kl_nats", "mean"),
        mean_gain_nats=("gain_nats", "mean"),
        hours_before_delivery=("hours_before_delivery", "max"),
        clinical_class=(labels.CLASS_COLUMN, "first"),
    ).reset_index(drop=True)
    table = table.rename(columns={"sample_index": SAMPLE_INDEX_COLUMN, "clinical_class": labels.CLASS_COLUMN})
    table["duration_s"] = (table["end_anchor"] - table["start_anchor"] + int(stride)) * SECONDS_PER_STEP
    lag_s = dict(zip(ordered["lag_index"], ordered["lag_s"]))
    table["modal_lag_s"] = table["modal_lag_index"].map(lag_s)
    return table[columns]


# =================================================================================================
# Per recording, then over recordings
# =================================================================================================
def per_recording_table(data: AnchorData) -> pd.DataFrame:
    """Reduce the anchors to one row per recording.

    Args:
        data: The anchors.

    Returns:
        One row per recording, indexed by ``guid``: the anchor count per population, the detected
        share of the scored and of the active anchors, the mean $K_t$ and gain of the immediate and
        the detected anchors, and the median detected lag in seconds.
    """
    n = len(data.recordings)
    table = pd.DataFrame(index=pd.Index(data.recordings, name="guid"))
    table[labels.CLASS_COLUMN] = data.cohort_of_recording
    table["n_scored"] = np.bincount(data.recording_codes, minlength=n)
    for key in POPULATION_KEYS:
        table[f"n_{key}"] = np.bincount(data.recording_codes[data.mask(key)], minlength=n)
    table["n_active"] = table["n_immediate"] + table["n_detected"]
    table["detected_share_of_scored"] = _safe_ratio(table["n_detected"], table["n_scored"])
    table["detected_share_of_active"] = _safe_ratio(table["n_detected"], table["n_active"])
    for key in ("immediate", "detected"):
        mask = data.mask(key)
        for name, values in (("kl_nats", data.kl), ("gain_nats", data.gain)):
            usable = mask & np.isfinite(values)
            sums = group_sums(values[usable], data.recording_codes[usable], n)
            counts = np.bincount(data.recording_codes[usable], minlength=n)
            table[f"{key}_{name.replace('_nats', '')}_nats"] = _safe_ratio(sums, counts)
    detected = data.mask("detected")
    lag_s = pd.Series(data.seconds[np.clip(data.argmax[detected], 0, data.n_lags - 1)])
    medians = lag_s.groupby(data.recording_codes[detected]).median()
    table["detected_median_lag_s"] = np.nan
    table.iloc[medians.index.to_numpy(), table.columns.get_loc("detected_median_lag_s")] = medians.to_numpy()
    return table


def summary_table(data: AnchorData, per_guid: pd.DataFrame, *, resamples: int, seed: int) -> pd.DataFrame:
    """Each population's size, KL and gain, per cohort.

    Args:
        data: The anchors.
        per_guid: From :func:`per_recording_table`.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per cohort and population: the anchor count and share, the mean over recordings of
        the recording's share with its interval, the number of recordings that have the
        population, and the mean $K_t$ and gain of its anchors.
    """
    rows: List[Dict[str, Any]] = []
    for cohort in data.cohorts():
        in_rows = data.rows_in(cohort)
        members = per_guid[data.recordings_in(cohort)]
        n_scored = int(in_rows.sum())
        for key in POPULATION_KEYS:
            mask = in_rows & data.mask(key)
            share = _safe_ratio(members[f"n_{key}"], members["n_scored"])
            mean, lo, hi, n_rec = _interval(share, resamples=resamples, seed=seed)
            gain = data.gain[mask]
            rows.append({
                "cohort": cohort, "population": key, "label": POPULATION_LABELS[key],
                "n_anchors": int(mask.sum()),
                "anchor_share": float(mask.sum() / n_scored) if n_scored else float("nan"),
                "recording_share_mean": mean, "recording_share_lo": lo, "recording_share_hi": hi,
                "n_recordings": n_rec,
                "n_recordings_with_population": int((members[f"n_{key}"] > 0).sum()),
                "kl_nats_mean": float(np.nanmean(data.kl[mask])) if mask.any() else float("nan"),
                "gain_nats_mean": (
                    float(np.nanmean(gain)) if np.isfinite(gain).any() else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def peak_table(data: AnchorData) -> pd.DataFrame:
    """Where the detected peaks sit on the lag axis, per cohort.

    Args:
        data: The anchors.

    Returns:
        One row per cohort and lag index from :data:`MIN_DETECTED_LAG_INDEX` up: the count of
        detected anchors peaking there, its share of the cohort's detected anchors, and the number
        of recordings with at least one such anchor.
    """
    rows: List[Dict[str, Any]] = []
    detected = data.mask("detected")
    for cohort in data.cohorts():
        mask = detected & data.rows_in(cohort)
        peaks = np.clip(data.argmax[mask], 0, data.n_lags - 1)
        counts = np.bincount(peaks, minlength=data.n_lags)
        recordings = pd.Series(data.recording_codes[mask]).groupby(peaks).nunique()
        total = int(mask.sum())
        for lag in range(MIN_DETECTED_LAG_INDEX, data.n_lags):
            rows.append({
                "cohort": cohort, "lag_index": lag, "lag_s": float(data.seconds[lag]),
                "n_anchors": int(counts[lag]),
                "share_of_detected": float(counts[lag] / total) if total else float("nan"),
                "n_recordings": int(recordings.get(lag, 0)),
            })
    return pd.DataFrame(rows)


def profile_table(data: AnchorData) -> pd.DataFrame:
    """The KL and attention lag profiles of the immediate, detected and active anchors, per cohort.

    Each recording's anchors are averaged first and the recordings then averaged, so a recording
    counts once. The KL profile is reported both in nats and as a share of its sum over lags.

    Args:
        data: The anchors.

    Returns:
        One row per cohort, population and lag.
    """
    n = len(data.recordings)
    rows: List[Dict[str, Any]] = []
    for cohort in data.cohorts():
        in_rows = data.rows_in(cohort)
        for key in ("immediate", "detected", "active"):
            mask = in_rows & data.mask(key)
            if not mask.any():
                continue
            codes = data.recording_codes[mask]
            counts = np.bincount(codes, minlength=n)
            present = counts > 0
            kl = group_sums(data.kl_map[mask], codes, n)[present] / counts[present, None]
            attention = group_sums(data.attention_map[mask], codes, n)[present] / counts[present, None]
            kl_mean = kl.mean(axis=0)
            attention_mean = attention.mean(axis=0)
            total = float(kl_mean.sum())
            for lag in range(data.n_lags):
                rows.append({
                    "cohort": cohort, "population": key, "lag_index": lag,
                    "lag_s": float(data.seconds[lag]),
                    "kl_nats": float(kl_mean[lag]),
                    "kl_share": float(kl_mean[lag] / total) if total > 0 else float("nan"),
                    "attention": float(attention_mean[lag]),
                    "n_recordings": int(present.sum()),
                })
    return pd.DataFrame(rows)


def contrast_table(data: AnchorData, per_guid: pd.DataFrame, *, resamples: int, seed: int) -> pd.DataFrame:
    """The paired, within-recording difference of the detected and the immediate anchors.

    For each recording with both populations, $d = \\bar x_{\\mathrm{detected}} -
    \\bar x_{\\mathrm{immediate}}$ for the forecast gain and for $K_t$. Pairing removes the
    between-recording spread, which dominates every readout here.

    Args:
        data: The anchors.
        per_guid: From :func:`per_recording_table`.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per cohort and readout: the pair count, the mean and median difference with the
        interval of the mean, the share of recordings where the detected anchors run higher, and
        the Wilcoxon signed-rank $p$-value.
    """
    rows: List[Dict[str, Any]] = []
    for cohort in data.cohorts():
        members = per_guid[data.recordings_in(cohort)]
        for readout in ("gain", "kl"):
            left = members[f"detected_{readout}_nats"].to_numpy(dtype=np.float64)
            right = members[f"immediate_{readout}_nats"].to_numpy(dtype=np.float64)
            paired = np.isfinite(left) & np.isfinite(right)
            difference = left[paired] - right[paired]
            mean, lo, hi, n_pairs = _interval(difference, resamples=resamples, seed=seed)
            test = shared_stats.wilcoxon_paired(
                left[paired], right[paired], label_left="detected", label_right="immediate"
            )
            rows.append({
                "cohort": cohort, "readout": f"{readout}_nats", "n_pairs": n_pairs,
                "mean_difference": mean, "ci_lo": lo, "ci_hi": hi,
                "median_difference": float(np.median(difference)) if difference.size else float("nan"),
                "detected_higher_fraction": (
                    float((difference > 0).mean()) if difference.size else float("nan")
                ),
                "wilcoxon_p": test.get("p_value", float("nan")),
                "note": test.get("note", ""),
            })
    return pd.DataFrame(rows)


def time_table(data: AnchorData, edges: np.ndarray, *, resamples: int, seed: int) -> pd.DataFrame:
    """The detected share and the detected lag against the time before delivery, per cohort.

    Args:
        data: The anchors.
        edges: The window edges in hours before delivery.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per cohort and window: the active and detected anchor counts, the anchor-level
        detected share, the mean over recordings of the recording's detected share (recordings
        with at least :data:`MIN_WINDOW_ANCHORS` active anchors in the window) with its interval,
        the median and quartiles of the detected lag in seconds, and the mean over recordings of the
        recording's mean detected lag with its interval.
    """
    n = len(data.recordings)
    window = np.digitize(data.hours, edges) - 1
    active = data.mask("active")
    detected = data.mask("detected")
    lag_s = data.seconds[np.clip(data.argmax, 0, data.n_lags - 1)]
    rows: List[Dict[str, Any]] = []
    for cohort in data.cohorts():
        in_rows = data.rows_in(cohort)
        for index in range(len(edges) - 1):
            here = in_rows & (window == index)
            n_active = np.bincount(data.recording_codes[here & active], minlength=n)
            n_detected = np.bincount(data.recording_codes[here & detected], minlength=n)
            eligible = n_active >= MIN_WINDOW_ANCHORS
            share = _safe_ratio(n_detected[eligible], n_active[eligible])
            mean, lo, hi, n_rec = _interval(share, resamples=resamples, seed=seed + index)
            total_active = int(n_active.sum())
            peaks = lag_s[here & detected]
            peak_codes = data.recording_codes[here & detected]
            lag_sum = group_sums(peaks, peak_codes, n) if peaks.size else np.zeros(n)
            lag_mean = _safe_ratio(lag_sum, n_detected)
            lag_mean_value, lag_lo, lag_hi, _ = _interval(
                lag_mean[eligible], resamples=resamples, seed=seed + 1000 + index
            )
            rows.append({
                "cohort": cohort, "hours_lo": float(edges[index]), "hours_hi": float(edges[index + 1]),
                "hours_mid": float(0.5 * (edges[index] + edges[index + 1])),
                "n_active": total_active, "n_detected": int(n_detected.sum()),
                "anchor_detected_share": (
                    float(n_detected.sum() / total_active) if total_active else float("nan")
                ),
                "recording_detected_share_mean": mean, "recording_detected_share_lo": lo,
                "recording_detected_share_hi": hi, "n_recordings": n_rec,
                "detected_median_lag_s": float(np.median(peaks)) if peaks.size else float("nan"),
                "detected_q25_lag_s": float(np.quantile(peaks, 0.25)) if peaks.size else float("nan"),
                "detected_q75_lag_s": float(np.quantile(peaks, 0.75)) if peaks.size else float("nan"),
                "recording_detected_lag_mean_s": lag_mean_value,
                "recording_detected_lag_lo_s": lag_lo, "recording_detected_lag_hi_s": lag_hi,
            })
    return pd.DataFrame(rows)


# =================================================================================================
# Figures
# =================================================================================================
def _population_colors() -> Dict[str, str]:
    """The colour of each population: neutral, pale red, secondary blue, key blue."""
    return {
        "gate_off": figures.COLOR_LIGHT_GRAY,
        "inactive": INACTIVE_COLOR,
        "immediate": IMMEDIATE_COLOR,
        "detected": figures.COLOR_BLUE,
    }


def _cohort_label(cohort: str) -> str:
    """The display name of a cohort."""
    return "all recordings" if cohort == ALL_COHORT else cohort


def visible_lags(peaks: pd.DataFrame, n_lags: int) -> int:
    """The number of lags the lag panels show.

    Args:
        peaks: From :func:`peak_table`.
        n_lags: The lag window width $L$.

    Returns:
        The smallest count of lags from lag 0 that holds :data:`VISIBLE_MASS` of the pooled
        detected peaks, at least :data:`MIN_VISIBLE_LAGS` and at most $L$.
    """
    pooled = peaks[peaks["cohort"] == ALL_COHORT].sort_values("lag_index")
    total = float(pooled["n_anchors"].sum())
    if total <= 0:
        return int(min(n_lags, MIN_VISIBLE_LAGS))
    cumulative = np.cumsum(pooled["n_anchors"].to_numpy(dtype=float)) / total
    last = int(pooled["lag_index"].to_numpy()[int(np.searchsorted(cumulative, VISIBLE_MASS))])
    return int(np.clip(last + 2, MIN_VISIBLE_LAGS, n_lags))


def _draw_shares(ax: Any, summary: pd.DataFrame, cohorts: Sequence[str]) -> None:
    """Horizontal 100 % bars of the four populations per cohort, black-edged, detected share printed."""
    colors = _population_colors()
    order = list(reversed(list(cohorts)))
    for row, cohort in enumerate(order):
        part = summary[summary["cohort"] == cohort].set_index("population")
        left = 0.0
        for key in POPULATION_KEYS:
            width = float(part.loc[key, "anchor_share"]) if key in part.index else 0.0
            if not np.isfinite(width):
                width = 0.0
            ax.barh(row, width, left=left, height=0.62, color=colors[key],
                    edgecolor=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE,
                    label=POPULATION_LABELS[key] if row == 0 else None)
            if key == "detected" and width > 0:
                ax.text(1.02, row, f"{100 * width:.1f}%", ha="left", va="center", clip_on=False,
                        color=colors["detected"], fontsize=figures.FONT_LABEL, fontweight="bold")
            left += width
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_cohort_label(name) for name in order])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("share of scored anchors (%)")
    ax.set_title("Where each scored anchor falls")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _draw_peaks(ax: Any, peaks: pd.DataFrame, cohorts: Sequence[str], n_visible: int) -> None:
    """Grouped bars of where the detected peaks sit, per clinical class, over the visible lags."""
    classes = [name for name in cohorts if name != ALL_COHORT] or [ALL_COHORT]
    colors = figures.group_colors(classes)
    colors.setdefault(ALL_COHORT, figures.COLOR_BLUE)
    used = peaks[peaks["n_anchors"] > 0]
    if used.empty:
        figures.style_axes(ax)
        ax.text(0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center")
        return
    lags = np.arange(MIN_DETECTED_LAG_INDEX, int(n_visible))
    width = 0.8 / len(classes)
    for position, cohort in enumerate(classes):
        part = peaks[peaks["cohort"] == cohort].set_index("lag_index").reindex(lags)
        ax.bar(lags + (position - (len(classes) - 1) / 2.0) * width,
               100.0 * part["share_of_detected"].fillna(0.0).to_numpy(), width=width,
               color=colors[cohort], edgecolor=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE,
               label=_cohort_label(cohort))
    seconds = peaks.drop_duplicates("lag_index").set_index("lag_index")["lag_s"]
    step = max(1, int(np.ceil(len(lags) / 12)))
    ticks = lags[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{seconds.get(lag, np.nan):g}" for lag in ticks])
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("share of detected anchors (%)")
    ax.set_title("Where the detected peaks sit")
    ax.set_ylim(bottom=0.0)
    figures.style_axes(ax)
    ax.legend(loc="upper right")


def _draw_profiles(ax: Any, profiles: pd.DataFrame, n_visible: int) -> None:
    """The pooled KL share per lag of the immediate and the detected anchors, over the visible lags."""
    pooled = profiles[(profiles["cohort"] == ALL_COHORT) & (profiles["lag_index"] < int(n_visible))]
    colors = _population_colors()
    for key, style in (("immediate", "--"), ("detected", "-")):
        part = pooled[pooled["population"] == key].sort_values("lag_index")
        if part.empty:
            continue
        ax.plot(part["lag_s"], 100.0 * part["kl_share"], linestyle=style, color=colors[key],
                linewidth=figures.LINE_EMPHASIS, marker="o", markersize=figures.MARKER_SMALL * 0.8,
                markeredgecolor="white", markeredgewidth=0.5, label=POPULATION_LABELS[key])
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("share of the anchor's KL (%)")
    ax.set_title("KL attribution over the lags")
    ax.set_ylim(bottom=0.0)
    figures.style_axes(ax)
    ax.legend(loc="upper right")


def _draw_contrast(ax: Any, per_guid: pd.DataFrame, contrast: pd.DataFrame, cohorts: Sequence[str]) -> None:
    """Per-recording gain difference, detected minus immediate: mean bars with intervals and points."""
    colors = figures.group_colors([name for name in cohorts if name != ALL_COHORT])
    colors[ALL_COHORT] = figures.COLOR_BLUE
    generator = np.random.default_rng(0)
    rows = contrast[contrast["readout"] == "gain_nats"].set_index("cohort")
    for position, cohort in enumerate(cohorts):
        members = per_guid if cohort == ALL_COHORT else per_guid[per_guid[labels.CLASS_COLUMN] == cohort]
        difference = (members["detected_gain_nats"] - members["immediate_gain_nats"]).to_numpy(dtype=float)
        difference = difference[np.isfinite(difference)]
        if cohort not in rows.index or difference.size == 0:
            continue
        row = rows.loc[cohort]
        ax.bar(position, row["mean_difference"], width=0.6, color=colors[cohort], alpha=0.85,
               edgecolor=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
        if np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"]):
            ax.errorbar(position, row["mean_difference"],
                        yerr=[[row["mean_difference"] - row["ci_lo"]], [row["ci_hi"] - row["mean_difference"]]],
                        color=figures.COLOR_BLACK, capsize=3.0, linewidth=figures.LINE_THIN)
        jitter = generator.uniform(-0.18, 0.18, difference.size)
        ax.scatter(position + jitter, difference, s=5, color=figures.COLOR_BLACK, alpha=0.25, linewidths=0,
                   zorder=3)

    def tick(cohort: str) -> str:
        """The cohort name, with its pair count and Wilcoxon p-value on two lines under it."""
        name = "all" if cohort == ALL_COHORT else cohort
        if cohort not in rows.index:
            return name
        row = rows.loc[cohort]
        p_value = float(row["wilcoxon_p"])
        test = f"p = {p_value:.1g}" if np.isfinite(p_value) else "not tested"
        return f"{name}\nn = {int(row['n_pairs'])}\n{test}"

    ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([tick(name) for name in cohorts], fontsize=figures.FONT_SMALL)
    ax.set_ylabel("gain, detected $-$ immediate (nats)")
    ax.set_title("Forecast gain per recording")
    figures.style_axes(ax)
    finite = (per_guid["detected_gain_nats"] - per_guid["immediate_gain_nats"]).to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        low, high = np.quantile(finite, [0.02, 0.98])
        span = max(high - low, 1e-6)
        ax.set_ylim(min(low, 0.0) - 0.1 * span, max(high, 0.0) + 0.1 * span)


def draw_overview_figure(
    summary: pd.DataFrame,
    peaks: pd.DataFrame,
    profiles: pd.DataFrame,
    per_guid: pd.DataFrame,
    contrast: pd.DataFrame,
    cohorts: Sequence[str],
    path: Path,
) -> Any:
    """Draw the four-panel overview, with the population key in a row of its own.

    Args:
        summary: From :func:`summary_table`.
        peaks: From :func:`peak_table`.
        profiles: From :func:`profile_table`.
        per_guid: From :func:`per_recording_table`.
        contrast: From :func:`contrast_table`.
        cohorts: The cohort keys, pooled first.
        path: The figure stem.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(figures.FIGURE_WIDTH, 6.4))
    # An explicit layout: the population key is a row of its own above the panels, and
    # tight_layout cannot place a figure-level legend.
    grid = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.42, left=0.14, right=0.95, top=0.88, bottom=0.17)
    colors = _population_colors()
    fig.legend(
        handles=[Patch(facecolor=colors[key], edgecolor=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE,
                       label=POPULATION_LABELS[key]) for key in POPULATION_KEYS],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(POPULATION_KEYS), frameon=False,
        handlelength=1.4, columnspacing=1.4,
    )
    n_lags = int(profiles["lag_index"].max()) + 1 if len(profiles) else MIN_VISIBLE_LAGS
    n_visible = visible_lags(peaks, n_lags)
    _draw_shares(fig.add_subplot(grid[0, 0]), summary, cohorts)
    _draw_peaks(fig.add_subplot(grid[0, 1]), peaks, cohorts, n_visible)
    _draw_profiles(fig.add_subplot(grid[1, 0]), profiles, n_visible)
    _draw_contrast(fig.add_subplot(grid[1, 1]), per_guid, contrast, cohorts)
    figures.caveat_note(fig)
    figures.mark_laid_out(fig)
    return figures.render_figure(fig, path)


def draw_time_figure(by_time: pd.DataFrame, cohorts: Sequence[str], path: Path) -> Any:
    """Draw the detected share and the detected lag against the time before delivery.

    Args:
        by_time: From :func:`time_table`.
        cohorts: The cohort keys, pooled first.
        path: The figure stem.

    Returns:
        The path written.
    """
    fig, axes = figures.panel_grid(2, n_cols=2, panel_height=2.9)
    colors = figures.group_colors([name for name in cohorts if name != ALL_COHORT])
    colors[ALL_COHORT] = figures.COLOR_BLACK
    for cohort in cohorts:
        part = by_time[by_time["cohort"] == cohort].sort_values("hours_mid")
        if part.empty:
            continue
        style = {"color": colors[cohort], "linewidth": figures.LINE_EMPHASIS if cohort == ALL_COHORT else figures.LINE_REGULAR,
                 "linestyle": "--" if cohort == ALL_COHORT else "-"}
        hours = part["hours_mid"].to_numpy()
        axes[0].plot(hours, 100.0 * part["recording_detected_share_mean"], label=_cohort_label(cohort), **style)
        if cohort != ALL_COHORT:
            axes[0].fill_between(hours, 100.0 * part["recording_detected_share_lo"],
                                 100.0 * part["recording_detected_share_hi"], color=colors[cohort],
                                 alpha=0.15, linewidth=0)
        axes[1].plot(hours, part["recording_detected_lag_mean_s"], label=_cohort_label(cohort), **style)
        if cohort != ALL_COHORT:
            axes[1].fill_between(hours, part["recording_detected_lag_lo_s"],
                                 part["recording_detected_lag_hi_s"], color=colors[cohort],
                                 alpha=0.15, linewidth=0)
    axes[0].set_ylabel("detected share of active anchors (%)")
    axes[0].set_title("How often a lag beyond 0 is detected")
    axes[1].set_ylabel("mean detected peak lag (s)")
    axes[1].set_title("How far back the detected peak sits")
    for ax in axes:
        ax.set_xlabel("hours before delivery")
        ax.set_xlim(MAX_HOURS, 0.0)
        figures.style_axes(ax)
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(loc="upper left", ncol=2)
    figures.caveat_note(fig)
    return figures.render_figure(fig, path)


# =================================================================================================
# The analysis
# =================================================================================================
def _skip(reason: str) -> Dict[str, Any]:
    """Build the result of a run this analysis cannot read.

    Args:
        reason: Why the analysis did not run.

    Returns:
        The protocol's keys, with ``skipped`` set and the reason recorded.
    """
    return {
        "n_samples": None, "composition": {}, "plan": {"capped": False},
        "skipped": True, "reason": reason, "files": [],
    }


def run_lag_detected_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select the anchors at which the model detects a lag beyond lag 0, and describe them.

    Args:
        context: The analysis context, read for the per-anchor table, the per-anchor vector
            sidecar, the per-sample table and the collection's lag block.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the selection rule, the population counts, the headline block and
        the files written. A model with neither a gate nor a UA mask, and a run without the
        per-anchor lag maps, each record a skip.
    """
    collection = context.collection
    per_anchor = getattr(collection, "per_anchor", None)
    per_sample = getattr(collection, "per_sample", None)
    gate = applied_gate(per_anchor)
    if gate is None:
        return _skip(
            "the per-anchor table carries neither event_gate nor source_open, so the model "
            "builds no posterior event gate and applies no UA mask"
        )
    lag = dict(dict(getattr(collection, "results", None) or {}).get("lag") or {})
    n_lags = int(lag.get("n_lags") or 0)
    maps = {
        name: array
        for name, array in dict(getattr(collection, "anchor_vectors", None) or {}).items()
        if np.asarray(array).ndim == 2 and np.asarray(array).shape == (len(per_anchor), n_lags)
    }
    if n_lags <= 0 or KL_MAP_KEY not in maps:
        return _skip(
            f"the run carries no {KL_MAP_KEY!r} per-anchor vector sidecar aligned with its "
            "per-anchor table, so no anchor's lag map can be read"
        )

    seconds = compensated_seconds_axis(n_lags, int(lag.get("delay_steps") or 0))
    labelled = per_recording_labels(per_sample) if per_sample is not None else pd.DataFrame()
    data = AnchorData(per_anchor, gate, maps, seconds, labelled)
    if not len(data.frame):
        return _skip("no anchor carries a finite gate value")

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))

    anchors = anchor_table(data)
    anchors.to_parquet(directory / ANCHORS_FILENAME, index=False)
    stride = anchor_stride(data.frame)
    episodes = episode_table(anchors, stride)
    per_guid = per_recording_table(data)
    summary = summary_table(data, per_guid, resamples=resamples, seed=seed)
    peaks = peak_table(data)
    profiles = profile_table(data)
    contrast = contrast_table(data, per_guid, resamples=resamples, seed=seed)
    by_time = time_table(
        data, np.arange(0.0, MAX_HOURS + TIME_BIN_HOURS, TIME_BIN_HOURS), resamples=resamples, seed=seed
    )
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)
    for table, filename in (
        (episodes, EPISODES_FILENAME), (summary, SUMMARY_FILENAME), (peaks, PEAKS_FILENAME),
        (profiles, PROFILE_FILENAME), (contrast, CONTRAST_FILENAME), (by_time, BY_TIME_FILENAME),
    ):
        table.to_csv(directory / filename, index=False)

    cohorts = data.cohorts()
    written = [
        draw_overview_figure(summary, peaks, profiles, per_guid, contrast, cohorts,
                             directory / OVERVIEW_FIGURE_STEM),
        draw_time_figure(by_time, cohorts, directory / TIME_FIGURE_STEM),
    ]

    pooled = summary[summary["cohort"] == ALL_COHORT].set_index("population")
    n_active = int(data.mask("active").sum())
    n_detected = int(data.mask("detected").sum())
    gain_row = contrast[(contrast["cohort"] == ALL_COHORT) & (contrast["readout"] == "gain_nats")]
    detected_lags = anchors["lag_s"].to_numpy(dtype=float)
    headline = {
        "n_detected": n_detected,
        "detected_share_of_scored": float(pooled.loc["detected", "anchor_share"]),
        "detected_share_of_active": float(n_detected / n_active) if n_active else float("nan"),
        "detected_recording_share_mean": float(pooled.loc["detected", "recording_share_mean"]),
        "n_recordings_with_detected": int(pooled.loc["detected", "n_recordings_with_population"]),
        "detected_median_lag_s": float(np.median(detected_lags)) if detected_lags.size else float("nan"),
        "n_episodes": int(len(episodes)),
        "episode_median_duration_s": (
            float(episodes["duration_s"].median()) if len(episodes) else float("nan")
        ),
        "gain_detected_minus_immediate_nats": (
            float(gain_row["mean_difference"].iloc[0]) if len(gain_row) else float("nan")
        ),
        "gain_detected_minus_immediate_p": (
            float(gain_row["wilcoxon_p"].iloc[0]) if len(gain_row) else float("nan")
        ),
    }
    return {
        "n_samples": (
            int(data.frame.groupby(["guid", "epoch"]).ngroups)
            if {"guid", "epoch"} <= set(data.frame.columns) else None
        ),
        "composition": {"n_recordings": int(len(data.recordings))},
        "plan": {
            "capped": False, "bootstrap_resamples": resamples, "seed": seed,
            "gate_column": APPLIED_COLUMN if APPLIED_COLUMN in per_anchor.columns else OPEN_COLUMN,
            "gain_column": data.gain_column,
        },
        "selection": {
            "gate_shut_below": GATE_SHUT_BELOW, "min_kl_nats": MIN_KL_NATS,
            "min_detected_lag_index": MIN_DETECTED_LAG_INDEX,
            "min_detected_lag_s": float(seconds[MIN_DETECTED_LAG_INDEX]) if n_lags > MIN_DETECTED_LAG_INDEX else None,
            "anchor_stride_steps": int(stride),
        },
        "populations": [
            {"key": key, "label": label, "n_anchors": int(data.mask(key).sum())}
            for key, label in POPULATIONS
        ],
        "headline": headline,
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            ANCHORS_FILENAME, EPISODES_FILENAME, SUMMARY_FILENAME, PER_RECORDING_FILENAME,
            PEAKS_FILENAME, PROFILE_FILENAME, CONTRAST_FILENAME, BY_TIME_FILENAME,
            *(Path(str(path)).name for path in written),
        ],
    }


__all__ = [
    "ANALYSIS_DIRNAME",
    "MIN_DETECTED_LAG_INDEX",
    "MIN_KL_NATS",
    "POPULATIONS",
    "AnchorData",
    "anchor_stride",
    "anchor_table",
    "applied_gate",
    "classify_anchors",
    "contrast_table",
    "episode_table",
    "peak_table",
    "per_recording_table",
    "profile_table",
    "run_lag_detected_analysis",
    "summary_table",
    "time_table",
]
