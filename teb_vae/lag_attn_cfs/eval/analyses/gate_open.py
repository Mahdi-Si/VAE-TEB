r"""The run read on the anchors where the posterior event gate is on, and on those where it is off.

The posterior applies the effective gate $g^{\mathrm{eff}}_t = g_t\,o_t$ to both of its residuals.
While the gate is off the posterior equals the prior, so the source adds no information and the
two forecast branches agree. A mean over every anchor therefore mixes two regimes, and on a run
whose gate is off for most anchors the mean mostly describes the off regime.

This analysis splits the scored anchors into four **populations** with the thresholds of
:mod:`~teb_vae.lag_attn_cfs.gate_states`, and reports each readout once per population:

* ``all`` -- every scored anchor;
* ``gate_on`` -- $g^{\mathrm{eff}}_t \ge g_{\mathrm{shut}}$, the anchors the source can move;
* ``fully_open`` -- $g^{\mathrm{eff}}_t \ge g_{\mathrm{open}}$, a subset of ``gate_on``;
* ``gate_off`` -- $g^{\mathrm{eff}}_t < g_{\mathrm{shut}}$, shut by the model or by the UA mask.

For each population it reports the source-conditioned KL $K_t$, the forecast gain, and the lag
structure. The lag structure comes from the two per-anchor lag maps of the collection pass: the
KL attribution $K_{t,\ell}$, which sums over lags to $K_t$, and the attention weight $a_{t,\ell}$,
which sums to $1$. It then reads the populations against three clocks: the time before delivery,
the time since the last contraction onset, and the gate value itself. A fourth reading follows
each population through labour per clinical class with one value per recording, and tests the
class difference in every half hour.

**Two aggregation rules, and each table says which one it uses.** A summary, a lag profile and
a class trajectory are statistics over recordings of the recording's own anchor mean, so a long
recording does not outvote a short one. A clock profile is an anchor-level mean per bin, because
one recording rarely spans every bin.

**The contribution columns add up.** For a readout $x$, the contribution of population $P$ is
$\frac{1}{N}\sum_{t \in P} x_t$ with $N$ the recording's scored anchors. The ``gate_on`` and
``gate_off`` contributions sum to the ``all`` mean, so they say how much of the run's mean KL or
gain each regime supplies.

**A lag axis here is stored-coefficient time**, as everywhere in this package. The stored timeline
is canonical, and this analysis applies no correction to it.

The analysis reads the per-anchor table and the per-anchor vector sidecar only, so it runs against
a finished run directory with no checkpoint. A model with neither a gate nor a UA mask records a
skip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
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
from teb_vae.lag_attn_cfs.gate_states import (
    GATE_OPEN_FROM,
    GATE_SHUT_BELOW,
    MODEL_SHUT,
    OPEN,
    PARTLY_OPEN,
    gate_states,
    run_lengths,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "gate_open"

#: What it writes.
SUMMARY_FILENAME = "gate_open_summary.csv"
PER_RECORDING_FILENAME = "gate_open_per_recording.csv"
LAG_PROFILE_FILENAME = "gate_open_lag_profile.csv"
BY_TIME_FILENAME = "gate_open_by_time_to_delivery.csv"
BY_CONTRACTION_FILENAME = "gate_open_by_contraction_age.csv"
BY_GATE_FILENAME = "gate_open_by_gate.csv"
LAG_BY_TIME_FILENAME = "gate_open_lag_by_time_to_delivery.csv"
LAG_BY_CONTRACTION_FILENAME = "gate_open_lag_by_contraction_age.csv"
LAG_BY_GATE_FILENAME = "gate_open_lag_by_gate.csv"
TRAJECTORY_FILENAME = "gate_open_trajectory.csv"
TRAJECTORY_TESTS_FILENAME = "gate_open_trajectory_significance.csv"
RUNS_FILENAME = "gate_open_run_durations.csv"
DISTRIBUTIONS_FILENAME = "gate_open_distributions.csv"
FIGURE_STEM = "gate_open"
LAG_FIGURE_STEM = "gate_open_lags"
TRAJECTORY_FIGURE_STEM = "gate_open_trajectory"

#: The per-anchor columns the collection pass writes.
APPLIED_COLUMN = "event_gate"
OPEN_COLUMN = "source_open"
KL_COLUMN = "kld_per_t"
ARGMAX_COLUMN = "argmax_lag"

#: The two per-anchor lag maps of the vector sidecar.
KL_MAP_KEY = "kl_lag_map"
ATTENTION_MAP_KEY = "attention_lag_map"

#: The per-anchor readouts averaged per population: the table column, then the name this
#: analysis's tables give it. A column the table lacks is reported as ``NaN``.
SCALAR_COLUMNS: Tuple[Tuple[str, str], ...] = (
    (KL_COLUMN, "kl_nats"),
    ("mc_pred_gap", "gain_mc_nats"),
    ("mean_pred_gap", "gain_mean_nats"),
    ("pred_gap", "gain_draw_nats"),
    ("mc_nll_base_block", "nll_base_nats"),
    ("mc_nll_full_block", "nll_full_nats"),
)

#: The populations, in the order every table and legend lists them: key, then label.
POPULATIONS: Tuple[Tuple[str, str], ...] = (
    ("all", "all scored anchors"),
    ("gate_on", f"gate on ($g \\geq {GATE_SHUT_BELOW:g}$)"),
    ("fully_open", f"fully open ($g \\geq {GATE_OPEN_FROM:g}$)"),
    ("gate_off", f"gate off ($g < {GATE_SHUT_BELOW:g}$)"),
)
POPULATION_KEYS: Tuple[str, ...] = tuple(key for key, _ in POPULATIONS)
POPULATION_LABELS: Dict[str, str] = dict(POPULATIONS)

#: One colour per population, so a population has one colour on every panel.
POPULATION_COLORS: Dict[str, str] = {
    "all": figures.COLOR_GRAY,
    "gate_on": figures.COLOR_BLUE,
    "fully_open": figures.COLOR_GREEN,
    "gate_off": figures.COLOR_VERMILLION,
}

#: The cohort key of the rows that pool every clinical class.
ALL_COHORT = "all"

#: Width of a time-before-delivery bin, in hours, and the last bin edge.
TIME_BIN_HOURS = 0.5
MAX_HOURS = 12.0

#: The fewest anchors of a population a recording needs in one window of the class trajectory.
#: Ten anchors are 40 s, so a recording's value in a window never rests on a few anchors.
MIN_WINDOW_ANCHORS = 10

#: The significance level of the per-window class tests, after the Holm adjustment.
TRAJECTORY_ALPHA = 0.05

#: The metrics the class trajectory follows, in table order.
TRAJECTORY_METRICS: Tuple[str, ...] = (
    "kl_nats", "gain_mc_nats", "gain_mean_nats", "share_of_anchors", "kl_centroid_s",
    "kl_lag0_share",
)

#: Width of a contraction-age bin, in seconds, and the last bin edge.
CONTRACTION_BIN_S = 15.0
MAX_CONTRACTION_AGE_S = 300.0

#: Width of a gate bin. Fixed edges on the gate value rather than quantiles, so a bin means the
#: same gate on every run.
GATE_BIN_WIDTH = 0.1

#: Width of a run-duration bin, in seconds.
RUN_BIN_S = 8.0

#: Histogram edges of the two per-anchor distributions the analysis writes. The first and the
#: last bin also hold what falls outside, so the counts of one population sum to its anchors.
KL_EDGES = np.linspace(0.0, 20.0, 81)
GAIN_EDGES = np.linspace(-60.0, 150.0, 106)

#: The smallest lag share the lag page draws. Its heatmaps and share panels use a logarithmic
#: scale, and a share below this floor is drawn at the floor.
HEATMAP_SHARE_FLOOR = 1e-3

#: The per-recording columns resolved by cohort, all of the ``gate_on`` population.
GROUPED_METRICS: Tuple[str, ...] = (
    "gate_on_frac",
    "gate_on_kl_nats",
    "gate_on_gain_mc_nats",
    "gate_on_kl_contribution_nats",
    "gate_on_gain_mc_contribution_nats",
    "gate_on_kl_centroid_s",
    "gate_on_kl_lag0_share",
    "gate_off_gain_mc_nats",
)


# =================================================================================================
# The anchors and their populations
# =================================================================================================
def applied_gate(per_anchor: pd.DataFrame) -> Optional[np.ndarray]:
    r"""Return the effective gate $g^{\mathrm{eff}}_t$ of every per-anchor row.

    Args:
        per_anchor: The collection pass's per-anchor table.

    Returns:
        The applied gate, or $o_t$ itself on a model that masks the UA but learns no gate. ``None``
        when the table carries neither column, which means the model builds neither.
    """
    if per_anchor is None or len(per_anchor) == 0:
        return None
    if APPLIED_COLUMN in per_anchor.columns:
        return per_anchor[APPLIED_COLUMN].to_numpy(dtype=np.float64)
    if OPEN_COLUMN in per_anchor.columns:
        return per_anchor[OPEN_COLUMN].to_numpy(dtype=np.float64)
    return None


def population_masks(gate: np.ndarray) -> Dict[str, np.ndarray]:
    r"""Assign every anchor to the populations it belongs to.

    Args:
        gate: The effective gate $g^{\mathrm{eff}}_t$. A ``NaN`` entry belongs to no population.

    Returns:
        One boolean mask per key of :data:`POPULATIONS`. ``gate_on`` and ``gate_off`` partition
        ``all``, and ``fully_open`` is a subset of ``gate_on``.
    """
    states = gate_states(gate)
    return {
        "all": states >= 0,
        "gate_on": states >= PARTLY_OPEN,
        "fully_open": states == OPEN,
        "gate_off": states == MODEL_SHUT,
    }


def group_sums(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of a matrix within each group.

    Args:
        values: The matrix, $(N, K)$.
        codes: The group of each row, $(N,)$. A negative code leaves the row out.
        n_groups: How many groups there are.

    Returns:
        The sums, $(G, K)$, as ``float64``.
    """
    kept = np.flatnonzero(codes >= 0)
    indicator = sparse.csr_matrix(
        (np.ones(kept.size, dtype=np.float64), (codes[kept], kept)),
        shape=(int(n_groups), int(len(codes))),
    )
    return np.asarray(indicator @ values, dtype=np.float64)


def lag_summaries(
    kl_profile: np.ndarray, attention_profile: np.ndarray, seconds: np.ndarray
) -> Dict[str, np.ndarray]:
    r"""Reduce lag profiles to the three scalars the tables carry.

    The centroid of a profile $p_\ell$ on the axis $\tau_\ell$ is
    $$\bar\tau = \frac{\sum_\ell \tau_\ell\, p^{+}_\ell}{\sum_\ell p^{+}_\ell}, \qquad
    p^{+}_\ell = \max(p_\ell, 0).$$

    Args:
        kl_profile: KL attribution per lag, $(G, L)$.
        attention_profile: Attention weight per lag, $(G, L)$.
        seconds: The lag axis $\tau_\ell$ in seconds, $(L,)$.

    Returns:
        ``kl_lag0_share`` (the share of the profile's total at lag $0$), ``kl_centroid_s`` and
        ``attention_centroid_s``, each $(G,)$ and ``NaN`` where the profile has no positive mass.
    """
    rectified = np.clip(np.nan_to_num(kl_profile), 0.0, None)
    total = rectified.sum(axis=1)
    attention = np.clip(np.nan_to_num(attention_profile), 0.0, None)
    attention_total = attention.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "kl_lag0_share": np.where(total > 0, rectified[:, 0] / total, np.nan),
            "kl_centroid_s": np.where(total > 0, rectified @ seconds / total, np.nan),
            "attention_centroid_s": np.where(
                attention_total > 0, attention @ seconds / attention_total, np.nan
            ),
        }


class AnchorData:
    """The per-anchor arrays every table of this analysis is reduced from.

    Attributes:
        frame: The per-anchor rows that carry a finite gate, re-indexed from zero.
        gate: The effective gate of each row.
        masks: The population masks of :func:`population_masks`.
        scalars: One column per entry of :data:`SCALAR_COLUMNS`, $(N, S)$, ``NaN`` where absent.
        scalars_filled: ``scalars`` with $0$ in place of a value that is not finite, for a sum.
        scalars_seen: $1$ where ``scalars`` is finite and $0$ elsewhere, for the count of that sum.
        kl_map: The KL attribution per lag, $(N, L)$.
        attention_map: The attention weight per lag, $(N, L)$, or zeros when the sidecar lacks it.
        argmax: The lag index holding each anchor's largest KL attribution.
        seconds: The lag axis in seconds, $(L,)$.
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
        self.masks = population_masks(self.gate)
        self.scalars = np.column_stack([
            self.frame[column].to_numpy(dtype=np.float64) if column in self.frame.columns
            else np.full(len(self.frame), np.nan)
            for column, _ in SCALAR_COLUMNS
        ])
        # Every reduction sums the readouts and counts the finite ones. Both arrays depend on the
        # anchors alone, so they are built once here and not once per population, cohort and clock.
        self.scalars_filled = np.nan_to_num(self.scalars)
        self.scalars_seen = np.isfinite(self.scalars).astype(np.float64)
        self.kl_map = np.asarray(maps[KL_MAP_KEY], dtype=np.float32)[finite]
        self.attention_map = (
            np.asarray(maps[ATTENTION_MAP_KEY], dtype=np.float32)[finite]
            if ATTENTION_MAP_KEY in maps else np.zeros_like(self.kl_map)
        )
        self.argmax = (
            self.frame[ARGMAX_COLUMN].to_numpy(dtype=np.int64)
            if ARGMAX_COLUMN in self.frame.columns else self.kl_map.argmax(axis=1)
        )
        self.seconds = np.asarray(seconds, dtype=np.float64)
        codes, recordings = pd.factorize(self.frame["guid"], sort=True)
        self.recording_codes = codes.astype(np.int64)
        self.recordings = [str(name) for name in recordings]
        classes = (
            labelled[labels.CLASS_COLUMN]
            if labels.CLASS_COLUMN in getattr(labelled, "columns", []) else pd.Series(dtype=object)
        )
        self.cohort_of_recording = [
            None if pd.isna(classes.get(name, None)) else str(classes.get(name))
            for name in self.recordings
        ]

    @property
    def n_lags(self) -> int:
        """The lag window width $L$."""
        return int(self.kl_map.shape[1])

    def cohorts(self) -> List[str]:
        """List the cohort keys.

        Returns:
            The pooled cohort first, then each clinical class the recordings carry, sorted.
        """
        present = sorted({name for name in self.cohort_of_recording if name is not None})
        return [ALL_COHORT, *present]

    def recordings_in(self, cohort: str) -> np.ndarray:
        """Select the recordings of one cohort.

        Args:
            cohort: A key of :meth:`cohorts`.

        Returns:
            A boolean mask over ``recordings``.
        """
        if cohort == ALL_COHORT:
            return np.ones(len(self.recordings), dtype=bool)
        return np.asarray([name == cohort for name in self.cohort_of_recording], dtype=bool)


# =================================================================================================
# Per recording, then over recordings
# =================================================================================================
def per_recording_reductions(data: AnchorData) -> Dict[str, Dict[str, np.ndarray]]:
    """Reduce the anchors to one row per recording and population.

    Args:
        data: The per-anchor arrays.

    Returns:
        Per population, a mapping from quantity to an array over recordings: ``n_anchors``,
        ``frac``, one mean and one ``*_contribution_*`` per scalar readout, ``gate_mean``, the two
        lag profiles ``kl_profile`` and ``attention_profile`` $(R, L)$, the argmax-lag histogram
        ``argmax_share`` $(R, L)$, ``argmax_lag0_share`` and the scalars of :func:`lag_summaries`.
    """
    n_recordings = len(data.recordings)
    total = np.bincount(data.recording_codes[data.masks["all"]], minlength=n_recordings).astype(float)
    reduced: Dict[str, Dict[str, np.ndarray]] = {}
    for key in POPULATION_KEYS:
        mask = data.masks[key]
        codes = np.where(mask, data.recording_codes, -1)
        count = np.bincount(data.recording_codes[mask], minlength=n_recordings).astype(float)
        sums = group_sums(data.scalars_filled, codes, n_recordings)
        seen = group_sums(data.scalars_seen, codes, n_recordings)
        kl_sum = group_sums(data.kl_map, codes, n_recordings)
        attention_sum = group_sums(data.attention_map, codes, n_recordings)
        argmax_count = np.bincount(
            data.recording_codes[mask] * data.n_lags + data.argmax[mask],
            minlength=n_recordings * data.n_lags,
        ).reshape(n_recordings, data.n_lags).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            row: Dict[str, np.ndarray] = {
                "n_anchors": count,
                "frac": np.where(total > 0, count / total, np.nan),
                "gate_mean": np.where(
                    count > 0,
                    np.bincount(data.recording_codes[mask], weights=data.gate[mask],
                                minlength=n_recordings) / count,
                    np.nan,
                ),
                "kl_profile": np.where(count[:, None] > 0, kl_sum / count[:, None], np.nan),
                "attention_profile": np.where(
                    count[:, None] > 0, attention_sum / count[:, None], np.nan
                ),
                "argmax_share": np.where(
                    count[:, None] > 0, argmax_count / count[:, None], np.nan
                ),
            }
            for index, (_, name) in enumerate(SCALAR_COLUMNS):
                row[name] = np.where(seen[:, index] > 0, sums[:, index] / seen[:, index], np.nan)
                stem = name[: -len("_nats")] if name.endswith("_nats") else name
                row[f"{stem}_contribution_nats"] = np.where(
                    total > 0, sums[:, index] / total, np.nan
                )
        row["argmax_lag0_share"] = row["argmax_share"][:, 0]
        row.update(lag_summaries(row["kl_profile"], row["attention_profile"], data.seconds))
        reduced[key] = row
    return reduced


def recording_scalar_names() -> List[str]:
    """List the per-recording scalar quantities of one population.

    Returns:
        The quantity names, in the column order of the per-recording table.
    """
    names = ["n_anchors", "frac", "gate_mean"]
    for _, name in SCALAR_COLUMNS:
        names.append(name)
    for _, name in SCALAR_COLUMNS:
        stem = name[: -len("_nats")] if name.endswith("_nats") else name
        names.append(f"{stem}_contribution_nats")
    names.extend(["kl_lag0_share", "kl_centroid_s", "attention_centroid_s", "argmax_lag0_share"])
    return names


def per_recording_table(
    data: AnchorData, reduced: Dict[str, Dict[str, np.ndarray]], labelled: pd.DataFrame
) -> pd.DataFrame:
    """Lay the per-recording reductions out as one wide row per recording.

    Args:
        data: The per-anchor arrays, for the recording identifiers.
        reduced: From :func:`per_recording_reductions`.
        labelled: The recording labels, indexed by ``guid``.

    Returns:
        One row per recording, indexed by ``guid``, with a ``<population>_<quantity>`` column per
        scalar quantity and the cohort labels.
    """
    columns: Dict[str, np.ndarray] = {}
    for key in POPULATION_KEYS:
        for name in recording_scalar_names():
            columns[f"{key}_{name}"] = reduced[key][name]
    table = pd.DataFrame(columns, index=pd.Index(data.recordings, name="guid"))
    for name in getattr(labelled, "columns", []):
        table[name] = labelled[name].reindex(table.index)
    return table


def _interval(values: np.ndarray, *, resamples: int, seed: int) -> Tuple[float, float, float, int]:
    """The mean of a per-recording vector with its percentile bootstrap interval.

    Args:
        values: One value per recording. Non-finite entries are dropped.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        ``(mean, lo, hi, n)``. The mean is the plain mean whenever one value is finite, and the
        bounds are ``NaN`` below three recordings.
    """
    finite = values[np.isfinite(values)]
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


def summary_table(
    data: AnchorData, reduced: Dict[str, Dict[str, np.ndarray]], *, resamples: int, seed: int
) -> pd.DataFrame:
    """Summarise each population over recordings, per cohort.

    Args:
        data: The per-anchor arrays.
        reduced: From :func:`per_recording_reductions`.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per population and cohort: the anchor and recording counts, then for each scalar
        quantity the mean over recordings with its interval.
    """
    rows: List[Dict[str, Any]] = []
    quantities = [name for name in recording_scalar_names() if name != "n_anchors"]
    for cohort in data.cohorts():
        members = data.recordings_in(cohort)
        for key in POPULATION_KEYS:
            row: Dict[str, Any] = {
                "population": key,
                "population_label": POPULATION_LABELS[key],
                "cohort": cohort,
                "n_anchors": int(np.nansum(reduced[key]["n_anchors"][members])),
                "n_recordings": int(np.sum(reduced[key]["n_anchors"][members] > 0)),
            }
            for name in quantities:
                mean, lo, hi, _ = _interval(
                    reduced[key][name][members], resamples=resamples, seed=seed
                )
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = mean, lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def lag_profile_table(
    data: AnchorData, reduced: Dict[str, Dict[str, np.ndarray]], *, resamples: int, seed: int
) -> pd.DataFrame:
    r"""The lag profile of each population over recordings, per cohort.

    Args:
        data: The per-anchor arrays.
        reduced: From :func:`per_recording_reductions`.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per population, cohort and lag: ``kl_nats`` (the mean over recordings of the
        recording's mean $K_{t,\ell}$) with its interval, ``kl_share`` (that profile divided by
        its sum over lags), ``attention`` (the mean attention weight) and ``argmax_share`` (the
        mean share of anchors whose largest attribution sits at this lag).
    """
    rows: List[pd.DataFrame] = []
    generator = np.random.default_rng(int(seed))
    for cohort in data.cohorts():
        members = data.recordings_in(cohort)
        for key in POPULATION_KEYS:
            used = members & (reduced[key]["n_anchors"] > 0)
            profile = reduced[key]["kl_profile"][used]
            if profile.shape[0] == 0:
                continue
            mean = profile.mean(axis=0)
            if profile.shape[0] >= 3:
                draws = generator.integers(0, profile.shape[0], size=(int(resamples), profile.shape[0]))
                # One matrix product per resample block keeps the peak memory at (B, L).
                weights = np.zeros((int(resamples), profile.shape[0]))
                np.add.at(weights, (np.arange(int(resamples))[:, None], draws), 1.0)
                boot = weights @ profile / profile.shape[0]
                lo, hi = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
            else:
                lo = hi = np.full(data.n_lags, np.nan)
            rectified = np.clip(mean, 0.0, None)
            total = rectified.sum()
            rows.append(pd.DataFrame({
                "population": key,
                "cohort": cohort,
                "lag_step": np.arange(data.n_lags),
                "lag_s": data.seconds,
                "n_recordings": int(profile.shape[0]),
                "kl_nats": mean,
                "kl_nats_lo": lo,
                "kl_nats_hi": hi,
                "kl_share": rectified / total if total > 0 else np.full(data.n_lags, np.nan),
                "attention": reduced[key]["attention_profile"][used].mean(axis=0),
                "argmax_share": reduced[key]["argmax_share"][used].mean(axis=0),
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =================================================================================================
# Against a clock
# =================================================================================================
def clock_tables(
    data: AnchorData,
    clock: np.ndarray,
    edges: np.ndarray,
    *,
    populations: Sequence[str] = POPULATION_KEYS,
    cohorts: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Read the populations in bins of one clock, at the anchor level.

    Args:
        data: The per-anchor arrays.
        clock: The clock value of each anchor. A non-finite value leaves the anchor out.
        edges: Bin edges, ascending.
        populations: The populations to read. Each one costs a reduction of both lag maps, so a
            caller that needs one population names it.
        cohorts: The cohorts to read. ``None`` reads the pooled cohort and every clinical class.

    Returns:
        ``(scalars, lags)``. ``scalars`` has one row per population, cohort and non-empty bin: the
        anchor and recording counts, ``share_of_anchors`` (the population's anchors over all
        anchors of the bin and cohort), the mean gate, the mean of each scalar readout and the
        scalars of :func:`lag_summaries`. ``lags`` has one row per population, bin and lag for the
        pooled cohort: ``kl_nats``, ``kl_share`` and ``attention``.
    """
    n_bins = len(edges) - 1
    position = np.digitize(clock, edges) - 1
    inside = np.isfinite(clock) & (position >= 0) & (position < n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cohort_of_anchor = np.asarray(data.cohort_of_recording, dtype=object)[data.recording_codes]
    scalar_rows: List[Dict[str, Any]] = []
    lag_rows: List[pd.DataFrame] = []
    for cohort in (data.cohorts() if cohorts is None else cohorts):
        in_cohort = inside if cohort == ALL_COHORT else inside & (cohort_of_anchor == cohort)
        cohort_total = np.bincount(position[in_cohort], minlength=n_bins).astype(float)
        for key in populations:
            mask = in_cohort & data.masks[key]
            codes = np.where(mask, position, -1)
            count = np.bincount(position[mask], minlength=n_bins).astype(float)
            if not count.any():
                continue
            sums = group_sums(data.scalars_filled, codes, n_bins)
            seen = group_sums(data.scalars_seen, codes, n_bins)
            kl_sum = group_sums(data.kl_map, codes, n_bins)
            attention_sum = group_sums(data.attention_map, codes, n_bins)
            gate_sum = np.bincount(position[mask], weights=data.gate[mask], minlength=n_bins)
            recordings = (
                pd.Series(data.recording_codes[mask]).groupby(position[mask]).nunique()
                .reindex(range(n_bins)).fillna(0).to_numpy()
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                kl_profile = kl_sum / count[:, None]
                attention_profile = attention_sum / count[:, None]
                means = sums / seen
            summaries = lag_summaries(kl_profile, attention_profile, data.seconds)
            for index in np.flatnonzero(count > 0):
                row: Dict[str, Any] = {
                    "population": key,
                    "cohort": cohort,
                    "bin_lo": float(edges[index]),
                    "bin_hi": float(edges[index + 1]),
                    "bin_center": float(centres[index]),
                    "n_anchors": int(count[index]),
                    "n_recordings": int(recordings[index]),
                    "share_of_anchors": float(count[index] / cohort_total[index]),
                    "gate_mean": float(gate_sum[index] / count[index]),
                }
                for column, (_, name) in enumerate(SCALAR_COLUMNS):
                    row[name] = float(means[index, column])
                for name, values in summaries.items():
                    row[name] = float(values[index])
                scalar_rows.append(row)
            if cohort != ALL_COHORT:
                continue
            rectified = np.clip(np.nan_to_num(kl_profile), 0.0, None)
            total = rectified.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(total > 0, rectified / total, np.nan)
            for index in np.flatnonzero(count > 0):
                lag_rows.append(pd.DataFrame({
                    "population": key,
                    "bin_center": float(centres[index]),
                    "n_anchors": int(count[index]),
                    "lag_step": np.arange(data.n_lags),
                    "lag_s": data.seconds,
                    "kl_nats": kl_profile[index],
                    "kl_share": share[index],
                    "attention": attention_profile[index],
                }))
    return (
        pd.DataFrame(scalar_rows),
        pd.concat(lag_rows, ignore_index=True) if lag_rows else pd.DataFrame(),
    )


def gate_tables(data: AnchorData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the readouts and the lag structure in fixed bins of the gate value.

    Args:
        data: The per-anchor arrays.

    Returns:
        ``(scalars, lags)`` in the layout of :func:`clock_tables`, restricted to the ``all``
        population and the pooled cohort, because the gate bin already is the selection.
    """
    edges = np.linspace(0.0, 1.0, int(round(1.0 / GATE_BIN_WIDTH)) + 1)
    # The last bin is closed on the right, so a gate of exactly one is counted.
    clock = np.minimum(data.gate, np.nextafter(1.0, 0.0))
    scalars, lags = clock_tables(data, clock, edges, populations=("all",), cohorts=(ALL_COHORT,))
    if not scalars.empty:
        scalars = scalars.drop(columns=["population", "cohort"]).reset_index(drop=True)
        scalars["share_of_anchors"] = scalars["n_anchors"] / scalars["n_anchors"].sum()
    if not lags.empty:
        lags = lags.drop(columns=["population"]).reset_index(drop=True)
    return scalars, lags


# =================================================================================================
# The class trajectories over labour, per recording
# =================================================================================================
def trajectory_tables(
    data: AnchorData, hours: np.ndarray, edges: np.ndarray, *, alpha: float = TRAJECTORY_ALPHA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Follow each population through labour per clinical class, with one value per recording.

    In each window of the clock, a recording gives one value of a metric: the mean over its own
    anchors of the population in that window. A recording counts in a window only when it has at
    least :data:`MIN_WINDOW_ANCHORS` anchors of the population there, so a value never rests on a
    few anchors. The trajectory is then the median, the quartiles and the mean over the
    recordings of each class.

    The class difference is tested per window with the Kruskal-Wallis test over the classes that
    have at least three recordings in the window. The $p$-values of one population and one metric
    are Holm-adjusted over its windows.

    Args:
        data: The per-anchor arrays.
        hours: The time before delivery of each anchor, in hours.
        edges: The window edges in hours, ascending.
        alpha: The significance level the adjusted $p$-value is compared with.

    Returns:
        ``(trajectory, tests)``. ``trajectory`` has one row per population, metric, cohort and
        window: the recording count and the mean, the quartiles and the median over recordings.
        ``tests`` has one row per population, metric and window: the classes tested, the
        statistic, the raw and the adjusted $p$-value, and the decision.
    """
    n_bins = len(edges) - 1
    n_recordings = len(data.recordings)
    n_cells = n_recordings * n_bins
    position = np.digitize(hours, edges) - 1
    inside = np.isfinite(hours) & (position >= 0) & (position < n_bins)
    cell = np.where(inside, data.recording_codes * n_bins + np.clip(position, 0, n_bins - 1), -1)
    total = np.bincount(cell[inside & data.masks["all"]], minlength=n_cells).astype(float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    scalar_names = [name for _, name in SCALAR_COLUMNS]
    cohorts = data.cohorts()
    classes = [name for name in cohorts if name != ALL_COHORT]
    members = {name: data.recordings_in(name) for name in cohorts}

    trajectory_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for key in POPULATION_KEYS:
        mask = inside & data.masks[key]
        codes = np.where(mask, cell, -1)
        count = np.bincount(cell[mask], minlength=n_cells).astype(float)
        enough = count >= MIN_WINDOW_ANCHORS
        sums = group_sums(data.scalars_filled, codes, n_cells)
        seen = group_sums(data.scalars_seen, codes, n_cells)
        kl_profile = group_sums(data.kl_map, codes, n_cells)
        summaries = lag_summaries(kl_profile, np.zeros_like(kl_profile), data.seconds)
        with np.errstate(invalid="ignore", divide="ignore"):
            values: Dict[str, np.ndarray] = {
                "share_of_anchors": np.where(total >= MIN_WINDOW_ANCHORS, count / total, np.nan),
            }
            for name in ("kl_nats", "gain_mc_nats", "gain_mean_nats"):
                column = scalar_names.index(name)
                values[name] = np.where(
                    enough & (seen[:, column] > 0), sums[:, column] / seen[:, column], np.nan
                )
        for name in ("kl_centroid_s", "kl_lag0_share"):
            values[name] = np.where(enough, summaries[name], np.nan)

        for metric in TRAJECTORY_METRICS:
            field = values[metric].reshape(n_recordings, n_bins)
            raw_p: List[float] = []
            pending: List[Dict[str, Any]] = []
            for index in range(n_bins):
                column = field[:, index]
                for cohort in cohorts:
                    finite = column[members[cohort] & np.isfinite(column)]
                    if not finite.size:
                        continue
                    trajectory_rows.append({
                        "population": key, "metric": metric, "cohort": cohort,
                        "time_bin": index, "bin_center_h": float(centres[index]),
                        "bin_lo_h": float(edges[index]), "bin_hi_h": float(edges[index + 1]),
                        "n_recordings": int(finite.size), "mean": float(finite.mean()),
                        "q25": float(np.quantile(finite, 0.25)), "median": float(np.median(finite)),
                        "q75": float(np.quantile(finite, 0.75)),
                    })
                samples = {
                    name: column[members[name] & np.isfinite(column)] for name in classes
                }
                samples = {
                    name: sample for name, sample in samples.items()
                    if sample.size >= shared_stats.MIN_GROUP_SIZE
                }
                if not samples:
                    continue
                outcome = shared_stats.kruskal_across_groups(samples)
                raw_p.append(float(outcome["p_value"]))
                pending.append({
                    "population": key, "metric": metric, "time_bin": index,
                    "bin_center_h": float(centres[index]), "n_classes": len(samples),
                    "classes": "|".join(samples),
                    "n_recordings": int(sum(sample.size for sample in samples.values())),
                    "statistic": float(outcome["statistic"]), "p_value": float(outcome["p_value"]),
                })
            for record, adjusted in zip(pending, shared_stats.holm_adjust(raw_p)):
                record["p_holm"] = float(adjusted)
                record["significant"] = bool(np.isfinite(adjusted) and adjusted < alpha)
                record["alpha"] = float(alpha)
                test_rows.append(record)
    return pd.DataFrame(trajectory_rows), pd.DataFrame(test_rows)


# =================================================================================================
# Runs and distributions
# =================================================================================================
def run_durations(data: AnchorData) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Measure every run of consecutive gate-on anchors, and of gate-off anchors, per segment.

    A run ends at a missing anchor as well as at a change of state. A segment holds a bounded
    number of anchors, so a run that touches a segment edge is cut short, and the count of such
    runs is returned beside the durations.

    Args:
        data: The per-anchor arrays, with ``guid``, ``epoch`` and ``anchor``.

    Returns:
        ``(durations, censored)``: the run durations in seconds and the number of runs that touch
        a segment edge, each keyed by ``gate_on`` and ``gate_off``.
    """
    durations: Dict[str, List[np.ndarray]] = {"gate_on": [], "gate_off": []}
    censored = {"gate_on": 0, "gate_off": 0}
    if not {"guid", "epoch", "anchor"} <= set(data.frame.columns):
        return {key: np.zeros(0) for key in durations}, censored
    anchors = data.frame["anchor"].to_numpy(dtype=np.int64)
    segment_codes = data.frame.groupby(["guid", "epoch"], sort=False).ngroup().to_numpy()
    order = np.lexsort((anchors, segment_codes))
    boundaries = np.flatnonzero(np.diff(segment_codes[order])) + 1
    for rows in np.split(order, boundaries):
        steps = anchors[rows]
        low, width = int(steps.min()), int(steps.max() - steps.min() + 1)
        for key in durations:
            dense = np.zeros(width, dtype=bool)
            dense[steps[data.masks[key][rows]] - low] = True
            lengths = run_lengths(dense)
            if lengths.size:
                durations[key].append(lengths * SECONDS_PER_STEP)
                # A run that fills the whole segment touches both edges and is still one run.
                censored[key] += int(dense[0]) + int(dense[-1]) - int(dense.all())
    return (
        {key: (np.concatenate(parts) if parts else np.zeros(0)) for key, parts in durations.items()},
        censored,
    )


def run_duration_table(durations: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Histogram the run durations.

    Args:
        durations: From :func:`run_durations`.

    Returns:
        One row per state and duration bin: the count of runs and the share of the state's
        anchors that sit in runs of that duration.
    """
    longest = max([float(values.max()) for values in durations.values() if values.size] + [RUN_BIN_S])
    edges = np.arange(0.0, longest + 2 * RUN_BIN_S, RUN_BIN_S)
    rows: List[pd.DataFrame] = []
    for key, values in durations.items():
        if not values.size:
            continue
        # A run of n anchors lasts n steps, so it is binned at its own duration, edge-inclusive.
        position = np.clip(np.digitize(values - 0.5 * SECONDS_PER_STEP, edges) - 1, 0, len(edges) - 2)
        count = np.bincount(position, minlength=len(edges) - 1)
        time = np.bincount(position, weights=values, minlength=len(edges) - 1)
        rows.append(pd.DataFrame({
            "state": key,
            "duration_lo_s": edges[:-1],
            "duration_hi_s": edges[1:],
            "n_runs": count,
            "share_of_runs": count / max(int(count.sum()), 1),
            "share_of_time": time / max(float(time.sum()), 1e-12),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def distribution_table(data: AnchorData) -> pd.DataFrame:
    """Histogram the per-anchor KL and forecast gain of each population.

    Args:
        data: The per-anchor arrays.

    Returns:
        One row per quantity, population and bin: the count and the density. The first and the
        last bin also hold the values outside the edges.
    """
    rows: List[pd.DataFrame] = []
    names = [name for _, name in SCALAR_COLUMNS]
    for quantity, edges in (("kl_nats", KL_EDGES), ("gain_mc_nats", GAIN_EDGES)):
        values = data.scalars[:, names.index(quantity)]
        for key in POPULATION_KEYS:
            selected = values[data.masks[key] & np.isfinite(values)]
            if not selected.size:
                continue
            clipped = np.clip(selected, edges[0], np.nextafter(edges[-1], edges[0]))
            count, _ = np.histogram(clipped, bins=edges)
            rows.append(pd.DataFrame({
                "quantity": quantity,
                "population": key,
                "bin_lo": edges[:-1],
                "bin_hi": edges[1:],
                "count": count,
                "density": count / (selected.size * np.diff(edges)),
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =================================================================================================
# Figures
# =================================================================================================
def _bars_by_population(
    ax: Any, summary: pd.DataFrame, quantity: str, cohorts: Sequence[str], *, title: str, ylabel: str
) -> None:
    """Draw one quantity per population as grouped bars, one bar per cohort.

    Args:
        ax: The target axes.
        summary: From :func:`summary_table`.
        quantity: The summary column to draw. Its ``_lo`` and ``_hi`` columns give the error bar.
        cohorts: The cohort keys, pooled first.
        title: The panel title.
        ylabel: The y axis label.
    """
    colours = figures.group_colors([c for c in cohorts if c != ALL_COHORT])
    width = 0.8 / max(len(cohorts), 1)
    positions = np.arange(len(POPULATION_KEYS))
    for index, cohort in enumerate(cohorts):
        rows = summary[summary["cohort"] == cohort].set_index("population").reindex(POPULATION_KEYS)
        point = rows[quantity].to_numpy(dtype=float)
        lo = rows[f"{quantity}_lo"].to_numpy(dtype=float)
        hi = rows[f"{quantity}_hi"].to_numpy(dtype=float)
        error = np.vstack([np.clip(point - lo, 0, None), np.clip(hi - point, 0, None)])
        offset = (index - 0.5 * (len(cohorts) - 1)) * width
        ax.bar(positions + offset, np.nan_to_num(point), width=width * 0.92,
               color=colours.get(cohort, figures.COLOR_GRAY), alpha=0.88,
               label="all classes" if cohort == ALL_COHORT else cohort,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(positions + offset, np.nan_to_num(point), yerr=np.nan_to_num(error),
                    fmt="none", ecolor=figures.COLOR_BLACK, elinewidth=0.7, capsize=1.5)
    ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([key.replace("_", " ") for key in POPULATION_KEYS])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)


def _clock_lines(
    ax: Any, table: pd.DataFrame, quantity: str, *, title: str, xlabel: str, ylabel: str,
    invert: bool = False, populations: Sequence[str] = POPULATION_KEYS,
) -> None:
    """Draw one quantity against a clock, one line per population, for the pooled cohort.

    Args:
        ax: The target axes.
        table: The scalar table of :func:`clock_tables`.
        quantity: The column to draw.
        title: The panel title.
        xlabel: The x axis label.
        ylabel: The y axis label.
        invert: Reverse the x axis, for a clock that counts down to delivery.
        populations: The populations to draw.
    """
    pooled = table[table["cohort"] == ALL_COHORT] if "cohort" in table.columns else table
    for key in populations:
        rows = pooled[pooled["population"] == key]
        if rows.empty:
            continue
        ax.plot(rows["bin_center"], rows[quantity], color=POPULATION_COLORS[key], linewidth=1.3,
                marker="o", markersize=2.3, label=key.replace("_", " "))
    if invert:
        ax.invert_xaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)


def _lag_field(
    table: pd.DataFrame, population: Optional[str], value: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a long lag-by-clock table into a ``(bins, lags)`` field on a uniform bin grid.

    The table holds only the bins that have anchors. An image draws its rows at equal steps, so a
    missing interior bin would shift every row after it. The field is therefore laid on the full
    grid from the first to the last bin, with ``NaN`` in a bin that has no anchors.

    Args:
        table: A lag table of :func:`clock_tables` or :func:`gate_tables`.
        population: The population to draw, or ``None`` for a table with no population column.
        value: The column to draw.

    Returns:
        ``(field, bin_centres, lag_seconds)``. All three are empty when the selection has no rows.
    """
    rows = table if population is None else table[table["population"] == population]
    if rows.empty:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    field = rows.pivot_table(index="bin_center", columns="lag_s", values=value, aggfunc="mean")
    centres = field.index.to_numpy(dtype=float)
    if centres.size > 1:
        step = float(np.min(np.diff(centres)))
        n_steps = int(round((centres[-1] - centres[0]) / step))
        grid = centres[0] + step * np.arange(n_steps + 1)
        full = np.full((grid.size, field.shape[1]), np.nan)
        full[np.rint((centres - centres[0]) / step).astype(int)] = field.to_numpy(dtype=float)
        return full, grid, field.columns.to_numpy(dtype=float)
    return field.to_numpy(dtype=float), centres, field.columns.to_numpy(dtype=float)


def draw_summary_figure(
    summary: pd.DataFrame,
    by_time: pd.DataFrame,
    by_contraction: pd.DataFrame,
    runs: pd.DataFrame,
    cohorts: Sequence[str],
    path: Path,
) -> Path:
    """Draw the six-panel population summary and return the path written.

    Args:
        summary: From :func:`summary_table`.
        by_time: The scalar table of :func:`clock_tables` on hours before delivery.
        by_contraction: The scalar table of :func:`clock_tables` on contraction age.
        runs: From :func:`run_duration_table`.
        cohorts: The cohort keys, pooled first.
        path: The figure stem.

    Returns:
        The file written.
    """
    fig, axes = figures.panel_grid(6, n_cols=2, panel_height=2.4)
    _bars_by_population(axes[0], summary, "frac", cohorts,
                        title="Share of anchors per population", ylabel="share of scored anchors")
    _bars_by_population(axes[1], summary, "kl_nats", cohorts,
                        title="$K_t$ per population", ylabel="nats per anchor")
    _bars_by_population(axes[2], summary, "gain_mc_nats", cohorts,
                        title="Forecast gain per population", ylabel="nats per anchor")
    if not by_time.empty:
        _clock_lines(axes[3], by_time, "gain_mc_nats", title="Forecast gain vs time to delivery",
                     xlabel="hours before delivery", ylabel="nats per anchor", invert=True)
    if not by_contraction.empty:
        _clock_lines(axes[4], by_contraction, "share_of_anchors",
                     title="Population share vs contraction age",
                     xlabel="s since contraction onset", ylabel="share of anchors",
                     populations=("gate_on", "fully_open", "gate_off"))
    ax = axes[5]
    for key in ("gate_on", "gate_off"):
        rows = runs[runs["state"] == key] if not runs.empty else runs
        if len(rows):
            ax.step(rows["duration_hi_s"], rows["share_of_time"], where="pre",
                    color=POPULATION_COLORS[key], linewidth=1.3, label=key.replace("_", " "))
    ax.set_title("Run duration, weighted by time")
    ax.set_xlabel("run duration (s)")
    ax.set_ylabel("share of the state's anchors")
    ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)
    return figures.render_figure(fig, path)


def draw_lag_figure(
    profile: pd.DataFrame,
    lag_by_time: pd.DataFrame,
    lag_by_contraction: pd.DataFrame,
    lag_by_gate: pd.DataFrame,
    cohorts: Sequence[str],
    path: Path,
) -> Path:
    """Draw the six-panel lag page and return the path written.

    Args:
        profile: From :func:`lag_profile_table`.
        lag_by_time: The lag table of :func:`clock_tables` on hours before delivery.
        lag_by_contraction: The lag table of :func:`clock_tables` on contraction age.
        lag_by_gate: The lag table of :func:`gate_tables`.
        cohorts: The cohort keys, pooled first.
        path: The figure stem.

    Returns:
        The file written.
    """
    # The heatmaps take the left column and the profiles the right one. A colourbar re-specs its
    # axes, and the shared label alignment then reads a right-column heatmap as a left-column
    # panel, so a heatmap on the right would print its y label over its neighbour.
    fig, axes = figures.panel_grid(6, n_cols=2, panel_height=2.5)
    pooled = profile[profile["cohort"] == ALL_COHORT] if not profile.empty else profile

    # (a), (c), (e) The gate-on lag share against the three clocks, on a logarithmic colour scale
    # so the lags beyond the first few stay visible beside lag 0.
    for ax, table, population, title, ylabel in (
        (axes[0], lag_by_contraction, "gate_on", "Gate-on KL share vs contraction age",
         "s since contraction onset"),
        (axes[2], lag_by_time, "gate_on", "Gate-on KL share vs time to delivery",
         "hours before delivery"),
        (axes[4], lag_by_gate, None, "KL share vs gate value", "$g$"),
    ):
        field, rows_axis, lag_axis = _lag_field(table, population, "kl_share")
        if field.size and np.isfinite(field).any():
            step = float(np.median(np.diff(rows_axis))) if rows_axis.size > 1 else 1.0
            lag_step = float(np.median(np.diff(lag_axis))) if lag_axis.size > 1 else 1.0
            extent = (
                lag_axis[0] - 0.5 * lag_step, lag_axis[-1] + 0.5 * lag_step,
                rows_axis[-1] + 0.5 * step, rows_axis[0] - 0.5 * step,
            )
            figures.heatmap_with_colorbar(
                fig, ax, np.clip(field, HEATMAP_SHARE_FLOOR, None), title=title,
                xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel=ylabel, symmetric=False,
                colorbar_label="share of $K_t$ (log scale)", extent=extent,
                norm=LogNorm(vmin=HEATMAP_SHARE_FLOOR, vmax=max(float(np.nanmax(field)), 2 * HEATMAP_SHARE_FLOOR)),
            )
        else:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no anchors on this clock", transform=ax.transAxes,
                    ha="center", va="center", color=figures.COLOR_GRAY)
            figures.style_axes(ax)

    # (b) The KL attribution per lag in nats, per population.
    ax = axes[1]
    for key in POPULATION_KEYS:
        rows = pooled[pooled["population"] == key] if len(pooled) else pooled
        if len(rows):
            ax.fill_between(rows["lag_s"], rows["kl_nats_lo"], rows["kl_nats_hi"],
                            color=POPULATION_COLORS[key], alpha=0.2, linewidth=0)
            ax.plot(rows["lag_s"], rows["kl_nats"], color=POPULATION_COLORS[key], linewidth=1.3,
                    label=key.replace("_", " "))
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_title("KL attribution per lag")
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("nats per anchor")
    ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)

    # (d) The same profiles as shares, with the gate-on attention beside them.
    ax = axes[3]
    for key in ("gate_on", "fully_open", "gate_off"):
        rows = pooled[pooled["population"] == key] if len(pooled) else pooled
        if len(rows):
            ax.plot(rows["lag_s"], rows["kl_share"], color=POPULATION_COLORS[key], linewidth=1.3,
                    label=f"KL share, {key.replace('_', ' ')}")
    rows = pooled[pooled["population"] == "gate_on"] if len(pooled) else pooled
    if len(rows):
        ax.plot(rows["lag_s"], rows["attention"], color=figures.COLOR_PURPLE, linewidth=1.1,
                linestyle="--", label="attention, gate on")
    ax.set_yscale("log")
    ax.set_ylim(HEATMAP_SHARE_FLOOR, 1.0)
    ax.set_title("Lag shares")
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("share (log scale)")
    ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)

    # (f) The gate-on KL share per clinical class.
    ax = axes[5]
    classes = [name for name in cohorts if name != ALL_COHORT]
    colours = figures.group_colors(classes)
    for name in classes:
        rows = profile[(profile["cohort"] == name) & (profile["population"] == "gate_on")]
        if len(rows):
            ax.plot(rows["lag_s"], rows["kl_share"], color=colours.get(name), linewidth=1.3,
                    label=f"{name} (n={int(rows['n_recordings'].iloc[0])})")
    ax.set_yscale("log")
    ax.set_ylim(HEATMAP_SHARE_FLOOR, 1.0)
    ax.set_title("Gate-on KL share per class")
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("share of $K_t$ (log scale)")
    if classes:
        ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)
    figures.caveat_note(fig)
    return figures.render_figure(fig, path)


def draw_trajectory_figure(
    trajectory: pd.DataFrame, tests: pd.DataFrame, cohorts: Sequence[str], path: Path
) -> Path:
    """Draw the gate-on class trajectories over labour and return the path written.

    Args:
        trajectory: The trajectory table of :func:`trajectory_tables`.
        tests: The test table of :func:`trajectory_tables`.
        cohorts: The cohort keys, pooled first.
        path: The figure stem.

    Returns:
        The file written.
    """
    panels = (
        ("kl_nats", "Gate-on $K_t$ per class", "nats per anchor"),
        ("gain_mc_nats", "Gate-on forecast gain per class", "nats per anchor"),
        ("share_of_anchors", "Gate-on share per class", "share of anchors"),
        ("kl_centroid_s", "Gate-on mean lag of the KL per class", "s, stored-coefficient time"),
    )
    fig, axes = figures.panel_grid(len(panels), n_cols=2, panel_height=2.5)
    classes = [name for name in cohorts if name != ALL_COHORT]
    colours = figures.group_colors(classes)
    for ax, (metric, title, ylabel) in zip(axes, panels):
        rows = trajectory[
            (trajectory["population"] == "gate_on") & (trajectory["metric"] == metric)
        ] if len(trajectory) else trajectory
        for name in classes:
            part = rows[rows["cohort"] == name].sort_values("bin_center_h") if len(rows) else rows
            if not len(part):
                continue
            ax.fill_between(part["bin_center_h"], part["q25"], part["q75"],
                            color=colours.get(name), alpha=0.18, linewidth=0)
            ax.plot(part["bin_center_h"], part["median"], color=colours.get(name), linewidth=1.3,
                    marker="o", markersize=2.2, label=name)
        if len(tests):
            marked = tests[
                (tests["population"] == "gate_on") & (tests["metric"] == metric) & tests["significant"]
            ]
            for centre in marked["bin_center_h"]:
                ax.axvline(float(centre), color=figures.COLOR_GRAY, linewidth=0.6, linestyle=":")
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_xlabel("hours before delivery")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, frameon=False)
        figures.style_axes(ax)
    figures.caveat_note(
        fig, "Median and quartiles over recordings. A dotted line marks a window with a class difference."
    )
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
        "n_samples": None, "composition": {},
        "plan": {"capped": False},
        "skipped": True, "reason": reason, "files": [],
    }


def run_gate_open_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the KL, the forecast gain and the lag structure per gate population.

    Args:
        context: The analysis context, read for the per-anchor table, the per-anchor vector
            sidecar, the per-sample table and the collection's lag block.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the population thresholds, the run-duration block, the headline
        block and the files written. A model with neither a gate nor a UA mask, and a run without
        the per-anchor lag maps, each record a skip.
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

    reduced = per_recording_reductions(data)
    per_guid = per_recording_table(data, reduced, labelled)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)
    summary = summary_table(data, reduced, resamples=resamples, seed=seed)
    summary.to_csv(directory / SUMMARY_FILENAME, index=False)
    profile = lag_profile_table(data, reduced, resamples=resamples, seed=seed)
    profile.to_csv(directory / LAG_PROFILE_FILENAME, index=False)

    frame = data.frame
    seconds_to_delivery = (
        frame["epoch"].to_numpy(dtype=float) + frame["anchor"].to_numpy(dtype=float) * SECONDS_PER_STEP
        if {"epoch", "anchor"} <= set(frame.columns) else np.full(len(frame), np.nan)
    )
    hours = np.clip(-seconds_to_delivery / 3600.0, 0.0, None)
    time_edges = np.arange(0.0, MAX_HOURS + TIME_BIN_HOURS, TIME_BIN_HOURS)
    by_time, lag_by_time = clock_tables(data, hours, time_edges)
    trajectory, trajectory_tests = trajectory_tables(data, hours, time_edges)
    age = (
        frame[CONTRACTION_AGE_COLUMN].to_numpy(dtype=float)
        if CONTRACTION_AGE_COLUMN in frame.columns else np.full(len(frame), np.nan)
    )
    by_contraction, lag_by_contraction = clock_tables(
        data, age, np.arange(0.0, MAX_CONTRACTION_AGE_S + CONTRACTION_BIN_S, CONTRACTION_BIN_S)
    )
    by_gate, lag_by_gate = gate_tables(data)
    durations, censored = run_durations(data)
    runs = run_duration_table(durations)
    distributions = distribution_table(data)
    for table, filename in (
        (by_time, BY_TIME_FILENAME), (lag_by_time, LAG_BY_TIME_FILENAME),
        (by_contraction, BY_CONTRACTION_FILENAME), (lag_by_contraction, LAG_BY_CONTRACTION_FILENAME),
        (by_gate, BY_GATE_FILENAME), (lag_by_gate, LAG_BY_GATE_FILENAME),
        (runs, RUNS_FILENAME), (distributions, DISTRIBUTIONS_FILENAME),
        (trajectory, TRAJECTORY_FILENAME), (trajectory_tests, TRAJECTORY_TESTS_FILENAME),
    ):
        table.to_csv(directory / filename, index=False)

    cohorts = data.cohorts()
    written = [
        draw_summary_figure(summary, by_time, by_contraction, runs, cohorts, directory / FIGURE_STEM),
        draw_lag_figure(
            profile, lag_by_time, lag_by_contraction, lag_by_gate, cohorts,
            directory / LAG_FIGURE_STEM,
        ),
        draw_trajectory_figure(
            trajectory, trajectory_tests, cohorts, directory / TRAJECTORY_FIGURE_STEM
        ),
    ]

    pooled = summary[summary["cohort"] == ALL_COHORT].set_index("population")

    def cell(population: str, quantity: str) -> float:
        """Read one pooled summary value.

        Args:
            population: A key of :data:`POPULATIONS`.
            quantity: A summary column.

        Returns:
            The value, ``NaN`` when the population is empty or the column is absent.
        """
        if population not in pooled.index or quantity not in pooled.columns:
            return float("nan")
        return float(pooled.loc[population, quantity])

    all_kl = cell("all", "kl_nats")
    headline = {
        "gate_on_frac": cell("gate_on", "frac"),
        "fully_open_frac": cell("fully_open", "frac"),
        "gate_on_kl_nats": cell("gate_on", "kl_nats"),
        "gate_off_kl_nats": cell("gate_off", "kl_nats"),
        "gate_on_gain_mc_nats": cell("gate_on", "gain_mc_nats"),
        "gate_off_gain_mc_nats": cell("gate_off", "gain_mc_nats"),
        "fully_open_gain_mc_nats": cell("fully_open", "gain_mc_nats"),
        "gate_on_kl_contribution_nats": cell("gate_on", "kl_contribution_nats"),
        "gate_on_gain_mc_contribution_nats": cell("gate_on", "gain_mc_contribution_nats"),
        "gate_off_gain_mc_contribution_nats": cell("gate_off", "gain_mc_contribution_nats"),
        "gate_on_kl_share_of_run": (
            cell("gate_on", "kl_contribution_nats") / all_kl if all_kl and np.isfinite(all_kl)
            else float("nan")
        ),
        "gate_on_kl_centroid_s": cell("gate_on", "kl_centroid_s"),
        "gate_on_kl_lag0_share": cell("gate_on", "kl_lag0_share"),
        "gate_on_median_run_s": (
            float(np.median(durations["gate_on"])) if durations["gate_on"].size else float("nan")
        ),
    }
    return {
        "n_samples": (
            int(frame.groupby(["guid", "epoch"]).ngroups)
            if {"guid", "epoch"} <= set(frame.columns) else None
        ),
        "composition": {"n_recordings": int(len(data.recordings))},
        "plan": {
            "capped": False, "bootstrap_resamples": resamples, "seed": seed,
            "gate_column": APPLIED_COLUMN if APPLIED_COLUMN in per_anchor.columns else OPEN_COLUMN,
        },
        "thresholds": {"gate_shut_below": GATE_SHUT_BELOW, "gate_open_from": GATE_OPEN_FROM},
        "populations": [
            {"key": key, "label": label, "n_anchors": int(data.masks[key].sum())}
            for key, label in POPULATIONS
        ],
        "runs": {
            key: {
                "n": int(values.size),
                "median_s": float(np.median(values)) if values.size else float("nan"),
                "q90_s": float(np.quantile(values, 0.9)) if values.size else float("nan"),
                "n_touching_segment_edge": int(censored[key]),
            }
            for key, values in durations.items()
        },
        "headline": headline,
        "trajectory": {
            "min_window_anchors": MIN_WINDOW_ANCHORS, "alpha": TRAJECTORY_ALPHA,
            "n_tests": int(len(trajectory_tests)),
            "n_significant": (
                int(trajectory_tests["significant"].sum()) if len(trajectory_tests) else 0
            ),
            "significant_by_population_and_metric": (
                {
                    f"{population}/{metric}": int(count)
                    for (population, metric), count in trajectory_tests[
                        trajectory_tests["significant"]
                    ].groupby(["population", "metric"]).size().items()
                } if len(trajectory_tests) else {}
            ),
        },
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            SUMMARY_FILENAME, PER_RECORDING_FILENAME, LAG_PROFILE_FILENAME, BY_TIME_FILENAME,
            LAG_BY_TIME_FILENAME, BY_CONTRACTION_FILENAME, LAG_BY_CONTRACTION_FILENAME,
            BY_GATE_FILENAME, LAG_BY_GATE_FILENAME, RUNS_FILENAME, DISTRIBUTIONS_FILENAME,
            TRAJECTORY_FILENAME, TRAJECTORY_TESTS_FILENAME,
            *(Path(str(path)).name for path in written),
        ],
    }


__all__ = [
    "ANALYSIS_DIRNAME",
    "POPULATIONS",
    "AnchorData",
    "applied_gate",
    "clock_tables",
    "distribution_table",
    "gate_tables",
    "group_sums",
    "lag_profile_table",
    "lag_summaries",
    "per_recording_reductions",
    "per_recording_table",
    "population_masks",
    "run_duration_table",
    "run_durations",
    "run_gate_open_analysis",
    "summary_table",
    "trajectory_tables",
]
