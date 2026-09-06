r"""Is the forecast right in the frequencies a fetal heart rate trace is read for, and for how long?

Every other analysis here scores the forecast in the time domain -- a block score, an RMSE in bpm,
a skill against three baselines, a rate of detected decelerations. None of them says *which*
frequencies the model reproduces, and that omission hides a distinction with clinical content: a
forecast that holds the baseline while flattening beat-to-beat variability and one that tracks
variability while drifting on baseline score identically on mean squared error, and they are
different findings about a fetus.

What this analysis measures, per frequency band and per **lead time**:

* **Coherence** $\gamma^2$ -- how much of the truth's variation at that frequency the forecast
  reproduces in phase.
* **Spectral gain** $g$ -- whether the forecast has the truth's amplitude there. $g < 1$ is the
  over-smoothing signature every mean-square-trained forecaster is prone to.
* **Cross-spectral phase** and a group delay -- whether it arrives at the right moment.
* **An exact three-way split of the residual spectrum** into what no rescaling could fix, what
  timing costs, and what amplitude costs. This is what turns "the coherence is $0.6$" into an
  actionable statement.
* **The source's contribution in frequency**, $\gamma^2_{\mathrm{full}} - \gamma^2_{\mathrm{base}}$
  -- the frequency-resolved counterpart of ``coupling``'s ``pred_gap``.
* **Whether the forecast reproduces the UP--FHR relationship** the truth carries.
* **A token-seam check**, because the decoder emits each $R$-sample sub-block from one linear map
  and a discontinuity at that period is architecturally plausible in $\hat\mu$ and impossible in
  the truth.

How it is possible at all
=========================

Spectral analysis of this forecast was deliberately deferred, for a correct reason: a single
forecast block is $H \cdot R$ samples at $4\,$Hz, and a Welch window that fits inside it puts
the whole $[0, 0.04)$ Hz deceleration span into the DC bin that detrending has already removed.

The **$\tau$-slice** answers it. Fixing a horizon step $\tau$ and concatenating over consecutive
anchors yields a contiguous, gap-free, non-overlapping $4\,$Hz series covering $960$ s -- see
:func:`~teb_vae.lag_attn_rws.eval.metrics.tau_slices`, whose identity against the model's own
target builder is asserted by test. That supports ``nperseg = 512`` ($\Delta f = 7.8125$ mHz, four
bins below $0.03$ Hz), and it makes lead time an *axis* rather than a trade: slice $\tau$ holds
lead times $[4\tau + 0.25,\ 4\tau + 4]$ s, and the thirty of them tile $0$--$120$ s at full
frequency resolution each.

Sums in, ratios out
===================

The collection pass stores **unnormalised** cross-spectral sums per segment; this module is the
only place a ratio is formed. Within a recording the statistics are therefore *summed*
(``groupby(...).sum()``), not averaged -- a deliberate, named departure from
:func:`~teb_vae.lag_attn_rws.eval.frames.per_recording_means`, which every other analysis uses.

Averaging per-segment coherences instead would be wrong rather than merely different. Coherence is
biased upward by $(1 - \gamma^2)/n_d$ at $n_d$ averaged windows, and one segment contributes at
most $14$ per $\tau$ -- up to $7$ percentage points. It is exactly $1.0$ on a single window, for any
two signals whatever. Summing first drives $n_d$ into the thousands, where the bias is $10^{-4}$.
Both estimators are emitted side by side (``..._segment_mean``) so the size of that bias is a
measured number on every run rather than an argument in this docstring.

Across recordings the chain resumes as usual: one value per recording, unweighted, with intervals
bootstrapped over recordings.

What it does not measure
========================

Nothing below $\Delta f = 7.8$ mHz is resolved, so a $0.003$ Hz VLF floor is unreachable at any
window length a $20$-minute segment supports. Every estimate pools windows, so a coherence that
*varies* within a recording reads as a lower constant one -- this is not a spectrogram. And the
per-window mean removal means the spectrum says nothing whatever about the forecast's level; that
is ``forecast``'s mean-error column, and the difference between the detrended and raw residual sums
is reported here so the size of what was removed is visible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval import spectra
from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_rws.eval.cohort import ordered_groups
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_labels,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "coherence"

#: What it writes.
BANDS_FILENAME = "coherence_bands.csv"
PER_RECORDING_FILENAME = "coherence_per_recording.csv"
SPECTRUM_FILENAME = "coherence_spectrum.csv"
COHORT_SPECTRUM_FILENAME = "coherence_cohort_spectrum.csv"
SOURCE_FILENAME = "coherence_source.csv"
SEAM_FILENAME = "coherence_seam.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them.
LEAD_TIME_FIGURE = "coherence_lead_time"
SPECTRUM_FIGURE = "coherence_spectrum"
BANDS_FIGURE = "coherence_bands"
DECOMPOSITION_FIGURE = "coherence_decomposition"
SOURCE_FIGURE = "coherence_source"
SEAM_FIGURE = "coherence_seam"

#: The two forecast branches, in the order every table and legend lists them.
BRANCHES: Tuple[str, ...] = ("base", "full")

#: Prefix the collection pass stores the cross-spectral sums under, restated here rather than
#: imported: this module is layer 2 and ``collect`` is layer 1, so the coupling is a string.
VECTOR_PREFIX = "coherence_"

#: Cohort key the whole split is pooled under in the spectra sidecar.
POOLED_KEY = "all"

#: The statistic names the spectra sidecar's keys end in, as ``<cohort>_<statistic>``.
#:
#: Listed rather than parsed. A cohort name is arbitrary -- it is a clinical class today and could
#: be a subgroup tomorrow, and subgroup names contain underscores -- so splitting a key on a
#: separator would work by accident until a cohort was named something ordinary like
#: ``healthy_bg_cs``. Stripping a *known* suffix cannot be fooled that way.
SPECTRAL_STATISTICS: Tuple[str, ...] = (
    "sxx",
    "syy_base",
    "syy_full",
    "sxy_base_re",
    "sxy_base_im",
    "sxy_full_re",
    "sxy_full_im",
    "suu",
    "sux_truth_re",
    "sux_truth_im",
    "sux_base_re",
    "sux_base_im",
    "sux_full_re",
    "sux_full_im",
)


def pooled_cohorts(pooled: Dict[str, np.ndarray]) -> List[str]:
    """Return the cohorts the spectra sidecar holds, pooled split first then the rest in order."""
    found = {
        name[: -(len(statistic) + 1)]
        for name in pooled
        for statistic in SPECTRAL_STATISTICS
        if name.endswith(f"_{statistic}")
    }
    rest = sorted(name for name in found if name != POOLED_KEY)
    return ([POOLED_KEY] if POOLED_KEY in found else []) + rest


def pooled_for(pooled: Dict[str, np.ndarray], cohort: str) -> Dict[str, np.ndarray]:
    """Return one cohort's statistics from the sidecar, keyed by statistic alone."""
    return {
        statistic: pooled[f"{cohort}_{statistic}"]
        for statistic in SPECTRAL_STATISTICS
        if f"{cohort}_{statistic}" in pooled
    }

#: The band whose numbers reach the headline. LF is where genuinely forecastable structure lives at
#: this horizon -- VLF is only four bins wide, MF carries the token-seam frequency, and HF is
#: mostly beyond what a horizon-long forecast can say anything about.
HEADLINE_BAND = "lf"

#: The metrics resolved by cohort. The three that answer different questions: how much of the truth
#: the forecast reproduces, whether it has the truth's amplitude, and what the source added.
GROUPED_METRICS: Tuple[str, ...] = (
    f"coherence_full_{HEADLINE_BAND}",
    f"gain_full_{HEADLINE_BAND}",
    f"delta_coherence_{HEADLINE_BAND}",
    f"residual_normalised_full_{HEADLINE_BAND}",
)

#: Which per-(band, lead) rows carry a bootstrap interval. Not every column: an interval is drawn
#: only where a figure shows one, and a bootstrap over recordings for all forty columns at thirty
#: lead times would dominate the analysis's runtime for numbers nothing reads.
INTERVAL_METRICS: Tuple[str, ...] = ("coherence", "gain", "delta_coherence")

#: Readouts that are **angles**, and must never be averaged linearly.
#:
#: A plain mean of $-3.1$ and $+3.1$ radians is $0$ -- the two are a tenth of a radian apart on the
#: circle and the mean puts them half a turn away. So the phase is never averaged across recordings
#: or across lead times anywhere in this module. Where a pooled phase is wanted it is taken as the
#: **argument of the summed cross-spectrum**, which is the correct pooled quantity and is what the
#: coherence is formed from anyway; where a pooled delay is wanted it comes from
#: :func:`~teb_vae.lag_attn_rws.eval.spectra.estimate_delay` over a band's bins.
CIRCULAR_METRICS: Tuple[str, ...] = ("phase_rad",)

#: Readouts that describe the estimator rather than the forecast, and are reported pooled rather
#: than per recording. ``decomposition_residual`` is the identity's own round-off; a per-recording
#: column of it would be forty numbers all reading $10^{-16}$.
DIAGNOSTIC_METRICS: Tuple[str, ...] = ("decomposition_residual",)

#: The sentence a run carries beside every source-coherence number, so the causality refusal this
#: pipeline makes everywhere is not quietly suspended by a figure that looks like a coupling
#: measurement.
SOURCE_COHERENCE_NOTE = (
    "source coherence is measured against the CONTEMPORANEOUS uterine pressure -- the pressure "
    "during the window being forecast, which the model never read, since it conditions on the "
    "source only up to each anchor. It is a linear-association and timing statement between two "
    "traces, not a directed or causal one, and the run's causality disclosure applies to it "
    "exactly as to every other readout here."
)


# =============================================================================
# Reading the stored sums
# =============================================================================
def stored_geometry(vectors: Dict[str, np.ndarray]) -> Optional[Tuple[int, int]]:
    r"""Recover $(H, B)$ from the stored vectors' own shapes.

    Read off the arrays rather than off the record, so a directory whose record and sidecar
    disagree fails on the reshape rather than silently relabelling lead times as bands.

    Args:
        vectors: The vectors sidecar.

    Returns:
        ``(horizon, n_bands)``, or ``None`` when the sidecar carries no cross-spectral sums.
    """
    counts = vectors.get(f"{VECTOR_PREFIX}n_windows")
    banded = vectors.get(f"{VECTOR_PREFIX}sxx")
    if counts is None or banded is None or counts.ndim != 2 or banded.ndim != 2:
        return None
    horizon = int(counts.shape[1])
    if horizon <= 0 or banded.shape[1] % horizon:
        return None
    return horizon, int(banded.shape[1] // horizon)


def per_recording_sums(
    vectors: Dict[str, np.ndarray], guids: Sequence[str], *, horizon: int, n_bands: int
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    r"""Sum every stored statistic within each recording.

    **Summed, not averaged.** See the module docstring: a mean of per-segment ratios reintroduces
    the coherence's small-sample bias, and a mean of per-segment *sums* is not a quantity anything
    downstream could ratio. Adding two segments' sums and ratioing once is exactly the ratio of
    their pooled windows, which is the property that makes this reduction correct.

    ``NaN`` rows -- segments that scored no anchor -- contribute zero, which is their true
    contribution: they held no valid window.

    Args:
        vectors: The vectors sidecar.
        guids: One recording identifier per row of ``per_sample``, in row order.
        horizon: $H$.
        n_bands: $B$.

    Returns:
        ``(recordings, sums)`` with the recordings sorted and each sum $(G, H, B)$ for a spectral
        statistic or $(G, H)$ for a count or a time-domain reference.
    """
    recordings = sorted({str(value) for value in guids})
    position = {name: index for index, name in enumerate(recordings)}
    rows = np.array([position[str(value)] for value in guids], dtype=np.int64)

    sums: Dict[str, np.ndarray] = {}
    for key, array in vectors.items():
        if not key.startswith(VECTOR_PREFIX) or array.ndim != 2 or array.shape[0] != rows.size:
            continue
        name = key[len(VECTOR_PREFIX) :]
        values = np.nan_to_num(np.asarray(array, dtype=np.float64), nan=0.0)
        if values.shape[1] == horizon * n_bands:
            values = spectra.reshape_band_horizon(values, horizon=horizon, n_bands=n_bands)
        elif values.shape[1] != horizon:
            continue
        total = np.zeros((len(recordings),) + values.shape[1:], dtype=np.float64)
        np.add.at(total, rows, values)
        sums[name] = total
    return recordings, sums


def _complex(sums: Dict[str, np.ndarray], stem: str) -> np.ndarray:
    """Rebuild one complex cross-spectrum from the real and imaginary halves the sidecar stores."""
    real = sums.get(f"{stem}_re")
    imaginary = sums.get(f"{stem}_im")
    if real is None or imaginary is None:
        return np.zeros(0)
    return np.asarray(real, dtype=np.float64) + 1j * np.asarray(imaginary, dtype=np.float64)


def branch_readouts(sums: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Derive the coherence readouts for both branches from summed statistics.

    Args:
        sums: Per-recording (or pooled) sums, each $(\ldots, H, B)$ or $(\ldots, F)$.

    Returns:
        Branch -> the readouts of :func:`~teb_vae.lag_attn_rws.eval.spectra.derive`, plus
        ``delta_msc`` and ``delta_residual_normalised`` on the ``full`` entry -- the
        frequency-resolved counterpart of ``coupling``'s ``pred_gap``.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    truth = sums.get("sxx")
    if truth is None:
        return out
    for branch in BRANCHES:
        power = sums.get(f"syy_{branch}")
        cross = _complex(sums, f"sxy_{branch}")
        if power is None or cross.size == 0:
            continue
        out[branch] = spectra.derive(truth, power, cross)
    if set(BRANCHES) <= set(out):
        out["full"]["delta_coherence"] = out["full"]["coherence"] - out["base"]["coherence"]
        out["full"]["delta_residual_normalised"] = (
            out["full"]["residual_normalised"] - out["base"]["residual_normalised"]
        )
    return out


def source_readouts(sums: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Derive the UP-versus-signal coherence for the truth and both branches.

    The comparison the three answer together: the truth's own UP--FHR coherence is what the
    relationship *is*, and each branch's is whether the forecast reproduces it. A source-conditioned
    branch that matches the truth where the target-only branch does not is frequency-resolved
    evidence that the source pathway carries something.

    Args:
        sums: Per-recording or pooled sums.

    Returns:
        Series name -> readouts, empty when the pass carried no raw UP.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    source_power = sums.get("suu")
    if source_power is None:
        return out
    for name, signal in (("truth", "sxx"), ("base", "syy_base"), ("full", "syy_full")):
        power = sums.get(signal)
        cross = _complex(sums, f"sux_{name}")
        if power is None or cross.size == 0:
            continue
        out[name] = spectra.derive(source_power, power, cross)
    return out


# =============================================================================
# Tables
# =============================================================================
def band_delays(
    pooled_sums: Dict[str, np.ndarray], *, frequencies: np.ndarray, band_names: Sequence[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""A group delay per (branch, band, lead time), from the pooled per-bin cross-spectra.

    A delay needs *several* bins to be identifiable -- it is the slope of the phase across
    frequency -- so it cannot come from the band-collapsed per-segment statistics, which hold one
    complex number per band. It is therefore taken from the pooled full-resolution maps, and is a
    property of the split rather than a per-recording quantity with an interval.

    Args:
        pooled_sums: The pooled statistics, each $(H, F)$.
        frequencies: The one-sided axis $(F,)$ in Hz.
        band_names: The bands, in table order.

    Returns:
        Branch -> ``{"delay": (H, B), "concentration": (H, B)}``. The delay is in seconds, positive
        where the forecast lags the truth. **The concentration travels with it and is not
        optional**: it is the aligned cross-spectral magnitude over its total, so it is near $1$
        when every bin in the band agrees on one delay and near $0$ when the phase is incoherent.
        An ``argmax`` over the search grid always returns *some* delay, so a delay reported without
        it cannot be told apart from phase noise. Empty when the pooled maps are absent.
    """
    truth = pooled_sums.get("sxx")
    if truth is None or frequencies.size < 2:
        return {}
    assigned = spectra.band_index(frequencies)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for branch in BRANCHES:
        cross = _complex(pooled_sums, f"sxy_{branch}")
        if cross.size == 0:
            continue
        shape = (cross.shape[0], len(band_names))
        delays = np.full(shape, np.nan, dtype=np.float64)
        concentration = np.full(shape, np.nan, dtype=np.float64)
        for position in range(len(band_names)):
            bins = assigned == position
            if int(bins.sum()) < 2:
                # One bin constrains a delay only modulo $1/f$, which is not a constraint.
                continue
            delays[:, position], concentration[:, position] = spectra.estimate_delay(
                cross[:, bins], frequencies[bins]
            )
        out[branch] = {"delay": delays, "concentration": concentration}
    return out


def band_rows(
    readouts: Dict[str, Dict[str, np.ndarray]],
    pooled_readouts: Dict[str, Dict[str, np.ndarray]],
    delays: Dict[str, Dict[str, np.ndarray]],
    *,
    layout_leads: np.ndarray,
    band_names: Sequence[str],
    bin_counts: Dict[str, int],
    resamples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    r"""One row per (branch, band, lead time), averaged across recordings.

    The per-recording ratio is the unit for everything that can be averaged, as everywhere else in
    this pipeline: a recording contributing thirty segments and one contributing two count the same,
    and the bootstrap has a per-recording quantity to resample.

    **The phase is the exception, and is not averaged at all.** It is an angle; the mean of
    $-3.1$ and $+3.1$ radians is $0$, half a turn from both. The reported ``phase_rad`` is the
    argument of the cross-spectrum summed over every recording -- the correct pooled quantity, and
    the one the pooled coherence is formed from -- so it is a whole-split number reported beside
    per-recording means rather than one of them. ``phase_n`` is absent for the same reason: there is
    no per-recording population behind it.

    Args:
        readouts: Per-recording readouts, each $(G, H, B)$.
        pooled_readouts: The same quantities from statistics summed over every recording, each
            $(H, B)$ -- the source of the phase.
        delays: Branch -> $(H, B)$ group delays.
        layout_leads: The lead-time axis $(H,)$ in seconds.
        band_names: The bands, in table order.
        bin_counts: How many frequency bins each band holds.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        The rows, in clinical reading order: branch, then band, then lead time.
    """
    rows: List[Dict[str, Any]] = []
    for branch in BRANCHES:
        block = readouts.get(branch)
        if not block:
            continue
        pooled_block = pooled_readouts.get(branch) or {}
        branch_delays = delays.get(branch)
        horizon = int(next(iter(block.values())).shape[1])
        for band_position, band in enumerate(band_names):
            for tau in range(horizon):
                row: Dict[str, Any] = {
                    "branch": branch,
                    "hrv_band": band,
                    "n_bins": int(bin_counts.get(band, 0)),
                    "tau": tau,
                    "lead_seconds": float(layout_leads[tau]) if tau < layout_leads.size else np.nan,
                }
                for metric, values in block.items():
                    if metric in CIRCULAR_METRICS:
                        continue
                    column = np.asarray(values)[:, tau, band_position]
                    row[metric] = float(np.nanmean(column)) if column.size else np.nan
                    row[f"{metric}_n"] = int(np.isfinite(column).sum())
                    if metric in INTERVAL_METRICS:
                        interval = shared_stats.bootstrap_ci(
                            column, resamples=resamples, seed=seed
                        )
                        row[f"{metric}_ci_lo"] = interval["lo"]
                        row[f"{metric}_ci_hi"] = interval["hi"]
                for metric in CIRCULAR_METRICS:
                    values = pooled_block.get(metric)
                    row[f"{metric}_pooled"] = (
                        float(np.asarray(values)[tau, band_position])
                        if values is not None
                        else np.nan
                    )
                # The delay and the concentration that says whether it means anything, together.
                row["group_delay_s"] = (
                    float(branch_delays["delay"][tau, band_position])
                    if branch_delays is not None
                    else np.nan
                )
                row["group_delay_concentration"] = (
                    float(branch_delays["concentration"][tau, band_position])
                    if branch_delays is not None
                    else np.nan
                )
                rows.append(row)
    return rows


def per_recording_frame(
    readouts: Dict[str, Dict[str, np.ndarray]],
    segment_means: Dict[str, Dict[str, np.ndarray]],
    recordings: Sequence[str],
    *,
    band_names: Sequence[str],
    labels: pd.DataFrame,
) -> pd.DataFrame:
    r"""One row per recording, pooled over lead time.

    Pooled as the **mean over $\tau$ of the per-$\tau$ ratio**, never as a ratio of $\tau$-summed
    spectra. The phase depends on lead time, so a coherent sum of $S_{xy}$ across $\tau$ cancels,
    and a forecast whose timing error grows with lead time would report as one with none at all.

    Args:
        readouts: Per-recording readouts from summed statistics, each $(G, H, B)$.
        segment_means: The same quantities computed as unweighted means of per-segment ratios --
            emitted beside them so the coherence's small-sample bias is measured on the run.
        recordings: The recording identifiers, in row order.
        band_names: The bands, in table order.
        labels: Cohort labels indexed by ``guid``.

    Returns:
        The frame, indexed by ``guid``, carrying the cohort columns every grouped variant
        resolves on.
    """
    frame = pd.DataFrame(index=pd.Index(list(recordings), name="guid"))
    for branch in BRANCHES:
        block = readouts.get(branch) or {}
        for metric, values in block.items():
            # The phase is an angle and is never averaged (see :data:`CIRCULAR_METRICS`); the
            # decomposition residual is the identity's own round-off and belongs in the record.
            if metric in CIRCULAR_METRICS or metric in DIAGNOSTIC_METRICS:
                continue
            pooled = spectra.mean_over_horizon(values)
            for position, band in enumerate(band_names):
                name = metric if metric.startswith("delta_") else f"{metric}_{branch}"
                frame[f"{name}_{band}"] = pooled[:, position]
        alternative = segment_means.get(branch) or {}
        for metric, values in alternative.items():
            name = metric if metric.startswith("delta_") else f"{metric}_{branch}"
            frame[f"{name}_{HEADLINE_BAND}_segment_mean"] = values[
                :, band_names.index(HEADLINE_BAND)
            ]
    for column in labels.columns:
        frame[column] = labels[column]
    return frame


def segment_mean_readouts(
    vectors: Dict[str, np.ndarray],
    guids: Sequence[str],
    recordings: Sequence[str],
    *,
    horizon: int,
    n_bands: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""The estimator this analysis deliberately does **not** use, computed so the gap is measured.

    Ratios each segment first, then averages over the recording's segments and over lead time. That
    is what ``frames.per_recording_means`` would produce, and it is biased upward by the coherence's
    $(1 - \gamma^2)/n_d$ at the $\le 14$ windows a single segment holds.

    Emitting both is cheaper than defending one: a run whose two columns agree says the bias is
    negligible on that population, and a run where they do not has said so in its own tables.

    Args:
        vectors: The vectors sidecar.
        guids: One recording identifier per row, in row order.
        recordings: The recordings, in output order.
        horizon: $H$.
        n_bands: $B$.

    Returns:
        Branch -> metric -> $(G, B)$, restricted to the metrics the comparison is drawn on.
    """
    per_segment: Dict[str, np.ndarray] = {}
    for key, array in vectors.items():
        if not key.startswith(VECTOR_PREFIX) or array.ndim != 2:
            continue
        values = np.asarray(array, dtype=np.float64)
        if values.shape[1] != horizon * n_bands:
            continue
        per_segment[key[len(VECTOR_PREFIX) :]] = spectra.reshape_band_horizon(
            values, horizon=horizon, n_bands=n_bands
        )
    if "sxx" not in per_segment:
        return {}

    position = {name: index for index, name in enumerate(recordings)}
    rows = np.array([position[str(value)] for value in guids], dtype=np.int64)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for branch in BRANCHES:
        power = per_segment.get(f"syy_{branch}")
        cross = _complex(per_segment, f"sxy_{branch}")
        if power is None or cross.size == 0:
            continue
        derived = spectra.derive(per_segment["sxx"], power, cross)
        out[branch] = {}
        for metric in ("coherence", "gain"):
            pooled = spectra.mean_over_horizon(derived[metric])  # (n_segments, B)
            totals = np.zeros((len(recordings), pooled.shape[1]), dtype=np.float64)
            counts = np.zeros_like(totals)
            finite = np.isfinite(pooled)
            np.add.at(totals, rows, np.where(finite, pooled, 0.0))
            np.add.at(counts, rows, finite.astype(np.float64))
            out[branch][metric] = spectra.ratio_of_sums(totals, counts)
    return out


def spectrum_frame(
    pooled: Dict[str, np.ndarray], *, frequencies: np.ndarray, leads: np.ndarray
) -> pd.DataFrame:
    r"""The pooled frequency $\times$ lead-time map, in long form.

    Args:
        pooled: The spectra sidecar, keyed ``<cohort>_<statistic>``.
        frequencies: The one-sided axis $(F,)$ in Hz.
        leads: The lead-time axis $(H,)$ in seconds.

    Returns:
        One row per (branch, lead time, bin). Empty with its columns present when the sidecar
        carries nothing.
    """
    columns = ["branch", "tau", "lead_seconds", "freq_hz", "coherence", "gain", "phase_rad"]
    readouts = branch_readouts(pooled_for(pooled, POOLED_KEY))
    if not readouts:
        return pd.DataFrame(columns=columns)

    blocks: List[pd.DataFrame] = []
    for branch, block in readouts.items():
        horizon, n_freq = block["coherence"].shape
        grid_tau, grid_freq = np.meshgrid(
            np.arange(horizon), np.arange(n_freq), indexing="ij"
        )
        blocks.append(
            pd.DataFrame(
                {
                    "branch": branch,
                    "tau": grid_tau.reshape(-1),
                    "lead_seconds": leads[grid_tau.reshape(-1)],
                    "freq_hz": frequencies[grid_freq.reshape(-1)],
                    "coherence": block["coherence"].reshape(-1),
                    "gain": block["gain"].reshape(-1),
                    "phase_rad": block["phase_rad"].reshape(-1),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)[columns]


def cohort_spectrum_frame(
    pooled: Dict[str, np.ndarray], *, frequencies: np.ndarray
) -> pd.DataFrame:
    r"""Each clinical class's own spectrum, pooled over lead time.

    The one place a cohort difference appears at full frequency resolution. It is **descriptive**:
    these are pooled over the class's recordings rather than per recording, so no interval and no
    test belongs here, and a cohort difference is adjudicated by ``cross_subgroup`` on the
    per-recording band columns exactly as everywhere else.

    Args:
        pooled: The spectra sidecar.
        frequencies: The one-sided axis $(F,)$ in Hz.

    Returns:
        One row per (cohort, branch, bin).
    """
    columns = ["cohort", "branch", "freq_hz", "coherence", "gain"]
    blocks: List[pd.DataFrame] = []
    for cohort in pooled_cohorts(pooled):
        for branch, block in branch_readouts(pooled_for(pooled, cohort)).items():
            # Coherence and gain only. The phase is an angle, and this frame is pooled over lead
            # time -- a linear mean of thirty phases is not a phase. The lead-resolved phase, which
            # is derived rather than averaged, is in `coherence_spectrum.csv`.
            blocks.append(
                pd.DataFrame(
                    {
                        "cohort": cohort,
                        "branch": branch,
                        "freq_hz": frequencies,
                        "coherence": np.nanmean(block["coherence"], axis=0),
                        "gain": np.nanmean(block["gain"], axis=0),
                    }
                )
            )
    if not blocks:
        return pd.DataFrame(columns=columns)
    return pd.concat(blocks, ignore_index=True)[columns]


def source_rows(
    readouts: Dict[str, Dict[str, np.ndarray]],
    *,
    leads: np.ndarray,
    band_names: Sequence[str],
) -> List[Dict[str, Any]]:
    r"""One row per (band, lead time): the UP coherence of the truth and of both branches.

    ``preservation`` is each branch's coherence over the truth's -- $1$ when the forecast carries
    the same UP relationship the record does, below it when the forecast has lost it. A ratio
    rather than a difference, because what matters is the *fraction* of an association reproduced
    and the truth's own level varies by band over an order of magnitude.
    """
    rows: List[Dict[str, Any]] = []
    truth = (readouts.get("truth") or {}).get("coherence")
    if truth is None:
        return rows
    horizon = int(truth.shape[1])
    for band_position, band in enumerate(band_names):
        for tau in range(horizon):
            row: Dict[str, Any] = {
                "hrv_band": band,
                "tau": tau,
                "lead_seconds": float(leads[tau]) if tau < leads.size else np.nan,
            }
            for series in ("truth", "base", "full"):
                block = readouts.get(series) or {}
                values = block.get("coherence")
                # Coherence only: the phase here is an angle, and averaging it across recordings
                # would put two nearly-aligned traces half a turn apart. The pooled UP-FHR phase
                # is in the pooled spectrum tables instead.
                row[f"coherence_{series}"] = (
                    float(np.nanmean(np.asarray(values)[:, tau, band_position]))
                    if values is not None
                    else np.nan
                )
            for series in ("base", "full"):
                row[f"preservation_{series}"] = float(
                    spectra.ratio_of_sums(
                        np.array([row[f"coherence_{series}"]]),
                        np.array([row["coherence_truth"]]),
                    )[0]
                )
            rows.append(row)
    return rows


def seam_rows(
    pooled: Dict[str, np.ndarray], *, frequencies: np.ndarray, raw_per_step: int, leads: np.ndarray
) -> List[Dict[str, Any]]:
    r"""The token-seam ratio, per series and per harmonic, pooled and per lead time.

    The truth's row is the control and is the reason the table is worth reading: the raw trace knows
    nothing about the decoder's token boundary, so anything the truth shows at $f_s/R$ is whatever
    the fetal heart rate happens to do there, and only the excess over it is an artifact.
    """
    rows: List[Dict[str, Any]] = []
    if not pooled:
        return rows
    if int(raw_per_step) <= 0 or frequencies.size < 2:
        # Without R there is no seam period to look for. No rows, rather than a divide by
        # zero -- an older run's record may carry no geometry at all.
        return rows
    n_freq = int(frequencies.size)
    nperseg = 2 * (n_freq - 1)
    bins = spectra.seam_bins(nperseg, int(raw_per_step))
    if bins.size == 0:
        return rows

    maps = pooled_for(pooled, POOLED_KEY)
    for series, key in (("truth", "sxx"), ("base", "syy_base"), ("full", "syy_full")):
        power = maps.get(key)
        if power is None:
            continue
        array = np.asarray(power, dtype=np.float64)
        overall = spectra.seam_ratio(array.sum(axis=0), bins)
        per_lead = spectra.seam_ratio(array, bins)
        for position, bin_index in enumerate(bins.tolist()):
            for tau in range(-1, int(array.shape[0])):
                rows.append(
                    {
                        "series": series,
                        "harmonic": position + 1,
                        "bin": int(bin_index),
                        "freq_hz": float(frequencies[bin_index]),
                        "tau": tau,
                        "lead_seconds": (
                            np.nan if tau < 0 else float(leads[tau]) if tau < leads.size else np.nan
                        ),
                        "seam_ratio": (
                            float(overall[position]) if tau < 0 else float(per_lead[tau, position])
                        ),
                    }
                )
    return rows


# =============================================================================
# Figures
# =============================================================================
def _empty(axis: Any) -> None:
    """Mark a panel that had nothing to draw, rather than leaving it blank."""
    axis.text(0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes)
    figures.style_axes(axis)


def _frequency_lead_map(
    figure: Any,
    axis: Any,
    grid: np.ndarray,
    frequencies: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    symmetric: bool,
    raw_per_step: int,
) -> None:
    r"""Draw one frequency $\times$ lead-time field on a **logarithmic** frequency axis.

    Deliberately not ``figures_seam.heatmap_with_colorbar``, which is used for every other heatmap
    in this pipeline. That helper places its rows with ``imshow`` and an ``extent``, which is a
    linear mapping, and setting a log scale on top of it would move the axis labels without moving
    the data -- every row would then be drawn at the wrong frequency.

    The distinction matters more here than anywhere else in the pipeline. The frequencies a fetal
    heart-rate trace is read for are crowded into the bottom of the range: VLF and LF together end
    at $0.15$ Hz, which is $7\%$ of a linear $0$--$2$ Hz axis. Drawn linearly, the two bands this
    analysis exists to report occupy a sliver at the bottom edge while the band above physiology
    takes half the panel.

    **The DC bin is not drawn**: a log axis has no room for $f = 0$. It is not lost -- it is part
    of the ``vlf`` band sum everywhere else -- but it is the Hann taper's residue of the removed
    mean rather than a frequency, and it has no place on a frequency axis.

    Args:
        figure: The figure, needed to attach the colourbar.
        axis: Target axes.
        grid: The field, $(F, H)$ -- frequency down, lead time across.
        frequencies: The one-sided axis $(F,)$ in Hz, ascending from DC.
        title: Panel title.
        colorbar_label: Label for the colourbar.
        symmetric: Centre the colour scale on zero, for a signed difference.
        raw_per_step: Raw samples per horizon token, for the lead-time edges.
    """
    spacing = float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else 1.0
    # Cell edges rather than centres: pcolormesh wants one more edge than cell in each direction.
    # Starting at bin 1 drops DC, which cannot sit on a log axis.
    frequency_edges = (np.arange(1, frequencies.size + 1) - 0.5) * spacing
    lead_edges = np.arange(grid.shape[1] + 1) * (raw_per_step / spectra.FS_RAW)

    finite = grid[1:][np.isfinite(grid[1:])]
    limit = float(np.abs(finite).max()) if finite.size else 1.0
    bounds = {"vmin": -limit, "vmax": limit} if symmetric else {}
    mesh = axis.pcolormesh(
        lead_edges,
        frequency_edges,
        grid[1:],
        cmap="coolwarm" if symmetric else "magma",
        shading="flat",
        **bounds,
    )
    axis.set_yscale("log")
    axis.set_title(title, fontsize=figures.FONT_NOTE)
    axis.set_xlabel("lead time (s)")
    axis.set_ylabel("frequency (Hz)")
    colorbar = figure.colorbar(mesh, ax=axis)
    colorbar.set_label(colorbar_label, fontsize=figures.FONT_SMALL)
    colorbar.ax.tick_params(labelsize=figures.FONT_TINY)
    figures.style_axes(axis, grid="none")


def build_lead_time_figure(frame: pd.DataFrame, *, raw_per_step: int) -> Any:
    r"""The frequency $\times$ lead-time map: coherence, and what the source added.

    The headline figure, and the one the whole $\tau$-slice construction exists to make drawable.
    Frequency runs up the $y$ axis on a log scale, lead time along $x$: a **column** is one
    $\tau$-slice's entire spectrum, a **row** is one frequency's decay as the forecast reaches
    further ahead.
    """
    figure, axes = figures.new_figure(2, height_per_row=3.2)
    if frame.empty or "branch" not in frame.columns:
        for row in range(2):
            _empty(axes[row, 0])
        return figure

    full = frame[frame["branch"] == "full"]
    base = frame[frame["branch"] == "base"]
    if full.empty:
        for row in range(2):
            _empty(axes[row, 0])
        return figure

    frequencies = np.sort(full["freq_hz"].unique())
    grid = full.pivot_table(index="freq_hz", columns="tau", values="coherence").to_numpy()
    _frequency_lead_map(
        figure, axes[0, 0], grid, frequencies,
        title="Forecast-truth coherence by frequency and lead time (source-conditioned)",
        colorbar_label="coherence",
        symmetric=False,
        raw_per_step=raw_per_step,
    )
    if base.empty:
        _empty(axes[1, 0])
        return figure
    delta = grid - base.pivot_table(
        index="freq_hz", columns="tau", values="coherence"
    ).to_numpy()
    _frequency_lead_map(
        figure, axes[1, 0], delta, frequencies,
        title="What the source added, per frequency and lead time (full minus base)",
        colorbar_label="coherence added",
        symmetric=True,
        raw_per_step=raw_per_step,
    )
    return figure


def build_spectrum_figure(
    frame: pd.DataFrame, cohorts: pd.DataFrame, *, seam_frequencies: Sequence[float]
) -> Any:
    r"""Coherence, gain and phase against frequency, at the nearest and furthest lead times.

    The gain panel is the over-smoothing diagnostic and is drawn against a reference at $1$: a
    forecast below it has less variance than the truth at that frequency, whatever its coherence.
    """
    figure, axes = figures.new_figure(4)
    if frame.empty:
        for row in range(4):
            _empty(axes[row, 0])
        return figure

    horizon = int(frame["tau"].max())
    chosen = [tau for tau in (0, horizon) if tau in set(frame["tau"].unique())]
    panels = (
        ("coherence", "coherence", None),
        ("gain", "spectral gain (forecast / truth amplitude)", 1.0),
        ("phase_rad", "cross-spectral phase (rad); negative = forecast lags", 0.0),
    )
    for row, (metric, ylabel, reference) in enumerate(panels):
        axis = axes[row, 0]
        curves, names = [], []
        for branch in BRANCHES:
            for tau in chosen:
                subset = frame[(frame["branch"] == branch) & (frame["tau"] == tau)]
                if subset.empty:
                    continue
                curves.append(subset.sort_values("freq_hz")[metric].to_numpy())
                names.append(f"{branch}, lead {float(subset['lead_seconds'].iloc[0]):.0f} s")
        if not curves:
            _empty(axis)
            continue
        frequencies = np.sort(frame["freq_hz"].unique())
        figures.multi_line_panel(
            axis, frequencies, np.vstack(curves), names,
            xlabel="frequency (Hz)", ylabel=ylabel,
        )
        axis.set_xscale("log")
        if reference is not None:
            axis.axhline(reference, color="0.4", linewidth=figures.LINE_HAIRLINE, linestyle="--")
        for value in seam_frequencies:
            axis.axvline(
                float(value), color="0.7", linewidth=figures.LINE_HAIRLINE, linestyle=":"
            )
        figures.style_axes(axis)

    axis = axes[3, 0]
    if cohorts.empty:
        _empty(axis)
        return figure
    groups = ordered_groups(
        [name for name in cohorts["cohort"].unique() if name != POOLED_KEY], "clinical_class"
    )
    palette = figures.group_colors(groups)
    drawn = False
    for group in groups:
        subset = cohorts[(cohorts["cohort"] == group) & (cohorts["branch"] == "full")]
        if subset.empty:
            continue
        ordered = subset.sort_values("freq_hz")
        axis.plot(
            ordered["freq_hz"], ordered["coherence"],
            color=palette.get(group), linewidth=figures.LINE_REGULAR, label=group,
        )
        drawn = True
    if not drawn:
        _empty(axis)
        return figure
    axis.set_xscale("log")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("coherence")
    axis.set_title("By clinical class, pooled over lead time (descriptive, not a test)")
    axis.legend(fontsize=figures.FONT_SMALL)
    figures.style_axes(axis)
    return figure


def build_bands_figure(frame: pd.DataFrame) -> Any:
    """Each band's coherence, gain and source contribution against lead time."""
    figure, axes = figures.new_figure(3)
    if frame.empty:
        for row in range(3):
            _empty(axes[row, 0])
        return figure

    bands = list(spectra.band_names())
    panels = (
        ("full", "coherence", "coherence"),
        ("full", "gain", "spectral gain"),
        ("full", "delta_coherence", "coherence added by the source"),
    )
    for row, (branch, metric, ylabel) in enumerate(panels):
        axis = axes[row, 0]
        subset = frame[frame["branch"] == branch]
        if subset.empty or metric not in subset.columns:
            _empty(axis)
            continue
        curves, names, leads = [], [], None
        for band in bands:
            band_rows_ = subset[subset["hrv_band"] == band].sort_values("tau")
            if band_rows_.empty:
                continue
            curves.append(band_rows_[metric].to_numpy())
            names.append(band)
            leads = band_rows_["lead_seconds"].to_numpy()
        if not curves or leads is None:
            _empty(axis)
            continue
        figures.multi_line_panel(
            axis, leads, np.vstack(curves), names,
            xlabel="lead time (s)", ylabel=ylabel,
        )
        if metric == "gain":
            axis.axhline(1.0, color="0.4", linewidth=figures.LINE_HAIRLINE, linestyle="--")
        if metric == "delta_coherence":
            axis.axhline(0.0, color="0.4", linewidth=figures.LINE_HAIRLINE, linestyle="--")
        figures.style_axes(axis)
    return figure


def build_decomposition_figure(bands: pd.DataFrame, spectrum: pd.DataFrame) -> Any:
    r"""Where the forecast's error goes: unpredictable, mistimed, or mis-scaled.

    The three terms sum exactly to the normalised residual spectrum, so the panels are a budget
    rather than three separate diagnostics -- and which term dominates says what would have to
    change.
    """
    figure, axes = figures.new_figure(2)
    terms = ("irreducible", "timing", "amplitude")
    if spectrum.empty and bands.empty:
        for row in range(2):
            _empty(axes[row, 0])
        return figure

    axis = axes[0, 0]
    subset = (
        spectrum[(spectrum["branch"] == "full") & (spectrum["tau"] == 0)]
        if not spectrum.empty
        else spectrum
    )
    if subset.empty or not set(terms) <= set(subset.columns):
        _empty(axis)
    else:
        ordered = subset.sort_values("freq_hz")
        figures.multi_line_panel(
            axis,
            ordered["freq_hz"].to_numpy(),
            np.vstack([ordered[name].to_numpy() for name in terms]),
            list(terms),
            title="Normalised residual spectrum at the nearest lead time",
            xlabel="frequency (Hz)",
            ylabel="share of the truth's power",
        )
        axis.set_xscale("log")
        figures.style_axes(axis)

    axis = axes[1, 0]
    subset = (
        bands[(bands["branch"] == "full") & (bands["hrv_band"] == HEADLINE_BAND)]
        if not bands.empty
        else bands
    )
    if subset.empty or not set(terms) <= set(subset.columns):
        _empty(axis)
        return figure
    ordered = subset.sort_values("tau")
    figures.multi_line_panel(
        axis,
        ordered["lead_seconds"].to_numpy(),
        np.vstack([ordered[name].to_numpy() for name in terms]),
        list(terms),
        title=f"The same budget against lead time, {HEADLINE_BAND} band",
        xlabel="lead time (s)",
        ylabel="share of the truth's power",
    )
    figures.style_axes(axis)
    return figure


def build_source_figure(frame: pd.DataFrame) -> Any:
    """Whether the forecast reproduces the UP-FHR relationship the record carries."""
    figure, axes = figures.new_figure(2)
    if frame.empty:
        for row in range(2):
            _empty(axes[row, 0])
        return figure

    axis = axes[0, 0]
    bands = [band for band in spectra.band_names() if band in set(frame["hrv_band"])]
    curves, names = [], []
    leads = None
    for series in ("truth", "base", "full"):
        column = f"coherence_{series}"
        if column not in frame.columns:
            continue
        subset = frame[frame["hrv_band"] == HEADLINE_BAND].sort_values("tau")
        if subset.empty:
            continue
        curves.append(subset[column].to_numpy())
        names.append(series)
        leads = subset["lead_seconds"].to_numpy()
    if curves and leads is not None:
        figures.multi_line_panel(
            axis, leads, np.vstack(curves), names,
            title=f"UP coherence in the {HEADLINE_BAND} band: the record, and each forecast",
            xlabel="lead time (s)", ylabel="coherence with uterine pressure",
        )
        figures.style_axes(axis)
    else:
        _empty(axis)

    axis = axes[1, 0]
    pooled = frame.groupby("hrv_band", sort=False)[
        [name for name in ("preservation_base", "preservation_full") if name in frame.columns]
    ].mean()
    if pooled.empty:
        _empty(axis)
        return figure
    ordered = pooled.reindex([band for band in bands if band in pooled.index])
    figures.multi_line_panel(
        axis,
        np.arange(len(ordered), dtype=np.float64),
        np.vstack([ordered[column].to_numpy() for column in ordered.columns]),
        list(ordered.columns),
        title="Fraction of the record's UP coherence each forecast reproduces",
        xlabel="",
        ylabel="branch coherence / truth coherence",
    )
    axis.set_xticks(np.arange(len(ordered)))
    axis.set_xticklabels(list(ordered.index))
    axis.axhline(1.0, color="0.4", linewidth=figures.LINE_HAIRLINE, linestyle="--")
    figures.style_axes(axis)
    return figure


def build_seam_figure(frame: pd.DataFrame) -> Any:
    """Power at the decoder's token-seam frequency, against its own neighbourhood."""
    figure, axes = figures.new_figure(2)
    if frame.empty:
        for row in range(2):
            _empty(axes[row, 0])
        return figure

    axis = axes[0, 0]
    pooled = frame[frame["tau"] < 0]
    curves, names = [], []
    harmonics = np.sort(pooled["harmonic"].unique()) if not pooled.empty else np.array([])
    for series in ("truth", "base", "full"):
        subset = pooled[pooled["series"] == series].sort_values("harmonic")
        if subset.empty:
            continue
        curves.append(subset["seam_ratio"].to_numpy())
        names.append(series)
    if curves:
        figures.multi_line_panel(
            axis, harmonics.astype(np.float64), np.vstack(curves), names,
            title="Seam power against neighbouring bins (truth is the control)",
            xlabel="harmonic of the token-seam frequency",
            ylabel="power / median of neighbours",
        )
        axis.axhline(1.0, color="0.4", linewidth=figures.LINE_HAIRLINE, linestyle="--")
        figures.style_axes(axis)
    else:
        _empty(axis)

    axis = axes[1, 0]
    fundamental = frame[(frame["harmonic"] == 1) & (frame["tau"] >= 0)]
    curves, names, leads = [], [], None
    for series in ("truth", "base", "full"):
        subset = fundamental[fundamental["series"] == series].sort_values("tau")
        if subset.empty:
            continue
        curves.append(subset["seam_ratio"].to_numpy())
        names.append(series)
        leads = subset["lead_seconds"].to_numpy()
    if curves and leads is not None:
        figures.multi_line_panel(
            axis, leads, np.vstack(curves), names,
            title="The fundamental against lead time",
            xlabel="lead time (s)", ylabel="power / median of neighbours",
        )
        axis.axhline(1.0, color="0.4", linewidth=figures.LINE_HAIRLINE, linestyle="--")
        figures.style_axes(axis)
    else:
        _empty(axis)
    return figure


# =============================================================================
# The analysis
# =============================================================================
def skip_record(reason: str) -> Dict[str, Any]:
    """Return the recorded skip for a run whose tables carry no cross-spectral sums.

    Args:
        reason: What is missing and what would fix it.

    Returns:
        The protocol's keys with ``n_samples`` set to ``None`` -- this analysis scored no
        population, and a zero there would enter the coverage block as a disagreement with every
        analysis that did.
    """
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False},
        "skipped": True,
        "reason": reason,
        "files": [],
    }


#: Column headers each table falls back to when it has no rows.
#:
#: A ``DataFrame`` built from an empty list of dicts has no columns, and ``to_csv`` then writes a
#: bare newline that ``pd.read_csv`` refuses with ``EmptyDataError``. That is the same unreadable
#: state :data:`SKIP_WRITES_TABLES` documents the skip path as existing to avoid -- and the success
#: path can reach it too, whenever a table's rows all fall away (no seam bins at a short
#: ``nperseg``, no source statistics when the pass carried no raw UP). A header-only CSV is a
#: readable empty table; a zero-byte one is a crash in whoever opens it next.
EMPTY_TABLE_COLUMNS: Dict[str, Tuple[str, ...]] = {
    BANDS_FILENAME: (
        "branch", "hrv_band", "n_bins", "tau", "lead_seconds",
        "coherence", "gain", "phase_rad_pooled", "group_delay_s",
    ),
    SOURCE_FILENAME: (
        "hrv_band", "tau", "lead_seconds",
        "coherence_truth", "coherence_base", "coherence_full",
        "preservation_base", "preservation_full",
    ),
    SEAM_FILENAME: ("series", "harmonic", "bin", "freq_hz", "tau", "lead_seconds", "seam_ratio"),
}


def write_table(frame: pd.DataFrame, directory: Path, filename: str) -> None:
    """Write one table, giving an empty one its headers so it stays readable.

    Args:
        frame: The table.
        directory: This analysis's subdirectory.
        filename: The file to write.
    """
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=list(EMPTY_TABLE_COLUMNS.get(filename, ())))
    frame.to_csv(directory / filename, index=False)


#: What a skipped run leaves behind, and what it deliberately does not.
#:
#: The six figures **are** rendered, because the committed figure manifest binds every emitted PDF
#: by name and a run that silently emitted none would fail that binding instead of reporting the
#: skip. The CSVs are **not** written, and that is a correctness choice rather than tidiness: an
#: empty file is not a readable table. ``pd.DataFrame().to_csv`` writes a bare newline, and
#: ``pd.read_csv`` raises ``EmptyDataError`` on it -- so a placeholder here would take down
#: ``cross_subgroup``, which reads ``coherence_per_recording.csv`` off disk and handles an
#: *absent* source gracefully by recording it. Absent is the state its guard is written for.
SKIP_WRITES_TABLES = False


def run_coherence_analysis(
    context: Any, *, eval_config: Dict[str, Any], output_dir: Any, probe: Any = None
) -> Dict[str, Any]:
    """Score the forecast in the frequency domain, resolved by lead time.

    Args:
        context: The analysis context, read for the stored cross-spectral sums and the pooled maps.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the band table, the headline block, the reconciliation and the
        seam accounting -- or the skip record, on a run whose tables predate this analysis or whose
        geometry cannot hold one Welch window.
    """
    collection = context.collection
    record = dict(getattr(collection, "record", None) or {})
    vectors = dict(getattr(collection, "vectors", None) or {})
    pooled = dict(getattr(collection, "spectra", None) or {})
    per_sample = collection.per_sample

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    filenames = [
        BANDS_FILENAME, PER_RECORDING_FILENAME, SPECTRUM_FILENAME,
        COHORT_SPECTRUM_FILENAME, SOURCE_FILENAME, SEAM_FILENAME,
    ]
    figure_builders = [
        (LEAD_TIME_FIGURE, lambda: build_lead_time_figure(pd.DataFrame(), raw_per_step=1)),
        (SPECTRUM_FIGURE, lambda: build_spectrum_figure(
            pd.DataFrame(), pd.DataFrame(), seam_frequencies=())),
        (BANDS_FIGURE, lambda: build_bands_figure(pd.DataFrame())),
        (DECOMPOSITION_FIGURE, lambda: build_decomposition_figure(
            pd.DataFrame(), pd.DataFrame())),
        (SOURCE_FIGURE, lambda: build_source_figure(pd.DataFrame())),
        (SEAM_FIGURE, lambda: build_seam_figure(pd.DataFrame())),
    ]

    geometry = stored_geometry(vectors)
    if geometry is None or per_sample.empty or "guid" not in per_sample.columns:
        # Figures yes, tables no -- see :data:`SKIP_WRITES_TABLES` for why an empty CSV here would
        # be worse than no CSV at all.
        for name, builder in figure_builders:
            figures.render_figure(builder(), directory / name)
        return {
            **skip_record(
                "the tables carry no cross-spectral sums. They were collected before the "
                "coherence analysis existed, or on a geometry whose trained-anchor span cannot "
                "hold one Welch window; re-collect with a checkpoint to produce them."
            ),
            "files": [name for name, _ in figure_builders],
        }

    horizon, n_bands = geometry
    band_names = list(spectra.band_names())
    coherence_record = dict(record.get("coherence") or {})

    # The layout the pass actually ran under, rebuilt from its own dump rather than re-derived from
    # the shipped constants -- so a run collected at a different window length is still readable,
    # and its frequencies and lead times are its own. The fallbacks below cover a record that
    # predates those fields: the frequency axis is still recoverable from the stored bin count, and
    # the lead axis from the raw-sample geometry, which is the only thing they are needed for.
    layout = spectra.layout_from_record(coherence_record)
    if layout is not None:
        nperseg, raw_per_step = layout.nperseg, layout.raw_per_step
        leads = layout.lead_center_seconds()
    else:
        sample = pooled.get(f"{POOLED_KEY}_sxx")
        nperseg = 2 * (int(np.asarray(sample).shape[-1]) - 1) if sample is not None else 0
        raw_per_step = int((record.get("geometry") or {}).get("raw_per_step") or 0)
        leads = (
            np.array(
                [
                    (raw_per_step * (2 * tau + 1) + 1 + raw_per_step) / (2.0 * spectra.FS_RAW)
                    for tau in range(horizon)
                ],
                dtype=np.float64,
            )
            if raw_per_step
            else np.full(horizon, np.nan, dtype=np.float64)
        )
    frequencies = spectra.frequency_axis(nperseg) if nperseg else np.zeros(0)
    bin_counts = (
        spectra.band_bin_counts(frequencies)
        if frequencies.size
        else {name: 0 for name in band_names}
    )

    guids = [str(value) for value in per_sample["guid"].tolist()]
    recordings, sums = per_recording_sums(vectors, guids, horizon=horizon, n_bands=n_bands)
    readouts = branch_readouts(sums)
    source = source_readouts(sums)
    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))

    # Summed over every recording, then ratioed once: the pooled phase and the pooled decomposition
    # come from here rather than from an average of per-recording values, because a phase is an
    # angle and the two are not the same quantity.
    pooled_sums = {name: values.sum(axis=0) for name, values in sums.items()}
    pooled_readouts = branch_readouts(pooled_sums)
    pooled_maps = pooled_for(pooled, POOLED_KEY)
    rows = band_rows(
        readouts,
        pooled_readouts,
        band_delays(pooled_maps, frequencies=frequencies, band_names=band_names),
        layout_leads=leads,
        band_names=band_names,
        bin_counts=bin_counts,
        resamples=resamples,
        seed=seed,
    )
    bands = pd.DataFrame(rows)
    write_table(bands, directory, BANDS_FILENAME)

    alternative = segment_mean_readouts(
        vectors, guids, recordings, horizon=horizon, n_bands=n_bands
    )
    per_guid = per_recording_frame(
        readouts, alternative, recordings,
        band_names=band_names, labels=per_recording_labels(per_sample),
    )
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    spectrum = spectrum_frame(pooled, frequencies=frequencies, leads=leads)
    write_table(spectrum, directory, SPECTRUM_FILENAME)
    cohorts = cohort_spectrum_frame(pooled, frequencies=frequencies)
    write_table(cohorts, directory, COHORT_SPECTRUM_FILENAME)
    source_table = pd.DataFrame(source_rows(source, leads=leads, band_names=band_names))
    write_table(source_table, directory, SOURCE_FILENAME)
    seams = pd.DataFrame(
        seam_rows(pooled, frequencies=frequencies, raw_per_step=raw_per_step, leads=leads)
    )
    write_table(seams, directory, SEAM_FILENAME)

    seam_frequencies = [
        float(frequencies[index])
        for index in (
            spectra.seam_bins(nperseg, raw_per_step)
            if nperseg and raw_per_step
            else np.zeros(0, dtype=np.int64)
        )
    ]
    # The residual-spectrum terms are needed on the pooled per-bin frame for the decomposition
    # figure; they are derived rather than stored, so they are attached here.
    if not spectrum.empty:
        for branch, block in branch_readouts(pooled_maps).items():
            mask = spectrum["branch"] == branch
            for term in ("irreducible", "timing", "amplitude", "residual_normalised"):
                spectrum.loc[mask, term] = block[term].reshape(-1)

    written = []
    for name, builder in (
        (LEAD_TIME_FIGURE, lambda: build_lead_time_figure(
            spectrum, raw_per_step=raw_per_step or 1)),
        (SPECTRUM_FIGURE, lambda: build_spectrum_figure(
            spectrum, cohorts, seam_frequencies=seam_frequencies)),
        (BANDS_FIGURE, lambda: build_bands_figure(bands)),
        (DECOMPOSITION_FIGURE, lambda: build_decomposition_figure(bands, spectrum)),
        (SOURCE_FIGURE, lambda: build_source_figure(source_table)),
        (SEAM_FIGURE, lambda: build_seam_figure(seams)),
    ):
        written.append(str(figures.render_figure(builder(), directory / name).name))

    # Segments, not recordings: the coverage block compares this against every other analysis's
    # count, and a recording count here would read as a population disagreement with all of them.
    # A segment counts when it held at least one Welch window at any lead time -- which is
    # *stricter* than the per-step mask the block scores use, so this can legitimately fall below
    # `forecast`'s count. That is the whole-window drop rule, and `EVAL.md` says so.
    window_counts = vectors.get(f"{VECTOR_PREFIX}n_windows")
    scored_segments = (
        int((np.nansum(np.asarray(window_counts, dtype=np.float64), axis=1) > 0.0).sum())
        if window_counts is not None
        else None
    )
    return {
        "n_samples": scored_segments,
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "estimator": coherence_record,
        "headline": headline_block(per_guid),
        "bands": [row for row in rows if row.get("hrv_band") == HEADLINE_BAND],
        "reconciliation": reconciliation(sums),
        "seam": {
            "note": (
                "a ratio far above one on a branch, with the truth's near one, is power at the "
                "decoder's token boundary rather than in the fetal heart rate; it sits inside the "
                f"{spectra.band_names()[2]!r} band and contaminates it"
            ),
            "rows": [row for row in seams.to_dict("records") if row.get("tau", 0) < 0],
        },
        "source_coherence_note": SOURCE_COHERENCE_NOTE,
        "segment_mean_note": (
            "every ..._segment_mean column is the estimator this analysis does NOT use -- "
            "per-segment ratios averaged -- emitted so the coherence's small-sample bias is a "
            "measured number rather than an assumed one"
        ),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [*filenames, *written],
    }


def headline_block(per_guid: pd.DataFrame) -> Dict[str, Any]:
    """The three scalars the acceptance gate and the arm tables read, with their honest $n$.

    Args:
        per_guid: The per-recording frame.

    Returns:
        The mean across recordings of each headline metric, and the count behind it.
    """
    block: Dict[str, Any] = {}
    for column in GROUPED_METRICS[:3]:
        values = finite_column(per_guid, column)
        summary = describe(values)
        block[column] = summary["mean"]
        block[f"{column}_n"] = summary["n"]
    return block


def reconciliation(sums: Dict[str, np.ndarray]) -> Dict[str, Any]:
    r"""Check the spectral residual against the time domain, and report what detrending removed.

    Two numbers with different jobs:

    * ``parseval_max_relative_error`` -- the FFT side against the time-domain side, both
      accumulated in the collection pass over the identical kept windows. Exact in real arithmetic
      by the DFT's Parseval theorem, so anything above round-off is a normalisation error: a wrong
      $N U$ divisor, a missing or duplicated one-sided doubling, or a band table that does not
      cover every bin.
    * ``detrended_share_of_raw`` -- how much of the raw residual power survives the per-window mean
      removal. The complement is the forecast's *level* error, which the spectrum by construction
      cannot see; reporting it makes the size of that blind spot a number rather than a caveat.

    Args:
        sums: Per-recording summed statistics.

    Returns:
        The reconciliation, with ``NaN`` where a term is absent.
    """
    out: Dict[str, Any] = {}
    truth = sums.get("sxx")
    if truth is None:
        return out
    # Seeded at NaN, not at 0.0, and raised only by a comparison that actually happened.
    #
    # A worst-case seeded at zero is the failure mode this whole identity exists to prevent. If the
    # stored sums carry no time-domain reference, or every window was dropped so that every
    # relative error is NaN, a zero seed reaches ``check_coherence_parseval`` as a finite value
    # below the tolerance and the check reports PASS -- "the residual spectrum matches the time
    # domain to 0" -- having compared nothing at all. The one exact correctness gate on the
    # estimator would then certify itself on precisely the runs where it measured nothing.
    # ``NaN`` routes that state to INCONCLUSIVE instead, which is what it is.
    worst = float("nan")
    compared = False
    for branch in BRANCHES:
        power = sums.get(f"syy_{branch}")
        cross = sums.get(f"sxy_{branch}_re")
        reference = sums.get(f"ss_detrended_{branch}")
        if power is None or cross is None or reference is None:
            continue
        spectral = (truth + power - 2.0 * cross).sum(axis=-1)
        relative = spectra.ratio_of_sums(np.abs(spectral - reference), np.abs(reference))
        finite = relative[np.isfinite(relative)]
        if finite.size:
            worst = float(finite.max()) if not compared else max(worst, float(finite.max()))
            compared = True
        raw = sums.get(f"ss_raw_{branch}")
        if raw is not None:
            share = spectra.ratio_of_sums(reference.sum(), raw.sum())
            out[f"detrended_share_of_raw_{branch}"] = float(share)
    out["parseval_max_relative_error"] = worst
    out["parseval_compared"] = bool(compared)
    return out
