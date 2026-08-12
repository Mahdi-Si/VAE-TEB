r"""Measure what a one-sided (causal) transform costs, against the production two-sided one.

.. code-block:: bash

    python -m hdf5_dataset.compare_causal_scattering                     # full comparison
    python -m hdf5_dataset.compare_causal_scattering --mode self-test    # filters only, no shard

or straight from an IDE's Run button, by editing :data:`RUN_ARGS` at the bottom of this file.

What this answers
-----------------
Not *"is the causal transform causal"* -- it is, by construction, and a figure of a flat zero line
would say nothing. The question is **what causality costs**, channel by channel, and the answer is
an exchange rate: a causal filter removes a forward leak $L_{95}$ by paying a backward group delay
$\tau_g$, and $\tau_g \approx 1.5\,L_{95}$. So against
:func:`~teb_vae.lag_attn.channel_reach.resolve_channel_budget` at the shipped $120$ s budget the
causal transform is *strictly staler* on every channel that budget already keeps, and buys back
only the channels the budget has to drop. Both arms are pushed through the **same**
``resolve_channel_budget`` so the comparison uses one predicate rather than two.

Three arms, so implementation error is separable from the effect under study
----------------------------------------------------------------------------
* **A** -- the shard itself, written by ``KymatioPhaseScattering1D`` or, on a causal shard, by
  :mod:`hdf5_dataset.causal_scattering_torch`.
* **B** -- :mod:`hdf5_dataset.causal_scattering` on the production Morlet bank.
* **C** -- the same code on :func:`~hdf5_dataset.causal_scattering.build_causal_bank`.

A-vs-B is the correctness gate and is reported in ``summary.json`` before anything else; B-vs-C is
the comparison. Arm B is run twice, in production's own decimation and smoothing conventions
(``'kymatio'`` / ``'kymatio_truncate'``) to validate against A, and in the conventions arm C
shares (``'full_rate'`` / ``'exact'``) to compare against C -- so neither number is doing two jobs.

Either variant of shard may be read
-----------------------------------
A shard says which transform wrote it through its root ``transform`` attribute (absent = legacy
two-sided). **Which arm the gate validates against follows from that**, because arm A *is* the
shard: a two-sided shard is checked against arm B, a causal one against arm C reduced by the stored
channel plan -- which additionally checks that the build kept the channels the plan says it keeps,
on production-scale data rather than on fixture segments. Everything after the gate -- the filter
measurements, the leak test, the delay estimates, the budgets and ``per_channel.csv`` -- describes
the two *banks* and is computed from the raw ``fhr``/``up`` rows alone, so it is the same study
whichever variant supplied those rows.

Outputs
-------
Everything lands under ``--output-dir`` (default ``output/causal_scattering``): eight figures, a
``per_channel.csv`` with one row per (block, channel), a ``summary.json`` carrying every scalar and
the provenance of every argument, and a generated ``REPORT.md``.

lean-limit: the delay and correlation estimates pool a handful of segments from one committed
shard; replace with a cohort-wide estimate if a decision rests on the tails rather than the medians.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not __package__ and _REPO_ROOT not in sys.path:
    # Launched as a script the file's own directory goes on sys.path instead of the repository
    # root, and every absolute import below then fails.
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np

from hdf5_dataset.causal_scattering import (
    CAUSAL_KERNEL_TAPS,
    GAMMATONE_ORDER,
    CausalBank,
    CausalChannelPlan,
    assert_matches_shard,
    build_causal_bank,
    build_channel_plan,
    build_truncated_morlet_bank,
    causal_support_samples,
    embed_on_two_sided_axis,
    future_energy_fraction,
    half_power_half_width,
    phase_block_causal,
    phase_block_two_sided,
    response_summary,
    scattering_block_causal,
    scattering_block_two_sided,
    selected_pairs,
    two_sided_taps,
)
from hdf5_dataset.hdf5_dataset import CAUSAL, TWO_SIDED, resolve_transform
from teb_vae.lag_attn.channel_reach import (
    SOURCE_PHASE_BAND_HZ,
    TARGET_PHASE_BAND_HZ,
    block_reach_seconds,
    resolve_channel_budget,
)
from teb_vae.lag_attn_rws.eval.launch import resolve_launch_args
from teb_vae.lag_attn.eval.representation_capacity_probe import (
    DECIMATION,
    FS,
    HORIZON_S,
    build_filter_bank,
    deceleration,
    forward_reach,
)

logger = logging.getLogger(__name__)

#: Blocks compared, in the order the model concatenates them. ``fhr_up_ph`` is deliberately absent:
#: ``teb_vae/lag_attn_rws/DESIGN.md`` S2 records that it is never loaded, so covering it would
#: measure something no model reads -- and it comes from a different selector with no ``sel_*``
#: attributes to cross-check channel identity against.
BLOCKS = ("fhr_st", "fhr_ph", "up_st", "up_ph")

#: Scattering blocks carry $S_0$ in channel $0$ and then the $42$ wavelets; phase blocks are one
#: channel per selected pair. The two need different channel-to-filter maps, hence the split.
SCATTERING_BLOCKS = ("fhr_st", "up_st")

#: Representative channels drawn in the per-filter figures: one fast, one mid, one in the
#: deceleration band. Chosen by index into the bank rather than by frequency so the figure cannot
#: silently move when the bank changes.
SHOWCASE_FILTERS = (6, 20, 30)

#: The loss warm-up the reach guard is checked against, from ``lag_attn_rws/configs/default.yaml``.
WARMUP_PERIOD = 30

#: Budgets tabulated in the survivorship figure, in seconds. ``None`` is the unguarded default.
DEFAULT_BUDGETS = (None, 240.0, 120.0, 60.0, 32.0)

#: Steps the loader trims from each end at ``trim_minutes: 1.0``, and the model's warm-up on top.
#: The A-vs-B gate is reported over the full segment *and* over this interior, because the residual
#: is an edge effect and quoting only the full-segment number would overstate it.
TRIM_STEPS = 15


# =================================================================================================
# Data
# =================================================================================================
def resolve_shard_variant(shard: str) -> str:
    """Which transform wrote a shard.

    Args:
        shard: Path to the HDF5 shard.

    Returns:
        ``'two_sided'`` or ``'causal'``. A shard without the attribute is a legacy two-sided one --
        the state of every shard written before the attribute existed, including the committed
        ``output/hie_cs.hdf5`` this study was built on -- so its absence is resolved silently.
    """
    with h5py.File(shard, "r") as handle:
        return resolve_transform(handle.attrs)


def load_segments(
    shard: str, indices: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Read raw signals, stored feature blocks and the phase-selection attributes from a shard.

    Args:
        shard: Path to the HDF5 shard.
        indices: Segment indices to read, ascending.

    Returns:
        ``(fhr, up, blocks, attrs)`` -- raw signals ``(n, 5280)``, the stored blocks keyed by name
        with a leading segment axis, and the ``sel_*`` attributes of the two phase blocks.
    """
    order = sorted(int(index) for index in indices)
    with h5py.File(shard, "r") as handle:
        fhr = np.asarray(handle["fhr"][order], dtype=np.float64)
        up = np.asarray(handle["up"][order], dtype=np.float64)
        blocks = {name: np.asarray(handle[name][order], dtype=np.float64) for name in BLOCKS}
        attrs = {name: dict(handle[name].attrs) for name in ("fhr_ph", "up_ph")}
    return fhr, up, blocks, attrs


def select_panel_samples(shard: str, n_panels: int) -> List[Tuple[int, str]]:
    """Pick segments for the per-sample coefficient panels, spread across recordings.

    One segment per distinct GUID first, because consecutive segments of the same recording are
    near-duplicates and a gallery of them would show sampling density rather than variation. Only
    once every recording is represented does it fall back to spreading further segments across the
    shard.

    Args:
        shard: Path to the HDF5 shard.
        n_panels: How many segments to return.

    Returns:
        ``(index, guid)`` pairs, ascending by index.
    """
    with h5py.File(shard, "r") as handle:
        raw_guids = np.asarray(handle["guid"][:])
    guids = [g.decode() if isinstance(g, bytes) else str(g) for g in raw_guids]

    first_of_recording: Dict[str, int] = {}
    for index, guid in enumerate(guids):
        first_of_recording.setdefault(guid, index)

    chosen = sorted(first_of_recording.values())[:n_panels]
    if len(chosen) < n_panels:
        # Fewer recordings than panels asked for: top up with segments spread evenly across the
        # shard, skipping any already chosen.
        remaining = n_panels - len(chosen)
        spread = np.linspace(0, len(guids) - 1, remaining + len(chosen) + 2).astype(int)
        for index in spread:
            if len(chosen) >= n_panels:
                break
            if index not in chosen:
                chosen.append(int(index))
        chosen = sorted(chosen)[:n_panels]
    return [(int(index), guids[index]) for index in chosen]


def _arm_b(
    fhr: np.ndarray, up: np.ndarray, bank: Any, pairs: Dict[str, np.ndarray], *, production: bool
) -> Dict[str, np.ndarray]:
    """Arm B for one segment, in either production's conventions or arm C's.

    Args:
        fhr: Raw fetal heart rate, ``(n_signal,)``.
        up: Raw uterine pressure, ``(n_signal,)``.
        bank: The production filter bank.
        pairs: ``{'fhr_ph': ..., 'up_ph': ...}`` pair index arrays.
        production: Whether to use ``'kymatio'`` / ``'kymatio_truncate'`` (validates against the
            shard) rather than ``'full_rate'`` / ``'exact'`` (comparable with arm C).

    Returns:
        The four blocks, each ``(n_channels, 330)``.
    """
    decimation_mode = "kymatio" if production else "full_rate"
    phi_mode = "kymatio_truncate" if production else "exact"
    return {
        "fhr_st": scattering_block_two_sided(fhr, bank, decimation_mode=decimation_mode),
        "up_st": scattering_block_two_sided(up, bank, decimation_mode=decimation_mode),
        "fhr_ph": phase_block_two_sided(fhr, fhr, pairs["fhr_ph"], bank, phi_mode=phi_mode),
        "up_ph": phase_block_two_sided(up, up, pairs["up_ph"], bank, phi_mode=phi_mode),
    }


def _arm_c(
    fhr: np.ndarray, up: np.ndarray, bank: CausalBank, pairs: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Arm C for one segment.

    Args:
        fhr: Raw fetal heart rate, ``(n_signal,)``.
        up: Raw uterine pressure, ``(n_signal,)``.
        bank: The causal bank.
        pairs: ``{'fhr_ph': ..., 'up_ph': ...}`` pair index arrays.

    Returns:
        The four blocks, each ``(n_channels, 330)``.
    """
    return {
        "fhr_st": scattering_block_causal(fhr, bank),
        "up_st": scattering_block_causal(up, bank),
        "fhr_ph": phase_block_causal(fhr, fhr, pairs["fhr_ph"], bank),
        "up_ph": phase_block_causal(up, up, pairs["up_ph"], bank),
    }


# =================================================================================================
# Measurements
# =================================================================================================
def _gathered_arm_c(
    fhr: np.ndarray,
    up: np.ndarray,
    causal: CausalBank,
    pairs: Dict[str, np.ndarray],
    plan: Dict[str, CausalChannelPlan],
) -> Dict[str, np.ndarray]:
    """Arm C reduced to the channels a causal file actually stores.

    The gather is what makes a causal shard checkable at all: the numpy chain produces every
    channel the bank has, while the file holds only the ones whose warm-up closes inside a
    segment. Reducing the reference by the *plan* rather than by the file's width means a build
    that kept 36 of the wrong channels fails here instead of matching on shape alone.

    Args:
        fhr: Raw fetal heart rate, ``(n_signal,)``.
        up: Raw uterine pressure, ``(n_signal,)``.
        causal: The causal bank.
        pairs: Pair index arrays.
        plan: The stored channel plan, keyed by block.

    Returns:
        The four blocks at their stored widths.
    """
    full = _arm_c(fhr, up, causal, pairs)
    return {name: full[name][plan[name].kept] for name in BLOCKS}


def measure_validation(
    stored: Dict[str, np.ndarray],
    fhr: np.ndarray,
    up: np.ndarray,
    bank: Any,
    pairs: Dict[str, np.ndarray],
    *,
    variant: str = TWO_SIDED,
    causal: Optional[CausalBank] = None,
    plan: Optional[Dict[str, CausalChannelPlan]] = None,
) -> Dict[str, Any]:
    r"""The correctness gate: does this module reproduce the shard, and by how much does S15.3 bite.

    Arm A *is* the shard, so which arm reproduces it follows from which transform wrote it. A
    two-sided shard is checked against arm B in production's own conventions; a causal shard
    against arm C reduced by the stored channel plan, which is the same comparison one level
    further along -- the numpy reference against the batched torch chain that wrote the file.

    Reported over the whole segment and over the interior the loader and warm-up leave, because
    the residual is concentrated at the segment start: quoting only the full-segment worst case
    would overstate the disagreement on the steps any model actually trains on.

    The S15.3 ratio is measured between the two ``phi_mode`` variants of the *same* code on the
    *same* product, so it isolates the smoothing operator. The ratio is not bounded by
    $[d/2, d]$: even for a real diagonal product it is
    $d(\mathrm{DC}+A)/(\mathrm{DC}+2A)$ and is unbounded where $A<0$. The exact diagnostic is
    instead the machine-precision identity between the truncating operator and $d$ times the
    analytic projection of the canonically smoothed product. It is measured on a **two-sided**
    shard only: the operator it describes is production's spectral truncation, which is an
    analytic projection and therefore not causal, so the causal chain offers no ``phi_mode`` for
    it to be a deviation from.

    Args:
        stored: Blocks read from the shard, with a leading segment axis.
        fhr: Raw fetal heart rate, ``(n_segments, n_signal)``.
        up: Raw uterine pressure, ``(n_segments, n_signal)``.
        bank: The production filter bank.
        pairs: Pair index arrays.
        variant: The shard's resolved transform.
        causal: The causal bank; required for a causal shard.
        plan: The stored channel plan; required for a causal shard.

    Returns:
        Worst-case relative errors per block, which arm produced them, and -- on a two-sided
        shard -- the S15.3 statistics.

    Raises:
        ValueError: If a causal shard is passed without its bank and plan, or if a stored block is
            a different width than the arm it is checked against.
    """
    two_sided = variant != CAUSAL
    if not two_sided and (causal is None or plan is None):
        raise ValueError(
            f"validating a '{variant}' shard needs the causal bank and the channel plan: the "
            f"reference arm is arm C gathered to the stored channels, not arm B."
        )

    worst_full: Dict[str, float] = {}
    worst_interior: Dict[str, float] = {}
    diagonal_ratios: List[float] = []
    selected_ratios: List[float] = []

    for index in range(fhr.shape[0]):
        if two_sided:
            produced = _arm_b(fhr[index], up[index], bank, pairs, production=True)
        else:
            assert causal is not None and plan is not None  # narrowed by the guard above
            produced = _gathered_arm_c(fhr[index], up[index], causal, pairs, plan)
        for name in BLOCKS:
            reference = stored[name][index]
            if reference.shape[0] != produced[name].shape[0]:
                raise ValueError(
                    f"'{name}': the shard stores {reference.shape[0]} channels but the "
                    f"{'two-sided' if two_sided else 'causal'} reference produces "
                    f"{produced[name].shape[0]}. The two channel axes mean different things, so "
                    f"channel c of one is not channel c of the other and no comparison of them "
                    f"would mean anything."
                )
            scale = float(np.abs(reference).max())
            error = np.abs(reference - produced[name]) / max(scale, 1e-30)
            worst_full[name] = max(worst_full.get(name, 0.0), float(error.max()))
            interior = error[:, TRIM_STEPS + WARMUP_PERIOD : -TRIM_STEPS]
            worst_interior[name] = max(worst_interior.get(name, 0.0), float(interior.max()))

        if not two_sided:
            continue

        # The S15.3 deviation on the channels actually stored. Measured per channel, because the
        # documented claim is that it is *not* a constant rescaling: the truncation keeps the
        # positive-frequency half, which off the diagonal mixes the real part of the product with
        # the Hilbert transform of its imaginary part.
        exact = phase_block_two_sided(fhr[index], fhr[index], pairs["fhr_ph"], bank,
                                      phi_mode="exact")
        truncated = phase_block_two_sided(fhr[index], fhr[index], pairs["fhr_ph"], bank,
                                          phi_mode="kymatio_truncate")
        for channel in range(exact.shape[0]):
            usable = np.abs(exact[channel]) > 1e-6 * np.abs(exact[channel]).max()
            if usable.any():
                selected_ratios.append(float(np.median(truncated[channel][usable]
                                                       / exact[channel][usable])))

        # The exact pin on what the truncation *is*. Not a bound on the ratio: the pointwise ratio
        # is $d(\mathrm{DC} + A)/(\mathrm{DC} + 2A)$ with $A$ the AC part, which is unbounded
        # wherever $A < 0$, so "between d/2 and d" describes the two limiting regimes and not any
        # individual sample. The identity that does hold exactly is the documented one: dropping
        # bins above $M/d$ is dropping the negative-frequency half, because $\hat\phi$ is already
        # $\approx 2\times10^{-22}$ at that edge. So the truncated operator equals $d$ times the
        # analytic projection of the smoothed product, and that is checked here directly.
        projection_error = _analytic_projection_error(fhr[index], pairs["fhr_ph"], bank)
        diagonal_ratios.append(projection_error)

    # Keyed on the gate rather than on an arm letter, because which arm reproduces the shard is a
    # property of the shard: naming the key after arm B would make it a lie on a causal file.
    validation: Dict[str, Any] = {
        "shard_transform": variant,
        "gate_arm": "B (two-sided)" if two_sided else "C (causal, gathered to the stored channels)",
        "gate_max_rel_full_segment": worst_full,
        "gate_max_rel_interior": worst_interior,
        "interior_slice": [TRIM_STEPS + WARMUP_PERIOD, -TRIM_STEPS],
    }
    if two_sided:
        validation.update({
            "s15_3_analytic_projection_max_rel_err": float(np.max(diagonal_ratios)),
            "s15_3_is_analytic_projection": bool(np.max(diagonal_ratios) < 1e-9),
            "s15_3_stored_channel_ratio_median": float(np.median(selected_ratios)),
            "s15_3_stored_channel_ratio_min": float(np.min(selected_ratios)),
            "s15_3_stored_channel_ratio_max": float(np.max(selected_ratios)),
            "s15_3_stored_channel_sign_flips": int(np.sum(np.asarray(selected_ratios) < 0)),
            "n_stored_diagonal_pairs": int((pairs["fhr_ph"][:, 0] == pairs["fhr_ph"][:, 1]).sum()),
        })
    return validation


def _analytic_projection_error(
    fhr: np.ndarray, pairs: np.ndarray, bank: Any
) -> float:
    r"""How far production's truncation is from $d \times$ the analytic projection.

    ``SCATTERING_PHASE_HARMONIC_MATH_Complete.md`` S15.3 states what the deviation *is*: keeping
    Fourier bins $0 \ldots M/d - 1$ drops the matching negative-frequency band, and $\hat\phi$ is
    already $\approx 2\times10^{-22}$ at that edge. The omitted positive bins are not algebraically
    zero, but the result equals $d$ times the analytic (positive-frequency) projection of the
    smoothed product to machine precision under this bank; that numerical identity is the right
    thing to pin.

    Args:
        fhr: One raw segment, ``(n_signal,)``.
        pairs: ``(n_pairs, 2)`` pair indices.
        bank: The production filter bank.

    Returns:
        The largest relative discrepancy between the two.
    """
    from hdf5_dataset.causal_scattering import (
        phase_products,
        production_padding,
        reflect_pad,
        two_sided_responses,
    )

    n_signal = int(fhr.shape[-1])
    pad_left, pad_right, n_padded = production_padding(n_signal)
    responses = two_sided_responses(fhr, bank)
    products = phase_products(responses, responses, pairs, bank.xi)

    smoothed = np.fft.fft(reflect_pad(products, pad_left, pad_right), axis=-1) * bank.phi[None, :]
    # The analytic projection: keep the non-negative frequencies, drop the rest, at full length.
    projected = smoothed.copy()
    projected[:, n_padded // 2 + 1 :] = 0.0
    expected = DECIMATION * np.fft.ifft(projected, axis=-1)
    expected = expected[:, pad_left : pad_left + n_signal][:, ::DECIMATION].real

    produced = phase_block_two_sided(fhr, fhr, pairs, bank, phi_mode="kymatio_truncate")
    return float(np.abs(produced - expected).max() / max(np.abs(expected).max(), 1e-30))


def measure_filters(bank: Any, causal: CausalBank, naive: CausalBank) -> Dict[str, Any]:
    r"""Per-filter facts for both banks: reach, delay, bandwidth, analyticity, warm-up.

    The headline lives here. ``forward_reach`` is applied to the **two-sided bank only**: it
    returns the $95\%$ quantile of future-tap energy, and for a causal kernel that energy is pure
    round-off, so its quantile would be an arbitrary number rather than $0$.
    :func:`~hdf5_dataset.causal_scattering.future_energy_fraction` is the measure that is
    well-defined for both, and it is exactly $0$ for a causal bank because there is no storage for
    a future tap at all.

    Args:
        bank: The production filter bank.
        causal: The gammatone bank.
        naive: The truncated-Morlet contrast bank.

    Returns:
        Per-filter arrays and the aggregate exchange rate.
    """
    reach = np.array([forward_reach(bank, bank.psi[index]) for index in range(bank.n_filters)])
    phi_reach = forward_reach(bank, bank.phi)
    delay = causal.group_delay_s

    embedded, embedded_taps = embed_on_two_sided_axis(causal)
    causal_future = np.array(
        [future_energy_fraction(embedded[k], embedded_taps) for k in range(causal.n_filters)]
    )
    two_sided_future = np.array(
        [future_energy_fraction(np.fft.ifft(bank.psi[k]), two_sided_taps(bank.phi.size))
         for k in range(bank.n_filters)]
    )

    causal_spectra = np.fft.fft(causal.psi, axis=-1)
    naive_spectra = np.fft.fft(naive.psi, axis=-1)
    summaries = [response_summary(causal_spectra[k], bank.xi[k]) for k in range(causal.n_filters)]
    naive_summaries = [response_summary(naive_spectra[k], bank.xi[k]) for k in range(naive.n_filters)]
    two_sided_bw = np.array([half_power_half_width(bank.psi[k]) * 2.0 * FS
                             for k in range(bank.n_filters)])
    support = np.array([causal_support_samples(causal.psi[k]) for k in range(causal.n_filters)]) / FS

    ratio = delay / reach
    return {
        "reach_s": reach,
        "phi_reach_s": float(phi_reach),
        "delay_s": delay,
        "phi_delay_s": causal.phi_group_delay_s,
        "delay_over_reach": ratio,
        "delay_over_reach_median": float(np.median(ratio)),
        "delay_over_reach_min": float(ratio.min()),
        "delay_over_reach_max": float(ratio.max()),
        "causal_future_energy_max": float(causal_future.max()),
        "two_sided_future_energy_median": float(np.median(two_sided_future)),
        "bw3db_two_sided_hz": two_sided_bw,
        "bw3db_causal_hz": np.array([s["bw3db_hz"] for s in summaries]),
        "bw3db_naive_hz": np.array([s["bw3db_hz"] for s in naive_summaries]),
        "dc_gain_rel": np.array([s["dc_gain_rel"] for s in summaries]),
        "neg_freq_gain_rel": np.array([s["neg_freq_gain_rel"] for s in summaries]),
        "neg_freq_gain_rel_naive": np.array([s["neg_freq_gain_rel"] for s in naive_summaries]),
        "causal_support_s": support,
        "phi_support_s": float(causal_support_samples(causal.phi) / FS),
        "n_support_over_segment": int((support > 1320.0).sum()),
        "n_reach_past_horizon": int((reach > HORIZON_S).sum()),
    }


def measure_leakage(
    fhr: np.ndarray,
    bank: Any,
    causal: CausalBank,
    pairs: Dict[str, np.ndarray],
    edit_time_s: float,
) -> Dict[str, Any]:
    r"""The direct test: edit the signal only in the future and see what moves in the past.

    A deceleration is injected at $t > t_0$ and both arms are recomputed. Arm B's coefficients move
    at steps *before* $t_0$ -- that is the leak, and how far back it reaches is the thing the reach
    budget exists to bound. Arm C's cannot move for any structural reason, but they do move by
    $\sim10^{-13}$ relative, because an FFT-based convolution mixes every input sample into every
    output through round-off. Reporting that floor rather than a flat zero is the honest figure,
    and the test module additionally pins one channel to **bitwise** zero through a direct
    time-domain convolution, where no such floor exists.

    Args:
        fhr: One raw segment, ``(n_signal,)``.
        bank: The production filter bank.
        causal: The causal bank.
        pairs: Pair index arrays.
        edit_time_s: $t_0$; the edit occupies $t > t_0$ only.

    Returns:
        Per-arm past-side movement curves and their maxima.
    """
    n_signal = int(fhr.shape[-1])
    time = np.arange(n_signal) / FS
    edited = fhr.copy()
    # A clinically ordinary deceleration, placed far enough after t0 that its own support does not
    # reach back across it -- so any movement before t0 is the filter's doing, not the edit's.
    injection = deceleration(time, nadir_s=edit_time_s + 120.0)
    injection[time <= edit_time_s] = 0.0
    edited = edited + injection

    step0 = int(edit_time_s * FS) // DECIMATION
    results: Dict[str, Any] = {"edit_step": step0, "edit_time_s": edit_time_s}
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for label, compute in (
        ("two_sided", lambda x: _arm_b(x, x, bank, pairs, production=False)),
        ("causal", lambda x: _arm_c(x, x, causal, pairs)),
    ):
        base, moved = compute(fhr), compute(edited)
        curves[label] = {}
        for name in ("fhr_st", "fhr_ph"):
            scale = max(float(np.abs(base[name]).max()), 1e-30)
            delta = np.abs(moved[name] - base[name]) / scale
            past = delta[:, : step0 + 1]
            curves[label][name] = past.max(axis=0)
            results[f"{label}_{name}_max_past_rel"] = float(past.max())
            # How far before the edit the movement is still above a visible threshold. This is
            # right-censored at the start of the segment: if the earliest step tested still moves,
            # the true extent is only known to be at least this, which the flag records rather
            # than letting the number read as a measured endpoint.
            visible = np.where(past.max(axis=0) > 1e-6)[0]
            results[f"{label}_{name}_leak_back_s"] = (
                float((step0 - visible.min()) * DECIMATION / FS) if visible.size else 0.0
            )
            results[f"{label}_{name}_leak_back_censored"] = bool(
                visible.size and visible.min() == 0
            )
    results["curves"] = curves
    return results


def _best_lag(
    reference: np.ndarray, other: np.ndarray, max_lag: int, predicted_lag: int
) -> Tuple[int, float, float, float]:
    r"""Align a causal channel against its two-sided counterpart, three ways.

    The argmax is reported but is **not** the trustworthy number on the phase blocks. A
    phase-harmonic channel is an oscillating real signal rather than a smooth envelope, so the
    cross-correlation has sidelobes one period apart and the argmax can lock onto the wrong one --
    which is what makes the measured phase lag zigzag while its prediction does not. The
    correlation evaluated *at the predicted lag* has no such ambiguity, so it is the one to read;
    the argmax then bounds how much better any alignment could possibly do.

    Args:
        reference: The two-sided series, ``(n_steps,)``.
        other: The causal series, ``(n_steps,)``.
        max_lag: Largest lag searched, in steps.
        predicted_lag: The analytic delay in steps.

    Returns:
        ``(best_lag, r_at_best, r_at_predicted, r_at_zero)``; the lag is positive when *other*
        trails.
    """
    a = reference - reference.mean()
    b = other - other.mean()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0, float("nan"), float("nan"), float("nan")

    def score(lag: int) -> float:
        """Pearson correlation with *other* shifted back by *lag* steps."""
        if lag <= 0:
            x, y = a, b
        else:
            x, y = a[:-lag], b[lag:]
        if x.size < 8 or x.std() < 1e-12 or y.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    scores = np.array([score(lag) for lag in range(max_lag + 1)])
    if np.all(np.isnan(scores)):
        return 0, float("nan"), float("nan"), float("nan")
    best = int(np.nanargmax(scores))
    at_predicted = score(int(min(max(predicted_lag, 0), a.size - 9)))
    return best, float(scores[best]), float(at_predicted), float(scores[0])


def measure_delay(
    fhr: np.ndarray,
    up: np.ndarray,
    bank: Any,
    causal: CausalBank,
    pairs: Dict[str, np.ndarray],
    filters: Dict[str, Any],
    max_lag: int,
) -> Dict[str, Any]:
    r"""Per-channel realised delay of the causal arm against the two-sided one.

    Restricted to steps past each channel's own causal warm-up, because before that the causal
    output is a function of the assumed history rather than of the signal, and correlating against
    it would report the pad.

    A phase channel's warm-up and delay are set by its **slower** leg plus the low-pass, so the
    phase blocks are measured with their own per-channel warm-up rather than the scattering
    blocks'.

    Args:
        fhr: Raw fetal heart rate, ``(n_segments, n_signal)``.
        up: Raw uterine pressure, ``(n_segments, n_signal)``.
        bank: The production filter bank.
        causal: The causal bank.
        pairs: Pair index arrays.
        filters: Output of :func:`measure_filters`, for the per-channel warm-up.
        max_lag: Largest lag searched, in steps.

    Returns:
        Per-block arrays of median best lag and mean correlations.
    """
    warmup = _channel_warmup_steps(pairs, filters)
    predicted = _predicted_channel_delay(filters, pairs)
    predicted_steps = {
        name: np.round(np.asarray(values) * FS / DECIMATION).astype(int)
        for name, values in predicted.items()
    }
    accumulated: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"lag": [], "r_best": [], "r_pred": [], "r_zero": []} for name in BLOCKS
    }

    for index in range(fhr.shape[0]):
        arm_b = _arm_b(fhr[index], up[index], bank, pairs, production=False)
        arm_c = _arm_c(fhr[index], up[index], causal, pairs)
        for name in BLOCKS:
            lags, r_best, r_pred, r_zero = [], [], [], []
            for channel in range(arm_b[name].shape[0]):
                start = min(int(warmup[name][channel]), arm_b[name].shape[1] - 32)
                lag, best, at_predicted, zero = _best_lag(
                    arm_b[name][channel, start:], arm_c[name][channel, start:], max_lag,
                    int(predicted_steps[name][channel]),
                )
                lags.append(lag)
                r_best.append(best)
                r_pred.append(at_predicted)
                r_zero.append(zero)
            accumulated[name]["lag"].append(np.array(lags, dtype=float))
            accumulated[name]["r_best"].append(np.array(r_best))
            accumulated[name]["r_pred"].append(np.array(r_pred))
            accumulated[name]["r_zero"].append(np.array(r_zero))

    return {
        name: {
            # Median over segments for the lag (an integer argmax, so a mean would invent values
            # between the grid points) and mean for the correlations.
            "lag_steps": np.median(np.stack(values["lag"]), axis=0),
            "r_at_best_lag": np.nanmean(np.stack(values["r_best"]), axis=0),
            "r_at_predicted_lag": np.nanmean(np.stack(values["r_pred"]), axis=0),
            "r_at_zero_lag": np.nanmean(np.stack(values["r_zero"]), axis=0),
            "warmup_steps": warmup[name],
            "predicted_delay_s": np.asarray(predicted[name]),
        }
        for name, values in accumulated.items()
    }


def _channel_hz_by_block(bank: Any, pairs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    r"""Representative centre frequency per stored channel, for the panel axes.

    $S_0$ is reported as ``nan`` rather than $0$ Hz -- "the low-pass" and "a $0$ Hz wavelet" are
    different claims, and the panel labels it ``$S_0$``. A phase channel is labelled by its
    **faster** leg $\xi_j$, which is the evaluation's own convention
    (:mod:`teb_vae.lag_attn.eval.band_partition`), not a new one.

    Args:
        bank: The production filter bank.
        pairs: Pair index arrays.

    Returns:
        ``{block: (n_channels,)}`` in Hz.
    """
    scattering = np.concatenate([[np.nan], bank.hz])
    return {
        "fhr_st": scattering,
        "up_st": scattering,
        "fhr_ph": bank.hz[pairs["fhr_ph"][:, 1]],
        "up_ph": bank.hz[pairs["up_ph"][:, 1]],
    }


def _channel_warmup_steps(
    pairs: Dict[str, np.ndarray], filters: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    r"""Causal warm-up per stored channel, in decimated steps -- the figures' black line.

    Supports **add** along a cascade: convolving a kernel of support $W_1$ with one of support
    $W_2$ gives support $W_1 + W_2$. Every stored channel is a cascade ending in the low-pass, so
    every warm-up carries $W_\phi$:

    $$W(S_0) = W_\phi, \qquad W(S_1^{(k)}) = W_k + W_\phi, \qquad
      W(\Phi_{ij}) = \max(W_i, W_j) + W_\phi ,$$

    the last because the phase product is formed pointwise from two responses at the same $t$, so
    it is usable only once the **slower** leg is, and is then smoothed by $\phi$.

    Args:
        pairs: Pair index arrays.
        filters: Output of :func:`measure_filters`.

    Returns:
        ``{block: (n_channels,)}`` in steps.
    """
    support = filters["causal_support_s"]
    phi_support = filters["phi_support_s"]
    step = DECIMATION / FS
    scattering = np.concatenate([[phi_support], support + phi_support]) / step
    return {
        "fhr_st": np.ceil(scattering),
        "up_st": np.ceil(scattering),
        "fhr_ph": np.ceil((np.maximum(support[pairs["fhr_ph"][:, 0]],
                                      support[pairs["fhr_ph"][:, 1]]) + phi_support) / step),
        "up_ph": np.ceil((np.maximum(support[pairs["up_ph"][:, 0]],
                                     support[pairs["up_ph"][:, 1]]) + phi_support) / step),
    }


def measure_survivorship(
    filters: Dict[str, Any], pairs: Dict[str, np.ndarray], budgets: Sequence[Optional[float]]
) -> Dict[str, Any]:
    r"""Channels kept, and how stale, under each budget -- for both arms, under one predicate.

    Both the two-sided reach vector and the causal delay vector are pushed through the **same**
    :func:`~teb_vae.lag_attn.channel_reach.resolve_channel_budget`. Using one function twice is the
    point: two arms compared under two different rules would not be a comparison.

    ``resolve_channel_budget`` raises when the resulting maximum delay exceeds ``warmup_period``,
    and that refusal is recorded as **the answer** rather than propagated as a failure: a budget at
    which the causal arm cannot satisfy the existing warm-up guard is a result about the causal
    arm, not a bug in the measurement.

    Args:
        filters: Output of :func:`measure_filters`.
        pairs: Pair index arrays.
        budgets: Budgets in seconds; ``None`` means unguarded.

    Returns:
        One record per budget per arm.
    """
    stored_reach = block_reach_seconds()
    causal_by_block = _causal_reach_by_block(filters, pairs)

    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for arm, per_block in (("two_sided", stored_reach), ("causal", causal_by_block)):
            target = tuple(per_block["fhr_st"]) + tuple(per_block["fhr_ph"])
            source = tuple(per_block["up_st"]) + tuple(per_block["up_ph"])
            record: Dict[str, Any] = {"budget_s": budget, "arm": arm}
            for stream, values in (("target", target), ("source", source)):
                try:
                    keep, delays = resolve_channel_budget(values, budget, WARMUP_PERIOD)
                    record[f"{stream}_kept"] = len(keep)
                    record[f"{stream}_max_delay_steps"] = max(delays) if delays else 0
                    record[f"{stream}_refused"] = None
                except ValueError as error:
                    record[f"{stream}_kept"] = None
                    record[f"{stream}_max_delay_steps"] = None
                    record[f"{stream}_refused"] = str(error).split(".")[0]
            record["target_total"] = len(target)
            record["source_total"] = len(source)
            rows.append(record)
    return {"rows": rows}


def _predicted_channel_delay(
    filters: Dict[str, Any], pairs: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    r"""The delay each stored channel should physically exhibit, in seconds.

    Distinct from :func:`_causal_reach_by_block`, and the difference is the point. This is what the
    *transform actually does*: $S_1 = |x \star \psi| \star \phi$ passes through both filters, so it
    carries $\tau_g^{\psi} + \tau_g^{\phi}$, and a phase channel carries its slower leg's delay plus
    the low-pass's. :func:`_causal_reach_by_block` instead mirrors
    :func:`~teb_vae.lag_attn.channel_reach.block_reach_seconds`'s composition exactly -- which omits
    the low-pass on $S_1$ -- because that vector is only ever used against the reach vector, where
    matching compositions matter more than either being complete.

    Using this one for the budget comparison would compare a complete delay against an incomplete
    reach; using the other one here understates the fast channels by the low-pass delay, which at
    $13.3$ s dominates everything above $\approx 0.05$ Hz.

    Args:
        filters: Output of :func:`measure_filters`.
        pairs: Pair index arrays.

    Returns:
        ``{block: (n_channels,)}`` in seconds.
    """
    delay = filters["delay_s"]
    phi_delay = filters["phi_delay_s"]
    scattering = np.concatenate([[phi_delay], delay + phi_delay])

    def phase(selection: np.ndarray) -> np.ndarray:
        """Slower leg plus the low-pass."""
        return np.maximum(delay[selection[:, 0]], delay[selection[:, 1]]) + phi_delay

    return {
        "fhr_st": scattering,
        "up_st": scattering,
        "fhr_ph": phase(pairs["fhr_ph"]),
        "up_ph": phase(pairs["up_ph"]),
    }


def _causal_reach_by_block(
    filters: Dict[str, Any], pairs: Dict[str, np.ndarray]
) -> Dict[str, Tuple[float, ...]]:
    r"""The causal arm's per-channel delay, laid out in the stored block order.

    The direct analogue of :func:`~teb_vae.lag_attn.channel_reach.block_reach_seconds`, composed
    the **same** way it is -- $S_0$ from the low-pass, $S_1$ from its wavelet alone, a phase
    channel from its slower leg plus the low-pass. That composition omits the low-pass on $S_1$,
    which understates the physical delay; it is used here anyway, and only here, because this
    vector's only job is to be compared against the reach vector under one predicate, and a
    comparison of two differently-composed vectors would be worse than a comparison of two
    equally-incomplete ones. :func:`_predicted_channel_delay` is the complete version.

    Args:
        filters: Output of :func:`measure_filters`.
        pairs: Pair index arrays.

    Returns:
        ``{block: (n_channels,)}`` in seconds.
    """
    delay = filters["delay_s"]
    phi_delay = filters["phi_delay_s"]
    scattering = tuple(np.concatenate([[phi_delay], delay]).tolist())

    def phase(selection: np.ndarray) -> Tuple[float, ...]:
        """Delay of each phase channel: the slower leg plus the low-pass."""
        return tuple(
            (np.maximum(delay[selection[:, 0]], delay[selection[:, 1]]) + phi_delay).tolist()
        )

    return {
        "fhr_st": scattering,
        "up_st": scattering,
        "fhr_ph": phase(pairs["fhr_ph"]),
        "up_ph": phase(pairs["up_ph"]),
    }


# =================================================================================================
# Entry point
# =================================================================================================
def build_parser() -> argparse.ArgumentParser:
    """The argument surface.

    Every argument defaults to ``None`` and nothing is ``required=True``: both would make the
    Run-button path unusable, the first by rendering a :data:`RUN_ARGS` entry unreachable (the
    launch merge treats any non-``None`` parsed value as having come from the command line) and the
    second by firing before :data:`RUN_ARGS` is read at all. Real defaults are applied in
    :func:`main` after the merge.

    This is also why ``--mode`` is a choice string rather than a ``--self-test`` flag:
    ``action='store_true'`` sets a non-``None`` default of ``False``.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Compare a one-sided (causal) scattering / phase-harmonic transform "
                    "against the production two-sided one."
    )
    parser.add_argument("--mode", choices=("compare", "self-test"),
                        help="'compare' runs the full study; 'self-test' checks the filters and "
                             "the causal chain without touching a shard. Default: compare.")
    parser.add_argument("--shard", help="HDF5 shard. Default: output/hie_cs.hdf5")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Where artifacts go. Default: output/causal_scattering")
    parser.add_argument("--n-samples", dest="n_samples", type=int,
                        help="Segments pooled for the delay estimates. Default: 6")
    parser.add_argument("--sample-index", dest="sample_index", type=int,
                        help="Segment used for the trace and leakage figures. Default: 0")
    parser.add_argument("--n-panels", dest="n_panels", type=int,
                        help="Segments given a full-coefficient panel, spread across distinct "
                             "recordings. 0 skips them. Default: 12")
    parser.add_argument("--order", type=int,
                        help=f"Gammatone order n. Default: {GAMMATONE_ORDER}")
    parser.add_argument("--kernel-taps", dest="kernel_taps", type=int,
                        help=f"Causal kernel length. Default: {CAUSAL_KERNEL_TAPS}")
    parser.add_argument("--edit-time-s", dest="edit_time_s", type=float,
                        help="t0 for the leakage test, in seconds. Default: 600")
    parser.add_argument("--max-lag-steps", dest="max_lag_steps", type=int,
                        help="Largest lag searched for the delay estimate. Default: 200")
    return parser


def main(
    *,
    mode: Optional[str] = None,
    shard: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_samples: Optional[int] = None,
    sample_index: Optional[int] = None,
    n_panels: Optional[int] = None,
    order: Optional[int] = None,
    kernel_taps: Optional[int] = None,
    edit_time_s: Optional[float] = None,
    max_lag_steps: Optional[int] = None,
    argument_sources: Optional[Dict[str, str]] = None,
) -> int:
    """Run the comparison, or the self-test, and write the artifacts.

    Args:
        mode: ``'compare'`` or ``'self-test'``.
        shard: HDF5 shard path.
        output_dir: Destination directory.
        n_samples: Segments pooled for the delay estimates.
        sample_index: Segment used for the trace and leakage figures.
        n_panels: Segments given a full-coefficient panel.
        order: Gammatone order.
        kernel_taps: Causal kernel length.
        edit_time_s: $t_0$ for the leakage test.
        max_lag_steps: Largest lag searched.
        argument_sources: Provenance from the launch merge, recorded in ``summary.json``.

    Returns:
        The process exit code.
    """
    mode = mode or "compare"
    shard = shard or os.path.join("output", "hie_cs.hdf5")
    output_dir = output_dir or os.path.join("output", "causal_scattering")
    n_samples = n_samples or 6
    sample_index = 0 if sample_index is None else sample_index
    n_panels = 12 if n_panels is None else n_panels
    order = order or GAMMATONE_ORDER
    kernel_taps = kernel_taps or CAUSAL_KERNEL_TAPS
    edit_time_s = 600.0 if edit_time_s is None else edit_time_s
    max_lag_steps = max_lag_steps or 200

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("building the production bank and its causal counterparts")
    bank = build_filter_bank()
    causal = build_causal_bank(bank, order=order, n_taps=kernel_taps)
    naive = build_truncated_morlet_bank(bank, n_taps=kernel_taps)
    pairs = {
        "fhr_ph": selected_pairs(TARGET_PHASE_BAND_HZ, bank),
        "up_ph": selected_pairs(SOURCE_PHASE_BAND_HZ, bank),
    }

    if mode == "self-test":
        return self_test(bank, causal, naive, pairs, shard)

    if not os.path.exists(shard):
        logger.error(f"shard not found: {shard}")
        return 2
    os.makedirs(output_dir, exist_ok=True)

    variant = resolve_shard_variant(shard)
    # The plan is what the causal build stored, so it is also what the gate must compare against.
    # Built for both variants because it costs nothing and keeps the branch below to one line.
    plan = build_channel_plan(causal, pairs["fhr_ph"], pairs["up_ph"])
    logger.info(f"shard transform: {variant}")

    indices = list(range(sample_index, sample_index + n_samples))
    logger.info(f"reading {n_samples} segments from {shard}")
    fhr, up, stored, attrs = load_segments(shard, indices)

    # Refuse before measuring anything: arm C channel c means arm A channel c only if this holds,
    # and a silent misalignment would produce a plausible-looking but meaningless comparison. The
    # causal variant carries the same sel_* attributes -- its phase selections are unchanged -- so
    # this check is the same check on either shard.
    assert_matches_shard(pairs["fhr_ph"], bank, attrs["fhr_ph"], name="fhr_ph")
    assert_matches_shard(pairs["up_ph"], bank, attrs["up_ph"], name="up_ph")
    logger.info("channel identity verified against the shard's sel_i/sel_j")

    logger.info("measuring filters")
    filters = measure_filters(bank, causal, naive)
    logger.info(f"validating the {variant} shard against its own arm")
    validation = measure_validation(
        stored, fhr, up, bank, pairs, variant=variant, causal=causal, plan=plan
    )
    logger.info("running the leakage test")
    leakage = measure_leakage(fhr[0], bank, causal, pairs, edit_time_s)
    logger.info(f"measuring per-channel delay over {n_samples} segments")
    delay = measure_delay(fhr, up, bank, causal, pairs, filters, max_lag_steps)
    logger.info("resolving budgets for both arms")
    survivorship = measure_survivorship(filters, pairs, DEFAULT_BUDGETS)

    from hdf5_dataset.causal_scattering_figures import (
        write_all_figures,
        write_gallery,
        write_report,
        write_sample_panels,
    )

    logger.info("writing figures")
    arm_b = _arm_b(fhr[0], up[0], bank, pairs, production=False)
    arm_c = _arm_c(fhr[0], up[0], causal, pairs)
    write_all_figures(
        output_dir, bank, causal, naive, filters, leakage, delay, survivorship,
        traces=(fhr[0], up[0], arm_b, arm_c), showcase=SHOWCASE_FILTERS,
    )
    _write_per_channel(output_dir, bank, filters, delay, pairs)

    panel_indices: List[Tuple[int, str]] = []
    if n_panels > 0:
        panel_indices = select_panel_samples(shard, n_panels)
        logger.info(
            f"transforming {len(panel_indices)} segments for the full-coefficient panels "
            f"(both arms, {len({guid for _, guid in panel_indices})} distinct recordings)"
        )
        panel_fhr, panel_up, _, _ = load_segments(shard, [i for i, _ in panel_indices])
        panels = []
        for position, (index, guid) in enumerate(panel_indices):
            panels.append({
                "index": index,
                "guid": guid,
                "fhr": panel_fhr[position],
                "up": panel_up[position],
                "arm_b": _arm_b(panel_fhr[position], panel_up[position], bank, pairs,
                                production=False),
                "arm_c": _arm_c(panel_fhr[position], panel_up[position], causal, pairs),
            })
        write_sample_panels(
            output_dir, panels, _channel_warmup_steps(pairs, filters),
            channel_hz=_channel_hz_by_block(bank, pairs),
        )
        write_gallery(output_dir, panels)
        logger.info(f"wrote {len(panels)} sample panels and the gallery")

    summary: Dict[str, Any] = {
        "arguments": {
            "mode": mode, "shard": shard, "output_dir": output_dir, "n_samples": n_samples,
            "sample_index": sample_index, "n_panels": n_panels, "order": order,
            "kernel_taps": kernel_taps,
            "edit_time_s": edit_time_s, "max_lag_steps": max_lag_steps,
        },
        "shard_transform": variant,
        # Read off the shard's own blocks, not off the channel plan: the plan describes what a
        # *causal* build stores, which is not what a two-sided shard holds.
        "stored_widths": {name: int(stored[name].shape[1]) for name in BLOCKS},
        "argument_sources": argument_sources or {},
        "validation": validation,
        "headline": {
            "delay_over_reach_median": filters["delay_over_reach_median"],
            "delay_over_reach_min": filters["delay_over_reach_min"],
            "delay_over_reach_max": filters["delay_over_reach_max"],
            "causal_future_energy_max": filters["causal_future_energy_max"],
            "two_sided_future_energy_median": filters["two_sided_future_energy_median"],
            "n_reach_past_horizon": filters["n_reach_past_horizon"],
            "n_causal_support_over_segment": filters["n_support_over_segment"],
            "neg_freq_gain_rel_median": float(np.median(filters["neg_freq_gain_rel"])),
            "neg_freq_gain_rel_max": float(filters["neg_freq_gain_rel"].max()),
        },
        "panel_samples": [{"index": index, "guid": guid} for index, guid in panel_indices],
        "leakage": {key: value for key, value in leakage.items() if key != "curves"},
        "survivorship": survivorship["rows"],
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_jsonable)
    write_report(output_dir, summary, filters, delay)
    logger.info(f"wrote artifacts to {output_dir}")
    return 0


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays for :func:`json.dump`.

    Args:
        value: Any value json cannot serialise natively.

    Returns:
        A serialisable equivalent.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)!r}")


def _write_per_channel(
    output_dir: str,
    bank: Any,
    filters: Dict[str, Any],
    delay: Dict[str, Any],
    pairs: Dict[str, np.ndarray],
) -> None:
    """Write one CSV row per (block, channel) with every per-channel measurement.

    The rows describe the **banks**, not a shard: every column comes from the filter design or
    from both arms recomputed off the raw signals, none of it from a stored block. So this file is
    the same width under either shard variant -- all $43$ scattering channels including the seven
    a causal build drops -- and it has to be, because the dropped channels are exactly the ones a
    reader comes here to find the warm-up of. Anything pinned against this file therefore stays
    valid after the drop landed in the writer.

    Args:
        output_dir: Destination directory.
        bank: The production filter bank.
        filters: Output of :func:`measure_filters`.
        delay: Output of :func:`measure_delay`.
        pairs: Pair index arrays.
    """
    reach_by_block = block_reach_seconds()
    causal_by_block = _causal_reach_by_block(filters, pairs)
    columns = [
        "block", "channel", "xi_hz", "reach_l95_s", "causal_delay_s", "delay_over_reach",
        "bw3db_two_sided_hz", "bw3db_causal_hz", "dc_gain_rel", "neg_freq_gain_rel",
        "causal_warmup_steps", "measured_lag_steps", "measured_lag_s",
        "r_at_best_lag", "r_at_predicted_lag", "r_at_zero_lag",
    ]

    with open(os.path.join(output_dir, "per_channel.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for block in BLOCKS:
            reach = reach_by_block[block]
            causal_delay = causal_by_block[block]
            measured = delay[block]
            for channel in range(len(reach)):
                if block in SCATTERING_BLOCKS:
                    # Channel 0 is S_0 and has no centre frequency; the rest index the bank
                    # directly. Reported as nan rather than 0 Hz -- "the low-pass" and "a 0 Hz
                    # wavelet" are different claims.
                    hz = float("nan") if channel == 0 else float(bank.hz[channel - 1])
                    filter_index = max(channel - 1, 0)
                else:
                    # A phase channel is labelled by the faster leg, the evaluation's convention.
                    hz = float(bank.hz[pairs[block][channel, 1]])
                    filter_index = int(pairs[block][channel, 0])
                lag = float(measured["lag_steps"][channel])
                writer.writerow({
                    "block": block,
                    "channel": channel,
                    "xi_hz": f"{hz:.6g}",
                    "reach_l95_s": f"{reach[channel]:.3f}",
                    "causal_delay_s": f"{causal_delay[channel]:.3f}",
                    "delay_over_reach": f"{causal_delay[channel] / reach[channel]:.4f}",
                    "bw3db_two_sided_hz": f"{filters['bw3db_two_sided_hz'][filter_index]:.6g}",
                    "bw3db_causal_hz": f"{filters['bw3db_causal_hz'][filter_index]:.6g}",
                    "dc_gain_rel": f"{filters['dc_gain_rel'][filter_index]:.3e}",
                    "neg_freq_gain_rel": f"{filters['neg_freq_gain_rel'][filter_index]:.3e}",
                    "causal_warmup_steps": int(measured["warmup_steps"][channel]),
                    "measured_lag_steps": f"{lag:.1f}",
                    "measured_lag_s": f"{lag * DECIMATION / FS:.1f}",
                    "r_at_best_lag": f"{measured['r_at_best_lag'][channel]:.4f}",
                    "r_at_predicted_lag": f"{measured['r_at_predicted_lag'][channel]:.4f}",
                    "r_at_zero_lag": f"{measured['r_at_zero_lag'][channel]:.4f}",
                })


def self_test(
    bank: Any, causal: CausalBank, naive: CausalBank, pairs: Dict[str, np.ndarray], shard: str
) -> int:
    """Check the filters and the causal chain without needing figures or a full comparison.

    Args:
        bank: The production filter bank.
        causal: The gammatone bank.
        naive: The truncated-Morlet bank.
        pairs: Pair index arrays.
        shard: Shard path; the channel-identity check is skipped if it is absent.

    Returns:
        ``0`` if every check passed, ``1`` otherwise.
    """
    failures: List[str] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        """Record and print one check."""
        print(f"  {'PASS' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not passed:
            failures.append(name)

    print("filters")
    embedded, taps = embed_on_two_sided_axis(causal)
    check("causal kernels have no future taps at all",
          int(np.count_nonzero(embedded[:, taps > 0])) == 0)
    check("causal future energy is exactly zero",
          max(future_energy_fraction(embedded[k], taps) for k in range(causal.n_filters)) == 0.0)
    check("zero mean (DC nulled)",
          float(np.abs(causal.psi.sum(axis=1)).max()) < 1e-12,
          f"max |sum psi| = {float(np.abs(causal.psi.sum(axis=1)).max()):.2e}")
    check("unit L1 norm", float(np.abs(np.abs(causal.psi).sum(axis=1) - 1).max()) < 1e-12)
    check("phi sums to one", abs(float(causal.phi.sum()) - 1.0) < 1e-12)

    spectra = np.fft.fft(causal.psi, axis=-1)
    from hdf5_dataset.causal_scattering import GAUSSIAN_HALF_POWER

    measured = np.array([half_power_half_width(spectra[k]) for k in range(causal.n_filters)])
    target = bank.sigma * GAUSSIAN_HALF_POWER
    ratio = measured / target
    check("bandwidth matches the Morlet's nominal sigma within 10%",
          bool(np.all(np.abs(ratio - 1.0) < 0.10)),
          f"ratio in [{ratio.min():.3f}, {ratio.max():.3f}]")

    print("the exchange rate")
    reach = np.array([forward_reach(bank, bank.psi[k]) for k in range(bank.n_filters)])
    exchange = causal.group_delay_s / reach
    check("causal delay is 1.3-1.9x the reach it removes",
          bool(np.all((exchange > 1.3) & (exchange < 1.9))),
          f"median {np.median(exchange):.2f}, range [{exchange.min():.2f}, {exchange.max():.2f}]")

    print("the causal chain, end to end")
    time = np.arange(2048) / FS
    signal = 140.0 + 5.0 * np.sin(2 * np.pi * 0.05 * time)
    edit_step = 1024
    edited = signal.copy()
    edited[edit_step:] += 30.0
    small = build_causal_bank(bank, order=causal.order, n_taps=1024)

    base = scattering_block_causal(signal, small)
    moved = scattering_block_causal(edited, small)
    horizon = edit_step // DECIMATION
    scale = max(float(np.abs(base).max()), 1e-30)
    fft_leak = float(np.abs(moved[:, :horizon] - base[:, :horizon]).max() / scale)
    check("FFT chain: past-side movement at the round-off floor", fft_leak < 1e-10,
          f"max rel = {fft_leak:.2e}")

    # A direct time-domain convolution has no round-off floor to hide behind: the past-side
    # difference must be exactly zero, bit for bit.
    kernel = small.psi[20]
    history = np.concatenate([np.full(kernel.size - 1, signal[0]), signal])
    history_edited = np.concatenate([np.full(kernel.size - 1, edited[0]), edited])
    direct_base = np.convolve(history, kernel)[kernel.size - 1 : kernel.size - 1 + edit_step]
    direct_moved = np.convolve(history_edited, kernel)[kernel.size - 1 : kernel.size - 1 + edit_step]
    check("direct convolution: past side is bitwise identical",
          bool(np.array_equal(direct_base, direct_moved)))

    print("channel identity")
    if os.path.exists(shard):
        with h5py.File(shard, "r") as handle:
            try:
                assert_matches_shard(pairs["fhr_ph"], bank, dict(handle["fhr_ph"].attrs), name="fhr_ph")
                assert_matches_shard(pairs["up_ph"], bank, dict(handle["up_ph"].attrs), name="up_ph")
                check("rebuilt pairs match the shard's sel_i/sel_j", True,
                      f"fhr_ph {len(pairs['fhr_ph'])}, up_ph {len(pairs['up_ph'])}")
            except ValueError as error:
                check("rebuilt pairs match the shard's sel_i/sel_j", False, str(error))
    else:
        print(f"  SKIP  channel identity (no shard at {shard})")

    print("\nnaive truncated Morlet (contrast arm)")
    naive_spectra = np.fft.fft(naive.psi, axis=-1)
    naive_bw = np.array([half_power_half_width(naive_spectra[k]) for k in range(naive.n_filters)])
    print(f"  bandwidth vs nominal: median {np.median(naive_bw / target):.2f}x "
          f"(gammatone {np.median(ratio):.2f}x) -- the cost of cutting at the peak")

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # The shard and output paths are repo-root-relative, and a relative path resolved against
        # an arbitrary working directory surfaces as "shard not found" with no mention of why.
        os.chdir(_REPO_ROOT)
    return main(**values, argument_sources=sources)


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key, so varying one value works without editing
#: anything else, and a key that is not an argparse ``dest`` raises at startup.
#:
#: **Nothing here is required.** With every key left ``None`` this runs the full comparison against
#: ``output/hie_cs.hdf5`` and writes to ``output/causal_scattering``; the working directory is moved
#: to the repository root for you. Set ``"mode": "self-test"`` to check the filters and the causal
#: chain without touching a shard.
#:
#: These are launch conveniences, not a second configuration surface: every value that shapes what
#: the run measures is echoed into ``summary.json`` together with whether it came from here or from
#: the command line, so a finished run's provenance is readable from its own artifacts.
RUN_ARGS: Dict[str, Any] = {
    "mode": None,           # "compare" (default) or "self-test"
    "shard": None,          # HDF5 shard; default output/hie_cs.hdf5
    "output_dir": None,     # default output/causal_scattering
    "n_samples": None,      # segments pooled for the delay estimates; default 6
    "sample_index": None,   # segment used for the trace and leakage figures; default 0
    "n_panels": None,       # segments given a full-coefficient panel, one per recording; default 12
    "order": None,          # gammatone order n; default 4
    "kernel_taps": None,    # causal kernel length; default 32768 (see CAUSAL_KERNEL_TAPS)
    "edit_time_s": None,    # t0 for the leakage test, seconds; default 600
    "max_lag_steps": None,  # largest lag searched for the delay estimate; default 200
}


if __name__ == "__main__":
    sys.exit(_cli())
