r"""The phase-harmonic operator versions, and the timing and start-up counterexamples (CFS-01/02).

Three groups of permanent checks, all on the canonical stored timeline (the dataset builder's UP
shift is part of the signal and appears nowhere here):

**The operator.** The legacy ratio-power operator $[y]^p = |y|e^{ip\operatorname{Arg}y}$ has a
finite jump across the principal-angle branch for non-integer $p$ -- $2|y||\sin\pi p| \approx
1.03|y|$ at the stored $p = 2^{3/2}$ -- and the polar construction does nothing about it. That is
pinned here as a counterexample against the legacy operator, so the defect can never be argued
away by a comment again. The corrected integer operator is then held to its contract: continuous
across the branch, carrier-phase consistent on a two-tone input, the same in NumPy and Torch, and
refusing a fractional pair by name.

**Timing.** The channel-alignment convention reads a channel's content at $\kappa\tau_g$ with
$\kappa = 0.875$, the energy centroid. Two direct measurements show that is a convention and not a
universal delay: a ramp through the actual low-pass is delayed by the tap-weighted mean, not the
energy centroid, and a slow narrowband modulation through the actual slow wavelet and low-pass is
delayed by the full composed group delay, $49.5$ s later than $\kappa$ predicts. Envelope leg
alignment is measured to put a phase pair's modulation at that delay while preserving the carrier
phase, and the pure-delay negative control is measured to rotate it.

**Start-up.** The $95\%$-energy warm-up threshold is an approximate initialisation policy, not
independence from the assumed prehistory: for a constant input the output error at the threshold
sample is exactly the kernel's remaining $L^1$ tail, $15.3\%$ of the input on the low-pass, and the
composed slow-channel envelope keeps $15.2\%$ of its mass beyond its rounded warm-up.

The measurements reproduce ``tmp/cfs_audit_2026_09_04.py``; the numbers pinned here are that
script's, rounded to the tolerance each assertion states.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest
import torch
from scipy.signal import fftconvolve

from hdf5_dataset import hdf5_dataset as loader_module
from hdf5_dataset.causal_scattering import (
    ALIGNMENT_DELAY_FACTOR,
    DECIMATION,
    FS,
    N_RAW,
    PHASE_K_STEPS,
    PHASE_K_STEPS_INTEGER,
    PHASE_OPERATOR_INTEGER,
    PHASE_OPERATOR_LEGACY,
    PHASE_OPERATORS,
    SOURCE_PHASE_BAND_HZ,
    TARGET_PHASE_BAND_HZ,
    CausalBank,
    FilterBank,
    assert_matches_shard,
    build_channel_plan,
    causal_convolve,
    causal_support_samples,
    harmonic_index,
    leg_alignment_shift,
    phase_k_steps_for,
    phase_products,
    resolve_phase_power,
    selected_pairs,
    transform_sample,
    validate_phase_operator,
)
from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy
from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset, read_causal_warmup
from hdf5_dataset.tests.conftest import scale_relative_errors

#: The committed fixtures the operator-mismatch refusals are asserted against.
_FIXTURES = Path(__file__).resolve().parents[2] / "teb_vae" / "lag_attn" / "tests" / "fixtures"
LEGACY_SHARD = _FIXTURES / "tiny_shard_causal.hdf5"
INTEGER_SHARD = _FIXTURES / "tiny_shard_causal_int.hdf5"
LEGACY_STATS = _FIXTURES / "tiny_stats_causal.hdf5"
INTEGER_STATS = _FIXTURES / "tiny_stats_causal_int.hdf5"

#: Two inputs on either side of the negative real axis, $2\times10^{-8}$ apart.
_BRANCH_EPSILON = 1e-8


def _branch_inputs() -> np.ndarray:
    """Two unit-modulus responses straddling the branch, plus a constant conjugated leg."""
    low = np.exp(1j * np.array([np.pi - _BRANCH_EPSILON, -np.pi + _BRANCH_EPSILON]))
    return np.stack((low, np.full(2, 1j)))


def _real_product_jump(power: float, phase_operator: str) -> float:
    """The jump in the real phase product across the branch, for one exponent and operator."""
    responses = _branch_inputs()
    product = phase_products(
        responses, responses, np.array([[0, 1]]), np.array([1.0, power]),
        phase_operator=phase_operator,
    )
    return float(abs(np.diff(product.real)[0, 0]))


def _modulation_delay(values: np.ndarray, times: np.ndarray, frequency: float) -> float:
    r"""Least-squares delay of a sinusoidal amplitude modulation $a\cos(2\pi f(t - d))$."""
    phase = 2.0 * np.pi * frequency * times
    design = np.column_stack((np.ones_like(times), np.cos(phase), np.sin(phase)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(np.arctan2(coefficients[2], coefficients[1]) / (2.0 * np.pi * frequency))


# =================================================================================================
# The operator versions
# =================================================================================================
def test_the_loader_restates_the_legacy_operator_name_the_transform_defines() -> None:
    """Two modules, one string: the loader cannot import the transform (kymatio at import), so it
    restates the legacy name, and this is what stops the two drifting apart."""
    assert loader_module.PHASE_OPERATOR_LEGACY == PHASE_OPERATOR_LEGACY
    assert PHASE_OPERATORS == (PHASE_OPERATOR_LEGACY, PHASE_OPERATOR_INTEGER)
    assert phase_k_steps_for(PHASE_OPERATOR_LEGACY) == PHASE_K_STEPS == (4, 6, 8)
    assert phase_k_steps_for(PHASE_OPERATOR_INTEGER) == PHASE_K_STEPS_INTEGER == (4, 8)
    with pytest.raises(ValueError, match="unknown phase_operator"):
        validate_phase_operator("polar_v2")


@pytest.mark.parametrize("power", [2.0, 2.0 ** 1.5, 4.0])
def test_the_legacy_operator_jumps_across_the_branch_exactly_for_the_fractional_family(
    power: float,
) -> None:
    r"""Inputs $2\times10^{-8}$ apart; images $2|\sin\pi p|$ apart -- $1.0266$ at $p = 2^{3/2}$.

    Pinned as the defect it is: this is the counterexample the 2026-09-05 review reproduced, and
    the legacy operator is retained only so shards already on disk keep their meaning.
    """
    jump = _real_product_jump(power, PHASE_OPERATOR_LEGACY)
    expected = 2.0 * abs(np.sin(np.pi * power))
    if abs(power - round(power)) < 1e-9:
        assert jump < 1e-6, f"integer p={power} must be continuous, jump {jump:.3e}"
    else:
        assert jump == pytest.approx(expected, abs=1e-6)
        assert jump > 1.0


@pytest.mark.parametrize("power", [2.0, 4.0])
def test_the_integer_operator_is_continuous_across_the_branch(power: float) -> None:
    """The corrected contract: an integer exponent makes the branch invisible."""
    assert _real_product_jump(power, PHASE_OPERATOR_INTEGER) < 1e-6


def test_the_integer_operator_refuses_a_fractional_pair_in_numpy_and_in_torch(
    causal_bank: CausalBank, bank: FilterBank,
) -> None:
    """A fractional pair under the integer operator is refused by name, not rounded silently."""
    responses = _branch_inputs()
    with pytest.raises(ValueError, match="not within"):
        phase_products(
            responses, responses, np.array([[0, 1]]), np.array([1.0, 2.0 ** 1.5]),
            phase_operator=PHASE_OPERATOR_INTEGER,
        )
    # The torch twin resolves the exponent through the same function, so the same pair list is
    # refused before any convolution is paid for.
    fractional = [
        pair for pair in selected_pairs(TARGET_PHASE_BAND_HZ, bank).tolist()
        if abs(bank.hz[pair[1]] / bank.hz[pair[0]] - 2.0 ** 1.5) < 0.05 * 2.0 ** 1.5
    ]
    assert fractional, "the legacy target selection carries a 2^(3/2) family"
    torch_bank = CausalTorchBank(causal_bank, "cpu", dtype=torch.complex128)
    signal = torch.zeros((1, N_RAW), dtype=torch.float64)
    with pytest.raises(ValueError, match="not within"):
        torch_bank.phase_block(
            signal, signal, np.asarray(fractional[:1]), phase_operator=PHASE_OPERATOR_INTEGER
        )
    with pytest.raises(ValueError, match="unknown phase_operator"):
        torch_bank.phase_block(signal, signal, np.asarray(fractional[:1]), phase_operator="x")


def test_the_integer_selections_are_the_legacy_ones_without_the_fractional_family(
    bank: FilterBank,
) -> None:
    """44 ``fhr_ph`` / 10 ``up_ph``, every one a legacy pair, at harmonics 2 and 4 only -- and
    the exponent the operator applies is the stored integer, not a recomputed ratio."""
    for band, n_legacy, n_integer in (
        (TARGET_PHASE_BAND_HZ, 66, 44), (SOURCE_PHASE_BAND_HZ, 15, 10)
    ):
        legacy = selected_pairs(band, bank)
        integer = selected_pairs(band, bank, PHASE_OPERATOR_INTEGER)
        assert legacy.shape[0] == n_legacy and integer.shape[0] == n_integer
        assert set(map(tuple, integer.tolist())) <= set(map(tuple, legacy.tolist()))
        harmonics = harmonic_index(integer, bank.xi)
        assert set(harmonics.tolist()) == {2, 4}
        power = resolve_phase_power(integer, bank.xi, PHASE_OPERATOR_INTEGER)
        assert np.array_equal(power, harmonics.astype(np.float64))
        # The legacy ratio is within 5% of the integer but is not the integer: the two operators
        # apply measurably different exponents to the same pairs.
        ratio = resolve_phase_power(integer, bank.xi, PHASE_OPERATOR_LEGACY)
        assert np.all(np.abs(ratio - harmonics) < 0.05 * harmonics)
        assert np.any(ratio != harmonics)
    with pytest.raises(ValueError, match="not within"):
        harmonic_index(selected_pairs(TARGET_PHASE_BAND_HZ, bank), bank.xi)


def test_the_integer_operator_is_carrier_phase_consistent_on_a_two_tone_input(
    causal_bank: CausalBank, bank: FilterBank,
) -> None:
    r"""For $x = \cos 2\pi f t + \cos(2\pi k f t + \varphi)$ the product's phase is $-\varphi$.

    With $y_i \approx Ae^{i2\pi ft}$ and $y_j \approx Be^{i(2\pi kft + \varphi)}$ the accelerated
    product $[y_i]^k\overline{y_j}$ is the constant $ABe^{-i\varphi}$, so its normalised real part
    is $\cos\varphi$. Asserted at three phases, under no leg alignment and under envelope
    alignment (whose de-rotation phasor is what keeps this true), and shown to FAIL for a delay
    without the phasor -- the negative control the alignment's own docstring promises.
    """
    integer = selected_pairs(TARGET_PHASE_BAND_HZ, bank, PHASE_OPERATOR_INTEGER)
    # The fastest k = 2 pair, so the filters warm up quickly and the test stays cheap.
    candidates = [p for p in integer.tolist() if harmonic_index(np.array([p]), bank.xi)[0] == 2]
    slow, fast = max(candidates, key=lambda p: bank.hz[p[0]])
    f_slow = float(causal_bank.hz[slow])
    count = 1 << 15
    times = np.arange(count) / FS
    steady = slice(count // 2, None)
    shifts, phasors = leg_alignment_shift(causal_bank, np.array([[slow, fast]]))
    pure_delay = (shifts, np.ones_like(phasors))
    for phi in (0.0, np.pi / 2.0, np.pi):
        x = np.cos(2.0 * np.pi * f_slow * times) + np.cos(2.0 * np.pi * 2.0 * f_slow * times + phi)
        response = causal_convolve(x, causal_bank.psi[[slow, fast]])
        local = np.array([[0, 1]])
        xi = causal_bank.xi[[slow, fast]]
        for name, shift in (("none", None), ("envelope", (shifts, phasors))):
            product = phase_products(
                response, response, local, xi, leg_shift=shift, phase_operator=PHASE_OPERATOR_INTEGER
            )[0, steady]
            normalised = float(np.mean(product.real) / np.mean(np.abs(product)))
            assert normalised == pytest.approx(np.cos(phi), abs=0.05), (name, phi)
        # Delay without the phasor rotates the carrier by 2*pi*xi_j*s: at the phase where the
        # aligned answer is +1 the control must land far from it.
        if phi == 0.0:
            control = phase_products(
                response, response, local, xi, leg_shift=pure_delay,
                phase_operator=PHASE_OPERATOR_INTEGER,
            )[0, steady]
            rotated = float(np.mean(control.real) / np.mean(np.abs(control)))
            expected = float(np.cos(2.0 * np.pi * causal_bank.xi[fast] * shifts[0]))
            assert rotated == pytest.approx(expected, abs=0.05)
            assert abs(rotated - 1.0) > 0.1


def test_numpy_and_torch_agree_under_the_integer_operator_on_real_segments(
    causal_bank: CausalBank, bank: FilterBank, raw_segments: Dict[str, np.ndarray],
) -> None:
    """The float64 gate the legacy operator already passes, now under the corrected one.

    Agreement between the two implementations cannot detect a shared mathematical mistake -- the
    branch tests above are what do that -- so this is only the check that the exponent reaches
    both chains identically.
    """
    fhr, up = raw_segments["fhr"][:2], raw_segments["up"][:2]
    target = selected_pairs(TARGET_PHASE_BAND_HZ, bank, PHASE_OPERATOR_INTEGER)
    source = selected_pairs(SOURCE_PHASE_BAND_HZ, bank, PHASE_OPERATOR_INTEGER)
    torch_bank = CausalTorchBank(causal_bank, "cpu", dtype=torch.complex128)
    produced = transform_batch_numpy(
        torch_bank, fhr.astype(np.float64), up.astype(np.float64), target, source,
        leg_alignment="envelope", phase_operator=PHASE_OPERATOR_INTEGER,
    )
    assert produced["fhr_ph"].shape[1] == 44 and produced["up_ph"].shape[1] == 10
    for index in range(2):
        expected = {
            "fhr_ph": __import__("hdf5_dataset.causal_scattering", fromlist=["phase_block_causal"])
            .phase_block_causal(
                fhr[index].astype(np.float64), fhr[index].astype(np.float64), target,
                causal_bank, leg_alignment="envelope", phase_operator=PHASE_OPERATOR_INTEGER,
            ),
        }
        e_inf, e_2 = scale_relative_errors(produced["fhr_ph"][index], expected["fhr_ph"])
        assert e_inf < 1e-9 and e_2 < 1e-9, (e_inf, e_2)


def test_the_two_sided_arm_refuses_the_integer_operator(bank: FilterBank) -> None:
    """Production computes the legacy operator by definition; a two-sided integer block would
    match no shard on disk, so ``transform_sample`` refuses it rather than inventing one."""
    zeros = np.zeros(N_RAW)
    with pytest.raises(ValueError, match="two-sided arm"):
        transform_sample(zeros, zeros, bank, phase_operator=PHASE_OPERATOR_INTEGER)


# =================================================================================================
# Provenance and refusals around the corrected fixture
# =================================================================================================
def test_the_integer_fixture_records_its_operator_and_harmonics(
    pipeline: Any, bank: FilterBank,
) -> None:
    """Root attribute, per-block attribute, integer harmonics, and 44/10 widths -- and the shard
    check passes on the integer selection and fails on the legacy one against those attributes."""
    assert INTEGER_SHARD.exists(), "regenerate with scripts/make_tiny_shard.py --variants causal_int"
    with h5py.File(INTEGER_SHARD, "r") as handle:
        assert handle.attrs["causal_phase_operator"] == PHASE_OPERATOR_INTEGER
        assert handle["fhr_ph"].shape[1] == 44 and handle["up_ph"].shape[1] == 10
        for name, band in (("fhr_ph", TARGET_PHASE_BAND_HZ), ("up_ph", SOURCE_PHASE_BAND_HZ)):
            attrs = dict(handle[name].attrs)
            assert attrs["sel_phase_operator"] == PHASE_OPERATOR_INTEGER
            assert set(np.asarray(attrs["sel_harmonic"]).tolist()) == {2, 4}
            assert_matches_shard(
                selected_pairs(band, bank, PHASE_OPERATOR_INTEGER), bank, attrs, name=name
            )
            with pytest.raises(ValueError, match="phase pairs"):
                assert_matches_shard(selected_pairs(band, bank), bank, attrs, name=name)
    with h5py.File(LEGACY_SHARD, "r") as handle:
        assert "causal_phase_operator" not in handle.attrs
        assert "sel_harmonic" not in handle["fhr_ph"].attrs
    # The masks the builder resolves under the integer operator describe exactly this file.
    masks = pipeline.compute_scattering_masks(
        N_RAW, scattering_T=DECIMATION, device=torch.device("cpu"), transform="causal",
        leg_alignment="envelope", phase_operator=PHASE_OPERATOR_INTEGER,
    )
    layout = pipeline.describe_layout(masks)
    assert layout["widths"] == {"fhr_st": 36, "fhr_ph": 44, "up_st": 36, "up_ph": 10,
                                "fhr_up_ph": None}
    assert (layout["c_y"], layout["c_u"]) == (80, 46)
    assert layout["causal_phase_operator"] == PHASE_OPERATOR_INTEGER
    with pytest.raises(ValueError, match="two-sided variant"):
        pipeline.compute_scattering_masks(
            N_RAW, scattering_T=DECIMATION, device=torch.device("cpu"),
            phase_operator=PHASE_OPERATOR_INTEGER,
        )


def test_a_legacy_shard_reads_as_the_legacy_operator_and_a_mixed_list_is_refused() -> None:
    """Absence is the legacy operator; two operators in one dataset are refused by name."""
    assert read_causal_warmup([str(LEGACY_SHARD)], 1.0).phase_operator == PHASE_OPERATOR_LEGACY
    assert read_causal_warmup([str(INTEGER_SHARD)], 1.0).phase_operator == PHASE_OPERATOR_INTEGER
    with pytest.raises(ValueError, match="Mixed causal phase operators"):
        read_causal_warmup([str(LEGACY_SHARD), str(INTEGER_SHARD)], 1.0)
    with pytest.raises(ValueError, match="Mixed causal phase operators"):
        CombinedHDF5Dataset(
            paths=[str(LEGACY_SHARD), str(INTEGER_SHARD)],
            cache_size=0, pin_memory=False, trim_minutes=1.0,
        )


def test_statistics_built_under_the_other_operator_are_refused_before_the_width_check() -> None:
    """A stats file is keyed to the operator it accumulated over. Refused by operator, not by the
    width that happens to differ here -- a width is a count and a future version could coincide."""
    for shard, stats in ((INTEGER_SHARD, LEGACY_STATS), (LEGACY_SHARD, INTEGER_STATS)):
        with pytest.raises(ValueError, match="phase-operator mismatch"):
            CombinedHDF5Dataset(
                paths=[str(shard)], cache_size=0, pin_memory=False, trim_minutes=1.0,
                stats_path=str(stats), normalize_fields=["fhr_ph", "up_ph"],
            )
    # And the matching pair loads.
    dataset = CombinedHDF5Dataset(
        paths=[str(INTEGER_SHARD)], cache_size=0, pin_memory=False, trim_minutes=1.0,
        stats_path=str(INTEGER_STATS), normalize_fields=["fhr_ph", "up_ph"],
    )
    sample = dataset[0]
    assert tuple(sample["fhr_ph"].shape) == (300, 44) and tuple(sample["up_ph"].shape) == (300, 10)


# =================================================================================================
# Timing: the kappa convention is a convention
# =================================================================================================
def test_the_low_pass_delays_a_ramp_by_its_mean_not_its_energy_centroid(
    causal_bank: CausalBank,
) -> None:
    r"""A unit-DC positive kernel delays a ramp by $\sum_t t\,\phi[t]$: $13.30$ s, not $11.64$ s."""
    tap_seconds = np.arange(causal_bank.n_taps) / FS
    mean_delay = float(np.sum(tap_seconds * causal_bank.phi))
    centroid = float(np.sum(tap_seconds * causal_bank.phi ** 2) / np.sum(causal_bank.phi ** 2))
    assert mean_delay == pytest.approx(13.3047, abs=2e-3)
    assert centroid == pytest.approx(11.6416, abs=2e-3)
    assert mean_delay == pytest.approx(causal_bank.phi_group_delay_s, abs=1e-5)

    count = 1 << 17
    times = np.arange(count) / FS
    filtered = fftconvolve(times, causal_bank.phi)[:count]
    # Past one full kernel length every output sample is a complete weighted average of the ramp,
    # and for a unit-DC kernel that average is exactly t minus the tap-weighted mean delay.
    steady = slice(2 * causal_bank.n_taps, None)
    assert times[steady].size > 0
    realised = float(np.mean(times[steady] - filtered[steady]))
    assert realised == pytest.approx(mean_delay, abs=0.02)
    assert abs(realised - centroid) > 1.0
    # kappa times the group delay is the energy centroid at gammatone order 4; the ramp does not
    # sit there.
    assert abs(realised - ALIGNMENT_DELAY_FACTOR * mean_delay) > 1.0


def test_a_slow_narrowband_modulation_realises_the_composed_group_delay_not_kappa_times_it(
    causal_bank: CausalBank,
) -> None:
    r"""The slow retained target channel delays a $1/8000$ Hz modulation by $401.4$ s.

    The sum of nominal group delays is $402.16$ s; the $\kappa$ convention predicts $351.9$ s,
    $49.5$ s too early. One counterexample disproves a universal claim; whether $\kappa$ is a
    useful average on real data is a separate, empirical question this test does not answer.
    """
    slow = 30
    count = 1 << 17
    times = np.arange(count) / FS
    modulation = 1.0 / 8000.0
    amplitude = 1.0 + 0.02 * np.cos(2.0 * np.pi * modulation * times)
    tone = np.cos(2.0 * np.pi * causal_bank.hz[slow] * times)
    response = causal_convolve(amplitude * tone, causal_bank.psi[[slow]])[0]
    scattering = fftconvolve(np.abs(response), causal_bank.phi)[:count]
    steady = slice(2 * causal_bank.n_taps, None)
    fitted = _modulation_delay(scattering[steady], times[steady], modulation)
    nominal = float(causal_bank.group_delay_s[slow] + causal_bank.phi_group_delay_s)
    assert nominal == pytest.approx(402.16, abs=0.01)
    assert fitted == pytest.approx(401.4, abs=1.0)
    assert abs(fitted - ALIGNMENT_DELAY_FACTOR * nominal) > 40.0


def test_envelope_alignment_puts_the_pair_at_the_composed_delay_and_the_pure_delay_control_rotates_it(
    causal_bank: CausalBank, bank: FilterBank,
) -> None:
    r"""The alignment's two claims, measured on the actual slow pair under the integer operator.

    Delaying the fast leg with its de-rotation phasor moves the pair's modulation timing from
    $304$ s (unaligned) to the nominal $402$ s and keeps the steady real product; the same delay
    without the phasor lands at the same time but rotates the real statistic to a quarter of its
    value. The pair $(30, 26)$ is a $k = 2$ harmonic, so the integer operator applies.
    """
    slow, fast = 30, 26
    assert harmonic_index(np.array([[slow, fast]]), bank.xi)[0] == 2
    count = 1 << 17
    times = np.arange(count) / FS
    modulation = 1.0 / 8000.0
    amplitude = 1.0 + 0.02 * np.cos(2.0 * np.pi * modulation * times)
    tones = (
        np.cos(2.0 * np.pi * causal_bank.hz[slow] * times)
        + np.cos(2.0 * np.pi * causal_bank.hz[fast] * times)
    )
    response = causal_convolve(amplitude * tones, causal_bank.psi[[slow, fast]])
    shifts, phasors = leg_alignment_shift(causal_bank, np.array([[slow, fast]]))
    steady = slice(2 * causal_bank.n_taps, None)
    results = {}
    for name, alignment in (
        ("none", None), ("envelope", (shifts, phasors)),
        ("delay_without_phasor", (shifts, np.ones_like(phasors))),
    ):
        product = phase_products(
            response, response, np.array([[0, 1]]), causal_bank.xi[[slow, fast]],
            leg_shift=alignment, phase_operator=PHASE_OPERATOR_INTEGER,
        )[0]
        smoothed = fftconvolve(product, causal_bank.phi)[:count]
        dc = float(np.mean(smoothed.real[steady]))
        results[name] = (dc, _modulation_delay(smoothed.real[steady] / dc, times[steady], modulation))
    nominal = float(causal_bank.group_delay_s[slow] + causal_bank.phi_group_delay_s)
    assert results["envelope"][1] == pytest.approx(nominal, abs=2.0)
    assert abs(results["none"][1] - nominal) > 50.0
    assert results["envelope"][0] == pytest.approx(results["none"][0], rel=0.01)
    assert results["delay_without_phasor"][0] < 0.5 * results["envelope"][0]


# =================================================================================================
# Start-up: the 95% energy threshold is a policy, not independence
# =================================================================================================
def test_the_energy_warm_up_leaves_the_l1_tail_as_the_error_on_a_constant_input(
    causal_bank: CausalBank,
) -> None:
    r"""Outside its $95\%$-energy support the low-pass still carries $15.3\%$ of its $L^1$ mass.

    Two prehistories -- ``'edge'`` (the constant continues) and ``'zero'`` -- differ by exactly
    the input level $c$, and the outputs at step $t$ differ by $c\sum_{r>t}\phi[r]$, the $L^1$
    tail beyond $t$. For a positive kernel the bound is attained. The support encloses taps
    $0 \ldots W-1$, so the last step the support calls "not yet valid", $t = W - 1$, still errs by
    the whole $15.3\%$ outside the support, and the first step it calls valid, $t = W$, by the
    $14.7\%$ beyond it. The threshold certifies the configured energy quantile, not independence
    from the assumed history.
    """
    warmup = causal_support_samples(causal_bank.phi)
    assert warmup == 80
    tail_outside_support = float(causal_bank.phi[warmup:].sum())
    tail_beyond_first_valid = float(causal_bank.phi[warmup + 1:].sum())
    assert tail_outside_support == pytest.approx(0.1534, abs=1e-3)
    assert tail_beyond_first_valid == pytest.approx(0.1467, abs=1e-3)
    energy_inside = float((causal_bank.phi[:warmup] ** 2).sum() / (causal_bank.phi ** 2).sum())
    assert energy_inside >= 0.95

    level = 7.0
    constant = np.full(N_RAW, level)
    smoothed = {
        pad: causal_convolve(constant, causal_bank.phi[None, :], pad=pad)[0] for pad in ("edge", "zero")
    }
    # 'edge' is exact for a constant; 'zero' is short by the tail mass beyond each step, so the
    # difference IS the tail bound, attained, at both the last invalid and the first valid step.
    assert np.allclose(smoothed["edge"], level, atol=1e-9)

    def relative_error(step: int) -> float:
        return float(abs(smoothed["edge"][step] - smoothed["zero"][step])) / level

    assert relative_error(warmup - 1) == pytest.approx(tail_outside_support, abs=2e-3)
    assert relative_error(warmup) == pytest.approx(tail_beyond_first_valid, abs=2e-3)
    assert relative_error(warmup) > 0.05


def test_the_slow_channels_composed_envelope_keeps_mass_beyond_its_rounded_warm_up(
    causal_bank: CausalBank,
) -> None:
    r"""$|\psi_{30}| \star \phi$ retains $15.2\%$ of its mass beyond the $596$ s composed warm-up."""
    slow = 30
    composed = fftconvolve(np.abs(causal_bank.psi[slow]), causal_bank.phi)
    warmup = int(
        np.ceil((causal_support_samples(causal_bank.psi[slow]) + causal_support_samples(causal_bank.phi))
                / DECIMATION) * DECIMATION
    )
    assert warmup / FS == pytest.approx(596.0)
    fraction = float(composed[warmup:].sum() / composed.sum())
    assert fraction == pytest.approx(0.152, abs=3e-3)
    # The plan stores this warm-up as the number a consumer honours, so the two must agree.
    target = selected_pairs(TARGET_PHASE_BAND_HZ, causal_bank if False else __import__(
        "hdf5_dataset.causal_scattering", fromlist=["build_filter_bank"]).build_filter_bank())
    plan = build_channel_plan(causal_bank, target, selected_pairs(SOURCE_PHASE_BAND_HZ))
    assert int(plan["fhr_st"].warmup_steps[np.flatnonzero(plan["fhr_st"].kept == slow + 1)[0]]) == (
        warmup // DECIMATION
    )
