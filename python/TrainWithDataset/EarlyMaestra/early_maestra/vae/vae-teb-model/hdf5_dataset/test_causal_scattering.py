r"""Verification for :mod:`hdf5_dataset.causal_scattering`.

Each test pins one property the comparison's conclusions rest on. Two of them are the load-bearing
ones and are worth naming here:

* :func:`test_causal_kernels_have_no_future_taps` and
  :func:`test_direct_convolution_past_side_is_bitwise_identical` together establish causality at
  two levels. The first is structural -- a :class:`~hdf5_dataset.causal_scattering.CausalBank`
  indexes kernels by delay, so there is no storage for a future tap and the future half of the
  embedded kernel is *bitwise* zero. The second runs a real edit through a real convolution and
  demands bitwise-unchanged output before the edit. The FFT path cannot make that claim -- an FFT
  convolution mixes every input into every output through round-off -- which is why the direct
  convolution is what carries the exact assertion and the FFT path only gets a floor.
* :func:`test_arm_b_reproduces_the_shard` is the gate that makes the whole comparison
  interpretable. If this module's two-sided implementation does not reproduce production, then
  every causal-versus-two-sided difference is confounded with a reimplementation difference.

Run from the repository root::

    .venv/Scripts/python.exe -m pytest hdf5_dataset/test_causal_scattering.py -q
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from hdf5_dataset.causal_scattering import (
    GAUSSIAN_HALF_POWER,
    assert_matches_shard,
    build_causal_bank,
    build_truncated_morlet_bank,
    embed_on_two_sided_axis,
    future_energy_fraction,
    half_power_half_width,
    phase_block_two_sided,
    production_padding,
    scattering_block_causal,
    scattering_block_two_sided,
    selected_pairs,
    two_sided_taps,
)
from teb_vae.lag_attn.channel_reach import SOURCE_PHASE_BAND_HZ, TARGET_PHASE_BAND_HZ
from teb_vae.lag_attn.eval.representation_capacity_probe import (
    DECIMATION,
    FS,
    N_RAW,
    build_filter_bank,
    forward_reach,
)

#: The committed shard the channel-identity and arm-B gates run against.
SHARD = os.path.join("output", "hie_cs.hdf5")

#: Kernel length for the tests that do not depend on the slowest filters being fully resolved.
#: Shorter than the shipped default so the suite stays fast; the length requirement itself is
#: checked separately by :func:`test_default_kernel_length_resolves_every_filter`.
TEST_TAPS = 1 << 12


@pytest.fixture(scope="module")
def bank():
    """The production two-sided filter bank."""
    return build_filter_bank()


@pytest.fixture(scope="module")
def causal(bank):
    """A causal gammatone bank at the shipped default length."""
    return build_causal_bank(bank)


@pytest.fixture(scope="module")
def pairs(bank):
    """The stored phase-harmonic selections, rebuilt."""
    return {
        "fhr_ph": selected_pairs(TARGET_PHASE_BAND_HZ, bank),
        "up_ph": selected_pairs(SOURCE_PHASE_BAND_HZ, bank),
    }


requires_shard = pytest.mark.skipif(
    not os.path.exists(SHARD), reason=f"no shard at {SHARD}"
)


# =================================================================================================
# The filters
# =================================================================================================
def test_causal_kernels_have_no_future_taps(causal):
    """Embedded on a signed tap axis, the future half of every causal kernel is bitwise zero.

    Asserted with ``count_nonzero`` rather than a tolerance: the kernels are *stored* delay-indexed,
    so a future tap is not small, it is absent.
    """
    embedded, taps = embed_on_two_sided_axis(causal)
    assert np.count_nonzero(embedded[:, taps > 0]) == 0
    assert all(
        future_energy_fraction(embedded[k], taps) == 0.0 for k in range(causal.n_filters)
    )


def test_two_sided_bank_reads_half_its_energy_from_the_future(bank):
    """The control for the test above: the production filters are symmetric, so ~50%.

    Without this, "the causal bank's future energy is zero" would be consistent with the measure
    simply returning zero for everything.
    """
    taps = two_sided_taps(bank.phi.size)
    fractions = np.array(
        [future_energy_fraction(np.fft.ifft(bank.psi[k]), taps) for k in range(bank.n_filters)]
    )
    assert 0.40 < float(np.median(fractions)) <= 0.5


def test_forward_reach_is_meaningless_on_a_causal_kernel(causal):
    r"""Why :func:`future_energy_fraction` exists rather than reusing ``forward_reach``.

    ``forward_reach`` returns the $95\%$ *quantile* of future-tap energy. For a causal kernel that
    energy is pure round-off, so the quantile of it is an arbitrary number rather than $0$ -- which
    would make "the causal bank's reach is zero" a claim that silently fails. Pinning the failure
    mode keeps a future reader from reintroducing it.
    """
    embedded, _ = embed_on_two_sided_axis(causal)

    class _Axis:
        """Minimal stand-in exposing the tap axis ``forward_reach`` reads."""

        taps = two_sided_taps(embedded.shape[1]) / FS

    reaches = [forward_reach(_Axis(), np.fft.fft(embedded[k])) for k in range(causal.n_filters)]
    assert max(reaches) > 1.0, "expected forward_reach to report a spurious nonzero reach"


def test_causal_wavelets_are_zero_mean(causal):
    r"""$\hat\psi(0) = 0$ exactly, so an FHR baseline near $140$ bpm cannot enter the passband.

    Without the $\kappa$ correction the slowest filter passes $11.9\%$ of DC relative to its peak.
    """
    assert float(np.abs(causal.psi.sum(axis=1)).max()) < 1e-12


def test_l1_normalisation_matches_kymatio(causal):
    r"""Unit $L^1$ norm in time, and $\phi$ summing to $1$ so its DC gain matches production's."""
    assert float(np.abs(np.abs(causal.psi).sum(axis=1) - 1.0).max()) < 1e-12
    assert abs(float(causal.phi.sum()) - 1.0) < 1e-12


def test_bandwidth_matches_the_morlet_it_replaces(bank, causal):
    r"""The matched bandwidth is a measurement, not an intention.

    Both widths are half-power. Pairing the Gaussian's half-*amplitude* width $\sigma\sqrt{2\ln 2}$
    with the gamma's half-power width would give a bank $41\%$ too wide and a group delay $41\%$
    too short -- flattering the causal arm on exactly the axis under study, and this test is what
    would catch it.
    """
    spectra = np.fft.fft(causal.psi, axis=-1)
    measured = np.array([half_power_half_width(spectra[k]) for k in range(causal.n_filters)])
    ratio = measured / (bank.sigma * GAUSSIAN_HALF_POWER)
    assert np.all(np.abs(ratio - 1.0) < 0.10), f"ratio range {ratio.min():.3f}..{ratio.max():.3f}"


def test_the_exchange_rate_is_worse_than_one(bank, causal):
    r"""The headline: causal delay is $\approx 1.5\times$ the forward reach it removes.

    Pinned as a range rather than a point so a change of gamma order moves it visibly instead of
    silently.
    """
    reach = np.array([forward_reach(bank, bank.psi[k]) for k in range(bank.n_filters)])
    ratio = causal.group_delay_s / reach
    assert np.all(ratio > 1.3) and np.all(ratio < 1.9)
    assert 1.4 < float(np.median(ratio)) < 1.6


def test_naive_truncation_costs_the_passband(bank):
    """The contrast arm earns its place: cutting the Morlet at its peak widens the band ~1.7x.

    This is what makes "use a matched gammatone" a measured recommendation rather than an
    assertion -- and why no taper is applied at the cut, the discontinuity being at the maximum.
    """
    naive = build_truncated_morlet_bank(bank, n_taps=TEST_TAPS)
    spectra = np.fft.fft(naive.psi, axis=-1)
    measured = np.array([half_power_half_width(spectra[k]) for k in range(12)])
    ratio = measured / (bank.sigma[:12] * GAUSSIAN_HALF_POWER)
    assert float(np.median(ratio)) > 1.3


def test_default_kernel_length_resolves_every_filter(bank):
    r"""The shipped kernel length is a measured requirement, not a round number.

    At the corrected rate $b = 1.914\,\sigma$ the slowest kernel retains only $\approx 71\%$ of its
    $L^1$ mass inside production's own $2^{13}$ taps. Enlarging the grid is legitimate *because*
    the filters are causal -- the extra taps reach further into the past, never forward.
    """
    from hdf5_dataset.causal_scattering import (
        CAUSAL_KERNEL_TAPS,
        GAMMATONE_ORDER,
        _gamma_envelope,
        gammatone_rate,
    )

    # Compare against the same envelope on a grid long enough to be effectively infinite, rather
    # than against a closed form: the kernels are peak-normalised, so a direct mass ratio is the
    # assumption-free measurement. The slowest filter is the binding one.
    reference_taps = 1 << 18
    rate = float(np.asarray(gammatone_rate(bank.sigma, GAMMATONE_ORDER))[-1])
    envelope = _gamma_envelope(np.arange(reference_taps, dtype=float), GAMMATONE_ORDER, rate)
    total = envelope.sum()

    assert envelope[: 1 << 13].sum() / total < 0.80, (
        "production's own 2^13 padded length should visibly truncate the slowest causal kernel"
    )
    assert envelope[:CAUSAL_KERNEL_TAPS].sum() / total > 0.999, (
        "the shipped kernel length should resolve the slowest filter's mass"
    )


# =================================================================================================
# The causal chain, end to end
# =================================================================================================
@pytest.fixture(scope="module")
def edited_pair():
    """A synthetic signal and a copy edited only after the midpoint.

    Full production length: ``production_padding`` runs kymatio's own support computation, which
    cannot build a $J = 11$ bank for a short signal and raises rather than returning a smaller
    pad. The two-sided arm is therefore only defined at the length the pipeline uses.
    """
    time = np.arange(N_RAW) / FS
    signal = 140.0 + 5.0 * np.sin(2 * np.pi * 0.05 * time)
    edit_step = N_RAW // 2
    edited = signal.copy()
    edited[edit_step:] += 30.0
    return signal, edited, edit_step


def test_fft_chain_past_side_holds_at_the_round_off_floor(bank, edited_pair):
    """An edit made only in the future must not move causal coefficients in the past.

    The bound is a floor rather than zero because an FFT convolution mixes every input sample into
    every output through round-off; the exact claim is carried by the direct-convolution test.
    """
    signal, edited, edit_step = edited_pair
    small = build_causal_bank(bank, n_taps=TEST_TAPS)
    base = scattering_block_causal(signal, small)
    moved = scattering_block_causal(edited, small)
    horizon = edit_step // DECIMATION
    scale = float(np.abs(base).max())
    assert float(np.abs(moved[:, :horizon] - base[:, :horizon]).max() / scale) < 1e-10


def test_two_sided_chain_past_side_moves(bank, edited_pair):
    """The control: the same edit *does* move the two-sided coefficients before it.

    Without this the test above would pass just as well on a transform that ignored its input.
    """
    signal, edited, edit_step = edited_pair
    base = scattering_block_two_sided(signal, bank)
    moved = scattering_block_two_sided(edited, bank)
    horizon = edit_step // DECIMATION
    scale = float(np.abs(base).max())
    assert float(np.abs(moved[:, :horizon] - base[:, :horizon]).max() / scale) > 1e-4


def test_direct_convolution_past_side_is_bitwise_identical(bank, edited_pair):
    """The exact causality claim, through a convolution with no round-off floor to hide behind."""
    signal, edited, edit_step = edited_pair
    small = build_causal_bank(bank, n_taps=TEST_TAPS)
    kernel = small.psi[20]
    history = kernel.size - 1

    def filtered(x):
        """Direct causal convolution with an edge-padded history."""
        padded = np.concatenate([np.full(history, x[0]), x])
        return np.convolve(padded, kernel)[history : history + edit_step]

    assert np.array_equal(filtered(signal), filtered(edited))


def test_edge_padding_is_annihilated_by_a_zero_mean_wavelet(bank):
    """Edge padding asserts only local constancy, and a zero-mean filter ignores a constant.

    This is what makes ``'edge'`` the safe causal pad: the assumption it introduces costs nothing
    in the passband. Reflection is the unsafe one, and is deliberately not offered.
    """
    small = build_causal_bank(bank, n_taps=TEST_TAPS)
    constant = np.full(1024, 137.0)
    coefficients = scattering_block_causal(constant, small)
    # Channel 0 is the low-pass and legitimately reproduces the constant; the wavelets must not.
    assert float(np.abs(coefficients[1:]).max()) < 1e-9


# =================================================================================================
# Geometry and channel identity
# =================================================================================================
def test_production_padding_matches_the_shipped_geometry():
    """The padding chain the two-sided arm places its signal with."""
    assert production_padding() == (1456, 1456, 8192)


@requires_shard
def test_rebuilt_pairs_match_the_shard(bank, pairs):
    """Channel $c$ here is channel $c$ on disk.

    The comparison is meaningless without this, so the runner refuses before measuring anything.
    """
    with h5py.File(SHARD, "r") as handle:
        assert_matches_shard(pairs["fhr_ph"], bank, dict(handle["fhr_ph"].attrs), name="fhr_ph")
        assert_matches_shard(pairs["up_ph"], bank, dict(handle["up_ph"].attrs), name="up_ph")
    assert len(pairs["fhr_ph"]) == 66
    assert len(pairs["up_ph"]) == 15


@requires_shard
def test_mismatched_pairs_are_refused(bank, pairs):
    """The guard above fails loudly rather than proceeding on a misaligned channel map."""
    scrambled = pairs["fhr_ph"].copy()
    scrambled[0, 0] += 1
    with h5py.File(SHARD, "r") as handle:
        attrs = dict(handle["fhr_ph"].attrs)
    with pytest.raises(ValueError, match="differ from the shard"):
        assert_matches_shard(scrambled, bank, attrs, name="fhr_ph")


# =================================================================================================
# The correctness gate
# =================================================================================================
@requires_shard
def test_arm_b_reproduces_the_shard(bank, pairs):
    """The two-sided implementation is production's, to float32 storage precision.

    Tolerances differ per block for a measured reason, not a fudge: three blocks land at
    ``~2e-7``, which is the tightest agreement float32 storage can express, while ``fhr_ph`` lands
    near ``1e-4``, concentrated at the segment start on the four channels whose harmonic power is
    $2^{6/4}$. The comparison is therefore run over the interior the loader's trim and the model's
    warm-up leave, which is the region any conclusion is drawn from.
    """
    with h5py.File(SHARD, "r") as handle:
        fhr = np.asarray(handle["fhr"][3], dtype=np.float64)
        up = np.asarray(handle["up"][3], dtype=np.float64)
        stored = {name: np.asarray(handle[name][3], dtype=np.float64)
                  for name in ("fhr_st", "up_st", "fhr_ph", "up_ph")}

    produced = {
        "fhr_st": scattering_block_two_sided(fhr, bank, decimation_mode="kymatio"),
        "up_st": scattering_block_two_sided(up, bank, decimation_mode="kymatio"),
        "fhr_ph": phase_block_two_sided(fhr, fhr, pairs["fhr_ph"], bank,
                                        phi_mode="kymatio_truncate"),
        "up_ph": phase_block_two_sided(up, up, pairs["up_ph"], bank,
                                       phi_mode="kymatio_truncate"),
    }
    tolerance = {"fhr_st": 1e-6, "up_st": 1e-6, "up_ph": 1e-5, "fhr_ph": 5e-4}
    interior = slice(45, -15)
    for name, expected in stored.items():
        error = np.abs(expected[:, interior] - produced[name][:, interior]).max()
        assert error / np.abs(expected).max() < tolerance[name], name


@requires_shard
def test_production_phase_smoothing_is_the_analytic_projection(bank, pairs):
    r"""S15.3, pinned as the exact identity it is rather than as an approximate ratio.

    Production truncates the smoothed spectrum to its first $M/d$ bins. $\hat\phi$ is already
    $\approx 2\times10^{-22}$ at that edge, so the omitted positive bins are numerically negligible
    and the result is $d$ times the analytic projection of the smoothed product to machine
    precision under this bank.
    """
    from hdf5_dataset.causal_scattering import (
        phase_products,
        reflect_pad,
        two_sided_responses,
    )

    with h5py.File(SHARD, "r") as handle:
        fhr = np.asarray(handle["fhr"][3], dtype=np.float64)

    pad_left, pad_right, n_padded = production_padding(fhr.size)
    responses = two_sided_responses(fhr, bank)
    products = phase_products(responses, responses, pairs["fhr_ph"], bank.xi)
    smoothed = np.fft.fft(reflect_pad(products, pad_left, pad_right), axis=-1) * bank.phi[None, :]
    smoothed[:, n_padded // 2 + 1 :] = 0.0
    expected = DECIMATION * np.fft.ifft(smoothed, axis=-1)
    expected = expected[:, pad_left : pad_left + fhr.size][:, ::DECIMATION].real

    produced = phase_block_two_sided(fhr, fhr, pairs["fhr_ph"], bank,
                                     phi_mode="kymatio_truncate")
    assert np.abs(produced - expected).max() / np.abs(expected).max() < 1e-9
