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
    build_channel_plan,
    build_truncated_morlet_bank,
    causal_convolve,
    causal_smooth,
    embed_on_two_sided_axis,
    future_energy_fraction,
    half_power_half_width,
    leg_alignment_shift,
    phase_block_causal,
    phase_block_two_sided,
    phase_products,
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

#: Eight real ``fhr``/``up`` segments, tracked in git, that the leg-alignment measurements run on.
#: :data:`SHARD` is git-ignored and exists only on the machine it was built on, so a fidelity claim
#: gated on it would be a claim that evaporates on a clean checkout and on the production box.
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "tests", "data", "causal_fixture.hdf5")

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


# =================================================================================================
# The leg alignment
# =================================================================================================
# The causal phase block multiplies two wavelet responses at one stored index, and the two come
# from filters of different group delay -- so they describe two different physical instants, up to
# 291.6 s apart. ``leg_alignment='envelope'`` delays the faster leg onto the slower one's clock and
# de-rotates its carrier. These tests establish, in order: that switching the mode on changes
# nothing while it is off, that the shift lands per pair rather than per filter, that the aligned
# block is still exactly causal, and that the block it produces is the one the repair predicts.
@pytest.fixture(scope="module")
def fixture_segments():
    """The committed raw signals, ``(8, 5280)`` each, as float64.

    Raises:
        FileNotFoundError: If the tracked fixture is missing, which would otherwise turn the
            fidelity measurement below into a silent skip.
    """
    if not os.path.exists(FIXTURE):
        raise FileNotFoundError(
            f"the committed raw-signal fixture is missing from {FIXTURE}; it is tracked in git, "
            f"so restore it rather than skipping -- the alignment's whole evidence rests on it"
        )
    with h5py.File(FIXTURE, "r") as handle:
        return (np.asarray(handle["fhr"][:], dtype=np.float64),
                np.asarray(handle["up"][:], dtype=np.float64))


def test_the_unaligned_product_is_the_operator_written_out(causal, pairs, fixture_segments):
    r"""The default path is the two-line formula, evaluated independently, bit for bit.

    "The default reproduces today's value" is only checkable against something other than another
    call of the same function, so the reference here is $[y_i]^{p}\,\overline{y_j}$ written out
    from the responses. Bitwise, because the ``None`` branch must take neither the gather nor the
    multiply the aligned branch takes.
    """
    fhr, _ = fixture_segments
    selection = pairs["fhr_ph"]
    responses = causal_convolve(fhr[0], causal.psi)

    power = (causal.xi[selection[:, 1]] / causal.xi[selection[:, 0]])[:, None]
    y_low = responses[selection[:, 0]]
    expected = (np.abs(y_low) * np.exp(1j * power * np.angle(y_low))) * np.conj(
        responses[selection[:, 1]]
    )
    assert np.array_equal(
        phase_products(responses, responses, selection, causal.xi), expected
    )

    # An identity shift must also be a no-op, which is what says the new branch adds nothing of
    # its own: a gather by zero and a multiply by exactly one.
    identity = (np.zeros(selection.shape[0], dtype=np.int64),
                np.ones(selection.shape[0], dtype=np.complex128))
    assert np.array_equal(
        phase_products(responses, responses, selection, causal.xi, leg_shift=identity), expected
    )


def test_each_pair_is_delayed_by_its_own_shift_after_the_gather(causal, pairs, fixture_segments):
    r"""Pair $r$'s conjugated leg is *its own* response, delayed by *its own* shift.

    This is the assertion that catches a per-filter implementation of a per-pair quantity. On the
    shipped bank $22$ of the $24$ distinct ``fhr_ph`` fast legs serve more than one slow partner,
    each at a different harmonic ratio and so at a different $s_{ij}$; a shift applied to the
    response array before the gather would satisfy one partner per filter and be silently wrong
    for the rest, with every shape correct and every existing gate green.

    The reference is built one pair at a time with an explicit roll, so it shares no index
    arithmetic with the implementation it checks.
    """
    fhr, _ = fixture_segments
    selection = pairs["fhr_ph"]
    responses = causal_convolve(fhr[0], causal.psi)
    shift, phasor = leg_alignment_shift(causal, selection)
    produced = phase_products(
        responses, responses, selection, causal.xi, leg_shift=(shift, phasor)
    )

    power = (causal.xi[selection[:, 1]] / causal.xi[selection[:, 0]])[:, None]
    y_low = responses[selection[:, 0]]
    accelerated = np.abs(y_low) * np.exp(1j * power * np.angle(y_low))
    for row in range(selection.shape[0]):
        steps = int(shift[row])
        leg = responses[int(selection[row, 1])]
        delayed = np.roll(leg, steps)
        delayed[:steps] = leg[0]  # edge replication, written out rather than clipped
        expected = accelerated[row] * np.conj(delayed * phasor[row])
        assert np.array_equal(produced[row], expected), row

    # Not vacuous: reused fast legs really do receive different shifts here.
    by_fast_leg = {}
    for (_, fast), steps in zip(selection.tolist(), shift.tolist()):
        by_fast_leg.setdefault(int(fast), set()).add(int(steps))
    assert sum(len(v) > 1 for v in by_fast_leg.values()) == 22


def test_a_leg_shift_of_the_wrong_length_is_refused(causal, pairs, fixture_segments):
    """A shift vector sized for another block's pairs is the mistake being guarded.

    ``fhr_ph`` has $66$ pairs and ``up_ph`` $15$, so passing one block's shift to the other is a
    live confusion; numpy would broadcast neither, but the refusal names the reason rather than
    surfacing a shape error from three frames down.
    """
    fhr, _ = fixture_segments
    responses = causal_convolve(fhr[0], causal.psi)
    shift, phasor = leg_alignment_shift(causal, pairs["up_ph"])
    with pytest.raises(ValueError, match="indexed by pair, not by filter"):
        phase_products(
            responses, responses, pairs["fhr_ph"], causal.xi, leg_shift=(shift, phasor)
        )


def test_the_causal_phase_block_defaults_to_the_unaligned_operator(causal, pairs,
                                                                   fixture_segments):
    """No mode argument, ``'none'``, and today's block are one array; an unknown mode is refused.

    The default carries the compatibility guarantee: every shard on disk, and the committed
    fixture the shared data-contract test rebuilds, came through the no-argument call.
    """
    fhr, _ = fixture_segments
    selection = pairs["fhr_ph"]
    implicit = phase_block_causal(fhr[0], fhr[0], selection, causal)
    explicit = phase_block_causal(fhr[0], fhr[0], selection, causal, leg_alignment="none")
    aligned = phase_block_causal(fhr[0], fhr[0], selection, causal, leg_alignment="envelope")
    assert np.array_equal(implicit, explicit)
    assert not np.array_equal(implicit, aligned)

    with pytest.raises(ValueError, match="unknown leg_alignment 'delay'"):
        phase_block_causal(fhr[0], fhr[0], selection, causal, leg_alignment="delay")
    # Refused before the pair list is consulted, so an empty selection cannot hide a typo.
    with pytest.raises(ValueError, match="unknown leg_alignment"):
        phase_block_causal(
            fhr[0], fhr[0], np.zeros((0, 2), dtype=int), causal, leg_alignment="envelop"
        )


def test_the_two_sided_arm_takes_no_shift(bank, pairs, fixture_segments):
    """Arm B's legs have no skew to remove, and its call site must stay the unaligned one.

    Checked at the level that matters -- the two-sided block equals the product formula with no
    shift anywhere -- because arm B reproducing the shard is what makes every causal-versus-
    two-sided difference attributable to the causal arm.
    """
    from hdf5_dataset.causal_scattering import smooth_products_exact, two_sided_responses

    fhr, _ = fixture_segments
    selection = pairs["fhr_ph"]
    responses = two_sided_responses(fhr[0], bank)
    expected = smooth_products_exact(
        phase_products(responses, responses, selection, bank.xi), bank.phi
    )
    assert np.array_equal(
        phase_block_two_sided(fhr[0], fhr[0], selection, bank, phi_mode="exact"), expected
    )


def test_the_aligned_phase_block_past_side_holds_at_the_round_off_floor(
    bank, causal, pairs, edited_pair
):
    """An edit made only in the future must not move an aligned coefficient before it.

    The alignment adds a delay and a pointwise complex multiply, neither of which can read
    forward, so this is a confirmation rather than a discovery -- but the whole value of the
    causal arm rests on it, and the delay is new index arithmetic that an off-by-one would make
    anticipative. The bound is the same round-off floor the scattering chain is held to.
    """
    signal, edited, edit_step = edited_pair
    selection = pairs["fhr_ph"]
    horizon = edit_step // DECIMATION

    for mode in ("none", "envelope"):
        base = phase_block_causal(signal, signal, selection, causal, leg_alignment=mode)
        moved = phase_block_causal(edited, edited, selection, causal, leg_alignment=mode)
        scale = float(np.abs(base).max())
        movement = float(np.abs(moved[:, :horizon] - base[:, :horizon]).max() / scale)
        assert movement < 1e-10, f"{mode}: {movement:.3e}"

    # The control: the two-sided block *does* move before the same edit, so the bound above is
    # measuring causality rather than a transform that ignored its input.
    two_sided = [
        phase_block_two_sided(x, x, selection, bank, phi_mode="exact") for x in (signal, edited)
    ]
    scale = float(np.abs(two_sided[0]).max())
    assert float(
        np.abs(two_sided[1][:, :horizon] - two_sided[0][:, :horizon]).max() / scale
    ) > 1e-4


def _correlation_at_the_predicted_delay(reference, produced, predicted_step):
    """Pearson correlation of a causal channel against its centred twin at the analytic delay.

    Delegates the arithmetic to the comparison tool's own scorer, so the floors pinned below mean
    the same thing as the ``r_at_predicted`` column of ``output/causal_scattering/per_channel.csv``
    rather than being a second definition that happens to agree today. ``max_lag=0`` skips the
    argmax scan, which this measurement does not read: on an oscillating phase channel the argmax
    can lock onto the wrong sidelobe, while the correlation at the predicted lag has no such
    ambiguity.

    Args:
        reference: The two-sided channel, ``(n_steps,)``.
        produced: The causal channel, ``(n_steps,)``.
        predicted_step: The composed delay in decimated steps.

    Returns:
        The correlation, or NaN on a degenerate channel.
    """
    from hdf5_dataset.compare_causal_scattering import _best_lag

    return _best_lag(reference, produced, 0, int(predicted_step))[2]


def test_the_envelope_alignment_beats_the_shipped_block_and_the_phasor_is_why(
    bank, causal, pairs, fixture_segments
):
    r"""The three-way comparison: aligned, shipped, and delay-without-phasor.

    Each causal phase channel is correlated against its centred counterpart **at the delay the
    channel plan predicts**, $\max(\tau_i,\tau_j) + \tau_\phi$, over the steps past that channel's
    own warm-up, and the median is taken over channel $\times$ segment.

    The delay-only arm is built here from public pieces rather than shipped as a third mode. It
    has no legitimate caller: a gammatone's phase delay at its own centre frequency is zero, so a
    plain shift moves the carrier as well as the envelope and injects a rotation of up to $9.6$
    turns. It exists only to show that the phasor is what does the work -- without it the block is
    *worse* than no alignment at all, which is the failure a reimplementation would produce with
    every shape correct.

    Measured on the eight committed segments -- shipped, aligned, delay-only:
    ``fhr_ph`` $+0.051$, $+0.799$, $-0.411$; ``up_ph`` $+0.110$, $+0.667$, $-0.274$.

    On twelve segments of the full shard the aligned medians are $0.805$ and $0.732$. The gap on
    ``up_ph`` is not a weaker repair: that block has $15$ channels of which $5$ carry the
    $p = 2^{6/4}$ branch-cut rotation that is deliberately measured and not applied, so its pooled
    median sits near the boundary between the two integer families' $+0.87$ / $+0.66$ and that
    family's $-0.72$, and moves with the segment draw. ``fhr_ph``'s $66$ channels put the same
    median well clear of it. The floors here are set from the measurement with margin, and
    :func:`test_the_aligned_residual_is_the_non_integer_harmonic_family` is what turns ``up_ph``'s
    lower figure from a soft number into a stated one.
    """
    fhr, up = fixture_segments
    plan = build_channel_plan(causal, pairs["fhr_ph"], pairs["up_ph"])
    floors = {"fhr_ph": 0.70, "up_ph": 0.55}
    measured = {}

    for name, signals in (("fhr_ph", fhr), ("up_ph", up)):
        selection = pairs[name]
        warmup = plan[name].warmup_steps
        predicted = np.round(plan[name].delay_s * FS / DECIMATION).astype(int)
        shift, phasor = leg_alignment_shift(causal, selection)
        scores = {arm: [] for arm in ("shipped", "aligned", "delay_only")}

        for index in range(signals.shape[0]):
            signal = signals[index]
            centred = phase_block_two_sided(signal, signal, selection, bank, phi_mode="exact")
            # One convolution, three products: the arms then differ in the product alone, which
            # is what makes their comparison a measurement of the alignment.
            responses = causal_convolve(signal, causal.psi)
            arms = {}
            for arm, leg_shift in (
                ("shipped", None),
                ("aligned", (shift, phasor)),
                ("delay_only", (shift, np.ones_like(phasor))),
            ):
                product = phase_products(
                    responses, responses, selection, causal.xi, leg_shift=leg_shift
                )
                arms[arm] = causal_smooth(product, causal.phi).real[:, ::DECIMATION]

            for channel in range(selection.shape[0]):
                start = min(int(warmup[channel]), centred.shape[1] - 32)
                for arm, block in arms.items():
                    scores[arm].append(_correlation_at_the_predicted_delay(
                        centred[channel, start:], block[channel, start:], predicted[channel]
                    ))

        summary = {arm: float(np.nanmedian(v)) for arm, v in scores.items()}
        measured[name] = summary
        assert summary["aligned"] >= floors[name], f"{name}: {summary}"
        assert summary["shipped"] <= 0.20, f"{name}: {summary}"
        assert summary["delay_only"] < summary["shipped"], f"{name}: {summary}"
        assert summary["delay_only"] <= -0.15, f"{name}: {summary}"
        # The mechanism claim, far more robust to the segment draw than any absolute floor: the
        # alignment multiplies the agreement several-fold, and the phasor is what buys it.
        assert summary["aligned"] >= 5.0 * abs(summary["shipped"]), f"{name}: {summary}"

    # The narrower block is the one the branch-cut family dominates, not the one the repair works
    # less well on; stated so the two floors above are not read as two different repairs.
    assert measured["fhr_ph"]["aligned"] > measured["up_ph"]["aligned"]


def test_the_aligned_residual_is_the_non_integer_harmonic_family(bank, causal, pairs,
                                                                 fixture_segments):
    r"""Split by harmonic ratio, the repair is complete on $p = 2$ and $p = 4$ and not on $2^{3/2}$.

    $[y]^{p}$ uses the principal argument, so an unwrapped phase
    $\Psi = \operatorname{Arg} + 2\pi m$ contributes a factor $e^{-i2\pi p\,m(t)}$ that is $1$ for
    integer $p$ and not otherwise. A causal leg and a centred one do not share a wrap count, so on
    the $p = 2^{6/4}$ family the two blocks differ by a rotation no shift can remove -- and taking
    $\Re\{\cdot\}$ of a rotated complex quantity flips its sign. The coefficient is still exactly
    causal and still informative; only its comparability to the centred block of the same name is
    lost, which is the stance the published limits already take for the whole block.

    Asserted rather than described, because it is the reason the pooled floors above sit where
    they do: without it ``up_ph`` merely looks like a weaker repair.
    """
    fhr, up = fixture_segments
    plan = build_channel_plan(causal, pairs["fhr_ph"], pairs["up_ph"])
    # One segment: the family separation is a property of the operator rather than a statistic,
    # and it is an order of magnitude larger than the segment-to-segment spread.
    for name, signal in (("fhr_ph", fhr[0]), ("up_ph", up[0])):
        selection = pairs[name]
        power = bank.hz[selection[:, 1]] / bank.hz[selection[:, 0]]
        integer_family = np.isclose(power, np.round(power), rtol=5e-2)
        assert integer_family.sum() and (~integer_family).sum(), name

        centred = phase_block_two_sided(signal, signal, selection, bank, phi_mode="exact")
        aligned = phase_block_causal(
            signal, signal, selection, causal, leg_alignment="envelope"
        )
        warmup = plan[name].warmup_steps
        predicted = np.round(plan[name].delay_s * FS / DECIMATION).astype(int)
        scores = np.array([
            _correlation_at_the_predicted_delay(
                centred[channel, min(int(warmup[channel]), centred.shape[1] - 32):],
                aligned[channel, min(int(warmup[channel]), centred.shape[1] - 32):],
                predicted[channel],
            )
            for channel in range(selection.shape[0])
        ])
        assert float(np.nanmedian(scores[integer_family])) >= 0.60, name
        assert float(np.nanmedian(scores[~integer_family])) <= -0.40, name
