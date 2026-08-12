r"""The causal transform: its geometry, its torch chain, its channel plan, and its import path.

Four things are pinned here that are easy to get wrong in ways nothing else would notice.

**The geometry is pinned by value, not by provenance.** The filter bank, the band edges and the
phase-pair rule used to be imported from ``teb_vae``; they now live in
:mod:`hdf5_dataset.causal_scattering` because the production package layout has no ``teb_vae`` on
its import path. Copying values across a package boundary is exactly the operation that silently
diverges later, so :func:`test_the_geometry_is_pinned_by_value` asserts the concrete numbers
independently of where they came from, and
:func:`test_the_absorbed_geometry_equals_the_probe_module` compares against the original while it
still exists -- guarded on importability, so it stops being a dependency the day that module moves.

**The torch chain is gated against the numpy reference, scale-relatively.** The batched path
exists because the validated numpy one cannot run at dataset scale; everything downstream is only
as trustworthy as their agreement. The gate normalises by each block's own scale rather than
pointwise, and a mutation test proves it can fail.

**The channel plan is the single source of widths.** A stored width, a warm-up vector and a
channel order that disagree would be silently wrong data, so the plan is pinned against the
published measurements where they exist and against hand-composed values everywhere.

**The import path is checked in a subprocess.** This ``conftest`` has already imported the causal
modules by the time any test runs, so an in-process assertion about how they import would pass
whatever their import statements said. Only a fresh interpreter can answer the question.
"""
from __future__ import annotations

import csv
import dataclasses
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import torch

from hdf5_dataset.causal_scattering import (
    CAUSAL_KERNEL_TAPS,
    CAUSAL_WARMUP_QUANTILE,
    DECIMATION,
    FS,
    J,
    N_RAW,
    PHASE_K_STEPS,
    PHASE_REL_TOL,
    Q,
    SOURCE_PHASE_BAND_HZ,
    T,
    TARGET_PHASE_BAND_HZ,
    CausalBank,
    CausalChannelPlan,
    FilterBank,
    build_channel_plan,
    build_filter_bank,
    causal_convolve,
    causal_smooth,
    causal_support_samples,
    gammatone_rate,
    production_padding,
    select_phase_pairs,
    selected_pairs,
    transform_sample,
)
from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy

from hdf5_dataset.tests.conftest import (
    FIXTURE_PATH,
    MEASUREMENTS_PATH,
    requires_cuda,
    requires_measurements,
    scale_relative_errors,
)

#: Repository root, the directory the production layout is rooted at as well.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules that must import with ``teb_vae`` absent, under either package name. The tuple is
#: written out rather than discovered so a module that forgot the rule fails instead of going
#: unseen. ``benchmark_causal_torch`` is deliberately absent: it is a Run-button script, launched
#: with no ``__package__``, and is not on the pipeline's import path.
IMPORT_GUARDED_MODULES = ("causal_scattering", "causal_scattering_torch")

#: Stored decimated length of a segment, and the layout the plan must produce.
LEN_SEQUENCE = N_RAW // DECIMATION
EXPECTED_CAUSAL_WIDTHS = {"fhr_st": 36, "fhr_ph": 66, "up_st": 36, "up_ph": 15}

#: Agreement required of the torch chain against the numpy reference, per block, scale-relatively.
#:
#: The float32 bound is a **measured** ceiling, not an aspiration: the worst case over the eight
#: fixture segments is $E_\infty = 1.11\times10^{-6}$ and $E_2 = 1.39\times10^{-6}$, which is
#: single-precision round-off accumulated through a modulus, an ``angle``/``polar`` round trip and
#: three transforms. Both metrics get the same bound because they are normalised by different
#: denominators and $E_2$ is not the smaller of the two here; see
#: :func:`~hdf5_dataset.tests.conftest.scale_relative_errors`.
GATE_FLOAT64 = {"e_inf": 1e-10, "e_2": 1e-12}
GATE_FLOAT32 = {"e_inf": 1e-5, "e_2": 1e-5}

#: Segments the gate runs over. The numpy reference costs $\approx 1.5$ s per segment, so this is
#: the smallest number that still crosses several GUIDs.
GATE_SEGMENTS = 4


# =================================================================================================
# The fixture
# =================================================================================================
def test_the_committed_fixture_is_real_signal(raw_segments: Dict[str, np.ndarray]) -> None:
    """Eight finite, non-constant segments at the production length -- and never a skip.

    This is the guard that the data-dependent half of the suite cannot silently evaporate: the
    fixture is tracked, so this test runs everywhere, and ``raw_segments`` raises rather than skips
    if it ever stops being tracked.
    """
    assert FIXTURE_PATH.exists()
    for name, signal in raw_segments.items():
        assert signal.shape == (8, N_RAW) and signal.dtype == np.float32, name
        assert np.isfinite(signal).all(), name
        assert (signal.std(axis=1) > 1.0).all(), f"{name} contains a flat segment"


# =================================================================================================
# The production geometry
# =================================================================================================
def test_the_geometry_is_pinned_by_value() -> None:
    r"""Every number the causal bank is matched against, asserted here and nowhere else.

    The causal filters take their $\xi$ and $\sigma$ from this bank, and the phase blocks take
    their channel order from this pair rule, so a change to either silently redefines what a stored
    channel means. The values are pinned literally rather than re-derived, because deriving them
    from the same constants under test would be circular.
    """
    assert (FS, J, Q, T, N_RAW, DECIMATION) == (4.0, 11, 4, 16, 5280, 16)
    assert production_padding() == (1456, 1456, 8192)

    bank = build_filter_bank()
    assert bank.psi.shape == (42, 8192)
    assert bank.phi.shape == (8192,)
    assert bank.xi.shape == bank.sigma.shape == (42,)

    # Descending centre frequency is the stored channel order: fhr_st channel c is filter c - 1.
    assert np.all(np.diff(bank.xi) < 0)
    assert bank.hz[0] == pytest.approx(1.4915395232983564, rel=1e-12)
    assert bank.hz[-1] == pytest.approx(0.0005149793512363386, rel=1e-12)
    assert bank.sigma[0] == pytest.approx(0.038709062824195895, rel=1e-12)
    assert bank.sigma[-1] == pytest.approx(4.8828125e-05, rel=1e-12)

    # Taps are signed and in seconds, centred on 0 with the past negative, at the 4 Hz raw rate.
    assert bank.taps[0] == 0.0 and bank.taps[1] == pytest.approx(0.25)
    assert bank.taps.min() == pytest.approx(-1023.75) and bank.taps.max() == pytest.approx(1024.0)

    # normalize='l1' makes the low-pass unit-DC-gain, which is what lets S_0 amplitudes from the
    # two banks be compared directly.
    assert abs(float(np.abs(bank.phi[0])) - 1.0) < 1e-12


def test_the_phase_selections_are_pinned_by_value() -> None:
    """The two stored selections: their widths, their band edges and their endpoint pairs.

    ``i`` indexes the **lower** frequency, and centre frequency descends with filter index, so
    ``i > j`` throughout is the correct ordering rather than a transposition.
    """
    assert TARGET_PHASE_BAND_HZ == (0.008, 1.00)
    assert SOURCE_PHASE_BAND_HZ == (0.008, 0.05)
    assert PHASE_K_STEPS == (4, 6, 8) and PHASE_REL_TOL == 0.05

    bank = build_filter_bank()
    target = select_phase_pairs(bank, *TARGET_PHASE_BAND_HZ)
    source = select_phase_pairs(bank, *SOURCE_PHASE_BAND_HZ)

    assert len(target) == 66 and target[0] == (7, 3) and target[-1] == (30, 26)
    assert len(source) == 15 and source[0] == (24, 20) and source[-1] == (30, 26)
    for pairs in (target, source):
        assert all(bank.hz[i] <= bank.hz[j] for i, j in pairs)

    # selected_pairs is the array-shaped wrapper the comparison code uses; same rule, same order.
    assert np.array_equal(selected_pairs(TARGET_PHASE_BAND_HZ, bank), np.asarray(target))
    assert np.array_equal(selected_pairs(SOURCE_PHASE_BAND_HZ, bank), np.asarray(source))


def test_the_causal_bank_is_matched_to_the_absorbed_bank(
    bank: FilterBank, causal_bank: CausalBank
) -> None:
    r"""The causal filters take their $\xi$ and $\sigma$ from the bank rebuilt here.

    The absorbed geometry is not decoration: it is the reference the causal bank is matched
    against, one filter at a time. This is what makes that wiring a checked fact rather than an
    assumption -- the existing filter-design suite builds its causal bank from the original module
    and so cannot see it.
    """
    assert causal_bank.n_filters == bank.n_filters == 42
    assert causal_bank.n_taps == CAUSAL_KERNEL_TAPS
    assert np.array_equal(causal_bank.xi, bank.xi)
    assert np.array_equal(causal_bank.sigma, bank.sigma)
    # b = sigma * sqrt(ln 2) / sqrt(2^(1/n) - 1): the half-power bandwidth match, not a fit.
    assert np.allclose(causal_bank.b, gammatone_rate(bank.sigma), rtol=0, atol=0)


def test_the_absorbed_geometry_equals_the_probe_module() -> None:
    """Array-equal against the module the geometry was absorbed from, while that module exists.

    Skipped rather than removed when ``teb_vae`` is unavailable -- on the production box it always
    is -- which is why the test above pins the values independently. This one catches a divergence
    on the dev box, where both definitions are reachable at once.
    """
    probe = pytest.importorskip(
        "teb_vae.lag_attn.eval.representation_capacity_probe",
        reason="teb_vae is not on the import path here; the value pins above still apply",
    )
    reach = pytest.importorskip("teb_vae.lag_attn.channel_reach")

    absorbed, original = build_filter_bank(), probe.build_filter_bank()
    for field in ("psi", "phi", "xi", "sigma", "taps"):
        assert np.array_equal(getattr(absorbed, field), getattr(original, field)), field
    assert (FS, J, Q, T, N_RAW, DECIMATION) == (
        probe.FS, probe.J, probe.Q, probe.T, probe.N_RAW, probe.DECIMATION
    )
    assert TARGET_PHASE_BAND_HZ == reach.TARGET_PHASE_BAND_HZ
    assert SOURCE_PHASE_BAND_HZ == reach.SOURCE_PHASE_BAND_HZ

    for band in (TARGET_PHASE_BAND_HZ, SOURCE_PHASE_BAND_HZ):
        assert select_phase_pairs(absorbed, *band) == probe.select_phase_pairs(
            original, band[0], band[1], PHASE_K_STEPS, PHASE_REL_TOL
        )


# =================================================================================================
# The channel plan and the drop rule
# =================================================================================================
def test_the_plan_composes_warm_up_and_delay_along_the_cascade(
    causal_bank: CausalBank,
    phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, CausalChannelPlan],
) -> None:
    r"""Supports add through a cascade; a phase pair takes the **max** of its two legs.

    The composition is the whole content of the plan, so it is asserted against the rule written
    out here rather than against another call of the same function. The max on a phase pair is the
    part worth an assertion of its own: the product is formed pointwise from two responses at the
    same $t$, so summing the two legs would overstate every phase channel -- and would still look
    plausible, being merely larger.
    """
    support = np.array(
        [causal_support_samples(causal_bank.psi[k]) for k in range(causal_bank.n_filters)],
        dtype=float,
    )
    phi_support = float(causal_support_samples(causal_bank.phi))
    delay, phi_delay = causal_bank.group_delay_s, causal_bank.phi_group_delay_s

    scattering_w = np.ceil(np.concatenate([[phi_support], support + phi_support]) / DECIMATION)
    scattering_d = np.concatenate([[phi_delay], delay + phi_delay])
    for name in ("fhr_st", "up_st"):
        kept = channel_plan[name].kept
        assert np.array_equal(channel_plan[name].warmup_steps, scattering_w[kept].astype(np.int32))
        assert np.allclose(channel_plan[name].delay_s, scattering_d[kept], rtol=0, atol=0)

    for name, pairs in (("fhr_ph", phase_pairs["fhr_ph"]), ("up_ph", phase_pairs["up_ph"])):
        expected_w = np.ceil(
            (np.maximum(support[pairs[:, 0]], support[pairs[:, 1]]) + phi_support) / DECIMATION
        )
        expected_d = np.maximum(delay[pairs[:, 0]], delay[pairs[:, 1]]) + phi_delay
        kept = channel_plan[name].kept
        assert np.array_equal(channel_plan[name].warmup_steps, expected_w[kept].astype(np.int32))
        assert np.allclose(channel_plan[name].delay_s, expected_d[kept], rtol=0, atol=0)


def test_the_plan_uses_the_published_warm_up_quantile() -> None:
    """The quantile is defined once and used as the measurement's own default.

    It is not a knob. Every published causal figure, the per-channel CSV and the stored
    ``causal_warmup_steps`` have to mean the same thing by "warm-up", and $q = 0.99$ would
    lengthen every one of them by $\\approx 18\\%$.
    """
    assert CAUSAL_WARMUP_QUANTILE == 0.95
    signature = inspect.signature(causal_support_samples)
    assert signature.parameters["quantile"].default == CAUSAL_WARMUP_QUANTILE


def test_the_plan_is_in_untrimmed_decimated_steps(
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    r"""Steps, ceiling-rounded, in the storage geometry -- not seconds, not trimmed.

    A step that is $40\%$ pad is not $40\%$ valid, so the rounding is a ceiling. Storing the
    vector trimmed would make it silently wrong for any consumer reading the file at a different
    trim, so it is untrimmed and the loader rebases it.
    """
    for name, plan in channel_plan.items():
        assert plan.warmup_steps.dtype == np.int32, name
        assert plan.warmup_steps.min() >= 1, name
        assert plan.warmup_steps.max() <= LEN_SEQUENCE, name
        assert plan.n_channels == plan.kept.size == plan.warmup_steps.size == plan.delay_s.size
    # phi alone: 80 raw samples of warm-up is 5 steps, and S_0 is nothing but phi.
    assert int(channel_plan["fhr_st"].warmup_steps[0]) == 5


def test_the_drop_rule_removes_exactly_the_never_valid_channels(
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    r"""Seven channels per scattering block, both phase blocks untouched, $c_y = 102$, $c_u = 51$.

    **Two index spaces, off by one.** A scattering block stores $S_0$ at channel $0$, so channel
    $c$ is filter $c - 1$: the dropped *channels* are $36 \ldots 42$ and the dropped *filters* are
    $35 \ldots 41$. An assertion that compared the phase selections against $36 \ldots 42$ would
    be testing the wrong set and would pass anyway.
    """
    for name, expected in EXPECTED_CAUSAL_WIDTHS.items():
        assert channel_plan[name].n_channels == expected, name

    for name in ("fhr_st", "up_st"):
        dropped = sorted(set(range(43)) - set(channel_plan[name].kept.tolist()))
        assert dropped == list(range(36, 43)), name
        assert int(channel_plan[name].warmup_steps.max()) == 293, name

    assert channel_plan["fhr_st"].n_channels + channel_plan["fhr_ph"].n_channels == 102
    assert channel_plan["up_st"].n_channels + channel_plan["up_ph"].n_channels == 51


def test_no_selected_phase_pair_uses_a_dropped_filter(
    phase_pairs: Dict[str, np.ndarray], channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    r"""What makes the drop a clean channel-axis operation rather than a re-selection.

    Both phase selections are band-limited at $0.008$ Hz, which excludes the slowest filters
    entirely: the highest filter index either one uses is $30$, against dropped filters
    $35 \ldots 41$. Compared as **filter** indices on both sides, which is the comparison that can
    actually fail.
    """
    dropped_filters = {
        channel - 1 for channel in set(range(43)) - set(channel_plan["fhr_st"].kept.tolist())
    }
    assert dropped_filters == set(range(35, 42))

    used = set(phase_pairs["fhr_ph"].ravel().tolist()) | set(phase_pairs["up_ph"].ravel().tolist())
    assert max(used) == 30
    assert not (used & dropped_filters)


def test_the_figures_usability_rule_stays_inside_the_drop_rule(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    r"""Two rules about warm-up live in this package, and they answer different questions.

    The plan drops a channel that is **never valid** ($W > 330$); the figure module shades one
    whose warm-up leaves too little segment to **estimate a lag from**
    ($W \ge 330 - 32$). Both land on 36 surviving scattering channels today, and only because the
    kept warm-ups stop at $293$ -- a gap of five steps from the figures' threshold. Two rules that
    agree by five steps will diverge silently, so what is pinned is the relationship (every
    measurable channel is one the dataset stores) *and* the count they currently share, which
    fails if either moves.
    """
    figures = pytest.importorskip("hdf5_dataset.causal_scattering_figures")

    # The full-width warm-up, before the drop: what the figures see, since both arms there are
    # computed from raw signals rather than read from a stored block.
    undropped = build_channel_plan(
        causal_bank, phase_pairs["fhr_ph"], phase_pairs["up_ph"], sequence_length=10 ** 6
    )
    assert figures.SEQUENCE_LENGTH == N_RAW // DECIMATION

    for name, plan in channel_plan.items():
        full = undropped[name].warmup_steps
        measurable = np.flatnonzero(figures._usable_mask({"warmup_steps": full}))
        stored = set(plan.kept.tolist())
        assert set(measurable.tolist()) <= stored, f"{name}: a shaded-in channel is not stored"
        assert measurable.size == plan.n_channels, name

    assert int(channel_plan["fhr_st"].warmup_steps.max()) < (
        figures.SEQUENCE_LENGTH - figures.MIN_LAG_WINDOW_STEPS
    )


def test_the_plan_matches_the_published_measurements(
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    """Hand-composed pins, so the plan is checked even where the measurement CSV is not present.

    The CSV is regenerated into the git-ignored ``output/``, so the comparison against it can only
    ever be optional; these are the values that always run.
    """
    warmup = {name: plan.warmup_steps for name, plan in channel_plan.items()}
    assert (int(warmup["fhr_st"].min()), int(warmup["fhr_st"].max())) == (5, 293)
    assert (int(warmup["fhr_ph"].min()), int(warmup["fhr_ph"].max())) == (8, 149)
    assert (int(warmup["up_ph"].min()), int(warmup["up_ph"].max())) == (56, 149)

    delay = channel_plan["fhr_st"].delay_s
    assert float(delay[0]) == pytest.approx(13.3, abs=0.05)     # S_0, the low-pass alone
    assert float(delay[-1]) == pytest.approx(791.0, abs=0.05)   # the slowest surviving wavelet
    assert float(channel_plan["fhr_ph"].delay_s.min()) == pytest.approx(20.5, abs=0.05)
    assert float(channel_plan["fhr_ph"].delay_s.max()) == pytest.approx(402.2, abs=0.05)
    assert float(channel_plan["up_ph"].delay_s.min()) == pytest.approx(150.8, abs=0.05)


@requires_measurements
def test_the_plan_agrees_with_the_per_channel_csv(
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    r"""Warm-up, against the published per-channel measurements, channel for channel.

    Only warm-up. The CSV's ``causal_delay_s`` column carries the **reach-comparable** composition
    -- $S_1$ from its wavelet alone, without $\phi$ -- because there it is only ever compared
    against the two-sided reach vector, which omits the low-pass the same way. The plan's delay is
    the *complete* one, so on the scattering blocks the two differ by exactly the low-pass delay of
    $13.3$ s, by design. Asserting them equal would force the plan to understate every scattering
    channel; the phase blocks, where the two compositions coincide, are checked.
    """
    rows: Dict[str, List[Dict[str, str]]] = {}
    with MEASUREMENTS_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["block"], []).append(row)

    for name, plan in channel_plan.items():
        measured = np.array([int(row["causal_warmup_steps"]) for row in rows[name]])
        assert np.array_equal(plan.warmup_steps, measured[plan.kept])

    for name in ("fhr_ph", "up_ph"):
        measured = np.array([float(row["causal_delay_s"]) for row in rows[name]])
        assert np.allclose(channel_plan[name].delay_s, measured[channel_plan[name].kept], atol=1e-3)

    # The scattering blocks: equal once the low-pass delay the CSV column omits is added back.
    for name in ("fhr_st", "up_st"):
        measured = np.array([float(row["causal_delay_s"]) for row in rows[name]])[1:]
        composed = channel_plan[name].delay_s[1:]
        assert np.allclose(composed - measured[channel_plan[name].kept[1:] - 1],
                           channel_plan[name].delay_s[0], atol=1e-3)


#: The dataset reference's schema table, which is the document consumers read instead of the code.
_REFERENCE_DOC = Path(__file__).resolve().parents[1] / "dataset_explained_research.md"


def _reference_schema_widths() -> Dict[str, Any]:
    r"""Parse the causal column of the reference's schema table.

    Args:
        None.

    Returns:
        ``{field: channels}`` for every coefficient block the table gives a causal shape for, with
        ``None`` where it says the block is absent.
    """
    rows: Dict[str, Any] = {}
    in_table = False
    for line in _REFERENCE_DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Field | dtype | Two-sided | Causal |"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 4 or cells[1] in {"---", "dtype"}:
                continue
            field, causal = cells[0].strip("`"), cells[3]
            if causal == "absent":
                rows[field] = None
            elif causal.startswith("$(") and "," in causal:
                rows[field] = int(causal[2:].split(",")[0])
    return rows


def test_the_reference_documents_the_widths_the_plan_produces(
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    """The prose a consumer reads must not drift from the code that writes the file.

    The reference is where a model author looks up $c_y$ and $c_u$ before configuring an encoder,
    so a stale number there is a shape error somewhere far away. Parsed rather than eyeballed.
    """
    documented = _reference_schema_widths()
    assert documented, "the schema table was not found in the reference"

    for name, plan in channel_plan.items():
        assert documented[name] == plan.n_channels, name
    # The one block the causal variant does not produce at all.
    assert documented["fhr_up_ph"] is None


def test_the_two_phase_selectors_agree_in_order(
    pipeline: Any, phase_pairs: Dict[str, np.ndarray]
) -> None:
    """The pipeline's selector and this module's rebuild return one list, element for element.

    Two independent implementations of one selection rule exist -- a torch mask over the
    transform's pair axis in the writer, a numpy rule over the rebuilt bank here -- and channel $c$
    of the data means channel $c$ of the ``sel_*`` provenance and of the warm-up vector only
    because they agree. The writer takes its pairs from the selection it writes the provenance
    from, so they cannot disagree *there*; this is what stops the two definitions drifting apart
    everywhere else.
    """
    masks = pipeline.compute_scattering_masks(N_RAW, scattering_T=DECIMATION,
                                              device=torch.device("cpu"))
    for name, key in (("fhr_ph", "fhr_ph_selection"), ("up_ph", "up_ph_selection")):
        selection = masks[key]
        assert np.array_equal(phase_pairs[name][:, 0], np.asarray(selection.i, dtype=int)), name
        assert np.array_equal(phase_pairs[name][:, 1], np.asarray(selection.j, dtype=int)), name


# =================================================================================================
# The torch bank
# =================================================================================================
@pytest.fixture(scope="module")
def torch_bank(causal_bank: CausalBank) -> CausalTorchBank:
    """The build-precision bank, on the CPU so the suite runs anywhere."""
    return CausalTorchBank(causal_bank, "cpu")


@pytest.fixture(scope="module")
def torch_bank_f64(causal_bank: CausalBank) -> CausalTorchBank:
    """The double-precision bank the numerical gate is run in."""
    return CausalTorchBank(causal_bank, "cpu", dtype=torch.complex128)


def test_the_fft_length_is_exact_for_the_retained_slice(torch_bank: CausalTorchBank) -> None:
    r"""$2^{16}$, and the arithmetic that makes it exact rather than merely adequate.

    Circular convolution at length $F$ folds the linear result's tail onto its head: output $n$
    picks up the linear result at $n + F$ for every $n < 2H + N - F$. The retained slice starts at
    $H$, so the requirement is $F \ge H + N$ -- the padded length, not the linear length. The numpy
    reference sizes its transform at $2^{17}$, which is exact everywhere rather than only where it
    is read; halving it halves every stage's time and memory and changes no retained sample, which
    the gate below measures.
    """
    history, n_signal = torch_bank.history, torch_bank.n_signal
    assert (history, n_signal) == (32767, 5280)
    assert torch_bank.fft_length == 1 << 16
    aliased_below = 2 * history + n_signal - torch_bank.fft_length
    assert aliased_below <= history, "wraparound would reach into the retained slice"
    assert torch_bank.fft_length >= history + n_signal


def test_the_device_is_explicit_and_the_dtype_is_checked(causal_bank: CausalBank) -> None:
    """No implicit ``cuda``: a build is a multi-hour commitment to one GPU of eight.

    A default would pick device 0 whatever the operator typed, and the mistake would only surface
    as an out-of-memory error hours later on somebody else's job.
    """
    assert inspect.signature(CausalTorchBank.__init__).parameters["device"].default is (
        inspect.Parameter.empty
    )
    with pytest.raises(TypeError):
        CausalTorchBank(causal_bank)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="complex"):
        CausalTorchBank(causal_bank, "cpu", dtype=torch.float32)


def test_the_torch_module_imports_no_private_filter_code() -> None:
    """It reimplements the convolution, never the filter design.

    Consuming only the public :func:`build_causal_bank` output is what makes one definition of the
    filter bank -- and what makes the gate below a measurement of a convolution rather than of two
    copies of the same formula agreeing with each other.
    """
    source = (_REPO_ROOT / "hdf5_dataset" / "causal_scattering_torch.py").read_text(
        encoding="utf-8"
    )
    imported = source.split("from .causal_scattering import (", 1)[1].split(")", 1)[0]
    names = [name.strip().rstrip(",") for name in imported.split() if name.strip()]
    assert names and not [name for name in names if name.startswith("_")]


def test_the_cached_spectra_are_the_numpy_transform_of_the_numpy_kernels(
    causal_bank: CausalBank, torch_bank: CausalTorchBank
) -> None:
    """Built once per bank, reused across every batch, and equal to what numpy would compute."""
    length = torch_bank.fft_length
    expected_psi = np.fft.fft(causal_bank.psi, n=length, axis=-1)
    expected_phi = np.fft.fft(causal_bank.phi, n=length)

    psi = torch_bank.psi_spectra.numpy()
    phi = torch_bank.phi_spectrum.numpy()
    assert psi.dtype == np.complex64 and phi.dtype == np.complex64
    assert np.abs(psi - expected_psi).max() / np.abs(expected_psi).max() < 1e-6
    assert np.abs(phi - expected_phi).max() / np.abs(expected_phi).max() < 1e-6

    # (K + 1) spectra of fft_length complex64 elements: the fixed cost of a build.
    assert torch_bank.spectra_bytes == (causal_bank.n_filters + 1) * length * 8
    pointer = torch_bank.psi_spectra.data_ptr()
    torch_bank.scattering_block(torch.zeros((1, N_RAW)))
    assert torch_bank.psi_spectra.data_ptr() == pointer, "the cache was rebuilt for a batch"


def test_one_signal_transform_per_batch_not_one_per_filter(
    torch_bank: CausalTorchBank, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of caching the spectra: 42 filters cost one forward transform, not 42."""
    calls = {"fft": 0}
    real_fft = torch.fft.fft

    def counting_fft(*args, **kwargs):
        """Count forward complex transforms without changing what they return."""
        calls["fft"] += 1
        return real_fft(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "fft", counting_fft)
    torch_bank.wavelet_responses(torch.zeros((3, N_RAW)))
    assert calls["fft"] == 1


def test_reflection_padding_is_refused_by_name(torch_bank: CausalTorchBank) -> None:
    """The one pad mode that would silently undo causality, and the message that says so."""
    with pytest.raises(ValueError, match="forward in time"):
        torch_bank.prepend_history(torch.zeros((1, N_RAW)), pad="reflect")
    with pytest.raises(ValueError, match="unknown pad mode"):
        torch_bank.prepend_history(torch.zeros((1, N_RAW)), pad="wrap")
    for pad in ("edge", "zero"):
        assert torch_bank.prepend_history(torch.zeros((1, N_RAW)), pad=pad).shape[-1] == (
            torch_bank.history + N_RAW
        )


def test_the_convolutions_match_the_numpy_reference(
    causal_bank: CausalBank, torch_bank_f64: CausalTorchBank, raw_segments: Dict[str, np.ndarray]
) -> None:
    """``causal_convolve`` and ``causal_smooth``, one signal, both pad modes.

    Checked before the blocks are, so a failure in the chain can be attributed to the convolution
    or exonerated of it.
    """
    signal = raw_segments["fhr"][0].astype(np.float64)
    for pad in ("edge", "zero"):
        expected = causal_convolve(signal, causal_bank.psi, pad=pad)
        produced = torch_bank_f64.wavelet_responses(
            torch.from_numpy(signal[None, :]), pad=pad
        ).numpy()[0]
        for part in (np.real, np.imag):
            e_inf, _ = scale_relative_errors(part(produced), part(expected))
            assert e_inf <= GATE_FLOAT64["e_inf"], f"psi {pad}"

        expected_phi = causal_smooth(signal[None, :], causal_bank.phi, pad=pad).real
        produced_phi = torch_bank_f64.smooth_real(
            torch.from_numpy(signal[None, :]), pad=pad
        ).numpy()
        e_inf, _ = scale_relative_errors(produced_phi, expected_phi)
        assert e_inf <= GATE_FLOAT64["e_inf"], f"phi {pad}"


def test_the_blocks_have_the_undropped_shapes(
    torch_bank: CausalTorchBank,
    phase_pairs: Dict[str, np.ndarray],
    raw_segments: Dict[str, np.ndarray],
) -> None:
    """43/66/43/15 before the drop, no ``fhr_up_ph``, and the pair array is the caller's.

    Passing a truncated pair list through is what proves the width follows the selection rather
    than a constant -- which is how the writer supplies pairs from the selection it also writes the
    provenance from.
    """
    fhr = torch.from_numpy(raw_segments["fhr"][:2])
    up = torch.from_numpy(raw_segments["up"][:2])
    blocks = torch_bank.transform_batch(fhr, up, phase_pairs["fhr_ph"], phase_pairs["up_ph"])

    assert set(blocks) == {"fhr_st", "fhr_ph", "up_st", "up_ph"}
    assert blocks["fhr_st"].shape == (2, 43, LEN_SEQUENCE)
    assert blocks["up_st"].shape == (2, 43, LEN_SEQUENCE)
    assert blocks["fhr_ph"].shape == (2, 66, LEN_SEQUENCE)
    assert blocks["up_ph"].shape == (2, 15, LEN_SEQUENCE)

    fewer = torch_bank.transform_batch(fhr, up, phase_pairs["fhr_ph"][:5], phase_pairs["up_ph"])
    assert fewer["fhr_ph"].shape == (2, 5, LEN_SEQUENCE)
    assert torch.equal(fewer["fhr_ph"], blocks["fhr_ph"][:, :5, :])


def test_a_batch_of_the_wrong_length_is_refused(torch_bank: CausalTorchBank) -> None:
    """The transform length is sized from the segment length, so a mismatch is a rebuild, not a crop."""
    with pytest.raises(ValueError, match="rebuild it"):
        torch_bank.transform_batch(
            torch.zeros((1, N_RAW // 2)), torch.zeros((1, N_RAW // 2)),
            np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int),
        )


# =================================================================================================
# The numerical gate
# =================================================================================================
@pytest.fixture(scope="module")
def numpy_reference(
    causal_bank: CausalBank, raw_segments: Dict[str, np.ndarray]
) -> List[Dict[str, np.ndarray]]:
    """The validated numpy chain on the first :data:`GATE_SEGMENTS` fixture segments.

    Module-scoped because it costs $\\approx 1.5$ s per segment and four tests read it.
    """
    return [
        transform_sample(
            raw_segments["fhr"][index].astype(np.float64),
            raw_segments["up"][index].astype(np.float64),
            causal_bank,
        )
        for index in range(GATE_SEGMENTS)
    ]


def _gate(
    torch_bank: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    reference: List[Dict[str, np.ndarray]],
    pairs: Dict[str, np.ndarray],
    dtype: type,
) -> Dict[str, Dict[str, float]]:
    """Worst-case scale-relative errors per block, over every gated segment.

    Args:
        torch_bank: The realised bank.
        raw_segments: The fixture signals.
        reference: The numpy chain's output per segment.
        pairs: The phase selections.
        dtype: numpy dtype the signals are fed in as.

    Returns:
        ``{block: {'e_inf': ..., 'e_2': ...}}``.
    """
    produced = transform_batch_numpy(
        torch_bank,
        raw_segments["fhr"][:GATE_SEGMENTS].astype(dtype),
        raw_segments["up"][:GATE_SEGMENTS].astype(dtype),
        pairs["fhr_ph"],
        pairs["up_ph"],
    )
    worst: Dict[str, Dict[str, float]] = {}
    for index, expected in enumerate(reference):
        for name, block in expected.items():
            e_inf, e_2 = scale_relative_errors(produced[name][index], block)
            record = worst.setdefault(name, {"e_inf": 0.0, "e_2": 0.0})
            record["e_inf"] = max(record["e_inf"], e_inf)
            record["e_2"] = max(record["e_2"], e_2)
    return worst


def test_the_torch_chain_reproduces_numpy_in_float64(
    torch_bank_f64: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """The gate that makes everything downstream interpretable.

    At matched precision the two paths differ only in transform length and in doing the work
    batched, so the residual is round-off: measured at $5\\times10^{-15}$, five orders inside the
    bound.
    """
    worst = _gate(torch_bank_f64, raw_segments, numpy_reference, phase_pairs, np.float64)
    for name, errors in worst.items():
        assert errors["e_inf"] <= GATE_FLOAT64["e_inf"], f"{name} {errors}"
        assert errors["e_2"] <= GATE_FLOAT64["e_2"], f"{name} {errors}"


def test_the_torch_chain_reproduces_numpy_in_float32(
    torch_bank: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """The precision the dataset is actually built in, against the float64 reference."""
    worst = _gate(torch_bank, raw_segments, numpy_reference, phase_pairs, np.float32)
    for name, errors in worst.items():
        assert errors["e_inf"] <= GATE_FLOAT32["e_inf"], f"{name} {errors}"
        assert errors["e_2"] <= GATE_FLOAT32["e_2"], f"{name} {errors}"


def test_a_pointwise_relative_error_would_be_unbounded_on_this_data(
    numpy_reference: List[Dict[str, np.ndarray]]
) -> None:
    r"""Why the gate is scale-relative, asserted as a property of the data rather than argued.

    The phase blocks are signed and cross zero constantly, so they really do contain coefficients
    within $10^{-6}$ of their own block maximum. A pointwise ratio there divides a round-off
    difference by a number that is numerically zero, and reports an enormous error for a difference
    of no consequence.
    """
    for name in ("fhr_ph", "up_ph"):
        block = np.abs(numpy_reference[0][name])
        assert (block < 1e-6 * block.max()).any(), name


def test_the_gate_fails_on_a_perturbed_kernel(
    causal_bank: CausalBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """The control. A gate that cannot fail proves nothing, so one kernel is deliberately moved.

    The perturbation is small -- a part in a thousand on one of 42 filters -- and it must still
    break **both** metrics on the blocks that read that filter.
    """
    perturbed = dataclasses.replace(causal_bank, psi=causal_bank.psi.copy())
    perturbed.psi[10] *= 1.001
    bank = CausalTorchBank(perturbed, "cpu", dtype=torch.complex128)

    worst = _gate(bank, raw_segments, numpy_reference, phase_pairs, np.float64)
    assert worst["fhr_st"]["e_inf"] > GATE_FLOAT64["e_inf"]
    assert worst["fhr_st"]["e_2"] > GATE_FLOAT64["e_2"]
    assert worst["fhr_ph"]["e_inf"] > GATE_FLOAT64["e_inf"]
    assert worst["fhr_ph"]["e_2"] > GATE_FLOAT64["e_2"]


@requires_cuda
def test_the_gate_holds_on_the_device_the_build_runs_on(
    causal_bank: CausalBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """cuFFT is not MKL, and the dataset is built on a GPU.

    Measured on the dev GPU the agreement is about twice the CPU's -- $E_\\infty$ and $E_2$ both
    near $2\\times10^{-6}$ against the CPU's $1.4\\times10^{-6}$ -- which is inside the bound and
    would not have been inside a bound set from the CPU figure alone. Runs wherever a device
    exists, which includes the production box.
    """
    bank = CausalTorchBank(causal_bank, "cuda:0")
    worst = _gate(bank, raw_segments, numpy_reference, phase_pairs, np.float32)
    for name, errors in worst.items():
        assert errors["e_inf"] <= GATE_FLOAT32["e_inf"], f"{name} {errors}"
        assert errors["e_2"] <= GATE_FLOAT32["e_2"], f"{name} {errors}"


def test_a_segment_is_transformed_the_same_alone_as_in_a_batch(
    torch_bank: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """Not academic: the writer's OOM retry recomputes a failed segment **alone**.

    If the transform were batch-sensitive, a retried segment would be stored on different terms
    than the peers it sits beside in the same file, and nothing downstream could tell.
    """
    batched = transform_batch_numpy(
        torch_bank, raw_segments["fhr"], raw_segments["up"],
        phase_pairs["fhr_ph"], phase_pairs["up_ph"],
    )
    index = 5
    alone = transform_batch_numpy(
        torch_bank,
        raw_segments["fhr"][index : index + 1], raw_segments["up"][index : index + 1],
        phase_pairs["fhr_ph"], phase_pairs["up_ph"],
    )
    for name, block in alone.items():
        e_inf, e_2 = scale_relative_errors(block[0], batched[name][index])
        assert e_inf <= GATE_FLOAT32["e_inf"] and e_2 <= GATE_FLOAT32["e_2"], name


# =================================================================================================
# The drop, applied
# =================================================================================================
def test_the_transform_gathers_the_plan_without_touching_the_values(
    torch_bank: CausalTorchBank,
    phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, CausalChannelPlan],
    raw_segments: Dict[str, np.ndarray],
) -> None:
    """36/66/36/15 out, and every kept row bit-identical to its row of the undropped output.

    The drop is a gather and nothing else: no renormalisation, no reordering, no recomputation.
    """
    fhr = torch.from_numpy(raw_segments["fhr"][:2])
    up = torch.from_numpy(raw_segments["up"][:2])
    full = torch_bank.transform_batch(fhr, up, phase_pairs["fhr_ph"], phase_pairs["up_ph"])
    dropped = torch_bank.transform_batch(
        fhr, up, phase_pairs["fhr_ph"], phase_pairs["up_ph"], plan=channel_plan
    )
    for name, expected_width in EXPECTED_CAUSAL_WIDTHS.items():
        assert dropped[name].shape == (2, expected_width, LEN_SEQUENCE), name
        kept = torch.from_numpy(channel_plan[name].kept.astype(np.int64))
        assert torch.equal(dropped[name], full[name].index_select(1, kept)), name


def test_a_plan_built_for_a_different_bank_is_refused(
    torch_bank: CausalTorchBank,
    phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, CausalChannelPlan],
    raw_segments: Dict[str, np.ndarray],
) -> None:
    """A plan indexing past the block's width is a bank disagreement, not an index error."""
    stale = dict(channel_plan)
    stale["fhr_ph"] = dataclasses.replace(
        channel_plan["fhr_ph"], kept=np.array([0, 900], dtype=np.int32)
    )
    with pytest.raises(ValueError, match="disagree about the filter bank"):
        torch_bank.transform_batch(
            torch.from_numpy(raw_segments["fhr"][:1]), torch.from_numpy(raw_segments["up"][:1]),
            phase_pairs["fhr_ph"], phase_pairs["up_ph"], plan=stale,
        )


def test_the_dropped_channels_are_the_pad_dominated_ones(
    torch_bank_f64: CausalTorchBank,
    phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, CausalChannelPlan],
    raw_segments: Dict[str, np.ndarray],
    causal_bank: CausalBank,
) -> None:
    r"""The behavioural half of the drop rule: swap the assumed history and see what moves.

    Two facts, one structural and one behavioural.

    *Structurally*, every dropped filter still carries $\ge 17\%$ of its $L^1$ mass beyond the
    stored segment, while every kept one carries $\le 10\%$ -- the support does not close inside
    the recording, which is what "never valid" means.

    *Behaviourally*, at the **last** stored step -- the step with the most history behind it --
    replacing the edge pad with a zero pad still moves the four slowest channels by more than
    $80\%$ of their own range, while every channel whose warm-up is under half the segment moves
    by less than $1\%$. Measured: $0.001$ against $0.87$.

    The transition between those two groups is **graded**, not a cliff (channels 34-36 measure
    $0.11$, $0.26$, $0.03$), and that is expected rather than a defect: $W_{0.95}$ is an energy
    quantile, not a support, so $5\%$ of a kept channel's kernel still lies beyond it. An assertion
    that kept channels agree between pads to round-off would therefore be false -- and would be
    testing the quantile's tail rather than the drop rule.
    """
    fhr = torch.from_numpy(raw_segments["fhr"][:1].astype(np.float64))
    up = torch.from_numpy(raw_segments["up"][:1].astype(np.float64))
    kwargs = dict(target_pairs=phase_pairs["fhr_ph"], source_pairs=phase_pairs["up_ph"])
    edge = torch_bank_f64.transform_batch(fhr, up, pad="edge", **kwargs)["fhr_st"][0].numpy()
    zero = torch_bank_f64.transform_batch(fhr, up, pad="zero", **kwargs)["fhr_st"][0].numpy()

    beyond_segment = np.array([
        float(np.abs(causal_bank.psi[k])[N_RAW:].sum() / np.abs(causal_bank.psi[k]).sum())
        for k in range(causal_bank.n_filters)
    ])
    kept_filters = channel_plan["fhr_st"].kept[1:] - 1
    dropped_filters = np.array(sorted(set(range(42)) - set(kept_filters.tolist())))
    assert beyond_segment[kept_filters].max() < 0.10
    assert beyond_segment[dropped_filters].min() > 0.17

    sensitivity = np.abs(edge[:, -1] - zero[:, -1]) / np.maximum(np.abs(zero).max(axis=1), 1e-30)
    warmup = np.ceil(
        np.concatenate([
            [causal_support_samples(causal_bank.phi)],
            np.array([causal_support_samples(causal_bank.psi[k])
                      for k in range(causal_bank.n_filters)]) + causal_support_samples(
                          causal_bank.phi),
        ]) / DECIMATION
    )
    settled = warmup <= LEN_SEQUENCE // 2
    never_closing = warmup > 1.5 * LEN_SEQUENCE
    assert sensitivity[settled].max() < 0.01
    assert sensitivity[never_closing].min() > 0.80
    assert not settled[never_closing].any()


# =================================================================================================
# The cost measurement
# =================================================================================================
def test_the_cost_measurement_reports_derivable_figures(
    causal_bank: CausalBank, tmp_path: Path
) -> None:
    """Run the benchmark small, on the CPU, and check what can be checked rather than timed.

    A wall-clock number cannot be asserted -- it is the thing being measured, and it differs by an
    order of magnitude between the two boxes this repository runs on. What can be asserted is that
    the fixed cost is the exact byte count the cache implies, that every stage produced a finite
    positive time, that the widths are the plan's, and that the artefact round-trips as JSON. The
    artefact matters because a figure that exists only in a terminal scrollback cannot be compared
    against the next run's.
    """
    from hdf5_dataset import benchmark_causal_torch

    output = tmp_path / "torch_cost.json"
    assert benchmark_causal_torch.main(
        device="cpu", batch_sizes=(2,), repeats=1, output_path=str(output)
    ) == 0

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["fft_length"] == 1 << 16
    assert record["spectra_bytes"] == (causal_bank.n_filters + 1) * (1 << 16) * 8
    assert record["widths"] == EXPECTED_CAUSAL_WIDTHS

    stages = record["batches"][0]["stages"]
    assert set(stages) >= {"wavelet_responses", "scattering_block", "whole_chain"}
    for name, values in stages.items():
        assert np.isfinite(values["ms_per_segment"]) and values["ms_per_segment"] > 0.0, name
        # No allocator watermark exists off CUDA, and reporting zero there would read as "free".
        assert values["peak_bytes_per_segment"] is None, name

    assert "No batch size is recommended" in benchmark_causal_torch.format_report(record)


# =================================================================================================
# The production import path
# =================================================================================================
def _run_child(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter, from a directory that is not the repository root.

    The working directory matters: a child launched from the repository root gets it on
    ``sys.path`` as ``''``, which would make the top-level ``hdf5_dataset`` name importable even in
    the aliased run that exists to prove it is not needed.

    Args:
        body: Source to execute.

    Returns:
        The completed process, with output captured.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
    )


def _blocker(names: Sequence[str]) -> str:
    """Source for an import hook that refuses *names* and anything beneath them."""
    return f"""
    import sys

    class _Blocked:
        \"\"\"Refuses the packages the production box does not have on its import path.\"\"\"

        blocked = {tuple(names)!r}

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in self.blocked:
                raise ImportError(f'{{fullname}} is not importable here')
            return None

    sys.meta_path.insert(0, _Blocked())
    """


@pytest.mark.parametrize("module", IMPORT_GUARDED_MODULES)
def test_imports_with_teb_vae_absent(module: str) -> None:
    """The dev-box layout, minus the model package: ``import hdf5_dataset.<module>`` must succeed.

    This is the half of the blocker that a plain ``pytest`` run would never catch, because
    ``teb_vae`` is importable here.
    """
    result = _run_child(f"""
    {_blocker(['teb_vae'])}
    sys.path.insert(0, {str(_REPO_ROOT)!r})
    import importlib
    module = importlib.import_module('hdf5_dataset.{module}')
    print(module.__name__)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"hdf5_dataset.{module}"


@pytest.mark.parametrize("module", IMPORT_GUARDED_MODULES)
def test_imports_under_the_production_package_name(module: str) -> None:
    """The production layout: this directory is ``Variational_AutoEncoder.seqvae_teb.hdf5_dataset``.

    The top-level ``hdf5_dataset`` name is blocked as well as ``teb_vae``, so an absolute
    ``from hdf5_dataset.X import Y`` reintroduced anywhere on this import path fails here instead
    of at the first production run. Only relative intra-package imports resolve under both names.
    """
    result = _run_child(f"""
    {_blocker(['teb_vae', 'hdf5_dataset'])}
    import importlib, types
    parent = types.ModuleType('Variational_AutoEncoder')
    parent.__path__ = []
    child = types.ModuleType('Variational_AutoEncoder.seqvae_teb')
    child.__path__ = [{str(_REPO_ROOT)!r}]
    sys.modules['Variational_AutoEncoder'] = parent
    sys.modules['Variational_AutoEncoder.seqvae_teb'] = child
    module = importlib.import_module(
        'Variational_AutoEncoder.seqvae_teb.hdf5_dataset.{module}'
    )
    print(module.__package__)
    """)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Variational_AutoEncoder.seqvae_teb.hdf5_dataset"


def test_the_import_guard_fails_on_a_blocked_import() -> None:
    """The guard's own control: with ``teb_vae`` blocked, importing it must fail.

    Without this, both tests above would pass just as well against a blocker that blocked nothing.
    """
    result = _run_child(f"""
    {_blocker(['teb_vae'])}
    sys.path.insert(0, {str(_REPO_ROOT)!r})
    import teb_vae.lag_attn.channel_reach
    """)
    assert result.returncode != 0
    assert "not importable here" in result.stderr


def test_an_absolute_intra_package_import_would_fail_under_the_production_name() -> None:
    """The second control: on the production path the top-level ``hdf5_dataset`` name does not exist.

    A ``from hdf5_dataset.X import Y`` inside this package resolves on the dev box and nowhere
    else, which is why the rule is relative intra-package imports. This asserts the property the
    rule rests on, so the guard above is known to bite rather than assumed to.
    """
    result = _run_child(f"""
    {_blocker(['teb_vae', 'hdf5_dataset'])}
    import types
    parent = types.ModuleType('Variational_AutoEncoder')
    parent.__path__ = []
    child = types.ModuleType('Variational_AutoEncoder.seqvae_teb')
    child.__path__ = [{str(_REPO_ROOT)!r}]
    sys.modules['Variational_AutoEncoder'] = parent
    sys.modules['Variational_AutoEncoder.seqvae_teb'] = child
    import hdf5_dataset.causal_scattering
    """)
    assert result.returncode != 0
    assert "not importable here" in result.stderr
