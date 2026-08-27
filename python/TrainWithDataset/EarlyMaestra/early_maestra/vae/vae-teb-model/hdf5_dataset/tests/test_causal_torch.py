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
    ALIGNMENT_DELAY_FACTOR,
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
    channel_alignment_delays,
    gammatone_rate,
    leg_alignment_shift,
    novelty_fraction,
    pair_leg_skew,
    phase_block_causal,
    production_padding,
    select_phase_pairs,
    selected_pairs,
    transform_sample,
)
from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy

from hdf5_dataset.tests.conftest import (
    FIXTURE_PATH,
    MEASUREMENTS_PATH,
    SHARD_PATH,
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

#: One stored step in seconds, and the shipped forecast horizon in stored steps. Both are
#: *model-side* geometry rather than bank geometry, which is why they are written here rather than
#: imported: the delay surface is a property of the filter bank alone, and these two numbers are
#: what a consumer of it happens to configure. They are the values every shipped config uses.
STEP_SECONDS = DECIMATION / FS
HORIZON_STEPS = 30

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


def _measurement_provenance() -> Dict[str, Any]:
    """What the measurement run recorded about itself, beside the CSV it wrote.

    Returns:
        The ``arguments`` block of ``summary.json``.

    Raises:
        AssertionError: If the CSV is present without its summary, which means the review
            directory was assembled by something other than the tool.
    """
    summary = MEASUREMENTS_PATH.parent / "summary.json"
    assert summary.exists(), (
        f"{MEASUREMENTS_PATH.name} is present but {summary.name} is not. The two are written by "
        f"one run, so a CSV without its summary is a directory nothing can date."
    )
    return json.loads(summary.read_text(encoding="utf-8"))["arguments"]


@requires_measurements
def test_the_measurement_csv_describes_this_bank_and_this_shard() -> None:
    r"""A **stale** CSV fails here; an absent one still skips.

    ``output/`` is git-ignored, so a fresh clone and the production box have never run the
    measurement tool and the comparison below can only ever be optional -- that is what
    :data:`requires_measurements` is for. What it must not do is cover the *other* case: a CSV
    left over from a run against a different shard, or at a different leg alignment, silently
    passing every comparison built on it because the comparison was skipped for a different
    reason. Absence is a skip; disagreement is a failure, and the message says which.

    The tool records both facts in ``summary.json`` beside the CSV, which is the only place they
    survive: the CSV's own rows are per channel and carry no run-level provenance at all.
    """
    arguments = _measurement_provenance()
    shard = str(arguments.get("shard", ""))
    alignment = str(arguments.get("leg_alignment", "none"))

    assert Path(shard).name == SHARD_PATH.name, (
        f"{MEASUREMENTS_PATH} was measured against {shard!r}, not {SHARD_PATH}. Every value in "
        f"it describes a different dataset's channels; re-run the comparison tool."
    )
    assert alignment in ("none", "envelope"), f"unknown leg alignment {alignment!r} in the summary"
    # The three shipped correlation columns describe whichever arm the run was made at, so a CSV
    # built aligned cannot be read as the unaligned baseline the pins below assume.
    assert alignment == "none", (
        f"{MEASUREMENTS_PATH} was measured at leg_alignment={alignment!r}. Its r_at_best_lag, "
        f"r_at_predicted_lag and r_at_zero_lag columns then describe the aligned arm, while the "
        f"envelope columns beside them describe the same one; re-run without --leg-alignment to "
        f"restore the comparison."
    )


@requires_measurements
def test_the_measurement_csv_carries_the_leg_alignment_columns() -> None:
    r"""The seven columns the alignment added, present and finite where they mean anything.

    Three describe the bank -- the intra-pair skew, the integer shift that removes it and the
    harmonic ratio the skew scales with. Four are measured against the centred block: the
    correlation at the predicted delay, and the complex coherence before $\Re\{\cdot\}$ with its
    residual rotation and that rotation's concentration across segments.

    The scattering blocks have no pairs, so their rows are ``nan`` by construction rather than
    absent -- the CSV keeps one row per stored channel of every block.
    """
    added = (
        "pair_skew_s", "leg_shift_samples", "harmonic_power",
        "r_at_predicted_lag_envelope", "coherence_abs_envelope",
        "coherence_deg_envelope", "coherence_concentration_envelope",
    )
    rows: Dict[str, List[Dict[str, str]]] = {}
    with MEASUREMENTS_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert set(added) <= set(reader.fieldnames), (
            f"missing: {sorted(set(added) - set(reader.fieldnames))}"
        )
        for row in reader:
            rows.setdefault(row["block"], []).append(row)

    for name in ("fhr_ph", "up_ph"):
        for column in added:
            values = np.array([float(row[column]) for row in rows[name]])
            assert np.isfinite(values).all(), f"{name}.{column}"
        skew = np.array([float(row["pair_skew_s"]) for row in rows[name]])
        assert (skew >= 0.0).all(), name
        # The alignment is measured, not merely recorded: it beats the shipped arm on both blocks.
        shipped = np.array([float(row["r_at_predicted_lag"]) for row in rows[name]])
        envelope = np.array([float(row["r_at_predicted_lag_envelope"]) for row in rows[name]])
        assert float(np.median(envelope)) > float(np.median(shipped)) + 0.3, name

    for name in ("fhr_st", "up_st"):
        for column in added:
            assert all(row[column] == "nan" for row in rows[name]), f"{name}.{column}"

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
# The delay surface
# =================================================================================================
# Four pure functions over the bank, the pair lists and a stored delay vector, and one assertion
# about what they cost. Nothing here changes a stored value; what it does is make the intra-pair
# skew, the shift that removes it and the per-channel alignment rule measurable rather than
# implicit, so that the operator built on them in the transform can be checked against numbers
# that were pinned before it existed.
def test_the_pair_skew_reproduces_the_harmonic_ratio_identity(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray]
) -> None:
    r"""$\Delta_{ij} = \tau_i - \tau_j = \tau_i(1 - 1/p_{ij})$, on both stored selections.

    The identity is the whole reason the skew is predictable rather than a table: on a constant-$Q$
    ladder $b \propto \xi$, so $\tau_i/\tau_j = p_{ij}$ exactly and the skew is a fixed fraction of
    the slow leg's own delay -- $\tfrac12$, $0.646$ and $\tfrac34$ for the three stored families.
    Asserted against the ratio rebuilt from the **production** centre frequencies rather than from
    the causal bank's own, so the two banks agreeing is part of what is checked.
    """
    reference = build_filter_bank()
    for name, pairs in phase_pairs.items():
        skew = pair_leg_skew(causal_bank, pairs)
        power = reference.hz[pairs[:, 1]] / reference.hz[pairs[:, 0]]
        identity = causal_bank.group_delay_s[pairs[:, 0]] * (1.0 - 1.0 / power)
        assert np.allclose(skew, identity, rtol=1e-12, atol=1e-12), name
        # Column 0 indexes the lower frequency, so the slow leg is always the first one.
        assert (skew >= 0.0).all(), name
        assert skew.shape == (pairs.shape[0],), name


def test_every_source_phase_channel_is_skewed_by_most_of_a_minute(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray]
) -> None:
    r"""The number that makes the defect worth repairing, on the block that carries it worst.

    ``up_ph`` is the block whose entire purpose is to carry contraction morphology into the lag
    attention, and the attention searches $360$ s. **Every one** of its fifteen channels is built
    from two legs at least $68.7$ s apart, with a median of $163.5$ s -- half the search range,
    inside a single channel, before any cross-channel effect. A floor rather than an equality, so
    a bank change that erodes the margin fails here rather than in a training curve.
    """
    source = pair_leg_skew(causal_bank, phase_pairs["up_ph"])
    assert source.min() >= 68.7
    assert float(np.median(source)) == pytest.approx(163.5, abs=0.1)
    assert source.max() == pytest.approx(291.6, abs=0.1)

    target = pair_leg_skew(causal_bank, phase_pairs["fhr_ph"])
    assert (float(target.min()), float(target.max())) == pytest.approx((3.6, 291.6), abs=0.1)
    assert float(np.median(target)) == pytest.approx(39.1, abs=0.1)


def test_a_reused_fast_leg_gets_a_different_shift_in_each_pair_that_uses_it(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray]
) -> None:
    r"""The shift is indexed by **pair**, and this is the test that says so.

    ``select_phase_pairs`` admits every $(i, j)$ meeting the band and ratio rule, so one fast
    filter serves several slow partners at several harmonic ratios and needs a different
    $s_{ij}$ in each. Measured on the shipped bank: $22$ of the $24$ distinct ``fhr_ph`` fast legs
    are reused -- $20$ of them by three slow partners each -- and $5$ of the $7$ in ``up_ph``.

    An implementation that shifted the *response array* before the per-pair gather could satisfy
    at most one pair per reused filter and would be silently wrong for the rest, with every shape
    correct and every existing gate green. That failure is invisible to a shape test and to a
    round-trip test; it is visible here.
    """
    expected_reuse = {"fhr_ph": (24, 22, 20), "up_ph": (7, 5, 3)}
    for name, pairs in phase_pairs.items():
        shift, _ = leg_alignment_shift(causal_bank, pairs)
        by_fast_leg: Dict[int, List[int]] = {}
        for (_, fast), value in zip(pairs.tolist(), shift.tolist()):
            by_fast_leg.setdefault(int(fast), []).append(int(value))

        distinct, reused, by_three = expected_reuse[name]
        assert len(by_fast_leg) == distinct, name
        assert sum(len(v) > 1 for v in by_fast_leg.values()) == reused, name
        assert sum(len(v) == 3 for v in by_fast_leg.values()) == by_three, name
        # Every partner of a reused leg asks it for a *different* shift, which is the claim.
        for fast, shifts in by_fast_leg.items():
            assert len(set(shifts)) == len(shifts), f"{name}: fast leg {fast} -> {shifts}"


def test_the_alignment_shift_is_non_negative_and_its_phasor_is_a_rotation(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray]
) -> None:
    r"""One shift and one unit-modulus phasor per pair, the shift derived from the stored delay.

    The phasor's modulus is the thing to pin: it is the de-rotation that keeps the carrier where
    it is while the envelope moves, so a modulus that drifted from $1$ would rescale every
    coefficient of the block on top of rotating it. Derived from
    :attr:`CausalBank.group_delay_s`, not from a second evaluation of $\gamma/(2\pi b)$, so the
    bank stays the one source of the number.
    """
    for name, pairs in phase_pairs.items():
        shift, phasor = leg_alignment_shift(causal_bank, pairs)
        assert shift.shape == phasor.shape == (pairs.shape[0],), name
        assert (shift >= 0).all(), name
        assert np.abs(np.abs(phasor) - 1.0).max() < 1e-12, name
        # The shift is the skew on the raw grid, rounded: no other quantity would round to this.
        assert np.array_equal(shift, np.round(pair_leg_skew(causal_bank, pairs) * FS)), name

    # A mis-ordered pair would need the faster leg advanced, which reads its own future.
    flipped = phase_pairs["up_ph"][:, ::-1].copy()
    with pytest.raises(ValueError, match="reading its own future"):
        leg_alignment_shift(causal_bank, flipped)


def test_the_leg_alignment_costs_no_warm_up_on_any_stored_pair(
    causal_bank: CausalBank, phase_pairs: Dict[str, np.ndarray]
) -> None:
    r"""$W_j + s_{ij} \le W_i$ for all $81$ stored pairs -- the whole cost argument, asserted.

    Delaying the fast leg lengthens *its* warm-up, and the pair's warm-up is
    $\max(W_i,\ W_j + s_{ij}) + W_\phi$. The alignment is free exactly when the delayed fast leg
    still warms up no later than the slow leg it was delayed onto, which holds because
    $W \approx \rho\tau$ with $\rho \approx 1.48$ roughly constant, giving
    $W_j + s_{ij} \approx f_s(\tau_i + (\rho - 1)\tau_j) \le f_s\rho\tau_i = W_i$.

    Asymptotically true is not the same as true, so this checks every pair and reports the
    tightest slack: a bank change that erodes it says by how much rather than merely failing. The
    shipped margin is $8$ raw samples on ``fhr_ph`` and $132$ on ``up_ph`` -- thin on the target
    block, which is why the composed warm-up rule, the stored widths and the drop rule all depend
    on a test rather than on the approximation above.
    """
    support = np.array(
        [causal_support_samples(causal_bank.psi[k]) for k in range(causal_bank.n_filters)],
        dtype=np.int64,
    )
    tightest = {"fhr_ph": 8, "up_ph": 132}
    for name, pairs in phase_pairs.items():
        shift, _ = leg_alignment_shift(causal_bank, pairs)
        slack = support[pairs[:, 0]] - (support[pairs[:, 1]] + shift)
        worst = int(np.argmin(slack))
        assert slack.min() >= 0, (
            f"{name}: pair {worst} = ({int(pairs[worst, 0])}, {int(pairs[worst, 1])}) has "
            f"{int(slack[worst])} raw samples of slack -- the delayed fast leg now warms up "
            f"after the slow leg, so the composed warm-up rule and the stored widths would move"
        )
        assert int(slack.min()) == tightest[name], name

    assert phase_pairs["fhr_ph"].shape[0] + phase_pairs["up_ph"].shape[0] == 81


def test_the_channel_alignment_rounds_and_refuses_a_channel_above_the_reference(
    channel_plan: Dict[str, CausalChannelPlan]
) -> None:
    r"""Rounding, a half-step residual, zero at the reference, and a named refusal.

    The reference is filter $30$'s composed delay, $402.1604$ s -- simultaneously the maximum of
    ``fhr_ph``, the maximum of ``up_ph`` and the lower band edge both phase selections already
    use. Both streams reach it, and both carry four scattering channels above it, which cannot be
    aligned at all: a negative shift reads a channel's own future.

    The shift carries :data:`ALIGNMENT_DELAY_FACTOR`, $\kappa = 1 - 1/(2\gamma) = 0.875$, because
    ``delay_s`` reports the envelope *mean* $\tau_g$ while a channel's content sits at the energy
    centroid $\kappa\tau_g$. Only the *difference* is scaled, so the reference channel still takes
    shift $0$ and the four refusals are unchanged -- $\tau_c \le \tau_{\mathrm{ref}}$ is
    scale-invariant. The span is $85$ rather than the $97$ this pinned before the factor existed.
    """
    reference = float(channel_plan["fhr_ph"].delay_s.max())
    assert reference == pytest.approx(402.1604, abs=5e-4)
    assert float(channel_plan["up_ph"].delay_s.max()) == pytest.approx(reference, abs=1e-9)

    for stream, blocks in (("target", ("fhr_st", "fhr_ph")), ("source", ("up_st", "up_ph"))):
        delay = np.concatenate([channel_plan[name].delay_s for name in blocks])
        above = delay > reference
        assert int(above.sum()) == 4, stream

        shifts = channel_alignment_delays(delay[~above], reference, STEP_SECONDS)
        assert (shifts >= 0).all(), stream
        assert (int(shifts.min()), int(shifts.max())) == (0, 85), stream
        # Rounding, not ceiling: both directions are causally safe, so the only criterion is the
        # residual, and it is bounded by half a step rather than by a whole one.
        # Taken against the SCALED difference, because that is what the shift rounds; measured
        # against the unscaled one it would report ~50 s and say nothing about the rounding.
        residual = np.abs(
            ALIGNMENT_DELAY_FACTOR * (reference - delay[~above]) - STEP_SECONDS * shifts
        )
        assert residual.max() <= STEP_SECONDS / 2.0, stream
        assert float(residual.max()) == pytest.approx(1.99, abs=0.01), stream
        # Zero at the reference itself, which is a channel of both streams.
        assert int(shifts[np.argmin(np.abs(delay[~above] - reference))]) == 0, stream

        with pytest.raises(ValueError, match="reads the channel's own future") as error:
            channel_alignment_delays(delay, reference, STEP_SECONDS)
        message = str(error.value)
        assert f"channel {int(np.flatnonzero(above)[0])}" in message, stream
        assert "402.16" in message, stream


def test_the_novelty_fraction_is_the_published_table(
    causal_bank: CausalBank,
    phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, CausalChannelPlan],
) -> None:
    r"""How much of a target coefficient is drawn from raw samples the anchor has not seen.

    Over the full $120$ s horizon the slowest kept channel draws $2.6\%$ of its value from the
    window it is being asked to forecast, while $S_0$ draws all of it. This is not a leak -- every
    coefficient still depends on samples after the anchor -- but it means the effective forecast
    horizon is per channel, and a block score summed over both mixes two different claims.

    Both ends of the range are pinned, because a bug that collapsed the fraction to a constant
    would look plausible at either end alone.
    """
    novelty = novelty_fraction(
        causal_bank, channel_plan, phase_pairs["fhr_ph"], phase_pairs["up_ph"], HORIZON_STEPS
    )
    assert sorted(novelty) == sorted(channel_plan)
    for name, plan in channel_plan.items():
        assert novelty[name].shape == (plan.n_channels,), name
        assert (novelty[name] >= 0.0).all() and (novelty[name] <= 1.0).all(), name

    # A scattering block stores $S_0$ at channel 0, so filter $k$ is channel $k + 1$.
    scattering = novelty["fhr_st"]
    assert float(scattering[0]) == pytest.approx(1.000, abs=5e-4)
    assert float(scattering[31]) == pytest.approx(0.026, abs=5e-3)
    for filter_index, expected in ((10, 1.000), (18, 0.974), (25, 0.267), (28, 0.073)):
        assert float(scattering[filter_index + 1]) == pytest.approx(expected, abs=5e-3)
    assert np.array_equal(novelty["up_st"], scattering)

    # A phase channel takes its slow leg's value, which is the conservative one; every pair of
    # both blocks tops out at filter 30, so both minima land on the reference channel's fraction.
    for name in ("fhr_ph", "up_ph"):
        assert float(novelty[name].min()) == pytest.approx(0.026, abs=5e-3), name
        expected = scattering[phase_pairs[name][:, 0] + 1]
        assert np.array_equal(novelty[name], expected), name

    with pytest.raises(ValueError, match="horizon_steps must be positive"):
        novelty_fraction(
            causal_bank, channel_plan, phase_pairs["fhr_ph"], phase_pairs["up_ph"], 0
        )


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


@pytest.fixture(scope="module")
def numpy_reference_aligned(
    causal_bank: CausalBank,
    raw_segments: Dict[str, np.ndarray],
    phase_pairs: Dict[str, np.ndarray],
    numpy_reference: List[Dict[str, np.ndarray]],
) -> List[Dict[str, np.ndarray]]:
    """The same reference with both phase blocks rebuilt envelope-aligned.

    The **scattering** blocks are carried over from the unaligned reference deliberately rather
    than recomputed. The alignment lives entirely inside the phase product, so those two blocks
    must come out identical -- and the torch side under test computes them fresh at
    ``leg_alignment='envelope'``, so gating them against the unaligned numpy values is what proves
    the mode does not leak out of the phase path.
    """
    return [
        {
            **sample,
            "fhr_ph": phase_block_causal(
                raw_segments["fhr"][index].astype(np.float64),
                raw_segments["fhr"][index].astype(np.float64),
                phase_pairs["fhr_ph"], causal_bank, leg_alignment="envelope",
            ),
            "up_ph": phase_block_causal(
                raw_segments["up"][index].astype(np.float64),
                raw_segments["up"][index].astype(np.float64),
                phase_pairs["up_ph"], causal_bank, leg_alignment="envelope",
            ),
        }
        for index, sample in enumerate(numpy_reference)
    ]


def _gate(
    torch_bank: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    reference: List[Dict[str, np.ndarray]],
    pairs: Dict[str, np.ndarray],
    dtype: type,
    leg_alignment: str = "none",
) -> Dict[str, Dict[str, float]]:
    """Worst-case scale-relative errors per block, over every gated segment.

    Args:
        torch_bank: The realised bank.
        raw_segments: The fixture signals.
        reference: The numpy chain's output per segment.
        pairs: The phase selections.
        dtype: numpy dtype the signals are fed in as.
        leg_alignment: The mode both chains are run at; *reference* must be the numpy output at
            the same mode.

    Returns:
        ``{block: {'e_inf': ..., 'e_2': ...}}``.
    """
    produced = transform_batch_numpy(
        torch_bank,
        raw_segments["fhr"][:GATE_SEGMENTS].astype(dtype),
        raw_segments["up"][:GATE_SEGMENTS].astype(dtype),
        pairs["fhr_ph"],
        pairs["up_ph"],
        leg_alignment=leg_alignment,
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


def test_the_torch_default_leg_alignment_is_bitwise_the_unaligned_block(
    torch_bank_f64: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """Asking for nothing and asking for ``'none'`` are the same tensor, bit for bit.

    Every shard on disk was built through the no-argument call, so the default carries the whole
    weight of "adding the mode changed no stored value". Bitwise rather than to a tolerance,
    because the ``'none'`` branch must not merely round to the old result -- it must *be* it,
    having taken no gather and no multiply on the way.
    """
    signals = torch.from_numpy(raw_segments["fhr"][:2].astype(np.float64))
    implicit = torch_bank_f64.phase_block(signals, signals, phase_pairs["fhr_ph"])
    explicit = torch_bank_f64.phase_block(
        signals, signals, phase_pairs["fhr_ph"], leg_alignment="none"
    )
    aligned = torch_bank_f64.phase_block(
        signals, signals, phase_pairs["fhr_ph"], leg_alignment="envelope"
    )
    assert torch.equal(implicit, explicit)
    # Not vacuous: the aligned mode really does move the block it is asked to move.
    assert not torch.equal(implicit, aligned)

    with pytest.raises(ValueError, match="unknown leg_alignment 'delay'"):
        torch_bank_f64.phase_block(
            signals, signals, phase_pairs["fhr_ph"], leg_alignment="delay"
        )


def test_the_torch_chain_reproduces_numpy_aligned_in_float64(
    torch_bank_f64: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference_aligned: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    r"""The alignment, gated -- two independent implementations of one per-pair shift.

    This is exactly the risk the gate exists for. The two paths delay the conjugated leg through
    different primitives (``take_along_axis`` against ``torch.gather``) and de-rotate it in
    different precisions, so an off-by-one in either index arithmetic, or a sign flip in either
    phasor, breaks the agreement here rather than showing up as a shard that looks plausible.
    """
    worst = _gate(
        torch_bank_f64, raw_segments, numpy_reference_aligned, phase_pairs, np.float64,
        leg_alignment="envelope",
    )
    for name, errors in worst.items():
        assert errors["e_inf"] <= GATE_FLOAT64["e_inf"], f"{name} {errors}"
        assert errors["e_2"] <= GATE_FLOAT64["e_2"], f"{name} {errors}"


def test_the_torch_chain_reproduces_numpy_aligned_in_float32(
    torch_bank: CausalTorchBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference_aligned: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    r"""The aligned mode at build precision, against the same bound the unaligned mode meets.

    The de-rotation is where single precision could plausibly have cost something: the phasor's
    angle reaches $9.6$ turns, and evaluating $e^{\,i2\pi\xi_j s}$ in float32 would lose about four
    digits of it. It does not, because both chains take the phasor from the float64 numpy bank and
    round it once -- and that is what this bound is checking.
    """
    worst = _gate(
        torch_bank, raw_segments, numpy_reference_aligned, phase_pairs, np.float32,
        leg_alignment="envelope",
    )
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
    numpy_reference_aligned: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
) -> None:
    """The control. A gate that cannot fail proves nothing, so one kernel is deliberately moved.

    The perturbation is small -- a part in a thousand on one of 42 filters -- and it must still
    break **both** metrics on the blocks that read that filter, at **both** leg-alignment modes.
    Run aligned as well because the alignment inserts a gather and a multiply between the
    convolution and the product, and a gate that had stopped depending on the kernel there would
    otherwise pass while measuring nothing.
    """
    perturbed = dataclasses.replace(causal_bank, psi=causal_bank.psi.copy())
    perturbed.psi[10] *= 1.001
    bank = CausalTorchBank(perturbed, "cpu", dtype=torch.complex128)

    for mode, reference in (
        ("none", numpy_reference), ("envelope", numpy_reference_aligned)
    ):
        worst = _gate(
            bank, raw_segments, reference, phase_pairs, np.float64, leg_alignment=mode
        )
        assert worst["fhr_st"]["e_inf"] > GATE_FLOAT64["e_inf"], mode
        assert worst["fhr_st"]["e_2"] > GATE_FLOAT64["e_2"], mode
        assert worst["fhr_ph"]["e_inf"] > GATE_FLOAT64["e_inf"], mode
        assert worst["fhr_ph"]["e_2"] > GATE_FLOAT64["e_2"], mode


@requires_cuda
@pytest.mark.parametrize("leg_alignment", ("none", "envelope"))
def test_the_gate_holds_on_the_device_the_build_runs_on(
    causal_bank: CausalBank,
    raw_segments: Dict[str, np.ndarray],
    numpy_reference: List[Dict[str, np.ndarray]],
    numpy_reference_aligned: List[Dict[str, np.ndarray]],
    phase_pairs: Dict[str, np.ndarray],
    leg_alignment: str,
) -> None:
    """cuFFT is not MKL, and the dataset is built on a GPU.

    Measured on the dev GPU the agreement is about twice the CPU's -- $E_\\infty$ and $E_2$ both
    near $2\\times10^{-6}$ against the CPU's $1.4\\times10^{-6}$ -- which is inside the bound and
    would not have been inside a bound set from the CPU figure alone. Runs wherever a device
    exists, which includes the production box.

    Parametrised over both leg-alignment modes because the aligned one is the only place in the
    chain that gathers along the time axis with a broadcast index, and a gather is exactly the
    operation whose device implementation is not the CPU's.
    """
    reference = numpy_reference if leg_alignment == "none" else numpy_reference_aligned
    bank = CausalTorchBank(causal_bank, "cuda:0")
    worst = _gate(
        bank, raw_segments, reference, phase_pairs, np.float32, leg_alignment=leg_alignment
    )
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
