r"""The two-sided writer, pinned against a recomputation of itself.

This is the baseline the rest of the effort is measured against: the existing datasets must stay
reproducible, so a two-sided build has to keep producing the same arrays it produces today. The
comparison recomputes the expected coefficients **in this process** rather than checking a stored
hash. A hash needs a committed baseline with no natural home, and its classic failure mode is
someone pasting in the new number the first time it trips; a recomputation cannot go stale and
cannot be re-pasted.

What is guaranteed, and what deliberately is not
------------------------------------------------
Every stored *array* is identical -- coefficients, raw signals, targets, weights and metadata
alike. Attributes are **additive**: a file written by this pipeline may gain documented new root
keys, and gains nothing else. So this asserts array identity plus an attribute set bounded by
:data:`DOCUMENTED_NEW_ROOT_ATTRS`, never whole-file equality, which the new keys make false by
construction. A test asserting byte equality would have to be weakened the moment the first
attribute lands -- exactly the failure a baseline exists to prevent.

The file is built through the real :func:`create_initial_hdf5` and :func:`append_samples_batch`
with coefficients from the real transform, so this exercises the write path itself and not a
reimplementation of it. Only the ``.mat`` reading and segmentation upstream are bypassed, because
they need production data and touch no coefficient.
"""
from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import numpy as np
import pytest
import torch

from hdf5_dataset.causal_scattering import DECIMATION, N_RAW, CausalBank, transform_sample

from hdf5_dataset.tests.conftest import SHARD_PATH, requires_shard, scale_relative_errors

#: Storage geometry of a stored segment: $5280$ raw samples, $330$ decimated steps.
LEN_SIGNAL = N_RAW
LEN_SEQUENCE = N_RAW // DECIMATION

#: The shipped two-sided widths. Literals on purpose -- deriving them from the selection under
#: test would make the assertion circular.
EXPECTED_WIDTHS = {
    "fhr_st": 43,
    "fhr_ph": 66,
    "fhr_up_ph": 79,
    "up_st": 43,
    "up_ph": 15,
}

#: Every array a two-sided file stores. All of them are compared, not only the coefficient blocks:
#: a writer change that reordered the metadata would be just as damaging and just as invisible.
STORED_DATASETS = (
    "fhr", "up", "fhr_st", "fhr_ph", "fhr_up_ph", "up_st", "up_ph",
    "target", "weight", "epoch", "cs_label", "bg_label",
    "time_from_labor_onset", "second_stage_onset", "guid",
)

#: The causal layout: seven never-valid channels dropped from each scattering block, both phase
#: blocks untouched, and no cross-phase block at all.
EXPECTED_CAUSAL_WIDTHS = {"fhr_st": 36, "fhr_ph": 66, "up_st": 36, "up_ph": 15}

#: Per-channel provenance carried by the two self-phase blocks, and by nothing else.
#: ``sel_phase_operator`` names the operator version the block's exponents follow; on a two-sided
#: file it is always the legacy ratio-power operator, which is what production computes. The
#: integer operator additionally writes ``sel_harmonic``, on causal files only.
SEL_ATTRS = frozenset(
    {
        "sel_i", "sel_j", "sel_xi_i_hz", "sel_xi_j_hz", "sel_power", "sel_band_hz",
        "sel_k_steps", "sel_phase_operator",
    }
)

#: Root attributes a two-sided file carries today: none. ``create_initial_hdf5`` writes no root
#: attribute at all, and every dataset already on disk is in exactly this state.
BASELINE_ROOT_ATTRS: frozenset = frozenset()

#: Root attributes this effort is allowed to add to a two-sided file. A new key outside this set
#: fails here; extending the causal variant is what this list is extended for, one documented key
#: at a time, and every entry must be additive -- never a rename, a retype or a removal.
DOCUMENTED_NEW_ROOT_ATTRS = frozenset(
    {"transform", "source_pickle_path", "source_guid_digest"}
)

#: Segments compared against the numpy reference. The reference costs $\approx 1.5$ s a segment
#: and the agreement is a property of the transform, not of the sample, so four is plenty; the
#: written shard still carries all eight, because the loader and statistics tests want the rows.
GATED_SEGMENTS = 4

#: Storage is float32 and the reference is float64, so the gate is the single-precision pair.
#: Both metrics are normalised by the block's own scale, never pointwise -- the phase blocks cross
#: zero constantly, and a pointwise ratio there is unbounded on a difference of no consequence.
GATE_FLOAT32 = {"e_inf": 1e-5, "e_2": 1e-5}


def _forward_blocks(
    model: Any, fhr: np.ndarray, up: np.ndarray, masks: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Run the writer's four transform passes over a batch and reduce them by the stored masks.

    Mirrors ``create_hdf5_dataset_from_records_list``'s transform block: one pass per stored
    product, each selecting its channel and phase legs, then boolean indexing on the pair axis.
    Written out here rather than imported because the writer's copy is embedded in the record loop
    with ``.mat`` reading either side of it -- and because a comparison against code the writer
    shares would not be an independent check of the writer.

    Args:
        model: A ``KymatioPhaseScattering1D`` at the production geometry.
        fhr: Raw fetal heart rate, ``(B, 5280)`` float32.
        up: Raw uterine pressure, ``(B, 5280)`` float32.
        masks: The output of ``compute_scattering_masks``.

    Returns:
        The five coefficient blocks, each ``(B, C, 330)`` float32.
    """
    batch = torch.from_numpy(np.stack([fhr, up], axis=1)).float()
    phase_mask = masks["fhr_ph_selection"].mask
    cross_mask = masks["cross_mask"]
    up_phase_mask = masks["up_ph_selection"].mask

    fhr_pass = model(x=batch, compute_phase=True, compute_cross_phase=False,
                     scattering_channel=0, phase_channels=[0])
    cross_pass = model(x=batch, compute_phase=False, compute_cross_phase=True,
                       scattering_channel=0, phase_channels=[0, 1])
    up_phase_pass = model(x=batch, compute_phase=True, compute_cross_phase=False,
                          scattering_channel=0, phase_channels=[1])
    up_scatter_pass = model(x=batch, compute_phase=False, compute_cross_phase=False,
                            scattering_channel=1)

    return {
        "fhr_st": fhr_pass["scattering"].numpy(),
        "fhr_ph": fhr_pass["phase_corr"][:, phase_mask, :].numpy(),
        "fhr_up_ph": cross_pass["cross_phase_corr"][:, cross_mask, :].numpy(),
        "up_st": up_scatter_pass["scattering"].numpy(),
        "up_ph": up_phase_pass["phase_corr"][:, up_phase_mask, :].numpy(),
    }


def _metadata(n_samples: int) -> Dict[str, Any]:
    """Distinctive per-sample metadata, so a misordered or truncated write cannot look correct.

    Every value differs per sample and no two fields share a value, which is what makes an
    off-by-one or a transposed pair of columns visible rather than plausible.

    Args:
        n_samples: Batch size.

    Returns:
        The metadata arrays keyed by dataset name, plus the GUID list.
    """
    index = np.arange(n_samples, dtype=np.float32)
    return {
        "epoch": (-3600.0 - 120.0 * index).astype(np.float32),
        "cs_label": (index % 2).astype(np.uint8),
        "bg_label": ((index + 1) % 2).astype(np.uint8),
        "time_from_labor_onset": (7200.0 + 13.0 * index).astype(np.float32),
        "second_stage_onset": (-900.0 - 7.0 * index).astype(np.float32),
        "guid": [f"FIXTURE{i:04d}" for i in range(n_samples)],
    }


@pytest.fixture(scope="module")
def causal_file(
    pipeline: Any, causal_masks: Dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """An empty causal shard: the schema, the attributes, and nothing written into it yet.

    This is the sprint's demoable artefact, and it needs no shard and no ``.mat`` file to exist.
    """
    path = tmp_path_factory.mktemp("causal") / "causal.hdf5"
    pipeline.create_hdf5_for_masks(str(path), causal_masks, len_sequence=LEN_SEQUENCE)
    return path


@pytest.fixture(scope="module")
def two_sided_file(
    pipeline: Any,
    masks: Dict[str, Any],
    st_model: Any,
    raw_segments: Dict[str, np.ndarray],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A real two-sided shard, written from the fixture signals through the real writer functions.

    Returns:
        Path to the written file.
    """
    path = tmp_path_factory.mktemp("two_sided") / "two_sided.hdf5"
    fhr, up = raw_segments["fhr"], raw_segments["up"]
    n_samples = fhr.shape[0]

    # Through the resolver the real writer uses, so the widths the baseline is built at are the
    # widths a production run would use rather than a second set that happens to agree today.
    pipeline.create_hdf5_for_masks(str(path), masks, len_sequence=LEN_SEQUENCE)

    blocks = _forward_blocks(st_model, fhr, up, masks)
    meta = _metadata(n_samples)
    # target is the class index scaled by the sample weight, as the writer forms it; a non-trivial
    # weight makes that product something an accidental swap of the two would not reproduce.
    weight = np.linspace(0.9, 1.0, LEN_SEQUENCE, dtype=np.float32)[None, :].repeat(n_samples, 0)
    weight = (weight * (1.0 - 0.01 * np.arange(n_samples, dtype=np.float32))[:, None])
    target = (3 * weight).astype(np.float32)

    pipeline.append_samples_batch(
        path=str(path),
        fhr_batch=fhr,
        up_batch=up,
        fhr_st_batch=blocks["fhr_st"],
        fhr_ph_batch=blocks["fhr_ph"],
        fhr_up_ph_batch=blocks["fhr_up_ph"],
        target_batch=target,
        weight_batch=weight.astype(np.float32),
        guid_batch=meta["guid"],
        epoch_batch=meta["epoch"],
        cs_label_batch=meta["cs_label"],
        bg_label_batch=meta["bg_label"],
        tlo_batch=meta["time_from_labor_onset"],
        second_stage_batch=meta["second_stage_onset"],
        up_st_batch=blocks["up_st"],
        up_ph_batch=blocks["up_ph"],
    )
    return path


# =================================================================================================
# Schema
# =================================================================================================
def test_the_written_file_has_the_shipped_schema(two_sided_file: Path) -> None:
    """Exactly the documented datasets, at the documented widths, with nothing extra."""
    with h5py.File(two_sided_file, "r") as handle:
        assert set(handle.keys()) == set(STORED_DATASETS)
        for field, expected in EXPECTED_WIDTHS.items():
            assert handle[field].shape == (8, expected, LEN_SEQUENCE), field
        assert handle["fhr"].shape == handle["up"].shape == (8, LEN_SIGNAL)
        assert handle["target"].shape == handle["weight"].shape == (8, LEN_SEQUENCE)


def test_only_the_self_phase_blocks_carry_selection_provenance(two_sided_file: Path) -> None:
    """``sel_*`` lives on ``fhr_ph`` and ``up_ph``, exactly and only.

    Pinned as an exact key set in both directions: the causal variant adds per-block warm-up
    attributes, and this is what proves it added them to the causal file rather than to every file.
    """
    with h5py.File(two_sided_file, "r") as handle:
        for field in ("fhr_ph", "up_ph"):
            assert set(handle[field].attrs.keys()) == SEL_ATTRS, field
            for key in ("sel_i", "sel_j", "sel_xi_i_hz", "sel_xi_j_hz", "sel_power"):
                assert len(handle[field].attrs[key]) == EXPECTED_WIDTHS[field], f"{field}.{key}"
        for field in ("fhr_st", "up_st", "fhr_up_ph"):
            assert set(handle[field].attrs.keys()) == set(), field


def test_root_attributes_are_additive_only(two_sided_file: Path) -> None:
    """The section that keeps "unchanged" and "byte-identical" from being confused.

    A newly built two-sided file may carry documented new root keys and must carry every key it
    carried before. Anything else -- a stray attribute, a removed one -- fails here.
    """
    with h5py.File(two_sided_file, "r") as handle:
        present = set(handle.attrs.keys())
    assert BASELINE_ROOT_ATTRS <= present, "a pre-existing root attribute was dropped"
    assert present <= BASELINE_ROOT_ATTRS | DOCUMENTED_NEW_ROOT_ATTRS, (
        f"undocumented root attribute(s): "
        f"{sorted(present - BASELINE_ROOT_ATTRS - DOCUMENTED_NEW_ROOT_ATTRS)}"
    )


# =================================================================================================
# Coefficient identity
# =================================================================================================
def test_stored_coefficients_equal_a_recomputation(
    masks: Dict[str, Any], st_model: Any, two_sided_file: Path
) -> None:
    """Every stored coefficient block equals the transform re-run on the file's own raw signals.

    Reading ``fhr``/``up`` back off disk rather than reusing the arrays that were written is what
    makes this a round trip: it would catch a writer that stored the coefficients of one segment
    against the raw signal of another.

    Equality is exact. The transform is deterministic and its output is already float32, so
    storage is lossless; a tolerance here would hide precisely the drift this test exists to catch.
    """
    with h5py.File(two_sided_file, "r") as handle:
        fhr = np.asarray(handle["fhr"][:], dtype=np.float32)
        up = np.asarray(handle["up"][:], dtype=np.float32)
        stored = {field: np.asarray(handle[field][:]) for field in EXPECTED_WIDTHS}

    recomputed = _forward_blocks(st_model, fhr, up, masks)
    for field, expected in recomputed.items():
        assert np.array_equal(stored[field], expected.astype(np.float32)), field


def test_stored_signals_and_metadata_round_trip(
    raw_segments: Dict[str, np.ndarray], two_sided_file: Path
) -> None:
    """The other nine datasets, which no coefficient check would cover."""
    meta = _metadata(raw_segments["fhr"].shape[0])
    weight = np.linspace(0.9, 1.0, LEN_SEQUENCE, dtype=np.float32)[None, :].repeat(8, 0)
    weight = (weight * (1.0 - 0.01 * np.arange(8, dtype=np.float32))[:, None]).astype(np.float32)

    with h5py.File(two_sided_file, "r") as handle:
        assert np.array_equal(handle["fhr"][:], raw_segments["fhr"])
        assert np.array_equal(handle["up"][:], raw_segments["up"])
        assert np.array_equal(handle["weight"][:], weight)
        assert np.array_equal(handle["target"][:], (3 * weight).astype(np.float32))
        for field in ("epoch", "cs_label", "bg_label", "time_from_labor_onset",
                      "second_stage_onset"):
            assert np.array_equal(handle[field][:], meta[field]), field
        guids = [g.decode() if isinstance(g, bytes) else g for g in handle["guid"][:]]
        assert guids == meta["guid"]


def test_the_recomputation_would_notice_a_changed_coefficient(
    masks: Dict[str, Any], st_model: Any, two_sided_file: Path
) -> None:
    """The control: a single perturbed coefficient must break the comparison above.

    Without it, an assertion comparing two references to one array would pass just as well.
    """
    with h5py.File(two_sided_file, "r") as handle:
        fhr = np.asarray(handle["fhr"][:], dtype=np.float32)
        up = np.asarray(handle["up"][:], dtype=np.float32)
        stored = np.asarray(handle["fhr_st"][:])

    tampered = stored.copy()
    tampered[0, 0, 0] = np.float32(tampered[0, 0, 0] + 1.0)
    recomputed = _forward_blocks(st_model, fhr, up, masks)["fhr_st"]
    assert np.array_equal(stored, recomputed.astype(np.float32))
    assert not np.array_equal(tampered, recomputed.astype(np.float32))


# =================================================================================================
# Variant selection
# =================================================================================================
def test_both_parameters_reach_every_write_path(pipeline: Any) -> None:
    """``transform`` and ``device`` on all three functions, checked by signature.

    A half-threaded parameter is the real failure mode here, not a wrong one: the pre-training
    files at the end of the pipeline call the writer **directly**, bypassing
    ``_build_hdf5_for_partition``, so a variant plumbed into the partition path alone produces an
    output directory whose classification and pre-training files disagree about what they contain.
    Reading the signatures catches that; running one path does not.
    """
    for function in (
        pipeline.create_new_pipeline,
        pipeline._build_hdf5_for_partition,
        pipeline.create_hdf5_dataset_from_records_list,
    ):
        parameters = inspect.signature(function).parameters
        assert "transform" in parameters, function.__name__
        assert "device" in parameters, function.__name__
        # Today's behaviour is the default on every one of them.
        assert parameters["transform"].default == "two_sided", function.__name__
        assert parameters["device"].default is None, function.__name__

    assert "transform" in inspect.signature(pipeline.compute_scattering_masks).parameters
    assert "device" in inspect.signature(pipeline.compute_scattering_masks).parameters


def test_an_unknown_transform_is_refused_before_anything_is_created(
    pipeline: Any, tmp_path: Path
) -> None:
    """The refusal is the **first** statement, before ``os.makedirs`` and before the CSV.

    Validated later it would leave an output directory behind and fail on a missing CSV instead,
    reporting the wrong problem and needing a manual cleanup before the retry.
    """
    output = tmp_path / "never_created"
    with pytest.raises(ValueError) as error:
        pipeline.create_new_pipeline(
            records_base_path=str(tmp_path),
            output_base_path=str(output),
            tlo_csv_path=str(tmp_path / "does_not_exist.csv"),
            transform="one_sided",
        )
    assert "two_sided" in str(error.value) and "causal" in str(error.value)
    assert not output.exists(), "the refusal left an output directory behind"


def test_the_writer_refuses_masks_from_the_other_variant(
    pipeline: Any, masks: Dict[str, Any], causal_file: Path
) -> None:
    """Two-sided masks against a causal build is a mistake with no safe interpretation.

    The causal path needs the channel plan, which only the causal masks carry; without this the
    build would fall through to the two-sided branch and write 43-wide blocks into 36-wide
    datasets.
    """
    with pytest.raises(ValueError, match="masks were computed for"):
        pipeline.create_hdf5_dataset_from_records_list(
            hdf5_path=str(causal_file),
            records_list=[],
            cs_label=False,
            bg_label=True,
            pre_defined_target=1,
            precomputed_masks=masks,
            labor_onset_map={},
            second_stage_map={},
            device=torch.device("cpu"),
            verbose=False,
            transform="causal",
        )


def test_the_layout_comes_from_the_model_two_sided_and_the_plan_causal(
    pipeline: Any, masks: Dict[str, Any], causal_masks: Dict[str, Any], st_model: Any
) -> None:
    """Where each width comes from, asserted rather than assumed.

    The two numbers are equal for the phase blocks and differ for the scattering ones, so a
    resolver that read the wrong source would still look right on two blocks out of four.
    """
    two_sided = pipeline.resolve_channel_layout(masks)
    n_scattering = 1 + len(st_model.center_freqs)
    assert two_sided == {
        "fhr_st": n_scattering,
        "fhr_ph": masks["fhr_ph_selection"].n_channels,
        "fhr_up_ph": int(masks["n_cross"]),
        "up_st": n_scattering,
        "up_ph": masks["up_ph_selection"].n_channels,
    }
    assert two_sided == dict(EXPECTED_WIDTHS)

    plan = causal_masks["channel_plan"]
    causal = pipeline.resolve_channel_layout(causal_masks)
    assert causal == {
        "fhr_st": plan["fhr_st"].n_channels,
        "fhr_ph": plan["fhr_ph"].n_channels,
        "fhr_up_ph": None,
        "up_st": plan["up_st"].n_channels,
        "up_ph": plan["up_ph"].n_channels,
    }
    assert causal == dict(EXPECTED_CAUSAL_WIDTHS, fhr_up_ph=None)


# =================================================================================================
# The causal schema
# =================================================================================================
def test_the_causal_file_has_the_causal_layout(causal_file: Path) -> None:
    """36/66/36/15 and no ``fhr_up_ph`` — the whole schema difference, on disk."""
    with h5py.File(causal_file, "r") as handle:
        assert "fhr_up_ph" not in handle
        assert set(handle.keys()) == set(STORED_DATASETS) - {"fhr_up_ph"}
        for field, expected in EXPECTED_CAUSAL_WIDTHS.items():
            assert handle[field].shape == (0, expected, LEN_SEQUENCE), field


def test_the_causal_file_records_what_its_warm_up_means(causal_file: Path) -> None:
    r"""The bank constants the warm-up vectors were measured under, at the root.

    Without them ``causal_warmup_steps`` is a bare integer per channel that cannot be checked,
    reproduced or compared against a differently-built file.
    """
    with h5py.File(causal_file, "r") as handle:
        attrs = dict(handle.attrs)
    assert attrs["transform"] == "causal"
    assert int(attrs["causal_kernel_taps"]) == 1 << 15
    assert int(attrs["gammatone_order"]) == 4
    assert float(attrs["causal_warmup_quantile"]) == pytest.approx(0.95)


def test_every_causal_block_carries_its_warm_up_and_delay(
    causal_file: Path, causal_masks: Dict[str, Any]
) -> None:
    """One value per stored channel, in the plan's order, at the documented dtypes."""
    plan = causal_masks["channel_plan"]
    with h5py.File(causal_file, "r") as handle:
        for field, width in EXPECTED_CAUSAL_WIDTHS.items():
            attrs = handle[field].attrs
            warmup = np.asarray(attrs["causal_warmup_steps"])
            delay = np.asarray(attrs["causal_delay_s"])
            assert warmup.dtype == np.int32 and delay.dtype == np.float32, field
            assert warmup.shape == delay.shape == (width,), field
            assert np.array_equal(warmup, plan[field].warmup_steps), field
            assert np.allclose(delay, plan[field].delay_s, rtol=1e-6), field
        # The measured layout, so a silent change to the drop rule or the bank moves this test.
        assert int(handle["fhr_st"].attrs["causal_warmup_steps"].max()) == 293
        assert int(handle["up_ph"].attrs["causal_warmup_steps"].min()) == 56


def test_the_causal_file_carries_the_two_sided_selection_unchanged(
    causal_file: Path, two_sided_file: Path
) -> None:
    """Both phase selections survive the causal build element for element.

    This is what makes the drop a clean channel-axis operation on the scattering blocks rather
    than a re-selection: the phase bands stop at $0.008$ Hz, above every dropped filter.
    """
    with h5py.File(causal_file, "r") as causal, h5py.File(two_sided_file, "r") as two_sided:
        for field in ("fhr_ph", "up_ph"):
            causal_attrs, two_sided_attrs = causal[field].attrs, two_sided[field].attrs
            assert SEL_ATTRS <= set(causal_attrs.keys()), field
            for key in SEL_ATTRS:
                assert np.array_equal(
                    np.asarray(causal_attrs[key]), np.asarray(two_sided_attrs[key])
                ), f"{field}.{key}"


def test_a_half_configured_causal_file_is_refused(
    pipeline: Any, causal_masks: Dict[str, Any], masks: Dict[str, Any], tmp_path: Path
) -> None:
    """Three ways to ask for a causal file that would be silently wrong, all refused.

    Each one produces a file that looks plausible and is not: no plan means no warm-up vectors at
    all; a cross-phase width means a dataset nothing will ever fill; a width that disagrees with
    the plan means a warm-up vector describing channels the data does not contain.
    """
    plan = causal_masks["channel_plan"]
    common = dict(
        path=str(tmp_path / "bad.hdf5"),
        len_signal=LEN_SIGNAL,
        len_sequence=LEN_SEQUENCE,
        fhr_ph_selection=causal_masks["fhr_ph_selection"],
        up_ph_selection=causal_masks["up_ph_selection"],
        transform="causal",
    )
    with pytest.raises(ValueError, match="needs its channel_plan"):
        pipeline.create_initial_hdf5(
            **common, n_fhr_st_channels=36, n_cross_phase_channels=None, n_up_st_channels=36
        )
    with pytest.raises(ValueError, match="does not produce fhr_up_ph"):
        pipeline.create_initial_hdf5(
            **common, n_fhr_st_channels=36, n_cross_phase_channels=int(masks["n_cross"]),
            n_up_st_channels=36, channel_plan=plan,
        )
    with pytest.raises(ValueError, match="channel plan for 'fhr_st'"):
        pipeline.create_initial_hdf5(
            **common, n_fhr_st_channels=43, n_cross_phase_channels=None,
            n_up_st_channels=36, channel_plan=plan,
        )
    with pytest.raises(ValueError, match="unknown transform"):
        pipeline.create_initial_hdf5(
            path=str(tmp_path / "bad.hdf5"), len_signal=LEN_SIGNAL, len_sequence=LEN_SEQUENCE,
            fhr_ph_selection=masks["fhr_ph_selection"], n_fhr_st_channels=43,
            n_cross_phase_channels=int(masks["n_cross"]), n_up_st_channels=43,
            up_ph_selection=masks["up_ph_selection"], transform="reversed",
        )


# =================================================================================================
# The geometry guard
# =================================================================================================
def test_the_guard_reads_the_required_block_set_from_the_widths(
    pipeline: Any, two_sided_file: Path, causal_file: Path
) -> None:
    """A width means "must exist this wide"; ``None`` means "must be absent".

    That mapping *is* the per-variant required-block set, which is why one implementation with no
    variant branch inside can keep a missing ``fhr_up_ph`` fatal for a two-sided build and correct
    for a causal one.
    """
    two_sided_widths = dict(EXPECTED_WIDTHS)
    causal_widths = dict(EXPECTED_CAUSAL_WIDTHS, fhr_up_ph=None)
    pipeline._validate_geometry(str(two_sided_file), two_sided_widths)
    pipeline._validate_geometry(str(causal_file), causal_widths)

    # Each direction is asked for with the *other* variant's cross-phase entry alone, so what
    # fails is the required-block rule rather than a scattering width that also differs.
    with pytest.raises(ValueError, match="Dataset 'fhr_up_ph' is missing"):
        pipeline._validate_geometry(
            str(causal_file), dict(EXPECTED_CAUSAL_WIDTHS, fhr_up_ph=79)
        )
    with pytest.raises(ValueError, match="does not produce it"):
        pipeline._validate_geometry(str(two_sided_file), dict(EXPECTED_WIDTHS, fhr_up_ph=None))


def test_the_guard_still_names_a_width_mismatch(pipeline: Any, causal_file: Path) -> None:
    """The original failure it was written for, now on the causal widths."""
    with pytest.raises(ValueError, match="Channel-count mismatch for 'fhr_st'") as error:
        pipeline._validate_geometry(str(causal_file), dict(EXPECTED_CAUSAL_WIDTHS, fhr_st=43))
    assert "36" in str(error.value) and "43" in str(error.value)


def test_the_pair_axis_check_is_skipped_without_a_model(
    pipeline: Any, masks: Dict[str, Any], two_sided_file: Path
) -> None:
    """The causal path indexes pairs directly, so it supplies no pair axis to check.

    Passing masks with no ``n_pairs`` must not silently *pass* a check it never ran, so the same
    masks are shown to fail when a pair count is supplied.
    """
    wrong_length = {"fhr_ph": masks["fhr_ph_selection"].mask[:10]}
    pipeline._validate_geometry(str(two_sided_file), dict(EXPECTED_WIDTHS),
                                pair_masks=wrong_length)
    with pytest.raises(ValueError, match="Phase-pair axis mismatch"):
        pipeline._validate_geometry(str(two_sided_file), dict(EXPECTED_WIDTHS),
                                    pair_masks=wrong_length, n_pairs=903)


# =================================================================================================
# The append guard
# =================================================================================================
def _minimal_batch(pipeline: Any, path: Path, widths: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """One all-zero sample's worth of arguments for :func:`append_samples_batch`."""
    del pipeline
    block = lambda width: np.zeros((1, width, LEN_SEQUENCE), dtype=np.float32)
    return dict(
        path=str(path),
        fhr_batch=np.zeros((1, LEN_SIGNAL), dtype=np.float32),
        up_batch=np.zeros((1, LEN_SIGNAL), dtype=np.float32),
        fhr_st_batch=block(widths["fhr_st"]),
        fhr_ph_batch=block(widths["fhr_ph"]),
        target_batch=np.zeros((1, LEN_SEQUENCE), dtype=np.float32),
        weight_batch=np.zeros((1, LEN_SEQUENCE), dtype=np.float32),
        guid_batch=["GUARD"],
        epoch_batch=np.zeros(1, dtype=np.float32),
        cs_label_batch=np.zeros(1, dtype=np.uint8),
        bg_label_batch=np.ones(1, dtype=np.uint8),
        tlo_batch=np.zeros(1, dtype=np.float32),
        second_stage_batch=np.zeros(1, dtype=np.float32),
        up_st_batch=block(widths["up_st"]),
        up_ph_batch=block(widths["up_ph"]),
    )


def test_appending_to_a_causal_file_needs_no_cross_phase_batch(
    pipeline: Any, causal_masks: Dict[str, Any], tmp_path: Path
) -> None:
    """The append path on a file with no ``fhr_up_ph``, including the resize loop.

    The loop iterates the datasets that exist, so an absent block is simply not resized — asserted
    here rather than assumed, because a resize that touched a missing name would raise mid-write
    with a batch already half-applied.
    """
    path = tmp_path / "causal_append.hdf5"
    pipeline.create_hdf5_for_masks(str(path), causal_masks, len_sequence=LEN_SEQUENCE)
    pipeline.append_samples_batch(
        **_minimal_batch(pipeline, path, dict(EXPECTED_CAUSAL_WIDTHS))
    )
    with h5py.File(path, "r") as handle:
        assert handle["fhr_st"].shape == (1, 36, LEN_SEQUENCE)
        assert handle["up_ph"].shape == (1, 15, LEN_SEQUENCE)
        assert "fhr_up_ph" not in handle


def test_the_cross_phase_guard_is_symmetric(
    pipeline: Any, masks: Dict[str, Any], causal_masks: Dict[str, Any], tmp_path: Path
) -> None:
    """Both directions raise, and for different reasons that are both silent otherwise.

    Skipping the block when the dataset is absent — the pattern the other optional blocks use —
    would compute a two-sided cross-phase block and drop it on the floor for the whole build.
    Writing it unconditionally would raise ``KeyError`` on every causal file.
    """
    two_sided_path = tmp_path / "two_sided_append.hdf5"
    pipeline.create_hdf5_for_masks(str(two_sided_path), masks, len_sequence=LEN_SEQUENCE)
    with pytest.raises(ValueError, match="no fhr_up_ph_batch was given"):
        pipeline.append_samples_batch(
            **_minimal_batch(pipeline, two_sided_path, dict(EXPECTED_WIDTHS))
        )

    causal_path = tmp_path / "causal_append_guard.hdf5"
    pipeline.create_hdf5_for_masks(str(causal_path), causal_masks, len_sequence=LEN_SEQUENCE)
    with pytest.raises(ValueError, match="dropped on the floor"):
        pipeline.append_samples_batch(
            **_minimal_batch(pipeline, causal_path, dict(EXPECTED_CAUSAL_WIDTHS)),
            fhr_up_ph_batch=np.zeros((1, 79, LEN_SEQUENCE), dtype=np.float32),
        )


# =================================================================================================
# The causal writer, end to end
# =================================================================================================
#: What a resumed run records, and what its shards must agree on afterwards.
FOLD_PICKLE = "/data1/fetal-heart-tracing/HDF5_Datasets/run_6h/classification_dataset_records.pickle"

#: A record list standing in for one subgroup of a fold. Only the basenames matter: they are the
#: GUIDs, and the digest is taken over them.
RECORDS = ["/records/EFMOut/GUID0007.mat", "/records/EFMOut/GUID0003.mat"]


def write_causal_shard(
    pipeline: Any,
    causal_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    path: Path,
    *,
    records_list: Optional[List[str]] = None,
    source_pickle_path: Optional[str] = None,
) -> Path:
    """Build a populated causal shard from the fixture signals, through the real writer functions.

    The transform stage is the writer's own :func:`_transform_causal_record`, and the write is the
    real :func:`append_samples_batch`, so this exercises the shipped path rather than a
    reimplementation of it. Only the ``.mat`` reading and segmentation upstream are replaced, and
    those touch no coefficient.

    Shared rather than inlined because the loader and the statistics calculator need exactly this
    file as their input, and a second builder would be free to produce a subtly different one.

    Args:
        pipeline: The writer module.
        causal_masks: Selections resolved for the causal variant, with the channel plan.
        raw_segments: The committed ``fhr``/``up`` fixture.
        path: Where to write.
        records_list: GUID provenance for the shard.
        source_pickle_path: Fold pickle provenance.

    Returns:
        *path*, for convenience at the call site.
    """
    fhr, up = raw_segments["fhr"], raw_segments["up"]
    n_samples = fhr.shape[0]
    pipeline.create_hdf5_for_masks(
        str(path), causal_masks, len_sequence=LEN_SEQUENCE,
        records_list=records_list, source_pickle_path=source_pickle_path,
    )

    blocks, failures = pipeline._transform_causal_record(
        pipeline.CausalTorchBank(causal_masks["causal_bank"], torch.device("cpu")),
        fhr, up,
        pipeline._selection_pairs(causal_masks["fhr_ph_selection"]),
        pipeline._selection_pairs(causal_masks["up_ph_selection"]),
        causal_masks["channel_plan"],
        3,
    )
    assert not failures, failures

    meta = _metadata(n_samples)
    weight = np.linspace(0.9, 1.0, LEN_SEQUENCE, dtype=np.float32)[None, :].repeat(n_samples, 0)
    weight = (weight * (1.0 - 0.01 * np.arange(n_samples, dtype=np.float32))[:, None])
    pipeline.append_samples_batch(
        path=str(path),
        fhr_batch=fhr,
        up_batch=up,
        fhr_st_batch=np.stack([b["fhr_st"] for b in blocks]),
        fhr_ph_batch=np.stack([b["fhr_ph"] for b in blocks]),
        # No fhr_up_ph_batch at all: the causal file has no such dataset, and passing one would be
        # refused rather than dropped.
        target_batch=(3 * weight).astype(np.float32),
        weight_batch=weight.astype(np.float32),
        guid_batch=meta["guid"],
        epoch_batch=meta["epoch"],
        cs_label_batch=meta["cs_label"],
        bg_label_batch=meta["bg_label"],
        tlo_batch=meta["time_from_labor_onset"],
        second_stage_batch=meta["second_stage_onset"],
        up_st_batch=np.stack([b["up_st"] for b in blocks]),
        up_ph_batch=np.stack([b["up_ph"] for b in blocks]),
    )
    return path


@pytest.fixture(scope="module")
def causal_written_file(
    pipeline: Any,
    causal_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """The sprint's headline artefact: a populated causal shard on disk, with its provenance."""
    return write_causal_shard(
        pipeline,
        causal_masks,
        raw_segments,
        tmp_path_factory.mktemp("causal_written") / "causal_written.hdf5",
        records_list=RECORDS,
        source_pickle_path=FOLD_PICKLE,
    )


@pytest.fixture(scope="module")
def causal_numpy_reference(
    causal_bank: CausalBank, raw_segments: Dict[str, np.ndarray]
) -> List[Dict[str, np.ndarray]]:
    """The validated numpy chain, undropped, on the gated segments.

    Undropped on purpose: gathering the plan's channels out of the reference here is what makes the
    comparison prove the writer stored the *right* channels, not merely 36 plausible ones.
    """
    return [
        transform_sample(
            raw_segments["fhr"][index].astype(np.float64),
            raw_segments["up"][index].astype(np.float64),
            causal_bank,
        )
        for index in range(GATED_SEGMENTS)
    ]


def test_the_written_causal_shard_has_the_causal_schema(causal_written_file: Path) -> None:
    """Eight real segments at 36/66/36/15, no cross-phase block, and the documented dtypes."""
    with h5py.File(causal_written_file, "r") as handle:
        assert "fhr_up_ph" not in handle
        assert set(handle.keys()) == set(STORED_DATASETS) - {"fhr_up_ph"}
        for field, expected in EXPECTED_CAUSAL_WIDTHS.items():
            assert handle[field].shape == (8, expected, LEN_SEQUENCE), field
            assert handle[field].dtype == np.float32, field
        assert handle["fhr"].shape == handle["up"].shape == (8, LEN_SIGNAL)
        assert handle["guid"].shape == (8,)


def test_the_written_causal_blocks_match_the_numpy_reference(
    causal_written_file: Path,
    causal_masks: Dict[str, Any],
    causal_numpy_reference: List[Dict[str, np.ndarray]],
) -> None:
    r"""What was stored is the validated transform, restricted to the surviving channels.

    Two things fail here and nowhere else: a batched chain that disagrees with the reference by
    more than single-precision round-off, and a gather that kept the wrong rows. The second is why
    the reference is compared undropped-then-gathered rather than recomputed at 36 channels.
    """
    plan = causal_masks["channel_plan"]
    with h5py.File(causal_written_file, "r") as handle:
        stored = {
            field: np.asarray(handle[field][:GATED_SEGMENTS]) for field in EXPECTED_CAUSAL_WIDTHS
        }

    for index, reference in enumerate(causal_numpy_reference):
        for field in EXPECTED_CAUSAL_WIDTHS:
            expected = reference[field][plan[field].kept, :]
            e_inf, e_2 = scale_relative_errors(stored[field][index], expected)
            assert e_inf <= GATE_FLOAT32["e_inf"], f"{field} segment {index}: {e_inf}"
            assert e_2 <= GATE_FLOAT32["e_2"], f"{field} segment {index}: {e_2}"


def test_the_comparison_would_notice_a_wrong_channel_or_a_changed_value(
    causal_written_file: Path,
    causal_masks: Dict[str, Any],
    causal_numpy_reference: List[Dict[str, np.ndarray]],
) -> None:
    """The control. A gate that cannot fail proves nothing, so it is made to fail twice over.

    Once on a perturbed coefficient, and once on a block whose channels were gathered one row off
    -- the failure a writer that took its channel map from a second selector would produce, and
    the one a shape check alone would never see.
    """
    plan = causal_masks["channel_plan"]
    reference = causal_numpy_reference[0]["fhr_st"]
    with h5py.File(causal_written_file, "r") as handle:
        stored = np.asarray(handle["fhr_st"][0])

    tampered = stored.copy()
    tampered[0, 0] = np.float32(tampered[0, 0] + 1.0)
    assert scale_relative_errors(tampered, reference[plan["fhr_st"].kept, :])[0] > \
        GATE_FLOAT32["e_inf"]

    shifted = reference[plan["fhr_st"].kept + 1, :]
    assert scale_relative_errors(stored, shifted)[0] > GATE_FLOAT32["e_inf"]


def test_the_written_warm_up_is_one_value_per_stored_channel(
    causal_written_file: Path, causal_masks: Dict[str, Any]
) -> None:
    """The attribute and the channel axis of the data it describes agree, after a real write."""
    plan = causal_masks["channel_plan"]
    with h5py.File(causal_written_file, "r") as handle:
        for field, width in EXPECTED_CAUSAL_WIDTHS.items():
            warmup = np.asarray(handle[field].attrs["causal_warmup_steps"])
            assert warmup.shape == (width,) == (handle[field].shape[1],), field
            assert np.array_equal(warmup, plan[field].warmup_steps), field


# =================================================================================================
# The record writer's causal branch, and its failure isolation
# =================================================================================================
#: Segment epochs the stand-in adaptor reports, one per fixture segment. Distinct and negative:
#: the writer deduplicates on epoch and skips anything at or after delivery.
def _epochs(n_samples: int) -> List[float]:
    """Distinct pre-delivery domain starts, twenty minutes apart."""
    return [-3600.0 - 1200.0 * index for index in range(n_samples)]


class _StandInMimo:
    """The subset of the production adaptor's interface the record writer actually touches.

    The writer reads ``.mat`` files through ``EarlyMaestraMimoAdaptor``, which exists only on the
    production box. Everything downstream of it — the quality filter, the transform, the collection
    and the write — is the shipped code, so replacing the reader is what lets the causal branch be
    exercised end to end off the committed fixture.
    """

    def __init__(self, fhr: np.ndarray, up: np.ndarray) -> None:
        n_samples = fhr.shape[0]
        # ``block_input`` is ``(N, samples, channel)`` with UP at 0 and FHR at 1 — the layout the
        # writer slices. Reproduced exactly, because swapping the two would be silent.
        self.block_input = np.stack([up, fhr], axis=2)
        self.domain_start = _epochs(n_samples)
        self.sample_weights = np.ones((n_samples, LEN_SEQUENCE), dtype=np.float32)
        self.mimo = self

    def read_single_input(self, record: str, **_kwargs: Any) -> None:
        """Accept the record path and ignore it; the signals are already loaded."""

    def prepare_data(self, **_kwargs: Any) -> Any:
        """Return the prepared segments in the ``(prepared, _)`` shape the writer unpacks."""
        return self, None


def _run_record_writer(
    pipeline: Any,
    run_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    transform: str = "causal",
    transform_hook: Optional[Callable[..., Any]] = None,
    run_guid_analysis: bool = False,
) -> List[str]:
    """Drive the real record writer over the fixture signals, on either branch.

    Args:
        pipeline: The writer module.
        run_masks: Selections for the requested variant, with a channel plan if causal.
        raw_segments: The committed fixture.
        path: Shard to create and fill.
        monkeypatch: For the adaptor stand-in and the optional transform hook.
        transform: Which branch to exercise.
        transform_hook: Replacement for the module-level causal transform, for the failure tests.
        run_guid_analysis: Whether per-GUID tracking is collected.

    Returns:
        The writer's list of errored records; empty on a clean run.
    """
    monkeypatch.setattr(
        pipeline, "EarlyMaestraMimoAdaptor",
        lambda **_kwargs: _StandInMimo(raw_segments["fhr"], raw_segments["up"]),
    )
    if transform_hook is not None:
        monkeypatch.setattr(pipeline, "transform_batch_numpy", transform_hook)

    pipeline.create_hdf5_for_masks(
        str(path), run_masks, len_sequence=LEN_SEQUENCE,
        records_list=RECORDS[:1], source_pickle_path=FOLD_PICKLE,
    )
    return pipeline.create_hdf5_dataset_from_records_list(
        hdf5_path=str(path),
        records_list=RECORDS[:1],
        cs_label=False,
        bg_label=True,
        pre_defined_target=1,
        precomputed_masks=run_masks,
        labor_onset_map={},
        second_stage_map={},
        device=torch.device("cpu"),
        run_guid_analysis=run_guid_analysis,
        # Smaller than the eight fixture segments, so the batching and its retry are both real.
        scatter_batch_size=3,
        verbose=False,
        transform=transform,
    )


def test_the_record_writer_still_fills_a_two_sided_shard(
    pipeline: Any,
    masks: Dict[str, Any],
    st_model: Any,
    raw_segments: Dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The two-sided record loop, unchanged, checked where the schema baseline cannot reach.

    The baseline above drives ``create_initial_hdf5`` and ``append_samples_batch`` directly, so the
    record loop between them — the four transform passes, the mask reduction and the collection —
    is covered by nothing else. It is the code the causal branch was added beside, which is exactly
    when it is worth pinning.
    """
    path = tmp_path / "two_sided_record_writer.hdf5"
    errors = _run_record_writer(
        pipeline, masks, raw_segments, path, monkeypatch, transform="two_sided"
    )
    assert errors == []

    expected = _forward_blocks(st_model, raw_segments["fhr"], raw_segments["up"], masks)
    with h5py.File(path, "r") as handle:
        for field, width in EXPECTED_WIDTHS.items():
            assert handle[field].shape == (8, width, LEN_SEQUENCE), field
            assert np.array_equal(handle[field][:], expected[field].astype(np.float32)), field


def test_the_record_writer_fills_a_causal_shard(
    pipeline: Any,
    causal_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    causal_written_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The causal branch of the writer, from segmentation to append, at the shipped widths.

    The stored coefficients are compared against the shard built through the transform stage
    directly, which the numpy reference already gates: what this adds is that the branch inside the
    record loop reaches the same values through the real collection and append path.
    """
    path = tmp_path / "record_writer.hdf5"
    assert _run_record_writer(pipeline, causal_masks, raw_segments, path, monkeypatch) == []

    with h5py.File(path, "r") as handle, h5py.File(causal_written_file, "r") as expected:
        assert "fhr_up_ph" not in handle
        for field, width in EXPECTED_CAUSAL_WIDTHS.items():
            assert handle[field].shape == (8, width, LEN_SEQUENCE), field
            assert np.array_equal(handle[field][:], expected[field][:]), field
        assert np.array_equal(handle["fhr"][:], raw_segments["fhr"])
        assert np.array_equal(handle["epoch"][:], np.array(_epochs(8), dtype=np.float32))


def test_a_batch_that_fails_is_retried_a_segment_at_a_time(
    pipeline: Any,
    causal_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    causal_written_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The out-of-memory guard: a batch that will not fit is recomputed one segment at a time.

    Every batch here fails, so every stored segment came through the retry — and every one still
    agrees with what the unbroken batched path produced. Agreement, not bit-equality: an FFT
    library is free to round differently at a different batch size, and it does, in the last
    single-precision place. The requirement is that a retried segment is stored on the same terms
    as its peers to the precision the dataset is built in, which is what the numerical gate
    measures.
    """
    real = pipeline.transform_batch_numpy

    def only_one_segment_fits(bank: Any, fhr: np.ndarray, up: np.ndarray, *args: Any,
                              **kwargs: Any) -> Any:
        if fhr.shape[0] > 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real(bank, fhr, up, *args, **kwargs)

    path = tmp_path / "retried.hdf5"
    errors = _run_record_writer(
        pipeline, causal_masks, raw_segments, path, monkeypatch,
        transform_hook=only_one_segment_fits,
    )
    assert errors == []

    with h5py.File(path, "r") as handle, h5py.File(causal_written_file, "r") as expected:
        for field in EXPECTED_CAUSAL_WIDTHS:
            assert handle[field].shape[0] == 8, field
            e_inf, e_2 = scale_relative_errors(handle[field][:], expected[field][:])
            assert e_inf <= GATE_FLOAT32["e_inf"], f"{field}: {e_inf}"
            assert e_2 <= GATE_FLOAT32["e_2"], f"{field}: {e_2}"


def test_a_segment_that_fails_twice_is_dropped_and_the_record_still_writes(
    pipeline: Any,
    causal_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One unusable segment costs that segment, not the record and not the run.

    It is also recorded rather than merely skipped: without the tracking entry a shard would be
    quietly short of segments with nothing anywhere saying which, or how many.
    """
    doomed = 2
    marker = raw_segments["fhr"][doomed]
    real = pipeline.transform_batch_numpy

    def one_segment_never_works(bank: Any, fhr: np.ndarray, up: np.ndarray, *args: Any,
                                **kwargs: Any) -> Any:
        if any(np.array_equal(row, marker) for row in np.asarray(fhr)):
            raise RuntimeError("segment is unusable (simulated)")
        return real(bank, fhr, up, *args, **kwargs)

    # The tracking is handed to guid_analysis, which is the only way out of the writer; a stand-in
    # module captures it instead of letting the real one run against a two-segment shard.
    captured: Dict[str, Any] = {}
    analysis = types.ModuleType("guid_analysis")

    def _capture(hdf5_path: str, guid_tracking: Dict[str, Any], **_kwargs: Any) -> None:
        captured.update(guid_tracking)

    analysis.run_guid_analysis = _capture  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "guid_analysis", analysis)

    path = tmp_path / "one_failed.hdf5"
    errors = _run_record_writer(
        pipeline, causal_masks, raw_segments, path, monkeypatch,
        transform_hook=one_segment_never_works, run_guid_analysis=True,
    )
    assert errors == [], "one bad segment must not fail its record"

    expected_epochs = [e for index, e in enumerate(_epochs(8)) if index != doomed]
    with h5py.File(path, "r") as handle:
        assert handle["fhr_st"].shape == (7, 36, LEN_SEQUENCE)
        assert np.array_equal(handle["epoch"][:], np.array(expected_epochs, dtype=np.float32))
        assert np.array_equal(
            handle["fhr"][:], np.delete(raw_segments["fhr"], doomed, axis=0)
        )

    entry = captured["GUID0007"]
    assert entry.skipped_scatter_failed == [_epochs(8)[doomed]]
    assert len(entry.included_domain_starts) == 7


# =================================================================================================
# Fold-pickle provenance
# =================================================================================================
def test_a_shard_records_the_pickle_and_the_guid_set_it_was_built_from(
    causal_written_file: Path, pipeline: Any
) -> None:
    """Comparability is the whole justification for the resumed run; this is its only evidence."""
    with h5py.File(causal_written_file, "r") as handle:
        attrs = dict(handle.attrs)
    assert attrs["source_pickle_path"] == FOLD_PICKLE
    assert attrs["source_guid_digest"] == pipeline.guid_set_digest(RECORDS)


def test_the_digest_is_the_guid_set_and_not_its_order_or_variant(
    pipeline: Any,
    masks: Dict[str, Any],
    causal_masks: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Two shards from one pickle agree whatever their variant, and disagree on a different set.

    Sorting before hashing is what makes the first half true: the record list arrives from a
    directory listing on one box and a pickle on another, and an order-sensitive digest would
    report two identical datasets as incomparable.
    """
    two_sided_path = tmp_path / "two_sided_provenance.hdf5"
    causal_path = tmp_path / "causal_provenance.hdf5"
    pipeline.create_hdf5_for_masks(
        str(two_sided_path), masks, len_sequence=LEN_SEQUENCE,
        records_list=list(reversed(RECORDS)), source_pickle_path=FOLD_PICKLE,
    )
    pipeline.create_hdf5_for_masks(
        str(causal_path), causal_masks, len_sequence=LEN_SEQUENCE,
        records_list=RECORDS, source_pickle_path=FOLD_PICKLE,
    )
    with h5py.File(two_sided_path, "r") as two_sided, h5py.File(causal_path, "r") as causal:
        assert two_sided.attrs["source_guid_digest"] == causal.attrs["source_guid_digest"]
        assert two_sided.attrs["source_pickle_path"] == causal.attrs["source_pickle_path"]
        assert two_sided.attrs["transform"] == "two_sided"
        assert causal.attrs["transform"] == "causal"

    assert pipeline.guid_set_digest(RECORDS) != pipeline.guid_set_digest(RECORDS[:1])


def test_a_fresh_run_records_the_absence_of_a_pickle_explicitly(
    pipeline: Any, masks: Dict[str, Any], tmp_path: Path
) -> None:
    """An empty string would be indistinguishable from an attribute written out of an unset name."""
    path = tmp_path / "fresh.hdf5"
    pipeline.create_hdf5_for_masks(str(path), masks, len_sequence=LEN_SEQUENCE)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["source_pickle_path"] == pipeline.NO_SOURCE_PICKLE
        assert handle.attrs["source_pickle_path"] != ""


# =================================================================================================
# The resolved layout
# =================================================================================================
def test_the_layout_dict_is_derived_from_the_channel_plan(
    pipeline: Any, causal_masks: Dict[str, Any]
) -> None:
    """The log is a formatting of this, so what is asserted is the layout and not an f-string."""
    plan = causal_masks["channel_plan"]
    layout = pipeline.describe_layout(causal_masks, torch.device("cuda:3"))

    assert layout["transform"] == "causal"
    assert layout["device"] == "cuda:3"
    assert layout["widths"] == dict(EXPECTED_CAUSAL_WIDTHS, fhr_up_ph=None)
    assert layout["c_y"] == plan["fhr_st"].n_channels + plan["fhr_ph"].n_channels
    assert layout["c_u"] == plan["up_st"].n_channels + plan["up_ph"].n_channels
    assert layout["gammatone_order"] == 4
    assert layout["causal_kernel_taps"] == 1 << 15
    assert layout["causal_warmup_quantile"] == pytest.approx(0.95)

    # Seven never-valid channels leave each scattering block and nothing leaves either phase block,
    # which is what makes the drop a channel-axis operation rather than a re-selection.
    for field in ("fhr_st", "up_st"):
        assert layout["dropped"][field] == {"count": 7, "first": 36, "last": 42}, field
    for field in ("fhr_ph", "up_ph"):
        assert layout["dropped"][field] == {"count": 0, "first": None, "last": None}, field

    for field in EXPECTED_CAUSAL_WIDTHS:
        assert layout["warmup_steps"][field] == (
            int(plan[field].warmup_steps.min()), int(plan[field].warmup_steps.max())
        ), field
        assert layout["delay_s"][field] == pytest.approx(
            (float(plan[field].delay_s.min()), float(plan[field].delay_s.max()))
        ), field


def test_the_two_sided_layout_keeps_its_content_and_gains_the_variant(
    pipeline: Any, masks: Dict[str, Any]
) -> None:
    """A two-sided run reports what it reported before, plus which variant it is."""
    layout = pipeline.describe_layout(masks)
    assert layout["transform"] == "two_sided"
    assert layout["device"] == "default"
    assert layout["widths"] == dict(EXPECTED_WIDTHS)
    assert layout["c_y"] == 43 + 66 and layout["c_u"] == 43 + 15
    # Causal-only keys stay off a two-sided layout rather than arriving as None.
    assert "dropped" not in layout and "warmup_steps" not in layout


def test_no_channel_count_is_a_literal_in_the_formatted_layout(
    pipeline: Any, masks: Dict[str, Any], causal_masks: Dict[str, Any]
) -> None:
    """The same formatter over two variants prints two different sets of numbers.

    The previous log carried ``43`` inside its f-string, twice; a formatter that still did would
    print ``43`` for the causal layout here.
    """
    causal_lines = "\n".join(pipeline.format_layout(pipeline.describe_layout(causal_masks)))
    two_sided_lines = "\n".join(pipeline.format_layout(pipeline.describe_layout(masks)))

    assert "fhr_st=36" in causal_lines and "c_y=102" in causal_lines
    assert "c_u=51" in causal_lines and "fhr_up_ph=absent" in causal_lines
    assert "43" not in causal_lines
    assert "fhr_st=43" in two_sided_lines and "fhr_up_ph=79" in two_sided_lines
    assert "36" not in two_sided_lines

    assert "gammatone n=4" in causal_lines and "32768 taps" in causal_lines
    assert "Dropped 7 never-valid channels from fhr_st (channels 36..42)" in causal_lines
    assert "Warm-up range: fhr_st 5..293" in causal_lines
    assert "Group delay:" in causal_lines
    assert "Device: default" in two_sided_lines


# =================================================================================================
# Fixture provenance
# =================================================================================================
@requires_shard
def test_the_fixture_segments_are_verbatim_shard_rows(
    raw_segments: Dict[str, np.ndarray], fixture_provenance: Dict[str, Any]
) -> None:
    """On a machine that has the shard, the committed fixture is provably an extract of it.

    The fixture records the rows it was taken from, so this is the check that keeps that record
    honest -- and the reason the fixture can be regenerated from its own attributes alone.
    """
    indices: List[int] = [int(i) for i in fixture_provenance["source_indices"]]
    with h5py.File(SHARD_PATH, "r") as handle:
        assert np.array_equal(raw_segments["fhr"], handle["fhr"][indices])
        assert np.array_equal(raw_segments["up"], handle["up"][indices])


def test_the_causal_file_records_which_operator_built_its_phase_blocks(
    pipeline: Any, causal_file: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    r"""``causal_leg_alignment``, on the root, on the causal variant only.

    Every other constant a causal file records -- the kernel length, the gamma order, the warm-up
    quantile, the widths, the per-channel warm-up and delay -- is either visible in the data's
    shape or recoverable from another. The leg alignment is not: an envelope-aligned file has
    exactly the widths, warm-ups and delays of an unaligned one and differs only in the values
    inside its two phase blocks. If this attribute is missing, nothing anywhere can tell the two
    apart, and a file list may mix them or a stats file normalise one with the other's constants.
    """
    with h5py.File(causal_file, "r") as handle:
        assert handle.attrs["causal_leg_alignment"] == "none"

    aligned_masks = pipeline.compute_scattering_masks(
        LEN_SIGNAL, scattering_T=16, device=torch.device("cpu"),
        transform="causal", leg_alignment="envelope",
    )
    aligned = tmp_path_factory.mktemp("aligned") / "causal_aligned.hdf5"
    pipeline.create_hdf5_for_masks(str(aligned), aligned_masks, len_sequence=LEN_SEQUENCE)
    with h5py.File(aligned, "r") as handle:
        assert handle.attrs["causal_leg_alignment"] == "envelope"
        # Everything else about the two files is the same, which is the point of the attribute.
        assert {name: handle[name].shape[1] for name in EXPECTED_CAUSAL_WIDTHS} == (
            EXPECTED_CAUSAL_WIDTHS
        )


def test_every_causal_block_carries_its_novelty_curve(causal_file: Path) -> None:
    r"""``causal_novelty_curve``: per channel, the envelope-mass share within $w$ stored steps.

    Horizon-free by design: the forecast horizon is a model-side choice, so the file tabulates the
    whole curve over the stored segment length and the model looks its own $H + s_c$ up in it. No
    ``causal_novelty_frac`` scalar is written any more -- that attribute baked one horizon into the
    dataset. Both ends of the window-$30$ column are pinned, because a bug that collapsed the
    table to a constant would look plausible at either one alone.
    """
    with h5py.File(causal_file, "r") as handle:
        for field, width in EXPECTED_CAUSAL_WIDTHS.items():
            assert "causal_novelty_frac" not in handle[field].attrs, field
            curve = np.asarray(handle[field].attrs["causal_novelty_curve"])
            assert curve.shape == (width, LEN_SEQUENCE + 1), field
            assert curve.dtype == np.float32, field
            assert (curve >= 0.0).all() and (curve <= 1.0).all(), field
            assert (curve[:, 0] == 0.0).all(), field
            assert (np.diff(curve.astype(np.float64), axis=1) >= -1e-7).all(), field

        # At window 30 (the legacy 120 s horizon): $S_0$ is the low-pass alone and is entirely
        # new; the slowest kept wavelet, at a 791 s composed delay, is almost entirely old.
        scattering = np.asarray(handle["fhr_st"].attrs["causal_novelty_curve"])[:, 30]
        assert float(scattering[0]) == pytest.approx(1.000, abs=5e-4)
        assert float(scattering[-1]) < 0.01
        # A phase channel takes its slow leg's value, so both phase blocks bottom out at the
        # reference channel's share rather than at their own.
        for field in ("fhr_ph", "up_ph"):
            assert float(
                np.asarray(handle[field].attrs["causal_novelty_curve"])[:, 30].min()
            ) == pytest.approx(0.026, abs=5e-3), field


def test_a_two_sided_file_carries_neither_new_causal_attribute(two_sided_file: Path) -> None:
    """Both are gated on the causal path, which the exact-key-set tests above already enforce.

    Stated separately because those tests read as being about ``sel_*`` provenance, and a future
    reader adding a third causal attribute needs one place that says the gating is the rule rather
    than an accident of which keys happened to be listed.
    """
    with h5py.File(two_sided_file, "r") as handle:
        assert "causal_leg_alignment" not in handle.attrs
        for field in EXPECTED_WIDTHS:
            assert "causal_novelty_frac" not in handle[field].attrs, field
            assert "causal_novelty_curve" not in handle[field].attrs, field


def test_an_unknown_leg_alignment_is_refused_before_anything_is_created(
    pipeline: Any, tmp_path: Path
) -> None:
    """Refused at mask time, not at transform time.

    The mode reaches a root attribute as well as the operator, so a typo caught inside the
    transform would already have created a directory of files claiming to be something they are
    not -- and a build that runs for hours before saying so.
    """
    with pytest.raises(ValueError, match="unknown leg_alignment 'envelop'"):
        pipeline.compute_scattering_masks(
            LEN_SIGNAL, scattering_T=16, device=torch.device("cpu"),
            transform="causal", leg_alignment="envelop",
        )
    assert not list(tmp_path.iterdir())


def test_the_novelty_attribute_must_describe_the_block_it_sits_on(
    pipeline: Any, causal_masks: Dict[str, Any], tmp_path: Path
) -> None:
    """A curve with the wrong row count would attribute one channel's novelty to another.

    It is read back as a parallel array to the warm-up, so a row disagreement is silent at
    every later step: the numbers are plausible, the shapes broadcast, and the split by novelty
    tertile would simply be a split by the wrong channels.
    """
    wrong = dict(causal_masks)
    wrong["causal_novelty_curve"] = dict(causal_masks["causal_novelty_curve"])
    wrong["up_ph"] = None  # unused key; present only to prove the copy is not the original
    wrong["causal_novelty_curve"]["up_ph"] = np.zeros((3, LEN_SEQUENCE + 1), dtype=np.float64)

    with pytest.raises(ValueError, match=r"novelty curve for 'up_ph' has shape \(3, "):
        pipeline.create_hdf5_for_masks(
            str(tmp_path / "bad_novelty.hdf5"), wrong, len_sequence=LEN_SEQUENCE
        )
