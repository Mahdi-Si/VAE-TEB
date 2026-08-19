r"""The loading stack against a real causal shard: validity, coherence, and the mask.

The input is the file :mod:`hdf5_dataset.tests.test_causal_pipeline` writes through the real
writer functions, so what is loaded here is the thing a production build produces rather than a
hand-assembled approximation of it. A two-sided shard is built beside it from the same fixture
signals, which is what makes every "unchanged on a two-sided file" claim below a comparison rather
than an assertion about one file.

What this file is really guarding
---------------------------------
Three failure modes, all of them silent:

* **A channel that is never valid.** Its column normalises to zeros that a model cannot tell from
  real coefficients, so it must raise rather than load.
* **A file list that does not cohere.** Mixed variants are accepted silently today and fail much
  later in two different ways -- ``default_collate`` raises something opaque on the first mixed
  batch, while the sequence dataset iterates the first segment's keys and simply **drops** the
  field the other file did not have.
* **A mask that is not a mask.** ``_create_tensor`` casts through float32 and the collate default
  is a float; either one turns a boolean validity mask into numbers that are all truthy.

The statistics section adds two more of the same kind: constants accumulated over the invalid
region, which no exception anywhere would report, and a stats file paired with the wrong variant,
which today would either normalise causal data with two-sided constants or -- worse -- be caught by
the calculator's broad ``except`` and degrade the run to *unnormalised* data.

Legacy files -- every dataset currently on disk, none of which carries a ``transform`` attribute --
must load exactly as they do today, silently. That is asserted here too, because "absence is
normal" is the kind of rule that decays into a warning on every load.
"""
from __future__ import annotations

import pickle
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pytest
import torch

from hdf5_dataset.calculate_dataset_stats import DatasetStatsCalculator
from hdf5_dataset.hdf5_dataset import (
    CAUSAL,
    TWO_SIDED,
    CombinedHDF5Dataset,
    _COEFFICIENT_FIELDS,
    decimated_trim_steps,
    read_causal_warmup,
    resolve_transform,
)
from hdf5_dataset.guid_hdf5_dataset import SignalSequenceDataset, sequence_collate_fn

from hdf5_dataset.tests.test_causal_pipeline import (
    EXPECTED_CAUSAL_WIDTHS,
    EXPECTED_WIDTHS,
    LEN_SEQUENCE,
    write_causal_shard,
)

#: ``trim_minutes=1.0`` discards $4 \times 60 = 240$ raw samples, i.e. $15$ decimated steps, from
#: each end -- leaving the $300$-step window the models are configured for.
TRIM_MINUTES = 1.0
TRIM_STEPS = 15
KEPT_STEPS = LEN_SEQUENCE - 2 * TRIM_STEPS

#: The measured extremes of the stored causal warm-up, and what they leave after rebasing. The
#: slowest surviving channel keeps $22$ steps of $300$ -- $7\%$ of the window, which is thin
#: enough that a silent change to it must fail a test rather than move a training curve.
SLOWEST_STORED_WARMUP = 293
SLOWEST_REBASED_WARMUP = SLOWEST_STORED_WARMUP - TRIM_STEPS
SLOWEST_VALID_STEPS = KEPT_STEPS - SLOWEST_REBASED_WARMUP


# =================================================================================================
# Shards under test
# =================================================================================================
@pytest.fixture(scope="module")
def causal_shard(
    pipeline: Any,
    causal_masks: Dict[str, Any],
    raw_segments: Dict[str, np.ndarray],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A populated causal shard, written through the real writer functions."""
    return write_causal_shard(
        pipeline,
        causal_masks,
        raw_segments,
        tmp_path_factory.mktemp("loader_causal") / "causal.hdf5",
    )


@pytest.fixture(scope="module")
def two_sided_shard(
    pipeline: Any,
    masks: Dict[str, Any],
    st_model: Any,
    raw_segments: Dict[str, np.ndarray],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A populated two-sided shard from the same signals, as the comparison arm."""
    from hdf5_dataset.tests.test_causal_pipeline import _forward_blocks, _metadata

    path = tmp_path_factory.mktemp("loader_two_sided") / "two_sided.hdf5"
    fhr, up = raw_segments["fhr"], raw_segments["up"]
    n_samples = fhr.shape[0]
    pipeline.create_hdf5_for_masks(str(path), masks, len_sequence=LEN_SEQUENCE)

    blocks = _forward_blocks(st_model, fhr, up, masks)
    meta = _metadata(n_samples)
    weight = np.ones((n_samples, LEN_SEQUENCE), dtype=np.float32)
    pipeline.append_samples_batch(
        path=str(path),
        fhr_batch=fhr, up_batch=up,
        fhr_st_batch=blocks["fhr_st"], fhr_ph_batch=blocks["fhr_ph"],
        fhr_up_ph_batch=blocks["fhr_up_ph"],
        target_batch=weight, weight_batch=weight,
        guid_batch=meta["guid"], epoch_batch=meta["epoch"],
        cs_label_batch=meta["cs_label"], bg_label_batch=meta["bg_label"],
        tlo_batch=meta["time_from_labor_onset"],
        second_stage_batch=meta["second_stage_onset"],
        up_st_batch=blocks["up_st"], up_ph_batch=blocks["up_ph"],
    )
    return path


@pytest.fixture(scope="module")
def legacy_shard(two_sided_shard: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same two-sided shard with its ``transform`` attribute stripped.

    This is what every dataset currently on disk looks like: the attribute postdates them. It is
    produced by deleting the attribute rather than by an older writer, so the arrays are provably
    identical and any behavioural difference is attributable to the attribute alone.
    """
    path = tmp_path_factory.mktemp("loader_legacy") / "legacy.hdf5"
    shutil.copyfile(two_sided_shard, path)
    with h5py.File(path, "a") as handle:
        del handle.attrs["transform"]
    return path


def _dataset(path: Path, **kwargs: Any) -> CombinedHDF5Dataset:
    """A dataset over one shard with the caches off, so each test reads what is on disk."""
    options: Dict[str, Any] = dict(
        paths=[str(path)], cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES
    )
    options.update(kwargs)
    return CombinedHDF5Dataset(**options)


# =================================================================================================
# Variant resolution
# =================================================================================================
def test_a_file_without_the_attribute_is_a_legacy_two_sided_file() -> None:
    """The default that keeps every existing dataset readable, at the resolver itself."""
    assert resolve_transform({}) == TWO_SIDED
    assert resolve_transform({"transform": "causal"}) == CAUSAL
    # h5py hands back bytes for some string attributes depending on how they were written.
    assert resolve_transform({"transform": b"causal"}) == CAUSAL


def test_loading_a_legacy_file_behaves_exactly_as_it_does_today(
    legacy_shard: Path, two_sided_shard: Path, recwarn: pytest.WarningsRecorder
) -> None:
    """No warm-up, no mask, and above all **no warning**.

    Reading a file without a ``transform`` attribute is the common case, not a degraded one. A
    "assuming two-sided" line on every load would be noise on every existing workflow, so its
    absence is pinned rather than assumed.
    """
    dataset = _dataset(legacy_shard)
    assert dataset.transform == TWO_SIDED
    assert dataset.causal_warmup_steps is None
    assert [str(w.message) for w in recwarn if "transform" in str(w.message)] == []

    with pytest.raises(ValueError, match="needs a causal dataset"):
        dataset.channel_valid_mask("fhr_st")

    # And it is the same data as the attributed file, so nothing above came from a different shard.
    attributed = _dataset(two_sided_shard)
    assert torch.equal(dataset[0]["fhr_st"], attributed[0]["fhr_st"])


def test_a_causal_file_resolves_and_reports_its_widths(causal_shard: Path) -> None:
    """The layout read from attributes and shapes alone, before any sample is served."""
    dataset = _dataset(causal_shard)
    assert dataset.transform == CAUSAL
    assert dataset._layout is not None
    assert dataset._layout.widths == dict(EXPECTED_CAUSAL_WIDTHS)
    assert "fhr_up_ph" not in dataset._layout.widths


# =================================================================================================
# Warm-up and the valid mask
# =================================================================================================
def test_the_warm_up_is_rebased_for_the_trim(causal_shard: Path) -> None:
    r"""$W' = \max(W - 15, 0)$ at ``trim_minutes=1.0``, against the stored untrimmed vector.

    The stored attribute describes a $330$-step segment; the loader serves $300$ of them, starting
    $15$ in. A loader that forwarded the stored number would overstate every channel's warm-up by
    exactly the trim, and one that stored it trimmed would be wrong for every other trim.
    """
    dataset = _dataset(causal_shard)
    rebased = dataset.causal_warmup_steps
    assert rebased is not None

    with h5py.File(causal_shard, "r") as handle:
        stored = {
            field: np.asarray(handle[field].attrs["causal_warmup_steps"], dtype=np.int64)
            for field in EXPECTED_CAUSAL_WIDTHS
        }

    for field, vector in stored.items():
        assert np.array_equal(rebased[field], np.maximum(vector - TRIM_STEPS, 0)), field
    assert int(stored["fhr_st"].max()) == SLOWEST_STORED_WARMUP
    assert int(rebased["fhr_st"].max()) == SLOWEST_REBASED_WARMUP


def test_the_untrimmed_dataset_reports_the_stored_warm_up_verbatim(causal_shard: Path) -> None:
    """With no trim there is nothing to rebase, which is the other half of the same rule."""
    dataset = _dataset(causal_shard, trim_minutes=None)
    rebased = dataset.causal_warmup_steps
    assert rebased is not None
    with h5py.File(causal_shard, "r") as handle:
        stored = np.asarray(handle["fhr_st"].attrs["causal_warmup_steps"], dtype=np.int64)
    assert np.array_equal(rebased["fhr_st"], stored)


def test_the_returned_warm_up_cannot_be_mutated_from_outside(causal_shard: Path) -> None:
    """The property hands out copies: a caller that scales it must not resize the valid region."""
    dataset = _dataset(causal_shard)
    first = dataset.causal_warmup_steps
    assert first is not None
    first["fhr_st"] += 1000

    second = dataset.causal_warmup_steps
    assert second is not None
    assert int(second["fhr_st"].max()) == SLOWEST_REBASED_WARMUP


def test_the_valid_mask_is_the_warm_up_in_the_models_layout(causal_shard: Path) -> None:
    r"""$(T, C)$ bool, matching the transposed data, and ``True`` exactly from $W'$ onwards."""
    dataset = _dataset(causal_shard)
    rebased = dataset.causal_warmup_steps
    assert rebased is not None

    for field, width in EXPECTED_CAUSAL_WIDTHS.items():
        mask = dataset.channel_valid_mask(field)
        assert mask.shape == (KEPT_STEPS, width), field
        assert mask.dtype == torch.bool, field
        expected = np.arange(KEPT_STEPS)[:, None] >= rebased[field][None, :]
        assert np.array_equal(mask.numpy(), expected), field
        # The sample's own data has the same (T, C) axes, so the mask applies without transposing.
        assert dataset[0][field].shape == mask.shape, field

    # The thin channel, pinned by value: 22 valid steps of 300.
    assert int(dataset.channel_valid_mask("fhr_st")[:, -1].sum()) == SLOWEST_VALID_STEPS


def test_the_valid_mask_is_built_once_and_reused(causal_shard: Path) -> None:
    """It is a filter-bank constant, identical for every sample; rebuilding it per call is waste."""
    dataset = _dataset(causal_shard)
    assert dataset.channel_valid_mask("fhr_st") is dataset.channel_valid_mask("fhr_st")


def test_a_block_this_dataset_does_not_store_is_named_in_the_refusal(causal_shard: Path) -> None:
    """``fhr_up_ph`` is absent from a causal file, and asking for its mask says so."""
    dataset = _dataset(causal_shard)
    with pytest.raises(ValueError, match="fhr_up_ph"):
        dataset.channel_valid_mask("fhr_up_ph")


def test_a_channel_with_no_valid_step_is_refused_at_index_build(
    causal_shard: Path, tmp_path: Path
) -> None:
    """An all-invalid channel normalises to zeros a model cannot distinguish from coefficients.

    It cannot arise at the shipped geometry -- the drop rule removes those channels at write time,
    and the slowest survivor keeps 22 steps -- which is exactly why it would go unnoticed if it
    ever started to. The refusal names the block and the channel so the cause is not a search.
    """
    path = tmp_path / "dead_channel.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        warmup = np.asarray(handle["fhr_st"].attrs["causal_warmup_steps"])
        warmup[7] = LEN_SEQUENCE  # never leaves the pad-dominated region at any trim
        handle["fhr_st"].attrs["causal_warmup_steps"] = warmup

    with pytest.raises(ValueError, match=r"'fhr_st' channel 7 has no valid step"):
        _dataset(path)


# =================================================================================================
# Warm-up attributes are all-or-nothing
# =================================================================================================
def test_a_causal_file_missing_a_warm_up_vector_is_refused(
    causal_shard: Path, tmp_path: Path
) -> None:
    """Half-attributed is worse than unattributed: some blocks would get a valid region, some not."""
    path = tmp_path / "half_attributed.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        del handle["up_ph"].attrs["causal_warmup_steps"]

    with pytest.raises(ValueError, match=r"'up_ph' block carries no causal_warmup_steps"):
        _dataset(path)


def test_a_warm_up_vector_of_the_wrong_length_is_refused(
    causal_shard: Path, tmp_path: Path
) -> None:
    """The attribute would then describe a different channel axis than the data it sits on."""
    path = tmp_path / "wrong_length.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        handle["fhr_ph"].attrs["causal_warmup_steps"] = np.arange(5, dtype=np.int32)

    with pytest.raises(ValueError, match="describes a different channel axis"):
        _dataset(path)


# =================================================================================================
# Coherence of a file list
# =================================================================================================
def test_a_mixed_variant_list_is_refused_naming_both_files(
    causal_shard: Path, two_sided_shard: Path
) -> None:
    """The refusal that replaces an opaque collate failure and a silently dropped field."""
    with pytest.raises(ValueError, match="Mixed transform variants") as error:
        CombinedHDF5Dataset(
            paths=[str(two_sided_shard), str(causal_shard)],
            cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
        )
    message = str(error.value)
    assert "causal" in message and "two_sided" in message
    assert causal_shard.name in message and two_sided_shard.name in message


def test_a_legacy_file_beside_a_two_sided_one_is_accepted(
    legacy_shard: Path, two_sided_shard: Path
) -> None:
    """Variants are compared **resolved**, so this coherent combination keeps working.

    Comparing the raw attribute instead would refuse a list mixing a shard built last year with
    one built today -- the same data, the same widths, and a workflow that works now.
    """
    dataset = CombinedHDF5Dataset(
        paths=[str(legacy_shard), str(two_sided_shard)],
        cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
    )
    assert dataset.transform == TWO_SIDED
    assert len(dataset) == 16


def test_a_list_disagreeing_on_a_block_width_is_refused(
    causal_shard: Path, tmp_path: Path
) -> None:
    """Same variant, same blocks, different channel axis -- which collates into garbage."""
    narrow = tmp_path / "narrow.hdf5"
    shutil.copyfile(causal_shard, narrow)
    # One block re-written narrower, which no writer would produce; the point is that the loader
    # notices a disagreement rather than that this particular file is reachable.
    with h5py.File(narrow, "a") as handle:
        data = np.asarray(handle["up_ph"][:, :10, :])
        warmup = np.asarray(handle["up_ph"].attrs["causal_warmup_steps"])[:10]
        delay = np.asarray(handle["up_ph"].attrs["causal_delay_s"])[:10]
        del handle["up_ph"]
        block = handle.create_dataset("up_ph", data=data, maxshape=(None, 10, LEN_SEQUENCE))
        block.attrs["causal_warmup_steps"] = warmup
        block.attrs["causal_delay_s"] = delay

    with pytest.raises(ValueError, match=r"Mismatched 'up_ph' width"):
        CombinedHDF5Dataset(
            paths=[str(causal_shard), str(narrow)],
            cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
        )


def test_a_list_disagreeing_on_the_block_set_is_refused(
    two_sided_shard: Path, tmp_path: Path
) -> None:
    """A file with an extra block makes ``__getitem__`` return different keys per sample.

    Both files are marked two-sided so the variant check cannot be what fires; what differs is the
    block set alone.
    """
    path = tmp_path / "no_cross.hdf5"
    shutil.copyfile(two_sided_shard, path)
    with h5py.File(path, "a") as handle:
        del handle["fhr_up_ph"]

    with pytest.raises(ValueError, match="Mismatched coefficient blocks") as error:
        CombinedHDF5Dataset(
            paths=[str(two_sided_shard), str(path)],
            cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
        )
    assert "fhr_up_ph" in str(error.value)


# =================================================================================================
# Reading the boundary without building a loader
# =================================================================================================
# A model resolves which channels it may use *before* it is constructed -- its input widths depend
# on the answer -- so it needs the boundary without a dataset, a stats file or a single sample
# read. The three properties that matter are that it is the same boundary the loader serves, that
# every configured file is checked rather than the first, and that a file list which does not
# describe one dataset is refused instead of resolved from whichever shard came first.
def test_the_boundary_read_without_a_loader_is_the_one_the_loader_serves(
    causal_shard: Path
) -> None:
    """Same function, same trim, same vectors -- which is the whole reason it is shared."""
    dataset = _dataset(causal_shard)
    served = dataset.causal_warmup_steps
    assert served is not None

    read = read_causal_warmup([str(causal_shard)], TRIM_MINUTES)
    assert sorted(read.warmup_steps) == sorted(served)
    for field, vector in served.items():
        assert np.array_equal(read.warmup_steps[field], vector), field
    assert read.trim_steps == TRIM_STEPS
    assert set(read.kept_steps.values()) == {KEPT_STEPS}

    with h5py.File(causal_shard, "r") as handle:
        assert read.quantile == pytest.approx(float(handle.attrs["causal_warmup_quantile"]))


def test_the_delay_is_the_shard_s_own_attribute_unrebased(causal_shard: Path) -> None:
    r"""The group delay arrives beside the warm-up, in seconds, untouched by the trim.

    A delay is not a step index. The warm-up says which steps of *this window* are usable and
    therefore moves when the window does; the delay says how far back in physical time a
    coefficient's content sits, which no trim can change. Rebasing it the way the warm-up is
    rebased would silently shorten every delay by a minute at ``trim_minutes=1.0``, and the shift
    vectors a consumer resolves from it would all be wrong by the same amount -- with every shape
    correct and nothing to report it.
    """
    read = read_causal_warmup([str(causal_shard)], TRIM_MINUTES)
    untrimmed = read_causal_warmup([str(causal_shard)], None)

    assert sorted(read.delay_s) == sorted(read.warmup_steps)
    with h5py.File(causal_shard, "r") as handle:
        for field, vector in read.delay_s.items():
            stored = np.asarray(handle[field].attrs["causal_delay_s"], dtype=np.float64)
            assert np.array_equal(vector, stored), field
            assert vector.shape == (handle[field].shape[1],), field
            # Unrebased: the same numbers at both trims, unlike the warm-up beside them.
            assert np.array_equal(untrimmed.delay_s[field], vector), field

    assert not np.array_equal(
        untrimmed.warmup_steps["fhr_st"], read.warmup_steps["fhr_st"]
    )
    # Seconds, and the published extremes of the composed one-sided delay.
    assert float(read.delay_s["fhr_st"].min()) == pytest.approx(13.30, abs=0.05)
    assert float(read.delay_s["fhr_st"].max()) == pytest.approx(791.02, abs=0.05)


def test_a_causal_file_missing_the_delay_is_refused_naming_the_block(
    causal_shard: Path, tmp_path: Path
) -> None:
    """A consumer that aligns channels has no other source for the number.

    Required by this reader and merely read where present by the loader: nothing on the loading
    path compensates for a delay, so refusing there would take a shard offline for a field no
    sample read uses.
    """
    path = tmp_path / "no_delay.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        del handle["up_ph"].attrs["causal_delay_s"]

    with pytest.raises(ValueError, match=r"'up_ph' block carries no causal_delay_s") as error:
        read_causal_warmup([str(path)], TRIM_MINUTES)
    assert path.name in str(error.value)
    # The loader itself is unmoved by the same file, which is the asymmetry being asserted.
    assert _dataset(path).transform == CAUSAL


def test_files_disagreeing_on_the_delay_are_refused_naming_one(
    causal_shard: Path, tmp_path: Path
) -> None:
    """It is a constant of the filter bank, exactly as the warm-up is.

    The two shards have identical widths, identical warm-ups and identical quantiles, so this is
    the only check that can fire -- which is the point: a shard rebuilt at a different bank whose
    delays alone moved would otherwise resolve one set of alignment shifts and be served another.
    """
    path = tmp_path / "shifted_delay.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        delay = np.asarray(handle["fhr_ph"].attrs["causal_delay_s"])
        delay[5] += np.float32(1.0)
        handle["fhr_ph"].attrs["causal_delay_s"] = delay

    with pytest.raises(ValueError, match=r"Mismatched 'fhr_ph' causal_delay_s") as error:
        read_causal_warmup([str(causal_shard), str(path)], TRIM_MINUTES)
    assert path.name in str(error.value)


def test_the_trim_conversion_is_the_loader_s_own(causal_shard: Path) -> None:
    """A consumer that rounded the trim differently would rebase against a window nothing serves."""
    dataset = _dataset(causal_shard)
    assert decimated_trim_steps(TRIM_MINUTES) == (
        dataset.trim_samples_raw,
        dataset.trim_samples_decimated,
    )
    assert decimated_trim_steps(TRIM_MINUTES) == (TRIM_STEPS * 16, TRIM_STEPS)
    assert decimated_trim_steps(None) == (0, 0)


@pytest.mark.parametrize("order", ("causal_first", "two_sided_first"))
def test_every_file_is_read_rather_than_the_first(
    causal_shard: Path, two_sided_shard: Path, order: str
) -> None:
    """A two-sided held-out shard beside causal training shards is the motivating case.

    Resolved off the first file alone it comes out clean either way, and the held-out numbers are
    then produced against a channel axis that means something else.
    """
    paths = [str(causal_shard), str(two_sided_shard)]
    if order == "two_sided_first":
        paths.reverse()
    with pytest.raises(ValueError, match=TWO_SIDED):
        read_causal_warmup(paths, TRIM_MINUTES)


def test_files_built_at_different_quantiles_are_refused_naming_one(
    causal_shard: Path, tmp_path: Path
) -> None:
    """The quantile sets the warm-up *and* which channels survive the build: it is a channel axis.

    Two files that disagree about it are two datasets, and a budget resolved from one of them
    describes a boundary the other does not have.
    """
    path = tmp_path / "other_quantile.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        handle.attrs["causal_warmup_quantile"] = np.float32(0.99)

    with pytest.raises(ValueError, match="causal_warmup_quantile") as error:
        read_causal_warmup([str(causal_shard), str(path)], TRIM_MINUTES)
    assert path.name in str(error.value)


def test_files_disagreeing_on_the_warm_up_itself_are_refused(
    causal_shard: Path, tmp_path: Path
) -> None:
    """It is a constant of the filter bank, so two shards that disagree had two banks."""
    path = tmp_path / "shifted_warmup.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        warmup = np.asarray(handle["fhr_ph"].attrs["causal_warmup_steps"])
        warmup[5] += 1
        handle["fhr_ph"].attrs["causal_warmup_steps"] = warmup

    with pytest.raises(ValueError, match=r"Mismatched 'fhr_ph' causal_warmup_steps") as error:
        read_causal_warmup([str(causal_shard), str(path)], TRIM_MINUTES)
    assert path.name in str(error.value)


def test_files_of_different_stored_length_are_refused_naming_the_block(
    causal_shard: Path, tmp_path: Path
) -> None:
    """A shorter shard beside a longer one resolves the whole boundary against the wrong window.

    This reader rebases against the first file alone and reports one ``kept_steps`` per block, so
    a test shard built to a different segment length would have the budget, the dead-channel
    refusal and a consumer's own trim cross-check all computed against a window it does not have
    -- and the shorter shard would then be served channels whose warm-up outruns it, with every
    valid-region readout still reporting a full one. The loader's own scan tolerates this because
    it serves one sample at a time and never shares a length between files.
    """
    path = tmp_path / "shorter_segments.hdf5"
    with h5py.File(causal_shard, "r") as source:
        block = np.asarray(source["fhr_ph"][:])
    with h5py.File(causal_shard, "r") as source, h5py.File(path, "w") as handle:
        for name, item in source.items():
            handle.create_dataset(
                name, data=item[..., :-1] if name == "fhr_ph" else item[...]
            )
            handle[name].attrs.update(item.attrs)
        handle.attrs.update(source.attrs)

    with pytest.raises(ValueError, match=r"Mismatched 'fhr_ph' stored length") as error:
        read_causal_warmup([str(causal_shard), str(path)], TRIM_MINUTES)
    assert path.name in str(error.value)
    # The probe is not vacuous: the same pair at equal length resolves without complaint.
    assert block.shape[2] == read_causal_warmup([str(causal_shard)], None).kept_steps["fhr_ph"]


def test_a_missing_file_is_refused_rather_than_skipped(
    causal_shard: Path, tmp_path: Path
) -> None:
    """Stricter than the loader's own scan, and deliberately.

    The loader tolerates a short list because it still has samples to serve. A boundary resolved
    from a subset of the configured shards is simply the wrong boundary, and nothing downstream
    would say so.
    """
    absent = tmp_path / "never_built.hdf5"
    with pytest.raises(ValueError, match="never_built"):
        read_causal_warmup([str(causal_shard), str(absent)], TRIM_MINUTES)


def test_an_empty_file_list_is_refused() -> None:
    """The boundary is a property of the shards; with none there is nothing to read it from."""
    with pytest.raises(ValueError, match="no files"):
        read_causal_warmup([], TRIM_MINUTES)


# =================================================================================================
# The consolidated field set
# =================================================================================================
def test_the_two_sided_sample_is_unchanged_by_the_field_set_refactor(
    two_sided_shard: Path
) -> None:
    """Trimming, normalisation membership and the transpose all read one definition now.

    Behaviour is what is pinned: every coefficient block trimmed on the time axis and handed back
    as $(T, C)$, the raw signals trimmed on their own axis, and no field gained or lost.
    """
    dataset = _dataset(two_sided_shard)
    sample = dataset[0]

    for field, width in EXPECTED_WIDTHS.items():
        assert field in _COEFFICIENT_FIELDS
        assert sample[field].shape == (KEPT_STEPS, width), field
    assert sample["fhr"].shape == (4800,)
    assert sample["up"].shape == (4800,)
    assert sample["target"].shape == (KEPT_STEPS,)
    assert not any(key.endswith("_valid") for key in sample)


def test_an_absent_cross_phase_block_needs_no_special_case(causal_shard: Path) -> None:
    """``__getitem__`` skips fields the file does not have, before any membership test."""
    sample = _dataset(causal_shard)[0]
    assert "fhr_up_ph" not in sample
    for field, width in EXPECTED_CAUSAL_WIDTHS.items():
        assert sample[field].shape == (KEPT_STEPS, width), field


# =================================================================================================
# Opt-in mask emission
# =================================================================================================
def test_the_mask_is_not_emitted_by_default(causal_shard: Path) -> None:
    """A dataset constant is not worth a per-sample, per-worker, per-collate copy by default."""
    sample = _dataset(causal_shard)[0]
    assert not any(key.endswith("_valid") for key in sample)


def test_the_emitted_mask_is_the_cached_one_and_stays_boolean(causal_shard: Path) -> None:
    """It must bypass ``_create_tensor``, normalisation and the transpose to survive as a mask.

    ``_create_tensor`` casts through float32; either that or a normalisation pass would leave a
    tensor of ones and zeros that is entirely truthy, i.e. a mask that masks nothing.
    """
    dataset = _dataset(causal_shard, emit_validity_mask=True)
    sample = dataset[0]
    for field, width in EXPECTED_CAUSAL_WIDTHS.items():
        mask = sample[f"{field}_valid"]
        assert mask.dtype == torch.bool, field
        assert mask.shape == (KEPT_STEPS, width), field
        assert torch.equal(mask, dataset.channel_valid_mask(field)), field
        assert not mask.all(), f"{field}: a mask that is all True masks nothing"


def test_the_mask_follows_a_restricted_field_list(causal_shard: Path) -> None:
    """Only blocks actually loaded get a mask; ``load_fields`` is not overridden from behind."""
    dataset = _dataset(
        causal_shard, emit_validity_mask=True,
        load_fields=["fhr_st", "target", "weight", "guid", "epoch", "cs_label", "bg_label"],
    )
    sample = dataset[0]
    assert "fhr_st_valid" in sample
    assert "up_st_valid" not in sample and "fhr_ph_valid" not in sample


def test_asking_for_a_mask_on_a_two_sided_dataset_is_refused(two_sided_shard: Path) -> None:
    """The only mask a two-sided file could produce is all-True, which asserts something false.

    Refusing at construction is the deliberate choice: emitting nothing would surface later as an
    ``AttributeError`` on ``sample.fhr_st_valid`` with nothing saying why.
    """
    with pytest.raises(ValueError, match="needs a causal dataset"):
        _dataset(two_sided_shard, emit_validity_mask=True)


def test_the_mask_cache_survives_pickling_to_a_worker(causal_shard: Path) -> None:
    """Kept rather than dropped: it is a small immutable constant, identical in every worker."""
    dataset = _dataset(causal_shard, emit_validity_mask=True)
    _ = dataset.channel_valid_mask("fhr_st")
    revived = pickle.loads(pickle.dumps(dataset))

    assert revived._valid_mask_cache, "the mask cache was dropped on the way to the worker"
    assert torch.equal(
        revived.channel_valid_mask("fhr_st"), dataset.channel_valid_mask("fhr_st")
    )
    # The sample cache, by contrast, is deliberately cleared.
    assert revived._cache == {}


# =================================================================================================
# Sequence stacking and padding
# =================================================================================================
#: Segments per GUID in the sequence fixture. Uneven on purpose: the collate path pads to the
#: longest sequence in the batch, so a shard whose GUIDs all have one segment would let every
#: padding assertion below pass without a single padded position ever existing.
SEQUENCE_GROUPS = (5, 2, 1)


@pytest.fixture(scope="module")
def sequence_shard(causal_shard: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The causal shard regrouped so its eight segments belong to three recordings.

    The writer gives each fixture segment its own GUID, which is right for a per-segment test and
    useless for a per-sequence one. Only ``guid`` and ``epoch`` are rewritten; every coefficient
    stays exactly what the writer produced.
    """
    path = tmp_path_factory.mktemp("loader_sequence") / "sequence.hdf5"
    shutil.copyfile(causal_shard, path)

    guids: List[str] = []
    epochs: List[float] = []
    for group, count in enumerate(SEQUENCE_GROUPS):
        for position in range(count):
            guids.append(f"SEQ{group:02d}")
            epochs.append(-40000.0 + 1200.0 * position)
    with h5py.File(path, "a") as handle:
        assert handle["guid"].shape[0] == len(guids), "the fixture segment count changed"
        handle["guid"][:] = guids
        handle["epoch"][:] = np.asarray(epochs, dtype=np.float32)
    return path


@pytest.fixture(scope="module")
def sequence_dataset(sequence_shard: Path) -> SignalSequenceDataset:
    """The per-GUID view of the causal shard, with masks emitted."""
    return SignalSequenceDataset(
        paths=[str(sequence_shard)], cache_size=0, guid_cache_size=0, pin_memory=False,
        trim_minutes=TRIM_MINUTES, emit_validity_mask=True,
    )


def test_masks_stack_like_any_other_tensor_field(sequence_dataset: SignalSequenceDataset) -> None:
    r"""$(S_i, T, C)$ and still boolean -- ``torch.stack`` must not promote it."""
    sample = sequence_dataset[0]
    n_segments = sample["num_segments"]
    for field, width in EXPECTED_CAUSAL_WIDTHS.items():
        mask = sample[f"{field}_valid"]
        assert mask.shape == (n_segments, KEPT_STEPS, width), field
        assert mask.dtype == torch.bool, field
        assert torch.equal(
            mask[0], sequence_dataset.inner_dataset.channel_valid_mask(field)
        ), field


def test_padded_segment_positions_read_as_invalid(
    sequence_dataset: SignalSequenceDataset
) -> None:
    """Asserted rather than configured.

    ``sequence_collate_fn`` pads with ``_DEFAULT_PAD = 0.0``, and ``torch.full(..., 0.0,
    dtype=torch.bool)`` is ``False`` -- which is what a padded position means. Adding
    ``<block>_valid`` to ``_PAD_VALUES`` would be the fragile option: the natural value to write
    beside ``target`` is ``-1.0``, and ``-1.0`` cast to bool is silently ``True``.
    """
    batch = sequence_collate_fn([sequence_dataset[i] for i in range(len(sequence_dataset))])
    # Asserted, not skipped past: a batch with nothing padded would let every check below pass.
    assert sorted(batch["lengths"].tolist(), reverse=True) == sorted(SEQUENCE_GROUPS, reverse=True)

    for index, length in enumerate(batch["lengths"].tolist()):
        padded = batch["fhr_st_valid"][index, length:]
        assert padded.dtype == torch.bool
        assert not padded.any(), "a padded segment must not claim its channels are valid"
        assert batch["fhr_st_valid"][index, :length].any()


def test_the_batched_mask_keeps_its_dtype_through_collate(
    sequence_dataset: SignalSequenceDataset
) -> None:
    """The dtype is taken from the sample tensors, so a float default cannot leak in."""
    batch = sequence_collate_fn([sequence_dataset[0]])
    assert batch["fhr_st_valid"].dtype == torch.bool
    assert batch["fhr_st_valid"].shape[-2:] == (KEPT_STEPS, EXPECTED_CAUSAL_WIDTHS["fhr_st"])
    assert "fhr_up_ph_valid" not in batch


def test_the_sequence_view_drops_no_field_on_a_causal_shard(
    sequence_dataset: SignalSequenceDataset
) -> None:
    """Every stored block survives the per-GUID grouping, and no cross-phase block appears."""
    sample = sequence_dataset[0]
    stored: List[str] = sorted(EXPECTED_CAUSAL_WIDTHS)
    assert [field for field in stored if field in sample] == stored
    assert "fhr_up_ph" not in sample


# =================================================================================================
# Statistics over the valid region
# =================================================================================================
#: Segments in the fixture, and therefore in every shard built from it.
N_FIXTURE_SEGMENTS = 8


def _calculator() -> DatasetStatsCalculator:
    """A calculator at the loader's trim, on the CPU so the result does not depend on the box."""
    return DatasetStatsCalculator(trim_minutes=TRIM_MINUTES, device="cpu")


def _channel_counts(stats: Dict[str, Any], field: str) -> np.ndarray:
    """How many values entered each channel's sums, as a plain array."""
    return stats[field]["channel_counts"].cpu().numpy()


def _stored_warmup(path: Path, field: str) -> np.ndarray:
    """The block's warm-up as written, in untrimmed steps."""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[field].attrs["causal_warmup_steps"], dtype=np.int64)


@pytest.fixture(scope="module")
def causal_stats(causal_shard: Path) -> Dict[str, Any]:
    """Statistics over the causal shard, computed once."""
    return _calculator().calculate_stats([str(causal_shard)], batch_size=4, progress_bar=False)


@pytest.fixture(scope="module")
def two_sided_stats(two_sided_shard: Path) -> Dict[str, Any]:
    """Statistics over the two-sided shard built from the same signals."""
    return _calculator().calculate_stats([str(two_sided_shard)], batch_size=4, progress_bar=False)


def test_per_channel_counts_fall_by_exactly_the_warm_up(
    causal_shard: Path, causal_stats: Dict[str, Any]
) -> None:
    r"""$N_c = n \cdot (T_{\text{trimmed}} - \max(W_c - \text{trim},\ 0))$, channel by channel.

    The headline of the sprint: a channel that is honest about the past for only 22 of its 300
    steps must contribute exactly those 22, not 300. Asserted against the file's own attribute so
    the expectation is not a second copy of the drop rule.
    """
    for field in EXPECTED_CAUSAL_WIDTHS:
        rebased = np.maximum(_stored_warmup(causal_shard, field) - TRIM_STEPS, 0)
        expected = N_FIXTURE_SEGMENTS * (KEPT_STEPS - rebased)
        assert np.array_equal(_channel_counts(causal_stats, field), expected), field

    # The thin channel, by value, so a silent widening of the valid region fails here too.
    assert int(_channel_counts(causal_stats, "fhr_st")[-1]) == N_FIXTURE_SEGMENTS * SLOWEST_VALID_STEPS


def test_a_two_sided_file_still_accumulates_over_the_whole_trimmed_window(
    two_sided_stats: Dict[str, Any]
) -> None:
    """No warm-up attribute, no exclusion: every channel keeps all $300$ steps of every segment."""
    for field, width in EXPECTED_WIDTHS.items():
        counts = _channel_counts(two_sided_stats, field)
        assert counts.shape == (width,), field
        assert np.array_equal(counts, np.full(width, N_FIXTURE_SEGMENTS * KEPT_STEPS)), field


def test_a_legacy_file_produces_exactly_the_same_arrays_as_an_attributed_one(
    legacy_shard: Path, two_sided_stats: Dict[str, Any]
) -> None:
    """The stored arrays of a two-sided stats file are what they always were.

    The legacy shard is the attributed one with ``transform`` deleted, so identical means the
    attribute changed nothing about the numbers -- which is the compatibility claim itself.
    """
    legacy = _calculator().calculate_stats([str(legacy_shard)], batch_size=4, progress_bar=False)
    for field in EXPECTED_WIDTHS:
        assert np.array_equal(legacy[field]["mean"], two_sided_stats[field]["mean"]), field
        assert np.array_equal(legacy[field]["variance"], two_sided_stats[field]["variance"]), field


def test_the_exclusion_changes_the_constants_it_is_supposed_to_change(
    causal_shard: Path, causal_stats: Dict[str, Any], tmp_path: Path
) -> None:
    """A gate that cannot fail proves nothing: this is the same file with the warm-up zeroed.

    Only the attribute differs, so any difference in the constants is attributable to the exclusion
    and to nothing else. The slowest channel is the one to look at -- it is $278$ of $300$ steps of
    pad-dominated output that would otherwise be averaged in as if it were signal.
    """
    path = tmp_path / "no_warmup.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        for field in EXPECTED_CAUSAL_WIDTHS:
            width = handle[field].shape[1]
            handle[field].attrs["causal_warmup_steps"] = np.zeros(width, dtype=np.int32)

    unmasked = _calculator().calculate_stats([str(path)], batch_size=4, progress_bar=False)
    assert int(_channel_counts(unmasked, "fhr_st")[-1]) == N_FIXTURE_SEGMENTS * KEPT_STEPS
    assert not np.isclose(
        causal_stats["fhr_st"]["mean"][-1], unmasked["fhr_st"]["mean"][-1]
    ), "excluding 278 of 300 steps left the channel's mean unchanged"


def test_a_channel_with_no_valid_step_is_refused_by_the_calculator(
    causal_shard: Path, tmp_path: Path
) -> None:
    """It must raise here for the same reason the loader raises: a zero-count channel gets
    variance $0$, which ``save_stats`` writes as ``std = 0``, which ``normalize_tensor_data``
    divides by $0 + 10^{-8}$ -- inflating that channel by $10^{8}$ with no exception anywhere."""
    path = tmp_path / "dead_channel_stats.hdf5"
    shutil.copyfile(causal_shard, path)
    with h5py.File(path, "a") as handle:
        warmup = np.asarray(handle["up_ph"].attrs["causal_warmup_steps"])
        warmup[3] = LEN_SEQUENCE
        handle["up_ph"].attrs["causal_warmup_steps"] = warmup

    with pytest.raises(ValueError, match=r"'up_ph' channel 3 has no valid step"):
        _calculator().calculate_stats([str(path)], batch_size=4, progress_bar=False)


def test_the_calculator_refuses_a_mixed_variant_file_list(
    causal_shard: Path, two_sided_shard: Path
) -> None:
    """Field shapes come from the first file alone, so a mixed list is a silent $10^{8}$ error.

    Seven channels of a 43-wide accumulator would never be written by a 36-wide file, finish at
    ``count == 0``, and come out of ``_finalize_stats`` with variance $0$.
    """
    with pytest.raises(ValueError, match="Mixed transform variants"):
        _calculator().calculate_stats(
            [str(two_sided_shard), str(causal_shard)], batch_size=4, progress_bar=False
        )


# =================================================================================================
# The histogram collection path
# =================================================================================================
def test_the_histogram_collection_excludes_the_same_region(causal_shard: Path) -> None:
    """``plot_histograms`` collects its own samples, so the exclusion has to reach that loop too.

    Left out, a causal file's normalisation constants would be warm-up aware while the raw
    distribution plotted next to them still contained the invalid region -- two panels of one
    figure describing different data.
    """
    calculator = _calculator()
    _, warmup = calculator._resolve_valid_region([str(causal_shard)])
    assert warmup is not None
    collected = calculator._collect_sample_data([str(causal_shard)], 100, warmup, progress_bar=False)

    with h5py.File(causal_shard, "r") as handle:
        stored = {field: np.asarray(handle[field][:]) for field in EXPECTED_CAUSAL_WIDTHS}

    for field, width in EXPECTED_CAUSAL_WIDTHS.items():
        data = collected[field]
        assert data.shape == (N_FIXTURE_SEGMENTS, width, KEPT_STEPS), field
        blanked = np.isnan(data).sum(axis=(0, 2))
        assert np.array_equal(blanked, N_FIXTURE_SEGMENTS * warmup[field]), field
        # What survives is the stored data itself, untouched beyond the trim and the blanking.
        valid_from = int(warmup[field].max())
        expected = stored[field][:, :, TRIM_STEPS:-TRIM_STEPS][:, :, valid_from:]
        assert np.array_equal(data[:, :, valid_from:], expected), field


def test_the_histogram_collection_is_untouched_on_a_two_sided_file(two_sided_shard: Path) -> None:
    """Nothing is blanked where there is no warm-up, and the raw signals are never blanked."""
    calculator = _calculator()
    _, warmup = calculator._resolve_valid_region([str(two_sided_shard)])
    assert warmup is None
    collected = calculator._collect_sample_data([str(two_sided_shard)], 100, warmup, progress_bar=False)

    for field in list(EXPECTED_WIDTHS) + ["fhr", "up"]:
        assert np.isfinite(collected[field]).all(), field


# =================================================================================================
# Stats provenance and pairing
# =================================================================================================
def _saved_stats(shard: Path, target: Path) -> Path:
    """Compute statistics over *shard* and write them to *target*."""
    calculator = _calculator()
    stats = calculator.calculate_stats([str(shard)], batch_size=4, progress_bar=False)
    calculator.save_stats(stats, str(target))
    return target


@pytest.fixture(scope="module")
def causal_stats_file(causal_shard: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stats file for the causal shard, written through ``save_stats``."""
    return _saved_stats(causal_shard, tmp_path_factory.mktemp("causal_stats") / "stats.hdf5")


@pytest.fixture(scope="module")
def two_sided_stats_file(two_sided_shard: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stats file for the two-sided shard."""
    return _saved_stats(two_sided_shard, tmp_path_factory.mktemp("two_sided_stats") / "stats.hdf5")


def test_a_causal_stats_file_records_the_variant_and_the_warm_up(
    causal_stats_file: Path, causal_shard: Path
) -> None:
    """Enough to tell afterwards which dataset the constants describe, and over what region."""
    with h5py.File(causal_stats_file, "r") as handle:
        assert resolve_transform(handle.attrs) == CAUSAL
        for field in EXPECTED_CAUSAL_WIDTHS:
            written = np.asarray(handle[field].attrs["causal_warmup_steps"], dtype=np.int64)
            # Untrimmed, exactly as the dataset stores it: one name, one meaning, and the file's
            # own trim_minutes is what rebases it.
            assert np.array_equal(written, _stored_warmup(causal_shard, field)), field
        assert "causal_warmup_steps" not in handle["fhr"].attrs


def test_a_two_sided_stats_file_gains_the_variant_and_no_warm_up(
    two_sided_stats_file: Path
) -> None:
    """The additive-attribute half of the compatibility contract, on the stats file."""
    with h5py.File(two_sided_stats_file, "r") as handle:
        assert resolve_transform(handle.attrs) == TWO_SIDED
        for field in EXPECTED_WIDTHS:
            assert "causal_warmup_steps" not in handle[field].attrs, field


def test_load_stats_round_trips_the_variant_and_the_warm_up(
    causal_stats_file: Path, causal_shard: Path
) -> None:
    """Both come back; the variant through the calculator, because every dict key is a group."""
    calculator = _calculator()
    loaded = calculator.load_stats(str(causal_stats_file))
    assert calculator.source_transform == CAUSAL
    for field in EXPECTED_CAUSAL_WIDTHS:
        assert np.array_equal(
            loaded[field]["causal_warmup_steps"], _stored_warmup(causal_shard, field)
        ), field


def test_load_stats_reads_a_file_written_before_these_attributes(
    two_sided_stats_file: Path, tmp_path: Path
) -> None:
    """Every stats file on disk predates them, and reading one is not a degraded case."""
    path = tmp_path / "legacy_stats.hdf5"
    shutil.copyfile(two_sided_stats_file, path)
    with h5py.File(path, "a") as handle:
        del handle.attrs["transform"]

    calculator = _calculator()
    loaded = calculator.load_stats(str(path))
    assert calculator.source_transform == TWO_SIDED
    for field in EXPECTED_WIDTHS:
        assert "causal_warmup_steps" not in loaded[field], field
        assert loaded[field]["mean"].shape == (EXPECTED_WIDTHS[field],), field


def test_a_causal_dataset_normalises_with_its_own_statistics(
    causal_shard: Path, causal_stats_file: Path
) -> None:
    """The pairing that is supposed to work, so the refusals below are not just a blanket refusal."""
    dataset = _dataset(causal_shard, stats_path=str(causal_stats_file))
    assert dataset.normalization_enabled
    sample = dataset[0]
    for field, width in EXPECTED_CAUSAL_WIDTHS.items():
        assert sample[field].shape == (KEPT_STEPS, width), field
        assert torch.isfinite(sample[field]).all(), field


def test_a_causal_dataset_paired_with_two_sided_statistics_is_refused(
    causal_shard: Path, two_sided_stats_file: Path
) -> None:
    """The pairing that would otherwise normalise causal data with two-sided constants.

    It must **raise**: the loader's stats handling wraps everything in ``except Exception -> warn
    and disable normalisation``, so a check inside it would answer a mispairing by training on
    unnormalised data instead.
    """
    with pytest.raises(ValueError, match="variant mismatch") as error:
        _dataset(causal_shard, stats_path=str(two_sided_stats_file))
    assert "causal" in str(error.value) and "two_sided" in str(error.value)


def test_a_legacy_stats_file_beside_a_two_sided_dataset_says_nothing(
    two_sided_shard: Path, legacy_shard: Path, two_sided_stats_file: Path, tmp_path: Path
) -> None:
    """Both resolve to two-sided, which is the common case for everything already on disk."""
    path = tmp_path / "legacy_pairing.hdf5"
    shutil.copyfile(two_sided_stats_file, path)
    with h5py.File(path, "a") as handle:
        del handle.attrs["transform"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all fails this test
        for shard in (two_sided_shard, legacy_shard):
            dataset = _dataset(shard, stats_path=str(path))
            assert dataset.normalization_enabled


def test_statistics_keyed_to_a_different_channel_selection_are_refused(
    causal_shard: Path, two_sided_stats_file: Path, tmp_path: Path
) -> None:
    """Same variant, wrong widths -- which reaches ``normalize_tensor_data`` as a broadcast error
    per sample, from a place that says nothing about the stats file."""
    path = tmp_path / "mislabelled_stats.hdf5"
    shutil.copyfile(two_sided_stats_file, path)
    with h5py.File(path, "a") as handle:
        handle.attrs["transform"] = CAUSAL  # now the variant matches and only the widths do not

    with pytest.raises(ValueError, match=r"width mismatch on 'fhr_st'") as error:
        _dataset(causal_shard, stats_path=str(path))
    assert "43" in str(error.value) and "36" in str(error.value)


# =================================================================================================
# The remaining consumers inside this package
# =================================================================================================
#: Segments the comparison tool is exercised over. Its reference arm is the numpy chain at roughly
#: $1.5$ s a segment per arm, and what is under test is that it reads either variant at all -- a
#: property of the shard, not of the sample -- so one segment is the whole of it.
COMPARED_SEGMENTS = 1


def _compare_module() -> Any:
    """The comparison tool, or a skip where its ``teb_vae`` dependencies are not installed."""
    return pytest.importorskip("hdf5_dataset.compare_causal_scattering")


def test_the_comparison_tool_reads_a_causal_shard(
    causal_shard: Path, causal_bank: Any, bank: Any, phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, Any]
) -> None:
    """Arm A is the shard, so a causal shard is gated against arm C reduced by the channel plan.

    This is the failure this task exists to remove: the gate used to subtract a 43-channel arm from
    a 36-channel stored block, which is a numpy broadcast error raised from the middle of a
    measurement loop.
    """
    compare = _compare_module()
    assert compare.resolve_shard_variant(str(causal_shard)) == CAUSAL

    fhr, up, stored, _ = compare.load_segments(str(causal_shard), range(COMPARED_SEGMENTS))
    validation = compare.measure_validation(
        stored, fhr, up, bank, phase_pairs,
        variant=CAUSAL, causal=causal_bank, plan=channel_plan,
    )

    assert validation["shard_transform"] == CAUSAL
    assert validation["gate_arm"].startswith("C")
    # float32 storage against a float64 reference; the same bound the write path is gated at.
    for field in EXPECTED_CAUSAL_WIDTHS:
        assert validation["gate_max_rel_full_segment"][field] < 1e-5, field
    # The truncation diagnostic describes a production-only operator, so it is absent here rather
    # than reported as zero -- there is no causal counterpart for it to deviate from.
    assert "s15_3_is_analytic_projection" not in validation


def test_the_comparison_tool_still_reads_a_two_sided_shard(
    two_sided_shard: Path, causal_bank: Any, bank: Any, phase_pairs: Dict[str, np.ndarray],
    channel_plan: Dict[str, Any]
) -> None:
    """The arm the study was built on, unchanged: arm B, with the S15.3 diagnostics."""
    compare = _compare_module()
    assert compare.resolve_shard_variant(str(two_sided_shard)) == TWO_SIDED

    fhr, up, stored, _ = compare.load_segments(str(two_sided_shard), range(COMPARED_SEGMENTS))
    validation = compare.measure_validation(
        stored, fhr, up, bank, phase_pairs,
        variant=TWO_SIDED, causal=causal_bank, plan=channel_plan,
    )

    assert validation["gate_arm"].startswith("B")
    assert set(validation["gate_max_rel_full_segment"]) == set(EXPECTED_WIDTHS) - {"fhr_up_ph"}
    assert np.isfinite(list(validation["gate_max_rel_interior"].values())).all()
    assert validation["s15_3_is_analytic_projection"] is True


def test_the_comparison_tool_names_a_variant_mismatch_instead_of_broadcasting(
    causal_shard: Path, bank: Any, phase_pairs: Dict[str, np.ndarray]
) -> None:
    """Both ways of asking it for the wrong arm say so, rather than failing inside numpy."""
    compare = _compare_module()
    fhr, up, stored, _ = compare.load_segments(str(causal_shard), range(COMPARED_SEGMENTS))

    # A causal shard read as two-sided: the widths cannot line up, and the refusal says why.
    with pytest.raises(ValueError, match="channel axes mean different things"):
        compare.measure_validation(stored, fhr, up, bank, phase_pairs, variant=TWO_SIDED)

    # A causal shard with no reference arm to gather.
    with pytest.raises(ValueError, match="needs the causal bank and the channel plan"):
        compare.measure_validation(stored, fhr, up, bank, phase_pairs, variant=CAUSAL)


def test_the_sample_plot_renders_both_variants(
    causal_shard: Path, two_sided_shard: Path, causal_stats_file: Path,
    two_sided_stats_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One panel per stored coefficient block: five rows two-sided, four causal.

    ``sample.fhr_up_ph`` raised ``AttributeError`` on a causal file, from inside a plotting loop
    that named nothing useful. The row count is captured rather than inferred from the PDF, since
    that is the thing that has to follow the file.
    """
    import matplotlib
    matplotlib.use("Agg")
    from hdf5_dataset import plot_dataset_samples

    rows: List[int] = []
    original = plot_dataset_samples.plt.subplots

    def record(n_rows: int, *args: Any, **kwargs: Any) -> Any:
        """Note how many panels the figure was asked for, then build it as usual."""
        rows.append(n_rows)
        return original(n_rows, *args, **kwargs)

    monkeypatch.setattr(plot_dataset_samples.plt, "subplots", record)

    for shard, stats, expected_rows in (
        (two_sided_shard, two_sided_stats_file, 5),
        (causal_shard, causal_stats_file, 4),
    ):
        rows.clear()
        output = tmp_path / shard.stem
        plot_dataset_samples.plot_random_dataset_samples(
            hdf5_file_path=str(shard), stats_file_path=str(stats),
            output_dir=str(output), n_samples=1, trim_minutes=TRIM_MINUTES,
        )
        assert rows == [expected_rows], shard.name
        assert list(output.glob("*.pdf")), shard.name


# =================================================================================================
# The leg alignment, which nothing else on the file reveals
# =================================================================================================
# An envelope-aligned causal shard has exactly the widths, warm-ups, delays and channel identities
# of an unaligned one. Every other coherence check in this file therefore passes on a mixed pair,
# and the only thing that can refuse one is the root attribute itself -- which is why absence has
# to mean 'none' rather than 'unknown', and why the comparison lands beside the variant check
# rather than beside the numeric ones.
def _realign(source: Path, destination: Path, value: Optional[str]) -> Path:
    """Copy a causal shard, setting or deleting its ``causal_leg_alignment``.

    Args:
        source: The shard to copy.
        destination: Where to write the copy.
        value: The alignment to record, or ``None`` to delete the attribute and produce the shape
            every causal shard written before it existed has.

    Returns:
        *destination*.
    """
    shutil.copyfile(source, destination)
    with h5py.File(destination, "a") as handle:
        if value is None:
            del handle.attrs["causal_leg_alignment"]
        else:
            handle.attrs["causal_leg_alignment"] = value
    return destination


def test_a_shard_without_the_attribute_reads_as_unaligned(
    causal_shard: Path, tmp_path: Path
) -> None:
    """Absence is an answer, not a gap: every causal shard on disk predates the attribute.

    Read as ``'none'`` rather than ``None``, so the comparisons below are between two strings and
    a legacy shard beside a freshly written unaligned one is a match rather than a mismatch.
    """
    legacy = _realign(causal_shard, tmp_path / "legacy_alignment.hdf5", None)
    with h5py.File(legacy, "r") as handle:
        assert "causal_leg_alignment" not in handle.attrs

    assert read_causal_warmup([str(legacy)], TRIM_MINUTES).leg_alignment == "none"
    assert read_causal_warmup([str(causal_shard)], TRIM_MINUTES).leg_alignment == "none"
    # And the pair loads as one dataset, which is the compatibility claim itself.
    dataset = CombinedHDF5Dataset(
        paths=[str(causal_shard), str(legacy)],
        cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
    )
    assert dataset.transform == CAUSAL


def test_a_mixed_leg_alignment_list_is_refused_naming_both(
    causal_shard: Path, tmp_path: Path
) -> None:
    """The refusal that no width, warm-up, delay or quantile check could ever make.

    Both files here have identical everything except one string, and the two hold different phase
    coefficients under the same channel names -- so a batch mixing them is a batch of two
    different representations with nothing to say so.
    """
    aligned = _realign(causal_shard, tmp_path / "aligned.hdf5", "envelope")
    with pytest.raises(ValueError, match="Mixed causal leg alignments") as error:
        CombinedHDF5Dataset(
            paths=[str(causal_shard), str(aligned)],
            cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
        )
    message = str(error.value)
    assert "'none'" in message and "'envelope'" in message
    assert causal_shard.name in message and aligned.name in message

    # The strict reader refuses it too, and for the same reason.
    with pytest.raises(ValueError, match="Mixed causal leg alignments"):
        read_causal_warmup([str(causal_shard), str(aligned)], TRIM_MINUTES)


def test_a_stats_file_built_at_another_alignment_is_refused_naming_both(
    causal_shard: Path, causal_stats_file: Path, tmp_path: Path
) -> None:
    """A stats file is keyed to the coefficients it was accumulated over, alignment included.

    The variant matches, every width matches, and the phase blocks hold different numbers. Without
    this the phase channels would be normalised with another transform's mean and scale, and the
    only symptom would be a training curve.
    """
    aligned = _realign(causal_shard, tmp_path / "aligned_shard.hdf5", "envelope")
    with pytest.raises(ValueError, match="leg-alignment mismatch") as error:
        _dataset(aligned, stats_path=str(causal_stats_file))
    message = str(error.value)
    assert "'none'" in message and "'envelope'" in message
    assert str(causal_stats_file) in message

    # The matching pair loads, so the refusal is about the disagreement and not about the check.
    assert _dataset(causal_shard, stats_path=str(causal_stats_file)).transform == CAUSAL


def test_the_stats_writer_records_the_alignment_it_accumulated_over(
    causal_shard: Path, causal_stats_file: Path, tmp_path: Path
) -> None:
    """Written on both variants, so absence is 'unaligned' rather than 'unlabelled'.

    That is what lets the pairing check above compare two strings instead of having to treat a
    missing attribute as compatible with everything.
    """
    with h5py.File(causal_stats_file, "r") as handle:
        assert handle.attrs["causal_leg_alignment"] == "none"

    aligned = _realign(causal_shard, tmp_path / "aligned_source.hdf5", "envelope")
    output = tmp_path / "aligned_stats.hdf5"
    calculator = _calculator()
    calculator.save_stats(
        calculator.calculate_stats([str(aligned)], batch_size=4, progress_bar=False), str(output)
    )
    with h5py.File(output, "r") as handle:
        assert handle.attrs["causal_leg_alignment"] == "envelope"
    # Read back onto the instance, which is where the loader's comparison takes it from.
    assert _calculator().load_stats(str(output)) is not None
