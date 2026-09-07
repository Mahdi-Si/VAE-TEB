r"""What the generated fixtures must be before any smoke result means anything.

**Execution-machine tests.** They read real HDF5 files and build a real dataset, so they need the
fixtures generated first::

    python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate
    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_fixtures.py -q

Without them every test here skips with the command that writes them, because "not generated yet"
and "the pipeline is broken" are different messages and only one deserves a traceback.

Four properties, each of which would make a green smoke run meaningless if it did not hold:

* **The splits are disjoint.** The same source shards are copied three times, so a missing identity
  rewrite would put every recording in all three splits and the disjointness assertion the pilot
  makes would be checking a property the fixture handed it for free.
* **Both classes are in every split.** A split carrying one class cannot produce an AUROC at all,
  and the smoke run would report an undefined metric as a finished stage.
* **The anchors are inside the window.** The cohort generator writes segments about eleven hours
  before delivery; if the time rewrite were dropped, every anchor would be filtered out and an
  empty cohort would run all the way to a report without anything in it.
* **The loader contract holds.** The pilot's own loader configuration, built from the fixture
  checkpoint's own resolved config, must open these shards and return the clinical fields --
  the loader skips a field it was not asked for without a word.

Nothing here asserts anything about latents, separation or outcomes. The identities, the times and
the labels are invented; the coefficients are the repository's tiny fixture bank.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train
from teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures import generate as fixtures


def _read(path: str) -> Dict[str, np.ndarray]:
    """The identity and clock columns of one shard."""
    import h5py

    with h5py.File(path, "r") as handle:
        guids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["guid"][:]
        ]
        return {
            "guid": np.asarray(guids, dtype=object),
            "epoch": np.asarray(handle["epoch"][:], dtype=np.float64),
            "target": np.asarray(handle["target"][:], dtype=np.float64),
            "weight": np.asarray(handle["weight"][:], dtype=np.float64),
        }


def _guids(paths: List[str]) -> Set[str]:
    """Every GUID in a split."""
    found: Set[str] = set()
    for path in paths:
        found.update(str(value) for value in _read(path)["guid"])
    return found


# =============================================================================
# What the generator wrote
# =============================================================================
def test_the_smoke_configuration_names_exactly_what_the_generator_writes(smoke_settings):
    """The one check that keeps the generator and ``configs/smoke.yaml`` from drifting apart. It
    needs no generated file: both sides are names."""
    paths = dict(smoke_settings["paths"])
    root = fixtures.GENERATED_ROOT
    assert Path(paths["checkpoint"]) == root / fixtures.CHECKPOINT_FILENAME
    assert Path(paths["statistics"]) == root / fixtures.STATISTICS_FILENAME
    for split in fixtures.SPLITS:
        configured = [Path(path) for path in paths[f"{split}_shards"]]
        assert configured == [
            root / split / f"{subgroup}.hdf5" for subgroup in fixtures.SMOKE_SUBGROUPS
        ]


def test_the_checkpoint_travels_with_the_configuration_it_was_trained_under(smoke_fixtures):
    """The strict loader reads the architecture from the file beside the checkpoint; a checkpoint
    copied without it cannot be rebuilt at all."""
    beside = Path(smoke_fixtures["checkpoint"]).parent / fixtures.RESOLVED_CONFIG_FILENAME
    assert beside.is_file()


def test_no_recording_appears_in_two_splits(smoke_fixtures):
    splits = {
        split: _guids(paths) for split, paths in smoke_fixtures["splits"].items()
    }
    for left in splits:
        for right in splits:
            if left < right:
                assert not splits[left] & splits[right], (
                    f"{sorted(splits[left] & splits[right])} appear in both {left} and {right}"
                )


def test_every_split_carries_both_binary_classes(smoke_fixtures):
    for split, paths in smoke_fixtures["splits"].items():
        outcomes = set()
        for path in paths:
            values = _read(path)
            for row in range(len(values["guid"])):
                code = labels.clinical_class_code(
                    values["target"][row], values["weight"][row]
                )
                outcome = data.binary_outcome(code)
                if outcome is not None:
                    outcomes.add(outcome)
        assert {0, 1} <= outcomes, f"{split} carries outcomes {sorted(outcomes)}"


def test_every_recording_contributes_a_late_and_an_early_segment(smoke_fixtures):
    """Which is what makes late eligibility, the trajectory bins and the paired early/late window
    comparison all have something to act on."""
    for _split, paths in smoke_fixtures["splits"].items():
        for path in paths:
            values = _read(path)
            per_recording: Dict[str, List[float]] = {}
            for guid, epoch in zip(values["guid"], values["epoch"]):
                per_recording.setdefault(str(guid), []).append(float(epoch))
            for guid, epochs in per_recording.items():
                hours = sorted(-value / data.SECONDS_PER_HOUR for value in epochs)
                assert any(hour <= 1.0 for hour in hours), f"{guid} has no late segment"
                assert any(hour > 2.0 for hour in hours), f"{guid} has no early segment"


def test_every_segment_starts_inside_the_coarse_window(smoke_fixtures):
    """A start earlier than the coarse filter would be dropped before its anchors were ever seen."""
    span = data.segment_span_seconds(330, 1.0)
    floor = data.coarse_epoch_min(3.0, span)
    for _split, paths in smoke_fixtures["splits"].items():
        for path in paths:
            epochs = _read(path)["epoch"]
            assert (epochs >= floor).all()
            assert (epochs < 0.0).all()


def test_no_late_segment_can_reach_delivery(smoke_fixtures):
    """A trimmed 330-step segment spans twenty minutes from its first anchor; a start too close to
    delivery would put scored coefficients at or after it, and the pilot would drop them."""
    trimmed = 330 - 2 * 15
    for _split, paths in smoke_fixtures["splits"].items():
        for path in paths:
            for epoch in _read(path)["epoch"]:
                last = float(epoch) + data.trim_seconds(1.0) + data.step_seconds() * (trimmed - 1)
                assert last < 0.0, f"a segment at {epoch} s reaches delivery at anchor {trimmed - 1}"


def test_the_generated_times_are_the_ones_the_generator_declares():
    """The layout is a property of the fixture, stated once and read here rather than re-derived."""
    epochs = fixtures.segment_epochs(6, segments_per_guid=2)
    assert epochs[0::2] == list(fixtures.LATE_EPOCHS)
    assert epochs[1::2] == list(fixtures.EARLY_EPOCHS)


# =============================================================================
# The loader interface, against the fixture checkpoint's own contract
# =============================================================================
@pytest.fixture(scope="module")
def loaded(smoke_fixtures):
    """The fixture checkpoint, rebuilt strictly through the pilot's own loader."""
    return pilot_model.load_pilot_checkpoint(smoke_fixtures["checkpoint"], device="cpu")


def test_the_fixture_checkpoint_rebuilds_strictly(loaded):
    assert loaded.geometry["model_class"]
    assert int(loaded.geometry["d_z"]) > 0
    assert loaded.digest


def test_the_pilot_loader_configuration_keeps_the_checkpoints_contract(loaded, smoke_fixtures):
    """Four things change and nothing else does: the shard lists, the statistics file, the loaded
    fields and the coarse epoch filter."""
    shards = smoke_fixtures["splits"]["train"]
    config = data.pilot_loader_config(
        loaded.config,
        shards=shards,
        statistics=smoke_fixtures["statistics"],
        epoch_min=-12120.0,
    )
    dataset = config["dataset_config"]
    kwargs = dataset["dataloader_config"]["dataset_kwargs"]
    assert dataset["vae_test_datasets"] == list(shards)
    assert dataset["vae_train_datasets"] == list(shards)
    assert dataset["stat_path"] == smoke_fixtures["statistics"]
    assert kwargs["epoch_min"] == -12120.0
    assert kwargs["label"] is None
    for field in data.REQUIRED_LOAD_FIELDS + data.CLINICAL_FIELDS:
        assert field in kwargs["load_fields"]
    original = dict(loaded.config["dataset_config"]["dataloader_config"]["dataset_kwargs"])
    assert kwargs["trim_minutes"] == original["trim_minutes"]


def test_the_generated_shards_open_through_that_configuration(loaded, smoke_fixtures):
    """The end of the loader contract: a real dataset over the real fixture files, indexed by the
    recording key every bag is pooled on."""
    shards = smoke_fixtures["splits"]["train"]
    config = data.pilot_loader_config(
        loaded.config, shards=shards, statistics=smoke_fixtures["statistics"]
    )
    source = pilot_train.RecordingSource.from_config(config, shards=shards)
    assert source.record["n_recordings"] == len(fixtures.SMOKE_SUBGROUPS) * len(
        fixtures.LATE_EPOCHS
    )
    assert source.record["n_repeated_segment_keys"] == 0

    guid, epoch = next(iter(source.index))
    batch = source.batch(guid, [epoch])
    for field in data.MODEL_INPUT_FIELDS + ("weight",):
        assert hasattr(batch, field) or field in batch, f"the loader returned no {field!r}"
