r"""The extraction pass, on the real net.

**Execution-machine tests**, like the model contract they sit beside: every one of these runs a real
forward. They need no checkpoint file, no clinical data and no GPU -- the model is the committed tiny
geometry, the loader is a list of the suite's synthetic stub batches, and the loaded-checkpoint
bundle is assembled here from the same ``geometry_record`` a real load would build::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_extraction.py -q

What they establish: the pass is repeatable, its rows are aligned with its arrays, its support is
the objective's, and a before/after pair read from two models describes the same anchors -- which is
the property every paired comparison in this pilot rests on.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    TINY_STRIDE,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)

#: Trim and window used throughout. The stub batch's epochs are chosen below so its anchors land
#: inside the window; a fixture whose anchors all fell outside would test the refusal, not the pass.
TRIM_MINUTES = 1.0
PRESERVATION_HOURS = 3.0
BIN_HOURS = 0.5


@pytest.fixture
def loaded():
    """A loaded-checkpoint bundle around the tiny model, without a checkpoint file.

    Everything the extraction reads -- the task, the net, and the geometry record its timestamps,
    fingerprint and forecast-endpoint check are derived from -- assembled through the same
    ``geometry_record`` a real load builds, so this exercises that function too.
    """
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    task = make_task(model_kwargs=kwargs)
    pilot_model.freeze_for_pilot(task.orig_model)
    blob = {
        "model_kwargs": dict(kwargs),
        "model_class": type(task.orig_model).__name__,
        "epoch": 0,
    }
    config = {
        "dataset_config": {
            "stat_path": "/synthetic/stats.hdf5",
            "dataloader_config": {"dataset_kwargs": {"trim_minutes": TRIM_MINUTES}},
        }
    }
    return types.SimpleNamespace(
        task=task,
        model=task.orig_model,
        blob=blob,
        config=config,
        checkpoint_path="/synthetic/tiny.ckpt",
        digest="synthetic-digest",
        geometry=pilot_model.geometry_record(task.orig_model, blob, config),
    )


def _loader(seed: int = 0, epochs=(-3600.0, -3000.0)):
    """A one-batch loader whose segments sit inside the preserved window."""
    batch = make_stub_batch(batch=len(epochs), seed=seed)
    batch.epoch = torch.tensor(list(epochs), dtype=torch.float32)
    batch.guid = [f"SYNTH-{index}" for index in range(len(epochs))]
    return [batch]


def _extract(loaded, loader, **overrides):
    """Run one extraction at this file's window settings."""
    return extract.extract_split(
        loaded,
        loader,
        split=overrides.pop("split", "train"),
        preservation_hours=PRESERVATION_HOURS,
        bin_hours=BIN_HOURS,
        **overrides,
    )


# =============================================================================
# The pass
# =============================================================================
def test_the_geometry_record_reads_the_checkpoint_and_the_net(loaded):
    """Declared values come from the kwargs, resolved ones from the model that was built."""
    geometry = loaded.geometry
    assert geometry["trim_minutes"] == TRIM_MINUTES
    assert geometry["d_z"] == loaded.model.d_z
    assert geometry["sequence_length"] == int(loaded.model.geometry.t)
    assert geometry["horizon"] == int(loaded.model.geometry.horizon)
    assert geometry["coverage_floor"] == float(loaded.model.coverage_floor)
    assert geometry["checkpoint_stat_path"] == "/synthetic/stats.hdf5"


def test_extraction_rows_are_aligned_with_its_arrays(loaded):
    """A row's latent is the model's own value at that anchor, not a neighbour's."""
    extraction = _extract(loaded, _loader())
    assert len(extraction.frame) == extraction.arrays["mu_post"].shape[0]
    assert extraction.arrays["mu_post"].shape[1] == int(loaded.model.d_z)
    assert extraction.frame[data.ROW_COLUMN].tolist() == list(range(len(extraction.frame)))

    batch = _loader()[0]
    with torch.no_grad():
        outputs = loaded.model(*pilot_model.forward_inputs(loaded.task, batch))
    guids = [str(value) for value in batch.guid]

    for position in (0, len(extraction.frame) // 2, len(extraction.frame) - 1):
        row = extraction.frame.iloc[position]
        sample = guids.index(str(row[data.GUID_COLUMN]))
        anchor = int(row[data.ANCHOR_COLUMN])
        expected = outputs["mu_post"][sample, anchor].detach().cpu().numpy()
        assert np.allclose(
            extraction.arrays["mu_post"][int(row[data.ROW_COLUMN])], expected, atol=1e-6
        )


def test_two_extractions_of_one_model_agree_exactly(loaded):
    """No draw is involved, so a repeated read is bit-identical."""
    first = _extract(loaded, _loader())
    second = _extract(loaded, _loader())

    for key in extract.LATENT_KEYS:
        assert np.array_equal(first.arrays[key], second.arrays[key])
    assert first.frame.equals(second.frame)
    assert first.fingerprint == second.fingerprint


def test_only_the_objective_s_contributing_anchors_are_retained(loaded):
    """The stub batch carries a deliberate weight gap, and the support rule sees it."""
    extraction = _extract(loaded, _loader())
    frame = extraction.frame

    assert (~frame["contributing"]).any(), "the gap must exclude something, or this proves nothing"
    excluded = frame[frame[data.EXCLUSION_COLUMN] == data.EXCLUDED_NOT_CONTRIBUTING]
    assert not excluded["contributing"].any()
    assert data.retained(frame)["contributing"].all()
    assert extraction.record["n_retained_anchors"] < extraction.record["n_in_window_anchors"]


def test_anchors_outside_the_window_are_counted_rather_than_stored(loaded):
    """The window is a definition, so it is reported as a count and not as a row per anchor."""
    extraction = _extract(loaded, _loader(epochs=(-3600.0, -3000.0)))
    assert extraction.record["n_outside_window"] >= 0
    hours = np.asarray(extraction.frame[data.HOURS_COLUMN], dtype=np.float64)
    assert np.all(hours > 0.0) and np.all(hours <= PRESERVATION_HOURS)


def test_a_split_with_nothing_near_delivery_is_refused(loaded):
    """Rather than returning an empty extraction that a later stage would read as a cohort."""
    with pytest.raises(PilotConfigError, match="no anchor inside"):
        # Ten hours before delivery: every anchor is outside the three-hour window.
        _extract(loaded, _loader(epochs=(-36000.0, -36000.0)))


# =============================================================================
# The paired reading
# =============================================================================
def test_two_models_are_read_at_identical_keys_and_differ_only_in_the_mean(loaded):
    """The whole before/after contract, end to end on one batch."""
    before = _extract(loaded, _loader())

    with torch.no_grad():
        for _name, parameter in pilot_model.mean_head_parameters(loaded.model):
            parameter.add_(0.05 * torch.ones_like(parameter))

    after = _extract(loaded, _loader())

    # Same anchors, same order: the comparison is paired by construction rather than by a join.
    extract.assert_same_keys(before, after)

    assert not np.array_equal(before.arrays["mu_post"], after.arrays["mu_post"])
    assert np.array_equal(before.arrays["mu_prior"], after.arrays["mu_prior"])
    assert np.array_equal(before.arrays["logvar_prior"], after.arrays["logvar_prior"])
    assert np.array_equal(before.arrays["logvar_post"], after.arrays["logvar_post"])


# =============================================================================
# Persistence and guards
# =============================================================================
def test_an_extraction_round_trips_and_refuses_a_foreign_fingerprint(loaded, tmp_path):
    extraction = _extract(loaded, _loader())
    extract.save_extraction(extraction, tmp_path, name="pretrained")

    reloaded = extract.load_extraction(
        tmp_path, name="pretrained", expected=extraction.fingerprint
    )
    for key in extract.LATENT_KEYS:
        assert np.array_equal(reloaded.arrays[key], extraction.arrays[key])
    assert len(reloaded.frame) == len(extraction.frame)

    foreign = dict(extraction.fingerprint, checkpoint_digest="another-checkpoint")
    with pytest.raises(PilotConfigError, match="checkpoint_digest"):
        extract.load_extraction(tmp_path, name="pretrained", expected=foreign)


def test_a_missing_extraction_names_the_stage_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="extraction stage"):
        extract.load_extraction(tmp_path, name="pretrained")


def test_the_test_split_is_refused_before_any_forward_runs(loaded):
    """The guard fires on the argument, so no model work happens on the way to the refusal."""
    with pytest.raises(PilotConfigError, match="evaluation stage"):
        _extract(loaded, _loader(), split="test")

    permitted = _extract(loaded, _loader(), split="test", allow_test=True)
    assert set(permitted.frame[data.SPLIT_COLUMN]) == {"test"}


def test_the_scaler_fits_on_a_real_extraction(loaded):
    """The training-only rule and the hierarchy hold on the pass's own output."""
    extraction = _extract(loaded, _loader())
    retained = extraction.retained
    # The full matrix, not a gathered one: ``row`` indexes the extraction's own arrays, and the
    # reductions read it, so a gathered matrix would be indexed by the ungathered positions.
    scaler = extract.fit_scaler(retained, extraction.arrays["mu_post"])

    assert scaler.center.size == int(loaded.model.d_z)
    assert np.all(scaler.scale > 0)
    assert scaler.record["n_recordings"] == retained[data.GUID_COLUMN].nunique()
