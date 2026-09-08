r"""The pilot checkpoint, the base-model export, and the refusals around both.

**Execution-machine tests.** They build the tiny net and write real tensors, which the synthetic
logic subset excludes; they need no production checkpoint, no clinical data and no GPU::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_persistence.py -q

What they establish: an adaptation round-trips and reproduces the same ``mu_post`` after being
applied to a freshly loaded model; it refuses to be applied to another checkpoint or another
geometry; the base-model export carries no classifier key and loads through the repository's own
strict loader; and neither writer can land on the pretrained checkpoint.

The run directory's own bookkeeping -- stage state, resume and the selection lock -- is checked
without a model in ``tests/logic/test_run_state.py``.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn_transformer_cfs.latent_pilot import extract, train
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    TINY_STRIDE,
    build,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)

TRIM_MINUTES = 1.0
FINGERPRINT = {"checkpoint_digest": "synthetic-digest", "support_policy": "forecast_contributing"}


@pytest.fixture
def kwargs():
    """The tiny geometry every model in this file is built at."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


@pytest.fixture
def bundle(kwargs, tmp_path):
    """A loaded-checkpoint bundle whose configuration is a real file the export can copy."""
    config = {
        "dataset_config": {
            "stat_path": "/synthetic/stats.hdf5",
            "dataloader_config": {"dataset_kwargs": {"trim_minutes": TRIM_MINUTES}},
        }
    }
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    task = make_task(model_kwargs=kwargs)
    pilot_model.freeze_for_pilot(task.orig_model)
    blob = {
        "model_kwargs": dict(kwargs),
        "model_class": type(task.orig_model).__name__,
        "hyper_parameters": {"likelihood": "gaussian_nll"},
        "epoch": 7,
    }
    return types.SimpleNamespace(
        task=task,
        model=task.orig_model,
        blob=blob,
        config=config,
        config_path=config_path,
        checkpoint_path=tmp_path / "source.ckpt",
        digest="synthetic-digest",
        geometry=pilot_model.geometry_record(task.orig_model, blob, config),
    )


def _fit(bundle, *, epoch: int = 3, shift: float = 0.05):
    """An adaptation that actually moved the mean heads, with a fitted classifier beside it."""
    with torch.no_grad():
        for _name, parameter in pilot_model.mean_head_parameters(bundle.model):
            parameter.add_(shift * torch.ones_like(parameter))
    d_z = int(bundle.model.d_z)
    classifier = pilot_model.LatentClassifier(
        d_z, center=np.zeros(d_z), scale=np.full(d_z, 2.0)
    )
    return train.AdaptationFit(
        mean_head_state={
            name: parameter.detach().clone()
            for name, parameter in pilot_model.mean_head_parameters(bundle.model)
        },
        classifier=classifier,
        threshold=0.25,
        selected_epoch=epoch,
        history=pd.DataFrame([{"epoch": 0}, {"epoch": epoch}]),
        record={"name": "finetuned", "selected_epoch": epoch},
    )


def _batch():
    """One stub batch, for comparing forwards before and after a round trip."""
    stub = make_stub_batch(batch=2, seed=0)
    stub.epoch = torch.tensor([-3600.0, -3000.0])
    stub.guid = ["SYNTH-0", "SYNTH-1"]
    return stub


# =============================================================================
# The pilot checkpoint
# =============================================================================
def test_an_adaptation_round_trips_with_its_classifier_and_threshold(bundle, tmp_path):
    fit = _fit(bundle)
    train.save_adapted(fit, bundle, tmp_path, fingerprint=FINGERPRINT)

    reloaded = train.load_adapted(tmp_path)

    assert reloaded.selected_epoch == fit.selected_epoch
    assert reloaded.threshold == pytest.approx(fit.threshold)
    assert reloaded.fingerprint == FINGERPRINT
    assert reloaded.source["checkpoint_digest"] == bundle.digest
    for name, tensor in fit.mean_head_state.items():
        assert torch.equal(reloaded.mean_head_state[name], tensor.cpu())

    # The classifier comes back with the scaler it was fitted under, as buffers.
    rebuilt = reloaded.classifier()
    values = torch.randn(4, int(bundle.model.d_z))
    assert torch.allclose(rebuilt(values), fit.classifier(values))
    assert torch.allclose(rebuilt.scale, torch.full((int(bundle.model.d_z),), 2.0))


def test_applying_a_saved_adaptation_reproduces_the_same_latent(bundle, kwargs, tmp_path):
    """The whole point of the artifact: another process gets this run's mu_post, not the pretrained one."""
    batch = _batch()
    fit = _fit(bundle)
    adapted = pilot_model.deterministic_outputs(
        bundle.task, batch, keys=("mu_post",)
    )["mu_post"]
    train.save_adapted(fit, bundle, tmp_path, fingerprint=FINGERPRINT)

    fresh = make_task(model_kwargs=kwargs)
    pilot_model.freeze_for_pilot(fresh.orig_model)
    reloaded_bundle = types.SimpleNamespace(
        task=fresh, model=fresh.orig_model, digest=bundle.digest
    )
    pretrained = pilot_model.deterministic_outputs(fresh, batch, keys=("mu_post",))["mu_post"]
    assert not torch.equal(pretrained, adapted), "the fixture must have moved something"

    record = train.apply_adapted(train.load_adapted(tmp_path), reloaded_bundle)

    after = pilot_model.deterministic_outputs(fresh, batch, keys=("mu_post",))["mu_post"]
    assert torch.allclose(after, adapted, atol=1e-6)
    assert record["selected_epoch"] == fit.selected_epoch
    assert record["n_parameters"] == sum(
        tensor.numel() for tensor in fit.mean_head_state.values()
    )


def test_an_adaptation_of_another_checkpoint_is_refused(bundle, kwargs, tmp_path):
    train.save_adapted(_fit(bundle), bundle, tmp_path, fingerprint=FINGERPRINT)
    other = make_task(model_kwargs=kwargs)
    pilot_model.freeze_for_pilot(other.orig_model)

    with pytest.raises(PilotConfigError, match="was fitted on checkpoint digest"):
        train.apply_adapted(
            train.load_adapted(tmp_path),
            types.SimpleNamespace(
                task=other, model=other.orig_model, digest="a-different-checkpoint"
            ),
        )


def test_an_adaptation_naming_parameters_this_model_lacks_is_refused(bundle, tmp_path):
    train.save_adapted(_fit(bundle), bundle, tmp_path, fingerprint=FINGERPRINT)
    checkpoint = train.load_adapted(tmp_path)
    renamed = train.PilotCheckpoint(
        mean_head_state={"posterior_head.delta_mu_head.99.weight": torch.zeros(2, 2)},
        classifier_state=checkpoint.classifier_state,
        threshold=checkpoint.threshold,
        selected_epoch=checkpoint.selected_epoch,
        source=checkpoint.source,
        fingerprint=checkpoint.fingerprint,
        record=checkpoint.record,
    )

    with pytest.raises(PilotConfigError, match="geometries differ"):
        train.apply_adapted(renamed, bundle)


def test_a_pilot_checkpoint_from_another_layout_is_refused(bundle, tmp_path):
    train.save_adapted(_fit(bundle), bundle, tmp_path, fingerprint=FINGERPRINT)
    target = tmp_path / train.PILOT_CHECKPOINT_FILENAME
    blob = torch.load(target, map_location="cpu", weights_only=False)
    blob["version"] = train.PILOT_CHECKPOINT_VERSION + 1
    torch.save(blob, target)

    with pytest.raises(PilotConfigError, match="pilot-checkpoint version"):
        train.load_adapted(tmp_path)


def test_a_missing_pilot_checkpoint_names_the_stage_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="finetune stage"):
        train.load_adapted(tmp_path)


# =============================================================================
# The base-model export
# =============================================================================
def test_the_export_loads_through_the_repository_s_own_strict_loader(bundle, kwargs, tmp_path):
    from train.graph_models_utils import check_model_class, load_checkpoint_strict

    _fit(bundle)
    target = train.export_base_checkpoint(bundle, tmp_path)
    blob = torch.load(target, map_location="cpu", weights_only=False)

    check_model_class(blob, type(bundle.model).__name__)
    fresh = build(kwargs)
    assert load_checkpoint_strict(model=fresh, checkpoint=blob) is not None
    for name, tensor in bundle.model.state_dict().items():
        assert torch.equal(fresh.state_dict()[name], tensor)


def test_the_export_carries_no_classifier_key(bundle, tmp_path):
    """The classifier is not part of the architecture, so it never travels in this file."""
    fit = _fit(bundle)
    blob = torch.load(
        train.export_base_checkpoint(bundle, tmp_path), map_location="cpu", weights_only=False
    )

    # Checked against the classifier's own key names and the net's own, rather than against a
    # substring: the net carries a ``target_adapter.linear.weight`` of its own, so "linear" in a
    # key says nothing about where that key came from.
    assert not set(fit.classifier.state_dict()) & set(blob["state_dict"])
    assert set(blob["state_dict"]) == set(bundle.model.state_dict())
    assert "classifier" not in blob
    assert blob["model_kwargs"] == bundle.blob["model_kwargs"]
    assert blob["hyper_parameters"] == bundle.blob["hyper_parameters"]
    assert blob["pilot"]["source_digest"] == bundle.digest


def test_the_export_carries_its_configuration_where_the_evaluator_looks(bundle, tmp_path):
    """``resolved_config_for`` looks beside a checkpoint, so the copy goes there."""
    from teb_vae.lag_attn_cfs.eval.probe import resolved_config_for

    target = train.export_base_checkpoint(bundle, tmp_path)

    assert resolved_config_for(target).is_file()
    assert yaml.safe_load(resolved_config_for(target).read_text()) == bundle.config


# =============================================================================
# The source checkpoint is never written to
# =============================================================================
def test_neither_writer_can_land_on_the_pretrained_checkpoint(bundle, tmp_path):
    fit = _fit(bundle)
    # A bundle whose source path IS the file the writer is about to produce.
    colliding = types.SimpleNamespace(
        task=bundle.task,
        model=bundle.model,
        blob=bundle.blob,
        config=bundle.config,
        config_path=bundle.config_path,
        checkpoint_path=tmp_path / train.PILOT_CHECKPOINT_FILENAME,
        digest=bundle.digest,
        geometry=bundle.geometry,
    )

    with pytest.raises(PilotConfigError, match="pretrained checkpoint's own file"):
        train.save_adapted(fit, colliding, tmp_path, fingerprint=FINGERPRINT)


def test_the_export_refuses_to_land_on_the_source_too(bundle, tmp_path):
    exporting = types.SimpleNamespace(
        task=bundle.task,
        model=bundle.model,
        blob=bundle.blob,
        config=bundle.config,
        config_path=bundle.config_path,
        checkpoint_path=(
            tmp_path / train.ADAPTED_EXPORT_DIRNAME / train.ADAPTED_CHECKPOINT_FILENAME
        ),
        digest=bundle.digest,
        geometry=bundle.geometry,
    )

    with pytest.raises(PilotConfigError, match="pretrained checkpoint's own file"):
        train.export_base_checkpoint(exporting, tmp_path)


def test_the_source_checkpoint_and_config_are_untouched_by_a_full_save(bundle, tmp_path):
    """Both writers run; neither file the pilot started from changes."""
    bundle.checkpoint_path.write_bytes(b"the pretrained checkpoint")
    before_config = bundle.config_path.read_bytes()

    fit = _fit(bundle)
    train.save_adapted(fit, bundle, tmp_path, fingerprint=FINGERPRINT)
    train.export_base_checkpoint(bundle, tmp_path)

    assert bundle.checkpoint_path.read_bytes() == b"the pretrained checkpoint"
    assert bundle.config_path.read_bytes() == before_config
