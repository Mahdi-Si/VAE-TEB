"""Config schema, ``base:`` chains, ``--set`` overrides and digest (SPEC §13; the schema half of T-F1)."""
from __future__ import annotations

import json
import sys

import pytest
from pydantic import ValidationError

from teb_vae.classifier import config

DEFAULT = config.DEFAULT_CONFIG
SHIPPED = sorted(DEFAULT.parent.glob("*.yaml"))
_T = "{name: t, kind: numeric, available_at: prospective}"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.name)
def test_shipped_configs_validate(path):
    assert config.load(path).classifier.labels.task in config.TASKS


def test_smoke_overrides_default():
    smoke = config.load(DEFAULT.parent / "smoke.yaml").classifier
    assert (smoke.run.device, smoke.source.kind, smoke.run.folds) == ("cpu", "hdf5", [1, 2])
    assert smoke.labels.strategy == "horizon_decay"  # inherited through base: default.yaml


@pytest.mark.parametrize("override", [
    "classifier.labels.bogus=1",
    "classifier.bogus.x=1",
    "classifier.source.vae.keys=[{name: mu_prior, rolle: value}]",
    "clasifier.run.name=x",
])
def test_unknown_keys_rejected(override):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        config.load(DEFAULT, [override])


@pytest.mark.parametrize("overrides", [
    ["classifier.labels.task=three_class"],  # head: binary needs K = 2
    ["classifier.labels.head=ordinal"],  # ordinal needs K >= 3
    ["classifier.eval.primary_policy=nope"],
    ["classifier.model.sequence.causal=false"],  # needs final_only | mil
    ["classifier.context.covariates.variables=[{name: t, kind: numeric, available_at: retro}]"],
    ["classifier.data.min_valid_frac=1.5"],
    ["classifier.train.cotrain.grad_segments=some"],
    ["classifier.model.scope=segment", "classifier.labels.k_warm=2"],  # k_warm is a sequence-scope warm-up
    [f"classifier.context.covariates.variables=[{_T}]"],  # no covariate table
    ["classifier.context.covariates.static_csv=s.csv", f"classifier.context.covariates.variables=[{_T}, {_T}]"],
    ["classifier.model.scope=segment", "classifier.context.fusion=token", "classifier.context.covariates.static_csv=s.csv",
     f"classifier.context.covariates.variables=[{_T}]"],
])
def test_invalid_values_rejected(overrides):
    with pytest.raises(ValidationError):
        config.load(DEFAULT, overrides)


def test_valid_cross_field_combinations():
    config.load(DEFAULT, ["classifier.model.sequence.causal=false",
                          "classifier.labels.strategy=mil"])
    config.load(DEFAULT, ["classifier.train.loss.beta_en=0"])
    config.load(DEFAULT, ["classifier.model.scope=segment", "classifier.labels.k_warm=0"])
    config.load(DEFAULT, ["classifier.labels.k_warm=3", "classifier.context.fusion=token",
                          "classifier.context.covariates.timed_csv=t.csv", f"classifier.context.covariates.variables=[{_T}]"])
    config.load(DEFAULT, ["advanced_config.callbacks.early_stopping.monitor=val/guid_auroc",
                          "advanced_config.callbacks.early_stopping.mode=max"])


_3C = ["classifier.labels.task=three_class", "classifier.labels.aux_3class_weight=0"]


@pytest.mark.parametrize("head, loss", [("multiclass", "ce"), ("multiclass", "weighted_ce"), ("ordinal", "coral"),
                                        ("ordinal", "cumulative_link")])
def test_three_class_heads_and_their_checks(head, loss):
    c = config.load(DEFAULT, _3C + [f"classifier.labels.head={head}", f"classifier.train.loss.name={loss}"]).classifier
    assert config.ovr_enabled(c) and not config.ovr_enabled(config.load(DEFAULT).classifier)
    assert not config.ovr_enabled(config.load(DEFAULT, _3C + [f"classifier.labels.head={head}",
                                                              f"classifier.train.loss.name={loss}",
                                                              "classifier.eval.ovr_thresholds=false"]).classifier)
    for sets, match in [([f"classifier.labels.head={head}"], "does not fit labels.head"),  # default loss: bce
                        ([f"classifier.labels.head={head}", f"classifier.train.loss.name={loss}",
                          "classifier.labels.aux_3class_weight=0.3"], "aux_3class_weight must be 0"),
                        ([f"classifier.labels.head={head}", f"classifier.train.loss.name={loss}",
                          "classifier.calibration.method=platt"], "platt is binary only")]:
        with pytest.raises(ValidationError, match=match):
            config.load(DEFAULT, _3C + sets)
    with pytest.raises(ValidationError, match="ovr_thresholds: true needs"):
        config.load(DEFAULT, ["classifier.eval.ovr_thresholds=true"])


def _policy(**kw):
    """A dict delta whose only threshold policy is ``kw`` (lists replace on merge)."""
    return {"classifier": {"eval": {"thresholds": [{"id": "p", "basis": "guid_final", **kw}], "primary_policy": "p"}}}


@pytest.mark.parametrize("policy, match", [
    ({"policy": "fpr_cap", "method": "empirical"}, "needs \\['alpha'\\]"),
    ({"policy": "fpr_cap", "alpha": 0.3}, "needs \\['method'\\]"),
    ({"policy": "fpr_cap", "alpha": 0.3, "method": "np_umbrella"}, "needs \\['delta'\\]"),
    ({"policy": "sens_target"}, "needs \\['beta'\\]"),
    ({"policy": "fixed"}, "needs \\['value'\\]"),
    ({"policy": "fpr_cap", "alpha": 0.3, "method": "np_umbrella", "delta": 0.05, "basis": "segment"},
     "allows only method empirical"),
])
def test_threshold_policy_keys_are_checked(policy, match):
    with pytest.raises(ValidationError, match=match):
        config.load(DEFAULT, _policy(**policy))
    ok = [{"policy": "fpr_cap", "alpha": 0.3, "method": "empirical", "basis": "segment"},
          {"policy": "sens_target", "beta": 0.8}, {"policy": "fixed", "value": 0.5}, {"policy": "youden"}]
    for kw in ok:
        config.load(DEFAULT, _policy(**kw))


def test_beta_en_one_is_refused():  # effective-number weights would be 0/0 = NaN
    with pytest.raises(ValidationError, match="beta_en"):
        config.load(DEFAULT, ["classifier.train.loss.beta_en=1.0"])


@pytest.mark.parametrize("key, value", [("monitor", "val/guid_auroc"), ("mode", "max")])
def test_checkpoint_monitor_must_be_the_early_stopping_one(key, value):
    with pytest.raises(ValidationError, match="differs from early_stopping"):
        config.load(DEFAULT, [f"advanced_config.callbacks.model_checkpoint.{key}={value}"])
    config.load(DEFAULT, ["advanced_config.callbacks.model_checkpoint.monitor=val/guid_logloss"])  # equal: fine
    assert config.selection_monitor({}) == ("val/guid_logloss", "min")


def test_set_values_parse_as_yaml_and_match_a_dict_delta():
    run = config.load(DEFAULT, [
        "classifier.labels.horizon_h=2", "classifier.data.patient_map=null",
        "classifier.run.folds=[1,3]", "classifier.run.device=cuda:1", "classifier.data.stride_s=660",
    ]).classifier
    assert run.labels.horizon_h == 2.0 and run.data.patient_map is None
    assert run.run.folds == [1, 3] and run.run.device == "cuda:1" and run.data.stride_s == 660.0
    delta = {"classifier": {"labels": {"horizon_h": 2}}}
    assert config.load(DEFAULT, delta).classifier.labels.horizon_h == 2.0


def test_base_chain(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {DEFAULT}\nclassifier:\n  labels: {{task: hie_vs_rest}}\n")
    labels = config.load(child).classifier.labels
    assert (labels.task, labels.strategy) == ("hie_vs_rest", "horizon_decay")


def test_digest_is_stable_ignores_execution_knobs_and_tracks_settings():
    base = config.digest(config.load(DEFAULT))
    assert len(base) == 64 and base == config.digest(config.load(DEFAULT))
    assert base == config.digest(config.load(
        DEFAULT, ["classifier.run.device=cpu", "classifier.run.num_workers=0", "classifier.run.report_workers=0"]))
    assert base != config.digest(config.load(DEFAULT, ["classifier.labels.horizon_h=2"]))
    assert base != config.digest(config.load(DEFAULT, ["advanced_config.trainer.precision=bf16"]))


def test_a_run_dir_of_another_schema_version_is_refused(tmp_path, monkeypatch):
    """A run dir written under an older run-dir schema (e.g. no ``patient`` column) never resumes."""
    from teb_vae.classifier import run

    cfg = config.load(DEFAULT)
    monkeypatch.setattr(config, "SCHEMA_VERSION", config.SCHEMA_VERSION - 1)
    old = config.digest(cfg)
    monkeypatch.undo()
    assert old != config.digest(cfg)
    (tmp_path / "manifest.json").write_text(json.dumps({"config_digest": old}))
    try:
        with pytest.raises(ValueError, match="different config"):
            run.main(config=str(DEFAULT), stage="cohort", run_dir=str(tmp_path))
    finally:  # run.main pointed loguru's sinks at tmp_path
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr)
