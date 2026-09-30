"""P7 adaptation regimes (SPEC §10.1, §8.2, §16 P7 accept): the trainable VaeSource contract on the tiny trf_cfs
checkpoint (gradient reach inside ``train.unfreeze``, none outside, frozen modules in eval whatever ``train()`` does),
the backbone's optimiser group, the LPFT stage transition, the regime schema, and one online run through predict."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from hdf5_dataset.hdf5_dataset import attribute_dict_collate
from teb_vae.classifier import cohort, config, sources
from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

ALLOW = "posterior_head"  # the pilot's allowlist family: the posterior's delta heads


def _restore_loguru() -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)


def _cfg(vae_overrides, *extra):
    return config.load(SMOKE_CONFIG, list(vae_overrides) + list(extra)).classifier


def _backbone(c, unfreeze=(ALLOW,), cotrain=None):
    """``(OnlineFeatures, raw rows)``: the tiny VAE behind an identity scaler, and 4 collated fold-1 train rows."""
    source = sources.VaeSource(c.source, unfreeze=unfreeze)
    dataset = source.dataset(cohort.fold_shards(c.data, 1)["train"])
    rows = attribute_dict_collate([dataset[i] for i in range(4)])
    probe = source(rows)
    channels = probe.channels
    scaler = sources.Scaler(channels=channels, center=np.zeros(len(channels)), scale=np.ones(len(channels)),
                            keep=np.ones(len(channels), bool), record={})
    n_values = probe.values.shape[-1]
    return sources.OnlineFeatures(source, scaler, n_values=n_values, chunk=3, cotrain=cotrain), rows, probe


def _batch(rows, probe):
    """A segment-scope batch around ``rows``: cached-looking ``x``/``attn`` (zeros) and the source's step mask."""
    return {"vae": rows, "x": torch.zeros_like(probe.values), "attn": torch.zeros_like(probe.attn),
            "step_mask": probe.step_mask.clone()}


# ---- the trainable-source contract -------------------------------------------------------------------------------
def test_gradient_reaches_the_allowlist_only_and_frozen_modules_stay_in_eval(vae_overrides):
    features, rows, probe = _backbone(_cfg(vae_overrides))
    source = features.source
    allowed = {n for n, _ in source.allowlist()}
    assert allowed and all(n.startswith(ALLOW + ".") for n in allowed)
    assert {n for n, p in source.model.named_parameters() if p.requires_grad} == allowed

    features.train()
    source.check_frozen()  # every module outside the allowlist is in eval, whatever train() did
    modes = {n: m.training for n, m in source.model.named_modules()}
    assert all(modes[n] == (n == ALLOW or n.startswith(ALLOW + ".")) for n in modes)
    out = features(_batch(rows, probe))
    x, mask = out["x"], probe.step_mask
    assert x.requires_grad and (x[~mask] == 0).all()
    features.eval()  # dropout off: the online features are the source's own (chunked in 3 + 1)
    torch.testing.assert_close(features(_batch(rows, probe))["x"][mask], probe.values[mask])

    features.train()
    (features(_batch(rows, probe))["x"] ** 2).sum().backward()
    grads = {n: p.grad for n, p in source.model.named_parameters()}
    assert all(grads[n] is None for n in grads if n not in allowed)  # no gradient outside the allowlist
    assert sum(float(grads[n].abs().sum()) for n in allowed if grads[n] is not None) > 0
    assert any(grads[n] is not None and grads[n].abs().sum() > 0 for n in allowed if ".delta_mu_head." in n)

    stray = next(p for n, p in source.model.named_parameters() if n not in allowed)
    stray.requires_grad_(True)
    with pytest.raises(RuntimeError, match="requiring grad"):
        source.check_frozen()
    stray.requires_grad_(False)
    source.model.target_encoder.train()
    with pytest.raises(RuntimeError, match="training mode"):
        source.check_frozen()


def test_frozen_online_source_has_no_gradient_and_the_step_mask_must_match(vae_overrides):
    features, rows, probe = _backbone(_cfg(vae_overrides), unfreeze=())
    features.train()
    assert not any(m.training for m in features.vae.modules())
    assert not features(_batch(rows, probe))["x"].requires_grad
    bad = _batch(rows, probe)
    bad["step_mask"][0, -1] = ~bad["step_mask"][0, -1]
    with pytest.raises(ValueError, match="step mask differs"):
        features(bad)


def test_sample_z_draws_the_posterior_in_training_only(vae_overrides):
    c = _cfg(vae_overrides, "classifier.source.vae.keys=[{name: mu_post}]")
    source = sources.VaeSource(c.source, sample_z=True)
    dataset = source.dataset(cohort.fold_shards(c.data, 1)["train"])
    rows = attribute_dict_collate([dataset[i] for i in range(2)])
    mean, drawn = source(rows), source(rows, sample_z=True)
    mask = mean.step_mask
    assert torch.equal(mean.values, source(rows).values)
    assert not torch.allclose(drawn.values[mask], mean.values[mask])


def _task(c, features):
    from teb_vae.classifier.model import ClassifierNet
    from teb_vae.classifier.train import ClassifierTask

    kwargs = dict(n_values=features.n_values, n_attn=1, n_ctx=0, model_cfg=c.model.model_dump(mode="json"),
                  labels_cfg=c.labels.model_dump(mode="json"), priors=None)
    task = ClassifierTask(ClassifierNet(**kwargs), lr=1e-3, weight_decay=0.01, classifier_kwargs=kwargs,
                          train_cfg=c.train.model_dump(mode="json"), class_weights=[1.0, 1.0], prior_offset=[0.0])
    task.backbone = features
    return task


def test_backbone_is_a_task_submodule_with_its_own_param_group(vae_overrides):
    from teb_vae.classifier.train import BACKBONE

    c = _cfg(vae_overrides, "classifier.train.regime=partial", f"classifier.train.unfreeze=[{ALLOW}]",
             "classifier.train.backbone_lr=1.0e-5")
    features, _, _ = _backbone(c)
    task = _task(c, features)
    assert "backbone" in dict(task.named_children())  # F11: not model/net/network/module
    decay, no_decay, backbone = task.configure_param_groups()
    assert (backbone["lr"], backbone["weight_decay"]) == (1e-5, 0.0)
    assert {id(p) for p in backbone["params"]} == {id(p) for _, p in features.source.allowlist()}
    net = {id(p) for p in task.orig_model.parameters()}
    assert {id(p) for p in decay["params"] + no_decay["params"]} == net
    task.train()
    features.source.check_frozen()  # Lightning's train() on the task leaves frozen VAE modules in eval
    keys = {k for k in task.state_dict() if k.startswith(BACKBONE)}
    assert {f"{BACKBONE}center", f"{BACKBONE}scale"} <= keys and any(k.startswith(f"{BACKBONE}vae.") for k in keys)
    assert not any(m is features.source.task for m in task.modules())  # the VAE's Lightning task stays outside


def test_lpft_callback_unfreezes_resets_early_stopping_and_records_the_transition(vae_overrides, tmp_path):
    from lightning.pytorch.callbacks import EarlyStopping

    from teb_vae.classifier.train import LpftUnfreezeCallback

    c = _cfg(vae_overrides, "classifier.train.regime=lpft", f"classifier.train.unfreeze=[{ALLOW}]",
             "classifier.train.lpft_head_epochs=2")
    features, _, _ = _backbone(c)
    for _, p in features.source.allowlist():  # stage 1 starts frozen (train.online_backbone)
        p.requires_grad_(False)
    task = _task(c, features)
    logged = []
    task.log = lambda name, value, **kw: logged.append((name, value))
    es = EarlyStopping(monitor="val/guid_logloss", mode="min", patience=3)
    trainer = SimpleNamespace(is_global_zero=True, early_stopping_callbacks=[es], current_epoch=0)
    cb = LpftUnfreezeCallback(head_epochs=2, output_dir=tmp_path)
    cb.on_fit_start(trainer, task)
    assert es.patience == float("inf")  # stage 1 runs its full length
    for epoch in (0, 1):
        trainer.current_epoch = epoch
        task.train()
        cb.on_train_epoch_start(trainer, task)
        assert not any(p.requires_grad for _, p in features.source.allowlist())
        assert not any(m.training for m in features.vae.modules())  # stage 1: the allowlist is frozen too
    es.wait_count, es.best_score = 2, torch.tensor(0.3)
    trainer.current_epoch = 2
    cb.on_train_epoch_start(trainer, task)
    assert all(p.requires_grad for _, p in features.source.allowlist())
    assert features.vae.posterior_head.training and not features.vae.target_encoder.training
    assert (es.patience, es.wait_count, float(es.best_score)) == (3, 0, float("inf"))
    features.source.check_frozen()
    lines = [json.loads(line) for line in (tmp_path / "stage_transitions.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["epoch"] == 2 and lines[0]["prefixes"] == [ALLOW]
    assert lines[0]["n_backbone_trainable"] == sum(p.numel() for _, p in features.source.allowlist())
    trainer.current_epoch = 3
    cb.on_train_epoch_start(trainer, task)  # once only
    assert len((tmp_path / "stage_transitions.jsonl").read_text().splitlines()) == 1
    assert [v for name, v in logged if name == "train/stage"] == [1.0, 1.0, 2.0, 2.0]


@pytest.mark.parametrize("overrides, match", [
    (["classifier.train.regime=partial"], "non-empty allowlist"),
    (["classifier.train.regime=frozen_online", "classifier.train.unfreeze=[prior_head]"], "non-empty allowlist"),
    (["classifier.train.unfreeze=[prior_head]"], "non-empty allowlist"),
    (["classifier.source.kind=hdf5", "classifier.train.regime=lpft", "classifier.train.unfreeze=[prior_head]"],
     "needs source.kind: vae"),
    (["classifier.source.vae.sample_z_train=true"], "frozen_online augmentation"),
])
def test_regime_schema_refusals(overrides, match):
    with pytest.raises(ValidationError, match=match):
        config.load(config.DEFAULT_CONFIG, overrides)


def test_frozen_baseline_config_and_units():
    from teb_vae.classifier.run import neural_units

    cfg = config.load(config.DEFAULT_CONFIG, ["classifier.source.kind=vae", "classifier.train.regime=lpft",
                                              "classifier.train.unfreeze=[prior_head]", "classifier.run.seeds=[1, 2]"])
    frozen = config.frozen_baseline(cfg).classifier
    assert (frozen.train.regime, frozen.train.unfreeze) == ("frozen_cached", [])
    assert cfg.classifier.train.regime == "lpft"  # a copy
    units = neural_units(cfg.classifier)
    assert ("frozen", 1) in units and ("frozen", 2) in units
    assert not [k for k, _ in neural_units(config.load(config.DEFAULT_CONFIG).classifier) if k == "frozen"]
    assert config.unit_dir("r", 1, 2, "frozen").parts[-2:] == ("seed_2", "frozen")


# ---- one online run --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_lpft_online_run_through_predict(vae_overrides, vae_checkpoint, tmp_path):
    """LPFT (stage 2 from epoch 1) on fold 1, sequence scope: the unit trains online, its last.ckpt carries the
    fine-tuned allowlist and nothing else moved, predict scores best.ckpt online beside the ``frozen`` baseline."""
    from teb_vae.classifier import run
    from teb_vae.classifier.config import unit_dir
    from teb_vae.classifier.train import backbone_state

    overrides = list(vae_overrides) + [
        "classifier.run.folds=[1]", f"classifier.source.cache_root={tmp_path / 'cache'}",
        "classifier.source.vae.keys=[{name: mu_prior}, {name: delta_mu}, {name: kld_per_t, transform: log1p, "
        "role: attention}]",
        "classifier.train.regime=lpft", f"classifier.train.unfreeze=[{ALLOW}.delta_mu_head]",
        "classifier.train.lpft_head_epochs=1", "classifier.train.max_epochs=2", "classifier.train.backbone_lr=1.0e-2",
        "classifier.baselines.shuffled_control=false", "classifier.context.auto_ablate_missing=false",
        "advanced_config.callbacks.classifier_plotting.train_eval_guids=4"]
    run_dir = tmp_path / "run"
    try:
        for stage in ("cohort", "extract", "train", "predict"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(run_dir))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert state["train"]["exit_code"] == 0 and sorted(state["train"]["units"]) == ["frozen|42|1", "model|42|1"]

    model, frozen = unit_dir(run_dir, 1, 42, "model"), unit_dir(run_dir, 1, 42, "frozen")
    lines = (model / "train_results" / "stage_transitions.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["epoch"] == 1
    setup = json.loads((model / "setup.json").read_text())
    assert setup["regime"]["name"] == "lpft" and setup["params"]["frozen"] > setup["params"]["trainable"] > 0
    assert backbone_state(model / "model_checkpoints" / "best.ckpt")  # epoch 0 or 1: either holds the backbone
    assert not backbone_state(frozen / "model_checkpoints" / "best.ckpt")
    state_ft = backbone_state(model / "model_checkpoints" / "last.ckpt")  # after stage 2's epoch
    pretrained = torch.load(vae_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    moved = {k for k, v in pretrained.items() if not torch.equal(state_ft[f"vae.{k}"], v)}
    assert moved and all(k.startswith(f"{ALLOW}.delta_mu_head.") for k in moved)  # the allowlist, nothing else
    assert "frozen_cached" in (frozen / "model_checkpoints" / "resolved_config.yaml").read_text()

    seg = pd.read_parquet(run_dir / "predictions" / "segments.parquet")
    gd = pd.read_parquet(run_dir / "predictions" / "guids.parquet")
    assert {"model", "frozen", "probe", "shortcut"} <= set(gd["model_id"])
    for m in ("model", "frozen"):
        rows = gd[gd["model_id"] == m]
        assert set(rows["split"]) == {"val", "test"} and np.isfinite(rows["score_final_cal"]).all()
        assert np.isfinite(seg.loc[seg["model_id"] == m, "logit_online_cal"]).all()
    a, b = (gd[gd["model_id"] == m].set_index(["split", "guid"])["score_final"].sort_index()
            for m in ("model", "frozen"))
    assert not np.allclose(a.to_numpy(), b.to_numpy())  # two different models, not one scored twice


# ---- co-training (§10.6) and the preservation gate (§10.10.2 #12) ------------------------------------------------------
COTRAIN = ("classifier.train.regime=cotrain", f"classifier.train.unfreeze=[{ALLOW}]")


def test_cotrain_step_runs_the_vae_once_with_the_joint_loss_and_allowlist_grads(vae_overrides):
    c = _cfg(vae_overrides, *COTRAIN, "classifier.model.scope=segment", "classifier.train.cotrain.l2sp=0.5",
             "classifier.train.cotrain.vae_weight=0.7", "classifier.train.cotrain.cls_weight=1.3")
    features, rows, probe = _backbone(c, cotrain=c.train.cotrain)
    forwards = []
    features.vae.register_forward_hook(lambda *_: forwards.append(1))
    task = _task(c, features)
    task.train()
    batch = _batch(rows, probe) | {"y": torch.tensor([0.0, 1.0, 0.0, 1.0]), "y3": torch.tensor([0, 1, 0, 2]),
                                   "w": torch.full((4,), 0.25), "guid": torch.arange(4)}
    total, m = task.compute_loss_and_metrics(batch, 0, "train")
    assert len(forwards) == 2  # 4 rows in chunks of 3: one forward per chunk, the objective's features reused
    assert all(np.isfinite(float(m[k].detach())) for k in ("total_loss", "loss_cls", "vae_total_loss", "vae_kld",
                                                           "vae_nll"))
    assert float(m["l2sp"]) == 0.0  # θ = θ₀ before any step
    torch.testing.assert_close(total, 1.3 * m["loss_cls"] + 0.7 * m["vae_total_loss"])
    total.backward()
    allowed = {n for n, _ in features.source.allowlist()}
    grads = {n: p.grad for n, p in features.vae.named_parameters()}
    assert all(grads[n] is None for n in grads if n not in allowed)
    assert sum(float(grads[n].abs().sum()) for n in allowed if grads[n] is not None) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.orig_model.parameters())  # the head too
    features.source.check_frozen()
    with torch.no_grad():
        next(p for _, p in features.source.allowlist()).add_(1.0)
    assert float(features.l2sp()) > 0
    features.eval()  # validation and scoring: the plain dense forward, no VAE objective
    assert "vae_loss" not in features(_batch(rows, probe))


def test_cotrain_last_k_trains_each_guids_last_segments_only(vae_overrides):
    c = _cfg(vae_overrides, *COTRAIN, "classifier.train.cotrain.grad_segments=last_k:1")
    features, rows, probe = _backbone(c, cotrain=c.train.cotrain)
    real = torch.tensor([[True, True, True], [True, False, False]])
    assert features._trains(real, 4).tolist() == [False, False, True, True]
    features.train()
    batch = {"vae": rows, "seg_mask": real, "x": torch.zeros(2, 3, *probe.values.shape[1:]),
             "attn": torch.zeros(2, 3, *probe.attn.shape[1:]), "step_mask": torch.zeros(2, 3, probe.step_mask.shape[1],
                                                                                        dtype=torch.bool)}
    batch["step_mask"][real] = probe.step_mask
    out = features(batch)
    torch.testing.assert_close(out["x"][real][[2, 3]], probe.values[[2, 3]], atol=1e-5, rtol=1e-5)
    out["vae_loss"].backward()  # two trained rows, one chunk
    assert any(p.grad is not None for _, p in features.source.allowlist())
    with pytest.raises(ValidationError, match="last_k"):
        _cfg(vae_overrides, *COTRAIN, "classifier.model.scope=segment", "classifier.train.cotrain.grad_segments=last_k:2")


def test_cotrain_pins_beta_to_the_pretrained_final_value(vae_overrides):
    from teb_vae.classifier.train import online_backbone

    c = _cfg(vae_overrides, *COTRAIN)
    features, _, probe = _backbone(c)
    unit = SimpleNamespace(scaler=sources.Scaler(channels=features.channels, center=np.zeros(len(features.channels)),
                                                 scale=np.ones(len(features.channels)),
                                                 keep=np.ones(len(features.channels), bool), record={}),
                           n_values=probe.values.shape[-1])
    joint = online_backbone(c, unit, trainable=True)
    task = joint.source.task
    assert task.hparams["beta_schedule"] == {"kind": "constant", "value": 1.0}  # the tiny checkpoint's kld_beta
    assert task._resolve_beta(0) == task._resolve_beta(10 ** 6) and joint.cotrain is c.train.cotrain
    assert online_backbone(c, unit, trainable=False).cotrain is None  # scoring: plain frozen forward


def test_preservation_gate_rejects_a_degraded_vae(vae_overrides, tmp_path):
    from teb_vae.classifier.train import PreservationGateCallback

    c = _cfg(vae_overrides, *COTRAIN)
    features, rows, _ = _backbone(c, cotrain=c.train.cotrain)
    task = _task(c, features)
    logged = {}
    task.log = lambda name, value, **kw: logged.__setitem__(name, value)
    task.val_metrics = {"val/guid_logloss": 0.5}
    trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False, current_epoch=0)
    gate = PreservationGateCallback(rows, tolerance=0.10, monitor="val/guid_logloss", mode="min", chunk=3,
                                    output_dir=tmp_path)
    gate.on_fit_start(trainer, task)
    gate.on_validation_epoch_end(trainer, task)
    assert logged["val/forecast_mse_rel"] == pytest.approx(0.0, abs=1e-6)  # mean decode: deterministic
    assert (logged["val/gate_ok"], logged["val/guid_logloss_gated"]) == (1.0, 0.5)
    assert all(np.isfinite(logged[f"val/{k}"]) for k in ("kld_active_frac", "logvar_prior_floor_frac",
                                                          "delta_mu_sat_frac"))
    with torch.no_grad():  # a synthetic degradation of the forecast
        for name, p in features.vae.decoder.named_parameters():
            if name.endswith("bias"):
                p.add_(3.0)
    trainer.current_epoch = 1
    gate.on_validation_epoch_end(trainer, task)
    assert logged["val/forecast_mse_rel"] > 0.10
    assert (logged["val/gate_ok"], logged["val/guid_logloss_gated"]) == (0.0, float("inf"))
    lines = [json.loads(line) for line in (tmp_path / "preservation.jsonl").read_text().splitlines()]
    assert [line["epoch"] for line in lines] == ["frozen", 0, 1]


def test_preservation_gate_reads_any_logged_monitor(monkeypatch, tmp_path):
    """The gated monitor need not be a GUID metric: any metric already logged this validation epoch (callback_metrics)
    serves; an unlogged one names itself."""
    from teb_vae.classifier import train

    monkeypatch.setattr(train, "preservation", lambda *a, **k: {"forecast_mse": 1.0, "kld_active_frac": 0.5,
                                                                "logvar_prior_floor_frac": 0.0, "delta_mu_sat_frac": 0.0})
    logged = {}
    task = SimpleNamespace(backbone=SimpleNamespace(source=None), val_metrics={"val/guid_logloss": 0.5},
                           log=lambda name, value, **kw: logged.__setitem__(name, value))
    trainer = SimpleNamespace(is_global_zero=True, sanity_checking=False, current_epoch=0,
                              callback_metrics={"val/total_loss": torch.tensor(0.7)})
    gate = train.PreservationGateCallback({}, tolerance=0.1, monitor="val/total_loss", mode="min", chunk=1,
                                          output_dir=tmp_path)
    gate.on_fit_start(trainer, task)
    gate.on_validation_epoch_end(trainer, task)
    assert logged["val/total_loss_gated"] == pytest.approx(0.7) and logged["val/gate_ok"] == 1.0
    gate.monitor = "val/not_logged"
    with pytest.raises(KeyError, match="val/not_logged"):
        gate.on_validation_epoch_end(trainer, task)


def test_cotrain_config_selects_on_the_gated_monitor():
    from teb_vae.classifier.train import tracked_metrics

    cfg = config.load(config.DEFAULT_CONFIG.parent / "cotrain.yaml")
    assert (cfg.classifier.train.regime, cfg.classifier.train.unfreeze) == ("cotrain", ["posterior_head.delta_mu_head"])
    assert config.selection_monitor(cfg.advanced_config, "cotrain") == ("val/guid_logloss_gated", "min")
    assert config.selection_monitor(cfg.advanced_config) == ("val/guid_logloss", "min")  # early stopping: ungated
    assert {"val/guid_logloss_gated", "val/gate_ok", "val/forecast_mse_rel",
            "train/vae_total_loss"} <= set(tracked_metrics(cfg.classifier, cfg.advanced_config))
    with pytest.raises(ValidationError, match="early_stopping"):
        config.load(config.DEFAULT_CONFIG.parent / "cotrain.yaml",
                    ["advanced_config.callbacks.model_checkpoint.monitor=val/guid_logloss"])


@pytest.mark.slow
def test_cotrain_online_run_through_predict_with_the_gate_and_delta_vs_frozen(vae_overrides, tmp_path):
    """Co-training on fold 1, sequence scope, last_k:2: the unit trains with the joint objective and the gate, best.ckpt
    is selected on the gated monitor, predict scores it online beside the ``frozen`` baseline, and block VF reads the
    model against it."""
    from teb_vae.classifier import metrics as M, run
    from teb_vae.classifier.config import unit_dir
    from teb_vae.classifier.report import _vs_frozen_md

    overrides = list(vae_overrides) + [
        "classifier.run.folds=[1]", f"classifier.source.cache_root={tmp_path / 'cache'}",
        "classifier.source.vae.keys=[{name: mu_prior}, {name: delta_mu}, {name: kld_per_t, transform: log1p, "
        "role: attention}]", *COTRAIN, "classifier.train.max_epochs=2", "classifier.train.backbone_lr=1.0e-3",
        "classifier.train.cotrain.grad_segments=last_k:2", "classifier.baselines.shuffled_control=false",
        "classifier.context.auto_ablate_missing=false", "classifier.eval.bootstrap.resamples=20",
        "advanced_config.callbacks.classifier_plotting.train_eval_guids=4"]
    run_dir = tmp_path / "run"
    try:
        for stage in ("cohort", "extract", "train", "predict"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(run_dir))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert state["train"]["exit_code"] == 0 and sorted(state["train"]["units"]) == ["frozen|42|1", "model|42|1"]
    model = unit_dir(run_dir, 1, 42, "model")
    record = json.loads((model / "fold_results.json").read_text())
    assert record["monitor"] == "val/guid_logloss_gated"
    history = pd.read_csv(model / "train_results" / "metrics_history.csv")
    for col in ("train/vae_total_loss", "train/vae_nll", "val/forecast_mse_rel", "val/gate_ok",
                "val/guid_logloss_gated"):
        assert history[col].notna().any(), col
    lines = (model / "train_results" / "preservation.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["epoch"] == "frozen" and len(lines) == 1 + record["epochs_run"]
    gd = pd.read_parquet(run_dir / "predictions" / "guids.parquet")
    for m in ("model", "frozen"):
        assert np.isfinite(gd.loc[gd["model_id"] == m, "score_final_cal"]).all()

    cfg = config.load(run_dir / "config.resolved.yaml")
    ctx = M.load_context(run_dir, cfg)
    res = M.run_VF(ctx, eval_config=ctx.cfg["eval"], out_dir=tmp_path)
    table = pd.DataFrame(res["table"])
    assert set(table["model_id"]) == {"model"} and "delta_auroc" in set(table["metric"])
    assert "Δ vs frozen" in "\n".join(_vs_frozen_md({"VF": res}))


def test_vf_pairs_model_with_frozen_on_shared_rows_only(tmp_path):
    """Block VF: ``model`` seeds only (not ``noind`` / ``*_covoff``) against ``frozen`` at the seed, on the (fold, guid)
    rows both have: a frozen unit missing fold 3 (``--allow-partial``) with identical scores gives Δ 0, not the
    difference between two populations, and the dropped rows are an L14 inclusion record."""
    from teb_vae.classifier import metrics as M
    from teb_vae.classifier.tests.test_eval_analyses import make_run

    run, cfg = make_run(tmp_path, resamples=20)
    pred = run / "predictions"
    seg, gd = pd.read_parquet(pred / "segments.parquet"), pd.read_parquet(pred / "guids.parquet")
    thr = json.loads((pred / "thresholds.json").read_text())
    copies = [("model", "42", ()), ("frozen", "42", (3,)), ("noind", "42", ()), ("model_covoff", "42", ())]
    for m, sd, drop in copies:
        seg = pd.concat([seg, seg[(seg["model_id"] == "probe") & ~seg["fold"].isin(drop)].assign(model_id=m, seed=sd)])
        gd = pd.concat([gd, gd[(gd["model_id"] == "probe") & ~gd["fold"].isin(drop)].assign(model_id=m, seed=sd)])
        thr |= {f"{m}|{sd}|{k}": thr[f"probe|42|{k}"] for k in (1, 2, 3) if k not in drop}
    seg.to_parquet(pred / "segments.parquet", index=False)
    gd.to_parquet(pred / "guids.parquet", index=False)
    (pred / "thresholds.json").write_text(json.dumps(thr))
    ctx = M.load_context(run, cfg)
    res = M.run_VF(ctx, eval_config=ctx.cfg["eval"], out_dir=tmp_path)
    t = pd.DataFrame(res["table"])
    assert set(t["model_id"]) == {"model"}
    whole = t[t["subgroup"].isna()].set_index("metric")
    assert whole.loc["delta_auroc", "value_raw"] == 0.0 and (whole["n_unpaired"] > 0).all()
    assert ctx.inclusion[-1]["analysis"] == "VF" and ctx.inclusion[-1]["n_excluded"] == whole["n_unpaired"].iloc[0]


@pytest.mark.slow
def test_cotrain_unit_whose_gate_fails_every_epoch_fails_and_is_never_locked(vae_overrides, tmp_path):
    """§10.6: only gate-passing epochs are selectable. With a tolerance no epoch can meet (forecast_mse_rel ≥ -1), no
    checkpoint is selectable: the unit is ``failed`` (train exits 1), keeps ``gate_failed_every_epoch``, writes no
    selection lock, so no gate-failing ``model`` row can be predicted; the frozen baseline trains and locks as usual."""
    from teb_vae.classifier import run
    from teb_vae.classifier.baselines import LOCK
    from teb_vae.classifier.config import unit_dir

    overrides = list(vae_overrides) + [
        "classifier.run.folds=[1]", f"classifier.source.cache_root={tmp_path / 'cache'}",
        "classifier.source.vae.keys=[{name: mu_prior}, {name: kld_per_t, transform: log1p, role: attention}]",
        *COTRAIN, "classifier.train.max_epochs=1", "classifier.train.cotrain.gates={forecast_mse_rel: -1.0}",
        "classifier.baselines.shuffled_control=false", "classifier.context.auto_ablate_missing=false",
        "advanced_config.callbacks.classifier_plotting.train_eval_guids=4"]
    run_dir = tmp_path / "run"
    try:
        for stage in ("cohort", "extract", "train"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(run_dir))
    finally:
        _restore_loguru()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert state["train"]["exit_code"] == 1
    assert state["train"]["units"]["model|42|1"]["status"] == "failed"
    assert state["train"]["units"]["frozen|42|1"]["status"] == "done"
    model = unit_dir(run_dir, 1, 42, "model")
    record = json.loads((model / "fold_results.json").read_text())
    assert record["status"] == "failed" and record["gate_failed_every_epoch"] is True
    assert "preservation gate failed at every epoch" in record["error"]
    assert not (model / LOCK).exists() and (unit_dir(run_dir, 1, 42, "frozen") / LOCK).exists()
