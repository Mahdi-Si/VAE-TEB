"""train.py and the unit config: framework integration (SPEC §10.10, §13.2, §15 T-F1, T-F2, T-F3, T-F5), the
optimiser/scheduler overrides, the §10.2 prior correction and §10.8 calibration, on fold 1 of the fixture tree."""
from __future__ import annotations

import atexit
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError

from teb_vae.classifier import config
from teb_vae.classifier.tests.conftest import SMOKE_CONFIG

SHIPPED = sorted(config.DEFAULT_CONFIG.parent.glob("*.yaml"))
SCOPES = ("sequence", "segment")
SEED = {"sequence": 42, "segment": 7}  # one run dir, two units
#: The early-stopping monitor per scope; best.ckpt must follow it (§10.10.2 #6).
MONITOR = {"sequence": ("val/guid_logloss", "min"), "segment": ("val/guid_auroc", "max")}
EPOCHS = 2


def _restore_loguru() -> None:
    """run.main / train_unit replace loguru's sinks with files under tmp."""
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr)


@pytest.fixture(scope="module")
def env(smoke_overrides, tmp_path_factory):
    """cohort -> extract (fold 1) into a private tmp tree: ``(overrides, run_dir, manifest)``."""
    from teb_vae.classifier import run

    tmp = tmp_path_factory.mktemp("train")
    overrides = list(smoke_overrides) + ["classifier.run.folds=[1]", f"classifier.source.cache_root={tmp / 'cache'}",
                                         f"classifier.train.max_epochs={EPOCHS}"]
    try:
        for stage in ("cohort", "extract"):
            run.main(config=str(SMOKE_CONFIG), stage=stage, overrides=overrides, run_dir=str(tmp / "run"))
    finally:
        _restore_loguru()
    return overrides, tmp / "run", json.loads((tmp / "run" / "manifest.json").read_text())


def _cfg(env, *sets):
    return config.load(SMOKE_CONFIG, env[0] + list(sets))


@pytest.fixture(scope="module")
def trained(env):
    """T-F2: one 2-epoch CPU unit per scope, MLflow off: ``{scope: (cfg, record, unit)}``."""
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.train import train_unit

    out = {}
    try:
        for scope in SCOPES:
            monitor, mode = MONITOR[scope]
            es = "advanced_config.callbacks.early_stopping"
            cfg = _cfg(env, f"classifier.model.scope={scope}", f"{es}.monitor={monitor}", f"{es}.mode={mode}")
            unit = build_unit(cfg, env[1], env[2]["source"], 1)
            out[scope] = cfg, train_unit(cfg, env[1], env[2], fold=1, seed=SEED[scope], kind="model", unit=unit), unit
    finally:
        _restore_loguru()
    return out


def _history(record) -> pd.DataFrame:
    return pd.read_csv(f"{record['unit_dir']}/train_results/metrics_history.csv")


def _summary(record) -> list:
    return [json.loads(line) for line in open(f"{record['unit_dir']}/train_results/epoch_summary.jsonl")]


# ---- T-F1: configs -> unit config -> the framework's validator ---------------------------------------------------
@pytest.fixture
def loguru_warnings():
    """``validate_config`` warns through loguru, not ``warnings`` (test_config_load.py pattern)."""
    from loguru import logger

    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.name)
def test_every_config_derives_a_unit_config_the_framework_accepts(path, tmp_path, loguru_warnings):
    from train.test_utils import make_graph_model

    cfg = config.load(path)
    for kind in ("model", "shuffled"):
        derived = config.unit_config(cfg, run_dir=tmp_path, fold=3, seed=42, kind=kind, mlflow_parent_id="p0")
        unit_yaml = tmp_path / f"{kind}.yaml"
        unit_yaml.write_text(yaml.safe_dump(derived, sort_keys=False))
        make_graph_model(unit_yaml).validate_config()
        general, mlflow = derived["general_config"], derived["advanced_config"]["tracking"]["mlflow"]
        assert general["seed"] == 42 + 3000 and general["epochs"] == cfg.classifier.train.max_epochs
        assert general["cuda_devices"] == ([] if cfg.classifier.run.device == "cpu" else [0])
        assert general["folders_config"]["out_dir_base"] == str(config.unit_dir(tmp_path, 3, 42, kind))
        assert derived["model_config"]["classifier"] == cfg.classifier.model_dump(mode="json")
        assert (mlflow["log_model"], mlflow["log_checkpoints"]) == (False, False)  # F10
        assert mlflow["run_name"] == f"fold3-seed42-{kind}"
        assert mlflow["tags"] == {"mlflow.parentRunId": "p0", "fold": "3", "seed": "42", "kind": kind}
    assert [message for message in loguru_warnings if "config:" in message] == []
    assert config.unit_dir(tmp_path, 3, 42, "shuffled") == tmp_path / "folds" / "fold_3" / "seed_42" / "shuffled"


@pytest.mark.parametrize("overrides, match", [
    (["classifier.model.scope=segment", "classifier.labels.strategy=mil"], "mil needs model.scope: sequence"),
    (["classifier.model.sequence.causal=false"], "causal: false requires"),
    (["classifier.train.loss.name=focal"], "focal_alpha"),
    (["classifier.train.loss.name=auc_margin"], "needs train.sampler: class_balanced"),
    (["classifier.train.loss.name=pauc", "classifier.train.sampler=class_balanced", "classifier.calibration.method=none"],
     "not probabilities"),
    (["classifier.train.cotrain.grad_segments=last_k:0"], "grad_segments"),
    (["classifier.labels.task=three_class", "classifier.labels.head=ordinal",
      "classifier.train.loss.name=cumulative_link"], "aux_3class_weight must be 0"),
])
def test_schema_refusals(overrides, match):
    with pytest.raises(ValidationError, match=match):
        config.load(config.DEFAULT_CONFIG, overrides)
    config.load(config.DEFAULT_CONFIG, ["classifier.train.loss.name=focal", "classifier.train.loss.focal_alpha=0.5"])
    for name in ("auc_margin", "pauc"):
        config.load(config.DEFAULT_CONFIG, [f"classifier.train.loss.name={name}", "classifier.train.sampler=class_balanced"])


def test_early_stopping_and_best_ckpt_share_one_default_monitor(tmp_path):
    """An early_stopping block without monitor/mode gets the checkpoint's (selection_monitor) in the unit config; the
    framework's own EarlyStopping default (val/total_loss) would otherwise select on another metric."""
    cfg = config.load(config.DEFAULT_CONFIG)
    advanced = json.loads(json.dumps(cfg.advanced_config))
    for key in ("monitor", "mode"):
        advanced["callbacks"]["early_stopping"].pop(key)
    cfg = cfg.model_copy(update={"advanced_config": advanced})
    es = config.unit_config(cfg, run_dir=tmp_path, fold=1, seed=42, kind="model")["advanced_config"]["callbacks"][
        "early_stopping"]
    assert (es["monitor"], es["mode"]) == config.selection_monitor(cfg.advanced_config) == ("val/guid_logloss", "min")


# ---- the task: F11, parameter groups, optimiser, schedule, prior correction ----------------------------------------
def _task(**labels):
    from teb_vae.classifier.model import ClassifierNet
    from teb_vae.classifier.train import ClassifierTask

    c = config.load(SMOKE_CONFIG).classifier  # labels bypass the root validator (the head / loss / λ3 checks)
    kwargs = dict(n_values=3, n_attn=1, n_ctx=2, model_cfg=c.model.model_dump(mode="json"),
                  labels_cfg=c.labels.model_copy(update=labels).model_dump(mode="json"), priors=None)
    return ClassifierTask(ClassifierNet(**kwargs), lr=1e-3, weight_decay=0.01, classifier_kwargs=kwargs,
                          train_cfg=c.train.model_dump(mode="json"), class_weights=[1.0] * 3, prior_offset=[0.0]), c


def test_task_naming_param_groups_and_optimizer():
    from train.graph_models_utils import _clean_state_dict

    task, c = _task(task="three_class", head="ordinal")
    net = task.orig_model
    assert not {name.split(".")[-1] for name, _ in net.named_modules()} & {"model", "net", "network", "module"}  # F11
    assert set(_clean_state_dict(task.state_dict())) == set(net.state_dict())
    assert task.model is net  # F1: eager, never torch.compile
    decay, no_decay = task.configure_param_groups()
    names = {id(p): n for n, p in net.named_parameters()}
    nd = {names[id(p)] for p in no_decay["params"]}
    assert {"head.bias_first", "head.bias_gaps", "aggregator.time_bias.weight", "head.mlp.0.bias",
            "step_encoder.proj.1.weight"} <= nd  # CORAL biases, time bias (an embedding), a bias, a LayerNorm
    assert "head.mlp.0.weight" in {names[id(p)] for p in decay["params"]}
    assert (decay["weight_decay"], no_decay["weight_decay"]) == (0.01, 0.0)
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(net.parameters()))
    assert task.build_optimizer([decay, no_decay]).defaults["betas"] == (0.9, 0.999)  # F2


def test_per_step_warmup_then_cosine_to_the_floor():
    task, c = _task()
    task._trainer = SimpleNamespace(estimated_stepping_batches=105)
    sched = task.build_lr_scheduler(torch.optim.SGD(task.parameters(), lr=1.0))
    f, warm, floor = sched["scheduler"].lr_lambdas[0], c.train.schedule.warmup_steps, c.train.schedule.min_lr_frac
    assert sched["interval"] == "step"  # F3
    assert f(0) == pytest.approx(1 / warm) and f(warm - 1) == pytest.approx(1.0) and f(warm) == pytest.approx(1.0)
    assert f(warm + 50) == pytest.approx(floor + (1 - floor) / 2) and f(105) == pytest.approx(floor)


def test_prior_correction_known_answers():
    from teb_vae.classifier.train import prior_correction

    unit = SimpleNamespace(class_counts=[30, 10], priors={"main": [0.75, 0.25]})

    def offset(*sets):
        return prior_correction(config.load(config.DEFAULT_CONFIG, list(sets)).classifier, unit)

    assert offset() == [0.0]
    assert offset("classifier.train.loss.name=weighted_bce", "classifier.train.loss.weighting=inverse") == \
        pytest.approx([math.log(3)])  # s - log(w1/w0), w = 1/n
    assert offset("classifier.train.loss.name=logit_adjusted") == pytest.approx([math.log(3)])  # s + tau log(pi1/pi0)
    assert offset("classifier.train.loss.name=weighted_ce", "classifier.labels.head=multiclass",
                  "classifier.train.loss.weighting=inverse") == pytest.approx([0.0, math.log(3)])
    # class_balanced trains under a uniform prior: the inverse-weighting offset log(n0/n1), added to any weighting's
    assert offset("classifier.train.sampler=class_balanced") == pytest.approx([math.log(3)])
    assert offset("classifier.train.sampler=class_balanced", "classifier.train.loss.name=weighted_bce",
                  "classifier.train.loss.weighting=inverse") == pytest.approx([2 * math.log(3)])
    assert offset("classifier.train.sampler=class_balanced", "classifier.labels.head=multiclass",
                  "classifier.train.loss.name=ce") == pytest.approx([0.0, math.log(3)])


# ---- T-F3: callback order and the trainer kwargs ------------------------------------------------------------------
def test_callback_order_cpu_and_unit_dirs(env, tmp_path):
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.train import ClassifierTrainer

    cfg = _cfg(env, "classifier.train.ema=0.9")
    path = tmp_path / "unit.yaml"
    path.write_text(yaml.safe_dump(config.unit_config(cfg, run_dir=tmp_path, fold=1, seed=42, kind="shuffled")))
    out = config.unit_dir(tmp_path, 1, 42, "shuffled")
    gm = ClassifierTrainer(path, unit_dir=out, unit=build_unit(cfg, env[1], env[2]["source"], 1, shuffle_seed=42))
    kw = gm._build_trainer_kwargs(gm._callbacks())
    assert [type(cb).__name__ for cb in kw["callbacks"]] == [
        "GuidEpochMetricsCallback", "MetricsLoggingCallback", "MetricsHistoryCsvCallback", "LossPlotCallback",
        "HyperparameterLoggingCallback", "ClassifierPlotCallback", "ModelCheckpoint", "EarlyStopping",
        "LearningRateMonitor", "EMAWeightAveraging"]
    assert kw["callbacks"][8].logging_interval == "step" and kw["callbacks"][9].should_update(step_idx=0)
    assert (kw["accelerator"], kw["devices"], kw["use_distributed_sampler"]) == ("cpu", 1, False)  # F5
    assert "strategy" not in kw and "profiler" not in kw
    assert (gm.train_results_dir, gm.model_checkpoint_dir) == (str(out / "train_results"),
                                                               str(out / "model_checkpoints"))  # F6


# ---- T-F2 / T-F3: the train smoke, both scopes ---------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("scope", SCOPES)
def test_train_smoke_files_metrics_and_checkpoint(trained, scope):
    from teb_vae.classifier.model import ClassifierNet
    from teb_vae.classifier.train import tracked_metrics
    from train.graph_models_utils import check_model_class, load_checkpoint_strict

    cfg, record, _ = trained[scope]
    assert record["status"] == "done" and record["epochs_run"] == EPOCHS, record
    root = record["unit_dir"]
    for rel in ("setup.json", "scaler.json", "fold_results.json", "model_checkpoints/resolved_config.yaml",
                "model_checkpoints/best.ckpt", "model_checkpoints/last.ckpt", "train_results/full.log",
                "train_results/loss_plot_epoch.html", "train_results/hyperparameters.html",
                "train_results/classifier_diagnostics/epoch0001_diagnostics.pdf"):
        assert (Path(root) / rel).is_file(), rel

    history = _history(record)
    names = tracked_metrics(cfg.classifier, cfg.advanced_config)
    assert "val/macro_f1" in names and "train/spike_skipped" not in names  # aux 3-class head on, breaker off
    assert [n for n in names if n not in history] == []
    assert [n for n in names if history[n].isna().all()] == []
    assert list(history["epoch"]) == list(range(EPOCHS))
    assert [line["epoch"] for line in _summary(record)] == list(range(EPOCHS))

    blob = torch.load(record["best_ckpt"], map_location="cpu", weights_only=False)
    check_model_class(blob, ClassifierNet.__name__)
    assert load_checkpoint_strict(ClassifierNet(**blob["classifier_kwargs"]), blob) is not None
    assert {"source_fingerprint", "scaler", "labels", "feature_channels"} <= set(blob)
    assert blob["hyper_parameters"]["compile_model"] is False
    setup = json.loads(open(f"{root}/setup.json").read())
    assert setup["params"]["frozen"] == 0 and setup["splits"]["val"]["n_guids"] == 12


@pytest.mark.slow
@pytest.mark.parametrize("scope", SCOPES)
def test_guid_metrics_land_in_their_own_epoch_row(trained, scope):
    """T-F3: the callback's epoch-e value is in CSV row e (F4), and best.ckpt follows the early-stopping monitor
    (§10.10.2 #6; AUROC/max for the segment unit)."""
    _, record, _ = trained[scope]
    history, summary = _history(record), _summary(record)
    monitor, mode = MONITOR[scope]
    np.testing.assert_allclose(history[monitor], [line[monitor] for line in summary], rtol=1e-6)
    best = history[monitor].idxmax() if mode == "max" else history[monitor].idxmin()  # first of ties, as Lightning
    assert record["monitor"] == monitor and record["best_epoch"] == history["epoch"][best]
    assert record["best_score"] == pytest.approx(history[monitor][best], rel=1e-6)
    assert record["best_val"]["epoch"] == record["best_epoch"]


@pytest.mark.slow
@pytest.mark.parametrize("scope", SCOPES)
def test_score_split_rows_are_finite_and_reproduce_validation(trained, scope):
    from teb_vae.classifier.train import guid_metrics, score_split

    cfg, record, unit = trained[scope]
    for split in ("val", "test"):
        seg, gd = score_split(record["unit_dir"], unit, split)
        frame = unit.frames[split]
        assert len(seg) == len(frame) and seg["row"].tolist() == frame["row"].tolist()
        assert sorted(gd["guid"]) == sorted(frame["guid"].unique())
        assert np.isfinite(seg[["logit_seg", "logit_online", "p_c0", "p_c1", "p_c2"]].to_numpy()).all()
        aggs = [f"score_{a}" for a in cfg.classifier.model.segment_aggregators[1:]] if scope == "segment" else []
        assert np.isfinite(gd[["score_final", *aggs]].to_numpy()).all()
        last = seg.groupby("guid").tail(1).set_index("guid")["logit_online"]
        np.testing.assert_allclose(gd.set_index("guid")["score_final"], last.loc[gd["guid"]])
    assert guid_metrics(seg, gd)["n_val_guids"] == len(gd)
    seg, gd = score_split(record["unit_dir"], unit, "val")  # best.ckpt reproduces the selected epoch
    assert guid_metrics(seg, gd)[record["monitor"]] == pytest.approx(record["best_score"], rel=1e-5)


@pytest.mark.slow
def test_non_causal_unit_has_no_online_score_and_locks_guid_level(env):
    """§9.1: a non-causal sequence unit writes ``logit_online`` = NaN on every row (its positions see later segments);
    without a segment head it has no causal per-position score, so it is thresholded like the shortcut."""
    from teb_vae.classifier import run
    from teb_vae.classifier.baselines import label_rows
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.train import score_split, train_unit

    cfg = _cfg(env, "classifier.model.sequence.causal=false", "classifier.labels.strategy=final_only",
               "classifier.model.segment_head=false", "classifier.train.max_epochs=1")
    unit = build_unit(cfg, env[1], env[2]["source"], 1)
    try:
        record = train_unit(cfg, env[1], env[2], fold=1, seed=11, kind="model", unit=unit)
    finally:
        _restore_loguru()
    assert record["status"] == "done", record
    for split in ("val", "test"):
        seg, gd = score_split(record["unit_dir"], unit, split)
        assert seg["logit_online"].isna().all() and seg["logit_seg"].isna().all()
        assert np.isfinite(gd["score_final"]).all()
    index, n_valid = run._cache_rows(env[2]["source"])
    out = Path(record["unit_dir"])
    run.lock_unit(cfg, out, unit, label_rows(env[1], 1, index, ["val"], n_valid), "model", 11, None)
    thr = json.loads((out / "thresholds.json").read_text())["guid"]
    assert (out / "selection_lock.json").is_file()
    for pid in ("inst30_1h", "cum30_1h", "ovr30_1h"):
        assert thr[pid] == {"skipped": "guid-level model"}
    assert {thr[pid]["basis"] for pid in ("np30", "emp30", "emp15", "youden")} == {"guid_final"}


# ---- T-F5: teardown ------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_teardown_after_units_and_failure_records(env, monkeypatch):
    """Two units (one ok, one failing) with a fake MLflow run: nothing registered at exit, every monitor stopped,
    logs uploaded explicitly, no live net; a failure is recorded and re-raised only with ``run.fail_fast``."""
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.model import ClassifierNet
    from teb_vae.classifier.train import ClassifierTrainer, train_unit
    from train.test_utils import FakeMLflowLogger

    loggers, monitors = [], []

    class Monitor:
        finished = False

        def finish(self):
            self.finished = True

    def fake_mlflow(self):
        self.mlflow_logger = FakeMLflowLogger(run_id=f"run-{len(loggers)}")
        self.lightning_loggers = []
        loggers.append(self.mlflow_logger)

    def fake_monitor(self):
        self._system_metrics_monitor = Monitor()
        monitors.append(self._system_metrics_monitor)

    def boom(self, *args, **kwargs):
        raise RuntimeError("planted failure")

    def live_nets() -> int:
        gc.collect()
        return sum(type(o) is ClassifierNet for o in gc.get_objects())

    registered, real_register = [], atexit.register
    monkeypatch.setattr(atexit, "register", lambda f, *a, **k: (registered.append(f), real_register(f, *a, **k))[1])
    monkeypatch.setattr(ClassifierTrainer, "_init_mlflow_logger", fake_mlflow)
    monkeypatch.setattr(ClassifierTrainer, "_start_system_metrics_monitor", fake_monitor)
    cfg = _cfg(env, "classifier.model.scope=segment", "classifier.train.max_epochs=1")
    unit = build_unit(cfg, env[1], env[2]["source"], 1)
    nets = live_nets()
    cuda = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    try:
        ok = train_unit(cfg, env[1], env[2], fold=1, seed=101, kind="model", unit=unit)
        monkeypatch.setattr(ClassifierTrainer, "train_model", boom)
        failed = train_unit(cfg, env[1], env[2], fold=1, seed=102, kind="shuffled")
        with pytest.raises(RuntimeError, match="planted failure"):
            train_unit(_cfg(env, "classifier.model.scope=segment", "classifier.train.max_epochs=1",
                            "classifier.run.fail_fast=true"), env[1], env[2], fold=1, seed=103, kind="model", unit=unit)
    finally:
        _restore_loguru()
    assert ok["status"] == "done" and failed["status"] == "failed" and "planted failure" in failed["error"]
    on_disk = json.loads(open(f"{failed['unit_dir']}/fold_results.json").read())
    assert on_disk["status"] == "failed" and "Traceback" in on_disk["error"] and failed["unit_dir"].endswith("shuffled")
    uploads = [f for f in registered if getattr(f, "__name__", "") == "upload_run_logs"]
    at_exit = atexit._ncallbacks()
    for f in uploads:  # a still-registered upload would lower the count here
        atexit.unregister(f)
    assert len(uploads) == 3 and atexit._ncallbacks() == at_exit
    assert len(monitors) == 3 and all(m.finished for m in monitors)
    assert all(any(c[0] == "log_artifact" and c[2][0].endswith("full.log") for c in lg.experiment.calls)
               for lg in loggers)
    assert live_nets() == nets
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == cuda


# ---- calibration (§10.8) -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 2.0, 20000)
    return z, (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(int)


def test_temperature_recovers_a_known_temperature(synthetic):
    from teb_vae.classifier.train import apply_calibration, fit_calibration

    z, y = synthetic
    cal = fit_calibration(2.5 * z, y, "temperature")  # over-confident by T = 2.5
    assert cal["temperature"] == pytest.approx(2.5, rel=0.05)
    np.testing.assert_allclose(apply_calibration(2.5 * z, cal), z * 2.5 / cal["temperature"])
    assert fit_calibration(z, y, "temperature")["temperature"] == pytest.approx(1.0, rel=0.05)


def test_platt_and_none(synthetic):
    from teb_vae.classifier.train import apply_calibration, fit_calibration

    z, y = synthetic
    cal = fit_calibration((z - 1.0) / 2.0, y, "platt")  # true logit = 2 s + 1
    assert (cal["a"], cal["b"]) == (pytest.approx(2.0, rel=0.05), pytest.approx(1.0, abs=0.1))
    np.testing.assert_allclose(apply_calibration([0.0, 1.0], cal), [cal["b"], cal["a"] + cal["b"]])
    assert fit_calibration(z, y, "none") == {"method": "none"}
    np.testing.assert_array_equal(apply_calibration(z, {"method": "none"}), z)
    with pytest.raises(ValueError, match="isotonic"):
        fit_calibration(z, y, "isotonic")


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _draw(P, rng):
    return (rng.random(len(P))[:, None] > P.cumsum(1)[:, :-1]).sum(1)


def test_multiclass_temperature_recovers_a_known_temperature():
    from teb_vae.classifier.train import alarm_logit, calibrate_probs, fit_calibration

    rng = np.random.default_rng(1)
    z = rng.normal(0.0, 1.5, (20000, 3))
    y, raw = _draw(_softmax(z), rng), _softmax(2.0 * z)  # the model is over-confident by T = 2
    cal = fit_calibration(alarm_logit(raw), y, "temperature", p3=raw)
    assert (cal["head"], cal["temperature"]) == ("multiclass", pytest.approx(2.0, rel=0.05))
    np.testing.assert_allclose(calibrate_probs(raw, cal), _softmax(2.0 * z / cal["temperature"]), atol=1e-9)
    assert fit_calibration(0, y, "none", p3=raw) == {"method": "none", "head": "multiclass"}
    with pytest.raises(ValueError, match="platt"):
        fit_calibration(0, y, "platt", p3=raw)


def test_ordinal_refit_keeps_offsets_ordered_and_the_ranking():
    """Truth P(Y > k) = σ(0.5 g + b_k), b = (1, -1.5); the head's raw g is over-confident by 2 and its biases are
    arbitrary. The refit recovers scale and offsets, and every calibrated cut-point ranks like g."""
    from teb_vae.classifier.train import _coral_logp, alarm_logit, apply_calibration, calibrate_probs, fit_calibration

    rng = np.random.default_rng(2)
    g = rng.normal(0.0, 3.0, 20000)
    y = _draw(np.exp(_coral_logp(0.5 * g[:, None] + np.array([1.0, -1.5]))), rng)
    b_raw = np.array([0.3, -0.2])
    cal = fit_calibration(g + b_raw[0], y, "temperature", ord_score=g)
    assert cal["head"] == "ordinal" and cal["scale"] == pytest.approx(0.5, rel=0.05)
    np.testing.assert_allclose(cal["offsets"], [1.0, -1.5], atol=0.1)
    assert cal["offsets"][0] >= cal["offsets"][1] and cal["temperature"] == pytest.approx(1 / cal["scale"])
    P = calibrate_probs(np.full((g.size, 3), np.nan), cal, g)
    np.testing.assert_allclose(P.sum(1), 1.0)
    s_cal = apply_calibration(g + b_raw[0], cal)  # the alarm logit stays logit P_cal(Y >= 1), increasing in g
    np.testing.assert_allclose(s_cal, alarm_logit(P), atol=1e-6)
    order = np.argsort(g)
    assert (np.diff(s_cal[order]) >= 0).all() and (np.diff(P[order, 2]) >= 0).all()
    # a degenerate val set (no middle class) still returns ordered offsets
    hard = fit_calibration(g + b_raw[0], np.where(y == 1, 2, y), "temperature", ord_score=g)
    assert hard["offsets"][0] >= hard["offsets"][1]
    sep = fit_calibration(g, 2 * (g > 0), "temperature", ord_score=g)  # separable: the MLE runs to a -> inf
    assert sep["scale"] == pytest.approx(100.0) and np.isfinite(sep["offsets"]).all()


def test_calibrate_frames_keeps_the_alarm_logit_on_the_calibrated_probabilities():
    from teb_vae.classifier.train import alarm_logit, apply_calibration, calibrate_frames, fit_calibration

    rng = np.random.default_rng(3)
    z, zs = rng.normal(0.0, 2.0, (9, 3)), rng.normal(0.0, 2.0, (9, 3))
    P, Ps = _softmax(z), _softmax(zs)
    seg = pd.DataFrame({"guid": np.repeat(["a", "b", "c"], 3), "seg_pos": np.tile([0, 1, 2], 3),
                        "logit_seg": alarm_logit(Ps), "logit_online": alarm_logit(P),
                        **{f"p_c{k}": P[:, k] for k in range(3)}, **{f"pseg_c{k}": Ps[:, k] for k in range(3)}})
    seg.loc[0, "logit_online"] = np.nan  # a non-causal row stays NaN
    last = seg.groupby("guid").tail(1)
    gd = pd.DataFrame({"guid": last["guid"].to_numpy(), "score_final": last["logit_online"].to_numpy(),
                       **{f"p_c{k}": last[f"p_c{k}"].to_numpy() for k in range(3)}})
    cal = fit_calibration(0, rng.integers(0, 3, 9), "temperature", p3=P) | {"temperature": 2.0}
    s, g = calibrate_frames(seg, gd, cal)
    pc = s[[f"p_c{k}_cal" for k in range(3)]].to_numpy()
    np.testing.assert_allclose(pc, _softmax(z / 2.0))
    np.testing.assert_allclose(s["logit_online_cal"][1:], alarm_logit(pc)[1:])
    assert np.isnan(s["logit_online_cal"][0]) and s["ord_score"].isna().all()
    np.testing.assert_allclose(s["logit_seg_cal"], alarm_logit(_softmax(zs / 2.0)))  # the segment head's own output
    np.testing.assert_allclose(1 / (1 + np.exp(-g["score_final_cal"])), 1 - g["p_c0_cal"])
    # segment scope: the online and GUID scores re-aggregate the calibrated segment logits
    seg_scope = cal | {"segment_scope": {"aggregators": ["max", "mean"], "lse_tau": 1.0}}
    s, g = calibrate_frames(seg.assign(logit_online=0.0), gd.assign(score_mean=0.0), seg_scope)
    np.testing.assert_allclose(s["logit_seg_cal"], alarm_logit(pc))
    np.testing.assert_allclose(s["logit_online_cal"], s.groupby("guid")["logit_seg_cal"].cummax())
    np.testing.assert_allclose(g["score_final_cal"], s.groupby("guid")["logit_seg_cal"].max().to_numpy())
    np.testing.assert_allclose(g["score_mean_cal"], s.groupby("guid")["logit_seg_cal"].mean().to_numpy())
    with pytest.raises(ValueError, match="class probabilities"):
        apply_calibration([0.0], cal)


def test_three_class_calibration_stays_finite_when_probabilities_underflow():
    """A separable val fold drives T to its 0.01 floor, and float32 softmax stores exact zeros: in probability space the
    calibrated p underflowed to 0 and the alarm logit to ±inf, which crashed evaluate. Log space keeps both finite, the
    ranking intact, and every stored p_c<k>_cal inside (0, 1), so the OvR logits are finite too."""
    from scipy.special import logit

    from teb_vae.classifier.train import calibrate_frames

    P = np.array([[0.9995, 4e-4, 1e-4], [0.0, 0.3, 0.7], [0.2, 0.0, 0.8]])
    probs = {f"p_c{k}": P[:, k] for k in range(3)}
    seg = pd.DataFrame({"guid": ["a", "b", "c"], "seg_pos": 0, "logit_seg": np.nan, "logit_online": 0.0, **probs})
    gd = pd.DataFrame({"guid": ["a", "b", "c"], "score_final": 0.0, **probs})
    for cal, g_score in (({"method": "temperature", "head": "multiclass", "temperature": 0.01}, None),
                         ({"method": "temperature", "head": "ordinal", "scale": 100.0, "offsets": [1.0, -1.0],
                           "shift": 0.0}, np.array([-60.0, 60.0, 5.0]))):
        s, g = calibrate_frames(seg.assign(ord_score=g_score) if g_score is not None else seg,
                                gd.assign(ord_score=g_score) if g_score is not None else gd, cal)
        pc = g[[f"p_c{k}_cal" for k in range(3)]].to_numpy()
        assert np.isfinite(g["score_final_cal"]).all() and np.isfinite(s["logit_online_cal"]).all()
        assert np.isfinite(logit(pc)).all() and np.isfinite(logit(s[[f"p_c{k}_cal" for k in range(3)]].to_numpy())).all()
        np.testing.assert_allclose(pc.sum(1), 1.0, atol=1e-9)
    s, g = calibrate_frames(seg, gd, {"method": "temperature", "head": "multiclass", "temperature": 0.01})
    assert g["score_final_cal"].iloc[0] < -700 < 700 < g["score_final_cal"].iloc[1]  # far apart, still ordered


def test_binary_temperature_is_bounded_on_a_separable_val_set():
    """A val set its score separates drives the temperature MLE to T -> 0 (with small logits unbounded L-BFGS ends near
    1e-9); it stops at the 3-class floor 1/100."""
    from teb_vae.classifier.train import fit_calibration

    cal = fit_calibration([-2e-3, -1e-3, 1e-3, 2e-3], [0, 0, 1, 1], "temperature")
    assert cal["temperature"] == pytest.approx(0.01) and np.isfinite(cal["nll"])


@pytest.mark.slow
@pytest.mark.parametrize("loss", ["auc_margin", "pauc"])
def test_auc_losses_train_a_unit(env, loss):
    """A short fit per AUC loss (class-balanced batches, Platt calibration): the unit trains, the loss is finite and
    moves, and best.ckpt scores val."""
    from teb_vae.classifier.data import build_unit
    from teb_vae.classifier.train import score_split, train_unit

    cfg = _cfg(env, "classifier.model.scope=segment", f"classifier.train.loss.name={loss}",
               "classifier.train.sampler=class_balanced", "classifier.calibration.method=platt",
               "classifier.train.max_epochs=3")
    unit = build_unit(cfg, env[1], env[2]["source"], 1)
    try:
        record = train_unit(cfg, env[1], env[2], fold=1, seed=31 if loss == "pauc" else 30, kind="model", unit=unit)
    finally:
        _restore_loguru()
    assert record["status"] == "done", record.get("error")
    loss_curve = _history(record)["val/total_loss"].dropna()
    assert np.isfinite(loss_curve).all() and loss_curve.nunique() > 1
    _, gd = score_split(record["unit_dir"], unit, "val")
    assert np.isfinite(gd["score_final"]).all()
