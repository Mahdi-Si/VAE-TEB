"""MLflow system-of-record logging: config/hparams/provenance, system metrics,
checkpoint decoupling, and the architecture/final-model run-logging callback.

All writes go through the run-bound client fake, so nothing here needs a live MLflow
server. DDP rank behaviour is checked with the ``is_global_zero`` flag on ``FakeTrainer``.
"""
import contextlib

import mlflow
import mlflow.pytorch
import mlflow.system_metrics.system_metrics_monitor as smm

from train.callbacks import MLflowRunLoggingCallback
from train.graph_model_base import GraphModelBase
from train.test_utils import FakeMLflowLogger, FakeTrainer, TinyLightningModel, make_graph_model


# --- S5-T02: flatten helper, provenance, run-metadata --------------------------

def test_flatten_config_dotted_and_json_encoded():
    node = {"a": 1, "b": {"c": "x", "d": [1, 2, 3]}, "e": True}
    flat = GraphModelBase._flatten_config(node, "root")
    assert flat["root.a"] == "1"
    assert flat["root.b.c"] == "x"
    assert flat["root.b.d"] == "[1, 2, 3]"  # non-scalar leaf JSON-encoded
    assert flat["root.e"] == "True"


def test_flatten_config_truncates_long_value():
    flat = GraphModelBase._flatten_config({"k": "y" * 1000}, "p", max_len=500)
    assert len(flat["p.k"]) == 500


def test_provenance_tags_omit_git_when_unavailable(config_path, monkeypatch):
    import subprocess

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "check_output", no_git)
    gm = make_graph_model(config_path)
    tags = gm._collect_provenance_tags()
    assert "git_commit" not in tags  # git absent -> omitted, no raise
    assert "host" in tags and "world_size" in tags and "precision" in tags
    assert tags["torch_version"]


def test_run_metadata_logs_config_params_and_tags(config_path):
    gm = make_graph_model(config_path)
    gm.mlflow_logger = FakeMLflowLogger(run_id="r1")
    gm._log_run_metadata_to_mlflow()
    calls = gm.mlflow_logger.experiment.calls

    text_calls = [c for c in calls if c[0] == "log_text"]
    assert any(c[2][0] == "config/resolved_config.yaml" for c in text_calls)

    # Hyperparameters are batched through log_hyperparams (more than the old 6-param set).
    hparams = gm.mlflow_logger.logged_hyperparams
    assert len(hparams) > 6
    assert any(k.startswith("general_config.") for k in hparams)

    tag_keys = {c[2][0] for c in calls if c[0] == "set_tag"}
    assert "host" in tag_keys and "precision" in tag_keys

    assert all(c[1] == "r1" for c in calls)  # every client write bound to the run id


def test_run_metadata_noop_without_logger(config_path):
    gm = make_graph_model(config_path)
    gm.mlflow_logger = None
    gm._log_run_metadata_to_mlflow()  # must not raise


# --- S5-T03: system metrics + checkpoint decouple ------------------------------

def test_system_metrics_monitor_started_with_run_id(config_path, monkeypatch):
    created = {"run_id": None, "starts": 0}

    class FakeMonitor:
        def __init__(self, run_id=None, **kwargs):
            created["run_id"] = run_id

        def start(self):
            created["starts"] += 1

    monkeypatch.setattr(smm, "SystemMetricsMonitor", FakeMonitor)
    gm = make_graph_model(config_path)
    gm.mlflow_logger = FakeMLflowLogger(run_id="rr")
    gm._start_system_metrics_monitor()
    assert created["run_id"] == "rr"
    assert created["starts"] == 1


def test_log_checkpoints_all_threaded_without_bool(config_path, tmp_path, monkeypatch):
    import train.graph_model_base as gmb

    captured = {}

    class _Exp:
        def set_tag(self, *args, **kwargs):
            pass

    class FakeLogger:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.run_id = "run-x"
            self.experiment = _Exp()

        def log_hyperparams(self, params):
            pass

    monkeypatch.setattr(gmb, "MLFlowLogger", FakeLogger)
    # A local file store keeps _init_mlflow_logger's soft-delete pre-check offline
    # (the shipped http URI would block on connection retries).
    file_uri = (tmp_path / "mlruns").as_uri()
    gm = make_graph_model(
        config_path,
        **{
            "advanced_config.tracking.mlflow.enabled": True,
            "advanced_config.tracking.mlflow.tracking_uri": file_uri,
            "advanced_config.tracking.mlflow.log_checkpoints": "all",
        },
    )
    gm._init_mlflow_logger()
    # "all" survives (no bool() collapse) into the Lightning logger constructor.
    assert captured["log_model"] == "all"


# --- S5-T04: architecture + final-model run-logging callback -------------------

def test_callback_on_fit_start_logs_architecture_and_counts():
    logger = FakeMLflowLogger(run_id="rid")
    cb = MLflowRunLoggingCallback(logger, experiment_tag="tagX", log_model=True)
    model = TinyLightningModel(compile_model=False)
    cb.on_fit_start(FakeTrainer(is_global_zero=True), model)

    calls = logger.experiment.calls
    assert any(c[0] == "log_text" and c[2][0] == "model/model_architecture.txt" for c in calls)
    assert "model_class" in {c[2][0] for c in calls if c[0] == "log_param"}
    metric_keys = {c[2][0] for c in calls if c[0] == "log_metric"}
    assert {"params_total", "params_trainable"} <= metric_keys


def test_callback_on_fit_start_noop_off_rank_zero():
    logger = FakeMLflowLogger()
    cb = MLflowRunLoggingCallback(logger, "tag", log_model=True)
    cb.on_fit_start(FakeTrainer(is_global_zero=False), TinyLightningModel(compile_model=False))
    assert logger.experiment.calls == []


def test_callback_on_fit_end_logs_orig_model(monkeypatch):
    captured = {}

    @contextlib.contextmanager
    def fake_start_run(run_id=None):
        captured["run_id"] = run_id
        yield

    def fake_log_model(model, name=None, registered_model_name=None, **kwargs):
        captured["model"] = model
        captured["name"] = name
        captured["registered"] = registered_model_name

    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow.pytorch, "log_model", fake_log_model)

    logger = FakeMLflowLogger(run_id="R")
    cb = MLflowRunLoggingCallback(logger, "myTag", log_model=True)
    model = TinyLightningModel(compile_model=False)
    cb.on_fit_end(FakeTrainer(is_global_zero=True), model)

    assert captured["run_id"] == "R"
    assert captured["model"] is model.orig_model
    assert captured["name"] == "model"
    assert captured["registered"] == "myTag"
    # The eager module carries no torch.compile prefix, so it reloads cleanly.
    assert not any(k.startswith("_orig_mod.") for k in model.orig_model.state_dict())


def test_callback_on_fit_end_skips_when_log_model_false(monkeypatch):
    called = {"n": 0}

    def fake_log_model(*args, **kwargs):
        called["n"] += 1

    monkeypatch.setattr(mlflow.pytorch, "log_model", fake_log_model)
    cb = MLflowRunLoggingCallback(FakeMLflowLogger(), "tag", log_model=False)
    cb.on_fit_end(FakeTrainer(is_global_zero=True), TinyLightningModel(compile_model=False))
    assert called["n"] == 0


def test_callback_on_fit_end_error_is_swallowed(monkeypatch):
    @contextlib.contextmanager
    def fake_start_run(run_id=None):
        yield

    def boom(*args, **kwargs):
        raise RuntimeError("registry down")

    monkeypatch.setattr(mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(mlflow.pytorch, "log_model", boom)
    cb = MLflowRunLoggingCallback(FakeMLflowLogger(), "t", log_model=True)
    cb.on_fit_end(FakeTrainer(is_global_zero=True), TinyLightningModel(compile_model=False))  # no raise


def test_build_trainer_attaches_callback_when_tracking_enabled(config_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gm = make_graph_model(config_path)
    gm.mlflow_logger = FakeMLflowLogger()
    kw = gm._build_trainer_kwargs([])
    assert any(isinstance(cb, MLflowRunLoggingCallback) for cb in kw["callbacks"])


def test_build_trainer_no_callback_when_tracking_disabled(config_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gm = make_graph_model(config_path)  # mlflow_logger stays None
    kw = gm._build_trainer_kwargs([])
    assert not any(isinstance(cb, MLflowRunLoggingCallback) for cb in kw["callbacks"])
