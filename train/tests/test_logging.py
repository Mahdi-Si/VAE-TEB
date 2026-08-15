"""Run-logging behaviour: rank isolation, JSONL mirror, PHI safety, MLflow upload.

Loguru's configuration is process-global, so every test here restores it on the way
out (which also closes the file handles Windows would otherwise refuse to unlink).
"""
import json
import logging as std_logging
import os
from pathlib import Path

import pytest
from loguru import logger

from train.graph_model_base import _RUN_STAMP_ENV, _resolve_run_stamp
from train.test_utils import FakeMLflowLogger, make_graph_model
from utils.custom_logger import (
    LoggingPaths,
    rank_suffixed_path,
    resolve_global_rank,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_loguru():
    """Drop every sink after each test so file handles close and state cannot leak."""
    yield
    logger.remove()


def _read(path) -> str:
    return Path(path).read_text(encoding="utf-8")


# --- rank resolution + per-rank file isolation ------------------------------

def test_resolve_global_rank_defaults_to_zero(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        monkeypatch.delenv(key, raising=False)
    assert resolve_global_rank() == 0


def test_resolve_global_rank_reads_local_rank(monkeypatch):
    # Lightning's own ddp launcher sets LOCAL_RANK and never sets RANK.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert resolve_global_rank() == 3


def test_resolve_global_rank_prefers_rank_over_local_rank(monkeypatch):
    # torchrun/SLURM set RANK; it is the global one, so it must win on multi-node.
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert resolve_global_rank() == 9


def test_resolve_global_rank_survives_malformed_value(monkeypatch):
    monkeypatch.setenv("RANK", "not-an-int")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("JSM_NAMESPACE_RANK", raising=False)
    assert resolve_global_rank() == 0


def test_rank_zero_keeps_plain_filename():
    assert rank_suffixed_path("/runs/full.log", 0) == "/runs/full.log"


def test_nonzero_rank_is_suffixed():
    assert rank_suffixed_path("/runs/full.log", 2) == "/runs/full.log.rank2"


def test_each_rank_writes_its_own_file(tmp_path):
    """The core DDP guard: two ranks must never share one file handle."""
    base = tmp_path / "full.log"
    paths_0 = setup_logging(file_path=str(base), log_to_console=False, rank=0)
    logger.info("from-rank-zero")
    logger.remove()

    paths_2 = setup_logging(file_path=str(base), log_to_console=False, rank=2)
    logger.info("from-rank-two")

    assert paths_0.text_log != paths_2.text_log
    assert "from-rank-zero" in _read(paths_0.text_log)
    assert "from-rank-two" not in _read(paths_0.text_log)
    assert "from-rank-two" in _read(paths_2.text_log)


def test_console_is_rank_zero_only(tmp_path, capsys):
    setup_logging(file_path=str(tmp_path / "full.log"), rank=1)
    logger.info("should-not-reach-console")
    assert "should-not-reach-console" not in capsys.readouterr().err


def test_console_present_on_rank_zero(tmp_path, capsys):
    setup_logging(file_path=str(tmp_path / "full.log"), rank=0)
    logger.info("hello-console")
    assert "hello-console" in capsys.readouterr().err


def test_rank_tagged_in_text_sink(tmp_path):
    paths = setup_logging(file_path=str(tmp_path / "full.log"), log_to_console=False, rank=5)
    logger.info("tagged")
    assert "rank5" in _read(paths.text_log)


# --- PHI safety: diagnose must never render locals to disk -------------------

def _raise_with_secret():
    """Fail on a line that *uses* the local, which is what diagnose renders."""
    patient_signal = "SECRET_PHI_VALUE"
    return patient_signal[9999]


def test_diagnose_off_keeps_locals_out_of_the_log_file(tmp_path):
    paths = setup_logging(
        file_path=str(tmp_path / "full.log"), log_to_console=False, rank=0, diagnose=False,
    )
    try:
        _raise_with_secret()
    except IndexError:
        logger.exception("step failed")

    text = _read(paths.text_log)
    assert "step failed" in text          # the traceback is still recorded ...
    assert "IndexError" in text
    assert "SECRET_PHI_VALUE" not in text  # ... without the variable's value


def test_diagnose_on_would_leak_locals(tmp_path):
    """Positive control: proves the assertion above is actually load-bearing."""
    paths = setup_logging(
        file_path=str(tmp_path / "full.log"), log_to_console=False, rank=0, diagnose=True,
    )
    try:
        _raise_with_secret()
    except IndexError:
        logger.exception("step failed")

    assert "SECRET_PHI_VALUE" in _read(paths.text_log)


def test_setup_config_forces_diagnose_off_on_disk(config_path, tmp_path, monkeypatch):
    """The shipped path must not leak locals even if a config asks it to."""
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return LoggingPaths(text_log=None, json_log=None, rank=0)

    monkeypatch.setattr("train.graph_model_base.setup_logging", _spy)
    gm = make_graph_model(
        config_path,
        **{
            "general_config.folders_config.out_dir_base": str(tmp_path),
            "advanced_config.logging.console_diagnose": True,
        },
    )
    gm.setup_config()

    assert captured["diagnose"] is False        # file sinks: never
    assert captured["console_diagnose"] is True  # console: caller's choice


# --- JSON Lines mirror ------------------------------------------------------

def test_jsonl_sink_emits_one_parseable_object_per_line(tmp_path):
    paths = setup_logging(
        file_path=str(tmp_path / "full.log"),
        json_path=str(tmp_path / "run.jsonl"),
        log_to_console=False,
        rank=0,
    )
    logger.info("first")
    logger.warning("second")

    lines = [ln for ln in _read(paths.json_log).splitlines() if ln.strip()]
    assert len(lines) == 2
    records = [json.loads(ln) for ln in lines]
    assert [r["record"]["message"] for r in records] == ["first", "second"]
    assert [r["record"]["level"]["name"] for r in records] == ["INFO", "WARNING"]


def test_jsonl_carries_rank_in_extra(tmp_path):
    paths = setup_logging(
        file_path=str(tmp_path / "full.log"),
        json_path=str(tmp_path / "run.jsonl"),
        log_to_console=False,
        rank=4,
    )
    logger.info("ranked")
    record = json.loads(_read(paths.json_log).splitlines()[0])
    assert record["record"]["extra"]["rank"] == 4


def test_jsonl_is_rank_suffixed_too(tmp_path):
    paths = setup_logging(
        file_path=str(tmp_path / "full.log"),
        json_path=str(tmp_path / "run.jsonl"),
        log_to_console=False,
        rank=1,
    )
    assert paths.json_log is not None
    assert paths.json_log.endswith("run.jsonl.rank1")


def test_json_log_disabled_by_default(tmp_path):
    paths = setup_logging(file_path=str(tmp_path / "full.log"), log_to_console=False, rank=0)
    assert paths.json_log is None


# --- interception + idempotency ---------------------------------------------

def test_stdlib_logging_is_intercepted(tmp_path):
    """Lightning/MLflow log through stdlib; those records must reach our sinks."""
    paths = setup_logging(file_path=str(tmp_path / "full.log"), log_to_console=False, rank=0)
    std_logging.getLogger("lightning.pytorch").warning("from-stdlib")
    assert "from-stdlib" in _read(paths.text_log)


def test_warnings_are_routed_into_logging(tmp_path):
    """rank_zero_warn goes through warnings.warn, not logging, so it needs capturing.

    Asserted as two links rather than by calling ``warnings.warn`` end-to-end:
    pytest's own warnings plugin wraps each test in ``catch_warnings(record=True)``
    and installs its recorder as ``showwarning``, so an end-to-end call here would
    measure pytest, not this code. (Verified manually outside pytest: a real
    ``warnings.warn`` does reach ``full.log``.)
    """
    import warnings

    paths = setup_logging(file_path=str(tmp_path / "full.log"), log_to_console=False, rank=0)

    # Link 1: captureWarnings(True) installed the stdlib bridge.
    assert warnings.showwarning.__name__ == "_showwarning"
    assert warnings.showwarning.__module__ == "logging"

    # Link 2: the logger that bridge writes to reaches our sink.
    std_logging.getLogger("py.warnings").warning("deprecated-thing")
    assert "deprecated-thing" in _read(paths.text_log)


def test_lightning_logger_is_reclaimed_from_its_import_time_handler(tmp_path):
    """Regression guard for the subtlest failure here.

    ``lightning/pytorch/__init__.py`` does, at import: if root has no handlers, add
    a private StreamHandler and set ``propagate = False``. Imports run long before
    the run configures logging, so that branch is normally taken and every Trainer
    message ("GPU available", "LOCAL_RANK: ...") bypasses root — and therefore
    bypasses the InterceptHandler and every sink. It previously worked only by
    accident, because some model modules called setup_logging at import time and
    happened to give root a handler first.
    """
    pl_logger = std_logging.getLogger("lightning.pytorch")
    pl_logger.handlers = [std_logging.StreamHandler()]  # reproduce the import-time grab
    pl_logger.propagate = False

    paths = setup_logging(file_path=str(tmp_path / "full.log"), log_to_console=False, rank=0)

    assert pl_logger.propagate is True
    assert pl_logger.handlers == []  # its private handler would also double-print
    pl_logger.info("GPU available: True (cuda), used: True")
    assert "GPU available" in _read(paths.text_log)


def test_repeated_setup_does_not_duplicate_records(tmp_path):
    """logger.configure() replaces handlers; a second call must not double every line."""
    target = str(tmp_path / "full.log")
    setup_logging(file_path=target, log_to_console=False, rank=0)
    paths = setup_logging(file_path=target, log_to_console=False, rank=0)
    logger.info("only-once")
    assert _read(paths.text_log).count("only-once") == 1


# --- run-stamp handshake across DDP ranks -----------------------------------

def test_parent_computes_fresh_stamp_and_exports_it(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv(_RUN_STAMP_ENV, "STALE-STAMP")
    stamp = _resolve_run_stamp()
    assert stamp != "STALE-STAMP"
    assert os.environ[_RUN_STAMP_ENV] == stamp  # exported for the children


def test_ddp_child_inherits_the_parent_stamp(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv(_RUN_STAMP_ENV, "PARENT-STAMP")
    assert _resolve_run_stamp() == "PARENT-STAMP"


def test_parent_after_fit_does_not_reuse_a_stale_stamp(monkeypatch):
    """Lightning sets LOCAL_RANK=0 in the parent once fit starts; a second run in
    that process must still get its own directory."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv(_RUN_STAMP_ENV, "FIRST-RUN-STAMP")
    assert _resolve_run_stamp() != "FIRST-RUN-STAMP"


def test_ranks_agree_on_the_run_directory(config_path, tmp_path, monkeypatch):
    """The end-to-end property: a child must not fragment the run into its own dir."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv(_RUN_STAMP_ENV, raising=False)
    parent = make_graph_model(
        config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    # Simulate the re-executed subprocess: same env, LOCAL_RANK set by the launcher.
    monkeypatch.setenv("LOCAL_RANK", "1")
    child = make_graph_model(
        config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )
    assert child.base_folder == parent.base_folder


# --- MLflow run-log upload --------------------------------------------------

def test_upload_run_logs_sends_both_files(config_path, tmp_path):
    gm = make_graph_model(
        config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )
    text = tmp_path / "full.log"
    js = tmp_path / "run.jsonl"
    text.write_text("log", encoding="utf-8")
    js.write_text("{}", encoding="utf-8")
    gm._log_paths = LoggingPaths(text_log=str(text), json_log=str(js), rank=0)
    gm.mlflow_logger = FakeMLflowLogger()

    gm.upload_run_logs()

    uploaded = [c[2][0] for c in gm.mlflow_logger.experiment.calls if c[0] == "log_artifact"]
    assert sorted(uploaded) == sorted([str(text), str(js)])


def test_upload_run_logs_skips_missing_files(config_path, tmp_path):
    gm = make_graph_model(
        config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )
    gm._log_paths = LoggingPaths(
        text_log=str(tmp_path / "absent.log"), json_log=None, rank=0,
    )
    gm.mlflow_logger = FakeMLflowLogger()
    gm.upload_run_logs()
    assert gm.mlflow_logger.experiment.calls == []


def test_upload_run_logs_is_noop_without_tracking(config_path, tmp_path):
    gm = make_graph_model(
        config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )
    gm.mlflow_logger = None
    gm.upload_run_logs()  # must not raise


def test_upload_run_logs_fails_closed(config_path, tmp_path):
    """A tracking outage must never take down a finished training run."""
    gm = make_graph_model(
        config_path, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )
    text = tmp_path / "full.log"
    text.write_text("log", encoding="utf-8")
    gm._log_paths = LoggingPaths(text_log=str(text), json_log=None, rank=0)

    class _Boom(FakeMLflowLogger):
        def __init__(self):
            super().__init__()
            self.experiment.log_artifact = self._raise

        @staticmethod
        def _raise(*_args, **_kwargs):
            raise RuntimeError("tracking server down")

    gm.mlflow_logger = _Boom()
    gm.upload_run_logs()  # warns, does not raise
