"""Uniform rank-0 MLflow artifact seam (``log_artifact_to_mlflow``).

The seam is a no-op when tracking is disabled or off the zero rank, uploads exactly once
through the run-bound client on rank 0, and swallows tracking errors so plotting can
never kill training.
"""
from utils.mlflow_utils import log_artifact_to_mlflow
from train.test_utils import FakeMLflowLogger, FakeTrainer


def test_noop_when_logger_none(tmp_path):
    # No logger -> nothing happens, no exception.
    log_artifact_to_mlflow(None, tmp_path / "x.html", FakeTrainer(is_global_zero=True))


def test_noop_on_non_zero_rank(tmp_path):
    logger = FakeMLflowLogger()
    log_artifact_to_mlflow(logger, tmp_path / "x.html", FakeTrainer(is_global_zero=False))
    assert logger.experiment.calls == []  # never uploads off rank 0


def test_uploads_once_on_rank_zero(tmp_path):
    logger = FakeMLflowLogger(run_id="run-7")
    path = tmp_path / "plot.html"
    log_artifact_to_mlflow(logger, path, FakeTrainer(is_global_zero=True))
    assert logger.experiment.calls == [("log_artifact", "run-7", (str(path),))]


class _RaisingExperiment:
    def log_artifact(self, *args, **kwargs):
        raise RuntimeError("tracking server down")


class _RaisingLogger:
    run_id = "run-0"
    experiment = _RaisingExperiment()


def test_upload_error_is_swallowed(tmp_path):
    # A tracking error is warned and swallowed, never propagated.
    log_artifact_to_mlflow(_RaisingLogger(), tmp_path / "x.html", FakeTrainer(is_global_zero=True))
