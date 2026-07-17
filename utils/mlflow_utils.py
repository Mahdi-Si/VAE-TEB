"""Shared MLflow helpers.

Leaf module: it imports nothing first-party and never imports ``mlflow`` itself --
every entry point takes an already-constructed, run-bound logger and only calls
methods on it. That keeps it importable from any layer.

It lives here rather than in ``train/callbacks.py`` (its original home) because both
layers need it: the agnostic callbacks in :mod:`train.callbacks` and the SeqVAE
plotters in :mod:`utils.seqvae_plot_callbacks`. Housing it in ``train/`` forced
``utils/ -> train/``, the one import that ran against the repo's layering
(``utils/`` <- ``train/`` <- ``model/``, one way). See ``train/tests/test_layering.py``.
"""

from __future__ import annotations

from loguru import logger


def log_artifact_to_mlflow(mlflow_logger, path, trainer) -> None:
    """Upload a file artifact to the active MLflow run on rank 0 only.

    Central seam so every file-writing callback logs artifacts identically. It is a
    no-op when tracking is disabled (``mlflow_logger is None``) or on a non-zero rank
    (a duplicate cross-rank write would corrupt the run under DDP), and a tracking
    error is warned and swallowed so plotting can never kill training.

    Args:
        mlflow_logger: The run-bound ``MLFlowLogger`` (or ``None`` when disabled).
        path: Path of the file to upload.
        trainer: The Lightning trainer, used only for its ``is_global_zero`` guard.
    """
    if mlflow_logger is None or trainer is None or not trainer.is_global_zero:
        return
    try:
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow artifact upload failed for {}: {}", path, exc)


__all__ = ["log_artifact_to_mlflow"]
