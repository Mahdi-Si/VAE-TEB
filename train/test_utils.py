"""Shared test helpers for the ``train/`` framework test suite.

Imported by the tests under ``train/tests/``. These subclass the real base classes
(so ``torch``/``lightning`` are required) but are otherwise lightweight and never
touch the filesystem, a GPU, HDF5, or an MLflow server.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch
from torch import nn

from train.pl_model_base import LightningModelBase


class TinyModule(nn.Module):
    """A minimal 2-layer MLP standing in for a real model in framework tests."""

    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TinyLightningModel(LightningModelBase):
    """Concrete :class:`LightningModelBase` subclass with a trivial MSE loss.

    Defaults to ``compile_model=False`` so instantiation is safe on any platform
    (no ``torch.compile`` backend involved); pass ``compile_model=True`` to exercise
    the compiled path.
    """

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        *,
        compile_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(base_model or TinyModule(), compile_model=compile_model, **kwargs)

    def compute_loss_and_metrics(self, batch, batch_idx, stage):
        x, y = batch
        pred = self.model(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss, {"total_loss": loss}


class StandInConsumer(L.LightningModule):
    """Faithful stand-in for the real consumers that bypass ``LightningModelBase``.

    Mirrors the patterns the migration guide will later remove: a grandparent
    ``LightningModule.__init__`` bypass (so no ``torch.compile``), an overridden
    ``training_step``, and an ``on_save_checkpoint`` that sets ``model_class``. Used
    by the backward-compatibility regression guard to prove such a consumer still
    trains one step and checkpoints against the changed base.
    """

    def __init__(self, base_model: Optional[nn.Module] = None) -> None:
        # Grandparent init on purpose — skips LightningModelBase.__init__ entirely.
        L.LightningModule.__init__(self)
        self._orig_model = base_model or TinyModule()
        self.model = self._orig_model  # eager, no compile

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.model(x), y)
        self.log("train/total_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["model_class"] = type(self._orig_model).__name__


class FakeMLflowExperiment:
    """Records the run-bound client-API calls for MLflow-seam assertions."""

    def __init__(self) -> None:
        self.calls = []  # list of (method, run_id, payload)

    def log_param(self, run_id, key, value):
        self.calls.append(("log_param", run_id, (key, value)))

    def log_metric(self, run_id, key, value, *args, **kwargs):
        self.calls.append(("log_metric", run_id, (key, value)))

    def log_text(self, run_id, text, artifact_file):
        self.calls.append(("log_text", run_id, (artifact_file,)))

    def set_tag(self, run_id, key, value):
        self.calls.append(("set_tag", run_id, (key, value)))

    def log_artifact(self, run_id, local_path, *args, **kwargs):
        self.calls.append(("log_artifact", run_id, (str(local_path),)))


class FakeMLflowLogger:
    """Minimal ``MLFlowLogger`` stand-in exposing ``.experiment`` and ``.run_id``.

    Also records ``log_hyperparams`` calls (the batched, rank-safe logger API) in
    ``logged_hyperparams`` so the run-metadata logging can be asserted on.
    """

    def __init__(self, run_id: str = "run-0") -> None:
        self.run_id = run_id
        self.experiment = FakeMLflowExperiment()
        self.logged_hyperparams: dict = {}

    def log_hyperparams(self, params) -> None:
        self.logged_hyperparams.update(dict(params))


class FakeStrategy:
    """Stand-in trainer strategy holding a settable ``world_size``.

    When ``other_value`` is set, :meth:`reduce` records the ``reduce_op`` it is called
    with and applies it element-wise against that injected "other rank" value, so the
    DDP MAX/MIN skip-sync can be unit-tested without a real process group.
    """

    def __init__(self, world_size: int = 1, other_value: Optional[float] = None) -> None:
        self.world_size = world_size
        self.other_value = other_value
        self.reduce_calls = []  # list of reduce_op values passed to reduce()

    def reduce(self, tensor, group=None, reduce_op=None):
        import torch

        self.reduce_calls.append(reduce_op)
        other_value = 0.0 if self.other_value is None else float(self.other_value)
        other = torch.tensor(other_value, device=tensor.device)
        name = getattr(reduce_op, "name", str(reduce_op)).upper()
        if "MIN" in name:
            return torch.minimum(tensor, other)
        return torch.maximum(tensor, other)


class FakeTrainer:
    """Settable stand-in for ``pl.Trainer`` used by callback/hook unit tests."""

    def __init__(
        self,
        *,
        is_global_zero: bool = True,
        current_epoch: int = 0,
        world_size: int = 1,
        sanity_checking: bool = False,
        callback_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.is_global_zero = is_global_zero
        self.current_epoch = current_epoch
        self.strategy = FakeStrategy(world_size=world_size)
        # Callbacks that skip the sanity-check pass read this; the default is the steady state.
        self.sanity_checking = sanity_checking
        self.callback_metrics: Dict[str, Any] = dict(callback_metrics or {})


def make_graph_model(config_path, **config_overrides):
    """Instantiate a concrete :class:`GraphModelBase` from ``config_path``.

    Returns a minimal concrete subclass (stub abstract methods) so tests can call
    ``validate_config`` / ``configure_determinism`` / ``_build_trainer_kwargs``
    without a real training run. ``config_overrides`` are dotted-path assignments
    applied to ``self.config`` after load (e.g.
    ``{"advanced_config.trainer.precision": 16}``); the handful of general-config
    attributes cached in ``__init__`` are re-synced so overrides to them take effect.
    """
    from train.graph_model_base import GraphModelBase

    class _ConcreteGraphModel(GraphModelBase):
        def create_model(self):
            raise NotImplementedError

        def train_model(self, train_loader, validation_loader):
            raise NotImplementedError

    gm = _ConcreteGraphModel(config_file_path=str(config_path))

    for dotted, value in config_overrides.items():
        node = gm.config
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    # Re-sync the general-config attributes __init__ cached, so overrides land.
    general = gm.config.get("general_config", {})
    gm.cuda_devices = general.get("cuda_devices", gm.cuda_devices)
    gm.epochs_num = general.get("epochs", gm.epochs_num)
    gm.accumulate_grad_batches = general.get("accumulate_grad_batches", gm.accumulate_grad_batches)

    # Re-derive the output directories too, so a folders_config.out_dir_base override
    # (e.g. a tmp_path in the smoke test) actually redirects where setup_config writes.
    import os as _os

    folders = general.get("folders_config", {})
    if "out_dir_base" in folders:
        gm.output_base_dir = _os.path.normpath(folders["out_dir_base"])
        run_dir = _os.path.join(gm.output_base_dir, gm.base_folder)
        gm.train_results_dir = _os.path.join(run_dir, "train_results")
        gm.test_results_dir = _os.path.join(run_dir, "test_results")
        gm.model_checkpoint_dir = _os.path.join(run_dir, "model_checkpoints")
        gm.aux_dir = _os.path.join(run_dir, "aux_tests")
        gm.tensorboard_dir = _os.path.join(run_dir, "tensorboard_log")
    return gm
