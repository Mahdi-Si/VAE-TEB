"""Backward-compatibility regression guard.

Proves a faithful unmigrated-style consumer (grandparent ``__init__`` bypass,
overridden ``training_step``, ``on_save_checkpoint`` setting ``model_class``) still
trains one step and saves a checkpoint against the changed base classes.
"""
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from train.test_utils import StandInConsumer


def _loader(n: int = 8):
    x = torch.randn(n, 4)
    y = torch.randn(n, 4)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_standin_trains_one_step_and_checkpoints(tmp_path):
    model = StandInConsumer()
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, _loader())

    ckpt_path = tmp_path / "standin.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    assert ckpt_path.exists()

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert checkpoint["model_class"] == "TinyModule"
