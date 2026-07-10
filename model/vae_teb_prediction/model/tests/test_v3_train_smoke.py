r"""S4-T07: a handful of real Lightning training steps on a tiny synthetic loader.

Everything the wrapper tests exercise in isolation has to survive the actual
``Trainer.fit`` loop -- the automatic-optimization path with gradient clipping and
accumulation switched on, the metrics callbacks, and the checkpoint contract.

Two assertions carry weight:

* **Step-0 ``kld_raw`` is exactly 0.** Both posterior delta heads are zero-initialised, so
  :math:`q = p` and every later nat of :math:`K` is earned by source conditioning (G1). Note
  this holds even in *train* mode: dropout cannot perturb a head whose final layer is zero.
* **``accumulate_grad_batches > 1`` and ``gradient_clip_val > 0`` are accepted.** Lightning
  rejects both under manual optimization. That they run here is the practical proof that the
  fused permutation control (G6) kept the automatic path.

Monotonic loss decrease is deliberately *not* asserted: on four random batches it means
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from train.callbacks import MetricsLoggingCallback


@dataclass
class _Batch:
    """Stand-in for the HDF5 batch object, holding the fields the wrapper reads.

    A dataclass, not a plain class: Lightning's ``apply_to_collection`` walks dataclass fields
    to move the batch to the device and to infer the batch size for ``self.log``.
    """

    fhr_st: torch.Tensor
    fhr_ph: torch.Tensor
    up_st: torch.Tensor
    up_ph: torch.Tensor
    weight: torch.Tensor


class _TinyDataset(Dataset):
    def __init__(self, n: int, seq_len: int = 16):
        self.n, self.seq_len = n, seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        T = self.seq_len
        return {
            "fhr_st": torch.randn(T, 43, generator=g),
            "fhr_ph": torch.randn(T, 44, generator=g),
            "up_st": torch.randn(T, 43, generator=g),
            "up_ph": torch.randn(T, 58, generator=g),
            "weight": torch.ones(T),
        }


def _collate(items):
    return _Batch(**{k: torch.stack([it[k] for it in items]) for k in items[0]})


def _loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    return DataLoader(_TinyDataset(n), batch_size=batch_size, collate_fn=_collate)


class _RecordingCallback(pl.Callback):
    """Capture the metrics dict of the very first optimisation step."""

    def __init__(self) -> None:
        self.first_step_metrics = None
        self.train_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.train_batches += 1
        if self.first_step_metrics is None:
            self.first_step_metrics = {
                k: float(v) for k, v in trainer.callback_metrics.items()
            }


@pytest.fixture
def trainer_kwargs(tmp_path):
    return dict(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )


def test_fit_runs_with_clipping_and_accumulation(v3_pl, trainer_kwargs):
    """Both settings are rejected by Lightning under manual optimization."""
    pl_module = v3_pl()
    assert pl_module.automatic_optimization is True

    recorder = _RecordingCallback()
    trainer = pl.Trainer(
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        callbacks=[recorder],
        **trainer_kwargs,
    )
    trainer.fit(pl_module, _loader(), _loader(n=4))

    assert recorder.train_batches == 2
    assert torch.isfinite(torch.tensor(trainer.callback_metrics["train/total_loss"].item()))


def test_step_zero_kld_raw_is_zero(v3_pl, trainer_kwargs):
    r"""Zero-KL at initialisation (G1) must survive the real training loop."""
    pl_module = v3_pl()
    recorder = _RecordingCallback()
    trainer = pl.Trainer(callbacks=[recorder], **trainer_kwargs)
    trainer.fit(pl_module, _loader(), _loader(n=4))

    metrics = recorder.first_step_metrics
    assert metrics is not None
    assert abs(metrics["train/kld_raw_step"]) < 1e-6, (
        f"step-0 kld_raw = {metrics['train/kld_raw_step']}, expected ~0"
    )


def test_all_new_metrics_reach_the_logger(v3_pl, trainer_kwargs):
    """G4/G6/G7 diagnostics must be logged, or the training run is unreadable."""
    pl_module = v3_pl()
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(pl_module, _loader(), _loader(n=4))

    logged = set(trainer.callback_metrics)
    for suffix in ("kld_raw", "kld_train", "kld_active_frac", "perm_loss",
                   "kld_shuffled", "kld_shuffled_ratio", "mean_logvar_full",
                   "mean_logvar_base", "main_loss"):
        assert any(k.startswith(f"train/{suffix}") for k in logged), (
            f"train/{suffix} never reached the logger; logged={sorted(logged)}"
        )
        assert f"val/{suffix}" in logged, f"val/{suffix} never reached the logger"


def test_metrics_history_callback_writes_a_csv(v3_pl, trainer_kwargs, tmp_path):
    """v1's MetricsLoggingCallback accumulates history but writes nothing; v3 dumps it."""
    from model.vae_teb_prediction.model.trainer_lag_attn_v3 import (
        _TRACKED_METRICS,
        MetricsHistoryCsvCallback,
    )

    metrics_cb = MetricsLoggingCallback(tracked_metrics=_TRACKED_METRICS)
    csv_cb = MetricsHistoryCsvCallback(source=metrics_cb, output_dir=str(tmp_path))
    trainer = pl.Trainer(callbacks=[metrics_cb, csv_cb], **trainer_kwargs)
    trainer.fit(v3_pl(), _loader(), _loader(n=4))

    path = tmp_path / "metrics_history.csv"
    assert path.is_file(), "metrics_history.csv was not written"
    header = path.read_text(encoding="utf-8").splitlines()[0]
    for column in ("val/kld_raw", "val/kld_shuffled", "val/mean_logvar_full", "lr"):
        assert column in header, f"{column} missing from the history CSV header"


def test_checkpoint_written_by_lightning_carries_the_contract(v3_pl, trainer_kwargs, tmp_path):
    from lightning.pytorch.callbacks import ModelCheckpoint

    from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import check_model_class

    kwargs = dict(trainer_kwargs)
    kwargs["enable_checkpointing"] = True
    ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_top_k=1,
                              monitor="val/total_loss", filename="v3-{epoch:02d}")
    trainer = pl.Trainer(callbacks=[ckpt_cb], **kwargs)
    trainer.fit(v3_pl(), _loader(), _loader(n=4))

    saved = list((tmp_path / "ckpt").glob("*.ckpt"))
    assert saved, "no checkpoint written"
    blob = torch.load(saved[0], map_location="cpu", weights_only=False)
    check_model_class(blob, "SeqVaeLagAttnV3")
    assert blob["model_kwargs"]["causal_norm"] is True


def test_losses_stay_finite_across_steps(v3_pl, trainer_kwargs):
    pl_module = v3_pl()
    kwargs = dict(trainer_kwargs, limit_train_batches=2, max_epochs=2)
    trainer = pl.Trainer(gradient_clip_val=0.5, **kwargs)
    trainer.fit(pl_module, _loader(), _loader(n=4))

    for name, value in trainer.callback_metrics.items():
        assert torch.isfinite(value), f"{name} went non-finite: {value}"
    assert pl_module._spike_skips_total == 0, "the circuit breaker fired on a healthy run"
