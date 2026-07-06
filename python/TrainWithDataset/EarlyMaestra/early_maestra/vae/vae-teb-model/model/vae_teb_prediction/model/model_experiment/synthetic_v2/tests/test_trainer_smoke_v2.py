r"""Sprint 6 (S6-T02): production HDF5 trainer drop-in on v2 (CPU smoke).

The assistant gate for the production pipeline. With :class:`SeqVaeLagAttnV2`
wrapped by the production Lightning module
(:class:`trainer_lag_attn_v1.SeqVaeLagAttnPl`), this exercises -- on CPU, with a
tiny v2 model and random HDF5-shaped tensors -- the three drop-in surfaces the
full ``GraphModelVaeTebLagAttnV1Trainer`` orchestration depends on:

* ``compute_loss_and_metrics`` returns finite, v1-keyed metrics (the shared
  metrics builder indexes only keys v2 also emits, so nothing leaks that would
  ``KeyError`` under the v1 alias).
* A short ``Trainer.fit`` runs N train + val steps and logs ``total_loss``.
* :class:`LagAttnV1PlotCallback` renders a diagnostic figure from the v2 forward
  dict.
* A v2 checkpoint passes ``check_model_class`` and reloads strict, while the same
  blob is rejected under the v1 alias name (the drop-in class guard).

The full multi-GPU DDP run on real HDF5 datasets on the A6000 box is operator-run
(not automated here). See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` S6-T02.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    dataset_v2 as ds2,
)
from model.vae_teb_prediction.model.plotting_callback_lag_attn_v1 import (  # noqa: E402
    LagAttnV1PlotCallback,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    SeqVaeLagAttnV2,
    check_model_class,
)

_T = 32
_C = (43, 44, 43, 58)   # fhr_st, fhr_ph, up_st, up_ph

# Tiny v2 model: production channel counts, everything else shrunk. entmax on so
# the sparse lag posterior path runs. num_heads * d_head == d_model.
_TINY_V2_KW: Dict[str, Any] = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": 4,
    "warmup_period": 2,
    "c_y": 87,
    "c_u": 101,
    "use_up_st": True,
    "max_lag": 8,
    "num_heads": 2,
    "d_head": 8,
    "dropout": 0.0,
    "decoder_hidden": 16,
    "logvar_clamp": (-5.0, 3.0),
    "mu_scale": 5.0,
    "delta_mu_scale": 3.0,
    "latent_stats_momentum": 0.01,
    "use_entmax": True,
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
    "target_encoder_blocks": 2,
    "target_kernel": 3,
    "target_dilations": (1, 2),
    "source_scales": (3, 5),
    "d_u": 16,
    "d_k": 8,
    "d_e": 8,
    "active_lags": 4,
    "active_lags_warmup": 6,
    "kappa_z": 0.05,
    "lambda_tv": 1.0e-4,
    "lambda_ent": 0.0,
    "context_dim": 5,
    "step_seconds": 4.0,
    "delta_up_seconds": 20.0,
}

# The v1 keys the production trainer's metrics builder reads (all emitted by v2).
_EXPECTED_METRIC_KEYS = {
    "total_loss", "feat_loss", "base_loss", "kld_loss", "kld_beta",
    "lambda_full", "lambda_base", "mu_prior_sat_frac", "delta_mu_sat_frac",
    "delta_mu_rms", "mu_post_prior_gap_rms", "pred_gap", "lag_smoothness",
}


def _model() -> SeqVaeLagAttnV2:
    return SeqVaeLagAttnV2(**_TINY_V2_KW)


def _batch(n: int = 2, *, seed: int = 0, with_guid: bool = True) -> ds2.AttributeDict:
    r"""A random HDF5-shaped batch (attribute access; no raw fhr/up trace).

    An :class:`AttributeDict` (dict subclass) so Lightning can infer the batch
    size for logging while ``batch.fhr_st`` attribute access still works.
    """
    g = torch.Generator().manual_seed(seed)
    batch = ds2.AttributeDict()
    batch["fhr_st"] = torch.randn(n, _T, _C[0], generator=g)
    batch["fhr_ph"] = torch.randn(n, _T, _C[1], generator=g)
    batch["up_st"] = torch.randn(n, _T, _C[2], generator=g)
    batch["up_ph"] = torch.randn(n, _T, _C[3], generator=g)
    batch["weight"] = torch.ones(n, _T)
    if with_guid:
        batch["guid"] = [f"g{i:04d}" for i in range(n)]
    return batch


def _pl_module(model: SeqVaeLagAttnV2):
    r"""Wrap in the production ``SeqVaeLagAttnPl`` and set the loss hparams."""
    trainer_mod = pytest.importorskip(
        "model.vae_teb_prediction.model.trainer_lag_attn_v1"
    )
    module = trainer_mod.SeqVaeLagAttnPl(model, lr=1e-3, lr_milestones=[])
    module.hparams["kld_beta"] = 5.0e-2
    module.hparams["lambda_full"] = 1.0
    module.hparams["lambda_base"] = 0.5
    module.hparams["likelihood"] = "gaussian_nll"
    module.hparams["sigma_obs"] = 1.0
    module.hparams["free_bits"] = 0.0
    module.hparams["detach_baseline_in_full"] = True
    module.hparams["lambda_lag"] = 1.0e-3
    return module


# ---------------------------------------------------------------------------
# Loss / metrics path
# ---------------------------------------------------------------------------
def test_production_loss_metrics_v2() -> None:
    r"""The production loss/metrics builder runs on v2 and returns finite v1 keys."""
    torch.manual_seed(0)
    module = _pl_module(_model())
    batch = _batch(seed=1)
    total_loss, metrics = module.compute_loss_and_metrics(batch, 0, stage="train")

    assert torch.isfinite(total_loss)
    # Every v1 metric is present (the shared builder still emits the full v1 set
    # under v2). Under v2 it ALSO emits the guarded S7-T03 diagnostics; the "no
    # v2 key leaks under the v1 alias" guarantee is covered by
    # test_diagnostics_v2::test_metrics_guarded_under_v1.
    assert _EXPECTED_METRIC_KEYS <= set(metrics)
    for key, value in metrics.items():
        assert torch.isfinite(torch.as_tensor(value)), key


# ---------------------------------------------------------------------------
# Short fit (train + val steps + logging)
# ---------------------------------------------------------------------------
class _RandomDS(Dataset):
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> ds2.AttributeDict:
        g = torch.Generator().manual_seed(idx)
        item = ds2.AttributeDict()
        item["fhr_st"] = torch.randn(_T, _C[0], generator=g)
        item["fhr_ph"] = torch.randn(_T, _C[1], generator=g)
        item["up_st"] = torch.randn(_T, _C[2], generator=g)
        item["up_ph"] = torch.randn(_T, _C[3], generator=g)
        item["weight"] = torch.ones(_T)
        return item


def _collate(items):
    r"""Stack per-sample :class:`AttributeDict` samples into a batched one."""
    return ds2.attribute_dict_collate(items)


def test_production_fit_steps_v2() -> None:
    r"""A short ``Trainer.fit`` runs train + val steps on v2 and logs ``total_loss``."""
    import lightning as pl

    module = _pl_module(_model())
    dl = DataLoader(_RandomDS(8), batch_size=2, collate_fn=_collate)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=dl, val_dataloaders=dl)

    assert trainer.state.finished
    keys = list(trainer.callback_metrics)
    assert any("total_loss" in k for k in keys), keys


# ---------------------------------------------------------------------------
# Diagnostic callback figure from v2 keys
# ---------------------------------------------------------------------------
def test_callback_figure_from_v2_keys(tmp_path) -> None:
    r"""``LagAttnV1PlotCallback`` renders a figure from the v2 forward dict."""
    module = _pl_module(_model())
    module.eval()
    callback = LagAttnV1PlotCallback(
        output_dir=tmp_path, plot_frequency=1, num_examples=1, file_format="png",
    )
    # Drive the internal plot builder directly (bypass the epoch-frequency gate).
    callback._generate_plots(_batch(n=2, seed=2), module, epoch=0)

    figs = list(callback.output_dir.glob("*.png"))
    assert figs, "callback produced no figure from the v2 forward dict"


# ---------------------------------------------------------------------------
# Checkpoint class guard + strict reload
# ---------------------------------------------------------------------------
def test_checkpoint_guard_and_reload_v2(tmp_path) -> None:
    r"""A v2 checkpoint passes its own guard, reloads strict, and is rejected as v1."""
    from train.graph_models_utils import load_checkpoint_strict

    model = _model()
    ckpt_path = tmp_path / "final.ckpt"
    torch.save(
        {
            "model_class": "SeqVaeLagAttnV2",
            "model_state_dict": model.state_dict(),
            "model_kwargs": dict(_TINY_V2_KW),
        },
        ckpt_path,
    )

    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Correct alias name passes; the v1 name is rejected before any rebuild.
    check_model_class(blob, "SeqVaeLagAttnV2")
    with pytest.raises(ValueError):
        check_model_class(blob, "SeqVaeLagAttnV1")

    fresh = _model()
    loaded = load_checkpoint_strict(fresh, str(ckpt_path), map_location="cpu")
    assert loaded is not None
