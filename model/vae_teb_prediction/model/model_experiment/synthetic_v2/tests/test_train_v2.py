r"""Tests for the Sprint 5 training layer (S5-T01/T02/T03).

Covers :mod:`pl_module_v2`: the Lightning wrapper
(:class:`SyntheticSeqVaeLagAttnV2Pl`) completing one epoch, the v1-compatible
checkpoint round-trip through :func:`train.graph_models_utils.load_checkpoint_strict`,
the ``mse`` / ``gaussian_nll`` likelihood switch and the ``free_bits`` KL floor, the
:func:`train_v2` pilot smoke (checkpoint + loss curves), and the optional
:func:`beta_select` enumeration.

Everything runs on CPU against a **tiny in-test fixture cache** (small ``T``, a
handful of samples) and a **tiny model** (small ``d_model`` / ``d_z`` / ``horizon``
with the production ``c_y=87`` / ``c_u=101`` channel counts), so the suite stays fast
and needs no GPU or a real dataset build. See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md``
Sprint 5.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

# Force the repo root ahead of the sibling ``model/vae_teb_prediction`` on ``sys.path``
# (which lacks ``utils.custom_logger``) so the model imports resolve under pytest -- the
# same guard ``pl_module_v2`` applies at import time. See test_build_dataset_v2's
# ``test_model_forward_compat``.
_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    dataset_v2 as ds2,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    pl_module_v2 as plm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    resolve_cache_dir,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"

# Channel counts are fixed by the transform / model contract.
_C_FHR_ST, _C_FHR_PH, _C_UP_ST, _C_UP_PH = 43, 44, 43, 58

# Tiny model: production channel counts, everything else shrunk for a fast CPU run.
_T = 32          # sequence length (== data T)
_HORIZON = 4
_WARMUP = 2
_MAX_LAG = 8
_DELAY = 4       # fixed source->target lag D in the fixture provenance

_TINY_MODEL: Dict[str, Any] = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": _HORIZON,
    "warmup_period": _WARMUP,
    "c_y": _C_FHR_ST + _C_FHR_PH,   # 87
    "c_u": _C_UP_ST + _C_UP_PH,     # 101
    "use_up_st": True,
    "max_lag": _MAX_LAG,
    "num_heads": 2,
    "d_head": 8,                    # num_heads * d_head == d_model
    "lstm_layers": 1,
    "dropout": 0.0,
    "decoder_hidden": 16,
    "logvar_clamp": [-5.0, 3.0],
    "mu_scale": 5.0,
    "delta_mu_scale": 3.0,
    "latent_stats_momentum": 0.01,
    "use_entmax": False,
    "attention_grad_checkpoint": False,
    "head_structured_latent": False,
    "lag_bias_init": "normal",
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
    "encoder_extra_dilations": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _write_split(path: Path, n: int, *, seed: int) -> None:
    r"""Write one tiny uncompressed split ``.npz`` with the v2 native + provenance fields."""
    rng = np.random.default_rng(seed)
    arrays = {
        "fhr_st": rng.standard_normal((n, _T, _C_FHR_ST)).astype(np.float32),
        "fhr_ph": rng.standard_normal((n, _T, _C_FHR_PH)).astype(np.float32),
        "up_st": rng.standard_normal((n, _T, _C_UP_ST)).astype(np.float32),
        "up_ph": rng.standard_normal((n, _T, _C_UP_PH)).astype(np.float32),
        "weight": np.ones((n, _T), dtype=np.float32),
        "true_lag_tt": np.full((n, _T), _DELAY, dtype=np.float32),
        "sample_te_true": np.full((n,), 1.0, dtype=np.float32),
        "sample_te_scat": np.full((n,), 1.1, dtype=np.float32),
        "sample_frac_phi": np.full((n,), 1.1, dtype=np.float32),
        "sample_delay": np.full((n,), _DELAY, dtype=np.int16),
        "sample_cell_id": np.zeros((n,), dtype=np.int16),
        "sample_held_out": np.zeros((n,), dtype=np.int8),
    }
    np.savez(path, **arrays)  # uncompressed -> memory-mappable


def _write_fixture_cache(cache_dir: Path, *, n_train: int = 8, n_val: int = 4) -> None:
    r"""Populate ``cache_dir`` with ``train.npz`` / ``val.npz`` + ``meta.json``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_split(cache_dir / "train.npz", n_train, seed=0)
    _write_split(cache_dir / "val.npz", n_val, seed=1)
    meta = {
        "te_true": 1.0,
        "true_lag_band": list(range(max(0, _DELAY - _HORIZON), _DELAY)),
        "tag": "test_tiny",
        "benchmark": "G1_raw",
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle)


@pytest.fixture
def tiny_config(tmp_path) -> Dict[str, Any]:
    r"""A trimmed config wired to a tiny fixture cache under ``tmp_path``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    cfg["model"] = copy.deepcopy(_TINY_MODEL)
    cfg["experiment"]["tag"] = "test_tiny"
    cfg["experiment"]["benchmark"] = "G1_raw"
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg["dataset"] = {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "mmap": "auto"}
    cfg["optim"]["batch_size"] = 4
    cfg["optim"]["epochs"] = 1
    cfg["optim"]["lr_milestones"] = []  # no scheduler churn in a 1-epoch smoke

    cache_dir = resolve_cache_dir(cfg, benchmark="G1_raw")
    _write_fixture_cache(cache_dir)
    return cfg


@pytest.fixture
def force_cpu(monkeypatch) -> None:
    r"""Force the training driver onto CPU (deterministic, no GPU contention)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _random_batch(n: int = 4, *, seed: int = 0) -> ds2.AttributeDict:
    r"""Build a random batched :class:`AttributeDict` matching the tiny model."""
    g = torch.Generator().manual_seed(seed)
    batch = ds2.AttributeDict()
    batch["fhr_st"] = torch.randn(n, _T, _C_FHR_ST, generator=g)
    batch["fhr_ph"] = torch.randn(n, _T, _C_FHR_PH, generator=g)
    batch["up_st"] = torch.randn(n, _T, _C_UP_ST, generator=g)
    batch["up_ph"] = torch.randn(n, _T, _C_UP_PH, generator=g)
    batch["weight"] = torch.ones(n, _T)
    return batch


# ---------------------------------------------------------------------------
# S5-T01: module one-epoch fit
# ---------------------------------------------------------------------------
def test_module_one_epoch(tiny_config) -> None:
    r"""The wrapper trains one CPU epoch on the fixture and logs ``kld_nats``."""
    import lightning as pl

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v2 import (
        SyntheticTEDataModuleV2,
    )

    model, _ = plm.build_model(tiny_config["model"], torch.device("cpu"))
    pl_model = plm.SyntheticSeqVaeLagAttnV2Pl(model, lr=1e-3, kld_beta=1e-3)
    dm = SyntheticTEDataModuleV2(tiny_config, batch_size=4, benchmark="G1_raw")

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
    trainer.fit(pl_model, datamodule=dm)

    assert trainer.state.finished
    keys = list(trainer.callback_metrics)
    assert any("kld_nats" in k for k in keys), keys
    assert any("total_loss" in k for k in keys), keys


# ---------------------------------------------------------------------------
# S5-T01: checkpoint round-trip (graph_models_utils)
# ---------------------------------------------------------------------------
def test_ckpt_roundtrip(tiny_config, tmp_path) -> None:
    r"""Save then reload (strict) reproduces the deterministic forward outputs."""
    from train.graph_models_utils import load_checkpoint_strict

    model_a, kwargs = plm.build_model(tiny_config["model"], torch.device("cpu"))
    ckpt_path = tmp_path / "final.ckpt"
    plm.save_checkpoint_v2(
        ckpt_path,
        model=model_a,
        model_kwargs=kwargs,
        config=tiny_config,
        data_meta={},
        epoch=1,
        val_loss=float("nan"),
        loss_settings={"beta": 1e-3},
        latent_stats_fitted=False,
    )
    assert ckpt_path.is_file()

    model_b, _ = plm.build_model(tiny_config["model"], torch.device("cpu"))
    loaded = load_checkpoint_strict(model_b, str(ckpt_path), map_location="cpu")
    assert loaded is not None

    batch = _random_batch(seed=7)
    u_stream = ds2.build_u_stream(batch)
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        out_a = model_a(batch.fhr_st, batch.fhr_ph, u_stream)
        torch.manual_seed(0)
        out_b = model_b(batch.fhr_st, batch.fhr_ph, u_stream)

    # ``mu_post`` / ``kld_per_t`` are deterministic (independent of the sampled z).
    assert torch.allclose(out_a["mu_post"], out_b["mu_post"], atol=1e-6)
    assert torch.allclose(out_a["kld_per_t"], out_b["kld_per_t"], atol=1e-6)


# ---------------------------------------------------------------------------
# S5-T01: likelihood switch + free_bits
# ---------------------------------------------------------------------------
def test_loss_switch_and_free_bits(tiny_config) -> None:
    r"""``mse`` and ``gaussian_nll`` both give finite losses; ``free_bits`` clamps KL."""
    model, _ = plm.build_model(tiny_config["model"], torch.device("cpu"))
    model.eval()
    batch = _random_batch(seed=3)
    u_stream = ds2.build_u_stream(batch)
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, u_stream)

    common = dict(
        forward_outputs=out, y_st=batch.fhr_st, y_ph=batch.fhr_ph,
        weight=batch.weight, beta=1e-3, lambda_full=1.0, lambda_base=0.5,
    )
    loss_mse = model.compute_loss(**common, likelihood="mse", sigma_obs=1.0, free_bits=0.0)
    loss_nll = model.compute_loss(**common, likelihood="gaussian_nll", sigma_obs=1.0, free_bits=0.0)
    assert torch.isfinite(loss_mse["total_loss"])
    assert torch.isfinite(loss_nll["total_loss"])

    # ``free_bits`` lower-bounds each per-dim per-step KL, so their mean is >= the floor.
    floor = 0.5
    loss_fb = model.compute_loss(**common, likelihood="mse", sigma_obs=1.0, free_bits=floor)
    assert float(loss_fb["kld_loss"]) >= floor - 1e-6


# ---------------------------------------------------------------------------
# S5-T02: train_v2 pilot smoke
# ---------------------------------------------------------------------------
def test_train_v2_smoke(tiny_config, force_cpu) -> None:
    r"""``train_v2`` (pilot overrides) writes a checkpoint and a loss-curve figure."""
    overrides = {
        "epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "batch_size": 4,
        "devices": 1,
        "progress_bar": False,
    }
    result = plm.train_v2(tiny_config, overrides, benchmark="G1_raw")

    assert result["checkpoint"] is not None and Path(result["checkpoint"]).is_file()
    assert Path(result["checkpoint"]).name == "final.ckpt"
    assert result["best"] is not None and Path(result["best"]).is_file()
    assert any("total_loss" in k for k in result["metrics"]), result["metrics"]
    assert result["figures"], "expected at least one loss-curve figure"
    for path in result["figures"]:
        assert Path(path).is_file()
    # The live HTML callback leaves a self-contained interactive curve at fit end
    # (sibling of the matplotlib figures); plotly is available in this env.
    html = Path(result["figures"][0]).with_suffix(".html")
    assert html.is_file(), f"expected live HTML loss curve at {html}"


# ---------------------------------------------------------------------------
# S5-T03: beta_select
# ---------------------------------------------------------------------------
def test_beta_select_disabled_is_noop(tiny_config) -> None:
    r"""With ``beta_select.enabled=false`` and no force, the stage is a reporting no-op."""
    tiny_config["beta_select"] = {"enabled": False, "beta_grid": [1e-4, 1e-3], "epochs": 1}
    result = plm.beta_select(tiny_config, {}, benchmark="G1_raw")
    assert result["enabled"] is False
    assert result["results"] == []
    assert result["selected_beta"] == pytest.approx(float(tiny_config["loss"]["kld_beta"]))


def test_beta_select_enumerates(tiny_config, force_cpu) -> None:
    r"""Force-run beta_select over a tiny grid: one record per beta + a selection."""
    grid = [1e-4, 1e-3]
    overrides = {
        "force": True,
        "beta_grid": grid,
        "epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "batch_size": 4,
        "devices": 1,
        "progress_bar": False,
    }
    result = plm.beta_select(tiny_config, overrides, benchmark="G1_raw")

    assert result["enabled"] is True
    assert len(result["results"]) == len(grid)
    for row in result["results"]:
        assert set(row) == {"beta", "kld_nats", "total_loss"}
    assert result["selected_beta"] in grid
    assert Path(result["out_path"]).is_file()
    with open(result["out_path"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["selected_beta"] in grid
    assert len(payload["results"]) == len(grid)


# ---------------------------------------------------------------------------
# Monitor-key resolver
# ---------------------------------------------------------------------------
def test_resolve_epoch_metric() -> None:
    r"""Bare names gain ``_epoch``; already-suffixed names are returned unchanged."""
    assert plm._resolve_epoch_metric("val/total_loss") == "val/total_loss_epoch"
    assert plm._resolve_epoch_metric("train/kld_nats") == "train/kld_nats_epoch"
    # Idempotent on the two Lightning-forked suffixes.
    assert plm._resolve_epoch_metric("val/total_loss_epoch") == "val/total_loss_epoch"
    assert plm._resolve_epoch_metric("val/total_loss_step") == "val/total_loss_step"


# ---------------------------------------------------------------------------
# EarlyStopping wiring + monitor resolution + live HTML callback
# ---------------------------------------------------------------------------
def _build_trainer(
    tmp_path: Path,
    *,
    es_enabled: bool,
    has_val: bool,
    plotting: Dict[str, Any] | None = None,
    patience: int = 15,
):
    r"""Build a trainer via ``_build_trainer_v2`` with a controllable ddp/plotting cfg."""
    ddp_cfg = {
        **plm._DDP_DEFAULTS,
        "early_stopping": {
            "enabled": es_enabled,
            "monitor": "val/total_loss",
            "mode": "min",
            "patience": patience,
            "min_delta": 0.0,
        },
    }
    return plm._build_trainer_v2(
        ddp_cfg=ddp_cfg,
        results_dir=tmp_path / "results",
        epochs=1,
        devices=1,
        n_gpus=0,  # forces the CPU trainer branch regardless of host CUDA
        has_val=has_val,
        overrides={"progress_bar": False},
        plotting_cfg=plotting if plotting is not None else {"enabled": True, "html": True},
    )


def test_early_stopping_present_when_enabled(tmp_path, force_cpu) -> None:
    r"""``enabled`` + a val split registers EarlyStopping on the resolved epoch key."""
    from lightning.pytorch.callbacks import EarlyStopping

    trainer = _build_trainer(tmp_path, es_enabled=True, has_val=True, patience=7)
    stoppers = [c for c in trainer.callbacks if isinstance(c, EarlyStopping)]
    assert len(stoppers) == 1
    assert stoppers[0].monitor == "val/total_loss_epoch"
    assert stoppers[0].patience == 7


def test_early_stopping_absent_when_disabled(tmp_path, force_cpu) -> None:
    r"""``enabled: false`` (the shipped default) registers no EarlyStopping."""
    from lightning.pytorch.callbacks import EarlyStopping

    trainer = _build_trainer(tmp_path, es_enabled=False, has_val=True)
    assert not any(isinstance(c, EarlyStopping) for c in trainer.callbacks)


def test_early_stopping_absent_without_val(tmp_path, force_cpu) -> None:
    r"""Enabled but no val split: skipped (nothing to monitor)."""
    from lightning.pytorch.callbacks import EarlyStopping

    trainer = _build_trainer(tmp_path, es_enabled=True, has_val=False)
    assert not any(isinstance(c, EarlyStopping) for c in trainer.callbacks)


def test_model_checkpoint_monitor_is_resolved(tmp_path, force_cpu) -> None:
    r"""The best-checkpoint monitor uses the real ``_epoch`` key (was the silent bug)."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    trainer = _build_trainer(tmp_path, es_enabled=False, has_val=True)
    checkpoints = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
    assert len(checkpoints) == 1
    assert checkpoints[0].monitor == "val/total_loss_epoch"


def test_html_callback_toggles_with_config(tmp_path, force_cpu) -> None:
    r"""The live HTML callback is present iff ``plotting.enabled`` and ``plotting.html``."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.callbacks_v2 import (
        LossPlotHtmlCallback,
    )

    on = _build_trainer(
        tmp_path, es_enabled=False, has_val=True, plotting={"enabled": True, "html": True}
    )
    assert any(isinstance(c, LossPlotHtmlCallback) for c in on.callbacks)

    off = _build_trainer(
        tmp_path, es_enabled=False, has_val=True, plotting={"enabled": True, "html": False}
    )
    assert not any(isinstance(c, LossPlotHtmlCallback) for c in off.callbacks)
