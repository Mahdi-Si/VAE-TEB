"""Config-driven ``_build_trainer_kwargs`` reproduces the consumer trainer configs.

Assertions target the capturable kwargs dict, not a constructed ``Trainer`` — the
shipped config names 7 CUDA devices while the dev box has one, so building a real
``Trainer`` would validate against hardware. ``torch.cuda.is_available`` is pinned so
the accelerator branch is deterministic regardless of the test host.
"""
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.profilers import SimpleProfiler

from train.test_utils import make_graph_model


def _cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# --- canonical trainer (shipped config) -------------------------------------

def test_canonical_scalar_kwargs(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(config_path)
    kw = gm._build_trainer_kwargs([])

    assert kw["precision"] == "bf16-mixed"
    assert kw["log_every_n_steps"] == 1
    assert kw["num_sanity_val_steps"] == 0
    assert kw["use_distributed_sampler"] is True
    assert kw["enable_checkpointing"] is True
    assert kw["gradient_clip_val"] == 0.5
    assert kw["gradient_clip_algorithm"] == "norm"
    assert kw["max_epochs"] == gm.epochs_num
    assert kw["accumulate_grad_batches"] == gm.accumulate_grad_batches
    assert isinstance(kw["profiler"], SimpleProfiler)


def test_sync_batchnorm_single_device_false(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(config_path, **{"general_config.cuda_devices": [0]})
    kw = gm._build_trainer_kwargs([])
    assert kw["sync_batchnorm"] is False


def test_sync_batchnorm_multi_device_true(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(config_path, **{"general_config.cuda_devices": [0, 1]})
    kw = gm._build_trainer_kwargs([])
    assert kw["sync_batchnorm"] is True


def test_lr_monitor_always_attached(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(config_path)
    kw = gm._build_trainer_kwargs([])
    assert any(isinstance(cb, LearningRateMonitor) for cb in kw["callbacks"])


# --- classifier bucket-sampler path (early stopping, no distributed sampler) --

def test_early_stopping_and_no_dist_sampler(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(
        config_path,
        **{
            "advanced_config.callbacks.early_stopping.enabled": True,
            "advanced_config.callbacks.early_stopping.patience": 7,
            "advanced_config.callbacks.early_stopping.monitor": "val/loss",
            "advanced_config.trainer.use_distributed_sampler": False,
        },
    )
    kw = gm._build_trainer_kwargs([])

    early = [cb for cb in kw["callbacks"] if isinstance(cb, EarlyStopping)]
    assert len(early) == 1
    assert early[0].patience == 7
    assert early[0].monitor == "val/loss"
    assert kw["use_distributed_sampler"] is False


def test_early_stopping_absent_by_default(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(config_path)  # shipped: early_stopping.enabled=false
    kw = gm._build_trainer_kwargs([])
    assert not any(isinstance(cb, EarlyStopping) for cb in kw["callbacks"])


# --- transformer path (sanity steps, profiler off) --------------------------

def test_num_sanity_and_profiler_off(config_path, monkeypatch):
    _cpu(monkeypatch)
    gm = make_graph_model(
        config_path,
        **{
            "advanced_config.trainer.num_sanity_val_steps": 2,
            "advanced_config.trainer.profiler": None,
        },
    )
    kw = gm._build_trainer_kwargs([])
    assert kw["num_sanity_val_steps"] == 2
    assert "profiler" not in kw  # off -> key omitted (Trainer default None)


# --- select_ddp_strategy hook -----------------------------------------------

def test_select_ddp_strategy_default(config_path):
    gm = make_graph_model(config_path)
    assert gm.select_ddp_strategy(1, gm.config) == "auto"
    assert gm.select_ddp_strategy(2, gm.config) == "ddp"


def test_strategy_override_reaches_kwargs(config_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    gm = make_graph_model(config_path, **{"general_config.cuda_devices": [0, 1]})
    monkeypatch.setattr(
        type(gm),
        "select_ddp_strategy",
        lambda self, num_devices, config, model=None: "ddp_find_unused_parameters_true",
    )
    kw = gm._build_trainer_kwargs([])
    assert kw["strategy"] == "ddp_find_unused_parameters_true"
