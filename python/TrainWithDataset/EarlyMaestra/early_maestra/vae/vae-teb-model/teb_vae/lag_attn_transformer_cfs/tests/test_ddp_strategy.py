r"""The DDP strategy this driver selects, and why every setting in it is licensed here too.

The selector is the shared driver's, reached through both parents, so what this file re-earns is not
the code but the *claim*: that on this composition every parameter is reachable, no buffer is a
running statistic, and the static-graph promise is one the loss-spike breaker would break. Each is a
property of the model rather than of the selector, and each would fail silently on a development box
and loudly on the first production step.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer

from .conftest import shipped_warmup_kwargs

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config; ``setup_config`` is never called.

    The shipped config rather than the tiny one: nothing here reads the shards, and what is under
    test is the strategy the *production* configuration selects.
    """
    driver = LagAttnTrfCfsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    return driver


def _config(**vae_overrides) -> dict:
    """A minimal config carrying only the keys the strategy selector reads."""
    return {"model_config": {"VAE_model": dict(vae_overrides)}}


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
def test_the_shipped_config_earns_every_parameter_reachable(trainer):
    """The payoff of the learned observation variance plus the unconditional ``W_o`` freeze: the
    reducer can expect every parameter."""
    assert trainer.ddp_kwargs(trainer.config)["find_unused_parameters"] is False


def test_a_single_device_needs_no_strategy(trainer):
    assert trainer.select_ddp_strategy(1, trainer.config) == "auto"


def test_the_smoke_configs_mse_selects_the_fallback(trainer):
    """``tiny.yaml`` ships ``likelihood: mse`` precisely so the smoke path exercises this branch
    where it is cheap to observe, rather than leaving it configured and never run."""
    from teb_vae.lag_attn.config import load_config

    tiny = load_config(str(_TINY))

    assert tiny["model_config"]["VAE_model"]["likelihood"] == "mse"
    assert trainer.ddp_kwargs(tiny)["find_unused_parameters"] is True
    assert trainer.ddp_kwargs(_config(likelihood="mse"))["find_unused_parameters"] is True


def test_the_buffer_broadcast_is_off_and_the_gradients_are_bucket_views(trainer):
    """Two performance settings the shorthand strategy strings cannot express, which is why the
    selector returns an instance."""
    kwargs = trainer.ddp_kwargs(trainer.config)

    assert kwargs["broadcast_buffers"] is False
    assert kwargs["gradient_as_bucket_view"] is True


def test_no_buffer_is_a_running_statistic_so_the_broadcast_is_safe_to_skip():
    """What licenses ``broadcast_buffers=False``: every buffer is a deterministic function of the
    config, built identically in each rank's constructor, so the broadcast restores values that were
    never going to differ. A ``BatchNorm`` running statistic is the one kind that genuinely diverges
    per rank, and there is none.

    This target domain adds three buffers of its own -- the warm-up tertile assignment and the two
    per-block source warmth patterns -- and every one is a function of the resolved budget and the
    geometry, so they belong in the same category. They are non-persistent for a second reason:
    their contents follow the budget, so a persistent copy would make a checkpoint trained at one
    budget fail to load at another and report it as misaligned keys rather than as a budget
    mismatch."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCfs(**shipped_warmup_kwargs())
    buffers = dict(model.named_buffers())

    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    for name in (
        "warm_tertile_id",
        "novelty_tertile_id",
        "source_block_warm_st",
        "source_block_warm_ph",
    ):
        assert name in buffers, name
        assert name not in model.state_dict(), f"{name} reaches a checkpoint"


def test_static_graph_is_not_claimed(trainer):
    """A correctness call rather than an omission: the loss-spike breaker substitutes a
    zero-weighted sum over every parameter on a skipped batch, which is a structurally different
    backward from the one iteration 1 recorded. ``static_graph=True`` promises DDP that never
    happens, and the breaker ships enabled."""
    assert "static_graph" not in trainer.ddp_kwargs(trainer.config)


def test_the_settings_reach_the_strategy_object(trainer):
    """``DDPStrategy`` forwards unrecognised kwargs into ``_ddp_kwargs`` and on to
    ``DistributedDataParallel``. That name is Lightning-internal, so it is asserted here only."""
    strategy = trainer.select_ddp_strategy(8, trainer.config)

    assert type(strategy).__name__ == "DDPStrategy"
    assert strategy._ddp_kwargs == trainer.ddp_kwargs(trainer.config)


def test_the_selector_is_a_pure_function_of_config(trainer):
    """The framework passes the *Lightning module* as ``model``, not the raw net; a selector that
    read a net attribute off it would find nothing and silently regress the shipped config to the
    slow strategy on the one box where it costs."""
    without_model = trainer.select_ddp_strategy(8, trainer.config)
    with_wrapper = trainer.select_ddp_strategy(8, trainer.config, model=object())

    assert without_model._ddp_kwargs == with_wrapper._ddp_kwargs


def test_the_hook_is_the_un_prefixed_name_the_framework_looks_up():
    """The framework calls ``select_ddp_strategy`` and nothing else. Inherited here through both
    parents, so what is asserted is that this driver did not shadow it with an underscore-prefixed
    copy that would never run."""
    assert "_select_ddp_strategy" not in vars(LagAttnTrfCfsTrainer)
    assert LagAttnTrfCfsTrainer.select_ddp_strategy is LagAttnRwsTrainer.select_ddp_strategy
