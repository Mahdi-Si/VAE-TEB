r"""Sprint 5 (S5-T03): DDP strategy selection + CPU grad-mask zero-grad check.

Checks that ``_select_ddp_strategy`` returns the correct Lightning strategy for
single-device, curriculum, learned-sigma, and unconsumed-logvar cases, and that
under the S4-T03 grad-mask a disabled-branch parameter receives a zero (not
``None``) gradient on CPU. Real multi-GPU DDP is operator-run on the A6000 box.
See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.trainer_lag_attn_v1 import (  # noqa: E402
    GraphModelVaeTebLagAttnV1Trainer as Trainer,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    SeqVaeLagAttnV2,
)


def test_strategy_single_device() -> None:
    """One device => no DDP strategy."""
    assert Trainer._select_ddp_strategy(1, "mse", 1.0) == "auto"
    assert Trainer._select_ddp_strategy(1, "gaussian_nll", "learned") == "auto"


def test_strategy_curriculum_pins_find_unused() -> None:
    """A curriculum run pins find-unused-true regardless of likelihood."""
    assert (
        Trainer._select_ddp_strategy(
            4, "gaussian_nll", "learned", curriculum_enabled=True
        )
        == "ddp_find_unused_parameters_true"
    )
    assert (
        Trainer._select_ddp_strategy(4, "mse", 1.0, curriculum_enabled=True)
        == "ddp_find_unused_parameters_true"
    )


def test_strategy_logvar_heads_consumed() -> None:
    """gaussian_nll + learned sigma consumes the logvar heads => plain ddp."""
    assert Trainer._select_ddp_strategy(4, "gaussian_nll", "learned") == "ddp"


def test_strategy_logvar_heads_unconsumed() -> None:
    """MSE or fixed-sigma NLL leaves the logvar heads unused => find-unused-true."""
    assert (
        Trainer._select_ddp_strategy(4, "mse", 1.0)
        == "ddp_find_unused_parameters_true"
    )
    assert (
        Trainer._select_ddp_strategy(4, "gaussian_nll", 1.0)
        == "ddp_find_unused_parameters_true"
    )


def test_gradmask_zero_grad_cpu() -> None:
    """Under grad-mask, a disabled-branch param gets a zero (not None) CPU grad."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True, enable_source=False).train()
    B, T = 2, 80
    y_st, y_ph, u = torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101)
    out = model(y_st, y_ph, u)
    model.zero_grad(set_to_none=True)
    model.compute_loss(out, y_st, y_ph, beta=0.05)["total_loss"].backward()

    g = model.source_encoder.proj.weight.grad
    assert g is not None, "grad-mask must leave a zero grad, not None"
    assert float(g.abs().sum()) == 0.0
