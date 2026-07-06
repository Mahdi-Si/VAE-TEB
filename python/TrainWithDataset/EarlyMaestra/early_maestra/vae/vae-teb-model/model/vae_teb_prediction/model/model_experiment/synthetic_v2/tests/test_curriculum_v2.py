r"""Sprint 4 (S4-T03/T04): curriculum flags, DDP grad-mask, stage-schedule mapping.

Checks that disabled branches receive zero (not ``None``) gradients under the
grad-mask, that ``active_lags`` is settable, and that the pure epoch->stage
mapping resolves the correct flags / active-lag count / beta at each boundary. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    SeqVaeLagAttnV2,
)


def test_gradmask_zero_not_none_when_source_disabled() -> None:
    """enable_source=False => conditional-branch params get zero (not None) grad."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True, enable_source=False).train()
    B, T = 2, 80
    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u = torch.randn(B, T, 101)
    out = model(y_st, y_ph, u)                       # baseline-only path
    model.zero_grad(set_to_none=True)
    model.compute_loss(out, y_st, y_ph, beta=0.05)["total_loss"].backward()

    conditional = [
        model.source_encoder.proj.weight,
        model.source_adapter.linear.weight,
        model.lag_posterior.q_proj.weight,
        model.lag_latent_head.logvar_head.weight,
        model.lag_prior.W_h.weight,
        model.residual_decoder.mean_head.weight,
    ]
    for p in conditional:
        assert p.grad is not None, "grad-mask must leave a zero grad, not None"
        assert float(p.grad.abs().sum()) == 0.0

    # The always-on baseline branch DOES train.
    assert model.baseline_decoder.mean_head.weight.grad is not None
    assert float(model.baseline_decoder.mean_head.weight.grad.abs().sum()) > 0.0


def test_active_lags_settable() -> None:
    """Changing active_lags changes the active count on the next forward."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    B, T = 2, 40
    args = (torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))
    assert model.active_lags == 8
    assert model(*args)["active_lag_indices"].shape[-1] == 8
    model.active_lags = 16
    assert model(*args)["active_lag_indices"].shape[-1] == 16


def test_stage_mapping() -> None:
    """The pure epoch->stage mapping returns the right flags / Ka / beta."""
    stages = SeqVaeLagAttnV2.default_curriculum_stages()

    s0 = SeqVaeLagAttnV2._resolve_stage(0, stages)
    assert s0["enable_source"] is False and s0["active_lags"] == 8
    assert s0["beta"] == 0.0

    s5 = SeqVaeLagAttnV2._resolve_stage(5, stages)
    assert s5["enable_source"] is True and s5["active_lags"] == 16
    assert abs(s5["beta"] - 1.0e-4) < 1e-12

    # Stage 3 warm-up: beta ramps 1e-4 -> 5e-2 over 50 epochs from epoch 10.
    s10 = SeqVaeLagAttnV2._resolve_stage(10, stages)
    assert s10["active_lags"] == 8 and abs(s10["beta"] - 1.0e-4) < 1e-12
    s35 = SeqVaeLagAttnV2._resolve_stage(35, stages)
    expected = 1.0e-4 + (5.0e-2 - 1.0e-4) * ((35 - 10) / 50.0)
    assert abs(s35["beta"] - expected) < 1e-9
    s60 = SeqVaeLagAttnV2._resolve_stage(60, stages)
    assert abs(s60["beta"] - 5.0e-2) < 1e-9

    assert SeqVaeLagAttnV2._resolve_active_lags(5, stages) == 16
    assert SeqVaeLagAttnV2._resolve_active_lags(0, stages) == 8


def test_resolve_stage_order_independent() -> None:
    """_resolve_stage picks the greatest start_epoch <= epoch even if stages unsorted."""
    stages = [
        {"start_epoch": 10, "active_lags": 8, "enable_source": True,
         "enable_residual": True, "enable_kl": True, "beta": 0.05},
        {"start_epoch": 0, "active_lags": 16, "enable_source": False,
         "enable_residual": False, "enable_kl": False, "beta": 0.0},
    ]
    s0 = SeqVaeLagAttnV2._resolve_stage(0, stages)
    assert s0["enable_source"] is False and s0["active_lags"] == 16
    s5 = SeqVaeLagAttnV2._resolve_stage(5, stages)
    assert s5["enable_source"] is False and s5["active_lags"] == 16
    s10 = SeqVaeLagAttnV2._resolve_stage(10, stages)
    assert s10["enable_source"] is True and s10["active_lags"] == 8


def test_enable_residual_gates_forward() -> None:
    """enable_residual=False => mu_full == mu_base and logvar_full == logvar_base."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True, enable_residual=False).eval()
    B, T = 2, 40
    out = model(torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))
    assert float(out["delta_mu_src"].abs().max()) == 0.0
    assert torch.equal(out["mu_full"], out["mu_base"])
    assert torch.equal(out["logvar_full"], out["logvar_base"])
    # The source latent / KL are still computed (the flag only gates the forecast).
    assert out["kld_per_t"].shape == (B, T)


def test_attended_source_width_stable_across_stages() -> None:
    """attended_source keeps width M*d_v in both baseline and source paths."""
    torch.manual_seed(0)
    # d_head=16 => M*d_v = 64 != d_model = 128, which would expose a width mismatch.
    model = SeqVaeLagAttnV2(use_entmax=True, d_head=16).eval()
    B, T = 2, 40
    args = (torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))

    model.enable_source = False
    base = model(*args)
    model.enable_source = True
    src = model(*args)

    assert base["attended_source"].shape == (B, T, 4 * 16)
    assert src["attended_source"].shape == (B, T, 4 * 16)
    assert base["attended_source_heads"].shape == (B, T, 4, 16)
    assert src["attended_source_heads"].shape == (B, T, 4, 16)


def test_set_curriculum_stage_mutates_model() -> None:
    """set_curriculum_stage applies flags / Ka in place and returns beta."""
    stages = SeqVaeLagAttnV2.default_curriculum_stages()
    model = SeqVaeLagAttnV2(use_entmax=True)

    beta0 = model.set_curriculum_stage(0, stages)
    assert model.enable_source is False and model.active_lags == 8 and beta0 == 0.0

    beta5 = model.set_curriculum_stage(5, stages)
    assert model.enable_source is True and model.active_lags == 16
    assert abs(beta5 - 1.0e-4) < 1e-12
