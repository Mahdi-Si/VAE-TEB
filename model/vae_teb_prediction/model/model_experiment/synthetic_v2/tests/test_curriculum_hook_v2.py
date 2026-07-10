r"""Sprint 5 (S5-T02): curriculum epoch hook in the Lightning modules.

Checks that ``_apply_curriculum`` flips ``enable_*`` / ``active_lags`` on the
wrapped model and updates ``kld_beta`` at the configured epoch boundary, that the
synthetic and production modules share the same stage-mapping source of truth, and
that a disabled / absent curriculum is a no-op. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402
    SyntheticSeqVaeLagAttnV2Pl,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_STAGES = [
    {
        "start_epoch": 0, "active_lags": 16, "enable_source": False,
        "enable_residual": False, "enable_kl": False, "beta": 0.0,
    },
    {
        "start_epoch": 2, "active_lags": 8, "enable_source": True,
        "enable_residual": True, "enable_kl": True,
        "beta": {"kind": "constant", "value": 5.0e-2},
    },
]


def test_synthetic_hook_flips_at_boundary() -> None:
    """The synthetic module flips flags / Ka and beta at the stage boundary."""
    model = SeqVaeLagAttnV2(use_entmax=True)
    module = SyntheticSeqVaeLagAttnV2Pl(
        model, curriculum={"enabled": True, "stages": _STAGES}
    )

    b0 = module._apply_curriculum(0)
    assert model.enable_source is False and model.active_lags == 16
    assert b0 == 0.0 and float(module.hparams["kld_beta"]) == 0.0

    # Just before the boundary the epoch-1 stage is still stage 1.
    module._apply_curriculum(1)
    assert model.enable_source is False and model.active_lags == 16

    b2 = module._apply_curriculum(2)
    assert model.enable_source is True and model.active_lags == 8
    assert abs(b2 - 5.0e-2) < 1e-12
    assert abs(float(module.hparams["kld_beta"]) - 5.0e-2) < 1e-12


def test_synthetic_hook_disabled_is_noop() -> None:
    """An absent or disabled curriculum leaves the model untouched."""
    model = SeqVaeLagAttnV2(use_entmax=True)
    src0 = model.enable_source
    m_none = SyntheticSeqVaeLagAttnV2Pl(model, curriculum=None)
    assert m_none._apply_curriculum(9) is None
    assert model.enable_source == src0

    m_off = SyntheticSeqVaeLagAttnV2Pl(
        model, curriculum={"enabled": False, "stages": _STAGES}
    )
    assert m_off._apply_curriculum(9) is None
    assert model.enable_source == src0


def test_v2_kld_nats_scale_and_lambda_lag() -> None:
    """Under a v2 model, kld_nats == kld_loss (no d_z rescale) and lambda_lag is forwarded."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True)
    module = SyntheticSeqVaeLagAttnV2Pl(model, lambda_lag=1.0e-3)
    assert float(module.hparams["lambda_lag"]) == 1.0e-3

    B, T = 2, 80  # > warmup_period + horizon so the loss mask is non-empty
    batch = SimpleNamespace(
        fhr_st=torch.randn(B, T, 43),
        fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43),
        up_ph=torch.randn(B, T, 58),
        weight=torch.ones(B, T),
    )
    _, metrics = module.compute_loss_and_metrics(batch, 0, stage="train")
    # v2 kld_loss is already nats/step (summed over heads/dims), so kld_nats must
    # equal it, NOT kld_loss * d_z (the v1 per-dim-mean correction).
    assert torch.allclose(
        torch.as_tensor(metrics["kld_nats"]),
        torch.as_tensor(metrics["kld_loss"]),
    )


def test_production_hook_shares_mapping() -> None:
    """The production module uses the same stage mapping (set hparams then apply)."""
    trainer_mod = pytest.importorskip(
        "model.vae_teb_prediction.model.trainer_lag_attn_v1"
    )
    model = SeqVaeLagAttnV2(use_entmax=True)
    module = trainer_mod.SeqVaeLagAttnPl(model, lr=1e-3)
    module.hparams["curriculum"] = {"enabled": True, "stages": _STAGES}

    b2 = module._apply_curriculum(2)
    assert model.enable_source is True and model.active_lags == 8
    assert abs(b2 - 5.0e-2) < 1e-12
    # Same resolver as the pure static mapping.
    assert (
        b2 == SeqVaeLagAttnV2._resolve_stage(2, _STAGES)["beta"]
    )
