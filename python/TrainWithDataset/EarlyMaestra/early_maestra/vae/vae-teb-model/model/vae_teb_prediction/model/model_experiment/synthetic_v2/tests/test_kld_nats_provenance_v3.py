r"""S2-T02: ``kld_nats`` reads ``kld_raw`` (G4 provenance).

Under G4 only the RAW per-step KL may be read as a TE surrogate; ``kld_loss`` is the
free-bit-FLOORED, optimised ``kld_train``. With ``free_bits > 0`` these diverge, and
``kld_nats`` must track ``kld_raw * d_z``, not ``kld_train * d_z`` -- so ``beta_select``'s
least-collapsed ranking and the K-bar calibration key on the raw KL regardless of the
free-bits value. Uses ``free_bits=0.2`` adversarially even though the shipped config sets 0.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    SyntheticSeqVaeLagAttnV2Pl,
    build_model,
)

_CPU = torch.device("cpu")
_D_Z = 8
_TINY = {
    "sequence_length": 32, "d_model": 16, "d_z": _D_Z, "horizon": 4, "warmup_period": 2,
    "c_y": 87, "c_u": 101, "use_up_st": True, "max_lag": 8, "num_heads": 4, "d_head": 4,
    "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
}
_V3 = {**_TINY, "class": "SeqVaeLagAttnV3",
       "v3": {"posterior_logvar": "residual", "kld_support": "anchor",
              "logvar_bound": "smooth"}}


def _batch(B=2, T=32):
    torch.manual_seed(0)
    return types.SimpleNamespace(
        fhr_st=torch.randn(B, T, 43), fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43), up_ph=torch.randn(B, T, 58),
        weight=torch.ones(B, T))


def _metrics(model_cfg, *, free_bits, likelihood="gaussian_nll", sigma_obs="learned"):
    model, _ = build_model(model_cfg, _CPU)
    wrap = SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood=likelihood, sigma_obs=sigma_obs,
        free_bits=free_bits, detach_baseline_in_full=True)
    wrap.eval()
    with torch.no_grad():
        _, metrics = wrap.compute_loss_and_metrics(_batch(), 0, "train")
    return {k: (float(v) if torch.is_tensor(v) else v) for k, v in metrics.items()}


def test_kld_nats_tracks_kld_raw_with_free_bits() -> None:
    """With free_bits=0.2, kld_nats == kld_raw * d_z (NOT kld_train * d_z)."""
    m = _metrics(_V3, free_bits=0.2)
    assert "kld_raw" in m and "kld_train" in m
    assert m["kld_train"] > m["kld_raw"]  # the floor is active
    assert abs(m["kld_nats"] - m["kld_raw"] * _D_Z) < 1e-6
    # And crucially NOT the floored training term.
    assert abs(m["kld_nats"] - m["kld_train"] * _D_Z) > 1e-6


def test_kld_nats_unchanged_at_free_bits_zero() -> None:
    """With free_bits=0, kld_raw == kld_loss, so kld_nats == kld_loss * d_z (as before)."""
    m = _metrics(_V3, free_bits=0.0)
    assert abs(m["kld_raw"] - m["kld_loss"]) < 1e-6
    assert abs(m["kld_nats"] - m["kld_loss"] * _D_Z) < 1e-6


def test_kld_nats_unchanged_under_v1() -> None:
    """A v1 model exposes no kld_raw; kld_nats falls back to kld_loss * d_z unchanged."""
    m = _metrics(_TINY, free_bits=0.0, likelihood="mse", sigma_obs=1.0)
    assert "kld_raw" not in m  # v1 loss_dict has no raw/train split
    assert abs(m["kld_nats"] - m["kld_loss"] * _D_Z) < 1e-6
