r"""Sprint 7 (S7-T02): supervised outcome head (arch spec section 22).

Checks the default-off Stage-4 classifier: the head builds
$r_t^{\mathrm{cls}} = [h^y \mid z \mid K_t \mid \bar\ell \mid \sigma^2_\ell]$,
attention-pools it into segment logits, and trains on a FROZEN variational model
(Stage 4a) so only the head moves; the light fine-tune path (Stage 4b) re-enables
exactly the configured subset. Labels never enter ``forward``. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` S7-T02.
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

from model.vae_teb_prediction.model.outcome_head_v2 import (  # noqa: E402
    outcome_loss,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_T = 32
_N = 8
_NUM_CLASSES = 3

_TINY: Dict[str, Any] = {
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
    "use_outcome_head": True,
    "outcome_classes": _NUM_CLASSES,
}


def _streams(n: int = _N, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, _T, 43, generator=g),
        torch.randn(n, _T, 44, generator=g),
        torch.randn(n, _T, 101, generator=g),
    )


def test_outcome_head_shapes_and_loss() -> None:
    r"""The head returns ``(B, num_classes)`` logits and a finite loss."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(**_TINY)
    assert model.outcome_head is not None
    model.eval()
    y_st, y_ph, u = _streams(seed=1)
    with torch.no_grad():
        fo = model(y_st, y_ph, u)
        logits = model.outcome_logits(fo)
    assert logits.shape == (_N, _NUM_CLASSES)
    assert torch.isfinite(logits).all()
    labels = torch.randint(0, _NUM_CLASSES, (_N,))
    loss = outcome_loss(logits, labels, _NUM_CLASSES)
    assert torch.isfinite(loss)


def test_outcome_logits_raises_when_disabled() -> None:
    r"""``outcome_logits`` raises when the head is not enabled."""
    model = SeqVaeLagAttnV2(**{**_TINY, "use_outcome_head": False})
    assert model.outcome_head is None
    y_st, y_ph, u = _streams(seed=2)
    with torch.no_grad():
        fo = model(y_st, y_ph, u)
    with pytest.raises(RuntimeError):
        model.outcome_logits(fo)


def test_frozen_vae_trains_only_head() -> None:
    r"""Stage 4a: a frozen VAE stays fixed while the head fits and the loss drops."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(**_TINY)
    model.eval()   # deterministic z (posterior means); head still receives grads
    model.freeze_vae()

    y_st, y_ph, u = _streams(seed=3)
    labels = torch.randint(0, _NUM_CLASSES, (_N,))

    # Snapshots: a deep VAE param must not move; a head param must move.
    vae_ref = model.target_encoder.final_norm.weight.detach().clone()
    head_ref = model.outcome_head.classifier.weight.detach().clone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    # Only head params are trainable under freeze_vae.
    assert all(
        id(p) in {id(hp) for hp in model.outcome_head.parameters()}
        for p in trainable
    )
    opt = torch.optim.Adam(trainable, lr=1e-2)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        fo = model(y_st, y_ph, u)
        loss = outcome_loss(model.outcome_logits(fo), labels, _NUM_CLASSES)
        loss.backward()
        opt.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0]                       # the head fits the labels
    assert torch.allclose(model.target_encoder.final_norm.weight, vae_ref)  # frozen
    assert not torch.allclose(model.outcome_head.classifier.weight, head_ref)


def test_unfreeze_finetune_subset() -> None:
    r"""Stage 4b re-enables exactly the last encoder block + source encoder + head."""
    model = SeqVaeLagAttnV2(**_TINY)
    model.unfreeze_finetune()

    def _all_grad(module) -> bool:
        return all(p.requires_grad for p in module.parameters())

    def _no_grad(module) -> bool:
        return all(not p.requires_grad for p in module.parameters())

    # Enabled subset.
    assert _all_grad(model.outcome_head)
    assert _all_grad(model.target_encoder.blocks[-1])
    assert _all_grad(model.target_encoder.final_norm)
    assert _all_grad(model.source_encoder)
    # Deep bottleneck stays frozen.
    assert _no_grad(model.target_encoder.blocks[0])
    assert _no_grad(model.prior_head)
    assert _no_grad(model.baseline_decoder)
    assert _no_grad(model.lag_posterior)
