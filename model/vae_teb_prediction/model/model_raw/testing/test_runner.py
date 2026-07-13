r"""S6-T02: the raw-adapted :class:`TestRunner` builds ``SeqVaeRawV4`` and runs the raw forward.

Validates that ``TestRunner.from_checkpoint`` rebuilds the raw model from the tiny fixture's stamped
``model_kwargs``, that ``forward`` returns the 25-key raw dict with decoder tensors on the full $T$
anchor axis, and that ``build_future_target`` gathers the crop-aligned raw future block on the
$T_{\mathrm{valid}}$ axis. Tiny geometry: $L_{\mathrm{raw}}=512, D=16 \Rightarrow T=28, T_{valid}=24,
H=4, R=16$.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.conftest import make_raw_stub_batch

# Tiny geometry constants (must match conftest.TINY_*).
_T = 28
_T_VALID = 24
_H = 4
_R = 16


def _runner(tiny_checkpoint, tmp_path) -> TestRunner:
    ckpt_path, _kwargs = tiny_checkpoint
    return TestRunner.from_checkpoint(ckpt_path, tmp_path / "out")


def test_from_checkpoint_builds_seqvaerawv4(tiny_checkpoint, tmp_path) -> None:
    """The runner rebuilds a ``SeqVaeRawV4`` from the stamped kwargs and mirrors its geometry."""
    from model.vae_teb_prediction.model.model_raw.vae_teb_raw_v4 import SeqVaeRawV4

    runner = _runner(tiny_checkpoint, tmp_path)
    assert isinstance(runner.model, SeqVaeRawV4)
    geo = runner.geometry()
    assert geo.t == _T
    assert geo.t_valid == _T_VALID
    assert geo.horizon == _H
    assert geo.r == _R
    assert runner.horizon == _H
    assert runner.warmup_steps == 2


def test_forward_returns_raw_25_key_dict(tiny_checkpoint, tmp_path) -> None:
    """``forward`` returns the raw forecast on the full $T$ anchor axis, shape $(B, T, H, R)$."""
    runner = _runner(tiny_checkpoint, tmp_path)
    batch = make_raw_stub_batch(batch_size=2, raw_len=512)
    with runner.inference_mode():
        b = next(iter(runner.iter_batches([batch])))
        outputs = runner.forward(b)

    assert outputs["raw_future_pred"] is not None
    assert tuple(outputs["mu_full"].shape) == (2, _T, _H, _R)
    assert tuple(outputs["mu_base"].shape) == (2, _T, _H, _R)
    assert tuple(outputs["logvar_full"].shape) == (2, _T, _H, _R)
    # Inherited v3 keys keep their shapes.
    assert tuple(outputs["attn_weights"].shape) == (2, _T, 4, 9)  # max_lag=8 -> 9 taps
    assert tuple(outputs["kld_per_t"].shape) == (2, _T)


def test_build_future_target_is_crop_aligned_raw_block(tiny_checkpoint, tmp_path) -> None:
    """``build_future_target`` gathers $(B, T_{valid}, H, R)$ from raw ``fhr``."""
    runner = _runner(tiny_checkpoint, tmp_path)
    batch = make_raw_stub_batch(batch_size=3, raw_len=512)
    x_plus = runner.build_future_target(batch)
    assert tuple(x_plus.shape) == (3, _T_VALID, _H, _R)


def test_forward_with_compute_loss_attaches_raw_loss_dict(tiny_checkpoint, tmp_path) -> None:
    """``forward(compute_loss=True)`` attaches the raw single-phase loss dict."""
    runner = _runner(tiny_checkpoint, tmp_path)
    batch = make_raw_stub_batch(batch_size=2, raw_len=512)
    with runner.inference_mode():
        b = next(iter(runner.iter_batches([batch])))
        outputs = runner.forward(b, compute_loss=True, beta=0.1)
    loss = outputs["loss_dict"]
    for key in ("feat_loss", "raw_loss", "base_loss", "kld_loss", "total_loss"):
        assert key in loss
        assert torch.isfinite(loss[key]).all()
