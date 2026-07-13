r"""S1-T06: strict stagewise and full-model causality of the raw front end.

Two-sided proof (for both ``stages`` configs and every ``norm_kind``):

1. **No-future-leak** -- backprop ``out[..., token].sum()`` and assert the raw gradient is exactly
   zero past the token's causal endpoint. This is the licence for reading the downstream $K_t$ as a
   transfer-entropy surrogate.
2. **Positive control** -- assert the token genuinely depends on its **own** endpoint sample. A
   too-tight left-offset decimation (``filtered[stride*t]`` instead of ``filtered[stride*t+stride-1]``)
   is still causally clean but would make the token immune to the last 15/16 of its present block and
   misalign with ``n_raw``/``future_block_start``; this control fails loudly on that mistake.

Small front ends (``raw_len=512``) are used because causality is structural, not length-dependent,
so the autograd sweep over stages x norm_kinds stays fast. The full-length ``(B,5280)->(B,300,128)``
geometry is covered in ``test_frontend.py``.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.model_raw.raw_frontend import CausalRawFrontend

_NORM_KINDS = ["causal_group_norm", "cln", "channel_affine"]
_STAGES = [(2, 2, 2, 2), (16,)]


def _small_fe(*, stream="y", stages=(2, 2, 2, 2), norm_kind="causal_group_norm", antialias=True):
    """A small (raw_len=512), otherwise-faithful front end for fast autograd causality checks."""
    return CausalRawFrontend(
        stream=stream,
        mean=0.0,
        std=1.0,
        raw_len=512,
        decimation=16,
        crop=2,
        stages=stages,
        norm_kind=norm_kind,
        antialias=antialias,
    ).eval()


def _proj_backward(out: torch.Tensor) -> None:
    """Backprop a fixed random projection of ``out``.

    A plain channel-sum is a degenerate probe: a channel-normalising norm (``causal_group_norm``,
    ``cln``) makes ``sum_c out[c, t]`` identically zero, so its gradient is exactly zero and the
    positive control would spuriously fail (or pass only on float noise). A random projection over
    channels breaks that cancellation and yields a genuine gradient at the endpoint, while leaving
    the no-future-leak check (exact zero past the endpoint) unaffected.
    """
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(out.shape, generator=gen)
    (out * w).sum().backward()


def _check_causal_token(module_out_fn, raw, endpoint) -> None:
    """Backprop a token and assert (no future leak past endpoint) + (depends on endpoint)."""
    raw_g = raw.clone().detach().requires_grad_(True)
    _proj_backward(module_out_fn(raw_g))
    g = raw_g.grad
    assert torch.all(g[:, endpoint + 1 :] == 0), f"future leak past raw index {endpoint}"
    assert torch.any(g[:, endpoint] != 0), f"token does not depend on its endpoint {endpoint}"


# ---------------------------------------------------------------------------
# Full-model causality: cropped token t depends only on raw <= n_raw(t).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stages", _STAGES)
@pytest.mark.parametrize("norm_kind", _NORM_KINDS)
def test_full_model_causality(stages, norm_kind) -> None:
    fe = _small_fe(stages=stages, norm_kind=norm_kind)
    raw = torch.randn(2, 512)
    mask = torch.ones(2, 512)
    for t in (0, 5, fe.t - 1):
        # Cropped anchor t is the uncropped token t + crop; endpoint = D*(t+crop+1) - 1.
        endpoint = fe.decimation * (t + fe.crop + 1) - 1
        _check_causal_token(lambda r: fe(r, mask)[:, t, :], raw, endpoint)


@pytest.mark.parametrize("norm_kind", _NORM_KINDS)
def test_full_model_causality_no_antialias(norm_kind) -> None:
    fe = _small_fe(stages=(2, 2, 2, 2), norm_kind=norm_kind, antialias=False)
    raw = torch.randn(2, 512)
    mask = torch.ones(2, 512)
    for t in (0, 5, fe.t - 1):
        endpoint = fe.decimation * (t + fe.crop + 1) - 1
        _check_causal_token(lambda r: fe(r, mask)[:, t, :], raw, endpoint)


# ---------------------------------------------------------------------------
# Stagewise causality: after s stages (cumulative stride S), output index p
# depends only on raw <= S*(p+1) - 1.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stages", _STAGES)
@pytest.mark.parametrize("norm_kind", _NORM_KINDS)
def test_stagewise_causality(stages, norm_kind) -> None:
    fe = _small_fe(stages=stages, norm_kind=norm_kind)
    raw = torch.randn(2, 512)
    mask = torch.ones(2, 512)
    cum = 1
    for j, s in enumerate(stages):
        cum *= s
        # A couple of output indices that keep the endpoint inside the length-512 window.
        for p in (0, 2, 5):
            endpoint = cum * (p + 1) - 1
            if endpoint >= raw.shape[-1]:
                continue

            def stage_out(r, _j=j, _p=p):
                x = fe.featurize(r, mask)
                for k in range(_j + 1):
                    x = fe.blocks[k](x)
                return x[:, :, _p]

            _check_causal_token(stage_out, raw, endpoint)


# ---------------------------------------------------------------------------
# The positive control is discriminating: a left-offset decimation is caught.
# ---------------------------------------------------------------------------
def test_positive_control_catches_left_offset(monkeypatch) -> None:
    """Sanity-check the control itself: patched left-offset decimation must fail the endpoint test."""
    import model.vae_teb_prediction.model.model_raw.raw_frontend as rf

    orig_forward = rf.CausalAntiAliasDownsample1d.forward

    def left_offset_forward(self, x):
        if self.antialias:
            import torch.nn.functional as F

            x = F.pad(x, (self.k_lp - 1, 0))
            x = F.conv1d(x, self.lp, groups=self.num_channels)
        # WRONG on purpose: left element of each stride group.
        return x[:, :, :: self.stride]

    monkeypatch.setattr(rf.CausalAntiAliasDownsample1d, "forward", left_offset_forward)
    fe = _small_fe(stages=(2, 2, 2, 2))
    raw = torch.randn(2, 512)
    mask = torch.ones(2, 512)
    t = 5
    endpoint = fe.decimation * (t + fe.crop + 1) - 1  # the CORRECT (right-offset) endpoint
    raw_g = raw.clone().detach().requires_grad_(True)
    _proj_backward(fe(raw_g, mask)[:, t, :])
    # With left-offset decimation the token does NOT reach its right-offset endpoint sample.
    assert torch.all(raw_g.grad[:, endpoint] == 0)

    monkeypatch.setattr(rf.CausalAntiAliasDownsample1d, "forward", orig_forward)
