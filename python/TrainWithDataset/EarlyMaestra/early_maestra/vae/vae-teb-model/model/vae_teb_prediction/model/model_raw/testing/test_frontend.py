r"""S1-T01/T03/T04/T05: shapes, causality, featurisation, and assembly of the raw front end.

Causality is checked with autograd: for output token ``t`` we backprop ``out[..., t].sum()`` and
assert the input gradient is exactly zero past the token's raw causal endpoint (no future leak)
and nonzero within it (the endpoint is real, not vacuously immune).
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.model_raw.raw_frontend import (
    CausalAntiAliasDownsample1d,
    CausalRawFrontend,
    RawFrontendBlock,
    assert_no_time_pooling_norm,
)


def _assert_causal_endpoint(fn, x, t, endpoint) -> None:
    """Assert ``fn(x)[..., t]`` depends only on ``x[..., <= endpoint]`` (and genuinely on it).

    The positive control backprops a fixed **random projection** of the output token, not a plain
    channel-sum: a channel-normalising norm (``causal_group_norm``) makes ``sum_c out[c, t]``
    identically zero, so a channel-sum probe gives zero gradient and the control would flakily fail.
    """
    x = x.clone().detach().requires_grad_(True)
    tok = fn(x)[:, :, t]  # (B, C)
    gen = torch.Generator().manual_seed(0)
    (tok * torch.randn(tok.shape, generator=gen)).sum().backward()
    g = x.grad
    assert torch.all(g[:, :, endpoint + 1 :] == 0), f"future leak past {endpoint} for token {t}"
    assert torch.any(g[:, :, : endpoint + 1] != 0), f"token {t} vacuously immune"


# ---------------------------------------------------------------------------
# S1-T01: CausalAntiAliasDownsample1d
# ---------------------------------------------------------------------------
def test_downsample_shape() -> None:
    for aa in (True, False):
        m = CausalAntiAliasDownsample1d(4, stride=2, antialias=aa).eval()
        y = m(torch.randn(2, 4, 5280))
        assert y.shape == (2, 4, 2640)


def test_downsample_shape_odd_length() -> None:
    # Generic guard: right-offset slice length == len(range(stride-1, L, stride)).
    m = CausalAntiAliasDownsample1d(3, stride=2, antialias=True).eval()
    y = m(torch.randn(1, 3, 7))
    assert y.shape[-1] == len(range(1, 7, 2)) == 3


def test_downsample_causal() -> None:
    m = CausalAntiAliasDownsample1d(3, stride=2, antialias=True).eval()
    x = torch.randn(2, 3, 64)
    for t in (0, 5, 20):
        _assert_causal_endpoint(m, x, t, 2 * t + 1)


def test_downsample_causal_no_antialias() -> None:
    m = CausalAntiAliasDownsample1d(3, stride=2, antialias=False).eval()
    x = torch.randn(2, 3, 64)
    for t in (0, 5, 20):
        _assert_causal_endpoint(m, x, t, 2 * t + 1)


def test_downsample_fir_is_buffer_not_parameter() -> None:
    m = CausalAntiAliasDownsample1d(4, stride=2, antialias=True)
    assert "lp" in dict(m.named_buffers())
    assert list(m.parameters()) == []  # FIR is a buffer; the module has no learnable params
    m_off = CausalAntiAliasDownsample1d(4, stride=2, antialias=False)
    assert "lp" not in dict(m_off.named_buffers())


# ---------------------------------------------------------------------------
# S1-T03: RawFrontendBlock
# ---------------------------------------------------------------------------
def _block(gated=True, norm_kind="causal_group_norm"):
    return RawFrontendBlock(
        8, 16, kernel_size=15, stride=2, gated=gated,
        norm_kind=norm_kind, norm_num_groups=8,
    ).eval()


@pytest.mark.parametrize("gated", [True, False])
def test_block_shape(gated) -> None:
    b = _block(gated=gated)
    y = b(torch.randn(2, 8, 128))
    assert y.shape == (2, 16, 64)


def test_block_first_stage_multiscale_shape() -> None:
    b = RawFrontendBlock(
        3, 32, stride=2, gated=True, norm_kind="causal_group_norm",
        norm_num_groups=8, first_stage_kernels=(7, 31, 65),
    ).eval()
    y = b(torch.randn(2, 3, 5280))
    assert y.shape == (2, 32, 2640)


def test_block_causal() -> None:
    b = _block().eval()
    x = torch.randn(2, 8, 128)
    for t in (0, 10, 40):
        _assert_causal_endpoint(b, x, t, 2 * t + 1)


@pytest.mark.parametrize("norm_kind", ["causal_group_norm", "cln", "channel_affine"])
def test_block_passes_norm_guard(norm_kind) -> None:
    assert_no_time_pooling_norm(_block(norm_kind=norm_kind))  # must not raise


# ---------------------------------------------------------------------------
# S1-T04: featurize
# ---------------------------------------------------------------------------
def _fe(**kw):
    return CausalRawFrontend(stream="y", mean=140.0, std=20.0, **kw).eval()


def test_featurize_shape_and_derivative() -> None:
    fe = _fe()
    raw = torch.randn(2, 5280) * 20 + 140
    mask = torch.ones(2, 5280)
    feat = fe.featurize(raw, mask)
    assert feat.shape == (2, 3, 5280)
    x_std, mask_ch, dx = feat[:, 0], feat[:, 1], feat[:, 2]
    # dx[n] = x_std[n] - x_std[n-1], with dx[0] == 0 (x[-1] := x[0]).
    assert torch.allclose(dx[:, 1:], x_std[:, 1:] - x_std[:, :-1], atol=1e-6)
    assert torch.allclose(dx[:, 0], torch.zeros(2), atol=1e-6)
    # mask channel is an un-normalized passthrough.
    assert torch.allclose(mask_ch, mask)


def test_featurize_fixed_stats_proof() -> None:
    # A per-segment-mean z-score would leave the shift invariant; fixed stats shift by delta/std.
    fe = _fe()
    raw = torch.randn(2, 256) * 20 + 140
    mask = torch.ones(2, 256)
    delta = 3.0
    shift = fe.featurize(raw + delta, mask)[:, 0] - fe.featurize(raw, mask)[:, 0]
    assert torch.allclose(shift, torch.full_like(shift, delta / 20.0), atol=1e-5)


def test_featurize_nan_and_sentinel_safety() -> None:
    fe = _fe(sentinel=0.0)
    raw = torch.randn(2, 256) * 20 + 140
    mask = torch.ones(2, 256)
    mask[:, 10:30] = 0.0
    raw[:, 10:20] = float("nan")
    raw[:, 20:30] = 0.0  # sentinel gap
    feat = fe.featurize(raw, mask)
    assert torch.isfinite(feat).all()


def test_featurize_gaps_are_neutral() -> None:
    # Gaps must read as NEUTRAL 0 in standardized space (not -mean/std ~ -7), with the mask channel
    # carrying invalidity; a sentinel-valued sample the caller's mask missed is also invalidated.
    fe = _fe(sentinel=0.0)  # mean=140, std=20
    raw = torch.randn(2, 256) * 20 + 140
    mask = torch.ones(2, 256)
    mask[:, 50:60] = 0.0     # masked gap
    raw[:, 100:110] = 0.0    # sentinel gap at mask==1 positions
    feat = fe.featurize(raw, mask)
    x_std, m, dx = feat[:, 0], feat[:, 1], feat[:, 2]
    # Masked gap: value neutral (~0, emphatically NOT -mean/std = -7), mask 0, derivative 0.
    assert x_std[:, 50:60].abs().max() < 1e-6
    assert torch.allclose(m[:, 50:60], torch.zeros(2, 10))
    assert dx[:, 50:60].abs().max() < 1e-6
    # Sentinel gap at mask==1 is invalidated by the effective-mask refinement.
    assert x_std[:, 100:110].abs().max() < 1e-6
    assert torch.allclose(m[:, 100:110], torch.zeros(2, 10))
    # Valid positions still carry a real (nonzero) standardized value.
    assert x_std[:, :50].abs().max() > 0.1


# ---------------------------------------------------------------------------
# S1-T05: CausalRawFrontend assembly
# ---------------------------------------------------------------------------
def test_frontend_shape_default() -> None:
    fe = _fe()
    y = fe(torch.randn(2, 5280) * 20 + 140, torch.ones(2, 5280))
    assert y.shape == (2, 300, 128)


def test_frontend_shape_up_stream() -> None:
    fe = CausalRawFrontend(stream="u", mean=0.0, std=1.0).eval()
    y = fe(torch.randn(2, 5280), torch.ones(2, 5280))
    assert y.shape == (2, 300, 128)


def test_frontend_shape_single_stride16() -> None:
    fe = _fe(stages=(16,))
    y = fe(torch.randn(2, 5280) * 20 + 140, torch.ones(2, 5280))
    assert y.shape == (2, 300, 128)


@pytest.mark.parametrize("norm_kind", ["causal_group_norm", "cln", "channel_affine"])
def test_frontend_shape_norm_kinds(norm_kind) -> None:
    fe = _fe(norm_kind=norm_kind)
    y = fe(torch.randn(2, 5280) * 20 + 140, torch.ones(2, 5280))
    assert y.shape == (2, 300, 128)


def test_frontend_shape_no_antialias() -> None:
    fe = _fe(antialias=False)
    y = fe(torch.randn(2, 5280) * 20 + 140, torch.ones(2, 5280))
    assert y.shape == (2, 300, 128)


def test_frontend_finite_under_gaps() -> None:
    fe = _fe()
    raw = torch.randn(2, 5280) * 20 + 140
    mask = torch.ones(2, 5280)
    mask[:, 100:200] = 0.0
    raw[:, 100:150] = float("nan")
    raw[:, 150:200] = 0.0
    y = fe(raw, mask)
    assert torch.isfinite(y).all()


def test_frontend_rejects_bad_stage_product() -> None:
    with pytest.raises(ValueError):
        CausalRawFrontend(stream="y", mean=0.0, std=1.0, stages=(2, 2, 2), decimation=16)


def test_frontend_rejects_bad_stream() -> None:
    with pytest.raises(ValueError):
        CausalRawFrontend(stream="z", mean=0.0, std=1.0)
