r"""Sprint 1 (S1-T01/T02): v2 deterministic encoders.

Checks the multi-scale gated causal target encoder (shape, strict causality,
two-sided receptive field $R = 253$) and the bounded source lag-atom encoder
(shape, two-sided bounded support). See ``vae-teb-lag-attn-v2-spec-and-sprints.md``
Sprint 1.
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
    SourceInputAdapterV2,
    SourceLagAtomEncoder,
    TargetCausalEncoderV2,
)


def _influence_profile(enc, x, t):
    r"""Return $\sum_c |(J^\top v)_{s}|$ over input positions $s$ for a random $v$.

    A gradient-based dependency profile is exact (structurally zero outside the
    receptive field, nonzero inside) and so is robust where a finite-difference
    check at the exact boundary would round the single farthest tap path -- a
    product of six small conv weights -- below tolerance. A random cotangent $v$
    is used (rather than a plain feature-sum) because at init the final
    ``LayerNorm`` has weight $1$ / bias $0$, so $\sum_c \mathrm{out}_{t,c}$ is
    identically zero and its gradient would vanish spuriously.
    """
    xg = x.clone().requires_grad_(True)
    out = enc(xg)
    gen = torch.Generator().manual_seed(12345)
    cot = torch.randn(out[:, t].shape, dtype=out.dtype, generator=gen)
    (grad,) = torch.autograd.grad((out[:, t] * cot).sum(), xg)
    return grad.abs().sum(dim=(0, 2))  # (T,)


def test_target_encoder() -> None:
    """Shape, strict causality, and an exact two-sided receptive field of 253."""
    torch.manual_seed(0)
    d_model, T = 16, 300
    enc = TargetCausalEncoderV2(
        d_model=d_model,
        num_blocks=6,
        kernel_size=5,
        dilations=(1, 2, 4, 8, 16, 32),
        dropout=0.0,
    ).double().eval()
    assert enc.receptive_field == 253

    x = torch.randn(1, T, d_model, dtype=torch.float64)
    with torch.no_grad():
        base = enc(x)
    assert base.shape == (1, T, d_model)

    r = enc.receptive_field
    t = T - 1  # last position sees a full receptive field

    influence = _influence_profile(enc, x, t)
    # Two-sided receptive field: the exact boundary at t-(R-1) still influences
    # output at t; one step further (t-R) is structurally outside the field.
    assert influence[t] > 0
    assert influence[t - (r - 1)] > 0, "the t-(R-1) boundary tap must reach t"
    assert influence[t - r].item() == 0.0, "t-R is outside the field (exact zero)"
    assert torch.all(influence[: t - (r - 1)] == 0), "no influence before the field"

    # Strict causality: for an earlier anchor, no future input influences it.
    anchor = 150
    infl_mid = _influence_profile(enc, x, anchor)
    assert torch.all(infl_mid[anchor + 1 :] == 0), "future input reached a past output"
    assert infl_mid[anchor] > 0


def test_source_atoms() -> None:
    """Source atoms have shape ``(B, T, d)`` and a bounded, causal support."""
    torch.manual_seed(0)
    c_u, d_u, d_model, T = 101, 8, 16, 60
    adapter = SourceInputAdapterV2(in_dim=c_u, d_u=d_u, dropout=0.0).double().eval()
    atoms = SourceLagAtomEncoder(
        d_u=d_u, d_model=d_model, scales=(3, 9, 21), dropout=0.0
    ).double().eval()
    assert atoms.max_lookback == 20

    u = torch.randn(2, T, c_u, dtype=torch.float64)
    with torch.no_grad():
        r_u = atoms(adapter(u))
    assert r_u.shape == (2, T, d_model)

    # Two-sided bounded support on the atom encoder's own input: the atom at s
    # depends only on positions [s - max_lookback, s].
    u_tilde = torch.randn(1, T, d_u, dtype=torch.float64)
    s = 40
    infl = _influence_profile(atoms, u_tilde, s)
    lb = atoms.max_lookback
    assert torch.all(infl[s + 1 :] == 0), "atom depends on a future position (not causal)"
    assert torch.all(infl[: s - lb] == 0), "atom depends beyond its bounded window"
    assert infl[s - lb] > 0, "the s-max_lookback boundary must contribute"
    assert infl[s] > 0, "the current position must contribute"
