"""P4-01: ``patchify``, ``patch_summaries`` and ``PatchEmbedding`` (plan B.3, B.4)."""
from __future__ import annotations

import math

import numpy as np
import torch

from teb_vae.lag_attn.nets.blocks import initialization
from teb_vae.lag_attn_transformer_patch.nets.patching import PatchEmbedding, patchify

from .conftest import build

NAN = math.nan


def test_patchify_layout_and_validity_channel_against_a_hand_built_input() -> None:
    """``[value, delta, m_t - 1]`` per token. Token 1 has weight 0, token 2 one NaN sample, token 3
    is all NaN. ``"finite"`` ignores the weight, so token 1 is valid there and its first delta
    crosses the (valid) boundary. Exact equality also pins finiteness of the all-NaN patch."""
    raw = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, NAN, 11, 12, NAN, NAN, NAN, NAN]], dtype=torch.float64
    )
    weight = torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    zeros = [0, 0, 0, 0, 0, 0, 0, 0]
    expected = {
        "fhr_weight": [
            [1, 2, 3, 4, 0, 1, 1, 1, 0],
            zeros + [-1],
            [9, 0, 11, 12, 0, 0, 0, 1, -1],
            zeros + [-1],
        ],
        "finite": [
            [1, 2, 3, 4, 0, 1, 1, 1, 0],
            [5, 6, 7, 8, 1, 1, 1, 1, 0],
            [9, 0, 11, 12, 1, 0, 0, 1, -1],
            zeros + [-1],
        ],
    }
    for validity, rows in expected.items():
        out = patchify(raw, weight, raw_per_step=4, validity=validity)
        assert torch.equal(out, torch.tensor([rows], dtype=torch.float64)), (validity, out)


def test_summary_target_matches_numpy_and_ignores_the_boundary_difference() -> None:
    """Through the model's target builder (patchify -> patch_summaries -> identity affine), so a
    target that read the delta block, whose first entry crosses the patch boundary, is caught.

    Dyadic samples make a whole-patch shift exact: every inside-patch difference is unchanged
    bitwise, while both boundary differences of the shifted patch change."""
    model = build()
    r, eps = model.raw_per_step, model.variability_eps
    gen = torch.Generator().manual_seed(0)
    raw = torch.randint(-64, 64, (2, 300 * r), generator=gen).double() / 8.0
    weight = torch.ones(2, 300, dtype=torch.float64)
    weight[0, 5] = 0.0
    raw[1, 7 * r + 3] = NAN

    out = model.summary_target(raw, weight).numpy()

    x = raw.numpy().reshape(2, 300, r)
    valid = (weight.numpy() >= 0.5) & np.isfinite(x).all(-1)
    ref = np.stack(
        [x.mean(-1), np.log(np.sqrt((np.diff(x, axis=-1) ** 2).mean(-1)) + eps)], axis=-1
    )
    ref = np.where(valid[..., None], ref, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert np.isfinite(out).all() and (out[0, 5] == 0).all() and (out[1, 7] == 0).all()

    k = 100
    shifted = raw.clone()
    shifted[:, k * r : (k + 1) * r] += 3.0
    moved = model.summary_target(shifted, weight).numpy()
    assert np.array_equal(moved[..., 1], out[..., 1]), "variability read a boundary difference"
    changed = np.flatnonzero((moved[..., 0] != out[..., 0]).any(0))
    assert changed.tolist() == [k]


def test_patch_embedding_missing_replaces_exactly_the_invalid_tokens() -> None:
    """``missing`` on ``m_t = 0`` tokens only; an all-zero stream (the source-null control's null)
    is all valid; the generic ``initialization`` pass leaves the bare parameter alone (a zeroed
    ``missing`` would put exact zero vectors into the norm layers)."""
    torch.manual_seed(0)
    emb = PatchEmbedding(in_dim=9, d_model=8, sequence_length=6, dropout=0.0).eval()
    before = emb.missing.detach().clone()
    initialization(emb)
    assert torch.equal(emb.missing, before) and before.abs().min() > 0

    x = torch.randn(2, 6, 9)
    invalid = torch.zeros(2, 6, dtype=torch.bool)
    invalid[0, 2] = invalid[1, 0] = invalid[1, 5] = True
    x[..., -1] = -invalid.float()
    with torch.no_grad():
        out, adapted = emb(x), emb.adapter(x)
        zero_in = torch.zeros(1, 6, 9)
        assert torch.equal(emb(zero_in), emb.adapter(zero_in))
    assert torch.equal(out[invalid], emb.missing.detach().expand(3, -1))
    assert torch.equal(out[~invalid], adapted[~invalid])
