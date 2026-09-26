r"""Construction: the one thing the composition changes, and the empty class body that guarantees it.

The *difference* is exactly one: each horizon token emits one value per surviving target channel
instead of $R$ raw samples. It shows up in the decoder's two output heads, the last axis of the four
forecast tensors and the parameter total, and nowhere else -- so the total against the architecture
parent must move by exactly the two widened heads.

The *sameness* is the empty class body: with nothing defined here, the forward keys, the posterior's
structure, the lag map, the construction-time refusals and every latent shape are the parents' own
code objects, pinned by the parents' own suites over the same functions.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    BATCH,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws


def _model(kwargs, cls=SeqVaeLagAttnTrfFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _n_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


#: The three keyword sets the width rule is checked at: a small hand-made gate, the production
#: geometry without a budget, and the production geometry at the shipped budget.
_KWARGS = {
    "tiny-gated": tiny_gated_kwargs,
    "shipped-ungated": lambda: shipped_gated_kwargs(None),
    "shipped-gated": shipped_gated_kwargs,
}


@pytest.mark.parametrize("arm", list(_KWARGS))
def test_the_forecast_tensors_carry_the_target_width_and_nothing_else_moves(arm):
    r"""$(B, T_{\mathrm{valid}}, H, C)$, where $C$ is the gate's surviving-channel count, or $c_y$
    when there is no gate. The width is not a constructor keyword, so it can only follow the gate;
    the latent side is untouched, which is what makes the last axis the *only* delta."""
    kwargs = _KWARGS[arm]()
    model = _model(kwargs, dropout=0.0).eval()
    channels = len(kwargs["target_keep_index"]) if "target_keep_index" in kwargs else kwargs["c_y"]
    split = SeqVaeLagAttnTrfFs.TARGET_BLOCK_SPLIT
    length = int(kwargs["sequence_length"])
    generator = torch.Generator().manual_seed(0)
    streams = (
        torch.randn(BATCH, length, split, generator=generator),
        torch.randn(BATCH, length, int(kwargs["c_y"]) - split, generator=generator),
        torch.randn(BATCH, length, int(kwargs["c_u"]), generator=generator),
    )

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*streams)

    assert model.decoder.mean_head.out_features == channels
    assert model.decoder.logvar_head.out_features == channels
    expected = (BATCH, model.geometry.t_valid, model.horizon, channels)
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == expected, key
    assert tuple(out["mu_prior"].shape) == (BATCH, length, int(kwargs["d_z"]))


@pytest.mark.parametrize("budget_s", [None, 120.0], ids=["ungated", "gated"])
def test_the_delta_against_the_architecture_parent_is_the_two_widened_heads(budget_s):
    r"""Both heads are $\mathrm{Linear}(d_{\mathrm{in}}, X)$, so widening $X$ from the raw
    parent's $R$ to this model's $C$ costs $2 (d_{\mathrm{in}} + 1)(C - R)$ parameters, and nothing
    anywhere else in the model may move."""
    kwargs = shipped_gated_kwargs(budget_s)
    raw = _model(kwargs, cls=SeqVaeLagAttnTrfRws)
    feature = _model(kwargs)
    d_in = feature.decoder.mean_head.in_features

    assert raw.decoder.mean_head.in_features == d_in
    assert _n_parameters(feature) - _n_parameters(raw) == 2 * (d_in + 1) * (
        feature.decoder_out_channels - raw.decoder_out_channels
    )


def test_the_class_itself_defines_nothing_at_all():
    """The cheapest guarantee the suite has: with nothing defined here, every behaviour the parents'
    suites pin is this class's behaviour too. A line count would pass a class that overrode
    ``forward``; an empty ``vars`` does not."""
    assert [name for name in vars(SeqVaeLagAttnTrfFs) if not name.startswith("__")] == []
    assert "__init__" not in vars(SeqVaeLagAttnTrfFs)
