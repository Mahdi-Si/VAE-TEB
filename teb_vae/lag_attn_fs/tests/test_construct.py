r"""Construction: the decoder emits one value per surviving target channel, and nothing else moves.

A subclass is only worth having if it is provably one. The first test pins the *difference* -- the
decoder width follows the target gate at every budget and at none, and an explicit
``decoder_out_channels`` still wins. The second pins the *sameness* as shapes: against the raw
sibling built from the same kwargs, every parameter tensor has the same name, and the only ones
whose shape differs are the decoder's two output heads, whose leading axis is the surviving
channel count instead of the raw grid's $R$.

The sibling's construction-time refusals, head structure and initialisation are inherited unchanged
and pinned in the sibling's own suite.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    SHIPPED_KWARGS,
    TINY_KEEP_INDEX,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

#: An explicit width no gate in this suite implies, so an override that was ignored is visible.
_EXPLICIT_WIDTH = 7


def _model(kwargs, cls=SeqVaeLagAttnFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _width_case(arm: str):
    """Return ``(kwargs, expected decoder width)`` for one construction arm.

    The expected width is read off the gate's own inputs -- the keep-index or the declared $c_y$ --
    rather than written down, so the rule is checked rather than a number.

    Args:
        arm: One of the parametrised arm names.

    Returns:
        The constructor kwargs and the width the decoder must emit.
    """
    if arm == "shipped-budget":
        kwargs = shipped_gated_kwargs()
        return kwargs, len(kwargs["target_keep_index"])
    if arm == "unguarded":
        return dict(SHIPPED_KWARGS), SHIPPED_KWARGS["c_y"]
    if arm == "tiny-guard":
        return tiny_gated_kwargs(), len(TINY_KEEP_INDEX)
    return dict(tiny_gated_kwargs(), decoder_out_channels=_EXPLICIT_WIDTH), _EXPLICIT_WIDTH


@pytest.mark.parametrize("arm", ["shipped-budget", "unguarded", "tiny-guard", "explicit-width"])
def test_the_decoder_width_follows_the_target_gate(arm):
    """$C_{\\mathrm{keep}}$ under a budget, $c_y$ without one, and an explicit keyword when given.

    Not a configuration key: the width follows the gate, so a run cannot decode a width its target
    does not have. The keyword is not removed, only defaulted differently, so a sweep arm that wants
    another width can still say so. The mixin must come ahead of the base in the MRO for its width
    hook to win; reversed, every gated arm here would decode the raw grid's $R$ instead.
    """
    kwargs, expected = _width_case(arm)
    model = _model(kwargs)

    assert model.decoder_out_channels == expected
    assert model.decoder.out_channels == expected


def test_only_the_decoder_heads_change_shape_against_the_raw_model(shipped_gated):
    """The structural claim, stated as shapes rather than values.

    Every parameter tensor has the same name in both models and the same shape in all but the
    decoder's two output heads. Values are *not* comparable: the heads are drawn at a different
    size, so the shared RNG stream shifts from that draw onward.
    """
    raw_model = _model(shipped_gated, cls=SeqVaeLagAttnRws)
    feature_model = _model(shipped_gated)
    raw = dict(raw_model.named_parameters())
    feature = dict(feature_model.named_parameters())

    assert set(raw) == set(feature)
    reshaped = sorted(name for name in raw if raw[name].shape != feature[name].shape)
    assert reshaped == ["decoder.logvar_head.bias", "decoder.logvar_head.weight",
                        "decoder.mean_head.bias", "decoder.mean_head.weight"], reshaped
    for name in reshaped:
        assert feature[name].shape[0] == len(shipped_gated["target_keep_index"])
        assert raw[name].shape[0] == raw_model.raw_per_step
