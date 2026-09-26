r"""The splice: the two halves of the conftest must agree on the geometry they share.

This suite's input and target half -- the committed causal shard, the configuration builder, the
tiny warm-up staircase, the stub batch, the seeded input streams and the seeded raw signal -- is the
conv-LSTM causal-input cell's, because it describes the *dataset*, the *target domain* and the
*anchor geometry*. The constructor keyword sets are written locally at the conv-Transformer schema,
because the conv-LSTM cell's carries five keywords this constructor refuses.

The two halves meet at :data:`~.conftest.SHARED_GEOMETRY_KEYS`, and here that seam is load-bearing:
the two conftests are independently maintained, so nothing but this file makes them agree. The
imported batch machinery, the imported budget resolution, the imported raw-signal builder and every
anchor count this suite asserts close over the conv-LSTM cell's values while every model here is
built from the local sets -- and a disagreement builds a model neither parent's suite tests, with no
shape differing, because $A_{\max}$ and the raw block width are geometry constants either way.
"""
from __future__ import annotations

from teb_vae.lag_attn_crws.tests import conftest as causal_conftest

from . import conftest as local

#: The seven keys that are the encoder edge between this cell and the conv-LSTM one. Written out
#: rather than derived from the difference between the two sets, because the difference is what the
#: test below is measuring.
_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)


def test_the_two_shipped_sets_differ_only_in_the_encoder():
    """Every key both production sets declare must agree in value, or the encoder edge is not the
    only edge -- and the splice would also be broken by two sets that agreed on *everything*,
    because then this package would be testing the conv-LSTM cell."""
    theirs = causal_conftest.shipped_warmup_kwargs()
    mine = local.shipped_warmup_kwargs()

    encoder_keys = set(_ENCODER_KEYS)
    # Directional rather than by set difference. Two keyword sets may also differ by a key one of
    # them states at the constructor's own default -- an inert declaration, not an architecture
    # difference -- and pinning the difference by equality would make this test fail on those.
    assert encoder_keys <= set(mine)
    assert encoder_keys.isdisjoint(theirs)
    assert set(local.CONV_LSTM_ONLY_KEYS).isdisjoint(mine)
    assert set(local.CONV_LSTM_ONLY_KEYS) & set(theirs)
    assert set(local.SHARED_GEOMETRY_KEYS) <= set(mine) & set(theirs)

    differing = [key for key in set(mine) & set(theirs) if mine[key] != theirs[key]]
    assert differing == [], differing


def test_the_tiny_sets_agree_on_every_shared_geometry_key_both_declare():
    """The same check at the tiny geometry, where the imported ``make_stub_batch``,
    ``make_streams`` and ``make_raw_signal`` close over the values directly."""
    theirs = causal_conftest.TINY_KWARGS
    shared = [
        key
        for key in local.SHARED_GEOMETRY_KEYS
        if key in theirs and key in local.TINY_KWARGS
    ]

    assert shared, "the two tiny sets declare no geometry key in common; the check is vacuous"
    for key in shared:
        assert local.TINY_KWARGS[key] == theirs[key], key
