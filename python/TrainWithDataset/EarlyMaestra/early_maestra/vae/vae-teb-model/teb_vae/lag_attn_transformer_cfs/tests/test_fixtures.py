r"""The splice: the two halves of this suite's conftest must agree where they meet.

The data half -- the committed causal shard, the configuration builder, the tiny warm-up staircase,
the stub batch carrying the two phase-key fields, the seeded input streams -- is the conv-LSTM
causal cell's, because it describes the *dataset* and the *target domain*. The constructor keyword
sets are written locally at the conv-Transformer schema, because the causal cell's would not
construct: they carry five keywords this constructor refuses.

The two halves meet at :data:`~.conftest.SHARED_GEOMETRY_KEYS`. The imported batch machinery, the
imported budget resolution and every anchor count this suite asserts close over those values while
every model here is built from the local sets, so the splice is sound only while the two agree --
and a disagreement builds a model neither parent's suite tests, with no shape differing, because
$A_{\max}$ and the block width are geometry constants either way.
"""
from __future__ import annotations

from teb_vae.lag_attn_cfs.tests import conftest as causal_conftest

from . import conftest as local

#: The seven keys this architecture adds, all of them describing the encoders being swapped in.
_ENCODER_KEYS = {
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
}


def test_the_two_shipped_sets_agree_everywhere_but_the_encoder():
    """Every shared geometry key is declared on both sides, and every key both sides declare agrees
    in value. Written out independently on both sides, so this is a real comparison rather than a
    tautology over one literal read twice.

    The other direction too: the splice would also be broken by two sets that agreed on
    *everything*, because then this package would be testing the conv-LSTM cell. Directional rather
    than by set difference, because two keyword sets may also differ by a key one of them states at
    the constructor's own default -- an inert declaration, not an architecture difference.
    """
    theirs = causal_conftest.shipped_warmup_kwargs()
    mine = local.shipped_warmup_kwargs()

    assert set(local.SHARED_GEOMETRY_KEYS) <= set(mine) & set(theirs)
    differing = [key for key in set(mine) & set(theirs) if mine[key] != theirs[key]]
    assert differing == [], differing

    assert _ENCODER_KEYS <= set(mine)
    assert _ENCODER_KEYS.isdisjoint(theirs)
    assert set(local.CONV_LSTM_ONLY_KEYS).isdisjoint(mine)
    assert set(local.CONV_LSTM_ONLY_KEYS) & set(theirs)


def test_the_tiny_sets_agree_on_every_shared_geometry_key_that_both_declare():
    """The same check at the tiny geometry, where the imported ``make_stub_batch`` and
    ``make_streams`` close over the values directly."""
    theirs = causal_conftest.TINY_KWARGS
    both = [
        key for key in local.SHARED_GEOMETRY_KEYS if key in theirs and key in local.TINY_KWARGS
    ]

    assert both, "no shared geometry key is declared in both tiny sets"
    differing = {key: (local.TINY_KWARGS[key], theirs[key]) for key in both
                 if local.TINY_KWARGS[key] != theirs[key]}
    assert differing == {}, differing
