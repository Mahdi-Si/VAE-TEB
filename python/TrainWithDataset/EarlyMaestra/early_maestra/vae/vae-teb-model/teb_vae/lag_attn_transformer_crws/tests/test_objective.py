r"""The objective's metric surface, as this composition reaches it.

The arithmetic is not tested here. ``lag_attn_rws/nets/losses.py`` owns every term, every reduction
and every reported metric, and its own suite pins them; ``compute_loss`` with the anchored gather,
the block width, the anchor denominator and the padding mask is ``lag_attn_crws``'s input mixin's,
reached here as the same code object, and that suite pins it. What this *composition* can still get
wrong is which ``compute_loss`` the base order resolves to: the architecture parent's dense objective
would score every anchor and report a different metric set.

So the metric surface is asserted exact in both directions against the conv-LSTM cell of this row's,
because the two are read side by side and a name in one and not the other is a column that silently
empties.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    tiny_warmup_kwargs,
)

#: What this model's ``compute_loss`` adds to the shared objective's metric dict: the three that
#: need the anchor set or the source warm-up, and none of the five that partition kept *target*
#: channels, because this block's last axis counts raw samples.
_ADDED_METRIC_KEYS = {
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
}

#: Two phases one anchor apart in count, so the tiled forward carries a padded slot.
_PHASES = (0, TINY_STRIDE - 1)


def _metric_names(model, kwargs) -> set:
    """The metric keys of one tiled forward's ``compute_loss``, at uniform validity."""
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.eval()(*make_streams(kwargs), torch.tensor(_PHASES), TINY_STRIDE)
    weight = torch.ones(BATCH, model.geometry.t)
    return set(model.compute_loss(out, make_raw_signal(kwargs), weight=weight)["metrics"])


def test_the_metric_key_set_is_the_conv_lstm_cells_plus_nothing() -> None:
    """Exact in both directions against the conv-LSTM cell of this row, and carrying the input
    mixin's own readouts -- which the architecture parent's dense objective does not emit."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    mine = _metric_names(build(kwargs), kwargs)

    conv_kwargs = conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    torch.manual_seed(0)
    theirs = _metric_names(SeqVaeLagAttnCrws(**conv_kwargs), conv_kwargs)

    assert mine == theirs
    assert _ADDED_METRIC_KEYS <= mine
