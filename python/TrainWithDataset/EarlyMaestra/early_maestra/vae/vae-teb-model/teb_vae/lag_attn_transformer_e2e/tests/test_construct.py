r"""Construction invariants of the assembled model: what it refuses, what it wires, what it saves.

Each is asserted on the **assembled** model, because several hold on the parts in isolation and fail
silently in composition: the constructor's own geometry refusals, the front ends' reach budget --
which is only connected to this model's geometry if the model is what passes it -- the dropout
sites pinned at zero while the rest run at the configured rate, the geometry-shaped buffers that
must stay out of the checkpoint, and the absence of any recurrent or time-pooling module on a
history path.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_e2e.nets.frontend import NUM_STAGES
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E


#: Recurrent and time-pooling module families that must not appear on a history path. Each would
#: make $H_t$ a function of the whole sequence (a pooling one) or reintroduce the recurrent state
#: this architecture family removed (a unidirectional recurrent one, which the bitwise causality
#: probes cannot see because it is causal).
_BANNED_ON_HISTORY_PATH = (
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveMaxPool1d,
    nn.AvgPool1d,
    nn.MaxPool1d,
)


def _model(kwargs, **overrides) -> SeqVaeLagAttnTrfE2E:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfE2E(**dict(kwargs, **overrides))


def test_no_recurrent_or_time_pooling_module_on_either_history_path(tiny_kwargs):
    """Scoped to the history path -- both front ends and both encoders -- because that is where a
    statistic pooled over time would make $H_t$ read its own future, and where a recurrent layer
    would bring back the bottleneck this family removed."""
    model = _model(tiny_kwargs)
    history = {
        "target_frontend": model.target_frontend,
        "source_frontend": model.source_frontend,
        "target_encoder": model.target_encoder,
        "source_encoder": model.source_encoder,
    }
    offenders = [
        f"{stem}.{name}"
        for stem, subtree in history.items()
        for name, module in subtree.named_modules()
        if isinstance(module, _BANNED_ON_HISTORY_PATH)
    ]

    assert not offenders, f"time-pooling or recurrent modules on a history path: {offenders}"


# ---------------------------------------------------------------------------------------
# Geometry refusals
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        # The head-structured latent: group m is written only by attention head m.
        ({"d_z": 9}, r"d_z=9.*num_heads=4"),
        ({"d_head": 16}, "d_model"),
        # An empty attention window: the model would train without ever reading the source.
        ({"max_lag": -1}, "max_lag"),
        # horizon == T leaves no valid anchor.
        ({"horizon": 16}, "degenerate"),
    ],
    ids=["indivisible-latent", "head-geometry", "negative-max-lag", "degenerate-geometry"],
)
def test_an_inconsistent_geometry_is_refused_by_name(tiny_kwargs, overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, **overrides)


# ---------------------------------------------------------------------------------------
# The front ends, and the budget that bounds them
# ---------------------------------------------------------------------------------------
def test_the_model_passes_the_warmup_as_the_front_ends_reach_budget(tiny_kwargs):
    """The one assertion that connects the front end's construction-time refusal to this model's
    geometry. With no configured budget it is ``warmup_period * raw_per_step``, the raw-sample span
    of the anchors that are excluded from every loss anyway."""
    model = _model(tiny_kwargs)
    expected = model.warmup_period * model.raw_per_step

    assert model.frontend_reach_budget == expected
    for frontend in (model.target_frontend, model.source_frontend):
        assert frontend.reach_budget == expected


# ---------------------------------------------------------------------------------------
# The dropout contract, built at a nonzero dropout so it is not vacuous
# ---------------------------------------------------------------------------------------
def test_every_structurally_zero_dropout_is_zero_while_the_model_is_built_at_a_tenth(tiny_kwargs):
    """Two sites, two different reasons.

    The lag-attention probabilities: dropout is applied to the weights *before* they are returned,
    and the per-lag KL attribution is exact only if the returned weights are the ones the posterior
    consumed. The decoder subtree: one module invoked twice draws two independent masks, so base
    and full would differ at initialisation even with $z^p = z^q$.
    """
    model = _model(tiny_kwargs, dropout=0.1)

    assert model.lag_attn.attn_dropout.p == 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and name.startswith(("decoder.", "horizon_core.")):
            assert module.p == 0.0, f"{name} has dropout {module.p}"
    # And the two stacks that *should* have received the configured value did, so the assertions
    # above are about zeros that were chosen rather than about a model built at zero throughout.
    for stem in (model.target_encoder, model.target_frontend):
        probabilities = {
            module.p for module in stem.modules() if isinstance(module, nn.Dropout)
        }
        assert probabilities == {0.1}, probabilities


# ---------------------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------------------
def test_the_geometry_shaped_buffers_stay_out_of_the_state_dict(tiny_kwargs):
    """Both are constants of the architecture rather than learned state. A persistent copy would
    make a checkpoint trained at one geometry -- or at one anti-alias tap count -- fail to load at
    another, reported as keys that did not align rather than as what it is."""
    model = _model(tiny_kwargs)
    keys = list(model.state_dict())

    assert not [name for name in keys if "future_index" in name]
    assert not [name for name in keys if name.endswith("fir")]
    # ...and they do exist as buffers, so the absence above is a choice rather than an omission:
    # one fixed filter per stage per stream.
    buffers = [name for name, _ in model.named_buffers()]
    assert "future_index" in buffers
    assert len([name for name in buffers if name.endswith("fir")]) == 2 * NUM_STAGES
