r"""What the forward returns, at this architecture, and that it is the causal-input cell's contract.

The architecture's keys plus the anchor index and its validity companion. This
package needs a forward-contract module where the conv-Transformer raw-signal parent does not, and
the reason is where the forward lives. That cell's forward is its own code object, pinned by its own
suite; this one's is the *causal-input mixin's*, so what has to be shown here is that composing the
mixin over a different architecture leaves the contract intact -- same key set, same dtypes, same
shapes, same arity.

The comparison is made against the conv-LSTM cell of this row rather than against a written-out
literal, so a change to the shared forward moves both sides at once instead of failing a constant
here.

The one shape this cell pins against the causal-*feature* cells rather than with them is the last
axis of the four forecast tensors: $R$ raw samples per horizon token, not the target gate's
surviving-channel count. That is the whole content of "no width hook is defined", read off a real
forward rather than off the constructor.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.tests.conftest import (
    TINY_KWARGS as CONV_LSTM_TINY_KWARGS,
)
from teb_vae.lag_attn_crws.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_streams,
    tiny_warmup_kwargs,
)

#: The two keys this input domain adds to the architecture's contract.
_ANCHOR_KEYS = ("anchor_index", "anchor_valid")


def _architecture_keys() -> set:
    """The keys the bare conv-Transformer architecture returns at this geometry."""
    kwargs = {
        name: value
        for name, value in tiny_warmup_kwargs().items()
        if name not in ("target_warmup_steps", "source_warmup_steps")
    }
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(
        **dict(kwargs, target_delays=None, source_delays=None)
    ).eval()
    with torch.no_grad():
        return set(model(*make_streams(kwargs)))


def _conv_lstm_keys() -> set:
    """And the keys the conv-LSTM cell of this row returns, which must be the same set."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**conv_lstm_tiny_warmup_kwargs()).eval()
    with torch.no_grad():
        return set(model(*make_streams(CONV_LSTM_TINY_KWARGS)))


@pytest.fixture(scope="module")
def outputs():
    """One forward of the tiny guarded model at a real tiling."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    model = build(kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*make_streams(kwargs), 0, TINY_STRIDE)


# =================================================================================================
# The key set
# =================================================================================================
def test_the_forward_returns_the_architectures_keys_plus_the_anchor_set(outputs) -> None:
    """By set equality against both neighbours in the grid, so neither a new key nor a lost one
    passes -- and so that a change to the shared forward fails on the change rather than here."""
    _model, out = outputs

    assert set(out) == _architecture_keys() | set(_ANCHOR_KEYS)
    assert set(out) == _conv_lstm_keys()


def test_the_anchor_keys_carry_the_dtypes_their_consumers_index_with(outputs) -> None:
    """``anchor_index`` gathers and scatters, so it must be ``long``; ``anchor_valid`` multiplies
    into a float mask, so it must be ``bool`` rather than a float that is silently truthy."""
    _model, out = outputs

    assert out["anchor_index"].dtype == torch.long
    assert out["anchor_valid"].dtype == torch.bool
    for key, value in out.items():
        if key in _ANCHOR_KEYS:
            continue
        assert value.dtype == torch.float32, key


# =================================================================================================
# Shapes
# =================================================================================================
def test_the_forecasts_carry_the_anchor_axis_and_the_raw_grid(outputs) -> None:
    r"""$(B, A_{\max}, H, R)$ with $A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$ derived from
    the geometry rather than written out -- and the last axis is the raw decimation, **not** the
    target gate's surviving-channel count. The gate is present and reaches the input adapters alone;
    a forecast whose last axis followed it would be the causal-feature cells' block."""
    model, out = outputs
    span = model.geometry.t_valid - model.warmup_period
    a_max = -(-span // TINY_STRIDE)
    assert model.target_gate is not None

    assert int(out["anchor_index"].shape[1]) == a_max
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == (BATCH, a_max, model.horizon, model.raw_per_step), key
    assert model.target_gate.out_channels != model.raw_per_step


def test_the_ungated_model_keeps_the_same_forecast_width(tiny_kwargs) -> None:
    """The other half of the width claim, and the half that separates this cell from the
    causal-feature ones: with no budget at all the forecast is still $R$ wide, because the raw block
    is geometry rather than a gate's survivor count."""
    model = build(tiny_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*make_streams(tiny_kwargs))

    a_max = int(out["anchor_index"].shape[1])
    assert model.target_gate is None
    assert tuple(out["mu_base"].shape) == (BATCH, a_max, model.horizon, model.raw_per_step)


def test_the_per_step_keys_keep_the_step_axis(outputs) -> None:
    """Only the anchor axis is sparse. The latent, the states and the KL are produced at every
    step regardless of which anchors were decoded, which is why the KL support has to scatter."""
    model, out = outputs
    length = model.sequence_length

    for key in ("mu_prior", "logvar_prior", "mu_post", "logvar_post", "z_prior", "z_post"):
        assert tuple(out[key].shape) == (BATCH, length, model.d_z), key
    for key in ("target_state", "source_state"):
        assert tuple(out[key].shape) == (BATCH, length, model.d_model), key
    assert tuple(out["kld_per_t"].shape) == (BATCH, length)
    assert tuple(out["source_kl_lag_map"].shape) == (BATCH, length, model.lag_attn.L)


def test_the_two_cells_of_this_row_agree_shape_for_shape(outputs) -> None:
    """The grid's premise on the forward: at the same tiny geometry the two cells differ in the
    *values* their encoders produce and in nothing about the contract."""
    _model, out = outputs
    kwargs = conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    torch.manual_seed(0)
    conv_lstm = SeqVaeLagAttnCrws(**kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        theirs = conv_lstm(*make_streams(CONV_LSTM_TINY_KWARGS), 0, TINY_STRIDE)

    for key in out:
        assert out[key].shape == theirs[key].shape, key
    # And the anchor set itself is identical, because it is a geometry constant either side.
    assert torch.equal(out["anchor_index"], theirs["anchor_index"])
    assert torch.equal(out["anchor_valid"], theirs["anchor_valid"])


# =================================================================================================
# The three-tensor call
# =================================================================================================
def test_the_three_tensor_call_still_works_and_agrees_with_a_zero_phase(tiny_warmup) -> None:
    """At the inert default stride both are the dense range, and they agree bitwise."""
    model = build(tiny_warmup).eval()
    streams = make_streams(tiny_warmup)

    torch.manual_seed(0)
    with torch.no_grad():
        implicit = model(*streams)
    torch.manual_seed(0)
    with torch.no_grad():
        explicit = model(*streams, 0, 1)

    assert set(implicit) == set(explicit)
    for key in implicit:
        assert torch.equal(implicit[key], explicit[key]), key
