r"""What the forward returns, at this architecture, and that it is the causal cell's contract.

Twenty-two keys: the architecture's twenty, plus the anchor index and its validity companion. This
package needs a forward-contract module where the two-sided conv-Transformer cell does not, and the
reason is where the forward lives. That cell's forward is its architecture parent's own code object,
pinned by that parent's own suite; this one's is the *causal mixin's*, so what has to be shown here
is that composing the mixin over a different architecture leaves the contract intact -- same key
set, same dtypes, same shapes, same arity.

The comparison is made against the conv-LSTM causal cell rather than against a written-out literal,
so a change to the shared forward moves both sides at once instead of failing a constant here.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    TINY_KWARGS as CONV_LSTM_TINY_KWARGS,
)
from teb_vae.lag_attn_cfs.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import (
    BATCH,
    CAUSAL_C_Y,
    TINY_STRIDE,
    build,
    make_streams,
    tiny_warmup_kwargs,
)

#: The two keys this target domain adds to the architecture's contract.
_ANCHOR_KEYS = ("anchor_index", "anchor_valid")


def _architecture_keys() -> set:
    """The twenty keys the bare conv-Transformer architecture returns at this geometry."""
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
    """And the twenty-two the conv-LSTM causal cell returns, which must be the same set."""
    kwargs = conv_lstm_tiny_warmup_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnCfs(**kwargs).eval()
    with torch.no_grad():
        return set(model(*make_streams(CONV_LSTM_TINY_KWARGS)))


@pytest.fixture(scope="module")
def outputs():
    """One forward of the tiny guarded model at a real tiling."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    model = build(kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*make_streams(kwargs), 1)


# =================================================================================================
# The key set
# =================================================================================================
def test_the_forward_returns_exactly_twenty_two_keys(outputs) -> None:
    """By set equality against both neighbours in the grid, so neither a new key nor a lost one
    passes -- and so that a change to the shared forward fails on the change rather than here."""
    _model, out = outputs

    assert len(out) == 22
    assert set(out) == _architecture_keys() | set(_ANCHOR_KEYS)
    assert set(out) == _conv_lstm_keys()


def test_the_pathways_this_architecture_does_not_have_are_absent(outputs) -> None:
    """No ``decoder_state`` and no ``delta_mu_src``: the decoder receives the latent and nothing
    else, so there is no bypass to report and no source term added around it."""
    _model, out = outputs

    assert "decoder_state" not in out
    assert "delta_mu_src" not in out


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
def test_the_forecasts_carry_the_anchor_axis_and_the_kept_width(outputs) -> None:
    r"""$(B, A_{\max}, H, C_{\mathrm{keep}})$, with the width the budget resolved and not $c_y$."""
    model, out = outputs
    a_max = int(out["anchor_index"].shape[1])
    assert model.target_gate is not None
    kept = model.target_gate.out_channels
    assert kept < CAUSAL_C_Y

    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == (BATCH, a_max, model.horizon, kept), key


def test_the_ungated_model_forecasts_every_declared_channel(tiny_kwargs) -> None:
    """The other half of the width claim: with no budget the decoder emits all $c_y$."""
    model = build(tiny_kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*make_streams(tiny_kwargs))

    a_max = int(out["anchor_index"].shape[1])
    assert tuple(out["mu_base"].shape) == (BATCH, a_max, model.horizon, CAUSAL_C_Y)


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


def test_the_two_causal_cells_agree_shape_for_shape(outputs) -> None:
    """The grid's premise on the forward: at the same tiny geometry the two cells differ in the
    *values* their encoders produce and in nothing about the contract."""
    _model, out = outputs
    kwargs = conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    torch.manual_seed(0)
    conv_lstm = SeqVaeLagAttnCfs(**kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        theirs = conv_lstm(*make_streams(CONV_LSTM_TINY_KWARGS), 1)

    for key in out:
        assert out[key].shape == theirs[key].shape, key
    # And the anchor set itself is identical, because it is a geometry constant either side.
    assert torch.equal(out["anchor_index"], theirs["anchor_index"])
    assert torch.equal(out["anchor_valid"], theirs["anchor_valid"])


# =================================================================================================
# The signature
# =================================================================================================
def test_the_signature_is_the_three_streams_and_the_two_anchor_arguments() -> None:
    """A literal list, as in every sibling's invariants file: the anchor geometry is an argument
    rather than something derived from ``self.training``, and the parameter list is where that
    decision is visible."""
    assert list(inspect.signature(SeqVaeLagAttnTrfCfs.forward).parameters) == [
        "self",
        "y_st",
        "y_ph",
        "u_stream",
        "anchor_phase",
        "anchor_stride",
    ]
    # The architecture parent's is three, which is what the mixin extends -- and it is the reason
    # the one arity-sensitive consumer in the tree needs a replacement here.
    assert list(inspect.signature(SeqVaeLagAttnTrfRws.forward).parameters) == [
        "self",
        "y_st",
        "y_ph",
        "u_stream",
    ]


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


def test_a_missing_phase_is_refused_at_a_real_stride(tiny_warmup) -> None:
    """Not defaulted: a forgotten phase would train every sample of every epoch on one tile grid at
    a fixed offset from the segment start, and $A_{\\max}$ is a geometry constant either way, so
    nothing about the shapes would say so."""
    model = build(tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()

    with pytest.raises(ValueError, match="anchor_phase is required"):
        model(*make_streams(tiny_warmup))


def test_a_non_zero_phase_is_refused_at_stride_one(tiny_warmup) -> None:
    """The anchor set truncates rather than rotating there, so a phase would silently drop leading
    anchors and ``anchors_per_sample`` would read $152 - \\varphi$."""
    model = build(tiny_warmup).eval()

    with pytest.raises(ValueError, match=r"outside \[0, anchor_stride\)"):
        model(*make_streams(tiny_warmup), 1, 1)
