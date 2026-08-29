r"""What the forward returns, and what its two extra arguments changed downstream.

Twenty-two keys: the base's twenty, plus the anchor index and its validity companion. They are
*returned* rather than recomputed by every consumer because the four forecast tensors and the target
must be gathered at the same anchors, and a second computation could disagree -- which would be a
wrong number rather than an exception.

The arity change is the part with a blast radius, and it is small and known: everything in the tree
that calls a model's forward does so star-splat or by index, with exactly one exception, and this
file pins the inventory rather than the replacement. Replacing that one consumer is a later task's,
and the reason it needs one is that its failure is *silent* -- it raises inside a handler that warns
and continues, so a missing seam costs two page rows and a figure with a green suite.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    CAUSAL_C_Y,
    TINY_STRIDE,
    build,
    make_streams,
    tiny_warmup_kwargs,
)
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs

#: The two keys this model adds to the base's contract.
_ANCHOR_KEYS = ("anchor_index", "anchor_valid")

#: The base's twenty, taken from the sibling that shares them rather than written out, so a change
#: to the shared forward moves both sides of the comparison instead of failing a literal here.
_SIBLING_KWARGS = dict(tiny_warmup_kwargs())


def _sibling_keys() -> set:
    """The twenty keys the two-sided feature sibling returns at the same geometry."""
    kwargs = {
        name: value
        for name, value in _SIBLING_KWARGS.items()
        if name not in ("target_warmup_steps", "source_warmup_steps")
    }
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(kwargs, target_delays=None, source_delays=None)).eval()
    with torch.no_grad():
        return set(model(*make_streams(kwargs)))


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
    """By set equality against the sibling's, so neither a new key nor a lost one passes."""
    _model, out = outputs
    assert len(out) == 22
    assert set(out) == _sibling_keys() | set(_ANCHOR_KEYS)


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


# =================================================================================================
# The signature and its one arity-sensitive consumer
# =================================================================================================
def test_the_signature_is_the_three_streams_and_the_two_anchor_arguments() -> None:
    """A literal list, as in every sibling's invariants file: the anchor geometry is an argument
    rather than something derived from ``self.training``, and the parameter list is where that
    decision is visible."""
    assert list(inspect.signature(SeqVaeLagAttnCfs.forward).parameters) == [
        "self",
        "y_st",
        "y_ph",
        "u_stream",
        "anchor_phase",
        "anchor_stride",
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


def test_the_one_arity_sensitive_consumer_is_the_input_budget_panel() -> None:
    """The inventory: this is the only consumer an arity change reaches at all.

    ``stream_panels`` refuses anything but a three-tensor call and then unpacks exactly three, and
    it raises *inside* a handler that warns and continues -- so reaching it on a five-argument
    forward costs two page rows and one figure with nothing failing. This package's own builder is
    what the callback resolves instead, and the refusal below is what keeps that a requirement
    rather than a preference. Everything else that calls a forward in this tree is star-splat or
    index-``[0]``, which no arity change touches.
    """
    from teb_vae.lag_attn_rws import input_budget

    source = Path(input_budget.__file__).read_text(encoding="utf-8")
    assert "len(forward_inputs) != 3" in source

    with pytest.raises(ValueError, match="got 5 tensors"):
        input_budget.stream_panels(model=None, forward_inputs=(1, 2, 3, 4, 5))


# =================================================================================================
# The raw grid is inert
# =================================================================================================
def test_the_raw_grid_moves_no_forecast_shape(tiny_warmup) -> None:
    r"""``raw_per_step`` remains a required geometry input -- ``raw_len`` is built from it -- but
    $T = \texttt{raw\_len} / \texttt{raw\_per\_step}$ is invariant under it, and this target's last
    axis is a channel count. So halving it moves nothing a forecast is measured against."""
    coarse = build(tiny_warmup).eval()
    fine = build(dict(tiny_warmup, raw_per_step=8)).eval()
    assert coarse.geometry.t == fine.geometry.t
    assert coarse.geometry.t_valid == fine.geometry.t_valid

    streams = make_streams(tiny_warmup)
    torch.manual_seed(0)
    with torch.no_grad():
        first = coarse(*streams)
    torch.manual_seed(0)
    with torch.no_grad():
        second = fine(*streams)

    assert first["mu_base"].shape == second["mu_base"].shape
    assert torch.equal(first["anchor_index"], second["anchor_index"])


# =================================================================================================
# The one key the persistence residual adds
#
# `persistence` is the target's own stored value at each anchor, and it is on the forward dict for a
# reason the shape alone does not say: three consumers need the SAME tensor. Both decoder calls take
# it, or the base-minus-full gap stops being a pure source readout; and the permutation control's
# re-decode takes it, or the shuffle gap shifts for a reason that has nothing to do with the source.
# Recomputing it at each site would be three chances to gather a different row.
# =================================================================================================
def test_the_forward_returns_no_persistence_key_when_the_residual_is_off(outputs) -> None:
    """The off-state on the contract itself. A key present and ``None`` would be indistinguishable
    from a mechanism that ran and produced nothing, and every consumer would have to test for it."""
    _model, out = outputs

    assert "persistence" not in out


def test_the_persistence_key_is_the_targets_own_value_at_each_anchor() -> None:
    r"""$(B, A_{\max}, C_{\mathrm{keep}})$, gathered from the TARGET stream on the kept axis.

    Target-only is the invariant: the residual adds $w_{	au,c}\, y_{t,c}$ to the decoder mean, and
    if $y_t$ carried any source content the no-bypass argument would fail and ``pred_gap`` would
    stop being a source readout. Asserted against a hand-built gather of the target features, so a
    tensor that had come to be built from anything else fails rather than merely being the right
    shape.
    """
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, persistence_residual=True)
    model = build(kwargs).eval()
    y_st, y_ph, u_stream = make_streams(kwargs)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(y_st, y_ph, u_stream, 1)

    features = torch.cat([y_st, y_ph], dim=-1)
    keep = model.target_gate.keep_index.to(torch.long)
    anchors = out["anchor_index"]
    expected = features[:, :, keep].gather(
        1, anchors[:, :, None].expand(-1, -1, keep.numel())
    )

    assert out["persistence"].shape == (BATCH, anchors.shape[1], keep.numel())
    assert torch.equal(out["persistence"], expected)
