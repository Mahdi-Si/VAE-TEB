r"""What the forward returns, and what its two extra arguments changed.

Twenty-two keys: the raw-signal architecture's twenty, plus the anchor index and its validity
companion. They are *returned* rather than recomputed by every consumer because the four forecast
tensors and the raw target must be gathered at the same anchors -- and a second computation could
disagree, which on this cell means one raw window scored against another with every shape correct.

The comparison arm is the architecture itself at the same keywords, which is available here and is
**not** available to the causal-feature cells: their decoder emits $C_{\mathrm{keep}}$ per horizon
token against the raw model's $R$, so a differently-shaped head consumes a different amount of the
initialisation stream and no key-set comparison between them is a comparison of one contract. Here
the two models are parameter-for-parameter identical, so the key set is compared against the object
that defines it rather than against a literal.

The forecast shapes are derived from the constructed geometry rather than written out. The one thing
worth stating in numbers is what they resolve to: $(B, 11, 15, 16)$ at the shipped configuration --
eleven tiles, fifteen horizon steps, sixteen raw samples per step -- which is a $240$-sample raw
block against ``lag_attn_rws``'s $480$, and about $10.1$ decoded anchors per step against its $240$.
"""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import (
    BATCH,
    SHIPPED_HORIZON,
    TINY_HORIZON,
    TINY_STRIDE,
    build,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: The two keys this model adds to the architecture's contract.
_ANCHOR_KEYS = ("anchor_index", "anchor_valid")


def _kwargs() -> dict:
    """The tiny guarded keyword set at a real tiling."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


def _a_max(model) -> int:
    r"""$A_{\max}$, from the constructed geometry rather than from the returned tensor."""
    return math.ceil((model.geometry.t_valid - model.warmup_period) / model.anchor_stride)


def _architecture_keys() -> set:
    """The twenty keys the raw-signal architecture returns at the same ungated keywords."""
    kwargs = {
        name: value
        for name, value in _kwargs().items()
        if name
        not in (
            "target_keep_index",
            "target_warmup_steps",
            "source_keep_index",
            "source_warmup_steps",
            "anchor_stride",
        )
    }
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**kwargs).eval()
    with torch.no_grad():
        return set(model(*make_streams(kwargs)))


@pytest.fixture(scope="module")
def outputs():
    """One forward of the tiny guarded model at a real tiling."""
    kwargs = _kwargs()
    model = build(kwargs).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        return model, model(*make_streams(kwargs), 1)


# =================================================================================================
# The key set
# =================================================================================================
def test_the_forward_returns_exactly_twenty_two_keys(outputs) -> None:
    """By set equality against the architecture's own, so neither a new key nor a lost one passes."""
    _model, out = outputs

    assert len(out) == 22
    assert set(out) == _architecture_keys() | set(_ANCHOR_KEYS)


def test_the_pathways_this_architecture_does_not_have_are_absent(outputs) -> None:
    """No ``decoder_state`` and no ``delta_mu_src``: the decoder receives the latent and nothing
    else, so there is no bypass to report and no source term added around it."""
    _model, out = outputs

    assert "decoder_state" not in out
    assert "delta_mu_src" not in out


def test_the_anchor_keys_carry_the_dtypes_their_consumers_index_with(outputs) -> None:
    """``anchor_index`` gathers a raw window and scatters the KL support, so it must be ``long``;
    ``anchor_valid`` multiplies into a float mask, so it must be ``bool`` rather than a float that
    is silently truthy."""
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
def test_the_forecasts_carry_the_anchor_axis_and_the_raw_block_width(outputs) -> None:
    r"""$(B, A_{\max}, H, R)$, with $R$ the raw samples per horizon token and **not**
    $C_{\mathrm{keep}}$ -- which is the whole difference between this cell and the causal-feature
    one, and the thing the excluded width hook would have changed."""
    model, out = outputs
    a_max = _a_max(model)

    assert model.target_gate is not None, "the guarded model gates its target stream"
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == (BATCH, a_max, model.horizon, model.geometry.r), key
    # The tiny geometry resolves that to four tiles of four steps of sixteen raw samples.
    assert (a_max, model.horizon, model.geometry.r) == (4, TINY_HORIZON, 16)


def test_the_per_step_keys_keep_the_step_axis(outputs) -> None:
    """Only the anchor axis is sparse. The latent, the states and the KL are produced at every step
    regardless of which anchors were decoded, which is why the KL support has to scatter."""
    model, out = outputs
    length = model.sequence_length

    for key in ("mu_prior", "logvar_prior", "mu_post", "logvar_post", "z_prior", "z_post"):
        assert tuple(out[key].shape) == (BATCH, length, model.d_z), key
    for key in ("target_state", "source_state"):
        assert tuple(out[key].shape) == (BATCH, length, model.d_model), key
    assert tuple(out["kld_per_t"].shape) == (BATCH, length)
    assert tuple(out["source_kl_lag_map"].shape) == (BATCH, length, model.lag_attn.L)


def test_the_shipped_geometry_forecasts_eleven_tiles_of_a_minute() -> None:
    r"""The shipped forecast shape: $(B, 11, 15, 16)$, from the budget the committed shard resolves.

    Every factor is derived: eleven is $\lceil 152/15 \rceil$, fifteen is the configured horizon,
    and sixteen is ``raw_per_step``. What is asserted as a literal is only that they resolve to
    those numbers, because that is the configuration every reported nat is produced at.
    """
    kwargs = shipped_warmup_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**kwargs).eval()

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*make_streams(kwargs), torch.tensor([0, 7]))

    expected = (BATCH, _a_max(model), model.horizon, model.geometry.r)
    assert expected == (BATCH, 11, SHIPPED_HORIZON, 16)
    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == expected, key


# =================================================================================================
# The signature and the arity change
# =================================================================================================
def test_the_signature_is_the_three_streams_and_the_two_anchor_arguments() -> None:
    """A literal list, as in every sibling's copy: the anchor geometry is an argument rather than
    something derived from ``self.training``, and the parameter list is where that decision is
    visible.

    Derived from the mode instead, ``total_loss`` would become a function of the dropout switch --
    the diagnostic callback calls ``eval()`` during training and then the objective.
    """
    assert list(inspect.signature(SeqVaeLagAttnCrws.forward).parameters) == [
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


# =================================================================================================
# The raw grid is no longer inert
# =================================================================================================
def test_the_raw_grid_moves_the_forecast_width(tiny_warmup) -> None:
    r"""The mirror of the causal-feature cells' claim, and it comes out the other way.

    There the block's last axis counts channels, so ``raw_per_step`` moves no forecast shape at all.
    Here it *is* the block width, while $T = \texttt{raw\_len} / \texttt{raw\_per\_step}$ stays
    invariant under it -- so halving it halves the raw block and leaves the anchor set untouched.
    """
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

    assert torch.equal(first["anchor_index"], second["anchor_index"])
    assert first["mu_base"].shape[:3] == second["mu_base"].shape[:3]
    assert first["mu_base"].shape[3] == 16 and second["mu_base"].shape[3] == 8
