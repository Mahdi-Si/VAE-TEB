r"""What the forward returns: its key set, dtypes and shapes, and what the raw grid moves.

The key set is the raw-signal architecture's plus the anchor index and its validity companion,
compared against the object that defines it rather than against a literal -- available here because
the two models are parameter-for-parameter identical on the ungated arm. The anchor keys are
*returned* rather than recomputed by every consumer because the four forecast tensors and the raw
target must be gathered at the same anchors.

The forecasts are $(B, A_{\max}, H, R)$ with $R$ the raw samples per horizon token, derived from the
constructed geometry; the per-step keys keep the step axis; and ``raw_per_step`` moves the forecast
width while leaving the step grid and the anchor set where they were.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_streams,
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
def test_the_key_set_is_the_architectures_plus_the_anchor_pair_at_their_dtypes(outputs) -> None:
    """By set equality against the architecture's own, so neither a new key nor a lost one passes.

    ``anchor_index`` gathers a raw window and scatters the KL support, so it must be ``long``;
    ``anchor_valid`` multiplies into a float mask, so it must be ``bool`` rather than a float that
    is silently truthy. Everything else is ``float32``."""
    _model, out = outputs

    assert set(out) == _architecture_keys() | set(_ANCHOR_KEYS)
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

    for key in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        assert tuple(out[key].shape) == (BATCH, a_max, model.horizon, model.geometry.r), key


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
