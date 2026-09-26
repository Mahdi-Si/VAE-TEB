r"""The tiled anchor set as this model decodes and scores it.

The index semantics -- which anchors a phase selects, the padding convention and the refusals -- are
those of ``_build_anchor_index``, inherited unchanged from the causal-feature cell and tested there.
What is checked here is what this composition does with the set: the forward returns the anchors it
decoded, the decoder is invoked on the latents gathered at them, the tiling does not change how much
randomness a forward consumes, and a padded slot -- a repeat of its row's last valid anchor, which
on this cell gathers a second copy of a **raw** window -- contributes exactly zero to the loss.
"""
from __future__ import annotations

import pytest
import torch

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    tiny_warmup_kwargs,
)

#: The forward keys carrying the anchor axis, which a trimmed anchor set must be sliced along
#: together. Named once so the padded-slot identity below cannot silently trim only some of them.
_ANCHOR_AXIS_KEYS = (
    "mu_base",
    "logvar_base",
    "mu_full",
    "logvar_full",
    "anchor_index",
    "anchor_valid",
)


def _kwargs() -> dict:
    """The tiny guarded keyword set at a real tiling."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


@pytest.fixture(scope="module")
def model():
    """The tiny guarded model, built once: nothing below trains and nothing below mutates it."""
    return build(_kwargs()).eval()


# =================================================================================================
# What a padded slot costs
# =================================================================================================
def test_a_padded_slot_contributes_exactly_zero_to_the_loss(model) -> None:
    """The property the padding convention exists for, as a loss identity rather than a mask shape.

    A padded slot's gathered window *is* its row's last valid window -- that is what repeating the
    index means -- so the only thing standing between it and a doubly-scored raw block is the mask.
    Scoring the same forward with the padding trimmed away must therefore reproduce every
    reconstruction readout: a padded slot that leaked would show as a block NLL inflated by roughly
    one anchor in four against an unchanged anchor count, which no shape would report.

    Compared to a relative tolerance rather than bitwise, and the reason is the comparison rather
    than the claim. A padded slot contributes exactly zero *terms*, but the two sums are reductions
    over differently-shaped tensors, so they accumulate in a different order; the observed gap is
    $2 \\times 10^{-7}$ relative against a leak of $0.33$.
    """
    stride = model.anchor_stride
    kwargs = _kwargs()
    signal = make_raw_signal(kwargs)
    weight = torch.ones(BATCH, model.geometry.t)

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*make_streams(kwargs), stride - 1)
    valid = out["anchor_valid"]
    assert bool((~valid).any()), "no padded slot at the widest phase; the identity is vacuous"

    count = int(valid[0].sum())
    assert bool((valid.sum(dim=1) == count).all()), "a scalar phase gives every row one count"
    trimmed = {
        name: (value[:, :count] if name in _ANCHOR_AXIS_KEYS else value)
        for name, value in out.items()
    }
    assert bool(trimmed["anchor_valid"].all())

    padded_metrics = model.compute_loss(out, signal, weight=weight, likelihood="mse")["metrics"]
    trimmed_metrics = model.compute_loss(
        trimmed, signal, weight=weight, likelihood="mse"
    )["metrics"]

    for name in ("nll_base_block", "nll_full_block", "source_conditioned_kl_raw"):
        assert float(padded_metrics[name]) == pytest.approx(
            float(trimmed_metrics[name]), rel=1e-5, abs=1e-6
        ), name
    assert float(padded_metrics["anchors_per_sample"]) == float(count)


def test_a_forward_consumes_the_same_randomness_at_every_phase(model) -> None:
    """The end-to-end form. Note what is *not* claimed: the forward draws nothing.
    ``_reparameterize_shared`` calls ``randn_like`` unconditionally, so the claim is that the tiling
    does not change how much randomness is consumed."""
    stride = model.anchor_stride
    streams = make_streams(_kwargs())

    states = []
    for phase in range(stride):
        torch.manual_seed(0)
        with torch.no_grad():
            model(*streams, phase)
        states.append(torch.random.get_rng_state())

    assert all(torch.equal(states[0], state) for state in states[1:])


def test_the_decoder_is_invoked_on_the_gathered_latents(model) -> None:
    """Not on the contiguous prefix: the anchors are no longer one.

    Read at the decoder's own input, so this is what the module received rather than what the
    forward meant to hand it.
    """
    streams = make_streams(_kwargs())
    seen: list = []
    handle = model.decoder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(*streams, 1)
    finally:
        handle.remove()

    index = out["anchor_index"]
    assert len(seen) == 2
    for latent, key in zip(seen, ("z_prior", "z_post")):
        expected = out[key].gather(1, index[:, :, None].expand(-1, -1, model.d_z))
        assert torch.equal(latent, expected), key
    assert tuple(out["mu_base"].shape)[:2] == tuple(index.shape)


def test_the_forward_returns_the_anchors_it_decoded(model) -> None:
    """Returned rather than recomputed, so the objective and the figures cannot disagree with it --
    which on this cell is the difference between one raw window and another."""
    streams = make_streams(_kwargs())
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*streams, 2)

    index, valid = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_phase=2
    )
    assert torch.equal(out["anchor_index"], index)
    assert torch.equal(out["anchor_valid"], valid)
