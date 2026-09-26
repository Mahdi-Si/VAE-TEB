r"""The anchored raw gather: the one genuinely new piece of arithmetic in this cell.

:func:`~teb_vae.lag_attn_rws.nets.raw_targets.build_future_target` takes no anchor set -- no
raw-target sibling tiled -- so there are two expressions of the same arithmetic in the repository,
and the test that matters pins them against each other: **at the dense anchor set the anchored path
must equal the dense builder elementwise**. The rest guards the ways the second expression could be
wrong while every shape stayed right.

**Why a ``gather`` and not an ``index_select``.** The anchor set is per sample, because the tile
phase is derived per segment, so the index is $(B, A, H, R)$. The property that distinguishes the two
is asserted directly: two rows with different anchors must produce two different windows, each the
raw samples its own anchor names.

**Why the bounds check is here rather than only in the mask.** Advanced indexing on a negative index
*wraps*, so an anchor of $-1$ would gather the last legal window and return every shape correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

from .conftest import BATCH, TINY_STRIDE, build, make_raw_signal, tiny_warmup_kwargs


def _kwargs() -> dict:
    """The tiny guarded keyword set at a real tiling."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


@pytest.fixture(scope="module")
def model():
    """The tiny guarded model, built once."""
    return build(_kwargs()).eval()


# =================================================================================================
# The gather itself
# =================================================================================================
def test_the_anchored_path_equals_the_dense_builder_at_the_dense_anchor_set(model) -> None:
    r"""At ``anchors = arange(T_valid)`` the two expressions of the same arithmetic must agree
    elementwise, under ``torch.equal``: both paths read the same integer grid and gather the same
    floats, so any difference is an indexing mistake rather than an accumulation one.
    """
    signal = make_raw_signal(_kwargs())
    dense = build_future_target(signal, model.geometry, future_index=model.future_index)
    anchors = torch.arange(model.geometry.t_valid)[None, :].expand(BATCH, -1)

    anchored = gather_anchored_future_target(
        signal, model.geometry, anchors, future_index=model.future_index
    )

    assert tuple(anchored.shape) == tuple(dense.shape)
    assert torch.equal(anchored, dense)


def test_a_per_sample_anchor_set_yields_per_sample_windows(model) -> None:
    """The property an ``index_select`` cannot have: two rows, two different windows, each checked
    against the raw samples its own anchor names."""
    signal = make_raw_signal(_kwargs())
    anchors = torch.tensor([[5, 9], [6, 10]])

    windows = gather_anchored_future_target(
        signal, model.geometry, anchors, future_index=model.future_index
    )

    assert tuple(windows.shape) == (2, 2, model.horizon, model.geometry.r)
    assert not torch.equal(windows[0], windows[1])
    for row in range(2):
        for slot in range(2):
            anchor = int(anchors[row, slot])
            start = model.geometry.future_block_start(anchor)
            stop = start + model.horizon * model.geometry.r
            expected = signal[row, start:stop].reshape(model.horizon, model.geometry.r)
            assert torch.equal(windows[row, slot], expected), (row, slot)


# =================================================================================================
# The refusals
# =================================================================================================
@pytest.mark.parametrize("where", ["past_the_last_valid_anchor", "negative"])
def test_an_anchor_outside_the_valid_range_is_refused_naming_it(model, where) -> None:
    r"""$\ge T_{\mathrm{valid}}$, not $\ge T$: the tail $H$ anchors have no fully observed window.
    And $-1$ is refused rather than wrapped to the last legal window."""
    signal = make_raw_signal(_kwargs())
    offending = model.geometry.t_valid if where == "past_the_last_valid_anchor" else -1

    with pytest.raises(ValueError, match=f"anchor {offending} "):
        gather_anchored_future_target(
            signal,
            model.geometry,
            torch.tensor([[5, offending]]),
            future_index=model.future_index,
        )


def test_a_signal_at_another_trim_is_refused_naming_both_lengths(model) -> None:
    """A loader at a different ``trim_minutes`` shifts every window by whole minutes, and a longer
    signal would gather silently rather than raise."""
    signal = make_raw_signal(_kwargs())

    with pytest.raises(ValueError, match="raw_len"):
        gather_anchored_future_target(
            signal[:, :-16],
            model.geometry,
            torch.tensor([[5]]),
            future_index=model.future_index,
        )
    with pytest.raises(ValueError, match="2-D"):
        gather_anchored_future_target(
            signal[0], model.geometry, torch.tensor([[5]]), future_index=model.future_index
        )
