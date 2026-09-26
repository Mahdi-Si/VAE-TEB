r"""What the forward accepts, and what each refusal has to say.

Two guards, and both exist because of a failure that produces *correctly shaped* tensors somewhere
else. The raw length is ``sequence_length * raw_per_step``, which is the trimmed geometry: a loader
configured at a different ``trim_minutes`` hands over a longer or shorter signal, and every
downstream shape then follows the input rather than the geometry the anchors, the masks and the
raw-target index grid were all built against. The weight is ``(B, sequence_length)`` on the
decimated grid; an untrimmed one is longer than the $T$ steps the model wants.

Both are asserted **by message content**, not by exception type. A ``ValueError`` alone tells an
operator that something about the batch is wrong; naming ``trim_minutes`` and the expected length
tells them which knob produced it, which is the difference between a five-minute fix and an
afternoon.

The guards read shape metadata only and never a tensor's content, which is what keeps them
compatible with the DDP walk in ``test_ddp_reachability.py``: a forward that branched on a value
would drop parameters from the graph on some ranks and not others and hang the run. A shape guard
that raises fails the run instead, which is the wanted outcome.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import BATCH, SEQ_LEN


def _model(tiny_kwargs) -> SeqVaeLagAttnTrfE2E:
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfE2E(**tiny_kwargs).eval()


def _raw(steps: int, raw_per_step: int = 16) -> torch.Tensor:
    return torch.randn(BATCH, steps * raw_per_step)


# ---------------------------------------------------------------------------------------
# The raw length
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("stream", [0, 1], ids=["target", "source"])
def test_a_raw_signal_of_the_wrong_length_is_refused_naming_the_trim(tiny_kwargs, stream):
    """The shape an untrimmed loader produces, on either stream. The message has to name
    ``trim_minutes`` and the expected length: those two together are the whole diagnosis."""
    model = _model(tiny_kwargs)
    inputs = [_raw(SEQ_LEN), _raw(SEQ_LEN), torch.ones(BATCH, SEQ_LEN)]
    inputs[stream] = _raw(SEQ_LEN + 2)

    with pytest.raises(ValueError, match=r"trim_minutes") as raised:
        model(*inputs)

    assert f"(B, {model.geometry.raw_len})" in str(raised.value), str(raised.value)
    assert str((SEQ_LEN + 2) * 16) in str(raised.value), str(raised.value)


def test_a_three_dimensional_raw_signal_is_refused(tiny_kwargs):
    """A feature block handed to a raw input -- the shape a config still listing ``fhr_st`` in
    ``load_fields`` would eventually produce."""
    model = _model(tiny_kwargs)

    with pytest.raises(ValueError, match=r"two raw signals of shape"):
        model(torch.randn(BATCH, SEQ_LEN, 43), _raw(SEQ_LEN), torch.ones(BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------------------
# The weight
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "weight_shape",
    [(BATCH, SEQ_LEN + 2), (BATCH, SEQ_LEN, 1)],
    ids=["untrimmed-grid", "channel-axis"],
)
def test_a_weight_off_the_trimmed_grid_is_refused_naming_it(tiny_kwargs, weight_shape):
    """An untrimmed weight is the shape a loader that skipped the trim hands over; the message names
    the trimmed $(B, T)$ grid the model wants."""
    model = _model(tiny_kwargs)

    with pytest.raises(ValueError, match=r"trimmed decimated grid") as raised:
        model(_raw(SEQ_LEN), _raw(SEQ_LEN), torch.ones(*weight_shape))

    assert f"(B, {SEQ_LEN})" in str(raised.value)


# ---------------------------------------------------------------------------------------
# Which guard fires
# ---------------------------------------------------------------------------------------
def test_the_model_guard_fires_before_the_front_ends_own(tiny_kwargs):
    """The front end has a length guard of its own, and it is the more general one -- it knows its
    stride but not this model's ``sequence_length``. The model's has to come first, or a
    wrong-trim batch would be refused with a message about a stride ratio rather than about the
    trim that caused it."""
    model = _model(tiny_kwargs)

    with pytest.raises(ValueError) as raised:
        model(_raw(SEQ_LEN + 2), _raw(SEQ_LEN + 2), torch.ones(BATCH, SEQ_LEN + 2))

    # Both lengths are internally consistent at stride 16, so the front end would have accepted
    # them; only the model knows the grid is the wrong size.
    assert "trim_minutes" in str(raised.value)
