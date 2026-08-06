r"""The assembled model's causality, measured at raw-sample resolution.

This is the claim the package exists to make, measured where it matters: on ``target_state`` and
``source_state``, through the front end, the encoder and everything the constructor wired between
them. The front end's own probe proves the component; this proves the composition, which is where a
wiring mistake lives -- a stream handed the wrong weight, a transpose that slipped, a front end
attached to the wrong encoder.

The cut is stated in **raw samples**, not in tokens, and that is the whole difference from the
sibling's version of this file. Anchor $t$'s causal endpoint is raw index $16t + 15$. Perturbing raw
$16t + 16$ -- the very next sample -- must leave the state at $t$ bitwise identical; perturbing
$16t + 15$ must move it. A token-resolution probe cannot express that cut at all, which is precisely
why the model this one replaces the input of can pass one and still read its own future.

Two things make the assertion mean what it says.

* **float64 and ``torch.equal``.** A threshold at this boundary would be a statement about float32
  round-off rather than about causality -- the trap the sibling's ``test_source_window.py`` records
  hitting. In float64 the two halves separate by many orders of magnitude.
* **The movement half is asserted at two magnitudes.** A single raw sample is weighted lightly by
  design: each of the four front-end stages puts the newest sample on its filter's leading tap
  $h_0 = 1/16$, so it reaches the token carrying about $16^{-4}$ of it. The encoder amplifies that
  again by a factor that depends on how much history the anchor has -- measured here at
  $1.5 \times 10^{-3}$ at the first trained anchor against $6.9 \times 10^{-5}$ at the last -- so no
  single threshold near the shared ``MOVEMENT_TOL`` fits both. The small bar is set nine orders above
  float64 round-off instead, and the anchor's **whole step** is asserted against the shared tolerance
  beside it, which is what stops the small number from being round-off. Weighting the edge of a
  window lightly is what a low-pass is for; what it may not do is weight anything *past* that edge,
  which is the bitwise half.

The recorded decision behind what is *not* here: the sibling's ``test_encoder_causality.py`` and
``test_prefix_equivalence.py`` are not mirrored. Both test imported code over $(B, T, d)$ inputs
whose behaviour cannot change in this package, and prefix equivalence has no consumer here while
evaluation is out of scope. This assembled probe is the one that catches this package's own wiring.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets import frontend as frontend_module
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    MOVEMENT_TOL,
    SEQ_LEN,
    relative_change,
    resample_raw_after,
)
from teb_vae.lag_attn_transformer_e2e.tests.test_frontend_causality import (
    _SymmetricallyPaddedDecimate,
)

#: Relative movement a change to an anchor's **single newest** raw sample must exceed. Six orders
#: below the shared :data:`MOVEMENT_TOL` on purpose, and the reason is arithmetic rather than slack:
#: the four-stage cascade's leading tap is $16^{-4} \approx 1.5 \times 10^{-5}$, and what the encoder
#: then makes of it depends on how much history the anchor has -- measured at $1.5 \times 10^{-3}$
#: at anchor $6$ and $6.9 \times 10^{-5}$ at anchor $11$. Nine orders above float64 round-off, which
#: is what separates "reads it faintly, as a low-pass must" from "does not read it".
_ANCHOR_MOVEMENT_TOL = 1e-9

#: Anchors the probe runs at, inside the trained range $[6, 12)$ of the tiny geometry. More than one,
#: because an alignment error that happened to cancel at a single $t$ should not need ruling out by
#: reading the code.
_ANCHORS = (6, 11)

#: Which raw input each state is claimed to depend on, and which key reports it.
_STREAMS = ((0, "target_state"), (1, "source_state"))


def _model(tiny_kwargs, **overrides) -> SeqVaeLagAttnTrfE2E:
    """A float64 model in eval mode: bitwise assertions need both."""
    torch.manual_seed(0)
    return (
        SeqVaeLagAttnTrfE2E(**dict(tiny_kwargs, dropout=0.0, **overrides)).eval().double()
    )


def _inputs(raw_per_step: int = 16, *, amplitude: float = 100.0):
    """Seeded large-amplitude float64 raw signals and a fully valid weight.

    Large so a genuine dependence is visible far above round-off. Fully valid so the probe measures
    the convolution stack rather than the mask: a perturbation landing inside a gap is zeroed by the
    featurisation, and the resulting bit-stability would be a statement about the gap.

    Args:
        raw_per_step: Raw samples per decimated step.
        amplitude: Scale of the drawn signals.

    Returns:
        ``(y_raw, u_raw, weight)``, freshly drawn from a fixed seed so two calls are identical.
    """
    generator = torch.Generator().manual_seed(0)
    shape = (BATCH, SEQ_LEN * raw_per_step)
    return (
        amplitude * torch.randn(shape, generator=generator, dtype=torch.float64),
        amplitude * torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.ones(BATCH, SEQ_LEN, dtype=torch.float64),
    )


def _state(model, inputs, key: str) -> torch.Tensor:
    """One forward's history state, with the generator re-seeded so the $\\epsilon$ draw matches."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*inputs)[key]


def _perturbed(inputs, stream: int, cut: int, *, seed: int = 5) -> list:
    """The input tuple with one stream resampled strictly after ``cut``."""
    perturbed = list(inputs)
    perturbed[stream] = resample_raw_after(perturbed[stream], cut, seed=seed)
    return perturbed


# ---------------------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("stream, key", _STREAMS, ids=["target", "source"])
@pytest.mark.parametrize("anchor", _ANCHORS, ids=lambda t: f"t={t}")
def test_the_state_at_an_anchor_reads_no_raw_sample_after_its_own_endpoint(
    tiny_kwargs, stream, key, anchor
):
    r"""Resample every raw sample after $16t + 15$: the state at $t$ must be **bitwise** unmoved.

    $16t + 15$ is ``TrimmedRawGeometry.n_raw(t)``, the newest raw sample anchor $t$ is allowed to
    have seen. Perturbing $16t + 16$ is the smallest wrong answer the geometry admits, and the one an
    off-by-one in the decimation offset produces.

    Paired with movement at the last anchor, which is the negative control: a stream that computed
    nothing would be bit-stable at every cut.
    """
    model = _model(tiny_kwargs)
    inputs = _inputs(model.raw_per_step)
    cut = model.raw_per_step * (anchor + 1) - 1

    reference = _state(model, inputs, key)
    perturbed = _state(model, _perturbed(inputs, stream, cut), key)

    assert torch.equal(reference[:, anchor], perturbed[:, anchor]), (
        f"{key} at anchor {anchor} moved when only raw samples after {cut} changed, so it reads "
        f"its own future"
    )
    at_end = relative_change(reference[:, -1], perturbed[:, -1])
    assert at_end > MOVEMENT_TOL, (
        f"{key}: the last anchor moved by only {at_end:.3e} -- the perturbation never reached the "
        f"model, so the bit-stability above proves nothing"
    )


@pytest.mark.parametrize("stream, key", _STREAMS, ids=["target", "source"])
@pytest.mark.parametrize("anchor", _ANCHORS, ids=lambda t: f"t={t}")
def test_the_state_at_an_anchor_does_read_its_own_endpoint(tiny_kwargs, stream, key, anchor):
    r"""The other side of the same boundary, at two magnitudes.

    Perturbing from $16t + 15$ onward -- one sample earlier than the test above -- must move the
    state at $t$ at all, or the stack would be discarding the newest quarter-second of every anchor
    and the bitwise test would be certifying something strictly more conservative than causality.
    Perturbing the anchor's whole step must move it substantially.

    The two bars are far apart, and that is a property of the design: four cascaded anti-alias
    filters put the newest raw sample on the leading tap each time, so it carries about $16^{-4}$ of
    the token before the encoder sees it. Measured, the newest sample moves the state by $6.9 \times
    10^{-5}$ to $1.5 \times 10^{-3}$ depending on the anchor, against $4.7 \times 10^{-3}$ to
    $3.8 \times 10^{-1}$ for its sixteen samples together. Without the first assertion the boundary
    is untested; without the second the first could be measuring round-off.
    """
    model = _model(tiny_kwargs)
    inputs = _inputs(model.raw_per_step)
    newest = model.raw_per_step * (anchor + 1) - 1

    reference = _state(model, inputs, key)
    from_anchor = _state(model, _perturbed(inputs, stream, newest - 1), key)
    from_step = _state(
        model, _perturbed(inputs, stream, model.raw_per_step * anchor - 1), key
    )

    moved = relative_change(reference[:, anchor], from_anchor[:, anchor])
    moved_step = relative_change(reference[:, anchor], from_step[:, anchor])

    assert moved > _ANCHOR_MOVEMENT_TOL, (
        f"{key} at anchor {anchor} moved by only {moved:.3e} when its own newest raw sample "
        f"{newest} changed, so the stack is not reading up to its anchor at all"
    )
    assert moved_step > MOVEMENT_TOL, (
        f"{key} at anchor {anchor} moved by only {moved_step:.3e} when its whole step was redrawn"
    )


def test_the_latent_carries_the_same_boundary(tiny_kwargs):
    """One step further down than the two states, because the states are not what the objective
    scores: the prior at anchor $t$ is what the base forecast is decoded from, so it is the tensor a
    leak would corrupt the ``pred_gap`` readout through."""
    model = _model(tiny_kwargs)
    inputs = _inputs(model.raw_per_step)
    anchor = _ANCHORS[0]
    cut = model.raw_per_step * (anchor + 1) - 1

    reference = _state(model, inputs, "mu_prior")
    perturbed = _state(model, _perturbed(inputs, 0, cut), "mu_prior")

    assert torch.equal(reference[:, anchor], perturbed[:, anchor])
    assert relative_change(reference[:, -1], perturbed[:, -1]) > MOVEMENT_TOL


# ---------------------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------------------
def test_the_probe_would_catch_a_model_whose_front_end_read_one_sample_ahead(
    tiny_kwargs, monkeypatch
):
    """The control the whole file rests on, planted where a real edit puts it.

    ``padding=(k-1)//2`` is what ``nn.Conv1d`` does by default and what anybody reaching for the
    ``padding`` argument would write. The planted decimator is the one the front end's own suite uses
    -- imported rather than rewritten -- and it changes no shape, no parameter count and no reach, so
    nothing but a causality probe could find it. A centred *offset* would not be a valid control: it
    makes an anchor depend on raw $\\le 16t$, which is strictly more conservative, so the broken model
    would pass the bitwise half and the control would prove nothing.
    """
    monkeypatch.setattr(frontend_module, "CausalAntiAliasDecimate", _SymmetricallyPaddedDecimate)
    broken = _model(tiny_kwargs)
    inputs = _inputs(broken.raw_per_step)
    anchor = _ANCHORS[0]
    cut = broken.raw_per_step * (anchor + 1) - 1

    reference = _state(broken, inputs, "target_state")
    perturbed = _state(broken, _perturbed(inputs, 0, cut), "target_state")

    assert not torch.equal(reference[:, anchor], perturbed[:, anchor]), (
        "a symmetrically padded front end passed the causality probe; the probe is not measuring "
        "what it claims to"
    )
