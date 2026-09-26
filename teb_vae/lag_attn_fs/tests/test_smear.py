r"""The smeared target is not a leak: the model reads no step after the anchor.

A stored coefficient at decimated step $s$ is a weighted average of raw signal over a window
**centred** at $s$, so a forecast target at short horizons is partly fixed by signal the model has
already observed. That is not a causality violation, and two claims carry the distinction:

1. *Nothing the model reads reaches past the anchor.* Per surviving channel the model reads
   channel $c$ at step $t - \delta_c$ at the latest, and $\delta_c$ covers the channel's forward
   reach. The delay arithmetic is pinned where the budget is resolved
   (``lag_attn_rws/tests/test_budget_resolution.py``); what it assumes -- that the network itself
   is causal step by step -- is asserted here, on this model.
2. *What is smeared into the target came from history the model legitimately holds*, so it
   cannot manufacture a base-minus-full gap; the zero-gap start is pinned in ``test_invariants.py``.

Step-wise causality holds under ``causal_norm: true``, which the shipped configuration sets and
the tiny keyword set does not. The paired test shows the qualifier is load-bearing: without it the
time-pooling normaliser mixes the whole sequence.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs

_CUT = 8


def _streams():
    """Seeded target and source streams at the tiny geometry."""
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(2, 16, 43, generator=generator),
        torch.randn(2, 16, 66, generator=generator),
        torch.randn(2, 16, 58, generator=generator),
        generator,
    )


def test_the_model_reads_no_step_after_the_anchor(tiny_gated):
    """Perturbing every target step after the cut leaves every output up to the cut bitwise
    unchanged, under ``causal_norm: true``."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(tiny_gated, causal_norm=True, dropout=0.0)).eval()
    y_st, y_ph, u_stream, generator = _streams()

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    perturbed_st, perturbed_ph = y_st.clone(), y_ph.clone()
    perturbed_st[:, _CUT + 1 :] = torch.randn(
        perturbed_st[:, _CUT + 1 :].shape, generator=generator
    )
    perturbed_ph[:, _CUT + 1 :] = torch.randn(
        perturbed_ph[:, _CUT + 1 :].shape, generator=generator
    )
    torch.manual_seed(0)
    with torch.no_grad():
        moved = model(perturbed_st, perturbed_ph, u_stream)

    assert torch.equal(reference["mu_prior"][:, : _CUT + 1], moved["mu_prior"][:, : _CUT + 1])
    assert torch.equal(reference["mu_base"][:, : _CUT + 1], moved["mu_base"][:, : _CUT + 1])
    # The paired control: the perturbation did reach the model, so the bit-stability above is
    # a statement about causality rather than about a dead pathway.
    assert not torch.equal(reference["mu_prior"][:, -1], moved["mu_prior"][:, -1])


def test_without_causal_norm_the_step_wise_claim_does_not_hold(tiny_gated):
    """The negative control: with the time-pooling normaliser left in, the same probe sees the
    future, which is what ``causal_norm`` exists to fix."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(tiny_gated, causal_norm=False, dropout=0.0)).eval()
    y_st, y_ph, u_stream, generator = _streams()

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    perturbed = y_st.clone()
    perturbed[:, _CUT + 1 :] = torch.randn(perturbed[:, _CUT + 1 :].shape, generator=generator)
    torch.manual_seed(0)
    with torch.no_grad():
        moved = model(perturbed, y_ph, u_stream)

    assert model.n_causalized_norms == 0
    assert not torch.equal(reference["mu_prior"][:, : _CUT + 1], moved["mu_prior"][:, : _CUT + 1])
