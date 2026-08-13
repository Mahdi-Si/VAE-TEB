r"""The objective as this model wires it, against the raw-signal suite's reassembly.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and its own suite pins them; a second copy of those
assertions would be a second copy of one piece of evidence. What this file supplies is what this
architecture owns -- its model, its target and its block width -- so that the shared harness is
driven against the conv-Transformer encoder as well as the conv-LSTM one.

That matters because the two reach the objective through different code. This encoder has no
recurrent branch, a different input adapter seam and a windowed source attention, and the seam
that lets a target domain hand the objective its own anchor set is shared by both. A harness run
against one of them says nothing about the other.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index, build_future_target
from teb_vae.lag_attn_rws.tests.test_objective import assert_objective_reassembles
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    TINY_KWARGS,
    make_stub_batch,
    tiny_gated_kwargs,
)

#: Weights every term is exercised at. Mutually distinct and none of them a default: at equal
#: weights a term swapped for another passes, at ``beta_prior=0`` the fourth term is multiplied
#: away, at ``free_bits=0`` the raw and trained KL are one tensor rather than two, and at a zero
#: shape weight the term is not computed at all.
_COEFFICIENTS = dict(
    beta=0.7,
    beta_prior=0.11,
    lambda_full=1.0,
    lambda_base=0.3,
    free_bits=0.05,
    lambda_ms=0.13,
    lambda_deriv=0.17,
    lambda_boundary=0.19,
)


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
@pytest.mark.parametrize("guard", ["ungated", "gated"], ids=["ungated", "gated"])
def test_every_metric_reassembles_from_the_primitives(perturb_posterior, likelihood, guard):
    """Both guard states, both likelihoods, every reported metric compared with ``torch.equal``."""
    kwargs = tiny_gated_kwargs(TINY_KWARGS) if guard == "gated" else dict(TINY_KWARGS)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**kwargs).eval()
    perturb_posterior(model)
    batch = make_stub_batch()

    torch.manual_seed(0)
    with torch.no_grad():
        outputs = model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )
        produced = model.compute_loss(
            outputs, batch.fhr, weight=batch.weight, likelihood=likelihood, **_COEFFICIENTS
        )["metrics"]

    geometry = model.geometry
    assert_objective_reassembles(
        model,
        outputs,
        build_future_target(
            batch.fhr, geometry, future_index=build_future_index(geometry)
        ),
        batch.weight,
        produced,
        likelihood=likelihood,
        coefficients=_COEFFICIENTS,
        # The raw grid's R, written out here rather than read off the objective: this denominator
        # is the one quantity a self-consistent objective cannot be checked against itself on.
        block_width=geometry.r,
    )

    # Not vacuous: the perturbation is what puts the KL and the posterior displacement on non-zero
    # values, so the reassembly compared them at something.
    assert float(produced["source_conditioned_kl_raw"]) > 0.0
    assert float(produced["delta_mu_rms"]) > 0.0
