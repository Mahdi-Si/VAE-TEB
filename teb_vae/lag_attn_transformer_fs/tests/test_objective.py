r"""The objective as this model wires it: the target it builds, and the width it declares.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and ``lag_attn_fs`` pins the mixin's target builder and gap
splits against its own model. What is this composition's to get wrong is the wiring:

* **The target.** Gathered from the caller's feature stream by the keep-index of a gate that *this*
  architecture builds at its own construction site, and unfolded into each anchor's future window --
  never delayed.
* **``block_width``.** $C_{\mathrm{keep}}$, the surviving-channel count. It feeds only the
  per-element log-variance diagnostics and the per-sample scores, so passing ``geometry.r`` instead
  would change no gradient, fail no shape check and silently rescale exactly those readouts.

Both are checked against **hand-written** quantities rather than against the implementation: the
target against a slice-and-stack that shares no arithmetic with ``unfold`` (a builder that wrongly
applied the delay would agree with a reference that wrongly applied it), and every metric against
the raw-signal suite's independent reassembly at a hand-written block width.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.tests.test_objective import assert_objective_reassembles
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    SHIPPED_KWARGS,
    TINY_KEEP_INDEX,
    TINY_KWARGS,
    make_patterned_batch,
    make_stub_batch,
    tiny_gated_kwargs,
)

#: Coefficients the recomposition runs at. Mutually distinct and none of them a default: at equal
#: weights a term swapped for another passes, at ``beta_prior=0`` the fourth term is multiplied
#: away, and at ``free_bits=0`` the raw and trained KL are one tensor rather than two.
_COEFFICIENTS = dict(
    beta=0.7, beta_prior=0.11, lambda_full=1.0, lambda_base=0.3, free_bits=0.05
)

#: The four resolved forecast gaps the feature target adds on top of the shared objective's keys.
_RESOLVED_GAP_KEYS = {
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
}


def _model(kwargs, cls=SeqVaeLagAttnTrfFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides)).eval()


def _features(batch) -> torch.Tensor:
    """The concatenated target stream, in the declared block order."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )


def _stacked_block(stream: torch.Tensor, horizon: int, t_valid: int) -> torch.Tensor:
    r"""The target block built by slicing and stacking, sharing no arithmetic with ``unfold``.

    Horizon step $\tau$ of every anchor is one contiguous slice of the stream, so the whole block is
    $H$ slices stacked on a new axis.

    Args:
        stream: A feature stream $(B, T, C)$.
        horizon: The forecast horizon $H$.
        t_valid: The number of valid anchors.

    Returns:
        The block $(B, T_{\mathrm{valid}}, H, C)$.
    """
    return torch.stack(
        [stream[:, 1 + tau : 1 + tau + t_valid, :] for tau in range(horizon)], dim=2
    )


def test_the_index_identity_holds_at_every_position_and_the_target_is_not_delayed(shipped_gated):
    r"""$Y^{+}[b, t, \tau, k] = Y[b,\, t + 1 + \tau,\, \mathrm{keep}[k]]$, whole block, at the
    shipped budget and against this architecture's own gate.

    ``ChannelGate.forward`` gathers *and* delays, so a builder that called the gate would be delayed
    and every shape check would still pass. The paired control asserts that at this budget the
    gate-built block genuinely differs, so the identity is not satisfied by a gate with no delay.
    """
    model = _model(shipped_gated)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    stream = _features(batch)

    built = model._build_forecast_target(stream)
    kept = torch.index_select(stream, -1, model.target_gate.keep_index)

    assert torch.equal(built, _stacked_block(kept, model.horizon, model.geometry.t_valid))
    delayed = _stacked_block(model.target_gate(stream), model.horizon, model.geometry.t_valid)
    assert delayed.shape == built.shape
    assert not torch.equal(built, delayed)


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
@pytest.mark.parametrize("guard", ["ungated", "gated"], ids=["ungated", "gated"])
def test_every_metric_reassembles_from_the_primitives(perturb_posterior, likelihood, guard):
    """This model's metrics, against the raw-signal suite's independent reassembly: the total, the
    per-sample scores, the log-variance diagnostics and the masking, every key, ``torch.equal``.

    What this file supplies is what the composition owns: its target, built by the slice-and-stack
    and gathered at the tiny guard's keep-index, its hand-written block width, and the four resolved
    forecast gaps as package-owned keys.
    """
    gated = guard == "gated"
    model = _model(tiny_gated_kwargs() if gated else dict(TINY_KWARGS))
    perturb_posterior(model)
    batch = make_stub_batch()
    outs = _forward(model, batch)

    target = _stacked_block(_features(batch), model.horizon, model.geometry.t_valid)
    if gated:
        target = torch.index_select(target, -1, torch.tensor(TINY_KEEP_INDEX))

    assert_objective_reassembles(
        model,
        outs,
        target,
        batch.weight,
        model.compute_loss(
            outs, _features(batch), weight=batch.weight, likelihood=likelihood, **_COEFFICIENTS
        )["metrics"],
        likelihood=likelihood,
        coefficients=_COEFFICIENTS,
        # Hand-written: the surviving width, or the declared one when nothing was dropped.
        block_width=len(TINY_KEEP_INDEX) if gated else int(TINY_KWARGS["c_y"]),
        package_owned=_RESOLVED_GAP_KEYS,
    )
