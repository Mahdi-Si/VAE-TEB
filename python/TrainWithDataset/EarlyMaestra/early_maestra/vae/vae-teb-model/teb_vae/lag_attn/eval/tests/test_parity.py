r"""The load-bearing test of the pipeline: eval's numbers must be the training loss's numbers.

Every conclusion an eval run supports rests on its masked feature loss and its masked KL being
the same quantities ``compute_loss`` optimised. If they are not, the two never reconcile and the
disagreement is invisible -- both sides produce plausible numbers of the same magnitude.

So each case here runs a real forward, computes the loss twice, and asserts agreement:

* through ``model.compute_loss``, which is what training used;
* through :mod:`~teb_vae.lag_attn.eval.masks` and :mod:`~teb_vae.lag_attn.eval.metrics`, which is
  what every analysis uses.

The cases are chosen for what separates the two implementations rather than for coverage:
``kld_support`` decides whether the untrained tail is in the KL; ``likelihood`` decides whether
the log-variance head enters the feature loss at all; ``free_bits`` is the *only* thing that
separates ``kld_train`` from ``kld_raw``; and ``detach_baseline_in_full`` makes ``compute_loss``
recompute ``mu_full`` internally, so the forward dict's own ``mu_full`` is **not** what was
scored -- a pipeline reading the forward key would be wrong by exactly the baseline's gradient
detachment and would look right in every other respect.

Every case runs on a ``perturb_posterior`` model. On an untouched one the posterior equals the
prior exactly, so every KL is $0$ and both sides agree vacuously -- including on a model that is
entirely wrong.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest
import torch

from teb_vae.lag_attn.eval import masks, metrics
from teb_vae.lag_attn.figure_primitives import future_target
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.tests.conftest import SEQ_LEN, SHIPPED_KWARGS, make_stub_batch

#: Relative tolerance. Both sides are fp32 reductions over the same tensors in a different
#: association order, so bit-equality is not available and 1e-5 is the meaningful bound.
RELATIVE_TOLERANCE = 1e-5

#: Objective settings crossed in the cases below. Each entry is (label, overrides).
OBJECTIVE_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("mse", {"likelihood": "mse", "sigma_obs": 1.0}),
    ("gaussian_nll_learned", {"likelihood": "gaussian_nll", "sigma_obs": "learned"}),
    ("gaussian_nll_scalar", {"likelihood": "gaussian_nll", "sigma_obs": 0.5}),
)


def _build(kld_support: str, perturb_posterior) -> Tuple[SeqVaeLagAttn, Any]:
    """Build a perturbed tiny model and a stub batch under the requested KL support."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(SHIPPED_KWARGS, kld_support=kld_support))
    perturb_posterior(model)
    model.eval()
    return model, make_stub_batch(batch_size=3, seq_len=SEQ_LEN, seed=1)


def _forward(model: SeqVaeLagAttn, batch: Any) -> Dict[str, torch.Tensor]:
    """Run one seeded forward, so both sides of every comparison see the same $z$."""
    torch.manual_seed(123)
    with torch.no_grad():
        return model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )


def _eval_feature_loss(
    model: SeqVaeLagAttn,
    batch: Any,
    outputs: Dict[str, torch.Tensor],
    objective: Dict[str, Any],
    *,
    detach_baseline_in_full: bool,
) -> torch.Tensor:
    """Compute the feature loss the way an analysis does, through masks.py and metrics.py."""
    seq_len = int(batch.fhr_st.shape[1])
    horizon = int(model.horizon)
    anchors = seq_len - horizon

    # Under detach_baseline_in_full, compute_loss recomputes mu_full = mu_base + delta_mu_src
    # rather than using the forward's own key. Detaching changes no value under no_grad, but the
    # *recomposition* does when the two differ -- so the pipeline must recompose too.
    if detach_baseline_in_full:
        mu_full = outputs["mu_base"] + outputs["delta_mu_src"]
    else:
        mu_full = outputs["mu_full"]

    target = future_target(batch.fhr_st, batch.fhr_ph, horizon)
    mask = masks.feature_mask(
        model, batch.weight, int(batch.fhr_st.shape[0]), seq_len, dtype=target.dtype
    )
    return metrics.feature_loss(
        mu_full[:, :anchors],
        target,
        outputs["logvar_full"][:, :anchors],
        mask,
        likelihood=objective["likelihood"],
        sigma_obs=objective["sigma_obs"],
    )


def _eval_kld(
    model: SeqVaeLagAttn,
    batch: Any,
    outputs: Dict[str, torch.Tensor],
    *,
    free_bits: float,
) -> torch.Tensor:
    """Compute the pooled KL the way an analysis does."""
    seq_len = int(batch.fhr_st.shape[1])
    kld_btd = metrics.kld_per_dim(outputs, model)
    mask_bt = masks.kld_mask(
        model, batch.weight, int(batch.fhr_st.shape[0]), seq_len, dtype=kld_btd.dtype
    )
    return metrics.kld_pooled(kld_btd, mask_bt, free_bits=free_bits)


# ---------------------------------------------------------------------------
# Feature loss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kld_support", ["full", "anchor"])
@pytest.mark.parametrize("label,objective", OBJECTIVE_CASES, ids=[case[0] for case in OBJECTIVE_CASES])
@pytest.mark.parametrize("detach_baseline_in_full", [False, True])
def test_eval_feature_loss_equals_compute_loss_feat_loss(
    kld_support: str,
    label: str,
    objective: Dict[str, Any],
    detach_baseline_in_full: bool,
    perturb_posterior,
) -> None:
    """The single assertion that proves the mask and the objective are right.

    It fails if the mask changes in either place, if the anchor slice moves, if the channel
    factor in the denominator is dropped, or if the pipeline reads the forward's ``mu_full``
    under ``detach_baseline_in_full``.
    """
    model, batch = _build(kld_support, perturb_posterior)
    outputs = _forward(model, batch)

    with torch.no_grad():
        reference = model.compute_loss(
            forward_outputs=outputs,
            y_st=batch.fhr_st,
            y_ph=batch.fhr_ph,
            weight=batch.weight,
            beta=0.0,
            detach_baseline_in_full=detach_baseline_in_full,
            **objective,
        )["feat_loss"]

    ours = _eval_feature_loss(
        model, batch, outputs, objective, detach_baseline_in_full=detach_baseline_in_full
    )
    assert float(ours) == pytest.approx(float(reference), rel=RELATIVE_TOLERANCE)


def test_the_parity_check_is_sensitive_to_a_wrong_mask(perturb_posterior) -> None:
    """A non-vacuity guard: perturbing the mask must break the agreement.

    Without this the parity test could be passing because both sides are trivially zero, or
    because the comparison is not actually comparing anything.
    """
    model, batch = _build("anchor", perturb_posterior)
    outputs = _forward(model, batch)
    objective = {"likelihood": "gaussian_nll", "sigma_obs": "learned"}

    with torch.no_grad():
        reference = float(
            model.compute_loss(
                forward_outputs=outputs,
                y_st=batch.fhr_st,
                y_ph=batch.fhr_ph,
                weight=batch.weight,
                beta=0.0,
                **objective,
            )["feat_loss"]
        )

    horizon = int(model.horizon)
    anchors = SEQ_LEN - horizon
    target = future_target(batch.fhr_st, batch.fhr_ph, horizon)
    # One anchor short of the real warm-up: the exact off-by-one the single-builder rule exists
    # to prevent.
    wrong_mask = masks.feature_mask(model, batch.weight, 3, SEQ_LEN, dtype=target.dtype).clone()
    wrong_mask[:, int(model.warmup_period)] = 0.0

    wrong = float(
        metrics.feature_loss(
            outputs["mu_full"][:, :anchors],
            target,
            outputs["logvar_full"][:, :anchors],
            wrong_mask,
            **objective,
        )
    )
    assert wrong != pytest.approx(reference, rel=RELATIVE_TOLERANCE)


# ---------------------------------------------------------------------------
# KL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kld_support", ["full", "anchor"])
def test_eval_kld_equals_kld_raw(kld_support: str, perturb_posterior) -> None:
    """``kld_raw`` is the un-floored KL over the support -- the only key readable as a TE surrogate."""
    model, batch = _build(kld_support, perturb_posterior)
    outputs = _forward(model, batch)

    with torch.no_grad():
        reference = model.compute_loss(
            forward_outputs=outputs,
            y_st=batch.fhr_st,
            y_ph=batch.fhr_ph,
            weight=batch.weight,
            beta=0.0,
            free_bits=0.5,
        )["kld_raw"]

    ours = _eval_kld(model, batch, outputs, free_bits=0.0)
    assert float(ours) == pytest.approx(float(reference), rel=RELATIVE_TOLERANCE)
    assert float(ours) > 0.0, "a perturbed posterior must give a nonzero KL, or this is vacuous"


@pytest.mark.parametrize("kld_support", ["full", "anchor"])
@pytest.mark.parametrize("free_bits", [0.0, 0.05, 0.5])
def test_eval_kld_equals_kld_train_under_free_bits(
    kld_support: str, free_bits: float, perturb_posterior
) -> None:
    """Free bits is the only thing separating ``kld_train`` from ``kld_raw``.

    The clamp is applied per dimension per step *before* masking. Clamping the aggregate instead
    would floor the total rather than each term and give a smaller number that still looks
    reasonable.
    """
    model, batch = _build(kld_support, perturb_posterior)
    outputs = _forward(model, batch)

    with torch.no_grad():
        reference = model.compute_loss(
            forward_outputs=outputs,
            y_st=batch.fhr_st,
            y_ph=batch.fhr_ph,
            weight=batch.weight,
            beta=0.0,
            free_bits=free_bits,
        )["kld_train"]

    ours = _eval_kld(model, batch, outputs, free_bits=free_bits)
    assert float(ours) == pytest.approx(float(reference), rel=RELATIVE_TOLERANCE)


def test_free_bits_raises_the_kl_so_the_two_keys_are_distinguishable(perturb_posterior) -> None:
    """``kld_train >= kld_raw`` always, and strictly so once the floor bites.

    If it did not, the previous test would be comparing two names for one number.
    """
    model, batch = _build("anchor", perturb_posterior)
    outputs = _forward(model, batch)
    raw = float(_eval_kld(model, batch, outputs, free_bits=0.0))
    floored = float(_eval_kld(model, batch, outputs, free_bits=0.5))
    assert floored > raw


def test_the_kl_ignores_weight_free_steps_the_same_way_the_model_does(
    perturb_posterior,
) -> None:
    """A gap must leave the KL unchanged whatever the posterior does there.

    The KL curve is only trustworthy as a coupling readout if invalid steps cannot contribute.
    """
    model, batch = _build("anchor", perturb_posterior)
    outputs = _forward(model, batch)

    weight = torch.ones_like(batch.weight)
    weight[:, -3:] = 0.0
    kld_btd = metrics.kld_per_dim(outputs, model)
    mask_bt = masks.kld_mask(model, weight, 3, SEQ_LEN, dtype=kld_btd.dtype)

    corrupted = kld_btd.clone()
    corrupted[:, -3:, :] += 1000.0
    assert float(metrics.kld_pooled(corrupted, mask_bt)) == pytest.approx(
        float(metrics.kld_pooled(kld_btd, mask_bt)), rel=RELATIVE_TOLERANCE
    )
