r"""The objective as this cell wires it: the anchors it averages over, the mask, and the delegation.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every reduction
and every reported metric, and its own suite pins them. What *this* cell can get wrong is narrower,
and is what is checked:

* **The delegation.** At the dense stride this model's ``compute_loss`` must reproduce the shared
  objective bitwise, given the same anchor index explicitly and a target gathered here -- which is
  also what catches a wrong ``block_width`` (it rescales only the log-variance diagnostics).
* **The headline gap.** ``pred_gap`` is recomputed by hand over the anchor set the forward actually
  decoded, branch by branch, at a tiling with a padded slot.
* **The mask.** A value planted at a padded anchor slot, or at a raw sample no surviving anchor
  scores, must move the loss by *exactly* zero -- multiplicatively, not approximately.
* **The refusals.** A weighted boundary term over a tiled anchor set, and a forward dict stripped of
  its anchor set, must both raise rather than score.

Every assertion that ``pred_gap`` is non-zero is paired with ``perturb_posterior``: the posterior
delta heads are zero-initialised, so at initialisation the posterior *is* the prior.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    tiny_warmup_kwargs,
)

#: What this cell's ``compute_loss`` adds to the shared objective's metric dict. Declared as one set
#: so a fourth addition fails here rather than arriving in a CSV no callback collects.
_ADDED_METRIC_KEYS = {
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
}

#: The two phases the tiled fixtures run at. Chosen so the second row is one anchor short of the
#: first, which is the only way a padded slot exists at all -- and every padding assertion below
#: would pass vacuously without one.
_PHASES = (0, TINY_STRIDE - 1)


def _weight(model, batch: int = BATCH, gap_step: int = -1, value: float = 1.0) -> torch.Tensor:
    """A uniform decimated weight at a model's own sequence length, optionally with one step zeroed."""
    weight = torch.full((batch, model.geometry.t), float(value))
    if gap_step >= 0:
        weight[:, gap_step] = 0.0
    return weight


def _tiled(stride: int = TINY_STRIDE, **overrides):
    """The tiny model at a tiling, its three input tensors, and a seeded raw target signal."""
    kwargs = tiny_warmup_kwargs(anchor_stride=stride, **overrides)
    model = build(kwargs).eval()
    return model, make_streams(kwargs), make_raw_signal(kwargs)


def _forward(model, streams, phase, stride=None):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams, phase, stride)


def _hand_block(model, out, signal, weight, *, branch: str, likelihood: str) -> float:
    r"""One branch's block score, reduced here rather than by the objective.

    The same three steps the objective takes and no shortcut through any of them: the per-sample
    score of :func:`~teb_vae.lag_attn_rws.nets.losses.raw_sample_score`, multiplied by the forecast
    mask built at the anchors the forward decoded, summed, and divided by the count of anchors that
    contribute at all.

    Args:
        model: The model whose geometry and coverage floor the mask is built at.
        out: The forward dict, read for its four forecast tensors and its anchor set.
        signal: The raw target signal $(B, L_{\mathrm{raw}})$.
        weight: The decimated validity signal $(B, T)$.
        branch: ``'base'`` or ``'full'``.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        The per-anchor block score.
    """
    target = gather_anchored_future_target(
        signal, model.geometry, out["anchor_index"], future_index=model.future_index
    )
    mask, _coverage = forecast_mask(
        weight,
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=out["anchor_index"],
        anchor_valid=out["anchor_valid"],
    )
    score = raw_sample_score(
        out[f"mu_{branch}"],
        target,
        likelihood=likelihood,
        logvar=out[f"logvar_{branch}"],
    )
    anchors = contributing_anchors(mask).sum().clamp_min(1.0)
    return float((score * mask[..., None]).sum() / anchors)


# =================================================================================================
# The headline gap
# =================================================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_headline_gap_survives_the_anchored_gather(perturb_posterior, likelihood) -> None:
    r"""$D_0 - D_1$ over the anchors the forward decoded, recomputed here from the mask up.

    The claim is split in two so neither half hides in the other's cancellation: each branch's
    block score is recomputed independently against the anchored target -- the masked sum over the
    decoded anchors divided by the count that contribute -- and the gap is then the exact difference
    of the two columns beside it.

    Asserted **after** ``perturb_posterior``, and paired with the zero it reads at initialisation, so
    a "the gap is finite" check cannot hold on a model that ignores its source entirely.
    """
    model, streams, signal = _tiled()
    weight = _weight(model)
    phase = torch.tensor(_PHASES)

    fresh = model.compute_loss(
        _forward(model, streams, phase), signal, weight=weight, likelihood=likelihood
    )["metrics"]
    assert float(fresh["pred_gap"]) == 0.0

    perturb_posterior(model)
    out = _forward(model, streams, phase)
    metrics = model.compute_loss(out, signal, weight=weight, likelihood=likelihood)["metrics"]

    for branch in ("base", "full"):
        assert float(metrics[f"nll_{branch}_block"]) == pytest.approx(
            _hand_block(model, out, signal, weight, branch=branch, likelihood=likelihood),
            rel=1e-6,
        ), branch
    assert torch.equal(
        metrics["pred_gap"], metrics["nll_base_block"] - metrics["nll_full_block"]
    )
    assert math.isfinite(float(metrics["pred_gap"]))
    assert float(metrics["pred_gap"]) != 0.0


# =================================================================================================
# The mask, multiplicatively
# =================================================================================================
def test_a_planted_value_at_a_padded_anchor_slot_moves_the_loss_by_exactly_zero() -> None:
    r"""The padding convention's whole justification, asserted at the loss.

    A padded slot repeats its row's last real anchor, so its forecast row is a *duplicate* of a live
    one -- and if the mask did not multiply ``anchor_valid`` in, that raw block would be scored twice
    by the reconstruction while the KL support, which is a set, counted it once.

    Exactly zero, not approximately: the mask is multiplicative, so an absurd value at a padded slot
    leaves every reported number bitwise unchanged.
    """
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    weight = _weight(model)

    padded = ~out["anchor_valid"]
    assert bool(padded.any()), "no padded slot in this batch; the probe would be vacuous"

    reference = model.compute_loss(out, signal, weight=weight)["metrics"]
    planted = dict(out)
    for key in ("mu_base", "mu_full", "logvar_base", "logvar_full"):
        tensor = out[key].clone()
        tensor[padded] = 1.0e9
        planted[key] = tensor
    moved = model.compute_loss(planted, signal, weight=weight)["metrics"]

    differing = [
        name for name, value in reference.items() if not torch.equal(value, moved[name])
    ]
    assert not differing, differing

    # Not vacuous: the same plant at a *live* slot moves the loss by a lot.
    live = out["anchor_valid"].clone()
    live[:, 1:] = False
    at_live = dict(out, mu_full=out["mu_full"].clone())
    at_live["mu_full"][live] = 1.0e9
    assert not torch.equal(
        reference["nll_full_block"],
        model.compute_loss(at_live, signal, weight=weight)["metrics"]["nll_full_block"],
    )


def test_a_planted_value_at_a_masked_raw_sample_moves_the_loss_by_exactly_zero() -> None:
    r"""The other half of the multiplicative-mask claim, on the target signal rather than the
    forecast.

    The only route by which the raw signal enters the loss is the anchored gather, so a planted
    $10^{9}$ among the samples no surviving anchor scores must leave every reported number bitwise
    unchanged.
    """
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    # A step the first decoded anchor's window covers. Zeroing its weight drops that anchor *whole*
    # -- its coverage falls below the floor -- so the planted samples are scored by nothing.
    # Decimated step $s$ occupies raw samples $[sD, (s+1)D)$.
    gap_step = int(out["anchor_index"][0, 0]) + 2
    weight = _weight(model, gap_step=gap_step)
    decimation = model.geometry.decimation

    reference = model.compute_loss(out, signal, weight=weight)["metrics"]
    planted = signal.clone()
    planted[:, gap_step * decimation : (gap_step + 1) * decimation] = 1.0e9
    moved = model.compute_loss(out, planted, weight=weight)["metrics"]

    differing = [
        name for name, value in reference.items() if not torch.equal(value, moved[name])
    ]
    assert not differing, differing

    # Not vacuous: a step the *second* decoded anchor reads, which the gap does not touch.
    live_step = int(out["anchor_index"][0, 1]) + 1
    unmasked = signal.clone()
    unmasked[:, live_step * decimation : (live_step + 1) * decimation] = 1.0e9
    assert not torch.equal(
        reference["nll_full_block"],
        model.compute_loss(out, unmasked, weight=weight)["metrics"]["nll_full_block"],
    )


def test_every_reported_number_is_finite_on_a_batch_with_no_valid_step() -> None:
    """A validation batch can be wholly invalid, and a NaN there poisons an epoch aggregate rather
    than raising. Every denominator in the objective is floored for exactly this case, and
    ``anchors_per_sample`` is counted off ``anchor_valid`` rather than off the mask, so it still
    reports the anchors the forward *built* instead of collapsing with the mask."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    metrics = model.compute_loss(
        out, signal, weight=_weight(model, value=0.0)
    )["metrics"]

    nonfinite = [name for name, value in metrics.items() if not bool(torch.isfinite(value))]
    assert not nonfinite, nonfinite
    assert float(metrics["nll_full_block"]) == 0.0
    assert float(metrics["anchors_per_sample"]) == float(out["anchor_valid"].sum()) / BATCH


# =================================================================================================
# The refusals
# =================================================================================================
def test_a_weighted_boundary_term_is_refused_naming_it_and_the_anchor_set() -> None:
    r"""``masked_boundary_gap`` identifies anchor $t$'s last observed sample with a slice of anchor
    $t-1$'s target block, which is a slicing identity only while the anchor axis is contiguous. On a
    tile the two rows are $S$ steps apart, so the term would compare unrelated samples -- silently,
    since every shape still lines up."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    with pytest.raises(ValueError) as error:
        model.compute_loss(out, signal, weight=_weight(model), lambda_boundary=0.1)

    message = str(error.value)
    assert "lambda_boundary" in message and "anchor_index" in message


def test_the_dense_range_is_not_the_none_anchor_set_and_the_objective_says_so() -> None:
    r"""``anchors=None`` means $[0, T_{\mathrm{valid}})$ -- the range a model decoding *every* anchor
    emits -- which is $F$ entries longer than this family's densest set, so a forward dict stripped
    of its anchor set builds a target that does not even have the forecast's shape. A shape refusal,
    not a value difference."""
    model, streams, signal = _tiled(stride=1)
    out = _forward(model, streams, None)

    assert out["anchor_index"].shape[1] == model.geometry.t_valid - model.warmup_period
    assert out["anchor_index"].shape[1] != model.geometry.t_valid

    stripped = {key: value for key, value in out.items() if not key.startswith("anchor_")}
    with pytest.raises(RuntimeError):
        model.compute_loss(stripped, signal, weight=_weight(model))


# =================================================================================================
# The tiling, isolated
# =================================================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_dense_stride_is_the_shared_objective_given_the_same_anchors(
    perturb_posterior, likelihood
) -> None:
    r"""At ``anchor_stride: 1`` the model decodes exactly $[F, T_{\mathrm{valid}})$, and scoring that
    forward through the **shared** objective with the same index supplied explicitly -- at
    ``block_width = geometry.r`` and a target gathered here rather than by the model -- reproduces
    every metric bitwise.

    This is what isolates the tiling: the delegation gathers the raw window at the decoded anchors,
    adds the three readouts of this input domain, and changes nothing whatever about the objective it
    delegates to.
    """
    model, streams, signal = _tiled(stride=1)
    perturb_posterior(model)
    out = _forward(model, streams, None)
    weight = _weight(model)

    dense = torch.arange(model.warmup_period, model.geometry.t_valid)
    explicit = dense[None, :].expand(BATCH, -1).contiguous()
    assert torch.equal(out["anchor_index"], explicit)
    assert bool(out["anchor_valid"].all())

    through_model = model.compute_loss(
        out, signal, weight=weight, likelihood=likelihood
    )["metrics"]
    reference = compute_shared_objective(
        dict(
            out,
            anchor_index=explicit,
            anchor_valid=torch.ones_like(explicit, dtype=torch.bool),
        ),
        gather_anchored_future_target(
            signal, model.geometry, explicit, future_index=model.future_index
        ),
        weight=weight,
        geometry=model.geometry,
        block_width=model.geometry.r,
        coverage_floor=model.coverage_floor,
        logvar_clamp=model.logvar_clamp,
        likelihood=likelihood,
    )["metrics"]

    assert set(through_model) - set(reference) == _ADDED_METRIC_KEYS
    assert set(reference) - set(through_model) == set()
    differing = [
        name
        for name, value in reference.items()
        if not torch.equal(value, through_model[name])
    ]
    assert not differing, differing
