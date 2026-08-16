r"""The objective as this cell wires it: the block it sums, the anchors it averages over, the mask.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every reduction
and every reported metric, and its own suite pins them; a second copy of those assertions would be a
second copy of one piece of evidence. What *this* cell can get wrong is narrower, and is what is
checked:

* **The block width.** ``geometry.r``, the raw grid's own $R$, and not the surviving-channel count a
  feature-target sibling passes. It feeds only the four per-element log-variance diagnostics, never a
  loss term, so a wrong value would change no gradient, fail no shape check, and rescale exactly
  those four numbers by a constant.
* **The anchor set.** Every per-anchor denominator is a count of *decoded* anchors, and this cell's
  decoded set is a tile rather than the dense range. A padded slot reaching the loss would score one
  raw block twice against a KL support that counted it once.
* **The mask.** A value planted where the mask is zero must move the loss by *exactly* zero --
  multiplicatively, not approximately.
* **The headline gap.** ``pred_gap`` is the readout this cell exists to produce: the same quantity
  ``lag_attn_rws`` reports, over the same raw target, from inputs that do not contain the answer. It
  is inherited whole from the shared objective, so nothing else in this package tests it, and it is
  recomputed here by hand over the anchor set the forward actually decoded.

The tiling itself is isolated by scoring one forward two ways: through this model's own
``compute_loss``, and through the **shared** objective given the same anchor index explicitly against
a target built here rather than by the model. Comparing against ``anchors=None`` instead is not
available, and the reason is worth stating rather than discovering: ``None`` means the dense range
$[0, T_{\mathrm{valid}})$, which is $F$ entries *longer* than this family's densest set, so the
target would not even have the forecast's shape.

Every assertion about a KL, and every assertion that ``pred_gap`` is non-zero, is paired with
``perturb_posterior``: the posterior delta heads are zero-initialised, so at initialisation the
posterior *is* the prior and both would hold on a model that ignores its source entirely.
"""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

from .conftest import (
    BATCH,
    TINY_KWARGS,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: What this cell's ``compute_loss`` adds to the shared objective's metric dict. Three, against the
#: causal-feature cells' eleven: the five that partition kept *target* channels are dropped, because
#: this block's last axis counts raw samples. Declared as one set so a fourth addition fails here
#: rather than arriving in a CSV no callback collects.
_ADDED_METRIC_KEYS = {
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
}

#: The shipped raw block: $H \cdot R = 15 \times 16$ samples per anchor, against the two-sided
#: sibling's $30 \times 16 = 480$. Recorded here because every loss-scale constant stated in nats has
#: to be re-derived rather than transferred, and this is the number that makes that true.
_SHIPPED_BLOCK = 240

#: The two phases the tiled fixtures run at. Chosen so the second row is one anchor short of the
#: first, which is the only way a padded slot exists at all -- and every padding assertion below
#: would pass vacuously without one.
_PHASES = (0, TINY_STRIDE - 1)

#: The three shape terms, and the weights that switch them off.
_SHAPE_TERMS = ("aux_multiscale", "aux_derivative", "aux_boundary")
_SHAPE_WEIGHTS = ("lambda_ms", "lambda_deriv", "lambda_boundary")


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
# The block the reconstruction sums
# =================================================================================================
def test_the_sample_score_divides_the_block_by_the_horizon_times_the_raw_grid() -> None:
    r"""$H \cdot R$, asserted as the product rather than as $240$, so a horizon change re-derives it.

    The rescaling is the only place the two reconstruction columns differ, and it is what makes
    ``nll_*_sample`` comparable across cells while ``nll_*_block`` is not: the shipped block here is
    $15 \times 16 = 240$ raw samples against the two-sided sibling's $480$ and the causal-feature
    cell's $1470$ coefficients, so a nat from this configuration is comparable to no sibling at all.
    """
    kwargs = shipped_warmup_kwargs()
    model = build(kwargs).eval()
    streams = make_streams(kwargs)
    out = _forward(model, streams, torch.zeros(BATCH, dtype=torch.long))

    metrics = model.compute_loss(
        out, make_raw_signal(kwargs), weight=_weight(model)
    )["metrics"]

    block = model.horizon * model.geometry.r
    for branch in ("full", "base"):
        assert float(metrics[f"nll_{branch}_sample"]) == pytest.approx(
            float(metrics[f"nll_{branch}_block"]) / block, rel=1e-6
        ), branch
    assert block == _SHIPPED_BLOCK
    # The causal-feature cell's block at the same horizon and the same inputs, for the same anchors:
    # a different quantity by a factor of 6.125, which is why no constant transfers between them.
    assert 15 * 98 / block == pytest.approx(6.125)


def test_the_block_width_is_the_raw_grids_own_and_the_logvar_diagnostics_follow_it() -> None:
    r"""``block_width`` is ``geometry.r``, and it is observable rather than asserted at the call site.

    It reaches only the four per-element log-variance diagnostics, so a wrong value changes no
    gradient and fails no shape check -- it rescales exactly those four numbers and nothing else. The
    check is therefore a *difference*: the same forward scored through the shared objective at twice
    the width must halve exactly those four columns and leave every other one bitwise where it was.
    """
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    weight = _weight(model)
    target = gather_anchored_future_target(
        signal, model.geometry, out["anchor_index"], future_index=model.future_index
    )

    def _through(width: int):
        return compute_shared_objective(
            out,
            target,
            weight=weight,
            geometry=model.geometry,
            block_width=width,
            coverage_floor=model.coverage_floor,
            logvar_clamp=model.logvar_clamp,
        )["metrics"]

    reported = model.compute_loss(out, signal, weight=weight)["metrics"]
    at_r = _through(model.geometry.r)
    at_double = _through(2 * model.geometry.r)

    assert model.geometry.r == int(TINY_KWARGS["raw_per_step"])
    scaled = {
        "mean_logvar_full",
        "mean_logvar_base",
        "logvar_full_floor_frac",
        "logvar_full_ceil_frac",
    }
    for name, value in at_r.items():
        # The model's own call is the one that has to be at geometry.r; the doubled call is only
        # here to prove those four columns are the ones the width moves.
        assert torch.equal(reported[name], value), name
        if name in scaled:
            assert float(at_double[name]) == pytest.approx(float(value) / 2.0, rel=1e-6), name
        else:
            assert torch.equal(at_double[name], value), name


def test_the_block_score_is_the_masked_sum_reduced_over_the_decoded_anchors() -> None:
    r"""$\sum_{b,a} m_{b,a} \sum_{\tau,r} s_{b,a,\tau,r}$ over $\sum_{b,a} c_{b,a}$, by hand.

    The denominator is the one thing a tiled objective can get wrong with no shape changing --
    ``contributing_anchors`` reduces the last axis alone, so a mask of another rank would inflate
    every per-anchor denominator silently -- and it is what makes every reported number nats *per
    anchor* rather than nats per batch.
    """
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    weight = _weight(model)

    metrics = model.compute_loss(out, signal, weight=weight)["metrics"]

    for branch in ("base", "full"):
        assert float(metrics[f"nll_{branch}_block"]) == pytest.approx(
            _hand_block(
                model, out, signal, weight, branch=branch, likelihood="gaussian_nll"
            ),
            rel=1e-6,
        ), branch


def test_the_anchor_denominator_is_the_hand_computed_tile_count() -> None:
    r"""$\sum_b |\mathcal A(\varphi_b)|$, computed here from the geometry rather than read off the
    model: $\lceil (T_{\mathrm{valid}} - F - \varphi)/S \rceil$ per sample."""
    model, streams, _signal = _tiled()
    phase = torch.tensor(_PHASES)
    out = _forward(model, streams, phase)

    span = model.geometry.t_valid - model.warmup_period
    expected = [-(-(span - int(value)) // TINY_STRIDE) for value in phase]
    assert len(set(expected)) == 2, "the fixture no longer produces a short row"

    mask, _coverage = forecast_mask(
        _weight(model),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=out["anchor_index"],
        anchor_valid=out["anchor_valid"],
    )

    assert int(out["anchor_valid"].sum()) == sum(expected)
    assert int(contributing_anchors(mask).sum()) == sum(expected)


# =================================================================================================
# The headline gap
# =================================================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_headline_gap_survives_the_anchored_gather(perturb_posterior, likelihood) -> None:
    r"""$D_0 - D_1$ over the anchors the forward decoded, recomputed here from the mask up.

    This is the readout the cell exists to produce -- the same quantity ``lag_attn_rws`` reports over
    the same raw target, from inputs that carry no future -- and it is inherited whole, so nothing
    else in this package tests it. The claim is split in two so neither half hides in the other's
    cancellation: each branch's block score is recomputed independently against the anchored target,
    and the gap is then the exact difference of the two columns beside it.

    Asserted **after** ``perturb_posterior``, and paired with the zero it reads at initialisation:
    the posterior delta heads are zero-initialised, so a fresh model has $D_0 = D_1$ identically and
    a "the gap is finite" check would hold on a model that ignores its source entirely.
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
    by the reconstruction while the KL support, which is a set, counted it once. The two per-anchor
    denominators would diverge and $\beta$ would quietly stop meaning what it means in every other
    cell of the grid.

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
    unchanged. Exactness is the point: an additive mask would leave $0 \times 10^{9}$, which is $0$
    in float -- but $10^{9}$ inside a squared error would already have overflowed the sum before the
    mask was applied.
    """
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    # A step the first decoded anchor's window covers. Zeroing its weight drops that anchor *whole*
    # -- one invalid step of four leaves its coverage below the floor -- so the planted samples are
    # scored by nothing. Decimated step $s$ occupies raw samples $[sD, (s+1)D)$.
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
    than raising. Every denominator in the objective is floored for exactly this case, and the three
    readouts merged onto it are counted off the *decoded* set, so they report the geometry rather
    than collapsing with the mask."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    metrics = model.compute_loss(
        out, signal, weight=_weight(model, value=0.0)
    )["metrics"]

    nonfinite = [name for name, value in metrics.items() if not bool(torch.isfinite(value))]
    assert not nonfinite, nonfinite
    assert float(metrics["nll_full_block"]) == 0.0
    assert float(metrics["anchors_per_sample"]) > 0.0


# =================================================================================================
# The shape terms, off
# =================================================================================================
def test_the_three_shape_terms_ship_off_and_report_exact_zeros() -> None:
    """Off at the signature, not merely off in a config, so a caller that names none of them gets
    the three-term objective the rest of the family is read against.

    Two of the three are meaningful on a raw block -- the flattened axis really is the recording's
    own time axis -- so this is a shipped-default claim rather than a domain one; the third is
    refused outright below.
    """
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    defaults = inspect.signature(SeqVaeLagAttnCrws.compute_loss).parameters
    metrics = model.compute_loss(out, signal, weight=_weight(model))["metrics"]

    for name in _SHAPE_WEIGHTS:
        assert defaults[name].default == 0.0, name
        assert float(metrics[name]) == 0.0, name
    for name in _SHAPE_TERMS:
        assert float(metrics[name]) == 0.0, name


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


# =================================================================================================
# The tiling, isolated
# =================================================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_dense_stride_is_the_shared_objective_given_the_same_anchors(
    perturb_posterior, likelihood
) -> None:
    r"""At ``anchor_stride: 1`` the model decodes exactly $[F, T_{\mathrm{valid}})$, and scoring that
    forward through the **shared** objective with the same index supplied explicitly -- and a target
    gathered here rather than by the model -- reproduces every metric bitwise.

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


def test_the_dense_range_is_not_the_none_anchor_set_and_the_objective_says_so() -> None:
    r"""Why the reference above supplies the index explicitly. ``anchors=None`` means
    $[0, T_{\mathrm{valid}})$ -- the range a model decoding *every* anchor emits -- which is $F$
    entries longer than this family's densest set, so the target it builds does not even have the
    forecast's shape. A shape refusal, not a value difference."""
    model, streams, signal = _tiled(stride=1)
    out = _forward(model, streams, None)

    assert out["anchor_index"].shape[1] == model.geometry.t_valid - model.warmup_period
    assert out["anchor_index"].shape[1] != model.geometry.t_valid

    stripped = {key: value for key, value in out.items() if not key.startswith("anchor_")}
    with pytest.raises(RuntimeError):
        model.compute_loss(stripped, signal, weight=_weight(model))


# =================================================================================================
# The metric surface
# =================================================================================================
def test_the_metric_key_set_is_the_raw_siblings_plus_this_packages_three() -> None:
    """Exact in both directions, against a declared addition rather than a free one. Every
    downstream reader is keyed by name, so a name in one model and not the other is a column that
    silently empties -- and a fourth addition arriving unannounced would be a readout no callback
    collects."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    causal = model.compute_loss(out, signal, weight=_weight(model))["metrics"]

    torch.manual_seed(0)
    raw = SeqVaeLagAttnRws(**dict(TINY_KWARGS)).eval()
    raw_streams = make_streams(TINY_KWARGS)
    torch.manual_seed(0)
    with torch.no_grad():
        raw_out = raw(*raw_streams)
    raw_metrics = raw.compute_loss(
        raw_out, torch.zeros(BATCH, raw.geometry.raw_len), weight=_weight(raw)
    )["metrics"]

    assert set(causal) - set(raw_metrics) == _ADDED_METRIC_KEYS
    assert set(raw_metrics) - set(causal) == set()
    assert all(isinstance(value, torch.Tensor) for value in causal.values())


def test_the_objective_carries_gradient_through_the_anchored_gather() -> None:
    """A smoke check that the assembled total is trainable *through* the gather, and that the
    gradient reaches the head emitting the raw block."""
    model, streams, signal = _tiled()
    model.train()

    out = model(*streams, torch.tensor(_PHASES))
    result = model.compute_loss(out, signal, weight=_weight(model))
    result["metrics"]["total_loss"].backward()

    assert model.decoder.mean_head.weight.grad is not None
    assert float(model.decoder.mean_head.weight.grad.abs().max()) > 0.0
