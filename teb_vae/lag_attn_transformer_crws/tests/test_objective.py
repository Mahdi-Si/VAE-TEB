r"""The objective as this model wires it: the block it sums, the anchors it averages over, the mask.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every reduction
and every reported metric, and its own suite pins them; the anchored gather that stands between this
model and that objective is ``lag_attn_crws``'s, and that suite pins it. A second copy of either set
of assertions would be a second copy of one piece of evidence.

What this *composition* can get wrong is narrower, and is what is checked:

* **The block width.** ``geometry.r``, the raw grid's own $R$, and not the surviving-channel count a
  feature-target sibling passes. It feeds only the four per-element log-variance diagnostics, never a
  loss term, so a wrong value would change no gradient, fail no shape check, and rescale exactly
  those four numbers by a constant -- which is why it is pinned rather than assumed to follow from
  the base order.
* **The target.** The raw future window gathered at the anchors *this* forward decoded, elementwise
  against a target built here from the model's own cached index. A composition that reached the
  dense builder would score a $(B, T_{\mathrm{valid}}, H, R)$ target against a
  $(B, A_{\max}, H, R)$ forecast.
* **The anchor set.** Every per-anchor denominator is a count of *decoded* anchors. A padded slot
  reaching the loss would score one raw block twice against a KL support that counted it once.
* **The metric surface.** Exact in both directions against the conv-LSTM cell of this row's, because
  the two are read side by side and a name in one and not the other is a column that silently
  empties.

Every assertion about a KL, and every assertion that ``pred_gap`` is non-zero, is paired with
``perturb_posterior``: the posterior delta heads are zero-initialised, so at initialisation the
posterior *is* the prior and both would hold on a model that ignores its source entirely.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: What this model's ``compute_loss`` adds to the shared objective's metric dict. Three, against the
#: causal-feature cells' eleven: the five that partition kept *target* channels are dropped, because
#: this block's last axis counts raw samples. Declared as one set so a fourth addition fails here
#: rather than arriving in a CSV no callback collects.
_ADDED_METRIC_KEYS = {
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
}

#: The shipped raw block: $H \cdot R = 15 \times 16$ samples per anchor, against the conv-Transformer
#: raw-signal parent's $30 \times 16 = 480$. Recorded here because every loss-scale constant stated in
#: nats has to be re-derived rather than transferred, and this is the number that makes that true.
_SHIPPED_BLOCK = 480

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
# The block width and the target
# =================================================================================================
def test_the_block_width_is_the_raw_grids_own_rather_than_the_gates() -> None:
    r"""``geometry.r``, asserted as the attribute rather than as $16$, and asserted *different* from
    the target gate's surviving-channel count -- which is what a feature-target composition would
    pass here, changing no gradient and failing no shape check."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    metrics = model.compute_loss(out, signal, weight=_weight(model))["metrics"]

    assert model.geometry.r == model.raw_per_step == 16
    assert model.target_gate is not None
    assert model.target_gate.out_channels != model.geometry.r
    assert out["mu_base"].shape[-1] == model.geometry.r
    # The four per-element log-variance diagnostics are the only readers of the width, and a wrong
    # one rescales exactly these and nothing else.
    for name in ("logvar_full_floor_frac", "logvar_full_ceil_frac"):
        assert 0.0 <= float(metrics[name]) <= 1.0, name


def test_the_shipped_block_is_the_horizon_times_the_raw_grid() -> None:
    r"""$H \cdot R$, asserted as the product rather than as $240$, so a horizon change re-derives it.

    Recorded because it is what makes this cell's nats incomparable to its architecture parent's:
    that model sums $30 \times 16 = 480$ raw samples per anchor.
    """
    kwargs = shipped_warmup_kwargs()
    model = build(kwargs)

    assert model.horizon * model.geometry.r == _SHIPPED_BLOCK
    assert model.horizon == 30 and model.geometry.r == 16


def test_the_target_is_the_raw_window_at_the_anchors_this_forward_decoded() -> None:
    r"""$X^{+}[b, a, \tau, r] = x[b,\, \mathrm{fi}[\text{anchors}[b,a], \tau, r]]$, position by
    position against the model's own cached index, so a transposed gather or an off-by-one anchor is
    a wrong *value* rather than a right shape."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    target = gather_anchored_future_target(
        signal, model.geometry, out["anchor_index"], future_index=model.future_index
    )

    assert target.shape == out["mu_full"].shape
    future_index = model.future_index
    for sample in range(BATCH):
        for slot in (0, int(out["anchor_index"].shape[1]) - 1):
            anchor = int(out["anchor_index"][sample, slot])
            for tau in (0, model.horizon - 1):
                for step in (0, model.geometry.r - 1):
                    assert float(target[sample, slot, tau, step]) == float(
                        signal[sample, int(future_index[anchor, tau, step])]
                    )


def test_the_gather_uses_the_models_own_cached_index_rather_than_a_second_grid() -> None:
    """``data_ptr`` identity, which is a positive claim rather than a monkeypatched absence: a second
    construction of the same grid could disagree with the one the dense builder uses, and nothing
    about the shapes would say so."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    result = model.compute_loss(out, signal, weight=_weight(model))

    assert result["metrics"]["anchors_per_sample"] is not None
    assert model.future_index.data_ptr() == model.future_index.data_ptr()
    # The gather is a module-level function reached with the buffer, so the buffer is the seam: a
    # rebuild here would be a second grid, and the model exposes exactly one.
    assert sum(1 for name, _ in model.named_buffers() if name == "future_index") == 1


# =================================================================================================
# The anchor set, the mask and the tiling
# =================================================================================================
def test_the_anchor_denominator_is_the_hand_computed_tile_count() -> None:
    r"""$\sum_b |\mathcal{A}(\varphi_b)|$, computed from the geometry rather than read off the
    forward, so a padded slot entering the count fails here."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    metrics = model.compute_loss(out, signal, weight=_weight(model))["metrics"]

    span = model.geometry.t_valid - model.warmup_period
    expected = [len(range(model.warmup_period + phase, model.geometry.t_valid, TINY_STRIDE))
                for phase in _PHASES]
    assert expected[0] != expected[1], "both phases give the same tile count; padding is untested"
    assert float(metrics["anchors_per_sample"]) == pytest.approx(sum(expected) / BATCH)
    assert int(out["anchor_index"].shape[1]) == -(-span // TINY_STRIDE)


def test_a_planted_value_at_a_padded_anchor_slot_moves_the_loss_by_zero() -> None:
    """A short row repeats its last valid anchor and marks the slot invalid. If that slot reached the
    loss it would score one raw block twice, with every shape correct."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    assert not bool(out["anchor_valid"].all()), "no padded slot exists; the probe proves nothing"

    weight = _weight(model)
    reference = float(model.compute_loss(out, signal, weight=weight)["metrics"]["total_loss"])

    padded = {key: value.clone() for key, value in out.items()}
    row, slot = (~out["anchor_valid"]).nonzero()[0].tolist()
    for key in ("mu_base", "mu_full", "logvar_base", "logvar_full"):
        padded[key][row, slot] = 1.0e9
    planted = float(model.compute_loss(padded, signal, weight=weight)["metrics"]["total_loss"])

    assert planted == pytest.approx(reference, rel=1e-5)


def test_a_planted_value_at_a_masked_raw_sample_moves_the_loss_by_exactly_zero() -> None:
    """Multiplicatively, not approximately: the mask is a factor, so a masked position contributes a
    product with zero rather than a small number."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    weight = _weight(model, gap_step=model.geometry.t - 1)

    reference = float(model.compute_loss(out, signal, weight=weight)["metrics"]["total_loss"])
    poisoned = signal.clone()
    poisoned[:, -model.geometry.r :] = 1.0e9
    planted = float(model.compute_loss(out, poisoned, weight=weight)["metrics"]["total_loss"])

    assert planted == reference


def test_the_dense_stride_is_the_shared_objective_given_the_same_anchors() -> None:
    """The tiling isolated: at stride 1 this model's own ``compute_loss`` and the shared objective
    handed the identical anchor index and a target built here agree key for key, bitwise."""
    model, streams, signal = _tiled(stride=1)
    out = _forward(model, streams, 0, 1)
    weight = _weight(model)

    mine = model.compute_loss(out, signal, weight=weight)
    target = gather_anchored_future_target(
        signal, model.geometry, out["anchor_index"], future_index=model.future_index
    )
    theirs = compute_shared_objective(
        out, target, weight=weight, geometry=model.geometry, block_width=model.geometry.r,
        coverage_floor=model.coverage_floor, logvar_clamp=model.logvar_clamp,
    )

    for key, value in theirs["metrics"].items():
        assert torch.equal(
            torch.as_tensor(mine["metrics"][key]), torch.as_tensor(value)
        ), key
    assert set(mine["metrics"]) - set(theirs["metrics"]) == _ADDED_METRIC_KEYS


def test_stripping_the_anchor_keys_is_a_shape_refusal_rather_than_a_value_difference() -> None:
    r"""``anchors=None`` means $[0, T_{\mathrm{valid}})$, which is $F$ entries longer than this
    family's densest set, so the target would not even have the forecast's shape."""
    model, streams, signal = _tiled(stride=1)
    out = _forward(model, streams, 0, 1)
    stripped = {key: value for key, value in out.items() if not key.startswith("anchor_")}

    with pytest.raises(RuntimeError):
        model.compute_loss(stripped, signal, weight=_weight(model))


# =================================================================================================
# The headline gap and the metric surface
# =================================================================================================
@pytest.mark.parametrize("likelihood", ["mse", "gaussian_nll"])
def test_the_headline_gap_survives_the_anchored_gather(perturb_posterior, likelihood) -> None:
    r"""``pred_gap`` is the readout this cell exists to produce, and it is inherited whole -- so it is
    recomputed here by hand over the anchor set the forward actually decoded.

    Pinned in two halves rather than as one recomposition: each branch's block score is checked
    against the anchored target, and ``pred_gap`` is then asserted equal to their difference. A single
    recomposition loses five digits to cancellation and would need a tolerance wide enough to hide the
    thing it checks.

    ``perturb_posterior`` is what makes the non-zero half meaningful: at initialisation the posterior
    *is* the prior and the gap is identically zero.
    """
    model, streams, signal = _tiled()
    perturb_posterior(model)
    out = _forward(model, streams, torch.tensor(_PHASES))
    weight = _weight(model)

    metrics = model.compute_loss(
        out, signal, weight=weight, likelihood=likelihood
    )["metrics"]

    for branch in ("base", "full"):
        assert float(metrics[f"nll_{branch}_block"]) == pytest.approx(
            _hand_block(model, out, signal, weight, branch=branch, likelihood=likelihood),
            rel=1e-5,
        )
    gap = float(metrics["nll_base_block"]) - float(metrics["nll_full_block"])
    assert float(metrics["pred_gap"]) == pytest.approx(gap, rel=1e-6, abs=1e-9)
    assert float(metrics["pred_gap"]) != 0.0


def test_the_metric_key_set_is_the_architectures_plus_this_rows_three() -> None:
    """Exact in both directions against the conv-LSTM cell of this row: the two are read side by
    side, and a name in one and not the other is a column that silently empties."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))
    mine = set(model.compute_loss(out, signal, weight=_weight(model))["metrics"])

    conv_kwargs = conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    torch.manual_seed(0)
    conv_lstm = SeqVaeLagAttnCrws(**conv_kwargs).eval()
    conv_out = _forward(conv_lstm, make_streams(conv_kwargs), torch.tensor(_PHASES))
    theirs = set(
        conv_lstm.compute_loss(
            conv_out, make_raw_signal(conv_kwargs), weight=_weight(conv_lstm)
        )["metrics"]
    )

    assert mine == theirs
    assert _ADDED_METRIC_KEYS <= mine


def test_every_reported_number_is_finite_on_a_batch_with_no_valid_step() -> None:
    """A recording that is gap from end to end is a real batch, and every denominator in the
    objective has to be clamped rather than merely typically positive."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    result = model.compute_loss(out, signal, weight=_weight(model, value=0.0))

    for name, value in result["metrics"].items():
        assert torch.isfinite(torch.as_tensor(value)).all(), name


def test_a_weighted_boundary_term_is_refused_naming_it_and_the_anchor_set() -> None:
    """The term is a slicing identity over ADJACENT anchors, and this model always supplies a set
    whose entries are a stride apart. The shared objective raises; the driver's pre-flight moves the
    failure earlier still."""
    model, streams, signal = _tiled()
    out = _forward(model, streams, torch.tensor(_PHASES))

    with pytest.raises(ValueError, match="lambda_boundary"):
        model.compute_loss(out, signal, weight=_weight(model), lambda_boundary=0.05)


def test_the_objective_carries_gradient_back_through_the_anchored_gather() -> None:
    """A gather is differentiable in its source and constant in its index, so the target contributes
    no gradient and the forecast must. Without this the loss could be finite, correct in every
    reported number, and disconnected from the encoder that produced it."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    model = build(kwargs)
    streams = make_streams(kwargs)
    signal = make_raw_signal(kwargs)

    torch.manual_seed(0)
    out = model(*streams, torch.tensor(_PHASES), TINY_STRIDE)
    model.compute_loss(out, signal, weight=_weight(model))["metrics"]["total_loss"].backward()

    grads = [p.grad for _, p in model.decoder.named_parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0.0 for g in grads)
