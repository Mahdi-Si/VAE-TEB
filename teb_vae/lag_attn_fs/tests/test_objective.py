r"""The objective as this model wires it: the target it builds, and the width it declares.

The arithmetic is not retested here. ``lag_attn_rws/nets/losses.py`` owns every term, every
reduction and every reported metric, and its own suite pins them; this file drives this model's
metrics through that suite's independent reassembly. What is this model's to get wrong is the
wiring, and it has three moving parts:

* **The target.** Gathered from the caller's feature stream by the target gate's keep-index and
  unfolded into each anchor's future window -- never delayed, and never rebuilt from a raw grid.
  Pinned against a slice-and-stack that shares no arithmetic with ``unfold``, and against the
  planted pattern, whose values name the stored step and channel they came from.
* **``block_width``.** $C_{\mathrm{keep}}$, the surviving-channel count. It feeds only the
  per-element log-variance diagnostics, never a loss term, so passing ``geometry.r`` here would
  change no gradient and fail no shape check. The reassembly harness is handed a hand-written
  width, which is what catches that.
* **The four resolved forecast gaps.** Partial sums of ``pred_gap`` by horizon step and by stored
  block, checked against a per-step and per-channel gap assembled here from the objective's own
  primitives, with the block boundary taken from the data rather than from the class constant.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    PATTERN_STEP_SCALE,
    SHIPPED_KWARGS,
    STUB_GAP_STEP,
    TINY_KEEP_INDEX,
    TINY_KWARGS,
    make_patterned_batch,
    make_stub_batch,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask
from teb_vae.lag_attn_rws.tests.test_objective import assert_objective_reassembles

#: Coefficients the recomposition runs at. Mutually distinct and none of them a default: at equal
#: weights a term swapped for another passes, at ``beta_prior=0`` the fourth term is multiplied
#: away, and at ``free_bits=0`` the raw and trained KL are one tensor rather than two.
_COEFFICIENTS = dict(
    beta=0.7, beta_prior=0.11, lambda_full=1.0, lambda_base=0.3, free_bits=0.05
)

#: The four metrics this model reports and the raw-signal sibling does not: ``pred_gap`` resolved
#: by horizon step and split by stored block.
_RESOLVED_GAP_KEYS = {
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
}


def _model(kwargs, cls=SeqVaeLagAttnFs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides)).eval()


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )


def _features(batch) -> torch.Tensor:
    """The concatenated target stream, in the declared block order."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


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


def _loss_batch(dtype=torch.float32):
    """A tiny batch whose feature values are order one, with the stub's deliberate weight gap.

    The planted-pattern batch is the right fixture for questions about *which* coefficient landed
    where, and the wrong one for questions about a loss value: its values run to $15{,}000$, a
    summed squared error over them reaches $10^{10}$, and a float32 recomposition at that
    magnitude fails on round-off rather than on a wiring mistake.

    Args:
        dtype: Dtype of the feature blocks and the weight. The resolved-gap tests run in float64:
            each gap is a small difference of two block-sized sums, and in float32 two summation
            orders of the same perturbed quantity disagree in the third significant digit.

    Returns:
        The batch.
    """
    batch = make_stub_batch()
    for name in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        setattr(batch, name, getattr(batch, name).to(dtype))
    return batch


#: A tiny target guard whose survivors straddle the stored-block boundary and are not contiguous,
#: so splitting the *kept* axis at the declared boundary puts a second-block channel in the first
#: block's total. The tiny suite's own guard keeps first-block channels only.
_STRADDLING_GUARD = dict(target_keep_index=(0, 5, 44, 60), target_delays=(0, 1, 2, 1))


# ---------------------------------------------------------------------------------------
# The target the objective is handed
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("guard", ["shipped-gated", "unguarded"])
def test_the_target_obeys_the_index_identity_at_every_position(guard):
    r"""$Y^{+}[b, t, \tau, k] = Y[b,\, t + 1 + \tau,\, \mathrm{keep}[k]]$, whole block, at the
    production geometry: against a slice-and-stack of the gathered stream, and against the planted
    pattern read back. Element $(0, t, \tau, k)$ of the pattern is
    $(t + 1 + \tau)\,S + \mathrm{keep}[k]$, so dividing by $S$ recovers the *stored step* and the
    remainder the *stored channel* -- which must be $\mathrm{keep}[k]$, not $k$. Unguarded, the
    keep-index is every declared channel."""
    model = _model(shipped_gated_kwargs() if guard == "shipped-gated" else dict(SHIPPED_KWARGS))
    batch = make_patterned_batch(2, SHIPPED_KWARGS["sequence_length"])
    stream = _features(batch)
    keep = (
        torch.arange(model.c_y) if model.target_gate is None else model.target_gate.keep_index
    )

    built = model._build_forecast_target(stream)

    t_valid, horizon = model.geometry.t_valid, model.horizon
    assert torch.equal(
        built, _stacked_block(torch.index_select(stream, -1, keep), horizon, t_valid)
    )
    sample = built[0]
    recovered_step = torch.div(sample, PATTERN_STEP_SCALE, rounding_mode="floor")
    recovered_channel = sample - recovered_step * PATTERN_STEP_SCALE
    expected_step = (
        torch.arange(t_valid).view(-1, 1) + 1 + torch.arange(horizon).view(1, -1)
    ).float()
    assert torch.equal(recovered_step, expected_step.unsqueeze(-1).expand_as(sample))
    assert torch.equal(recovered_channel, keep.float().view(1, 1, -1).expand_as(sample))


def test_the_target_is_not_delayed(tiny_gated):
    """The sharpest correctness trap in the model, asserted where the target is actually built.
    The gate's delays are non-zero and distinct here, so a builder that called the gate would be
    wrong by a different number of steps in each channel -- and nothing downstream would fail."""
    model = _model(tiny_gated)
    batch = make_patterned_batch()
    stream = _features(batch)

    built = model._build_forecast_target(stream)
    delayed = (
        model.target_gate(stream)[:, 1:, :]
        .unfold(dimension=1, size=model.horizon, step=1)
        .permute(0, 1, 3, 2)
    )

    assert built.shape == delayed.shape
    assert not torch.equal(built, delayed)
    # And specifically: the delayed block at anchor t is the correct block at t - delta.
    delays = model.target_gate.delay.delay_steps
    channel = int(torch.argmax(delays))
    shift = int(delays[channel])
    assert shift > 0
    assert torch.equal(delayed[:, 5, :, channel], built[:, 5 - shift, :, channel])


def test_a_named_anchor_set_takes_the_dense_blocks_it_names(tiny_gated):
    """The gathered window, against the rows of the dense block it selects.

    A selection, not a second construction: anchor $t$'s block must be the same tensor whether it
    was built as one of every anchor's or as one of three.
    """
    model = _model(tiny_gated)
    stream = _features(make_patterned_batch())
    dense = model._build_forecast_target(stream)
    anchors = torch.tensor([[2, 6, 9], [3, 6, 11]])

    gathered = model._build_forecast_target(stream, anchors)

    assert gathered.shape == (2, 3, model.horizon, dense.shape[-1])
    for row in range(2):
        for slot, anchor in enumerate(anchors[row].tolist()):
            assert torch.equal(gathered[row, slot], dense[row, anchor])


def test_the_anchored_target_never_unfolds(tiny_gated, monkeypatch):
    r"""No ``unfold`` in the anchored path, asserted by making one fail.

    The two do not compose: ``unfold`` produces *every* window in order, so unfolding first and
    selecting after would materialise the whole $(B, T_{\mathrm{valid}}, H, C)$ block -- a third
    of a gigabyte at the production batch, and exactly the tensor a sparse anchor set exists to
    avoid. The dense path still unfolds, which is why the same patch is checked to bite there.
    """
    model = _model(tiny_gated)
    stream = _features(make_patterned_batch())
    anchors = torch.tensor([[2, 6, 9], [3, 6, 11]])

    def _refuse(*args, **kwargs):
        raise AssertionError("unfold was called")

    monkeypatch.setattr(torch.Tensor, "unfold", _refuse)

    model._build_forecast_target(stream, anchors)  # must not raise
    with pytest.raises(AssertionError, match="unfold was called"):
        model._build_forecast_target(stream)


@pytest.mark.parametrize(
    "shape, match",
    [
        ((2, 16), "3-D"),
        ((2, 20, 109), "trim_minutes"),
        ((2, 16, 78), "c_y=109"),
    ],
    ids=["not-3d", "wrong-length", "already-gathered"],
)
def test_a_target_stream_that_does_not_match_the_geometry_is_refused(
    tiny_kwargs, shape, match
):
    """The third case is the one worth having: a caller that gathered the channels itself would
    hand over a correctly-ranked tensor whose keep-index positions no longer mean what the model
    thinks, and the gather would silently take the wrong channels."""
    model = _model(tiny_kwargs)

    with pytest.raises(ValueError, match=match):
        model._build_forecast_target(torch.zeros(shape))


# ---------------------------------------------------------------------------------------
# The resolved forecast gaps
#
# Four partial sums of ``pred_gap``, and the only reason they exist is that with no evaluation
# pipeline the summed number cannot separate forecasting from reconstruction of the part of the
# target the model's own history already determines. So what is checked is exactly that: each
# equals the matching slice of a per-step or per-channel gap assembled here, per anchor, and the
# block split follows the reach budget's keep-index rather than assuming the survivors are
# contiguous.
# ---------------------------------------------------------------------------------------
def _gap_by_axis(model, outs, batch, likelihood: str):
    r"""The forecast gap by horizon step and by kept channel, from the objective's own primitives.

    Written out rather than read off the model so the reported metrics are checked against a
    quantity this file computed: a curve derived from the same private method that produced the
    metrics would be self-consistent whatever either did.

    Args:
        model: The net.
        outs: Its forward dict.
        batch: The batch the forward was run on.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        ``(by_tau, by_channel)``, shapes $(H,)$ and $(C_{\mathrm{keep}},)$, in nats per anchor.
    """
    target = model._build_forecast_target(_features(batch))
    mask, _coverage = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
    n_anchors = contributing_anchors(mask).to(target.dtype).sum().clamp_min(1.0)
    gap = (
        raw_sample_score(outs["mu_base"], target, likelihood=likelihood, logvar=outs["logvar_base"])
        - raw_sample_score(
            outs["mu_full"], target, likelihood=likelihood, logvar=outs["logvar_full"]
        )
    ) * mask[..., None]
    return gap.sum(dim=(0, 1, 3)) / n_anchors, gap.sum(dim=(0, 1, 2)) / n_anchors


def _gap_case(guard: str, perturb_posterior):
    """A perturbed float64 model and its batch and forward, for the resolved-gap tests.

    Args:
        guard: ``'straddling'`` for :data:`_STRADDLING_GUARD`, ``'unguarded'`` for none.
        perturb_posterior: The perturbation factory fixture.

    Returns:
        ``(model, batch, outs)``.
    """
    kwargs = dict(TINY_KWARGS, **(_STRADDLING_GUARD if guard == "straddling" else {}))
    model = _model(kwargs).double()
    perturb_posterior(model)
    batch = _loss_batch(torch.float64)
    return model, batch, _forward(model, batch)


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_horizon_split_recomposes_to_the_reported_gap(perturb_posterior, likelihood):
    r"""Summed over $\tau$ the horizon curve is ``pred_gap``, and its two endpoints are the two
    reported scalars. Perturbed, so the curve is not identically zero and its two ends genuinely
    differ: the endpoints alone would pass for a curve that was not a decomposition of anything,
    and the recomposition alone would pass for endpoints read off the wrong end."""
    model, batch, outs = _gap_case("straddling", perturb_posterior)

    by_tau, _ = _gap_by_axis(model, outs, batch, likelihood)
    metrics = model.compute_loss(
        outs, _features(batch), weight=batch.weight, likelihood=likelihood
    )["metrics"]

    assert by_tau.numel() == model.horizon
    assert float(by_tau[0]) != pytest.approx(float(by_tau[-1]), rel=1e-6)
    assert float(metrics["pred_gap_tau_first"]) == pytest.approx(float(by_tau[0]), rel=1e-9)
    assert float(metrics["pred_gap_tau_last"]) == pytest.approx(float(by_tau[-1]), rel=1e-9)
    assert float(by_tau.sum()) == pytest.approx(float(metrics["pred_gap"]), rel=1e-9)


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
@pytest.mark.parametrize("guard", ["straddling", "unguarded"])
def test_the_block_split_is_the_declared_boundary_over_the_kept_channels(
    perturb_posterior, likelihood, guard
):
    """The other axis, and the one where a wrong split is invisible to recomposition: the two parts
    add back to ``pred_gap`` for **any** partition of the channels. So each part is compared with
    the per-channel gap summed over the kept channels whose *declared* index falls in the first
    stored block -- the boundary read off the batch's first block width, not off the class
    constant. Splitting the kept axis at the declared boundary instead, the natural mistake, fails
    on the straddling guard. Unguarded, every declared channel survives and the split is the two
    blocks' full widths."""
    model, batch, outs = _gap_case(guard, perturb_posterior)
    keep = (
        torch.arange(model.c_y) if model.target_gate is None else model.target_gate.keep_index
    )
    first_block = keep < batch.fhr_st.shape[-1]

    _, by_channel = _gap_by_axis(model, outs, batch, likelihood)
    metrics = model.compute_loss(
        outs, _features(batch), weight=batch.weight, likelihood=likelihood
    )["metrics"]

    assert 0 < int(first_block.sum()) < keep.numel()
    assert float(metrics["pred_gap_st"]) == pytest.approx(
        float(by_channel[first_block].sum()), rel=1e-9
    )
    assert float(metrics["pred_gap_ph"]) == pytest.approx(
        float(by_channel[~first_block].sum()), rel=1e-9
    )
    assert float(metrics["pred_gap_st"]) + float(metrics["pred_gap_ph"]) == pytest.approx(
        float(metrics["pred_gap"]), rel=1e-9
    )


def test_the_resolved_gaps_carry_no_gradient(tiny_gated, perturb_posterior):
    """They are diagnostics, computed under ``no_grad``: a term that held the graph would keep a
    block-sized score tensor alive for every step it was logged."""
    model = _model(tiny_gated)
    perturb_posterior(model)
    batch = _loss_batch()

    outs = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    metrics = model.compute_loss(outs, _features(batch), weight=batch.weight)["metrics"]

    for name in sorted(_RESOLVED_GAP_KEYS):
        assert not metrics[name].requires_grad, name


# ---------------------------------------------------------------------------------------
# The loss path
# ---------------------------------------------------------------------------------------
def test_an_unknown_likelihood_is_rejected_listing_the_choices(tiny_kwargs):
    model = _model(tiny_kwargs)
    batch = _loss_batch()

    with pytest.raises(ValueError, match=r"mse.*gaussian_nll"):
        model.compute_loss(
            _forward(model, batch), _features(batch), weight=batch.weight, likelihood="huber"
        )


def test_the_objective_carries_gradient(tiny_gated, perturb_posterior):
    """A smoke check that the assembled total is trainable, and that the gradient reaches the
    decoder head whose width this model changed."""
    model = _model(tiny_gated).train()
    perturb_posterior(model)
    batch = _loss_batch()

    out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    result = model.compute_loss(out, _features(batch), weight=batch.weight)
    result["metrics"]["total_loss"].backward()

    assert model.decoder.mean_head.weight.grad is not None
    assert float(model.decoder.mean_head.weight.grad.abs().max()) > 0.0


def test_a_gapped_step_is_invisible_end_to_end(tiny_gated):
    """The only route by which the target stream enters the loss is the gather, so planting an
    absurd value at the gapped step must leave every reconstruction number bitwise unchanged.

    Multiplicative masking is what makes this exact rather than merely small: an additive mask
    would leave $10^{9}$ contributing $0 \\times 10^{9}$ in float, which is $0$, but $10^{9}$
    inside a squared error would already have overflowed the sum before the mask was applied."""
    model = _model(tiny_gated)
    batch = _loss_batch()
    out = _forward(model, batch)
    stream = _features(batch)

    reference = model.compute_loss(out, stream, weight=batch.weight)
    planted = stream.clone()
    planted[:, STUB_GAP_STEP, :] = 1.0e9
    result = model.compute_loss(out, planted, weight=batch.weight)

    differing = [
        key
        for key, value in reference["metrics"].items()
        if not torch.equal(value, result["metrics"][key])
    ]
    assert not differing, differing

    # Not vacuous: an unmasked step moves the loss by a lot. Step 5, not the gap's neighbour --
    # at the tiny horizon and coverage_floor = 0.9 the anchors whose window covers the gap are
    # dropped *whole*, so the steps just past the gap are unscored too and planting there would
    # prove nothing.
    unmasked = stream.clone()
    unmasked[:, 5, :] = 1.0e9
    moved = model.compute_loss(out, unmasked, weight=batch.weight)
    assert not torch.equal(
        reference["metrics"]["nll_full_block"], moved["metrics"]["nll_full_block"]
    )


# ---------------------------------------------------------------------------------------
# The shared reassembly harness
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
@pytest.mark.parametrize("guard", ["ungated", "gated"], ids=["ungated", "gated"])
def test_every_metric_reassembles_from_the_primitives(perturb_posterior, likelihood, guard):
    """This model's metrics, against the sibling suite's independent reassembly: the total, the
    per-sample scores, the log-variance diagnostics and the masking, every key, ``torch.equal``.

    What this file supplies is what this model owns: its target, built by the slice-and-stack and
    gathered at the tiny guard's keep-index, its hand-written block width -- which differs from the
    raw grid's $R$ in both arms, so ``block_width`` passed as ``geometry.r`` fails here -- and the
    four resolved forecast gaps as package-owned keys, so an unannounced *fifth* addition fails
    rather than passing unnoticed.
    """
    gated = guard == "gated"
    kwargs = tiny_gated_kwargs() if gated else dict(TINY_KWARGS)
    model = _model(kwargs)
    perturb_posterior(model)
    batch = _loss_batch()
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
        block_width=len(TINY_KEEP_INDEX) if gated else 109,
        package_owned=_RESOLVED_GAP_KEYS,
    )
