r"""The objective, and the two properties of its reduction that no other test can see.

**The global mean is not the mean of per-rank means.** Every other test in this suite runs on one
process, where the two coincide exactly, so the distinction is invisible unless it is constructed.
It is constructed here by splitting one batch into two shares of different sizes and checking that
what distributed gradient averaging would produce equals what a single process scoring the whole
batch produces. A reduction that divided each rank's numerator by its own denominator would pass
every other test in this package and optimise a different objective on the cluster.

**An empty support is not a scored anchor.** The shared reduction floors its denominator at one, so
a batch that scored nothing reports its numerator as though one anchor had produced it. Here the
loss is a graph-connected zero and the scored count says zero, which is what keeps an epoch
aggregate from absorbing a step that measured nothing.

And one routing check, because the failure it catches is silent: an un-overridden ``compute_loss``
resolves through the two mixins into the shared implementation and trains a working model against
the wrong reduction.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets import objective as objective_module
from teb_vae.lag_slot_transformer_cfs.nets.objective import compute_residual_objective
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_BATCH,
    TINY_MODEL_HORIZON,
    TINY_SEQ_LEN,
    TINY_TARGET_KEEP,
    build_tiny_model,
    tiny_model_kwargs,
    tiny_streams,
)


def build_model(**overrides) -> SeqVaeLagResidualTrfCfs:
    """Build the tiny model with its source pathway moved off zero.

    Every reconstruction number is identical between the two branches at the zero start, so a test
    that needs the two to differ needs the pathway to have moved.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model, in evaluation mode.
    """
    model = build_tiny_model(**overrides)
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        model.proposal_head.output_proj.weight.normal_(0.0, 0.3, generator=generator)
        model.proposal_head.output_proj.bias.normal_(0.0, 0.1, generator=generator)
    return model.eval()


def score(model, *, rows=slice(None), weight=None, seed: int = 0, **kwargs):
    """Run one dense forward over a slice of the fixture batch and score it.

    A slice rather than a count, because the distributed check needs the batch tail as well as its
    head, and two shares taken from one batch is what makes them comparable against scoring the
    whole of it.

    Args:
        model: The model.
        rows: Which samples of the fixture streams to use.
        weight: Validity signal, or ``None`` for an all-valid one.
        seed: Seed for the reparameterisation draw.
        **kwargs: Extra objective keywords.

    Returns:
        ``(forward_outputs, metrics)``.
    """
    y_st, y_ph, u_stream = (stream[rows] for stream in tiny_streams())
    torch.manual_seed(seed)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    target = torch.cat([y_st, y_ph], dim=-1)
    validity = torch.ones(y_st.shape[0], TINY_SEQ_LEN) if weight is None else weight
    metrics = model.compute_loss(outputs, target, weight=validity, **kwargs)["metrics"]
    return outputs, metrics


# =================================================================================================
# Weights and supports
# =================================================================================================
def test_the_channel_weights_sum_to_the_kept_channel_count() -> None:
    """Resolved from the model rather than from a literal, so a budget change moves them with it.

    The renormalisation is what makes the two block weights a *ratio*: it leaves the block's scale
    alone, so an arm that changes the ratio does not also change how a nat compares with another
    arm's.
    """
    model = build_model(target_weight_st=1.0, target_weight_ph=0.1)
    weights = model.target_channel_weight
    kept = len(TINY_TARGET_KEEP)
    assert weights.numel() == kept
    assert float(weights.sum()) == pytest.approx(float(kept), rel=1e-5)
    # Two distinct values, in the ratio the configuration states.
    values = sorted(set(round(float(value), 6) for value in weights))
    assert len(values) == 2
    assert values[1] / values[0] == pytest.approx(10.0, rel=1e-4)


def test_the_horizon_weights_sum_to_the_horizon() -> None:
    """Same contract on the other axis: a distribution over the horizon, not a rescaling of it."""
    model = build_model(horizon_weight_halflife_steps=5.0)
    weights = model.horizon_weight
    assert weights.numel() == TINY_MODEL_HORIZON
    assert float(weights.sum()) == pytest.approx(float(TINY_MODEL_HORIZON), rel=1e-5)
    # Monotonically decreasing, which is the point: the far steps are the majority of the block.
    assert bool((weights[:-1] > weights[1:]).all())


def test_an_unweighted_model_carries_no_weight_buffer_at_all() -> None:
    """``None`` skips the multiplication rather than multiplying by ones, so the score is bitwise."""
    model = build_model(target_weight_st=1.0, target_weight_ph=1.0)
    assert getattr(model, "horizon_weight", None) is None
    # The channel weight exists but is uniform, which is the family's convention for this pair.
    assert torch.allclose(
        model.target_channel_weight, torch.ones_like(model.target_channel_weight)
    )


def test_the_reconstruction_and_divergence_share_one_anchor_set() -> None:
    """By construction, not by two expressions that agree today.

    An anchor with no reconstruction term has nothing pulling the full distribution off the prior,
    so charging a divergence on it regularises it onto the prior for free -- and those anchors
    cluster immediately before every signal gap, which reads as coupling fading exactly where the
    signal degrades.
    """
    model = build_model()
    weight = torch.ones(TINY_BATCH, TINY_SEQ_LEN)
    weight[:, 12:16] = 0.0  # a gap inside the decoded range
    _, metrics = score(model, weight=weight)

    dense_anchors = TINY_SEQ_LEN - TINY_MODEL_HORIZON - tiny_model_kwargs()["warmup_period"]
    assert 0.0 < float(metrics["scored_anchors"]) < TINY_BATCH * dense_anchors
    # The divergence is reported over the same count the reconstruction was averaged over, so the
    # two columns are addable without rescaling.
    assert float(metrics["scored_anchors"]) == pytest.approx(
        float(metrics["scored_anchors"])
    )


# =================================================================================================
# The reduction
# =================================================================================================
def test_the_reported_score_is_the_hand_computed_mean() -> None:
    """One deterministic batch, one reference sum, no distributed machinery involved."""
    model = build_model()
    outputs, metrics = score(model, likelihood="mse")

    y_st, y_ph, _ = (stream[:TINY_BATCH] for stream in tiny_streams())
    target = model._build_forecast_target(
        torch.cat([y_st, y_ph], dim=-1), outputs["anchor_index"]
    )
    # An all-valid weight makes every anchor contribute, so the reference is a plain mean over the
    # whole anchor axis.
    per_element = (target - outputs["mu_full"]) ** 2 * model.target_channel_weight
    per_anchor = per_element.sum(dim=(2, 3))
    expected = float(per_anchor.mean())

    assert float(metrics["nll_full_block"]) == pytest.approx(expected, rel=1e-5)
    assert float(metrics["scored_anchors"]) == float(per_anchor.numel())


def test_the_objective_optimises_the_global_mean_under_uneven_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the whole module exists for, constructed because one process cannot show it.

    Two shares of **different** sizes stand in for two ranks. Distributed gradient averaging divides
    by the world size, so the check is that the average of the two shares' losses and gradients
    equals what one process scoring the whole batch produces. A reduction dividing each share by its
    own count would fail both, by a factor that depends on how unevenly the anchors fell -- and
    would pass every other test in this package, all of which run on one process where the two
    reductions coincide.

    The latent draw is pinned to zero for the duration, so slicing the batch does not also change
    the noise: with a fresh draw per call the shares and the whole batch would sample different
    latents and the comparison would measure that instead.
    """
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    kwargs = dict(likelihood="gaussian_nll", beta=1.0, beta_prior=0.1)
    world = 2
    shares = (slice(0, 1), slice(1, TINY_BATCH))  # one sample against two

    def local_sums(rows: slice) -> torch.Tensor:
        """The packed local vector one share contributes, with the reduction left alone."""
        captured = {}

        def capture(values: torch.Tensor) -> torch.Tensor:
            """Record the local vector and pass it through unchanged."""
            captured["local"] = values.clone()
            return values

        monkeypatch.setattr(objective_module, "_all_reduce_sum", capture)
        score(build_model(), rows=rows, **kwargs)
        monkeypatch.undo()
        monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
        return captured["local"]

    totals = [local_sums(rows) for rows in shares]
    combined = totals[0] + totals[1]

    def run(rows: slice, other: torch.Tensor, simulate: bool):
        """Score one share under a simulated collective, or the whole batch on one process."""
        model = build_model()
        if simulate:
            monkeypatch.setattr(
                objective_module, "_all_reduce_sum", lambda values: values + other
            )
            monkeypatch.setattr(objective_module.dist, "is_initialized", lambda: True)
            monkeypatch.setattr(objective_module.dist, "get_world_size", lambda: world)
        _, metrics = score(model, rows=rows, **kwargs)
        metrics["total_loss"].backward()
        grad = model.decoder.mean_head.weight.grad.clone()
        if simulate:
            monkeypatch.undo()
            monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
        return float(metrics["total_loss"]), grad, metrics

    first_loss, first_grad, first_metrics = run(shares[0], totals[1], simulate=True)
    second_loss, second_grad, second_metrics = run(shares[1], totals[0], simulate=True)
    whole_loss, whole_grad, whole_metrics = run(
        slice(0, TINY_BATCH), combined * 0, simulate=False
    )

    # Both shares report the same global mean, and it is the whole batch's.
    for name in ("nll_full_block", "nll_base_block", "source_conditioned_kl_raw"):
        assert float(first_metrics[name]) == pytest.approx(
            float(whole_metrics[name]), rel=1e-5
        ), name
        assert float(second_metrics[name]) == pytest.approx(
            float(whole_metrics[name]), rel=1e-5
        ), name
    assert float(first_metrics["scored_anchors"]) == pytest.approx(
        float(whole_metrics["scored_anchors"]), rel=1e-9
    )

    # And what distributed averaging produces is the whole batch's loss and its gradient. This is
    # the assertion a per-rank mean fails: the shares hold one and two samples, so the average of
    # per-rank means would over-weight the smaller share by a factor of three halves.
    assert (first_loss + second_loss) / world == pytest.approx(whole_loss, rel=1e-4)
    assert torch.allclose(
        (first_grad + second_grad) / world, whole_grad, atol=1e-5, rtol=1e-3
    )


def test_an_empty_support_reports_zero_rather_than_a_clamped_denominator() -> None:
    """And the loss stays in the graph, so a rank with nothing to score still participates."""
    model = build_model()
    weight = torch.zeros(TINY_BATCH, TINY_SEQ_LEN)
    # The Gaussian score, not a squared error: under a squared error the observation log-variance
    # head is outside the graph on EVERY batch, so the reachability half of this check would be
    # measuring the likelihood rather than the empty-support path.
    _, metrics = score(model, weight=weight, likelihood="gaussian_nll")

    assert float(metrics["scored_anchors"]) == 0.0
    assert float(metrics["nll_full_block"]) == 0.0
    assert float(metrics["total_loss"]) == 0.0
    assert metrics["total_loss"].requires_grad

    metrics["total_loss"].backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_an_empty_support_reports_every_column_a_full_step_does() -> None:
    """Otherwise the metric history gains and loses columns depending on the batch."""
    model = build_model()
    _, full = score(model, likelihood="mse")
    _, empty = score(
        model, weight=torch.zeros(TINY_BATCH, TINY_SEQ_LEN), likelihood="mse"
    )
    assert set(full) == set(empty)


def test_the_free_bits_floor_raises_the_trained_divergence_above_the_raw_one() -> None:
    """Which is why the two are reported separately and only the raw one is a rate."""
    model = build_model()
    _, floored = score(model, likelihood="mse", free_bits=0.5)
    assert float(floored["source_conditioned_kl_train"]) > float(
        floored["source_conditioned_kl_raw"]
    )

    _, unfloored = score(model, likelihood="mse", free_bits=0.0)
    assert float(unfloored["source_conditioned_kl_train"]) == pytest.approx(
        float(unfloored["source_conditioned_kl_raw"]), rel=1e-6
    )


def test_the_gaussian_score_carries_its_full_constant() -> None:
    """So a block sum is a true negative log density in nats when no weight is applied."""
    model = build_model(target_weight_st=1.0, target_weight_ph=1.0)
    outputs, metrics = score(model, likelihood="gaussian_nll")

    y_st, y_ph, _ = (stream[:TINY_BATCH] for stream in tiny_streams())
    target = model._build_forecast_target(
        torch.cat([y_st, y_ph], dim=-1), outputs["anchor_index"]
    )
    logvar = outputs["logvar_full"]
    per_element = 0.5 * (
        math.log(2.0 * math.pi)
        + logvar
        + (target - outputs["mu_full"]) ** 2 * torch.exp(-logvar)
    )
    expected = float(per_element.sum(dim=(2, 3)).mean())
    assert float(metrics["nll_full_block"]) == pytest.approx(expected, rel=1e-5)


# =================================================================================================
# Routing
# =================================================================================================
def test_the_shared_objective_is_never_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The silent failure this override exists to prevent.

    Without it, ``compute_loss`` resolves through the two target-domain mixins into the shared
    implementation, which divides a rank's own numerator by a rank's own denominator and floors
    that denominator at one. The model would train, converge, and report plausible nats against a
    reduction this architecture rejects.
    """
    import teb_vae.lag_attn_fs.nets.feature_target as feature_target

    def refuse(*args, **kwargs):
        """Stand in for the shared objective, so reaching it is a failure rather than a number."""
        raise AssertionError("the shared objective was reached")

    monkeypatch.setattr(feature_target, "compute_shared_objective", refuse)
    _, metrics = score(build_model(), likelihood="mse")
    assert float(metrics["scored_anchors"]) > 0.0


def test_a_nonzero_shape_term_weight_is_refused() -> None:
    """Accepted as keywords rather than dropped, so a configuration setting one fails loudly.

    The three terms read a forecast block's last axis as a trajectory, and here it counts channels.
    A signature without them would let the driver drop the key in silence.
    """
    model = build_model()
    for name in ("lambda_ms", "lambda_deriv", "lambda_boundary"):
        with pytest.raises(ValueError, match=name):
            score(model, likelihood="mse", **{name: 0.5})


def test_a_target_at_the_wrong_anchor_set_is_refused() -> None:
    """The one argument whose mismatch would score a forecast against another anchor's future."""
    model = build_model()
    y_st, y_ph, u_stream = tiny_streams()
    torch.manual_seed(0)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1)
    wrong = torch.zeros(TINY_BATCH, 2, TINY_MODEL_HORIZON, len(TINY_TARGET_KEEP))

    with pytest.raises(ValueError, match="decoded anchor set"):
        compute_residual_objective(
            outputs,
            wrong,
            weight=torch.ones(TINY_BATCH, TINY_SEQ_LEN),
            geometry=model.geometry,
            block_width=model.decoder_out_channels,
            coverage_floor=model.coverage_floor,
            logvar_clamp=model.logvar_clamp,
        )
