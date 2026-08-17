r"""The loss-spike breaker at this encoder, on a sign-indefinite loss over roughly ten anchors.

``main_loss`` here is a learned-variance Gaussian NLL summed over $H \cdot C_{\mathrm{keep}} = 2940$
coefficients and averaged over the anchors the tiling decoded -- about $4.57$ at the shipped stride.
Both numbers are the conv-LSTM causal cell's, because the **encoder edge changes neither**, which is
what makes the two breaker constants below inherited-and-re-measured rather than re-derived.

The one constant the edge does move is ``gradient_clip_val``, which is a gradient statistic rather
than a loss scale, and it is checked here beside the two that did not move so the asymmetry is a
passing test rather than a claim in a comment.

Every test drives the breaker with the block **this** ``configs/default.yaml`` ships, at magnitudes
this objective actually reaches -- the instrumented run the config's comments record saw
``main_loss`` inside $[-3660, +4235]$ within 600 steps -- so a config edit that regressed the
behaviour fails here rather than on the production box.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENCODER_SIBLING = _REPO_ROOT / "teb_vae" / "lag_attn_cfs" / "configs" / "default.yaml"
_TARGET_SIBLING = (
    _REPO_ROOT / "teb_vae" / "lag_attn_transformer_fs" / "configs" / "default.yaml"
)

#: Surviving target channels at the shipped warm-up budget. Written out because this file reasons
#: about the block's arithmetic bound rather than building a model.
KEPT_TARGET_CHANNELS = 98

#: A loss magnitude this objective genuinely reaches, used to settle a healthy negative EMA. Chosen
#: from the instrumented run rather than scaled: that run's post-ramp half sat well below zero with
#: a minimum of $-7464$, so a few hundred negative is comfortably inside the regime the breaker has
#: to be quiet in.
_HEALTHY_LOSS = -500.0

#: The worst excursion above the EMA the instrumented run measured, in the noisiest regime the
#: committed fixture can produce (batch 1, four distinct batches), after the breaker's own priming
#: window. The margin has to clear it or ordinary batches are skipped -- which reads in the log
#: exactly like a model that keeps blowing up.
#:
#: This is the CONV-LSTM cell's $5090.2$ rather than this cell's own $3927.5$, deliberately: the two
#: cells share one margin because the encoder edge changes neither the block nor the anchor count,
#: so the bar that has to be cleared is the larger of the two measurements.
_WORST_MEASURED_EXCURSION = 5090.2


def _shipped_breaker(**overrides) -> dict:
    """The spike-breaker block the shipped config carries, with test-friendly overrides."""
    config = dict(load_config(str(_CONFIG))["advanced_config"]["spike_breaker"])
    config["warmup_batches"] = 0  # the priming window is not what is under test
    config.update(overrides)
    return config


def _feed(module, value, config, main=None):
    """Run one breaker decision on a scalar loss.

    Calls ``_apply_spike_breaker`` directly rather than going through a step: the breaker's own
    behaviour is what is under test, and a real step would need a Trainer to log through.

    Args:
        module: The task.
        value: The returned loss.
        config: The breaker block.
        main: ``metrics['main_loss']``; defaults to ``value``.

    Returns:
        ``(metrics, returned_loss)``.
    """
    main_value = value if main is None else main
    metrics = {
        "total_loss": torch.tensor(float(value)),
        "main_loss": torch.tensor(float(main_value)),
    }
    loss = torch.tensor(float(value), requires_grad=True)
    returned = module._apply_spike_breaker(loss, metrics, config)
    return metrics, returned


def _skipped(metrics) -> bool:
    return bool(metrics["spike_skipped"].item())


# --------------------------------------------------------------------------------------
# Which constants the encoder edge moves, and which it does not
# --------------------------------------------------------------------------------------
def test_the_loss_scale_constants_are_the_encoder_siblings_and_the_clip_is_not():
    """The asymmetry the re-derivation found, as a test. ``additive_margin`` and ``ema_floor`` are
    stated in nats of the summed block; the encoder edge changes neither the block ($2940$) nor the
    anchor count ($\\approx 4.57$), so both must equal the conv-LSTM causal cell's exactly.
    ``gradient_clip_val`` is a gradient-norm statistic, and the encoder is precisely what moves
    it."""
    mine = load_config(str(_CONFIG))["advanced_config"]
    theirs = load_config(str(_ENCODER_SIBLING))["advanced_config"]

    assert mine["spike_breaker"]["additive_margin"] == theirs["spike_breaker"]["additive_margin"]
    assert mine["spike_breaker"]["ema_floor"] == theirs["spike_breaker"]["ema_floor"] >= 1.0e9
    assert mine["spike_breaker"]["multiplier"] == theirs["spike_breaker"]["multiplier"]
    assert (
        mine["spike_breaker"]["max_consecutive_skips"]
        == theirs["spike_breaker"]["max_consecutive_skips"]
    )
    assert mine["trainer"]["gradient_clip_val"] != theirs["trainer"]["gradient_clip_val"]


def test_the_margin_is_larger_than_the_two_sided_conv_transformer_cells():
    """The other edge: the block *does* move across the transform, from $2340$ coefficients at
    $\\approx 240$ anchors to $2940$ at $\\approx 4.57$, so the margin that cell ships is out of
    range here.

    The direction reversed when both causal cells went to the two-minute horizon. At $H = 15$ this
    block was the *smaller* of the two ($1470$ against $2340$) and the margin shrank below the
    two-sided cell's; at $H = 30$ it is the larger, and the margin grows past it. The sign of this
    comparison is a readout of which side has the bigger block, which is why it is asserted rather
    than assumed.
    """
    mine = load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]
    theirs = load_config(str(_TARGET_SIBLING))["advanced_config"]["spike_breaker"]

    assert mine["additive_margin"] > theirs["additive_margin"]


def test_the_floor_still_exceeds_any_loss_this_objective_can_reach():
    r"""Confirmed rather than assumed, at the block that has to stay under it. The reconstruction is
    summed over $2940$ coefficients and the per-coefficient Gaussian NLL is bounded below by
    $\tfrac{1}{2}(\log 2\pi + \ell_{\min}) \approx -1.58$ at the shipped ``logvar_clamp`` floor of
    $-5$, so the two reconstruction terms cannot fall below about
    $2 \times 2940 \times 1.58 \approx 9.3 \times 10^{3}$ in magnitude. The KL and the prior anchor
    are both nonnegative and only add.
    """
    config = load_config(str(_CONFIG))
    vae = config["model_config"]["VAE_model"]
    block = vae["horizon"] * KEPT_TARGET_CHANNELS
    logvar_floor = float(vae["logvar_clamp"][0])

    most_negative = 2.0 * block * 0.5 * (math.log(2.0 * math.pi) + logvar_floor)

    assert abs(most_negative) < 1.0e5
    assert config["advanced_config"]["spike_breaker"]["ema_floor"] > 100.0 * abs(most_negative)


def test_the_margin_stays_inside_the_range_the_objective_can_reach():
    r"""A margin above the whole reachable magnitude makes the additive test **decoration**: no
    finite value could ever exceed $\mathrm{EMA} + \mathrm{margin}$, and the breaker would
    degenerate to its non-finite guard alone with nothing in the log saying so."""
    config = load_config(str(_CONFIG))
    vae = config["model_config"]["VAE_model"]
    reachable = abs(
        2.0
        * vae["horizon"]
        * KEPT_TARGET_CHANNELS
        * 0.5
        * (math.log(2.0 * math.pi) + float(vae["logvar_clamp"][0]))
    )

    assert float(config["advanced_config"]["spike_breaker"]["additive_margin"]) < reachable


def test_the_margin_clears_the_worst_excursion_the_instrumented_run_measured():
    """The lower of the two bounds the margin sits between; ``_the_margin_stays_inside_the_range``
    is the upper one, and at this horizon they are close enough that the pair genuinely fixes the
    value rather than leaving a wide band."""
    margin = float(
        load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]["additive_margin"]
    )

    assert margin > _WORST_MEASURED_EXCURSION


def test_the_clip_sits_above_the_measured_q99_and_below_the_measured_maximum():
    """The family's rule, and the property that makes the value a blow-up guard rather than a
    rescaler: above q99 so a healthy step is untouched, below the observed maximum so the guard has
    something to catch. Both numbers are the instrumented run's, recorded in the config."""
    clip = float(load_config(str(_CONFIG))["advanced_config"]["trainer"]["gradient_clip_val"])

    assert 13059.7 < clip < 14380.7


# --------------------------------------------------------------------------------------
# The negative-loss regime the model actually trains in
# --------------------------------------------------------------------------------------
def test_a_sustained_negative_loss_never_spikes(task):
    """A breaker that skipped here would zero-gradient the entire run. This is the failure the huge
    ``ema_floor`` exists to prevent, at the scale this objective actually reaches."""
    module = task()
    config = _shipped_breaker()

    skips = [
        _skipped(_feed(module, value, config)[0])
        for value in (-500.0, -3660.0, -1200.0, -300.0, -2000.0, -100.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


def test_a_sign_crossing_batch_is_not_a_spike(task):
    """With the EMA negative, a batch landing above zero but inside the margin must train. At a zero
    ``ema_floor`` the relative test would discard it; the shipped floor plus the additive margin
    leave it alone.

    Both magnitudes are expressed in margins rather than in nats, so this keeps testing the
    sign-crossing property rather than a particular pair of numbers.
    """
    module = task()
    config = _shipped_breaker()
    margin = float(config["additive_margin"])
    for _ in range(5):
        _feed(module, -0.5 * margin, config)
    assert module._spike_ema_loss < 0.0

    crossing = 0.4 * margin  # 0.9 margins above the EMA: over zero, inside the threshold
    metrics, _ = _feed(module, crossing, config)

    assert crossing > 0.0 > module._spike_ema_loss, "the batch did not cross zero"
    assert not _skipped(metrics)


def test_the_relative_test_is_genuinely_off(task):
    """Under the huge floor, even a value far above ``multiplier * EMA`` passes when it stays inside
    the additive margin -- so the margin, not the ratio, is the active finite test."""
    module = task()
    config = _shipped_breaker(additive_margin=0.0)  # isolate the relative test
    for _ in range(5):
        _feed(module, 500.0, config)

    assert not _skipped(_feed(module, 20000.0, config)[0])


def test_the_shipped_margin_catches_a_finite_blowup(task):
    """A finite jump with no NaN anywhere, re-enacted at this objective's scale. The non-finite
    guard has nothing to catch; the additive test is the one that must fire, against the raw
    (negative) EMA."""
    module = task()
    config = _shipped_breaker()
    for _ in range(5):
        _feed(module, _HEALTHY_LOSS, config)
    ema_before = module._spike_ema_loss
    margin = float(config["additive_margin"])

    metrics, returned = _feed(module, ema_before + 2.0 * margin, config)

    assert _skipped(metrics)
    assert torch.isfinite(returned), "a skipped step must still return a finite loss"
    assert module._spike_ema_loss == ema_before, "a skipped spike must not drag the EMA up"
    assert float(metrics["main_loss"]) < 0.0, "the logged main_loss was replaced by the EMA"


def test_the_margin_leaves_an_ordinary_step_to_step_move_alone(task):
    """The other side of the same threshold, and the one a too-tight margin breaks."""
    module = task()
    config = _shipped_breaker()
    for _ in range(5):
        _feed(module, _HEALTHY_LOSS, config)
    margin = float(config["additive_margin"])

    metrics, _ = _feed(module, module._spike_ema_loss + 0.5 * margin, config)

    assert not _skipped(metrics)


def test_a_non_finite_loss_still_skips(task):
    """The guard that survives every threshold setting, because it consults none of them."""
    module = task()
    config = _shipped_breaker()
    for value in (-500.0, -690.0, -450.0):
        _feed(module, value, config)

    for bad in (float("nan"), float("inf")):
        metrics, returned = _feed(module, bad, config)
        assert _skipped(metrics)
        assert torch.isfinite(returned)


def test_max_consecutive_skips_releases_the_breaker_rather_than_deadlocking(task):
    """After the cap, the next finite batch is force-accepted and the EMA hard re-seeded -- the
    escape from the frozen-EMA deadlock that once cost a run ~160 epochs."""
    module = task()
    config = _shipped_breaker()
    cap = int(config["max_consecutive_skips"])
    for _ in range(5):
        _feed(module, _HEALTHY_LOSS, config)  # settle a healthy negative EMA
    spike = module._spike_ema_loss + 10.0 * float(config["additive_margin"])

    skips = [_skipped(_feed(module, spike, config)[0]) for _ in range(cap + 2)]

    assert module._spike_forced_accepts_total >= 1, "the escape hatch never fired"
    assert not all(skips), "every batch skipped; the breaker deadlocked"
    # The hard re-seed: after the forced accept the EMA sits at the new level, so the same value is
    # no longer a spike.
    assert not _skipped(_feed(module, spike, config)[0])


def test_the_configured_comparison_metric_is_one_the_task_emits(
    task, stub_batch, perturb_posterior
):
    """``comparison_metric`` falls back to the returned loss silently when the named metric is
    missing, so the config must name something the task genuinely emits."""
    config = _shipped_breaker()
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert config["comparison_metric"] in metrics


def test_a_skipped_step_still_touches_every_parameter(task):
    """The skip path is a zero-gradient step, not an absent one: the forward already armed DDP's
    reducer, which expects one gradient hook per parameter. The breaker returns ``torch.where`` over
    the REAL loss -- backward still traverses the full graph, so every hook fires -- and
    ``on_after_backward`` zeroes the NaN that a poisoned graph pushes through the zero incoming
    gradient."""
    module = task()

    real = torch.stack([p.sum() for p in module.parameters() if p.requires_grad]).sum()
    poisoned = real * float("nan")
    metrics = {"total_loss": poisoned.detach(), "main_loss": poisoned.detach()}
    returned = module._apply_spike_breaker(poisoned, metrics, _shipped_breaker())
    assert _skipped(metrics)

    module.zero_grad(set_to_none=True)
    returned.backward()
    module.on_after_backward()

    starved = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not starved, f"parameters left without a gradient hook on a skipped step: {starved}"
    assert math.isfinite(float(returned))
    poisoned_grads = [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    ]
    assert not poisoned_grads, f"non-zero gradients survived a skipped step: {poisoned_grads}"
