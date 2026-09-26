r"""The loss-spike breaker at this encoder, on a sign-indefinite loss over the decoded anchors.

``main_loss`` here is a learned-variance Gaussian NLL summed over the $H \cdot C_{\mathrm{keep}}$
block and averaged over the anchors the tiling decoded. The breaker constants were MEASURED once, at
the $2940$-coefficient block of the $H = 30$, legacy-operator geometry; every later value, including
the 2026-09-23 final revision's, is that measurement scaled by the block ratio. The revision moved
this cell's block and not the conv-LSTM cell's, so since then neither ``additive_margin`` nor
``gradient_clip_val`` equals that cell's -- the switches (``ema_floor``, ``multiplier``, the escape
hatch) still do.

Every test drives the breaker with the block **this** ``configs/default.yaml`` ships, at magnitudes
this objective actually reaches, so a config edit that regressed the behaviour fails here rather
than on the production box. The breaker's own mechanics -- the non-finite guard, the forced accept
after the skip cap, the zero-gradient skip step -- are the shared task base's and are tested with it;
what is tested here is that this cell's constants sit where they have to: the floor above anything
the objective can reach, the margin between the worst measured excursion and the reachable range,
and the clip between the measured q99 and maximum.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: Surviving target channels at the shipped warm-up budget on the integer-operator shards. Written
#: out because this file reasons about the block's arithmetic bound rather than building a model.
KEPT_TARGET_CHANNELS = 76

#: The block the instrumented run measured the breaker and clip statistics at: $H = 30$ over the
#: legacy operator's $98$ survivors. Every shipped value since is scaled from it by the block ratio.
_MEASURED_BLOCK = 30 * 98

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
#: This is the CONV-LSTM cell's $5090.2$ rather than this cell's own $3927.5$, and it is left
#: UNSCALED although it was measured at the larger $2940$-coefficient block: the larger of the two
#: measurements at the larger block is the conservative bar for a margin that was itself scaled
#: rather than re-measured.
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
# Where the shipped constants sit
# --------------------------------------------------------------------------------------
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
    something to catch. Both numbers are the instrumented run's, recorded in the config, SCALED by
    the block ratio to the shipped block -- which is how the shipped value was derived, and which
    the headline run's own ``train/grad_norm`` column must replace."""
    config = load_config(str(_CONFIG))
    clip = float(config["advanced_config"]["trainer"]["gradient_clip_val"])
    ratio = config["model_config"]["VAE_model"]["horizon"] * KEPT_TARGET_CHANNELS / _MEASURED_BLOCK

    assert 13059.7 * ratio < clip < 14380.7 * ratio


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


def test_the_shipped_margin_catches_a_finite_blowup_and_leaves_an_ordinary_move_alone(task):
    """A finite jump with no NaN anywhere, re-enacted at this objective's scale. The non-finite
    guard has nothing to catch; the additive test is the one that must fire, against the raw
    (negative) EMA. And the other side of the same threshold, the one a too-tight margin breaks:
    half a margin above the EMA must train."""
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

    metrics, _ = _feed(module, module._spike_ema_loss + 0.5 * margin, config)
    assert not _skipped(metrics)


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
