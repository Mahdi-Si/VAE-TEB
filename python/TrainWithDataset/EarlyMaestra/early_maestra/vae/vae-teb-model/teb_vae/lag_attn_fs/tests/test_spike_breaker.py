r"""The loss-spike breaker under this model's sign-indefinite, 2340-coefficient-summed loss.

``main_loss`` here is a learned-variance Gaussian NLL summed over the largest forecast block in the
family -- $H \cdot C_{\mathrm{keep}} = 2340$ coefficients against the raw models' $480$ samples --
so it goes negative harder and earlier than any of them. The breaker's relative test is
$\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$, which silently assumes a loss bounded below by
zero: once the EMA is negative it degenerates to "skip every positive batch", the failure that has
already cost this repository a run. The shipped block therefore disables the relative test with a
floor far above any reachable loss and carries the finite-blow-up detection in ``additive_margin``,
which compares against the *raw* EMA and keeps working at a negative baseline.

Ported rather than inherited, and the port earns its place: every threshold in that block is stated
in nats of the summed block, so none of the sibling's values transfer and none of its tests are
evidence about this configuration. Each test below drives the breaker with the block **this**
``configs/default.yaml`` ships (warm-up shortened to zero so the gate is active), at magnitudes this
objective actually reaches, so a config edit that regressed the behaviour fails here rather than on
the production box.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_SIBLING_CONFIG = (
    Path(__file__).resolve().parents[3] / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"
)

#: A loss magnitude this objective genuinely reaches, used to settle a healthy negative EMA. Chosen
#: at the scale of the summed block rather than at the sibling's: a test that settled its EMA at
#: $-500$ would be testing the sibling's regime under this config's thresholds.
_HEALTHY_LOSS = -4000.0


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
# The thresholds this target domain had to move
# --------------------------------------------------------------------------------------
def test_the_margin_is_larger_than_the_sibling_s_and_the_floor_is_not():
    """One of the two moved and the other did not, and the reason is the whole design of the block.
    ``additive_margin`` is stated in nats of the summed block, so it scales with the block; the
    ``ema_floor`` is not a scale at all -- it is a switch that turns the relative test off by
    sitting above any reachable loss, and "above any reachable loss" is satisfied by the same
    number in both models."""
    mine = load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]
    theirs = load_config(str(_SIBLING_CONFIG))["advanced_config"]["spike_breaker"]

    assert mine["additive_margin"] > theirs["additive_margin"]
    assert mine["ema_floor"] == theirs["ema_floor"] >= 1.0e9
    assert mine["multiplier"] == theirs["multiplier"]
    assert mine["max_consecutive_skips"] == theirs["max_consecutive_skips"]


def test_the_floor_still_exceeds_any_loss_this_objective_can_reach():
    r"""Confirmed rather than assumed, because the block that has to stay under it grew by
    $4.9\times$. The reconstruction is summed over $2340$ coefficients and the per-coefficient
    Gaussian NLL is bounded below by $\tfrac{1}{2}(\log 2\pi + \ell_{\min}) \approx -1.6$ at the
    shipped ``logvar_clamp`` floor of $-5$, so the two reconstruction terms cannot fall below about
    $2 \times 2340 \times 1.6 \approx 7.5 \times 10^{3}$ in magnitude -- five orders of magnitude
    inside the floor. The KL and the prior anchor are both nonnegative and only add.
    """
    config = load_config(str(_CONFIG))
    vae = config["model_config"]["VAE_model"]
    block = vae["horizon"] * 78  # C_keep at the shipped reach budget
    logvar_floor = float(vae["logvar_clamp"][0])

    most_negative = 2.0 * block * 0.5 * (math.log(2.0 * math.pi) + logvar_floor)

    assert abs(most_negative) < 1.0e5
    assert config["advanced_config"]["spike_breaker"]["ema_floor"] > 100.0 * abs(most_negative)


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
        for value in (-4000.0, -6000.0, -25000.0, -1500.0, -40000.0, -500.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


def test_a_sign_crossing_batch_is_not_a_spike(task):
    """With the EMA negative, a batch landing above zero but inside the margin must train. At a
    zero ``ema_floor`` the relative test would discard it; the shipped floor plus the additive
    margin leave it alone.

    Both magnitudes are expressed in margins rather than in nats, so this keeps testing the
    sign-crossing property rather than a particular pair of numbers when the margin is re-derived.
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
        _feed(module, 1000.0, config)

    assert not _skipped(_feed(module, 25000.0, config)[0])


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


def test_the_margin_leaves_an_ordinary_epoch_to_epoch_move_alone(task):
    """The other side of the same threshold, and the one a too-tight margin breaks. A margin below
    the objective's own healthy fluctuation skips ordinary batches, which reads in the log exactly
    like a model that keeps blowing up."""
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
    for value in (-4000.0, -6000.0, -4500.0):
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

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert config["comparison_metric"] in metrics


def test_a_skipped_step_still_touches_every_parameter(task):
    """The skip path is a zero-gradient step, not an absent one: the forward already armed DDP's
    reducer, which expects one gradient hook per parameter. The breaker returns ``torch.where`` over
    the REAL loss -- backward still traverses the full graph, so every hook fires -- and
    ``on_after_backward`` zeroes the NaN that a poisoned graph pushes through the zero incoming
    gradient."""
    module = task()

    # A non-finite loss whose autograd graph spans every trainable parameter, as the real loss
    # does; a leaf NaN would prove nothing about the hooks.
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
