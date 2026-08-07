r"""The loss-spike breaker under this model's sign-indefinite, 480-sample-summed loss.

``main_loss`` here is a learned-variance Gaussian NLL summed over the forecast block, so it
goes negative harder and earlier than the sibling's per-feature loss ever did. The breaker's
relative test is $\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$, which silently assumes a
loss bounded below by zero: once the EMA is negative the test degenerates to "skip every
positive batch" -- the failure that has already cost this repository a run. The shipped block
therefore disables the relative test with a floor far above any reachable loss and carries the
finite-blow-up detection in ``additive_margin``, which compares against the *raw* EMA and keeps
working at a negative baseline.

Every test below drives the breaker with the block ``configs/default.yaml`` actually ships
(warm-up shortened to zero so the gate is active), so a config edit that regressed the
behaviour fails here rather than on the production box.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


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
# The negative-loss regime the model actually trains in
# --------------------------------------------------------------------------------------
def test_a_sustained_negative_loss_never_spikes(task):
    """A breaker that skipped here would zero-gradient the entire run. This is the failure the
    huge ``ema_floor`` exists to prevent, at the scale this objective actually reaches."""
    module = task()
    config = _shipped_breaker()

    skips = [
        _skipped(_feed(module, value, config)[0])
        for value in (-800.0, -1200.0, -5000.0, -300.0, -8000.0, -100.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


def test_a_sign_crossing_batch_is_not_a_spike(task):
    """With the EMA negative, an ordinary batch landing just above zero must train. At a zero
    ``ema_floor`` the relative test would discard it; the shipped floor plus the additive
    margin leave it alone."""
    module = task()
    config = _shipped_breaker()
    for _ in range(5):
        _feed(module, -500.0, config)
    assert module._spike_ema_loss < 0.0

    metrics, _ = _feed(module, 300.0, config)  # an ordinary fluctuation, not a blow-up

    assert not _skipped(metrics)


def test_the_relative_test_is_genuinely_off(task):
    """Under the huge floor, even a value far above ``multiplier * EMA`` passes when it stays
    inside the additive margin -- so the margin, not the ratio, is the active finite test."""
    module = task()
    config = _shipped_breaker(additive_margin=0.0)  # isolate the relative test
    for _ in range(5):
        _feed(module, 200.0, config)

    assert not _skipped(_feed(module, 5000.0, config)[0])


def test_the_shipped_margin_catches_a_finite_blowup(task):
    """The event the sibling's baseline actually had -- a finite jump with no NaN anywhere --
    re-enacted at this objective's scale. The non-finite guard has nothing to catch; the
    additive test is the one that must fire, against the raw (negative) EMA."""
    module = task()
    config = _shipped_breaker()
    for _ in range(5):
        _feed(module, -500.0, config)
    ema_before = module._spike_ema_loss
    margin = float(config["additive_margin"])

    metrics, returned = _feed(module, ema_before + 2.0 * margin, config)

    assert _skipped(metrics)
    assert torch.isfinite(returned), "a skipped step must still return a finite loss"
    assert module._spike_ema_loss == ema_before, "a skipped spike must not drag the EMA up"
    assert float(metrics["main_loss"]) < 0.0, "the logged main_loss was replaced by the EMA"


def test_a_non_finite_loss_still_skips(task):
    """The guard that survives every threshold setting, because it consults none of them."""
    module = task()
    config = _shipped_breaker()
    for value in (-800.0, -1200.0, -900.0):
        _feed(module, value, config)

    for bad in (float("nan"), float("inf")):
        metrics, returned = _feed(module, bad, config)
        assert _skipped(metrics)
        assert torch.isfinite(returned)


def test_max_consecutive_skips_releases_the_breaker_rather_than_deadlocking(task):
    """After the cap, the next finite batch is force-accepted and the EMA hard re-seeded --
    the escape from the frozen-EMA deadlock that once cost a v3 run ~160 epochs."""
    module = task()
    config = _shipped_breaker()
    cap = int(config["max_consecutive_skips"])
    for _ in range(5):
        _feed(module, -500.0, config)  # settle a healthy negative EMA
    spike = module._spike_ema_loss + 10.0 * float(config["additive_margin"])

    skips = [_skipped(_feed(module, spike, config)[0]) for _ in range(cap + 2)]

    assert module._spike_forced_accepts_total >= 1, "the escape hatch never fired"
    assert not all(skips), "every batch skipped; the breaker deadlocked"
    # The hard re-seed: after the forced accept the EMA sits at the new level, so the same
    # value is no longer a spike.
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
    """The skip path is a zero-gradient step, not an absent one: the forward already armed
    DDP's reducer, which expects one gradient hook per parameter. The breaker returns
    ``torch.where`` over the REAL loss -- backward still traverses the full graph, so every
    hook fires -- and ``on_after_backward`` zeroes the NaN that a poisoned graph pushes
    through the zero incoming gradient."""
    module = task()

    # A non-finite loss whose autograd graph spans every trainable parameter, as the real
    # loss does; a leaf NaN would prove nothing about the hooks.
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
