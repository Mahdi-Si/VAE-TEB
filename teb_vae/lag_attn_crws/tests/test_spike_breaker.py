r"""The loss-spike breaker driven with the block this cell's ``configs/default.yaml`` ships.

``main_loss`` here is a learned-variance Gaussian NLL summed over an $H \cdot R$ raw block and
averaged over the anchors the tiling decoded, so it is sign-indefinite. The breaker's relative test
is $\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$, which silently assumes a loss bounded below by
zero -- once the EMA is negative it degenerates to "skip every positive batch". The shipped block
therefore disables the relative test with a floor far above any reachable loss and carries the
finite-blow-up detection in ``additive_margin``, which compares against the *raw* EMA.

The breaker's mechanics (non-finite skips, the forced-accept escape, the zero-gradient skip path)
belong to ``train/pl_model_base.py`` and are pinned in ``lag_attn`` and ``lag_attn_rws``. What is
this cell's is the configuration, so what is checked here is that the shipped block behaves:

* the floor exceeds the most negative loss this objective can reach, computed from the shipped
  horizon, raw grid and log-variance clamp;
* a batch crossing zero from a negative EMA is not a spike;
* a finite blow-up above the margin is caught, without dragging the EMA up;
* the configured comparison metric is one the task emits.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

#: A loss magnitude this objective genuinely reaches, used to settle a healthy EMA: the instrumented
#: run's post-priming mean with the whole shard in one batch.
_HEALTHY_LOSS = 215.0


def _shipped_breaker(**overrides) -> dict:
    """The spike-breaker block the shipped config carries, with test-friendly overrides."""
    config = dict(load_config(str(_CONFIG))["advanced_config"]["spike_breaker"])
    config["warmup_batches"] = 0  # the priming window is not what is under test
    config.update(overrides)
    return config


def _reachable_magnitude(config: dict) -> float:
    r"""The largest magnitude the two reconstruction terms can reach on the negative side.

    Each per-sample Gaussian NLL is bounded below by
    $\tfrac{1}{2}\big(\log 2\pi + \texttt{logvar\_clamp\_lo}\big)$, and each of the two terms sums
    over the whole $H \cdot R$ block, so

    $$\ell \;\ge\; 2 \cdot H \cdot R \cdot \tfrac{1}{2}\big(\log 2\pi + \ell_{\min}\big).$$

    The KL, the prior scale anchor and both auxiliary shape terms are nonnegative and only add, so
    this is the whole downside span of a healthy loss. Computed from the config's own keys, so a
    clamp or horizon change moves the bound with it.

    Args:
        config: A loaded run config.

    Returns:
        The bound's absolute value, in nats.
    """
    vae = config["model_config"]["VAE_model"]
    block = int(vae["horizon"]) * int(vae["raw_per_step"])
    return abs(2.0 * block * 0.5 * (math.log(2.0 * math.pi) + float(vae["logvar_clamp"][0])))


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


def test_the_floor_still_exceeds_any_loss_this_objective_can_reach():
    """Confirmed rather than assumed, at the block that has to stay under it. The bound is
    :func:`_reachable_magnitude`, which reads the shipped horizon, raw grid and clamp rather than
    restating them."""
    config = load_config(str(_CONFIG))

    most_negative = _reachable_magnitude(config)

    assert most_negative < 1.0e5
    assert config["advanced_config"]["spike_breaker"]["ema_floor"] > 100.0 * most_negative


def test_a_sign_crossing_batch_is_not_a_spike(task):
    """With the EMA negative, a batch landing above zero but inside the margin must train. At a zero
    ``ema_floor`` the relative test would discard it; the shipped floor plus the additive margin
    leave it alone.

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


def test_the_shipped_margin_catches_a_finite_blowup(task):
    """A finite jump with no NaN anywhere, at this objective's scale. The non-finite guard has
    nothing to catch; the additive test is the one that must fire, against the raw EMA."""
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
    assert float(metrics["main_loss"]) == float(ema_before), (
        "the logged main_loss was not replaced by the EMA"
    )


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
