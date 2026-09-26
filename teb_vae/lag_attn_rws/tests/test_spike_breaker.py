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
behaviour fails here rather than on the production box. The breaker's own mechanics -- the
non-finite skip, the escape hatch, the zero-gradient skipped step -- are the framework's and are
pinned in ``train/tests/test_spike_breaker.py`` and ``teb_vae/lag_attn/tests/test_spike_breaker.py``.
"""
from __future__ import annotations

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
