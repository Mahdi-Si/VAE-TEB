r"""The loss-spike breaker's shipped block, checked against this model's summed loss.

``main_loss`` here is a learned-variance Gaussian NLL summed over $H \cdot C_{\mathrm{keep}}$
coefficients, the largest forecast block in the family, so it goes negative harder and earlier
than the raw models' loss. The shipped block disables the relative test with an ``ema_floor`` far
above any reachable loss and carries finite blow-up detection in ``additive_margin``, which is
stated in nats of the summed block and was re-derived for it.

The breaker's own behaviour -- the non-finite guard, the escape hatch, the zero-gradient skip, and
every threshold test expressed in margins -- is pinned in ``lag_attn`` and ``lag_attn_rws`` against
the same code, and the margin's retuning against the sibling is pinned in ``test_config_load.py``.
What is checked here is what depends on this model's numbers: that the floor still sits above the
most negative loss this larger block can reach, that the shipped margin lets a sustained negative
loss at this block's scale train, and that the configured comparison metric is one this task
emits.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_fs.tests.conftest import resolve_target_budget

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _shipped_breaker(**overrides) -> dict:
    """The spike-breaker block the shipped config carries, with test-friendly overrides."""
    config = dict(load_config(str(_CONFIG))["advanced_config"]["spike_breaker"])
    config["warmup_batches"] = 0  # the priming window is not what is under test
    config.update(overrides)
    return config


def _feed(module, value, config):
    """Run one breaker decision on a scalar loss and return the metrics it wrote.

    Calls ``_apply_spike_breaker`` directly rather than going through a step: the breaker's own
    behaviour is what is under test, and a real step would need a Trainer to log through.

    Args:
        module: The task.
        value: The returned loss, also used as ``metrics['main_loss']``.
        config: The breaker block.

    Returns:
        The metrics dict, carrying ``spike_skipped``.
    """
    metrics = {
        "total_loss": torch.tensor(float(value)),
        "main_loss": torch.tensor(float(value)),
    }
    module._apply_spike_breaker(torch.tensor(float(value), requires_grad=True), metrics, config)
    return metrics


def test_the_floor_still_exceeds_any_loss_this_objective_can_reach():
    r"""Confirmed rather than assumed, because the block that has to stay under it is wider than
    the raw models'. The per-coefficient Gaussian NLL is bounded below by
    $\tfrac{1}{2}(\log 2\pi + \ell_{\min})$ at the shipped ``logvar_clamp`` floor $\ell_{\min}$, so
    the two reconstruction terms together cannot fall below
    $2 \cdot H C_{\mathrm{keep}} \cdot \tfrac{1}{2}(\log 2\pi + \ell_{\min})$. The KL and the prior
    anchor are both nonnegative and only add.
    """
    config = load_config(str(_CONFIG))
    vae = config["model_config"]["VAE_model"]
    block = vae["horizon"] * len(
        resolve_target_budget(vae["causal_reach_budget_s"]).target_keep_index
    )
    logvar_floor = float(vae["logvar_clamp"][0])

    most_negative = 2.0 * block * 0.5 * (math.log(2.0 * math.pi) + logvar_floor)

    assert most_negative < 0.0
    assert config["advanced_config"]["spike_breaker"]["ema_floor"] > 100.0 * abs(most_negative)


def test_a_sustained_negative_loss_never_spikes(task):
    """A breaker that skipped here would zero-gradient the entire run. Driven at magnitudes this
    summed block actually reaches, so it is the shipped ``additive_margin`` -- the one value in the
    block this model re-derived -- that has to absorb the batch-to-batch swings."""
    module = task()
    config = _shipped_breaker()

    skips = [
        bool(_feed(module, value, config)["spike_skipped"].item())
        for value in (-4000.0, -6000.0, -25000.0, -1500.0, -40000.0, -500.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


def test_the_configured_comparison_metric_is_one_the_task_emits(
    task, stub_batch, perturb_posterior
):
    """``comparison_metric`` falls back to the returned loss silently when the named metric is
    missing, so the config must name something the task genuinely emits."""
    config = load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert config["comparison_metric"] in metrics
