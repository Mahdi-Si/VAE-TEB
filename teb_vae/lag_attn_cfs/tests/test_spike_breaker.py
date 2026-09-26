r"""The loss-spike breaker under a sign-indefinite loss, with the block this cell ships.

``main_loss`` here is a learned-variance Gaussian NLL summed over $H \cdot C_{\mathrm{keep}}$
coefficients and averaged over the anchors the tiling decoded -- far fewer per step than the
two-sided cell's dense set, so the per-step *variance* is much larger. The breaker's relative test is
$\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$, which silently assumes a loss bounded below by
zero -- once the EMA is negative it degenerates to "skip every positive batch", the failure that has
already cost this repository a run. The shipped block therefore disables the relative test with a
floor far above any reachable loss and carries the finite-blow-up detection in ``additive_margin``,
which compares against the *raw* EMA and keeps working at a negative baseline.

The breaker's own mechanics (non-finite skips, the escape hatch, the additive test against a
negative EMA) are the framework's and are tested in ``train/tests/test_spike_breaker.py``. What is
checked here is the block ``configs/default.yaml`` ships: its thresholds against the magnitudes this
objective can reach, its scale-free thresholds against the two-sided sibling's, its behaviour in the
negative-loss regime this model trains in, the comparison metric it names, and that a skipped step
still touches every parameter of this model.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

from .conftest import absolutize_dataset_paths

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_TINY = _CONFIG.parent / "tiny.yaml"
_SIBLING_CONFIG = (
    Path(__file__).resolve().parents[3] / "teb_vae" / "lag_attn_fs" / "configs" / "default.yaml"
)


def _shipped_breaker(**overrides) -> dict:
    """The spike-breaker block the shipped config carries, with test-friendly overrides."""
    config = dict(load_config(str(_CONFIG))["advanced_config"]["spike_breaker"])
    config["warmup_batches"] = 0  # the priming window is not what is under test
    config.update(overrides)
    return config


def _reachable_block_magnitude() -> float:
    r"""The most negative the two reconstruction terms can sum to, in magnitude.

    Each is summed over $H \cdot C_{\mathrm{keep}}$ coefficients and the per-coefficient Gaussian NLL
    is bounded below by $\tfrac{1}{2}(\log 2\pi + \ell_{\min})$ at the ``logvar_clamp`` floor. The
    kept width is resolved from the committed fixture through the tiny config, which is the shipped
    config pointed at it, so a budget or dataset change moves the bound with the model.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    vae = load_config(str(_CONFIG))["model_config"]["VAE_model"]
    budget = resolve_warmup_budget(absolutize_dataset_paths(load_config(str(_TINY))))
    assert budget is not None
    block = int(vae["horizon"]) * len(budget.target.keep_index)
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


# --------------------------------------------------------------------------------------
# The thresholds, against the objective and the sibling
# --------------------------------------------------------------------------------------
def test_the_scale_free_thresholds_match_the_two_sided_siblings():
    """``additive_margin`` is stated in nats of the summed block, so it follows the block and is
    this cell's own. The others are not scales: the ``ema_floor`` is a switch that turns the
    relative test off by sitting above any reachable loss, and the multiplier and the skip cap are
    policy -- so all three are the sibling's, whatever the block size."""
    mine = load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]
    theirs = load_config(str(_SIBLING_CONFIG))["advanced_config"]["spike_breaker"]

    assert mine["ema_floor"] == theirs["ema_floor"] >= 1.0e9
    assert mine["multiplier"] == theirs["multiplier"]
    assert mine["max_consecutive_skips"] == theirs["max_consecutive_skips"]


def test_the_floor_still_exceeds_any_loss_this_objective_can_reach():
    r"""Confirmed rather than assumed, at the block that has to stay under it. The KL and the prior
    anchor are both nonnegative and only add, so the reconstruction bound is the whole of it."""
    most_negative = _reachable_block_magnitude()

    assert most_negative < 1.0e5
    assert _shipped_breaker()["ema_floor"] > 100.0 * most_negative


def test_the_margin_stays_inside_the_range_the_objective_can_reach():
    r"""The **upper** bound that fixes the margin. A margin above the whole reachable magnitude
    makes the additive test **decoration**: no finite value could ever exceed
    $\mathrm{EMA} + \mathrm{margin}$, and the breaker would degenerate to its non-finite guard alone
    with nothing in the log saying so."""
    assert float(_shipped_breaker()["additive_margin"]) < _reachable_block_magnitude()


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
        for value in (-500.0, -690.0, -1200.0, -300.0, -2000.0, -100.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


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


def test_the_relative_test_is_genuinely_off(task):
    """Under the huge floor, even a value far above ``multiplier * EMA`` passes when it stays inside
    the additive margin -- so the margin, not the ratio, is the active finite test."""
    module = task()
    config = _shipped_breaker(additive_margin=0.0)  # isolate the relative test
    for _ in range(5):
        _feed(module, 500.0, config)

    assert not _skipped(_feed(module, 20000.0, config)[0])


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
