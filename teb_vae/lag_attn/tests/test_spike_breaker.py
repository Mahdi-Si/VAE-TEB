r"""The loss-spike breaker's real behaviour under this model's sign-indefinite loss.

This model's ``main_loss`` is a Gaussian NLL with a *learned* observation variance, so it is not
bounded below by zero and routinely goes negative. The breaker's relative test is
$\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$ -- note $\max(\mathrm{EMA}, \mathrm{floor})$,
**not** $\max(|\mathrm{EMA}|, \mathrm{floor})$ -- which silently assumes a loss bounded below by
zero. Once the EMA is negative the test degenerates and starts discarding healthy batches, so the
shipped config switches the relative test off with a floor far above any reachable loss. The first
block below is the evidence for that choice.

The 2026-07 baseline then measured what the non-finite guard alone costs: a *finite* blow-up
(``main_loss`` $\approx -0.5 \to +4..+10$, no NaN in the run) sailed through and the run was lost,
so the shipped config also enables ``additive_margin`` -- skip when
$\ell > \mathrm{EMA} + \mathrm{margin}$, against the raw EMA, so it works at a negative baseline.
The additive test, the escape hatch and the DDP skip synchronisation are the framework's and are
tested in ``train/tests/test_spike_breaker.py``. What stays here depends on this model's loss: its
sign, its perm-free ``main_loss``, and its real autograd graph on a skipped step.
"""
from __future__ import annotations

import math

import torch

from teb_vae.lag_attn.tests.conftest import make_stub_batch


def _breaker_config(**overrides) -> dict:
    """The shipped spike-breaker block, with ``warmup_batches`` shortened for the tests."""
    config = {
        "enabled": True,
        "multiplier": 5.0,
        "ema_decay": 0.02,
        "ema_floor": 0.0,
        "warmup_batches": 0,
        "max_consecutive_skips": 25,
        "comparison_metric": "main_loss",
    }
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
# A sign-indefinite loss at ema_floor: 0.0
# --------------------------------------------------------------------------------------
def test_a_sustained_negative_loss_never_spikes(task):
    """The regime this model actually trains in.

    ``ema_ref = max(ema_before, 0.0) = 0.0`` while the EMA is negative, and no negative loss
    exceeds ``5 * 0.0``. A breaker that skipped here would zero-gradient the entire run.
    """
    module = task()
    config = _breaker_config()

    skips = [
        _skipped(_feed(module, value, config)[0])
        for value in (-10.0, -12.0, -50.0, -3.0, -80.0, -1.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


def test_at_a_zero_floor_a_negative_ema_makes_every_positive_batch_a_spike(task):
    """Why the shipped config does NOT use ``ema_floor: 0.0``.

    ``max(EMA, 0.0)`` is ``0.0`` once the EMA is negative, so the test degenerates to
    ``watched > 0`` and an ordinary batch landing just above zero during the sign crossing is
    treated as a blow-up: its gradient is discarded and the value logged as ``main_loss`` is
    replaced by the EMA. The EMA updates only on accepted batches, so it stays negative and the
    run keeps dropping precisely its hardest batches.

    The trainer this replaced gated on ``ema_before > 0.0`` and never took the spike branch at all
    in this regime, so the two do **not** agree -- despite a claim to the contrary that this test
    exists to refute.
    """
    module = task()
    zero_floor = _breaker_config(ema_floor=0.0)
    for _ in range(5):
        _feed(module, -0.5, zero_floor)  # a healthy negative-loss run
    assert module._spike_ema_loss < 0.0

    metrics, _ = _feed(module, 0.3, zero_floor)  # an ordinary fluctuation, not a blow-up

    assert _skipped(metrics)
    assert float(metrics["main_loss"]) < 0.0, "the logged main_loss was replaced by the EMA"


def test_the_shipped_floor_still_catches_a_nan(task):
    """The protection that survives, and the reason the breaker stays enabled at all.

    The non-finite check never consults the threshold, so it is unaffected by the floor. Without
    it a NaN loss would write NaN gradients into every weight and the run would be dead with no
    error -- which is worse than any spike.
    """
    module = task()
    shipped = _breaker_config(ema_floor=1.0e9)
    for _ in range(5):
        _feed(module, -0.5, shipped)

    metrics, returned = _feed(module, float("nan"), shipped)

    assert _skipped(metrics)
    assert torch.isfinite(returned)


# Note the ``enabled`` gate is not tested here. It lives in the framework's step dispatcher, not in
# the breaker itself -- ``_apply_spike_breaker`` runs whatever it is handed -- and the framework's
# own suite covers it. What this model owns is whether its config block reaches the module at all,
# which the trainer's wiring test asserts.


# --------------------------------------------------------------------------------------
# The periodic control must not look like a spike
# --------------------------------------------------------------------------------------
def test_periodic_perm_steps_do_not_trip_the_breaker(task, perturb_posterior):
    r"""Why ``comparison_metric: main_loss`` is configured.

    The permutation control fires every ``perm_every_n_batches`` steps and adds
    $\lambda_{\mathrm{perm}} L_{\mathrm{perm}}$ to the returned loss. A breaker watching the
    returned loss would see a periodic step change, settle its EMA between the two levels, and
    start skipping every perm step -- reacting to its own statistic's artefact. Watching the
    perm-free ``main_loss`` removes the periodicity from what it sees.

    Driven with the real losses from the real task, so this fails if the metric ever stops being
    perm-free.
    """
    module = task()
    perturb_posterior(module.orig_model)
    config = _breaker_config(warmup_batches=2)
    batch = make_stub_batch()

    skips = []
    for batch_idx in range(8):
        loss, metrics = module.compute_loss_and_metrics(batch, batch_idx, "train")
        module._apply_spike_breaker(loss, metrics, config)
        skips.append(_skipped(metrics))

    assert not any(skips), "the breaker skipped a step; the perm jump is reaching its statistic"


def test_the_escape_hatch_does_fire_when_every_rank_is_healthy(task):
    """On a single healthy rank the escape hatch force-accepts a run of spikes past the cap."""
    module = task()
    config = _breaker_config(max_consecutive_skips=3)
    for _ in range(5):
        _feed(module, 2.0, config)  # settle a positive EMA so the spikes below are spikes

    skips = [_skipped(_feed(module, 100.0, config)[0]) for _ in range(6)]

    assert module._spike_forced_accepts_total >= 1
    assert not all(skips), "the escape hatch never fired on a single healthy rank"


# --------------------------------------------------------------------------------------
# A skipped step under DDP
# --------------------------------------------------------------------------------------
def test_a_skipped_step_still_touches_every_parameter(task):
    """The skip path is a zero-gradient step, not an absent one.

    The forward has already armed DDP's reducer, which expects one gradient hook per parameter. A
    ``None`` return, or a loss built from a single parameter, leaves the rest unreduced and the
    next iteration raises "Expected to have finished reduction in the prior iteration". The
    breaker therefore returns ``torch.where`` over the REAL loss -- backward still traverses the
    full graph, so every hook fires -- and ``on_after_backward`` zeroes the NaN that a poisoned
    graph pushes through the zero incoming gradient.
    """
    module = task()

    # A non-finite loss whose autograd graph spans every trainable parameter, as the real
    # loss does; a leaf NaN would prove nothing about the hooks.
    real = torch.stack([p.sum() for p in module.parameters() if p.requires_grad]).sum()
    poisoned = real * float("nan")
    metrics = {"total_loss": poisoned.detach(), "main_loss": poisoned.detach()}
    returned = module._apply_spike_breaker(poisoned, metrics, _breaker_config())
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
