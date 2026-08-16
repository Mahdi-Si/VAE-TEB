r"""The loss-spike breaker under a sign-indefinite loss averaged over roughly ten anchors a step.

``main_loss`` here is a learned-variance Gaussian NLL summed over $H \cdot R = 240$ raw samples and
averaged over the anchors the tiling decoded -- about $10.1$ at the shipped stride. The breaker's
relative test is $\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$, which silently assumes a loss
bounded below by zero -- once the EMA is negative it degenerates to "skip every positive batch", the
failure that has already cost this repository a run. The shipped block therefore disables the
relative test with a floor far above any reachable loss and carries the finite-blow-up detection in
``additive_margin``, which compares against the *raw* EMA and keeps working at a negative baseline.

Ported rather than inherited, and the port earns its place for a reason that is different from the
conv-LSTM cell of this row's. There the block itself changed, so no threshold could transfer by
argument. **Here the block does not change at all** -- the same $240$ raw samples over the same
tiling -- so both constants *could* have transferred, and the only way to know whether they do is to
run this encoder and look. One of the two moved and one did not, and this file is where the
distribution that settled both lives.

**The threshold assertions are falsifiable independently of the derivation that set the values.**
The margin is bracketed from above by an arithmetic bound on what this objective can reach, computed
here from the shipped horizon, raw grid and log-variance clamp, and from below by the excursion the
instrumented run measured. Neither side reads the constant under test, so a margin edited to any
value outside that bracket fails rather than moving the goalposts with itself.

``gradient_clip_val`` is bracketed here too, though it is a trainer key rather than a breaker one.
It is the *other* constant the same instrumented run re-derived, and the alternative is two files
each carrying half of one measurement -- so the run's recorded distribution lives here, in one
place, and ``tests/test_config_load.py`` asserts only that it is declared and diverges.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_SIBLING_CONFIG = (
    Path(__file__).resolve().parents[3] / "teb_vae" / "lag_attn_crws" / "configs" / "default.yaml"
)

#: The run every number below comes from: ``configs/smoke_causal.yaml`` at the shipped widths over
#: the committed causal fixture, $600$ optimizer steps, with the clip parked at $10^{9}$ so nothing
#: rescaled the steps the norms were drawn from and the step-granular ramp shortened to $100$ steps
#: so five sixths of the run sits at the shipped learning rate. Two regimes were recorded and they
#: are not interchangeable -- the whole committed shard in one batch, which is the closest reachable
#: analogue of a large-batch production step, and one sample per batch, which is the noisiest regime
#: the fixture can produce. Neither run skipped a batch or bound the clip, so both records describe
#: the objective rather than the guards that were watching it.
#:
#: Every constant here is **provisional**: four in-sample windows are a thinner tail than a
#: production run's, so the distribution describes a memorised window rather than the production
#: objective.

#: The largest excursion of ``main_loss`` above the EMA the breaker carried into the step, over the
#: $500$ steps after the $100$-batch priming window, in the **noisiest** regime -- which is the one
#: the margin has to survive. The breaker's own statistic at this file's ``ema_decay``, rather than a
#: proxy for it. The whole-shard batch reads $18.5$ over the same window.
MEASURED_EXCURSION_MAX = 283.4

#: The $99$th percentile of the same statistic over the same steps ($7.3$ with the whole shard in
#: one batch). The margin must clear it, or ordinary batches are skipped -- which reads in a log
#: exactly like a model that keeps blowing up.
MEASURED_EXCURSION_Q99 = 241.0

#: Pre-clip ``train/grad_norm`` over the same run, **whole-shard batch**: the regime the shipped
#: batch of $128$ is the analogue of, and therefore the one the clip is set from. The batch-1 run
#: reads q99 $2634$ and max $4389$; recorded in the config beside these and deliberately not used.
MEASURED_GRAD_Q99 = 1054.7
MEASURED_GRAD_MAX = 1332.8

#: A loss magnitude this objective genuinely reaches, used to settle a healthy EMA in the
#: behavioural tests below. The instrumented run's post-priming mean with the whole shard in one
#: batch, rather than a value scaled from a sibling's.
_HEALTHY_LOSS = 257.6


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
    this is the whole downside span of a healthy loss. Computed from the config's own keys rather
    than written out, so a clamp or horizon change moves the bracket with it.

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


# --------------------------------------------------------------------------------------
# The threshold this encoder moved, and the one it did not
# --------------------------------------------------------------------------------------
def test_the_clip_moved_and_the_whole_breaker_block_did_not() -> None:
    """Which of the two constants this edge moved, stated as an assertion so the *equality* is a
    measurement rather than an oversight.

    ``additive_margin`` is stated in nats of the summed block, and the encoder edge changes neither
    the block nor the anchor count -- but a faster-fitting encoder could still produce larger
    excursions above the breaker's own EMA, so it was re-measured rather than argued. It came back
    to the conv-LSTM cell's value; the clip did not.
    """
    mine = load_config(str(_CONFIG))
    theirs = load_config(str(_SIBLING_CONFIG))

    assert mine["advanced_config"]["spike_breaker"] == theirs["advanced_config"]["spike_breaker"]
    assert float(mine["advanced_config"]["trainer"]["gradient_clip_val"]) != float(
        theirs["advanced_config"]["trainer"]["gradient_clip_val"]
    )


def test_the_floor_still_exceeds_any_loss_this_objective_can_reach() -> None:
    """Confirmed rather than assumed, at the block that has to stay under it. The bound is
    :func:`_reachable_magnitude`, which reads the shipped horizon, raw grid and clamp rather than
    restating them, so a config that moved any of the three moves this test with it."""
    config = load_config(str(_CONFIG))

    most_negative = _reachable_magnitude(config)

    assert most_negative < 1.0e5
    assert config["advanced_config"]["spike_breaker"]["ema_floor"] > 100.0 * most_negative


def test_the_margin_stays_inside_the_range_the_objective_can_reach() -> None:
    r"""The upper bracket. A margin above the whole reachable magnitude makes the additive test
    **decoration**: the healthy loss cannot swing that far, so nothing short of a genuine divergence
    exceeds $\mathrm{EMA} + \mathrm{margin}$ and the breaker degenerates to its non-finite guard
    alone, with nothing in the log saying so."""
    config = load_config(str(_CONFIG))
    reachable = _reachable_magnitude(config)
    margin = float(config["advanced_config"]["spike_breaker"]["additive_margin"])

    assert margin < reachable


def test_the_margin_clears_the_worst_excursion_the_instrumented_run_measured() -> None:
    """The lower bracket. The breaker's own statistic is the excursion above the EMA it carried into
    the step, and the instrumented run measured it in the noisiest regime the committed fixture can
    produce -- four in-sample windows one at a time, which is $128\\times$ smaller than the shipped
    batch. A margin below that would skip ordinary batches, and a run that skipped ordinary batches
    reads in a log exactly like one that keeps blowing up."""
    margin = float(
        load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]["additive_margin"]
    )

    assert margin > MEASURED_EXCURSION_Q99
    assert margin > MEASURED_EXCURSION_MAX


def test_the_bracket_the_margin_sits_in_is_not_empty() -> None:
    """Stated as its own assertion because the two tests above would both pass on a configuration in
    which no value could satisfy them together. It is a real possibility rather than a hypothetical:
    the ceiling follows the block, which this cell halved against its architecture parent, while the
    floor follows the per-step variance, which the tiling grew."""
    reachable = _reachable_magnitude(load_config(str(_CONFIG)))

    assert MEASURED_EXCURSION_MAX < reachable, (
        f"the measured worst excursion {MEASURED_EXCURSION_MAX} already exceeds the objective's "
        f"reachable magnitude {reachable:.1f}; no additive margin can both fire and stay quiet"
    )


def test_this_encoders_excursions_are_larger_than_the_conv_lstm_cells() -> None:
    """Why the equality above is a measurement rather than an inheritance. This encoder's worst
    measured excursion is *above* the conv-LSTM cell's, so the shared margin sits nearer the floor of
    its bracket here -- close enough that a smaller value would have had to move, and far enough that
    this one did not."""
    margin = float(
        load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]["additive_margin"]
    )

    assert MEASURED_EXCURSION_MAX > 247.7, (
        "this encoder's worst excursion is no longer above the conv-LSTM cell's 247.7, so the "
        "reason recorded for re-measuring rather than inheriting no longer holds"
    )
    assert 1.5 < margin / MEASURED_EXCURSION_MAX < 2.5


def test_the_clip_clears_the_gradient_distribution_the_instrumented_run_measured() -> None:
    """``gradient_clip_val`` is the smallest round value above the pre-clip norm's measured q99, so
    the clip bites on the tail rather than on the body of the distribution -- a threshold below the
    body rescales most steps and turns the optimizer into a sign-descent method with nothing in the
    log saying so."""
    clip = float(load_config(str(_CONFIG))["advanced_config"]["trainer"]["gradient_clip_val"])

    assert MEASURED_GRAD_Q99 < clip <= MEASURED_GRAD_MAX


def test_the_conv_lstm_cells_clip_would_bind_on_this_encoders_body() -> None:
    """The falsifiable half of "it moved". The comparison model's $1000$ sits *below* this encoder's
    measured q99, so inheriting it would rescale more than one step in a hundred -- and both failure
    modes of a wrong clip are silent, a threshold that never fires and one that fires constantly."""
    sibling_clip = float(
        load_config(str(_SIBLING_CONFIG))["advanced_config"]["trainer"]["gradient_clip_val"]
    )

    assert sibling_clip < MEASURED_GRAD_Q99


def test_no_rounder_value_would_still_be_a_guard() -> None:
    """Why the shipped value is rounded to $100$ rather than to $500$ or $1000$: both of the rounder
    candidates sit above the measured maximum, so a blow-up guard set at either would not have bound
    on a single step of the run it was derived from."""
    assert MEASURED_GRAD_MAX < 1500.0
    assert MEASURED_GRAD_MAX < 2000.0
    clip = float(load_config(str(_CONFIG))["advanced_config"]["trainer"]["gradient_clip_val"])
    assert clip % 100.0 == 0.0


# --------------------------------------------------------------------------------------
# The regime the model actually trains in
# --------------------------------------------------------------------------------------
def test_a_sustained_negative_loss_never_spikes(task) -> None:
    """A breaker that skipped here would zero-gradient the entire run. This is the failure the huge
    ``ema_floor`` exists to prevent, and the regime is reachable on this objective: a confident
    decoder's per-sample NLL is negative, so a summed block of $240$ of them crosses zero as soon as
    the forecast is any good. The instrumented run did not get there -- ``main_loss`` stayed inside
    $[+173, +646]$ -- which sizes how far four in-sample windows are from it rather than showing the
    guard is unnecessary."""
    module = task()
    config = _shipped_breaker()

    skips = [
        _skipped(_feed(module, value, config)[0])
        for value in (-100.0, -300.0, -500.0, -700.0, -200.0, -50.0)
    ]

    assert skips == [False] * 6
    assert module._spike_skips_total == 0


def test_a_sign_crossing_batch_is_not_a_spike(task) -> None:
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


def test_the_relative_test_is_genuinely_off(task) -> None:
    """Under the huge floor, even a value far above ``multiplier * EMA`` passes when it stays inside
    the additive margin -- so the margin, not the ratio, is the active finite test."""
    module = task()
    config = _shipped_breaker(additive_margin=0.0)  # isolate the relative test
    for _ in range(5):
        _feed(module, 500.0, config)

    assert not _skipped(_feed(module, 20000.0, config)[0])


def test_the_shipped_margin_catches_a_finite_blowup(task) -> None:
    """A finite jump with no NaN anywhere, re-enacted at this objective's scale. The non-finite
    guard has nothing to catch; the additive test is the one that must fire, against the raw EMA."""
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


def test_the_margin_leaves_an_ordinary_step_to_step_move_alone(task) -> None:
    """The other side of the same threshold, and the one a too-tight margin breaks."""
    module = task()
    config = _shipped_breaker()
    for _ in range(5):
        _feed(module, _HEALTHY_LOSS, config)
    margin = float(config["additive_margin"])

    metrics, _ = _feed(module, module._spike_ema_loss + 0.5 * margin, config)

    assert not _skipped(metrics)


def test_the_measured_excursion_is_left_alone_by_the_shipped_margin(task) -> None:
    """The bracket's lower half, driven through the breaker rather than compared as two numbers: an
    excursion the size of the instrumented run's worst must train, or the shipped configuration
    would have skipped a batch of that run."""
    module = task()
    config = _shipped_breaker()
    for _ in range(5):
        _feed(module, _HEALTHY_LOSS, config)

    metrics, _ = _feed(module, module._spike_ema_loss + MEASURED_EXCURSION_MAX, config)

    assert not _skipped(metrics)


def test_a_non_finite_loss_still_skips(task) -> None:
    """The guard that survives every threshold setting, because it consults none of them."""
    module = task()
    config = _shipped_breaker()
    for value in (_HEALTHY_LOSS, _HEALTHY_LOSS * 1.2, _HEALTHY_LOSS * 0.8):
        _feed(module, value, config)

    for bad in (float("nan"), float("inf")):
        metrics, returned = _feed(module, bad, config)
        assert _skipped(metrics)
        assert torch.isfinite(returned)


def test_max_consecutive_skips_releases_the_breaker_rather_than_deadlocking(task) -> None:
    """After the cap, the next finite batch is force-accepted and the EMA hard re-seeded -- the
    escape from the frozen-EMA deadlock that once cost a run ~160 epochs."""
    module = task()
    config = _shipped_breaker()
    cap = int(config["max_consecutive_skips"])
    for _ in range(5):
        _feed(module, _HEALTHY_LOSS, config)  # settle a healthy EMA
    spike = module._spike_ema_loss + 10.0 * float(config["additive_margin"])

    skips = [_skipped(_feed(module, spike, config)[0]) for _ in range(cap + 2)]

    assert module._spike_forced_accepts_total >= 1, "the escape hatch never fired"
    assert not all(skips), "every batch skipped; the breaker deadlocked"
    # The hard re-seed: after the forced accept the EMA sits at the new level, so the same value is
    # no longer a spike.
    assert not _skipped(_feed(module, spike, config)[0])


def test_the_configured_comparison_metric_is_one_the_task_emits(
    task, stub_batch, perturb_posterior
) -> None:
    """``comparison_metric`` falls back to the returned loss silently when the named metric is
    missing, so the config must name something the task genuinely emits."""
    config = _shipped_breaker()
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert config["comparison_metric"] in metrics


def test_a_skipped_step_still_touches_every_parameter(task) -> None:
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
