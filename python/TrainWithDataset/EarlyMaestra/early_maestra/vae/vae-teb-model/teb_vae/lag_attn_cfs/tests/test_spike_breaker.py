r"""The loss-spike breaker under a sign-indefinite loss averaged over roughly ten anchors a step.

``main_loss`` here is a learned-variance Gaussian NLL summed over $H \cdot C_{\mathrm{keep}} = 2940$
coefficients and averaged over the anchors the tiling decoded -- about $10.1$ at the shipped stride
against the two-sided cell's $\approx 240$. Both halves of that matter and they pull in opposite
directions: the block is $0.63\times$, so the loss *level* is smaller, while the anchor count is a
twenty-fourth, so the per-step *variance* is much larger. The breaker's relative test is
$\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$, which silently assumes a loss bounded below by
zero -- once the EMA is negative it degenerates to "skip every positive batch", the failure that has
already cost this repository a run. The shipped block therefore disables the relative test with a
floor far above any reachable loss and carries the finite-blow-up detection in ``additive_margin``,
which compares against the *raw* EMA and keeps working at a negative baseline.

Ported rather than inherited, and the port earns its place: every threshold in that block is stated
in nats of the summed block, so none of the two-sided cell's values transfer and none of its tests
are evidence about this configuration. Each test below drives the breaker with the block **this**
``configs/default.yaml`` ships, at magnitudes this objective actually reaches -- the instrumented run
the config's comments record saw ``main_loss`` inside $[-7556, +8487]$ within 600 steps -- so a config
edit that regressed the behaviour fails here rather than on the production box.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.config import load_config

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_SIBLING_CONFIG = (
    Path(__file__).resolve().parents[3] / "teb_vae" / "lag_attn_fs" / "configs" / "default.yaml"
)

#: Surviving target channels at the shipped warm-up budget. Written out because this file reasons
#: about the block's arithmetic bound rather than building a model.
KEPT_TARGET_CHANNELS = 98

#: A loss magnitude this objective genuinely reaches, used to settle a healthy negative EMA. Chosen
#: from the instrumented run rather than scaled from the sibling's: that run's post-ramp half sat at
#: $-119 \pm 295$ with a minimum of $-7556$, so a few hundred negative is the regime the breaker has
#: to be quiet in.
_HEALTHY_LOSS = -500.0

#: The worst excursion above the EMA the instrumented run measured at $H = 30$, in the noisiest
#: regime the committed fixture can produce (batch 1, four distinct batches), after the breaker's own
#: priming window. The margin has to clear it or ordinary batches are skipped -- which reads in the
#: log exactly like a model that keeps blowing up. It is also the value the conv-Transformer cell's
#: suite pins, because the two cells share one margin across the encoder edge and this is the larger
#: of the two measurements.
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
# The thresholds this cell had to move, and the one it did not
# --------------------------------------------------------------------------------------
def test_the_margin_moved_up_and_the_floor_did_not():
    """One of the two moved and the other did not, and the reason is the whole design of the block.
    ``additive_margin`` is stated in nats of the summed block, so it follows the block -- which is
    $30 \\times 98 = 2940$ coefficients against the comparison model's $30 \\times 78 = 2340$, so the
    margin sits above that model's.

    **The direction reversed with the horizon.** At $H = 15$ this block was $1470$, the smaller of
    the two, and this was the first re-derivation in the family to move a threshold *downwards*; at
    $H = 30$ the same rule moves it up instead. The rule did not change and the block did, which is
    what makes the sign of this comparison worth asserting rather than assuming.

    The ``ema_floor`` is not a scale at all: it is a switch that turns the relative test off by
    sitting above any reachable loss, and "above any reachable loss" is satisfied by the same number
    at either block size."""
    mine = load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]
    theirs = load_config(str(_SIBLING_CONFIG))["advanced_config"]["spike_breaker"]

    assert mine["additive_margin"] > theirs["additive_margin"]
    assert mine["ema_floor"] == theirs["ema_floor"] >= 1.0e9
    assert mine["multiplier"] == theirs["multiplier"]
    assert mine["max_consecutive_skips"] == theirs["max_consecutive_skips"]


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
    r"""The **upper** of the two bounds that fix the margin, and at this horizon it is the binding
    one. A margin above the whole reachable magnitude makes the additive test **decoration**: no
    finite value could ever exceed $\mathrm{EMA} + \mathrm{margin}$, and the breaker would degenerate
    to its non-finite guard alone with nothing in the log saying so.

    The lower bound is
    :func:`test_the_margin_clears_the_worst_excursion_the_instrumented_run_measured`, and the two
    are now close: the measured worst excursion is $5090$ and the reachable magnitude is
    $\approx 9296$, so the admissible band is roughly $[5090, 9296]$ and the shipped $9000$ sits
    near its top. At $H = 15$ the same band was $[1089, 4646]$ and the shipped $3000$ sat in the
    middle of it -- the band did not scale with the block, because the excursion grew faster than
    the block did. If a future change narrows it to nothing, this pair of tests is what says so.
    """
    config = load_config(str(_CONFIG))
    vae = config["model_config"]["VAE_model"]
    # Two reconstruction terms, each summed over the block, each per-coefficient value bounded below
    # by 0.5 * (log 2pi + logvar_clamp_lo).
    reachable = abs(
        2.0
        * vae["horizon"]
        * KEPT_TARGET_CHANNELS
        * 0.5
        * (math.log(2.0 * math.pi) + float(vae["logvar_clamp"][0]))
    )
    margin = float(config["advanced_config"]["spike_breaker"]["additive_margin"])

    assert margin < reachable

    # The band is real rather than nominal: the margin has to fit between the measured excursion and
    # the reachable magnitude, and at this block those two are within a factor of two of each other.
    # Asserted here rather than left to the pair of tests, because "there is still room" is the
    # property that would be lost silently by a further horizon increase.
    assert _WORST_MEASURED_EXCURSION < margin < reachable


def test_the_margin_clears_the_worst_excursion_the_instrumented_run_measured():
    """The **lower** of the two bounds that fix the margin. The breaker's own statistic is the
    excursion above the EMA, and the run recorded in the config's comments measured its maximum at
    $5{,}090$ nats after the priming window -- in the noisiest regime the committed fixture can
    produce, four distinct single-sample batches, which is $32\\times$ smaller than the shipped
    batch. A margin below that would skip ordinary batches, which reads in the log exactly like a
    model that keeps blowing up."""
    margin = float(
        load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]["additive_margin"]
    )

    assert margin > _WORST_MEASURED_EXCURSION


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
    """The other side of the same threshold, and the one a too-tight margin breaks -- which is the
    live risk here rather than a hypothetical, because the re-derivation moved the margin **down**.
    """
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


# --------------------------------------------------------------------------------------
# The horizon weighting, and why it leaves these two constants alone
# --------------------------------------------------------------------------------------
def test_the_horizon_weight_redistributes_the_block_rather_than_rescaling_it():
    r"""The one mechanism that could have invalidated both constants above, and the property that
    stops it.

    ``additive_margin`` and ``gradient_clip_val`` are both stated in nats of the **summed block**,
    so anything that changes the block's scale puts both out of date at once -- and silently, since
    a shrunken block is a perfectly healthy-looking loss curve at a smaller number. A decaying
    horizon weight is exactly such a thing: $2^{-\tau/\lambda}$ over $H$ steps sums to far less than
    $H$, so an un-renormalised weight would shrink the reconstruction against a KL that did not
    move -- an increase in the effective $\beta$ wearing a horizon weight's name.

    :func:`~teb_vae.lag_attn_rws.nets.losses.horizon_decay_weight` renormalises to $\sum_\tau w_\tau
    = H$, so only the *distribution* across horizon steps moves. Asserted as a ratio against the
    unweighted sum on a symmetric per-step score: what has to hold is that the two are the same
    number, not that the weight is uniform -- it is emphatically not, and the spread is asserted
    beside the ratio so a weight that had quietly become flat would fail here too.
    """
    from teb_vae.lag_attn_rws.nets.losses import horizon_decay_weight

    vae = load_config(str(_CONFIG))["model_config"]["VAE_model"]
    horizon = int(vae["horizon"])
    halflife = float(vae["horizon_weight_halflife_steps"])
    weight = horizon_decay_weight(halflife, horizon)

    # A per-step score that is the same at every horizon step, so the two block sums differ by the
    # weight alone. Any constant would do; one is the one that makes the ratio readable.
    per_step = torch.ones(horizon, dtype=torch.float64)
    unweighted = float(per_step.sum())
    weighted = float((per_step * weight.to(torch.float64)).sum())

    assert weighted == pytest.approx(unweighted, rel=1e-4)
    # And it is a real reweighting rather than a no-op: the near steps carry several times what the
    # far ones do, which is the whole mechanism.
    assert float(weight[0]) / float(weight[-1]) > 3.0


def test_an_un_renormalised_decay_would_have_moved_the_block_enough_to_matter():
    r"""Non-vacuity for the ratio above, and the number that says why the renormalisation is not a
    detail.

    Without it the block would shrink by $H / \sum_\tau 2^{-\tau/\lambda}$ -- a factor near two at
    the shipped half-life -- which is a change of the same order as the gap between the measured
    worst excursion and the shipped margin. So the admissible band this file brackets would have
    moved out from under both constants, with no column in any run's CSV saying so.
    """
    from teb_vae.lag_attn_rws.nets.losses import horizon_decay_weight

    vae = load_config(str(_CONFIG))["model_config"]["VAE_model"]
    horizon = int(vae["horizon"])
    halflife = float(vae["horizon_weight_halflife_steps"])

    raw = torch.tensor(
        [2.0 ** (-step / halflife) for step in range(horizon)], dtype=torch.float64
    )
    shrinkage = horizon / float(raw.sum())
    normalised = horizon_decay_weight(halflife, horizon).to(torch.float64)

    assert shrinkage > 1.5
    assert torch.allclose(normalised, raw * shrinkage, rtol=1e-5)
