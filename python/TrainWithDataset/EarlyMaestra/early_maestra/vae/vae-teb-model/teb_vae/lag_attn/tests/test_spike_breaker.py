r"""The loss-spike breaker's real behaviour under this model's sign-indefinite loss.

This model's ``main_loss`` is a Gaussian NLL with a *learned* observation variance, so it is not
bounded below by zero and routinely goes negative. The breaker's relative test is
$\ell > m \cdot \max(\mathrm{EMA}, \mathrm{floor})$ -- note $\max(\mathrm{EMA}, \mathrm{floor})$,
**not** $\max(|\mathrm{EMA}|, \mathrm{floor})$ -- which silently assumes a loss bounded below by
zero. Once the EMA is negative the test degenerates and starts discarding healthy batches, so the
shipped config switches the relative test off with a floor far above any reachable loss and keeps
only the non-finite guard, which never consults the threshold. The first block below is the
evidence for that choice, in both directions.

The other reachable failure is DDP-only, and the last block demonstrates it rather than asserting
it absent: it is the one that would cost days of a headline run to diagnose from a training curve.

No tuning happens here. What the breaker should look like once the real loss's sign and magnitude
are known at production scale is a question a tiny local run cannot answer.
"""
from __future__ import annotations

import math

import torch

from teb_vae.lag_attn.tests.conftest import make_stub_batch
from train.test_utils import FakeStrategy, FakeTrainer


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


def test_a_nan_still_skips_in_that_same_sequence(task):
    """The guard that survives a zero threshold, because it does not consult the threshold."""
    module = task()
    config = _breaker_config()
    for value in (-10.0, -12.0, -11.0):
        _feed(module, value, config)

    metrics, returned = _feed(module, float("nan"), config)

    assert _skipped(metrics)
    assert torch.isfinite(returned), "a skipped step must still return a finite loss"


def test_an_infinite_loss_still_skips(task):
    module = task()
    config = _breaker_config()
    _feed(module, -10.0, config)

    assert _skipped(_feed(module, float("inf"), config)[0])


def test_a_genuine_positive_spike_is_caught(task):
    """The breaker's actual job, in the positive regime where the threshold means something."""
    module = task()
    config = _breaker_config()
    for _ in range(5):
        _feed(module, 2.0, config)  # EMA settles near 2.0

    assert _skipped(_feed(module, 100.0, config)[0])


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


def test_the_shipped_floor_leaves_an_ordinary_positive_batch_alone(task):
    """The shipped configuration: relative detection off, so no healthy batch is ever discarded."""
    module = task()
    shipped = _breaker_config(ema_floor=1.0e9)
    for _ in range(5):
        _feed(module, -0.5, shipped)

    assert not _skipped(_feed(module, 0.3, shipped)[0])
    assert not _skipped(_feed(module, 50.0, shipped)[0])  # relative detection is genuinely off


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


def test_the_ema_never_learns_from_a_skipped_batch(task):
    """Otherwise one spike drags the threshold up and the next spike looks normal."""
    module = task()
    config = _breaker_config()
    for _ in range(5):
        _feed(module, 2.0, config)
    ema_before = module._spike_ema_loss

    _feed(module, 100.0, config)

    assert module._spike_ema_loss == ema_before


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


# --------------------------------------------------------------------------------------
# The DDP failure that is reachable
# --------------------------------------------------------------------------------------
class _PeerWithNonFiniteLoss(FakeStrategy):
    """A second rank whose loss is persistently non-finite.

    Such a peer reports ``is_spike_local = True`` unconditionally (the non-finite guard), so the
    MAX reduce over the skip decision returns 1 -- every rank skips. And its ``forced_local`` is
    gated on ``not is_nonfinite``, so it is always False and the MIN reduce over the force-accept
    returns 0 -- the escape hatch never fires anywhere.

    Two reduces with opposite senses, which one injected scalar cannot express; hence this class.
    """

    def __init__(self) -> None:
        super().__init__(world_size=2)

    def reduce(self, tensor, group=None, reduce_op=None):
        self.reduce_calls.append(reduce_op)
        name = getattr(reduce_op, "name", str(reduce_op)).upper()
        peer_value = 0.0 if "MIN" in name else 1.0
        return torch.minimum(tensor, torch.tensor(peer_value)) if "MIN" in name else torch.maximum(
            tensor, torch.tensor(peer_value)
        )


def test_a_rank_with_a_non_finite_loss_vetoes_the_escape_hatch_forever(task):
    """The freeze, demonstrated.

    A healthy rank, given a healthy loss, every step, forever -- and it never trains. The MAX
    reduce keeps it skipping because a peer is skipping; the MIN reduce keeps the escape hatch shut
    because that same peer, being non-finite, refuses to force-accept. The result is permanent
    zero-gradient training with no error, no exception, and a loss curve that simply stops moving.

    Nothing here is a bug in this model: it is the framework's arithmetic, and this test exists so
    the shape of the failure is on record before a multi-day run hits it. From the outside the only
    tell is ``spike_skipped`` pinned at 1 while ``spike_ema_loss`` never moves.
    """
    module = task()
    module._trainer = FakeTrainer()
    module._trainer.strategy = _PeerWithNonFiniteLoss()
    config = _breaker_config(max_consecutive_skips=3)

    skips = [_skipped(_feed(module, -10.0, config)[0]) for _ in range(20)]

    assert all(skips), "expected the MAX reduce to make this rank skip alongside its peer"
    assert module._spike_forced_accepts_total == 0, (
        "the escape hatch fired; the veto is no longer reachable and this test is obsolete"
    )
    assert module._spike_consecutive == 20  # the run length grows without bound: the freeze


def test_the_escape_hatch_does_fire_when_every_rank_is_healthy(task):
    """The mirror image. Without this, the test above would pass on a breaker that never forces."""
    module = task()
    config = _breaker_config(max_consecutive_skips=3)
    for _ in range(5):
        _feed(module, 2.0, config)  # settle a positive EMA so the spikes below are spikes

    skips = [_skipped(_feed(module, 100.0, config)[0]) for _ in range(6)]

    assert module._spike_forced_accepts_total >= 1
    assert not all(skips), "the escape hatch never fired on a single healthy rank"


def test_a_skipped_step_still_touches_every_parameter(task):
    """The skip path is a zero-gradient step, not an absent one.

    The forward has already armed DDP's reducer, which expects one gradient hook per parameter. A
    ``None`` return, or a loss built from a single parameter, leaves the rest unreduced and the
    next iteration raises "Expected to have finished reduction in the prior iteration".
    """
    module = task()

    _, returned = _feed(module, float("nan"), _breaker_config())
    module.zero_grad(set_to_none=True)
    returned.backward()

    starved = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not starved, f"parameters left without a gradient hook on a skipped step: {starved}"
    assert math.isfinite(float(returned))
