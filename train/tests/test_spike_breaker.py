"""Config-gated loss-spike circuit breaker: train-only, EMA threshold, escape hatch,
metric selection, and DDP MAX/MIN skip synchronisation.

The breaker logic is exercised directly through ``_apply_spike_breaker`` (which never
touches ``self.log``); the train-only gate is exercised through the step dispatcher with
logging stubbed out. Skips are zero-gradient steps, so a skipped batch yields all-zero
gradients rather than a ``None`` return.
"""
import torch
from torch.distributed import ReduceOp

from train.test_utils import FakeStrategy, FakeTrainer, TinyLightningModel


def _cfg(**overrides):
    cfg = {
        "enabled": True,
        "multiplier": 5.0,
        "ema_decay": 0.02,
        "ema_floor": 0.0,
        "warmup_batches": 0,
        "max_consecutive_skips": 25,
        "comparison_metric": "total_loss",
    }
    cfg.update(overrides)
    return cfg


def _model(**kwargs):
    return TinyLightningModel(compile_model=False, **kwargs)


def _feed(model, value, cfg, main=None):
    """Run one breaker decision on a scalar loss; return (metrics, returned_loss)."""
    metrics = {"total_loss": torch.tensor(float(value))}
    if main is not None:
        metrics["main_loss"] = torch.tensor(float(main))
    loss = torch.tensor(float(value), requires_grad=True)
    out = model._apply_spike_breaker(loss, metrics, cfg)
    return metrics, out


# --- S4-T01: gate, train-only, non-finite guard, zero-grad skip ----------------

def test_breaker_off_returns_raw_loss(monkeypatch):
    model = _model()  # no spike_breaker -> disabled
    monkeypatch.setattr(model, "log", lambda *a, **k: None)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    out = model.training_step((x, y), 0)
    expected = torch.nn.functional.mse_loss(model.model(x), y)
    assert torch.allclose(out, expected)


def test_nonfinite_train_batch_is_zero_gradient_step():
    # The full skip contract, end to end: the returned loss is finite, backward through the
    # REAL (poisoned) graph still fires a gradient hook for every parameter DDP's reducer
    # armed, and the on_after_backward guard zeroes the NaN the poisoned graph produced.
    model = _model()
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    real_loss = torch.nn.functional.mse_loss(model.model(x), y)
    poisoned = real_loss * float("nan")  # NaN loss whose graph still reaches every parameter
    metrics = {"total_loss": poisoned.detach()}
    out = model._apply_spike_breaker(poisoned, metrics, _cfg())
    assert metrics["spike_skipped"].item() == 1.0
    assert torch.isfinite(out).item()  # the returned loss is finite despite the NaN input
    model.zero_grad(set_to_none=True)
    out.backward()
    model.on_after_backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in trainable)
    assert all(torch.count_nonzero(p.grad) == 0 for p in trainable)


def test_finite_spike_yields_exact_zero_gradients_without_the_guard():
    # A finite spike needs no gradient guard at all: torch.where routes a zero incoming
    # gradient through the healthy graph, and backward is linear in it.
    model = _model()
    cfg = _cfg(multiplier=5.0, ema_decay=0.5)
    _feed(model, 1.0, cfg)  # seed EMA = 1.0
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    spike_loss = torch.nn.functional.mse_loss(model.model(x), y) + 100.0  # 100x the EMA
    metrics = {"total_loss": spike_loss.detach()}
    out = model._apply_spike_breaker(spike_loss, metrics, cfg)
    assert metrics["spike_skipped"].item() == 1.0
    model.zero_grad(set_to_none=True)
    out.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in trainable)
    assert all(torch.count_nonzero(p.grad) == 0 for p in trainable)


def test_val_stage_never_triggers(monkeypatch):
    model = _model(spike_breaker=_cfg())
    monkeypatch.setattr(model, "log", lambda *a, **k: None)
    x, y = torch.full((2, 4), float("nan")), torch.zeros(2, 4)
    out = model.validation_step((x, y), 0)
    # val is untouched by the breaker: the real (non-finite) loss is returned as-is.
    assert not torch.isfinite(out).item()


def test_nonfinite_returned_loss_skips_even_in_main_loss_mode():
    # The guard must protect the backpropagated total_loss: a NaN total_loss with a
    # finite watched main_loss must still skip, not corrupt the weights.
    model = _model()
    cfg = _cfg(comparison_metric="main_loss")
    metrics = {"total_loss": torch.tensor(1.0), "main_loss": torch.tensor(1.0)}
    out = model._apply_spike_breaker(torch.tensor(float("nan"), requires_grad=True), metrics, cfg)
    assert metrics["spike_skipped"].item() == 1.0
    assert torch.isfinite(out).item()


def test_skip_overwrites_poisoned_loss_metric():
    # A skipped NaN loss must not reach the logger as NaN and poison the epoch aggregate.
    model = _model()
    metrics = {"total_loss": torch.tensor(float("nan"))}
    model._apply_spike_breaker(torch.tensor(float("nan"), requires_grad=True), metrics, _cfg())
    assert torch.isfinite(metrics["total_loss"]).item()


# --- S4-T02: EMA-multiplier threshold + ema_floor ------------------------------

def test_threshold_skips_spike_and_ema_not_polluted():
    model = _model()
    cfg = _cfg(multiplier=5.0, ema_decay=0.5)
    _feed(model, 1.0, cfg)                     # seed EMA = 1.0
    assert model._spike_ema_loss == 1.0
    metrics, _ = _feed(model, 100.0, cfg)      # 100 > 5*1 -> skip
    assert metrics["spike_skipped"].item() == 1.0
    assert model._spike_ema_loss == 1.0        # a skipped spike must not move the EMA
    metrics, _ = _feed(model, 1.0, cfg)        # normal loss trains again
    assert metrics["spike_skipped"].item() == 0.0


def test_ema_floor_clamps_comparison_base():
    # With a floor, the comparison base cannot drop below it, so a mid-size loss stays
    # under threshold; without the floor the same loss spikes.
    with_floor = _model()
    _feed(with_floor, 1.0, _cfg(ema_floor=10.0))
    metrics, _ = _feed(with_floor, 40.0, _cfg(ema_floor=10.0))   # 40 vs 5*max(1,10)=50
    assert metrics["spike_skipped"].item() == 0.0

    no_floor = _model()
    _feed(no_floor, 1.0, _cfg(ema_floor=0.0))
    metrics, _ = _feed(no_floor, 40.0, _cfg(ema_floor=0.0))      # 40 vs 5*1=5
    assert metrics["spike_skipped"].item() == 1.0


# --- Additive margin: sign-agnostic finite-spike detection ----------------------
# The relative test compares against max(EMA, ema_floor) and so cannot express "a finite jump
# above a NEGATIVE baseline"; the additive test compares against the raw EMA and can. These
# tests drive the negative-EMA regime directly, with the relative test disabled the same way a
# negative-NLL consumer disables it (a floor far above any reachable loss).

def test_additive_margin_catches_finite_spike_over_negative_ema():
    model = _model()
    cfg = _cfg(ema_floor=1.0e9, additive_margin=3.0)
    for _ in range(3):
        _feed(model, -0.5, cfg)                    # settle a negative EMA
    ema_before = model._spike_ema_loss
    assert ema_before is not None and ema_before < 0.0

    metrics, _ = _feed(model, 5.0, cfg)            # 5.0 > -0.5 + 3.0 -> skip

    assert metrics["spike_skipped"].item() == 1.0
    assert model._spike_ema_loss == ema_before     # a skipped spike must not move the EMA


def test_additive_margin_leaves_fluctuations_within_margin_alone():
    # The sign-crossing batch that a zero ema_floor would have discarded: +0.3 above an EMA of
    # -0.5 is inside the margin and must train.
    model = _model()
    cfg = _cfg(ema_floor=1.0e9, additive_margin=3.0)
    for _ in range(3):
        _feed(model, -0.5, cfg)

    metrics, _ = _feed(model, 0.3, cfg)

    assert metrics["spike_skipped"].item() == 0.0


def test_additive_margin_zero_disables_the_test():
    # 0.0 must mean "off", not "margin of zero": half of all healthy batches land above the EMA,
    # and a zero margin would skip every one of them.
    model = _model()
    cfg = _cfg(ema_floor=1.0e9, additive_margin=0.0)
    _feed(model, -0.5, cfg)

    metrics, _ = _feed(model, 50.0, cfg)           # far above the EMA, but both tests are off

    assert metrics["spike_skipped"].item() == 0.0


def test_additive_margin_respects_warmup():
    model = _model()
    cfg = _cfg(ema_floor=1.0e9, additive_margin=3.0, warmup_batches=3)
    _feed(model, -0.5, cfg)                        # batch 1 seeds the EMA
    metrics, _ = _feed(model, 50.0, cfg)           # batch 2: inside warmup -> never skip
    assert metrics["spike_skipped"].item() == 0.0


def test_additive_and_relative_tests_are_ored():
    # With both enabled, either firing skips: a spike below the relative threshold but above
    # EMA + margin is still caught.
    model = _model()
    cfg = _cfg(multiplier=5.0, ema_floor=0.0, additive_margin=2.0)
    _feed(model, 1.0, cfg)                         # EMA = 1.0; relative threshold = 5.0
    metrics, _ = _feed(model, 4.0, cfg)            # 4 < 5*1, but 4 > 1 + 2 -> skip
    assert metrics["spike_skipped"].item() == 1.0


# --- S4-T03: escape hatch + comparison_metric ----------------------------------

def test_escape_hatch_force_accepts_after_cap():
    n = 3
    model = _model()
    cfg = _cfg(max_consecutive_skips=n)
    _feed(model, 1.0, cfg)                      # seed EMA = 1.0
    for _ in range(n):                          # n consecutive spikes -> n skips
        metrics, _ = _feed(model, 100.0, cfg)
        assert metrics["spike_skipped"].item() == 1.0
    metrics, _ = _feed(model, 100.0, cfg)       # the (n+1)th spike is force-accepted
    assert metrics["spike_skipped"].item() == 0.0
    assert model._spike_forced_accepts_total == 1
    assert model._spike_ema_loss == 100.0       # EMA hard re-seeded on force-accept


def test_comparison_metric_watches_main_loss():
    model = _model()
    cfg = _cfg(comparison_metric="main_loss")
    _feed(model, 1.0, cfg, main=1.0)                 # seed EMA from main_loss = 1.0
    metrics, _ = _feed(model, 1000.0, cfg, main=1.0)  # total huge but main normal -> train
    assert metrics["spike_skipped"].item() == 0.0
    metrics, _ = _feed(model, 1.0, cfg, main=100.0)   # main spikes -> skip
    assert metrics["spike_skipped"].item() == 1.0


# --- S4-T04: DDP skip-decision sync --------------------------------------------

def test_reduce_flags_use_max_and_skip_if_any_rank_skips():
    # One MAX-reduce carries both cross-rank facts as a tensor (no .item() sync): element 0
    # is the skip decision, element 1 the non-finite flag.
    model = _model()
    trainer = FakeTrainer(world_size=2)
    trainer.strategy = FakeStrategy(world_size=2, other_value=1.0)  # other rank flags both
    model._trainer = trainer
    flags = model._reduce_spike_flags(torch.tensor([0.0, 0.0]))
    assert trainer.strategy.reduce_calls[-1] == ReduceOp.MAX
    assert flags.tolist() == [1.0, 1.0]  # skip if any rank skips / any rank non-finite


def test_remote_skip_flag_skips_the_local_batch():
    # A healthy local loss must still skip when another rank spiked: the reduced flags,
    # not the local ones, drive the decision.
    model = _model()
    trainer = FakeTrainer(world_size=2)
    trainer.strategy = FakeStrategy(world_size=2, other_value=1.0)  # other rank skips
    model._trainer = trainer
    metrics, _ = _feed(model, 1.0, _cfg())
    assert metrics["spike_skipped"].item() == 1.0


def test_force_accept_is_vetoed_while_any_rank_is_nonfinite():
    # The escape hatch used to MIN-reduce "finite and past the cap"; the equivalent
    # single-collective form is "past the cap AND no rank non-finite". A remote NaN
    # arrives as flags[1] = 1 through the MAX-reduce and must keep vetoing the force.
    model = _model()
    trainer = FakeTrainer(world_size=2)
    trainer.strategy = FakeStrategy(world_size=2, other_value=1.0)  # remote: spike + NaN
    model._trainer = trainer
    cfg = _cfg(max_consecutive_skips=2)
    for _ in range(5):  # far past the cap
        metrics, _ = _feed(model, 1.0, cfg)
        assert metrics["spike_skipped"].item() == 1.0
    assert int(model._spike_forced_accepts_total) == 0


def test_reduce_is_noop_on_single_device():
    model = _model()
    trainer = FakeTrainer(world_size=1)
    trainer.strategy = FakeStrategy(world_size=1, other_value=1.0)
    model._trainer = trainer
    flags = torch.tensor([0.0, 0.0])
    assert model._reduce_spike_flags(flags) is flags  # returned unchanged
    assert trainer.strategy.reduce_calls == []        # reduce never called
