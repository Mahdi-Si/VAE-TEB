"""``build_lr_scheduler`` warmup branch and milestone-shift correctness.

The warmup path composes a ``LinearLR`` ramp with a ``MultiStepLR`` decay via
``SequentialLR``; the decay milestones are shifted back by ``lr_warmup_epochs`` so the
LR drop lands at its intended *absolute* epoch. These tests step a scheduler across
epochs and assert the drop happens at the absolute milestone (which fails if the shift
is omitted), that the no-warmup path is unchanged, and that ``None`` is returned when
nothing is configured.
"""
import pytest
import torch

from train.test_utils import TinyLightningModel


def _sgd(base_lr: float = 1.0) -> torch.optim.Optimizer:
    """A one-parameter SGD so ``param_groups[0]['lr']`` tracks the schedule cleanly."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=base_lr)


def _lr_trace(scheduler, optimizer, n_epochs: int):
    """Record the LR at epoch 0 then after each of ``n_epochs`` ``step()`` calls."""
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(n_epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def _model(**hparams) -> TinyLightningModel:
    model = TinyLightningModel()
    for key, value in hparams.items():
        setattr(model.hparams, key, value)
    return model


def test_no_milestones_no_warmup_returns_none():
    model = _model(lr_milestones=[], lr_warmup_epochs=0)
    assert model.build_lr_scheduler(_sgd()) is None


def test_bare_multistep_drops_at_milestone():
    model = _model(lr_milestones=[3], lr_gamma=0.1, lr_warmup_epochs=0)
    optimizer = _sgd(base_lr=1.0)
    scheduler = model.build_lr_scheduler(optimizer)["scheduler"]
    lrs = _lr_trace(scheduler, optimizer, n_epochs=5)
    # No warmup: full LR through epoch 2, decays at epoch 3.
    assert lrs[2] == pytest.approx(1.0)
    assert lrs[3] == pytest.approx(0.1)


def test_warmup_ramps_then_decays_at_absolute_milestone():
    warmup, milestone = 3, 5
    model = _model(lr_milestones=[milestone], lr_gamma=0.1, lr_warmup_epochs=warmup)
    optimizer = _sgd(base_lr=1.0)
    scheduler = model.build_lr_scheduler(optimizer)["scheduler"]
    lrs = _lr_trace(scheduler, optimizer, n_epochs=7)

    # Warmup ramps up from start_factor*base towards base.
    assert lrs[0] == pytest.approx(0.1)          # start_factor = 0.1
    assert lrs[0] < lrs[2] < lrs[3]              # strictly increasing during warmup
    assert lrs[4] == pytest.approx(1.0)          # at full LR, decay not yet fired
    # The decay lands at the ABSOLUTE milestone (epoch 5), not milestone + warmup (8).
    assert lrs[5] == pytest.approx(0.1)
    assert lrs[6] == pytest.approx(0.1)


def test_warmup_reachable_via_constructor():
    # lr_warmup_epochs must be a real constructor knob, not settable only via hparams.
    model = TinyLightningModel(lr_warmup_epochs=3, lr_milestones=[5], lr_gamma=0.1)
    optimizer = _sgd(base_lr=1.0)
    scheduler = model.build_lr_scheduler(optimizer)["scheduler"]
    lrs = _lr_trace(scheduler, optimizer, n_epochs=6)
    assert lrs[0] == pytest.approx(0.1)   # warmup start_factor active
    assert lrs[4] == pytest.approx(1.0)   # full LR before the decay
    assert lrs[5] == pytest.approx(0.1)   # decay at the absolute milestone


def test_warmup_only_no_milestones_ramps_and_holds():
    model = _model(lr_milestones=[], lr_warmup_epochs=2)
    optimizer = _sgd(base_lr=1.0)
    scheduler = model.build_lr_scheduler(optimizer)["scheduler"]
    lrs = _lr_trace(scheduler, optimizer, n_epochs=4)
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[2] == pytest.approx(1.0)          # reached base at end of warmup
    assert lrs[3] == pytest.approx(1.0)          # holds, no decay configured
