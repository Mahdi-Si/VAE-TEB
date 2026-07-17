"""Opt-in ``torch.compile`` via the ``compile_model`` flag."""
import torch

from train.pl_model_base import LightningModelBase
from train.test_utils import TinyModule


class _Bare(LightningModelBase):
    """Minimal subclass that keeps the base ``compile_model`` default (True)."""

    def compute_loss_and_metrics(self, batch, batch_idx, stage):
        raise NotImplementedError


def test_compile_model_false_is_eager():
    base = TinyModule()
    wrapper = _Bare(base, compile_model=False)
    assert wrapper.model is base


def test_default_compiles_exactly_once(monkeypatch):
    calls = []

    def fake_compile(module, *args, **kwargs):
        calls.append(module)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    base = TinyModule()
    wrapper = _Bare(base)  # compile_model omitted -> base default True
    assert len(calls) == 1
    assert calls[0] is base
    assert wrapper.model is base  # our fake returns the module unchanged
