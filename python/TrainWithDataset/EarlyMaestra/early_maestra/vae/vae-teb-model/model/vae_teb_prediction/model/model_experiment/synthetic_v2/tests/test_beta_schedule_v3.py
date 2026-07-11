r"""S2-T01a / S2-T01b: per-epoch beta schedule and its precedence.

v3 starts at $K \equiv 0$ and must *grow* $K$ to earn prediction gain, so the training
wrapper resolves a per-epoch $\beta$ from ``loss.beta_schedule`` on the epoch-start seam.
These fast (no-training) tests pin:

* the ``linear_warmup`` / ``constant`` arithmetic of ``_resolve_beta`` (S2-T01a);
* an absent schedule returns the constant ``kld_beta`` (v1 behaviour unchanged);
* the epoch hook writes the resolved $\beta$ into ``hparams['kld_beta']`` (read by
  ``compute_loss_and_metrics``);
* precedence (S2-T01b): a ``beta_schedule`` wins over the (v3 no-op) curriculum, and nulling
  the schedule -- as ``beta_select`` does -- keeps a fixed $\beta$ constant across epochs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    SyntheticSeqVaeLagAttnV2Pl,
    build_model,
)

_CPU = torch.device("cpu")
_TINY = {
    "sequence_length": 16, "d_model": 16, "d_z": 8, "horizon": 4, "warmup_period": 2,
    "c_y": 87, "c_u": 101, "use_up_st": True, "max_lag": 8, "num_heads": 4, "d_head": 4,
    "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
}


def _make_wrapper(*, beta_schedule=None, kld_beta=1e-3, curriculum=None, v3=False):
    cfg = dict(_TINY)
    if v3:
        cfg = {**cfg, "class": "SeqVaeLagAttnV3", "v3": {"posterior_logvar": "residual"}}
    model, _ = build_model(cfg, _CPU)
    return SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=kld_beta, beta_schedule=beta_schedule, curriculum=curriculum)


def _set_epoch(monkeypatch, wrapper, epoch: int) -> None:
    """Force ``current_epoch`` (a trainer-backed property) for a trainer-less unit test."""
    wrapper._test_epoch = epoch
    monkeypatch.setattr(type(wrapper), "current_epoch",
                        property(lambda self: self._test_epoch), raising=False)


# ---------------------------------------------------------------------------
# S2-T01a: _resolve_beta arithmetic
# ---------------------------------------------------------------------------
def test_linear_warmup_matches_formula() -> None:
    start, end, w = 1.0e-5, 1.0e-3, 20
    wrap = _make_wrapper(
        beta_schedule={"kind": "linear_warmup", "start": start, "end": end,
                       "warmup_epochs": w}, kld_beta=0.5)
    for e in (0, 1, w - 1, w, w + 10):
        expected = start + (end - start) * min(1.0, e / w)
        assert abs(wrap._resolve_beta(e) - expected) < 1e-12, e


def test_linear_warmup_then_ramp_arithmetic() -> None:
    """S8-T04: open low (warmup), hold at ``end``, then ramp UP to ``ramp_end``."""
    s, end, w = 1.0e-5, 3.0e-4, 20
    ramp_end, rs, re = 3.0e-3, 55, 90
    wrap = _make_wrapper(beta_schedule={
        "kind": "linear_warmup_then_ramp", "start": s, "end": end, "warmup_epochs": w,
        "ramp_end": ramp_end, "ramp_start_epoch": rs, "ramp_end_epoch": re})
    # warm-up ramp (start -> end)
    assert abs(wrap._resolve_beta(0) - s) < 1e-15
    assert abs(wrap._resolve_beta(10) - (s + (end - s) * 0.5)) < 1e-15
    # held open at ``end`` through [w, rs)
    assert wrap._resolve_beta(w) == end
    assert wrap._resolve_beta(rs - 1) == end
    assert wrap._resolve_beta(rs) == end  # ramp fraction 0 at rs
    # up-ramp (end -> ramp_end)
    mid = end + (ramp_end - end) * ((72 - rs) / (re - rs))
    assert abs(wrap._resolve_beta(72) - mid) < 1e-15
    # held at ramp_end after re
    assert wrap._resolve_beta(re) == ramp_end
    assert wrap._resolve_beta(re + 50) == ramp_end
    # monotone non-decreasing across the whole schedule
    seq = [wrap._resolve_beta(e) for e in range(0, re + 5)]
    assert all(b2 >= b1 - 1e-15 for b1, b2 in zip(seq, seq[1:]))


def test_linear_warmup_then_ramp_misordered_stays_continuous() -> None:
    """ramp_start_epoch < warmup_epochs must NOT jump: ramp_start is floored at warmup."""
    end, ramp_end = 3.0e-4, 3.0e-3
    wrap = _make_wrapper(beta_schedule={
        "kind": "linear_warmup_then_ramp", "start": 0.0, "end": end, "warmup_epochs": 30,
        "ramp_end": ramp_end, "ramp_start_epoch": 10, "ramp_end_epoch": 50})
    # At the warm-up boundary the value is exactly ``end`` (no jump into mid-ramp).
    assert abs(wrap._resolve_beta(29) - end * (29 / 30)) < 1e-15
    assert abs(wrap._resolve_beta(30) - end) < 1e-15
    # The ramp only begins at the floored start (== warmup_epochs == 30), not at epoch 10.
    assert abs(wrap._resolve_beta(40) - (end + (ramp_end - end) * ((40 - 30) / (50 - 30)))) < 1e-15
    # Whole curve is continuous (no step > a small per-epoch delta) and monotone.
    seq = [wrap._resolve_beta(e) for e in range(0, 55)]
    steps = [b2 - b1 for b1, b2 in zip(seq, seq[1:])]
    assert all(s >= -1e-15 for s in steps)              # monotone non-decreasing
    assert max(steps) < (ramp_end - end) / (50 - 30) + 1e-9  # no discontinuous jump


def test_linear_warmup_then_ramp_no_squeeze_holds_end() -> None:
    """With no squeeze window (ramp_end_epoch <= ramp_start_epoch) it holds at ``end``."""
    wrap = _make_wrapper(beta_schedule={
        "kind": "linear_warmup_then_ramp", "start": 0.0, "end": 5e-4, "warmup_epochs": 10,
        "ramp_end": 5e-3, "ramp_start_epoch": 50, "ramp_end_epoch": 50})
    assert wrap._resolve_beta(10) == 5e-4
    assert wrap._resolve_beta(200) == 5e-4


def test_warmup_epochs_nonpositive_returns_end() -> None:
    wrap = _make_wrapper(
        beta_schedule={"kind": "linear_warmup", "start": 1e-5, "end": 1e-3,
                       "warmup_epochs": 0})
    assert wrap._resolve_beta(0) == 1e-3


def test_constant_schedule() -> None:
    wrap = _make_wrapper(beta_schedule={"kind": "constant", "value": 7e-4})
    assert wrap._resolve_beta(5) == 7e-4
    # constant without an explicit value falls back to kld_beta.
    wrap2 = _make_wrapper(beta_schedule={"kind": "constant"}, kld_beta=0.02)
    assert wrap2._resolve_beta(5) == 0.02


def test_absent_schedule_returns_constant_kld_beta() -> None:
    wrap = _make_wrapper(beta_schedule=None, kld_beta=0.003)
    assert wrap._resolve_beta(0) == 0.003
    assert wrap._resolve_beta(1000) == 0.003


def test_unknown_kind_raises() -> None:
    wrap = _make_wrapper(beta_schedule={"kind": "bogus"})
    with pytest.raises(ValueError):
        wrap._resolve_beta(0)


def test_schedule_exposed_in_hparams() -> None:
    sched = {"kind": "linear_warmup", "start": 1e-5, "end": 1e-3, "warmup_epochs": 20}
    wrap = _make_wrapper(beta_schedule=sched)
    assert wrap.hparams["beta_schedule"] == sched


def test_epoch_hook_writes_resolved_beta(monkeypatch) -> None:
    sched = {"kind": "linear_warmup", "start": 0.0, "end": 1.0, "warmup_epochs": 10}
    wrap = _make_wrapper(beta_schedule=sched, kld_beta=99.0)
    _set_epoch(monkeypatch, wrap, 5)
    wrap._on_train_epoch_start_hook()
    assert abs(wrap.hparams["kld_beta"] - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# S2-T01b: precedence
# ---------------------------------------------------------------------------
def test_schedule_wins_over_noop_curriculum_v3(monkeypatch) -> None:
    """For v3 (no set_curriculum_stage) the curriculum is a no-op and the schedule wins."""
    wrap = _make_wrapper(
        beta_schedule={"kind": "constant", "value": 0.123}, kld_beta=99.0,
        curriculum={"enabled": True, "stages": [{"start_epoch": 0, "beta": 0.0}]}, v3=True)
    assert not hasattr(wrap.orig_model, "set_curriculum_stage")
    _set_epoch(monkeypatch, wrap, 3)
    wrap._on_train_epoch_start_hook()
    assert wrap.hparams["kld_beta"] == 0.123


def test_beta_select_null_schedule_keeps_beta_constant(monkeypatch) -> None:
    """With the schedule nulled (as beta_select forces), the swept beta stays constant."""
    wrap = _make_wrapper(beta_schedule=None, kld_beta=0.007)
    for e in range(5):
        _set_epoch(monkeypatch, wrap, e)
        wrap._on_train_epoch_start_hook()
        assert wrap.hparams["kld_beta"] == 0.007
