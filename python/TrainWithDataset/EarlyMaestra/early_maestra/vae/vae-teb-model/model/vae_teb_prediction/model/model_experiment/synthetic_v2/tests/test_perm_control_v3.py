r"""S3-T01: source-permutation control in the training wrapper (readout only).

The wrapper deranges ``source_state`` along the batch axis and re-scores the forecast,
yielding ``kld_shuffled`` / ``kld_shuffled_ratio`` (the demoted KL-space readout, near $1$
on an honest model per v3 Finding F2) and ``feat_loss_perm`` / ``shuffle_penalty`` (the
prediction-space penalty, which *does* discriminate).

Four invariants are asserted here:

* the metrics appear on scheduled train batches and on **every** validation batch, are
  zero-filled otherwise, and are never ``None`` (``_log_metrics`` cannot log a ``None``);
* a batch with $B < 2$ is skipped without raising (it cannot be deranged);
* the schedule is rank-invariant (simulated over a 2-rank ``batch_idx`` sweep);
* under v1 the metrics are absent and nothing else changes.

The control is a **no-grad readout**: ``lambda_perm`` is pinned to $0$ in the synthetic
pipeline, so ``total_loss`` must come back untouched and no autograd graph may be built for
the permuted branch. That last property is what licenses ``find_unused_parameters=False``
and is asserted in ``test_ddp_strategy_v3.py``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    SyntheticSeqVaeLagAttnV2Pl,
    build_model,
)

_CPU = torch.device("cpu")
_TINY = {
    "sequence_length": 32, "d_model": 16, "d_z": 8, "horizon": 4, "warmup_period": 2,
    "c_y": 87, "c_u": 101, "use_up_st": True, "max_lag": 8, "num_heads": 4, "d_head": 4,
    "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
}
_V3 = {**_TINY, "class": "SeqVaeLagAttnV3",
       "v3": {"posterior_logvar": "residual", "kld_support": "anchor",
              "logvar_bound": "smooth", "lambda_perm": 0.0, "perm_every_n_batches": 4}}

#: The four readouts S3-T01 adds. ``perm_loss`` is deliberately NOT among them: the
#: synthetic wrapper never optimises the control, so the column would be a constant zero.
_PERM_KEYS = ("kld_shuffled", "kld_shuffled_ratio", "feat_loss_perm", "shuffle_penalty")


def _batch(B: int = 4, T: int = 32) -> types.SimpleNamespace:
    torch.manual_seed(0)
    return types.SimpleNamespace(
        fhr_st=torch.randn(B, T, 43), fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43), up_ph=torch.randn(B, T, 58),
        weight=torch.ones(B, T))


def _activate_source_path(model) -> None:
    r"""Un-zero the source-dependent heads so the source actually moves the model.

    ``SeqVaeLagAttnV3._zero_init_delta_heads`` zeroes ``posterior_head.delta_mu_head`` (which
    carries the source into the posterior) and ``residual_decoder.mean_head`` (which carries
    it into the forecast). That is the whole point of the zero-KL-at-init property (S1-T03):
    at step 0 the posterior *equals* the prior, so $K \equiv 0$ and permuting the source
    changes literally nothing. Every perm readout is therefore an exact ``0.0`` on a
    freshly-built model, and a test asserting "non-zero == the control ran" would be measuring
    initialisation, not the control. Tests needing live values re-randomise both heads; tests
    needing to know whether the control *ran* count calls instead.
    """
    torch.manual_seed(1)
    heads = [model.posterior_head.delta_mu_head, model.residual_decoder.mean_head]
    for head in heads:
        mods = list(head) if isinstance(head, torch.nn.ModuleList) else [head]
        for m in mods:
            for p in m.parameters():
                torch.nn.init.normal_(p, std=0.5)


def _wrap(cfg: dict, *, live_source: bool = False) -> SyntheticSeqVaeLagAttnV2Pl:
    model, _ = build_model(cfg, _CPU)
    if live_source:
        _activate_source_path(model)
    wrap = SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood="gaussian_nll", sigma_obs="learned",
        free_bits=0.0, detach_baseline_in_full=True)
    wrap.eval()
    return wrap


def _metrics(wrap, *, batch_idx: int, stage: str, B: int = 4):
    with torch.no_grad():
        loss, metrics = wrap.compute_loss_and_metrics(_batch(B=B), batch_idx, stage)
    return loss, metrics


def _count_perm_calls(wrap, monkeypatch) -> list:
    """Record every ``perm_kl_from_forward`` invocation, so 'ran' != 'happens to be zero'."""
    calls: list = []
    original = wrap.orig_model.perm_kl_from_forward

    def _spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(wrap.orig_model, "perm_kl_from_forward", _spy)
    return calls


def test_perm_metrics_present_on_scheduled_train_batch(monkeypatch) -> None:
    """``perm_every_n_batches=4`` -> batch 0 runs the control, batch 1 does not."""
    wrap = _wrap(_V3)
    calls = _count_perm_calls(wrap, monkeypatch)

    _, metrics = _metrics(wrap, batch_idx=0, stage="train")
    assert len(calls) == 1, "scheduled batch did not run the control"
    for key in _PERM_KEYS:
        assert key in metrics, f"perm metric {key} missing on a scheduled batch"
        assert metrics[key] is not None
        assert torch.isfinite(metrics[key]).all()

    _metrics(wrap, batch_idx=1, stage="train")
    assert len(calls) == 1, "off-schedule batch ran the control"


def test_perm_metrics_are_live_once_the_source_path_is(monkeypatch) -> None:
    """With a non-zero ``delta_mu_head`` the derangement moves both the KL and the forecast."""
    wrap = _wrap(_V3, live_source=True)
    _, metrics = _metrics(wrap, batch_idx=0, stage="train")
    assert float(metrics["kld_shuffled"]) > 0.0
    assert torch.isfinite(metrics["kld_shuffled_ratio"]).all()
    assert float(metrics["shuffle_penalty"]) != 0.0


def test_kld_shuffled_is_exactly_zero_at_init() -> None:
    r"""A freshly-built v3 has $K \equiv 0$, so the deranged KL is exactly $0$ too.

    ``delta_mu_head`` and ``delta_logvar_head`` are zero-initialised, so the posterior equals
    the prior *whatever* the source is. Permuting the source therefore cannot move the KL.
    This is the same property S1-T03 pins on real cached data, seen through the control.
    """
    wrap = _wrap(_V3)
    _, metrics = _metrics(wrap, batch_idx=0, stage="train")
    assert float(metrics["kld_shuffled"]) == 0.0


def test_shuffle_penalty_at_init_is_zero_under_mse_but_not_under_learned_sigma() -> None:
    r"""At init the forecast *mean* ignores the source; the learned *variance* does not.

    ``residual_decoder.mean_head`` is zero-initialised, so under ``mse`` the permuted forecast
    is bit-identical and $\mathcal L_{\mathrm{feat}}^{\pi(U)} - \mathcal L_{\mathrm{feat}} = 0$
    exactly. The residual decoder's log-variance head is **not** zero-initialised, so under
    ``gaussian_nll`` + ``sigma_obs='learned'`` a deranged source already changes
    $\log\sigma^2$ and hence the NLL -- at step 0 typically *improving* it (a negative
    penalty). The prediction-space gate is something the model has to **earn** during
    training; it does not hold at initialisation, and a run whose ``shuffle_penalty`` never
    turns positive is a collapsed run, not a passing one.
    """
    model, _ = build_model(_V3, _CPU)
    wrap_mse = SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood="mse", sigma_obs=1.0,
        free_bits=0.0, detach_baseline_in_full=True)
    wrap_mse.eval()
    with torch.no_grad():
        _, m_mse = wrap_mse.compute_loss_and_metrics(_batch(), 0, "train")
    assert float(m_mse["shuffle_penalty"]) == 0.0
    assert float(m_mse["feat_loss_perm"]) == float(m_mse["feat_loss"])

    _, m_nll = _metrics(_wrap(_V3), batch_idx=0, stage="train")
    assert float(m_nll["shuffle_penalty"]) != 0.0


def test_perm_metrics_zero_filled_off_schedule() -> None:
    """Off-schedule train batches keep the columns dense (zeros), never ``None``."""
    wrap = _wrap(_V3, live_source=True)
    _, metrics = _metrics(wrap, batch_idx=1, stage="train")  # 1 % 4 != 0
    for key in _PERM_KEYS:
        assert key in metrics, f"perm metric {key} missing off-schedule"
        assert metrics[key] is not None
        assert float(metrics[key]) == 0.0


def test_perm_metrics_on_every_validation_batch(monkeypatch) -> None:
    """Validation ignores the schedule -- the readout is cheap and is the headline gate."""
    wrap = _wrap(_V3)
    calls = _count_perm_calls(wrap, monkeypatch)
    for batch_idx in (0, 1, 2, 3):
        _metrics(wrap, batch_idx=batch_idx, stage="val")
    assert len(calls) == 4, "validation skipped the control on some batch"


def test_small_batch_is_skipped_without_raising(monkeypatch) -> None:
    """``B = 1`` cannot be deranged; ``perm_kl_from_forward`` would raise, so we skip."""
    wrap = _wrap(_V3, live_source=True)
    calls = _count_perm_calls(wrap, monkeypatch)
    _, metrics = _metrics(wrap, batch_idx=0, stage="val", B=1)
    assert not calls, "the control ran on a batch it cannot derange"
    for key in _PERM_KEYS:
        assert float(metrics[key]) == 0.0


def test_schedule_is_rank_invariant() -> None:
    """``_should_run_perm`` depends only on ``(batch_idx, batch_size, stage)``.

    Two ranks holding different data but the same ``batch_idx`` and a non-degenerate batch
    must agree, so the MIN reduction in ``_sync_perm_decision`` is a no-op. A rank whose
    local batch is degenerate vetoes the step for everyone -- asserted via the MIN semantics.
    """
    wrap = _wrap(_V3)
    for batch_idx in range(12):
        rank0 = wrap._should_run_perm(batch_idx, 4, "train")
        rank1 = wrap._should_run_perm(batch_idx, 8, "train")
        assert rank0 == rank1 == (batch_idx % 4 == 0)
    # A degenerate rank says no; MIN(no, yes) == no.
    assert wrap._should_run_perm(0, 1, "train") is False
    assert min(int(wrap._should_run_perm(0, 1, "train")),
               int(wrap._should_run_perm(0, 8, "train"))) == 0


def test_total_loss_untouched_by_the_control() -> None:
    """The readout must not enter the objective (the spike breaker reads the returned loss)."""
    wrap = _wrap(_V3, live_source=True)
    loss_on, metrics_on = _metrics(wrap, batch_idx=0, stage="train")   # control runs
    loss_off, _ = _metrics(wrap, batch_idx=1, stage="train")           # control skipped
    assert torch.allclose(loss_on, metrics_on["total_loss"])
    # Same batch, same weights, same seed -> identical loss whether the control ran or not.
    assert torch.allclose(loss_on, loss_off, atol=1e-6)


def test_v1_logs_no_perm_metrics() -> None:
    """A v1 model has no perm API; the block is a no-op and today's metric set survives."""
    model, _ = build_model(_TINY, _CPU)
    wrap = SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood="mse", sigma_obs=1.0,
        free_bits=0.0, detach_baseline_in_full=True)
    wrap.eval()
    with torch.no_grad():
        _, metrics = wrap.compute_loss_and_metrics(_batch(), 0, "train")
    for key in _PERM_KEYS:
        assert key not in metrics, f"v1 run unexpectedly logged {key}"


def test_perm_readout_builds_no_autograd_graph() -> None:
    """The permuted branch runs under ``no_grad``: its outputs carry no ``grad_fn``."""
    wrap = _wrap(_V3, live_source=True)
    wrap.train()
    loss, metrics = wrap.compute_loss_and_metrics(_batch(), 0, "train")
    assert loss.requires_grad, "the main loss must still be differentiable"
    for key in _PERM_KEYS:
        assert not metrics[key].requires_grad, f"{key} leaked an autograd graph"
