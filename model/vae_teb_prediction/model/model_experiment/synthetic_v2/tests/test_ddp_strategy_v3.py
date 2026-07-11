r"""S2-T04 (+ S2-T07): DDP strategy resolution and the grad-coverage proof.

``_select_ddp_strategy`` resolves to plain ``'ddp'`` (``find_unused_parameters=False``) for
the v3 arms only when every parameter is guaranteed a gradient each step. The assertion that
actually *licenses* that is not the strategy string but a grad-coverage check: after one
``training_step`` + ``backward``, every ``requires_grad`` parameter has a non-``None`` grad.

The spike-breaker + 2-rank gloo DDP smoke (S2-T07) extend this module and are marked slow.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    SyntheticSeqVaeLagAttnV2Pl,
    _select_ddp_strategy,
    build_model,
)

_CPU = torch.device("cpu")
_V3_PROD = {
    "sequence_length": 32, "d_model": 16, "d_z": 8, "horizon": 4, "warmup_period": 2,
    "c_y": 87, "c_u": 101, "use_up_st": True, "max_lag": 8, "num_heads": 4, "d_head": 4,
    "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
    "class": "SeqVaeLagAttnV3",
    "v3": {"posterior_logvar": "residual", "kld_support": "anchor", "logvar_bound": "smooth",
           "causal_norm": True, "freeze_unused_attn_proj": True},
}


# ---------------------------------------------------------------------------
# Strategy matrix
# ---------------------------------------------------------------------------
def test_v3_arms_resolve_to_plain_ddp() -> None:
    """gaussian_nll + learned sigma + head-structured + frozen attn proj -> 'ddp' for n>1."""
    assert _select_ddp_strategy(
        8, "gaussian_nll", "learned",
        head_structured_latent=True, freeze_unused_attn_proj=True) == "ddp"
    assert _select_ddp_strategy(
        1, "gaussian_nll", "learned",
        head_structured_latent=True, freeze_unused_attn_proj=True) == "auto"


def test_unfrozen_attn_proj_needs_find_unused() -> None:
    """A head-structured latent with an UNfrozen W_o starves it -> find_unused."""
    assert _select_ddp_strategy(
        8, "gaussian_nll", "learned",
        head_structured_latent=True, freeze_unused_attn_proj=False
    ) == "ddp_find_unused_parameters_true"


def test_mse_and_curriculum_need_find_unused() -> None:
    assert _select_ddp_strategy(8, "mse", 1.0) == "ddp_find_unused_parameters_true"
    assert _select_ddp_strategy(
        8, "gaussian_nll", "learned", curriculum_enabled=True,
        head_structured_latent=True, freeze_unused_attn_proj=True
    ) == "ddp_find_unused_parameters_true"


# ---------------------------------------------------------------------------
# W_o freezing + grad coverage
# ---------------------------------------------------------------------------
def test_w_o_frozen_when_head_structured() -> None:
    model, _ = build_model(_V3_PROD, _CPU)
    for p in model.lag_attn.W_o.parameters():
        assert p.requires_grad is False


def _grad_names(model, wrap, batch_idx: int) -> set:
    """Return the set of parameter names that received a gradient after one step."""
    torch.manual_seed(0)
    B, T = 2, 32
    batch = types.SimpleNamespace(
        fhr_st=torch.randn(B, T, 43), fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43), up_ph=torch.randn(B, T, 58),
        weight=torch.ones(B, T))
    wrap.train()
    for p in model.parameters():
        p.grad = None
    loss, _ = wrap.compute_loss_and_metrics(batch, batch_idx, "train")
    loss.backward()
    return {n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def _grad_coverage_missing(model, wrap, batch_idx: int = 1):
    """Return the list of requires_grad params that received no grad after one step."""
    got = _grad_names(model, wrap, batch_idx)
    return [n for n, p in model.named_parameters() if p.requires_grad and n not in got]


def _v3_wrap():
    model, _ = build_model(_V3_PROD, _CPU)
    return model, SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood="gaussian_nll", sigma_obs="learned",
        free_bits=0.0, detach_baseline_in_full=True)


def test_grad_coverage_on_non_perm_step() -> None:
    """Every requires_grad parameter gets a gradient on a (non-perm) step -- licenses ddp.

    ``perm_every_n_batches`` defaults to 4, so ``batch_idx=1`` skips the permutation control.
    """
    model, wrap = _v3_wrap()
    missing = _grad_coverage_missing(model, wrap, batch_idx=1)
    assert missing == [], f"parameters without grad: {missing}"


def test_grad_coverage_identical_on_perm_step() -> None:
    """S3-T01: the perm step's parameter-grad set equals the non-perm step's.

    The permutation control is a ``no_grad`` readout, so it must add no parameter to the
    backward graph and remove none. This -- not the strategy string -- is what actually
    licenses ``find_unused_parameters=False`` across perm and non-perm steps, and closes the
    half of S2-T04 that was deferred until the control existed.
    """
    model_a, wrap_a = _v3_wrap()
    non_perm = _grad_names(model_a, wrap_a, batch_idx=1)   # 1 % 4 != 0 -> control skipped
    model_b, wrap_b = _v3_wrap()
    perm = _grad_names(model_b, wrap_b, batch_idx=0)       # 0 % 4 == 0 -> control runs

    assert perm == non_perm, (
        f"perm step changed the grad set: only-on-perm={sorted(perm - non_perm)}, "
        f"only-on-non-perm={sorted(non_perm - perm)}")
    assert not _grad_coverage_missing(model_b, wrap_b, batch_idx=0)


# ---------------------------------------------------------------------------
# S2-T07: spike breaker (with the beta schedule active) + 2-rank gloo DDP smoke
# ---------------------------------------------------------------------------
def _batch(B=2, T=32):
    torch.manual_seed(0)
    return types.SimpleNamespace(
        fhr_st=torch.randn(B, T, 43), fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43), up_ph=torch.randn(B, T, 58),
        weight=torch.ones(B, T))


def _wrap_with_schedule():
    model, _ = build_model(_V3_PROD, _CPU)
    wrap = SyntheticSeqVaeLagAttnV2Pl(
        model, kld_beta=1e-3, likelihood="gaussian_nll", sigma_obs="learned",
        free_bits=0.0, detach_baseline_in_full=True,
        beta_schedule={"kind": "linear_warmup", "start": 1e-4, "end": 1e-2,
                       "warmup_epochs": 1})
    wrap.train()
    return model, wrap


def test_spike_breaker_fires_on_nan_batch(monkeypatch) -> None:
    """A NaN batch triggers the MAX-reduced skip even with a beta schedule configured."""
    _, wrap = _wrap_with_schedule()
    monkeypatch.setattr(wrap, "_log_metrics", lambda *a, **k: None)
    batch = _batch()
    batch.fhr_st = batch.fhr_st.clone()
    batch.fhr_st[0, 0, 0] = float("nan")
    out = wrap.training_step(batch, 0)
    assert wrap._spike_skips_total == 1
    assert torch.isfinite(out).all() and float(out) == 0.0


def test_spike_breaker_fires_on_ema_spike(monkeypatch) -> None:
    """A loss far above the EMA (>5x) triggers the skip once the EMA is primed."""
    _, wrap = _wrap_with_schedule()
    monkeypatch.setattr(wrap, "_log_metrics", lambda *a, **k: None)
    wrap._spike_batches_seen = int(wrap._spike_cfg["warmup_batches"])  # past priming
    wrap._spike_ema_loss = 1e-6  # a tiny EMA makes any real loss a >5x spike
    out = wrap.training_step(_batch(), 0)
    assert wrap._spike_skips_total == 1
    assert float(out) == 0.0


def test_no_spike_while_priming(monkeypatch) -> None:
    """During EMA priming (batches < warmup) a normal batch is never flagged."""
    _, wrap = _wrap_with_schedule()
    monkeypatch.setattr(wrap, "_log_metrics", lambda *a, **k: None)
    out = wrap.training_step(_batch(), 0)
    assert wrap._spike_skips_total == 0
    assert torch.isfinite(out).all()


def test_spike_breaker_force_accepts_after_cap(monkeypatch) -> None:
    """At the consecutive-skip cap the breaker force-accepts one batch and re-seeds the EMA.

    This is the regression for the headline-run freeze: a collapsed EMA (near-zero
    learned-variance NLL loss) otherwise skips every batch forever. Sitting exactly at the
    cap, the next real batch must be accepted, the EMA hard-re-seeded off the collapsed
    value, the run-length reset, and the forced accept must NOT count as a skip.
    """
    _, wrap = _wrap_with_schedule()
    monkeypatch.setattr(wrap, "_log_metrics", lambda *a, **k: None)
    cap = int(wrap._spike_cfg["max_consecutive_skips"])
    wrap._spike_batches_seen = int(wrap._spike_cfg["warmup_batches"])  # past priming
    wrap._spike_ema_loss = 1e-6           # collapsed EMA: any real loss is a >5x spike
    wrap._spike_consecutive = cap         # exactly at the cap -> next step is force-accepted

    out = wrap.training_step(_batch(), 0)

    assert wrap._spike_skips_total == 0, "a forced accept must not count as a skip"
    assert wrap._spike_forced_accepts_total == 1
    assert wrap._spike_consecutive == 0
    assert torch.isfinite(out).all() and float(out) != 0.0
    # Hard re-seed off the collapsed 1e-6 to the real batch-loss scale.
    assert wrap._spike_ema_loss == float(out)


# -- 2-rank gloo DDP reducer smoke ------------------------------------------
# Run in spawned worker processes; must be importable at module level for pickling.
def _ddp_worker(rank: int, world_size: int, err_queue) -> None:  # pragma: no cover
    import os

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        model, _ = build_model(_V3_PROD, torch.device("cpu"))
        wrap = SyntheticSeqVaeLagAttnV2Pl(
            model, kld_beta=1e-3, likelihood="gaussian_nll", sigma_obs="learned",
            free_bits=0.0, detach_baseline_in_full=True)
        wrap.train()
        # find_unused_parameters=False is the setting `strategy='ddp'` implies; a starved
        # parameter would raise "Expected to mark a variable ready only once".
        ddp_wrap = DDP(wrap, find_unused_parameters=False)

        torch.manual_seed(rank)
        B, T = 2, 32
        batch = types.SimpleNamespace(
            fhr_st=torch.randn(B, T, 43), fhr_ph=torch.randn(B, T, 44),
            up_st=torch.randn(B, T, 43), up_ph=torch.randn(B, T, 58),
            weight=torch.ones(B, T))
        loss, _ = ddp_wrap.module.compute_loss_and_metrics(batch, 0, "train")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.5)

        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        if missing:
            err_queue.put(f"rank {rank}: params without grad: {missing[:5]}")
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        err_queue.put(f"rank {rank}: {type(exc).__name__}: {exc}")


@pytest.mark.slow
def test_two_rank_gloo_ddp_smoke() -> None:
    """2-rank gloo DDP forward+backward with find_unused_parameters=False, grads on every rank.

    Catches a reducer mismatch here rather than on 8x A6000. Uses ``DistributedDataParallel``
    directly (the component that raises "Expected to mark a variable ready only once"), so the
    check does not depend on a full Lightning ``Trainer.fit`` spawn.
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    err_queue = ctx.Queue()
    world = 2
    procs = [ctx.Process(target=_ddp_worker, args=(r, world, err_queue))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)

    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get())
    assert not errors, "\n".join(errors)
    for p in procs:
        assert p.exitcode == 0, f"rank process exited {p.exitcode}"
