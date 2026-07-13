r"""S5-T04: DDP grad-coverage proof + a 2-rank gloo smoke for the synthetic_v4 arms.

`GraphModelVaeTebRawV4Trainer` resolves to plain ``'ddp'`` (``find_unused_parameters=False``) under
the v4 recipe (learned variance + ``freeze_unused_attn_proj`` + the always-used causal front ends).
What actually *licenses* that -- not the strategy string -- is a grad-coverage check: after one
``compute_loss_and_metrics`` + ``backward``, **every** ``requires_grad`` parameter has a non-``None``
grad, on both the permutation-control step and the perm-free step. The permutation control is a
``no_grad`` readout, so it must add/remove no parameter from the backward graph.

The arms that change the parameter set -- ``prod`` (causal front end), ``frontend_noncausal`` (the
leaky negative control, which swaps a time-pooling norm into a `CausalRawFrontend` subclass), and
``disable_source`` (the no-UP ablation) -- are each proved here at the tiny CPU geometry. The
2-rank gloo ``DistributedDataParallel`` smoke (marked ``slow``) catches a reducer mismatch here
rather than on the 8x A6000 headline; it is skipped where a spawn/gloo backend is unavailable.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402

pytestmark = pytest.mark.v4

_ARMS = ("prod", "frontend_noncausal", "disable_source")


def _build_arm(arm: str) -> Tuple[Any, Any]:
    r"""Build the tiny-geometry model + provenance Pl wrapper for ``arm``.

    ``prod`` / ``disable_source`` are :class:`SeqVaeRawV4`; ``frontend_noncausal`` is the leaky
    negative control :class:`LeakyRawFrontendSeqVaeRawV4` (built via ``super().__init__`` so the
    causal guard passes, then its front ends are replaced).

    Returns:
        ``(model, pl_wrap)`` -- the raw model and a :class:`SeqVaeRawV4Pl` around it.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
        LeakyRawFrontendSeqVaeRawV4,
    )
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        make_tiny_raw_model,
        tiny_raw_kwargs,
    )
    from model.vae_teb_prediction.model.model_raw.trainer_raw_v4 import SeqVaeRawV4Pl

    if arm == "frontend_noncausal":
        kwargs = tiny_raw_kwargs()
        model = LeakyRawFrontendSeqVaeRawV4(**kwargs)
    elif arm == "disable_source":
        kwargs = {**tiny_raw_kwargs(), "disable_source": True}
        model = make_tiny_raw_model(disable_source=True)
    else:
        kwargs = tiny_raw_kwargs()
        model = make_tiny_raw_model()

    pl_wrap = SeqVaeRawV4Pl(model, lr=1e-3, model_kwargs=kwargs)
    pl_wrap.train()
    return model, pl_wrap


def _tiny_batch(batch_size: int = 2, seed: int = 0):
    r"""A tiny-geometry raw batch (``fhr`` / ``up`` / decimated ``weight``)."""
    from model.vae_teb_prediction.model.model_raw.testing.conftest import make_raw_stub_batch

    return make_raw_stub_batch(batch_size=batch_size, seed=seed)


def _grad_names(model, pl_wrap, batch_idx: int) -> set:
    r"""Set of ``requires_grad`` parameter names that received a grad after one step at ``batch_idx``."""
    for p in model.parameters():
        p.grad = None
    loss, _ = pl_wrap.compute_loss_and_metrics(_tiny_batch(), batch_idx, "train")
    loss.backward()
    return {n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def _missing(model, pl_wrap, batch_idx: int) -> list:
    r"""``requires_grad`` params that received NO grad after one step."""
    got = _grad_names(model, pl_wrap, batch_idx)
    return [n for n, p in model.named_parameters() if p.requires_grad and n not in got]


@pytest.mark.parametrize("arm", _ARMS)
def test_grad_coverage_on_perm_free_step(arm: str) -> None:
    r"""Every ``requires_grad`` parameter gets a grad on a perm-free step (licenses plain ddp)."""
    model, pl_wrap = _build_arm(arm)
    # perm schedule fires on batch_idx % perm_every == 0; batch_idx=1 skips the control.
    missing = _missing(model, pl_wrap, batch_idx=1)
    assert missing == [], f"[{arm}] parameters without grad on the perm-free step: {missing}"


@pytest.mark.parametrize("arm", _ARMS)
def test_grad_coverage_identical_on_perm_step(arm: str) -> None:
    r"""The perm step's grad set equals the perm-free step's -- the no_grad control adds no param."""
    model_a, wrap_a = _build_arm(arm)
    perm_free = _grad_names(model_a, wrap_a, batch_idx=1)     # control skipped
    model_b, wrap_b = _build_arm(arm)
    perm = _grad_names(model_b, wrap_b, batch_idx=0)          # control runs

    assert perm == perm_free, (
        f"[{arm}] perm step changed the grad set: only-on-perm={sorted(perm - perm_free)}, "
        f"only-on-perm-free={sorted(perm_free - perm)}")
    assert not _missing(model_b, wrap_b, batch_idx=0), f"[{arm}] missing grads on the perm step"


# ---------------------------------------------------------------------------
# 2-rank gloo DistributedDataParallel reducer smoke (slow; skipped where unavailable).
# Must be importable at module level so ``spawn`` can pickle the worker target.
# ---------------------------------------------------------------------------
def _ddp_worker(rank: int, world_size: int, arm: str, port: str, err_queue) -> None:  # pragma: no cover
    import os

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        model, pl_wrap = _build_arm(arm)
        # find_unused_parameters=False is what strategy='ddp' implies; a starved parameter raises
        # "Expected to mark a variable ready only once" inside the reducer.
        ddp_wrap = DDP(pl_wrap, find_unused_parameters=False)
        loss, _ = ddp_wrap.module.compute_loss_and_metrics(_tiny_batch(seed=rank), 1, "train")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.5)

        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            err_queue.put(f"[{arm}] rank {rank}: params without grad: {missing[:5]}")
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        err_queue.put(f"[{arm}] rank {rank}: {type(exc).__name__}: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize("arm", _ARMS)
def test_two_rank_gloo_ddp_smoke(arm: str) -> None:
    r"""2-rank gloo DDP forward+backward with ``find_unused_parameters=False``, grads on every rank."""
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    try:
        ctx = mp.get_context("spawn")
    except (ValueError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"spawn context unavailable: {exc}")

    # A distinct port per arm avoids a bind collision when the tests run back to back.
    port = str(29520 + _ARMS.index(arm))
    err_queue = ctx.Queue()
    world = 2
    procs = [ctx.Process(target=_ddp_worker, args=(r, world, arm, port, err_queue))
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
        assert p.exitcode == 0, f"[{arm}] rank process exited {p.exitcode}"
