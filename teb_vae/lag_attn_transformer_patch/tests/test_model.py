"""Model invariants of plan B.10 on this composition: raw causality (P4-02), source purity, zero KL,
the single decoder and the lag attribution identity (P4-03), and DDP reachability (P4-05)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.frontend import refuse_time_pooling_norms
from teb_vae.lag_attn_transformer_patch.nets.patching import patchify
from teb_vae.lag_attn_transformer_rws.tests.test_ddp_reachability import (
    _forward_conditionals,
    _reads_a_tensor_value,
)

from .conftest import MOVEMENT_TOL, build, make_stub_batch, relative_change

#: The training geometry: stride 15, one phase per sample.
STRIDE = 15
PHASE = torch.tensor([0, 7])

#: Keys that must not read the source (B.10.2), including the base decode (B.10.4: no bypass).
SOURCE_FREE = ("mu_prior", "logvar_prior", "z_prior", "target_state", "mu_base", "logvar_base")


def _forward(model, fhr, up, weight):
    """Raw streams -> ``patchify`` as the task does -> the model, with a fixed epsilon draw."""
    r = model.raw_per_step
    y = patchify(fhr, weight, raw_per_step=r, validity="fhr_weight")
    u = patchify(up, weight, raw_per_step=r, validity=model.source_validity)
    torch.manual_seed(0)
    return model(y, u, PHASE, STRIDE)


def test_states_at_token_t_read_no_raw_sample_after_16t_plus_15() -> None:
    """B.10.1 in float64: resampling raw ``>= 16t + 16`` of either stream leaves every state at
    ``<= t`` bitwise unchanged; resampling from ``16t + 15`` moves the stream's own state at ``t``.
    Plus the structural half: no history-path module normalizes over time or batch (a BatchNorm
    leaks only in train mode, which an eval-mode probe cannot see)."""
    model = build().double()
    for name, child in model.named_children():
        if name not in ("horizon_core", "decoder"):
            refuse_time_pooling_norms(child, label=name)

    r, t = model.raw_per_step, 100
    gen = torch.Generator().manual_seed(0)
    raw = [torch.randn(2, 300 * r, generator=gen, dtype=torch.float64) for _ in range(2)]
    weight = torch.ones(2, 300, dtype=torch.float64)
    own = {0: ("target_state", "mu_prior"), 1: ("source_state",)}
    with torch.no_grad():
        ref = _forward(model, *raw, weight)
        for stream, keys in own.items():
            for first in (r * t + r, r * t + r - 1):
                moved = [x.clone() for x in raw]
                moved[stream][:, first:] = torch.randn(
                    moved[stream][:, first:].shape, generator=gen, dtype=torch.float64
                )
                out = _forward(model, *moved, weight)
                if first == r * t + r:
                    for key in ("target_state", "source_state", "mu_prior"):
                        assert torch.equal(ref[key][:, : t + 1], out[key][:, : t + 1]), (stream, key)
                else:
                    for key in keys:
                        change = relative_change(ref[key][:, t], out[key][:, t])
                        assert change > MOVEMENT_TOL, (stream, key, change)


def test_source_purity_and_one_decoder_invoked_twice(perturb_posterior) -> None:
    """B.10.2 and B.10.4. Perturbed first: at init ``z_post == z_prior``, so a base decode wired
    to the posterior would pass the purity check vacuously. Each equality has its control."""
    model = build()
    perturb_posterior(model)
    b = make_stub_batch(2, 300)
    gen = torch.Generator().manual_seed(1)
    calls = []
    hook = model.decoder.register_forward_pre_hook(lambda module, args: calls.append(len(args)))
    with torch.no_grad():
        ref = _forward(model, b.fhr, b.up, b.weight)
        hook.remove()
        new_up = _forward(model, b.fhr, torch.randn(b.up.shape, generator=gen), b.weight)
        new_fhr = _forward(model, torch.randn(b.fhr.shape, generator=gen), b.up, b.weight)

    assert calls == [1, 1], "the decoder must be called twice per forward, on z alone"
    for key in SOURCE_FREE:
        assert torch.equal(ref[key], new_up[key]), key
    assert not torch.equal(ref["mu_full"], new_up["mu_full"])
    assert torch.equal(ref["source_state"], new_fhr["source_state"])
    assert not torch.equal(ref["target_state"], new_fhr["target_state"])


def test_kl_is_exactly_zero_at_init_and_its_lag_map_sums_to_it(perturb_posterior) -> None:
    """B.10.3 and B.10.6. The identity is asserted after the perturbation, where both sides are
    non-zero."""
    model = build()
    b = make_stub_batch(2, 300)
    with torch.no_grad():
        kld = _forward(model, b.fhr, b.up, b.weight)["kld_per_t"]
        assert torch.equal(kld, torch.zeros_like(kld))

        perturb_posterior(model)
        out = _forward(model, b.fhr, b.up, b.weight)
    assert float(out["kld_per_t"].min()) > 0.0
    torch.testing.assert_close(out["source_kl_lag_map"].sum(-1), out["kld_per_t"])


@pytest.mark.parametrize(
    "invalid", [slice(0, 0), slice(100, 101), slice(None)], ids=["valid", "gap", "all-invalid"]
)
def test_every_trainable_parameter_gets_a_gradient(invalid) -> None:
    """B.10.5 under ``find_unused_parameters=False``, in train mode at stride 15. ``invalid``
    zeroes FHR's weight and NaNs UP over those tokens, so each adapter's ``missing`` is selected on
    some batches and not others; the loss and every gradient must also stay finite."""
    model = build().train()
    b = make_stub_batch(2, 300)
    weight = torch.ones_like(b.weight)
    weight[:, invalid] = 0.0
    up = b.up.clone()
    up.view(2, 300, -1)[:, invalid] = math.nan

    out = _forward(model, b.fhr, up, weight)
    loss = model.compute_loss(out, b.fhr, weight=weight)["metrics"]["total_loss"]
    loss.backward()

    params = dict(model.named_parameters())
    assert {"target_adapter.missing", "source_adapter.missing", "target_ar_logit"} <= set(params)
    starved = [n for n, p in params.items() if p.requires_grad and p.grad is None]
    assert not starved, starved
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None)


def test_no_forward_branches_on_a_tensor_value() -> None:
    """The sibling AST walk over this package's ``nets``: a backward on three batches cannot rule
    out a branch that fires on a fourth."""
    import ast

    nets = Path(__file__).resolve().parents[1] / "nets"
    offenders = [
        f"{path.name}:{line}: {ast.unparse(test)}"
        for path in sorted(nets.glob("*.py"))
        for _, line, test in _forward_conditionals(path.read_text(encoding="utf-8"))
        if _reads_a_tensor_value(test)
    ]
    assert not offenders, offenders
