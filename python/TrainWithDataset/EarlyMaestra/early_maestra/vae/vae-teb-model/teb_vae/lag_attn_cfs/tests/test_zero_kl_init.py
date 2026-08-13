r"""At initialisation the source says exactly nothing, and every nat later is earned.

The posterior is a zero-initialised residual on the prior under one shared $\epsilon$, and the
decoder is one module invoked twice with no dropout -- so at step $0$ the KL is exactly $0$ and the
base and full forecasts are bitwise identical, in **train** mode, which is where the objective runs.

That property is inherited rather than new, and the reason it is asserted here anyway is that this
model changes which anchors are decoded. A gather is where a batch axis and an anchor axis could be
transposed, or where the two branches could be handed different indices -- and either would break
the identity while leaving every shape correct.

**Every assertion here needs its negative control.** A model whose posterior was structurally stuck
at the prior -- a detached graph, a dead fusion -- passes all of them. ``perturb_posterior`` is the
escape, and it is applied to the same construction rather than to a differently-built model.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    TINY_HORIZON,
    make_streams,
    tiny_warmup_kwargs,
)

_TOL = 1e-6

#: The legal ends of the stride range, and the two guard states. A gather that transposed an axis
#: could easily do so only at one of them.
_STRIDES = (1, TINY_HORIZON)


def _closed_form_kl(out: dict) -> torch.Tensor:
    r"""$\mathrm{KL}(q \Vert p)$ per step per dimension, from the returned parameters alone.

    Written out rather than taken from the model, so a model whose own KL readout was wrong cannot
    certify itself.

    Args:
        out: A forward return dict.

    Returns:
        The per-step per-dimension KL.
    """
    return 0.5 * (
        out["logvar_prior"]
        - out["logvar_post"]
        + (out["logvar_post"].exp() + (out["mu_post"] - out["mu_prior"]) ** 2)
        / out["logvar_prior"].exp()
        - 1.0
    )


def _train_mode_forward(kwargs, stride: int, perturb=None):
    """A forward in **train** mode with dropout on, which is where the identities must hold."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnCfs(**dict(kwargs, dropout=0.1))
    if perturb is not None:
        perturb(model)
    model.train()
    torch.manual_seed(0)
    phase = 0 if stride == 1 else 1
    return model, model(*make_streams(kwargs), phase, stride)


def _guard_states():
    """The guarded and the unguarded keyword sets, so the identity is checked at both."""
    guarded = tiny_warmup_kwargs()
    unguarded = {
        name: value
        for name, value in guarded.items()
        if name
        not in (
            "target_keep_index",
            "target_warmup_steps",
            "source_keep_index",
            "source_warmup_steps",
        )
    }
    return (("gated", guarded), ("ungated", unguarded))


_GUARDS = _guard_states()
_GUARD_IDS = [name for name, _ in _GUARDS]
_GUARD_KWARGS = [kwargs for _, kwargs in _GUARDS]


@pytest.mark.parametrize("kwargs", _GUARD_KWARGS, ids=_GUARD_IDS)
@pytest.mark.parametrize("stride", _STRIDES)
def test_the_kl_is_exactly_zero_at_init(kwargs, stride: int) -> None:
    """Both readouts and the closed form, so a model reporting a wrong KL cannot pass on its own."""
    _model, out = _train_mode_forward(kwargs, stride)

    assert float(_closed_form_kl(out).abs().max()) == 0.0
    assert float(out["kld_per_t"].abs().max()) == 0.0
    assert float(out["source_kl_lag_map"].abs().max()) == 0.0


@pytest.mark.parametrize("kwargs", _GUARD_KWARGS, ids=_GUARD_IDS)
@pytest.mark.parametrize("stride", _STRIDES)
def test_the_two_forecasts_are_bitwise_identical_at_init(kwargs, stride: int) -> None:
    """One decoder, two invocations, one $\\epsilon$, one anchor index -- so the gather cannot have
    handed the branches different rows."""
    _model, out = _train_mode_forward(kwargs, stride)

    assert torch.equal(out["z_prior"], out["z_post"])
    assert torch.equal(out["mu_base"], out["mu_full"])
    assert torch.equal(out["logvar_base"], out["logvar_full"])


@pytest.mark.parametrize("kwargs", _GUARD_KWARGS, ids=_GUARD_IDS)
@pytest.mark.parametrize("stride", _STRIDES)
def test_the_objective_reports_the_same_zero(kwargs, stride: int) -> None:
    """``source_conditioned_kl_raw`` is the column a run is read by, and it is what must read zero.

    Asserted through ``compute_loss`` rather than off the forward, because the objective averages
    the KL over the *scored anchor support* -- so a support that had drifted from the decoded set
    would show here even while the per-step KL was fine.
    """
    model, out = _train_mode_forward(kwargs, stride)
    y_st, y_ph, u_stream = make_streams(kwargs)
    features = torch.cat([y_st, y_ph], dim=-1)
    weight = torch.ones(BATCH, model.geometry.t)

    metrics = model.compute_loss(out, features, weight=weight, likelihood="mse")["metrics"]
    assert float(metrics["source_conditioned_kl_raw"]) == 0.0
    assert float(metrics["source_conditioned_kl_train"]) == 0.0
    assert float(metrics["pred_gap"]) == 0.0


@pytest.mark.parametrize("kwargs", _GUARD_KWARGS, ids=_GUARD_IDS)
@pytest.mark.parametrize("stride", _STRIDES)
def test_everything_above_becomes_false_once_perturbed(
    kwargs, stride: int, perturb_posterior
) -> None:
    """The zero must be a property of the init, not of the model being unable to produce a KL.

    Without this, a model whose posterior was structurally stuck at the prior -- a detached graph,
    a fusion that never reads the source -- would pass every test above.
    """
    model, out = _train_mode_forward(kwargs, stride, perturb=perturb_posterior)

    assert float(_closed_form_kl(out).abs().max()) > _TOL
    assert float(out["kld_per_t"].abs().max()) > _TOL
    assert not torch.equal(out["z_prior"], out["z_post"])
    assert not torch.equal(out["mu_base"], out["mu_full"])

    y_st, y_ph, u_stream = make_streams(kwargs)
    features = torch.cat([y_st, y_ph], dim=-1)
    weight = torch.ones(BATCH, model.geometry.t)
    metrics = model.compute_loss(out, features, weight=weight, likelihood="mse")["metrics"]
    assert float(metrics["source_conditioned_kl_raw"]) > _TOL
