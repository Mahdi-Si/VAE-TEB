r"""S6-T01a/b: the neural CMI estimator's bounds and its synthetic recovery.

Two things are pinned here. First, that both lower bounds are correct in the small: on a
score matrix with a strong diagonal (positives score high) they are positive and
gradient-carrying, and on an anti-diagonal matrix they are negative. Second, that the fitted
estimator *recovers* information where it exists -- a clearly positive bound when :math:`Y`
depends on :math:`U`, and a near-zero bound when they are independent -- because a CMI
estimator that reports the same number in both cases corroborates nothing.
"""
from __future__ import annotations

import numpy as np
import torch

from model.vae_teb_prediction.testing.analyses.cmi_comparison import (
    NeuralCMIEstimator,
    estimate_cmi,
    fit_cmi_estimator,
)


def _fixed_scores(est: NeuralCMIEstimator, matrix: torch.Tensor) -> None:
    """Pin the critic's score matrix so the bound formulas can be checked in isolation."""
    est.scores = lambda u, y, c: matrix  # type: ignore[method-assign]


def test_infonce_bound_sign_and_extremes():
    est = NeuralCMIEstimator(dim_u=4, dim_y=4, dim_c=4)
    dummy = torch.zeros(4, 4)

    _fixed_scores(est, 5.0 * torch.eye(4))
    strong, _ = est.infonce_bound(dummy, dummy, dummy)
    assert strong.item() > 1.0  # near the log(N)=log4 ceiling

    _fixed_scores(est, torch.zeros(4, 4))
    flat, _ = est.infonce_bound(dummy, dummy, dummy)
    assert abs(flat.item()) < 1e-5  # no discrimination -> zero

    _fixed_scores(est, -5.0 * torch.eye(4))
    anti, _ = est.infonce_bound(dummy, dummy, dummy)
    assert anti.item() < -1.0


def test_mine_bound_sign():
    est = NeuralCMIEstimator(dim_u=4, dim_y=4, dim_c=4)
    dummy = torch.zeros(4, 4)

    _fixed_scores(est, 5.0 * torch.eye(4))
    strong, _ = est.mine_bound(dummy, dummy, dummy)
    assert strong.item() > 0.0

    _fixed_scores(est, -5.0 * torch.eye(4))
    anti, _ = est.mine_bound(dummy, dummy, dummy)
    assert anti.item() < 0.0


def test_bounds_are_finite_and_carry_gradient():
    torch.manual_seed(0)
    est = NeuralCMIEstimator(dim_u=8, dim_y=8, dim_c=8, hidden=32, embed=16)
    u, y, c = (torch.randn(16, 8) for _ in range(3))
    for bound_fn in (est.infonce_bound, est.mine_bound):
        est.zero_grad(set_to_none=True)
        value, pointwise = bound_fn(u, y, c)
        assert torch.isfinite(value)
        assert pointwise.shape == (16,)
        assert value.requires_grad
        value.backward()
        grads = [p.grad for p in est.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def _synthetic(n: int, dim: int, *, dependent: bool, seed: int):
    """Return ``(u, y, c)`` where ``y`` either depends on ``u`` or is independent of it."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, dim)).astype(np.float32)
    c = rng.standard_normal((n, dim)).astype(np.float32)
    if dependent:
        w = rng.standard_normal((dim, dim)).astype(np.float32)
        y = (u @ w + 0.1 * rng.standard_normal((n, dim))).astype(np.float32)
    else:
        y = rng.standard_normal((n, dim)).astype(np.float32)
    return u, y, c


def test_fit_loop_runs_and_records_capacity():
    """The fit primitive optimises the bound and reports the resolved capacity."""
    u, y, c = _synthetic(64, 8, dependent=True, seed=3)
    _, hist = fit_cmi_estimator(u, y, c, bound="infonce", n_iters=50, hidden=32, embed=16)
    assert np.isfinite(hist["bound"])
    assert hist["pointwise"].shape == (64,)
    assert hist["critic_hidden"] == 32 and hist["n_iters"] == 50
    # The bound should have moved off its starting value over the fit.
    assert hist["trajectory"][-1] >= hist["trajectory"][0] - 1e-3


def test_recovers_positive_cmi_on_dependent_and_zero_on_independent():
    """Cross-fitted (held-out) estimation: positive on dependence, ~0 on independence.

    Resolved capacity (recorded for the spec's Section-11 open question): critic hidden 64,
    embedding 32, depth 2, 250 iters, 2-fold cross-fit.
    """
    n, dim = 512, 8

    def _run(u, y, c, bound):
        return estimate_cmi(
            u, y, c, bound=bound, n_folds=2, n_iters=250, lr=1e-3,
            hidden=64, embed=32, weight_decay=1e-3, seed=7,
        )

    for bound in ("infonce", "mine"):
        u_d, y_d, c_d = _synthetic(n, dim, dependent=True, seed=0)
        dep = _run(u_d, y_d, c_d, bound)

        u_i, y_i, c_i = _synthetic(n, dim, dependent=False, seed=1)
        ind = _run(u_i, y_i, c_i, bound)

        assert dep["bound"] > 0.3, f"{bound}: dependent bound too small ({dep['bound']:.3f})"
        assert ind["bound"] < 0.3, f"{bound}: independent bound too large ({ind['bound']:.3f})"
        assert dep["bound"] > ind["bound"] + 0.2, (
            f"{bound}: no separation (dep={dep['bound']:.3f}, ind={ind['bound']:.3f})"
        )
        # Every sample gets an out-of-fold density for the downstream comparison.
        assert dep["pointwise"].shape == (n,)
        assert np.isfinite(dep["pointwise"]).all()
