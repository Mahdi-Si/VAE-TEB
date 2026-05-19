"""Pytest unit tests for ``analytic_te`` -- Task 1.3 of the synthetic-TE plan.

Validates the closed-form transfer-entropy formulas against the reference
tables in ``synthetic_te_validation_plan.md`` Section 8 (asserted to 1e-3),
plus edge cases. Run from the repo root with ``python -m pytest``.
"""

import math

import numpy as np
import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    a_for_te_per_step,
    binary_entropy,
    te_block_gaussian,
    te_block_gaussian_mc,
    te_block_xor,
    te_categorical_switch,
)

TOL = 1e-3

# Section 8 reference tables -- H=30, sigma2=1, M=1, block TE in nats.
GAUSSIAN_REF = [(0.0, 0.0), (0.25, 0.909), (0.5, 3.347), (1.0, 10.397), (2.0, 24.142)]
XOR_REF = [(0.01, 19.114), (0.10, 11.042), (0.25, 3.924), (0.50, 0.0)]


@pytest.mark.parametrize("a, expected", GAUSSIAN_REF)
def test_te_block_gaussian_reference(a, expected):
    assert te_block_gaussian(a, 1.0, H=30, M=1) == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("q, expected", XOR_REF)
def test_te_block_xor_reference(q, expected):
    assert te_block_xor(q, H=30, M=1) == pytest.approx(expected, abs=TOL)


def test_te_categorical_switch_reference():
    assert te_categorical_switch(0.5, 10) == pytest.approx(1.67689, abs=TOL)


def test_gaussian_zero_transfer():
    assert te_block_gaussian(0.0, 1.0, 30, 8) == 0.0


def test_gaussian_m_scaling():
    single = te_block_gaussian(1.0, 1.0, 30, 1)
    assert te_block_gaussian(1.0, 1.0, 30, 5) == pytest.approx(5 * single, abs=TOL)


def test_gaussian_scalar_array_agree():
    scalar = te_block_gaussian(0.7, 1.3, 30, 3)
    array = te_block_gaussian([0.7, 0.7, 0.7], [1.3, 1.3, 1.3], 30, 3)
    assert scalar == pytest.approx(array, abs=1e-9)


def test_gaussian_per_channel_array():
    # Heterogeneous channels sum independently.
    expected = 0.5 * 30 * (math.log1p(0.25 / 1.0) + math.log1p(1.0 / 2.0))
    assert te_block_gaussian([0.5, 1.0], [1.0, 2.0], 30, 2) == pytest.approx(
        expected, abs=TOL
    )


def test_gaussian_invalid_args():
    with pytest.raises(ValueError):
        te_block_gaussian(1.0, 0.0, 30, 1)         # sigma2 <= 0
    with pytest.raises(ValueError):
        te_block_gaussian(1.0, 1.0, 0, 1)          # H <= 0
    with pytest.raises(ValueError):
        te_block_gaussian([1.0, 1.0], 1.0, 30, 3)  # length mismatch


@pytest.mark.parametrize("te_per_step", [0.05, 0.10, 0.15, 0.20, 0.30, 0.45])
@pytest.mark.parametrize("M", [1, 4, 8])
def test_a_for_te_per_step_roundtrip(te_per_step, M):
    # a_for_te_per_step inverts te_block_gaussian: block TE = per-step TE * H.
    a = a_for_te_per_step(te_per_step, sigma2=1.0, M=M)
    assert te_block_gaussian(a, 1.0, H=30, M=M) == pytest.approx(
        te_per_step * 30, abs=1e-9
    )


def test_a_for_te_per_step_zero_and_sigma_scaling():
    assert a_for_te_per_step(0.0, 1.0, 4) == pytest.approx(0.0, abs=1e-12)
    # a scales as sqrt(sigma2) for a fixed target TE.
    a1 = a_for_te_per_step(0.15, 1.0, 4)
    a4 = a_for_te_per_step(0.15, 4.0, 4)
    assert a4 == pytest.approx(2.0 * a1, rel=1e-9)


def test_a_for_te_per_step_invalid():
    with pytest.raises(ValueError):
        a_for_te_per_step(-0.1, 1.0, 4)   # te_per_step < 0
    with pytest.raises(ValueError):
        a_for_te_per_step(0.1, 0.0, 4)    # sigma2 <= 0
    with pytest.raises(ValueError):
        a_for_te_per_step(0.1, 1.0, 0)    # M < 1


def test_binary_entropy_endpoints():
    assert binary_entropy(0.0) == pytest.approx(0.0, abs=1e-12)
    assert binary_entropy(1.0) == pytest.approx(0.0, abs=1e-12)
    assert binary_entropy(0.5) == pytest.approx(math.log(2.0), abs=TOL)


def test_binary_entropy_array():
    out = binary_entropy(np.array([0.0, 0.5, 1.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.0, math.log(2.0), 0.0], atol=TOL)


def test_binary_entropy_invalid():
    with pytest.raises(ValueError):
        binary_entropy(1.5)


def test_xor_zero_transfer():
    assert te_block_xor(0.5, 30, 4) == pytest.approx(0.0, abs=TOL)


def test_xor_m_scaling():
    single = te_block_xor(0.1, 30, 1)
    assert te_block_xor(0.1, 30, 6) == pytest.approx(6 * single, abs=TOL)


def test_categorical_uniform_redraw():
    # p=1 -> full uniform redraw -> TE = ln(K).
    assert te_categorical_switch(1.0, 10) == pytest.approx(math.log(10.0), abs=TOL)


def test_categorical_no_switch():
    # p=0 -> regime never changes -> source reveals nothing -> TE = 0.
    assert te_categorical_switch(0.0, 5) == pytest.approx(0.0, abs=TOL)


def test_categorical_invalid():
    with pytest.raises(ValueError):
        te_categorical_switch(0.5, 1)    # K < 2
    with pytest.raises(ValueError):
        te_categorical_switch(1.5, 10)   # p out of range


# =============================================================================
# Benchmark B -- Monte-Carlo determinant-ratio TE cross-check (task 7.1)
# =============================================================================

# Smaller H / n_samples keep these tests fast while still resolving the claim.
_MC_H = 10
_MC_D = 12
_MC_N = 40_000


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("a", [0.3, 1.0])
def test_te_block_gaussian_mc_matches_closed_form(a, rho):
    # The AR self-term cancels in the determinant ratio, so the MC estimate
    # matches the closed form independently of rho.
    closed = te_block_gaussian(a, 1.0, H=_MC_H, M=1)
    mc = te_block_gaussian_mc(
        a, 1.0, rho, H=_MC_H, M=1, D=_MC_D, n_samples=_MC_N, seed=0
    )
    assert mc == pytest.approx(closed, rel=0.08, abs=0.05)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99])
def test_te_block_gaussian_mc_zero_a_is_zero(rho):
    # Headline Benchmark-B claim: no source transfer (a=0) -> TE ~ 0 for every
    # rho, however autocorrelated the target is.
    mc = te_block_gaussian_mc(
        0.0, 1.0, rho, H=_MC_H, M=1, D=_MC_D, n_samples=_MC_N, seed=0
    )
    assert abs(mc) < 0.05


def test_te_block_gaussian_mc_rho_independent():
    # The MC estimate is (within MC noise) the same across rho.
    vals = [
        te_block_gaussian_mc(
            0.6, 1.0, rho, H=_MC_H, M=1, D=_MC_D, n_samples=_MC_N, seed=0
        )
        for rho in (0.0, 0.5, 0.9)
    ]
    assert max(vals) - min(vals) < 0.05


def test_te_block_gaussian_mc_m_scaling():
    single = te_block_gaussian_mc(
        0.6, 1.0, 0.7, H=_MC_H, M=1, D=_MC_D, n_samples=_MC_N, seed=0
    )
    quad = te_block_gaussian_mc(
        0.6, 1.0, 0.7, H=_MC_H, M=4, D=_MC_D, n_samples=_MC_N, seed=0
    )
    assert quad == pytest.approx(4.0 * single, rel=1e-9)


def test_te_block_gaussian_mc_invalid():
    with pytest.raises(ValueError):
        te_block_gaussian_mc(0.5, 1.0, 0.5, H=10, M=1, D=5)    # D < H
    with pytest.raises(ValueError):
        te_block_gaussian_mc(0.5, 1.0, 1.0, H=10, M=1, D=12)   # rho >= 1
    with pytest.raises(ValueError):
        te_block_gaussian_mc(0.5, 0.0, 0.5, H=10, M=1, D=12)   # sigma2 <= 0
