r"""Tests for the ported single-pathway analytic TE math (S0-T03).

Covers the inverter round-trip on the v2 operating regime, the SNR law
(monotone + round-trip identity + edge cases), the zero-coupling block-TE floor,
the null-cell short-circuit, and seed stability across bisection.

MC size is deliberately reduced (``_N_SAMPLES``) so the suite stays fast; a 5 %
relative tolerance still catches a broken bisection or a mis-ported determinant
ratio. The ``run_pipeline_v2.py --solve-te`` demo uses the production
``mix.inverter`` knobs from the config instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.analytic_te import (
    B_y_for_mean_te_block_state_space,
    snr_per_step_for_te_block,
    te_block_state_space_gaussian,
)

# v2 single-pathway operating regime (config benchmarks.G1_raw.data).
_OSC = [(0.80, 0.10)]
_TARGET_AR = 0.40
_SIGMA2_Y = 1.0
_SIGMA2_ETA = 0.01
_H = 30
_K = 80
_D = 8
_N_SAMPLES = 4000   # reduced MC for CI runtime; 5 % tol still catches breakage


@pytest.mark.parametrize("target_te", [0.5, 1.0, 2.0, 3.0])
def test_inverter_roundtrip(target_te: float) -> None:
    r"""The inverter lands within 5 % of the requested block TE at ``D = 8``.

    Also checks the returned-dict contract: ``B_y`` is a single-element list
    equal to ``B_y_scalar`` (single pathway), and ``te_per_step == te_block / H``.
    """
    sol = B_y_for_mean_te_block_state_space(
        target_te_block=target_te,
        delay_min=_D,
        delay_max=_D,
        oscillators=_OSC,
        target_ar=_TARGET_AR,
        sigma2_y=_SIGMA2_Y,
        sigma2_eta=_SIGMA2_ETA,
        H=_H,
        K_history=_K,
        n_samples=_N_SAMPLES,
        lo=1e-4,
        hi=60.0,
        tol=1e-2,
        max_iter=40,
        seed=0,
    )
    assert sol["te_block"] == pytest.approx(target_te, rel=5e-2), (
        f"target={target_te}, got te_block={sol['te_block']:.4f}, "
        f"B_y_scalar={sol['B_y_scalar']:.5f}, n_iter={sol['n_iter']}"
    )
    assert sol["B_y_scalar"] > 0.0
    assert set(sol) >= {"B_y", "B_y_scalar", "te_block", "te_per_step", "n_iter"}
    assert len(sol["B_y"]) == 1
    assert sol["B_y"][0] == pytest.approx(sol["B_y_scalar"], rel=1e-9)
    assert sol["te_per_step"] == pytest.approx(sol["te_block"] / _H, rel=1e-9, abs=1e-9)


def test_null_cell_short_circuits() -> None:
    r"""``target_te == 0`` returns $B = 0$ immediately (a null cell)."""
    sol = B_y_for_mean_te_block_state_space(
        target_te_block=0.0,
        delay_min=_D,
        delay_max=_D,
        oscillators=_OSC,
        target_ar=_TARGET_AR,
        sigma2_y=_SIGMA2_Y,
        sigma2_eta=_SIGMA2_ETA,
        H=_H,
        K_history=_K,
        n_samples=_N_SAMPLES,
    )
    assert sol["B_y_scalar"] == 0.0
    assert sol["te_block"] == 0.0
    assert sol["n_iter"] == 0


def test_snr_monotone_in_te() -> None:
    r"""The SNR law is strictly increasing in TE at fixed $H$."""
    snrs = [snr_per_step_for_te_block(t, _H, 1) for t in [0.5, 1.0, 2.0, 3.0]]
    assert all(later > earlier for earlier, later in zip(snrs, snrs[1:]))


def test_snr_roundtrip_and_edges() -> None:
    r"""The SNR law round-trips and honours its edge cases.

    ``snr_per_step_for_te_block(2.0, 30, 1) ~= 0.143`` (the S0 demo value),
    inverts ``0.5 * H * M * log1p(snr) == te``, is ``0`` at ``te = 0``, and
    rejects ``H <= 0``.
    """
    snr = snr_per_step_for_te_block(2.0, _H, 1)
    assert snr == pytest.approx(0.1427, abs=1e-3)
    assert 0.5 * _H * 1 * np.log1p(snr) == pytest.approx(2.0, rel=1e-9)
    assert snr_per_step_for_te_block(0.0, _H, 1) == 0.0
    with pytest.raises(ValueError):
        snr_per_step_for_te_block(1.0, 0, 1)


def test_te_block_zero_coupling_is_floor() -> None:
    r"""With $B = 0$ the block TE collapses to the MC floor (~0)."""
    te = te_block_state_space_gaussian(
        oscillators=_OSC,
        target_ar=_TARGET_AR,
        delays=[_D],
        B_y=[0.0],
        sigma2_y=_SIGMA2_Y,
        sigma2_eta=_SIGMA2_ETA,
        H=_H,
        K_history=_K,
        n_samples=_N_SAMPLES,
        seed=0,
    )
    assert abs(te) < 0.1


def _solve_default(seed: int = 0) -> dict:
    r"""Solve the ``target_te = 2.0``, ``D = 8`` cell at the reduced MC size."""
    return B_y_for_mean_te_block_state_space(
        target_te_block=2.0,
        delay_min=_D,
        delay_max=_D,
        oscillators=_OSC,
        target_ar=_TARGET_AR,
        sigma2_y=_SIGMA2_Y,
        sigma2_eta=_SIGMA2_ETA,
        H=_H,
        K_history=_K,
        n_samples=_N_SAMPLES,
        lo=1e-4,
        hi=60.0,
        tol=1e-2,
        max_iter=40,
        seed=seed,
    )


def test_inverter_seed_stable() -> None:
    r"""Two identical inverter calls (same seed) return identical results.

    The Monte-Carlo seed is held fixed across bisection, so the solve is
    deterministic and does not chase noise.
    """
    first = _solve_default(seed=0)
    second = _solve_default(seed=0)
    assert first["B_y_scalar"] == second["B_y_scalar"]
    assert first["te_block"] == second["te_block"]
    assert first["n_iter"] == second["n_iter"]
