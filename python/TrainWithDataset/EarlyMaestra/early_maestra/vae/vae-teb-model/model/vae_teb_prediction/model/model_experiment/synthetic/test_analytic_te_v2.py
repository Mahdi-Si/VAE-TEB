r"""Unit tests for the v2 closed-form / Monte-Carlo block transfer entropies.

Covers the additions to :mod:`analytic_te`:

    * :func:`te_block_arx_gaussian` (closed-form, G2)
    * :func:`_te_block_arx_gaussian_mc` (OLS cross-check, tests-only)
    * :func:`te_block_state_space_gaussian` (Monte-Carlo, G1)
    * :func:`te_categorical_switch_block` (G3)

Test plan follows ``model_validation_v2_plan.md`` Sprint 1.6.
"""

from __future__ import annotations

import numpy as np
import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    B_y_for_mean_te_block_state_space,
    B_y_for_te_block_state_space,
    _simulate_state_space_gaussian,
    _te_block_arx_gaussian_mc,
    c_for_mean_te_block_arx,
    c_for_te_block_arx,
    mean_te_block_arx_over_delays,
    mean_te_block_state_space_over_delays,
    te_block_arx_gaussian,
    te_block_gaussian,
    te_block_state_space_gaussian,
    te_categorical_switch,
    te_categorical_switch_block,
)


# ---------------------------------------------------------------------------
# (a) te_block_arx_gaussian — iid limit reduces to te_block_gaussian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("sigma2_eps", [0.5, 1.0, 2.0])
def test_arx_iid_limit_matches_te_block_gaussian(c: float, sigma2_eps: float) -> None:
    r"""At $\rho_u = \rho_y = 0$ and $\sigma_\eta^2 = 1$, the ARX block TE
    must equal the iid Gaussian benchmark closed form.
    """
    H, D = 10, 12
    arx = te_block_arx_gaussian(
        rho_u=0.0, rho_y=0.0, c=c,
        sigma2_eta=1.0, sigma2_eps=sigma2_eps, H=H, D=D,
    )
    ref = te_block_gaussian(a=c, sigma2=sigma2_eps, H=H, M=1)
    assert arx == pytest.approx(ref, abs=1e-9)


def test_arx_zero_coupling_is_zero() -> None:
    """``c == 0`` is the no-information control; TE must be exactly 0."""
    for rho_u in (0.0, 0.5, 0.9, 0.99):
        for rho_y in (0.0, 0.5, 0.9, 0.99):
            te = te_block_arx_gaussian(
                rho_u=rho_u, rho_y=rho_y, c=0.0,
                sigma2_eta=1.0, sigma2_eps=1.0, H=10, D=12,
            )
            assert te == 0.0


# ---------------------------------------------------------------------------
# (b) Monotone in |c|, independent of rho_y in the determinant ratio
# ---------------------------------------------------------------------------


def test_arx_monotone_in_abs_c() -> None:
    """Block TE must strictly increase with the source-coupling magnitude."""
    cs = [0.1, 0.25, 0.5, 1.0]
    last = -1.0
    for c in cs:
        te = te_block_arx_gaussian(
            rho_u=0.99, rho_y=0.95, c=c,
            sigma2_eta=1.0, sigma2_eps=1.0, H=30, D=60,
        )
        assert te > last, f"TE not monotone: c={c} gives {te} <= prev {last}"
        last = te


def test_arx_signed_c_invariance() -> None:
    """The determinant ratio depends on $c^2$, not on the sign of $c$."""
    pos = te_block_arx_gaussian(
        rho_u=0.9, rho_y=0.9, c=0.5,
        sigma2_eta=1.0, sigma2_eps=1.0, H=10, D=12,
    )
    neg = te_block_arx_gaussian(
        rho_u=0.9, rho_y=0.9, c=-0.5,
        sigma2_eta=1.0, sigma2_eps=1.0, H=10, D=12,
    )
    assert pos == pytest.approx(neg, abs=1e-9)


# ---------------------------------------------------------------------------
# (c) Closed form vs Monte-Carlo cross-check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho_u", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("rho_y", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("c", [0.25, 0.5, 1.0])
def test_arx_closed_form_matches_mc(rho_u: float, rho_y: float, c: float) -> None:
    """At small/moderate $\\rho$, closed-form and MC TE agree within 5%."""
    H, D = 10, 12
    closed = te_block_arx_gaussian(
        rho_u=rho_u, rho_y=rho_y, c=c,
        sigma2_eta=1.0, sigma2_eps=1.0, H=H, D=D,
    )
    mc = _te_block_arx_gaussian_mc(
        rho_u=rho_u, rho_y=rho_y, c=c,
        sigma2_eta=1.0, sigma2_eps=1.0, H=H, D=D,
        n_samples=80_000, seed=0,
    )
    # Absolute slack for the near-zero case; relative slack otherwise.
    abs_err = abs(closed - mc)
    rel_err = abs_err / max(abs(closed), 1e-6)
    ok = abs_err < 0.05 or rel_err < 0.05
    assert ok, (
        f"ARX TE mismatch at rho_u={rho_u}, rho_y={rho_y}, c={c}: "
        f"closed={closed:.4f}, mc={mc:.4f}, abs={abs_err:.4f}, rel={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# (d) State-space MC reduces to ARX when the oscillator collapses
# ---------------------------------------------------------------------------


def test_state_space_collapses_to_arx() -> None:
    r"""At $r = 0$, $\omega = 0$ the oscillator degenerates to iid noise
    ($s_t = \eta_t$), so the G1 process is identical to ARX with
    $\rho_u = 0$. The MC TE must match the closed-form ARX TE within 10%
    Monte-Carlo slack.
    """
    H, D = 10, 12
    ss = te_block_state_space_gaussian(
        oscillators=[(0.0, 0.0)],
        target_ar=0.5,
        delays=[D],
        B_y=[0.5],
        sigma2_y=1.0,
        sigma2_eta=1.0,
        H=H,
        n_samples=60_000,
        seed=0,
    )
    arx = te_block_arx_gaussian(
        rho_u=0.0, rho_y=0.5, c=0.5,
        sigma2_eta=1.0, sigma2_eps=1.0, H=H, D=D,
    )
    rel_err = abs(ss - arx) / max(abs(arx), 1e-6)
    assert rel_err < 0.10, (
        f"state-space MC vs ARX collapse: ss={ss:.4f}, arx={arx:.4f}, "
        f"rel_err={rel_err:.4f}"
    )


def test_state_space_oscillator_positive_te() -> None:
    """An AR(2) oscillator drive with nonzero coupling has strictly
    positive block TE.
    """
    ss = te_block_state_space_gaussian(
        oscillators=[(0.99, 0.05)],
        target_ar=0.95,
        delays=[60],
        B_y=[0.5],
        sigma2_y=1.0,
        sigma2_eta=0.01,
        H=30,
        n_samples=30_000,
        seed=0,
    )
    assert ss > 0.5, f"oscillator TE collapsed: {ss}"


# ---------------------------------------------------------------------------
# (d2) B_y_for_te_block_state_space — bisection round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_per_step", [0.05, 0.15, 0.30])
def test_b_y_inverter_roundtrips_state_space(target_per_step: float) -> None:
    r"""``B_y_for_te_block_state_space`` should land within 5% relative of
    the requested block TE on the default G1 operating regime.

    Tolerance: the bisection loop stops at ``tol=0.01`` on either the
    bracket width or the achieved-TE relative error, but the MC noise in
    ``te_block_state_space_gaussian`` adds ~1% std. error at the default
    ``n_samples``. 5% is a safe ceiling that still catches a broken
    bisection.
    """
    H = 30
    target = target_per_step * H
    # Use the same shared seed for the bracket evaluations and the bisection
    # so monotonicity in B_y holds against MC noise.
    result = B_y_for_te_block_state_space(
        target_te_block=target,
        oscillators=[(0.99, 0.05)],
        target_ar=0.95,
        delays=[60],
        sigma2_y=1.0,
        sigma2_eta=0.01,
        H=H,
        n_samples=10_000,        # keep CI runtime bounded
        seed=0,
        tol=0.02,                # bracket OR TE relative tol
        max_iter=24,
    )
    assert result["te_block"] == pytest.approx(target, rel=0.05), (
        f"bisection missed target: target={target:.3f}, "
        f"got te_block={result['te_block']:.3f}, "
        f"B_y_scalar={result['B_y_scalar']:.4f}, n_iter={result['n_iter']}"
    )
    # Per-step bookkeeping matches the contract.
    assert result["te_per_step"] == pytest.approx(
        result["te_block"] / H, rel=1e-9, abs=1e-9
    )
    # The returned B_y list has length M and is uniform.
    assert len(result["B_y"]) == 1
    assert result["B_y"][0] == pytest.approx(result["B_y_scalar"], rel=1e-9)


def test_b_y_inverter_zero_target_short_circuits() -> None:
    """``target_te_block=0`` must return B_y=0 without running the bisection."""
    result = B_y_for_te_block_state_space(
        target_te_block=0.0,
        oscillators=[(0.99, 0.05)],
        target_ar=0.95,
        delays=[60],
        sigma2_y=1.0,
        sigma2_eta=0.01,
        H=30,
        n_samples=2_000,
        seed=0,
    )
    assert result["B_y_scalar"] == 0.0
    assert result["B_y"] == [0.0]
    assert result["te_block"] == 0.0
    assert result["te_per_step"] == 0.0
    assert result["n_iter"] == 0


def test_b_y_inverter_rejects_bad_bracket() -> None:
    """Bracket must contain the target; otherwise a helpful ValueError."""
    with pytest.raises(ValueError, match="bracket"):
        B_y_for_te_block_state_space(
            target_te_block=100.0,        # unreachable from the default bracket
            oscillators=[(0.99, 0.05)],
            target_ar=0.95,
            delays=[60],
            sigma2_y=1.0,
            sigma2_eta=0.01,
            H=30,
            n_samples=2_000,
            lo=1e-4, hi=0.1,              # too narrow on the high side
            seed=0,
            max_iter=4,
        )


def test_b_y_inverter_rejects_bad_args() -> None:
    """Sanity guards on H, target, lo/hi, max_iter."""
    base_kwargs = dict(
        oscillators=[(0.99, 0.05)],
        target_ar=0.95,
        delays=[60],
        sigma2_y=1.0,
        sigma2_eta=0.01,
        H=30,
        n_samples=1_000,
        seed=0,
    )
    with pytest.raises(ValueError, match="target_te_block"):
        B_y_for_te_block_state_space(target_te_block=-1.0, **base_kwargs)
    with pytest.raises(ValueError, match="H"):
        B_y_for_te_block_state_space(
            target_te_block=1.0,
            **{**base_kwargs, "H": 0},
        )
    with pytest.raises(ValueError, match="lo"):
        B_y_for_te_block_state_space(
            target_te_block=1.0, lo=0.0, **base_kwargs,
        )
    with pytest.raises(ValueError, match="hi"):
        B_y_for_te_block_state_space(
            target_te_block=1.0, lo=1.0, hi=0.5, **base_kwargs,
        )
    with pytest.raises(ValueError, match="max_iter"):
        B_y_for_te_block_state_space(
            target_te_block=1.0, max_iter=0, **base_kwargs,
        )


# ---------------------------------------------------------------------------
# (e) te_categorical_switch_block = H * te_categorical_switch
# ---------------------------------------------------------------------------


def test_categorical_block_equals_H_times_per_step() -> None:
    r"""$\mathrm{TE}^{(H)}_{\text{cat}} = H \cdot \mathrm{TE}^{(1)}_{\text{cat}}$."""
    for p in (0.05, 0.25, 0.5, 0.75):
        for K in (2, 5, 10, 20):
            for H in (1, 10, 30):
                block = te_categorical_switch_block(p, K, H)
                per_step = te_categorical_switch(p, K)
                assert block == pytest.approx(H * per_step, abs=1e-9)


def test_categorical_block_rotating_mnist_value() -> None:
    """The TEB-paper reference value: $H=30, K=10, p=0.5 \\to 50.31$ nats."""
    got = te_categorical_switch_block(0.5, 10, 30)
    assert got == pytest.approx(30 * 1.67689, abs=1e-3)


def test_categorical_block_rejects_invalid_H() -> None:
    """``H <= 0`` must raise ``ValueError``."""
    with pytest.raises(ValueError):
        te_categorical_switch_block(0.5, 10, 0)
    with pytest.raises(ValueError):
        te_categorical_switch_block(0.5, 10, -3)


# ---------------------------------------------------------------------------
# Misc: input validation
# ---------------------------------------------------------------------------


def test_arx_rejects_invalid_rho() -> None:
    """``rho_u`` and ``rho_y`` must lie in $[0, 1)$."""
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            te_block_arx_gaussian(
                rho_u=bad, rho_y=0.5, c=0.5,
                sigma2_eta=1.0, sigma2_eps=1.0, H=10, D=12,
            )
        with pytest.raises(ValueError):
            te_block_arx_gaussian(
                rho_u=0.5, rho_y=bad, c=0.5,
                sigma2_eta=1.0, sigma2_eps=1.0, H=10, D=12,
            )


def test_arx_rejects_invalid_sigma() -> None:
    """Both innovation variances must be strictly positive."""
    with pytest.raises(ValueError):
        te_block_arx_gaussian(
            rho_u=0.5, rho_y=0.5, c=0.5,
            sigma2_eta=0.0, sigma2_eps=1.0, H=10, D=12,
        )
    with pytest.raises(ValueError):
        te_block_arx_gaussian(
            rho_u=0.5, rho_y=0.5, c=0.5,
            sigma2_eta=1.0, sigma2_eps=-1.0, H=10, D=12,
        )


# ---------------------------------------------------------------------------
# (f) c_for_te_block_arx -- bisection round-trip on the G2 process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_per_step", [0.05, 0.15, 0.30])
def test_c_inverter_roundtrips_arx(target_per_step: float) -> None:
    r"""``c_for_te_block_arx`` should land within 0.5% of the requested
    per-channel block TE on the default G2 operating regime.

    The inverter calls the closed-form :func:`te_block_arx_gaussian` (no
    Monte-Carlo noise), so the only error source is the bisection
    tolerance itself.
    """
    H, D = 30, 60
    target = target_per_step * H
    # tol=5e-3 halves the bisection iterations vs 1e-3 while still keeping
    # the achieved TE within the assertion's 0.5% relative tolerance --
    # the `te_rel` early-exit triggers on TE-relative error, not bracket
    # width.
    result = c_for_te_block_arx(
        target_te_block=target,
        rho_u=0.99, rho_y=0.95,
        sigma2_eta=1.0, sigma2_eps=1.0,
        H=H, D=D, tol=5e-3,
    )
    assert result["te_block"] == pytest.approx(target, rel=0.005), (
        f"bisection missed target: target={target:.3f}, "
        f"got te_block={result['te_block']:.3f}, "
        f"c={result['c_scalar']:.4f}, n_iter={result['n_iter']}"
    )
    assert result["te_per_step"] == pytest.approx(
        result["te_block"] / H, rel=1e-9, abs=1e-9
    )
    # The recovered c should be strictly positive (we requested non-zero TE).
    assert result["c_scalar"] > 0.0


def test_c_inverter_zero_target_short_circuits() -> None:
    """``target_te_block=0`` returns c=0 without running the bisection."""
    result = c_for_te_block_arx(
        target_te_block=0.0,
        rho_u=0.99, rho_y=0.95,
        sigma2_eta=1.0, sigma2_eps=1.0, H=30, D=60,
    )
    assert result["c_scalar"] == 0.0
    assert result["te_block"] == 0.0
    assert result["te_per_step"] == 0.0
    assert result["n_iter"] == 0


def test_c_inverter_rejects_bad_bracket() -> None:
    """Bracket must contain the target; otherwise a helpful ValueError."""
    with pytest.raises(ValueError, match="bracket"):
        c_for_te_block_arx(
            target_te_block=200.0,             # unreachable from a tight bracket
            rho_u=0.99, rho_y=0.95,
            sigma2_eta=1.0, sigma2_eps=1.0,
            H=30, D=60,
            lo=1e-4, hi=0.01,                  # too narrow on the high side
            max_iter=4,
        )


def test_c_inverter_rejects_bad_args() -> None:
    """Sanity guards on H, D, target, lo/hi, max_iter."""
    base_kwargs = dict(
        rho_u=0.99, rho_y=0.95,
        sigma2_eta=1.0, sigma2_eps=1.0,
        H=30, D=60,
    )
    with pytest.raises(ValueError, match="target_te_block"):
        c_for_te_block_arx(target_te_block=-1.0, **base_kwargs)
    with pytest.raises(ValueError, match="H"):
        c_for_te_block_arx(
            target_te_block=1.0,
            **{**base_kwargs, "H": 0},
        )
    with pytest.raises(ValueError, match="D"):
        c_for_te_block_arx(
            target_te_block=1.0,
            **{**base_kwargs, "D": 0},
        )
    with pytest.raises(ValueError, match="lo"):
        c_for_te_block_arx(
            target_te_block=1.0, lo=0.0, **base_kwargs,
        )
    with pytest.raises(ValueError, match="hi"):
        c_for_te_block_arx(
            target_te_block=1.0, lo=1.0, hi=0.5, **base_kwargs,
        )
    with pytest.raises(ValueError, match="max_iter"):
        c_for_te_block_arx(
            target_te_block=1.0, max_iter=0, **base_kwargs,
        )


# ---------------------------------------------------------------------------
# (g) variable per-sample delays + mean-over-delays TE (real-data small-lag)
# ---------------------------------------------------------------------------


def test_simulate_per_sample_delays_match_scalar_path() -> None:
    """An (n, M) delay array of a constant ``d`` reproduces the scalar path
    bit-for-bit, so the variable-delay gather introduces no drift in TE."""
    osc = [(0.99, 0.05), (0.98, 0.07)]
    common = dict(
        oscillators=osc, target_ar=0.95, B_y=[0.3, 0.5],
        sigma2_y=1.0, sigma2_eta=0.01, burn_in=80, seed=11,
    )
    S1, Y1 = _simulate_state_space_gaussian(n=12, T=60, delays=[7, 7], **common)
    d2 = np.full((12, 2), 7, dtype=int)
    S2, Y2 = _simulate_state_space_gaussian(n=12, T=60, delays=d2, **common)
    assert np.array_equal(S1, S2)
    assert np.array_equal(Y1, Y2)


def test_simulate_per_time_constant_walk_matches_scalar_path() -> None:
    """An (n, T_total, M) delay array constant in time/sample reproduces the
    scalar path bit-for-bit, so the random-walk per-time gather adds no TE drift
    when the lag does not move."""
    osc = [(0.99, 0.05), (0.98, 0.07)]
    common = dict(
        oscillators=osc, target_ar=0.95, B_y=[0.3, 0.5],
        sigma2_y=1.0, sigma2_eta=0.01, burn_in=80, seed=11,
    )
    n, T, burn = 12, 60, 80
    S1, Y1 = _simulate_state_space_gaussian(n=n, T=T, delays=[7, 7], **common)
    d3 = np.full((n, burn + T, 2), 7, dtype=int)        # (n, T_total, M) constant
    S3, Y3 = _simulate_state_space_gaussian(n=n, T=T, delays=d3, **common)
    assert np.array_equal(S1, S3)
    assert np.array_equal(Y1, Y3)


def test_simulate_per_time_wrong_T_total_raises() -> None:
    """A 3-D delay array whose time span != burn_in + T is rejected."""
    osc = [(0.99, 0.05)]
    with pytest.raises(ValueError, match="burn_in"):
        _simulate_state_space_gaussian(
            n=4, T=30, oscillators=osc, target_ar=0.9, B_y=[0.3],
            sigma2_y=1.0, sigma2_eta=0.01, burn_in=50,
            delays=np.full((4, 30, 1), 5, dtype=int),  # should be 50+30
        )


def test_mean_te_single_delay_equals_block_te() -> None:
    """A degenerate range (delay_min == delay_max) reduces to the single-delay
    block TE."""
    osc = [(0.99, 0.05)]
    single = te_block_state_space_gaussian(
        oscillators=osc, target_ar=0.95, delays=[8], B_y=[0.5],
        sigma2_y=1.0, sigma2_eta=0.01, H=30, K_history=120,
        n_samples=4000, seed=3 + 8,
    )
    mean = mean_te_block_state_space_over_delays(
        delay_min=8, delay_max=8, oscillators=osc, target_ar=0.95,
        B_y=0.5, sigma2_y=1.0, sigma2_eta=0.01, H=30, K_history=120,
        n_samples=4000, seed=3,
    )
    assert mean == pytest.approx(single, rel=1e-9)


def test_mean_te_arx_monotone_in_coupling() -> None:
    """The mean-over-delays ARX TE is monotone increasing in ``c``."""
    kw = dict(
        delay_min=1, delay_max=15, rho_u=0.99, rho_y=0.95,
        sigma2_eta=1.0, sigma2_eps=1.0, H=30, M=4,
    )
    vals = [mean_te_block_arx_over_delays(c=c, **kw) for c in (0.0, 0.02, 0.05, 0.1)]
    assert vals[0] == pytest.approx(0.0, abs=1e-9)
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("target", [0.1, 1.0, 3.0])
def test_c_for_mean_te_block_arx_roundtrip(target: float) -> None:
    """The closed-form ARX mean inverter lands on a real-data-band target."""
    sol = c_for_mean_te_block_arx(
        target_te_block=target, delay_min=1, delay_max=15,
        rho_u=0.99, rho_y=0.95, sigma2_eta=1.0, sigma2_eps=1.0,
        H=30, M=4,
    )
    assert sol["te_block"] == pytest.approx(target, rel=5e-3)
    assert sol["c_scalar"] > 0.0


def test_B_y_for_mean_te_block_state_space_roundtrip() -> None:
    """The MC state-space mean inverter lands on a real-data-band target."""
    sol = B_y_for_mean_te_block_state_space(
        target_te_block=1.0, delay_min=1, delay_max=15,
        oscillators=[(0.99, 0.05)] * 4, target_ar=0.95,
        sigma2_y=1.0, sigma2_eta=0.01, H=30, K_history=160,
        n_samples=4000, lo=1e-4, hi=10.0, tol=1e-2,
    )
    assert sol["te_block"] == pytest.approx(1.0, rel=5e-2)
    assert sol["B_y_scalar"] > 0.0


def test_mean_te_zero_coupling_is_zero() -> None:
    """B_y = 0 ⇒ mean block TE is ~0 (null control)."""
    te = mean_te_block_state_space_over_delays(
        delay_min=1, delay_max=10, oscillators=[(0.99, 0.05)], target_ar=0.95,
        B_y=0.0, sigma2_y=1.0, sigma2_eta=0.01, H=30, K_history=120,
        n_samples=3000,
    )
    assert te == pytest.approx(0.0, abs=2e-2)
