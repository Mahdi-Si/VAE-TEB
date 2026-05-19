r"""Closed-form ground-truth transfer-entropy formulas (Phase 1).

This module holds the analytic block transfer-entropy (TE) expressions used as
ground truth for every synthetic benchmark. All functions are pure -- ``numpy``
and ``math`` only, no ``torch`` and no I/O -- and return values in **nats**.
They are unit-tested in ``test_analytic_te.py`` against the reference tables in
``synthetic_te_validation_plan.md`` Section 8.

Public API:
    binary_entropy: Binary Shannon entropy $h_b(q)$ in nats.
    te_block_gaussian: Delayed linear-Gaussian benchmark TE (Benchmark A / E).
    te_block_gaussian_mc: Monte-Carlo determinant-ratio block TE for the
        AR-target process (Benchmark B rho-cancellation cross-check).
    a_for_te_per_step: Inverse of the Gaussian per-step TE -> transfer coeff $a$.
    te_block_xor: Delayed binary-XOR benchmark TE (Benchmark C).
    te_categorical_switch: Categorical regime-switch benchmark TE (Benchmark D).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

ArrayLike = Union[float, int, list, tuple, np.ndarray]


def _broadcast_to_m(x: ArrayLike, m: int, name: str) -> np.ndarray:
    r"""Coerce a scalar or length-$M$ array-like into a length-$M$ array.

    Args:
        x: A scalar (broadcast to all $M$ entries) or a 1-D array-like of
            length $M$.
        m: The target length $M$.
        name: Parameter name, used only for error messages.

    Returns:
        A 1-D ``float`` ``np.ndarray`` of length ``m``.

    Raises:
        ValueError: If ``x`` is not 1-D, or its length is neither 1 nor ``m``.
    """
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D array, got ndim={arr.ndim}.")
    if arr.size == 1:
        return np.full(m, arr.item(), dtype=float)
    if arr.size != m:
        raise ValueError(f"{name} has length {arr.size}, expected 1 or M={m}.")
    return arr


def binary_entropy(q: ArrayLike) -> Union[float, np.ndarray]:
    r"""Binary Shannon entropy $h_b(q)$ in nats.

    Computes

    $$
    h_b(q) = -q \ln q - (1 - q) \ln(1 - q),
    $$

    with the convention $0 \ln 0 := 0$, so $q \in \{0, 1\}$ yields $0$.

    Args:
        q: Bernoulli probability, or array thereof, each element in $[0, 1]$.

    Returns:
        The binary entropy in nats: a Python ``float`` for scalar input,
        otherwise an ``np.ndarray`` of the same shape as ``q``.

    Raises:
        ValueError: If any element of ``q`` lies outside $[0, 1]$.
    """
    q_arr = np.asarray(q, dtype=float)
    if np.any(q_arr < 0.0) or np.any(q_arr > 1.0):
        raise ValueError("binary_entropy: q must lie in [0, 1].")
    # np.where evaluates both branches, so 0 * log(0) = nan and log(0) = -inf
    # are produced but discarded; errstate silences the corresponding warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        term_q = np.where(q_arr > 0.0, q_arr * np.log(q_arr), 0.0)
        term_1mq = np.where(q_arr < 1.0, (1.0 - q_arr) * np.log1p(-q_arr), 0.0)
    h = -(term_q + term_1mq)
    return float(h) if h.ndim == 0 else h


def te_block_gaussian(a: ArrayLike, sigma2: ArrayLike, H: int, M: int) -> float:
    r"""Block transfer entropy of the delayed linear-Gaussian process.

    For $M$ independent transferred channels with per-channel transfer
    coefficient $a_j$ and noise variance $\sigma_j^2$, the block TE over a
    forecast horizon $H$ is

    $$
    \mathrm{TE}^{(H)} = \frac{H}{2} \sum_{j=1}^{M}
        \ln\!\left(1 + \frac{a_j^2}{\sigma_j^2}\right).
    $$

    This is the Benchmark A (and per-lag Benchmark E) ground truth.

    Args:
        a: Transfer coefficient(s) -- a scalar (broadcast to all $M$ channels)
            or an array-like of length $M$.
        sigma2: Noise variance(s) $\sigma_j^2 > 0$ -- scalar or length-$M$.
        H: Forecast horizon (number of future steps), $H > 0$.
        M: Number of informative channels, $M \ge 0$.

    Returns:
        The block transfer entropy in nats.

    Raises:
        ValueError: If ``H <= 0``, ``M < 0``, any ``sigma2 <= 0``, or ``a`` /
            ``sigma2`` have a length inconsistent with ``M``.
    """
    if H <= 0:
        raise ValueError(f"te_block_gaussian: H must be > 0, got {H}.")
    if M < 0:
        raise ValueError(f"te_block_gaussian: M must be >= 0, got {M}.")
    if M == 0:
        return 0.0
    a_arr = _broadcast_to_m(a, M, "a")
    s_arr = _broadcast_to_m(sigma2, M, "sigma2")
    if np.any(s_arr <= 0.0):
        raise ValueError("te_block_gaussian: sigma2 must be strictly positive.")
    te = 0.5 * H * float(np.sum(np.log1p(a_arr ** 2 / s_arr)))
    return float(te)


def a_for_te_per_step(te_per_step: float, sigma2: float, M: int) -> float:
    r"""Transfer coefficient $a$ that yields a target per-step Gaussian TE.

    Inverts the delayed linear-Gaussian per-step transfer entropy

    $$
    \mathrm{TE}^{(1)} = \frac{M}{2}\,\ln\!\left(1 + \frac{a^2}{\sigma^2}\right)
    $$

    for the shared transfer coefficient $a$. The block TE over a horizon $H$ is
    then simply $H \cdot \mathrm{TE}^{(1)}$, so this is the natural knob for
    authoring a benchmark at a chosen per-step regime (e.g. the real-data
    range $\mathrm{TE}^{(1)} \in [0.05, 0.3]$ nats):

    $$
    a = \sqrt{\sigma^2\,\bigl(e^{2\,\mathrm{TE}^{(1)}/M} - 1\bigr)}.
    $$

    Args:
        te_per_step: Target per-step transfer entropy in nats, $\ge 0$.
        sigma2: Noise variance $\sigma^2 > 0$ (shared across channels).
        M: Number of informative channels, $M \ge 1$.

    Returns:
        The shared transfer coefficient $a \ge 0$. Round-trips with
        :func:`te_block_gaussian`:
        ``te_block_gaussian(a_for_te_per_step(t, s, M), s, H, M) == t * H``.

    Raises:
        ValueError: If ``te_per_step < 0``, ``sigma2 <= 0``, or ``M < 1``.
    """
    if te_per_step < 0.0:
        raise ValueError(
            f"a_for_te_per_step: te_per_step must be >= 0, got {te_per_step}."
        )
    if sigma2 <= 0.0:
        raise ValueError(
            f"a_for_te_per_step: sigma2 must be > 0, got {sigma2}."
        )
    if M < 1:
        raise ValueError(f"a_for_te_per_step: M must be >= 1, got {M}.")
    return float(math.sqrt(sigma2 * math.expm1(2.0 * te_per_step / M)))


def _residual_logdet(design: np.ndarray, targets: np.ndarray) -> float:
    r"""Log-determinant of the unbiased residual covariance of an OLS fit.

    Regresses every column of ``targets`` on ``design`` by ordinary least
    squares, forms the unbiased residual covariance
    $\hat\Sigma = R^\top R / (n - k)$ (with $k$ the number of regressors), and
    returns $\ln\det\hat\Sigma$.

    Args:
        design: Regressor matrix of shape $(n, k)$ (an intercept column should
            already be included by the caller).
        targets: Response matrix of shape $(n, H)$.

    Returns:
        The natural log-determinant of the unbiased $H \times H$ residual
        covariance.
    """
    n, k = design.shape
    coef, *_ = np.linalg.lstsq(design, targets, rcond=None)
    resid = targets - design @ coef
    cov = resid.T @ resid / float(n - k)
    _sign, logdet = np.linalg.slogdet(cov)
    return float(logdet)


def te_block_gaussian_mc(
    a: float,
    sigma2: float,
    rho: float,
    H: int,
    M: int,
    D: int,
    *,
    n_samples: int = 200_000,
    T: int = None,
    seed: int = 0,
) -> float:
    r"""Monte-Carlo block transfer entropy of the AR-target process (Benchmark B).

    Estimates, by simulation, the determinant-ratio block transfer entropy

    $$
    \mathrm{TE}^{(H)} = \frac12 \ln
        \frac{\det \Sigma_{Y^+ \mid Y^-}}{\det \Sigma_{Y^+ \mid Y^-, X^-}}
    $$

    of the autoregressive-target delayed-Gaussian process
    $Y_t = \rho\,Y_{t-1} + a\,X_{t-D} + \varepsilon_t$ with $X$ i.i.d.
    $\mathcal{N}(0,1)$ and $\varepsilon$ i.i.d. $\mathcal{N}(0,\sigma^2)$. The
    two conditional covariances are obtained as the residual covariances of
    ordinary-least-squares fits of the future block $Y^+$ on the history
    windows $Y^-$ (numerator) and $[Y^-, X^-]$ (denominator).

    This is a numerical *cross-check*, not a runtime path: it confirms that the
    AR self-term cancels in the determinant ratio, so the estimate matches
    :func:`te_block_gaussian` (``a``, ``sigma2``, ``H``, ``M``) **independently
    of $\rho$**. Channels are independent, so the single-channel estimate is
    scaled by $M$.

    Args:
        a: Transfer coefficient (shared across the $M$ channels).
        sigma2: Innovation variance $\sigma^2 > 0$.
        rho: AR(1) self-coefficient, $0 \le \rho < 1$.
        H: Forecast horizon, $H > 0$.
        M: Number of informative channels, $M \ge 0$.
        D: Source-to-target delay; must satisfy $D \ge H$.
        n_samples: Number of simulated sequences (Monte-Carlo sample size).
        T: Simulated sequence length. ``None`` picks a length that fits a
            burn-in, the history depth and the future block.
        seed: Seed for the NumPy random generator.

    Returns:
        The Monte-Carlo block transfer entropy estimate in nats.

    Raises:
        ValueError: If ``H <= 0``, ``M < 0``, ``D < H``, ``sigma2 <= 0``,
            ``rho`` is outside $[0, 1)$, or ``T`` is too short.
    """
    if H <= 0:
        raise ValueError(f"te_block_gaussian_mc: H must be > 0, got {H}.")
    if M < 0:
        raise ValueError(f"te_block_gaussian_mc: M must be >= 0, got {M}.")
    if D < H:
        raise ValueError(
            f"te_block_gaussian_mc: requires D >= H, got D={D}, H={H}."
        )
    if sigma2 <= 0.0:
        raise ValueError(f"te_block_gaussian_mc: sigma2 must be > 0, got {sigma2}.")
    if not 0.0 <= rho < 1.0:
        raise ValueError(
            f"te_block_gaussian_mc: rho must lie in [0, 1), got {rho}."
        )
    if M == 0:
        return 0.0

    burn_in = 100
    depth = D + H  # history-window depth for both Y^- and X^-
    if T is None:
        T = burn_in + depth + H + 1
    anchor = T - H - 1  # last step with a full future block
    if anchor - depth + 1 < 0:
        raise ValueError(
            f"te_block_gaussian_mc: T={T} too short for D={D}, H={H} "
            f"(need T >= {depth + H + 1})."
        )

    rng = np.random.default_rng(seed)
    sigma = math.sqrt(sigma2)
    n = int(n_samples)

    # Simulate one informative channel; draw innovations step-by-step so only
    # the source X and the target Y are held as full (n, T) arrays.
    X = rng.standard_normal((n, T))
    Y = np.empty((n, T), dtype=float)
    prev = np.zeros(n, dtype=float)
    for k in range(T):
        drive = a * X[:, k - D] if k >= D else 0.0
        innov = sigma * rng.standard_normal(n)
        prev = rho * prev + drive + innov
        Y[:, k] = prev

    lo = anchor - depth + 1
    y_minus = Y[:, lo : anchor + 1]            # (n, depth)
    x_minus = X[:, lo : anchor + 1]            # (n, depth)
    y_plus = Y[:, anchor + 1 : anchor + 1 + H]  # (n, H)
    ones = np.ones((n, 1), dtype=float)

    logdet_num = _residual_logdet(np.concatenate([ones, y_minus], axis=1), y_plus)
    logdet_den = _residual_logdet(
        np.concatenate([ones, y_minus, x_minus], axis=1), y_plus
    )
    te_one_channel = 0.5 * (logdet_num - logdet_den)
    return float(M * te_one_channel)


def te_block_xor(q: float, H: int, M: int) -> float:
    r"""Block transfer entropy of the delayed binary-XOR process.

    For $M$ independent binary channels copied through a delay with bit-flip
    noise probability $q$, the block TE over a forecast horizon $H$ is

    $$
    \mathrm{TE}^{(H)} = M\,H\,\bigl(\ln 2 - h_b(q)\bigr),
    $$

    where $h_b$ is the binary entropy. This is the Benchmark C ground truth.

    Args:
        q: Bit-flip probability in $[0, 1]$.
        H: Forecast horizon, $H > 0$.
        M: Number of informative binary channels, $M \ge 0$.

    Returns:
        The block transfer entropy in nats.

    Raises:
        ValueError: If ``H <= 0``, ``M < 0``, or ``q`` is outside $[0, 1]$.
    """
    if H <= 0:
        raise ValueError(f"te_block_xor: H must be > 0, got {H}.")
    if M < 0:
        raise ValueError(f"te_block_xor: M must be >= 0, got {M}.")
    per_step = math.log(2.0) - float(binary_entropy(q))
    return float(M * H * per_step)


def te_categorical_switch(p: float, K: int) -> float:
    r"""One-step transfer entropy of the categorical regime-switch process.

    A target regime in $\{1, \dots, K\}$ keeps its class with probability
    $1 - p$ and otherwise redraws uniformly over all $K$ classes. A source
    that reveals the next regime transfers, per step,

    $$
    \mathrm{TE}^{(1)} = -s \ln s - (K - 1)\,\frac{p}{K}\,\ln\frac{p}{K},
    \qquad s = 1 - p + \frac{p}{K},
    $$

    which equals $H(C_{t+1} \mid C_t)$. This is the Benchmark D ground truth;
    for $K = 10,\ p = 0.5$ it evaluates to $1.67689$ nats.

    Args:
        p: Switch probability in $[0, 1]$.
        K: Number of categorical classes, $K \ge 2$.

    Returns:
        The one-step transfer entropy in nats.

    Raises:
        ValueError: If ``p`` is outside $[0, 1]$ or ``K < 2``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"te_categorical_switch: p must be in [0, 1], got {p}.")
    if K < 2:
        raise ValueError(f"te_categorical_switch: K must be >= 2, got {K}.")
    s = 1.0 - p + p / K
    p_off = p / K
    term_stay = -s * math.log(s) if s > 0.0 else 0.0
    term_off = -(K - 1) * p_off * math.log(p_off) if p_off > 0.0 else 0.0
    return float(term_stay + term_off)


if __name__ == "__main__":
    # Self-check against synthetic_te_validation_plan.md Section 8 (nats).
    _TOL = 1e-3

    _gauss_ref = {0.0: 0.0, 0.25: 0.909, 0.5: 3.347, 1.0: 10.397, 2.0: 24.142}
    for _a, _expected in _gauss_ref.items():
        _got = te_block_gaussian(_a, 1.0, H=30, M=1)
        assert abs(_got - _expected) < _TOL, f"gaussian a={_a}: {_got} vs {_expected}"
    print(f"[gaussian] {len(_gauss_ref)} reference values OK")

    _xor_ref = {0.01: 19.114, 0.10: 11.042, 0.25: 3.924, 0.50: 0.0}
    for _q, _expected in _xor_ref.items():
        _got = te_block_xor(_q, H=30, M=1)
        assert abs(_got - _expected) < _TOL, f"xor q={_q}: {_got} vs {_expected}"
    print(f"[xor] {len(_xor_ref)} reference values OK")

    _cat = te_categorical_switch(0.5, 10)
    assert abs(_cat - 1.67689) < _TOL, f"categorical: {_cat} vs 1.67689"
    print(f"[categorical] K=10, p=0.5 -> {_cat:.5f} OK")

    # M-scaling and scalar/array agreement.
    assert abs(te_block_gaussian(1.0, 1.0, 30, 4) - 4 * 10.397) < 4 * _TOL
    assert abs(
        te_block_gaussian([1.0, 1.0], [1.0, 1.0], 30, 2)
        - te_block_gaussian(1.0, 1.0, 30, 2)
    ) < 1e-9
    print("[scaling] M-scaling and scalar/array agreement OK")

    # a_for_te_per_step round-trips with te_block_gaussian across the regime.
    for _tau in (0.05, 0.10, 0.15, 0.20, 0.30):
        for _M in (1, 4, 8):
            _a = a_for_te_per_step(_tau, 1.0, _M)
            _block = te_block_gaussian(_a, 1.0, 30, _M)
            assert abs(_block - _tau * 30) < 1e-9, (_tau, _M, _a, _block)
    print("[inverse] a_for_te_per_step round-trip OK")

    # Benchmark B cross-check: the MC determinant-ratio TE matches the closed
    # form and is rho-independent (the AR self-term cancels).
    for _a in (0.0, 0.5):
        _closed = te_block_gaussian(_a, 1.0, H=10, M=1)
        for _rho in (0.0, 0.5, 0.9, 0.99):
            _mc = te_block_gaussian_mc(
                _a, 1.0, _rho, H=10, M=1, D=12, n_samples=40_000, seed=0
            )
            _abs = abs(_mc - _closed)
            _ok = _abs < 0.05 or _abs < 0.08 * max(_closed, 1e-9)
            assert _ok, f"MC a={_a} rho={_rho}: {_mc:.4f} vs closed {_closed:.4f}"
        print(f"[mc] a={_a}: MC TE matches closed form {_closed:.4f} for all rho")

    print("All analytic-TE checks passed.")
