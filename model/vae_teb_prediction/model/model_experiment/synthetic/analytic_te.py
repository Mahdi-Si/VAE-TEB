r"""Closed-form ground-truth transfer-entropy formulas (v2 — low-frequency).

This module holds analytic block transfer-entropy (TE) expressions used as
ground truth for every synthetic benchmark. All functions are pure
(``numpy`` / ``scipy.linalg`` / ``math`` only; no ``torch``, no I/O) and
return values in **nats**. They are unit-tested in ``test_analytic_te_v2.py``.

Public API (v2):
    binary_entropy: Binary Shannon entropy $h_b(q)$ in nats.
    te_block_gaussian: Per-step-additive delayed linear-Gaussian block TE.
    a_for_te_per_step: Inverse of the Gaussian per-step TE -> transfer coeff $a$.
    te_block_xor: Delayed binary-XOR benchmark TE.
    te_categorical_switch: One-step categorical regime-switch TE.
    te_block_gaussian_mc: Monte-Carlo determinant-ratio block TE for the
        delayed linear-Gaussian AR-target process (cross-check helper).

v2 additions:
    te_block_arx_gaussian: Closed-form block TE for the smooth AR(1)-ARX
        process (G2). Uses an augmented state-space, the discrete Lyapunov
        stationary covariance, and Schur-complement log-det ratios.
    te_block_state_space_gaussian: Monte-Carlo block TE for the AR(2)-driven
        oscillator state-space (G1).
    te_categorical_switch_block: Block TE for the G3 inclusive-redraw regime
        switch (= H * te_categorical_switch, per channel).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

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


# ---------------------------------------------------------------------------
# v2 additions: ARX, state-space oscillator, regime-switch block TE.
# ---------------------------------------------------------------------------


def _arx_state_space_matrices(
    rho_u: float,
    rho_y: float,
    c: float,
    sigma2_eta: float,
    sigma2_eps: float,
    D: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Augmented state-space $(A, Q)$ for the AR(1)-ARX process.

    The augmented state is

    $$
    x_t = \bigl[Y_t,\; U_t,\; U_{t-1},\; \ldots,\; U_{t-D}\bigr]^{\top}
    \in \mathbb{R}^{D+2}.
    $$

    Under $U_t = \rho_u U_{t-1} + \eta_t$ and
    $Y_t = \rho_y Y_{t-1} + c\,U_{t-D} + \varepsilon_t$, the state evolves as
    $x_{t+1} = A x_t + w_{t+1}$ where $w_t$ has covariance $Q$ with only
    the $Y$ and $U$ entries nonzero ($\sigma_\varepsilon^2$ and
    $\sigma_\eta^2$ respectively).

    Args:
        rho_u: AR(1) coefficient of $U$, $0 \le \rho_u < 1$.
        rho_y: AR(1) coefficient of $Y$, $0 \le \rho_y < 1$.
        c: Source-to-target transfer coefficient.
        sigma2_eta: Innovation variance of $U$, $> 0$.
        sigma2_eps: Innovation variance of $Y$, $> 0$.
        D: Source-to-target delay, $D \ge 1$.

    Returns:
        A tuple ``(A, Q)`` of ``(D+2, D+2)`` ``np.ndarray`` matrices.
    """
    n = D + 2
    A = np.zeros((n, n))
    A[0, 0] = rho_y
    # Y_{t+1} = rho_y * Y_t + c * U_{(t+1)-D} + eps; with x_t^{(i)} = U_{t-i+1}
    # for i >= 1, the term U_{(t+1)-D} = U_{t-D+1} = x_t^{(D)}.
    A[0, D] = c
    A[1, 1] = rho_u
    # U-lag shift: x_{t+1}^{(i)} = x_t^{(i-1)} for i = 2, ..., D+1.
    for i in range(2, D + 2):
        A[i, i - 1] = 1.0
    Q = np.zeros((n, n))
    Q[0, 0] = sigma2_eps
    Q[1, 1] = sigma2_eta
    return A, Q


def te_block_arx_gaussian(
    rho_u: float,
    rho_y: float,
    c: float,
    sigma2_eta: float,
    sigma2_eps: float,
    H: int,
    D: int,
    *,
    K_history: Optional[int] = None,
) -> float:
    r"""Closed-form block transfer entropy of the AR(1)-driven ARX process.

    Computes the analytic block TE of the smooth ARX process

    $$
    U_t = \rho_u\,U_{t-1} + \eta_t,
    \qquad
    Y_t = \rho_y\,Y_{t-1} + c\,U_{t-D} + \varepsilon_t,
    $$

    with $\eta \sim \mathcal{N}(0, \sigma_\eta^2)$ and
    $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^2)$, as the
    log-determinant ratio

    $$
    \mathrm{TE}^{(H)}_{U \to Y}
        = \tfrac{1}{2}\,
        \ln\!\frac{\det \Sigma_{Y^{+} \mid Y^{-}}}{
        \det \Sigma_{Y^{+} \mid Y^{-},\,U^{-}}},
    $$

    where $Y^{-} = Y_{t_{0}-K+1:t_{0}}$, $U^{-} = U_{t_{0}-K+1:t_{0}}$ and
    $Y^{+} = Y_{t_{0}+1:t_{0}+H}$. The two conditional covariances are
    obtained by Schur complement of the stationary joint covariance of
    $(Y^{-}, U^{-}, Y^{+})$, which is built from the discrete-Lyapunov
    stationary covariance of the augmented state
    $x_t = [Y_t, U_t, U_{t-1}, \ldots, U_{t-D}]^{\top}$ propagated through
    $\operatorname{Cov}(x_{t+\tau}, x_t) = A^{\tau} P$.

    For Gaussian processes the determinant ratio **is** the true block
    transfer entropy (Barnett–Barrett–Seth, 2009).

    Args:
        rho_u: AR(1) coefficient of $U$, $0 \le \rho_u < 1$.
        rho_y: AR(1) coefficient of $Y$, $0 \le \rho_y < 1$.
        c: Source-to-target transfer coefficient.
        sigma2_eta: Innovation variance of $U$, $> 0$.
        sigma2_eps: Innovation variance of $Y$, $> 0$.
        H: Forecast horizon, $H \ge 1$.
        D: Source-to-target delay, $D \ge 1$.
        K_history: History depth used for both $Y^{-}$ and $U^{-}$. Defaults
            to $D + 2H$, which is empirically sufficient for the conditional
            covariance to have converged to its block-marginal limit.

    Returns:
        The block transfer entropy in nats.

    Raises:
        ValueError: On invalid ranges of $\rho_{*}$, $\sigma^2_{*}$, $H$, or
            $D$.
    """
    if H <= 0:
        raise ValueError(f"te_block_arx_gaussian: H must be > 0, got {H}.")
    if D < 1:
        raise ValueError(f"te_block_arx_gaussian: D must be >= 1, got {D}.")
    if not 0.0 <= rho_u < 1.0:
        raise ValueError(
            f"te_block_arx_gaussian: rho_u must be in [0, 1), got {rho_u}."
        )
    if not 0.0 <= rho_y < 1.0:
        raise ValueError(
            f"te_block_arx_gaussian: rho_y must be in [0, 1), got {rho_y}."
        )
    if sigma2_eta <= 0.0 or sigma2_eps <= 0.0:
        raise ValueError(
            "te_block_arx_gaussian: sigma2_eta and sigma2_eps must be > 0."
        )
    if c == 0.0:
        return 0.0

    K = int(K_history) if K_history is not None else D + 2 * H

    A, Q = _arx_state_space_matrices(rho_u, rho_y, c, sigma2_eta, sigma2_eps, D)
    P = solve_discrete_lyapunov(A, Q)

    # Precompute A^tau @ P for tau = 0, ..., max_lag where
    # max_lag = (K + H - 1) covers every |s - t| arising in the joint cov.
    max_lag = K + H - 1
    APs = [P]
    for _ in range(max_lag):
        APs.append(A @ APs[-1])
    R_YY = np.array([APs[t][0, 0] for t in range(max_lag + 1)])
    R_UU = np.array([APs[t][1, 1] for t in range(max_lag + 1)])
    R_YU_pos = np.array([APs[t][0, 1] for t in range(max_lag + 1)])
    R_YU_neg = np.array([APs[t][1, 0] for t in range(max_lag + 1)])

    # Joint covariance of (Y^-, U^-, Y^+). Pick t0 = K - 1 so Y^- and U^-
    # occupy absolute times 0..K-1 and Y^+ occupies K..K+H-1.
    t_Y_minus = np.arange(K)
    t_U_minus = np.arange(K)
    t_Y_plus = np.arange(K, K + H)

    def _sym_block(times_a: np.ndarray, times_b: np.ndarray,
                   R_arr: np.ndarray) -> np.ndarray:
        return R_arr[np.abs(np.subtract.outer(times_a, times_b))]

    def _yu_block(times_Y: np.ndarray, times_U: np.ndarray) -> np.ndarray:
        delta = np.subtract.outer(times_Y, times_U)
        out = np.where(
            delta >= 0,
            R_YU_pos[np.clip(delta, 0, max_lag)],
            R_YU_neg[np.clip(-delta, 0, max_lag)],
        )
        return out

    YY_mm = _sym_block(t_Y_minus, t_Y_minus, R_YY)
    UU_mm = _sym_block(t_U_minus, t_U_minus, R_UU)
    YU_mm = _yu_block(t_Y_minus, t_U_minus)

    YY_pm = _sym_block(t_Y_plus, t_Y_minus, R_YY)   # Y^+ vs Y^-
    YU_pm = _yu_block(t_Y_plus, t_U_minus)           # Y^+ vs U^-
    YY_pp = _sym_block(t_Y_plus, t_Y_plus, R_YY)     # Y^+ vs Y^+

    Sigma_YY_minus = YY_mm
    Sigma_Yplus_Yminus = YY_pm
    Sigma_Yplus_Yplus = YY_pp

    # Sigma_{[Y-, U-], [Y-, U-]} block matrix.
    Sigma_BB = np.block([[YY_mm, YU_mm], [YU_mm.T, UU_mm]])
    Sigma_Yplus_B = np.concatenate([YY_pm, YU_pm], axis=1)

    # Schur complements: Sigma_{Y+|Y-} and Sigma_{Y+|Y-,U-}.
    rhs_Y = np.linalg.solve(Sigma_YY_minus, Sigma_Yplus_Yminus.T)
    Sigma_cond_Y = Sigma_Yplus_Yplus - Sigma_Yplus_Yminus @ rhs_Y

    rhs_YU = np.linalg.solve(Sigma_BB, Sigma_Yplus_B.T)
    Sigma_cond_YU = Sigma_Yplus_Yplus - Sigma_Yplus_B @ rhs_YU

    _sign_n, logdet_num = np.linalg.slogdet(Sigma_cond_Y)
    _sign_d, logdet_den = np.linalg.slogdet(Sigma_cond_YU)
    if _sign_n <= 0.0 or _sign_d <= 0.0:
        raise RuntimeError(
            "te_block_arx_gaussian: non-positive determinant in Schur "
            "complement; check input ranges or raise K_history."
        )
    return float(0.5 * (logdet_num - logdet_den))


def _te_block_arx_gaussian_mc(
    rho_u: float,
    rho_y: float,
    c: float,
    sigma2_eta: float,
    sigma2_eps: float,
    H: int,
    D: int,
    *,
    K_history: Optional[int] = None,
    n_samples: int = 200_000,
    burn_in: int = 200,
    seed: int = 0,
) -> float:
    r"""Monte-Carlo cross-check of :func:`te_block_arx_gaussian`.

    Simulates ``n_samples`` long ARX sequences, regresses the future block
    $Y^{+}$ on $Y^{-}$ and on $[Y^{-}, U^{-}]$ by ordinary least squares,
    and returns the half-log-det-ratio of the unbiased residual covariances.
    Used only in unit tests as a numerical sanity check on the closed-form
    determinant ratio; never called at runtime.

    Args:
        rho_u, rho_y, c, sigma2_eta, sigma2_eps, H, D, K_history: As in
            :func:`te_block_arx_gaussian`.
        n_samples: Number of simulated sequences (Monte-Carlo sample size).
        burn_in: Number of warm-up steps discarded so the kept window is
            stationary.
        seed: Seed for the NumPy random generator.

    Returns:
        The Monte-Carlo block transfer entropy estimate in nats.
    """
    K = int(K_history) if K_history is not None else D + 2 * H
    rng = np.random.default_rng(seed)
    n = int(n_samples)
    T = burn_in + K + H

    eta = math.sqrt(sigma2_eta) * rng.standard_normal((n, T))
    eps = math.sqrt(sigma2_eps) * rng.standard_normal((n, T))

    U = np.empty((n, T), dtype=float)
    Y = np.empty((n, T), dtype=float)
    U_prev = np.zeros(n, dtype=float)
    Y_prev = np.zeros(n, dtype=float)
    for k in range(T):
        U_prev = rho_u * U_prev + eta[:, k]
        drive = c * U[:, k - D] if k >= D else 0.0
        Y_prev = rho_y * Y_prev + drive + eps[:, k]
        U[:, k] = U_prev
        Y[:, k] = Y_prev

    lo = burn_in
    y_minus = Y[:, lo : lo + K]
    u_minus = U[:, lo : lo + K]
    y_plus = Y[:, lo + K : lo + K + H]
    ones = np.ones((n, 1), dtype=float)

    logdet_num = _residual_logdet(
        np.concatenate([ones, y_minus], axis=1), y_plus
    )
    logdet_den = _residual_logdet(
        np.concatenate([ones, y_minus, u_minus], axis=1), y_plus
    )
    return float(0.5 * (logdet_num - logdet_den))


def _simulate_state_space_gaussian(
    n: int,
    T: int,
    *,
    oscillators: Sequence[Tuple[float, float]],
    target_ar: float,
    delays: Union[Sequence[int], np.ndarray],
    B_y: Sequence[float],
    sigma2_y: float,
    sigma2_eta: Union[float, Sequence[float]],
    burn_in: int = 500,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Simulate the G1 AR(2)-oscillator-driven Gaussian state-space.

    Each informative source channel $m \in \{0, \ldots, M-1\}$ is generated
    as an AR(2) oscillator

    $$
    s^{(m)}_t = 2\,r_m \cos(\omega_m)\,s^{(m)}_{t-1}
        - r_m^{2}\,s^{(m)}_{t-2} + \eta^{(m)}_t,
    \qquad
    \eta^{(m)}_t \sim \mathcal{N}(0, \sigma^{2}_{\eta, m}).
    $$

    The target channel $m$ couples the lagged source state through

    $$
    y^{(m)}_t = A_y\,y^{(m)}_{t-1} + B_y^{(m)}\,s^{(m)}_{t-D_m}
        + \varepsilon^{(m)}_t,
    \qquad
    \varepsilon^{(m)}_t \sim \mathcal{N}(0, \sigma^{2}_{y}),
    $$

    with $\sigma^2_y$ shared across channels. The simulator is shared
    between :func:`te_block_state_space_gaussian` and the v2 generator
    ``gen_state_space_oscillator`` so the two cannot disagree.

    Args:
        n: Number of independent sequences (samples).
        T: Sequence length (post-burn-in).
        oscillators: Length-$M$ list of $(r_m, \omega_m)$ pairs.
        target_ar: Scalar target self-coefficient $A_y \in [0, 1)$.
        delays: Source-to-target delays $D_m$. One of three shapes: a
            length-$M$ sequence (one delay per channel, shared across all $n$
            samples — the fast path used by the analytic TE); an $(n, M)$
            integer array of **per-sample, per-channel** delays (the
            per-sample-constant variable-delay generator); or an
            $(n, T_{\text{total}}, M)$ integer array of **per-sample,
            per-time, per-channel** delays (the within-signal random-walk
            generator, where the lag $d_{i,t}$ drifts over time). $T_{\text{
            total}} = \text{burn\_in} + T$. Every entry must be $\ge 1$.
        B_y: Length-$M$ list of target loadings $B_y^{(m)}$.
        sigma2_y: Target innovation variance $\sigma^2_y > 0$ (shared).
        sigma2_eta: Source innovation variance — scalar (broadcast) or
            length-$M$ array of positive floats.
        burn_in: Warm-up steps discarded so the kept window is stationary.
        seed: Seed for the NumPy random generator.

    Returns:
        A tuple ``(S, Y)`` of ``np.ndarray`` of shape ``(n, T, M)`` —
        the oscillator state stream ``S`` (which the generator surfaces as
        the informative source channels) and the target stream ``Y``.
    """
    M = len(oscillators)
    if len(B_y) != M:
        raise ValueError(
            "_simulate_state_space_gaussian: oscillators / B_y must have the "
            f"same length; got {M}, {len(B_y)}."
        )
    # ``delays`` may be a length-M sequence (shared across samples), an
    # (n, M) array of per-sample / per-channel delays, or an (n, T_total, M)
    # array of per-sample / per-time / per-channel delays (the random-walk
    # generator). ``per_time`` selects the time-indexed gather in the loop.
    delays_in = np.asarray(delays, dtype=int)
    per_sample = False
    per_time = False
    if delays_in.ndim == 1:
        if delays_in.shape[0] != M:
            raise ValueError(
                "_simulate_state_space_gaussian: 1-D delays must have length "
                f"M={M}, got {delays_in.shape[0]}."
            )
    elif delays_in.ndim == 2:
        if delays_in.shape != (n, M):
            raise ValueError(
                "_simulate_state_space_gaussian: 2-D (per-sample) delays must "
                f"have shape (n, M)=({n}, {M}), got {tuple(delays_in.shape)}."
            )
        per_sample = True
    elif delays_in.ndim == 3:
        if delays_in.shape[0] != n or delays_in.shape[2] != M:
            raise ValueError(
                "_simulate_state_space_gaussian: 3-D (per-time) delays must "
                f"have shape (n, T_total, M)=({n}, *, {M}), got "
                f"{tuple(delays_in.shape)}."
            )
        per_time = True
    else:
        raise ValueError(
            "_simulate_state_space_gaussian: delays must be 1-D (M,), "
            f"2-D (n, M), or 3-D (n, T_total, M), got ndim={delays_in.ndim}."
        )
    if np.any(delays_in < 1):
        raise ValueError(
            "_simulate_state_space_gaussian: every delay must be >= 1."
        )
    sigma2_eta_arr = _broadcast_to_m(sigma2_eta, M, "sigma2_eta")
    if np.any(sigma2_eta_arr <= 0.0):
        raise ValueError(
            "_simulate_state_space_gaussian: every sigma2_eta entry must be > 0."
        )
    if sigma2_y <= 0.0:
        raise ValueError(
            "_simulate_state_space_gaussian: sigma2_y must be > 0."
        )
    if not 0.0 <= target_ar < 1.0:
        raise ValueError(
            f"_simulate_state_space_gaussian: target_ar must be in [0, 1), "
            f"got {target_ar}."
        )
    rs = np.array([r for r, _ in oscillators], dtype=float)
    omegas = np.array([w for _, w in oscillators], dtype=float)
    if np.any((rs < 0.0) | (rs >= 1.0)):
        raise ValueError(
            "_simulate_state_space_gaussian: every oscillator r must lie in "
            "[0, 1)."
        )
    B_y_arr = np.array(B_y, dtype=float)

    rng = np.random.default_rng(seed)
    T_total = burn_in + T
    if per_time and delays_in.shape[1] != T_total:
        raise ValueError(
            "_simulate_state_space_gaussian: 3-D (per-time) delays must span "
            f"burn_in + T = {T_total} time steps, got {delays_in.shape[1]}."
        )
    D_max = int(delays_in.max())
    if D_max >= T_total:
        raise ValueError(
            "_simulate_state_space_gaussian: max delay must be < burn_in + T."
        )
    row = np.arange(n)

    # AR(2) coefficients per channel (one-time).
    ar1 = 2.0 * rs * np.cos(omegas)                   # (M,)
    ar2 = -(rs ** 2)                                   # (M,)
    eta_std = np.sqrt(sigma2_eta_arr)                  # (M,)
    eps_std = math.sqrt(sigma2_y)

    eta = rng.standard_normal((n, T_total, M)) * eta_std  # broadcast
    eps = eps_std * rng.standard_normal((n, T_total, M))

    S = np.zeros((n, T_total, M), dtype=float)
    Y = np.zeros((n, T_total, M), dtype=float)
    for t in range(T_total):
        if t == 0:
            S[:, t, :] = eta[:, t, :]
        elif t == 1:
            S[:, t, :] = ar1 * S[:, t - 1, :] + eta[:, t, :]
        else:
            S[:, t, :] = (
                ar1 * S[:, t - 1, :] + ar2 * S[:, t - 2, :] + eta[:, t, :]
            )
        # Target update: y_t = A_y y_{t-1} + B_y * s_{t-D} + eps_t.
        if t == 0:
            Y[:, t, :] = eps[:, t, :]
        else:
            drive = np.zeros((n, M), dtype=float)
            if not (per_sample or per_time):
                # Fast path: one delay per channel, shared across samples.
                for m in range(M):
                    d_m = int(delays_in[m])
                    if t >= d_m:
                        drive[:, m] = B_y_arr[m] * S[:, t - d_m, m]
            else:
                # Per-sample (2-D) or per-time random-walk (3-D) delay: masked
                # gather over samples. The per-time path reads the lag at the
                # current step, ``d_{i,t}``, shared across the M channels.
                for m in range(M):
                    d_col = delays_in[:, t, m] if per_time else delays_in[:, m]
                    active = t >= d_col                     # (n,) bool
                    if active.any():
                        idx = np.clip(t - d_col, 0, None)   # (n,)
                        src = S[row, idx, m]                # gather over samples
                        drive[active, m] = B_y_arr[m] * src[active]
            Y[:, t, :] = target_ar * Y[:, t - 1, :] + drive + eps[:, t, :]

    return (
        S[:, burn_in:, :].astype(float, copy=False),
        Y[:, burn_in:, :].astype(float, copy=False),
    )


def te_block_state_space_gaussian(
    oscillators: Sequence[Tuple[float, float]],
    target_ar: float,
    delays: Sequence[int],
    B_y: Sequence[float],
    sigma2_y: float,
    sigma2_eta: Union[float, Sequence[float]],
    H: int,
    *,
    K_history: Optional[int] = None,
    n_samples: int = 50_000,
    burn_in: int = 500,
    seed: int = 0,
) -> float:
    r"""Monte-Carlo block transfer entropy for the G1 oscillator state-space.

    For the AR(2)-oscillator-driven Gaussian state-space simulated by
    :func:`_simulate_state_space_gaussian`, computes

    $$
    \widehat{\mathrm{TE}}^{(H)}_{U \to Y}
        = \tfrac{1}{2}\,\bigl(
            \ln\det \widehat\Sigma_{Y^{+} \mid Y^{-}}
            - \ln\det \widehat\Sigma_{Y^{+} \mid Y^{-},\,U^{-}}
        \bigr),
    $$

    where the conditional covariances are unbiased residual covariances of
    OLS fits of $Y^{+}$ on $Y^{-}$ and on $[Y^{-}, U^{-}]$ over
    ``n_samples`` simulated sequences. For jointly Gaussian processes the
    determinant ratio **is** the true block transfer entropy
    (Barnett–Barrett–Seth, 2009). The Monte-Carlo variance scales as
    $O(K^{2} / n_{\text{samples}})$ in the worst case, so the default
    ``n_samples = 50_000`` keeps the standard error below $\sim 1\%$ of
    the mean across the planned benchmark grid; raise it if needed.

    Args:
        oscillators, target_ar, delays, B_y, sigma2_y, sigma2_eta,
        burn_in, seed: As in :func:`_simulate_state_space_gaussian`.
        H: Forecast horizon, $H \ge 1$.
        K_history: History depth used for $Y^{-}$ and $U^{-}$. Defaults to
            ``max(delays) + 2H``, matching the closed-form ARX setting.
        n_samples: Number of simulated sequences.

    Returns:
        The Monte-Carlo block transfer entropy estimate in nats.
    """
    if H <= 0:
        raise ValueError(f"te_block_state_space_gaussian: H must be > 0, got {H}.")
    K = int(K_history) if K_history is not None else int(max(delays)) + 2 * H

    T = K + H
    S, Y = _simulate_state_space_gaussian(
        n=int(n_samples),
        T=T,
        oscillators=oscillators,
        target_ar=target_ar,
        delays=delays,
        B_y=B_y,
        sigma2_y=sigma2_y,
        sigma2_eta=sigma2_eta,
        burn_in=burn_in,
        seed=seed,
    )
    # Flatten the per-channel history into the design matrices (each
    # informative channel contributes independently to Y^- and U^-).
    n = S.shape[0]
    M = S.shape[2]
    y_minus = Y[:, :K, :].reshape(n, K * M)            # (n, K*M)
    u_minus = S[:, :K, :].reshape(n, K * M)            # (n, K*M)
    y_plus = Y[:, K : K + H, :].reshape(n, H * M)      # (n, H*M)
    ones = np.ones((n, 1), dtype=float)

    logdet_num = _residual_logdet(
        np.concatenate([ones, y_minus], axis=1), y_plus
    )
    logdet_den = _residual_logdet(
        np.concatenate([ones, y_minus, u_minus], axis=1), y_plus
    )
    return float(0.5 * (logdet_num - logdet_den))


def snr_per_step_for_te_block(te_block: float, H: int, M: int) -> float:
    r"""Per-step, per-channel innovation SNR implied by a block TE.

    Inverts the block$\to$per-step decomposition $\mathrm{TE}^{(H)}_{\text{block}}
    \approx H\,M\cdot\tfrac12\ln(1+\mathrm{SNR})$ for a Gaussian channel:

    $$
    \mathrm{SNR}
        \;=\; \frac{\operatorname{Var}(\text{source drive})}{\sigma_y^2}
        \;\approx\; \exp\!\Big(\frac{2\,\mathrm{TE}^{(H)}_{\text{block}}}{H\,M}\Big) - 1 .
    $$

    A back-of-envelope extractability gauge: a finite model needs the
    source-driven component to be a non-trivial fraction of the target
    innovation variance. Values $\lesssim 0.01$ (≈1 %) are effectively
    unextractable at the usual per-cell sample sizes.

    Args:
        te_block: Block transfer entropy in nats ($\ge 0$).
        H: Forecast horizon ($\ge 1$).
        M: Informative-channel count ($\ge 1$).

    Returns:
        The implied per-step, per-channel SNR (dimensionless, $\ge 0$).
    """
    if H <= 0 or M <= 0:
        raise ValueError("snr_per_step_for_te_block: H and M must be > 0.")
    return float(math.expm1(2.0 * max(float(te_block), 0.0) / (float(H) * float(M))))


def _ridge_holdout_logdet(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_te: np.ndarray, y_te: np.ndarray, ridge: float,
) -> float:
    r"""Log-det of the held-out residual covariance of a ridge fit.

    Fits $\hat B = (X_{\mathrm{tr}}^\top X_{\mathrm{tr}} + \lambda I)^{-1}
    X_{\mathrm{tr}}^\top Y_{\mathrm{tr}}$ (the intercept column is left
    unpenalised) and returns $\ln\det\hat\Sigma$ with
    $\hat\Sigma = R_{\mathrm{te}}^\top R_{\mathrm{te}} / n_{\mathrm{te}}$ on the
    held-out residuals $R_{\mathrm{te}} = Y_{\mathrm{te}} - X_{\mathrm{te}}\hat B$.
    Evaluating on held-out rows is what stops over-fitting from spuriously
    shrinking the residual covariance (and inflating the gain). A small jitter
    keeps the covariance positive-definite for ``slogdet``.

    Args:
        x_tr, y_tr: Training design / target matrices.
        x_te, y_te: Held-out design / target matrices.
        ridge: $L_2$ penalty scaled by the mean diagonal of the training Gram.

    Returns:
        The natural log-determinant of the held-out residual covariance.
    """
    gram = x_tr.T @ x_tr
    p = gram.shape[0]
    lam = float(ridge) * (float(np.trace(gram)) / max(p, 1))
    reg = lam * np.eye(p)
    reg[0, 0] = 0.0  # do not penalise the intercept column
    coef = np.linalg.solve(gram + reg, x_tr.T @ y_tr)
    resid = y_te - x_te @ coef
    cov = resid.T @ resid / float(resid.shape[0])
    h = cov.shape[0]
    jitter = 1e-8 * (float(np.trace(cov)) / max(h, 1) + 1e-12)
    _sign, logdet = np.linalg.slogdet(cov + jitter * np.eye(h))
    return float(logdet)


def realizable_te_block_from_arrays(
    Y: np.ndarray,
    U: np.ndarray,
    *,
    M: int,
    K: int,
    H: int,
    delay_max: Optional[int] = None,
    anchor: Optional[int] = None,
    ridge: float = 1e-2,
    train_frac: float = 0.7,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""Block TE *realizable* by a finite ridge predictor on cached arrays.

    Estimates the same Barnett--Barrett--Seth determinant-ratio block transfer
    entropy as :func:`te_block_state_space_gaussian`, but on **already-generated,
    z-scored cache arrays** and at the *training* sample size, using a held-out
    split so over-fitting cannot inflate the gain:

    $$
    \widehat{\mathrm{TE}}^{(H),\,\text{real}}_{U\to Y}
        = \tfrac12\Big(
            \ln\det\widehat\Sigma^{\text{test}}_{Y^{+}\mid Y^{-}}
          - \ln\det\widehat\Sigma^{\text{test}}_{Y^{+}\mid Y^{-},\,U^{-}}
        \Big),
    $$

    where the two ridge regressions ($Y^{+}\!\sim\!Y^{-}$ and
    $Y^{+}\!\sim\![Y^{-},U^{-}]$) are fit on a training partition and the
    residual covariances are evaluated on a held-out partition. The informative
    source channels ``U[:, :, :M]`` are exactly the oscillator state the analytic
    TE conditions on (the generator writes them into the front channels), so the
    ratio is directly comparable to a cell's ``te_cell_realised``.

    Comparing this to the analytic block TE answers the realizability question
    that gates the experiment: a ratio $\to 1$ means an optimal-ish predictor
    *can* extract the TE at this sample size (so a downstream calibration failure
    is a model/training problem); a ratio $\ll 1$ means the TE is *not
    extractable* at this $n$ (a data-design problem -- concentrate the signal).

    Args:
        Y: Target cache array of shape $(n, T, C_y)$ (z-scored).
        U: Source cache array of shape $(n, T, C_u)$; channels $[0, M)$ are the
            informative oscillator state.
        M: Informative-channel count of the cell.
        K: History depth for the target self-history $Y^{-}$ (use the cell's
            ``K_history``).
        H: Forecast horizon.
        delay_max: Upper lag of the cell's band; the source regressors $U^{-}$
            are scoped to lags $0..\,$``delay_max`` (the band the true coupling
            lives in), so the held-out gain reflects a lag-selecting predictor
            rather than a blind $K$-lag regression that pays the estimation
            variance of $K\,M$ noisy source coefficients. ``None`` uses the full
            $K$ source lags (matches the analytic support but is noisier at
            finite $n$).
        anchor: Anchor index $t_0$ (history $[t_0-K, t_0)$, future
            $[t_0, t_0+H)$). Defaults to a mid-sequence position for
            stationarity.
        ridge: $L_2$ penalty added to the regressor Gram diagonal (relative to
            its mean), shared across the reduced and full fits. Negligible for
            the low-$M$ headline cells where regressors $\ll$ samples.
        train_frac: Fraction of samples used to fit; the remainder evaluates the
            residual covariance.
        seed: Seed for the train/test row shuffle.

    Returns:
        Dict with ``realizable_gain`` (nats, block), ``snr_per_step``,
        ``m_used``, ``anchor``, ``n_train``, ``n_test`` and ``ill_conditioned``
        (``True`` with ``realizable_gain = nan`` when the held-out set is too
        small to estimate the $HM \times HM$ residual covariance, i.e.
        ``n_test <= H * m_used`` -- which happens only for the heavily diluted,
        non-gated high-$M$ cells).
    """
    Y = np.asarray(Y, dtype=float)
    U = np.asarray(U, dtype=float)
    if Y.ndim != 3 or U.ndim != 3:
        raise ValueError(
            "realizable_te_block_from_arrays: Y and U must be (n, T, C)."
        )
    n, T, _ = Y.shape
    m_used = int(min(int(M), Y.shape[2], U.shape[2]))
    if m_used <= 0:
        raise ValueError("realizable_te_block_from_arrays: M must be >= 1.")
    t0 = (max(int(K), (T - int(H)) // 2) if anchor is None else int(anchor))
    if t0 < int(K) or t0 + int(H) > T:
        raise ValueError(
            f"realizable_te_block_from_arrays: anchor {t0} needs K={K} history "
            f"and H={H} future within T={T}."
        )
    Yc = Y[:, :, :m_used]
    Uc = U[:, :, :m_used]
    # Target self-history: full K lags (shared by both fits, so its estimation
    # variance cancels in the gain). Source history: scoped to the band lags
    # 0..delay_max, since the true coupling lies in-band -- regressing on all K
    # source lags is asymptotically identical but pays the held-out estimation
    # variance of K*M noisy coefficients, which a lag-attention model avoids.
    w_u = (int(K) if delay_max is None
           else int(min(int(delay_max) + 1, t0 + 1, int(K))))
    y_minus = Yc[:, t0 - K : t0, :].reshape(n, K * m_used)
    u_minus = Uc[:, t0 - w_u + 1 : t0 + 1, :].reshape(n, w_u * m_used)
    y_plus = Yc[:, t0 : t0 + H, :].reshape(n, H * m_used)
    ones = np.ones((n, 1), dtype=float)
    x_red = np.concatenate([ones, y_minus], axis=1)
    x_full = np.concatenate([ones, y_minus, u_minus], axis=1)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_tr = max(1, int(round(float(train_frac) * n)))
    tr, te = perm[:n_tr], perm[n_tr:]
    out: Dict[str, Any] = {
        "m_used": m_used, "anchor": t0,
        "n_train": int(n_tr), "n_test": int(te.size),
    }
    if int(te.size) <= H * m_used:
        out.update(realizable_gain=float("nan"), snr_per_step=float("nan"),
                   ill_conditioned=True)
        return out
    ld_red = _ridge_holdout_logdet(x_red[tr], y_plus[tr], x_red[te], y_plus[te],
                                   ridge)
    ld_full = _ridge_holdout_logdet(x_full[tr], y_plus[tr], x_full[te],
                                    y_plus[te], ridge)
    gain = float(0.5 * (ld_red - ld_full))
    out.update(realizable_gain=gain,
               snr_per_step=float(math.expm1(2.0 * max(gain, 0.0)
                                             / (H * m_used))),
               ill_conditioned=False)
    return out


def _bisect_for_te_target(
    eval_fn: Callable[[float], float],
    target_te_block: float,
    *,
    lo: float,
    hi: float,
    tol: float,
    max_iter: int,
    label: str,
) -> Tuple[float, float, int]:
    r"""Bisect a monotone TE estimator to land on a target block TE.

    Solves :math:`\widehat{\mathrm{TE}}(x) = \mathrm{target\_te\_block}` on
    :math:`x \in [\mathrm{lo}, \mathrm{hi}]` by bisection. Assumes
    ``eval_fn`` is monotone increasing on the bracket; the caller is
    responsible for holding any stochastic seed fixed across iterations.

    Stop conditions: relative bracket width *or* relative TE error falls
    below ``tol``; the loop is capped at ``max_iter`` (returning the final
    midpoint on the falling edge).

    Args:
        eval_fn: Maps a coupling magnitude :math:`x` to its block TE in
            nats. Called twice for the bracket check, then up to
            ``max_iter`` more times in the loop.
        target_te_block: Target block TE in nats; must satisfy
            :math:`\mathrm{eval\_fn}(\mathrm{lo}) < \mathrm{target} <
            \mathrm{eval\_fn}(\mathrm{hi})`.
        lo, hi: Bracket on the coupling magnitude.
        tol, max_iter: Stop conditions.
        label: Prefix used in the ``ValueError`` raised by the bracket
            check (so the caller's error message identifies the inverter).

    Returns:
        ``(root, te_block_at_root, n_iter)``.

    Raises:
        ValueError: If the initial bracket does not contain ``target_te_block``.
    """
    te_lo = eval_fn(lo)
    te_hi = eval_fn(hi)
    if not (te_lo < target_te_block < te_hi):
        raise ValueError(
            f"{label}: initial bracket does not contain the target; "
            f"TE({lo})={te_lo:.4f}, TE({hi})={te_hi:.4f}, "
            f"target={target_te_block:.4f}. Widen [lo, hi]."
        )

    n_iter = 0
    while n_iter < max_iter:
        mid = 0.5 * (lo + hi)
        te_mid = eval_fn(mid)
        n_iter += 1
        bracket_rel = (hi - lo) / max(mid, 1e-12)
        te_rel = abs(te_mid - target_te_block) / max(target_te_block, 1e-12)
        if bracket_rel < tol or te_rel < tol:
            return mid, te_mid, n_iter
        if te_mid < target_te_block:
            lo = mid
        else:
            hi = mid

    root = 0.5 * (lo + hi)
    return root, eval_fn(root), n_iter + 1


def B_y_for_te_block_state_space(
    target_te_block: float,
    *,
    oscillators: Sequence[Tuple[float, float]],
    target_ar: float,
    delays: Sequence[int],
    sigma2_y: float,
    sigma2_eta: Union[float, Sequence[float]],
    H: int,
    K_history: Optional[int] = None,
    n_samples: int = 50_000,
    burn_in: int = 500,
    seed: int = 0,
    lo: float = 1e-4,
    hi: float = 10.0,
    tol: float = 1e-3,
    max_iter: int = 40,
) -> Dict[str, Any]:
    r"""Invert :func:`te_block_state_space_gaussian` for a target block TE.

    Solves for the **uniform** target-loading magnitude $b$ such that

    $$
    \widehat{\mathrm{TE}}^{(H)}_{U \to Y}\!\bigl(B_y = b \cdot \mathbf{1}_M\bigr)
        = \mathrm{target\_te\_block}
    $$

    by bisection on the Monte-Carlo determinant-ratio estimator. The
    estimator is monotone increasing in $|b|$ for fixed oscillator /
    delay / variance hyperparameters (the cross-covariance scales linearly
    in $b$, the determinant ratio monotonically in the cross-contribution),
    so bisection converges robustly. ``seed`` is held fixed across
    iterations so the Monte-Carlo noise does not break monotonicity.

    The bracket $[\mathrm{lo}, \mathrm{hi}]$ must satisfy
    $\mathrm{TE}(\mathrm{lo}) < \mathrm{target\_te\_block} < \mathrm{TE}(\mathrm{hi})$;
    otherwise the function raises ``ValueError`` with the achieved values
    so the caller can widen the bracket. For the
    ``model_validation_v2_plan §5.3`` operating regime
    (per-step TE $\in \{0.05, 0.15, 0.30\}$, block TE $\in \{1.5, 4.5, 9.0\}$
    nats, the default G1 oscillator at $r=0.99$, $\omega=0.05$, $D=60$,
    $A_y=0.95$, $M=4$), the default bracket $[10^{-4}, 10]$ comfortably
    contains the root.

    Args:
        target_te_block: Target block transfer entropy in nats. Must be in
            $[0, \mathrm{TE}(\mathrm{hi}))$; ``0`` returns immediately with
            $B_y = 0$.
        oscillators, target_ar, delays, sigma2_y, sigma2_eta, H, K_history,
        n_samples, burn_in, seed: Forwarded to
            :func:`te_block_state_space_gaussian`. The number of informative
            channels $M$ is inferred from ``len(oscillators)``.
        lo, hi: Initial bisection bracket on the uniform $B_y$ magnitude.
        tol: Relative tolerance on either the bracket width or the achieved
            block TE — the loop stops when either falls below it.
        max_iter: Maximum bisection iterations.

    Returns:
        A dict with keys ``B_y`` (length-$M$ list of identical floats;
        matches the forward API), ``B_y_scalar`` (the bisected magnitude),
        ``te_block`` (achieved block TE in nats), ``te_per_step``
        (= ``te_block / H``), and ``n_iter`` (iterations used).

    Raises:
        ValueError: If ``target_te_block < 0``, ``H <= 0``,
            ``lo <= 0``, ``hi <= lo``, ``max_iter <= 0``, or if the initial
            bracket does not contain the target TE.
    """
    if H <= 0:
        raise ValueError(f"B_y_for_te_block_state_space: H must be > 0, got {H}.")
    if target_te_block < 0.0:
        raise ValueError(
            f"B_y_for_te_block_state_space: target_te_block must be >= 0, "
            f"got {target_te_block}."
        )
    if lo <= 0.0:
        raise ValueError(
            f"B_y_for_te_block_state_space: lo must be > 0, got {lo}."
        )
    if hi <= lo:
        raise ValueError(
            f"B_y_for_te_block_state_space: hi must be > lo, "
            f"got lo={lo}, hi={hi}."
        )
    if max_iter <= 0:
        raise ValueError(
            f"B_y_for_te_block_state_space: max_iter must be > 0, "
            f"got {max_iter}."
        )
    M = len(oscillators)
    if M == 0:
        raise ValueError(
            "B_y_for_te_block_state_space: at least one oscillator is required."
        )

    if target_te_block == 0.0:
        return {
            "B_y": [0.0] * M,
            "B_y_scalar": 0.0,
            "te_block": 0.0,
            "te_per_step": 0.0,
            "n_iter": 0,
        }

    def _eval(b: float) -> float:
        return te_block_state_space_gaussian(
            oscillators=oscillators,
            target_ar=target_ar,
            delays=delays,
            B_y=[b] * M,
            sigma2_y=sigma2_y,
            sigma2_eta=sigma2_eta,
            H=H,
            K_history=K_history,
            n_samples=n_samples,
            burn_in=burn_in,
            seed=seed,
        )

    b_star, te_block, n_iter = _bisect_for_te_target(
        _eval, target_te_block, lo=lo, hi=hi, tol=tol, max_iter=max_iter,
        label="B_y_for_te_block_state_space",
    )

    return {
        "B_y": [float(b_star)] * M,
        "B_y_scalar": float(b_star),
        "te_block": float(te_block),
        "te_per_step": float(te_block) / float(H),
        "n_iter": int(n_iter),
    }


def c_for_te_block_arx(
    target_te_block: float,
    rho_u: float,
    rho_y: float,
    sigma2_eta: float,
    sigma2_eps: float,
    H: int,
    D: int,
    *,
    K_history: Optional[int] = None,
    lo: float = 1.0e-4,
    hi: float = 5.0,
    tol: float = 1.0e-3,
    max_iter: int = 40,
) -> Dict[str, float]:
    r"""Invert the ARX coupling :math:`c` to hit a target block TE.

    Bisects the (positive) coupling magnitude :math:`c` so that the closed-form
    block TE of the v2-G2 process matches ``target_te_block``. Because
    :func:`te_block_arx_gaussian` is monotone increasing in :math:`|c|` for
    fixed AR / variance hyperparameters, bisection converges robustly with no
    Monte-Carlo noise.

    The bracket :math:`[\mathrm{lo}, \mathrm{hi}]` must satisfy
    :math:`\mathrm{TE}(\mathrm{lo}) < \mathrm{target\_te\_block} <
    \mathrm{TE}(\mathrm{hi})`; otherwise the function raises ``ValueError``
    with the achieved values so the caller can widen the bracket. For the
    default G2 calibration regime (per-step TE :math:`\in \{0.05, 0.15, 0.30\}`,
    block TE :math:`\in \{1.5, 4.5, 9.0\}` nats, the default G2 process at
    :math:`\rho_u = 0.99`, :math:`\rho_y = 0.95`, :math:`D = 60`,
    :math:`\sigma^2_\eta = \sigma^2_\varepsilon = 1`), the default bracket
    :math:`[10^{-4}, 5]` comfortably contains the root.

    Args:
        target_te_block: Target *per-channel* block TE in nats. Must be
            :math:`\ge 0`; ``0`` returns immediately with :math:`c = 0`.
        rho_u: AR(1) coefficient of :math:`U`, :math:`0 \le \rho_u < 1`.
        rho_y: AR(1) coefficient of :math:`Y`, :math:`0 \le \rho_y < 1`.
        sigma2_eta: Innovation variance of :math:`U`, :math:`> 0`.
        sigma2_eps: Innovation variance of :math:`Y`, :math:`> 0`.
        H: Forecast horizon, :math:`H \ge 1`.
        D: Source-to-target delay, :math:`D \ge 1`.
        K_history: History depth forwarded to :func:`te_block_arx_gaussian`
            (default :math:`D + 2H`).
        lo, hi: Initial bisection bracket on :math:`c`.
        tol: Relative tolerance on either the bracket width or the achieved
            block TE -- the loop stops when either falls below it.
        max_iter: Maximum bisection iterations.

    Returns:
        A dict with keys ``c_scalar`` (the bisected magnitude), ``te_block``
        (achieved per-channel block TE in nats), ``te_per_step``
        (``te_block / H``), and ``n_iter`` (iterations used).

    Raises:
        ValueError: If ``target_te_block < 0``, ``H <= 0``, ``D < 1``,
            ``lo <= 0``, ``hi <= lo``, ``max_iter <= 0``, or if the initial
            bracket does not contain the target TE.
    """
    if H <= 0:
        raise ValueError(f"c_for_te_block_arx: H must be > 0, got {H}.")
    if D < 1:
        raise ValueError(f"c_for_te_block_arx: D must be >= 1, got {D}.")
    if target_te_block < 0.0:
        raise ValueError(
            f"c_for_te_block_arx: target_te_block must be >= 0, "
            f"got {target_te_block}."
        )
    if lo <= 0.0:
        raise ValueError(f"c_for_te_block_arx: lo must be > 0, got {lo}.")
    if hi <= lo:
        raise ValueError(
            f"c_for_te_block_arx: hi must be > lo, got lo={lo}, hi={hi}."
        )
    if max_iter <= 0:
        raise ValueError(
            f"c_for_te_block_arx: max_iter must be > 0, got {max_iter}."
        )

    if target_te_block == 0.0:
        return {
            "c_scalar": 0.0,
            "te_block": 0.0,
            "te_per_step": 0.0,
            "n_iter": 0,
        }

    def _eval(c: float) -> float:
        return te_block_arx_gaussian(
            rho_u=rho_u, rho_y=rho_y, c=c,
            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
            H=H, D=D, K_history=K_history,
        )

    c_star, te_block, n_iter = _bisect_for_te_target(
        _eval, target_te_block, lo=lo, hi=hi, tol=tol, max_iter=max_iter,
        label="c_for_te_block_arx",
    )

    return {
        "c_scalar": float(c_star),
        "te_block": float(te_block),
        "te_per_step": float(te_block) / float(H),
        "n_iter": int(n_iter),
    }


def _validate_delay_range(delay_min: int, delay_max: int, label: str) -> None:
    r"""Validate an inclusive integer delay range $\{d_{\min},\dots,d_{\max}\}$.

    Args:
        delay_min: Smallest delay in the range, $\ge 1$.
        delay_max: Largest delay in the range, $\ge d_{\min}$.
        label: Caller name, used only in the error message.

    Raises:
        ValueError: If ``delay_min < 1`` or ``delay_max < delay_min``.
    """
    if delay_min < 1:
        raise ValueError(f"{label}: delay_min must be >= 1, got {delay_min}.")
    if delay_max < delay_min:
        raise ValueError(
            f"{label}: delay_max must be >= delay_min, got "
            f"delay_min={delay_min}, delay_max={delay_max}."
        )


def mean_te_block_state_space_over_delays(
    *,
    delay_min: int,
    delay_max: int,
    oscillators: Sequence[Tuple[float, float]],
    target_ar: float,
    B_y: Union[float, Sequence[float]],
    sigma2_y: float,
    sigma2_eta: Union[float, Sequence[float]],
    H: int,
    K_history: Optional[int] = None,
    n_samples: int = 50_000,
    burn_in: int = 500,
    seed: int = 0,
) -> float:
    r"""Mean block TE of the G1 oscillator state-space over a delay range.

    For the variable-delay G1 generator each sample draws a single delay
    $d \sim \mathrm{Uniform}\{d_{\min},\dots,d_{\max}\}$ shared across its $M$
    informative channels. Because every channel of a sample uses the same
    delay, the exact block TE of that sample is
    :func:`te_block_state_space_gaussian` evaluated at ``delays = [d]*M``. The
    dataset-level ground-truth TE is the expectation over the (uniform) delay
    distribution,

    $$
    \overline{\mathrm{TE}}
        = \frac{1}{d_{\max}-d_{\min}+1}
          \sum_{d=d_{\min}}^{d_{\max}}
          \widehat{\mathrm{TE}}^{(H)}_{U\to Y}\!\bigl(\text{delays}=[d]\!*\!M\bigr),
    $$

    which this helper computes (the build reports the *realised* sample mean,
    which converges to this as $n\to\infty$).

    Args:
        delay_min: Smallest delay in the range, $\ge 1$.
        delay_max: Largest delay in the range (inclusive), $\ge d_{\min}$.
        oscillators: Length-$M$ list of $(r_m, \omega_m)$ oscillator specs.
        target_ar: Target self-coefficient $A_y \in [0, 1)$.
        B_y: Uniform target loading magnitude (scalar) or length-$M$ list.
        sigma2_y: Target innovation variance $\sigma^2_y > 0$.
        sigma2_eta: Source innovation variance (scalar or length-$M$).
        H: Forecast horizon, $H \ge 1$.
        K_history: History depth forwarded to
            :func:`te_block_state_space_gaussian`. With a near-unit-root source
            this should be set generously (e.g. 120-200) so the conditioning
            captures the long source memory; defaults to ``max(delays)+2H``.
        n_samples: Monte-Carlo sample size per delay.
        burn_in: Warm-up steps discarded per simulation.
        seed: Base seed; delay $d$ uses ``seed + d`` so each MC estimate is
            distinct but held fixed across coupling-bisection iterations.

    Returns:
        The mean block transfer entropy over the delay range, in nats.
    """
    if H <= 0:
        raise ValueError(
            f"mean_te_block_state_space_over_delays: H must be > 0, got {H}."
        )
    _validate_delay_range(
        delay_min, delay_max, "mean_te_block_state_space_over_delays"
    )
    M = len(oscillators)
    if M == 0:
        raise ValueError(
            "mean_te_block_state_space_over_delays: at least one oscillator "
            "is required."
        )
    B_y_arr = _broadcast_to_m(B_y, M, "B_y")
    total = 0.0
    n_delays = delay_max - delay_min + 1
    for d in range(delay_min, delay_max + 1):
        total += te_block_state_space_gaussian(
            oscillators=oscillators,
            target_ar=target_ar,
            delays=[int(d)] * M,
            B_y=B_y_arr.tolist(),
            sigma2_y=sigma2_y,
            sigma2_eta=sigma2_eta,
            H=H,
            K_history=K_history,
            n_samples=n_samples,
            burn_in=burn_in,
            seed=int(seed) + int(d),
        )
    return float(total / n_delays)


def B_y_for_mean_te_block_state_space(
    target_te_block: float,
    *,
    delay_min: int,
    delay_max: int,
    oscillators: Sequence[Tuple[float, float]],
    target_ar: float,
    sigma2_y: float,
    sigma2_eta: Union[float, Sequence[float]],
    H: int,
    K_history: Optional[int] = None,
    n_samples: int = 50_000,
    burn_in: int = 500,
    seed: int = 0,
    lo: float = 1e-4,
    hi: float = 10.0,
    tol: float = 1e-3,
    max_iter: int = 40,
) -> Dict[str, Any]:
    r"""Invert the mean-over-delays G1 block TE for a uniform $B_y$ magnitude.

    Bisects the uniform target-loading magnitude $b$ so that
    :func:`mean_te_block_state_space_over_delays` (with ``B_y = b``) equals
    ``target_te_block``. The mean of monotone-increasing MC TE estimators is
    itself monotone increasing in $|b|$, so bisection converges; seeds are
    held fixed across iterations.

    Args:
        target_te_block: Target mean block TE in nats, $\ge 0$. ``0`` returns
            immediately with $B_y = 0$.
        delay_min, delay_max, oscillators, target_ar, sigma2_y, sigma2_eta,
        H, K_history, n_samples, burn_in, seed: Forwarded to
            :func:`mean_te_block_state_space_over_delays`.
        lo, hi: Initial bisection bracket on the uniform $B_y$ magnitude.
        tol, max_iter: Stop conditions, as in :func:`_bisect_for_te_target`.

    Returns:
        A dict with ``B_y`` (length-$M$ list of identical floats), ``B_y_scalar``,
        ``te_block`` (achieved mean block TE), ``te_per_step`` (= ``te_block/H``)
        and ``n_iter``.

    Raises:
        ValueError: On invalid arguments or if the bracket misses the target.
    """
    if H <= 0:
        raise ValueError(
            f"B_y_for_mean_te_block_state_space: H must be > 0, got {H}."
        )
    if target_te_block < 0.0:
        raise ValueError(
            "B_y_for_mean_te_block_state_space: target_te_block must be >= 0, "
            f"got {target_te_block}."
        )
    if lo <= 0.0:
        raise ValueError(
            f"B_y_for_mean_te_block_state_space: lo must be > 0, got {lo}."
        )
    if hi <= lo:
        raise ValueError(
            f"B_y_for_mean_te_block_state_space: hi must be > lo, "
            f"got lo={lo}, hi={hi}."
        )
    if max_iter <= 0:
        raise ValueError(
            f"B_y_for_mean_te_block_state_space: max_iter must be > 0, "
            f"got {max_iter}."
        )
    _validate_delay_range(
        delay_min, delay_max, "B_y_for_mean_te_block_state_space"
    )
    M = len(oscillators)
    if M == 0:
        raise ValueError(
            "B_y_for_mean_te_block_state_space: at least one oscillator is "
            "required."
        )

    if target_te_block == 0.0:
        return {
            "B_y": [0.0] * M,
            "B_y_scalar": 0.0,
            "te_block": 0.0,
            "te_per_step": 0.0,
            "n_iter": 0,
        }

    def _eval(b: float) -> float:
        return mean_te_block_state_space_over_delays(
            delay_min=delay_min,
            delay_max=delay_max,
            oscillators=oscillators,
            target_ar=target_ar,
            B_y=b,
            sigma2_y=sigma2_y,
            sigma2_eta=sigma2_eta,
            H=H,
            K_history=K_history,
            n_samples=n_samples,
            burn_in=burn_in,
            seed=seed,
        )

    b_star, te_block, n_iter = _bisect_for_te_target(
        _eval, target_te_block, lo=lo, hi=hi, tol=tol, max_iter=max_iter,
        label="B_y_for_mean_te_block_state_space",
    )
    return {
        "B_y": [float(b_star)] * M,
        "B_y_scalar": float(b_star),
        "te_block": float(te_block),
        "te_per_step": float(te_block) / float(H),
        "n_iter": int(n_iter),
    }


def mean_te_block_arx_over_delays(
    *,
    delay_min: int,
    delay_max: int,
    rho_u: float,
    rho_y: float,
    c: float,
    sigma2_eta: float,
    sigma2_eps: float,
    H: int,
    M: int = 1,
    K_history: Optional[int] = None,
) -> float:
    r"""Mean block TE of the G2 ARX process over a delay range.

    For the variable-delay G2 generator each sample draws one delay
    $d \sim \mathrm{Uniform}\{d_{\min},\dots,d_{\max}\}$ shared across its $M$
    channels. The per-channel block TE is the closed-form
    :func:`te_block_arx_gaussian`; the $M$ channels are independent so the
    sample TE is $M$ times the per-channel value, and the dataset mean is the
    uniform delay average

    $$
    \overline{\mathrm{TE}}
        = \frac{M}{d_{\max}-d_{\min}+1}
          \sum_{d=d_{\min}}^{d_{\max}}
          \mathrm{TE}^{(H)}_{U\to Y}\!\bigl(D=d\bigr).
    $$

    Args:
        delay_min: Smallest delay, $\ge 1$.
        delay_max: Largest delay (inclusive), $\ge d_{\min}$.
        rho_u, rho_y, c, sigma2_eta, sigma2_eps, H, K_history: Forwarded to
            :func:`te_block_arx_gaussian`.
        M: Number of independent informative channels (the per-channel TE is
            scaled by $M$, matching the generator's ``te_true``).

    Returns:
        The mean block transfer entropy over the delay range, in nats.
    """
    if H <= 0:
        raise ValueError(
            f"mean_te_block_arx_over_delays: H must be > 0, got {H}."
        )
    if M < 1:
        raise ValueError(
            f"mean_te_block_arx_over_delays: M must be >= 1, got {M}."
        )
    _validate_delay_range(delay_min, delay_max, "mean_te_block_arx_over_delays")
    total = 0.0
    n_delays = delay_max - delay_min + 1
    for d in range(delay_min, delay_max + 1):
        total += te_block_arx_gaussian(
            rho_u=rho_u, rho_y=rho_y, c=c,
            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
            H=H, D=int(d), K_history=K_history,
        )
    return float(M * total / n_delays)


def c_for_mean_te_block_arx(
    target_te_block: float,
    *,
    delay_min: int,
    delay_max: int,
    rho_u: float,
    rho_y: float,
    sigma2_eta: float,
    sigma2_eps: float,
    H: int,
    M: int = 1,
    K_history: Optional[int] = None,
    lo: float = 1.0e-4,
    hi: float = 5.0,
    tol: float = 1.0e-3,
    max_iter: int = 40,
) -> Dict[str, float]:
    r"""Invert the mean-over-delays G2 block TE for the ARX coupling $c$.

    Bisects the coupling magnitude $c$ so that
    :func:`mean_te_block_arx_over_delays` equals ``target_te_block`` (the
    $M$-scaled dataset TE). The closed-form per-delay TE is monotone in $|c|$,
    so the mean is too and bisection converges with no Monte-Carlo noise.

    Args:
        target_te_block: Target $M$-scaled mean block TE in nats, $\ge 0$.
            ``0`` returns immediately with $c = 0$.
        delay_min, delay_max, rho_u, rho_y, sigma2_eta, sigma2_eps, H, M,
        K_history: Forwarded to :func:`mean_te_block_arx_over_delays`.
        lo, hi: Initial bisection bracket on $c$.
        tol, max_iter: Stop conditions, as in :func:`_bisect_for_te_target`.

    Returns:
        A dict with ``c_scalar`` (bisected coupling), ``te_block`` (achieved
        $M$-scaled mean block TE), ``te_per_step`` (= ``te_block/H``) and
        ``n_iter``.

    Raises:
        ValueError: On invalid arguments or if the bracket misses the target.
    """
    if H <= 0:
        raise ValueError(f"c_for_mean_te_block_arx: H must be > 0, got {H}.")
    if M < 1:
        raise ValueError(f"c_for_mean_te_block_arx: M must be >= 1, got {M}.")
    if target_te_block < 0.0:
        raise ValueError(
            "c_for_mean_te_block_arx: target_te_block must be >= 0, "
            f"got {target_te_block}."
        )
    if lo <= 0.0:
        raise ValueError(f"c_for_mean_te_block_arx: lo must be > 0, got {lo}.")
    if hi <= lo:
        raise ValueError(
            f"c_for_mean_te_block_arx: hi must be > lo, got lo={lo}, hi={hi}."
        )
    if max_iter <= 0:
        raise ValueError(
            f"c_for_mean_te_block_arx: max_iter must be > 0, got {max_iter}."
        )
    _validate_delay_range(delay_min, delay_max, "c_for_mean_te_block_arx")

    if target_te_block == 0.0:
        return {
            "c_scalar": 0.0,
            "te_block": 0.0,
            "te_per_step": 0.0,
            "n_iter": 0,
        }

    def _eval(c: float) -> float:
        return mean_te_block_arx_over_delays(
            delay_min=delay_min,
            delay_max=delay_max,
            rho_u=rho_u, rho_y=rho_y, c=c,
            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
            H=H, M=M, K_history=K_history,
        )

    c_star, te_block, n_iter = _bisect_for_te_target(
        _eval, target_te_block, lo=lo, hi=hi, tol=tol, max_iter=max_iter,
        label="c_for_mean_te_block_arx",
    )
    return {
        "c_scalar": float(c_star),
        "te_block": float(te_block),
        "te_per_step": float(te_block) / float(H),
        "n_iter": int(n_iter),
    }


def te_categorical_switch_block(p: float, K: int, H: int) -> float:
    r"""Block transfer entropy of the G3 inclusive-redraw regime switch.

    For an inclusive-redraw categorical Markov chain on $K$ classes with
    switch probability $p$, the regime at each step is renewed
    independently of the past, so the per-step transfer entropy
    :func:`te_categorical_switch` accumulates additively over a horizon
    of $H$ future steps (per informative channel):

    $$
    \mathrm{TE}^{(H)}_{U \to Y}
        = H \cdot \mathrm{TE}^{(1)}_{U \to Y}.
    $$

    The G3 generator scales by the number of independent regime processes
    ($M$ informative channels), which is **not** done inside this helper.

    Args:
        p: Switch probability in $[0, 1]$.
        K: Number of categorical classes, $K \ge 2$.
        H: Forecast horizon, $H \ge 1$.

    Returns:
        The per-channel block transfer entropy in nats.

    Raises:
        ValueError: If ``p`` is outside $[0, 1]$, ``K < 2``, or ``H <= 0``.
    """
    if H <= 0:
        raise ValueError(f"te_categorical_switch_block: H must be > 0, got {H}.")
    return float(H * te_categorical_switch(p, K))


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
