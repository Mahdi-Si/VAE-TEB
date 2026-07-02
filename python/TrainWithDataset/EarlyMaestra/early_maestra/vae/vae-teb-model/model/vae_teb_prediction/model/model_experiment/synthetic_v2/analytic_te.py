r"""Ground-truth block transfer-entropy for ``synthetic_v2`` (single pathway).

Ported from ``synthetic/analytic_te.py`` and trimmed to the AR(2)-oscillator
state-space family that ``synthetic_v2`` needs. ``synthetic_v2`` couples exactly
**one** UP-contraction source to **one** FHR-deceleration target, so every call
here uses a single oscillator ($M = 1$); the functions keep their general
length-$M$ signatures (the determinant-ratio machinery is byte-for-byte the v1
math) but there is **no** $M$ grid / dilution axis anywhere in v2. "Single
pathway" therefore means: v2 always passes a one-element ``oscillators`` list
and receives a scalar coupling $B$.

All functions are pure (``numpy`` / ``math`` only; no ``torch``, no I/O) and
return values in **nats**.

Ported API:
    snr_per_step_for_te_block: Per-step, per-channel innovation SNR implied by a
        block TE — the extractability gauge $\mathrm{SNR} \approx
        e^{2\,\mathrm{TE}^{(H)}/(H M)} - 1$.
    te_block_state_space_gaussian: Monte-Carlo determinant-ratio block TE for the
        AR(2)-oscillator-driven Gaussian state-space (the exact latent label
        $\mathrm{TE}_{\mathrm{inj}}$).
    mean_te_block_state_space_over_delays: Delay-averaged block TE. A fixed lag is
        the degenerate range $d_{\min} = d_{\max} = D$; the band-averaging form is
        retained so a future ``band`` lag mode can be added without new math.
    B_y_for_mean_te_block_state_space: The inverter — bisects the uniform coupling
        magnitude $B$ so the (mean) block TE hits a requested ``target_te_block``.
    realizable_te_block_from_arrays: The R0 probe — block TE realizable by a
        finite ridge predictor on cached, z-scored feature arrays (the
        $\mathrm{TE}_{\mathrm{scat}}$ estimator of §14.3).

See ``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` (§6, §8, §9, §14.3) for the
derivations. Unit tests: ``synthetic_v2/tests/test_analytic_te.py``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

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
    r"""Simulate the AR(2)-oscillator-driven Gaussian latent state-space.

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

    with $\sigma^2_y$ shared across channels. In ``synthetic_v2`` this is the
    decimated-grid latent pair $(c_k, d_k)$ of §6 with $M = 1$: the single
    source is the contraction-strength latent $c_k$ and the single target is the
    deceleration-depth latent $d_k$.

    Args:
        n: Number of independent sequences (samples).
        T: Sequence length (post-burn-in).
        oscillators: Length-$M$ list of $(r_m, \omega_m)$ pairs (one entry in v2).
        target_ar: Scalar target self-coefficient $A_y \in [0, 1)$.
        delays: Source-to-target delays $D_m$. One of three shapes: a
            length-$M$ sequence (one delay per channel, shared across all $n$
            samples — the fast path used by the analytic TE); an $(n, M)$
            integer array of **per-sample, per-channel** delays; or an
            $(n, T_{\text{total}}, M)$ integer array of **per-sample, per-time,
            per-channel** delays. $T_{\text{total}} = \text{burn\_in} + T$.
            Every entry must be $\ge 1$.
        B_y: Length-$M$ list of target loadings $B_y^{(m)}$.
        sigma2_y: Target innovation variance $\sigma^2_y > 0$ (shared).
        sigma2_eta: Source innovation variance — scalar (broadcast) or
            length-$M$ array of positive floats.
        burn_in: Warm-up steps discarded so the kept window is stationary.
        seed: Seed for the NumPy random generator.

    Returns:
        A tuple ``(S, Y)`` of ``np.ndarray`` of shape ``(n, T, M)`` — the
        oscillator source stream ``S`` and the coupled target stream ``Y``.
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
    r"""Monte-Carlo block transfer entropy for the oscillator state-space.

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
    the mean; raise it if needed.

    Args:
        oscillators, target_ar, delays, B_y, sigma2_y, sigma2_eta,
        burn_in, seed: As in :func:`_simulate_state_space_gaussian`.
        H: Forecast horizon, $H \ge 1$.
        K_history: History depth used for $Y^{-}$ and $U^{-}$. Defaults to
            ``max(delays) + 2H``.
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
    unextractable at the usual sample sizes. In ``synthetic_v2`` the single
    pathway means $M = 1$, so ``snr_per_step_for_te_block(2.0, 30, 1) ≈ 0.143``.

    Args:
        te_block: Block transfer entropy in nats ($\ge 0$).
        H: Forecast horizon ($\ge 1$).
        M: Informative-channel count ($\ge 1$; $= 1$ in v2).

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
    source channels ``U[:, :, :M]`` are the columns the analytic TE conditions on.
    In ``synthetic_v2`` this is the $\mathrm{TE}_{\mathrm{scat}}$ estimator of
    §14.3: the caller slices the fs-correct coupled decel/contraction pulse-shape
    scattering channels into single-channel arrays (``M = 1``) before calling in.

    Comparing this to the analytic block TE answers the realizability question
    that gates the experiment: a ratio $\to 1$ means an optimal-ish predictor
    *can* extract the TE at this sample size (so a downstream calibration failure
    is a model/training problem); a ratio $\ll 1$ means the TE is *not
    extractable* at this $n$ (a data/transform-design problem).

    Args:
        Y: Target cache array of shape $(n, T, C_y)$ (z-scored).
        U: Source cache array of shape $(n, T, C_u)$; channels $[0, M)$ are the
            informative (coupled) columns.
        M: Informative-channel count of the cell ($= 1$ in v2).
        K: History depth for the target self-history $Y^{-}$ (use the cell's
            ``K_history``).
        H: Forecast horizon.
        delay_max: Upper lag of the cell's band; the source regressors $U^{-}$
            are scoped to lags $0..\,$``delay_max``. ``None`` uses the full $K$
            source lags (matches the analytic support but is noisier at finite
            $n$).
        anchor: Anchor index $t_0$ (history $[t_0-K, t_0)$, future
            $[t_0, t_0+H)$). Defaults to a mid-sequence position.
        ridge: $L_2$ penalty added to the regressor Gram diagonal (relative to
            its mean), shared across the reduced and full fits.
        train_frac: Fraction of samples used to fit; the remainder evaluates the
            residual covariance.
        seed: Seed for the train/test row shuffle.

    Returns:
        Dict with ``realizable_gain`` (nats, block), ``snr_per_step``,
        ``m_used``, ``anchor``, ``n_train``, ``n_test`` and ``ill_conditioned``
        (``True`` with ``realizable_gain = nan`` when the held-out set is too
        small to estimate the $HM \times HM$ residual covariance, i.e.
        ``n_test <= H * m_used``).
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
    below ``tol``; the loop is capped at ``max_iter``.

    Args:
        eval_fn: Maps a coupling magnitude :math:`x` to its block TE in nats.
        target_te_block: Target block TE in nats; must satisfy
            :math:`\mathrm{eval\_fn}(\mathrm{lo}) < \mathrm{target} <
            \mathrm{eval\_fn}(\mathrm{hi})`.
        lo, hi: Bracket on the coupling magnitude.
        tol, max_iter: Stop conditions.
        label: Prefix used in the ``ValueError`` raised by the bracket check.

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


def _validate_delay_range(delay_min: int, delay_max: int, label: str) -> None:
    r"""Validate an inclusive integer delay range $\{d_{\min},\dots,d_{\max}\}$.

    A fixed lag $D$ is the degenerate range $d_{\min} = d_{\max} = D$.

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
    r"""Mean block TE of the oscillator state-space over a delay range.

    For a fixed lag $D$ (``synthetic_v2`` default) pass
    $d_{\min} = d_{\max} = D$ and this reduces to a single
    :func:`te_block_state_space_gaussian` evaluation at ``delays = [D] * M``. The
    band form averages the per-delay block TE over a uniform delay distribution,

    $$
    \overline{\mathrm{TE}}
        = \frac{1}{d_{\max}-d_{\min}+1}
          \sum_{d=d_{\min}}^{d_{\max}}
          \widehat{\mathrm{TE}}^{(H)}_{U\to Y}\!\bigl(\text{delays}=[d]\!*\!M\bigr),
    $$

    and is retained so a future ``band`` lag mode can be added without new math.

    Args:
        delay_min: Smallest delay in the range, $\ge 1$.
        delay_max: Largest delay in the range (inclusive), $\ge d_{\min}$.
        oscillators: Length-$M$ list of $(r_m, \omega_m)$ oscillator specs (one
            entry in v2).
        target_ar: Target self-coefficient $A_y \in [0, 1)$.
        B_y: Uniform target loading magnitude (scalar) or length-$M$ list.
        sigma2_y: Target innovation variance $\sigma^2_y > 0$.
        sigma2_eta: Source innovation variance (scalar or length-$M$).
        H: Forecast horizon, $H \ge 1$.
        K_history: History depth forwarded to
            :func:`te_block_state_space_gaussian`; defaults to ``max(delays)+2H``.
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
    r"""Invert the (mean-over-delays) block TE for a uniform $B_y$ magnitude.

    Bisects the uniform target-loading magnitude $b$ so that
    :func:`mean_te_block_state_space_over_delays` (with ``B_y = b``) equals
    ``target_te_block``. The mean of monotone-increasing MC TE estimators is
    itself monotone increasing in $|b|$, so bisection converges; the MC seed is
    held fixed across iterations so bisection does not chase noise. For a fixed
    lag $D$ (``synthetic_v2`` default) pass $d_{\min} = d_{\max} = D$; the solved
    $B$ is the exact injected-TE label $\mathrm{TE}_{\mathrm{inj}}$ of the cell.

    Args:
        target_te_block: Target (mean) block TE in nats, $\ge 0$. ``0`` returns
            immediately with $B_y = 0$ (a null cell).
        delay_min, delay_max, oscillators, target_ar, sigma2_y, sigma2_eta,
        H, K_history, n_samples, burn_in, seed: Forwarded to
            :func:`mean_te_block_state_space_over_delays`.
        lo, hi: Initial bisection bracket on the uniform $B_y$ magnitude. In v2
            the config widens ``hi`` to ``60`` so the higher single-pathway
            couplings ($M = 1$) at ``target_te`` up to 3 nats stay in-bracket.
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
