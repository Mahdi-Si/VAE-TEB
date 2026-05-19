r"""Synthetic data-generating processes for the TE benchmarks (Phase 1 / 7).

Implements the source/target processes of ``model_validation.md`` Section 3.
Each generator is a **pure function** -- it returns in-memory tensors plus a
metadata dict and performs no disk I/O (persistence is ``build_dataset``'s job,
per Decision D7). Generated tensors use the model's native channel layout
(Decision D1): target $Y \in \mathbb{R}^{n \times T \times 87}$ and source
$U \in \mathbb{R}^{n \times T \times 101}$, with non-informative channels
filled by i.i.d. noise distractors.

Public API:
    gen_delayed_gaussian: Benchmark A (delayed linear-Gaussian); the
        ``reverse_roles`` flag produces the Benchmark G directionality variant.
    gen_ar_gaussian: Benchmark B (autoregressive target self-information).
    gen_delayed_xor: Benchmark C (delayed binary XOR).
    gen_two_lag_gaussian: Benchmark E (two-lag Gaussian).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Tuple

import torch

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    te_block_gaussian,
    te_block_xor,
)


def _standardize_per_channel(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Z-score every channel of ``x`` over the batch and time axes.

    Standardisation is a per-channel invertible affine map, so it leaves the
    transfer entropy between any pair of channels unchanged while putting all
    channels on a common $\mathcal{N}(0, 1)$ scale.

    Args:
        x: Tensor of shape $(n, T, C)$.
        eps: Small constant added to the standard deviation for stability.

    Returns:
        The per-channel standardised tensor, same shape and dtype as ``x``.
    """
    mean = x.mean(dim=(0, 1), keepdim=True)
    std = x.std(dim=(0, 1), keepdim=True)
    return (x - mean) / (std + eps)


def gen_delayed_gaussian(
    n: int,
    T: int,
    delay: int,
    a: float,
    sigma2: float,
    M: int,
    *,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    easy_variant: bool = False,
    standardize: bool = True,
    reverse_roles: bool = False,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the Benchmark A delayed linear-Gaussian source/target process.

    The data-generating process (``model_validation.md`` Section 3, Benchmark A)
    draws an i.i.d. Gaussian source and copies $M$ of its channels into the
    target through a fixed ``delay`` $D$ with additive Gaussian noise:

    $$
    Y_j(t) = a\,U_j(t - D) + \varepsilon_j(t), \qquad
    \varepsilon_j(t) \sim \mathcal{N}(0, \sigma^2), \qquad j = 1, \dots, M,
    $$

    for $t \ge D$; earlier target steps and all non-informative (distractor)
    channels are pure i.i.d. $\mathcal{N}(0, 1)$ noise. With $D \ge H$ every
    future target step of a valid anchor depends on an already-observed source
    value, so the block transfer entropy is known in closed form:
    $\mathrm{TE}^{(H)} = \tfrac{H}{2}\,M\,\ln(1 + a^2 / \sigma^2)$.

    **Directionality variant (Benchmark G).** With ``reverse_roles=True`` the
    *same* $X \to Y$ process is built, but the i.i.d. driver $X$ is returned in
    the 87-channel target slot and the dependent stream in the 101-channel
    source slot. The model then measures $\mathrm{TE}_{\text{source} \to
    \text{target}}$ on the *anti-causal* arrangement, whose true value is $0$
    (the i.i.d. target's future is unpredictable). ``te_true`` is set to $0$ and
    ``true_lag_band`` to the empty list.

    Args:
        n: Number of independent sequences (samples) to generate.
        T: Sequence length (decimated time steps).
        delay: Source-to-target delay $D$. Must satisfy $D \ge H$.
        a: Scalar transfer coefficient applied to every informative channel.
        sigma2: Target additive-noise variance $\sigma^2 > 0$.
        M: Number of informative channels. Ignored (forced to ``c_y``) when
            ``easy_variant`` is set.
        c_y: Target channel count (default 87, the model's native width).
        c_u: Source channel count (default 101, the model's native width).
        horizon: Forecast horizon $H$; sets ``true_lag_band`` and the
            block-TE horizon factor.
        easy_variant: If ``True`` make every target channel informative
            ($M = c_y$) -- the pipeline proof-of-life variant that avoids
            channel dilution.
        standardize: If ``True`` (default) z-score every channel to unit
            variance. TE-invariant; see :func:`_standardize_per_channel`.
        reverse_roles: If ``True`` produce the Benchmark G directionality
            variant (i.i.d. driver in the target slot, dependent stream in the
            source slot; ``te_true = 0``).
        seed: Seed for the CPU ``torch.Generator`` driving all randomness.

    Returns:
        A tuple ``(Y, U, meta)`` where ``Y`` is the target tensor of shape
        $(n, T, c_y)$, ``U`` the source tensor of shape $(n, T, c_u)$ (both
        ``float32``), and ``meta`` a dict of ground-truth metadata
        (``te_true``, ``true_lag_band``, ``informative_channels``,
        ``clean_anchor_range``, ``direction`` and the generator arguments).

    Raises:
        ValueError: If the arguments are inconsistent (non-positive sizes,
            ``delay < horizon``, ``sigma2 <= 0``, or ``M`` out of range).
    """
    if easy_variant:
        M = c_y
    if n <= 0 or T <= 0:
        raise ValueError(
            f"gen_delayed_gaussian: n and T must be > 0, got n={n}, T={T}."
        )
    if horizon <= 0:
        raise ValueError(
            f"gen_delayed_gaussian: horizon must be > 0, got {horizon}."
        )
    if delay < horizon:
        raise ValueError(
            f"gen_delayed_gaussian: Benchmark A requires delay >= horizon "
            f"(D >= H), got delay={delay}, horizon={horizon}."
        )
    if delay + 1 > T:
        raise ValueError(
            f"gen_delayed_gaussian: need delay + 1 <= T, got delay={delay}, T={T}."
        )
    if not 1 <= M <= min(c_y, c_u):
        raise ValueError(
            f"gen_delayed_gaussian: require 1 <= M <= min(c_y, c_u)="
            f"{min(c_y, c_u)}, got M={M}."
        )
    if sigma2 <= 0.0:
        raise ValueError(
            f"gen_delayed_gaussian: sigma2 must be > 0, got {sigma2}."
        )
    if delay > 91:
        warnings.warn(
            f"gen_delayed_gaussian: delay={delay} exceeds the model's attention "
            f"window (max_lag + 1 = 91); this is the intended A-wrong-lag setup.",
            stacklevel=2,
        )

    gen = torch.Generator().manual_seed(int(seed))
    sigma = math.sqrt(sigma2)

    if not reverse_roles:
        # --- forward: i.i.d. source U (c_u), dependent target Y (c_y) -------
        U = torch.randn(n, T, c_u, generator=gen, dtype=torch.float32)
        eps = sigma * torch.randn(n, T, M, generator=gen, dtype=torch.float32)
        y_inf = eps.clone()
        # Y_j(t) = a * U_j(t - D) + eps  for t >= D; copy from the source.
        y_inf[:, delay:, :] += a * U[:, : T - delay, :M]
        if M < c_y:
            y_dist = torch.randn(
                n, T, c_y - M, generator=gen, dtype=torch.float32
            )
            Y = torch.cat([y_inf, y_dist], dim=-1)
        else:
            Y = y_inf
        te_true = te_block_gaussian(a, sigma2, horizon, M)
        true_lag_band = list(range(delay - horizon, delay))
    else:
        # --- reverse (Benchmark G): i.i.d. driver in the 87-ch target slot,
        # dependent stream in the 101-ch source slot -> model-slot TE is 0. ---
        Y = torch.randn(n, T, c_y, generator=gen, dtype=torch.float32)
        eps = sigma * torch.randn(n, T, M, generator=gen, dtype=torch.float32)
        u_inf = eps.clone()
        u_inf[:, delay:, :] += a * Y[:, : T - delay, :M]
        if M < c_u:
            u_dist = torch.randn(
                n, T, c_u - M, generator=gen, dtype=torch.float32
            )
            U = torch.cat([u_inf, u_dist], dim=-1)
        else:
            U = u_inf
        te_true = 0.0
        true_lag_band = []

    if standardize:
        Y = _standardize_per_channel(Y)
        U = _standardize_per_channel(U)

    meta: Dict[str, Any] = {
        "benchmark": "G" if reverse_roles else "A",
        "te_true": te_true,
        "te_per_step": te_true / horizon,
        "true_lag_band": true_lag_band,
        "informative_channels": list(range(M)),
        "clean_anchor_range": [delay - 1, T - horizon],
        "delay": delay,
        "a": float(a),
        "sigma2": float(sigma2),
        "M": M,
        "horizon": horizon,
        "sequence_length": T,
        "c_y": c_y,
        "c_u": c_u,
        "easy_variant": easy_variant,
        "standardized": standardize,
        "reverse_roles": reverse_roles,
        "direction": "Y_to_X" if reverse_roles else "X_to_Y",
        "seed": seed,
    }
    return Y, U, meta


def gen_ar_gaussian(
    n: int,
    T: int,
    delay: int,
    a: float,
    sigma2: float,
    M: int,
    rho: float,
    *,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    burn_in: int = 200,
    easy_variant: bool = False,
    standardize: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the Benchmark B autoregressive-target delayed-Gaussian process.

    The target carries strong self-predictability through an AR(1) recursion on
    top of the delayed i.i.d. source drive (``model_validation.md`` Section 3,
    Benchmark B):

    $$
    Y_j(t) = \rho\,Y_j(t - 1) + a\,U_j(t - D) + \varepsilon_j(t), \qquad
    \varepsilon_j(t) \sim \mathcal{N}(0, \sigma^2),
    $$

    for the $M$ informative channels; distractor channels are i.i.d.
    $\mathcal{N}(0, 1)$. The recursion is run for $T + \texttt{burn\_in}$ steps
    and the leading ``burn_in`` steps are discarded so the kept window is
    AR-stationary.

    **Key fact (block-TE $\rho$-independence).** Because the source $X$ is
    i.i.d. and $D \ge H$, the AR self-term $\rho\,Y_j(t-1)$ is a deterministic
    function of the target history $Y_{\le t}$ and therefore *cancels* in the
    block-TE determinant ratio. The exact block transfer entropy is hence
    identical to Benchmark A's and **independent of $\rho$**:
    $\mathrm{TE}^{(H)} = \tfrac{H}{2}\,M\,\ln(1 + a^2 / \sigma^2)$. The function
    :func:`analytic_te.te_block_gaussian_mc` numerically cross-checks this.

    Args:
        n: Number of independent sequences (samples) to generate.
        T: Sequence length (decimated time steps), excluding the burn-in.
        delay: Source-to-target delay $D$. Must satisfy $D \ge H$.
        a: Scalar transfer coefficient applied to every informative channel.
        sigma2: Innovation variance $\sigma^2 > 0$.
        M: Number of informative channels. Ignored (forced to ``c_y``) when
            ``easy_variant`` is set.
        rho: AR(1) self-coefficient, $0 \le \rho < 1$ for stationarity.
        c_y: Target channel count (default 87).
        c_u: Source channel count (default 101).
        horizon: Forecast horizon $H$.
        burn_in: Number of leading steps generated and discarded so the kept
            window is free of the $\mathcal{O}(1/(1-\rho))$ AR transient.
        easy_variant: If ``True`` make every target channel informative.
        standardize: If ``True`` (default) z-score every channel.
        seed: Seed for the CPU ``torch.Generator``.

    Returns:
        A tuple ``(Y, U, meta)`` -- ``Y`` of shape $(n, T, c_y)$, ``U`` of shape
        $(n, T, c_u)$ (both ``float32``), and the ground-truth ``meta`` dict
        (``benchmark="B"``, plus ``rho`` and ``burn_in``).

    Raises:
        ValueError: If the arguments are inconsistent or ``rho`` is outside
            $[0, 1)$.
    """
    if easy_variant:
        M = c_y
    if n <= 0 or T <= 0:
        raise ValueError(f"gen_ar_gaussian: n and T must be > 0, got n={n}, T={T}.")
    if horizon <= 0:
        raise ValueError(f"gen_ar_gaussian: horizon must be > 0, got {horizon}.")
    if delay < horizon:
        raise ValueError(
            f"gen_ar_gaussian: requires delay >= horizon, got delay={delay}, "
            f"horizon={horizon}."
        )
    if delay + 1 > T:
        raise ValueError(
            f"gen_ar_gaussian: need delay + 1 <= T, got delay={delay}, T={T}."
        )
    if not 1 <= M <= min(c_y, c_u):
        raise ValueError(
            f"gen_ar_gaussian: require 1 <= M <= min(c_y, c_u)="
            f"{min(c_y, c_u)}, got M={M}."
        )
    if sigma2 <= 0.0:
        raise ValueError(f"gen_ar_gaussian: sigma2 must be > 0, got {sigma2}.")
    if not 0.0 <= rho < 1.0:
        raise ValueError(
            f"gen_ar_gaussian: rho must satisfy 0 <= rho < 1 for stationarity, "
            f"got rho={rho}."
        )
    if burn_in < 0:
        raise ValueError(f"gen_ar_gaussian: burn_in must be >= 0, got {burn_in}.")
    if delay > 91:
        warnings.warn(
            f"gen_ar_gaussian: delay={delay} exceeds the model's attention "
            f"window (max_lag + 1 = 91); this is the intended wrong-lag setup.",
            stacklevel=2,
        )

    gen = torch.Generator().manual_seed(int(seed))
    sigma = math.sqrt(sigma2)
    length = T + burn_in

    # --- i.i.d. source over the full (burn-in + kept) length ----------------
    u_full = torch.randn(n, length, c_u, generator=gen, dtype=torch.float32)
    eps = sigma * torch.randn(n, length, M, generator=gen, dtype=torch.float32)

    # Source drive s_j(t) = a * U_j(t - D) (zero for t < D).
    src = torch.zeros(n, length, M, dtype=torch.float32)
    src[:, delay:, :] = a * u_full[:, : length - delay, :M]

    # AR(1) recursion over the informative channels.
    y_inf = torch.zeros(n, length, M, dtype=torch.float32)
    y_inf[:, 0, :] = src[:, 0, :] + eps[:, 0, :]
    for t in range(1, length):
        y_inf[:, t, :] = rho * y_inf[:, t - 1, :] + src[:, t, :] + eps[:, t, :]

    # Discard the burn-in prefix from both streams.
    y_inf = y_inf[:, burn_in:, :].contiguous()
    U = u_full[:, burn_in:, :].contiguous()

    if M < c_y:
        y_dist = torch.randn(n, T, c_y - M, generator=gen, dtype=torch.float32)
        Y = torch.cat([y_inf, y_dist], dim=-1)
    else:
        Y = y_inf

    if standardize:
        Y = _standardize_per_channel(Y)
        U = _standardize_per_channel(U)

    te_true = te_block_gaussian(a, sigma2, horizon, M)
    meta: Dict[str, Any] = {
        "benchmark": "B",
        "te_true": te_true,
        "te_per_step": te_true / horizon,
        "true_lag_band": list(range(delay - horizon, delay)),
        "informative_channels": list(range(M)),
        "clean_anchor_range": [delay - 1, T - horizon],
        "delay": delay,
        "a": float(a),
        "sigma2": float(sigma2),
        "M": M,
        "rho": float(rho),
        "burn_in": int(burn_in),
        "horizon": horizon,
        "sequence_length": T,
        "c_y": c_y,
        "c_u": c_u,
        "easy_variant": easy_variant,
        "standardized": standardize,
        "direction": "X_to_Y",
        "seed": seed,
    }
    return Y, U, meta


def gen_delayed_xor(
    n: int,
    T: int,
    delay: int,
    q: float,
    M: int,
    *,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    obs_noise: float = 0.1,
    easy_variant: bool = False,
    standardize: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the Benchmark C delayed binary-XOR source/target process.

    A binary source is copied into the target through a fixed ``delay`` $D$
    with bit-flip (XOR) noise (``model_validation.md`` Section 3, Benchmark C):

    $$
    X_j(t) \sim \mathrm{Bernoulli}(\tfrac12), \qquad
    Y_j(t) = X_j(t - D) \oplus E_j(t), \qquad
    E_j(t) \sim \mathrm{Bernoulli}(q),
    $$

    for the $M$ informative channels and $t \ge D$. Distractor channels and
    pre-delay target steps are independent $\mathrm{Bernoulli}(\tfrac12)$. Bits
    are embedded as $\pm 1$ with additive Gaussian observation noise so the
    model consumes continuous tensors while the analytic ground truth stays
    that of the *discrete* process:
    $\mathrm{TE}^{(H)} = M\,H\,(\ln 2 - h_b(q))$.

    Args:
        n: Number of independent sequences (samples) to generate.
        T: Sequence length (decimated time steps).
        delay: Source-to-target delay $D$. Must satisfy $D \ge H$.
        q: Bit-flip probability in $[0, 1]$.
        M: Number of informative channels. Ignored (forced to ``c_y``) when
            ``easy_variant`` is set.
        c_y: Target channel count (default 87).
        c_u: Source channel count (default 101).
        horizon: Forecast horizon $H$.
        obs_noise: Standard deviation of the Gaussian observation noise added
            to the $\pm 1$ embedding. The discrete-process ``te_true`` is
            unaffected; only the model's task realism changes.
        easy_variant: If ``True`` make every target channel informative.
        standardize: If ``True`` (default) z-score every channel.
        seed: Seed for the CPU ``torch.Generator``.

    Returns:
        A tuple ``(Y, U, meta)`` -- ``Y`` of shape $(n, T, c_y)$, ``U`` of shape
        $(n, T, c_u)$ (both ``float32``), and the ground-truth ``meta`` dict
        (``benchmark="C"``, with ``q`` / ``obs_noise`` and **no** ``a`` /
        ``sigma2``).

    Raises:
        ValueError: If the arguments are inconsistent or ``q`` is outside
            $[0, 1]$.
    """
    if easy_variant:
        M = c_y
    if n <= 0 or T <= 0:
        raise ValueError(f"gen_delayed_xor: n and T must be > 0, got n={n}, T={T}.")
    if horizon <= 0:
        raise ValueError(f"gen_delayed_xor: horizon must be > 0, got {horizon}.")
    if delay < horizon:
        raise ValueError(
            f"gen_delayed_xor: requires delay >= horizon, got delay={delay}, "
            f"horizon={horizon}."
        )
    if delay + 1 > T:
        raise ValueError(
            f"gen_delayed_xor: need delay + 1 <= T, got delay={delay}, T={T}."
        )
    if not 1 <= M <= min(c_y, c_u):
        raise ValueError(
            f"gen_delayed_xor: require 1 <= M <= min(c_y, c_u)="
            f"{min(c_y, c_u)}, got M={M}."
        )
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"gen_delayed_xor: q must lie in [0, 1], got {q}.")
    if obs_noise < 0.0:
        raise ValueError(f"gen_delayed_xor: obs_noise must be >= 0, got {obs_noise}.")
    if delay > 91:
        warnings.warn(
            f"gen_delayed_xor: delay={delay} exceeds the model's attention "
            f"window (max_lag + 1 = 91); this is the intended wrong-lag setup.",
            stacklevel=2,
        )

    gen = torch.Generator().manual_seed(int(seed))

    # --- source bits: every channel i.i.d. Bernoulli(1/2) ------------------
    x_bit = (
        torch.rand(n, T, c_u, generator=gen) < 0.5
    ).to(torch.float32)

    # --- informative target bits: Y_bit(t) = X_bit(t - D) XOR E(t) ---------
    e_flip = (torch.rand(n, T, M, generator=gen) < q).to(torch.float32)
    y_bit_inf = torch.empty(n, T, M, dtype=torch.float32)
    # XOR of {0,1} values via modulo-2 addition.
    y_bit_inf[:, delay:, :] = (
        x_bit[:, : T - delay, :M] + e_flip[:, delay:, :]
    ).remainder(2.0)
    # Pre-delay steps carry no transfer -> independent Bernoulli(1/2).
    y_bit_inf[:, :delay, :] = (
        torch.rand(n, delay, M, generator=gen) < 0.5
    ).to(torch.float32)

    if M < c_y:
        y_bit_dist = (
            torch.rand(n, T, c_y - M, generator=gen) < 0.5
        ).to(torch.float32)
        y_bit = torch.cat([y_bit_inf, y_bit_dist], dim=-1)
    else:
        y_bit = y_bit_inf

    # --- embed bits as +/-1 with Gaussian observation noise ----------------
    Y = (2.0 * y_bit - 1.0) + obs_noise * torch.randn(
        n, T, c_y, generator=gen, dtype=torch.float32
    )
    U = (2.0 * x_bit - 1.0) + obs_noise * torch.randn(
        n, T, c_u, generator=gen, dtype=torch.float32
    )

    if standardize:
        Y = _standardize_per_channel(Y)
        U = _standardize_per_channel(U)

    te_true = te_block_xor(q, horizon, M)
    meta: Dict[str, Any] = {
        "benchmark": "C",
        "te_true": te_true,
        "te_per_step": te_true / horizon,
        "true_lag_band": list(range(delay - horizon, delay)),
        "informative_channels": list(range(M)),
        "clean_anchor_range": [delay - 1, T - horizon],
        "delay": delay,
        "q": float(q),
        "obs_noise": float(obs_noise),
        "M": M,
        "horizon": horizon,
        "sequence_length": T,
        "c_y": c_y,
        "c_u": c_u,
        "easy_variant": easy_variant,
        "standardized": standardize,
        "direction": "X_to_Y",
        "seed": seed,
    }
    return Y, U, meta


def gen_two_lag_gaussian(
    n: int,
    T: int,
    delay1: int,
    delay2: int,
    a1: float,
    a2: float,
    sigma2: float,
    M1: int,
    M2: int,
    *,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    standardize: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the Benchmark E two-lag Gaussian source/target process.

    Two *independent* groups of source channels transfer into the target at two
    distinct delays (``model_validation.md`` Section 3, Benchmark E). Group 1
    occupies channels $[0, M_1)$ and transfers at delay $D_1$; group 2 occupies
    channels $[M_1, M_1 + M_2)$ and transfers at delay $D_2$:

    $$
    Y_j(t) = a_1\,U_j(t - D_1) + \varepsilon_j(t), \quad j \in [0, M_1), \qquad
    Y_j(t) = a_2\,U_j(t - D_2) + \varepsilon_j(t), \quad j \in [M_1, M_1+M_2).
    $$

    Because the two groups use disjoint source channels, the block transfer
    entropy decomposes additively:
    $\mathrm{TE}^{(H)} = \tfrac{H}{2} M_1 \ln(1 + a_1^2/\sigma^2)
    + \tfrac{H}{2} M_2 \ln(1 + a_2^2/\sigma^2)$, and the lag map should resolve
    two separate bands whose mass ratio tracks the per-band TE ratio.

    Args:
        n: Number of independent sequences (samples) to generate.
        T: Sequence length (decimated time steps).
        delay1: Group-1 source-to-target delay $D_1$. Must satisfy $D_1 \ge H$.
        delay2: Group-2 source-to-target delay $D_2$. Must satisfy $D_2 \ge H$
            and $D_2 \ne D_1$.
        a1: Group-1 transfer coefficient.
        a2: Group-2 transfer coefficient.
        sigma2: Target additive-noise variance $\sigma^2 > 0$.
        M1: Number of group-1 informative channels, $M_1 \ge 1$.
        M2: Number of group-2 informative channels, $M_2 \ge 1$.
        c_y: Target channel count (default 87).
        c_u: Source channel count (default 101).
        horizon: Forecast horizon $H$.
        standardize: If ``True`` (default) z-score every channel.
        seed: Seed for the CPU ``torch.Generator``.

    Returns:
        A tuple ``(Y, U, meta)`` -- ``Y`` of shape $(n, T, c_y)$, ``U`` of shape
        $(n, T, c_u)$ (both ``float32``), and the ground-truth ``meta`` dict
        (``benchmark="E"``, with per-band ``lag_band_1`` / ``lag_band_2`` and
        ``te_true_1`` / ``te_true_2``).

    Raises:
        ValueError: If the arguments are inconsistent (equal delays, delays
            below the horizon, or $M_1 + M_2$ out of range).
    """
    if n <= 0 or T <= 0:
        raise ValueError(
            f"gen_two_lag_gaussian: n and T must be > 0, got n={n}, T={T}."
        )
    if horizon <= 0:
        raise ValueError(
            f"gen_two_lag_gaussian: horizon must be > 0, got {horizon}."
        )
    if delay1 < horizon or delay2 < horizon:
        raise ValueError(
            f"gen_two_lag_gaussian: both delays must be >= horizon, got "
            f"delay1={delay1}, delay2={delay2}, horizon={horizon}."
        )
    if delay1 == delay2:
        raise ValueError(
            f"gen_two_lag_gaussian: delay1 and delay2 must differ, both "
            f"equal {delay1}."
        )
    if max(delay1, delay2) + 1 > T:
        raise ValueError(
            f"gen_two_lag_gaussian: need max(delay1, delay2) + 1 <= T, got "
            f"delays=({delay1}, {delay2}), T={T}."
        )
    if M1 < 1 or M2 < 1:
        raise ValueError(
            f"gen_two_lag_gaussian: M1 and M2 must be >= 1, got M1={M1}, M2={M2}."
        )
    if M1 + M2 > min(c_y, c_u):
        raise ValueError(
            f"gen_two_lag_gaussian: require M1 + M2 <= min(c_y, c_u)="
            f"{min(c_y, c_u)}, got M1+M2={M1 + M2}."
        )
    if sigma2 <= 0.0:
        raise ValueError(
            f"gen_two_lag_gaussian: sigma2 must be > 0, got {sigma2}."
        )
    if max(delay1, delay2) > 91:
        warnings.warn(
            f"gen_two_lag_gaussian: max delay {max(delay1, delay2)} exceeds the "
            f"model's attention window (max_lag + 1 = 91).",
            stacklevel=2,
        )

    gen = torch.Generator().manual_seed(int(seed))
    sigma = math.sqrt(sigma2)
    m_total = M1 + M2

    # --- i.i.d. source ------------------------------------------------------
    U = torch.randn(n, T, c_u, generator=gen, dtype=torch.float32)

    eps = sigma * torch.randn(n, T, m_total, generator=gen, dtype=torch.float32)
    y_inf = eps.clone()
    # Group 1: channels [0, M1) <- source channels [0, M1) at delay1.
    y_inf[:, delay1:, :M1] += a1 * U[:, : T - delay1, :M1]
    # Group 2: channels [M1, M1+M2) <- source channels [M1, M1+M2) at delay2.
    y_inf[:, delay2:, M1:m_total] += a2 * U[:, : T - delay2, M1:m_total]

    if m_total < c_y:
        y_dist = torch.randn(
            n, T, c_y - m_total, generator=gen, dtype=torch.float32
        )
        Y = torch.cat([y_inf, y_dist], dim=-1)
    else:
        Y = y_inf

    if standardize:
        Y = _standardize_per_channel(Y)
        U = _standardize_per_channel(U)

    te_true_1 = te_block_gaussian(a1, sigma2, horizon, M1)
    te_true_2 = te_block_gaussian(a2, sigma2, horizon, M2)
    te_true = te_true_1 + te_true_2
    lag_band_1 = list(range(delay1 - horizon, delay1))
    lag_band_2 = list(range(delay2 - horizon, delay2))
    true_lag_band = sorted(set(lag_band_1) | set(lag_band_2))

    meta: Dict[str, Any] = {
        "benchmark": "E",
        "te_true": te_true,
        "te_per_step": te_true / horizon,
        "te_true_1": te_true_1,
        "te_true_2": te_true_2,
        "true_lag_band": true_lag_band,
        "lag_band_1": lag_band_1,
        "lag_band_2": lag_band_2,
        "informative_channels": list(range(m_total)),
        "informative_channels_1": list(range(M1)),
        "informative_channels_2": list(range(M1, m_total)),
        "clean_anchor_range": [max(delay1, delay2) - 1, T - horizon],
        "delay1": delay1,
        "delay2": delay2,
        "a1": float(a1),
        "a2": float(a2),
        "sigma2": float(sigma2),
        "M1": M1,
        "M2": M2,
        "M": m_total,
        "horizon": horizon,
        "sequence_length": T,
        "c_y": c_y,
        "c_u": c_u,
        "standardized": standardize,
        "direction": "X_to_Y",
        "seed": seed,
    }
    return Y, U, meta


if __name__ == "__main__":
    # Self-check -- shapes, metadata, determinism, and the null cases for every
    # Phase-1/Phase-7 generator.

    # --- Benchmark A (delayed linear-Gaussian) -----------------------------
    _Y, _U, _meta = gen_delayed_gaussian(
        n=8, T=300, delay=60, a=1.0, sigma2=1.0, M=4, seed=0
    )
    assert _Y.shape == (8, 300, 87) and _U.shape == (8, 300, 101)
    assert _Y.dtype == torch.float32 and _U.dtype == torch.float32
    _Y2, _U2, _ = gen_delayed_gaussian(
        n=8, T=300, delay=60, a=1.0, sigma2=1.0, M=4, seed=0
    )
    assert torch.equal(_Y, _Y2) and torch.equal(_U, _U2), "A non-deterministic"
    _, _, _m0 = gen_delayed_gaussian(
        n=8, T=300, delay=60, a=0.0, sigma2=1.0, M=4, seed=0
    )
    assert _m0["te_true"] == 0.0
    print(f"[A] shapes/determinism/null OK  te_true={_meta['te_true']:.4f}")

    # --- Benchmark G (reverse-roles directionality) ------------------------
    _Yg, _Ug, _mg = gen_delayed_gaussian(
        n=8, T=300, delay=60, a=1.0, sigma2=1.0, M=4, reverse_roles=True, seed=0
    )
    assert _Yg.shape == (8, 300, 87) and _Ug.shape == (8, 300, 101)
    assert _mg["te_true"] == 0.0 and _mg["true_lag_band"] == []
    assert _mg["benchmark"] == "G" and _mg["direction"] == "Y_to_X"
    print("[G] reverse-roles te_true=0, empty lag band OK")

    # --- Benchmark B (AR target) -- te_true is rho-independent -------------
    _Yb, _Ub, _mb = gen_ar_gaussian(
        n=8, T=300, delay=60, a=0.5, sigma2=1.0, M=4, rho=0.9, seed=0
    )
    assert _Yb.shape == (8, 300, 87) and _Ub.shape == (8, 300, 101)
    _te_rho = {
        r: gen_ar_gaussian(
            n=4, T=200, delay=40, a=0.5, sigma2=1.0, M=4, rho=r, seed=0
        )[2]["te_true"]
        for r in (0.0, 0.5, 0.9, 0.99)
    }
    assert max(_te_rho.values()) - min(_te_rho.values()) < 1e-9, _te_rho
    assert abs(_mb["te_true"] - te_block_gaussian(0.5, 1.0, 30, 4)) < 1e-9
    print(f"[B] shapes OK  te_true rho-independent ({_mb['te_true']:.4f})")

    # --- Benchmark C (delayed XOR) -----------------------------------------
    _Yc, _Uc, _mc = gen_delayed_xor(n=8, T=300, delay=60, q=0.10, M=4, seed=0)
    assert _Yc.shape == (8, 300, 87) and _Uc.shape == (8, 300, 101)
    assert abs(_mc["te_true"] - te_block_xor(0.10, 30, 4)) < 1e-9
    _, _, _mc_null = gen_delayed_xor(n=4, T=200, delay=40, q=0.5, M=4, seed=0)
    assert abs(_mc_null["te_true"]) < 1e-9, _mc_null["te_true"]
    # Bit-agreement on raw (unstandardised) data: P(round(Y)==round(X(t-D)))~=1-q.
    _Yr, _Ur, _mr = gen_delayed_xor(
        n=400, T=200, delay=40, q=0.10, M=4, obs_noise=0.1,
        standardize=False, seed=0,
    )
    _ybit = (_Yr[:, 40:, 0] > 0).float()
    _xbit = (_Ur[:, :160, 0] > 0).float()
    _agree = (_ybit == _xbit).float().mean().item()
    assert abs(_agree - 0.90) < 0.03, _agree
    print(f"[C] shapes/null OK  bit-agreement={_agree:.3f} (expect ~0.90)")

    # --- Benchmark E (two-lag Gaussian) ------------------------------------
    _Ye, _Ue, _me = gen_two_lag_gaussian(
        n=8, T=300, delay1=50, delay2=80, a1=0.4, a2=0.25,
        sigma2=1.0, M1=4, M2=4, seed=0,
    )
    assert _Ye.shape == (8, 300, 87) and _Ue.shape == (8, 300, 101)
    _add = te_block_gaussian(0.4, 1.0, 30, 4) + te_block_gaussian(0.25, 1.0, 30, 4)
    assert abs(_me["te_true"] - _add) < 1e-9
    assert _me["lag_band_1"] == list(range(20, 50))
    assert _me["lag_band_2"] == list(range(50, 80))
    print(f"[E] shapes OK  te_true={_me['te_true']:.4f} additive  two bands OK")

    print("All generator checks passed.")
