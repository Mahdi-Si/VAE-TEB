r"""Synthetic data-generating processes for the v2 TE benchmarks.

Implements the source/target processes of ``model_validation_v2.md`` Sections
3-5. Each generator is a **pure function** -- it returns in-memory tensors
plus a metadata dict and performs no disk I/O (persistence is
:mod:`build_dataset`'s job). Generated tensors use the model's native
channel layout: target $Y \in \mathbb{R}^{n \times T \times 87}$ and source
$U \in \mathbb{R}^{n \times T \times 101}$, with non-informative channels
filled by i.i.d. $\mathcal{N}(0, 1)$ distractor noise.

Public API (v2):
    gen_state_space_oscillator: Benchmark G1 -- AR(2)-driven state-space
        oscillator. ``reverse_roles=True`` produces the G1-rev directionality
        variant ($\bar K_{Y \to X}$ control).
    gen_smooth_arx: Benchmark G2 -- smooth AR(1)-ARX, the debugging stepping
        stone with a closed-form analytic TE.
    gen_regime_switch_smooth: Benchmark G3 -- slow categorical regime switch
        with a phase-continuous oscillator target and per-channel one-hot
        source encoding.

G4 (switched sinusoid) is deferred to Sprint 7 per ``model_validation_v2_plan
.md`` decision V2-D9.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    _simulate_state_space_gaussian,
    te_block_arx_gaussian,
    te_block_state_space_gaussian,
    te_categorical_switch_block,
)


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------


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


def _place_into_buffers(
    informative_target: torch.Tensor,
    informative_source: torch.Tensor,
    *,
    M: int,
    c_y: int,
    c_u: int,
    reverse_roles: bool,
    distractor_generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Stitch informative streams into the model's native channel layout.

    Places the $M$ informative streams in target channels $[0, M)$ and source
    channels $[0, M)$; the remaining channels are i.i.d. $\mathcal{N}(0, 1)$
    distractors. ``reverse_roles=True`` swaps which physical series lands in
    the target slot vs. the source slot, so the model measures
    $\mathrm{TE}_{\text{target slot} \to \text{source slot}}$ on a process
    whose true direction was $\text{src} \to \text{tgt}$ -- the lag-attention
    control where the true TE is zero.

    Args:
        informative_target: ``(n, T, M)`` tensor that goes into the target
            buffer in the forward orientation.
        informative_source: ``(n, T, M)`` tensor that goes into the source
            buffer in the forward orientation.
        M: Number of informative channels.
        c_y: Target channel count.
        c_u: Source channel count.
        reverse_roles: When ``True`` swap the two streams.
        distractor_generator: ``torch.Generator`` for the i.i.d. distractor
            noise; passed in so the caller controls determinism.

    Returns:
        A tuple ``(Y_buf, U_buf)`` of ``float32`` tensors with the v2 native
        shapes $(n, T, c_y)$ and $(n, T, c_u)$.
    """
    n, T, _ = informative_target.shape
    y_stream = informative_source if reverse_roles else informative_target
    u_stream = informative_target if reverse_roles else informative_source

    Y_buf = torch.empty(n, T, c_y, dtype=torch.float32)
    Y_buf[..., :M] = y_stream
    if M < c_y:
        Y_buf[..., M:] = torch.randn(
            n, T, c_y - M, generator=distractor_generator, dtype=torch.float32
        )

    U_buf = torch.empty(n, T, c_u, dtype=torch.float32)
    U_buf[..., :M] = u_stream
    if M < c_u:
        U_buf[..., M:] = torch.randn(
            n, T, c_u - M, generator=distractor_generator, dtype=torch.float32
        )
    return Y_buf, U_buf


# ---------------------------------------------------------------------------
# Benchmark G1: AR(2)-oscillator state-space
# ---------------------------------------------------------------------------


def gen_state_space_oscillator(
    n: int,
    T: int,
    *,
    oscillators: Sequence[Tuple[float, float]],
    target_ar: float,
    delays: Sequence[int],
    B_y: Sequence[float],
    sigma2_y: float,
    sigma2_eta: Union[float, Sequence[float]],
    M: int,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    easy_variant: bool = False,
    standardize: bool = True,
    reverse_roles: bool = False,
    seed: int = 0,
    te_n_samples: int = 50_000,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the G1 AR(2)-oscillator state-space benchmark.

    For each informative channel $m \in \{0, \ldots, M-1\}$ the source state
    follows an AR(2) oscillator

    $$
    s^{(m)}_t = 2\,r_m \cos(\omega_m)\,s^{(m)}_{t-1}
        - r_m^{2}\,s^{(m)}_{t-2} + \eta^{(m)}_t,
    \qquad
    \eta^{(m)}_t \sim \mathcal{N}(0, \sigma^{2}_{\eta, m}),
    $$

    and the target couples through the lagged source state

    $$
    y^{(m)}_t = A_y\,y^{(m)}_{t-1} + B_y^{(m)}\,s^{(m)}_{t-D_m}
        + \varepsilon^{(m)}_t,
    \qquad
    \varepsilon^{(m)}_t \sim \mathcal{N}(0, \sigma^{2}_{y}).
    $$

    The observed source $U$ holds $s^{(m)}_t$ directly (no further delay
    between $S$ and $U$); the delay lives entirely in $Y$'s dependence on
    $S$. Therefore, for an anchor time $t$ and forecast offset $\tau$, the
    future target $y_{t+\tau}$ depends on $s_{t+\tau - D_m}$, i.e. on
    $u_{t-(D_m-\tau)}$. The true source-lag band is

    $$
    \mathcal{L}^{\star}_m = \{D_m - H,\,\ldots,\,D_m - 1\}
    $$

    per oscillator. With ``reverse_roles=True`` the same process is generated
    but the source-side ($S$) stream is returned in the target slot and the
    target-side ($Y$) stream in the source slot, so the slot-wise TE is zero.

    Args:
        n: Number of independent sequences (samples) to generate.
        T: Sequence length (decimated time steps).
        oscillators: Length-$M$ list of $(r_m, \omega_m)$ pairs with
            $r_m \in [0, 1)$.
        target_ar: Scalar target self-coefficient $A_y \in [0, 1)$.
        delays: Length-$M$ list of source-to-target delays $D_m \ge H$.
        B_y: Length-$M$ list of target loadings $B_y^{(m)}$.
        sigma2_y: Shared target innovation variance $\sigma^{2}_{y} > 0$.
        sigma2_eta: Source innovation variance -- scalar (broadcast) or
            length-$M$ array of positive floats.
        M: Number of informative channels; must equal ``len(oscillators)``
            and must satisfy $1 \le M \le \min(c_y, c_u)$.
        c_y: Target channel count (default 87).
        c_u: Source channel count (default 101).
        horizon: Forecast horizon $H$.
        easy_variant: If ``True``, tile the user-supplied oscillator specs to
            fill every target channel ($M$ becomes $c_y$, avoiding channel
            dilution).
        standardize: If ``True`` (default), z-score every channel.
        reverse_roles: If ``True`` produce the G1-rev directionality variant.
        seed: Seed for both the simulator and the distractor RNG.
        te_n_samples: Monte-Carlo sample size for ``meta.te_true``. Skipped
            when ``reverse_roles=True``.

    Returns:
        ``(Y, U, meta)`` with $Y \in \mathbb{R}^{n \times T \times c_y}$,
        $U \in \mathbb{R}^{n \times T \times c_u}$ (both ``float32``), and
        a ground-truth metadata dict.

    Raises:
        ValueError: On any inconsistent input.
    """
    if n <= 0 or T <= 0:
        raise ValueError(
            f"gen_state_space_oscillator: n, T must be > 0, got n={n}, T={T}."
        )
    if horizon <= 0:
        raise ValueError(
            f"gen_state_space_oscillator: horizon must be > 0, got {horizon}."
        )
    if not 1 <= M <= min(c_y, c_u):
        raise ValueError(
            f"gen_state_space_oscillator: require 1 <= M <= min(c_y, c_u)="
            f"{min(c_y, c_u)}, got M={M}."
        )

    oscillators_list: List[Tuple[float, float]] = [
        (float(o[0]), float(o[1])) for o in oscillators
    ]
    delays_list: List[int] = [int(d) for d in delays]
    B_y_list: List[float] = [float(b) for b in B_y]

    if easy_variant:
        # Tile the supplied specs to fill c_y target channels.
        base = len(oscillators_list)
        if base == 0:
            raise ValueError(
                "gen_state_space_oscillator: easy_variant requires at least "
                "one oscillator spec to tile."
            )
        reps = (c_y + base - 1) // base
        oscillators_list = (oscillators_list * reps)[:c_y]
        delays_list = (delays_list * reps)[:c_y]
        B_y_list = (B_y_list * reps)[:c_y]
        if not isinstance(sigma2_eta, (int, float)):
            sigma2_eta_list = list(sigma2_eta)
            sigma2_eta = (sigma2_eta_list * reps)[:c_y]
        M = c_y

    if not (len(oscillators_list) == M and len(delays_list) == M
            and len(B_y_list) == M):
        raise ValueError(
            "gen_state_space_oscillator: len(oscillators) / len(delays) / "
            f"len(B_y) must all equal M={M}; got {len(oscillators_list)} / "
            f"{len(delays_list)} / {len(B_y_list)}."
        )
    if min(delays_list) < horizon:
        raise ValueError(
            f"gen_state_space_oscillator: every delay must be >= horizon="
            f"{horizon}, got delays={delays_list}."
        )

    # Simulate via the same helper that computes the analytic TE.
    S_np, Y_np = _simulate_state_space_gaussian(
        n=n,
        T=T,
        oscillators=oscillators_list,
        target_ar=target_ar,
        delays=delays_list,
        B_y=B_y_list,
        sigma2_y=sigma2_y,
        sigma2_eta=sigma2_eta,
        seed=seed,
    )
    S_inf = torch.from_numpy(S_np.astype(np.float32))
    Y_inf = torch.from_numpy(Y_np.astype(np.float32))

    distractor_gen = torch.Generator().manual_seed(int(seed) + 7919)
    Y_buf, U_buf = _place_into_buffers(
        informative_target=Y_inf,
        informative_source=S_inf,
        M=M,
        c_y=c_y,
        c_u=c_u,
        reverse_roles=reverse_roles,
        distractor_generator=distractor_gen,
    )

    if reverse_roles:
        te_true = 0.0
        true_lag_band: List[int] = []
        direction = "Y_to_X"
    else:
        te_true = te_block_state_space_gaussian(
            oscillators=oscillators_list,
            target_ar=target_ar,
            delays=delays_list,
            B_y=B_y_list,
            sigma2_y=sigma2_y,
            sigma2_eta=sigma2_eta,
            H=horizon,
            n_samples=te_n_samples,
            seed=int(seed) + 1_337,
        )
        lag_set = set()
        for D_m in delays_list:
            for ell in range(max(0, D_m - horizon), D_m):
                lag_set.add(ell)
        true_lag_band = sorted(lag_set)
        direction = "X_to_Y"

    if standardize:
        Y_buf = _standardize_per_channel(Y_buf)
        U_buf = _standardize_per_channel(U_buf)

    meta: Dict[str, Any] = {
        "benchmark": "G1",
        "te_true": float(te_true),
        "te_per_step": float(te_true) / horizon,
        "true_lag_band": true_lag_band,
        "informative_channels": list(range(M)),
        "clean_anchor_range": [max(delays_list) - 1, T - horizon],
        "oscillators": [list(o) for o in oscillators_list],
        "target_ar": float(target_ar),
        "delays": delays_list,
        "B_y": B_y_list,
        "sigma2_y": float(sigma2_y),
        "sigma2_eta": (
            float(sigma2_eta)
            if isinstance(sigma2_eta, (int, float))
            else [float(s) for s in sigma2_eta]
        ),
        "M": M,
        "horizon": horizon,
        "sequence_length": T,
        "c_y": c_y,
        "c_u": c_u,
        "easy_variant": easy_variant,
        "standardized": standardize,
        "reverse_roles": reverse_roles,
        "direction": direction,
        "seed": seed,
    }
    return Y_buf, U_buf, meta


# ---------------------------------------------------------------------------
# Benchmark G2: smooth AR(1)-ARX
# ---------------------------------------------------------------------------


def gen_smooth_arx(
    n: int,
    T: int,
    *,
    rho_u: float,
    rho_y: float,
    c: float,
    sigma2_eta: float,
    sigma2_eps: float,
    delay: int,
    M: int,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    burn_in: Optional[int] = None,
    easy_variant: bool = False,
    standardize: bool = True,
    reverse_roles: bool = False,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the G2 smooth AR(1)-ARX benchmark.

    For each informative channel $m \in \{0, \ldots, M-1\}$ the source and
    target follow

    $$
    U^{(m)}_t = \rho_u\,U^{(m)}_{t-1} + \eta^{(m)}_t,
    \qquad
    Y^{(m)}_t = \rho_y\,Y^{(m)}_{t-1} + c\,U^{(m)}_{t-D}
        + \varepsilon^{(m)}_t,
    $$

    with $\eta \sim \mathcal{N}(0, \sigma_\eta^{2})$,
    $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon^{2})$, all parameters
    shared across channels (only noise realisations differ). For
    $D \ge H$ every future target step depends on an already-observed source
    value, so the analytic block TE is :func:`te_block_arx_gaussian`
    (computed once per cache build and written to ``meta.te_true``).

    Args:
        n: Number of independent sequences (samples).
        T: Sequence length (decimated time steps).
        rho_u: AR(1) source coefficient, $0 \le \rho_u < 1$.
        rho_y: AR(1) target coefficient, $0 \le \rho_y < 1$.
        c: Scalar source-to-target transfer coefficient.
        sigma2_eta: Source innovation variance $> 0$.
        sigma2_eps: Target innovation variance $> 0$.
        delay: Source-to-target delay $D$ with $D \ge H$.
        M: Number of informative channels, $1 \le M \le \min(c_y, c_u)$.
        c_y, c_u: Target / source channel counts (defaults 87 / 101).
        horizon: Forecast horizon $H$.
        burn_in: Warm-up steps discarded for stationarity. Defaults to
            ``max(200, 5 / (1 - max(rho_u, rho_y)))``.
        easy_variant: If ``True`` set $M = c_y$ (all target channels
            informative).
        standardize: If ``True`` (default), z-score every channel.
        reverse_roles: If ``True`` produce the G2-rev directionality variant.
        seed: Seed for the CPU ``torch.Generator``.

    Returns:
        ``(Y, U, meta)`` per the v2 contract; ``meta.te_true`` is the
        per-channel ARX block TE scaled by $M$ (or $0$ when
        ``reverse_roles=True``).

    Raises:
        ValueError: On any inconsistent input.
    """
    if n <= 0 or T <= 0:
        raise ValueError(
            f"gen_smooth_arx: n, T must be > 0, got n={n}, T={T}."
        )
    if horizon <= 0:
        raise ValueError(
            f"gen_smooth_arx: horizon must be > 0, got {horizon}."
        )
    if delay < horizon:
        raise ValueError(
            f"gen_smooth_arx: require delay >= horizon, got delay={delay}, "
            f"horizon={horizon}."
        )
    if not 0.0 <= rho_u < 1.0:
        raise ValueError(
            f"gen_smooth_arx: rho_u must be in [0, 1), got {rho_u}."
        )
    if not 0.0 <= rho_y < 1.0:
        raise ValueError(
            f"gen_smooth_arx: rho_y must be in [0, 1), got {rho_y}."
        )
    if sigma2_eta <= 0.0 or sigma2_eps <= 0.0:
        raise ValueError(
            f"gen_smooth_arx: sigma2_eta and sigma2_eps must be > 0, "
            f"got {sigma2_eta} and {sigma2_eps}."
        )
    if easy_variant:
        M = c_y
    if not 1 <= M <= min(c_y, c_u):
        raise ValueError(
            f"gen_smooth_arx: require 1 <= M <= min(c_y, c_u)="
            f"{min(c_y, c_u)}, got M={M}."
        )

    if burn_in is None:
        denom = max(1.0 - max(rho_u, rho_y), 1e-3)
        burn_in = max(200, int(5.0 / denom))

    gen = torch.Generator().manual_seed(int(seed))
    T_total = burn_in + T
    if delay >= T_total:
        raise ValueError(
            f"gen_smooth_arx: delay={delay} must be < burn_in + T={T_total}."
        )
    eta = math.sqrt(sigma2_eta) * torch.randn(
        n, T_total, M, generator=gen, dtype=torch.float32
    )
    eps = math.sqrt(sigma2_eps) * torch.randn(
        n, T_total, M, generator=gen, dtype=torch.float32
    )

    U = torch.empty(n, T_total, M, dtype=torch.float32)
    Y = torch.empty(n, T_total, M, dtype=torch.float32)
    U[:, 0, :] = eta[:, 0, :]
    Y[:, 0, :] = eps[:, 0, :]
    for t in range(1, T_total):
        U[:, t, :] = rho_u * U[:, t - 1, :] + eta[:, t, :]
        if t >= delay:
            drive = c * U[:, t - delay, :]
            Y[:, t, :] = rho_y * Y[:, t - 1, :] + drive + eps[:, t, :]
        else:
            Y[:, t, :] = rho_y * Y[:, t - 1, :] + eps[:, t, :]

    U_inf = U[:, burn_in:, :].contiguous()
    Y_inf = Y[:, burn_in:, :].contiguous()

    distractor_gen = torch.Generator().manual_seed(int(seed) + 7919)
    Y_buf, U_buf = _place_into_buffers(
        informative_target=Y_inf,
        informative_source=U_inf,
        M=M,
        c_y=c_y,
        c_u=c_u,
        reverse_roles=reverse_roles,
        distractor_generator=distractor_gen,
    )

    if reverse_roles:
        te_true = 0.0
        true_lag_band: List[int] = []
        direction = "Y_to_X"
    else:
        te_per_channel = te_block_arx_gaussian(
            rho_u=rho_u, rho_y=rho_y, c=c,
            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
            H=horizon, D=delay,
        )
        te_true = M * te_per_channel
        true_lag_band = list(range(delay - horizon, delay))
        direction = "X_to_Y"

    if standardize:
        Y_buf = _standardize_per_channel(Y_buf)
        U_buf = _standardize_per_channel(U_buf)

    meta: Dict[str, Any] = {
        "benchmark": "G2",
        "te_true": float(te_true),
        "te_per_step": float(te_true) / horizon,
        "true_lag_band": true_lag_band,
        "informative_channels": list(range(M)),
        "clean_anchor_range": [delay - 1, T - horizon],
        "rho_u": float(rho_u),
        "rho_y": float(rho_y),
        "c": float(c),
        "sigma2_eta": float(sigma2_eta),
        "sigma2_eps": float(sigma2_eps),
        "delay": delay,
        "M": M,
        "horizon": horizon,
        "sequence_length": T,
        "burn_in": burn_in,
        "c_y": c_y,
        "c_u": c_u,
        "easy_variant": easy_variant,
        "standardized": standardize,
        "reverse_roles": reverse_roles,
        "direction": direction,
        "seed": seed,
    }
    return Y_buf, U_buf, meta


# ---------------------------------------------------------------------------
# Benchmark G3: slow categorical regime switch with smooth oscillator target
# ---------------------------------------------------------------------------


def _sample_regime_chains(
    n_chains: int,
    T: int,
    K: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Sample ``n_chains`` independent inclusive-redraw Markov chains.

    Each chain $R_t \in \{0, \ldots, K-1\}$ evolves under the inclusive-redraw
    kernel of ``model_validation_v2.md`` Section 5: with probability $p$ the
    next state is redrawn uniformly from all $K$ classes (so the same class
    can be re-selected); with probability $1 - p$ the state is kept. The
    stationary distribution is uniform.

    Args:
        n_chains: Number of independent chains to sample.
        T: Sequence length per chain.
        K: Number of categorical classes, $K \ge 2$.
        p: Switch probability in $[0, 1]$.
        rng: NumPy random generator (caller-supplied for determinism).

    Returns:
        An ``(n_chains, T)`` ``int`` array of regime indices in $[0, K)$.
    """
    R = np.empty((n_chains, T), dtype=np.int64)
    R[:, 0] = rng.integers(K, size=n_chains)
    switch_mask = rng.random((n_chains, T)) < p
    redraws = rng.integers(K, size=(n_chains, T))
    for t in range(1, T):
        R[:, t] = np.where(switch_mask[:, t], redraws[:, t], R[:, t - 1])
    return R


def gen_regime_switch_smooth(
    n: int,
    T: int,
    *,
    K_classes: int,
    p_switch: float,
    delta: int,
    M: int,
    omega_grid: Optional[Sequence[float]] = None,
    amp_grid: Optional[Sequence[float]] = None,
    sigma2_y: float = 0.1,
    sigma2_u: float = 0.1,
    c_y: int = 87,
    c_u: int = 101,
    horizon: int = 30,
    shared_regime: bool = False,
    template_period_min: int = 40,
    standardize: bool = True,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Generate the G3 slow categorical regime-switch benchmark.

    For each informative channel $m$, a slow inclusive-redraw Markov chain
    $R^{(m)}_t$ on $K$ classes governs an instantaneous oscillator frequency
    $\omega(R^{(m)}_t)$ and amplitude $A(R^{(m)}_t)$. The phase is
    accumulated continuously, so only the instantaneous frequency jumps at
    regime change:

    $$
    \Phi^{(m)}_t = \Phi^{(m)}_{t-1} + \omega_{R^{(m)}_t},
    \quad
    Y^{(m)}_t = A_{R^{(m)}_t}\,\sin\Phi^{(m)}_t + \varepsilon^{y,m}_t.
    $$

    The source at time $t$ leaks the regime that will be active at time
    $t + \delta$ via a noisy per-channel one-hot encoding:

    $$
    U^{(m,k)}_t = \mathbb{1}[\,R^{(m)}_{t+\delta} = k\,]
        + \varepsilon^{u,m,k}_t.
    $$

    Channel layout: target channels $[0, M)$ hold the $Y^{(m)}_t$; source
    channels $[m K,\,(m+1) K)$ for $m \in [0, M)$ hold the per-channel
    one-hot encoding -- the source must therefore satisfy
    $M \cdot K \le c_u$. Distractor channels are i.i.d. $\mathcal{N}(0, 1)$.

    Since regimes are renewed per step, the block TE is simply
    $M \cdot H \cdot \mathrm{TE}^{(1)}_{\text{cat}}(p, K)$ (or
    $H \cdot \mathrm{TE}^{(1)}_{\text{cat}}$ when ``shared_regime=True``).
    True lag band: $\{\delta - H,\,\ldots,\,\delta - 1\}$ (requires
    $\delta \ge H$).

    Args:
        n: Number of independent sequences.
        T: Sequence length.
        K_classes: Number of categorical classes, $K \ge 2$.
        p_switch: Per-step switch probability in $[0, 1]$.
        delta: Source-lead lag (source at time $t$ reveals $R_{t+\delta}$);
            must satisfy $\delta \ge H$.
        M: Number of informative channels.
        omega_grid: Optional length-$K$ list of per-regime frequencies.
            Defaults to $K$ values linearly spaced in
            $[\,\pi / \text{template\_period\_min},\,
            2\pi / \text{template\_period\_min}\,]$.
        amp_grid: Optional length-$K$ list of per-regime amplitudes.
            Defaults to ``linspace(0.5, 1.5, K)``.
        sigma2_y: Target observation-noise variance.
        sigma2_u: Source observation-noise variance.
        c_y, c_u: Target / source channel counts (defaults 87 / 101).
        horizon: Forecast horizon $H$.
        shared_regime: If ``True`` all $M$ channels share one chain.
        template_period_min: Lower bound on per-regime period (in time
            steps). Must satisfy ``template_period_min > horizon`` so the
            future window is intra-regime predictable.
        standardize: If ``True`` (default), z-score every channel.
        seed: Seed for the NumPy random generator and the distractor
            ``torch.Generator``.

    Returns:
        ``(Y, U, meta)`` per the v2 contract.

    Raises:
        ValueError: On any inconsistent input -- including the
            $M \cdot K \le c_u$ channel-budget constraint and
            $\delta \ge H$.
    """
    if n <= 0 or T <= 0:
        raise ValueError(
            f"gen_regime_switch_smooth: n, T must be > 0, got n={n}, T={T}."
        )
    if horizon <= 0:
        raise ValueError(
            f"gen_regime_switch_smooth: horizon must be > 0, got {horizon}."
        )
    if K_classes < 2:
        raise ValueError(
            f"gen_regime_switch_smooth: K_classes must be >= 2, got {K_classes}."
        )
    if not 0.0 <= p_switch <= 1.0:
        raise ValueError(
            f"gen_regime_switch_smooth: p_switch must be in [0, 1], got "
            f"{p_switch}."
        )
    if delta < horizon:
        raise ValueError(
            f"gen_regime_switch_smooth: delta must be >= horizon, got "
            f"delta={delta}, horizon={horizon}."
        )
    if template_period_min <= horizon:
        raise ValueError(
            f"gen_regime_switch_smooth: template_period_min must exceed "
            f"horizon for intra-regime predictability; got "
            f"template_period_min={template_period_min}, horizon={horizon}."
        )
    if M < 1 or M > c_y:
        raise ValueError(
            f"gen_regime_switch_smooth: require 1 <= M <= c_y={c_y}, "
            f"got M={M}."
        )
    if M * K_classes > c_u:
        raise ValueError(
            f"gen_regime_switch_smooth: per-channel one-hot encoding needs "
            f"M * K_classes = {M * K_classes} source channels, but "
            f"c_u={c_u}."
        )
    if sigma2_y <= 0.0 or sigma2_u <= 0.0:
        raise ValueError(
            f"gen_regime_switch_smooth: sigma2_y and sigma2_u must be > 0, "
            f"got {sigma2_y} and {sigma2_u}."
        )

    if omega_grid is None:
        # K slow frequencies; all periods >= template_period_min > horizon.
        omega_max = 2.0 * math.pi / template_period_min
        omega_min = math.pi / template_period_min
        omega_grid = list(np.linspace(omega_min, omega_max, K_classes))
    if amp_grid is None:
        amp_grid = list(np.linspace(0.5, 1.5, K_classes))
    if len(omega_grid) != K_classes or len(amp_grid) != K_classes:
        raise ValueError(
            f"gen_regime_switch_smooth: omega_grid / amp_grid must each have "
            f"length K_classes={K_classes}."
        )

    rng = np.random.default_rng(int(seed))

    # Need regimes for t in [0, T + delta) so:
    #   - Y_t (t in [0, T))         consumes R[:, t]
    #   - U_t = onehot(R_{t+delta}) consumes R[:, t+delta] for t in [0, T)
    T_R = T + delta
    n_chains = 1 if shared_regime else n * M
    R_flat = _sample_regime_chains(n_chains, T_R, K_classes, p_switch, rng)
    if shared_regime:
        # _sample_regime_chains returned shape (1, T_R); drop the chain axis.
        R = np.broadcast_to(R_flat[0][None, :, None], (n, T_R, M)).copy()
    else:
        R = R_flat.reshape(n, M, T_R).transpose(0, 2, 1)  # (n, T_R, M)

    omega_arr = np.asarray(omega_grid, dtype=np.float32)  # (K,)
    amp_arr = np.asarray(amp_grid, dtype=np.float32)       # (K,)

    R_target = R[:, :T, :]                                 # (n, T, M)
    omega_R = omega_arr[R_target]                          # (n, T, M)
    amp_R = amp_arr[R_target]                              # (n, T, M)
    Phi = np.cumsum(omega_R, axis=1)                       # (n, T, M)
    eps_y = math.sqrt(sigma2_y) * rng.standard_normal((n, T, M)).astype(np.float32)
    Y_inf_np = (amp_R * np.sin(Phi) + eps_y).astype(np.float32)

    R_source = R[:, delta : T + delta, :]                  # (n, T, M)
    # Per-channel one-hot: U[:, :, m*K : (m+1)*K] = onehot(R_source[:, :, m]).
    U_inf_np = np.zeros((n, T, M * K_classes), dtype=np.float32)
    for m in range(M):
        for k in range(K_classes):
            U_inf_np[:, :, m * K_classes + k] = (
                R_source[:, :, m] == k
            ).astype(np.float32)
    eps_u = math.sqrt(sigma2_u) * rng.standard_normal(
        (n, T, M * K_classes)
    ).astype(np.float32)
    U_inf_np = U_inf_np + eps_u

    Y_inf = torch.from_numpy(Y_inf_np)
    U_inf = torch.from_numpy(U_inf_np)

    distractor_gen = torch.Generator().manual_seed(int(seed) + 7919)
    Y_buf = torch.empty(n, T, c_y, dtype=torch.float32)
    Y_buf[..., :M] = Y_inf
    if M < c_y:
        Y_buf[..., M:] = torch.randn(
            n, T, c_y - M, generator=distractor_gen, dtype=torch.float32
        )

    n_source_inf = M * K_classes
    U_buf = torch.empty(n, T, c_u, dtype=torch.float32)
    U_buf[..., :n_source_inf] = U_inf
    if n_source_inf < c_u:
        U_buf[..., n_source_inf:] = torch.randn(
            n, T, c_u - n_source_inf, generator=distractor_gen, dtype=torch.float32
        )

    if standardize:
        Y_buf = _standardize_per_channel(Y_buf)
        U_buf = _standardize_per_channel(U_buf)

    n_independent_chains = 1 if shared_regime else M
    te_true = n_independent_chains * te_categorical_switch_block(
        p_switch, K_classes, horizon
    )

    meta: Dict[str, Any] = {
        "benchmark": "G3",
        "te_true": float(te_true),
        "te_per_step": float(te_true) / horizon,
        "true_lag_band": list(range(delta - horizon, delta)),
        "informative_channels": list(range(M)),
        "informative_source_channels": list(range(M * K_classes)),
        "clean_anchor_range": [delta - 1, T - horizon],
        "K_classes": K_classes,
        "p_switch": float(p_switch),
        "delta": delta,
        "shared_regime": shared_regime,
        "omega_grid": [float(w) for w in omega_grid],
        "amp_grid": [float(a) for a in amp_grid],
        "sigma2_y": float(sigma2_y),
        "sigma2_u": float(sigma2_u),
        "M": M,
        "horizon": horizon,
        "sequence_length": T,
        "c_y": c_y,
        "c_u": c_u,
        "template_period_min": template_period_min,
        "standardized": standardize,
        "reverse_roles": False,
        "direction": "X_to_Y",
        "seed": seed,
    }
    return Y_buf, U_buf, meta


# TODO Sprint 7: gen_switched_sinusoid (G4) -- multi-component sinusoid with
# source-announced frequency switch. Deferred per v2-D9.
