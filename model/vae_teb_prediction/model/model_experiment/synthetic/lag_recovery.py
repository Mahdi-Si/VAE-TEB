r"""Lag-recovery metrics -- does the model find the true source lag (Phase 5).

Loads a trained :class:`SeqVaeLagAttnV1` checkpoint and quantifies whether the
model localises transfer to the true source-lag band
$\mathcal{L}^\star = \{D-H,\dots,D-1\}$, using three complementary measures
(``model_validation_v2_plan.md`` Sprint 4):

    Task 5.2 -- attention lag-mass: the fraction of the ``te_lag_map`` mass
        $\widetilde{\mathrm{TE}}_{t,\ell} = K_t\,\bar\alpha_{t,\ell}$ that lands
        inside $\mathcal{L}^\star$, over valid anchors, compared to the uniform
        baseline $|\mathcal{L}^\star| / (L_{\max} + 1)$.
    Task 5.3 -- peak-lag error: the per-anchor $\arg\max_\ell$ of the lag map
        versus the band centre, plus the in-band fraction.
    Sprint 4.1 -- sliding-window leave-one-lag-out (v2-D5, the faithful LOLO
        for smooth sources): the recurrent ``SourceEncoder`` makes internal
        lag-memory masking non-causal, so ablation corrupts the **raw source
        input**. For each lag $\ell$ in the lag grid a contiguous window of
        ``u_stream`` centred on $t-\ell$ across all valid anchors is replaced
        with $\mathcal{N}(0,1)$ noise; the resulting feat-loss degradation
        $\delta_\ell$ is the per-lag importance. The v1 i.i.d.-only LOLO (whole
        source corruption + $\tau\to\ell$ scatter) was deleted in Sprint 4
        because the $\tau\to\ell$ bijection breaks for autocorrelated sources.
        See :func:`run_sliding_window_lolo`.

It also surfaces the Section 4.2 finding (Task 5.5): when the delay $D$ exceeds
the attention window $L_{\max}$, the recurrent encoder still carries the source,
so $K$ can stay non-zero. :func:`_delay_window_report` reports this for any
checkpoint; the dedicated large-$D$ run is deferred.

For two-band benchmarks (e.g. ``G1_twoband`` with two distinct delays) the
harness adds the **two-band mass ratio** (:func:`compute_two_band_mass_ratio`):
the lag map should resolve two separate bands whose mass ratio tracks the
per-band TE ratio. The same two-band split is also applied to the
sliding-window LOLO profile $A_\ell$ via :func:`compute_lag_mass_from_profile`.

This module **reuses** :func:`evaluate_te.load_eval_checkpoint` /
:func:`evaluate_te.make_test_loader` and the :mod:`train_minimal` helpers so it
scores models with the exact model / loss code the earlier phases used.

Run modes (project convention -- Decision D9 / V2-D8): like every
``synthetic/`` runner this file supports **both** a CLI and an edit-and-run
``__main__``, auto-detected from whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.lag_recovery --checkpoint PATH [--config PATH]
            [--data-tag TAG] [--benchmark B] [--batch-size N]
            [--mode {analyze,width_sweep}] [--widths "1,5,10,20"]
            [--window-width N] [--n-ablation-samples N] [--device DEV]
            [--seed S]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.lag_recovery
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    evaluate_te as ev,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    build_u_stream,
)

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.*`` config values are resolved relative to ``model_experiment/``
# (identical convention to evaluate_te.py / train_minimal.py).
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Columns of the one-row summary CSV (task 5.6). The full per-lag vectors
# (``A_lag``, ``delta_per_lag``, lag profiles) go to ``metrics.json``. ``B_y`` /
# ``c`` / ``p_switch`` are the v2 benchmark-specific sweep knobs (G1 / G2 / G3)
# matching :data:`evaluate_te._SUMMARY_FIELDS`; whichever the active benchmark
# does not use is left blank. The ``lag_mass_band*`` columns are populated only
# for multi-band benchmarks (e.g. ``G1_twoband``). ``window_width`` records the
# sliding-window LOLO width chosen for this row.
_SUMMARY_FIELDS = [
    "run_tag", "data_tag", "benchmark",
    "B_y", "c", "p_switch", "M",
    "delay", "horizon", "te_true", "warmup", "n_test", "L",
    "lag_band_lo", "lag_band_hi",
    "lag_mass_attn", "lag_mass_attn_uniform", "lag_mass_attn_ratio",
    "peak_lag_err_mean", "peak_lag_err_median", "peak_in_band_frac",
    "lag_mass_lolo", "window_width", "delta_oob_max",
    "lag_mass_band1", "lag_mass_band2", "lag_mass_ratio",
    "lag_mass_te_ratio", "lag_mass_ratio_err",
    "lag_mass_lolo_band1", "lag_mass_lolo_band2",
    "k_bar", "epoch", "ckpt_path",
]


# =============================================================================
# Path helper
# =============================================================================

def _lag_out_dir(config: Dict[str, Any], benchmark: str) -> Path:
    """Resolve (and create) the Phase-5 output directory.

    Args:
        config: The parsed ``config_synth.yaml``.
        benchmark: Benchmark identifier (e.g. ``"A"``).

    Returns:
        ``<results_root>/<benchmark>/lag_recovery`` -- created if absent. Keeps
        the lag-recovery artifacts separate from ``eval_te`` and per-run
        training directories.
    """
    results_root = tm.resolve_user_path(config["paths"]["results_dir"])
    out_dir = results_root / str(benchmark) / "lag_recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# Task 5.1 -- attention / te-lag-map collection
# =============================================================================

@torch.no_grad()
def collect_lag_tensors(
    model: SeqVaeLagAttnV1,
    loader: Any,
    device: torch.device,
    *,
    warmup: int,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Collect the head-averaged attention and lag map over the test loader.

    One ``eval``-mode :meth:`SeqVaeLagAttnV1.forward` pass, accumulating
    batch-axis means while keeping the ``(T, L)`` axes intact. The full
    ``(B, T, L)`` tensors are never stacked -- only running ``(T, L)`` sums.

    Args:
        model: The trained model (on ``device``).
        loader: A test :class:`DataLoader`.
        device: Compute device.
        warmup: Leading time steps excluded from the $\bar K$ aggregate.
        max_batches: Optional cap on the number of batches consumed.

    Returns:
        Dict with ``mean_alpha_tl`` $(T, L)$ -- head-averaged attention
        $\bar\alpha_{t,\ell}$; ``te_lag_map_tl`` $(T, L)$; ``kld_per_t_t``
        $(T,)$ -- the per-step KL profile (summed over $d_z$); ``k_bar`` --
        mean ``kld_per_t`` over $t \in [\text{warmup}, T)$; and ``T``,
        ``L``, ``n_samples``.
    """
    model.eval()
    sum_alpha: Optional[torch.Tensor] = None  # (T, L)
    sum_te: Optional[torch.Tensor] = None     # (T, L)
    sum_kld: Optional[torch.Tensor] = None    # (T,)
    n_samples = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = tm.move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)
        bs = int(y_st.size(0))

        out = model(y_st, y_ph, u_stream)
        # attn_weights (B, T, num_heads, L) -> head-mean (B, T, L).
        alpha = out["attn_weights"].mean(dim=2)
        te = out["te_lag_map"]                       # (B, T, L)
        kld = out["kld_per_t"]                       # (B, T)

        a_sum = alpha.sum(dim=0)                     # (T, L)
        t_sum = te.sum(dim=0)                        # (T, L)
        k_sum = kld.sum(dim=0)                       # (T,)
        sum_alpha = a_sum if sum_alpha is None else sum_alpha + a_sum
        sum_te = t_sum if sum_te is None else sum_te + t_sum
        sum_kld = k_sum if sum_kld is None else sum_kld + k_sum
        n_samples += bs

    if n_samples == 0:
        raise RuntimeError("collect_lag_tensors: the loader yielded no batches")

    mean_alpha = (sum_alpha / n_samples).cpu().numpy()
    mean_te = (sum_te / n_samples).cpu().numpy()
    mean_kld = (sum_kld / n_samples).cpu().numpy()
    T, L = mean_alpha.shape
    k_bar = (
        float(np.mean(mean_kld[warmup:])) if warmup < T else float("nan")
    )
    return {
        "mean_alpha_tl": mean_alpha,
        "te_lag_map_tl": mean_te,
        "kld_per_t_t": mean_kld,
        "k_bar": k_bar,
        "T": int(T),
        "L": int(L),
        "n_samples": int(n_samples),
    }


# =============================================================================
# Task 5.2 -- attention lag-mass
# =============================================================================

def compute_lag_mass_attn(
    te_lag_map_tl: np.ndarray,
    *,
    lag_band: Sequence[int],
    anchor_lo: int,
    anchor_hi: int,
    max_lag: int,
) -> Dict[str, Any]:
    r"""Fraction of ``te_lag_map`` mass inside the true lag band (task 5.2).

    Because $\bar\alpha_{t,\ell}$ sums to 1 over the lag axis,
    $\sum_\ell \widetilde{\mathrm{TE}}_{t,\ell} = K_t$, so this is a
    $K_t$-weighted in-band attention fraction. Compared against the uniform
    baseline $|\mathcal{L}^\star| / (L_{\max} + 1)$.

    Args:
        te_lag_map_tl: Head-averaged lag map $(T, L)$.
        lag_band: The true source-lag indices $\mathcal{L}^\star$.
        anchor_lo: First valid anchor (inclusive).
        anchor_hi: Last valid anchor (exclusive).
        max_lag: Maximum attention lag ($L = L_{\max} + 1$).

    Returns:
        Dict with ``lag_mass_attn`` (``nan`` if the total mass collapses to
        ~0), ``uniform_baseline``, ``ratio_to_uniform``, the resolved anchor
        window and the raw band / total masses.
    """
    te = np.asarray(te_lag_map_tl, dtype=float)
    T, L = te.shape
    lo = max(0, int(anchor_lo))
    hi = min(T, int(anchor_hi))
    valid = te[lo:hi, :]                              # (T', L)

    band_idx = np.asarray(list(lag_band), dtype=int)
    band_idx = band_idx[(band_idx >= 0) & (band_idx < L)]

    total = float(valid.sum())
    band = float(valid[:, band_idx].sum()) if band_idx.size else 0.0
    lag_mass = band / total if total >= 1e-12 else float("nan")
    uniform = band_idx.size / float(max_lag + 1)
    ratio = (
        lag_mass / uniform
        if (uniform > 0.0 and np.isfinite(lag_mass))
        else float("nan")
    )
    return {
        "lag_mass_attn": lag_mass,
        "uniform_baseline": uniform,
        "ratio_to_uniform": ratio,
        "anchor_lo": lo,
        "anchor_hi": hi,
        "n_anchors": hi - lo,
        "band_mass": band,
        "total_mass": total,
    }


# =============================================================================
# Task 7.3 -- two-band lag mass ratio (Benchmark E)
# =============================================================================

def compute_two_band_mass_ratio(
    te_lag_map_tl: np.ndarray,
    *,
    lag_band_1: Sequence[int],
    lag_band_2: Sequence[int],
    anchor_lo: int,
    anchor_hi: int,
    max_lag: int,
    te_true_1: float,
    te_true_2: float,
) -> Dict[str, Any]:
    r"""Two-band lag-mass ratio for the two-lag benchmark E (task 7.3).

    Calls :func:`compute_lag_mass_attn` once per band on the *same* lag map --
    the two calls share the full-axis denominator, so the ratio of their raw
    band masses is a clean mass ratio. A model that resolves the two source
    bands in proportion to their information content has

    $$
    \frac{\operatorname{mass}(\mathcal L_1)}{\operatorname{mass}(\mathcal L_2)}
    \;\approx\;
    \frac{\mathrm{TE}_1}{\mathrm{TE}_2}.
    $$

    Args:
        te_lag_map_tl: Head-averaged lag map $(T, L)$.
        lag_band_1: Band-1 source-lag indices $\mathcal L_1$.
        lag_band_2: Band-2 source-lag indices $\mathcal L_2$.
        anchor_lo: First valid anchor (inclusive).
        anchor_hi: Last valid anchor (exclusive).
        max_lag: Maximum attention lag.
        te_true_1: Analytic block TE of band 1.
        te_true_2: Analytic block TE of band 2.

    Returns:
        Dict with the per-band :func:`compute_lag_mass_attn` results
        (``band1`` / ``band2`` / ``union``), the observed ``mass_ratio``, the
        analytic ``te_ratio`` and their absolute ``ratio_error``.
    """
    band1 = compute_lag_mass_attn(
        te_lag_map_tl, lag_band=lag_band_1, anchor_lo=anchor_lo,
        anchor_hi=anchor_hi, max_lag=max_lag,
    )
    band2 = compute_lag_mass_attn(
        te_lag_map_tl, lag_band=lag_band_2, anchor_lo=anchor_lo,
        anchor_hi=anchor_hi, max_lag=max_lag,
    )
    union_band = sorted(set(int(x) for x in lag_band_1)
                        | set(int(x) for x in lag_band_2))
    union = compute_lag_mass_attn(
        te_lag_map_tl, lag_band=union_band, anchor_lo=anchor_lo,
        anchor_hi=anchor_hi, max_lag=max_lag,
    )
    m1, m2 = band1["band_mass"], band2["band_mass"]
    mass_ratio = m1 / m2 if m2 > 1e-12 else float("nan")
    te_ratio = (
        te_true_1 / te_true_2 if abs(te_true_2) > 1e-12 else float("nan")
    )
    ratio_error = (
        abs(mass_ratio - te_ratio)
        if (np.isfinite(mass_ratio) and np.isfinite(te_ratio))
        else float("nan")
    )
    return {
        "band1": band1,
        "band2": band2,
        "union": union,
        "lag_mass_1": band1["lag_mass_attn"],
        "lag_mass_2": band2["lag_mass_attn"],
        "band_mass_1": m1,
        "band_mass_2": m2,
        "union_mass": union["lag_mass_attn"],
        "mass_ratio": mass_ratio,
        "te_ratio": te_ratio,
        "ratio_error": ratio_error,
        "te_true_1": float(te_true_1),
        "te_true_2": float(te_true_2),
    }


# =============================================================================
# Task 5.3 -- peak-lag error
# =============================================================================

def compute_peak_lag_error(
    te_lag_map_anchor: np.ndarray,
    *,
    lag_band: Sequence[int],
) -> Dict[str, Any]:
    r"""Per-anchor peak-lag error versus the true band centre (task 5.3).

    For each valid anchor the peak lag is $\arg\max_\ell \widetilde{\mathrm{TE}}
    _{t,\ell}$ (identical to the $\arg\max$ of $\bar\alpha$, since $K_t \ge 0$
    is a per-lag-constant scale factor). The error is the absolute distance to
    the band centre $(D-H + D-1)/2$.

    Args:
        te_lag_map_anchor: ``te_lag_map`` restricted to valid anchors $(T', L)$.
        lag_band: The true source-lag indices $\mathcal{L}^\star$.

    Returns:
        Dict with ``peak_lag_err_mean`` / ``peak_lag_err_median``,
        ``peak_in_band_frac``, ``peak_lag_centre`` and ``peak_lag_hist`` (a
        length-$L$ histogram of where the per-anchor peak lands).
    """
    te = np.asarray(te_lag_map_anchor, dtype=float)
    n_anchor, L = te.shape
    band = np.asarray(list(lag_band), dtype=int)
    lo, hi = int(band.min()), int(band.max())
    centre = 0.5 * (lo + hi)

    if n_anchor == 0:
        return {
            "peak_lag_err_mean": float("nan"),
            "peak_lag_err_median": float("nan"),
            "peak_in_band_frac": float("nan"),
            "peak_lag_centre": centre,
            "peak_lag_hist": [0.0] * L,
            "n_anchors": 0,
        }

    peak = te.argmax(axis=-1)                          # (T',)
    err = np.abs(peak.astype(float) - centre)
    in_band = ((peak >= lo) & (peak <= hi)).astype(float)
    hist = np.bincount(peak, minlength=L)[:L].astype(float)
    return {
        "peak_lag_err_mean": float(err.mean()),
        "peak_lag_err_median": float(np.median(err)),
        "peak_in_band_frac": float(in_band.mean()),
        "peak_lag_centre": centre,
        "peak_lag_hist": hist.tolist(),
        "n_anchors": int(n_anchor),
    }


# =============================================================================
# Task 5.4 -- input-level leave-one-lag-out ablation
# =============================================================================

def _per_tau_terms(
    mu_full: torch.Tensor,
    Y: torch.Tensor,
    *,
    warmup: int,
    channels: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, float]:
    r"""Masked squared error summed over $(B, T_{valid}, C)$, kept per horizon.

    Replicates the future-target construction of
    :meth:`SeqVaeLagAttnV1.compute_loss` (``vae_teb_lag_attn_v1.py:1664-1700``):
    ``Y_plus[b, t, tau, c] = Y[b, t+1+tau, c]`` via ``unfold``. Synthetic data
    has an all-ones ``weight`` field, so the mask reduces to the warm-up anchor
    mask.

    Args:
        mu_full: Forecast tensor $(B, T, H_d, C)$.
        Y: Concatenated target $(B, T, C)$.
        warmup: Leading anchors excluded from the loss.
        channels: Optional channel subset to score (defaults to all $C$).

    Returns:
        Tuple ``(num_tau, denom)`` -- ``num_tau`` is the length-$H_d$ vector of
        masked SE summed over $(B, T_{valid}, C)$; ``denom`` is the scalar
        $\sum \text{warmup} \cdot B \cdot C$. ``num_tau / denom`` is the per-
        horizon MSE; ``num_tau.sum() / (H_d \cdot denom)`` equals ``feat_loss``.
    """
    B, T, Hd, _ = mu_full.shape
    T_valid = T - Hd
    # Y_shift[:, t, :] = Y[:, t+1, :]; unfold -> (B, T_valid, C, Hd).
    Y_plus = Y[:, 1:, :].unfold(dimension=1, size=Hd, step=1)
    Y_plus = Y_plus.permute(0, 1, 3, 2).contiguous()   # (B, T_valid, Hd, C)
    mu_valid = mu_full[:, :T_valid, :, :]

    if channels is not None:
        ch = torch.as_tensor(
            list(channels), dtype=torch.long, device=mu_full.device
        )
        Y_plus = Y_plus.index_select(-1, ch)
        mu_valid = mu_valid.index_select(-1, ch)
    C = int(Y_plus.shape[-1])

    warmup_t = torch.zeros(T_valid, dtype=mu_full.dtype, device=mu_full.device)
    if warmup < T_valid:
        warmup_t[warmup:] = 1.0

    diff = (mu_valid - Y_plus) ** 2                    # (B, T_valid, Hd, C)
    num = (diff * warmup_t[None, :, None, None]).sum(dim=(0, 1, 3))  # (Hd,)
    denom = float(warmup_t.sum()) * B * C
    return num, denom


@torch.no_grad()
def _per_tau_mse(
    model: SeqVaeLagAttnV1,
    loader: Any,
    device: torch.device,
    *,
    warmup: int,
    horizon: int,
    channels: Optional[Sequence[int]] = None,
    corrupt: Optional[Any] = None,
    seed: int = 0,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    r"""Per-future-step MSE over the loader, optionally corrupting the source.

    The reparameterisation RNG is reseeded once per call so that a clean and a
    corrupted pass with the same ``seed`` share their latent draws -- the
    decomposed degradation $\delta(\tau)$ then reflects the source perturbation,
    not sampling noise.

    Args:
        model: The trained model (``eval`` mode).
        loader: A test :class:`DataLoader`.
        device: Compute device.
        warmup: Leading anchors excluded from the loss.
        horizon: Forecast horizon $H_d$.
        channels: Optional channel subset to score.
        corrupt: ``None`` -> clean; ``'all'`` -> the whole ``u_stream`` is
            replaced by fresh $\mathcal N(0, 1)$ noise; ``(lo, hi)`` -> only the
            absolute-time window ``[lo, hi)`` of ``u_stream`` is replaced.
        seed: Seed for both the reparam RNG (pairing) and the corruption RNG.
        max_batches: Optional cap on the number of batches consumed.

    Returns:
        Length-$H_d$ array of per-horizon MSE averaged over valid anchors and
        scored channels; ``mse.mean()`` equals the (channel-restricted)
        ``feat_loss``.
    """
    model.eval()
    # Pair the reparam draws across clean / corrupt passes (same seed).
    torch.manual_seed(int(seed))
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(int(seed) + 12345)

    sum_num: Optional[torch.Tensor] = None
    sum_denom = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = tm.move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)               # (B, T, c_u)
        B, T, c_u = u_stream.shape

        if corrupt == "all":
            u_stream = torch.randn(
                (B, T, c_u), generator=noise_gen,
                device=device, dtype=u_stream.dtype,
            )
        elif isinstance(corrupt, (tuple, list)):
            lo = max(0, int(corrupt[0]))
            hi = min(T, int(corrupt[1]))
            if hi > lo:
                u_stream = u_stream.clone()
                u_stream[:, lo:hi, :] = torch.randn(
                    (B, hi - lo, c_u), generator=noise_gen,
                    device=device, dtype=u_stream.dtype,
                )

        out = model(y_st, y_ph, u_stream)
        Y = torch.cat([y_st, y_ph], dim=-1)            # clean target
        num, denom = _per_tau_terms(
            out["mu_full"], Y, warmup=warmup, channels=channels,
        )
        sum_num = num if sum_num is None else sum_num + num
        sum_denom += denom

    if sum_num is None or sum_denom <= 0.0:
        return np.full(int(horizon), np.nan)
    return (sum_num / sum_denom).cpu().numpy()


@torch.no_grad()
def _crosscheck_per_tau(
    model: SeqVaeLagAttnV1,
    loader: Any,
    device: torch.device,
    *,
    warmup: int,
    beta: float,
    lambda_full: float,
    lambda_base: float,
    n_batches: int = 2,
) -> float:
    r"""Assert the per-horizon MSE wiring reproduces ``compute_loss``'s feat_loss.

    For the first ``n_batches`` batches, ``num_tau.sum() / (H_d \cdot denom)``
    (from :func:`_per_tau_terms`) must equal
    :meth:`SeqVaeLagAttnV1.compute_loss`'s ``feat_loss`` on the *same* forward
    output -- the single most important correctness gate of task 5.4.

    Args:
        model: The trained model.
        loader: A test :class:`DataLoader`.
        device: Compute device.
        warmup: Leading anchors excluded from the loss.
        beta: KL weight (forwarded to ``compute_loss``).
        lambda_full: Full-forecast loss weight.
        lambda_base: Baseline-forecast loss weight.
        n_batches: Number of batches to check.

    Returns:
        The worst observed relative error across the checked batches.

    Raises:
        AssertionError: If the relative error exceeds ``1e-4`` on any batch.
    """
    model.eval()
    worst = 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = tm.move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)
        out = model(y_st, y_ph, u_stream)
        Y = torch.cat([y_st, y_ph], dim=-1)
        num, denom = _per_tau_terms(out["mu_full"], Y, warmup=warmup)
        Hd = int(out["mu_full"].shape[2])
        mine = float(num.sum()) / (Hd * denom) if denom > 0 else float("nan")
        losses = model.compute_loss(
            out, y_st, y_ph, weight=batch.weight,
            beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
        )
        theirs = float(losses["feat_loss"])
        rel = abs(mine - theirs) / max(abs(theirs), 1e-8)
        worst = max(worst, rel)
        if not (rel <= 1e-4):
            raise AssertionError(
                f"per-tau MSE cross-check failed on batch {i}: "
                f"mean(mse_tau)={mine:.8g} vs feat_loss={theirs:.8g} "
                f"(relative error {rel:.2e}). The unfold / mask wiring in "
                f"_per_tau_terms does not match compute_loss."
            )
    return worst


def compute_lag_mass_from_profile(
    profile: Sequence[float],
    *,
    lag_band: Sequence[int],
) -> Dict[str, Any]:
    r"""Normalise a per-lag importance vector and read off its in-band mass.

    Used by the sliding-window LOLO (whose raw $\delta_\ell$ is on the same
    lag axis as $\bar\alpha_\ell$) and by the two-band wiring of Sprint 4.5
    (separate per-band masses on a shared denominator).

    Steps: positive-clip the profile, divide by the total positive mass, sum
    the entries inside ``lag_band``. The non-clip step is the same convention
    as the v1 LOLO -- negative degradations are not informative.

    Args:
        profile: Raw per-lag importance, length $L_{\max}+1$. ``nan`` /
            missing entries become 0.
        lag_band: True source-lag indices $\mathcal{L}^\star$ (or one band of
            a multi-band benchmark). Out-of-range entries are ignored.

    Returns:
        Dict with ``A_lag`` (normalised, length $L_{\max}+1$), ``A_lag_raw``
        (positive-clipped, length $L_{\max}+1$), ``total`` (denominator),
        ``band_mass`` (raw in-band sum), ``lag_mass`` (normalised in-band
        fraction, ``nan`` on a collapsed total).
    """
    arr = np.asarray(profile, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    raw = np.clip(arr, 0.0, None)
    total = float(raw.sum())
    L = raw.size
    band_idx = np.asarray(list(lag_band), dtype=int)
    band_idx = band_idx[(band_idx >= 0) & (band_idx < L)]
    band_mass = float(raw[band_idx].sum()) if band_idx.size else 0.0
    if total < 1e-12:
        norm = np.full(L, np.nan)
        lag_mass = float("nan")
    else:
        norm = raw / total
        lag_mass = float(norm[band_idx].sum()) if band_idx.size else 0.0
    return {
        "A_lag": norm.tolist(),
        "A_lag_raw": raw.tolist(),
        "total": total,
        "band_mass": band_mass,
        "lag_mass": lag_mass,
    }


def _resolve_lag_grid(
    max_lag: int,
    *,
    lag_band: Sequence[int],
    coarse_step: int = 5,
    fine_step: int = 1,
) -> List[int]:
    r"""Build the default lag grid for the sliding-window LOLO.

    Coarse-spaced over $[0, L_{\max}]$ for cost (every ``coarse_step`` lags)
    plus a refined stride inside the true band so the in-band shape is fully
    resolved. The lag ``0`` is always present (zero-offset sanity), as is
    ``max_lag`` (window edge).

    Args:
        max_lag: Maximum attention lag.
        lag_band: True lag band indices to refine over.
        coarse_step: Coarse stride.
        fine_step: Fine stride inside ``lag_band``.

    Returns:
        Sorted unique list of lag indices in $[0, L_{\max}]$.
    """
    coarse = list(range(0, max_lag + 1, max(1, int(coarse_step))))
    if max_lag not in coarse:
        coarse.append(int(max_lag))
    band = [int(ell) for ell in lag_band if 0 <= int(ell) <= max_lag]
    if band:
        lo, hi = min(band), max(band)
        coarse.extend(range(lo, hi + 1, max(1, int(fine_step))))
    return sorted(set(coarse))


@torch.no_grad()
def run_sliding_window_lolo(
    model: SeqVaeLagAttnV1,
    loader: Any,
    device: torch.device,
    *,
    meta: Dict[str, Any],
    warmup: int,
    max_lag: int,
    beta: float,
    lambda_full: float,
    lambda_base: float,
    batch_size: int,
    window_width: int = 10,
    lag_grid: Optional[Sequence[int]] = None,
    n_ablation_samples: Optional[int] = None,
    do_oob_probe: bool = True,
    seed: int = 0,
    coarse_step: int = 5,
    fine_step: int = 1,
) -> Dict[str, Any]:
    r"""Sliding-window leave-one-lag-out ablation (Sprint 4.1, V2-D5).

    For each lag $\ell$ in ``lag_grid``, replaces the source slice
    $[t_{\min}-\ell-w/2,\;t_{\max}-\ell+w/2)$ of ``u_stream`` with
    $\mathcal{N}(0,1)$ noise and measures the resulting per-anchor feat-loss
    degradation $\delta_\ell = \sum_\tau (\mathrm{MSE}_{\rm corrupt}(\tau) -
    \mathrm{MSE}_{\rm clean}(\tau))$. The window is the union of per-anchor
    width-$w$ windows centred on $t-\ell$ across the valid anchor range
    $[t_{\min}, t_{\max})$, which is one contiguous block -- so the routine
    needs **one forward pass per $\ell$**, not per anchor (cheap implementation
    per Sprint 4 decision 1).

    The reparameterisation RNG is paired between clean and corrupt passes so
    $\delta_\ell$ reflects the source perturbation, not sampling noise.

    Args:
        model: The trained model (``eval`` mode).
        loader: A test :class:`DataLoader`.
        device: Compute device.
        meta: The dataset ``meta.json`` (carries ``delay``, ``horizon``,
            ``true_lag_band``, ``informative_channels``, ``sequence_length``,
            ``clean_anchor_range``).
        warmup: Leading anchors excluded from the loss.
        max_lag: Maximum attention lag.
        beta: KL weight (forwarded to the correctness cross-check only).
        lambda_full: Full-forecast loss weight (cross-check only).
        lambda_base: Baseline-forecast loss weight (cross-check only).
        batch_size: Loader batch size (maps ``n_ablation_samples`` to a batch
            cap).
        window_width: Width $w$ of the corruption window centred on each
            anchor's source-lag $t-\ell$. Default 10 (typical autocorrelation
            scale $1/(1-\rho_u) \approx 20$--$200$ for v2 sources).
        lag_grid: Lag indices to probe. ``None`` -> :func:`_resolve_lag_grid`
            (coarse stride + fine stride inside the true band).
        n_ablation_samples: Optional cap on the samples used per pass.
        do_oob_probe: Whether to run the out-of-band sanity pass (corrupt the
            tail ``[T-w, T)`` -- no scored anchor depends on it).
        seed: Base seed for the corruption / reparam RNGs.
        coarse_step: Coarse-stride for the default lag grid.
        fine_step: Fine-stride inside the true band for the default lag grid.

    Returns:
        Dict with:

        * ``A_lag``      -- normalised $A_\ell$, length $L_{\max}+1$
                            (off-grid lags are 0 before normalisation).
        * ``A_lag_raw``  -- positive-clipped $[\delta_\ell]_+$ on the same axis.
        * ``A_overflow`` -- mass that landed outside $[0, L_{\max}]$ during
                            grid scattering (always 0 here since the grid is
                            in-range; kept for downstream compatibility).
        * ``lag_grid``        -- the lag indices that were probed.
        * ``delta_per_lag``    -- raw $\delta_\ell$ on the lag grid (length
                                  ``len(lag_grid)``).
        * ``mse_per_lag``      -- per-$\ell$ horizon-summed corrupt MSE
                                  (length ``len(lag_grid)``).
        * ``mse_clean_total``  -- horizon-summed clean MSE (scalar).
        * ``mse_clean_per_tau`` -- per-horizon clean MSE (length $H$).
        * ``window_width``     -- $w$ used.
        * ``window_per_lag``   -- the $(lo, hi)$ window tuple per probed lag.
        * ``lag_mass_lolo``    -- in-band fraction of $A_\ell$
                                  (``nan`` if total collapses).
        * ``total_delta``      -- $\sum_\ell [\delta_\ell]_+$.
        * ``delta_oob_max``    -- max $|\delta|$ over the OOB probe (``nan``
                                  if ``do_oob_probe=False``).
        * ``n_ablation_batches`` -- batch cap actually applied.
        * ``crosscheck_rel_err`` -- worst per-tau-MSE vs ``feat_loss`` error.
    """
    H = int(meta["horizon"])
    T = int(meta["sequence_length"])
    lag_band = np.asarray(list(meta["true_lag_band"]), dtype=int)
    clean_range = meta.get("clean_anchor_range", [warmup, T - H])
    t_lo = int(clean_range[0])
    t_hi = int(clean_range[1])
    if t_hi <= t_lo:
        t_lo, t_hi = int(warmup), int(T - H)

    max_batches: Optional[int] = None
    if n_ablation_samples is not None and batch_size > 0:
        max_batches = max(1, math.ceil(int(n_ablation_samples) / batch_size))

    if lag_grid is None:
        grid = _resolve_lag_grid(
            max_lag, lag_band=lag_band.tolist(),
            coarse_step=coarse_step, fine_step=fine_step,
        )
    else:
        grid = sorted({int(ell) for ell in lag_grid if 0 <= int(ell) <= max_lag})
    if not grid:
        raise ValueError(
            f"run_sliding_window_lolo: empty lag_grid (max_lag={max_lag})."
        )

    # --- correctness gate ----------------------------------------------------
    crosscheck = _crosscheck_per_tau(
        model, loader, device, warmup=warmup,
        beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
    )

    # --- clean baseline ------------------------------------------------------
    mse_clean = _per_tau_mse(
        model, loader, device, warmup=warmup, horizon=H,
        corrupt=None, seed=seed, max_batches=max_batches,
    )
    mse_clean_total = float(np.nansum(mse_clean))

    # --- per-lag corruption window: union of per-anchor windows --------------
    w = int(max(1, window_width))
    half_lo = w // 2
    half_hi = w - half_lo
    delta_per_lag: List[float] = []
    mse_per_lag: List[float] = []
    window_per_lag: List[Tuple[int, int]] = []
    A_raw = np.zeros(max_lag + 1, dtype=float)

    for ell in grid:
        lo = max(0, t_lo - int(ell) - half_lo)
        hi = min(T, t_hi - int(ell) + half_hi)
        if hi <= lo:
            window_per_lag.append((lo, lo))
            delta_per_lag.append(0.0)
            mse_per_lag.append(mse_clean_total)
            continue
        mse_corrupt = _per_tau_mse(
            model, loader, device, warmup=warmup, horizon=H,
            corrupt=(lo, hi), seed=seed, max_batches=max_batches,
        )
        mse_corrupt_total = float(np.nansum(mse_corrupt))
        d_ell = mse_corrupt_total - mse_clean_total
        window_per_lag.append((lo, hi))
        delta_per_lag.append(d_ell)
        mse_per_lag.append(mse_corrupt_total)
        A_raw[int(ell)] = max(0.0, d_ell)

    profile = compute_lag_mass_from_profile(
        A_raw.tolist(), lag_band=lag_band.tolist(),
    )

    # --- out-of-band probe: corrupt the source tail [T - w, T) ---------------
    # Source samples in [T - w, T) lie at lag <= w - 1 relative to anchors at
    # T - 1; for every clean anchor t < T - H this is outside [t - max_lag, t],
    # so no scored forecast depends on it. Delta should be ~ 0.
    delta_oob_max = float("nan")
    if do_oob_probe:
        oob_lo, oob_hi = max(0, T - w), T
        if oob_hi > oob_lo:
            mse_oob = _per_tau_mse(
                model, loader, device, warmup=warmup, horizon=H,
                corrupt=(oob_lo, oob_hi), seed=seed,
                max_batches=max_batches,
            )
            delta_oob_max = float(np.nanmax(np.abs(mse_oob - mse_clean)))

    return {
        "A_lag": profile["A_lag"],
        "A_lag_raw": profile["A_lag_raw"],
        "A_overflow": 0.0,
        "lag_grid": list(grid),
        "delta_per_lag": delta_per_lag,
        "mse_per_lag": mse_per_lag,
        "mse_clean_total": mse_clean_total,
        "mse_clean_per_tau": mse_clean.tolist(),
        "window_width": int(w),
        "window_per_lag": [list(ww) for ww in window_per_lag],
        "lag_mass_lolo": profile["lag_mass"],
        "total_delta": profile["total"],
        "delta_oob_max": delta_oob_max,
        "n_ablation_batches": max_batches,
        "crosscheck_rel_err": crosscheck,
    }


# =============================================================================
# Task 5.5 -- delay versus attention window
# =============================================================================

_LARGE_D_NOTE = (
    "Caveat 4.2: the SourceEncoder is a CNN+LSTM, so the source state h^u_t "
    "encodes all of U_{<=t} regardless of the (max_lag+1)-step attention "
    "window. When the true delay D exceeds max_lag the true lag band runs off "
    "the attention axis (A_overflow > 0), yet K (kld_per_t) can stay non-zero "
    "because the recurrent encoder still carries the delayed source. Measuring "
    "this directly needs a large-D (D > max_lag) dataset + checkpoint; this "
    "harness is ready for that run -- the deferred Task 5.5 verdict."
)


def _resolve_repr_delay(meta: Dict[str, Any]) -> int:
    r"""Resolve a representative scalar delay $D$ from a cache ``meta``.

    Handles every v2 layout: a fixed scalar ``delay`` (G2 fixed mode), a
    variable per-sample range (``delay = None`` + ``delay_max``), a multi-band
    ``delays`` list (returns the largest), the G3 reveal-lead ``delta``, and
    the legacy two-lag ``delay2``. Defaults to 0 when none is present.

    Args:
        meta: The dataset metadata dict.

    Returns:
        The representative delay (the upper end for variable / multi-band).
    """
    d = meta.get("delay")
    if d is None:
        delays = meta.get("delays") or []
        d = meta.get("delay_max")
        if d is None:
            d = meta.get("delta")
        if d is None:
            d = meta.get("delay2")
        if d is None:
            d = delays[-1] if delays else 0
    return int(d) if d is not None else 0


def _delay_window_report(
    meta: Dict[str, Any], max_lag: int, k_bar: float
) -> Dict[str, Any]:
    r"""Compare the true delay $D$ to the attention window (task 5.5).

    Args:
        meta: The dataset ``meta.json``.
        max_lag: Maximum attention lag ($L = L_{\max} + 1$).
        k_bar: The TE surrogate $\bar K$ for this checkpoint.

    Returns:
        Dict with ``delay``, ``max_lag``, ``attn_window_L``,
        ``lag_band_in_window``, ``lag_band_overflow_count``, ``k_bar`` and the
        Section 4.2 ``large_d_note``.
    """
    D = _resolve_repr_delay(meta)
    L = int(max_lag) + 1
    band = np.asarray(list(meta["true_lag_band"]), dtype=int)
    overflow = int(np.sum(band > max_lag))
    return {
        "delay": D,
        "max_lag": int(max_lag),
        "attn_window_L": L,
        "lag_band_in_window": bool(D <= L),
        "lag_band_overflow_count": overflow,
        "k_bar": float(k_bar),
        "large_d_note": _LARGE_D_NOTE,
    }


# =============================================================================
# Task 5.6 -- plots
# =============================================================================

def _band_spans(meta: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return the ``(lo, hi)`` span(s) of the true source-lag band(s).

    Splits ``meta["true_lag_band"]`` into maximal contiguous runs of integers.
    For a single-delay benchmark this returns one span; for the multi-delay
    ``G1_twoband`` (Sprint 4.5) it returns two non-contiguous spans
    automatically -- no extra metadata is required. Returns an empty list
    when ``true_lag_band`` is empty (e.g. ``G1-rev`` directionality control).

    Args:
        meta: The dataset ``meta.json``.

    Returns:
        A list of inclusive ``(lo, hi)`` integer spans, one per contiguous
        sub-band, in ascending order.
    """
    band = sorted({int(x) for x in meta.get("true_lag_band", [])})
    if not band:
        return []
    spans: List[Tuple[int, int]] = []
    run_lo = band[0]
    prev = run_lo
    for x in band[1:]:
        if x == prev + 1:
            prev = x
            continue
        spans.append((run_lo, prev))
        run_lo, prev = x, x
    spans.append((run_lo, prev))
    return spans


def _make_plots(
    collected: Dict[str, Any],
    lag_mass: Dict[str, Any],
    ablation: Dict[str, Any],
    meta: Dict[str, Any],
    out_dir: Path,
) -> None:
    r"""Render the three Phase-5 lag-recovery plots (task 5.6).

    Produces ``attn_heatmap.{pdf,png}`` (head-averaged attention, time x lag,
    with the true band(s) shaded), ``lolo_abar.{pdf,png}`` (the per-lag
    importance $A_\ell$ bar chart) and ``lag_profile.{pdf,png}`` (the 1-D
    ``te_lag_map`` lag profile and the per-horizon $\delta(\tau)$, drawn as a
    lag-aligned two-panel stack). The two-lag benchmark E shades two bands and
    shows an empty LOLO panel (its single-$D$ decomposition does not
    generalise -- LOLO is skipped for E). All figures use the shared
    publication style in :mod:`plot_style`.

    Args:
        collected: The :func:`collect_lag_tensors` output.
        lag_mass: The :func:`compute_lag_mass_attn` output.
        ablation: The :func:`run_sliding_window_lolo` output.
        meta: The dataset ``meta.json``.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()

    mean_alpha = np.asarray(collected["mean_alpha_tl"], dtype=float)  # (T, L)
    te_tl = np.asarray(collected["te_lag_map_tl"], dtype=float)
    T, L = mean_alpha.shape
    spans = _band_spans(meta)
    anchor_lo = int(lag_mass["anchor_lo"])
    anchor_hi = int(lag_mass["anchor_hi"])

    def _shade(ax, *, line: bool = False) -> None:
        """Shade every true lag band on ``ax`` (the source-lag x-axis)."""
        for i, (lo, hi) in enumerate(spans):
            ax.axvspan(
                lo - 0.5, hi + 0.5, color=ps.COLOR_VERMILLION, alpha=0.16,
                lw=0.0,
                label=(r"true band $\mathcal{L}^\star$" if i == 0 else None),
            )
            if line:
                for edge in (lo - 0.5, hi + 0.5):
                    ax.axvline(edge, color=ps.COLOR_VERMILLION, ls=":",
                               lw=0.8, alpha=0.7)

    # --- Plot 1: attention heatmap (time x lag) ---------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    im = ax.imshow(
        mean_alpha, aspect="auto", origin="lower",
        extent=(-0.5, L - 0.5, -0.5, T - 0.5), cmap="magma",
    )
    _shade(ax)
    for anchor in (anchor_lo, anchor_hi):
        ax.axhline(anchor, color="white", ls="--", lw=0.8, alpha=0.85)
    ax.set_title(r"head-averaged attention $\bar\alpha_{t,\ell}$")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel("anchor time $t$")
    if spans:
        ax.legend(loc="upper right")
    ps.style_axes(ax, grid="none")
    ps.add_colorbar(fig, im, ax, label=r"$\bar\alpha$")
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "attn_heatmap")

    # --- Plot 2: per-lag LOLO importance A_ell ----------------------------
    A = np.asarray(ablation.get("A_lag", []), dtype=float)
    w = int(ablation.get("window_width", 0) or 0)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if A.size == 0 or not np.isfinite(np.nansum(A)):
        ax.text(
            0.5, 0.5, "sliding-window LOLO collapsed\n"
            "(total positive delta < 1e-12)",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=ps.FONT_LEGEND,
            color=ps.COLOR_GRAY,
        )
        ax.set_title("sliding-window LOLO per-lag importance")
    else:
        lag_axis = np.arange(A.size)
        # Sprint 4.6: colour bars by band (one colour per span, +1 for
        # out-of-band) so two-band runs read off both bands at a glance.
        band_of_ell = np.full(A.size, -1, dtype=int)
        for i, (lo, hi) in enumerate(spans):
            in_i = (lag_axis >= lo) & (lag_axis <= hi)
            band_of_ell[in_i] = i
        band_colours = (
            ps.COLOR_VERMILLION,
            ps.COLOR_GREEN,
            ps.COLOR_PURPLE,
            ps.COLOR_SAGE,
        )
        colours = [
            band_colours[band_of_ell[ell] % len(band_colours)]
            if band_of_ell[ell] >= 0 else ps.COLOR_BLUE
            for ell in lag_axis
        ]
        ax.bar(lag_axis, np.nan_to_num(A, nan=0.0), width=1.0, color=colours)
        lm = ablation.get("lag_mass_lolo", float("nan"))
        title_band = (
            "  in-band = vermillion"
            if len(spans) <= 1
            else f"  bands = {len(spans)} (vermillion / green)"
        )
        ax.set_title(
            f"sliding-window LOLO per-lag importance $A_\\ell$  "
            f"(LagMass$_{{\\rm LOLO}}$={lm:.3f}, "
            f"$w$={w}){title_band}"
        )
        ax.set_xlabel(r"source lag $\ell$")
        ax.set_ylabel(r"$A_\ell$ (normalised)")
        ax.set_xlim(-0.5, A.size - 0.5)
        # Window-width annotation in the upper-right corner (Sprint 4.6).
        ax.text(
            0.99, 0.97,
            f"window width $w$ = {w} steps\n"
            f"|delta|$_{{\\rm OOB}}$ = "
            f"{ablation.get('delta_oob_max', float('nan')):.3g}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=ps.FONT_LEGEND - 1, color=ps.COLOR_GRAY,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": ps.COLOR_LIGHT_GRAY, "alpha": 0.85},
        )
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "lolo_abar")

    # --- Plot 3: lag profile (TE-map + attention) + delta_per_lag ---------
    fig, axes, _ = ps.stacked_figure([1.2, 1.0], width=8.0, hspace=0.45)
    lag_grid = np.arange(L)
    te_profile = te_tl[anchor_lo:anchor_hi, :].mean(axis=0)
    a_profile = mean_alpha[anchor_lo:anchor_hi, :].mean(axis=0)

    ax = axes[0]
    line_te = ax.plot(lag_grid, te_profile, color=ps.COLOR_BLUE, lw=1.2,
                      label=r"$\overline{\widetilde{\mathrm{TE}}}_\ell$")
    _shade(ax)
    ax.set_title("lag profile (mean over valid anchors)")
    ax.set_ylabel(r"$\widetilde{\mathrm{TE}}_\ell$", color=ps.COLOR_BLUE)
    ax.set_xlim(-0.5, L - 0.5)
    ax.tick_params(axis="y", colors=ps.COLOR_BLUE)
    ax.tick_params(labelbottom=False)
    ps.style_axes(ax)
    ax2 = ax.twinx()
    line_a = ax2.plot(lag_grid, a_profile, color=ps.COLOR_SKY, lw=1.0,
                      alpha=0.85, label=r"$\bar\alpha_\ell$")
    ax2.set_ylabel(r"$\bar\alpha_\ell$", color=ps.COLOR_SKY)
    ax2.tick_params(axis="y", colors=ps.COLOR_SKY)
    handles = line_te + line_a
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left")

    ax = axes[1]
    grid = np.asarray(ablation.get("lag_grid", []), dtype=float)
    delta = np.asarray(ablation.get("delta_per_lag", []), dtype=float)
    if grid.size == 0 or delta.size == 0:
        ax.text(
            0.5, 0.5, "sliding-window LOLO produced no per-lag deltas",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=ps.FONT_LEGEND, color=ps.COLOR_GRAY,
        )
        ax.set_title(r"LOLO per-lag degradation $\delta_\ell$")
    else:
        _shade(ax)
        ax.bar(grid, delta, width=1.0, color=ps.COLOR_ORANGE,
               label=r"$\delta_\ell$ ($w$=" f"{w}" r")")
        ax.axhline(0.0, color=ps.COLOR_GRAY, lw=0.8)
        ax.set_title(
            r"sliding-window LOLO per-lag degradation $\delta_\ell$"
        )
        ax.set_ylabel(
            r"$\delta_\ell = \sum_\tau (\mathrm{MSE}_{\rm corrupt}"
            r"-\mathrm{MSE}_{\rm clean})$"
        )
        ax.legend(loc="upper left")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_xlim(-0.5, L - 0.5)
    ps.style_axes(ax)
    ps.save_figure(fig, out_dir / "lag_profile")

    # --- Plot 4: LOLO vs attention overlay (Sprint 4.6) -------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    lag_axis = np.arange(L)
    if A.size > 0 and np.isfinite(np.nansum(A)):
        line_lolo = ax.plot(
            lag_axis, np.nan_to_num(A[:L], nan=0.0),
            color=ps.COLOR_BLUE, lw=1.4,
            label=r"$A_\ell$ (sliding-window LOLO)",
        )
    else:
        line_lolo = []
    _shade(ax)
    ax.set_ylabel(r"$A_\ell$ (LOLO, normalised)", color=ps.COLOR_BLUE)
    ax.tick_params(axis="y", colors=ps.COLOR_BLUE)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_xlabel(r"source lag $\ell$")

    ax2 = ax.twinx()
    # Normalise attention profile so the two curves share a comparable y-range
    # (LOLO's A_ell sums to 1; alpha sums to 1 per anchor but the per-anchor
    # mean has L-dependent scale). Z-score-free: just rescale so the lag-sum
    # is 1, mirroring the LOLO normalisation.
    a_norm = np.nan_to_num(a_profile, nan=0.0)
    a_total = float(a_norm.sum())
    if a_total > 1e-12:
        a_norm = a_norm / a_total
    line_attn = ax2.plot(
        lag_axis, a_norm, color=ps.COLOR_ORANGE, lw=1.2, alpha=0.9,
        label=r"$\bar\alpha_\ell$ (normalised attention)",
    )
    ax2.set_ylabel(
        r"$\bar\alpha_\ell$ (normalised)", color=ps.COLOR_ORANGE,
    )
    ax2.tick_params(axis="y", colors=ps.COLOR_ORANGE)

    handles = list(line_lolo) + list(line_attn)
    if handles:
        ax.legend(
            handles, [h.get_label() for h in handles], loc="upper left",
        )
    lm = ablation.get("lag_mass_lolo", float("nan"))
    ax.set_title(
        r"LOLO vs attention attribution on the same lag axis"
        f"  ($w$={w}, "
        f"LagMass$_{{\\rm LOLO}}$={lm:.3f})"
    )
    ax.text(
        0.99, 0.97,
        f"window width $w$ = {w} steps",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=ps.FONT_LEGEND - 1, color=ps.COLOR_GRAY,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": ps.COLOR_LIGHT_GRAY, "alpha": 0.85},
    )
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "lolo_vs_attn_overlay")


# =============================================================================
# Output: CSV / JSON
# =============================================================================

def write_summary_csv(row: Dict[str, Any], path: Path) -> None:
    """Write the one-row lag-recovery summary CSV (task 5.6).

    Args:
        row: The flat metrics row (the :data:`_SUMMARY_FIELDS` keys).
        path: Destination CSV path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({k: row.get(k) for k in _SUMMARY_FIELDS})


def write_metrics_json(payload: Dict[str, Any], path: Path) -> None:
    """Write the structured Phase-5 metrics JSON.

    Args:
        payload: The nested metrics dict assembled by
            :func:`analyze_lag_recovery`.
        path: Destination JSON path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _print_summary(
    row: Dict[str, Any],
    ablation: Dict[str, Any],
    two_band: Optional[Dict[str, Any]] = None,
) -> None:
    """Print the Phase-5 / 7.3 metrics block.

    Args:
        row: The flat metrics row.
        ablation: The :func:`run_sliding_window_lolo` output.
        two_band: The :func:`compute_two_band_mass_ratio` output for the
            two-lag benchmark E, or ``None``.
    """
    print(
        f"\n[lag-recovery] {row['run_tag']}  "
        f"(te_true={row['te_true']:.4g} nats, epoch={row['epoch']})\n"
        f"  Task 5.2  attention lag-mass : {row['lag_mass_attn']:.4f}  "
        f"(uniform {row['lag_mass_attn_uniform']:.4f}, "
        f"ratio {row['lag_mass_attn_ratio']:.3f})\n"
        f"  Task 5.3  peak-lag error     : mean {row['peak_lag_err_mean']:.3f}"
        f"  median {row['peak_lag_err_median']:.3f}  "
        f"in-band {row['peak_in_band_frac']:.3f}\n"
        f"  Sprint 4  sliding-LOLO       : LagMass_LOLO={row['lag_mass_lolo']:.4f}"
        f"  (w={row['window_width']}, total_delta={ablation['total_delta']:.4g},"
        f" oob |delta| max={row['delta_oob_max']:.4g},"
        f" cross-check {ablation['crosscheck_rel_err']:.2e})\n"
        f"  Task 5.5  delay {row['delay']} vs attn window {row['L']}  "
        f"K_bar={row['k_bar']:.5f}"
    )
    if two_band is not None:
        print(
            f"  Task 7.3  two-band mass      : "
            f"band1 {two_band['lag_mass_1']:.4f}  "
            f"band2 {two_band['lag_mass_2']:.4f}  "
            f"mass ratio {two_band['mass_ratio']:.3f}  "
            f"(TE ratio {two_band['te_ratio']:.3f}, "
            f"|err| {two_band['ratio_error']:.3f})"
        )


# =============================================================================
# Library entry point (Decision D9)
# =============================================================================

def analyze_lag_recovery(
    ckpt_path: Any,
    config: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    data_tag: Optional[str] = None,
    batch_size: Optional[int] = None,
    n_ablation_samples: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Run the Phase-5 lag-recovery analysis on one trained checkpoint.

    Resolves the checkpoint and its cached test split (mirroring
    :func:`evaluate_te.evaluate_checkpoint`), runs Tasks 5.1-5.6 and writes
    ``summary.csv`` / ``metrics.json`` / the three plots under
    ``results/<benchmark>/lag_recovery/``.

    Args:
        ckpt_path: Path to a ``.ckpt`` written by :func:`train_minimal.train`.
        config: The parsed ``config_synth.yaml``.
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        data_tag: Test-split tag. Defaults to the tag the checkpoint trained on.
        batch_size: Inference batch size. Defaults to the ``lag_recovery``
            config value, else the checkpoint's training batch size.
        n_ablation_samples: Cap on the samples used per LOLO pass. Defaults to
            the ``lag_recovery`` config value.

    Returns:
        Dict with ``row`` (the flat metrics row), ``metrics`` (the JSON
        payload) and ``out_dir``.
    """
    device = device or tm.resolve_device(config["runtime"])
    model, ckpt = ev.load_eval_checkpoint(ckpt_path, device)

    data_meta: Dict[str, Any] = ckpt.get("data_meta", {}) or {}
    ckpt_config: Dict[str, Any] = ckpt.get("config", {}) or {}
    ckpt_exp = ckpt_config.get("experiment", {})
    lr_cfg: Dict[str, Any] = config.get("lag_recovery", {}) or {}

    benchmark = str(ckpt_exp.get("benchmark", config["experiment"]["benchmark"]))
    tag = data_tag or data_meta.get("tag") or ckpt_exp.get("tag")
    if tag is None:
        raise ValueError(
            f"cannot resolve a test-split tag for {ckpt_path}: the checkpoint "
            f"carries no data_meta['tag']; pass an explicit data_tag."
        )
    if batch_size is None:
        batch_size = lr_cfg.get("batch_size") or int(
            ckpt_config.get("optim", {}).get(
                "batch_size", config["optim"]["batch_size"]
            )
        )
    batch_size = int(batch_size)

    test_loader, test_meta = ev.make_test_loader(
        config, benchmark, str(tag), batch_size
    )

    # Loss settings: prefer what the checkpoint trained with (cross-check only).
    loss_settings = ckpt.get("loss_settings", {}) or {}
    beta = float(loss_settings.get("beta", config["loss"]["kld_beta"]))
    lambda_full = float(
        loss_settings.get("lambda_full", config["loss"]["lambda_full"])
    )
    lambda_base = float(
        loss_settings.get("lambda_base", config["loss"]["lambda_base"])
    )

    warmup = int(getattr(model, "warmup_period", 0) or 0)
    n_attn = lr_cfg.get("n_attention_samples")
    n_abl = (
        n_ablation_samples
        if n_ablation_samples is not None
        else lr_cfg.get("n_ablation_samples", 512)
    )
    do_oob = bool(lr_cfg.get("do_oob_probe", True))
    ablation_seed = int(lr_cfg.get("ablation_seed", 0))
    anchor_window = str(lr_cfg.get("anchor_window", "clean")).lower()
    window_width = int(lr_cfg.get("window_width", 10))
    coarse_step = int(lr_cfg.get("lag_grid_step", 5))
    fine_step = int(lr_cfg.get("fine_lag_grid_step", 1))

    # --- Task 5.1: attention / lag-map tensors -------------------------------
    max_attn_batches = (
        max(1, math.ceil(int(n_attn) / batch_size)) if n_attn else None
    )
    collected = collect_lag_tensors(
        model, test_loader, device, warmup=warmup, max_batches=max_attn_batches,
    )
    T, L = collected["T"], collected["L"]
    max_lag = L - 1

    # Ground-truth lag structure from the dataset meta.
    lag_band = [int(x) for x in test_meta["true_lag_band"]]
    clean_range = test_meta.get("clean_anchor_range", [warmup, T])
    if anchor_window == "warmup":
        anchor_lo, anchor_hi = warmup, T
    else:
        anchor_lo, anchor_hi = int(clean_range[0]), int(clean_range[1])

    # --- Task 5.2: attention lag-mass (headline + warmup window) -------------
    te_tl = collected["te_lag_map_tl"]
    lag_mass = compute_lag_mass_attn(
        te_tl, lag_band=lag_band, anchor_lo=anchor_lo, anchor_hi=anchor_hi,
        max_lag=max_lag,
    )
    lag_mass_warmup = compute_lag_mass_attn(
        te_tl, lag_band=lag_band, anchor_lo=warmup, anchor_hi=T,
        max_lag=max_lag,
    )

    # --- Task 5.3: peak-lag error over the headline anchor window ------------
    peak = compute_peak_lag_error(
        te_tl[anchor_lo:anchor_hi, :], lag_band=lag_band,
    )

    # --- Sprint 4.1: sliding-window LOLO (replaces v1 run_lag_ablation) ------
    k_bar = tm.compute_kbar(model, test_loader, device)
    ablation = run_sliding_window_lolo(
        model, test_loader, device, meta=test_meta, warmup=warmup,
        max_lag=max_lag, beta=beta, lambda_full=lambda_full,
        lambda_base=lambda_base, batch_size=batch_size,
        window_width=window_width,
        n_ablation_samples=n_abl, do_oob_probe=do_oob, seed=ablation_seed,
        coarse_step=coarse_step, fine_step=fine_step,
    )

    te_true = float(
        test_meta.get("te_true", data_meta.get("te_true", float("nan")))
    )

    # --- Sprint 4.5: two-band wiring (multi-delay benchmarks like G1_twoband)
    spans = _band_spans(test_meta)
    two_band: Optional[Dict[str, Any]] = None
    lolo_band1: Optional[Dict[str, Any]] = None
    lolo_band2: Optional[Dict[str, Any]] = None
    if len(spans) >= 2:
        # Take the first two non-contiguous bands; ignore any beyond (the v2
        # multi-delay benchmark is two-band by construction).
        (b1_lo, b1_hi), (b2_lo, b2_hi) = spans[0], spans[1]
        band_1 = list(range(b1_lo, b1_hi + 1))
        band_2 = list(range(b2_lo, b2_hi + 1))
        # Per-band TE split: prefer generator-provided ``te_true_band1`` /
        # ``te_true_band2`` if present. The symmetric (equal-share) prior is
        # the safe fallback -- the ``mass_ratio`` diagnostic still flags
        # gross imbalance even when ``te_ratio = 1``.
        te_band_1 = float(test_meta.get("te_true_band1", te_true / 2.0))
        te_band_2 = float(test_meta.get("te_true_band2", te_true / 2.0))
        two_band = compute_two_band_mass_ratio(
            te_tl, lag_band_1=band_1, lag_band_2=band_2,
            anchor_lo=anchor_lo, anchor_hi=anchor_hi, max_lag=max_lag,
            te_true_1=te_band_1, te_true_2=te_band_2,
        )
        # LOLO per-band masses (shared denominator: total positive LOLO mass).
        lolo_band1 = compute_lag_mass_from_profile(
            ablation["A_lag_raw"], lag_band=band_1,
        )
        lolo_band2 = compute_lag_mass_from_profile(
            ablation["A_lag_raw"], lag_band=band_2,
        )

    # --- Task 5.5: delay versus attention window -----------------------------
    dwin = _delay_window_report(test_meta, max_lag, k_bar)
    run_tag = Path(ckpt_path).resolve().parent.name

    # v2 sweep knobs (V2-D6 / V2-D7): G1 -> B_y, G2 -> c, G3 -> p_switch.
    # Pull from the cache's meta; whichever the active benchmark does not use
    # stays None. Mirrors :data:`evaluate_te._SUMMARY_FIELDS`.
    def _meta_scalar(key: str) -> Optional[float]:
        val = test_meta.get(key, data_meta.get(key))
        if val is None:
            return None
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    row: Dict[str, Any] = {
        "run_tag": run_tag,
        "data_tag": str(tag),
        "benchmark": benchmark,
        "B_y": _meta_scalar("B_y"),
        "c": _meta_scalar("c"),
        "p_switch": _meta_scalar("p_switch"),
        "M": data_meta.get("M", test_meta.get("M")),
        "delay": _resolve_repr_delay(test_meta),
        "horizon": int(test_meta["horizon"]),
        "te_true": te_true,
        "warmup": warmup,
        "n_test": int(len(test_loader.dataset)),
        "L": int(L),
        "lag_band_lo": int(min(lag_band)) if lag_band else None,
        "lag_band_hi": int(max(lag_band)) if lag_band else None,
        "lag_mass_attn": lag_mass["lag_mass_attn"],
        "lag_mass_attn_uniform": lag_mass["uniform_baseline"],
        "lag_mass_attn_ratio": lag_mass["ratio_to_uniform"],
        "peak_lag_err_mean": peak["peak_lag_err_mean"],
        "peak_lag_err_median": peak["peak_lag_err_median"],
        "peak_in_band_frac": peak["peak_in_band_frac"],
        "lag_mass_lolo": ablation["lag_mass_lolo"],
        "window_width": ablation["window_width"],
        "delta_oob_max": ablation["delta_oob_max"],
        "lag_mass_band1": two_band["lag_mass_1"] if two_band else None,
        "lag_mass_band2": two_band["lag_mass_2"] if two_band else None,
        "lag_mass_ratio": two_band["mass_ratio"] if two_band else None,
        "lag_mass_te_ratio": two_band["te_ratio"] if two_band else None,
        "lag_mass_ratio_err": two_band["ratio_error"] if two_band else None,
        "lag_mass_lolo_band1": lolo_band1["lag_mass"] if lolo_band1 else None,
        "lag_mass_lolo_band2": lolo_band2["lag_mass"] if lolo_band2 else None,
        "k_bar": float(k_bar),
        "epoch": ckpt.get("epoch"),
        "ckpt_path": str(Path(ckpt_path).resolve()),
    }

    # te_lag_map / attention lag profiles over the headline anchor window.
    te_profile = te_tl[anchor_lo:anchor_hi, :].mean(axis=0)
    a_profile = collected["mean_alpha_tl"][anchor_lo:anchor_hi, :].mean(axis=0)

    metrics: Dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "run_tag": run_tag,
        "data_tag": str(tag),
        "benchmark": benchmark,
        "ground_truth": {
            "delay": row["delay"],
            "horizon": int(test_meta["horizon"]),
            "te_true": te_true,
            "true_lag_band": lag_band,
            "lag_band_spans": [list(s) for s in spans],
            "lag_band_centre": peak["peak_lag_centre"],
            "informative_channels": test_meta.get("informative_channels"),
            "clean_anchor_range": [int(clean_range[0]), int(clean_range[1])],
            "max_lag": int(max_lag),
            "L": int(L),
            "warmup": warmup,
        },
        "task_5_2_lag_mass_attn": {
            "anchor_window": anchor_window,
            "lag_mass_attn": lag_mass["lag_mass_attn"],
            "uniform_baseline": lag_mass["uniform_baseline"],
            "ratio_to_uniform": lag_mass["ratio_to_uniform"],
            "anchor_lo": lag_mass["anchor_lo"],
            "anchor_hi": lag_mass["anchor_hi"],
            "total_mass": lag_mass["total_mass"],
            "band_mass": lag_mass["band_mass"],
            "lag_mass_attn_warmup_window": lag_mass_warmup["lag_mass_attn"],
        },
        "task_5_3_peak_lag": {
            "peak_lag_err_mean": peak["peak_lag_err_mean"],
            "peak_lag_err_median": peak["peak_lag_err_median"],
            "peak_in_band_frac": peak["peak_in_band_frac"],
            "peak_lag_centre": peak["peak_lag_centre"],
            "n_anchors": peak["n_anchors"],
            "peak_lag_hist": peak["peak_lag_hist"],
        },
        "sprint_4_1_sliding_lolo": {
            "lag_mass_lolo": ablation["lag_mass_lolo"],
            "total_delta": ablation["total_delta"],
            "A_overflow": ablation["A_overflow"],
            "delta_oob_max": ablation["delta_oob_max"],
            "crosscheck_rel_err": ablation["crosscheck_rel_err"],
            "n_ablation_batches": ablation["n_ablation_batches"],
            "window_width": ablation["window_width"],
            "lag_grid": ablation["lag_grid"],
            "delta_per_lag": ablation["delta_per_lag"],
            "mse_per_lag": ablation["mse_per_lag"],
            "mse_clean_total": ablation["mse_clean_total"],
            "mse_clean_per_tau": ablation["mse_clean_per_tau"],
            "window_per_lag": ablation["window_per_lag"],
            "A_lag": ablation["A_lag"],
            "A_lag_raw": ablation["A_lag_raw"],
        },
        "task_5_5_delay_vs_max_lag": dwin,
        "per_anchor": {
            "lag_profile": te_profile.tolist(),
            "attn_lag_profile": a_profile.tolist(),
            "kld_per_t_mean": collected["kld_per_t_t"].tolist(),
        },
    }
    if two_band is not None:
        metrics["sprint_4_5_two_band_attn"] = two_band
        # Backwards-compat alias for final_report.py (which reads the legacy
        # v1 key); the new sprint-named key above is the canonical one.
        metrics["task_7_3_two_band"] = two_band
    if lolo_band1 is not None and lolo_band2 is not None:
        metrics["sprint_4_5_two_band_lolo"] = {
            "band1": lolo_band1, "band2": lolo_band2,
        }
    # Backwards-compat alias for final_report.py's lag-mass panel.
    metrics["task_5_4_lolo"] = metrics["sprint_4_1_sliding_lolo"]

    out_dir = _lag_out_dir(config, benchmark)
    write_summary_csv(row, out_dir / "summary.csv")
    write_metrics_json(metrics, out_dir / "metrics.json")
    _make_plots(collected, lag_mass, ablation, test_meta, out_dir)
    _print_summary(row, ablation, two_band)
    print(f"[done] lag-recovery analysis -> {out_dir}")
    return {"row": row, "metrics": metrics, "out_dir": str(out_dir)}


# =============================================================================
# Sprint 4.4 -- window-width sweep
# =============================================================================

def _select_window_width(
    widths: Sequence[int], lag_masses: Sequence[float], *, frac: float = 0.95
) -> int:
    r"""Smallest $w$ such that ``LagMass_LOLO(w) >= frac * max(LagMass_LOLO)``.

    "Smallest" = most lag-localising; "above ``frac`` of the peak" = not
    sacrificing measurable mass. Sprint 4.4 default ``frac=0.95``.

    Args:
        widths: The width grid (must align with ``lag_masses``).
        lag_masses: ``LagMass_LOLO`` per width. ``nan`` entries are skipped.
        frac: Acceptance threshold relative to the peak mass.

    Returns:
        The chosen width. If every entry is ``nan``, returns ``widths[0]``.
    """
    arr_w = [int(w) for w in widths]
    arr_m = [float(m) for m in lag_masses]
    finite = [(w, m) for w, m in zip(arr_w, arr_m) if np.isfinite(m)]
    if not finite:
        return arr_w[0]
    peak = max(m for _, m in finite)
    threshold = frac * peak
    eligible = sorted(w for w, m in finite if m >= threshold)
    return int(eligible[0]) if eligible else int(finite[0][0])


def sweep_window_widths(
    ckpt_path: Any,
    config: Dict[str, Any],
    *,
    widths: Sequence[int] = (1, 5, 10, 20),
    device: Optional[torch.device] = None,
    data_tag: Optional[str] = None,
    batch_size: Optional[int] = None,
    n_ablation_samples: Optional[int] = None,
    selection_frac: float = 0.95,
) -> Dict[str, Any]:
    r"""Sweep the sliding-window LOLO across ``widths`` (Sprint 4.4).

    Loads the checkpoint and test loader **once**, then re-runs
    :func:`run_sliding_window_lolo` for each $w \in$ ``widths`` -- reusing the
    same model, loader, lag grid and clean baseline (the cost dominator).
    Writes ``lolo_width_sweep.csv`` (one row per width: ``window_width``,
    ``lag_mass_lolo``, ``total_delta``, ``peak_lag``, ``delta_oob_max``,
    ``n_ablation_batches``) and ``lolo_width_sweep.pdf`` (two panels:
    ``LagMass_LOLO`` vs $w$ with the chosen width highlighted, and the per-
    width $A_\ell$ profiles overlaid with the true band shaded).

    The chosen width is reported in the JSON summary but **not** written back
    into ``config_synth.yaml`` -- the user inspects the plot and edits the
    config manually (V2-D8 / plan-mode discipline).

    Args:
        ckpt_path: Path to a ``.ckpt`` written by :func:`train_minimal.train`.
        config: The parsed ``config_synth.yaml``.
        widths: Window-width grid.
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        data_tag: Test-split tag. Defaults to the checkpoint's training tag.
        batch_size: Inference batch size.
        n_ablation_samples: Cap on samples per LOLO pass.
        selection_frac: Threshold for :func:`_select_window_width`.

    Returns:
        Dict with ``widths``, ``per_width`` (list of result dicts, one per
        $w$), ``lag_mass_lolo`` (list aligned with ``widths``),
        ``chosen_width``, ``out_dir`` and ``summary_csv`` / ``summary_pdf``
        paths.
    """
    widths = sorted({int(w) for w in widths if int(w) >= 1})
    if not widths:
        raise ValueError("sweep_window_widths: empty `widths` list.")

    device = device or tm.resolve_device(config["runtime"])
    model, ckpt = ev.load_eval_checkpoint(ckpt_path, device)

    data_meta: Dict[str, Any] = ckpt.get("data_meta", {}) or {}
    ckpt_config: Dict[str, Any] = ckpt.get("config", {}) or {}
    ckpt_exp = ckpt_config.get("experiment", {})
    lr_cfg: Dict[str, Any] = config.get("lag_recovery", {}) or {}

    benchmark = str(ckpt_exp.get("benchmark", config["experiment"]["benchmark"]))
    tag = data_tag or data_meta.get("tag") or ckpt_exp.get("tag")
    if tag is None:
        raise ValueError(
            f"cannot resolve a test-split tag for {ckpt_path}: the checkpoint "
            f"carries no data_meta['tag']; pass an explicit data_tag."
        )
    if batch_size is None:
        batch_size = lr_cfg.get("batch_size") or int(
            ckpt_config.get("optim", {}).get(
                "batch_size", config["optim"]["batch_size"]
            )
        )
    batch_size = int(batch_size)

    test_loader, test_meta = ev.make_test_loader(
        config, benchmark, str(tag), batch_size
    )

    loss_settings = ckpt.get("loss_settings", {}) or {}
    beta = float(loss_settings.get("beta", config["loss"]["kld_beta"]))
    lambda_full = float(
        loss_settings.get("lambda_full", config["loss"]["lambda_full"])
    )
    lambda_base = float(
        loss_settings.get("lambda_base", config["loss"]["lambda_base"])
    )

    warmup = int(getattr(model, "warmup_period", 0) or 0)
    max_lag = int(getattr(model, "max_lag", 90))
    n_abl = (
        n_ablation_samples
        if n_ablation_samples is not None
        else lr_cfg.get("n_ablation_samples", 512)
    )
    do_oob = bool(lr_cfg.get("do_oob_probe", True))
    ablation_seed = int(lr_cfg.get("ablation_seed", 0))
    coarse_step = int(lr_cfg.get("lag_grid_step", 5))
    fine_step = int(lr_cfg.get("fine_lag_grid_step", 1))

    per_width: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for w in widths:
        result = run_sliding_window_lolo(
            model, test_loader, device, meta=test_meta, warmup=warmup,
            max_lag=max_lag, beta=beta, lambda_full=lambda_full,
            lambda_base=lambda_base, batch_size=batch_size,
            window_width=int(w), n_ablation_samples=n_abl,
            do_oob_probe=do_oob, seed=ablation_seed,
            coarse_step=coarse_step, fine_step=fine_step,
        )
        per_width.append(result)
        A_lag = np.nan_to_num(
            np.asarray(result["A_lag"], dtype=float), nan=0.0,
        )
        peak_lag = int(np.argmax(A_lag)) if A_lag.size else -1
        rows.append({
            "window_width": int(w),
            "lag_mass_lolo": result["lag_mass_lolo"],
            "total_delta": result["total_delta"],
            "peak_lag": peak_lag,
            "delta_oob_max": result["delta_oob_max"],
            "n_ablation_batches": result["n_ablation_batches"],
        })

    lag_masses = [r["lag_mass_lolo"] for r in rows]
    chosen_width = _select_window_width(
        widths, lag_masses, frac=selection_frac,
    )

    out_dir = _lag_out_dir(config, benchmark)
    csv_path = out_dir / "lolo_width_sweep.csv"
    fields = [
        "window_width", "lag_mass_lolo", "total_delta",
        "peak_lag", "delta_oob_max", "n_ablation_batches",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    pdf_path = _plot_width_sweep(
        widths, rows, per_width, test_meta, chosen_width,
        max_lag=max_lag, out_dir=out_dir,
    )

    print(
        f"\n[width sweep] {benchmark}  widths={widths}  "
        f"chosen w={chosen_width} (>= {selection_frac:.0%} of peak)\n"
        f"  CSV -> {csv_path}\n  PDF -> {pdf_path}"
    )
    return {
        "widths": list(widths),
        "per_width": per_width,
        "lag_mass_lolo": lag_masses,
        "chosen_width": int(chosen_width),
        "selection_frac": float(selection_frac),
        "out_dir": str(out_dir),
        "summary_csv": str(csv_path),
        "summary_pdf": str(pdf_path),
    }


def _plot_width_sweep(
    widths: Sequence[int],
    rows: Sequence[Dict[str, Any]],
    per_width: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    chosen_width: int,
    *,
    max_lag: int,
    out_dir: Path,
) -> Path:
    r"""Render ``lolo_width_sweep.{pdf,png}`` (Sprint 4.4).

    Panel 1: ``LagMass_LOLO`` vs $w$ as a line+marker plot, chosen width
    highlighted. Panel 2: $A_\ell$ profiles overlaid (one curve per $w$,
    faint-to-bold colour ramp). True band(s) shaded via :func:`_band_spans`.

    Args:
        widths: The width grid.
        rows: One :func:`sweep_window_widths` summary row per width.
        per_width: Full LOLO result dict per width.
        meta: Dataset ``meta.json`` (for shading bands).
        chosen_width: Width selected by :func:`_select_window_width`.
        max_lag: Maximum attention lag.
        out_dir: Destination directory.

    Returns:
        Path of the saved PDF (PNG is also written alongside).
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()
    spans = _band_spans(meta)
    fig, axes, _ = ps.stacked_figure([1.0, 1.2], width=8.0, hspace=0.45)

    # Panel 1 -- LagMass_LOLO vs window width.
    ax = axes[0]
    masses = [float(r["lag_mass_lolo"]) for r in rows]
    ax.plot(
        list(widths), masses, marker="o", color=ps.COLOR_BLUE, lw=1.4,
        label=r"LagMass$_{\rm LOLO}$",
    )
    # Highlight the chosen width.
    chosen_idx = list(widths).index(int(chosen_width))
    ax.plot(
        [chosen_width], [masses[chosen_idx]],
        marker="o", markersize=10, mfc="none", mec=ps.COLOR_VERMILLION,
        mew=1.6, label=f"chosen $w^*$ = {chosen_width}",
    )
    ax.set_title(
        r"sliding-window LOLO mass vs window width "
        r"($\mathcal{L}^\star$ in-band fraction)"
    )
    ax.set_ylabel(r"LagMass$_{\rm LOLO}(w)$")
    ax.set_xlabel(r"window width $w$ (time steps)")
    ax.legend(loc="best")
    ps.style_axes(ax)

    # Panel 2 -- per-width A_ell profiles overlaid.
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    n = max(1, len(per_width))
    L = int(max_lag) + 1
    lag_axis = np.arange(L)
    for i, (w, result) in enumerate(zip(widths, per_width)):
        A = np.nan_to_num(
            np.asarray(result["A_lag"], dtype=float), nan=0.0,
        )
        if A.size < L:
            A = np.pad(A, (0, L - A.size))
        colour = cmap((i + 1) / (n + 1))
        ax.plot(
            lag_axis, A[:L],
            color=colour, lw=1.2 if int(w) == int(chosen_width) else 0.9,
            alpha=0.95 if int(w) == int(chosen_width) else 0.7,
            label=f"$w$ = {w}{' *' if int(w) == int(chosen_width) else ''}",
        )
    for i, (lo, hi) in enumerate(spans):
        ax.axvspan(
            lo - 0.5, hi + 0.5, color=ps.COLOR_VERMILLION, alpha=0.14,
            lw=0.0,
            label=(r"true band $\mathcal{L}^\star$" if i == 0 else None),
        )
    ax.set_title(r"$A_\ell$ profile per window width")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel(r"$A_\ell$ (normalised)")
    ax.set_xlim(-0.5, L - 0.5)
    ax.legend(loc="best", fontsize=ps.FONT_LEGEND - 1.5)
    ps.style_axes(ax)
    ps.save_figure(fig, out_dir / "lolo_width_sweep")
    return out_dir / "lolo_width_sweep.pdf"


# =============================================================================
# Overrides + dispatch
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the config-level overrides onto ``config`` in place.

    Only ``benchmark`` / ``device`` / ``seed`` are config fields; the rest
    (``checkpoint``, ``data_tag``, ``batch_size``, ``n_ablation_samples``) are
    passed as call arguments by :func:`_dispatch` and are ignored here.

    Args:
        config: The config dict (mutated in place).
        overrides: Flat ``{key: value}`` overrides; ``None`` values ignored.

    Returns:
        The same ``config`` dict.
    """
    if overrides.get("benchmark") is not None:
        config["experiment"]["benchmark"] = overrides["benchmark"]
    if overrides.get("device") is not None:
        config["runtime"]["device"] = overrides["device"]
    if overrides.get("seed") is not None:
        config["experiment"]["seed"] = overrides["seed"]
    # data_dir / results_dir overrides -> config["paths"] (None -> YAML default).
    tm.apply_path_overrides(config, overrides)
    return config


def _parse_widths(value: Any) -> Sequence[int]:
    """Parse a CLI / config widths value into a list of positive ints.

    Accepts a comma-/space-separated string (``"1,5,10,20"`` or
    ``"1 5 10 20"``), a list/tuple of ints, or ``None`` (returns the Sprint
    4.4 default).

    Args:
        value: The raw widths spec.

    Returns:
        Sorted unique list of widths $\\ge 1$.
    """
    if value is None:
        return (1, 5, 10, 20)
    if isinstance(value, (list, tuple)):
        return [int(w) for w in value if int(w) >= 1]
    s = str(value).strip().strip("[](){}")
    parts = [p for p in s.replace(",", " ").split() if p]
    return [int(p) for p in parts if int(p) >= 1]


def _dispatch(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve overrides, seed, device and run the analysis.

    Dispatches on ``overrides['mode']``: ``"analyze"`` (default) ->
    :func:`analyze_lag_recovery`; ``"width_sweep"`` ->
    :func:`sweep_window_widths`.

    Args:
        config: The parsed ``config_synth.yaml``.
        overrides: Flat overrides (from ``vars(args)`` or ``RUN_CONFIG``).

    Returns:
        The result dict of the chosen entry point.

    Raises:
        ValueError: If no checkpoint is supplied or ``mode`` is unknown.
    """
    config = deepcopy(config)
    _apply_overrides(config, overrides)
    tm.set_seed(int(config["experiment"].get("seed", 0)))
    device = tm.resolve_device(config["runtime"])

    ckpt = overrides.get("checkpoint")
    if not ckpt:
        raise ValueError(
            "a checkpoint is required -- pass --checkpoint PATH (CLI) or set "
            "RUN_CONFIG['checkpoint'] (edit-and-run)."
        )

    if overrides.get("window_width") is not None:
        config.setdefault("lag_recovery", {})["window_width"] = int(
            overrides["window_width"]
        )

    mode = str(overrides.get("mode") or "analyze").lower()
    if mode == "analyze":
        return analyze_lag_recovery(
            ckpt, config, device=device,
            data_tag=overrides.get("data_tag"),
            batch_size=overrides.get("batch_size"),
            n_ablation_samples=overrides.get("n_ablation_samples"),
        )
    if mode == "width_sweep":
        widths_arg = overrides.get("widths")
        if widths_arg is None:
            widths_arg = (config.get("lag_recovery") or {}).get("window_widths")
        widths = _parse_widths(widths_arg)
        return sweep_window_widths(
            ckpt, config, widths=widths, device=device,
            data_tag=overrides.get("data_tag"),
            batch_size=overrides.get("batch_size"),
            n_ablation_samples=overrides.get("n_ablation_samples"),
        )
    raise ValueError(
        f"unknown --mode {mode!r}; expected 'analyze' or 'width_sweep'."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`. Every flag defaults to ``None``
        (fall back to ``config_synth.yaml`` / the checkpoint).
    """
    p = argparse.ArgumentParser(
        description="Lag-recovery harness for SeqVaeLagAttnV1 on synthetic "
                    "benchmark data (Phase 5)."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="path to the .ckpt to analyse (required)",
    )
    p.add_argument(
        "--data-tag", type=str, default=None, dest="data_tag",
        help="test-split tag (defaults to the checkpoint's training tag)",
    )
    p.add_argument(
        "--benchmark", type=str, default=None,
        help="override experiment.benchmark",
    )
    p.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="inference batch size (defaults to the checkpoint's value)",
    )
    p.add_argument(
        "--n-ablation-samples", type=int, default=None,
        dest="n_ablation_samples",
        help="cap on the samples used per LOLO pass (Sprint 4.1)",
    )
    p.add_argument(
        "--mode", type=str, default="analyze",
        choices=("analyze", "width_sweep"),
        help="'analyze' (default, single-w LOLO + headline metrics) or "
             "'width_sweep' (LOLO across multiple window widths; Sprint 4.4).",
    )
    p.add_argument(
        "--widths", type=str, default=None,
        help="comma- or space-separated widths for --mode width_sweep "
             "(e.g. \"1,5,10,20\"). Defaults to config lag_recovery.window_widths.",
    )
    p.add_argument(
        "--window-width", type=int, default=None, dest="window_width",
        help="override config lag_recovery.window_width for --mode analyze.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override experiment.seed",
    )
    p.add_argument(
        "--data-dir", type=str, default=None, dest="data_dir",
        help="override paths.data_dir (absolute/relative path, ~, or $VAR); "
             "None -> config paths.data_dir",
    )
    p.add_argument(
        "--results-dir", type=str, default=None, dest="results_dir",
        help="override paths.results_dir (same format as --data-dir); "
             "None -> config paths.results_dir",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse args, load config, dispatch.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    config = tm.load_config(args.config)
    _dispatch(config, vars(args))


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision D9)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly;
    #                      no terminal flags required.
    #
    # Every key in RUN_CONFIG mirrors a CLI flag and is forwarded to
    # `_dispatch`; `None` means "fall back to config_synth.yaml".
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        # which checkpoint to analyse.
        "checkpoint": str(
            _EXPERIMENT_DIR / "results" / "G1" / "G1_baseline" / "final.ckpt"
        ),
        "data_tag": None,            # None -> the checkpoint's own tag
        "benchmark": None,           # None -> config experiment.benchmark
        "batch_size": None,          # None -> checkpoint's batch size
        "n_ablation_samples": None,  # None -> config lag_recovery value
        "device": None,              # None -> config runtime.device
        "seed": None,                # None -> config experiment.seed
        # Sprint 4 additions:
        "mode": "analyze",           # "analyze" | "width_sweep"
        "widths": None,              # None -> config lag_recovery.window_widths
        "window_width": None,        # None -> config lag_recovery.window_width
        "data_dir": None,            # None -> config paths.data_dir
        "results_dir": None,         # None -> config paths.results_dir
    }

    if len(sys.argv) > 1:
        main()                       # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["lag_recovery"]["n_ablation_samples"] = 256
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
