r"""Lag-recovery metrics -- does the model find the true source lag (Phase 5).

Loads a trained :class:`SeqVaeLagAttnV1` checkpoint and quantifies whether the
model localises transfer to the true source-lag band
$\mathcal{L}^\star = \{D-H,\dots,D-1\}$, using three complementary measures
(``synthetic_te_validation_plan.md`` Phase 5):

    Task 5.2 -- attention lag-mass: the fraction of the ``te_lag_map`` mass
        $\widetilde{\mathrm{TE}}_{t,\ell} = K_t\,\bar\alpha_{t,\ell}$ that lands
        inside $\mathcal{L}^\star$, over valid anchors, compared to the uniform
        baseline $|\mathcal{L}^\star| / (L_{\max} + 1)$.
    Task 5.3 -- peak-lag error: the per-anchor $\arg\max_\ell$ of the lag map
        versus the band centre, plus the in-band fraction.
    Task 5.4 -- input-level leave-one-lag-out (faithful LOLO, Decision D5): the
        recurrent ``SourceEncoder`` makes internal lag-memory masking
        non-causal, so ablation corrupts the **raw source input**. Benchmark A
        is i.i.d. white noise, so future step $\tau$'s target depends on
        exactly one source lag $\ell = D-1-\tau$ -- corrupting the whole source
        and decomposing the loss degradation per future step is therefore a
        faithful per-lag LOLO. See :func:`run_lag_ablation`.

It also surfaces the Section 4.2 finding (Task 5.5): when the delay $D$ exceeds
the attention window $L_{\max}$, the recurrent encoder still carries the source,
so $K$ can stay non-zero. :func:`_delay_window_report` reports this for any
checkpoint; the dedicated large-$D$ run is deferred.

For the two-lag benchmark E (Task 7.3) the harness adds the **two-band mass
ratio** (:func:`compute_two_band_mass_ratio`): the lag map should resolve two
separate bands whose mass ratio tracks the per-band TE ratio. The single-delay
input-level LOLO (Task 5.4) does not generalise to two delays and is skipped
for E.

This module **reuses** :func:`evaluate_te.load_eval_checkpoint` /
:func:`evaluate_te.make_test_loader` and the :mod:`train_minimal` helpers so it
scores models with the exact model / loss code the earlier phases used.

Run modes (project convention -- Decision D9 in
``synthetic_te_validation_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``, auto-detected from
whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.lag_recovery --checkpoint PATH [--config PATH]
            [--data-tag TAG] [--benchmark B] [--batch-size N]
            [--n-ablation-samples N] [--device DEV] [--seed S]

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
# (``A_lag``, ``delta_per_tau``, lag profiles) go to ``metrics.json``. The
# ``lag_mass_band*`` columns are populated only for the two-lag benchmark E.
_SUMMARY_FIELDS = [
    "run_tag", "data_tag", "benchmark", "a", "M", "delay", "horizon",
    "te_true", "warmup", "n_test", "L", "lag_band_lo", "lag_band_hi",
    "lag_mass_attn", "lag_mass_attn_uniform", "lag_mass_attn_ratio",
    "peak_lag_err_mean", "peak_lag_err_median", "peak_in_band_frac",
    "lag_mass_lolo", "delta_oob_max",
    "lag_mass_band1", "lag_mass_band2", "lag_mass_ratio",
    "lag_mass_te_ratio", "lag_mass_ratio_err",
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
    results_root = (
        _EXPERIMENT_DIR / str(config["paths"]["results_dir"])
    ).resolve()
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


_LOLO_NOTE = (
    "For Benchmark A the source is i.i.d. white noise, so each future step's "
    "target depends on exactly one source lag (ell = D-1-tau, which ranges "
    "over the true band {D-H,...,D-1} as tau spans [0,H)). All transferable "
    "information is in-band by construction, so LagMass_LOLO is ~1.0 whenever "
    "the model uses the source at all (or NaN on a collapsed latent). The "
    "discriminating LOLO signal for Benchmark A is the within-band shape of "
    "A_lag / delta_per_tau, not the scalar LagMass_LOLO; the scalar becomes a "
    "real discriminator only for autocorrelated or multi-lag source benchmarks."
)


def _skipped_ablation_stub(reason: str) -> Dict[str, Any]:
    """Return a :func:`run_lag_ablation`-shaped stub for a skipped LOLO pass.

    Args:
        reason: Why the LOLO ablation was skipped (recorded in the JSON).

    Returns:
        A dict with the :func:`run_lag_ablation` keys, all numeric fields set
        to ``nan`` and all vector fields empty.
    """
    return {
        "delta_per_tau": [],
        "delta_per_tau_informative": [],
        "lag_of_tau": [],
        "A_lag": [],
        "A_lag_raw": [],
        "A_overflow": float("nan"),
        "lag_mass_lolo": float("nan"),
        "total_delta": float("nan"),
        "delta_oob_max": float("nan"),
        "mse_clean": [],
        "mse_corrupt": [],
        "n_ablation_batches": None,
        "crosscheck_rel_err": float("nan"),
        "lag_mass_lolo_note": reason,
    }


@torch.no_grad()
def run_lag_ablation(
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
    n_ablation_samples: Optional[int] = None,
    do_oob_probe: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""Faithful input-level leave-one-lag-out ablation (task 5.4, Decision D5).

    Corrupts the whole raw source with $\mathcal N(0, 1)$ noise and decomposes
    the per-future-step loss degradation $\delta(\tau) = \mathrm{MSE}_{\rm
    corrupt}(\tau) - \mathrm{MSE}_{\rm clean}(\tau)$. Because future step
    $\tau$'s target depends only on source lag $\ell = D-1-\tau$ (white-noise
    DGP), $[\delta(\tau)]_+$ scattered by $\ell$ yields the per-lag importance
    $A_\ell$; lags off the attention axis accumulate into ``A_overflow`` (the
    large-$D$ safety valve). An optional out-of-band probe corrupts the source
    window $[T-D, T)$ -- which no scored future step depends on -- and should
    leave the loss unchanged.

    Args:
        model: The trained model.
        loader: A test :class:`DataLoader`.
        device: Compute device.
        meta: The dataset ``meta.json`` (carries ``delay``, ``horizon``,
            ``true_lag_band``, ``informative_channels``, ``sequence_length``).
        warmup: Leading anchors excluded from the loss.
        max_lag: Maximum attention lag.
        beta: KL weight (cross-check only).
        lambda_full: Full-forecast loss weight (cross-check only).
        lambda_base: Baseline-forecast loss weight (cross-check only).
        batch_size: Loader batch size (to map ``n_ablation_samples`` to a
            batch cap).
        n_ablation_samples: Optional cap on the samples used per pass.
        do_oob_probe: Whether to run the out-of-band sanity pass.
        seed: Base seed for the corruption / reparam RNGs.

    Returns:
        Dict with ``delta_per_tau`` / ``delta_per_tau_informative``,
        ``lag_of_tau``, ``A_lag`` (normalised) / ``A_lag_raw``, ``A_overflow``,
        ``lag_mass_lolo``, ``total_delta``, ``delta_oob_max``, ``mse_clean`` /
        ``mse_corrupt``, ``crosscheck_rel_err`` and ``lag_mass_lolo_note``.
    """
    D = int(meta["delay"])
    H = int(meta["horizon"])
    T = int(meta["sequence_length"])
    informative = meta.get("informative_channels")
    lag_band = np.asarray(list(meta["true_lag_band"]), dtype=int)

    max_batches: Optional[int] = None
    if n_ablation_samples is not None and batch_size > 0:
        max_batches = max(1, math.ceil(int(n_ablation_samples) / batch_size))

    # --- correctness gate: per-tau wiring must match compute_loss ------------
    crosscheck = _crosscheck_per_tau(
        model, loader, device, warmup=warmup,
        beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
    )

    # --- clean / whole-source-corrupt passes (paired reparam draws) ----------
    mse_clean = _per_tau_mse(
        model, loader, device, warmup=warmup, horizon=H,
        corrupt=None, seed=seed, max_batches=max_batches,
    )
    mse_corrupt = _per_tau_mse(
        model, loader, device, warmup=warmup, horizon=H,
        corrupt="all", seed=seed, max_batches=max_batches,
    )
    delta = mse_corrupt - mse_clean                    # (H,)

    # Informative-channel variant (feat_loss restricted to the M signal
    # channels -- compute_loss never exposes this; channel dilution cancels in
    # the normalised A_ell either way).
    mse_clean_inf = _per_tau_mse(
        model, loader, device, warmup=warmup, horizon=H,
        channels=informative, corrupt=None, seed=seed, max_batches=max_batches,
    )
    mse_corrupt_inf = _per_tau_mse(
        model, loader, device, warmup=warmup, horizon=H,
        channels=informative, corrupt="all", seed=seed,
        max_batches=max_batches,
    )
    delta_inf = mse_corrupt_inf - mse_clean_inf

    # --- scatter delta(tau) into per-lag importance A_ell --------------------
    lag_of_tau = D - 1 - np.arange(H)                  # (H,)
    dpos = np.clip(delta, 0.0, None)
    A = np.zeros(max_lag + 1, dtype=float)
    A_overflow = 0.0
    for tau in range(H):
        ell = int(lag_of_tau[tau])
        if 0 <= ell <= max_lag:
            A[ell] += float(dpos[tau])
        else:
            A_overflow += float(dpos[tau])

    total = float(A.sum() + A_overflow)
    if total < 1e-12:
        A_norm = np.full(max_lag + 1, np.nan)
        lag_mass_lolo = float("nan")
    else:
        A_norm = A / total
        band_idx = lag_band[(lag_band >= 0) & (lag_band <= max_lag)]
        lag_mass_lolo = float(A_norm[band_idx].sum())

    # --- out-of-band probe: corrupt [T-D, T) -> expect delta ~ 0 -------------
    delta_oob_max = float("nan")
    if do_oob_probe:
        mse_oob = _per_tau_mse(
            model, loader, device, warmup=warmup, horizon=H,
            corrupt=(T - D, T), seed=seed, max_batches=max_batches,
        )
        delta_oob_max = float(np.nanmax(np.abs(mse_oob - mse_clean)))

    return {
        "delta_per_tau": delta.tolist(),
        "delta_per_tau_informative": delta_inf.tolist(),
        "lag_of_tau": lag_of_tau.tolist(),
        "A_lag": A_norm.tolist(),
        "A_lag_raw": A.tolist(),
        "A_overflow": A_overflow,
        "lag_mass_lolo": lag_mass_lolo,
        "total_delta": total,
        "delta_oob_max": delta_oob_max,
        "mse_clean": mse_clean.tolist(),
        "mse_corrupt": mse_corrupt.tolist(),
        "n_ablation_batches": max_batches,
        "crosscheck_rel_err": crosscheck,
        "lag_mass_lolo_note": _LOLO_NOTE,
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
    # The two-lag benchmark E has no single ``delay``; report the larger.
    D = int(meta.get("delay", meta.get("delay2", meta.get("delay1", 0))))
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

    Args:
        meta: The dataset ``meta.json``.

    Returns:
        A list of ``(lo, hi)`` inclusive spans -- two entries for the two-lag
        benchmark E, one otherwise (empty if there is no causal lag band).
    """
    if str(meta.get("benchmark")) == "E" and "lag_band_1" in meta:
        spans = []
        for key in ("lag_band_1", "lag_band_2"):
            b = [int(x) for x in meta.get(key, [])]
            if b:
                spans.append((min(b), max(b)))
        return spans
    band = [int(x) for x in meta.get("true_lag_band", [])]
    return [(min(band), max(band))] if band else []


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
        ablation: The :func:`run_lag_ablation` output (or the E skip stub).
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
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if A.size == 0:
        ax.text(
            0.5, 0.5, "LOLO skipped (benchmark E:\nsingle-$D$ decomposition "
            "does not generalise)", ha="center", va="center",
            transform=ax.transAxes, fontsize=ps.FONT_LEGEND,
            color=ps.COLOR_GRAY,
        )
        ax.set_title("input-level LOLO per-lag importance")
    else:
        lag_axis = np.arange(A.size)
        in_band = np.zeros(A.size, dtype=bool)
        for lo, hi in spans:
            in_band |= (lag_axis >= lo) & (lag_axis <= hi)
        colours = [ps.COLOR_VERMILLION if in_band[ell] else ps.COLOR_BLUE
                   for ell in lag_axis]
        ax.bar(lag_axis, np.nan_to_num(A, nan=0.0), width=1.0, color=colours)
        lm = ablation.get("lag_mass_lolo", float("nan"))
        ax.set_title(
            f"input-level LOLO per-lag importance "
            f"$A_\\ell$  (LagMass$_{{\\rm LOLO}}$={lm:.3f}, "
            f"overflow={ablation.get('A_overflow', float('nan')):.3g})"
        )
        ax.set_xlabel(r"source lag $\ell$  (orange = true band)")
        ax.set_ylabel(r"$A_\ell$ (normalised)")
        ax.set_xlim(-0.5, A.size - 0.5)
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "lolo_abar")

    # --- Plot 3: lag profile + delta(tau), aligned on the source-lag axis -
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
    delta = np.asarray(ablation.get("delta_per_tau", []), dtype=float)
    lag_of_tau = np.asarray(ablation.get("lag_of_tau", []), dtype=float)
    if delta.size == 0 or lag_of_tau.size == 0:
        ax.text(
            0.5, 0.5, "LOLO degradation skipped (benchmark E)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=ps.FONT_LEGEND, color=ps.COLOR_GRAY,
        )
        ax.set_title(r"LOLO degradation $\delta(\tau)$")
    else:
        _shade(ax)
        ax.bar(lag_of_tau, delta, width=1.0, color=ps.COLOR_ORANGE,
               label=r"$\delta(\tau)$ at lag $D-1-\tau$")
        ax.axhline(0.0, color=ps.COLOR_GRAY, lw=0.8)
        ax.set_title(
            r"LOLO degradation $\delta(\tau)$ placed at lag $D-1-\tau$"
        )
        ax.set_ylabel(
            r"$\delta = \mathrm{MSE}_{\rm corrupt}-\mathrm{MSE}_{\rm clean}$"
        )
        ax.legend(loc="upper left")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_xlim(-0.5, L - 0.5)
    ps.style_axes(ax)
    ps.save_figure(fig, out_dir / "lag_profile")


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
        ablation: The :func:`run_lag_ablation` output (or the E skip stub).
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
        f"  Task 5.4  LOLO lag-mass      : {row['lag_mass_lolo']:.4f}  "
        f"(total delta {ablation['total_delta']:.4g}, "
        f"oob |delta| max {row['delta_oob_max']:.4g}, "
        f"cross-check {ablation['crosscheck_rel_err']:.2e})\n"
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

    # --- Task 5.4 / 7.3: LOLO ablation or the two-band mass ratio ------------
    k_bar = tm.compute_kbar(model, test_loader, device)
    is_two_lag = str(test_meta.get("benchmark")) == "E"
    two_band: Optional[Dict[str, Any]] = None
    if is_two_lag:
        # Benchmark E -- the single-delay LOLO decomposition does not
        # generalise to two delays (comment C27); skip it and compute the
        # two-band attention mass ratio (task 7.3) instead.
        ablation = _skipped_ablation_stub(
            "LOLO skipped for benchmark E: run_lag_ablation's single-delay "
            "lag_of_tau = D-1-tau decomposition does not generalise to two "
            "delays. The two-band attention mass ratio (task 7.3) is the "
            "lag-recovery metric for E."
        )
        two_band = compute_two_band_mass_ratio(
            te_tl,
            lag_band_1=[int(x) for x in test_meta["lag_band_1"]],
            lag_band_2=[int(x) for x in test_meta["lag_band_2"]],
            anchor_lo=anchor_lo, anchor_hi=anchor_hi, max_lag=max_lag,
            te_true_1=float(test_meta["te_true_1"]),
            te_true_2=float(test_meta["te_true_2"]),
        )
    else:
        ablation = run_lag_ablation(
            model, test_loader, device, meta=test_meta, warmup=warmup,
            max_lag=max_lag, beta=beta, lambda_full=lambda_full,
            lambda_base=lambda_base, batch_size=batch_size,
            n_ablation_samples=n_abl, do_oob_probe=do_oob, seed=ablation_seed,
        )

    # --- Task 5.5: delay versus attention window -----------------------------
    dwin = _delay_window_report(test_meta, max_lag, k_bar)

    te_true = float(
        test_meta.get("te_true", data_meta.get("te_true", float("nan")))
    )
    run_tag = Path(ckpt_path).resolve().parent.name

    row: Dict[str, Any] = {
        "run_tag": run_tag,
        "data_tag": str(tag),
        "benchmark": benchmark,
        "a": data_meta.get("a", test_meta.get("a")),
        "M": data_meta.get("M", test_meta.get("M")),
        # Benchmark E has no single ``delay`` -- report the larger of the two.
        "delay": int(
            test_meta.get("delay", test_meta.get("delay2", 0))
        ),
        "horizon": int(test_meta["horizon"]),
        "te_true": te_true,
        "warmup": warmup,
        "n_test": int(len(test_loader.dataset)),
        "L": int(L),
        "lag_band_lo": int(min(lag_band)),
        "lag_band_hi": int(max(lag_band)),
        "lag_mass_attn": lag_mass["lag_mass_attn"],
        "lag_mass_attn_uniform": lag_mass["uniform_baseline"],
        "lag_mass_attn_ratio": lag_mass["ratio_to_uniform"],
        "peak_lag_err_mean": peak["peak_lag_err_mean"],
        "peak_lag_err_median": peak["peak_lag_err_median"],
        "peak_in_band_frac": peak["peak_in_band_frac"],
        "lag_mass_lolo": ablation["lag_mass_lolo"],
        "delta_oob_max": ablation["delta_oob_max"],
        "lag_mass_band1": two_band["lag_mass_1"] if two_band else None,
        "lag_mass_band2": two_band["lag_mass_2"] if two_band else None,
        "lag_mass_ratio": two_band["mass_ratio"] if two_band else None,
        "lag_mass_te_ratio": two_band["te_ratio"] if two_band else None,
        "lag_mass_ratio_err": two_band["ratio_error"] if two_band else None,
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
            # Benchmark E has no single ``delay``; report the larger.
            "delay": int(test_meta.get("delay", test_meta.get("delay2", 0))),
            "horizon": int(test_meta["horizon"]),
            "te_true": te_true,
            "true_lag_band": lag_band,
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
        "task_5_4_lolo": {
            "lag_mass_lolo": ablation["lag_mass_lolo"],
            "total_delta": ablation["total_delta"],
            "A_overflow": ablation["A_overflow"],
            "delta_oob_max": ablation["delta_oob_max"],
            "crosscheck_rel_err": ablation["crosscheck_rel_err"],
            "n_ablation_batches": ablation["n_ablation_batches"],
            "lag_of_tau": ablation["lag_of_tau"],
            "delta_per_tau": ablation["delta_per_tau"],
            "delta_per_tau_informative": ablation["delta_per_tau_informative"],
            "mse_clean": ablation["mse_clean"],
            "mse_corrupt": ablation["mse_corrupt"],
            "A_lag": ablation["A_lag"],
            "A_lag_raw": ablation["A_lag_raw"],
            "lag_mass_lolo_note": ablation["lag_mass_lolo_note"],
        },
        "task_5_5_delay_vs_max_lag": dwin,
        "per_anchor": {
            "lag_profile": te_profile.tolist(),
            "attn_lag_profile": a_profile.tolist(),
            "kld_per_t_mean": collected["kld_per_t_t"].tolist(),
        },
    }
    if two_band is not None:
        metrics["task_7_3_two_band"] = two_band

    out_dir = _lag_out_dir(config, benchmark)
    write_summary_csv(row, out_dir / "summary.csv")
    write_metrics_json(metrics, out_dir / "metrics.json")
    _make_plots(collected, lag_mass, ablation, test_meta, out_dir)
    _print_summary(row, ablation, two_band)
    print(f"[done] lag-recovery analysis -> {out_dir}")
    return {"row": row, "metrics": metrics, "out_dir": str(out_dir)}


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
    return config


def _dispatch(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve overrides, seed, device and run the analysis.

    Args:
        config: The parsed ``config_synth.yaml``.
        overrides: Flat overrides (from ``vars(args)`` or ``RUN_CONFIG``).

    Returns:
        The :func:`analyze_lag_recovery` result dict.

    Raises:
        ValueError: If no checkpoint is supplied.
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
    return analyze_lag_recovery(
        ckpt, config, device=device,
        data_tag=overrides.get("data_tag"),
        batch_size=overrides.get("batch_size"),
        n_ablation_samples=overrides.get("n_ablation_samples"),
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
        help="cap on the samples used per LOLO pass (task 5.4)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override experiment.seed",
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
            _EXPERIMENT_DIR / "results" / "A" / "pol_easy_a1" / "final.ckpt"
        ),
        "data_tag": None,            # None -> the checkpoint's own tag
        "benchmark": None,           # None -> config experiment.benchmark
        "batch_size": None,          # None -> checkpoint's batch size
        "n_ablation_samples": None,  # None -> config lag_recovery value
        "device": None,              # None -> config runtime.device
        "seed": None,                # None -> config experiment.seed
    }

    if len(sys.argv) > 1:
        main()                       # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["lag_recovery"]["n_ablation_samples"] = 256
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
