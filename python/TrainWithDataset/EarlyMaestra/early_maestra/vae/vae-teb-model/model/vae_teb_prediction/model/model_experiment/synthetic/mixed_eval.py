r"""``mixed_eval`` -- per-group recovery + generalization for the ``G1_mix`` model.

A single model is trained (under Gaussian-NLL) on the heterogeneous pool built
by :mod:`mixed_dataset`. This module reads that one checkpoint and asks, **per
sub-population**, whether the model recovers the true KLD / lag / TE, and whether
it **generalizes** to held-out $(M, \mathrm{TE}, \text{lag})$ triples it never
trained on.

The per-sample machinery already exists in the model output -- ``kld_per_t``
$(B,T)$ -- it is simply averaged away elsewhere. Here each sample's
$\bar K_i$ is read over its **own** clean window
$[\max(\text{warmup}, d_{\max,i}-1),\,T-H)$ (the lag band varies per sample),
then grouped by the per-sample provenance ``cell_id`` / ``M`` / ``band_id`` and
marginally by $M$ / $\mathrm{TE}$ / band.

Metrics (per :mod:`model_validation_v2_plan` §8):
    * **Calibration** $\gamma$ in $\bar K = \alpha + \gamma\,\mathrm{TE}$
      (overall + per-$M$ + per-band) via :func:`calibration.fit_calibration_slope`.
    * **TE recovery** $\widehat{\mathrm{TE}}_i = (\bar K_i - \alpha)/\gamma$,
      per-cell RMSE / bias against the cell TE.
    * **Lag recovery** per cell via
      :func:`lag_recovery.run_sliding_window_lolo` on a per-cell subset loader
      (each cell's band is $\{0,\dots,d_{\max}-1\}$).
    * **Null controls** per cell: $\bar K$ under shuffled / time-reversed source.
    * **Generalization gap**: held-out cell metric minus the trained-marginal
      mean.

Run modes (Decision V2-D8): both a CLI and an edit-and-run ``RUN_CONFIG``.

    python -m ...synthetic.mixed_eval --run-tag G1_mix_gnll \
        --in-mix-tag G1_mix_base --holdout-tag G1_mix_base_holdout
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model.vae_teb_prediction.model.model_experiment.synthetic.calibration import (
    fit_calibration_slope,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
    attribute_dict_collate,
    build_u_stream,
    make_dataloader,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te import (
    load_eval_checkpoint,
    reverse_source_batch,
    shuffle_source_batch,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.lag_recovery import (
    compute_lag_mass_attn,
    compute_peak_lag_error,
    run_sliding_window_lolo,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import plot_style as ps
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    apply_path_overrides,
    load_config,
    move_batch,
    resolve_active_benchmark,
    resolve_device,
    resolve_user_path,
)

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_BENCHMARK = "G1_mix"
_OUT_SUBDIR = "mixed_eval"


# =============================================================================
# Per-sample K-bar collection
# =============================================================================

@torch.no_grad()
def collect_per_sample_kbar(
    model: Any,
    loader: Any,
    device: torch.device,
    *,
    warmup: int,
    horizon: int,
    controls: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    r"""One eval pass collecting each sample's $\bar K$ and provenance.

    For sample $i$ the clean window is
    $[\max(\text{warmup}, d_{\max,i}-1),\,T-H)$ -- the per-sample lag floor, not
    a pooled range -- and $\bar K_i$ is the mean of ``kld_per_t`` over it.

    Args:
        model: The trained model in ``eval`` mode.
        loader: A non-shuffled mixed-cache loader (samples carry ``te_true`` /
            ``M`` / ``delay_max`` / ``band_id`` / ``cell_id`` / ``held_out``).
        device: Compute device.
        warmup: Warm-up steps excluded from every window.
        horizon: Forecast horizon $H$ (window upper bound is $T-H$).
        controls: Subset of ``{"shuffle", "reverse"}`` -- additionally compute
            $\bar K$ under those source corruptions (same target / cell label).

    Returns:
        A dict of aligned length-$N$ arrays: ``kbar``, ``te_true``, ``M``,
        ``delay_max``, ``band_id``, ``cell_id``, ``held_out``, plus
        ``kbar_shuffle`` / ``kbar_reverse`` when requested. In addition three
        **nested** diagnostic structures (not length-$N$ arrays) are attached:
        ``per_dim_kl_by_cell`` / ``per_dim_kl_by_M`` (clean-window mean KL per
        latent dimension, $(\,d_z,)$ each) and ``kld_time_by_band`` (the mean
        ``kld_per_t`` trajectory $(\,T,)$ per lag-band, with that band's
        ``delay_max`` for the clean-window floor).
    """
    out: Dict[str, List[np.ndarray]] = {
        k: [] for k in (
            "kbar", "te_true", "M", "delay_max", "band_id", "cell_id", "held_out",
        )
    }
    for ctrl in controls:
        out[f"kbar_{ctrl}"] = []

    # Per-dim KL accumulators (clean-window masked sum over (samples, t) and the
    # matching valid-step count), grouped by cell and by M; kld-vs-time
    # accumulators (full-T mean over samples) grouped by band.
    pdk_cell_sum: Dict[int, np.ndarray] = {}
    pdk_cell_cnt: Dict[int, float] = {}
    pdk_m_sum: Dict[int, np.ndarray] = {}
    pdk_m_cnt: Dict[int, float] = {}
    kt_band_sum: Dict[int, np.ndarray] = {}
    kt_band_cnt: Dict[int, int] = {}
    kt_band_dmax: Dict[int, int] = {}

    for batch in loader:
        batch = move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        delay_max = _batch_int(batch, "delay_max", device)
        kbar, kld_sum_dz, valid_cnt, kld_bt = _per_sample_kld(
            model, y_st, y_ph, build_u_stream(batch), delay_max,
            warmup=warmup, horizon=horizon,
        )
        out["kbar"].append(kbar)
        for ctrl in controls:
            corrupt = (
                shuffle_source_batch(batch) if ctrl == "shuffle"
                else reverse_source_batch(batch)
            )
            out[f"kbar_{ctrl}"].append(_per_sample_kld(
                model, y_st, y_ph, build_u_stream(corrupt), delay_max,
                warmup=warmup, horizon=horizon,
            )[0])
        m_b = _batch_int(batch, "M", device).cpu().numpy()
        band_b = _batch_int(batch, "band_id", device).cpu().numpy()
        cell_b = _batch_int(batch, "cell_id", device).cpu().numpy()
        dmax_b = delay_max.cpu().numpy()
        out["te_true"].append(_batch_float(batch, "te_true"))
        out["M"].append(m_b)
        out["delay_max"].append(dmax_b)
        out["band_id"].append(band_b)
        out["cell_id"].append(cell_b)
        out["held_out"].append(_batch_int(batch, "held_out", device).cpu().numpy())

        # Accumulate the nested per-dim / per-time diagnostics for this batch.
        _accumulate_per_dim(
            cell_b, kld_sum_dz, valid_cnt, pdk_cell_sum, pdk_cell_cnt,
        )
        _accumulate_per_dim(m_b, kld_sum_dz, valid_cnt, pdk_m_sum, pdk_m_cnt)
        _accumulate_kld_time(
            band_b, dmax_b, kld_bt, kt_band_sum, kt_band_cnt, kt_band_dmax,
        )

    result: Dict[str, Any] = {
        k: np.concatenate(v, axis=0) for k, v in out.items() if v
    }
    result["per_dim_kl_by_cell"] = {
        int(c): (pdk_cell_sum[c] / max(pdk_cell_cnt[c], 1.0)).tolist()
        for c in pdk_cell_sum
    }
    result["per_dim_kl_by_M"] = {
        int(m): (pdk_m_sum[m] / max(pdk_m_cnt[m], 1.0)).tolist()
        for m in pdk_m_sum
    }
    result["kld_time_by_band"] = {
        int(b): {
            "kld_t": (kt_band_sum[b] / max(kt_band_cnt[b], 1)).tolist(),
            "delay_max": int(kt_band_dmax[b]),
            "T": int(kt_band_sum[b].shape[0]),
            "warmup": int(warmup),
        }
        for b in kt_band_sum
    }
    return result


def _accumulate_per_dim(
    key_arr: np.ndarray,
    kld_sum_dz: np.ndarray,
    valid_cnt: np.ndarray,
    sum_acc: Dict[int, np.ndarray],
    cnt_acc: Dict[int, float],
) -> None:
    r"""Add a batch's clean-window per-dim KL into ``{key: (sum_dz, count)}``.

    Args:
        key_arr: Per-sample grouping key (``cell_id`` or ``M``), shape $(B,)$.
        kld_sum_dz: Per-sample clean-window KL summed over $t$, shape
            $(B, d_z)$.
        valid_cnt: Per-sample count of valid (clean-window) steps, shape $(B,)$.
        sum_acc: Mutated ``{key -> running $(d_z,)$ KL sum}``.
        cnt_acc: Mutated ``{key -> running valid-step count}``.
    """
    for k in np.unique(key_arr):
        sel = key_arr == k
        ki = int(k)
        contrib = kld_sum_dz[sel].sum(axis=0)
        if ki not in sum_acc:
            sum_acc[ki] = np.zeros_like(contrib)
            cnt_acc[ki] = 0.0
        sum_acc[ki] += contrib
        cnt_acc[ki] += float(valid_cnt[sel].sum())


def _accumulate_kld_time(
    band_arr: np.ndarray,
    dmax_arr: np.ndarray,
    kld_bt: np.ndarray,
    sum_acc: Dict[int, np.ndarray],
    cnt_acc: Dict[int, int],
    dmax_acc: Dict[int, int],
) -> None:
    r"""Add a batch's full-$T$ ``kld_per_t`` into per-band sample-mean accumulators.

    Args:
        band_arr: Per-sample ``band_id``, shape $(B,)$.
        dmax_arr: Per-sample ``delay_max``, shape $(B,)$ (constant within a band).
        kld_bt: Per-sample ``kld_per_t`` over the full sequence, shape $(B, T)$.
        sum_acc: Mutated ``{band -> running $(T,)$ KL sum over samples}``.
        cnt_acc: Mutated ``{band -> running sample count}``.
        dmax_acc: Mutated ``{band -> delay_max}`` (first value seen).
    """
    for b in np.unique(band_arr):
        sel = band_arr == b
        bi = int(b)
        contrib = kld_bt[sel].sum(axis=0)
        if bi not in sum_acc:
            sum_acc[bi] = np.zeros_like(contrib)
            cnt_acc[bi] = 0
            dmax_acc[bi] = int(dmax_arr[sel][0])
        sum_acc[bi] += contrib
        cnt_acc[bi] += int(sel.sum())


def _batch_float(batch: Any, key: str) -> np.ndarray:
    """Return a batch scalar field as a 1-D ``float64`` numpy array."""
    val = batch[key]
    if torch.is_tensor(val):
        return val.detach().cpu().to(torch.float64).numpy()
    return np.asarray(val, dtype=np.float64)


def _batch_int(batch: Any, key: str, device: torch.device) -> torch.Tensor:
    """Return a batch integer field as a 1-D ``long`` tensor on ``device``."""
    val = batch[key]
    if torch.is_tensor(val):
        return val.to(device=device, dtype=torch.long)
    return torch.as_tensor(val, dtype=torch.long, device=device)


@torch.no_grad()
def _per_sample_kld(
    model: Any,
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    u_stream: torch.Tensor,
    delay_max: torch.Tensor,
    *,
    warmup: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Per-sample $\bar K$ plus the raw KL tensors over each sample's window.

    For sample $i$ the clean window is
    $[\max(\text{warmup}, d_{\max,i}-1),\,T-H)$. The function returns the scalar
    $\bar K_i$ *and* the building blocks for the per-dim / per-time diagnostics
    so the caller can accumulate them without a second encoder pass.

    Args:
        model: The model (``encode_only`` + ``kld_tensor`` are used so no
            decoder pass is needed).
        y_st, y_ph: Target streams $(B, T, \cdot)$.
        u_stream: Source stream $(B, T, c_u)$.
        delay_max: Per-sample lag ceiling $(B,)$ long tensor.
        warmup: Warm-up floor.
        horizon: Forecast horizon $H$.

    Returns:
        A 4-tuple of numpy arrays:
            * ``kbar`` $(B,)$ ``float64`` -- per-sample mean clean-window KL.
            * ``kld_sum_dz`` $(B, d_z)$ -- clean-window KL summed over $t$.
            * ``valid_cnt`` $(B,)$ -- number of clean-window steps per sample.
            * ``kld_bt`` $(B, T)$ -- full-sequence ``kld_per_t`` (unmasked).
    """
    enc = model.encode_only(y_st, y_ph, u_stream, sample_z=True)
    kld = model.kld_tensor(
        mu_prior=enc["mu_prior"], logvar_prior=enc["logvar_prior"],
        mu_post=enc["mu_post"], logvar_post=enc["logvar_post"],
        mask_warmup=False,
    )                                                      # (B, T, d_z)
    kld_bt = kld.sum(dim=-1)                               # (B, T)
    T = kld_bt.shape[1]
    hi = int(T - horizon)
    t_idx = torch.arange(T, device=kld_bt.device).unsqueeze(0)   # (1, T)
    lo = torch.clamp(delay_max.long() - 1, min=int(warmup)).unsqueeze(1)  # (B,1)
    valid = (t_idx >= lo) & (t_idx < hi)                  # (B, T)
    valid_f = valid.to(kld_bt.dtype)
    denom = valid_f.sum(dim=1).clamp(min=1.0)
    kbar = (kld_bt * valid_f).sum(dim=1) / denom
    kld_sum_dz = (kld * valid_f.unsqueeze(-1)).sum(dim=1)  # (B, d_z)
    return (
        kbar.detach().cpu().to(torch.float64).numpy(),
        kld_sum_dz.detach().cpu().to(torch.float64).numpy(),
        valid_f.sum(dim=1).detach().cpu().to(torch.float64).numpy(),
        kld_bt.detach().cpu().to(torch.float64).numpy(),
    )


# =============================================================================
# Grouping + calibration
# =============================================================================

def group_recovery(
    arrs: Dict[str, np.ndarray],
    cells_by_id: Dict[int, Dict[str, Any]],
    *,
    alpha: float,
    gamma: float,
    controls: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    r"""Aggregate per-sample $\bar K$ into per-cell recovery stats.

    Args:
        arrs: The aligned arrays from :func:`collect_per_sample_kbar`.
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.
        alpha, gamma: The overall calibration fit, used for
            $\widehat{\mathrm{TE}}_i = (\bar K_i - \alpha)/\gamma$.
        controls: Null-control names present in ``arrs``.

    Returns:
        A per-cell list of dicts (one row per ``cell_id``), each carrying the
        cell identity, $\bar K$ stats, TE-recovery error and null ratios.
    """
    rows: List[Dict[str, Any]] = []
    cell_ids = np.asarray(arrs["cell_id"], dtype=int)
    kbar = np.asarray(arrs["kbar"], dtype=float)
    te = np.asarray(arrs["te_true"], dtype=float)
    safe_gamma = gamma if abs(gamma) > 1e-12 else float("nan")
    te_pred_all = (kbar - alpha) / safe_gamma
    for cid in sorted(set(cell_ids.tolist())):
        sel = cell_ids == cid
        cell = cells_by_id.get(int(cid), {})
        te_cell = float(np.mean(te[sel]))
        kb = kbar[sel]
        te_pred = te_pred_all[sel]
        row: Dict[str, Any] = {
            "cell_id": int(cid),
            "M": int(cell.get("M", arrs["M"][sel][0])),
            "target_te": float(cell.get("target_te", float("nan"))),
            "band": str(cell.get("band", "")),
            "delay_min": int(cell.get("delay_min", 0)),
            "delay_max": int(cell.get("delay_max", int(arrs["delay_max"][sel][0]))),
            "B_y": float(cell.get("B_y_scalar", float("nan"))),
            "te_true": te_cell,
            "n": int(sel.sum()),
            "kbar_mean": float(np.mean(kb)),
            "kbar_std": float(np.std(kb)),
            "te_pred_mean": float(np.mean(te_pred)),
            "te_rmse": float(np.sqrt(np.mean((te_pred - te_cell) ** 2))),
            "te_bias": float(np.mean(te_pred) - te_cell),
            "held_out": int(arrs["held_out"][sel][0]),
        }
        for ctrl in controls:
            kc = np.asarray(arrs[f"kbar_{ctrl}"], dtype=float)[sel]
            denom = row["kbar_mean"] if abs(row["kbar_mean"]) > 1e-12 else float("nan")
            row[f"null_{ctrl}_kbar"] = float(np.mean(kc))
            row[f"null_{ctrl}_ratio"] = float(np.mean(kc) / denom)
        rows.append(row)
    return rows


def _fit_slices_from_response(
    arrs: Dict[str, np.ndarray],
    cells_by_id: Dict[int, Dict[str, Any]],
    response: np.ndarray,
) -> Dict[str, Any]:
    r"""Fit $\text{response} = \alpha + \gamma\,\mathrm{TE}$ overall + per-$M$ + per-band.

    Shared core of :func:`fit_calibration_slices` (response $= \bar K$) and
    :func:`fit_calibration_slices_nullsub` (response $= \bar K - \bar K_{\text{null}}$).
    The headline slope uses **per-cell means** (one point per cell); a singular
    or under-determined slice (fewer than two distinct cell TEs) is skipped.

    Args:
        arrs: The aligned per-sample arrays (carry ``cell_id`` / ``te_true`` / ``M``).
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.
        response: Per-sample response array aligned with ``arrs['cell_id']``.

    Returns:
        ``{"overall": fit, "by_M": {M: fit}, "by_band": {band: fit}}`` where each
        ``fit`` is the :func:`fit_calibration_slope` dict, or ``None`` if singular.
    """
    cell_ids = np.asarray(arrs["cell_id"], dtype=int)
    resp = np.asarray(response, dtype=float)
    te = np.asarray(arrs["te_true"], dtype=float)
    per_cell: List[Tuple[float, float, int, str]] = []
    for cid in sorted(set(cell_ids.tolist())):
        sel = cell_ids == cid
        cell = cells_by_id.get(int(cid), {})
        per_cell.append((
            float(np.mean(te[sel])), float(np.mean(resp[sel])),
            int(cell.get("M", arrs["M"][sel][0])),
            str(cell.get("band", "")),
        ))

    def _fit(pairs: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
        try:
            return fit_calibration_slope(pairs)
        except ValueError:
            return None

    overall = _fit([(t, k) for t, k, _, _ in per_cell])
    by_M: Dict[str, Any] = {}
    for m in sorted({c[2] for c in per_cell}):
        by_M[str(m)] = _fit([(t, k) for t, k, mm, _ in per_cell if mm == m])
    by_band: Dict[str, Any] = {}
    for b in sorted({c[3] for c in per_cell}):
        by_band[b] = _fit([(t, k) for t, k, _, bb in per_cell if bb == b])
    return {"overall": overall, "by_M": by_M, "by_band": by_band}


def fit_calibration_slices(
    arrs: Dict[str, np.ndarray],
    cells_by_id: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    r"""Fit $\bar K = \alpha + \gamma\,\mathrm{TE}$ overall + per-$M$ + per-band.

    Args:
        arrs: The aligned per-sample arrays.
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.

    Returns:
        ``{"overall": fit, "by_M": {M: fit}, "by_band": {band: fit}}``.
    """
    return _fit_slices_from_response(arrs, cells_by_id, np.asarray(arrs["kbar"]))


def fit_calibration_slices_nullsub(
    arrs: Dict[str, np.ndarray],
    cells_by_id: Dict[int, Dict[str, Any]],
    *,
    control: str = "shuffle",
) -> Dict[str, Any]:
    r"""Fit the **null-subtracted** calibration $(\bar K - \bar K_{\text{null}})$ vs TE.

    Shuffling the source $U$ destroys the directed term $I_q(Z;U\mid Y)$ while
    leaving the prior-mismatch / estimation floor
    $\mathbb{E}_Y[\mathrm{KL}(q_\phi(z\mid Y)\,\|\,p_\psi(z\mid Y))]$ intact, so
    $\bar K_{\text{shuffle}}$ estimates that floor. Regressing the
    floor-subtracted response should drive the intercept $\alpha \to 0$ while
    leaving $\gamma$ essentially unchanged -- a direct check that the calibration
    intercept *is* the floor (model_validation_v3_mixed identity validation).

    Args:
        arrs: The aligned per-sample arrays (must carry ``kbar_<control>``).
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.
        control: Null-control name (``shuffle`` / ``reverse``) present in ``arrs``.

    Returns:
        The same shape as :func:`fit_calibration_slices`, or ``{}`` if the
        control was not collected.
    """
    key = f"kbar_{control}"
    if key not in arrs:
        return {}
    response = np.asarray(arrs["kbar"], dtype=float) - np.asarray(arrs[key], dtype=float)
    return _fit_slices_from_response(arrs, cells_by_id, response)


def calibration_primary_summary(
    slices: Dict[str, Any], *, gamma_tol: float = 0.2,
) -> Dict[str, Any]:
    r"""Summarise the **per-$M$** calibration as the headline result.

    The mixed pool crosses the channel-dilution axis $M$, so a single pooled
    slope conflates the $M$-dependence of $\bar K$ with the TE-dependence. The
    per-$M$ slopes are therefore the primary report; this collapses them into a
    compact summary (mean / median $\gamma$, spread, and the fraction of $M$
    slices that meet $|\gamma - 1| \le \texttt{gamma\_tol}$).

    Args:
        slices: The ``fit_calibration_slices`` output (``{"overall", "by_M", ...}``).
        gamma_tol: Tolerance for the per-$M$ calibration pass (validation plan §8,
            Metric 3 threshold ``0.2``).

    Returns:
        ``{"gamma_by_M", "alpha_by_M", "mean_gamma", "median_gamma",
        "gamma_spread", "n_M", "frac_M_calibrated"}`` (empty if no per-$M$ fit).
    """
    by_M = (slices or {}).get("by_M", {}) or {}
    gamma_by_M: Dict[str, float] = {}
    alpha_by_M: Dict[str, float] = {}
    for m, fit in by_M.items():
        if isinstance(fit, dict) and fit:
            gamma_by_M[str(m)] = float(fit.get("gamma", float("nan")))
            alpha_by_M[str(m)] = float(fit.get("alpha", float("nan")))
    gammas = np.asarray(
        [g for g in gamma_by_M.values() if np.isfinite(g)], dtype=float
    )
    if gammas.size == 0:
        return {"gamma_by_M": gamma_by_M, "alpha_by_M": alpha_by_M, "n_M": 0}
    return {
        "gamma_by_M": gamma_by_M,
        "alpha_by_M": alpha_by_M,
        "mean_gamma": float(np.mean(gammas)),
        "median_gamma": float(np.median(gammas)),
        "gamma_spread": float(np.max(gammas) - np.min(gammas)),
        "n_M": int(gammas.size),
        "frac_M_calibrated": float(np.mean(np.abs(gammas - 1.0) <= float(gamma_tol))),
    }


# =============================================================================
# Per-cell lag recovery
# =============================================================================

def _cell_subset_loader(
    dataset: SyntheticTEDataset,
    cell_id: int,
    cell_id_array: np.ndarray,
    batch_size: int,
    *,
    cap: Optional[int],
) -> Optional[DataLoader]:
    """Build a non-shuffled loader over one cell's samples.

    Args:
        dataset: The full mixed-cache dataset.
        cell_id: The cell to select.
        cell_id_array: Per-sample ``sample_cell_id`` (dataset order).
        batch_size: Loader batch size.
        cap: Optional cap on the number of samples (first ``cap`` of the cell).

    Returns:
        A :class:`DataLoader` over the cell's samples, or ``None`` if empty.
    """
    idx = np.nonzero(cell_id_array == int(cell_id))[0]
    if cap is not None and idx.size > cap:
        idx = idx[:cap]
    if idx.size == 0:
        return None
    subset = Subset(dataset, idx.tolist())
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=attribute_dict_collate,
    )


def _cell_meta(cell: Dict[str, Any], horizon: int, T: int) -> Dict[str, Any]:
    r"""Build the per-cell ``meta`` the LOLO routine needs.

    The cell's lag band is $\{0,\dots,d_{\max}-1\}$ and its clean anchor range
    is $[d_{\max}-1,\,T-H)$ (the within-signal walk floor is $0$).

    Args:
        cell: A manifest cell dict.
        horizon: Forecast horizon $H$.
        T: Sequence length.

    Returns:
        A ``meta`` dict with ``horizon`` / ``sequence_length`` /
        ``true_lag_band`` / ``clean_anchor_range``.
    """
    dmax = int(cell["delay_max"])
    return {
        "horizon": int(horizon),
        "sequence_length": int(T),
        "true_lag_band": list(range(dmax)),
        "clean_anchor_range": [dmax - 1, int(T) - int(horizon)],
    }


def per_cell_lag_recovery(
    model: Any,
    dataset: SyntheticTEDataset,
    cells_by_id: Dict[int, Dict[str, Any]],
    cell_id_array: np.ndarray,
    device: torch.device,
    *,
    horizon: int,
    T: int,
    max_lag: int,
    warmup: int,
    loss_settings: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    r"""Run the sliding-window LOLO once per cell and read its LagMass.

    Args:
        model: The trained model.
        dataset: The mixed-cache dataset.
        cells_by_id: Manifest cells keyed by ``cell_id``.
        cell_id_array: Per-sample ``sample_cell_id`` (dataset order).
        device: Compute device.
        horizon, T, max_lag, warmup: Model / data geometry.
        loss_settings: Checkpoint ``loss_settings`` (``kld_beta`` /
            ``lambda_full`` / ``lambda_base``) for the LOLO cross-check.
        eval_cfg: The ``benchmarks.G1_mix.eval`` block.

    Returns:
        ``{cell_id: {"lag_mass_lolo", "A_lag", "peak_lag_err", "lag_grid"}}``.
    """
    beta = float(loss_settings.get("kld_beta", 0.001))
    lam_full = float(loss_settings.get("lambda_full", 1.0))
    lam_base = float(loss_settings.get("lambda_base", 0.5))
    width = int(eval_cfg.get("window_width", 10))
    coarse = int(eval_cfg.get("lag_grid_step", 5))
    fine = int(eval_cfg.get("fine_lag_grid_step", 1))
    cap = eval_cfg.get("n_lolo_per_cell")
    cap = None if cap is None else int(cap)
    bs = int(eval_cfg.get("batch_size") or 32)

    out: Dict[int, Dict[str, Any]] = {}
    for cid, cell in sorted(cells_by_id.items()):
        loader = _cell_subset_loader(dataset, cid, cell_id_array, bs, cap=cap)
        if loader is None:
            continue
        meta = _cell_meta(cell, horizon, T)
        res = run_sliding_window_lolo(
            model, loader, device, meta=meta, warmup=warmup, max_lag=max_lag,
            beta=beta, lambda_full=lam_full, lambda_base=lam_base,
            batch_size=bs, window_width=width, n_ablation_samples=cap,
            do_oob_probe=False, coarse_step=coarse, fine_step=fine,
        )
        band = list(range(int(cell["delay_max"])))
        out[int(cid)] = {
            "lag_mass_lolo": float(res.get("lag_mass_lolo", float("nan"))),
            "A_lag": [float(x) for x in res.get("A_lag", [])],
            "peak_lag_err": _peak_lag_err(res.get("A_lag", []), band),
            "lag_grid": [int(x) for x in res.get("lag_grid", [])],
        }
    return out


def _peak_lag_err(a_lag: Sequence[float], band: Sequence[int]) -> float:
    r"""Distance from $\arg\max_\ell A_\ell$ to the nearest in-band lag.

    Args:
        a_lag: Normalised $A_\ell$ profile.
        band: True lag-band indices.

    Returns:
        ``nan`` if ``a_lag`` is empty / all-NaN, else the integer distance.
    """
    arr = np.nan_to_num(np.asarray(a_lag, dtype=float), nan=0.0)
    if arr.size == 0 or float(arr.sum()) <= 0.0 or not band:
        return float("nan")
    peak = int(np.argmax(arr))
    return float(min(abs(peak - int(b)) for b in band))


# =============================================================================
# Per-cell prediction gain + attention-vs-lag (decoder pass)
# =============================================================================


@torch.no_grad()
def collect_per_cell_pred_gain(
    model: Any,
    dataset: SyntheticTEDataset,
    cells_by_id: Dict[int, Dict[str, Any]],
    cell_id_array: np.ndarray,
    device: torch.device,
    *,
    warmup: int,
    horizon: int,
    loss_settings: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    r"""Per-cell prediction gain $\Delta\mathcal L$ and attention-vs-lag profile.

    Runs **one** decoder forward pass per cell (capped at
    ``eval_cfg['n_lolo_per_cell']`` samples) and from the same output computes
    both diagnostics, so no second pass is needed:

    * $\Delta\mathcal L = \mathcal L_{\mathrm{base}} - \mathcal L_{\mathrm{feat}}$
      -- the size-weighted Gaussian-NLL prediction gain (validation plan §9.2),
      reusing :meth:`SeqVaeLagAttnV1.compute_loss` exactly as
      :func:`train_minimal.evaluate` does.
    * $\bar\alpha_\ell$ -- the head-averaged lag-attention profile averaged over
      each cell's clean window $[\max(\text{warmup}, d_{\max}-1),\,T-H)$ and
      samples, $(L,)$.

    Args:
        model: The trained model in ``eval`` mode.
        dataset: The mixed-cache dataset.
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.
        cell_id_array: Per-sample ``sample_cell_id`` (dataset order).
        device: Compute device.
        warmup: Warm-up floor.
        horizon: Forecast horizon $H$.
        loss_settings: Checkpoint ``loss_settings`` (``kld_beta`` /
            ``lambda_full`` / ``lambda_base`` / ``likelihood`` / ``sigma_obs`` /
            ``free_bits``).
        eval_cfg: The ``benchmarks.G1_mix.eval`` block (``n_lolo_per_cell`` cap,
            ``batch_size``).

    Returns:
        ``(pred_gain_by_cell, attn_profile_by_cell)`` -- each keyed by
        ``cell_id``. ``pred_gain`` rows carry ``delta_L`` / ``feat_loss`` /
        ``base_loss`` / ``te_true`` / ``M`` / ``band`` / ``target_te``;
        ``attn_profile`` rows carry ``attn_lag`` / ``delay_max`` / ``band``.
    """
    beta = float(loss_settings.get("kld_beta", 0.001))
    lam_full = float(loss_settings.get("lambda_full", 1.0))
    lam_base = float(loss_settings.get("lambda_base", 0.5))
    likelihood = str(loss_settings.get("likelihood", "gaussian_nll"))
    sigma_obs = loss_settings.get("sigma_obs", "learned")
    free_bits = float(loss_settings.get("free_bits", 0.0))
    cap = eval_cfg.get("n_lolo_per_cell")
    cap = None if cap is None else int(cap)
    bs = int(eval_cfg.get("batch_size") or 32)

    pred_gain: Dict[int, Dict[str, Any]] = {}
    attn_prof: Dict[int, Dict[str, Any]] = {}
    for cid, cell in sorted(cells_by_id.items()):
        loader = _cell_subset_loader(dataset, cid, cell_id_array, bs, cap=cap)
        if loader is None:
            continue
        dmax = int(cell["delay_max"])
        feat_sum = base_sum = 0.0
        n = 0
        attn_acc: Optional[np.ndarray] = None
        attn_cnt = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            y_st, y_ph = batch.fhr_st, batch.fhr_ph
            out = model(y_st, y_ph, build_u_stream(batch))
            losses = model.compute_loss(
                out, y_st, y_ph, weight=batch.weight, beta=beta,
                lambda_full=lam_full, lambda_base=lam_base,
                likelihood=likelihood, sigma_obs=sigma_obs, free_bits=free_bits,
            )
            b = int(y_st.size(0))
            feat_sum += float(losses["feat_loss"]) * b
            base_sum += float(losses["base_loss"]) * b
            n += b
            mean_alpha = out["attn_weights"].mean(dim=2)      # (B, T, L)
            T = int(mean_alpha.shape[1])
            hi = int(T - horizon)
            lo = max(int(warmup), dmax - 1)
            win = mean_alpha[:, lo:hi, :] if hi > lo else mean_alpha
            attn_b = win.sum(dim=(0, 1)).detach().cpu().to(torch.float64).numpy()
            attn_acc = attn_b if attn_acc is None else attn_acc + attn_b
            attn_cnt += float(win.shape[0] * win.shape[1])
        if n == 0:
            continue
        feat = feat_sum / n
        base = base_sum / n
        pred_gain[int(cid)] = {
            "delta_L": float(base - feat),
            "feat_loss": float(feat),
            "base_loss": float(base),
            "te_true": float(cell.get("te_cell_realised",
                                      cell.get("target_te", float("nan")))),
            "M": int(cell.get("M", 0)),
            "band": str(cell.get("band", "")),
            "target_te": float(cell.get("target_te", float("nan"))),
        }
        if attn_acc is not None and attn_cnt > 0:
            attn_prof[int(cid)] = {
                "attn_lag": (attn_acc / attn_cnt).tolist(),
                "delay_max": dmax,
                "band": str(cell.get("band", "")),
            }
    return pred_gain, attn_prof


# =============================================================================
# Per-cell attention LagMass / peak-lag (tasks 5.2-5.3) + lag-walk recovery
# =============================================================================


def _pearson(
    sx: float, sy: float, sxx: float, syy: float, sxy: float, n: float,
) -> float:
    """Streaming Pearson correlation from running sums (``nan`` if degenerate)."""
    if n < 2.0:
        return float("nan")
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    denom = float(np.sqrt(vx * vy))
    return float(cov / denom) if denom > 1e-12 else float("nan")


@torch.no_grad()
def collect_per_cell_attn_diag(
    model: Any,
    dataset: SyntheticTEDataset,
    cells_by_id: Dict[int, Dict[str, Any]],
    cell_id_array: np.ndarray,
    device: torch.device,
    *,
    warmup: int,
    horizon: int,
    T: int,
    max_lag: int,
    eval_cfg: Dict[str, Any],
    do_lag_walk: bool = True,
) -> Dict[int, Dict[str, Any]]:
    r"""Per-cell attention LagMass / peak-lag (tasks 5.2-5.3) + lag-walk recovery.

    One light encoder-only ``@torch.no_grad`` pass per cell (no decoder, no
    backward). For each cell it accumulates the head-averaged $K_t$-weighted lag
    map $\widetilde{\mathrm{TE}}_{t,\ell} = K_t\,\bar\alpha_{t,\ell}$ over the
    cell's samples and, when the cache carries the per-step true lag walk
    ``true_lag_tt`` ($d_t$), the per-step recovered lag
    $\hat d_t = \arg\max_\ell \bar\alpha_{t,\ell}$.

    From the mean lag map it derives the attention LagMass (task 5.2, with the
    uniform baseline + ratio) via :func:`lag_recovery.compute_lag_mass_attn` and
    the per-anchor peak-lag error / in-band fraction (task 5.3) via
    :func:`lag_recovery.compute_peak_lag_error` -- exactly the two diagnostics
    that :func:`per_cell_lag_recovery` (LOLO-only) does not produce. The
    lag-walk block reports the clean-window per-step $|\hat d_t - d_t|$ MAE, the
    within-1 fraction, the Pearson correlation and the mean true / recovered
    trajectories (for the figure).

    Args:
        model: The trained model in ``eval`` mode.
        dataset: The mixed-cache dataset.
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.
        cell_id_array: Per-sample ``sample_cell_id`` (dataset order).
        device: Compute device.
        warmup: Warm-up floor.
        horizon: Forecast horizon $H$ (clean-window upper bound is $T-H$).
        T: Sequence length.
        max_lag: Maximum attention lag ($L = L_{\max}+1$).
        eval_cfg: The ``benchmarks.G1_mix.eval`` block (``n_lolo_per_cell`` cap,
            ``batch_size``).
        do_lag_walk: When ``True`` and the cache carries ``true_lag_tt``, also
            compute the per-step lag-walk recovery.

    Returns:
        ``{cell_id: {...}}`` -- each row carries ``lag_mass_attn`` /
        ``lag_mass_attn_uniform`` / ``lag_mass_attn_ratio`` /
        ``peak_lag_err_mean`` / ``peak_lag_err_median`` / ``peak_in_band_frac``
        and, when ``true_lag_tt`` is present, ``lag_walk_mae`` /
        ``lag_walk_within1_frac`` / ``lag_walk_corr`` / ``lag_walk_true_mean`` /
        ``lag_walk_pred_mean`` (length-$T$ lists) / ``lag_walk_band`` /
        ``lag_walk_delay_max``.
    """
    cap = eval_cfg.get("n_lolo_per_cell")
    cap = None if cap is None else int(cap)
    bs = int(eval_cfg.get("batch_size") or 32)

    out: Dict[int, Dict[str, Any]] = {}
    for cid, cell in sorted(cells_by_id.items()):
        loader = _cell_subset_loader(dataset, cid, cell_id_array, bs, cap=cap)
        if loader is None:
            continue
        dmax = int(cell["delay_max"])
        band = list(range(dmax))
        anchor_lo = max(int(warmup), dmax - 1)
        anchor_hi = int(T - horizon)

        te_sum: Optional[np.ndarray] = None
        n_samples = 0
        lw_err_sum = lw_within1 = lw_count = 0.0
        lw_sx = lw_sy = lw_sxx = lw_syy = lw_sxy = 0.0
        true_sum_t: Optional[np.ndarray] = None
        pred_sum_t: Optional[np.ndarray] = None
        n_walk = 0
        have_walk = False

        for batch in loader:
            batch = move_batch(batch, device)
            y_st, y_ph = batch.fhr_st, batch.fhr_ph
            enc = model.encode_only(y_st, y_ph, build_u_stream(batch), sample_z=True)
            kld = model.kld_tensor(
                mu_prior=enc["mu_prior"], logvar_prior=enc["logvar_prior"],
                mu_post=enc["mu_post"], logvar_post=enc["logvar_post"],
                mask_warmup=False,
            )                                                    # (B, T, d_z)
            kld_bt = kld.sum(dim=-1)                             # (B, T)
            mean_alpha = enc["attn_weights"].mean(dim=2)         # (B, T, L)
            te_lag = kld_bt.unsqueeze(-1) * mean_alpha           # (B, T, L)
            te_b = te_lag.sum(dim=0).detach().cpu().to(torch.float64).numpy()
            te_sum = te_b if te_sum is None else te_sum + te_b
            n_samples += int(y_st.size(0))

            tlt = batch.get("true_lag_tt") if do_lag_walk else None
            if tlt is not None:
                have_walk = True
                pred_lag = mean_alpha.argmax(dim=-1)             # (B, T) long
                true_lag = tlt.to(device=pred_lag.device, dtype=torch.long)
                Bn, Tn = int(pred_lag.shape[0]), int(pred_lag.shape[1])
                lo = min(anchor_lo, Tn)
                hi = min(anchor_hi, Tn)
                if hi > lo:
                    p = pred_lag[:, lo:hi].to(torch.float64)
                    t = true_lag[:, lo:hi].to(torch.float64)
                    err = (p - t).abs()
                    lw_err_sum += float(err.sum())
                    lw_within1 += float((err <= 1.0).sum())
                    lw_count += float(err.numel())
                    lw_sx += float(t.sum())
                    lw_sy += float(p.sum())
                    lw_sxx += float((t * t).sum())
                    lw_syy += float((p * p).sum())
                    lw_sxy += float((t * p).sum())
                t_full = true_lag.to(torch.float64).sum(dim=0).cpu().numpy()
                p_full = pred_lag.to(torch.float64).sum(dim=0).cpu().numpy()
                true_sum_t = t_full if true_sum_t is None else true_sum_t + t_full
                pred_sum_t = p_full if pred_sum_t is None else pred_sum_t + p_full
                n_walk += Bn

        if te_sum is None or n_samples == 0:
            continue
        te_mean = te_sum / float(n_samples)
        lm = compute_lag_mass_attn(
            te_mean, lag_band=band, anchor_lo=anchor_lo,
            anchor_hi=anchor_hi, max_lag=max_lag,
        )
        anchor_slice = (
            te_mean[anchor_lo:anchor_hi, :] if anchor_hi > anchor_lo else te_mean
        )
        pk = compute_peak_lag_error(anchor_slice, lag_band=band)
        row: Dict[str, Any] = {
            "lag_mass_attn": lm["lag_mass_attn"],
            "lag_mass_attn_uniform": lm["uniform_baseline"],
            "lag_mass_attn_ratio": lm["ratio_to_uniform"],
            "peak_lag_err_mean": pk["peak_lag_err_mean"],
            "peak_lag_err_median": pk["peak_lag_err_median"],
            "peak_in_band_frac": pk["peak_in_band_frac"],
        }
        if have_walk and lw_count > 0:
            row["lag_walk_mae"] = lw_err_sum / lw_count
            row["lag_walk_within1_frac"] = lw_within1 / lw_count
            row["lag_walk_corr"] = _pearson(
                lw_sx, lw_sy, lw_sxx, lw_syy, lw_sxy, lw_count,
            )
            if n_walk > 0 and true_sum_t is not None and pred_sum_t is not None:
                row["lag_walk_true_mean"] = (true_sum_t / n_walk).tolist()
                row["lag_walk_pred_mean"] = (pred_sum_t / n_walk).tolist()
                row["lag_walk_band"] = str(cell.get("band", ""))
                row["lag_walk_delay_max"] = dmax
        out[int(cid)] = row
    return out


def _attach_attn_diag(
    arrs: Dict[str, Any],
    rows: List[Dict[str, Any]],
    model: Any,
    dataset: SyntheticTEDataset,
    cells_by_id: Dict[int, Dict[str, Any]],
    cell_id_array: np.ndarray,
    device: torch.device,
    *,
    warmup: int,
    horizon: int,
    T: int,
    max_lag: int,
    eval_cfg: Dict[str, Any],
    do_lag_walk: bool = True,
) -> None:
    r"""Run :func:`collect_per_cell_attn_diag` and merge into rows / arrs.

    Defensive: a forward failure leaves the new fields as ``nan`` (so the CSV
    columns still exist) and the lag-walk trajectory dict empty -- the encoder
    calibration / LOLO artifacts stay valid.

    Args:
        arrs: Per-sample dict (mutated: ``lag_walk_by_cell`` added).
        rows: Per-cell rows (mutated: ``lag_mass_attn`` / ``peak_lag_*`` /
            ``lag_walk_*`` added per row).
        model, dataset, cells_by_id, cell_id_array, device: As for
            :func:`collect_per_cell_attn_diag`.
        warmup, horizon, T, max_lag, eval_cfg, do_lag_walk: Forwarded unchanged.
    """
    _new_fields = (
        "lag_mass_attn", "lag_mass_attn_ratio", "peak_lag_err_mean",
        "peak_in_band_frac", "lag_walk_mae", "lag_walk_within1_frac",
        "lag_walk_corr",
    )
    for row in rows:
        for k in _new_fields:
            row.setdefault(k, float("nan"))
    try:
        diag = collect_per_cell_attn_diag(
            model, dataset, cells_by_id, cell_id_array, device,
            warmup=warmup, horizon=horizon, T=T, max_lag=max_lag,
            eval_cfg=eval_cfg, do_lag_walk=do_lag_walk,
        )
    except Exception as exc:  # noqa: BLE001 -- diagnostics must never gate eval
        print(f"[mixed_eval] attention / lag-walk diagnostics skipped: {exc}")
        diag = {}
    walk: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        d = diag.get(int(row["cell_id"]), {})
        for k in _new_fields:
            if k in d:
                row[k] = d[k]
        if "lag_walk_true_mean" in d:
            walk[int(row["cell_id"])] = {
                "true_mean": d["lag_walk_true_mean"],
                "pred_mean": d["lag_walk_pred_mean"],
                "band": d.get("lag_walk_band", ""),
                "delay_max": d.get("lag_walk_delay_max", 0),
                "mae": d.get("lag_walk_mae", float("nan")),
            }
    arrs["lag_walk_by_cell"] = walk


def lag_recovery_summary(
    rows: List[Dict[str, Any]], *, threshold: float = 0.8,
) -> Dict[str, Any]:
    r"""Roll the per-cell LagMass up to the criterion-#4 pass/fail fractions.

    Success criterion #4 (``model_validation_v3_mixed.md`` §6) asks for per-cell
    LagMass $\ge 0.8$. The per-cell ``lag_mass_lolo`` (LOLO) and ``lag_mass_attn``
    (attention) are collected by the eval but were never thresholded; this
    summary reports the fraction of cells clearing ``threshold`` overall and
    per $M$ / per band.

    Args:
        rows: Per-cell rows carrying ``lag_mass_lolo`` / ``lag_mass_attn`` /
            ``M`` / ``band``.
        threshold: The LagMass pass threshold (default $0.8$).

    Returns:
        ``{}`` when ``rows`` is empty, else a dict with the overall and
        per-$M$ / per-band pass fractions and mean LagMass.
    """
    if not rows:
        return {}

    def _frac_ge(vals: Sequence[float]) -> float:
        a = np.asarray(list(vals), dtype=float)
        a = a[np.isfinite(a)]
        return float(np.mean(a >= threshold)) if a.size else float("nan")

    def _mean(vals: Sequence[float]) -> float:
        a = np.asarray(list(vals), dtype=float)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    lolo = [r.get("lag_mass_lolo", float("nan")) for r in rows]
    attn = [r.get("lag_mass_attn", float("nan")) for r in rows]
    by_M: Dict[str, Any] = {}
    for m in sorted({int(r["M"]) for r in rows}):
        sub = [r for r in rows if int(r["M"]) == m]
        by_M[str(m)] = {
            "frac_lolo_ge": _frac_ge([r.get("lag_mass_lolo", float("nan")) for r in sub]),
            "frac_attn_ge": _frac_ge([r.get("lag_mass_attn", float("nan")) for r in sub]),
            "mean_lolo": _mean([r.get("lag_mass_lolo", float("nan")) for r in sub]),
        }
    by_band: Dict[str, Any] = {}
    for b in sorted({str(r.get("band", "")) for r in rows}):
        sub = [r for r in rows if str(r.get("band", "")) == b]
        by_band[b] = {
            "frac_lolo_ge": _frac_ge([r.get("lag_mass_lolo", float("nan")) for r in sub]),
            "frac_attn_ge": _frac_ge([r.get("lag_mass_attn", float("nan")) for r in sub]),
        }
    return {
        "threshold": float(threshold),
        "n_cells": len(rows),
        "frac_cells_lolo_ge": _frac_ge(lolo),
        "frac_cells_attn_ge": _frac_ge(attn),
        "mean_lag_mass_lolo": _mean(lolo),
        "mean_lag_mass_attn": _mean(attn),
        "by_M": by_M,
        "by_band": by_band,
    }


def _lag_walk_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Aggregate the per-cell lag-walk recovery (``{}`` if never computed)."""
    mae = np.asarray(
        [r.get("lag_walk_mae", float("nan")) for r in rows], dtype=float
    )
    mae = mae[np.isfinite(mae)]
    if mae.size == 0:
        return {}
    w = np.asarray(
        [r.get("lag_walk_within1_frac", float("nan")) for r in rows], dtype=float
    )
    c = np.asarray(
        [r.get("lag_walk_corr", float("nan")) for r in rows], dtype=float
    )
    w = w[np.isfinite(w)]
    c = c[np.isfinite(c)]
    return {
        "n_cells": int(mae.size),
        "mean_mae": float(mae.mean()),
        "max_mae": float(mae.max()),
        "mean_within1_frac": float(w.mean()) if w.size else float("nan"),
        "mean_corr": float(c.mean()) if c.size else float("nan"),
    }


def _cell_eval_meta(
    cell: Dict[str, Any], horizon: int, T: int,
) -> Dict[str, Any]:
    r"""Single-cell-style ``meta`` for one mixed-pool cell.

    Superset of :func:`_cell_meta`: adds the scalar provenance
    (``te_true`` / ``te_per_step`` / ``M`` / ``delay_*`` / ``B_y`` / ``band``)
    that the injected single-cell evaluators
    (:func:`evaluate_te.evaluate_checkpoint`,
    :func:`lag_recovery.analyze_lag_recovery`) read from a cache ``meta.json``.
    Used by :mod:`mixed_per_cell_diag` so those functions run per sub-population
    without materialising a one-cell cache on disk.

    Args:
        cell: A manifest cell dict.
        horizon: Forecast horizon $H$.
        T: Sequence length.

    Returns:
        The per-cell ``meta`` dict.
    """
    dmax = int(cell["delay_max"])
    te_true = float(
        cell.get("te_cell_realised", cell.get("target_te", float("nan")))
    )
    meta = _cell_meta(cell, horizon, T)
    meta.update({
        "te_true": te_true,
        "te_per_step": te_true / float(horizon) if horizon else float("nan"),
        "M": int(cell.get("M", 0)),
        "delay_min": int(cell.get("delay_min", 0)),
        "delay_max": dmax,
        "delay": dmax,
        "B_y": float(cell.get("B_y_scalar", float("nan"))),
        "target_te": float(cell.get("target_te", float("nan"))),
        "band": str(cell.get("band", "")),
    })
    return meta


# =============================================================================
# Orchestration
# =============================================================================

def _load_mixed(
    config: Dict[str, Any], tag: str,
) -> Tuple[SyntheticTEDataset, Dict[str, Any], np.ndarray]:
    """Load a mixed cache's test split + its manifest + per-sample cell ids.

    Args:
        config: The parsed config (for ``paths.data_dir``).
        tag: Cache tag under ``data/G1_mix/``.

    Returns:
        ``(dataset, mixture_manifest, cell_id_array)``.

    Raises:
        FileNotFoundError: If the cache's ``test.npz`` is absent.
    """
    data_root = resolve_user_path(config["paths"]["data_dir"])
    test_npz = data_root / _BENCHMARK / tag / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(
            f"mixed test split not found: {test_npz}. Build it with "
            f"`python -m ...synthetic.mixed_dataset --tag {tag}"
            f"{' --holdout' if tag.endswith('_holdout') else ''}`."
        )
    dataset = SyntheticTEDataset(test_npz)
    manifest = dataset.meta.get("mixture", {})
    with np.load(test_npz) as npz:
        cell_id_array = np.asarray(npz["sample_cell_id"], dtype=int)
    return dataset, manifest, cell_id_array


def _cells_by_id(manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index the manifest cell list by ``cell_id``."""
    return {int(c["cell_id"]): c for c in manifest.get("cells", [])}


def _attach_lag_profiles(
    arrs: Dict[str, Any],
    lag_dict: Dict[int, Dict[str, Any]],
    cells_by_id: Dict[int, Dict[str, Any]],
) -> None:
    r"""Store the per-cell LOLO $A_\ell$ profile into ``arrs['lag_profile_by_cell']``.

    The sliding-window LOLO already produced ``A_lag`` / ``lag_grid`` per cell;
    they are retained here (keyed by ``cell_id``, with the cell's ``delay_max`` /
    ``band``) so the lag-profile figures can show the actual per-lag importance
    curve with the true band shaded, not just the scalar ``lag_mass_lolo``.

    Args:
        arrs: The per-sample array dict (mutated in place).
        lag_dict: ``{cell_id -> per_cell_lag_recovery output}``.
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.
    """
    prof: Dict[int, Dict[str, Any]] = {}
    for cid, lr in lag_dict.items():
        cell = cells_by_id.get(int(cid), {})
        prof[int(cid)] = {
            "A_lag": [float(x) for x in lr.get("A_lag", [])],
            "lag_grid": [int(x) for x in lr.get("lag_grid", [])],
            "delay_max": int(cell.get("delay_max", 0)),
            "band": str(cell.get("band", "")),
        }
    arrs["lag_profile_by_cell"] = prof


def _attach_pred_gain(
    arrs: Dict[str, Any],
    rows: List[Dict[str, Any]],
    model: Any,
    dataset: SyntheticTEDataset,
    cells_by_id: Dict[int, Dict[str, Any]],
    cell_id_array: np.ndarray,
    device: torch.device,
    *,
    warmup: int,
    horizon: int,
    loss_settings: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> None:
    r"""Run the decoder-pass collector and merge $\Delta\mathcal L$ / attention.

    Defensive: a non-Gaussian-NLL checkpoint (or any decoder failure) leaves the
    ``pred_gain``/``attn_profile`` structures empty so the ΔL figures simply skip
    -- the encoder-side calibration / lag artifacts remain valid.

    Args:
        arrs: The per-sample array dict (mutated: ``pred_gain_by_cell`` /
            ``attn_profile_by_cell`` added).
        rows: The per-cell rows (mutated: ``delta_L`` / ``feat_loss`` /
            ``base_loss`` added per row).
        model, dataset, cells_by_id, cell_id_array, device: As for
            :func:`collect_per_cell_pred_gain`.
        warmup, horizon, loss_settings, eval_cfg: Forwarded unchanged.
    """
    try:
        pred_gain, attn_prof = collect_per_cell_pred_gain(
            model, dataset, cells_by_id, cell_id_array, device,
            warmup=warmup, horizon=horizon, loss_settings=loss_settings,
            eval_cfg=eval_cfg,
        )
    except Exception as exc:  # noqa: BLE001 -- ΔL must never gate the eval
        print(f"[mixed_eval] prediction-gain collection skipped: {exc}")
        pred_gain, attn_prof = {}, {}
    arrs["pred_gain_by_cell"] = pred_gain
    arrs["attn_profile_by_cell"] = attn_prof
    for row in rows:
        pg = pred_gain.get(int(row["cell_id"]), {})
        row["delta_L"] = pg.get("delta_L", float("nan"))
        row["feat_loss"] = pg.get("feat_loss", float("nan"))
        row["base_loss"] = pg.get("base_loss", float("nan"))


def percell_association(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""KLD-vs-TE association computed on **per-cell** aggregates.

    Unlike :func:`kld_te_association` (per-sample, diluted by within-cell KLD
    variance), this uses one $(\mathrm{TE}_{\mathrm{true}}, \bar K)$ point per
    cell -- the statistically correct granularity for the headline calibration.

    Args:
        rows: Per-cell rows carrying ``te_true`` / ``kbar_mean`` / ``M``.

    Returns:
        ``{"overall": stats, "by_M": {M: stats}}`` with :func:`_assoc_stats`
        dicts.
    """
    te = np.asarray([r["te_true"] for r in rows], dtype=float)
    kb = np.asarray([r["kbar_mean"] for r in rows], dtype=float)
    m = np.asarray([r["M"] for r in rows], dtype=int)
    by_M: Dict[str, Any] = {}
    for mm in sorted(set(m.tolist())):
        sel = m == mm
        by_M[str(int(mm))] = _assoc_stats(kb[sel], te[sel])
    return {"overall": _assoc_stats(kb, te), "by_M": by_M}


def _pred_gain_summary(arrs: Dict[str, Any]) -> Dict[str, Any]:
    r"""Summarise per-cell $\Delta\mathcal L$ (mean, fraction positive, range).

    Args:
        arrs: The per-sample dict carrying ``pred_gain_by_cell``.

    Returns:
        ``{"n_cells", "mean_delta_L", "frac_positive", "min_delta_L",
        "max_delta_L"}`` (empty dict if no ΔL was collected).
    """
    pg = arrs.get("pred_gain_by_cell", {}) or {}
    vals = np.asarray([v["delta_L"] for v in pg.values()], dtype=float)
    if vals.size == 0:
        return {}
    return {
        "n_cells": int(vals.size),
        "mean_delta_L": float(np.nanmean(vals)),
        "frac_positive": float(np.mean(vals > 0.0)),
        "min_delta_L": float(np.nanmin(vals)),
        "max_delta_L": float(np.nanmax(vals)),
    }


def evaluate_mixed(
    config: Dict[str, Any],
    *,
    run_tag: str,
    in_mix_tag: str,
    holdout_tag: Optional[str] = None,
    ckpt_name: str = "final.ckpt",
    out_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Evaluate the ``G1_mix`` model per-group on in-mix + held-out caches.

    Args:
        config: The parsed config carrying ``benchmarks.G1_mix``.
        run_tag: Training run under ``results/G1_mix/<run_tag>/``.
        in_mix_tag: In-mix cache tag (``data/G1_mix/<in_mix_tag>/``).
        holdout_tag: Optional held-out cache tag for extrapolation.
        ckpt_name: Checkpoint file name (``final.ckpt`` or ``best.ckpt``).
        out_subdir: Output subdirectory under the run dir. ``None`` keeps the
            default ``mixed_eval``; pass a distinct name (e.g.
            ``mixed_eval_extrap_m64``) when evaluating several held-out /
            extrapolation caches against the same checkpoint, so each pass
            keeps its own artifacts instead of overwriting the previous one.

    Returns:
        The full metrics dict (also written to ``metrics.json``).
    """
    device = resolve_device(config.get("runtime", {}))
    results_root = resolve_user_path(config["paths"]["results_dir"])
    run_dir = results_root / _BENCHMARK / run_tag
    ckpt_path = run_dir / ckpt_name
    model, ckpt = load_eval_checkpoint(ckpt_path, device)
    loss_settings = dict(ckpt.get("loss_settings", {}))

    model_cfg = config["model"]
    horizon = int(model_cfg["horizon"])
    max_lag = int(model_cfg["max_lag"])
    eval_cfg = config["benchmarks"][_BENCHMARK].get("eval", {}) or {}
    warmup_cfg = eval_cfg.get("warmup")
    warmup = int(model_cfg["warmup_period"]) if warmup_cfg is None else int(warmup_cfg)
    controls = tuple(eval_cfg.get("null_controls", ["shuffle", "reverse"]))
    bs = int(eval_cfg.get("batch_size") or config.get("optim", {}).get("batch_size", 32))
    lag_walk_cfg = eval_cfg.get("lag_walk", {}) or {}
    do_lag_walk = bool(lag_walk_cfg.get("enabled", True))
    lag_mass_threshold = float(eval_cfg.get("lag_mass_threshold", 0.8))

    out_dir = run_dir / (out_subdir or _OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- in-mix recovery -----------------------------------------------------
    print(f"[mixed_eval] in-mix cache '{in_mix_tag}' ...")
    ds_in, manifest_in, cellids_in = _load_mixed(config, in_mix_tag)
    cells_in = _cells_by_id(manifest_in)
    T = int(ds_in.meta.get("sequence_length", model_cfg["sequence_length"]))
    loader_in = make_dataloader(ds_in, bs, shuffle=False, drop_last=False)
    arrs_in = collect_per_sample_kbar(
        model, loader_in, device, warmup=warmup, horizon=horizon,
        controls=controls,
    )
    slices_in = fit_calibration_slices(arrs_in, cells_in)
    slices_in_nullsub = (
        fit_calibration_slices_nullsub(arrs_in, cells_in, control=controls[0])
        if controls else {}
    )
    overall = slices_in["overall"] or {"alpha": 0.0, "gamma": 1.0, "r2": float("nan")}
    rows_in = group_recovery(
        arrs_in, cells_in, alpha=float(overall["alpha"]),
        gamma=float(overall["gamma"]), controls=controls,
    )
    print("[mixed_eval] per-cell lag recovery (in-mix) ...")
    lag_in = per_cell_lag_recovery(
        model, ds_in, cells_in, cellids_in, device, horizon=horizon, T=T,
        max_lag=max_lag, warmup=warmup, loss_settings=loss_settings,
        eval_cfg={**eval_cfg, "batch_size": bs},
    )
    for row in rows_in:
        lr = lag_in.get(row["cell_id"], {})
        row["lag_mass_lolo"] = lr.get("lag_mass_lolo", float("nan"))
        row["peak_lag_err"] = lr.get("peak_lag_err", float("nan"))
    _attach_lag_profiles(arrs_in, lag_in, cells_in)
    print("[mixed_eval] per-cell prediction gain (in-mix) ...")
    _attach_pred_gain(
        arrs_in, rows_in,
        model, ds_in, cells_in, cellids_in, device, warmup=warmup,
        horizon=horizon, loss_settings=loss_settings,
        eval_cfg={**eval_cfg, "batch_size": bs},
    )
    print("[mixed_eval] per-cell attention LagMass / lag-walk (in-mix) ...")
    _attach_attn_diag(
        arrs_in, rows_in,
        model, ds_in, cells_in, cellids_in, device, warmup=warmup,
        horizon=horizon, T=T, max_lag=max_lag,
        eval_cfg={**eval_cfg, "batch_size": bs}, do_lag_walk=do_lag_walk,
    )

    # --- held-out extrapolation ---------------------------------------------
    rows_ho: List[Dict[str, Any]] = []
    slices_ho: Dict[str, Any] = {}
    slices_ho_nullsub: Dict[str, Any] = {}
    arrs_ho: Dict[str, Any] = {}
    if holdout_tag is not None:
        try:
            print(f"[mixed_eval] held-out cache '{holdout_tag}' ...")
            ds_ho, manifest_ho, cellids_ho = _load_mixed(config, holdout_tag)
            cells_ho = _cells_by_id(manifest_ho)
            loader_ho = make_dataloader(ds_ho, bs, shuffle=False, drop_last=False)
            arrs_ho = collect_per_sample_kbar(
                model, loader_ho, device, warmup=warmup, horizon=horizon,
                controls=controls,
            )
            # Score held-out cells on the IN-MIX calibration (extrapolation):
            rows_ho = group_recovery(
                arrs_ho, cells_ho, alpha=float(overall["alpha"]),
                gamma=float(overall["gamma"]), controls=controls,
            )
            slices_ho = fit_calibration_slices(arrs_ho, cells_ho)
            slices_ho_nullsub = (
                fit_calibration_slices_nullsub(arrs_ho, cells_ho, control=controls[0])
                if controls else {}
            )
            lag_ho = per_cell_lag_recovery(
                model, ds_ho, cells_ho, cellids_ho, device, horizon=horizon,
                T=int(ds_ho.meta.get("sequence_length", T)), max_lag=max_lag,
                warmup=warmup, loss_settings=loss_settings,
                eval_cfg={**eval_cfg, "batch_size": bs},
            )
            for row in rows_ho:
                lr = lag_ho.get(row["cell_id"], {})
                row["lag_mass_lolo"] = lr.get("lag_mass_lolo", float("nan"))
                row["peak_lag_err"] = lr.get("peak_lag_err", float("nan"))
            _attach_lag_profiles(arrs_ho, lag_ho, cells_ho)
            _attach_pred_gain(
                arrs_ho, rows_ho,
                model, ds_ho, cells_ho, cellids_ho, device, warmup=warmup,
                horizon=horizon, loss_settings=loss_settings,
                eval_cfg={**eval_cfg, "batch_size": bs},
            )
            _attach_attn_diag(
                arrs_ho, rows_ho,
                model, ds_ho, cells_ho, cellids_ho, device, warmup=warmup,
                horizon=horizon, T=int(ds_ho.meta.get("sequence_length", T)),
                max_lag=max_lag,
                eval_cfg={**eval_cfg, "batch_size": bs}, do_lag_walk=do_lag_walk,
            )
        except FileNotFoundError as exc:
            print(f"[mixed_eval] held-out cache skipped: {exc}")

    generalization = _generalization_gaps(rows_in, rows_ho)

    metrics = {
        "run_tag": run_tag,
        "ckpt": str(ckpt_path),
        "warmup": warmup,
        "horizon": horizon,
        "calibration": {
            "in_mix": slices_in,
            "in_mix_nullsub": slices_in_nullsub,
            "holdout": slices_ho,
            "holdout_nullsub": slices_ho_nullsub,
        },
        "calibration_primary": {
            "in_mix": calibration_primary_summary(slices_in),
            "holdout": calibration_primary_summary(slices_ho) if rows_ho else {},
        },
        "kld_te_association": {
            "in_mix": kld_te_association(arrs_in),
            "holdout": kld_te_association(arrs_ho) if rows_ho else {},
        },
        "kld_te_association_percell": {
            "in_mix": percell_association(rows_in),
            "holdout": percell_association(rows_ho) if rows_ho else {},
        },
        "pred_gain": {
            "in_mix": _pred_gain_summary(arrs_in),
            "holdout": _pred_gain_summary(arrs_ho) if rows_ho else {},
        },
        "lag_recovery_summary": {
            "in_mix": lag_recovery_summary(rows_in, threshold=lag_mass_threshold),
            "holdout": (
                lag_recovery_summary(rows_ho, threshold=lag_mass_threshold)
                if rows_ho else {}
            ),
        },
        "lag_walk": {
            "in_mix": _lag_walk_summary(rows_in),
            "holdout": _lag_walk_summary(rows_ho) if rows_ho else {},
        },
        "n_cells_in_mix": len(rows_in),
        "n_cells_holdout": len(rows_ho),
        "generalization": generalization,
    }

    _write_per_cell_csv(out_dir / "per_cell.csv", rows_in, rows_ho, controls)
    _write_per_sample_csv(out_dir / "per_sample.csv", arrs_in, controls)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump({"in_mix": slices_in, "in_mix_nullsub": slices_in_nullsub,
                   "holdout": slices_ho, "holdout_nullsub": slices_ho_nullsub,
                   "primary_per_M": metrics["calibration_primary"]["in_mix"],
                   "alpha": float(overall["alpha"]),
                   "gamma": float(overall["gamma"])}, fh, indent=2)
    with open(out_dir / "generalization.json", "w", encoding="utf-8") as fh:
        json.dump(generalization, fh, indent=2)

    _render_figures(out_dir, rows_in, rows_ho, arrs_in, slices_in, controls)

    g = overall.get("gamma", float("nan"))
    a = overall.get("alpha", float("nan"))
    prim = metrics["calibration_primary"]["in_mix"]
    gamma_by_M = prim.get("gamma_by_M", {})
    per_m_str = "  ".join(f"M={m}:{gv:.3f}" for m, gv in sorted(
        gamma_by_M.items(), key=lambda kv: int(kv[0])))
    print(
        f"[mixed_eval] done -> {out_dir}\n"
        f"  PRIMARY per-M gamma: {per_m_str or '(none)'}  "
        f"(mean={prim.get('mean_gamma', float('nan')):.3f}, "
        f"frac|gamma-1|<=0.2={prim.get('frac_M_calibrated', float('nan')):.2f})\n"
        f"  overall calibration: gamma={g:.3f} alpha={a:.3f} "
        f"r2={overall.get('r2', float('nan')):.3f} over {len(rows_in)} cells"
    )
    return metrics


def _generalization_gaps(
    rows_in: List[Dict[str, Any]], rows_ho: List[Dict[str, Any]],
) -> Dict[str, Any]:
    r"""Held-out cell metric minus the trained-marginal mean.

    For each held-out cell, compare its TE-RMSE and LagMass to the mean over
    *trained* cells that share its $M$ (the simplest interpolation reference).

    Args:
        rows_in: In-mix per-cell rows.
        rows_ho: Held-out per-cell rows.

    Returns:
        ``{"cells": [...], "summary": {...}}``.
    """
    if not rows_ho:
        return {"cells": [], "summary": {}}
    by_M_rmse: Dict[int, List[float]] = {}
    by_M_lag: Dict[int, List[float]] = {}
    for r in rows_in:
        by_M_rmse.setdefault(r["M"], []).append(r.get("te_rmse", float("nan")))
        by_M_lag.setdefault(r["M"], []).append(r.get("lag_mass_lolo", float("nan")))
    cells: List[Dict[str, Any]] = []
    for r in rows_ho:
        ref_rmse = float(np.nanmean(by_M_rmse.get(r["M"], [float("nan")])))
        ref_lag = float(np.nanmean(by_M_lag.get(r["M"], [float("nan")])))
        cells.append({
            "M": r["M"], "target_te": r["target_te"], "band": r["band"],
            "te_rmse": r.get("te_rmse"), "te_rmse_ref": ref_rmse,
            "te_rmse_gap": float(r.get("te_rmse", float("nan")) - ref_rmse),
            "lag_mass_lolo": r.get("lag_mass_lolo"), "lag_mass_ref": ref_lag,
            "lag_mass_gap": float(r.get("lag_mass_lolo", float("nan")) - ref_lag),
        })
    return {
        "cells": cells,
        "summary": {
            "mean_te_rmse_gap": float(np.nanmean([c["te_rmse_gap"] for c in cells])),
            "mean_lag_mass_gap": float(np.nanmean([c["lag_mass_gap"] for c in cells])),
        },
    }


# =============================================================================
# CSV / figure writers
# =============================================================================

_BASE_CELL_FIELDS = [
    "split", "cell_id", "M", "target_te", "band", "delay_min", "delay_max",
    "B_y", "te_true", "n", "kbar_mean", "kbar_std", "te_pred_mean", "te_rmse",
    "te_bias", "lag_mass_lolo", "peak_lag_err",
    "lag_mass_attn", "lag_mass_attn_ratio", "peak_lag_err_mean",
    "peak_in_band_frac", "lag_walk_mae", "lag_walk_within1_frac",
    "lag_walk_corr", "delta_L", "feat_loss",
    "base_loss", "held_out",
]


def _write_per_cell_csv(
    path: Path, rows_in: List[Dict[str, Any]], rows_ho: List[Dict[str, Any]],
    controls: Sequence[str],
) -> None:
    """Write the combined in-mix + held-out per-cell table."""
    fields = list(_BASE_CELL_FIELDS)
    for ctrl in controls:
        fields += [f"null_{ctrl}_kbar", f"null_{ctrl}_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for split, rows in (("in_mix", rows_in), ("holdout", rows_ho)):
            for r in rows:
                writer.writerow({**r, "split": split})


def _write_per_sample_csv(
    path: Path, arrs: Dict[str, np.ndarray], controls: Sequence[str],
) -> None:
    """Write the in-mix per-sample table (cell id / M / band / kbar / te)."""
    fields = ["cell_id", "M", "band_id", "delay_max", "te_true", "kbar"]
    for ctrl in controls:
        fields.append(f"kbar_{ctrl}")
    n = arrs["kbar"].shape[0]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for i in range(n):
            writer.writerow([
                int(arrs["cell_id"][i]), int(arrs["M"][i]),
                int(arrs["band_id"][i]), int(arrs["delay_max"][i]),
                float(arrs["te_true"][i]), float(arrs["kbar"][i]),
                *[float(arrs[f"kbar_{c}"][i]) for c in controls],
            ])


# Shared semantic maps (one source of truth in plot_style) so mixed_eval and
# mixed_calibration colour M / mark bands identically; ``ps.color_for_M`` adds a
# deterministic fallback for the M-extrapolation caches (M=4 / M=64).
_M_COLORS = ps.M_COLORS
_BAND_MARKERS = ps.BAND_MARKERS


def _caption(fig, text: str) -> None:
    r"""Add an italic "how to read this" caption beneath a figure.

    Placed just below the figure box (figure $y<0$) so it never overlaps the
    axes; :func:`plot_style.save_figure` writes with a tight bounding box, which
    captures the caption. Keep ``text`` to roughly one line in the form
    "what is plotted -- what good looks like".

    Args:
        fig: The figure to annotate.
        text: The caption string (plain text; ``wrap`` re-flows long lines).
    """
    fig.text(0.5, -0.015, text, ha="center", va="top", fontsize=7.0,
             color=ps.COLOR_GRAY, style="italic", wrap=True)


def _render_figures(
    out_dir: Path,
    rows_in: List[Dict[str, Any]],
    rows_ho: List[Dict[str, Any]],
    arrs: Dict[str, np.ndarray],
    slices: Dict[str, Any],
    controls: Sequence[str],
) -> None:
    """Render the per-group / generalization figures (defensive).

    Each figure is wrapped so a plotting failure never aborts the eval (the
    JSON / CSV artifacts are the source of truth).
    """
    ps.apply_style()
    for name, fn in (
        # Calibration + recovery (per-cell). The headline KLD-vs-TE view is the
        # ``kld_vs_te`` master grid (M rows x band cols); ``kld_vs_te_overview``
        # shows the two KLD scales (dim-summed nats vs per-dim) side by side; and
        # ``calibration_health`` collapses the per-M slopes into a pass/fail view.
        ("kld_vs_te", _fig_kld_vs_te),
        ("kld_vs_te_overview", _fig_kld_vs_te_overview),
        ("calibration_health", _fig_calibration_health),
        ("te_recovery_percell", _fig_te_recovery_percell),
        ("prior_mismatch", _fig_prior_mismatch),
        # Prediction gain (decoder pass)
        ("pred_gain_vs_te", _fig_pred_gain_vs_te),
        ("pred_gain_vs_kbar", _fig_pred_gain_vs_kbar),
        # Lag diagnostics
        ("lag_band_recovery", _fig_lag_recovery),
        ("lag_profiles", _fig_lag_profiles),
        ("attn_vs_lag", _fig_attn_vs_lag),
        ("lag_walk", _fig_lag_walk),
        # KLD structure
        ("per_dim_kl", _fig_per_dim_kl),
        ("kld_vs_time", _fig_kld_vs_time),
        # Controls + generalization
        ("null_controls", _fig_null_controls),
        ("generalization_gap", _fig_generalization),
    ):
        try:
            fn(out_dir / name, rows_in, rows_ho, arrs, slices, controls)
        except Exception as exc:  # noqa: BLE001 -- a plot must never gate eval
            print(f"[mixed_eval] figure '{name}' skipped: {exc}")


def _fig_kld_vs_te(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Master KLD-vs-TE grid -- rows $= M$, cols $=$ lag-band ("all together").

    The single headline view of the calibration across every axis at once. Each
    panel is one $(M, \text{band})$ cell-group and shows three layers:

    * the faint per-sample $\bar K$ cloud (jittered over the few discrete
      true-TE levels so each cell reads as a vertical column);
    * the per-cell mean $\bar K$ with a $\pm\,$std error bar on top
      (filled $=$ in-mix, hollow $=$ held-out / extrapolation);
    * the panel's own OLS line with its slope $\gamma$ annotated (fitted on the
      cell means; falls back to the per-$M$ slope when a panel has fewer than
      two distinct true-TE values).

    Each $M$-row gets its **own** adaptive $\bar K$ scale (``sharey="row"``)
    because $\bar K$ grows with the channel count $M$; the old figures shared a
    single TE-locked scale that crushed every point into a thin strip. There is
    **no** $y=x$ line: at the default $\beta$ the bottleneck is under-regularised
    so $\gamma \gg 1$, which would make $y=x$ misleading (see the banner).

    Args:
        path: Output path stem (``.pdf`` / ``.png`` appended by ``save_figure``).
        rows_in: In-mix per-cell rows.
        rows_ho: Held-out / extrapolation per-cell rows.
        arrs: In-mix per-sample arrays (``kbar`` / ``te_true`` / ``M`` /
            ``band_id`` and the ``per_dim_kl_by_M`` diagnostic for $d_z$).
        slices: The calibration slices (``overall`` / ``by_M`` / ``by_band``).
        controls: Null-control names (unused here; kept for signature parity).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    all_rows = rows_in + rows_ho
    ms, tes, bands = _grid_axes(all_rows)
    if not (ms and bands):
        return
    by_M = slices.get("by_M", {}) or {}
    cells = _cells_by_M_band(all_rows)
    rows_by_m: Dict[int, List[Dict[str, Any]]] = {}
    for r in all_rows:
        rows_by_m.setdefault(int(r["M"]), []).append(r)
    extrap_ms = {m for m, rs in rows_by_m.items()
                 if rs and all(int(x.get("held_out", 0)) for x in rs)}

    te = np.asarray(arrs.get("te_true", []), dtype=float)
    kb = np.asarray(arrs.get("kbar", []), dtype=float)
    m_arr = np.asarray(arrs.get("M", []), dtype=int)
    band_arr = np.asarray(arrs.get("band_id", []), dtype=int)
    # Horizontal jitter so each cell's single true-TE value reads as a column.
    levels = (np.array(sorted(set(np.round(te, 6).tolist())), dtype=float)
              if te.size else np.array([0.0]))
    span = float(np.min(np.diff(levels))) if levels.size > 1 else 1.0
    rng = np.random.default_rng(0)
    jitter = (rng.random(te.size) - 0.5) * 0.18 * span if te.size else np.zeros(0)
    te_hi = (max(tes) if tes else 1.0) * 1.05 + 1e-6
    te_lo = (min(tes) if tes else 0.0) - 0.12
    xs_line = np.linspace(0.0, te_hi, 50)

    fig, axes = plt.subplots(
        len(ms), len(bands),
        figsize=(3.4 * len(bands), 2.7 * len(ms) + 0.8),
        squeeze=False, sharex=True, sharey="row",
    )
    for ri, m in enumerate(ms):
        color = ps.color_for_M(m)
        row_vals: List[float] = []
        for ci, band in enumerate(bands):
            ax = axes[ri][ci]
            bid = _BAND_ID.get(band)
            if te.size and bid is not None:
                sel = (m_arr == m) & (band_arr == bid)
                if np.any(sel):
                    ax.scatter(te[sel] + jitter[sel], kb[sel], s=5, alpha=0.16,
                               linewidths=0.0, color=color, zorder=1)
                    row_vals.extend(kb[sel].tolist())
            cell_te: List[float] = []
            cell_kb: List[float] = []
            for r in cells.get((m, band), []):
                filled = not int(r.get("held_out", 0))
                ax.errorbar(
                    r["te_true"], r["kbar_mean"],
                    yerr=float(r.get("kbar_std", 0.0)), fmt="o", ms=5,
                    color=color, markerfacecolor=color if filled else "none",
                    markeredgecolor=color, ecolor=color, elinewidth=1.0,
                    capsize=2, zorder=3,
                )
                cell_te.append(float(r["te_true"]))
                cell_kb.append(float(r["kbar_mean"]))
                row_vals.append(float(r["kbar_mean"]))
            fit: Optional[Dict[str, Any]] = None
            if len(set(np.round(cell_te, 6).tolist())) >= 2:
                cand = _assoc_stats(np.asarray(cell_kb), np.asarray(cell_te))
                if np.isfinite(cand.get("gamma", float("nan"))):
                    fit = cand
            if fit is None:
                fit = by_M.get(str(m))
            if fit and np.isfinite(fit.get("gamma", float("nan"))):
                ax.plot(xs_line, fit["alpha"] + fit["gamma"] * xs_line,
                        color=ps.COLOR_BLACK, lw=1.2, alpha=0.85, zorder=2)
                ax.text(0.04, 0.96, fr"$\gamma$={fit['gamma']:.1f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=7, color=ps.COLOR_BLACK)
            if ri == 0:
                ax.set_title(f"band = {band}", fontsize=9)
            if ri == len(ms) - 1:
                ax.set_xlabel(
                    r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
            if ci == 0:
                tag = "  (extrap)" if m in extrap_ms else ""
                ax.set_ylabel(f"M = {m}{tag}\n" + r"$\bar K$  (nats)")
            ps.style_axes(ax)
        if row_vals:
            ps.auto_ylim(axes[ri][0], np.asarray(row_vals, dtype=float),
                         min_zero=True, pad_frac=0.10)
    axes[0][0].set_xlim(te_lo, te_hi)

    legend_handles = [
        Line2D([0], [0], marker="o", ls="", color=ps.COLOR_GRAY,
               markerfacecolor=ps.COLOR_GRAY, label="in-mix"),
        Line2D([0], [0], marker="o", ls="", color=ps.COLOR_GRAY,
               markerfacecolor="none", label="held-out / extrap"),
    ]
    axes[0][-1].legend(handles=legend_handles, fontsize=6, loc="lower right",
                       frameon=False)

    alpha, gamma = _overall_fit(slices)
    sup = r"Master KLD-vs-TE  (rows $= M$, cols $=$ lag-band)"
    if np.isfinite(gamma):
        sup += fr"  —  overall $\gamma$={gamma:.1f}, $\alpha$={alpha:.2f}"
    fig.suptitle(sup, fontsize=11)
    fig.subplots_adjust(top=0.93, hspace=0.22, wspace=0.08)

    dz = _dz_from(arrs)
    dz_txt = (f"$\\bar K$ is summed over $d_z$={dz} latent dims "
              f"(≈ $d_z$ × loss-side kld_loss).  " if dz else
              r"$\bar K$ is summed over the latent dims.  ")
    banner = (dz_txt + r"$\gamma \gg 1$ means the bottleneck is under-regularised "
              r"at this $\beta$; $\gamma \to 1$ ($\bar K \approx \mathrm{TE}$) needs "
              r"$\beta$-selection (mixed_calibration).")
    banner_color = (ps.COLOR_VERMILLION if np.isfinite(gamma) and gamma > 1.5
                    else ps.COLOR_GRAY)
    fig.text(0.5, -0.045, banner, ha="center", va="top", fontsize=7.5,
             color=banner_color, wrap=True)
    _caption(fig, "Per-cell mean K-bar (points, +-std) over the faint per-sample "
                  "cloud, vs true block TE; rows = M, cols = band, each row on "
                  "its own K-bar scale, no y=x. Good: monotone rise with TE and a "
                  "consistent per-panel gamma across bands.")
    ps.save_figure(fig, path)


def _fig_kld_vs_te_overview(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""KLD-vs-TE on **both scales side by side**: dim-summed nats and per-dim.

    One point per cell (colour $= M$, marker $=$ band, filled $=$ in-mix /
    hollow $=$ held-out). The **left** panel plots the headline $\bar K$ in
    dim-summed nats -- the correct unit to compare against the analytic block TE.
    The **right** panel plots the same cells in per-dimension units
    $\bar K / d_z$, the scale of the loss-side ``kld_loss``: it lands near the TE
    range only because of the $\div d_z$ factor, **not** because the model is
    calibrated. Both carry the overall OLS line and the null-source floor (the
    empirical $\mathrm{TE} \approx 0$ anchor).

    Args:
        path: Output path stem.
        rows_in: In-mix per-cell rows.
        rows_ho: Held-out per-cell rows.
        arrs: In-mix per-sample arrays (for $d_z$ and the null floor).
        slices: The calibration slices (overall fit).
        controls: Null-control names (unused; signature parity).
    """
    import matplotlib.pyplot as plt

    all_rows = rows_in + rows_ho
    if not all_rows:
        return
    alpha, gamma = _overall_fit(slices)
    st = percell_association(rows_in).get("overall", {})
    dz = _dz_from(arrs)
    null_key = next((k for k in ("null_shuffle_kbar", "null_reverse_kbar")
                     if rows_in and k in rows_in[0]), None)
    floor = float("nan")
    if null_key is not None:
        vals = [r[null_key] for r in rows_in
                if np.isfinite(r.get(null_key, float("nan")))]
        floor = float(np.mean(vals)) if vals else float("nan")
    te_hi = max((r["te_true"] for r in all_rows), default=1.0) * 1.05 + 1e-6

    panels: List[Tuple[str, float, str]] = [
        ("dim-summed nats (TE unit)", 1.0,
         r"mean latent KL  $\bar K$  (nats, summed over $d_z$)"),
    ]
    if dz:
        panels.append((
            f"per-dim  $\\bar K/d_z$  ($d_z$={dz}, loss-side scale)", 1.0 / dz,
            r"per-dim latent KL  $\bar K/d_z$  (nats)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.0),
                             squeeze=False)
    for ci, (title, scale, ylabel) in enumerate(panels):
        ax = axes[0][ci]
        for r in all_rows:
            r["_kbar_scaled"] = float(r["kbar_mean"]) * scale
        _scatter_cells(ax, rows_in, rows_ho, "te_true", "_kbar_scaled")
        ref_handles = []
        if np.isfinite(gamma):
            xs = np.linspace(0.0, te_hi, 50)
            ref_handles += ax.plot(
                xs, (alpha + gamma * xs) * scale, color=ps.COLOR_BLACK, lw=1.6,
                zorder=4,
                label=fr"OLS: $\gamma$={gamma * scale:.2f} TE+{alpha * scale:.2f}")
        if np.isfinite(floor):
            ref_handles.append(ax.axhline(
                floor * scale, ls=":", lw=1.2, color=ps.COLOR_GREEN, zorder=2,
                label=rf"null floor $\approx$ {floor * scale:.2f} "
                      rf"(TE$\approx$0)"))
        ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        if ci == 0 and st:
            ax.text(0.03, 0.97,
                    f"cells = {int(st.get('n', 0))}\n"
                    f"Pearson r = {st.get('pearson', float('nan')):.3f}\n"
                    f"Spearman $\\rho$ = {st.get('spearman', float('nan')):.3f}\n"
                    f"MI = {st.get('mi', float('nan')):.3f} nats\n"
                    f"$R^2$ = {st.get('r2', float('nan')):.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec=ps.COLOR_GRAY,
                              alpha=0.85))
        ax.legend(handles=ref_handles + _cell_legend_handles(all_rows),
                  fontsize=7, loc="lower right", frameon=False, ncol=2)
        ps.style_axes(ax)
    for r in all_rows:
        r.pop("_kbar_scaled", None)
    fig.suptitle("KLD vs TE on both scales (per cell)", fontsize=11)
    _caption(fig, "Same per-cell points on two y-scales: dim-summed nats (left, "
                  "the TE unit) and per-dim K-bar/d_z (right, the loss-side "
                  "scale). The per-dim panel only looks closer to TE because of "
                  "the /d_z factor -- it is not a calibration.")
    ps.save_figure(fig, path)


def _fig_calibration_health(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Calibration health: per-$M$ slope $\gamma_M$ + local-$\gamma$ heatmaps.

    Collapses the per-$M$ / per-cell calibration into a pass-fail glance. The
    **left** panel is a $\gamma_M$ lollipop with the $|\gamma - 1| \le 0.2$
    target band shaded and $\gamma = 1$ marked; the y-axis switches to log when
    any $\gamma_M \ge 5$ so the calibrated target and an under-regularised
    $\gamma \approx 40$ both stay legible. The **right** panels (one per band)
    show the local slope $\gamma_{\mathrm{loc}} = \bar K / \mathrm{TE}$ per cell
    over the $M \times \mathrm{TE}$ grid (the $\mathrm{TE} = 0$ column is
    dropped) on a diverging colour map centred at $\gamma = 1$: white cells are
    calibrated, red cells over-encode (under-regularised), blue cells
    under-encode.

    Args:
        path: Output path stem.
        rows_in: In-mix per-cell rows.
        rows_ho: Held-out per-cell rows.
        arrs: Per-sample arrays (unused here; signature parity).
        slices: The calibration slices (``by_M`` for the slopes).
        controls: Null-control names (unused; signature parity).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    all_rows = rows_in + rows_ho
    ms, tes, bands = _grid_axes(all_rows)
    if not ms:
        return
    by_M = slices.get("by_M", {}) or {}
    prim = calibration_primary_summary(slices)
    gamma_by_M: Dict[int, float] = {}
    for m in ms:
        fit = by_M.get(str(m))
        if isinstance(fit, dict) and np.isfinite(fit.get("gamma", float("nan"))):
            gamma_by_M[m] = float(fit["gamma"])

    tes_pos = [t for t in tes if t > 1e-9]
    grids = {b: _local_gamma_grid(all_rows, ms, tes_pos, b) for b in bands}
    ncols = 1 + len(bands)
    fig, axes = plt.subplots(
        1, ncols, figsize=(3.2 + 2.7 * len(bands), 4.2), squeeze=False,
        gridspec_kw={"width_ratios": [1.4] + [1.0] * len(bands)},
    )

    axg = axes[0][0]
    xs = np.arange(len(ms))
    axg.axhspan(0.8, 1.2, color=ps.COLOR_SAGE, alpha=0.3, zorder=0,
                label=r"$|\gamma-1|$ <= 0.2")
    axg.axhline(1.0, ls=":", lw=1.0, color=ps.COLOR_BLACK, zorder=1)
    for x, m in zip(xs, ms):
        g = gamma_by_M.get(m, float("nan"))
        if not np.isfinite(g):
            continue
        axg.vlines(x, 1.0, g, color=ps.color_for_M(m), lw=1.6, zorder=2)
        axg.scatter([x], [g], s=70, color=ps.color_for_M(m), zorder=3)
    gammas = np.array([gamma_by_M.get(m, np.nan) for m in ms], dtype=float)
    finite_g = gammas[np.isfinite(gammas)]
    if finite_g.size and float(np.max(finite_g)) >= 5.0:
        axg.set_yscale("log")
    axg.set_xticks(xs)
    axg.set_xticklabels([f"M={m}" for m in ms])
    axg.set_ylabel(r"per-$M$ calibration slope $\gamma_M$")
    title = "Per-M calibration"
    if prim:
        title += (f"\nmean $\\gamma$={prim.get('mean_gamma', float('nan')):.2f},  "
                  f"frac $|\\gamma-1|$<=0.2 = "
                  f"{prim.get('frac_M_calibrated', float('nan')):.2f}")
    axg.set_title(title, fontsize=9)
    axg.legend(fontsize=7, loc="best", frameon=False)
    ps.style_axes(axg)

    finite_vals = np.concatenate(
        [g[np.isfinite(g)].ravel() for g in grids.values()]
    ) if grids else np.array([])
    vmax = max(float(np.max(finite_vals)) if finite_vals.size else 2.0, 1.2)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=vmax)
    im = None
    for bi, band in enumerate(bands):
        ax = axes[0][1 + bi]
        im = ax.imshow(grids[band], aspect="auto", origin="lower",
                       cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(tes_pos)))
        ax.set_xticklabels([f"{t:g}" for t in tes_pos])
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([str(m) for m in ms])
        ax.set_xlabel(r"true block TE (nats)")
        if bi == 0:
            ax.set_ylabel("M")
        ax.set_title(f"local $\\gamma$ | band={band}", fontsize=8)
    if im is not None:
        ps.add_colorbar(fig, im, axes[0][-1],
                        label=r"local $\gamma=\bar K/\mathrm{TE}$")
    fig.suptitle(r"Calibration health ($\gamma \to 1$ = calibrated)", fontsize=11)
    _caption(fig, "Left: per-M slope gamma_M vs the |gamma-1|<=0.2 target band "
                  "(log-y when gamma>>1). Right: local gamma = K-bar/TE per cell "
                  "(TE>0); white = calibrated, red = under-regularised. gamma->1 "
                  "needs beta-selection.")
    ps.save_figure(fig, path)


def _fig_prior_mismatch(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-$M$: calibration intercept $\alpha$ vs the null floors (identity check).

    The decomposition
    $\mathbb{E}[K_t] = I_q(Z;U\mid Y) + \mathbb{E}_Y[\mathrm{KL}(q_\phi(z\mid Y)\|p_\psi(z\mid Y))]$
    means the calibration intercept $\alpha$ *should equal* the irreducible floor:
    the latent KL when no directed source information is present. Two independent
    estimates of that floor are the shuffled-source $\bar K_{\text{shuffle}}$ and
    (if built) the zero-coupling TE$=0$ cell's $\bar K$. A well-specified model
    has, per $M$, $\alpha \approx \bar K_{\text{shuffle}} \approx \bar K(\mathrm{TE}=0)$.
    """
    import matplotlib.pyplot as plt

    if "shuffle" not in controls:
        return
    ms, _, _ = _grid_axes(rows_in)
    if not ms:
        return
    by_M = slices.get("by_M", {}) or {}

    def _mean_key(rows, m, key):
        vals = [float(r[key]) for r in rows
                if r["M"] == m and key in r and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    alpha = [float((by_M.get(str(m)) or {}).get("alpha", np.nan)) for m in ms]
    shuf = [_mean_key(rows_in, m, "null_shuffle_kbar") for m in ms]
    te0 = [float(np.mean([r["kbar_mean"] for r in rows_in
                          if r["M"] == m and float(r.get("target_te", -1.0)) == 0.0])
                 ) if any(r["M"] == m and float(r.get("target_te", -1.0)) == 0.0
                          for r in rows_in) else float("nan") for m in ms]

    x = np.arange(len(ms), dtype=float)
    w = 0.26
    fig, ax = plt.subplots(figsize=(1.6 * len(ms) + 2.5, 4.4))
    ax.bar(x - w, alpha, width=w, color=ps.COLOR_VERMILLION,
           label=r"calibration intercept $\alpha_M$")
    ax.bar(x, shuf, width=w, color=ps.COLOR_BLUE,
           label=r"$\bar K_{\mathrm{shuffle}}$ (null floor)")
    ax.bar(x + w, te0, width=w, color=ps.COLOR_GREEN,
           label=r"$\bar K(\mathrm{TE}{=}0)$ cell")
    ax.axhline(0.0, color=ps.COLOR_BLACK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M={m}" for m in ms])
    ax.set_ylabel(r"latent KL floor (nats)")
    ax.set_title(r"Prior-mismatch check: intercept $\alpha$ vs null floors")
    ax.legend(fontsize=7, frameon=False, loc="best")
    ps.style_axes(ax)
    _caption(fig, "Per M: the calibration intercept should equal the no-transfer "
                  "floor (shuffled source, and the TE=0 cell). Large alpha above "
                  "the floors = leaked directed signal into the intercept.")
    ps.save_figure(fig, path)


def _grid_axes(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[str]]:
    """Return sorted unique (M, target_te, band) axes present in ``rows``."""
    ms = sorted({r["M"] for r in rows})
    tes = sorted({r["target_te"] for r in rows})
    bands = sorted({r["band"] for r in rows},
                   key=lambda b: {"short": 0, "mid": 1, "long": 2}.get(b, 99))
    return ms, tes, bands


# Inverse of the band-name map used by ``_fig_kld_vs_time`` (band_id -> name), so
# the master grid can select per-sample strip points by ``arrs['band_id']``.
_BAND_ID = {"short": 0, "mid": 1, "long": 2}


def _dz_from(arrs: Dict[str, Any]) -> Optional[int]:
    r"""Read the latent dimensionality $d_z$ from the per-dim KL diagnostics.

    The figures need $d_z$ only for the per-dimension $\bar K / d_z$ scale; it is
    recovered from the length of any ``per_dim_kl_by_M`` (or ``per_dim_kl_by_cell``)
    entry so the ``_fig_*`` signature does not have to carry ``config``.

    Args:
        arrs: The per-sample array dict from :func:`collect_per_sample_kbar`.

    Returns:
        The latent dimension $d_z$, or ``None`` if no per-dim diagnostic is
        present (the per-dim panel is then silently skipped).
    """
    for key in ("per_dim_kl_by_M", "per_dim_kl_by_cell"):
        table = arrs.get(key) or {}
        for vec in table.values():
            if vec is not None and len(vec) > 0:
                return int(len(vec))
    return None


def _cells_by_M_band(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    r"""Group per-cell rows by the $(M, \text{band})$ key for the master grid.

    Args:
        rows: Per-cell rows (in-mix and/or held-out).

    Returns:
        ``{(M, band) -> [rows]}``.
    """
    out: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault((int(r["M"]), str(r["band"])), []).append(r)
    return out


def _local_gamma_grid(
    rows: List[Dict[str, Any]],
    ms: Sequence[int],
    tes: Sequence[float],
    band: str,
) -> np.ndarray:
    r"""Build the $M \times \mathrm{TE}$ grid of the local slope $\bar K/\mathrm{TE}$.

    The local slope $\gamma_{\mathrm{loc}} = \bar K / \mathrm{TE}_{\mathrm{true}}$
    is the per-cell calibration deviation: $\gamma_{\mathrm{loc}} \approx 1$ is
    calibrated, $\gg 1$ is under-regularised. Cells with $\mathrm{TE} \le 0$ (the
    null anchor) are left ``nan`` since the ratio is undefined.

    Args:
        rows: Per-cell rows (in-mix and/or held-out).
        ms: Row order (informative-channel counts $M$).
        tes: Column order (target block TE values, $> 0$).
        band: The lag-band to slice.

    Returns:
        A $(\,|ms|, |tes|)$ array of $\bar K / \mathrm{TE}$ (``nan`` where the
        cell is absent or $\mathrm{TE} \le 0$).
    """
    lut = {(r["M"], r["target_te"], r["band"]): r for r in rows}
    grid = np.full((len(ms), len(tes)), np.nan)
    for i, m in enumerate(ms):
        for j, t in enumerate(tes):
            r = lut.get((m, t, band))
            if r is None:
                continue
            te_true = float(r.get("te_true", float("nan")))
            kb = float(r.get("kbar_mean", float("nan")))
            if np.isfinite(te_true) and te_true > 1e-9 and np.isfinite(kb):
                grid[i, j] = kb / te_true
    return grid


def _fig_lag_recovery(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    """Per-cell LOLO LagMass as an M x TE heatmap, faceted by band."""
    import matplotlib.pyplot as plt

    rows = rows_in + rows_ho
    ms, tes, bands = _grid_axes(rows)
    if not (ms and tes and bands):
        return
    lut = {(r["M"], r["target_te"], r["band"]): r for r in rows}
    fig, axes = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 3.4),
                             squeeze=False)
    for ci, band in enumerate(bands):
        ax = axes[0][ci]
        grid = np.full((len(ms), len(tes)), np.nan)
        for i, m in enumerate(ms):
            for j, te in enumerate(tes):
                r = lut.get((m, te, band))
                if r is not None:
                    grid[i, j] = r.get("lag_mass_lolo", np.nan)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma",
                       vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(tes)))
        ax.set_xticklabels([f"{t:g}" for t in tes])
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([str(m) for m in ms])
        ax.set_xlabel(r"target TE (nats)")
        if ci == 0:
            ax.set_ylabel("M")
        ax.set_title(f"LagMass | band={band}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(r"Sliding-window LOLO lag mass in $\mathcal{L}^\star$", fontsize=11)
    _caption(fig, "Leave-one-lag-out importance mass inside the true lag band "
                  "(0-1), as an M x TE heatmap per band. Good: near 1 -- ablating "
                  "in-band source lags hurts the forecast most.")
    ps.save_figure(fig, path)


def _fig_generalization(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    """Dumbbell of held-out vs trained-marginal TE-RMSE and LagMass."""
    import matplotlib.pyplot as plt

    if not rows_ho:
        return
    gaps = _generalization_gaps(rows_in, rows_ho)["cells"]
    if not gaps:
        return
    labels = [f"M{c['M']} TE{c['target_te']:g} {c['band']}" for c in gaps]
    y = np.arange(len(gaps))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.5 * len(gaps) + 2.0),
                             squeeze=False)
    for ax, key, ref, title in (
        (axes[0][0], "te_rmse", "te_rmse_ref", "TE RMSE (held-out vs trained-M mean)"),
        (axes[0][1], "lag_mass_lolo", "lag_mass_ref", "LagMass (held-out vs trained-M mean)"),
    ):
        ho_vals = [c.get(key, np.nan) for c in gaps]
        ref_vals = [c.get(ref, np.nan) for c in gaps]
        for yi, (hv, rv) in enumerate(zip(ho_vals, ref_vals)):
            ax.plot([rv, hv], [yi, yi], color=ps.COLOR_LIGHT_GRAY, lw=2, zorder=1)
        ax.scatter(ref_vals, y, color=ps.COLOR_GRAY, label="trained-M mean", zorder=2)
        ax.scatter(ho_vals, y, color=ps.COLOR_VERMILLION, label="held-out", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ps.style_axes(ax)
    fig.suptitle("Held-out extrapolation gap", fontsize=11)
    _caption(fig, "Held-out cells (vermillion) vs the trained-M-mean reference "
                  "(gray) for TE-RMSE and lag-mass. Good: held-out markers close "
                  "to the reference -- interpolates to unseen (M,TE,band).")
    ps.save_figure(fig, path)


def _fig_null_controls(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Clean vs corrupted-source $\bar K$ per cell -- one panel per control.

    Each panel scatters clean $\bar K$ (x) against the control's $\bar K$ (y),
    coloured by $M$, with a dashed $y=x$ reference. A faithful TE estimate
    **collapses** the control toward $0$, so points should fall **far below**
    $y=x$ (the further below, the cleaner the directional signal). The median
    control/clean ratio is printed per panel.
    """
    import matplotlib.pyplot as plt

    if not controls or not rows_in:
        return
    hi = max((r["kbar_mean"] for r in rows_in), default=1.0) * 1.05 + 1e-6
    fig, axes = plt.subplots(1, len(controls), figsize=(4.2 * len(controls), 4.2),
                             squeeze=False, sharex=True, sharey=True)
    for ci, ctrl in enumerate(controls):
        ax = axes[0][ci]
        ax.plot([0, hi], [0, hi], ls="--", lw=1.0, color=ps.COLOR_GRAY,
                label="$y=x$ (no collapse)", zorder=1)
        ratios: List[float] = []
        for mm in sorted({r["M"] for r in rows_in}):
            xs = [r["kbar_mean"] for r in rows_in if r["M"] == mm]
            ys = [r.get(f"null_{ctrl}_kbar", np.nan) for r in rows_in if r["M"] == mm]
            ax.scatter(xs, ys, s=45, color=_M_COLORS.get(mm, ps.COLOR_PURPLE),
                       edgecolors=ps.COLOR_BLACK, linewidths=0.5, zorder=3,
                       label=f"M={mm}")
        ratios = [r.get(f"null_{ctrl}_ratio", np.nan) for r in rows_in]
        med = float(np.nanmedian(ratios)) if ratios else float("nan")
        ax.set_xlabel(r"clean $\bar K$  (nats)")
        if ci == 0:
            ax.set_ylabel(r"corrupted-source $\bar K$  (nats)")
        ax.set_title(f"{ctrl}  (median ratio = {med:.2f})", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ps.style_axes(ax)
    fig.suptitle("Null controls: source corruption should collapse $\\bar K$",
                 fontsize=11)
    _caption(fig, "Clean K-bar (x) vs source-corrupted K-bar (y) per cell, one "
                  "panel per control; dashed y=x = no collapse. Good: points far "
                  "below y=x (median ratio << 1) -- corruption destroys the signal.")
    ps.save_figure(fig, path)


# =============================================================================
# KLD-vs-TE association: per-sample scatter, recovery, residual, binning
# =============================================================================


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    r"""Estimate the mutual information $I(X; Y)$ in nats between two 1-D arrays.

    Prefers the k-NN estimator :func:`sklearn.feature_selection.mutual_info_regression`
    (output already in nats); falls back to the Gaussian closed form
    $-\tfrac12\ln(1-\rho^2)$ from the Pearson correlation $\rho$ when sklearn is
    unavailable or the estimator fails.

    Args:
        x: Predictor samples $(N,)$.
        y: Target samples $(N,)$.

    Returns:
        The estimated mutual information in nats (``nan`` if degenerate).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    try:
        from sklearn.feature_selection import mutual_info_regression

        mi = mutual_info_regression(
            x.reshape(-1, 1), y, discrete_features=False, random_state=0
        )
        return float(mi[0])
    except Exception:  # noqa: BLE001 -- fall back to the Gaussian approximation
        rho = float(np.corrcoef(x, y)[0, 1])
        rho = min(max(rho, -0.999999), 0.999999)
        return float(-0.5 * np.log(1.0 - rho ** 2))


def _assoc_stats(kbar: np.ndarray, te: np.ndarray) -> Dict[str, float]:
    r"""Compute the KLD-vs-TE association summary (Pearson / Spearman / MI / OLS).

    Args:
        kbar: Per-sample mean latent KL $\bar K$ $(N,)$.
        te: Per-sample true block TE $(N,)$.

    Returns:
        ``{"pearson", "spearman", "mi", "alpha", "gamma", "r2", "n"}`` -- any
        entry is ``nan`` when its estimator is undefined (e.g. a single distinct
        TE value).
    """
    kbar = np.asarray(kbar, dtype=float).ravel()
    te = np.asarray(te, dtype=float).ravel()
    out: Dict[str, float] = {
        "pearson": float("nan"), "spearman": float("nan"), "mi": float("nan"),
        "alpha": float("nan"), "gamma": float("nan"), "r2": float("nan"),
        "n": float(kbar.size),
    }
    if kbar.size < 2:
        return out
    try:
        from scipy.stats import pearsonr, spearmanr

        if np.std(te) > 0 and np.std(kbar) > 0:
            out["pearson"] = float(pearsonr(te, kbar)[0])
            out["spearman"] = float(spearmanr(te, kbar).statistic)
    except Exception:  # noqa: BLE001
        if np.std(te) > 0 and np.std(kbar) > 0:
            out["pearson"] = float(np.corrcoef(te, kbar)[0, 1])
    out["mi"] = _mutual_info(kbar, te)
    try:
        fit = fit_calibration_slope(list(zip(te.tolist(), kbar.tolist())))
        out.update({"alpha": float(fit["alpha"]), "gamma": float(fit["gamma"]),
                    "r2": float(fit["r2"])})
    except ValueError:
        pass
    return out


def kld_te_association(arrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    r"""Headline KLD-vs-TE association over all samples + per-$M$ slices.

    Args:
        arrs: The aligned per-sample arrays from :func:`collect_per_sample_kbar`.

    Returns:
        ``{"overall": stats, "by_M": {M: stats}}`` with :func:`_assoc_stats`
        dicts.
    """
    kbar = np.asarray(arrs["kbar"], dtype=float)
    te = np.asarray(arrs["te_true"], dtype=float)
    m = np.asarray(arrs["M"], dtype=int)
    by_M: Dict[str, Any] = {}
    for mm in sorted(set(m.tolist())):
        sel = m == mm
        by_M[str(int(mm))] = _assoc_stats(kbar[sel], te[sel])
    return {"overall": _assoc_stats(kbar, te), "by_M": by_M}


def _overall_fit(slices: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(alpha, gamma)`` of the overall in-mix calibration fit."""
    overall = (slices or {}).get("overall") or {}
    alpha = float(overall.get("alpha", 0.0))
    gamma = float(overall.get("gamma", 1.0))
    if abs(gamma) <= 1e-12:
        gamma = float("nan")
    return alpha, gamma


def _scatter_cells(ax, rows_in, rows_ho, xkey: str, ykey: str) -> None:
    r"""Scatter one point per cell, colour=$M$, marker=band, filled=in / hollow=held-out.

    Args:
        ax: Target axes.
        rows_in: In-mix per-cell rows.
        rows_ho: Held-out per-cell rows.
        xkey, ykey: Row keys for the x / y coordinate.
    """
    for rows, filled in ((rows_in, True), (rows_ho, False)):
        for r in rows:
            x, y = r.get(xkey, np.nan), r.get(ykey, np.nan)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            color = _M_COLORS.get(r["M"], ps.COLOR_PURPLE)
            ax.scatter(
                x, y, c=color if filled else "none", edgecolors=color,
                marker=_BAND_MARKERS.get(r["band"], "D"), s=55, linewidths=1.4,
                zorder=3,
            )


def _cell_legend_handles(rows: List[Dict[str, Any]]):
    """Build the shared M-colour / band-marker legend handles."""
    import matplotlib.pyplot as plt

    ms = sorted({r["M"] for r in rows})
    bands = sorted({r["band"] for r in rows},
                   key=lambda b: {"short": 0, "mid": 1, "long": 2}.get(b, 99))
    h_m = [plt.Line2D([0], [0], marker="o", ls="",
                      color=_M_COLORS.get(m, ps.COLOR_PURPLE), label=f"M={m}")
           for m in ms]
    h_b = [plt.Line2D([0], [0], marker=_BAND_MARKERS.get(b, "D"), ls="",
                      color=ps.COLOR_GRAY, label=f"band={b}") for b in bands]
    return h_m + h_b


def _fig_te_recovery_percell(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-cell recovered TE $(\bar K-\alpha)/\gamma$ vs true TE, with $y=x$.

    Recovers each cell's TE from the overall in-mix calibration and plots it
    against the cell's true block TE. RMSE / bias are over cells.
    """
    import matplotlib.pyplot as plt

    all_rows = rows_in + rows_ho
    if not all_rows:
        return
    alpha, gamma = _overall_fit(slices)
    for r in all_rows:
        r["_te_pred"] = (r["kbar_mean"] - alpha) / gamma
    te_in = np.asarray([r["te_true"] for r in rows_in], dtype=float)
    pred_in = np.asarray([r["_te_pred"] for r in rows_in], dtype=float)
    rmse = float(np.sqrt(np.nanmean((pred_in - te_in) ** 2))) if te_in.size else float("nan")
    bias = float(np.nanmean(pred_in - te_in)) if te_in.size else float("nan")

    preds = [r["_te_pred"] for r in all_rows if np.isfinite(r["_te_pred"])]
    hi = max([r["te_true"] for r in all_rows] + preds, default=1.0) * 1.05 + 1e-6
    lo = min([0.0] + preds) - 0.02 * abs(hi)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=ps.COLOR_GRAY, label="$y=x$")
    _scatter_cells(ax, rows_in, rows_ho, "te_true", "_te_pred")
    # Both axes are in TE-nats, so equalise their ranges to keep y=x an exact
    # diagonal (independent autoscaling would tilt the reference line).
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"recovered TE  $(\bar K-\alpha)/\gamma$  (nats)")
    ax.set_title("Per-cell TE recovery (overall in-mix calibration)")
    _caption(fig, "Each cell's recovered TE (K-bar - alpha)/gamma vs its true TE; "
                  "dashed y=x. Recovery divides out the overall gamma, so it works "
                  "even when gamma>>1 (under-regularised); residual scatter is the "
                  "per-cell calibration error. Good: points on y=x, low RMSE, "
                  "near-zero bias.")
    ax.text(0.03, 0.97,
            f"RMSE = {rmse:.3f} nats\nbias = {bias:+.3f} nats\n"
            f"$\\gamma$ = {gamma:.2f}, $\\alpha$ = {alpha:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec=ps.COLOR_GRAY, alpha=0.85))
    ax.legend(handles=_cell_legend_handles(all_rows), fontsize=7,
              loc="lower right", frameon=False, ncol=2)
    ps.style_axes(ax)
    ps.save_figure(fig, path)
    for r in all_rows:
        r.pop("_te_pred", None)


# =============================================================================
# Prediction-gain (Delta-L) figures
# =============================================================================


def _fig_pred_gain_vs_te(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-cell prediction gain $\Delta\mathcal L$ vs true TE (validation plan §9.2).

    $\Delta\mathcal L = \mathcal L_{\mathrm{base}} - \mathcal L_{\mathrm{feat}} > 0$
    only when the source path improves the forecast. Expect $\Delta\mathcal L
    \approx 0$ at the lowest TE and rising with TE.
    """
    import matplotlib.pyplot as plt

    pg_in = arrs.get("pred_gain_by_cell", {}) or {}
    if not pg_in:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.axhline(0.0, ls="--", lw=1.0, color=ps.COLOR_GRAY, zorder=1,
               label=r"$\Delta\mathcal{L}=0$")
    for v in pg_in.values():
        color = _M_COLORS.get(v["M"], ps.COLOR_PURPLE)
        ax.scatter(v["te_true"], v["delta_L"], c=color, edgecolors=ps.COLOR_BLACK,
                   marker=_BAND_MARKERS.get(v["band"], "D"), s=55,
                   linewidths=0.5, zorder=3)
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"prediction gain  $\Delta\mathcal{L}=\mathcal{L}_{\mathrm{base}}"
                  r"-\mathcal{L}_{\mathrm{feat}}$  (nats)")
    ax.set_title("Prediction gain vs true TE")
    handles = _cell_legend_handles(
        [{"M": v["M"], "band": v["band"]} for v in pg_in.values()]
    )
    ax.legend(handles=handles, fontsize=7, loc="upper left", frameon=False, ncol=2)
    _caption(fig, "Prediction gain dL = L_base - L_feat per cell vs true TE; "
                  "dashed dL=0. Good: dL ~ 0 at low TE and rising with TE -- the "
                  "source path helps the forecast only when info is real.")
    ps.style_axes(ax)
    ps.save_figure(fig, path)


def _fig_pred_gain_vs_kbar(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-cell $(\bar K,\ \Delta\mathcal L)$ plane, coloured by true TE.

    The validation-plan §10.1 view of *whether latent KL buys predictive
    improvement*: points above the dashed $\Delta\mathcal L=0$ line mean the
    source path helped the forecast.
    """
    import matplotlib.pyplot as plt

    pg = arrs.get("pred_gain_by_cell", {}) or {}
    kbar_by_cell = {int(r["cell_id"]): float(r["kbar_mean"]) for r in rows_in}
    pts: List[Tuple[float, float, float]] = []
    for cid, v in pg.items():
        x = kbar_by_cell.get(int(cid))
        y = float(v["delta_L"])
        if x is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts.append((float(x), y, float(v["te_true"])))
    if not pts:
        return
    xs, ys, ts = zip(*pts)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.axhline(0.0, ls="--", lw=1.0, color=ps.COLOR_GRAY, zorder=1)
    sc = ax.scatter(xs, ys, c=ts, cmap="viridis", s=60, edgecolors=ps.COLOR_BLACK,
                    linewidths=0.5, zorder=3)
    ps.add_colorbar(fig, sc, ax, label=r"true block TE (nats)")
    ax.set_xlabel(r"mean latent KL  $\bar K$  (nats)")
    ax.set_ylabel(r"prediction gain  $\Delta\mathcal{L}$  (nats)")
    ax.set_title(r"Does latent KL buy prediction? $(\bar K,\ \Delta\mathcal{L})$ plane")
    _caption(fig, "Per-cell (K-bar, dL) plane coloured by true TE. Above dL=0 = "
                  "latent KL bought real predictive gain; an upward trend means "
                  "more KL converts to more forecast improvement.")
    ps.style_axes(ax)
    ps.save_figure(fig, path)


# =============================================================================
# Lag-profile figures (LOLO + attention)
# =============================================================================


def _shade_band(ax, dmax: int) -> None:
    r"""Shade the true lag band $\{0,\dots,d_{\max}-1\}$ on the source-lag axis."""
    if dmax and dmax > 0:
        ax.axvspan(-0.5, dmax - 0.5, color=ps.COLOR_VERMILLION, alpha=0.16,
                   lw=0.0, label=r"true band $\mathcal{L}^\star$")


def _fig_lag_profiles(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-cell LOLO importance $A_\ell$ vs lag, faceted by band, true band shaded.

    Overlays every cell's $A_\ell$ profile (from the sliding-window
    leave-one-lag-out ablation) within its band panel; a faithful model peaks
    inside the shaded true band $\{0,\dots,d_{\max}-1\}$.
    """
    import matplotlib.pyplot as plt

    prof = arrs.get("lag_profile_by_cell", {}) or {}
    if not prof:
        return
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    for v in prof.values():
        by_band.setdefault(v.get("band", ""), []).append(v)
    bands = sorted(by_band, key=lambda b: {"short": 0, "mid": 1, "long": 2}.get(b, 99))
    fig, axes = plt.subplots(1, len(bands), figsize=(4.0 * len(bands), 3.8),
                             squeeze=False, sharey=True)
    for ci, band in enumerate(bands):
        ax = axes[0][ci]
        dmax = max((v.get("delay_max", 0) for v in by_band[band]), default=0)
        _shade_band(ax, dmax)
        for v in by_band[band]:
            grid = np.asarray(v.get("lag_grid", []), dtype=float)
            a = np.asarray(v.get("A_lag", []), dtype=float)
            if grid.size == 0 or a.size == 0:
                continue
            ax.plot(grid, a[grid.astype(int)] if a.size > grid.max() else a[:grid.size],
                    color=ps.COLOR_BLUE, lw=1.0, alpha=0.5)
        ax.set_xlabel(r"source lag $\ell$")
        if ci == 0:
            ax.set_ylabel(r"LOLO importance $A_\ell$")
        ax.set_title(f"band = {band}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", frameon=False)
        ps.style_axes(ax)
    fig.suptitle(r"Per-cell sliding-window LOLO lag profile $A_\ell$", fontsize=11)
    _caption(fig, "Per-cell leave-one-lag-out importance A_l vs source lag, one "
                  "panel per band; shaded = true band {0..dmax-1}. Good: every "
                  "curve peaks inside the shaded band.")
    ps.save_figure(fig, path)


def _fig_lag_walk(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-step lag-walk recovery: true $d_t$ vs recovered $\hat d_t$ over time.

    Uses the per-cell mean trajectories from
    :func:`collect_per_cell_attn_diag` (the heterogeneous random-walk-lag DGP's
    per-step ground truth ``true_lag_tt``). A faithful model tracks the true lag
    walk $d_t$ (solid) with its argmax-attention recovery
    $\hat d_t = \arg\max_\ell \bar\alpha_{t,\ell}$ (dashed); the band-mean MAE
    annotates each panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    walk = arrs.get("lag_walk_by_cell", {}) or {}
    if not walk:
        return
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    for v in walk.values():
        by_band.setdefault(str(v.get("band", "")), []).append(v)
    bands = sorted(by_band, key=lambda b: {"short": 0, "mid": 1, "long": 2}.get(b, 99))
    fig, axes = plt.subplots(1, len(bands), figsize=(4.2 * len(bands), 3.8),
                             squeeze=False, sharey=True)
    for ci, band in enumerate(bands):
        ax = axes[0][ci]
        maes: List[float] = []
        for v in by_band[band]:
            true_t = np.asarray(v.get("true_mean", []), dtype=float)
            pred_t = np.asarray(v.get("pred_mean", []), dtype=float)
            if true_t.size == 0 or pred_t.size == 0:
                continue
            t_axis = np.arange(true_t.size)
            ax.plot(t_axis, true_t, color=ps.COLOR_BLUE, lw=1.1, alpha=0.7)
            ax.plot(t_axis, pred_t, color=ps.COLOR_ORANGE, lw=1.0, ls="--",
                    alpha=0.7)
            maes.append(float(v.get("mae", float("nan"))))
        mae_arr = np.asarray(maes, dtype=float)
        mae_arr = mae_arr[np.isfinite(mae_arr)]
        mae_txt = f"MAE={mae_arr.mean():.2f}" if mae_arr.size else "MAE=n/a"
        ax.set_xlabel(r"time step $t$")
        if ci == 0:
            ax.set_ylabel(r"source lag $d_t$")
        ax.set_title(f"band = {band}  ({mae_txt})", fontsize=9)
        ps.style_axes(ax)
    handles = [
        Line2D([0], [0], color=ps.COLOR_BLUE, lw=1.1, label=r"true $d_t$"),
        Line2D([0], [0], color=ps.COLOR_ORANGE, lw=1.0, ls="--",
               label=r"recovered $\hat d_t$"),
    ]
    fig.legend(handles=handles, fontsize=8, loc="upper right", frameon=False)
    fig.suptitle(r"Per-step lag-walk recovery: true vs argmax-attention $d_t$",
                 fontsize=11)
    _caption(fig, "Mean true lag walk d_t (blue) vs recovered argmax-attention "
                  "d_hat_t (orange dashed) over time, one panel per band. Good: "
                  "dashed tracks solid; low MAE.")
    ps.save_figure(fig, path)


def _fig_attn_vs_lag(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Band-mean attention-vs-lag with the LOLO $A_\ell$ overlaid (twin axis).

    Per band: the head-averaged, clean-window attention profile
    $\bar\alpha_\ell$ (orange, normalised to unit lag-sum) and the mean LOLO
    $A_\ell$ (blue), both with the true band $\{0,\dots,d_{\max}-1\}$ shaded.
    Agreement of the two on the shaded band is the strong lag-recovery signal.
    """
    import matplotlib.pyplot as plt

    attn = arrs.get("attn_profile_by_cell", {}) or {}
    prof = arrs.get("lag_profile_by_cell", {}) or {}
    if not attn:
        return
    attn_by_band: Dict[str, List[np.ndarray]] = {}
    dmax_by_band: Dict[str, int] = {}
    for v in attn.values():
        b = v.get("band", "")
        attn_by_band.setdefault(b, []).append(np.asarray(v["attn_lag"], dtype=float))
        dmax_by_band[b] = max(dmax_by_band.get(b, 0), int(v.get("delay_max", 0)))
    lolo_by_band: Dict[str, List[np.ndarray]] = {}
    for v in prof.values():
        a = np.asarray(v.get("A_lag", []), dtype=float)
        if a.size:
            lolo_by_band.setdefault(v.get("band", ""), []).append(a)

    bands = sorted(attn_by_band, key=lambda b: {"short": 0, "mid": 1, "long": 2}.get(b, 99))
    fig, axes = plt.subplots(1, len(bands), figsize=(4.2 * len(bands), 3.8),
                             squeeze=False)
    for ci, band in enumerate(bands):
        ax = axes[0][ci]
        a_mean = np.mean(np.vstack(attn_by_band[band]), axis=0)
        a_sum = float(a_mean.sum())
        a_norm = a_mean / a_sum if a_sum > 1e-12 else a_mean
        lag_axis = np.arange(a_norm.size)
        _shade_band(ax, dmax_by_band.get(band, 0))
        ln1 = ax.plot(lag_axis, a_norm, color=ps.COLOR_ORANGE, lw=1.2,
                      label=r"$\bar\alpha_\ell$ (attention)")
        ax.set_xlabel(r"source lag $\ell$")
        ax.set_ylabel(r"$\bar\alpha_\ell$ (norm.)", color=ps.COLOR_ORANGE)
        ax.tick_params(axis="y", colors=ps.COLOR_ORANGE)
        ax.set_xlim(-0.5, a_norm.size - 0.5)
        ax.set_title(f"band = {band}", fontsize=9)
        ln2 = []
        if band in lolo_by_band:
            ax2 = ax.twinx()
            lo_mean = np.mean(np.vstack(lolo_by_band[band]), axis=0)
            ln2 = ax2.plot(np.arange(lo_mean.size), lo_mean, color=ps.COLOR_BLUE,
                           lw=1.2, label=r"$A_\ell$ (LOLO)")
            ax2.set_ylabel(r"$A_\ell$ (LOLO)", color=ps.COLOR_BLUE)
            ax2.tick_params(axis="y", colors=ps.COLOR_BLUE)
        ax.legend(handles=ln1 + ln2, fontsize=7, loc="upper right", frameon=False)
        ps.style_axes(ax)
    fig.suptitle("Attention vs LOLO lag profile (band-averaged)", fontsize=11)
    _caption(fig, "Band-averaged lag attention (orange) and LOLO importance "
                  "(blue) vs source lag; shaded = true band. Good: both "
                  "concentrate on and agree over the shaded band.")
    ps.save_figure(fig, path)


# =============================================================================
# KLD-structure figures (per-dimension, vs time)
# =============================================================================


def _fig_per_dim_kl(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-latent-dimension KL as an $M \times d_z$ heatmap + overall-mean bar.

    Reveals which of the $d_z$ latent dimensions carry source information and
    flags posterior collapse (a dimension dark across every $M$).
    """
    import matplotlib.pyplot as plt

    by_M = arrs.get("per_dim_kl_by_M", {}) or {}
    if not by_M:
        return
    ms = sorted(int(m) for m in by_M)
    grid = np.array([by_M[str(m)] if str(m) in by_M else by_M[m] for m in ms],
                    dtype=float)
    d_z = grid.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(max(6.0, 0.28 * d_z + 2.0), 5.4),
                             gridspec_kw={"height_ratios": [len(ms), 1.4]})
    im = axes[0].imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_yticks(range(len(ms)))
    axes[0].set_yticklabels([str(m) for m in ms])
    axes[0].set_ylabel("M")
    axes[0].set_xlabel(r"latent dimension index")
    axes[0].set_title(r"Per-dimension mean KL  $\overline{\mathrm{KL}}_d$  (clean window)")
    ps.add_colorbar(fig, im, axes[0], label="nats")
    axes[1].bar(np.arange(d_z), grid.mean(axis=0), color=ps.COLOR_BLUE)
    axes[1].set_xlabel(r"latent dimension index")
    axes[1].set_ylabel("mean over M")
    axes[1].set_xlim(-0.5, d_z - 0.5)
    ps.style_axes(axes[1])
    fig.suptitle("Latent KL allocation across dimensions", fontsize=11)
    _caption(fig, "Mean KL per latent dimension: per-M heatmap (top), "
                  "averaged-over-M bar (bottom). A dimension dark across all M is "
                  "collapsed/unused; broad use = healthy latent.")
    ps.save_figure(fig, path)


def _fig_kld_vs_time(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Mean ``kld_per_t`` trajectory vs timestep $t$, one line per lag-band.

    Vertical dashed lines mark the warm-up floor and each band's clean-window
    floor $\max(\text{warmup}, d_{\max}-1)$, so the steady-state KL away from
    the transient is visible.
    """
    import matplotlib.pyplot as plt

    by_band = arrs.get("kld_time_by_band", {}) or {}
    if not by_band:
        return
    band_names = {0: "short", 1: "mid", 2: "long"}
    palette = ps.PALETTE_PRIMARY
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    warmup_drawn = False
    for i, (b, v) in enumerate(sorted(by_band.items())):
        kt = np.asarray(v["kld_t"], dtype=float)
        t = np.arange(kt.size)
        color = palette[i % len(palette)]
        label = band_names.get(int(b), f"band {b}")
        ax.plot(t, kt, color=color, lw=1.3, label=label)
        floor = max(int(v.get("warmup", 0)), int(v.get("delay_max", 0)) - 1)
        ax.axvline(floor, ls=":", lw=0.9, color=color, alpha=0.8)
        if not warmup_drawn:
            ax.axvline(int(v.get("warmup", 0)), ls="--", lw=1.0,
                       color=ps.COLOR_GRAY, label="warm-up")
            warmup_drawn = True
    ps.tighten_xaxis(ax, np.arange(max(len(v["kld_t"]) for v in by_band.values())))
    ax.set_xlabel(r"timestep $t$")
    ax.set_ylabel(r"mean per-step KL  $K_t$  (nats)")
    ax.set_title("KLD trajectory over time by lag-band")
    _caption(fig, "Mean per-step KL K_t vs time, one line per band; dashed = "
                  "warm-up, dotted = each band's clean-window floor. Good: a "
                  "decaying transient then a stable steady state past the floor.")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ps.style_axes(ax)
    ps.save_figure(fig, path)


# =============================================================================
# CLI / dual-mode
# =============================================================================

def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply path overrides in place (``data_dir`` / ``results_dir``)."""
    apply_path_overrides(config, overrides)


def main() -> None:
    """CLI entry point: parse arguments, load config, evaluate."""
    parser = argparse.ArgumentParser(
        description="Per-group recovery + generalization for the G1_mix model."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--run-tag", type=str, required=True, dest="run_tag",
                        help="training run under results/G1_mix/<run_tag>/")
    parser.add_argument("--in-mix-tag", type=str, default="G1_mix_base",
                        dest="in_mix_tag", help="in-mix cache tag")
    parser.add_argument("--holdout-tag", type=str, default=None,
                        dest="holdout_tag", help="held-out cache tag")
    parser.add_argument("--ckpt-name", type=str, default="final.ckpt",
                        dest="ckpt_name")
    parser.add_argument("--out-subdir", type=str, default=None,
                        dest="out_subdir",
                        help="output subdir under the run dir (default "
                             "'mixed_eval'); use a distinct name per "
                             "extrapolation cache to avoid overwriting")
    parser.add_argument("--data-dir", type=str, default=None, dest="data_dir")
    parser.add_argument("--results-dir", type=str, default=None, dest="results_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    config["experiment"]["benchmark"] = _BENCHMARK
    resolve_active_benchmark(config)
    _apply_overrides(config, vars(args))
    evaluate_mixed(
        config, run_tag=args.run_tag, in_mix_tag=args.in_mix_tag,
        holdout_tag=args.holdout_tag, ckpt_name=args.ckpt_name,
        out_subdir=args.out_subdir,
    )


if __name__ == "__main__":
    CONFIG_PATH = _DEFAULT_CONFIG
    RUN_CONFIG = {
        "run_tag": "G1_mix_gnll",
        "in_mix_tag": "G1_mix_base",
        "holdout_tag": "G1_mix_base_holdout",
        "ckpt_name": "final.ckpt",
        "out_subdir": None,      # None -> 'mixed_eval'; set per extrap cache
        "data_dir": None,
        "results_dir": None,
    }

    if len(sys.argv) > 1:
        main()
    else:
        config = load_config(CONFIG_PATH)
        config["experiment"]["benchmark"] = _BENCHMARK
        resolve_active_benchmark(config)
        _apply_overrides(config, RUN_CONFIG)
        evaluate_mixed(
            config, run_tag=RUN_CONFIG["run_tag"],
            in_mix_tag=RUN_CONFIG["in_mix_tag"],
            holdout_tag=RUN_CONFIG["holdout_tag"],
            ckpt_name=RUN_CONFIG["ckpt_name"],
            out_subdir=RUN_CONFIG["out_subdir"],
        )
