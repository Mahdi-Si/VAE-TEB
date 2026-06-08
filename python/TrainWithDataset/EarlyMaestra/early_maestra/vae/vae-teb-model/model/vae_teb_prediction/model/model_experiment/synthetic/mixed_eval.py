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
        ``kbar_shuffle`` / ``kbar_reverse`` when requested.
    """
    out: Dict[str, List[np.ndarray]] = {
        k: [] for k in (
            "kbar", "te_true", "M", "delay_max", "band_id", "cell_id", "held_out",
        )
    }
    for ctrl in controls:
        out[f"kbar_{ctrl}"] = []

    for batch in loader:
        batch = move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        delay_max = _batch_int(batch, "delay_max", device)
        kbar = _per_sample_kbar(
            model, y_st, y_ph, build_u_stream(batch), delay_max,
            warmup=warmup, horizon=horizon,
        )
        out["kbar"].append(kbar)
        for ctrl in controls:
            corrupt = (
                shuffle_source_batch(batch) if ctrl == "shuffle"
                else reverse_source_batch(batch)
            )
            out[f"kbar_{ctrl}"].append(_per_sample_kbar(
                model, y_st, y_ph, build_u_stream(corrupt), delay_max,
                warmup=warmup, horizon=horizon,
            ))
        out["te_true"].append(_batch_float(batch, "te_true"))
        out["M"].append(_batch_int(batch, "M", device).cpu().numpy())
        out["delay_max"].append(delay_max.cpu().numpy())
        out["band_id"].append(_batch_int(batch, "band_id", device).cpu().numpy())
        out["cell_id"].append(_batch_int(batch, "cell_id", device).cpu().numpy())
        out["held_out"].append(_batch_int(batch, "held_out", device).cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in out.items() if v}


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
def _per_sample_kbar(
    model: Any,
    y_st: torch.Tensor,
    y_ph: torch.Tensor,
    u_stream: torch.Tensor,
    delay_max: torch.Tensor,
    *,
    warmup: int,
    horizon: int,
) -> np.ndarray:
    r"""Per-sample $\bar K$ over each sample's own clean window.

    Args:
        model: The model (``encode_only`` + ``kld_tensor`` are used so no
            decoder pass is needed).
        y_st, y_ph: Target streams $(B, T, \cdot)$.
        u_stream: Source stream $(B, T, c_u)$.
        delay_max: Per-sample lag ceiling $(B,)$ long tensor.
        warmup: Warm-up floor.
        horizon: Forecast horizon $H$.

    Returns:
        A length-$B$ ``float64`` numpy array of per-sample $\bar K_i$.
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
    return kbar.detach().cpu().to(torch.float64).numpy()


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


def fit_calibration_slices(
    arrs: Dict[str, np.ndarray],
    cells_by_id: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    r"""Fit $\bar K = \alpha + \gamma\,\mathrm{TE}$ overall + per-$M$ + per-band.

    The headline slope uses **per-cell means** (one point per cell); a singular
    or under-determined slice (fewer than two distinct cell TEs) is skipped.

    Args:
        arrs: The aligned per-sample arrays.
        cells_by_id: Manifest cell dicts keyed by ``cell_id``.

    Returns:
        ``{"overall": fit, "by_M": {M: fit}, "by_band": {band: fit}}`` where
        each ``fit`` is the :func:`fit_calibration_slope` dict, or ``None`` if
        the slice was singular.
    """
    cell_ids = np.asarray(arrs["cell_id"], dtype=int)
    kbar = np.asarray(arrs["kbar"], dtype=float)
    te = np.asarray(arrs["te_true"], dtype=float)
    per_cell: List[Tuple[float, float, int, str]] = []
    for cid in sorted(set(cell_ids.tolist())):
        sel = cell_ids == cid
        cell = cells_by_id.get(int(cid), {})
        per_cell.append((
            float(np.mean(te[sel])), float(np.mean(kbar[sel])),
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


def evaluate_mixed(
    config: Dict[str, Any],
    *,
    run_tag: str,
    in_mix_tag: str,
    holdout_tag: Optional[str] = None,
    ckpt_name: str = "final.ckpt",
) -> Dict[str, Any]:
    r"""Evaluate the ``G1_mix`` model per-group on in-mix + held-out caches.

    Args:
        config: The parsed config carrying ``benchmarks.G1_mix``.
        run_tag: Training run under ``results/G1_mix/<run_tag>/``.
        in_mix_tag: In-mix cache tag (``data/G1_mix/<in_mix_tag>/``).
        holdout_tag: Optional held-out cache tag for extrapolation.
        ckpt_name: Checkpoint file name (``final.ckpt`` or ``best.ckpt``).

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

    out_dir = run_dir / _OUT_SUBDIR
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

    # --- held-out extrapolation ---------------------------------------------
    rows_ho: List[Dict[str, Any]] = []
    slices_ho: Dict[str, Any] = {}
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
        except FileNotFoundError as exc:
            print(f"[mixed_eval] held-out cache skipped: {exc}")

    generalization = _generalization_gaps(rows_in, rows_ho)

    metrics = {
        "run_tag": run_tag,
        "ckpt": str(ckpt_path),
        "warmup": warmup,
        "horizon": horizon,
        "calibration": {"in_mix": slices_in, "holdout": slices_ho},
        "kld_te_association": {
            "in_mix": kld_te_association(arrs_in),
            "holdout": kld_te_association(arrs_ho) if rows_ho else {},
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
        json.dump({"in_mix": slices_in, "holdout": slices_ho,
                   "alpha": float(overall["alpha"]),
                   "gamma": float(overall["gamma"])}, fh, indent=2)
    with open(out_dir / "generalization.json", "w", encoding="utf-8") as fh:
        json.dump(generalization, fh, indent=2)

    _render_figures(out_dir, rows_in, rows_ho, arrs_in, slices_in, controls)

    g = overall.get("gamma", float("nan"))
    a = overall.get("alpha", float("nan"))
    print(
        f"[mixed_eval] done -> {out_dir}\n"
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
    "te_bias", "lag_mass_lolo", "peak_lag_err", "held_out",
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


_M_COLORS = {8: ps.COLOR_BLUE, 16: ps.COLOR_VERMILLION, 32: ps.COLOR_GREEN}
_BAND_MARKERS = {"short": "o", "mid": "s", "long": "^"}


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
        ("calibration_scatter", _fig_calibration),
        ("grid_heatmaps", _fig_grid_heatmaps),
        ("lag_band_recovery", _fig_lag_recovery),
        ("generalization_gap", _fig_generalization),
        ("null_controls", _fig_null_controls),
        ("kld_vs_te_scatter", _fig_kld_vs_te_scatter),
        ("te_recovery_scatter", _fig_te_recovery_scatter),
        ("te_residual_vs_te", _fig_te_residual_vs_te),
        ("kld_vs_te_binned", _fig_kld_vs_te_binned),
        ("corr_by_M_bars", _fig_corr_by_m_bars),
    ):
        try:
            fn(out_dir / name, rows_in, rows_ho, arrs, slices, controls)
        except Exception as exc:  # noqa: BLE001 -- a plot must never gate eval
            print(f"[mixed_eval] figure '{name}' skipped: {exc}")


def _fig_calibration(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    """K-bar vs TE_true scatter, colour=M, marker=band; y=x + per-M OLS lines."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    all_rows = rows_in + rows_ho
    te_vals = [r["te_true"] for r in all_rows] or [0.0, 1.0]
    kb_vals = [r["kbar_mean"] for r in all_rows] or [0.0, 1.0]
    hi = max(max(te_vals), max(kb_vals)) * 1.05 + 1e-6
    ax.plot([0, hi], [0, hi], ls="--", lw=1.0, color=ps.COLOR_GRAY, label="$y=x$")
    for rows, filled in ((rows_in, True), (rows_ho, False)):
        for r in rows:
            color = _M_COLORS.get(r["M"], ps.COLOR_PURPLE)
            marker = _BAND_MARKERS.get(r["band"], "D")
            ax.scatter(
                r["te_true"], r["kbar_mean"], c=color if filled else "none",
                edgecolors=color, marker=marker, s=55, linewidths=1.4,
                zorder=3,
            )
    by_M = slices.get("by_M", {}) or {}
    te_line = np.linspace(0, hi, 50)
    for m_str, fit in by_M.items():
        if not fit:
            continue
        color = _M_COLORS.get(int(m_str), ps.COLOR_PURPLE)
        ax.plot(te_line, fit["alpha"] + fit["gamma"] * te_line, color=color,
                lw=1.3, alpha=0.8, label=f"M={m_str} ($\\gamma$={fit['gamma']:.2f})")
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"mean latent KL  $\bar K$  (nats)")
    overall = slices.get("overall")
    title = "Mixed-population calibration"
    if overall:
        title += (f"  ($\\gamma$={overall['gamma']:.2f}, "
                  f"$\\alpha$={overall['alpha']:.2f})")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ps.style_axes(ax)
    ps.save_figure(fig, path)


def _grid_axes(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[str]]:
    """Return sorted unique (M, target_te, band) axes present in ``rows``."""
    ms = sorted({r["M"] for r in rows})
    tes = sorted({r["target_te"] for r in rows})
    bands = sorted({r["band"] for r in rows},
                   key=lambda b: {"short": 0, "mid": 1, "long": 2}.get(b, 99))
    return ms, tes, bands


def _fig_grid_heatmaps(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    """Per-cell K-bar and TE-recovery error as M x TE heatmaps, faceted by band."""
    import matplotlib.pyplot as plt

    rows = rows_in + rows_ho
    ms, tes, bands = _grid_axes(rows)
    if not (ms and tes and bands):
        return
    lut = {(r["M"], r["target_te"], r["band"]): r for r in rows}
    nrows, ncols = 2, len(bands)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 6.0),
                             squeeze=False)
    metrics = [("kbar_mean", r"$\bar K$"), ("te_rmse", "TE RMSE")]
    for ri, (key, label) in enumerate(metrics):
        for ci, band in enumerate(bands):
            ax = axes[ri][ci]
            grid = np.full((len(ms), len(tes)), np.nan)
            for i, m in enumerate(ms):
                for j, te in enumerate(tes):
                    r = lut.get((m, te, band))
                    if r is not None:
                        grid[i, j] = r.get(key, np.nan)
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(tes)))
            ax.set_xticklabels([f"{t:g}" for t in tes])
            ax.set_yticks(range(len(ms)))
            ax.set_yticklabels([str(m) for m in ms])
            if ri == nrows - 1:
                ax.set_xlabel(r"target TE (nats)")
            if ci == 0:
                ax.set_ylabel(f"{label}\nM")
            ax.set_title(f"{label} | band={band}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-cell recovery over the (M x TE x band) grid", fontsize=11)
    ps.save_figure(fig, path)


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
    ps.save_figure(fig, path)


def _fig_null_controls(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    """Per-cell K-bar: clean vs each source-corruption control."""
    import matplotlib.pyplot as plt

    if not controls:
        return
    rows = sorted(rows_in, key=lambda r: (r["M"], r["target_te"], r["band"]))
    if not rows:
        return
    x = np.arange(len(rows))
    width = 0.8 / (1 + len(controls))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.32 * len(rows)), 4.0))
    ax.bar(x, [r["kbar_mean"] for r in rows], width, label="clean",
           color=ps.COLOR_BLUE)
    palette = [ps.COLOR_VERMILLION, ps.COLOR_GREEN, ps.COLOR_ORANGE]
    for k, ctrl in enumerate(controls):
        ax.bar(x + (k + 1) * width, [r.get(f"null_{ctrl}_kbar", np.nan) for r in rows],
               width, label=ctrl, color=palette[k % len(palette)])
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"M{r['M']}/{r['target_te']:g}/{r['band']}" for r in rows],
                       rotation=90, fontsize=6)
    ax.set_ylabel(r"mean latent KL  $\bar K$")
    ax.set_title("Null controls: source shuffle / reverse collapse")
    ax.legend(fontsize=8, frameon=False)
    ps.style_axes(ax)
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


def _scatter_by_m(ax, te: np.ndarray, kbar: np.ndarray, m: np.ndarray) -> None:
    """Scatter ``(te, kbar)`` coloured by ``M`` onto ``ax`` (one label per M)."""
    for mm in sorted(set(np.asarray(m, dtype=int).tolist())):
        sel = np.asarray(m, dtype=int) == mm
        ax.scatter(
            te[sel], kbar[sel], s=10, alpha=0.35, linewidths=0.0,
            color=_M_COLORS.get(int(mm), ps.COLOR_PURPLE), label=f"M={int(mm)}",
            zorder=2,
        )


def _fig_kld_vs_te_scatter(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-sample $\bar K$ vs true TE scatter with OLS line + correlation / MI."""
    import matplotlib.pyplot as plt

    te = np.asarray(arrs["te_true"], dtype=float)
    kbar = np.asarray(arrs["kbar"], dtype=float)
    m = np.asarray(arrs["M"], dtype=int)
    if te.size == 0:
        return
    st = _assoc_stats(kbar, te)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _scatter_by_m(ax, te, kbar, m)
    if np.isfinite(st["gamma"]):
        xs = np.linspace(float(te.min()), float(te.max()), 50)
        ax.plot(xs, st["alpha"] + st["gamma"] * xs, color=ps.COLOR_BLACK, lw=1.6,
                zorder=4,
                label=(rf"OLS: $\bar K$={st['gamma']:.2f}$\,$TE+{st['alpha']:.2f}"))
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"mean latent KL  $\bar K$  (nats)")
    ax.set_title("Per-sample KLD vs true TE")
    txt = (
        f"N = {int(st['n'])}\n"
        f"Pearson r = {st['pearson']:.3f}\n"
        f"Spearman $\\rho$ = {st['spearman']:.3f}\n"
        f"MI = {st['mi']:.3f} nats\n"
        f"$R^2$ = {st['r2']:.3f}"
    )
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec=ps.COLOR_GRAY,
                                  alpha=0.85))
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ps.style_axes(ax)
    ps.save_figure(fig, path)


def _overall_fit(slices: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(alpha, gamma)`` of the overall in-mix calibration fit."""
    overall = (slices or {}).get("overall") or {}
    alpha = float(overall.get("alpha", 0.0))
    gamma = float(overall.get("gamma", 1.0))
    if abs(gamma) <= 1e-12:
        gamma = float("nan")
    return alpha, gamma


def _fig_te_recovery_scatter(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-sample recovered TE $(\bar K-\alpha)/\gamma$ vs true TE, with $y=x$."""
    import matplotlib.pyplot as plt

    te = np.asarray(arrs["te_true"], dtype=float)
    kbar = np.asarray(arrs["kbar"], dtype=float)
    m = np.asarray(arrs["M"], dtype=int)
    if te.size == 0:
        return
    alpha, gamma = _overall_fit(slices)
    te_pred = (kbar - alpha) / gamma
    rmse = float(np.sqrt(np.nanmean((te_pred - te) ** 2)))
    bias = float(np.nanmean(te_pred - te))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    finite = np.isfinite(te_pred)
    hi = max(float(te.max()), float(np.nanmax(te_pred[finite])) if finite.any() else 0.0)
    lo = min(0.0, float(np.nanmin(te_pred[finite])) if finite.any() else 0.0)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=ps.COLOR_GRAY, label="$y=x$")
    _scatter_by_m(ax, te, te_pred, m)
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"recovered TE  $(\bar K-\alpha)/\gamma$  (nats)")
    ax.set_title("Per-sample TE recovery (overall in-mix calibration)")
    ax.text(0.03, 0.97,
            f"RMSE = {rmse:.3f} nats\nbias = {bias:+.3f} nats\n"
            f"$\\gamma$ = {gamma:.2f}, $\\alpha$ = {alpha:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec=ps.COLOR_GRAY, alpha=0.85))
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ps.style_axes(ax)
    ps.save_figure(fig, path)


def _fig_te_residual_vs_te(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Per-sample recovery residual $(\widehat{TE}-\mathrm{TE})$ vs true TE."""
    import matplotlib.pyplot as plt

    te = np.asarray(arrs["te_true"], dtype=float)
    kbar = np.asarray(arrs["kbar"], dtype=float)
    m = np.asarray(arrs["M"], dtype=int)
    if te.size == 0:
        return
    alpha, gamma = _overall_fit(slices)
    resid = (kbar - alpha) / gamma - te

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0.0, ls="--", lw=1.0, color=ps.COLOR_GRAY, zorder=1)
    _scatter_by_m(ax, te, resid, m)
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"recovery residual  $\widehat{\mathrm{TE}}-\mathrm{TE}$  (nats)")
    ax.set_title("TE recovery residual vs true TE (heteroscedasticity)")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ps.style_axes(ax)
    ps.save_figure(fig, path)


def _fig_kld_vs_te_binned(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""$\bar K$ mean $\pm$ std per distinct true-TE level over the sample cloud."""
    import matplotlib.pyplot as plt

    te = np.asarray(arrs["te_true"], dtype=float)
    kbar = np.asarray(arrs["kbar"], dtype=float)
    if te.size == 0:
        return
    levels = np.array(sorted(set(np.round(te, 6).tolist())), dtype=float)
    means = np.array([kbar[np.round(te, 6) == lv].mean() for lv in levels])
    stds = np.array([kbar[np.round(te, 6) == lv].std() for lv in levels])

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(te, kbar, s=8, alpha=0.18, linewidths=0.0, color=ps.COLOR_LIGHT_GRAY,
               zorder=1, label="samples")
    ax.errorbar(levels, means, yerr=stds, fmt="o-", color=ps.COLOR_VERMILLION,
                ecolor=ps.COLOR_GRAY, elinewidth=1.0, capsize=3, lw=1.6, ms=5,
                zorder=3, label=r"mean $\pm$ std per TE level")
    ax.set_xlabel(r"true block TE  $\mathrm{TE}_{\mathrm{true}}$  (nats)")
    ax.set_ylabel(r"mean latent KL  $\bar K$  (nats)")
    ax.set_title("KLD monotonicity across true-TE levels")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ps.style_axes(ax)
    ps.save_figure(fig, path)


def _fig_corr_by_m_bars(path, rows_in, rows_ho, arrs, slices, controls) -> None:
    r"""Grouped bars of Pearson $r$ / Spearman $\rho$ / MI of $\bar K$ vs TE per $M$."""
    import matplotlib.pyplot as plt

    assoc = kld_te_association(arrs)["by_M"]
    if not assoc:
        return
    ms = sorted(int(k) for k in assoc)
    x = np.arange(len(ms))
    metrics = [("pearson", "Pearson r", ps.COLOR_BLUE),
               ("spearman", r"Spearman $\rho$", ps.COLOR_GREEN),
               ("mi", "MI (nats)", ps.COLOR_VERMILLION)]
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(ms) + 2.0), 4.2))
    for k, (key, label, color) in enumerate(metrics):
        vals = [assoc[str(mm)].get(key, float("nan")) for mm in ms]
        ax.bar(x + k * width, vals, width, label=label, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(mm) for mm in ms])
    ax.set_xlabel("informative-channel count $M$")
    ax.set_ylabel("association of $\\bar K$ with TE")
    ax.set_title("KLD-vs-TE association by channel count $M$")
    ax.legend(fontsize=8, frameon=False)
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
    )


if __name__ == "__main__":
    CONFIG_PATH = _DEFAULT_CONFIG
    RUN_CONFIG = {
        "run_tag": "G1_mix_gnll",
        "in_mix_tag": "G1_mix_base",
        "holdout_tag": "G1_mix_base_holdout",
        "ckpt_name": "final.ckpt",
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
        )
