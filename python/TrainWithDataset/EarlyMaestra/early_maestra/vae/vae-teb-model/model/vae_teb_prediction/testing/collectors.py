"""Data collection utilities for the VAE-TEB Lag-Attentive v1 testing pipeline.

Each collector iterates through a DataLoader, runs :class:`TestRunner` in
``inference_mode``, pulls the forward-dict outputs that matter for a given
analysis, and returns data in a standard format (``pandas.DataFrame``,
``numpy.ndarray``, or ``list[dict]``).

Every collector here is tightly tied to the lag-attn v1 forward contract:
``mu_full``, ``mu_base``, ``delta_mu_src``, ``z``, ``attn_weights``,
``te_lag_map``, and ``kld_per_t`` (see ``new_architecture.md``).

Example:
    >>> from testing.collectors import collect_metrics, collect_latents
    >>> df = collect_metrics(runner, test_loader, max_samples=1000)
    >>> latents = collect_latents(runner, test_loader, max_samples=500)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.metrics import (
    aggregate_te_lag_map,
    compute_attention_diagnostics,
    compute_forecast_metrics,
    compute_kld_per_sample,
    compute_residual_usage,
    compute_uplift_metrics,
)


# -----------------------------------------------------------------------------
# Per-sample metadata extractors
# -----------------------------------------------------------------------------


def _extract_guid(batch: Any, idx: int) -> Optional[str]:
    """Extract a GUID string from a batch at a given index.

    Handles tensor / numpy / list / bytes formats; returns None if the
    GUID field is absent or unparseable.

    Args:
        batch: Batch object with ``guid`` attribute.
        idx: Index within the batch.

    Returns:
        GUID string, or None if extraction fails.
    """
    guid_attr = getattr(batch, "guid", None)
    if guid_attr is None:
        return None

    try:
        raw = guid_attr[idx]
        if isinstance(raw, torch.Tensor):
            raw = raw.item() if raw.numel() == 1 else int(raw.item())
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return str(raw)
    except Exception:
        return None


def _extract_epoch(batch: Any, idx: int) -> Optional[float]:
    """Extract the per-sample epoch (seconds relative to delivery).

    Args:
        batch: Batch object with ``epoch`` attribute.
        idx: Index within the batch.

    Returns:
        Epoch in seconds (negative = before delivery), or None on failure.
    """
    epoch_attr = getattr(batch, "epoch", None)
    if epoch_attr is None:
        return None

    try:
        raw = epoch_attr[idx]
        if isinstance(raw, torch.Tensor):
            return float(raw.item())
        return float(raw)
    except Exception:
        return None


def _extract_label(batch: Any, idx: int) -> Optional[int]:
    """Extract the per-sample class label from ``target``.

    The ``target`` HDF5 field stores ``class_id * weight`` per timestep, so
    the class ID is recovered by taking the first non-zero value.

    Args:
        batch: Batch object with ``target`` attribute.
        idx: Index within the batch.

    Returns:
        Class label integer (1=HEALTHY, 2=ACIDOSIS, 3=HIE), 0 if the
        sample is all-pad, or None on failure.
    """
    target_attr = getattr(batch, "target", None)
    if target_attr is None:
        return None

    try:
        raw = target_attr[idx]
        if isinstance(raw, torch.Tensor):
            if raw.dim() > 0:
                nonzero = raw[raw > 0]
                if len(nonzero) > 0:
                    return int(nonzero[0].item())
                return 0
            return int(raw.item())
        return int(raw)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Raw-signal denormalisation (fhr / up)
# -----------------------------------------------------------------------------


def resolve_fhr_up_denorm_stats(
    loader: Any,
) -> Dict[str, Dict[str, float]]:
    """Resolve ``(mean, std)`` for ``fhr`` and ``up`` from a loader.

    The HDF5 dataloader z-score normalises ``fhr`` and ``up`` using
    per-field scalar stats loaded from the stats HDF5. To plot the
    **actual raw** traces we need to invert that normalisation via
    ``x_raw = x_norm * std + mean``. This helper reaches through the
    loader to fetch those stats when normalisation is enabled, and
    returns an empty dict when it is not (e.g. a loader without a
    stats file, or when ``fhr`` / ``up`` are not in ``normalize_fields``).

    Args:
        loader: DataLoader (or anything exposing a ``.dataset``) that
            wraps a ``CombinedHDF5Dataset``.

    Returns:
        ``{"fhr": {"mean": .., "std": ..}, "up": {"mean": .., "std": ..}}``
        containing only the fields that were actually normalised. Empty
        dict when the loader has no stats.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return {}

    # Unwrap common PyTorch wrappers (Subset / ConcatDataset).
    for _ in range(3):
        inner = getattr(dataset, "dataset", None)
        if inner is None or inner is dataset:
            break
        dataset = inner

    getter = getattr(dataset, "get_normalization_stats", None)
    stats_raw = getter() if callable(getter) else getattr(
        dataset, "normalization_stats", None
    )
    if not isinstance(stats_raw, dict) or not stats_raw:
        return {}
    stats_all: Dict[str, Any] = stats_raw

    out: Dict[str, Dict[str, float]] = {}
    for field in ("fhr", "up"):
        entry = stats_all.get(field)
        if not isinstance(entry, dict):
            continue
        mean = entry.get("mean")
        std = entry.get("std")
        if mean is None or std is None:
            continue
        try:
            out[field] = {"mean": float(mean), "std": float(std)}
        except (TypeError, ValueError):
            continue
    return out


def denormalize_signal(
    signal: Optional[np.ndarray],
    stats: Optional[Dict[str, float]],
) -> Optional[np.ndarray]:
    """Invert the z-score normalisation applied to ``fhr`` / ``up``.

    Args:
        signal: Normalised 1-D array (or ``None``).
        stats: ``{"mean": ..., "std": ...}`` from
            :func:`resolve_fhr_up_denorm_stats`, or ``None`` to skip
            denormalisation.

    Returns:
        ``signal * std + mean`` as a ``float32`` array when stats are
        available; the input unchanged otherwise.
    """
    if signal is None or stats is None:
        return signal
    return np.asarray(signal, dtype=np.float32) * float(stats["std"]) + float(
        stats["mean"]
    )


# -----------------------------------------------------------------------------
# Primary collectors (replace the old raw-FHR workflow)
# -----------------------------------------------------------------------------


def collect_metrics(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-sample feature-forecast, uplift, residual and KL metrics.

    For every sample this runs the model once, builds the unfolded future
    feature target via :meth:`TestRunner.build_future_target`, and calls
    :func:`compute_forecast_metrics`, :func:`compute_uplift_metrics`,
    :func:`compute_residual_usage`, and :func:`compute_kld_per_sample`.

    The resulting DataFrame preserves a ``kld`` column (an alias of
    ``kld_mean``) so downstream consumers that key on ``kld`` (notably
    ``TE_Calculated/te_kld_analysis.py``) keep working unchanged.

    Args:
        runner: :class:`TestRunner` with a loaded model and device.
        loader: PyTorch DataLoader yielding the batch objects consumed by
            the runner.
        max_samples: Maximum samples to process (None = all).

    Returns:
        DataFrame with columns ``[guid, epoch, label, feat_mse_total,
        feat_mse_st, feat_mse_ph, feat_r2_total, base_mse_total,
        uplift_abs, uplift_rel, residual_ratio, kld_mean, kld]``.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)

            fcst = compute_forecast_metrics(
                outputs["mu_full"], y_plus, runner.warmup_steps, runner.horizon
            )
            uplift = compute_uplift_metrics(
                outputs["mu_full"],
                outputs["mu_base"],
                y_plus,
                runner.warmup_steps,
                runner.horizon,
            )
            usage = compute_residual_usage(
                outputs["delta_mu_src"],
                outputs["mu_full"],
                runner.warmup_steps,
                runner.horizon,
            )
            kld_sample = compute_kld_per_sample(outputs, runner.warmup_steps)

            # Per-dim KLD (closed-form), averaged over the post-warmup
            # anchor range. Enables the per-dimension heatmap in the
            # TE_Calculated pipeline. Only emitted when all four moment
            # tensors are present on the forward output.
            per_dim_means: Optional[Any] = None
            if (
                "mu_prior" in outputs
                and "logvar_prior" in outputs
                and "mu_post" in outputs
                and "logvar_post" in outputs
            ):
                mu_prior = outputs["mu_prior"]
                logvar_prior = outputs["logvar_prior"]
                mu_post = outputs["mu_post"]
                logvar_post = outputs["logvar_post"]
                kld_per_dim_t = 0.5 * (
                    logvar_prior
                    - logvar_post
                    + (logvar_post.exp() + (mu_post - mu_prior) ** 2)
                    / logvar_prior.exp()
                    - 1.0
                )  # (B, T, d_z)
                T = kld_per_dim_t.shape[1]
                warm = min(max(0, int(runner.warmup_steps)), T)
                if warm < T:
                    kld_valid = kld_per_dim_t[:, warm:, :]
                else:
                    kld_valid = kld_per_dim_t
                per_dim_means = kld_valid.mean(dim=1).detach().cpu().numpy()

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                kld_val = float(kld_sample[idx].cpu().item())
                record: Dict[str, Any] = {
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "feat_mse_total": float(fcst["feat_mse_total"][idx].cpu().item()),
                    "feat_mse_st": float(fcst["feat_mse_st"][idx].cpu().item()),
                    "feat_mse_ph": float(fcst["feat_mse_ph"][idx].cpu().item()),
                    "feat_r2_total": float(fcst["feat_r2_total"][idx].cpu().item()),
                    "base_mse_total": float(uplift["l_base"][idx].cpu().item()),
                    "uplift_abs": float(uplift["uplift_abs"][idx].cpu().item()),
                    "uplift_rel": float(uplift["uplift_rel"][idx].cpu().item()),
                    "residual_ratio": float(usage["residual_ratio"][idx].cpu().item()),
                    "kld_mean": kld_val,
                    "kld": kld_val,  # alias for backward compatibility
                }
                if per_dim_means is not None:
                    dim_vec = per_dim_means[idx]
                    for d in range(dim_vec.shape[0]):
                        record[f"kld_dim_{d}"] = float(dim_vec[d])
                records.append(record)
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)


def collect_latents(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Collect latent trajectories for all processed samples.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        ``(N * T, d_z)`` array with the full latent trajectory of each
        sample flattened along the time axis.
    """
    chunks: List[np.ndarray] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            latent = outputs.get("z")
            if latent is None:
                continue

            latent_np = latent.detach().cpu().numpy()
            for i in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                chunks.append(latent_np[i])
                processed += 1

            if max_samples and processed >= max_samples:
                break

    if not chunks:
        return np.array([])

    return np.concatenate(chunks, axis=0)


def collect_predictions(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect detailed per-sample predictions for diagnostic plots.

    Each record holds numpy copies of every quantity a sample-level
    diagnostic plot might need: the full forecast, baseline forecast,
    residual, ground-truth feature trajectory, latent, attention weights,
    TE lag map, per-timestep KL, raw FHR/UP traces, and summary metrics.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        List of dicts, one per sample, with keys described in the module
        docstring of ``plot_single_samples.py``.
    """
    samples: List[Dict[str, Any]] = []
    processed = 0

    # Stats for reversing fhr/up z-score normalisation before plotting.
    # When the loader is not normalising these fields, the helper returns
    # an empty dict and ``denormalize_signal`` becomes a no-op.
    denorm_stats = resolve_fhr_up_denorm_stats(loader)
    fhr_stats = denorm_stats.get("fhr")
    up_stats = denorm_stats.get("up")

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)

            fcst = compute_forecast_metrics(
                outputs["mu_full"], y_plus, runner.warmup_steps, runner.horizon
            )
            uplift = compute_uplift_metrics(
                outputs["mu_full"],
                outputs["mu_base"],
                y_plus,
                runner.warmup_steps,
                runner.horizon,
            )
            usage = compute_residual_usage(
                outputs["delta_mu_src"],
                outputs["mu_full"],
                runner.warmup_steps,
                runner.horizon,
            )
            kld_sample = compute_kld_per_sample(outputs, runner.warmup_steps)

            mu_full_np = outputs["mu_full"].detach().cpu().numpy()
            mu_base_np = outputs["mu_base"].detach().cpu().numpy()
            delta_np = outputs["delta_mu_src"].detach().cpu().numpy()
            z_np = outputs["z"].detach().cpu().numpy()
            attn_np = outputs["attn_weights"].detach().cpu().numpy()
            te_lag_np = outputs["te_lag_map"].detach().cpu().numpy()
            kld_t_np = outputs["kld_per_t"].detach().cpu().numpy()
            y_plus_np = y_plus.detach().cpu().numpy()

            # Per-dimension KL (for dim-heat plots). Closed-form KL tensor
            # is computed by the model for us when we ask; replicate here
            # using the same formula without re-running the encoder.
            mu_prior = outputs["mu_prior"]
            logvar_prior = outputs["logvar_prior"]
            mu_post = outputs["mu_post"]
            logvar_post = outputs["logvar_post"]
            kld_per_dim = 0.5 * (
                logvar_prior
                - logvar_post
                + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
                - 1.0
            )
            kld_per_dim_np = kld_per_dim.detach().cpu().numpy()

            fhr_np = batch.fhr.detach().cpu().numpy() if hasattr(batch, "fhr") and isinstance(batch.fhr, torch.Tensor) else None
            up_np = batch.up.detach().cpu().numpy() if hasattr(batch, "up") and isinstance(batch.up, torch.Tensor) else None

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                samples.append({
                    "mu_full": mu_full_np[idx],
                    "mu_base": mu_base_np[idx],
                    "delta_src": delta_np[idx],
                    "y_plus": y_plus_np[idx],
                    "z": z_np[idx],
                    "attn": attn_np[idx],
                    "te_lag": te_lag_np[idx],
                    "kld_t": kld_t_np[idx],
                    "kld_per_dim": kld_per_dim_np[idx],
                    "fhr": denormalize_signal(
                        fhr_np[idx] if fhr_np is not None else None, fhr_stats
                    ),
                    "up": denormalize_signal(
                        up_np[idx] if up_np is not None else None, up_stats
                    ),
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "metrics": {
                        "feat_mse_total": float(fcst["feat_mse_total"][idx].cpu().item()),
                        "feat_r2_total": float(fcst["feat_r2_total"][idx].cpu().item()),
                        "base_mse_total": float(uplift["l_base"][idx].cpu().item()),
                        "uplift_abs": float(uplift["uplift_abs"][idx].cpu().item()),
                        "uplift_rel": float(uplift["uplift_rel"][idx].cpu().item()),
                        "residual_ratio": float(usage["residual_ratio"][idx].cpu().item()),
                        "kld_mean": float(kld_sample[idx].cpu().item()),
                    },
                })
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return samples


def collect_kld_trajectory(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-timestep KL trajectory records for every sample.

    Reads ``outputs["kld_per_t"]`` directly (the model's TE analysis head
    produces this as ``sum_d KL(q || p)``), so no recomputation is needed.
    Warmup timesteps are dropped by the model's warmup masking; we skip
    non-finite values defensively.

    The CSV schema is preserved from the legacy pipeline so
    ``TE_Calculated`` modules keep working:
    ``[guid, epoch, hours_before, label, timestep, kld_mean, latent_0 ... latent_{d_z-1}]``.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        DataFrame with per-(sample, timestep) rows.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            kld_t = outputs.get("kld_per_t")
            latent = outputs.get("z")
            if kld_t is None:
                continue

            # Mask warmup region with NaN so we can skip it below.
            kld_t_f = kld_t.detach().to(torch.float32).clone()
            warmup = int(runner.warmup_steps)
            if warmup > 0 and kld_t_f.size(1) > warmup:
                kld_t_f[:, :warmup] = float("nan")

            T = int(kld_t_f.size(1))

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)
                hours_before = -epoch / 3600.0 if epoch is not None else None

                kld_vals = kld_t_f[idx].cpu().numpy()
                latent_vals = latent[idx].cpu().numpy() if latent is not None else None

                for t in range(T):
                    v = kld_vals[t]
                    if not np.isfinite(v):
                        continue
                    record: Dict[str, Any] = {
                        "guid": guid,
                        "epoch": epoch,
                        "hours_before": hours_before,
                        "label": label,
                        "timestep": t,
                        "kld_mean": float(v),
                    }
                    if latent_vals is not None:
                        for d in range(latent_vals.shape[1]):
                            record[f"latent_{d}"] = float(latent_vals[t, d])
                    records.append(record)

                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# New collectors for lag-attention diagnostics and forecast profiling
# -----------------------------------------------------------------------------


def collect_attention_maps(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect per-sample lag-attention summaries.

    For each sample this runs :func:`compute_attention_diagnostics` and
    stores the head-averaged attention, argmax lag, per-head entropy,
    head diversity, and the time-averaged ``alpha_mass_by_lag`` vector.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        List of dicts ``{guid, epoch, label, alpha_bar (T,L),
        argmax_lag (T,), entropy (T,M), head_diversity (T,),
        alpha_mass_by_lag (L,)}`` — all numpy arrays with NaN in warmup
        regions.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            diag = compute_attention_diagnostics(
                outputs["attn_weights"], runner.warmup_steps
            )

            alpha_bar_np = diag["alpha_bar"].detach().cpu().numpy()
            argmax_np = diag["argmax_lag"].detach().cpu().numpy()
            entropy_np = diag["entropy"].detach().cpu().numpy()
            head_div_np = diag["head_diversity"].detach().cpu().numpy()
            mass_np = diag["alpha_mass_by_lag"].detach().cpu().numpy()

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                records.append({
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "alpha_bar": alpha_bar_np[idx],
                    "argmax_lag": argmax_np[idx],
                    "entropy": entropy_np[idx],
                    "head_diversity": head_div_np[idx],
                    "alpha_mass_by_lag": mass_np[idx],
                })
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return records


def collect_te_lag_maps(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect time-averaged TE lag signatures for every sample.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        DataFrame one-row-per-sample with columns ``[guid, epoch, label,
        te_lag_mean_0 ... te_lag_mean_{L-1}, te_lag_argmax]``.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            agg = aggregate_te_lag_map(outputs["te_lag_map"], runner.warmup_steps)

            mean_np = agg["te_lag_mean"].detach().cpu().numpy()
            argmax_np = agg["te_lag_argmax"].detach().cpu().numpy()
            L = int(mean_np.shape[1])

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                rec: Dict[str, Any] = {
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "te_lag_argmax": int(argmax_np[idx]),
                }
                for k in range(L):
                    rec[f"te_lag_mean_{k}"] = float(mean_np[idx, k])
                records.append(rec)
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)


def collect_forecast_errors_per_horizon(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-(sample, horizon step) forecast error.

    For each sample we compute :func:`compute_forecast_metrics` and unpack
    ``feat_mse_per_horizon (B, H_d)`` into one row per horizon step, plus
    the scattering/phase block split evaluated only at that horizon step.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        DataFrame with columns ``[guid, epoch, label, h, mse_step, mse_st,
        mse_ph]``.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr_st.size(0))

            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)

            mu_full = outputs["mu_full"]
            T, H_d, C = mu_full.shape[1], mu_full.shape[2], mu_full.shape[3]
            c_st = min(43, int(C))

            warmup = int(runner.warmup_steps)
            T_valid = max(T - int(H_d), 0)
            start = max(0, min(warmup, T_valid))

            mu_v = mu_full[:, start:T_valid, :, :]
            y_v = y_plus[:, start:T_valid, :, :]
            if mu_v.numel() == 0:
                continue

            diff_sq = (mu_v - y_v).pow(2)                   # (B, T_v, H_d, C)
            mse_step = diff_sq.mean(dim=(1, 3))              # (B, H_d)
            mse_st = diff_sq[..., :c_st].mean(dim=(1, 3)) if c_st > 0 else torch.zeros_like(mse_step)
            mse_ph = diff_sq[..., c_st:].mean(dim=(1, 3)) if c_st < int(C) else torch.zeros_like(mse_step)

            mse_step_np = mse_step.detach().cpu().numpy()
            mse_st_np = mse_st.detach().cpu().numpy()
            mse_ph_np = mse_ph.detach().cpu().numpy()

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)
                for h in range(int(H_d)):
                    records.append({
                        "guid": guid,
                        "epoch": epoch,
                        "label": label,
                        "h": h,
                        "mse_step": float(mse_step_np[idx, h]),
                        "mse_st": float(mse_st_np[idx, h]),
                        "mse_ph": float(mse_ph_np[idx, h]),
                    })
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)
