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

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.metrics import (
    aggregate_selected_pca_scores,
    aggregate_te_lag_map,
    compute_attention_diagnostics,
    compute_forecast_metrics,
    compute_raw_forecast_metrics,
    compute_kld_aggregate_tensors,
    compute_kld_aggregates_per_sample,
    compute_kld_per_sample,
    compute_posterior_drift,
    compute_residual_usage,
    compute_uplift_metrics,
    fit_pca_kld_per_dim,
    project_kld_per_dim,
    select_pca_components,
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


def _extract_te_true(batch: Any, idx: int) -> Optional[float]:
    r"""Extract the per-sample analytic true transfer entropy (nats).

    Synthetic batches built from :class:`SyntheticTEDataset` carry a
    ``te_true`` field per sample (the dataset-level block TE, or the per-cell
    ``sample_te_true`` for the mixed-population cache). Real CTG / HDF5 batches
    have no such field, so this returns ``None`` and the caller simply omits the
    annotation.

    Args:
        batch: Batch object that may carry a ``te_true`` attribute.
        idx: Index within the batch.

    Returns:
        The true block TE $\mathrm{TE}_{\mathrm{true}}$ in nats, or ``None`` if
        the field is absent or unparseable.
    """
    te_attr = getattr(batch, "te_true", None)
    if te_attr is None:
        return None

    try:
        raw = te_attr[idx]
        if isinstance(raw, torch.Tensor):
            return float(raw.item())
        return float(raw)
    except Exception:
        return None


def _extract_scalar_field(batch: Any, field: str, idx: int) -> Optional[float]:
    r"""Extract an optional per-sample float ``field`` from a batch (guarded).

    Synthetic v2 batches carry per-sample scalars such as ``te_scat``, ``frac_phi``, and
    ``te_raw``; real CTG / HDF5 batches do not. Returns ``None`` when the field is absent
    or unparseable so callers can simply omit the annotation.

    Args:
        batch: Batch object that may carry a ``field`` attribute of shape ``(B,)``.
        field: The attribute name to read (e.g. ``"te_scat"``).
        idx: Index within the batch.

    Returns:
        The per-sample float, or ``None`` if the field is absent / unparseable.
    """
    attr = getattr(batch, field, None)
    if attr is None:
        return None
    try:
        raw = attr[idx]
        if isinstance(raw, torch.Tensor):
            return float(raw.item())
        return float(raw)
    except Exception:
        return None


def _extract_int_field(batch: Any, field: str, idx: int) -> Optional[int]:
    r"""Extract an optional per-sample integer ``field`` from a batch (guarded).

    Args:
        batch: Batch object that may carry a ``field`` attribute of shape ``(B,)``.
        field: The attribute name (e.g. ``"cell_id"``).
        idx: Index within the batch.

    Returns:
        The per-sample integer, or ``None`` if absent / unparseable.
    """
    value = _extract_scalar_field(batch, field, idx)
    return None if value is None else int(round(value))


def _extract_delay(batch: Any, idx: int) -> Optional[int]:
    r"""Extract the per-sample fixed lag $D$ (``delay`` or ``sample_delay``), guarded."""
    for field in ("delay", "sample_delay"):
        if getattr(batch, field, None) is not None:
            return _extract_int_field(batch, field, idx)
    return None


def _extract_array_field(batch: Any, field: str, idx: int) -> Optional[np.ndarray]:
    r"""Extract an optional per-sample array ``field`` (e.g. ``true_lag_tt``) from a batch.

    Args:
        batch: Batch object that may carry a ``field`` attribute of shape ``(B, ...)``.
        field: The attribute name (e.g. ``"true_lag_tt"`` or ``"true_lag_band"``).
        idx: Index within the batch.

    Returns:
        The per-sample array as ``numpy``, or ``None`` if the field is absent.
    """
    attr = getattr(batch, field, None)
    if attr is None:
        return None
    try:
        raw = attr[idx]
        if isinstance(raw, torch.Tensor):
            return raw.detach().cpu().numpy()
        return np.asarray(raw)
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
    *,
    pca_components: int = 3,
    pca_output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Collect per-sample feature-forecast, uplift, residual and KL metrics.

    For every sample this runs the model once, builds the unfolded future
    feature target via :meth:`TestRunner.build_future_target`, and calls
    :func:`compute_forecast_metrics`, :func:`compute_uplift_metrics`,
    :func:`compute_residual_usage`, :func:`compute_kld_per_sample`,
    :func:`compute_attention_diagnostics`, :func:`aggregate_te_lag_map`,
    and :func:`compute_posterior_drift`.

    Beyond the legacy ``kld``/``kld_mean`` scalar, additional TE-surrogate
    columns are emitted for the comparison pipeline:
    ``posterior_drift_norm``, ``delta_src_norm``, ``attention_entropy_mean``,
    ``attention_concentration_mean``, ``te_lag_peak``,
    ``te_lag_total_mass``, plus PCA top-3 of the per-dim KL trajectory
    (``kld_pc1``, ``kld_pc2``, ``kld_pc3``).

    Args:
        runner: :class:`TestRunner` with a loaded model and device.
        loader: PyTorch DataLoader yielding the batch objects consumed by
            the runner.
        max_samples: Maximum samples to process (None = all).
        pca_components: Number of PCA components to retain on the per-dim
            KL trajectory (default 3).
        pca_output_dir: Optional directory for persisting PCA artifacts
            (``ev_ratio.json``, ``components.npy``, ``mean.npy``). When
            ``None`` the artifacts are written under
            ``runner.output_dir / "pca_kld"``.

    Returns:
        DataFrame with the legacy columns plus the new TE surrogates and
        ``kld_pc1``/``kld_pc2``/``kld_pc3`` (when fitting succeeds). The
        ``kld`` alias is preserved for backward compatibility with
        ``TE_Calculated/te_kld_analysis.py``.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    # Per-batch buffers for PCA on the per-time per-dim KL.
    per_dim_t_chunks: List[np.ndarray] = []
    record_index_to_sample_offset: List[int] = []
    sample_T: Optional[int] = None
    sample_dz: Optional[int] = None

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr.size(0))

            outputs = runner.forward(batch)
            x_plus = runner.build_future_target(batch)      # (B, T_valid, H, R)

            fcst = compute_raw_forecast_metrics(
                outputs["mu_full"], x_plus, runner.warmup_steps, runner.horizon
            )
            uplift = compute_uplift_metrics(
                outputs["mu_full"],
                outputs["mu_base"],
                x_plus,
                runner.warmup_steps,
                runner.horizon,
            )
            usage = compute_residual_usage(
                outputs["delta_mu_src"],
                outputs["mu_full"],
                runner.warmup_steps,
                runner.horizon,
            )
            kld_agg_sample = compute_kld_aggregates_per_sample(
                outputs, runner.warmup_steps
            )
            kld_sample = kld_agg_sample["kld_mean"]
            kld_sum_sample = kld_agg_sample["kld_sum"]
            kld_l2_sample = kld_agg_sample["kld_l2"]

            # Lag-attention diagnostics (entropy / concentration).
            attn_entropy_mean: Optional[np.ndarray] = None
            attn_conc_mean: Optional[np.ndarray] = None
            attn_weights = outputs.get("attn_weights")
            if attn_weights is not None and attn_weights.dim() == 4:
                diag = compute_attention_diagnostics(
                    attn_weights, runner.warmup_steps
                )
                # entropy: (B, T, M) with NaN in warmup; collapse over heads
                # then over time using nanmean. Warmup rows are entirely NaN
                # by design, so ``nanmean`` emits ``RuntimeWarning: Mean of
                # empty slice`` for them -- expected, not a bug, suppress.
                ent = diag["entropy"].detach().cpu().numpy()
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", category=RuntimeWarning)
                    head_mean = np.nanmean(ent, axis=2)  # (B, T)
                    ent_mean = np.asarray(np.nanmean(head_mean, axis=1))  # (B,)
                attn_entropy_mean = ent_mean
                L = int(attn_weights.shape[-1])
                norm = math.log(L) if L > 1 else 1.0
                attn_conc_mean = 1.0 - ent_mean / max(norm, 1e-12)

            # TE lag map: peak lag and total mass per sample.
            te_lag_peak: Optional[np.ndarray] = None
            te_lag_total_mass: Optional[np.ndarray] = None
            te_lag_map = outputs.get("te_lag_map")
            if te_lag_map is not None and te_lag_map.dim() == 3:
                agg = aggregate_te_lag_map(te_lag_map, runner.warmup_steps)
                te_lag_peak = agg["te_lag_argmax"].detach().cpu().numpy()
                te_lag_total_mass = (
                    agg["te_lag_mean"].detach().cpu().numpy().sum(axis=-1)
                )

            # Posterior drift surrogate (||mu_q - mu_p||^2 averaged over t).
            drift: Optional[np.ndarray] = None
            if "mu_prior" in outputs and "mu_post" in outputs:
                drift = compute_posterior_drift(
                    outputs["mu_prior"], outputs["mu_post"], runner.warmup_steps
                ).detach().cpu().numpy()

            # Per-dim KLD (closed-form), per-time per-dim tensor needed for
            # both per-sample mean (existing) AND PCA fit (new).
            per_dim_means: Optional[np.ndarray] = None
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

                # Stash the (B, T, d_z) tensor with NaN in warmup for PCA.
                per_dim_t_np = kld_per_dim_t.detach().to(torch.float32).cpu().numpy()
                if warm > 0 and warm < T:
                    per_dim_t_np = per_dim_t_np.copy()
                    per_dim_t_np[:, :warm, :] = np.nan
                per_dim_t_chunks.append(per_dim_t_np)
                if sample_T is None:
                    sample_T = int(per_dim_t_np.shape[1])
                    sample_dz = int(per_dim_t_np.shape[2])

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                kld_val = float(kld_sample[idx].cpu().item())
                record: Dict[str, Any] = {
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    # Raw-forecast quality (no scattering st/ph split). ``feat_mse_total`` /
                    # ``feat_r2_total`` are kept as aliases of the raw MSE / R^2 so the
                    # domain-agnostic histogram + per-class post-processors keep finding a headline
                    # forecast column.
                    "feat_mse_total": float(fcst["raw_mse"][idx].cpu().item()),
                    "feat_r2_total": float(fcst["raw_r2"][idx].cpu().item()),
                    "raw_mse": float(fcst["raw_mse"][idx].cpu().item()),
                    "raw_vaf": float(fcst["raw_vaf"][idx].cpu().item()),
                    "raw_snr": float(fcst["raw_snr"][idx].cpu().item()),
                    "raw_r2": float(fcst["raw_r2"][idx].cpu().item()),
                    "raw_lowpass_mse": float(fcst["raw_lowpass_mse"][idx].cpu().item()),
                    "base_mse_total": float(uplift["l_base"][idx].cpu().item()),
                    "uplift_abs": float(uplift["uplift_abs"][idx].cpu().item()),
                    "uplift_rel": float(uplift["uplift_rel"][idx].cpu().item()),
                    "residual_ratio": float(usage["residual_ratio"][idx].cpu().item()),
                    "delta_src_norm": float(usage["delta_norm"][idx].cpu().item()),
                    "kld_mean": kld_val,
                    "kld_sum": float(kld_sum_sample[idx].cpu().item()),
                    "kld_l2": float(kld_l2_sample[idx].cpu().item()),
                    "kld": kld_val,  # alias for backward compatibility
                }
                if drift is not None:
                    record["posterior_drift_norm"] = float(drift[idx])
                if attn_entropy_mean is not None and attn_conc_mean is not None:
                    record["attention_entropy_mean"] = float(attn_entropy_mean[idx])
                    record["attention_concentration_mean"] = float(
                        attn_conc_mean[idx]
                    )
                if te_lag_peak is not None and te_lag_total_mass is not None:
                    record["te_lag_peak"] = int(te_lag_peak[idx])
                    record["te_lag_total_mass"] = float(te_lag_total_mass[idx])
                if per_dim_means is not None:
                    dim_vec = per_dim_means[idx]
                    for d in range(dim_vec.shape[0]):
                        record[f"kld_dim_{d}"] = float(dim_vec[d])

                # Track which (chunk_idx, in_chunk_idx) this record maps to
                # so we can back-fill PCA scores after the loop.
                record_index_to_sample_offset.append(len(records))
                records.append(record)
                processed += 1

            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(records)

    # --- PCA on stacked per-time per-dim KL trajectories --------------------
    if (
        per_dim_t_chunks
        and sample_T is not None
        and sample_dz is not None
        and pca_components > 0
    ):
        try:
            stacked = np.concatenate(per_dim_t_chunks, axis=0)  # (N, T, d_z)
            stacked = stacked[: len(df)]
            # Fit as many PCs as the latent KL space supports, then select the
            # most contrastive subset. This avoids hard-coding "top 3 by
            # eigenvalue" while keeping legacy kld_pc1..3 columns available.
            n_fit = max(1, int(sample_dz))
            pca_model, projected, ev_ratio = fit_pca_kld_per_dim(
                stacked, n_components=n_fit
            )
            # Per-sample mean of each component (ignoring NaN warmup).
            with np.errstate(invalid="ignore"):
                pc_means = np.nanmean(projected, axis=1)  # (N, k)

            for k in range(pc_means.shape[1]):
                df[f"kld_pc{k + 1}"] = pc_means[:, k].astype(float)
            legacy_count = min(3, pc_means.shape[1])
            if legacy_count >= 3:
                df["kld_pca_l2_top3"] = np.sqrt(
                    np.nansum(pc_means[:, :3] ** 2, axis=1)
                ).astype(float)

            labels = df["label"].to_numpy() if "label" in df.columns else None
            selection = select_pca_components(
                projected,
                ev_ratio,
                n_select=max(1, int(pca_components)),
                labels=labels,
            )
            selected_aggs = aggregate_selected_pca_scores(
                selection["pc_means"],
                selection["selected_indices"],
                selection["signs"],
            )
            selected_scores = selected_aggs["selected_scores"]
            for k in range(selected_scores.shape[1]):
                original_pc = int(selection["selected_1based"][k])
                df[f"kld_pc_selected_{k + 1}"] = selected_scores[:, k].astype(float)
                df[f"kld_pc_selected_{k + 1}_source_pc"] = original_pc
            df["kld_pca_l2_selected"] = selected_aggs["l2"].astype(float)
            df["kld_pca_abs_sum_selected"] = selected_aggs["abs_sum"].astype(float)
            df["kld_pca_signed_sum_selected"] = selected_aggs["signed_sum"].astype(float)

            target_dir = (
                Path(pca_output_dir)
                if pca_output_dir is not None
                else Path(runner.output_dir) / "pca_kld"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            with open(target_dir / "ev_ratio.json", "w") as fh:
                json.dump(
                    {
                        "n_components": int(pca_model.n_components_),
                        "explained_variance_ratio": [float(x) for x in ev_ratio],
                        "n_samples_fitted": int(stacked.shape[0]),
                        "T": int(sample_T),
                        "d_z": int(sample_dz),
                        "legacy_top_components_emitted": int(legacy_count),
                    },
                    fh,
                    indent=2,
                )
            with open(target_dir / "selection.json", "w") as fh:
                json.dump(
                    {
                        "n_selected": int(len(selection["selected_indices"])),
                        "selected_indices_0based": [
                            int(x) for x in selection["selected_indices"]
                        ],
                        "selected_indices_1based": [
                            int(x) for x in selection["selected_1based"]
                        ],
                        "signs": [float(x) for x in selection["signs"]],
                        "contrast_type": str(selection["contrast_type"]),
                        "score": [float(x) for x in selection["score"]],
                        "contrast": [float(x) for x in selection["contrast"]],
                        "explained_variance_ratio": [float(x) for x in ev_ratio],
                    },
                    fh,
                    indent=2,
                )
            np.save(
                target_dir / "components.npy",
                np.asarray(pca_model.components_, dtype=np.float32),
            )
            np.save(
                target_dir / "mean.npy",
                np.asarray(pca_model.mean_, dtype=np.float32),
            )
        except Exception as exc:  # noqa: BLE001
            # PCA is auxiliary; never let it break the main metrics CSV,
            # but surface the failure so a missing kld_pc* column can be
            # distinguished from "no per-dim KLD was collected".
            logger.warning(
                f"collect_metrics: PCA fit on per-dim KLD failed "
                f"({type(exc).__name__}: {exc}); kld_pc*, kld_pca_l2_*, "
                f"and kld_pc_selected_* columns will be omitted."
            )

    return df


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
            batch_size = int(batch.fhr.size(0))

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


def _optional_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    """Detach a tensor to numpy, tolerating ``None`` (a key an older model never emitted)."""
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def collect_calibration(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
    n_bins: int = 20,
) -> Dict[str, Any]:
    r"""Collect calibration statistics for the learned predictive distribution (G10).

    The decoder emits ``logvar_full`` on every forward, so no model change is needed -- but no
    collector read it before Sprint 5. Reading it turns the point forecast ``mu_full`` into a
    per-element Gaussian :math:`\mathcal{N}(\mu, \sigma^2)`, which can then be scored as a
    *distribution*: NLL, CRPS, interval coverage, and PIT reliability.

    A homoscedastic reference is fitted alongside. It matters twice: it says whether the
    learned :math:`\sigma` earns its keep, and it keeps the report meaningful on a checkpoint
    trained with a fixed ``sigma_obs``, whose ``logvar_full`` head never received a gradient
    (the model records no ``sigma_obs``, so this cannot be detected automatically).

    Args:
        runner: The configured :class:`TestRunner`.
        loader: A dataloader yielding batches with the model's input fields.
        max_samples: Cap on the number of samples to consume.
        levels: Nominal central-interval levels for coverage.
        n_bins: Quantile resolution of the reliability curves.

    Returns:
        A dict with ``per_sample`` (:class:`pandas.DataFrame`, one row per sample),
        ``per_horizon`` (long-format DataFrame keyed by ``h``), ``reliability`` (long-format
        DataFrame of nominal vs empirical PIT quantiles per horizon), and ``summary``
        (scalar dict, including the constant-sigma baseline).

    Raises:
        RuntimeError: If the model emits no ``logvar_full`` key at all.
    """
    from model.vae_teb_prediction.model.model_raw.testing.metrics import (
        compute_crps,
        compute_interval_coverage,
        compute_nll,
        compute_reliability_by_horizon,
        fit_constant_sigma,
    )

    rows: List[Dict[str, Any]] = []
    horizon_rows: List[Dict[str, Any]] = []
    reliability_sum: Optional[torch.Tensor] = None
    nominal: Optional[torch.Tensor] = None
    n_batches = 0
    resid_sq_sum, resid_count = 0.0, 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            logvar_full = outputs.get("logvar_full")
            if logvar_full is None:
                raise RuntimeError(
                    "model emitted no 'logvar_full'; calibration needs the decoder's "
                    "observation log-variance head."
                )
            mu_full = outputs["mu_full"]
            y_plus = runner.build_future_target(batch)
            warmup, horizon = runner.warmup_steps, runner.horizon

            nll = compute_nll(mu_full, logvar_full, y_plus, warmup, horizon)
            crps = compute_crps(mu_full, logvar_full, y_plus, warmup, horizon)
            cov = compute_interval_coverage(
                mu_full, logvar_full, y_plus, warmup, horizon, levels=levels
            )
            rel = compute_reliability_by_horizon(
                mu_full, logvar_full, y_plus, warmup, horizon, n_bins=n_bins
            )

            if nominal is None:
                nominal = rel["nominal"].detach().cpu()
            empirical = rel["empirical"].detach().cpu()
            if torch.isfinite(empirical).all():
                reliability_sum = (
                    empirical if reliability_sum is None else reliability_sum + empirical
                )
                n_batches += 1

            # Pooled residual scale, for the constant-sigma reference.
            sigma_hat = fit_constant_sigma(mu_full, y_plus, warmup, horizon)
            n_elem = int(mu_full[:, warmup : max(mu_full.shape[1] - horizon, warmup)].numel())
            resid_sq_sum += float(sigma_hat) ** 2 * n_elem
            resid_count += n_elem

            batch_size = int(mu_full.shape[0])
            for idx in range(batch_size):
                # Raw calibration (S6-T05): single-block scores -- no scattering st/ph split.
                row: Dict[str, Any] = {
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "nll": float(nll["nll_total"][idx]),
                    "crps": float(crps["crps_total"][idx]),
                    "sharpness": float(cov["sharpness"][idx]),
                }
                for j, level in enumerate(levels):
                    row[f"coverage_{int(round(level * 100))}"] = float(cov["coverage"][idx, j])
                rows.append(row)

                for h in range(int(nll["nll_per_horizon"].shape[1])):
                    horizon_rows.append({
                        "guid": row["guid"],
                        "label": row["label"],
                        "h": h,
                        "nll": float(nll["nll_per_horizon"][idx, h]),
                        "crps": float(crps["crps_per_horizon"][idx, h]),
                        "sharpness": float(cov["sharpness_per_horizon"][idx, h]),
                    })

    per_sample = pd.DataFrame(rows)
    per_horizon = pd.DataFrame(horizon_rows)

    reliability = pd.DataFrame()
    if reliability_sum is not None and nominal is not None and n_batches > 0:
        mean_empirical = (reliability_sum / n_batches).numpy()
        reliability = pd.DataFrame([
            {"h": h, "nominal": float(nominal[b]), "empirical": float(mean_empirical[h, b])}
            for h in range(mean_empirical.shape[0])
            for b in range(mean_empirical.shape[1])
        ])

    const_sigma = math.sqrt(resid_sq_sum / resid_count) if resid_count else float("nan")
    summary: Dict[str, Any] = {
        "n_samples": int(len(per_sample)),
        "constant_sigma": const_sigma,
    }
    if not per_sample.empty:
        summary.update(
            nll_mean=float(per_sample["nll"].mean()),
            crps_mean=float(per_sample["crps"].mean()),
            sharpness_mean=float(per_sample["sharpness"].mean()),
        )
        for level in levels:
            key = f"coverage_{int(round(level * 100))}"
            summary[key] = float(per_sample[key].mean())
            summary[f"{key}_error"] = summary[key] - float(level)
        # What the learned heteroscedastic sigma buys over one global scale.
        if math.isfinite(const_sigma) and const_sigma > 0.0:
            summary["nll_constant_sigma"] = (
                0.9189385332046727
                + math.log(const_sigma)
                + 0.5  # E[(y-mu)^2] / const_sigma^2 == 1 by construction
            )
            summary["nll_gain_over_constant"] = (
                summary["nll_constant_sigma"] - summary["nll_mean"]
            )
    if not reliability.empty:
        summary["reliability_max_deviation"] = float(
            (reliability["empirical"] - reliability["nominal"]).abs().max()
        )

    logger.info(
        "calibration: {} samples, NLL {:.4f}, CRPS {:.4f}, sharpness {:.4f}, "
        "constant sigma {:.4f}",
        summary["n_samples"], summary.get("nll_mean", float("nan")),
        summary.get("crps_mean", float("nan")), summary.get("sharpness_mean", float("nan")),
        const_sigma,
    )
    return {
        "per_sample": per_sample,
        "per_horizon": per_horizon,
        "reliability": reliability,
        "summary": summary,
    }


def _anchor_support_mask(runner: TestRunner, seq_len: int, device: torch.device) -> torch.Tensor:
    r"""Return the ``(T,)`` training-KL anchor mask ``[warmup, T-H)`` used by the v3 model.

    Prefers the model's own :meth:`_kld_support_mask` so the CMI features are summarised over
    exactly the anchors that carry supervised gradient (and feed :math:`K_{\mathrm{raw}}`),
    falling back to an explicit ``[warmup, T-H)`` window for models without that method.

    Args:
        runner: The configured :class:`TestRunner`.
        seq_len: Sequence length ``T``.
        device: Device for the returned mask.

    Returns:
        A ``(T,)`` float tensor of 1.0 (in support) / 0.0 (excluded).
    """
    mask_fn = getattr(runner.model, "_kld_support_mask", None)
    if callable(mask_fn):
        try:
            return mask_fn(seq_len, device=device, dtype=torch.float32)
        except Exception:  # noqa: BLE001 - fall back to the explicit window
            pass
    mask = torch.zeros(seq_len, device=device, dtype=torch.float32)
    warmup, t_valid = runner.valid_anchor_range(seq_len)
    if t_valid > warmup:
        mask[warmup:t_valid] = 1.0
    return mask


def collect_cmi_features(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Collect per-sample features for the neural-CMI comparison (G11, S6-T02).

    For every sample the estimator needs one triple :math:`(u, y, c)` plus the model's raw KL:

    - ``u`` -- the source causal summary :math:`H^u`, anchor-mean of ``source_state``;
    - ``c`` -- the target causal summary :math:`H^y = c_t`, anchor-mean of ``target_state``;
    - ``y`` -- the future-target summary, anchor-mean of the unfolded forecast window
      :math:`Y^+` flattened over ``(H, C_y)``;
    - ``k_raw`` -- :math:`K_{\mathrm{raw}}[b] = \sum_t m_t\,\texttt{kld\_per\_t}[b,t] / \sum_t
      m_t`, the per-step raw KL averaged over the anchor support (summed over latent dims).

    All summaries are anchor-means over the training support :math:`[w_{\mathrm{warm}}, T-H)`
    (:func:`_anchor_support_mask`), so ``c`` is the fixed-dimensional causal target history the
    spec pins as the conditioning set.

    Args:
        runner: The configured :class:`TestRunner`.
        loader: Dataloader yielding batches with the model's input fields.
        max_samples: Cap on the number of samples to consume.

    Returns:
        A dict with numpy arrays ``u`` ``(N, d_u)``, ``y`` ``(N, d_y)``, ``c`` ``(N, d_c)``,
        ``k_raw`` ``(N,)``, ``guids`` ``(N,)`` (object), ``labels`` ``(N,)`` (object), and the
        scalar ``n_samples``.

    Raises:
        RuntimeError: If the model emits no ``target_state`` / ``source_state`` / ``kld_per_t``.
    """
    u_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    c_rows: List[np.ndarray] = []
    k_raw: List[float] = []
    guids: List[Optional[str]] = []
    labels: List[Optional[int]] = []

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            target_state = outputs.get("target_state")
            source_state = outputs.get("source_state")
            kld_per_t = outputs.get("kld_per_t")
            if target_state is None or source_state is None or kld_per_t is None:
                raise RuntimeError(
                    "model emitted no 'target_state'/'source_state'/'kld_per_t'; the CMI "
                    "comparison needs the encoder states and the raw per-step KL."
                )

            y_plus = runner.build_future_target(batch)  # (B, T-H, H, C_y)
            T = int(target_state.shape[1])
            device = target_state.device
            mask = _anchor_support_mask(runner, T, device)  # (T,)
            denom = float(mask.sum().item())
            if denom <= 0.0:
                mask = torch.ones(T, device=device, dtype=torch.float32)
                denom = float(T)

            m_state = mask.view(1, T, 1)
            # Anchor-mean of the encoder states over the support window.
            u_avg = (source_state * m_state).sum(dim=1) / denom  # (B, d_u)
            c_avg = (target_state * m_state).sum(dim=1) / denom  # (B, d_c)
            # Future-target summary: mean over the anchor axis of Y_plus, then flatten (H, C).
            warmup, t_valid = runner.valid_anchor_range(T)
            y_slice = y_plus[:, warmup:t_valid]  # (B, n_anchor, H, C_y)
            if y_slice.shape[1] == 0:
                y_slice = y_plus
            y_avg = y_slice.mean(dim=1)  # (B, H, C_y)
            y_flat = y_avg.reshape(y_avg.shape[0], -1)  # (B, H*C_y)
            # Per-sample raw K over the anchor support.
            k_b = (kld_per_t * mask.view(1, T)).sum(dim=1) / denom  # (B,)

            u_np = _optional_numpy(u_avg)
            c_np = _optional_numpy(c_avg)
            y_np = _optional_numpy(y_flat)
            k_np = _optional_numpy(k_b)
            for idx in range(int(target_state.shape[0])):
                u_rows.append(u_np[idx])
                c_rows.append(c_np[idx])
                y_rows.append(y_np[idx])
                k_raw.append(float(k_np[idx]))
                guids.append(_extract_guid(batch, idx))
                labels.append(_extract_label(batch, idx))

    n = len(k_raw)
    logger.info("cmi_comparison: collected {} per-sample (u, y, c, K_raw) triples", n)
    return {
        "u": np.asarray(u_rows, dtype=np.float32) if n else np.zeros((0, 0), np.float32),
        "y": np.asarray(y_rows, dtype=np.float32) if n else np.zeros((0, 0), np.float32),
        "c": np.asarray(c_rows, dtype=np.float32) if n else np.zeros((0, 0), np.float32),
        "k_raw": np.asarray(k_raw, dtype=np.float64),
        "guids": np.asarray(guids, dtype=object),
        "labels": np.asarray(labels, dtype=object),
        "n_samples": n,
    }


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
            batch_size = int(batch.fhr.size(0))

            outputs = runner.forward(batch)
            x_plus = runner.build_future_target(batch)      # (B, T_valid, H, R)

            fcst = compute_raw_forecast_metrics(
                outputs["mu_full"], x_plus, runner.warmup_steps, runner.horizon
            )
            uplift = compute_uplift_metrics(
                outputs["mu_full"],
                outputs["mu_base"],
                x_plus,
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
            kld_agg_sample = compute_kld_aggregates_per_sample(
                outputs, runner.warmup_steps
            )

            mu_full_np = outputs["mu_full"].detach().cpu().numpy()
            mu_base_np = outputs["mu_base"].detach().cpu().numpy()
            # Decoder observation log-variances (G7). Present in every forward, but no
            # collector read them before Sprint 5; they turn the point forecast into a
            # predictive distribution.
            logvar_full_np = _optional_numpy(outputs.get("logvar_full"))
            logvar_base_np = _optional_numpy(outputs.get("logvar_base"))
            delta_np = outputs["delta_mu_src"].detach().cpu().numpy()
            z_np = outputs["z"].detach().cpu().numpy()
            attn_np = outputs["attn_weights"].detach().cpu().numpy()
            te_lag_np = outputs["te_lag_map"].detach().cpu().numpy()
            kld_t_np = outputs["kld_per_t"].detach().cpu().numpy()
            kld_agg_t = compute_kld_aggregate_tensors(outputs, runner.warmup_steps)
            kld_sum_t_np = (
                kld_agg_t["kld_sum_t"].detach().cpu().numpy()
                if kld_agg_t is not None else kld_t_np
            )
            kld_l2_t_np = (
                kld_agg_t["kld_l2_t"].detach().cpu().numpy()
                if kld_agg_t is not None else kld_t_np
            )
            y_plus_np = x_plus.detach().cpu().numpy()  # raw future target (B, T_valid, H, R)

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
                    "logvar_full": None if logvar_full_np is None else logvar_full_np[idx],
                    "logvar_base": None if logvar_base_np is None else logvar_base_np[idx],
                    "delta_src": delta_np[idx],
                    "y_plus": y_plus_np[idx],
                    "z": z_np[idx],
                    "attn": attn_np[idx],
                    "te_lag": te_lag_np[idx],
                    "kld_t": kld_t_np[idx],
                    "kld_sum_t": kld_sum_t_np[idx],
                    "kld_l2_t": kld_l2_t_np[idx],
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
                    # Synthetic-TE provenance (S7-T06): present only for synthetic v2
                    # batches; ``None`` for real CTG / HDF5 batches (additive + guarded,
                    # so the real-data path is behaviourally unchanged).
                    "te_true": _extract_te_true(batch, idx),
                    "te_scat": _extract_scalar_field(batch, "te_scat", idx),
                    "te_raw": _extract_scalar_field(batch, "te_raw", idx),
                    "frac_phi": _extract_scalar_field(batch, "frac_phi", idx),
                    "sample_delay": _extract_delay(batch, idx),
                    "cell_id": _extract_int_field(batch, "cell_id", idx),
                    "true_lag_tt": _extract_array_field(batch, "true_lag_tt", idx),
                    "true_lag_band": _extract_array_field(batch, "true_lag_band", idx),
                    "metrics": {
                        # Raw-forecast headline metrics (no scattering st/ph split); ``feat_*``
                        # names kept as aliases of the raw MSE / R^2 for downstream compatibility.
                        "feat_mse_total": float(fcst["raw_mse"][idx].cpu().item()),
                        "feat_r2_total": float(fcst["raw_r2"][idx].cpu().item()),
                        "raw_mse": float(fcst["raw_mse"][idx].cpu().item()),
                        "raw_vaf": float(fcst["raw_vaf"][idx].cpu().item()),
                        "raw_lowpass_mse": float(fcst["raw_lowpass_mse"][idx].cpu().item()),
                        "base_mse_total": float(uplift["l_base"][idx].cpu().item()),
                        "uplift_abs": float(uplift["uplift_abs"][idx].cpu().item()),
                        "uplift_rel": float(uplift["uplift_rel"][idx].cpu().item()),
                        "residual_ratio": float(usage["residual_ratio"][idx].cpu().item()),
                        "kld_mean": float(kld_sample[idx].cpu().item()),
                        "kld_sum": float(kld_agg_sample["kld_sum"][idx].cpu().item()),
                        "kld_l2": float(kld_agg_sample["kld_l2"][idx].cpu().item()),
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
    *,
    pca_model: Any = None,
    pca_model_top: Any = None,
) -> pd.DataFrame:
    """Collect per-timestep KL trajectory records for every sample.

    Reads ``outputs["kld_per_t"]`` directly (the model's TE analysis head
    produces this as ``sum_d KL(q || p)``), so no recomputation is needed.
    Warmup timesteps are dropped by the model's warmup masking; we skip
    non-finite values defensively.

    The CSV schema is preserved from the legacy pipeline so
    ``TE_Calculated`` modules keep working:
    ``[guid, epoch, hours_before, label, timestep, kld_mean, latent_0 ... latent_{d_z-1}]``.

    When ``pca_model`` is supplied (an sklearn PCA fitted on per-time
    per-dim KL via :func:`fit_pca_kld_per_dim`), the closed-form per-dim
    KL is recomputed on the fly and projected through the model. The
    resulting per-time scores land in extra columns ``kld_pc1_t``,
    ``kld_pc2_t``, ``kld_pc3_t`` (or as many components as the model
    has). These typically reflect the **contrast-selected** PCs (rows
    of ``components.npy`` indexed by ``selection.json``).

    When ``pca_model_top`` is supplied, the per-dim KL is projected
    through that second model and emitted under
    ``kld_pc_top1_t``, ``kld_pc_top2_t``, ... — used for the
    "first-N-by-eigenvalue" trajectory view (orthogonal to the
    contrast-selection view).

    Both projections share a single inference pass; pass either or
    both, or neither.

    Args:
        runner: :class:`TestRunner` with a loaded model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).
        pca_model: Optional fitted sklearn ``PCA`` whose components live
            in the per-dim-KL space ``(n_components, d_z)``. When None,
            no ``kld_pc{i}_t`` columns are emitted.
        pca_model_top: Optional second fitted PCA (typically the first
            N PCs by eigenvalue). When set, emits ``kld_pc_top{i}_t``
            columns alongside the selected-PC columns.

    Returns:
        DataFrame with per-(sample, timestep) rows.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    n_pcs = 0
    if pca_model is not None:
        n_pcs = int(getattr(pca_model, "n_components_", 0))
    n_pcs_top = 0
    if pca_model_top is not None:
        n_pcs_top = int(getattr(pca_model_top, "n_components_", 0))

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr.size(0))

            outputs = runner.forward(batch)
            kld_t = outputs.get("kld_per_t")
            latent = outputs.get("z")
            if kld_t is None:
                continue
            kld_agg_t = compute_kld_aggregate_tensors(outputs, runner.warmup_steps)

            # Mask warmup region with NaN so we can skip it below.
            kld_t_f = kld_t.detach().to(torch.float32).clone()
            warmup = int(runner.warmup_steps)
            if warmup > 0 and kld_t_f.size(1) > warmup:
                kld_t_f[:, :warmup] = float("nan")

            T = int(kld_t_f.size(1))
            kld_mean_t_f = None
            kld_sum_t_f = None
            kld_l2_t_f = None
            if kld_agg_t is not None:
                kld_mean_t_f = kld_agg_t["kld_mean_t"].detach().to(torch.float32).clone()
                kld_sum_t_f = kld_agg_t["kld_sum_t"].detach().to(torch.float32).clone()
                kld_l2_t_f = kld_agg_t["kld_l2_t"].detach().to(torch.float32).clone()
                if warmup > 0 and T > warmup:
                    kld_mean_t_f[:, :warmup] = float("nan")
                    kld_sum_t_f[:, :warmup] = float("nan")
                    kld_l2_t_f[:, :warmup] = float("nan")

            # Optional: per-time per-dim KL projected through PCA(s).
            pc_traj_np: Optional[np.ndarray] = None
            pc_traj_top_np: Optional[np.ndarray] = None
            if (n_pcs > 0 or n_pcs_top > 0) and all(
                k in outputs
                for k in ("mu_prior", "logvar_prior", "mu_post", "logvar_post")
            ):
                mu_p = outputs["mu_prior"]
                lv_p = outputs["logvar_prior"]
                mu_q = outputs["mu_post"]
                lv_q = outputs["logvar_post"]
                kld_per_dim_t = 0.5 * (
                    lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0
                )
                arr = kld_per_dim_t.detach().to(torch.float32).cpu().numpy()
                if warmup > 0 and warmup < T:
                    arr = arr.copy()
                    arr[:, :warmup, :] = np.nan
                if n_pcs > 0:
                    pc_traj_np = project_kld_per_dim(arr, pca_model)
                if n_pcs_top > 0:
                    pc_traj_top_np = project_kld_per_dim(arr, pca_model_top)

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)
                hours_before = -epoch / 3600.0 if epoch is not None else None

                kld_vals = kld_t_f[idx].cpu().numpy()
                kld_dim_mean_vals = (
                    kld_mean_t_f[idx].cpu().numpy()
                    if kld_mean_t_f is not None else None
                )
                kld_sum_vals = (
                    kld_sum_t_f[idx].cpu().numpy()
                    if kld_sum_t_f is not None else None
                )
                kld_l2_vals = (
                    kld_l2_t_f[idx].cpu().numpy()
                    if kld_l2_t_f is not None else None
                )
                latent_vals = latent[idx].cpu().numpy() if latent is not None else None
                pc_vals = pc_traj_np[idx] if pc_traj_np is not None else None
                pc_top_vals = pc_traj_top_np[idx] if pc_traj_top_np is not None else None

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
                    if kld_dim_mean_vals is not None:
                        record["kld_dim_mean_t"] = float(kld_dim_mean_vals[t])
                    if kld_sum_vals is not None:
                        record["kld_sum_t"] = float(kld_sum_vals[t])
                    if kld_l2_vals is not None:
                        record["kld_l2_t"] = float(kld_l2_vals[t])
                    if latent_vals is not None:
                        for d in range(latent_vals.shape[1]):
                            record[f"latent_{d}"] = float(latent_vals[t, d])
                    if pc_vals is not None:
                        for k in range(pc_vals.shape[1]):
                            record[f"kld_pc{k + 1}_t"] = float(pc_vals[t, k])
                    if pc_top_vals is not None:
                        for k in range(pc_top_vals.shape[1]):
                            record[f"kld_pc_top{k + 1}_t"] = float(pc_top_vals[t, k])
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
            batch_size = int(batch.fhr.size(0))

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
            batch_size = int(batch.fhr.size(0))

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
    r"""Collect per-(sample, horizon step) raw-forecast error.

    For each sample we slice the raw forecast to the valid anchor range $[w, T-H)$ and reduce the
    squared error over anchors and the $R$ raw substeps, giving one row per horizon step. There is
    **no** scattering / phase channel split (the raw target's last axis is a block of raw substeps,
    not feature channels), so the row carries only ``mse_step``.

    Args:
        runner: :class:`TestRunner` with a loaded raw model.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process (None = all).

    Returns:
        DataFrame with columns ``[guid, epoch, label, h, mse_step]``.
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = int(batch.fhr.size(0))

            outputs = runner.forward(batch)
            x_plus = runner.build_future_target(batch)      # (B, T_valid, H, R)

            mu_full = outputs["mu_full"]                    # (B, T, H, R)
            T, H_d = mu_full.shape[1], mu_full.shape[2]

            warmup = int(runner.warmup_steps)
            T_valid = max(T - int(H_d), 0)
            start = max(0, min(warmup, T_valid))

            mu_v = mu_full[:, start:T_valid, :, :]
            x_v = x_plus[:, start:T_valid, :, :]
            if mu_v.numel() == 0:
                continue

            diff_sq = (mu_v - x_v).pow(2)                   # (B, T_v, H_d, R)
            mse_step = diff_sq.mean(dim=(1, 3))             # (B, H_d)
            mse_step_np = mse_step.detach().cpu().numpy()

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
                    })
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)
