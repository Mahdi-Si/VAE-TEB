"""Per-sample KLD and lag-attention diagnostic figures.

This analysis emits **two multi-panel figures per sample**:

- ``<guid>_<epoch>_signals_kld.pdf`` — raw FHR / UP traces, FHR and UP
  scattering transforms, per-dimension KLD traces (one subplot per
  latent dim), and the mean ± std of KLD aggregated across latent
  dimensions.
- ``<guid>_<epoch>_lag_attention.pdf`` — raw traces, head-averaged lag
  attention as a ``(L, T)`` heatmap with the lag y-axis expressed in
  minutes (via ``decim / fs_raw``), TE lag attribution on the same
  minute-by-minute grid, an attention-analysis row (argmax lag +
  head-averaged entropy) and a time-averaged attention-mass bar chart
  per lag bin.

Both figures share a common physical-time x-axis in **minutes**
(derived from the raw sampling rate and decimation factor), so the
raw signal row and every decimated heatmap row line up column for
column.

Example:
    >>> run_kld_lag_diagnostics(runner, loader, max_samples=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
    _extract_te_true,
    denormalize_signal,
    resolve_fhr_up_denorm_stats,
)
from model.vae_teb_prediction.testing.metrics import project_kld_per_dim
from model.vae_teb_prediction.testing.plot_single_samples import (
    plot_sample_lag_attention,
    plot_sample_signals_kld,
    plot_sample_signals_kld_pca,
)


def _load_pca_artifacts_for_plotting(pca_dir: Path):
    """Load the PCA fit written by :func:`collect_metrics` (if present).

    Returns a tuple ``(pca_model, ev_ratio)`` or ``(None, None)`` when
    the artifacts are missing or unreadable. The returned ``pca_model``
    is a ``sklearn.decomposition.PCA`` instance reconstructed from the
    persisted ``components.npy`` + ``mean.npy`` so downstream callers
    can use :func:`project_kld_per_dim` without re-fitting.
    """
    ev_path = pca_dir / "ev_ratio.json"
    comp_path = pca_dir / "components.npy"
    mean_path = pca_dir / "mean.npy"
    if not ev_path.exists() or not comp_path.exists():
        return None, None
    try:
        with open(ev_path) as fh:
            ev = json.load(fh)
        components = np.load(comp_path)
        mean = np.load(mean_path) if mean_path.exists() else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"kld_lag_diagnostics: failed to read PCA artifacts: {exc}")
        return None, None

    try:
        from sklearn.decomposition import PCA
    except Exception:
        logger.warning(
            "kld_lag_diagnostics: sklearn unavailable — PCA variant skipped"
        )
        return None, None

    components = np.asarray(components, dtype=np.float32)
    n_components, d_z = components.shape
    pca = PCA(n_components=n_components)
    pca.components_ = components
    pca.mean_ = (
        np.asarray(mean, dtype=np.float32)
        if mean is not None
        else np.zeros(d_z, dtype=np.float32)
    )
    pca.n_components_ = n_components
    pca.n_features_in_ = d_z
    ev_ratio = np.asarray(
        ev.get("explained_variance_ratio", []), dtype=np.float32
    )
    pca.explained_variance_ratio_ = ev_ratio
    pca.explained_variance_ = ev_ratio.copy()
    return pca, ev_ratio


# Conversion factors: one decimated step corresponds to ``decim / fs_raw``
# raw seconds. For the defaults (4 Hz sampling and stride-16 decimation)
# that is 4 seconds per decimated step, so the lag window spans
# ``(max_lag + 1) × 4 / 60`` minutes.
_DEFAULT_FS_RAW = 4.0
_DEFAULT_DECIM = 16


def run_kld_lag_diagnostics(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 10,
    output_dir: Optional[Path] = None,
    *,
    fs_raw: float = _DEFAULT_FS_RAW,
    decim: int = _DEFAULT_DECIM,
) -> Dict[str, Any]:
    """Emit per-sample KLD and lag-attention diagnostic PDFs.

    For every sample this runs :class:`SeqVaeLagAttnV1` once, pulls the
    scattering features + forward-dict outputs, computes the closed-form
    per-dim KL, and hands off to the plotters in
    :mod:`plot_single_samples`.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader yielding the new HDF5 batch schema
            (``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``, optionally
            ``fhr``/``up``).
        max_samples: Number of samples to diagnose (None → all).
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("kld_lag_diag")``).
        fs_raw: Raw sampling rate in Hz (default 4.0).
        decim: Raw→decimated stride (default 16).

    Returns:
        Dict with ``n_plotted, output_dir, summary_csv, samples``,
        where ``samples`` is a list of per-sample metadata dicts that
        also carry basic KLD summary statistics.
    """
    if max_samples is not None and max_samples <= 0:
        logger.info("kld_lag_diagnostics: skipped (max_samples <= 0)")
        return {"n_plotted": 0, "output_dir": None, "samples": []}

    if output_dir is None:
        output_dir = runner.ensure_dir("kld_lag_diag")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    processed = 0

    # Stats for reversing fhr/up z-score normalisation before plotting.
    # The HDF5 dataloader normalises the raw traces; inverting that here
    # lets the plots show the actual physiological signal (bpm / mmHg).
    denorm_stats = resolve_fhr_up_denorm_stats(loader)
    fhr_stats = denorm_stats.get("fhr")
    up_stats = denorm_stats.get("up")

    # Optional PCA fit produced earlier in the pipeline by
    # ``collect_metrics``. When available, emit a companion per-sample
    # figure that plots the top-k PCA scores of the KL trajectory in
    # place of the 24 per-dim traces.
    pca_dir = Path(runner.output_dir) / "pca_kld"
    pca_model, pca_ev_ratio = _load_pca_artifacts_for_plotting(pca_dir)
    if pca_model is None:
        logger.info(
            "kld_lag_diagnostics: PCA artifacts not found in "
            f"{pca_dir} — skipping per-sample KLD-PCA variant. Run the "
            "histogram step first so collect_metrics writes the fit."
        )
    else:
        ev_str = (
            [round(float(x), 4) for x in np.asarray(pca_ev_ratio).tolist()]
            if pca_ev_ratio is not None
            else []
        )
        logger.info(
            f"kld_lag_diagnostics: PCA fit loaded "
            f"(n_components={pca_model.n_components_}, ev_ratio={ev_str})"
        )

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)

            # Normalised scattering features go straight to plotting.
            fhr_st_np = batch.fhr_st.detach().cpu().numpy()  # (B, T, 43)
            fhr_ph_np = None
            if hasattr(batch, "fhr_ph") and isinstance(batch.fhr_ph, torch.Tensor):
                fhr_ph_np = batch.fhr_ph.detach().cpu().numpy()
            up_st_np = None
            if hasattr(batch, "up_st") and isinstance(batch.up_st, torch.Tensor):
                up_st_np = batch.up_st.detach().cpu().numpy()
            up_ph_np = None
            if hasattr(batch, "up_ph") and isinstance(batch.up_ph, torch.Tensor):
                up_ph_np = batch.up_ph.detach().cpu().numpy()

            # Synthetic-data ground truth for the lag-attention overlay: the
            # per-step true lag d_{i,t} (B, T) and the dataset-level informative
            # band. Both are absent for real (HDF5) batches -> overlay skipped.
            true_lag_tt_np = None
            if hasattr(batch, "true_lag_tt") and isinstance(batch.true_lag_tt, torch.Tensor):
                true_lag_tt_np = batch.true_lag_tt.detach().cpu().numpy()
            true_lag_band_np = None
            if hasattr(batch, "true_lag_band") and isinstance(batch.true_lag_band, torch.Tensor):
                true_lag_band_np = batch.true_lag_band.detach().cpu().numpy()

            # Raw fhr / up traces: invert the z-score normalisation the
            # dataloader applied so the plots show physiological units.
            fhr_np = None
            if hasattr(batch, "fhr") and isinstance(batch.fhr, torch.Tensor):
                fhr_np = batch.fhr.detach().cpu().numpy()
                fhr_np = denormalize_signal(fhr_np, fhr_stats)
            up_np = None
            if hasattr(batch, "up") and isinstance(batch.up, torch.Tensor):
                up_np = batch.up.detach().cpu().numpy()
                up_np = denormalize_signal(up_np, up_stats)

            # Closed-form KL per latent dim per timestep.
            mu_prior = outputs["mu_prior"]
            logvar_prior = outputs["logvar_prior"]
            mu_post = outputs["mu_post"]
            logvar_post = outputs["logvar_post"]
            kld_per_dim_t = 0.5 * (
                logvar_prior
                - logvar_post
                + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
                - 1.0
            )
            kld_per_dim_np = kld_per_dim_t.detach().cpu().numpy()  # (B, T, d_z)

            attn_np = outputs["attn_weights"].detach().cpu().numpy()  # (B, T, M, L)
            te_lag_np = outputs["te_lag_map"].detach().cpu().numpy()  # (B, T, L)

            batch_size = int(fhr_st_np.shape[0])
            for idx in range(batch_size):
                if max_samples is not None and processed >= max_samples:
                    break

                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)
                # Per-sample analytic true TE (None for real / HDF5 batches).
                true_te = _extract_te_true(batch, idx)

                # Scalar TE surrogate K-bar for the title bar: sum over latent
                # dims (the ``mixed_eval`` reduction), then mean over the
                # warmup-trimmed window [warm:T) -- the same window as the
                # ``kld_mean_all`` summary stat below (no per-sample d_max floor /
                # T-H bound, so it stays portable to real batches). ``None`` if
                # the window is empty.
                warm = int(runner.warmup_steps)
                kld_one = kld_per_dim_np[idx]                      # (T, d_z)
                kld_one_tail = (
                    kld_one[warm:] if 0 < warm < kld_one.shape[0] else kld_one
                )
                kbar_sample = (
                    float(kld_one_tail.sum(axis=-1).mean())
                    if kld_one_tail.size
                    else None
                )

                safe_guid = (
                    str(guid).replace("/", "_")
                    if guid is not None
                    else f"sample_{processed}"
                )
                epoch_str = (
                    f"ep{int(epoch):+d}" if epoch is not None else f"idx{processed}"
                )
                stem = f"{safe_guid}_{epoch_str}"

                signals_out = output_dir / f"{stem}_signals_kld.pdf"
                pca_out = output_dir / f"{stem}_signals_kld_pca.pdf"
                lag_out = output_dir / f"{stem}_lag_attention.pdf"

                # -- Figure 1: signals + scattering + KLD per dim --
                try:
                    plot_sample_signals_kld(
                        fhr=fhr_np[idx] if fhr_np is not None else None,
                        up=up_np[idx] if up_np is not None else None,
                        fhr_st=fhr_st_np[idx],
                        fhr_ph=fhr_ph_np[idx] if fhr_ph_np is not None else None,
                        up_st=up_st_np[idx] if up_st_np is not None else None,
                        up_ph=up_ph_np[idx] if up_ph_np is not None else None,
                        kld_per_dim=kld_per_dim_np[idx],
                        warmup=int(runner.warmup_steps),
                        out_path=signals_out,
                        guid=guid,
                        epoch=epoch,
                        label=label,
                        kld_value=kbar_sample,
                        true_te=true_te,
                        fs_raw=fs_raw,
                        decim=decim,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"kld_lag_diagnostics: signals_kld failed for {guid}: {exc}"
                    )
                    signals_out = None  # type: ignore[assignment]

                # -- Figure 1b (PCA variant): same layout with the
                # per-dim KL trace block replaced by top-k PCA scores
                # of the KL trajectory. Only emitted when the PCA fit
                # from collect_metrics is available on disk.
                if pca_model is not None:
                    try:
                        kld_pcs_one = project_kld_per_dim(
                            kld_per_dim_np[idx:idx + 1], pca_model
                        )[0]   # (T, n_components)
                        plot_sample_signals_kld_pca(
                            fhr=fhr_np[idx] if fhr_np is not None else None,
                            up=up_np[idx] if up_np is not None else None,
                            fhr_st=fhr_st_np[idx],
                            fhr_ph=fhr_ph_np[idx] if fhr_ph_np is not None else None,
                            up_st=up_st_np[idx] if up_st_np is not None else None,
                            up_ph=up_ph_np[idx] if up_ph_np is not None else None,
                            kld_pcs=kld_pcs_one,
                            explained_variance_ratio=pca_ev_ratio,
                            warmup=int(runner.warmup_steps),
                            out_path=pca_out,
                            guid=guid,
                            epoch=epoch,
                            label=label,
                            kld_value=kbar_sample,
                            true_te=true_te,
                            fs_raw=fs_raw,
                            decim=decim,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            f"kld_lag_diagnostics: signals_kld_pca failed for {guid}: {exc}"
                        )
                        pca_out = None  # type: ignore[assignment]
                else:
                    pca_out = None  # type: ignore[assignment]

                # -- Figure 2: attention + TE lag + lag analysis --
                try:
                    plot_sample_lag_attention(
                        fhr=fhr_np[idx] if fhr_np is not None else None,
                        up=up_np[idx] if up_np is not None else None,
                        attn_weights=attn_np[idx],
                        te_lag_map=te_lag_np[idx],
                        warmup=int(runner.warmup_steps),
                        out_path=lag_out,
                        fhr_st=fhr_st_np[idx],
                        fhr_ph=fhr_ph_np[idx] if fhr_ph_np is not None else None,
                        up_st=up_st_np[idx] if up_st_np is not None else None,
                        up_ph=up_ph_np[idx] if up_ph_np is not None else None,
                        true_lag_tt=(
                            true_lag_tt_np[idx] if true_lag_tt_np is not None else None
                        ),
                        true_lag_band=(
                            true_lag_band_np[idx] if true_lag_band_np is not None else None
                        ),
                        guid=guid,
                        epoch=epoch,
                        label=label,
                        kld_value=kbar_sample,
                        true_te=true_te,
                        fs_raw=fs_raw,
                        decim=decim,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"kld_lag_diagnostics: lag_attention failed for {guid}: {exc}"
                    )
                    lag_out = None  # type: ignore[assignment]

                # Summary metadata for the CSV (per-sample KLD summary
                # stats restricted to the valid anchor range).
                warm = int(runner.warmup_steps)
                kld_tail = kld_per_dim_np[idx]
                if warm > 0 and warm < kld_tail.shape[0]:
                    kld_tail = kld_tail[warm:]
                mean_per_dim = kld_tail.mean(axis=0) if kld_tail.size else None

                records.append({
                    "guid": guid,
                    "epoch": epoch,
                    "label": label,
                    "te_true": true_te,
                    "kbar_sum": kbar_sample,
                    "signals_kld_pdf": str(signals_out.name) if signals_out else None,
                    "signals_kld_pca_pdf": str(pca_out.name) if pca_out else None,
                    "lag_attention_pdf": str(lag_out.name) if lag_out else None,
                    "kld_mean_all": (
                        float(mean_per_dim.mean()) if mean_per_dim is not None else None
                    ),
                    "kld_std_all": (
                        float(mean_per_dim.std()) if mean_per_dim is not None else None
                    ),
                    "kld_argmax_dim": (
                        int(mean_per_dim.argmax()) if mean_per_dim is not None else None
                    ),
                })
                processed += 1

            if max_samples is not None and processed >= max_samples:
                break

    summary_df = pd.DataFrame(records)
    summary_csv = output_dir / "sample_kld_lag_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    logger.info(
        f"kld_lag_diagnostics: plotted {processed} samples → {output_dir}"
    )
    return {
        "n_plotted": int(processed),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "samples": records,
    }
