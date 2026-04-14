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

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
)
from model.vae_teb_prediction.testing.plot_single_samples import (
    plot_sample_lag_attention,
    plot_sample_signals_kld,
)


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

            fhr_np = None
            if hasattr(batch, "fhr") and isinstance(batch.fhr, torch.Tensor):
                fhr_np = batch.fhr.detach().cpu().numpy()
            up_np = None
            if hasattr(batch, "up") and isinstance(batch.up, torch.Tensor):
                up_np = batch.up.detach().cpu().numpy()

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
                        fs_raw=fs_raw,
                        decim=decim,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"kld_lag_diagnostics: signals_kld failed for {guid}: {exc}"
                    )
                    signals_out = None  # type: ignore[assignment]

                # -- Figure 2: attention + TE lag + lag analysis --
                try:
                    plot_sample_lag_attention(
                        fhr=fhr_np[idx] if fhr_np is not None else None,
                        up=up_np[idx] if up_np is not None else None,
                        attn_weights=attn_np[idx],
                        te_lag_map=te_lag_np[idx],
                        warmup=int(runner.warmup_steps),
                        out_path=lag_out,
                        guid=guid,
                        epoch=epoch,
                        label=label,
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
                    "signals_kld_pdf": str(signals_out.name) if signals_out else None,
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
