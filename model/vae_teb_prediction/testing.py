from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import yaml
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from trainer import GraphModelVaeTebSmallTrainer
from train.graph_models_utils import denormalize_signal_data
from utils.plot_utils import (
    plot_metrics_histograms,
    plot_model_analysis,
    plot_single_prediction_windows,
    plot_vae_reconstruction,
)


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_test_dataloader(config: Dict):
    dataset_cfg = config.get("dataset_config", {})
    dataloader_cfg = dataset_cfg.get("dataloader_config", {})
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {}) or {}
    test_files = dataset_cfg.get("vae_test_datasets", [])
    if not test_files:
        raise ValueError("dataset_config.vae_test_datasets is empty.")
    stats_path = dataset_cfg.get("stat_path")
    if stats_path is None:
        raise ValueError("dataset_config.stat_path must be provided.")
    normalize_fields = dataloader_cfg.get("normalize_fields")
    batch_size = config.get("general_config", {}).get("batch_size", {}).get("test", 1)
    return create_optimized_dataloader(
        hdf5_files=test_files,
        batch_size=batch_size,
        num_workers=dataloader_cfg.get("num_workers", 0),
        shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        pin_memory=torch.cuda.is_available(),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )


class GraphModelVaeTebSmallTester(GraphModelVaeTebSmallTrainer):
    """Lightweight test harness focused on basic reconstruction diagnostics."""

    def run_histogram_test(self, test_loader, *, num_samples: Optional[int] = None, max_samples: Optional[int] = None) -> None:
        """Evaluate VAF, MSE, SNR, and KLD histograms on the provided loader."""
        if self.pytorch_model is None:
            self.create_model()
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        output_dir = Path(self.test_results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vaf_values: List[float] = []
        mse_values: List[float] = []
        snr_values: List[float] = []
        kld_values: List[float] = []

        limit_primary = None if num_samples is None else max(0, int(num_samples))
        limit_secondary = None if max_samples is None else max(0, int(max_samples))
        if limit_primary is not None and limit_secondary is not None:
            total_limit = min(limit_primary, limit_secondary)
        else:
            total_limit = limit_primary if limit_primary is not None else limit_secondary
        processed = 0
        with torch.inference_mode():
            for batch in test_loader:
                if total_limit is not None and processed >= total_limit:
                    break

                y_st = batch.fhr_st.to(device)
                y_ph = batch.fhr_ph.to(device)
                x_ph = batch.fhr_up_ph.to(device)
                y_raw = batch.fhr.to(device)

                forward_outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                recon_segments = forward_outputs.get("mu_pr")
                logvar_segments = forward_outputs.get("logvar_pr")
                recon, _, valid_mask = self._average_raw_prediction_segments(
                    model,
                    recon_segments,
                    logvar_segments=logvar_segments,
                )
                if recon is None:
                    logger.warning("Forward pass did not return usable raw predictions; skipping batch.")
                    continue
                if recon.shape != y_raw.shape:
                    logger.warning(
                        "Predictions have shape %s but targets are %s; skipping batch.",
                        tuple(recon.shape),
                        tuple(y_raw.shape),
                    )
                    continue

                metrics = self._compute_basic_metrics(y_raw, recon, valid_mask)
                if metrics is None:
                    logger.warning("Metric computation failed for batch; skipping.")
                    continue
                kld_batch = self._kld_from_forward(forward_outputs, device).to(recon.device)

                if valid_mask is not None:
                    sample_mask = valid_mask.view(valid_mask.size(0), -1).any(dim=1)
                else:
                    sample_mask = torch.ones(recon.size(0), dtype=torch.bool, device=recon.device)
                if not torch.any(sample_mask):
                    logger.warning("Predictions did not cover any raw samples in this batch; skipping.")
                    continue

                batch_vaf = metrics["vaf"][sample_mask].detach().cpu().tolist()
                batch_mse = metrics["mse"][sample_mask].detach().cpu().tolist()
                batch_snr = metrics["snr"][sample_mask].detach().cpu().tolist()
                batch_kld = kld_batch[sample_mask].detach().cpu().tolist()

                if total_limit is not None:
                    remaining = total_limit - processed
                    if remaining <= 0:
                        break
                    batch_vaf = batch_vaf[:remaining]
                    batch_mse = batch_mse[:remaining]
                    batch_snr = batch_snr[:remaining]
                    batch_kld = batch_kld[:remaining]

                if not batch_vaf:
                    continue

                vaf_values.extend(batch_vaf)
                mse_values.extend(batch_mse)
                snr_values.extend(batch_snr)
                kld_values.extend(batch_kld)
                processed += len(batch_vaf)

        if not vaf_values:
            logger.error("No metrics were collected; check the dataloader and model outputs.")
            return

        def _finite_only(values: List[float]) -> List[float]:
            if not values:
                return []
            arr = np.asarray(values, dtype=float)
            mask = np.isfinite(arr)
            return arr[mask].tolist()

        vaf_values = _finite_only(vaf_values)
        mse_values = _finite_only(mse_values)
        snr_values = _finite_only(snr_values)
        kld_values = _finite_only(kld_values)

        if not any((vaf_values, mse_values, snr_values, kld_values)):
            logger.error("All histogram metrics are NaN/invalid; cannot plot.")
            return

        def _safe_mean(values: List[float]) -> float:
            return float(np.mean(values)) if values else float("nan")

        logger.info(
            "Histogram metrics collected for %d samples (VAF mean=%.4f, MSE mean=%.6f, SNR mean=%.2f dB, KLD mean=%.6f)",
            len(vaf_values),
            _safe_mean(vaf_values),
            _safe_mean(mse_values),
            _safe_mean(snr_values),
            _safe_mean(kld_values),
        )

        plot_metrics_histograms(vaf_values, mse_values, snr_values, kld_values, output_dir)

    def run_latent_distribution(self, test_loader, *, num_samples: int = 500) -> None:
        """Plot latent-dimension distributions aggregated across samples."""
        if num_samples <= 0:
            logger.info("Latent distribution skipped (num_samples <= 0)")
            return
        if self.pytorch_model is None:
            self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model unavailable; cannot compute latent distributions.")
            return
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        latent_chunks: List[np.ndarray] = []
        collected = 0
        with torch.inference_mode():
            for batch in test_loader:
                outputs = model(y_st=batch.fhr_st.to(device), y_ph=batch.fhr_ph.to(device), x_ph=batch.fhr_up_ph.to(device))
                latent = outputs.get("z")
                if latent is None:
                    continue
                latent_np = latent.detach().cpu().numpy()  # (B, T, D)
                batch_size = latent_np.shape[0]
                for i in range(batch_size):
                    if collected >= num_samples:
                        break
                    latent_chunks.append(latent_np[i].reshape(-1, latent_np.shape[-1]))
                    collected += 1
                if collected >= num_samples:
                    break
        if not latent_chunks:
            logger.warning("No latent samples collected for distribution plot.")
            return
        combined = np.concatenate(latent_chunks, axis=0)
        latent_dim = combined.shape[1]
        cols = 4
        rows = math.ceil(latent_dim / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.atleast_2d(axes)
        for idx in range(rows * cols):
            row, col = divmod(idx, cols)
            ax = axes[row, col]
            if idx < latent_dim:
                ax.hist(combined[:, idx], bins=50, color="#4C72B0", alpha=0.8)
                ax.set_title(f"z[{idx}]")
            else:
                ax.axis("off")
        fig.tight_layout()
        out_dir = Path(self.test_results_dir) / "latent_distribution"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "latent_histograms.png", dpi=200)
        plt.close(fig)
        logger.info("Latent distribution plot saved to %s", out_dir)

    @staticmethod
    def _extract_guid_epoch(batch, index: int) -> tuple[Optional[str], Optional[float]]:
        guid_attr = getattr(batch, "guid", None)
        epoch_attr = getattr(batch, "epoch", None)
        guid_val: Optional[str] = None
        epoch_val: Optional[float] = None
        if guid_attr is not None:
            try:
                raw_guid = guid_attr[index]
                if isinstance(raw_guid, torch.Tensor):
                    if raw_guid.dtype == torch.int64:
                        raw_guid = int(raw_guid.item())
                    else:
                        raw_guid = raw_guid.item()
                if isinstance(raw_guid, bytes):
                    raw_guid = raw_guid.decode("utf-8")
                guid_val = str(raw_guid)
            except Exception:
                guid_val = None
        if epoch_attr is not None:
            try:
                raw_epoch = epoch_attr[index]
                if isinstance(raw_epoch, torch.Tensor):
                    epoch_val = float(raw_epoch.item())
                else:
                    epoch_val = float(raw_epoch)
            except Exception:
                epoch_val = None
        return guid_val, epoch_val

    @staticmethod
    def _build_consecutive_pairs(grouped: Dict[str, List[Dict[str, Any]]], target: int) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
        pairs: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        for entries in grouped.values():
            if not entries:
                continue
            entries.sort(key=lambda item: item.get("epoch", 0.0))
            for idx in range(len(entries) - 1):
                pairs.append((entries[idx], entries[idx + 1]))
                if len(pairs) >= target:
                    return pairs
        return pairs

    @staticmethod
    def _kld_tensor_from_forward(outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Return the unreduced KLD tensor for downstream analysis with warmup masking applied."""
        mu_prior = outputs.get("mu_prior")
        logvar_prior = outputs.get("logvar_prior")
        mu_post = outputs.get("mu_post")
        logvar_post = outputs.get("logvar_post")
        if any(t is None for t in (mu_prior, logvar_prior, mu_post, logvar_post)):
            return None
        kld = (
            logvar_prior
            - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
            - 1.0
        )
        kld = 0.5 * kld
        mask = outputs.get("warmup_mask")
        if mask is not None:
            valid = mask
            if valid.dim() == 0:
                valid = valid.view(1, 1, 1)
            if valid.dim() == 1:
                valid = valid.unsqueeze(0).unsqueeze(-1)
            elif valid.dim() == 2:
                valid = valid.unsqueeze(-1)
            valid = valid.to(device=kld.device, dtype=torch.bool)
            if valid.size(0) == 1 and kld.size(0) > 1:
                valid = valid.expand(kld.size(0), -1, -1)
            if valid.size(-1) == 1 and kld.size(-1) > 1:
                valid = valid.expand(-1, -1, kld.size(-1))
            kld = kld.masked_fill(~valid, float("nan"))
        return kld

    @staticmethod
    def _compute_aligned_kld_mean(kld_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute mean KLD over latent dimensions while preserving warmup NaN alignment.

        Args:
            kld_tensor: KLD tensor of shape (B, T, D) where warmup timesteps are NaN

        Returns:
            Mean KLD of shape (B, T) with NaN preserved for warmup timesteps
        """
        # Take mean over latent dimension (dim=-1)
        # nanmean will return NaN if all values in a timestep are NaN (warmup period)
        kld_mean = torch.nanmean(kld_tensor, dim=-1)
        return kld_mean

    @staticmethod
    def _kld_from_forward(outputs: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Compute per-sample KLD scalars from forward outputs."""
        tensor = GraphModelVaeTebSmallTester._kld_tensor_from_forward(outputs)
        if tensor is None:
            fallback_shape = outputs.get("mu_pr")
            batch = 1 if fallback_shape is None else fallback_shape.shape[0]
            return torch.zeros(batch, device=device)
        dims = tuple(range(1, tensor.ndim))
        if not dims:
            return tensor
        return torch.nanmean(tensor, dim=dims)

    @staticmethod
    def _average_raw_prediction_segments(
        model: torch.nn.Module,
        segments: Optional[torch.Tensor],
        *,
        logvar_segments: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Collapse per-timestep prediction windows into a single raw-length sequence.

        Returns averaged predictions, averaged log-variance (if provided), and a boolean
        mask marking positions with valid coverage.
        """
        if segments is None or segments.dim() != 3:
            return segments, logvar_segments, None
        if not hasattr(model, "average_raw_prediction"):
            return segments, logvar_segments, None

        avg_mu = model.average_raw_prediction(segments)
        valid_mask = torch.isfinite(avg_mu)
        avg_mu = torch.nan_to_num(avg_mu, nan=0.0)

        avg_logvar = None
        if logvar_segments is not None:
            avg_var = model.average_raw_prediction(logvar_segments.exp())
            valid_mask = valid_mask & torch.isfinite(avg_var)
            avg_var = torch.nan_to_num(avg_var, nan=1.0)
            avg_logvar = avg_var.clamp_min(1e-12).log()

        return avg_mu, avg_logvar, valid_mask

    @staticmethod
    def _compute_basic_metrics(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return per-sample tensors for MSE, SNR (dB), and VAF."""
        if y_true.shape != y_pred.shape:
            return None
        residual = y_true - y_pred
        dims = tuple(range(1, residual.ndim))
        if not dims:
            return None
        residual = y_true - y_pred
        if valid_mask is not None:
            if valid_mask.shape != y_true.shape:
                raise ValueError("valid_mask must match tensor shapes.")
            weight = valid_mask.to(y_true.dtype)
            denom = weight.sum(dim=dims).clamp_min(1.0)
            mse = ((residual ** 2) * weight).sum(dim=dims) / denom
            signal_power = ((y_true ** 2) * weight).sum(dim=dims) / denom
            noise_power = ((residual ** 2) * weight).sum(dim=dims) / denom
            mean_residual = (residual * weight).sum(dim=dims) / denom
            mean_true = (y_true * weight).sum(dim=dims) / denom
            var_res = (noise_power - mean_residual ** 2).clamp_min(1e-12)
            var_orig = (((y_true ** 2) * weight).sum(dim=dims) / denom - mean_true ** 2).clamp_min(1e-12)
        else:
            mse = (residual ** 2).mean(dim=dims)
            signal_power = (y_true ** 2).mean(dim=dims)
            noise_power = (residual ** 2).mean(dim=dims)
            var_res = residual.var(dim=dims, unbiased=False)
            var_orig = y_true.var(dim=dims, unbiased=False).clamp_min(1e-12)
        snr = torch.where(
            noise_power > 1e-12,
            10.0 * torch.log10(signal_power.clamp_min(1e-12) / noise_power.clamp_min(1e-12)),
            torch.full_like(signal_power, 100.0),
        )
        vaf = (1.0 - (var_res / var_orig)).clamp(0.0, 1.0)
        return {"mse": mse, "snr": snr, "vaf": vaf}

    def run_analysis_and_plot(
        self,
        test_loader,
        *,
        num_samples: int = 10,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Generate qualitative plots for a subset of samples."""
        if num_samples <= 0:
            logger.info("run_analysis_and_plot: num_samples<=0, skipping analysis.")
            return
        if self.pytorch_model is None:
            self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model is unavailable; cannot run analysis.")
            return
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        stats = self._get_normalization_stats(test_loader)
        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        beta_value = float(vae_cfg.get("kld_beta", 1.0))
        out_dir = Path(output_dir or self.test_results_dir) / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        with torch.inference_mode():
            for batch in test_loader:
                batch_size = batch.fhr_st.size(0)
                for idx in range(batch_size):
                    if processed >= num_samples:
                        break
                    y_st = batch.fhr_st[idx : idx + 1].to(device)
                    y_ph = batch.fhr_ph[idx : idx + 1].to(device)
                    x_ph = batch.fhr_up_ph[idx : idx + 1].to(device)
                    y_raw = batch.fhr[idx : idx + 1].to(device)
                    up_raw_tensor = getattr(batch, "up", None)
                    if up_raw_tensor is None:
                        up_raw = torch.zeros_like(y_raw)
                    else:
                        up_raw = up_raw_tensor[idx : idx + 1].to(device)

                    forward_outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                    recon_segments = forward_outputs.get("mu_pr")
                    logvar_segments = forward_outputs.get("logvar_pr")
                    latent = forward_outputs.get("z")
                    linear_output = forward_outputs.get("linear_output")
                    recon, logvar, valid_mask = self._average_raw_prediction_segments(
                        model,
                        recon_segments,
                        logvar_segments=logvar_segments,
                    )
                    if recon is None or latent is None:
                        logger.warning("Forward outputs missing required tensors; skipping sample.")
                        continue

                    loss_dict = model.compute_loss(
                        forward_outputs=forward_outputs,
                        y_st=y_st,
                        y_ph=y_ph,
                        y_raw=y_raw,
                        beta=beta_value,
                    )
                    loss_floats = {}
                    for key, val in loss_dict.items():
                        if isinstance(val, torch.Tensor):
                            scalar = torch.nan_to_num(val.detach()).cpu().item()
                        else:
                            scalar = float(val)
                        loss_floats[key] = scalar

                    kld_tensor = self._kld_tensor_from_forward(forward_outputs)
                    if kld_tensor is None:
                        kld_tensor_np = np.zeros((latent.shape[-1], y_st.shape[1]))
                        kld_mean_np = np.zeros(y_st.shape[1])
                    else:
                        kld_sample = kld_tensor[0]  # (T, D)
                        kld_tensor_np = kld_sample.detach().cpu().numpy().T  # (D, T)
                        # Compute aligned mean - preserves NaN for warmup timesteps
                        kld_mean_np = self._compute_aligned_kld_mean(kld_sample.unsqueeze(0))[0].detach().cpu().numpy()  # (T,)

                    fhr_norm = y_raw[0]
                    up_norm = up_raw[0]
                    fhr_denorm = self._maybe_denormalize(fhr_norm, "fhr", stats)
                    up_denorm = self._maybe_denormalize(up_norm, "up", stats)

                    raw_fhr_norm_np = fhr_norm.detach().cpu().numpy()
                    raw_up_norm_np = up_norm.detach().cpu().numpy()
                    raw_fhr_denorm_np = fhr_denorm.detach().cpu().numpy()
                    raw_up_denorm_np = up_denorm.detach().cpu().numpy()

                    fhr_st_np = y_st[0].detach().cpu().numpy().T
                    fhr_ph_np = y_ph[0].detach().cpu().numpy().T
                    fhr_up_ph_np = x_ph[0].detach().cpu().numpy().T
                    latent_np = latent[0].detach().cpu().numpy().T
                    recon_np = recon[0].detach().cpu().numpy()
                    if logvar is not None:
                        logvar_np = logvar[0].detach().cpu().numpy()
                    else:
                        logvar_np = np.zeros_like(recon_np)
                    if valid_mask is not None:
                        mask_np = valid_mask[0].detach().cpu().numpy().astype(bool)
                        recon_np = np.where(mask_np, recon_np, np.nan)
                        logvar_np = np.where(mask_np, logvar_np, np.nan)

                    recon_st_np, recon_ph_np = self._extract_reconstruction_features(
                        linear_output,
                        st_channels=fhr_st_np.shape[0],
                        ph_channels=fhr_ph_np.shape[0],
                        seq_len=fhr_st_np.shape[1],
                    )

                    plot_model_analysis(
                        output_dir=str(out_dir),
                        raw_fhr=raw_fhr_denorm_np,
                        raw_up=raw_up_denorm_np,
                        fhr_st=fhr_st_np,
                        fhr_ph=fhr_ph_np,
                        fhr_up_ph=fhr_up_ph_np,
                        latent_z=latent_np,
                        reconstructed_fhr_mu=recon_np,
                        reconstructed_fhr_logvar=logvar_np,
                        kld_tensor=kld_tensor_np,
                        kld_mean_over_channels=kld_mean_np,
                        batch_idx=processed,
                        loss_dict=loss_floats,
                        raw_fhr_normalized=raw_fhr_norm_np,
                        raw_up_normalized=raw_up_norm_np,
                    )

                    plot_vae_reconstruction(
                        output_dir=str(out_dir),
                        raw_fhr_unnormalized=raw_fhr_denorm_np,
                        raw_up_unnormalized=raw_up_denorm_np,
                        raw_fhr_normalized=raw_fhr_norm_np,
                        raw_up_normalized=raw_up_norm_np,
                        reconstructed_fhr=recon_np,
                        original_scattering_transform=fhr_st_np,
                        reconstructed_scattering_transform=recon_st_np,
                        original_phase_harmonic=fhr_ph_np,
                        reconstructed_phase_harmonic=recon_ph_np,
                        scattering_channel_data=None,
                        batch_idx=processed,
                        loss_dict=loss_floats,
                    )

                    processed += 1
                if processed >= num_samples:
                    break

        logger.info("Analysis plots saved to %s (samples=%d)", out_dir, processed)

    def run_single_prediction_probe(
        self,
        test_loader,
        *,
        num_samples: int = 0,
        start_index: int = 20,
        step_size: Optional[int] = None,
        windows_per_sample: int = 4,
    ) -> None:
        """Visualize non-overlapping prediction windows without averaging."""
        if num_samples <= 0 or windows_per_sample <= 0:
            logger.info("Single prediction probe skipped (num_samples<=0 or windows<=0).")
            return
        if self.pytorch_model is None:
            self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model unavailable; cannot run single prediction probe.")
            return

        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()

        stats = self._get_normalization_stats(test_loader)
        out_dir = Path(self.test_results_dir) / "single_prediction_windows"
        out_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        with torch.inference_mode():
            for batch in test_loader:
                if processed >= num_samples:
                    break
                y_st = batch.fhr_st.to(device)
                y_ph = batch.fhr_ph.to(device)
                x_ph = batch.fhr_up_ph.to(device)
                y_raw = batch.fhr.to(device)

                forward_outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                mu_segments = forward_outputs.get("mu_pr")
                logvar_segments = forward_outputs.get("logvar_pr")
                if mu_segments is None:
                    logger.warning("Model outputs missing 'mu_pr'; skipping batch.")
                    continue

                batch_size = mu_segments.size(0)
                for idx in range(batch_size):
                    if processed >= num_samples:
                        break
                    guid_val, epoch_val = self._extract_guid_epoch(batch, idx)
                    fhr_norm = y_raw[idx]
                    fhr_denorm = self._maybe_denormalize(fhr_norm, "fhr", stats)
                    windows = self._prepare_single_prediction_windows(
                        predictions=mu_segments[idx],
                        logvar_predictions=logvar_segments[idx] if logvar_segments is not None else None,
                        raw_norm=fhr_norm,
                        raw_denorm=fhr_denorm,
                        start_index=start_index,
                        step_size=step_size,
                        max_windows=windows_per_sample,
                        stats=stats,
                    )
                    if not windows:
                        continue

                    raw_fhr_norm_np = fhr_norm.detach().cpu().numpy()
                    agg_pred_norm = np.full_like(raw_fhr_norm_np, np.nan)
                    agg_uncert_norm = np.full_like(raw_fhr_norm_np, np.nan)
                    for window in windows:
                        start = window["raw_start"]
                        end = window["raw_end"]
                        pred_norm = window.get("prediction_norm")
                        if pred_norm is not None:
                            agg_pred_norm[start:end] = pred_norm
                        uncert_norm = window.get("uncertainty_norm")
                        if uncert_norm is not None:
                            agg_uncert_norm[start:end] = uncert_norm

                    raw_fhr_denorm_np = fhr_denorm.detach().cpu().numpy()
                    plot_single_prediction_windows(
                        output_dir=str(out_dir),
                        raw_fhr_unnormalized=raw_fhr_denorm_np,
                        raw_fhr_normalized=raw_fhr_norm_np,
                        windows=windows,
                        aggregated_pred_norm=agg_pred_norm,
                        aggregated_uncertainty_norm=agg_uncert_norm,
                        sample_idx=processed,
                        sample_guid=guid_val,
                        epoch=epoch_val,
                    )
                    processed += 1

        if processed == 0:
            logger.warning("Single prediction probe did not generate any plots.")
        else:
            logger.info("Single prediction probe saved %d samples to %s", processed, out_dir)

    def run_temporal_accuracy_analysis(
        self,
        test_loader,
        *,
        num_samples: int = 50,
        start_index: int = 30,
        max_timesteps: Optional[int] = None,
    ) -> None:
        """Analyze prediction accuracy as a function of position and timestep.

        Generates two analyses:
        1. Within-window: VAF/SNR vs position (0-480) within each prediction window
        2. Across-timesteps: VAF/SNR vs timestep index across the recording

        Args:
            test_loader: DataLoader for test samples
            num_samples: Number of samples to aggregate statistics over
            start_index: Starting timestep (default: warmup_period)
            max_timesteps: Maximum number of timesteps to analyze (None = all)
        """
        if num_samples <= 0:
            logger.info("Temporal accuracy analysis skipped (num_samples<=0).")
            return

        if self.pytorch_model is None:
            self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model unavailable; cannot run temporal accuracy analysis.")
            return

        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()

        stride = int(getattr(model, "decimation_factor", 16))
        warmup = int(getattr(model, "warmup_period", 30))
        start_t = max(start_index, warmup)

        out_dir = Path(self.test_results_dir) / "temporal_accuracy"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        processed = 0

        with torch.inference_mode():
            for batch in test_loader:
                if processed >= num_samples:
                    break

                y_st = batch.fhr_st.to(device)
                y_ph = batch.fhr_ph.to(device)
                x_ph = batch.fhr_up_ph.to(device)
                y_raw = batch.fhr.to(device)

                forward_outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                mu_pr = forward_outputs.get("mu_pr")

                if mu_pr is None:
                    logger.warning("Model outputs missing 'mu_pr'; skipping batch.")
                    continue

                batch_size = mu_pr.size(0)
                T = mu_pr.size(1)
                H = mu_pr.size(2)
                end_t = T if max_timesteps is None else min(T, start_t + max_timesteps)

                for idx in range(batch_size):
                    if processed >= num_samples:
                        break

                    sample_preds: List[torch.Tensor] = []
                    sample_targets: List[torch.Tensor] = []

                    for t in range(start_t, end_t):
                        raw_start = t * stride
                        raw_end = raw_start + H

                        if raw_end > y_raw.size(1):
                            break

                        pred = mu_pr[idx, t, :]
                        target = y_raw[idx, raw_start:raw_end]

                        sample_preds.append(pred)
                        sample_targets.append(target)

                    if sample_preds:
                        all_predictions.append(torch.stack(sample_preds))
                        all_targets.append(torch.stack(sample_targets))
                        processed += 1

        if not all_predictions:
            logger.warning("No valid predictions collected for temporal accuracy analysis.")
            return

        all_predictions_tensor = torch.stack(all_predictions)
        all_targets_tensor = torch.stack(all_targets)

        N, T_prime, H = all_predictions_tensor.shape
        logger.info(
            "Collected %d samples with %d timesteps each (horizon=%d)",
            N, T_prime, H
        )

        preds_flat = all_predictions_tensor.view(N * T_prime, H)
        targets_flat = all_targets_tensor.view(N * T_prime, H)

        position_vaf: List[float] = []
        position_snr: List[float] = []

        logger.info("Computing within-window metrics (position 0-%d)...", H)
        for pos in range(H):
            pred_at_pos = preds_flat[:, pos]
            target_at_pos = targets_flat[:, pos]

            residual = target_at_pos - pred_at_pos
            mse = (residual ** 2).mean()
            signal_power = (target_at_pos ** 2).mean()
            noise_power = mse

            var_res = residual.var(unbiased=False)
            var_orig = target_at_pos.var(unbiased=False).clamp_min(1e-12)

            snr = 10.0 * torch.log10(signal_power.clamp_min(1e-12) / noise_power.clamp_min(1e-12))
            vaf = (1.0 - (var_res / var_orig)).clamp(0.0, 1.0)

            position_vaf.append(vaf.item())
            position_snr.append(snr.item())

        timestep_vaf: List[float] = []
        timestep_snr: List[float] = []

        logger.info("Computing across-timesteps metrics (timesteps %d-%d)...", start_t, start_t + T_prime)
        for t_idx in range(T_prime):
            metrics = self._compute_basic_metrics(
                all_targets_tensor[:, t_idx, :],
                all_predictions_tensor[:, t_idx, :],
            )
            if metrics is not None:
                timestep_vaf.append(metrics['vaf'].mean().item())
                timestep_snr.append(metrics['snr'].mean().item())
            else:
                timestep_vaf.append(float('nan'))
                timestep_snr.append(float('nan'))

        self._plot_within_window_analysis(position_vaf, position_snr, out_dir)
        self._plot_across_timesteps_analysis(timestep_vaf, timestep_snr, start_t, out_dir)

        logger.info(
            "Temporal accuracy analysis complete: %d samples, saved to %s",
            processed, out_dir
        )

    def run_latent_interpolation(self, test_loader, *, pair_count: int = 10, steps: int = 11) -> None:
        """Decode linear latent blends and export animated Matplotlib HTML dashboards."""
        if pair_count <= 0 or steps < 2:
            logger.info("Latent interpolation skipped (invalid pair_count/steps)")
            return
        if self.pytorch_model is None:
            self.create_model()
        decoder = getattr(self.pytorch_model, "decoder", None)
        if decoder is None:
            logger.error("SeqVAE core does not expose a decoder attribute; cannot run interpolation.")
            return
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        out_dir = Path(self.test_results_dir) / "latent_interpolation_matplotlib"
        out_dir.mkdir(parents=True, exist_ok=True)

        required_samples = pair_count * 2
        samples: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        metadata_complete = True
        with torch.inference_mode():
            for batch in test_loader:
                batch_size = batch.fhr_st.size(0)
                for idx in range(batch_size):
                    guid_val, epoch_val = self._extract_guid_epoch(batch, idx)
                    entry = {
                        "fhr_st": batch.fhr_st[idx : idx + 1].to(device),
                        "fhr_ph": batch.fhr_ph[idx : idx + 1].to(device),
                        "fhr_up_ph": batch.fhr_up_ph[idx : idx + 1].to(device),
                        "fhr": batch.fhr[idx : idx + 1].to(device),
                        "guid": guid_val,
                        "epoch": epoch_val,
                    }
                    samples.append(entry)
                    if guid_val is not None and epoch_val is not None:
                        grouped.setdefault(guid_val, []).append(entry)
                    else:
                        metadata_complete = False
                    if len(samples) >= required_samples:
                        break
                if len(samples) >= required_samples:
                    break
        if len(samples) < 2:
            logger.error("Insufficient samples for interpolation.")
            return

        if metadata_complete and grouped:
            sample_pairs = self._build_consecutive_pairs(grouped, pair_count)
        else:
            sample_pairs = []
        if not sample_pairs:
            logger.warning("Falling back to sequential pairing for interpolation.")
            for idx in range(0, min(len(samples) - 1, pair_count * 2), 2):
                sample_pairs.append((samples[idx], samples[idx + 1]))
        if not sample_pairs:
            logger.error("Unable to construct interpolation pairs.")
            return

        weights = np.linspace(0.0, 1.0, steps)
        rendered = 0
        for pair_idx, (sample_a, sample_b) in enumerate(sample_pairs[:pair_count]):
            with torch.inference_mode():
                outputs_a = model(y_st=sample_a["fhr_st"], y_ph=sample_a["fhr_ph"], x_ph=sample_a["fhr_up_ph"])
                outputs_b = model(y_st=sample_b["fhr_st"], y_ph=sample_b["fhr_ph"], x_ph=sample_b["fhr_up_ph"])
            latent_a = outputs_a.get("z")
            latent_b = outputs_b.get("z")
            if latent_a is None or latent_b is None:
                logger.warning("Skipping pair %d because latents are missing.", pair_idx)
                continue

            y_a = sample_a["fhr"]
            y_b = sample_b["fhr"]
            latent_dim = latent_a.size(-1)
            seq_len = latent_a.size(1)
            dim_template = list(range(latent_dim))

            recon_sequences: List[np.ndarray] = []
            target_sequences: List[np.ndarray] = []
            latent_means: List[np.ndarray] = []
            heatmaps: List[np.ndarray] = []
            latent_hists: List[np.ndarray] = []
            latent_dim_hists: List[np.ndarray] = []
            latent_hist_bins: Optional[np.ndarray] = None
            latent_hist_values: List[np.ndarray] = []
            latent_hist_bins: Optional[np.ndarray] = None
            latent_hist_values: List[np.ndarray] = []
            latent_hist_bins: Optional[np.ndarray] = None

            for alpha in weights:
                latent_interp = torch.lerp(latent_a, latent_b, float(alpha))
                decoded_linear, decoded_mu, _ = decoder(latent_interp)
                averaged_mu, _, _ = self._average_raw_prediction_segments(model, decoded_mu)
                if averaged_mu is None:
                    logger.warning("Decoder did not return usable raw predictions; skipping alpha %.3f.", alpha)
                    continue
                target_interp = torch.lerp(y_a, y_b, float(alpha))

                recon_flat = self._flatten_signal_for_plot(averaged_mu[0])
                target_flat = self._flatten_signal_for_plot(target_interp[0])
                series_len = min(len(recon_flat), len(target_flat))
                if series_len <= 1:
                    continue

                recon_sequences.append(recon_flat[:series_len])
                target_sequences.append(target_flat[:series_len])
                latent_mean_np = latent_interp.mean(dim=1)[0].detach().cpu().numpy()
                latent_means.append(self._sanitize_finite_array(latent_mean_np))
                heat_np = latent_interp[0].detach().cpu().numpy().T  # (latent_dim, seq_len)
                heatmaps.append(self._sanitize_finite_array(heat_np))
                latent_flat = latent_interp[0].detach().cpu().numpy().ravel()
                latent_hists.append(self._sanitize_finite_array(latent_flat))

            if not recon_sequences or not target_sequences:
                logger.warning("Interpolation pair %d produced no valid decoded signals.", pair_idx)
                continue

            min_series_len = min(arr.shape[0] for arr in (recon_sequences + target_sequences))
            if min_series_len <= 1:
                logger.warning("Skipping pair %d (series length %d).", pair_idx, min_series_len)
                continue

            common_len = min(len(recon_sequences), len(target_sequences), len(latent_means), len(heatmaps))
            if common_len == 0:
                logger.warning("Skipping pair %d due to empty synchronized sequences.", pair_idx)
                continue

            recon_sequences = [seq[:min_series_len] for seq in recon_sequences[:common_len]]
            target_sequences = [seq[:min_series_len] for seq in target_sequences[:common_len]]
            latent_means = [seq for seq in latent_means[:common_len] if seq.shape[0] == latent_dim]
            heatmaps = [img for img in heatmaps[:common_len] if img.shape == (latent_dim, seq_len)]
            if not latent_means or not heatmaps:
                logger.warning("Skipping pair %d due to invalid latent summaries/heatmaps.", pair_idx)
                continue
            # Build common histogram bins for latent distributions
            latent_hist_bins = np.linspace(
                float(np.min(np.concatenate(latent_hists, axis=0))),
                float(np.max(np.concatenate(latent_hists, axis=0))) + 1e-6,
                41,
            )
            latent_hist_values = []
            for arr in latent_hists[:common_len]:
                hist, _ = np.histogram(arr, bins=latent_hist_bins, density=True)
                latent_hist_values.append(hist)
            # Per-dimension distributions over temporal values: shape (steps, latent_dim, bins)
            latent_dim_hist_values: List[np.ndarray] = []
            for step_idx, img in enumerate(heatmaps):
                # img shape: (latent_dim, seq_len)
                dim_hists = []
                for d in range(latent_dim):
                    h, _ = np.histogram(img[d], bins=latent_hist_bins, density=True)
                    dim_hists.append(h)
                latent_dim_hist_values.append(np.stack(dim_hists, axis=0))  # (latent_dim, bins)

            heat_min = float(min(float(np.min(img)) for img in heatmaps))
            heat_max = float(max(float(np.max(img)) for img in heatmaps))
            if heat_max <= heat_min:
                heat_max = heat_min + 1e-6
            if latent_hist_bins is None:
                logger.warning("Skipping pair %d due to missing histogram bins.", pair_idx)
                continue

            x_values = np.arange(min_series_len)

            fig, axes = plt.subplots(
                5,
                1,
                figsize=(10, 13),
                gridspec_kw={"height_ratios": [3.0, 2.0, 3.5, 2.0, 2.0], "hspace": 0.45},
            )
            ax_sig, ax_bar, ax_heat, ax_hist, ax_dimhist = axes

            line_target, = ax_sig.plot(x_values, target_sequences[0], color="#268bd2", label="Target", linewidth=1.8)
            line_recon, = ax_sig.plot(x_values, recon_sequences[0], color="#d33682", label="Reconstruction", linewidth=1.2)
            ax_sig.set_title(f"Pair {pair_idx} – Signal")
            ax_sig.set_xlabel("Sample Index")
            ax_sig.set_ylabel("Amplitude")
            ax_sig.legend(loc="upper right")

            bars = ax_bar.bar(dim_template, latent_means[0], color="#6c71c4")
            ax_bar.set_title("Latent Mean by Dimension")
            ax_bar.set_xlabel("Latent Dimension")
            ax_bar.set_ylabel("Mean Value")

            heat_img = ax_heat.imshow(
                heatmaps[0],
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=heat_min,
                vmax=heat_max,
                extent=[0, seq_len, 0, latent_dim],
            )
            cbar = fig.colorbar(heat_img, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Latent Activation")
            ax_heat.set_title("Latent Heatmap")
            ax_heat.set_xlabel("Time Step")
            ax_heat.set_ylabel("Latent Dimension")

            hist_centers = 0.5 * (latent_hist_bins[:-1] + latent_hist_bins[1:])
            line_hist, = ax_hist.plot(hist_centers, latent_hist_values[0], color="#2c3e50", linewidth=1.5)
            ax_hist.set_title("Latent Distribution (flattened)")
            ax_hist.set_xlabel("Latent Value")
            ax_hist.set_ylabel("Density")
            ax_hist.grid(True, alpha=0.3)

            # Per-dimension distributions: show a few dims stacked
            dim_hist_lines = []
            for d in range(min(latent_dim, 4)):
                ln, = ax_dimhist.plot(
                    hist_centers,
                    latent_dim_hist_values[0][d],
                    label=f"z[{d}]",
                    linewidth=1.2,
                )
                dim_hist_lines.append(ln)
            ax_dimhist.set_title("Latent Distribution per Dimension")
            ax_dimhist.set_xlabel("Latent Value")
            ax_dimhist.set_ylabel("Density")
            ax_dimhist.legend(loc="upper right", fontsize=8, ncol=2)
            ax_dimhist.grid(True, alpha=0.3)

            def _update(frame_idx: int):
                line_target.set_ydata(target_sequences[frame_idx])
                line_recon.set_ydata(recon_sequences[frame_idx])
                frame_latent = latent_means[frame_idx]
                for bar, val in zip(bars, frame_latent):
                    bar.set_height(val)
                heat_img.set_data(heatmaps[frame_idx])
                line_hist.set_ydata(latent_hist_values[frame_idx])
                frame_dim_hists = latent_dim_hist_values[frame_idx]
                for ln, arr in zip(dim_hist_lines, frame_dim_hists[: len(dim_hist_lines)]):
                    ln.set_ydata(arr)
                return [line_target, line_recon, heat_img, line_hist, *dim_hist_lines, *bars]

            anim = animation.FuncAnimation(
                fig,
                _update,
                frames=len(recon_sequences),
                interval=400,
                blit=False,
            )
            html_str = anim.to_jshtml()
            output_path = out_dir / f"latent_interp_pair_{pair_idx:02d}.html"
            output_path.write_text(html_str, encoding="utf-8")
            plt.close(fig)
            logger.info(
                "Saved matplotlib interpolation pair %d -> %s (steps=%d, series_len=%d, latent_dim=%d, seq_len=%d)",
                pair_idx,
                output_path,
                len(recon_sequences),
                min_series_len,
                latent_dim,
                seq_len,
            )
            rendered += 1

        if rendered == 0:
            logger.error("No latent interpolation plots were generated.")
        else:
            logger.info("Latent interpolation animations saved to %s (pairs=%d).", out_dir, rendered)

    def run_latent_interpolation_plotly(self, test_loader, *, pair_count: int = 10, steps: int = 11) -> None:
        """Export latent interpolation as interactive Plotly HTML (signals, latent bars, latent heatmap)."""
        if pair_count <= 0 or steps < 2:
            logger.info("Latent interpolation (plotly) skipped (invalid pair_count/steps)")
            return
        if self.pytorch_model is None:
            self.create_model()
        decoder = getattr(self.pytorch_model, "decoder", None)
        if decoder is None:
            logger.error("SeqVAE core does not expose a decoder attribute; cannot run interpolation.")
            return
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        out_dir = Path(self.test_results_dir) / "latent_interpolation_plotly"
        out_dir.mkdir(parents=True, exist_ok=True)

        required_samples = pair_count * 2
        samples: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        metadata_complete = True
        with torch.inference_mode():
            for batch in test_loader:
                batch_size = batch.fhr_st.size(0)
                for idx in range(batch_size):
                    guid_val, epoch_val = self._extract_guid_epoch(batch, idx)
                    entry = {
                        "fhr_st": batch.fhr_st[idx : idx + 1].to(device),
                        "fhr_ph": batch.fhr_ph[idx : idx + 1].to(device),
                        "fhr_up_ph": batch.fhr_up_ph[idx : idx + 1].to(device),
                        "fhr": batch.fhr[idx : idx + 1].to(device),
                        "guid": guid_val,
                        "epoch": epoch_val,
                    }
                    samples.append(entry)
                    if guid_val is not None and epoch_val is not None:
                        grouped.setdefault(guid_val, []).append(entry)
                    else:
                        metadata_complete = False
                    if len(samples) >= required_samples:
                        break
                if len(samples) >= required_samples:
                    break
        if len(samples) < 2:
            logger.error("Insufficient samples for interpolation.")
            return

        if metadata_complete and grouped:
            sample_pairs = self._build_consecutive_pairs(grouped, pair_count)
        else:
            sample_pairs = []
        if not sample_pairs:
            for idx in range(0, min(len(samples) - 1, pair_count * 2), 2):
                sample_pairs.append((samples[idx], samples[idx + 1]))
        if not sample_pairs:
            logger.error("Unable to construct interpolation pairs.")
            return

        weights = np.linspace(0.0, 1.0, steps)
        rendered = 0
        for pair_idx, (sample_a, sample_b) in enumerate(sample_pairs[:pair_count]):
            with torch.inference_mode():
                outputs_a = model(y_st=sample_a["fhr_st"], y_ph=sample_a["fhr_ph"], x_ph=sample_a["fhr_up_ph"])
                outputs_b = model(y_st=sample_b["fhr_st"], y_ph=sample_b["fhr_ph"], x_ph=sample_b["fhr_up_ph"])
            latent_a = outputs_a.get("z")
            latent_b = outputs_b.get("z")
            if latent_a is None or latent_b is None:
                logger.warning("Skipping pair %d because latents are missing.", pair_idx)
                continue

            y_a = sample_a["fhr"]
            y_b = sample_b["fhr"]
            latent_dim = latent_a.size(-1)
            seq_len = latent_a.size(1)
            dim_template = list(range(latent_dim))

            recon_sequences: List[List[float]] = []
            target_sequences: List[List[float]] = []
            latent_means: List[List[float]] = []
            heatmaps: List[List[List[float]]] = []
            latent_hists: List[np.ndarray] = []
            latent_dim_hist_values: List[List[List[float]]] = []
            latent_hist_bins: Optional[np.ndarray] = None
            latent_hist_values: List[np.ndarray] = []

            for alpha in weights:
                latent_interp = torch.lerp(latent_a, latent_b, float(alpha))
                decoded_linear, decoded_mu, _ = decoder(latent_interp)
                averaged_mu, _, _ = self._average_raw_prediction_segments(model, decoded_mu)
                if averaged_mu is None:
                    logger.warning("Decoder did not return usable raw predictions; skipping alpha %.3f.", alpha)
                    continue
                target_interp = torch.lerp(y_a, y_b, float(alpha))

                recon_flat = self._flatten_signal_for_plot(averaged_mu[0])
                target_flat = self._flatten_signal_for_plot(target_interp[0])
                series_len = min(len(recon_flat), len(target_flat))
                if series_len <= 1:
                    continue

                recon_sequences.append(recon_flat[:series_len].tolist())
                target_sequences.append(target_flat[:series_len].tolist())
                latent_mean_np = latent_interp.mean(dim=1)[0].detach().cpu().numpy()
                latent_means.append(self._sanitize_finite_array(latent_mean_np).tolist())
                heat_np = latent_interp[0].detach().cpu().numpy().T  # (latent_dim, seq_len)
                heatmaps.append(self._sanitize_finite_array(heat_np).tolist())
                latent_flat = latent_interp[0].detach().cpu().numpy().ravel()
                latent_hists.append(self._sanitize_finite_array(latent_flat))

            if not recon_sequences or not target_sequences:
                logger.warning("Interpolation pair %d produced no valid decoded signals.", pair_idx)
                continue

            min_series_len = min(len(seq) for seq in (recon_sequences + target_sequences))
            if min_series_len <= 1:
                logger.warning("Skipping pair %d (series length %d).", pair_idx, min_series_len)
                continue

            common_len = min(
                len(recon_sequences),
                len(target_sequences),
                len(latent_means),
                len(heatmaps),
            )
            if common_len == 0:
                logger.warning("Skipping pair %d due to empty synchronized sequences.", pair_idx)
                continue
            recon_sequences = recon_sequences[:common_len]
            target_sequences = target_sequences[:common_len]
            latent_means = latent_means[:common_len]
            heatmaps = heatmaps[:common_len]
            latent_hists = latent_hists[:common_len]

            x_values = np.arange(min_series_len).tolist()
            recon_sequences = [seq[:min_series_len] for seq in recon_sequences]
            target_sequences = [seq[:min_series_len] for seq in target_sequences]
            latent_means = [seq for seq in latent_means if len(seq) == latent_dim]
            heatmaps = [np.asarray(h, dtype=float).tolist() for h in heatmaps if np.asarray(h).shape == (latent_dim, seq_len)]
            if not latent_means or not heatmaps or not latent_hists:
                logger.warning("Skipping pair %d due to invalid latent summaries/heatmaps.", pair_idx)
                continue

            all_latent_flat = np.concatenate(latent_hists, axis=0)
            hist_min = float(np.min(all_latent_flat))
            hist_max = float(np.max(all_latent_flat))
            if hist_max <= hist_min:
                hist_max = hist_min + 1e-6
            latent_hist_bins = np.linspace(hist_min, hist_max, 41)
            latent_hist_values = []
            for arr in latent_hists[: len(heatmaps)]:
                hist, _ = np.histogram(arr, bins=latent_hist_bins, density=True)
                latent_hist_values.append(hist.tolist())
            # per-dimension histograms (first up to 4 dims)
            for img in heatmaps:
                arr = np.asarray(img)
                dim_hists = []
                for d in range(min(latent_dim, 4)):
                    h, _ = np.histogram(arr[d], bins=latent_hist_bins, density=True)
                    dim_hists.append(h.tolist())
                latent_dim_hist_values.append(dim_hists)

            heat_min = min(float(np.min(np.array(h))) for h in heatmaps)
            heat_max = max(float(np.max(np.array(h))) for h in heatmaps)
            if heat_max <= heat_min:
                heat_max = heat_min + 1e-6

            fig = make_subplots(
                rows=5,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.05,
                row_heights=[0.28, 0.18, 0.32, 0.12, 0.18],
                specs=[[{"type": "xy"}], [{"type": "bar"}], [{"type": "heatmap"}], [{"type": "xy"}], [{"type": "xy"}]],
            )
            fig.add_trace(
                go.Scatter(x=x_values, y=target_sequences[0], name="Target", line=dict(color="#268bd2")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=x_values, y=recon_sequences[0], name="Reconstruction", line=dict(color="#d33682")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=dim_template, y=latent_means[0], name="Latent Mean", marker_color="#6c71c4"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=heatmaps[0],
                    coloraxis="coloraxis",
                    zmin=heat_min,
                    zmax=heat_max,
                ),
                row=3,
                col=1,
            )
            hist_centers = 0.5 * (latent_hist_bins[:-1] + latent_hist_bins[1:])
            fig.add_trace(
                go.Scatter(x=hist_centers.tolist(), y=latent_hist_values[0], mode="lines", name="Latent PDF", line=dict(color="#2c3e50")),
                row=4,
                col=1,
            )
            dim_traces = []
            dims_shown = min(latent_dim, 4)
            for d in range(dims_shown):
                tr = go.Scatter(
                    x=hist_centers.tolist(),
                    y=latent_dim_hist_values[0][d],
                    mode="lines",
                    name=f"z[{d}]",
                    line=dict(width=1.2),
                )
                fig.add_trace(tr, row=5, col=1)
                dim_traces.append(tr)

            frames = []
            for step_idx in range(len(recon_sequences)):
                frames.append(
                    go.Frame(
                        data=[
                            go.Scatter(y=target_sequences[step_idx]),
                            go.Scatter(y=recon_sequences[step_idx]),
                            go.Bar(y=latent_means[step_idx]),
                            go.Heatmap(z=heatmaps[step_idx]),
                            go.Scatter(y=latent_hist_values[step_idx]),
                            *[
                                go.Scatter(y=latent_dim_hist_values[step_idx][d])
                                for d in range(dims_shown)
                            ],
                        ],
                        name=f"step{step_idx}",
                    )
                )

            sliders = [
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"font": {"size": 12}, "prefix": "Step: ", "visible": True},
                    "pad": {"t": 5, "b": 5},
                    "steps": [
                        {
                            "args": [[f"step{idx}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                            "label": f"{idx}",
                            "method": "animate",
                        }
                        for idx in range(len(frames))
                    ],
                }
            ]

            fig.update_layout(
                title=f"Latent Interpolation Pair {pair_idx}",
                coloraxis={
                    "colorscale": "Viridis",
                    "cmin": heat_min,
                    "cmax": heat_max,
                    "colorbar": {"len": 0.2, "thickness": 10, "y": 0.21},
                },
                sliders=sliders,
                height=1100,
                showlegend=True,
            )
            fig.frames = frames

            output_path = out_dir / f"latent_interp_pair_{pair_idx:02d}_plotly.html"
            fig.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
            logger.info("Saved plotly interpolation for pair %d -> %s", pair_idx, output_path)
            rendered += 1

        if rendered == 0:
            logger.error("No plotly latent interpolation plots were generated.")

    @staticmethod
    def _maybe_denormalize(tensor: torch.Tensor, field: str, stats: Optional[Dict]) -> torch.Tensor:
        if not stats:
            return tensor
        try:
            return denormalize_signal_data(tensor, field, stats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to denormalize %s: %s. Returning tensor as-is.", field, exc)
            return tensor

    @staticmethod
    def _denormalize_std(
        std_tensor: torch.Tensor,
        field: str,
        stats: Optional[Dict],
        *,
        raw_start: Optional[int] = None,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        if not stats or field not in stats:
            return std_tensor
        try:
            field_stats = stats[field] or {}
            scale = field_stats.get("std_tensor")
            if scale is None:
                scale = field_stats.get("std", 1.0)
            scale_tensor = torch.as_tensor(scale, dtype=std_tensor.dtype, device=std_tensor.device)
            if (
                scale_tensor.dim() > 0
                and raw_start is not None
                and length is not None
                and scale_tensor.size(-1) >= raw_start + length
            ):
                scale_tensor = scale_tensor.narrow(-1, raw_start, length)
            while scale_tensor.dim() < std_tensor.dim():
                scale_tensor = scale_tensor.unsqueeze(0)
            return std_tensor * (scale_tensor + 1e-8)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to denormalize std for %s: %s. Returning std as-is.", field, exc)
            return std_tensor

    @staticmethod
    def _get_normalization_stats(loader) -> Optional[Dict]:
        dataset = getattr(loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "get_normalization_stats"):
            return None
        try:
            return dataset.get_normalization_stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch normalization stats: %s", exc)
            return None

    @staticmethod
    def _extract_reconstruction_features(
        linear_output: Optional[torch.Tensor],
        st_channels: int,
        ph_channels: int,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            linear_output is None
            or linear_output.dim() != 3
            or linear_output.size(-1) < st_channels + ph_channels
        ):
            return np.zeros((st_channels, seq_len)), np.zeros((ph_channels, seq_len))
        linear_np = linear_output[0].detach().cpu().numpy()  # (T, channels_total)
        recon_st = linear_np[:, :st_channels].T
        recon_ph = linear_np[:, st_channels : st_channels + ph_channels].T
        return recon_st, recon_ph

    def _prepare_single_prediction_windows(
        self,
        *,
        predictions: torch.Tensor,
        logvar_predictions: Optional[torch.Tensor],
        raw_norm: torch.Tensor,
        raw_denorm: torch.Tensor,
        start_index: int,
        step_size: Optional[int],
        max_windows: int,
        stats: Optional[Dict],
    ) -> List[Dict[str, Any]]:
        if predictions.dim() == 3:
            predictions = predictions.squeeze(0)
        if predictions.dim() != 2:
            return []
        stride = int(getattr(self.pytorch_model, "decimation_factor", 16))
        horizon = predictions.size(-1)
        warmup = int(getattr(self.pytorch_model, "warmup_period", 0))
        effective_start = max(0, max(start_index, warmup))
        if step_size is None or step_size <= 0:
            effective_step = max(1, horizon // stride)
        else:
            effective_step = int(step_size)
        raw_norm = raw_norm.view(-1)
        raw_denorm = raw_denorm.view(-1)
        raw_len = raw_norm.size(0)

        windows: List[Dict[str, Any]] = []
        total_steps = predictions.size(0)
        t_idx = effective_start
        while t_idx < total_steps and len(windows) < max_windows:
            raw_start = t_idx * stride
            raw_end = raw_start + horizon
            if raw_end > raw_len:
                break
            pred_segment = predictions[t_idx]
            logvar_segment = logvar_predictions[t_idx] if logvar_predictions is not None else None
            target_norm = raw_norm[raw_start:raw_end]
            target_denorm = raw_denorm[raw_start:raw_end]
            pred_denorm = self._maybe_denormalize(pred_segment.unsqueeze(0), "fhr", stats)[0]
            std_denorm_np = None
            std_norm_np = None
            if logvar_segment is not None:
                std_norm = torch.exp(0.5 * logvar_segment)
                std_norm_np = std_norm.detach().cpu().numpy()
                std_denorm = self._denormalize_std(
                    std_norm.unsqueeze(0),
                    "fhr",
                    stats,
                    raw_start=raw_start,
                    length=horizon,
                )[0]
                std_denorm_np = std_denorm.detach().cpu().numpy()
            metrics = self._compute_basic_metrics(
                target_denorm.unsqueeze(0),
                pred_denorm.unsqueeze(0),
            )
            metrics_map = (
                {key: float(val.detach().cpu().item()) for key, val in metrics.items()}
                if metrics is not None
                else {}
            )
            windows.append(
                {
                    "t_index": int(t_idx),
                    "raw_start": int(raw_start),
                    "raw_end": int(raw_end),
                    "prediction": pred_denorm.detach().cpu().numpy(),
                    "target": target_denorm.detach().cpu().numpy(),
                    "prediction_norm": pred_segment.detach().cpu().numpy(),
                    "target_norm": target_norm.detach().cpu().numpy(),
                    "uncertainty": std_denorm_np,
                    "uncertainty_norm": std_norm_np,
                    "metrics": metrics_map,
                }
            )
            t_idx += effective_step

        return windows

    @staticmethod
    def _flatten_signal_for_plot(tensor: torch.Tensor) -> np.ndarray:
        """Return a 1D numpy series suitable for plotting."""
        array = tensor.detach().cpu().numpy()
        squeezed = np.squeeze(array)
        if squeezed.ndim == 0:
            flat = np.asarray([float(squeezed)], dtype=float)
        elif squeezed.ndim == 1:
            flat = np.asarray(squeezed, dtype=float)
        elif squeezed.ndim == 2:
            flat = np.asarray(squeezed[0], dtype=float)
        else:
            flat = np.asarray(squeezed.reshape(-1), dtype=float)
        np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        flat = np.clip(flat, -1e6, 1e6)
        return flat

    @staticmethod
    def _sanitize_finite_array(arr: np.ndarray, clip_val: float = 1e6) -> np.ndarray:
        """Ensure array is finite and reasonably bounded for JSON serialization."""
        arr = np.asarray(arr, dtype=float)
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return np.clip(arr, -clip_val, clip_val)

    @staticmethod
    def _plot_within_window_analysis(
        vaf_values: List[float],
        snr_values: List[float],
        output_dir: Path,
    ) -> None:
        """Plot VAF and SNR vs position within 480-sample prediction window."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(range(len(vaf_values)), vaf_values, linewidth=1.5, color='#2E86AB')
        ax1.set_xlabel('Position in Window (samples)', fontsize=11)
        ax1.set_ylabel('VAF', fontsize=11)
        ax1.set_title('Prediction Quality vs Position Within Prediction Window', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        ax2.plot(range(len(snr_values)), snr_values, linewidth=1.5, color='#A23B72')
        ax2.set_xlabel('Position in Window (samples)', fontsize=11)
        ax2.set_ylabel('SNR (dB)', fontsize=11)
        ax2.set_title('SNR vs Position Within Prediction Window', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'temporal_accuracy_within_window.png'
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        logger.info("Saved within-window analysis plot to %s", output_path)

    @staticmethod
    def _plot_across_timesteps_analysis(
        vaf_values: List[float],
        snr_values: List[float],
        start_timestep: int,
        output_dir: Path,
    ) -> None:
        """Plot VAF and SNR vs timestep index across recording."""
        timestep_indices = list(range(start_timestep, start_timestep + len(vaf_values)))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(timestep_indices, vaf_values, linewidth=1.5, color='#2E86AB')
        ax1.set_xlabel('Timestep Index', fontsize=11)
        ax1.set_ylabel('VAF', fontsize=11)
        ax1.set_title('Prediction Quality vs Timestep in Recording', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        ax2.plot(timestep_indices, snr_values, linewidth=1.5, color='#A23B72')
        ax2.set_xlabel('Timestep Index', fontsize=11)
        ax2.set_ylabel('SNR (dB)', fontsize=11)
        ax2.set_title('SNR vs Timestep in Recording', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'temporal_accuracy_across_timesteps.png'
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        logger.info("Saved across-timesteps analysis plot to %s", output_path)

    @staticmethod
    def _plot_latent_metric_curves(metric_map: Dict[str, Dict[float, List[float]]], scales: List[float], path: Path) -> None:
        metrics = ["vaf", "mse", "snr"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for idx, metric in enumerate(metrics):
            axis = axes[0][idx]
            values = metric_map.get(metric, {})
            means = [np.mean(values.get(scale, [np.nan])) for scale in scales]
            axis.plot(scales, means, marker="o")
            axis.set_title(metric.upper())
            axis.set_xlabel("Scale factor")
            axis.set_ylabel(metric.upper())
            axis.grid(True, alpha=0.4)
        fig.suptitle("Latent Magnitude Sweep", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    @staticmethod
    def _plot_feature_importance(values: np.ndarray, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        dim_ids = np.arange(len(values))
        ax.bar(dim_ids, values)
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Mean |∂Loss/∂z|")
        ax.set_title("Latent Feature Attribution")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeqVAE-TEB small tester")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config file.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for histogram plots (None = all, 0 = skip).",
    )
    parser.add_argument(
        "--metrics-max-samples",
        type=int,
        default=None,
        help="Additional limit for histogram metric samples (None = unrestricted).",
    )
    parser.add_argument(
        "--analysis-samples",
        type=int,
        default=10,
        help="Number of samples to visualize with analysis plots (0 = skip).",
    )
    parser.add_argument(
        "--latent-dist-samples",
        type=int,
        default=500,
        help="Samples to use when plotting latent distributions (0 = skip).",
    )
    parser.add_argument(
        "--latent-interp-pairs",
        type=int,
        default=10,
        help="Number of sample pairs for interpolation animations.",
    )
    parser.add_argument(
        "--latent-interp-steps",
        type=int,
        default=11,
        help="Number of discrete interpolation steps (>=2).",
    )
    parser.add_argument(
        "--latent-interp-plotly",
        action="store_true",
        help="Also export latent interpolation animations using Plotly.",
    )
    parser.add_argument(
        "--single-pred-samples",
        type=int,
        default=0,
        help="Samples for non-overlapping single prediction plots (0 = skip).",
    )
    parser.add_argument(
        "--single-pred-start",
        type=int,
        default=20,
        help="Starting timestep index in scattering/latent domain for single predictions.",
    )
    parser.add_argument(
        "--single-pred-step",
        type=int,
        default=None,
        help="Stride in scattering timesteps between single predictions (default: horizon/decimation).",
    )
    parser.add_argument(
        "--single-pred-windows",
        type=int,
        default=4,
        help="Number of non-overlapping windows to visualize per sample.",
    )
    parser.add_argument(
        "--temporal-accuracy-samples",
        type=int,
        default=480,
        help="Samples for temporal accuracy analysis (0 = skip).",
    )
    return parser.parse_args()


def _parse_int_list(value: Optional[Any]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError:
            logger.warning("Failed to parse int list from %s; ignoring.", value)
            return None
    if isinstance(value, Iterable):
        try:
            return [int(item) for item in value]
        except (TypeError, ValueError):
            logger.warning("Failed to coerce iterable %s into int list; ignoring.", value)
            return None
    try:
        return [int(value)]
    except (TypeError, ValueError):
        logger.warning("Failed to parse value %s as int list; ignoring.", value)
        return None


def _parse_float_list(value: Optional[Any]) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            return [float(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError:
            logger.warning("Failed to parse float list from %s; ignoring.", value)
            return None
    if isinstance(value, Iterable):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            logger.warning("Failed to coerce iterable %s into float list; ignoring.", value)
            return None
    try:
        return [float(value)]
    except (TypeError, ValueError):
        logger.warning("Failed to parse value %s as float list; ignoring.", value)
        return None


def main(
    *,
    config: Path | str = Path("config.yaml"),
    max_samples: Optional[int] = 100,
    metrics_max_samples: Optional[int] = None,
    analysis_samples: int = 10,
    latent_dist_samples: int = 100,
    latent_interp_pairs: int = 10,
    latent_interp_steps: int = 10,
    latent_interp_plotly: bool = False,
    single_pred_samples: int = 20,
    single_pred_start: int = 20,
    single_pred_step: Optional[int] = 30,
    single_pred_windows: int = 4,
    temporal_accuracy_samples: int = 480,
) -> None:
    config_path = Path(config)
    config_data = load_config(config_path)
    tester = GraphModelVaeTebSmallTester(str(config_path))
    tester.setup_config()
    tester.create_model()
    test_loader = build_test_dataloader(config_data)
    # if latent_interp_pairs and latent_interp_steps and latent_interp_steps >= 2:
    #     tester.run_latent_interpolation(
    #         test_loader,
    #         pair_count=latent_interp_pairs,
    #         steps=latent_interp_steps,
    #     )
    #     if latent_interp_plotly:
    #         tester.run_latent_interpolation_plotly(
    #             test_loader,
    #             pair_count=latent_interp_pairs,
    #             steps=latent_interp_steps,
    #         )
    # if latent_dist_samples and latent_dist_samples > 0:
    #     tester.run_latent_distribution(test_loader, num_samples=latent_dist_samples)
    # if analysis_samples and analysis_samples > 0:
    #     tester.run_analysis_and_plot(test_loader, num_samples=analysis_samples)
    if single_pred_samples and single_pred_samples > 0:
        tester.run_single_prediction_probe(
            test_loader,
            num_samples=single_pred_samples,
            start_index=single_pred_start,
            step_size=single_pred_step,
            windows_per_sample=single_pred_windows,
        )
    if temporal_accuracy_samples and temporal_accuracy_samples > 0:
        tester.run_temporal_accuracy_analysis(
            test_loader,
            num_samples=temporal_accuracy_samples,
        )
    # if max_samples is None or max_samples > 0 or (metrics_max_samples and metrics_max_samples > 0):
    #     tester.run_histogram_test(
    #         test_loader,
    #         num_samples=max_samples,
    #         max_samples=metrics_max_samples,
    #     )


if __name__ == "__main__":
    cli_args = parse_args()
    main()
