from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, ColorBar, CustomJS, LinearColorMapper, Slider
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, output_file, save
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from trainer import GraphModelVaeTebSmallTrainer
from train.graph_models_utils import denormalize_signal_data
from utils.plot_utils import plot_metrics_histograms, plot_model_analysis, plot_vae_reconstruction


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
                recon = forward_outputs.get("mu_pr")
                if recon is None:
                    logger.warning("Forward pass did not return 'mu_pr'; skipping batch.")
                    continue

                metrics = self._compute_basic_metrics(y_raw, recon)
                if metrics is None:
                    logger.warning("Metric computation failed for batch; skipping.")
                    continue
                kld_batch = self._kld_from_forward(forward_outputs, device).to(recon.device)

                batch_vaf = metrics["vaf"].detach().cpu().tolist()
                batch_mse = metrics["mse"].detach().cpu().tolist()
                batch_snr = metrics["snr"].detach().cpu().tolist()
                batch_kld = kld_batch.detach().cpu().tolist()

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

        logger.info(
            "Histogram metrics collected for %d samples (VAF mean=%.4f, MSE mean=%.6f, SNR mean=%.2f dB, KLD mean=%.6f)",
            len(vaf_values),
            float(np.mean(vaf_values)),
            float(np.mean(mse_values)),
            float(np.mean(snr_values)),
            float(np.mean(kld_values)),
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
        """Return the unreduced KLD tensor for downstream analysis."""
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
        return kld

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
        return tensor.mean(dim=dims)

    @staticmethod
    def _compute_basic_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """Return per-sample tensors for MSE, SNR (dB), and VAF."""
        if y_true.shape != y_pred.shape:
            return None
        residual = y_true - y_pred
        dims = tuple(range(1, residual.ndim))
        if not dims:
            return None
        mse = (residual ** 2).mean(dim=dims)
        signal_power = (y_true ** 2).mean(dim=dims)
        noise_power = (residual ** 2).mean(dim=dims)
        snr = torch.where(
            noise_power > 1e-12,
            10.0 * torch.log10(signal_power.clamp_min(1e-12) / noise_power.clamp_min(1e-12)),
            torch.full_like(signal_power, 100.0),
        )
        var_res = residual.var(dim=dims, unbiased=False)
        var_orig = y_true.var(dim=dims, unbiased=False).clamp_min(1e-12)
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
                    recon = forward_outputs.get("mu_pr")
                    logvar = forward_outputs.get("logvar_pr")
                    latent = forward_outputs.get("z")
                    linear_output = forward_outputs.get("linear_output")
                    if recon is None or logvar is None or latent is None:
                        logger.warning("Forward outputs missing required tensors; skipping sample.")
                        continue

                    loss_dict = model.compute_loss(
                        forward_outputs=forward_outputs,
                        y_st=y_st,
                        y_ph=y_ph,
                        y_raw=y_raw,
                        beta=beta_value,
                    )
                    loss_floats = {
                        key: float(val.detach().cpu().item()) if isinstance(val, torch.Tensor) else float(val)
                        for key, val in loss_dict.items()
                    }

                    kld_tensor = self._kld_tensor_from_forward(forward_outputs)
                    if kld_tensor is None:
                        kld_tensor_np = np.zeros((latent.shape[-1], y_st.shape[1]))
                        kld_mean_np = np.zeros(y_st.shape[1])
                    else:
                        kld_sample = kld_tensor[0]
                        kld_tensor_np = kld_sample.detach().cpu().numpy().T
                        kld_mean_np = kld_sample.mean(dim=-1).detach().cpu().numpy()

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
                    logvar_np = logvar[0].detach().cpu().numpy()

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

    def run_latent_ablation_test(
        self,
        test_loader,
        *,
        num_samples: int = 16,
        dims: Optional[List[int]] = None,
        visuals_per_dim: int = 1,
    ) -> None:
        """Zero-out selected latent dimensions and evaluate reconstruction impact."""
        if num_samples <= 0:
            logger.info("latent ablation skipped (num_samples <= 0)")
            return
        if self.pytorch_model is None:
            self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model unavailable; cannot run latent ablation.")
            return
        decoder = getattr(self.pytorch_model, "decoder", None)
        if decoder is None:
            logger.error("SeqVAE core does not expose a decoder attribute; cannot ablate latents.")
            return
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        stats = self._get_normalization_stats(test_loader)
        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        beta_value = float(vae_cfg.get("kld_beta", 1.0))

        out_root = Path(self.test_results_dir) / "latent_ablation"
        out_root.mkdir(parents=True, exist_ok=True)

        dims_to_eval: Optional[List[int]] = None
        dim_metrics: Dict[int, Dict[str, List[float]]] = {}
        visual_counts: Dict[int, int] = {}
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
                    up_raw = up_raw_tensor[idx : idx + 1].to(device) if up_raw_tensor is not None else torch.zeros_like(y_raw)

                    outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                    latent = outputs.get("z")
                    recon = outputs.get("mu_pr")
                    logvar = outputs.get("logvar_pr")
                    linear_output = outputs.get("linear_output")
                    if latent is None or recon is None or logvar is None or linear_output is None:
                        logger.warning("Forward outputs missing latent/decoder tensors; skipping sample.")
                        continue

                    dims_source = latent.size(-1)
                    if dims_to_eval is None:
                        dims_to_eval = dims if dims else list(range(dims_source))
                        for dim in dims_to_eval:
                            dim_metrics[dim] = {"vaf": [], "mse": [], "snr": [], "delta_loss": []}
                            visual_counts[dim] = 0

                    metrics_base = self._compute_basic_metrics(y_raw, recon)
                    if metrics_base is None:
                        continue
                    loss_base = model.compute_loss(
                        forward_outputs=outputs,
                        y_st=y_st,
                        y_ph=y_ph,
                        y_raw=y_raw,
                        beta=beta_value,
                    )
                    base_recon_loss = float(loss_base["reconstruction_loss"].detach().cpu().item())

                    for dim in dims_to_eval:
                        latent_mod = latent.clone()
                        latent_mod[:, :, dim] = 0.0
                        linear_mod, recon_mod, logvar_mod = decoder(latent_mod)
                        metrics_mod = self._compute_basic_metrics(y_raw, recon_mod)
                        if metrics_mod is None:
                            continue

                        mutated_outputs = dict(outputs)
                        mutated_outputs["linear_output"] = linear_mod
                        mutated_outputs["mu_pr"] = recon_mod
                        mutated_outputs["logvar_pr"] = logvar_mod
                        loss_mod = model.compute_loss(
                            forward_outputs=mutated_outputs,
                            y_st=y_st,
                            y_ph=y_ph,
                            y_raw=y_raw,
                            beta=beta_value,
                        )
                        delta_loss = float(
                            loss_mod["reconstruction_loss"].detach().cpu().item() - base_recon_loss
                        )

                        dim_metrics[dim]["vaf"].extend(metrics_mod["vaf"].detach().cpu().tolist())
                        dim_metrics[dim]["mse"].extend(metrics_mod["mse"].detach().cpu().tolist())
                        dim_metrics[dim]["snr"].extend(metrics_mod["snr"].detach().cpu().tolist())
                        dim_metrics[dim]["delta_loss"].append(delta_loss)

                        # Save qualitative plots if requested
                        if visual_counts[dim] < max(0, int(visuals_per_dim)):
                            visual_counts[dim] += 1
                            self._save_ablation_visual(
                                out_root / f"dim_{dim:02d}",
                                sample_idx=processed,
                                dim_index=dim,
                                y_st=y_st,
                                y_ph=y_ph,
                                x_ph=x_ph,
                                y_raw=y_raw,
                                up_raw=up_raw,
                                recon=recon_mod,
                                logvar=logvar_mod,
                                linear_output=linear_mod,
                                latent=latent_mod,
                                stats=stats,
                                loss_dict=loss_mod,
                                label_suffix="ablate",
                            )

                    processed += 1
                if processed >= num_samples:
                    break

        if not dim_metrics:
            logger.warning("No latent ablation metrics were collected.")
            return

        for dim, metrics in dim_metrics.items():
            if not metrics["vaf"]:
                continue
            dim_dir = out_root / f"dim_{dim:02d}"
            dim_dir.mkdir(parents=True, exist_ok=True)
            plot_metrics_histograms(
                metrics["vaf"],
                metrics["mse"],
                metrics["snr"],
                metrics["delta_loss"],
                dim_dir,
            )
            logger.info(
                "Latent dim %d -> mean VAF=%.3f, MSE=%.5f, Δrecon=%.5f",
                dim,
                float(np.mean(metrics["vaf"])),
                float(np.mean(metrics["mse"])),
                float(np.mean(metrics["delta_loss"])),
            )

    def _save_ablation_visual(
        self,
        output_dir: Path,
        *,
        sample_idx: int,
        dim_index: int,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
        y_raw: torch.Tensor,
        up_raw: torch.Tensor,
        recon: torch.Tensor,
        logvar: torch.Tensor,
        linear_output: torch.Tensor,
        latent: torch.Tensor,
        stats: Optional[Dict],
        loss_dict: Dict[str, torch.Tensor],
        label_suffix: str = "",
    ) -> None:
        """Store qualitative reconstruction plot for a latent-ablation sample."""
        output_dir.mkdir(parents=True, exist_ok=True)
        fhr_norm = y_raw[0]
        up_norm = up_raw[0]
        fhr_denorm = self._maybe_denormalize(fhr_norm, "fhr", stats)
        up_denorm = self._maybe_denormalize(up_norm, "up", stats)

        fhr_norm_np = fhr_norm.detach().cpu().numpy()
        up_norm_np = up_norm.detach().cpu().numpy()
        fhr_denorm_np = fhr_denorm.detach().cpu().numpy()
        up_denorm_np = up_denorm.detach().cpu().numpy()

        fhr_st_np = y_st[0].detach().cpu().numpy().T
        fhr_ph_np = y_ph[0].detach().cpu().numpy().T
        fhr_up_ph_np = x_ph[0].detach().cpu().numpy().T
        latent_np = latent[0].detach().cpu().numpy().T
        recon_np = recon[0].detach().cpu().numpy()
        logvar_np = logvar[0].detach().cpu().numpy()

        recon_st_np, recon_ph_np = self._extract_reconstruction_features(
            linear_output,
            st_channels=fhr_st_np.shape[0],
            ph_channels=fhr_ph_np.shape[0],
            seq_len=fhr_st_np.shape[1],
        )

        loss_floats = {
            key: float(val.detach().cpu().item()) if isinstance(val, torch.Tensor) else float(val)
            for key, val in loss_dict.items()
        }

        extra = f"_{label_suffix}" if label_suffix else ""
        tag = f"sample{sample_idx:03d}_dim{dim_index:02d}{extra}"
        plot_model_analysis(
            output_dir=str(output_dir),
            raw_fhr=fhr_denorm_np,
            raw_up=up_denorm_np,
            fhr_st=fhr_st_np,
            fhr_ph=fhr_ph_np,
            fhr_up_ph=fhr_up_ph_np,
            latent_z=latent_np,
            reconstructed_fhr_mu=recon_np,
            reconstructed_fhr_logvar=logvar_np,
            kld_tensor=np.zeros_like(latent_np),
            kld_mean_over_channels=np.zeros(latent_np.shape[1]),
            batch_idx=f"{tag}_analysis",
            loss_dict=loss_floats,
            raw_fhr_normalized=fhr_norm_np,
            raw_up_normalized=up_norm_np,
        )

        plot_vae_reconstruction(
            output_dir=str(output_dir),
            raw_fhr_unnormalized=fhr_denorm_np,
            raw_up_unnormalized=up_denorm_np,
            raw_fhr_normalized=fhr_norm_np,
            raw_up_normalized=up_norm_np,
            reconstructed_fhr=recon_np,
            original_scattering_transform=fhr_st_np,
            reconstructed_scattering_transform=recon_st_np,
            original_phase_harmonic=fhr_ph_np,
            reconstructed_phase_harmonic=recon_ph_np,
            scattering_channel_data=None,
            batch_idx=f"{tag}_recon",
            loss_dict=loss_floats,
        )

    def run_latent_magnitude_sweep(
        self,
        test_loader,
        *,
        num_samples: int = 16,
        dims: Optional[List[int]] = None,
        scales: Optional[List[float]] = None,
        visuals_per_dim: int = 1,
    ) -> None:
        """Scale latent dimensions and track reconstruction metrics."""
        if num_samples <= 0:
            logger.info("latent magnitude sweep skipped (num_samples <= 0)")
            return
        if self.pytorch_model is None:
            self.create_model()
        decoder = getattr(self.pytorch_model, "decoder", None)
        if decoder is None:
            logger.error("SeqVAE core does not expose a decoder attribute; cannot run sweep.")
            return
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        stats = self._get_normalization_stats(test_loader)
        beta_value = float(self.config.get("model_config", {}).get("VAE_model", {}).get("kld_beta", 1.0))
        sweep_scales = scales if scales else [0.0, 0.5, 1.0, 1.5, 2.0]
        out_root = Path(self.test_results_dir) / "latent_sweep"
        out_root.mkdir(parents=True, exist_ok=True)

        dim_metrics: Dict[int, Dict[str, Dict[float, List[float]]]] = {}
        visual_counts: Dict[int, int] = {}
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
                    up_raw = up_raw_tensor[idx : idx + 1].to(device) if up_raw_tensor is not None else torch.zeros_like(y_raw)

                    outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                    latent = outputs.get("z")
                    if latent is None:
                        continue
                    dims_to_eval = dims if dims else list(range(latent.size(-1)))
                    for dim in dims_to_eval:
                        dim_metrics.setdefault(dim, {})
                        for metric_name in ("vaf", "mse", "snr"):
                            dim_metrics[dim].setdefault(metric_name, {})
                        visual_counts.setdefault(dim, 0)
                        for scale in sweep_scales:
                            latent_mod = latent.clone()
                            latent_mod[:, :, dim] = latent[:, :, dim] * scale
                            linear_mod, recon_mod, logvar_mod = decoder(latent_mod)
                            metrics_mod = self._compute_basic_metrics(y_raw, recon_mod)
                            if metrics_mod is None:
                                continue
                            for metric_name in ("vaf", "mse", "snr"):
                                dim_metrics[dim][metric_name].setdefault(scale, []).extend(
                                    metrics_mod[metric_name].detach().cpu().tolist()
                                )
                            if visual_counts[dim] < max(0, int(visuals_per_dim)) and scale != 1.0:
                                visual_counts[dim] += 1
                                mutated_outputs = dict(outputs)
                                mutated_outputs["linear_output"] = linear_mod
                                mutated_outputs["mu_pr"] = recon_mod
                                mutated_outputs["logvar_pr"] = logvar_mod
                                loss_mod = model.compute_loss(
                                    forward_outputs=mutated_outputs,
                                    y_st=y_st,
                                    y_ph=y_ph,
                                    y_raw=y_raw,
                                    beta=beta_value,
                                )
                                self._save_ablation_visual(
                                    out_root / f"dim_{dim:02d}",
                                    sample_idx=processed,
                                    dim_index=dim,
                                    y_st=y_st,
                                    y_ph=y_ph,
                                    x_ph=x_ph,
                                    y_raw=y_raw,
                                    up_raw=up_raw,
                                    recon=recon_mod,
                                    logvar=logvar_mod,
                                    linear_output=linear_mod,
                                    latent=latent_mod,
                                    stats=stats,
                                    loss_dict=loss_mod,
                                    label_suffix=f"scale{scale}",
                                )
                    processed += 1
                if processed >= num_samples:
                    break

        if not dim_metrics:
            logger.warning("No latent sweep metrics were collected.")
            return
        for dim, metric_map in dim_metrics.items():
            dim_dir = out_root / f"dim_{dim:02d}"
            dim_dir.mkdir(parents=True, exist_ok=True)
            self._plot_latent_metric_curves(metric_map, sweep_scales, dim_dir / "metric_curves.png")
            logger.info("Latent sweep plotted for dimension %d -> %s", dim, dim_dir)

    def run_latent_interpolation(self, test_loader, *, pair_count: int = 10, steps: int = 11) -> None:
        """Create Bokeh animations showing interpolation between sample pairs."""
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
        out_dir = Path(self.test_results_dir) / "latent_interpolation_bokeh"
        out_dir.mkdir(parents=True, exist_ok=True)

        required_samples = pair_count * 2
        samples: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        metadata_available = True
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
                        metadata_available = False
                    if len(samples) >= required_samples:
                        break
                if len(samples) >= required_samples:
                    break
        if len(samples) < 2:
            logger.error("Insufficient samples for interpolation.")
            return

        if metadata_available and grouped:
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

        actual_pairs = min(pair_count, len(sample_pairs))
        weights = np.linspace(0.0, 1.0, steps)
        for pair_idx in range(actual_pairs):
            sample_a, sample_b = sample_pairs[pair_idx]
            with torch.inference_mode():
                outputs_a = model(y_st=sample_a["fhr_st"], y_ph=sample_a["fhr_ph"], x_ph=sample_a["fhr_up_ph"])
                outputs_b = model(y_st=sample_b["fhr_st"], y_ph=sample_b["fhr_ph"], x_ph=sample_b["fhr_up_ph"])
            latent_a = outputs_a.get("z")
            latent_b = outputs_b.get("z")
            if latent_a is None or latent_b is None:
                logger.warning("Skipping pair %d due to missing latents.", pair_idx)
                continue
            y_a = sample_a["fhr"]
            y_b = sample_b["fhr"]
            recon_store: List[np.ndarray] = []
            target_store: List[np.ndarray] = []
            latent_store: List[np.ndarray] = []
            heatmaps: List[List[List[float]]] = []
            time_axis = np.arange(y_a.shape[-1])
            latent_dim = latent_a.size(-1)
            sequence_len = latent_a.size(1)
            for alpha in weights:
                latent_interp = latent_a * (1.0 - alpha) + latent_b * alpha
                decoded = decoder(latent_interp)
                recon_interp = decoded[1]
                target_interp = y_a * (1.0 - alpha) + y_b * alpha
                recon_store.append(recon_interp[0].detach().cpu().numpy())
                target_store.append(target_interp[0].detach().cpu().numpy())
                latent_mean = latent_interp.mean(dim=1)[0].detach().cpu().numpy()
                latent_store.append(latent_mean)
                heatmaps.append(latent_interp[0].detach().cpu().numpy().tolist())

            x_values = time_axis.tolist()
            recon_series = [arr.tolist() for arr in recon_store]
            target_series = [arr.tolist() for arr in target_store]
            latent_series = [arr.tolist() for arr in latent_store]

            signal_source = ColumnDataSource(
                data=dict(
                    x=x_values,
                    recon=recon_series[0],
                    target=target_series[0],
                )
            )
            latent_source = ColumnDataSource(
                data=dict(dim=np.arange(latent_dim).tolist(), value=latent_series[0])
            )
            signal_fig = figure(
                title=f"Pair {pair_idx} - Raw vs Reconstruction",
                width=900,
                height=300,
                x_axis_label="Time Index",
                y_axis_label="Amplitude",
            )
            signal_fig.line("x", "target", source=signal_source, color="#1b9e77", legend_label="Target", line_width=2)
            signal_fig.line("x", "recon", source=signal_source, color="#d95f02", legend_label="Reconstruction", line_width=2)
            signal_fig.legend.location = "top_right"
            signal_fig.legend.click_policy = "hide"

            latent_fig = figure(
                title=f"Pair {pair_idx} - Latent Mean by Dimension",
                width=900,
                height=250,
                x_axis_label="Latent Dimension",
                y_axis_label="Mean Value",
            )
            latent_fig.vbar(x="dim", top="value", width=0.8, source=latent_source, color="#7570b3")

            heat_min = min(float(np.min(np.array(arr))) for arr in heatmaps)
            heat_max = max(float(np.max(np.array(arr))) for arr in heatmaps)
            color_mapper = LinearColorMapper(palette=Viridis256, low=heat_min, high=heat_max)
            heat_source = ColumnDataSource(data=dict(image=[heatmaps[0]]))
            heat_fig = figure(
                title=f"Pair {pair_idx} - Latent Heatmap",
                width=900,
                height=300,
                x_axis_label="Latent Dimension",
                y_axis_label="Time Step",
                y_range=(sequence_len, 0),
            )
            heat_fig.image(
                source=heat_source,
                image="image",
                x=0,
                y=0,
                dw=latent_dim,
                dh=sequence_len,
                color_mapper=color_mapper,
            )
            heat_fig.add_layout(ColorBar(color_mapper=color_mapper, width=10, label_standoff=8), "right")

            slider = Slider(start=0, end=len(weights) - 1, value=0, step=1, title="Interpolation Step")
            callback = CustomJS(
                args=dict(
                    source_signal=signal_source,
                    source_latent=latent_source,
                    source_heat=heat_source,
                    recon_data=recon_series,
                    target_data=target_series,
                    latent_data=latent_series,
                    heat_data=heatmaps,
                ),
                code="""
                    const idx = Math.round(cb_obj.value);
                    source_signal.data['recon'] = recon_data[idx];
                    source_signal.data['target'] = target_data[idx];
                    source_signal.change.emit();
                    source_latent.data['value'] = latent_data[idx];
                    source_latent.change.emit();
                    source_heat.data['image'] = [heat_data[idx]];
                    source_heat.change.emit();
                """,
            )
            slider.js_on_change("value", callback)

            layout = column(signal_fig, latent_fig, heat_fig, slider)
            output_file(str(out_dir / f"latent_interp_pair_{pair_idx:02d}.html"), title=f"Latent Interpolation Pair {pair_idx}")
            save(layout)
        logger.info("Latent interpolation animations saved to %s", out_dir)

    def run_latent_feature_attribution(
        self,
        test_loader,
        *,
        num_samples: int = 8,
    ) -> None:
        """Approximate latent importance via gradients of reconstruction loss."""
        if num_samples <= 0:
            logger.info("Feature attribution skipped (num_samples <= 0)")
            return
        if self.pytorch_model is None:
            self.create_model()
        decoder = getattr(self.pytorch_model, "decoder", None)
        if decoder is None:
            logger.error("SeqVAE core does not expose a decoder attribute; cannot run attribution.")
            return
        device = torch.device(
            f"cuda:{self.cuda_devices[0]}" if torch.cuda.is_available() and self.cuda_devices else "cpu"
        )
        model = self.pytorch_model.to(device)
        model.eval()
        out_dir = Path(self.test_results_dir) / "latent_feature_attribution"
        out_dir.mkdir(parents=True, exist_ok=True)

        grad_accum: Optional[torch.Tensor] = None
        processed = 0
        for batch in test_loader:
            batch_size = batch.fhr_st.size(0)
            for idx in range(batch_size):
                if processed >= num_samples:
                    break
                y_st = batch.fhr_st[idx : idx + 1].to(device)
                y_ph = batch.fhr_ph[idx : idx + 1].to(device)
                x_ph = batch.fhr_up_ph[idx : idx + 1].to(device)
                y_raw = batch.fhr[idx : idx + 1].to(device)

                with torch.no_grad():
                    outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
                    latent = outputs.get("z")
                if latent is None:
                    continue
                latent = latent.detach().clone().requires_grad_(True)
                decoder.zero_grad(set_to_none=True)
                reconstruction = decoder(latent)[1]
                loss = torch.mean((reconstruction - y_raw) ** 2)
                loss.backward()
                if latent.grad is None:
                    decoder.zero_grad(set_to_none=True)
                    continue
                grad_importance = latent.grad.detach().abs().mean(dim=(0, 1))
                if grad_accum is None:
                    grad_accum = grad_importance
                else:
                    grad_accum = grad_accum + grad_importance
                decoder.zero_grad(set_to_none=True)
                processed += 1
            if processed >= num_samples:
                break

        if grad_accum is None:
            logger.warning("No gradients collected for feature attribution.")
            return
        grad_mean = (grad_accum / processed).cpu().numpy()
        self._plot_feature_importance(grad_mean, out_dir / "latent_feature_importance.png")
        logger.info("Feature attribution plot saved to %s", out_dir)

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
        "--latent-ablation-samples",
        type=int,
        default=0,
        help="Number of samples to run latent ablation on (0 = skip).",
    )
    parser.add_argument(
        "--latent-ablation-dims",
        type=str,
        default=None,
        help="Comma-separated latent dimension indices to zero (default: all).",
    )
    parser.add_argument(
        "--latent-ablation-visuals",
        type=int,
        default=1,
        help="Number of reconstruction plots to save per ablated dimension.",
    )
    parser.add_argument(
        "--latent-dist-samples",
        type=int,
        default=500,
        help="Samples to use when plotting latent distributions (0 = skip).",
    )
    parser.add_argument(
        "--latent-sweep-samples",
        type=int,
        default=0,
        help="Number of samples for latent magnitude sweep (0 = skip).",
    )
    parser.add_argument(
        "--latent-sweep-dims",
        type=str,
        default=None,
        help="Comma-separated latent dims for magnitude sweep (default: all).",
    )
    parser.add_argument(
        "--latent-sweep-scales",
        type=str,
        default=None,
        help="Comma-separated scale factors for sweep (default: 0.0,0.5,1.0,1.5,2.0).",
    )
    parser.add_argument(
        "--latent-sweep-visuals",
        type=int,
        default=1,
        help="Visuals per-dimension during latent sweep.",
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
        "--latent-attr-samples",
        type=int,
        default=8,
        help="Samples for latent feature attribution (0 = skip).",
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
    max_samples: Optional[int] = None,
    metrics_max_samples: Optional[int] = None,
    analysis_samples: int = 10,
    latent_ablation_samples: int = 0,
    latent_ablation_dims: Optional[Any] = None,
    latent_ablation_visuals: int = 1,
    latent_dist_samples: int = 1000,
    latent_sweep_samples: int = 0,
    latent_sweep_dims: Optional[Any] = None,
    latent_sweep_scales: Optional[Any] = None,
    latent_sweep_visuals: int = 1,
    latent_interp_pairs: int = 10,
    latent_interp_steps: int = 20,
    latent_attr_samples: int = 8,
) -> None:
    config_path = Path(config)
    config_data = load_config(config_path)
    tester = GraphModelVaeTebSmallTester(str(config_path))
    tester.setup_config()
    tester.create_model()
    test_loader = build_test_dataloader(config_data)
    if latent_dist_samples and latent_dist_samples > 0:
        tester.run_latent_distribution(test_loader, num_samples=latent_dist_samples)
    if analysis_samples and analysis_samples > 0:
        tester.run_analysis_and_plot(test_loader, num_samples=analysis_samples)
    ablation_dims_list = _parse_int_list(latent_ablation_dims)
    if latent_ablation_samples and latent_ablation_samples > 0:
        tester.run_latent_ablation_test(
            test_loader,
            num_samples=latent_ablation_samples,
            dims=ablation_dims_list,
            visuals_per_dim=latent_ablation_visuals,
        )
    sweep_dims_list = _parse_int_list(latent_sweep_dims)
    sweep_scale_list = _parse_float_list(latent_sweep_scales)
    if latent_sweep_samples and latent_sweep_samples > 0:
        tester.run_latent_magnitude_sweep(
            test_loader,
            num_samples=latent_sweep_samples,
            dims=sweep_dims_list,
            scales=sweep_scale_list,
            visuals_per_dim=latent_sweep_visuals,
        )
    if latent_interp_pairs and latent_interp_steps and latent_interp_steps >= 2:
        tester.run_latent_interpolation(
            test_loader,
            pair_count=latent_interp_pairs,
            steps=latent_interp_steps,
        )
    if latent_attr_samples and latent_attr_samples > 0:
        tester.run_latent_feature_attribution(test_loader, num_samples=latent_attr_samples)
    if max_samples is None or max_samples > 0 or (metrics_max_samples and metrics_max_samples > 0):
        tester.run_histogram_test(
            test_loader,
            num_samples=max_samples,
            max_samples=metrics_max_samples,
        )


if __name__ == "__main__":
    main(**vars(parse_args()))
