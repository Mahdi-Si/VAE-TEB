import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
import numpy as np
import torch.nn as nn
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple

from vae_teb_model import SeqVaeTeb, ensure_compiled_module, is_compiled_module

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON'] = "NO"

torch.backends.cudnn.enabled = True

from loguru import logger


# ------------------------------------------------------------------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------------------------------------------------------------------
class PlottingCallBack(Callback):
    def __init__(self, output_dir, plot_every_epoch, predictive_horizon: Optional[int] = None):
        super().__init__()
        self.output_dir = output_dir    
        self.plot_every_epoch = plot_every_epoch
        self.predictive_horizon = predictive_horizon

    def on_validation_epoch_end(self, pl_trainer, pl_module):
        if pl_trainer.current_epoch % self.plot_every_epoch != 0 or not pl_trainer.is_global_zero:
            return

        logger.info(f"Starting plotting callback for epoch {pl_trainer.current_epoch}")

        try:
            if hasattr(pl_trainer, 'datamodule') and pl_trainer.datamodule is not None:
                val_dataloader = pl_trainer.datamodule.val_dataloader()
            else:
                val_dataloader = pl_trainer.val_dataloaders
                if isinstance(val_dataloader, list):
                    val_dataloader = val_dataloader[0]

            batch = next(iter(val_dataloader))
            logger.info("Successfully fetched batch from validation dataloader")
        except (StopIteration, AttributeError, IndexError) as e:
            logger.warning(f"Could not get a batch from validation dataloader for plotting: {e}")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)

        pl_module.eval()
        try:
            with torch.no_grad():
                # Check if this is the correct Lightning module type
                if not isinstance(pl_module, LightSeqVaeTeb):
                    logger.warning(f"PlottingCallback received unexpected module type: {type(pl_module)}. Expected LightSeqVaeTeb.")
                    return

                logger.info("Accessing batch data...")
                # SPEED OPTIMIZATION: Data now comes pre-permuted from dataset - no permute needed
                # Optimized dataloader provides tensors in (batch, sequence, channels) format:
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph  # All (B, seq, channels)
                y_raw_normalized = batch.fhr  # (B, 4800)
                up_raw_normalized = batch.up  # (B, 4800)

                model_outputs = pl_module.model(y_st, y_ph, x_ph)
                mu_pr = model_outputs.get("mu_pr")
                logvar_pr = model_outputs.get("logvar_pr")
                latent_z_full = model_outputs.get("z")
                mu_post_full = model_outputs.get("mu_post")
                mu_prior_full = model_outputs.get("mu_prior")

                self._plot_reconstruction_overview(
                    y_raw_normalized=y_raw_normalized,
                    up_raw_normalized=up_raw_normalized,
                    mu_pr=mu_pr,
                    logvar_pr=logvar_pr,
                    latent_z=latent_z_full,
                    mu_post=mu_post_full,
                    mu_prior=mu_prior_full,
                    epoch=pl_trainer.current_epoch,
                )

                if pl_module.model.has_forecaster():
                    forecast_out = pl_module.model.forecast(
                        y_st, y_ph, x_ph, anchors=None, use_posterior_mean=True
                    )
                    anchors = forecast_out["anchors"]
                    mu_future = forecast_out["mu_future"]          # (B,N,480)
                    logvar_future = forecast_out["logvar_future"]  # (B,N,480)
                    z_future = forecast_out.get("z_future")
                    latent_logvar_future = forecast_out.get("latent_logvar_future")
                    enc = forecast_out["enc"]

                    stability_penalty = forecast_out.get("stability_penalty")
                    if stability_penalty is not None:
                        logger.info(
                            "LGSSM stability penalty at epoch %s: %.4e",
                            pl_trainer.current_epoch,
                            float(stability_penalty),
                        )

                    canvas_mu, mean_mu = pl_module.model.aggregate_forecasts_to_canvas(
                        mu_future, anchors, total_len=y_raw_normalized.shape[1], stride=pl_module.model.decimation_factor)

                    var_future = logvar_future.exp()
                    _, mean_var = pl_module.model.aggregate_forecasts_to_canvas(
                        var_future, anchors, total_len=y_raw_normalized.shape[1], stride=pl_module.model.decimation_factor)
                    std_mu = mean_var.clamp_min(1e-8).sqrt()

                    self._plot_latent_forecast_samples(
                        mu_post_sequence=enc.get("mu_post"),
                        z_future=z_future,
                        latent_logvar_future=latent_logvar_future,
                        anchors=anchors,
                        epoch=pl_trainer.current_epoch,
                    )
                    self._plot_channel_forecasts(
                        mu_post_sequence=enc.get("mu_post"),
                        z_future=z_future,
                        latent_logvar_future=latent_logvar_future,
                        anchors=anchors,
                        epoch=pl_trainer.current_epoch,
                    )
                    self._plot_latent_trajectory_analysis(
                        mu_post_sequence=enc.get("mu_post"),
                        mu_prior_sequence=enc.get("mu_prior"),
                        epoch=pl_trainer.current_epoch,
                    )

                    self._plot_latent_statistics(
                        mu_prior=mu_prior_full if mu_prior_full is not None else enc.get("mu_prior"),
                        mu_post=mu_post_full if mu_post_full is not None else enc.get("mu_post"),
                        epoch=pl_trainer.current_epoch,
                    )

                    self._plot_forecast_results(
                        y_raw_normalized,
                        mean_mu,
                        std_mu,
                        canvas_mu,
                        anchors,
                        enc.get('mu_post'),
                        pl_trainer.current_epoch)

                    try:
                        self._plot_batch_aggregated_forecast(
                            y_raw_batch=y_raw_normalized,
                            mean_mu_batch=mean_mu,
                            std_mu_batch=std_mu,
                            epoch=pl_trainer.current_epoch,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to plot batch aggregated forecast: {e}")
                else:
                    self._plot_latent_statistics(
                        mu_prior=mu_prior_full,
                        mu_post=mu_post_full,
                        epoch=pl_trainer.current_epoch,
                    )
        except Exception as e:
            logger.error(f"Error during plotting: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            pl_module.train()

    def _plot_reconstruction_overview(
        self,
        y_raw_normalized: torch.Tensor,
        up_raw_normalized: Optional[torch.Tensor],
        mu_pr: Optional[torch.Tensor],
        logvar_pr: Optional[torch.Tensor],
        latent_z: Optional[torch.Tensor],
        mu_post: Optional[torch.Tensor],
        mu_prior: Optional[torch.Tensor],
        epoch: int,
    ):
        """Plot ground-truth vs reconstruction along with latent diagnostics."""
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt

        if y_raw_normalized is None or mu_pr is None or logvar_pr is None:
            return

        batch_idx = 0
        try:
            y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
            recon = mu_pr[batch_idx].detach().cpu().numpy()
            logvar = logvar_pr[batch_idx].detach().cpu().numpy()
        except Exception:
            return

        std = np.exp(0.5 * logvar)
        diff = y_raw - recon

        up_raw = None
        if up_raw_normalized is not None:
            try:
                up_raw = up_raw_normalized[batch_idx].detach().cpu().numpy()
            except Exception:
                up_raw = None

        latent_matrix = None
        if latent_z is not None:
            try:
                latent_matrix = latent_z[batch_idx].detach().cpu().numpy()
            except Exception:
                latent_matrix = None
        if latent_matrix is None and mu_post is not None:
            try:
                latent_matrix = mu_post[batch_idx].detach().cpu().numpy()
            except Exception:
                latent_matrix = None

        prior_matrix = None
        if mu_prior is not None:
            try:
                prior_matrix = mu_prior[batch_idx].detach().cpu().numpy()
            except Exception:
                prior_matrix = None

        Fs = 4.0
        t = np.arange(y_raw.shape[0]) / Fs

        corr = float('nan')
        if np.std(y_raw) > 1e-8 and np.std(recon) > 1e-8:
            try:
                corr = float(np.corrcoef(y_raw, recon)[0, 1])
            except Exception:
                corr = float('nan')

        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))

        colors = {
            'fhr': "#055C9A",
            'up': "#0DD8A2",
            'gt': '#456882',
            'recon': '#BB3E00',
            'uncertainty': '#F7AD45',
            'background': '#F9F3EF'
        }

        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.linewidth': 0.7,
            'axes.edgecolor': "#9E9D9D",
            'axes.facecolor': colors['background'],
            'grid.color': "#838383",
            'grid.linewidth': 0.4,
            'grid.alpha': 0.6,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 0.95,
            'legend.edgecolor': '#A2B9A7',
            'legend.facecolor': colors['background'],
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.dpi': 300
        })

        n_rows = 4
        fig, ax = plt.subplots(
            nrows=n_rows, ncols=2, figsize=(20, n_rows * 3.2),
            gridspec_kw={"width_ratios": [80, 1]}, constrained_layout=True)

        for i in range(n_rows):
            ax[i, 0].grid(True, linestyle='-', alpha=0.35, linewidth=0.4, color='#D2C1B6')
            ax[i, 0].grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax[i, 0].minorticks_on()
            ax[i, 0].set_axisbelow(True)
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)
            ax[i, 0].spines['left'].set_color('#A2B9A7')
            ax[i, 0].spines['bottom'].set_color('#A2B9A7')
            ax[i, 0].spines['left'].set_linewidth(0.7)
            ax[i, 0].spines['bottom'].set_linewidth(0.7)

        ax[0, 1].set_axis_off()
        ax[0, 0].plot(t, y_raw, linewidth=1.2, color=colors['fhr'], label='FHR', alpha=0.9)
        if up_raw is not None:
            ax[0, 0].plot(t, up_raw, linewidth=1.0, color=colors['up'], label='UP', alpha=0.75)
        ax[0, 0].set_ylabel('Amplitude')
        ax[0, 0].set_title('Raw FHR/UP Signals')
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)
        ax[0, 0].legend(loc='upper right', framealpha=0.95)

        ax[1, 1].set_axis_off()
        ax[1, 0].plot(t, y_raw, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.9, zorder=3)
        ax[1, 0].plot(t, recon, linewidth=1.3, color=colors['recon'], label='Reconstruction', alpha=0.85, zorder=2)
        ax[1, 0].fill_between(t, recon - std, recon + std, color=colors['uncertainty'], alpha=0.25, label='+/- 1 sigma')
        ax[1, 0].set_ylabel('FHR (bpm)')
        ax[1, 0].set_title('Ground Truth vs Reconstruction')
        ax[1, 0].legend(loc='upper right', framealpha=0.95)
        ax[1, 0].autoscale(enable=True, axis='x', tight=True)
        ax[1, 0].text(
            0.01,
            0.92,
            f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nCorr: {corr:.3f}",
            transform=ax[1, 0].transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        ax[2, 0].set_axis_off()
        ax[2, 1].set_axis_off()

        if prior_matrix is not None and latent_matrix is not None and ax.shape[1] > 1:
            ax[2, 1].set_axis_on()
            step_idx = np.arange(latent_matrix.shape[0])
            prior_norm = np.linalg.norm(prior_matrix, axis=1)
            post_norm = np.linalg.norm(latent_matrix, axis=1)
            delta_norm = np.linalg.norm(latent_matrix - prior_matrix, axis=1)
            ax[2, 1].plot(step_idx, prior_norm, color='#7F8C8D', linewidth=1.0, label='||mu_prior||')
            ax[2, 1].plot(step_idx, post_norm, color='#BB3E00', linewidth=1.2, label='||mu_post||')
            ax[2, 1].plot(step_idx, delta_norm, color='#2E86AB', linewidth=1.2, linestyle='--', label='||delta||')
            ax[2, 1].set_title('Latent Norm Dynamics')
            ax[2, 1].set_xlabel('Decimated step')
            ax[2, 1].grid(True, alpha=0.3)
            ax[2, 1].legend(loc='upper right', fontsize=9, framealpha=0.9)

        ax[3, 1].set_axis_off()
        if latent_matrix is not None:
            img = ax[3, 0].imshow(latent_matrix.T, aspect='auto', cmap='RdBu_r', origin='lower')
            ax[3, 0].set_ylabel('Latent dim')
            ax[3, 0].set_xlabel('Decimated step')
            title_suffix = ''
            if prior_matrix is not None:
                mean_delta = float(np.mean(np.abs(latent_matrix - prior_matrix)))
                title_suffix = f' | mean |delta| = {mean_delta:.3f}'
            ax[3, 0].set_title(f'Posterior Latent Trajectory{title_suffix}')
            cbar = fig.colorbar(img, cax=ax[3, 1])
            cbar.ax.tick_params(labelsize=9, colors='#666666')
            cbar.set_label('Activation', fontsize=10, color='#666666')
        else:
            ax[3, 0].text(0.5, 0.5, 'Latents not available', ha='center', va='center', fontsize=12)
            ax[3, 0].set_axis_off()

        fig.suptitle(f'Reconstruction Overview - Epoch {epoch}', fontsize=14, color='#456882')
        save_path = os.path.join(self.output_dir, f'reconstruction_overview_epoch_{epoch}.pdf')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Reconstruction plot saved to {save_path}")

    def _plot_latent_forecast_samples(
        self,
        mu_post_sequence: Optional[torch.Tensor],
        z_future: Optional[torch.Tensor],
        latent_logvar_future: Optional[torch.Tensor],
        anchors: Optional[torch.Tensor],
        epoch: int,
    ):
        """Visualize forecasted latent trajectories against ground truth for spaced anchors."""
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt

        if mu_post_sequence is None or z_future is None or anchors is None or anchors.numel() == 0:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post_sequence[batch_idx].detach().cpu().numpy()
            z_future_np = z_future[batch_idx].detach().cpu().numpy()
            anchors_np = anchors.detach().cpu().numpy().astype(int)
            latent_std_np = None
            if latent_logvar_future is not None:
                latent_std_np = np.sqrt(
                    np.exp(latent_logvar_future[batch_idx].detach().cpu().numpy())
                )
        except Exception:
            return

        if mu_post_np.ndim != 2 or z_future_np.ndim != 3:
            return

        horizon = z_future_np.shape[1]
        step = 30
        selected = []
        last_anchor = -step
        for idx, anchor in enumerate(anchors_np):
            if anchor - last_anchor >= step and anchor + 1 + horizon <= mu_post_np.shape[0]:
                selected.append((idx, anchor))
                last_anchor = anchor
            if len(selected) >= 4:
                break

        if not selected:
            for idx, anchor in enumerate(anchors_np[:4]):
                if anchor + 1 + horizon <= mu_post_np.shape[0]:
                    selected.append((idx, anchor))

        if not selected:
            return

        global_max = 1e-6
        valid_segments = []
        for idx, anchor in selected:
            start = anchor + 1
            end = start + horizon
            gt = mu_post_np[start:end]
            if gt.shape[0] != horizon:
                continue
            pred = z_future_np[idx]
            global_max = max(global_max, np.abs(gt).max(), np.abs(pred).max())
            std_seg = latent_std_np[idx] if latent_std_np is not None else None
            valid_segments.append((idx, anchor, gt, pred, std_seg))

        if not valid_segments:
            return

        n_rows = len(valid_segments)
        fig, axes = plt.subplots(
            n_rows, 4, figsize=(22, n_rows * 3.2),
            gridspec_kw={'width_ratios': [1, 1, 1, 1.3]},
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, (idx, anchor, gt, pred, std_segment) in enumerate(valid_segments):
            err = pred - gt

            im0 = axes[row, 0].imshow(gt.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-global_max, vmax=global_max)
            axes[row, 0].set_title(f'Ground truth mu_post | anchor={anchor}')
            axes[row, 0].set_ylabel('Latent dim')
            axes[row, 0].set_xlabel('Forecast step')

            im1 = axes[row, 1].imshow(pred.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-global_max, vmax=global_max)
            axes[row, 1].set_title('Forecast mu_future')
            axes[row, 1].set_xlabel('Forecast step')

            abs_err = np.abs(err)
            im2 = axes[row, 2].imshow(abs_err.T, aspect='auto', origin='lower', cmap='magma')
            axes[row, 2].set_title('Absolute error')
            axes[row, 2].set_xlabel('Forecast step')

            for c in range(3):
                axes[row, c].grid(False)

            if abs_err.size > 0:
                per_channel_energy = abs_err.sum(axis=0)
                best_channel = int(np.argmax(per_channel_energy))
            else:
                best_channel = 0
            time_axis = np.arange(horizon)
            axes[row, 3].plot(time_axis, gt[:, best_channel], color='#2E86AB', linewidth=1.4, label='GT')
            axes[row, 3].plot(time_axis, pred[:, best_channel], color='#BB3E00', linewidth=1.2, linestyle='--', label='Forecast')
            if std_segment is not None and best_channel < std_segment.shape[1]:
                std_vec = std_segment[:, best_channel]
                upper = pred[:, best_channel] + 1.96 * std_vec
                lower = pred[:, best_channel] - 1.96 * std_vec
                axes[row, 3].fill_between(
                    time_axis,
                    lower,
                    upper,
                    color='#BB3E00',
                    alpha=0.18,
                    label='Forecast +/- 1.96 std' if row == 0 else None,
                )
            axes[row, 3].fill_between(
                time_axis,
                gt[:, best_channel],
                pred[:, best_channel],
                color='#F5B7B1',
                alpha=0.3,
            )
            axes[row, 3].set_title(f'Latent dim {best_channel} trajectory')
            axes[row, 3].set_xlabel('Forecast step')
            axes[row, 3].set_ylabel('Activation')
            axes[row, 3].grid(True, alpha=0.3)
            axes[row, 3].legend(loc='upper right', fontsize=8, framealpha=0.85)

            if row == 0:
                fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        save_path = os.path.join(self.output_dir, f'latent_forecast_samples_epoch_{epoch}.pdf')
        fig.suptitle(f'Latent Forecast Diagnostics - Epoch {epoch}', fontsize=14, color='#456882')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent forecast comparison saved to {save_path}")

    def _plot_channel_forecasts(
        self,
        mu_post_sequence: Optional[torch.Tensor],
        z_future: Optional[torch.Tensor],
        latent_logvar_future: Optional[torch.Tensor],
        anchors: Optional[torch.Tensor],
        epoch: int,
    ) -> None:
        """Plot per-dimension latent trajectories for specific anchors (75 and 224)."""
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt

        if mu_post_sequence is None or z_future is None or anchors is None or anchors.numel() == 0:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post_sequence[batch_idx].detach().cpu().numpy()
            z_future_np = z_future[batch_idx].detach().cpu().numpy()
            anchors_np = anchors.detach().cpu().numpy().astype(int)
            std_future_np = None
            if latent_logvar_future is not None:
                std_future_np = np.sqrt(
                    np.exp(latent_logvar_future[batch_idx].detach().cpu().numpy())
                )
        except Exception:
            return

        if mu_post_np.ndim != 2 or z_future_np.ndim != 3:
            return

        horizon = z_future_np.shape[1]
        latent_dim = z_future_np.shape[2]
        desired_anchors = [75, 224]
        anchor_pairs = []
        for anchor_val in desired_anchors:
            if anchor_val in anchors_np:
                idx_anchor = int(np.where(anchors_np == anchor_val)[0][0])
                start = anchor_val + 1
                end = start + horizon
                if end <= mu_post_np.shape[0]:
                    std_slice = std_future_np[idx_anchor] if std_future_np is not None else None
                    anchor_pairs.append((anchor_val, mu_post_np[start:end], z_future_np[idx_anchor], std_slice))

        if len(anchor_pairs) != len(desired_anchors):
            return

        fig, axes = plt.subplots(
            latent_dim,
            len(anchor_pairs),
            figsize=(len(anchor_pairs) * 5.5, latent_dim * 1.6),
            sharex=True,
            constrained_layout=True,
        )
        if latent_dim == 1:
            axes = axes.reshape(1, -1)

        time_axis = np.arange(horizon)
        colors = {
            'gt': '#2E86AB',
            'pred': '#BB3E00',
        }

        for col, (anchor_val, gt, pred, std_block) in enumerate(anchor_pairs):
            for row in range(latent_dim):
                ax = axes[row, col]
                ax.plot(time_axis, gt[:, row], color=colors['gt'], linewidth=1.2, label='GT')
                ax.plot(time_axis, pred[:, row], color=colors['pred'], linewidth=1.0, linestyle='--', label='Forecast')
                if std_block is not None:
                    std_vec = std_block[:, row]
                    upper = pred[:, row] + 1.96 * std_vec
                    lower = pred[:, row] - 1.96 * std_vec
                    ax.fill_between(
                        time_axis,
                        lower,
                        upper,
                        color=colors['pred'],
                    alpha=0.18,
                    label='Forecast +/- 1.96 std' if (row == 0 and col == 0) else None,
                    )
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(f'Anchor {anchor_val}')
                if col == 0:
                    ax.set_ylabel(f'Latent {row}')
                if row == latent_dim - 1:
                    ax.set_xlabel('Forecast step')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=2)

        save_path = os.path.join(self.output_dir, f'latent_forecast_channels_epoch_{epoch}.pdf')
        fig.suptitle(f'Per-channel Latent Forecasts - Epoch {epoch}', fontsize=14, color='#456882')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent forecast per-channel plot saved to {save_path}")
    def _plot_latent_trajectory_analysis(
        self,
        mu_post_sequence: Optional[torch.Tensor],
        mu_prior_sequence: Optional[torch.Tensor],
        epoch: int,
    ) -> None:
        """Comprehensive latent trajectory diagnostics for a single validation sample."""
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt

        if mu_post_sequence is None or mu_post_sequence.numel() == 0:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post_sequence[batch_idx].detach().cpu().numpy()
        except Exception:
            return

        if mu_post_np.ndim != 2:
            return

        prior_np = None
        if mu_prior_sequence is not None:
            try:
                prior_np = mu_prior_sequence[batch_idx].detach().cpu().numpy()
                if prior_np.ndim != 2 or prior_np.shape != mu_post_np.shape:
                    prior_np = None
            except Exception:
                prior_np = None

        time_axis = np.arange(mu_post_np.shape[0])
        latent_dim = mu_post_np.shape[1]
        if latent_dim == 0:
            return

        if prior_np is not None:
            delta_np = mu_post_np - prior_np
            ranking_signal = delta_np.var(axis=0)
        else:
            delta_np = None
            ranking_signal = mu_post_np.var(axis=0)
        order = np.argsort(ranking_signal)[::-1]
        top_k = int(min(latent_dim, 8))
        total_rows = top_k + 1

        fig, axes = plt.subplots(
            total_rows,
            2,
            figsize=(12, 2.2 * total_rows),
            constrained_layout=True,
        )
        axes = np.asarray(axes)

        colors = {'posterior': '#1f77b4', 'prior': '#ff7f0e', 'delta': '#2ca02c'}
        legend_handles = []
        legend_labels = []

        for row, dim in enumerate(order[:top_k]):
            ax_left = axes[row, 0]
            line_post, = ax_left.plot(time_axis, mu_post_np[:, dim], color=colors['posterior'], linewidth=1.2, label='mu_post')
            if 'mu_post' not in legend_labels:
                legend_handles.append(line_post)
                legend_labels.append('mu_post')
            if prior_np is not None:
                line_prior, = ax_left.plot(time_axis, prior_np[:, dim], color=colors['prior'], linewidth=1.0, linestyle='--', label='mu_prior')
                if 'mu_prior' not in legend_labels:
                    legend_handles.append(line_prior)
                    legend_labels.append('mu_prior')
            ax_left.grid(True, alpha=0.3)
            if row == 0:
                ax_left.set_title('Latent trajectory')
            if row == top_k - 1:
                ax_left.set_xlabel('Decimated step')
            ax_left.set_ylabel(f'z[{dim}]')

            ax_right = axes[row, 1]
            if delta_np is not None:
                line_delta, = ax_right.plot(time_axis, delta_np[:, dim], color=colors['delta'], linewidth=1.2, label='delta')
                if row == 0:
                    ax_right.set_title('Delta (mu_post - mu_prior)')
            else:
                centered = mu_post_np[:, dim] - mu_post_np[:, dim].mean()
                line_delta, = ax_right.plot(time_axis, centered, color=colors['delta'], linewidth=1.2, label='delta')
                if row == 0:
                    ax_right.set_title('Centered trajectory')
            if 'delta' not in legend_labels:
                legend_handles.append(line_delta)
                legend_labels.append('delta')
            ax_right.grid(True, alpha=0.3)
            if row == top_k - 1:
                ax_right.set_xlabel('Decimated step')

        energy = np.sum(mu_post_np ** 2, axis=0)
        summary_ax = axes[top_k, 0]
        summary_ax.bar(np.arange(latent_dim), energy, color='#345995')
        summary_ax.set_title('Posterior energy per latent')
        summary_ax.set_xlabel('Latent dim')
        summary_ax.set_ylabel('Energy')
        summary_ax.grid(True, axis='y', alpha=0.3)

        heat_ax = axes[top_k, 1]
        heat_data = mu_post_np[:, order[:top_k]].T
        im = heat_ax.imshow(heat_data, aspect='auto', origin='lower', cmap='RdBu_r')
        heat_ax.set_title('Posterior heatmap (top dims)')
        heat_ax.set_xlabel('Decimated step')
        heat_ax.set_yticks(range(top_k))
        heat_ax.set_yticklabels([f'z[{dim}]' for dim in order[:top_k]])
        fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04, label='Activation')

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_handles))

        overall_energy = float(np.linalg.norm(mu_post_np))
        if delta_np is not None:
            delta_energy = float(np.linalg.norm(delta_np))
            stats_text = f'||mu_post||_2={overall_energy:.2f}  |  ||delta||_2={delta_energy:.2f}'
        else:
            stats_text = f'||mu_post||_2={overall_energy:.2f}'

        save_path = os.path.join(self.output_dir, f'latent_trajectory_analysis_epoch_{epoch}.pdf')
        fig.suptitle(f'Latent Trajectory Analysis - Epoch {epoch}\n{stats_text}', fontsize=14, color='#456882')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f'Latent trajectory analysis plot saved to {save_path}')

    def _plot_latent_statistics(
        self,
        mu_prior: Optional[torch.Tensor],
        mu_post: Optional[torch.Tensor],
        epoch: int,
    ):
        """Plot summary statistics for latent trajectories (prior vs posterior)."""
        import os
        import gc
        import numpy as np
        import matplotlib.pyplot as plt

        if mu_post is None:
            return

        batch_idx = 0
        try:
            mu_post_np = mu_post[batch_idx].detach().cpu().numpy()
        except Exception:
            return

        if mu_post_np.ndim != 2:
            return

        if mu_prior is not None:
            try:
                mu_prior_np = mu_prior[batch_idx].detach().cpu().numpy()
            except Exception:
                mu_prior_np = np.zeros_like(mu_post_np)
        else:
            mu_prior_np = np.zeros_like(mu_post_np)

        delta = mu_post_np - mu_prior_np
        prior_norm = np.linalg.norm(mu_prior_np, axis=1)
        post_norm = np.linalg.norm(mu_post_np, axis=1)
        delta_norm = np.linalg.norm(delta, axis=1)
        steps = np.arange(mu_post_np.shape[0])

        with np.errstate(all='ignore'):
            corr = np.corrcoef(mu_post_np.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        delta_heatmap = delta.T
        energy_per_dim = np.sum(delta ** 2, axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

        axes[0, 0].plot(steps, prior_norm, color='#7F8C8D', linewidth=1.0, label='||mu_prior||')
        axes[0, 0].plot(steps, post_norm, color='#BB3E00', linewidth=1.2, label='||mu_post||')
        axes[0, 0].plot(steps, delta_norm, color='#2E86AB', linewidth=1.2, linestyle='--', label='||delta||')
        axes[0, 0].set_title('Latent Norms over Time')
        axes[0, 0].set_xlabel('Decimated step')
        axes[0, 0].set_ylabel('Norm')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc='upper right', framealpha=0.9)

        im_corr = axes[0, 1].imshow(corr, aspect='auto', origin='lower', cmap='Spectral', vmin=-1.0, vmax=1.0)
        axes[0, 1].set_title('Posterior Latent Correlation')
        axes[0, 1].set_xlabel('Latent dim')
        axes[0, 1].set_ylabel('Latent dim')
        fig.colorbar(im_corr, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im_delta = axes[1, 0].imshow(delta_heatmap, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[1, 0].set_title('delta = mu_post - mu_prior')
        axes[1, 0].set_xlabel('Decimated step')
        axes[1, 0].set_ylabel('Latent dim')
        fig.colorbar(im_delta, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].bar(np.arange(delta_heatmap.shape[0]), energy_per_dim, color='#3C6E71')
        axes[1, 1].set_title('Energy per Latent Dimension')
        axes[1, 1].set_xlabel('Latent dim')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].grid(True, axis='y', alpha=0.3)

        fig.suptitle(f'Latent Statistics - Epoch {epoch}', fontsize=14, color='#456882')
        save_path = os.path.join(self.output_dir, f'latent_statistics_epoch_{epoch}.pdf')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        gc.collect()
        logger.info(f"Latent statistics plot saved to {save_path}")

    def _plot_forecast_results(
        self,
        y_raw_normalized: torch.Tensor,
        mean_mu: torch.Tensor,
        std_mu: torch.Tensor,
        canvas_mu: torch.Tensor,
        anchors: torch.Tensor,
        latent_z: torch.Tensor,
        epoch: int,
    ):
        """Plot forecast results: ground truth vs aggregated forecast with uncertainty; sample windows; latent heatmap."""
        import os
        import gc
        
        # Select one sample from the batch (first sample)
        batch_idx = 0
        
        # Convert tensors to numpy and move to CPU
        y_raw = y_raw_normalized[batch_idx].cpu().numpy()  # (4800,)
        pred_mean = mean_mu[batch_idx].cpu().numpy()       # (4800,)
        pred_std = std_mu[batch_idx].cpu().numpy()         # (4800,)
        z_latent = None
        if latent_z is not None:
            z_latent = latent_z[batch_idx].detach().cpu().numpy()  # (T,D)

        mask = ~np.isnan(pred_mean)
        coverage = float(np.mean(mask)) if mask.size > 0 else float('nan')
        coverage_display = f"{coverage:.2%}" if not np.isnan(coverage) else "nan"
        if np.any(mask):
            gt_masked = y_raw.copy()
            pred_masked = pred_mean.copy()
            gt_masked[~mask] = np.nan
            pred_masked[~mask] = np.nan
            sample_mse = float(np.nanmean((pred_masked - gt_masked) ** 2))
            sample_mae = float(np.nanmean(np.abs(pred_masked - gt_masked)))
            if np.sum(mask) > 2:
                sample_corr = float(np.corrcoef(gt_masked[mask], pred_masked[mask])[0, 1])
            else:
                sample_corr = float('nan')
        else:
            sample_mse = float('nan')
            sample_mae = float('nan')
            sample_corr = float('nan')

        logger.info(
            "[PlottingCallback] Epoch {} sample forecast diagnostics | horizon={} | MSE={:.4f} | MAE={:.4f} | Corr={:.4f} | Coverage={:.2%}".format(
                epoch,
                self.predictive_horizon if self.predictive_horizon is not None else 'model',
                sample_mse,
                sample_mae,
                sample_corr,
                coverage if not np.isnan(coverage) else float('nan'),
            )
        )

        # Setup plotting parameters following the style from data_utils
        Fs = 4
        N = len(y_raw)
        t_in = np.arange(0, N) / Fs
        
        # Professional scientific paper color palette
        colors = {
            'fhr': "#055C9A",           # Deep blue-gray
            'up': "#0DD8A2",            # Sage green
            'gt': '#456882',            # Medium blue-gray
            'recon': '#BB3E00',         # Deep orange-red
            'uncertainty': '#F7AD45',    # Golden yellow
            'samples': "#4BD605",       # Muted green-gray
            'background': '#F9F3EF'     # Warm off-white
        }
        
        # Set professional scientific paper styling
        plt.style.use('default')  # Reset to default first
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.linewidth': 0.7,
            'axes.edgecolor': "#9E9D9D",
            'axes.facecolor': colors['background'],
            'grid.color': "#838383",
            'grid.linewidth': 0.4,
            'grid.alpha': 0.6,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.framealpha': 0.95,
            'legend.edgecolor': '#A2B9A7',
            'legend.facecolor': colors['background'],
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.dpi': 300
        })
        
        # Create figure with 3 rows, 2 columns (main plot + colorbar)
        n_rows = 3
        fig, ax = plt.subplots(
            nrows=n_rows, ncols=2, figsize=(20, n_rows * 3.5),
            gridspec_kw={"width_ratios": [80, 1]}, constrained_layout=True)
        
        # Configure scientific paper grid style for all subplots
        for i in range(n_rows):
            ax[i, 0].grid(True, linestyle='-', alpha=0.4, linewidth=0.4, color='#D2C1B6')
            ax[i, 0].grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax[i, 0].minorticks_on()
            ax[i, 0].set_axisbelow(True)
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)
            ax[i, 0].spines['left'].set_color('#A2B9A7')
            ax[i, 0].spines['bottom'].set_color('#A2B9A7')
            ax[i, 0].spines['left'].set_linewidth(0.7)
            ax[i, 0].spines['bottom'].set_linewidth(0.7)
        
        # Subplot 1: GT vs aggregated forecast with uncertainty band
        ax[0, 1].set_axis_off()
        ax[0, 0].plot(t_in, y_raw, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.9)
        ax[0, 0].plot(t_in, pred_mean, linewidth=1.2, color='#BB3E00', label='Forecast (mean)', alpha=0.9)
        ax[0, 0].fill_between(
            t_in,
            pred_mean - pred_std,
            pred_mean + pred_std,
            alpha=0.3,
            color=colors['uncertainty'],
            label='Uncertainty band',
        )
        ax[0, 0].set_ylabel('FHR (bpm)')
        ax[0, 0].set_title('Raw FHR vs Aggregated Forecast')
        ax[0, 0].legend(loc='upper right', framealpha=0.95)
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)

        # Subplot 2: show some example windows overlayed
        ax[1, 1].set_axis_off()
        ax[1, 0].plot(t_in, y_raw, color=colors['gt'], alpha=0.4, linewidth=1.0)
        if anchors.numel() > 0:
            anc = anchors.detach().cpu().numpy()
            cmu = canvas_mu[batch_idx].detach().cpu().numpy()  # (N,4800)
            picks = [anc[0], anc[len(anc)//2], anc[-1]] if len(anc) >= 3 else list(anc)
            for a in picks:
                idx = int(np.where(anc == a)[0][0])
                w = cmu[idx]
                ax[1, 0].plot(t_in, w, color='#D7263D', linewidth=1.0, alpha=0.8)
        ax[1, 0].set_ylabel('FHR (bpm)')
        ax[1, 0].set_title('Sample Forecast Windows')
        ax[1, 0].autoscale(enable=True, axis='x', tight=True)

        # Subplot 3: latent heatmap (posterior mean)
        if z_latent is not None:
            imgplot = ax[2, 0].imshow(z_latent.T, aspect='auto', cmap='bwr', origin='lower')
            ax[2, 1].set_axis_on()
            cbar = fig.colorbar(imgplot, cax=ax[2, 1])
            cbar.ax.tick_params(labelsize=10, colors='#666666')
            cbar.set_label('Activation', fontweight='normal', fontsize=11, color='#666666')
            cbar.outline.set_color('#A2B9A7')
            cbar.outline.set_linewidth(0.7)
            ax[2, 0].set_ylabel('Latent Dimensions')
            ax[2, 0].set_xlabel('Decimated steps')
            ax[2, 0].set_title('Latent mu_post (T x D)')
        else:
            ax[2, 0].text(0.5, 0.5, 'Latents not available', ha='center', va='center')
        
        # Set overall title with scientific paper styling
        diag_str = (
            f"H={self.predictive_horizon if self.predictive_horizon is not None else 'model'} | "
            f"MSE={sample_mse:.4f} | MAE={sample_mae:.4f} | Corr={sample_corr:.4f} | Cov={coverage_display}"
        )
        fig.suptitle(
            f'Forecasting Results - Epoch {epoch}\n{diag_str}',
            fontsize=14,
            fontweight='normal',
            y=0.98,
            color='#456882'
        )
        # Save plot as PDF with high quality
        save_path = os.path.join(self.output_dir, f'forecast_results_epoch_{epoch}.pdf')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Clean up memory  
        del y_raw, pred_mean, pred_std
        if z_latent is not None:
            del z_latent
        gc.collect()
        logger.info(f"Forecast results plot saved to {save_path}")

    def _plot_batch_aggregated_forecast(
        self,
        y_raw_batch: torch.Tensor,
        mean_mu_batch: torch.Tensor,
        std_mu_batch: torch.Tensor,
        epoch: int,
    ):
        """Plot average aggregated forecast across validation batch with uncertainty band.

        Uses per-sample coverage masks from mean_mu_batch (NaNs denote uncovered points).
        Uncertainty combines within-sample variance and across-sample variability (law of total variance).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # Compute per-time coverage mask and masked batch means
        mask = ~torch.isnan(mean_mu_batch)  # (B,4800)
        # Predicted mean across samples (nanmean)
        pred_mean = torch.nanmean(mean_mu_batch, dim=0)  # (4800,)
        # Ground-truth mean across samples restricted to covered points
        gt_masked = y_raw_batch.masked_fill(~mask, float('nan'))
        gt_mean = torch.nanmean(gt_masked, dim=0)  # (4800,)

        # Uncertainty via total variance: E[var] + var(E)
        # E[var] term
        e_var = torch.nanmean(std_mu_batch.pow(2), dim=0)
        # var(E) term: variability of per-sample predicted means
        mean_centered = mean_mu_batch - pred_mean.unsqueeze(0)
        mean_centered = mean_centered.masked_fill(~mask, float('nan'))
        var_e = torch.nanmean(mean_centered.pow(2), dim=0)
        total_var = (e_var + var_e).clamp_min(0.0)
        pred_std = total_var.sqrt()

        t = np.arange(pred_mean.shape[0]) / 4.0
        pm = pred_mean.detach().cpu().numpy()
        ps = pred_std.detach().cpu().numpy()
        gm = gt_mean.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(16, 4.5), constrained_layout=True)
        ax.plot(t, gm, color='#2E86AB', label='GT (batch mean)', linewidth=1.5)
        ax.plot(t, pm, color='#BB3E00', label='Forecast (batch mean)', linewidth=1.2)
        ax.fill_between(
            t,
            pm - ps,
            pm + ps,
            color='#F5B7B1',
            alpha=0.4,
            label='Total uncertainty band',
        )
        ax.set_title('Batch-Aggregated Forecast vs Ground Truth')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FHR (bpm)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        save_path = os.path.join(self.output_dir, f'forecast_results_epoch_{epoch}_avg.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Batch-aggregated forecast plot saved to {save_path}")


class LossPlotCallback(Callback):
    def __init__(self, output_dir, plot_frequency=10, max_history_size=1000):
        """
        Args:
            output_dir (str): Directory where the loss plot HTML files will be saved.
            plot_frequency (int): Frequency (in epochs) to generate the loss plot.
            max_history_size (int): Maximum number of epochs to keep in history to prevent memory issues.
        """
        super().__init__()
        self.output_dir = output_dir
        self.plot_frequency = plot_frequency
        self.max_history_size = max_history_size
        # Standard TEB metrics
        self.history = {
            "epoch": [],
            "train/total_loss": [],
            "train/recon_loss": [],
            "train/mse_loss": [],
            "train/nll_loss": [],
            "train/predictive_loss": [],
            "train/latent_nll_loss": [],
            "train/latent_consistency_loss": [],
            "train/forecast_nll": [],
            "train/predictive_kl_loss": [],
            "train/stability_penalty": [],
            "train/kld_loss": [],
            "train/agg_mse": [],
            "val/total_loss": [],
            "val/recon_loss": [],
            "val/mse_loss": [],
            "val/nll_loss": [],
            "val/predictive_loss": [],
            "val/latent_nll_loss": [],
            "val/latent_consistency_loss": [],
            "val/forecast_nll": [],
            "val/predictive_kl_loss": [],
            "val/stability_penalty": [],
            "val/kld_loss": [],
            "val/agg_mse": [],
            "val/agg_mae": [],
            "val/agg_corr": [],
            "val/agg_std": [],
            "val/agg_coverage": [],
            # Hyperparameters
            "hyperparams/beta": [],
            "hyperparams/lr": []
        }

    def _trim_history(self):
        """Trim history to prevent unlimited memory growth."""
        if len(self.history["epoch"]) > self.max_history_size:
            # Keep only the last max_history_size entries
            trim_size = len(self.history["epoch"]) - self.max_history_size
            for key in self.history:
                self.history[key] = self.history[key][trim_size:]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Extract the current epoch number
        epoch = trainer.current_epoch

        # Retrieve logged metrics from the trainer
        metrics = trainer.callback_metrics

        def to_float(x):
            return x.item() if x is not None and hasattr(x, 'item') else float('nan')

        # Store losses in history
        self.history["epoch"].append(epoch)
        for key in self.history:
            if key != "epoch":
                self.history[key].append(to_float(metrics.get(key)))

        # Trim history to prevent memory issues
        self._trim_history()

        # Check if it's time to plot the losses and only do so on the main process
        if (epoch + 1) % self.plot_frequency == 0 and trainer.is_global_zero:
            self.plot_losses()
            self.plot_hyperparameters()

    def plot_losses(self):
        import os
        import plotly.graph_objects as go
        import gc

        # Create a Plotly figure and add a trace for each metric.
        fig = go.Figure()

        for key, values in self.history.items():
            if key == "epoch" or not any(v is not None and not np.isnan(v) for v in values):
                continue

            fig.add_trace(go.Scatter(
                x=self.history["epoch"],
                y=values,
                mode='lines+markers',
                name=key.replace('/', ' ').title()
            ))

        # Customize layout
        fig.update_layout(
            title="Training and Validation Losses",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Metrics",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Save the figure as an HTML file
        plot_path = os.path.join(self.output_dir, f"loss_plot_epoch.html")
        fig.write_html(plot_path)
        logger.info(f"Loss plot saved to {plot_path}")

        # Clean up figure to free memory
        del fig
        gc.collect()

    def plot_hyperparameters(self):
        """Create a plot of hyperparameter evolution"""
        import os
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import gc

        if len(self.history["epoch"]) == 0:
            return

        # Check if we have hyperparameter data
        has_beta = any(v is not None and not np.isnan(v) for v in self.history.get("hyperparams/beta", []))
        has_lr = any(v is not None and not np.isnan(v) for v in self.history.get("hyperparams/lr", []))

        if not (has_beta or has_lr):
            logger.info("No hyperparameter data available for plotting")
            return

        # Create subplots for different hyperparameters
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Beta (KLD Weight)', 'Learning Rate'),
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Plot Beta if available
        if has_beta:
            fig.add_trace(
                go.Scatter(x=self.history["epoch"], y=self.history["hyperparams/beta"],
                          mode='lines+markers', name='Beta', line=dict(color='red', width=2)),
                row=1, col=1
            )

        # Plot Learning Rate if available
        if has_lr:
            fig.add_trace(
                go.Scatter(x=self.history["epoch"], y=self.history["hyperparams/lr"],
                          mode='lines+markers', name='Learning Rate', line=dict(color='blue', width=2)),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title="Training Hyperparameters Evolution",
            showlegend=False,
            template="plotly_white",
            height=400
        )

        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)

        fig.update_yaxes(title_text="Beta Value", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log")

        # Save the figure
        plot_path = os.path.join(self.output_dir, "hyperparameters_evolution.html")
        fig.write_html(plot_path)
        logger.info(f"Hyperparameters plot saved to {plot_path}")

        # Clean up
        del fig
        gc.collect()


class HyperparameterLoggingCallback(Callback):
    """
    Callback to track and log hyperparameters like beta, learning rate, alpha, gamma, etc.
    """
    def __init__(self):
        super().__init__()
        self.history = {
            "epoch": [],
            "beta": [],
            "lr": []
        }

    def on_train_epoch_start(self, trainer, pl_module):
        """Log hyperparameters at the start of each epoch"""
        if trainer.is_global_zero:  # Only log on main process for multi-GPU
            epoch = trainer.current_epoch
            
            # Get current beta value (always calculate fresh to reflect any config changes)
            beta = pl_module._calculate_beta()
            
            # Get current learning rate
            lr = 0.0
            try:
                if hasattr(trainer, 'optimizers') and trainer.optimizers:
                    optimizer = trainer.optimizers[0] if isinstance(trainer.optimizers, list) else trainer.optimizers
                    lr = optimizer.param_groups[0]['lr']
            except (IndexError, AttributeError):
                pass
            
            # Store in history
            self.history["epoch"].append(epoch)
            self.history["beta"].append(beta)
            self.history["lr"].append(lr)
            
            # LightningModule handles logging; callback only tracks history
            
            logger.info(f"Epoch {epoch}: beta={beta:.4f}, lr={lr:.6f}")

    def plot_hyperparameters(self, output_dir):
        """Create a plot of hyperparameter evolution"""
        import os
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import gc

        if len(self.history["epoch"]) == 0:
            return

        # Create subplots for different hyperparameters
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Beta (KLD Weight)', 'Learning Rate'),
            vertical_spacing=0.12,
            horizontal_spacing=0.10
        )

        # Plot Beta
        fig.add_trace(
            go.Scatter(x=self.history["epoch"], y=self.history["beta"],
                      mode='lines+markers', name='Beta', line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Plot Learning Rate
        fig.add_trace(
            go.Scatter(x=self.history["epoch"], y=self.history["lr"],
                      mode='lines+markers', name='Learning Rate', line=dict(color='blue', width=2)),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title="Training Hyperparameters Evolution",
            showlegend=False,
            template="plotly_white",
            height=400
        )

        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)

        fig.update_yaxes(title_text="Beta Value", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=2, type="log")

        # Save the figure
        plot_path = os.path.join(output_dir, "hyperparameters_evolution.html")
        fig.write_html(plot_path)
        logger.info(f"Hyperparameters plot saved to {plot_path}")

        # Clean up
        del fig
        gc.collect()


class MetricsLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss_history = []
        self.val_loss_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        train_loss = logs.get("train/total_loss")
        self.train_loss_history.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        val_loss = logs.get("val/total_loss")
        self.val_loss_history.append(val_loss)


class LightSeqVaeTeb(L.LightningModule):
    """
    PyTorch Lightning module for the SeqVaeTeb model.

    This module handles the training, validation, and optimization loops,
    including learning rate scheduling and KLD beta annealing.
    Supports both standard TEB and beta-TCVAE training modes.
    """

    def __init__(
        self,
        seqvae_teb_model: SeqVaeTeb,
        lr: float = 1e-4,
        lr_milestones: list = None,
        beta_schedule: str = "linear",
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_anneal_epochs: int = 100,
        beta_cycle_len: int = 1000,
        beta_const_val: float = 1.0,
        predictive_weight: float = 0.0,
        latent_consistency_weight: float = 0.0,
        predictive_horizon: Optional[int] = None,
        predictive_context_len: Optional[int] = None,
        log_forecast_metrics: bool = True,
        predictive_max_anchors: Optional[int] = None,
        *,
        forecast_weight: Optional[float] = None,
        latent_nll_weight: Optional[float] = None,
        predictive_kl_weight: float = 0.0,
        stability_weight: float = 0.0,
        ):
        """
        Args:
            seqvae_teb_model: An instance of the SeqVaeTeb model.
            lr: Learning rate.
            lr_milestones: Epochs at which to decay the learning rate.
            beta_schedule: Type of beta annealing schedule. Options: 'constant', 'linear', 'cyclic'.
            beta_start: Starting value for beta in annealing schedules.
            beta_end: Final value for beta in annealing schedules.
            beta_anneal_epochs: Number of epochs for linear annealing.
            beta_cycle_len: Length of a cycle for cyclic annealing.
            beta_const_val: Constant value for beta if schedule is 'constant'.
            predictive_weight: (legacy) raw forecasting loss weight.
            latent_consistency_weight: (legacy) latent consistency weight.
            predictive_horizon: Forecast horizon (decimated steps) used for auxiliary objectives.
            predictive_context_len: Context length supplied to the latent forecaster (decimated steps).
            log_forecast_metrics: Whether to compute/log aggregated forecast metrics during validation.
            predictive_max_anchors: Optional cap on anchors sampled for auxiliary loss to control memory.
            forecast_weight: Weight for raw forecast NLL (defaults to predictive_weight).
            latent_nll_weight: Weight for latent Gaussian NLL (defaults to latent_consistency_weight).
            predictive_kl_weight: Weight for KL between LGSSM predictions and posterior latents.
            stability_weight: Weight for the LGSSM stability regulariser.
        """
        super().__init__()

        # Default predictive settings to model attributes when not provided
        model_horizon = getattr(seqvae_teb_model, "horizon_len", None)
        model_context = getattr(seqvae_teb_model, "context_len", None)

        if predictive_horizon is None:
            predictive_horizon = model_horizon if model_horizon is not None else 1
        if predictive_context_len is None:
            if model_context is not None:
                predictive_context_len = model_context
            else:
                predictive_context_len = max(predictive_horizon, 1)
        if predictive_max_anchors is None:
            predictive_max_anchors = 0

        if forecast_weight is None:
            forecast_weight = predictive_weight
        if latent_nll_weight is None:
            latent_nll_weight = latent_consistency_weight

        # Using save_hyperparameters to automatically save arguments to self.hparams
        self.save_hyperparameters(ignore=['seqvae_teb_model'])
        self.model = seqvae_teb_model
        self._forecaster_enabled = getattr(self.model, "has_forecaster", lambda: True)()
        self.hparams.enable_forecaster = self._forecaster_enabled
        if not self._forecaster_enabled:
            self.hparams.predictive_weight = 0.0
            self.hparams.latent_consistency_weight = 0.0
            self.hparams.forecast_weight = 0.0
            self.hparams.latent_nll_weight = 0.0
            self.hparams.predictive_kl_weight = 0.0
            self.hparams.stability_weight = 0.0
            self.hparams.log_forecast_metrics = False

        # Only compile if not already compiled to avoid double compilation
        if not is_compiled_module(self.model):
            self.model, self._model_compiled = ensure_compiled_module(
                self.model,
                module_name="SeqVaeTeb Lightning wrapper",
            )
        else:
            self._model_compiled = True
            logger.info("[LightSeqVaeTeb] Model already compiled, skipping compilation")

    def forward(self, y_st, y_ph, x_ph):
        """Forward pass through the SeqVaeTeb model."""
        return self.model(y_st, y_ph, x_ph)

    def _has_forecaster(self) -> bool:
        return getattr(self.model, "has_forecaster", lambda: True)()

    def _calculate_beta(self):
        """Calculates the KLD weight (beta) based on the current epoch and schedule."""
        schedule = self.hparams.beta_schedule
        epoch = self.current_epoch

        if schedule == 'linear':
            # Linear annealing from beta_start to beta_end
            progress = min(1.0, epoch / self.hparams.beta_anneal_epochs)
            beta = self.hparams.beta_start + (self.hparams.beta_end - self.hparams.beta_start) * progress
        elif schedule == 'cyclic':
            # Cyclic annealing
            cycle_progress = (epoch % self.hparams.beta_cycle_len) / self.hparams.beta_cycle_len
            beta = self.hparams.beta_start + (self.hparams.beta_end - self.hparams.beta_start) * cycle_progress
        elif schedule == 'constant':
            beta = self.hparams.beta_const_val
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

        # Update beta in the underlying model
        return beta

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        self.hparams.beta = self._calculate_beta()
        self.log('kld_beta', self.hparams.beta, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('hyperparams/beta', self.hparams.beta, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        # Log learning rate at the start of each epoch
        try:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr', lr, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('hyperparams/lr', lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        except IndexError:
            # This can happen if the optimizer is not yet configured
            pass
        
        # Validate hyperparameters are correctly set (first epoch only)
        if self.current_epoch == 0:
            self._validate_hyperparameters()

    def _validate_hyperparameters(self):
        """Validate that hyperparameters are correctly set from config."""
        logger.info(" Validating hyperparameters...")
        logger.info(f"  Current beta_schedule: {self.hparams.beta_schedule}")
        logger.info(f"  Current beta_const_val: {self.hparams.beta_const_val}")
        logger.info(f"  Current beta_start: {self.hparams.beta_start}")
        logger.info(f"  Current beta_end: {self.hparams.beta_end}")
        logger.info(f"  Current lr: {self.hparams.lr}")
        logger.info(f"  Current lr_milestones: {self.hparams.lr_milestones}")
        logger.info(" Hyperparameter validation complete")

    def _compute_forecast_metrics(
        self,
        mu_post: torch.Tensor,
        y_raw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate aggregated forecast metrics from posterior means without autograd."""
        metrics: Dict[str, torch.Tensor] = {}

        horizon = max(int(self.hparams.predictive_horizon), 1)
        context_len = max(int(self.hparams.predictive_context_len), 1)
        stride = self.model.decimation_factor

        # Ensure requested horizons fit within available sequence length
        horizon = min(horizon, mu_post.size(1))
        context_len = min(context_len, mu_post.size(1))

        anchors = self.model.anchor_range(mu_post.size(1), context_len, horizon)
        if anchors.numel() == 0:
            return metrics

        anchors = anchors.to(mu_post.device)

        max_anchors = int(getattr(self.hparams, 'predictive_max_anchors', 0) or 0)
        if max_anchors > 0 and anchors.numel() > max_anchors:
            perm = torch.randperm(anchors.numel(), device=anchors.device)
            anchors = anchors[perm[:max_anchors]]
        contexts = self.model._gather_context(mu_post, anchors, context_len)  # (B, N, Lc, D)
        B, N, Lc, D = contexts.shape
        contexts_flat = contexts.reshape(B * N, Lc, D)

        mu_latent_flat, _, _ = self.model.latent_forecaster(contexts_flat, horizon=horizon)
        _, mu_flat, logvar_flat = self.model.decoder(mu_latent_flat)
        mu_future = mu_flat.reshape(B, N, -1)
        logvar_future = torch.clamp(logvar_flat.reshape(B, N, -1), min=-10, max=10)

        canvas_mu, mean_mu = self.model.aggregate_forecasts_to_canvas(
            mu_future, anchors, total_len=y_raw.shape[1], stride=stride
        )
        var_future = logvar_future.exp()
        _, mean_var = self.model.aggregate_forecasts_to_canvas(
            var_future, anchors, total_len=y_raw.shape[1], stride=stride
        )

        mask = ~torch.isnan(mean_mu)
        if mask.any():
            pred = mean_mu.masked_fill(~mask, 0.0)
            gt = y_raw.masked_fill(~mask, 0.0)
            denom = mask.sum(dim=1).clamp_min(1)
            mse = (pred - gt).pow(2).sum(dim=1) / denom
            mae = (pred - gt).abs().sum(dim=1) / denom
            corr = self.model._masked_corrcoef(pred, gt, mask)
            coverage = mask.float().mean(dim=1)

            metrics['agg_mse'] = torch.nanmean(mse)
            metrics['agg_mae'] = torch.nanmean(mae)
            metrics['agg_corr'] = torch.nanmean(corr)
            metrics['agg_coverage'] = torch.nanmean(coverage)

        metrics['agg_std'] = torch.nanmean(mean_var.clamp_min(1e-8).sqrt())
        return metrics

    def _compute_losses_and_metrics(self, batch, stage: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run forward pass, compute losses, and (optionally) auxiliary forecast metrics."""
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        x_ph = batch.fhr_up_ph
        y_raw = batch.fhr
        use_forecaster = self._has_forecaster()

        def _get_weight(primary: str, legacy: str = "", default: float = 0.0) -> float:
            if not use_forecaster:
                return 0.0
            if hasattr(self.hparams, primary):
                return getattr(self.hparams, primary)
            if legacy and hasattr(self.hparams, legacy):
                return getattr(self.hparams, legacy)
            return default

        forecast_weight = _get_weight("forecast_weight", "predictive_weight")
        latent_nll_weight = _get_weight("latent_nll_weight", "latent_consistency_weight")
        predictive_kl_weight = _get_weight("predictive_kl_weight")
        stability_weight = _get_weight("stability_weight")
        predictive_weight = _get_weight("predictive_weight")
        latent_consistency_weight = _get_weight("latent_consistency_weight")

        predictive_horizon = max(1, int(self.hparams.predictive_horizon))
        predictive_context = max(1, int(self.hparams.predictive_context_len))
        predictive_max = None
        if use_forecaster and getattr(self.hparams, 'predictive_max_anchors', None) is not None:
            predictive_max = int(self.hparams.predictive_max_anchors)
            if predictive_max <= 0:
                predictive_max = None
        forward_outputs = self.model(y_st, y_ph, x_ph)

        loss_dict = self.model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            y_raw=y_raw,
            compute_kld_loss=True,
            beta=self.hparams.beta,
            predictive_weight=predictive_weight,
            predictive_horizon=predictive_horizon,
            latent_consistency_weight=latent_consistency_weight,
            predictive_context_len=predictive_context,
            predictive_max_anchors=predictive_max,
            latent_nll_weight=latent_nll_weight,
            forecast_weight=forecast_weight,
            predictive_kl_weight=predictive_kl_weight,
            stability_weight=stability_weight,
        )

        aux_metrics: Dict[str, torch.Tensor] = {}
        if use_forecaster and self.hparams.log_forecast_metrics and stage != "train":
            with torch.no_grad():
                aux_metrics = self._compute_forecast_metrics(
                    mu_post=forward_outputs["mu_post"].detach(),
                    y_raw=y_raw,
                )

        return loss_dict, aux_metrics

    def training_step(self, batch, batch_idx):
        """Defines the training loop with memory optimization."""
        loss_dict, aux_metrics = self._compute_losses_and_metrics(batch, stage="train")
        total_loss = loss_dict['total_loss']

        # Core reconstruction / KL logging
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train/recon_loss', loss_dict['reconstruction_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/mse_loss', loss_dict['mse_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/nll_loss', loss_dict['nll_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/kld_loss', loss_dict['kld_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Forecast-specific losses
        if 'predictive_loss' in loss_dict:
            self.log('train/predictive_loss', loss_dict['predictive_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if 'forecast_nll' in loss_dict:
            self.log('train/forecast_nll', loss_dict['forecast_nll'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if 'latent_nll_loss' in loss_dict:
            self.log('train/latent_nll_loss', loss_dict['latent_nll_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/latent_consistency_loss', loss_dict['latent_nll_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if 'predictive_kl_loss' in loss_dict:
            self.log('train/predictive_kl_loss', loss_dict['predictive_kl_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if 'stability_penalty' in loss_dict:
            self.log('train/stability_penalty', loss_dict['stability_penalty'], on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Auxiliary metrics (if any were computed)
        for name, value in aux_metrics.items():
            self.log(f'train/{name}', value, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation loop with memory optimization."""
        loss_dict, aux_metrics = self._compute_losses_and_metrics(batch, stage="val")
        total_loss = loss_dict['total_loss']

        # Core reconstruction / KL logging
        self.log('val/total_loss', total_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/recon_loss', loss_dict['reconstruction_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/mse_loss', loss_dict['mse_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/nll_loss', loss_dict['nll_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/kld_loss', loss_dict['kld_loss'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if 'predictive_loss' in loss_dict:
            self.log('val/predictive_loss', loss_dict['predictive_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if 'forecast_nll' in loss_dict:
            self.log('val/forecast_nll', loss_dict['forecast_nll'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if 'latent_nll_loss' in loss_dict:
            self.log('val/latent_nll_loss', loss_dict['latent_nll_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log('val/latent_consistency_loss', loss_dict['latent_nll_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if 'predictive_kl_loss' in loss_dict:
            self.log('val/predictive_kl_loss', loss_dict['predictive_kl_loss'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if 'stability_penalty' in loss_dict:
            self.log('val/stability_penalty', loss_dict['stability_penalty'], on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        for name, value in aux_metrics.items():
            self.log(f'val/{name}', value, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Minimal cleanup after each training batch - removed frequent cache clearing for multi-GPU."""
        # Only clean up batch references - no cache clearing for better multi-GPU performance
        del batch

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Minimal cleanup after each validation batch - removed frequent cache clearing for multi-GPU."""
        # Only clean up batch references - no cache clearing for better multi-GPU performance
        del batch

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers with SOTA optimizations."""
        # OPTIMIZATION: Use AdamW with gradient clipping compatibility
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,     # L2 regularization
            eps=1e-8,              # Numerical stability
            betas=(0.9, 0.95),     # SOTA: Slightly higher beta2 for better convergence
            amsgrad=False,         # Standard AdamW
            # foreach=True,          # SOTA: Vectorized optimizer updates (faster)
            maximize=False,
            capturable=False,      # Standard mode for compatibility
            differentiable=False,
            fused=False,           # Disable fused for gradient clipping compatibility
        )

        if self.hparams.lr_milestones:
            # Use simple milestone-based learning rate scheduler
            from torch.optim.lr_scheduler import MultiStepLR
            scheduler = MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_milestones,
                gamma=0.1  # Decay factor at each milestone
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Epoch-wise for milestone scheduling
                    "frequency": 1,
                },
            }
        return optimizer


class MemoryMonitorCallback(Callback):
    """
    Callback to monitor GPU memory usage and automatically clear cache when needed.
    Optimized for multi-GPU training with reduced monitoring frequency.
    """

    def __init__(self, threshold_gb=12.0, log_frequency=200):
        """
        Args:
            threshold_gb (float): GPU memory threshold in GB above which cache is cleared.
            log_frequency (int): Frequency (in batches) to log memory usage.
        """
        super().__init__()
        self.threshold_gb = threshold_gb
        self.log_frequency = log_frequency
        self.batch_count = 0

    def _log_memory_usage(self, prefix=""):
        """Log current GPU memory usage for all devices."""
        if torch.cuda.is_available():
            total_allocated = 0.0
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3  # GB
                logger.info(f"{prefix} GPU {device_id}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                total_allocated += allocated
            return total_allocated
        return 0.0

    def _clear_memory_if_needed(self):
        """Clear GPU memory on all devices if usage exceeds threshold."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cleared_any = False
            for device_id in range(device_count):
                allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3  # GB
                if allocated > self.threshold_gb:
                    logger.warning(
                        f"GPU {device_id} memory usage ({allocated:.2f}GB) exceeds threshold ({self.threshold_gb}GB). Clearing cache...")
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                    cleared_any = True
            return cleared_any
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor memory after each training batch."""
        self.batch_count += 1

        # Log memory usage periodically
        if self.batch_count % self.log_frequency == 0:
            self._log_memory_usage(f"Train batch {batch_idx}")

        # Clear memory if needed
        self._clear_memory_if_needed()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor memory after each validation batch."""
        # Clear memory if needed during validation
        self._clear_memory_if_needed()

    def on_train_epoch_start(self, trainer, pl_module):
        """Log memory at the start of each epoch."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} start")

    def on_train_epoch_end(self, trainer, pl_module):
        """Log usage at the end of each epoch - reduced cache clearing for multi-GPU."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} end")
        # Only clear cache at epoch end, not during training for better multi-GPU performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


