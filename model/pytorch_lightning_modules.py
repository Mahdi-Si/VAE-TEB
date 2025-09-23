import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import os
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple

from vae_teb_model import SeqVaeTeb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON'] = "NO"

matplotlib.use('Agg')
torch.backends.cudnn.enabled = False

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

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, pl_module.local_rank)

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

                # Forecast across all valid anchors using posterior mean latents for stability
                forecast_out = pl_module.model.forecast(y_st, y_ph, x_ph, anchors=None, use_posterior_mean=True)
                anchors = forecast_out["anchors"]
                mu_future = forecast_out["mu_future"]          # (B,N,480)
                logvar_future = forecast_out["logvar_future"]  # (B,N,480)
                enc = forecast_out["enc"]

                # Aggregate forecasts to full raw timeline
                canvas_mu, mean_mu = pl_module.model.aggregate_forecasts_to_canvas(
                    mu_future, anchors, total_len=y_raw_normalized.shape[1], stride=pl_module.model.decimation_factor)

                # Uncertainty: aggregate variances then sqrt to get std
                var_future = logvar_future.exp()
                _, mean_var = pl_module.model.aggregate_forecasts_to_canvas(
                    var_future, anchors, total_len=y_raw_normalized.shape[1], stride=pl_module.model.decimation_factor)
                std_mu = mean_var.clamp_min(1e-8).sqrt()

                # Plot forecast results
                self._plot_forecast_results(
                    y_raw_normalized,
                    mean_mu,
                    std_mu,
                    canvas_mu,
                    anchors,
                    enc.get('mu_post'),
                    pl_trainer.current_epoch)

                # Additionally, plot batch-aggregated forecast across samples
                try:
                    self._plot_batch_aggregated_forecast(
                        y_raw_batch=y_raw_normalized,
                        mean_mu_batch=mean_mu,
                        std_mu_batch=std_mu,
                        epoch=pl_trainer.current_epoch,
                    )
                except Exception as e:
                    logger.warning(f"Batch-aggregated forecast plotting failed: {e}")
                


        except Exception as e:
            logger.error(f"Error during plotting: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            pl_module.train()

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
        ax[0, 0].fill_between(t_in, pred_mean - pred_std, pred_mean + pred_std, alpha=0.3, color=colors['uncertainty'], label='±1σ')
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
            ax[2, 0].set_title('Latent μ_post (T×D)')
        else:
            ax[2, 0].text(0.5, 0.5, 'Latents not available', ha='center', va='center')
        
        # Set overall title with scientific paper styling
        diag_str = (
            f"H={self.predictive_horizon if self.predictive_horizon is not None else 'model'} | "
            f"MSE={sample_mse:.4f} | MAE={sample_mae:.4f} | Corr={sample_corr:.4f} | Cov={coverage_display}"
        )
        fig.suptitle(
            f'Forecasting Results – Epoch {epoch}\n{diag_str}',
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
        ax.fill_between(t, pm - ps, pm + ps, color='#F5B7B1', alpha=0.4, label='±1σ (total)')
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
            "train/latent_consistency_loss": [],
            "train/forecast_nll": [],
            "train/kld_loss": [],
            "train/agg_mse": [],
            "val/total_loss": [],
            "val/recon_loss": [],
            "val/mse_loss": [],
            "val/nll_loss": [],
            "val/predictive_loss": [],
            "val/latent_consistency_loss": [],
            "val/forecast_nll": [],
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
            
            # Log to trainer for tensorboard/wandb
            pl_module.log('hyperparams/beta', beta, on_epoch=True, logger=True)
            pl_module.log('hyperparams/lr', lr, on_epoch=True, logger=True)
            
            logger.info(f"Epoch {epoch}: β={beta:.4f}, lr={lr:.6f}")

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
        train_loss = logs.get("train_loss")
        self.train_loss_history.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        val_loss = logs.get("validation_loss")
        self.val_loss_history.append(val_loss)


class LightSeqVaeTeb(L.LightningModule):
    """
    PyTorch Lightning module for the SeqVaeTeb model.

    This module handles the training, validation, and optimization loops,
    including learning rate scheduling and KLD beta annealing.
    Supports both standard TEB and β-TCVAE training modes.
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
            predictive_weight: Weight for auxiliary raw forecasting NLL during training.
            latent_consistency_weight: Weight for latent forecast consistency (MSE) term.
            predictive_horizon: Forecast horizon (decimated steps) used for auxiliary objectives.
            predictive_context_len: Context length supplied to the latent forecaster (decimated steps).
            log_forecast_metrics: Whether to compute/log aggregated forecast metrics during validation.
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

        # Using save_hyperparameters to automatically save arguments to self.hparams
        self.save_hyperparameters(ignore=['seqvae_teb_model'])
        self.model = seqvae_teb_model

    def forward(self, y_st, y_ph, x_ph):
        """Forward pass through the SeqVaeTeb model."""
        return self.model(y_st, y_ph, x_ph)

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
        self.log('kld_beta', self.hparams.beta, on_epoch=True, prog_bar=True)
        self.log('hyperparams/beta', self.hparams.beta, on_epoch=True, prog_bar=False, logger=True)
        
        # Log learning rate at the start of each epoch
        try:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr', lr, on_epoch=True, prog_bar=True, logger=True)
            self.log('hyperparams/lr', lr, on_epoch=True, prog_bar=False, logger=True)
        except IndexError:
            # This can happen if the optimizer is not yet configured
            pass
        
        # Validate hyperparameters are correctly set (first epoch only)
        if self.current_epoch == 0:
            self._validate_hyperparameters()

    def _validate_hyperparameters(self):
        """Validate that hyperparameters are correctly set from config."""
        logger.info("🔍 Validating hyperparameters...")
        logger.info(f"  Current beta_schedule: {self.hparams.beta_schedule}")
        logger.info(f"  Current beta_const_val: {self.hparams.beta_const_val}")
        logger.info(f"  Current beta_start: {self.hparams.beta_start}")
        logger.info(f"  Current beta_end: {self.hparams.beta_end}")
        logger.info(f"  Current lr: {self.hparams.lr}")
        logger.info(f"  Current lr_milestones: {self.hparams.lr_milestones}")
        logger.info("✅ Hyperparameter validation complete")

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
        contexts = self.model._gather_context(mu_post, anchors, context_len)  # (B, N, Lc, D)
        B, N, Lc, D = contexts.shape
        contexts_flat = contexts.reshape(B * N, Lc, D)

        z_future_flat = self.model.latent_forecaster(contexts_flat, horizon=horizon)
        _, mu_flat, logvar_flat = self.model.decoder(z_future_flat)
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

        forward_outputs = self.model(y_st, y_ph, x_ph)

        loss_dict = self.model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            y_raw=y_raw,
            compute_kld_loss=True,
            beta=self.hparams.beta,
            predictive_weight=self.hparams.predictive_weight,
            predictive_horizon=max(1, int(self.hparams.predictive_horizon)),
            latent_consistency_weight=self.hparams.latent_consistency_weight,
            predictive_context_len=max(1, int(self.hparams.predictive_context_len)),
        )

        aux_metrics: Dict[str, torch.Tensor] = {}
        if self.hparams.log_forecast_metrics and stage != "train":
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
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/recon_loss', loss_dict['reconstruction_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/mse_loss', loss_dict['mse_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/nll_loss', loss_dict['nll_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train/kld_loss', loss_dict['kld_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Forecast-specific losses
        if 'predictive_loss' in loss_dict:
            self.log('train/predictive_loss', loss_dict['predictive_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            # Backward-compat alias for prior dashboards
            self.log('train/forecast_nll', loss_dict['predictive_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'latent_consistency_loss' in loss_dict:
            self.log('train/latent_consistency_loss', loss_dict['latent_consistency_loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Auxiliary metrics (if any were computed)
        for name, value in aux_metrics.items():
            self.log(f'train/{name}', value, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation loop with memory optimization."""
        loss_dict, aux_metrics = self._compute_losses_and_metrics(batch, stage="val")
        total_loss = loss_dict['total_loss']

        # Core reconstruction / KL logging
        self.log('val/total_loss', total_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/recon_loss', loss_dict['reconstruction_loss'], on_epoch=True, prog_bar=False, logger=True)
        self.log('val/mse_loss', loss_dict['mse_loss'], on_epoch=True, prog_bar=False, logger=True)
        self.log('val/nll_loss', loss_dict['nll_loss'], on_epoch=True, prog_bar=False, logger=True)
        self.log('val/kld_loss', loss_dict['kld_loss'], on_epoch=True, prog_bar=True, logger=True)

        if 'predictive_loss' in loss_dict:
            self.log('val/predictive_loss', loss_dict['predictive_loss'], on_epoch=True, prog_bar=False, logger=True)
            self.log('val/forecast_nll', loss_dict['predictive_loss'], on_epoch=True, prog_bar=True, logger=True)
        if 'latent_consistency_loss' in loss_dict:
            self.log('val/latent_consistency_loss', loss_dict['latent_consistency_loss'], on_epoch=True, prog_bar=False, logger=True)

        for name, value in aux_metrics.items():
            self.log(f'val/{name}', value, on_epoch=True, prog_bar=False, logger=True)

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
            betas=(0.9, 0.95),     # SOTA: Slightly higher β2 for better convergence
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
