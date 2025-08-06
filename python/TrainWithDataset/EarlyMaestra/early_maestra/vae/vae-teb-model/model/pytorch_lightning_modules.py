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
    def __init__(self, output_dir, plot_every_epoch, input_channel_num):
        super().__init__()
        self.output_dir = output_dir
        self.plot_every_epoch = plot_every_epoch
        self.input_channel_num = input_channel_num

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

                model_outputs = pl_module.model(y_st, y_ph, x_ph)
                latent_z = model_outputs['z']
                # Note: mu_pr and logvar_pr are now (B, 4800) raw signal reconstructions
                mu_pr_raw = model_outputs['mu_pr']  # (B, 4800)
                logvar_pr_raw = model_outputs['logvar_pr']  # (B, 4800)
                # For compatibility with existing plotting, create dummy means
                mu_pr_means = mu_pr_raw
                log_var_means = logvar_pr_raw
                
                # Plot results
                self._plot_results(
                    y_raw_normalized,
                    up_raw_normalized,
                    mu_pr_means,
                    log_var_means,
                    mu_pr_raw,  # Remove unsqueeze since we'll handle this in plotting
                    logvar_pr_raw,  # Remove unsqueeze since we'll handle this in plotting
                    latent_z,
                    pl_trainer.current_epoch)
                


        except Exception as e:
            logger.error(f"Error during plotting: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            pl_module.train()

    def _plot_results(
        self, y_raw_normalized, up_raw_normalized, mu_pr_means, log_var_means,
        mu_pr, logvar_pr, latent_z, epoch):
        """Plot model results with 4 subplots following the style of plot_forward_pass"""
        import os
        import gc
        
        # Select one sample from the batch (first sample)
        batch_idx = 0
        
        # Convert tensors to numpy and move to CPU
        y_raw = y_raw_normalized[batch_idx].cpu().numpy()  # Shape: (4800,)
        up_raw = up_raw_normalized[batch_idx].cpu().numpy()  # Shape: (4800,)
        mu_means = mu_pr_means[batch_idx].cpu().numpy()  # Shape: (4800,)
        log_var = log_var_means[batch_idx].cpu().numpy()  # Shape: (4800,)
        # Handle the case where mu_pr and logvar_pr are now (B, 4800) instead of (B, 300, 4800)
        if len(mu_pr.shape) == 2:  # (B, 4800) format
            mu_samples = mu_pr[batch_idx].cpu().numpy()  # Shape: (4800,)
            logvar_samples = logvar_pr[batch_idx].cpu().numpy()  # Shape: (4800,)
        else:  # (B, 300, 4800) format (legacy)
            mu_samples = mu_pr[batch_idx].cpu().numpy()  # Shape: (300, 4800)
            logvar_samples = logvar_pr[batch_idx].cpu().numpy()  # Shape: (300, 4800)
        z_latent = latent_z[batch_idx].cpu().numpy()  # Shape: (300, 32)
        
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
        
        # Create figure with 4 rows, 2 columns (main plot + colorbar)
        n_rows = 4
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
        
        # Subplot 1: y_raw_normalized and up_raw_normalized
        ax[0, 1].set_axis_off()
        ax[0, 0].plot(t_in, y_raw, linewidth=1.2, color=colors['fhr'], label='FHR', alpha=0.85)
        ax[0, 0].plot(t_in, up_raw, linewidth=1.2, color=colors['up'], label='UP', alpha=0.85)
        ax[0, 0].set_ylabel('Amplitude', fontweight='normal')
        ax[0, 0].set_title('Raw FHR and UP Signals', fontweight='normal', pad=12)
        ax[0, 0].legend(loc='upper right', framealpha=0.95)
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)
        
        # Subplot 2: y_raw_normalized and mu_pr_means with uncertainty
        ax[1, 1].set_axis_off()
        ax[1, 0].plot(t_in, y_raw, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.85, zorder=3)
        ax[1, 0].plot(t_in, mu_means, linewidth=1.5, color=colors['recon'], label='Reconstruction', alpha=0.85, zorder=2)
        
        # Add uncertainty visualization using log_var_means
        std_dev = np.exp(0.5 * log_var)  # Convert log variance to standard deviation
        ax[1, 0].fill_between(t_in, mu_means - std_dev, mu_means + std_dev, 
                                alpha=0.3, color=colors['uncertainty'], label='Uncertainty (±1σ)', zorder=1)
        
        ax[1, 0].set_ylabel('FHR (bpm)', fontweight='normal')
        ax[1, 0].set_title('FHR Reconstruction with Uncertainty', fontweight='normal', pad=12)
        ax[1, 0].legend(loc='upper right', framealpha=0.95)
        ax[1, 0].autoscale(enable=True, axis='x', tight=True)
        
        # Subplot 3: y_raw_normalized and mu_pr samples
        ax[2, 1].set_axis_off()
        ax[2, 0].plot(t_in, y_raw, linewidth=1.5, color=colors['gt'], label='Ground Truth', alpha=0.85, zorder=2)
        
        # Handle different formats of mu_samples
        if len(mu_samples.shape) == 1:  # (4800,) format - single prediction
            ax[2, 0].plot(
                t_in, mu_samples, linewidth=1.5, color=colors['samples'], 
                label='Model Prediction', alpha=0.85, zorder=1)
        else:  # (300, 4800) format - multiple predictions
            # Select specific time indices: [30, 60, 90, 120, 150, 180, 210, 240, 270]
            selected_indices = [idx for idx in [30, 60, 90, 120, 150, 180, 210, 240, 270] if idx < mu_samples.shape[0]]
            
            if selected_indices:
                # Handle NaN values and sum selected samples
                selected_samples = mu_samples[selected_indices, :]  # Shape: (len(selected_indices), 4800)
                
                # Remove NaN values and compute mean
                valid_mask = ~np.isnan(selected_samples)
                summed_samples = np.zeros(4800)
                
                for i in range(4800):
                    valid_values = selected_samples[:, i][valid_mask[:, i]]
                    if len(valid_values) > 0:
                        summed_samples[i] = np.sum(valid_values)
                    else:
                        summed_samples[i] = 0
                
                ax[2, 0].plot(
                    t_in, summed_samples, linewidth=1.5, color=colors['samples'], 
                    label='Selected Samples Sum', alpha=0.85, zorder=1)
            else:
                # Fallback to first sample if no valid indices
                ax[2, 0].plot(
                    t_in, mu_samples[0, :], linewidth=1.5, color=colors['samples'], 
                    label='First Sample', alpha=0.85, zorder=1)
        
        ax[2, 0].set_ylabel('FHR (bpm)', fontweight='normal')
        ax[2, 0].set_title('FHR vs Model Reconstructions', fontweight='normal', pad=12)
        ax[2, 0].legend(loc='upper right', framealpha=0.95)
        ax[2, 0].autoscale(enable=True, axis='x', tight=True)
        
        # Subplot 4: latent_z with imshow
        imgplot = ax[3, 0].imshow(z_latent.T, aspect='auto', cmap='bwr', origin='lower')
        ax[3, 1].set_axis_on()
        cbar = fig.colorbar(imgplot, cax=ax[3, 1])
        cbar.ax.tick_params(labelsize=10, colors='#666666')
        cbar.set_label('Activation', fontweight='normal', fontsize=11, color='#666666')
        cbar.outline.set_color('#A2B9A7')
        cbar.outline.set_linewidth(0.7)
        ax[3, 0].set_ylabel('Latent Dimensions', fontweight='normal')
        ax[3, 0].set_xlabel('Time Steps', fontweight='normal')
        ax[3, 0].set_title('Latent Space Representation', fontweight='normal', pad=12)
        
        # Set overall title with scientific paper styling
        fig.suptitle(f'Model Performance Analysis — Epoch {epoch}', 
                    fontsize=14, fontweight='normal', y=0.97, color='#456882')
        
        # Save plot as PDF with high quality
        save_path = os.path.join(self.output_dir, f'model_results_epoch_{epoch}.pdf')
        plt.savefig(save_path, bbox_inches='tight', orientation='landscape', dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Clean up memory  
        del y_raw, up_raw, mu_means, log_var, z_latent
        if 'mu_samples' in locals():
            del mu_samples
        if 'logvar_samples' in locals():
            del logvar_samples
        gc.collect()
        
        logger.info(f"Model results plot saved to {save_path}")


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
        self.history = {
            "epoch": [],
            "train/total_loss": [],
            "train/recon_loss": [],
            "train/mse_loss": [],
            "train/nll_loss": [],
            "train/kld_loss": [],
            "val/total_loss": [],
            "val/recon_loss": [],
            "val/mse_loss": [],
            "val/nll_loss": [],
            "val/kld_loss": []
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
        beta_const_val: float = 1.0
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
        """
        super().__init__()
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
        # Log learning rate at the start of each epoch
        try:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr', lr, on_epoch=True, prog_bar=True, logger=True)
        except IndexError:
            # This can happen if the optimizer is not yet configured
            pass

    def _common_step(self, batch, batch_idx):
        """SPEED OPTIMIZED: Removed expensive permute operations - data comes pre-permuted from dataset."""
        # SPEED OPTIMIZATION: Data now comes in the correct format from the optimized dataset
        # No expensive permute operations needed - significant speedup!
        # Optimized dataloader provides tensors in model-ready format:
        y_st = batch.fhr_st    # Scattering transform features (batch, sequence, channels) - ready for model
        y_ph = batch.fhr_ph    # Phase harmonic features (batch, sequence, channels) - ready for model
        x_ph = batch.fhr_up_ph # Cross-phase features (batch, sequence, channels) - ready for model
        y_raw = batch.fhr      # Raw signal for reconstruction (batch, sequence_length)

        # Forward pass without gradient checkpointing for speed
        forward_outputs = self.model(y_st, y_ph, x_ph)

        loss_dict = self.model.compute_loss(
            forward_outputs, y_st, y_ph, y_raw, compute_kld_loss=True, beta=self.hparams.beta
        )

        return loss_dict

    def training_step(self, batch, batch_idx):
        """Defines the training loop with memory optimization."""
        loss_dict = self._common_step(batch, batch_idx)
        total_loss = loss_dict['total_loss']  # Total loss already includes beta-weighted KLD

        # Log training metrics
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/recon_loss', loss_dict['reconstruction_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/mse_loss', loss_dict['mse_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/nll_loss', loss_dict['nll_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/kld_loss', loss_dict['kld_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Clear loss_dict to free memory
        del loss_dict

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation loop with memory optimization."""
        loss_dict = self._common_step(batch, batch_idx)
        total_loss = loss_dict['total_loss']  # Total loss already includes beta-weighted KLD

        # Log validation metrics
        self.log('val/total_loss', total_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/recon_loss', loss_dict['reconstruction_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val/mse_loss', loss_dict['mse_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val/nll_loss', loss_dict['nll_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val/kld_loss', loss_dict['kld_loss'], on_epoch=True, prog_bar=True, logger=True)

        # Clear loss_dict to free memory
        del loss_dict

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
        """Configure optimizers and learning rate schedulers with memory-efficient settings."""
        # Use AdamW with weight decay for better generalization and memory efficiency
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,  # L2 regularization
            eps=1e-8,  # Numerical stability
            betas=(0.9, 0.999)  # Default Adam betas
        )

        if self.hparams.lr_milestones:
            # Use cosine annealing with restarts for better convergence
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(self.hparams.lr_milestones) // 4,  # Restart every quarter of training
                T_mult=1,
                eta_min=self.hparams.lr * 0.01  # Minimum LR is 1% of initial
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Step-wise for smoother updates
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
