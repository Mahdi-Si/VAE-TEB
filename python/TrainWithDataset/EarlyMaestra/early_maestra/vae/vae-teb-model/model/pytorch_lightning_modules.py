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
os.environ['PYDEVD_USE_CYTHON']="NO"


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

        # Fetch a single batch from the validation dataloader
        try:
            if hasattr(pl_trainer, 'datamodule') and pl_trainer.datamodule is not None:
                val_dataloader = pl_trainer.datamodule.val_dataloader()
            else:
                # Access validation dataloader directly from trainer
                val_dataloader = pl_trainer.val_dataloaders
                if isinstance(val_dataloader, list):
                    val_dataloader = val_dataloader[0]

            batch = next(iter(val_dataloader))
            logger.info("Successfully fetched batch from validation dataloader")
        except (StopIteration, AttributeError, IndexError) as e:
            logger.warning(f"Could not get a batch from validation dataloader for plotting: {e}")
            return

        # Ensure batch is on the correct device (use proper device, not hardcoded 0)
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, pl_module.local_rank)

        pl_module.eval()
        try:
            with torch.no_grad():
                # Check if this is the correct Lightning module type
                if not isinstance(pl_module, LightSeqVaeTeb):
                    logger.warning(f"PlottingCallback received unexpected module type: {type(pl_module)}. Expected LightSeqVaeTeb.")
                    return

                logger.info("Accessing batch data...")
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw_normalized = batch.fhr  # This is the normalized FHR from dataset
                up_raw_normalized = batch.up   # This is the normalized UP from dataset

                logger.info(f"Batch shapes - y_st: {y_st.shape}, y_ph: {y_ph.shape}, x_ph: {x_ph.shape}, y_raw: {y_raw_normalized.shape}")

                # Simple full forward pass - no windowed prediction
                logger.info("Running model forward pass...")
                model_outputs = pl_module.model(y_st, y_ph, x_ph)
                logger.info(f"Model forward pass successful. Output keys: {list(model_outputs.keys())}")
                
                # OPTIMIZATION: Move data to CPU immediately after forward pass to reduce GPU 0 load
                # Transfer all required data to CPU for plotting operations
                device_info = {
                    'device_id': pl_module.device.index if hasattr(pl_module.device, 'index') else 0,
                    'device_type': pl_module.device.type
                }

                if 'raw_predictions' not in model_outputs:
                    logger.error("Model output missing 'raw_predictions' key")
                    return

                raw_predictions = model_outputs['raw_predictions']
                if 'raw_signal_mu' not in raw_predictions or 'raw_signal_logvar' not in raw_predictions:
                    logger.error(f"Raw predictions missing required keys. Available keys: {list(raw_predictions.keys())}")
                    return

                # Get model parameters for plotting
                warmup_period = getattr(pl_module.model, 'warmup_period', 30)
                decimation_factor = getattr(pl_module.model, 'decimation_factor', 16)
                B, S, prediction_horizon = raw_predictions['raw_signal_mu'].shape
                logger.info(f"Model parameters - warmup_period: {warmup_period}, decimation_factor: {decimation_factor}")
                logger.info(f"Prediction shape: (B={B}, S={S}, prediction_horizon={prediction_horizon})")

                # OPTIMIZATION: Move all tensor operations to CPU immediately to reduce GPU load
                # Extract predictions for the first sample in the batch and move to CPU
                # raw_predictions now contain (B, S, prediction_horizon) - predictions from each timestep
                pred_mu = raw_predictions['raw_signal_mu'][0].detach().cpu().numpy()  # (S, prediction_horizon)
                pred_logvar = raw_predictions['raw_signal_logvar'][0].detach().cpu().numpy()  # (S, prediction_horizon)
                pred_std = np.exp(0.5 * pred_logvar)

                # Ground truth (first sample in batch) - normalized raw signals - move to CPU immediately
                ground_truth_fhr = y_raw_normalized[0].squeeze().detach().cpu().numpy()
                ground_truth_up = up_raw_normalized[0].squeeze().detach().cpu().numpy()
                
                # Get latent representation and move to CPU immediately
                z_latent = None
                if 'z' in model_outputs:
                    z_latent = model_outputs['z'][0].permute(1, 0).detach().cpu().numpy()  # (latent_dim, seq_len)
                
                # Get batch info and move to CPU
                try:
                    guid = batch.guid[0] if hasattr(batch, 'guid') else 'unknown'
                    epoch_info = batch.epoch[0].item() if hasattr(batch, 'epoch') else 'unknown'
                except:
                    guid = 'unknown'
                    epoch_info = 'unknown'
                
                # Free GPU memory early by deleting GPU tensors
                del model_outputs, raw_predictions
                del y_st, y_ph, x_ph, y_raw_normalized, up_raw_normalized
                # Force GPU memory cleanup before heavy CPU plotting
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                logger.info(f"Moved data to CPU for plotting. Device was: {device_info}")

                # Select a representative timepoint for simple visualization
                # Use a middle timepoint after warmup for plotting
                if S > warmup_period + 10:
                    selected_timestep = min(int(S * 0.6), S - 1)  # Use 60% through sequence
                else:
                    selected_timestep = max(warmup_period, S - 1)
                
                # Extract prediction for the selected timestep
                pred_mu_selected = pred_mu[selected_timestep]  # (prediction_horizon,)
                pred_std_selected = pred_std[selected_timestep]  # (prediction_horizon,)
                
                # Calculate where this prediction should be placed in the raw signal
                raw_start = selected_timestep * decimation_factor
                raw_end = raw_start + prediction_horizon
                
                logger.info(f"Selected timestep {selected_timestep} for plotting, raw_start: {raw_start}, raw_end: {raw_end}")

                logger.info(f"Data shapes for plotting - pred_mu: {pred_mu.shape}, ground_truth_fhr: {ground_truth_fhr.shape}, ground_truth_up: {ground_truth_up.shape}")
                if z_latent is not None:
                    logger.info(f"Latent shape: {z_latent.shape}")

                # --- Create plots with proper data handling and professional styling ---
                # Professional pastel color palette for research papers
                colors = {
                    'ground_truth': '#2E5984',  # Deep blue-gray (professional)
                    'prediction': '#C7522A',  # Muted red-orange
                    'uncertainty': '#E5B181',  # Light peach
                    'grid': '#E8E8E8',  # Light gray for grid
                    'background': '#FAFAFA'  # Very light gray background
                }

                # Set the overall style
                plt.style.use('default')  # Reset to default first
                plt.rcParams.update({
                    'figure.facecolor': colors['background'],
                    'axes.facecolor': 'white',
                    'axes.edgecolor': '#CCCCCC',
                    'axes.linewidth': 0.8,
                    'grid.color': colors['grid'],
                    'grid.linewidth': 0.5,
                    'font.size': 10,
                    'axes.titlesize': 11,
                    'axes.labelsize': 10,
                    'legend.fontsize': 9,
                    'xtick.labelsize': 9,
                    'ytick.labelsize': 9
                })

                # Create a 3x2 grid for a more comprehensive dashboard
                fig, axes = plt.subplots(3, 2, figsize=(22, 18), facecolor=colors['background'])
                ax = axes.flatten()

                # --- Plot 1: Main Prediction View (Ground Truth vs. Prediction) ---
                ax[0].plot(ground_truth_fhr, color='gray', linewidth=1, alpha=0.8, label='Full Ground Truth')
                # Show history leading up to the prediction
                context_window = 150
                context_start = max(0, raw_start - context_window)
                ax[0].plot(range(context_start, raw_start), ground_truth_fhr[context_start:raw_start], color='black', linestyle='--', label='History Window')
                
                # Plot the ground truth for the predicted window for direct comparison
                if raw_end <= len(ground_truth_fhr):
                    ax[0].plot(range(raw_start, raw_end), ground_truth_fhr[raw_start:raw_end], color=colors['ground_truth'], linewidth=1.5, label='Actual Future')
                
                # Plot the prediction with uncertainty
                prediction_indices = range(raw_start, raw_end)
                ax[0].plot(prediction_indices, pred_mu_selected, color=colors['prediction'], linewidth=1.5, label='Predicted Future (μ)')
                ax[0].fill_between(prediction_indices, 
                                 pred_mu_selected - pred_std_selected, 
                                 pred_mu_selected + pred_std_selected,
                                 color=colors['uncertainty'], alpha=0.6, label='Uncertainty (±1σ)')
                ax[0].axvline(x=raw_start, color='red', linestyle=':', linewidth=1.5, label='Prediction Start')
                ax[0].set_title(f'Main Prediction View (from Timestep {selected_timestep})')
                ax[0].set_ylabel('FHR (Normalized)')
                ax[0].legend(fontsize=8)
                ax[0].grid(True)

                # --- Plot 2: Prediction Error Analysis ---
                if raw_end <= len(ground_truth_fhr):
                    target_window = ground_truth_fhr[raw_start:raw_end]
                    error = target_window - pred_mu_selected
                    mse = np.mean(error**2)
                    
                    ax[1].plot(prediction_indices, error, color='purple', label='Prediction Error (Actual - Predicted)')
                    ax[1].axhline(0, color='black', linestyle='--', linewidth=1)
                    ax[1].set_title(f'Prediction Error (MSE: {mse:.4f})')
                    ax[1].set_ylabel('Error')
                    ax[1].legend(fontsize=8)
                    ax[1].grid(True)
                else:
                    ax[1].text(0.5, 0.5, 'Ground truth not available for error plot', ha='center', va='center')
                    ax[1].set_title('Prediction Error')

                # --- Plot 3: Latent Space (z) Heatmap ---
                if z_latent is not None:
                    im = ax[2].imshow(z_latent.T, aspect='auto', cmap='viridis', interpolation='nearest', origin='lower')
                    ax[2].set_title('Latent Representation (z)')
                    ax[2].set_xlabel('Time Steps')
                    ax[2].set_ylabel('Latent Dimensions')
                    fig.colorbar(im, ax=ax[2])
                else:
                    ax[2].text(0.5, 0.5, 'Latent z not available', ha='center', va='center')
                    ax[2].set_title('Latent Representation (z)')

                # --- Plot 4: Latent Variable Distributions (Posterior) ---
                if 'mu_post' in model_outputs and 'logvar_post' in model_outputs:
                    mu_post_sample = model_outputs['mu_post'][0].detach().cpu().numpy().flatten()
                    logvar_post_sample = model_outputs['logvar_post'][0].detach().cpu().numpy().flatten()
                    
                    ax[3].hist(mu_post_sample, bins=50, alpha=0.7, label='Posterior μ', color='skyblue', density=True)
                    ax[3].set_xlabel('Value')
                    ax[3].set_ylabel('Density')
                    ax[3].legend(loc='upper left', fontsize=8)
                    
                    ax3_twin = ax[3].twinx()
                    ax3_twin.hist(logvar_post_sample, bins=50, alpha=0.7, label='Posterior log(σ²)', color='salmon', density=True)
                    ax3_twin.set_ylabel('Density')
                    ax3_twin.legend(loc='upper right', fontsize=8)
                    ax[3].set_title('Posterior Latent Distributions')
                    ax[3].grid(True)

                # --- Plot 5: Prior vs. Posterior (Mu) ---
                if 'mu_prior' in model_outputs and 'mu_post' in model_outputs:
                    mu_prior_sample = model_outputs['mu_prior'][0].detach().cpu().numpy()
                    mu_post_sample = model_outputs['mu_post'][0].detach().cpu().numpy()
                    dims_to_plot = min(3, mu_prior_sample.shape[1])
                    for i in range(dims_to_plot):
                        ax[4].plot(mu_prior_sample[:, i], '--', color=f'C{i}', label=f'Prior μ (dim {i})')
                        ax[4].plot(mu_post_sample[:, i], '-', color=f'C{i}', label=f'Posterior μ (dim {i})')
                    ax[4].set_title('Prior vs. Posterior (μ)')
                    ax[4].set_xlabel('Time Steps')
                    ax[4].set_ylabel('Value')
                    ax[4].legend(fontsize=8)
                    ax[4].grid(True)

                # --- Plot 6: Prior vs. Posterior (Logvar) ---
                if 'logvar_prior' in model_outputs and 'logvar_post' in model_outputs:
                    logvar_prior_sample = model_outputs['logvar_prior'][0].detach().cpu().numpy()
                    logvar_post_sample = model_outputs['logvar_post'][0].detach().cpu().numpy()
                    dims_to_plot = min(3, logvar_prior_sample.shape[1])
                    for i in range(dims_to_plot):
                        ax[5].plot(logvar_prior_sample[:, i], '--', color=f'C{i}', label=f'Prior log(σ²) (dim {i})')
                        ax[5].plot(logvar_post_sample[:, i], '-', color=f'C{i}', label=f'Posterior log(σ²) (dim {i})')
                    ax[5].set_title('Prior vs. Posterior (log σ²)')
                    ax[5].set_xlabel('Time Steps')
                    ax[5].set_ylabel('Value')
                    ax[5].legend(fontsize=8)
                    ax[5].grid(True)
                
                # Finalize and save the plot
                fig.suptitle(f"SeqVaeTeb Performance Dashboard - Epoch {pl_trainer.current_epoch} - GUID: {guid}", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle and legend
                
                save_path = os.path.join(self.output_dir, f"epoch_{pl_trainer.current_epoch}_dashboard.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close(fig)
                logger.info(f"Saved performance dashboard to {save_path}")

        except Exception as e:
            logger.error(f"Error during plotting: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            pl_module.train()


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
            "train/kld_loss": [],
            "train/raw_signal_loss": [],
            "val/total_loss": [],
            "val/recon_loss": [],
            "val/kld_loss": [],
            "val/raw_signal_loss": []
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
    def __init__(self,
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
        self.model.kld_beta = beta
        return beta

    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        beta = self._calculate_beta()
        self.log('kld_beta', beta, on_epoch=True, prog_bar=True)
        # Log learning rate at the start of each epoch
        try:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr', lr, on_epoch=True, prog_bar=True, logger=True)
        except IndexError:
            # This can happen if the optimizer is not yet configured
            pass
        
        # Clear GPU cache at the start of each epoch - device-aware
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

    def _common_step(self, batch, batch_idx):
        """Optimized common logic for training and validation steps with aggressive memory management."""
        # Access data using correct HDF5 dataset field names  
        y_st = batch.fhr_st      # Scattering transform features
        y_ph = batch.fhr_ph      # Phase harmonic features
        x_ph = batch.fhr_up_ph   # Cross-phase features
        y_raw = batch.fhr        # Raw signal for reconstruction
        
        # Use gradient checkpointing for forward pass to save memory
        if self.training:
            forward_outputs = torch.utils.checkpoint.checkpoint(
                self.model, y_st, y_ph, x_ph, use_reentrant=False
            )
        else:
            forward_outputs = self.model(y_st, y_ph, x_ph)
        
        loss_dict = self.model.compute_loss(
            forward_outputs, y_raw, compute_kld_loss=True
        )
        
        # Aggressive cleanup to free memory immediately
        del y_st, y_ph, x_ph, y_raw
        
        # Clean up forward outputs except what's needed for loss
        if isinstance(forward_outputs, dict):
            keys_to_keep = set()  # Don't keep anything after loss computation
            for key in list(forward_outputs.keys()):
                if key not in keys_to_keep:
                    if key in forward_outputs:
                        del forward_outputs[key]
        
        # Force garbage collection on every 10th step
        if batch_idx % 10 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return loss_dict

    def training_step(self, batch, batch_idx):
        """Defines the training loop with memory optimization."""
        loss_dict = self._common_step(batch, batch_idx)
        total_loss = loss_dict['total_loss']

        # Log training metrics
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/recon_loss', loss_dict['reconstruction_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/kld_loss', loss_dict['kld_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/raw_signal_loss', loss_dict['raw_signal_loss'], on_step=False, on_epoch=True, logger=True)

        # Clear loss_dict to free memory
        del loss_dict
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation loop with memory optimization."""
        loss_dict = self._common_step(batch, batch_idx)
        total_loss = loss_dict['total_loss']

        # Log validation metrics
        self.log('val/total_loss', total_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/recon_loss', loss_dict['reconstruction_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val/kld_loss', loss_dict['kld_loss'], on_epoch=True, prog_bar=True, logger=True)
        self.log('val/raw_signal_loss', loss_dict['raw_signal_loss'], on_epoch=True, logger=True)

        # Clear loss_dict to free memory
        del loss_dict
        
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Optimized cleanup after each training batch."""
        # More aggressive memory management for training
        if batch_idx % 5 == 0 and torch.cuda.is_available():  # More frequent clearing
            current_device = torch.cuda.current_device()
            with torch.cuda.device(current_device):
                torch.cuda.empty_cache()
        
        # Clean up batch references
        del batch
        
        # Periodic garbage collection
        if batch_idx % 20 == 0:
            import gc
            gc.collect()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Optimized cleanup after each validation batch."""
        # Aggressive validation cleanup
        if batch_idx % 3 == 0 and torch.cuda.is_available():  # More frequent for validation
            current_device = torch.cuda.current_device()
            with torch.cuda.device(current_device):
                torch.cuda.empty_cache()
        
        # Clean up batch references
        del batch

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers with memory-efficient settings."""
        # Use AdamW with weight decay for better generalization and memory efficiency
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=1e-4,  # L2 regularization
            eps=1e-8,          # Numerical stability
            betas=(0.9, 0.999) # Default Adam betas
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
    """
    def __init__(self, threshold_gb=10.0, log_frequency=50):
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
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
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
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                if allocated > self.threshold_gb:
                    logger.warning(f"GPU {device_id} memory usage ({allocated:.2f}GB) exceeds threshold ({self.threshold_gb}GB). Clearing cache...")
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
        """Clear memory and log usage at the end of each epoch."""
        self._log_memory_usage(f"Epoch {trainer.current_epoch} end")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
