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
                sequence_length = raw_predictions['raw_signal_mu'].shape[1]  # S timesteps
                
                logger.info(f"Model parameters - warmup_period: {warmup_period}, decimation_factor: {decimation_factor}, sequence_length: {sequence_length}")

                # OPTIMIZATION: Move all tensor operations to CPU immediately to reduce GPU load
                # Extract predictions for the first sample in the batch and move to CPU
                # raw_predictions contain (B, S, 480) - predictions of next 480 samples from each timestep
                pred_mu_all = raw_predictions['raw_signal_mu'][0].detach().cpu().numpy()  # (S, 480)
                pred_logvar_all = raw_predictions['raw_signal_logvar'][0].detach().cpu().numpy()  # (S, 480)
                pred_std_all = np.exp(0.5 * pred_logvar_all)

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

                # Create average predictions considering warmup period
                # For timesteps after warmup, average all predictions that contribute to each raw signal sample
                raw_signal_length = len(ground_truth_fhr)
                avg_pred_mu = np.zeros(raw_signal_length)
                avg_pred_var = np.zeros(raw_signal_length)
                prediction_counts = np.zeros(raw_signal_length)

                # Only use predictions from timesteps after warmup period
                for t in range(warmup_period, sequence_length):
                    # Each timestep t predicts the next 480 samples starting from its position
                    start_raw = t * decimation_factor  # Start from current timestep position
                    end_raw = start_raw + 480  # Next 480 samples
                    
                    if end_raw <= raw_signal_length:
                        # Add this prediction to the average
                        avg_pred_mu[start_raw:end_raw] += pred_mu_all[t, :]
                        avg_pred_var[start_raw:end_raw] += pred_std_all[t, :] ** 2  # Add variances
                        prediction_counts[start_raw:end_raw] += 1

                # Normalize by count to get average
                valid_mask = prediction_counts > 0
                avg_pred_mu[valid_mask] /= prediction_counts[valid_mask]
                avg_pred_var[valid_mask] /= prediction_counts[valid_mask]
                avg_pred_std = np.sqrt(avg_pred_var)

                logger.info(f"Data shapes for plotting - avg_pred_mu: {avg_pred_mu.shape}, ground_truth_fhr: {ground_truth_fhr.shape}, ground_truth_up: {ground_truth_up.shape}")
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

                fig = plt.figure(figsize=(16, 15), facecolor=colors['background'])

                # Create a grid layout with 5 rows for the new subplot
                gs = fig.add_gridspec(5, 30, hspace=0.4, wspace=0.05)

                # Main plots take up most of the width (columns 0-27), colorbar is thinner (28-29)
                ax1 = fig.add_subplot(gs[0, :27])  # Ground truth
                ax2 = fig.add_subplot(gs[1, :27])  # Average predictions with uncertainty
                ax3 = fig.add_subplot(gs[2, :27])  # Single predictions (non-overlapping)
                ax4 = fig.add_subplot(gs[3, :27])  # Comparison overlay
                ax5 = fig.add_subplot(gs[4, :27])  # Latent representation

                # Thinner colorbar axis
                cbar_ax = fig.add_subplot(gs[4, 28:])

                # Time axis in minutes (assuming 4Hz sampling rate)
                time_axis = np.arange(len(ground_truth_fhr)) / 4.0 / 60  # Convert to minutes

                # Plot 1: Ground truth raw signals (FHR and UP)
                colors['ground_truth_up'] = '#8E44AD'  # Purple for UP signal
                
                ax1.plot(time_axis, ground_truth_fhr, color=colors['ground_truth'],
                         linewidth=1.2, label='Ground Truth FHR', alpha=0.9)
                ax1.plot(time_axis, ground_truth_up, color=colors['ground_truth_up'],
                         linewidth=1.2, label='Ground Truth UP', alpha=0.9)
                ax1.set_title('Ground Truth Raw FHR and UP Signals', fontweight='medium', pad=15)
                ax1.set_ylabel('Normalized Amplitude')
                ax1.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
                ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                ax1.set_facecolor('white')

                # Plot 2: Average predicted raw signal with uncertainty
                if np.any(valid_mask):
                    # Only plot where we have valid predictions
                    valid_time = time_axis[valid_mask]
                    valid_avg_mu = avg_pred_mu[valid_mask]
                    valid_avg_std = avg_pred_std[valid_mask]

                    # Plot uncertainty band first (so it appears behind the line)
                    ax2.fill_between(valid_time,
                                     valid_avg_mu - valid_avg_std,
                                     valid_avg_mu + valid_avg_std,
                                     alpha=0.25, color=colors['uncertainty'],
                                     label='±1σ Uncertainty', edgecolor='none')

                    # Plot prediction line on top
                    ax2.plot(valid_time, valid_avg_mu,
                             color=colors['prediction'], linewidth=1.2,
                             label='Average Predicted Mean', alpha=0.9)

                    ax2.set_title('Average Predicted Raw FHR Signal with Uncertainty (After Warmup)', fontweight='medium', pad=15)
                    ax2.set_ylabel('Normalized Amplitude')
                    ax2.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
                    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                    ax2.set_facecolor('white')
                else:
                    ax2.text(0.5, 0.5, 'No valid predictions after warmup period',
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax2.transAxes, fontsize=11, color='#666666')
                    ax2.set_title('Average Predicted Raw FHR Signal (No Valid Data)', fontweight='medium', pad=15)
                    ax2.set_facecolor('white')

                # Plot 3: Non-overlapping single predictions with gaps
                # Use current warmup period from model
                warmup_raw_samples = warmup_period * decimation_factor  # warmup in raw signal samples
                
                colors['single_pred1'] = '#E74C3C'  # Red
                colors['single_pred2'] = '#3498DB'  # Blue  
                colors['single_pred3'] = '#2ECC71'  # Green
                colors['single_pred4'] = '#F39C12'  # Orange
                colors['single_pred5'] = '#9B59B6'  # Purple
                colors['raw_signal'] = '#34495E'     # Dark gray for raw signal
                
                # Calculate truly non-overlapping prediction windows
                # Start from warmup period and create consecutive 2-minute (480-sample) windows
                prediction_windows = []
                
                # Start right after warmup period and create non-overlapping windows
                current_timestep = warmup_period
                
                # Create consecutive 2-minute prediction windows to cover entire signal
                # Each timestep predicts NEXT 480 samples, so we need to account for gaps
                while current_timestep < sequence_length:
                    start_raw = current_timestep * decimation_factor
                    end_raw = start_raw + 480
                    
                    if end_raw <= raw_signal_length:
                        # Calculate the gap before next window
                        next_window_start = end_raw  # Next window should start where this one ends
                        next_timestep = next_window_start // decimation_factor
                        
                        prediction_windows.append({
                            'timestep': current_timestep,
                            'raw_start': start_raw,
                            'raw_end': end_raw,
                            'gap_start': end_raw,
                            'gap_end': next_timestep * decimation_factor  # Gap until next aligned timestep
                        })
                        
                        # Move to next timestep that aligns with end of current window
                        current_timestep = next_timestep
                    else:
                        break

                if prediction_windows:
                    pred_colors = [colors['single_pred1'], colors['single_pred2'], colors['single_pred3'], 
                                  colors['single_pred4'], colors['single_pred5']]
                    
                    # Plot the raw FHR signal as background
                    ax3.plot(time_axis, ground_truth_fhr, color=colors['raw_signal'], 
                            linewidth=0.8, alpha=0.6, label='Ground Truth FHR', zorder=1)
                    
                    # Plot each non-overlapping prediction window - show all windows to cover full 20 minutes
                    for i, window in enumerate(prediction_windows):  # Show all available windows
                        t = window['timestep']
                        start_raw = window['raw_start']
                        end_raw = window['raw_end']
                        color = pred_colors[i % len(pred_colors)]
                        
                        # Extract prediction for this timestep
                        if t < pred_mu_all.shape[0]:
                            single_pred = pred_mu_all[t, :]  # (480,)
                            single_std = pred_std_all[t, :]  # (480,)
                            single_time = time_axis[start_raw:end_raw]
                            
                            # Plot uncertainty band
                            ax3.fill_between(single_time, single_pred - single_std, single_pred + single_std,
                                            alpha=0.2, color=color, edgecolor='none', zorder=2)
                            
                            # Plot prediction line
                            ax3.plot(single_time, single_pred, color=color, linewidth=1.2,
                                    label=f'Window {i+1} (t={t})' if i < 5 else "", alpha=0.9, zorder=3)
                    
                    # Mark gaps between prediction windows
                    for i, window in enumerate(prediction_windows):
                        if 'gap_start' in window and 'gap_end' in window:
                            gap_start = window['gap_start']
                            gap_end = window['gap_end']
                            
                            # Only show gap if it's meaningful (more than a few samples)
                            if gap_end > gap_start and gap_start < len(time_axis) and gap_end <= len(time_axis):
                                gap_time_start = time_axis[gap_start]
                                gap_time_end = time_axis[gap_end-1] if gap_end-1 < len(time_axis) else time_axis[-1]
                                ax3.axvspan(gap_time_start, gap_time_end, alpha=0.3, color='red', 
                                           label='Gap (no predictions)' if i == 0 else "", zorder=0)

                    ax3.set_title(f'Non-Overlapping 2-Min Prediction Windows (After {warmup_period}-timestep Warmup)', 
                                 fontweight='medium', pad=15)
                    ax3.set_ylabel('Normalized Amplitude')
                    ax3.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9, fontsize=8)
                    ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                    ax3.set_facecolor('white')
                    
                    # Add annotation about the implementation
                    warmup_minutes = warmup_raw_samples / 4.0 / 60.0  # Convert to minutes
                    total_gaps = sum(1 for w in prediction_windows if w.get('gap_end', 0) > w.get('gap_start', 0))
                    ax3.text(0.02, 0.98, f'Warmup: {warmup_period} timesteps ({warmup_raw_samples} samples = {warmup_minutes:.1f} min)\nEach window: 480 samples (2 min)\nWindows: {len(prediction_windows)} total, Gaps: {total_gaps}', 
                            transform=ax3.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    ax3.text(0.5, 0.5, 'No valid prediction windows available',
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax3.transAxes, fontsize=11, color='#666666')
                    ax3.set_title('Single Predictions (No Valid Data)', fontweight='medium', pad=15)
                    ax3.set_facecolor('white')

                # Plot 4: Comparison overlay with refined styling
                if np.any(valid_mask):
                    # Use valid predictions for comparison
                    comparison_len = min(len(time_axis), len(avg_pred_mu))
                    time_comp = time_axis[:comparison_len]
                    gt_comp = ground_truth_fhr[:comparison_len]
                    pred_comp = avg_pred_mu[:comparison_len]
                    
                    # Only plot where we have valid predictions
                    valid_comp_mask = valid_mask[:comparison_len]
                    
                    if np.any(valid_comp_mask):
                        # Ground truth with slightly thicker line
                        ax4.plot(time_comp, gt_comp, color=colors['ground_truth'],
                                linewidth=1.4, alpha=0.85, label='Ground Truth')

                        # Average prediction with thinner solid line for better comparison
                        ax4.plot(time_comp[valid_comp_mask], pred_comp[valid_comp_mask],
                                color=colors['prediction'], linewidth=0.9, alpha=0.9, 
                                label='Average Predicted', linestyle='-')

                        ax4.set_title('Ground Truth vs Average Predicted Raw FHR Signal', fontweight='medium', pad=15)
                        ax4.set_ylabel('Normalized Amplitude')
                        ax4.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
                        ax4.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                        ax4.set_facecolor('white')

                        # Calculate metrics on valid data only
                        gt_valid = gt_comp[valid_comp_mask]
                        pred_valid = pred_comp[valid_comp_mask]
                        
                        if len(gt_valid) > 0:
                            mse = np.mean((gt_valid - pred_valid) ** 2)
                            mae = np.mean(np.abs(gt_valid - pred_valid))
                            correlation = np.corrcoef(gt_valid, pred_valid)[0, 1] if len(gt_valid) > 1 else 0
                        else:
                            mse, mae, correlation = np.nan, np.nan, np.nan
                    else:
                        ax4.text(0.5, 0.5, 'No valid predictions for comparison',
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax4.transAxes, fontsize=11, color='#666666')
                        ax4.set_title('Ground Truth vs Predicted (No Valid Data)', fontweight='medium', pad=15)
                        ax4.set_facecolor('white')
                        mse, mae, correlation = np.nan, np.nan, np.nan
                else:
                    ax4.text(0.5, 0.5, 'No valid predictions for comparison',
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax4.transAxes, fontsize=11, color='#666666')
                    ax4.set_title('Ground Truth vs Predicted (No Valid Data)', fontweight='medium', pad=15)
                    ax4.set_facecolor('white')
                    mse, mae, correlation = np.nan, np.nan, np.nan

                # Plot 5: Latent representation with professional colormap and thin colorbar
                if z_latent is not None and z_latent.size > 0:
                    # Use a professional colormap suitable for research papers
                    im = ax5.imshow(z_latent, aspect='auto', cmap='RdYlBu_r',
                                    interpolation='nearest', alpha=0.9)
                    ax5.set_title('Latent Representation', fontweight='medium', pad=15)
                    ax5.set_xlabel('Time Steps')
                    ax5.set_ylabel('Latent Dimensions')
                    ax5.set_facecolor('white')

                    # Add thin colorbar to the dedicated axis on the right
                    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
                    cbar_ax.set_ylabel('Latent Value', rotation=270, labelpad=12, fontsize=9)
                    cbar.ax.tick_params(labelsize=8)

                    # Style the colorbar
                    cbar.outline.set_linewidth(0.5)
                    cbar.outline.set_edgecolor('#CCCCCC')
                else:
                    ax5.text(0.5, 0.5, 'Latent representation not available',
                             horizontalalignment='center', verticalalignment='center',
                             transform=ax5.transAxes, fontsize=11, color='#666666')
                    ax5.set_title('Latent Representation (N/A)', fontweight='medium', pad=15)
                    ax5.set_xlabel('Time Steps')
                    ax5.set_ylabel('Latent Dimensions')
                    ax5.set_facecolor('white')
                    # Hide the colorbar axis if no latent data
                    cbar_ax.set_visible(False)

                # Batch info already retrieved above

                # Overall title with metrics and warmup info
                plt.suptitle(f"Raw Signal Prediction - GUID: {guid}, Epoch: {epoch_info}\n"
                             f"Warmup Period: {warmup_period} timesteps, MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}",
                             fontsize=13, y=0.98)

                plot_path = f"{self.output_dir}/raw_signal_prediction_e-{pl_trainer.current_epoch}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Raw signal prediction plot saved to {plot_path}")

                # Explicit cleanup - GPU tensors already cleaned up earlier
                plt.close('all')

                # Clean up remaining CPU arrays
                if z_latent is not None:
                    del z_latent
                del ground_truth_fhr, ground_truth_up
                del pred_mu_all, pred_logvar_all, pred_std_all
                del avg_pred_mu, avg_pred_var, avg_pred_std, prediction_counts
                # Final GPU cache cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Error during plotting: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Ensure cleanup even if plotting fails
            plt.close('all')
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        finally:
            # Ensure we return to training mode
            pl_module.train()
            # Clean up the batch from memory
            del batch


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
        """Common logic for training and validation steps with memory optimization."""
        # Access data using correct HDF5 dataset field names  
        y_st = batch.fhr_st      # Scattering transform features
        y_ph = batch.fhr_ph      # Phase harmonic features
        x_ph = batch.fhr_up_ph   # Cross-phase features
        y_raw = batch.fhr        # Raw signal for reconstruction
        
        # Clear any cached computations
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        
        forward_outputs = self.model(y_st, y_ph, x_ph)
        loss_dict = self.model.compute_loss(
            forward_outputs, y_raw, compute_kld_loss=True
        )
        
        # Clean up intermediate tensors to free memory
        del y_st, y_ph, x_ph, y_raw
        if 'forward_outputs' in locals():
            # Only delete if we have a reference to avoid errors
            for key in list(forward_outputs.keys()):
                if key not in ['z', 'raw_predictions']:  # Keep essential outputs for loss computation
                    forward_outputs.pop(key, None)
        
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
        """Clean up after each training batch."""
        # Periodic GPU cache clearing to prevent memory buildup - device-aware
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            # Clear cache on current device, not just device 0
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            if current_device is not None:
                with torch.cuda.device(current_device):
                    torch.cuda.empty_cache()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Clean up after each validation batch."""
        # Periodic GPU cache clearing to prevent memory buildup - device-aware
        if batch_idx % 5 == 0 and torch.cuda.is_available():
            # Clear cache on current device, not just device 0
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            if current_device is not None:
                with torch.cuda.device(current_device):
                    torch.cuda.empty_cache()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        if self.hparams.lr_milestones:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_milestones,
                gamma=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
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
