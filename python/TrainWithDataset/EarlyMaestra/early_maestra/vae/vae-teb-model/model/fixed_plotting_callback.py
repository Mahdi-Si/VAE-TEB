import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from loguru import logger


class FixedPlottingCallBack(Callback):
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

        # Ensure batch is on the correct device
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)

        pl_module.eval()
        try:
            with torch.no_grad():
                # Check if this is the correct Lightning module type
                from pytorch_lightning_modules import LightSeqVaeTeb
                if not isinstance(pl_module, LightSeqVaeTeb):
                    logger.warning(f"PlottingCallback received unexpected module type: {type(pl_module)}. Expected LightSeqVaeTeb.")
                    return

                logger.info("Accessing batch data...")
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw_normalized = batch.fhr  # This is already normalized from the dataset
                
                logger.info(f"Batch shapes - y_st: {y_st.shape}, y_ph: {y_ph.shape}, x_ph: {x_ph.shape}, y_raw: {y_raw_normalized.shape}")
                
                # Simple full-sequence prediction for plotting
                try:
                    logger.info("Running model forward pass...")
                    # Forward pass through the model
                    model_outputs = pl_module.model(y_st, y_ph, x_ph)
                    logger.info(f"Model forward pass successful. Output keys: {list(model_outputs.keys())}")
                    
                    if 'raw_predictions' not in model_outputs:
                        logger.error("Model output missing 'raw_predictions' key")
                        return
                    
                    raw_predictions = model_outputs['raw_predictions']
                    if 'raw_signal_mu' not in raw_predictions or 'raw_signal_logvar' not in raw_predictions:
                        logger.error(f"Raw predictions missing required keys. Available keys: {list(raw_predictions.keys())}")
                        return
                    
                    logger.info("Extracting predictions...")
                    # Extract predictions - now shape (B, S, prediction_horizon)
                    pred_mu = raw_predictions['raw_signal_mu'][0].detach().cpu().numpy()  # (S, prediction_horizon)
                    pred_logvar = raw_predictions['raw_signal_logvar'][0].detach().cpu().numpy()  # (S, prediction_horizon)
                    pred_std = np.exp(0.5 * pred_logvar)
                    
                    B, S, prediction_horizon = raw_predictions['raw_signal_mu'].shape
                    logger.info(f"Prediction shape: (B={B}, S={S}, prediction_horizon={prediction_horizon})")
                    
                    # Ground truth
                    ground_truth = y_raw_normalized[0].squeeze().detach().cpu().numpy()
                    
                    # Get model parameters
                    warmup_period = getattr(pl_module.model, 'warmup_period', 30)
                    decimation_factor = getattr(pl_module.model, 'decimation_factor', 16)
                    
                    # Get latent representation if available
                    z_latent = None
                    if 'z' in model_outputs:
                        z_latent = model_outputs['z'][0].permute(1, 0).detach().cpu().numpy()  # (latent_dim, seq_len)
                    
                    logger.info(f"Extracted data for plotting - pred_mu: {pred_mu.shape}, ground_truth: {ground_truth.shape}")
                    
                except Exception as e:
                    logger.error(f"Error during model forward pass: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return

                # Create enhanced visualization for multi-timepoint predictions
                logger.info("Creating enhanced multi-timepoint plots...")
                fig, axes = plt.subplots(6, 1, figsize=(18, 20))
                
                # Time axes
                raw_time_axis = np.arange(len(ground_truth)) / 4.0 / 60  # Raw signal time in minutes
                
                # Plot 1: Ground truth raw signal
                axes[0].plot(raw_time_axis, ground_truth, 'b-', linewidth=1.2, label='Ground Truth FHR')
                axes[0].set_title('Ground Truth Raw FHR Signal')
                axes[0].set_ylabel('Normalized Amplitude')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Multi-timepoint predictions overview
                # Select representative timepoints for visualization
                selected_timepoints = []
                if S > warmup_period:
                    selected_timepoints = [
                        warmup_period + 10,
                        int(S * 0.25),
                        int(S * 0.5), 
                        int(S * 0.75),
                        min(S - 20, S - 1)
                    ]
                    selected_timepoints = [t for t in selected_timepoints if warmup_period <= t < S]
                
                # Plot ground truth as reference
                axes[1].plot(raw_time_axis, ground_truth, color='gray', 
                            linewidth=1.0, alpha=0.5, label='Ground Truth')
                
                # Plot predictions from selected timepoints
                prediction_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
                for i, t in enumerate(selected_timepoints[:5]):
                    raw_start = t * decimation_factor
                    raw_end = raw_start + prediction_horizon
                    
                    if raw_end <= len(ground_truth):
                        pred_time = np.arange(raw_start, raw_end) / 4.0 / 60
                        color = prediction_colors[i % len(prediction_colors)]
                        
                        axes[1].plot(pred_time, pred_mu[t], color=color, linewidth=1.2,
                                    alpha=0.8, label=f'Pred t={t} ({t*decimation_factor/4/60:.1f}min)')
                        axes[1].fill_between(pred_time, 
                                           pred_mu[t] - pred_std[t],
                                           pred_mu[t] + pred_std[t],
                                           alpha=0.15, color=color, edgecolor='none')
                
                # Mark warmup region
                warmup_time = warmup_period * decimation_factor / 4.0 / 60
                axes[1].axvline(x=warmup_time, color='orange', linestyle='--', 
                               alpha=0.7, label=f'Warmup End (t={warmup_period})')
                
                axes[1].set_title('Multi-Timepoint Future Predictions')
                axes[1].set_ylabel('Normalized Amplitude')
                axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1].grid(True, alpha=0.3)
                
                # Plot 3: Detailed view of specific timepoint
                if selected_timepoints:
                    detail_t = selected_timepoints[len(selected_timepoints)//2]
                    raw_start = detail_t * decimation_factor
                    raw_end = raw_start + prediction_horizon
                    
                    if raw_end <= len(ground_truth):
                        # Show context
                        context_samples = 200
                        context_start = max(0, raw_start - context_samples)
                        context_time = np.arange(context_start, raw_start) / 4.0 / 60
                        pred_time = np.arange(raw_start, raw_end) / 4.0 / 60
                        
                        if len(context_time) > 0:
                            axes[2].plot(context_time, ground_truth[context_start:raw_start],
                                        'b-', linewidth=1.2, alpha=0.8, label='Historical')
                        
                        axes[2].fill_between(pred_time,
                                           pred_mu[detail_t] - pred_std[detail_t],
                                           pred_mu[detail_t] + pred_std[detail_t],
                                           alpha=0.3, color='red', label='±1σ Uncertainty')
                        axes[2].plot(pred_time, pred_mu[detail_t], 'r-', linewidth=1.5, 
                                    label=f'Prediction from t={detail_t}')
                        axes[2].plot(pred_time, ground_truth[raw_start:raw_end], 'b:', 
                                    linewidth=1.0, alpha=0.7, label='Actual Future')
                        axes[2].axvline(x=raw_start / 4.0 / 60, color='red', linestyle='--', 
                                       alpha=0.7, label='Prediction Start')
                
                axes[2].set_title(f'Detailed View - Timepoint {detail_t if selected_timepoints else "N/A"}')
                axes[2].set_ylabel('Normalized Amplitude')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Plot 4: Prediction quality over time
                if S > warmup_period:
                    timepoints = range(warmup_period, min(S, len(ground_truth)//decimation_factor))
                    mse_over_time = []
                    correlation_over_time = []
                    
                    for t in timepoints:
                        raw_start = t * decimation_factor
                        raw_end = raw_start + prediction_horizon
                        
                        if raw_end <= len(ground_truth):
                            actual = ground_truth[raw_start:raw_end]
                            pred = pred_mu[t]
                            
                            mse = np.mean((actual - pred) ** 2)
                            corr = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else 0
                            
                            mse_over_time.append(mse)
                            correlation_over_time.append(corr)
                        else:
                            break
                    
                    if mse_over_time:
                        timepoint_minutes = [t * decimation_factor / 4.0 / 60 for t in timepoints[:len(mse_over_time)]]
                        
                        ax4_twin = axes[3].twinx()
                        line1 = axes[3].plot(timepoint_minutes, mse_over_time, 'r-', 
                                           linewidth=1.5, label='MSE', marker='o', markersize=3)
                        line2 = ax4_twin.plot(timepoint_minutes, correlation_over_time, 'b-',
                                            linewidth=1.5, label='Correlation', marker='s', markersize=3)
                        
                        axes[3].set_ylabel('MSE', color='red')
                        ax4_twin.set_ylabel('Correlation', color='blue')
                        axes[3].tick_params(axis='y', labelcolor='red')
                        ax4_twin.tick_params(axis='y', labelcolor='blue')
                        
                        lines = line1 + line2
                        labels = [l.get_label() for l in lines]
                        axes[3].legend(lines, labels, loc='upper right')
                
                axes[3].set_title('Prediction Quality Over Time')
                axes[3].set_xlabel('Time (minutes)')
                axes[3].grid(True, alpha=0.3)
                
                # Plot 5: Warmup period analysis
                if warmup_period < S:
                    early_timepoints = range(max(0, warmup_period-10), min(warmup_period+20, S))
                    
                    for i, t in enumerate(early_timepoints):
                        if t < len(pred_mu):
                            alpha = 0.3 if t < warmup_period else 0.8
                            color = 'orange' if t < warmup_period else 'red'
                            
                            subset_samples = min(100, prediction_horizon)
                            raw_start = t * decimation_factor
                            
                            if raw_start + subset_samples <= len(ground_truth):
                                pred_subset_time = np.arange(raw_start, raw_start + subset_samples) / 4.0 / 60
                                axes[4].plot(pred_subset_time, pred_mu[t][:subset_samples], 
                                           color=color, alpha=alpha, linewidth=1.0,
                                           label=f't={t}' if i % 5 == 0 else "")
                    
                    warmup_time = warmup_period * decimation_factor / 4.0 / 60
                    axes[4].axvline(x=warmup_time, color='red', linestyle='--', 
                                   alpha=0.8, label=f'Warmup End (t={warmup_period})')
                
                axes[4].set_title('Prediction Evolution Through Warmup Period')
                axes[4].set_ylabel('Normalized Amplitude')
                axes[4].legend()
                axes[4].grid(True, alpha=0.3)
                
                # Plot 6: Latent representation
                if z_latent is not None:
                    im = axes[5].imshow(z_latent, aspect='auto', cmap='viridis', interpolation='nearest')
                    axes[5].set_title('Latent Representation')
                    axes[5].set_xlabel('Time Steps')
                    axes[5].set_ylabel('Latent Dimensions')
                    plt.colorbar(im, ax=axes[5])
                else:
                    axes[5].text(0.5, 0.5, 'Latent representation not available', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[5].transAxes)
                    axes[5].set_title('Latent Representation (N/A)')
                
                # Calculate overall metrics (using middle timepoint as representative)
                if selected_timepoints:
                    mid_t = selected_timepoints[len(selected_timepoints)//2]
                    raw_start = mid_t * decimation_factor
                    raw_end = raw_start + prediction_horizon
                    
                    if raw_end <= len(ground_truth):
                        actual_segment = ground_truth[raw_start:raw_end]
                        pred_segment = pred_mu[mid_t]
                        
                        mse = np.mean((actual_segment - pred_segment) ** 2)
                        mae = np.mean(np.abs(actual_segment - pred_segment))
                        correlation = np.corrcoef(actual_segment, pred_segment)[0, 1]
                    else:
                        mse, mae, correlation = np.nan, np.nan, np.nan
                else:
                    mse, mae, correlation = np.nan, np.nan, np.nan
                
                # Get batch info
                try:
                    guid = batch.guid[0] if hasattr(batch, 'guid') else 'unknown'
                    epoch_info = batch.epoch[0].item() if hasattr(batch, 'epoch') else 'unknown'
                except:
                    guid = 'unknown'
                    epoch_info = 'unknown'
                
                future_minutes = prediction_horizon / 4.0 / 60.0
                plt.suptitle(f"Enhanced Multi-Timepoint Prediction Analysis\n"
                           f"GUID: {guid}, Epoch: {epoch_info}, Training Epoch: {pl_trainer.current_epoch}\n"
                           f"Prediction Window: {prediction_horizon} samples ({future_minutes:.1f} min), "
                           f"Warmup: {warmup_period} steps, Sequence: {S} steps\n"
                           f"Representative MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}")
                
                plt.tight_layout()
                plot_path = f"{self.output_dir}/enhanced_multi_timepoint_prediction_e-{pl_trainer.current_epoch}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Raw signal prediction plot saved to {plot_path}")
                
                # Explicit cleanup
                plt.close('all')
                
                # Clean up tensors to free GPU memory
                del model_outputs, raw_predictions
                if z_latent is not None:
                    del z_latent
                del y_st, y_ph, x_ph, y_raw_normalized
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