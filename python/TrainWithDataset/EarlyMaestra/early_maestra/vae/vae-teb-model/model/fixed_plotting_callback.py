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
                
                # Create figure with improved layout and styling
                plt.style.use('default')
                plt.rcParams.update({
                    'figure.facecolor': '#FAFAFA',
                    'axes.facecolor': 'white',
                    'axes.edgecolor': '#CCCCCC',
                    'axes.linewidth': 0.8,
                    'grid.color': '#E8E8E8',
                    'grid.linewidth': 0.5,
                    'font.size': 10,
                    'axes.titlesize': 12,
                    'axes.labelsize': 10,
                    'legend.fontsize': 9,
                })
                
                fig, axes = plt.subplots(7, 1, figsize=(20, 24))
                
                # Time axes
                raw_time_axis = np.arange(len(ground_truth)) / 4.0 / 60  # Raw signal time in minutes
                
                # Color palette for consistent styling
                colors = {
                    'ground_truth': '#2E5984',  # Deep blue
                    'prediction': '#C7522A',   # Red-orange
                    'uncertainty': '#E5B181',  # Light peach
                    'warmup': '#F39C12',       # Orange
                    'context': '#7F8C8D'       # Gray
                }
                
                # Plot 1: Ground truth raw signal with clearer annotations
                axes[0].plot(raw_time_axis, ground_truth, color=colors['ground_truth'], 
                            linewidth=1.2, label='Ground Truth FHR', alpha=0.9)
                
                # Mark important regions
                warmup_time = warmup_period * decimation_factor / 4.0 / 60
                axes[0].axvline(x=warmup_time, color=colors['warmup'], linestyle='--', 
                               alpha=0.7, label=f'Warmup End ({warmup_period} steps)')
                
                # Add sequence region indicators
                seq_end_time = S * decimation_factor / 4.0 / 60
                axes[0].axvspan(0, warmup_time, alpha=0.1, color=colors['warmup'], label='Warmup Period')
                axes[0].axvspan(warmup_time, seq_end_time, alpha=0.1, color='green', label='Active Period')
                
                axes[0].set_title('Complete Raw FHR Signal with Analysis Regions', fontweight='bold')
                axes[0].set_ylabel('Normalized Amplitude')
                axes[0].legend(loc='upper right')
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Multi-timepoint predictions overview with improved visualization
                # Select representative timepoints for visualization (more strategic selection)
                selected_timepoints = []
                if S > warmup_period:
                    # Create more evenly distributed timepoints
                    valid_range = S - warmup_period - 5  # Leave some buffer at the end
                    if valid_range > 0:
                        n_points = min(6, valid_range // 10)  # Up to 6 points, at least 10 steps apart
                        selected_timepoints = [
                            warmup_period + i * (valid_range // n_points) 
                            for i in range(n_points)
                        ]
                        # Ensure we don't exceed bounds
                        selected_timepoints = [t for t in selected_timepoints if warmup_period <= t < S-2]
                
                # Plot ground truth as reference with better styling
                axes[1].plot(raw_time_axis, ground_truth, color=colors['context'], 
                            linewidth=1.0, alpha=0.6, label='Ground Truth', zorder=1)
                
                # Enhanced prediction colors with better contrast
                prediction_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#E67E22']
                
                # Plot predictions from selected timepoints with improved visualization
                for i, t in enumerate(selected_timepoints):
                    raw_start = t * decimation_factor
                    raw_end = raw_start + prediction_horizon
                    
                    if raw_end <= len(ground_truth):
                        pred_time = np.arange(raw_start, raw_end) / 4.0 / 60
                        color = prediction_colors[i % len(prediction_colors)]
                        
                        # Plot uncertainty band first (lower z-order)
                        axes[1].fill_between(pred_time, 
                                           pred_mu[t] - pred_std[t],
                                           pred_mu[t] + pred_std[t],
                                           alpha=0.2, color=color, edgecolor='none', zorder=2)
                        
                        # Plot prediction line with marker at start
                        axes[1].plot(pred_time, pred_mu[t], color=color, linewidth=1.5,
                                    alpha=0.9, label=f'Pred t={t} (start: {t*decimation_factor/4/60:.1f}min)', 
                                    zorder=3)
                        
                        # Mark prediction start point
                        axes[1].scatter([pred_time[0]], [pred_mu[t][0]], color=color, s=40, 
                                      zorder=4, edgecolors='white', linewidth=1)
                
                # Mark important regions
                axes[1].axvline(x=warmup_time, color=colors['warmup'], linestyle='--', 
                               alpha=0.8, linewidth=2, label=f'Warmup End (t={warmup_period})', zorder=5)
                
                # Add prediction horizon indicator
                if selected_timepoints:
                    sample_t = selected_timepoints[0]
                    sample_start = sample_t * decimation_factor / 4.0 / 60
                    sample_end = (sample_t * decimation_factor + prediction_horizon) / 4.0 / 60
                    axes[1].annotate('', xy=(sample_end, 0.95*np.max(ground_truth)), 
                                   xytext=(sample_start, 0.95*np.max(ground_truth)),
                                   arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                                   annotation_clip=False)
                    axes[1].text((sample_start + sample_end)/2, 0.97*np.max(ground_truth), 
                               f'{prediction_horizon/4/60:.1f} min prediction window',
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                axes[1].set_title('Multi-Timepoint Future Predictions (2-Minute Windows)', fontweight='bold')
                axes[1].set_ylabel('Normalized Amplitude')
                axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                axes[1].grid(True, alpha=0.3)
                
                # Plot 2.5: NEW - Sequential predictions at different timepoints
                if selected_timepoints:
                    # Show how predictions evolve temporally
                    prediction_start_times = []
                    prediction_mse_values = []
                    prediction_correlations = []
                    
                    for t in selected_timepoints:
                        raw_start = t * decimation_factor
                        raw_end = raw_start + prediction_horizon
                        
                        if raw_end <= len(ground_truth):
                            start_time = raw_start / 4.0 / 60
                            prediction_start_times.append(start_time)
                            
                            # Calculate quality metrics
                            actual = ground_truth[raw_start:raw_end]
                            pred = pred_mu[t]
                            
                            mse = np.mean((actual - pred) ** 2)
                            corr = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else 0
                            
                            prediction_mse_values.append(mse)
                            prediction_correlations.append(corr)
                    
                    if prediction_start_times:
                        # Plot temporal evolution of prediction quality
                        ax2_twin = axes[2].twinx()
                        
                        line1 = axes[2].plot(prediction_start_times, prediction_mse_values, 
                                           'o-', color='red', linewidth=2, markersize=6,
                                           label='MSE', markerfacecolor='white', markeredgewidth=2)
                        line2 = ax2_twin.plot(prediction_start_times, prediction_correlations, 
                                            's-', color='blue', linewidth=2, markersize=6,
                                            label='Correlation', markerfacecolor='white', markeredgewidth=2)
                        
                        # Styling
                        axes[2].set_ylabel('Mean Squared Error', color='red', fontweight='bold')
                        ax2_twin.set_ylabel('Correlation Coefficient', color='blue', fontweight='bold')
                        axes[2].tick_params(axis='y', labelcolor='red')
                        ax2_twin.tick_params(axis='y', labelcolor='blue')
                        axes[2].set_xlabel('Prediction Start Time (minutes)')
                        
                        # Add warmup indicator
                        axes[2].axvline(x=warmup_time, color=colors['warmup'], linestyle='--', 
                                       alpha=0.8, linewidth=2)
                        
                        # Combined legend
                        lines = line1 + line2
                        labels = [l.get_label() for l in lines]
                        axes[2].legend(lines, labels, loc='upper left')
                        
                        axes[2].set_title('Prediction Quality vs. Temporal Position', fontweight='bold')
                        axes[2].grid(True, alpha=0.3)
                
                # Plot 3: Enhanced detailed view of specific timepoint  
                if selected_timepoints:
                    detail_t = selected_timepoints[len(selected_timepoints)//2]
                    raw_start = detail_t * decimation_factor
                    raw_end = raw_start + prediction_horizon
                    
                    if raw_end <= len(ground_truth):
                        # Show expanded context with clearer transitions
                        context_samples = 400  # Increased context
                        context_start = max(0, raw_start - context_samples)
                        context_time = np.arange(context_start, raw_start) / 4.0 / 60
                        pred_time = np.arange(raw_start, raw_end) / 4.0 / 60
                        
                        # Plot historical context with gradient effect
                        if len(context_time) > 0:
                            axes[3].plot(context_time, ground_truth[context_start:raw_start],
                                        color=colors['ground_truth'], linewidth=1.5, alpha=0.8, 
                                        label='Historical Context')
                        
                        # Plot actual future for comparison
                        axes[3].plot(pred_time, ground_truth[raw_start:raw_end], 
                                    color=colors['ground_truth'], linewidth=1.5, linestyle=':', 
                                    alpha=0.9, label='Actual Future', zorder=4)
                        
                        # Plot uncertainty band
                        axes[3].fill_between(pred_time,
                                           pred_mu[detail_t] - pred_std[detail_t],
                                           pred_mu[detail_t] + pred_std[detail_t],
                                           alpha=0.25, color=colors['prediction'], 
                                           label='±1σ Uncertainty', zorder=2)
                        
                        # Plot prediction
                        axes[3].plot(pred_time, pred_mu[detail_t], 
                                    color=colors['prediction'], linewidth=2.0, 
                                    label=f'Prediction from t={detail_t}', zorder=3)
                        
                        # Mark prediction boundary clearly
                        axes[3].axvline(x=raw_start / 4.0 / 60, color='black', linestyle='-', 
                                       alpha=0.8, linewidth=2, label='Prediction Boundary', zorder=5)
                        
                        # Add transition region highlighting
                        transition_width = 20  # samples
                        transition_start = max(context_start, raw_start - transition_width)
                        axes[3].axvspan((transition_start) / 4.0 / 60, raw_start / 4.0 / 60, 
                                       alpha=0.1, color='yellow', label='Transition Zone')
                        
                        # Calculate and display metrics for this prediction
                        actual_segment = ground_truth[raw_start:raw_end]
                        pred_segment = pred_mu[detail_t]
                        mse_detail = np.mean((actual_segment - pred_segment) ** 2)
                        corr_detail = np.corrcoef(actual_segment, pred_segment)[0, 1]
                        
                        # Add text box with metrics
                        textstr = f'MSE: {mse_detail:.4f}\nCorr: {corr_detail:.3f}\nTime: {detail_t*decimation_factor/4/60:.1f} min'
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                        axes[3].text(0.02, 0.98, textstr, transform=axes[3].transAxes, fontsize=9,
                                    verticalalignment='top', bbox=props)
                
                axes[3].set_title(f'Detailed Analysis - Prediction from Timepoint {detail_t if selected_timepoints else "N/A"}', 
                                 fontweight='bold')
                axes[3].set_ylabel('Normalized Amplitude')
                axes[3].legend(loc='lower right', fontsize=8)
                axes[3].grid(True, alpha=0.3)
                
                # Plot 4: NEW - Stacked predictions showing temporal progression
                if selected_timepoints and len(selected_timepoints) >= 3:
                    # Create a stacked view of predictions at different time points
                    n_show = min(4, len(selected_timepoints))  # Show up to 4 predictions
                    stack_timepoints = selected_timepoints[:n_show]
                    
                    # Create offset for stacking
                    y_offset = 0
                    stack_spacing = 1.5 * np.std(ground_truth)
                    
                    for i, t in enumerate(stack_timepoints):
                        raw_start = t * decimation_factor
                        raw_end = raw_start + prediction_horizon
                        
                        if raw_end <= len(ground_truth):
                            pred_time = np.arange(raw_start, raw_end) / 4.0 / 60
                            color = prediction_colors[i % len(prediction_colors)]
                            
                            # Offset the signals for stacking
                            offset_pred = pred_mu[t] + y_offset
                            offset_actual = ground_truth[raw_start:raw_end] + y_offset
                            offset_std = pred_std[t]
                            
                            # Plot uncertainty band
                            axes[4].fill_between(pred_time, 
                                               offset_pred - offset_std,
                                               offset_pred + offset_std,
                                               alpha=0.2, color=color)
                            
                            # Plot prediction and actual
                            axes[4].plot(pred_time, offset_pred, color=color, linewidth=1.5, 
                                        label=f't={t} Pred ({raw_start/4/60:.1f}min)')
                            axes[4].plot(pred_time, offset_actual, color=color, linewidth=1.0, 
                                        linestyle='--', alpha=0.7, 
                                        label=f't={t} Actual')
                            
                            # Add time point marker
                            axes[4].text(pred_time[0]-0.2, y_offset, f't={t}', 
                                        rotation=90, va='center', ha='right', fontsize=8,
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))
                            
                            y_offset += stack_spacing
                    
                    axes[4].set_title('Stacked View: Predictions at Different Timepoints', fontweight='bold')
                    axes[4].set_ylabel('Normalized Amplitude (Stacked)')
                    axes[4].set_xlabel('Time (minutes)')
                    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
                    axes[4].grid(True, alpha=0.3)
                else:
                    # Fallback message if not enough timepoints
                    axes[4].text(0.5, 0.5, 'Insufficient timepoints for stacked view\n(Need at least 3 valid predictions)', 
                               ha='center', va='center', transform=axes[4].transAxes, fontsize=10)
                    axes[4].set_title('Stacked View: Not Available', fontweight='bold')
                
                # Plot 5: Comprehensive prediction quality analysis over time
                if S > warmup_period:
                    timepoints = range(warmup_period, min(S, len(ground_truth)//decimation_factor))
                    mse_over_time = []
                    correlation_over_time = []
                    mae_over_time = []
                    
                    for t in timepoints:
                        raw_start = t * decimation_factor
                        raw_end = raw_start + prediction_horizon
                        
                        if raw_end <= len(ground_truth):
                            actual = ground_truth[raw_start:raw_end]
                            pred = pred_mu[t]
                            
                            mse = np.mean((actual - pred) ** 2)
                            mae = np.mean(np.abs(actual - pred))
                            corr = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else 0
                            
                            mse_over_time.append(mse)
                            mae_over_time.append(mae)
                            correlation_over_time.append(corr)
                        else:
                            break
                    
                    if mse_over_time:
                        timepoint_minutes = [t * decimation_factor / 4.0 / 60 for t in timepoints[:len(mse_over_time)]]
                        
                        # Create twin axes for different metrics
                        ax5_twin = axes[5].twinx()
                        
                        # Plot MSE and MAE on primary axis
                        line1 = axes[5].plot(timepoint_minutes, mse_over_time, 'r-', 
                                           linewidth=2, label='MSE', marker='o', markersize=4,
                                           markerfacecolor='white', markeredgewidth=1.5)
                        line3 = axes[5].plot(timepoint_minutes, mae_over_time, 'orange', 
                                           linewidth=2, label='MAE', marker='^', markersize=4,
                                           markerfacecolor='white', markeredgewidth=1.5)
                        
                        # Plot correlation on secondary axis
                        line2 = ax5_twin.plot(timepoint_minutes, correlation_over_time, 'b-',
                                            linewidth=2, label='Correlation', marker='s', markersize=4,
                                            markerfacecolor='white', markeredgewidth=1.5)
                        
                        # Styling
                        axes[5].set_ylabel('MSE / MAE', color='red', fontweight='bold')
                        ax5_twin.set_ylabel('Correlation Coefficient', color='blue', fontweight='bold')
                        axes[5].tick_params(axis='y', labelcolor='red')
                        ax5_twin.tick_params(axis='y', labelcolor='blue')
                        axes[5].set_xlabel('Prediction Start Time (minutes)')
                        
                        # Add warmup indicator
                        axes[5].axvline(x=warmup_time, color=colors['warmup'], linestyle='--', 
                                       alpha=0.8, linewidth=2, label='Warmup End')
                        
                        # Combined legend
                        lines = line1 + line2 + line3
                        labels = [l.get_label() for l in lines]
                        axes[5].legend(lines, labels, loc='upper left', fontsize=9)
                        
                        # Add statistics text
                        avg_mse = np.mean(mse_over_time)
                        avg_corr = np.mean(correlation_over_time)
                        stats_text = f'Avg MSE: {avg_mse:.4f}\nAvg Corr: {avg_corr:.3f}'
                        axes[5].text(0.98, 0.98, stats_text, transform=axes[5].transAxes, 
                                    fontsize=9, va='top', ha='right',
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                axes[5].set_title('Comprehensive Prediction Quality Over Time', fontweight='bold')
                axes[5].grid(True, alpha=0.3)
                
                # Plot 6: Enhanced latent representation with analysis
                if z_latent is not None:
                    # Create latent visualization with enhanced features
                    im = axes[6].imshow(z_latent, aspect='auto', cmap='viridis', interpolation='nearest')
                    
                    # Add warmup period indicator
                    if warmup_period < z_latent.shape[1]:
                        axes[6].axvline(x=warmup_period, color='red', linestyle='--', 
                                       alpha=0.8, linewidth=2, label=f'Warmup End (t={warmup_period})')
                    
                    # Add colorbar with better positioning
                    cbar = plt.colorbar(im, ax=axes[6], shrink=0.8)
                    cbar.set_label('Latent Activation', rotation=270, labelpad=15)
                    
                    # Add latent statistics
                    latent_mean = np.mean(z_latent)
                    latent_std = np.std(z_latent)
                    latent_range = np.max(z_latent) - np.min(z_latent)
                    
                    stats_text = f'Mean: {latent_mean:.3f}\nStd: {latent_std:.3f}\nRange: {latent_range:.3f}'
                    axes[6].text(0.02, 0.98, stats_text, transform=axes[6].transAxes, 
                                fontsize=9, va='top', ha='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    axes[6].set_title('Latent Space Representation', fontweight='bold')
                    axes[6].set_xlabel('Time Steps')
                    axes[6].set_ylabel('Latent Dimensions')
                    
                    # Add dimension labels if not too many
                    if z_latent.shape[0] <= 20:
                        axes[6].set_yticks(range(0, z_latent.shape[0], max(1, z_latent.shape[0]//10)))
                    
                    # Add time step labels
                    n_ticks = min(10, z_latent.shape[1])
                    tick_positions = np.linspace(0, z_latent.shape[1]-1, n_ticks, dtype=int)
                    tick_labels = [f't={pos}' for pos in tick_positions]
                    axes[6].set_xticks(tick_positions)
                    axes[6].set_xticklabels(tick_labels, rotation=45)
                    
                else:
                    axes[6].text(0.5, 0.5, 'Latent representation not available\n(Check model output keys)', 
                               ha='center', va='center', transform=axes[6].transAxes, fontsize=12)
                    axes[6].set_title('Latent Representation (N/A)', fontweight='bold')
                
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
                
                # Enhanced title with comprehensive information
                future_minutes = prediction_horizon / 4.0 / 60.0
                sequence_minutes = S * decimation_factor / 4.0 / 60.0
                warmup_minutes = warmup_period * decimation_factor / 4.0 / 60.0
                
                plt.suptitle(f"Enhanced TEB-VAE Multi-Timepoint Prediction Analysis\n"
                           f"Sample: {guid} (Epoch {epoch_info}) | Training Epoch: {pl_trainer.current_epoch}\n"
                           f"Signal: {len(ground_truth)} samples ({len(ground_truth)/4/60:.1f}min) | "
                           f"Sequence: {S} steps ({sequence_minutes:.1f}min) | "
                           f"Warmup: {warmup_period} steps ({warmup_minutes:.1f}min)\n"
                           f"Prediction Window: {prediction_horizon} samples ({future_minutes:.1f}min) | "
                           f"Decimation: {decimation_factor}x | "
                           f"Timepoints Analyzed: {len(selected_timepoints) if selected_timepoints else 0}\n"
                           f"Representative Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}",
                           fontsize=11, y=0.98)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
                plot_path = f"{self.output_dir}/enhanced_multi_timepoint_prediction_e-{pl_trainer.current_epoch}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
                logger.info(f"Enhanced raw signal prediction plot saved to {plot_path}")
                
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