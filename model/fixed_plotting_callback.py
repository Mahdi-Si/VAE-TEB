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
                    # Extract predictions
                    pred_mu = raw_predictions['raw_signal_mu'][0].squeeze().detach().cpu().numpy()  # First sample in batch
                    pred_logvar = raw_predictions['raw_signal_logvar'][0].squeeze().detach().cpu().numpy()
                    pred_std = np.exp(0.5 * pred_logvar)
                    
                    # Ground truth
                    ground_truth = y_raw_normalized[0].squeeze().detach().cpu().numpy()
                    
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

                # Create simple visualization
                logger.info("Creating plots...")
                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                
                # Plot 1: Ground truth raw signal
                time_axis = np.arange(len(ground_truth)) / 4.0 / 60  # Convert to minutes (4Hz sampling)
                axes[0].plot(time_axis, ground_truth, 'b-', linewidth=1, label='Ground Truth')
                axes[0].set_title('Ground Truth Raw FHR Signal')
                axes[0].set_ylabel('Normalized Amplitude')
                axes[0].set_xlabel('Time (minutes)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Predicted raw signal with uncertainty
                axes[1].plot(time_axis, pred_mu, 'r-', linewidth=1, label='Predicted Mean')
                axes[1].fill_between(time_axis, pred_mu - pred_std, pred_mu + pred_std,
                                   alpha=0.3, color='red', label='±1σ Uncertainty')
                axes[1].set_title('Predicted Raw FHR Signal with Uncertainty')
                axes[1].set_ylabel('Normalized Amplitude')
                axes[1].set_xlabel('Time (minutes)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Plot 3: Comparison overlay
                axes[2].plot(time_axis, ground_truth, 'b-', linewidth=1, alpha=0.7, label='Ground Truth')
                axes[2].plot(time_axis, pred_mu, 'r--', linewidth=1, alpha=0.7, label='Predicted')
                axes[2].set_title('Ground Truth vs Predicted Raw FHR Signal')
                axes[2].set_ylabel('Normalized Amplitude')
                axes[2].set_xlabel('Time (minutes)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                # Plot 4: Latent representation (if available)
                if z_latent is not None:
                    im = axes[3].imshow(z_latent, aspect='auto', cmap='viridis', interpolation='nearest')
                    axes[3].set_title('Latent Representation')
                    axes[3].set_xlabel('Time Steps')
                    axes[3].set_ylabel('Latent Dimensions')
                    plt.colorbar(im, ax=axes[3])
                else:
                    axes[3].text(0.5, 0.5, 'Latent representation not available', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[3].transAxes)
                    axes[3].set_title('Latent Representation (N/A)')
                
                # Calculate and display metrics
                mse = np.mean((ground_truth - pred_mu) ** 2)
                mae = np.mean(np.abs(ground_truth - pred_mu))
                correlation = np.corrcoef(ground_truth, pred_mu)[0, 1] if len(ground_truth) > 1 else 0
                
                # Get batch info
                try:
                    guid = batch.guid[0] if hasattr(batch, 'guid') else 'unknown'
                    epoch_info = batch.epoch[0].item() if hasattr(batch, 'epoch') else 'unknown'
                except:
                    guid = 'unknown'
                    epoch_info = 'unknown'
                
                plt.suptitle(f"Raw Signal Prediction - GUID: {guid}, Epoch: {epoch_info}\n"
                           f"MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}")
                
                plt.tight_layout()
                plot_path = f"{self.output_dir}/raw_signal_prediction_e-{pl_trainer.current_epoch}.png"
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