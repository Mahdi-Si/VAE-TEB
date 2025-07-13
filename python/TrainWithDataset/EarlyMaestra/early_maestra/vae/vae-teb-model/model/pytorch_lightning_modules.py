import lightning as L
import torch.nn.functional as F
# from keras.src.backend.jax.random import gamma
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks import TQDMProgressBar
from torchmetrics.classification import BinaryConfusionMatrix, BinaryPrecisionRecallCurve, BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryROC, BinaryAveragePrecision
import numpy as np
import torch.nn as nn

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import torch
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib
import os
from utils.data_utils import plot_forward_pass


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from vae_teb_model import SeqVaeTeb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"


matplotlib.use('Agg')
torch.backends.cudnn.enabled = False

from loguru import logger

def compute_sample_weights(targets, weight_map):
    """
    Given a tensor of targets (0 or 1) and a mapping (dict) from class to weight,
    returns a tensor of weights of the same shape.
    """
    # Make sure that targets is a tensor
    # Use a list comprehension to create a tensor of weights corresponding to the target values.
    weights = torch.tensor([weight_map[int(t.item())] for t in targets], dtype=torch.float32, device=targets.device)
    return weights


def remove_invalid_samples(true_labels, logits, mask):
    """
    Removes samples from true labels and logits where class 0 (invalid) is present in true labels.

    Args:
        true_labels (torch.Tensor): Tensor of one-hot encoded true labels with shape (batch_size, time_steps, num_classes).
                                    Class 0 is considered invalid.
        logits (torch.Tensor): Tensor of logits with shape (batch_size, time_steps, num_logit_classes).

    Returns:
        tuple: A tuple containing:
            - filtered_true_labels (torch.Tensor): True labels with invalid samples removed.
            - filtered_logits (torch.Tensor): Logits with corresponding invalid samples removed.
    """
    # 1. Identify invalid samples based on true_labels.
    #    A sample is invalid if ANY timestep has class 0 as the true label.

    # Check if class 0 is the maximum value (one-hot encoded) along the last dimension (classes)
    # and if this occurs for ANY timestep (dimension 1) in each batch (dimension 0).
    # mask[2, 2] = 0
    true_labels  = torch.argmax(true_labels, dim=-1) * mask
    valid_mask = (true_labels >= 0)

    # 3. Filter both true_labels and logits using the valid_mask.
    filtered_true_labels = true_labels[valid_mask]
    filtered_logits = logits[valid_mask]

    return filtered_true_labels, filtered_logits



class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.

    For each class j, a margin delta_j is computed as:
        delta_j = max_m * (1 / sqrt(sqrt(n_j))) / max(1 / sqrt(sqrt(n_j)))
    where n_j is the number of samples in class j.

    The loss subtracts the margin from the logit corresponding to the true class
    and then computes a scaled cross-entropy loss.

    Parameters
    ----------
    cls_num_list : list or array-like
        A list of sample counts for each class (e.g., [num_class0, num_class1]).
    max_m : float, optional
        The maximum margin value (default is 0.5).
    s : float, optional
        The scaling factor applied to logits (default is 30).
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        # Compute margin for each class.
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s

    def forward(self, logits, target):
        """
        Parameters
        ----------
        logits : torch.Tensor of shape (N, C)
            The model predictions (before softmax).
        target : torch.Tensor of shape (N,)
            The ground truth labels (0, 1, ..., C-1).
        """
        if logits.is_cuda:
            self.m_list = self.m_list.cuda()
        # Create a binary mask that selects the ground-truth classes.
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.unsqueeze(1), True)
        index_float = index.float()
        # Compute per-sample margins by multiplying the mask with the margins.
        batch_m = torch.matmul(index_float, self.m_list.unsqueeze(1)).squeeze(1)
        # Subtract the margin from the logit corresponding to the true class.
        logits_m = logits.clone()
        logits_m[torch.arange(logits.size(0)), target] -= batch_m
        # Scale the logits and compute standard cross-entropy.
        return F.cross_entropy(self.s * logits_m, target)

# ------------------------------------------------------------------------------------------------------------------------------------------
# Modules for models
# ------------------------------------------------------------------------------------------------------------------------------------------
class LightSeqVAE(L.LightningModule):
    def __init__(self, seqvae_model, kl_beta, lr, lr_milestones, batch_size_train, batch_size_test):
        super().__init__()
        self.seqvae_model = seqvae_model
        self.lr_milestones = lr_milestones
        self.lr = lr
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.validation_min = 1e+8
        self.kl_beta = kl_beta

    # todo: handel this zero_source_value
    def forward(self, x, epoch_num):
        return self.seqvae_model(x, zero_source=False, epoch_num=epoch_num)

    def training_step(self, batch=None, batch_idx=None):
        data = batch[0]
        mask = batch[4]
        epoch_num = batch[3]
        results = self.seqvae_model(data, zero_source=False, epoch_num=epoch_num)
        loss = (self.kl_beta * results.kld_loss) + results.nll_loss
        self.log('train_loss', loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train_kld_loss', results.kld_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train_reconstruction_loss', results.nll_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch=None, batch_idx=None):
        data = batch[0]
        mask = batch[4]
        epoch_num = batch[3]
        results = self.seqvae_model(data, epoch_num=epoch_num)
        loss = (self.kl_beta * results.kld_loss) + results.nll_loss
        self.log('validation_loss', loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True,  sync_dist=True)
        logger.info(f'validation loss: {loss.item()}')
        self.log('validation_kld_loss', results.kld_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('validation_reconstruction_loss', results.nll_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        # lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones,
                                                         gamma=0.5)
        # return optimizer
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': schedular,
                'interval': 'epoch',
                'frequency': 1
            }
        }


class LightSeqVAEClassifier(L.LightningModule):
    def __init__(self, seqvae_model, kl_beta, lr, lr_milestones, batch_size_train, batch_size_test,
                 use_ldam=False, cls_num_list=None):
        super().__init__()
        self.seqvae_model = seqvae_model
        self.lr_milestones = lr_milestones
        self.lr = lr
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.validation_min = 1e+8
        self.kl_beta = kl_beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.cls_num_list = cls_num_list

        self.train_accuracy = BinaryAccuracy()
        self.train_recall = BinaryRecall()
        self.train_confmat = BinaryConfusionMatrix()

        self.val_accuracy = BinaryAccuracy()
        self.val_recall = BinaryRecall()
        self.val_confmat = BinaryConfusionMatrix()
        self.use_ldam = use_ldam
        # todo: Connect this to graph
        self.per_step_label = True
        self.sample_weights = None
        self.labeling_start_idx = 100
        if self.use_ldam:
            # self.ldam_loss = LDAMLoss(cls_num_list)
            total_samples = self.cls_num_list[0] + self.cls_num_list[1]
            num_classes = 2
            weight_for_positive = total_samples / (num_classes * self.cls_num_list[0])
            weight_for_negative = total_samples / (num_classes * self.cls_num_list[1])
            self.class_weights = {0: weight_for_negative, 1: weight_for_positive}
            self.ldam_loss = nn.BCEWithLogitsLoss(reduce='None')



    def forward(self, x, epoch_num):
        return self.seqvae_model(x, zero_source=False, epoch_num=epoch_num)

    def training_step(self, batch=None, batch_idx=None):
        data = batch[0]
        labels = batch[1]
        weights_mask = batch[4]  # shape (batch_size,
        epoch_num = batch[3]

        results = self.seqvae_model(data, zero_source=False, epoch_num=epoch_num)
        if self.per_step_label:
            # Slice according to your original code
            weights_mask = weights_mask[:, ::16]
            weights_mask = weights_mask[:, self.labeling_start_idx:]
            labels = labels[:, self.labeling_start_idx:, :]
            logits = results.logits[:, self.labeling_start_idx:, :]

            # Create a boolean mask from weights
            valid_mask = weights_mask.to(torch.bool)
            # Create a mask where the true label is not the invalid [1, 0, 0]
            label_valid_mask = (labels.argmax(dim=-1) != 0)
            # Combine both masks
            combined_mask = valid_mask & label_valid_mask

            # Convert one-hot labels to class indices
            true_labels = labels.argmax(dim=-1)  # This yields 0, 1, or 2 for each position.
            # For valid labels, subtract 1 so that the classes become 0 or 1.
            adjusted_true_labels = true_labels - 1

            # Apply the combined mask to filter logits and labels.
            # After masking, these tensors are flattened.
            filtered_logits = logits[combined_mask]  # Shape: (N, 2)
            filtered_labels = adjusted_true_labels[combined_mask]  # Shape: (N,)

            # Compute the cross entropy loss using the filtered tensors.
            if self.use_ldam:
                loss = self.ldam_loss(filtered_logits, filtered_labels)
                loss = loss * self.sample_weights #+ self.kl_beta * results.kld_loss
                loss = loss.mean()
            else:
                loss = self.ce_loss(filtered_logits, filtered_labels) # + self.kl_beta * results.kld_loss

            predicted_label = filtered_logits.argmax(dim=-1).to(torch.long)
            self.train_accuracy(predicted_label, filtered_labels)
            self.train_recall(predicted_label, filtered_labels)
            self.train_confmat(predicted_label, filtered_labels)
        else:
            true_labels = labels.argmax(dim=-1)
            true_labels, _ = torch.max(true_labels, dim=1)
            true_labels =  true_labels - 1
            true_labels_ = true_labels
            self.sample_weights = compute_sample_weights(true_labels, self.class_weights)
            true_labels = F.one_hot(true_labels, num_classes=2).float()
            if self.ldam_loss:
                loss = self.ldam_loss(results.logits, true_labels.squeeze(-1)) #+ self.kl_beta * results.kld_loss
                loss = loss * self.sample_weights
                loss = loss.mean()
            else:
                loss = self.ce_loss(results.logits, true_labels.squeeze(-1)) # + self.kl_beta * results.kld_loss
            predicted_label = results.logits.argmax(dim=-1).to(torch.long)
            self.train_accuracy(predicted_label, true_labels_)
            self.train_recall(predicted_label, true_labels_)
            self.train_confmat(predicted_label, true_labels_)

        loss_vae = (self.kl_beta * results.kld_loss) + results.nll_loss
        self.log('train_loss', loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_kld_loss', results.kld_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_reconstruction_loss', results.nll_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True)
        # for i, per_coefficient_loss in enumerate(results.nll_per_coefficient_loss):
        #     self.log(f'train_loss_/st-{i}', per_coefficient_loss.item(),
        #              on_epoch=True, on_step=False, prog_bar=False)
        return loss

    def validation_step(self, batch=None, batch_idx=None):
        data = batch[0]
        labels = batch[1]
        weights_mask = batch[4]  # shape (batch_size,
        epoch_num = batch[3]

        results = self.seqvae_model(data, zero_source=False, epoch_num=epoch_num)
        if self.per_step_label:
            # Slice according to your original code
            weights_mask = weights_mask[:, ::16]
            weights_mask = weights_mask[:, self.labeling_start_idx:]
            labels = labels[:, self.labeling_start_idx:, :]
            logits = results.logits[:, self.labeling_start_idx:, :]

            # Create a boolean mask from weights
            valid_mask = weights_mask.to(torch.bool)
            # Create a mask where the true label is not the invalid [1, 0, 0]
            label_valid_mask = (labels.argmax(dim=-1) != 0)
            # Combine both masks
            combined_mask = valid_mask & label_valid_mask

            # Convert one-hot labels to class indices
            true_labels = labels.argmax(dim=-1)  # This yields 0, 1, or 2 for each position.
            # For valid labels, subtract 1 so that the classes become 0 or 1.
            adjusted_true_labels = true_labels - 1

            # Apply the combined mask to filter logits and labels.
            # After masking, these tensors are flattened.
            filtered_logits = logits[combined_mask]  # Shape: (N, 2)
            filtered_labels = adjusted_true_labels[combined_mask]  # Shape: (N,)

            # Compute the cross entropy loss using the filtered tensors.
            if self.use_ldam:
                loss = self.ldam_loss(filtered_logits, filtered_labels) #+ self.kl_beta * results.kld_loss
                loss = loss * self.sample_weights
                loss = loss.mean()
            else:
                loss = self.ce_loss(filtered_logits, filtered_labels) #+ self.kl_beta * results.kld_loss
            predicted_label = filtered_logits.argmax(dim=-1).to(torch.long)
            probs = torch.softmax(filtered_logits, dim=-1)
            self.val_accuracy(predicted_label, filtered_labels)
            self.val_recall(predicted_label, filtered_labels)
            self.val_confmat(predicted_label, filtered_labels)
        else:
            true_labels = labels.argmax(dim=-1)
            true_labels, _ = torch.max(true_labels, dim=1)
            true_labels = true_labels - 1
            filtered_labels = true_labels
            true_labels = F.one_hot(true_labels, num_classes=2).float()
            if self.use_ldam:
                loss = self.ldam_loss(results.logits, true_labels.squeeze(-1)) #+ self.kl_beta * results.kld_loss
                loss = loss * self.sample_weights
                loss = loss.mean()
            else:
                loss = self.ce_loss(results.logits, true_labels.squeeze(-1)) #+ self.kl_beta * results.kld_loss
            predicted_label = results.logits.argmax(dim=-1).to(torch.long)
            probs = torch.softmax(results.logits, dim=-1)
            self.val_accuracy(predicted_label, filtered_labels)
            self.train_recall(predicted_label, filtered_labels)
            self.val_confmat(predicted_label, filtered_labels)
        loss_vae = (self.kl_beta * results.kld_loss) + results.nll_loss

        self.log('validation_loss', loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True)
        self.log('validation_kld_loss', results.kld_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True)
        self.log('validation_reconstruction_loss', results.nll_loss.item(),
                 on_epoch=True, on_step=False, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss, 'preds': predicted_label, 'labels': filtered_labels, 'probs': probs}
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.lr_milestones)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': schedular,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        confmat = self.train_confmat.compute()
        tn, fp, fn, tp = confmat.flatten()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)

        self.log('train_binary_accuracy', self.train_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log('train_binary_sensitivity', self.train_recall.compute(), on_epoch=True, prog_bar=True)
        self.log('train_binary_specificity', specificity, on_epoch=True, prog_bar=True)

        # Reset training metrics
        self.train_accuracy.reset()
        self.train_recall.reset()
        self.train_confmat.reset()


    def on_validation_epoch_end(self):
        # Compute specificity
        confmat = self.val_confmat.compute()
        tn, fp, fn, tp = confmat.flatten()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)

        # Log validation metrics
        self.log('val_binary_accuracy', self.val_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log('val_binary_sensitivity', self.val_recall.compute(), on_epoch=True, prog_bar=True)
        self.log('val_binary_specificity', specificity, on_epoch=True, prog_bar=True)

        # Reset validation metrics
        self.val_accuracy.reset()
        self.val_recall.reset()
        self.val_confmat.reset()

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

        # Ensure batch is on the correct device
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)

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
                
                logger.info(f"Batch shapes - y_st: {y_st.shape}, y_ph: {y_ph.shape}, x_ph: {x_ph.shape}, y_raw: {y_raw_normalized.shape}")

                # Implement windowed prediction logic
                logger.info("Running windowed prediction...")
                
                # Parameters for windowed prediction
                sampling_rate_hz = 4
                decimation_factor = 16
                warmup_minutes = 2
                prediction_window_minutes = 2
                
                # Calculate indices
                warmup_steps_raw = int(warmup_minutes * 60 * sampling_rate_hz)  # t=2*60*4 = 480
                warmup_steps_decimated = warmup_steps_raw // decimation_factor  # t=2*60*4/16 = 30
                prediction_window_steps_raw = int(prediction_window_minutes * 60 * sampling_rate_hz)  # 480 steps
                prediction_window_steps_decimated = prediction_window_steps_raw // decimation_factor  # 30 steps
                
                total_decimated_len = y_st.shape[2]
                total_raw_len = y_raw_normalized.shape[1]
                
                logger.info(f"Windowed prediction setup: warmup_raw={warmup_steps_raw}, warmup_decimated={warmup_steps_decimated}")
                logger.info(f"Prediction window: raw={prediction_window_steps_raw}, decimated={prediction_window_steps_decimated}")
                
                # Calculate number of prediction windows
                num_predictions = max(1, (total_decimated_len - warmup_steps_decimated) // prediction_window_steps_decimated)
                logger.info(f"Number of prediction windows: {num_predictions}")
                
                predicted_segments_mu = []
                predicted_segments_std = []
                prediction_start_times_minutes = []
                prediction_start_times_decimated = []
                model_outputs = None  # To store last output
                
                for i in range(num_predictions):
                    # Current prediction window start in decimated coordinates
                    current_input_end_decimated = warmup_steps_decimated + i * prediction_window_steps_decimated
                    
                    # Prepare windowed inputs (truncate to current time)
                    y_st_window = y_st[:, :, :current_input_end_decimated]
                    y_ph_window = y_ph[:, :, :current_input_end_decimated]
                    x_ph_window = x_ph[:, :, :current_input_end_decimated]
                    
                    logger.info(f"Window {i+1}: input ends at decimated step {current_input_end_decimated}")
                    
                    # Run model on the window
                    model_outputs = pl_module.model(y_st_window, y_ph_window, x_ph_window)
                    
                    if 'raw_predictions' not in model_outputs:
                        logger.error(f"Window {i+1}: Model output missing 'raw_predictions' key")
                        continue
                    
                    raw_predictions = model_outputs['raw_predictions']
                    if 'raw_signal_mu' not in raw_predictions or 'raw_signal_logvar' not in raw_predictions:
                        logger.error(f"Window {i+1}: Raw predictions missing required keys")
                        continue
                    
                    pred_mu_full = raw_predictions['raw_signal_mu'][0].squeeze()
                    pred_logvar_full = raw_predictions['raw_signal_logvar'][0].squeeze()
                    
                    # Extract the prediction for the next 2-minute window in raw coordinates
                    prediction_start_raw = current_input_end_decimated * decimation_factor
                    prediction_end_raw = prediction_start_raw + prediction_window_steps_raw
                    
                    # Ensure we don't exceed the predicted signal length
                    if prediction_end_raw <= pred_mu_full.size(0):
                        segment_mu = pred_mu_full[prediction_start_raw:prediction_end_raw].detach().cpu().numpy()
                        segment_logvar = pred_logvar_full[prediction_start_raw:prediction_end_raw].detach().cpu().numpy()
                        segment_std = np.exp(0.5 * segment_logvar)
                        
                        predicted_segments_mu.append(segment_mu)
                        predicted_segments_std.append(segment_std)
                        
                        # Store timing information
                        start_time_minutes = (current_input_end_decimated * decimation_factor) / sampling_rate_hz / 60.0
                        prediction_start_times_minutes.append(start_time_minutes)
                        prediction_start_times_decimated.append(current_input_end_decimated)
                        
                        logger.info(f"Window {i+1}: extracted segment of {len(segment_mu)} samples starting at {start_time_minutes:.2f} min")
                
                # Check if we have any predictions
                if not predicted_segments_mu:
                    logger.warning("No prediction segments were generated. Using full forward pass.")
                    # Fallback to full forward pass
                    model_outputs = pl_module.model(y_st, y_ph, x_ph)
                    raw_predictions = model_outputs['raw_predictions']
                    pred_mu = raw_predictions['raw_signal_mu'][0].squeeze().detach().cpu().numpy()
                    pred_logvar = raw_predictions['raw_signal_logvar'][0].squeeze().detach().cpu().numpy()
                    pred_std = np.exp(0.5 * pred_logvar)
                    prediction_start_times_minutes = []
                else:
                    # Stitch predictions together
                    pred_mu = np.concatenate(predicted_segments_mu)
                    pred_std = np.concatenate(predicted_segments_std)
                    logger.info(f"Stitched predictions: {len(pred_mu)} samples from {len(predicted_segments_mu)} windows")
                
                # Ground truth (first sample in batch)
                ground_truth = y_raw_normalized[0].squeeze().detach().cpu().numpy()
                
                # Get latent representation from the final prediction window
                z_latent = None
                if model_outputs and 'z' in model_outputs:
                    z_latent = model_outputs['z'][0].permute(1, 0).detach().cpu().numpy()  # (latent_dim, seq_len)
                
                logger.info(f"Data shapes for plotting - pred_mu: {pred_mu.shape}, ground_truth: {ground_truth.shape}")
                if z_latent is not None:
                    logger.info(f"Latent shape: {z_latent.shape}")

                # --- Create plots with proper data handling and professional styling ---
                # Professional pastel color palette for research papers
                colors = {
                    'ground_truth': '#2E5984',      # Deep blue-gray (professional)
                    'prediction': '#C7522A',       # Muted red-orange  
                    'uncertainty': '#E5B181',      # Light peach
                    'grid': '#E8E8E8',             # Light gray for grid
                    'background': '#FAFAFA'        # Very light gray background
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
                
                fig = plt.figure(figsize=(16, 12), facecolor=colors['background'])
                
                # Create a grid layout with very thin colorbar and wider main plots
                gs = fig.add_gridspec(4, 40, hspace=0.35, wspace=0.05)
                
                # Main plots take up most of the width (columns 0-37), colorbar is very thin (38-39)
                ax1 = fig.add_subplot(gs[0, :37])
                ax2 = fig.add_subplot(gs[1, :37])
                ax3 = fig.add_subplot(gs[2, :37])
                ax4 = fig.add_subplot(gs[3, :37])
                
                # Very thin colorbar axis
                cbar_ax = fig.add_subplot(gs[3, 38:])

                # Time axis in minutes (assuming 4Hz sampling rate)
                time_axis = np.arange(len(ground_truth)) / 4.0 / 60  # Convert to minutes
                pred_time_axis = np.arange(len(pred_mu)) / 4.0 / 60  # Prediction time axis

                # Plot 1: Ground truth raw signal
                ax1.plot(time_axis, ground_truth, color=colors['ground_truth'], 
                        linewidth=1.2, label='Ground Truth', alpha=0.9)
                ax1.set_title('Ground Truth Raw FHR Signal', fontweight='medium', pad=15)
                ax1.set_ylabel('Normalized Amplitude')
                ax1.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
                ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                ax1.set_facecolor('white')
                
                # Plot 2: Predicted raw signal with uncertainty
                # Ensure arrays have compatible lengths
                min_len = min(len(pred_time_axis), len(pred_mu), len(pred_std))
                if min_len > 0:
                    pred_time_cropped = pred_time_axis[:min_len]
                    pred_mu_cropped = pred_mu[:min_len]
                    pred_std_cropped = pred_std[:min_len]
                    
                    # Plot uncertainty band first (so it appears behind the line)
                    ax2.fill_between(pred_time_cropped, 
                                   pred_mu_cropped - pred_std_cropped, 
                                   pred_mu_cropped + pred_std_cropped,
                                   alpha=0.25, color=colors['uncertainty'], 
                                   label='±1σ Uncertainty', edgecolor='none')
                    
                    # Plot prediction line on top
                    ax2.plot(pred_time_cropped, pred_mu_cropped, 
                            color=colors['prediction'], linewidth=1.2, 
                            label='Predicted Mean', alpha=0.9)
                    
                    # Add vertical lines to show all prediction window starts
                    if prediction_start_times_minutes:
                        for i, start_time in enumerate(prediction_start_times_minutes):
                            # Only show lines that are within the current plot range
                            if start_time <= pred_time_cropped[-1]:
                                line_label = 'Prediction Windows' if i == 0 else ""
                                ax2.axvline(x=start_time, color='#2D8B2D', linestyle='--', 
                                          linewidth=1.2, alpha=0.7, label=line_label)
                        
                        # Add annotation for the first prediction window
                        if prediction_start_times_minutes[0] <= pred_time_cropped[-1]:
                            y_range = pred_mu_cropped.max() - pred_mu_cropped.min()
                            annotation_y = pred_mu_cropped.min() + y_range * 0.85
                            first_start = prediction_start_times_minutes[0]
                            ax2.annotate(f'2-min Windows\n({len(prediction_start_times_minutes)} total)', 
                                       xy=(first_start, annotation_y),
                                       xytext=(first_start + pred_time_cropped[-1] * 0.05, annotation_y),
                                       fontsize=8, color='#2D8B2D', alpha=0.8,
                                       arrowprops=dict(arrowstyle='->', color='#2D8B2D', alpha=0.6, lw=0.8),
                                       ha='left', va='center')
                    
                    ax2.set_title('Predicted Raw FHR Signal with Uncertainty', fontweight='medium', pad=15)
                    ax2.set_ylabel('Normalized Amplitude')
                    ax2.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
                    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                    ax2.set_facecolor('white')
                else:
                    ax2.text(0.5, 0.5, 'No prediction data available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax2.transAxes, fontsize=11, color='#666666')
                    ax2.set_title('Predicted Raw FHR Signal (No Data)', fontweight='medium', pad=15)
                    ax2.set_facecolor('white')
                
                # Plot 3: Comparison overlay with refined styling
                # Match the lengths for comparison
                comparison_len = min(len(time_axis), len(pred_time_axis), len(ground_truth), len(pred_mu))
                if comparison_len > 0:
                    time_comp = time_axis[:comparison_len]
                    gt_comp = ground_truth[:comparison_len]
                    pred_comp = pred_mu[:comparison_len]
                    
                    # Ground truth with slightly thicker line
                    ax3.plot(time_comp, gt_comp, color=colors['ground_truth'], 
                            linewidth=1.4, alpha=0.85, label='Ground Truth')
                    
                    # Prediction with thinner solid line for better comparison
                    ax3.plot(time_comp, pred_comp, color=colors['prediction'], 
                            linewidth=0.9, alpha=0.9, label='Predicted', linestyle='-')
                    
                    ax3.set_title('Ground Truth vs Predicted Raw FHR Signal', fontweight='medium', pad=15)
                    ax3.set_ylabel('Normalized Amplitude')
                    ax3.set_xlabel('Time (minutes)')
                    ax3.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
                    ax3.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                    ax3.set_facecolor('white')
                    
                    # Calculate metrics
                    mse = np.mean((gt_comp - pred_comp) ** 2)
                    mae = np.mean(np.abs(gt_comp - pred_comp))
                    correlation = np.corrcoef(gt_comp, pred_comp)[0, 1] if len(gt_comp) > 1 else 0
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data for comparison', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax3.transAxes, fontsize=11, color='#666666')
                    ax3.set_title('Ground Truth vs Predicted (No Data)', fontweight='medium', pad=15)
                    ax3.set_facecolor('white')
                    mse, mae, correlation = np.nan, np.nan, np.nan
                
                # Plot 4: Latent representation with professional colormap and thin colorbar
                if z_latent is not None and z_latent.size > 0:
                    # Use a professional colormap suitable for research papers
                    im = ax4.imshow(z_latent, aspect='auto', cmap='RdYlBu_r', 
                                  interpolation='nearest', alpha=0.9)
                    ax4.set_title('Latent Representation', fontweight='medium', pad=15)
                    ax4.set_xlabel('Time Steps')
                    ax4.set_ylabel('Latent Dimensions')
                    ax4.set_facecolor('white')
                    
                    # Add thin colorbar to the dedicated axis on the right
                    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
                    cbar_ax.set_ylabel('Latent Value', rotation=270, labelpad=12, fontsize=9)
                    cbar.ax.tick_params(labelsize=8)
                    
                    # Style the colorbar
                    cbar.outline.set_linewidth(0.5)
                    cbar.outline.set_edgecolor('#CCCCCC')
                else:
                    ax4.text(0.5, 0.5, 'Latent representation not available', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax4.transAxes, fontsize=11, color='#666666')
                    ax4.set_title('Latent Representation (N/A)', fontweight='medium', pad=15)
                    ax4.set_xlabel('Time Steps')
                    ax4.set_ylabel('Latent Dimensions')
                    ax4.set_facecolor('white')
                    # Hide the colorbar axis if no latent data
                    cbar_ax.set_visible(False)
                
                # Get batch info
                try:
                    guid = batch.guid[0] if hasattr(batch, 'guid') else 'unknown'
                    epoch_info = batch.epoch[0].item() if hasattr(batch, 'epoch') else 'unknown'
                except:
                    guid = 'unknown'
                    epoch_info = 'unknown'
                
                # Overall title with metrics
                plt.suptitle(f"Raw Signal Prediction - GUID: {guid}, Epoch: {epoch_info}\n"
                           f"MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {correlation:.4f}", 
                           fontsize=14, y=0.98)
                
                plot_path = f"{self.output_dir}/raw_signal_prediction_e-{pl_trainer.current_epoch}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                logger.info(f"Raw signal prediction plot saved to {plot_path}")
                
                # Explicit cleanup
                plt.close('all')
                
                # Clean up tensors to free GPU memory
                del model_outputs, raw_predictions
                if z_latent is not None:
                    del z_latent
                del y_st, y_ph, x_ph, y_raw_normalized, pred_mu, pred_std, ground_truth
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



class MetricsLoggingCallbackOriginal(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve the logged training metrics for this epoch.
        logs = trainer.callback_metrics
        # You can change the keys below if your module uses different names.
        train_loss = logs.get("train_loss")
        train_acc = logs.get("train_acc")
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Retrieve the logged validation metrics for this epoch.
        logs = trainer.callback_metrics
        val_loss = logs.get("validation_loss")
        val_acc = logs.get("validation_acc")
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_acc)


class BinaryClassificationMetricsPlotter(Callback):
    """
    This callback collects the predictions, true labels, and probabilities from
    each validation batch. At the end of the validation epoch (if the current epoch
    is a multiple of `plot_freq`), it computes and saves the following plots:

      - Confusion Matrix
      - ROC Curve (with AUROC)
      - Precision-Recall Curve (with Average Precision)
      - A summary figure with Accuracy, F1 Score, AUROC, and PRAUC

    Args:
        plot_freq (int): Frequency (in epochs) at which to generate the plots.
        output_dir (str): Directory where the plots will be saved.
    """
    def __init__(self, plot_freq=20, output_dir="metrics_plots"):
        super().__init__()
        self.plot_freq = plot_freq
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "confusion_matrix"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "roc"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "pr"), exist_ok=True)
        self.reset_buffers()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams.update({
            'font.size': 10
        })

    def reset_buffers(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Collect predictions, labels, and probabilities from each validation batch.
        if outputs is None:
            return
        preds = outputs.get("preds")
        labels = outputs.get("labels")
        probs = outputs.get("probs")
        if preds is not None:
            self.all_preds.append(preds.detach().cpu())
        if labels is not None:
            self.all_labels.append(labels.detach().cpu())
        if probs is not None:
            self.all_probs.append(probs.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Only plot at the specified frequency.
        if (epoch + 1) % self.plot_freq != 0:
            self.reset_buffers()
            return

        # Aggregate results across validation batches.
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()
        probs = torch.cat(self.all_probs).numpy()

        if probs.ndim > 1 and probs.shape[1] == 2:
            probs = probs[:, 1]

        # --- Confusion Matrix ---
        cm = confusion_matrix(labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix (Epoch {epoch+1})")
        cm_path = os.path.join(self.output_dir, f"confusion_matrix/confusion_matrix_epoch_{epoch+1}.png")
        plt.savefig(cm_path)
        plt.close()

        # --- ROC Curve and AUROC ---
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='#D55E00', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')  # Deep orange
        plt.plot([0, 1], [0, 1], color='#7F4F24', lw=2, linestyle='--')  # Brown diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title(f'ROC Curve (Epoch {epoch + 1})', fontsize=12)
        plt.legend(loc="lower right", fontsize=12)
        roc_path = os.path.join(self.output_dir, f"roc/roc_curve_epoch_{epoch + 1}.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()

        # --- Precision-Recall Curve and Average Precision ---
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (Epoch {epoch+1})', fontsize=12)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(loc="lower left", fontsize=12)
        pr_path = os.path.join(self.output_dir, f"pr/pr_curve_epoch_{epoch+1}.png")
        plt.savefig(pr_path)
        plt.close()

        # --- Summary of Key Metrics ---
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        plt.figure(figsize=(6, 4))
        plt.text(0.1, 0.8, f'Accuracy: {acc:.2f}', fontsize=12)
        plt.text(0.1, 0.65, f'F1 Score: {f1:.2f}', fontsize=12)
        plt.text(0.1, 0.5, f'AUROC: {roc_auc:.2f}', fontsize=12)
        plt.text(0.1, 0.35, f'Average Precision (PRAUC): {pr_auc:.2f}', fontsize=12)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.axis('off')
        plt.title(f'Classification Metrics Summary (Epoch {epoch+1})')
        summary_path = os.path.join(self.output_dir, f"metrics_summary_epoch_{epoch+1}.png")
        plt.savefig(summary_path)
        plt.close()

        # Clear buffers for the next epoch.
        self.reset_buffers()



class MetricsLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_binary_acc_history = []
        self.val_binary_acc_history = []
        self.train_binary_sensitivity_history = []
        self.val_binary_sensitivity_history = []
        self.train_binary_specificity_history = []
        self.val_binary_specificity_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve the logged training metrics for this epoch.
        logs = trainer.callback_metrics
        # Change keys to those logged in your module for the binary metrics.
        train_loss = logs.get("train_loss")
        # For example, if your module logs binary accuracy, sensitivity, and specificity:
        train_bin_acc = logs.get("train_binary_accuracy")
        train_bin_sens = logs.get("train_binary_sensitivity")
        train_bin_spec = logs.get("train_binary_specificity")
        self.train_loss_history.append(train_loss)
        self.train_binary_acc_history.append(train_bin_acc)
        self.train_binary_sensitivity_history.append(train_bin_sens)
        self.train_binary_specificity_history.append(train_bin_spec)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Retrieve the logged validation metrics for this epoch.
        logs = trainer.callback_metrics
        val_loss = logs.get("validation_loss")
        # Replace the keys with your binary metric keys.
        val_bin_acc = logs.get("val_binary_accuracy")
        val_bin_sens = logs.get("val_binary_sensitivity")
        val_bin_spec = logs.get("val_binary_specificity")
        self.val_loss_history.append(val_loss)
        self.val_binary_acc_history.append(val_bin_acc)
        self.val_binary_sensitivity_history.append(val_bin_sens)
        self.val_binary_specificity_history.append(val_bin_spec)



# ------------------------------------------------------------------------------------------------------------------------------------------
# Utility Methods
# ------------------------------------------------------------------------------------------------------------------------------------------
custom_theme = RichProgressBarTheme(
    description="blue",
    progress_bar="green1",
    progress_bar_finished="bright_green",
    progress_bar_pulse="cyan",
    batch_progress="magenta",
    time="grey82",
    processing_speed="white",
    metrics="yellow",
)


class CustomTQDMProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format = "{l_bar}{bar:20}{r_bar}{bar:-10b}"
        return bar


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
        
        # Clear GPU cache at the start of each epoch
        if torch.cuda.is_available():
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
        # Periodic GPU cache clearing to prevent memory buildup
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Clean up after each validation batch."""
        # Periodic GPU cache clearing to prevent memory buildup
        if batch_idx % 5 == 0 and torch.cuda.is_available():
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
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
            logger.info(f"{prefix} GPU {device}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            return allocated
        return 0.0
    
    def _clear_memory_if_needed(self):
        """Clear GPU memory if usage exceeds threshold."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            if allocated > self.threshold_gb:
                logger.warning(f"GPU memory usage ({allocated:.2f}GB) exceeds threshold ({self.threshold_gb}GB). Clearing cache...")
                torch.cuda.empty_cache()
                return True
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
