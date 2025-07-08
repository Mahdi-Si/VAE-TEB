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

from vae_teb_model_improved import SeqVaeTeb

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

        # Fetch a single batch from the validation dataloader
        val_dataloader = pl_trainer.datamodule.val_dataloader() if hasattr(pl_trainer.datamodule, 'val_dataloader') else pl_trainer.val_dataloaders[0]
        try:
            batch = next(iter(val_dataloader))
        except StopIteration:
            logger.warning("Could not get a batch from validation dataloader for plotting.")
            return

        # Ensure batch is on the correct device
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)

        pl_module.eval()
        try:
            with torch.no_grad():
                if not isinstance(pl_module, LightSeqVaeTeb):
                    return

                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                forward_outputs = pl_module.model(y_st, y_ph, x_ph)
                avg_preds = pl_module.model.get_average_predictions(forward_outputs)

                fhr_st_mean_pred = avg_preds['scattering_mu']
                fhr_ph_mean_pred = avg_preds['phase_harmonic_mu']
                z_latent = forward_outputs['z'].permute(0, 2, 1)[0].detach().cpu().numpy()

                from utils.data_utils import plot_forward_pass
                plot_forward_pass(
                    raw_fhr=batch.fhr[0].detach().cpu().numpy(),
                    raw_up=batch.up[0].detach().cpu().numpy(),
                    fhr_st=y_st[0].detach().cpu().numpy(),
                    fhr_ph=y_ph[0].detach().cpu().numpy(),
                    fhr_st_mean_pred=fhr_st_mean_pred[0].detach().cpu().numpy(),
                    fhr_ph_mean_pred=fhr_ph_mean_pred[0].detach().cpu().numpy(),
                    z_latent=z_latent,
                    plot_dir=self.output_dir,
                    plot_title=f"GUID: {batch.guid[0]}, Epoch: {batch.epoch[0].item()}",
                    tag=f'e-{pl_trainer.current_epoch}'
                )
                
                # Explicit cleanup of matplotlib figures and memory
                plt.close('all')
                
                # Clean up tensors to free GPU memory
                del forward_outputs, avg_preds, fhr_st_mean_pred, fhr_ph_mean_pred, z_latent
                del y_st, y_ph, x_ph
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            logger.error(f"Error during plotting: {e}")
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
            "train/scattering_loss": [],
            "train/phase_loss": [],
            "val/total_loss": [],
            "val/recon_loss": [],
            "val/kld_loss": [],
            "val/scattering_loss": [],
            "val/phase_loss": []
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
        
        # Clear any cached computations
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        
        forward_outputs = self.model(y_st, y_ph, x_ph)
        loss_dict = self.model.compute_loss(
            forward_outputs, y_st, y_ph,
            compute_scattering_loss=True,
            compute_phase_loss=True,
            compute_kld_loss=True
        )
        
        # Clean up intermediate tensors to free memory
        del y_st, y_ph, x_ph
        if 'forward_outputs' in locals():
            # Only delete if we have a reference to avoid errors
            for key in list(forward_outputs.keys()):
                if key not in ['mu', 'logvar', 'z']:  # Keep essential outputs for loss computation
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
        self.log('train/scattering_loss', loss_dict['scattering_loss'], on_step=False, on_epoch=True, logger=True)
        self.log('train/phase_loss', loss_dict['phase_loss'], on_step=False, on_epoch=True, logger=True)

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
        self.log('val/scattering_loss', loss_dict['scattering_loss'], on_epoch=True, logger=True)
        self.log('val/phase_loss', loss_dict['phase_loss'], on_epoch=True, logger=True)

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
