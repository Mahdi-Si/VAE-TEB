import lightning as L
import sklearn.utils
import glob
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from pytorch_lightning_modules_te import LightSeqVAE, LightSeqVAEClassifier, PlottingCallBack, CustomTQDMProgressBar, \
    BinaryClassificationMetricsPlotter, MetricsLoggingCallback, LossPlotCallback
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import OrderedDict
import yaml
import logging
from datetime import datetime
import sys
import pickle
import argparse
from tqdm import tqdm
import time
from pathlib import Path
import numpy as np
# todo: import the correct model
# from SeqVAE_model_2_Channel import VRNNGauss, ClassifierBlock, VRNNClassifier
# from seqvae_model import SeqVaeTebRec, SeqVaeTebBase, SeqVaeTebPrd, SeqVaeTebSimCls
# from seqvae_model_1min import SeqVaeTebRec, SeqVaeTebBase, SeqVaeTebPrd, SeqVaeTebSimCls
from EarlyMaestra.early_maestra.vae.Variational_AutoEncoder.seqvae_teb.model.seqvae_teb import SeqVaeTebRec, SeqVaeTebBase, SeqVaeTebPrd, SeqVaeTebSimCls
from Variational_AutoEncoder.utils.data_utils import \
    plot_forward_pass, \
    plot_averaged_results, \
    plot_generated_samples, \
    plot_distributions, \
    plot_histogram, \
    plot_loss_dict, \
    plot_latent_interpolation, \
    animate_latent_interpolation, \
    plot_original_reconstructed, \
    analyze_and_plot_classification_metrics, \
    analyze_class_stats_and_plot
from Variational_AutoEncoder.utils.classification_analysis_BD import run_bd_analysis, aggregate_and_plot_fold_comparison
from Variational_AutoEncoder.utils.graph_model_utils import \
    calculate_log_likelihood, \
    interpolate_latent, \
    calculate_vaf
from Variational_AutoEncoder.utils.run_utils import \
    log_resource_usage, \
    StreamToLogger, \
    setup_logging, \
    setup_logging_lightning, \
    StreamToLoggerLightning
from Variational_AutoEncoder.datasets.custom_datasets import \
    CTGDataset, \
    CTGDatasetWithGuid, \
    RepeatSampleDataset,\
    NumpyArrayDataset, \
    TensorDataset, \
    CTGDatasetComplete, \
    CTGDatasetWithLabel

from Variational_AutoEncoder.utils.analyze_folds import aggregate_folds, plot_raw_values, find_threshold_for_fpr, apply_strike_labeling
from Variational_AutoEncoder.utils.classification_analysis_BD import run_bd_fold_aggregation_for_folds

import re
from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
from sklearn.manifold import TSNE
import pandas as pd
from inv_st import RecScatteringNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "7"

matplotlib.use('Agg')
torch.backends.cudnn.enabled = False


import numpy as np

def reorder_signals_to_end(fhr_list, up_list, weight_list):
    """
    Reorder each (fhr, up, weight) triple so that the valid samples
    (where weight != 0) come first, and the invalid samples (weight == 0)
    are shifted to the end.

    fhr_list, up_list, weight_list are all lists of 1D NumPy arrays.
    They all must have the same length.
    """
    for i in range(len(fhr_list)):
        # Identify valid vs invalid samples by checking weights
        valid_mask = (weight_list[i] != 0)
        invalid_mask = ~valid_mask  # or (weight_list[i] == 0)

        # Reorder so that valid samples come first, invalid come second
        fhr_list[i] = np.concatenate([fhr_list[i][valid_mask],
                                      fhr_list[i][invalid_mask]])
        up_list[i] = np.concatenate([up_list[i][valid_mask],
                                     up_list[i][invalid_mask]])
        weight_list[i] = np.concatenate([weight_list[i][valid_mask],
                                         weight_list[i][invalid_mask]])

    return fhr_list, up_list, weight_list


def flip_16_before_first_zero(mask):
    """
    mask: 1D NumPy array of length N (values in {0,1}).
    Find the *first* zero in mask, and set the 16 entries immediately
    prior to it (if they exist) to 0 as well.

    This does NOT change the array's length. It only flips up to 16 ones
    to zero, keeping everything else intact.
    """
    # 1) argmax(...) returns the *first* index where (mask == 0) is True
    #    because for booleans argmax finds the first 'True' (value=1).
    first_zero_idx = (mask == 0).argmax()

    # 2) Check if we actually found a zero
    #    - If mask has no zero at all, then argmax() will return 0
    #      but mask[0] won't be zero => do nothing.
    if mask[first_zero_idx] == 0:
        # 3) Flip up to 16 samples before the zero
        start = max(0, first_zero_idx - 16)
        mask[start:first_zero_idx] = 0

    return mask


def extend_mask_left(mask_list, n_extend=16):
    """
    Given a list of 1D NumPy arrays representing masks,
    prepend `n_extend` zeros to the *left* of each mask.
    """
    new_list = []
    for m in mask_list:
        # If m.shape == (T,), then new shape will be (T + n_extend,).
        # Prepend n_extend zeros on the left.
        m_extended = np.pad(m, pad_width=(n_extend, 0),
                            mode='constant', constant_values=0)
        new_list.append(m_extended)
    return new_list


def compute_cls_num_list(dataloader, labeling_start_idx=200):
    """
    Computes the number of valid time steps for each class (Healthy and HIE)
    in the training dataloader.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The training dataloader. It is assumed that each batch is a tuple where
        the second element (index 1) is the labels tensor of shape (B, seq_len, 3).
    labeling_start_idx : int, optional
        The time step index from which valid labels start (to ignore early time steps).

    Returns
    -------
    list
        A list [healthy_count, hie_count] representing the number of valid samples
        for the Healthy and HIE classes.
    """
    healthy_count = 0
    hie_count = 0
    for batch in dataloader:
        # batch[1] is assumed to be the labels tensor with shape (B, seq_len, 3)
        labels = batch[1]
        # Slice to ignore early time steps (if applicable)
        labels = labels[:, labeling_start_idx:, :]
        # Convert one-hot to class indices:
        # 0 = pad, 1 = Healthy, 2 = HIE.
        true_labels = labels.argmax(dim=-1)
        # Filter out the padding values (class 0)
        valid_mask = (true_labels != 0)
        valid_labels = true_labels[valid_mask]
        # Count occurrences for Healthy (1) and HIE (2)
        max_class = true_labels.max().item()
        healthy_count += (valid_labels == 1).sum().item()
        hie_count += (valid_labels == max_class).sum().item()
    return [healthy_count, hie_count]


def convert_onehot(arr):
    converted_list = []

    for arr in arr:
        new_arr = []
        for row in arr:
            if np.array_equal(row, [1, 0, 0, 0]):
                new_arr.append([1, 0, 0])
            elif np.array_equal(row, [0, 1, 0, 0]):
                new_arr.append([0, 1, 0])
            else:
                new_arr.append([0, 0, 1])
        converted_list.append(np.array(new_arr))

    return converted_list



class SeqVAEGraphModel:
    def __init__(self, config_file_path=None, device_id=None, cuda_devices=None):
        super(SeqVAEGraphModel, self).__init__()
        self.cuda_devices = cuda_devices
        if config_file_path is None:
            self.config_file_path = os.path.dirname(os.path.realpath(__file__)) + '/seqvae_configs/config_args_cls.yaml'
        else:
            self.config_file_path = config_file_path

        with open(self.config_file_path) as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        now = datetime.now()
        run_date = now.strftime("%Y-%m-%d--[%H-%M]-")
        self.experiment_tag = self.config['general_config']['tag']
        self.mimo_trainer_mode = self.config['general_config']['mimo_trainer_mode']
        if self.mimo_trainer_mode:
            self.output_base_dir = os.getcwd()
            self.cuda_devices = self.config['general_config']['mimo_trainer_cuda']
        else:
            self.output_base_dir = os.path.normpath(self.config['folders_config']['out_dir_base'])
        self.base_folder = f'{run_date}-{self.experiment_tag}'
        self.train_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'train_results')
        self.test_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'test_results')
        self.model_checkpoint_dir = os.path.join(self.output_base_dir, self.base_folder, 'model_checkpoints')
        self.general_logs = os.path.join(self.output_base_dir, self.base_folder, 'general_logs')
        self.device = device_id
        self.log_file = None
        self.logger = None

        # print yaml file properly -------------------------------------------------------------------------------------
        print(yaml.dump(self.config, sort_keys=False, default_flow_style=False))
        print('==' * 50)
        self.stat_path = os.path.normpath(self.config['dataset_config']['stat_path'])

        self.plot_every_epoch = self.config['general_config']['plot_frequency']
        self.previous_check_point = self.config['general_config']['checkpoint_path']


        self.raw_input_size = self.config['model_config']['VAE_model']['raw_input_size']
        self.input_size = self.config['model_config']['VAE_model']['input_size']

        self.input_dim = self.config['model_config']['VAE_model']['input_dim']
        self.input_channel_num = self.config['model_config']['VAE_model']['channel_num']

        self.latent_dim = self.config['model_config']['VAE_model']['latent_size']
        self.num_layers = self.config['model_config']['VAE_model']['num_RNN_layers']
        self.rnn_hidden_dim = self.config['model_config']['VAE_model']['RNN_hidden_dim']
        self.y_module_only = self.config['model_config']['VAE_model']['Y_module_only']
        self.epochs_num = self.config['general_config']['epochs']
        self.lr = self.config['general_config']['lr']
        self.lr_milestones = self.config['general_config']['lr_milestone']
        self.kld_beta_ = float(self.config['model_config']['VAE_model']['kld_beta'])
        self.seqvae_ckp = self.config['model_config']['seqvae_checkpoint']

        self.train_classifier = self.config['general_config']['train_classifier']
        if self.train_classifier:
            self.pytorch_ckp = self.config['model_config']['classification_checkpoint']
        else:
            self.pytorch_ckp = self.seqvae_ckp
        self.freeze_seqvae = self.config['model_config']['VAE_model']['freeze_seqvae']
        self.batch_size_train = self.config['general_config']['batch_size']['train']
        self.batch_size_test = self.config['general_config']['batch_size']['test']

        self.test_checkpoint_path = None
        self.seqvae_testing_checkpoint = self.config['seqvae_testing']['test_checkpoint_path']
        self.base_model_ckp = self.config['model_config']['base_model_checkpoint']

        self.inv_scattering_checkpoint = self.config['inv_scattering_model']['inv_st_checkpoint']
        self.do_inv_st = self.config['inv_scattering_model']['do_inv_st']
        self.train_inv_st = self.config['inv_scattering_model']['train_inv_st']
        self.blood_gas_file = self.config['dataset_config']['blood_gas_file']
        self.folds_dataset = self.config['dataset_config']['folds_dataset']
        self.batch_size = config['general_config']['batch_size']['train']
        self.per_step_label = config['model_config']['per_step_label']
        self.use_balanced_loss = config['model_config']['use_balanced_loss']

        self.clip = 10
        plt.ion()

        self.strike = config['classification_config']['strike']
        self.folds = self.config['classification_config']['folds']
        self.target_fpr = self.config['classification_config']['target_fpr']

        self.log_stat = None
        self.latent_stats = None
        self.model = None
        self.seqvae_lightning_model = None
        self.classifier = None
        self.inv_scattering_model = None
        self.csv_logger = None
        self.plotting_callback = None
        self.classification_performance_callback = None
        self.base_model = None
        self.pytorch_model = None
        self.prd_base_model = None
        self.checkpoint_callback = None
        self.early_stop_callback = None
        self.loss_plot_callback = None
        self.cls_num_list = None
        self.base_fold_folder = None
        self.fold_path_list = []
        self.min_domain_start = None
        self.max_domain_start = None

    def setup_config(self):
        folders_list = [
            self.output_base_dir,
            self.train_results_dir,
            self.test_results_dir,
            self.model_checkpoint_dir,
        ]
        for folder in folders_list:
            os.makedirs(folder, exist_ok=True)

        self.log_file = os.path.join(self.train_results_dir, 'log.txt')
        self.logger = setup_logging(self.log_file)
        sys.stdout = StreamToLogger(self.logger, logging.INFO)
        print(yaml.dump(self.config, sort_keys=False, default_flow_style=False))
        print('==' * 50)

        self.log_stat = np.load(self.stat_path, allow_pickle=True).item()

    def load_pytorch_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is not None:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['state_dict']
            # filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'scattering_transform' not in k}
            state_dict = {k.replace('seqvae_model.', ''): v for k, v in state_dict.items()}
            self.pytorch_model.load_state_dict(state_dict)
            print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            if self.seqvae_ckp is not None:
                print(f"Loading checkpoint: {self.seqvae_ckp}")
                # checkpoint = torch.load(self.seqvae_checkpoint_path,  map_location=self.device)
                checkpoint = torch.load(self.seqvae_ckp)
                state_dict = checkpoint['state_dict']
                # filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'scattering_transform' not in k}
                state_dict = {k.replace('seqvae_model.', ''): v for k, v in state_dict.items()}
                self.pytorch_model.load_state_dict(state_dict)
                print(f"Loaded checkpoint '{self.seqvae_ckp}' (epoch {checkpoint['epoch']})")

    def create_model(self):
        self.setup_config()
        self.base_model = SeqVaeTebBase(
            input_size=self.raw_input_size,
            input_dim=self.input_dim,
            h_dim=self.rnn_hidden_dim,
            z_dim=self.latent_dim,
            n_layers=self.num_layers,
            log_stat=self.log_stat,
        )
        self.plotting_callback = PlottingCallBack(
            output_dir=self.train_results_dir,
            plot_every_epoch=self.plot_every_epoch,
            input_channel_num=self.input_channel_num,
        )
        classification_performance_folder = os.path.join(self.train_results_dir, 'classification_performance')
        os.makedirs(classification_performance_folder, exist_ok=True)
        self.classification_performance_callback = BinaryClassificationMetricsPlotter(
            output_dir=classification_performance_folder, plot_freq=self.plot_every_epoch
        )
        self.metrics_callback = MetricsLoggingCallback()
        if self.train_classifier:
            self.prd_base_model = SeqVaeTebPrd(base_model=self.base_model)
            if self.seqvae_ckp is not None:
                print("=" * 100)
                print(f"Loading vae base checkpoint for classification: {self.seqvae_ckp}")
                checkpoint = torch.load(self.seqvae_ckp)
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace('seqvae_model.', ''): v for k, v in state_dict.items()}
                self.prd_base_model.load_state_dict(state_dict)
                print(f"Loaded checkpoint '{self.seqvae_ckp}' (epoch {checkpoint['epoch']})")
                print("=" * 100)
            self.pytorch_model = SeqVaeTebSimCls(vae_base_model=self.prd_base_model, freeze_vae=self.freeze_seqvae)
            if self.pytorch_ckp is not None:
                self.seqvae_lightning_model = LightSeqVAEClassifier.load_from_checkpoint(
                    self.pytorch_ckp,
                    seqvae_model=self.pytorch_model,
                    lr=self.lr,
                    lr_milestones=self.lr_milestones,
                    batch_size_train=self.batch_size_train,
                    batch_size_test=self.batch_size_test,
                    kl_beta=self.kld_beta_,
                    cls_num_list = self.cls_num_list,
                    use_ldam=self.use_balanced_loss,

                )
            else:
                self.seqvae_lightning_model = LightSeqVAEClassifier(
                    seqvae_model=self.pytorch_model,
                    lr=self.lr,
                    lr_milestones=self.lr_milestones,
                    batch_size_train=self.batch_size_train,
                    batch_size_test=self.batch_size_test,
                    kl_beta=self.kld_beta_,
                    cls_num_list=self.cls_num_list,
                    use_ldam=self.use_balanced_loss,
                )

        else:
            if self.base_model_ckp is not None:
                base_model_ckp = torch.load(self.base_model_ckp)
                base_model_state_dict = base_model_ckp['state_dict']
                prefix = 'seqvae_model.base_model.'
                base_state_dict = OrderedDict()
                for key, value in base_model_state_dict.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        base_state_dict[new_key] = value
                self.base_model.load_state_dict(base_state_dict)
            self.pytorch_model =  SeqVaeTebPrd(base_model=self.base_model)
            if self.pytorch_ckp is not None:
                self.seqvae_lightning_model = LightSeqVAE.load_from_checkpoint(
                    self.pytorch_ckp,
                    seqvae_model=self.pytorch_model,
                    lr=self.lr,
                    lr_milestones=self.lr_milestones,
                    batch_size_train=self.batch_size_train,
                    batch_size_test=self.batch_size_test,
                    kl_beta = self.kld_beta_,
                )
            else:
                self.seqvae_lightning_model = LightSeqVAE(
                    seqvae_model=self.pytorch_model,
                    lr=self.lr,
                    lr_milestones=self.lr_milestones,
                    batch_size_train=self.batch_size_train,
                    batch_size_test=self.batch_size_test,
                    kl_beta=self.kld_beta_,
                )
        logger.info('==' * 50)
        trainable_params = sum(p.numel() for p in self.seqvae_lightning_model.parameters() if p.requires_grad)
        logger.info(f'Trainable params of SeqVAE: {trainable_params:,}')
        total_params = sum(p.numel() for p in self.seqvae_lightning_model.parameters())
        logger.info(f'Total params of SeqVAE: {total_params:,}')
        model_size_mb = sum(p.numel() * p.element_size() for p in self.seqvae_lightning_model.parameters()) / (1024 * 1024)
        logger.info(f'Model size: {model_size_mb:.2f} MB')
        logger.info('==' * 50)
        logger.info('MODEL ARCHITECTURE:')
        logger.info(str(self.seqvae_lightning_model))
        
        # Log detailed module breakdown
        logger.info('\n' + '==' * 50)
        logger.info('DETAILED MODULE BREAKDOWN:')
        for name, module in self.seqvae_lightning_model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    logger.info(f'{name}: {type(module).__name__} - {params:,} parameters')
        
        # Log model blocks if available
        logger.info('\n' + '==' * 50)
        logger.info('MODEL BLOCKS:')
        if hasattr(self.seqvae_lightning_model, 'model'):
            model = self.seqvae_lightning_model.model
            if hasattr(model, 'encoder'):
                logger.info(f'Encoder: {type(model.encoder).__name__}')
                logger.info(str(model.encoder))
            if hasattr(model, 'decoder'):
                logger.info(f'Decoder: {type(model.decoder).__name__}')
                logger.info(str(model.decoder))
            if hasattr(model, 'prediction_head') or hasattr(model, 'predictor'):
                pred_head = getattr(model, 'prediction_head', getattr(model, 'predictor', None))
                if pred_head:
                    logger.info(f'Prediction Head: {type(pred_head).__name__}')
                    logger.info(str(pred_head))
        logger.info('==' * 50)


    def set_cuda_devices(self, device_list=None):
        self.cuda_devices = device_list if device_list is not None else [0]


    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    def train_seqvae_model(self, train_loader_seqvae=None,
                           validation_loader_seqvae=None):
        self.early_stop_callback = EarlyStopping(
            monitor="validation_loss",
            min_delta=0.0,
            patience=50,
            verbose=True,
            mode="min"
        )
        self.checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            mode="min",
            dirpath=self.model_checkpoint_dir,
            filename="best_checkpoint-{epoch:02d}-{validation_loss:.2f}",
            save_top_k=1,
            save_last=True,
        )
        self.loss_plot_callback = LossPlotCallback(output_dir=self.train_results_dir, plot_frequency=self.plot_every_epoch)
        profiler_g = SimpleProfiler(dirpath=self.train_results_dir, filename="profiler_log.txt")
        if len(self.cuda_devices) > 0:
            loging_steps = (len(train_loader_seqvae.dataset) // self.batch_size_train) // len(self.cuda_devices)
        else:
            loging_steps = (len(train_loader_seqvae.dataset) // self.batch_size_train)

        callbacks_list = [
            ModelSummary(max_depth=-1),
            self.plotting_callback,
            self.checkpoint_callback,
            self.loss_plot_callback
        ]
        if self.train_classifier:
            callbacks_list.append(self.metrics_callback)
            callbacks_list.append(self.classification_performance_callback)
            callbacks_list.append(self.early_stop_callback)

        trainer_graph_model = L.Trainer(
            devices=self.cuda_devices,
            log_every_n_steps=loging_steps,
            gradient_clip_val=0.5,
            accelerator="gpu",
            max_epochs=self.epochs_num,
            enable_checkpointing=True,
            enable_progress_bar=True,
            default_root_dir=os.path.normpath(self.train_results_dir),
            profiler=profiler_g,
            num_sanity_val_steps=0,
            # accumulate_grad_batches=2,
            callbacks=callbacks_list,
            strategy=DDPStrategy(find_unused_parameters=True)
        )
        #
        # tuner = Tuner(trainer_graph_model)
        # lr_finder = tuner.lr_find(self.seqvae_lightning_model, train_loader_seqvae, validation_loader_seqvae)
        # print(lr_finder.results)
        # fig = lr_finder.plot(suggest=True)
        # plt.savefig(os.path.join(self.train_results_dir, "lr.png"))
        # plt.close()

        trainer_graph_model.fit(self.seqvae_lightning_model, train_dataloaders=train_loader_seqvae,
                                val_dataloaders=validation_loader_seqvae)
        print('=' * 50)
        training_hist = self.loss_plot_callback.history
        return training_hist


    def do_train_with_dataset(self, train_dataset, validation_dataset, tag='', weights_filename=None):
        # self.calculate_latent_stats(train_dataset)
        self.lr = self.config['general_config']['lr']
        guid_lists, epoch_numbs_list, labels_list = train_dataset.dataset.get_the_lists()
        labels_true_list = []
        for label_i in labels_list:
            class_labels = np.argmax(label_i, axis=1)
            label_epoch = np.bincount(class_labels).argmax()
            if label_epoch == 0:
                labels_true_list.append('pad')
            elif label_epoch == 1:
                labels_true_list.append('Healthy')
            elif label_epoch == 2:
                labels_true_list.append('HIE')
        train_df = pd.DataFrame({
            "guid": guid_lists,
            "epoch": epoch_numbs_list,
            "label": labels_true_list,
        })
        train_df.to_csv(os.path.join(self.train_results_dir, 'train_dataset.csv'), index=False)

        guid_lists, epoch_numbs_list, labels_list = validation_dataset.dataset.get_the_lists()
        labels_true_list = []
        for label_i in labels_list:
            class_labels = np.argmax(label_i, axis=1)
            label_epoch = np.bincount(class_labels).argmax()
            if label_epoch == 0:
                labels_true_list.append('pad')
            elif label_epoch == 1:
                labels_true_list.append('Healthy')
            elif label_epoch == 2:
                labels_true_list.append('HIE')
        train_df = pd.DataFrame({
            "guid": guid_lists,
            "epoch": epoch_numbs_list,
            "label": labels_true_list,
        })
        train_df.to_csv(os.path.join(self.train_results_dir, 'validation_dataset.csv'), index=False)
        self.create_model()
        history_dict = self.train_seqvae_model(train_dataset, validation_dataset)
        return history_dict

    def train_and_test_folds(self,
                             class_positive_train='acidosis_train.pkl',
                             class_positive_val='acidosis_val.pkl',
                             class_positive_test='acidosis_test.pkl',
                             class_negative_train='healthy_train.pkl',
                             class_negative_val='healthy_val.pkl',
                             class_negative_test='healthy_test.pkl',
                             class_aux_train='hie_train.pkl',
                             class_aux_val='hie_val.pkl',
                             class_aux_test='hie_test.pkl',):
        for fold in self.folds:
            self.base_fold_folder = os.path.join(self.output_base_dir, self.base_folder, f'fold_{fold}')
            self.train_results_dir = os.path.join(self.base_fold_folder, f'train_fold_{fold}')
            self.test_results_dir = os.path.join(self.base_fold_folder, f'test_fold_{fold}')
            self.model_checkpoint_dir = os.path.join(self.base_fold_folder, 'model_checkpoints')

            os.makedirs(self.base_fold_folder, exist_ok=True)

            fold_folder = self.folds_dataset + f'fold_{fold}'
            class_positive_train_path = os.path.join(fold_folder, class_positive_train)
            class_positive_val_path = os.path.join(fold_folder, class_positive_val)
            class_positive_test_path = os.path.join(fold_folder, class_positive_test)
            class_negative_train_path = os.path.join(fold_folder, class_negative_train)
            class_negative_val_path = os.path.join(fold_folder, class_negative_val)
            class_negative_test_path = os.path.join(fold_folder, class_negative_test)
            class_aux_train_path = os.path.join(fold_folder, class_aux_train)
            class_aux_val_path = os.path.join(fold_folder, class_aux_val)
            class_aux_test_path = os.path.join(fold_folder, class_aux_test)

            with open(class_positive_train_path, 'rb') as file:
                class_positive_train_data = pickle.load(file)
            with open(class_positive_val_path, 'rb') as file:
                class_positive_val_data = pickle.load(file)
            with open(class_positive_test_path, 'rb') as file:
                class_positive_test_data = pickle.load(file)
            with open(class_negative_train_path, 'rb') as file:
                class_negative_train_data = pickle.load(file)
            with open(class_negative_val_path, 'rb') as file:
                class_negative_val_data = pickle.load(file)
            with open(class_negative_test_path, 'rb') as file:
                class_negative_test_data = pickle.load(file)
            with open(class_aux_train_path, 'rb') as file:
                class_aux_train_data = pickle.load(file)
            with open(class_aux_val_path, 'rb') as file:
                class_aux_val_data = pickle.load(file)
            with open(class_aux_test_path, 'rb') as file:
                class_aux_test_data = pickle.load(file)
            len_train = min(len(class_positive_train_data['fhr']), len(class_negative_train_data['fhr']))
            fhr_train_fold = class_positive_train_data['fhr'][:len_train] + class_negative_train_data['fhr'][:len_train]
            up_train_fold = class_positive_train_data['up'][:len_train] + class_negative_train_data['up'][:len_train]
            epoch_num_train_fold = class_positive_train_data['epoch'][:len_train] + class_negative_train_data['epoch'][:len_train]
            weight_train_fold = class_positive_train_data['sample_weight'][:len_train] + class_negative_train_data['sample_weight'][:len_train]
            labels_train_fold = convert_onehot(class_positive_train_data['target'][:len_train]) + convert_onehot(class_negative_train_data['target'][:len_train])
            guid_train_fold = class_positive_train_data['guid'][:len_train] + class_negative_train_data['guid'][:len_train]


            len_val = min(len(class_positive_val_data['fhr']), len(class_negative_val_data['fhr']))
            fhr_val_fold = class_positive_val_data['fhr'][:len_val] + class_negative_val_data['fhr'][:len_val]
            up_val_fold = class_positive_val_data['up'][:len_val] + class_negative_val_data['up'][:len_val]
            epoch_num_val_fold = class_positive_val_data['epoch'][:len_val] + class_negative_val_data['epoch'][:len_val]
            weight_val_fold = class_positive_val_data['sample_weight'][:len_val] + class_negative_val_data['sample_weight'][:len_val]
            labels_val_fold = convert_onehot(class_positive_val_data['target'][:len_val]) + convert_onehot(class_negative_val_data['target'][:len_val])
            guid_val_fold = class_positive_val_data['guid'][:len_val] + class_negative_val_data['guid'][:len_val]


            len_test = min(len(class_positive_test_data['fhr']), len(class_negative_test_data['fhr']))
            fhr_test_fold = class_positive_test_data['fhr'][:len_test] + class_negative_test_data['fhr'][:len_test]
            up_test_fold = class_positive_test_data['up'][:len_test] + class_negative_test_data['up'][:len_test]
            epoch_num_test_fold = class_positive_test_data['epoch'][:len_test] + class_negative_test_data['epoch'][:len_test]
            weight_test_fold = class_positive_test_data['sample_weight'][:len_test] + class_negative_test_data['sample_weight'][:len_test]
            labels_test_fold = convert_onehot(class_positive_test_data['target'][:len_test]) + convert_onehot(class_negative_test_data['target'][:len_test])
            guid_test_fold = class_positive_test_data['guid'][:len_test] + class_negative_test_data['guid'][:len_test]

            len_test += len_val
            fhr_test_fold += fhr_val_fold
            up_test_fold += up_val_fold
            epoch_num_test_fold += epoch_num_val_fold
            weight_test_fold += weight_val_fold
            labels_test_fold += labels_val_fold
            guid_test_fold += guid_val_fold

            fhr_aux_fold = class_aux_train_data['fhr'] + class_aux_test_data['fhr'] + class_negative_test_data['fhr']
            up_aux_fold = class_aux_train_data['up'] + class_aux_test_data['up'] + class_negative_test_data['up']
            epoch_num_aux_fold = class_aux_train_data['epoch'] + class_aux_test_data['epoch'] + class_negative_test_data['epoch']
            weight_aux_fold = class_aux_train_data['sample_weight'] + class_aux_test_data['sample_weight'] + class_negative_test_data['sample_weight']
            labels_aux_fold = convert_onehot(class_aux_train_data['target']) + convert_onehot(class_aux_test_data['target']) + convert_onehot(class_negative_test_data['target'])
            guid_aux_fold = class_aux_train_data['guid'] + class_aux_test_data['guid'] + class_negative_test_data['guid']

            #todo: Handel correct targets
            self.min_domain_start = min(epoch_num_train_fold)
            self.max_domain_start = max(epoch_num_train_fold)
            dataset_train_fold = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_train_fold[:],
                up_list_ctg_dataset=up_train_fold[:],
                guids_list_ctg_dataset=guid_train_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_train_fold[:],
                sample_weights_list_ctg_dataset=weight_train_fold[:],
                fhr_labels=labels_train_fold[:],
                num_channel=2,
                min_domain_start=min(epoch_num_train_fold),
                max_domain_start=max(epoch_num_train_fold)
            )
            dataset_test_fold = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_test_fold[:],
                up_list_ctg_dataset=up_test_fold[:],
                guids_list_ctg_dataset=guid_test_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_test_fold[:],
                sample_weights_list_ctg_dataset=weight_test_fold[:],
                fhr_labels=labels_test_fold[:],
                num_channel=2,
                min_domain_start=min(epoch_num_train_fold),
                max_domain_start=max(epoch_num_train_fold)
            )

            dataset_test_aux_fold = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_aux_fold[:],
                up_list_ctg_dataset=up_aux_fold[:],
                guids_list_ctg_dataset=guid_aux_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_aux_fold[:],
                sample_weights_list_ctg_dataset=weight_aux_fold[:],
                fhr_labels=labels_aux_fold[:],
                num_channel=2,
                min_domain_start=min(epoch_num_train_fold),
                max_domain_start=max(epoch_num_train_fold)
            )

            train_dataloader_fold = DataLoader(dataset_train_fold, batch_size=self.batch_size, shuffle=True, num_workers=0)
            test_dataloader_fold = DataLoader(dataset_test_fold, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_aux_dataloader_fold = DataLoader(dataset_test_aux_fold, batch_size=self.batch_size, shuffle=False, num_workers=0)
            # cls_num_list_fold = [len(class_positive_train_data['fhr']), len(class_negative_train_data['fhr']),]
            # cls_num_list_fold = compute_cls_num_list(train_dataloader_fold)
            # self.cls_num_list = cls_num_list_fold
            self.create_model()
            _ = self.train_seqvae_model(train_loader_seqvae=train_dataloader_fold, validation_loader_seqvae=test_dataloader_fold)
            self.test_classifier(test_dataloader_cls=test_dataloader_fold, tag="cls_test", do_bd_analysis=True)
            self.test_classifier(test_dataloader_cls=test_aux_dataloader_fold, tag="aux_test", do_bd_analysis=False)
        aggregate_folds(self.base_fold_folder, test_dir='cls_test', tag="acidosis")
        aggregate_folds(self.base_fold_folder, test_dir='aux_test', tag="HIE")
        run_bd_fold_aggregation_for_folds(os.path.join(self.output_base_dir, self.base_folder))

    def test_classifier(self, threshold=0.5, test_dataloader_cls=None, tag='cls_tests',
                        labeling_start_idx=100, checkpoint_for_test=None, do_bd_analysis=True):
        save_dir_cls_tests = os.path.join(self.test_results_dir, tag)
        os.makedirs(save_dir_cls_tests, exist_ok=True)
        if checkpoint_for_test is None:
            if self.pytorch_ckp is not None:
                best_model_checkpoint = self.pytorch_ckp
            else:
                best_model_checkpoint = self.checkpoint_callback.best_model_path
        else:
            best_model_checkpoint = checkpoint_for_test
        adjusted_threshold = 0.5
        self.load_pytorch_checkpoint(checkpoint_path=best_model_checkpoint)
        cuda_device = f"cuda:{self.cuda_devices[0]}"
        self.pytorch_model.to(cuda_device)
        self.pytorch_model.eval()
        results = []
        with torch.no_grad():
            for idx, batched_data in enumerate(test_dataloader_cls):
                input_data = batched_data[0].to(cuda_device)
                guids_list = batched_data[2]
                epoch_nums_list = batched_data[3]
                true_labels_list = batched_data[1].argmax(dim=2).mode(dim=1)[0] - 1
                sample_weights_list = batched_data[4]
                output = self.pytorch_model(input_data, zero_source=False, epoch_num=epoch_nums_list)
                logits = output.logits  # shape (batch_size, 300, 2)
                if self.per_step_label:
                    probabilities = F.softmax(logits[:, labeling_start_idx:, :], dim=2)  # shape: (batch_size, 300, 2)
                    avg_probs = torch.mean(probabilities, dim=1)  # shape: (batch_size, 2)
                else:
                    avg_probs = F.softmax(logits, dim=-1)
                avg_probs_cpu = avg_probs.cpu().numpy()

                predicted = (avg_probs[:, 1] > adjusted_threshold).long().cpu().numpy()
                min_epoch_num = min(epoch_nums_list)
                max_epoch_num = max(epoch_nums_list)
                for i in range(len(guids_list)):
                    results.append({
                        'guid': guids_list[i],
                        'epoch_num': epoch_nums_list[i].item() * (self.max_domain_start - self.min_domain_start) + self.min_domain_start,
                        'prob_class_0': avg_probs_cpu[i, 0],
                        'prob_class_1': avg_probs_cpu[i, 1],
                        'predicted_class': predicted[i],
                        'true_label': true_labels_list[i].item()
                    })
            df_results = pd.DataFrame(results,
                                      columns=['guid', 'epoch_num', 'prob_class_0', 'prob_class_1',
                                               'predicted_class', 'true_label'])
            csv_file_path = os.path.join(save_dir_cls_tests, 'classification_results.csv')
            cs_df = pd.read_csv(r"/data/deid/isilon/MS_model/cs_df.csv")
            df_results = df_results.merge(cs_df[['guid', 'CS']], on='guid', how='left')

            no_bg_df = pd.read_csv(r"/data/deid/isilon/MS_model/no_bg_df_complete.csv")
            df_results['no_bg'] = df_results['guid'].isin(no_bg_df['guid'])

            df_results.to_csv(csv_file_path, index=False)
            print(f"Test classification results saved to {csv_file_path}")
            analyze_and_plot_classification_metrics(csv_file_path, output_cls_test_dir=save_dir_cls_tests)
            analyze_class_stats_and_plot(csv_file_path, save_dir_cls_tests)
            if do_bd_analysis:
                bd_analysis_save_path = os.path.join(save_dir_cls_tests, 'bd_analysis')
                os.makedirs(bd_analysis_save_path, exist_ok=True)
                df_blood_gas = pd.read_csv(self.blood_gas_file)
                run_bd_analysis(output_base_dir=bd_analysis_save_path, df_classification=df_results, df_blood_gas=df_blood_gas)
        return csv_file_path

    def run_post_training_test(self, checkpoints_base_folder, tag,
                               class_positive_val='acidosis_val.pkl',
                               class_positive_test='acidosis_test.pkl',
                               class_negative_val='healthy_val.pkl',
                               class_negative_test='healthy_test.pkl',
                               class_aux_train='hie_train.pkl',
                               class_aux_val='hie_val.pkl',
                               class_aux_test='hie_test.pkl',
                               class_aux_healthy_cs_val='healthy_cs_val.pkl',
                               class_aux_healthy_cs_test='healthy_cs_test.pkl',
                               class_aux_healthy_no_bg_val='healthy_no_bg_val.pkl',
                               class_aux_healthy_no_bg_test='healthy_no_bg_test.pkl',
                               class_aux_healthy_no_bg_cs_val='healthy_no_bg_cs_val.pkl',
                               class_aux_healthy_no_bg_cs_test='healthy_no_bg_cs_test.pkl',
                               ):

        pattern = re.compile(r"^fold_(\d+)$")
        fold_indices = []
        for entry in Path(checkpoints_base_folder).iterdir():
            if entry.is_dir():
                m = pattern.match(entry.name)
                if m:
                    fold_indices.append(int(m.group(1)))
        # fold_indices = [1, 2, 4, 5, 6, 7, 8, 10]
        # fold_indices = [8, 10]
        for fold in fold_indices:
            print(f"Processing fold {fold}...")
            self.base_fold_folder = os.path.join(self.output_base_dir, self.base_folder, f'fold_{fold}')
            self.val_results_dir = os.path.join(self.base_fold_folder, f'val_fold_{fold}')
            self.test_results_dir = os.path.join(self.base_fold_folder, f'test_fold_{fold}')


            os.makedirs(self.base_fold_folder, exist_ok=True)
            os.makedirs(self.val_results_dir, exist_ok=True)
            os.makedirs(self.test_results_dir, exist_ok=True)

            # -- Dataset paths:
            fold_folder = os.path.join(self.folds_dataset, f'fold_{fold}')
            class_positive_val_path = os.path.join(fold_folder, class_positive_val)
            class_positive_test_path = os.path.join(fold_folder, class_positive_test)
            class_negative_val_path = os.path.join(fold_folder, class_negative_val)
            class_negative_test_path = os.path.join(fold_folder, class_negative_test)
            class_aux_train_path = os.path.join(fold_folder, class_aux_train)
            # class_aux_val_path = os.path.join(fold_folder, class_aux_val)
            # class_aux_test_path = os.path.join(fold_folder, class_aux_test)
            class_aux_healthy_cs_val_path = os.path.join(fold_folder, class_aux_healthy_cs_val)
            class_aux_healthy_cs_test_path = os.path.join(fold_folder, class_aux_healthy_cs_test)
            class_aux_healthy_no_bg_val_path = os.path.join(fold_folder, class_aux_healthy_no_bg_val)
            class_aux_healthy_no_bg_test_path = os.path.join(fold_folder, class_aux_healthy_no_bg_test)

            class_aux_healthy_no_bg_cs_val_path = os.path.join(fold_folder, class_aux_healthy_no_bg_cs_val)
            class_aux_healthy_no_bg__cs_test_path = os.path.join(fold_folder, class_aux_healthy_no_bg_cs_test)

            # -- Load datasets for valing and testing.
            with open(class_positive_val_path, 'rb') as file:
                class_positive_val_data = pickle.load(file)
            with open(class_positive_test_path, 'rb') as file:
                class_positive_test_data = pickle.load(file)
            with open(class_negative_val_path, 'rb') as file:
                class_negative_val_data = pickle.load(file)
            with open(class_negative_test_path, 'rb') as file:
                class_negative_test_data = pickle.load(file)
            with open(class_aux_train_path, 'rb') as file:
                class_aux_train_data = pickle.load(file)
            # with open(class_aux_val_path, 'rb') as file:
            #     class_aux_val_data = pickle.load(file)
            # with open(class_aux_test_path, 'rb') as file:
            #     class_aux_test_data = pickle.load(file)
            with open(class_aux_healthy_cs_val_path, 'rb') as file:
                class_aux_healthy_cs_val_data = pickle.load(file)
            with open(class_aux_healthy_cs_test_path, 'rb') as file:
                class_aux_healthy_cs_test_data = pickle.load(file)
            with open(class_aux_healthy_no_bg_val_path, 'rb') as file:
                class_aux_healthy_no_bg_val_data = pickle.load(file)
            with open(class_aux_healthy_no_bg_test_path, 'rb') as file:
                class_aux_healthy_no_bg_test_data = pickle.load(file)


            with open(class_aux_healthy_no_bg_cs_val_path, 'rb') as file:
                class_aux_healthy_no_bg_cs_val_data = pickle.load(file)
            with open(class_aux_healthy_no_bg__cs_test_path, 'rb') as file:
                class_aux_healthy_no_bg_cs_test_data = pickle.load(file)


            epoch_num_val_fold = class_positive_val_data['epoch'][:] + class_negative_val_data['epoch'][:]
            # -- Create validation dataset objects.
            len_val = min(len(class_positive_val_data['fhr']), len(class_negative_val_data['fhr']))
            fhr_val_fold = class_positive_val_data['fhr'][:] + class_negative_val_data['fhr'][:]
            up_val_fold = class_positive_val_data['up'][:] + class_negative_val_data['up'][:]
            epoch_num_val_fold = class_positive_val_data['epoch'][:] + class_negative_val_data['epoch'][:]
            weight_val_fold = class_positive_val_data['sample_weight'][:] + class_negative_val_data['sample_weight'][:]
            labels_val_fold = convert_onehot(class_positive_val_data['target'][:]) + convert_onehot(class_negative_val_data['target'][:])
            guid_val_fold = class_positive_val_data['guid'][:] + class_negative_val_data['guid'][:]

            # -- Create testing dataset objects.
            len_test = min(len(class_positive_test_data['fhr']), len(class_negative_test_data['fhr']))
            fhr_test_fold = class_positive_test_data['fhr'][:] + class_negative_test_data['fhr'][:]
            up_test_fold = class_positive_test_data['up'][:] + class_negative_test_data['up'][:]
            epoch_num_test_fold = class_positive_test_data['epoch'][:] + class_negative_test_data['epoch'][:]
            weight_test_fold = class_positive_test_data['sample_weight'][:] + class_negative_test_data['sample_weight'][:]
            labels_test_fold = convert_onehot(class_positive_test_data['target'][:]) + convert_onehot(class_negative_test_data['target'][:])
            guid_test_fold = class_positive_test_data['guid'][:] + class_negative_test_data['guid'][:]

            # -- Create auxiliary dataset objects.
            # fhr_aux_fold = class_aux_train_data['fhr'] + class_aux_test_data['fhr'] + class_negative_test_data['fhr']
            # up_aux_fold = class_aux_train_data['up'] + class_aux_test_data['up'] + class_negative_test_data['up']
            # epoch_num_aux_fold = class_aux_train_data['epoch'] + class_aux_test_data['epoch'] + class_negative_test_data['epoch']
            # weight_aux_fold = class_aux_train_data['sample_weight'] + class_aux_test_data['sample_weight'] + class_negative_test_data['sample_weight']
            # labels_aux_fold = convert_onehot(class_aux_train_data['target']) + convert_onehot(class_aux_test_data['target']) + convert_onehot(class_negative_test_data['target'])
            # guid_aux_fold = class_aux_train_data['guid'] + class_aux_test_data['guid'] + class_negative_test_data['guid']

            # -- Create auxiliary val healthy dataset
            fhr_val_aux_healthy_fold = class_positive_val_data['fhr'][:] + class_negative_val_data['fhr'][:] + class_aux_healthy_cs_val_data['fhr'][0:200] + class_aux_healthy_no_bg_val_data['fhr'][0:500] + class_aux_healthy_no_bg_cs_val_data['fhr'][0:500]
            up_val_aux_healthy_fold = class_positive_val_data['up'][:] + class_negative_val_data['up'][:] + class_aux_healthy_cs_val_data['up'][0:200] + class_aux_healthy_no_bg_val_data['up'][0:500] + class_aux_healthy_no_bg_cs_val_data['up'][0:500]
            epoch_num_val_aux_healthy_fold = class_positive_val_data['epoch'][:] + class_negative_val_data['epoch'][:] + class_aux_healthy_cs_val_data['epoch'][0:200] + class_aux_healthy_no_bg_val_data['epoch'][0:500] + class_aux_healthy_no_bg_cs_val_data['epoch'][0:500]
            weight_val_aux_healthy_fold = class_positive_val_data['sample_weight'][:] + class_negative_val_data['sample_weight'][:] + class_aux_healthy_cs_val_data['sample_weight'][0:520] + class_aux_healthy_no_bg_val_data['sample_weight'][0:500] + class_aux_healthy_no_bg_cs_val_data['sample_weight'][0:500]
            labels_val_aux_healthy_fold = convert_onehot(class_positive_val_data['target'][:]) + convert_onehot(class_negative_val_data['target'][:]) + convert_onehot(class_aux_healthy_cs_val_data['target'][0:200]) + convert_onehot(class_aux_healthy_no_bg_val_data['target'][0:500]) + convert_onehot(class_aux_healthy_no_bg_cs_val_data['target'][0:500])
            guid_val_aux_healthy_fold = class_positive_val_data['guid'][:] + class_negative_val_data['guid'][:] + class_aux_healthy_cs_val_data['guid'][0:200] + class_aux_healthy_no_bg_val_data['guid'][0:500] + class_aux_healthy_no_bg_cs_val_data['guid'][0:500]

            # -- Create auxiliary test healthy dataset
            fhr_test_aux_healthy_fold = class_positive_test_data['fhr'][:] + class_negative_test_data['fhr'][:] + class_aux_healthy_cs_test_data['fhr'][0:100] + class_aux_healthy_no_bg_test_data['fhr'][0:500] + class_aux_healthy_no_bg_cs_test_data['fhr'][0:500]
            up_test_aux_healthy_fold = class_positive_test_data['up'][:] + class_negative_test_data['up'][:] + class_aux_healthy_cs_test_data['up'][0:100] + class_aux_healthy_no_bg_test_data['up'][0:500] + class_aux_healthy_no_bg_cs_test_data['up'][0:500]
            epoch_num_test_aux_healthy_fold = class_positive_test_data['epoch'][:] + class_negative_test_data['epoch'][:] + class_aux_healthy_cs_test_data['epoch'][0:100] + class_aux_healthy_no_bg_test_data['epoch'][0:500] + class_aux_healthy_no_bg_cs_test_data['epoch'][0:500]
            weight_test_aux_healthy_fold = class_positive_test_data['sample_weight'][:] + class_negative_test_data['sample_weight'][:] + class_aux_healthy_cs_test_data['sample_weight'][0:100] + class_aux_healthy_no_bg_test_data['sample_weight'][0:500] + class_aux_healthy_no_bg_cs_test_data['sample_weight'][0:500]
            labels_test_aux_healthy_fold = convert_onehot(class_positive_test_data['target'][:]) + convert_onehot(class_negative_test_data['target'][:]) + convert_onehot(class_aux_healthy_cs_test_data['target'][0:100]) + convert_onehot(class_aux_healthy_no_bg_test_data['target'][0:500]) + convert_onehot(class_aux_healthy_no_bg_cs_test_data['target'][0:500])
            guid_test_aux_healthy_fold = class_positive_test_data['guid'][:] + class_negative_test_data['guid'][:] + class_aux_healthy_cs_test_data['guid'][0:100] + class_aux_healthy_no_bg_test_data['guid'][0:500] + class_aux_healthy_no_bg_cs_test_data['guid'][0:500]

            # Set domain boundaries using training data.
            self.min_domain_start = min(epoch_num_val_fold)
            self.max_domain_start = max(epoch_num_val_fold)

            dataset_test_fold = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_test_fold[:],
                up_list_ctg_dataset=up_test_fold[:],
                guids_list_ctg_dataset=guid_test_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_test_fold[:],
                sample_weights_list_ctg_dataset=weight_test_fold[:],
                fhr_labels=labels_test_fold[:],
                num_channel=2,
                min_domain_start=self.min_domain_start,
                max_domain_start=self.max_domain_start
            )

            dataset_val_fold = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_val_fold[:],
                up_list_ctg_dataset=up_val_fold[:],
                guids_list_ctg_dataset=guid_val_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_val_fold[:],
                sample_weights_list_ctg_dataset=weight_val_fold[:],
                fhr_labels=labels_val_fold[:],
                num_channel=2,
                min_domain_start=self.min_domain_start,
                max_domain_start=self.max_domain_start
            )


            dataset_val_aux_healthy_cs_fold  = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_val_aux_healthy_fold[:],
                up_list_ctg_dataset=up_val_aux_healthy_fold[:],
                guids_list_ctg_dataset=guid_val_aux_healthy_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_val_aux_healthy_fold[:],
                sample_weights_list_ctg_dataset=weight_val_aux_healthy_fold[:],
                fhr_labels=labels_val_aux_healthy_fold,
                num_channel=2,
                min_domain_start=self.min_domain_start,
                max_domain_start=self.max_domain_start
            )


            dataset_test_aux_healthy_cs_fold  = CTGDatasetWithLabel(
                fhr_list_ctg_dataset=fhr_test_aux_healthy_fold[:],
                up_list_ctg_dataset=up_test_aux_healthy_fold[:],
                guids_list_ctg_dataset=guid_test_aux_healthy_fold[:],
                epoch_nums_list_ctg_dataset=epoch_num_test_aux_healthy_fold[:],
                sample_weights_list_ctg_dataset=weight_test_aux_healthy_fold[:],
                fhr_labels=labels_test_aux_healthy_fold,
                num_channel=2,
                min_domain_start=self.min_domain_start,
                max_domain_start=self.max_domain_start
            )

            # dataset_test_aux_fold = CTGDatasetWithLabel(
            #     fhr_list_ctg_dataset=fhr_aux_fold[:],
            #     up_list_ctg_dataset=up_aux_fold[:],
            #     guids_list_ctg_dataset=guid_aux_fold[:],
            #     epoch_nums_list_ctg_dataset=epoch_num_aux_fold[:],
            #     sample_weights_list_ctg_dataset=weight_aux_fold[:],
            #     fhr_labels=labels_aux_fold[:],
            #     num_channel=2,
            #     min_domain_start=self.min_domain_start,
            #     max_domain_start=self.max_domain_start
            # )

            # Create DataLoaders.
            test_dataloader_fold = DataLoader(dataset_test_fold, batch_size=self.batch_size, shuffle=False, num_workers=0)
            val_dataloader_fold = DataLoader(dataset_val_fold, batch_size=self.batch_size, shuffle=False, num_workers=0)
            val_aux_dataloader_fold = DataLoader(dataset_val_aux_healthy_cs_fold, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_aux_dataloader_fold = DataLoader(dataset_test_aux_healthy_cs_fold, batch_size=self.batch_size, shuffle=False, num_workers=0)

            # -- Get the best checkpoint file.
            checkpoint_folder = os.path.join(checkpoints_base_folder, f"fold_{fold}", "model_checkpoints")
            checkpoint_files = []
            for filename in os.listdir(checkpoint_folder):
                # Check if the file ends with '.ckpt' and starts with 'best'
                if filename.endswith('.ckpt') and filename.startswith('best'):
                    # Construct the full path to the file.
                    checkpoint_files.append(os.path.join(checkpoint_folder, filename))
            if not checkpoint_files:
                print(f"No best checkpoint found in {checkpoint_folder} for fold {fold}; skipping analysis for this fold.")
                continue
            best_checkpoint = checkpoint_files[0]
            print(f"Fold {fold}: using best checkpoint {best_checkpoint}")

            # -- Create model and load checkpoint.
            self.create_model()
            self.load_pytorch_checkpoint(checkpoint_path=best_checkpoint)
            cuda_device = f"cuda:{self.cuda_devices[0]}"
            self.pytorch_model.to(cuda_device)
            self.pytorch_model.eval()

            # -- Run your testing/analysis pipeline.
            # Here you can call your test_classifier method or any other analysis functions.
            # For example, the following line runs test_classifier on the test dataloader:
            print(f"Running analysis for fold {fold} using test dataset...")
            # csv_path_val = self.test_classifier(test_dataloader_cls=val_dataloader_fold, tag='cls_val', checkpoint_for_test=best_checkpoint)
            # csv_path_test = self.test_classifier(test_dataloader_cls=test_dataloader_fold, tag='cls_test',
            #                                      checkpoint_for_test=best_checkpoint)
            csv_path_aux_val = self.test_classifier(test_dataloader_cls=val_aux_dataloader_fold, tag='cls_aux_val',
                                                 checkpoint_for_test=best_checkpoint)
            csv_path_aux_test = self.test_classifier(test_dataloader_cls=test_aux_dataloader_fold, tag='cls_aux_test',
                                                 checkpoint_for_test=best_checkpoint)


            threshold_optimal = self.test_classifier_with_strike_and_fpr(
                tag='v_ep_nocon',
                results_csv_path=csv_path_aux_val,
                strike=self.strike,
                target_fpr=self.target_fpr,
                pre_threshold_adjust=False,
                epoch_based=True,
                use_defined_threshold=False,
                consecutive_logic=False,
            )

            _ = self.test_classifier_with_strike_and_fpr(
                tag='t_ep_nocon_ax',
                results_csv_path=csv_path_aux_test,
                strike=self.strike,
                target_fpr=self.target_fpr,
                pre_threshold_adjust=False,
                use_defined_threshold=True,
                pre_defined_threshold=threshold_optimal,
                consecutive_logic=False,
            )

            #
            # threshold_optimal = self.test_classifier_with_strike_and_fpr(
            #     tag='v_ep_con',
            #     results_csv_path=csv_path_val,
            #     strike=self.strike,
            #     target_fpr=self.target_fpr,
            #     pre_threshold_adjust=False,
            #     epoch_based=True,
            #     use_defined_threshold=False,
            #     consecutive_logic=True,
            # )
            #
            # _ = self.test_classifier_with_strike_and_fpr(
            #     tag='t_ep_con_ax',
            #     results_csv_path=csv_path_aux_test,
            #     strike=self.strike,
            #     target_fpr=self.target_fpr,
            #     pre_threshold_adjust=False,
            #     use_defined_threshold=True,
            #     pre_defined_threshold=threshold_optimal,
            #     consecutive_logic=True,
            # )

            threshold_optimal = self.test_classifier_with_strike_and_fpr(
                tag='v_ep_con_cu',
                results_csv_path=csv_path_aux_val,
                strike=self.strike,
                target_fpr=self.target_fpr,
                pre_threshold_adjust=False,
                epoch_based=True,
                use_defined_threshold=False,
                consecutive_logic=True,
                cumulative_fill=True
            )

            _ = self.test_classifier_with_strike_and_fpr(
                tag='t_ep_con_cu_ax',
                results_csv_path=csv_path_aux_test,
                strike=self.strike,
                target_fpr=self.target_fpr,
                epoch_based=True,
                pre_threshold_adjust=False,
                use_defined_threshold=True,
                pre_defined_threshold=threshold_optimal,
                consecutive_logic=True,
                cumulative_fill=True
            )

        print("Fold analysis pipeline complete.")
        # aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='cls_test', tag="acidosis",
        #                 base_output_folder=os.path.join(self.output_base_dir, self.base_folder))
        # aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='cls_val', tag="acidosis",
        #                 base_output_folder=os.path.join(self.output_base_dir, self.base_folder))

        # aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='v_ep_nocon', tag="v_ep_nocon",
        #                 base_output_folder=os.path.join(self.output_base_dir, self.base_folder))
        # aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='t_ep_nocon_ax', tag="t_ep_nocon_ax",
        #                 base_output_folder=os.path.join(self.output_base_dir, self.base_folder))
        # aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='t_ep_con_ax', tag="t_ep_con_ax",
        #                 base_output_folder=os.path.join(self.output_base_dir, self.base_folder))
        aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='v_ep_con_cu', tag="v_ep_con_cu",
                        base_output_folder=os.path.join(self.output_base_dir, self.base_folder))


        aggregate_folds(os.path.join(self.output_base_dir, self.base_folder), test_dir='t_ep_con_cu_ax', tag="t_ep_con_cu_ax",
                        base_output_folder=os.path.join(self.output_base_dir, self.base_folder))


        # plot_raw_values(os.path.join(base_folder, f'{tag}'), test_dir='aux_test', tag="HIE")
        # run_bd_fold_aggregation_for_folds(os.path.join(base_folder, f'{tag}'))


    def test_classifier_with_strike_and_fpr(
        self,
        strike: int = 1,
        target_fpr: float = 0.3,
        pre_threshold_adjust: bool = False,
        tag: str = 'cls_tests_strike',
        results_csv_path: str = None,
        do_bd_analysis: bool = False,
        epoch_based: bool = False,
        pre_defined_threshold: float = 0.5,
        use_defined_threshold: bool = False,
        consecutive_logic: bool = False,
        cumulative_fill: bool = False,
    ):
        # ———————————————————————————————
        # 1) Run your standard test to get df_results
        # ———————————————————————————————
        output_dir_strike = os.path.join(self.test_results_dir, tag)
        os.makedirs(output_dir_strike, exist_ok=True)

        df_results = pd.read_csv(results_csv_path, sep=',')

        # ———————————————————————————————
        # 2) Find optimal threshold for your target FPR
        # ———————————————————————————————
        y_true = df_results['true_label'].values
        scores = df_results['prob_class_1'].values

        from Variational_AutoEncoder.utils.analyze_folds import find_threshold_after_strike
        if not use_defined_threshold:
            if pre_threshold_adjust:
                opt_thresh, fpr, tpr, threshs = find_threshold_for_fpr(y_true, scores, target_fpr)
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label='ROC curve')
                # find idx for marker
                idx = np.argmin(np.abs(fpr - target_fpr))
                plt.scatter(fpr[idx], tpr[idx], s=80, marker='o',
                            label=f'Thresh={opt_thresh:.3f}\nFPR={fpr[idx]:.2f}, TPR={tpr[idx]:.2f}')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve with Selected Operating Point")
                plt.legend(loc="lower right")
                roc_path = os.path.join(output_dir_strike, 'roc_curve_with_threshold.png')
                plt.savefig(roc_path, bbox_inches='tight')
                plt.close()
                print(f"ROC curve saved to {roc_path}")
            else:
                opt_thresh = find_threshold_after_strike(
                    df_results,
                    strike=strike,
                    target_fpr=target_fpr,
                    n_steps=400,
                    epoch_based=epoch_based,
                    output_dir=output_dir_strike,
                    consecutive=consecutive_logic,
                    cumulative_fill=cumulative_fill
                )
            print(f"Optimal threshold for FPR≈{target_fpr:.2f} is {opt_thresh:.3f}")
        else:
            opt_thresh = pre_defined_threshold
            print(f"Pre defined Optimal threshold for FPR≈{target_fpr:.2f} is {opt_thresh:.3f}")


        # ———————————————————————————————
        # 4) Apply strike‐based relabeling
        # ———————————————————————————————
        from Variational_AutoEncoder.utils.analyze_folds import apply_strike_labeling
        df_strike = apply_strike_labeling(df_results, strike=strike, threshold=opt_thresh, consecutive=consecutive_logic, cumulate=cumulative_fill)

        # ———————————————————————————————
        # 5) Save & run analytics
        # ———————————————————————————————
        out_dir = output_dir_strike
        csv_path = os.path.join(out_dir, f'classification_results_strike_{strike}.csv')
        df_strike.to_csv(csv_path, index=False)
        print(f"Strike‐labeled results saved to {csv_path}")

        # reuse your analysis functions on the new CSV
        analyze_and_plot_classification_metrics(csv_path, output_cls_test_dir=out_dir)
        analyze_class_stats_and_plot(csv_path, out_dir)

        if do_bd_analysis:
            bd_dir = os.path.join(out_dir, 'bd_analysis')
            os.makedirs(bd_dir, exist_ok=True)
            df_blood = pd.read_csv(self.blood_gas_file)
            run_bd_analysis(
                output_base_dir=bd_dir,
                df_classification=df_strike,
                df_blood_gas=df_blood
            )
        return opt_thresh

    def predict_dataset(self, dataloader=None, model=None, model_name='default', validation_checkpoint=None):
        self.set_cuda_devices([0])
        self.pytorch_ckp = validation_checkpoint
        self.create_model()
        self.seqvae_lightning_model = LightSeqVAEClassifier.load_from_checkpoint(
            validation_checkpoint,
            seqvae_model=self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            batch_size_train=self.batch_size_train,
            batch_size_test=self.batch_size_test,
            kl_beta=self.kld_beta_,
        )
        self.set_cuda_devices([0])
        all_predictions = []
        self.seqvae_lightning_model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                input_data = batch[0].to(self.seqvae_lightning_model.device)
                sample_mask = batch[4].to(self.seqvae_lightning_model.device).to(torch.bool)
                sample_mask = sample_mask[:, ::16]
                output_data = self.seqvae_lightning_model(input_data)
                logits = output_data.logits
                predictions_ = torch.softmax(logits, dim=-1)

                zero_column = torch.zeros(predictions_.size(0), predictions_.size(1), 1,
                                          device=predictions_.device, dtype=predictions_.dtype)
                new_logits = torch.cat([zero_column, predictions_], dim=2)
                new_logits[~sample_mask] = torch.tensor([1, 0, 0], device=new_logits.device, dtype=new_logits.dtype)

                all_predictions.append(new_logits.cpu().detach().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        return all_predictions

    def calculate_latent_stats(self, input_dataloader):
        if self.seqvae_ckp is not None:
            print(f"Loading checkpoint: {self.seqvae_ckp}")
            checkpoint = torch.load(self.seqvae_ckp)
            self.model.seqvae_lightning_model.load_state_dict(checkpoint['state_dict'])
        else:
            print("No checkpoint is defined")

        mean_sum = torch.zeros(self.latent_dim, device=self.device)
        square_sum = torch.zeros(self.latent_dim, device=self.device)
        num_samples = 0
        latent_stats_loader_tqdm = tqdm(enumerate(input_dataloader), total=len(input_dataloader))
        self.model.seqvae_lightning_model.eval()
        with torch.no_grad():
            for batch_idx, batched_data_latent_stats in latent_stats_loader_tqdm:
                data = batched_data_latent_stats[0].to(self.device)
                one_batch_size = data.shape[0]
                results_latent_stats = self.model.seqvae_lightning_model(data)

                # new approach -----------------------------------------------------------------------------------------
                results_latent_stats = results_latent_stats.z_latent.permute(1, 0, 2).contiguous().view(self.latent_dim, -1)
                num_samples += results_latent_stats.size(1)
                mean_sum += results_latent_stats.sum(dim=1)
                square_sum += results_latent_stats.pow(2).sum(dim=1)
                #-------------------------------------------------------------------------------------------------------
        latent_mean_tensor = mean_sum / num_samples
        latent_variance_tensor = (square_sum / num_samples) - (latent_mean_tensor ** 2)
        mean_latent_np = latent_mean_tensor
        var_latent_np = latent_variance_tensor

        latent_stats_path = os.path.join(self.output_base_dir, 'latent_stats.npy')
        np.save(latent_stats_path, (mean_latent_np, var_latent_np))
        self.latent_stats = (mean_latent_np, var_latent_np)


if __name__ == '__main__':
    # np.random.seed(42)
    # torch.manual_seed(42)
    # sklearn.utils.check_random_state(42)
    """
    aggregate_folds(root_folder, test_dir='cls_tests', tag="acidosis")
    aggregate_folds(root_folder, test_dir='aux_test', tag="HIE")
    plot_raw_values()
    run_bd_fold_aggregation_for_folds()
    """
    # from Variational_AutoEncoder.utils.analyze_folds import aggregate_folds, plot_raw_values
    # from Variational_AutoEncoder.utils.classification_analysis_BD import run_bd_fold_aggregation_for_folds
    # aggregate_folds(r"/data/deid/isilon/MS_model/Labmda Machine Second/2025-04-12--[08-02]--Acidosis_cs_nocs_Healhty_no_cs_frozen_vae_LSTM_ADTH---", test_dir='cls_test', tag="acidosis")
    # aggregate_folds(r"/data/deid/isilon/MS_model/Labmda Machine Second/2025-04-12--[08-02]--Acidosis_cs_nocs_Healhty_no_cs_frozen_vae_LSTM_ADTH---", test_dir='aux_test', tag="HIE")
    # run_bd_fold_aggregation_for_folds(r'/data/deid/isilon/MS_model/Labmda Machine Second/2025-04-12--[08-02]--Acidosis_cs_nocs_Healhty_no_cs_frozen_vae_LSTM_ADTH---')
    #
    start = time.time()
    parser = argparse.ArgumentParser(description="Train SeqVAE model.")
    parser.add_argument('--cuda_devices', required=True, default=0, type=int, nargs='+',
                        help="Cuda devices participating in training")
    parser.add_argument('--config', type=str, default='config_arguments.yaml', required=True,
                        help="Path to config file")
    parser.add_argument('--train_SeqVAE', type=int, default=-1, help="Run the training code")
    parser.add_argument('--test_SeqVAE',type=int, default=-1, help="Run the test code")
    args = parser.parse_args()
    config_file_path = os.path.dirname(
        os.path.realpath(__file__)) + '/seqvae_configs/' + args.config

    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    vae_train_dataset_file = os.path.normpath(config['dataset_config']['train_dataset_file'])
    vae_test_dataset_file = os.path.normpath(config['dataset_config']['test_dataset_file'])

    batch_size = config['general_config']['batch_size']['train']
    seqvae_testing_dataset_dir = config['seqvae_testing']['test_data_dir']
    input_channel_num = config['model_config']['VAE_model']['channel_num']

    batch_size_test = config['general_config']['batch_size']['test']
    with open(vae_train_dataset_file, 'rb') as file:
        vae_train_dict = pickle.load(file)
    with open(vae_test_dataset_file, 'rb') as file:
        vae_test_dict = pickle.load(file)

    graph_model = SeqVAEGraphModel(config_file_path=config_file_path, cuda_devices=args.cuda_devices, device_id=None)
    if args.train_SeqVAE > 0:
        print('done')
        # print('done')
        # graph_model.cls_num_list = cls_num_list
        # graph_model.create_model()
        # loss_history = graph_model.train_seqvae_model(train_loader_seqvae=train_dataloader,
        #                                               validation_loader_seqvae=test_dataloader)
        graph_model.train_and_test_folds()
        # graph_model.test_classifier(test_dataloader_cls=test_dataloader)
        end = time.time()
        elapsed_time = start - end
        hours, rem = divmod(elapsed_time, 3600)
        minutes, rem = divmod(rem, 60)
        seconds, milliseconds = divmod(rem, 1)
        milliseconds *= 1000
        print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(milliseconds):03}")
    if args.test_SeqVAE > 0:
        # aggregate_folds(r"/data/deid/isilon/MS_model/Labmda Machine Second/2025-05-07--[18-13]--1_strike_complete_tests_MMM", test_dir='test_acidosis_epoch_based_threshold_adjust_consecutive_strike_aux_cumulate_decisions', tag="FPR_015",
        #                 base_output_folder=r"/data/deid/isilon/MS_model/Labmda Machine Second/Poster_figs")
        graph_model.run_post_training_test(
            checkpoints_base_folder=r"/data/deid/isilon/MS_model/Labmda Machine Second/all_folds_last_12",
            tag="THIRD_try")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Create data loaders


