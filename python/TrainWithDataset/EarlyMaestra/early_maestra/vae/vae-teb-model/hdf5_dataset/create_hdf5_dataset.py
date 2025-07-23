from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch.utils.data
import matplotlib
import os
import pickle
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib.pyplot as plt
import random

from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
import shutil
import logging
import math
import random
import h5py
from typing import Dict, List
from sklearn.model_selection import KFold

from hdf5_dataset import create_initial_hdf5, append_sample


from Variational_AutoEncoder.seqvae_teb.hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

matplotlib.use('Agg')
torch.backends.cudnn.enabled = False

random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)


# ----------------------------------------------------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------------------------------------------------

def find_flat_regions(signal, tolerance=1e-3, min_length=20):
    """
    Finds flat regions in the signal that are at least `min_length` samples long.

    Parameters:
      - signal: array-like, the input signal
      - tolerance: float, the threshold for considering consecutive points as "flat"
      - min_length: int, minimum number of consecutive "flat" samples to qualify

    Returns:
      - flat_regions: list of tuples, each tuple contains the start and end indices
                      of a flat region of length >= min_length
    """
    flat_regions = []
    start_idx = None

    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i-1]) <= tolerance:
            # we're in a flat segment
            if start_idx is None:
                start_idx = i-1
        else:
            # break in flatness
            if start_idx is not None:
                end_idx = i-1
                if (end_idx - start_idx + 1) >= min_length:
                    flat_regions.append((start_idx, end_idx))
                start_idx = None

    # handle case where signal ends in a flat region
    if start_idx is not None:
        end_idx = len(signal) - 1
        if (end_idx - start_idx + 1) >= min_length:
            flat_regions.append((start_idx, end_idx))

    return flat_regions


def detect_flat_region(signal, threshold=0.5, window=5):
    """
    Helper method
    Detects flat regions in the given signal.
    :param signal: List or numpy array containing the signal values.
    :param threshold: The threshold for the derivative to consider the signal as flat.
    :param window: The window size for smoothing the derivative.
    :return: A list of tuples indicating the start and end indices of flat regions.
    """
    # Calculate the derivative (rate of change) of the signal
    derivative = np.diff(signal, n=1)
    # Smooth the derivative using a uniform filter to reduce noise
    smooth_derivative = uniform_filter1d(np.abs(derivative), size=window)
    # Find indices where the absolute value of the smoothed derivative is below the threshold
    flat_indices = np.where(smooth_derivative < threshold)[0]
    # Group the flat indices into contiguous flat regions
    flat_regions = []
    if flat_indices.size > 0:
        start_idx = flat_indices[0]
        for i in range(1, len(flat_indices)):
            # If there is a gap between indices, it means the end of the current flat region
            if flat_indices[i] > flat_indices[i - 1] + 1:
                end_idx = flat_indices[i - 1]
                flat_regions.append((start_idx, end_idx))
                start_idx = flat_indices[i]
        # Add the last flat region
        flat_regions.append((start_idx, flat_indices[-1]))

    return flat_regions


def plot_fhr_signals(fhr_data, domain_starts, start_idx=0, sampling_rate=4, save_path=None):
    """
    Plot 4 consecutive FHR signals in vertically stacked subplots.

    Parameters:
    -----------
    fhr_data : np.ndarray
        FHR data with shape (n_segments, n_samples)
    domain_starts : list
        List of domain start values for each segment
    start_idx : int, default=0
        Starting index for plotting (will plot signals at start_idx, start_idx+1, start_idx+2, start_idx+3)
    sampling_rate : float, default=4
        Sampling rate in Hz for time axis conversion
    save_path : str, optional
        Path to save the plot. If None, plot will be displayed.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """

    # Ensure we have enough signals to plot
    n_signals = fhr_data.shape[0]
    if start_idx + 3 >= n_signals:
        raise ValueError(f"Not enough signals to plot. Have {n_signals}, need at least {start_idx + 4}")

    # Create time axis in minutes
    n_samples = fhr_data.shape[1]
    time_minutes = np.arange(n_samples) / (sampling_rate * 60)  # Convert to minutes

    # Create figure with 4 vertically stacked subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 6), sharex=True)
    fig.suptitle(f'FHR Signals - Starting from Index {start_idx}', fontsize=12)

    for i in range(4):
        signal_idx = start_idx + i
        ax = axes[i]

        # Plot the FHR signal
        ax.plot(time_minutes, fhr_data[signal_idx, :], linewidth=1.0)

        # Add labels and formatting
        ax.set_ylabel('FHR (bpm)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 200)  # Typical FHR range

        # Add domain start information
        domain_start_minutes = domain_starts[signal_idx] / 60  # Convert to minutes
        ax.set_title(f'Signal {signal_idx} - Domain Start: {domain_start_minutes:.1f} min',
                     fontsize=12, pad=10)

        # Add some statistics
        mean_fhr = np.mean(fhr_data[signal_idx, :])
        std_fhr = np.std(fhr_data[signal_idx, :])
        ax.text(0.02, 0.95, f'Mean: {mean_fhr:.1f} ± {std_fhr:.1f} bpm',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')

    # Set x-axis label only for bottom subplot
    axes[-1].set_xlabel('Time (minutes)', fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_channel_range(tensor: torch.Tensor,
                       start_chan: int,
                       end_chan: int,
                       save_path: str=None,
                       figsize=(10, 6)):
    """
    Plots channels [start_chan…end_chan] of a (m, n) tensor as an image.

    Args:
        tensor (torch.Tensor): shape (m, n), where m=channels, n=time steps
        start_chan (int): 1-based index of first channel to plot
        end_chan (int):   1-based index of last  channel to plot
        save_path (str):  path (including filename) to save the PNG
        figsize (tuple):  matplotlib figure size
    """
    # convert to NumPy and select (convert 1-based to 0-based indices)
    # ensure tensor is on CPU
    arr = tensor.detach().cpu().numpy()
    sel = arr[start_chan - 1 : end_chan, :]  # shape = (end_chan-start_chan+1, n)

    plt.figure(figsize=figsize)
    plt.imshow(sel,
               aspect='auto',
               interpolation='none',
               origin='lower')      # channels on the y-axis
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time step')
    plt.ylabel('Channel index')
    plt.title(f'Channels {start_chan}–{end_chan}')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def plot_stacked_channels(tensor: torch.Tensor,
                          start_chan: int = 1,
                          end_chan: int = None,
                          save_path: str = None,
                          figsize=(12, 2),
                          dpi=150):
    """
    Plots each channel in [start_chan…end_chan] of a (m, n) tensor as individual
    time-series subplots stacked vertically.

    Args:
        tensor (torch.Tensor): shape (m, n), where m=channels, n=time steps
        start_chan (int): 1-based index of first channel to plot (default: 1)
        end_chan (int):   1-based index of last channel to plot (default: m)
        save_path (str):  if provided, path (including filename) to save the figure
        figsize (tuple):  width & height of each subplot row, total height = figsize[1] * num_chans
        dpi (int):        resolution of saved figure
    """
    # Move to CPU/NumPy and handle channel bounds
    arr = tensor.detach().cpu().numpy()
    m, n = arr.shape
    if end_chan is None or end_chan > m:
        end_chan = m
    if start_chan < 1 or start_chan > end_chan:
        raise ValueError("start_chan must be ≥1 and ≤ end_chan")

    # Select channels (convert 1-based to 0-based)
    sel = arr[start_chan - 1:end_chan, :]  # shape = (num_chans, n)
    num_chans = sel.shape[0]

    # Create figure & axes
    fig, axes = plt.subplots(num_chans,
                             1,
                             figsize=(figsize[0], figsize[1] * num_chans),
                             sharex=True)

    # If only one channel, axes is not an array
    if num_chans == 1:
        axes = [axes]

    # Plot each channel
    for idx, ax in enumerate(axes, start=start_chan):
        ax.plot(sel[idx - start_chan, :])
        ax.set_ylabel(f"Ch {idx}")
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"Channels {start_chan}–{end_chan} (stacked)", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.close()

# ------------------------------------------------------------------------------------------
# folds creation method
# ------------------------------------------------------------------------------------------

def create_cv_splits(
    data: dict[str, list[str]],
    n_splits: int = 10,
    val_ratio: float = 0.1,
    random_state: int = 42
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """
    Perform stratified-by-subgroup 10-fold CV, with an inner train/validation split.

    Args:
        data: Mapping subgroup name → list of file paths.
        n_splits: Number of outer folds (here: 10).
        val_ratio: Fraction of the remaining after test to use as validation.
        random_state: Seed for reproducibility.

    Returns:
        folds: {
            'fold_1': {
                'train': { subgroup: [paths], … },
                'val':   { subgroup: [paths], … },
                'test':  { subgroup: [paths], … },
            },
            … up to 'fold_10'
        }
    """
    # prepare outer KFold per subgroup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits_per_group = {
        group: list(kf.split(file_list))
        for group, file_list in data.items()
    }

    folds: dict[str, dict] = {}
    for fold_idx in range(n_splits):
        fold_name = f"fold_{fold_idx+1}"
        fold_data = {'train': {}, 'val': {}, 'test': {}}

        for group, splits in splits_per_group.items():
            train_val_idx, test_idx = splits[fold_idx]

            # build test set for this group
            test_files = [data[group][i] for i in test_idx]

            # split train/val from the remaining indices
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                shuffle=True,
                random_state=random_state
            )
            train_files = [data[group][i] for i in train_idx]
            val_files   = [data[group][i] for i in val_idx]

            # store
            fold_data['train'][group] = train_files
            fold_data['val'][group]   = val_files
            fold_data['test'][group]  = test_files

        folds[fold_name] = fold_data

    return folds

# ----------------------------------------------------------------------------------------------------------------------
# Dataset creation method
#-----------------------------------------------------------------------------------------------------------------------
def create_hdf5_dataset_from_records_list(hdf5_path=None, records_list=None, file_limit=-1,
                                          base_block_size=3840, save_name=None, min_domain_start=None,
                                          cs_label=None, bg_label=None, pre_defined_target=None, device=None, overlap_percentage=0.5):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize scattering transform with optimal configuration for FHR analysis
    st_model = KymatioPhaseScattering1D(J=11, Q=4, T=16, shape=5760, device=device, tukey_alpha=None, max_order=1)
    
    # Get optimal coefficient selection for FHR analysis
    optimal_selection = st_model.get_optimal_coefficients_for_fhr(11, 4, 16)
    phase_mask = optimal_selection['recommendations']['use_phase_mask']
    cross_mask = optimal_selection['recommendations']['use_cross_mask']
    
    logger.info(f"Using optimal coefficient selection:")
    logger.info(f"  - FHR scattering: 45 coefficients (first order)")
    logger.info(f"  - FHR phase: {phase_mask.sum().item()} coefficients (95.1% reduction)")
    logger.info(f"  - FHR-UP cross-phase: {cross_mask.sum().item()} coefficients")
    logger.info(f"  - Total features: {optimal_selection['recommendations']['total_selected_features']}")
    errors_list = []
    counter_rec = 0
    if file_limit > 0:
        records_list = records_list[:file_limit]
    for record in tqdm(records_list):
        counter_rec += 1
        logger.info(f'The count is  --------->  {counter_rec}')
        try:
            mimo_adaptor = EarlyMaestraMimoAdaptor(do_transpose=True,
                                                   process_targets=True,
                                                   n_aux_labels=None,
                                                   signal_indices=range(0, 2),
                                                   n_input_chan=2,
                                                   labels=["HIE", "ACIDOSIS", "HEALTHY"],
                                                   # labels=["HIE", "ACIDOSIS"],
                                                   )
            mimo_adaptor.read_single_input(record, out_dec_factor=16, out_dec_factor_offset=0, target_is_onehot=True,
                                           dtype=np.float32)
            mimo_prepared, n_padded = mimo_adaptor.mimo.prepare_data(
                batch_size=1, do_evaluate=True, align_left=True,
                do_split=True,
                do_pad=True,
                do_reflect=True,
                base_length=base_block_size,
                do_equalize=True,
                do_merge=True,
                min_domain_start=[-44640, -44640],
                max_domain_start=[np.inf, np.inf],
                overlap_percentage=overlap_percentage,
                )

            epoch_samples = mimo_prepared.block_input.shape[1]
            fhr = mimo_prepared.block_input[:, :, 1]

            up = mimo_prepared.block_input[:, :, 0]
            domain_starts = mimo_prepared.domain_start
            block_targets = mimo_prepared.block_target
            if isinstance(block_targets, np.ndarray) and block_targets.ndim < 3:
                block_targets = []
            elif isinstance(block_targets, list):
                block_targets = []
            # sample_weights = np.repeat(mimo_prepared.sample_weights, repeats=16, axis=1)
            sample_weights = mimo_prepared.sample_weights

            st_input = torch.from_numpy(np.stack([fhr, up], axis=1)).float().to(device)
            
            # Compute all coefficients in one pass for efficiency
            st_results = st_model(x=st_input,
                                  compute_phase=True,
                                  compute_cross_phase=True,
                                  scattering_channel=0,
                                  phase_channels=[0, 1])
            
            # Extract and apply optimal coefficient selection
            fhr_st = st_results.get('scattering')  # Use all scattering coefficients (first order)
            fhr_st_phase_full = st_results.get('phase_corr')
            fhr_up_cc_phase_full = st_results.get('cross_phase_corr')
            
            # Apply optimal selection masks to reduce coefficients
            fhr_st_phase = fhr_st_phase_full[:, phase_mask, :] if fhr_st_phase_full is not None else None
            fhr_up_cc_phase = fhr_up_cc_phase_full[:, cross_mask, :] if fhr_up_cc_phase_full is not None else None
            # ===========================================
            # Testing the overlap plot:

            # if fhr.shape[0] >= 4:  # Only plot if we have at least 4 segments
            #     try:
            #         plot_fhr_signals(fhr, domain_starts, start_idx=0,
            #                          save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_{os.path.splitext(os.path.split(record)[1])}.png")
            #         logger.info(f"FHR plot saved for record {counter_rec}")
            #     except Exception as plot_error:
            #         logger.warning(f"Could not create FHR plot: {plot_error}")
            # =============================================

            for i in range(fhr.shape[0]):
                record_file = os.path.split(record)
                record_name = os.path.splitext(record_file[1])

                if np.mean(sample_weights[i, :]) < 0.90:
                    continue

                # --- Corrected flat region detection ---
                fhr_flat_regions = find_flat_regions(fhr[i, :], tolerance=1e-9)
                up_flat_regions = find_flat_regions(up[i, :], tolerance=1e-9)

                # Calculate lengths of flat regions
                fhr_flat_lengths = [end - start + 1 for start, end in fhr_flat_regions]
                up_flat_lengths = [end - start + 1 for start, end in up_flat_regions]

                max_flat_fhr_len = max(fhr_flat_lengths, default=0)
                max_flat_up_len = max(up_flat_lengths, default=0)
                total_flat_fhr_len = sum(fhr_flat_lengths)
                total_flat_up_len = sum(up_flat_lengths)

                if (max_flat_fhr_len > 480 or
                        max_flat_up_len > 1200 or
                        total_flat_fhr_len > 1200 or
                        total_flat_up_len > 1200):
                    logger.info(f'Flat region detected for {record_name} in {domain_starts[i]}')
                else:
                    # plotting for test =============================================================================================================================================
                    # plot_channel_range(fhr_st[i, :, :], 2, 23, save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_st_{os.path.splitext(os.path.split(record)[1])}.png")
                    # plot_stacked_channels(fhr_st[i, :, :], 2, 23, save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_st_chs_{os.path.splitext(os.path.split(record)[1])}.png")
                    #
                    # plot_channel_range(fhr_st_phase[i, :, :], 2, 23, save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_ph_{os.path.splitext(os.path.split(record)[1])}.png")
                    # plot_stacked_channels(fhr_st_phase[i, :, :], 2, 23, save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_ph_chs_{os.path.splitext(os.path.split(record)[1])}.png")
                    #
                    # plot_channel_range(fhr_up_cc_phase[i, :, :], 2, 23, save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_up_ph_{os.path.splitext(os.path.split(record)[1])}.png")
                    # plot_stacked_channels(fhr_up_cc_phase[i, :, :], 2, 23, save_path=f"/data/deid/isilon/MS_model/testing_code_plots/fhr_up_ph_chs_{os.path.splitext(os.path.split(record)[1])}.png")
                    # # plotting for test =============================================================================================================================================

                    append_sample(path=hdf5_path,
                                  fhr=fhr[i, :],
                                  up=up[i, :],
                                  fhr_st=fhr_st[i, :, :].detach().cpu().numpy(),
                                  fhr_ph=fhr_st_phase[i, :, :].detach().cpu().numpy(),
                                  fhr_up_ph=fhr_up_cc_phase[i, :, :].detach().cpu().numpy(),
                                  target=pre_defined_target * sample_weights[i, :],
                                  weight=sample_weights[i, :],
                                  guid=record_name[0],
                                  epoch=domain_starts[i],
                                  bg_label=bg_label,
                                  cs_label=cs_label)

        except Exception as e:
            errors_list.append(record)
            logger.error(e)
    return errors_list


def create_records(records_base_path_ = None, output_base_path_ = None):

    list_of_folders_dict = {
        1: "ACIDOSIS_NO_HIE_CS",
        2: "ACIDOSIS_NO_HIE_NoCS",
        3: "DEATH_lt_6_CS",
        4: "DEATH_lt_6_NoCS",
        5: "DISTANT_HIE_CS",
        6: "DISTANT_HIE_NoCS",
        7: "HEALTHY_NO_ACIDOSIS_CS",
        8: "HEALTHY_NO_ACIDOSIS_NoCS",
        9: "HEALTHY_NO_BG_CS",
        10: "HEALTHY_NO_BG_NoCS",
        11: "HIE_CS",
        12: "HIE_NoCS",
        13: "INTERVENTION_NO_ACIDOSIS_CS",
        14: "INTERVENTION_NO_ACIDOSIS_NoCS",
        15: "INTERVENTION_NO_BG_CS",
        16: "INTERVENTION_NO_BG_NoCS",
    }

    healthy_no_bg_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[10], 'EFMOut')
    healthy_no_bg_cs_path = os.path.join(records_base_path_, list_of_folders_dict[9], 'EFMOut')
    healthy_bg_cs_path = os.path.join(records_base_path_, list_of_folders_dict[7], 'EFMOut')
    healthy_bg_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[8], 'EFMOut')

    acidosis_cs_path = os.path.join(records_base_path_, list_of_folders_dict[1], 'EFMOut')
    acidosis_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[2], 'EFMOut')

    hie_cs_path =  os.path.join(records_base_path_, list_of_folders_dict[11], 'EFMOut')
    hie_no_cs_path = os.path.join(records_base_path_, list_of_folders_dict[12], 'EFMOut')

    healthy_no_bg_no_cs_files = [os.path.join(healthy_no_bg_no_cs_path, f) for f in os.listdir(healthy_no_bg_no_cs_path) if f.endswith('.mat')]
    healthy_no_bg_cs_files = [os.path.join(healthy_no_bg_cs_path, f) for f in os.listdir(healthy_no_bg_cs_path) if f.endswith('.mat')]
    healthy_bg_cs_files = [os.path.join(healthy_bg_cs_path, f) for f in os.listdir(healthy_bg_cs_path) if f.endswith('.mat')]
    healthy_bg_no_cs_files = [os.path.join(healthy_bg_no_cs_path, f) for f in os.listdir(healthy_bg_no_cs_path) if f.endswith('.mat')]

    acidosis_cs_files = [os.path.join(acidosis_cs_path, f) for f in os.listdir(acidosis_cs_path) if f.endswith('.mat')]
    acidosis_no_cs_files = [os.path.join(acidosis_no_cs_path, f) for f in os.listdir(acidosis_no_cs_path) if f.endswith('.mat')]

    hie_cs_files = [os.path.join(hie_cs_path, f) for f in os.listdir(hie_cs_path) if f.endswith('.mat')]
    hie_no_cs_files = [os.path.join(hie_no_cs_path, f) for f in os.listdir(hie_no_cs_path) if f.endswith('.mat')]

    n_healthy_no_bg_no_cs = len(healthy_no_bg_no_cs_files)
    n_healthy_no_bg_cs = len(healthy_no_bg_cs_files)
    n_healthy_bg_cs = len(healthy_bg_cs_files)
    n_healthy_bg_no_cs = len(healthy_bg_no_cs_files)
    n_healthy_total = n_healthy_no_bg_no_cs + n_healthy_no_bg_cs + n_healthy_bg_cs + n_healthy_bg_no_cs

    n_acidosis_cs = len(acidosis_cs_files)
    n_acidosis_no_cs = len(acidosis_no_cs_files)
    n_acidosis_total = n_acidosis_cs + n_acidosis_no_cs

    n_hie_cs = len(hie_cs_files)
    n_hie_no_cs = len(hie_no_cs_files)
    n_hie_total = n_hie_cs + n_hie_no_cs

    n_unhealthy_total = n_acidosis_total + n_hie_total

    counts_healthy = {
        "NoBG_NoCS": n_healthy_no_bg_no_cs,
        "NoBG_CS": n_healthy_no_bg_cs,
        "BG_CS": n_healthy_bg_cs,
        "BG_NoCS": n_healthy_bg_no_cs,
    }

    total_healthy = sum(counts_healthy.values())
    target_healthy = {k: int(round((v / total_healthy) * n_unhealthy_total)) for k, v in counts_healthy.items()}

    diff = n_unhealthy_total - sum(target_healthy.values())
    if diff:
        largest = max(counts_healthy, key=counts_healthy.get)
        target_healthy[largest] = target_healthy[largest] + diff

    random.shuffle(healthy_bg_cs_files)
    healthy_bg_cs_files_vae_train = healthy_bg_cs_files[:int(n_healthy_bg_cs*0.9)]
    healthy_bg_cs_files_vae_test = healthy_bg_cs_files[int(n_healthy_bg_cs*0.9):]

    random.shuffle(healthy_bg_no_cs_files)
    healthy_bg_no_cs_files_vae_train = healthy_bg_no_cs_files[:int(n_healthy_bg_no_cs*0.9)]
    healthy_bg_no_cs_files_vae_test = healthy_bg_no_cs_files[int(n_healthy_bg_no_cs*0.9):]

    healthy_no_bg_no_cs_files_subsampled = random.sample(healthy_no_bg_no_cs_files, target_healthy['NoBG_NoCS'])
    healthy_no_bg_cs_files_subsampled = random.sample(healthy_no_bg_cs_files, target_healthy['NoBG_CS'])
    healthy_bg_cs_files_subsampled = random.sample(healthy_bg_cs_files_vae_test, target_healthy['BG_CS'])
    healthy_bg_no_cs_files_subsampled = random.sample(healthy_bg_no_cs_files_vae_test, target_healthy['BG_NoCS'])

    cross_validation_records = {
        'healthy_no_bg_no_cs': healthy_no_bg_no_cs_files_subsampled,
        'healthy_no_bg_cs':    healthy_no_bg_cs_files_subsampled,
        'healthy_bg_cs':       healthy_bg_cs_files_subsampled,
        'healthy_bg_no_cs':    healthy_bg_no_cs_files_subsampled,
        'acidosis_cs':         acidosis_cs_files,
        'acidosis_no_cs':      acidosis_no_cs_files,
        'hie_cs':              hie_cs_files,
        'hie_no_cs':           hie_no_cs_files,
    }
    classification_folds = create_cv_splits(cross_validation_records, n_splits=10, val_ratio=0.1, random_state=42)
    classification_dataset_records_path = os.path.join(output_base_path_, "classification_dataset_records.pickle")
    with open(classification_dataset_records_path, 'wb') as f:
        pickle.dump(classification_folds, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---------------------------
    # Vae Train and Test
    # ---------------------------
    pre_train_path = os.path.join(output_base_path_, "pre_training_dataset")
    os.makedirs(pre_train_path, exist_ok=True)
    pre_training_dataset = os.path.join(pre_train_path, "train_dataset_cs.hdf5")
    # Calculate optimal number of channels based on coefficient selection
    total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
    create_initial_hdf5(path=pre_training_dataset, len_signal=5760, n_channels=total_channels, len_sequence=360)
    create_hdf5_dataset_from_records_list(records_list=healthy_bg_cs_files_vae_train,
                                          hdf5_path=pre_training_dataset,
                                          cs_label=True,
                                          bg_label=True,
                                          pre_defined_target=1)

    pre_training_dataset = os.path.join(pre_train_path, "train_dataset_no_cs.hdf5")
    # Calculate optimal number of channels based on coefficient selection
    total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
    create_initial_hdf5(path=pre_training_dataset, len_signal=5760, n_channels=total_channels, len_sequence=360)
    create_hdf5_dataset_from_records_list(records_list=healthy_bg_no_cs_files_vae_train,
                                          hdf5_path=pre_training_dataset,
                                          cs_label=False,
                                          bg_label=True,
                                          pre_defined_target=1)


    pre_training_dataset = os.path.join(pre_train_path, "test_dataset_cs.hdf5")
    # Calculate optimal number of channels based on coefficient selection
    total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
    create_initial_hdf5(path=pre_training_dataset, len_signal=5760, n_channels=total_channels, len_sequence=360)
    create_hdf5_dataset_from_records_list(records_list=healthy_bg_cs_files_vae_test,
                                          hdf5_path=pre_training_dataset,
                                          cs_label=True,
                                          bg_label=True,
                                          pre_defined_target=1)

    pre_training_dataset = os.path.join(pre_train_path, "test_dataset_no_cs.hdf5")
    # Calculate optimal number of channels based on coefficient selection
    total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
    create_initial_hdf5(path=pre_training_dataset, len_signal=5760, n_channels=total_channels, len_sequence=360)
    create_hdf5_dataset_from_records_list(records_list=healthy_bg_no_cs_files_vae_test,
                                          hdf5_path=pre_training_dataset,
                                          cs_label=False,
                                          bg_label=True,
                                          pre_defined_target=1)
    # ---------------------------
    # Classifications
    # ---------------------------
    k_fold_cross_validation_path = os.path.join(output_base_path_, "k_fold_cross_validation_dataset")
    os.makedirs(k_fold_cross_validation_path, exist_ok=True)
    for fold in classification_folds:
        print('done')
        fold_path = os.path.join(k_fold_cross_validation_path, str(fold))
        os.makedirs(fold_path, exist_ok=True)
        fold_datasets = classification_folds.get(fold)
        for dataset_partition in fold_datasets:
            dataset_partition_path = os.path.join(fold_path, str(dataset_partition))
            os.makedirs(dataset_partition_path, exist_ok=True)
            sub_groups_list = fold_datasets.get(dataset_partition)

            selected_sub_group = "healthy_no_bg_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=False,
                                                  bg_label=False,
                                                  pre_defined_target=1)

            selected_sub_group = "healthy_no_bg_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=True,
                                                  bg_label=False,
                                                  pre_defined_target=1)

            selected_sub_group = "healthy_bg_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=True,
                                                  bg_label=True,
                                                  pre_defined_target=1)

            selected_sub_group = "healthy_bg_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=False,
                                                  bg_label=True,
                                                  pre_defined_target=1)

            selected_sub_group = "acidosis_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=True,
                                                  bg_label=True,
                                                  pre_defined_target=2)

            selected_sub_group = "acidosis_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=False,
                                                  bg_label=True,
                                                  pre_defined_target=2)

            selected_sub_group = "hie_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=True,
                                                  bg_label=True,
                                                  pre_defined_target=3)

            selected_sub_group = "hie_no_cs"
            sub_group_path = os.path.join(dataset_partition_path, f"{selected_sub_group}.hdf5")
            sub_group_records_list = sub_groups_list.get(selected_sub_group)
            # Use optimal number of channels for all dataset creation
            total_channels = 174  # 44 phase + 130 cross-phase coefficients from optimal selection
            create_initial_hdf5(path=sub_group_path, len_signal=5760, n_channels=total_channels, len_sequence=360)
            create_hdf5_dataset_from_records_list(records_list=sub_group_records_list,
                                                  hdf5_path=sub_group_path,
                                                  cs_label=False,
                                                  bg_label=True,
                                                  pre_defined_target=3)


if __name__ == "__main__":
    # base_folder = r'/data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/'
    # base_output_folder = r'/data1/fetal-heart-tracing/HDF5_Datasets/last_12_hours'
    # create_records(records_base_path_=base_folder, output_base_path_=base_output_folder)

    hdf_file = "test_dataset_no_cs.hdf5"
    
    try:
        print("Opening HDF5 dataset...")
        with h5py.File(hdf_file, "r") as dataset:
            
            # Print dataset structure
            print("\n" + "="*60)
            print("DATASET STRUCTURE")
            print("="*60)
            print("Available fields in dataset:")
            for key in dataset.keys():
                shape = dataset[key].shape
                dtype = dataset[key].dtype
                print(f"  {key}: shape={shape}, dtype={dtype}")
            
            # Check if dataset has samples
            if len(dataset.keys()) == 0:
                print("Dataset is empty!")
            else:
                # Get the number of samples (assuming all fields have same first dimension)
                first_key = list(dataset.keys())[0]
                n_samples = dataset[first_key].shape[0]
                print(f"\nTotal number of samples: {n_samples}")
                
                if n_samples > 0:
                    # Example: Get the first sample (index 0)
                    sample_idx = 100
                    print(f"\n" + "="*60)
                    print(f"SAMPLE {sample_idx} DETAILS")
                    print("="*60)
                    
                    sample_data = {}
                    for field in dataset.keys():
                        sample_data[field] = dataset[field][sample_idx]
                    
                    # Display sample information
                    for field, data in sample_data.items():
                        if isinstance(data, np.ndarray):
                            print(f"{field}:")
                            print(f"  Shape: {data.shape}")
                            print(f"  Dtype: {data.dtype}")
                            if data.size > 0:
                                if data.ndim == 1:
                                    # 1D array - show basic stats
                                    print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
                                    print(f"  Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
                                    print(f"  First 5 values: {data[:5]}")
                                elif data.ndim == 2:
                                    # 2D array - show shape and channel stats
                                    print(f"  Channels: {data.shape[0]}, Sequence length: {data.shape[1]}")
                                    print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
                                    print(f"  Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
                                    # Show stats for first few channels
                                    for i in range(min(3, data.shape[0])):
                                        ch_data = data[i, :]
                                        print(f"    Ch {i}: mean={np.mean(ch_data):.4f}, std={np.std(ch_data):.4f}")
                                    if data.shape[0] > 3:
                                        print(f"    ... ({data.shape[0] - 3} more channels)")
                        else:
                            # Scalar values
                            print(f"{field}: {data}")
                        print()
                    
                    # Example: How to use the sample data
                    print("="*60)
                    print("EXAMPLE USAGE")  
                    print("="*60)
                    print("\n# Example code for using the loaded sample:")
                    print("# Access specific fields:")
                    if 'fhr' in sample_data:
                        print(f"# fhr_signal = sample_data['fhr']  # Shape: {sample_data['fhr'].shape}")
                    if 'up' in sample_data:
                        print(f"# up_signal = sample_data['up']    # Shape: {sample_data['up'].shape}")
                    if 'fhr_st' in sample_data:
                        print(f"# fhr_st = sample_data['fhr_st']   # Shape: {sample_data['fhr_st'].shape}")
                    if 'fhr_ph' in sample_data:
                        print(f"# fhr_ph = sample_data['fhr_ph']   # Shape: {sample_data['fhr_ph'].shape}")
                    if 'fhr_up_ph' in sample_data:
                        print(f"# fhr_up_ph = sample_data['fhr_up_ph'] # Shape: {sample_data['fhr_up_ph'].shape}")
                    
                    print("\n# Convert to torch tensors for deep learning:")
                    print("# import torch")
                    if 'fhr' in sample_data and 'up' in sample_data:
                        print("# fhr_tensor = torch.from_numpy(sample_data['fhr']).float()")
                        print("# up_tensor = torch.from_numpy(sample_data['up']).float()")
                    if 'fhr_st' in sample_data:
                        print("# fhr_st_tensor = torch.from_numpy(sample_data['fhr_st']).float()")
                    
                    print("\n# Example batch processing (multiple samples):")
                    batch_size = min(4, n_samples)
                    print(f"# batch_fhr = dataset['fhr'][:{batch_size}]  # Shape: {dataset['fhr'][:batch_size].shape}")
                    if 'fhr_st' in dataset:
                        print(f"# batch_fhr_st = dataset['fhr_st'][:{batch_size}]  # Shape: {dataset['fhr_st'][:batch_size].shape}")
                    
                    # Show another sample if available
                    if n_samples > 1:
                        sample_idx = min(1, n_samples - 1)
                        print(f"\n" + "="*60)
                        print(f"QUICK VIEW: SAMPLE {sample_idx}")
                        print("="*60)
                        for field in ['guid', 'epoch', 'target', 'cs_label', 'bg_label']:
                            if field in dataset:
                                value = dataset[field][sample_idx]
                                print(f"{field}: {value}")
                else:
                    print("\nDataset contains no samples!")
                    
        print(f"\nSuccessfully examined HDF5 dataset: {hdf_file}")
        
    except FileNotFoundError:
        print(f"HDF5 file not found: {hdf_file}")
        print("Please make sure the file exists or create it first using the dataset creation functions.")
    except Exception as e:
        print(f"Error reading HDF5 dataset: {e}")
    
    print('\nDone!')