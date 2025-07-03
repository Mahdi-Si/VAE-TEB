import os
import sys
early_maestra_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
mimo_trainer_path = os.path.join(os.path.dirname(early_maestra_path), 'MIMO_Sequence_Trainer')
sys.path.insert(0, early_maestra_path)
sys.path.insert(0, mimo_trainer_path)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch.utils.data
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
import numpy as np
# todo: import the correct model
# from SeqVAE_model_2_Channel import VRNNGauss, ClassifierBlock, VRNNClassifier
import random
from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
import shutil
import logging
import math
import random
import h5py
from typing import Dict, List
from sklearn.model_selection import KFold


from early_maestra.vae.Variational_AutoEncoder.hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D

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
# Helper functions
#-----------------------------------------------------------------------------------------------------------------------
def find_flat_regions(signal, tolerance=1e-3):
    """
    Finds flat regions in the signal.

    Parameters:
    - signal: array-like, the input signal
    - tolerance: float, the threshold for considering points as flat

    Returns:
    - flat_regions: list of tuples, each tuple contains the start and end indices of a flat region
    """
    flat_regions = []
    start_idx = None

    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i-1]) <= tolerance:
            if start_idx is None:
                start_idx = i-1
        else:
            if start_idx is not None:
                flat_regions.append((start_idx, i-1))
                start_idx = None

    # Check if the last region is flat
    if start_idx is not None:
        flat_regions.append((start_idx, len(signal) - 1))

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
    import matplotlib.pyplot as plt
    
    # Ensure we have enough signals to plot
    n_signals = fhr_data.shape[0]
    if start_idx + 3 >= n_signals:
        raise ValueError(f"Not enough signals to plot. Have {n_signals}, need at least {start_idx + 4}")
    
    # Create time axis in minutes
    n_samples = fhr_data.shape[1]
    time_minutes = np.arange(n_samples) / (sampling_rate * 60)  # Convert to minutes
    
    # Create figure with 4 vertically stacked subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'FHR Signals - Starting from Index {start_idx}', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i in range(4):
        signal_idx = start_idx + i
        ax = axes[i]
        
        # Plot the FHR signal
        ax.plot(time_minutes, fhr_data[signal_idx, :], color=colors[i], linewidth=1.0)
        
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
    
    return fig


# ----------------------------------------------------------------------------------------------------------------------
# Dataset creation method
#-----------------------------------------------------------------------------------------------------------------------
def create_hdf5_dataset_from_records_list(hdf5_path=None, records_list=None, file_limit=-1,
                                          base_block_size=3200, save_name=None, min_domain_start=None,
                                          cs_label=None, bg_label=None, pre_defined_target=None, device=None, overlap_percentage=0.0):
    """
    Create HDF5 dataset from a list of records with optional overlapping segments.
    
    Parameters:
    -----------
    hdf5_path : str, optional
        Path for HDF5 output file
    records_list : list
        List of record file paths to process
    file_limit : int, default=-1
        Maximum number of files to process (-1 for all)
    base_block_size : int, default=3200
        Base length for segmentation (actual segment size will be base_block_size * 1.5)
    save_name : str, optional
        Name for saving the dataset
    min_domain_start : list, optional
        Minimum domain start values
    cs_label : str, optional
        CS label
    bg_label : str, optional
        Background label
    pre_defined_target : optional
        Predefined target values
    device : torch.device, optional
        Device for computations
    overl_samples : int, default=1920
        Overlap samples (deprecated parameter)
    overlap_percentage : float, default=0.0
        Percentage of overlap between consecutive segments (0.0-1.0).
        For example:
        - 0.0 = no overlap (default behavior)
        - 0.5 = 50% overlap (for 22-minute segments, shift by 11 minutes)
        - 0.75 = 75% overlap (for 22-minute segments, shift by 5.5 minutes)
    
    Returns:
    --------
    tuple
        (features_list_fhr, features_list_up, guid_list, epoch_num_list, tracing_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_model = KymatioPhaseScattering1D(J=11, Q=1, T=16, shape=4800, device=device, tukey_alpha=0.1)
    errors_list = []
    features_list_fhr = []
    features_list_up = []
    guid_list = []
    sample_weight_list = []
    target_list = []
    epoch_num_list = []
    counter = 0
    counter_rec = 0
    if file_limit > 0:
        records_list = records_list[:file_limit]
    for record in tqdm(records_list):
        counter_rec += 1
        logger.info(f'The count is  --------->  {counter_rec}')

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
        mimo_prepared, n_padded = mimo_adaptor.mimo.prepare_data(batch_size=1, do_evaluate=True, align_left=True,
                                                                    do_split=True,
                                                                    do_pad=True,
                                                                    do_reflect=True,
                                                                    base_length=base_block_size,
                                                                    do_equalize=True,
                                                                    do_merge=True,
                                                                    min_domain_start=[-43200, -43200],
                                                                    max_domain_start=[np.inf, np.inf],
                                                                    overlap_percentage=overlap_percentage)

        epoch_samples = mimo_prepared.block_input.shape[1]
        fhr = mimo_prepared.block_input[:, :, 1]
        up = mimo_prepared.block_input[:, :, 0]
        domain_starts = mimo_prepared.domain_start
        
        # Plot FHR signals to visualize overlap (optional - can be commented out)
        if fhr.shape[0] >= 4:  # Only plot if we have at least 4 segments
            try:
                plot_fhr_signals(fhr, domain_starts, start_idx=0, 
                               save_path=f"fhr_overlap_plot_{counter_rec}.png")
                logger.info(f"FHR plot saved for record {counter_rec}")
            except Exception as plot_error:
                logger.warning(f"Could not create FHR plot: {plot_error}")
        
        block_targets = mimo_prepared.block_target
        targets = np.argmax(block_targets, axis=2)
        sample_weights = np.repeat(mimo_prepared.sample_weights, repeats=16, axis=1)
        for i in range(fhr.shape[0]):
            record_file = os.path.split(record)
            record_name = os.path.splitext(record_file[1])
            if np.mean(sample_weights[i, :]) >= 0.90 and np.mean(targets[i, :] / max(targets[i, :])) >= 0.90:
                flat_regions_fhr = find_flat_regions(fhr[i, :], tolerance=1e-9)
                total_flat_samples_fhr = sum(end - start + 1 for start, end in flat_regions_fhr)
                flat_regions_up = find_flat_regions(up[i, :], tolerance=1e-9)
                total_flat_samples_up = sum(end - start + 1 for start, end in flat_regions_up)
                if max(abs(x - y) for x, y in flat_regions_fhr) > 480 or \
                        max(abs(x - y) for x, y in flat_regions_up) > 1200 or total_flat_samples_fhr > 1200 or \
                        total_flat_samples_up > 1200:
                    logger.info(f'Flat region detected for {record_name} in {domain_starts[i]}')
                    counter += 1
                else:
                    st_input = np.concatenate([fhr[i, :], up[i, :]], axis=0)
                    features_list_fhr.append(fhr[i, :])
                    features_list_up.append(up[i, :])
                    guid_list.append(record_name[0])
                    epoch_num_list.append(domain_starts[i])
                    sample_weight_list.append(sample_weights[i, :])
                    target_list.append(block_targets[i, :])
    
    if save_name and 'healthy' in save_name.lower():
        new_target_list = []
        for target_array in target_list:
            shifted_array = np.roll(target_array, -1, axis=1)
            new_target_list.append(shifted_array)
        target_list = new_target_list
    tracing_dict = {
        'fhr': features_list_fhr,
        'up': features_list_up,
        'guid': guid_list,
        'epoch': epoch_num_list,
        'sample_weight': sample_weight_list,
        'target': target_list

    }
    return features_list_fhr, features_list_up, guid_list, epoch_num_list, tracing_dict

if __name__ == "__main__":
    # base_folder = r'/data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/'
    # base_output_folder = r'/data1/fetal-heart-tracing/HDF5_Datasets/last_12_hours'
    
    # Example usage with 50% overlap (0.5 overlap_percentage)
    create_hdf5_dataset_from_records_list(
        records_list=[r"EarlyMaestra/early_maestra/vae/Variational_AutoEncoder/hdf5_dataset/A97E3DEA25464EC2BC5C4A6B4AB90147.mat"],
        overlap_percentage=0.5  # 50% overlap between consecutive segments
    )