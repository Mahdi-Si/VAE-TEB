import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List, Dict, Any

# Assuming the script is run from the same directory or the path is correctly set up
from hdf5_dataset import CombinedHDF5Dataset
from calculate_dataset_stats import calculate_and_save_dataset_stats, plot_dataset_histograms


def plot_random_samples(dataset: CombinedHDF5Dataset, output_dir: str, num_samples: int = 10):
    """
    Selects random samples from the dataset and generates detailed plots for visualization.

    Args:
        dataset: The initialized CombinedHDF5Dataset instance.
        output_dir: Directory to save the plot files.
        num_samples: The number of random samples to plot.
    """
    if len(dataset) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(dataset)}. Plotting all samples.")
        num_samples = len(dataset)

    random_indices = random.sample(range(len(dataset)), k=num_samples)
    print(f"\nPlotting {num_samples} random samples...")

    for i, sample_idx in enumerate(random_indices):
        print(f"  - Processing sample {i+1}/{num_samples} (index {sample_idx})...")
        sample = dataset[sample_idx]
        guid = sample.get('guid', f'sample_{sample_idx}')

        # Plot 1: Normalized FHR and UP signals
        _plot_fhr_up(sample, output_dir, guid, sample_idx)

        # Plot 2: Imshow for multi-channel data
        _plot_imshow(sample, output_dir, guid, sample_idx)

        # Plot 3: Channel-wise line plots for multi-channel data
        for field in ['fhr_st', 'fhr_ph', 'fhr_up_ph']:
            if field in sample:
                _plot_channel_subplots(sample, field, output_dir, guid, sample_idx)

def _plot_fhr_up(sample: Dict[str, Any], output_dir: str, guid: str, sample_idx: int):
    """Plots normalized FHR and UP signals on the same graph."""
    fig, ax = plt.subplots(figsize=(15, 5))
    
    if 'fhr' in sample:
        ax.plot(sample['fhr'].numpy(), label='FHR (Normalized)', color='blue')
    if 'up' in sample:
        ax.plot(sample['up'].numpy(), label='UP (Normalized)', color='red', alpha=0.7)
    
    ax.set_title(f'Normalized FHR & UP Signals\nSample Index: {sample_idx}, GUID: {guid}')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Normalized Value')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(output_dir, f'sample_{sample_idx}_fhr_up.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def _plot_imshow(sample: Dict[str, Any], output_dir: str, guid: str, sample_idx: int):
    """Creates imshow plots for multi-channel data in one figure."""
    fields_to_plot = [f for f in ['fhr_st', 'fhr_ph', 'fhr_up_ph'] if f in sample]
    if not fields_to_plot:
        return
        
    fig, axes = plt.subplots(len(fields_to_plot), 1, figsize=(12, 5 * len(fields_to_plot)), squeeze=False)
    fig.suptitle(f'Imshow of Normalized Multi-Channel Signals\nSample Index: {sample_idx}, GUID: {guid}', fontsize=16)

    for i, field in enumerate(fields_to_plot):
        ax = axes[i, 0]
        data = sample[field].numpy()
        im = ax.imshow(data, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'{field.upper()}')
        ax.set_xlabel('Time (decimated samples)')
        ax.set_ylabel('Channel Index')
        fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f'sample_{sample_idx}_imshow.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def _plot_channel_subplots(sample: Dict[str, Any], field: str, output_dir: str, guid: str, sample_idx: int):
    """Creates line plots for each channel of a given field."""
    data = sample[field].numpy()
    n_channels = data.shape[0]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True, squeeze=False)
    fig.suptitle(f'Normalized Channel Signals for {field.upper()}\nSample Index: {sample_idx}, GUID: {guid}', fontsize=16)

    for i in range(n_channels):
        ax = axes[i, 0]
        ax.plot(data[i, :])
        ax.set_ylabel(f'Ch {i}')
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1, 0].set_xlabel('Time (decimated samples)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f'sample_{sample_idx}_{field}_channels.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def run_full_test(hdf5_files: List[str], output_dir: str, trim_minutes: float = 2.0):
    """
    Executes a full test suite: calculates stats, plots histograms, and visualizes samples.

    Args:
        hdf5_files: List of paths to HDF5 dataset files.
        output_dir: Directory to save all generated files (stats, plots).
        trim_minutes: The duration in minutes to trim from the start and end of signals.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # 1. Calculate and save statistics with trimming
    print("\n--- Step 1: Calculating Dataset Statistics ---")
    stats_path = os.path.join(output_dir, 'test_stats_trimmed.hdf5')
    calculate_and_save_dataset_stats(
        hdf5_files,
        stats_path,
        trim_minutes=trim_minutes,
        metadata={'description': f'Test stats with {trim_minutes} min trim'}
    )
    print(f"Statistics saved to {stats_path}")

    # 2. Plot histograms of raw and normalized data
    print("\n--- Step 2: Plotting Histograms ---")
    # The plot_dataset_histograms function will save separate files for each field
    # into the specified output directory.
    plot_dataset_histograms(
        hdf5_files,
        output_dir,
        trim_minutes=trim_minutes,
        max_channels=None, # Plot all channels for a full overview
        max_samples=20000
    )
    print("Histograms saved.")

    # 3. Create Dataset and DataLoader with trimming
    print("\n--- Step 3: Initializing Dataset and DataLoader ---")
    dataset = CombinedHDF5Dataset(
        paths=hdf5_files,
        stats_path=stats_path,
        trim_minutes=trim_minutes,
        pin_memory=False # Pin memory is best for training, not needed for sample inspection
    )
    print(f"Dataset initialized with {len(dataset)} samples.")
    
    # Example of creating a DataLoader
    # data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # first_batch = next(iter(data_loader))
    # print(f"First batch 'fhr' shape: {first_batch['fhr'].shape}")
    
    # 4. Visualize random samples
    print("\n--- Step 4: Visualizing Random Samples ---")
    plot_random_samples(dataset, output_dir, num_samples=5)
    print("\nSample visualization complete.")
    
    print(f"\nFull test finished. Check the '{output_dir}' directory for all outputs.")


if __name__ == "__main__":
    # =============================================================================
    # Configuration
    # =============================================================================
    # Get the directory where this script is located to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Assumes the HDF5 file is in the same directory as this script.
    # You can add multiple files to the list.
    HDF5_FILE_PATHS = [
        os.path.join(script_dir, "hie_cs.hdf5")
    ]
    
    # All test outputs (stats file, plots) will be saved in an 'output_dir'
    # created in the same directory as this script.
    OUTPUT_DIRECTORY = os.path.join(script_dir, "output_dir")
    
    # Duration in minutes to trim from the start and end of each signal.
    TRIM_MINUTES = 2.0

    # =============================================================================
    
    # Check if the input file exists
    if not all(os.path.exists(p) for p in HDF5_FILE_PATHS):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: One or more HDF5 input files not found.")
        print(f"!!! Please check if the file exists or update the 'HDF5_FILE_PATHS' list in this script.")
        print("!!! Attempted path(s):", HDF5_FILE_PATHS)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        run_full_test(
            hdf5_files=HDF5_FILE_PATHS,
            output_dir=OUTPUT_DIRECTORY,
            trim_minutes=TRIM_MINUTES
        )
