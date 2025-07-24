import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import torch
from typing import List, Optional

from hdf5_dataset import CombinedHDF5Dataset, normalize_tensor_data
from calculate_dataset_stats import DatasetStatsCalculator

matplotlib.use('Agg')


def plot_random_dataset_samples(
    hdf5_file_path: str,
    stats_file_path: str,
    output_dir: str = "./dataset_sample_plots",
    n_samples: int = 10,
    trim_minutes: Optional[float] = None,
    seed: int = 42
):
    """
    Plot randomly selected samples from the HDF5 dataset showing:
    - FHR and UP signals (line plots)
    - Normalized fhr_st, fhr_ph, fhr_up_ph (imshow plots)
    
    Uses professional color scheme from the plotting callback.
    
    Args:
        hdf5_file_path: Path to the HDF5 dataset file
        stats_file_path: Path to the statistics file for normalization
        output_dir: Directory to save the plots
        n_samples: Number of random samples to plot (default: 10)
        trim_minutes: Optional trimming time in minutes
        seed: Random seed for reproducible sample selection
    """
    # Set random seed for reproducible results
    random.seed(seed)
    np.random.seed(seed)
    
    # Professional scientific paper color palette (from plotting callback)
    colors = {
        'fhr': "#055C9A",           # Deep blue-gray
        'up': "#0DD8A2",            # Sage green
        'gt': '#456882',            # Medium blue-gray
        'recon': '#BB3E00',         # Deep orange-red
        'uncertainty': '#F7AD45',    # Golden yellow
        'samples': "#4BD605",       # Muted green-gray
        'background': '#F9F3EF'     # Warm off-white
    }
    
    # Set professional scientific paper styling
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'axes.linewidth': 0.7,
        'axes.edgecolor': "#9E9D9D",
        'axes.facecolor': colors['background'],
        'grid.color': "#838383",
        'grid.linewidth': 0.4,
        'grid.alpha': 0.6,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#A2B9A7',
        'legend.facecolor': colors['background'],
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {hdf5_file_path}")
    dataset = CombinedHDF5Dataset(
        paths=hdf5_file_path,
        stats_path=stats_file_path,
        trim_minutes=trim_minutes,
        cache_size=0  # Disable caching for memory efficiency
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Randomly select sample indices
    total_samples = len(dataset)
    if n_samples > total_samples:
        print(f"Warning: Requested {n_samples} samples but dataset only contains {total_samples}. Using all samples.")
        n_samples = total_samples
    
    sample_indices = random.sample(range(total_samples), n_samples)
    print(f"Selected sample indices: {sample_indices}")
    
    # Setup time axis for signal plots (assuming 4 Hz sampling rate)
    Fs = 4
    
    for i, sample_idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{n_samples} (index {sample_idx})")
        
        # Load sample
        sample = dataset[sample_idx]
        
        # Extract data
        fhr_signal = sample.fhr.cpu().numpy()  # Raw FHR signal
        up_signal = sample.up.cpu().numpy()    # Raw UP signal
        fhr_st_norm = sample.fhr_st.cpu().numpy()    # Normalized scattering transform
        fhr_ph_norm = sample.fhr_ph.cpu().numpy()    # Normalized phase harmonics
        fhr_up_ph_norm = sample.fhr_up_ph.cpu().numpy()  # Normalized cross-phase
        
        # Create time axis for signals
        N = len(fhr_signal)
        t_signal = np.arange(0, N) / Fs
        
        # Create figure with 5 subplots (2 signals + 3 imshow)
        fig, axes = plt.subplots(5, 1, figsize=(20, 20), constrained_layout=True)
        
        # Configure scientific paper grid style for all subplots
        for ax in axes[:2]:  # Only for signal plots
            ax.grid(True, linestyle='-', alpha=0.4, linewidth=0.4, color='#D2C1B6')
            ax.grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax.minorticks_on()
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#A2B9A7')
            ax.spines['bottom'].set_color('#A2B9A7')
            ax.spines['left'].set_linewidth(0.7)
            ax.spines['bottom'].set_linewidth(0.7)
        
        # Plot 1: FHR Signal
        axes[0].plot(t_signal, fhr_signal, linewidth=1.2, color=colors['fhr'], alpha=0.85)
        axes[0].set_ylabel('FHR (bpm)', fontweight='normal')
        axes[0].set_title('Fetal Heart Rate (FHR) Signal', fontweight='normal', pad=12)
        axes[0].autoscale(enable=True, axis='x', tight=True)
        
        # Plot 2: UP Signal
        axes[1].plot(t_signal, up_signal, linewidth=1.2, color=colors['up'], alpha=0.85)
        axes[1].set_ylabel('UP Amplitude', fontweight='normal')
        axes[1].set_xlabel('Time (s)', fontweight='normal')
        axes[1].set_title('Uterine Pressure (UP) Signal', fontweight='normal', pad=12)
        axes[1].autoscale(enable=True, axis='x', tight=True)
        
        # Plot 3: Normalized FHR Scattering Transform
        im1 = axes[2].imshow(fhr_st_norm, aspect='auto', cmap='viridis', origin='lower', interpolation='bilinear')
        axes[2].set_ylabel('Scattering Channels', fontweight='normal')
        axes[2].set_title('Normalized FHR Scattering Transform', fontweight='normal', pad=12)
        cbar1 = plt.colorbar(im1, ax=axes[2], shrink=0.8)
        cbar1.ax.tick_params(labelsize=10, colors='#666666')
        cbar1.set_label('Normalized Amplitude', fontweight='normal', fontsize=11, color='#666666')
        cbar1.outline.set_color('#A2B9A7')
        cbar1.outline.set_linewidth(0.7)
        
        # Plot 4: Normalized FHR Phase Harmonics
        im2 = axes[3].imshow(fhr_ph_norm, aspect='auto', cmap='plasma', origin='lower', interpolation='bilinear')
        axes[3].set_ylabel('Phase Channels', fontweight='normal')
        axes[3].set_title('Normalized FHR Phase Harmonics', fontweight='normal', pad=12)
        cbar2 = plt.colorbar(im2, ax=axes[3], shrink=0.8)
        cbar2.ax.tick_params(labelsize=10, colors='#666666')
        cbar2.set_label('Normalized Phase', fontweight='normal', fontsize=11, color='#666666')
        cbar2.outline.set_color('#A2B9A7')
        cbar2.outline.set_linewidth(0.7)
        
        # Plot 5: Normalized FHR-UP Cross-Phase
        im3 = axes[4].imshow(fhr_up_ph_norm, aspect='auto', cmap='inferno', origin='lower', interpolation='bilinear')
        axes[4].set_ylabel('Cross-Phase Channels', fontweight='normal')
        axes[4].set_xlabel('Time Steps', fontweight='normal')
        axes[4].set_title('Normalized FHR-UP Cross-Phase', fontweight='normal', pad=12)
        cbar3 = plt.colorbar(im3, ax=axes[4], shrink=0.8)
        cbar3.ax.tick_params(labelsize=10, colors='#666666')
        cbar3.set_label('Normalized Cross-Phase', fontweight='normal', fontsize=11, color='#666666')
        cbar3.outline.set_color('#A2B9A7')
        cbar3.outline.set_linewidth(0.7)
        
        # Set overall title
        sample_guid = sample.guid if hasattr(sample, 'guid') else f"Sample_{sample_idx}"
        fig.suptitle(f'Dataset Sample Analysis — {sample_guid}', 
                    fontsize=14, fontweight='normal', y=0.97, color='#456882')
        
        # Save plot
        save_path = os.path.join(output_dir, f'sample_{i+1:02d}_idx_{sample_idx}.pdf')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"Saved plot to {save_path}")
    
    print(f"\nCompleted plotting {n_samples} random samples")
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    # Configuration
    hdf5_file = r"C:\Users\mahdi\Desktop\teb_vae_model\hdf5_dataset\train_dataset_cs.hdf5"
    stats_file = r"C:\Users\mahdi\Desktop\teb_vae_model\output\stats.hdf5"
    output_directory = "./dataset_sample_plots"
    
    # Plot 10 random samples
    plot_random_dataset_samples(
        hdf5_file_path=hdf5_file,
        stats_file_path=stats_file,
        output_dir=output_directory,
        n_samples=10,
        trim_minutes=2,  # Match the trimming used in stats calculation
        seed=42  # For reproducible results
    )
    
    print("\nScript completed successfully!")