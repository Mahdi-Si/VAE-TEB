
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_model_analysis(
    output_dir: str,
    raw_fhr: np.ndarray,
    raw_up: np.ndarray,
    fhr_st: np.ndarray,
    fhr_ph: np.ndarray,
    fhr_up_ph: np.ndarray,
    latent_z: np.ndarray,
    reconstructed_fhr_mu: np.ndarray,
    reconstructed_fhr_logvar: np.ndarray,
    kld_tensor: np.ndarray,
    kld_mean_over_channels: np.ndarray,
    batch_idx: int = 0
):
    """
    Generates and saves a comprehensive plot for model analysis.

    Args:
        output_dir (str): Directory to save the plot.
        raw_fhr (np.ndarray): Raw FHR signal. Shape: (N,).
        raw_up (np.ndarray): Raw UP signal. Shape: (N,).
        fhr_st (np.ndarray): Scattering transform of FHR. Shape: (C, L).
        fhr_ph (np.ndarray): Phase harmonics of FHR. Shape: (C, L).
        fhr_up_ph (np.ndarray): Phase harmonics of UP. Shape: (C, L).
        latent_z (np.ndarray): Latent space representation z. Shape: (D, L).
        reconstructed_fhr_mu (np.ndarray): Mean of the reconstructed FHR. Shape: (N,).
        reconstructed_fhr_logvar (np.ndarray): Log variance of the reconstructed FHR. Shape: (N,).
        kld_tensor (np.ndarray): KLD tensor. Shape: (D, L).
        kld_mean_over_channels (np.ndarray): KLD mean over channels. Shape: (L,).
        batch_idx (int): Index of the sample in the batch for file naming.
    """
    
    # Professional scientific paper color palette
    colors = {
        'fhr': "#055C9A",
        'up': "#0DD8A2",
        'gt': '#456882',
        'recon': '#BB3E00',
        'uncertainty': '#F7AD45',
        'kld': '#D95319',
        'background': '#F9F3EF'
    }

    plt.style.use('default')
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

    fig, ax = plt.subplots(4, 2, figsize=(18, 20), constrained_layout=True)

    # Common settings for subplots
    for axis in ax.flatten():
        axis.grid(True, linestyle='-', alpha=0.4, linewidth=0.4, color='#D2C1B6')
        axis.grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
        axis.minorticks_on()
        axis.set_axisbelow(True)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_color('#A2B9A7')
        axis.spines['bottom'].set_color('#A2B9A7')
        axis.spines['left'].set_linewidth(0.7)
        axis.spines['bottom'].set_linewidth(0.7)

    # 1. Raw FHR and UP
    t_raw = np.arange(raw_fhr.shape[0]) / 4.0  # Assuming 4Hz
    ax[0, 0].plot(t_raw, raw_fhr, color=colors['fhr'], label='Raw FHR', linewidth=1.2)
    ax[0, 0].plot(t_raw, raw_up, color=colors['up'], label='Raw UP', linewidth=1.2)
    ax[0, 0].set_title('Raw Input Signals')
    ax[0, 0].set_xlabel('Time (s)')
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].legend()
    ax[0, 0].autoscale(enable=True, axis='x', tight=True)

    # 2. FHR Reconstruction with Uncertainty
    ax[0, 1].plot(t_raw, raw_fhr, color=colors['gt'], label='Ground Truth FHR', linewidth=1.5)
    ax[0, 1].plot(t_raw, reconstructed_fhr_mu, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5)
    std_dev = np.exp(0.5 * reconstructed_fhr_logvar)
    ax[0, 1].fill_between(t_raw, reconstructed_fhr_mu - std_dev, reconstructed_fhr_mu + std_dev,
                           color=colors['uncertainty'], alpha=0.4, label='Uncertainty (±1σ)')
    ax[0, 1].set_title('FHR Reconstruction')
    ax[0, 1].set_xlabel('Time (s)')
    ax[0, 1].set_ylabel('Amplitude')
    ax[0, 1].legend()
    ax[0, 1].autoscale(enable=True, axis='x', tight=True)

    # 3. Latent Space z
    im_z = ax[1, 0].imshow(latent_z, aspect='auto', cmap='bwr', origin='lower')
    ax[1, 0].set_title('Latent Space (z)')
    ax[1, 0].set_xlabel('Time Steps')
    ax[1, 0].set_ylabel('Latent Dimensions')
    fig.colorbar(im_z, ax=ax[1, 0])

    # 4. KLD Tensor
    im_kld = ax[1, 1].imshow(kld_tensor, aspect='auto', cmap='bwr', origin='lower')
    ax[1, 1].set_title('KLD Tensor')
    ax[1, 1].set_xlabel('Time Steps')
    ax[1, 1].set_ylabel('Latent Dimensions')
    fig.colorbar(im_kld, ax=ax[1, 1])

    # 5. Mean KLD over time
    t_latent = np.arange(kld_mean_over_channels.shape[0])
    ax[2, 0].plot(t_latent, kld_mean_over_channels, color=colors['kld'], linewidth=1.5)
    ax[2, 0].set_title('Mean KLD Across Channels')
    ax[2, 0].set_xlabel('Time Steps')
    ax[2, 0].set_ylabel('KLD')
    ax[2, 0].autoscale(enable=True, axis='x', tight=True)

    # 6. fhr_st
    im_st = ax[2, 1].imshow(fhr_st, aspect='auto', cmap='viridis', origin='lower')
    ax[2, 1].set_title('FHR Scattering Transform (fhr_st)')
    ax[2, 1].set_xlabel('Time Steps')
    ax[2, 1].set_ylabel('Channels')
    fig.colorbar(im_st, ax=ax[2, 1])

    # 7. fhr_ph
    im_ph = ax[3, 0].imshow(fhr_ph, aspect='auto', cmap='viridis', origin='lower')
    ax[3, 0].set_title('FHR Phase Harmonics (fhr_ph)')
    ax[3, 0].set_xlabel('Time Steps')
    ax[3, 0].set_ylabel('Channels')
    fig.colorbar(im_ph, ax=ax[3, 0])

    # 8. fhr_up_ph
    im_up_ph = ax[3, 1].imshow(fhr_up_ph, aspect='auto', cmap='viridis', origin='lower')
    ax[3, 1].set_title('UP Phase Harmonics (fhr_up_ph)')
    ax[3, 1].set_xlabel('Time Steps')
    ax[3, 1].set_ylabel('Channels')
    fig.colorbar(im_up_ph, ax=ax[3, 1])

    fig.suptitle(f'Model Analysis - Best Checkpoint - Sample {batch_idx}', fontsize=16, fontweight='bold')
    
    save_path = os.path.join(output_dir, f'analysis_plot_best_checkpoint_sample_{batch_idx}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Analysis plot saved to {save_path}")
