
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_model_analysis(
    output_dir: str,
    raw_fhr: np.ndarray = None,
    raw_up: np.ndarray = None,
    fhr_st: np.ndarray = None,
    fhr_ph: np.ndarray = None,
    fhr_up_ph: np.ndarray = None,
    latent_z: np.ndarray = None,
    reconstructed_fhr_mu: np.ndarray = None,
    reconstructed_fhr_logvar: np.ndarray = None,
    kld_tensor: np.ndarray = None,
    kld_mean_over_channels: np.ndarray = None,
    batch_idx: int = 0,
    # New parameters for training callback
    y_raw_normalized: np.ndarray = None,
    up_raw_normalized: np.ndarray = None,
    mu_pr_means: np.ndarray = None,
    log_var_means: np.ndarray = None,
    mu_pr: np.ndarray = None,
    logvar_pr: np.ndarray = None,
    loss_dict: dict = None,
    epoch: int = 0,
    training_mode: bool = False
):
    """
    Generates and saves a comprehensive plot for model analysis.

    Args:
        output_dir (str): Directory to save the plot.
        # Original analysis mode parameters
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
        # Training callback mode parameters
        y_raw_normalized (np.ndarray): Raw FHR signal. Shape: (4800,).
        up_raw_normalized (np.ndarray): Raw UP signal. Shape: (4800,).
        mu_pr_means (np.ndarray): Mean reconstruction of FHR. Shape: (4800,).
        log_var_means (np.ndarray): Log variance of reconstruction. Shape: (4800,).
        mu_pr (np.ndarray): Per-timestep reconstructions. Shape: (300, 4800).
        logvar_pr (np.ndarray): Per-timestep log variances. Shape: (300, 4800).
        loss_dict (dict): Dictionary containing loss values (KLD, MSE, NLL, total_rec, total_loss).
        epoch (int): Current training epoch.
        training_mode (bool): Whether to use training callback mode (4 subplots) or analysis mode (8 subplots).
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

    if training_mode:
        # Training callback mode: 2x2 layout for 4 specific subplots
        fig, ax = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        ax = ax.flatten()
        
        # Use training callback data
        if y_raw_normalized is None or up_raw_normalized is None:
            raise ValueError("Training mode requires y_raw_normalized and up_raw_normalized")
    else:
        # Original analysis mode: 4x2 layout for 8 subplots  
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

    if training_mode:
        # Training callback mode: 4 specific subplots
        
        # Calculate KLD mean across all dimensions for display
        kld_mean_value = loss_dict.get('kld_loss', 0) if loss_dict else 0
        if isinstance(kld_mean_value, (np.ndarray, list)):
            kld_mean_value = np.mean(kld_mean_value)
        
        # Time axis for raw signals (assuming 4Hz sampling)
        t_raw = np.arange(len(y_raw_normalized)) / 4.0

        # 1. Raw FHR and UP signals
        ax[0].plot(t_raw, y_raw_normalized, color=colors['fhr'], label='Raw FHR', linewidth=1.2)
        ax[0].plot(t_raw, up_raw_normalized, color=colors['up'], label='Raw UP', linewidth=1.2)
        ax[0].set_title('Raw Input Signals')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        ax[0].legend()
        ax[0].autoscale(enable=True, axis='x', tight=True)
        
        # Add loss info at bottom
        if loss_dict:
            loss_text = f"KLD: {kld_mean_value:.4f} | MSE: {loss_dict.get('mse_loss', 0):.4f}"
            ax[0].text(0.5, -0.15, loss_text, transform=ax[0].transAxes, ha='center', 
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # 2. FHR Reconstruction with Uncertainty
        if mu_pr_means is not None and log_var_means is not None:
            ax[1].plot(t_raw, y_raw_normalized, color=colors['gt'], label='Ground Truth FHR', linewidth=1.5)
            ax[1].plot(t_raw, mu_pr_means, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5)
            std_dev = np.exp(0.5 * log_var_means)
            ax[1].fill_between(
                t_raw, mu_pr_means - std_dev, mu_pr_means + std_dev,
                color=colors['uncertainty'], alpha=0.4, label='Uncertainty (±1σ)')
            ax[1].set_title('FHR Reconstruction with Uncertainty')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Amplitude')
            ax[1].legend()
            ax[1].autoscale(enable=True, axis='x', tight=True)
            
            # Add loss info at bottom
            if loss_dict:
                loss_text = f"NLL: {loss_dict.get('nll_loss', 0):.4f} | Total Rec: {loss_dict.get('total_rec', loss_dict.get('reconstruction_loss', 0)):.4f}"
                ax[1].text(0.5, -0.15, loss_text, transform=ax[1].transAxes, ha='center',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # 3. Selected timesteps aggregation (handling NaN values)
        if mu_pr is not None:
            selected_timesteps = [30, 60, 90, 120, 150, 180, 210, 240, 270]
            # Filter timesteps that are within bounds
            valid_timesteps = [t for t in selected_timesteps if t < mu_pr.shape[0]]
            
            if valid_timesteps:
                # Extract selected timesteps and handle NaN values
                selected_mu = mu_pr[valid_timesteps, :]  # Shape: (n_valid_timesteps, 4800)
                
                # Use nanmean to aggregate, ignoring NaN values
                aggregated_mu = np.nanmean(selected_mu, axis=0)  # Shape: (4800,)
                
                # Only plot if we have valid data
                if not np.all(np.isnan(aggregated_mu)):
                    ax[2].plot(t_raw, y_raw_normalized, color=colors['gt'], label='Ground Truth FHR', linewidth=1.5)
                    ax[2].plot(t_raw, aggregated_mu, color=colors['recon'], label='Aggregated Reconstruction', linewidth=1.5)
                    ax[2].set_title(f'Aggregated Reconstruction (Timesteps: {valid_timesteps})')
                else:
                    ax[2].text(0.5, 0.5, 'No valid data (all NaN)', transform=ax[2].transAxes, 
                              ha='center', va='center', fontsize=12)
                    ax[2].set_title('Aggregated Reconstruction (No Valid Data)')
            else:
                ax[2].text(0.5, 0.5, 'No valid timesteps', transform=ax[2].transAxes, 
                          ha='center', va='center', fontsize=12)
                ax[2].set_title('Aggregated Reconstruction (No Valid Timesteps)')
            
            ax[2].set_xlabel('Time (s)')
            ax[2].set_ylabel('Amplitude')
            ax[2].legend()
            ax[2].autoscale(enable=True, axis='x', tight=True)
            
            # Add loss info at bottom
            if loss_dict:
                loss_text = f"Total Loss: {loss_dict.get('total_loss', 0):.4f}"
                ax[2].text(0.5, -0.15, loss_text, transform=ax[2].transAxes, ha='center',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # 4. Latent Space z
        if latent_z is not None:
            im_z = ax[3].imshow(latent_z.T, aspect='auto', cmap='bwr', origin='lower')
            ax[3].set_title('Latent Space (z)')
            ax[3].set_xlabel('Time Steps')
            ax[3].set_ylabel('Latent Dimensions')
            fig.colorbar(im_z, ax=ax[3])
            
            # Add KLD mean to latent space plot
            if loss_dict:
                loss_text = f"KLD Mean: {kld_mean_value:.4f} | Epoch: {epoch}"
                ax[3].text(0.5, -0.15, loss_text, transform=ax[3].transAxes, ha='center',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        fig.suptitle(f'Training Analysis - Epoch {epoch} - Sample {batch_idx}', fontsize=16, fontweight='bold')
        save_path = os.path.join(output_dir, f'training_analysis_epoch_{epoch}_sample_{batch_idx}.pdf')
        
    else:
        # Original analysis mode: 8 subplots
        
        # 1. Raw FHR and UP
        t_raw = np.arange(raw_fhr.shape[0]) / 4.0  # Assuming 4Hz
        ax[0, 0].plot(t_raw, raw_fhr, color=colors['fhr'], label='Raw FHR', linewidth=1.2)
        ax[0, 0].plot(t_raw, raw_up, color=colors['up'], label='Raw UP', linewidth=1.2)
        ax[0, 0].set_title('Raw Input Signals')
        ax[0, 0].set_xlabel('Time (s)')
        ax[0, 0].set_ylabel('Amplitude')
        ax[0, 0].legend()
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)

        # Calculate KLD mean for display in original mode
        kld_overall_mean = np.mean(kld_mean_over_channels) if kld_mean_over_channels is not None else 0
        
        # 2. FHR Reconstruction with Uncertainty
        ax[0, 1].plot(t_raw, raw_fhr, color=colors['gt'], label='Ground Truth FHR', linewidth=1.5)
        ax[0, 1].plot(t_raw, reconstructed_fhr_mu, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5)
        std_dev = np.exp(0.5 * reconstructed_fhr_logvar)
        ax[0, 1].fill_between(
            t_raw, reconstructed_fhr_mu - std_dev, reconstructed_fhr_mu + std_dev,
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
        ax[1, 1].set_title(f'KLD Tensor (Mean: {kld_overall_mean:.4f})')
        ax[1, 1].set_xlabel('Time Steps')
        ax[1, 1].set_ylabel('Latent Dimensions')
        fig.colorbar(im_kld, ax=ax[1, 1])

        # 5. Mean KLD over time
        t_latent = np.arange(kld_mean_over_channels.shape[0])
        ax[2, 0].plot(t_latent, kld_mean_over_channels, color=colors['kld'], linewidth=1.5)
        ax[2, 0].set_title(f'Mean KLD Across Channels (Overall Mean: {kld_overall_mean:.4f})')
        ax[2, 0].set_xlabel('Time Steps')
        ax[2, 0].set_ylabel('KLD')
        ax[2, 0].autoscale(enable=True, axis='x', tight=True)

        # 6. fhr_st
        im_st = ax[2, 1].imshow(fhr_st, aspect='auto', cmap='bwr', origin='lower')
        ax[2, 1].set_title('FHR Scattering Transform (fhr_st)')
        ax[2, 1].set_xlabel('Time Steps')
        ax[2, 1].set_ylabel('Channels')
        fig.colorbar(im_st, ax=ax[2, 1])

        # 7. fhr_ph
        im_ph = ax[3, 0].imshow(fhr_ph, aspect='auto', cmap='bwr', origin='lower')
        ax[3, 0].set_title('FHR Phase Harmonics (fhr_ph)')
        ax[3, 0].set_xlabel('Time Steps')
        ax[3, 0].set_ylabel('Channels')
        fig.colorbar(im_ph, ax=ax[3, 0])

        # 8. fhr_up_ph
        im_up_ph = ax[3, 1].imshow(fhr_up_ph, aspect='auto', cmap='bwr', origin='lower')
        ax[3, 1].set_title('UP Phase Harmonics (fhr_up_ph)')
        ax[3, 1].set_xlabel('Time Steps')
        ax[3, 1].set_ylabel('Channels')
        fig.colorbar(im_up_ph, ax=ax[3, 1])

        fig.suptitle(f'Model Analysis - Best Checkpoint - Sample {batch_idx}', fontsize=16, fontweight='bold')
        save_path = os.path.join(output_dir, f'analysis_plot_best_checkpoint_sample_{batch_idx}.pdf')
    
    # Save and close (common for both modes)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Analysis plot saved to {save_path}")
