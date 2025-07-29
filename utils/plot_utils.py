
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
    y_raw_unnormalized: np.ndarray = None,
    up_raw_unnormalized: np.ndarray = None,
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
        y_raw_normalized (np.ndarray): Normalized raw FHR signal. Shape: (4800,).
        up_raw_normalized (np.ndarray): Normalized raw UP signal. Shape: (4800,).
        y_raw_unnormalized (np.ndarray): Unnormalized raw FHR signal. Shape: (4800,).
        up_raw_unnormalized (np.ndarray): Unnormalized raw UP signal. Shape: (4800,).
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
        # Training callback mode: 4 rows, 2 columns (main plot + colorbar) like PlottingCallBack
        n_rows = 4
        fig, ax = plt.subplots(
            nrows=n_rows, ncols=2, figsize=(20, n_rows * 3.5),
            gridspec_kw={"width_ratios": [80, 1]}, constrained_layout=True)
        
        # Use training callback data
        if y_raw_normalized is None or up_raw_normalized is None:
            raise ValueError("Training mode requires y_raw_normalized and up_raw_normalized")
        if y_raw_unnormalized is None or up_raw_unnormalized is None:
            raise ValueError("Training mode requires y_raw_unnormalized and up_raw_unnormalized for the first plot")
    else:
        # Original analysis mode: 8 rows, single column layout like _plot_results  
        n_rows = 8
        fig, ax = plt.subplots(n_rows, 1, figsize=(16, n_rows * 3), constrained_layout=True)

    # Common settings for subplots
    if training_mode:
        # Configure scientific paper grid style for main plots only (left column)
        for i in range(n_rows):
            ax[i, 0].grid(True, linestyle='-', alpha=0.4, linewidth=0.4, color='#D2C1B6')
            ax[i, 0].grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax[i, 0].minorticks_on()
            ax[i, 0].set_axisbelow(True)
            ax[i, 0].spines['top'].set_visible(False)
            ax[i, 0].spines['right'].set_visible(False)
            ax[i, 0].spines['left'].set_color('#A2B9A7')
            ax[i, 0].spines['bottom'].set_color('#A2B9A7')
            ax[i, 0].spines['left'].set_linewidth(0.7)
            ax[i, 0].spines['bottom'].set_linewidth(0.7)
    else:
        # Original mode: apply to all subplots (now single column)
        for i in range(n_rows):
            ax[i].grid(True, linestyle='-', alpha=0.4, linewidth=0.4, color='#D2C1B6')
            ax[i].grid(True, which='minor', linestyle=':', alpha=0.25, linewidth=0.3, color='#D2C1B6')
            ax[i].minorticks_on()
            ax[i].set_axisbelow(True)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_color('#A2B9A7')
            ax[i].spines['bottom'].set_color('#A2B9A7')
            ax[i].spines['left'].set_linewidth(0.7)
            ax[i].spines['bottom'].set_linewidth(0.7)

    if training_mode:
        # Training callback mode: 4 specific subplots in single column layout
        
        # Calculate KLD mean the same way as during training
        # Use the exact same procedure as in training to match logged values
        if loss_dict and 'kld_loss' in loss_dict:
            kld_mean_value = loss_dict['kld_loss']
            # If it's a tensor or array, convert to scalar (it should already be scalar from training)
            if hasattr(kld_mean_value, 'item'):
                kld_mean_value = kld_mean_value.item()
            elif isinstance(kld_mean_value, (np.ndarray, list)):
                kld_mean_value = float(np.mean(kld_mean_value))
            else:
                kld_mean_value = float(kld_mean_value)
        else:
            kld_mean_value = 0.0
        
        # Time axis for raw signals (assuming 4Hz sampling)
        t_raw = np.arange(len(y_raw_unnormalized)) / 4.0

        # 1. Raw unnormalized FHR and UP signals
        ax[0, 1].set_axis_off()  # Turn off colorbar column for this plot
        ax[0, 0].plot(t_raw, y_raw_unnormalized, color=colors['fhr'], label='Raw FHR', linewidth=1.2, alpha=0.85)
        ax[0, 0].plot(t_raw, up_raw_unnormalized, color=colors['up'], label='Raw UP', linewidth=1.2, alpha=0.85)
        ax[0, 0].set_title('Raw Unnormalized FHR and UP Signals', fontweight='normal', pad=12)
        ax[0, 0].set_ylabel('Amplitude', fontweight='normal')
        ax[0, 0].legend(loc='upper right', framealpha=0.95)
        ax[0, 0].autoscale(enable=True, axis='x', tight=True)
        
        # Add loss info at bottom
        if loss_dict:
            loss_text = f"KLD: {kld_mean_value:.4f} | MSE: {loss_dict.get('mse_loss', 0):.4f}"
            ax[0, 0].text(0.5, -0.15, loss_text, transform=ax[0, 0].transAxes, ha='center', 
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # 2. FHR Reconstruction with Uncertainty
        if mu_pr_means is not None and log_var_means is not None:
            # Time axis for normalized signals (may be different length)
            t_raw_norm = np.arange(len(y_raw_normalized)) / 4.0
            
            ax[1, 1].set_axis_off()  # Turn off colorbar column for this plot
            ax[1, 0].plot(t_raw_norm, y_raw_normalized, color=colors['gt'], label='Ground Truth', linewidth=1.5, alpha=0.85, zorder=3)
            ax[1, 0].plot(t_raw_norm, mu_pr_means, color=colors['recon'], label='Reconstruction', linewidth=1.5, alpha=0.85, zorder=2)
            
            # Add uncertainty visualization using log_var_means
            std_dev = np.exp(0.5 * log_var_means)  # Convert log variance to standard deviation
            ax[1, 0].fill_between(t_raw_norm, mu_pr_means - std_dev, mu_pr_means + std_dev, 
                                alpha=0.3, color=colors['uncertainty'], label='Uncertainty (±1σ)', zorder=1)
            
            ax[1, 0].set_title('FHR Reconstruction with Uncertainty', fontweight='normal', pad=12)
            ax[1, 0].set_ylabel('FHR (bpm)', fontweight='normal')
            ax[1, 0].legend(loc='upper right', framealpha=0.95)
            ax[1, 0].autoscale(enable=True, axis='x', tight=True)
            
            # Add loss info at bottom
            if loss_dict:
                loss_text = f"NLL: {loss_dict.get('nll_loss', 0):.4f} | Total Rec: {loss_dict.get('total_rec', loss_dict.get('reconstruction_loss', 0)):.4f}"
                ax[1, 0].text(0.5, -0.15, loss_text, transform=ax[1, 0].transAxes, ha='center',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # 3. Selected timesteps aggregation (handling NaN values)
        if mu_pr is not None:
            ax[2, 1].set_axis_off()  # Turn off colorbar column for this plot
            selected_timesteps = [30, 60, 90, 120, 150, 180, 210, 240, 270]
            
            # Handle different formats of mu_pr
            if len(mu_pr.shape) == 1:  # (4800,) format - single prediction
                t_raw_norm = np.arange(len(y_raw_normalized)) / 4.0
                ax[2, 0].plot(t_raw_norm, y_raw_normalized, color=colors['gt'], label='Ground Truth', linewidth=1.5, alpha=0.85, zorder=2)
                ax[2, 0].plot(t_raw_norm, mu_pr, linewidth=1.5, color=colors['recon'], 
                            label='Model Prediction', alpha=0.85, zorder=1)
                ax[2, 0].set_title('FHR vs Model Reconstructions', fontweight='normal', pad=12)
            else:  # (300, 4800) format - multiple predictions
                # Filter timesteps that are within bounds
                valid_timesteps = [t for t in selected_timesteps if t < mu_pr.shape[0]]
                
                if valid_timesteps:
                    # Handle NaN values and sum selected samples - matching PlottingCallBack logic
                    selected_samples = mu_pr[valid_timesteps, :]  # Shape: (len(valid_timesteps), 4800)
                    
                    # Remove NaN values and compute sum
                    valid_mask = ~np.isnan(selected_samples)
                    summed_samples = np.zeros(len(y_raw_normalized))
                    
                    for i in range(len(y_raw_normalized)):
                        valid_values = selected_samples[:, i][valid_mask[:, i]]
                        if len(valid_values) > 0:
                            summed_samples[i] = np.sum(valid_values)
                        else:
                            summed_samples[i] = 0
                    
                    t_raw_norm = np.arange(len(y_raw_normalized)) / 4.0
                    ax[2, 0].plot(t_raw_norm, y_raw_normalized, color=colors['gt'], label='Ground Truth', linewidth=1.5, alpha=0.85, zorder=2)
                    ax[2, 0].plot(t_raw_norm, summed_samples, linewidth=1.5, color=colors['recon'], 
                                label='Selected Samples Sum', alpha=0.85, zorder=1)
                    ax[2, 0].set_title('FHR vs Model Reconstructions', fontweight='normal', pad=12)
                else:
                    # Fallback to first sample if no valid indices
                    t_raw_norm = np.arange(len(y_raw_normalized)) / 4.0
                    ax[2, 0].plot(t_raw_norm, y_raw_normalized, color=colors['gt'], label='Ground Truth', linewidth=1.5, alpha=0.85, zorder=2)
                    ax[2, 0].plot(t_raw_norm, mu_pr[0, :], linewidth=1.5, color=colors['recon'], 
                                label='First Sample', alpha=0.85, zorder=1)
                    ax[2, 0].set_title('FHR vs Model Reconstructions', fontweight='normal', pad=12)
            
            ax[2, 0].set_ylabel('FHR (bpm)', fontweight='normal')
            ax[2, 0].legend(loc='upper right', framealpha=0.95)
            ax[2, 0].autoscale(enable=True, axis='x', tight=True)
            
            # Add loss info at bottom
            if loss_dict:
                loss_text = f"Total Loss: {loss_dict.get('total_loss', 0):.4f}"
                ax[2, 0].text(0.5, -0.15, loss_text, transform=ax[2, 0].transAxes, ha='center',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # 4. Latent Space z
        if latent_z is not None:
            imgplot = ax[3, 0].imshow(latent_z.T, aspect='auto', cmap='bwr', origin='lower')
            
            # Remove grid lines from imshow plot
            ax[3, 0].grid(False)
            
            ax[3, 1].set_axis_on()
            cbar = fig.colorbar(imgplot, cax=ax[3, 1])
            cbar.ax.tick_params(labelsize=10, colors='#666666')
            cbar.set_label('Activation', fontweight='normal', fontsize=11, color='#666666')
            cbar.outline.set_color('#A2B9A7')
            cbar.outline.set_linewidth(0.7)
            ax[3, 0].set_ylabel('Latent Dimensions', fontweight='normal')
            ax[3, 0].set_xlabel('Time Steps', fontweight='normal')
            ax[3, 0].set_title('Latent Space Representation', fontweight='normal', pad=12)
            
            # Add KLD mean to latent space plot
            if loss_dict:
                loss_text = f"KLD Mean: {kld_mean_value:.4f} | Epoch: {epoch}"
                ax[3, 0].text(0.5, -0.15, loss_text, transform=ax[3, 0].transAxes, ha='center',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))

        # Set overall title with scientific paper styling
        fig.suptitle(f'Model Performance Analysis — Epoch {epoch}', 
                    fontsize=14, fontweight='normal', y=0.97, color='#456882')
        save_path = os.path.join(output_dir, f'model_results_epoch_{epoch}.pdf')
        
    else:
        # Original analysis mode: 8 subplots
        
        # 1. Raw FHR and UP
        t_raw = np.arange(raw_fhr.shape[0]) / 4.0  # Assuming 4Hz
        ax[0].plot(t_raw, raw_fhr, color=colors['fhr'], label='Raw FHR', linewidth=1.2)
        ax[0].plot(t_raw, raw_up, color=colors['up'], label='Raw UP', linewidth=1.2)
        ax[0].set_title('Raw Input Signals')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        ax[0].legend()
        ax[0].autoscale(enable=True, axis='x', tight=True)

        # Calculate KLD mean for display in original mode
        kld_overall_mean = np.mean(kld_mean_over_channels) if kld_mean_over_channels is not None else 0
        
        # 2. FHR Reconstruction with Uncertainty
        ax[1].plot(t_raw, raw_fhr, color=colors['gt'], label='Ground Truth FHR', linewidth=1.5)
        ax[1].plot(t_raw, reconstructed_fhr_mu, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5)
        std_dev = np.exp(0.5 * reconstructed_fhr_logvar)
        ax[1].fill_between(
            t_raw, reconstructed_fhr_mu - std_dev, reconstructed_fhr_mu + std_dev,
            color=colors['uncertainty'], alpha=0.4, label='Uncertainty (±1σ)')
        ax[1].set_title('FHR Reconstruction')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Amplitude')
        ax[1].legend()
        ax[1].autoscale(enable=True, axis='x', tight=True)

        # 3. Latent Space z
        im_z = ax[2].imshow(latent_z, aspect='auto', cmap='bwr', origin='lower')
        ax[2].grid(False)  # Remove grid lines
        ax[2].set_title('Latent Space (z)')
        ax[2].set_xlabel('Time Steps')
        ax[2].set_ylabel('Latent Dimensions')
        fig.colorbar(im_z, ax=ax[2])

        # 4. KLD Tensor
        im_kld = ax[3].imshow(kld_tensor, aspect='auto', cmap='bwr', origin='lower')
        ax[3].grid(False)  # Remove grid lines
        ax[3].set_title(f'KLD Tensor (Mean: {kld_overall_mean:.4f})')
        ax[3].set_xlabel('Time Steps')
        ax[3].set_ylabel('Latent Dimensions')
        fig.colorbar(im_kld, ax=ax[3])

        # 5. Mean KLD over time
        t_latent = np.arange(kld_mean_over_channels.shape[0])
        ax[4].plot(t_latent, kld_mean_over_channels, color=colors['kld'], linewidth=1.5)
        ax[4].set_title(f'Mean KLD Across Channels (Overall Mean: {kld_overall_mean:.4f})')
        ax[4].set_xlabel('Time Steps')
        ax[4].set_ylabel('KLD')
        ax[4].autoscale(enable=True, axis='x', tight=True)

        # 6. fhr_st
        im_st = ax[5].imshow(fhr_st, aspect='auto', cmap='bwr', origin='lower')
        ax[5].grid(False)  # Remove grid lines
        ax[5].set_title('FHR Scattering Transform (fhr_st)')
        ax[5].set_xlabel('Time Steps')
        ax[5].set_ylabel('Channels')
        fig.colorbar(im_st, ax=ax[5])

        # 7. fhr_ph
        im_ph = ax[6].imshow(fhr_ph, aspect='auto', cmap='bwr', origin='lower')
        ax[6].grid(False)  # Remove grid lines
        ax[6].set_title('FHR Phase Harmonics (fhr_ph)')
        ax[6].set_xlabel('Time Steps')
        ax[6].set_ylabel('Channels')
        fig.colorbar(im_ph, ax=ax[6])

        # 8. fhr_up_ph
        im_up_ph = ax[7].imshow(fhr_up_ph, aspect='auto', cmap='bwr', origin='lower')
        ax[7].grid(False)  # Remove grid lines
        ax[7].set_title('UP Phase Harmonics (fhr_up_ph)')
        ax[7].set_xlabel('Time Steps')
        ax[7].set_ylabel('Channels')
        fig.colorbar(im_up_ph, ax=ax[7])

        fig.suptitle(f'Model Analysis - Best Checkpoint - Sample {batch_idx}', fontsize=16, fontweight='bold')
        save_path = os.path.join(output_dir, f'analysis_plot_best_checkpoint_sample_{batch_idx}.pdf')
    
    # Save and close (common for both modes)
    if training_mode:
        # Save plot as PDF with high quality - matching PlottingCallBack
        plt.savefig(save_path, bbox_inches='tight', orientation='landscape', dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Clean up memory like PlottingCallBack
        import gc
        if 'y_raw_normalized' in locals():
            del y_raw_normalized
        if 'up_raw_normalized' in locals():
            del up_raw_normalized
        if 'y_raw_unnormalized' in locals():
            del y_raw_unnormalized
        if 'up_raw_unnormalized' in locals():
            del up_raw_unnormalized
        if 'mu_pr_means' in locals():
            del mu_pr_means
        if 'log_var_means' in locals():
            del log_var_means
        if 'mu_pr' in locals():
            del mu_pr
        if 'latent_z' in locals():
            del latent_z
        gc.collect()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Analysis plot saved to {save_path}")
