
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
    training_mode: bool = False,
    # New parameters for analysis mode with both normalized and unnormalized data
    raw_fhr_normalized: np.ndarray = None,
    raw_up_normalized: np.ndarray = None,
    # Optional channel index splits
    phase_auto_indices: np.ndarray = None,
    phase_cross_indices: np.ndarray = None,
    cross_auto_indices: np.ndarray = None,
    cross_cross_indices: np.ndarray = None
):
    """
    Generates and saves a comprehensive plot for model analysis.

    Args:
        output_dir (str): Directory to save the plot.
        # Original analysis mode parameters
        raw_fhr (np.ndarray): Raw FHR signal (unnormalized for first plot). Shape: (N,).
        raw_up (np.ndarray): Raw UP signal (unnormalized for first plot). Shape: (N,).
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
        # Analysis mode additional parameters
        raw_fhr_normalized (np.ndarray): Normalized FHR signal for reconstruction comparison. Shape: (N,).
        raw_up_normalized (np.ndarray): Normalized UP signal (if needed). Shape: (N,).
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
        ax[0].set_title('Raw Input Signals (Unnormalized)')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Original Amplitude')
        ax[0].legend()
        ax[0].autoscale(enable=True, axis='x', tight=True)

        # Calculate KLD mean for display in original mode
        kld_overall_mean = np.mean(kld_mean_over_channels) if kld_mean_over_channels is not None else 0
        
        # 2. FHR Reconstruction with Uncertainty
        # Use normalized FHR for reconstruction comparison if available, otherwise use raw_fhr
        fhr_for_reconstruction = raw_fhr_normalized if raw_fhr_normalized is not None else raw_fhr
        ax[1].plot(t_raw, fhr_for_reconstruction, color=colors['gt'], label='Ground Truth FHR', linewidth=1.5)
        ax[1].plot(t_raw, reconstructed_fhr_mu, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5)
        std_dev = np.exp(0.5 * reconstructed_fhr_logvar)
        ax[1].fill_between(
            t_raw, reconstructed_fhr_mu - std_dev, reconstructed_fhr_mu + std_dev,
            color=colors['uncertainty'], alpha=0.4, label='Uncertainty (±1σ)')
        ax[1].set_title('FHR Reconstruction (Normalized Space)')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Normalized Amplitude')
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

        # 6. fhr_st (channel 0 at top)
        im_st = ax[5].imshow(fhr_st, aspect='auto', cmap='bwr', origin='upper')
        ax[5].grid(False)  # Remove grid lines
        ax[5].set_title('FHR Scattering Transform (fhr_st)')
        ax[5].set_xlabel('Time Steps')
        ax[5].set_ylabel('Channels')
        fig.colorbar(im_st, ax=ax[5])

        # 7-8. fhr_ph split into autocorr (same freq) and cross (different freq)
        if phase_auto_indices is not None and phase_cross_indices is not None:
            ph_auto = fhr_ph[phase_auto_indices, :] if len(phase_auto_indices) > 0 else None
            ph_cross = fhr_ph[phase_cross_indices, :] if len(phase_cross_indices) > 0 else None
            # Autocorr
            if ph_auto is not None and ph_auto.size > 0:
                im_ph_auto = ax[6].imshow(ph_auto, aspect='auto', cmap='bwr', origin='upper')
                ax[6].grid(False)
                ax[6].set_title('FHR Phase Harmonics — Autocorr (same freq)')
                ax[6].set_xlabel('Time Steps')
                ax[6].set_ylabel('Channels')
                fig.colorbar(im_ph_auto, ax=ax[6])
            else:
                ax[6].set_title('FHR Phase Harmonics — Autocorr (none)')
                ax[6].set_axis_off()
            # Cross
            if ph_cross is not None and ph_cross.size > 0:
                im_ph_cross = ax[7].imshow(ph_cross, aspect='auto', cmap='bwr', origin='upper')
                ax[7].grid(False)
                ax[7].set_title('FHR Phase Harmonics — Cross (different freq)')
                ax[7].set_xlabel('Time Steps')
                ax[7].set_ylabel('Channels')
                fig.colorbar(im_ph_cross, ax=ax[7])
            else:
                ax[7].set_title('FHR Phase Harmonics — Cross (none)')
                ax[7].set_axis_off()
        else:
            im_ph = ax[6].imshow(fhr_ph, aspect='auto', cmap='bwr', origin='upper')
            ax[6].grid(False)  # Remove grid lines
            ax[6].set_title('FHR Phase Harmonics (fhr_ph)')
            ax[6].set_xlabel('Time Steps')
            ax[6].set_ylabel('Channels')
            fig.colorbar(im_ph, ax=ax[6])

        # 9-10. fhr_up_ph split into autocorr (same freq) and cross (different freq)
        if phase_auto_indices is not None and phase_cross_indices is not None and \
           cross_auto_indices is not None and cross_cross_indices is not None:
            # Shift indices for positions (axes 8 and 9 weren't defined originally)
            # Extend figure if needed
            # If we already used ax[7] above for phase split, we need more rows.
            pass
        
        # Backward compatible single-plot for cross if splits not provided
        if cross_auto_indices is None or cross_cross_indices is None:
            # If we used two slots for phase above, the next available axis index is 8
            # But since the figure was created with fixed rows earlier, we keep legacy placement
            im_up_ph = ax[7].imshow(fhr_up_ph, aspect='auto', cmap='bwr', origin='upper')
            ax[7].grid(False)
            ax[7].set_title('UP→FHR Cross-Phase Harmonics (fhr_up_ph)')
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

        # Additional separate figures for phase/cross-phase splits when indices are provided
        try:
            # Phase harmonic split
            if phase_auto_indices is not None and phase_cross_indices is not None and fhr_ph is not None:
                if isinstance(phase_auto_indices, (list, tuple)):
                    phase_auto_idx = np.array(phase_auto_indices)
                else:
                    phase_auto_idx = phase_auto_indices
                if isinstance(phase_cross_indices, (list, tuple)):
                    phase_cross_idx = np.array(phase_cross_indices)
                else:
                    phase_cross_idx = phase_cross_indices

                if phase_auto_idx.size > 0:
                    fig_pa, ax_pa = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
                    im_pa = ax_pa.imshow(fhr_ph[phase_auto_idx, :], aspect='auto', cmap='bwr', origin='upper')
                    ax_pa.grid(False)
                    ax_pa.set_title('FHR Phase Harmonics — Autocorr (same freq)')
                    ax_pa.set_xlabel('Time Steps')
                    ax_pa.set_ylabel('Channels')
                    fig_pa.colorbar(im_pa, ax=ax_pa)
                    pa_path = os.path.join(output_dir, f'phase_harmonics_autocorr_sample_{batch_idx}.png')
                    plt.savefig(pa_path, bbox_inches='tight')
                    plt.close(fig_pa)

                if phase_cross_idx.size > 0:
                    fig_pc, ax_pc = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
                    im_pc = ax_pc.imshow(fhr_ph[phase_cross_idx, :], aspect='auto', cmap='bwr', origin='upper')
                    ax_pc.grid(False)
                    ax_pc.set_title('FHR Phase Harmonics — Cross (different freq)')
                    ax_pc.set_xlabel('Time Steps')
                    ax_pc.set_ylabel('Channels')
                    fig_pc.colorbar(im_pc, ax=ax_pc)
                    pc_path = os.path.join(output_dir, f'phase_harmonics_cross_sample_{batch_idx}.png')
                    plt.savefig(pc_path, bbox_inches='tight')
                    plt.close(fig_pc)

            # Cross-channel phase split
            if cross_auto_indices is not None and cross_cross_indices is not None and fhr_up_ph is not None:
                if isinstance(cross_auto_indices, (list, tuple)):
                    cross_auto_idx = np.array(cross_auto_indices)
                else:
                    cross_auto_idx = cross_auto_indices
                if isinstance(cross_cross_indices, (list, tuple)):
                    cross_cross_idx = np.array(cross_cross_indices)
                else:
                    cross_cross_idx = cross_cross_indices

                if cross_auto_idx.size > 0:
                    fig_cpa, ax_cpa = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
                    im_cpa = ax_cpa.imshow(fhr_up_ph[cross_auto_idx, :], aspect='auto', cmap='bwr', origin='upper')
                    ax_cpa.grid(False)
                    ax_cpa.set_title('UP→FHR Cross-Phase — Autocorr (same filter)')
                    ax_cpa.set_xlabel('Time Steps')
                    ax_cpa.set_ylabel('Channels')
                    fig_cpa.colorbar(im_cpa, ax=ax_cpa)
                    cpa_path = os.path.join(output_dir, f'cross_phase_autocorr_sample_{batch_idx}.png')
                    plt.savefig(cpa_path, bbox_inches='tight')
                    plt.close(fig_cpa)

                if cross_cross_idx.size > 0:
                    fig_cpc, ax_cpc = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
                    im_cpc = ax_cpc.imshow(fhr_up_ph[cross_cross_idx, :], aspect='auto', cmap='bwr', origin='upper')
                    ax_cpc.grid(False)
                    ax_cpc.set_title('UP→FHR Cross-Phase — Cross (different filters)')
                    ax_cpc.set_xlabel('Time Steps')
                    ax_cpc.set_ylabel('Channels')
                    fig_cpc.colorbar(im_cpc, ax=ax_cpc)
                    cpc_path = os.path.join(output_dir, f'cross_phase_cross_sample_{batch_idx}.png')
                    plt.savefig(cpc_path, bbox_inches='tight')
                    plt.close(fig_cpc)
        except Exception as e:
            # Avoid breaking analysis if extra plots fail
            print(f"Warning: failed to save split phase/cross-phase plots: {e}")

    print(f"Analysis plot saved to {save_path}")


def plot_vae_reconstruction(
    output_dir: str,
    raw_fhr_unnormalized: np.ndarray,
    raw_up_unnormalized: np.ndarray,
    raw_fhr_normalized: np.ndarray,
    raw_up_normalized: np.ndarray,
    reconstructed_fhr: np.ndarray,
    original_scattering_transform: np.ndarray,  # Shape: (43, 300)
    reconstructed_scattering_transform: np.ndarray,  # Shape: (43, 300)
    original_phase_harmonic: np.ndarray,  # Shape: (44, 300)
    reconstructed_phase_harmonic: np.ndarray,  # Shape: (44, 300)
    scattering_channel_data: dict = None,  # From frequency analysis
    batch_idx: int = 0,
    loss_dict: dict = None
):
    """
    Generates and saves comprehensive VAE reconstruction analysis plots.
    
    Args:
        output_dir (str): Directory to save the plot.
        raw_fhr_unnormalized (np.ndarray): Raw unnormalized FHR signal. Shape: (4800,).
        raw_up_unnormalized (np.ndarray): Raw unnormalized UP signal. Shape: (4800,).
        raw_fhr_normalized (np.ndarray): Raw normalized FHR signal. Shape: (4800,).
        raw_up_normalized (np.ndarray): Raw normalized UP signal. Shape: (4800,).
        reconstructed_fhr (np.ndarray): Reconstructed FHR signal. Shape: (4800,).
        original_scattering_transform (np.ndarray): Original scattering coeffs. Shape: (43, 300).
        reconstructed_scattering_transform (np.ndarray): Reconstructed scattering coeffs. Shape: (43, 300).
        original_phase_harmonic (np.ndarray): Original phase harmonic coeffs. Shape: (44, 300).
        reconstructed_phase_harmonic (np.ndarray): Reconstructed phase harmonic coeffs. Shape: (44, 300).
        scattering_channel_data (dict): Frequency analysis data for channel annotations.
        batch_idx (int): Index of the sample for file naming.
        loss_dict (dict): Dictionary containing loss values.
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

    # Create figure with multiple rows for comprehensive analysis
    # Total: 8 + N channel plots (where N is number of scattering channels to plot)
    n_main_plots = 8
    n_channel_plots = min(10, original_scattering_transform.shape[0])  # Show first 10 channels
    n_rows = n_main_plots + n_channel_plots
    
    fig, ax = plt.subplots(n_rows, 1, figsize=(16, n_rows * 2.5), constrained_layout=True)

    # Configure scientific paper grid style for all plots
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

    # Time axes
    t_raw = np.arange(len(raw_fhr_unnormalized)) / 4.0  # 4Hz sampling
    t_coeffs = np.arange(original_scattering_transform.shape[1])  # Time steps for coefficients

    # 1. Raw unnormalized FHR and UP signals
    ax[0].plot(t_raw, raw_fhr_unnormalized, color=colors['fhr'], label='Raw FHR', linewidth=1.2, alpha=0.85)
    ax[0].plot(t_raw, raw_up_unnormalized, color=colors['up'], label='Raw UP', linewidth=1.2, alpha=0.85)
    ax[0].set_title('Raw Unnormalized FHR and UP Signals', fontweight='normal', pad=12)
    ax[0].set_ylabel('Amplitude', fontweight='normal')
    ax[0].set_xlabel('Time (s)', fontweight='normal')
    ax[0].legend(loc='upper right', framealpha=0.95)
    ax[0].autoscale(enable=True, axis='x', tight=True)

    # 2. Raw normalized FHR and reconstructed
    ax[1].plot(t_raw, raw_fhr_normalized, color=colors['gt'], label='Normalized FHR', linewidth=1.5, alpha=0.85)
    ax[1].plot(t_raw, reconstructed_fhr, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5, alpha=0.85)
    ax[1].set_title('Normalized FHR vs Reconstructed FHR', fontweight='normal', pad=12)
    ax[1].set_ylabel('Normalized Amplitude', fontweight='normal')
    ax[1].set_xlabel('Time (s)', fontweight='normal')
    ax[1].legend(loc='upper right', framealpha=0.95)
    ax[1].autoscale(enable=True, axis='x', tight=True)

    # 3. Only reconstructed FHR
    ax[2].plot(t_raw, reconstructed_fhr, color=colors['recon'], label='Reconstructed FHR', linewidth=1.5, alpha=0.85)
    ax[2].set_title('Reconstructed FHR Signal', fontweight='normal', pad=12)
    ax[2].set_ylabel('Normalized Amplitude', fontweight='normal')
    ax[2].set_xlabel('Time (s)', fontweight='normal')
    ax[2].legend(loc='upper right', framealpha=0.95)
    ax[2].autoscale(enable=True, axis='x', tight=True)

    # 4. Original scattering transform (imshow)
    im_st_orig = ax[3].imshow(original_scattering_transform, aspect='auto', cmap='bwr', origin='upper', vmin=-3, vmax=3)
    ax[3].grid(False)  # Remove grid lines for imshow
    ax[3].set_title('Original Scattering Transform Coefficients', fontweight='normal', pad=12)
    ax[3].set_xlabel('Time Steps', fontweight='normal')
    ax[3].set_ylabel('Scattering Channels', fontweight='normal')
    fig.colorbar(im_st_orig, ax=ax[3], shrink=0.8)

    # 5. Reconstructed scattering transform (imshow)
    im_st_recon = ax[4].imshow(reconstructed_scattering_transform, aspect='auto', cmap='bwr', origin='upper', vmin=-3, vmax=3)
    ax[4].grid(False)  # Remove grid lines for imshow
    ax[4].set_title('Reconstructed Scattering Transform Coefficients', fontweight='normal', pad=12)
    ax[4].set_xlabel('Time Steps', fontweight='normal')
    ax[4].set_ylabel('Scattering Channels', fontweight='normal')
    fig.colorbar(im_st_recon, ax=ax[4], shrink=0.8)

    # 6. Original phase harmonic (imshow)
    im_ph_orig = ax[5].imshow(original_phase_harmonic, aspect='auto', cmap='bwr', origin='upper', vmin=-3, vmax=3)
    ax[5].grid(False)  # Remove grid lines for imshow
    ax[5].set_title('Original Phase Harmonic Coefficients', fontweight='normal', pad=12)
    ax[5].set_xlabel('Time Steps', fontweight='normal')
    ax[5].set_ylabel('Phase Harmonic Channels', fontweight='normal')
    fig.colorbar(im_ph_orig, ax=ax[5], shrink=0.8)

    # 7. Reconstructed phase harmonic (imshow)
    im_ph_recon = ax[6].imshow(reconstructed_phase_harmonic, aspect='auto', cmap='bwr', origin='upper', vmin=-3, vmax=3)
    ax[6].grid(False)  # Remove grid lines for imshow
    ax[6].set_title('Reconstructed Phase Harmonic Coefficients', fontweight='normal', pad=12)
    ax[6].set_xlabel('Time Steps', fontweight='normal')
    ax[6].set_ylabel('Phase Harmonic Channels', fontweight='normal')
    fig.colorbar(im_ph_recon, ax=ax[6], shrink=0.8)

    # 8. Reconstruction error heatmap
    st_error = np.abs(original_scattering_transform - reconstructed_scattering_transform)
    ph_error = np.abs(original_phase_harmonic - reconstructed_phase_harmonic)
    combined_error = np.vstack([st_error, ph_error])
    
    im_error = ax[7].imshow(combined_error, aspect='auto', cmap='Reds', origin='upper')
    ax[7].grid(False)  # Remove grid lines for imshow
    ax[7].set_title('Reconstruction Error (|Original - Reconstructed|)', fontweight='normal', pad=12)
    ax[7].set_xlabel('Time Steps', fontweight='normal')
    ax[7].set_ylabel('All Channels (ST + PH)', fontweight='normal')
    # Add horizontal line to separate ST and PH
    ax[7].axhline(y=original_scattering_transform.shape[0]-0.5, color='white', linewidth=2, alpha=0.8)
    ax[7].text(
        combined_error.shape[1]*0.02, original_scattering_transform.shape[0]/2, 'ST', 
        color='white', fontweight='bold', fontsize=10, va='center')
    ax[7].text(combined_error.shape[1]*0.02, original_scattering_transform.shape[0] + original_phase_harmonic.shape[0]/2, 
               'PH', color='white', fontweight='bold', fontsize=10, va='center')
    fig.colorbar(im_error, ax=ax[7], shrink=0.8)

    # 9-N. Individual scattering channel plots with frequency information
    for i in range(n_channel_plots):
        plot_idx = n_main_plots + i
        channel = i
        
        # Plot original and reconstructed for this channel
        ax[plot_idx].plot(t_coeffs, original_scattering_transform[channel, :], 
                         color=colors['gt'], label='Original', linewidth=1.2, alpha=0.85)
        ax[plot_idx].plot(t_coeffs, reconstructed_scattering_transform[channel, :], 
                         color=colors['recon'], label='Reconstructed', linewidth=1.2, alpha=0.85)
        
        # Add frequency information from scattering analysis if available
        freq_info = ""
        if scattering_channel_data and 'first_order_filters' in scattering_channel_data:
            if channel < len(scattering_channel_data['first_order_filters']):
                filter_info = scattering_channel_data['first_order_filters'][channel]
                freq_hz = filter_info.get('center_freq_hz', 0)
                bandwidth_hz = filter_info.get('bandwidth_hz', 0)
                freq_range = filter_info.get('frequency_range_hz', (0, 0))
                physiol_band = filter_info.get('physiological_band', 'Unknown')
                freq_info = f" (f={freq_hz:.3f}Hz, BW={bandwidth_hz:.3f}Hz, {physiol_band})"
        
        ax[plot_idx].set_title(f'Scattering Channel {channel}{freq_info}', fontweight='normal', pad=8)
        ax[plot_idx].set_ylabel('Coefficient Value', fontweight='normal')
        if i == n_channel_plots - 1:  # Only add xlabel to the last plot
            ax[plot_idx].set_xlabel('Time Steps', fontweight='normal')
        ax[plot_idx].legend(loc='upper right', framealpha=0.95, fontsize=9)
        ax[plot_idx].autoscale(enable=True, axis='x', tight=True)
        
        # Add error metrics for this channel
        mse_channel = np.mean((original_scattering_transform[channel, :] - 
                              reconstructed_scattering_transform[channel, :])**2)
        mae_channel = np.mean(np.abs(original_scattering_transform[channel, :] - 
                                   reconstructed_scattering_transform[channel, :]))
        
        # Add small text box with error metrics
        error_text = f"MSE: {mse_channel:.4f}, MAE: {mae_channel:.4f}"
        ax[plot_idx].text(0.98, 0.95, error_text, transform=ax[plot_idx].transAxes, 
                         ha='right', va='top', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                                 alpha=0.8, edgecolor='#A2B9A7'))

    # Add overall loss information if available
    loss_text = ""
    if loss_dict:
        if 'mse_loss' in loss_dict and 'nll_loss' in loss_dict:
            loss_text = f"MSE Loss: {loss_dict['mse_loss']:.4f}, NLL Loss: {loss_dict['nll_loss']:.4f}"
        elif 'total_loss' in loss_dict:
            loss_text = f"Total Loss: {loss_dict['total_loss']:.4f}"

    # Set overall title with scientific paper styling
    title_text = f'VAE Reconstruction Analysis — Sample {batch_idx}'
    if loss_text:
        title_text += f" — {loss_text}"
    
    fig.suptitle(title_text, fontsize=14, fontweight='normal', y=0.99, color='#456882')
    
    # Save plot
    save_path = os.path.join(output_dir, f'vae_reconstruction_analysis_sample_{batch_idx}.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Clean up memory
    import gc
    del fig, ax
    gc.collect()
    
    print(f"VAE reconstruction analysis plot saved to {save_path}")


def plot_transfer_entropy_vs_shift(shifts_seconds, kld_values, output_dir):
    """
    Plot transfer entropy (KLD) as a function of temporal shift.
    
    Args:
        shifts_seconds: List of shift values in seconds
        kld_values: List of corresponding KLD values
        output_dir: Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    # Professional scientific paper color palette
    colors = {
        'main_line': "#055C9A",
        'minimum_point': "#BB3E00",
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # Configure scientific paper grid style for both plots
    for ax in [ax1, ax2]:
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
    
    # Main plot
    ax1.plot(shifts_seconds, kld_values, color=colors['main_line'], linewidth=2, 
             marker='o', markersize=3, alpha=0.8, label='Transfer Entropy')
    ax1.set_xlabel('UP Signal Shift (seconds)', fontweight='normal')
    ax1.set_ylabel('Average Transfer Entropy (KLD)', fontweight='normal')
    ax1.set_title('Transfer Entropy vs UP Signal Temporal Shift', fontweight='normal', pad=12)
    
    # Find and mark minimum
    min_idx = np.argmin(kld_values)
    min_shift = shifts_seconds[min_idx]
    min_kld = kld_values[min_idx]
    ax1.plot(min_shift, min_kld, color=colors['minimum_point'], marker='o', markersize=8, 
             label=f'Minimum: {min_shift}s (KLD={min_kld:.6f})', zorder=5)
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.autoscale(enable=True, axis='x', tight=True)
    
    # Add physiological interpretation zones
    ax1.axvspan(-30, -10, alpha=0.1, color='green', label='Early contraction effect')
    ax1.axvspan(-10, 10, alpha=0.1, color='blue', label='Immediate coupling')  
    ax1.axvspan(10, 30, alpha=0.1, color='orange', label='Delayed response')
    
    # Zoomed plot around minimum
    zoom_range = 15  # ±15 seconds around minimum
    zoom_mask = np.abs(np.array(shifts_seconds) - min_shift) <= zoom_range
    zoom_shifts = np.array(shifts_seconds)[zoom_mask]
    zoom_klds = np.array(kld_values)[zoom_mask]
    
    if len(zoom_shifts) > 1:
        ax2.plot(zoom_shifts, zoom_klds, color=colors['main_line'], linewidth=2, 
                 marker='o', markersize=4, alpha=0.8)
        ax2.plot(min_shift, min_kld, color=colors['minimum_point'], marker='o', 
                 markersize=8, zorder=5)
        ax2.set_xlabel('UP Signal Shift (seconds)', fontweight='normal')
        ax2.set_ylabel('Average Transfer Entropy (KLD)', fontweight='normal')
        ax2.set_title(f'Zoomed View: ±{zoom_range}s around minimum', fontweight='normal', pad=12)
        ax2.autoscale(enable=True, axis='x', tight=True)
        
        # Add vertical line at minimum
        ax2.axvline(x=min_shift, color=colors['minimum_point'], linestyle='--', 
                   alpha=0.7, linewidth=1.5)
        
        # Add text annotation for the optimal shift
        ax2.text(min_shift, min_kld, f'  Optimal: {min_shift}s\n  KLD: {min_kld:.6f}', 
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], 
                          alpha=0.9, edgecolor='#A2B9A7'), fontsize=10)
    
    # Add overall title with physiological context
    fig.suptitle('Fetal-Maternal Coupling: Transfer Entropy Analysis', 
                fontsize=14, fontweight='normal', y=0.97, color='#456882')
    
    # Add interpretation text
    interpretation_text = (
        "Negative shifts: UP leads FHR (contraction precedes heart rate change)\n"
        "Positive shifts: FHR leads UP (heart rate change precedes contraction)\n" 
        "Minimum KLD indicates optimal temporal coupling for information transfer"
    )
    
    fig.text(0.02, 0.02, interpretation_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], 
                      alpha=0.9, edgecolor='#A2B9A7'),
             verticalalignment='bottom', horizontalalignment='left')
    
    # Save plot
    plot_path = os.path.join(output_dir, 'transfer_entropy_vs_shift.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Clean up memory
    import gc
    del fig, ax1, ax2
    gc.collect()
    
    return plot_path


def plot_metrics_histograms(vaf_values, mse_values, snr_values, kld_values, output_dir):
    """
    Plot histograms for VAF, MSE, SNR, and KLD metrics using the same style as other plots.
    
    Args:
        vaf_values: List of VAF values
        mse_values: List of MSE values 
        snr_values: List of SNR values in dB
        kld_values: List of KLD values
        output_dir: Directory to save plots
    """
    # Professional scientific paper color palette
    colors = {
        'vaf': '#055C9A',      # Deep blue
        'mse': '#BB3E00',      # Deep orange-red
        'snr': '#0DD8A2',      # Sage green
        'kld': '#F7AD45',      # Golden yellow
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
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()
    
    metrics_data = [
        (vaf_values, 'VAF (Variance Accounted For)', colors['vaf'], 'VAF', axes[0]),
        (mse_values, 'MSE (Mean Squared Error)', colors['mse'], 'MSE', axes[1]), 
        (snr_values, 'SNR (Signal-to-Noise Ratio) [dB]', colors['snr'], 'SNR (dB)', axes[2]),
        (kld_values, 'KLD (Kullback-Leibler Divergence)', colors['kld'], 'KLD', axes[3])
    ]
    
    # Configure scientific paper grid style for all subplots
    for ax in axes:
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
    
    for values, title, color, xlabel, ax in metrics_data:
        if len(values) == 0:
            continue
            
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Plot histogram
        n, bins, patches = ax.hist(values, bins=50, alpha=0.7, color=color, 
                                 edgecolor='white', linewidth=0.5, density=True)
        
        # Add vertical lines for mean and ±1std
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.4f}')
        ax.axvline(mean_val + std_val, color='red', linestyle=':', linewidth=1.5,
                  alpha=0.7, label=f'±1σ: {std_val:.4f}')
        ax.axvline(mean_val - std_val, color='red', linestyle=':', linewidth=1.5,
                  alpha=0.7)
        
        # Styling
        ax.set_title(title, fontweight='normal', fontsize=12, pad=12)
        ax.set_xlabel(xlabel, fontweight='normal')
        ax.set_ylabel('Density', fontweight='normal')
        ax.legend(framealpha=0.95, loc='upper right')
        
        # Add text box with statistics
        stats_text = f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nStd: {std_val:.4f}\nSamples: {len(values)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle('Reconstruction Quality Metrics Distribution', fontsize=14, fontweight='normal', y=0.98, color='#456882')
    
    # Save plot
    save_path = os.path.join(output_dir, 'metrics_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Clean up memory
    import gc
    del fig, axes
    gc.collect()
    
    print(f"Metrics histograms saved to: {save_path}")
    return save_path


def plot_te_ablation_results(kld_with_up, kld_without_up, vaf_with_up, vaf_without_up, output_dir):
    """
    Visualize transfer entropy (KLD) and reconstruction quality (VAF) distributions
    with and without UP input (ablation), using the same plot style.

    Args:
        kld_with_up (List[float]): KLD per-sample with UP features present. e.g. [0.12, 0.08, ...]
        kld_without_up (List[float]): KLD per-sample with UP features ablated (zeros). e.g. [0.01, 0.00, ...]
        vaf_with_up (List[float]): VAF per-sample with UP features present. e.g. [0.65, 0.71, ...]
        vaf_without_up (List[float]): VAF per-sample with UP features ablated. e.g. [0.52, 0.58, ...]
        output_dir (str): Directory to save the figure. e.g. 'output/run_123/test_results'

    Returns:
        str: Path to the saved plot. e.g. '.../te_ablation_results.png'
    """
    import numpy as np

    # Professional scientific paper color palette (consistent styling)
    colors = {
        'kld_up': '#F7AD45',    # Golden yellow
        'kld_no': '#D95319',    # Darker orange/red
        'vaf_up': '#055C9A',    # Deep blue
        'vaf_no': '#0DD8A2',    # Sage green
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
        'axes.edgecolor': '#9E9D9D',
        'axes.facecolor': colors['background'],
        'grid.color': '#838383',
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

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    # Grid/spine style
    for ax in axes:
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

    # 1) KLD histogram overlay
    if len(kld_with_up) > 0 or len(kld_without_up) > 0:
        axes[0].hist(kld_with_up, bins=50, alpha=0.6, color=colors['kld_up'],
                     edgecolor='white', linewidth=0.5, density=True, label='KLD (with UP)')
        axes[0].hist(kld_without_up, bins=50, alpha=0.6, color=colors['kld_no'],
                     edgecolor='white', linewidth=0.5, density=True, label='KLD (ablated)')
        axes[0].set_title('Transfer Entropy (KLD) Distribution')
        axes[0].set_xlabel('KLD')
        axes[0].set_ylabel('Density')
        axes[0].legend(loc='upper right', framealpha=0.95)

    # 2) VAF histogram overlay
    if len(vaf_with_up) > 0 or len(vaf_without_up) > 0:
        axes[1].hist(vaf_with_up, bins=50, alpha=0.6, color=colors['vaf_up'],
                     edgecolor='white', linewidth=0.5, density=True, label='VAF (with UP)')
        axes[1].hist(vaf_without_up, bins=50, alpha=0.6, color=colors['vaf_no'],
                     edgecolor='white', linewidth=0.5, density=True, label='VAF (ablated)')
        axes[1].set_title('Reconstruction Quality (VAF) Distribution')
        axes[1].set_xlabel('VAF')
        axes[1].set_ylabel('Density')
        axes[1].legend(loc='upper right', framealpha=0.95)

    # 3) KLD mean ± std bars
    if len(kld_with_up) > 0 or len(kld_without_up) > 0:
        means = [np.mean(kld_with_up) if len(kld_with_up) else 0.0,
                 np.mean(kld_without_up) if len(kld_without_up) else 0.0]
        stds = [np.std(kld_with_up) if len(kld_with_up) else 0.0,
                np.std(kld_without_up) if len(kld_without_up) else 0.0]
        axes[2].bar([0, 1], means, yerr=stds, color=[colors['kld_up'], colors['kld_no']],
                    alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(['with UP', 'ablated'])
        axes[2].set_title('KLD: Mean ± 1σ')
        axes[2].set_ylabel('KLD')

    # 4) VAF mean ± std bars
    if len(vaf_with_up) > 0 or len(vaf_without_up) > 0:
        means = [np.mean(vaf_with_up) if len(vaf_with_up) else 0.0,
                 np.mean(vaf_without_up) if len(vaf_without_up) else 0.0]
        stds = [np.std(vaf_with_up) if len(vaf_with_up) else 0.0,
                np.std(vaf_without_up) if len(vaf_without_up) else 0.0]
        axes[3].bar([0, 1], means, yerr=stds, color=[colors['vaf_up'], colors['vaf_no']],
                    alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[3].set_xticks([0, 1])
        axes[3].set_xticklabels(['with UP', 'ablated'])
        axes[3].set_title('VAF: Mean ± 1σ')
        axes[3].set_ylabel('VAF')

    fig.suptitle('UP Ablation: Transfer Entropy and Reconstruction', fontsize=14, fontweight='normal', y=0.98, color='#456882')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'te_ablation_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    # Clean up
    import gc
    del fig, axes
    gc.collect()
    print(f"TE ablation results saved to: {save_path}")
    return save_path


def plot_te_gain_sweep(gains, kld_means, vaf_means, output_dir):
    """
    Plot KLD and VAF as a function of scaling applied to UP features.

    Args:
        gains (List[float]): Multiplicative gains applied to UP features. e.g. [0.0, 0.5, 1.0, 1.5]
        kld_means (List[float]): Average KLD per gain. e.g. [0.00, 0.05, 0.10, 0.13]
        vaf_means (List[float]): Average VAF per gain. e.g. [0.55, 0.62, 0.68, 0.67]
        output_dir (str): Directory to save the figure.

    Returns:
        str: Path to the saved plot.
    """
    import numpy as np

    colors = {
        'kld': '#F7AD45',
        'vaf': '#055C9A',
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
        'axes.edgecolor': '#9E9D9D',
        'axes.facecolor': colors['background'],
        'grid.color': '#838383',
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax in (ax1, ax2):
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

    gains_np = np.array(gains, dtype=float)

    # KLD vs gain
    ax1.plot(gains_np, kld_means, color=colors['kld'], marker='o', linewidth=2, label='KLD')
    ax1.set_xlabel('UP Gain')
    ax1.set_ylabel('Average KLD')
    ax1.set_title('Transfer Entropy vs UP Gain')
    ax1.legend(loc='upper left', framealpha=0.95)

    # VAF vs gain
    ax2.plot(gains_np, vaf_means, color=colors['vaf'], marker='o', linewidth=2, label='VAF')
    ax2.set_xlabel('UP Gain')
    ax2.set_ylabel('Average VAF')
    ax2.set_title('Reconstruction Quality vs UP Gain')
    ax2.legend(loc='lower right', framealpha=0.95)

    fig.suptitle('UP Gain Sweep: Information Flow and Reconstruction', fontsize=14, fontweight='normal', y=0.98, color='#456882')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'te_gain_sweep.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    # Clean up
    import gc
    del fig, ax1, ax2
    gc.collect()
    print(f"TE gain sweep saved to: {save_path}")
    return save_path
