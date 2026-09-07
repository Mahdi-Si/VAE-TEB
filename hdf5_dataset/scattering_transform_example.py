#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Examples for Scattering Transform and Phase Harmonics
=====================================================================

This script generates illustrative figures for presentation showing:
1. Standard scattering transform (zeroth, first, second order)
2. Phase harmonic coefficients (within-channel)
3. Cross-channel phase harmonics (UP-FHR coupling)

Outputs are saved as high-quality images suitable for presentation slides.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d
from kymatio_phase_scattering import KymatioPhaseScattering1D

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    pass

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Enable LaTeX rendering for proper mathematical symbols
plt.rcParams['text.usetex'] = False  # Don't require full LaTeX installation
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font for math
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def create_synthetic_fhr_signals(length=4800, fs=4.0):
    """
    Create synthetic FHR and UP signals with realistic characteristics.
    Smoother signals with reduced noise for better visualization.

    Args:
        length (int): Signal length in samples
        fs (float): Sampling frequency in Hz

    Returns:
        dict: Contains 'fhr', 'up', and 'time' arrays
    """
    t = np.arange(length) / fs

    # Set random seed for reproducibility
    np.random.seed(42)

    # UP signal: Slow contractions (0.003-0.008 Hz) - smoother
    contraction_freq1 = 0.005  # ~3 contractions per 10 min
    contraction_freq2 = 0.0035
    contraction_freq3 = 0.007
    up_signal = (
        35 * np.sin(2 * np.pi * contraction_freq1 * t) +
        20 * np.sin(2 * np.pi * contraction_freq2 * t + np.pi/3) +
        10 * np.sin(2 * np.pi * contraction_freq3 * t + np.pi/6) +
        3 * np.random.randn(length)  # Reduced noise from 10 to 3
    )
    up_signal = np.maximum(up_signal, 0)  # Contractions are non-negative

    # Smooth the UP signal slightly
    up_signal = gaussian_filter1d(up_signal, sigma=2)

    # FHR signal: Baseline + variability + coupling to contractions - smoother
    baseline = 140  # bpm

    # Low-frequency variability (0.04-0.15 Hz)
    lf_variability = (
        10 * np.sin(2 * np.pi * 0.08 * t) +
        5 * np.sin(2 * np.pi * 0.12 * t + np.pi/4)
    )

    # Mid-frequency variability (0.15-0.5 Hz)
    mf_variability = (
        5 * np.sin(2 * np.pi * 0.25 * t) +
        3 * np.sin(2 * np.pi * 0.35 * t + np.pi/5) +
        2 * np.sin(2 * np.pi * 0.45 * t)
    )

    # High-frequency component (respiratory)
    hf_variability = 4 * np.sin(2 * np.pi * 0.7 * t + np.pi/3)

    # Coupling: FHR responds to contractions with phase lag
    coupling_response = -6 * np.sin(2 * np.pi * contraction_freq1 * t - np.pi/4)

    # Combine components with reduced noise
    fhr_signal = (
        baseline +
        lf_variability +
        mf_variability +
        hf_variability +
        coupling_response +
        0.8 * np.random.randn(length)  # Reduced noise from 2 to 0.8
    )

    # Smooth the FHR signal slightly
    fhr_signal = gaussian_filter1d(fhr_signal, sigma=1.5)

    return {
        'fhr': torch.tensor(fhr_signal, dtype=torch.float32),
        'up': torch.tensor(up_signal, dtype=torch.float32),
        'time': t
    }


def plot_input_signals(signals, save_path='output/01_input_signals.png'):
    """Plot the input FHR and UP signals."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot UP signal
    axes[0].plot(signals['time'], signals['up'].numpy(), 'b-', linewidth=1.5)
    axes[0].set_ylabel('UP (mmHg)', fontsize=12, fontweight='bold')
    axes[0].set_title('Uterine Pressure Signal', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot FHR signal
    axes[1].plot(signals['time'], signals['fhr'].numpy(), 'r-', linewidth=1.5)
    axes[1].set_ylabel('FHR (bpm)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Fetal Heart Rate Signal', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_wavelet_filters(scattering, save_path='output/02_wavelet_filters.png'):
    """Visualize the wavelet filter bank in time and frequency domain."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Time domain - subset of wavelets
    ax1 = fig.add_subplot(gs[0, :])
    n_display = 8  # Show subset of filters
    step = len(scattering.center_freqs) // n_display
    colors = plt.cm.viridis(np.linspace(0, 1, n_display))

    t_support = np.arange(scattering.psi1_filters.shape[-1])
    for idx, i in enumerate(range(0, len(scattering.center_freqs), step)):
        if idx >= n_display:
            break
        psi = scattering.psi1_filters[i].cpu().numpy()
        psi_real = np.real(np.fft.ifft(psi))
        psi_real = psi_real / np.max(np.abs(psi_real))  # Normalize
        offset = idx * 2.5
        ax1.plot(t_support[:500], psi_real[:500] + offset,
                color=colors[idx], linewidth=1.5,
                label=f'$\\xi={scattering.center_freqs[i].item():.4f}$')

    ax1.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('Wavelet Filter Bank (Time Domain)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Frequency domain - all filters
    ax2 = fig.add_subplot(gs[1, 0])
    freqs = scattering.center_freqs.cpu().numpy()
    freq_axis = np.linspace(0, 0.5, scattering.psi1_filters.shape[-1])

    for i in range(len(scattering.psi1_filters)):
        psi_fft = np.abs(scattering.psi1_filters[i].cpu().numpy())
        ax2.plot(freq_axis, psi_fft, alpha=0.5, linewidth=1)

    ax2.set_xlabel('Normalized Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
    ax2.set_title('Filters in Frequency Domain', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.5])

    # Center frequencies distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(range(len(freqs)), freqs, c=freqs, cmap='viridis', s=80, edgecolors='k')
    ax3.set_xlabel('Filter Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Center Frequency (normalized)', fontsize=12, fontweight='bold')
    ax3.set_title('Filter Center Frequencies', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scattering_cascade(scattering, signals, save_path='output/03_scattering_cascade.png'):
    r"""Visualize the scattering cascade: x -> |x*psi| -> |x*psi|*phi."""
    # Prepare input
    x = torch.stack([signals['up'], signals['fhr']], dim=0).unsqueeze(0)

    # Compute intermediate representations
    x_fhr = x[:, 1, :].contiguous()  # FHR channel

    # Ensure x_fhr is on the same device as scattering
    x_fhr = x_fhr.to(scattering.device)

    # Step 1: Apply wavelets
    x_padded = scattering._pad_signal(x_fhr)
    x_fft = torch.fft.fft(x_padded, dim=-1)
    filtered_fft = x_fft.unsqueeze(1) * scattering.psi1_filters.unsqueeze(0)
    U1 = torch.fft.ifft(filtered_fft, dim=-1)
    U1_unpadded = scattering._unpad_signal(U1, scale=0)

    # Step 2: Apply modulus
    U1_mod = torch.abs(U1_unpadded)

    # Step 3: Apply low-pass filter
    S1 = scattering._apply_phi_filter(U1_mod)

    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Create proper time axis for downsampled scattering coefficients
    time_axis = np.linspace(0, signals['time'][-1], S1.shape[-1])

    # Original signal
    ax_orig = fig.add_subplot(gs[0, :])
    ax_orig.plot(signals['time'], signals['fhr'].numpy(), 'k-', linewidth=1.5)
    ax_orig.set_title('Step 0: Input Signal x(t) [FHR]', fontsize=14, fontweight='bold')
    ax_orig.set_ylabel('Amplitude', fontsize=11)
    ax_orig.grid(True, alpha=0.3)
    ax_orig.set_xlim([0, 100])  # Show first 100 seconds

    # Wavelet convolutions (show 3 examples)
    filter_indices = [5, 15, 25]  # Low, mid, high frequency
    for idx, filt_idx in enumerate(filter_indices):
        ax = fig.add_subplot(gs[1, idx])

        # Plot real and imaginary parts
        U1_real = U1_unpadded[0, filt_idx, :len(time_axis)].real.numpy()
        U1_imag = U1_unpadded[0, filt_idx, :len(time_axis)].imag.numpy()

        ax.plot(time_axis, U1_real, 'b-', alpha=0.7, linewidth=1, label=r'Real')
        ax.plot(time_axis, U1_imag, 'r-', alpha=0.7, linewidth=1, label=r'Imag')
        ax.plot(time_axis, U1_mod[0, filt_idx, :len(time_axis)].numpy(),
               'k-', linewidth=2, label=r'$|U_1|$')

        freq = scattering.center_freqs[filt_idx].item()
        ax.set_title(f'Step 1: $x \\star \\psi_{{{filt_idx}}}$ ($\\xi={freq:.4f}$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])

    # Scattering coefficients (low-pass filtered)
    for idx, filt_idx in enumerate(filter_indices):
        ax = fig.add_subplot(gs[2, idx])

        S1_coeff = S1[0, filt_idx, :].numpy()
        ax.plot(time_axis, S1_coeff, 'g-', linewidth=2)

        freq = scattering.center_freqs[filt_idx].item()
        ax.set_title(f'Step 2: $S_1(\\xi={freq:.4f})$', fontsize=12, fontweight='bold')
        ax.set_ylabel('$S_1$ Coefficient', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scattering_scalogram(scattering, signals, save_path='output/04_scattering_scalogram.png'):
    """Create a scalogram showing all first-order scattering coefficients."""
    x = torch.stack([signals['up'], signals['fhr']], dim=0).unsqueeze(0)

    # Ensure on correct device
    x = x.to(scattering.device)

    # Compute scattering coefficients
    output = scattering.forward(x[:, 1:2, :], compute_phase=False, compute_cross_phase=False)
    S = output['scattering']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Create proper time axis for downsampled coefficients
    time_axis = np.linspace(0, signals['time'][-1], S.shape[-1])

    # Scalogram
    im1 = axes[0].imshow(S[0, 1:, :].numpy(), aspect='auto', origin='lower',
                         cmap='viridis', interpolation='bilinear')
    axes[0].set_ylabel('Scattering Coefficient Index', fontsize=12, fontweight='bold')
    axes[0].set_title(r'First-Order Scattering Scalogram $S_1(t, \lambda)$',
                     fontsize=14, fontweight='bold')

    # Add frequency labels
    n_filters = len(scattering.center_freqs)
    tick_positions = np.linspace(0, S.shape[1]-2, 8, dtype=int)
    tick_labels = [f'{scattering.center_freqs[min(i, n_filters-1)].item():.3f}'
                  for i in tick_positions]
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels(tick_labels)

    plt.colorbar(im1, ax=axes[0], label='Coefficient Value')

    # Select and plot individual coefficients
    coeff_indices = [0, 10, 20, 30, 40]
    colors = plt.cm.tab10(range(len(coeff_indices)))

    for idx, coeff_idx in enumerate(coeff_indices):
        if coeff_idx < S.shape[1]:
            axes[1].plot(time_axis, S[0, coeff_idx, :].numpy(),
                        color=colors[idx], linewidth=2,
                        label=f'$S_1[{coeff_idx}]$', alpha=0.8)

    axes[1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Selected Scattering Coefficients', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_phase_harmonic_concept(scattering, signals, save_path='output/05_phase_harmonic_concept.png'):
    """Illustrate phase acceleration and phase correlation concept."""
    x = signals['fhr'].unsqueeze(0)  # Shape: (1, N)
    x = x.to(scattering.device)

    # Apply filters - this returns (B, n_filters, N) for 2D input
    filtered = scattering._apply_filters(x)

    # Select two filters with harmonic relationship
    # Make sure indices are valid for the number of filters
    n_filters = filtered.shape[1]
    i_idx = min(10, n_filters - 1)
    j_idx = min(20, n_filters - 1)

    U_i = filtered[0, i_idx, :]
    U_j = filtered[0, j_idx, :]

    # Extract amplitude and phase
    A_i = torch.abs(U_i)
    A_j = torch.abs(U_j)
    phi_i = torch.atan2(U_i.imag, U_i.real)
    phi_j = torch.atan2(U_j.imag, U_j.real)

    # Compute harmonic power
    xi_i = scattering.center_freqs[i_idx].item()
    xi_j = scattering.center_freqs[j_idx].item()
    power = xi_j / xi_i

    # Apply phase acceleration
    U_i_accelerated = scattering._accelerate_phase(U_i, power)
    phi_i_acc = torch.atan2(U_i_accelerated.imag, U_i_accelerated.real)

    # Compute phase correlation
    phase_corr = U_i_accelerated * U_j.conj()
    # Add batch and channel dimensions properly: (B, C, N) -> (1, 1, N)
    phase_corr_for_filter = phase_corr.unsqueeze(0)  # Add batch dim
    if phase_corr_for_filter.dim() == 2:
        phase_corr_for_filter = phase_corr_for_filter.unsqueeze(0)  # Add channel dim if needed
    phase_corr_smoothed = scattering._apply_phi_filter(phase_corr_for_filter)

    # Plotting
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    time_plot = signals['time'][:2000]  # First 500 seconds

    # Amplitudes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_plot, A_i[:len(time_plot)].numpy(), 'b-', linewidth=2, label=f'$A_i$ ($\\xi={xi_i:.4f}$)')
    ax1.plot(time_plot, A_j[:len(time_plot)].numpy(), 'r-', linewidth=2, label=f'$A_j$ ($\\xi={xi_j:.4f}$)')
    ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax1.set_title('Wavelet Amplitudes $A_i(t)$, $A_j(t)$', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Original phases
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_plot, phi_i[:len(time_plot)].numpy(), 'b-', linewidth=1.5,
            label=f'$\\phi_i$ ($\\xi={xi_i:.4f}$)', alpha=0.7)
    ax2.plot(time_plot, phi_j[:len(time_plot)].numpy(), 'r-', linewidth=1.5,
            label=f'$\\phi_j$ ($\\xi={xi_j:.4f}$)', alpha=0.7)
    ax2.set_ylabel('Phase (radians)', fontsize=11, fontweight='bold')
    ax2.set_title('Original Instantaneous Phases', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-np.pi, np.pi])

    # Phase acceleration
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_plot, phi_i[:len(time_plot)].numpy(), 'b--', linewidth=1.5,
            label=r'$\phi_i$ (original)', alpha=0.5)
    ax3.plot(time_plot, phi_i_acc[:len(time_plot)].numpy(), 'b-', linewidth=2,
            label=f'$p \\cdot \\phi_i$ ($p={power:.2f}$)')
    ax3.plot(time_plot, phi_j[:len(time_plot)].numpy(), 'r-', linewidth=1.5,
            label=r'$\phi_j$', alpha=0.7)
    ax3.set_ylabel('Phase (radians)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Phase Acceleration: p={power:.2f}', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-np.pi, np.pi])

    # Phase difference
    ax4 = fig.add_subplot(gs[1, 1])
    phase_diff = phi_i_acc - phi_j
    # Wrap to [-pi, pi]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    ax4.plot(time_plot, phase_diff[:len(time_plot)].numpy(), 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Phase Difference (radians)', fontsize=11, fontweight='bold')
    ax4.set_title(r'Phase Difference: $p \cdot \phi_i - \phi_j$', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-np.pi, np.pi])

    # Phase correlation (complex)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time_plot, phase_corr.real[:len(time_plot)].numpy(), 'g-',
            linewidth=1.5, label=r'Re$[\cdot]$', alpha=0.7)
    ax5.plot(time_plot, phase_corr.imag[:len(time_plot)].numpy(), 'orange',
            linewidth=1.5, label=r'Im$[\cdot]$', alpha=0.7)
    ax5.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Correlation Value', fontsize=11, fontweight='bold')
    ax5.set_title(r'Phase Correlation: $U_i^{(p)} \cdot \overline{U_j}$', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Smoothed phase correlation coefficient
    ax6 = fig.add_subplot(gs[2, 1])
    time_smooth = signals['time'][:phase_corr_smoothed.shape[-1]]
    ax6.plot(time_smooth, phase_corr_smoothed[0, 0, :].real.numpy(),
            'darkgreen', linewidth=2.5)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('S_phase Coefficient', fontsize=11, fontweight='bold')
    ax6.set_title(f'Final Coefficient: $S_{{\\mathrm{{phase}}}}({i_idx},{j_idx})$',
                 fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Add text annotation
    textstr = f'Filter pair: ({i_idx}, {j_idx})\n$\\xi_i$ = {xi_i:.4f}, $\\xi_j$ = {xi_j:.4f}\nPower p = {power:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_phase_harmonic_matrix(scattering, signals, save_path='output/06_phase_harmonic_matrix.png'):
    """Visualize the phase harmonic coefficient matrix."""
    x = signals['fhr'].unsqueeze(0).unsqueeze(0)

    # Compute phase harmonics
    output = scattering.forward(x, compute_phase=True, compute_cross_phase=False)
    phase_corr = output['phase_corr']

    # Get optimal coefficient selection
    selection = scattering.get_optimal_coefficients_for_fhr(scattering.J, scattering.Q, scattering.T)
    optimal_mask = selection['recommendations']['use_phase_mask']

    # Create matrix representation
    n_filters = len(scattering.center_freqs)
    phase_matrix = np.zeros((n_filters, n_filters))

    # Fill the matrix with average coefficient values
    for k, (i, j) in enumerate(zip(scattering.i_idx, scattering.j_idx)):
        avg_value = phase_corr[0, k, :].mean().item()
        phase_matrix[i, j] = avg_value

    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Full matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(phase_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(phase_matrix).max(), vmax=np.abs(phase_matrix).max())
    ax1.set_xlabel('Filter Index j', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Filter Index i', fontsize=12, fontweight='bold')
    ax1.set_title('Complete Phase Harmonic Matrix\\n$S_{\\mathrm{phase}}(i,j)$ [averaged over time]',
                 fontsize=14, fontweight='bold')

    # Add frequency labels
    tick_positions = np.linspace(0, n_filters-1, 8, dtype=int)
    tick_labels = [f'{scattering.center_freqs[i].item():.3f}' for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)

    plt.colorbar(im1, ax=ax1, label='Average Phase Correlation')

    # Selected coefficients only
    ax2 = fig.add_subplot(gs[0, 1])

    # Create mask matrix
    mask_matrix = np.zeros((n_filters, n_filters), dtype=bool)
    for k, (i, j) in enumerate(zip(scattering.i_idx, scattering.j_idx)):
        if optimal_mask[k]:
            mask_matrix[i, j] = True

    # Create masked matrix
    selected_matrix = np.ma.masked_where(~mask_matrix, phase_matrix)

    im2 = ax2.imshow(selected_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(phase_matrix).max(), vmax=np.abs(phase_matrix).max())
    ax2.set_xlabel('Filter Index j', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Filter Index i', fontsize=12, fontweight='bold')

    n_selected = optimal_mask.sum().item()
    ax2.set_title(f'Selected Phase Harmonics for FHR\n({n_selected} coefficients)',
                 fontsize=14, fontweight='bold')

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)

    plt.colorbar(im2, ax=ax2, label='Average Phase Correlation')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_channel_concept(scattering, signals, save_path='output/07_cross_channel_concept.png'):
    """Illustrate cross-channel phase correlation between UP and FHR."""
    x = torch.stack([signals['up'], signals['fhr']], dim=0).unsqueeze(0)
    x = x.to(scattering.device)

    # Apply filters to both channels
    filtered = scattering._apply_filters(x)

    # Select filter pair: low freq from UP, higher freq from FHR
    n_filters = filtered.shape[2]
    i_idx = min(15, n_filters - 1)  # UP channel - contraction frequency
    j_idx = min(25, n_filters - 1)  # FHR channel - variability frequency

    U_i_up = filtered[0, 0, i_idx, :]  # Channel 0 (UP)
    U_j_fhr = filtered[0, 1, j_idx, :]  # Channel 1 (FHR)

    # Extract components
    A_i = torch.abs(U_i_up)
    A_j = torch.abs(U_j_fhr)
    phi_i = torch.atan2(U_i_up.imag, U_i_up.real)
    phi_j = torch.atan2(U_j_fhr.imag, U_j_fhr.real)

    # Compute power and acceleration
    xi_i = scattering.center_freqs[i_idx].item()
    xi_j = scattering.center_freqs[j_idx].item()
    power = xi_j / xi_i

    U_i_acc = scattering._accelerate_phase(U_i_up, power)

    # Cross-channel correlation
    cross_corr = U_i_acc * U_j_fhr.conj()
    # Add batch and channel dimensions properly
    cross_corr_for_filter = cross_corr.unsqueeze(0)  # Add batch dim
    if cross_corr_for_filter.dim() == 2:
        cross_corr_for_filter = cross_corr_for_filter.unsqueeze(0)  # Add channel dim if needed
    cross_corr_smoothed = scattering._apply_phi_filter(cross_corr_for_filter)

    # Plotting
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    time_plot = signals['time'][:2000]

    # Original signals in the selected frequency bands
    ax1 = fig.add_subplot(gs[0, :])
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(time_plot, signals['up'][:len(time_plot)].numpy(),
                    'b-', linewidth=2, label='UP Signal', alpha=0.7)
    line2 = ax1_twin.plot(time_plot, signals['fhr'][:len(time_plot)].numpy(),
                         'r-', linewidth=2, label='FHR Signal', alpha=0.7)

    ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('UP (mmHg)', fontsize=11, fontweight='bold', color='b')
    ax1_twin.set_ylabel('FHR (bpm)', fontsize=11, fontweight='bold', color='r')
    ax1.set_title('Original Two-Channel Signals', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    # Filtered amplitudes from both channels
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_plot, A_i[:len(time_plot)].numpy(), 'b-', linewidth=2,
            label=f'UP: $A_i$ ($\\xi={xi_i:.4f}$)')
    ax2.plot(time_plot, A_j[:len(time_plot)].numpy(), 'r-', linewidth=2,
            label=f'FHR: $A_j$ ($\\xi={xi_j:.4f}$)')
    ax2.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax2.set_title('Filtered Amplitudes from Each Channel', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Phases from both channels
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_plot, phi_i[:len(time_plot)].numpy(), 'b-', linewidth=1.5,
            label=r'UP: $\phi_i$', alpha=0.7)
    ax3.plot(time_plot, phi_j[:len(time_plot)].numpy(), 'r-', linewidth=1.5,
            label=r'FHR: $\phi_j$', alpha=0.7)
    ax3.set_ylabel('Phase (radians)', fontsize=11, fontweight='bold')
    ax3.set_title('Instantaneous Phases', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-np.pi, np.pi])

    # Cross-channel phase difference
    ax4 = fig.add_subplot(gs[2, 0])
    phi_i_acc = torch.atan2(U_i_acc.imag, U_i_acc.real)
    phase_diff = phi_i_acc - phi_j
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    ax4.plot(time_plot, phase_diff[:len(time_plot)].numpy(), 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.fill_between(time_plot, -np.pi/2, np.pi/2, alpha=0.2, color='green',
                     label='Coupling region')
    ax4.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Phase Difference (radians)', fontsize=11, fontweight='bold')
    ax4.set_title(r'Cross-Channel Phase Difference: UP $\rightarrow$ FHR', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-np.pi, np.pi])

    # Final cross-channel coefficient
    ax5 = fig.add_subplot(gs[2, 1])
    time_smooth = signals['time'][:cross_corr_smoothed.shape[-1]]
    ax5.plot(time_smooth, cross_corr_smoothed[0, 0, :].real.numpy(),
            'darkmagenta', linewidth=2.5)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('S_cross Coefficient', fontsize=11, fontweight='bold')
    ax5.set_title(f'Cross-Channel Coefficient: $S_{{\\mathrm{{cross}}}}({i_idx},{j_idx})$',
                 fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Add annotation
    textstr = f'UP filter: i={i_idx} ($\\xi={xi_i:.4f}$)\nFHR filter: j={j_idx} ($\\xi={xi_j:.4f}$)\nPower p = {power:.2f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cross_channel_matrix(scattering, signals, save_path='output/08_cross_channel_matrix.png'):
    """Visualize the cross-channel phase correlation matrix."""
    x = torch.stack([signals['up'], signals['fhr']], dim=0).unsqueeze(0)
    x = x.to(scattering.device)

    # Compute cross-channel phase harmonics
    output = scattering.forward(x, compute_phase=False, compute_cross_phase=True)
    cross_phase = output['cross_phase_corr']

    # Get optimal coefficient selection
    selection = scattering.get_optimal_coefficients_for_fhr(scattering.J, scattering.Q, scattering.T)
    cross_mask = selection['recommendations']['use_cross_mask']
    up_band_mask = selection['cross_selection']['up_band_mask']
    fhr_band_mask = selection['cross_selection']['fhr_band_mask']

    # Create matrix
    n_filters = len(scattering.center_freqs)
    cross_matrix = np.zeros((n_filters, n_filters))

    for k, (i, j) in enumerate(zip(scattering.i_idx, scattering.j_idx)):
        avg_value = cross_phase[0, k, :].mean().item()
        cross_matrix[i, j] = avg_value

    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Full matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(cross_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(cross_matrix).max(), vmax=np.abs(cross_matrix).max())
    ax1.set_xlabel('FHR Filter Index j', fontsize=12, fontweight='bold')
    ax1.set_ylabel('UP Filter Index i', fontsize=12, fontweight='bold')
    ax1.set_title('Complete Cross-Channel Matrix\\n$S_{\\mathrm{cross}}(\\mathrm{UP}_i, \\mathrm{FHR}_j)$ [averaged over time]',
                 fontsize=14, fontweight='bold')

    tick_positions = np.linspace(0, n_filters-1, 8, dtype=int)
    tick_labels = [f'{scattering.center_freqs[i].item():.3f}' for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)

    plt.colorbar(im1, ax=ax1, label='Average Cross-Channel Correlation')

    # Selected coefficients with frequency band highlighting
    ax2 = fig.add_subplot(gs[0, 1])

    # Create mask matrix
    mask_matrix = np.zeros((n_filters, n_filters), dtype=bool)
    for k, (i, j) in enumerate(zip(scattering.i_idx, scattering.j_idx)):
        if cross_mask[k]:
            mask_matrix[i, j] = True

    selected_matrix = np.ma.masked_where(~mask_matrix, cross_matrix)

    im2 = ax2.imshow(selected_matrix, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(cross_matrix).max(), vmax=np.abs(cross_matrix).max())

    # Highlight frequency bands
    up_indices = torch.where(up_band_mask)[0].cpu().numpy()
    fhr_indices = torch.where(fhr_band_mask)[0].cpu().numpy()

    if len(up_indices) > 0:
        ax2.add_patch(Rectangle((0, up_indices[0]), n_filters, len(up_indices),
                                fill=False, edgecolor='blue', linewidth=3,
                                linestyle='--', label='UP band'))

    if len(fhr_indices) > 0:
        ax2.add_patch(Rectangle((fhr_indices[0], 0), len(fhr_indices), n_filters,
                                fill=False, edgecolor='red', linewidth=3,
                                linestyle='--', label='FHR band'))

    ax2.set_xlabel('FHR Filter Index j', fontsize=12, fontweight='bold')
    ax2.set_ylabel('UP Filter Index i', fontsize=12, fontweight='bold')

    n_selected = cross_mask.sum().item()
    ax2.set_title(f'Selected Cross-Channel Coefficients\n({n_selected} coefficients)',
                 fontsize=14, fontweight='bold')

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)

    ax2.legend(loc='upper right', fontsize=10)
    plt.colorbar(im2, ax=ax2, label='Average Cross-Channel Correlation')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_complete_feature_summary(scattering, signals, save_path='output/09_complete_features.png'):
    """Summary visualization showing all feature types together."""
    x = torch.stack([signals['up'], signals['fhr']], dim=0).unsqueeze(0)
    x = x.to(scattering.device)

    # Compute scattering features
    output_scat = scattering.forward(x, compute_phase=False, compute_cross_phase=False,
                                     scattering_channel=1)  # Use FHR for scattering

    # Compute phase features (within-channel on FHR)
    output_phase = scattering.forward(x[:, 1:2, :], compute_phase=True, compute_cross_phase=False)

    # Compute cross-channel features
    output_cross = scattering.forward(x, compute_phase=False, compute_cross_phase=True)

    # Get selection
    selection = scattering.get_optimal_coefficients_for_fhr(scattering.J, scattering.Q, scattering.T)

    # Extract features
    S = output_scat['scattering']
    phase = output_phase['phase_corr'][:, selection['recommendations']['use_phase_mask'], :]
    cross = output_cross['cross_phase_corr'][:, selection['recommendations']['use_cross_mask'], :]

    # Create summary figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    time_axis = signals['time'][:S.shape[-1]]

    # Original signals
    ax1 = fig.add_subplot(gs[0, :])
    ax1_twin = ax1.twinx()
    ax1.plot(signals['time'], signals['up'].numpy(), 'b-', linewidth=1.5, alpha=0.7, label='UP')
    ax1_twin.plot(signals['time'], signals['fhr'].numpy(), 'r-', linewidth=1.5, alpha=0.7, label='FHR')
    ax1.set_ylabel('UP (mmHg)', color='b', fontweight='bold')
    ax1_twin.set_ylabel('FHR (bpm)', color='r', fontweight='bold')
    ax1.set_title('Input Signals', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Scattering coefficients
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(S[0, :, :].numpy(), aspect='auto', cmap='viridis',
                     interpolation='bilinear', origin='lower')
    ax2.set_ylabel('Coefficient Index', fontweight='bold')
    ax2.set_title(f'Scattering Coefficients ({S.shape[1]} coefficients)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Value')

    # Phase harmonic coefficients
    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(phase[0, :, :].numpy(), aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(phase).max(), vmax=np.abs(phase).max(),
                     interpolation='bilinear', origin='lower')
    ax3.set_ylabel('Coefficient Index', fontweight='bold')
    ax3.set_title(f'Phase Harmonics ({phase.shape[1]} coefficients)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Value')

    # Cross-channel coefficients
    ax4 = fig.add_subplot(gs[2, 1])
    im4 = ax4.imshow(cross[0, :, :].numpy(), aspect='auto', cmap='RdBu_r',
                     vmin=-np.abs(cross).max(), vmax=np.abs(cross).max(),
                     interpolation='bilinear', origin='lower')
    ax4.set_ylabel('Coefficient Index', fontweight='bold')
    ax4.set_title(f'Cross-Channel Phase ({cross.shape[1]} coefficients)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Value')

    # Feature count summary
    ax5 = fig.add_subplot(gs[3, :])
    feature_types = [r'Scattering\n$(S_0, S_1, S_2)$', 'Phase\nHarmonics',
                    'Cross-Channel\nPhase', 'Total\nFeatures']
    feature_counts = [S.shape[1], phase.shape[1], cross.shape[1],
                     S.shape[1] + phase.shape[1] + cross.shape[1]]
    colors_bar = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

    bars = ax5.bar(feature_types, feature_counts, color=colors_bar, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Number of Coefficients', fontsize=12, fontweight='bold')
    ax5.set_title('Feature Set Summary', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_frequency_band_analysis(scattering, save_path='output/10_frequency_bands.png'):
    """Visualize frequency band selection for FHR analysis."""
    freqs = scattering.center_freqs.cpu().numpy()

    # Get selection
    selection = scattering.get_optimal_coefficients_for_fhr(scattering.J, scattering.Q, scattering.T)

    # Frequency bands
    up_max = 0.02
    fhr_min = 0.04
    fhr_max = 0.5
    phase_min = 0.006

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Frequency distribution
    ax1 = axes[0]
    ax1.scatter(range(len(freqs)), freqs, s=100, c='gray', alpha=0.5, label='All filters')

    # Highlight bands
    up_band = freqs < up_max
    fhr_band = (freqs >= fhr_min) & (freqs <= fhr_max)
    phase_band = freqs >= phase_min

    ax1.scatter(np.where(up_band)[0], freqs[up_band], s=150, c='blue',
               label=f'UP band (<{up_max} Hz)', zorder=3)
    ax1.scatter(np.where(fhr_band)[0], freqs[fhr_band], s=150, c='red',
               label=f'FHR band ({fhr_min}-{fhr_max} Hz)', zorder=3)

    ax1.axhline(y=up_max, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=fhr_min, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=fhr_max, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=phase_min, color='green', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Filter Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Center Frequency (normalized)', fontsize=12, fontweight='bold')
    ax1.set_title('Filter Bank Frequency Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Coefficient selection statistics
    ax2 = axes[1]

    categories = ['Total\nPossible', 'Selected\nPhase', 'Selected\nCross-Channel']
    total_phase = len(scattering.i_idx)
    selected_phase = selection['recommendations']['use_phase_mask'].sum().item()
    selected_cross = selection['recommendations']['use_cross_mask'].sum().item()

    counts = [total_phase, selected_phase, selected_cross]
    colors = ['gray', '#e74c3c', '#9b59b6']

    bars = ax2.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Number of Coefficients', fontsize=12, fontweight='bold')
    ax2.set_title('Coefficient Selection Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if count == total_phase:
            label = f'{int(count)}\n(100%)'
        else:
            pct = 100 * count / total_phase
            label = f'{int(count)}\n({pct:.1f}%)'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_first_order_scattering(scattering, signals, save_path='output/11_first_order_scattering.png'):
    """Detailed visualization of first-order scattering coefficients."""
    x_fhr = signals['fhr'].unsqueeze(0)
    x_fhr = x_fhr.to(scattering.device)

    # Compute first-order scattering
    output = scattering.forward(x_fhr, compute_phase=False, compute_cross_phase=False)
    S = output['scattering']

    # Get metadata
    meta = scattering.meta()

    # meta is a dict with arrays, not a list
    orders = meta['order']
    xis = meta['xi']

    # Separate zeroth and first order
    order_0_idx = [i for i, order in enumerate(orders) if order == 0]
    order_1_idx = [i for i, order in enumerate(orders) if order == 1]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Create proper time axis
    time_axis = np.linspace(0, signals['time'][-1], S.shape[-1])

    # Original signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(signals['time'], signals['fhr'].numpy(), 'k-', linewidth=1.5)
    ax1.set_ylabel('FHR (bpm)', fontsize=12, fontweight='bold')
    ax1.set_title('Input Signal: Fetal Heart Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 200])

    # Zeroth-order (low-pass average)
    ax2 = fig.add_subplot(gs[1, 0])
    if len(order_0_idx) > 0:
        S0 = S[0, order_0_idx[0], :].numpy()
        ax2.plot(time_axis, S0, 'darkblue', linewidth=3)
        ax2.set_ylabel(r'$S_0$ Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title(r'Zeroth-Order: $S_0 = x \star \phi$', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 200])

    # First-order scalogram
    ax3 = fig.add_subplot(gs[1, 1])
    if len(order_1_idx) > 0:
        S1 = S[0, order_1_idx, :].numpy()
        im = ax3.imshow(S1, aspect='auto', origin='lower', cmap='viridis',
                       interpolation='bilinear', extent=[time_axis[0], time_axis[-1], 0, len(order_1_idx)])
        ax3.set_ylabel('Filter Index', fontsize=12, fontweight='bold')
        ax3.set_title(r'First-Order Scalogram: $S_1 = |x \star \psi_\lambda| \star \phi$',
                     fontsize=13, fontweight='bold')
        ax3.set_xlim([0, 200])
        plt.colorbar(im, ax=ax3, label='Coefficient Value')

    # Select representative first-order coefficients
    ax4 = fig.add_subplot(gs[2, :])
    if len(order_1_idx) > 0:
        # Select coefficients spanning different frequencies
        indices_to_plot = [0, len(order_1_idx)//4, len(order_1_idx)//2,
                          3*len(order_1_idx)//4, len(order_1_idx)-1]
        colors = plt.cm.tab10(range(len(indices_to_plot)))

        for idx, coeff_idx in enumerate(indices_to_plot):
            if coeff_idx < len(order_1_idx):
                meta_idx = order_1_idx[coeff_idx]
                freq = xis[meta_idx, 0]  # First xi value

                ax4.plot(time_axis, S[0, meta_idx, :].numpy(),
                        color=colors[idx], linewidth=2,
                        label=f'$S_1$ at $\\xi={freq:.4f}$', alpha=0.8)

        ax4.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax4.set_title('Selected First-Order Coefficients', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10, ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 200])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_second_order_scattering(scattering, signals, save_path='output/12_second_order_scattering.png'):
    """Detailed visualization of second-order scattering coefficients."""
    x_fhr = signals['fhr'].unsqueeze(0)
    x_fhr = x_fhr.to(scattering.device)

    # Compute scattering
    output = scattering.forward(x_fhr, compute_phase=False, compute_cross_phase=False)
    S = output['scattering']

    # Get metadata
    meta = scattering.meta()

    # meta is a dict with arrays
    orders = meta['order']
    xis = meta['xi']

    # Separate orders
    order_1_idx = [i for i, order in enumerate(orders) if order == 1]
    order_2_idx = [i for i, order in enumerate(orders) if order == 2]

    if len(order_2_idx) == 0:
        print("Warning: No second-order coefficients found. Skipping this visualization.")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Create proper time axis
    time_axis = np.linspace(0, signals['time'][-1], S.shape[-1])

    # First-order scalogram for reference
    ax1 = fig.add_subplot(gs[0, :])
    S1 = S[0, order_1_idx, :].numpy()
    im1 = ax1.imshow(S1, aspect='auto', origin='lower', cmap='viridis',
                    interpolation='bilinear', extent=[time_axis[0], time_axis[-1], 0, len(order_1_idx)])
    ax1.set_ylabel('Filter Index', fontsize=12, fontweight='bold')
    ax1.set_title(r'First-Order Reference: $S_1 = |x \star \psi_{\lambda_1}| \star \phi$',
                 fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 200])
    plt.colorbar(im1, ax=ax1, label='Coefficient Value')

    # Second-order scalogram
    ax2 = fig.add_subplot(gs[1, :])
    S2 = S[0, order_2_idx, :].numpy()
    im2 = ax2.imshow(S2, aspect='auto', origin='lower', cmap='plasma',
                    interpolation='bilinear', extent=[time_axis[0], time_axis[-1], 0, len(order_2_idx)])
    ax2.set_ylabel('Coefficient Pair Index', fontsize=12, fontweight='bold')
    ax2.set_title(r'Second-Order Scalogram: $S_2 = ||x \star \psi_{\lambda_1}| \star \psi_{\lambda_2}| \star \phi$',
                 fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 200])
    plt.colorbar(im2, ax=ax2, label='Coefficient Value')

    # Select representative second-order coefficients
    ax3 = fig.add_subplot(gs[2, :])
    # Select diverse pairs
    indices_to_plot = np.linspace(0, len(order_2_idx)-1, min(6, len(order_2_idx)), dtype=int)
    colors = plt.cm.tab10(range(len(indices_to_plot)))

    for idx, coeff_idx in enumerate(indices_to_plot):
        meta_idx = order_2_idx[coeff_idx]
        xi1 = xis[meta_idx, 0]
        xi2 = xis[meta_idx, 1]

        ax3.plot(time_axis, S[0, meta_idx, :].numpy(),
                color=colors[idx], linewidth=2,
                label=f'$S_2(\\xi_1={xi1:.3f}, \\xi_2={xi2:.3f})$', alpha=0.8)

    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax3.set_title('Selected Second-Order Coefficients', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 200])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_phase_harmonics_detail(scattering, signals, save_path='output/13_phase_harmonics_detail.png'):
    """Detailed visualization of phase harmonic coefficients."""
    x_fhr = signals['fhr'].unsqueeze(0)
    x_fhr = x_fhr.to(scattering.device)

    # Compute phase harmonics
    output = scattering.forward(x_fhr, compute_phase=True, compute_cross_phase=False)
    phase_corr = output['phase_corr']

    # Get optimal selection
    selection = scattering.get_optimal_coefficients_for_fhr(scattering.J, scattering.Q, scattering.T)
    optimal_mask = selection['recommendations']['use_phase_mask']

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Create proper time axis
    time_axis = np.linspace(0, signals['time'][-1], phase_corr.shape[-1])

    # All phase harmonics
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(phase_corr[0, :, :].numpy(), aspect='auto', origin='lower', cmap='RdBu_r',
                    vmin=-np.abs(phase_corr).max(), vmax=np.abs(phase_corr).max(),
                    interpolation='bilinear', extent=[time_axis[0], time_axis[-1], 0, phase_corr.shape[1]])
    ax1.set_ylabel('Coefficient Pair Index', fontsize=12, fontweight='bold')
    ax1.set_title(r'All Phase Harmonic Coefficients: $\Phi_{ij} = U_i^{(p)} \cdot \overline{U_j}$ (smoothed)',
                 fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 200])
    plt.colorbar(im1, ax=ax1, label='Phase Correlation')

    # Selected phase harmonics for FHR
    ax2 = fig.add_subplot(gs[1, :])
    phase_selected = phase_corr[:, optimal_mask, :]
    im2 = ax2.imshow(phase_selected[0, :, :].numpy(), aspect='auto', origin='lower', cmap='RdBu_r',
                    vmin=-np.abs(phase_selected).max(), vmax=np.abs(phase_selected).max(),
                    interpolation='bilinear', extent=[time_axis[0], time_axis[-1], 0, phase_selected.shape[1]])
    ax2.set_ylabel('Selected Coefficient Index', fontsize=12, fontweight='bold')
    ax2.set_title(f'Optimal Phase Harmonics for FHR ({optimal_mask.sum().item()} coefficients)',
                 fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 200])
    plt.colorbar(im2, ax=ax2, label='Phase Correlation')

    # Individual phase harmonic coefficients
    ax3 = fig.add_subplot(gs[2, :])
    # Plot a few representative coefficients
    selected_indices = torch.where(optimal_mask)[0]
    plot_indices = selected_indices[::max(1, len(selected_indices)//6)][:6]
    colors = plt.cm.tab10(range(len(plot_indices)))

    for idx, coeff_idx in enumerate(plot_indices):
        i = scattering.i_idx[coeff_idx].item()
        j = scattering.j_idx[coeff_idx].item()
        power = scattering.powers[coeff_idx].item()
        xi_i = scattering.center_freqs[i].item()
        xi_j = scattering.center_freqs[j].item()

        ax3.plot(time_axis, phase_corr[0, coeff_idx, :].numpy(),
                color=colors[idx], linewidth=2,
                label=f'$\\Phi({i},{j})$: $\\xi_i={xi_i:.3f}$, $p={power:.1f}$', alpha=0.8)

    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Phase Correlation Value', fontsize=12, fontweight='bold')
    ax3.set_title('Selected Phase Harmonic Time Series', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 200])

    # Add statistics text
    stats_text = f"Total pairs: {len(scattering.i_idx)}\n"
    stats_text += f"Selected: {optimal_mask.sum().item()}\n"
    stats_text += f"Reduction: {100*(1-optimal_mask.sum().item()/len(scattering.i_idx)):.1f}%"
    ax3.text(0.98, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all visualization examples."""
    import os

    # Create output directory
    os.makedirs('output', exist_ok=True)

    print("="*60)
    print("Generating Scattering Transform Visualizations")
    print("="*60)

    # Initialize scattering transform
    print("\n1. Initializing scattering transform...")
    scattering = KymatioPhaseScattering1D(
        J=11,
        Q=4,
        T=16,
        shape=4800,
        tukey_alpha=0.15
    )
    print(f"   - Filters: {len(scattering.center_freqs)}")
    print(f"   - Phase pairs: {len(scattering.i_idx)}")

    # Generate synthetic signals
    print("\n2. Creating synthetic FHR/UP signals...")
    signals = create_synthetic_fhr_signals(length=4800, fs=4.0)
    print(f"   - Signal length: {len(signals['fhr'])} samples (20 minutes @ 4 Hz)")

    # Generate visualizations
    print("\n3. Generating visualizations...")
    print("   [1/13] Input signals...")
    plot_input_signals(signals)

    print("   [2/13] Wavelet filter bank...")
    plot_wavelet_filters(scattering)

    print("   [3/13] Scattering cascade...")
    plot_scattering_cascade(scattering, signals)

    print("   [4/13] Scattering scalogram...")
    plot_scattering_scalogram(scattering, signals)

    print("   [5/13] Phase harmonic concept...")
    plot_phase_harmonic_concept(scattering, signals)

    print("   [6/13] Phase harmonic matrix...")
    plot_phase_harmonic_matrix(scattering, signals)

    print("   [7/13] Cross-channel concept...")
    plot_cross_channel_concept(scattering, signals)

    print("   [8/13] Cross-channel matrix...")
    plot_cross_channel_matrix(scattering, signals)

    print("   [9/13] Complete feature summary...")
    plot_complete_feature_summary(scattering, signals)

    print("   [10/13] Frequency band analysis...")
    plot_frequency_band_analysis(scattering)

    print("   [11/13] First-order scattering detail...")
    plot_first_order_scattering(scattering, signals)

    print("   [12/13] Second-order scattering detail...")
    plot_second_order_scattering(scattering, signals)

    print("   [13/13] Phase harmonics detail...")
    plot_phase_harmonics_detail(scattering, signals)

    print("\n" + "="*60)
    print("All visualizations complete!")
    print("Output saved in: ./output/")
    print("="*60)


if __name__ == "__main__":
    main()
