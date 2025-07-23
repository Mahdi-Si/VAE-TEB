import torch
import torch.nn as nn
import numpy as np
import math

from kymatio.torch import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.scattering1d.utils import compute_border_indices, compute_padding, compute_minimum_support_to_pad


class KymatioPhaseScattering1D(nn.Module):
    """
    1D Scattering Transform with Phase Harmonics and Cross-Channel Phase Correlation.
    
    Combines kymatio's scattering coefficients with phase correlation features.
    Supports both within-channel and cross-channel phase correlations with 
    proper boundary handling following kymatio's approach.
    
    OPTIMAL COEFFICIENT SELECTION FOR FHR ANALYSIS:
    =============================================
    This class includes specialized methods for automatic coefficient selection 
    optimized for Fetal Heart Rate (FHR) analysis and prediction tasks.
    
    Key Features:
    - Physiologically-motivated coefficient selection for FHR-UP analysis
    - Clinical frequency band filtering (>0.006 Hz for 2-minute prediction relevance)
    - Cross-channel analysis for contraction-FHR relationships
    - Efficient dimensionality reduction while preserving clinical information
    
    For J=11, Q=4, T=16 configuration:
    - Reduces phase coefficients by ~95% (from 903 to ~44 relevant pairs)
    - Selects ~130 cross-channel coefficients for UP→FHR coupling
    - Focuses on contraction frequencies (<0.02 Hz) and FHR variability (0.04-0.5 Hz)
    - Total feature reduction from ~232 to ~219 coefficients with better clinical relevance
    
    Usage for FHR Analysis:
    ----------------------
    scattering = KymatioPhaseScattering1D(J=11, Q=4, T=16, shape=4800)
    selection = scattering.get_optimal_coefficients_for_fhr(11, 4, 16)
    
    # Apply selection masks to computed coefficients
    output = scattering.forward(x, compute_phase=True, compute_cross_phase=True)
    selected_phase = output['phase_corr'][:, selection['recommendations']['use_phase_mask'], :]
    selected_cross = output['cross_phase_corr'][:, selection['recommendations']['use_cross_mask'], :]

    Args:
        J (int): Maximum scattering scale (octaves).
        Q (int): Number of wavelets per octave.
        T (int): Temporal support of low-pass filter.
        shape (int): Input signal length.
        device (torch.device, optional): Computation device.
        oversampling (int, optional): Oversampling factor. Defaults to 0.
        max_order (int, optional): Maximum scattering order (1 or 2). Defaults to 2.
        border_mode (str, optional): Padding mode. Defaults to 'reflect'.
        tukey_alpha (float, optional): Alpha parameter for the Tukey window applied to the
            input signal to reduce edge effects. If None, no window is applied. 
            A value between 0.1 and 0.25 is a good starting point. Defaults to None.
    """
    
    def __init__(
        self, J, Q, T, shape, device=None, oversampling=0, max_order=2, 
        border_mode='reflect', tukey_alpha=None):
        super().__init__()
        self.J = J
        
        # Kymatio's Scattering1D uses (Q, 1) for an integer Q. To allow full
        # control over both first and second-order Q values, this class now
        # accepts a tuple for Q. If a tuple (Q1, Q2) is passed, it is used
        # for scattering, and Q1 is used for the phase part. If an integer
        # is passed, it's used for both (and becomes (Q, 1) in scattering).
        if isinstance(Q, tuple):
            self.Q_scattering = Q
            self.Q = Q[0]
        else:
            self.Q_scattering = Q
            self.Q = Q
            
        self.T = T
        self.oversampling = oversampling
        self.max_order = max_order
        self.border_mode = border_mode
        self.tukey_alpha = tukey_alpha
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-14
        
        self.N = int(shape) if isinstance(shape, (int, float)) else int(shape[0])
        
        # Compute padding parameters using kymatio's exact approach
        self._compute_padding_parameters()
        
        # Initialize kymatio scattering transform
        self.scattering = Scattering1D(
            J=J, shape=shape, Q=self.Q_scattering, max_order=max_order, average=True,
            oversampling=oversampling, vectorize=True, out_type='array', T=T
        ).to(self.device)
        
        # Build phase filters using kymatio's filter factory
        self._build_phase_filters()
        
    def _compute_padding_parameters(self):
        """Compute padding parameters using kymatio's exact approach."""
        # Use kymatio's minimum support computation for consistency
        min_to_pad = compute_minimum_support_to_pad(self.N, self.J, self.Q, self.T)
        min_to_pad = min(min_to_pad, self.N - 1)
        
        # Compute J_pad as in kymatio
        J_max_support = int(np.floor(np.log2(3 * self.N - 2)))
        self.J_pad = min(int(np.ceil(np.log2(self.N + 2 * min_to_pad))), J_max_support)
        
        # Use kymatio's padding and border computation
        self.pad_left, self.pad_right = compute_padding(self.J_pad, self.N)
        self.ind_start, self.ind_end = compute_border_indices(self.J, self.pad_left, self.pad_left + self.N)
        self.N_padded = 2 ** self.J_pad
        
    def _build_phase_filters(self):
        """Build filters using kymatio's filter factory."""
        J_support = int(np.ceil(np.log2(self.N_padded)))
        phi_f, psi1_f, _, _ = scattering_filter_factory(
            J_support=J_support, J_scattering=self.J, Q=self.Q, T=self.T
        )
        
        # Extract and register filters as buffers
        filters = np.stack([d['levels'][0] for d in psi1_f], axis=0)
        self.register_buffer('psi1_filters', torch.from_numpy(filters).to(torch.complex64).to(self.device))
        self.register_buffer('phi_filter', torch.from_numpy(phi_f['levels'][0]).to(torch.complex64).to(self.device))
        
        # Extract center frequencies
        center_freqs = torch.tensor([d['xi'] for d in psi1_f], dtype=torch.float32, device=self.device)
        self.register_buffer('center_freqs', center_freqs)
        
        # Precompute phase coupling indices with correct powers
        self._build_coupling_indices()
        
    def _build_coupling_indices(self):
        """Build indices for phase coupling pairs with harmonically related powers."""
        n_filters = len(self.center_freqs)
        indices = []
        powers = []
        
        # Build only valid pairs where xi_j >= xi_i (following kymatio convention)
        for i in range(n_filters):
            xi_i = self.center_freqs[i]
            for j in range(n_filters):
                xi_j = self.center_freqs[j]
                if xi_j >= xi_i:  # Only consider valid pairs
                    indices.append((i, j))
                    # Use harmonic ratios for phase acceleration
                    if xi_i > 1e-8:  # Avoid division by zero
                        power = xi_j / xi_i
                    else:
                        power = 1.0
                    powers.append(power)
        
        i_idx, j_idx = zip(*indices) if indices else ([], [])
        autoc_idx = [k for k, (i, j) in enumerate(indices) if i == j]
        
        self.register_buffer('i_idx', torch.tensor(i_idx, dtype=torch.long, device=self.device))
        self.register_buffer('j_idx', torch.tensor(j_idx, dtype=torch.long, device=self.device))  
        self.register_buffer('powers', torch.tensor(powers, dtype=torch.float32, device=self.device))
        self.register_buffer('autoc_idx', torch.tensor(autoc_idx, dtype=torch.long, device=self.device))
        
    def _pad_signal(self, x):
        """Pad signal using specified border mode."""
        pad_modes = {
            'reflect': lambda: self._reflect_pad(x),
            'constant': lambda: torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode='constant', value=0),
            'circular': lambda: torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode='circular')
        }
        
        if self.border_mode not in pad_modes:
            raise ValueError(f"Unsupported border_mode: {self.border_mode}")
        return pad_modes[self.border_mode]()
    
    def _reflect_pad(self, x):
        """Handle reflection padding with large pad sizes iteratively."""
        signal_len = x.shape[-1]
        
        # If the signal is empty or has only one sample, reflection is not well-defined.
        # We'll fall back to zero-padding in this edge case.
        if signal_len <= 1:
            return torch.nn.functional.pad(x, (self.pad_left, self.pad_right), mode='constant', value=0)

        # Iteratively apply reflection padding in chunks smaller than the signal length.
        # This correctly builds up the reflection for any padding size.
        padded_x = x
        
        # Left padding
        remaining_pad = self.pad_left
        current_signal_len = padded_x.shape[-1]
        while remaining_pad > 0:
            chunk_size = min(remaining_pad, current_signal_len - 1)
            padded_x = torch.nn.functional.pad(padded_x, (chunk_size, 0), mode='reflect')
            remaining_pad -= chunk_size
            current_signal_len = padded_x.shape[-1]

        # Right padding
        remaining_pad = self.pad_right
        current_signal_len = padded_x.shape[-1]
        while remaining_pad > 0:
            chunk_size = min(remaining_pad, current_signal_len - 1)
            padded_x = torch.nn.functional.pad(padded_x, (0, chunk_size), mode='reflect')
            remaining_pad -= chunk_size
            current_signal_len = padded_x.shape[-1]
            
        return padded_x
    
    def _unpad_signal(self, x, scale=0):
        """Remove padding using precomputed boundary indices."""
        return x[..., self.ind_start[scale]:self.ind_end[scale]]
        
    def _accelerate_phase(self, x, power):
        """Phase acceleration: A*exp(i*phi) -> A*exp(i*power*phi)."""
        # More numerically stable polar coordinate approach following kymatio
        magnitude = x.abs()
        phase = torch.atan2(x.imag, x.real) * power
        
        # Convert back to cartesian coordinates
        return magnitude * torch.complex(torch.cos(phase), torch.sin(phase))
    
    def _apply_filters(self, x):
        """Apply first-order wavelets efficiently."""
        x_padded = self._pad_signal(x)
        x_fft = torch.fft.fft(x_padded, dim=-1)
        
        if x.dim() == 3:  # Multi-channel
            filtered_fft = x_fft.unsqueeze(2) * self.psi1_filters.unsqueeze(0).unsqueeze(0)
        else:  # Single channel
            filtered_fft = x_fft.unsqueeze(1) * self.psi1_filters.unsqueeze(0)
            
        filtered_signals = torch.fft.ifft(filtered_fft, dim=-1)
        return self._unpad_signal(filtered_signals, scale=0)
    
    def _apply_phi_filter(self, x, decimation_factor=None):
        """Apply low-pass filter and optionally subsample."""
        x_padded = self._pad_signal(x)
        x_fft = torch.fft.fft(x_padded, dim=-1)
        smoothed_fft = x_fft * self.phi_filter.unsqueeze(0).unsqueeze(0)
        
        # Apply proper frequency domain subsampling like kymatio
        if decimation_factor is not None and decimation_factor > 1:
            # Subsample in frequency domain to avoid aliasing
            N_orig = smoothed_fft.shape[-1]
            N_sub = N_orig // decimation_factor
            if N_sub == 0:
                # Prevent zero-length output
                N_sub = 1
            # Take every decimation_factor-th frequency component
            # This is equivalent to cropping in the frequency domain, which corresponds
            # to low-pass filtering and decimating in time.
            smoothed_fft = smoothed_fft[..., :N_sub]
        
        smoothed = torch.fft.ifft(smoothed_fft, dim=-1)
        
        # Unpad correctly based on whether decimation occurred
        if decimation_factor is not None and decimation_factor > 1:
            # After decimation, padding and signal length are scaled down.
            # We extract the central part corresponding to the original signal.
            pad_left_decimated = self.pad_left // decimation_factor
            target_len_decimated = self.N // decimation_factor
            
            start_idx = pad_left_decimated
            end_idx = start_idx + target_len_decimated
            
            # Clamp indices to be within the bounds of the decimated signal
            if end_idx > smoothed.shape[-1]:
                end_idx = smoothed.shape[-1]
            
            smoothed = smoothed[..., start_idx:end_idx]
        else:
            # Use the original unpadding method if no decimation was done
            smoothed = self._unpad_signal(smoothed, scale=0)
        
        return smoothed
    
    def _compute_phase_correlation(self, filtered_signals, target_length):
        """Compute phase correlation between filter pairs."""
        # Extract signal pairs and compute phase correlation
        signal_i = filtered_signals[:, self.i_idx, :]
        signal_j = filtered_signals[:, self.j_idx, :]
        
        # Apply phase acceleration and compute correlation
        accelerated_i = self._accelerate_phase(signal_i, self.powers.view(1, -1, 1))
        phase_corr = accelerated_i * signal_j.conj()
        
        # Apply low-pass filtering and decimation to match target length
        # Ensure we don't over-decimate to avoid zero-length outputs
        if target_length > 0 and phase_corr.shape[-1] > target_length:
            decimation_factor = min(phase_corr.shape[-1], phase_corr.shape[-1] // target_length)
            decimation_factor = max(1, decimation_factor)  # Ensure at least 1
        else:
            decimation_factor = 1
            
        smoothed = self._apply_phi_filter(phase_corr, decimation_factor)
        
        # Ensure we have at least some temporal samples
        if smoothed.shape[-1] == 0:
            # Fallback: just apply phi filter without decimation
            smoothed = self._apply_phi_filter(phase_corr, 1)
        
        # Return real part (imaginary part should be negligible for properly computed phase correlations)
        return smoothed.real
    
    def _compute_cross_channel_phase_correlation(self, filtered_signals, target_length, same_pairs_only=False, apply_low_pass=True):
        """
        Compute cross-channel phase correlation.

        If `same_pairs_only` is True, this restricts the computation to pairs of
        identical filters across the two channels (e.g., ch1-psi_k vs ch2-psi_k).
        Otherwise, it computes correlations for all harmonically related filter pairs.
        
        Args:
            filtered_signals (torch.Tensor): Filtered signals of shape (B, 2, n_filters, N).
            target_length (int): Target temporal length for the output coefficients.
            same_pairs_only (bool): If True, only compute correlation for identical
                filter pairs across channels. Defaults to False.
            apply_low_pass (bool): If True, applies a low-pass filter to the
                phase correlation coefficients. Defaults to True.
        
        Returns:
            torch.Tensor: The computed cross-channel phase correlation coefficients.
        """
        if filtered_signals.shape[1] != 2:
            raise ValueError("Cross-channel correlation requires exactly 2 channels")

        if same_pairs_only:
            i_idx = self.i_idx[self.autoc_idx]
            j_idx = self.j_idx[self.autoc_idx]
            powers = self.powers[self.autoc_idx]
        else:
            i_idx = self.i_idx
            j_idx = self.j_idx
            powers = self.powers

        signal_i = filtered_signals[:, 0, i_idx, :]  # Channel 0
        signal_j = filtered_signals[:, 1, j_idx, :]  # Channel 1
        
        # Apply phase acceleration and compute cross-channel correlation
        accelerated_i = self._accelerate_phase(signal_i, powers.view(1, -1, 1))
        cross_phase_corr = accelerated_i * signal_j.conj()
        
        if apply_low_pass:
            # Apply low-pass filtering and decimation to match target length
            # Ensure we don't over-decimate to avoid zero-length outputs
            if target_length > 0 and cross_phase_corr.shape[-1] > target_length:
                decimation_factor = min(cross_phase_corr.shape[-1], cross_phase_corr.shape[-1] // target_length)
                decimation_factor = max(1, decimation_factor)  # Ensure at least 1
            else:
                decimation_factor = 1
                
            smoothed = self._apply_phi_filter(cross_phase_corr, decimation_factor)
            
            # Ensure we have at least some temporal samples
            if smoothed.shape[-1] == 0:
                # Fallback: just apply phi filter without decimation
                smoothed = self._apply_phi_filter(cross_phase_corr, 1)
        else:
            smoothed = cross_phase_corr
        
        # Return real part (imaginary part should be negligible for properly computed phase correlations)
        return smoothed.real
    
    def _create_tukey_window(self, n, alpha, device):
        """Creates a Tukey window of length n."""
        if alpha is None or not (0 < alpha <= 1):
            return torch.ones(n, device=device)
        if alpha >= 1.0:
            return torch.hann_window(n, periodic=False, device=device)
            
        taper_len = int(alpha * (n - 1) / 2.0)
        if taper_len == 0:
            return torch.ones(n, device=device)

        # Taper sections
        taper = torch.hann_window(2 * taper_len, periodic=False, device=device)
        
        window = torch.ones(n, device=device)
        window[:taper_len] = taper[:taper_len]
        window[n - taper_len:] = taper[taper_len:]
        return window

    def _apply_tukey_window(self, x):
        """Apply a Tukey window to the signal to reduce edge effects."""
        n = x.shape[-1]
        window = self._create_tukey_window(n, alpha=self.tukey_alpha, device=x.device)
        
        # The window should be broadcastable to the signal shape
        # (B, N) -> (1, N)
        # (B, C, N) -> (1, 1, N)
        while window.dim() < x.dim():
            window.unsqueeze_(0)
            
        return x * window
    
    def forward(self, x, compute_phase=True, compute_cross_phase=False,
                cross_phase_same_pairs_only=False, 
                cross_phase_low_pass=True,
                scattering_channel=0, phase_channels=None):
        """
        Forward pass computing scattering and phase correlation coefficients.
        
        Args:
            x (torch.Tensor): Input tensor (B, N) or (B, C, N).
            compute_phase (bool): Compute within-channel phase correlation.
            compute_cross_phase (bool): Compute cross-channel phase correlation.
            cross_phase_same_pairs_only (bool): If True, compute cross-channel
                correlation only for same filter pairs (e.g., ch1-psi_k vs ch2-psi_k).
                This is a subset of the full cross-phase correlation. Defaults to False.
            cross_phase_low_pass (bool): If True, applies a low-pass filter to the
                cross-channel phase correlation coefficients. Defaults to True.
            scattering_channel (int): Channel for scattering coefficients.
            phase_channels (list): Channel indices for phase correlation.
            
        Returns:
            dict: Computed coefficients with keys 'scattering', 'phase_corr', 
                  'cross_phase_corr', and 'autoc_idx'.
        """
        x = x.to(self.device)
        
        # Apply Tukey window to reduce edge effects before any processing
        if self.tukey_alpha is not None:
            x = self._apply_tukey_window(x)

        # Input validation and setup
        if x.dim() == 3:
            B, n_channels, N = x.shape
            if scattering_channel >= n_channels:
                raise ValueError(f"scattering_channel {scattering_channel} >= {n_channels}")
            
            scattering_input = x[:, scattering_channel, :].contiguous()
            phase_signals = self._setup_phase_channels(x, compute_cross_phase, phase_channels, scattering_channel, n_channels)
            
        elif x.dim() == 2:
            if scattering_channel != 0:
                raise ValueError("scattering_channel must be 0 for single-channel input")
            if compute_cross_phase:
                raise ValueError("Cross-channel correlation requires multi-channel input")
            
            scattering_input = x
            phase_signals = x if compute_phase else None
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {x.shape}")
        
        # Compute scattering coefficients
        scattering_coeffs, _ = self.scattering(scattering_input)
        target_length = scattering_coeffs.shape[-1]
        results = {'scattering': scattering_coeffs}
        
        # Debug: ensure target_length is reasonable
        if target_length == 0:
            raise ValueError(f"Scattering output has zero temporal dimension: {scattering_coeffs.shape}")
        
        # Compute phase correlations
        if (compute_phase or compute_cross_phase) and phase_signals is not None:
            # Unify the processing path by ensuring phase_signals is always 3D.
            # This avoids numerical discrepancies between single- and multi-channel processing.
            if phase_signals.dim() == 2:
                phase_signals = phase_signals.unsqueeze(1)  # Shape: (B, 1, N)

            filtered_signals = self._apply_filters(phase_signals)
            
            if compute_cross_phase:
                results['cross_phase_corr'] = self._compute_cross_channel_phase_correlation(
                    filtered_signals, target_length, same_pairs_only=cross_phase_same_pairs_only,
                    apply_low_pass=cross_phase_low_pass)
            elif compute_phase:
                # Squeeze channel dimension before passing to the correlation function
                # to match its expectation of a (B, n_filters, N) tensor.
                squeezed_signals = filtered_signals.squeeze(1)
                results['phase_corr'] = self._compute_phase_correlation(squeezed_signals, target_length)
            
            results['autoc_idx'] = self.autoc_idx
                
        return results
    
    def _setup_phase_channels(self, x, compute_cross_phase, phase_channels, scattering_channel, n_channels):
        """Setup phase correlation channels with validation."""
        if compute_cross_phase:
            if phase_channels is None:
                if n_channels < 2:
                    raise ValueError("Cross-channel correlation requires at least 2 channels")
                phase_channels = [0, 1]
            if len(phase_channels) != 2 or any(ch >= n_channels for ch in phase_channels):
                raise ValueError("Invalid phase_channels for cross-channel correlation")
            return x[:, phase_channels, :].contiguous()
        
        elif phase_channels is not None:
            if len(phase_channels) != 1:
                raise ValueError("Single-channel phase correlation requires exactly 1 channel")
            ch = phase_channels[0]
            if ch >= n_channels:
                raise ValueError(f"phase_channel {ch} >= {n_channels}")
            return x[:, ch, :].contiguous()
        
        else:
            return x[:, scattering_channel, :].contiguous()
    
    def meta(self):
        """Return scattering transform metadata."""
        return self.scattering.meta()
    
    def select_fhr_phase_coefficients(self, min_freq=0.006, max_harmonic_power=8, 
                                      include_autocorr=True, harmonic_ratios=[2, 3]):
        """
        Select optimal phase harmonic coefficients for FHR analysis.
        
        Based on clinical requirements from FHR analysis:
        - Focus on frequencies >0.006 Hz (clinical relevance for 2-min prediction)
        - Include auto-correlations for phase stability
        - Include specific harmonic ratios for rhythm detection
        - Limit harmonic powers to avoid noise
        
        Args:
            min_freq (float): Minimum frequency for coefficient selection (Hz)
            max_harmonic_power (float): Maximum harmonic power to include
            include_autocorr (bool): Include auto-correlation coefficients
            harmonic_ratios (list): List of harmonic ratios to include (e.g., [2, 3] for 1:2 and 2:3)
            
        Returns:
            dict: Selection masks and metadata
        """
        # Create frequency mask for clinically relevant frequencies
        freq_mask = self.center_freqs >= min_freq
        
        # Create masks for different coefficient types
        masks = {}
        
        # Auto-correlation mask (i == j)
        if include_autocorr:
            autocorr_mask = torch.zeros(len(self.i_idx), dtype=torch.bool, device=self.device)
            autocorr_mask[self.autoc_idx] = True
            # Filter by frequency
            valid_autocorr = (freq_mask[self.i_idx] & freq_mask[self.j_idx] & autocorr_mask)
            masks['autocorr'] = valid_autocorr
        
        # Harmonic ratio masks
        for ratio in harmonic_ratios:
            ratio_mask = torch.zeros(len(self.i_idx), dtype=torch.bool, device=self.device)
            
            # Find pairs where power is approximately the target ratio
            power_tolerance = 0.1
            target_power_mask = torch.abs(self.powers - ratio) < power_tolerance
            
            # Combine with frequency and power constraints
            harmonic_mask = (freq_mask[self.i_idx] & freq_mask[self.j_idx] & 
                           target_power_mask & (self.powers <= max_harmonic_power))
            
            masks[f'harmonic_{ratio}'] = harmonic_mask
        
        # Combined optimal mask
        optimal_mask = torch.zeros(len(self.i_idx), dtype=torch.bool, device=self.device)
        for mask in masks.values():
            optimal_mask |= mask
            
        # Metadata for analysis
        metadata = {
            'total_pairs': len(self.i_idx),
            'selected_pairs': optimal_mask.sum().item(),
            'frequency_range': (self.center_freqs.min().item(), self.center_freqs.max().item()),
            'selected_freq_range': (
                self.center_freqs[self.i_idx[optimal_mask]].min().item() if optimal_mask.any() else 0,
                self.center_freqs[self.j_idx[optimal_mask]].max().item() if optimal_mask.any() else 0
            ),
            'power_range': (
                self.powers[optimal_mask].min().item() if optimal_mask.any() else 0,
                self.powers[optimal_mask].max().item() if optimal_mask.any() else 0
            )
        }
        
        return {
            'masks': masks,
            'optimal_mask': optimal_mask,
            'metadata': metadata,
            'i_idx_selected': self.i_idx[optimal_mask],
            'j_idx_selected': self.j_idx[optimal_mask],
            'powers_selected': self.powers[optimal_mask],
            'freqs_i_selected': self.center_freqs[self.i_idx[optimal_mask]],
            'freqs_j_selected': self.center_freqs[self.j_idx[optimal_mask]]
        }
    
    def select_fhr_up_cross_coefficients(self, up_max_freq=0.02, fhr_min_freq=0.04, 
                                       fhr_max_freq=0.5, max_harmonic_power=32):
        """
        Select optimal cross-channel coefficients for FHR-UP analysis.
        
        Based on physiological requirements:
        - UP channel: contraction frequencies <0.02 Hz (2-5 contractions per 10 min)
        - FHR channel: variability bands 0.04-0.5 Hz (LF and MF bands)
        - Harmonic powers 1-32 for physiological coupling
        
        Args:
            up_max_freq (float): Maximum frequency for UP channel filters (Hz)
            fhr_min_freq (float): Minimum frequency for FHR channel filters (Hz)  
            fhr_max_freq (float): Maximum frequency for FHR channel filters (Hz)
            max_harmonic_power (float): Maximum harmonic power to include
            
        Returns:
            dict: Selection masks and metadata for cross-channel analysis
        """
        # UP channel frequency mask (slow contraction band)
        up_band_mask = self.center_freqs < up_max_freq
        
        # FHR channel frequency mask (variability bands)
        fhr_band_mask = (self.center_freqs >= fhr_min_freq) & (self.center_freqs <= fhr_max_freq)
        
        # Cross-channel pairs: UP (channel 0) -> FHR (channel 1)
        cross_mask = (up_band_mask[self.i_idx] & fhr_band_mask[self.j_idx] & 
                     (self.powers >= 1) & (self.powers <= max_harmonic_power))
        
        # Metadata for analysis
        metadata = {
            'total_pairs': len(self.i_idx),
            'cross_selected_pairs': cross_mask.sum().item(),
            'up_freq_range': (0.0, up_max_freq),
            'fhr_freq_range': (fhr_min_freq, fhr_max_freq),
            'up_filters_available': up_band_mask.sum().item(),
            'fhr_filters_available': fhr_band_mask.sum().item(),
            'power_range': (
                self.powers[cross_mask].min().item() if cross_mask.any() else 0,
                self.powers[cross_mask].max().item() if cross_mask.any() else 0
            )
        }
        
        return {
            'cross_mask': cross_mask,
            'up_band_mask': up_band_mask,
            'fhr_band_mask': fhr_band_mask,
            'metadata': metadata,
            'i_idx_selected': self.i_idx[cross_mask],  # UP channel filters
            'j_idx_selected': self.j_idx[cross_mask],  # FHR channel filters
            'powers_selected': self.powers[cross_mask],
            'up_freqs_selected': self.center_freqs[self.i_idx[cross_mask]],
            'fhr_freqs_selected': self.center_freqs[self.j_idx[cross_mask]]
        }
    
    def get_optimal_coefficients_for_fhr(self, j_config=11, q_config=4, t_config=16):
        """
        Get optimal coefficient selection for FHR analysis based on current configuration.
        
        RATIONALE FOR COEFFICIENT SELECTION:
        ===================================
        
        1. CLINICAL MOTIVATION:
        ----------------------
        - FHR signals contain clinically relevant patterns in specific frequency bands
        - Baseline FHR: 110-180 bpm (1.8-3.0 Hz) requires good frequency resolution
        - Variability bands: LF (0.04-0.15 Hz), MF (0.15-0.5 Hz) indicate fetal health
        - Contraction frequency: 2-5 contractions per 10 min (0.003-0.008 Hz)
        - Prediction horizon: 2 minutes requires focus on patterns <2.8 min periods
        
        2. FREQUENCY SELECTION CRITERIA:
        --------------------------------
        - min_freq=0.006 Hz: Excludes ultra-low frequencies (>2.8 min periods) irrelevant 
        for 2-minute prediction tasks
        - UP channel: <0.02 Hz captures contraction frequencies (2-5 per 10 min)
        - FHR channel: 0.04-0.5 Hz captures both LF and MF variability bands
        - This focuses on physiologically meaningful frequency interactions
        
        3. PHASE HARMONIC SELECTION:
        ----------------------------
        - Auto-correlations (i=j): Phase stability at each frequency scale
        - 1:2 harmonic ratios: Rhythm detection and period doubling
        - 2:3 harmonic ratios: Complex rhythmic patterns (often absent in FHR)
        - Power limits ≤8: Avoid noise from high harmonic interactions
        
        4. CROSS-CHANNEL SELECTION:
        ---------------------------
        - UP→FHR directionality: Contractions influence FHR (physiological causality)
        - Harmonic powers 1-32: Captures physiological coupling ranges
        - Asymmetric frequency bands: Slow contractions → Fast FHR variability
        
        5. EFFICIENCY GAINS:
        -------------------
        For J=11, Q=4, T=16 configuration:
        - Total possible phase pairs: 903
        - Selected phase pairs: ~44 (95.1% reduction)
        - Selected cross-channel pairs: ~130
        - Result: Focused feature set with better clinical interpretability
        
        TEST RESULTS (J=11, Q=4, T=16):
        ===============================
        Configuration: {'J': 11, 'Q': 4, 'T': 16}
        Total scattering coefficients: 45
        Selected phase coefficients: 44
        Selected cross-channel coefficients: 130
        Total selected features: 219
        Phase reduction: 95.1%
        
        Phase Selection Breakdown:
        - autocorr: 24 coefficients (phase stability)
        - harmonic_2: 20 coefficients (1:2 rhythm detection)
        - harmonic_3: 0 coefficients (no 2:3 patterns found)
        
        Cross-Channel Analysis:
        - UP filters available: 25 (in contraction band <0.02 Hz)
        - FHR filters available: 13 (in variability bands 0.04-0.5 Hz)
        - Selected pairs: 130 (physiologically relevant UP→FHR coupling)
        - Power range: 2.4-32 (appropriate harmonic coupling)
        
        Automatically selects coefficients based on:
        - Current scattering parameters (J, Q, T)
        - Clinical requirements for FHR analysis
        - Physiological frequency bands
        
        Args:
            j_config (int): Current J parameter
            q_config (int): Current Q parameter  
            t_config (int): Current T parameter
            
        Returns:
            dict: Complete coefficient selection strategy with masks and metadata
        """
        # Adjust frequency thresholds based on configuration
        # For J=11, lowest frequency is ~0.0007 Hz, so we set min_freq to focus on relevant scales
        if j_config >= 11:
            min_freq = 0.006  # Focus on scales 0-8 for clinical relevance
        else:
            min_freq = 0.003  # Include more scales for smaller J
            
        # Get phase coefficient selection
        phase_selection = self.select_fhr_phase_coefficients(
            min_freq=min_freq,
            max_harmonic_power=8,
            include_autocorr=True,
            harmonic_ratios=[2, 3]
        )
        
        # Get cross-channel coefficient selection  
        cross_selection = self.select_fhr_up_cross_coefficients(
            up_max_freq=0.02,
            fhr_min_freq=0.04,
            fhr_max_freq=0.5,
            max_harmonic_power=32
        )
        
        # Provide configuration analysis
        config_analysis = {
            'current_config': {'J': j_config, 'Q': q_config, 'T': t_config},
            'total_scattering_coeffs': j_config * q_config + 1,  # Approximate for first-order
            'selected_phase_coeffs': phase_selection['optimal_mask'].sum().item(),
            'selected_cross_coeffs': cross_selection['cross_mask'].sum().item(),
            'efficiency_gain': {
                'phase_reduction': f"{100 * (1 - phase_selection['optimal_mask'].sum().item() / len(self.i_idx)):.1f}%",
                'focus_improvement': f"Focused on {phase_selection['metadata']['selected_pairs']} most relevant pairs"
            }
        }
        
        return {
            'phase_selection': phase_selection,
            'cross_selection': cross_selection,
            'config_analysis': config_analysis,
            'recommendations': {
                'use_phase_mask': phase_selection['optimal_mask'],
                'use_cross_mask': cross_selection['cross_mask'],
                'total_selected_features': (
                    config_analysis['total_scattering_coeffs'] + 
                    config_analysis['selected_phase_coeffs'] + 
                    config_analysis['selected_cross_coeffs']
                )
            }
        }

    def verify_phase_correlation_properties(self, x, tol=1e-6):
        """
        Verify mathematical properties of phase correlations.
        
        Returns:
            dict: Verification results with 'passed' bool and 'details' dict
        """
        results = {'passed': True, 'details': {}}
        
        # Test with simple signal
        if x.dim() == 2:
            x_test = x[:1]  # Single batch
        else:
            x_test = x[:1, :1]  # Single batch, single channel
            
        try:
            # Apply filters
            filtered = self._apply_filters(x_test)
            
            # Check that filtered signals can be processed
            # The phase correlations themselves are now returned as real values
            try:
                phase_corr = self._compute_phase_correlation(filtered, filtered.shape[-1])
                # Check that autocorrelations are positive (they're already real)
                for k, idx in enumerate(self.autoc_idx):
                    autocorr_value = phase_corr[0, idx, :]
                    if torch.any(autocorr_value < -tol):
                        results['passed'] = False
                        results['details'][f'autocorr_{k}_negative'] = autocorr_value.min().item()
            except Exception as e:
                results['passed'] = False
                results['details']['phase_computation_error'] = str(e)
            
            # Check frequency ordering constraint
            for k in range(len(self.i_idx)):
                xi_i = self.center_freqs[self.i_idx[k]]
                xi_j = self.center_freqs[self.j_idx[k]]
                if xi_j < xi_i - tol:
                    results['passed'] = False
                    results['details'][f'frequency_ordering_violation_{k}'] = (xi_i.item(), xi_j.item())
            
            # Check power values are >= 1
            if torch.any(self.powers < 1.0 - tol):
                results['passed'] = False
                results['details']['invalid_powers'] = self.powers[self.powers < 1.0 - tol].tolist()
                
        except Exception as e:
            results['passed'] = False
            results['details']['error'] = str(e)
            
        return results