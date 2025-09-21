import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def entropy_discrete(samples: np.ndarray) -> float:
    """
    Compute entropy of discrete samples.
    
    Args:
        samples: 1D array of discrete values
        
    Returns:
        Entropy value
    """
    _, counts = np.unique(samples, return_counts=True)
    probabilities = counts / len(samples)
    return -np.sum(probabilities * np.log(probabilities + 1e-12))


def compute_mig(latent_codes: np.ndarray, factors: np.ndarray) -> float:
    """
    Compute Mutual Information Gap (MIG) metric for disentanglement evaluation.
    
    Args:
        latent_codes: [N, latent_dim] encoded representations
        factors: [N, factor_dim] ground truth factors
        
    Returns:
        MIG score (0 to 1, higher is better)
    """
    if latent_codes.shape[0] != factors.shape[0]:
        raise ValueError("Number of samples must match between latent codes and factors")
    
    num_factors = factors.shape[1]
    num_latents = latent_codes.shape[1]
    
    if num_latents == 0 or num_factors == 0:
        return 0.0
    
    # Compute mutual information matrix
    mi_matrix = np.zeros((num_latents, num_factors))
    
    for i in range(num_latents):
        for j in range(num_factors):
            # Handle both continuous and discrete factors
            try:
                mi_matrix[i, j] = mutual_info_regression(
                    latent_codes[:, i:i+1], factors[:, j]
                )[0]
            except:
                mi_matrix[i, j] = 0.0
    
    # Compute factor entropy (handle both discrete and continuous)
    factor_entropy = np.zeros(num_factors)
    for j in range(num_factors):
        factor_values = factors[:, j]
        if len(np.unique(factor_values)) < len(factor_values) * 0.1:
            # Discrete factor
            factor_entropy[j] = entropy_discrete(factor_values)
        else:
            # Continuous factor - use differential entropy approximation
            factor_entropy[j] = 0.5 * np.log(2 * np.pi * np.e * np.var(factor_values))
    
    # Prevent division by zero
    factor_entropy = np.maximum(factor_entropy, 1e-12)
    
    # Normalize MI matrix by factor entropy
    mi_matrix_norm = mi_matrix / factor_entropy[np.newaxis, :]
    
    # Compute MIG for each factor
    mig_scores = []
    for j in range(num_factors):
        sorted_mi = np.sort(mi_matrix_norm[:, j])[::-1]
        if len(sorted_mi) > 1:
            gap = sorted_mi[0] - sorted_mi[1]
            mig_scores.append(gap)
        else:
            mig_scores.append(sorted_mi[0] if len(sorted_mi) > 0 else 0.0)
    
    return np.mean(mig_scores)


def compute_fhr_clinical_features(fhr_raw: torch.Tensor) -> torch.Tensor:
    """
    Extract clinically relevant FHR features for evaluation.
    
    Args:
        fhr_raw: Raw FHR signal (batch, sequence_length) in bpm
        
    Returns:
        torch.Tensor: Clinical features (batch, num_features)
    """
    if fhr_raw.dim() == 3 and fhr_raw.size(-1) == 1:
        fhr_raw = fhr_raw.squeeze(-1)
    
    batch_size = fhr_raw.shape[0]
    features_list = []
    
    for i in range(batch_size):
        signal = fhr_raw[i].cpu().numpy()
        
        # Remove NaN and infinite values
        signal = signal[np.isfinite(signal)]
        
        if len(signal) < 10:  # Skip if too few valid samples
            # Return zero features for invalid signals
            features_list.append(np.zeros(12))
            continue
        
        # Basic statistics
        baseline = np.median(signal)
        variability = np.std(signal)
        mean_fhr = np.mean(signal)
        
        # Range features
        fhr_range = np.ptp(signal)  # peak-to-peak
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        
        # Accelerations (>15 bpm increase for >15 seconds at 4Hz = 60 samples)
        accelerations = count_accelerations(signal, baseline, threshold=15, min_duration=60)
        
        # Decelerations (>15 bpm decrease for >15 seconds)
        decelerations = count_decelerations(signal, baseline, threshold=15, min_duration=60)
        
        # Frequency domain features
        freq_features = compute_frequency_features(signal)
        
        # Trend analysis
        slope = compute_trend_slope(signal)
        
        sample_features = [
            baseline,           # 0: FHR baseline (median)
            variability,        # 1: Short-term variability (std)
            mean_fhr,          # 2: Mean FHR
            fhr_range,         # 3: FHR range
            iqr,               # 4: Interquartile range
            accelerations,     # 5: Number of accelerations
            decelerations,     # 6: Number of decelerations
            slope,             # 7: Overall trend slope
            freq_features[0],  # 8: Low frequency power
            freq_features[1],  # 9: High frequency power
            freq_features[2],  # 10: LF/HF ratio
            freq_features[3],  # 11: Total power
        ]
        features_list.append(sample_features)
    
    return torch.tensor(features_list, dtype=torch.float32)


def count_accelerations(signal: np.ndarray, baseline: float, threshold: float = 15, min_duration: int = 60) -> int:
    """
    Count FHR accelerations (increases >threshold bpm for >min_duration samples).
    
    Args:
        signal: FHR signal in bpm
        baseline: Baseline FHR value
        threshold: Minimum increase in bpm
        min_duration: Minimum duration in samples
        
    Returns:
        Number of accelerations
    """
    # Find regions above baseline + threshold
    above_threshold = signal > (baseline + threshold)
    
    # Find continuous regions
    accelerations = 0
    current_duration = 0
    
    for value in above_threshold:
        if value:
            current_duration += 1
        else:
            if current_duration >= min_duration:
                accelerations += 1
            current_duration = 0
    
    # Check final region
    if current_duration >= min_duration:
        accelerations += 1
    
    return accelerations


def count_decelerations(signal: np.ndarray, baseline: float, threshold: float = 15, min_duration: int = 60) -> int:
    """
    Count FHR decelerations (decreases >threshold bpm for >min_duration samples).
    
    Args:
        signal: FHR signal in bpm
        baseline: Baseline FHR value
        threshold: Minimum decrease in bpm
        min_duration: Minimum duration in samples
        
    Returns:
        Number of decelerations
    """
    # Find regions below baseline - threshold
    below_threshold = signal < (baseline - threshold)
    
    # Find continuous regions
    decelerations = 0
    current_duration = 0
    
    for value in below_threshold:
        if value:
            current_duration += 1
        else:
            if current_duration >= min_duration:
                decelerations += 1
            current_duration = 0
    
    # Check final region
    if current_duration >= min_duration:
        decelerations += 1
    
    return decelerations


def compute_frequency_features(signal: np.ndarray, fs: float = 4.0) -> List[float]:
    """
    Compute frequency domain features for FHR analysis.
    
    Args:
        signal: FHR signal
        fs: Sampling frequency (Hz)
        
    Returns:
        List of frequency features [lf_power, hf_power, lf_hf_ratio, total_power]
    """
    try:
        from scipy.signal import welch
        
        # Compute power spectral density
        freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)//4))
        
        # Define frequency bands for FHR analysis
        lf_band = (0.04, 0.15)  # Low frequency band (Hz)
        hf_band = (0.15, 0.4)   # High frequency band (Hz)
        
        # Find frequency indices
        lf_indices = np.where((freqs >= lf_band[0]) & (freqs < lf_band[1]))[0]
        hf_indices = np.where((freqs >= hf_band[0]) & (freqs < hf_band[1]))[0]
        
        # Compute power in each band
        lf_power = np.trapz(psd[lf_indices], freqs[lf_indices]) if len(lf_indices) > 0 else 0.0
        hf_power = np.trapz(psd[hf_indices], freqs[hf_indices]) if len(hf_indices) > 0 else 0.0
        total_power = np.trapz(psd, freqs)
        
        # LF/HF ratio
        lf_hf_ratio = lf_power / (hf_power + 1e-12)
        
        return [lf_power, hf_power, lf_hf_ratio, total_power]
    
    except ImportError:
        # Fallback if scipy is not available
        return [0.0, 0.0, 0.0, np.var(signal)]


def compute_trend_slope(signal: np.ndarray) -> float:
    """
    Compute the overall trend slope of the signal using linear regression.
    
    Args:
        signal: Input signal
        
    Returns:
        Slope coefficient
    """
    if len(signal) < 2:
        return 0.0
    
    x = np.arange(len(signal))
    slope, _ = np.polyfit(x, signal, 1)
    return slope


def compute_total_correlation_metric(latent_codes: torch.Tensor) -> float:
    """
    Compute total correlation metric for evaluating disentanglement.
    
    Args:
        latent_codes: (N, seq_len, latent_dim) or (N, latent_dim) tensor
        
    Returns:
        Total correlation value
    """
    if latent_codes.dim() == 3:
        # Average over sequence dimension for global representation
        latent_codes = latent_codes.mean(dim=1)
    
    latent_codes_np = latent_codes.cpu().numpy()
    
    # Estimate marginal distributions using histograms
    num_samples, num_dims = latent_codes_np.shape
    bins = min(50, int(np.sqrt(num_samples)))
    
    # Compute joint entropy (using correlation-based approximation)
    correlation_matrix = np.corrcoef(latent_codes_np.T)
    
    # Total correlation approximation using correlation determinant
    det_corr = np.linalg.det(correlation_matrix + 1e-6 * np.eye(num_dims))
    tc_approx = -0.5 * np.log(det_corr + 1e-12)
    
    return max(0.0, tc_approx)


def compute_clinical_mutual_information(latent_codes: torch.Tensor, clinical_features: torch.Tensor) -> Dict[str, float]:
    """
    Compute mutual information between latent factors and clinical FHR features.
    
    Args:
        latent_codes: (N, seq_len, latent_dim) or (N, latent_dim) tensor
        clinical_features: (N, num_clinical_features) tensor
        
    Returns:
        Dictionary of mutual information scores
    """
    if latent_codes.dim() == 3:
        latent_codes = latent_codes.mean(dim=1)
    
    latent_np = latent_codes.cpu().numpy()
    clinical_np = clinical_features.cpu().numpy()
    
    num_latents = latent_np.shape[1]
    num_clinical = clinical_np.shape[1]
    
    clinical_names = [
        'baseline', 'variability', 'mean_fhr', 'fhr_range', 'iqr',
        'accelerations', 'decelerations', 'trend_slope',
        'lf_power', 'hf_power', 'lf_hf_ratio', 'total_power'
    ]
    
    # Compute MI matrix
    mi_results = {}
    max_mi_per_latent = []
    
    for i in range(num_latents):
        latent_factor = latent_np[:, i:i+1]
        mi_scores = []
        
        for j in range(min(num_clinical, len(clinical_names))):
            clinical_factor = clinical_np[:, j]
            try:
                mi_score = mutual_info_regression(latent_factor, clinical_factor)[0]
                mi_scores.append(mi_score)
            except:
                mi_scores.append(0.0)
        
        max_mi_per_latent.append(max(mi_scores) if mi_scores else 0.0)
        
        # Find best clinical match for this latent factor
        if mi_scores:
            best_idx = np.argmax(mi_scores)
            best_clinical = clinical_names[best_idx] if best_idx < len(clinical_names) else f"feature_{best_idx}"
            mi_results[f'latent_{i}_best_match'] = best_clinical
            mi_results[f'latent_{i}_max_mi'] = mi_scores[best_idx]
    
    mi_results['mean_max_mi'] = np.mean(max_mi_per_latent)
    mi_results['total_interpretability'] = sum(1 for mi in max_mi_per_latent if mi > 0.1)
    
    return mi_results


def compute_beta_vae_metric(latent_codes: torch.Tensor, factors: torch.Tensor) -> float:
    """
    Compute β-VAE disentanglement metric.
    
    Args:
        latent_codes: (N, latent_dim) tensor
        factors: (N, num_factors) tensor
        
    Returns:
        β-VAE metric score
    """
    if latent_codes.dim() == 3:
        latent_codes = latent_codes.mean(dim=1)
    
    latent_np = latent_codes.cpu().numpy()
    factors_np = factors.cpu().numpy()
    
    num_latents = latent_np.shape[1]
    num_factors = factors_np.shape[1]
    
    # Compute mutual information matrix
    mi_matrix = np.zeros((num_latents, num_factors))
    
    for i in range(num_latents):
        for j in range(num_factors):
            try:
                mi_matrix[i, j] = mutual_info_regression(
                    latent_np[:, i:i+1], factors_np[:, j]
                )[0]
            except:
                mi_matrix[i, j] = 0.0
    
    # β-VAE metric: average of max MI per factor
    max_mi_per_factor = np.max(mi_matrix, axis=0)
    return np.mean(max_mi_per_factor)


def assess_factor_interpretability(latent_codes: torch.Tensor, clinical_features: torch.Tensor) -> Dict[str, Dict]:
    """
    Assess how well individual latent factors correspond to interpretable FHR features.
    
    Args:
        latent_codes: (N, seq_len, latent_dim) or (N, latent_dim) tensor
        clinical_features: (N, num_clinical_features) tensor
        
    Returns:
        Dictionary of interpretability scores per latent factor
    """
    if latent_codes.dim() == 3:
        latent_codes = latent_codes.mean(dim=1)
    
    latent_np = latent_codes.cpu().numpy()
    clinical_np = clinical_features.cpu().numpy()
    
    num_latents = latent_np.shape[1]
    num_clinical = clinical_np.shape[1]
    
    clinical_names = [
        'baseline', 'variability', 'mean_fhr', 'fhr_range', 'iqr',
        'accelerations', 'decelerations', 'trend_slope',
        'lf_power', 'hf_power', 'lf_hf_ratio', 'total_power'
    ]
    
    interpretability_scores = {}
    
    for i in range(num_latents):
        latent_factor = latent_np[:, i]
        
        correlations = []
        mi_scores = []
        
        for j in range(min(num_clinical, len(clinical_names))):
            clinical_factor = clinical_np[:, j]
            
            # Compute correlation
            try:
                corr = np.corrcoef(latent_factor, clinical_factor)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
            except:
                correlations.append(0.0)
            
            # Compute mutual information
            try:
                mi = mutual_info_regression(latent_factor.reshape(-1, 1), clinical_factor)[0]
                mi_scores.append(mi)
            except:
                mi_scores.append(0.0)
        
        # Find the clinical feature most correlated with this latent factor
        max_corr_idx = np.argmax(correlations) if correlations else 0
        max_mi_idx = np.argmax(mi_scores) if mi_scores else 0
        
        max_correlation = correlations[max_corr_idx] if correlations else 0.0
        max_mi = mi_scores[max_mi_idx] if mi_scores else 0.0
        
        best_clinical_corr = clinical_names[max_corr_idx] if max_corr_idx < len(clinical_names) else f"feature_{max_corr_idx}"
        best_clinical_mi = clinical_names[max_mi_idx] if max_mi_idx < len(clinical_names) else f"feature_{max_mi_idx}"
        
        interpretability_scores[f'latent_{i}'] = {
            'max_correlation': max_correlation,
            'best_clinical_match_corr': best_clinical_corr,
            'max_mutual_info': max_mi,
            'best_clinical_match_mi': best_clinical_mi,
            'all_correlations': correlations,
            'all_mi_scores': mi_scores,
            'interpretability_score': (max_correlation + max_mi) / 2.0  # Combined score
        }
    
    return interpretability_scores


def compute_fhr_disentanglement_metrics(model, fhr_dataloader, device) -> Dict[str, float]:
    """
    Compute comprehensive disentanglement metrics specific to FHR analysis.
    
    Args:
        model: Trained SeqVaeTeb model with β-TCVAE
        fhr_dataloader: DataLoader for FHR dataset
        device: torch.device
        
    Returns:
        dict: Comprehensive disentanglement metrics
    """
    model.eval()
    
    # Collect latent representations and reconstructions
    latent_codes = []
    fhr_features = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for i, batch in enumerate(fhr_dataloader):
            if i >= 10:  # Limit to first 10 batches for efficiency
                break
                
            # Extract batch data
            if hasattr(batch, 'fhr_st'):
                y_st, y_ph, x_ph = batch.fhr_st.to(device), batch.fhr_ph.to(device), batch.fhr_up_ph.to(device)
                y_raw = batch.fhr.to(device)
            else:
                # Handle different batch formats
                y_st, y_ph, x_ph, y_raw = batch
                y_st, y_ph, x_ph, y_raw = y_st.to(device), y_ph.to(device), x_ph.to(device), y_raw.to(device)
            
            outputs = model(y_st, y_ph, x_ph)
            z = outputs['z']  # (batch, seq_len, latent_dim)
            
            # Compute FHR clinical features
            fhr_stats = compute_fhr_clinical_features(y_raw)
            
            latent_codes.append(z.cpu())
            fhr_features.append(fhr_stats)
            
            # Reconstruction quality
            if 'mu_pr' in outputs:
                recon_loss = F.mse_loss(outputs['mu_pr'], y_raw.view(y_raw.size(0), -1))
                reconstruction_errors.append(recon_loss.item())
    
    if not latent_codes:
        return {'error': 'No valid batches processed'}
    
    latent_codes = torch.cat(latent_codes, dim=0)  # (total_samples, seq_len, latent_dim)
    fhr_features = torch.cat(fhr_features, dim=0)  # (total_samples, num_clinical_features)
    
    # Compute metrics
    metrics = {
        'total_correlation': compute_total_correlation_metric(latent_codes),
        'reconstruction_quality': np.mean(reconstruction_errors) if reconstruction_errors else 0.0,
        'beta_vae_metric': compute_beta_vae_metric(latent_codes, fhr_features),
    }
    
    # Clinical mutual information
    clinical_mi = compute_clinical_mutual_information(latent_codes, fhr_features)
    metrics.update(clinical_mi)
    
    # Factor interpretability
    interpretability = assess_factor_interpretability(latent_codes, fhr_features)
    
    # Summarize interpretability
    interpretability_scores = [
        interpretability[key]['interpretability_score'] 
        for key in interpretability.keys()
    ]
    metrics['mean_interpretability'] = np.mean(interpretability_scores)
    metrics['num_interpretable_factors'] = sum(1 for score in interpretability_scores if score > 0.3)
    
    return metrics
