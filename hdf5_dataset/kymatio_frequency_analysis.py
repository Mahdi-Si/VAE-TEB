#!/usr/bin/env python3
"""
Frequency Analysis for Scattering Transform
Analyzes the frequency ranges and characteristics for specific scattering parameters.
"""

import numpy as np
import math
from typing import Tuple, List, Dict

def compute_xi_max(Q: int) -> float:
    """Compute the maximal xi to use for the Morlet family, depending on Q."""
    xi_max = max(1. / (1. + math.pow(2., 3. / Q)), 0.35)
    return xi_max

def compute_sigma_psi(xi: float, Q: int, r: float = math.sqrt(0.5)) -> float:
    """Compute the frequential width sigma for a Morlet filter."""
    factor = 1. / math.pow(2, 1. / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / math.sqrt(2 * math.log(1. / r))
    return xi * term1 * term2

def get_max_dyadic_subsampling(xi: float, sigma: float, alpha: float = 5.) -> int:
    """Compute the maximal dyadic subsampling possible for a Gabor filter."""
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j

def move_one_dyadic_step(xi: float, sigma: float, Q: int, alpha: float = 5.) -> Tuple[float, float, int]:
    """Compute the parameters of the next wavelet on the low frequency side."""
    factor = 1. / math.pow(2., 1. / Q)
    new_xi = xi * factor
    new_sigma = sigma * factor
    new_j = get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha)
    return new_xi, new_sigma, new_j

def compute_params_filterbank(sigma_min: float, Q: int, r_psi: float = math.sqrt(0.5), 
                             alpha: float = 5.) -> Tuple[List[float], List[float], List[int]]:
    """Compute the parameters of a Morlet wavelet filterbank."""
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    xi = []
    sigma = []
    j = []

    if sigma_max <= sigma_min:
        last_xi = sigma_max
    else:
        # Fill all the dyadic wavelets as long as possible
        current_xi, current_sigma = xi_max, sigma_max
        current_j = get_max_dyadic_subsampling(current_xi, current_sigma, alpha=alpha)
        
        while current_sigma > sigma_min:
            xi.append(current_xi)
            sigma.append(current_sigma)
            j.append(current_j)
            current_xi, current_sigma, current_j = move_one_dyadic_step(current_xi, current_sigma, Q, alpha=alpha)
        
        last_xi = xi[-1]
    
    # Fill num_interm wavelets between last_xi and 0, both excluded
    num_intermediate = Q - 1
    for q in range(1, num_intermediate + 1):
        factor = (num_intermediate + 1. - q) / (num_intermediate + 1.)
        new_xi = factor * last_xi
        new_sigma = sigma_min
        xi.append(new_xi)
        sigma.append(new_sigma)
        j.append(get_max_dyadic_subsampling(new_xi, new_sigma, alpha=alpha))
    
    return xi, sigma, j

def calibrate_scattering_filters(J: int, Q: int, T: int, r_psi: float = math.sqrt(0.5), 
                                sigma0: float = 0.1, alpha: float = 5.) -> Tuple:
    """Calibrate the parameters of the filters used at the 1st and 2nd orders."""
    if Q < 1:
        raise ValueError('Q should always be >= 1, got {}'.format(Q))

    # Lower bound of band-pass filter frequential widths
    sigma_min = sigma0 / math.pow(2, J)

    xi1s, sigma1s, j1 = compute_params_filterbank(sigma_min, Q, r_psi=r_psi, alpha=alpha)
    xi2s, sigma2s, j2 = compute_params_filterbank(sigma_min, 1, r_psi=r_psi, alpha=alpha)

    # Width of the low-pass filter
    sigma_low = sigma0 / T
    return sigma_low, xi1s, sigma1s, j1, xi2s, sigma2s, j2

def analyze_scattering_frequencies(J: int, Q: int, T: int, sampling_rate: float, 
                                 signal_duration_minutes: float, 
                                 analyze_phase_harmonics: bool = True,
                                 analyze_cross_phase: bool = True) -> Dict:
    """
    Analyze the frequency characteristics for specific scattering transform parameters,
    including phase harmonics and cross-channel phase correlations.
    
    Parameters:
    -----------
    J : int
        Maximum scale (controls number of wavelets)
    Q : int
        Number of wavelets per octave for first order
    T : int
        Temporal support of low-pass filter
    sampling_rate : float
        Sampling rate in Hz
    signal_duration_minutes : float
        Signal duration in minutes
    analyze_phase_harmonics : bool
        Whether to analyze within-channel phase harmonic correlations
    analyze_cross_phase : bool
        Whether to analyze cross-channel phase correlations
    
    Returns:
    --------
    Dict containing frequency analysis results
    """
    
    # Calculate signal characteristics
    signal_duration_seconds = signal_duration_minutes * 60
    num_samples = int(sampling_rate * signal_duration_seconds)
    nyquist_freq = sampling_rate / 2
    
    # Get filter parameters
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = calibrate_scattering_filters(
        J, Q, T, r_psi=math.sqrt(0.5), sigma0=0.1, alpha=5.0)
    
    # Convert normalized frequencies to Hz
    def normalized_to_hz(xi_norm):
        return xi_norm * sampling_rate
    
    def compute_bandwidth_hz(sigma_norm, xi_norm):
        # Approximate bandwidth based on the standard deviation in frequency
        # For Morlet wavelets, effective bandwidth is approximately 2*sigma*sampling_rate
        return 2 * sigma_norm * sampling_rate
    
    # Analyze first-order filters
    first_order_analysis = []
    for i, (xi, sigma, j) in enumerate(zip(xi1s, sigma1s, j1s)):
        center_freq_hz = normalized_to_hz(xi)
        bandwidth_hz = compute_bandwidth_hz(sigma, xi)
        first_order_analysis.append({
            'filter_index': i,
            'xi_normalized': xi,
            'sigma_normalized': sigma,
            'center_freq_hz': center_freq_hz,
            'bandwidth_hz': bandwidth_hz,
            'frequency_range_hz': (center_freq_hz - bandwidth_hz/2, center_freq_hz + bandwidth_hz/2),
            'max_subsampling_j': j
        })
    
    # Analyze second-order filters
    second_order_analysis = []
    for i, (xi, sigma, j) in enumerate(zip(xi2s, sigma2s, j2s)):
        center_freq_hz = normalized_to_hz(xi)
        bandwidth_hz = compute_bandwidth_hz(sigma, xi)
        second_order_analysis.append({
            'filter_index': i,
            'xi_normalized': xi,
            'sigma_normalized': sigma,
            'center_freq_hz': center_freq_hz,
            'bandwidth_hz': bandwidth_hz,
            'frequency_range_hz': (center_freq_hz - bandwidth_hz/2, center_freq_hz + bandwidth_hz/2),
            'max_subsampling_j': j
        })
    
    # Low-pass filter analysis
    low_pass_analysis = {
        'sigma_normalized': sigma_low,
        'bandwidth_hz': 2 * sigma_low * sampling_rate,
        'cutoff_freq_hz': sigma_low * sampling_rate
    }
    
    # Count coefficients
    def count_second_order_pairs():
        count = 0
        for i, j1 in enumerate(j1s):
            for j, j2 in enumerate(j2s):
                if j2 > j1:  # Second-order condition: j2 > j1
                    count += 1
        return count
    
    def count_phase_harmonic_pairs(xi_list):
        """Count phase harmonic correlation pairs (all pairs where xi_j >= xi_i)"""
        n_filters = len(xi_list)
        count = 0
        auto_count = 0
        pairs_info = []
        
        for i in range(n_filters):
            xi_i = xi_list[i]
            for j in range(n_filters):
                xi_j = xi_list[j]
                if xi_j >= xi_i:  # Phase harmonic condition
                    count += 1
                    if i == j:
                        auto_count += 1
                    
                    # Compute harmonic ratio (power)
                    if xi_i > 1e-8:
                        power = xi_j / xi_i
                    else:
                        power = 1.0
                    
                    pairs_info.append({
                        'pair_index': len(pairs_info),
                        'filter_i': i,
                        'filter_j': j,
                        'xi_i': xi_i,
                        'xi_j': xi_j,
                        'xi_i_hz': xi_i * sampling_rate,
                        'xi_j_hz': xi_j * sampling_rate,
                        'power': power,
                        'is_auto': i == j,
                        'harmonic_type': classify_harmonic_relationship(power)
                    })
        
        return count, auto_count, pairs_info
    
    def classify_harmonic_relationship(power):
        """Classify the type of harmonic relationship based on frequency ratio"""
        if abs(power - 1.0) < 0.01:
            return "Auto-correlation (1:1)"
        elif abs(power - 2.0) < 0.1:
            return "Octave (1:2)"
        elif abs(power - 1.5) < 0.1:
            return "Perfect Fifth (2:3)"
        elif abs(power - 3.0) < 0.1:
            return "Octave + Fifth (1:3)"
        elif abs(power - 4.0) < 0.1:
            return "Double Octave (1:4)"
        elif power < 1.5:
            return "Close frequencies"
        elif power < 3.0:
            return "Harmonic relationship"
        else:
            return "High harmonic ratio"
    
    # Physiological frequency band mapping
    def map_to_physiological_bands(freq_hz):
        if freq_hz < 0.1:
            return "Ultra-low frequency (ULF)"
        elif freq_hz < 0.5:
            return "Very low frequency (VLF)" 
        elif freq_hz < 1.0:
            return "Low frequency (LF)"
        elif freq_hz < 2.0:
            return "High frequency (HF)"
        else:
            return "Very high frequency (VHF)"
    
    # Add physiological band mapping to filter analyses
    for filter_data in first_order_analysis:
        filter_data['physiological_band'] = map_to_physiological_bands(filter_data['center_freq_hz'])
    
    for filter_data in second_order_analysis:
        filter_data['physiological_band'] = map_to_physiological_bands(filter_data['center_freq_hz'])
    
    # Analyze phase harmonics if requested
    phase_harmonic_analysis = {}
    if analyze_phase_harmonics:
        phase_count, auto_count, phase_pairs = count_phase_harmonic_pairs(xi1s)
        phase_harmonic_analysis = {
            'total_pairs': phase_count,
            'auto_correlations': auto_count,
            'cross_correlations': phase_count - auto_count,
            'pairs_detail': phase_pairs
        }
    
    # Analyze cross-channel phase harmonics if requested
    cross_phase_analysis = {}
    if analyze_cross_phase:
        # Cross-channel uses the same pair structure as within-channel
        cross_count, cross_auto_count, cross_pairs = count_phase_harmonic_pairs(xi1s)
        cross_phase_analysis = {
            'total_pairs': cross_count,
            'auto_correlations': cross_auto_count,  # Same filter across channels
            'cross_correlations': cross_count - cross_auto_count,
            'pairs_detail': cross_pairs,
            'description': 'Cross-channel phase correlations between two different signals'
        }

    def get_second_order_relevance(freq1_hz, freq2_hz):
        """Describe physiological relevance of second-order interactions"""
        if 0.04 <= freq1_hz <= 0.15 and 0.15 <= freq2_hz <= 0.4:
            return "LF-HF interaction (autonomic balance)"
        elif freq1_hz < 0.04 and 0.04 <= freq2_hz <= 0.15:
            return "VLF-LF interaction (long-term regulation)"
        elif freq1_hz < 0.04 and 0.15 <= freq2_hz <= 0.4:
            return "VLF-HF interaction (respiratory-autonomic coupling)"
        elif 0.15 <= freq1_hz <= 0.4 and freq2_hz > 0.4:
            return "HF modulation of higher frequencies"
        else:
            return "General amplitude modulation pattern"


    # Detailed second-order analysis
    second_order_detailed = []
    for i, (xi1, j1) in enumerate(zip(xi1s, j1s)):
        for k, (xi2, j2) in enumerate(zip(xi2s, j2s)):
            if j2 > j1:  # Valid second-order pair
                second_order_detailed.append({
                    'pair_index': len(second_order_detailed),
                    'first_filter_idx': i,
                    'second_filter_idx': k,
                    'xi1_norm': xi1,
                    'xi2_norm': xi2,
                    'xi1_hz': xi1 * sampling_rate,
                    'xi2_hz': xi2 * sampling_rate,
                    'j1': j1,
                    'j2': j2,
                    'frequency_interaction': f'{xi1*sampling_rate:.4f}Hz → {xi2*sampling_rate:.4f}Hz',
                    'modulation_type': 'Amplitude modulation analysis',
                    'physiological_relevance': get_second_order_relevance(xi1*sampling_rate, xi2*sampling_rate)
                })
    

    return {
        'signal_info': {
            'sampling_rate_hz': sampling_rate,
            'duration_minutes': signal_duration_minutes,
            'duration_seconds': signal_duration_seconds,
            'num_samples': num_samples,
            'nyquist_freq_hz': nyquist_freq
        },
        'scattering_params': {
            'J': J,
            'Q': Q,
            'T': T,
            'sigma_min': sigma_low * T  # This is sigma0/2^J
        },
        'coefficient_counts': {
            'zeroth_order': 1,
            'first_order': len(xi1s),
            'second_order': count_second_order_pairs(),
            'total_scattering': 1 + len(xi1s) + count_second_order_pairs(),
            'phase_harmonics': phase_harmonic_analysis.get('total_pairs', 0) if analyze_phase_harmonics else 0,
            'cross_phase_harmonics': cross_phase_analysis.get('total_pairs', 0) if analyze_cross_phase else 0,
            'total_all_coefficients': (1 + len(xi1s) + count_second_order_pairs() + 
                                     (phase_harmonic_analysis.get('total_pairs', 0) if analyze_phase_harmonics else 0) +
                                     (cross_phase_analysis.get('total_pairs', 0) if analyze_cross_phase else 0))
        },
        'first_order_filters': first_order_analysis,
        'second_order_filters': second_order_analysis,
        'low_pass_filter': low_pass_analysis,
        'frequency_summary': {
            'min_freq_hz': min([f['center_freq_hz'] for f in first_order_analysis + second_order_analysis]),
            'max_freq_hz': max([f['center_freq_hz'] for f in first_order_analysis + second_order_analysis]),
            'first_order_range_hz': (
                min([f['center_freq_hz'] for f in first_order_analysis]),
                max([f['center_freq_hz'] for f in first_order_analysis])
            ),
            'second_order_range_hz': (
                min([f['center_freq_hz'] for f in second_order_analysis]),
                max([f['center_freq_hz'] for f in second_order_analysis])
            )
        },
        'phase_harmonic_analysis': phase_harmonic_analysis if analyze_phase_harmonics else None,
        'cross_phase_analysis': cross_phase_analysis if analyze_cross_phase else None,
        'second_order_detailed': second_order_detailed
    }

def print_detailed_analysis(analysis: Dict):
    """Print a detailed analysis of the scattering transform frequency characteristics."""
    
    print("=" * 80)
    print("SCATTERING TRANSFORM FREQUENCY ANALYSIS")
    print("=" * 80)
    
    # Signal information
    print("\n📊 SIGNAL CHARACTERISTICS:")
    print(f"  • Sampling rate: {analysis['signal_info']['sampling_rate_hz']} Hz")
    print(f"  • Duration: {analysis['signal_info']['duration_minutes']} minutes ({analysis['signal_info']['duration_seconds']} seconds)")
    print(f"  • Total samples: {analysis['signal_info']['num_samples']}")
    print(f"  • Nyquist frequency: {analysis['signal_info']['nyquist_freq_hz']} Hz")
    
    # Scattering parameters
    print(f"\n🔧 SCATTERING PARAMETERS:")
    print(f"  • J (max scale): {analysis['scattering_params']['J']}")
    print(f"  • Q (wavelets per octave): {analysis['scattering_params']['Q']}")
    print(f"  • T (low-pass support): {analysis['scattering_params']['T']}")
    print(f"  • σ_min: {analysis['scattering_params']['sigma_min']:.6f}")
    
    # Coefficient counts
    print(f"\n📈 SCATTERING COEFFICIENTS:")
    counts = analysis['coefficient_counts']
    print(f"  • Zeroth order (S0): {counts['zeroth_order']} coefficient")
    print(f"  • First order (S1): {counts['first_order']} coefficients")
    print(f"  • Second order (S2): {counts['second_order']} coefficients")
    print(f"  • Total scattering: {counts['total_scattering']} coefficients")
    
    if counts['phase_harmonics'] > 0:
        print(f"\n🌊 PHASE HARMONIC COEFFICIENTS:")
        print(f"  • Within-channel pairs: {counts['phase_harmonics']} coefficients")
        phase_analysis = analysis['phase_harmonic_analysis']
        if phase_analysis:
            print(f"  • Auto-correlations: {phase_analysis['auto_correlations']} (phase stability)")
            print(f"  • Cross-correlations: {phase_analysis['cross_correlations']} (frequency coupling)")
    
    if counts['cross_phase_harmonics'] > 0:
        print(f"\n🔄 CROSS-CHANNEL PHASE COEFFICIENTS:")
        print(f"  • Cross-channel pairs: {counts['cross_phase_harmonics']} coefficients")
        cross_analysis = analysis['cross_phase_analysis']
        if cross_analysis:
            print(f"  • Same filter pairs: {cross_analysis['auto_correlations']} (inter-signal coherence)")
            print(f"  • Different filter pairs: {cross_analysis['cross_correlations']} (complex coupling)")
    
    print(f"\n📊 TOTAL COEFFICIENT COUNT: {counts['total_all_coefficients']} coefficients")
    
    # Frequency ranges
    print(f"\n🎵 FREQUENCY RANGES:")
    freq_sum = analysis['frequency_summary']
    print(f"  • Overall range: {freq_sum['min_freq_hz']:.4f} - {freq_sum['max_freq_hz']:.4f} Hz")
    print(f"  • First-order range: {freq_sum['first_order_range_hz'][0]:.4f} - {freq_sum['first_order_range_hz'][1]:.4f} Hz")
    print(f"  • Second-order range: {freq_sum['second_order_range_hz'][0]:.4f} - {freq_sum['second_order_range_hz'][1]:.4f} Hz")
    
    # Low-pass filter
    print(f"\n🔽 LOW-PASS FILTER:")
    lp = analysis['low_pass_filter']
    print(f"  • σ_low (normalized): {lp['sigma_normalized']:.6f}")
    print(f"  • Bandwidth: {lp['bandwidth_hz']:.4f} Hz")
    print(f"  • Cutoff frequency: {lp['cutoff_freq_hz']:.4f} Hz")
    
    # First-order filters detail
    print(f"\n🔵 FIRST-ORDER FILTERS (Q={analysis['scattering_params']['Q']}):")
    print("   #  |   ξ (norm)  |  σ (norm)  |  Freq (Hz)  |  BW (Hz)   |  Range (Hz)        | Physiol. Band    | Max j")
    print("   ---|-------------|------------|-------------|------------|-------------------|------------------|------")
    for f in analysis['first_order_filters']:
        range_str = f"{f['frequency_range_hz'][0]:.3f}-{f['frequency_range_hz'][1]:.3f}"
        print(f"   {f['filter_index']:2d} |   {f['xi_normalized']:.6f} |  {f['sigma_normalized']:.6f} |   {f['center_freq_hz']:7.4f} |  {f['bandwidth_hz']:8.4f} | {range_str:17s} | {f['physiological_band']:16s} | {f['max_subsampling_j']:4d}")
    
    # Second-order filters detail  
    print(f"\n🔴 SECOND-ORDER FILTERS (Q=1):")
    print("   #  |   ξ (norm)  |  σ (norm)  |  Freq (Hz)  |  BW (Hz)   |  Range (Hz)        | Physiol. Band    | Max j")
    print("   ---|-------------|------------|-------------|------------|-------------------|------------------|------")
    for f in analysis['second_order_filters']:
        range_str = f"{f['frequency_range_hz'][0]:.3f}-{f['frequency_range_hz'][1]:.3f}"
        print(f"   {f['filter_index']:2d} |   {f['xi_normalized']:.6f} |  {f['sigma_normalized']:.6f} |   {f['center_freq_hz']:7.4f} |  {f['bandwidth_hz']:8.4f} | {range_str:17s} | {f['physiological_band']:16s} | {f['max_subsampling_j']:4d}")
    
    # Phase harmonic details
    if analysis.get('phase_harmonic_analysis'):
        print(f"\n🌊 PHASE HARMONIC PAIRS:")
        phase_analysis = analysis['phase_harmonic_analysis']
        print(f"   Total pairs: {phase_analysis['total_pairs']} | Auto-correlations: {phase_analysis['auto_correlations']} | Cross-correlations: {phase_analysis['cross_correlations']}")
        print("   Pair# | Filter i→j | Freq i→j (Hz)     | Power  | Harmonic Type")
        print("   ------|------------|-------------------|--------|-----------------")
        
        # Show ALL pairs - no truncation
        for pair in phase_analysis['pairs_detail']:
            freq_str = f"{pair['xi_i_hz']:.3f}→{pair['xi_j_hz']:.3f}"
            print(f"   {pair['pair_index']:4d}  |     {pair['filter_i']:2d}→{pair['filter_j']:2d}    | {freq_str:17s} | {pair['power']:6.2f} | {pair['harmonic_type']}")
    
    # Cross-channel phase harmonic details
    if analysis.get('cross_phase_analysis'):
        print(f"\n🔄 CROSS-CHANNEL PHASE PAIRS:")
        cross_analysis = analysis['cross_phase_analysis']
        print(f"   Total pairs: {cross_analysis['total_pairs']} | Same filter across channels: {cross_analysis['auto_correlations']} | Cross-correlations: {cross_analysis['cross_correlations']}")
        print(f"   Applications: Inter-signal coherence, bilateral symmetry, cross-system coupling")
        print("   Pair# | Filter i→j | Freq i→j (Hz)     | Power  | Cross-Channel Type")
        print("   ------|------------|-------------------|--------|-----------------")
        
        # Show ALL cross-channel pairs - no truncation
        for pair in cross_analysis['pairs_detail']:
            freq_str = f"{pair['xi_i_hz']:.3f}→{pair['xi_j_hz']:.3f}"
            cross_type = "Same filter" if pair['is_auto'] else pair['harmonic_type']
            print(f"   {pair['pair_index']:4d}  |     {pair['filter_i']:2d}→{pair['filter_j']:2d}    | {freq_str:17s} | {pair['power']:6.2f} | {cross_type}")
    
    # Second-order interaction details
    if analysis.get('second_order_detailed'):
        print(f"\n🔴 SECOND-ORDER INTERACTIONS:")
        s2_detailed = analysis['second_order_detailed']
        print(f"   Pair# | S1[i]→S2[j] | Frequency Interaction     | Physiological Relevance")
        print("   ------|-------------|---------------------------|---------------------------")
        
        # Show ALL second-order pairs - no truncation
        for pair in s2_detailed:
            print(f"   {pair['pair_index']:4d}  |   {pair['first_filter_idx']:2d}→{pair['second_filter_idx']:2d}     | {pair['frequency_interaction']:25s} | {pair['physiological_relevance']}")

if __name__ == "__main__":
    # Your specific parameters
    J = 11
    Q = 4
    T = 16
    sampling_rate = 4.0  # Hz
    signal_duration_minutes = 20.0  # minutes
    
    # Perform analysis with phase harmonics
    analysis = analyze_scattering_frequencies(
        J, Q, T, sampling_rate, signal_duration_minutes,
        analyze_phase_harmonics=True,
        analyze_cross_phase=True
    )
    
    # Print detailed results
    print_detailed_analysis(analysis)
    
    print(f"\n" + "=" * 80)
    print("SUMMARY FOR YOUR CONFIGURATION:")
    print("=" * 80)
    print(f"With J={J}, Q={Q}, T={T} at {sampling_rate}Hz sampling:")
    counts = analysis['coefficient_counts']
    print(f"• Scattering coefficients: {counts['total_scattering']} total")
    print(f"• Phase harmonic coefficients: {counts['phase_harmonics']} total")
    print(f"• Cross-channel phase coefficients: {counts['cross_phase_harmonics']} total")
    print(f"• TOTAL ALL COEFFICIENTS: {counts['total_all_coefficients']} coefficients")
    print(f"• First-order coefficients cover {analysis['frequency_summary']['first_order_range_hz'][0]:.4f}-{analysis['frequency_summary']['first_order_range_hz'][1]:.4f} Hz")
    print(f"• Second-order coefficients cover {analysis['frequency_summary']['second_order_range_hz'][0]:.4f}-{analysis['frequency_summary']['second_order_range_hz'][1]:.4f} Hz")
    print(f"• This captures frequency content from very low frequencies up to {analysis['frequency_summary']['max_freq_hz']:.4f} Hz")
    print(f"• All frequencies are well below Nyquist ({analysis['signal_info']['nyquist_freq_hz']} Hz)")
    
    # Analysis recommendations
    print(f"\n📋 COEFFICIENT SELECTION RECOMMENDATIONS:")
    print(f"• For FHR prediction: Focus on first-order + phase harmonics ({counts['first_order'] + counts['phase_harmonics']} coefficients)")
    print(f"• For multi-channel analysis: Add cross-phase harmonics (+{counts['cross_phase_harmonics']} coefficients)")
    print(f"• For complex pattern detection: Include selected second-order coefficients")
    print(f"• Dimensionality reduction: Use subset selection to reduce from {counts['total_all_coefficients']} to 50-100 most informative coefficients")