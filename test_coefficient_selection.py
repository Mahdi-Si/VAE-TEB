#!/usr/bin/env python3
"""
Test script to analyze optimal coefficient selection for FHR analysis
with J=11, Q=4, T=16 configuration.
"""

import sys
import os
sys.path.append('/mnt/c/Users/mahdi/Desktop/teb_vae_model/hdf5_dataset')

import torch
import numpy as np
from kymatio_phase_scattering import KymatioPhaseScattering1D

def test_coefficient_selection():
    """Test optimal coefficient selection for FHR analysis."""
    
    # Configuration
    J, Q, T = 11, 4, 16
    shape = 4800  # 20 minutes at 4 Hz
    
    print(f"Testing coefficient selection for J={J}, Q={Q}, T={T}")
    print("=" * 60)
    
    # Initialize scattering transform
    scattering = KymatioPhaseScattering1D(
        J=J, Q=Q, T=T, shape=shape, max_order=1  # First order only
    )
    
    # Get optimal coefficients
    selection = scattering.get_optimal_coefficients_for_fhr(J, Q, T)
    
    # Print configuration analysis
    config = selection['config_analysis']
    print("CONFIGURATION ANALYSIS:")
    print(f"Current config: {config['current_config']}")
    print(f"Total scattering coefficients: {config['total_scattering_coeffs']}")
    print(f"Selected phase coefficients: {config['selected_phase_coeffs']}")
    print(f"Selected cross-channel coefficients: {config['selected_cross_coeffs']}")
    print(f"Phase reduction: {config['efficiency_gain']['phase_reduction']}")
    print()
    
    # Analyze phase selection
    phase_sel = selection['phase_selection']
    print("PHASE COEFFICIENT SELECTION:")
    print(f"Total possible phase pairs: {phase_sel['metadata']['total_pairs']}")
    print(f"Selected pairs: {phase_sel['metadata']['selected_pairs']}")
    print(f"Frequency range: {phase_sel['metadata']['frequency_range'][0]:.6f} - {phase_sel['metadata']['frequency_range'][1]:.6f} Hz")
    print(f"Selected freq range: {phase_sel['metadata']['selected_freq_range'][0]:.6f} - {phase_sel['metadata']['selected_freq_range'][1]:.6f} Hz")
    print(f"Power range: {phase_sel['metadata']['power_range'][0]:.2f} - {phase_sel['metadata']['power_range'][1]:.2f}")
    print()
    
    # Breakdown by mask type
    print("Phase Selection Breakdown:")
    for mask_name, mask in phase_sel['masks'].items():
        count = mask.sum().item()
        print(f"  {mask_name}: {count} coefficients")
    print()
    
    # Analyze cross-channel selection
    cross_sel = selection['cross_selection']
    print("CROSS-CHANNEL COEFFICIENT SELECTION:")
    print(f"Total possible cross pairs: {cross_sel['metadata']['total_pairs']}")
    print(f"Selected cross pairs: {cross_sel['metadata']['cross_selected_pairs']}")
    print(f"UP frequency range: {cross_sel['metadata']['up_freq_range'][0]:.3f} - {cross_sel['metadata']['up_freq_range'][1]:.3f} Hz")
    print(f"FHR frequency range: {cross_sel['metadata']['fhr_freq_range'][0]:.3f} - {cross_sel['metadata']['fhr_freq_range'][1]:.3f} Hz")
    print(f"UP filters available: {cross_sel['metadata']['up_filters_available']}")
    print(f"FHR filters available: {cross_sel['metadata']['fhr_filters_available']}")
    print(f"Power range: {cross_sel['metadata']['power_range'][0]:.2f} - {cross_sel['metadata']['power_range'][1]:.2f}")
    print()
    
    # Show specific filter frequencies for selected coefficients
    print("SELECTED PHASE COEFFICIENT DETAILS:")
    phase_mask = selection['recommendations']['use_phase_mask']
    if phase_mask.any():
        selected_i = scattering.i_idx[phase_mask]
        selected_j = scattering.j_idx[phase_mask]
        selected_powers = scattering.powers[phase_mask]
        
        print("Filter pairs (i->j) with frequencies and powers:")
        for k in range(min(10, len(selected_i))):  # Show first 10
            i_freq = scattering.center_freqs[selected_i[k]].item()
            j_freq = scattering.center_freqs[selected_j[k]].item()
            power = selected_powers[k].item()
            print(f"  {selected_i[k].item():2d}->{selected_j[k].item():2d}: {i_freq:.6f}Hz -> {j_freq:.6f}Hz (power={power:.2f})")
        if len(selected_i) > 10:
            print(f"  ... and {len(selected_i) - 10} more pairs")
    print()
    
    print("SELECTED CROSS-CHANNEL COEFFICIENT DETAILS:")
    cross_mask = selection['recommendations']['use_cross_mask']
    if cross_mask.any():
        selected_i = scattering.i_idx[cross_mask]
        selected_j = scattering.j_idx[cross_mask]
        selected_powers = scattering.powers[cross_mask]
        
        print("UP->FHR filter pairs with frequencies and powers:")
        for k in range(min(10, len(selected_i))):  # Show first 10
            up_freq = scattering.center_freqs[selected_i[k]].item()
            fhr_freq = scattering.center_freqs[selected_j[k]].item()
            power = selected_powers[k].item()
            print(f"  UP{selected_i[k].item():2d}->FHR{selected_j[k].item():2d}: {up_freq:.6f}Hz -> {fhr_freq:.6f}Hz (power={power:.2f})")
        if len(selected_i) > 10:
            print(f"  ... and {len(selected_i) - 10} more pairs")
    print()
    
    # Final recommendations
    print("FINAL RECOMMENDATIONS:")
    recs = selection['recommendations']
    print(f"Total selected features: {recs['total_selected_features']}")
    print(f"Use phase mask shape: {recs['use_phase_mask'].shape} ({recs['use_phase_mask'].sum()} selected)")
    print(f"Use cross mask shape: {recs['use_cross_mask'].shape} ({recs['use_cross_mask'].sum()} selected)")
    
    return selection

if __name__ == "__main__":
    selection = test_coefficient_selection()