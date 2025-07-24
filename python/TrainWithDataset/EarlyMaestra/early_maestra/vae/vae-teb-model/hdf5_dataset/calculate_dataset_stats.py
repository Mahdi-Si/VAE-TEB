import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from hdf5_dataset import normalize_tensor_data
import argparse


class DatasetStatsCalculator:
    """
    Calculate statistics (mean and variance) for HDF5 datasets created with create_initial_hdf5.
    
    Updated for optimal coefficient selection (J=11, Q=4, T=16):
    - FHR scattering: 43 coefficients (first order, channel 0 regular, others log-transformed)
    - FHR phase: 44 selected coefficients (all asinh-transformed)
    - FHR-UP cross-phase: 130 selected coefficients (all asinh-transformed)
    
    Efficiently computes statistics using online algorithms to handle large datasets
    that may not fit entirely in memory.
    
    Transformation strategy:
    - fhr_st (43 channels): channel 0 regular, channels 1-42 log-transformed  
    - fhr_ph (44 channels): all asinh-transformed for phase stability
    - fhr_up_ph (130 channels): all asinh-transformed for cross-phase correlation
    """
    
    def __init__(self, trim_minutes: Optional[float] = None, device: Optional[str] = None):
        self.stats_fields = ['fhr', 'up', 'fhr_st', 'fhr_ph', 'fhr_up_ph']
        self.trim_minutes = trim_minutes

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")

        if self.trim_minutes is not None:
            self.trim_samples_raw = int(4 * 60 * self.trim_minutes)
            self.trim_samples_decimated = self.trim_samples_raw // 16
        else:
            self.trim_samples_raw = 0
            self.trim_samples_decimated = 0
        
        # Define transformations for optimal coefficient selection
        # LOG normalization for scattering coefficients (except order 0)
        self.log_norm_channels_config = {
            'fhr_st': 'all_except_0',  # 42 of 43 scattering coefficients (exclude channel 0)
        }
        
        # ASINH normalization for phase coefficients (better for phase data)
        self.asinh_norm_channels_config = {
            'fhr_ph': 'all',    # All 44 selected phase coefficients
            'fhr_up_ph': 'all'  # All 130 selected cross-phase coefficients
        }
        
    def _initialize_stats(self, field_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Dict[str, Any]]:
        """
        Initialize statistics accumulators for each field.
        
        Args:
            field_shapes: Dictionary mapping field names to their shapes (excluding batch dimension)
            
        Returns:
            Dictionary containing initialized statistics accumulators
        """
        stats = {}
        
        for field in self.stats_fields:
            if field not in field_shapes:
                continue
                
            shape = field_shapes[field]
            
            if field in ['fhr', 'up']:
                # Single-channel data: compute global mean/variance
                stats[field] = {
                    'count': 0,
                    'sum': torch.tensor(0.0, dtype=torch.float64, device=self.device),
                    'sum_squares': torch.tensor(0.0, dtype=torch.float64, device=self.device),
                    'mean': 0.0,
                    'variance': 0.0,
                    'shape': shape
                }
            else:
                # Multi-channel data: compute per-channel mean/variance
                # Separate statistics for order 0 and order 1+ channels
                n_channels = shape[0]
                
                # Determine which transform to use for each channel
                log_channels_config = self.log_norm_channels_config.get(field, [])
                asinh_channels_config = self.asinh_norm_channels_config.get(field, [])

                log_channels = []
                if log_channels_config == 'all_except_0':
                    if n_channels > 0:
                        log_channels = [c for c in range(n_channels) if c != 0]
                elif isinstance(log_channels_config, list):
                    log_channels = log_channels_config
                
                asinh_channels = []
                if asinh_channels_config == 'all':
                    asinh_channels = list(range(n_channels))
                elif isinstance(asinh_channels_config, list):
                    asinh_channels = asinh_channels_config

                if set(log_channels) & set(asinh_channels):
                    warnings.warn(f"Field {field} has overlapping channels for log and asinh normalization. Log takes precedence.")
                    asinh_channels = [c for c in asinh_channels if c not in log_channels]

                regular_channels = [c for c in range(n_channels) if c not in log_channels and c not in asinh_channels]
                
                stats[field] = {
                    'count': 0,
                    'sum': torch.zeros(n_channels, dtype=torch.float64, device=self.device),
                    'sum_squares': torch.zeros(n_channels, dtype=torch.float64, device=self.device),
                    'mean': np.zeros(n_channels, dtype=np.float32),
                    'variance': np.zeros(n_channels, dtype=np.float32),
                    'shape': shape,
                    'n_channels': n_channels,
                    'regular_channels': regular_channels,
                    'log_channels': log_channels,
                    'asinh_channels': asinh_channels,
                    'log_epsilon': 1e-6  # For log transformation
                }
                
        return stats
    
    def _update_single_channel_stats(
        self, 
        stats: Dict[str, Any], 
        data: np.ndarray
    ) -> None:
        """
        Update statistics for single-channel data (fhr, up).
        
        Args:
            stats: Statistics accumulator for this field
            data: Data array of shape (batch_size, signal_length)
        """
        # Convert numpy data to torch tensor on the specified device
        data_tensor = torch.from_numpy(data.astype(np.float64)).to(self.device)
        
        # Flatten all data points
        flat_data = data_tensor.flatten()
        
        # Remove any NaN or infinite values
        valid_mask = torch.isfinite(flat_data)
        valid_data = flat_data[valid_mask]
        
        if valid_data.numel() == 0:
            return
            
        stats['count'] += valid_data.numel()
        stats['sum'] += torch.sum(valid_data)
        stats['sum_squares'] += torch.sum(valid_data ** 2)
    
    def _update_multi_channel_stats(
        self, 
        stats: Dict[str, Any], 
        data: np.ndarray
    ) -> None:
        """
        Update statistics for multi-channel data (fhr_st, fhr_ph, fhr_up_ph).
        Applies log transformation to order 1+ coefficients.
        
        Args:
            stats: Statistics accumulator for this field
            data: Data array of shape (batch_size, n_channels, sequence_length)
        """
        batch_size, n_channels, seq_length = data.shape
        regular_channels = stats.get('regular_channels', [])
        log_channels = stats.get('log_channels', [])
        asinh_channels = stats.get('asinh_channels', [])
        log_epsilon = stats.get('log_epsilon', 1e-6)
        
        # Convert numpy data to torch tensor on the specified device
        data_tensor = torch.from_numpy(data.astype(np.float64)).to(self.device)
        
        for channel in range(n_channels):
            # Extract data for this channel
            channel_data = data_tensor[:, channel, :].flatten()
            
            # Remove any NaN or infinite values
            valid_mask = torch.isfinite(channel_data)
            
            # Apply appropriate transformation based on coefficient order
            if channel in regular_channels:
                # Regular data (no transformation)
                valid_data = channel_data[valid_mask]
            elif channel in log_channels:
                # Log transformation
                channel_data = torch.maximum(channel_data, torch.tensor(0.0, device=self.device, dtype=torch.float64))
                log_data = torch.log(channel_data + log_epsilon)
                valid_mask = valid_mask & torch.isfinite(log_data)
                valid_data = log_data[valid_mask]
            elif channel in asinh_channels:
                # Hyperbolic sine transformation
                asinh_data = torch.asinh(channel_data)
                valid_mask = valid_mask & torch.isfinite(asinh_data)
                valid_data = asinh_data[valid_mask]
            else:
                # Fallback to regular for any channel not specified
                valid_data = channel_data[valid_mask]
            
            if valid_data.numel() == 0:
                continue
                
            # Update statistics for this channel
            # Note: count is now per-channel to handle varying numbers of valid data points
            if 'channel_counts' not in stats:
                stats['channel_counts'] = torch.zeros(n_channels, dtype=torch.int64, device=self.device)
                
            stats['channel_counts'][channel] += valid_data.numel()
            stats['sum'][channel] += torch.sum(valid_data)
            stats['sum_squares'][channel] += torch.sum(valid_data ** 2)
    
    def _finalize_stats(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """
        Calculate final mean and variance from accumulated sums.
        
        Args:
            stats: Statistics accumulators to finalize
        """
        for field, field_stats in stats.items():
            if field in ['fhr', 'up']:
                count = field_stats['count']
                if count == 0:
                    warnings.warn(f"No valid data found for field '{field}'")
                    continue

                # Single-channel calculation
                mean = field_stats['sum'] / count
                variance = (field_stats['sum_squares'] / count) - (mean ** 2)
                
                field_stats['mean'] = mean.item()
                field_stats['variance'] = max(0.0, variance.item())  # Ensure non-negative
                
            else:
                # Multi-channel calculation
                if 'channel_counts' not in field_stats or torch.sum(field_stats['channel_counts']) == 0:
                    warnings.warn(f"No valid data found for field '{field}'")
                    continue
                    
                n_channels = field_stats['n_channels']
                count_per_channel = field_stats['channel_counts']

                # To avoid division by zero for channels with no data, set count to 1 (mean will be 0)
                # These channels will have zero variance anyway.
                count_per_channel_safe = torch.where(count_per_channel > 0, count_per_channel, 1)

                # Vectorized calculation for all channels
                mean_vec = field_stats['sum'] / count_per_channel_safe
                variance_vec = (field_stats['sum_squares'] / count_per_channel_safe) - (mean_vec ** 2)
                
                # For channels with no data, mean and variance should be 0
                mean_vec[count_per_channel == 0] = 0
                variance_vec[count_per_channel == 0] = 0

                # Ensure variance is non-negative
                variance_vec[variance_vec < 0] = 0
                
                # Move to CPU and convert to numpy for storage
                field_stats['mean'] = mean_vec.cpu().numpy().astype(np.float32)
                field_stats['variance'] = variance_vec.cpu().numpy().astype(np.float32)
                
                # Update total count for summary purposes
                field_stats['count'] = torch.sum(field_stats['channel_counts']).item()
    
    def calculate_stats(
        self, 
        hdf5_files: List[str], 
        batch_size: int = 100,
        progress_bar: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics across multiple HDF5 files.
        
        Args:
            hdf5_files: List of paths to HDF5 files
            batch_size: Number of samples to process at once for memory efficiency
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing calculated statistics for each field
        """
        # Validate input files
        valid_files = []
        for file_path in hdf5_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                warnings.warn(f"File not found: {file_path}")
        
        if not valid_files:
            raise ValueError("No valid HDF5 files found")
        
        # Get field shapes from first file
        field_shapes = {}
        with h5py.File(valid_files[0], 'r') as f:
            for field in self.stats_fields:
                if field in f:
                    shape = list(f[field].shape[1:])  # Exclude batch dimension
                    if self.trim_minutes is not None:
                        if field in ['fhr', 'up']:
                            shape[-1] -= 2 * self.trim_samples_raw
                        else:
                            shape[-1] -= 2 * self.trim_samples_decimated
                    field_shapes[field] = tuple(shape)
        
        if not field_shapes:
            raise ValueError("No statistics fields found in HDF5 files")
        
        # Initialize statistics
        stats = self._initialize_stats(field_shapes)
        
        # Process each file
        total_samples = 0
        for file_path in (tqdm(valid_files, desc="Processing files") if progress_bar else valid_files):
            with h5py.File(file_path, 'r') as f:
                file_samples = f[next(iter(self.stats_fields))].shape[0]
                total_samples += file_samples
                
                # Process in batches for memory efficiency
                for start_idx in (tqdm(range(0, file_samples, batch_size), 
                                     desc=f"Processing {os.path.basename(file_path)}", 
                                     leave=False) if progress_bar else range(0, file_samples, batch_size)):
                    
                    end_idx = min(start_idx + batch_size, file_samples)
                    
                    # Load batch data for each field
                    for field in self.stats_fields:
                        if field not in f:
                            continue
                            
                        data = f[field][start_idx:end_idx]

                        if self.trim_minutes is not None:
                            if field in ['fhr', 'up']:
                                start_trim = self.trim_samples_raw
                                end_trim = -self.trim_samples_raw if self.trim_samples_raw > 0 else None
                                data = data[:, start_trim:end_trim]
                            else:
                                start_trim = self.trim_samples_decimated
                                end_trim = -self.trim_samples_decimated if self.trim_samples_decimated > 0 else None
                                data = data[:, :, start_trim:end_trim]
                        
                        if field in ['fhr', 'up']:
                            self._update_single_channel_stats(stats[field], data)
                        else:
                            self._update_multi_channel_stats(stats[field], data)
        
        # Finalize statistics
        self._finalize_stats(stats)
        
        print(f"\nProcessed {total_samples} total samples from {len(valid_files)} files")
        return stats
    
    def save_stats(
        self, 
        stats: Dict[str, Dict[str, Any]], 
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save calculated statistics to an HDF5 file.
        
        Args:
            stats: Statistics dictionary from calculate_stats
            output_path: Path for output HDF5 file
            metadata: Optional metadata to include
        """
        # Remove existing file
        try:
            os.remove(output_path)
        except OSError:
            pass
        
        with h5py.File(output_path, 'w', libver='latest') as f:
            # Save metadata
            if metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    meta_group.attrs[key] = value
            
            # Save timestamp
            import datetime
            f.attrs['created_at'] = datetime.datetime.now().isoformat()
            f.attrs['trim_minutes'] = self.trim_minutes if self.trim_minutes is not None else -1.0
            
            # Save information about log transformation
            f.attrs['log_epsilon'] = 1e-6
            f.attrs['description'] = 'Statistics for optimal coefficient selection: 43 scattering (log), 44 phase (asinh), 130 cross-phase (asinh)'
            
            # Save statistics for each field
            for field, field_stats in stats.items():
                field_group = f.create_group(field)
                
                # Save basic info
                field_group.attrs['shape'] = field_stats['shape']
                field_group.attrs['count'] = field_stats['count']
                
                if field in ['fhr', 'up']:
                    # Single-channel data
                    field_group.create_dataset('mean', data=field_stats['mean'], dtype='f4')
                    field_group.create_dataset('variance', data=field_stats['variance'], dtype='f4')
                    field_group.create_dataset('std', data=np.sqrt(field_stats['variance']), dtype='f4')
                    
                    # Store as scalars as well for convenience
                    field_group.attrs['mean_scalar'] = field_stats['mean']
                    field_group.attrs['variance_scalar'] = field_stats['variance']
                    field_group.attrs['std_scalar'] = np.sqrt(field_stats['variance'])
                    
                else:
                    # Multi-channel data
                    field_group.attrs['n_channels'] = field_stats['n_channels']
                    field_group.create_dataset('mean', data=field_stats['mean'], dtype='f4')
                    field_group.create_dataset('variance', data=field_stats['variance'], dtype='f4')
                    field_group.create_dataset('std', data=np.sqrt(field_stats['variance']), dtype='f4')
                    
                    # Save information about which channels use log transformation
                    if 'regular_channels' in field_stats:
                        field_group.attrs['regular_channels'] = field_stats.get('regular_channels', [])
                        field_group.attrs['log_channels'] = field_stats.get('log_channels', [])
                        field_group.attrs['asinh_channels'] = field_stats.get('asinh_channels', [])
                        field_group.attrs['uses_log_transform'] = len(field_stats.get('log_channels', [])) > 0
                        field_group.attrs['uses_asinh_transform'] = len(field_stats.get('asinh_channels', [])) > 0
                    elif 'order0_channels' in field_stats: # Backward compatibility for old stats files
                        order0_channels = field_stats['order0_channels']
                        field_group.attrs['order0_channels'] = order0_channels
                        field_group.attrs['log_transformed_channels'] = [i for i in range(field_stats['n_channels']) 
                                                                        if i not in order0_channels]
                        field_group.attrs['uses_log_transform'] = True
                        field_group.attrs['uses_asinh_transform'] = False
                    else:
                        field_group.attrs['uses_log_transform'] = False
                        field_group.attrs['uses_asinh_transform'] = False
        
        print(f"Statistics saved to: {output_path}")
    
    def load_stats(self, stats_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load previously calculated statistics from HDF5 file.
        
        Args:
            stats_path: Path to statistics HDF5 file
            
        Returns:
            Dictionary containing loaded statistics
        """
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        
        stats = {}
        with h5py.File(stats_path, 'r') as f:
            # Load global metadata
            log_epsilon = f.attrs.get('log_epsilon', 1e-6)
            
            for field in f.keys():
                if field == 'metadata':
                    continue
                    
                field_group = f[field]
                field_stats = {
                    'shape': tuple(field_group.attrs['shape']),
                    'count': field_group.attrs['count'],
                    'log_epsilon': log_epsilon
                }
                
                if field in ['fhr', 'up']:
                    # Single-channel data
                    field_stats['mean'] = field_group.attrs['mean_scalar']
                    field_stats['variance'] = field_group.attrs['variance_scalar']
                    field_stats['std'] = field_group.attrs['std_scalar']
                else:
                    # Multi-channel data
                    field_stats['n_channels'] = field_group.attrs['n_channels']
                    field_stats['mean'] = field_group['mean'][()]
                    field_stats['variance'] = field_group['variance'][()]
                    field_stats['std'] = field_group['std'][()]
                    
                    # Load log transformation metadata if available
                    if 'regular_channels' in field_group.attrs:
                        field_stats['regular_channels'] = list(field_group.attrs.get('regular_channels', []))
                        field_stats['log_channels'] = list(field_group.attrs.get('log_channels', []))
                        field_stats['asinh_channels'] = list(field_group.attrs.get('asinh_channels', []))
                        field_stats['uses_log_transform'] = field_group.attrs.get('uses_log_transform', False)
                        field_stats['uses_asinh_transform'] = field_group.attrs.get('uses_asinh_transform', False)
                    # Backward compatibility for 'order0_channels'
                    elif 'order0_channels' in field_group.attrs:
                        order0_channels = list(field_group.attrs.get('order0_channels', []))
                        log_channels = [i for i in range(field_stats['n_channels']) if i not in order0_channels]
                        field_stats['regular_channels'] = order0_channels
                        field_stats['log_channels'] = log_channels
                        field_stats['asinh_channels'] = []
                        field_stats['uses_log_transform'] = True
                        field_stats['uses_asinh_transform'] = False
                    else:
                        # Fallback for very old stats files
                        field_stats['regular_channels'] = []
                        field_stats['log_channels'] = []
                        field_stats['asinh_channels'] = []
                        field_stats['uses_log_transform'] = False
                        field_stats['uses_asinh_transform'] = False
                
                stats[field] = field_stats
        
        return stats
    
    def print_stats_summary(self, stats: Dict[str, Dict[str, Any]]) -> None:
        """
        Print a human-readable summary of the statistics.
        
        Args:
            stats: Statistics dictionary
        """
        print("\n" + "="*60)
        print("DATASET STATISTICS SUMMARY")
        print("="*60)
        print("Note: Optimal coefficient selection with specialized transformations:")
        print("- FHR scattering (43 ch): channel 0 regular, others log(x + 1e-6)")
        print("- FHR phase (44 ch): all asinh(x) transformed")
        print("- FHR-UP cross-phase (130 ch): all asinh(x) transformed")
        
        for field, field_stats in stats.items():
            print(f"\n{field.upper()}:")
            print(f"  Shape: {field_stats['shape']}")
            print(f"  Total samples: {field_stats['count']:,}")
            
            if field in ['fhr', 'up']:
                # Single-channel data
                mean = field_stats['mean']
                variance = field_stats['variance']
                std = np.sqrt(variance)
                print(f"  Mean: {mean:.6f}")
                print(f"  Variance: {variance:.6f}")
                print(f"  Std: {std:.6f}")
                print(f"  Transformation: Regular normalization")
            else:
                # Multi-channel data
                n_channels = field_stats['n_channels']
                means = field_stats['mean']
                variances = field_stats['variance']
                stds = np.sqrt(variances)
                regular_channels = field_stats.get('regular_channels', [])
                log_channels = field_stats.get('log_channels', [])
                asinh_channels = field_stats.get('asinh_channels', [])
                
                print(f"  Channels: {n_channels}")
                if regular_channels: print(f"  Regular channels: {regular_channels}")
                if log_channels: print(f"  Log-transformed channels: {log_channels}")
                if asinh_channels: print(f"  Asinh-transformed channels: {asinh_channels}")
                    
                print(f"  Mean (per channel):")
                for i, mean in enumerate(means):
                    transform_type = "regular"
                    if i in log_channels: transform_type = "log"
                    elif i in asinh_channels: transform_type = "asinh"
                    print(f"    Ch {i:2d}: {mean:.6f} ({transform_type})")
                print(f"  Variance (per channel):")
                for i, var in enumerate(variances):
                    transform_type = "regular"
                    if i in log_channels: transform_type = "log"
                    elif i in asinh_channels: transform_type = "asinh"
                    print(f"    Ch {i:2d}: {var:.6f} ({transform_type})")
                print(f"  Std (per channel):")
                for i, std in enumerate(stds):
                    transform_type = "regular"
                    if i in log_channels: transform_type = "log"
                    elif i in asinh_channels: transform_type = "asinh"
                    print(f"    Ch {i:2d}: {std:.6f} ({transform_type})")

    def plot_histograms(
        self,
        hdf5_files: List[str],
        save_path: str,
        max_channels: Optional[int] = 5,
        max_samples: int = 50000,
        figsize: Tuple[int, int] = (15, 10),
        bins: int = 50,
        progress_bar: bool = True
    ) -> None:
        """
        Plot histograms of the 5-95th percentile of raw and normalized values for each field.
        This removes 10% of outliers from the view to provide a clearer histogram.
        
        Args:
            hdf5_files: List of paths to HDF5 files
            save_path: Directory where the output figures will be saved.
            max_channels: Maximum number of channels to plot for multi-channel data (0:max_channels).
                          If None or <= 0, all channels will be plotted.
            max_samples: Maximum number of samples to use for plotting (for memory efficiency)
            bins: The number of bins to use for the histogram.
            progress_bar: Whether to show progress bar
        """
        # First, calculate statistics which are needed for normalization
        print("Calculating statistics for normalization...")
        stats = self.calculate_stats(hdf5_files, progress_bar=progress_bar)
        
        # Validate input files
        valid_files = []
        for file_path in hdf5_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                warnings.warn(f"File not found: {file_path}")
        if not valid_files:
            print("No data available for plotting")
            return
        
        print(f"Collecting sample data for histogram plotting (max {max_samples:,} samples)...")
        sample_data = {field: [] for field in self.stats_fields}
        total_collected = 0
        
        for file_path in (tqdm(valid_files, desc="Collecting data") if progress_bar else valid_files):
            if total_collected >= max_samples:
                break
                
            with h5py.File(file_path, 'r') as f:
                available_fields = [field for field in self.stats_fields if field in f]
                if not available_fields:
                    continue
                
                file_samples = f[available_fields[0]].shape[0]
                samples_needed = max_samples - total_collected
                samples_to_take = min(file_samples, samples_needed)
                
                if samples_to_take < file_samples:
                    indices = np.linspace(0, file_samples - 1, samples_to_take, dtype=int)
                else:
                    indices = np.arange(file_samples)
                
                for field in available_fields:
                    data = f[field][indices]
                    if self.trim_minutes is not None:
                        if field in ['fhr', 'up']:
                            start_trim = self.trim_samples_raw
                            end_trim = -self.trim_samples_raw if self.trim_samples_raw > 0 else None
                            data = data[:, start_trim:end_trim]
                        else:
                            start_trim = self.trim_samples_decimated
                            end_trim = -self.trim_samples_decimated if self.trim_samples_decimated > 0 else None
                            data = data[:, :, start_trim:end_trim]
                    sample_data[field].append(data)
                
                total_collected += samples_to_take
        
        for field in self.stats_fields:
            if sample_data[field]:
                sample_data[field] = np.concatenate(sample_data[field], axis=0)
            else:
                sample_data[field] = None
        
        available_fields = [field for field in self.stats_fields if sample_data[field] is not None]
        if not available_fields:
            print("No data available for plotting")
            return
        
        save_directory = save_path
        os.makedirs(save_directory, exist_ok=True)
        
        for field in available_fields:
            data = sample_data[field]
            
            # Normalize the data for the second set of plots
            normalized_data = None
            if field in stats:
                try:
                    # Convert to tensor, normalize, then convert back to numpy
                    data_tensor = torch.from_numpy(data.astype(np.float32)).to(self.device)
                    
                    # For multi-channel data, ensure tensor has correct channel dimension first
                    if data_tensor.dim() == 3: # (batch, channels, seq_len)
                        pass
                    elif data_tensor.dim() == 2 and field in ['fhr', 'up']: # (batch, seq_len)
                        pass # No change needed
                    else:
                        warnings.warn(f"Unexpected tensor shape for {field}: {data_tensor.shape}")

                    # Create device-compatible stats - ensure tensors are created on the correct device
                    device_compatible_stats = {}
                    for stat_field, stat_data in stats.items():
                        device_compatible_stats[stat_field] = {}
                        for key, value in stat_data.items():
                            device_compatible_stats[stat_field][key] = value
                        
                        # Pre-create mean_tensor and std_tensor on the correct device
                        if stat_field in ['fhr', 'up']:
                            device_compatible_stats[stat_field]['mean_tensor'] = torch.tensor(
                                stat_data['mean'], dtype=torch.float32, device=self.device
                            )
                            device_compatible_stats[stat_field]['std_tensor'] = torch.tensor(
                                np.sqrt(stat_data['variance']), dtype=torch.float32, device=self.device
                            )
                        else:
                            mean_array = np.array(stat_data['mean'], dtype=np.float32)
                            std_array = np.array(np.sqrt(stat_data['variance']), dtype=np.float32)
                            device_compatible_stats[stat_field]['mean_tensor'] = torch.from_numpy(mean_array).to(
                                dtype=torch.float32, device=self.device
                            ).unsqueeze(-1)
                            device_compatible_stats[stat_field]['std_tensor'] = torch.from_numpy(std_array).to(
                                dtype=torch.float32, device=self.device
                            ).unsqueeze(-1)
                    
                    normalized_tensor = normalize_tensor_data(
                        data=data_tensor,
                        field_name=field,
                        normalization_stats=device_compatible_stats,
                        log_norm_channels_config=self.log_norm_channels_config,
                        asinh_norm_channels_config=self.asinh_norm_channels_config,
                        log_epsilon=1e-6,
                        dtype=torch.float32
                    )
                    normalized_data = normalized_tensor.cpu().numpy()
                except Exception as e:
                    warnings.warn(f"Could not normalize data for field {field}: {e}")

            if field in ['fhr', 'up']:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Raw data plot
                ax_raw = axes[0]
                valid_data = data.flatten()
                valid_data = valid_data[np.isfinite(valid_data)]
                
                if len(valid_data) > 0:
                    p5, p95 = np.percentile(valid_data, [5, 95])
                    clipped_data = valid_data[(valid_data >= p5) & (valid_data <= p95)]
                    if len(clipped_data) > 0:
                        sns.histplot(clipped_data, bins=bins, kde=False, ax=ax_raw)
                        ax_raw.set_title(f'{field.upper()} Raw (5-95th Percentile)')
                        ax_raw.set_xlabel('Raw Value')
                        ax_raw.set_ylabel('Count')
                    else:
                        ax_raw.text(0.5, 0.5, 'No data in 5-95th percentile', ha='center', va='center', transform=ax_raw.transAxes)
                else:
                    ax_raw.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax_raw.transAxes)
                
                # Normalized data plot
                ax_norm = axes[1]
                if normalized_data is not None:
                    valid_norm_data = normalized_data.flatten()
                    valid_norm_data = valid_norm_data[np.isfinite(valid_norm_data)]
                    
                    if len(valid_norm_data) > 0:
                        p5, p95 = np.percentile(valid_norm_data, [5, 95])
                        clipped_norm_data = valid_norm_data[(valid_norm_data >= p5) & (valid_norm_data <= p95)]
                        if len(clipped_norm_data) > 0:
                            sns.histplot(clipped_norm_data, bins=bins, kde=False, ax=ax_norm)
                            ax_norm.set_title(f'{field.upper()} Normalized (5-95th Percentile)')
                            ax_norm.set_xlabel('Normalized Value')
                        else:
                            ax_norm.text(0.5, 0.5, 'No data in 5-95th percentile', ha='center', va='center', transform=ax_norm.transAxes)
                    else:
                        ax_norm.text(0.5, 0.5, 'No valid normalized data', ha='center', va='center', transform=ax_norm.transAxes)
                else:
                    ax_norm.text(0.5, 0.5, 'Normalization failed', ha='center', va='center', transform=ax_norm.transAxes)

                plt.tight_layout()
                field_save_path = os.path.join(save_directory, f'histogram_{field}.png')
                plt.savefig(field_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {field} histogram to: {field_save_path}")
                
            else:
                n_channels = data.shape[1]
                if max_channels is None or max_channels <= 0:
                    channels_to_plot = n_channels
                else:
                    channels_to_plot = min(n_channels, max_channels)
                
                fig, axes = plt.subplots(channels_to_plot, 2, figsize=(14, 5 * channels_to_plot), squeeze=False)
                
                for channel in range(channels_to_plot):
                    # Raw data plot
                    ax_raw = axes[channel, 0]
                    valid_data = data[:, channel, :].flatten()
                    valid_data = valid_data[np.isfinite(valid_data)]
                    
                    if len(valid_data) > 0:
                        p5, p95 = np.percentile(valid_data, [5, 95])
                        clipped_data = valid_data[(valid_data >= p5) & (valid_data <= p95)]
                        if len(clipped_data) > 0:
                            sns.histplot(clipped_data, bins=bins, kde=False, ax=ax_raw)
                            ax_raw.set_title(f'{field.upper()} Ch {channel} Raw (5-95th %ile)')
                            ax_raw.set_ylabel('Count')
                        else:
                            ax_raw.text(0.5, 0.5, 'No data in 5-95th percentile', ha='center', va='center', transform=ax_raw.transAxes)
                    else:
                        ax_raw.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax_raw.transAxes)

                    # Normalized data plot
                    ax_norm = axes[channel, 1]
                    if normalized_data is not None:
                        valid_norm_data = normalized_data[:, channel, :].flatten()
                        valid_norm_data = valid_norm_data[np.isfinite(valid_norm_data)]
                        
                        if len(valid_norm_data) > 0:
                            p5, p95 = np.percentile(valid_norm_data, [5, 95])
                            clipped_norm_data = valid_norm_data[(valid_norm_data >= p5) & (valid_norm_data <= p95)]
                            if len(clipped_norm_data) > 0:
                                sns.histplot(clipped_norm_data, bins=bins, kde=False, ax=ax_norm)
                                ax_norm.set_title(f'{field.upper()} Ch {channel} Normalized (5-95th %ile)')
                            else:
                                ax_norm.text(0.5, 0.5, 'No data in 5-95th percentile', ha='center', va='center', transform=ax_norm.transAxes)
                        else:
                             ax_norm.text(0.5, 0.5, 'No valid normalized data', ha='center', va='center', transform=ax_norm.transAxes)
                    else:
                        ax_norm.text(0.5, 0.5, 'Normalization failed', ha='center', va='center', transform=ax_norm.transAxes)

                plt.tight_layout()
                field_save_path = os.path.join(save_directory, f'histogram_{field}.png')
                plt.savefig(field_save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {field} histogram ({channels_to_plot} channels) to: {field_save_path}")
        
        print(f"\nHistogram plotting completed!")
        print(f"Created separate histogram plots for: {', '.join(available_fields)}")
        print(f"All plots show raw and normalized distributions for the 5-95th percentile of data with {bins} bins.")
        for field in available_fields:
            if field not in ['fhr', 'up']:
                data_shape = sample_data[field].shape
                n_channels = data_shape[1] if len(data_shape) > 1 else 1
                
                if max_channels is None or max_channels <= 0:
                    channels_plotted = n_channels
                else:
                    channels_plotted = min(n_channels, max_channels)
                print(f"  {field}: {channels_plotted}/{n_channels} channels plotted in vertical layout")


def plot_dataset_histograms(
    hdf5_files: List[str],
    save_path: str,
    max_channels: Optional[int] = 5,
    max_samples: int = 10000,
    figsize: Tuple[int, int] = (15, 10),
    bins: int = 50,
    progress_bar: bool = True,
    trim_minutes: Optional[float] = None,
    device: Optional[str] = None
) -> None:
    """
    Convenience function to plot histograms of the 5-95th percentile of raw and normalized values.
    This removes 10% of outliers from the view to provide a clearer histogram.
    
    Args:
        hdf5_files: List of paths to HDF5 dataset files
        save_path: Directory where the output figures will be saved.
        max_channels: Maximum number of channels to plot for multi-channel data (0:max_channels).
                      If None or <= 0, all channels will be plotted.
        max_samples: Maximum number of samples to use for plotting (for memory efficiency)
        bins: The number of bins to use for the histogram.
        progress_bar: Whether to show progress bars
        trim_minutes: Optional trimming time in minutes
        device: The device to use for calculation (e.g., 'cpu', 'cuda:0'). Autodetects if None.
        
    Example:
        >>> hdf5_files = ['dataset1.h5', 'dataset2.h5']
        >>> plot_dataset_histograms(hdf5_files, './histogram_plots', max_channels=None)
        # Creates files like: ./histogram_plots/histogram_fhr.png, ./histogram_plots/histogram_up.png, etc.
    """
    calculator = DatasetStatsCalculator(trim_minutes=trim_minutes, device=device)
    calculator.plot_histograms(hdf5_files, save_path, max_channels, max_samples, figsize, bins, progress_bar)


def calculate_and_save_dataset_stats(
    hdf5_files: List[str],
    output_path: str,
    batch_size: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
    progress_bar: bool = True,
    trim_minutes: Optional[float] = None,
    device: Optional[str] = None,
    plot_histograms: bool = True,
    histograms_dir: Optional[str] = None,
    max_histogram_samples: int = 50000
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to calculate and save dataset statistics and plot histograms.
    
    Args:
        hdf5_files: List of paths to HDF5 dataset files
        output_path: Path for output statistics HDF5 file
        batch_size: Batch size for processing (affects memory usage)
        metadata: Optional metadata to include in output file
        progress_bar: Whether to show progress bars
        trim_minutes: Optional trimming time in minutes
        device: The device to use for calculation (e.g., 'cpu', 'cuda:0'). Autodetects if None.
        plot_histograms: Whether to generate histogram plots
        histograms_dir: Directory to save histogram plots (defaults to same dir as output_path)
        max_histogram_samples: Maximum number of samples to use for histogram plotting
        
    Returns:
        Dictionary containing calculated statistics
        
    Example:
        >>> hdf5_files = ['dataset1.h5', 'dataset2.h5']
        >>> stats = calculate_and_save_dataset_stats(
        ...     hdf5_files, 
        ...     'dataset_stats.h5',
        ...     metadata={'description': 'Training dataset statistics'},
        ...     plot_histograms=True
        ... )
    """
    calculator = DatasetStatsCalculator(trim_minutes=trim_minutes, device=device)
    
    # Calculate statistics
    print("Calculating dataset statistics...")
    stats = calculator.calculate_stats(hdf5_files, batch_size, progress_bar)
    
    # Print summary
    calculator.print_stats_summary(stats)
    
    # Save to file
    if metadata is None:
        metadata = {}
    metadata['trim_minutes'] = trim_minutes
    calculator.save_stats(stats, output_path, metadata)
    
    # Plot histograms if requested
    if plot_histograms:
        if histograms_dir is None:
            # Default to same directory as output file
            histograms_dir = os.path.dirname(output_path)
            if not histograms_dir:
                histograms_dir = '.'
            histograms_dir = os.path.join(histograms_dir, 'histograms')
        
        print(f"\n--- Generating Histograms (all channels) ---")
        plot_dataset_histograms(
            hdf5_files=hdf5_files,
            save_path=histograms_dir,
            max_channels=None,  # Plot all channels
            max_samples=max_histogram_samples,
            bins=50,
            trim_minutes=trim_minutes,
            device=device
        )
    
    return stats


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_files = [r"C:\Users\mahdi\Desktop\teb_vae_model\hdf5_dataset\train_dataset_cs.hdf5"]
    output_file = r'C:\Users\mahdi\Desktop\teb_vae_model\output\stats.hdf5'
    
    # Calculate and save stats with histogram plotting enabled
    stats = calculate_and_save_dataset_stats(
        input_files,
        output_file,
        metadata={
            'input_files': input_files,
            'num_files': len(input_files),
            'description': 'Statistics for optimal coefficient selection (J=11, Q=4, T=16): 217 total features'
        },
        trim_minutes=2,
        device=device,
        plot_histograms=True,  # Enable histogram plotting
        max_histogram_samples=50000
    )
    
    # Verify saved stats by reloading
    calculator = DatasetStatsCalculator(device=device)
    loaded_stats = calculator.load_stats(output_file)
    print("\n--- Verifying saved stats by reloading and printing summary ---")
    calculator.print_stats_summary(loaded_stats)

    print("\nScript finished successfully.")
    