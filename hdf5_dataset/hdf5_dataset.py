import os
import h5py
import numpy as np
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pickle
import atexit
import threading
import warnings
from functools import lru_cache
import gc
from torch.utils.data.dataloader import default_collate
import yaml


def normalize_tensor_data(
    data: torch.Tensor,
    field_name: str,
    normalization_stats: Dict[str, Dict[str, Any]],
    log_norm_channels_config: Dict[str, Any],
    asinh_norm_channels_config: Dict[str, Any],
    log_epsilon: float,
    pin_memory: bool = False,
    normalize_fields: Optional[set] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Normalizes a tensor using precomputed statistics from a stats dictionary.

    This function can be used independently of the CombinedHDF5Dataset class.
    It prepares the statistics (mean/std tensors) and then applies normalization.

    Args:
        data: The data tensor to normalize.
        field_name: The name of the field (e.g., 'fhr_st').
        normalization_stats: A dictionary of statistics, like one from DatasetStatsCalculator.
        log_norm_channels_config: A dictionary defining which channels use log normalization.
        asinh_norm_channels_config: A dictionary defining which channels use asinh normalization.
        log_epsilon: A small value to add before taking the logarithm.
        pin_memory: If True, pin the created mean/std tensors.
        normalize_fields: An optional set of fields to normalize. If None, all are attempted.
        dtype: The torch dtype for the tensors.

    Returns:
        The normalized data tensor.
    """
    if field_name not in normalization_stats:
        return data

    if normalize_fields is not None and field_name not in normalize_fields:
        return data

    stats = normalization_stats[field_name]

    # Prepare mean and std tensors from numpy arrays in stats
    if 'mean_tensor' not in stats or 'std_tensor' not in stats:
        if field_name in ['fhr', 'up']:
            mean_tensor = torch.tensor(stats['mean'], dtype=dtype)
            std_tensor = torch.tensor(np.sqrt(stats['variance']), dtype=dtype)
        else:
            mean_array = np.array(stats['mean'], dtype=np.float32)
            std_array = np.array(np.sqrt(stats['variance']), dtype=np.float32)
            mean_tensor = torch.from_numpy(mean_array).to(dtype).unsqueeze(-1)
            std_tensor = torch.from_numpy(std_array).to(dtype).unsqueeze(-1)
    else:
        mean_tensor = stats['mean_tensor']
        std_tensor = stats['std_tensor']

    if pin_memory and data.is_pinned():
        mean_tensor = mean_tensor.pin_memory()
        std_tensor = std_tensor.pin_memory()

    is_batch = data.dim() == 3

    # Apply transformation based on field type
    if field_name in ['fhr', 'up']:
        epsilon = 1e-8
        normalized_data = (data - mean_tensor) / (std_tensor + epsilon)
    else:
        # Multi-channel scattering data: apply log or standard normalization per channel.
        n_channels = data.shape[1] if is_batch else data.shape[0]

        # Determine which channels use log normalization from the config.
        log_config = log_norm_channels_config.get(field_name, [])
        log_channels = []
        if log_config == 'all_except_0':
            log_channels = [c for c in range(n_channels) if c != 0] if n_channels > 0 else []
        elif isinstance(log_config, list):
            log_channels = log_config

        # Determine which channels use asinh normalization from the config.
        asinh_config = asinh_norm_channels_config.get(field_name, [])
        asinh_channels = []
        if asinh_config == 'all':
            asinh_channels = [c for c in range(n_channels)]
        elif isinstance(asinh_config, list):
            asinh_channels = asinh_config

        # Start with a clone of the data; we will transform it in-place.
        data_transformed = data.clone()

        # Apply log transform to the specified channels.
        if log_channels:
            log_channels_tensor = torch.tensor(log_channels, device=data.device, dtype=torch.long)
            if is_batch:
                selected_data = data_transformed[:, log_channels_tensor, :]
                log_transformed_data = torch.log(torch.clamp(selected_data, min=0.0) + log_epsilon)
                data_transformed[:, log_channels_tensor, :] = log_transformed_data
            else:
                selected_data = data_transformed[log_channels_tensor, :]
                log_transformed_data = torch.log(torch.clamp(selected_data, min=0.0) + log_epsilon)
                data_transformed[log_channels_tensor, :] = log_transformed_data

        # Apply asinh transform to the specified channels.
        if asinh_channels:
            asinh_channels_tensor = torch.tensor(asinh_channels, device=data.device, dtype=torch.long)
            if is_batch:
                selected_data = data_transformed[:, asinh_channels_tensor, :]
                asinh_transformed_data = torch.asinh(selected_data)
                data_transformed[:, asinh_channels_tensor, :] = asinh_transformed_data
            else:
                selected_data = data_transformed[asinh_channels_tensor, :]
                asinh_transformed_data = torch.asinh(selected_data)
                data_transformed[asinh_channels_tensor, :] = asinh_transformed_data
        
        # Reshape stats tensors for broadcasting across all dimensions.
        if is_batch:
            mean_tensor = mean_tensor.view(1, -1, 1)
            std_tensor = std_tensor.view(1, -1, 1)
        
        # Apply standard normalization to the (potentially log-transformed) data.
        epsilon = 1e-8
        normalized_data = (data_transformed - mean_tensor) / (std_tensor + epsilon)

    return normalized_data


def create_initial_hdf5(
    path: str,
    len_signal: int,
    n_channels: int,
    len_sequence: int = 300
) -> None:
    """
    Create a new HDF5 file with empty, resizable datasets for signal storage.
    
    Updated for optimal coefficient selection (J=11, Q=4, T=16):
    - FHR scattering: 43 coefficients (first order only)
    - FHR phase: 44 coefficients (95.1% reduction from optimal selection)
    - FHR-UP cross-phase: 130 coefficients (UP→FHR coupling)
    - Total phase/cross-phase channels: 174

    Datasets created (first dim unlimited):
      - "fhr"       : float32, shape (N, len_signal)
      - "up"        : float32, shape (N, len_signal)
      - "fhr_st"    : float32, shape (N, 43, len_sequence) - Scattering coefficients
      - "fhr_ph"    : float32, shape (N, 44, len_sequence) - Selected phase coefficients
      - "fhr_up_ph" : float32, shape (N, 130, len_sequence) - Selected cross-phase coefficients
      - "target"    : float32, shape (N, len_sequence)
      - "weight"    : float32, shape (N, len_sequence)
      - "epoch"     : float32, shape (N,)
      - "cs_label"  : uint8 (0 or 1), shape (N,)
      - "bg_label"  : uint8 (0 or 1), shape (N,)
      - "guid"      : variable-length UTF-8 strings, shape (N,)

    All datasets use per-sample chunking and LZF compression.

    Args:
        path:         Path to output HDF5 file (overwrites if exists).
        len_signal:   Length of raw signal arrays (e.g. 4800).
        n_channels:   Number of channels in combined phase/cross-phase arrays (174).
        len_sequence: Length of sequence dimension (default: 300).
    """
    try:
        os.remove(path)
    except OSError:
        pass

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w", libver="latest") as h5f:
        h5f.create_dataset(
            "fhr", shape=(0, len_signal), maxshape=(None, len_signal),
            dtype="f4", chunks=(1, len_signal), compression="lzf"
        )
        h5f.create_dataset(
            "up", shape=(0, len_signal), maxshape=(None, len_signal),
            dtype="f4", chunks=(1, len_signal), compression="lzf"
        )
        # Create datasets with optimal channel counts
        # fhr_st: 43 scattering coefficients (first order)
        h5f.create_dataset(
            "fhr_st", shape=(0, 43, len_sequence), maxshape=(None, 43, len_sequence),
            dtype="f4", chunks=(1, 43, len_sequence), compression="lzf"
        )
        # fhr_ph: 44 selected phase coefficients  
        h5f.create_dataset(
            "fhr_ph", shape=(0, 44, len_sequence), maxshape=(None, 44, len_sequence),
            dtype="f4", chunks=(1, 44, len_sequence), compression="lzf"
        )
        # fhr_up_ph: 130 selected cross-phase coefficients
        h5f.create_dataset(
            "fhr_up_ph", shape=(0, 130, len_sequence), maxshape=(None, 130, len_sequence),
            dtype="f4", chunks=(1, 130, len_sequence), compression="lzf"
        )
        h5f.create_dataset(
            "target", shape=(0, len_sequence), maxshape=(None, len_sequence),
            dtype="f4", chunks=(1, len_sequence), compression="lzf"
        )
        h5f.create_dataset(
            "weight", shape=(0, len_sequence), maxshape=(None, len_sequence),
            dtype="f4", chunks=(1, len_sequence), compression="lzf"
        )
        h5f.create_dataset(
            "epoch", shape=(0,), maxshape=(None,),
            dtype="f4", chunks=(1,), compression="lzf"
        )
        h5f.create_dataset(
            "cs_label", shape=(0,), maxshape=(None,),
            dtype="u1", chunks=(1,), compression="lzf"
        )
        h5f.create_dataset(
            "bg_label", shape=(0,), maxshape=(None,),
            dtype="u1", chunks=(1,), compression="lzf"
        )
        h5f.create_dataset(
            "guid", shape=(0,), maxshape=(None,),
            dtype=str_dt, chunks=(1,)
        )


def append_sample(
    path: str,
    fhr: np.ndarray,
    up: np.ndarray,
    fhr_st: np.ndarray,
    fhr_ph: np.ndarray,
    fhr_up_ph: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    guid: str,
    epoch: float,
    cs_label: bool,
    bg_label: bool
) -> None:
    """
    Append a single sample to an existing HDF5 dataset.

    Datasets are resized by +1 along axis=0, and new values written in-place.

    Args:
        path:      Path to existing HDF5 file.
        fhr:       Raw FHR array, shape (len_signal,).
        up:        Raw UP array, shape (len_signal,).
        fhr_st:    Scattering array, shape (n_channels, len_sequence).
        fhr_ph:    Phase array, shape (n_channels, len_sequence).
        fhr_up_ph: Cross-phase array, shape (n_channels, len_sequence).
        target:    Target array, shape (len_sequence,).
        weight:    Weight array, shape (len_sequence,).
        guid:      Unique identifier string.
        epoch:     Epoch as float.
        cs_label:  Case label flag.
        bg_label:  Background label flag.
    """
    with h5py.File(path, "a", libver="latest") as h5f:
        idx = h5f["fhr"].shape[0]
        new_size = idx + 1
        for name, ds in h5f.items():
            ds.resize((new_size,) + ds.shape[1:])
        h5f["fhr"][idx]       = fhr
        h5f["up"][idx]        = up
        h5f["fhr_st"][idx]    = fhr_st
        h5f["fhr_ph"][idx]    = fhr_ph
        h5f["fhr_up_ph"][idx] = fhr_up_ph
        h5f["target"][idx]    = target
        h5f["weight"][idx]    = weight
        h5f["epoch"][idx]     = epoch
        h5f["cs_label"][idx]  = np.uint8(cs_label)
        h5f["bg_label"][idx]  = np.uint8(bg_label)
        h5f["guid"][idx]      = guid


class AttributeDict(dict):
    """A dictionary that allows attribute-style access."""
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'AttributeDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class CombinedHDF5Dataset(Dataset):
    """
    High-performance PyTorch Dataset for one or more HDF5 files with identical structure.
    
    Updated for optimal coefficient selection (J=11, Q=4, T=16):
    - FHR scattering: 43 coefficients (first order)
    - FHR phase: 44 selected coefficients (95.1% reduction)
    - FHR-UP cross-phase: 130 selected coefficients (UP→FHR coupling)
    
    Optimized for:
    - Multi-GPU training with DistributedDataParallel
    - Multi-worker data loading
    - Memory efficiency and fast I/O
    - Advanced filtering and selective loading
    - Data normalization using precomputed statistics with optimal coefficient selection
    
    Args:
        paths: Path(s) to HDF5 file(s).
        load_fields: Specific fields to load (None loads all). Target and weight are always included.
        allowed_guids: Only samples with these GUIDs are included.
        cs_label: Filter by cs_label value (True/False/None for no filtering).
        bg_label: Filter by bg_label value (True/False/None for no filtering).
        epoch_min: Minimum epoch value (inclusive).
        epoch_max: Maximum epoch value (inclusive).
        label: Target class to filter by. Only samples where the one-hot encoded target
               has this class as 1 in at least one valid timestep are included.
        cache_size: Number of samples to keep in memory cache (0 disables caching).
        pin_memory: Pre-allocate tensors in pinned memory for faster GPU transfer.
        dtype: Data type for tensors (torch.float32 or torch.float16 for mixed precision).
        stats_path: Path to HDF5 statistics file for data normalization (None disables normalization).
        normalize_fields: List of fields to normalize (None normalizes all available fields with stats).
        trim_minutes: Optional trimming time in minutes for signal data
    """
    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        load_fields: Optional[Sequence[str]] = None,
        allowed_guids: Optional[Sequence[str]] = None,
        cs_label: Optional[bool] = None,
        bg_label: Optional[bool] = None,
        epoch_min: Optional[float] = None,
        epoch_max: Optional[float] = None,
        label: Optional[int] = None,
        cache_size: int = 2000,
        pin_memory: bool = True,
        dtype: torch.dtype = torch.float32,
        stats_path: Optional[str] = None,
        normalize_fields: Optional[Sequence[str]] = None,
        trim_minutes: Optional[float] = None
    ):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.load_fields = None if load_fields is None else set(load_fields)
        self.allowed_guids = set(allowed_guids) if allowed_guids is not None else None
        self.cs_label = cs_label
        self.bg_label = bg_label
        self.epoch_min = epoch_min
        self.epoch_max = epoch_max
        self.label = label
        self.cache_size = cache_size
        self.pin_memory = pin_memory
        self.dtype = dtype
        self.stats_path = stats_path
        self.normalize_fields = set(normalize_fields) if normalize_fields is not None else None
        self.trim_minutes = trim_minutes
        if self.trim_minutes is not None:
            self.trim_samples_raw = int(4 * 60 * self.trim_minutes)
            self.trim_samples_decimated = self.trim_samples_raw // 16
        else:
            self.trim_samples_raw = 0
            self.trim_samples_decimated = 0
        
        # Thread-safe file handle management
        self.file_handles: List[Any] = [None] * len(self.paths)
        self._handle_locks = [threading.Lock() for _ in self.paths]
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, sample_idx)
        
        # Performance optimizations
        self._cache: Dict[int, "AttributeDict"] = {}
        self._cache_lock = threading.Lock()
        self._access_count = 0
        
        # Normalization statistics
        self.normalization_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self.normalization_enabled = False
        
        # Define which channels should use LOG normalization for optimal coefficients.
        # Updated for optimal coefficient selection (44 phase + 130 cross-phase channels).
        self.log_norm_channels_config = {
            'fhr_st': 'all_except_0',  # 42 of 43 scattering coefficients (exclude order 0)
        }
        self.asinh_norm_channels_config = {
            'fhr_ph': 'all',     # All 44 selected phase coefficients
            'fhr_up_ph': 'all'   # All 130 selected cross-phase coefficients
        }
        
        # This will be populated from the stats file, but the config above provides a fallback.
        self.order0_channels: Dict[str, List[int]] = {}
        self.log_epsilon = 1e-6  # For log transformation
        
        # Register cleanup for proper file handle management
        atexit.register(self._cleanup_handles)
        
        # Load normalization statistics if provided
        if self.stats_path is not None:
            self._load_normalization_stats()
        
        # Build index with optimized filtering
        self._build_index()
        
        # Validate dataset
        if not self.index_map:
            raise ValueError("No samples match the specified filters.")
        
        print(f"Initialized HDF5Dataset: {len(self.index_map)} samples from {len(self.paths)} files")
        if self.cache_size > 0:
            print(f"Caching enabled: {min(self.cache_size, len(self.index_map))} samples")
        if self.normalization_enabled:
            normalized_fields = list(self.normalization_stats.keys())
            print(f"Normalization enabled for fields: {normalized_fields}")

    def _load_normalization_stats(self):
        """
        Load normalization statistics from HDF5 file.
        
        The stats file should be created by DatasetStatsCalculator.save_stats().
        """
        if not os.path.exists(self.stats_path):
            warnings.warn(f"Statistics file not found: {self.stats_path}. Normalization disabled.")
            return
        
        try:
            stats = {}
            with h5py.File(self.stats_path, 'r') as f:
                # Load global metadata
                self.log_epsilon = f.attrs.get('log_epsilon', 1e-6)
                stats_trim_minutes = f.attrs.get('trim_minutes', -1.0)
                if self.trim_minutes is not None and stats_trim_minutes != self.trim_minutes:
                    warnings.warn(f"Dataset trim_minutes ({self.trim_minutes}) does not match stats file trim_minutes ({stats_trim_minutes}). This may lead to incorrect normalization.")
                elif self.trim_minutes is None and stats_trim_minutes > 0:
                     warnings.warn(f"Stats file was created with trim_minutes={stats_trim_minutes}, but dataset is not using trimming. Normalization might be incorrect.")

                for field in f.keys():
                    if field == 'metadata':
                        continue
                    
                    field_group = f[field]
                    field_stats = {
                        'shape': tuple(field_group.attrs['shape']),
                        'count': field_group.attrs['count']
                    }
                    
                    if field in ['fhr', 'up']:
                        # Single-channel data - scalar values
                        field_stats['mean'] = field_group.attrs['mean_scalar']
                        field_stats['std'] = field_group.attrs['std_scalar']
                        
                        # Convert to tensors for efficient computation
                        field_stats['mean_tensor'] = torch.tensor(
                            field_stats['mean'], dtype=self.dtype
                        )
                        field_stats['std_tensor'] = torch.tensor(
                            field_stats['std'], dtype=self.dtype
                        )
                        
                    else:
                        # Multi-channel data - per-channel arrays
                        field_stats['mean'] = field_group['mean'][()]
                        field_stats['std'] = field_group['std'][()]
                        
                        # Load transformation metadata if available
                        if 'regular_channels' in field_group.attrs:
                            field_stats['uses_log_transform'] = field_group.attrs.get('uses_log_transform', False)
                            field_stats['uses_asinh_transform'] = field_group.attrs.get('uses_asinh_transform', False)
                            field_stats['regular_channels'] = list(field_group.attrs.get('regular_channels', []))
                            field_stats['log_channels'] = list(field_group.attrs.get('log_channels', []))
                            field_stats['asinh_channels'] = list(field_group.attrs.get('asinh_channels', []))
                        # Backward compatibility for 'order0_channels' from old stats files
                        elif 'order0_channels' in field_group.attrs:
                            order0_channels = list(field_group.attrs.get('order0_channels', []))
                            n_channels = len(field_stats['mean'])
                            log_channels = [i for i in range(n_channels) if i not in order0_channels]
                            field_stats['uses_log_transform'] = True
                            field_stats['uses_asinh_transform'] = False
                            field_stats['regular_channels'] = order0_channels
                            field_stats['log_channels'] = log_channels
                            field_stats['asinh_channels'] = []
                        else:
                            field_stats['uses_log_transform'] = False
                            field_stats['uses_asinh_transform'] = False
                            field_stats['regular_channels'] = []
                            field_stats['log_channels'] = []
                            field_stats['asinh_channels'] = []
                        
                        # Convert to tensors with proper shape for broadcasting
                        # Shape will be (n_channels, 1) for broadcasting over sequence dimension
                        mean_array = np.array(field_stats['mean'], dtype=np.float32)
                        std_array = np.array(field_stats['std'], dtype=np.float32)
                        
                        field_stats['mean_tensor'] = torch.from_numpy(mean_array).to(self.dtype).unsqueeze(-1)
                        field_stats['std_tensor'] = torch.from_numpy(std_array).to(self.dtype).unsqueeze(-1)
                    
                    stats[field] = field_stats
            
            self.normalization_stats = stats
            self.normalization_enabled = True
            
            # Overwrite the default transformation configs with what was loaded from the stats file.
            log_config = {}
            asinh_config = {}
            if self.normalization_stats:
                for field, stats_dict in self.normalization_stats.items():
                    if stats_dict.get('uses_log_transform') and 'log_channels' in stats_dict:
                        log_config[field] = stats_dict['log_channels']
                    if stats_dict.get('uses_asinh_transform') and 'asinh_channels' in stats_dict:
                        asinh_config[field] = stats_dict['asinh_channels']
            
            self.log_norm_channels_config = log_config
            self.asinh_norm_channels_config = asinh_config

            # Report the actual transformations that will be used.
            log_transformed_fields = list(self.log_norm_channels_config.keys())
            asinh_transformed_fields = list(self.asinh_norm_channels_config.keys())

            if log_transformed_fields:
                print(f"Log transformation enabled for fields: {log_transformed_fields}")
                for field in log_transformed_fields:
                    log_channels = self.log_norm_channels_config.get(field, [])
                    # Infer regular channels from the full channel list
                    try:
                        # shape is (channels, sequence_len) for these fields
                        n_channels = self.normalization_stats[field]['shape'][0]
                        regular_channels = [c for c in range(n_channels) if c not in log_channels]
                        if regular_channels or log_channels:
                            print(f"  {field}: regular channels {regular_channels}, log channels {log_channels}")
                    except (KeyError, IndexError):
                         print(f"  {field}: log channels {log_channels}")

            if asinh_transformed_fields:
                print(f"Asinh transformation enabled for fields: {asinh_transformed_fields}")
                for field in asinh_transformed_fields:
                    asinh_channels = self.asinh_norm_channels_config.get(field, [])
                    if asinh_channels:
                        print(f"  {field}: asinh channels {asinh_channels}")
            
        except Exception as e:
            warnings.warn(f"Failed to load statistics from {self.stats_path}: {e}. Normalization disabled.")
            self.normalization_stats = None
            self.normalization_enabled = False

    def _normalize_data(self, field_name: str, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data using precomputed statistics.
        Applies log transformation to scattering coefficients (except order 0).
        
        Args:
            field_name: Name of the field being normalized
            data: Data tensor to normalize
            
        Returns:
            Normalized data tensor
        """
        if not self.normalization_enabled or field_name not in self.normalization_stats:
            return data
        
        # Check if this field should be normalized
        if self.normalize_fields is not None and field_name not in self.normalize_fields:
            return data
        
        return normalize_tensor_data(
            data=data,
            field_name=field_name,
            normalization_stats=self.normalization_stats,
            log_norm_channels_config=self.log_norm_channels_config,
            asinh_norm_channels_config=self.asinh_norm_channels_config,
            log_epsilon=self.log_epsilon,
            pin_memory=self.pin_memory,
            normalize_fields=self.normalize_fields,
            dtype=self.dtype
        )

    def get_normalization_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get the loaded normalization statistics.
        
        Returns:
            Dictionary containing normalization statistics for each field, or None if not loaded
        """
        return self.normalization_stats

    def is_normalization_enabled(self) -> bool:
        """
        Check if normalization is enabled.
        
        Returns:
            True if normalization is enabled, False otherwise
        """
        return self.normalization_enabled

    def _build_index(self):
        """Build sample index with optimized filtering."""
        for fidx, path in enumerate(self.paths):
            if not os.path.exists(path):
                warnings.warn(f"HDF5 file not found: {path}")
                continue
                
            try:
                with h5py.File(path, 'r', libver='latest') as f:
                    # Load all metadata at once for efficiency
                    guids = f['guid'][()]
                    epochs = f['epoch'][()]
                    cs_lbl = f['cs_label'][()]
                    bg_lbl = f['bg_label'][()]
                    n_samples = len(guids)
                    
                    # Vectorized filtering where possible
                    valid_mask = np.ones(n_samples, dtype=bool)
                    
                    # Apply epoch filtering
                    if self.epoch_min is not None:
                        valid_mask &= (epochs >= self.epoch_min)
                    if self.epoch_max is not None:
                        valid_mask &= (epochs <= self.epoch_max)
                    
                    # Apply label filtering
                    if self.cs_label is not None:
                        valid_mask &= (cs_lbl == self.cs_label)
                    if self.bg_label is not None:
                        valid_mask &= (bg_lbl == self.bg_label)
                    
                    # Process remaining samples
                    for i in np.where(valid_mask)[0]:
                        # GUID filtering
                        guid = guids[i].decode('utf-8') if isinstance(guids[i], bytes) else str(guids[i])
                        if self.allowed_guids and guid not in self.allowed_guids:
                            continue
                        
                        # Target label filtering
                        if self.label is not None:
                            target_data = f['target'][i]  # shape: (len_sequence,)
                            # Check if any timestep has the target label value
                            has_label = np.any(target_data == self.label)
                            if not has_label:
                                continue
                        
                        self.index_map.append((fidx, i))
                        
            except Exception as e:
                warnings.warn(f"Error processing {path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.index_map)

    def _open_handle(self, file_idx: int):
        """Thread-safe file handle opening with optimizations."""
        with self._handle_locks[file_idx]:
            if self.file_handles[file_idx] is None:
                try:
                    # Optimal HDF5 settings for performance
                    self.file_handles[file_idx] = h5py.File(
                        self.paths[file_idx], 'r',
                        libver='latest',
                        swmr=True,
                        rdcc_nbytes=1024**2 * 128,    # 128MB cache per file
                        rdcc_nslots=10007,            # Prime number for hash table
                        rdcc_w0=0.75,                 # Cache write policy
                        driver='sec2'                 # System call driver for better performance
                    )
                except Exception as e:
                    # Fallback to default settings
                    warnings.warn(f"Using default HDF5 settings for {self.paths[file_idx]}: {e}")
                    self.file_handles[file_idx] = h5py.File(
                        self.paths[file_idx], 'r', libver='latest', swmr=True
                    )
            return self.file_handles[file_idx]
    
    def _cleanup_handles(self):
        """Thread-safe cleanup of file handles."""
        for i, (handle, lock) in enumerate(zip(self.file_handles, self._handle_locks)):
            with lock:
                if handle is not None:
                    try:
                        handle.close()
                        self.file_handles[i] = None
                    except:
                        pass

    @lru_cache(maxsize=128)
    def _get_sample_fields(self, file_idx: int) -> Tuple[str, ...]:
        """Cache available fields for each file."""
        f = self._open_handle(file_idx)
        return tuple(f.keys())

    def _create_tensor(self, data: np.ndarray, pin_memory: bool = None) -> torch.Tensor:
        """Optimized tensor creation with optional memory pinning."""
        if pin_memory is None:
            pin_memory = self.pin_memory
            
        # Convert to tensor with specified dtype
        if data.dtype == np.float32 and self.dtype == torch.float32:
            # Direct conversion without copy for matching dtypes
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.from_numpy(data.astype(np.float32 if self.dtype == torch.float32 else np.float16))
        
        # Pin memory for faster GPU transfer
        if pin_memory and tensor.is_floating_point():
            tensor = tensor.pin_memory()
            
        return tensor.to(dtype=self.dtype)

    def __getitem__(self, idx: int) -> "AttributeDict":
        """Optimized sample loading with caching, memory management, and normalization."""
        # Check cache first
        if self.cache_size > 0:
            with self._cache_lock:
                if idx in self._cache:
                    return self._cache[idx]
        
        file_idx, sample_idx = self.index_map[idx]
        f = self._open_handle(file_idx)
        out: Dict[str, Any] = {}
        
        # Determine fields to load
        available_fields = self._get_sample_fields(file_idx)
        if self.load_fields is None:
            fields = list(available_fields)
        else:
            fields = list(self.load_fields)

        # Load data efficiently
        try:
            for name in fields:
                if name not in available_fields:
                    continue
                    
                data = f[name][sample_idx]

                if self.trim_minutes is not None:
                    if name in ['fhr', 'up']:
                        start_trim = self.trim_samples_raw
                        end_trim = -self.trim_samples_raw if self.trim_samples_raw > 0 else None
                        data = data[start_trim:end_trim]
                    elif name in ['fhr_st', 'fhr_ph', 'fhr_up_ph']:
                        start_trim = self.trim_samples_decimated
                        end_trim = -self.trim_samples_decimated if self.trim_samples_decimated > 0 else None
                        data = data[:, start_trim:end_trim]

                if name in ('guid',):
                    out[name] = data.decode('utf-8') if isinstance(data, bytes) else str(data)
                elif name in ('cs_label', 'bg_label'):
                    out[name] = bool(data)
                else:
                    # Optimized tensor creation
                    tensor = self._create_tensor(np.asarray(data))
                    
                    # Apply normalization if enabled and applicable
                    if (self.normalization_enabled and 
                        name in ['fhr', 'up', 'fhr_st', 'fhr_ph', 'fhr_up_ph']):
                        tensor = self._normalize_data(name, tensor)
                    
                    # SPEED OPTIMIZATION: Apply permutation here once instead of multiple times in training
                    # Convert from HDF5 format (channels, sequence) to model format (sequence, channels)
                    if name in ['fhr_st', 'fhr_ph', 'fhr_up_ph'] and tensor.dim() == 2:
                        tensor = tensor.transpose(0, 1)  # (channels, seq) -> (seq, channels)
                    
                    out[name] = tensor
                    
        except Exception as e:
            warnings.warn(f"Error loading sample {idx} from {self.paths[file_idx]}: {e}")
            raise
        
        sample = AttributeDict(out)

        # Cache management
        if self.cache_size > 0:
            with self._cache_lock:
                if len(self._cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[idx] = sample
        
        self._access_count += 1
        return sample
    
    def get_the_lists(self):
        """
        Retrieves lists of GUIDs, epochs, and targets for all samples in the dataset index.
        Note: This can be slow for large datasets as it iterates through all samples.
        """
        guids, epochs, targets = [], [], []
        
        indices_by_file = {}
        for f_idx, s_idx in self.index_map:
            if f_idx not in indices_by_file:
                indices_by_file[f_idx] = []
            indices_by_file[f_idx].append(s_idx)

        for f_idx, s_indices in indices_by_file.items():
            handle = self._open_handle(f_idx)
            # Sort indices to improve read performance, h5py recommends this
            s_indices.sort()
            
            # Use fancy indexing to read all required samples from this file at once
            guids.extend([g.decode('utf-8') for g in handle['guid'][s_indices]])
            epochs.extend(handle['epoch'][s_indices])
            targets.extend(handle['target'][s_indices])
            
        return guids, epochs, targets

    def clear_cache(self):
        """Clear the sample cache to free memory."""
        with self._cache_lock:
            self._cache.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics for monitoring."""
        return {
            'total_samples': len(self.index_map),
            'num_files': len(self.paths),
            'cache_size': len(self._cache),
            'access_count': self._access_count,
            'dtype': str(self.dtype),
            'pin_memory': self.pin_memory,
            'normalization_enabled': self.normalization_enabled,
            'stats_path': self.stats_path,
        }
    
    def __del__(self):
        """Cleanup when dataset is garbage collected."""
        self._cleanup_handles()
        self.clear_cache()


def attribute_dict_collate(batch):
    """
    Collate a batch of AttributeDicts into a single AttributeDict of batched tensors.
    """
    collated = default_collate(batch)
    return AttributeDict(collated)


def create_optimized_dataloader(
    hdf5_files: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader for multi-GPU training.
    
    Args:
        hdf5_files: List of HDF5 file paths
        batch_size: Batch size per GPU
        num_workers: Number of worker processes per GPU
        rank: Current GPU rank for distributed training
        world_size: Total number of GPUs
        stats_path: Path to HDF5 statistics file for data normalization
        normalize_fields: List of fields to normalize (None normalizes all available fields with stats)
        **dataset_kwargs: Additional arguments for CombinedHDF5Dataset
    
    Returns:
        Optimized DataLoader instance
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Create dataset
    dataset = CombinedHDF5Dataset(
        paths=hdf5_files,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        **dataset_kwargs
    )
    
    # Setup distributed sampler if multi-GPU
    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True  # Ensures consistent batch sizes across GPUs
        )
        shuffle = False
    
    # Optimal DataLoader settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
        collate_fn=attribute_dict_collate
    )
