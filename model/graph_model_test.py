import torch
import torch.distributed as dist
import numpy as np
import sklearn.utils
import time
import sys
import os
import yaml
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('Agg')

from utils.plot_utils import (
    plot_model_analysis,
    plot_vae_reconstruction,
    plot_transfer_entropy_vs_shift,
    plot_metrics_histograms,
    plot_te_ablation_results,
    plot_te_gain_sweep,
)
from loguru import logger
from hdf5_dataset.kymatio_frequency_analysis import analyze_scattering_frequencies
from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D
from hdf5_dataset.hdf5_dataset import normalize_tensor_data, create_optimized_dataloader
from model.graph_model_train import SeqVAEGraphModel, denormalize_signal_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"   # set to 1 only in debugging

class SeqVAEGraphModelTest(SeqVAEGraphModel):
    def __init__(self, config_file_path=None):
        super().__init__(config_file_path)

    def run_tests(self, test_loader, cuda_device=None):
        """Run SeqVAE test analyses and plots with optional CUDA device selection.

        Args:
            test_loader (DataLoader): Loader providing required tensors (e.g., `fhr_st`, `fhr_ph`,
                `fhr_up_ph`, `fhr`, `up`, `guid`). Example: created via `create_optimized_dataloader(...)`.
            cuda_device (int | str | None): Desired device index or 'cpu'. If `None`, uses the first
                configured GPU in `self.cuda_devices` when available. Examples: `0`, `1`, `'cpu'`.

        Returns:
            None: Saves analysis figures and artifacts into the test results directory. For example,
            plots are written under `.../test_results/...`.
        """
        analysis_dir = os.path.join(self.test_results_dir, 'analysis_and_plot')
        te_shift_dir = os.path.join(self.test_results_dir, 'te_shift_left')
        metrics_dir = os.path.join(self.test_results_dir, 'metrics_histograms')
        ablation_dir = os.path.join(self.test_results_dir, 'up_ablation')
        gain_sweep_dir = os.path.join(self.test_results_dir, 'up_gain_sweep')

        for d in [analysis_dir, te_shift_dir, metrics_dir, ablation_dir, gain_sweep_dir]:
            os.makedirs(d, exist_ok=True)

        try:
            if cuda_device is not None:
                if isinstance(cuda_device, str) and cuda_device.lower() == 'cpu':
                    self.set_cuda_devices([])
                    logger.info("run_tests: Using CPU as requested.")
                else:
                    device_index = int(cuda_device)
                    if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
                        self.set_cuda_devices([device_index])
                        logger.info(f"run_tests: Using CUDA device cuda:{device_index} as requested.")
                    else:
                        logger.warning(
                            f"run_tests: Requested CUDA device {cuda_device} is not available. "
                            "Falling back to default device configuration."
                        )
            else:
                # Use CUDA devices from config.yaml if no specific device requested
                if self.cuda_devices and len(self.cuda_devices) > 0 and torch.cuda.is_available():
                    # For testing, use the first available GPU from config
                    available_devices = [d for d in self.cuda_devices if d < torch.cuda.device_count()]
                    if available_devices:
                        self.set_cuda_devices([available_devices[0]])  # Use first available GPU for testing
                        logger.info(f"run_tests: Using CUDA device cuda:{available_devices[0]} from config (available: {available_devices})")
                    else:
                        logger.warning("run_tests: No configured CUDA devices available. Using CPU.")
                        self.set_cuda_devices([])
                else:
                    logger.info("run_tests: No CUDA devices configured or CUDA not available. Using CPU.")
                    
            torch.set_float32_matmul_precision('high')
        except Exception as e:
            logger.warning(f"run_tests: Device selection setup failed: {e}")

        # CRITICAL FIX: Create model ONCE at the beginning to ensure consistency
        logger.info("Creating model once for all test analyses...")
        logger.info(f"Config CUDA devices: {self.config['general_config']['cuda_devices']}")
        logger.info(f"Selected CUDA devices for testing: {self.cuda_devices}")
        
        self.create_model()
        
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting all tests.")
            return
            
        # Set model to eval mode and log the beta value being used
        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        logger.info(f"Moving model to device: {device}")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        
        # Log beta value for debugging
        effective_beta = getattr(self.lightning_base_model, 'current_beta', self.kld_beta_) if hasattr(self, 'lightning_base_model') else self.kld_beta_
        logger.info(f"Using beta value for testing: {effective_beta}")
        logger.info(f"Config beta_const_val: {getattr(self, 'beta_const_val', 'N/A')}")
        logger.info(f"Config beta_schedule: {getattr(self, 'beta_schedule', 'N/A')}")
        
        # Verify model is in correct mode and device
        logger.info(f"Model training mode: {self.pytorch_model.training}")
        if hasattr(self.pytorch_model, 'parameters'):
            sample_param = next(iter(self.pytorch_model.parameters()))
            logger.info(f"Model device: {sample_param.device}")

        target_count = 50
        selected_guids = []
        try:
            for batch in tqdm(test_loader, desc="Selecting GUIDs for tests"):
                guids_batch = batch.guid
                if isinstance(guids_batch, (list, tuple)):
                    guids_iter = guids_batch
                else:
                    try:
                        guids_iter = [str(g) for g in guids_batch]
                    except Exception:
                        guids_iter = []
                for g in guids_iter:
                    if g not in selected_guids:
                        selected_guids.append(g)
                        if len(selected_guids) >= target_count:
                            break
                if len(selected_guids) >= target_count:
                    break
        except Exception as e:
            logger.warning(f"Could not preselect GUIDs: {e}. Tests will pick samples as available.")

        # Pass the created model to avoid re-creation in each analysis function
        self.run_analysis_and_plot(test_loader, 50, output_dir=analysis_dir, selected_guids=selected_guids, model_created=True)
        self.run_transfer_entropy_shift_analysis(test_loader, output_dir=te_shift_dir, selected_guids=selected_guids, model_created=True)
        self.run_metrics_histogram_analysis(test_loader, output_dir=metrics_dir, model_created=True)
        self.run_up_ablation_analysis(test_loader, output_dir=ablation_dir, model_created=True)
        self.run_up_gain_sweep_analysis(test_loader, output_dir=gain_sweep_dir, model_created=True)

        # New: forecasting evaluation and plots (keeps legacy tests intact)
        forecast_dir = os.path.join(self.test_results_dir, 'forecast_eval')
        os.makedirs(forecast_dir, exist_ok=True)
        self.run_forecast_evaluation_and_plot(test_loader, num_samples=100, output_dir=forecast_dir, selected_guids=selected_guids, model_created=True)


    def run_analysis_and_plot(self, test_loader, num_samples=200, output_dir=None, selected_guids=None, model_created=False):
        """
        Runs a full analysis on randomly selected samples from the test loader and plots the results.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of random samples to analyze and plot (default: 50)
            model_created: If True, skip model creation (model already created)
        """
        out_dir = output_dir or self.test_results_dir
        logger.info(f"Starting model analysis and plotting on {num_samples} random samples...")
        
        if not model_created:
            self.create_model()

        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting analysis.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        if not model_created:
            self.pytorch_model.to(device)
            self.pytorch_model.eval()

        # Get normalization stats from the dataset for denormalization
        normalization_stats = None
        if hasattr(test_loader.dataset, 'get_normalization_stats'):
            normalization_stats = test_loader.dataset.get_normalization_stats()
            if normalization_stats:
                logger.info("Found normalization stats for denormalizing FHR and UP signals")
            else:
                logger.warning("No normalization stats available - will use normalized data for plotting")

        # Get scattering transform frequency analysis for channel annotations
        scattering_analysis = None
        # Build phase/cross indices for splitting plots
        phase_auto_indices = None
        phase_cross_indices = None
        cross_auto_indices = None
        cross_cross_indices = None
        try:
            # Parameters from fhr_st_setting.md - J=11, Q=4, T=16, sampling_rate=4Hz
            scattering_analysis = analyze_scattering_frequencies(
                J=11, Q=4, T=16, sampling_rate=4.0, signal_duration_minutes=20.0,
                analyze_phase_harmonics=True, analyze_cross_phase=True
            )
            logger.info("Generated scattering transform frequency analysis for channel annotations")
            # Also get selection masks to derive auto/cross indices matching dataset channel order
            st_helper = KymatioPhaseScattering1D(J=11, Q=4, T=16, shape=5760, device=device, tukey_alpha=None, max_order=1)
            sel = st_helper.get_optimal_coefficients_for_fhr(11, 4, 16)
            # Phase selected order
            i_phase = sel['phase_selection']['i_idx_selected'].detach().cpu().numpy()
            j_phase = sel['phase_selection']['j_idx_selected'].detach().cpu().numpy()
            phase_auto_indices = np.where(i_phase == j_phase)[0]
            phase_cross_indices = np.where(i_phase != j_phase)[0]
            # Cross selected order
            i_cross = sel['cross_selection']['i_idx_selected'].detach().cpu().numpy()
            j_cross = sel['cross_selection']['j_idx_selected'].detach().cpu().numpy()
            cross_auto_indices = np.where(i_cross == j_cross)[0]
            cross_cross_indices = np.where(i_cross != j_cross)[0]
        except Exception as e:
            logger.warning(f"Could not generate scattering frequency analysis: {e}")
            scattering_analysis = None

        # Collect all samples from the test loader
        logger.info("Collecting all samples from test loader...")
        all_samples = []
        try:
            with torch.inference_mode():
                for batch_data in tqdm(test_loader, desc="Collecting samples"):
                    batch_size = batch_data.fhr_st.size(0)
                    for i in range(batch_size):
                        guid_val = None
                        try:
                            guid_val = batch_data.guid[i] if isinstance(batch_data.guid, (list, tuple)) else str(batch_data.guid[i])
                        except Exception:
                            guid_val = None
                        if selected_guids and guid_val not in selected_guids:
                            continue
                        sample = {
                            'fhr_st': batch_data.fhr_st[i],
                            'fhr_ph': batch_data.fhr_ph[i],
                            'fhr_up_ph': batch_data.fhr_up_ph[i],
                            'fhr': batch_data.fhr[i],
                            'up': batch_data.up[i]
                        }
                        all_samples.append(sample)
        except Exception as e:
            logger.error(f"Error collecting samples: {e}")
            return

        if len(all_samples) == 0:
            logger.error("No samples found in test loader. Cannot perform analysis.")
            return

        np.random.seed(42)  # For reproducibility
        total_samples = len(all_samples)
        num_samples = min(num_samples, total_samples)
        selected_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        
        logger.info(f"Selected {num_samples} random samples from {total_samples} total samples")
        logger.info(f"Selected sample indices: {selected_indices[:10]}..." if num_samples > 10 else f"Selected sample indices: {selected_indices}")

        # Process each selected sample
        with torch.inference_mode():
            for plot_idx, sample_idx in enumerate(tqdm(selected_indices, desc="Processing selected samples")):
                try:
                    sample = all_samples[sample_idx]
                    
                    # Move sample data to device and add batch dimension
                    y_st = sample['fhr_st'].unsqueeze(0).to(device)
                    y_ph = sample['fhr_ph'].unsqueeze(0).to(device)
                    x_ph = sample['fhr_up_ph'].unsqueeze(0).to(device)
                    y_raw = sample['fhr'].unsqueeze(0).to(device)
                    up_raw = sample['up'].unsqueeze(0).to(device)

                    # Get model outputs
                    forward_outputs = self.pytorch_model(y_st, y_ph, x_ph)
                    latent_z = forward_outputs['z']
                    reconstructed_fhr_mu = forward_outputs['mu_pr']
                    reconstructed_fhr_logvar = forward_outputs['logvar_pr']

                    # Compute loss the same way as training to get consistent KLD values
                    # CRITICAL FIX: Use the same beta as training (beta_const_val from config)
                    effective_beta = getattr(self, 'beta_const_val', self.kld_beta_)
                    loss_dict = self.pytorch_model.compute_loss(
                        forward_outputs, y_st, y_ph, y_raw, 
                        compute_kld_loss=True, 
                        beta=effective_beta)
                    
                    # Also get KLD tensor for detailed analysis (original method)
                    kld_tensor = self.pytorch_model.measure_transfer_entropy(y_st, y_ph, x_ph, reduce_mean=False)
                    kld_mean_over_channels = kld_tensor.mean(dim=-1)

                    # Always keep normalized versions for reconstruction comparison
                    raw_fhr_normalized_np = y_raw[0].cpu().numpy()
                    raw_up_normalized_np = up_raw[0].cpu().numpy()
                    
                    # Denormalize FHR and UP signals if normalization stats are available
                    if normalization_stats:
                        # Denormalize the normalized signals to get the original raw signals for first plot
                        raw_fhr_denormalized = denormalize_signal_data(y_raw[0], 'fhr', normalization_stats)
                        raw_up_denormalized = denormalize_signal_data(up_raw[0], 'up', normalization_stats)
                        raw_fhr_unnormalized_np = raw_fhr_denormalized.cpu().numpy()
                        raw_up_unnormalized_np = raw_up_denormalized.cpu().numpy()
                        
                        # Log info for first sample to confirm denormalization is working
                        if plot_idx == 0:
                            logger.info(f"Using denormalized FHR and UP signals for first plot, normalized for reconstruction plot")
                            logger.info(f"Unnormalized FHR range: [{raw_fhr_unnormalized_np.min():.2f}, {raw_fhr_unnormalized_np.max():.2f}]")
                            logger.info(f"Unnormalized UP range: [{raw_up_unnormalized_np.min():.2f}, {raw_up_unnormalized_np.max():.2f}]")
                            logger.info(f"Normalized FHR range: [{raw_fhr_normalized_np.min():.2f}, {raw_fhr_normalized_np.max():.2f}]")
                    else:
                        # Use normalized data if no stats available
                        raw_fhr_unnormalized_np = raw_fhr_normalized_np
                        raw_up_unnormalized_np = raw_up_normalized_np
                        
                        # Log warning for first sample
                        if plot_idx == 0:
                            logger.warning("Using normalized FHR and UP signals for plotting (no denormalization stats available)")
                        
                    # Move other data to CPU and convert to numpy for plotting (remove batch dimension)
                    fhr_st_np = y_st[0].cpu().numpy().T
                    fhr_ph_np = y_ph[0].cpu().numpy().T
                    fhr_up_ph_np = x_ph[0].cpu().numpy().T
                    latent_z_np = latent_z[0].cpu().numpy().T
                    reconstructed_fhr_mu_np = reconstructed_fhr_mu[0].cpu().numpy()
                    reconstructed_fhr_logvar_np = reconstructed_fhr_logvar[0].cpu().numpy()
                    kld_tensor_np = kld_tensor[0].cpu().numpy().T
                    kld_mean_over_channels_np = kld_mean_over_channels[0].cpu().numpy()
                    
                    # Extract reconstructed scattering and phase harmonic coefficients from linear_output
                    linear_output = forward_outputs['linear_output']  # Shape: (1, 300, 87)
                    linear_output_np = linear_output[0].cpu().numpy()  # Shape: (300, 87)
                    
                    # Split into scattering (43) and phase harmonic (44) components
                    reconstructed_st_np = linear_output_np[:, :43].T  # Shape: (43, 300)
                    reconstructed_ph_np = linear_output_np[:, 43:].T  # Shape: (44, 300)

                    # Generate plots for this sample
                    plot_model_analysis(
                        output_dir=out_dir,
                        raw_fhr=raw_fhr_unnormalized_np,  # Unnormalized for first plot
                        raw_up=raw_up_unnormalized_np,    # Unnormalized for first plot
                        fhr_st=fhr_st_np,
                        fhr_ph=fhr_ph_np,
                        fhr_up_ph=fhr_up_ph_np,
                        latent_z=latent_z_np,
                        reconstructed_fhr_mu=reconstructed_fhr_mu_np,
                        reconstructed_fhr_logvar=reconstructed_fhr_logvar_np,
                        kld_tensor=kld_tensor_np,
                        kld_mean_over_channels=kld_mean_over_channels_np,
                        batch_idx=sample_idx,  # Use original sample index for unique file naming
                        loss_dict=loss_dict,  # Pass training-consistent loss values
                        # Pass normalized versions for reconstruction comparison
                        raw_fhr_normalized=raw_fhr_normalized_np,
                        raw_up_normalized=raw_up_normalized_np,
                        phase_auto_indices=phase_auto_indices,
                        phase_cross_indices=phase_cross_indices,
                        cross_auto_indices=cross_auto_indices,
                        cross_cross_indices=cross_cross_indices
                    )
                    
                    # Generate VAE reconstruction plots for this sample
                    plot_vae_reconstruction(
                        output_dir=out_dir,
                        raw_fhr_unnormalized=raw_fhr_unnormalized_np,
                        raw_up_unnormalized=raw_up_unnormalized_np,
                        raw_fhr_normalized=raw_fhr_normalized_np,
                        raw_up_normalized=raw_up_normalized_np,
                        reconstructed_fhr=reconstructed_fhr_mu_np,
                        original_scattering_transform=fhr_st_np,  # Already transposed to (43, 300)
                        reconstructed_scattering_transform=reconstructed_st_np,  # Shape: (43, 300)
                        original_phase_harmonic=fhr_ph_np,  # Already transposed to (44, 300)
                        reconstructed_phase_harmonic=reconstructed_ph_np,  # Shape: (44, 300)
                        scattering_channel_data=scattering_analysis,  # Frequency analysis data
                        batch_idx=sample_idx,
                        loss_dict=loss_dict
                    )
                    
                    # Log progress every 10 samples
                    if (plot_idx + 1) % 10 == 0:
                        logger.info(f"Completed analysis for {plot_idx + 1}/{num_samples} samples")
                        
                except Exception as e:
                    logger.warning(f"Failed to process sample {sample_idx}: {e}")
                    continue

        logger.info(f"Model analysis and plotting complete for {num_samples} samples.")
        logger.info(f"Plots saved to: {out_dir}")

    def run_transfer_entropy_shift_analysis(self, test_loader, num_samples=None, max_left_shift_seconds=60, step_seconds=1, output_dir=None, selected_guids=None, model_created=False):
        """Measure and plot TE (KLD) vs UP left-shift per sample (no averaging).

        Args:
            test_loader: DataLoader with normalized coeffs and raw signals.
            num_samples (int | None): Limit number of samples to analyze (None = all).
            max_left_shift_seconds (int): Maximum LEFT shift seconds (negative direction).
            step_seconds (int): Shift step in seconds.
            output_dir (str | None): Directory to save figures.
            selected_guids (List[str] | None): Subset of GUIDs to analyze (e.g., the 50 chosen in run_tests).
        """
        out_dir = output_dir or self.test_results_dir
        logger.info(
            f"Starting per-sample TE vs shift (LEFT only). Samples: {('ALL' if num_samples is None else num_samples)}, max_left_shift_seconds={max_left_shift_seconds}, step={step_seconds}"
        )
        
        if not model_created:
            self.create_model()
        
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting shift analysis.")
            return
            
        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        if not model_created:
            self.pytorch_model.to(device)
            self.pytorch_model.eval()
        
        # Get normalization stats for the fhr_up_ph field
        normalization_stats = None
        if hasattr(test_loader.dataset, 'get_normalization_stats'):
            normalization_stats = test_loader.dataset.get_normalization_stats()
            if not normalization_stats or 'fhr_up_ph' not in normalization_stats:
                logger.error("No normalization stats found for fhr_up_ph field. Cannot proceed with analysis.")
                return
        else:
            logger.error("Dataset does not provide normalization stats. Cannot proceed with analysis.")
            return
            
        # Initialize scattering transform for cross-phase computation
        # Use parameters matching dataset creation: J=11, Q=4, T=16, shape=5760, max_order=1
        scattering_transform = KymatioPhaseScattering1D(
            J=11, Q=4, T=16, shape=5760, device=device, tukey_alpha=None, max_order=1
        )
        scattering_transform.to(device)
        scattering_transform.eval()
        
        # Get optimal coefficient selection masks (same as dataset creation)
        optimal_selection = scattering_transform.get_optimal_coefficients_for_fhr(11, 4, 16)
        cross_mask = optimal_selection['recommendations']['use_cross_mask']
        logger.info(f"Using cross-channel mask with {cross_mask.sum().item()} selected coefficients")
        
        # Create a temporary dataset without trimming to get raw signals
        logger.info("Creating dataset without trimming to access raw signals...")
        from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset
        
        # Get dataset config from test_loader
        dataset_paths = test_loader.dataset.paths
        stats_path = test_loader.dataset.stats_path
        allowed_guids = None
        if hasattr(test_loader.dataset, 'allowed_guids'):
            allowed_guids = list(test_loader.dataset.allowed_guids) if test_loader.dataset.allowed_guids else None
            
        # Create dataset without trimming for raw signal access
        raw_dataset = CombinedHDF5Dataset(
            paths=dataset_paths,
            load_fields=['fhr', 'up', 'fhr_st', 'fhr_ph', 'guid'],  # Include guid for per-sample outputs
            allowed_guids=selected_guids if selected_guids else allowed_guids,
            stats_path=stats_path,
            trim_minutes=None,  # No trimming to get full raw signals
            normalize_fields=['fhr_st', 'fhr_ph']  # Only normalize what we need, keep fhr/up raw
        )

        # Iterate samples on-the-fly (no averaging across samples)
        logger.info("Processing samples for per-sample shift analysis...")
        
        # Define LEFT shift range only: [-max_left_shift_seconds, 0] in step_seconds increments
        # At 4Hz sampling rate: 1 second = 4 samples
        sampling_rate = 4.0  # Hz
        shift_seconds = np.arange(-int(max_left_shift_seconds), 0 + 1, int(step_seconds))
        shift_samples = (shift_seconds * sampling_rate).astype(int)
        
        logger.info(f"Testing {len(shift_samples)} left shifts from {shift_seconds[0]}s to {shift_seconds[-1]}s")
        
        # Calculate 2-minute trimming parameters
        trim_minutes = 2.0
        trim_samples_raw = int(4 * 60 * trim_minutes)  # 480 samples at 4Hz
        trim_samples_decimated = trim_samples_raw // 16  # 30 samples for coefficients
        

        # Iterate samples, compute per-sample KLD vs shift, and plot per-sample figures
        with torch.inference_mode():
            total_items = len(raw_dataset)
            processed = 0
            for sample_idx in range(total_items):
                if num_samples is not None and processed >= num_samples:
                    break

                try:
                    sample = raw_dataset[sample_idx]

                    # Prepare raw and coeff data
                    fhr_raw = sample['fhr'].cpu().numpy()  # (5760,)
                    up_raw = sample['up'].cpu().numpy()    # (5760,)
                    fhr_st = sample['fhr_st']              # (300, 43) normalized
                    fhr_ph = sample['fhr_ph']              # (300, 44) normalized
                    guid = sample.get('guid', None)
                    guid_safe = ''.join([c if str(c).isalnum() else '_' for c in str(guid)]) if guid is not None else 'NA'

                    # Per-sample accumulators
                    kld_per_shift = []
                    per_sample_signal_plot_data = []

                    for shift_idx, (shift_sec, shift_samp) in enumerate(zip(shift_seconds, shift_samples)):
                        # Shift UP
                        up_shifted = self._apply_circular_shift(up_raw, shift_samp)

                        # Stack raw for scattering
                        st_input = torch.from_numpy(np.stack([fhr_raw, up_shifted], axis=0)).float().unsqueeze(0).to(device)  # (1, 2, 5760)

                        # Cross-phase computation (as in dataset creation)
                        st_results_cross = scattering_transform(
                            x=st_input,
                            compute_phase=False,
                            compute_cross_phase=True,
                            scattering_channel=0,
                            phase_channels=[0, 1]
                        )
                        fhr_up_cc_phase_full = st_results_cross.get('cross_phase_corr')
                        cross_phase_raw = fhr_up_cc_phase_full[:, cross_mask, :] if fhr_up_cc_phase_full is not None else None
                        cross_phase_formatted = cross_phase_raw.transpose(1, 2)  # (1, 300, 130)

                        # Normalize with existing stats
                        cross_phase_normalized = normalize_tensor_data(
                            data=cross_phase_formatted,
                            field_name='fhr_up_ph',
                            normalization_stats=normalization_stats,
                            log_norm_channels_config=raw_dataset.log_norm_channels_config,
                            asinh_norm_channels_config=raw_dataset.asinh_norm_channels_config,
                            log_epsilon=raw_dataset.log_epsilon,
                            pin_memory=False,
                            normalize_fields=raw_dataset.normalize_fields,
                            dtype=torch.float32
                        )

                        # Trim 2 minutes from both ends
                        if trim_samples_decimated > 0:
                            cross_phase_trimmed = cross_phase_normalized[:, trim_samples_decimated:-trim_samples_decimated, :]
                            fhr_st_trimmed = fhr_st[trim_samples_decimated:-trim_samples_decimated, :]
                            fhr_ph_trimmed = fhr_ph[trim_samples_decimated:-trim_samples_decimated, :]
                        else:
                            cross_phase_trimmed = cross_phase_normalized
                            fhr_st_trimmed = fhr_st
                            fhr_ph_trimmed = fhr_ph

                        # Inputs for model
                        y_st_input = fhr_st_trimmed.unsqueeze(0).to(device)
                        y_ph_input = fhr_ph_trimmed.unsqueeze(0).to(device)
                        x_ph_input = cross_phase_trimmed.to(device)

                        # TE (KLD)
                        kld_tensor = self.pytorch_model.measure_transfer_entropy(
                            y_st=y_st_input, y_ph=y_ph_input, x_ph=x_ph_input, reduce_mean=False
                        )
                        kld_value = kld_tensor.mean().item()
                        kld_per_shift.append(kld_value)

                        # Store trimmed signals for the second plot
                        if trim_samples_raw > 0:
                            fhr_trimmed = fhr_raw[trim_samples_raw:-trim_samples_raw]
                            up_trimmed = up_raw[trim_samples_raw:-trim_samples_raw]
                            up_shifted_trimmed = up_shifted[trim_samples_raw:-trim_samples_raw]
                        else:
                            fhr_trimmed = fhr_raw
                            up_trimmed = up_raw
                            up_shifted_trimmed = up_shifted

                        per_sample_signal_plot_data.append({
                            'sample_idx': sample_idx,
                            'shift_sec': shift_sec,
                            'fhr': fhr_trimmed,
                            'up_original': up_trimmed,
                            'up_shifted': up_shifted_trimmed,
                            'kld': kld_value
                        })

                    # Per-sample plot: TE vs shift
                    try:
                        import matplotlib.pyplot as plt
                        os.makedirs(out_dir, exist_ok=True)
                        fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
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

                        ax.plot(shift_seconds, kld_per_shift, color="#055C9A", marker='o', linewidth=2)
                        ax.set_xlabel('UP Shift (seconds)')
                        ax.set_ylabel('Transfer Entropy (KLD)')
                        ax.set_title(f'TE vs Shift — Sample {sample_idx} — GUID: {guid_safe}')
                        # Mark minimum
                        min_idx = int(np.argmin(kld_per_shift)) if len(kld_per_shift) > 0 else None
                        if min_idx is not None:
                            ax.plot(shift_seconds[min_idx], kld_per_shift[min_idx], color="#BB3E00", marker='o', markersize=8)
                            ax.text(
                                shift_seconds[min_idx], kld_per_shift[min_idx], 
                                f'  Min {shift_seconds[min_idx]}s\n  KLD {kld_per_shift[min_idx]:.6f}',
                                va='bottom', ha='left',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='#A2B9A7'), fontsize=9
                            )

                        save_path_curve = os.path.join(out_dir, f'te_vs_shift_sample_{sample_idx}_{guid_safe}.png')
                        plt.savefig(save_path_curve, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                        plt.close(fig)
                        logger.info(f"Saved TE vs shift for sample {sample_idx} (GUID: {guid_safe}) → {save_path_curve}")
                    except Exception as e:
                        logger.warning(f"Failed to plot TE vs shift for sample {sample_idx}: {e}")

                    # Per-sample plot: shifted UP vs FHR with KLD annotations
                    try:
                        self._plot_signal_shift_examples(per_sample_signal_plot_data, sampling_rate, output_dir=out_dir)
                    except Exception as e:
                        logger.warning(f"Failed to plot signal shift examples for sample {sample_idx}: {e}")

                    processed += 1

                except Exception as e:
                    logger.warning(f"Failed to process sample index {sample_idx}: {e}")
                    continue

    def _apply_circular_shift(self, signal, shift_samples):
        """
        Apply circular shift to a signal (no zero-padding, preserves all information).
        
        Args:
            signal: 1D numpy array of signal values
            shift_samples: Number of samples to shift (positive = shift right/delay, negative = shift left/advance)
        
        Returns:
            Circularly shifted signal of the same length
        """
        if shift_samples == 0:
            return signal.copy()
        
        # Use numpy's roll for circular shift
        return np.roll(signal, shift_samples)

    def _plot_signal_shift_examples(self, signal_plot_data, sampling_rate, output_dir=None):
        """
        Plot examples of FHR, original UP, and shifted UP signals.
        
        Args:
            signal_plot_data: List of dictionaries containing signal data for different shifts
            sampling_rate: Sampling rate in Hz
        """
        out_dir = output_dir or self.test_results_dir
        if not signal_plot_data:
            return
            
        # Group data by sample
        samples_data = {}
        for data in signal_plot_data:
            sample_idx = data['sample_idx']
            if sample_idx not in samples_data:
                samples_data[sample_idx] = []
            samples_data[sample_idx].append(data)
        
        # Plot each sample
        for sample_idx, sample_shifts in samples_data.items():
            fig, axes = plt.subplots(len(sample_shifts), 1, figsize=(16, len(sample_shifts) * 4), constrained_layout=True)
            if len(sample_shifts) == 1:
                axes = [axes]
                
            for i, data in enumerate(sample_shifts):
                t = np.arange(len(data['fhr'])) / sampling_rate
                
                axes[i].plot(t, data['fhr'], color='#055C9A', label='FHR', linewidth=1.2, alpha=0.8)
                axes[i].plot(t, data['up_original'], color='#0DD8A2', label='UP Original', linewidth=1.2, alpha=0.8)
                axes[i].plot(t, data['up_shifted'], color='#BB3E00', label=f'UP Shifted ({data["shift_sec"] }s)', linewidth=1.2, alpha=0.8)
                
                axes[i].set_title(f'Sample {sample_idx} - Shift: {data["shift_sec"]}s - KLD: {data["kld"]:.6f}', fontweight='normal', pad=12)
                axes[i].set_ylabel('Amplitude', fontweight='normal')
                axes[i].legend(loc='upper right', framealpha=0.95)
                axes[i].grid(True, alpha=0.3)
                
                if i == len(sample_shifts) - 1:
                    axes[i].set_xlabel('Time (s)', fontweight='normal')
            
            fig.suptitle(f'Signal Shift Examples - Sample {sample_idx}', fontsize=14, fontweight='normal', y=0.98)
            
            # Save plot
            plot_path = os.path.join(out_dir, f'signal_shift_examples_sample_{sample_idx}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Signal shift examples for sample {sample_idx} saved to: {plot_path}")

    # ------------------------------
    # New: Forecasting evaluation and plotting
    # ------------------------------
    def run_forecast_evaluation_and_plot(self, test_loader, num_samples=100, output_dir=None, selected_guids=None, model_created=False):
        out_dir = output_dir or os.path.join(self.test_results_dir, 'forecast_eval')
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Starting forecasting evaluation on up to {num_samples} samples...")

        if not model_created:
            self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting forecast evaluation.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        if not model_created:
            self.pytorch_model.to(device)
            self.pytorch_model.eval()

        # Collect samples
        samples = []
        count = 0
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Collecting samples for forecast eval"):
                bsz = batch.fhr_st.size(0)
                for i in range(bsz):
                    if count >= num_samples:
                        break
                    guid_ok = True
                    if selected_guids is not None:
                        try:
                            guid_val = batch.guid[i] if isinstance(batch.guid, (list, tuple)) else str(batch.guid[i])
                            guid_ok = (guid_val in selected_guids)
                        except Exception:
                            guid_ok = True
                    if not guid_ok:
                        continue
                    samples.append({
                        'fhr_st': batch.fhr_st[i],
                        'fhr_ph': batch.fhr_ph[i],
                        'fhr_up_ph': batch.fhr_up_ph[i],
                        'fhr': batch.fhr[i],
                    })
                    count += 1
                if count >= num_samples:
                    break

        if len(samples) == 0:
            logger.warning("No samples collected for forecasting evaluation.")
            return

        # Metrics arrays
        mse_list, mae_list, corr_list = [], [], []

        # Plot a few examples
        example_indices = list(range(min(6, len(samples))))

        with torch.inference_mode():
            for idx, sample in enumerate(tqdm(samples, desc="Forecasting eval")):
                y_st = sample['fhr_st'].unsqueeze(0).to(device)
                y_ph = sample['fhr_ph'].unsqueeze(0).to(device)
                x_ph = sample['fhr_up_ph'].unsqueeze(0).to(device)
                y_raw = sample['fhr'].unsqueeze(0).to(device)

                # Evaluate forecast metrics and get aggregated predictions
                out = self.pytorch_model.evaluate_forecast_batch(y_st, y_ph, x_ph, y_raw, use_posterior_mean=True)
                mse = float(out['mse'][0].item())
                mae = float(out['mae'][0].item())
                corr = float(out['corr'][0].item()) if not torch.isnan(out['corr'][0]) else float('nan')
                mse_list.append(mse)
                mae_list.append(mae)
                corr_list.append(corr)

                # Plot sample forecasts for first few samples
                if idx in example_indices:
                    self._plot_forecast_example(
                        y_raw=y_raw[0].detach().cpu(),
                        mean_mu=out['mean_mu'][0].detach().cpu(),
                        std_mu=out['std_mu'][0].detach().cpu(),
                        anchors=out['anchors'].detach().cpu(),
                        canvas_mu=self.pytorch_model.aggregate_forecasts_to_canvas(
                            self.pytorch_model.forecast(y_st, y_ph, x_ph, use_posterior_mean=True)["mu_future"],
                            out['anchors'], total_len=y_raw.shape[1], stride=self.pytorch_model.decimation_factor
                        )[0][0].detach().cpu(),
                        save_path=os.path.join(out_dir, f'forecast_sample_{idx}.pdf')
                    )

        # Save metrics summary and histograms
        self._plot_forecast_metrics_histograms(mse_list, mae_list, corr_list, out_dir)
        with open(os.path.join(out_dir, 'forecast_metrics.pkl'), 'wb') as f:
            pickle.dump({"mse": mse_list, "mae": mae_list, "corr": corr_list}, f)
        logger.info(
            f"Forecast metrics — MSE: mean={np.nanmean(mse_list):.6f}, std={np.nanstd(mse_list):.6f}; "
            f"MAE: mean={np.nanmean(mae_list):.6f}, std={np.nanstd(mae_list):.6f}; "
            f"Corr: mean={np.nanmean(corr_list):.4f}, std={np.nanstd(corr_list):.4f}"
        )

    def _plot_forecast_example(self, y_raw, mean_mu, std_mu, anchors, canvas_mu, save_path):
        import matplotlib.pyplot as plt
        import numpy as np
        t = np.arange(y_raw.shape[0]) / 4.0
        fig, ax = plt.subplots(2, 1, figsize=(16, 7), constrained_layout=True)
        # Aggregated view
        ax[0].plot(t, y_raw.numpy(), color='#2E86AB', label='GT', linewidth=1.5)
        ax[0].plot(t, mean_mu.numpy(), color='#D7263D', label='Forecast mean', linewidth=1.2)
        ax[0].fill_between(t, (mean_mu - std_mu).numpy(), (mean_mu + std_mu).numpy(), color='#F5B7B1', alpha=0.4, label='±1σ')
        ax[0].set_title('Forecast: Aggregated prediction with uncertainty')
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('FHR (bpm)')
        # Overlay a few windows
        ax[1].plot(t, y_raw.numpy(), color='#2E86AB', alpha=0.3, linewidth=1.0)
        cmu = canvas_mu.numpy()  # (N,4800) with NaNs
        if anchors.numel() > 0:
            anc = anchors.numpy()
            picks = [anc[0], anc[len(anc)//2], anc[-1]] if len(anc) >= 3 else list(anc)
            for a in picks:
                idx = int(np.where(anc == a)[0][0])
                ax[1].plot(t, cmu[idx], color='#D7263D', linewidth=1.0, alpha=0.8)
        ax[1].set_title('Sample forecast windows')
        ax[1].set_ylabel('FHR (bpm)')
        ax[1].set_xlabel('Time (s)')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved forecast example to {save_path}")

    def _plot_forecast_metrics_histograms(self, mse_list, mae_list, corr_list, out_dir):
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
        ax[0].hist([v for v in mse_list if np.isfinite(v)], bins=40, color='#6AAED6')
        ax[0].set_title('MSE (forecast)')
        ax[1].hist([v for v in mae_list if np.isfinite(v)], bins=40, color='#FF9F80')
        ax[1].set_title('MAE (forecast)')
        ax[2].hist([v for v in corr_list if np.isfinite(v)], bins=40, color='#A0D683')
        ax[2].set_title('Pearson Corr (forecast)')
        for a in ax:
            a.grid(True, alpha=0.3)
        save_path = os.path.join(out_dir, 'forecast_metrics_histograms.png')
        fig.suptitle('Forecasting Metrics')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved forecast metrics histograms to {save_path}")

    def run_metrics_histogram_analysis(self, test_loader, num_samples=None, output_dir=None, selected_guids=None, model_created=False):
        """
        Calculate VAF, MSE, SNR between normalized raw FHR and reconstructed FHR,
        and KLD loss for each sample, then plot histograms of these metrics.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to analyze (None = all samples)
        """
        out_dir = output_dir or self.test_results_dir
        logger.info("Starting metrics histogram analysis...")
        
        if not model_created:
            self.create_model()
        
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting metrics analysis.")
            return
            
        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        if not model_created:
            self.pytorch_model.to(device)
            self.pytorch_model.eval()
        
        # Get normalization stats for denormalization
        normalization_stats = None
        if hasattr(test_loader.dataset, 'get_normalization_stats'):
            normalization_stats = test_loader.dataset.get_normalization_stats()
            
        # Collect all samples
        all_samples = []
        sample_count = 0
        max_samples = num_samples if num_samples is not None else float('inf')
        
        try:
            with torch.inference_mode():
                for batch_data in tqdm(test_loader, desc="Collecting samples"):
                    if sample_count >= max_samples:
                        break
                        
                    batch_size = batch_data.fhr_st.size(0)
                    for i in range(batch_size):
                        if sample_count >= max_samples:
                            break
                        guid_val = None
                        try:
                            guid_val = batch_data.guid[i] if isinstance(batch_data.guid, (list, tuple)) else str(batch_data.guid[i])
                        except Exception:
                            guid_val = None
                        if selected_guids and guid_val not in selected_guids:
                            continue
                        sample = {
                            'fhr_st': batch_data.fhr_st[i],
                            'fhr_ph': batch_data.fhr_ph[i], 
                            'fhr_up_ph': batch_data.fhr_up_ph[i],
                            'fhr': batch_data.fhr[i]
                        }
                        all_samples.append(sample)
                        sample_count += 1
                        
        except Exception as e:
            logger.error(f"Error collecting samples: {e}")
            return
            
        if len(all_samples) == 0:
            logger.error("No samples found in test loader.")
            return
            
        logger.info(f"Analyzing {len(all_samples)} samples for metrics calculation")
        
        # Storage for metrics
        vaf_values = []
        mse_values = []
        snr_values = []
        kld_values = []
        
        # Process each sample
        with torch.inference_mode():
            for sample_idx, sample in enumerate(tqdm(all_samples, desc="Computing metrics")):
                try:
                    # Move sample data to device and add batch dimension
                    y_st = sample['fhr_st'].unsqueeze(0).to(device)
                    y_ph = sample['fhr_ph'].unsqueeze(0).to(device) 
                    x_ph = sample['fhr_up_ph'].unsqueeze(0).to(device)
                    y_raw = sample['fhr'].unsqueeze(0).to(device)
                    
                    # Get model outputs
                    forward_outputs = self.pytorch_model(y_st, y_ph, x_ph)
                    reconstructed_fhr_mu = forward_outputs['mu_pr']  # (1, 4800)
                    
                    # Compute KLD using the model's method
                    kld_tensor = self.pytorch_model.measure_transfer_entropy(
                        y_st, y_ph, x_ph, reduce_mean=False
                    )
                    # Average KLD over sequence length and latent dimensions
                    sample_kld = kld_tensor.mean().item()
                    kld_values.append(sample_kld)
                    
                    # Move to CPU for metric calculations
                    y_raw_np = y_raw[0].cpu().numpy()  # (4800,)
                    reconstructed_fhr_np = reconstructed_fhr_mu[0].cpu().numpy()  # (4800,)
                    
                    # Handle normalization - we want normalized versions for fair comparison
                    if normalization_stats and 'fhr' in normalization_stats:
                        # Both signals should be normalized to compute metrics fairly
                        original_fhr_normalized = y_raw_np  # Already normalized from dataset
                        reconstructed_fhr_normalized = reconstructed_fhr_np  # Model output should be in same scale
                    else:
                        # Use as-is if no normalization stats
                        original_fhr_normalized = y_raw_np
                        reconstructed_fhr_normalized = reconstructed_fhr_np
                    
                    # Calculate VAF (Variance Accounted For)
                    # VAF = 1 - var(original - reconstructed) / var(original)
                    residual = original_fhr_normalized - reconstructed_fhr_normalized
                    var_residual = np.var(residual)
                    var_original = np.var(original_fhr_normalized)
                    
                    if var_original > 1e-12:  # Avoid division by zero
                        vaf = 1.0 - (var_residual / var_original)
                        vaf = max(0.0, min(1.0, vaf))  # Clamp to [0, 1]
                    else:
                        vaf = 0.0
                    vaf_values.append(vaf)
                    
                    # Calculate MSE
                    mse = np.mean((original_fhr_normalized - reconstructed_fhr_normalized) ** 2)
                    mse_values.append(mse)
                    
                    # Calculate SNR (Signal-to-Noise Ratio) in dB
                    # SNR = 10 * log10(signal_power / noise_power)
                    signal_power = np.mean(original_fhr_normalized ** 2)
                    noise_power = np.mean(residual ** 2)
                    
                    if noise_power > 1e-12:  # Avoid division by zero
                        snr_db = 10.0 * np.log10(signal_power / noise_power)
                    else:
                        snr_db = 100.0  # Very high SNR when noise is negligible
                    snr_values.append(snr_db)
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {sample_idx}: {e}")
                    continue
        
        # Log statistics
        logger.info(f"Computed metrics for {len(vaf_values)} samples")
        logger.info(f"VAF - Mean: {np.mean(vaf_values):.4f}, Std: {np.std(vaf_values):.4f}")
        logger.info(f"MSE - Mean: {np.mean(mse_values):.6f}, Std: {np.std(mse_values):.6f}")
        logger.info(f"SNR - Mean: {np.mean(snr_values):.2f} dB, Std: {np.std(snr_values):.2f} dB")
        logger.info(f"KLD - Mean: {np.mean(kld_values):.6f}, Std: {np.std(kld_values):.6f}")
        
        # Plot histograms using the plotting function from utils
        plot_metrics_histograms(vaf_values, mse_values, snr_values, kld_values, out_dir)
        
        # Save metrics data
        metrics_data = {
            'vaf': vaf_values,
            'mse': mse_values, 
            'snr': snr_values,
            'kld': kld_values,
            'num_samples': len(vaf_values),
            'statistics': {
                'vaf': {'mean': np.mean(vaf_values), 'std': np.std(vaf_values)},
                'mse': {'mean': np.mean(mse_values), 'std': np.std(mse_values)},
                'snr': {'mean': np.mean(snr_values), 'std': np.std(snr_values)},
                'kld': {'mean': np.mean(kld_values), 'std': np.std(kld_values)}
            }
        }
        
        results_path = os.path.join(out_dir, 'metrics_histogram_analysis.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(metrics_data, f)
            
        logger.info(f"Metrics histogram analysis complete. Results saved to: {results_path}")

    def run_up_ablation_analysis(self, test_loader, num_samples=None, output_dir=None, selected_guids=None, model_created=False):
        """Compare TE (KLD) and reconstruction quality (VAF) with and without UP input.

        Args:
            test_loader (DataLoader): Loader providing normalized tensors. e.g., create_optimized_dataloader(...)
            num_samples (int | None): Limit number of samples evaluated. e.g., 200; None = all.

        Returns:
            None: Saves an ablation plot showing distributions and mean±std bars.
        """
        out_dir = output_dir or self.test_results_dir
        logger.info("Starting UP ablation analysis (with vs without UP)...")
        
        if not model_created:
            self.create_model()

        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting ablation analysis.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        model = self.pytorch_model.to(device) if not model_created else self.pytorch_model
        model.eval()

        kld_with_up, kld_without_up = [], []
        vaf_with_up, vaf_without_up = [], []

        processed = 0
        max_samples = num_samples  # None means no limit

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="UP Ablation"):
                # Respect sample cap
                batch_size = batch.fhr_st.size(0)
                if max_samples is not None and processed >= max_samples:
                    break
                if max_samples is None:
                    take = batch_size
                else:
                    remaining = max_samples - processed
                    if remaining <= 0:
                        break
                    take = min(batch_size, remaining)

                # Build a mask for selected GUIDs
                if selected_guids:
                    guids_batch = batch.guid if isinstance(batch.guid, (list, tuple)) else [str(g) for g in batch.guid]
                    idx_keep = [i for i in range(take) if guids_batch[i] in selected_guids]
                else:
                    idx_keep = list(range(take))

                if len(idx_keep) == 0:
                    continue

                y_st = batch.fhr_st[idx_keep].to(device)
                y_ph = batch.fhr_ph[idx_keep].to(device)
                x_ph = batch.fhr_up_ph[idx_keep].to(device)
                y_raw = batch.fhr[idx_keep].to(device)

                # With UP
                out_up = model(y_st, y_ph, x_ph)
                mu_pr_up = out_up['mu_pr']  # (B, 4800)
                kld_tensor_up = model.measure_transfer_entropy(y_st, y_ph, x_ph, reduce_mean=False)
                kld_up = kld_tensor_up.mean(dim=(1, 2))  # per-sample

                # Without UP (zeroed source)
                x_zero = torch.zeros_like(x_ph)
                out_no = model(y_st, y_ph, x_zero)
                mu_pr_no = out_no['mu_pr']
                kld_tensor_no = model.measure_transfer_entropy(y_st, y_ph, x_zero, reduce_mean=False)
                kld_no = kld_tensor_no.mean(dim=(1, 2))

                # VAF per-sample (normalized space)
                for i in range(y_st.size(0)):
                    gt = y_raw[i].detach().cpu().numpy()
                    pr_up = mu_pr_up[i].detach().cpu().numpy()
                    pr_no = mu_pr_no[i].detach().cpu().numpy()

                    res_up = gt - pr_up
                    res_no = gt - pr_no
                    var_gt = np.var(gt)
                    if var_gt > 1e-12:
                        vaf_w = 1.0 - (np.var(res_up) / var_gt)
                        vaf_wo = 1.0 - (np.var(res_no) / var_gt)
                        # Keep within [0,1] as elsewhere
                        vaf_w = max(0.0, min(1.0, float(vaf_w)))
                        vaf_wo = max(0.0, min(1.0, float(vaf_wo)))
                    else:
                        vaf_w = 0.0
                        vaf_wo = 0.0

                    vaf_with_up.append(vaf_w)
                    vaf_without_up.append(vaf_wo)
                    kld_with_up.append(float(kld_up[i].item()))
                    kld_without_up.append(float(kld_no[i].item()))

                processed += y_st.size(0)

        # Plot
        try:
            plot_te_ablation_results(kld_with_up, kld_without_up, vaf_with_up, vaf_without_up, out_dir)
            logger.info("UP ablation analysis complete.")
        except Exception as e:
            logger.warning(f"Failed to plot ablation analysis: {e}")

    def run_up_gain_sweep_analysis(self, test_loader, gains=None, num_samples=None, output_dir=None, selected_guids=None, model_created=False):
        """Sweep multiplicative gains on UP features and track TE (KLD) and VAF trends.

        Args:
            test_loader (DataLoader): Loader for normalized tensors.
            gains (List[float] | None): Multiplicative gains to apply to UP features. e.g., [0.0, 0.5, 1.0, 1.5, 2.0]
            num_samples (int | None): Limit the number of samples. None = all.

        Returns:
            None: Saves a plot of mean KLD and VAF vs gain.
        """
        out_dir = output_dir or self.test_results_dir
        logger.info("Starting UP gain sweep analysis...")
        
        if not model_created:
            self.create_model()

        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting gain sweep analysis.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        model = self.pytorch_model.to(device) if not model_created else self.pytorch_model
        model.eval()

        gains = gains if gains is not None else [0.0, 0.5, 1.0, 1.5, 2.0]

        # Accumulators per gain
        kld_sums = {g: 0.0 for g in gains}
        vaf_sums = {g: 0.0 for g in gains}
        counts = 0
        max_samples = num_samples  # None means no limit

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="UP Gain Sweep"):
                if max_samples is not None and counts >= max_samples:
                    break
                batch_size = batch.fhr_st.size(0)
                if max_samples is None:
                    take = batch_size
                else:
                    remaining = max_samples - counts
                    if remaining <= 0:
                        break
                    take = min(batch_size, remaining)

                # Filter by selected GUIDs if provided
                if selected_guids:
                    guids_batch = batch.guid if isinstance(batch.guid, (list, tuple)) else [str(g) for g in batch.guid]
                    idx_keep = [i for i in range(take) if guids_batch[i] in selected_guids]
                else:
                    idx_keep = list(range(take))

                if len(idx_keep) == 0:
                    continue

                y_st = batch.fhr_st[idx_keep].to(device)
                y_ph = batch.fhr_ph[idx_keep].to(device)
                x_ph_base = batch.fhr_up_ph[idx_keep].to(device)
                y_raw = batch.fhr[idx_keep].to(device)

                for g in gains:
                    x_scaled = x_ph_base * float(g)
                    out = model(y_st, y_ph, x_scaled)
                    mu_pr = out['mu_pr']
                    kld_tensor = model.measure_transfer_entropy(y_st, y_ph, x_scaled, reduce_mean=False)

                    # Per-sample KLD mean
                    kld_ps = kld_tensor.mean(dim=(1, 2))  # (B,)

                    # Per-sample VAF
                    for i in range(take):
                        gt = y_raw[i].detach().cpu().numpy()
                        pr = mu_pr[i].detach().cpu().numpy()
                        res = gt - pr
                        var_gt = np.var(gt)
                        if var_gt > 1e-12:
                            vaf = 1.0 - (np.var(res) / var_gt)
                            vaf = max(0.0, min(1.0, float(vaf)))
                        else:
                            vaf = 0.0

                        kld_sums[g] += float(kld_ps[i].item())
                        vaf_sums[g] += vaf

                counts += y_st.size(0)

        if counts == 0:
            logger.warning("No samples processed for gain sweep.")
            return

        gains_list = list(gains)
        kld_means = [kld_sums[g] / counts for g in gains_list]
        vaf_means = [vaf_sums[g] / counts for g in gains_list]

        try:
            plot_te_gain_sweep(gains_list, kld_means, vaf_means, out_dir)
            logger.info("UP gain sweep analysis complete.")
        except Exception as e:
            logger.warning(f"Failed to plot gain sweep analysis: {e}")


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    sklearn.utils.check_random_state(42)
    start = time.time()

    config_file_path = 'model/config.yaml'
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(project_root, config_file_path)

    config_file_path = os.path.normpath(config_file_path)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at the resolved path: {config_file_path}")
        logger.error("This might be because the file is missing or the path is incorrect.")
        logger.error(f"The path was set to 'model/config.yaml'.")
        logger.error("Please check your project structure and the config path.")
        sys.exit(1)

    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    # For PyTorch Lightning, DDP is handled by the Trainer.
    # We initialize rank and world_size for single-process dataloader creation.
    # Lightning will correctly handle distributed sampling when the DDP strategy is active.
    rank = 0
    world_size = 1

    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')

    def resolve_path(p):
        if not p or os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(project_root, p))

    if 'dataset_config' in config:
        if 'vae_train_datasets' in config['dataset_config']:
            config['dataset_config']['vae_train_datasets'] = [resolve_path(p) for p in config['dataset_config']['vae_train_datasets']]
        if 'vae_test_datasets' in config['dataset_config']:
            config['dataset_config']['vae_test_datasets'] = [resolve_path(p) for p in config['dataset_config']['vae_test_datasets']]
        if 'stat_path' in config['dataset_config']:
            config['dataset_config']['stat_path'] = resolve_path(config['dataset_config']['stat_path'])
    
    if 'seqvae_testing' in config and 'test_data_dir' in config['seqvae_testing']:
        config['seqvae_testing']['test_data_dir'] = resolve_path(config['seqvae_testing']['test_data_dir'])
    

    # Dataloader configuration for testing
    dataloader_config = config['dataset_config'].get('dataloader_config', {})
    dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
    num_workers = 0
    normalize_fields = dataloader_config.get('normalize_fields', None)
    stat_path = config['dataset_config'].get('stat_path')

    # SPEED OPTIMIZED: Enhanced test dataloader
    test_loader_seqvae = create_optimized_dataloader(
        hdf5_files=config['dataset_config']['vae_test_datasets'],
        batch_size=config['general_config']['batch_size']['test'],
        num_workers=0,  # Set to 0 to avoid pickle issues
        rank=0,
        world_size=1,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        pin_memory=True,  # Speed optimization
        **dataset_kwargs
    )

    # Initialize model for testing
    graph_model = SeqVAEGraphModelTest(config_file_path=config_file_path)
    graph_model.run_tests(test_loader_seqvae)

    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
