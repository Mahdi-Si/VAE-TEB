"""
Base module for the VAE-TEB testing pipeline.

This module provides the TestRunner class which handles all common boilerplate
for model testing: device management, model loading, batch iteration, and
output directory management.

Example:
    >>> from testing.base import TestRunner
    >>> runner = TestRunner.from_checkpoint("model.ckpt", output_dir="results")
    >>> with runner.inference_mode():
    ...     for batch in runner.iter_batches(loader, max_samples=100):
    ...         outputs = runner.model(y_st=batch.fhr_st, ...)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch
import torch.nn as nn

from train.graph_models_utils import load_checkpoint_strict
from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae


@dataclass
class TestRunner:
    """
    Minimal test harness with zero boilerplate.

    Handles device management, model setup, batch iteration, and output
    directory organization. Designed to be composed with analysis functions
    rather than inherited from.

    Attributes:
        model: The PyTorch model to test (must be on correct device).
        device: The torch device for inference (cuda or cpu).
        output_dir: Base directory for saving test results.
        warmup_steps: Number of initial timesteps to mask (default 30).
        decimation_factor: Temporal decimation factor (default 16).

    Example:
        >>> runner = TestRunner.from_checkpoint("checkpoint.ckpt", "results/")
        >>> with runner.inference_mode():
        ...     for batch in runner.iter_batches(test_loader):
        ...         outputs = runner.model(y_st=batch.fhr_st, ...)
    """

    model: nn.Module
    device: torch.device
    output_dir: Path
    warmup_steps: int = 30
    decimation_factor: int = 16

    # Private field to track if model is in eval mode
    _in_inference: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Ensure output_dir is a Path object."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[torch.device] = None,
        **model_kwargs: Any,
    ) -> "TestRunner":
        """
        Create a TestRunner by loading a model from a checkpoint file.

        Args:
            checkpoint_path: Path to the model checkpoint file (.ckpt or .pt).
            output_dir: Directory for saving test results.
            device: Torch device to use. If None, auto-detects (cuda:0 or cpu).
            **model_kwargs: Additional arguments passed to SeqVae constructor.

        Returns:
            TestRunner: Configured test runner with loaded model.

        Example:
            >>> runner = TestRunner.from_checkpoint(
            ...     "checkpoints/best_model.ckpt",
            ...     output_dir="test_results/",
            ...     device=torch.device("cuda:0")
            ... )
        """
        # Auto-detect device if not provided
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create model with architecture parameters
        # Note: model_kwargs should match the checkpoint's architecture
        # (e.g., latent_dim, hidden_dim, etc.)
        model = SeqVae()

        # Load checkpoint using the robust loader that handles:
        # - PyTorch Lightning checkpoints
        # - torch.compile() wrapped models
        # - Various state_dict key prefixes
        loaded = load_checkpoint_strict(model, checkpoint_path)

        if loaded is None:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}'. "
                f"Check logs for details. Ensure model_kwargs match the checkpoint architecture."
            )

        model = model.to(device)

        # Extract warmup and decimation from loaded model
        warmup = int(getattr(model, "warmup_period", 30))
        decimation = int(getattr(model, "decimation_factor", 16))

        return cls(
            model=model,
            device=device,
            output_dir=Path(output_dir),
            warmup_steps=warmup,
            decimation_factor=decimation,
        )

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        output_subdir: str = "test_results",
    ) -> "TestRunner":
        """
        Create a TestRunner from an existing trainer instance.

        This is useful when you already have a trainer with a loaded model
        and want to run tests without reloading.

        Args:
            trainer: A GraphModelVaeTebSmallTrainer instance with pytorch_model.
            output_subdir: Subdirectory name under trainer's output for results.

        Returns:
            TestRunner: Configured test runner using trainer's model.

        Raises:
            ValueError: If trainer.pytorch_model is None.

        Example:
            >>> trainer = GraphModelVaeTebSmallTrainer("config.yaml")
            >>> trainer.create_model()
            >>> runner = TestRunner.from_trainer(trainer)
        """
        if trainer.pytorch_model is None:
            raise ValueError("Trainer's pytorch_model is None. Call create_model() first.")

        # Determine device from trainer's cuda_devices
        cuda_devices = getattr(trainer, "cuda_devices", [])
        if cuda_devices and torch.cuda.is_available():
            device = torch.device(f"cuda:{cuda_devices[0]}")
        else:
            device = torch.device("cpu")

        # Get output directory from trainer
        output_dir = Path(getattr(trainer, "test_results_dir", "test_results")) / output_subdir

        # Move model to device
        model = trainer.pytorch_model.to(device)

        # Extract warmup and decimation from model if available
        warmup = int(getattr(model, "warmup_period", 30))
        decimation = int(getattr(model, "decimation_factor", 16))

        return cls(
            model=model,
            device=device,
            output_dir=output_dir,
            warmup_steps=warmup,
            decimation_factor=decimation,
        )

    @contextmanager
    def inference_mode(self):
        """
        Context manager for model evaluation with inference optimizations.

        Sets model to eval mode and enables torch.inference_mode() for
        faster inference without gradient tracking.

        Yields:
            None

        Example:
            >>> with runner.inference_mode():
            ...     outputs = runner.model(y_st=batch.fhr_st, ...)
        """
        # Save original training state
        was_training = self.model.training
        self.model.eval()
        self._in_inference = True

        try:
            with torch.inference_mode():
                yield
        finally:
            # Restore original state (though usually we leave in eval)
            self._in_inference = False
            if was_training:
                self.model.train()

    def iter_batches(
        self,
        loader: Any,
        max_samples: Optional[int] = None,
    ) -> Iterator[Any]:
        """
        Iterate over batches with tensors moved to the correct device.

        Automatically moves fhr_st, fhr_ph, fhr_up_ph, and fhr tensors to
        self.device. Metadata (guid, epoch) is kept on CPU.

        Args:
            loader: PyTorch DataLoader yielding batch objects.
            max_samples: Maximum number of samples to yield. If None, yields all.

        Yields:
            batch: Batch object with tensors on self.device.

        Example:
            >>> for batch in runner.iter_batches(test_loader, max_samples=100):
            ...     y_st = batch.fhr_st  # Already on runner.device
        """
        processed = 0

        for batch in loader:
            # Check sample limit
            if max_samples is not None and processed >= max_samples:
                break

            # Move main tensors to device (in-place modification of batch)
            batch.fhr_st = batch.fhr_st.to(self.device)
            batch.fhr_ph = batch.fhr_ph.to(self.device)
            batch.fhr_up_ph = batch.fhr_up_ph.to(self.device)
            batch.fhr = batch.fhr.to(self.device)

            # Move UP signal if present
            if hasattr(batch, "up") and batch.up is not None:
                batch.up = batch.up.to(self.device)

            yield batch

            # Track samples processed (batch size)
            batch_size = batch.fhr_st.size(0)
            processed += batch_size

    def ensure_dir(self, subdir: str) -> Path:
        """
        Create and return an output subdirectory.

        Args:
            subdir: Name of subdirectory under self.output_dir.

        Returns:
            Path: The created (or existing) subdirectory path.

        Example:
            >>> histograms_dir = runner.ensure_dir("histograms")
            >>> fig.savefig(histograms_dir / "vaf_histogram.pdf")
        """
        path = self.output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def forward(
        self,
        batch: Any,
        compute_loss: bool = False,
        beta: float = 1.0,
    ) -> dict:
        """
        Run forward pass on a batch and return outputs.

        Convenience method that extracts tensors from batch and calls model.

        Args:
            batch: Batch object with fhr_st, fhr_ph, fhr_up_ph, fhr attributes.
            compute_loss: If True, also compute and return loss dict.
            beta: KLD weight for loss computation (only used if compute_loss=True).

        Returns:
            dict: Model outputs including 'z', 'mu_pr', 'logvar_pr', etc.
                  If compute_loss=True, also includes 'loss_dict'.

        Example:
            >>> outputs = runner.forward(batch, compute_loss=True)
            >>> latent = outputs['z']
            >>> loss = outputs['loss_dict']['total_loss']
        """
        outputs = self.model(
            y_st=batch.fhr_st,
            y_ph=batch.fhr_ph,
            x_ph=batch.fhr_up_ph,
        )

        if compute_loss:
            loss_dict = self.model.compute_loss(
                forward_outputs=outputs,
                y_st=batch.fhr_st,
                y_ph=batch.fhr_ph,
                y_raw=batch.fhr,
                beta=beta,
            )
            outputs["loss_dict"] = loss_dict

        return outputs
