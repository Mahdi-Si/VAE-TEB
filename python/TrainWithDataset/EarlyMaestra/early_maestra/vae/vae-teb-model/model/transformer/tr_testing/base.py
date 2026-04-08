"""TransformerTestRunner — core test infrastructure for the transformer model.

Handles model loading, inference, intermediate extraction, and batch iteration.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from model.transformer.model import TransformerConfig, sample_anchors
from model.transformer.model.model import (
    CausalMultimodalTransformer,
    CausalTransformerLoss,
)


@dataclass
class TransformerTestRunner:
    """Core test runner for the Causal Multimodal Forecasting Transformer.

    Attributes:
        model: The CausalMultimodalTransformer instance.
        config: TransformerConfig with all hyperparameters.
        loss_fn: CausalTransformerLoss for computing loss components.
        device: Compute device.
        output_dir: Base output directory for results.
    """

    model: CausalMultimodalTransformer
    config: TransformerConfig
    loss_fn: CausalTransformerLoss
    device: torch.device
    output_dir: Path

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> "TransformerTestRunner":
        """Load a trained transformer model from a checkpoint.

        Uses ``train.graph_models_utils.load_checkpoint_strict`` per
        CLAUDE.md rules for checkpoint loading.

        Args:
            checkpoint_path: Path to ``.ckpt`` or ``.pt`` checkpoint file.
            output_dir: Directory where test results will be saved.
            device: Compute device (auto-detected if ``None``).
            config_overrides: Optional overrides for TransformerConfig fields.

        Returns:
            Configured TransformerTestRunner instance.
        """
        from train.graph_models_utils import load_checkpoint_strict

        checkpoint_path = Path(checkpoint_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str):
            device = torch.device(device)

        # Load checkpoint to extract config
        ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                          weights_only=False)

        # Extract TransformerConfig from checkpoint hparams
        config = cls._extract_config(ckpt, config_overrides)

        # Build model
        model = CausalMultimodalTransformer(config)
        loss_fn = CausalTransformerLoss(config)

        # Load weights
        load_checkpoint_strict(model, ckpt)
        model = model.to(device)
        model.eval()

        logger.info(
            f"Loaded transformer from {checkpoint_path.name} "
            f"({sum(p.numel() for p in model.parameters()):,} params) "
            f"on {device}"
        )

        return cls(
            model=model,
            config=config,
            loss_fn=loss_fn,
            device=device,
            output_dir=output_dir,
        )

    @staticmethod
    def _extract_config(
        ckpt: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> TransformerConfig:
        """Extract TransformerConfig from a checkpoint.

        Looks for config in ``hyper_parameters`` (Lightning) or
        ``config`` / ``transformer_config`` keys.

        Args:
            ckpt: Loaded checkpoint dictionary.
            overrides: Optional field overrides.

        Returns:
            TransformerConfig instance.
        """
        config_dict = {}

        # Try Lightning hyper_parameters
        hparams = ckpt.get("hyper_parameters", {})
        if "transformer_config" in hparams:
            cfg = hparams["transformer_config"]
            if isinstance(cfg, TransformerConfig):
                config_dict = {
                    f.name: getattr(cfg, f.name)
                    for f in cfg.__dataclass_fields__.values()
                }
            elif isinstance(cfg, dict):
                config_dict = cfg

        # Try top-level config
        if not config_dict and "config" in ckpt:
            cfg = ckpt["config"]
            if isinstance(cfg, dict) and "transformer" in cfg:
                config_dict = cfg["transformer"]
            elif isinstance(cfg, TransformerConfig):
                config_dict = {
                    f.name: getattr(cfg, f.name)
                    for f in cfg.__dataclass_fields__.values()
                }

        # Apply overrides
        if overrides:
            config_dict.update(overrides)

        # Filter to valid TransformerConfig fields
        valid_fields = {f.name for f in TransformerConfig.__dataclass_fields__.values()}
        config_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        if config_dict:
            return TransformerConfig(**config_dict)

        logger.warning(
            "Could not extract config from checkpoint; using defaults"
        )
        return TransformerConfig()

    # -----------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------

    @contextmanager
    def inference_mode(self):
        """Context manager: sets eval mode and ``torch.inference_mode``."""
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            yield
        if was_training:
            self.model.train()

    def iter_batches(
        self,
        loader: Any,
        max_samples: Optional[int] = None,
    ) -> Iterator[Any]:
        """Iterate batches, moving tensors to the runner's device.

        Moves ``fhr_st``, ``up_st``, ``fhr``, ``up`` to device.
        Keeps ``guid``, ``epoch``, ``cs_label``, ``bg_label``,
        ``time_from_labor_onset`` on CPU.

        Args:
            loader: A DataLoader yielding AttributeDict batches.
            max_samples: Stop after this many samples.

        Yields:
            Batch objects with signal tensors on ``self.device``.
        """
        count = 0
        for batch in loader:
            if max_samples is not None and count >= max_samples:
                break

            # Move signal tensors to device
            for key in ("fhr_st", "up_st", "fhr", "up"):
                val = getattr(batch, key, None)
                if val is not None and isinstance(val, Tensor):
                    setattr(batch, key, val.to(self.device))

            batch_size = batch.fhr_st.shape[0]
            if max_samples is not None and count + batch_size > max_samples:
                # Trim batch to fit max_samples
                trim = max_samples - count
                for key in ("fhr_st", "fhr_ph", "fhr_up_ph", "up_st",
                            "fhr", "up", "target", "weight",
                            "guid", "epoch", "cs_label",
                            "bg_label", "time_from_labor_onset"):
                    val = getattr(batch, key, None)
                    if val is not None:
                        if isinstance(val, Tensor):
                            setattr(batch, key, val[:trim])
                        elif isinstance(val, (list, tuple)):
                            setattr(batch, key, val[:trim])
                batch_size = trim

            yield batch
            count += batch_size

    # -----------------------------------------------------------------
    # Forward pass variants
    # -----------------------------------------------------------------

    def forward_with_anchors(
        self, Y: Tensor, U: Tensor,
    ) -> Dict[str, Any]:
        """Training-mode forward with eval-grid anchors.

        Uses ``sample_anchors(training=False)`` for a fixed grid every 15
        steps, giving ~16 deterministic anchors per window.

        Args:
            Y: FHR scattering features ``(B, T, d_F)``.
            U: UP scattering features ``(B, T, d_U)``.

        Returns:
            Model output dict with all 3 forecast heads, TE latent
            parameters, H_F, H_FU, and anchor_indices.
        """
        anchors = sample_anchors(Y, U, self.config, training=False)
        return self.model(Y, U, anchor_indices=anchors)

    def forward_for_embedding(
        self, Y: Tensor, U: Tensor,
    ) -> Tensor:
        """Inference-mode forward returning the window embedding.

        Args:
            Y: FHR scattering features ``(B, T, d_F)``.
            U: UP scattering features ``(B, T, d_U)``.

        Returns:
            Window embedding ``e_win`` of shape ``(B, output_dim)``.
        """
        out = self.model(Y, U)
        return out["e_win"]

    def extract_intermediates(
        self, Y: Tensor, U: Tensor,
    ) -> Dict[str, Tensor]:
        """Extract intermediate representations including gate activations.

        Runs a partial forward pass through stems, encoders, and fusion to
        extract gate activations (not returned by ``model.forward()``).

        Args:
            Y: FHR scattering features ``(B, T, d_F)``.
            U: UP scattering features ``(B, T, d_U)``.

        Returns:
            Dictionary with ``H_F``, ``H_U``, ``H_FU``, ``gate``,
            ``context`` tensors.
        """
        model = self.model
        F_out = model.fhr_stem(Y)       # (B, T, d)
        S_out = model.up_stem(U)        # (B, T, d)
        H_F = model.fhr_encoder(F_out)  # (B, T, d)
        H_U = model.up_encoder(S_out)   # (B, T, d)

        # Cross-attention and gating (first fusion layer for diagnostics)
        context = model.fusion.cross_attns[0](target=H_F, source=H_U)
        gate = torch.sigmoid(
            model.fusion.gate_projs[0](
                torch.cat([H_F, context], dim=-1)
            )
        )

        # Full fusion (all layers) + fused encoder
        H_tilde = model.fusion(H_F, H_U)
        H_FU = model.fused_encoder(H_tilde)

        return {
            "H_F": H_F,
            "H_U": H_U,
            "H_FU": H_FU,
            "gate": gate,
            "context": context,
        }

    def compute_losses(
        self,
        outputs: Dict[str, Any],
        Y: Tensor,
        beta: float = 0.0,
    ) -> Dict[str, Tensor]:
        """Compute all loss components for a batch.

        Args:
            outputs: Output dict from ``forward_with_anchors()``.
            Y: FHR scattering features ``(B, T, d_F)``.
            beta: KL weight (default 0 for test — just report raw L_kl).

        Returns:
            Dictionary with ``L_fus``, ``L_delta``, ``L_delta2``,
            ``L_spectral``, ``L_self``, ``L_te``, ``L_kl``,
            ``total_loss`` (without beta*L_kl).
        """
        return self.loss_fn(outputs, Y)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def ensure_dir(self, subdir: str) -> Path:
        """Create and return a subdirectory under ``output_dir``.

        Args:
            subdir: Subdirectory name.

        Returns:
            Path to the created directory.
        """
        path = self.output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
