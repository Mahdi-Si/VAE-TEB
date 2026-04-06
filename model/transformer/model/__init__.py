"""Causal Multimodal Forecasting Transformer for FHR/UP self-supervised pretraining.

Public API:
    - TransformerConfig: Dataclass with all model hyperparameters.
    - CausalMultimodalTransformer: Main nn.Module.
    - CausalTransformerLoss: Loss computation module.
    - sample_anchors: Utility for anchor index sampling.
    - validate_anchor_indices: Validate caller-supplied anchors before forward().
    - SelfLatentModule: Intrinsic FHR latent module (v2).
"""

from .config import TransformerConfig
from .heads import SelfLatentModule
from .model import (
    CausalMultimodalTransformer,
    CausalTransformerLoss,
    sample_anchors,
    validate_anchor_indices,
)

__all__ = [
    "TransformerConfig",
    "CausalMultimodalTransformer",
    "CausalTransformerLoss",
    "sample_anchors",
    "validate_anchor_indices",
    "SelfLatentModule",
]
