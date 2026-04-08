"""Configuration dataclass for the Causal Multimodal Forecasting Transformer."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TransformerConfig:
    """All hyperparameters for the CausalMultimodalTransformer.

    Attributes:
        d_model: Backbone hidden dimension (width of all internal representations).
        d_f: Number of FHR scattering-transform input channels.
        d_u: Number of UP scattering-transform input channels.
        d_z_self: Deprecated compatibility field from the experimental v2 path.
            It is unused by the default document-aligned model.
        d_z_transfer: TE coupling latent dimension (the only latent in the
            default document-aligned model).
        n_heads: Number of attention heads in all multi-head attention layers.
        dropout: Dropout probability used throughout the model.
        stem_num_blocks: Number of residual causal-convolution blocks per stem.
        stem_kernels: Kernel sizes for each stem conv block.
        stem_dilations: Dilation factors for each stem conv block.
        stem_expansion: Pointwise expansion ratio in stem conv blocks (d -> expansion*d).
        fhr_encoder_layers: Number of causal transformer blocks in the FHR-only encoder.
        up_encoder_layers: Number of causal transformer blocks in the UP-only encoder.
        fused_encoder_layers: Number of causal transformer blocks in the fused encoder.
        fusion_layers: Number of stacked cross-attention fusion layers.
        ff_expansion: Feed-forward expansion ratio in transformer blocks.
        seq_len: Effective sequence length T (after trimming).
        ctx_len: Local context length L_ctx for anchor-based pooling.
        guard_gap: Guard gap g between anchor and prediction target start.
        horizons: Prediction horizons (in time steps).
        horizon_weights: Loss weights per horizon (longer = higher weight).
        num_anchors: Number of anchors K sampled per window during training.
        anchor_uniform_ratio: Fraction of anchors drawn uniformly (rest activity-biased).
        lambda_fus: Weight for the main fused forecasting loss.
        lambda_delta: Weight for the dynamics (first-order temporal difference) loss.
        lambda_delta2: Optional legacy extension weight. Defaults to 0.0.
        lambda_spectral: Optional legacy extension weight. Defaults to 0.0.
        lambda_self: Weight for the self-only baseline loss.
        lambda_te: Weight for the TE residual loss.
        huber_delta: Threshold delta for the Huber loss.
        free_bits: Optional legacy extension for the TE KL term. Defaults to 0.0.
        use_swiglu: Whether to use SwiGLU feed-forward networks instead of GELU.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
        gradient_checkpointing: Whether to use gradient checkpointing in encoder blocks.
    """

    # --- Dimensions ---
    d_model: int = 192
    d_f: int = 43
    d_u: int = 43
    d_z_self: int = 0
    d_z_transfer: int = 16
    n_heads: int = 4
    dropout: float = 0.1

    # --- Stems ---
    stem_num_blocks: int = 3
    stem_kernels: Tuple[int, ...] = (3, 5, 5)
    stem_dilations: Tuple[int, ...] = (1, 2, 4)
    stem_expansion: int = 2

    # --- Encoders ---
    fhr_encoder_layers: int = 4
    up_encoder_layers: int = 4
    fused_encoder_layers: int = 4
    fusion_layers: int = 1
    ff_expansion: int = 4

    # --- Sequence / Anchors / Prediction ---
    seq_len: int = 300
    ctx_len: int = 30
    guard_gap: int = 4
    horizons: Tuple[int, ...] = (8, 15, 30)
    horizon_weights: Tuple[float, ...] = (1.0, 1.5, 2.0)
    num_anchors: int = 4
    anchor_uniform_ratio: float = 0.5

    # --- Loss weights ---
    lambda_fus: float = 1.0
    lambda_delta: float = 0.5
    lambda_delta2: float = 0.0
    lambda_spectral: float = 0.0
    lambda_self: float = 0.25
    lambda_te: float = 0.25
    huber_delta: float = 1.0

    # --- Latent ---
    free_bits: float = 0.0

    # --- Architecture options ---
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    # --- Efficiency ---
    gradient_checkpointing: bool = False

    @property
    def d_z(self) -> int:
        """Backward-compatible alias for d_z_transfer."""
        return self.d_z_transfer

    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def max_horizon(self) -> int:
        """Largest prediction horizon."""
        return max(self.horizons)

    @property
    def valid_anchor_start(self) -> int:
        """First valid anchor index (inclusive).

        The context window for anchor a is ``[a - L + 1, ..., a]``.  For the
        earliest index ``a - L + 1`` to be >= 0 (0-based), we need
        ``a >= L - 1`` where ``L = ctx_len``.
        """
        return self.ctx_len - 1

    @property
    def valid_anchor_end(self) -> int:
        """Last valid anchor index (inclusive).

        The target block for anchor a spans indices [a+g+1, a+g+h].  The last
        index a+g+h must be <= T-1 (0-based), so a <= T - g - h - 1.
        """
        return self.seq_len - self.guard_gap - self.max_horizon - 1

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert len(self.stem_kernels) == self.stem_num_blocks, (
            f"stem_kernels length ({len(self.stem_kernels)}) must match "
            f"stem_num_blocks ({self.stem_num_blocks})"
        )
        assert len(self.stem_dilations) == self.stem_num_blocks, (
            f"stem_dilations length ({len(self.stem_dilations)}) must match "
            f"stem_num_blocks ({self.stem_num_blocks})"
        )
        assert len(self.horizons) == len(self.horizon_weights), (
            f"horizons ({len(self.horizons)}) and horizon_weights "
            f"({len(self.horizon_weights)}) must have the same length"
        )
        valid_range = self.valid_anchor_end - self.valid_anchor_start + 1
        assert valid_range > 0, (
            f"No valid anchors: start={self.valid_anchor_start}, "
            f"end={self.valid_anchor_end}. Check seq_len, ctx_len, "
            f"guard_gap, and horizons."
        )
        assert self.num_anchors <= valid_range, (
            f"num_anchors ({self.num_anchors}) exceeds valid anchor range "
            f"({valid_range}). Reduce num_anchors or increase seq_len."
        )
        assert 0.0 <= self.anchor_uniform_ratio <= 1.0, (
            f"anchor_uniform_ratio ({self.anchor_uniform_ratio}) must be in [0, 1]."
        )
        assert self.d_z_self >= 0, (
            f"d_z_self ({self.d_z_self}) must be non-negative."
        )
        assert self.d_z_transfer > 0, (
            f"d_z_transfer ({self.d_z_transfer}) must be positive."
        )
        assert self.free_bits >= 0.0, (
            f"free_bits ({self.free_bits}) must be non-negative."
        )
        assert self.fusion_layers >= 1, (
            f"fusion_layers ({self.fusion_layers}) must be >= 1."
        )
