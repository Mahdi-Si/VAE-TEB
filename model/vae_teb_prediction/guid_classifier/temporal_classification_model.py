"""Temporal VAE classifier for GUID-level sequence classification.

Implements a two-level architecture: a frozen VAE encoder produces per-segment
latent representations, which are then processed by a segment-level encoder
(mean-pool, LSTM, or CNN) and a temporal LSTM across the segment sequence.
Each segment receives a context-informed binary prediction (healthy vs unhealthy).

The segment encoder LSTM can optionally persist its hidden state across
segments (``persist_segment_state=True``), with time-decay gating to account
for variable inter-segment gaps.  This allows within-segment temporal
patterns (e.g. decelerations at segment boundaries) to inform the encoding
of subsequent segments.

The model takes batch dicts from ``sequence_collate_fn`` and produces
per-segment predictions of shape ``(B, S_max, num_classes)``.

Example::

    from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
    from train.graph_models_utils import load_checkpoint_strict

    vae = SeqVae()
    load_checkpoint_strict(vae, checkpoint="/path/to/vae.ckpt")

    model = TemporalVaeClassifier(
        vae_model=vae,
        segment_encoder_type="lstm",
        d_seg=64,
        temporal_lstm_hidden=128,
        persist_segment_state=True,
        tlo_embed_dim=8,
        delta_t_embed_dim=8,
    )

    outputs = model(batch)  # batch from sequence_collate_fn
    loss_dict = model.compute_loss(outputs, batch)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from loguru import logger

from model.vae_teb_prediction.prediction_classification_model import (
    FocalBCEWithLogitsLoss,
    map_to_hierarchical_labels,
)


# ------------------------------------------------------------------ #
#  Learned temporal feature embeddings                                  #
# ------------------------------------------------------------------ #


class TemporalTLOEmbedding(nn.Module):
    """Learned embedding for Time from Labour Onset (per-segment).

    Converts scalar TLO (seconds) to a learned vector via an MLP.
    NaN values (unavailable TLO) are replaced by a learned
    ``missing_embedding`` parameter so the model can distinguish
    "unknown TLO" from any real value.

    Follows the same pattern as
    :class:`~model.vae_teb_prediction.prediction_classification_model.TLOEmbedding`
    but operates on ``(B, S_max)`` temporal sequences instead of
    ``(B,)`` scalars.

    Args:
        embed_dim: Dimensionality of the output embedding.
        dropout: Dropout probability inside the MLP.
    """

    def __init__(self, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.missing_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, tlo_seconds: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            tlo_seconds: TLO in seconds ``(B, S_max)``.  May contain ``NaN``
                when TLO is unavailable.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Embedding tensor ``(B, S_max, embed_dim)``.  Padded positions
            are zeroed.  Guaranteed NaN-free.
        """
        is_valid = ~torch.isnan(tlo_seconds) & mask  # (B, S_max)
        tlo_hours = tlo_seconds / 3600.0
        # Replace NaN with 0 for MLP input (NaN positions get missing_embedding).
        tlo_hours = torch.where(
            torch.isnan(tlo_hours),
            torch.zeros_like(tlo_hours),
            tlo_hours,
        )

        # MLP: (B, S_max, 1) → (B, S_max, embed_dim)
        tlo_embed = self.mlp(tlo_hours.unsqueeze(-1))

        # Replace NaN positions with learned missing_embedding.
        missing_mask = (~is_valid).unsqueeze(-1)  # (B, S_max, 1)
        tlo_embed = torch.where(
            missing_mask,
            self.missing_embedding.unsqueeze(0).unsqueeze(0).expand_as(tlo_embed),
            tlo_embed,
        )

        # Zero out padded positions.
        tlo_embed = tlo_embed * mask.unsqueeze(-1).float()
        return tlo_embed


class DeltaTEmbedding(nn.Module):
    """Learned embedding for inter-segment time gaps.

    Converts scalar delta_t (seconds) to a learned vector via an MLP,
    replacing the raw ``[hours, log]`` 2-dim concatenation with a
    richer learned representation.

    Args:
        embed_dim: Dimensionality of the output embedding.
        dropout: Dropout probability inside the MLP.
    """

    def __init__(self, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, delta_t: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            delta_t: Inter-segment gap in seconds ``(B, S_max)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Embedding tensor ``(B, S_max, embed_dim)``.  Padded positions
            are zeroed.
        """
        delta_t_hours = delta_t / 3600.0
        dt_embed = self.mlp(delta_t_hours.unsqueeze(-1))  # (B, S_max, embed_dim)
        return dt_embed * mask.unsqueeze(-1).float()


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block for the classifier head.

    Applies LayerNorm before the transformation (pre-norm pattern), then a
    bottleneck MLP, and adds the result back to the input (residual
    connection).  This stabilises gradient flow in deeper classifier heads.

    Architecture::

        out = x + Dropout(Linear_up(GELU(Linear_down(LayerNorm(x)))))

    where ``Linear_down`` projects from ``hidden_dim`` to ``bottleneck_dim``
    and ``Linear_up`` projects back.

    Args:
        hidden_dim: Input and output dimension (must match for the residual
            addition).
        bottleneck_dim: Internal bottleneck dimension of the MLP.
        dropout: Dropout probability applied after the expansion linear
            layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., hidden_dim)``.

        Returns:
            Output tensor of the same shape as *x*.
        """
        return x + self.mlp(self.norm(x))


class SegmentAttentionPooling(nn.Module):
    """Learned attention pooling over within-segment LSTM hidden states.

    Instead of using only the last hidden state of the segment LSTM (which
    has a recency bias toward the final timesteps), this module computes a
    learned attention-weighted sum over ALL hidden states.  This allows the
    model to focus on the most informative timesteps within each 20-minute
    segment (e.g., deceleration events, variability changes).

    Architecture::

        score_t = w^T tanh(W h_t + b)
        alpha   = softmax(score, dim=T)
        v       = sum(alpha * h, dim=T)

    Args:
        hidden_dim: Dimension of the LSTM hidden states (d_seg).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute attention-weighted sum of hidden states.

        Args:
            hidden_states: LSTM hidden states ``(*, T, H)`` where ``T`` is
                the number of timesteps (300) and ``H`` is the hidden dim.

        Returns:
            Tuple of ``(pooled, alpha)`` where ``pooled`` has shape ``(*, H)``
            and ``alpha`` has shape ``(*, T)`` (attention weights for
            interpretability).
        """
        scores = self.attention_net(hidden_states)  # (*, T, 1)
        alpha = F.softmax(scores, dim=-2)           # (*, T, 1)
        pooled = (alpha * hidden_states).sum(dim=-2)  # (*, H)
        return pooled, alpha.squeeze(-1)


class TemporalCellAttention(nn.Module):
    """ATTAIN-style time-decayed attention over past temporal LSTM cell states.

    At each segment position $j$, computes scaled dot-product attention over
    all cell states $c_0, c_1, \\ldots, c_j$ from the temporal LSTM, modulated
    by an exponential time-decay that down-weights temporally distant segments.
    This allows the classifier to directly retrieve relevant past states
    without relying solely on the temporal LSTM's compressed hidden state.

    Based on ATTAIN (Zhang et al., IJCAI 2019).

    Architecture::

        q_j = W_q h_j,   k_i = W_k c_i,   v_i = W_v c_i
        score_{j,i} = (q_j^T k_i) / sqrt(A) + log(exp(-gamma * dt_{j->i}))
        Causal mask: score_{j,i} = -inf  for i > j
        alpha = softmax(score, dim=keys)
        context_j = sum_i(alpha_{j,i} * v_i)
        output_j  = W_o [h_j || context_j]

    Args:
        hidden_dim: Dimension of temporal LSTM hidden/cell states.
        attn_dim: Projection dimension for queries and keys.
        dropout: Dropout probability on attention weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gamma_attn_log = nn.Parameter(torch.tensor(-2.0))
        self.scale = attn_dim ** -0.5
        self.attn_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        h: Tensor,
        c_all: Tensor,
        delta_t: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute time-decayed causal attention over past cell states.

        Args:
            h: Temporal LSTM hidden states ``(B, S, H)``.
            c_all: Temporal LSTM cell states ``(B, S, H)``.
            delta_t: Inter-segment time gaps in seconds ``(B, S)``.
            mask: Boolean validity mask ``(B, S)``.

        Returns:
            Tuple of ``(fused, alpha)`` where ``fused`` has shape
            ``(B, S, H)`` (h augmented with attended context) and ``alpha``
            has shape ``(B, S, S)`` (attention weights).
        """
        B, S, H = h.shape
        device = h.device

        q = self.query_proj(h)       # (B, S, A)
        k = self.key_proj(c_all)     # (B, S, A)
        v = self.value_proj(c_all)   # (B, S, H)

        # Content-based attention scores.
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, S, S)

        # Causal mask: position j can attend to 0..j (inclusive, not future).
        causal_mask = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        # Time-decay: exp(-gamma * elapsed_hours) on attention logits.
        gamma = F.softplus(self.gamma_attn_log)
        cum_time = torch.cumsum(delta_t, dim=1)  # (B, S)
        # time_diff[b, j, k] = elapsed time from segment k to segment j.
        time_diff = cum_time.unsqueeze(1) - cum_time.unsqueeze(2)  # (B, S, S)
        time_decay = torch.exp(-gamma * time_diff.clamp(min=0) / 3600.0)
        scores = scores + torch.log(time_decay.clamp(min=1e-8))

        # Mask padded key positions.
        key_pad_mask = ~mask.unsqueeze(1).expand(B, S, S)  # (B, S, S)
        scores = scores.masked_fill(key_pad_mask, float("-inf"))

        alpha = F.softmax(scores, dim=-1)  # (B, S, S)
        # Guard against NaN from all-masked rows.
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
        alpha = self.attn_dropout(alpha)

        context = torch.bmm(alpha, v)  # (B, S, H)

        # Fuse h with attended context via concatenation + projection.
        fused = self.output_proj(torch.cat([h, context], dim=-1))  # (B, S, H)
        fused = fused * mask.unsqueeze(-1).float()

        return fused, alpha


class TemporalVaeClassifier(nn.Module):
    """Two-level temporal classifier: frozen VAE + segment encoder + temporal LSTM.

    Level 1 (segment): Frozen VAE encoder → segment encoder → v_j (D_seg).
    Level 2 (temporal): Temporal LSTM across segments → per-segment predictions.

    Args:
        vae_model: Pre-trained ``SeqVae`` instance (frozen during training).
        segment_encoder_type: How to reduce within-segment temporal dim.
            ``'mean_pool'`` | ``'lstm'`` | ``'cnn'``.  Default ``'mean_pool'``.
        d_seg: Segment representation dimension.  Forced to VAE latent dim (16)
            when ``segment_encoder_type='mean_pool'``.  Default 64.
        temporal_lstm_hidden: Temporal LSTM hidden dimension.  Default 128.
        temporal_lstm_layers: Number of temporal LSTM layers.  Default 2.
        temporal_lstm_dropout: Dropout between temporal LSTM layers.  Default 0.1.
        gap_encoding: How to encode delta_t gaps.
            ``'concat'`` | ``'time_decay'`` | ``'both'``.  Default ``'concat'``.
        position_embed_dim: Dimension of learned position embedding.  0 disables.
            Default 0 (enabled in Sprint 6).
        max_position_index: Max grid slot index for the embedding table.
            Default 40.
        tlo_enabled: Whether to use ``time_from_labor_onset`` features.
            Default ``False`` (enabled in Sprint 6).
        tlo_embed_dim: Dimension of learned TLO embedding.  0 uses raw
            ``[hours, flag]`` (2 dims).  Default 0.
        tlo_dropout: Dropout for TLO embedding MLP.  Default 0.1.
        delta_t_embed_dim: Dimension of learned delta_t embedding.  0 uses raw
            ``[hours, log]`` (2 dims).  Default 0.
        delta_t_dropout: Dropout for delta_t embedding MLP.  Default 0.1.
        persist_segment_state: If ``True`` and ``segment_encoder_type='lstm'``,
            carry LSTM hidden state across segments (with optional time-decay).
            Default ``False``.
        segment_state_decay: If ``True`` (and ``persist_segment_state=True``),
            apply time-decay gating to the segment LSTM state between segments.
            Default ``True``.
        num_classes: Number of output classes.  Default 2.
        temporal_lstm_residual: If ``True``, add a skip connection around the
            temporal LSTM: project input to hidden dim and add to LSTM output.
            Default ``False``.
        classifier_dropout: Dropout probability in the classifier head.
            Default 0.1.
        mlp_multiplier: MLP hidden expansion factor (legacy head only).
            Default 2.0.
        classifier_num_residual_blocks: Number of pre-norm
            :class:`ResidualMLPBlock` layers before the final linear
            projection.  0 uses the legacy shallow head.  Default 0.
        classifier_bottleneck_dim: Bottleneck dimension inside each
            :class:`ResidualMLPBlock`.  Default 64.
        output_dropout: Dropout probability applied to the temporal LSTM
            output before the classifier head.  0 disables.  Default 0.0.
        class_weights: Optional per-class loss weights for CE loss.
        vae_chunk_size: Segments per VAE encoding chunk to limit peak GPU
            memory.  Default 32.
        use_posterior: If ``True`` use posterior mean ``mu_post``; otherwise use
            prior mean ``mu_prior``.  Default ``True``.
        freeze_vae: Whether to freeze all VAE parameters.  Default ``True``.
        cnn_kernel: Kernel size for CNN segment encoder.  Default 7.
        debug: If ``True``, add NaN assertion guards at critical points in the
            forward pass.  Default ``False``.
    """

    # VAE latent dimension (fixed by pre-trained VAE architecture).
    _VAE_LATENT_DIM = 16

    def __init__(
        self,
        vae_model: nn.Module,
        *,
        segment_encoder_type: str = "mean_pool",
        d_seg: int = 64,
        temporal_lstm_hidden: int = 128,
        temporal_lstm_layers: int = 2,
        temporal_lstm_dropout: float = 0.1,
        gap_encoding: str = "concat",
        position_embed_dim: int = 0,
        max_position_index: int = 40,
        tlo_enabled: bool = False,
        tlo_embed_dim: int = 0,
        tlo_dropout: float = 0.1,
        delta_t_embed_dim: int = 0,
        delta_t_dropout: float = 0.1,
        persist_segment_state: bool = False,
        segment_state_decay: bool = True,
        temporal_lstm_residual: bool = False,
        num_classes: int = 2,
        classifier_dropout: float = 0.1,
        mlp_multiplier: float = 2.0,
        classifier_num_residual_blocks: int = 0,
        classifier_bottleneck_dim: int = 64,
        output_dropout: float = 0.0,
        class_weights: Optional[Sequence[float]] = None,
        vae_chunk_size: int = 32,
        use_posterior: bool = True,
        freeze_vae: bool = True,
        cnn_kernel: int = 7,
        segment_attention_pool: bool = False,
        temporal_attention: bool = False,
        temporal_attention_dim: int = 64,
        temporal_attention_dropout: float = 0.1,
        debug: bool = False,
        enriched_features: bool = False,
        label_mode: str = "binary",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        bit_weights: Optional[Sequence[float]] = None,
        augment_posterior_sample: bool = False,
        augment_noise_scale: float = 0.5,
        augment_temporal_jitter: int = 0,
    ) -> None:
        super().__init__()

        # -- Store config -------------------------------------------------- #
        self.segment_encoder_type = segment_encoder_type
        self.gap_encoding = gap_encoding
        self.position_embed_dim = position_embed_dim
        self.tlo_enabled = tlo_enabled
        self.tlo_embed_dim = tlo_embed_dim
        self.delta_t_embed_dim = delta_t_embed_dim
        self.persist_segment_state = persist_segment_state
        self.segment_state_decay = segment_state_decay
        self.temporal_lstm_residual = temporal_lstm_residual
        self.num_classes = num_classes
        self.vae_chunk_size = vae_chunk_size
        self.use_posterior = use_posterior
        self.freeze_vae = freeze_vae
        self.debug = debug
        self.enriched_features = enriched_features
        self.label_mode = label_mode
        self.augment_posterior_sample = augment_posterior_sample
        self.augment_noise_scale = augment_noise_scale
        self.augment_temporal_jitter = augment_temporal_jitter

        # Feature dimension: 64 when enriched (mu_post + logvar_post + residual + kld), else 16
        self._feature_dim = self._VAE_LATENT_DIM * 4 if enriched_features else self._VAE_LATENT_DIM

        # Focal loss for hierarchical mode
        if label_mode == "hierarchical":
            bw = torch.as_tensor(bit_weights, dtype=torch.float32) if bit_weights else None
            self.focal_loss_fn = FocalBCEWithLogitsLoss(
                gamma=focal_gamma, alpha=bw, label_smoothing=label_smoothing,
            )

        # -- VAE model (frozen) -------------------------------------------- #
        self.vae_model = vae_model
        if self.freeze_vae:
            self.vae_model.eval()
            for param in self.vae_model.parameters():
                param.requires_grad = False

        # -- Segment encoder ----------------------------------------------- #
        # Use _feature_dim (64 when enriched, 16 otherwise) as input size
        if segment_encoder_type == "mean_pool":
            self.d_seg = self._feature_dim
        elif segment_encoder_type == "lstm":
            self.d_seg = d_seg
            self.segment_lstm = nn.LSTM(
                input_size=self._feature_dim,
                hidden_size=d_seg,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            # Segment LSTM state persistence with time-decay.
            if persist_segment_state and segment_state_decay:
                self.gamma_seg_log = nn.Parameter(
                    torch.full((d_seg,), -2.0)
                )
        elif segment_encoder_type == "cnn":
            self.d_seg = d_seg
            self._cnn_kernel = cnn_kernel
            self.segment_cnn = nn.Sequential(
                nn.Conv1d(self._feature_dim, d_seg, kernel_size=cnn_kernel, padding=0),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            raise ValueError(
                f"Unknown segment_encoder_type '{segment_encoder_type}'. "
                "Expected 'mean_pool', 'lstm', or 'cnn'."
            )

        # -- Segment attention pooling ------------------------------------- #
        self.segment_attention_pool = (
            segment_attention_pool and segment_encoder_type == "lstm"
        )
        if self.segment_attention_pool:
            self.seg_attn_pool = SegmentAttentionPooling(self.d_seg)

        # -- Learned temporal feature embeddings --------------------------- #
        if delta_t_embed_dim > 0:
            self.delta_t_embedding = DeltaTEmbedding(
                embed_dim=delta_t_embed_dim,
                dropout=delta_t_dropout,
            )

        if tlo_enabled and tlo_embed_dim > 0:
            self.tlo_embedding = TemporalTLOEmbedding(
                embed_dim=tlo_embed_dim,
                dropout=tlo_dropout,
            )

        # -- Temporal input dimension (computed dynamically) --------------- #
        temporal_input_dim = self.d_seg

        # delta_t features: learned embedding or raw [hours, log].
        if gap_encoding in ("concat", "both"):
            if delta_t_embed_dim > 0:
                temporal_input_dim += delta_t_embed_dim
            else:
                temporal_input_dim += 2

        # Position embedding.
        if position_embed_dim > 0:
            self.position_embedding = nn.Embedding(
                num_embeddings=max_position_index,
                embedding_dim=position_embed_dim,
            )
            temporal_input_dim += position_embed_dim

        # Time from labor onset (TLO): learned embedding or raw [hours, flag].
        if tlo_enabled:
            if tlo_embed_dim > 0:
                temporal_input_dim += tlo_embed_dim
            else:
                temporal_input_dim += 2

        self.temporal_input_dim = temporal_input_dim

        # -- LayerNorm on temporal input ----------------------------------- #
        self.temporal_input_norm = nn.LayerNorm(temporal_input_dim)

        # -- Time-decay gating (T-LSTM) ------------------------------------ #
        self._temporal_lstm_hidden = temporal_lstm_hidden
        self._temporal_lstm_layers = temporal_lstm_layers

        if gap_encoding in ("time_decay", "both"):
            # Learned per-dimension decay rate.  softplus ensures γ > 0.
            # Init -2.0 → softplus(-2.0) ≈ 0.127 → moderate decay.
            self.gamma_log = nn.Parameter(
                torch.full((temporal_lstm_hidden,), -2.0)
            )
            # Use LSTMCell for per-step gating.  One cell per layer.
            self.temporal_lstm_cells = nn.ModuleList()
            for layer_idx in range(temporal_lstm_layers):
                cell_input = temporal_input_dim if layer_idx == 0 else temporal_lstm_hidden
                self.temporal_lstm_cells.append(
                    nn.LSTMCell(cell_input, temporal_lstm_hidden)
                )
            if temporal_lstm_layers > 1 and temporal_lstm_dropout > 0:
                self._td_dropout = nn.Dropout(temporal_lstm_dropout)
            else:
                self._td_dropout = None

        # -- Standard temporal LSTM (only when concat mode needs it) -------- #
        if gap_encoding == "concat":
            self.temporal_lstm = nn.LSTM(
                input_size=temporal_input_dim,
                hidden_size=temporal_lstm_hidden,
                num_layers=temporal_lstm_layers,
                batch_first=True,
                bidirectional=False,
                dropout=temporal_lstm_dropout if temporal_lstm_layers > 1 else 0.0,
            )

        # -- Residual projection around temporal LSTM ---------------------- #
        if temporal_lstm_residual:
            self.temporal_residual_proj = nn.Linear(
                temporal_input_dim, temporal_lstm_hidden, bias=False,
            )

        # -- Output dropout on temporal LSTM output ----------------------- #
        self.output_drop = (
            nn.Dropout(output_dropout) if output_dropout > 0.0
            else nn.Identity()
        )

        # -- Classifier head ----------------------------------------------- #
        self._classifier_num_residual_blocks = classifier_num_residual_blocks
        if classifier_num_residual_blocks > 0:
            # Deep head: pre-norm residual blocks + final projection.
            self.residual_blocks = nn.ModuleList([
                ResidualMLPBlock(
                    hidden_dim=temporal_lstm_hidden,
                    bottleneck_dim=classifier_bottleneck_dim,
                    dropout=classifier_dropout,
                )
                for _ in range(classifier_num_residual_blocks)
            ])
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(temporal_lstm_hidden),
                nn.Linear(temporal_lstm_hidden, num_classes),
            )
        else:
            # Legacy shallow head (backward compatible).
            self.residual_blocks = nn.ModuleList()
            mlp_hidden = int(temporal_lstm_hidden * mlp_multiplier)
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(temporal_lstm_hidden),
                nn.Linear(temporal_lstm_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(mlp_hidden, num_classes),
            )

        # -- ATTAIN-style temporal attention -------------------------------- #
        self.temporal_attention_enabled = temporal_attention
        if temporal_attention:
            if gap_encoding not in ("time_decay", "both"):
                raise ValueError(
                    "temporal_attention requires gap_encoding='time_decay' or "
                    "'both' (custom LSTMCell loop needed for cell states)"
                )
            self.temporal_cell_attention = TemporalCellAttention(
                hidden_dim=temporal_lstm_hidden,
                attn_dim=temporal_attention_dim,
                dropout=temporal_attention_dropout,
            )

        # -- Class weights for loss ---------------------------------------- #
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        logger.info(
            "TemporalVaeClassifier created: encoder={}, d_seg={}, "
            "temporal_input_dim={}, lstm_hidden={}, lstm_layers={}, "
            "lstm_residual={}, gap={}, pos_dim={}, tlo={}, tlo_embed={}, "
            "dt_embed={}, persist_seg={}, seg_decay={}, classes={}, "
            "residual_blocks={}, bottleneck={}, output_dropout={}, "
            "seg_attn_pool={}, temporal_attn={}",
            segment_encoder_type,
            self.d_seg,
            temporal_input_dim,
            temporal_lstm_hidden,
            temporal_lstm_layers,
            temporal_lstm_residual,
            gap_encoding,
            position_embed_dim,
            tlo_enabled,
            tlo_embed_dim,
            delta_t_embed_dim,
            persist_segment_state,
            segment_state_decay,
            num_classes,
            classifier_num_residual_blocks,
            classifier_bottleneck_dim,
            output_dropout,
            self.segment_attention_pool,
            temporal_attention,
        )

    # ------------------------------------------------------------------ #
    #  Segment encoders                                                    #
    # ------------------------------------------------------------------ #

    def _encode_segments_mean_pool(
        self, mu_post: Tensor, mask: Tensor,
    ) -> Tensor:
        """Mean-pool VAE latent over the 300 within-segment timesteps.

        Args:
            mu_post: Per-segment VAE posterior mean of shape
                ``(B, S_max, T, D_latent)`` where ``T=300`` and
                ``D_latent=16``.
            mask: Boolean validity mask ``(B, S_max)``.  ``True`` for real
                segments, ``False`` for padding.

        Returns:
            Segment vectors ``(B, S_max, 16)`` with padded positions zeroed.
        """
        # mu_post: (B, S_max, 300, 16) → mean over T → (B, S_max, 16)
        v = mu_post.mean(dim=2)
        # Zero out padded positions.
        v = v * mask.unsqueeze(-1).float()
        return v

    def _encode_segments_lstm(
        self, mu_post: Tensor, mask: Tensor,
    ) -> Tensor:
        """Encode segments via a causal LSTM, taking the last hidden state.

        Only processes valid (non-padded) segments for efficiency.  Flattens
        valid segments, runs through the segment LSTM, and scatters results
        back into the full ``(B, S_max, d_seg)`` tensor.

        Args:
            mu_post: Per-segment VAE posterior mean ``(B, S_max, 300, 16)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Segment vectors ``(B, S_max, d_seg)`` with padded positions zeroed.
        """
        B, S_max, T, D = mu_post.shape
        device = mu_post.device

        # Flatten → (B*S_max, T, D) and select valid.
        flat = mu_post.reshape(B * S_max, T, D)
        mask_flat = mask.reshape(B * S_max)
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]

        if valid_idx.numel() == 0:
            return torch.zeros(B, S_max, self.d_seg, device=device)

        mu_valid = flat[valid_idx]  # (N_valid, 300, 16)
        output_all, (h_n, _) = self.segment_lstm(mu_valid)
        if self.segment_attention_pool:
            seg_vectors = self.seg_attn_pool(output_all)[0]  # (N_valid, d_seg)
        else:
            seg_vectors = h_n.squeeze(0)  # (N_valid, d_seg)

        # Scatter back.
        v_flat = torch.zeros(B * S_max, self.d_seg, device=device)
        v_flat[valid_idx] = seg_vectors
        return v_flat.reshape(B, S_max, self.d_seg)

    def _encode_segments_lstm_persistent(
        self, mu_post: Tensor, mask: Tensor, delta_t: Tensor,
    ) -> Tensor:
        """Encode segments via LSTM with state persistence across segments.

        Processes segments in temporal order, carrying the LSTM hidden state
        from one segment to the next.  When ``segment_state_decay`` is enabled,
        the hidden and cell states are decayed proportionally to the
        inter-segment time gap before processing each subsequent segment.

        This allows within-segment patterns (e.g. a deceleration at the end
        of segment j) to influence the encoding of segment j+1.

        Args:
            mu_post: Per-segment VAE posterior mean ``(B, S_max, 300, 16)``.
            mask: Boolean validity mask ``(B, S_max)``.
            delta_t: Inter-segment gap in seconds ``(B, S_max)``.

        Returns:
            Segment vectors ``(B, S_max, d_seg)`` with padded positions zeroed.
        """
        B, S_max, T, D = mu_post.shape
        device = mu_post.device
        d_seg = self.d_seg

        v = torch.zeros(B, S_max, d_seg, device=device)

        # Initialise LSTM hidden/cell state.
        h = torch.zeros(1, B, d_seg, device=device)
        c = torch.zeros(1, B, d_seg, device=device)

        # Precompute decay rates if enabled.
        if self.segment_state_decay and hasattr(self, "gamma_seg_log"):
            gamma_seg = F.softplus(self.gamma_seg_log)  # (d_seg,)
        else:
            gamma_seg = None

        for j in range(S_max):
            step_valid = mask[:, j]  # (B,) bool

            # Skip steps where NO batch item has a valid segment — avoids
            # a full 300-timestep LSTM forward pass on all-zero input.
            if not step_valid.any():
                continue

            # -- Time-decay before step (skip j=0, no predecessor) --------- #
            if j > 0 and gamma_seg is not None:
                dt_j = delta_t[:, j].unsqueeze(-1) / 3600.0  # (B, 1) hours
                decay = torch.exp(
                    -gamma_seg.unsqueeze(0) * dt_j
                )  # (B, d_seg)
                h = h * decay.unsqueeze(0)  # (1, B, d_seg)
                c = c * decay.unsqueeze(0)

            # -- Run segment LSTM over 300 timesteps ----------------------- #
            seg_input = mu_post[:, j, :, :]  # (B, T, D)
            output_all, (h_new, c_new) = self.segment_lstm(seg_input, (h, c))

            # -- Store segment vector -------------------------------------- #
            if self.segment_attention_pool:
                v[:, j, :] = self.seg_attn_pool(output_all)[0]  # (B, d_seg)
            else:
                v[:, j, :] = h_new.squeeze(0)  # (B, d_seg)

            # -- Update state, zeroing invalid positions ------------------- #
            step_mask = step_valid.unsqueeze(0).unsqueeze(-1).float()  # (1, B, 1)
            # For valid segments, carry forward new state.
            # For invalid segments, carry forward old state (unchanged).
            h = h_new * step_mask + h * (1.0 - step_mask)
            c = c_new * step_mask + c * (1.0 - step_mask)

        # Zero out padded positions in output.
        v = v * mask.unsqueeze(-1).float()
        return v

    def _encode_segments_cnn(
        self, mu_post: Tensor, mask: Tensor,
    ) -> Tensor:
        """Encode segments via causal 1-D convolution + global average pool.

        Applies causal left-padding, 1-D convolution, GELU activation, and
        global average pooling.

        Args:
            mu_post: Per-segment VAE posterior mean ``(B, S_max, 300, 16)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Segment vectors ``(B, S_max, d_seg)`` with padded positions zeroed.
        """
        B, S_max, T, D = mu_post.shape
        device = mu_post.device

        flat = mu_post.reshape(B * S_max, T, D)
        mask_flat = mask.reshape(B * S_max)
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]

        if valid_idx.numel() == 0:
            return torch.zeros(B, S_max, self.d_seg, device=device)

        mu_valid = flat[valid_idx]  # (N_valid, 300, 16)
        # Conv1d expects (N, C, L) → transpose to (N, 16, 300).
        x_conv = mu_valid.transpose(1, 2)
        # Causal left-padding.
        x_conv = F.pad(x_conv, (self._cnn_kernel - 1, 0))
        seg_vectors = self.segment_cnn(x_conv).squeeze(-1)  # (N_valid, d_seg)

        v_flat = torch.zeros(B * S_max, self.d_seg, device=device)
        v_flat[valid_idx] = seg_vectors
        return v_flat.reshape(B, S_max, self.d_seg)

    # ------------------------------------------------------------------ #
    #  VAE encoding                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_kld_per_dim(mu_post, logvar_post, mu_prior, logvar_prior):
        """Per-dimension KL divergence (transfer entropy signal)."""
        var_post = logvar_post.exp()
        var_prior = logvar_prior.exp().clamp(min=1e-8)
        return 0.5 * (
            logvar_prior - logvar_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1.0
        )

    def _encode_vae_chunked(
        self,
        fhr_st: Tensor,
        fhr_ph: Tensor,
        fhr_up_ph: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Encode all valid segments through the frozen VAE in chunks.

        When ``enriched_features=True``, concatenates mu_post, logvar_post,
        posterior-prior residual, and per-dim KLD into a 64-dim feature.

        Args:
            fhr_st: Scattering features ``(B, S_max, 300, C_st)``.
            fhr_ph: Phase-harmonic features ``(B, S_max, 300, C_ph)``.
            fhr_up_ph: Cross-phase features ``(B, S_max, 300, C_x)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Features of shape ``(B, S_max, 300, D)`` where D is
            ``_feature_dim`` (64 when enriched, 16 otherwise).
        """
        B, S_max, T, C_st = fhr_st.shape
        device = fhr_st.device
        D = self._feature_dim

        fhr_st_flat = fhr_st.reshape(B * S_max, T, C_st)
        fhr_ph_flat = fhr_ph.reshape(B * S_max, T, fhr_ph.shape[-1])
        fhr_up_ph_flat = fhr_up_ph.reshape(B * S_max, T, fhr_up_ph.shape[-1])
        mask_flat = mask.reshape(B * S_max)

        valid_idx = mask_flat.nonzero(as_tuple=True)[0]
        N_valid = valid_idx.numel()

        features_flat = torch.zeros(B * S_max, T, D, device=device)

        if N_valid == 0:
            return features_flat.reshape(B, S_max, T, D)

        if self.freeze_vae:
            self.vae_model.eval()

        ctx = torch.no_grad() if self.freeze_vae else torch.enable_grad()
        with ctx:
            for i in range(0, N_valid, self.vae_chunk_size):
                chunk_idx = valid_idx[i: i + self.vae_chunk_size]
                enc = self.vae_model.encode_only(
                    y_st=fhr_st_flat[chunk_idx],
                    y_ph=fhr_ph_flat[chunk_idx],
                    x_ph=fhr_up_ph_flat[chunk_idx],
                    sample_z=False,
                )

                mu_post = enc["mu_post"]
                z = mu_post if self.use_posterior else enc["mu_prior"]

                # Training-time augmentation
                if self.training and self.augment_posterior_sample:
                    noise = (
                        torch.randn_like(z)
                        * (0.5 * enc["logvar_post"]).exp()
                        * self.augment_noise_scale
                    )
                    z = z + noise

                if self.training and self.augment_temporal_jitter > 0:
                    shift = torch.randint(
                        -self.augment_temporal_jitter,
                        self.augment_temporal_jitter + 1,
                        (1,),
                    ).item()
                    if shift != 0:
                        z = torch.roll(z, shifts=shift, dims=1)

                if self.enriched_features:
                    residual = mu_post - enc["mu_prior"]
                    kld = self._compute_kld_per_dim(
                        mu_post, enc["logvar_post"],
                        enc["mu_prior"], enc["logvar_prior"],
                    )
                    chunk_features = torch.cat(
                        [z, enc["logvar_post"], residual, kld], dim=-1,
                    )
                else:
                    chunk_features = z

                features_flat[chunk_idx] = chunk_features

        return features_flat.reshape(B, S_max, T, D)

    # ------------------------------------------------------------------ #
    #  Temporal feature encoders                                           #
    # ------------------------------------------------------------------ #

    def _encode_delta_t_concat(self, delta_t: Tensor) -> Tensor:
        """Encode delta_t as ``[hours, log-scaled]`` features.

        Args:
            delta_t: Inter-segment gap in seconds ``(B, S_max)``.

        Returns:
            Delta-t features ``(B, S_max, 2)``.
        """
        delta_t_hours = delta_t / 3600.0
        delta_t_log = torch.log1p(delta_t.abs()) / 10.0
        return torch.stack([delta_t_hours, delta_t_log], dim=-1)

    def _encode_position(
        self, segment_indices: Tensor, mask: Tensor,
    ) -> Tensor:
        """Encode segment grid indices as learned embeddings.

        Clamps ``-1`` (padding sentinel) to ``0`` before lookup, then zeroes
        padded positions so the embedding at index 0 receives no spurious
        gradient from padded slots.

        Args:
            segment_indices: Integer grid indices ``(B, S_max)``, padded with
                ``-1``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Position embeddings ``(B, S_max, position_embed_dim)``.
        """
        idx_clamped = segment_indices.clamp(
            min=0, max=self.position_embedding.num_embeddings - 1,
        )
        pos_embed = self.position_embedding(idx_clamped)
        return pos_embed * mask.unsqueeze(-1).float()

    def _encode_tlo(
        self, time_from_labor_onset: Tensor, mask: Tensor,
    ) -> Tensor:
        """Encode time_from_labor_onset with NaN-safe zero+flag strategy.

        NaN values are replaced with ``0.0`` and flagged with an availability
        indicator of ``0.0``.  Valid values are normalised to hours and flagged
        with ``1.0``.  The output is guaranteed NaN-free.

        Args:
            time_from_labor_onset: TLO in seconds ``(B, S_max)``, may contain
                ``NaN``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            TLO features ``(B, S_max, 2)`` — ``[tlo_hours, tlo_flag]``.
            Guaranteed no NaN in output.
        """
        tlo = time_from_labor_onset.clone()
        tlo_valid = ~torch.isnan(tlo) & mask
        tlo_hours = tlo / 3600.0
        tlo_hours = torch.where(
            torch.isnan(tlo_hours),
            torch.zeros_like(tlo_hours),
            tlo_hours,
        )
        tlo_flag = tlo_valid.float()
        # Zero padded positions.
        tlo_hours = tlo_hours * mask.float()
        tlo_flag = tlo_flag * mask.float()
        return torch.stack([tlo_hours, tlo_flag], dim=-1)

    # ------------------------------------------------------------------ #
    #  Time-decay temporal LSTM (custom loop)                              #
    # ------------------------------------------------------------------ #

    def _temporal_lstm_time_decay(
        self,
        x: Tensor,
        delta_t: Tensor,
        mask: Tensor,
        lengths: Tensor,
        return_cell_states: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Custom LSTM loop with time-decay gating on hidden/cell state.

        Before each step ``j > 0`` the hidden and cell states are decayed
        proportionally to the inter-segment time gap::

            gamma = softplus(gamma_log)            # (hidden_dim,)
            decay = exp(-gamma * delta_t_j / 3600) # (B, hidden_dim)
            h = h * decay;  c = c * decay

        Uses ``nn.LSTMCell`` for per-step control (cannot use ``nn.LSTM``).
        Multi-layer support: each layer's output feeds the next.

        Args:
            x: Temporal LSTM input ``(B, S_max, input_dim)``.
            delta_t: Inter-segment gap in seconds ``(B, S_max)``.
            mask: Boolean validity mask ``(B, S_max)``.
            lengths: Segment counts per GUID ``(B,)`` int.
            return_cell_states: If ``True``, also return per-step top-layer
                cell states for use by :class:`TemporalCellAttention`.

        Returns:
            When ``return_cell_states=False``: LSTM output ``(B, S_max, H)``.
            When ``return_cell_states=True``: tuple of
            ``(h_out, c_out)`` both ``(B, S_max, H)``.
        """
        B, S_max, _ = x.shape
        device = x.device
        hidden_dim = self._temporal_lstm_hidden
        num_layers = self._temporal_lstm_layers

        gamma = F.softplus(self.gamma_log)  # (hidden_dim,) — positive rates

        # Initialise hidden/cell states per layer.
        h_states = [torch.zeros(B, hidden_dim, device=device) for _ in range(num_layers)]
        c_states = [torch.zeros(B, hidden_dim, device=device) for _ in range(num_layers)]

        outputs = []
        cell_buf = [] if return_cell_states else None
        for j in range(S_max):
            # -- Time-decay before step (skip j=0, no predecessor) ---------- #
            if j > 0:
                dt_j = delta_t[:, j].unsqueeze(-1) / 3600.0  # (B, 1) hours
                decay = torch.exp(-gamma.unsqueeze(0) * dt_j)  # (B, hidden_dim)
                for layer_idx in range(num_layers):
                    h_states[layer_idx] = h_states[layer_idx] * decay
                    c_states[layer_idx] = c_states[layer_idx] * decay

            # -- LSTMCell forward per layer --------------------------------- #
            layer_input = x[:, j, :]  # (B, input_dim)
            for layer_idx in range(num_layers):
                h_states[layer_idx], c_states[layer_idx] = self.temporal_lstm_cells[layer_idx](
                    layer_input, (h_states[layer_idx], c_states[layer_idx]),
                )
                layer_input = h_states[layer_idx]
                # Dropout between layers (not after last).
                if self._td_dropout is not None and layer_idx < num_layers - 1:
                    layer_input = self._td_dropout(layer_input)

            # -- Mask padded positions -------------------------------------- #
            step_mask = mask[:, j].unsqueeze(-1).float()  # (B, 1)
            for layer_idx in range(num_layers):
                h_states[layer_idx] = h_states[layer_idx] * step_mask
                c_states[layer_idx] = c_states[layer_idx] * step_mask

            outputs.append(h_states[-1])  # Top-layer hidden state
            if cell_buf is not None:
                cell_buf.append(c_states[-1].clone())

        h_out = torch.stack(outputs, dim=1)  # (B, S_max, hidden_dim)
        if return_cell_states:
            c_out = torch.stack(cell_buf, dim=1)  # (B, S_max, hidden_dim)
            return h_out, c_out
        return h_out

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Full forward pass through the temporal classifier.

        Orchestrates:
          1. NaN firewall on TLO (if enabled).
          2. Chunked VAE encoding of all valid segments.
          3. Segment-level encoding (mean-pool / LSTM / CNN).
          4. Temporal feature encoding (delta_t, position, TLO).
          5. Feature concatenation.
          6. Packed temporal LSTM forward.
          7. Per-segment classification.

        Args:
            batch: Dict from ``sequence_collate_fn`` with at least:
                ``fhr_st`` ``(B, S_max, 300, C_st)``,
                ``fhr_ph`` ``(B, S_max, 300, C_ph)``,
                ``fhr_up_ph`` ``(B, S_max, 300, C_x)``,
                ``delta_t`` ``(B, S_max)``,
                ``mask`` ``(B, S_max)`` bool,
                ``lengths`` ``(B,)`` int.
                Optionally: ``segment_indices``, ``time_from_labor_onset``,
                ``mu_post_precomputed``.

        Returns:
            Dict with keys:
                ``logits`` ``(B, S_max, num_classes)``,
                ``probs`` ``(B, S_max, num_classes)``,
                ``preds`` ``(B, S_max)``,
                ``mask`` ``(B, S_max)`` (forwarded for downstream use).
        """
        mask = batch["mask"]                  # (B, S_max) bool
        lengths = batch["lengths"]            # (B,)
        delta_t = batch["delta_t"]            # (B, S_max)
        B, S_max = mask.shape

        # ---- Step 1: NaN firewall on TLO -------------------------------- #
        # Must happen before ANY tensor operations to prevent NaN propagation.
        if self.tlo_enabled and "time_from_labor_onset" in batch:
            if self.tlo_embed_dim > 0:
                tlo_feat = self.tlo_embedding(
                    batch["time_from_labor_onset"], mask,
                )
            else:
                tlo_feat = self._encode_tlo(
                    batch["time_from_labor_onset"], mask,
                )
            if self.debug:
                assert not torch.isnan(tlo_feat).any(), "NaN in TLO features after encoding"
        else:
            tlo_feat = None

        # ---- Step 2: VAE encoding --------------------------------------- #
        if "mu_post_precomputed" in batch:
            mu_post = batch["mu_post_precomputed"]  # (B, S_max, 300, 16)
        else:
            mu_post = self._encode_vae_chunked(
                fhr_st=batch["fhr_st"],
                fhr_ph=batch["fhr_ph"],
                fhr_up_ph=batch["fhr_up_ph"],
                mask=mask,
            )

        # ---- Step 3: Segment encoding ----------------------------------- #
        if self.segment_encoder_type == "mean_pool":
            v = self._encode_segments_mean_pool(mu_post, mask)
        elif self.segment_encoder_type == "lstm":
            if self.persist_segment_state:
                v = self._encode_segments_lstm_persistent(mu_post, mask, delta_t)
            else:
                v = self._encode_segments_lstm(mu_post, mask)
        elif self.segment_encoder_type == "cnn":
            v = self._encode_segments_cnn(mu_post, mask)
        else:
            raise RuntimeError(f"Unexpected encoder type: {self.segment_encoder_type}")

        # ---- Step 4: Temporal feature encoding --------------------------- #
        features = [v]  # Start with segment vectors.

        if self.gap_encoding in ("concat", "both"):
            if self.delta_t_embed_dim > 0:
                features.append(self.delta_t_embedding(delta_t, mask))
            else:
                features.append(self._encode_delta_t_concat(delta_t))

        if self.position_embed_dim > 0 and "segment_indices" in batch:
            features.append(
                self._encode_position(batch["segment_indices"], mask)
            )

        if tlo_feat is not None:
            features.append(tlo_feat)

        # ---- Step 5: Concatenate + LayerNorm ----------------------------- #
        x = torch.cat(features, dim=-1)  # (B, S_max, temporal_input_dim)
        x = self.temporal_input_norm(x)

        if self.debug:
            assert not torch.isnan(x).any(), "NaN in temporal LSTM input"

        # ---- Step 6: Temporal LSTM --------------------------------------- #
        c_all = None
        if self.gap_encoding in ("time_decay", "both"):
            if self.temporal_attention_enabled:
                h, c_all = self._temporal_lstm_time_decay(
                    x, delta_t, mask, lengths, return_cell_states=True,
                )
            else:
                h = self._temporal_lstm_time_decay(x, delta_t, mask, lengths)
        else:
            # Standard packed-sequence LSTM (concat gap encoding).
            lengths_cpu = lengths.cpu().clamp(min=1)
            x_packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            h_packed, _ = self.temporal_lstm(x_packed)
            h, _ = pad_packed_sequence(
                h_packed, batch_first=True, total_length=S_max,
            )
        # h: (B, S_max, temporal_lstm_hidden)

        # Residual skip connection around temporal LSTM.
        if self.temporal_lstm_residual:
            h = h + self.temporal_residual_proj(x)

        # Output dropout on LSTM hidden states.
        h = self.output_drop(h)

        if self.debug:
            assert not torch.isnan(h).any(), "NaN in temporal LSTM output"

        # ---- Step 6b: ATTAIN temporal attention (optional) --------------- #
        temporal_attn_weights = None
        if self.temporal_attention_enabled and c_all is not None:
            h, temporal_attn_weights = self.temporal_cell_attention(
                h, c_all, delta_t, mask,
            )

        # ---- Step 7: Per-segment classification -------------------------- #
        for block in self.residual_blocks:
            h = block(h)
        logits = self.classifier_head(h)        # (B, S_max, num_classes)
        probs = F.softmax(logits, dim=-1)       # (B, S_max, num_classes)
        preds = logits.argmax(dim=-1)           # (B, S_max)

        result = {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "mask": mask,
        }
        if temporal_attn_weights is not None:
            result["temporal_attention_weights"] = temporal_attn_weights
        return result

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute masked loss over valid segments.

        Supports both binary (cross-entropy) and hierarchical (focal BCE)
        modes. Per-segment labels are derived by taking
        ``target.max(dim=-1)`` over the 300 within-segment timesteps.

        Args:
            outputs: Dict from :meth:`forward` containing ``logits`` and
                ``mask``.
            batch: Dict from ``sequence_collate_fn`` containing ``target``
                ``(B, S_max, 300)``.

        Returns:
            Dict with ``loss``, ``accuracy``, ``class_0_acc``,
            ``class_1_acc``.
        """
        logits = outputs["logits"]   # (B, S_max, num_classes)
        mask = outputs["mask"]       # (B, S_max) bool
        target = batch["target"]     # (B, S_max, 300)

        logits_valid = logits[mask]                     # (N_valid, num_classes)
        target_valid = target[mask]                     # (N_valid, 300)

        if logits_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=logits.device)
            return {
                "loss": zero, "accuracy": 0.0,
                "class_0_acc": 0.0, "class_1_acc": 0.0,
            }

        seg_labels = target_valid.max(dim=-1)[0]        # (N_valid,)
        binary_labels = (seg_labels > 1).long()         # (N_valid,) {0,1}

        # --- Loss computation ---
        if self.label_mode == "hierarchical":
            hier_targets = map_to_hierarchical_labels(seg_labels.long())
            loss = self.focal_loss_fn(logits_valid, hier_targets)
            # Primary prediction: unhealthy bit (index 1)
            preds_valid = (torch.sigmoid(logits_valid[:, 1]) > 0.5).long()
        else:
            loss = F.cross_entropy(
                logits_valid, binary_labels, weight=self.class_weights,
            )
            preds_valid = logits_valid.argmax(dim=-1)

        # --- Metrics ---
        accuracy = (preds_valid == binary_labels).float().mean()

        cls0_mask = binary_labels == 0
        cls1_mask = binary_labels == 1
        cls0_acc = (
            (preds_valid[cls0_mask] == 0).float().mean()
            if cls0_mask.any()
            else torch.tensor(0.0, device=loss.device)
        )
        cls1_acc = (
            (preds_valid[cls1_mask] == 1).float().mean()
            if cls1_mask.any()
            else torch.tensor(0.0, device=loss.device)
        )

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "class_0_acc": cls0_acc.item() if isinstance(cls0_acc, Tensor) else cls0_acc,
            "class_1_acc": cls1_acc.item() if isinstance(cls1_acc, Tensor) else cls1_acc,
        }
