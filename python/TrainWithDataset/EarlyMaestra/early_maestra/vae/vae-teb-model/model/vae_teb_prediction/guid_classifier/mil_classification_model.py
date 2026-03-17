"""MIL-based temporal classifiers for GUID-level CTG classification.

Implements three Multiple Instance Learning (MIL) variants that treat each
GUID recording as a *bag* of segment *instances*.  Unlike the per-segment
cross-entropy approach in :class:`TemporalVaeClassifier`, MIL applies a
single GUID-level loss, eliminating the label-noise problem where every
segment inherits the baby's outcome label regardless of whether that
segment actually exhibits pathology.

Variants:

- **ABMIL** (``ABMILClassifier``): Gated attention pooling (Ilse et al. 2018).
- **TransMIL** (``TransMILClassifier``): Self-attention between segments via
  Transformer encoder with a CLS token (Shao et al. NeurIPS 2021).
- **CausalMIL** (``CausalMILClassifier``): Causal self-attention producing a
  per-segment risk trajectory with monotonicity regularisation for early
  detection.

All three share a common base (``BaseMILClassifier``) that handles frozen VAE
encoding, segment encoding, and temporal feature construction.

Example::

    from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
    from train.graph_models_utils import load_checkpoint_strict

    vae = SeqVae()
    load_checkpoint_strict(vae, checkpoint="/path/to/vae.ckpt")

    model = ABMILClassifier(
        vae_model=vae,
        segment_encoder_type="simple",
        d_seg=128,
        attn_dim=128,
    )
    outputs = model(batch)        # batch from sequence_collate_fn
    loss_dict = model.compute_loss(outputs, batch)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loguru import logger

from model.vae_teb_prediction.guid_classifier.temporal_classification_model import (
    DeltaTEmbedding,
    TemporalTLOEmbedding,
)


# ================================================================== #
#  Segment Encoders                                                    #
# ================================================================== #


class SegmentEncoderSimple(nn.Module):
    """Multi-statistic pooling over within-segment timesteps + MLP projection.

    Reduces ``(B, S_max, T, D_latent)`` to ``(B, S_max, d_seg)`` by computing
    five summary statistics over the ``T=300`` within-segment timesteps:

    - **Mean**: Average coupling strength across the segment.
    - **Std**: Variability of coupling (captures decelerations, accelerations).
    - **Min**: Lowest coupling value (nadir of decelerations).
    - **Max**: Peak coupling (baseline or acceleration peaks).
    - **Slope**: Difference between last-third and first-third means,
      capturing within-segment temporal trends (deterioration vs recovery).

    The concatenated ``5 * D_latent`` vector is projected through a 3-layer
    MLP with residual connection, providing sufficient capacity for the
    challenging binary classification task.

    Args:
        latent_dim: VAE latent dimension (default 16).
        d_seg: Output segment embedding dimension.
        dropout: Dropout probability inside the MLP and on the output.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        d_seg: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_seg = d_seg
        # 5 statistics × latent_dim: mean, std, min, max, slope.
        pool_dim = latent_dim * 5
        self.input_norm = nn.LayerNorm(pool_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, d_seg * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_seg * 2, d_seg),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_seg, d_seg),
        )
        # Residual skip projection for the case pool_dim != d_seg.
        self.skip_proj = nn.Linear(pool_dim, d_seg, bias=False)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, mu_post: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            mu_post: Per-segment VAE posterior mean of shape
                ``(B, S_max, T, D_latent)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Segment embeddings ``(B, S_max, d_seg)`` with padded positions
            zeroed.
        """
        # Five pooled statistics over T=300 timesteps.
        mean_val = mu_post.mean(dim=2)        # (B, S_max, D)
        std_val = mu_post.std(dim=2)          # (B, S_max, D)
        min_val = mu_post.min(dim=2)[0]       # (B, S_max, D)
        max_val = mu_post.max(dim=2)[0]       # (B, S_max, D)

        # Slope: last-third mean minus first-third mean.
        T = mu_post.size(2)
        third = T // 3
        first_third = mu_post[:, :, :third, :].mean(dim=2)  # (B, S_max, D)
        last_third = mu_post[:, :, -third:, :].mean(dim=2)  # (B, S_max, D)
        slope = last_third - first_third                     # (B, S_max, D)

        pooled = torch.cat(
            [mean_val, std_val, min_val, max_val, slope], dim=-1,
        )  # (B, S_max, 5*D)

        normed = self.input_norm(pooled)
        v = self.mlp(normed) + self.skip_proj(normed)  # residual
        v = self.output_dropout(v)
        return v * mask.unsqueeze(-1).float()


class SegmentEncoderRich(nn.Module):
    """Rich segment encoder reusing causal depthwise-separable conv + BiLSTM.

    Mirrors the feature-extraction backbone of
    :class:`~model.vae_teb_prediction.prediction_classification_model.CausalCNNLSTMClassifier`
    but operates per-segment: flattens valid segments, runs through conv stages
    and BiLSTM, applies mean-max pooling, and projects to ``d_seg``.

    Args:
        latent_dim: VAE latent dimension (default 16).
        d_seg: Output segment embedding dimension.
        conv_channels: Channel count for each causal conv stage.
        kernel_sizes: Kernel size per stage.
        dilations: Dilation factor per stage.
        lstm_hidden: BiLSTM hidden dim per direction.
        lstm_layers: Number of stacked BiLSTM layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        d_seg: int = 128,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_sizes: Sequence[int] = (5, 7, 11),
        dilations: Sequence[int] = (1, 2, 4),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_seg = d_seg

        # Import CausalConvBlock from the independent classifier module.
        from model.vae_teb_prediction.prediction_classification_model import (
            CausalConvBlock,
        )

        conv_blocks: List[nn.Module] = []
        in_ch = latent_dim
        for ch, ks, dil in zip(conv_channels, kernel_sizes, dilations):
            conv_blocks.append(
                CausalConvBlock(in_ch, ch, ks, dilation=dil, dropout=dropout)
            )
            in_ch = ch
        self.conv_stages = nn.Sequential(*conv_blocks)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # mean-max pool → 2 * lstm_out_dim → project to d_seg
        pooled_dim = lstm_out_dim * 2
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, d_seg),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, mu_post: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            mu_post: Per-segment VAE posterior mean ``(B, S_max, T, D_latent)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Segment embeddings ``(B, S_max, d_seg)`` with padded positions
            zeroed.
        """
        B, S_max, T, D = mu_post.shape
        device = mu_post.device

        flat = mu_post.reshape(B * S_max, T, D)
        mask_flat = mask.reshape(B * S_max)
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]

        if valid_idx.numel() == 0:
            return torch.zeros(B, S_max, self.d_seg, device=device)

        mu_valid = flat[valid_idx]  # (N_valid, T, D)

        # Conv stages: (N, T, D) → (N, D, T) → conv → (N, C_last, T) → (N, T, C_last)
        h = mu_valid.transpose(1, 2)
        h = self.conv_stages(h)
        h = h.transpose(1, 2)

        # BiLSTM
        h, _ = self.lstm(h)
        h = self.lstm_norm(h)

        # Mean-max pool
        mean_val = h.mean(dim=1)
        max_val, _ = h.max(dim=1)
        pooled = torch.cat([mean_val, max_val], dim=1)  # (N_valid, 2*lstm_out)

        seg_vectors = self.proj(pooled)  # (N_valid, d_seg)

        # Scatter back
        v_flat = torch.zeros(B * S_max, self.d_seg, device=device)
        v_flat[valid_idx] = seg_vectors
        return v_flat.reshape(B, S_max, self.d_seg)


# ================================================================== #
#  Temporal Feature Encoder                                            #
# ================================================================== #


class TemporalFeatureEncoder(nn.Module):
    """Encodes and concatenates temporal context features with segment embeddings.

    Reuses :class:`DeltaTEmbedding` and :class:`TemporalTLOEmbedding` from
    the temporal classification model.  Produces a LayerNorm-normalised
    feature vector per segment.

    Args:
        d_seg: Dimension of segment embeddings (input).
        delta_t_embed_dim: Dimension of learned delta_t embedding (0 = raw 2-dim).
        delta_t_dropout: Dropout for delta_t MLP.
        position_embed_dim: Dimension of position embedding (0 = disabled).
        max_position_index: Max grid slot index for embedding table.
        tlo_enabled: Whether to use TLO features.
        tlo_embed_dim: Dimension of learned TLO embedding (0 = raw 2-dim).
        tlo_dropout: Dropout for TLO MLP.
    """

    def __init__(
        self,
        d_seg: int,
        *,
        delta_t_embed_dim: int = 8,
        delta_t_dropout: float = 0.1,
        position_embed_dim: int = 8,
        max_position_index: int = 40,
        tlo_enabled: bool = True,
        tlo_embed_dim: int = 8,
        tlo_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tlo_enabled = tlo_enabled
        self.delta_t_embed_dim = delta_t_embed_dim
        self.position_embed_dim = position_embed_dim
        self.tlo_embed_dim = tlo_embed_dim

        output_dim = d_seg

        # Delta-t embedding
        if delta_t_embed_dim > 0:
            self.delta_t_embedding = DeltaTEmbedding(
                embed_dim=delta_t_embed_dim, dropout=delta_t_dropout,
            )
            output_dim += delta_t_embed_dim
        else:
            self.delta_t_embedding = None
            output_dim += 2  # raw [hours, log]

        # Position embedding
        if position_embed_dim > 0:
            self.position_embedding = nn.Embedding(
                num_embeddings=max_position_index,
                embedding_dim=position_embed_dim,
            )
            output_dim += position_embed_dim
        else:
            self.position_embedding = None

        # TLO embedding
        if tlo_enabled:
            if tlo_embed_dim > 0:
                self.tlo_embedding = TemporalTLOEmbedding(
                    embed_dim=tlo_embed_dim, dropout=tlo_dropout,
                )
                output_dim += tlo_embed_dim
            else:
                self.tlo_embedding = None
                output_dim += 2  # raw [hours, flag]
        else:
            self.tlo_embedding = None

        self.output_dim = output_dim
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        v: Tensor,
        batch: Dict[str, Tensor],
        mask: Tensor,
    ) -> Tensor:
        """Concatenate segment embeddings with temporal features.

        Args:
            v: Segment embeddings ``(B, S_max, d_seg)``.
            batch: Batch dict from ``sequence_collate_fn``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Feature tensor ``(B, S_max, output_dim)`` after LayerNorm.
        """
        features = [v]
        delta_t = batch["delta_t"]

        # Delta-t
        if self.delta_t_embedding is not None:
            features.append(self.delta_t_embedding(delta_t, mask))
        else:
            dt_hours = delta_t / 3600.0
            dt_log = torch.log1p(delta_t.abs()) / 10.0
            features.append(torch.stack([dt_hours, dt_log], dim=-1))

        # Position
        if self.position_embedding is not None and "segment_indices" in batch:
            idx = batch["segment_indices"].clamp(
                min=0, max=self.position_embedding.num_embeddings - 1,
            )
            pos_embed = self.position_embedding(idx)
            features.append(pos_embed * mask.unsqueeze(-1).float())

        # TLO
        if self.tlo_enabled and "time_from_labor_onset" in batch:
            tlo = batch["time_from_labor_onset"]
            if self.tlo_embedding is not None:
                features.append(self.tlo_embedding(tlo, mask))
            else:
                tlo_valid = ~torch.isnan(tlo) & mask
                tlo_hours = torch.where(
                    torch.isnan(tlo / 3600.0),
                    torch.zeros_like(tlo),
                    tlo / 3600.0,
                )
                tlo_flag = tlo_valid.float()
                tlo_hours = tlo_hours * mask.float()
                tlo_flag = tlo_flag * mask.float()
                features.append(torch.stack([tlo_hours, tlo_flag], dim=-1))

        x = torch.cat(features, dim=-1)  # (B, S_max, output_dim)
        return self.layer_norm(x)


# ================================================================== #
#  Gated Attention Pooling                                             #
# ================================================================== #


class GatedAttentionPooling(nn.Module):
    """Gated attention mechanism for MIL bag-level aggregation.

    Implements the gated attention from Ilse et al. (2018):

    .. math::

        a_j = \\frac{\\exp\\bigl(\\mathbf{w}^\\top
            (\\tanh(\\mathbf{V}\\mathbf{h}_j) \\odot
             \\sigma(\\mathbf{U}\\mathbf{h}_j))\\bigr)}
            {\\sum_k \\exp(\\ldots)}

    .. math::

        \\mathbf{z} = \\sum_j a_j \\mathbf{h}_j

    Args:
        input_dim: Dimension of segment feature vectors.
        attn_dim: Internal attention dimension.
    """

    def __init__(self, input_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.V = nn.Linear(input_dim, attn_dim)
        self.U = nn.Linear(input_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(
        self, h: Tensor, mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            h: Segment features ``(B, S_max, D)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Tuple of:
                - Bag representation ``(B, D)``.
                - Attention weights ``(B, S_max)`` (0 for padded positions).
        """
        # Gated attention scores
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # (B, S_max, 1)
        a = a.squeeze(-1)  # (B, S_max)

        # Mask: set padded positions to -inf before softmax
        a = a.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(a, dim=-1)  # (B, S_max)

        # Handle all-padded edge case (softmax of all -inf → NaN)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum
        z = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return z, attn_weights


# ================================================================== #
#  Base MIL Classifier                                                 #
# ================================================================== #


class BaseMILClassifier(nn.Module):
    """Abstract base class for MIL-based GUID-level classifiers.

    Handles:
        - Frozen VAE encoding (chunked for memory efficiency).
        - Segment encoding (simple or rich).
        - Temporal feature concatenation (delta_t, position, TLO).

    Subclasses implement :meth:`_aggregate_and_classify` and
    :meth:`compute_loss`.

    Args:
        vae_model: Pre-trained ``SeqVae`` instance.
        segment_encoder_type: ``'simple'`` or ``'rich'``.
        d_seg: Segment embedding dimension.
        delta_t_embed_dim: Learned delta_t embedding dim (0 = raw 2-dim).
        delta_t_dropout: Dropout for delta_t MLP.
        position_embed_dim: Position embedding dim (0 = disabled).
        max_position_index: Max position index for embedding table.
        tlo_enabled: Whether to use TLO features.
        tlo_embed_dim: Learned TLO embedding dim (0 = raw 2-dim).
        tlo_dropout: Dropout for TLO MLP.
        num_classes: Number of output classes.
        class_weights: Optional per-class loss weights.
        vae_chunk_size: Segments per VAE forward chunk.
        use_posterior: Use posterior mean (True) or prior mean (False).
        freeze_vae: Whether to freeze VAE parameters.
        rich_conv_channels: Conv channels for rich encoder.
        rich_kernel_sizes: Kernel sizes for rich encoder.
        rich_dilations: Dilations for rich encoder.
        debug: Enable NaN assertion guards.
    """

    _VAE_LATENT_DIM = 16

    def __init__(
        self,
        vae_model: nn.Module,
        *,
        segment_encoder_type: str = "simple",
        d_seg: int = 128,
        delta_t_embed_dim: int = 8,
        delta_t_dropout: float = 0.1,
        position_embed_dim: int = 8,
        max_position_index: int = 40,
        tlo_enabled: bool = True,
        tlo_embed_dim: int = 8,
        tlo_dropout: float = 0.1,
        num_classes: int = 2,
        class_weights: Optional[Sequence[float]] = None,
        vae_chunk_size: int = 32,
        use_posterior: bool = True,
        freeze_vae: bool = True,
        rich_conv_channels: Sequence[int] = (32, 64, 128),
        rich_kernel_sizes: Sequence[int] = (5, 7, 11),
        rich_dilations: Sequence[int] = (1, 2, 4),
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # kwargs absorbs variant-specific config keys passed from the factory.

        self.num_classes = num_classes
        self.vae_chunk_size = vae_chunk_size
        self.use_posterior = use_posterior
        self.freeze_vae = freeze_vae
        self.debug = debug
        self.segment_encoder_type = segment_encoder_type

        # -- VAE model (frozen) --
        self.vae_model = vae_model
        if freeze_vae:
            self.vae_model.eval()
            for param in self.vae_model.parameters():
                param.requires_grad = False

        # -- Segment encoder --
        if segment_encoder_type == "simple":
            self.segment_encoder = SegmentEncoderSimple(
                latent_dim=self._VAE_LATENT_DIM, d_seg=d_seg,
            )
        elif segment_encoder_type == "rich":
            self.segment_encoder = SegmentEncoderRich(
                latent_dim=self._VAE_LATENT_DIM,
                d_seg=d_seg,
                conv_channels=rich_conv_channels,
                kernel_sizes=rich_kernel_sizes,
                dilations=rich_dilations,
            )
        else:
            raise ValueError(
                f"Unknown segment_encoder_type '{segment_encoder_type}'. "
                "Expected 'simple' or 'rich'."
            )
        self.d_seg = d_seg

        # -- Temporal feature encoder --
        self.temporal_encoder = TemporalFeatureEncoder(
            d_seg=d_seg,
            delta_t_embed_dim=delta_t_embed_dim,
            delta_t_dropout=delta_t_dropout,
            position_embed_dim=position_embed_dim,
            max_position_index=max_position_index,
            tlo_enabled=tlo_enabled,
            tlo_embed_dim=tlo_embed_dim,
            tlo_dropout=tlo_dropout,
        )
        self.feature_dim = self.temporal_encoder.output_dim

        # -- Class weights --
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    # ------------------------------------------------------------------ #
    #  VAE encoding (copied from TemporalVaeClassifier)                    #
    # ------------------------------------------------------------------ #

    def _encode_vae_chunked(
        self,
        fhr_st: Tensor,
        fhr_ph: Tensor,
        fhr_up_ph: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Encode all valid segments through the frozen VAE in chunks.

        Args:
            fhr_st: Scattering features ``(B, S_max, 300, C_st)``.
            fhr_ph: Phase-harmonic features ``(B, S_max, 300, C_ph)``.
            fhr_up_ph: Cross-phase features ``(B, S_max, 300, C_x)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            ``mu_post`` of shape ``(B, S_max, 300, 16)``.  Padded segments
            are all-zero.
        """
        B, S_max, T, C_st = fhr_st.shape
        device = fhr_st.device

        fhr_st_flat = fhr_st.reshape(B * S_max, T, C_st)
        fhr_ph_flat = fhr_ph.reshape(B * S_max, T, fhr_ph.shape[-1])
        fhr_up_ph_flat = fhr_up_ph.reshape(B * S_max, T, fhr_up_ph.shape[-1])
        mask_flat = mask.reshape(B * S_max)

        valid_idx = mask_flat.nonzero(as_tuple=True)[0]
        N_valid = valid_idx.numel()

        mu_post_flat = torch.zeros(
            B * S_max, T, self._VAE_LATENT_DIM, device=device,
        )

        if N_valid == 0:
            return mu_post_flat.reshape(B, S_max, T, self._VAE_LATENT_DIM)

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
                key = "mu_post" if self.use_posterior else "mu_prior"
                mu_post_flat[chunk_idx] = enc[key]

        return mu_post_flat.reshape(B, S_max, T, self._VAE_LATENT_DIM)

    # ------------------------------------------------------------------ #
    #  Shared forward pipeline                                             #
    # ------------------------------------------------------------------ #

    def _get_segment_features(
        self, batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run VAE encoding, segment encoding, and temporal feature construction.

        Args:
            batch: Batch dict from ``sequence_collate_fn``.

        Returns:
            Tuple of ``(features, mask, lengths)`` where features is
            ``(B, S_max, feature_dim)`` after LayerNorm.
        """
        mask = batch["mask"]
        lengths = batch["lengths"]

        # VAE encoding
        if "mu_post_precomputed" in batch:
            mu_post = batch["mu_post_precomputed"]
        else:
            mu_post = self._encode_vae_chunked(
                fhr_st=batch["fhr_st"],
                fhr_ph=batch["fhr_ph"],
                fhr_up_ph=batch["fhr_up_ph"],
                mask=mask,
            )

        # Segment encoding
        v = self.segment_encoder(mu_post, mask)  # (B, S_max, d_seg)

        # Temporal features + LayerNorm
        features = self.temporal_encoder(v, batch, mask)  # (B, S_max, feature_dim)

        if self.debug:
            assert not torch.isnan(features).any(), "NaN in MIL segment features"

        return features, mask, lengths

    def _extract_guid_labels(
        self, batch: Dict[str, Tensor],
    ) -> Tensor:
        """Extract binary GUID-level labels from per-segment targets.

        For each GUID in the batch: if ANY valid segment has ``max(target) > 1``,
        the GUID is labelled unhealthy (1); otherwise healthy (0).

        Args:
            batch: Contains ``target`` ``(B, S_max, 300)`` and ``mask``
                ``(B, S_max)``.

        Returns:
            Binary GUID labels ``(B,)`` as long tensor.
        """
        target = batch["target"]  # (B, S_max, 300)
        mask = batch["mask"]  # (B, S_max)

        # Per-segment max class: (B, S_max)
        seg_labels = target.max(dim=-1)[0]
        # Zero out padding positions
        seg_labels = seg_labels * mask.float()
        # GUID label: any segment > 1 → unhealthy
        guid_labels = (seg_labels.max(dim=-1)[0] > 1).long()  # (B,)
        return guid_labels

    @abstractmethod
    def _aggregate_and_classify(
        self,
        features: Tensor,
        mask: Tensor,
        lengths: Tensor,
    ) -> Dict[str, Tensor]:
        """Subclass hook: MIL aggregation and classification.

        Args:
            features: Segment features ``(B, S_max, feature_dim)``.
            mask: Boolean validity mask ``(B, S_max)``.
            lengths: Segment counts per GUID ``(B,)``.

        Returns:
            Dict with at least: ``logits_guid``, ``probs_guid``, ``preds_guid``,
            ``logits``, ``probs``, ``preds``, ``mask``, ``attention_weights``.
        """

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Full forward pass: VAE → segments → features → MIL aggregation.

        Args:
            batch: Dict from ``sequence_collate_fn``.

        Returns:
            Dict with GUID-level and per-segment predictions, plus attention
            weights for interpretability.
        """
        features, mask, lengths = self._get_segment_features(batch)
        return self._aggregate_and_classify(features, mask, lengths)

    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute loss and metrics.

        Args:
            outputs: Dict from :meth:`forward`.
            batch: Dict from ``sequence_collate_fn``.

        Returns:
            Dict with ``loss``, ``accuracy``, ``class_0_acc``, ``class_1_acc``.
        """

    def _compute_guid_metrics(
        self,
        guid_logits: Tensor,
        guid_labels: Tensor,
    ) -> Dict[str, float]:
        """Compute GUID-level accuracy metrics.

        Args:
            guid_logits: GUID-level logits ``(B, num_classes)``.
            guid_labels: GUID-level binary labels ``(B,)``.

        Returns:
            Dict with ``accuracy``, ``class_0_acc``, ``class_1_acc``.
        """
        guid_preds = guid_logits.argmax(dim=-1)
        accuracy = (guid_preds == guid_labels).float().mean()

        cls0_mask = guid_labels == 0
        cls1_mask = guid_labels == 1
        cls0_acc = (
            (guid_preds[cls0_mask] == 0).float().mean()
            if cls0_mask.any()
            else torch.tensor(0.0, device=guid_logits.device)
        )
        cls1_acc = (
            (guid_preds[cls1_mask] == 1).float().mean()
            if cls1_mask.any()
            else torch.tensor(0.0, device=guid_logits.device)
        )

        return {
            "accuracy": accuracy.item(),
            "class_0_acc": cls0_acc.item() if isinstance(cls0_acc, Tensor) else cls0_acc,
            "class_1_acc": cls1_acc.item() if isinstance(cls1_acc, Tensor) else cls1_acc,
        }


# ================================================================== #
#  Variant 1: ABMIL                                                   #
# ================================================================== #


class ABMILClassifier(BaseMILClassifier):
    """Attention-Based MIL classifier for GUID-level CTG classification.

    Uses :class:`GatedAttentionPooling` to aggregate segment features into a
    single GUID representation, then classifies at the GUID level.  Per-segment
    predictions are produced by an independent projection head for evaluation
    pipeline compatibility.

    Args:
        vae_model: Pre-trained ``SeqVae`` instance.
        attn_dim: Internal dimension of the gated attention mechanism.
        classifier_dropout: Dropout in the classifier head.
        mlp_multiplier: Hidden expansion factor for the classifier head.
        **kwargs: Forwarded to :class:`BaseMILClassifier`.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        *,
        attn_dim: int = 128,
        classifier_dropout: float = 0.1,
        mlp_multiplier: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(vae_model=vae_model, **kwargs)

        self.attention = GatedAttentionPooling(self.feature_dim, attn_dim)

        mlp_hidden = int(self.feature_dim * mlp_multiplier)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(mlp_hidden, self.num_classes),
        )

        # Independent per-segment projector (for eval pipeline compatibility).
        # Uses a small MLP rather than a single Linear to give per-segment
        # predictions enough capacity for a hard binary classification task.
        self.segment_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.feature_dim // 2, self.num_classes),
        )

        logger.info(
            "ABMILClassifier created: feature_dim={}, attn_dim={}, "
            "segment_encoder={}, d_seg={}, classes={}",
            self.feature_dim, attn_dim,
            self.segment_encoder_type, self.d_seg, self.num_classes,
        )

    def _aggregate_and_classify(
        self,
        features: Tensor,
        mask: Tensor,
        lengths: Tensor,
    ) -> Dict[str, Tensor]:
        """ABMIL aggregation: gated attention pooling → GUID classification.

        Args:
            features: ``(B, S_max, feature_dim)``.
            mask: ``(B, S_max)`` boolean.
            lengths: ``(B,)`` int.

        Returns:
            Dict with GUID-level and per-segment predictions + attention weights.
        """
        # GUID-level via attention pooling
        z, attn_weights = self.attention(features, mask)  # z: (B, D), attn: (B, S_max)
        guid_logits = self.classifier_head(z)  # (B, num_classes)
        guid_probs = F.softmax(guid_logits, dim=-1)
        guid_preds = guid_logits.argmax(dim=-1)

        # Per-segment predictions (independent, for eval pipeline)
        seg_logits = self.segment_projector(features)  # (B, S_max, num_classes)
        seg_probs = F.softmax(seg_logits, dim=-1)
        seg_preds = seg_logits.argmax(dim=-1)

        return {
            "logits_guid": guid_logits,
            "probs_guid": guid_probs,
            "preds_guid": guid_preds,
            "logits": seg_logits,
            "probs": seg_probs,
            "preds": seg_preds,
            "mask": mask,
            "attention_weights": attn_weights,
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """GUID-level cross-entropy loss.

        Args:
            outputs: Dict from :meth:`forward`.
            batch: Dict from ``sequence_collate_fn``.

        Returns:
            Dict with ``loss``, ``accuracy``, ``class_0_acc``, ``class_1_acc``.
        """
        guid_labels = self._extract_guid_labels(batch)
        guid_logits = outputs["logits_guid"]

        loss = F.cross_entropy(guid_logits, guid_labels, weight=self.class_weights)
        metrics = self._compute_guid_metrics(guid_logits, guid_labels)

        return {"loss": loss, **metrics}


# ================================================================== #
#  Variant 2: TransMIL                                                 #
# ================================================================== #


class TransMILClassifier(BaseMILClassifier):
    """Transformer-based MIL classifier with self-attention between segments.

    Prepends a learnable CLS token to the segment sequence, processes via a
    standard ``nn.TransformerEncoder``, and classifies from the CLS output.
    Per-segment predictions are produced from each segment's transformer output.

    Args:
        vae_model: Pre-trained ``SeqVae`` instance.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_ff: Feedforward dimension in each transformer layer.
        d_model: Transformer model dimension (0 = match feature_dim).
        use_cls_token: If True, prepend a learnable CLS token.
        classifier_dropout: Dropout in the classifier head.
        mlp_multiplier: Hidden expansion factor for the classifier head.
        **kwargs: Forwarded to :class:`BaseMILClassifier`.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        d_model: int = 0,
        use_cls_token: bool = True,
        classifier_dropout: float = 0.1,
        mlp_multiplier: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(vae_model=vae_model, **kwargs)

        self.use_cls_token = use_cls_token
        self.d_model = d_model if d_model > 0 else self.feature_dim

        # Validate d_model is divisible by n_heads (required by
        # nn.TransformerEncoderLayer for multi-head attention).
        if self.d_model % n_heads != 0:
            raise ValueError(
                f"TransMIL d_model ({self.d_model}) must be divisible by "
                f"n_heads ({n_heads}).  feature_dim={self.feature_dim}.  "
                f"Set d_model explicitly to a multiple of n_heads."
            )

        # Input projection (if feature_dim != d_model)
        if self.feature_dim != self.d_model:
            self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        else:
            self.input_proj = nn.Identity()

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, self.d_model) * 0.02
            )

        # Learnable positional encoding
        max_segments = 50  # safety margin over typical max ~40
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_segments + 1, self.d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=classifier_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Classifier head (from CLS output)
        mlp_hidden = int(self.d_model * mlp_multiplier)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(mlp_hidden, self.num_classes),
        )

        # Per-segment projector (small MLP for sufficient capacity).
        self.segment_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.d_model // 2, self.num_classes),
        )

        # Fallback attention pool if no CLS token
        if not use_cls_token:
            self.attention_pool = GatedAttentionPooling(self.d_model, self.d_model // 2)

        logger.info(
            "TransMILClassifier created: d_model={}, n_heads={}, n_layers={}, "
            "d_ff={}, cls_token={}, segment_encoder={}, d_seg={}, classes={}",
            self.d_model, n_heads, n_layers, d_ff, use_cls_token,
            self.segment_encoder_type, self.d_seg, self.num_classes,
        )

    def _aggregate_and_classify(
        self,
        features: Tensor,
        mask: Tensor,
        lengths: Tensor,
    ) -> Dict[str, Tensor]:
        """TransMIL aggregation: self-attention + CLS classification.

        Args:
            features: ``(B, S_max, feature_dim)``.
            mask: ``(B, S_max)`` boolean.
            lengths: ``(B,)`` int.

        Returns:
            Dict with GUID-level and per-segment predictions.
        """
        B, S_max, _ = features.shape
        device = features.device

        # Project to d_model
        h = self.input_proj(features)  # (B, S_max, d_model)

        if self.use_cls_token:
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            h = torch.cat([cls_tokens, h], dim=1)  # (B, 1 + S_max, d_model)

            # Extend mask for CLS (always valid)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)  # (B, 1 + S_max)

            # Add positional encoding
            seq_len = h.size(1)
            h = h + self.pos_encoding[:, :seq_len, :]

            # Padding mask for transformer: True = IGNORE
            padding_mask = ~extended_mask  # (B, 1 + S_max)

            # Transformer encoder
            h = self.transformer(h, src_key_padding_mask=padding_mask)

            # CLS output → GUID classification
            cls_out = h[:, 0, :]  # (B, d_model)
            guid_logits = self.classifier_head(cls_out)

            # Segment outputs → per-segment predictions
            seg_out = h[:, 1:, :]  # (B, S_max, d_model)
            seg_logits = self.segment_head(seg_out)

            # Attention weights: use last layer's CLS-to-segment attention
            # For simplicity, compute proxy attention via softmax of seg_logits[:,:,1]
            attn_scores = seg_logits[:, :, 1].masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        else:
            # No CLS token — use attention pooling
            seq_len = h.size(1)
            h = h + self.pos_encoding[:, :seq_len, :]

            padding_mask = ~mask
            h = self.transformer(h, src_key_padding_mask=padding_mask)

            z, attn_weights = self.attention_pool(h, mask)
            guid_logits = self.classifier_head(z)
            seg_logits = self.segment_head(h)

        guid_probs = F.softmax(guid_logits, dim=-1)
        guid_preds = guid_logits.argmax(dim=-1)
        seg_probs = F.softmax(seg_logits, dim=-1)
        seg_preds = seg_logits.argmax(dim=-1)

        return {
            "logits_guid": guid_logits,
            "probs_guid": guid_probs,
            "preds_guid": guid_preds,
            "logits": seg_logits,
            "probs": seg_probs,
            "preds": seg_preds,
            "mask": mask,
            "attention_weights": attn_weights,
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """GUID-level cross-entropy loss.

        Args:
            outputs: Dict from :meth:`forward`.
            batch: Dict from ``sequence_collate_fn``.

        Returns:
            Dict with ``loss``, ``accuracy``, ``class_0_acc``, ``class_1_acc``.
        """
        guid_labels = self._extract_guid_labels(batch)
        guid_logits = outputs["logits_guid"]

        loss = F.cross_entropy(guid_logits, guid_labels, weight=self.class_weights)
        metrics = self._compute_guid_metrics(guid_logits, guid_labels)

        return {"loss": loss, **metrics}


# ================================================================== #
#  Variant 3: CausalMIL                                                #
# ================================================================== #


class CausalMILClassifier(BaseMILClassifier):
    """Causal attention MIL for early detection of fetal distress.

    Each segment can only attend to preceding segments (causal mask).
    A per-segment risk head produces a risk trajectory in ``[0, 1]``.
    GUID prediction aggregates risk scores via max, attention, or last-segment.

    A monotonicity regulariser encourages risk to increase (or stay stable)
    toward delivery for unhealthy GUIDs.

    Args:
        vae_model: Pre-trained ``SeqVae`` instance.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        d_ff: Feedforward dimension.
        d_model: Transformer model dimension (0 = match feature_dim).
        monotonicity_weight: Weight of the monotonicity regulariser.
        monotonicity_margin: Margin for the hinge loss.
        aggregation: How to aggregate for GUID prediction:
            ``'max_risk'``, ``'attention'``, or ``'last'``.
        classifier_dropout: Dropout in classifier head.
        mlp_multiplier: MLP hidden expansion factor.
        **kwargs: Forwarded to :class:`BaseMILClassifier`.
    """

    def __init__(
        self,
        vae_model: nn.Module,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        d_model: int = 0,
        monotonicity_weight: float = 0.1,
        monotonicity_margin: float = 0.0,
        aggregation: str = "max_risk",
        classifier_dropout: float = 0.1,
        mlp_multiplier: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(vae_model=vae_model, **kwargs)

        self.monotonicity_weight = monotonicity_weight
        self.monotonicity_margin = monotonicity_margin
        self.aggregation = aggregation
        self.d_model = d_model if d_model > 0 else self.feature_dim

        # Validate d_model is divisible by n_heads.
        if self.d_model % n_heads != 0:
            raise ValueError(
                f"CausalMIL d_model ({self.d_model}) must be divisible by "
                f"n_heads ({n_heads}).  feature_dim={self.feature_dim}.  "
                f"Set d_model explicitly to a multiple of n_heads."
            )

        # Input projection
        if self.feature_dim != self.d_model:
            self.input_proj = nn.Linear(self.feature_dim, self.d_model)
        else:
            self.input_proj = nn.Identity()

        # Positional encoding
        max_segments = 50
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_segments, self.d_model) * 0.02
        )

        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=classifier_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Per-segment risk head: scalar sigmoid risk score
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.d_model // 2, 1),
        )

        # GUID-level classifier head (from aggregated representation)
        mlp_hidden = int(self.d_model * mlp_multiplier)
        self.guid_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(mlp_hidden, self.num_classes),
        )

        # Attention pooling for 'attention' aggregation mode
        if aggregation == "attention":
            self.agg_attention = GatedAttentionPooling(
                self.d_model, self.d_model // 2,
            )

        logger.info(
            "CausalMILClassifier created: d_model={}, n_heads={}, n_layers={}, "
            "d_ff={}, mono_weight={}, mono_margin={}, aggregation={}, "
            "segment_encoder={}, d_seg={}, classes={}",
            self.d_model, n_heads, n_layers, d_ff,
            monotonicity_weight, monotonicity_margin, aggregation,
            self.segment_encoder_type, self.d_seg, self.num_classes,
        )

    def _make_causal_mask(self, S: int, device: torch.device) -> Tensor:
        """Create upper-triangular causal attention mask.

        Position j can attend to positions <= j.  Masked positions are ``True``
        (following PyTorch's convention for ``src_mask``).

        Args:
            S: Sequence length.
            device: Target device.

        Returns:
            Boolean mask ``(S, S)`` where ``True`` = block attention.
        """
        return torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1,
        )

    def _aggregate_and_classify(
        self,
        features: Tensor,
        mask: Tensor,
        lengths: Tensor,
    ) -> Dict[str, Tensor]:
        """CausalMIL: causal transformer → risk scores → GUID classification.

        Args:
            features: ``(B, S_max, feature_dim)``.
            mask: ``(B, S_max)`` boolean.
            lengths: ``(B,)`` int.

        Returns:
            Dict with GUID-level and per-segment predictions, risk scores,
            and attention weights.
        """
        B, S_max, _ = features.shape
        device = features.device

        # Project + positional encoding
        h = self.input_proj(features)
        h = h + self.pos_encoding[:, :S_max, :]

        # Causal + padding masks
        causal_mask = self._make_causal_mask(S_max, device)
        padding_mask = ~mask  # True = ignore

        # Causal transformer
        h = self.transformer(
            h, mask=causal_mask, src_key_padding_mask=padding_mask,
        )  # (B, S_max, d_model)

        # Per-segment risk scores
        risk_logits = self.risk_head(h).squeeze(-1)  # (B, S_max)
        risk_scores = torch.sigmoid(risk_logits)  # (B, S_max)
        risk_scores = risk_scores * mask.float()  # zero padded positions

        # GUID-level aggregation
        if self.aggregation == "max_risk":
            # Use the representation of the segment with highest risk
            risk_for_select = risk_scores.masked_fill(~mask, -1.0)
            max_idx = risk_for_select.argmax(dim=1)  # (B,)
            batch_idx = torch.arange(B, device=device)
            guid_repr = h[batch_idx, max_idx, :]  # (B, d_model)
        elif self.aggregation == "attention":
            guid_repr, _ = self.agg_attention(h, mask)  # (B, d_model)
        elif self.aggregation == "last":
            # Last valid segment
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(B, device=device)
            guid_repr = h[batch_idx, last_idx, :]  # (B, d_model)
        else:
            raise ValueError(f"Unknown aggregation '{self.aggregation}'")

        guid_logits = self.guid_head(guid_repr)  # (B, num_classes)
        guid_probs = F.softmax(guid_logits, dim=-1)
        guid_preds = guid_logits.argmax(dim=-1)

        # Per-segment predictions from risk scores: [1-risk, risk]
        seg_probs = torch.stack(
            [1.0 - risk_scores, risk_scores], dim=-1,
        )  # (B, S_max, 2)
        seg_logits = torch.log(seg_probs.clamp(min=1e-7))
        seg_preds = (risk_scores > 0.5).long()

        # Attention weights = risk scores (interpretable)
        attn_weights = risk_scores

        return {
            "logits_guid": guid_logits,
            "probs_guid": guid_probs,
            "preds_guid": guid_preds,
            "logits": seg_logits,
            "probs": seg_probs,
            "preds": seg_preds,
            "mask": mask,
            "attention_weights": attn_weights,
            "risk_scores": risk_scores,
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Hybrid loss: GUID-level CE + monotonicity regulariser.

        The monotonicity term penalises risk *decreases* between consecutive
        valid segments for unhealthy GUIDs only (risk should increase toward
        delivery for pathological cases).

        Args:
            outputs: Dict from :meth:`forward`.
            batch: Dict from ``sequence_collate_fn``.

        Returns:
            Dict with ``loss``, ``accuracy``, ``class_0_acc``, ``class_1_acc``,
            ``loss_guid``, ``loss_mono``.
        """
        guid_labels = self._extract_guid_labels(batch)
        guid_logits = outputs["logits_guid"]
        risk_scores = outputs["risk_scores"]  # (B, S_max)
        mask = outputs["mask"]

        # GUID-level CE
        loss_guid = F.cross_entropy(
            guid_logits, guid_labels, weight=self.class_weights,
        )

        # Monotonicity regulariser (unhealthy GUIDs only)
        loss_mono = torch.tensor(0.0, device=guid_logits.device)
        if self.monotonicity_weight > 0:
            unhealthy_mask = guid_labels == 1
            if unhealthy_mask.any():
                r = risk_scores[unhealthy_mask]  # (N_unhealthy, S_max)
                m = mask[unhealthy_mask]  # (N_unhealthy, S_max)

                # Adjacent valid segments: penalise risk decreases
                r_prev = r[:, :-1]
                r_next = r[:, 1:]
                m_valid = m[:, :-1] & m[:, 1:]

                violations = F.relu(
                    r_prev - r_next + self.monotonicity_margin
                )
                denom = m_valid.float().sum().clamp(min=1.0)
                loss_mono = (violations * m_valid.float()).sum() / denom

        loss = loss_guid + self.monotonicity_weight * loss_mono

        metrics = self._compute_guid_metrics(guid_logits, guid_labels)
        metrics["loss_guid"] = loss_guid.item()
        metrics["loss_mono"] = loss_mono.item()

        return {"loss": loss, **metrics}
