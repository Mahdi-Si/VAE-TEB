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

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from loguru import logger


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
        classifier_dropout: Dropout probability in the classifier head.
            Default 0.1.
        mlp_multiplier: MLP hidden expansion factor.  Default 2.0.
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
        num_classes: int = 2,
        classifier_dropout: float = 0.1,
        mlp_multiplier: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        vae_chunk_size: int = 32,
        use_posterior: bool = True,
        freeze_vae: bool = True,
        cnn_kernel: int = 7,
        debug: bool = False,
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
        self.num_classes = num_classes
        self.vae_chunk_size = vae_chunk_size
        self.use_posterior = use_posterior
        self.freeze_vae = freeze_vae
        self.debug = debug

        # -- VAE model (frozen) -------------------------------------------- #
        self.vae_model = vae_model
        if self.freeze_vae:
            self.vae_model.eval()
            for param in self.vae_model.parameters():
                param.requires_grad = False

        # -- Segment encoder ----------------------------------------------- #
        if segment_encoder_type == "mean_pool":
            # Mean-pool directly uses VAE latent dim; no extra parameters.
            self.d_seg = self._VAE_LATENT_DIM
        elif segment_encoder_type == "lstm":
            self.d_seg = d_seg
            self.segment_lstm = nn.LSTM(
                input_size=self._VAE_LATENT_DIM,
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
                nn.Conv1d(self._VAE_LATENT_DIM, d_seg, kernel_size=cnn_kernel, padding=0),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            raise ValueError(
                f"Unknown segment_encoder_type '{segment_encoder_type}'. "
                "Expected 'mean_pool', 'lstm', or 'cnn'."
            )

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

        # -- Standard temporal LSTM (used when gap_encoding != 'time_decay') #
        self.temporal_lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=temporal_lstm_hidden,
            num_layers=temporal_lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=temporal_lstm_dropout if temporal_lstm_layers > 1 else 0.0,
        )

        # -- Classifier head ----------------------------------------------- #
        mlp_hidden = int(temporal_lstm_hidden * mlp_multiplier)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(temporal_lstm_hidden),
            nn.Linear(temporal_lstm_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(mlp_hidden, num_classes),
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
            "gap={}, pos_dim={}, tlo={}, tlo_embed={}, dt_embed={}, "
            "persist_seg={}, seg_decay={}, classes={}",
            segment_encoder_type,
            self.d_seg,
            temporal_input_dim,
            temporal_lstm_hidden,
            temporal_lstm_layers,
            gap_encoding,
            position_embed_dim,
            tlo_enabled,
            tlo_embed_dim,
            delta_t_embed_dim,
            persist_segment_state,
            segment_state_decay,
            num_classes,
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
        _, (h_n, _) = self.segment_lstm(mu_valid)  # h_n: (1, N_valid, d_seg)
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
            _, (h_new, c_new) = self.segment_lstm(seg_input, (h, c))

            # -- Store segment vector -------------------------------------- #
            v[:, j, :] = h_new.squeeze(0)  # (B, d_seg)

            # -- Update state, zeroing invalid positions ------------------- #
            step_mask = mask[:, j].unsqueeze(0).unsqueeze(-1).float()  # (1, B, 1)
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

    def _encode_vae_chunked(
        self,
        fhr_st: Tensor,
        fhr_ph: Tensor,
        fhr_up_ph: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Encode all valid segments through the frozen VAE in chunks.

        Flattens ``(B, S_max, ...)`` to ``(B*S_max, ...)``, selects valid
        segments via *mask*, processes them in chunks of ``vae_chunk_size``,
        and scatters the results back into the full tensor.

        Args:
            fhr_st: Scattering features ``(B, S_max, 300, C_st)``.
            fhr_ph: Phase-harmonic features ``(B, S_max, 300, C_ph)``.
            fhr_up_ph: Cross-phase features ``(B, S_max, 300, C_x)`` where
                ``C_x`` is dynamic (depends on coefficient selection version).
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

        # Ensure VAE is in eval mode.
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
    ) -> Tensor:
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

        Returns:
            LSTM output ``(B, S_max, hidden_dim)``.
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

        return torch.stack(outputs, dim=1)  # (B, S_max, hidden_dim)

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
        if self.gap_encoding in ("time_decay", "both"):
            # Custom loop with time-decay gating on hidden/cell state.
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

        if self.debug:
            assert not torch.isnan(h).any(), "NaN in temporal LSTM output"

        # ---- Step 7: Per-segment classification -------------------------- #
        logits = self.classifier_head(h)        # (B, S_max, num_classes)
        probs = F.softmax(logits, dim=-1)       # (B, S_max, num_classes)
        preds = logits.argmax(dim=-1)           # (B, S_max)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "mask": mask,
        }

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute masked cross-entropy loss over valid segments.

        Per-segment labels are derived by taking ``target.max(dim=-1)`` over
        the 300 within-segment timesteps and mapping to binary:
        ``{0, 1} → 0 (healthy)``, ``{2, 3} → 1 (unhealthy)``.

        Only valid (non-padded) segments contribute to the loss, enforced by
        the ``mask`` tensor.

        Note:
            Segments where all timesteps have ``weight=0`` yield
            ``target.max()=0``, which maps to ``binary_labels=0`` (healthy).
            This is acceptable because such segments are extremely rare
            (quality filter requires >90% weight) and genuinely uninformative.

        Args:
            outputs: Dict from :meth:`forward` containing ``logits`` and
                ``mask``.
            batch: Dict from ``sequence_collate_fn`` containing ``target``
                ``(B, S_max, 300)``.

        Returns:
            Dict with keys:
                ``loss`` (scalar tensor),
                ``accuracy`` (float),
                ``class_0_acc`` (float),
                ``class_1_acc`` (float).
        """
        logits = outputs["logits"]   # (B, S_max, num_classes)
        mask = outputs["mask"]       # (B, S_max) bool
        target = batch["target"]     # (B, S_max, 300)

        # Extract valid-segment logits and targets.
        logits_valid = logits[mask]                     # (N_valid, num_classes)
        target_valid = target[mask]                     # (N_valid, 300)

        # Guard: all segments padded → return zero loss.
        if logits_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=logits.device)
            return {
                "loss": zero,
                "accuracy": 0.0,
                "class_0_acc": 0.0,
                "class_1_acc": 0.0,
            }

        # Per-segment label: max over 300 timesteps → class ID {0,1,2,3}.
        seg_labels = target_valid.max(dim=-1)[0]        # (N_valid,)
        binary_labels = (seg_labels > 1).long()         # (N_valid,) {0,1}

        # Cross-entropy with optional class weights.
        loss = F.cross_entropy(
            logits_valid, binary_labels, weight=self.class_weights,
        )

        # Metrics.
        preds_valid = logits_valid.argmax(dim=-1)       # (N_valid,)
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
