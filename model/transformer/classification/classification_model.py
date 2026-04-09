"""Time-aware GRU classifier for GUID-level binary classification.

Implements the architecture from ``classification_model.md``: a frozen (or
fine-tunable) ``CausalMultimodalTransformer`` produces 416-dim segment
embeddings, which are processed by a time-aware GRU with decay gating to
produce per-segment binary predictions (healthy vs unhealthy).

Segment embedding composition (416-dim)::

    e = [s_F(192) | s_FU(192) | mean_TE(16) | std_TE(16)]

where:
    - s_F: pooled FHR-only causal states (full 300-step sequence)
    - s_FU: pooled fused multimodal states (full 300-step sequence)
    - mean_TE: average TE posterior means across dense anchor grid
    - std_TE: standard deviation of TE posterior means (coupling variability)

The classifier concatenates each segment's embedding, its delta from the
previous segment, and a learned time embedding, then processes the sequence
through a GRU with time-decay gating before a binary prediction head.

Example::

    from model.transformer.model import CausalMultimodalTransformer
    from train.graph_models_utils import load_checkpoint_strict

    transformer = CausalMultimodalTransformer()
    load_checkpoint_strict(transformer, "/path/to/transformer.ckpt")

    model = TimeAwareGRUClassifier(
        transformer_model=transformer,
        freeze_strategy="frozen",
    )

    outputs = model(batch)  # batch from sequence_collate_fn
    loss_dict = model.compute_loss(outputs, batch)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loguru import logger


# ------------------------------------------------------------------ #
#  Time Feature Encoder                                                #
# ------------------------------------------------------------------ #


class TimeFeatureEncoder(nn.Module):
    """Encode raw time features into a learned embedding.

    Computes a 6-dim raw time feature vector from time-from-labour-onset
    (TLO) and inter-segment gap (delta_t), then passes it through a
    small MLP to produce a learned embedding.

    Raw feature vector r (per segment)::

        r = [tau, log(1+tau), delta_tau, log(1+delta_tau), delta, m]

    where:
        - tau = TLO in minutes (NaN → 0)
        - delta_tau = inter-segment gap in minutes
        - delta = gap deviation from nominal (minutes)
        - m = missingness indicator (1 if gap > threshold)

    Args:
        embed_dim: Output embedding dimension.
        nominal_gap_minutes: Expected inter-segment gap for deviation
            computation.
        gap_threshold_minutes: Threshold above which the missingness
            indicator fires.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        nominal_gap_minutes: float = 20.0,
        gap_threshold_minutes: float = 22.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.nominal_gap_minutes = nominal_gap_minutes
        self.gap_threshold_minutes = gap_threshold_minutes

        self.mlp = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        time_from_labor_onset: Tensor,
        delta_t: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Compute time embedding from raw temporal features.

        Args:
            time_from_labor_onset: TLO in seconds ``(B, S_max)``.  May
                contain ``NaN`` when TLO is unavailable.
            delta_t: Inter-segment gap in seconds ``(B, S_max)``.
            mask: Boolean validity mask ``(B, S_max)``.

        Returns:
            Time embedding ``(B, S_max, embed_dim)``.  Padded positions
            are zeroed.  Guaranteed NaN-free.
        """
        # Convert to minutes for scale stability.
        tlo_minutes = time_from_labor_onset / 60.0
        # Replace NaN with 0 — the model learns from tau=0 + gap context.
        tlo_minutes = torch.where(
            torch.isnan(tlo_minutes),
            torch.zeros_like(tlo_minutes),
            tlo_minutes,
        )

        delta_tau_minutes = delta_t / 60.0

        # Raw 6-dim feature vector.
        tau = tlo_minutes
        log_tau = torch.log1p(tau.clamp(min=0.0))
        delta_tau = delta_tau_minutes
        log_delta_tau = torch.log1p(delta_tau.clamp(min=0.0))
        delta = delta_tau - self.nominal_gap_minutes
        m = (delta_tau > self.gap_threshold_minutes).float()

        # (B, S_max, 6)
        raw_features = torch.stack(
            [tau, log_tau, delta_tau, log_delta_tau, delta, m], dim=-1
        )

        # MLP: (B, S_max, 6) → (B, S_max, embed_dim)
        time_embed = self.mlp(raw_features)

        # Zero out padded positions.
        time_embed = time_embed * mask.unsqueeze(-1).float()
        return time_embed


# ------------------------------------------------------------------ #
#  Time-Aware GRU Classifier                                           #
# ------------------------------------------------------------------ #


class TimeAwareGRUClassifier(nn.Module):
    """Time-aware GRU classifier for binary segment-level prediction.

    Operates on 416-dim segment embeddings extracted from a pretrained
    ``CausalMultimodalTransformer``.  Supports both on-the-fly encoding
    (transformer forward per batch) and precomputed embeddings loaded
    from HDF5.

    Architecture overview::

        embeddings (416) + deltas (416) + time_embed (32) = 864
        → input_proj (864 → 256)
        → time-decay GRU loop (hidden 256)
        → output feature [h | x] (512)
        → binary head (512 → 1 or 2)

    Args:
        transformer_model: Pretrained ``CausalMultimodalTransformer``.
            Pass ``None`` when using precomputed embeddings exclusively.
        d_embedding: Segment embedding dimension (default 416).
        time_embed_dim: Time MLP output dimension (default 32).
        input_proj_dim: Input projection output dimension (default 256).
        gru_hidden_dim: GRU hidden state dimension (default 256).
        dropout: Dropout probability (default 0.1).
        loss_type: ``"bce"`` for BCEWithLogitsLoss or ``"ce"`` for
            CrossEntropyLoss with 2 classes.
        pos_weight: Positive class weight for BCE loss.  Used when
            ``loss_type="bce"`` and class balancing is enabled.
        class_weights: Per-class weights for CE loss.  Used when
            ``loss_type="ce"`` and class balancing is enabled.
        label_smoothing: Label smoothing factor (0 = disabled).
        transformer_chunk_size: Segments per transformer forward chunk.
        freeze_strategy: One of ``"frozen"``, ``"trainable"``,
            ``"phased"``.
        pooling: ``"mean"`` for mean pooling or ``"attention"`` for
            trainable ``AttentionPool`` over H_F and H_FU.
        anchor_step: Step size for the TE anchor grid (default 5).
        nominal_gap_minutes: Expected gap between consecutive segments.
        gap_threshold_minutes: Gap threshold for missingness indicator.
    """

    def __init__(
        self,
        transformer_model: Optional[nn.Module] = None,
        d_embedding: int = 416,
        time_embed_dim: int = 32,
        input_proj_dim: int = 256,
        gru_hidden_dim: int = 256,
        dropout: float = 0.1,
        loss_type: str = "bce",
        pos_weight: Optional[float] = None,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        transformer_chunk_size: int = 16,
        freeze_strategy: str = "frozen",
        pooling: str = "mean",
        anchor_step: int = 5,
        nominal_gap_minutes: float = 20.0,
        gap_threshold_minutes: float = 22.0,
    ) -> None:
        super().__init__()

        self.d_embedding = d_embedding
        self.time_embed_dim = time_embed_dim
        self.input_proj_dim = input_proj_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.transformer_chunk_size = transformer_chunk_size
        self.freeze_strategy = freeze_strategy
        self.pooling = pooling
        self.anchor_step = anchor_step

        # --- Transformer (may be None for precomputed mode) -----------
        self.transformer = transformer_model
        if self.transformer is not None:
            self._apply_freeze_strategy()

        # --- Trainable attention pools (only for pooling="attention") --
        if pooling == "attention":
            from model.transformer.model.layers import AttentionPool
            self.pool_f = AttentionPool(192)
            self.pool_fu = AttentionPool(192)

        # --- Time feature encoder -------------------------------------
        self.time_encoder = TimeFeatureEncoder(
            embed_dim=time_embed_dim,
            nominal_gap_minutes=nominal_gap_minutes,
            gap_threshold_minutes=gap_threshold_minutes,
        )

        # --- Input projection -----------------------------------------
        cat_dim = d_embedding + d_embedding + time_embed_dim  # 864
        self.input_proj = nn.Sequential(
            nn.Linear(cat_dim, input_proj_dim),
            nn.LayerNorm(input_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Time-decay gate ------------------------------------------
        self.decay_gate = nn.Linear(time_embed_dim, gru_hidden_dim)

        # --- GRU cell -------------------------------------------------
        self.gru_cell = nn.GRUCell(input_proj_dim, gru_hidden_dim)

        # --- Binary prediction head -----------------------------------
        output_dim = gru_hidden_dim + input_proj_dim  # 512
        if loss_type == "bce":
            self.head = nn.Linear(output_dim, 1)
        elif loss_type == "ce":
            self.head = nn.Linear(output_dim, 2)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

        # --- Loss weights (registered as buffers) ---------------------
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        logger.info(
            "TimeAwareGRUClassifier — d_emb={}, proj={}, gru={}, "
            "loss={}, pooling={}, anchor_step={}",
            d_embedding, input_proj_dim, gru_hidden_dim,
            loss_type, pooling, anchor_step,
        )

    # ------------------------------------------------------------------ #
    #  Freeze / Unfreeze                                                   #
    # ------------------------------------------------------------------ #

    def _apply_freeze_strategy(self) -> None:
        """Apply the initial freeze strategy to the transformer."""
        if self.freeze_strategy in ("frozen", "phased"):
            self.freeze_transformer()
        elif self.freeze_strategy == "trainable":
            pass  # Leave all parameters trainable.
        else:
            raise ValueError(
                f"Unknown freeze_strategy: {self.freeze_strategy!r}"
            )

    def freeze_transformer(self) -> None:
        """Freeze all transformer parameters."""
        if self.transformer is None:
            return
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.transformer.eval()
        logger.info("Transformer frozen ({:,} params)",
                     sum(p.numel() for p in self.transformer.parameters()))

    def unfreeze_transformer(self) -> None:
        """Unfreeze all transformer parameters for fine-tuning."""
        if self.transformer is None:
            return
        for param in self.transformer.parameters():
            param.requires_grad = True
        self.transformer.train()
        logger.info("Transformer unfrozen ({:,} params)",
                     sum(p.numel() for p in self.transformer.parameters()))

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass producing per-segment predictions.

        Args:
            batch: Dict from ``sequence_collate_fn`` with keys:

                - ``mask`` ``(B, S_max)`` bool
                - ``lengths`` ``(B,)`` int
                - ``delta_t`` ``(B, S_max)`` float seconds
                - ``time_from_labor_onset`` ``(B, S_max)`` float seconds
                - ``fhr_st`` ``(B, S_max, 300, 43)`` (if not precomputed)
                - ``up_st`` ``(B, S_max, 300, 43)`` (if not precomputed)
                - ``embeddings_precomputed`` ``(B, S_max, 416)`` (optional)

        Returns:
            Dict with keys:

                - ``logits`` ``(B, S_max)`` or ``(B, S_max, 2)``
                - ``probs`` ``(B, S_max)``
                - ``preds`` ``(B, S_max)``
                - ``mask`` ``(B, S_max)``
        """
        mask = batch["mask"]                        # (B, S_max)
        delta_t = batch["delta_t"]                  # (B, S_max)

        # NaN firewall on TLO.
        tlo = batch.get("time_from_labor_onset")
        if tlo is None:
            tlo = torch.zeros_like(delta_t)

        # Step 1: Segment embeddings (B, S_max, 416).
        if "embeddings_precomputed" in batch:
            embeddings = batch["embeddings_precomputed"]
        else:
            embeddings = self._encode_transformer_chunked(batch)

        # Step 2: Segment deltas (B, S_max, 416).
        delta_e = torch.zeros_like(embeddings)
        delta_e[:, 1:, :] = embeddings[:, 1:, :] - embeddings[:, :-1, :]
        delta_e = delta_e * mask.unsqueeze(-1).float()

        # Step 3: Time embedding (B, S_max, 32).
        time_embed = self.time_encoder(tlo, delta_t, mask)

        # Step 4: Concatenated classifier token (B, S_max, 864).
        x_cat = torch.cat([embeddings, delta_e, time_embed], dim=-1)

        # Step 5: Input projection (B, S_max, 256).
        x = self.input_proj(x_cat)

        # Step 6: Time-decay gate (B, S_max, 256).
        gamma = torch.exp(-F.softplus(self.decay_gate(time_embed)))

        # Step 7: GRU loop with time-decay gating.
        h_all = self._gru_loop(x, gamma, mask)     # (B, S_max, 256)

        # Step 8: Output feature [h | x] (B, S_max, 512).
        o = torch.cat([h_all, x], dim=-1)

        # Step 9: Prediction head.
        raw = self.head(o)                          # (B, S_max, 1) or (B, S_max, 2)

        if self.loss_type == "bce":
            logits = raw.squeeze(-1)                # (B, S_max)
            probs = torch.sigmoid(logits)
            preds = (logits > 0.0).long()
        else:
            logits = raw                            # (B, S_max, 2)
            probs = F.softmax(logits, dim=-1)[:, :, 1]  # P(unhealthy)
            preds = logits.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "mask": mask,
        }

    def _gru_loop(
        self,
        x: Tensor,
        gamma: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Run the time-decay GRU loop.

        At each step j:
            1. Decay previous hidden: h_tilde = gamma_j * h
            2. GRU update: h = GRUCell(x_j, h_tilde)
            3. Zero out padded positions.

        Args:
            x: Projected tokens ``(B, S_max, input_proj_dim)``.
            gamma: Decay gates ``(B, S_max, gru_hidden_dim)``.
            mask: Boolean mask ``(B, S_max)``.

        Returns:
            Stacked hidden states ``(B, S_max, gru_hidden_dim)``.
        """
        B, S_max, _ = x.shape
        device = x.device

        if S_max == 0:
            return torch.zeros(B, 0, self.gru_hidden_dim, device=device)

        h = torch.zeros(B, self.gru_hidden_dim, device=device)
        h_list: List[Tensor] = []

        for j in range(S_max):
            h_tilde = gamma[:, j, :] * h            # decay
            h = self.gru_cell(x[:, j, :], h_tilde)  # update
            h = h * mask[:, j].unsqueeze(-1).float() # zero padding
            h_list.append(h)

        return torch.stack(h_list, dim=1)

    # ------------------------------------------------------------------ #
    #  Transformer Encoding                                                #
    # ------------------------------------------------------------------ #

    def _encode_transformer_chunked(
        self,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """Encode segments through the transformer in chunks.

        For each valid segment, runs the transformer forward pass and
        extracts a 416-dim embedding:
            [pool(H_F)(192) | pool(H_FU)(192) | mean_TE(16) | std_TE(16)]

        Note:
            Gradient flow is controlled by a context manager inside this
            method, not a decorator.  When ``freeze_strategy`` is
            ``"trainable"`` or ``"phased"`` (after unfreezing),
            ``torch.enable_grad()`` is used so gradients propagate to
            the transformer.  Otherwise ``torch.no_grad()`` is used.

        Args:
            batch: Batch dict containing ``fhr_st``, ``up_st``, ``mask``,
                ``lengths``.

        Returns:
            Segment embeddings ``(B, S_max, d_embedding)``.
        """
        if self.transformer is None:
            raise RuntimeError(
                "Transformer model is None but precomputed embeddings "
                "were not provided in the batch."
            )

        transformer = self.transformer  # Narrow type for Pyright.

        # Determine if gradients should flow (for trainable/phased mode).
        needs_grad = (
            self.freeze_strategy == "trainable"
            or (self.freeze_strategy == "phased" and transformer.training)
        )
        ctx = torch.enable_grad() if needs_grad else torch.no_grad()

        fhr_st = batch["fhr_st"]  # (B, S_max, 300, 43)
        up_st = batch["up_st"]    # (B, S_max, 300, 43)
        mask = batch["mask"]      # (B, S_max)

        B, S_max, T, d_f = fhr_st.shape
        device = fhr_st.device

        # Build dense TE anchor grid.
        cfg = transformer.config
        grid = torch.arange(
            cfg.valid_anchor_start,
            cfg.valid_anchor_end + 1,
            self.anchor_step,
            device=device,
        )
        K = grid.shape[0]

        # Flatten valid segments.
        valid_mask = mask.view(-1)               # (B * S_max)
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        n_valid = valid_indices.shape[0]

        if n_valid == 0:
            return torch.zeros(B, S_max, self.d_embedding, device=device)

        fhr_flat = fhr_st.view(B * S_max, T, d_f)
        up_flat = up_st.view(B * S_max, T, d_f)

        Y_valid = fhr_flat[valid_indices]  # (n_valid, 300, 43)
        U_valid = up_flat[valid_indices]   # (n_valid, 300, 43)

        # Process in chunks.
        embeddings_valid = []
        chunk_size = self.transformer_chunk_size

        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            Y_chunk = Y_valid[start:end]
            U_chunk = U_valid[start:end]
            C = Y_chunk.shape[0]

            grid_chunk = grid.unsqueeze(0).expand(C, -1)

            with ctx:
                outputs = transformer(Y_chunk, U_chunk,
                                      anchor_indices=grid_chunk)

            H_F = outputs["H_F"]         # (C, 300, 192)
            H_FU = outputs["H_FU"]       # (C, 300, 192)
            mu_post = outputs["mu_post"]  # (C * K, 16)

            # Pool H_F and H_FU (full-sequence, anchor-independent).
            if self.pooling == "mean":
                s_F = H_F.mean(dim=1)    # (C, 192)
                s_FU = H_FU.mean(dim=1)  # (C, 192)
            else:
                s_F = self.pool_f(H_F)   # (C, 192)
                s_FU = self.pool_fu(H_FU)  # (C, 192)

            # TE statistics from dense anchor grid.
            te_mus = mu_post.view(C, K, -1)     # (C, K, 16)
            mean_te = te_mus.mean(dim=1)         # (C, 16)
            std_te = te_mus.std(dim=1)           # (C, 16)

            emb = torch.cat([s_F, s_FU, mean_te, std_te], dim=-1)
            embeddings_valid.append(emb)

        embeddings_valid = torch.cat(embeddings_valid, dim=0)  # (n_valid, 416)

        # Scatter back into (B * S_max, d_embedding).
        all_embeddings = torch.zeros(
            B * S_max, self.d_embedding, device=device,
        )
        all_embeddings[valid_indices] = embeddings_valid

        return all_embeddings.view(B, S_max, self.d_embedding)

    # ------------------------------------------------------------------ #
    #  Loss Computation                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute classification loss over valid (non-padding) segments.

        Args:
            outputs: Dict from :meth:`forward` with ``logits`` and ``mask``.
            batch: Original batch dict with ``target`` field.

        Returns:
            Dict with keys ``loss``, ``accuracy``, ``class_0_acc``,
            ``class_1_acc``.
        """
        mask = outputs["mask"]                      # (B, S_max)
        target = batch["target"]                    # (B, S_max, 300)

        # Per-segment binary labels: max target value > 1 = unhealthy.
        seg_labels_raw = target.max(dim=-1).values  # (B, S_max)
        binary_labels = (seg_labels_raw > 1).float()  # {0.0, 1.0}

        # Select valid (non-padding) segments.
        logits_valid = outputs["logits"][mask]       # (N,) or (N, 2)
        labels_valid = binary_labels[mask]           # (N,)

        if logits_valid.numel() == 0:
            zero = torch.tensor(0.0, device=mask.device, requires_grad=True)
            return {
                "loss": zero,
                "accuracy": torch.tensor(0.0, device=mask.device),
                "class_0_acc": torch.tensor(0.0, device=mask.device),
                "class_1_acc": torch.tensor(0.0, device=mask.device),
            }

        # Compute loss.
        if self.loss_type == "bce":
            targets_for_loss = labels_valid
            if self.label_smoothing > 0:
                alpha = self.label_smoothing
                targets_for_loss = targets_for_loss * (1 - alpha) + 0.5 * alpha
            loss = F.binary_cross_entropy_with_logits(
                logits_valid, targets_for_loss,
                pos_weight=self.pos_weight,
            )
            preds_valid = (logits_valid > 0.0).long()
        else:
            labels_long = labels_valid.long()
            loss = F.cross_entropy(
                logits_valid, labels_long,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            preds_valid = logits_valid.argmax(dim=-1)

        # Accuracy metrics.
        labels_long = labels_valid.long()
        correct = (preds_valid == labels_long).float()
        accuracy = correct.mean()

        mask_0 = labels_long == 0
        mask_1 = labels_long == 1
        class_0_acc = correct[mask_0].mean() if mask_0.any() else torch.tensor(
            0.0, device=mask.device
        )
        class_1_acc = correct[mask_1].mean() if mask_1.any() else torch.tensor(
            0.0, device=mask.device
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "class_0_acc": class_0_acc,
            "class_1_acc": class_1_acc,
        }
