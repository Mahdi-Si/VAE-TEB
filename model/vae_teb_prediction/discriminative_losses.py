"""Discriminative loss modules for class-aware VAE-TEB fine-tuning.

This module provides two components that augment the standard VAE-TEB objective
with class-discriminative signals:

1. **TemporalCenterLoss** — pulls per-timestep latent vectors toward learnable
   class centroids (intra-class compactness).
2. **AuxiliaryClassifierHead** — attention-weighted temporal pooling followed by
   a lightweight MLP that produces inter-class separation gradients.

Together they enable phased fine-tuning of a pretrained SeqVae so that distinct
outcome classes (e.g. HEALTHY vs UNHEALTHY, or HEALTHY/ACIDOSIS/HIE) occupy
separable regions of the latent space while preserving reconstruction quality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCenterLoss(nn.Module):
    """Per-timestep center loss with EMA-updated class centroids.

    Maintains a set of class centroids as non-learnable buffers and computes
    the average squared distance between each latent vector and the centroid
    of its assigned class.  Centroids are updated via exponential moving
    average (EMA) during training.

    Attributes:
        num_classes: Number of outcome classes (default 3).
        latent_dim: Dimensionality of the latent space (default 16).
        ema_decay: Decay factor for centroid EMA updates.
        centers: Buffer of shape ``(num_classes, latent_dim)`` holding current
            centroid positions.
    """

    def __init__(
        self,
        num_classes: int = 3,
        latent_dim: int = 16,
        ema_decay: float = 0.99,
    ) -> None:
        """Initialize TemporalCenterLoss.

        Args:
            num_classes: Number of distinct outcome classes.
            latent_dim: Dimensionality of latent vectors ``z_t``.
            ema_decay: EMA decay coefficient for centroid updates.  Higher
                values make centroids more stable (recommended >=0.99 when
                there are many timesteps per batch).
        """
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.register_buffer(
            "centers", torch.zeros(num_classes, latent_dim)
        )
        self.register_buffer(
            "centers_initialized", torch.zeros(num_classes, dtype=torch.bool)
        )

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        warmup_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute temporal center loss and update centroids via EMA.

        Args:
            z: Latent vectors of shape ``(B, T, D)`` where *D* equals
                ``latent_dim``.
            labels: Per-sample 0-indexed class indices of shape ``(B,)``
                with integer values in ``{0, ..., num_classes - 1}``.
                The caller is responsible for mapping raw dataset labels
                to this range.
            warmup_mask: Optional boolean mask of shape ``(T,)`` where
                ``True`` indicates a valid (post-warmup) timestep.  If
                ``None`` all timesteps are used.

        Returns:
            Scalar center loss averaged over valid timesteps and samples.
        """
        B, T, _ = z.shape

        class_ids = labels.long()  # (B,)

        # Expand class ids to every timestep: (B,) -> (B, T)
        class_ids_exp = class_ids.unsqueeze(1).expand(B, T)

        # Gather the centroid for each (sample, timestep)
        # centers: (C, D) -> index by class_ids_exp: (B, T) -> (B, T, D)
        selected_centers = self.centers[class_ids_exp]  # (B, T, D)

        # Compute squared distance
        diff = z - selected_centers.detach()  # detach centers for loss
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, T)

        # Apply warmup mask
        if warmup_mask is not None:
            sq_dist = sq_dist * warmup_mask.unsqueeze(0).float()
            n_valid = warmup_mask.sum().clamp_min(1).float() * B
        else:
            n_valid = float(B * T)

        loss = 0.5 * sq_dist.sum() / n_valid

        # Update centers via EMA during training
        if self.training:
            self._update_centers(z.detach(), class_ids, warmup_mask)

        return loss

    @torch.no_grad()
    def _update_centers(
        self,
        z: torch.Tensor,
        class_ids: torch.Tensor,
        warmup_mask: torch.Tensor | None,
    ) -> None:
        """Update class centroids using exponential moving average.

        Args:
            z: Detached latent vectors of shape ``(B, T, D)``.
            class_ids: 0-indexed class indices of shape ``(B,)`` in
                ``{0, ..., num_classes - 1}``.
            warmup_mask: Optional boolean mask of shape ``(T,)``.
        """
        alpha = self.ema_decay

        for c in range(self.num_classes):
            mask_b = (class_ids == c)  # (B,)
            if not mask_b.any():
                continue

            z_c = z[mask_b]  # (N_c, T, D)

            if warmup_mask is not None:
                # Only average over valid timesteps
                z_valid = z_c[:, warmup_mask]  # (N_c, T_valid, D)
            else:
                z_valid = z_c

            if z_valid.numel() == 0:
                continue

            new_mean = z_valid.mean(dim=(0, 1))  # (D,)

            if not self.centers_initialized[c]:
                # First time seeing this class — initialize directly
                self.centers[c] = new_mean
                self.centers_initialized[c] = True
            else:
                self.centers[c] = alpha * self.centers[c] + (1 - alpha) * new_mean

    def get_center_distances(self) -> torch.Tensor:
        """Compute pairwise L2 distances between all class centroids.

        Returns:
            Symmetric matrix of shape ``(num_classes, num_classes)`` with
            pairwise Euclidean distances.
        """
        return torch.cdist(
            self.centers.unsqueeze(0), self.centers.unsqueeze(0)
        ).squeeze(0)


class AuxiliaryClassifierHead(nn.Module):
    """Lightweight attention-pooling classifier for inter-class separation.

    Performs attention-weighted temporal pooling over latent vectors, then
    classifies via a small 2-layer MLP.  Designed to be intentionally small
    (~5K parameters) so its gradient signal does not overpower the VAE
    reconstruction objective.

    Attributes:
        attn_linear: Linear projection for computing attention scores.
        mlp: Two-layer MLP that maps pooled features to class logits.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        num_classes: int = 3,
        hidden_dim: int = 32,
    ) -> None:
        """Initialize the auxiliary classifier head.

        Args:
            latent_dim: Dimensionality of input latent vectors.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension of the MLP (kept small to limit
                parameter count).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Attention scoring: project D -> 1 for attention weights
        self.attn_linear = nn.Linear(latent_dim, 1)

        # 2-layer MLP: D -> hidden_dim -> num_classes
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        z: torch.Tensor,
        warmup_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: attention pool then classify.

        Args:
            z: Latent vectors of shape ``(B, T, D)``.
            warmup_mask: Optional boolean mask of shape ``(T,)`` where
                ``True`` marks valid timesteps.  Warmup timesteps are
                excluded from the attention pool.

        Returns:
            Dictionary containing:
                - ``logits``: Raw classification scores, shape ``(B, C)``.
                - ``probs``: Softmax probabilities, shape ``(B, C)``.
                - ``preds``: Predicted class indices, shape ``(B,)``.
        """
        # Compute attention scores
        attn_scores = self.attn_linear(z).squeeze(-1)  # (B, T)

        # Mask out warmup timesteps
        if warmup_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~warmup_mask.unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T)

        # Weighted sum pooling
        pooled = torch.bmm(
            attn_weights.unsqueeze(1), z
        ).squeeze(1)  # (B, D)

        # Classify
        logits = self.mlp(pooled)  # (B, C)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {"logits": logits, "probs": probs, "preds": preds}
