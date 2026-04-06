"""Top-level Causal Multimodal Forecasting Transformer (v2).

Contains:
    - sample_anchors: Standalone utility for anchor index sampling.
    - CausalMultimodalTransformer: The main nn.Module wiring all components.
    - CausalTransformerLoss: Loss computation module (spec §16, v2 extensions).
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TransformerConfig
from .encoder import (
    CausalCrossAttentionFusion,
    CausalTransformerEncoder,
)
from .heads import (
    ForecastHead,
    SelfLatentModule,
    TELatentModule,
    WindowRepresentationExport,
)
from .layers import AttentionPool, _make_norm
from .stems import CausalStem


# ---------------------------------------------------------------------------
# Anchor sampling utility
# ---------------------------------------------------------------------------

def sample_anchors(
    Y: Tensor,
    U: Tensor,
    config: TransformerConfig,
    training: bool = True,
) -> Tensor:
    """Sample anchor indices for prediction (spec §10).

    During training, anchors are sampled from a 50/50 mixture of uniform and
    activity-biased distributions.  During evaluation, a fixed grid (every 15
    steps) is returned for deterministic window-level representation export.

    Args:
        Y: FHR scattering features of shape ``(B, T, d_F)``.
        U: UP scattering features of shape ``(B, T, d_U)``.
        config: Transformer configuration.
        training: Whether the model is in training mode.

    Returns:
        Anchor indices of shape ``(B, K)`` as a LongTensor on the same device
        as ``Y``.
    """
    B = Y.shape[0]
    device = Y.device
    a_start = config.valid_anchor_start
    a_end = config.valid_anchor_end
    valid_range = a_end - a_start + 1

    if not training:
        # Fixed grid every 15 steps for deterministic inference
        grid = torch.arange(a_start, a_end + 1, 15, device=device)
        return grid.unsqueeze(0).expand(B, -1)  # (B, K_grid)

    K = config.num_anchors
    eta = config.anchor_uniform_ratio

    # --- Activity scores (spec §10.4) ---
    # s_t = |U_t|_1 + |delta U_t|_1 + |delta Y_t|_1
    U_valid = U[:, a_start:a_end + 1, :]             # (B, R, d_U)
    Y_valid = Y[:, a_start:a_end + 1, :]             # (B, R, d_F)

    score = U_valid.abs().sum(dim=-1)                 # (B, R)
    if valid_range > 1:
        delta_U = torch.diff(U_valid, dim=1).abs().sum(dim=-1)  # (B, R-1)
        delta_Y = torch.diff(Y_valid, dim=1).abs().sum(dim=-1)  # (B, R-1)
        # Pad first position with 0 to keep shape (B, R)
        delta_U = F.pad(delta_U, (1, 0), value=0.0)
        delta_Y = F.pad(delta_Y, (1, 0), value=0.0)
        score = score + delta_U + delta_Y

    # --- Mixture sampling ---
    # Normalize activity scores to a probability distribution
    score_sum = score.sum(dim=-1, keepdim=True)
    score_prob = score / (score_sum + 1e-8)                      # (B, R)

    # Uniform component
    uniform_prob = torch.ones(B, valid_range, device=device) / valid_range

    # Mixed distribution — fall back to uniform when scores are degenerate
    mixed_prob = eta * uniform_prob + (1.0 - eta) * score_prob   # (B, R)
    # Guard against all-zero rows (e.g. eta=0 and constant input)
    row_sums = mixed_prob.sum(dim=-1, keepdim=True)
    mixed_prob = torch.where(
        row_sums > 1e-8, mixed_prob, uniform_prob,
    )

    # Sample K anchors per batch element
    indices_in_range = torch.multinomial(mixed_prob, K, replacement=False)  # (B, K)
    anchors = indices_in_range + a_start  # shift to absolute positions

    return anchors


def validate_anchor_indices(
    anchor_indices: Tensor,
    config: TransformerConfig,
) -> None:
    """Validate that anchor indices are within the legal range.

    Call this **before** passing anchors to the model's ``forward()`` method.
    This runs on the CPU and is safe for use outside ``torch.compile`` graphs.

    Args:
        anchor_indices: Anchor positions of shape ``(B, K)``.
        config: Transformer configuration.

    Raises:
        ValueError: If any anchor is outside
            ``[config.valid_anchor_start, config.valid_anchor_end]``.
    """
    a_min = anchor_indices.min().item()
    a_max = anchor_indices.max().item()
    if a_min < config.valid_anchor_start or a_max > config.valid_anchor_end:
        raise ValueError(
            f"anchor_indices out of valid range "
            f"[{config.valid_anchor_start}, {config.valid_anchor_end}]: "
            f"got min={a_min}, max={a_max}"
        )


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    """Initialize weights following existing project conventions.

    - Linear / Conv1d: Xavier uniform, zeros bias.
    - LayerNorm / RMSNorm: ones weight, zeros bias (if present).
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CausalMultimodalTransformer(nn.Module):
    """Causal dual-branch FHR-ST / UP-ST forecasting transformer (v2).

    v2 changes over v1:
        - Wider backbone (d_model=256, n_heads=8).
        - Asymmetric encoder depths (FHR: 6, UP: 3, Fused: 6).
        - RoPE positional encoding in all attention layers.
        - Multi-layer cross-attention fusion.
        - SwiGLU FFN and RMSNorm throughout.
        - Factorized latents: z_self (intrinsic FHR) + z_transfer (TE coupling).
        - Deeper residual MLP posterior/prior networks with free bits KL.
        - Shared-backbone forecast heads.
        - Enriched window representation export.

    Args:
        config: Transformer configuration (or keyword arguments to construct
            one).
    """

    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig(**kwargs)
        self.config = config
        d = config.d_model

        # --- Stems (spec §7) ---
        self.fhr_stem = CausalStem(
            in_dim=config.d_f,
            d_model=d,
            num_blocks=config.stem_num_blocks,
            kernels=config.stem_kernels,
            dilations=config.stem_dilations,
            expansion=config.stem_expansion,
            dropout=config.dropout,
            use_rmsnorm=config.use_rmsnorm,
            use_swiglu=config.use_swiglu,
        )
        self.up_stem = CausalStem(
            in_dim=config.d_u,
            d_model=d,
            num_blocks=config.stem_num_blocks,
            kernels=config.stem_kernels,
            dilations=config.stem_dilations,
            expansion=config.stem_expansion,
            dropout=config.dropout,
            use_rmsnorm=config.use_rmsnorm,
            use_swiglu=config.use_swiglu,
        )

        # --- Modality encoders (spec §8) ---
        self.fhr_encoder = CausalTransformerEncoder(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=config.fhr_encoder_layers,
            ff_expansion=config.ff_expansion,
            dropout=config.dropout,
            use_checkpoint=config.gradient_checkpointing,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
        )
        self.up_encoder = CausalTransformerEncoder(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=config.up_encoder_layers,
            ff_expansion=config.ff_expansion,
            dropout=config.dropout,
            use_checkpoint=config.gradient_checkpointing,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
        )

        # --- Cross-attention fusion (spec §9, v2: multi-layer) ---
        self.fusion = CausalCrossAttentionFusion(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=config.fusion_layers,
            ff_expansion=config.ff_expansion,
            dropout=config.dropout,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
        )

        # --- Fused encoder (spec §9.4) ---
        self.fused_encoder = CausalTransformerEncoder(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=config.fused_encoder_layers,
            ff_expansion=config.ff_expansion,
            dropout=config.dropout,
            use_checkpoint=config.gradient_checkpointing,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
        )

        # --- Anchor attention pools (spec §12) ---
        self.pool_f = AttentionPool(d)
        self.pool_u = AttentionPool(d)
        self.pool_fu = AttentionPool(d)

        # --- Forecast heads (spec §13, v2: shared backbone) ---
        self.self_head = ForecastHead(
            in_dim=d,
            d_out=config.d_f,
            horizons=config.horizons,
            dropout=config.dropout,
        )
        self.fused_head = ForecastHead(
            in_dim=d,
            d_out=config.d_f,
            horizons=config.horizons,
            dropout=config.dropout,
        )
        self.te_head = ForecastHead(
            in_dim=d + config.d_z_self + config.d_z_transfer,
            d_out=config.d_f,
            horizons=config.horizons,
            dropout=config.dropout,
        )

        # --- Self latent module (v2) ---
        self.self_latent = SelfLatentModule(
            d_model=d,
            d_z_self=config.d_z_self,
            dropout=config.dropout,
        )

        # --- TE latent module (spec §14, v2: deeper) ---
        self.te_module = TELatentModule(
            d_model=d,
            d_z=config.d_z_transfer,
            dropout=config.dropout,
        )

        # --- Window representation export (spec §20, v2: enriched) ---
        self.window_export = WindowRepresentationExport(
            d, config.d_z_self, config.d_z_transfer,
        )

        # Initialize weights
        self.apply(_init_weights)

    def _gather_anchor_contexts(
        self,
        h: Tensor,
        anchor_indices: Tensor,
    ) -> Tensor:
        """Gather local context windows at anchor positions.

        For each anchor a, extracts ``h[:, a-L_ctx+1 : a+1, :]``.

        Args:
            h: Encoder states of shape ``(B, T, d)``.
            anchor_indices: Anchor positions of shape ``(B, K)``.

        Returns:
            Context windows of shape ``(B*K, L_ctx, d)``.
        """
        B, T, d = h.shape
        K = anchor_indices.shape[1]
        L = self.config.ctx_len

        # Build index tensor: for each anchor a, indices [a-L+1, a-L+2, ..., a]
        offsets = torch.arange(-L + 1, 1, device=h.device)     # (L,)
        # anchor_indices: (B, K) -> (B, K, 1) + (1, 1, L) -> (B, K, L)
        ctx_indices = anchor_indices.unsqueeze(-1) + offsets.view(1, 1, -1)

        # Gather: expand h and index
        ctx_indices_exp = ctx_indices.unsqueeze(-1).expand(B, K, L, d)  # (B, K, L, d)
        h_exp = h.unsqueeze(1).expand(B, K, T, d)                       # (B, K, T, d)
        windows = torch.gather(h_exp, dim=2, index=ctx_indices_exp)     # (B, K, L, d)

        return windows.reshape(B * K, L, d)

    def forward(
        self,
        Y: Tensor,
        U: Tensor,
        anchor_indices: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, Dict[int, Tensor]]]:
        """Forward pass.

        Args:
            Y: FHR scattering features of shape ``(B, T, d_F)``.
            U: UP scattering features of shape ``(B, T, d_U)``.
            anchor_indices: Anchor positions of shape ``(B, K)``.  When
                ``None``, the model runs in inference mode and returns only the
                window-level embedding.

        Returns:
            Dictionary with either:
                - Inference mode: ``{"e_win": (B, embed_dim)}``
                - Training mode: all forecast predictions, latent parameters,
                  and intermediate states for loss computation.
        """
        # --- Step 1-2: Stems ---
        F_out = self.fhr_stem(Y)                         # (B, T, d)
        S_out = self.up_stem(U)                          # (B, T, d)

        # --- Step 3: Modality encoders ---
        H_F = self.fhr_encoder(F_out)                    # (B, T, d)
        H_U = self.up_encoder(S_out)                     # (B, T, d)

        # --- Step 4: Cross-attention fusion + fused encoder ---
        H_tilde = self.fusion(H_F, H_U)                 # (B, T, d)
        H_FU = self.fused_encoder(H_tilde)              # (B, T, d)

        # --- Inference mode: export window embedding ---
        if anchor_indices is None:
            return self._inference_forward(Y, H_F, H_U, H_FU)

        # --- Training mode: anchor-based prediction ---
        B = Y.shape[0]
        K = anchor_indices.shape[1]

        # Step 5-6: Gather context windows and pool
        ctx_f = self._gather_anchor_contexts(H_F, anchor_indices)    # (B*K, L, d)
        ctx_u = self._gather_anchor_contexts(H_U, anchor_indices)    # (B*K, L, d)
        ctx_fu = self._gather_anchor_contexts(H_FU, anchor_indices)  # (B*K, L, d)

        s_f = self.pool_f(ctx_f)                         # (B*K, d)
        s_u = self.pool_u(ctx_u)                         # (B*K, d)
        s_fu = self.pool_fu(ctx_fu)                      # (B*K, d)

        # Step 7: Self-only and fused forecasts
        Y_hat_self = self.self_head(s_f)                 # {h: (B*K, h, d_f)}
        Y_hat_fus = self.fused_head(s_fu)                # {h: (B*K, h, d_f)}

        # Step 8a: Self latent (v2)
        self_out = self.self_latent(s_f)
        z_self = self_out["z_self"]                      # (B*K, d_z_self)

        # Step 8b: TE posterior / prior / sample
        te_out = self.te_module(s_f, s_u)
        z_transfer = te_out["z_te"]                      # (B*K, d_z_transfer)

        # Step 9: TE residual forecast (input includes both latents)
        te_input = torch.cat([s_f, z_self, z_transfer], dim=-1)
        R_hat = self.te_head(te_input)                   # {h: (B*K, h, d_f)}

        # Y_hat_te = stop_grad(Y_hat_self) + R_hat  (per horizon)
        Y_hat_te = {}
        for h in self.config.horizons:
            Y_hat_te[h] = Y_hat_self[h].detach() + R_hat[h]

        return {
            "Y_hat_self": Y_hat_self,
            "Y_hat_fus": Y_hat_fus,
            "Y_hat_te": Y_hat_te,
            "R_hat": R_hat,
            # Self latent outputs
            "mu_self": self_out["mu_self"],
            "logvar_self": self_out["logvar_self"],
            # TE latent outputs
            "mu_post": te_out["mu_post"],
            "logvar_post": te_out["logvar_post"],
            "mu_prior": te_out["mu_prior"],
            "logvar_prior": te_out["logvar_prior"],
            "anchor_indices": anchor_indices,
            "H_F": H_F,
            "H_FU": H_FU,
        }

    def _inference_forward(
        self,
        Y: Tensor,
        H_F: Tensor,
        H_U: Tensor,
        H_FU: Tensor,
    ) -> Dict[str, Tensor]:
        """Inference mode: produce enriched window embedding.

        Args:
            Y: FHR input of shape ``(B, T, d_F)``.
            H_F: FHR encoder states ``(B, T, d)``.
            H_U: UP encoder states ``(B, T, d)``.
            H_FU: Fused encoder states ``(B, T, d)``.

        Returns:
            Dictionary with ``e_win`` of shape ``(B, output_dim)``.
        """
        cfg = self.config
        B = Y.shape[0]

        # Fixed anchor grid for inference
        grid = torch.arange(
            cfg.valid_anchor_start, cfg.valid_anchor_end + 1, 15,
            device=Y.device,
        ).unsqueeze(0).expand(B, -1)          # (B, K_grid)
        K_grid = grid.shape[1]

        # Gather context and pool
        ctx_f = self._gather_anchor_contexts(H_F, grid)
        ctx_u = self._gather_anchor_contexts(H_U, grid)
        s_f_grid = self.pool_f(ctx_f)                    # (B*K_grid, d)
        s_u_grid = self.pool_u(ctx_u)                    # (B*K_grid, d)

        # Self latent at grid
        self_out = self.self_latent(s_f_grid)
        self_mus = self_out["mu_self"].view(B, K_grid, -1)  # (B, K_grid, d_z_self)

        # TE latent at grid
        te_out = self.te_module(s_f_grid, s_u_grid)
        te_mus = te_out["mu_post"].view(B, K_grid, -1)          # (B, K_grid, d_z_transfer)
        te_mu_priors = te_out["mu_prior"].view(B, K_grid, -1)   # (B, K_grid, d_z_transfer)
        te_logvar_posts = te_out["logvar_post"].view(B, K_grid, -1)
        te_logvar_priors = te_out["logvar_prior"].view(B, K_grid, -1)

        e_win = self.window_export(
            H_F, H_FU, self_mus,
            te_mus, te_mu_priors, te_logvar_posts, te_logvar_priors,
        )
        return {"e_win": e_win}


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class CausalTransformerLoss(nn.Module):
    """Loss computation for the Causal Multimodal Forecasting Transformer (v2).

    Computes seven loss terms:
        - ``L_fus``: Huber loss on fused forecasts (main loss, §16.2).
        - ``L_delta``: Huber loss on first-order temporal differences (§16.3).
        - ``L_delta2``: Huber loss on second-order temporal differences (v2).
        - ``L_spectral``: Huber loss on spectral magnitude consistency (v2).
        - ``L_self``: Huber loss on self-only forecasts (baseline, §16.4).
        - ``L_te``: Huber loss on TE residual forecasts (§16.5).
        - ``L_kl_self``: KL divergence for z_self (v2).
        - ``L_kl_transfer``: Conditional KL divergence for z_transfer (§16.6).

    Args:
        config: Transformer configuration.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon_weight_map = dict(
            zip(config.horizons, config.horizon_weights)
        )

    def _extract_targets(
        self,
        Y: Tensor,
        anchor_indices: Tensor,
        horizon: int,
    ) -> Tensor:
        """Extract future FHR target blocks from Y for given anchors and horizon.

        For anchor a and guard gap g, extracts ``Y[:, a+g+1 : a+g+1+h, :]``.

        Args:
            Y: FHR input of shape ``(B, T, d_F)``.
            anchor_indices: Anchor positions of shape ``(B, K)``.
            horizon: Number of future time steps to extract.

        Returns:
            Target blocks of shape ``(B*K, h, d_F)``.
        """
        B, T, d_f = Y.shape
        K = anchor_indices.shape[1]
        g = self.config.guard_gap

        # Start indices: a + g + 1 for each anchor
        starts = anchor_indices + g + 1                    # (B, K)

        # Build time indices: (B, K, h)
        offsets = torch.arange(horizon, device=Y.device)   # (h,)
        time_idx = starts.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)  # (B, K, h)

        # Gather from Y: need (B, K, h, d_f) index into (B, T, d_f)
        time_idx_exp = time_idx.unsqueeze(-1).expand(B, K, horizon, d_f)  # (B, K, h, d_f)
        Y_exp = Y.unsqueeze(1).expand(B, K, T, d_f)                       # (B, K, T, d_f)
        targets = torch.gather(Y_exp, dim=2, index=time_idx_exp)          # (B, K, h, d_f)

        return targets.reshape(B * K, horizon, d_f)

    def _weighted_huber(
        self,
        pred: Dict[int, Tensor],
        target_fn,
        Y: Tensor,
        anchor_indices: Tensor,
    ) -> Tensor:
        """Compute horizon-weighted Huber loss across all horizons.

        Args:
            pred: Dict mapping horizon h to predictions ``(B*K, h, d_f)``.
            target_fn: Callable(targets) -> targets (identity or diff).
            Y: FHR input for target extraction.
            anchor_indices: Anchor positions.

        Returns:
            Scalar loss.
        """
        total = torch.tensor(0.0, device=Y.device)
        weight_sum = 0.0
        for h in self.config.horizons:
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            loss_h = F.huber_loss(
                pred[h], target_fn(targets),
                reduction="mean",
                delta=self.config.huber_delta,
            )
            total = total + w * loss_h
            weight_sum += w
        return total / weight_sum

    def forward(
        self,
        outputs: Dict[str, Union[Tensor, Dict[int, Tensor]]],
        Y: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute all loss terms.

        Args:
            outputs: Output dictionary from ``CausalMultimodalTransformer.forward()``.
            Y: FHR scattering features of shape ``(B, T, d_F)`` (used to
                extract future targets).

        Returns:
            Dictionary with keys ``total_loss``, ``L_fus``, ``L_delta``,
            ``L_delta2``, ``L_spectral``, ``L_self``, ``L_te``,
            ``L_kl_self``, ``L_kl_transfer``.
        """
        anchor_indices = outputs["anchor_indices"]
        identity = lambda x: x

        # L_fus: main fused forecasting loss (spec §16.2)
        L_fus = self._weighted_huber(
            outputs["Y_hat_fus"], identity, Y, anchor_indices
        )

        # L_delta: first-order dynamics loss (spec §16.3)
        L_delta = torch.tensor(0.0, device=Y.device)
        weight_sum = 0.0
        for h in self.config.horizons:
            if h < 2:
                continue
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            pred_diff = torch.diff(outputs["Y_hat_fus"][h], dim=1)
            target_diff = torch.diff(targets, dim=1)
            loss_h = F.huber_loss(
                pred_diff, target_diff,
                reduction="mean",
                delta=self.config.huber_delta,
            )
            L_delta = L_delta + w * loss_h
            weight_sum += w
        if weight_sum > 0:
            L_delta = L_delta / weight_sum

        # L_delta2: second-order dynamics loss (v2)
        L_delta2 = torch.tensor(0.0, device=Y.device)
        weight_sum_d2 = 0.0
        for h in self.config.horizons:
            if h < 3:
                continue
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            pred_diff2 = torch.diff(torch.diff(outputs["Y_hat_fus"][h], dim=1), dim=1)
            target_diff2 = torch.diff(torch.diff(targets, dim=1), dim=1)
            loss_h = F.huber_loss(
                pred_diff2, target_diff2,
                reduction="mean",
                delta=self.config.huber_delta,
            )
            L_delta2 = L_delta2 + w * loss_h
            weight_sum_d2 += w
        if weight_sum_d2 > 0:
            L_delta2 = L_delta2 / weight_sum_d2

        # L_spectral: spectral consistency loss (v2)
        L_spectral = torch.tensor(0.0, device=Y.device)
        weight_sum_sp = 0.0
        for h in self.config.horizons:
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            # Cast to float32 for FFT — cuFFT requires power-of-2 sizes in fp16
            pred_f32 = outputs["Y_hat_fus"][h].float()
            tgt_f32 = targets.float()
            pred_fft = torch.fft.rfft(pred_f32, dim=1)
            target_fft = torch.fft.rfft(tgt_f32, dim=1)
            loss_h = F.huber_loss(
                pred_fft.abs(), target_fft.abs(),
                reduction="mean",
                delta=0.5,
            )
            L_spectral = L_spectral + w * loss_h
            weight_sum_sp += w
        if weight_sum_sp > 0:
            L_spectral = L_spectral / weight_sum_sp

        # L_self: self-only baseline loss (spec §16.4)
        L_self = self._weighted_huber(
            outputs["Y_hat_self"], identity, Y, anchor_indices
        )

        # L_te: TE residual loss (spec §16.5)
        L_te = torch.tensor(0.0, device=Y.device)
        weight_sum_te = 0.0
        for h in self.config.horizons:
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            residual_target = targets - outputs["Y_hat_self"][h].detach()
            loss_h = F.huber_loss(
                outputs["R_hat"][h], residual_target,
                reduction="mean",
                delta=self.config.huber_delta,
            )
            L_te = L_te + w * loss_h
            weight_sum_te += w
        L_te = L_te / weight_sum_te

        # L_kl_self: KL divergence for z_self (v2)
        L_kl_self = SelfLatentModule.kl_divergence(
            outputs["mu_self"],
            outputs["logvar_self"],
            free_bits=self.config.free_bits,
        )

        # L_kl_transfer: conditional KL divergence for z_transfer (§16.6)
        L_kl_transfer = TELatentModule.kl_divergence(
            outputs["mu_post"],
            outputs["logvar_post"],
            outputs["mu_prior"],
            outputs["logvar_prior"],
            free_bits=self.config.free_bits,
        )

        # Total loss (spec §16.7, v2 extensions)
        # Note: beta_self and beta_transfer are NOT included here — they are
        # applied by the training loop for scheduling flexibility.
        cfg = self.config
        total_loss = (
            cfg.lambda_fus * L_fus
            + cfg.lambda_delta * L_delta
            + cfg.lambda_delta2 * L_delta2
            + cfg.lambda_spectral * L_spectral
            + cfg.lambda_self * L_self
            + cfg.lambda_te * L_te
        )

        return {
            "total_loss": total_loss,
            "L_fus": L_fus,
            "L_delta": L_delta,
            "L_delta2": L_delta2,
            "L_spectral": L_spectral,
            "L_self": L_self,
            "L_te": L_te,
            "L_kl_self": L_kl_self,
            "L_kl_transfer": L_kl_transfer,
        }
