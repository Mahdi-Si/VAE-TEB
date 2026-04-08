"""Top-level Causal Multimodal Forecasting Transformer.

Contains:
    - sample_anchors: Standalone utility for anchor index sampling.
    - CausalMultimodalTransformer: Main nn.Module wiring all components.
    - CausalTransformerLoss: Loss computation module matching model.md §16.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TransformerConfig
from .encoder import CausalCrossAttentionFusion, CausalTransformerEncoder
from .heads import ForecastHead, TELatentModule, WindowRepresentationExport
from .layers import AttentionPool
from .stems import CausalStem


def sample_anchors(
    Y: Tensor,
    U: Tensor,
    config: TransformerConfig,
    training: bool = True,
) -> Tensor:
    """Sample 0-based anchor indices for prediction (model.md §10)."""
    b = Y.shape[0]
    device = Y.device
    a_start = config.valid_anchor_start
    a_end = config.valid_anchor_end
    valid_range = a_end - a_start + 1

    if not training:
        grid = torch.arange(a_start, a_end + 1, 15, device=device)
        return grid.unsqueeze(0).expand(b, -1)

    k = config.num_anchors
    eta = config.anchor_uniform_ratio

    u_valid = U[:, a_start:a_end + 1, :]
    y_valid = Y[:, a_start:a_end + 1, :]

    score = u_valid.abs().sum(dim=-1)
    if valid_range > 1:
        delta_u = F.pad(torch.diff(u_valid, dim=1).abs().sum(dim=-1), (1, 0))
        delta_y = F.pad(torch.diff(y_valid, dim=1).abs().sum(dim=-1), (1, 0))
        score = score + delta_u + delta_y

    score_prob = score / (score.sum(dim=-1, keepdim=True) + 1e-8)
    uniform_prob = torch.ones(b, valid_range, device=device) / valid_range
    mixed_prob = eta * uniform_prob + (1.0 - eta) * score_prob

    row_sums = mixed_prob.sum(dim=-1, keepdim=True)
    mixed_prob = torch.where(row_sums > 1e-8, mixed_prob, uniform_prob)

    indices_in_range = torch.multinomial(mixed_prob, k, replacement=False)
    return indices_in_range + a_start


def validate_anchor_indices(anchor_indices: Tensor, config: TransformerConfig) -> None:
    """Validate that anchor indices are within the legal 0-based range."""
    a_min = anchor_indices.min().item()
    a_max = anchor_indices.max().item()
    if a_min < config.valid_anchor_start or a_max > config.valid_anchor_end:
        raise ValueError(
            f"anchor_indices out of valid range "
            f"[{config.valid_anchor_start}, {config.valid_anchor_end}]: "
            f"got min={a_min}, max={a_max}"
        )


def _init_weights(module: nn.Module) -> None:
    """Initialize weights following existing project conventions."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class CausalMultimodalTransformer(nn.Module):
    """Document-aligned causal multimodal forecasting transformer."""

    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig(**kwargs)
        self.config = config
        d = config.d_model

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

        self.fusion = CausalCrossAttentionFusion(
            d_model=d,
            n_heads=config.n_heads,
            n_layers=config.fusion_layers,
            ff_expansion=config.ff_expansion,
            dropout=config.dropout,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
        )
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

        self.pool_f = AttentionPool(d)
        self.pool_u = AttentionPool(d)
        self.pool_fu = AttentionPool(d)

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
            in_dim=d + config.d_z_transfer,
            d_out=config.d_f,
            horizons=config.horizons,
            dropout=config.dropout,
        )

        self.te_module = TELatentModule(
            d_model=d,
            d_z=config.d_z_transfer,
            dropout=config.dropout,
        )
        self.window_export = WindowRepresentationExport(d, config.d_z_transfer)

        self.apply(_init_weights)

    def _gather_anchor_contexts(self, h: Tensor, anchor_indices: Tensor) -> Tensor:
        """Gather local context windows h[a-L+1:a+1] for each anchor."""
        b, t, d = h.shape
        k = anchor_indices.shape[1]
        l = self.config.ctx_len

        offsets = torch.arange(-l + 1, 1, device=h.device)
        ctx_indices = anchor_indices.unsqueeze(-1) + offsets.view(1, 1, -1)

        ctx_indices_exp = ctx_indices.unsqueeze(-1).expand(b, k, l, d)
        h_exp = h.unsqueeze(1).expand(b, k, t, d)
        windows = torch.gather(h_exp, dim=2, index=ctx_indices_exp)
        return windows.reshape(b * k, l, d)

    def forward(
        self,
        Y: Tensor,
        U: Tensor,
        anchor_indices: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, Dict[int, Tensor]]]:
        """Forward pass.

        When ``anchor_indices`` is ``None``, returns only ``{"e_win": ...}``.
        Otherwise returns forecast outputs and TE latent parameters.
        """
        f_out = self.fhr_stem(Y)
        s_out = self.up_stem(U)

        h_f = self.fhr_encoder(f_out)
        h_u = self.up_encoder(s_out)

        h_tilde = self.fusion(h_f, h_u)
        h_fu = self.fused_encoder(h_tilde)

        if anchor_indices is None:
            return self._inference_forward(Y, h_f, h_u, h_fu)

        ctx_f = self._gather_anchor_contexts(h_f, anchor_indices)
        ctx_u = self._gather_anchor_contexts(h_u, anchor_indices)
        ctx_fu = self._gather_anchor_contexts(h_fu, anchor_indices)

        s_f = self.pool_f(ctx_f)
        s_u = self.pool_u(ctx_u)
        s_fu = self.pool_fu(ctx_fu)

        y_hat_self = self.self_head(s_f)
        y_hat_fus = self.fused_head(s_fu)

        te_out = self.te_module(s_f, s_u)
        z_te = te_out["z_te"]

        r_hat = self.te_head(torch.cat([s_f, z_te], dim=-1))
        y_hat_te = {
            h: y_hat_self[h].detach() + r_hat[h]
            for h in self.config.horizons
        }

        return {
            "Y_hat_self": y_hat_self,
            "Y_hat_fus": y_hat_fus,
            "Y_hat_te": y_hat_te,
            "R_hat": r_hat,
            "mu_post": te_out["mu_post"],
            "logvar_post": te_out["logvar_post"],
            "mu_prior": te_out["mu_prior"],
            "logvar_prior": te_out["logvar_prior"],
            "anchor_indices": anchor_indices,
            "H_F": h_f,
            "H_FU": h_fu,
        }

    def _inference_forward(
        self,
        Y: Tensor,
        H_F: Tensor,
        H_U: Tensor,
        H_FU: Tensor,
    ) -> Dict[str, Tensor]:
        """Inference mode: produce the document-defined window embedding."""
        cfg = self.config
        b = Y.shape[0]

        grid = torch.arange(
            cfg.valid_anchor_start,
            cfg.valid_anchor_end + 1,
            15,
            device=Y.device,
        ).unsqueeze(0).expand(b, -1)
        k_grid = grid.shape[1]

        ctx_f = self._gather_anchor_contexts(H_F, grid)
        ctx_u = self._gather_anchor_contexts(H_U, grid)
        s_f_grid = self.pool_f(ctx_f)
        s_u_grid = self.pool_u(ctx_u)

        te_out = self.te_module(s_f_grid, s_u_grid)
        te_mus = te_out["mu_post"].view(b, k_grid, -1)

        return {"e_win": self.window_export(H_F, H_FU, te_mus)}


class CausalTransformerLoss(nn.Module):
    """Loss computation matching model.md §16.

    Returns ``L_kl`` as the TE conditional KL. Legacy alias keys
    ``L_delta2``, ``L_spectral``, ``L_kl_self``, and ``L_kl_transfer`` are kept
    to reduce breakage in surrounding utilities.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon_weight_map = dict(zip(config.horizons, config.horizon_weights))

    def _extract_targets(
        self,
        Y: Tensor,
        anchor_indices: Tensor,
        horizon: int,
    ) -> Tensor:
        b, t, d_f = Y.shape
        k = anchor_indices.shape[1]
        g = self.config.guard_gap

        starts = anchor_indices + g + 1
        offsets = torch.arange(horizon, device=Y.device)
        time_idx = starts.unsqueeze(-1) + offsets.view(1, 1, -1)

        time_idx_exp = time_idx.unsqueeze(-1).expand(b, k, horizon, d_f)
        y_exp = Y.unsqueeze(1).expand(b, k, t, d_f)
        targets = torch.gather(y_exp, dim=2, index=time_idx_exp)
        return targets.reshape(b * k, horizon, d_f)

    def _weighted_huber(
        self,
        pred: Dict[int, Tensor],
        target_fn,
        Y: Tensor,
        anchor_indices: Tensor,
    ) -> Tensor:
        total = torch.tensor(0.0, device=Y.device)
        weight_sum = 0.0
        for h in self.config.horizons:
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            loss_h = F.huber_loss(
                pred[h],
                target_fn(targets),
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
        anchor_indices = outputs["anchor_indices"]
        identity = lambda x: x

        l_fus = self._weighted_huber(outputs["Y_hat_fus"], identity, Y, anchor_indices)

        l_delta = torch.tensor(0.0, device=Y.device)
        weight_sum = 0.0
        for h in self.config.horizons:
            if h < 2:
                continue
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            pred_diff = torch.diff(outputs["Y_hat_fus"][h], dim=1)
            target_diff = torch.diff(targets, dim=1)
            loss_h = F.huber_loss(
                pred_diff,
                target_diff,
                reduction="mean",
                delta=self.config.huber_delta,
            )
            l_delta = l_delta + w * loss_h
            weight_sum += w
        if weight_sum > 0:
            l_delta = l_delta / weight_sum

        l_self = self._weighted_huber(outputs["Y_hat_self"], identity, Y, anchor_indices)

        l_te = torch.tensor(0.0, device=Y.device)
        weight_sum_te = 0.0
        for h in self.config.horizons:
            w = self.horizon_weight_map[h]
            targets = self._extract_targets(Y, anchor_indices, h)
            residual_target = targets - outputs["Y_hat_self"][h].detach()
            loss_h = F.huber_loss(
                outputs["R_hat"][h],
                residual_target,
                reduction="mean",
                delta=self.config.huber_delta,
            )
            l_te = l_te + w * loss_h
            weight_sum_te += w
        l_te = l_te / weight_sum_te

        l_kl = TELatentModule.kl_divergence(
            outputs["mu_post"],
            outputs["logvar_post"],
            outputs["mu_prior"],
            outputs["logvar_prior"],
            free_bits=self.config.free_bits,
        )

        cfg = self.config
        total_loss = (
            cfg.lambda_fus * l_fus
            + cfg.lambda_delta * l_delta
            + cfg.lambda_self * l_self
            + cfg.lambda_te * l_te
        )

        zero = torch.tensor(0.0, device=Y.device)
        return {
            "total_loss": total_loss,
            "L_fus": l_fus,
            "L_delta": l_delta,
            "L_delta2": zero,
            "L_spectral": zero,
            "L_self": l_self,
            "L_te": l_te,
            "L_kl": l_kl,
            "L_kl_self": zero,
            "L_kl_transfer": l_kl,
        }
