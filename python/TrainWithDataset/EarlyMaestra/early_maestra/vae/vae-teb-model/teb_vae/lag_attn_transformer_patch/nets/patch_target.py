"""The target half of the patch model: two standardized per-patch summaries, ``[level, variability]``.

:class:`PatchSummaryTarget` sets the decoder width to 2, owns ``compute_loss`` (the shared raw
objective at ``block_width=2``) and admits the persistence residual. The loss target and the
persistence input both go through :meth:`PatchSummaryTarget._standardized_summaries`, so the model
has one definition of the target (B.10 item 7). The AR(1) parameter and three helpers are bound by
reference from the feature-target mixins; none of them reads a feature-specific attribute here.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_raw_objective
from teb_vae.lag_attn_transformer_patch.nets.patching import patch_summaries, patchify


class PatchSummaryTarget:
    """Per-patch summary forecast target. Place before the architecture in the model's bases."""

    # Bound, not copied. With ``target_scored_horizon=None`` the register step reads only
    # ``decoder_out_channels`` and ``forecast_ar_residual`` and builds ``target_ar_logit = 0``.
    _set_likelihood_structure = CausalFeatureForecastTarget._set_likelihood_structure
    _register_likelihood_structure = CausalFeatureForecastTarget._register_likelihood_structure
    _anchors_per_sample = CausalFeatureForecastTarget._anchors_per_sample
    forecast_likelihood_kwargs = FeatureForecastTarget.forecast_likelihood_kwargs

    def _default_decoder_out_channels(self) -> int:
        """``[level, variability]``. A constant: the base calls this mid-``__init__``."""
        return 2

    def _check_persistence_target(self) -> None:
        """Admit the persistence residual: each summary channel has a level to carry forward."""

    def scored_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """The scored clock is the stored clock, so the weight is unchanged."""
        return weight

    def _standardized_summaries(self, patches: torch.Tensor) -> torch.Tensor:
        """``(..., 2R + 1)`` patch tokens to standardized ``(..., 2)`` summaries."""
        r = self.raw_per_step
        s = patch_summaries(patches[..., :r], patches[..., -1] + 1.0, eps=self.variability_eps)
        loc, scale = self.target_summary_loc, self.target_summary_scale
        # Python-float affine: no host-to-device copy per step.
        return torch.stack([(s[..., c] - loc[c]) / scale[c] for c in range(2)], dim=-1)

    def summary_target(self, fhr_raw: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The forecast target on the token grid, ``(B, T, 2)``."""
        patches = patchify(fhr_raw, weight, raw_per_step=self.raw_per_step, validity="fhr_weight")
        return self._standardized_summaries(patches)

    def _build_forecast_target(self, summaries: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """The scored block: anchor ``a``'s target at step ``τ`` is token ``a + 1 + τ``.

        The one gather both :meth:`compute_loss` and the shared evaluation (``metrics``, ``oracle``,
        ``attributions``) score against, so the eval's ``nll_*`` is the training loss's.

        Args:
            summaries: Standardized summaries on the token grid ``(B, T, 2)``.
            anchors: Decoded anchors ``(B, A)``.

        Returns:
            ``(B, A, H, 2)``.
        """
        taus = torch.arange(1, self.horizon + 1, device=summaries.device)
        steps = anchors.to(summaries.device).long()[:, :, None] + taus  # (B, A, H)
        rows = torch.arange(summaries.shape[0], device=summaries.device)[:, None, None]
        return summaries[rows, steps]

    def _anchor_target_values(self, target: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Persistence input: the anchor's own patch summaries, ``(B, A, 2)``.

        ``target`` is the pre-gate ``(B, T, 2R + 1)`` target patch stream.
        """
        index = anchors.long()[:, :, None].expand(-1, -1, target.shape[-1])
        return self._standardized_summaries(target.gather(1, index))

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        fhr_raw: torch.Tensor,
        *,
        weight: torch.Tensor,
        beta: float = 1.0,
        beta_prior: float = 0.0,
        lambda_full: float = 1.0,
        lambda_base: float = 1.0,
        likelihood: str = "gaussian_nll",
        free_bits: float = 0.0,
        lambda_ms: float = 0.0,
        lambda_deriv: float = 0.0,
        lambda_boundary: float = 0.0,
    ) -> Dict[str, Any]:
        """The shared objective on the summaries of tokens ``a + 1 .. a + H`` per decoded anchor.

        Same signature as ``CausalRawInputs.compute_loss``. The forecast mask is built inside the
        shared objective from ``weight``, so an invalid target patch is not scored.

        Returns:
            ``{'metrics': ..., 'likelihood': ...}``: the shared objective's metrics plus
            ``anchors_per_sample``.
        """
        summaries = self.summary_target(fhr_raw, weight)
        batch = summaries.shape[0]
        anchors = forward_outputs.get("anchor_index")
        if anchors is None:  # a dense forward: every anchor below the ceiling
            anchors = torch.arange(self.anchor_ceiling, device=summaries.device).expand(batch, -1)
        target = self._build_forecast_target(summaries, anchors)

        result = compute_raw_objective(
            forward_outputs,
            target,
            weight=weight,
            geometry=self.geometry,
            block_width=self.decoder_out_channels,
            coverage_floor=self.coverage_floor,
            logvar_clamp=self.logvar_clamp,
            beta=beta,
            beta_prior=beta_prior,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            free_bits=free_bits,
            lambda_ms=lambda_ms,
            lambda_deriv=lambda_deriv,
            lambda_boundary=lambda_boundary,
            horizon_weight=getattr(self, "horizon_weight", None),
            **self.forecast_likelihood_kwargs(),
        )
        result["metrics"]["anchors_per_sample"] = self._anchors_per_sample(forward_outputs, target)
        return result


__all__ = ["PatchSummaryTarget"]
