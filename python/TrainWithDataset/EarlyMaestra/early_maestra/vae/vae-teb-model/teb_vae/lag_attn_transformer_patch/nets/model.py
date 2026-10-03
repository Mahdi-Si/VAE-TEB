"""The patch-token conv-Transformer lag-attention VAE: two mixins, one architecture, one constructor.

``SeqVaeLagAttnTrfRws`` supplies the network; :class:`PatchStreamInputs` the patch adapters and the
tiled-anchor forward; :class:`PatchSummaryTarget` the 2-channel summary target and the objective.
The mixins come first in the bases, and that order is load-bearing: reversed, the architecture's
dense forward, raw decoder width and raw ``compute_loss`` win (``lag_attn_transformer_cfs/DESIGN.md``
§6).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from teb_vae.lag_attn.nets.blocks import validate_choice
from teb_vae.lag_attn_cfs.nets.causal_inputs import FORWARDED_EXCLUSIONS
from teb_vae.lag_attn_transformer_patch.nets.patch_inputs import PatchStreamInputs
from teb_vae.lag_attn_transformer_patch.nets.patch_target import PatchSummaryTarget
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: This model's own keywords, kept out of the base's ``**forwarded`` (C4).
PATCH_ONLY: Tuple[str, ...] = (
    "target_summary_loc",
    "target_summary_scale",
    "variability_eps",
    "source_validity",
)

SOURCE_VALIDITY_CHOICES: Tuple[str, ...] = ("finite", "fhr_weight")


class SeqVaeLagAttnTrfPatch(PatchStreamInputs, PatchSummaryTarget, SeqVaeLagAttnTrfRws):
    """Lag-attentive conv-Transformer VAE over 16-sample raw patches, forecasting patch summaries."""

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 48,
        horizon: int = 30,
        raw_per_step: int = 16,
        warmup_period: int = 30,
        max_lag: int = 37,
        num_heads: int = 4,
        d_head: int = 32,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        horizon_depth: int = 2,
        horizon_kernel: int = 3,
        horizon_film: bool = True,  # False cannot construct: the core hardcodes per-block FiLM
        horizon_attention_blocks: int = 0,
        horizon_embed_std: float = 0.02,
        head_init_calibration: bool = False,
        a_head_gain: float = 1.0,
        encoder_conv_kernels: Sequence[int] = (5, 9),
        encoder_conv_dilations: Sequence[int] = (1, 2),
        encoder_num_heads: int = 4,
        encoder_d_ff: int = 256,
        target_attention_blocks: int = 4,
        source_attention_blocks: int = 3,
        source_attention_window: Optional[int] = 16,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        mu_scale: float = 5.0,
        delta_mu_scale: float = 3.0,
        delta_logvar_scale: float = 2.0,
        posterior_logvar_mode: str = "residual",
        source_dropout: Optional[float] = None,
        lag_kv_source: str = "encoder",
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
        query_uses_logvar: bool = False,
        prior_availability_input: bool = False,
        coverage_floor: float = 0.9,
        base_decode: str = "sample",
        persistence_residual: bool = False,
        horizon_weight_halflife_steps: Optional[float] = None,
        anchor_stride: int = 1,
        lag_floor: int = 0,
        forecast_ar_residual: bool = False,
        target_summary_loc: Sequence[float] = (0.0, 0.0),
        target_summary_scale: Sequence[float] = (1.0, 1.0),
        variability_eps: float = 0.01,
        source_validity: str = "finite",
        init_weights: bool = True,
    ) -> None:
        """Initialize the model.

        The ``SeqVaeLagAttnTrfCrws`` keyword list without the feature-width, gate, warm-up and
        alignment keys; both streams are ``2R + 1`` wide and ungated. New keywords:

        Args:
            forecast_ar_residual: AR(1) residual likelihood, ``phi_c = tanh(a_c)`` seeded at 0.
            target_summary_loc: Per-channel location subtracted from ``[level, variability]``.
            target_summary_scale: Per-channel scale the centred summaries are divided by.
            variability_eps: ``eps`` in ``log(rms(diff) + eps)``, in loader z-units.
            source_validity: How the task masks the UP patches: ``"finite"`` or ``"fhr_weight"``.
                Read by the task, not by the model.
        """
        forwarded = {
            name: value
            for name, value in locals().items()
            if name not in FORWARDED_EXCLUSIONS + PATCH_ONLY
        }

        # Plain Python values: legal before Module.__init__, and read only after it.
        self.target_summary_loc = tuple(float(v) for v in target_summary_loc)
        self.target_summary_scale = tuple(float(v) for v in target_summary_scale)
        self.variability_eps = float(variability_eps)
        if (
            len(self.target_summary_loc) != 2
            or len(self.target_summary_scale) != 2
            or min(self.target_summary_scale) <= 0.0
            or self.variability_eps <= 0.0
        ):
            raise ValueError(
                f"target_summary_loc/scale must have 2 entries with scale > 0, and "
                f"variability_eps must be > 0; got loc={self.target_summary_loc}, "
                f"scale={self.target_summary_scale}, eps={self.variability_eps}"
            )
        self.source_validity = validate_choice(
            source_validity, SOURCE_VALIDITY_CHOICES, "source_validity"
        )

        self._set_causal_inputs(
            horizon=horizon,
            target_keep_index=None,
            target_warmup_steps=None,
            source_keep_index=None,
            source_warmup_steps=None,
            anchor_stride=anchor_stride,
            lag_floor=lag_floor,
        )
        self._set_likelihood_structure(
            target_scored_horizon=None, forecast_ar_residual=forecast_ar_residual
        )
        width = 2 * int(raw_per_step) + 1
        super().__init__(
            **forwarded,
            c_y=width,
            c_u=width,
            use_up_st=False,
            target_delays=None,
            source_delays=None,
        )
        self._validate_causal_geometry()
        # After the base's generic initialization, so phi starts at exactly 0.
        self._register_likelihood_structure()


__all__ = ["PATCH_ONLY", "SOURCE_VALIDITY_CHOICES", "SeqVaeLagAttnTrfPatch"]
