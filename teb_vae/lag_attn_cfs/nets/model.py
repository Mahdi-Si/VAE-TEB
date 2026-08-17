r"""The causal-feature forecaster: the raw-signal architecture, told where its inputs begin.

:class:`SeqVaeLagAttnCfs` is :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws` composed with
the two halves of this target domain, and it holds **nothing but a constructor**:

* :class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs` -- the input warm-up mask, the
  lag validity floor, the tiled anchor set and the forward that decodes at it;
* :class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget` -- the
  one-sided channel layout, the budget-and-floor refusal and the eleven resolved forecast readouts.

Neither mixin names an encoder, which is what lets the conv-Transformer cell compose exactly the same
two objects over a different architecture. What cannot be shared is this file's one member: the
experiment driver builds a run's kwargs by sweeping ``inspect.signature(MODEL_CLS.__init__)``, so each
cell writes out its **own** architecture's keyword list in full. A ``**kwargs`` signature would
forward four keys and silently build an all-defaults model.

``target_delays`` and ``source_delays`` are the only names removed from the base's list, and removing
them is the point: a warm-up is a leading *mask*, ``ChannelDelay`` is a *shift*, and a warm-up routed
under a delay name would train a different model with every shape intact.

Everything else -- both encoders, the channel gates, the prior and posterior heads, the lag
cross-attention, the shared decoder invoked twice, the paired reparameterisation and the seven-term
objective -- is the base's, reached by inheritance.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import (
    FORWARDED_EXCLUSIONS,
    CausalWarmupInputs,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


class SeqVaeLagAttnCfs(CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnRws):
    r"""Single-latent lag-attentive VAE forecasting one-sided target coefficients.

    The two mixins come **first** in the bases, which is what makes the tiled forward, the warm-up
    adapter, the floored lag mask, the width hook and the resolved gaps win method resolution over
    the raw-target ones; reversing them would build a decoder emitting $R = 16$ raw samples and
    score a $C_{\mathrm{keep}}$-wide block against it.

    The constructor signature is the base's, written out in full rather than narrowed to the new
    keywords. That is not style: the experiment driver builds a run's kwargs by sweeping
    ``inspect.signature(MODEL_CLS.__init__)``, so a ``**kwargs`` signature would forward four keys
    and silently build an all-defaults model.
    """

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 48,
        horizon: int = 30,
        raw_per_step: int = 16,
        warmup_period: int = 133,
        c_y: int = 102,
        c_u: int = 51,
        use_up_st: bool = True,
        max_lag: int = 90,
        num_heads: int = 4,
        d_head: int = 32,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_out_channels: Optional[int] = None,
        horizon_depth: int = 2,
        horizon_kernel: int = 3,
        horizon_film: bool = False,
        horizon_attention_blocks: int = 0,
        horizon_embed_std: float = 0.02,
        head_init_calibration: bool = False,
        a_head_gain: float = 1.0,
        encoder_extra_dilations: Tuple[int, ...] = (),
        encoder_extra_kernel: int = 15,
        conv_norm_groups: Optional[int] = None,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        mu_scale: float = 5.0,
        delta_mu_scale: float = 3.0,
        delta_logvar_scale: float = 2.0,
        posterior_logvar_mode: str = "residual",
        source_dropout: Optional[float] = None,
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
        query_uses_logvar: bool = False,
        causal_norm: bool = False,
        coverage_floor: float = 0.9,
        base_decode: str = "sample",
        target_keep_index: Optional[Sequence[int]] = None,
        target_warmup_steps: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_warmup_steps: Optional[Sequence[int]] = None,
        anchor_stride: int = 1,
        lag_floor: int = 0,
        target_weight_st: float = 1.0,
        target_weight_ph: float = 1.0,
        init_weights: bool = True,
    ) -> None:
        r"""Initialize the model.

        Every keyword the base takes is forwarded unchanged; only the four below are this target
        domain's, and only ``target_delays`` / ``source_delays`` are gone. The defaults that differ
        from the base's -- ``warmup_period`` $133$, ``c_y`` $102$, ``c_u`` $51$ -- are this target
        domain's geometry rather than a preference, and a run that leaves them at the base's values
        would be describing a dataset that does not exist. ``horizon`` is **no longer** among them:
        at $30$ it agrees with the base and with every other cell of the grid.

        Args:
            target_warmup_steps: $W'_c$ per **surviving** target channel, positional against
                ``target_keep_index``. ``None`` builds no warm-up mask, which is the ungated model.
            source_warmup_steps: The same for the source stream.
            anchor_stride: $S$, the spacing between decoded anchors, in $[1, H]$. Defaults to $1$
                -- the dense range every sibling decodes, and the inert value -- so a model
                constructed without an opinion behaves like the rest of the family. The tiling is a
                configuration decision, and the shipped configuration states it.
            lag_floor: $F_u$, the earliest source step lag attention may read. Ships at $0$,
                where the lag mask is bitwise the sibling's.
            target_weight_st: Relative reconstruction weight of the first stored target block,
                the scattering coefficients. Both weights default to $1.0$, where the objective is
                bitwise the uniform one.
            target_weight_ph: The same for the second stored block, the phase-harmonic
                coefficients. The pair is renormalised to leave the block scale unchanged, so what
                the configuration states is a **ratio**: $(1.0, 0.1)$ and $(10.0, 1.0)$ describe
                the same objective, agreeing to float32 rounding rather than bitwise.

        Raises:
            ValueError: If ``anchor_stride`` is outside $[1, H]$ or leaves a phase with no anchor;
                if ``lag_floor`` is negative; if a warm-up vector arrives without its keep-index;
                or if ``warmup_period`` is below the floor the kept channels require. Everything
                else is the base's own validation.
        """
        # Captured before anything else runs, so the forwarded set is exactly this signature minus
        # the four keywords the mixin owns. Written out as forty explicit `name=name` pairs it would
        # be the same dict with one silent failure mode: a keyword added to the base and forgotten
        # here would be forwarded at its default with nothing raising.
        forwarded = {
            name: value
            for name, value in locals().items()
            if name not in FORWARDED_EXCLUSIONS
        }

        # Before the base constructor, which builds the input adapters that read the two vectors.
        self._set_causal_inputs(
            horizon=horizon,
            target_keep_index=target_keep_index,
            target_warmup_steps=target_warmup_steps,
            source_keep_index=source_keep_index,
            source_warmup_steps=source_warmup_steps,
            anchor_stride=anchor_stride,
            lag_floor=lag_floor,
            target_weight_st=target_weight_st,
            target_weight_ph=target_weight_ph,
        )

        super().__init__(**forwarded, target_delays=None, source_delays=None)

        # After the base, which is what validates the geometry the anchor checks read.
        self._validate_causal_geometry()
