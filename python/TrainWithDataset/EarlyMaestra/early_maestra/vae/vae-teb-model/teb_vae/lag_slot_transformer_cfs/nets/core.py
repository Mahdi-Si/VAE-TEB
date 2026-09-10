r"""The architecture base: everything this design builds, and nothing it does not.

A plain :class:`~torch.nn.Module` that constructs the target encoder, the prior, the pointwise
source pathway, the proposal head and the shared decoder, and supplies the construction hooks the
two target-domain mixins call into. It is never used alone: it is the last base of
:class:`~teb_vae.lag_slot_transformer_cfs.nets.model.SeqVaeLagResidualTrfCfs`, and the mixins ahead
of it own the warm-up mask, the anchor tiling, the target gather and the scored clock.

**Why a base of its own rather than a subclass of the lag-attentive architecture.** That
constructor builds six modules this design has no use for -- the lag cross-attention, its query
projection, the head-structured posterior, the attention-attribution head and one of the two source
memories -- and the design's requirement is not that they go unused but that they are never
constructed. An unreachable parameter block is a starved entry in a distributed run's expectation
set and a claim in the checkpoint that the model attends over a source state it does not have. So
the two architectures share their *objective* and their components, imported from the modules that
own them, rather than their construction.

**What is deliberately absent, and how the absence is enforced.** There is no attention over the
source, no head-structured posterior, no independently bounded posterior log-variance and no source
dropout. Each of those is a keyword some sibling configuration sets, and the experiment driver
builds a run's kwargs by sweeping the constructor signature -- so a key this signature does not
mention is **silently dropped**, and a configuration asking for entmax attention over a model with
no attention would train quietly to completion. :data:`REFUSED_KEYWORDS` is why the composing model
lists them anyway: they reach :func:`refuse_incompatible_keywords`, which refuses each by name.

**The metadata clock is this module's, not the prior head's.** The head can carry a clock path of
its own, and this design does not use it, because the conditioning state

$$h_t = h^Y_t + W_A \operatorname{LayerNorm}(\chi_t)$$

is read by the **proposal head** as well as by the prior, and a projection living inside the prior
head would have to be reached across a module boundary to serve it -- or computed twice, which is
two clocks. It is built here, applied once in the forward, and the result is handed to both.

$\chi_t$ reads stored **position** and nothing else: no source value, no source-path parameter, no
recording-dependent quantity. That is the whole of its contract, and it is what lets the prior and
the full distribution both be told which channels have arrived without the prior being told
anything about what they said.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import initialization
from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
from teb_vae.lag_attn.nets.delays import ChannelGate
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
from teb_vae.lag_attn_rws.nets.losses import horizon_decay_weight
from teb_vae.lag_attn_rws.nets.losses import kld_tensor as closed_form_kld
from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
from teb_vae.lag_attn_transformer_rws.nets.blocks import init_depthwise_
from teb_vae.lag_attn_transformer_rws.nets.encoders import (
    AvailabilityInputAdapter,
    CausalConvTransformerEncoder,
)
from teb_vae.lag_slot_transformer_cfs.nets.conv_source import ConvSourceStem
from teb_vae.lag_slot_transformer_cfs.nets.lag_attention import LagAttentionFusion
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import (
    LagProposalHead,
    default_lag_scale,
)
from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import PointwiseSourceEncoder

#: How each lag's source vector may be formed, and what each choice is for.
#:
#: ``pointwise`` is the recommended representation and the one every claim in this design rests on:
#: each coefficient carries its own value and its own availability, nothing mixes two stored times,
#: and the additional neural source receptive field is one stored sample. ``conv`` is the
#: comparator -- the bounded causal convolution stack this design replaced -- and it exists so that
#: removing the stem can be measured on its own rather than inside a package of changes.
SOURCE_STEMS: Tuple[str, ...] = ("pointwise", "conv")

#: How the per-lag source vectors may become the latent update.
#:
#: ``local`` is the recommended explicit sum of one proposal per lag, bounded after summation.
#: ``attention`` is the comparator: a learned distribution over lags. Everything downstream of the
#: choice is identical, which is the condition that makes a difference between the two attributable
#: to the fusion rather than to five changes at once.
LAG_FUSIONS: Tuple[str, ...] = ("local", "attention")

#: Constructor keywords a sibling configuration sets that describe machinery this architecture does
#: not have, mapped to why each one cannot be honoured.
#:
#: **These are refused rather than omitted, and the difference is the whole point.** The experiment
#: driver forwards a ``model_config.VAE_model`` key only when it names a real constructor parameter,
#: so a key absent from the signature is dropped without a word: a run configured for windowed
#: source attention over a model that has none would train to completion and report its divergence
#: as a coupling measurement of a pathway it never built. Listing them and raising is what turns
#: that into a startup failure.
REFUSED_KEYWORDS: Dict[str, str] = {
    "num_heads": (
        "no latent here is head-structured, on any arm: the update is read out through one "
        "projection into the full latent width, so a head count would partition nothing. The "
        "attention comparator's own head count is lag_attention_heads, which splits the "
        "attention and nothing else"
    ),
    "d_head": (
        "the same: the comparator's per-head width follows from d_model and lag_attention_heads, "
        "and no other arm has an attention head at all"
    ),
    "source_attention_blocks": (
        "neither source representation stacks attention over the stream. The pointwise one holds "
        "no temporal operator at all and the comparator's stem is a fixed convolution schedule; "
        "the only attention on any arm is the lag fusion, which is one layer chosen by lag_fusion"
    ),
    "source_attention_window": "the same: there is no source self-attention to window",
    "source_dropout": (
        "the source pathway holds no parameters on the recommended arm and no dropout on any arm; "
        "a rate here would describe a stochastic source encoder that does not exist, and it would "
        "break the exact pairing of the two branches in training mode"
    ),
    "lag_kv_source": (
        "the source representation is chosen by source_stem, which names what forms each lag's "
        "vector rather than which of two memories a key-value projection reads"
    ),
    "use_entmax": (
        "the attention comparator uses plain softmax deliberately. A sparsifying normaliser "
        "changes what the distribution over lags looks like, so switching it on alongside the "
        "fusion would be a second declared change inside a comparison built to isolate one"
    ),
    "lag_bias_init": (
        "the comparator's per-lag key bias is learned from a small symmetric draw on every arm, "
        "for the same reason: a seeded long-lag penalty is a prior on the readout and would be a "
        "second change"
    ),
    "alibi_slope_scale": "the same, and it has no effect without that seeded penalty",
    "attention_grad_checkpoint": (
        "the comparator's fusion is one layer over an anchor-indexed window and is bounded by "
        "anchor chunking, which is exact; recomputation would trade the same memory for compute "
        "with no setting recording that it happened"
    ),
    "query_uses_logvar": (
        "no arm poses a query from the prior's parameters: both fusions read the conditioning "
        "state whole, which is the same tensor the prior heads read"
    ),
    "posterior_logvar_mode": (
        "the full log-variance is a residual on the prior's by construction, not a configurable "
        "mode, and an independent head would break the exact equality at a zero update"
    ),
    "delta_mu_scale": (
        "the mean bound is in PRIOR-STANDARD-DEVIATION units here, not in raw latent units; "
        "reusing the name would silently reinterpret its value. Set residual_mu_scale instead"
    ),
    "delta_logvar_scale": (
        "the scale bound is on the log-standard-deviation residual, not on a raw log-variance "
        "delta. Set residual_logsigma_scale instead"
    ),
    "a_head_gain": (
        "no arm fuses an attended source summary into a head-structured posterior, so there is "
        "no per-head gain to set"
    ),
    "base_decode": (
        "both branches are always sampled under one shared noise draw; decoding the base at its "
        "mean would make the base and full scores two different estimators"
    ),
    "prior_availability_input": (
        "the metadata clock is unconditional and explicit here, so there is no arm without it"
    ),
}


def refuse_incompatible_keywords(values: Mapping[str, Any]) -> None:
    """Refuse any keyword naming machinery this architecture does not have.

    Args:
        values: The composing constructor's own locals, or any mapping keyed by keyword name. A
            key present with value ``None`` is treated as unset, because that is what the
            signature default is and what an absent configuration key resolves to.

    Raises:
        ValueError: On the first refused keyword that carries a value, naming it, the reason, and
            what to set instead where there is a replacement.
    """
    for name, reason in REFUSED_KEYWORDS.items():
        if values.get(name) is None:
            continue
        raise ValueError(
            f"{name}={values[name]!r} was given, and this architecture cannot honour it: {reason}. "
            f"Remove the key from the configuration rather than setting it to a value that looks "
            f"inert -- it is refused here because the experiment driver forwards a key only when "
            f"the constructor names it, so a key this model quietly ignored would leave a run "
            f"describing a mechanism it never built."
        )


class LagResidualCore(nn.Module):
    r"""Build the modules this architecture has, and supply the hooks the mixins call.

    Ordered exactly as the family's constructors are, because the order is load-bearing: the gates
    decide the adapters' widths, the adapters and the encoder decide the conditioning state's width,
    the target gate decides the decoder's output width, and the whole initialisation sequence runs
    last so that the generic pass cannot undo the zeroings that make the model start at the prior.

    **Five arms, and every one of them is a construction rather than a branch in the forward.** The
    recommended arm builds a pointwise source representation and a local proposal head. The
    mean-only arm builds no scale half of the update. The target-only arm builds no source pathway
    at all. The capacity-control arm builds the whole pathway and withholds the source *values* from
    it, so a head of identical trainable capacity sees lag identity and availability and nothing
    else. And the two attention comparators replace the local sum with a distribution over lags,
    over either source representation. Each changes the module tree, so each is a different
    checkpoint and a run cannot silently be one arm while its configuration describes another.

    Attributes:
        geometry: The trimmed-grid geometry every mask and target gather is built against.
        source_encoder: The source representation, or ``None`` on the target-only arm. Pointwise on
            every recommended arm, where it holds no parameters unless the scalar lift is
            configured, and a bounded causal convolution stem on the comparator arm that isolates
            the stem's removal.
        proposal_head: The only learned fusion on the source pathway, or ``None`` on the
            target-only arm. One name for both fusions, because it is one role -- the map from the
            per-lag source vectors to the latent update -- and because a run's transfer, its
            re-zeroing hook and its checkpoint prefixes all address it by that role.
        clock_proj: $W_A$, the bias-free projection of the normalised metadata clock. Zeroed after
            the generic initialisation, so the conditioning state starts as the target state alone.
    """

    #: The causal input guards, or ``None`` when no budget is configured. Declared so they type as
    #: gates rather than as the ``Tensor | Module`` a bare submodule attribute would.
    target_gate: Optional[ChannelGate]
    source_gate: Optional[ChannelGate]

    #: The metadata clock, $(T, d_{\mathrm{model}})$. A deterministic function of stored position.
    metadata_clock: torch.Tensor

    #: The resolved horizon weighting $(H,)$, absent on a model built without a halflife.
    #:
    #: Declared without a value on purpose, exactly as the family's other optional weight buffers
    #: are: a model that configures one registers this name as a *buffer*, and ``register_buffer``
    #: refuses a name the class already carries, which a ``= None`` default here would create.
    horizon_weight: Optional[torch.Tensor]

    def __init__(
        self,
        *,
        sequence_length: int,
        d_model: int,
        d_z: int,
        horizon: int,
        raw_per_step: int,
        warmup_period: int,
        c_y: int,
        c_u: int,
        use_up_st: bool,
        max_lag: int,
        dropout: float,
        decoder_hidden: int,
        horizon_depth: int,
        horizon_kernel: int,
        horizon_film: bool,
        horizon_attention_blocks: int,
        horizon_embed_std: float,
        head_init_calibration: bool,
        encoder_conv_kernels: Sequence[int],
        encoder_conv_dilations: Sequence[int],
        encoder_num_heads: int,
        encoder_d_ff: int,
        target_attention_blocks: int,
        logvar_clamp: Tuple[float, float],
        mu_scale: float,
        coverage_floor: float,
        persistence_residual: bool,
        horizon_weight_halflife_steps: Optional[float],
        residual_mu_scale: float,
        residual_logsigma_scale: float,
        lag_embed_dim: int,
        proposal_hidden: Optional[int],
        mean_only_residual: bool,
        source_scalar_lift: bool,
        source_disabled: bool,
        source_values_withheld: bool,
        source_stem: str,
        lag_fusion: str,
        lag_attention_heads: int,
        lag_scale: Optional[float],
        anchor_chunk: Optional[int],
        lag_chunk: Optional[int],
        target_keep_index: Optional[Sequence[int]],
        source_keep_index: Optional[Sequence[int]],
        target_delays: Optional[Sequence[int]],
        source_delays: Optional[Sequence[int]],
        init_weights: bool,
    ) -> None:
        r"""Build every module and run the initialisation sequence.

        Args:
            sequence_length: Stored steps $T$ after trimming.
            d_model: Encoder and conditioning-state width $d_h$.
            d_z: Latent width $d_z$.
            horizon: Future steps per forecast $H$.
            raw_per_step: Raw samples per stored step. Carried for the geometry, which every mask
                is built against; this target domain scores coefficients rather than raw samples.
            warmup_period: The anchor floor $F$.
            c_y: Declared target channels before the gate.
            c_u: Declared source channels before the gate.
            use_up_st: Whether the source stream's first stored block is present, which the source
                warmth readout partitions on.
            max_lag: Furthest candidate lag, so $L = \texttt{max\_lag} + 1$.
            dropout: Dropout inside the target adapter, the target encoder and the prior heads.
                The source pathway and the decoder take none.
            decoder_hidden: Shared decoder hidden width.
            horizon_depth: Dilated blocks in the horizon core.
            horizon_kernel: Horizon convolution kernel width.
            horizon_film: Whether the latent modulates each horizon block.
            horizon_attention_blocks: Bidirectional attention blocks over the generated forecast
                tokens. Permitted because they operate on tokens generated from the latent, never
                on source history.
            horizon_embed_std: Standard deviation the horizon-step embedding is re-seeded at.
            head_init_calibration: Whether to place the decoder and the prior scale on the trivial
                predictor at initialisation. For a freshly trained model only; a transferred
                target encoder, prior and decoder must not be recalibrated.
            encoder_conv_kernels: Kernel width per target stem block.
            encoder_conv_dilations: Dilation per target stem block.
            encoder_num_heads: Attention heads inside the target encoder. Unrelated to anything in
                the latent, which this architecture does not partition.
            encoder_d_ff: Feed-forward width inside the target encoder.
            target_attention_blocks: Causal Transformer blocks in the target encoder.
            logvar_clamp: The prior's log-variance bound, and the decoder's observation
                log-variance bound. The **full** log-variance is not bounded by it: it is the
                prior's plus a bounded residual, so its range is wider by twice the scale bound.
            mu_scale: Bound on the prior mean.
            coverage_floor: Minimum valid fraction of an anchor's forecast window.
            persistence_residual: Whether the decoder's mean carries a weighted copy of the
                anchor's own target vector.
            horizon_weight_halflife_steps: Half-life of the horizon weighting, or ``None`` for the
                horizon-uniform objective.
            residual_mu_scale: $a_{\max}$, the mean bound **in prior standard deviations**.
            residual_logsigma_scale: $b_{\max}$, the bound on the log-standard-deviation residual.
            lag_embed_dim: Width of the lag embedding.
            proposal_hidden: Body width of the proposal head, or ``None`` for ``d_model``.
            mean_only_residual: Build no scale-proposal parameters, so the full distribution
                differs from the prior in its mean alone.
            source_disabled: Build **no** source pathway at all -- no pointwise encoder, no lag
                embeddings, no proposal head. The full distribution is then the prior, the
                divergence is exactly zero, and the model is a target-only forecaster on the same
                task, the same decoder and the same objective as the joint candidate.

                A constructor decision rather than a flag the forward consults, for the reason the
                mean-only arm is one: an arm that built the modules and never used them would be a
                starved parameter block under a distributed run and a claim in the manifest that
                the model reads a source it does not. It also makes the checkpoint's key set
                exactly the transferable half, which is what a warm start needs.
            source_scalar_lift: Widen each source coefficient's representation with a per-channel
                lift, retaining the identity and mask coordinates.
            source_values_withheld: The capacity control. The source pathway is built whole and the
                encoder emits an exact zero in place of every coefficient, so the fusion receives
                lag identity and the availability announcement and no source value at all. Widths,
                parameter count, optimizer state and checkpoint key set are the candidate's exactly,
                which is what makes a predictive gain on this arm evidence about extra nonlinear
                target capacity rather than about the source.
            source_stem: How each lag's source vector is formed. ``'pointwise'`` is the recommended
                representation: value and availability per coefficient, nothing mixing two stored
                times. ``'conv'`` is the comparator, a bounded causal convolution stack whose output
                at one step summarises a window ending there -- the representation this design
                replaced, built here so that its removal can be isolated from everything else that
                changed with it.
            lag_fusion: How the per-lag vectors become the update. ``'local'`` is the recommended
                explicit sum of one proposal per lag. ``'attention'`` is the comparator, a learned
                distribution over lags read out through one projection; everything downstream of it
                -- the bound, the residual parameters, the divergence, the sampling, the decoder --
                is identical, which is what makes a difference between the two attributable to the
                fusion.
            lag_attention_heads: Attention heads on the attention fusion, which must divide
                ``d_model``. It partitions the attention and nothing in the latent: no arm here has
                a head-structured posterior, which is why the sibling's head-count keyword is
                refused rather than reused.
            lag_scale: $c_L$, or ``None`` for $L^{-1/2}$. Resolved once here and then fixed; it is
                never renormalised by the number of available lags. Under attention fusion there is
                no sum to scale, so it resolves to one and an explicit value is refused.
            anchor_chunk: Anchors evaluated per proposal chunk, or ``None`` for all of them.
            lag_chunk: Lags evaluated per proposal chunk, or ``None`` for all of them.
            target_keep_index: Surviving target channels.
            source_keep_index: Surviving source channels.
            target_delays: Per-survivor target shift, which is where the channel gate stops being a
                pure gather.
            source_delays: The same for the source stream.
            init_weights: Run the generic initialisation pass and the repairs that follow it.

        Raises:
            ValueError: If a channel count, width or lag count is not positive or is otherwise
                inconsistent -- each of these builds a model that is wrong rather than one that
                fails, so each is refused here.
        """
        super().__init__()

        if int(c_y) < 1 or int(c_u) < 1:
            raise ValueError(
                f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
            )
        if int(max_lag) < 0:
            raise ValueError(f"max_lag must be >= 0, got {max_lag}")
        if int(d_model) % 2 != 0:
            # The clock pairs a sine with a cosine per frequency, so an odd width would leave one
            # coordinate of a pair unbuilt and the two halves misaligned by one.
            raise ValueError(
                f"d_model must be even for the metadata clock's sine and cosine pairs, got "
                f"{d_model}"
            )
        if float(residual_mu_scale) <= 0.0 or float(residual_logsigma_scale) <= 0.0:
            raise ValueError(
                f"the residual bounds must be > 0, got residual_mu_scale={residual_mu_scale}, "
                f"residual_logsigma_scale={residual_logsigma_scale}. A zero bound pins the update "
                f"at zero for every input while leaving a fully built head in the graph."
            )
        for name, value in (("anchor_chunk", anchor_chunk), ("lag_chunk", lag_chunk)):
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be >= 1 when given, got {value}")
        if str(source_stem) not in SOURCE_STEMS:
            raise ValueError(
                f"source_stem={source_stem!r} is not one of {sorted(SOURCE_STEMS)}. Each names a "
                f"different module tree and therefore a different checkpoint; an unrecognised "
                f"value cannot be resolved to one of them by anything downstream."
            )
        if str(lag_fusion) not in LAG_FUSIONS:
            raise ValueError(
                f"lag_fusion={lag_fusion!r} is not one of {sorted(LAG_FUSIONS)}. The two form the "
                f"latent update in different ways and share nothing downstream of it, so there is "
                f"no default to fall back on."
            )
        if bool(source_scalar_lift) and str(source_stem) != "pointwise":
            raise ValueError(
                f"source_scalar_lift widens each COEFFICIENT's own representation, and under "
                f"source_stem={source_stem!r} there are no per-coefficient representations to "
                f"widen: the stem emits one model-width state per stored step. Accepted and "
                f"ignored, the configuration would describe a lift the run never built."
            )
        if bool(source_values_withheld) and str(source_stem) != "pointwise":
            raise ValueError(
                f"source_values_withheld is the capacity control and is defined against the "
                f"pointwise representation, where withholding the value leaves exactly the "
                f"availability bit. Under source_stem={source_stem!r} it would leave a "
                f"convolution stack over an all-zero stream, whose output is a constant its own "
                f"bias already provides -- a different arm answering a different question, under "
                f"a name that says otherwise."
            )
        if str(lag_fusion) == "attention" and lag_scale is not None:
            raise ValueError(
                f"lag_scale={lag_scale} was given under attention fusion, which performs no "
                f"summation to scale: its aggregation is a normalised convex combination. A "
                f"factor here would shrink the update by an amount that has nothing to do with "
                f"the mechanism being compared. Leave it null."
            )
        if str(lag_fusion) == "attention" and lag_chunk is not None:
            raise ValueError(
                f"lag_chunk={lag_chunk} was given under attention fusion. The distribution is "
                f"normalised over the whole lag axis, so a chunked pass would renormalise within "
                f"each chunk and compute a different model rather than the same one in a different "
                f"order. Chunk the anchor axis instead, which is exact."
            )

        # The trimmed-grid geometry, which every mask, every target gather and the anchor ceiling
        # are built against.
        self.geometry = TrimmedRawGeometry(
            raw_len=int(sequence_length) * int(raw_per_step),
            decimation=int(raw_per_step),
            horizon=int(horizon),
            warmup=int(warmup_period),
        )

        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.horizon = int(horizon)
        self.raw_per_step = int(raw_per_step)
        self.warmup_period = int(warmup_period)
        self.c_y = int(c_y)
        self.c_u = int(c_u)
        self.use_up_st = bool(use_up_st)
        self.max_lag = int(max_lag)
        self.n_lags = int(max_lag) + 1
        self.mu_scale = float(mu_scale)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.coverage_floor = float(coverage_floor)
        self.persistence_residual = bool(persistence_residual)
        self.residual_mu_scale = float(residual_mu_scale)
        self.residual_logsigma_scale = float(residual_logsigma_scale)
        self.mean_only_residual = bool(mean_only_residual)
        self.source_scalar_lift = bool(source_scalar_lift)
        self.source_disabled = bool(source_disabled)
        self.source_values_withheld = bool(source_values_withheld)
        self.source_stem = str(source_stem)
        self.lag_fusion = str(lag_fusion)
        self.lag_attention_heads = int(lag_attention_heads)
        # Resolved once, from the CONFIGURED lag count, and then a plain float for the rest of the
        # model's life. Deriving it per batch from the available lags is the one change to this
        # line that would be silent: the same evidence would weigh differently at two anchors.
        #
        # Under attention fusion it is one: the aggregation is already a convex combination, so
        # there is no summation convention to apply and the forward's multiplication is the
        # identity. Resolved to a number rather than skipped, so every readout that reports $c_L$
        # reports what the arm actually used.
        self.lag_scale = (
            1.0
            if self.lag_fusion == "attention"
            else (default_lag_scale(self.n_lags) if lag_scale is None else float(lag_scale))
        )
        self.anchor_chunk = None if anchor_chunk is None else int(anchor_chunk)
        self.lag_chunk = None if lag_chunk is None else int(lag_chunk)
        self.horizon_embed_std = float(horizon_embed_std)
        self.head_init_calibration = bool(head_init_calibration)

        self.horizon_weight_halflife_steps = (
            None
            if horizon_weight_halflife_steps is None
            else float(horizon_weight_halflife_steps)
        )
        if self.horizon_weight_halflife_steps is not None:
            # Non-persistent, like every geometry-shaped tensor here: its length is $H$, so a
            # persistent copy would make a checkpoint trained at one horizon fail to load at
            # another and report it as a missing key rather than as a geometry mismatch.
            self.register_buffer(
                "horizon_weight",
                horizon_decay_weight(self.horizon_weight_halflife_steps, self.horizon),
                persistent=False,
            )

        # The metadata clock. Built once because it is a function of stored position and the
        # sequence length alone -- no source value and no recording-dependent quantity reaches it.
        self.register_buffer(
            "metadata_clock",
            self._build_metadata_clock(self.sequence_length, self.d_model),
            persistent=False,
        )

        self.target_gate = self._build_channel_gate(
            self.c_y, target_keep_index, target_delays
        )
        self.source_gate = self._build_channel_gate(
            self.c_u, source_keep_index, source_delays
        )

        # The adapters read their availability terms off the gates the forward actually applies,
        # never off a second copy of the constructor arguments. The composing mixin overrides this
        # hook to announce the warm-up as well as the shift, and falls back to the version below
        # for a stream with no warm-up at all.
        self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)

        self.target_encoder = CausalConvTransformerEncoder(
            d_model=self.d_model,
            sequence_length=self.sequence_length,
            conv_kernels=encoder_conv_kernels,
            conv_dilations=encoder_conv_dilations,
            num_attention_blocks=int(target_attention_blocks),
            num_heads=int(encoder_num_heads),
            d_ff=int(encoder_d_ff),
            attention_window=None,
            dropout=dropout,
        )

        # $W_A$, and its normalisation. Built here rather than inside the prior head -- which can
        # carry a clock path of its own -- because the conditioning state it produces is read by
        # the proposal head too, and a projection reachable only from inside the prior would have
        # to be applied twice to serve both. Bias-free so that a zeroed weight makes the term
        # EXACTLY zero whatever the normaliser's affine does, which is what makes the start exact
        # rather than approximate.
        self.clock_norm = nn.LayerNorm(self.d_model)
        self.clock_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # The prior head is built WITHOUT a clock path, deliberately: it receives $h_t$, already
        # summed above, so its two input norms see the conditioning state whole rather than one of
        # them seeing a pre-normalised sum.
        self.prior_head = FullLatentPriorHead(
            d_model=self.d_model,
            d_z=self.d_z,
            logvar_clamp=self.logvar_clamp,
            dropout=dropout,
            mu_scale=self.mu_scale,
            clock_dim=None,
        )

        # The source pathway. Two objects: one that forms each lag's source vector, and one that
        # turns those vectors into the latent update. Both are absent on the target-only arm --
        # ``None`` rather than an inert module, so a module-tree walk and a state dict both report
        # the pathway as missing rather than as present and unused.
        #
        # The pair is chosen by the two arm keywords and by nothing else. Neither object is asked
        # which the other is: the encoder reports its own per-lag width and owns its own gather, and
        # the fusion sizes itself from that width, so the four combinations compose without any of
        # them holding a case for the others.
        if self.source_disabled:
            self.source_encoder = None
            self.proposal_head = None
        else:
            source_width = self.c_u if self.source_gate is None else self.source_gate.out_channels
            if self.source_stem == "conv":
                self.source_encoder = ConvSourceStem(
                    c_u=source_width,
                    d_model=self.d_model,
                    # The mixin's own resolution, which combines each channel's warm-up wait with
                    # whatever shift the gate applied. Asked for here rather than recomputed, so
                    # the source stream announces its availability by exactly the rule the target
                    # stream does.
                    adapter=self._build_adapter(self.source_gate, self.c_u, dropout),
                    conv_kernels=encoder_conv_kernels,
                    conv_dilations=encoder_conv_dilations,
                    warmup_steps=self._combined_source_steps(),
                )
            else:
                self.source_encoder = PointwiseSourceEncoder(
                    c_u=source_width,
                    warmup_steps=self._combined_source_steps(),
                    scalar_lift=self.source_scalar_lift,
                    withhold_values=self.source_values_withheld,
                )
            if self.lag_fusion == "attention":
                self.proposal_head = LagAttentionFusion(
                    d_model=self.d_model,
                    d_z=self.d_z,
                    n_lags=self.n_lags,
                    source_dim=self.source_encoder.source_dim,
                    num_heads=self.lag_attention_heads,
                    mean_only=self.mean_only_residual,
                )
            else:
                self.proposal_head = LagProposalHead(
                    d_model=self.d_model,
                    d_z=self.d_z,
                    n_lags=self.n_lags,
                    source_dim=self.source_encoder.source_dim,
                    lag_embed_dim=int(lag_embed_dim),
                    hidden=proposal_hidden,
                    mean_only=self.mean_only_residual,
                )

        # One shared decoder, invoked twice per forward and receiving no source-derived tensor.
        # Its dropout must be zero, not merely small: invoking one module twice draws two
        # independent dropout masks, so the two branches would differ at initialisation even with
        # identical latent parameters.
        self.horizon_core = HorizonDecoderCore(
            d_hidden=int(decoder_hidden),
            horizon=self.horizon,
            kernel_size=int(horizon_kernel),
            depth=int(horizon_depth),
            film=bool(horizon_film),
            film_per_block=True,
            attention_blocks=int(horizon_attention_blocks),
        )
        self.decoder_out_channels = self._default_decoder_out_channels()
        self.decoder = BaselineFutureDecoder(
            core=self.horizon_core,
            d_model=self.d_z,
            out_channels=self.decoder_out_channels,
            d_hidden=int(decoder_hidden),
            dropout=0.0,
            logvar_clamp=self.logvar_clamp,
            persistence_residual=self.persistence_residual,
        )

        # The initialisation order is load-bearing, top to bottom.
        #
        #: How many depthwise convolutions the variance-preserving pass re-initialised. Recorded
        #: because the count is the only evidence that pass was not a silent no-op.
        self.n_depthwise_init = 0
        if init_weights:
            initialization(self)
            # Immediately after the generic pass. That pass reads a depthwise weight's fan wrongly
            # and starts the stem an order of magnitude too quiet, independent of the kernel, so no
            # kernel sweep could reveal it.
            self.n_depthwise_init = init_depthwise_(self)
        # After the generic pass, never before: it xavier-fills every linear layer and would
        # otherwise undo all three zeroings below, and with them the exact zero-update start.
        if self.proposal_head is not None:
            self.proposal_head.zero_output()
        nn.init.zeros_(self.clock_proj.weight)
        self._zero_init_film_generators()
        # The scalar lift is a loose-parameter block the generic pass does not reach, so this is
        # not a repair; it is here so the whole initialisation sequence reads in one place. Guarded
        # on the encoder's type rather than on the presence of a method: the comparator's stem has
        # no lift and never will, and a hook it answered would say otherwise.
        if isinstance(self.source_encoder, PointwiseSourceEncoder):
            self.source_encoder.reset_lift_parameters()

        if self.horizon_embed_std != 0.02:
            self._reinit_horizon_embedding()
        if self.head_init_calibration:
            self._calibrate_output_heads()
            self._calibrate_prior_scale()

    # ------------------------------------------------------------------
    # Construction hooks the mixins and this constructor call
    # ------------------------------------------------------------------
    @staticmethod
    def _build_channel_gate(
        declared_width: int,
        keep_index: Optional[Sequence[int]],
        delays: Optional[Sequence[int]],
    ) -> Optional[ChannelGate]:
        """Build one stream's causal input guard, or ``None`` when it has none.

        With neither argument the stream is ungated and no module is created, which is how this
        family represents an unguarded stream -- by the absence of a gate rather than by an
        identity one. With either, the missing half is filled in, because a gather without delays
        is far more likely to be a resolution bug than an intent.

        Args:
            declared_width: The stream's full declared channel count.
            keep_index: Surviving channel indices, or ``None`` for all of them.
            delays: Per-survivor delay in steps, or ``None`` for none.

        Returns:
            The gate, or ``None``.
        """
        if keep_index is None and delays is None:
            return None
        return ChannelGate(
            declared_width=int(declared_width), keep_index=keep_index, delays=delays
        )

    def _build_adapter(
        self, gate: Optional[ChannelGate], declared_width: int, dropout: float
    ) -> AvailabilityInputAdapter:
        r"""Build one stream's input adapter at the width and delays its gate emits.

        The fallback the composing mixin delegates to for a stream with no warm-up vector. The gate
        is the single source of truth for both width and delays: reading them back off the
        constructed gate rather than off the constructor arguments means the availability pattern
        cannot describe a guard the stream never received.

        Args:
            gate: The stream's guard, or ``None`` when it is unguarded.
            declared_width: The stream's declared channel count, used when there is no gate.
            dropout: Dropout probability inside the projection stack.

        Returns:
            The adapter.
        """
        width = declared_width if gate is None else gate.out_channels
        delays = None if gate is None else [int(value) for value in gate.delay.delay_steps]
        return AvailabilityInputAdapter(
            in_dim=width,
            d_model=self.d_model,
            sequence_length=self.sequence_length,
            dropout=dropout,
            delays=delays,
        )

    @staticmethod
    def _build_metadata_clock(sequence_length: int, width: int) -> torch.Tensor:
        r"""The value-free position encoding the prior and the proposal head both condition on.

        $$\chi_{t, 2k-2} = \sin\!\left(\frac{2\pi k t}{T}\right),
          \qquad
          \chi_{t, 2k-1} = \cos\!\left(\frac{2\pi k t}{T}\right),
          \qquad k = 1, \ldots, \tfrac{d}{2}.$$

        **What it is.** A deterministic function of stored position, identical for every recording
        and unchanged by any intervention on the source. With the warm-up vector fixed it contains
        the position from which the deterministic availability schedule can be inferred, which is
        why conditioning the prior on it lets an availability term cancel out of the divergence
        instead of being attributed to the source by every readout downstream.

        **What it is not.** It is not the same representation as an encode of a zeroed source
        stream, and no claim is made that the two are equivalent -- both are functions of position
        and configuration, and their representational adequacy can differ. It is also not
        sufficient for recording-dependent availability: if a genuine source-quality mask is
        introduced, this tensor cannot carry it, and the metadata input has to be extended
        explicitly and both branches retrained under the same declared conditioning.

        Args:
            sequence_length: $T$.
            width: $d$, which must be even so every frequency has both of its coordinates.

        Returns:
            The clock, $(T, d)$, in ``float32``.
        """
        steps = torch.arange(int(sequence_length), dtype=torch.float32)[:, None]
        frequencies = torch.arange(1, int(width) // 2 + 1, dtype=torch.float32)[None, :]
        angle = 2.0 * math.pi * steps * frequencies / float(sequence_length)
        clock = torch.empty(int(sequence_length), int(width), dtype=torch.float32)
        clock[:, 0::2] = torch.sin(angle)
        clock[:, 1::2] = torch.cos(angle)
        return clock

    def conditioning_state(self, target_state: torch.Tensor) -> torch.Tensor:
        r"""The state both the prior heads and the proposal head read: $h_t$.

        $$h_t = h^Y_t + W_A \operatorname{LayerNorm}(\chi_t).$$

        Computed once per forward and handed to both, which is the reason the projection lives on
        this module rather than inside the prior head. At initialisation $W_A$ is exactly zero, so
        $h_t$ is the target state and nothing else.

        Args:
            target_state: The target encoder's output $(B, T, d_h)$, over the full stored grid.

        Returns:
            The conditioning state $(B, T, d_h)$.
        """
        clock = self.metadata_clock[: target_state.shape[1]]
        return target_state + self.clock_proj(self.clock_norm(clock))

    # ------------------------------------------------------------------
    # Latent plumbing
    # ------------------------------------------------------------------
    def reparameterize_shared(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_full: torch.Tensor,
        logvar_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Draw one $\epsilon$ and produce both latents from it.

        $$z^{p,(k)} = \mu^p + \sigma^p \odot \epsilon^{(k)},
          \qquad z^{q,(k)} = \mu^q + \sigma^q \odot \epsilon^{(k)},$$

        so equal parameters give bitwise equal samples and the base-minus-full readout carries no
        independent sampling noise. Both branches are always sampled: decoding the base at its mean
        instead would make the two columns different estimators, and the difference between them
        partly a difference of sampling policy.

        The exact paired difference is
        $\sigma^p \odot [a + (e^{b} - 1)\odot\epsilon]$, which is deterministic only in the
        mean-only case. It is not an independent additive source random variable.

        Args:
            mu_prior: Prior mean $(B, A, d_z)$.
            logvar_prior: Prior log-variance $(B, A, d_z)$.
            mu_full: Full mean $(B, A, d_z)$.
            logvar_full: Full log-variance $(B, A, d_z)$.

        Returns:
            ``(z_prior, z_full)``, both $(B, A, d_z)$.
        """
        epsilon = torch.randn_like(mu_prior)
        z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
        z_full = mu_full + epsilon * torch.exp(0.5 * logvar_full)
        return z_prior, z_full

    def kld_tensor(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
    ) -> torch.Tensor:
        """The family's closed-form diagonal-Gaussian divergence, per anchor and per dimension.

        Kept because every consumer -- a control, a diagnostic figure, an offline evaluation --
        reaches this quantity through the model it was handed. The forward computes the same number
        from the residual form instead, which is cheaper and better conditioned near zero; the two
        agree, and a test in this package measures that rather than asserting it.

        Args:
            mu_prior: Prior mean.
            logvar_prior: Prior log-variance.
            mu_post: Full mean.
            logvar_post: Full log-variance.

        Returns:
            The per-anchor per-dimension divergence.
        """
        return closed_form_kld(mu_prior, logvar_prior, mu_post, logvar_post)

    def saturation_fractions(
        self, mu_prior: torch.Tensor, a: torch.Tensor, b: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """How often each bound is actually binding.

        A bound that is always active is a mis-set hyperparameter rather than a guard, and the only
        way to know is to count. Reported for the prior mean and for both residual channels, so the
        two residual bounds can be moved independently on evidence.

        Args:
            mu_prior: Prior mean.
            a: The bounded mean update.
            b: The bounded scale update, or ``None`` on the mean-only arm.

        Returns:
            Scalar fractions, keyed for the metric surface.
        """
        with torch.no_grad():
            fractions = {
                "mu_prior_sat_frac": (
                    mu_prior.abs() >= SATURATION_FRAC * self.mu_scale
                ).float().mean(),
                "residual_mu_sat_frac": (
                    a.abs() >= SATURATION_FRAC * self.residual_mu_scale
                ).float().mean(),
            }
            fractions["residual_logsigma_sat_frac"] = (
                torch.zeros((), device=a.device, dtype=torch.float32)
                if b is None
                else (
                    b.abs() >= SATURATION_FRAC * self.residual_logsigma_scale
                ).float().mean()
            )
        return fractions

    # ------------------------------------------------------------------
    # Initialisation repairs, run after the generic pass
    # ------------------------------------------------------------------
    @staticmethod
    def _zero_linear(layer: nn.Linear) -> None:
        """Zero a linear layer's weight and, if present, its bias."""
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _zero_init_film_generators(self) -> None:
        """Zero every modulation generator, so the horizon core starts as an identity.

        The core zeroes them in its own constructor and the generic pass xavier-fills them again.
        Re-zeroing is what makes the identity actually true: at step zero the latent enters the
        trajectory only through the projected state, not through the modulation, and it begins
        modulating as training drives these off zero.

        Initialisation only. Calling it on a trained model would discard exactly the modulation the
        latent learned to apply.
        """
        core = self.horizon_core
        layers: List[nn.Module] = []
        if core.film_gen is not None:
            layers.append(core.film_gen)
        if core.refine.film is not None:
            layers.extend(core.refine.film)
        for layer in layers:
            self._zero_linear(cast(nn.Linear, layer))

    def _reinit_horizon_embedding(self) -> None:
        """Re-seed the horizon-step embedding, so the forecast tokens start distinguishable.

        At the core's own small default the horizon tokens enter the refine stack almost perfectly
        correlated and the stack has to manufacture the whole trajectory shape in a
        latent-independent direction. A larger spread gives each token a distinct starting offset
        for the modulation to work from.

        Initialisation only; the constructor calls it on a non-default spread.
        """
        nn.init.normal_(
            self.horizon_core.horizon_embedding, mean=0.0, std=self.horizon_embed_std
        )

    def _calibrate_output_heads(self) -> None:
        r"""Place the shared decoder on the trivial predictor at initialisation.

        Xavier-filled heads emit a high-variance mean and an over-confident log-variance, so the
        initial score of a standardized target sits far above the trivial unit-Gaussian
        predictor's -- pressure the optimiser spends its first epochs undoing, and a confound in
        any comparison that reads them. The mean head is shrunk rather than zeroed, so a residual
        update still moves the two forecasts apart; the log-variance bias is seeded at the
        pre-image of zero under the bound, and its weight shrunk so the initial spread is small.

        Both branches share this one decoder, so they stay calibrated identically and every
        bitwise-at-initialisation property is preserved. Initialisation only, for a freshly trained
        model: applying it to transferred weights would discard what they carry.
        """
        lo, hi = self.logvar_clamp
        if not lo < 0.0 < hi:
            raise ValueError(
                f"output-head calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
            )
        self.decoder.mean_head.weight.data.mul_(0.02)
        self.decoder.logvar_head.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))
        self.decoder.logvar_head.weight.data.mul_(0.1)

    def _calibrate_prior_scale(self) -> None:
        r"""Pin the prior's log-variance at unit scale at initialisation.

        Nothing else in the initialisation places the prior's scale, and left xavier-filled it
        starts well below zero with only the scale regulariser pushing it back up. The head is a
        residual multilayer perceptron with no single bias governing its output level, so the
        decoder's shrink-and-bias recipe does not transfer: instead the final body layer's weight
        and the whole skip projection are zeroed and the final bias seeded at the pre-image of zero
        under the bound. The zeroed layers still receive gradient, and the residual update is built
        on the same raw tensor, so the exact zero-update start is untouched.

        Raises:
            ValueError: If the bound does not contain zero, which makes unit scale unreachable, or
                if the head's skip path is an identity, which the recipe cannot silence.
        """
        lo, hi = self.logvar_clamp
        if not lo < 0.0 < hi:
            raise ValueError(
                f"prior scale calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
            )
        head = self.prior_head.logvar_prior_head
        if not isinstance(head.skip_proj, nn.Linear):
            raise ValueError(
                "prior scale calibration requires a projected skip on the log-variance head; "
                "with d_model == d_z the skip is an identity and the output cannot be pinned"
            )
        self._zero_linear(head.skip_proj)
        final = cast(nn.Linear, head.body[-1])
        nn.init.zeros_(final.weight)
        final.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))


#: Module-name prefixes that make up the source pathway, for a parameter census that can tell an
#: arm's source budget from its target budget.
#:
#: The same two prefixes the warm start leaves at zero, and deliberately so: what a transfer treats
#: as the source pathway and what a comparison counts as the source pathway have to be one set, or
#: an arm's reported source budget describes a different boundary from the one its checkpoint was
#: built across.
SOURCE_MODULE_PREFIXES: Tuple[str, ...] = ("source_encoder.", "proposal_head.")


def pathway_parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Split a model's trainable parameter count into its source half and everything else.

    A comparison of arms is only readable beside their budgets: two arms differing by a fusion
    differ by however many parameters that fusion holds, and a predictive difference between them
    is not attributable to the mechanism until the budgets are on the table. The split is the
    number that says so, and one total cannot.

    Args:
        model: The constructed net, or any module whose source pathway carries the names in
            :data:`SOURCE_MODULE_PREFIXES`.

    Returns:
        ``{'total', 'source', 'target'}`` parameter counts, where *target* is the remainder --
        the encoder, the prior, the clock projection and the shared decoder.
    """
    total = source = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if name.startswith(SOURCE_MODULE_PREFIXES):
            source += count
    return {"total": total, "source": source, "target": total - source}


__all__ = [
    "LAG_FUSIONS",
    "REFUSED_KEYWORDS",
    "SOURCE_MODULE_PREFIXES",
    "SOURCE_STEMS",
    "LagResidualCore",
    "pathway_parameter_counts",
    "refuse_incompatible_keywords",
]
