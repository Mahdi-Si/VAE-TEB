r"""A strictly causal, anti-aliased map from a raw $4$ Hz signal onto the $4$ s token grid.

This is the one module in the package that is written rather than imported, because nothing under
``teb_vae`` performs causal anti-aliased decimation of a raw signal and the requirement cannot be
met by composing what exists. Everything downstream of it -- the encoders, the heads, the lag
attention, the decoder, the objective -- is the sibling package's, unchanged, so that a difference
in results is attributable to the input representation and to nothing else.

**The property being bought.** A stored wavelet or phase-harmonic coefficient at decimated step $t$
is a weighted average over raw samples on *both* sides of $t$, so a model conditioning on "the past
up to $t$" is conditioning on part of the interval it is asked to forecast. The front end below
cannot do that at all: every operation in it is either position-wise on the channel axis or
left-padded in time, and the decimation takes the **right** element of each stride group. Token $t$
is therefore a function of raw samples at index $\le s(t+1) - 1$ with $s$ the total stride, which
for $s = 16$ is exactly the anchor convention the rest of the model already uses -- there is no
off-by-one to negotiate between the two.

**The stack**, per stream:

$$
\text{featurise} \rightarrow
\underbrace{\text{stage}_1 \rightarrow \text{stage}_2 \rightarrow \text{stage}_3 \rightarrow
            \text{stage}_4}_{\text{stride } 2 \text{ each}}
\rightarrow \operatorname{RMSNorm}(d_{\mathrm{model}}) \rightarrow (B, T, d_{\mathrm{model}}) .
$$

Two bans hold throughout, and both are enforced rather than intended. No normaliser may reduce over
the time axis -- ``nn.GroupNorm`` reduces over $(C/G, T)$ within a group and is exactly the leak
this package exists to remove -- which :func:`refuse_time_pooling_norms` checks at construction. And
the anti-alias filter is a **buffer** applied with ``F.conv1d``, never an ``nn.Conv1d``:
``teb_vae/lag_attn/nets/blocks.py::initialization`` Xavier-fills every ``nn.Conv1d`` weight in the
model, so a fixed kernel stored as a layer would be silently overwritten with random values and the
anti-aliasing would quietly stop happening.

Run this module to print the stage, width, stride, reach and parameter table::

    python -m teb_vae.lag_attn_transformer_e2e.nets.frontend
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn

from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD
from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    LAYER_SCALE_INIT,
    GatedCausalConvBlock,
    RMSNorm,
)

#: Tap count $\tau$ of the fixed binomial anti-alias filter. Five, following the front end in the
#: tree this design descends from. A binomial kernel has an exact null at the old Nyquist for any
#: $\tau \ge 2$; more taps buy stopband depth at the fold point -- $|H(\pi/2)|$ is $0.5$ at three
#: taps and $0.25$ at five -- and cost reach, which is the budgeted resource. No ablation varies
#: it, so it is a module constant rather than a constructor argument, and there is deliberately no
#: switch that turns the filtering off.
ANTI_ALIAS_TAPS = 5

#: Number of stride-$2$ stages, and hence the total decimation $2^4 = 16$. Fixed rather than
#: configurable because the stage widths are derived from $d_{\mathrm{model}}$ as
#: $(d/4,\, d/2,\, 3d/4,\, d)$: the count and the width schedule are one decision, not two.
NUM_STAGES = 4

#: Depthwise kernel per stage at the production geometry, widest at the finest rate. Stage $1$ runs
#: at $4$ Hz where a $65$-tap kernel spans $16$ s -- the deceleration scale -- and stages $2$ to $4$
#: run at $2$, $1$ and $0.5$ Hz where a $15$-tap kernel already spans $7$, $14$ and $28$ s. Both
#: figures are the schedule of the front end in the tree this design descends from, with its
#: parallel multi-kernel first stage collapsed to its widest branch.
#:
#: A constructor default rather than a configuration key, following the precedent
#: ``ROPE_BASE`` sets in the sibling's blocks: no arm varies it, so a config key would be a
#: configuration surface with nothing behind it. The reach guard bounds any future choice.
FRONTEND_KERNELS: Tuple[int, ...] = (65, 15, 15, 15)

#: Channels the featurisation emits: masked value, validity, gated first difference.
FEATURE_CHANNELS = 3

#: Normalisers that either pool over time or couple the batch, and are therefore refused anywhere
#: inside the front end. ``SyncBatchNorm`` is listed explicitly because it does not subclass
#: ``BatchNorm1d``; the ``InstanceNorm`` family is listed because it reduces over the spatial --
#: here temporal -- axis by construction.
TIME_POOLING_NORMS = (
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def binomial_lowpass(taps: int) -> torch.Tensor:
    r"""Return the unit-sum binomial low-pass kernel of ``taps`` taps.

    $$h_i = \binom{\tau - 1}{i} \Big/ 2^{\tau - 1}, \qquad i = 0, \ldots, \tau - 1 .$$

    Its transfer function is $H(\omega) = \left(\cos(\omega/2)\right)^{\tau - 1} e^{-j\omega(\tau-1)/2}$,
    so $H(\pi) = 0$ **exactly** for every $\tau \ge 2$: the old Nyquist, which is what a factor-2
    decimation would fold onto DC, is annihilated rather than merely attenuated. That exactness is
    what lets the alias test assert against $10^{-12}$ in float64 instead of against a threshold
    somebody had to choose.

    Args:
        taps: Tap count $\tau$. Must be at least $2$; a single tap is the identity, which is the
            no-anti-aliasing case this front end does not offer.

    Returns:
        A ``(taps,)`` float32 tensor summing to $1$.

    Raises:
        ValueError: If ``taps`` is below $2$.
    """
    if int(taps) < 2:
        raise ValueError(
            f"an anti-alias filter needs at least 2 taps to have a null at the old Nyquist, got "
            f"taps={taps}; a 1-tap kernel is the identity and would decimate without filtering"
        )
    coefficients = torch.tensor(
        [math.comb(int(taps) - 1, index) for index in range(int(taps))], dtype=torch.float32
    )
    return coefficients / coefficients.sum()


class CausalAntiAliasDecimate(nn.Module):
    r"""Fixed causal low-pass FIR, then right-offset subsampling.

    $$
    \tilde x[n] = \sum_{i=0}^{\tau-1} h_i\, x[n - i], \qquad
    \mathrm{out}[t] = \tilde x[s\,t + s - 1] .
    $$

    **The offset is right, not left or centre**, and that is the whole reason this module exists in
    this form. Composed over the front end's four stride-2 stages the total stride is $16$ and token
    $t$'s newest input sample is raw index $16t + 15$, which is exactly
    ``TrimmedRawGeometry.n_raw(t)``. A centred offset would be *more* conservative and would still
    pass a causality probe while silently discarding the newest quarter-second of every token; a
    left offset would read the future.

    **The coefficients are a non-persistent buffer applied with** ``F.conv1d``, never an
    ``nn.Conv1d``. The model-wide ``initialization`` helper Xavier-fills every ``nn.Conv1d`` weight
    it walks, so a fixed kernel held as a layer would be replaced by random values with no error and
    no observable symptom beyond aliased high-frequency energy appearing as real variability.
    Non-persistent because it is a constant of the architecture: a checkpoint that carried it would
    fail to load the moment the tap count changed, reported as a missing key rather than as what it
    is.

    There is deliberately no ``antialias=False`` switch. A decimation without the filter is a
    different model, not a setting of this one.

    Shapes:
        Input:  ``(B, C, L)``
        Output: ``(B, C, L // stride)``
    """

    # Declared so the buffer types as a tensor rather than as the ``Tensor | Module`` union
    # ``__getattr__`` advertises.
    fir: torch.Tensor

    def __init__(self, channels: int, stride: int) -> None:
        r"""Build the fixed depthwise filter bank and record the padding it needs.

        Args:
            channels: Channel count $C$. The same kernel is applied to every channel, so the
                filter bank is depthwise and carries no cross-channel mixing.
            stride: Decimation factor $s$.

        Raises:
            ValueError: If ``channels`` or ``stride`` is not positive.
        """
        super().__init__()
        if int(channels) < 1 or int(stride) < 1:
            raise ValueError(
                f"channels and stride must both be positive; got channels={channels}, "
                f"stride={stride}"
            )
        self.channels = int(channels)
        self.stride = int(stride)
        self.taps = int(ANTI_ALIAS_TAPS)
        self.left_padding = self.taps - 1

        # (C, 1, tau): one group per channel, so F.conv1d(groups=C) applies the same fixed kernel
        # to each independently. Registered rather than assigned so it follows the module across
        # devices and dtypes.
        kernel = binomial_lowpass(self.taps).view(1, 1, -1).repeat(self.channels, 1, 1)
        self.register_buffer("fir", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Low-pass with left-only padding, then take the last sample of each stride group.

        Args:
            x: Input of shape ``(B, C, L)``.

        Returns:
            Output of shape ``(B, C, L // stride)``.

        Raises:
            ValueError: If the channel axis does not match the filter bank.
        """
        # ``.ndim`` and a bare ``.shape`` subscript rather than ``.dim()`` and ``int(...)``:
        # ``tests/test_ddp_reachability.py`` walks every ``forward`` in this package and rejects a
        # conditional holding any call, because a call is how a forward reads a tensor's *content*
        # -- and a content-dependent branch drops parameters from the graph on some ranks and not
        # others, which hangs a DDP run rather than failing it. Shape metadata is admitted; this
        # guard says the same thing in the admitted form.
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected (B, {self.channels}, L), got shape {tuple(x.shape)}"
            )
        padded = F.pad(x, (self.left_padding, 0))
        filtered = F.conv1d(padded, self.fir.to(dtype=x.dtype), groups=self.channels)
        return filtered[..., self.stride - 1 :: self.stride]

    def extra_repr(self) -> str:
        """Report the width, the stride and the filter, for readable module trees."""
        return f"{self.channels}, stride={self.stride}, taps={self.taps}, offset=right"


def featurize(raw: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    r"""Turn a normalized raw signal and its decimated validity into three raw-rate channels.

    $$
    \left[\; \bar x_n m_n, \quad m_n, \quad
        (\bar x_n - \bar x_{n-1})\, m_n m_{n-1} \;\right],
    \qquad
    m_n = \mathbb 1\!\left[\,\mathrm{weight}_{\lfloor n/r \rfloor} \ge \theta\,\right]
          \wedge \operatorname{isfinite}(x_n),
    $$

    where $\bar x$ is the signal as the loader produced it -- already z-scored, so this function
    owns no statistics of its own -- and $r$ is the raw samples per decimated step, derived from the
    two lengths rather than passed, so a mismatched pair cannot be silently accepted.

    Three properties are load-bearing:

    * **The validity mask is a channel.** A gap is then representable, rather than being
      indistinguishable from a genuine normalised zero.
    * **Invalid positions are neutralised in standardised space**, not raw space. Zeroing the raw
      value *before* a z-score would map a gap to $-\mu/\sigma \approx -7$ for the target signal --
      an extreme-bradycardia-looking constant that the low-pass then smears across the following
      tokens.
    * **The first difference is gated on both endpoints**, so the first valid sample after a gap
      injects no spurious slope. The replicate pad at $n = 0$ makes $\Delta x_0$ exactly zero for
      the same reason.

    The threshold $\theta$ is :data:`~teb_vae.lag_attn_rws.nets.raw_masks.VALID_THRESHOLD`, imported
    rather than restated: it is this repository's definition of a valid step and it is what the loss
    masks use, so a second float comparison here could drift from the mask the objective scores
    against.

    Args:
        raw: The loader-normalized signal, $(B, L)$.
        weight: The decimated validity signal, $(B, T)$ with $L = rT$.

    Returns:
        A $(B, 3, L)$ tensor, channel-major, all-finite regardless of the input.

    Raises:
        ValueError: If either tensor is not 2-D, if the batch axes disagree, or if the raw length
            is not a positive multiple of the weight length.
    """
    if raw.dim() != 2 or weight.dim() != 2:
        raise ValueError(
            f"expected a 2-D raw signal (B, L) and a 2-D weight (B, T), got shapes "
            f"{tuple(raw.shape)} and {tuple(weight.shape)}"
        )
    if int(raw.shape[0]) != int(weight.shape[0]):
        raise ValueError(
            f"raw and weight disagree on the batch axis: {int(raw.shape[0])} against "
            f"{int(weight.shape[0])}"
        )
    raw_length, steps = int(raw.shape[-1]), int(weight.shape[-1])
    if steps < 1 or raw_length % steps != 0:
        raise ValueError(
            f"raw length {raw_length} is not a positive multiple of the weight length {steps}; "
            f"the two grids must agree or the expanded mask would slip against the signal"
        )

    # A step is valid only when it is fully valid *and* the sample itself is finite. The finiteness
    # term is kept even though the pipeline sanitises before writing: one NaN would otherwise
    # propagate through the low-pass into every following token.
    valid = (weight >= VALID_THRESHOLD).repeat_interleave(raw_length // steps, dim=-1)
    valid = valid & torch.isfinite(raw)

    mask = valid.to(raw.dtype)
    # ``where`` rather than a multiply: a non-finite sample times a zero mask is still NaN.
    value = torch.where(valid, raw, raw.new_zeros(()))

    # Replicate padding, written as a concatenation because F.pad's replicate mode needs a channel
    # axis this tensor does not have. At n = 0 both the value and the mask repeat, so the gated
    # difference is exactly zero there.
    previous_value = torch.cat((value[..., :1], value[..., :-1]), dim=-1)
    previous_mask = torch.cat((mask[..., :1], mask[..., :-1]), dim=-1)
    delta = (value - previous_value) * mask * previous_mask

    # ``value`` is already zero wherever the mask is, so no second multiply is needed.
    return torch.stack((value, mask, delta), dim=1)


class CausalFrontendStage(nn.Module):
    r"""One stride-2 front-end stage: widen, gated causal convolution, anti-aliased decimation.

    $$
    P = W\,x, \qquad
    G = \operatorname{GatedCausalConvBlock}_k(P), \qquad
    \mathrm{out} = \operatorname{Decimate}_2(G) .
    $$

    The convolution block is the sibling's, imported and used whole in the $(B, T, C)$ layout it
    already presents, which settles a hazard rather than defending against it: the block adds its
    residual at the **full** rate and one decimator then runs on the sum, so there is no separate
    skip path that could be decimated by a different operator and drift out of sample alignment.

    The widening lives in the pointwise projection ahead of the block because the block holds its
    width constant, and it carries a bias -- the one place in this stack that does. A fully invalid
    window featurises to an exactly zero vector, and an exactly zero token entering repeated
    pre-normalisation is the numerical accident the sibling's input adapter documents reaching
    gradient norms around $10^{26}$; a bias turns "this window is empty" into a learnable constant
    instead.

    Shapes:
        Input:  ``(B, L, C_in)``
        Output: ``(B, L // 2, C_out)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 2,
        dropout: float = 0.0,
        layer_scale_init: float = LAYER_SCALE_INIT,
    ) -> None:
        """Build the projection, the gated convolution and the decimator.

        Args:
            in_channels: Input width $C_{\\mathrm{in}}$.
            out_channels: Output width $C_{\\mathrm{out}}$.
            kernel_size: Depthwise kernel width $k$, applied at ``out_channels``.
            stride: Decimation factor of this stage.
            dropout: Dropout probability on the convolution block's residual branch.
            layer_scale_init: Initial LayerScale gain inside the convolution block.
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = nn.Linear(self.in_channels, self.out_channels)
        self.block = GatedCausalConvBlock(
            self.out_channels,
            kernel_size=int(kernel_size),
            dropout=dropout,
            layer_scale_init=layer_scale_init,
        )
        self.decimate = CausalAntiAliasDecimate(self.out_channels, int(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Widen, convolve causally, and decimate.

        Args:
            x: Input of shape ``(B, L, C_in)``.

        Returns:
            Output of shape ``(B, L // stride, C_out)``.
        """
        h = self.block(self.proj(x))
        # The decimator is channel-major, like every convolution in this repository; the transpose
        # pair is internal and does not leak into the stage's interface.
        return self.decimate(h.transpose(1, 2)).transpose(1, 2)


def refuse_time_pooling_norms(module: nn.Module, *, label: str = "front end") -> None:
    """Raise if any submodule normalises over time or couples the batch.

    The front end's entire claim is that token $t$ is a function of raw samples at index
    $\\le s(t+1) - 1$. A normaliser whose statistics run over the time axis breaks that claim
    everywhere at once and leaves no symptom in a loss curve: ``nn.GroupNorm`` reduces over
    $(C/G, T)$ within a group, and the ``BatchNorm`` family additionally makes every sample's output
    depend on the rest of the batch. Both are refused at construction, so this is a standing guard
    rather than something a test happens to notice.

    Args:
        module: Subtree to walk.
        label: Named in the message, so a failure says which stack was rejected.

    Raises:
        ValueError: Naming the first offending submodule, its class and the reason.
    """
    for name, child in module.named_modules():
        if isinstance(child, TIME_POOLING_NORMS):
            raise ValueError(
                f"{label} holds {type(child).__name__} at '{name or '<root>'}'; normalisers that "
                f"reduce over the time axis, or over the batch, make every history state carry an "
                f"image of its own future -- which is the leak this input representation exists to "
                f"remove. Normalise over the channel axis only (RMSNorm)."
            )


class CausalRawFrontend(nn.Module):
    r"""Map one raw $4$ Hz signal onto the token grid, reading no sample after each token's anchor.

    $$
    (B, rT) \times (B, T) \;\longrightarrow\; (B, T, d_{\mathrm{model}}),
    \qquad r = \prod_i s_i = 2^{4} = 16 .
    $$

    Four stride-2 stages at widths $(d/4,\, d/2,\, 3d/4,\, d)$ -- $32/64/96/128$ at the production
    width, $8/16/24/32$ at the tiny one -- so there is no width configuration to get wrong and no
    "the last stage must equal $d_{\mathrm{model}}$" invariant to violate. The trailing
    ``RMSNorm`` mirrors the encoder's own final norm: everything downstream is calibrated to a
    normalised state, and an unnormalised residual stream whose scale grows with depth would move
    the operating point for reasons that have nothing to do with the input representation.

    **At initialisation this is approximately a linear mix of the decimated**
    ``[value, mask, delta]`` **channels.** ``LayerScale`` starts every convolution block's residual
    branch at $10^{-2}$, so each stage is close to its pointwise projection composed with the fixed
    low-pass. That is a sane start rather than a defect, but it is worth stating: a freshly built
    front end has not yet learned any temporal structure, and the first epochs are the stages
    finding it.

    Shapes:
        Input:  ``(B, r T)`` raw, ``(B, T)`` weight
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        *,
        d_model: int,
        raw_per_step: int,
        reach_budget: int,
        kernels: Sequence[int] = FRONTEND_KERNELS,
        dropout: float = 0.0,
        layer_scale_init: float = LAYER_SCALE_INIT,
    ) -> None:
        r"""Build the stages and refuse a stack that reaches further back than the warm-up allows.

        Args:
            d_model: Output width $d$. Must be divisible by $4$, which the width schedule needs.
            raw_per_step: Raw samples per decimated step $r$. Must equal the front end's own total
                stride, or the token grid it emits would not be the grid the rest of the model
                indexes.
            reach_budget: Raw samples the front end may reach back over. The model passes
                ``warmup_period * raw_per_step``: a stack reaching further would let a *trained*
                anchor read the zero-padded convolution transient at the segment's start. Not a
                configuration key and not a caller's preference -- it is a fact about the geometry.
            kernels: One depthwise kernel per stage. Defaults to :data:`FRONTEND_KERNELS`.
            dropout: Dropout probability on each convolution block's residual branch.
            layer_scale_init: Initial LayerScale gain on each branch.

        Raises:
            ValueError: If ``d_model`` is not divisible by $4$, if ``kernels`` does not have
                :data:`NUM_STAGES` entries, if ``raw_per_step`` disagrees with the total stride, or
                if the accumulated reach exceeds ``reach_budget`` -- each naming both values.
        """
        super().__init__()
        if int(d_model) % 4 != 0:
            raise ValueError(
                f"d_model must be divisible by 4 so the stage widths (d/4, d/2, 3d/4, d) are "
                f"integral, got d_model={d_model}"
            )
        if len(tuple(kernels)) != NUM_STAGES:
            raise ValueError(
                f"expected {NUM_STAGES} kernels, one per stride-2 stage, got "
                f"{len(tuple(kernels))}: {tuple(kernels)}"
            )

        self.d_model = int(d_model)
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.reach_budget = int(reach_budget)

        width = self.d_model // 4
        widths = (width, 2 * width, 3 * width, self.d_model)
        self.stages = nn.ModuleList(
            [
                CausalFrontendStage(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for in_channels, out_channels, kernel in zip(
                    (FEATURE_CHANNELS,) + widths[:-1], widths, self.kernels
                )
            ]
        )
        self.output_norm = RMSNorm(self.d_model)

        if int(raw_per_step) != self.total_stride:
            raise ValueError(
                f"raw_per_step={raw_per_step} disagrees with the front end's total stride "
                f"{self.total_stride}; token t's newest input sample is raw index "
                f"{self.total_stride} * (t + 1) - 1, so a mismatch would silently emit a different "
                f"grid from the one the model's anchors index"
            )
        if self.reach_samples > self.reach_budget:
            raise ValueError(
                f"front end reaches {self.reach_samples} raw samples but the budget is "
                f"{self.reach_budget} (warmup_period * raw_per_step); a stack reaching further "
                f"lets a trained anchor read the zero-padded convolution transient at the "
                f"segment's start. Narrow the kernels {self.kernels} or raise warmup_period."
            )
        refuse_time_pooling_norms(self)

    @property
    def stage_modules(self) -> Tuple[CausalFrontendStage, ...]:
        """The stages as their own type; ``nn.ModuleList`` iterates as bare ``nn.Module``."""
        return tuple(cast(CausalFrontendStage, stage) for stage in self.stages)

    @property
    def total_stride(self) -> int:
        """Raw samples per emitted token: the product of the stage strides."""
        stride = 1
        for stage in self.stage_modules:
            stride *= stage.decimate.stride
        return stride

    @property
    def stage_reach_samples(self) -> Tuple[int, ...]:
        r"""Raw samples one output sample of each stage depends on, itself included.

        A **count**, matching the ``receptive_field`` convention the sibling's blocks use, so a
        reach of $1$ means "this sample only". Accumulated from the *built* modules rather than
        recomputed from the constructor arguments, so the reported number cannot disagree with the
        stack that produced it:

        $$
        R = 2 + \sum_{i=1}^{4} \left(k_i + \tau - 2\right) \prod_{j<i} s_j ,
        $$

        where each stage contributes $(k_i - 1)$ samples from its depthwise kernel and $(\tau - 1)$
        from the anti-alias filter, scaled by the stride already accumulated below it, and the
        leading $2$ is the featurisation's one-sample first difference plus the sample itself.
        """
        reaches = []
        extra = 1  # the first difference reaches one raw sample further back than the sample itself
        stride_below = 1
        for stage in self.stage_modules:
            per_sample = (stage.block.receptive_field - 1) + (stage.decimate.taps - 1)
            extra += per_sample * stride_below
            stride_below *= stage.decimate.stride
            reaches.append(extra + 1)
        return tuple(reaches)

    @property
    def reach_samples(self) -> int:
        """Raw samples one output token depends on, itself included."""
        return self.stage_reach_samples[-1]

    def forward(self, raw: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Featurise, run the four stages, and normalise the emitted token.

        Args:
            raw: The loader-normalized signal, ``(B, r T)``.
            weight: The decimated validity signal, ``(B, T)``.

        Returns:
            The stream's token-major representation, ``(B, T, d_model)``.

        Raises:
            ValueError: If either input is not 2-D, or if the raw length is not ``total_stride``
                times the weight length. :func:`featurize` only requires the two grids to be
                commensurate; this is the stricter check, that the ratio is the stride this front
                end actually decimates by, and it is the one that keeps the emitted grid the grid
                the model's anchors index.
        """
        # Shape metadata only, in the form the DDP forward walk admits; see the note in
        # :meth:`CausalAntiAliasDecimate.forward` for why a call in a conditional is refused here.
        if raw.ndim != 2 or weight.ndim != 2:
            raise ValueError(
                f"expected a 2-D raw signal (B, L) and a 2-D weight (B, T), got shapes "
                f"{tuple(raw.shape)} and {tuple(weight.shape)}"
            )
        expected = int(weight.shape[-1]) * self.total_stride
        if raw.shape[-1] != expected:
            raise ValueError(
                f"expected a raw signal of {expected} samples for a weight of "
                f"{tuple(weight.shape)} at stride {self.total_stride}, got shape "
                f"{tuple(raw.shape)}"
            )

        # (B, 3, L) channel-major from the featurisation, transposed once into the token-major
        # layout the convolution blocks present, and held there for the whole cascade.
        h = featurize(raw, weight).transpose(1, 2)
        for stage in self.stages:
            h = stage(h)
        return self.output_norm(h)

    def extra_repr(self) -> str:
        """Report the width, the stride, the kernels and the reach against its budget."""
        return (
            f"d_model={self.d_model}, total_stride={self.total_stride}, kernels={self.kernels}, "
            f"reach={self.reach_samples}/{self.reach_budget} raw samples"
        )


def _print_table(name: str, frontend: CausalRawFrontend) -> None:
    """Print one geometry's stage, width, stride, reach and parameter table.

    Args:
        name: Label for the geometry.
        frontend: The built front end to describe.
    """
    seconds_per_sample = SECONDS_PER_STEP / frontend.total_stride
    total = sum(parameter.numel() for parameter in frontend.parameters())

    print(f"{name}: d_model={frontend.d_model}, total stride {frontend.total_stride}")
    print(f"  {'stage':>5}  {'width':>5}  {'kernel':>6}  {'stride':>6}  {'reach':>18}  {'params':>8}")
    stride_below = 1
    for index, (stage, reach) in enumerate(
        zip(frontend.stage_modules, frontend.stage_reach_samples), 1
    ):
        stride_below *= stage.decimate.stride
        parameters = sum(parameter.numel() for parameter in stage.parameters())
        print(
            f"  {index:>5}  {stage.out_channels:>5}  {stage.block.conv.kernel_size:>6}  "
            f"{stride_below:>6}  {reach:>7} raw / {reach * seconds_per_sample:>6.1f} s  "
            f"{parameters:>8,}"
        )
    print(f"  {'norm':>5}  {frontend.d_model:>5}  {'-':>6}  {'-':>6}  {'-':>18}  "
          f"{frontend.output_norm.weight.numel():>8,}")
    print(
        f"  reach {frontend.reach_samples} raw samples "
        f"({frontend.reach_samples * seconds_per_sample:.1f} s) against a budget of "
        f"{frontend.reach_budget} ({frontend.reach_budget * seconds_per_sample:.1f} s)"
    )
    print(f"  total {total:,} parameters per stream, {2 * total:,} for both\n")


def main() -> None:
    """Print the stage table for the production and the smoke geometry.

    The numbers below are written here so the table can be printed without a YAML file or a test
    fixture; ``configs/default.yaml`` is the source of truth for an actual run. The budget is
    ``warmup_period * raw_per_step`` in both cases.
    """
    print("Causal raw front end: one per stream, anti-aliased and strictly one-sided")
    print(f"  seconds per decimated step : {SECONDS_PER_STEP:.0f}")
    print(f"  anti-alias filter          : binomial, {ANTI_ALIAS_TAPS} taps, null at old Nyquist\n")

    _print_table(
        "production (warmup_period 30)",
        CausalRawFrontend(d_model=128, raw_per_step=16, reach_budget=30 * 16),
    )
    _print_table(
        "smoke (warmup_period 6)",
        CausalRawFrontend(
            d_model=32, raw_per_step=16, reach_budget=6 * 16, kernels=(5, 3, 3, 3)
        ),
    )
    print(
        "  Token t reads raw samples up to index 16(t + 1) - 1 and no further, which is exactly\n"
        "  the anchor convention the rest of the model uses."
    )


if __name__ == "__main__":
    main()
