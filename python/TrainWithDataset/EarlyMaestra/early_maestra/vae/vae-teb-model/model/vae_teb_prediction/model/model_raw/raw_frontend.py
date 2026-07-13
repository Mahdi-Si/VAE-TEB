r"""The strictly-causal, anti-aliased, multiscale raw front end (Sprint 1).

This module maps a raw $4$ Hz FHR/UP signal $(B, L_{\mathrm{raw}})$ plus a validity mask onto the
low-rate token grid $(B, T, d_{\mathrm{raw}})$ that the inherited v3 encoders consume, replacing
the fixed scattering/phase adapters. Everything here is **strictly causal** (left-only padding,
per-timestep / cumulative normalisation, right-offset decimation), which is the licence for
reading the downstream $K_t$ as a transfer-entropy surrogate (approach doc G0, extended into the
front end).

Pipeline per stream $s \in \{y, u\}$:

$$
X^{(s)}_0 = [\,x_{\mathrm{std}},\, m,\, \Delta x_{\mathrm{std}}\,]
\xrightarrow{\text{4 stride-2 blocks}} (B, d_{\mathrm{raw}}, \tilde T)
\xrightarrow{\text{crop } [\mathrm{CROP}:\tilde T-\mathrm{CROP})} (B, d_{\mathrm{raw}}, T)
\xrightarrow{\text{transpose}} (B, T, d_{\mathrm{raw}}).
$$

Key design choices (see the Sprint-1 design notes):
- The anti-alias low-pass is a **fixed FIR applied functionally** (:func:`torch.nn.functional.conv1d`
  over a ``register_buffer``), never an :class:`torch.nn.Conv1d` -- the shared ``initialization``
  helper Xavier-overwrites every ``nn.Conv1d.weight``, which would clobber the binomial kernel.
- **Decimation takes the right element of each stride group** (``x[..., stride-1::stride]``), so the
  uncropped token $t'$ has causal endpoint $D(t'+1)-1$, matching :func:`geometry.n_raw`.
- **All convolutions are stride-1**; the only length reduction happens in the anti-alias step, so
  the low-pass genuinely precedes subsampling (sampling theorem).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vae_teb_prediction.model.model_raw.raw_mask_constants import SENTINEL as _DEFAULT_SENTINEL
from model.vae_teb_prediction.model.model_raw.reuse import CausalConv1d, CausalGroupNorm


# =============================================================================
# Anti-alias downsampling (S1-T01)
# =============================================================================
_FIR_KERNELS = {
    "binomial5": [1.0, 4.0, 6.0, 4.0, 1.0],
    "binomial3": [1.0, 2.0, 1.0],
    "triangular5": [1.0, 2.0, 3.0, 2.0, 1.0],
}


def antialias_fir(name: str) -> torch.Tensor:
    r"""Return a normalised (unit-sum) 1-D low-pass FIR by name.

    Args:
        name: One of ``binomial5`` ($[1,4,6,4,1]/16$), ``binomial3``, ``triangular5``.

    Returns:
        A 1-D float tensor summing to $1$.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in _FIR_KERNELS:
        raise ValueError(
            f"unknown antialias_kernel {name!r}; choose from {sorted(_FIR_KERNELS)}"
        )
    w = torch.tensor(_FIR_KERNELS[name], dtype=torch.float32)
    return w / w.sum()


class CausalAntiAliasDownsample1d(nn.Module):
    r"""Causal low-pass FIR followed by right-offset stride subsampling.

    The FIR is left-padded by $K_{\mathrm{lp}}-1$ (causal) and applied depthwise via
    :func:`torch.nn.functional.conv1d` over a **buffer** (so it is invisible to Xavier
    re-initialisation and absent from :meth:`parameters`). Subsampling keeps the **right**
    element of each stride group, ``out[t] = filtered[stride*t + (stride-1)]``, so token $t$'s
    causal endpoint is $\mathrm{stride}\cdot t + (\mathrm{stride}-1)$.

    With ``antialias=False`` the low-pass is skipped and the same right-offset decimation is
    applied directly (the ``stages:[16]`` / anti-alias ablation).

    Shapes:
        Input:  $(B, C, L)$
        Output: $(B, C, \lceil L / \mathrm{stride} \rceil)$ (exact $L/\mathrm{stride}$ when
            ``stride`` divides $L$, which holds for every production length).
    """

    def __init__(
        self,
        num_channels: int,
        *,
        stride: int = 2,
        antialias: bool = True,
        antialias_kernel: str = "binomial5",
    ) -> None:
        """Initialise the downsampler.

        Args:
            num_channels: Channel count $C$ (the FIR is depthwise at this width).
            stride: Subsampling stride (also the decimation factor of this stage).
            antialias: If ``True`` low-pass before subsampling; if ``False`` subsample only.
            antialias_kernel: FIR name passed to :func:`antialias_fir`.
        """
        super().__init__()
        self.num_channels = int(num_channels)
        self.stride = int(stride)
        self.antialias = bool(antialias)
        if self.antialias:
            fir = antialias_fir(antialias_kernel)  # (K,)
            self.k_lp = int(fir.numel())
            lp = fir.view(1, 1, -1).repeat(self.num_channels, 1, 1).contiguous()  # (C, 1, K)
            self.register_buffer("lp", lp)
        else:
            self.k_lp = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Low-pass (optional) then right-offset subsample ``(B, C, L) -> (B, C, ceil(L/stride))``."""
        if self.antialias:
            x = F.pad(x, (self.k_lp - 1, 0))
            x = F.conv1d(x, self.lp, groups=self.num_channels)
        return x[:, :, self.stride - 1 :: self.stride]


# =============================================================================
# Causal normalisation (S1-T02a) + forbidden-norm guard (S1-T02b)
# =============================================================================
class CumulativeLayerNorm(nn.Module):
    r"""Cumulative layer norm (cLN), the causal replacement for global layer norm (Conv-TasNet).

    Statistics pool over **channels and cumulative time (past-and-present only)** with count
    $C\,(t+1)$ at step $t$:

    $$
    \mu_t = \frac{1}{C(t+1)}\sum_{c}\sum_{t'\le t} x_{c,t'},\qquad
    \sigma^2_t = \frac{1}{C(t+1)}\sum_{c}\sum_{t'\le t} x_{c,t'}^2 - \mu_t^2,
    $$

    $$
    \hat x_{c,t} = \frac{x_{c,t} - \mu_t}{\sqrt{\sigma^2_t + \epsilon}}\,\gamma_c + \beta_c .
    $$

    Because the pooling spans channels, $\sigma^2_0$ is a *cross-channel* variance (generally
    positive), so the per-channel $t=0$ variance-collapse worry does not arise; ``clamp_min(0)``
    plus ``eps`` still covers the constant/$C=1$ edge case. Being an inclusive prefix over time,
    $\hat x_{:,t}$ depends only on $x_{:,\le t}$ (strictly causal).

    Shapes:
        Input/Output: $(B, C, T)$.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        """Initialise with per-channel affine parameters.

        Args:
            num_channels: Channel count $C$.
            eps: Numerical-stability term added to the variance.
        """
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``(B, C, T)`` over channels and cumulative (past-and-present) time.

        The cumulative reduction runs in ``float32`` regardless of the input dtype: the pooling
        count $C(t+1)$ and the cumulative sum-of-squares grow with $T$ and overflow ``float16``
        (max $65504$) within a single length-$5280$ stage under AMP, and the one-pass
        $E[x^2]-E[x]^2$ variance is cancellation-prone; ``float32`` avoids both. The output is cast
        back to the input dtype.
        """
        in_dtype = x.dtype
        xf = x.float()
        _, C, T = xf.shape
        s1 = xf.sum(dim=1)             # (B, T)
        s2 = (xf * xf).sum(dim=1)      # (B, T)
        cs1 = s1.cumsum(dim=-1)        # (B, T) inclusive prefix over time
        cs2 = s2.cumsum(dim=-1)        # (B, T)
        cnt = C * torch.arange(1, T + 1, device=xf.device, dtype=torch.float32)  # (T,)
        mean = cs1 / cnt               # (B, T)
        var = (cs2 / cnt - mean * mean).clamp_min(0.0)                    # (B, T)
        xhat = (xf - mean.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + self.eps)
        out = xhat * self.weight.float().view(1, -1, 1) + self.bias.float().view(1, -1, 1)
        return out.to(in_dtype)


class ChannelAffine(nn.Module):
    r"""A per-channel affine transform $y_{c,t} = \gamma_c x_{c,t} + \beta_c$ (parameterless over time).

    Trivially per-timestep and per-sample, hence strictly causal; the lightest ``norm_kind``.

    Shapes:
        Input/Output: $(B, C, T)$.
    """

    def __init__(self, num_channels: int) -> None:
        """Initialise with identity affine parameters.

        Args:
            num_channels: Channel count $C$.
        """
        super().__init__()
        self.num_channels = int(num_channels)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-channel affine to ``(B, C, T)``."""
        return x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)


def make_frontend_norm(kind: str, num_channels: int, num_groups: int = 8) -> nn.Module:
    r"""Build a causal, per-sample front-end normaliser.

    Args:
        kind: ``causal_group_norm`` (v3's :class:`CausalGroupNorm`, the default), ``cln``
            (:class:`CumulativeLayerNorm`), or ``channel_affine`` (:class:`ChannelAffine`).
        num_channels: Channel count $C$.
        num_groups: Group count for ``causal_group_norm``; **must divide** ``num_channels``.

    Returns:
        A ready :class:`torch.nn.Module` operating on $(B, C, T)$.

    Raises:
        ValueError: If ``kind`` is unknown, or ``num_channels`` is not divisible by ``num_groups``
            for ``causal_group_norm`` (raised by :class:`CausalGroupNorm`; a misconfigured
            ``norm_num_groups`` must fail loudly, not silently degrade to a single group).
    """
    if kind == "causal_group_norm":
        # Delegate divisibility validation to CausalGroupNorm, which raises ValueError -- do NOT
        # silently walk num_groups down to a divisor (that would train an unconfigured norm).
        return CausalGroupNorm(num_groups, num_channels)
    if kind == "cln":
        return CumulativeLayerNorm(num_channels)
    if kind == "channel_affine":
        return ChannelAffine(num_channels)
    raise ValueError(
        f"unknown norm_kind {kind!r}; choose from "
        "{'causal_group_norm', 'cln', 'channel_affine'}"
    )


#: Normalisers that pool over the time axis (or couple the batch) and therefore leak the future
#: (or break the batch-independent source path). Forbidden anywhere in the front end.
_FORBIDDEN_NORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,  # the time-pooling GroupNorm over (C, T); CausalGroupNorm is a *different* class
)


def assert_no_time_pooling_norm(module: nn.Module) -> None:
    r"""Raise if ``module`` contains any time-pooling or batch-coupling normaliser.

    Forbidden: ``BatchNorm{1,2,3}d``, ``SyncBatchNorm``, ``InstanceNorm{1,2,3}d``, and
    :class:`torch.nn.GroupNorm` (which reduces over $(C, T)$ within a group -- the exact G0 leak).
    Permitted: :class:`CausalGroupNorm`, :class:`CumulativeLayerNorm`, :class:`ChannelAffine`, and
    a channel-axis :class:`torch.nn.LayerNorm` (the inherited v3 core uses those legitimately, so
    plain ``LayerNorm`` is **not** blanket-forbidden here).

    Args:
        module: Subtree to check.

    Raises:
        ValueError: On the first forbidden normaliser found, naming the offending submodule.
    """
    for name, sub in module.named_modules():
        if isinstance(sub, _FORBIDDEN_NORMS):
            raise ValueError(
                f"forbidden time-pooling/batch-coupling norm {type(sub).__name__} at "
                f"{name!r}: the front end must use causal per-sample norms "
                "(causal_group_norm / cln / channel_affine)."
            )


# =============================================================================
# Convolutional stage helpers
# =============================================================================
class MultiScaleCausalConv1d(nn.Module):
    r"""A parallel multi-kernel causal filterbank: concat branches, then a $1\times1$ mix.

    Used for the first stage, where a single stride sees both a fast deceleration notch and a slow
    contraction envelope. Each branch is a full causal conv at its own kernel width; the branches
    are concatenated and projected by a $1\times1$ conv (learned per-scale, per-channel mixing).

    Shapes:
        Input:  $(B, C_{\mathrm{in}}, L)$
        Output: $(B, C_{\mathrm{out}}, L)$
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: Sequence[int],
        branch_channels: Optional[int] = None,
    ) -> None:
        """Initialise the filterbank.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count (the pre-gate width when the stage is gated).
            kernels: Causal kernel sizes, one branch each.
            branch_channels: Per-branch channel count before the mix (defaults to ``out_channels``).
        """
        super().__init__()
        bc = int(branch_channels) if branch_channels is not None else int(out_channels)
        self.branches = nn.ModuleList(
            [CausalConv1d(in_channels, bc, int(k)) for k in kernels]
        )
        self.project = nn.Conv1d(len(kernels) * bc, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every branch, concatenate on the channel axis, and $1\\times1$-project."""
        cat = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(cat)


class DepthwiseSeparableCausalConv1d(nn.Module):
    r"""Depthwise causal conv (spatial mix) then a $1\times1$ pointwise conv (channel mix).

    Much cheaper than a full conv, which keeps the length-$5280$ front end trainable. Stride is
    **not** applied here (all downsampling lives in the anti-alias step).

    Shapes:
        Input:  $(B, C_{\mathrm{in}}, L)$
        Output: $(B, C_{\mathrm{out}}, L)$
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialise the depthwise-separable conv.

        Args:
            in_channels: Input channel count (also the depthwise group count).
            out_channels: Output channel count (the pre-gate width when the stage is gated).
            kernel_size: Depthwise causal kernel size.
        """
        super().__init__()
        self.dw = CausalConv1d(in_channels, in_channels, int(kernel_size), groups=in_channels)
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Depthwise then pointwise: ``(B, Cin, L) -> (B, Cout, L)``."""
        return self.pw(self.dw(x))


# =============================================================================
# One front-end stage (S1-T03)
# =============================================================================
class RawFrontendBlock(nn.Module):
    r"""One stride-$s$ front-end stage.

    $$
    X_{j+1} = \operatorname{Norm}_{\text{causal}}\big(\mathrm{main} + \mathrm{skip}\big),\quad
    \mathrm{main} = \mathrm{Refine}\big(\mathrm{AA}_{\text{main}}(\mathrm{Gate}(\mathrm{Conv}(X_j)))\big),\quad
    \mathrm{skip} = \mathrm{AA}_{\text{skip}}\big(\mathrm{Proj}(X_j)\big).
    $$

    ``Conv`` is a :class:`MultiScaleCausalConv1d` on the first stage (``first_stage_kernels`` set)
    and a :class:`DepthwiseSeparableCausalConv1d` otherwise; it emits the pre-gate width
    $G\,C_{\mathrm{out}}$ ($G=2$ gated, $1$ otherwise). Both the main and skip branches downsample
    with the **same** right-offset stride, so the residual stays sample-aligned. All ops are
    causal.

    Shapes:
        Input:  $(B, C_{\mathrm{in}}, L)$
        Output: $(B, C_{\mathrm{out}}, \lceil L/s \rceil)$
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 15,
        stride: int = 2,
        gated: bool = True,
        antialias: bool = True,
        antialias_kernel: str = "binomial5",
        norm_kind: str = "causal_group_norm",
        norm_num_groups: int = 8,
        first_stage_kernels: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        """Build one stage; the gated/plain branch is construction-gated (no unused params)."""
        super().__init__()
        self.gated = bool(gated)
        gate_mult = 2 if self.gated else 1
        pre_gate = gate_mult * int(out_channels)

        if first_stage_kernels is not None:
            self.conv: nn.Module = MultiScaleCausalConv1d(
                in_channels, pre_gate, first_stage_kernels, branch_channels=out_channels
            )
        else:
            self.conv = DepthwiseSeparableCausalConv1d(in_channels, pre_gate, kernel_size)

        self.aa_main = CausalAntiAliasDownsample1d(
            out_channels, stride=stride, antialias=antialias, antialias_kernel=antialias_kernel
        )
        self.refine = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.proj_skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.aa_skip = CausalAntiAliasDownsample1d(
            out_channels, stride=stride, antialias=antialias, antialias_kernel=antialias_kernel
        )
        self.norm = make_frontend_norm(norm_kind, out_channels, norm_num_groups)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _gate(self, y: torch.Tensor) -> torch.Tensor:
        r"""Gated activation $\tanh(a)\odot\sigma(b)$ (split channels) or GELU when not gated."""
        if self.gated:
            a, b = y.chunk(2, dim=1)
            return torch.tanh(a) * torch.sigmoid(b)
        return F.gelu(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stage: ``(B, Cin, L) -> (B, Cout, ceil(L/stride))``."""
        y = self.conv(x)                       # (B, pre_gate, L)
        y = self._gate(y)                      # (B, Cout, L)
        y = self.aa_main(y)                    # (B, Cout, L/s)
        main = self.refine(y)                  # (B, Cout, L/s)
        skip = self.aa_skip(self.proj_skip(x))  # (B, Cout, L/s)
        return self.drop(self.norm(main + skip))


# =============================================================================
# The full front end (S1-T04 featurize + S1-T05 assembly)
# =============================================================================
class CausalRawFrontend(nn.Module):
    r"""Strictly-causal multiscale front end mapping raw $(B, L)$ + mask to $(B, T, d_{\mathrm{raw}})$.

    Owns its own $3$-channel featurisation ($[\text{std value},\ \text{mask},\ \Delta\text{std value}]$)
    with **fixed dataset z-score stats** (per-segment stats would leak the future, G0), runs four
    stride-$2$ stages ($4\to2\to1\to0.5\to0.25$ Hz) with a multiscale first stage, then crops the
    central $T$ tokens (transform-then-trim, so the useful region is edge-clean) and transposes to
    the token layout.
    """

    def __init__(
        self,
        *,
        stream: str,
        mean: float,
        std: float,
        raw_len: int = 5280,
        decimation: int = 16,
        crop: int = 15,
        channels: Sequence[int] = (32, 64, 96, 128),
        d_raw: int = 128,
        stages: Sequence[int] = (2, 2, 2, 2),
        gated: bool = True,
        antialias: bool = True,
        antialias_kernel: str = "binomial5",
        norm_kind: str = "causal_group_norm",
        norm_num_groups: int = 8,
        first_kernels_fhr: Sequence[int] = (7, 31, 65),
        first_kernels_up: Sequence[int] = (15, 65, 129),
        generic_kernel: int = 15,
        dropout: float = 0.0,
        sentinel: Optional[float] = _DEFAULT_SENTINEL,
    ) -> None:
        r"""Assemble the front end for one stream.

        Args:
            stream: ``"y"`` (FHR / target) or ``"u"`` (UP / source); selects the first-stage kernels.
            mean: Fixed dataset mean for the z-score (a scalar; never per-segment).
            std: Fixed dataset std for the z-score (a scalar).
            raw_len: Raw samples per segment $L_{\mathrm{raw}}$.
            decimation: Total front-end stride $D$ (must equal the product of ``stages``).
            crop: Tokens trimmed each side after the stages.
            channels: Per-stage output channels for the progressive path.
            d_raw: Final token width (must equal the model width $d_{\mathrm{model}}$).
            stages: Per-stage strides (``[2,2,2,2]`` default, ``[16]`` single-stride ablation).
            gated: Gated activation ($\tanh\odot\sigma$) vs GELU.
            antialias: Low-pass before each decimation.
            antialias_kernel: FIR name for the anti-alias filter.
            norm_kind: Front-end causal normaliser (see :func:`make_frontend_norm`).
            norm_num_groups: Group count for ``causal_group_norm``.
            first_kernels_fhr: Multiscale first-stage kernels for the FHR stream.
            first_kernels_up: Multiscale first-stage kernels for the UP stream.
            generic_kernel: Depthwise kernel size for stages $\ge 2$.
            dropout: Stage dropout.
            sentinel: Gap sentinel value in the raw signal; defaults to
                :data:`raw_mask_constants.SENTINEL` (the single source of truth, $0.0$ for this
                dataset). ``None`` disables sentinel refinement (``weight_only`` masking).

        Raises:
            ValueError: If ``stream`` is invalid, the stages do not multiply to ``decimation``,
                the crop is degenerate, or a forbidden norm is configured.
        """
        super().__init__()
        if stream not in ("y", "u"):
            raise ValueError(f"stream must be 'y' or 'u', got {stream!r}")
        self.stream = stream
        self.raw_len = int(raw_len)
        self.decimation = int(decimation)
        self.crop = int(crop)
        self.d_raw = int(d_raw)
        self.sentinel = sentinel

        prod = 1
        for s in stages:
            prod *= int(s)
        if prod != self.decimation:
            raise ValueError(
                f"product of stages {tuple(stages)} = {prod} must equal decimation {self.decimation}"
            )
        self.t_tilde = self.raw_len // self.decimation
        self.t = self.t_tilde - 2 * self.crop
        if self.t <= 0:
            raise ValueError(f"non-positive T={self.t} (t_tilde={self.t_tilde}, crop={self.crop})")

        # Fixed z-score stats as buffers (never per-segment).
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

        first_kernels = first_kernels_fhr if stream == "y" else first_kernels_up

        # Per-stage output channels. Progressive path follows `channels` (last forced to d_raw);
        # a single-stride ablation collapses to one stage at d_raw.
        if len(stages) == 1:
            stage_out = [self.d_raw]
        else:
            if len(channels) != len(stages):
                raise ValueError(
                    f"channels {tuple(channels)} must match stages {tuple(stages)} in length"
                )
            stage_out = [int(c) for c in channels]
            stage_out[-1] = self.d_raw

        blocks = []
        in_ch = 3  # featurised input channels
        for j, s in enumerate(stages):
            blocks.append(
                RawFrontendBlock(
                    in_ch,
                    stage_out[j],
                    kernel_size=generic_kernel,
                    stride=int(s),
                    gated=gated,
                    antialias=antialias,
                    antialias_kernel=antialias_kernel,
                    norm_kind=norm_kind,
                    norm_num_groups=norm_num_groups,
                    first_stage_kernels=first_kernels if j == 0 else None,
                    dropout=dropout,
                )
            )
            in_ch = stage_out[j]
        self.blocks = nn.ModuleList(blocks)

        # G0 guard: fail loudly on any time-pooling / batch-coupling normaliser.
        assert_no_time_pooling_norm(self)

    # -- Featurisation (S1-T04) --------------------------------------------
    def featurize(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""Build the $3$-channel input $[\text{std value},\ \text{mask},\ \Delta\text{std value}]$, NaN-safe.

        The invalid positions are neutralised in **standardised** space (set to $0$), not in raw
        space: zeroing the raw value *before* the z-score would map a gap to $-\mu/\sigma$
        (${\approx}-7$ for FHR $\mu{=}140,\sigma{=}20$) -- an extreme-bradycardia-looking constant
        that the anti-alias low-pass then smears across the following ${\sim}16$-$32$ s of valid
        tokens -- whereas a neutral $0$ lets the mask channel (not a spurious value) carry the
        invalidity. An **effective validity** mask refines the caller's mask with finiteness and the
        gap sentinel, so a sentinel-valued sample the caller's mask missed is still invalidated
        (and appears as $0$ in the mask channel). The first difference is taken on the neutralised,
        standardised value and zeroed wherever **either** endpoint is invalid, so a gap -- and, in
        particular, the first valid sample *after* a gap -- injects no spurious slope; it is
        replicate-padded ($x[-1] := x[0]$) so $\Delta x[0] = 0$.

        Args:
            raw: Raw signal $(B, L)$.
            mask: Validity mask $(B, L)$ (1 = valid).

        Returns:
            Featurised tensor $(B, 3, L)$, all-finite, with invalid positions neutral in the value
            and derivative channels and $0$ in the (refined) mask channel.
        """
        # (1) Effective validity: caller mask AND finite AND (optionally) not the gap sentinel.
        valid = mask.to(torch.bool) & torch.isfinite(raw)
        if self.sentinel is not None:
            valid = valid & (raw != self.sentinel)
        m = valid.to(raw.dtype)
        # (2) Finite value for the z-score (non-finite -> 0; the position is neutralised below).
        val = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        # (3) Fixed-stats z-score, then neutralise invalid positions to 0 in standardised space.
        x_std = ((val - self.mean) / self.std) * m
        # (4) First difference of the neutralised standardised value (replicate-padded so
        #     dx[0] == 0), zeroed wherever EITHER endpoint is invalid. Gating on `m` alone left the
        #     first valid sample after a gap with dx = x_std - 0 = its full value -- a fake step the
        #     causal anti-alias low-pass would then smear across the next ~16-32 s of valid tokens;
        #     the extra `m_prev` factor removes it.
        x_prev = torch.cat([x_std[:, :1], x_std[:, :-1]], dim=1)
        m_prev = torch.cat([m[:, :1], m[:, :-1]], dim=1)
        dx = (x_std - x_prev) * m * m_prev
        return torch.stack([x_std, m, dx], dim=1)

    # -- Full forward (S1-T05) ---------------------------------------------
    def forward(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""Map raw $(B, L)$ + mask to low-rate tokens $(B, T, d_{\mathrm{raw}})$.

        Args:
            raw: Raw signal $(B, L_{\mathrm{raw}})$.
            mask: Validity mask $(B, L_{\mathrm{raw}})$.

        Returns:
            Token tensor $(B, T, d_{\mathrm{raw}})$ (cropped, transposed).
        """
        x = self.featurize(raw, mask)          # (B, 3, L)
        for block in self.blocks:
            x = block(x)                       # (B, C_j, L / prod strides)
        # x is (B, d_raw, t_tilde); crop the central T tokens, then transpose.
        x = x[:, :, self.crop : self.crop + self.t]  # (B, d_raw, T)
        return x.transpose(1, 2).contiguous()        # (B, T, d_raw)
