r"""Causal Transformer primitives: normalisation, positions, gating, convolution, attention.

Nothing under ``teb_vae`` provides per-token RMSNorm, rotary position encoding, LayerScale,
SwiGLU, depthwise causal convolution or windowed causal self-attention, so they are written here
against the architecture's own equations rather than imported from outside the package tree.

Everything in this module obeys one invariant, and it is worth stating rather than assuming:
**the output at step $t$ depends only on inputs at steps $\le t$.** Two mechanisms deliver it,
and they are different in kind.

*Position-wise modules* -- :class:`RMSNorm`, :class:`LayerScale`, :class:`SwiGLUFeedForward` --
touch the channel axis only. They are causal for free, with no mask and no padding, because there
is no path between steps at all. This is the reason the architecture normalises per token: a
statistic pooled over time makes every "history" state carry a low-bandwidth image of its own
future, which is invisible in a loss curve and corrupts exactly the quantity the model exists to
measure.

*Time-mixing modules* -- :class:`CausalDepthwiseConv1d`, :class:`CausalSelfAttention` -- are causal
by explicit construction: left-only padding for the convolution, and a mask that admits only
$j \le t$ (optionally only $t - j < W$) for the attention.

The following must therefore never appear inside a history encoder, and none of them appears here:
``BatchNorm`` over $(B, C, T)$; ``GroupNorm`` whose statistics include the time axis; any
sequence-wide mean or variance; centred or symmetric convolutional padding; bidirectional
recurrence; non-causal self-attention; and pooling that reads future tokens.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

#: Variance floor inside the RMSNorm denominator. The architecture's own value; small enough not
#: to bias an $O(1)$ activation, large enough that an exactly-zero token has a bounded derivative.
RMS_NORM_EPS = 1e-5

#: LayerScale initialisation. A residual branch starts at a hundredth of its eventual weight, so a
#: pre-norm stack begins close to the identity and is stable at depth without a warm-up trick.
LAYER_SCALE_INIT = 1e-2

#: Rotary frequency base $\theta$. The standard value; no ablation varies it, so it is a
#: constructor default rather than a configuration key.
ROPE_BASE = 10000.0


class RMSNorm(nn.Module):
    r"""Root-mean-square normalisation over the channel axis only.

    $$
    \operatorname{RMSNorm}(x_t) =
    \frac{x_t}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_{t,i}^2 + \epsilon}} \odot g .
    $$

    The reduction runs over channels at each position independently, so no statistic mixes $t$
    with any $t' \neq t$ and the module is causal without a mask. That is the property, not a
    happy accident: a normaliser that pooled over time would leak the future into every history
    state.

    Unlike ``LayerNorm`` this does **not** centre. A constant offset shared by every channel of a
    token survives into the output, scaled -- which is intended: the input adapter upstream already
    centres, and re-centring at every sublayer discards a degree of freedom the residual stream
    uses.

    Shapes:
        Input:  ``(..., d)``
        Output: ``(..., d)``
    """

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS) -> None:
        """Initialize the scale vector at $1$.

        Args:
            dim: Channel count $d$; the size of the last axis.
            eps: Variance floor added inside the square root.
        """
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last axis and apply the learned scale.

        Args:
            x: Input whose last axis is ``dim``.

        Returns:
            A tensor shaped like ``x``.
        """
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps) * self.weight

    def extra_repr(self) -> str:
        """Report the width and the epsilon, for readable module trees."""
        return f"{self.dim}, eps={self.eps}"


class LayerScale(nn.Module):
    r"""Per-channel learned gain on a residual branch, initialised to $10^{-2}$.

    $$x \mapsto \gamma \odot x, \qquad \gamma \in \mathbb R^{d},\ \gamma_i(0) = 10^{-2}.$$

    Applied to the branch, never to the residual stream itself, so the identity path through every
    sublayer stays unmodified. Starting at a hundredth means a freshly constructed stack is close
    to the identity, which is what makes a deep pre-norm residual network trainable without
    per-layer scaling tricks -- and what makes "near-identity at initialisation" a testable
    property rather than a hope.

    Shapes:
        Input:  ``(..., d)``
        Output: ``(..., d)``
    """

    def __init__(self, dim: int, init_value: float = LAYER_SCALE_INIT) -> None:
        """Initialize every channel's gain to ``init_value``.

        Args:
            dim: Channel count $d$.
            init_value: The initial gain $\\gamma_i(0)$.
        """
        super().__init__()
        self.dim = int(dim)
        self.init_value = float(init_value)
        self.weight = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the last axis channel-wise.

        Args:
            x: Input whose last axis is ``dim``.

        Returns:
            A tensor shaped like ``x``.
        """
        return x * self.weight

    def extra_repr(self) -> str:
        """Report the width and the initialisation, for readable module trees."""
        return f"{self.dim}, init={self.init_value}"


class SwiGLUFeedForward(nn.Module):
    r"""Bias-free gated feed-forward network.

    $$F(x) = W^o\left[\operatorname{SiLU}(W^g x) \odot W^v x\right],$$

    with $W^g, W^v: \mathbb R^{d} \to \mathbb R^{d_{\mathrm{ff}}}$ and
    $W^o: \mathbb R^{d_{\mathrm{ff}}} \to \mathbb R^{d}$. Exactly $3 d\, d_{\mathrm{ff}}$
    parameters: every projection is bias-free, so the count is a clean function of the two widths.

    The nonlinearity sits on the *gate* branch and the value branch is linear. Position-wise, like
    every other channel-mixing module here, so it carries no time dependence of its own.

    Shapes:
        Input:  ``(..., d)``
        Output: ``(..., d)``
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        """Build the three projections and the output dropout.

        Args:
            d_model: Input and output width $d$.
            d_ff: Hidden width $d_{\\mathrm{ff}}$.
            dropout: Dropout probability applied to the output, before the caller's residual add.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate, mix and project back.

        Args:
            x: Input whose last axis is ``d_model``.

        Returns:
            A tensor shaped like ``x``.
        """
        gated = F.silu(self.gate_proj(x)) * self.value_proj(x)
        return self.dropout(self.out_proj(gated))


class RotaryPositionEncoding(nn.Module):
    r"""Rotary position encoding applied to queries and keys.

    Each adjacent coordinate pair of a head vector is rotated by an angle proportional to its
    absolute position:

    $$
    R(t) = \begin{bmatrix} \cos(t\omega_i) & -\sin(t\omega_i) \\
                           \sin(t\omega_i) & \phantom{-}\cos(t\omega_i) \end{bmatrix},
    \qquad \omega_i = \theta^{-2i/d_h}.
    $$

    Applying it to both sides of an attention score makes the score depend on the *displacement*:
    $\langle R(t)q, R(j)k\rangle$ is a function of $t - j$ alone. That is the whole point. It adds
    no learned absolute anchor, so the encoder cannot come to associate a particular position
    inside an arbitrarily trimmed segment with a particular signal behaviour.

    Positions are **absolute and start at zero**, which is what makes prefix equivalence hold:
    encoding $X_{0:t}$ and reading its last step must give what encoding $X_{0:T-1}$ and reading
    step $t$ gives, and any end-relative indexing breaks it.

    The transform is deterministic and carries no parameters. The late lag cross-attention has its
    own learned lag-specific biases; those answer a different question -- which source-target delay
    is useful -- and the two mechanisms are kept strictly independent.

    The tables are built once at ``max_seq_len`` and held as non-persistent buffers: they follow
    the module across devices and dtypes, never enter a ``state_dict``, and never grow. A longer
    input raises rather than silently reallocating.

    Shapes:
        Input:  ``(..., T, d_head)``
        Output: ``(..., T, d_head)``
    """

    # Declared so the buffers type as tensors rather than as the ``Module`` union
    # ``__getattr__`` advertises.
    cos_table: torch.Tensor
    sin_table: torch.Tensor

    def __init__(self, d_head: int, max_seq_len: int, base: float = ROPE_BASE) -> None:
        """Precompute the cosine and sine tables.

        Args:
            d_head: Head width $d_h$. Must be even -- the rotation acts on coordinate pairs.
            max_seq_len: Longest sequence the tables cover.
            base: Frequency base $\\theta$.

        Raises:
            ValueError: If ``d_head`` is odd or ``max_seq_len`` is not positive.
        """
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(
                f"d_head must be even for rotary position encoding, which rotates coordinate "
                f"pairs; got d_head={d_head}"
            )
        if max_seq_len < 1:
            raise ValueError(f"max_seq_len must be at least 1, got {max_seq_len}")

        self.d_head = int(d_head)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)

        exponents = torch.arange(0, d_head, 2, dtype=torch.float32) / float(d_head)
        frequencies = torch.pow(torch.tensor(float(base)), -exponents)
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), frequencies)
        self.register_buffer("cos_table", angles.cos(), persistent=False)
        self.register_buffer("sin_table", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate every coordinate pair by its absolute position.

        Args:
            x: Tensor whose last two axes are ``(T, d_head)``; any leading axes are broadcast over.

        Returns:
            A tensor shaped like ``x``.

        Raises:
            ValueError: If the last axis is not ``d_head`` or the sequence exceeds ``max_seq_len``.
        """
        seq_len = int(x.shape[-2])
        if x.shape[-1] != self.d_head:
            raise ValueError(
                f"expected a head width of {self.d_head}, got {int(x.shape[-1])}"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence of {seq_len} steps exceeds the rotary tables built for "
                f"max_seq_len={self.max_seq_len}; the tables are fixed at construction so a "
                f"longer input cannot be served"
            )

        cos = self.cos_table[:seq_len].to(dtype=x.dtype)
        sin = self.sin_table[:seq_len].to(dtype=x.dtype)
        pairs = x.reshape(*x.shape[:-1], self.d_head // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.reshape(x.shape)

    def extra_repr(self) -> str:
        """Report the head width, the table length and the base."""
        return f"d_head={self.d_head}, max_seq_len={self.max_seq_len}, base={self.base}"


class CausalDepthwiseConv1d(nn.Module):
    r"""Depthwise convolution with left-only padding.

    One filter per channel, no cross-channel mixing, no bias, and $(k-1)r$ steps of zero padding
    applied to the **left** before the convolution rather than through the ``padding`` argument --
    which would pad symmetrically and read the future. The output length equals the input length
    and position $t$ sees exactly $[t - (k-1)r,\ t]$.

    Shapes:
        Input:  ``(B, C, T)``
        Output: ``(B, C, T)``
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1) -> None:
        """Build the grouped convolution and record its left padding.

        Args:
            channels: Channel count $C$; also the group count, which is what makes it depthwise.
            kernel_size: Kernel width $k$.
            dilation: Dilation $r$.

        Raises:
            ValueError: If any argument is not positive.
        """
        super().__init__()
        if channels < 1 or kernel_size < 1 or dilation < 1:
            raise ValueError(
                f"channels, kernel_size and dilation must all be positive; got "
                f"channels={channels}, kernel_size={kernel_size}, dilation={dilation}"
            )
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad on the left and convolve.

        Args:
            x: Input of shape ``(B, C, T)``.

        Returns:
            Output of shape ``(B, C, T)``.
        """
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)

    def extra_repr(self) -> str:
        """Report the shape, the dilation and the left padding actually applied."""
        return (
            f"{self.channels}, kernel_size={self.kernel_size}, dilation={self.dilation}, "
            f"left_padding={self.left_padding}"
        )


def init_depthwise_(module: nn.Module) -> int:
    r"""Re-initialise every depthwise convolution weight in place, variance-preservingly.

    A depthwise filter sums $k$ terms, so preserving the activation variance needs
    $\sigma = 1/\sqrt k$.

    This exists because the generic initialiser gets it badly wrong on this shape, and silently.
    Xavier reads the fan pair off a $(C, 1, k)$ weight as $\mathrm{fan\_in} = k$ against
    $\mathrm{fan\_out} = Ck$, giving $\sigma = \sqrt{2 / (k + Ck)}$. At $C = 128$ that is a factor
    $\sqrt{(1 + C)/2} = 8.03$ too small, **independent of $k$** -- so the stem starts an order of
    magnitude quieter than intended and the error cannot be spotted by varying the kernel.

    Call it *after* any generic pass, never before: a later generic pass would undo it.

    Args:
        module: Subtree to re-initialise in place.

    Returns:
        The number of depthwise convolutions re-initialised, so a caller can assert the pass was
        not a no-op.
    """
    count = 0
    for child in module.modules():
        if isinstance(child, CausalDepthwiseConv1d):
            std = 1.0 / math.sqrt(float(child.kernel_size))
            nn.init.normal_(child.conv.weight, mean=0.0, std=std)
            count += 1
    return count


class GatedCausalConvBlock(nn.Module):
    r"""Pre-normalised gated depthwise convolution block with a LayerScale residual.

    $$
    N = \operatorname{RMSNorm}(C^{\mathrm{in}}), \qquad
    [G^{(1)}, G^{(2)}] = W_{\mathrm{in}} N, \qquad
    G = G^{(1)} \odot \sigma\!\left(G^{(2)}\right),
    $$
    $$
    D = \operatorname{DWConv}^{\mathrm{causal}}_{k, r}(G), \qquad
    R = W_{\mathrm{out}} \operatorname{SiLU}\!\left(\operatorname{RMSNorm}(D)\right), \qquad
    C^{\mathrm{out}} = C^{\mathrm{in}} + \gamma \odot \operatorname{Dropout}(R).
    $$

    $W_{\mathrm{in}}: \mathbb R^{d} \to \mathbb R^{2d}$ and
    $W_{\mathrm{out}}: \mathbb R^{d} \to \mathbb R^{d}$ are bias-free and position-wise; the only
    time mixing is the depthwise convolution, and it is left-padded. So the block is causal, and
    its reach is exactly $(k-1)r$ steps.

    The parameter count is exactly $3d^2 + 3d + dk$: $2d^2$ for the gated input projection, $d^2$
    for the output projection, $d$ for each of the two norms and the LayerScale, and $dk$ for the
    depthwise filter bank.

    The public interface is $(B, T, d)$ throughout. The transpose to the convolution's
    $(B, d, T)$ is internal and does not leak.

    Shapes:
        Input:  ``(B, T, d)``
        Output: ``(B, T, d)``
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        layer_scale_init: float = LAYER_SCALE_INIT,
    ) -> None:
        """Build the gate, the convolution, the projection and the residual scale.

        Args:
            d_model: Channel width $d$, unchanged from input to output.
            kernel_size: Depthwise kernel width $k$.
            dilation: Depthwise dilation $r$.
            dropout: Dropout probability on the branch output, before the residual add.
            layer_scale_init: Initial LayerScale gain.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.norm_in = RMSNorm(d_model)
        self.proj_in = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv = CausalDepthwiseConv1d(d_model, kernel_size, dilation)
        self.norm_conv = RMSNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = LayerScale(d_model, layer_scale_init)

    @property
    def receptive_field(self) -> int:
        """Steps of history the block's output at $t$ can reach, itself included."""
        return self.conv.left_padding + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated convolution branch and add it to the residual stream.

        Args:
            x: Input of shape ``(B, T, d)``.

        Returns:
            Output of shape ``(B, T, d)``.
        """
        value, gate = self.proj_in(self.norm_in(x)).chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)
        convolved = self.conv(gated.transpose(1, 2)).transpose(1, 2)
        branch = self.proj_out(F.silu(self.norm_conv(convolved)))
        return x + self.layer_scale(self.dropout(branch))


def build_causal_window_mask(seq_len: int, window: Optional[int]) -> torch.Tensor:
    r"""Build the boolean attention mask admitting $j \le t$, optionally within a window.

    With ``window`` set, the admitted band is $0 \le t - j < W$; without it, the full causal
    prefix. ``True`` means *participate*, which is what
    ``torch.nn.functional.scaled_dot_product_attention`` expects of a boolean mask.

    Every row admits $j = t$, in both cases. That is worth noting because it settles a question
    the architecture would otherwise have to answer: an all-masked attention row cannot occur here,
    so there is no NaN path to defend against. The encoder attention does no data-driven validity
    masking -- see the design record for why that is deliberate -- so the diagonal is the only
    guarantee needed.

    Args:
        seq_len: Sequence length $T$.
        window: Window width $W$ in steps, or ``None`` for the full causal prefix.

    Returns:
        A ``(T, T)`` boolean tensor.

    Raises:
        ValueError: If ``seq_len`` is not positive, or ``window`` is given and not positive.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")
    if window is not None and window < 1:
        raise ValueError(
            f"attention window must be at least 1 step so that a row can attend to itself, "
            f"got {window}"
        )

    positions = torch.arange(seq_len)
    displacement = positions[:, None] - positions[None, :]
    allowed = displacement >= 0
    if window is not None:
        allowed = allowed & (displacement < int(window))
    return allowed


class CausalSelfAttention(nn.Module):
    r"""Pre-normalised multi-head causal self-attention with rotary positions.

    Normalises per token, projects $Q$, $K$, $V$ bias-free, rotates $Q$ and $K$ by their absolute
    positions, and attends through ``scaled_dot_product_attention``:

    $$
    e_{h,t,j} = \frac{\widetilde q_{h,t}^{\mathsf T}\widetilde k_{h,j}}{\sqrt{d_h}} +
    \mathcal M_{t,j}, \qquad
    O_{h,t} = \sum_j \operatorname{softmax}_j(e_{h,t,j})\, v_{h,j},
    $$

    then concatenates the heads and applies $W^O$.

    Causality arrives through exactly one of two mechanisms, never both. Without a window the
    kernel's own ``is_causal`` flag supplies the lower triangle; with one, an explicit boolean band
    mask does. Passing both would double-specify the constraint and, worse, make the two
    disagree silently if one were later changed.

    Attention-probability dropout is structurally zero, not configurable. It is unnecessary in an
    encoder this size and it makes reproducibility harder; the ``dropout`` argument is the
    *output* dropout of the equations.

    These heads are unrelated to the late lag cross-attention's heads and to the latent groups.
    They happen to number the same at the shipped configuration; nothing may couple them.

    Shapes:
        Input:  ``(B, T, d)``
        Output: ``(B, T, d)``
    """

    # Declared so the buffer types as an optional tensor rather than as the ``Module`` union
    # ``__getattr__`` advertises. ``None`` is the full-causal case, which carries no mask at all.
    attn_mask: Optional[torch.Tensor]

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        window: Optional[int] = None,
        is_causal: Optional[bool] = None,
        dropout: float = 0.0,
        rope_base: float = ROPE_BASE,
    ) -> None:
        """Build the projections, the rotary tables and whichever mask mechanism applies.

        Args:
            d_model: Model width $d$. Must be divisible by ``num_heads``.
            num_heads: Encoder attention heads $H_e$. The head width $d_h = d / H_e$ must be even,
                which rotary position encoding requires.
            max_seq_len: Longest sequence served; sizes the rotary tables and the window mask.
            window: Causal window $W$ in steps, or ``None`` for the full causal prefix.
            is_causal: Which masking mechanism to use, or ``None`` to derive it from ``window``.
                Only the derived combinations are legal; the argument exists so a caller that
                states it and gets it wrong is told, rather than silently double-masked or
                silently non-causal.
            dropout: Dropout probability on the attention output.
            rope_base: Rotary frequency base.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``num_heads``, or if ``is_causal``
                contradicts ``window``.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}) so every head "
                f"has the same width"
            )
        derived_causal = window is None
        if is_causal is not None and bool(is_causal) != derived_causal:
            raise ValueError(
                f"is_causal={is_causal} contradicts window={window}: causality comes from the "
                f"kernel's is_causal flag when there is no window and from the explicit band mask "
                f"when there is one, never from both and never from neither"
            )

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = int(d_model) // int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.window = None if window is None else int(window)
        self.is_causal = derived_causal

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Raises on an odd head width, naming it.
        self.rope = RotaryPositionEncoding(self.d_head, max_seq_len, base=rope_base)
        self.dropout = nn.Dropout(dropout)

        mask = None if self.window is None else build_causal_window_mask(max_seq_len, self.window)
        self.register_buffer("attn_mask", mask, persistent=False)

    @property
    def receptive_field(self) -> Optional[int]:
        """Steps of history one block reaches, or ``None`` when the context is unbounded."""
        return self.window

    def _project(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project the normalised input to per-head $Q$, $K$, $V$ with rotary positions applied.

        Args:
            h: Normalised input of shape ``(B, T, d)``.

        Returns:
            Three ``(B, H, T, d_head)`` tensors; $Q$ and $K$ are rotated, $V$ is not.
        """
        batch, seq_len, _ = h.shape
        shape = (batch, seq_len, self.num_heads, self.d_head)
        query = self.q_proj(h).view(shape).transpose(1, 2)
        key = self.k_proj(h).view(shape).transpose(1, 2)
        value = self.v_proj(h).view(shape).transpose(1, 2)
        return self.rope(query), self.rope(key), value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise, attend causally, and project the concatenated heads back to ``d_model``.

        Args:
            x: Input of shape ``(B, T, d)``.

        Returns:
            Output of shape ``(B, T, d)``.
        """
        batch, seq_len, _ = x.shape
        query, key, value = self._project(self.norm(x))
        mask = None if self.attn_mask is None else self.attn_mask[:seq_len, :seq_len]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
        )
        merged = attended.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.dropout(self.out_proj(merged))

    def extra_repr(self) -> str:
        """Report the head layout and the context the block reads."""
        context = "full causal prefix" if self.window is None else f"causal window {self.window}"
        return f"d_model={self.d_model}, num_heads={self.num_heads}, context={context}"


class CausalTransformerBlock(nn.Module):
    r"""Pre-normalised causal Transformer block: attention, then SwiGLU, each LayerScaled.

    $$
    S^{\mathrm{attn}} = S + \gamma^{\mathrm{attn}} \odot
    \operatorname{MHSA}\!\left(\operatorname{RMSNorm}(S)\right),
    $$
    $$
    S^{\mathrm{out}} = S^{\mathrm{attn}} + \gamma^{\mathrm{ffn}} \odot
    F\!\left(\operatorname{RMSNorm}\!\left(S^{\mathrm{attn}}\right)\right),
    $$

    where each sublayer owns its own pre-norm and its own output dropout. The residual path is
    an unmodified identity through both, so gradients reach the first block without passing
    through a normaliser at every hop, and with LayerScale at $10^{-2}$ the whole stack starts
    close to the identity.

    The parameter count is exactly $4d^2 + 3d\,d_{\mathrm{ff}} + 4d$: four bias-free attention
    projections, the SwiGLU triple, and $d$ apiece for two norms and two LayerScale vectors.

    Shapes:
        Input:  ``(B, T, d)``
        Output: ``(B, T, d)``
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        window: Optional[int] = None,
        dropout: float = 0.0,
        layer_scale_init: float = LAYER_SCALE_INIT,
        rope_base: float = ROPE_BASE,
    ) -> None:
        """Build the two sublayers and their residual scales.

        Args:
            d_model: Model width $d$.
            num_heads: Encoder attention heads $H_e$.
            d_ff: Feed-forward hidden width $d_{\\mathrm{ff}}$.
            max_seq_len: Longest sequence served.
            window: Causal window $W$ in steps, or ``None`` for the full causal prefix.
            dropout: Dropout probability on each sublayer's output.
            layer_scale_init: Initial LayerScale gain on both branches.
            rope_base: Rotary frequency base.
        """
        super().__init__()
        self.attn = CausalSelfAttention(
            d_model,
            num_heads,
            max_seq_len,
            window=window,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.attn_scale = LayerScale(d_model, layer_scale_init)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout=dropout)
        self.ffn_scale = LayerScale(d_model, layer_scale_init)

    @property
    def window(self) -> Optional[int]:
        """The block's attention window, or ``None`` when the context is the full prefix."""
        return self.attn.window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both residual sublayers.

        Args:
            x: Input of shape ``(B, T, d)``.

        Returns:
            Output of shape ``(B, T, d)``.
        """
        x = x + self.attn_scale(self.attn(x))
        return x + self.ffn_scale(self.ffn(self.ffn_norm(x)))
