"""Shared building blocks: causal convolution, normalisation, MLPs, and bound helpers.

Everything here is small, generic and used by more than one of the modules beside it. Nothing
here knows what a batch looks like or what the model is for.

These blocks are vendored rather than imported from the tree this package replaces. That is
deliberate and is what keeps this layer importable on its own: the module they came from pulls in
``numpy``, ``loguru``, the standard ``logging`` package and ``torch._dynamo`` at import time,
none of which a network component needs, and all of which would make a net impossible to
construct without a configured logger.
"""
from __future__ import annotations

import copy
from typing import Callable, Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn


def geometric_schedule(
    input_size: int,
    output_size: int,
    n_hidden: int,
    *,
    round_fn: Callable[[float], float] = round,
) -> Tuple[int, ...]:
    """Interpolate ``n_hidden`` layer widths geometrically between two sizes.

    Widths follow $s_i = s_0 r^i$ with $r = (s_{out} / s_{in})^{1/(n+1)}$, so each layer
    changes width by the same *ratio* rather than the same amount. For a funnel from $109$ to
    $8$ that keeps the per-layer compression even instead of dropping most of it in one step.

    Args:
        input_size: Width $s_{in}$ before the first hidden layer. Not included in the result.
        output_size: Width $s_{out}$ of the final layer.
        n_hidden: Number of hidden layers between the two.
        round_fn: How to round each interpolated width to an integer.

    Returns:
        The ``n_hidden + 1`` widths, ending at ``output_size``. The input width is excluded, so
        the result is directly usable as ``hidden_dims``.
    """
    steps = n_hidden + 1
    ratio = (output_size / input_size) ** (1 / steps)

    sizes = [input_size]
    current_ratio = ratio
    for _ in range(n_hidden):
        sizes.append(int(round_fn(input_size * current_ratio)))
        current_ratio *= ratio
    sizes.append(output_size)

    return tuple(sizes[1:])


def initialization(model: nn.Module) -> None:
    """Apply per-layer-type weight initialisation in place.

    Xavier-uniform for linear and convolutional weights, orthogonal for LSTM recurrent and input
    weights, zeros for biases, ones for ``LayerNorm`` scales.

    The one non-obvious step is the LSTM forget-gate bias. PyTorch packs the four gate biases
    into one ``bias_hh`` vector ordered $(i, f, g, o)$; this sets the $f$ slice to $1$, which
    starts the forget gate open so gradients survive the early steps of a long sequence instead
    of decaying through a near-zero gate.

    Args:
        model: Module tree to initialise in place.
    """
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for param_name, param in module.named_parameters():
                if "weight_ih" in param_name or "weight_hh" in param_name:
                    nn.init.orthogonal_(param)
                elif "bias" in param_name:
                    nn.init.zeros_(param)
                    if "bias_hh" in param_name:
                        hidden_size = module.hidden_size
                        param.data[hidden_size : 2 * hidden_size].fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class CausalMultiChannelConvBlock(nn.Module):
    """Pre-norm residual convolution block with causal padding.

    Normalisation and activation come *before* the convolution rather than after, which leaves
    the residual path an unmodified identity all the way through the stack -- gradients reach
    early layers without passing through a normaliser at every hop.

    Shapes:
        Input:  ``(B, C_in, L)``
        Output: ``(B, C_out, L')``; ``L' == L`` unless upsampling or a stride is requested.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        groups: int = 1,
        filter_size: int = 3,
        activation: type[nn.Module] = nn.ReLU,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            groups: Convolution groups.
            filter_size: Convolution kernel width.
            activation: Activation class, instantiated once.
            dilation: Convolution dilation.
            stride: Convolution stride.
            bias: Whether the convolution carries a bias.
            dropout: Dropout probability; ``0.0`` disables the layer entirely.
        """
        super().__init__()

        self.left_padding = (filter_size - 1) * dilation
        # min(8, C): GroupNorm needs num_groups to divide num_channels, and a narrow block can
        # have fewer than 8 channels.
        self.pre_norm = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=filter_size,
            groups=groups,
            bias=bias,
            padding=0,
            dilation=dilation,
            stride=stride,
        )
        self.act_fn = activation()

        # The residual can only be added if its channel count matches the output's.
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise, activate, convolve causally, and add the residual.

        Args:
            x: Input of shape ``(B, C_in, L)``.

        Returns:
            Output of shape ``(B, C_out, L')``.
        """
        residual = x
        x = self.pre_norm(x)
        x = self.act_fn(x)
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        output = self.conv(x)

        if self.dropout is not None:
            output = self.dropout(output)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return output + residual


class ResidualMLP(nn.Module):
    """Per-timestep residual MLP.

    Applied to the channel dimension independently at each step, so it mixes features without
    ever mixing time -- which is what lets it sit inside a causal stack without a mask.

    Shapes:
        Input:  ``(B, L, C_in)``
        Output: ``(B, L, hidden_dims[-1])``
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (72, 68, 64),
        final_activation: bool = True,
        activation: type[nn.Module] = nn.GELU,
        use_skip_connection: bool = True,
        use_input_layer_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the MLP stack and its skip path.

        Args:
            input_dim: Input channel count.
            hidden_dims: Width of each layer; the last is the output width.
            final_activation: Whether the last layer is normalised and activated.
            activation: Activation class, instance, or factory.
            use_skip_connection: Whether to add a projected skip from input to output.
            use_input_layer_norm: Whether to normalise the input before anything else.
            dropout: Dropout probability applied after intermediate activations only.
        """
        super().__init__()
        self.final_activation = final_activation
        self.use_skip_connection = use_skip_connection
        self.dropout = dropout
        self.input_norm = nn.LayerNorm(input_dim) if use_input_layer_norm else nn.Identity()
        self.activation_factory = self._build_activation_factory(activation)

        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for index in range(len(hidden_dims)):
            is_final_layer = index == len(hidden_dims) - 1
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if not is_final_layer or final_activation:
                layers.append(nn.LayerNorm(dims[index + 1]))
            if not is_final_layer:
                layers.append(self.activation_factory())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)

        final_dim = hidden_dims[-1]
        if self.use_skip_connection:
            # Identity rather than None when the widths already match: it keeps the skip path a
            # single uniform call site.
            self.skip_proj = (
                nn.Linear(input_dim, final_dim) if input_dim != final_dim else nn.Identity()
            )
        else:
            self.skip_proj = None

        self.final_act = self.activation_factory() if final_activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack and add the skip.

        The skip is taken from the *normalised* input, not the raw one, so both branches of the
        sum are on the same scale.

        Args:
            x: Input of shape ``(B, L, C_in)``.

        Returns:
            Output of shape ``(B, L, C_out)``.
        """
        x_norm = self.input_norm(x)
        y = self.body(x_norm)
        if self.use_skip_connection and self.skip_proj is not None:
            y = y + self.skip_proj(x_norm)
        if self.final_activation and self.final_act is not None:
            y = self.final_act(y)
        return y

    @staticmethod
    def _build_activation_factory(activation) -> Callable[[], nn.Module]:
        """Normalise the three accepted ``activation`` spellings into one factory.

        An instance is deep-copied per call site rather than shared, because a shared stateful
        activation would tie unrelated layers together.

        Args:
            activation: An ``nn.Module`` instance, an ``nn.Module`` subclass, or a callable
                returning one.

        Returns:
            A zero-argument callable producing a fresh activation module.

        Raises:
            TypeError: If ``activation`` is none of the three.
        """
        if isinstance(activation, nn.Module):
            return lambda: copy.deepcopy(activation)
        if isinstance(activation, type) and issubclass(activation, nn.Module):
            return activation
        if callable(activation):
            # The caller's contract says this returns an nn.Module; nothing here can prove it.
            return cast(Callable[[], nn.Module], activation)
        raise TypeError(
            "activation must be an nn.Module instance, nn.Module subclass, or callable "
            "returning an nn.Module."
        )


class CausalGroupNorm(nn.Module):
    r"""Group normalisation over channels only, with **no pooling across time**.

    ``torch.nn.GroupNorm`` on a ``(B, C, T)`` tensor reduces over every non-batch dimension
    inside a group, i.e. over $(C/G, T)$ -- so the mean and variance at step $t$ are functions of
    the whole sequence, including $t' > t$. Every "history" state would therefore carry a
    low-bandwidth image of its own future, which silently invalidates the transfer-entropy
    reading of $K_t$, whose prior is supposed to condition on $Y_{\le t}$ alone. The leak is
    small and entirely invisible in a loss curve; it corrupts only the quantity the model exists
    to measure.

    This module reduces over the channels of each group **at each timestep independently**:

    $$\hat{x}_{b,c,t} = \frac{x_{b,c,t} - \mu_{b,g(c),t}}{\sqrt{\sigma^2_{b,g(c),t} +
    \epsilon}}\,\gamma_c + \beta_c, \qquad
    \mu_{b,g,t} = \frac{G}{C}\sum_{c \in g} x_{b,c,t}.$$

    It registers exactly the parameters ``torch.nn.GroupNorm`` does (``weight`` and ``bias``,
    both of shape ``(C,)``) under the same names, so swapping one for the other leaves a
    ``state_dict`` aligned key-for-key and shape-for-shape.

    Shapes:
        Input:  ``(B, C, T)``
        Output: ``(B, C, T)``
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        """Initialize with the same signature as ``torch.nn.GroupNorm``.

        Args:
            num_groups: Number of channel groups $G$; must divide ``num_channels``.
            num_channels: Channel count $C$.
            eps: Numerical-stability term added to the variance.

        Raises:
            ValueError: If ``num_channels`` is not divisible by ``num_groups``.
        """
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``(B, C, T)`` per group, per timestep.

        Args:
            x: Input of shape ``(B, C, T)``.

        Returns:
            Output of shape ``(B, C, T)``.
        """
        batch, channels, _ = x.shape
        grouped = x.view(batch, self.num_groups, channels // self.num_groups, -1)
        mean = grouped.mean(dim=2, keepdim=True)
        var = grouped.var(dim=2, unbiased=False, keepdim=True)
        normed = ((grouped - mean) / torch.sqrt(var + self.eps)).view(batch, channels, -1)
        return normed * self.weight[None, :, None] + self.bias[None, :, None]

    def extra_repr(self) -> str:
        """Mirror ``torch.nn.GroupNorm``'s repr for readable module trees."""
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"


def causalize_norms(module: nn.Module) -> int:
    """Recursively replace every ``torch.nn.GroupNorm`` with :class:`CausalGroupNorm`.

    The replacement inherits the original affine parameters, so this is numerically a no-op for
    the affine transform and changes only which elements the normalising statistics pool over.

    Args:
        module: Subtree to rewrite **in place**.

    Returns:
        The number of modules replaced.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            causal = CausalGroupNorm(child.num_groups, child.num_channels, child.eps)
            if child.affine:
                with torch.no_grad():
                    causal.weight.copy_(child.weight)
                    causal.bias.copy_(child.bias)
                causal.to(child.weight.device)
            setattr(module, name, causal)
            replaced += 1
        else:
            replaced += causalize_norms(child)
    return replaced


def smooth_bound(r: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    r"""Smoothly map a raw value into the open interval $(lo, hi)$ via a scaled sigmoid.

    Computes $\ell = lo + (hi - lo)\,\sigma(r)$. Unlike ``torch.clamp``, the gradient is
    strictly positive everywhere, with no zero-gradient plateaus, so a log-variance that
    saturates can still recover -- under a hard clamp it is stuck, because the gradient that
    would pull it back is exactly zero.

    The map is **not** idempotent. Callers must apply it to a raw value and never to an
    already-bounded one, which is why the heads return their pre-bound raw log-variance
    alongside the bounded one.

    Args:
        r: Raw pre-bound tensor.
        lo: Lower asymptote of the output range.
        hi: Upper asymptote of the output range.

    Returns:
        A tensor shaped like ``r``, lying strictly inside ``(lo, hi)``.
    """
    return lo + (hi - lo) * torch.sigmoid(r)


def validate_choice(value: str, choices: Tuple[str, ...], name: str) -> str:
    """Check that ``value`` is one of ``choices``.

    Args:
        value: The value to check.
        choices: The permitted values.
        name: The option's name, used in the error message.

    Returns:
        ``value`` unchanged, so this can wrap an assignment.

    Raises:
        ValueError: If ``value`` is not in ``choices``.
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value
