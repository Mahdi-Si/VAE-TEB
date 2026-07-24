"""Input projection and the shared causal encoder body.

Both streams -- the target (fetal heart rate) and the source (uterine pressure) -- are projected
to the model width by an :class:`InputAdapter` and then encoded by a
:class:`CausalConvLstmEncoder` into a per-step history state. The two streams differ only in
their channel count and their convolution kernels; the architecture is identical, so there is one
class for each job rather than a target and a source variant of each.

Causality is the property that matters here. The history state at step $t$ must be a function of
$t' \\le t$ only: the model's whole purpose is to read how much the source's past tells us about
the target's future, and a state that has seen its own future answers that question with the
answer already in it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import (
    CausalMultiChannelConvBlock,
    ResidualMLP,
    geometric_schedule,
)


class InputAdapter(nn.Module):
    """Project a raw feature stream into the internal model width.

    One class serves both streams. They differ only in ``in_dim`` -- $109$ for the target ($43$
    scattering plus $66$ phase-harmonic channels), $58$ or $15$ for the source (with or without
    its $43$ scattering channels) -- and a channel count is an argument, not a subclass.

    Shapes:
        Input:  ``(B, T, in_dim)``
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        dropout: float = 0.1,
        post_residual_activation: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            in_dim: Number of input feature channels. Required rather than defaulted: the two
                streams have different widths and a default would silently fit only one of them.
            d_model: Internal model width.
            dropout: Dropout probability applied after the activation.
            post_residual_activation: Whether the projection's residual MLP ends in a normalise +
                GELU. ``True`` (the default) reproduces the original seam. ``False`` drops the
                post-residual GELU, so the seam does not gate the exported representation or
                attenuate the backward gradient through it.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_mlp = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=post_residual_activation,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, T, in_dim)`` to ``(B, T, d_model)``.

        Args:
            x: Raw feature stream.

        Returns:
            The projected stream at the model width.
        """
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return self.res_mlp(x)


class CausalConvLstmEncoder(nn.Module):
    """Causal encoder producing a per-step history state.

    Two branches run in parallel over the projected stream and are then fused:

    * a stack of dilated causal convolutions, which sees a wide but fixed window -- the
      exponential dilation schedule buys receptive field without buying depth;
    * a unidirectional LSTM, which carries unbounded history but through a bottleneck.

    They fail in different directions, which is why both are here: the conv stack cannot
    remember past its receptive field, and the LSTM blurs what it does remember. Concatenating
    and fusing them lets each cover the other.

    Every component is causal by construction -- left-padded convolutions, a unidirectional
    LSTM, and normalisers that never pool across time -- so the output at $t$ depends only on
    inputs at $t' \\le t$.

    Shapes:
        Input:  ``(B, T, d_model)``
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        *,
        d_model: int,
        cnn_kernels: Tuple[int, ...],
        cnn_dilations: Tuple[int, ...],
        lstm_layers: int,
        lstm_dropout: float,
        conv_dropout: float,
        stack_skip_connection: bool = True,
        post_residual_activation: bool = True,
        conv_norm_groups: Optional[int] = None,
    ) -> None:
        """Initialize the encoder.

        Args:
            d_model: Internal model width, held constant through every stage.
            cnn_kernels: Kernel width per convolution block.
            cnn_dilations: Dilation per convolution block; parallel to ``cnn_kernels``.
            lstm_layers: LSTM depth.
            lstm_dropout: Dropout between LSTM layers. Ignored by PyTorch at a single layer.
            conv_dropout: Dropout inside the convolution blocks and the MLPs.
            stack_skip_connection: Whether to add a second, stack-level residual on top of each
                conv block's own pre-norm residual. ``True`` (the default) reproduces the
                original double-residual stack exactly. ``False`` drops the redundant term --
                and does not build its ``GroupNorm``s at all -- leaving a single clean residual
                chain whose activation scale is stable through depth and whose input skip is
                preserved.
            post_residual_activation: Whether the ``front_mlp`` and ``fusion`` residual MLPs end in
                a normalise + GELU. ``True`` (the default) reproduces the original seams. ``False``
                drops the post-residual GELU at both, so those seams neither gate the encoded
                representation nor attenuate the backward gradient flowing through them.
            conv_norm_groups: Number of groups for every conv block's pre-norm ``GroupNorm``.
                ``None`` (the default) keeps each block's ``min(8, d_model)``. Threaded into the
                blocks' ``norm_groups`` -- distinct from the ``nn.Conv1d`` group count -- so a
                value of ``1`` normalises over all channels per timestep without changing the
                parameter count.

        Raises:
            ValueError: If the kernel and dilation schedules disagree in length, or if no
                convolution block is requested.
        """
        super().__init__()
        self.d_model = d_model

        if len(cnn_kernels) != len(cnn_dilations):
            raise ValueError(
                "cnn_kernels and cnn_dilations must have equal length, got "
                f"{len(cnn_kernels)} and {len(cnn_dilations)}"
            )
        if len(cnn_kernels) < 1:
            raise ValueError("need at least one causal conv block")

        self.post_residual_activation = bool(post_residual_activation)

        # Stage A: front-end residual MLP, holding the channel count at d_model.
        self.front_mlp = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=self.post_residual_activation,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=conv_dropout,
        )

        # Stage B: the dilated causal conv stack. The schedule length is configurable so
        # longer-dilation blocks can be appended for a wider receptive field.
        self.convs = nn.ModuleList(
            [
                CausalMultiChannelConvBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    filter_size=kernel,
                    dilation=dilation,
                    dropout=conv_dropout,
                    activation=nn.GELU,
                    norm_groups=conv_norm_groups,
                )
                for kernel, dilation in zip(cnn_kernels, cnn_dilations)
            ]
        )
        self.stack_skip_connection = bool(stack_skip_connection)
        # One inter-block skip norm between each adjacent conv pair -- built only when the
        # stack-level skip is in use. When off, an unused GroupNorm would still sit in DDP's
        # expectation set as a starved parameter (or force find_unused_parameters=True), so it is
        # not created at all rather than created and skipped.
        self.stack_skip_norms: Optional[nn.ModuleList]
        if self.stack_skip_connection:
            self.stack_skip_norms = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
                    for _ in range(len(self.convs) - 1)
                ]
            )
        else:
            self.stack_skip_norms = None
        self.conv_out_norm = nn.LayerNorm(d_model)

        # Stage C: the unidirectional LSTM branch. Bidirectional would read the future.
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # Stage D: fuse the two branches back down to d_model.
        self.fusion = ResidualMLP(
            input_dim=2 * d_model,
            hidden_dims=geometric_schedule(2 * d_model, d_model, 3),
            final_activation=self.post_residual_activation,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=conv_dropout,
        )
        # Caps the encoder exit to roughly per-step N(0, 1). Without it the exit drifts
        # unbounded, which downstream shows up as a single latent dimension sitting at an
        # absurd prior mean.
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a projected stream into its history state.

        Args:
            x: Projected stream of shape ``(B, T, d_model)``.

        Returns:
            The history state, shape ``(B, T, d_model)``.
        """
        x_lin = self.front_mlp(x)

        # The conv blocks work in (B, C, T); the rest of the model works in (B, T, C).
        x_conv = x_lin.transpose(1, 2).contiguous()
        out = self.convs[0](x_conv)
        skip_norms = self.stack_skip_norms
        for index in range(1, len(self.convs)):
            block_out = self.convs[index](out)
            if skip_norms is not None:
                # The redundant second residual: each conv block is already a pre-norm residual,
                # so this stack-level term adds a second, GroupNorm-rescaled copy of the stream on
                # top of the one the block added. Kept for the sibling; off, the stream stays a
                # single clean residual chain and x_lin re-enters at full weight below.
                block_out = block_out + skip_norms[index - 1](out)
            out = block_out
        conv_out = self.conv_out_norm(out.transpose(1, 2).contiguous() + x_lin)

        lstm_out, _ = self.lstm(x_lin)
        lstm_out = self.lstm_norm(lstm_out)

        fused = torch.cat([conv_out, lstm_out], dim=-1)
        return self.output_norm(self.fusion(fused))
