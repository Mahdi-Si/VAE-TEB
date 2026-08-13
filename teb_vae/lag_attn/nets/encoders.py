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

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import (
    CausalMultiChannelConvBlock,
    ResidualMLP,
    geometric_schedule,
)

#: Standard deviation of the learned start embedding in :class:`AvailabilityInputAdapter`. Small
#: enough that a fully unavailable token starts as a quiet, learnable constant rather than as a
#: large perturbation of the residual stream.
START_EMBED_STD = 0.02


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


class AvailabilityInputAdapter(nn.Module):
    r"""Project a gated feature stream to the model width, telling it which channels are real.

    The stack is :class:`InputAdapter` exactly -- ``Linear -> LayerNorm -> GELU -> Dropout ->
    ResidualMLP``, submodule for submodule and name for name, at whichever residual seam
    ``post_residual_activation`` selects, so a matched state dict transfers between the two and an
    unguarded adapter is parameter-for-parameter the plain one it replaces -- with two terms added
    to the first linear's output:

    $$
    e_t = W_x \left(x_t \odot m_t\right) + W_m\!\left(m_t - \mathbf 1\right)
        + \mathbb 1\!\left[\textstyle\sum_c m_{t,c} = 0\right] e_{\mathrm{start}},
    \qquad m_{t,c} = \mathbb 1[t \ge \delta_c].
    $$

    **The mask and the announcement are one vector.** $m_t$ both zeroes the input and drives the
    announcement term, so the two cannot describe different regions -- which a separate masking
    module beside this one could, silently, with every shape still correct. For a stream that
    arrived through :class:`~teb_vae.lag_attn.nets.delays.ChannelDelay` the multiply changes
    nothing: that module already returns ``gathered * available`` under the same $\delta$, so
    those positions are exactly zero before they get here. It is load-bearing for a stream whose
    unavailable prefix holds *real values on no defined scale* -- coefficients a one-sided
    transform emitted from assumed pre-recording history, normalised with constants accumulated
    while deliberately excluding them.

    **Why this exists.** When the per-channel causal delay guard is active, the first
    $\max_c \delta_c$ steps of a channel are exact zeros -- no data, a fill value. An exactly zero
    token entering repeated normalisation layers produces derivatives of order
    $1/\sqrt{\epsilon}$, and a guarded configuration built on the plain :class:`InputAdapter`
    reaches global gradient norms around $10^{26}$ that way at *every* finite budget -- a switch,
    not a gradient, so raising the warm-up does not help and the clip coefficient leaves the run
    optimising nothing but weight decay. The two terms turn "this position is empty" from a
    numerical accident into a representation the model is told about.

    **Why the projection reads $m_t - \mathbf 1$ rather than $m_t$.** The two differ by the
    constant $W_m \mathbf 1$, which the first linear's own bias already spans, so they are the same
    model. Written this way the term is *exactly* zero wherever every channel is available -- which
    is everywhere past the delayed prefix -- so the availability mechanism cannot quietly shift the
    representation on the part of the sequence where nothing is missing, and a comparison between a
    guarded and an unguarded arm stays clean.

    **What is conditional and what is not.** Both terms are added **unconditionally in the
    forward**; whether each exists is settled at construction. That order matters, and the
    intuitive reading of why is wrong: a parameter multiplied by an identically-zero tensor is
    still reachable -- its ``AccumulateGrad`` node fires and it receives a zeros gradient, so
    ``DistributedDataParallel`` marks it ready. What breaks ``find_unused_parameters=False`` is a
    parameter left *out of the graph*, which is what a data-dependent
    ``if indicator.any(): e = e + e_start`` would do, on some ranks and not others, on some batches
    and not others. There is no such branch here.

    Construction is conditional for parameter economy and honesty instead, and the two terms have
    different conditions because they become non-trivial at different points:

    * $W_m$ exists when $\max_c \delta_c > 0$. Below that $m \equiv 1$, the term is identically
      zero, and the projection would be a parameter that can never receive a gradient.
    * $e_{\mathrm{start}}$ exists when $\min_c \delta_c > 0$. The indicator is non-zero for some
      $t$ exactly when *every* channel is delayed, so a mixed delay vector such as $(0, 3, 5)$
      satisfies $\max > 0$ while leaving the start token permanently inert.

    With no delays at all -- the unguarded configuration, where the model builds no gate object --
    neither exists, which is correct: without delays there is no all-zero prefix to repair.

    Shapes:
        Input:  ``(B, T, in_dim)``
        Output: ``(B, T, d_model)``
    """

    #: Declared so the registered buffers type as tensors rather than as ``Tensor | Module``. They
    #: are registered only when their term is built, so on an unguarded adapter these attributes
    #: are absent rather than ``None`` -- which is the same convention the models use for the gate
    #: itself, and which keeps an inert buffer out of every ``named_buffers`` listing.
    availability: torch.Tensor
    start_indicator: torch.Tensor

    def __init__(
        self,
        *,
        in_dim: int,
        d_model: int,
        sequence_length: int,
        dropout: float = 0.1,
        delays: Optional[Sequence[int]] = None,
        post_residual_activation: bool = False,
    ) -> None:
        r"""Build the projection stack and whichever availability terms the delays call for.

        Args:
            in_dim: Input channel count -- the *surviving* width the gate emits, not the declared
                one.
            d_model: Internal model width.
            sequence_length: Sequence length $T$ the availability pattern is built for. The
                pattern is a constant of the delays, not a function of the batch, so it is built
                once here.
            dropout: Dropout probability after the activation and inside the residual MLP.
            delays: One delay $\delta_c$ per surviving channel, in decimated steps, or ``None`` for
                the unguarded case -- which the models represent by having no gate object at all,
                not by an identity one.
            post_residual_activation: Whether the residual MLP ends in a normalise + GELU, exactly
                as on :class:`InputAdapter`. Defaults to ``False``, the ungated seam the
                raw-signal models want; ``True`` reproduces the original gated seam. A parameter
                rather than a fixed choice so that swapping a plain adapter for this one leaves an
                *unguarded* model bitwise unchanged -- otherwise adopting the availability terms
                would silently move a model that has no delays for them to repair.

        Raises:
            ValueError: If ``delays`` is given with a length other than ``in_dim``, or contains a
                negative entry.
        """
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.sequence_length = int(sequence_length)

        # InputAdapter, submodule for submodule and name for name.
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

        delay_values = self._validate_delays(delays, self.in_dim)
        self.max_delay = max(delay_values) if delay_values else 0
        self.min_delay = min(delay_values) if delay_values else 0

        self.mask_proj: Optional[nn.Linear] = None
        self.start_embed: Optional[nn.Parameter] = None
        if self.max_delay > 0:
            pattern = self._availability_pattern(delay_values, self.sequence_length)
            # Non-persistent, like every geometry- and budget-shaped tensor in these models: its
            # width is the surviving-channel count, so a persistent copy would make a checkpoint
            # trained at one reach budget fail to load at another, reported as misaligned keys
            # rather than as a budget mismatch.
            self.register_buffer("availability", pattern, persistent=False)
            self.mask_proj = nn.Linear(self.in_dim, d_model, bias=False)
            if self.min_delay > 0:
                indicator = (pattern.sum(dim=-1) == 0).to(pattern.dtype).unsqueeze(-1)
                self.register_buffer("start_indicator", indicator, persistent=False)
                self.start_embed = nn.Parameter(torch.randn(d_model) * START_EMBED_STD)

    @staticmethod
    def _validate_delays(delays: Optional[Sequence[int]], in_dim: int) -> List[int]:
        """Return the delays as a list of ints, or an empty list for the unguarded case.

        Args:
            delays: The per-survivor delays, or ``None``.
            in_dim: The surviving channel count the delays must be positional against.

        Returns:
            The delays, or ``[]``.

        Raises:
            ValueError: If the length disagrees with ``in_dim`` or any entry is negative.
        """
        if delays is None:
            return []
        values = [int(value) for value in delays]
        if len(values) != in_dim:
            raise ValueError(
                f"delays has {len(values)} entries but the adapter reads {in_dim} channels; the "
                f"delay vector is positional against the surviving channels, so a length mismatch "
                f"would mark the wrong channels unavailable with no other failure signal"
            )
        negative = [(index, value) for index, value in enumerate(values) if value < 0]
        if negative:
            raise ValueError(
                f"delays must be >= 0; got negative entries at {negative}"
            )
        return values

    @staticmethod
    def _availability_pattern(delays: Sequence[int], sequence_length: int) -> torch.Tensor:
        r"""Build $m_{t,c} = \mathbb 1[t \ge \delta_c]$ as a $(T, C)$ float tensor.

        Args:
            delays: One delay per channel. Empty gives an all-ones $(T, 0)$ pattern, which no
                caller uses.
            sequence_length: Sequence length $T$.

        Returns:
            The availability pattern, ``float32``.
        """
        steps = torch.arange(sequence_length).unsqueeze(-1)
        return (steps >= torch.tensor(list(delays), dtype=torch.long)).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the stream and add whichever availability terms were built.

        Args:
            x: Feature stream ``(B, T, in_dim)`` at the surviving width. Whatever it holds inside
                each channel's unavailable prefix is masked away here, so a caller need not have
                zeroed it.

        Returns:
            The projected stream ``(B, T, d_model)``.

        Raises:
            ValueError: If the channel count is not ``in_dim``, or if the sequence is longer than
                the availability pattern was built for.
        """
        seq_len = int(x.shape[1])
        self._validate_stream(x)

        # Both terms are added whenever they exist, on every rank and every batch. The tests are
        # `is None` checks on modules built in __init__, never on tensor content: see the class
        # docstring for why that distinction is the one DDP cares about. The same pattern zeroes
        # the input and announces it -- one vector, so the two cannot disagree.
        if self.mask_proj is not None:
            available = self._slice(self.availability, seq_len)
            embedded = self.linear(x * available) + self.mask_proj(available - 1.0)
        else:
            embedded = self.linear(x)
        if self.start_embed is not None:
            embedded = embedded + self._slice(self.start_indicator, seq_len) * self.start_embed

        return self.res_mlp(self.drop(self.act(self.norm(embedded))))

    def _validate_stream(self, x: torch.Tensor) -> None:
        """Refuse a stream whose channel count is not the width this adapter was built for.

        Checked rather than left to the projection. Without the availability term ``self.linear``
        refuses a wrong width itself, but ``x * available`` **broadcasts**: a squeezed or
        mis-sliced single-channel stream fans out to every channel and produces a plausible
        encoding, so the guarded adapters -- the ones with a warm-up to respect -- would be the
        lenient ones.

        A method rather than a branch inside :meth:`forward`, and the distinction is the DDP rule
        rather than style: the forward's own control flow must stay a set of construction-time
        ``is None`` tests, so that no step can skip a projection and leave its parameter unready on
        the ranks whose batch did not need it. This runs unconditionally, on every rank and every
        batch, and either returns or raises.

        Args:
            x: The stream handed to :meth:`forward`.

        Raises:
            ValueError: If ``x`` is not 3-D or its last axis is not ``in_dim``.
        """
        if x.dim() != 3 or int(x.shape[-1]) != self.in_dim:
            raise ValueError(
                f"the stream is {tuple(x.shape)} but this adapter reads {self.in_dim} channels; "
                f"the availability pattern is positional against that width and broadcasts over a "
                f"narrower one rather than refusing it"
            )

    def _slice(self, pattern: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Take the first ``seq_len`` steps of a constant pattern, refusing a longer request.

        Args:
            pattern: A ``(T, ...)`` constant built at ``sequence_length``.
            seq_len: The batch's sequence length.

        Returns:
            The leading ``seq_len`` steps.

        Raises:
            ValueError: If ``seq_len`` exceeds what the pattern was built for.
        """
        if seq_len > self.sequence_length:
            raise ValueError(
                f"sequence of {seq_len} steps exceeds the availability pattern built for "
                f"sequence_length={self.sequence_length}"
            )
        return pattern[:seq_len]

    def extra_repr(self) -> str:
        """Report the widths and which availability terms were built."""
        terms = [
            name
            for name, built in (
                ("W_m", self.mask_proj is not None),
                ("e_start", self.start_embed is not None),
            )
            if built
        ]
        return (
            f"{self.in_dim} -> {self.d_model}, max_delay={self.max_delay}, "
            f"availability_terms={terms or 'none'}"
        )


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
