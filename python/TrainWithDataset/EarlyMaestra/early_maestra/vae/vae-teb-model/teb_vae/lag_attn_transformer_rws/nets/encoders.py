r"""The availability-aware input adapter and the causal conv-Transformer stream encoder.

For each stream $s \in \{Y, U\}$ the path is

$$
X^s \longrightarrow \mathcal G_s \longrightarrow \operatorname{InputAdapter}_s
\longrightarrow \operatorname{CausalConvStem}_s \longrightarrow \operatorname{CausalTransformer}_s
\longrightarrow H^s \in \mathbb R^{B \times T \times 128},
$$

with the channel gate $\mathcal G_s$ living on the model and the last three stages here. The two
streams run through the same two classes at different settings: the target sees the full causal
prefix, the source a bounded causal window, so a source state stays a *local* neighbourhood
summary and the late lag cross-attention keeps its ability to tell adjacent delays apart.

Run this module to print the receptive-field table::

    python -m teb_vae.lag_attn_transformer_rws.nets.encoders
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import ResidualMLP, geometric_schedule
from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    LAYER_SCALE_INIT,
    ROPE_BASE,
    CausalTransformerBlock,
    GatedCausalConvBlock,
    RMSNorm,
)

#: Standard deviation of the learned start embedding, per the architecture's initialisation list.
#: Small enough that a fully unavailable token starts as a quiet, learnable constant rather than
#: as a large perturbation of the residual stream.
START_EMBED_STD = 0.02


def conv_receptive_field(
    kernels: Sequence[int], dilations: Sequence[int]
) -> int:
    r"""Steps of history a stack of causal depthwise convolutions reaches, itself included.

    $$R_{\mathrm{conv}} = 1 + \sum_b (k_b - 1) r_b.$$

    An empty stack reaches one step, which is the identity and is what the stem-free architecture
    arm builds.

    Args:
        kernels: Kernel width per block.
        dilations: Dilation per block; parallel to ``kernels``.

    Returns:
        The reach in decimated steps.

    Raises:
        ValueError: If the two schedules disagree in length -- they are positional against each
            other, so a mismatch would pair the wrong kernel with the wrong dilation.
    """
    if len(kernels) != len(dilations):
        raise ValueError(
            f"kernels and dilations must have equal length, got {len(kernels)} and "
            f"{len(dilations)}; they are positional against each other"
        )
    return 1 + sum((int(kernel) - 1) * int(dilation) for kernel, dilation in zip(kernels, dilations))


class AvailabilityInputAdapter(nn.Module):
    r"""Project a gated feature stream to the model width, telling it which channels are real.

    The stack is the sibling's ``InputAdapter(post_residual_activation=False)`` exactly --
    ``Linear -> LayerNorm -> GELU -> Dropout -> ResidualMLP``, submodule for submodule and name
    for name, so a matched state dict transfers between the two -- with two terms added to the
    first linear's output:

    $$
    e_t = W_x \bar x_t + W_m\!\left(m_t - \mathbf 1\right)
        + \mathbb 1\!\left[\textstyle\sum_c m_{t,c} = 0\right] e_{\mathrm{start}},
    \qquad m_{t,c} = \mathbb 1[t \ge \delta_c].
    $$

    **Why this exists.** When the per-channel causal delay guard is active, the first
    $\max_c \delta_c$ steps of a channel are exact zeros -- no data, a fill value. An exactly zero
    token entering repeated normalisation layers produces derivatives of order
    $1/\sqrt{\epsilon}$, and the as-built guarded configurations of the encoder this replaces reach
    global gradient norms around $10^{26}$ that way. The two terms turn "this position is empty"
    from a numerical accident into a representation the model is told about.

    **Why the projection reads $m_t - \mathbf 1$ rather than $m_t$.** The two differ by the
    constant $W_m \mathbf 1$, which the first linear's own bias already spans, so they are the same
    model. Written this way the term is *exactly* zero wherever every channel is available -- which
    is everywhere past the delayed prefix -- so the availability mechanism cannot quietly shift the
    representation on the part of the sequence where nothing is missing, and the comparison this
    package exists to make stays clean.

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
    #: are absent rather than ``None`` -- which is the same convention the model uses for the gate
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
                the unguarded case -- which the model represents by having no gate object at all,
                not by an identity one.

        Raises:
            ValueError: If ``delays`` is given with a length other than ``in_dim``, or contains a
                negative entry.
        """
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.sequence_length = int(sequence_length)

        # The sibling adapter, submodule for submodule and name for name.
        self.linear = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_mlp = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=False,
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
            # Non-persistent, like every geometry- and budget-shaped tensor in this model: its
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
            x: Gated feature stream ``(B, T, in_dim)``.

        Returns:
            The projected stream ``(B, T, d_model)``.

        Raises:
            ValueError: If the sequence is longer than the availability pattern was built for.
        """
        seq_len = int(x.shape[1])
        embedded = self.linear(x)

        # Both terms are added whenever they exist, on every rank and every batch. The tests are
        # `is None` checks on modules built in __init__, never on tensor content: see the class
        # docstring for why that distinction is the one DDP cares about.
        if self.mask_proj is not None:
            embedded = embedded + self.mask_proj(self._slice(self.availability, seq_len) - 1.0)
        if self.start_embed is not None:
            embedded = embedded + self._slice(self.start_indicator, seq_len) * self.start_embed

        return self.res_mlp(self.drop(self.act(self.norm(embedded))))

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


class CausalConvTransformerEncoder(nn.Module):
    r"""A causal depthwise convolution stem, causal self-attention blocks, and a final norm.

    The stem captures local temporal morphology over a bounded window; the attention blocks supply
    a content-dependent direct path from any admitted earlier step to the current one. Both are
    causal by construction, so $H_t = f(X_{\le t})$ holds per block and therefore per encoder.

    **The final normalisation is a deliberate addition** to the architecture's parameter
    arithmetic, and it is a contract rather than a preference. The encoder this replaces ends in a
    ``LayerNorm``, so the prior head, the posterior fusion and the lag attention's key-value
    normalisation are all calibrated to a normalised state. A pre-norm residual stack without a
    final norm exports an unnormalised residual stream whose scale grows with depth; dropping it
    would move the downstream operating point for reasons that have nothing to do with the encoder
    architecture, and would confound the comparison this model exists to make. It costs $d$
    parameters.

    A stem of zero blocks is legal and builds a working module: it is what the stem-free
    architecture arm needs.

    Shapes:
        Input:  ``(B, T, d_model)``
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        *,
        d_model: int,
        sequence_length: int,
        conv_kernels: Sequence[int],
        conv_dilations: Sequence[int],
        num_attention_blocks: int,
        num_heads: int,
        d_ff: int,
        attention_window: Optional[int] = None,
        dropout: float = 0.0,
        layer_scale_init: float = LAYER_SCALE_INIT,
        rope_base: float = ROPE_BASE,
    ) -> None:
        r"""Build the stem and the attention stack.

        Args:
            d_model: Model width $d$, held constant end to end.
            sequence_length: Longest sequence served; sizes the rotary tables and the window mask.
            conv_kernels: Kernel width per stem block. Empty builds no stem.
            conv_dilations: Dilation per stem block; parallel to ``conv_kernels``.
            num_attention_blocks: Number of causal Transformer blocks $N_s$.
            num_heads: Encoder attention heads $H_e$. Unrelated to the late lag-attention heads and
                to the latent groups; nothing may couple them.
            d_ff: Feed-forward hidden width $d_{\mathrm{ff}}$.
            attention_window: Causal window $W_s$ in steps, or ``None`` for the full causal prefix.
            dropout: Dropout probability on every residual branch output.
            layer_scale_init: Initial LayerScale gain on every branch.
            rope_base: Rotary frequency base.

        Raises:
            ValueError: If the two stem schedules disagree in length, if fewer than one attention
                block is requested, or if the window is not positive.
        """
        super().__init__()
        self.d_model = int(d_model)
        self.sequence_length = int(sequence_length)
        self.conv_kernels = tuple(int(kernel) for kernel in conv_kernels)
        self.conv_dilations = tuple(int(dilation) for dilation in conv_dilations)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.attention_window = None if attention_window is None else int(attention_window)

        # Raises if the two schedules disagree; computed before the blocks so the message names the
        # schedules rather than a kernel that happened to be missing a dilation.
        self.conv_reach = conv_receptive_field(self.conv_kernels, self.conv_dilations)

        if int(num_attention_blocks) < 1:
            raise ValueError(
                f"num_attention_blocks must be at least 1, got {num_attention_blocks}; an encoder "
                f"with no attention is the convolution stack this architecture replaces"
            )
        self.num_attention_blocks = int(num_attention_blocks)

        self.conv_blocks = nn.ModuleList(
            [
                GatedCausalConvBlock(
                    d_model,
                    kernel_size=kernel,
                    dilation=dilation,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                )
                for kernel, dilation in zip(self.conv_kernels, self.conv_dilations)
            ]
        )
        # Raises on a non-positive window, naming it.
        self.attention_blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    self.sequence_length,
                    window=self.attention_window,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                    rope_base=rope_base,
                )
                for _ in range(self.num_attention_blocks)
            ]
        )
        self.output_norm = RMSNorm(d_model)

    @property
    def receptive_field(self) -> Optional[int]:
        r"""Steps of history the encoder's output at $t$ can reach, or ``None`` if unbounded.

        $$R_s = \min\!\left(R_{\mathrm{conv}} + N_s (W_s - 1),\ T\right)$$

        for a windowed encoder. A full-prefix encoder is reported as ``None`` rather than as $T$:
        the bound is *absent*, not merely equal to the sequence length, and a caller that wants to
        say "unbounded" should not have to compare against a geometry constant to find out.
        """
        if self.attention_window is None:
            return None
        reach = self.conv_reach + self.num_attention_blocks * (self.attention_window - 1)
        return min(reach, self.sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, the attention stack and the final normalisation.

        Args:
            x: Projected stream ``(B, T, d_model)``.

        Returns:
            The history state ``(B, T, d_model)``.
        """
        for block in self.conv_blocks:
            x = block(x)
        for block in self.attention_blocks:
            x = block(x)
        return self.output_norm(x)

    def extra_repr(self) -> str:
        """Report the block counts, the attention context and the resulting bound."""
        context = (
            "full causal prefix"
            if self.attention_window is None
            else f"causal window {self.attention_window}"
        )
        bound = "unbounded" if self.receptive_field is None else f"{self.receptive_field} steps"
        return (
            f"d_model={self.d_model}, conv_blocks={len(self.conv_blocks)}, "
            f"attention_blocks={self.num_attention_blocks}, context={context}, "
            f"receptive_field={bound}"
        )


def _describe(steps: Optional[int]) -> str:
    """Render a reach as steps and seconds, or as ``unbounded``."""
    if steps is None:
        return "unbounded"
    return f"{steps} steps / {steps * SECONDS_PER_STEP:.0f} s"


def main() -> None:
    """Print the receptive-field table for the shipped architecture.

    The numbers below are the shipped configuration, written here so the table can be printed
    without a YAML file; ``configs/default.yaml`` is the source of truth for an actual run.
    """
    sequence_length = 300
    kernels, dilations = (5, 9), (1, 2)
    target_blocks, source_blocks, source_window = 4, 3, 16
    # The late lag cross-attention searches $\ell \in \{0, \ldots, 90\}$, spanning 90 steps.
    lag_search_steps = 90

    stem = conv_receptive_field(kernels, dilations)
    source_reach = min(stem + source_blocks * (source_window - 1), sequence_length)

    print("Causal conv-Transformer encoder receptive fields")
    print(f"  seconds per decimated step : {SECONDS_PER_STEP:.0f}")
    print(f"  sequence length            : {sequence_length} steps")
    print(f"  lag search range           : {_describe(lag_search_steps)}")
    print()
    print(f"  stem, kernels {kernels} dilations {dilations} : {_describe(stem)}")
    print(
        f"  target : {target_blocks} blocks, full causal prefix   -> {_describe(None)}"
    )
    print(
        f"  source : {source_blocks} blocks, causal window {source_window:>2} -> "
        f"{_describe(source_reach)}"
    )
    print()
    print(
        "  The source bound is shorter than the lag search range on purpose: the encoder "
        "characterises\n  a local source neighbourhood and the lag attention selects which "
        "neighbourhood matters."
    )


if __name__ == "__main__":
    main()
