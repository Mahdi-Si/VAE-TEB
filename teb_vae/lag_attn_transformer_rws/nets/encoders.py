r"""The causal conv-Transformer stream encoder.

The availability-aware input adapter that precedes it is re-exported here but lives in
``teb_vae/lag_attn/nets/encoders.py``: three packages build one, so it sits in the shared layer
beside the plain :class:`~teb_vae.lag_attn.nets.encoders.InputAdapter` whose stack it reproduces.

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

from typing import Optional, Sequence

import torch
from torch import nn

from teb_vae.lag_attn.nets.encoders import START_EMBED_STD, AvailabilityInputAdapter
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    LAYER_SCALE_INIT,
    ROPE_BASE,
    CausalTransformerBlock,
    GatedCausalConvBlock,
    RMSNorm,
)

#: Re-exported so ``from ...nets.encoders import AvailabilityInputAdapter`` keeps resolving beside
#: :class:`CausalConvTransformerEncoder`, which is the only thing this package pairs it with. The
#: adapter itself lives in ``teb_vae/lag_attn/nets/encoders.py`` because three packages build one.
__all__ = [
    "AvailabilityInputAdapter",
    "CausalConvTransformerEncoder",
    "START_EMBED_STD",
    "conv_receptive_field",
]


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
