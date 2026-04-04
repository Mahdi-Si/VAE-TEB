"""Causal convolutional stems for the Causal Multimodal Forecasting Transformer.

The stems project raw scattering-transform channels to the backbone dimension
and apply residual causal convolution blocks to learn local temporal motifs
(spec §7).  A single ``CausalStem`` class is instantiated separately for the
FHR and UP modalities.
"""

import torch.nn as nn
from torch import Tensor

from .layers import CausalConvBlock


class CausalStem(nn.Module):
    """Causal convolutional stem shared by FHR and UP pathways (spec §7.3).

    Architecture::

        Linear(in_dim → d_model)
        → CausalConvBlock(k=k_0, dil=dil_0)  (block 0)
        → CausalConvBlock(k=k_1, dil=dil_1)  (block 1)
        → ...
        → CausalConvBlock(k=k_{N-1}, dil=dil_{N-1})  (block N-1)

    Each block uses depthwise causal convolution with increasing receptive
    field, followed by pointwise expansion and residual connection.

    Args:
        in_dim: Number of input channels (``d_F`` or ``d_U``).
        d_model: Backbone hidden dimension.
        num_blocks: Number of residual causal-conv blocks.
        kernels: Kernel sizes per block (length must equal ``num_blocks``).
        dilations: Dilation factors per block (length must equal ``num_blocks``).
        expansion: Pointwise expansion ratio inside each block.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_blocks: int = 3,
        kernels: tuple = (3, 5, 5),
        dilations: tuple = (1, 2, 4),
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            CausalConvBlock(
                d_model=d_model,
                kernel_size=k,
                dilation=dil,
                expansion=expansion,
                dropout=dropout,
            )
            for k, dil in zip(kernels, dilations)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input scattering-transform features of shape
                ``(B, T, in_dim)``.

        Returns:
            Stem output of shape ``(B, T, d_model)``.
        """
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return h
