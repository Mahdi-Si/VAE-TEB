"""The input half of the patch model: ``PatchEmbedding`` adapters and the tiled-anchor forward.

:class:`PatchStreamInputs` reuses ``CausalWarmupInputs`` (tiled anchors, no gates, no warm-up, no
alignment) and changes only what a patch token changes: the adapter, the two warm-up hooks (a patch
has no per-channel warm-up) and the forward's arity.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch

from teb_vae.lag_attn.nets.delays import ChannelGate
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_transformer_patch.nets.patching import PatchEmbedding


class PatchStreamInputs(CausalWarmupInputs):
    """Patch-token inputs over the CFS tiled-anchor forward. Place first in the model's bases."""

    def _build_adapter(
        self, gate: Optional[ChannelGate], declared_width: int, dropout: float
    ) -> PatchEmbedding:
        """Both streams: ``PatchEmbedding`` at the declared width ``2R + 1`` (gates are ``None``)."""
        return PatchEmbedding(
            in_dim=declared_width,
            d_model=self.d_model,
            sequence_length=self.sequence_length,
            dropout=dropout,
        )

    @staticmethod
    def _check_anchor_floor(*_: object) -> None:
        """No per-channel input warm-up, so no floor beyond ``warmup_period`` to enforce."""

    def _resolve_warmup_readout_constants(self) -> None:
        """No warm-up readouts. Must live here: CWI defines it, so a later base would lose the MRO."""

    def forward(
        self,
        y_patch: torch.Tensor,
        u_patch: torch.Tensor,
        anchor_phase: Optional[Union[int, torch.Tensor]] = None,
        anchor_stride: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the CFS forward on ``(B, T, 2R + 1)`` patch streams; returns its key set.

        The parameter is ``u_patch``, not ``u_stream``, so the classifier's ``kld_excess`` gate
        refuses this model instead of handing it the phase tensor.
        """
        # The CFS forward takes the target as two blocks it concatenates; the empty second block
        # makes that concatenation the patch stream itself, which is why this wrapper exists.
        return super().forward(y_patch, y_patch[..., :0], u_patch, anchor_phase, anchor_stride)


__all__ = ["PatchStreamInputs"]
