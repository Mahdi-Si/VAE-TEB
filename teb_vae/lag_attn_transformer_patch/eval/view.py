r"""The evaluation view of the patch model and task (plan D3): the CFS calling convention, no new weights.

Every shared call site in ``teb_vae/lag_attn_cfs/eval`` builds inputs through ``metrics.model_inputs``
(the task's three ``_build_*`` builders) and calls ``model(y_st, y_ph, u_stream, anchor_phase=,
anchor_stride=)``. These two classes present the patch cell in exactly that form:

* ``y_st = y_patch`` ``(B, T, 2R + 1)``, ``y_ph = y_patch[..., :0]`` (empty), ``u_stream = u_patch``;
* ``target_features = model.summary_target(fhr, weight)`` ``(B, T, 2)`` -- the standardized
  ``[level, variability]`` the forecast is scored against, gathered by the model's own
  ``_build_forecast_target`` (plan D4), so an evaluated ``nll_*`` is the training loss's.

:class:`SeqVaeLagAttnTrfPatch` keeps the training class's **name** (the checkpoint class guard compares
names) and adds no parameter or buffer (the strict state-dict load must still align). It declares the
eval constants a patch model makes true by construction. Nothing here is used by training.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_transformer_patch.nets import model as _nets_model
from teb_vae.lag_attn_transformer_patch.nets.patching import patchify
from teb_vae.lag_attn_transformer_patch.task import SeqVaeLagAttnTrfPatchTask


class SeqVaeLagAttnTrfPatch(_nets_model.SeqVaeLagAttnTrfPatch):
    """The patch model under the CFS five-argument forward. Same name, same weights.

    Constants the shared collection reads, each true by construction for patch tokens:

    * ``TARGET_BLOCK_SPLIT = 1``: kept channel 0 (``level``) lands in the ``*_st`` columns and channel
      1 (``variability``) in the ``*_ph`` ones (``pred_gap_st/_ph``, ``ar_coef_mean_st/_ph``).
    * ``target_warm_frac = 1.0``: there is no per-channel warm-up, so every scored cell is warm.
    * ``source_block_warm_st/_ph``: all-True ``(T,)``, so ``source_lag_warmth_frac_*`` is 1.0.

    ``warm_tertile_id`` is deliberately **absent**: the shared pass then writes NaN
    ``pred_gap_warm_*`` columns instead of a fabricated split, and ``warmup`` is excluded.
    """

    TARGET_BLOCK_SPLIT = 1
    target_warm_frac = 1.0

    @property
    def source_block_warm_st(self) -> torch.Tensor:
        """All stored steps are warm: no source warm-up on a raw patch."""
        return torch.ones(self.sequence_length, dtype=torch.bool)

    @property
    def source_block_warm_ph(self) -> torch.Tensor:
        """See :attr:`source_block_warm_st`."""
        return torch.ones(self.sequence_length, dtype=torch.bool)

    def forward(  # type: ignore[override]
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        anchor_phase: Optional[Union[int, torch.Tensor]] = None,
        anchor_stride: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """The CFS forward: ``(y_patch, empty, u_patch, φ, S)``, exactly what the training forward runs."""
        return CausalWarmupInputs.forward(self, y_st, y_ph, u_stream, anchor_phase, anchor_stride)


class SeqVaeLagAttnTrfPatchEvalTask(SeqVaeLagAttnTrfPatchTask):
    """The patch task with every shared builder returning patch streams in the CFS layout."""

    def _build_target_streams(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(y_patch, y_patch[..., :0])``: the FHR patch stream, masked by ``weight``."""
        model = self.orig_model
        y_patch = patchify(batch.fhr, batch.weight, raw_per_step=model.raw_per_step, validity="fhr_weight")
        return y_patch, y_patch[..., :0]

    def _build_source_stream(self, batch: Any) -> torch.Tensor:
        """``u_patch``: the UP patch stream under the model's own ``source_validity``."""
        model = self.orig_model
        return patchify(batch.up, batch.weight, raw_per_step=model.raw_per_step, validity=model.source_validity)

    def _build_raw_target(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(summaries (B, T, 2), weight)``: the standardized target on the token grid."""
        return self.orig_model.summary_target(batch.fhr, batch.weight), batch.weight

    def _build_forward_inputs(self, batch: Any) -> Tuple[Any, ...]:
        """``(y_patch, empty, u_patch, phase, stride)``; dense ``(0, 1)`` outside a training step."""
        y_st, y_ph = self._build_target_streams(batch)
        phase, stride = self.resolve_anchor_geometry(self._stage, batch)
        return y_st, y_ph, self._build_source_stream(batch), phase, stride


__all__ = ["SeqVaeLagAttnTrfPatch", "SeqVaeLagAttnTrfPatchEvalTask"]
