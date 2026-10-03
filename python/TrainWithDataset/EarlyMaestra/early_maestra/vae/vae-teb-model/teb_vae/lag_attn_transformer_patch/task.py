r"""The training task for the patch-input cell.

* **Parent.** :class:`~teb_vae.lag_attn_transformer_crws.task.SeqVaeLagAttnTrfCrwsTask` (plan C.3,
  C6), which already carries the run seed, the class-default ``_stage`` that ``VaeSource`` relies on
  when it calls ``_build_forward_inputs`` outside a step, the stage-setting
  ``compute_loss_and_metrics``, the bound anchor-phase members and the step-granular LR ramp. Only
  the two members that name the input layout are written here.
* **Source-null.** The forward tuple is ``(y_patch, u_patch, phase, stride)``, so the source stream
  is ``inputs[1]``; the CFS ``_added_metrics`` reads ``inputs[2]``, which is the phase here.
* **No forecast page.** The inherited plotting seams (``forecast_rows``, ``input_stream_panels``,
  ``input_budget_figure``) draw raw or feature rows and are read only by the plotting callback,
  which this package's configs disable.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask
from teb_vae.lag_attn_transformer_patch.nets.patching import patchify


class SeqVaeLagAttnTrfPatchTask(SeqVaeLagAttnTrfCrwsTask):
    """Lightning task for :class:`~teb_vae.lag_attn_transformer_patch.nets.model.SeqVaeLagAttnTrfPatch`."""

    def _build_forward_inputs(self, batch: Any) -> Tuple[Any, ...]:
        """Return ``(y_patch, u_patch, anchor_phase, anchor_stride)`` for the net's forward.

        The target stream is masked by FHR's ``weight``; the source stream by the model's
        ``source_validity`` (``"finite"`` ignores ``weight``). The anchor geometry follows
        ``self._stage``: dense ``(0, 1)`` outside a step and on val/test.
        """
        model = self.orig_model
        r = model.raw_per_step
        y_patch = patchify(batch.fhr, batch.weight, raw_per_step=r, validity="fhr_weight")
        u_patch = patchify(batch.up, batch.weight, raw_per_step=r, validity=model.source_validity)
        phase, stride = self.resolve_anchor_geometry(self._stage, batch)
        return y_patch, u_patch, phase, stride

    def _added_metrics(
        self,
        inputs: Tuple[Any, ...],
        forward_outputs: Dict[str, torch.Tensor],
        weight: torch.Tensor,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        """The source-null KL floor on val/test; absent (not zero-filled) on train."""
        if stage == "train":
            return {}
        return {
            "kld_source_null": controls.source_null_kld(
                self.orig_model, forward_outputs, inputs[1], weight
            )
        }


__all__ = ["SeqVaeLagAttnTrfPatchTask"]
