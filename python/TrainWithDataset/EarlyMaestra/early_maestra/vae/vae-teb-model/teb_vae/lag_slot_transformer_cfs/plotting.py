r"""Validation-epoch diagnostic callback for the lag-residual causal-feature forecaster.

The page itself lives in :mod:`teb_vae.lag_slot_transformer_cfs.sample_page`, at the package root,
because the evaluation may draw the same page and may import neither Lightning nor this module.
What is here is the callback: when to draw, which samples, what to run to get the tensors the page
needs, and where the files go.

**Almost all of it is inherited.** The rank-zero guard, the sanity-pass guard, the plot-frequency
gate, the validation-batch fetch, the broad handler that keeps a figure failure out of the training
loop, the once-per-run input-budget figure and the MLflow artifact seam are all
:class:`~teb_vae.lag_attn_rws.plotting.LagAttnRwsPlotCallback`'s, and a copy of them here would be
free to drift from the ones every comparison model runs under.

**What is overridden is the forward pass and the page call**, and each for a reason the shared one
cannot serve:

* The page's four lag rows are computed from the **per-lag proposals**, which a forward returns
  only when asked. The shared callback never asks, because on every other model in the family
  there is nothing to ask for.
* Those proposals are the largest tensor this architecture holds -- one entry per sample, anchor,
  candidate lag and latent coordinate -- so the forward runs on the **drawn samples alone** rather
  than on the whole validation batch. A page draws a handful of samples and would otherwise carry
  the whole batch's proposal array to discard all but those.
* The page builder is this package's, because every latent tensor here carries an anchor axis
  rather than a time axis; see the page module for why that is not a row subset away from the
  shared one.

The publication style is applied once, by the inherited constructor. It mutates global
``rcParams``, so applying it per figure would make the timing of any other figure's appearance
decide how it looked.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch

from teb_vae.lag_attn_rws.plotting import (
    LagAttnRwsPlotCallback,
    _guid_of,
    _source_delay_steps,
    input_stream_panels,
    normalization_stats_of,
)
from teb_vae.lag_slot_transformer_cfs.sample_page import (
    build_residual_page,
    residual_lag_panels,
)
from utils.mlflow_utils import log_artifact_to_mlflow
from utils.style import SAVE_DPI, save_figure

__all__ = ["LagResidualTrfCfsPlotCallback"]

#: Batch fields a sliced micro-batch carries. Written out rather than discovered, because the two
#: that are not tensors -- the recording identity and the epoch stamp -- are exactly the ones a
#: generic "slice every tensor" would drop, and the page titles itself from the first of them.
_SLICED_FIELDS: Tuple[str, ...] = (
    "fhr",
    "up",
    "fhr_st",
    "fhr_ph",
    "up_st",
    "up_ph",
    "weight",
    "guid",
    "epoch",
    "source_file_basename",
    "cs_label",
    "bg_label",
)


def _slice_batch(batch: Any, count: int) -> Any:
    """Return a batch holding the first ``count`` samples, and nothing else changed.

    The forward this callback runs retains the per-lag proposals, whose size is one entry per
    sample, anchor, candidate lag and latent coordinate. Drawing two samples out of a production
    batch would otherwise materialise that array for every sample in it.

    Args:
        batch: A batch from the data module, a mapping or an object with the fields as attributes.
        count: How many leading samples to keep.

    Returns:
        A ``SimpleNamespace`` carrying the sliced fields, which is what every consumer of a batch
        in this package reads through.
    """
    fields: Dict[str, Any] = {}
    for name in _SLICED_FIELDS:
        value = batch.get(name) if isinstance(batch, dict) else getattr(batch, name, None)
        if isinstance(value, torch.Tensor):
            fields[name] = value[:count]
        elif isinstance(value, (list, tuple)):
            fields[name] = list(value[:count])
        elif value is not None:
            fields[name] = value
    return SimpleNamespace(**fields)


class LagResidualTrfCfsPlotCallback(LagAttnRwsPlotCallback):
    """Writes this model's validation diagnostic page and routes it to MLflow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the callback, under this package's own output subdirectory.

        Args:
            *args: The inherited positional arguments; see the parent.
            **kwargs: The inherited keyword arguments. ``subdir`` defaults to this package's own,
                so a run that somehow wrote both pages into one output tree keeps them apart.
        """
        kwargs.setdefault("subdir", "lag_residual_trf_cfs_diagnostics")
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _generate_plots(self, trainer: Any, batch: Any, pl_module: Any, epoch: int) -> None:
        """Run one forward over the drawn samples and write one page each.

        The inputs are assembled through the task's own ``_build_forward_inputs`` and the loss
        through the net's own ``compute_loss``, so a figure cannot quietly disagree with the
        objective it illustrates about what the model was fed or what it scored.

        The title's readouts are the drawn samples' own rather than the whole batch's, which is
        what running the forward on the slice costs and what it buys: the page a reader opens and
        the numbers printed on it describe the same recordings.

        Args:
            trainer: The Lightning trainer.
            batch: The validation batch to draw from.
            pl_module: The task.
            epoch: The current epoch.
        """
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        # orig_model, not model: the latter may be a compiled wrapper without the net's methods.
        model = pl_module.orig_model

        # Sliced before the forward, not after: the proposals are what makes this pass expensive.
        drawn = _slice_batch(batch, int(self.num_examples))
        inputs = pl_module._build_forward_inputs(drawn)
        target_features, weight = pl_module._build_raw_target(drawn)

        was_training = pl_module.training
        pl_module.eval()
        try:
            outs = model(*inputs, return_proposals=True)
            # The schedule's value for this epoch, not the raw hyperparameter: under any warm-up
            # the latter is the endpoint and the figure would report a constant.
            beta = float(pl_module._resolve_beta(pl_module.current_epoch))
            # Every objective weight the task passes, not a subset: the figure's recorded scores
            # are read against the training curve, and a weight left at its default here would
            # make the two disagree by exactly that term with nothing on the page saying so.
            scalars = model.compute_loss(
                outs,
                target_features,
                weight=weight,
                beta=beta,
                beta_prior=float(pl_module.hparams.get("beta_prior", 0.0)),
                lambda_full=float(pl_module.hparams.get("lambda_full", 1.0)),
                lambda_base=float(pl_module.hparams.get("lambda_base", 1.0)),
                likelihood=str(pl_module.hparams.get("likelihood", "gaussian_nll")),
                free_bits=float(pl_module.hparams.get("free_bits", 0.0)),
            )["metrics"]
        finally:
            if was_training:
                pl_module.train()

        self._write_budget_figure(trainer, pl_module, model)

        stats = normalization_stats_of(trainer)
        # Resolved off the task rather than captured, so the existing monkeypatch seams still
        # intercept and so a task that supplies none falls back to the page's own defaults.
        forecast_rows = getattr(pl_module, "forecast_rows", None)
        panel_builder = getattr(pl_module, "input_stream_panels", None)
        extra_rows = tuple(getattr(pl_module, "forecast_extra_rows", ()) or ())
        delay_steps = _source_delay_steps(model)

        drawn_count = min(int(self.num_examples), int(inputs[0].shape[0]))
        for index in range(drawn_count):
            guid = _guid_of(drawn, index)
            figure = build_residual_page(
                outs=outs,
                target_features=target_features,
                geometry=model.geometry,
                sample_index=index,
                epoch=epoch,
                guid=guid,
                beta=beta,
                scalars={name: float(value) for name, value in scalars.items()},
                # The raw source, taken from the batch rather than from the forward inputs: this
                # model consumes the decimated blocks and never sees the trace, and the row draws
                # it so a contraction can be found in the same column as the response it is
                # claimed to drive.
                up_raw=getattr(drawn, "up", None),
                normalization_stats=stats,
                delay_steps=delay_steps,
                forecast_rows=forecast_rows,
                batch=drawn,
                # The encoders' actual input: the same tensors the forward above consumed, put
                # through the model's own gates. Built per sample so the row is the recording the
                # rest of the page is about.
                input_streams=input_stream_panels(model, inputs, index, panel_builder),
                forecast_extra_rows=extra_rows,
                lag_panels=residual_lag_panels(model, outs, sample_index=index),
            )
            path = self.output_dir / (
                f"lag_residual_trf_cfs_epoch{epoch:04d}_sample{index}_{guid[:16]}."
                f"{self.file_format}"
            )
            self._save(figure, path, trainer)

    def _save(self, figure: Any, path: Path, trainer: Any) -> None:
        """Write one figure and log it, through the family's own two seams.

        Args:
            figure: The figure to write; closed by the save.
            path: Where it goes.
            trainer: The Lightning trainer, for the rank-zero artifact seam.
        """
        save_figure(figure, path, dpi=SAVE_DPI, close=True)
        log_artifact_to_mlflow(self._mlflow_logger, path, trainer)
