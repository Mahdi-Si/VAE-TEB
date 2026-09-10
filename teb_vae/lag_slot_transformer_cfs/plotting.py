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

* **The objective is not called here, and calling it would stop the fit.** The shared callback
  recomputes the loss so that a figure cannot disagree with the objective it illustrates, which is
  right for every cell whose objective is collective-free. This one's is not: it ends in a single
  ``all_reduce`` over its packed denominators, and this callback runs on rank zero alone. Rank zero
  would block there for six peers that are already running the next epoch; they would block at
  their own next collective; and the run would stop with nothing written and nothing raised. The
  title's numbers come from what the epoch already logged instead -- see :func:`_epoch_readouts`,
  which explains why that is the better source and not merely the safe one.
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

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
from loguru import logger

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn_rws.plotting import (  # noqa: E402
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

__all__ = ["LagResidualTrfCfsPlotCallback", "PAGE_DPI"]

#: Resolution the page is written at, and it is deliberately **not** the family's publication
#: :data:`~utils.style.SAVE_DPI`.
#:
#: This page is roughly $14 \times 52$ inches, and every heatmap on it is a small array -- one
#: column per decoded anchor, one row per channel or per candidate lag. Matplotlib resamples each
#: image to the axes' size in *device* pixels, so at $600$ dpi a map of a few hundred columns is
#: upsampled to eight thousand, once per row, twice per save.
#:
#: MEASURED at the production geometry on one CPU, for a single page: $23.5$ s at $600$ dpi with
#: the shared helper's tight bounding box, against $4.0$ s here. The written PDF is the same size
#: to within a few percent -- $244$ kB against $216$ kB -- because the file is dominated by vector
#: text and lines, so the extra twenty seconds bought no detail at all. On rank zero inside a
#: distributed fit that time is not merely this callback's: every other rank waits at the next
#: collective until it finishes.
PAGE_DPI = 200

#: Stage prefix the framework gives a validation metric on its way out. The page's title names its
#: readouts without one, as the objective produces them.
_VALIDATION_PREFIX = "val/"

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


def _epoch_readouts(trainer: Any) -> Dict[str, float]:
    r"""The validation readouts this epoch already logged, for the page's title.

    **This is deliberately not a fresh call to the objective, and that is the whole point.**

    This architecture's objective performs one ``all_reduce`` per call, on the packed vector its
    denominators and reported sums are built from: see
    :func:`~teb_vae.lag_slot_transformer_cfs.nets.objective.compute_residual_objective`. That is
    correct inside a step, where every rank calls it together. This callback runs on **rank zero
    alone** -- the inherited handler returns early everywhere else -- so a call from here enters a
    collective that no other rank will ever join. Rank zero waits for six peers already running
    the next epoch, they reach their own next collective and wait for it, and the fit stops with
    nothing written and nothing raised. The sibling cell's objective has no collective at all,
    which is why the same callback shape is safe there and why this had to change here.

    Taking the epoch's own logged values is better than a guarded recompute would be, not merely
    safer. They are the numbers on the training curve, already reduced across every rank over the
    whole validation set, so the figure and the curve a reader holds it against cannot disagree.

    Args:
        trainer: The Lightning trainer, for the metrics it has already collected.

    Returns:
        The validation metrics with their stage prefix removed, as floats. Empty when the trainer
        carries none, which is what a hand-built one gives and which the title renders by simply
        omitting the line.
    """
    metrics = getattr(trainer, "callback_metrics", None) or {}
    readouts: Dict[str, float] = {}
    for key, value in metrics.items():
        name = str(key)
        if not name.startswith(_VALIDATION_PREFIX):
            continue
        try:
            readouts[name[len(_VALIDATION_PREFIX) :]] = float(value)
        except (TypeError, ValueError):
            # A non-scalar entry is some other callback's, and the title wants numbers.
            continue
    return readouts


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

        The inputs are assembled through the task's own ``_build_forward_inputs``, so the figure
        cannot quietly disagree with the run about what the model was fed.

        **One forward, and nothing that synchronises.** Everything here happens on rank zero
        alone, so this method must not enter a collective; the module docstring and
        :func:`_epoch_readouts` say what that ruled out and why. The forward runs over the drawn
        samples only, because the per-lag proposals it must retain are the largest tensor this
        architecture holds.

        The title's numbers are the epoch's, taken from what the run already logged. The rows
        below are one recording's, and the page says which is which.

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
        # The validity signal is not taken here: the page's own score row reads it off the batch,
        # through the same field the objective's mask is built from.
        target_features, _weight = pl_module._build_raw_target(drawn)

        # THE FORWARD, AND NOTHING ELSE THAT COULD SYNCHRONISE. The objective is **not** called
        # here, however tempting it is to have the figure recompute what it reports: it ends in an
        # all-reduce, this runs on rank zero alone, and the fit would stop there. See
        # :func:`_epoch_readouts`, which is where the title's numbers come from instead.
        was_training = pl_module.training
        pl_module.eval()
        try:
            outs = model(*inputs, return_proposals=True)
        finally:
            if was_training:
                pl_module.train()

        # The schedule's value for this epoch, not the raw hyperparameter: under any warm-up the
        # latter is the endpoint and the figure would report a constant.
        beta = float(pl_module._resolve_beta(pl_module.current_epoch))
        scalars = _epoch_readouts(trainer)

        self._write_budget_figure(trainer, pl_module, model)

        stats = normalization_stats_of(trainer)
        # Resolved off the task rather than captured, so the existing monkeypatch seams still
        # intercept and so a task that supplies none falls back to the page's own defaults.
        forecast_rows = getattr(pl_module, "forecast_rows", None)
        panel_builder = getattr(pl_module, "input_stream_panels", None)
        extra_rows = tuple(getattr(pl_module, "forecast_extra_rows", ()) or ())
        delay_steps = _source_delay_steps(model)

        drawn_count = min(int(self.num_examples), int(inputs[0].shape[0]))
        # Announced before the first page rather than after the last, and this is not decoration.
        # On rank zero inside a distributed fit every other rank waits at the next collective for
        # as long as this takes, and the page's own file does not exist until its save returns --
        # so a run that is drawing looks from the outside exactly like a run that has stopped.
        started = time.perf_counter()
        logger.info(
            f"drawing {drawn_count} diagnostic page(s) for epoch {epoch} into {self.output_dir}"
        )
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
        logger.info(
            f"diagnostic page(s) for epoch {epoch} written in "
            f"{time.perf_counter() - started:.1f} s"
        )

    def _save(self, figure: Any, path: Path, trainer: Any) -> None:
        """Write one page and log it, at this page's own resolution.

        ``utils.style.save_figure`` is deliberately not used, and it is the only place this
        package departs from a family seam. That helper always passes ``bbox_inches="tight"``,
        which costs a **second** full draw of the figure to measure the artists before the one
        that writes them, and it saves at the publication resolution. Both are right for a figure
        going into a paper and wrong for a seventeen-row diagnostic drawn inside a training loop:
        together they were nineteen of this page's twenty-three seconds, and this page's margins
        are set explicitly in the ``GridSpec`` rather than discovered, so there is no surrounding
        whitespace for a tight box to crop.

        The close stays in a ``finally`` for the reason that helper documents: the caller has
        given up its handle, and pyplot holds every unclosed figure in a global registry, so a
        save that raises must not be the one path that keeps this one alive.

        Args:
            figure: The figure to write; closed here either way.
            path: Where it goes.
            trainer: The Lightning trainer, for the rank-zero artifact seam.
        """
        try:
            figure.savefig(str(path), dpi=PAGE_DPI)
        finally:
            plt.close(figure)
        log_artifact_to_mlflow(self._mlflow_logger, path, trainer)
