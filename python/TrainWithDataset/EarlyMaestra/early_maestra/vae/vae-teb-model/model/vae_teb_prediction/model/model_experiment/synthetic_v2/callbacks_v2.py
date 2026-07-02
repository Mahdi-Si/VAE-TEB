r"""Training callbacks for ``synthetic_v2``.

Currently a single callback: :class:`LossPlotHtmlCallback`, which refreshes the
interactive Plotly HTML training curve (one distinctly-coloured trace per logged metric)
*during* training so a long headline run can be watched mid-flight. This interactive HTML
is the only training-curve output -- there is no static matplotlib PDF/PNG twin. It is
wired into the trainer by
:func:`~model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2._build_trainer_v2`
when the ``plotting`` config block enables it (``plotting.enabled`` and ``plotting.html``).

See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import lightning.pytorch as pl
from loguru import logger


class LossPlotHtmlCallback(pl.Callback):
    r"""Rewrite an interactive Plotly HTML loss curve every ``every_n_epochs`` epochs.

    Delegates the actual rendering to
    :func:`~model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2.plot_loss_curves_html`,
    which reads the Lightning ``CSVLogger`` ``metrics.csv`` and writes a single
    self-contained ``.html``. The callback flushes the logger first
    (``trainer.logger.save()``) so the most recent epoch's rows are on disk before the
    read.

    It renders on both :meth:`on_train_epoch_end` (covers no-validation runs) and
    :meth:`on_validation_epoch_end` (guarantees ``val/*`` epoch metrics are present),
    every ``every_n_epochs`` epochs, and unconditionally once on :meth:`on_fit_end` so a
    complete final curve exists even when the last epoch is not a multiple of the period.
    Rendering is **rank-0 only** and fully **non-fatal**: any failure (including a missing
    optional ``plotly`` install, surfaced by ``plot_loss_curves_html`` returning ``None``)
    is logged and swallowed so it can never break training.

    Args:
        out_stem: Output path stem for the HTML (``.html`` is appended by
            :func:`plot_loss_curves_html`), e.g. ``results/<tag>/figures/training_curves``.
        every_n_epochs: Refresh cadence in epochs (clamped to ``>= 1``).
    """

    def __init__(
        self, out_stem: Union[str, Path], every_n_epochs: int = 10
    ) -> None:
        super().__init__()
        self._out_stem = Path(out_stem)
        self._every = max(1, int(every_n_epochs))

    def _render(self, trainer: "pl.Trainer", *, force: bool = False) -> None:
        r"""Flush the CSV logger and rewrite the HTML curve (rank-0, non-fatal).

        Args:
            trainer: The active trainer (source of the logger's ``log_dir``).
            force: When ``True`` render regardless of the epoch cadence (used at
                :meth:`on_fit_end`).
        """
        if not getattr(trainer, "is_global_zero", True):
            return
        epoch = int(getattr(trainer, "current_epoch", 0))
        if not force and (epoch + 1) % self._every != 0:
            return

        experiment_logger = getattr(trainer, "logger", None)
        log_dir = getattr(experiment_logger, "log_dir", None)
        if experiment_logger is None or log_dir is None:
            return

        try:
            # Flush buffered rows so the latest epoch is on disk before we read it.
            save = getattr(experiment_logger, "save", None)
            if callable(save):
                save()
            from .visualize_v2 import plot_loss_curves_html

            plot_loss_curves_html(Path(log_dir) / "metrics.csv", self._out_stem)
        except Exception as exc:  # pragma: no cover - plotting must never break training
            logger.warning(
                "[LossPlotHtmlCallback] HTML render failed (non-fatal): {}", exc
            )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._render(trainer)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._render(trainer)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._render(trainer, force=True)
