"""SeqVAE-specific Lightning callbacks that plot validation diagnostics.

Three qualitative-plotting callbacks that render reconstructions, predictions and
latents for the SeqVAE family of models during validation:

* :class:`ReconstructionPlotCallback` -- a matplotlib PNG of target-vs-reconstruction
  for a handful of validation samples.
* :class:`PlottingCallBack` -- the four-panel analysis PDF (raw signals,
  reconstruction with uncertainty, prediction overlay, latent heatmap).
* :class:`PlottingAvgPredCallBack` -- the averaged post-warmup raw prediction PDF,
  plus latent, linear output and the scattering/phase feature heatmaps.

Why these are not in ``train/callbacks.py``
-------------------------------------------
``train/callbacks.py`` is the model-agnostic callback module for the shared training
framework: its callbacks read only ``trainer.callback_metrics`` and ``nn.Module``
APIs, so they work for any :class:`~lightning.pytorch.LightningModule`. These three
do not. They are welded to the SeqVAE model and its $4\\,\\mathrm{Hz}$ FHR/UP batch
contract:

* **Batch fields.** All three require ``fhr_st``, ``fhr_ph``, ``fhr_up_ph`` and
  ``fhr``; the two PDF callbacks additionally require ``up`` and ``guid``, and read
  them by *attribute* (``batch.fhr_st``), so a plain ``dict`` batch raises.
* **Forward signature.** ``ReconstructionPlotCallback`` calls the wrapped model with
  keywords -- ``model(y_st=, y_ph=, x_ph=)`` -- while both PDF callbacks call it
  positionally. A model with a different forward signature cannot be plotted.
* **Output keys.** ``mu_pr`` is required throughout; ``PlottingCallBack`` also
  requires ``z`` and ``logvar_pr``. The uncertainty band is
  $\\mu_{pr} \\pm k\\exp(\\tfrac{1}{2}\\mathrm{logvar}_{pr})$, with $k=2$ except in
  :class:`PlottingCallBack`, which draws $k=1$.
* **Model attributes.** ``PlottingAvgPredCallBack`` is the most coupled: it calls
  ``average_raw_prediction(mu_pr)`` on the wrapped model and reads
  ``decimation_factor`` and ``warmup_period`` off it.
* **Fixed sampling rate.** ``PlottingCallBack`` hardcodes a $4\\,\\mathrm{Hz}$ time
  axis (``np.arange(len(y_raw)) / 4.0``) and "FHR (bpm)" axis labels, so a compliant
  model at another sampling rate still gets mislabeled axes.

Keeping them alongside the agnostic callbacks made that agnosticism a convention
that nothing enforced. Housing them here makes it a property of the file.

Why ``utils/`` rather than one model package
--------------------------------------------
Their consumers span three *different* model families -- ``model/vae_teb_small``,
``model/lstm_cnn_vae_teb`` and ``model/vae_teb_prediction`` -- so any single model
home would force two of the three to reach sideways into a peer package. ``utils/``
is the existing precedent for cross-family SeqVAE plotting: :mod:`utils.plot_utils`
is already SeqVAE-specific and is already imported by all three families' testing
modules.

This does *not* generalise to every plotting callback. The single-consumer plotters
(``model/vae_teb_prediction/model/plotting_callback_lag_attn_v1.py`` and siblings)
correctly live next to the one model that uses them; only the shared ones belong here.

Dependency direction
--------------------
This module imports nothing outside ``utils/``: the rank-0 artifact seam it shares
with :mod:`train.callbacks` lives in :mod:`utils.mlflow_utils`, so the repo's layering
(``utils/`` <- ``train/`` <- ``model/``, one way) holds here with no exception.
``train/tests/test_layering.py`` enforces it.

Known limitations, carried over unchanged
-----------------------------------------
These classes were relocated verbatim; the following pre-existing defects moved with
them and are **not** endorsed -- fixing them is separate work:

* Both PDF callbacks call ``pl_module.train()`` unconditionally in their ``finally``
  block, so validating from an eval-mode module (e.g. ``trainer.validate()``) leaves
  the module in **train** mode. ``ReconstructionPlotCallback`` handles this correctly
  with a ``was_training`` flag.
* Both PDF callbacks wrap their whole forward in a bare ``except Exception`` that only
  emits a ``logger.error`` line, so plots stop appearing silently.
  ``ReconstructionPlotCallback`` is quieter still: it returns without logging anything
  when the batch fields are missing.
* All three run a rank-0-only forward pass while other ranks proceed; a collective
  inside the model's forward would deadlock.
* Both PDF callbacks write their artifacts to disk only -- unlike the agnostic
  callbacks, they never upload through :func:`train.callbacks.log_artifact_to_mlflow`.
* Both PDF callbacks mutate global ``plt.style`` / ``plt.rcParams`` without restoring
  them.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from lightning.pytorch.callbacks import Callback
from loguru import logger
from pathlib import Path
import numpy as np
import torch

# Mirrors utils/style.py: force the non-interactive backend at import, but tolerate a
# matplotlib-less environment. HAS_MPL is exported for symmetry with utils.style and is
# deliberately not branched on here -- these callbacks cannot function without
# matplotlib, so an absent backend surfaces as a NameError at plot time rather than a
# silent no-op.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from utils.mlflow_utils import log_artifact_to_mlflow

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger


def _resolve_validation_dataloader(trainer):
    """Return the first validation dataloader regardless of trainer setup."""
    if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
        dataloader = trainer.datamodule.val_dataloader()
    else:
        dataloader = trainer.val_dataloaders
    if isinstance(dataloader, list):
        return dataloader[0] if dataloader else None
    return dataloader


def _first_validation_batch(trainer):
    """Fetch the very first validation batch for qualitative callbacks."""
    dataloader = _resolve_validation_dataloader(trainer)
    if dataloader is None:
        return None
    iterator = iter(dataloader)
    return next(iterator, None)


class ReconstructionPlotCallback(Callback):
    """Plot reconstructions for a handful of validation samples."""

    def __init__(
        self,
        output_dir: Path | str,
        plot_frequency: int = 5,
        num_examples: int = 3,
        *,
        file_format: str = "png",
        mlflow_logger: "MLFlowLogger" | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir) / "reconstructions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_frequency = max(1, int(plot_frequency))
        self.num_examples = max(1, int(num_examples))
        file_format = file_format.lower()
        self.file_format = file_format
        self._mlflow_logger = mlflow_logger

    @staticmethod
    def _extract(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid_from_batch(guid_field, index: int = 0) -> str:
        if guid_field is None:
            return "unknown"
        if isinstance(guid_field, (list, tuple)):
            if not guid_field:
                return "unknown"
            value = guid_field[index % len(guid_field)]
            return str(value)
        if isinstance(guid_field, torch.Tensor):
            try:
                return str(guid_field[index].item())
            except Exception:  # noqa: BLE001
                return "unknown"
        return str(guid_field)

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_frequency != 0:
            return
        batch = _first_validation_batch(trainer)
        if batch is None:
            return
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        forward_module = getattr(pl_module, "model", pl_module)
        y_st = self._extract(batch, "fhr_st")
        y_ph = self._extract(batch, "fhr_ph")
        x_ph = self._extract(batch, "fhr_up_ph")
        target = self._extract(batch, "fhr")
        if not isinstance(y_st, torch.Tensor) or not isinstance(y_ph, torch.Tensor) or not isinstance(target, torch.Tensor):
            return
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            outputs = forward_module(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        if was_training:
            pl_module.train()
        mu_pr = outputs.get("mu_pr")
        logvar_pr = outputs.get("logvar_pr")
        if not isinstance(mu_pr, torch.Tensor):
            return
        count = min(self.num_examples, mu_pr.shape[0], target.shape[0])
        if count == 0:
            return
        target_np = target[:count].detach().cpu().numpy()
        recon_np = mu_pr[:count].detach().cpu().numpy()
        std_np = None
        if isinstance(logvar_pr, torch.Tensor):
            std_np = torch.exp(0.5 * logvar_pr[:count]).detach().cpu().numpy()
        fig, axes = plt.subplots(count, 1, figsize=(14, 3 * count), sharex=True)
        if count == 1:
            axes = [axes]
        for idx in range(count):
            axis = axes[idx]
            target_series = np.squeeze(target_np[idx])
            recon_series = np.squeeze(recon_np[idx])
            length = min(target_series.shape[-1], recon_series.shape[-1])
            t_axis = np.arange(length)
            axis.plot(t_axis, target_series[..., :length], label="target", color="tab:blue", linewidth=1.2)
            axis.plot(t_axis, recon_series[..., :length], label="reconstruction", color="tab:orange", linewidth=1.0)
            if std_np is not None:
                std_series = np.squeeze(std_np[idx])[..., :length]
                axis.fill_between(
                    t_axis,
                    recon_series[..., :length] - 2 * std_series,
                    recon_series[..., :length] + 2 * std_series,
                    color="tab:orange",
                    alpha=0.2,
                )
            axis.set_ylabel(f"Example {idx + 1}")
            axis.grid(True, alpha=0.3)
        axes[0].set_title("Validation reconstructions")
        axes[-1].set_xlabel("Time index")
        axes[0].legend(loc="upper right", frameon=False)
        fig.tight_layout()
        path = self.output_dir / f"reconstruction_epoch_{epoch:04d}.{self.file_format}"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved reconstruction plot to {path}")
        log_artifact_to_mlflow(self._mlflow_logger, path, trainer)


class PlottingCallBack(Callback):
    """Reproduce the original SeqVAE qualitative plotting callback."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, input_channel_num: int = 0):
        super().__init__()
        self.output_dir = Path(output_dir) / "analysis_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.input_channel_num = input_channel_num
        # Vestigial: never read. Retained by the verbatim relocation rather than
        # pruned, so the move stays a pure move.
        self._plotly_available = True

    @staticmethod
    def _get(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid(batch, index: int = 0) -> str:
        guid_field = None
        if isinstance(batch, dict):
            guid_field = batch.get("guid")
        elif hasattr(batch, "guid"):
            guid_field = getattr(batch, "guid")
        if guid_field is None:
            return "unknown"
        if isinstance(guid_field, (list, tuple)):
            if not guid_field:
                return "unknown"
            return str(guid_field[index % len(guid_field)])
        if isinstance(guid_field, torch.Tensor):
            try:
                return str(guid_field[index].item())
            except Exception:  # noqa: BLE001
                return "unknown"
        return str(guid_field)

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_epoch != 0:
            return

        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("PlottingCallBack: could not fetch validation batch")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        module = getattr(pl_module, "model", pl_module)

        pl_module.eval()
        try:
            with torch.no_grad():
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw = batch.fhr
                up_raw = batch.up
                outputs = module(y_st, y_ph, x_ph)
                latent_z = outputs["z"]
                mu_pr = outputs["mu_pr"]
                logvar_pr = outputs["logvar_pr"]
            guid = self._guid(batch)
            self._plot_results(
                y_raw,
                up_raw,
                mu_pr,
                logvar_pr,
                latent_z,
                epoch,
                guid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"PlottingCallBack failed: {exc}")
        finally:
            pl_module.train()

    def _plot_results(
        self,
        y_raw_normalized,
        up_raw_normalized,
        mu_pr,
        logvar_pr,
        latent_z,
        epoch: int,
        guid: str,
    ) -> None:
        batch_idx = 0
        y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
        up_raw = up_raw_normalized[batch_idx].detach().cpu().numpy()
        mu_samples = mu_pr[batch_idx].detach().cpu().numpy()
        logvar_samples = logvar_pr[batch_idx].detach().cpu().numpy()
        z_latent = latent_z[batch_idx].detach().cpu().numpy()
        time_axis = np.arange(0, len(y_raw)) / 4.0

        colors = {
            "fhr": "#055C9A",
            "up": "#0DD8A2",
            "gt": "#456882",
            "recon": "#BB3E00",
            "uncertainty": "#F7AD45",
            "samples": "#4BD605",
            "background": "#F9F3EF",
        }

        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.linewidth": 0.7,
                "axes.edgecolor": "#9E9D9D",
                "axes.facecolor": colors["background"],
                "grid.color": "#838383",
                "grid.linewidth": 0.4,
                "grid.alpha": 0.6,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 300,
            }
        )

        n_rows = 4
        fig, ax = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(20, n_rows * 3.5),
            gridspec_kw={"width_ratios": [80, 1]},
            constrained_layout=True,
        )
        for i in range(n_rows):
            ax[i, 0].grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
            ax[i, 0].grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
            ax[i, 0].minorticks_on()
            ax[i, 0].set_axisbelow(True)
            ax[i, 0].spines["top"].set_visible(False)
            ax[i, 0].spines["right"].set_visible(False)
            ax[i, 0].spines["left"].set_color("#A2B9A7")
            ax[i, 0].spines["bottom"].set_color("#A2B9A7")
            ax[i, 0].spines["left"].set_linewidth(0.7)
            ax[i, 0].spines["bottom"].set_linewidth(0.7)
            ax[i, 1].set_axis_off()

        ax[0, 0].plot(time_axis, y_raw, linewidth=1.2, color=colors["fhr"], label="FHR", alpha=0.85)
        ax[0, 0].plot(time_axis, up_raw, linewidth=1.2, color=colors["up"], label="UP", alpha=0.85)
        ax[0, 0].set_ylabel("Amplitude")
        ax[0, 0].set_title("Raw FHR and UP Signals", pad=12)
        ax[0, 0].legend(loc="upper right", framealpha=0.95)

        ax[1, 0].plot(time_axis, y_raw, linewidth=1.5, color=colors["gt"], label="Ground Truth", alpha=0.85, zorder=3)
        ax[1, 0].plot(time_axis, mu_samples, linewidth=1.5, color=colors["recon"], label="Reconstruction", alpha=0.85, zorder=2)
        std_dev = np.exp(0.5 * logvar_samples)
        ax[1, 0].fill_between(
            time_axis,
            mu_samples - std_dev,
            mu_samples + std_dev,
            alpha=0.3,
            color=colors["uncertainty"],
            label="Uncertainty (±1σ)",
            zorder=1,
        )
        ax[1, 0].set_ylabel("FHR (bpm)")
        ax[1, 0].set_title("FHR Reconstruction with Uncertainty", pad=12)
        ax[1, 0].legend(loc="upper right", framealpha=0.95)

        ax[2, 0].plot(time_axis, y_raw, linewidth=1.5, color=colors["gt"], label="Ground Truth", alpha=0.85, zorder=2)
        ax[2, 0].plot(time_axis, mu_samples, linewidth=1.5, color=colors["samples"], label="Model Prediction", alpha=0.85, zorder=1)
        ax[2, 0].set_ylabel("FHR (bpm)")
        ax[2, 0].set_title("FHR vs Model Reconstructions", pad=12)
        ax[2, 0].legend(loc="upper right", framealpha=0.95)

        imgplot = ax[3, 0].imshow(z_latent.T, aspect="auto", cmap="bwr", origin="lower")
        ax[3, 0].set_ylabel("Latent Dimensions")
        ax[3, 0].set_xlabel("Time Steps")
        ax[3, 0].set_title("Latent Space Representation", pad=12)
        ax[3, 1].set_axis_on()
        cbar = fig.colorbar(imgplot, cax=ax[3, 1])
        cbar.ax.tick_params(labelsize=10, colors="#666666")
        cbar.set_label("Activation", fontweight="normal", fontsize=11, color="#666666")

        fig.suptitle(f"Model Performance Analysis — Epoch {epoch} | guid: {guid}", fontsize=14, y=0.97, color="#456882")
        save_path = self.output_dir / f"model_results_epoch_{epoch:04d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", orientation="landscape", dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)


class PlottingAvgPredCallBack(Callback):
    """Plot averaged raw prediction (post-warmup) versus ground truth with uncertainty, latent, and UP."""

    def __init__(self, output_dir: Path | str, plot_every_epoch: int = 5, input_channel_num: int = 0):
        super().__init__()
        self.output_dir = Path(output_dir) / "analysis_plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_every_epoch = max(1, int(plot_every_epoch))
        self.input_channel_num = input_channel_num
        # Vestigial: never read. See PlottingCallBack.__init__.
        self._plotly_available = True

    @staticmethod
    def _get(batch, name):
        if isinstance(batch, dict):
            return batch.get(name)
        return getattr(batch, name, None)

    @staticmethod
    def _guid(batch, index: int = 0) -> str:
        guid_field = None
        if isinstance(batch, dict):
            guid_field = batch.get("guid")
        elif hasattr(batch, "guid"):
            guid_field = getattr(batch, "guid")
        if guid_field is None:
            return "unknown"
        if isinstance(guid_field, (list, tuple)):
            if not guid_field:
                return "unknown"
            return str(guid_field[index % len(guid_field)])
        if isinstance(guid_field, torch.Tensor):
            try:
                return str(guid_field[index].item())
            except Exception:  # noqa: BLE001
                return "unknown"
        return str(guid_field)

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_epoch != 0:
            return

        batch = _first_validation_batch(trainer)
        if batch is None:
            logger.warning("PlottingCallBack: could not fetch validation batch")
            return

        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
        module = getattr(pl_module, "model", pl_module)

        pl_module.eval()
        try:
            with torch.no_grad():
                y_st, y_ph, x_ph = batch.fhr_st, batch.fhr_ph, batch.fhr_up_ph
                y_raw = batch.fhr
                up_raw = getattr(batch, "up", None)
                outputs = module(y_st, y_ph, x_ph)
                mu_pr = outputs["mu_pr"]
                logvar_pr = outputs.get("logvar_pr")
                avg_pred = module.average_raw_prediction(mu_pr)
                avg_std = None
                if isinstance(logvar_pr, torch.Tensor):
                    std_pr = torch.exp(0.5 * logvar_pr)
                    avg_std = self._average_segments(
                        std_pr,
                        decimation_factor=getattr(module, "decimation_factor", 16),
                        warmup=getattr(module, "warmup_period", 0),
                        raw_len=y_raw.shape[-1],
                    )
                linear_output = outputs.get("linear_output")
                latent_z = outputs.get("z")
                st_feats = y_st
                ph_feats = y_ph
                cross_feats = x_ph
            guid = self._guid(batch)
            self._plot_results(
                y_raw,
                up_raw,
                avg_pred,
                avg_std,
                linear_output,
                latent_z,
                st_feats,
                ph_feats,
                cross_feats,
                epoch,
                guid,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"PlottingCallBack failed: {exc}")
        finally:
            pl_module.train()

    @staticmethod
    def _average_segments(segments: torch.Tensor, decimation_factor: int, warmup: int, raw_len: int) -> torch.Tensor:
        """Average per-step horizon segments onto the raw axis (mirrors average_raw_prediction)."""
        if segments.dim() != 3:
            return torch.full((segments.size(0), raw_len), float("nan"), device=segments.device, dtype=segments.dtype)
        B, T, H = segments.shape
        s = int(decimation_factor)
        warmup = int(warmup)
        device = segments.device
        dtype = segments.dtype

        avg = torch.full((B, raw_len), float("nan"), device=device, dtype=dtype)
        max_valid_t = max(0, min(T, (raw_len - H) // s + 1))
        start_t = min(warmup, max_valid_t)
        if start_t >= max_valid_t:
            return avg
        valid_steps = torch.arange(start_t, max_valid_t, device=device)
        start_idx = valid_steps[:, None] * s
        h_idx = torch.arange(H, device=device)[None, :]
        idx = start_idx + h_idx  # (T_valid, H)
        mask = (idx < raw_len).to(dtype)
        idx_clamped = idx.clamp(max=raw_len - 1)

        pred_sum = torch.zeros((B, raw_len), device=device, dtype=dtype)
        count = torch.zeros((B, raw_len), device=device, dtype=dtype)

        flat_idx = idx_clamped.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        flat_seg = segments[:, valid_steps, :].reshape(B, -1)
        flat_mask = mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)

        pred_sum.scatter_add_(1, flat_idx, flat_seg * flat_mask)
        count.scatter_add_(1, flat_idx, flat_mask)

        avg = pred_sum / count.clamp_min(1.0)
        avg = avg.masked_fill(count == 0, float("nan"))
        return avg

    def _plot_results(
        self,
        y_raw_normalized,
        up_raw_normalized,
        avg_pred,
        avg_std,
        linear_output,
        latent_z,
        st_feats,
        ph_feats,
        cross_feats,
        epoch: int,
        guid: str,
    ) -> None:
        batch_idx = 0
        y_raw = y_raw_normalized[batch_idx].detach().cpu().numpy()
        up_raw = None
        if isinstance(up_raw_normalized, torch.Tensor):
            up_raw = up_raw_normalized[batch_idx].detach().cpu().numpy()
        avg_pred_np = avg_pred[batch_idx].detach().cpu().numpy()
        avg_std_np = None
        if isinstance(avg_std, torch.Tensor):
            avg_std_np = avg_std[batch_idx].detach().cpu().numpy()
        linear_np = None
        if isinstance(linear_output, torch.Tensor):
            linear_np = linear_output[batch_idx].detach().cpu().numpy()
        latent_np = None
        if isinstance(latent_z, torch.Tensor):
            latent_np = latent_z[batch_idx].detach().cpu().numpy()
        st_np = None
        if isinstance(st_feats, torch.Tensor):
            st_np = st_feats[batch_idx].detach().cpu().numpy()
        ph_np = None
        if isinstance(ph_feats, torch.Tensor):
            ph_np = ph_feats[batch_idx].detach().cpu().numpy()
        cross_np = None
        if isinstance(cross_feats, torch.Tensor):
            cross_np = cross_feats[batch_idx].detach().cpu().numpy()
        time_axis = np.arange(0, len(y_raw)) / 4.0

        colors = {
            "fhr": "#055C9A",
            "avg": "#BB3E00",
            "up": "#0DD8A2",
            "uncertainty": "#F7AD45",
            "background": "#F9F3EF",
        }

        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.linewidth": 0.7,
                "axes.edgecolor": "#9E9D9D",
                "axes.facecolor": colors["background"],
                "grid.color": "#838383",
                "grid.linewidth": 0.4,
                "grid.alpha": 0.6,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 300,
            }
        )

        extra_rows = int(st_np is not None) + int(ph_np is not None) + int(cross_np is not None)
        n_rows = 4 + extra_rows
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.5 * n_rows), sharex=False)

        ax0 = axes[0]
        ax0.grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
        ax0.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
        ax0.minorticks_on()
        ax0.plot(time_axis, y_raw, linewidth=1.2, color=colors["fhr"], label="FHR", alpha=0.85)
        if up_raw is not None:
            ax0.plot(time_axis, up_raw, linewidth=1.0, color=colors["up"], label="UP", alpha=0.8)
        ax0.set_ylabel("Amplitude")
        ax0.set_title("Raw FHR/UP")
        ax0.legend(loc="upper right", framealpha=0.95)

        ax1 = axes[1]
        ax1.grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color="#D2C1B6")
        ax1.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color="#D2C1B6")
        ax1.minorticks_on()
        ax1.plot(time_axis, y_raw, linewidth=1.2, color=colors["fhr"], label="Ground Truth", alpha=0.85)
        ax1.plot(time_axis, avg_pred_np, linewidth=1.0, color=colors["avg"], label="Avg Prediction", alpha=0.9)
        if avg_std_np is not None:
            ax1.fill_between(
                time_axis,
                avg_pred_np - 2 * avg_std_np,
                avg_pred_np + 2 * avg_std_np,
                color=colors["uncertainty"],
                alpha=0.25,
                label="Uncertainty (±2σ)",
            )
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_title("Average Raw Prediction (post-warmup)")
        ax1.legend(loc="upper right", framealpha=0.95)

        ax2 = axes[2]
        if latent_np is not None:
            im = ax2.imshow(latent_np.T, aspect="auto", cmap="bwr", origin="lower")
            ax2.set_ylabel("Latent dims")
            ax2.set_xlabel("Time steps (decimated)")
            ax2.set_title("Latent Representation")
            fig.colorbar(im, ax=ax2, fraction=0.015, pad=0.02)
        else:
            ax2.set_visible(False)

        ax3 = axes[3]
        if linear_np is not None:
            im2 = ax3.imshow(linear_np.T, aspect="auto", cmap="viridis", origin="lower")
            ax3.set_ylabel("Linear Output Ch")
            ax3.set_xlabel("Time steps (decimated)")
            ax3.set_title("Linear Output (B,T,40)")
            fig.colorbar(im2, ax=ax3, fraction=0.015, pad=0.02)
        else:
            ax3.set_visible(False)

        idx = 4
        if st_np is not None and idx < len(axes):
            ax_st = axes[idx]
            im_st = ax_st.imshow(st_np.T, aspect="auto", cmap="bwr", origin="lower")
            ax_st.set_ylabel("ST ch")
            ax_st.set_xlabel("Time steps (decimated)")
            ax_st.set_title("FHR Scattering Transform")
            fig.colorbar(im_st, ax=ax_st, fraction=0.015, pad=0.02)
            idx += 1
        if ph_np is not None and idx < len(axes):
            ax_ph = axes[idx]
            im_ph = ax_ph.imshow(ph_np.T, aspect="auto", cmap="bwr", origin="lower")
            ax_ph.set_ylabel("Phase ch")
            ax_ph.set_xlabel("Time steps (decimated)")
            ax_ph.set_title("FHR Phase Harmonics")
            fig.colorbar(im_ph, ax=ax_ph, fraction=0.015, pad=0.02)
            idx += 1
        if cross_np is not None and idx < len(axes):
            ax_cross = axes[idx]
            im_cross = ax_cross.imshow(cross_np.T, aspect="auto", cmap="bwr", origin="lower")
            ax_cross.set_ylabel("Cross ch")
            ax_cross.set_xlabel("Time steps (decimated)")
            ax_cross.set_title("UP+FHR Cross Phase")
            fig.colorbar(im_cross, ax=ax_cross, fraction=0.015, pad=0.02)

        fig.suptitle(f"Avg Prediction — Epoch {epoch} | guid: {guid}", fontsize=12, y=0.98, color="#456882")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_path = self.output_dir / f"avg_prediction_epoch_{epoch:04d}.pdf"
        fig.savefig(save_path, bbox_inches="tight", orientation="landscape", dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)


__all__ = [
    "ReconstructionPlotCallback",
    "PlottingCallBack",
    "PlottingAvgPredCallBack",
]
