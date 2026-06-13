r"""Multi-GPU (DDP) training entry point for the ``G1_mix`` final model.

The single-GPU :mod:`train_minimal` loop is the throughput bottleneck for the
**one large heterogeneous** ``G1_mix`` run (a pool mixing informative-channel
counts $M$, block transfer entropies $\mathrm{TE}$ and lag bands; see
:mod:`mixed_dataset` / :mod:`mixed_eval`). This script trains that single model
**data-parallel across all GPUs** using PyTorch Lightning DDP, while preserving
the synthetic pipeline's semantics end-to-end:

* the cached ``.npz`` pool via :class:`datamodule_synth.SyntheticTEDataModule`,
* the synthetic loss settings (``likelihood`` / ``sigma_obs`` / ``free_bits`` /
  ``lambda_full`` / ``lambda_base`` / ``kld_beta``) threaded through
  :class:`pl_module_synth.SyntheticSeqVaeLagAttnPl`, and
* a ``final.ckpt`` written in the **exact** :func:`train_minimal.save_checkpoint`
  format, so :mod:`mixed_eval` / :mod:`evaluate_te` load it unchanged.

Scope (locked): DDP is for this single run only. The $\beta\times M\times
\mathrm{TE}$ sweeps stay on the task-parallel :mod:`gpu_pool` (one independent
cell per GPU), which already saturates the box and is faster than DDP for many
tiny runs.

Multi-GPU correctness highlights:

* **Latent stats.** The model's ``mu_post_running_*`` EMA buffers diverge across
  ranks during training. After ``fit`` we call
  :meth:`SeqVaeLagAttnV1.fit_latent_stats` on **every** rank (it ``all_reduce``s
  exact sums) over a **sharded** loader so the aggregated count is truthful.
* **LR scaling.** The locked policy keeps the per-GPU batch (so the global batch
  is ``batch_size * n_gpus``) and scales ``lr`` linearly with a short warmup.
* **Checkpoint bridge.** ``final.ckpt`` (post latent-stats fit) and ``best.ckpt``
  (lowest ``val/total_loss``, converted from the Lightning best snapshot) store
  the bare ``SeqVaeLagAttnV1`` state dict (unprefixed) + ``model_kwargs`` so the
  strict loader ``load_checkpoint_strict`` matches exactly. Both are also written
  **every validation epoch during training** (rank-0
  :class:`_SyntheticCheckpointCallback`, ``latent_stats_fitted=False``) so a run
  stopped before the last epoch (Ctrl-C on a rising val loss, early stop, crash)
  still leaves consumable checkpoints; the post-fit export then finalizes them.
* **Early stopping.** Opt-in via ``ddp.early_stopping`` (or ``--early-stop-patience``)
  on the ``sync_dist``-reduced ``val/total_loss`` -- a graceful stop still runs the
  post-fit latent-stats fit + checkpoint finalization.
* **Monitoring.** A rank-0 callback emits the single-GPU-equivalent
  ``metrics.csv`` + ``training_curves.{pdf,png}`` / ``loss_plot_epoch.html``
  figures every ``plotting.plot_every`` epochs.

Run modes (Decision V2-D8): a CLI and an edit-and-run ``RUN_CONFIG``,
auto-detected from whether any command-line argument is present::

    # Linux 8-GPU box (Lightning spawns its own DDP workers):
    python -m ...synthetic.train_ddp --tag G1_mix_base --run-tag G1_mix_base --devices 8
    # Windows single-GPU dev:
    python -m ...synthetic.train_ddp --tag G1_mix_base --devices 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as pl
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from model.vae_teb_prediction.model.model_experiment.synthetic.datamodule_synth import (
    SyntheticTEDataModule,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    build_u_stream,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.pl_module_synth import (
    SyntheticSeqVaeLagAttnPl,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    _EVAL_KEYS,
    _FIELDNAMES,
    _OVERRIDE_MAP,
    _TRAIN_KEYS,
    _TRAIN_METRIC_KEYS,
    _assemble_row,
    _refresh_training_curves,
    append_csv_row,
    apply_path_overrides,
    build_model,
    load_config,
    resolve_active_benchmark,
    resolve_user_path,
    save_checkpoint,
    set_seed,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from train.graph_models_utils import load_checkpoint_strict

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_BENCHMARK = "G1_mix"

# Defaults for the optional top-level ``ddp:`` config block (only this script
# reads it). Every key is overridable from the YAML / CLI.
_DDP_DEFAULTS: Dict[str, Any] = {
    "devices": 8,
    "strategy": "ddp",
    "precision": "32-true",
    "sync_batchnorm": True,
    "lr_scaling": True,
    "warmup_epochs": 5,
    "num_sanity_val_steps": 0,
    "gradient_clip_val": 0.5,
    "gradient_clip_algorithm": "norm",
    # Optional early stopping (off by default -> unchanged 100-epoch behaviour).
    # Only active when a validation split exists; see ``_build_trainer``.
    "early_stopping": {
        "enabled": False,
        "monitor": "val/total_loss",
        "mode": "min",
        "patience": 15,
        "min_delta": 0.0,
    },
}


# =============================================================================
# Helpers
# =============================================================================
def _parse_devices(spec: Union[int, str, List[int]]) -> Tuple[Union[int, List[int]], int]:
    """Parse a ``devices`` spec into ``(lightning_devices, n_gpus)``.

    Accepts an int (use the first N GPUs), a comma list ``"0,1,2"`` (explicit
    indices), or an actual list. Returns the value to hand to ``pl.Trainer`` and
    the GPU count used for LR scaling.
    """
    if isinstance(spec, int):
        return spec, int(spec)
    if isinstance(spec, (list, tuple)):
        idx = [int(x) for x in spec]
        return idx, len(idx)
    text = str(spec).strip()
    if "," in text:
        idx = [int(x) for x in text.split(",") if x != ""]
        return idx, len(idx)
    return int(text), int(text)


def _resolve_loss_settings(loss_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the synthetic loss settings, mirroring ``train_minimal.train``.

    Parses ``sigma_obs`` (positive float or the literal ``'learned'``) exactly
    as the single-GPU loop so DDP training is bit-for-bit comparable.
    """
    beta = float(loss_cfg["kld_beta"])
    lambda_full = float(loss_cfg["lambda_full"])
    lambda_base = float(loss_cfg["lambda_base"])
    likelihood = str(loss_cfg.get("likelihood", "mse"))
    sigma_obs_raw = loss_cfg.get("sigma_obs", 1.0)
    if isinstance(sigma_obs_raw, str) and sigma_obs_raw != "learned":
        try:
            sigma_obs: "float | str" = float(sigma_obs_raw)
        except ValueError as exc:
            raise ValueError(
                f"loss.sigma_obs must be a positive float or 'learned', "
                f"got {sigma_obs_raw!r}"
            ) from exc
    else:
        sigma_obs = sigma_obs_raw if isinstance(sigma_obs_raw, str) else float(sigma_obs_raw)
    free_bits = float(loss_cfg.get("free_bits", 0.0))
    return {
        "beta": beta, "lambda_full": lambda_full, "lambda_base": lambda_base,
        "likelihood": likelihood, "sigma_obs": sigma_obs, "free_bits": free_bits,
    }


def _env_rank_zero() -> bool:
    """Return ``True`` on the global-rank-0 process, from the environment.

    Used for output that must happen **before** ``trainer.fit`` (the banner and
    the ``config_used.yaml`` dump). ``trainer.is_global_zero`` is unreliable
    pre-fit under the DDP subprocess launcher -- every freshly re-executed rank
    reports ``global_rank == 0`` until the strategy initialises inside ``fit``,
    so all ranks would race on the same write. The launcher sets ``LOCAL_RANK``
    / ``RANK`` in each child, so the environment is authoritative here.
    """
    for var in ("LOCAL_RANK", "RANK", "SLURM_PROCID", "NODE_RANK"):
        value = os.environ.get(var)
        if value is not None:
            return int(value) == 0
    return True


def _as_float(value: Any) -> float:
    """Coerce a (possibly tensor) metric to a Python float."""
    if torch.is_tensor(value):
        return float(value.detach().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _augment_loss_settings(loss_settings: Dict[str, Any]) -> Dict[str, Any]:
    r"""Return a copy of ``loss_settings`` with ``kld_beta`` mirroring ``beta``.

    ``train_minimal`` persists only ``beta`` in the checkpoint, but
    :func:`mixed_eval.per_cell_lag_recovery` reads ``kld_beta``; duplicating the
    key keeps the LOLO ablation on the trained $\beta$ rather than the ``0.001``
    default fallback. Used by both :func:`_export_checkpoints` and
    :class:`_SyntheticCheckpointCallback` so every persisted ``loss_settings`` is
    identical regardless of when (mid-training vs end-of-fit) it is written.

    Args:
        loss_settings: The resolved loss settings from :func:`_resolve_loss_settings`.

    Returns:
        A shallow copy with ``kld_beta == beta``.
    """
    augmented = dict(loss_settings)
    augmented["kld_beta"] = loss_settings["beta"]
    return augmented


def _apply_train_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the shared flat overrides (``_OVERRIDE_MAP`` keys + ``run_tag``).

    DDP-specific keys (``devices``) are consumed by the caller, not here, so we
    only route keys this config understands and ignore the rest.
    """
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "run_tag":
            config["run_tag"] = value
        elif key in _OVERRIDE_MAP:
            section, field = _OVERRIDE_MAP[key]
            config.setdefault(section, {})[field] = value
    return config


# =============================================================================
# Loss-curve plotting (rank-0 only)
# =============================================================================
class _SyntheticCurveCallback(pl.Callback):
    r"""Rank-0 callback that mirrors the single-GPU loss-curve pipeline.

    Each epoch it appends one :data:`train_minimal._FIELDNAMES`-schema row to
    ``metrics.csv`` -- built from the epoch-level, distributed-synced metrics in
    ``trainer.callback_metrics`` (validation metrics are reduced across ranks
    via ``sync_dist=True`` in :class:`train.pl_model_base.LightningModelBase`) --
    and, every ``plot_every`` epochs plus once at fit end, re-renders
    ``training_curves.{pdf,png}`` and ``loss_plot_epoch.html`` via
    :func:`train_minimal._refresh_training_curves`. The figures are therefore
    identical in look and location to a single-GPU :mod:`train_minimal` run.

    DDP-safety: every disk write / render happens **only on global rank 0** and
    reads already-reduced metrics, so no collective op runs inside the rank-0
    branch (one would otherwise deadlock the other ranks). The render path is
    wrapped in ``try/except`` inside ``_refresh_training_curves``, so a plotting
    failure never aborts training.

    Three columns have no DDP source and fall back gracefully (the renderers
    tolerate NaN): ``train_grad_norm`` (gradient clipping is ``Trainer``-managed),
    ``nan_skips`` (taken from ``train/spike_skips_total`` when present, else 0),
    and ``epoch_seconds`` (timed by this callback).

    Args:
        csv_path: Destination ``metrics.csv``; figures are written next to it.
        run_tag: Run label used in figure titles.
        plotting_cfg: The ``plotting`` config block (carries the ``enabled``
            toggle that ``_refresh_training_curves`` honours).
        plot_every: Render cadence in epochs; ``0`` renders only at fit end.
        has_val: Whether a validation loader exists. When ``True`` the row is
            written at ``on_validation_epoch_end`` (so val metrics are present);
            when ``False`` it is written at ``on_train_epoch_end``.
    """

    def __init__(
        self,
        *,
        csv_path: Path,
        run_tag: str,
        plotting_cfg: Dict[str, Any],
        plot_every: int,
        has_val: bool,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.run_tag = run_tag
        self.plotting_cfg = plotting_cfg
        self.plot_every = int(plot_every)
        self.has_val = bool(has_val)
        self._epoch_t0: Optional[float] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Stamp the epoch start time for the ``epoch_seconds`` column."""
        self._epoch_t0 = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Write the metrics row when there is no validation loop."""
        if not self.has_val:
            self._write_and_maybe_plot(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Write the metrics row after validation (the common path)."""
        if self.has_val:
            self._write_and_maybe_plot(trainer)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Final render -- covers ``epochs < plot_every`` (cf. train_minimal)."""
        if not trainer.is_global_zero:
            return
        _refresh_training_curves(self.csv_path, self.run_tag, self.plotting_cfg)

    def _write_and_maybe_plot(self, trainer: pl.Trainer) -> None:
        """Append one CSV row on rank 0 and refresh figures on schedule."""
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return
        cm = trainer.callback_metrics
        train_m: Dict[str, float] = {
            k: _as_float(cm.get(f"train/{k}", float("nan"))) for k in _TRAIN_KEYS
        }
        train_m["grad_norm"] = float("nan")  # clipping is Trainer-managed
        train_m["nan_skips"] = int(
            _as_float(cm.get("train/spike_skips_total", 0.0))
        )
        val_m: Dict[str, float] = {
            k: _as_float(cm.get(f"val/{k}", float("nan"))) for k in _EVAL_KEYS
        }
        epoch = int(trainer.current_epoch) + 1  # 1-based, matching train_minimal
        lr = _as_float(cm.get("lr", float("nan")))
        dt = time.time() - self._epoch_t0 if self._epoch_t0 is not None else float("nan")
        append_csv_row(
            self.csv_path, _assemble_row(epoch, train_m, val_m, lr, dt), _FIELDNAMES
        )
        if self.plot_every > 0 and epoch % self.plot_every == 0:
            _refresh_training_curves(self.csv_path, self.run_tag, self.plotting_cfg)


# =============================================================================
# Synthetic-format checkpointing (rank-0, written DURING training)
# =============================================================================
class _SyntheticCheckpointCallback(pl.Callback):
    r"""Rank-0 callback that writes the synthetic ``best.ckpt`` / ``final.ckpt``
    **every validation epoch**, so an interrupted run is always evaluable.

    The end-of-fit :func:`_export_checkpoints` only runs if ``trainer.fit``
    returns normally. A hard interruption (``Ctrl-C`` because the validation loss
    is climbing, a crash, or a killed job) skips it entirely, leaving only the
    Lightning ``lightning_best-*.ckpt`` -- which carries no ``model_kwargs`` and
    cannot be loaded by :mod:`mixed_eval` / :mod:`run_pipeline_tests`. This
    callback closes that gap by persisting both consumable artifacts as training
    progresses:

    * ``final.ckpt`` -- the *latest* model, rewritten every epoch.
    * ``best.ckpt`` -- the lowest ``val/total_loss`` snapshot so far.

    Both are written via :func:`train_minimal.save_checkpoint` (byte-identical to
    the end-of-fit format) from ``pl_module.orig_model``. In DDP the model
    parameters are gradient-synced across ranks, so rank-0's copy is
    authoritative; only the EMA ``mu_post_running_*`` buffers are rank-local and
    un-fitted, hence ``latent_stats_fitted=False`` (matching the single-GPU
    ``best.ckpt`` semantics -- :func:`mixed_eval`'s $\bar K$ reads the encoder
    outputs, not those buffers). On graceful completion the post-fit
    :func:`_export_checkpoints` **overwrites** both with the finalized versions
    (latent-fitted ``final.ckpt``; ``best.ckpt`` re-derived from the Lightning
    best across all epochs), so a completed run's artifacts are unchanged.

    DDP-safety: every write happens **only on global rank 0** and reads the
    already-reduced ``trainer.callback_metrics`` (``val/total_loss`` is logged
    with ``sync_dist=True``), so no collective op runs in the rank-0 branch.

    Args:
        results_dir: Run directory; ``best.ckpt`` / ``final.ckpt`` land here.
        model_kwargs: Exact ``SeqVaeLagAttnV1`` constructor kwargs (stored so the
            strict loader can rebuild the architecture).
        config: The effective (post-override) config.
        datamodule: The :class:`SyntheticTEDataModule`; its ``data_meta`` (the
            dataset ``meta.json``) is read **lazily** at write time, because the
            datamodule only populates it in ``setup`` (called inside
            ``trainer.fit``) -- a snapshot taken at construction would be empty.
        loss_settings: The augmented loss settings (must already carry
            ``kld_beta``; see :func:`_augment_loss_settings`).
        epochs: Configured ``max_epochs`` (used as the epoch fallback).
        has_val: Whether a validation loader exists. When ``True`` the write
            happens at ``on_validation_epoch_end``; otherwise at
            ``on_train_epoch_end`` and ``best.ckpt`` mirrors ``final.ckpt``.
        monitor: Metric key tracked for the best snapshot.
    """

    def __init__(
        self,
        *,
        results_dir: Path,
        model_kwargs: Dict[str, Any],
        config: Dict[str, Any],
        datamodule: Any,
        loss_settings: Dict[str, Any],
        epochs: int,
        has_val: bool,
        monitor: str = "val/total_loss",
    ) -> None:
        self.results_dir = Path(results_dir)
        self.model_kwargs = model_kwargs
        self.config = config
        self.datamodule = datamodule
        self.loss_settings = loss_settings
        self.epochs = int(epochs)
        self.has_val = bool(has_val)
        self.monitor = monitor
        self._best = float("inf")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Write checkpoints when there is no validation loop."""
        if not self.has_val:
            self._write(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Write checkpoints after validation (the common path)."""
        if self.has_val:
            self._write(trainer, pl_module)

    def _write(self, trainer: pl.Trainer, pl_module: Any) -> None:
        """Persist ``final.ckpt`` (always) and ``best.ckpt`` (on improvement)."""
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return
        cm = trainer.callback_metrics
        train_metrics = {
            k: _as_float(cm[f"train/{k}"])
            for k in _TRAIN_METRIC_KEYS
            if f"train/{k}" in cm
        }
        cur = _as_float(cm.get(self.monitor, float("nan")))
        epoch = int(trainer.current_epoch) + 1  # 1-based, matching train_minimal
        data_meta = dict(getattr(self.datamodule, "data_meta", {}) or {})

        save_checkpoint(
            self.results_dir / "final.ckpt",
            model=pl_module.orig_model,
            model_kwargs=self.model_kwargs,
            config=self.config,
            data_meta=data_meta,
            epoch=epoch,
            val_loss=cur,
            train_metrics=train_metrics,
            loss_settings=self.loss_settings,
            latent_stats_fitted=False,
        )

        # Best snapshot: lowest val/total_loss so far. With no validation split
        # there is no monitor, so best.ckpt simply mirrors final.ckpt. The
        # ``cur == cur`` guard rejects NaN (NaN compares unequal to itself).
        improved = (not self.has_val) or (cur == cur and cur < self._best)
        if improved:
            if self.has_val and cur == cur:
                self._best = cur
            save_checkpoint(
                self.results_dir / "best.ckpt",
                model=pl_module.orig_model,
                model_kwargs=self.model_kwargs,
                config=self.config,
                data_meta=data_meta,
                epoch=epoch,
                val_loss=cur,
                train_metrics=train_metrics,
                loss_settings=self.loss_settings,
                latent_stats_fitted=False,
            )


# =============================================================================
# Trainer
# =============================================================================
def _build_trainer(
    *,
    ddp_cfg: Dict[str, Any],
    results_dir: Path,
    epochs: int,
    devices: Union[int, List[int]],
    n_gpus: int,
    has_val: bool,
    run_tag: str,
    plotting_cfg: Dict[str, Any],
    extra_callbacks: Optional[List[Any]] = None,
) -> pl.Trainer:
    """Construct the ``pl.Trainer``, mirroring ``trainer_lag_attn_v1``.

    Uses plain ``ddp`` for >1 GPU -- the forced ``G1_mix`` benchmark trains with
    ``gaussian_nll`` + ``sigma_obs='learned'`` so both logvar heads receive
    gradients (no unused parameters),
    ``use_distributed_sampler=True`` so the DataModule's plain loaders are
    sharded automatically, and a ``CSVLogger`` + ``ModelCheckpoint`` for
    monitoring. The synthetic-format checkpoints are written separately by
    :func:`_export_checkpoints` (the Lightning ``ModelCheckpoint`` file lives in
    ``lightning_ckpts/`` and is **not** consumed by ``mixed_eval``; it is the
    snapshot of the best-val weights that ``_export_checkpoints`` converts into
    the synthetic-format ``best.ckpt``). When ``plotting.enabled`` a rank-0
    :class:`_SyntheticCurveCallback` emits the single-GPU-equivalent
    ``metrics.csv`` + loss-curve figures.

    Args:
        extra_callbacks: Additional callbacks appended after the built-in ones
            (e.g. the :class:`_SyntheticCheckpointCallback` that writes the
            consumable ``best.ckpt`` / ``final.ckpt`` during training).
    """
    callbacks: List[Any] = [LearningRateMonitor(logging_interval="epoch")]
    if has_val:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(results_dir / "lightning_ckpts"),
                monitor="val/total_loss",
                mode="min",
                save_top_k=1,
                filename="lightning_best-{epoch:02d}",
            )
        )
    es_cfg = ddp_cfg.get("early_stopping") or {}
    if has_val and bool(es_cfg.get("enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(es_cfg.get("monitor", "val/total_loss")),
                mode=str(es_cfg.get("mode", "min")),
                patience=int(es_cfg.get("patience", 15)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )
        )
    if plotting_cfg.get("enabled", True):
        callbacks.append(
            _SyntheticCurveCallback(
                csv_path=results_dir / "metrics.csv",
                run_tag=run_tag,
                plotting_cfg=plotting_cfg,
                plot_every=int(plotting_cfg.get("plot_every", 10)),
                has_val=has_val,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer_kwargs: Dict[str, Any] = {
        "max_epochs": epochs,
        "callbacks": callbacks,
        "default_root_dir": str(results_dir),
        "precision": ddp_cfg["precision"],
        "deterministic": False,
        "benchmark": True,
        "gradient_clip_val": ddp_cfg["gradient_clip_val"],
        "gradient_clip_algorithm": ddp_cfg["gradient_clip_algorithm"],
        "enable_checkpointing": has_val,
        "log_every_n_steps": 1,
        "num_sanity_val_steps": int(ddp_cfg["num_sanity_val_steps"]),
        "use_distributed_sampler": True,
        "sync_batchnorm": bool(ddp_cfg["sync_batchnorm"]) and n_gpus > 1,
        "enable_progress_bar": True,
        "logger": CSVLogger(save_dir=str(results_dir), name="logs"),
    }

    if torch.cuda.is_available():
        strategy = str(ddp_cfg["strategy"]) if n_gpus > 1 else "auto"
        trainer_kwargs.update(
            {"accelerator": "gpu", "devices": devices, "strategy": strategy}
        )
    else:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1})
    return pl.Trainer(**trainer_kwargs)


def _export_checkpoints(
    *,
    trainer: pl.Trainer,
    pl_model: SyntheticSeqVaeLagAttnPl,
    model_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    data_meta: Dict[str, Any],
    loss_settings: Dict[str, Any],
    results_dir: Path,
    epochs: int,
    n_stats: int,
) -> None:
    """Write ``final.ckpt`` and ``best.ckpt`` in the synthetic format (rank 0).

    The bare ``SeqVaeLagAttnV1`` (``pl_model.orig_model``) is checkpointed so the
    state-dict keys are unprefixed and align exactly with the strict loader. The
    ``loss_settings`` carries **both** ``beta`` and ``kld_beta`` (and the
    lambda / likelihood / sigma_obs / free_bits knobs): ``train_minimal`` stores
    only ``beta`` but ``mixed_eval.per_cell_lag_recovery`` reads ``kld_beta``, so
    the duplicate keeps the LOLO ablation on the trained $\\beta$ rather than the
    ``0.001`` default fallback.

    Two artifacts are produced, matching the single-GPU :mod:`train_minimal`
    contract:

    * ``final.ckpt`` -- the trained model after :meth:`fit_latent_stats`, so
      ``latent_stats_fitted=(n_stats > 0)``.
    * ``best.ckpt`` -- the lowest ``val/total_loss`` snapshot. The DDP loop keeps
      no per-epoch model copy of its own, so the Lightning ``ModelCheckpoint``
      (``lightning_ckpts/lightning_best-*.ckpt``) is the only record of the
      best-val weights; they are loaded into a fresh
      ``SeqVaeLagAttnV1(**model_kwargs)`` via ``load_checkpoint_strict`` (which
      strips the ``model.`` / ``_orig_model.`` wrapper prefixes and dedups the
      duplicate registration) and re-saved here with ``latent_stats_fitted=False``
      (the best snapshot predates the latent-stats fit, exactly as single-GPU).
      With no validation split the Lightning best does not exist, so ``best.ckpt``
      falls back to a copy of ``final.ckpt`` (cf. ``train_minimal``). Either way
      the synthetic ``best.ckpt`` / ``final.ckpt`` are the consumable artifacts;
      the Lightning file stays in ``lightning_ckpts/`` and is never named
      ``best.ckpt`` / ``final.ckpt``.
    """
    cm = trainer.callback_metrics
    train_metrics = {
        k: _as_float(cm[f"train/{k}"])
        for k in _TRAIN_METRIC_KEYS
        if f"train/{k}" in cm
    }
    val_loss = _as_float(cm.get("val/total_loss", float("nan")))

    loss_settings_aug = _augment_loss_settings(loss_settings)

    save_checkpoint(
        results_dir / "final.ckpt",
        model=pl_model.orig_model,
        model_kwargs=model_kwargs,
        config=config,
        data_meta=data_meta,
        epoch=int(trainer.current_epoch or epochs),
        val_loss=val_loss,
        train_metrics=train_metrics,
        loss_settings=loss_settings_aug,
        latent_stats_fitted=(n_stats > 0),
    )
    print(f"[train_ddp] wrote {results_dir / 'final.ckpt'} "
          f"(latent_stats_fitted={n_stats > 0}, samples={n_stats})")

    # --- best.ckpt: synthetic-format mirror of the Lightning best snapshot -----
    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
    best_path = getattr(ckpt_cb, "best_model_path", "") or ""
    if best_path and Path(best_path).is_file():
        # YAML / checkpoint store ``logvar_clamp`` as a list; the constructor
        # wants a tuple (same coercion ``evaluate_te.load_eval_checkpoint`` does).
        best_kwargs = dict(model_kwargs)
        clamp = best_kwargs.get("logvar_clamp")
        if clamp is not None and not isinstance(clamp, tuple):
            best_kwargs["logvar_clamp"] = (float(clamp[0]), float(clamp[1]))
        best_model = SeqVaeLagAttnV1(**best_kwargs)
        if load_checkpoint_strict(best_model, best_path, map_location="cpu") is None:
            raise RuntimeError(
                f"load_checkpoint_strict could not align the Lightning best "
                f"checkpoint {best_path} with SeqVaeLagAttnV1(**model_kwargs)."
            )
        best_score = _as_float(getattr(ckpt_cb, "best_model_score", float("nan")))
        save_checkpoint(
            results_dir / "best.ckpt",
            model=best_model,
            model_kwargs=model_kwargs,
            config=config,
            data_meta=data_meta,
            epoch=int(trainer.current_epoch or epochs),
            val_loss=best_score,
            train_metrics=train_metrics,
            loss_settings=loss_settings_aug,
            latent_stats_fitted=False,
        )
        print(f"[train_ddp] wrote {results_dir / 'best.ckpt'} "
              f"(from {Path(best_path).name}, val/total_loss={best_score:.4f}, "
              f"latent_stats_fitted=False)")
    else:
        # No validation split / no best tracked -> mirror train_minimal's
        # fallback: best.ckpt == final.ckpt (with the fitted latent stats).
        save_checkpoint(
            results_dir / "best.ckpt",
            model=pl_model.orig_model,
            model_kwargs=model_kwargs,
            config=config,
            data_meta=data_meta,
            epoch=int(trainer.current_epoch or epochs),
            val_loss=val_loss,
            train_metrics=train_metrics,
            loss_settings=loss_settings_aug,
            latent_stats_fitted=(n_stats > 0),
        )
        print(f"[train_ddp] wrote {results_dir / 'best.ckpt'} "
              f"(fallback = final.ckpt, latent_stats_fitted={n_stats > 0})")


# =============================================================================
# Main training routine
# =============================================================================
def train_ddp(
    config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Train ``SeqVaeLagAttnV1`` on the ``G1_mix`` pool across all GPUs (DDP).

    Args:
        config: The parsed ``config_synth.yaml`` (any active benchmark; this
            routine forces ``G1_mix`` and re-overlays its data / loss block).
        overrides: Optional flat overrides -- the shared ``_OVERRIDE_MAP`` keys
            (``data_tag`` / ``beta`` / ``epochs`` / ``batch_size`` / ``lr`` /
            ``seed`` / ``data_dir`` / ``results_dir``), ``run_tag``, the
            DDP-only ``devices``, and ``resume_ckpt`` (path to a previous
            run's synthetic-format ``final.ckpt`` / ``best.ckpt`` whose model
            weights warm-start this run; ``None`` trains from scratch).

    Returns:
        A small results dict: ``run_tag``, ``results_dir``, ``n_gpus``,
        ``effective_lr``, ``latent_stats_samples`` (rank-zero values).
    """
    config = deepcopy(config)
    overrides = dict(overrides or {})

    # Force the mixed-population benchmark and re-overlay its data / loss block
    # (``resolve_active_benchmark`` is idempotent and pins gaussian_nll/learned).
    config["experiment"]["benchmark"] = _BENCHMARK
    apply_path_overrides(config, overrides)
    config = resolve_active_benchmark(config)
    _apply_train_overrides(config, overrides)

    exp = config["experiment"]
    optim_cfg = config["optim"]
    ddp_cfg = {**_DDP_DEFAULTS, **(config.get("ddp") or {})}
    # Deep-merge the early_stopping sub-block (the shallow ** above would drop
    # the defaults if the YAML sets only a subset of keys) and apply the CLI
    # ``--early-stop-patience`` override (setting it also enables early stopping).
    ddp_cfg["early_stopping"] = {
        **_DDP_DEFAULTS["early_stopping"],
        **((config.get("ddp") or {}).get("early_stopping") or {}),
    }
    es_patience = overrides.get("early_stop_patience")
    if es_patience is not None:
        ddp_cfg["early_stopping"]["enabled"] = True
        ddp_cfg["early_stopping"]["patience"] = int(es_patience)

    seed = int(exp.get("seed", 0))
    pl.seed_everything(seed, workers=True)
    set_seed(seed)

    # Resolve devices + LR scaling.
    devices_spec = overrides.get("devices", ddp_cfg["devices"])
    devices, n_gpus = _parse_devices(devices_spec)
    base_lr = float(optim_cfg["lr"])
    do_scale = bool(ddp_cfg["lr_scaling"]) and n_gpus > 1
    effective_lr = base_lr * n_gpus if do_scale else base_lr

    # Build the model on CPU so no rank pre-allocates on cuda:0 before Lightning
    # assigns devices. Keep ``model_kwargs`` for the checkpoint bridge.
    model, model_kwargs = build_model(config["model"], torch.device("cpu"))

    # Optional warm start (--resume-ckpt): continue training from the weights
    # of a previous run instead of a random init. Provide a SYNTHETIC-format
    # checkpoint written by train_ddp / train_minimal -- i.e.
    # ``results/G1_mix/<run_tag>/final.ckpt`` (or ``best.ckpt``) -- NOT a
    # Lightning ``lightning_ckpts/lightning_best-*.ckpt``. Only the model
    # weights are restored; the optimizer / LR schedule / epoch counter start
    # fresh (the synthetic checkpoints carry no optimizer state). Every DDP
    # rank loads identically, so the ranks start in sync.
    resume_ckpt = overrides.get("resume_ckpt")
    resume_meta: Optional[Dict[str, Any]] = None
    if resume_ckpt:
        resume_path = Path(resume_ckpt)
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"--resume-ckpt not found: {resume_path}. Provide the "
                f"synthetic-format checkpoint of a previous run, e.g. "
                f"results/G1_mix/<run_tag>/final.ckpt."
            )
        # Read the checkpoint once and reuse the dict for BOTH the strict weight
        # load and the warm-start banner's "checkpoint was trained with"
        # metadata, so the (large) state dict is loaded from disk only once.
        ckpt_obj = torch.load(resume_path, map_location="cpu", weights_only=False)
        if load_checkpoint_strict(model, ckpt_obj,
                                  map_location="cpu") is None:
            raise RuntimeError(
                f"load_checkpoint_strict could not align {resume_path} with "
                f"SeqVaeLagAttnV1(**model_kwargs) -- the checkpoint must come "
                f"from a run with the same model.* config (final.ckpt / "
                f"best.ckpt, not a Lightning lightning_best-*.ckpt)."
            )
        # Snapshot the source run's training hyperparameters for the banner so
        # the warm-start summary can contrast them against THIS run's fresh
        # config values. These are read-only -- they never feed back into the
        # optimizer / scheduler / loss, which are rebuilt from the current config.
        ckpt_dict = ckpt_obj if isinstance(ckpt_obj, dict) else {}
        ckpt_optim = (ckpt_dict.get("config") or {}).get("optim", {}) or {}
        ckpt_loss = ckpt_dict.get("loss_settings", {}) or {}
        resume_meta = {
            "path": resume_path,
            "epoch": ckpt_dict.get("epoch"),
            "lr": ckpt_optim.get("lr"),
            "lr_milestones": ckpt_optim.get("lr_milestones"),
            "lr_gamma": ckpt_optim.get("lr_gamma"),
            "weight_decay": ckpt_optim.get("weight_decay"),
            "beta": ckpt_loss.get("beta", ckpt_loss.get("kld_beta")),
        }
        del ckpt_obj, ckpt_dict  # weights are already in `model`; free the rest

    loss_settings = _resolve_loss_settings(config["loss"])

    pl_model = SyntheticSeqVaeLagAttnPl(
        model,
        lr=effective_lr,
        lr_milestones=optim_cfg.get("lr_milestones"),
        lr_gamma=float(optim_cfg.get("lr_gamma", 0.1)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        kld_beta=loss_settings["beta"],
        lambda_full=loss_settings["lambda_full"],
        lambda_base=loss_settings["lambda_base"],
        likelihood=loss_settings["likelihood"],
        sigma_obs=loss_settings["sigma_obs"],
        free_bits=loss_settings["free_bits"],
        warmup_epochs=int(ddp_cfg["warmup_epochs"]) if do_scale else 0,
        loss_spike_skip=config.get("loss_spike_skip"),
    )

    per_gpu_batch = int(optim_cfg["batch_size"])
    dm = SyntheticTEDataModule(config, batch_size=per_gpu_batch)

    run_tag = str(config.get("run_tag") or exp["tag"])
    results_root = resolve_user_path(config["paths"]["results_dir"])
    results_dir = results_root / str(exp["benchmark"]) / run_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    has_val = (
        resolve_user_path(config["paths"]["data_dir"])
        / str(exp["benchmark"]) / str(exp["tag"]) / "val.npz"
    ).is_file()

    epochs = int(optim_cfg["epochs"])
    plotting_cfg = config.get("plotting") or {}

    # Warm-start verification banner (rank-0). When resuming, make it explicit
    # that THIS run takes every training hyperparameter fresh from the config and
    # restarts the optimizer / LR schedule / epoch counter from zero -- so a
    # milestone configured for epoch N fires at epoch N of the resumed run, not
    # at original-epoch-N. The checkpoint values are printed only for contrast.
    if resume_meta is not None and _env_rank_zero():
        warmup_used = int(ddp_cfg["warmup_epochs"]) if do_scale else 0
        print(
            f"\n[train_ddp] WARM START from {resume_meta['path']}\n"
            f"        checkpoint trained: epoch={resume_meta['epoch']} "
            f"lr={resume_meta['lr']} milestones={resume_meta['lr_milestones']} "
            f"gamma={resume_meta['lr_gamma']} "
            f"weight_decay={resume_meta['weight_decay']} beta={resume_meta['beta']}\n"
            f"        THIS run uses (from config; fresh optimizer/schedule; epoch 0):\n"
            f"          lr={base_lr:g} (effective {effective_lr:g})  "
            f"milestones={optim_cfg.get('lr_milestones')}  "
            f"gamma={float(optim_cfg.get('lr_gamma', 0.1)):g}  "
            f"weight_decay={float(optim_cfg.get('weight_decay', 0.0)):g}\n"
            f"          warmup_epochs={warmup_used}  "
            f"beta={float(loss_settings['beta']):g}  epochs={epochs}\n"
            f"        -> schedules restart from epoch 0; a milestone at epoch N "
            f"fires at epoch N of THIS run."
        )

    # Write the consumable best.ckpt / final.ckpt DURING training (rank-0) so an
    # interrupted run (Ctrl-C on a rising val loss, crash) stays evaluable. On
    # graceful completion the post-fit _export_checkpoints overwrites both with
    # the finalized (latent-fitted) versions.
    loss_settings_aug = _augment_loss_settings(loss_settings)
    synthetic_ckpt_cb = _SyntheticCheckpointCallback(
        results_dir=results_dir,
        model_kwargs=model_kwargs,
        config=config,
        datamodule=dm,
        loss_settings=loss_settings_aug,
        epochs=epochs,
        has_val=has_val,
    )

    trainer = _build_trainer(
        ddp_cfg=ddp_cfg, results_dir=results_dir, epochs=epochs,
        devices=devices, n_gpus=n_gpus, has_val=has_val,
        run_tag=run_tag, plotting_cfg=plotting_cfg,
        extra_callbacks=[synthetic_ckpt_cb],
    )

    if _env_rank_zero():
        # Start the per-epoch metrics log fresh (mirrors train_minimal), so the
        # rank-0 _SyntheticCurveCallback does not append to a stale run's CSV.
        csv_path = results_dir / "metrics.csv"
        if csv_path.is_file():
            csv_path.unlink()
        with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        print(
            f"[train_ddp] run='{run_tag}' benchmark={exp['benchmark']} "
            f"data_tag='{exp['tag']}'\n"
            f"        n_gpus={n_gpus} per_gpu_batch={per_gpu_batch} "
            f"global_batch={per_gpu_batch * max(1, n_gpus)} epochs={epochs}\n"
            f"        base_lr={base_lr:g} effective_lr={effective_lr:g} "
            f"(scaling={'on' if do_scale else 'off'}) "
            f"likelihood={loss_settings['likelihood']} beta={loss_settings['beta']:g}\n"
            f"        results_dir={results_dir}"
        )

    trainer.fit(pl_model, datamodule=dm)

    # --- Post-fit latent-stats sync (ALL ranks, process group still alive) ----
    # Replace the per-rank EMA buffers with exact, cross-rank-reduced stats over
    # a sharded loader (each valid time step counted once).
    def _batch_to_inputs(batch: Any):
        return batch.fhr_st, batch.fhr_ph, build_u_stream(batch)

    n_stats = 0
    try:
        n_stats = int(
            pl_model.orig_model.fit_latent_stats(
                dm.make_plain_train_loader(),
                device=pl_model.device,
                batch_to_inputs=_batch_to_inputs,
            )
        )
    except RuntimeError as exc:  # pragma: no cover - defensive
        if trainer.is_global_zero:
            print(f"[train_ddp][warn] fit_latent_stats failed: {exc}")

    # Barrier so every rank has finished the all_reduce before rank 0 reads the
    # (now identical) buffers, then export the synthetic-format checkpoint.
    trainer.strategy.barrier()
    if trainer.is_global_zero:
        _export_checkpoints(
            trainer=trainer, pl_model=pl_model, model_kwargs=model_kwargs,
            config=config, data_meta=dm.data_meta, loss_settings=loss_settings,
            results_dir=results_dir, epochs=epochs, n_stats=n_stats,
        )
    trainer.strategy.barrier()

    return {
        "run_tag": run_tag,
        "results_dir": str(results_dir),
        "n_gpus": n_gpus,
        "effective_lr": effective_lr,
        "latent_stats_samples": n_stats,
    }


# =============================================================================
# CLI / edit-and-run
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments (every override defaults to ``None``)."""
    p = argparse.ArgumentParser(
        description="DDP training for the G1_mix mixed-population final model."
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--tag", "--data-tag", type=str, default=None, dest="data_tag",
                   help="experiment.tag -- which cached G1_mix pool to load")
    p.add_argument("--run-tag", type=str, default=None, dest="run_tag",
                   help="results subdirectory name (defaults to the data tag)")
    p.add_argument("--devices", type=str, default=None,
                   help="GPU count (e.g. 8) or explicit indices (e.g. 0,1,2,3)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="PER-GPU batch size (global batch = batch_size * n_gpus)")
    p.add_argument("--lr", type=float, default=None, help="override optim.lr")
    p.add_argument("--beta", type=float, default=None, help="override loss.kld_beta")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None, dest="data_dir")
    p.add_argument("--results-dir", type=str, default=None, dest="results_dir")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   dest="early_stop_patience",
                   help="enable EarlyStopping on val/total_loss with this "
                        "patience (epochs); omitted -> use config ddp.early_stopping")
    p.add_argument("--resume-ckpt", type=str, default=None, dest="resume_ckpt",
                   help="continue training from this synthetic-format "
                        "checkpoint (results/G1_mix/<run_tag>/final.ckpt or "
                        "best.ckpt): the model WEIGHTS are loaded, while every "
                        "training hyperparameter (lr, lr_milestones/lr_gamma, "
                        "weight_decay, beta) is taken fresh from the config and "
                        "the optimizer / LR schedule / epoch counter restart "
                        "from zero -- edit config_synth.yaml optim/loss to "
                        "change them; omitted -> train from scratch")
    return p.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry: parse args, load config, run :func:`train_ddp`."""
    args = parse_args(argv)
    config = load_config(args.config)
    overrides = {
        "data_tag": args.data_tag,
        "run_tag": args.run_tag,
        "devices": args.devices,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta": args.beta,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "early_stop_patience": args.early_stop_patience,
        "resume_ckpt": args.resume_ckpt,
    }
    train_ddp(config, overrides=overrides)


if __name__ == "__main__":
    # Two modes (Decision V2-D8), auto-detected from the command line:
    #   * CLI mode      -- launched with any --flag -> argparse main().
    #   * EDIT-AND-RUN  -- no arguments -> the RUN_CONFIG dict below.
    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "data_tag": "G1_mix_base",   # cached pool: data/G1_mix/<tag>/
        "run_tag": "G1_mix_base",    # output dir: results/G1_mix/<tag>/
        "devices": 1,                # 8 on the Linux box; 1 on Windows dev
        "epochs": None,              # None -> config optim.epochs
        "batch_size": None,          # None -> config optim.batch_size (PER-GPU)
        "lr": None,                  # None -> config optim.lr
        "beta": None,                # None -> config loss.kld_beta
        "seed": None,                # None -> config experiment.seed
        "data_dir": None,
        "results_dir": None,
        "early_stop_patience": None, # int -> enable EarlyStopping with this patience;
                                     # None -> use config ddp.early_stopping block
        "resume_ckpt": None,         # path to a previous run's synthetic-format
                                     # checkpoint -- results/G1_mix/<run_tag>/
                                     # final.ckpt (or best.ckpt) -- to continue
                                     # training from its weights (fresh
                                     # optimizer); None -> train from scratch
    }

    if len(sys.argv) > 1:
        main()
    else:
        cfg = load_config(CONFIG_PATH)
        train_ddp(cfg, overrides=RUN_CONFIG)
