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
* **Checkpoint bridge.** ``final.ckpt`` stores the bare ``SeqVaeLagAttnV1``
  state dict (unprefixed) + ``model_kwargs`` so the strict loader
  ``load_checkpoint_strict`` matches exactly.

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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as pl
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
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
    _OVERRIDE_MAP,
    _TRAIN_METRIC_KEYS,
    apply_path_overrides,
    build_model,
    load_config,
    resolve_active_benchmark,
    resolve_user_path,
    save_checkpoint,
    set_seed,
)

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_BENCHMARK = "G1_mix"

# Defaults for the optional top-level ``ddp:`` config block (only this script
# reads it). Every key is overridable from the YAML / CLI.
_DDP_DEFAULTS: Dict[str, Any] = {
    "devices": 8,
    "strategy": "ddp_find_unused_parameters_true",
    "precision": "32-true",
    "sync_batchnorm": True,
    "lr_scaling": True,
    "warmup_epochs": 5,
    "num_sanity_val_steps": 0,
    "gradient_clip_val": 0.5,
    "gradient_clip_algorithm": "norm",
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
) -> pl.Trainer:
    """Construct the ``pl.Trainer``, mirroring ``trainer_lag_attn_v1``.

    Uses ``ddp_find_unused_parameters_true`` for >1 GPU (the model keeps
    auxiliary logvar heads that may be ungraded under the ``mse`` likelihood),
    ``use_distributed_sampler=True`` so the DataModule's plain loaders are
    sharded automatically, and a ``CSVLogger`` + ``ModelCheckpoint`` for
    monitoring. The synthetic-format checkpoints are written separately by
    :func:`_export_checkpoints` (the Lightning ``ModelCheckpoint`` file is a
    convenience artifact and is **not** consumed by ``mixed_eval``).
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
    """Write ``final.ckpt`` in the synthetic ``save_checkpoint`` format (rank 0).

    The bare ``SeqVaeLagAttnV1`` (``pl_model.orig_model``) is checkpointed so the
    state-dict keys are unprefixed and align exactly with the strict loader. The
    ``loss_settings`` carries **both** ``beta`` and ``kld_beta`` (and the
    lambda / likelihood / sigma_obs / free_bits knobs): ``train_minimal`` stores
    only ``beta`` but ``mixed_eval.per_cell_lag_recovery`` reads ``kld_beta``, so
    the duplicate keeps the LOLO ablation on the trained $\\beta$ rather than the
    ``0.001`` default fallback.
    """
    cm = trainer.callback_metrics
    train_metrics = {
        k: _as_float(cm[f"train/{k}"])
        for k in _TRAIN_METRIC_KEYS
        if f"train/{k}" in cm
    }
    val_loss = _as_float(cm.get("val/total_loss", float("nan")))

    loss_settings_aug = dict(loss_settings)
    loss_settings_aug["kld_beta"] = loss_settings["beta"]

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
            ``seed`` / ``data_dir`` / ``results_dir``), ``run_tag``, and the
            DDP-only ``devices``.

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
    trainer = _build_trainer(
        ddp_cfg=ddp_cfg, results_dir=results_dir, epochs=epochs,
        devices=devices, n_gpus=n_gpus, has_val=has_val,
    )

    if _env_rank_zero():
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
    }

    if len(sys.argv) > 1:
        main()
    else:
        cfg = load_config(CONFIG_PATH)
        train_ddp(cfg, overrides=RUN_CONFIG)
