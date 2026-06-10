r"""Standalone minimal training loop for ``SeqVaeLagAttnV1`` (Decision D2).

A compact, dependency-light PyTorch loop -- **no PyTorch Lightning, no DDP, no
HDF5, no MLflow**. It exercises the *real* model and the *real* loss
(:meth:`SeqVaeLagAttnV1.forward` / :meth:`SeqVaeLagAttnV1.compute_loss`) so
fidelity to the production training path is preserved while keeping the
synthetic $\beta$-sweep fast and fully controlled.

The model claims its per-step KL divergence
$K_t = \mathrm{KL}(q_\phi(z_t\mid Y_{\le t},U_{\le t})\;\|\;p_\psi(z_t\mid
Y_{\le t}))$ is a surrogate for block transfer entropy
$\mathrm{TE}^{(H)}_{U\to Y,t}$. This loop trains the model on cached synthetic
data of *known* TE so later phases can measure whether $K_t$ tracks it.

Device handling:
    The loop is **device-agnostic**: it resolves a device from the ``runtime``
    block of ``config_synth.yaml`` (``device: auto`` -> CUDA when available,
    else CPU) and moves the model and every batch tensor onto it. It runs
    single-GPU -- this machine's RTX 4080, or one of the production box's 7
    GPUs selected by ``cuda_device``.

Public API:
    train: Reusable entry point ``train(config, overrides=None) -> results``.
        Phase-6 ``beta_sweep`` imports this directly.
    main: CLI wrapper around :func:`train`.

Run modes (project convention -- see Decision D9 in
``synthetic_te_validation_plan.md``): every ``synthetic/`` runner supports
**both** a CLI and an edit-and-run ``__main__``, auto-detected from whether any
command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.train_minimal --data-tag TAG --epochs 100 ...
        [--config PATH] [--run-tag TAG] [--beta B] [--batch-size N]
        [--grad-checkpoint/--no-grad-checkpoint] [--lr LR] [--a A] [--m M]
        [--device DEV] [--seed S]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook / no
      terminal flags needed)::

        python -m ...synthetic.train_minimal
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
    build_u_stream,
    make_dataloader,
)

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.data_dir`` / ``paths.results_dir`` config values are resolved
# relative to ``model_experiment/`` (identical convention to build_dataset.py).
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Tensor fields carried by a synthetic batch (native model channel layout).
_TENSOR_FIELDS = ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight")

# Per-epoch training metrics tracked and reduced as size-weighted means.
# ``mean_logvar_full`` / ``mean_logvar_base`` are the Sprint-5 collapse
# diagnostics — they record the mean observation logvar over the
# loss-masked support, so a single-channel collapse of the ``logvar_full``
# head (its lower clamp at -5) is visible in the metrics CSV.
#
# KLD scale note (two units, both correct, $\times d_z$ apart):
#   * ``kld_loss`` is the loss-side KL — the masked mean over
#     $(b, t, d)$, i.e. **nats per latent dimension per step**. This is the
#     quantity $\beta$ multiplies in the objective.
#   * ``kld_nats`` $= d_z \cdot$ ``kld_loss`` — the same masked mean but with
#     the latent dimensions **summed**, i.e. **nats per step**. For diagonal
#     Gaussians $\mathrm{KL}(q\,\|\,p) = \sum_d \mathrm{KL}_d$, so this is the
#     scale of the TE surrogate $\bar K$ that ``mixed_eval`` / ``evaluate_te``
#     plot against the analytic block TE. Read ``kld_nats`` (not ``kld_loss``)
#     when comparing training curves with the $\bar K$-vs-TE figures.
_TRAIN_KEYS = (
    "feat_loss", "base_loss", "kld_loss", "kld_nats", "total_loss", "pred_gap",
    "mu_prior_sat_frac", "delta_mu_sat_frac",
    "mean_logvar_full", "mean_logvar_base",
    "grad_norm",
)
# Subset persisted into the checkpoint's ``train_metrics`` block.
_TRAIN_METRIC_KEYS = (
    "feat_loss", "base_loss", "kld_loss", "kld_nats", "total_loss", "pred_gap",
    "mu_prior_sat_frac", "delta_mu_sat_frac",
    "mean_logvar_full", "mean_logvar_base",
)
# Per-epoch evaluation metrics (no gradient-related fields).
_EVAL_KEYS = (
    "feat_loss", "base_loss", "kld_loss", "kld_nats", "total_loss", "pred_gap",
    "mean_logvar_full", "mean_logvar_base",
)

# CSV column order (one row per epoch).
_FIELDNAMES = [
    "epoch",
    "train_feat_loss", "train_base_loss", "train_kld_loss", "train_kld_nats",
    "train_total_loss",
    "train_pred_gap", "train_mu_prior_sat_frac", "train_delta_mu_sat_frac",
    "train_mean_logvar_full", "train_mean_logvar_base",
    "train_grad_norm",
    "val_feat_loss", "val_base_loss", "val_kld_loss", "val_kld_nats",
    "val_total_loss",
    "val_pred_gap",
    "val_mean_logvar_full", "val_mean_logvar_base",
    "lr", "epoch_seconds", "nan_skips",
]

# Flat-override key -> (config section, field) for :func:`_apply_overrides`.
_OVERRIDE_MAP: Dict[str, Tuple[str, str]] = {
    "data_tag": ("experiment", "tag"),
    "beta": ("loss", "kld_beta"),
    "epochs": ("optim", "epochs"),
    "batch_size": ("optim", "batch_size"),
    "lr": ("optim", "lr"),
    "a": ("data", "a"),
    "m": ("data", "M"),
    "grad_checkpoint": ("model", "attention_grad_checkpoint"),
    "device": ("runtime", "device"),
    "seed": ("experiment", "seed"),
    "plot_every": ("plotting", "plot_every"),
    # Path overrides -- accept absolute paths on any drive, ``~`` for the
    # user's home, or ``$VAR``/``${VAR}`` env-var references. Resolution is
    # done lazily by :func:`resolve_user_path` so an override applied at any
    # stage (CLI flag, RUN_CONFIG, gpu_pool patch) takes effect at use-time.
    "data_dir": ("paths", "data_dir"),
    "results_dir": ("paths", "results_dir"),
}


# =============================================================================
# Config / device / seed helpers
# =============================================================================

def resolve_user_path(value: Any) -> Path:
    r"""Resolve a config-supplied path to an absolute :class:`~pathlib.Path`.

    Handles the three shapes a user is likely to put in
    ``paths.data_dir`` / ``paths.results_dir`` (or pass via
    ``--data-dir`` / ``--results-dir``):

        * **Relative** (``./data``, ``results``) -- joined with
          ``model_experiment/`` so the default ``./data`` keeps the
          baked-in behaviour.
        * **Absolute on any drive** (``D:/teb_data``, ``E:\caches``,
          ``/mnt/scratch/teb``) -- used as-is; the ``model_experiment/``
          prefix is dropped.
        * **Home-relative** (``~/teb``) or **env-var-prefixed**
          (``$DATA_ROOT/teb``, ``${SCRATCH}/teb``) -- expanded via
          :func:`os.path.expanduser` and :func:`os.path.expandvars` before
          the relative / absolute check.

    Calling this helper at *use-time* (rather than canonicalising at
    config-load time) means a late override -- a CLI flag, a
    ``RUN_CONFIG`` dict, or a :mod:`gpu_pool` worker patch -- is honoured
    transparently without any extra plumbing.

    Args:
        value: A path-like value (``str`` / :class:`os.PathLike`).

    Returns:
        The resolved absolute path. Symlinks are followed via
        :meth:`Path.resolve`; missing parents do not raise.
    """
    raw = str(value)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = Path(expanded)
    if not p.is_absolute():
        p = _EXPERIMENT_DIR / p
    return p.resolve()


def apply_path_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    r"""Route ``data_dir`` / ``results_dir`` overrides into ``config['paths']``.

    Shared by every synthetic runner (``beta_sweep``, ``evaluate_te``,
    ``calibration``, ``null_controls``, ``lag_recovery``, ``directionality``,
    ``build_dataset``, ``gpu_pool``) so a custom dataset / output location can
    be supplied **once** -- via a ``--data-dir`` / ``--results-dir`` CLI flag or
    the in-file ``RUN_CONFIG`` dict -- instead of editing
    ``config_synth.yaml``. A ``None`` value is ignored, so the YAML's
    ``paths.data_dir`` / ``paths.results_dir`` stay the default (Decision V2-D8
    "respect the config" semantics): the override is the exception, the config
    is the rule.

    The raw value is stored verbatim; resolution is deferred to
    :func:`resolve_user_path` at use-time, so a relative path, an absolute path
    on any drive, ``~`` and ``$VAR`` / ``${VAR}`` references are all honoured
    exactly as they are for the YAML values.

    Args:
        config: The config dict (mutated in place). A missing ``paths`` block
            is created only when an override is actually written, so the
            all-``None`` no-op path leaves ``config`` untouched.
        overrides: Flat ``{key: value}`` overrides; only ``data_dir`` /
            ``results_dir`` are consumed, every other key is ignored.

    Returns:
        The same ``config`` dict.
    """
    for key in ("data_dir", "results_dir"):
        value = overrides.get(key)
        if value is not None:
            config.setdefault("paths", {})[key] = value
    return config


def resolve_active_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay the active benchmark's block onto the flat top-level config.

    The Phase-7 config groups per-benchmark settings under a ``benchmarks:``
    mapping -- one block per benchmark (A/B/C/E/G), each holding a ``data`` and
    an optional ``sweep`` / ``rho_null`` sub-block. ``experiment.benchmark``
    names the active block. This resolver copies that block's ``data`` to
    ``config["data"]``, its ``sweep`` to ``config["sweep"]`` and its
    ``rho_null`` (when present) to ``config["rho_null"]`` so every downstream
    reader keeps the flat shape used since Phase 1.

    Idempotent, and a no-op for a config without a ``benchmarks`` key (so a
    hand-built test config still works unchanged).

    Args:
        config: The parsed config dict (mutated in place).

    Returns:
        The same ``config`` dict, with ``data`` / ``sweep`` / ``rho_null``
        populated from the active benchmark block.

    Raises:
        KeyError: If ``experiment.benchmark`` names no block in ``benchmarks``.
    """
    benchmarks = config.get("benchmarks")
    if not benchmarks:
        return config
    benchmark = str(config["experiment"]["benchmark"])
    if benchmark not in benchmarks:
        raise KeyError(
            f"experiment.benchmark={benchmark!r} has no block under "
            f"`benchmarks:` (available: {sorted(benchmarks)})."
        )
    block = benchmarks[benchmark]
    config["data"] = deepcopy(block["data"])
    config["sweep"] = deepcopy(block.get("sweep", {}))
    if "rho_null" in block:
        config["rho_null"] = deepcopy(block["rho_null"])
    # Optional per-benchmark loss overlay: a benchmark block may carry a
    # ``loss`` sub-block whose keys are merged onto (not replacing) the global
    # ``loss`` block. This lets a benchmark pin its own likelihood / sigma_obs
    # (e.g. ``G1_mix`` trains under ``gaussian_nll`` for nat-scale calibration)
    # without flipping the global default for every other benchmark. No-op for
    # any block that does not define ``loss``, so existing benchmarks are
    # unaffected.
    if "loss" in block:
        config.setdefault("loss", {})
        config["loss"].update(deepcopy(block["loss"]))
    return config


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse the synthetic-experiment YAML config.

    The active benchmark block (``benchmarks[experiment.benchmark]``) is
    overlaid onto the flat ``data`` / ``sweep`` keys via
    :func:`resolve_active_benchmark`.

    Args:
        config_path: Path to ``config_synth.yaml``.

    Returns:
        The parsed, benchmark-resolved config as a nested dict.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return resolve_active_benchmark(config)


def resolve_device(runtime_cfg: Dict[str, Any]) -> torch.device:
    """Resolve the compute device from the ``runtime`` config block.

    Args:
        runtime_cfg: The ``runtime`` block: ``device`` (``auto`` / ``cpu`` /
            ``cuda`` / ``cuda:N``) and ``cuda_device`` (GPU index).

    Returns:
        The resolved :class:`torch.device`. ``auto`` maps to
        ``cuda:{cuda_device}`` when CUDA is available, else ``cpu``.
    """
    spec = str(runtime_cfg.get("device", "auto")).lower()
    idx = int(runtime_cfg.get("cuda_device", 0))
    if spec == "auto":
        return torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    if spec == "cuda":
        return torch.device(f"cuda:{idx}")
    return torch.device(spec)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    ``cudnn`` is intentionally left non-deterministic: the proof-of-life run
    needs training *stability*, not bitwise reproducibility, and determinism
    costs throughput.

    Args:
        seed: The RNG seed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model / optimiser / data builders
# =============================================================================

def build_model(
    model_cfg: Dict[str, Any], device: torch.device
) -> Tuple[SeqVaeLagAttnV1, Dict[str, Any]]:
    """Construct :class:`SeqVaeLagAttnV1` from the ``model`` config block.

    Args:
        model_cfg: The ``model`` block -- a 1:1 map of the keyword-only
            constructor arguments.
        device: Device to move the model onto.

    Returns:
        Tuple ``(model, model_kwargs)`` where ``model_kwargs`` is the exact
        resolved constructor kwargs (stored verbatim in the checkpoint so
        downstream phases can rebuild the architecture without the config).

    Raises:
        ValueError: Propagated from the constructor when ``c_u`` and
            ``use_up_st`` are inconsistent.
    """
    kwargs = deepcopy(dict(model_cfg))
    # YAML gives ``logvar_clamp`` as a list; the constructor expects a tuple.
    clamp = kwargs.get("logvar_clamp")
    if clamp is not None:
        kwargs["logvar_clamp"] = (float(clamp[0]), float(clamp[1]))
    model = SeqVaeLagAttnV1(**kwargs)
    model.to(device)
    return model, kwargs


def build_optimizer(
    model: torch.nn.Module, optim_cfg: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Build the ``AdamW`` optimiser, mirroring the production trainer.

    Args:
        model: The model whose parameters are optimised.
        optim_cfg: The ``optim`` block (``lr``, ``weight_decay``).

    Returns:
        A configured :class:`torch.optim.AdamW` (``eps=1e-8``,
        ``betas=(0.9, 0.95)`` -- identical to ``trainer_lag_attn_v1.py``).
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        eps=1e-8,
        betas=(0.9, 0.95),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, optim_cfg: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler.MultiStepLR]:
    """Build a per-epoch ``MultiStepLR`` scheduler if milestones are set.

    Args:
        optimizer: The optimiser to schedule.
        optim_cfg: The ``optim`` block (``lr_milestones``, ``lr_gamma``).

    Returns:
        A :class:`MultiStepLR` stepped once per epoch, or ``None`` when no
        milestones are configured.
    """
    milestones = optim_cfg.get("lr_milestones") or []
    if not milestones:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(m) for m in milestones],
        gamma=float(optim_cfg.get("lr_gamma", 0.1)),
    )


def make_dataloaders(
    config: Dict[str, Any], batch_size: int
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    """Build train / val dataloaders over the cached benchmark dataset.

    Args:
        config: The (post-override) config dict.
        batch_size: Samples per batch.

    Returns:
        Tuple ``(train_loader, val_loader, data_meta)``. ``val_loader`` is
        ``None`` when no ``val.npz`` is cached. ``data_meta`` is the dataset
        ``meta.json`` (analytic ``te_true``, ``true_lag_band``, RNG seeds ...).

    Raises:
        FileNotFoundError: If the train split is not cached -- the message
            points at :mod:`build_dataset`.
    """
    exp = config["experiment"]
    data_root = resolve_user_path(config["paths"]["data_dir"])
    cache_dir = data_root / str(exp["benchmark"]) / str(exp["tag"])
    train_npz = cache_dir / "train.npz"
    val_npz = cache_dir / "val.npz"

    if not train_npz.is_file():
        raise FileNotFoundError(
            f"cached dataset not found: {train_npz}\n"
            f"Build it first, e.g.:\n"
            f"  python -m model.vae_teb_prediction.model.model_experiment."
            f"synthetic.build_dataset --tag {exp['tag']}"
        )

    # Optional DataLoader knobs (``config.dataset`` block, all defaults
    # preserve pre-existing single-GPU behaviour). Useful on multi-GPU boxes
    # with bigger memory where host->device copies start to dominate.
    ds_cfg = (config.get("dataset") or {})
    num_workers = int(ds_cfg.get("num_workers", 0))
    pin_memory = bool(ds_cfg.get("pin_memory", False))
    persistent_workers = bool(ds_cfg.get("persistent_workers", False))

    train_ds = SyntheticTEDataset(train_npz)
    train_loader = make_dataloader(
        train_ds, batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_npz.is_file():
        val_ds = SyntheticTEDataset(val_npz)
        val_loader = make_dataloader(
            val_ds, batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    return train_loader, val_loader, train_ds.meta


def move_batch(batch: Any, device: torch.device) -> Any:
    """Move every tensor field of a synthetic batch onto ``device`` in place.

    Args:
        batch: A batched :class:`AttributeDict`.
        device: Destination device.

    Returns:
        The same ``batch`` object, with its tensor fields on ``device``.
    """
    for field in _TENSOR_FIELDS:
        if field in batch and torch.is_tensor(batch[field]):
            batch[field] = batch[field].to(device, non_blocking=True)
    return batch


# =============================================================================
# Train / eval epoch
# =============================================================================

def train_one_epoch(
    model: SeqVaeLagAttnV1,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    beta: float,
    lambda_full: float,
    lambda_base: float,
    grad_clip_norm: float,
    likelihood: str = "mse",
    sigma_obs: "float | str" = 1.0,
    free_bits: float = 0.0,
) -> Dict[str, float]:
    r"""Run one training epoch: forward -> loss -> backward -> clip -> step.

    Non-finite training steps (NaN/Inf in the loss or gradient norm) are
    skipped and counted in ``nan_skips`` rather than corrupting the weights.

    Args:
        model: The model in train mode.
        loader: Training :class:`DataLoader`.
        optimizer: The optimiser.
        device: Compute device.
        beta: KL weight $\beta$.
        lambda_full: Weight on the full-forecast feature loss.
        lambda_base: Weight on the baseline feature loss.
        grad_clip_norm: Gradient-norm clip threshold.
        likelihood: Reconstruction likelihood passed to
            :meth:`SeqVaeLagAttnV1.compute_loss`. Defaults to ``'mse'`` so
            legacy configs train identically to pre-Sprint-5 behaviour.
        sigma_obs: Observation noise (scalar or ``'learned'``).
        free_bits: Per-dim KL floor (0.0 is a no-op).

    Returns:
        Size-weighted epoch means of all entries in :data:`_TRAIN_KEYS`, plus
        the integer ``nan_skips`` count.
    """
    model.train()
    accum = {k: 0.0 for k in _TRAIN_KEYS}
    n_samples = 0
    nan_skips = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)
        bs = int(y_st.size(0))

        optimizer.zero_grad(set_to_none=True)
        try:
            out = model(y_st, y_ph, u_stream)
            losses = model.compute_loss(
                out, y_st, y_ph, weight=batch.weight,
                beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
                likelihood=likelihood, sigma_obs=sigma_obs,
                free_bits=free_bits,
            )
            total = losses["total_loss"]
            if not torch.isfinite(total):
                nan_skips += 1
                continue
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
            if not torch.isfinite(grad_norm):
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
        except RuntimeError as exc:  # pragma: no cover - hardware dependent
            if "out of memory" in str(exc).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory in a training step "
                    f"(batch_size={bs}). Lower --batch-size and/or keep "
                    f"--grad-checkpoint enabled."
                ) from exc
            raise

        accum["feat_loss"] += float(losses["feat_loss"]) * bs
        accum["base_loss"] += float(losses["base_loss"]) * bs
        accum["kld_loss"] += float(losses["kld_loss"]) * bs
        # Dim-summed KL in nats/step: exact rescale on the same loss mask
        # ($\mathrm{KL} = \sum_d \mathrm{KL}_d$), comparable to eval-side $\bar K$.
        accum["kld_nats"] += (
            float(losses["kld_loss"]) * int(out["mu_post"].shape[-1]) * bs
        )
        accum["total_loss"] += float(total) * bs
        accum["pred_gap"] += float(losses["base_loss"] - losses["feat_loss"]) * bs
        accum["mu_prior_sat_frac"] += float(out["mu_prior_sat_frac"]) * bs
        accum["delta_mu_sat_frac"] += float(out["delta_mu_sat_frac"]) * bs
        accum["mean_logvar_full"] += float(losses["mean_logvar_full"]) * bs
        accum["mean_logvar_base"] += float(losses["mean_logvar_base"]) * bs
        accum["grad_norm"] += float(grad_norm) * bs
        n_samples += bs

    if n_samples == 0:
        metrics = {k: float("nan") for k in _TRAIN_KEYS}
    else:
        metrics = {k: accum[k] / n_samples for k in _TRAIN_KEYS}
    metrics["nan_skips"] = nan_skips
    return metrics


@torch.no_grad()
def evaluate(
    model: SeqVaeLagAttnV1,
    loader: Optional[Any],
    device: torch.device,
    *,
    beta: float,
    lambda_full: float,
    lambda_base: float,
    likelihood: str = "mse",
    sigma_obs: "float | str" = 1.0,
    free_bits: float = 0.0,
) -> Dict[str, float]:
    r"""Run one evaluation pass (no backward, no optimiser step).

    ``model.eval()`` also halts the ``mu_post_running_*`` EMA, so validation
    never perturbs the latent-stats buffers.

    Args:
        model: The model.
        loader: Validation :class:`DataLoader`, or ``None``.
        device: Compute device.
        beta: KL weight $\beta$.
        lambda_full: Weight on the full-forecast feature loss.
        lambda_base: Weight on the baseline feature loss.
        likelihood: Reconstruction likelihood passed to
            :meth:`SeqVaeLagAttnV1.compute_loss`. Defaults to ``'mse'``.
        sigma_obs: Observation noise (scalar or ``'learned'``).
        free_bits: Per-dim KL floor (0.0 is a no-op).

    Returns:
        Size-weighted means of :data:`_EVAL_KEYS`; all-NaN when ``loader`` is
        ``None``.
    """
    if loader is None:
        return {k: float("nan") for k in _EVAL_KEYS}

    model.eval()
    accum = {k: 0.0 for k in _EVAL_KEYS}
    n_samples = 0

    for batch in loader:
        batch = move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)
        bs = int(y_st.size(0))

        out = model(y_st, y_ph, u_stream)
        losses = model.compute_loss(
            out, y_st, y_ph, weight=batch.weight,
            beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
            likelihood=likelihood, sigma_obs=sigma_obs,
            free_bits=free_bits,
        )
        accum["feat_loss"] += float(losses["feat_loss"]) * bs
        accum["base_loss"] += float(losses["base_loss"]) * bs
        accum["kld_loss"] += float(losses["kld_loss"]) * bs
        accum["kld_nats"] += (
            float(losses["kld_loss"]) * int(out["mu_post"].shape[-1]) * bs
        )
        accum["total_loss"] += float(losses["total_loss"]) * bs
        accum["pred_gap"] += float(losses["base_loss"] - losses["feat_loss"]) * bs
        accum["mean_logvar_full"] += float(losses["mean_logvar_full"]) * bs
        accum["mean_logvar_base"] += float(losses["mean_logvar_base"]) * bs
        n_samples += bs

    if n_samples == 0:
        return {k: float("nan") for k in _EVAL_KEYS}
    return {k: accum[k] / n_samples for k in _EVAL_KEYS}


@torch.no_grad()
def compute_kbar(
    model: SeqVaeLagAttnV1,
    loader: Optional[Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    r"""Mean TE surrogate $\bar K$ over a loader.

    $\bar K$ is the per-step KL $K_t$ averaged over valid (non-warm-up) time
    steps, computed via :meth:`SeqVaeLagAttnV1.measure_transfer_entropy`.

    Args:
        model: The (trained) model.
        loader: A :class:`DataLoader`, or ``None``.
        device: Compute device.
        max_batches: Optional cap on the number of batches.

    Returns:
        The size-weighted mean $\bar K$ in nats, or NaN if ``loader`` is empty.
    """
    if loader is None:
        return float("nan")
    total, n_samples = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)
        bs = int(y_st.size(0))
        k = model.measure_transfer_entropy(y_st, y_ph, u_stream, reduce_mean=True)
        total += float(k) * bs
        n_samples += bs
    return total / n_samples if n_samples else float("nan")


# =============================================================================
# Persistence
# =============================================================================

def append_csv_row(
    csv_path: Path, row: Dict[str, Any], fieldnames: list
) -> None:
    """Append one metrics row to ``csv_path``, writing the header on first use.

    The file handle is flushed on close every epoch, so a crash leaves a
    readable partial log.

    Args:
        csv_path: Destination CSV path.
        row: A row dict keyed by ``fieldnames``.
        fieldnames: The CSV column order.
    """
    csv_path = Path(csv_path)
    write_header = not csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    *,
    model: SeqVaeLagAttnV1,
    model_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    data_meta: Dict[str, Any],
    epoch: int,
    val_loss: float,
    train_metrics: Dict[str, float],
    loss_settings: Dict[str, float],
    latent_stats_fitted: bool,
) -> None:
    """Save a training checkpoint.

    The bare (unprefixed) ``state_dict`` is stored under ``model_state_dict``
    -- the key :func:`train.graph_models_utils.load_checkpoint_strict` scans
    for. Phase 4 rebuilds the model via ``SeqVaeLagAttnV1(**model_kwargs)``
    then loads this file ``strict=True``.

    Args:
        path: Destination ``.ckpt`` path.
        model: The model to checkpoint.
        model_kwargs: Exact resolved constructor kwargs.
        config: The effective (post-override) config.
        data_meta: The dataset ``meta.json`` (carries the generator seeds).
        epoch: Epoch index this checkpoint corresponds to.
        val_loss: Validation total loss at this checkpoint.
        train_metrics: Last-epoch training metrics.
        loss_settings: ``{beta, lambda_full, lambda_base}``.
        latent_stats_fitted: Whether ``fit_latent_stats`` was run (exact
            buffers) vs. the noisy EMA buffers.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "config": config,
        "data_meta": data_meta,
        "epoch": int(epoch),
        "val_total_loss": float(val_loss),
        "train_metrics": {
            k: float(train_metrics.get(k, float("nan")))
            for k in _TRAIN_METRIC_KEYS
        },
        "loss_settings": loss_settings,
        "latent_stats_fitted": bool(latent_stats_fitted),
        "torch_version": torch.__version__,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _maybe_fit_latent_stats(
    model: SeqVaeLagAttnV1, train_loader: Any, device: torch.device
) -> int:
    """Run :meth:`SeqVaeLagAttnV1.fit_latent_stats` over the training set.

    Replaces the noisy EMA ``mu_post_running_*`` buffers (momentum 0.01) with
    exact statistics, so a checkpoint is trustworthy for downstream consumers.

    Args:
        model: The trained model.
        train_loader: The training :class:`DataLoader`.
        device: Compute device.

    Returns:
        The number of time-step samples aggregated, or ``0`` on failure.
    """
    def batch_to_inputs(batch: Any):
        return batch.fhr_st, batch.fhr_ph, build_u_stream(batch)

    try:
        return int(model.fit_latent_stats(
            train_loader, device=device, batch_to_inputs=batch_to_inputs
        ))
    except RuntimeError as exc:  # pragma: no cover - defensive
        print(f"[warn] fit_latent_stats failed: {exc}")
        return 0


def _refresh_training_curves(
    csv_path: Path, run_tag: str, plotting_cfg: Dict[str, Any]
) -> None:
    """Re-render the train/val loss-curve figures from the metrics CSV.

    Writes both the matplotlib ``training_curves.{pdf,png}`` grid and the
    interactive plotly ``loss_plot_epoch.html`` (mirrors
    :class:`train.callbacks.LossPlotCallback`) next to the CSV. Both
    renderers are imported lazily and wrapped in independent
    try/except blocks -- a plotting bug in one backend must never abort
    training nor suppress the other backend's output.

    Args:
        csv_path: The run's ``metrics.csv`` (flushed every epoch).
        run_tag: Run label, used in the figure title.
        plotting_cfg: The ``plotting`` config block (``enabled`` toggle).
    """
    if not plotting_cfg.get("enabled", True):
        return
    try:
        from model.vae_teb_prediction.model.model_experiment.synthetic.plot_training_curves import (
            plot_training_curves,
        )
        plot_training_curves(csv_path, run_tag=run_tag)
    except Exception as exc:  # noqa: BLE001 - plotting must not kill training
        print(f"[warn] training-curve plot failed: {type(exc).__name__}: {exc}")
    try:
        from model.vae_teb_prediction.model.model_experiment.synthetic.plot_training_curves import (
            plot_training_curves_html,
        )
        plot_training_curves_html(csv_path, run_tag=run_tag)
    except Exception as exc:  # noqa: BLE001 - plotting must not kill training
        print(
            f"[warn] training-curve HTML plot failed: "
            f"{type(exc).__name__}: {exc}"
        )


# =============================================================================
# Overrides + main training entry point
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply flat CLI overrides onto a config dict in place.

    ``run_tag`` is stored as a top-level key; every other recognised key is
    routed through :data:`_OVERRIDE_MAP`. ``None`` values are ignored.

    Args:
        config: The config dict (mutated in place).
        overrides: Flat ``{key: value}`` overrides.

    Returns:
        The same ``config`` dict.

    Raises:
        KeyError: On an unrecognised override key (fails loud on sweep typos).
    """
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "run_tag":
            config["run_tag"] = value
            continue
        if key not in _OVERRIDE_MAP:
            raise KeyError(f"unknown override key: {key!r}")
        section, field = _OVERRIDE_MAP[key]
        # ``setdefault`` so an override into an optional section (e.g.
        # ``plotting``) still works when a hand-built config omits the block.
        config.setdefault(section, {})[field] = value
    return config


def _assemble_row(
    epoch: int,
    train_m: Dict[str, float],
    val_m: Dict[str, float],
    lr: float,
    epoch_seconds: float,
) -> Dict[str, Any]:
    """Build one :data:`_FIELDNAMES`-keyed CSV row from epoch metrics."""
    return {
        "epoch": epoch,
        "train_feat_loss": train_m["feat_loss"],
        "train_base_loss": train_m["base_loss"],
        "train_kld_loss": train_m["kld_loss"],
        "train_kld_nats": train_m.get("kld_nats", float("nan")),
        "train_total_loss": train_m["total_loss"],
        "train_pred_gap": train_m["pred_gap"],
        "train_mu_prior_sat_frac": train_m["mu_prior_sat_frac"],
        "train_delta_mu_sat_frac": train_m["delta_mu_sat_frac"],
        "train_mean_logvar_full": train_m["mean_logvar_full"],
        "train_mean_logvar_base": train_m["mean_logvar_base"],
        "train_grad_norm": train_m["grad_norm"],
        "val_feat_loss": val_m["feat_loss"],
        "val_base_loss": val_m["base_loss"],
        "val_kld_loss": val_m["kld_loss"],
        "val_kld_nats": val_m.get("kld_nats", float("nan")),
        "val_total_loss": val_m["total_loss"],
        "val_pred_gap": val_m["pred_gap"],
        "val_mean_logvar_full": val_m["mean_logvar_full"],
        "val_mean_logvar_base": val_m["mean_logvar_base"],
        "lr": lr,
        "epoch_seconds": epoch_seconds,
        "nan_skips": train_m["nan_skips"],
    }


def train(
    config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    r"""Train :class:`SeqVaeLagAttnV1` on a cached synthetic benchmark dataset.

    This is the reusable entry point. The Phase-6 ``beta_sweep`` calls it
    directly with ``overrides={"beta": ..., "run_tag": ...}``.

    Workflow: apply overrides -> seed -> resolve device -> load cached data ->
    build model/optimiser/scheduler -> epoch loop (train + eval + checkpoint
    best) -> :meth:`fit_latent_stats` -> save ``final.ckpt`` -> report $\bar K$.

    Args:
        config: The parsed ``config_synth.yaml``.
        overrides: Optional flat overrides (see :func:`_apply_overrides`).

    Returns:
        A results dict: ``run_tag``, ``results_dir``, ``best_epoch``,
        ``best_val_total_loss``, ``final_train_metrics``, ``final_val_metrics``,
        ``kbar_val``, ``kbar_train``, ``te_true``, and checkpoint paths.
    """
    config = deepcopy(config)
    if overrides:
        config = _apply_overrides(config, overrides)

    exp = config["experiment"]
    optim_cfg = config["optim"]
    loss_cfg = config["loss"]
    # Optional ``plotting`` block -- a config without it still trains fine.
    plotting_cfg = config.get("plotting", {}) or {}
    plot_every = int(plotting_cfg.get("plot_every", 10))

    set_seed(exp.get("seed", 0))
    device = resolve_device(config["runtime"])

    batch_size = int(optim_cfg["batch_size"])
    epochs = int(optim_cfg["epochs"])
    grad_clip = float(optim_cfg["grad_clip_norm"])
    beta = float(loss_cfg["kld_beta"])
    lambda_full = float(loss_cfg["lambda_full"])
    lambda_base = float(loss_cfg["lambda_base"])
    # Sprint-5 likelihood switch. Defaults preserve pre-Sprint-5 behaviour
    # (MSE feat_loss / base_loss, no free-bits floor). ``sigma_obs`` accepts
    # a positive scalar or the literal string ``'learned'``; YAML carries it
    # as a string when set to ``learned`` and as a number otherwise.
    likelihood = str(loss_cfg.get("likelihood", "mse"))
    sigma_obs_raw = loss_cfg.get("sigma_obs", 1.0)
    if isinstance(sigma_obs_raw, str) and sigma_obs_raw != "learned":
        # Allow YAML to carry a stringified float like "1.0" — coerce.
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
    loss_settings = {
        "beta": beta, "lambda_full": lambda_full, "lambda_base": lambda_base,
        "likelihood": likelihood, "sigma_obs": sigma_obs,
        "free_bits": free_bits,
    }

    train_loader, val_loader, data_meta = make_dataloaders(config, batch_size)

    # Provenance check: --a / --m are recorded, not acted on -- warn if they
    # disagree with what the cache was actually generated with.
    for cfg_key, meta_key in (("a", "a"), ("M", "M")):
        cfg_val = config["data"].get(cfg_key)
        meta_val = data_meta.get(meta_key)
        if cfg_val is not None and meta_val is not None:
            if abs(float(cfg_val) - float(meta_val)) > 1e-6:
                print(
                    f"[warn] config data.{cfg_key}={cfg_val} != cached dataset "
                    f"{meta_key}={meta_val}. The cache is used as-is; check "
                    f"that --data-tag points at the intended dataset."
                )

    model, model_kwargs = build_model(config["model"], device)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg)

    run_tag = str(config.get("run_tag") or exp["tag"])
    results_root = resolve_user_path(config["paths"]["results_dir"])
    results_dir = results_root / str(exp["benchmark"]) / run_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    # Dump the effective config and start a fresh metrics CSV for this run.
    with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    csv_path = results_dir / "metrics.csv"
    if csv_path.is_file():
        csv_path.unlink()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] run='{run_tag}' device={device} "
        f"benchmark={exp['benchmark']} data_tag='{exp['tag']}'\n"
        f"        params={n_params/1e6:.2f}M  batch_size={batch_size}  "
        f"epochs={epochs}  beta={beta:g}  lr={optim_cfg['lr']:g}\n"
        f"        te_true={data_meta.get('te_true'):.4f} nats  "
        f"(per-step {data_meta.get('te_per_step'):.4f})  "
        f"grad_checkpoint={config['model']['attention_grad_checkpoint']}\n"
        f"        results_dir={results_dir}"
    )

    best_val = float("inf")
    best_epoch = -1
    nan_train = {k: float("nan") for k in _TRAIN_KEYS}
    nan_train["nan_skips"] = 0
    train_m: Dict[str, float] = dict(nan_train)
    val_m: Dict[str, float] = {k: float("nan") for k in _EVAL_KEYS}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
            grad_clip_norm=grad_clip,
            likelihood=likelihood, sigma_obs=sigma_obs,
            free_bits=free_bits,
        )
        val_m = evaluate(
            model, val_loader, device,
            beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
            likelihood=likelihood, sigma_obs=sigma_obs,
            free_bits=free_bits,
        )
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        append_csv_row(
            csv_path, _assemble_row(epoch, train_m, val_m, lr, dt), _FIELDNAMES
        )
        skip_note = f"  SKIP={train_m['nan_skips']}" if train_m["nan_skips"] else ""
        print(
            f"  epoch {epoch:3d}/{epochs} | "
            f"train total={train_m['total_loss']:.4f} feat={train_m['feat_loss']:.4f} "
            f"base={train_m['base_loss']:.4f} kld={train_m['kld_loss']:.4f} "
            f"gap={train_m['pred_gap']:+.4f} | "
            f"val total={val_m['total_loss']:.4f} gap={val_m['pred_gap']:+.4f} | "
            f"sat={train_m['mu_prior_sat_frac']:.3f}/{train_m['delta_mu_sat_frac']:.3f} "
            f"lr={lr:.1e} {dt:.1f}s{skip_note}"
        )

        # Refresh the loss-curve figure (the CSV row above is already flushed).
        if plot_every > 0 and epoch % plot_every == 0:
            _refresh_training_curves(csv_path, run_tag, plotting_cfg)

        val_total = val_m["total_loss"]
        if math.isfinite(val_total) and val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            save_checkpoint(
                results_dir / "best.ckpt",
                model=model, model_kwargs=model_kwargs, config=config,
                data_meta=data_meta, epoch=epoch, val_loss=val_total,
                train_metrics=train_m, loss_settings=loss_settings,
                latent_stats_fitted=False,
            )

    # Replace the EMA latent-stats buffers with exact statistics.
    n_stats = _maybe_fit_latent_stats(model, train_loader, device)

    final_val = val_m["total_loss"]
    save_checkpoint(
        results_dir / "final.ckpt",
        model=model, model_kwargs=model_kwargs, config=config,
        data_meta=data_meta, epoch=epochs, val_loss=final_val,
        train_metrics=train_m, loss_settings=loss_settings,
        latent_stats_fitted=(n_stats > 0),
    )
    # Fallback: with no val split (or only non-finite val losses) the best
    # checkpoint was never written -- use the final model for it.
    if not (results_dir / "best.ckpt").is_file():
        best_epoch = epochs
        best_val = final_val
        save_checkpoint(
            results_dir / "best.ckpt",
            model=model, model_kwargs=model_kwargs, config=config,
            data_meta=data_meta, epoch=epochs, val_loss=final_val,
            train_metrics=train_m, loss_settings=loss_settings,
            latent_stats_fitted=(n_stats > 0),
        )

    # Final loss-curve refresh -- covers the case where ``epochs`` is not a
    # multiple of ``plot_every``.
    _refresh_training_curves(csv_path, run_tag, plotting_cfg)

    kbar_val = compute_kbar(model, val_loader, device)
    kbar_train = compute_kbar(model, train_loader, device, max_batches=8)

    results = {
        "run_tag": run_tag,
        "results_dir": str(results_dir),
        "best_epoch": best_epoch,
        "best_val_total_loss": best_val,
        "final_train_metrics": train_m,
        "final_val_metrics": val_m,
        "kbar_val": kbar_val,
        "kbar_train": kbar_train,
        "te_true": float(data_meta.get("te_true", float("nan"))),
        "best_ckpt": str(results_dir / "best.ckpt"),
        "final_ckpt": str(results_dir / "final.ckpt"),
    }
    print(
        f"[done] run='{run_tag}'  best epoch {best_epoch} "
        f"(val total {best_val:.4f})\n"
        f"       final pred_gap: train={train_m['pred_gap']:+.4f} "
        f"val={val_m['pred_gap']:+.4f}\n"
        f"       K_bar: val={kbar_val:.4f}  train={kbar_train:.4f}  "
        f"(te_true={results['te_true']:.4f} nats)\n"
        f"       checkpoints -> {results_dir}"
    )
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`. Every override defaults to
        ``None`` so an unspecified flag falls back to the config value.
    """
    p = argparse.ArgumentParser(
        description="Standalone training loop for SeqVaeLagAttnV1 on "
                    "synthetic transfer-entropy benchmark data."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--data-tag", type=str, default=None, dest="data_tag",
        help="override experiment.tag (which cached dataset to load)",
    )
    p.add_argument(
        "--run-tag", type=str, default=None, dest="run_tag",
        help="results subdirectory name (defaults to the data tag)",
    )
    p.add_argument("--beta", type=float, default=None, help="override loss.kld_beta")
    p.add_argument("--epochs", type=int, default=None, help="override optim.epochs")
    p.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="override optim.batch_size",
    )
    p.add_argument(
        "--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None,
        dest="grad_checkpoint",
        help="override model.attention_grad_checkpoint",
    )
    p.add_argument("--lr", type=float, default=None, help="override optim.lr")
    p.add_argument(
        "--a", type=float, default=None,
        help="record data.a (provenance only; does not regenerate data)",
    )
    p.add_argument(
        "--m", type=int, default=None,
        help="record data.M (provenance only; does not regenerate data)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    p.add_argument(
        "--plot-every", type=int, default=None, dest="plot_every",
        help="override plotting.plot_every (epochs between loss-curve "
             "refreshes; 0 = render only at the end)",
    )
    p.add_argument(
        "--data-dir", type=str, default=None, dest="data_dir",
        help="override paths.data_dir (absolute path, relative path, ~, or "
             "$VAR; resolved via train_minimal.resolve_user_path)",
    )
    p.add_argument(
        "--results-dir", type=str, default=None, dest="results_dir",
        help="override paths.results_dir (same format as --data-dir)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point: parse args, load config, run :func:`train`.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    config = load_config(args.config)
    overrides = {
        "data_tag": args.data_tag,
        "run_tag": args.run_tag,
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "a": args.a,
        "m": args.m,
        "grad_checkpoint": args.grad_checkpoint,
        "device": args.device,
        "seed": args.seed,
        "plot_every": args.plot_every,
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
    }
    train(config, overrides=overrides)


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision D9)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly;
    #                      no terminal flags required.
    #
    # Every key in RUN_CONFIG mirrors a CLI flag and is passed straight to
    # `train()` as `overrides`; `None` means "fall back to config_synth.yaml".
    # For any setting with no override key (e.g. a model kwarg such as `d_z`),
    # edit the loaded `config` dict directly at the marked spot below.
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "data_tag": "benchmark_A_easy_a1",  # cached dataset: data/<bench>/<tag>/
        "run_tag": "pol_easy_a1",           # output dir: results/<bench>/<tag>/
        "beta": None,                       # None -> config loss.kld_beta
        "epochs": 100,                      # full proof-of-life run length
        "batch_size": None,                 # None -> config optim.batch_size
        "lr": None,                         # None -> config optim.lr
        "a": 1.0,                           # provenance only (recorded+checked)
        "m": 87,                            # provenance only
        "grad_checkpoint": None,            # None -> config model.attention_grad_checkpoint
        "device": "cuda:0",                 # None/"auto" -> config runtime.device
        "seed": None,                       # None -> config experiment.seed
        "plot_every": None,                 # None -> config plotting.plot_every
        # Path overrides -- absolute path on any drive, ``~``, or ``$VAR``.
        # None -> use config_synth.yaml paths.data_dir / paths.results_dir.
        "data_dir": None,                   # e.g. r"D:/teb_data" or "~/teb"
        "results_dir": None,                # e.g. r"E:/teb_results"
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["model"]["d_z"] = 32
        # ---------------------------------------------------------------------
        train(config, overrides=RUN_CONFIG)

