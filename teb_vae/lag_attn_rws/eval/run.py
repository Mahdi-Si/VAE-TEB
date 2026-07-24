r"""The evaluation command line: a checkpoint in, ``summary.json`` out.

.. code-block:: bash

    python -m teb_vae.lag_attn_rws.eval.run --checkpoint <path> [--output-dir <dir>]

There is deliberately **no** ``--config``. A training run writes its fully resolved configuration
beside its checkpoints, and this reads that file, so the evaluation reconstructs the exact data
contract the run trained under. A second configuration file would be a second thing to keep in
step, and the failure mode of letting it drift -- evaluating on differently normalized or
differently trimmed data -- produces plausible numbers and no error.

Three things the run forces regardless of what that config says:

* **A single-process loader.** ``create_optimized_dataloader`` turns on persistent workers
  whenever ``num_workers > 0``, and spawn workers over a multi-file HDF5 dataset degrade after
  the first full pass, silently truncating later passes to the first file's index range. An
  evaluation makes many passes.
* **Evaluation mode and no gradient**, which the metrics module owns.
* **A timestamped output directory** at second resolution, because two evaluation runs in the
  same minute is normal while iterating.

The architecture is rebuilt from the checkpoint's own ``model_kwargs`` and the objective from its
``hyper_parameters``, so what is evaluated is what was trained rather than what a config file
currently says.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Repository root: ``teb_vae/lag_attn_rws/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_rws.eval.metrics import (  # noqa: E402
    DEFAULT_MIN_ACTIVE_DIMS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PRIOR_SHUFFLE_MIN_NATS,
    evaluate,
)
from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws  # noqa: E402
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask  # noqa: E402
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: The results subdirectory created inside every run directory.
RESULTS_DIRNAME = "eval_results"

#: The written summary's filename.
SUMMARY_FILENAME = "summary.json"

#: Seed for the derangement generator, so a re-run of the same checkpoint reproduces its controls.
_PERM_SEED = 1234


# =============================================================================
# Locating the run's own configuration
# =============================================================================
def resolved_config_for(checkpoint: Path) -> Path:
    """Find the resolved config the training run wrote for this checkpoint.

    A run's layout is ``<out_dir_base>/<stamp>-<tag>/model_checkpoints/<name>.ckpt``, and the
    config is written into that checkpoint directory. The run root and its ``train_results`` are
    searched as well, so a config placed there by hand is still found.

    Args:
        checkpoint: Path to the checkpoint being evaluated.

    Returns:
        Path to the resolved config.

    Raises:
        FileNotFoundError: Naming every location tried. A checkpoint moved out of its run
            directory has lost the record of what it was trained on, and evaluating it against a
            guessed configuration is worse than not evaluating it.
    """
    checkpoint = Path(checkpoint)
    run_root = checkpoint.parent.parent
    candidates = [
        checkpoint.parent / RESOLVED_CONFIG_FILENAME,
        run_root / "train_results" / RESOLVED_CONFIG_FILENAME,
        run_root / RESOLVED_CONFIG_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no {RESOLVED_CONFIG_FILENAME} found for checkpoint {checkpoint}. Tried: "
        + ", ".join(str(path) for path in candidates)
        + f". The training entry point writes it beside the checkpoints; a checkpoint copied "
        f"out of its run directory must be copied together with that file."
    )


def force_single_process_loader(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pin the dataloader to the calling process, mutating ``config`` in place.

    Not tuning. ``create_optimized_dataloader`` sets ``persistent_workers=True`` whenever
    ``num_workers > 0``, and with spawn multiprocessing over a multi-file HDF5 dataset those
    workers degrade after the first complete iteration -- later passes are silently truncated to
    the first file's index range. An evaluation iterates repeatedly, so it must not use them.

    Args:
        config: The resolved run config.

    Returns:
        The same config.
    """
    loader_config = config.setdefault("dataset_config", {}).setdefault("dataloader_config", {})
    requested = loader_config.get("num_workers", 0)
    if requested:
        logger.warning(
            f"dataloader_config.num_workers={requested} overridden to 0: spawn workers over a "
            f"multi-file HDF5 dataset silently truncate every pass after the first, and an "
            f"evaluation makes many passes."
        )
    loader_config["num_workers"] = 0
    loader_config["persistent_workers"] = False
    return config


def make_output_dir(config: Dict[str, Any], explicit: Optional[Any] = None) -> Path:
    """Create and return the results directory for this evaluation run.

    Args:
        config: The resolved run config, for its ``out_dir_base`` and ``tag``.
        explicit: An explicit run directory from ``--output-dir``, used as given.

    Returns:
        The ``eval_results`` directory inside the run directory.
    """
    if explicit is not None:
        results_dir = Path(explicit) / RESULTS_DIRNAME
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    general = config.get("general_config") or {}
    base = Path(str((general.get("folders_config") or {}).get("out_dir_base", "output")))
    tag = str(general.get("tag", "lag_attn_rws"))
    # Second resolution, unlike training's minute stamp: two evaluation runs in the same minute
    # is normal, and the numeric suffix below is the backstop for the same second.
    stamp = datetime.now().strftime("%Y-%m-%d--[%H-%M-%S]")
    candidate = base / f"{tag}-eval" / stamp
    suffix = 2
    while candidate.exists():
        candidate = base / f"{tag}-eval" / f"{stamp}-{suffix}"
        suffix += 1
    results_dir = candidate / RESULTS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# =============================================================================
# JSON safety
# =============================================================================
def json_safe(value: Any) -> Any:
    """Convert a value into something ``json.dump`` can write and a reader can parse.

    Applied *before* serialisation rather than passed as ``default=``, which is consulted only
    for types the encoder does not already recognise -- and ``float('nan')`` is recognised.
    Left alone, ``json.dump`` emits the bare token ``NaN``, which round-trips through Python and
    is rejected by every other JSON parser. Non-finite floats therefore become ``null``.

    Args:
        value: Anything.

    Returns:
        A JSON-serialisable structure.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    # Before the int branch: numpy's bool is not a Python bool and would serialise as 0 or 1.
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # Anything else is recorded by its repr rather than dropped, so an unexpected type shows up
    # in the output instead of failing the write at the end of a long run.
    return str(value)


# =============================================================================
# Model reconstruction
# =============================================================================
def load_task(checkpoint_path: Path, device: torch.device) -> SeqVaeLagAttnRwsTask:
    """Rebuild the net and its task from a checkpoint, and load the weights.

    The order is load-bearing: the class guard runs before construction, because the net's
    constructor is keyword-only and another model's ``model_kwargs`` would otherwise surface as a
    ``TypeError`` naming a parameter rather than as a message naming both classes.

    Args:
        checkpoint_path: Path to the checkpoint.
        device: Device to place the model on.

    Returns:
        The task, in evaluation mode on ``device``.

    Raises:
        FileNotFoundError: If the checkpoint is not there.
        RuntimeError: If it carries no ``model_kwargs``, no ``hyper_parameters``, or if its state
            dict does not align into the rebuilt model. ``load_checkpoint_strict`` returns
            ``None`` rather than raising, so an unchecked call would evaluate randomly
            initialised weights and report the result as a measurement.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    check_model_class(blob, SeqVaeLagAttnRws.__name__)

    model_kwargs = blob.get("model_kwargs") if isinstance(blob, dict) else None
    if not model_kwargs:
        raise RuntimeError(
            f"checkpoint {str(checkpoint_path)!r} carries no 'model_kwargs', so the architecture "
            f"cannot be rebuilt. SeqVaeLagAttnRws() with no arguments builds the production "
            f"geometry rather than raising, so guessing would silently evaluate the wrong model."
        )
    hparams = blob.get("hyper_parameters") if isinstance(blob, dict) else None
    if not hparams:
        raise RuntimeError(
            f"checkpoint {str(checkpoint_path)!r} carries no 'hyper_parameters', so the "
            f"likelihood and loss weights the run trained under are unknown; scoring it under "
            f"assumed defaults would report a different objective's numbers."
        )

    model = SeqVaeLagAttnRws(**model_kwargs)
    task = SeqVaeLagAttnRwsTask(
        model,
        model_kwargs=model_kwargs,
        beta_schedule=hparams.get("beta_schedule"),
        kld_beta=hparams.get("kld_beta", 1.0),
        lambda_full=hparams.get("lambda_full", 1.0),
        lambda_base=hparams.get("lambda_base", 1.0),
        likelihood=hparams.get("likelihood", "gaussian_nll"),
        free_bits=hparams.get("free_bits", 0.0),
    )
    if load_checkpoint_strict(model=task.orig_model, checkpoint=blob) is None:
        raise RuntimeError(
            f"could not align checkpoint {str(checkpoint_path)!r} into SeqVaeLagAttnRws: no "
            f"module matched its state dict. Evaluating would otherwise proceed on randomly "
            f"initialised weights and report the numbers as a result."
        )
    task.to(device)
    task.eval()
    return task


def resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve the evaluation device.

    Args:
        requested: An explicit device string, or ``None`` to choose automatically.

    Returns:
        The device: what was asked for, else CUDA when available, else CPU.
    """
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Entry point
# =============================================================================
def main(
    checkpoint: Any,
    output_dir: Optional[Any] = None,
    *,
    device: Optional[str] = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    max_batches: Optional[int] = None,
    prior_shuffle_min_nats: float = DEFAULT_PRIOR_SHUFFLE_MIN_NATS,
    min_active_dims: int = DEFAULT_MIN_ACTIVE_DIMS,
) -> Path:
    """Evaluate a checkpoint and write ``summary.json``.

    Args:
        checkpoint: Path to the checkpoint.
        output_dir: Run directory; a timestamped one is created when omitted.
        device: Torch device string; chosen automatically when omitted.
        num_samples: Monte Carlo draws $K$.
        max_batches: Stop after this many batches, for a smoke run.
        prior_shuffle_min_nats: Verdict margin for the prior-shuffle control.
        min_active_dims: Verdict threshold for latent collapse.

    Returns:
        Path to the written ``summary.json``.
    """
    checkpoint_path = Path(checkpoint)
    config_path = resolved_config_for(checkpoint_path)
    logger.info(f"evaluating {checkpoint_path} against {config_path}")

    # load_config is a no-op on an already-resolved file (it carries no `base:` key) and is used
    # anyway so there is exactly one config reader in the tree.
    config = load_config(str(config_path))
    force_single_process_loader(config)

    results_dir = make_output_dir(config, output_dir)
    logger.info(f"writing results to {results_dir}")
    with open(results_dir / RESOLVED_CONFIG_FILENAME, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    resolved_device = resolve_device(device)
    task = load_task(checkpoint_path, resolved_device)
    loader = GraphDataModule(config).test_dataloader()

    # The model is what was trained, so it -- not the config -- is asked what guard it carries.
    delay_steps = int(task.orig_model.source_delay_steps)
    if delay_steps:
        logger.info(
            f"source channels are delayed by up to {delay_steps} steps "
            f"({delay_steps * SECONDS_PER_STEP:g} s); the reported lag adds this back as an "
            f"upper bound."
        )

    generator = torch.Generator()
    generator.manual_seed(_PERM_SEED)
    results = evaluate(
        task,
        loader,
        num_samples=num_samples,
        max_batches=max_batches,
        perm_generator=generator,
        delay_steps=delay_steps,
        prior_shuffle_min_nats=prior_shuffle_min_nats,
        min_active_dims=min_active_dims,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "output_dir": str(results_dir),
        "device": str(resolved_device),
        # Per-channel delays have no single representative; the maximum is used, so the reported
        # lag is an upper bound. Recorded beside the number so the choice travels with it.
        "source_delay_steps": delay_steps,
        "source_delay_is_max_over_channels": True,
        "results": results,
    }

    summary_path = results_dir / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        # allow_nan=False so an unsanitised non-finite value raises here rather than producing a
        # file only Python can read back.
        json.dump(json_safe(summary), handle, indent=2, allow_nan=False)
    logger.info(f"wrote {summary_path}")
    for verdict in results["verdicts"]:
        logger.info(f"{verdict['status']:<13} {verdict['name']}: {verdict['criterion']}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_rws.eval.run",
        description="Evaluate a trained SeqVaeLagAttnRws checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Run directory. Default: a timestamped directory under out_dir_base/<tag>-eval.",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device. Default: cuda:0 when available, else cpu."
    )
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help="Monte Carlo draws per anchor for the predictive scores.",
    )
    parser.add_argument(
        "--max-batches", dest="max_batches", type=int, default=None,
        help="Stop after this many batches. For a smoke run only.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # The shard paths inside a resolved config are repo-root-relative for the tiny variant,
        # and a relative path resolved against an arbitrary working directory surfaces as "no
        # samples match the specified filters" with no mention of the real cause.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    main(
        args.checkpoint,
        args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        max_batches=args.max_batches,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
