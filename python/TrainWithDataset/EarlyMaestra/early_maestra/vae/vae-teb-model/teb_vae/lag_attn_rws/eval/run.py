r"""The evaluation command line: a checkpoint in, ``summary.json`` out.

.. code-block:: bash

    python -m teb_vae.lag_attn_rws.eval.run --checkpoint <path> [--output-dir <dir>]

**There is no second config, only a delta.** A training run writes its fully resolved
configuration beside its checkpoints, and that file -- not a committed YAML -- is the record of
what the model was trained on. The evaluation reads it and deep-merges the committed override
delta over the top, so the run's own contract stays authoritative and everything that genuinely
differs (the holdout shards, the five extra ``load_fields`` the clinical questions are asked in,
and the ``eval_config`` block) is one small reviewable file. A ``base:`` chain cannot do that
job: ``base:`` resolves relative to the file naming it, and a checkpoint's resolved config is a
runtime path no committed file can reference.

Four things the run forces regardless of what either file says:

* **A single-process loader.** ``create_optimized_dataloader`` turns on persistent workers
  whenever ``num_workers > 0``, and spawn workers over a multi-file HDF5 dataset degrade after
  the first full pass, silently truncating later passes to the first file's index range. An
  evaluation makes many passes.
* **A fixed-seed shuffle over that loader.** The test split is eight per-subgroup shards read in
  order, so an unshuffled batch holds consecutive segments of one recording -- and the
  permutation control then has no stranger in the batch to borrow a source from. The shuffle is
  drawn from the run's own seed, so it is a reordering, not a source of run-to-run variation.
* **Evaluation mode and no gradient**, which the metrics module owns.
* **A timestamped output directory** at second resolution, because two evaluation runs in the
  same minute is normal while iterating.

The architecture is rebuilt from the checkpoint's own ``model_kwargs`` and the objective from its
``hyper_parameters``, so what is evaluated is what was trained rather than what a config file
currently says -- and preflight refuses the run outright when the merged config contradicts
either.

**Preflight runs outside any failure isolation.** A refused run leaves its *inputs* -- the merged
configuration and the log holding the refusal -- and no ``summary.json``: a rejected input must
not produce a file that reads like a result, and a refusal an operator cannot read afterwards is
half a refusal.

**The forward pass happens once per run directory.** It writes two durable tables --
``per_sample.csv`` and ``per_anchor.parquet`` -- beside the summary, and a provenance sidecar
naming the checkpoint, the seed and the ``eval_config`` they were collected under. A later
invocation into the same directory reads them back instead of decoding four branches over every
anchor again, and refuses outright when the sidecar describes a different run.

**Every number a re-run must reproduce comes from one seed.** ``configure_numerics`` pins the
numeric environment and seeds the global generators; the Monte Carlo draw, the derangement and
the loader shuffle each take an explicit generator derived from the same value, so the readouts
of two runs of one checkpoint agree exactly rather than approximately.

**Every analysis runs inside a wrapper that captures its failure.** An evaluation is a sequence of
largely independent readings of one checkpoint and it can take hours; one of them raising must not
discard the ones that already succeeded, and must also not be mistakable for a clean run. So each
runs through ``Report.step``, each writes its own artifacts as it finishes, ``summary.json`` is
assembled from whatever exists at the end, and the process exits non-zero. Preflight is the
deliberate exception: it runs outside the wrapper, because a refused input must not produce a file
that reads like a result.

**A checkpoint is not required.** Everything after the collection pass reads the tables rather
than the model, so ``--output-dir <a finished run>`` with no ``--checkpoint`` re-runs the analyses
offline, with no model built and no GPU touched -- which is the point of splitting collection from
emission, and the form a re-run takes after a multi-hour pass failed at its ninth step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_rws/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler, Subset  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_rws.eval import cohort, collect, preflight, probe as probe_module  # noqa: E402
from teb_vae.lag_attn_rws.eval._reuse import configure_numerics, subsample_indices  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import (  # noqa: E402
    GROUPED_FRAMES_KEY,
    AnalysisContext,
)
from teb_vae.lag_attn_rws.eval.analyses import band_partition as band_partition_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import calibration as calibration_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import coupling as coupling_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import attention as attention_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import cross_subgroup as cross_subgroup_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import events as events_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import forecast as forecast_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import lag_kl as lag_kl_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import latent as latent_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import perm_control as perm_control_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import residual as residual_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import samples as samples_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import sufficiency as sufficiency_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import time_to_delivery as time_to_delivery_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.analyses import trajectory as trajectory_analysis  # noqa: E402
from teb_vae.lag_attn_rws.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides_with_provenance,
    validate_eval_config,
)
from teb_vae.lag_attn_rws.eval.figures_seam import configure_figure_style  # noqa: E402

# Re-exported rather than reimplemented: one serialiser writes every summary this repository
# produces, so a value that survives a round trip in one package cannot fail the write in the
# other. Named here because ``summary.json`` is written from this module.
from teb_vae.lag_attn_rws.eval.report_seam import (  # noqa: E402,F401
    STEPS_FILENAME,
    SUMMARY_FILENAME,
    Report,
    console_summary,
    emit_grouped_variants,
    finalise,
    json_safe,
    step_records,
    write_steps,
)
from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws  # noqa: E402
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask  # noqa: E402
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: The results subdirectory created inside every run directory.
RESULTS_DIRNAME = "eval_results"

#: The run's log file, beside the artifacts it explains.
LOG_FILENAME = "eval.log"

#: Steps that always run, in run order, and are **not** selectable.
#:
#: The input channel map belongs here rather than on the registry below because it describes the
#: *data* rather than the model: it reads the shards' own ``sel_*`` provenance, needs no forward
#: pass and no tables, and is the data-side companion to the causality disclosure. A run whose
#: channel map could be skipped would be a run whose frequency-resolved statements have no
#: definition of a band behind them.
UNSKIPPABLE_ANALYSES: Dict[str, Any] = {
    "band_partition": band_partition_analysis.run_band_partition_analysis,
}

#: Analyses ``--only`` and ``--skip`` select between, in run order.
#:
#: Registering one is a line here: every entry takes the same arguments and returns the same four
#: keys, which is what the protocol in ``analyses/__init__.py`` exists to guarantee. The rest of
#: the readouts a run reports come from the shared collection pass rather than from a registered
#: analysis, and the analyses that read its tables land sprint by sprint.
#:
#: There is deliberately no dependency table beside it. The sibling pipeline's is still empty
#: after fourteen analyses, and its comment gives the reason this design makes even more true: the
#: real dependency is on *files existing on disk* rather than on an analysis having run in this
#: pass, which is exactly what makes ``--only <name> --output-dir <a finished run>`` work at all.
#: One line adds the table the day a genuine correctness dependency appears.
#: ``cross_subgroup`` is deliberately **last**, and that ordering is load-bearing: it reads the
#: per-recording CSVs the analyses above it write, so on a single pass it can only test the
#: metrics they have already produced. It does not *depend* on them having run in this pass -- a
#: source that is absent is recorded rather than raised, which is what keeps ``--only`` working --
#: but a run of everything should test everything.
ANALYSIS_FUNCTIONS: Dict[str, Any] = {
    "forecast": forecast_analysis.run_forecast_analysis,
    "coupling": coupling_analysis.run_coupling_analysis,
    "perm_control": perm_control_analysis.run_perm_control_analysis,
    "latent": latent_analysis.run_latent_analysis,
    "lag_kl": lag_kl_analysis.run_lag_kl_analysis,
    "attention": attention_analysis.run_attention_analysis,
    "calibration": calibration_analysis.run_calibration_analysis,
    "residual": residual_analysis.run_residual_analysis,
    "trajectory": trajectory_analysis.run_trajectory_analysis,
    "time_to_delivery": time_to_delivery_analysis.run_time_to_delivery_analysis,
    "events": events_analysis.run_events_analysis,
    # Last but for the two below, and deliberately so: it is the only analysis whose cost is a
    # training loop rather than a forward pass, so a run that fails earlier fails before paying
    # for it -- and everything it reports is a comparison against readouts the pass already has.
    "sufficiency": sufficiency_analysis.run_sufficiency_analysis,
    "samples": samples_analysis.run_samples_analysis,
    "cross_subgroup": cross_subgroup_analysis.run_cross_subgroup_analysis,
}

#: Selectable analysis names, in run order. Published for the command line's help text and for a
#: reader; the run itself derives the same tuple from the registry when it selects, so the two
#: cannot disagree about what is registered.
ANALYSES: Tuple[str, ...] = tuple(ANALYSIS_FUNCTIONS)

#: Offsets applied to ``eval_config.seed`` for the four explicit generators. All non-zero and
#: distinct, so the sample cap's draw, the loader's shuffle, the derangement and the Monte Carlo
#: draw are five independent streams together with the global one ``configure_numerics`` seeds --
#: a generator seeded with the bare seed would replay exactly the numbers the global stream is
#: already handing to the model's own reparameterisation. Fixed, so all still follow from one
#: recorded value.
_SEED_OFFSET_LOADER, _SEED_OFFSET_PERM, _SEED_OFFSET_MC = 1, 2, 3
_SEED_OFFSET_SAMPLE_CAP = 4


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


def finished_run_config(output_dir: Optional[Any]) -> Path:
    """Find the config a finished run dumped, for a pass that was given no checkpoint.

    The dumped merged configuration is the record of what that run evaluated, so an offline
    re-run reads it rather than re-deriving one -- there is no checkpoint to find a resolved
    config beside, and rebuilding the merge from the committed delta would evaluate against what
    a config file says today rather than against what the tables in this directory were collected
    under.

    Args:
        output_dir: The run directory named by ``--output-dir``.

    Returns:
        Path to the dumped config.

    Raises:
        FileNotFoundError: If there is no such directory or no config in it. Both mean the same
            thing: there is nothing here to re-run, and a checkpoint is required after all.
    """
    if output_dir is None:
        raise FileNotFoundError(
            "--checkpoint is required unless --output-dir names a finished run directory to "
            "re-run the analyses against."
        )
    candidate = Path(output_dir) / RESULTS_DIRNAME / RESOLVED_CONFIG_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(
            f"--checkpoint is required unless --output-dir names a finished run directory: "
            f"{candidate} does not exist, so this directory holds no evaluation to re-run. A "
            f"finished run leaves that file beside its tables and its summary."
        )
    return candidate


def preserve_prior_summary(results_dir: Path) -> List[Path]:
    r"""Rename a finished run's summary aside before this pass overwrites it.

    ``--output-dir <a finished run>`` is the documented way to re-run analyses against an existing
    directory, and it is also destructive: the summary is opened with mode ``'w'`` and the
    artifact manifest classifies every earlier file as stale, so a one-analysis re-run replaces a
    complete summary with one whose headline is mostly ``null`` and whose manifest lists four
    files instead of forty -- and exits $0$. The tables survive that, but the sanity block, the
    coverage record and the promoted verdicts exist nowhere else.

    Renaming rather than merging is the smaller fix: the new pass still writes a truthful summary
    of *itself*, and the prior one is recoverable byte for byte.

    ``preflight.json`` is deliberately **not** preserved. A run with a checkpoint rewrites it, and
    a run without one reads it back for the causality disclosure -- renaming it aside would take
    that away from exactly the pass that cannot regenerate it. It is stable within a directory
    anyway: a different checkpoint's tables are refused before it is ever read.

    Args:
        results_dir: The ``eval_results`` directory this pass is about to write into.

    Returns:
        The backup paths created, in the order attempted. Empty when the directory is fresh.
    """
    stamp = datetime.now().strftime("%Y-%m-%d--[%H-%M-%S]")
    preserved: List[Path] = []
    for filename in (SUMMARY_FILENAME, STEPS_FILENAME):
        existing = results_dir / filename
        if not existing.is_file():
            continue
        backup = results_dir / f"{existing.stem}.bak.{stamp}{existing.suffix}"
        suffix = 2
        while backup.exists():
            backup = results_dir / f"{existing.stem}.bak.{stamp}-{suffix}{existing.suffix}"
            suffix += 1
        existing.rename(backup)
        preserved.append(backup)
        logger.warning(
            f"{filename} already existed in {results_dir} and this pass would overwrite it; "
            f"preserved the prior run's copy as {backup.name}. A partial re-run's summary "
            f"describes only the analyses this pass actually ran."
        )
    return preserved


def make_output_dir(config: Dict[str, Any], explicit: Optional[Any] = None) -> Path:
    """Create and return the results directory for this evaluation run.

    An explicit directory is used as given, which is what makes the documented offline re-run
    possible -- and also what makes it destructive, so any prior summary is preserved first. The
    timestamped default cannot collide and needs no such guard.

    Args:
        config: The resolved run config, for its ``out_dir_base`` and ``tag``.
        explicit: An explicit run directory from ``--output-dir``, used as given.

    Returns:
        The ``eval_results`` directory inside the run directory.
    """
    if explicit is not None:
        results_dir = Path(explicit) / RESULTS_DIRNAME
        results_dir.mkdir(parents=True, exist_ok=True)
        preserve_prior_summary(results_dir)
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


def dump_resolved_config(config: Dict[str, Any], results_dir: Path) -> Path:
    """Write the merged config into the run directory as its durable record.

    The merged result, not the delta and not the run's own file: what the evaluation actually ran
    under is one document, and it is the one a reader should be able to hand back to
    ``load_config`` unchanged.

    Args:
        config: The merged run config.
        results_dir: The results directory.

    Returns:
        The path written.
    """
    path = results_dir / RESOLVED_CONFIG_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def configure_logging(results_dir: Path, config: Dict[str, Any]) -> int:
    """Add a file sink beside the run's artifacts, keeping the default console sink.

    Args:
        results_dir: The results directory.
        config: The merged run config, for its ``advanced_config.logging`` levels.

    Returns:
        The sink id, so the caller can remove it when the run ends. A run that left its sink
        attached would keep writing a *later* run's lines into this run's log file.
    """
    logging_config = (config.get("advanced_config") or {}).get("logging") or {}
    return logger.add(
        str(results_dir / LOG_FILENAME),
        level=str(logging_config.get("file_level", "INFO")),
        rotation=str(logging_config.get("rotation", "100 MB")),
        retention=str(logging_config.get("retention", "14 days")),
        enqueue=False,
    )


def dataset_shard_keys(loader: DataLoader) -> Optional[List[str]]:
    """Return the source shard of every dataset index, or ``None`` when it cannot be read.

    The stratum a sample cap must be drawn within. Taken from the dataset's own file index rather
    than from a batch, because the draw has to be resolved before anything is loaded.

    Args:
        loader: The evaluation dataloader.

    Returns:
        One shard basename per dataset index, or ``None`` when the dataset does not expose its
        file layout -- which a caller reads as "draw without strata", never as "draw a prefix".
    """
    dataset = getattr(loader, "dataset", None)
    index_map = getattr(dataset, "index_map", None)
    paths = getattr(dataset, "paths", None)
    if not index_map or not paths:
        return None
    try:
        return [Path(str(paths[int(file_index)])).name for file_index, _ in index_map]
    except (IndexError, TypeError, ValueError):
        return None


def capped_sample_loader(
    loader: DataLoader, cap: Optional[int], seed: int
) -> Tuple[DataLoader, Dict[str, Any]]:
    """Restrict the loader to a seeded **stratified** draw of at most ``cap`` samples.

    Never a prefix. The test loader is built ``shuffle=False`` over eight concatenated
    per-subgroup files, so a prefix cap yields one subgroup and one clinical class -- the
    predecessor's documented "only 1 class found" failure. Stratifying by shard gives every file
    a share of the cap proportional to its size, which guarantees each one appears whenever the
    cap is at least the shard count rather than merely making it likely.

    Args:
        loader: The loader the data module built.
        cap: Maximum samples to evaluate. ``None``, or a cap at or above the split size, returns
            the loader untouched.
        seed: Seed, so a re-run draws the same samples.

    Returns:
        ``(loader, record)``. The record says what the cap did -- including that it did nothing --
        so a summary never has to be read as though the whole split was seen when it was not.
    """
    n_total = len(loader.dataset)  # type: ignore[arg-type]
    keys = dataset_shard_keys(loader)
    indices = subsample_indices(n_total, cap, seed, groups=keys)
    if indices is None:
        return loader, {
            "max_samples": None if cap is None else int(cap),
            "applied": False,
            "n_total": int(n_total),
            "n_drawn": int(n_total),
            "stratified_by": "source_file_basename" if keys else None,
        }
    drawn = [int(value) for value in indices]
    record = {
        "max_samples": int(cap),
        "applied": True,
        "n_total": int(n_total),
        "n_drawn": len(drawn),
        # Named rather than implied: an unstratified draw is still not a prefix, but it does not
        # carry the coverage guarantee, and the difference matters when a shard goes missing.
        "stratified_by": "source_file_basename" if keys else None,
        "n_shards_drawn": len({keys[index] for index in drawn}) if keys else None,
    }
    logger.info(
        f"eval_config.max_samples: evaluating {len(drawn)} of {n_total} sample(s), "
        f"stratified by {'shard' if keys else 'nothing available'}"
    )
    return (
        DataLoader(
            Subset(loader.dataset, drawn),
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=loader.collate_fn,
        ),
        record,
    )


def shuffled_control_loader(loader: DataLoader, seed: int) -> DataLoader:
    """Return the same dataset, re-batched under a fixed-seed shuffle.

    The permutation control pairs each target with another *sample in its own batch*, so which
    samples share a batch is part of the measurement rather than an implementation detail. The
    test loader is unshuffled over eight concatenated per-subgroup files and one recording
    contributes tens of consecutive segments, so an unshuffled batch is frequently one or two
    recordings -- and a batch holding one recording has no stranger in it at all, which costs the
    control that batch entirely.

    Shuffling is a reordering, not a resampling: every sample is still scored exactly once, and
    the per-recording aggregation does not depend on the order. The generator is explicit so the
    reordering is a function of the run's seed rather than of whatever else drew from the global
    stream first.

    Args:
        loader: The loader the data module built, read for its dataset, batch size and collation.
        seed: Seed for the shuffle.

    Returns:
        A new single-process loader over the same dataset.
    """
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=RandomSampler(loader.dataset, generator=torch.Generator().manual_seed(int(seed))),
        num_workers=0,
        drop_last=False,
        collate_fn=loader.collate_fn,
    )


# =============================================================================
# Model reconstruction
# =============================================================================
def read_checkpoint(checkpoint_path: Any) -> Dict[str, Any]:
    """Read a checkpoint blob off disk.

    Separated from :func:`load_task` so a run reads a multi-gigabyte file once and then answers
    both questions from it: what to rebuild, and what the rebuilt thing must be reconciled
    against.

    Args:
        checkpoint_path: Path to the checkpoint.

    Returns:
        The blob.

    Raises:
        FileNotFoundError: If the checkpoint is not there.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)


def load_task(
    checkpoint_path: Path, device: torch.device, *, blob: Optional[Dict[str, Any]] = None
) -> SeqVaeLagAttnRwsTask:
    """Rebuild the net and its task from a checkpoint, and load the weights.

    The order is load-bearing: the class guard runs before construction, because the net's
    constructor is keyword-only and another model's ``model_kwargs`` would otherwise surface as a
    ``TypeError`` naming a parameter rather than as a message naming both classes.

    Args:
        checkpoint_path: Path to the checkpoint.
        device: Device to place the model on.
        blob: An already-read checkpoint, so a caller that needed its ``model_kwargs`` before
            construction does not pay a second read. Read from ``checkpoint_path`` when omitted.

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
    if blob is None:
        blob = read_checkpoint(checkpoint_path)
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
# Analysis selection
# =============================================================================
def select_analyses(
    available: Sequence[str], only: Optional[str], skip: Optional[str]
) -> List[str]:
    """Resolve ``--only`` / ``--skip`` into the ordered list of analyses to run.

    Args:
        available: Every selectable analysis, in run order.
        only: Comma-separated names to run exclusively, or ``None`` for all of them.
        skip: Comma-separated names to skip, or ``None``.

    Returns:
        The analyses to run, in ``available`` order -- not in the order they were typed. The run
        order is the pipeline's, and a later analysis may read what an earlier one wrote.

    Raises:
        ValueError: If a named analysis does not exist. A misspelling would otherwise silently
            run everything (``--only``) or nothing extra (``--skip``), which is indistinguishable
            in the output from having asked for exactly that.
    """
    known = list(available)

    def _parse(raw: Optional[str], flag: str) -> List[str]:
        if not raw:
            return []
        names = [name.strip() for name in str(raw).split(",") if name.strip()]
        unknown = [name for name in names if name not in known]
        if not unknown:
            return names
        unskippable = [name for name in unknown if name in UNSKIPPABLE_ANALYSES]
        detail = (
            f" {unskippable} always run(s) and is not selectable."
            if unskippable else ""
        )
        raise ValueError(
            f"--{flag} names unknown analyses: {unknown}. Available: "
            f"{known or '(none registered yet)'}.{detail}"
        )

    requested = set(_parse(only, "only")) or set(known)
    excluded = set(_parse(skip, "skip"))
    return [name for name in known if name in requested and name not in excluded]


# =============================================================================
# The analysis loop
# =============================================================================
def emit_grouped_for(result: Dict[str, Any], output_dir: Any) -> Dict[str, Any]:
    """Fan the by-class and by-subgroup variants over whatever an analysis declared.

    **The fan-out is the runner's job, not each analysis's.** The alternative is a cross-cutting
    edit that every analysis added later has to remember to make, and the one that forgets emits
    a pooled number over a mixed cohort with nothing saying so. An analysis therefore *declares*
    a per-sample CSV and the columns worth resolving by group, and this reads it.

    A declared path is resolved **against the results directory** when it is relative, which is
    the form :func:`~teb_vae.lag_attn_rws.eval.frames.grouped_frame_entry` writes: an absolute
    path would put a machine-specific string into the summary, and the emitter's own record is
    relativised on the way back out for the same reason.

    Args:
        result: One analysis's return value, read for its optional grouped-frames declaration.
        output_dir: The results directory, which relative declarations resolve against.

    Returns:
        Frame stem to the emitter's record, empty when the analysis declared nothing. Never
        raises: a grouped variant is an addition to a run, and an analysis whose pooled output
        succeeded must not be marked failed because its split turned out to hold one cohort.
    """
    import pandas as pd

    base = Path(output_dir)
    declared = result.get(GROUPED_FRAMES_KEY) or []
    emitted: Dict[str, Any] = {}
    for entry in declared:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        stem = str(entry.get("stem") or path.stem)
        directory = Path(entry.get("directory") or path.parent)
        if not directory.is_absolute():
            directory = base / directory
        try:
            frame = pd.read_csv(path)
            emitted[stem] = _relative_files(
                emit_grouped_variants(
                    frame,
                    directory,
                    value_columns=list(entry.get("value_columns") or []),
                    stem=stem,
                    references=entry.get("references"),
                ),
                base,
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.error(
                f"could not emit grouped variants for {path}: {type(exc).__name__}: {exc}. "
                f"The pooled output stands."
            )
            emitted[stem] = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
    return emitted


def _relative_files(emitted: Dict[str, Any], base: Path) -> Dict[str, Any]:
    """Rewrite an emitter record's absolute paths as paths inside the run directory.

    The shared emitter reports what it wrote as absolute paths, which is right for a caller
    holding them and wrong for ``summary.json``: that block is one two runs of a checkpoint must
    compare equal, and an absolute path differs between two runs of the same checkpoint for no
    reason but where they were written.

    Args:
        emitted: The emitter's per-axis record.
        base: The results directory the paths are made relative to.

    Returns:
        The same record with its ``files`` entries relativised. A path outside the run directory
        -- which nothing here produces -- is left as it is rather than mangled.
    """
    for record in emitted.values():
        files = record.get("files") if isinstance(record, dict) else None
        if not isinstance(files, dict):
            continue
        for name, value in list(files.items()):
            try:
                files[name] = Path(value).relative_to(base).as_posix()
            except ValueError:
                continue
    return emitted


def run_analyses(
    report: Report,
    names: Sequence[str],
    functions: Dict[str, Any],
    *,
    context: AnalysisContext,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> None:
    """Run each analysis under the failure-isolating wrapper, in order.

    Each writes its own artifacts as it finishes and the step heartbeat is rewritten after every
    one, so a run killed outright still says how far it got; ``summary.json`` is then assembled
    from whatever reached ``report.results``. A failure costs the remaining *blocks* of that
    analysis, not the run.

    Args:
        report: The run's report, which owns the step wrapper and the accumulated results.
        names: The analyses to run, in order.
        functions: Name to callable.
        context: What every analysis is given about the run.
        eval_config: The validated ``eval_config`` block.
        output_dir: The results directory.
        probe: The loader probe's record, or ``None`` when this pass did not build a loader.
    """
    for name in names:
        result = report.step(
            name,
            functions[name],
            context,
            eval_config=eval_config,
            output_dir=output_dir,
            probe=probe,
        )
        if isinstance(result, dict):
            grouped = emit_grouped_for(result, output_dir)
            report.set(name, {**result, "grouped": grouped} if grouped else result)
        write_steps(report.steps, output_dir)


# =============================================================================
# Run context
# =============================================================================
def _beta_end(weights: Dict[str, Any]) -> float:
    r"""Resolve the end-of-training $\beta$ from a run's objective weights.

    The same resolution rule the task applies at its final epoch -- ``linear_warmup`` ends at
    ``beta_schedule.end``, ``constant`` reads ``value`` and falls back to ``kld_beta`` -- so the
    loss-scale estimate below weighs the KL the way the finished run actually did.

    Args:
        weights: The checkpoint's ``hyper_parameters`` (or the config's ``VAE_model`` block on a
            pass with no checkpoint).

    Returns:
        The scalar $\beta$ in force after any warm-up.
    """
    schedule = weights.get("beta_schedule")
    if isinstance(schedule, dict):
        if str(schedule.get("kind", "constant")) == "linear_warmup":
            return float(schedule.get("end", weights.get("kld_beta", 1.0)))
        value = schedule.get("value")
        if value is not None:
            return float(value)
    return float(weights.get("kld_beta", 1.0))


def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` when it is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_run_context(
    *,
    task: Optional[SeqVaeLagAttnRwsTask],
    blob: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    collection: collect.Collection,
) -> Dict[str, Any]:
    r"""Record the run facts the calibration study's tables consume.

    Four things a summary must carry so the arm comparison and the first-run checklist are
    arithmetic rather than archaeology: the parameter count (the architecture arms are judged
    per parameter), the checkpoint's training epoch, the anchor-coverage distribution the
    ``coverage_floor`` is confirmed or revised against, and the observed magnitude of the
    training objective the spike breaker's ``additive_margin`` is re-derived from.

    Outside ``results`` deliberately: the first two are facts about the *checkpoint*, absent on
    a pass that built no model, and ``results`` is the block an offline re-run must reproduce
    byte for byte.

    Args:
        task: The loaded task, or ``None`` on a pass with no checkpoint.
        blob: The checkpoint blob, read for its training epoch. ``None`` without a checkpoint.
        config: The merged run config, the fallback source of the objective weights.
        collection: The shared pass's output, read for the per-anchor coverage column and the
            aggregated readouts.

    Returns:
        The run-context block for ``summary.json``.
    """
    import numpy as np

    coverage: Optional[Dict[str, Any]] = None
    per_anchor = collection.per_anchor
    if per_anchor is not None and len(per_anchor) and "coverage" in per_anchor.columns:
        values = np.asarray(per_anchor["coverage"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            coverage = {
                "n_anchors": int(values.size),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(values.max()),
                "frac_at_one": float(np.mean(values >= 1.0)),
                # The floor truncates what this distribution can show, and a reader revising the
                # floor from it must know that.
                "note": (
                    "over contributing anchors only: an anchor whose valid fraction falls below "
                    "the model's coverage_floor is never scored and has no row here"
                ),
            }

    # The checkpoint's own hyperparameters when there is one; the dumped config's objective block
    # otherwise. Preflight reconciles the two on every checkpointed run, so on the offline path
    # the config is the record of what the tables were collected under.
    weights = (
        dict(task.hparams) if task is not None
        else dict((config.get("model_config") or {}).get("VAE_model") or {})
    )
    readouts = dict(collection.results.get("readouts") or {})
    nll_full = _finite(readouts.get("nll_full_block"))
    nll_base = _finite(readouts.get("nll_base_block"))
    kl_raw = _finite(readouts.get("source_conditioned_kl_raw"))
    lambda_full = float(weights.get("lambda_full", 1.0))
    lambda_base = float(weights.get("lambda_base", 1.0))
    beta_end = _beta_end(weights)
    estimate = (
        lambda_full * nll_full + lambda_base * nll_base + beta_end * kl_raw
        if nll_full is not None and nll_base is not None and kl_raw is not None
        else None
    )

    return {
        "n_parameters": (
            None if task is None
            else int(sum(parameter.numel() for parameter in task.orig_model.parameters()))
        ),
        "train_epoch": None if blob is None else blob.get("epoch"),
        "anchor_coverage_frac": coverage,
        "observed_loss_scale": {
            "nll_full_block": nll_full,
            "nll_base_block": nll_base,
            "source_conditioned_kl_raw": kl_raw,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
            "beta_end": beta_end,
            "free_bits": float(weights.get("free_bits", 0.0)),
            "main_loss_estimate": estimate,
            # The raw KL stands in for the trained one, which the evaluation deliberately does
            # not aggregate; the two coincide at the shipped free_bits: 0.0 and the estimate is a
            # lower bound otherwise.
            "note": (
                "training-path per-anchor magnitudes recombined with the objective's weights at "
                "the end-of-ramp beta, for re-deriving the spike breaker's additive_margin; the "
                "raw KL stands in for the trained one (exact at free_bits 0.0)"
            ),
        },
    }


# =============================================================================
# Entry point
# =============================================================================
def main(
    checkpoint: Optional[Any] = None,
    output_dir: Optional[Any] = None,
    *,
    overrides: Optional[Any] = None,
    device: Optional[str] = None,
    num_samples: Optional[int] = None,
    max_batches: Optional[int] = None,
    only: Optional[str] = None,
    skip: Optional[str] = None,
    argument_sources: Optional[Dict[str, str]] = None,
) -> int:
    """Evaluate a checkpoint -- or re-read a finished run -- and write ``summary.json``.

    Everything that shapes the run comes from the merged configuration's ``eval_config`` block --
    the seed, the Monte Carlo draw count, the two verdict thresholds -- rather than from
    arguments, because that block is dumped into the run directory and is the durable record; a
    value injected from Python would appear in no artifact. ``num_samples`` and ``max_batches``
    are the two exceptions, both smoke-run conveniences, and the effective draw count is recorded
    either way.

    Args:
        checkpoint: Path to the checkpoint. ``None`` re-runs the analyses against the finished run
            in ``output_dir``, with no model built.
        output_dir: Run directory; a timestamped one is created when omitted.
        overrides: The evaluation override delta merged over the checkpoint's own resolved
            config. Defaults to the committed one. Ignored without a checkpoint, where the
            finished run's own dumped config is already the merged result.
        device: Torch device string; chosen automatically when omitted.
        num_samples: Monte Carlo draws $K$, overriding ``eval_config.num_mc_samples``.
        max_batches: Stop after this many batches, for a smoke run.
        only: Comma-separated analyses to run exclusively, from :data:`ANALYSES`. ``None`` runs
            every one of them, which is the default.
        skip: Comma-separated analyses to skip, from :data:`ANALYSES`. ``None`` skips none.
        argument_sources: Which source supplied each argument, recorded in ``summary.json``.

    Returns:
        The process exit code: non-zero when any step failed. **Not** the sanity block's warning
        flag -- a run whose every step succeeded can still be one nobody should quote a number
        from, and that asymmetry is why an offline acceptance gate exists separately.

    Raises:
        EvalPreconditionUnmet: If any precondition fails. The run directory then holds the merged
            config and the log carrying the refusal, and no ``summary.json``.
        FileNotFoundError: If neither a checkpoint nor a finished run directory was given, or if
            a finished directory holds no tables to re-read.
    """
    started_at = time.time()
    checkpoint_path = None if checkpoint is None else Path(checkpoint)

    if checkpoint_path is not None:
        config_path = resolved_config_for(checkpoint_path)
        logger.info(f"evaluating {checkpoint_path} against {config_path}")
        # load_config is a no-op on an already-resolved file (it carries no `base:` key) and is
        # used anyway so there is exactly one config reader in the tree.
        run_config = load_config(str(config_path))
        config, overridden = merge_eval_overrides_with_provenance(run_config, overrides)
    else:
        # No checkpoint: the run being re-read already dumped the merged configuration it used,
        # and re-deriving one from today's committed delta would evaluate the tables in this
        # directory against a contract they were not collected under.
        config_path = finished_run_config(output_dir)
        logger.info(f"re-running analyses against {config_path} with no checkpoint")
        config, overridden = load_config(str(config_path)), []

    force_single_process_loader(config)
    # Before the checkpoint, the loader and the output directory: a misspelled eval_config key or
    # analysis name must cost a parse rather than a model load and a first pass over the shards.
    eval_config = validate_eval_config(config)
    config["eval_config"] = eval_config
    selected = select_analyses(tuple(ANALYSIS_FUNCTIONS), only, skip)

    seed = int(eval_config["seed"])
    numerics = configure_numerics(seed)
    resolved_device = resolve_device(device)

    results_dir = make_output_dir(config, output_dir)
    sink_id = configure_logging(results_dir, config)
    report = Report()
    try:
        logger.info(f"writing results to {results_dir}")
        dump_resolved_config(config, results_dir)
        # Once, here, rather than as an import side effect: the publication style mutates global
        # rcParams, and an import that did it would restyle every other figure in the process.
        configure_figure_style()

        task = None
        blob: Optional[Dict[str, Any]] = None
        preflight_record: Dict[str, Any]
        if checkpoint_path is not None:
            # The blob is read once and handed on, so the reconciliation below compares against
            # the checkpoint's own record rather than against a second read of the same file.
            blob = read_checkpoint(checkpoint_path)
            task = load_task(checkpoint_path, resolved_device, blob=blob)
            # Outside any failure isolation: a refusal must reach the operator as a refusal
            # rather than as a step that happened to fail, and must leave no summary behind it.
            preflight_record = preflight.run_preflight(
                config=config,
                model=task.orig_model,
                checkpoint_path=checkpoint_path,
                model_kwargs=blob.get("model_kwargs") or {},
                hyper_parameters=dict(task.hparams),
            )
            preflight.write_preflight(preflight_record, results_dir)
        else:
            preflight_record = read_preflight(results_dir)

        # The model is what was trained, so it -- not the config -- is asked what guard it
        # carries; a pass with no model reads the figure back off what the tables were collected
        # under, which is the same number by construction.
        delay_steps = 0 if task is None else int(task.orig_model.source_delay_steps)
        if delay_steps:
            logger.info(
                f"source channels are delayed by up to {delay_steps} steps "
                f"({delay_steps * SECONDS_PER_STEP:g} s); the reported lag adds this back as an "
                f"upper bound."
            )

        collection, probe_record, loader = load_or_collect_tables(
            results_dir,
            task=task,
            config=config,
            eval_config=eval_config,
            checkpoint_path=checkpoint_path,
            seed=seed,
            device=resolved_device,
            num_samples=num_samples,
            max_batches=max_batches,
            delay_steps=delay_steps,
            report=report,
        )
        if task is None:
            delay_steps = int((collection.results.get("lag") or {}).get("delay_steps") or 0)

        # Seeded before the analyses so the headline, the sanity block and any analysis that
        # reads a readout all see the same numbers. Into a *separate* mapping: the collection's
        # own record carries these blocks too and is already on disk, and a run that grew a
        # headline inside it would hand the next pass a headline of the previous one's making.
        report.results.update(dict(collection.results))
        # Who was evaluated, computed from the tables and both resolved dataset lists rather than
        # asserted: whether the two cohorts are disjoint, which of the canonical subgroups the
        # pretraining split never covered, and the two sentences a reader needs before quoting a
        # class contrast or comparing a number against a training log.
        report.set(
            "cohort", cohort.build_cohort_block(collection.per_sample, config, probe_record)
        )
        context = AnalysisContext(
            collection=collection, config=config, task=task, loader=loader
        )
        run_analyses(
            report, list(UNSKIPPABLE_ANALYSES), UNSKIPPABLE_ANALYSES,
            context=context, eval_config=eval_config, output_dir=results_dir,
            probe=probe_record,
        )
        run_analyses(
            report, selected, ANALYSIS_FUNCTIONS,
            context=context, eval_config=eval_config, output_dir=results_dir,
            probe=probe_record,
        )

        # Only on CUDA, and **absent** rather than zero on CPU: a 0.00 GB peak reads as "measured,
        # and the run used no memory", which is a claim a CPU box cannot make.
        if resolved_device.type == "cuda":
            report.set(
                "max_memory_allocated_gb",
                float(torch.cuda.max_memory_allocated(resolved_device)) / (1024**3),
            )

        # The run facts the calibration study's tables read: parameter count, training epoch,
        # the coverage distribution and the observed objective magnitude. Outside `results`,
        # because the first two are checkpoint facts a model-free re-run cannot reproduce.
        run_context = build_run_context(
            task=task, blob=blob, config=config, collection=collection
        )

        artifacts = finalise(
            report,
            output_dir=results_dir,
            analyses=list(UNSKIPPABLE_ANALYSES) + selected,
            eval_config=eval_config,
            started_at=started_at,
            per_sample=collection.per_sample,
            per_anchor=collection.per_anchor,
            probe=probe_record,
        )

        summary = {
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "config": str(config_path),
            "output_dir": str(results_dir),
            "device": str(resolved_device),
            # Per-channel delays have no single representative; the maximum is used, so the
            # reported lag is an upper bound. Recorded beside the number so the choice travels
            # with it.
            "source_delay_steps": delay_steps,
            "source_delay_is_max_over_channels": True,
            "eval_config": eval_config,
            # Read back from global state rather than echoed from the assignments, so the record
            # is what was in force rather than what was asked for.
            "numerics": numerics,
            # What this evaluation changed about the run's own contract, per key, with both
            # values. A divergence recorded nowhere is indistinguishable from an accident.
            "config_overrides": {
                "path": str(overrides) if overrides is not None else None,
                "entries": overridden,
            },
            "arguments": {
                "values": {
                    "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                    "output_dir": None if output_dir is None else str(output_dir),
                    "overrides": None if overrides is None else str(overrides),
                    "device": None if device is None else str(device),
                    "num_samples": num_samples,
                    "max_batches": max_batches,
                    "only": only,
                    "skip": skip,
                },
                "sources": dict(argument_sources or {}),
            },
            "analyses_selected": selected,
            "analyses_unskippable": list(UNSKIPPABLE_ANALYSES),
            # What the shared pass wrote down, minus the readouts it also carries: the row
            # counts, the honest denominator behind every column, what was excluded for scoring
            # no anchors, what was retained under which cap, and the streamed accumulators.
            "collection": collect.record_summary_view(collection.record),
            # The run facts the arm tables and the first-run checklist consume; see
            # build_run_context for what each is and why the block sits outside `results`.
            "run_context": run_context,
            "preflight": preflight_record,
            # Promoted out of the preflight record as well as left inside it: what the readout is
            # not must be legible to a reader who opens only the summary.
            "causality": preflight_record.get("causality"),
            "results": report.results,
            "steps": step_records(report.steps),
            "n_steps": len(report.steps),
            "n_failed": len(report.failed_steps),
            "failed": [record.name for record in report.failed_steps],
            # Beside the results rather than inside them: it describes the directory rather than
            # the model, and two passes into one directory legitimately produce different
            # manifests from identical results.
            "artifacts": artifacts,
            "exit_code": report.exit_code(),
        }

        summary_path = results_dir / SUMMARY_FILENAME
        with open(summary_path, "w", encoding="utf-8") as handle:
            # allow_nan=False so an unsanitised non-finite value raises here rather than
            # producing a file only Python can read back.
            json.dump(json_safe(summary), handle, indent=2, allow_nan=False)
        logger.info(f"wrote {summary_path}")

        logger.info(console_summary(report.results, report.steps))
        # At the end deliberately: in a run whose log is tens of thousands of lines the original
        # ERROR is long gone and the operator's attention is on the tail.
        for record in report.failed_steps:
            logger.error(f"[{record.name}] {record.error}")
            logger.error(f"{record.traceback}")
        return report.exit_code()
    finally:
        logger.remove(sink_id)


def read_preflight(results_dir: Path) -> Dict[str, Any]:
    """Read back the preflight record a checkpointed pass over this directory wrote.

    A pass with no checkpoint has no model to run the guards against, and cannot regenerate the
    causality disclosure the summary promotes. It reads the record instead, and says so where it
    cannot: an absent record is reported as skipped rather than filled in with defaults, because
    a disclosure nobody produced must not read as one that passed.

    Args:
        results_dir: The results directory.

    Returns:
        The record, or a skip record naming why there is none.
    """
    path = Path(results_dir) / preflight.PREFLIGHT_FILENAME
    if not path.is_file():
        return {
            "skipped": True,
            "reason": (
                f"this pass built no model, and {preflight.PREFLIGHT_FILENAME} is not in "
                f"{results_dir}, so the guards were neither run now nor run before"
            ),
        }
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    record["reused_from"] = str(path)
    return record


def load_or_collect_tables(
    results_dir: Path,
    *,
    task: Optional[SeqVaeLagAttnRwsTask],
    config: Dict[str, Any],
    eval_config: Dict[str, Any],
    checkpoint_path: Optional[Path],
    seed: int,
    device: torch.device,
    num_samples: Optional[int],
    max_batches: Optional[int],
    delay_steps: int,
    report: Report,
) -> Tuple[collect.Collection, Optional[Dict[str, Any]], Optional[DataLoader]]:
    """Reuse this directory's tables, or run the one shared pass that produces them.

    The **probe** runs only in the second case, and that is what makes an offline re-run cheap: it
    is a full iteration of the split, and a pass reusing tables has no reason to pay for one when
    the record it left behind carries what the population checks read.

    A **loader** is still built whenever there is a checkpoint, even against a finished directory.
    That costs the dataset's index build and no forward pass, and it buys a property worth more
    than the cost: the per-sample pages are rendered from the loader rather than read off a table,
    so without one a re-run of the same checkpoint into the same directory would silently skip
    them and its ``results`` would stop comparing equal to the run it re-ran -- which is the
    property that block exists to have. A pass with no checkpoint has nothing to render *with*
    anyway, so it builds nothing.

    Args:
        results_dir: The run's results directory.
        task: The loaded task, or ``None`` on a pass with no checkpoint.
        config: The merged run config.
        eval_config: The validated ``eval_config`` block.
        checkpoint_path: The checkpoint being evaluated, for the provenance comparison.
        seed: The run's seed, offset per generator.
        device: The device the Monte Carlo draw is made on.
        num_samples: An explicit draw count, or ``None`` for the configured one.
        max_batches: Batch cap for a smoke run.
        delay_steps: The causal input delay, for the lag report.
        report: The run's report, so the probe runs as an isolated step.

    Returns:
        ``(collection, probe_record, loader)``. The probe record is ``None`` only when neither
        this pass nor any earlier one produced it; the loader is ``None`` exactly when there is no
        checkpoint, which is when there is nothing to render a page with.

    Raises:
        FileNotFoundError: If there is no model to collect with and no tables to read.
        TablesProvenanceMismatch: If the tables here belong to another run.
    """
    draws = int(eval_config["num_mc_samples"] if num_samples is None else num_samples)
    if collect.has_collection(results_dir):
        if task is None and num_samples is None:
            # Nothing to collect *with*, so the draw count the tables were collected under is a
            # fact about them rather than a setting to compare against: refusing here would refuse
            # every offline re-run of a smoke pass, and there is no forward to redo at another K.
            # An explicitly requested count still reaches the comparison, and is still refused.
            recorded = (
                (collect.read_record(results_dir) or {}).get("provenance") or {}
            ).get("num_mc_samples")
            if recorded is not None and int(recorded) != draws:
                logger.info(
                    f"the tables here were collected with {int(recorded)} Monte Carlo draw(s), "
                    f"not the configured {draws}; this pass builds no model, so it reports them "
                    f"as collected."
                )
                draws = int(recorded)
        collection = collect.load_or_collect(
            results_dir,
            _refuse_to_collect,
            checkpoint_path=checkpoint_path,
            eval_config=eval_config,
            num_samples=draws,
        )
        # Unshuffled, unlike the collection pass's: nothing here is batched through the model, and
        # the per-sample pages build their own strictly sequential loader over a Subset of this
        # one's dataset anyway.
        cached_loader = (
            None if task is None else GraphDataModule(config).test_dataloader()
        )
        return collection, probe_module.read_probe(results_dir), cached_loader

    if task is None:
        raise FileNotFoundError(
            f"{results_dir} holds no {collect.COLLECTION_FILENAME}, so there are no tables to "
            f"re-run the analyses against, and no --checkpoint was given to collect them with."
        )

    # The sample cap first, so the shuffle re-batches what will actually be scored. Applied to the
    # dataset rather than by stopping early: a prefix of the concatenated per-subgroup index is
    # one subgroup and one clinical class.
    capped, cap_record = capped_sample_loader(
        GraphDataModule(config).test_dataloader(),
        eval_config.get("max_samples"),
        seed + _SEED_OFFSET_SAMPLE_CAP,
    )
    loader = shuffled_control_loader(capped, seed + _SEED_OFFSET_LOADER)
    # One loader pass, no forward. It is what raises on a shard that contributed nothing, on a
    # missing clinical field and on a GUID appearing in two holdout shards -- and what the
    # population sanity checks read afterwards. Inside the wrapper: its failure is worth a
    # non-zero exit and a named step, not the loss of the readouts.
    probe_record = report.step(
        "probe",
        probe_module.run_probe,
        loader,
        configured_files=(config.get("dataset_config") or {}).get("vae_test_datasets"),
        max_batches=max_batches,
        output_dir=results_dir,
    )
    write_steps(report.steps, results_dir)

    perm_generator = torch.Generator().manual_seed(seed + _SEED_OFFSET_PERM)
    # On the model's device, because the draw it seeds is ``normal_`` into a tensor there.
    mc_generator = torch.Generator(device=device).manual_seed(seed + _SEED_OFFSET_MC)
    # One pass, two durable tables, and the readouts. Every later analysis reads the tables
    # rather than the model, so the decoder pass -- four branches over 480 raw samples per
    # anchor at K draws -- happens exactly once per run directory.
    def _collect() -> collect.Collection:
        """Run the pass and record what the sample cap did, before the tables are written."""
        collected = collect.collect_tables(
            task,
            loader,
            eval_config=eval_config,
            num_samples=draws,
            # What the pass can actually reach, so the retention draw is resolved against the
            # positions the collector will see. A batch cap stops the loop early, and the sink's
            # index space is compacted over scored batches -- a draw over the whole dataset would
            # land mostly past the end and retain nothing while still reporting a plan.
            n_total=_reachable_samples(loader, max_batches),
            max_batches=max_batches,
            perm_generator=perm_generator,
            mc_generator=mc_generator,
            delay_steps=delay_steps,
        )
        # Inside the pass, so it is written with the tables: a directory whose forward pass is
        # skipped must still be able to say what population its numbers describe.
        collected.record["sample_cap"] = cap_record
        return collected

    collection = collect.load_or_collect(
        results_dir,
        _collect,
        checkpoint_path=checkpoint_path,
        eval_config=eval_config,
        num_samples=draws,
    )
    return collection, probe_record, loader


def _reachable_samples(loader: DataLoader, max_batches: Optional[int]) -> Optional[int]:
    """How many samples the collection pass can reach, given a batch cap.

    Args:
        loader: The loader the pass will iterate.
        max_batches: The smoke-run batch cap, or ``None``.

    Returns:
        The sample count the retention draw should be resolved against, or ``None`` when the
        loader cannot say -- which the draw reads as "no cap is resolvable".
    """
    try:
        n_total = len(loader.dataset)  # type: ignore[arg-type]
    except TypeError:
        return None
    if max_batches is None:
        return int(n_total)
    return min(int(n_total), int(max_batches) * int(loader.batch_size or 1))


def _refuse_to_collect() -> collect.Collection:
    """Stand in for the collection pass where the caller has already established there is none.

    Reached only if :func:`collect.has_collection` and :func:`collect.load_or_collect` disagree
    about whether this directory holds tables, which is a bug rather than a state.
    """
    raise AssertionError(
        "the collection pass was invoked on a directory whose tables were already found"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Every ``dest`` here is also a valid :data:`RUN_ARGS` key."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_rws.eval.run",
        description="Evaluate a trained SeqVaeLagAttnRws checkpoint.",
    )
    # Not required: every analysis reads the run's tables rather than the model, so a re-run
    # against a finished directory needs no checkpoint at all. What still does is the collection
    # pass, and the run says so by name when the directory holds no tables.
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to the checkpoint to evaluate. Required unless --output-dir names a finished "
             "run whose tables the analyses can be re-run against.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Run directory. Default: a timestamped directory under out_dir_base/<tag>-eval.",
    )
    parser.add_argument(
        "--overrides", default=None,
        help="Evaluation override delta merged over the checkpoint's own resolved config. "
             "Default: the committed eval_overrides.yaml.",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device. Default: cuda:0 when available, else cpu."
    )
    parser.add_argument(
        "--num-samples", dest="num_samples", type=int, default=None,
        help="Monte Carlo draws per anchor. Default: eval_config.num_mc_samples.",
    )
    parser.add_argument(
        "--max-batches", dest="max_batches", type=int, default=None,
        help="Stop after this many batches. For a smoke run only.",
    )
    # Both flags take a comma-separated list of ANALYSIS_FUNCTIONS keys. An unknown name raises
    # at startup rather than silently running everything (--only) or nothing extra (--skip).
    # `band_partition` is NOT valid for either: it always runs and is not selectable.
    # Interpolated from the registry rather than restated, so ``--help`` names exactly what is
    # registered today and a fifteenth analysis appears in the help text by being registered.
    selectable = ", ".join(ANALYSES)
    parser.add_argument(
        "--only", default=None,
        help=f"Comma-separated analyses to run exclusively. Default: all of them. One or more "
             f"of: {selectable}.",
    )
    parser.add_argument(
        "--skip", default=None,
        help=f"Comma-separated analyses to skip. Default: skip none. One or more of: "
             f"{selectable}.",
    )
    return parser


def resolve_arguments(
    argv: Optional[Sequence[str]] = None, run_args: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Resolve the argument set from the command line and :data:`RUN_ARGS`, **per key**.

    Per key rather than all-or-nothing: the common iteration is varying one thing, and a fallback
    that was discarded the moment any flag appeared would make the launch dict useless for exactly
    that. So ``--checkpoint other.ckpt`` overrides that one value and leaves the output directory
    and the analysis selection to the dict.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv[1:]``.
        run_args: The fallback dict. ``None`` uses the module-level :data:`RUN_ARGS`.

    Returns:
        ``(values, sources)``, where ``sources`` maps each key to ``'cli'``, ``'config'`` or
        ``'default'``. Recorded in ``summary.json``, so a run's provenance is unambiguous after
        the fact rather than reconstructed from a shell history.

    Raises:
        ValueError: If ``run_args`` carries a key that is not an argparse ``dest``. A typo there
            would otherwise silently do nothing, which is the same class of failure the
            ``eval_config`` validator guards against.
    """
    parser = build_parser()
    fallback = dict(RUN_ARGS if run_args is None else run_args)

    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(fallback) - valid_dests)
    if unknown:
        raise ValueError(
            f"RUN_ARGS carries key(s) that are not command-line arguments: "
            f"{', '.join(repr(key) for key in unknown)}. Valid keys are: "
            f"{', '.join(sorted(valid_dests))}. RUN_ARGS is a launch convenience, not a second "
            f"configuration surface -- settings that shape the run belong in the override delta, "
            f"which is dumped into the run directory."
        )

    parsed = vars(parser.parse_args(list(argv) if argv is not None else None))
    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for key in sorted(valid_dests):
        if parsed.get(key) is not None:
            values[key], sources[key] = parsed[key], "cli"
        elif fallback.get(key) is not None:
            values[key], sources[key] = fallback[key], "config"
        else:
            values[key], sources[key] = None, "default"
    return values, sources


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    values, sources = resolve_arguments(argv)
    if values["checkpoint"] is None and not _finished_run(values["output_dir"]):
        raise SystemExit(
            "--checkpoint is required unless --output-dir names a finished run directory: the "
            "analyses read the tables the collection pass wrote, and there are none here to read."
        )
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # The shard paths inside a resolved config are repo-root-relative for the tiny variant,
        # and a relative path resolved against an arbitrary working directory surfaces as "no
        # samples match the specified filters" with no mention of the real cause.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(**values, argument_sources=sources)


def _finished_run(output_dir: Optional[Any]) -> bool:
    """Whether ``output_dir`` holds a run the analyses could be re-run against."""
    try:
        finished_run_config(output_dir)
    except FileNotFoundError:
        return False
    return True


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key, so varying only the checkpoint works without
#: editing anything here, and a key that is not an argparse ``dest`` raises at startup.
#:
#: Do not add run settings here. The seed, the caps and the draw count belong in the override
#: delta, which is dumped into the run directory as the durable record; a value injected from
#: Python would appear in no artifact.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,
    "output_dir": None,
    "overrides": None,
    "device": None,
    "num_samples": None,
    "max_batches": None,
    # Comma-separated analysis names, or None for all of them. `band_partition` always runs and
    # is not selectable.
    "only": None,
    "skip": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
