r"""The evaluation command line: a checkpoint in, ``summary.json`` out.

.. code-block:: bash

    python -m teb_vae.lag_attn_cfs.eval.run --checkpoint <path> [--output-dir <dir>]

**There is no second config, only a delta.** A training run writes its fully resolved
configuration beside its checkpoints, and that file -- not a committed YAML -- is the record of
what the model was trained on. The evaluation reads it and deep-merges the committed override
delta over the top, so the run's own contract stays authoritative and everything that genuinely
differs (the causal holdout shards, the statistics accumulated from them, the five extra
``load_fields`` the clinical questions are asked in, and the ``eval_config`` block) is one small
reviewable file. A ``base:`` chain cannot do that job: ``base:`` resolves relative to the file
naming it, and a checkpoint's resolved config is a runtime path no committed file can reference.

Four things the run forces regardless of what either file says:

* **A single-process loader.** ``create_optimized_dataloader`` turns on persistent workers
  whenever ``num_workers > 0``, and spawn workers over a multi-file HDF5 dataset degrade after
  the first full pass, silently truncating later passes to the first file's index range. An
  evaluation makes many passes.
* **A fixed-seed shuffle over that loader.** The test split is eight per-subgroup shards read in
  order, so an unshuffled batch holds consecutive segments of one recording -- and the
  permutation control then has no stranger in the batch to borrow a source from. The shuffle is
  drawn from the run's own seed, so it is a reordering, not a source of run-to-run variation.
* **Evaluation mode and no gradient**, which the readout module owns.
* **A timestamped output directory** at second resolution, because two evaluation runs in the
  same minute is normal while iterating.

The architecture is rebuilt from the checkpoint's own ``model_kwargs`` and the objective from its
``hyper_parameters``, so what is evaluated is what was trained rather than what a config file
currently says -- and preflight refuses the run outright when the merged config contradicts
either. On this cell that reconciliation reaches further than the sibling's: the warm-up budget is
re-resolved against the configured shards and compared with the four tuples the checkpoint stamped,
because those tuples decide which target channels exist at all.

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
deliberate exception, for the reason above.

**A checkpoint is not required.** Everything after the collection pass reads the tables rather
than the model, so ``--output-dir <a finished run>`` with no ``--checkpoint`` re-runs the analyses
offline, with no model built and no GPU touched -- which is the point of splitting collection from
emission, and the form a re-run takes after a multi-hour pass failed at its ninth step.

**The checkpoint is rebuilt through** :mod:`teb_vae.lag_attn_cfs.eval.probe`, which owns that
loading path already: this model's forward takes five arguments and refuses a missing anchor phase
above stride $1$, so the probe measures its contract against a rebuilt model rather than reading
one, and a second reconstruction here would be a second place for the two to disagree.
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

#: Repository root: ``teb_vae/lag_attn_cfs/eval/run.py`` -> up four.
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
from teb_vae.lag_attn_cfs.eval import (  # noqa: E402
    cohort,
    collect,
    launch,
    metrics,
    preflight,
    probe as probe_module,
)
from teb_vae.lag_attn_cfs.eval._reuse import configure_numerics, subsample_indices  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    GROUPED_FRAMES_KEY,
    AnalysisContext,
)
from teb_vae.lag_attn_cfs.eval.analyses import attention as attention_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    band_partition as band_partition_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    calibration as calibration_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import coupling as coupling_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    cross_subgroup as cross_subgroup_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    distributions as distributions_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import events as events_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import forecast as forecast_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import lag_kl as lag_kl_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import latent as latent_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    perm_control as perm_control_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import residual as residual_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import samples as samples_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    sufficiency as sufficiency_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import (  # noqa: E402
    time_to_delivery as time_to_delivery_analysis,
)
from teb_vae.lag_attn_cfs.eval.analyses import trajectory as trajectory_analysis  # noqa: E402
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING, ModelBinding  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides_with_provenance,
    validate_eval_config,
)
from teb_vae.lag_attn_cfs.eval.figures_seam import configure_figure_style  # noqa: E402

# Re-exported rather than reimplemented, in both directions. One serialiser writes every summary
# this repository produces, so a value that survives a round trip in one package cannot fail the
# write in another; and one module rebuilds a checkpoint of this cell, so the contract the probe
# measures is the contract the run scores through.
from teb_vae.lag_attn_cfs.eval.probe import (  # noqa: E402,F401
    load_task,
    read_checkpoint,
    resolve_device,
    resolved_config_for,
)
from teb_vae.lag_attn_cfs.eval.report_seam import (  # noqa: E402,F401
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
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402

#: The results subdirectory created inside every run directory.
RESULTS_DIRNAME = "eval_results"

#: The run's log file, beside the artifacts it explains.
LOG_FILENAME = "eval.log"

#: Steps that always run, in run order, and are **not** selectable.
#:
#: The target channel map belongs here rather than on the registry below because it describes the
#: *data* rather than the model: it reads the shards' own ``sel_*`` provenance and per-block causal
#: attributes, needs no forward pass and no tables, and is the data-side companion to the causality
#: disclosure. It is also what the band-resolved skill readout joins against, and that join is over
#: the **kept** channel axis, so a run whose channel map could be skipped would be a run whose
#: frequency-resolved statements have no definition of a band behind them.
#:
UNSKIPPABLE_ANALYSES: Dict[str, Any] = {
    "band_partition": band_partition_analysis.run_band_partition_analysis,
}

#: Analyses ``--only`` and ``--skip`` select between, in run order.
#:
#: Registering one is a line here: every entry takes the same arguments and returns the same four
#: keys, which is what the protocol in ``analyses/__init__.py`` exists to guarantee. The rest of
#: the readouts a run reports come from the shared collection pass rather than from a registered
#: analysis, which is why this dict being empty still produces a summary carrying every readout,
#: every verdict and both durable tables.
#:
#: There is deliberately no dependency table beside it. The real dependency is on *files existing
#: on disk* rather than on an analysis having run in this pass, which is exactly what makes
#: ``--only <name> --output-dir <a finished run>`` work at all. One line adds the table the day a
#: genuine correctness dependency appears.
#:
#: ``cross_subgroup`` is registered **last**, and that ordering is load-bearing: it reads the
#: per-recording CSVs the analyses above it write, so on a single pass it can only test the
#: metrics they have already produced. It does not *depend* on them having run in this pass --
#: a source that is absent is recorded rather than raised, which is what keeps ``--only`` working
#: -- but a run of everything should test everything.
#:
#: The order is the sibling's with the analyses this package does not have left out, rather than a
#: new one: the two summaries are read side by side in the arm comparison, and a registry that
#: reordered what it kept would make that comparison an exercise in re-sorting. ``coherence`` sits
#: between ``residual`` and ``distributions`` there and is simply absent here -- the stored
#: coefficients are moduli, so phase agreement and group delay have no analogue at any window
#: length, and the frequency-resolved readout this package has instead is ``spectral_skill``.
ANALYSIS_FUNCTIONS: Dict[str, Any] = {
    "forecast": forecast_analysis.run_forecast_analysis,
    "coupling": coupling_analysis.run_coupling_analysis,
    "perm_control": perm_control_analysis.run_perm_control_analysis,
    "latent": latent_analysis.run_latent_analysis,
    "lag_kl": lag_kl_analysis.run_lag_kl_analysis,
    "attention": attention_analysis.run_attention_analysis,
    "calibration": calibration_analysis.run_calibration_analysis,
    "residual": residual_analysis.run_residual_analysis,
    # The last of the table-describing analyses, before the against-time ones. It reads
    # ``per_sample.csv`` and nothing else, so its position is a reading order rather than a
    # dependency.
    "distributions": distributions_analysis.run_distributions_analysis,
    "trajectory": trajectory_analysis.run_trajectory_analysis,
    "time_to_delivery": time_to_delivery_analysis.run_time_to_delivery_analysis,
    # One readout of the sibling's three: contraction-conditioned coupling. The two that scored a
    # bpm waveform are absent and the analysis's own record names them.
    "events": events_analysis.run_events_analysis,
    # Last but for the two below, and deliberately so: it is the only analysis whose cost is a
    # training loop rather than a forward pass, so a run that fails earlier fails before paying
    # for it -- and everything it reports is a comparison against readouts the pass already has.
    "sufficiency": sufficiency_analysis.run_sufficiency_analysis,
    "samples": samples_analysis.run_samples_analysis,
    "cross_subgroup": cross_subgroup_analysis.run_cross_subgroup_analysis,
}

#: The **shared** analysis names, in run order, for a reader and for the second cfs cell to merge
#: its own onto. Deliberately not what ``--only`` and ``--skip`` accept: selection runs against the
#: registry a *binding* merges (:func:`merged_analysis_functions`), and this cell's own three are
#: registered on ``CFS_BINDING`` rather than above, so a run of this package selects from eighteen
#: names while this tuple holds fifteen.
ANALYSES: Tuple[str, ...] = tuple(ANALYSIS_FUNCTIONS)


#: Shared analyses that stay **last** in the merged registry, after a binding's own.
#:
#: Exactly one, and it is there because its inputs are the per-recording CSVs the analyses above it
#: wrote: ``cross_subgroup`` tests cohort differences in metrics that other steps produce, and its
#: source table names three of this cell's own analyses. Appending the binding's extras after it --
#: which is what a plain ``{**shared, **extras}`` does -- would make a full run test those three
#: sources every time and find every one of them absent, with the summary recording it as a partial
#: directory rather than as a run order that cannot work.
#:
#: It is not a dependency table and deliberately not the start of one. The real dependency is on
#: *files existing on disk*, which is what makes ``--only <name> --output-dir <a finished run>``
#: work at all; this is only about which order a single pass writes them in.
TRAILING_ANALYSES: Tuple[str, ...] = ("cross_subgroup",)


def merged_analysis_functions(binding: ModelBinding) -> Dict[str, Any]:
    """Return the shared registry with a binding's own analyses merged in, in run order.

    Built fresh on every call rather than once at import, so a test that replaces
    :data:`ANALYSIS_FUNCTIONS` still drives what the run selects from -- and so the extras the
    second cfs cell registers appear in its help text, its selection and its ``summary.json``
    record from one place.

    Args:
        binding: The model binding whose ``extra_analyses`` are merged.

    Returns:
        A new ordered mapping: every shared analysis but the trailing ones, then the binding's own
        in declaration order, then :data:`TRAILING_ANALYSES`.

    Raises:
        ValueError: If an extra analysis reuses a shared name. Replacing a shared implementation
            silently would leave two models reporting different things under one name, which is
            indistinguishable in the output from them agreeing.
    """
    collisions = sorted(set(binding.extra_analyses) & set(ANALYSIS_FUNCTIONS))
    if collisions:
        raise ValueError(
            f"binding for {binding.model_cls.__name__} registers analyses whose names are "
            f"already in the shared registry: {collisions}. An extra analysis is an addition, "
            f"never an override: rename it, or -- if the shared implementation is genuinely "
            f"wrong -- fix it there, where both models get the fix."
        )
    leading = {
        name: function
        for name, function in ANALYSIS_FUNCTIONS.items()
        if name not in TRAILING_ANALYSES
    }
    trailing = {
        name: ANALYSIS_FUNCTIONS[name]
        for name in TRAILING_ANALYSES
        if name in ANALYSIS_FUNCTIONS
    }
    return {**leading, **binding.extra_analyses, **trailing}


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


def make_output_dir(
    config: Dict[str, Any],
    explicit: Optional[Any] = None,
    *,
    binding: ModelBinding = CFS_BINDING,
) -> Path:
    """Create and return the results directory for this evaluation run.

    An explicit directory is used as given, which is what makes the documented offline re-run
    possible -- and also what makes it destructive, so any prior summary is preserved first. The
    timestamped default cannot collide and needs no such guard.

    Args:
        config: The resolved run config, for its ``out_dir_base`` and ``tag``.
        explicit: An explicit run directory from ``--output-dir``, used as given.
        binding: The model being evaluated, whose ``tag`` is the fallback when the config
            declares none -- so two models' runs land in two directories rather than in one told
            apart only by timestamp.

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
    tag = str(general.get("tag", binding.tag))
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
    per-subgroup files, so a prefix cap yields one subgroup and one clinical class. Stratifying by
    shard gives every file a share of the cap proportional to its size, which guarantees each one
    appears whenever the cap is at least the shard count rather than merely making it likely.

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
    the form :func:`~teb_vae.lag_attn_cfs.eval.frames.grouped_frame_entry` writes: an absolute
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
# The one verdict an analysis has to finish
# =============================================================================
#: The analysis that produces the interval the availability-clock criterion is decided on, and the
#: path into its block. Named here rather than reached for inline so the coupling between the two is
#: one declaration a reader can find.
CLOCK_INTERVAL_SOURCE = ("source_null", "difference")


def _finite_or_none(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` when it is missing or not finite.

    ``None`` rather than a ``NaN`` reaching the verdict: every comparison against a ``NaN`` is
    false, so a missing measurement would read as a failed criterion instead of an unevaluated one.

    Args:
        value: A number, ``None``, or anything a run's JSON block might carry in that slot.

    Returns:
        The finite float, or ``None``.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def revise_clock_verdict(
    results: Dict[str, Any], *, eval_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    r"""Re-decide ``coupling_exceeds_availability_clock`` once its interval exists.

    Every other criterion is decided in the collection pass, from readouts that pass already has.
    This one cannot be: its criterion is stated on the **lower end of a bootstrap interval over
    recordings**, and that interval is produced by the ``source_null`` analysis, which runs after
    the pass. So the pass emits the criterion as ``INCONCLUSIVE`` with the measured difference
    beside it, and this replaces it with the decided form when -- and only when -- the analysis
    that owns the interval has run and reported one.

    Deliberately **not** a second bootstrap here. One interval exists per run, computed once, in
    the analysis that writes it to disk beside the per-recording table it came from; a second
    resample with the same seed would agree today and would be a second place to drift tomorrow.

    Args:
        results: The run's accumulated results, holding the collected ``verdicts`` block and
            whatever the analyses reported. Mutated in place: the verdict list is replaced rather
            than edited, so the collection's own record on disk is untouched.
        eval_config: The validated block, for ``clock_margin_min_nats``. Unset -- the shipped
            value -- leaves the criterion INCONCLUSIVE with the measurement attached, which is the
            state the threshold is meant to be *set from*.

    Returns:
        The revised verdict, or ``None`` when nothing was revised: no collected verdict block, no
        ``source_null`` block (the analysis was skipped, or this is an ``--only`` re-run of
        something else), or a block reporting no finite interval. Every one of those leaves the
        collection pass's INCONCLUSIVE standing, which is the honest answer -- an unevaluated
        criterion is never upgraded to a pass by the absence of its evidence.
    """
    verdicts = results.get("verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        return None
    block: Any = results
    for key in CLOCK_INTERVAL_SOURCE:
        block = block.get(key) if isinstance(block, dict) else None
    if not isinstance(block, dict):
        return None

    lower, upper = _finite_or_none(block.get("ci_lo")), _finite_or_none(block.get("ci_hi"))
    if lower is None:
        return None

    revised = metrics.availability_clock_verdict(
        _finite_or_none(block.get("source_conditioned_kl_raw_nats")),
        _finite_or_none(block.get("kld_source_null_nats")),
        margin_min_nats=eval_config.get("clock_margin_min_nats"),
        interval=(lower, upper),
    ).as_dict()
    # The denominators the difference was measured over, carried into the verdict rather than left
    # one block away. Under the shipped unset threshold the status says nothing, so what the
    # criterion emits *instead* of a decision is all the reader gets -- and a difference without
    # the recording count behind it is one nobody can weigh.
    for name in ("positive_fraction", "n_positive", "n_recordings"):
        value = _finite_or_none(block.get(name))
        if value is not None:
            revised["values"][name] = value
    # One entry substituted, the list rebuilt rather than mutated in place: the collection's own
    # record is already on disk and a run that edited it would hand a later pass a verdict of this
    # pass's making. Length and order are preserved exactly -- the block is read by name *and* by
    # position, so a dropped entry reads as a criterion that passed.
    results["verdicts"] = [
        revised
        if isinstance(verdict, dict) and str(verdict.get("name")) == revised["name"]
        else verdict
        for verdict in verdicts
    ]
    return revised


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


def _max_or_none(values: Any) -> Optional[int]:
    """Return the largest entry of a warm-up vector, or ``None`` for an ungated stream."""
    return None if not values else int(max(int(step) for step in values))


def warmup_budget_record(model: Any) -> Dict[str, Any]:
    r"""Record what the warm-up budget resolved to on the checkpoint being evaluated.

    Beside the parameter count rather than inside ``results`` because it is a fact about the
    *checkpoint*: the budget is resolved by the training driver against its own shards and stamped
    into ``model_kwargs``, so two arms at two budgets are two mutually unloadable checkpoints
    scored over two different channel axes. An arm table that did not carry these numbers would
    put their block scores in one column.

    Args:
        model: The rebuilt net.

    Returns:
        Declared and surviving widths per stream, the realised warm-up maxima, the anchor floor
        and the warm target fraction the constructor resolved.
    """
    return {
        "anchor_floor": int(model.warmup_period),
        "target_declared": int(model.c_y),
        "target_kept": int(model.decoder_out_channels),
        "target_max_warmup_steps": _max_or_none(getattr(model, "target_warmup_steps", None)),
        "source_declared": int(model.c_u),
        "source_kept": (
            None if getattr(model, "source_gate", None) is None
            else int(model.source_gate.out_channels)
        ),
        "source_max_warmup_steps": _max_or_none(getattr(model, "source_warmup_steps", None)),
        # Identically 1.0 on a model whose constructor accepted the floor, which is the point:
        # a value below it means the checkpoint predates that refusal.
        "target_warm_frac": float(model.target_warm_frac),
    }


def build_run_context(
    *,
    # Whichever task the binding named; every read below is through the shared interface.
    task: Optional[Any],
    blob: Optional[Dict[str, Any]],
    config: Dict[str, Any],
    collection: collect.Collection,
) -> Dict[str, Any]:
    r"""Record the run facts the arm tables and the first-run checklist consume.

    Seven things a summary must carry so the arm comparison is arithmetic rather than archaeology:
    the parameter count (the architecture arms are judged per parameter), the checkpoint's training
    epoch, the class of model that produced the run (which architecture a row belongs to, when a
    comparison spans more than one), the anchor geometry the pass decoded at **beside the stride
    the run trained at**, the resolved warm-up budget, the anchor-coverage distribution the
    ``coverage_floor`` is confirmed or revised against, and the observed magnitude of the training
    objective the spike breaker's ``additive_margin`` is re-derived from.

    The two geometry entries are this cell's own and neither is optional. A table read against the
    training CSV is unreadable without the stride -- $A_{\max}$ differs by a factor of it between
    the two -- and a block score is a sum over $H \cdot C_{\mathrm{keep}}$ coefficients, so a
    comparison of two budgets' nats is a comparison of two different denominators.

    Outside ``results`` deliberately: the checkpoint facts are absent on a pass that built no
    model, and ``results`` is the block an offline re-run must reproduce byte for byte.

    Args:
        task: The loaded task, or ``None`` on a pass with no checkpoint.
        blob: The checkpoint blob, read for its training epoch and its model-class stamp.
            ``None`` without a checkpoint.
        config: The merged run config, the fallback source of the objective weights.
        collection: The shared pass's output, read for the per-anchor coverage column, the
            recorded geometry and the aggregated readouts.

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
    prior_rate = _finite(readouts.get("prior_rate"))
    lambda_full = float(weights.get("lambda_full", 1.0))
    lambda_base = float(weights.get("lambda_base", 1.0))
    beta_end = _beta_end(weights)
    beta_prior = float(weights.get("beta_prior", 0.0))
    estimate = (
        lambda_full * nll_full
        + lambda_base * nll_base
        + beta_end * kl_raw
        + beta_prior * (prior_rate if prior_rate is not None else 0.0)
        if nll_full is not None and nll_base is not None and kl_raw is not None
        else None
    )

    # Read back off the collection's own record rather than off the model, so the block is
    # identical on the offline path -- which has no model to ask and is the path a re-run's
    # ``results`` must compare equal on.
    geometry = dict(collection.record.get("geometry") or {})
    return {
        "n_parameters": (
            None if task is None
            else int(sum(parameter.numel() for parameter in task.orig_model.parameters()))
        ),
        "train_epoch": None if blob is None else blob.get("epoch"),
        # Which architecture produced the run. Copied out of the checkpoint's own stamp -- the
        # only place it is written -- because a run directory otherwise records it nowhere: the
        # dumped config carries every constructor keyword and not the class they build. A table
        # that ranks two architectures against each other has to key its rows on something, and
        # the alternatives are the directory name, which a rename would relabel, or the geometry,
        # which two arms of one model can differ in. ``None`` on a pass that read no checkpoint.
        "model_class": None if blob is None else blob.get("model_class"),
        # The population every number below was computed over, and the geometry it was trained at.
        "anchor_geometry": {
            name: geometry.get(name)
            for name in (
                "anchor_phase",
                "anchor_stride",
                "training_anchor_stride",
                "anchor_floor",
                "anchors_per_sample",
                "anchor_first",
                "anchor_last",
            )
        },
        "target_axis": {
            name: geometry.get(name)
            for name in ("target_declared_width", "target_kept_width", "horizon", "block_width")
        },
        "warmup_budget": (
            None if task is None else warmup_budget_record(task.orig_model)
        ),
        "anchor_coverage_frac": coverage,
        "observed_loss_scale": {
            "nll_full_block": nll_full,
            "nll_base_block": nll_base,
            "source_conditioned_kl_raw": kl_raw,
            "prior_rate": prior_rate,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
            "beta_end": beta_end,
            "beta_prior": beta_prior,
            "free_bits": float(weights.get("free_bits", 0.0)),
            "main_loss_estimate": estimate,
            # The raw KL stands in for the trained one, which the evaluation deliberately does
            # not aggregate; the two coincide at the shipped free_bits: 0.0 and the estimate is a
            # lower bound otherwise.
            "note": (
                "training-path per-anchor magnitudes recombined with the objective's weights at "
                "the end-of-ramp beta, for re-deriving the spike breaker's additive_margin; the "
                "raw KL stands in for the trained one (exact at free_bits 0.0), and the prior "
                "scale rate contributes only when the collection recorded it (exact at "
                "beta_prior 0.0). Every block magnitude here is per anchor over H*C_keep "
                "coefficients, so it is comparable only against a run at the same warm-up budget."
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
    binding: ModelBinding = CFS_BINDING,
) -> int:
    """Evaluate a checkpoint -- or re-read a finished run -- and write ``summary.json``.

    **The binding decides which model is rebuilt**, which override delta is merged when none is
    given, which constructor keys preflight reconciles, and which analyses run on top of the
    shared registry. Everything else here is architecture-independent, which is what lets the
    second cfs cell reuse this function instead of copying it.

    Everything that shapes the run comes from the merged configuration's ``eval_config`` block --
    the seed, the Monte Carlo draw count, the three verdict thresholds -- rather than from
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
        binding: The model being evaluated. Defaults to this package's, so every existing call
            site says nothing extra.

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
        # The delta is the binding's own when none was named, so the default follows the model
        # rather than whichever module the merge happens to live in: each package owns a
        # self-contained delta documenting its own launch commands, and an operator's repointed
        # copy of one must not be read for a run of the other.
        config, overridden = merge_eval_overrides_with_provenance(
            run_config, binding.overrides_path if overrides is None else overrides
        )
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
    # Built once and used for the selection, the run loop and the record below, so the three
    # cannot disagree about what this model's registry holds.
    registry = merged_analysis_functions(binding)
    selected = select_analyses(tuple(registry), only, skip)

    seed = int(eval_config["seed"])
    numerics = configure_numerics(seed)
    resolved_device = resolve_device(device)

    results_dir = make_output_dir(config, output_dir, binding=binding)
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
            task = load_task(checkpoint_path, resolved_device, blob=blob, binding=binding)
            # Outside any failure isolation: a refusal must reach the operator as a refusal
            # rather than as a step that happened to fail, and must leave no summary behind it.
            preflight_record = preflight.run_preflight(
                config=config,
                model=task.orig_model,
                checkpoint_path=checkpoint_path,
                model_kwargs=blob.get("model_kwargs") or {},
                hyper_parameters=dict(task.hparams),
                binding=binding,
            )
            preflight.write_preflight(preflight_record, results_dir)
        else:
            preflight_record = read_preflight(results_dir)

        # The model is what was trained, so it -- not the config -- is asked what guard it
        # carries; a pass with no model reads the figure back off what the tables were collected
        # under, which is the same number by construction. Preflight refuses a reach budget on
        # this cell, so the value is structurally zero here and is read rather than assumed: an
        # arm that acquired one would report it instead of quietly shifting every lag.
        delay_steps = 0 if task is None else int(task.orig_model.source_delay_steps)
        # The reference is NOT source_delay_steps and is deliberately read from somewhere else.
        # That scalar is the largest stored-step shift, which under a channel alignment belongs to
        # the *fastest* channel; the reference is the physical instant every aligned source channel
        # reports at a step, and is resolved from the shards rather than from the gate. Only the
        # second is a constant a lag in seconds can be computed from, and only the first builds the
        # stored-coefficient axis every figure here draws -- so both travel, and neither stands in
        # for the other.
        reference_delay_s = (
            (preflight_record.get("causality") or {}).get("source_reference_delay_s")
        )
        if delay_steps:
            logger.info(
                f"source channels are delayed by up to {delay_steps} steps "
                f"({delay_steps * SECONDS_PER_STEP:g} s); the reported lag adds this back as an "
                f"upper bound."
            )
        if reference_delay_s is not None:
            logger.info(
                f"source channels are aligned onto a common reference of "
                f"{float(reference_delay_s):g} s; the lag axis below stays in stored-coefficient "
                f"time and does NOT apply it."
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
            report, selected, registry,
            context=context, eval_config=eval_config, output_dir=results_dir,
            probe=probe_record,
        )
        # The one criterion the collection pass cannot finish, because it is decided on a bootstrap
        # interval over recordings that only the source-null analysis produces. Absent that
        # analysis the pass's own INCONCLUSIVE stands, which is the honest answer rather than a
        # pass by default.
        revised = revise_clock_verdict(report.results, eval_config=eval_config)
        if revised is not None:
            logger.info(
                f"{revised['name']}: {revised['status']} -- "
                f"coupling minus clock {revised['values'].get('coupling_minus_clock_nats')} "
                f"nats/anchor, interval "
                f"[{revised['values'].get('interval_lo')}, "
                f"{revised['values'].get('interval_hi')}]"
            )

        # Only on CUDA, and **absent** rather than zero on CPU: a 0.00 GB peak reads as "measured,
        # and the run used no memory", which is a claim a CPU box cannot make.
        if resolved_device.type == "cuda":
            report.set(
                "max_memory_allocated_gb",
                float(torch.cuda.max_memory_allocated(resolved_device)) / (1024**3),
            )

        # The run facts the arm tables read: parameter count, training epoch, both geometry
        # blocks, the resolved budget, the coverage distribution and the observed objective
        # magnitude. Outside `results`, because the checkpoint facts a model-free re-run cannot
        # reproduce are in it.
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
            # What this model's own analyses put in the headline, appended to the shared registry.
            # Empty for a model that registers none, which is why the block below is identical for
            # a run that adds nothing.
            headline_scalars=binding.headline_scalars,
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
            # The alignment reference, beside the maximum and not merged into it. See the comment
            # at the read site: they are a physical constant and a stored-step count, and a reader
            # who takes one for the other gets a lag wrong by minutes with nothing failing.
            "source_reference_delay_s": reference_delay_s,
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
            # no anchors, what was retained under which cap, the streamed accumulators, and the
            # anchor and target-channel geometry an offline analysis cannot re-derive.
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
    # Whichever task the binding named; every read below is through the shared interface.
    task: Optional[Any],
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
    # rather than the model, so the decoder pass -- four scored branches over H*C_keep
    # coefficients per anchor at K draws, plus the fifth KL-only arm -- happens exactly once per
    # run directory.
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
        prog="python -m teb_vae.lag_attn_cfs.eval.run",
        description="Evaluate a trained SeqVaeLagAttnCfs checkpoint.",
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
    # Both flags take a comma-separated list of registered analysis names. An unknown name raises
    # at startup rather than silently running everything (--only) or nothing extra (--skip).
    # An unskippable step is NOT valid for either: it always runs and is not selectable.
    #
    # Interpolated from the registry the *default binding* merges rather than from the shared
    # ANALYSES tuple, because that merged registry is what `main` selects against: this cell's own
    # three analyses are registered on CFS_BINDING, so a help text built from the shared fifteen
    # would tell an operator that `--only warmup` is invalid while the run accepts it.
    selectable = ", ".join(merged_analysis_functions(CFS_BINDING)) or "(none registered yet)"
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
    argv: Optional[Sequence[str]] = None,
    run_args: Optional[Dict[str, Any]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Resolve the argument set from the command line and :data:`RUN_ARGS`, **per key**.

    Per key rather than all-or-nothing: the common iteration is varying one thing, and a fallback
    that was discarded the moment any flag appeared would make the launch dict useless for exactly
    that. So ``--checkpoint other.ckpt`` overrides that one value and leaves the output directory
    and the analysis selection to the dict.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv[1:]``.
        run_args: The fallback dict. ``None`` uses the module-level :data:`RUN_ARGS`.
        parser: The parser to read the valid ``dest`` set from and to parse with. ``None`` builds
            this package's. The second cfs cell passes its own, so its ``prog=`` names it in a
            usage error and its ``--only`` help lists *its* registry -- while the resolution rule
            below stays one implementation for both.

    Returns:
        ``(values, sources)``, where ``sources`` maps each key to ``'cli'``, ``'config'`` or
        ``'default'``. Recorded in ``summary.json``, so a run's provenance is unambiguous after
        the fact rather than reconstructed from a shell history.

    Raises:
        ValueError: If ``run_args`` carries a key that is not an argparse ``dest``. A typo there
            would otherwise silently do nothing, which is the same class of failure the
            ``eval_config`` validator guards against.
    """
    return launch.resolve_launch_args(
        build_parser() if parser is None else parser,
        RUN_ARGS if run_args is None else run_args,
        argv,
    )


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
#: **Running this file directly needs exactly one of two things filled in below**: ``checkpoint``,
#: or an ``output_dir`` naming a finished run whose tables the analyses re-read. With both left
#: ``None`` the run refuses at startup, because there is then neither a model to collect the
#: tables with nor tables to read. Nothing else is required: the working directory is moved to the
#: repository root for you, and every other value falls back to the merged configuration.
#:
#: Do not add run settings here. The seed, the caps, the draw count and the availability-clock
#: margin belong in the override delta, which is dumped into the run directory as the durable
#: record; a value injected from Python would appear in no artifact and could not be recovered
#: from the output afterwards.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,
    "output_dir": None,
    "overrides": None,
    "device": None,
    "num_samples": None,
    "max_batches": None,
    # Which analyses run. Both keys take a comma-separated string of the names below --
    # ``"forecast,coupling"`` -- and both default to ``None``, which runs **every** one of them,
    # in the order listed. `band_partition` runs regardless and is not selectable by either key;
    # naming it raises rather than being read as a typo. An unknown name raises at startup too,
    # before the checkpoint is loaded, so a misspelling costs a parse rather than a first pass
    # over the shards.
    #
    #   forecast:         Is the forecast any good. Skill against persistence, climatology and
    #                     the segment's own mean, in the loader's z units, by horizon step.
    #   coupling:         What the source added. `pred_gap` per recording in both estimators,
    #                     with a paired Wilcoxon, bootstrap intervals and the positive fraction.
    #   perm_control:     Is it *this* recording's source. The GUID-aware shuffle control, whose
    #                     verdict is three losses and deliberately not the KL.
    #   latent:           The per-dimension KL spectrum and active dimensions -- plus the
    #                     prior-variance-pinned detector that catches an inflated coupling number.
    #   lag_kl:           Where in the past the source informed the future. The per-lag KL
    #                     attribution in its raw, support-corrected and untruncated forms.
    #   attention:        The attention per head, and its entropy against the ceiling the measured
    #                     lag support actually allows rather than against an assumed log L.
    #   calibration:      Is the decoder's learned variance the spread of its own errors. PIT,
    #                     coverage, CRPS. An `mse` checkpoint records a skip.
    #   residual:         How far apart the two forecasts are, in z units, and the two
    #                     latent-drift quantities behind them.
    #   distributions:    The shape of each metric over 20-minute segments, by cohort. Histograms
    #                     at both levels; descriptive only, and deliberately tests nothing.
    #   trajectory:       The readouts against time -- within one segment, and assembled across a
    #                     whole delivery on the absolute time axis.
    #   time_to_delivery: The readouts binned on a 0.5 h grid of time before delivery,
    #                     class-stratified, with Holm across windows.
    #   events:           Contraction-conditioned coupling. One readout of the raw pipeline's
    #                     three; the two that scored a bpm waveform have no analogue here.
    #   sufficiency:      What the latent bottleneck costs, against an evaluation-only oracle
    #                     decoder. The one analysis whose cost is a training loop, not a forward.
    #   samples:          Per-recording fifteen-row diagnostic PDF pages -- a stratified draw, plus
    #                     the extremes of each headline metric. Needs a checkpoint; skips without.
    #   warmup:           What the causal front end cost. The gap by warm-up tertile, the source
    #                     lag warmth, and the two FAIL-able geometry guards.
    #   source_null:      How much of the coupling readout survives zeroing the source -- the
    #                     availability-clock hazard no permutation of rows can see.
    #   spectral_skill:   The forecast gap resolved by the frequency band of the target
    #                     coefficient, joined through the kept-axis channel map.
    #   cross_subgroup:   Do the cohorts actually differ. Kruskal, Holm, then Mann-Whitney, over
    #                     the per-recording CSVs the analyses above it wrote, so it runs last.
    "only": None,
    "skip": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
