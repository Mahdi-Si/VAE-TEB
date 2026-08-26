r"""Evaluate one checkpoint end to end: preflight, probe, analyses, report.

Two launch paths, one ``main``.

**Command line**, from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_attn.eval.run \
        --config teb_vae/lag_attn/eval/configs/eval.yaml \
        --checkpoint /path/to/lag-attn-epoch=412.ckpt

**An IDE's Run button**, with no run configuration and no command line: :data:`RUN_ARGS` at the
bottom of this file supplies whatever the command line did not. Resolution is **per key** --
``--checkpoint other.ckpt`` overrides that one value and leaves the config, the output
directory and the analysis selection to the dict -- because the common iteration is varying
one thing, and an all-or-nothing fallback would discard the dict the moment any flag appeared.
``trainer.py`` establishes this pattern with a single ``RUN_CONFIG``; eval extends it to every
argument because an eval run varies its checkpoint far more often than its config.

The dict is a launch convenience, not a second configuration surface. It carries only what
argparse carries, and a key that is not an argparse ``dest`` raises at startup. Everything that
shapes the *run* -- seed, caps, bands, batch size -- stays in the YAML, which is dumped into
the run directory and forms the durable record; a value injected from Python would appear in
neither that dump nor MLflow. Which source supplied each value is logged and written into
``summary.json``, so a run's provenance is unambiguous after the fact.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` and `train.` imports below would fail
# with ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn.eval.run` from the repo root sets __package__ and needs none of
# this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn.eval import band_partition, metrics, preflight  # noqa: E402
from teb_vae.lag_attn.eval.analyses import attention as attention_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import calibration as calibration_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import cross_subgroup as cross_subgroup_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import forecast as forecast_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import frequency_band as frequency_band_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import kld_time_to_delivery as kld_time_to_delivery_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import lag_ablation as lag_ablation_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import latent as latent_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import perm_control as perm_control_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import probe as probe_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import residual as residual_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import samples as samples_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import scalars as scalars_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import te_lag as te_lag_analysis  # noqa: E402
from teb_vae.lag_attn.eval.analyses import uplift as uplift_analysis  # noqa: E402
from teb_vae.lag_attn.eval.config_schema import validate_eval_config  # noqa: E402
from teb_vae.lag_attn.eval.figures import configure_figure_style  # noqa: E402
from teb_vae.lag_attn.eval.numerics import configure_numerics  # noqa: E402
from teb_vae.lag_attn.eval.report import SUMMARY_FILENAME, Report  # noqa: E402
from teb_vae.lag_attn.eval.runner import EvalRunner  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402

#: Analyses ``--only`` and ``--skip`` select between, in run order. Grows with each sprint. The
#: loader probe is not on it: it validates the input and records the per-file composition every
#: stratified capped draw stratifies over, so skipping it would remove the run's only coverage
#: record and silently unstratify every cap.
#:
#: Every entry takes the same arguments and returns a JSON-safe summary, so registering one is a
#: line here rather than a branch in ``main``.
ANALYSIS_FUNCTIONS: Dict[str, Any] = {
    "forecast": forecast_analysis.run_forecast_analysis,
    "frequency_band": frequency_band_analysis.run_frequency_band_analysis,
    "uplift": uplift_analysis.run_uplift_analysis,
    "residual": residual_analysis.run_residual_analysis,
    "scalars": scalars_analysis.run_scalar_analysis,
    "attention": attention_analysis.run_attention_analysis,
    "te_lag": te_lag_analysis.run_te_lag_analysis,
    "latent": latent_analysis.run_latent_analysis,
    # Latent's time-resolved companion: the same per-segment KL, cut by time to delivery and
    # tested across clinical classes. Runs its own forward pass, so it needs no other analysis.
    "kld_time_to_delivery": kld_time_to_delivery_analysis.run_kld_time_to_delivery_analysis,
    "calibration": calibration_analysis.run_calibration_analysis,
    "perm_control": perm_control_analysis.run_perm_control_analysis,
    "lag_ablation": lag_ablation_analysis.run_lag_ablation_analysis,
    "samples": samples_analysis.run_samples_analysis,
    # Last, and that ordering is load-bearing: it reads the per-sample CSVs every analysis above
    # wrote. It is *not* declared in ANALYSIS_DEPENDENCIES, because the dependency is on those
    # files existing on disk rather than on the analyses having run in this pass -- which is what
    # makes `--only cross_subgroup --output-dir <a finished run>` work, with no forward pass.
    "cross_subgroup": cross_subgroup_analysis.run_cross_subgroup_analysis,
}

ANALYSES: Tuple[str, ...] = tuple(ANALYSIS_FUNCTIONS)

#: Analyses that cannot produce correct results unless another analysis ran in the same pass.
#:
#: **Empty, and that is a finding rather than an omission.** Every analysis here takes the same
#: three inputs -- the runner, the loader, and the loader probe's record -- and none reads
#: another's return value: ``analyses/__init__.py`` states the rule and the import graph keeps
#: it. The probe, which every capped analysis genuinely does depend on for its stratification, is
#: deliberately not selectable, so it cannot be skipped in the first place.
#:
#: The check exists anyway because the cost of the table being wrong is asymmetric. An analysis
#: added later that *does* consume another's output has one line to write here; without the
#: mechanism it would instead produce quietly wrong numbers under ``--only``, which is exactly
#: the failure mode ``--only`` is most used in -- re-running one analysis after a multi-hour run
#: failed at its ninth step.
#:
#: Only a *correctness* dependency belongs here, never an interpretive convenience. Several
#: analyses read better beside another -- the scalar pass's collapse verdict is easier to trust
#: next to the residual distribution -- but none of them is *wrong* alone, and over-declaring
#: would turn ``--only`` into a flag that keeps refusing.
ANALYSIS_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {}

#: Subdirectory of the run directory receiving every artifact.
RESULTS_DIRNAME = "eval_results"


# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Every ``dest`` here is also a valid :data:`RUN_ARGS` key."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn.eval.run",
        description="Evaluate a trained SeqVaeLagAttn checkpoint.",
    )
    parser.add_argument("--config", default=None, help="Path to the eval YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Path to the checkpoint to evaluate.")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Run directory. Default: a timestamped directory under out_dir_base/tag.",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device. Default: cuda:0 when available, else cpu."
    )
    parser.add_argument(
        "--max-samples",
        dest="max_samples",
        type=int,
        default=None,
        help="Cap on test samples. Overrides eval_config.max_samples.",
    )
    # Valid names for --only / --skip are the keys of ANALYSIS_FUNCTIONS above, in run order:
    #   forecast, frequency_band, uplift, residual, scalars, attention, te_lag, latent,
    #   kld_time_to_delivery, calibration, perm_control, lag_ablation, samples, cross_subgroup
    # Both flags take a comma-separated list, so several can be named at once (e.g.
    # --only latent,kld_time_to_delivery  or  --skip samples,perm_control). `probe` and
    # `band_partition` are NOT valid: they always run and are not selectable. An unknown name
    # raises at startup rather than silently running everything (--only) or nothing extra (--skip).
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated analyses to run exclusively (names from ANALYSIS_FUNCTIONS).",
    )
    parser.add_argument(
        "--skip", default=None,
        help="Comma-separated analyses to skip (names from ANALYSIS_FUNCTIONS).",
    )
    return parser


def resolve_arguments(
    argv: Optional[Sequence[str]] = None, run_args: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Resolve the argument set from the command line and :data:`RUN_ARGS`, per key.

    For each argument the command line wins when the flag was passed, otherwise ``RUN_ARGS``
    supplies it, otherwise the documented default applies. A required argument absent from both
    is an error naming both sources.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv[1:]``.
        run_args: The fallback dict. ``None`` uses the module-level :data:`RUN_ARGS`.

    Returns:
        ``(values, sources)`` where ``sources`` maps each key to ``'cli'``, ``'RUN_ARGS'`` or
        ``'default'``.

    Raises:
        ValueError: If ``run_args`` carries a key that is not an argparse ``dest`` -- otherwise
            a typo there silently does nothing, which is the same class of failure the YAML
            validator guards against.
        SystemExit: Via ``parser.error`` when ``config`` or ``checkpoint`` is absent from both
            sources.
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
            f"configuration surface -- settings that shape the run belong in the YAML, where "
            f"they are dumped into the run directory."
        )

    parsed = vars(parser.parse_args(list(argv) if argv is not None else None))

    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for key in sorted(valid_dests):
        if parsed.get(key) is not None:
            values[key], sources[key] = parsed[key], "cli"
        elif fallback.get(key) is not None:
            values[key], sources[key] = fallback[key], "RUN_ARGS"
        else:
            values[key], sources[key] = None, "default"

    for required in ("config", "checkpoint"):
        if values[required] is None:
            parser.error(
                f"--{required} is required. Pass it on the command line, or set "
                f"RUN_ARGS[{required!r}] near the bottom of "
                f"teb_vae/lag_attn/eval/run.py to launch from an IDE Run button."
            )
    return values, sources


def select_analyses(
    available: Sequence[str],
    only: Optional[str],
    skip: Optional[str],
    *,
    dependencies: Optional[Dict[str, Tuple[str, ...]]] = None,
) -> List[str]:
    """Resolve ``--only`` / ``--skip`` into the ordered list of analyses to run.

    Args:
        available: Every analysis the pipeline knows about, in run order.
        only: Comma-separated names to run exclusively, or ``None``.
        skip: Comma-separated names to skip, or ``None``.
        dependencies: Declared inter-analysis dependencies. ``None`` uses
            :data:`ANALYSIS_DEPENDENCIES`.

    Returns:
        The analyses to run, in ``available`` order.

    Raises:
        ValueError: If a named analysis does not exist -- a misspelling would otherwise silently
            run everything (``--only``) or nothing extra (``--skip``) -- or if the resulting
            subset drops an analysis another selected one needs, which would produce output that
            looks complete and cannot be correctly read.
    """
    known = list(available)
    declared = ANALYSIS_DEPENDENCIES if dependencies is None else dependencies

    def _parse(raw: Optional[str], flag: str) -> List[str]:
        if not raw:
            return []
        names = [name.strip() for name in str(raw).split(",") if name.strip()]
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                f"--{flag} names unknown analyses: {unknown}. Available: {known or '(none yet)'}."
            )
        return names

    requested = set(_parse(only, "only")) or set(known)
    excluded = set(_parse(skip, "skip"))
    # Filtered out of ``known`` rather than returned in the order the operator typed: the run
    # order is the pipeline's, and some later analyses will consume what earlier ones produced.
    selected = [name for name in known if name in requested and name not in excluded]

    # Checked after the subset is final rather than per flag, because --only and --skip can
    # each individually be innocent and jointly drop a dependency.
    unmet = [
        f"{name} needs {needed!r}"
        for name in selected
        for needed in declared.get(name, ())
        if needed not in selected
    ]
    if unmet:
        raise ValueError(
            f"the selected analyses are missing a declared dependency: {'; '.join(unmet)}. "
            f"Selected: {selected}. Running the subset anyway would produce output that looks "
            f"complete but cannot be read correctly. Add the named analysis to --only, or drop "
            f"it from --skip."
        )
    return selected


# ---------------------------------------------------------------------------
# Run directory and loader
# ---------------------------------------------------------------------------
def _preserve_prior_summary(results_dir: Path) -> List[Path]:
    r"""Rename a finished run's summary aside before this pass overwrites it.

    ``--output-dir <a finished run>`` is the documented way to re-run one analysis against an
    existing directory, and ``analyses/cross_subgroup.py`` needs exactly that -- it reads the
    per-sample CSVs the other analyses already wrote. But ``report.write`` opens ``summary.json``
    with mode ``'w'`` and ``build_manifest(since=started_at)`` classifies every earlier file as
    stale, so a one-analysis re-run replaced a complete summary with one whose ``headline`` was
    entirely ``null`` and whose manifest listed four files instead of forty-seven -- and exited 0.

    The per-sample CSVs survive that, but the sanity block, the coverage record, the resolved
    geometry and the two promoted verdicts (``collapse``, ``source_specificity``) exist nowhere
    else, so the loss is silent and total. Renaming rather than merging is the smaller fix: the
    new pass still writes a truthful summary of *itself*, and the prior one is recoverable.

    Args:
        results_dir: The ``eval_results`` directory this pass is about to write into.

    Returns:
        The backup paths created, in the order attempted. Empty when the directory is fresh.
    """
    stamp = datetime.now().strftime("%Y-%m-%d--[%H-%M-%S]")
    preserved: List[Path] = []
    for filename in (SUMMARY_FILENAME, preflight.PREFLIGHT_FILENAME):
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
            f"preserved the prior run's copy as {backup.name}. Note that a partial re-run's "
            f"summary describes only the analyses this pass actually ran."
        )
    return preserved


def make_output_dir(config: Dict[str, Any], explicit: Optional[Any] = None) -> Path:
    """Return the run directory, creating it.

    Timestamped to *second* resolution with a numeric collision guard, so two runs launched in
    the same minute -- which is a normal thing to do while iterating on a checkpoint -- cannot
    write into each other's directory.

    An explicit directory is used as given, which is what makes the documented single-analysis
    re-run possible -- and also what makes it destructive: ``report.write`` opens ``summary.json``
    with mode ``'w'``, and the artifact manifest classifies every file older than this pass as
    stale. Re-running one analysis into a finished run therefore replaced a complete summary with
    one whose headline is entirely ``null``, and exited 0. The prior summary is preserved here
    instead, because it is the only place the sanity block, the coverage record and the two
    promoted verdicts exist.

    Args:
        config: The merged run config.
        explicit: An explicit directory, which is used as given.

    Returns:
        The ``eval_results`` directory inside the run directory.
    """
    if explicit is not None:
        results_dir = Path(explicit) / RESULTS_DIRNAME
        results_dir.mkdir(parents=True, exist_ok=True)
        _preserve_prior_summary(results_dir)
        return results_dir

    general = config.get("general_config") or {}
    base = Path(str((general.get("folders_config") or {}).get("out_dir_base", "output")))
    tag = str(general.get("tag", "lag_attn_eval"))
    stamp = datetime.now().strftime("%Y-%m-%d--[%H-%M-%S]")

    candidate = base / tag / stamp
    suffix = 2
    while candidate.exists():
        candidate = base / tag / f"{stamp}-{suffix}"
        suffix += 1
    results_dir = candidate / RESULTS_DIRNAME
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def force_single_process_loader(config: Dict[str, Any]) -> Dict[str, Any]:
    """Force ``num_workers`` to 0, warning when the config asked otherwise.

    Not tuning. ``create_optimized_dataloader`` hardcodes ``persistent_workers=True`` whenever
    ``num_workers > 0``, and with spawn multiprocessing over a multi-file HDF5 dataset the
    workers degrade after the first full iteration, silently truncating the second pass to the
    first file's index range. Eval makes many passes over one loader, so it meets that failure
    every time; in the predecessor it presented as "only 1 class found" and cost days.

    Args:
        config: The merged run config. Mutated in place.

    Returns:
        The same config.
    """
    loader_config = (config.setdefault("dataset_config", {})).setdefault("dataloader_config", {})
    requested = loader_config.get("num_workers", 0)
    if requested:
        logger.warning(
            f"dataloader_config.num_workers={requested} overridden to 0: with spawn workers "
            f"over a multi-file HDF5 dataset the loader silently truncates its second pass to "
            f"the first file's index range, and eval makes many passes."
        )
    loader_config["num_workers"] = 0
    loader_config["persistent_workers"] = False
    return config


def dump_resolved_config(config: Dict[str, Any], output_dir: Path) -> Path:
    """Write the merged config into the run directory as its durable record.

    Args:
        config: The merged run config.
        output_dir: The results directory.

    Returns:
        The path written.
    """
    path = output_dir / "resolved_config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    logger.info(f"wrote {path}")
    return path


def configure_logging(output_dir: Path, config: Dict[str, Any]) -> None:
    """Add a file sink beside the run's artifacts, keeping the default console sink.

    Args:
        output_dir: The results directory.
        config: The merged run config, for ``advanced_config.logging`` levels.
    """
    logging_config = (config.get("advanced_config") or {}).get("logging") or {}
    logger.add(
        str(output_dir / "eval.log"),
        level=str(logging_config.get("file_level", "INFO")),
        rotation=str(logging_config.get("rotation", "100 MB")),
        retention=str(logging_config.get("retention", "14 days")),
        enqueue=False,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(
    config: Any,
    checkpoint: Any,
    output_dir: Optional[Any] = None,
    device: Optional[Any] = None,
    max_samples: Optional[int] = None,
    only: Optional[str] = None,
    skip: Optional[str] = None,
    argument_sources: Optional[Dict[str, str]] = None,
) -> int:
    """Run the pipeline against one checkpoint.

    Args:
        config: Path to the eval YAML config.
        checkpoint: Path to the checkpoint to evaluate.
        output_dir: Explicit run directory, or ``None`` for the timestamped default.
        device: Torch device, or ``None`` to auto-select.
        max_samples: Sample cap overriding ``eval_config.max_samples``.
        only: Comma-separated analyses to run exclusively.
        skip: Comma-separated analyses to skip.
        argument_sources: Which source supplied each argument, recorded in ``summary.json``.

    Returns:
        The process exit code: non-zero when any step failed.
    """
    started_at = time.time()
    merged_config = load_config(str(config))

    # Folded into eval_config rather than threaded separately, because every analysis reads its
    # cap from this block: a --max-samples that only reached the probe would bound the coverage
    # record and nothing else, while the flag says it overrides eval_config.max_samples.
    #
    # Injected *before* validation rather than assigned onto the resolved block afterwards, so
    # the override goes through the same ``minimum=1`` bound the YAML value does. Assigned
    # afterwards it bypassed that check entirely: ``--max-samples 0`` and ``--max-samples -1``
    # both evaluated a single batch (``seen >= 0`` fires after the first yield), which is
    # indistinguishable in the output from the legal ``--max-samples 1`` -- so a ``-1`` meaning
    # "no cap" was silently reinterpreted as the smallest run the pipeline can do.
    if max_samples is not None:
        block = merged_config.get("eval_config")
        merged_config["eval_config"] = {
            **(block if isinstance(block, dict) else {}),
            "max_samples": max_samples,
        }

    # Before the model, the loader or the output directory: a misspelled eval_config key must
    # cost a parse, not a checkpoint load and a first pass over the shards.
    eval_config = validate_eval_config(merged_config)
    force_single_process_loader(merged_config)

    selected = select_analyses(ANALYSES, only, skip)
    numerics = configure_numerics(int(eval_config["seed"]))

    results_dir = make_output_dir(merged_config, output_dir)
    configure_logging(results_dir, merged_config)
    logger.info(f"eval run directory: {results_dir}")
    dump_resolved_config(merged_config, results_dir)

    report = Report()
    report.set("checkpoint", str(checkpoint))
    report.set("config", str(config))
    report.set("output_dir", str(results_dir))
    report.set("numerics", numerics)
    report.set("eval_config", eval_config)
    report.set("arguments", {
        "values": {
            "config": str(config),
            "checkpoint": str(checkpoint),
            "output_dir": None if output_dir is None else str(output_dir),
            "device": None if device is None else str(device),
            "max_samples": max_samples,
            "only": only,
            "skip": skip,
        },
        "sources": dict(argument_sources or {}),
    })
    report.set("analyses_selected", selected)

    runner = EvalRunner.from_checkpoint(checkpoint, results_dir, device=device)
    report.set("geometry", runner.geometry())
    report.set("objective", runner.objective.as_dict())

    # Hard-fail guards. Deliberately NOT inside report.step: a run that cannot be trusted must
    # not produce a summary that looks like a result. They run before any analysis, so a
    # rejected run leaves a directory holding its *inputs* -- the resolved config and the log,
    # which are what one needs to see why it was rejected -- and no results of any kind.
    preflight_record = preflight.run_preflight(config=merged_config, runner=runner)

    data_module = GraphDataModule(merged_config)
    loader = data_module.test_dataloader()

    health_batch = preflight.first_batch(loader)
    if health_batch is None:
        raise RuntimeError(
            "the test loader yielded no batches, so nothing can be evaluated. Check "
            "dataset_config.vae_test_datasets and the dataset_kwargs filters in the resolved "
            "config dumped into this run directory."
        )
    preflight_record["health_probe"] = preflight.probe_load_health(
        runner, runner.to_device(health_batch), floor=float(eval_config["health_probe_floor"])
    )
    # Every lag figure converts its axis with this offset, and ``metrics.lag_to_seconds`` says the
    # value used is recorded here -- so record it, rather than leaving the claim to be checked
    # against a key that was never written. The sign is carried forward unverified on purpose:
    # the dataset adaptor is built with up_shift_secs=-20 and DESIGN.md reads a peak at lag l as a
    # delay of 4(l + 5) seconds, which agree, but plotting.py's own axis applies no offset at all,
    # so no figure in this repository has ever actually used one.
    preflight_record["lag_seconds_convention"] = {
        "up_shift_secs": float(eval_config["up_shift_secs"]),
        "step_seconds": float(metrics.STEP_SECONDS),
        "formula": "seconds = step_seconds * lag - up_shift_secs",
        "sign_verified": False,
        "note": (
            "the physical-seconds axis is provisional until the sign is pinned against the "
            "dataset pipeline documentation; the model-lag axis is exact regardless."
        ),
    }
    preflight.write_preflight(preflight_record, results_dir)
    report.set("preflight", preflight_record)

    sample_cap = eval_config["max_samples"]

    probe_record = report.step(
        "probe",
        probe_analysis.run_probe,
        runner,
        loader,
        configured_files=(merged_config.get("dataset_config") or {}).get("vae_test_datasets"),
        max_samples=sample_cap,
        output_dir=results_dir,
    )
    if probe_record is not None:
        report.set("probe", probe_analysis.summary_view(probe_record))

    # The channel map, from the shards' own provenance. Emitted here rather than inside an
    # analysis because it describes the *data*, not the model, and nothing about it depends on
    # a forward pass. It skips cleanly on a shard without sel_* attributes -- see emit_partition.
    band_record = report.step(
        "band_partition",
        band_partition.emit_partition,
        (merged_config.get("dataset_config") or {}).get("vae_test_datasets") or [],
        int(runner.build_target_streams(health_batch)[0].shape[-1]),
        results_dir,
    )
    if band_record is not None:
        report.set("band_partition", band_record)

    # Once, here, rather than as an import side effect: apply_publication_style mutates global
    # rcParams, and an import that did it would restyle any figure produced in the same process.
    # The run's figure format is fixed in the same call and for the same reason -- it is a
    # property of the pass, and every figure written after this line follows it.
    configure_figure_style(eval_config["figure_format"])

    # Each analysis under report.step, so one failure does not discard the rest of a multi-hour
    # run. The probe's record is threaded through because it carries the sample count and the
    # per-file grouping every capped draw stratifies over.
    for name in selected:
        result = report.step(
            name,
            ANALYSIS_FUNCTIONS[name],
            runner,
            loader,
            eval_config=eval_config,
            output_dir=results_dir,
            probe=probe_record,
        )
        if result is not None:
            report.set(name, result)

    # Promoted to a top-level field rather than left inside the scalar pass's block: it is the
    # run's headline conclusion, and a reader should not have to know which analysis produced it.
    scalar_summary = report.results.get("scalars")
    if isinstance(scalar_summary, dict) and "collapse" in scalar_summary:
        report.set("collapse", scalar_summary["collapse"])

    # Likewise promoted: source specificity is the question the model exists to answer, and it
    # must not be reachable only by knowing which analysis produced it.
    perm_summary = report.results.get("perm_control")
    if isinstance(perm_summary, dict) and "specificity" in perm_summary:
        report.set("source_specificity", perm_summary["specificity"])

    # Only on CUDA, and absent rather than zero on CPU: a 0.00 GB peak reads as "measured, and
    # the run used no memory", which is a claim this pipeline cannot make on a CPU box.
    if runner.device.type == "cuda":
        import torch

        report.set(
            "max_memory_allocated_gb",
            float(torch.cuda.max_memory_allocated(runner.device)) / (1024**3),
        )

    report.finalise(results_dir, analyses=selected, started_at=started_at)
    logger.info(report.console_table())
    report.write(results_dir)
    return report.exit_code()


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key: a flag passed on the command line wins
#: for that key alone and leaves the rest to this dict, so varying only the checkpoint works
#: without editing anything here. A key that is not an argparse ``dest`` raises at startup.
#:
#: Do not add run settings here. Seed, caps, bands and batch size belong in the YAML, which is
#: dumped into the run directory as the durable record; a value injected from Python would
#: appear in neither that dump nor MLflow.
RUN_ARGS: Dict[str, Any] = {
    "config": "teb_vae/lag_attn/eval/configs/eval.yaml",
    "checkpoint": None,
    "output_dir": None,
    "device": None,
    "max_samples": None,
    # Comma-separated analysis names, or None for all. Valid names are the keys of
    # ANALYSIS_FUNCTIONS: forecast, frequency_band, uplift, residual, scalars, attention, te_lag,
    # latent, kld_time_to_delivery, calibration, perm_control, lag_ablation, samples,
    # cross_subgroup. Several may be listed (e.g. "latent,kld_time_to_delivery"); `probe` and
    # `band_partition` always run and are not selectable.
    "only": None,
    "skip": None,
}


if __name__ == "__main__":
    _values, _sources = resolve_arguments()

    # Repo-root-relative, because that is the convention every documented invocation uses and
    # an IDE's working directory is not something this module can rely on. The paths *inside* a
    # config are repo-root-relative too, so the chdir is what makes a relative shard path
    # resolve the same way under a Run button as on the command line.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    logger.info(
        "resolved arguments: "
        + ", ".join(f"{key}={_values[key]!r} (from {_sources[key]})" for key in sorted(_values))
    )
    sys.exit(main(**_values, argument_sources=_sources))
