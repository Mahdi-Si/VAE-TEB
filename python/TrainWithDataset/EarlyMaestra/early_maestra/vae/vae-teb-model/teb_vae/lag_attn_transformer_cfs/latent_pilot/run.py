r"""The pilot's single entry point: an editable dictionary, a Run button, and every stage.

**The IDE Run button is a first-class requirement, not a convenience.** The surrounding package's
``trainer.py`` already establishes the pattern -- a module-level constant naming what an
argument-less launch should do, a guarded repository-root ``sys.path`` bootstrap before the package
imports, and path resolution that does not depend on the IDE's working directory. This runner
extends it from a single config path to an editable **dictionary**: open this file, edit
:data:`RUN_ARGS` at the bottom, hit Run. No CLI flags, no launcher script, no IDE working-directory
setting.

Three launch modes, one resolver, so none of them can drift:

.. code-block:: bash

    python teb_vae/lag_attn_transformer_cfs/latent_pilot/run.py            # uses RUN_ARGS as edited
    python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run --stage tests
    python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run \
        --config teb_vae/lag_attn_transformer_cfs/latent_pilot/configs/pilot.yaml --stage all

and programmatically, ``main(**RUN_ARGS)``. The direct-file mode is the one that needs the guarded
bootstrap below: running a file puts *its own* directory on ``sys.path`` rather than the repository
root, so every ``teb_vae.`` import would fail before ``__main__`` was reached. This file sits one
level deeper than ``trainer.py``, so the root is four directories up rather than three -- a depth
that is easy to copy wrongly and impossible to notice afterwards, because the symptom is an import
error that looks like a broken environment.

Precedence, per key: an explicitly supplied command-line value beats a non-``None``
:data:`RUN_ARGS` entry, which beats the YAML, which beats the built-in defaults. A parser default is
*not* a value -- it is the absence of one -- so an unsupplied flag never outranks the file.
``--set a.b=c`` supplies structured pilot overrides, merged over the dictionary's ``overrides``
block; its value is parsed as YAML, so ``--set optim.max_epochs=3`` is an integer and
``--set paths.train_shards=[a.hdf5,b.hdf5]`` is a list. Neither :func:`main` nor
:func:`resolve_run_args` mutates :data:`RUN_ARGS` or anything nested inside it: everything is
deep-copied on the way in, and the effective arguments are persisted with the run.

Importing this module reads nothing, builds nothing, creates no directory and parses no arguments.

Stages, every one of them selectable through ``RUN_ARGS["stage"]`` and the same guard: ``tests``,
``smoke``, ``preflight``, ``extract``, ``baseline``, ``finetune``, ``control``, ``evaluate``,
``report``, ``all``. ``all`` runs::

    tests -> smoke -> preflight -> extract(train/validation)
          -> baseline -> finetune -> control
          -> freeze selection/settings -> evaluate(test + analyses) -> report

``tests`` invokes pytest with the current interpreter from a fixed repository-root working
directory. ``smoke`` uses its own small non-clinical fixture configuration and output subtree and
dispatches the pipeline internally -- it never recurses into ``all`` and never touches production
arguments or artifacts. Both stop the sequence on failure. Every runtime stage consumes the saved
outputs of the stages before it, rejects artifacts that do not match its own fingerprints, and fails
with a clear message when a prerequisite is missing: asking for ``report`` never silently trains a
model.

The stage bodies are registered in :data:`STAGE_HANDLERS` and each of them takes the one shared
``context`` dictionary, reads what it needs off disk, and writes its own artifacts back. Four rules
they all obey, each of which is a way a multi-stage pipeline reports something that did not happen:

* **A stage reads the cohort rather than re-deriving it.** The recording table is established once,
  by ``preflight``, and every later stage reads it back: two derivations of "which recordings are
  in" are two chances for one of them to keep a recording the other dropped.
* **A checkpoint bundle is loaded fresh wherever it is needed.** ``finetune`` adapts the weights of
  the bundle it holds, so a bundle cached across stages would hand ``evaluate`` an adapted model
  under the name ``pretrained``. Loading twice costs seconds; the alternative costs the comparison.
* **The test split is opened only after the lock.** ``evaluate`` locks the selection first and
  passes the lock's own permission into the extraction, so the ordinary path to the held-out split
  runs through the check rather than beside it.
* **A missing prerequisite is a refusal, never a silent redo.** Asking for ``report`` in a directory
  that never fitted a model says so and stops.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/latent_pilot/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead of
# the repository root, and every absolute import below fails before __main__ is reached. Launching
# as `python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run` sets __package__ and needs none
# of this, which is why the whole block is guarded rather than unconditional.
#
# Two things have to be true, and adding the root is only the first of them. This package's own
# modules are named `train`, `model`, `data`, `config`, `report`, `evaluate` and `run`, and the
# repository has a top-level `train` package -- so while the script's directory stays on the path
# ahead of the root, `import train.graph_models_utils` finds `latent_pilot/train.py` and fails as
# "No module named 'train.pl_model_base'; 'train' is not a package", several stages into a run.
# Merely testing `_REPO_ROOT not in sys.path` is not enough either: an inherited PYTHONPATH (which
# is what a PyCharm remote interpreter sets) already carries the root further down the list, where
# it loses to the script's directory at position zero.
if not __package__:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [
        entry for entry in sys.path
        if os.path.abspath(entry or os.getcwd()) != _SCRIPT_DIR
    ]
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)

import yaml  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import _deep_merge  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.latent_pilot import config as pilot_config  # noqa: E402

#: The pilot configuration used when nothing names another one.
DEFAULT_CONFIG_PATH = "teb_vae/lag_attn_transformer_cfs/latent_pilot/configs/pilot.yaml"

#: Where a smoke run lands inside the operator's configured run root. A sibling of the production
#: folds rather than a directory inside one: a smoke run is its own run, with its own protocol and
#: its own stage state, and burying it in a production run directory would put two runs' records in
#: one place.
SMOKE_SUBDIR = "smoke"

#: The fixture configuration the ``smoke`` stage runs under. Never the production one: a smoke run
#: must not be able to read production shards or write into a production run directory.
SMOKE_CONFIG_PATH = "teb_vae/lag_attn_transformer_cfs/latent_pilot/configs/smoke.yaml"

#: What an unspecified run argument means. A key absent from here is not a run argument, and naming
#: one raises rather than being carried along to be ignored.
RUN_ARG_DEFAULTS: Dict[str, Any] = {
    "config_path": DEFAULT_CONFIG_PATH,
    "stage": pilot_config.ALL_STAGE,
    "device": None,
    "run_dir": None,
    "resume": False,
    "overrides": {},
}

#: The production stages, in the order they consume each other's artifacts. ``tests`` and ``smoke``
#: are deliberately absent: this is the sequence the smoke stage dispatches internally, and putting
#: either of them here would make the smoke stage run itself.
PIPELINE_STAGES: List[str] = [
    "preflight", "extract", "baseline", "finetune", "control", "evaluate", "report",
]

#: The two model versions every paired artifact is filed under. ``pretrained`` is the checkpoint as
#: it was loaded; ``adapted`` is the selected candidate, which may *be* the pretrained model when
#: selection retained epoch zero -- and the report says so rather than the filename.
PRETRAINED = "pretrained"
ADAPTED = "adapted"

#: The frozen model's preservation reading on the gate subset, measured before any candidate is
#: fitted and used as the gates' reference throughout the fit.
GATE_REFERENCE = "gate_reference"

#: Artifacts the evaluation stage writes for the report stage to render from. The report stage
#: refits nothing, so everything it draws has to be here.
PREFLIGHT_FILENAME = "preflight.json"
RESULTS_FILENAME = "results.json"
METRICS_FILENAME = "metrics.csv"
BIN_TABLE_FILENAME = "per_recording_bin.parquet"
BAND_TABLE_FILENAME = "trajectory_bands.parquet"
TEST_TABLE_FILENAME = "per_recording_test.parquet"
BAG_VALUES_FILENAME = "figure_bags.npz"

#: One row per held-out recording observed in both the early and the late window, per model: the
#: two scores, their difference and the class behind it. Written so the acidosis/HIE split of the
#: paired change stays auditable from the run directory rather than only from the report.
PAIRED_TABLE_FILENAME = "per_recording_paired_windows.parquet"

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Every ``dest`` is also a :data:`RUN_ARGS` key.

    Every default is ``None`` on purpose, including the boolean: the resolver has to tell an
    unsupplied flag from one supplied at its default value, and a parser default of ``False`` would
    make ``--no-resume`` indistinguishable from saying nothing -- which would silently outrank a
    ``resume: true`` in the dictionary.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.run",
        description=(
            "Run the latent-class fine-tuning pilot. With no arguments the RUN_ARGS dictionary at "
            "the bottom of run.py is used, which is what an IDE Run button launches."
        ),
    )
    parser.add_argument(
        "--config", dest="config_path", default=None,
        help=f"Pilot YAML. Relative to the repository root. Default: {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--stage", default=None,
        help=(
            "Stage to run: " + ", ".join(pilot_config.STAGES)
            + f", or {pilot_config.ALL_STAGE!r} for the whole ordered sequence."
        ),
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device, e.g. cuda:0 or cpu. Default: the config's, else chosen at runtime.",
    )
    parser.add_argument(
        "--run-dir", dest="run_dir", default=None,
        help="An existing run directory, for resuming or for regenerating a report from saved "
             "artifacts. Default: a new directory under the configured run root.",
    )
    parser.add_argument(
        "--resume", default=None, action=argparse.BooleanOptionalAction,
        help="Continue the run named by --run-dir instead of starting a new one.",
    )
    parser.add_argument(
        "--set", dest="set_overrides", default=None, action="append", metavar="KEY=VALUE",
        help="Pilot setting override, e.g. --set optim.max_epochs=3. Repeatable. The value is "
             "parsed as YAML, so numbers, booleans, null and [a,b] lists all work. Architecture "
             "and preprocessing are not settable: those keys do not exist and are refused.",
    )
    return parser


def _nest(dotted: str, value: Any) -> Dict[str, Any]:
    """Turn ``a.b.c`` and a value into ``{'a': {'b': {'c': value}}}``."""
    node: Any = value
    for part in reversed(dotted.split(".")):
        node = {part: node}
    return node


def _parse_set_overrides(entries: Sequence[str]) -> Dict[str, Any]:
    """Fold ``--set KEY=VALUE`` entries into one nested override mapping.

    Args:
        entries: The raw ``KEY=VALUE`` strings, in the order given.

    Returns:
        The merged overrides. Later entries win over earlier ones.

    Raises:
        PilotConfigError: If an entry has no ``=``, has an empty key, or carries a value that is
            not valid YAML. A silently dropped override is worse than a refused one: the run would
            proceed under settings the operator believes they changed.
    """
    merged: Dict[str, Any] = {}
    for entry in entries:
        key, separator, raw = str(entry).partition("=")
        key = key.strip()
        if not separator or not key:
            raise pilot_config.PilotConfigError(
                f"--set expects KEY=VALUE, got {entry!r}. The key is a dotted settings path, e.g. "
                f"--set optim.max_epochs=3."
            )
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise pilot_config.PilotConfigError(
                f"--set {entry!r} carries a value that is not valid YAML: {error}"
            ) from error
        merged = _deep_merge_overrides(merged, _nest(key, value))
    return merged


def _deep_merge_overrides(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge override mappings under the repository's one merge rule.

    The same function the config resolver merges with, so a partial override behaves identically
    whether it arrives from the dictionary, from ``--set``, or from the YAML's own base chain.

    Args:
        base: The lower-precedence mapping. Not mutated.
        over: The higher-precedence mapping. Not mutated.

    Returns:
        A new mapping sharing no value with either input: nested dicts merge key by key, lists and
        scalars replace.
    """
    return _deep_merge(dict(base), dict(over))


def _validate_run_args(values: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    """Check one set of run arguments, whatever supplied it.

    One function for the dictionary and the command line, so the two cannot come to accept
    different things: a value that is refused when typed is refused when written down.

    Args:
        values: The arguments to check.
        source: Where they came from, for every message.

    Returns:
        The arguments, with paths left as given and ``overrides`` copied.

    Raises:
        PilotConfigError: On an unknown key, a wrong type, an unknown stage, or ``resume`` without
            a run directory to resume into.
    """
    unknown = sorted(set(values) - set(RUN_ARG_DEFAULTS))
    if unknown:
        raise pilot_config.PilotConfigError(
            f"unknown run argument(s) from {source}: "
            f"{', '.join(repr(key) for key in unknown)}. Valid keys: "
            f"{', '.join(sorted(RUN_ARG_DEFAULTS))}. Pilot *settings* are not run arguments: they "
            f"belong in the YAML or in the 'overrides' mapping."
        )

    resolved = {**deepcopy(RUN_ARG_DEFAULTS), **{k: v for k, v in values.items() if v is not None}}

    if not isinstance(resolved["config_path"], (str, os.PathLike)):
        raise pilot_config.PilotConfigError(
            f"config_path (from {source}) must be a path, got "
            f"{type(resolved['config_path']).__name__}."
        )
    # Raises with the list of valid stages when it is not one.
    pilot_config.stage_plan(resolved["stage"])
    for name in ("device", "run_dir"):
        value = resolved[name]
        if value is not None and not isinstance(value, (str, os.PathLike)):
            raise pilot_config.PilotConfigError(
                f"{name} (from {source}) must be a string or a path, got {type(value).__name__}."
            )
    if not isinstance(resolved["resume"], bool):
        raise pilot_config.PilotConfigError(
            f"resume (from {source}) must be a boolean, got {type(resolved['resume']).__name__}."
        )
    if not isinstance(resolved["overrides"], Mapping):
        raise pilot_config.PilotConfigError(
            f"overrides (from {source}) must be a mapping of pilot settings, got "
            f"{type(resolved['overrides']).__name__}."
        )
    resolved["overrides"] = deepcopy(dict(resolved["overrides"]))
    if resolved["resume"] and resolved["run_dir"] is None:
        raise pilot_config.PilotConfigError(
            f"resume was requested (from {source}) with no run_dir. There is nothing to resume: a "
            f"new run starts with resume left false, and resuming names the directory to continue."
        )
    return resolved


def resolve_run_args(
    run_args: Optional[Mapping[str, Any]] = None,
    *,
    argv: Optional[Sequence[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> Dict[str, Any]:
    """Resolve the dictionary and the command line into :func:`main`'s keyword arguments.

    Args:
        run_args: The editable dictionary. Defaults to :data:`RUN_ARGS`. Never mutated -- neither
            it nor its nested ``overrides`` mapping.
        argv: Command-line arguments, excluding the program name. ``None`` skips parsing entirely,
            which is the programmatic and Run-button path.
        parser: The parser to use. Defaults to :func:`build_parser`.

    Returns:
        Keyword arguments for :func:`main`.

    Raises:
        PilotConfigError: If either source carries an unknown key, a wrong type, an unknown stage,
            a malformed ``--set`` entry, or ``resume`` without a run directory.
    """
    base = _validate_run_args(
        deepcopy(dict(RUN_ARGS if run_args is None else run_args)), source="RUN_ARGS"
    )
    if argv is None:
        return base

    parsed = vars((parser or build_parser()).parse_args(list(argv)))
    entries = parsed.pop("set_overrides", None)
    # Only what was actually typed: a parser default is the absence of a value, not a value, so it
    # must not outrank the dictionary.
    supplied = {key: value for key, value in parsed.items() if value is not None}
    if entries:
        supplied["overrides"] = _deep_merge_overrides(
            base["overrides"], _parse_set_overrides(entries)
        )
    if not supplied:
        return base
    return _validate_run_args({**base, **supplied}, source="the command line")


# =============================================================================
# What every stage shares
# =============================================================================
def _load_checkpoint(settings: Mapping[str, Any], *, freeze: bool = True) -> Any:
    """Rebuild the pretrained checkpoint strictly, and freeze it for this pilot.

    **Loaded fresh at every call, never cached in the context.** The fitting stage adapts the
    weights of the bundle it holds, so a bundle shared across stages would hand a later stage an
    adapted model under the name ``pretrained`` -- and every paired comparison in this run would be
    between two models that differ by nothing.

    Args:
        settings: The resolved settings.
        freeze: Apply the pilot's freeze and evaluation mode. Off only where the caller is about to
            install adapted weights and freeze afterwards.

    Returns:
        The loaded bundle.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model

    loaded = pilot_model.load_pilot_checkpoint(
        settings["paths"]["checkpoint"], device=settings["device"]
    )
    if freeze:
        pilot_model.freeze_for_pilot(loaded.model)
    return loaded


def _loader_config(settings: Mapping[str, Any], loaded: Any, split: str) -> Dict[str, Any]:
    """The loader configuration for one split, from the checkpoint's own resolved config.

    Args:
        settings: The resolved settings.
        loaded: The loaded bundle, whose configuration and geometry are the authority.
        split: ``'train'``, ``'val'`` or ``'test'``.

    Returns:
        The configuration, with this split's shards, the configured statistics and the coarse epoch
        bound. The bound is derived from the checkpoint's own sequence length and trim rather than
        from the length today's shards happen to have.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data

    configured = settings["windows"]["segment_span_seconds"]
    span = (
        float(configured) if configured is not None
        else data.segment_span_seconds(
            int(loaded.geometry["sequence_length"]), loaded.geometry["trim_minutes"]
        )
    )
    return data.pilot_loader_config(
        loaded.config,
        shards=settings["paths"][f"{split}_shards"],
        statistics=settings["paths"]["statistics"],
        epoch_min=data.coarse_epoch_min(
            float(settings["windows"]["preservation_hours"]), span
        ),
    )


def _loader(settings: Mapping[str, Any], loaded: Any, split: str) -> Any:
    """One split's re-iterable dataloader.

    Args:
        settings: The resolved settings.
        loaded: The loaded bundle.
        split: The split.

    Returns:
        The dataloader.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data

    return data.split_loader(_loader_config(settings, loaded, split))


def _extraction_name(split: str, version: str) -> str:
    """The name one split's latents are filed under for one model version.

    Split first, so a directory listing groups the two readings of the held-out split together --
    which is what a reader compares.

    Args:
        split: The split.
        version: :data:`PRETRAINED` or :data:`ADAPTED`.

    Returns:
        The artifact prefix.
    """
    return f"{split}_{version}"


def _attach_eligibility(
    recordings: Any, retained: Any, *, splits: Sequence[str], settings: Mapping[str, Any]
) -> Any:
    """Attach late eligibility to the recordings of the splits an extraction covers.

    Eligibility is a statement about retained anchors, so it can only be decided for the splits that
    have been extracted -- and it is decided **per split**, because judging every recording against
    one split's anchors would mark the rest ineligible for having no anchors in a pass that never
    read them.

    Args:
        recordings: The recording table.
        retained: The retained anchors of the covered splits.
        splits: Which splits those anchors cover.
        settings: The resolved settings; the supervised window and the eligibility rules.

    Returns:
        A new table: the covered splits judged, every other row carried through untouched.
    """
    import pandas as pd

    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data

    wanted = {str(split) for split in splits}
    covered = recordings[recordings[data.SPLIT_COLUMN].astype(str).isin(wanted)]
    others = recordings[~recordings[data.SPLIT_COLUMN].astype(str).isin(wanted)]
    judged = data.late_eligibility(
        retained,
        covered,
        supervised_hours=float(settings["windows"]["supervised_hours"]),
        min_late_segments=int(settings["eligibility"]["min_late_segments"]),
        final_anchor_within_minutes=float(
            settings["eligibility"]["final_anchor_within_minutes"]
        ),
    )
    merged = pd.concat([judged, others], ignore_index=True) if len(others) else judged
    return merged.sort_values(data.GUID_COLUMN).reset_index(drop=True)


def _rewrite_cohort(recordings: Any, directory: Any) -> None:
    """Persist the recording table and rewrite the manifest and coverage from it.

    Called by every stage that learns something new about the cohort -- which for this pilot means
    eligibility, and only after the anchors it rests on have been read.

    Args:
        recordings: The recording table.
        directory: The run directory.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data

    data.write_cohort_table(recordings, directory, name=data.RECORDINGS_FILENAME)
    data.write_manifest(recordings, directory)
    segments = data.read_cohort_table(directory, name=data.SEGMENTS_FILENAME)
    data.write_coverage(data.coverage_summary(segments, recordings), directory)


def _bags(
    extraction: Any, recordings: Any, *, split: str, settings: Mapping[str, Any],
    key: str = "mu_post",
) -> Any:
    """One split's supervised bags at this run's window and recency weighting."""
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train

    return pilot_train.build_bags(
        extraction,
        recordings,
        split=split,
        supervised_hours=float(settings["windows"]["supervised_hours"]),
        halflife_hours=float(settings["bag"]["halflife_hours"]),
        key=key,
    )


def _bin_summaries(extraction: Any, recordings: Any, *, split: str, settings: Mapping[str, Any]):
    """One split's per-recording, per-bin summaries at this run's bins."""
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze

    return analyze.bin_summaries(
        extraction,
        recordings,
        split=split,
        bin_hours=float(settings["windows"]["bin_hours"]),
        preservation_hours=float(settings["windows"]["preservation_hours"]),
    )


# =============================================================================
# The stages
# =============================================================================
def stage_tests(context: Dict[str, Any]) -> Dict[str, Any]:
    """Run the pilot's own test suite with the current interpreter.

    From a fixed repository-root working directory, because every path in the suite and in the
    shipped configurations is root-relative and an IDE's working directory is not.

    The end-to-end smoke scenario is **excluded** here and run by the next stage instead: it drives
    the whole pipeline, so running it inside ``tests`` would fit every model twice per ``all``.

    Args:
        context: The run context.

    Returns:
        The command, its exit code and where it ran.

    Raises:
        RuntimeError: If pytest is not installed, or if any test failed. Either stops the sequence,
            which is the point of running the suite first.
    """
    import importlib.util
    import subprocess

    if importlib.util.find_spec("pytest") is None:
        raise RuntimeError(
            f"pytest is not installed in the environment this interpreter runs in "
            f"({sys.executable}), so the test stage cannot run. It is the only package this pilot "
            f"needs beyond the ones the repository already uses."
        )
    tests = pilot_config.PILOT_ROOT / "tests"
    command = [
        sys.executable, "-m", "pytest", str(tests), "-q",
        f"--ignore={tests / 'test_smoke.py'}",
    ]
    logger.info(f"tests: {' '.join(command)}")
    finished = subprocess.run(command, cwd=str(pilot_config.REPO_ROOT))
    if finished.returncode != 0:
        raise RuntimeError(
            f"the pilot test suite failed (pytest exit code {finished.returncode}). The sequence "
            f"stops here rather than fitting a model against code its own tests reject."
        )
    return {
        "command": command,
        "returncode": int(finished.returncode),
        "working_directory": str(pilot_config.REPO_ROOT),
        "excluded": "test_smoke.py, which the smoke stage runs",
    }


def stage_smoke(context: Dict[str, Any]) -> Dict[str, Any]:
    """Run the whole pipeline on the small non-clinical fixtures, in its own output subtree.

    Isolated in what it measures: its own configuration, which names its own fixture shards and its
    own ``fold``, and its own settings resolved from that file rather than inherited -- the outer
    run's device, checkpoint, statistics, shards and protocol overrides never reach it. It
    dispatches :data:`PIPELINE_STAGES` directly and so cannot recurse into ``all``.

    **One leaf is inherited, and only one: the destination.** The outer run's ``paths.run_root``
    becomes this run's, under :data:`SMOKE_SUBDIR`, because where a run writes is the operator's
    choice and a stage that ignored it would scatter output between the configured location and
    this package. Nothing that shapes a measurement travels with it, and the smoke run still gets a
    directory of its own, so it cannot overwrite the production run that invoked it.

    The fixtures it runs on are **written here when they are absent**, rather than being an
    operator step this stage assumes has happened. They are generated, git-ignored and reproducible,
    so a checkout that has never run them is the ordinary case rather than an error -- and the
    alternative was the failure this replaced: a missing-input refusal listing six fixture paths,
    from a stage whose whole claim is that it needs no data. Present fixtures are never rewritten:
    regenerating them would spend the fit again and would move the ground under a smoke run that
    already read them.

    Args:
        context: The run context. Read for nothing but the log line: a smoke run that inherited a
            production argument would not be a wiring check of the shipped configuration.

    Returns:
        Where the fixture run was written, the stages it ran, and whether the fixtures were written
        by this call.

    Raises:
        PilotConfigError: If generation finishes without producing something the smoke
            configuration names -- a drift between that file and the generator, which is what
            :func:`~...tests.fixtures.generate.manifest_matches` exists to name.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures import generate as fixtures

    destination = Path(context["settings"]["paths"]["run_root"]) / SMOKE_SUBDIR
    overrides = {"paths": {"run_root": str(destination)}}
    # Resolved here rather than taken from the outer run, for the same reason the pipeline below
    # re-resolves it: the smoke configuration is what these fixtures have to satisfy. The one
    # override is the destination, so this reading and the pipeline's agree on where output goes.
    settings = pilot_config.resolve_settings(SMOKE_CONFIG_PATH, overrides=overrides)
    generated = bool(pilot_config.missing_inputs(settings, list(PIPELINE_STAGES)))
    if generated:
        logger.info(
            "smoke: the fixtures this stage runs on are absent, writing them now. They are "
            "artificial and git-ignored, and the checkpoint among them is a real one-epoch fit -- "
            "expect minutes, once per checkout."
        )
        manifest = fixtures.generate()
        undelivered = fixtures.manifest_matches(manifest, settings)
        if undelivered:
            raise pilot_config.PilotConfigError(
                "the fixture generator finished without writing "
                + ", ".join(repr(path) for path in undelivered)
                + f", which {SMOKE_CONFIG_PATH} names. The configuration and the generator have "
                f"drifted apart: one of them names a subgroup or a directory the other does not."
            )
        logger.info(
            f"smoke: fixtures written under {manifest['root']}. They stay there rather than "
            f"following the run root: they are a per-checkout cache, and re-rooting them would "
            f"re-run the fit every time an operator changed where output goes."
        )

    logger.info(
        f"smoke: dispatching {', '.join(PIPELINE_STAGES)} under {SMOKE_CONFIG_PATH}, "
        f"writing to {destination}"
    )
    smoke = run_pipeline(SMOKE_CONFIG_PATH, overrides=overrides)
    return {
        "config_path": SMOKE_CONFIG_PATH,
        "run_root": str(destination),
        "run_dir": str(smoke["run_dir"]),
        "stages": list(PIPELINE_STAGES),
        "fixtures_generated": generated,
        "clinical": False,
        "note": (
            "artificial identities, times and labels; a finished smoke run is evidence that the "
            "stages connect and the artifacts round-trip, and evidence of nothing else"
        ),
    }


def stage_preflight(context: Dict[str, Any]) -> Dict[str, Any]:
    """Establish the cohort, the provenance and the geometry, without fitting anything.

    This is the stage that reads the shards' metadata and decides who is in the run: identities,
    class codes, split disjointness, patient grouping, exposure, class presence and coverage. It
    fits nothing and it writes the tables every later stage reads back.

    Eligibility is **not** decided here: it is a statement about retained anchors, and no anchor has
    been read yet.

    Args:
        context: The run context.

    Returns:
        The cohort record: counts, exclusions, exposure, grouping and the checkpoint's geometry.
    """
    import json

    import pandas as pd

    from teb_vae.lag_attn.eval.report import json_safe
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model

    settings, directory = context["settings"], Path(context["run_dir"])

    # Eligibility is decided by `extract` and lives in the same table this stage writes. Rewriting
    # that table after extraction has run would drop the `eligible` column, and a missing column
    # reads downstream as "no eligibility rule at all" -- so the baseline, the fit and the gate
    # subset would quietly be taken over recordings the declared coverage rules excluded, with
    # normal-looking counts and nothing logged. Refuse instead; the operator can start a new run.
    if "extract" in pilot_config.read_stage_state(directory).get("completed", []):
        raise pilot_config.RunStateError(
            f"{directory} already has a completed 'extract' stage, whose eligibility columns live "
            f"in the same recording table this stage rewrites. Re-running 'preflight' here would "
            f"drop them, and every later stage reads a missing 'eligible' column as no "
            f"eligibility rule -- fitting on recordings the coverage rules excluded. Start a new "
            f"run directory, or re-run from 'extract' onwards."
        )

    loaded = _load_checkpoint(settings)
    statistics = pilot_model.statistics_record(
        settings["paths"]["statistics"],
        trim_minutes=loaded.geometry["trim_minutes"],
        checkpoint_stat_path=loaded.geometry.get("checkpoint_stat_path"),
    )

    frames = []
    for split in ("train", "val", "test"):
        if not settings["paths"][f"{split}_shards"]:
            logger.info(f"preflight: no {split!r} shards configured, so that split is not read")
            continue
        frames.append(
            data.segment_frame(_loader(settings, loaded, split), split=split)
        )
    segments = pd.concat(frames, ignore_index=True)
    recordings = data.recording_frame(segments)
    recordings, grouping = data.attach_patient_groups(
        recordings, patient_map=settings["provenance"]["patient_map"]
    )
    # On the patient column, which falls back to the GUID when no mapping was supplied -- so this
    # is the stronger check wherever a mapping exists and the same one everywhere else.
    data.check_split_disjoint(recordings, group_column=data.PATIENT_COLUMN)
    exposure = data.exposure_record(
        recordings,
        pretraining_guids=data.load_guid_list(settings["provenance"]["pretraining_guids"]),
        selection_guids=data.load_guid_list(settings["provenance"]["selection_guids"]),
        statistics_population=settings["provenance"]["statistics_population"],
    )
    # Before eligibility, so this asks the smaller question: does the cohort carry both classes at
    # all? The eligible-only version of it runs once eligibility exists.
    data.require_both_classes(
        recordings,
        splits=sorted({str(value) for value in recordings[data.SPLIT_COLUMN]}),
        eligible_only=False,
    )

    data.write_cohort_table(segments, directory, name=data.SEGMENTS_FILENAME)
    data.write_cohort_table(recordings, directory, name=data.RECORDINGS_FILENAME)
    data.write_manifest(recordings, directory)
    data.write_coverage(data.coverage_summary(segments, recordings), directory)

    record = {
        "checkpoint": {
            "checkpoint": str(loaded.checkpoint_path),
            "checkpoint_digest": loaded.digest,
            "model_class": loaded.geometry.get("model_class"),
            "d_z": loaded.geometry.get("d_z"),
            "geometry": dict(loaded.geometry),
        },
        "statistics": statistics,
        "exposure": exposure,
        "grouping": grouping,
        # Section 4.1: whether the shards were built in holdout or augmented mode. Recorded from
        # the operator's setting and left as null when they did not supply it -- the two modes
        # partition differently, so an unrecorded one is a gap in what a held-out claim rests on
        # rather than something a run may infer from a path.
        "dataset_build_mode": settings["provenance"]["dataset_build_mode"],
        "n_segments": int(len(segments)),
        "n_recordings": int(len(recordings)),
        "exclusions": data.exclusion_counts(recordings),
        "splits": sorted({str(value) for value in recordings[data.SPLIT_COLUMN]}),
    }
    # Persisted, not merely returned: the stages run one at a time as often as they run in one
    # sequence, and a report assembled in a later process must still be able to say which
    # checkpoint produced it and what is known about exposure.
    (directory / PREFLIGHT_FILENAME).write_text(
        json.dumps(json_safe(record), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        f"preflight: {record['n_recordings']} recording(s) over {record['n_segments']} segment(s); "
        f"exclusions {record['exclusions']}"
    )
    return record


def stage_extract(context: Dict[str, Any]) -> Dict[str, Any]:
    """Read the pretrained latents of the fitting splits, decide eligibility, freeze the scaler.

    The training and validation splits only. The held-out split is extracted by ``evaluate``, after
    the selection lock, and :func:`~latent_pilot.extract.extract_split` refuses it here.

    Args:
        context: The run context.

    Returns:
        The extraction record: the support fingerprint, the per-split counts and the scaler.
    """
    import pandas as pd

    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract

    settings, directory = context["settings"], Path(context["run_dir"])
    pilot_config.require_completed(context["state"], "preflight", needed_by="extract")
    recordings = data.read_cohort_table(directory, name=data.RECORDINGS_FILENAME)
    loaded = _load_checkpoint(settings)

    extractions = {}
    for split in ("train", "val"):
        extraction = extract.extract_split(
            loaded,
            _loader(settings, loaded, split),
            split=split,
            preservation_hours=float(settings["windows"]["preservation_hours"]),
            bin_hours=float(settings["windows"]["bin_hours"]),
        )
        extract.save_extraction(
            extraction, directory, name=_extraction_name(split, PRETRAINED)
        )
        extractions[split] = extraction

    recordings = _attach_eligibility(
        recordings,
        pd.concat(
            [extractions["train"].retained, extractions["val"].retained], ignore_index=True
        ),
        splits=("train", "val"),
        settings=settings,
    )
    data.require_both_classes(recordings, splits=("train", "val"), eligible_only=True)
    _rewrite_cohort(recordings, directory)

    scaler = extract.fit_scaler(
        extractions["train"].retained, extractions["train"].arrays["mu_post"]
    )
    extract.save_scaler(scaler, directory)

    return {
        "fingerprint": dict(extractions["train"].fingerprint),
        "splits": {
            split: dict(extraction.record) for split, extraction in extractions.items()
        },
        "scaler": dict(scaler.record),
        "n_eligible": {
            split: len(data.eligible_outcomes(recordings, split=split))
            for split in ("train", "val")
        },
    }


def stage_baseline(context: Dict[str, Any]) -> Dict[str, Any]:
    """Fit the frozen linear baseline on the pretrained bags.

    No model is built here at all: the bags are pooled posterior means already extracted from the
    pretrained checkpoint, so nothing in this stage can move a latent coordinate.

    Args:
        context: The run context.

    Returns:
        The fit's record.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train

    settings, directory = context["settings"], Path(context["run_dir"])
    pilot_config.require_completed(context["state"], "extract", needed_by="baseline")
    recordings = data.read_cohort_table(directory, name=data.RECORDINGS_FILENAME)
    scaler = extract.load_scaler(directory)

    bags = {}
    fingerprints = {}
    for split in ("train", "val"):
        extraction = extract.load_extraction(
            directory, name=_extraction_name(split, PRETRAINED)
        )
        fingerprints[split] = extraction.fingerprint
        bags[split] = _bags(extraction, recordings, split=split, settings=settings)
    extract.check_compatible(
        fingerprints["train"], fingerprints["val"],
        what="the training and validation extractions",
    )

    fit = pilot_train.fit_baseline(
        bags["train"], bags["val"],
        scaler=scaler,
        lr=float(settings["baseline"]["lr"]),
        max_steps=int(settings["baseline"]["max_steps"]),
        patience=int(settings["baseline"]["patience"]),
        weight_decay=float(settings["optim"]["weight_decay"]),
        seed=int(settings["seed"]),
    )
    pilot_train.save_fit(fit, directory, name=pilot_train.BASELINE_NAME)
    return dict(fit.record)


def stage_finetune(context: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt the posterior mean-output layers, select an epoch, and read the adapted latents.

    The gate subset and the frozen model's reading on it are established **before** the fit, so
    every candidate is measured against one reference on one fixed set of recordings. After
    selection the bundle holds the selected weights, so the adapted training and validation latents
    are read here rather than reloaded and re-adapted somewhere else.

    Args:
        context: The run context.

    Returns:
        The selection record, the gate subset and the adapted extraction records.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate, extract
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train

    settings, directory = context["settings"], Path(context["run_dir"])
    pilot_config.require_completed(context["state"], "baseline", needed_by="finetune")
    recordings = data.read_cohort_table(directory, name=data.RECORDINGS_FILENAME)
    scaler = extract.load_scaler(directory)
    baseline = pilot_train.load_fit(directory, name=pilot_train.BASELINE_NAME)
    train_extraction = extract.load_extraction(
        directory, name=_extraction_name("train", PRETRAINED)
    )

    loaded = _load_checkpoint(settings)
    val_loader = _loader(settings, loaded, "val")
    outcomes = evaluate.outcome_map(recordings)

    subset = evaluate.gate_subset(
        recordings,
        size=int(settings["gates"]["subset_recordings"]),
        seed=int(settings["seed"]),
    )
    evaluate.save_gate_subset(subset, directory)
    reference = evaluate.preservation_pass(
        loaded,
        val_loader,
        guids=subset["guids"],
        outcomes=outcomes,
        preservation_hours=float(settings["windows"]["preservation_hours"]),
    )
    evaluate.save_preservation(reference, directory, name=GATE_REFERENCE)

    fit = pilot_train.fit_adaptation(
        loaded,
        source=pilot_train.RecordingSource.from_config(
            _loader_config(settings, loaded, "train"),
            shards=settings["paths"]["train_shards"],
        ),
        plans=pilot_train.build_plans(
            train_extraction, recordings, split="train",
            supervised_hours=float(settings["windows"]["supervised_hours"]),
            halflife_hours=float(settings["bag"]["halflife_hours"]),
        ),
        val_loader=val_loader,
        recordings=recordings,
        scaler=scaler,
        baseline=baseline,
        gate_guids=subset["guids"],
        gate_baseline=reference.record,
        outcomes=outcomes,
        settings=settings,
    )
    pilot_train.save_adapted(
        fit, loaded, directory, fingerprint=train_extraction.fingerprint
    )
    pilot_train.export_base_checkpoint(loaded, directory, selected_epoch=fit.selected_epoch)

    # The bundle now holds the selected epoch's weights, so this is the adapted reading -- and it is
    # taken here, where that is true by construction, rather than reconstructed later.
    adapted = {}
    for split in ("train", "val"):
        extraction = extract.extract_split(
            loaded,
            _loader(settings, loaded, split),
            split=split,
            preservation_hours=float(settings["windows"]["preservation_hours"]),
            bin_hours=float(settings["windows"]["bin_hours"]),
        )
        extract.assert_same_keys(
            extract.load_extraction(directory, name=_extraction_name(split, PRETRAINED)),
            extraction,
        )
        extract.save_extraction(extraction, directory, name=_extraction_name(split, ADAPTED))
        adapted[split] = dict(extraction.record)

    return {
        "selection": dict(fit.record),
        "selected_epoch": int(fit.selected_epoch),
        "adapted": bool(fit.adapted),
        "threshold": float(fit.threshold),
        "gate_subset": dict(subset),
        "gate_reference": dict(reference.record),
        "adapted_extractions": adapted,
        "history": fit.history.to_dict(orient="records"),
    }


def stage_control(context: Dict[str, Any]) -> Dict[str, Any]:
    """Fit the shuffled-label control and, when enabled, the frozen prior probe.

    Both are fitted on the **pretrained** latents. The control gets its own permuted-label
    classifier rather than the true-label baseline's, which would hand it a head already fitted to
    the association it exists to test the absence of.

    Args:
        context: The run context.

    Returns:
        The control's record, the permutation that produced it, and the probe's.
    """
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train

    settings, directory = context["settings"], Path(context["run_dir"])
    pilot_config.require_completed(context["state"], "finetune", needed_by="control")
    recordings = data.read_cohort_table(directory, name=data.RECORDINGS_FILENAME)
    scaler = extract.load_scaler(directory)
    extractions = {
        split: extract.load_extraction(directory, name=_extraction_name(split, PRETRAINED))
        for split in ("train", "val")
    }

    permuted, permutation = pilot_train.control_recordings(
        recordings, seed=int(settings["seed"])
    )
    control = pilot_train.fit_control_baseline(
        extractions["train"], extractions["val"], permuted,
        permutation=permutation, scaler=scaler, settings=settings,
    )
    pilot_train.save_fit(control, directory, name=pilot_train.CONTROL_NAME)

    probe, probe_record = pilot_train.fit_prior_probe(
        extractions["train"], extractions["val"], recordings, settings=settings
    )
    if probe is not None:
        pilot_train.save_fit(probe, directory, name=pilot_train.PRIOR_PROBE_NAME)

    return {
        "control": dict(control.record),
        "permutation": dict(permutation),
        "prior_probe": probe_record,
    }


def stage_evaluate(context: Dict[str, Any]) -> Dict[str, Any]:
    """Lock the selection, then read the held-out split once and measure everything on it.

    The order is the point. Every model, threshold, gate tolerance, control and seed is written into
    the lock **before** the test shards are opened, and the extraction's permission to open them is
    the lock's own check -- so the ordinary path to the held-out split runs through it rather than
    beside it.

    Args:
        context: The run context.

    Returns:
        The complete report record, which is also written to disk for the report stage.
    """
    import json

    import numpy as np
    import pandas as pd

    from teb_vae.lag_attn.eval import labels as eval_labels
    from teb_vae.lag_attn.eval.report import json_safe
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze, data, evaluate, extract
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import report as pilot_report
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train

    settings, directory = context["settings"], Path(context["run_dir"])
    pilot_config.require_completed(context["state"], "control", needed_by="evaluate")
    recordings = data.read_cohort_table(directory, name=data.RECORDINGS_FILENAME)
    scaler = extract.load_scaler(directory)
    baseline = pilot_train.load_fit(directory, name=pilot_train.BASELINE_NAME)
    control = pilot_train.load_fit(directory, name=pilot_train.CONTROL_NAME)
    adapted_checkpoint = pilot_train.load_adapted(directory)
    subset = evaluate.load_gate_subset(directory)
    probe = None
    if (directory / f"{pilot_train.PRIOR_PROBE_NAME}_{pilot_train.FIT_FILENAME}").is_file():
        probe = pilot_train.load_fit(directory, name=pilot_train.PRIOR_PROBE_NAME)

    # Locked once. A retry of this stage -- a crash after the lock, a rerun under resume -- reuses
    # the lock it already wrote rather than replacing it: the choices were frozen at that moment,
    # ``open_run`` has already refused any settings change, and a second lock would be a record
    # written after the held-out split was available.
    locked = {
        "models": {
            PRETRAINED: {"threshold": float(baseline.threshold), "source": "frozen checkpoint"},
            ADAPTED: {
                "threshold": float(adapted_checkpoint.threshold),
                "selected_epoch": int(adapted_checkpoint.selected_epoch),
            },
            pilot_train.CONTROL_NAME: {"threshold": float(control.threshold)},
            pilot_train.PRIOR_PROBE_NAME: (
                None if probe is None else {"threshold": float(probe.threshold)}
            ),
        },
        "gates": dict(settings["gates"]),
        "gate_subset": subset,
        "bootstrap": dict(settings["bootstrap"]),
        "mc": dict(settings["mc"]),
        "seed": int(settings["seed"]),
        "windows": dict(settings["windows"]),
        "projection": "one label-free two-component PCA, fitted on training bin vectors of both "
                      "model versions after the frozen scaler",
        "settings_digest": pilot_config.settings_digest(settings),
    }
    if (directory / pilot_config.SELECTION_LOCK_FILENAME).is_file():
        locked = pilot_config.read_selection_lock(directory)
        logger.info(
            f"selection was already locked at {locked.get('locked_utc')}; this run reuses that "
            f"record rather than writing a second one"
        )
    else:
        pilot_config.lock_selection(directory, locked)

    # The permission and the extraction in one expression, so there is no path to the test split
    # that does not pass the lock.
    allow_test = pilot_config.require_selection_locked(directory)

    bundles = {PRETRAINED: _load_checkpoint(settings), ADAPTED: _load_checkpoint(settings)}
    applied = pilot_train.apply_adapted(adapted_checkpoint, bundles[ADAPTED])

    test_extractions = {}
    for version, loaded in bundles.items():
        extraction = extract.extract_split(
            loaded,
            _loader(settings, loaded, "test"),
            split="test",
            preservation_hours=float(settings["windows"]["preservation_hours"]),
            bin_hours=float(settings["windows"]["bin_hours"]),
            allow_test=allow_test,
        )
        extract.save_extraction(
            extraction, directory, name=_extraction_name("test", version)
        )
        test_extractions[version] = extraction
    extract.assert_same_keys(test_extractions[PRETRAINED], test_extractions[ADAPTED])

    recordings = _attach_eligibility(
        recordings, test_extractions[PRETRAINED].retained,
        splits=("test",), settings=settings,
    )
    data.require_both_classes(recordings, splits=("test",), eligible_only=True)
    _rewrite_cohort(recordings, directory)

    # ---------------------------------------------------------------- held-out discrimination
    test_bags = {
        version: _bags(extraction, recordings, split="test", settings=settings)
        for version, extraction in test_extractions.items()
    }
    if test_bags[PRETRAINED].guids != test_bags[ADAPTED].guids:
        raise pilot_config.PilotConfigError(
            "the two model versions produced bags for different held-out recordings, so a paired "
            "comparison between them would be a comparison of two populations."
        )
    outcomes = test_bags[PRETRAINED].labels
    guids = test_bags[PRETRAINED].guids

    adapted_classifier = adapted_checkpoint.classifier()
    columns = {
        PRETRAINED: baseline.logits(test_bags[PRETRAINED].values),
        ADAPTED: analyze.score_frame(
            test_bags[ADAPTED].frame, test_bags[ADAPTED].values, adapted_classifier
        )[analyze.SCORE_COLUMN].to_numpy(),
        pilot_train.CONTROL_NAME: control.logits(test_bags[PRETRAINED].values),
    }
    thresholds = {
        PRETRAINED: float(baseline.threshold),
        ADAPTED: float(adapted_checkpoint.threshold),
        pilot_train.CONTROL_NAME: float(control.threshold),
    }
    if probe is not None:
        prior_bags = _bags(
            test_extractions[PRETRAINED], recordings, split="test", settings=settings,
            key="mu_prior",
        )
        columns[pilot_train.PRIOR_PROBE_NAME] = probe.logits(prior_bags.values)
        thresholds[pilot_train.PRIOR_PROBE_NAME] = float(probe.threshold)

    metrics = {
        name: evaluate.recording_metrics(
            outcomes, values, threshold=thresholds[name]
        )
        for name, values in columns.items()
    }
    # Clustered resampling only where a real mapping exists. The patient column falls back to the
    # GUID when none was supplied, so passing it unconditionally would report GUID-only grouping as
    # patient grouping and suppress the disclosure that says the interval may be too narrow.
    preflight = (
        json.loads((directory / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
        if (directory / PREFLIGHT_FILENAME).is_file() else {}
    )
    patients = None
    if dict(preflight.get("grouping") or {}).get("grouping") == "patient":
        mapping = {
            str(row[data.GUID_COLUMN]): str(row[data.PATIENT_COLUMN])
            for _index, row in recordings.iterrows()
        }
        patients = {guid: mapping[guid] for guid in guids}
    try:
        bootstrap = evaluate.paired_bootstrap(
            outcomes, columns,
            guids=guids, thresholds=thresholds, patients=patients,
            resamples=int(settings["bootstrap"]["resamples"]),
            seed=int(settings["seed"]),
        )
    except evaluate.GateEvaluationError as error:
        # A cohort too small to resample is a result about this fold, not a software failure, and
        # the report has to be able to tell the two apart.
        bootstrap = {"error": str(error), "n_recordings": int(len(guids))}
        logger.warning(f"paired bootstrap not estimated: {error}")

    test_frame = recordings[
        recordings[data.GUID_COLUMN].astype(str).isin(set(guids))
    ].set_index(data.GUID_COLUMN).loc[guids].reset_index()
    for name, values in columns.items():
        test_frame[f"logit_{name}"] = np.asarray(values, dtype=np.float64)
    subgroups = evaluate.subgroup_table(test_frame, columns, thresholds=thresholds)

    # ---------------------------------------------------------------- geometry, in full latent space
    train_bags = {
        version: _bags(
            extract.load_extraction(directory, name=_extraction_name("train", version)),
            recordings, split="train", settings=settings,
        )
        for version in (PRETRAINED, ADAPTED)
    }
    centroid_metrics = {}
    for version in (PRETRAINED, ADAPTED):
        centroids = analyze.class_centroids(
            scaler.apply(train_bags[version].values), train_bags[version].labels
        )
        centroid_metrics[version] = evaluate.recording_metrics(
            outcomes,
            analyze.nearest_centroid_scores(
                scaler.apply(test_bags[version].values), centroids
            ),
            threshold=0.0,
        )
    geometry = {
        "movement": analyze.movement_summary(
            test_bags[PRETRAINED].values, test_bags[ADAPTED].values, scale=scaler.scale
        ),
        "covariance": {
            version: analyze.covariance_summary(scaler.apply(bags.values))
            for version, bags in test_bags.items()
        },
    }

    # ---------------------------------------------------------------- preservation, side by side
    readings = {}
    for version, loaded in bundles.items():
        reading = evaluate.preservation_pass(
            loaded,
            _loader(settings, loaded, "val"),
            guids=subset["guids"],
            outcomes=evaluate.outcome_map(recordings),
            preservation_hours=float(settings["windows"]["preservation_hours"]),
            mc_draws=int(settings["mc"]["draws"]),
            mc_seed=int(settings["seed"]),
        )
        evaluate.save_preservation(reading, directory, name=version)
        readings[version] = reading.record
    gate = evaluate.gate_decision(
        readings[PRETRAINED], readings[ADAPTED],
        forecast_mse_max_increase=float(settings["gates"]["forecast_mse_max_increase"]),
        saturation_max_increase_pp=float(settings["gates"]["saturation_max_increase_pp"]),
    )
    convergence = None
    if settings["mc"]["large_draws"] is not None:
        large = {
            version: evaluate.preservation_pass(
                loaded,
                _loader(settings, loaded, "val"),
                guids=subset["guids"],
                outcomes=evaluate.outcome_map(recordings),
                preservation_hours=float(settings["windows"]["preservation_hours"]),
                mc_draws=int(settings["mc"]["large_draws"]),
                mc_seed=int(settings["seed"]),
            ).record
            for version, loaded in bundles.items()
        }
        convergence = {
            version: evaluate.nll_convergence(readings[version], large[version])
            for version in readings
        }

    # ---------------------------------------------------------------- trajectories and the paired windows
    classifiers = {PRETRAINED: baseline.classifier, ADAPTED: adapted_classifier}
    supervised = analyze.supervised_bins(
        bin_hours=float(settings["windows"]["bin_hours"]),
        supervised_hours=float(settings["windows"]["supervised_hours"]),
    )
    scored_bins, bands, temporal, paired_frames = [], [], {}, []
    for version, extraction in test_extractions.items():
        frame, values = _bin_summaries(
            extraction, recordings, split="test", settings=settings
        )
        scored = analyze.score_frame(frame, values, classifiers[version])
        scored["model"] = version
        scored_bins.append(scored)
        for group_column in (data.OUTCOME_COLUMN, eval_labels.CLASS_COLUMN):
            band = analyze.group_bands(
                scored, group_column=group_column,
                resamples=int(settings["bootstrap"]["resamples"]),
                seed=int(settings["seed"]), supervised=supervised,
            )
            band["model"], band["grouped_by"] = version, group_column
            bands.append(band)
        paired = analyze.window_scores(
            extraction, recordings, classifiers[version],
            split="test",
            early=settings["windows"]["early_window_hours"],
            late=(0.0, float(settings["windows"]["supervised_hours"])),
        )
        temporal[version] = analyze.paired_contrast(
            paired,
            resamples=int(settings["bootstrap"]["resamples"]),
            seed=int(settings["seed"]),
        )
        # The two subgroups against the same healthy controls, computed together so neither can be
        # promoted to the headline afterwards. Training is binary; these are descriptive, and a
        # stratum with no paired recording yields a nan point with the bootstrap's own note rather
        # than an exception.
        for stratum in ("acidosis", "hie"):
            temporal[f"{version} ({stratum} vs healthy)"] = analyze.paired_contrast(
                paired,
                group_column=eval_labels.CLASS_COLUMN,
                adverse=stratum,
                healthy="healthy",
                resamples=int(settings["bootstrap"]["resamples"]),
                seed=int(settings["seed"]),
            )
        # Persisted: the run directory is how a later process re-reads what a stage established,
        # and without this the subgroup split could not be audited after the fact.
        paired_frames.append(paired.assign(model=version))

    # ---------------------------------------------------------------- the one shared projection
    training_bins = {}
    for version in (PRETRAINED, ADAPTED):
        frame, values = _bin_summaries(
            extract.load_extraction(directory, name=_extraction_name("train", version)),
            recordings, split="train", settings=settings,
        )
        training_bins[version] = (frame, scaler.apply(values))
    projection = analyze.fit_projection(training_bins)
    analyze.save_projection(projection, directory)

    # ---------------------------------------------------------------- what the report will draw from
    scored_bins = pd.concat(scored_bins, ignore_index=True)
    scored_bins.to_parquet(directory / BIN_TABLE_FILENAME, index=False)
    pd.concat(bands, ignore_index=True).to_parquet(directory / BAND_TABLE_FILENAME, index=False)
    pd.concat(paired_frames, ignore_index=True).to_parquet(
        directory / PAIRED_TABLE_FILENAME, index=False
    )
    test_frame.to_parquet(directory / TEST_TABLE_FILENAME, index=False)
    np.savez_compressed(
        directory / BAG_VALUES_FILENAME,
        **{version: scaler.apply(test_bags[version].values) for version in (PRETRAINED, ADAPTED)},
    )
    pd.DataFrame([
        {"model": name, **{key: value for key, value in measured.items()}}
        for name, measured in metrics.items()
    ]).to_csv(directory / METRICS_FILENAME, index=False)

    results = {
        "protocol": context["protocol"],
        "checkpoint": dict(preflight.get("checkpoint") or {}),
        # Deliberately NOT filed under `geometry` as "invariants": this is `apply_adapted`'s
        # record of which tensors were copied into the bundle -- names, count, selected epoch,
        # source digest -- and not a measurement of the section 5.1 invariance of `mu_prior`, the
        # two log-variances and the attention weights. Those are checked by
        # `model.assert_invariants` in the contract tests; no stage of a run measures them, and a
        # key named for them here would imply one did.
        "adaptation_applied": applied,
        "exposure": dict(preflight.get("exposure") or {}),
        "statistics": dict(preflight.get("statistics") or {}),
        "grouping": dict(preflight.get("grouping") or {}),
        "dataset_build_mode": preflight.get("dataset_build_mode"),
        "cohort": {
            "coverage": pd.read_csv(directory / data.COVERAGE_FILENAME).to_dict(
                orient="records"
            ),
            "exclusions": data.exclusion_counts(recordings),
            "coverage_contrast": evaluate.coverage_contrast(
                recordings[recordings[data.SPLIT_COLUMN].astype(str) == "test"]
            ).to_dict(orient="records"),
        },
        "selection": {
            "baseline": dict(baseline.record),
            "adaptation": dict(adapted_checkpoint.record),
        },
        "preservation": {
            "gate": dict(gate.record),
            "nll": [
                {
                    "model": version,
                    "nll_full": record.get("nll_full"),
                    "nll_base": record.get("nll_base"),
                    "mse_full": record.get("mse_full"),
                    "source_conditioned_kl": record.get("source_conditioned_kl"),
                    "mc_draws": record.get("mc_draws"),
                }
                for version, record in readings.items()
            ],
            "convergence": convergence,
        },
        "metrics": {
            "models": metrics,
            "bootstrap": bootstrap,
            "nearest_centroid": centroid_metrics,
        },
        "geometry": geometry,
        "controls": {
            "metrics": {
                name: metrics[name]
                for name in (pilot_train.CONTROL_NAME, pilot_train.PRIOR_PROBE_NAME)
                if name in metrics
            },
            "disclosure": evaluate.control_disclosure(
                n_control_fits=1, prior_probe=probe is not None
            ),
            # Read off the control's own saved fit rather than off this process's memory, so a
            # report assembled in a later run still names the draw that produced it.
            "permutation": dict(control.record.get("permutation") or {}),
        },
        "temporal": temporal,
        "subgroups": subgroups.to_dict(orient="records"),
        "projection": dict(projection.record),
        "figures": {},
        "reproduction": pilot_report.reproduction_commands(context["protocol"]),
    }
    (directory / RESULTS_FILENAME).write_text(
        json.dumps(json_safe(results), indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        f"evaluate: {len(guids)} held-out recording(s); "
        + ", ".join(f"{name} AUROC {measured['auroc']:.4f}" for name, measured in metrics.items())
    )
    return results


def stage_report(context: Dict[str, Any]) -> Dict[str, Any]:
    """Render the three figures and the measured report from what the evaluation stage wrote.

    It refits nothing and re-reads nothing from the shards, which is what makes it re-runnable in a
    finished directory on a machine where the data is no longer mounted.

    Args:
        context: The run context.

    Returns:
        The paths written.
    """
    import json

    import numpy as np
    import pandas as pd

    from teb_vae.lag_attn.eval import labels as eval_labels
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze
    from teb_vae.lag_attn_transformer_cfs.latent_pilot import report as pilot_report

    settings, directory = context["settings"], Path(context["run_dir"])
    results_path = directory / RESULTS_FILENAME
    if not results_path.is_file():
        raise pilot_config.RunStateError(
            f"{results_path} is missing, so there is nothing measured to report. Run the "
            f"evaluation stage first; this stage renders what that one established and never "
            f"fits a model to fill a gap."
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))

    pilot_report.configure_figures(settings["figure_format"])
    projection = analyze.load_projection(directory)
    test_frame = pd.read_parquet(directory / TEST_TABLE_FILENAME)
    scored_bins = pd.read_parquet(directory / BIN_TABLE_FILENAME)
    bands = pd.read_parquet(directory / BAND_TABLE_FILENAME)
    with np.load(directory / BAG_VALUES_FILENAME) as stored:
        # Pretrained first, always: the before/after arrows run from the first panel to the last,
        # and an archive whose order changed would reverse them without changing a number.
        bag_values = {
            version: stored[version] for version in (PRETRAINED, ADAPTED) if version in stored.files
        }

    figures = {
        pilot_report.FIGURE_LATENT_SPACE: str(pilot_report.figure_latent_space(
            {
                f"{version} (test)": (test_frame, values)
                for version, values in bag_values.items()
            },
            projection, directory,
            seed=int(settings["seed"]),
        )),
        pilot_report.FIGURE_SUPERVISED_AXIS: str(pilot_report.figure_supervised_axis(
            test_frame,
            {
                column[len("logit_"):]: test_frame[column].to_numpy()
                for column in test_frame.columns if str(column).startswith("logit_")
            },
            directory,
            metrics=dict(results.get("metrics", {}).get("models") or {}),
            thresholds={
                name: float(measured.get("threshold"))
                for name, measured in (
                    results.get("metrics", {}).get("models") or {}
                ).items()
                if measured.get("threshold") is not None
            },
        )),
        pilot_report.FIGURE_COVERAGE_SPACE: str(pilot_report.figure_coverage_space(
            test_frame, bag_values[PRETRAINED], projection, directory,
        )),
        pilot_report.FIGURE_TRAJECTORIES: str(pilot_report.figure_trajectories(
            {
                version: (
                    bands[
                        (bands["model"] == version)
                        & (bands["grouped_by"] == eval_labels.CLASS_COLUMN)
                    ],
                    scored_bins[scored_bins["model"] == version],
                )
                for version in (PRETRAINED, ADAPTED)
                if version in set(str(value) for value in scored_bins["model"])
            },
            directory,
            bin_hours=float(settings["windows"]["bin_hours"]),
            preservation_hours=float(settings["windows"]["preservation_hours"]),
            supervised_hours=float(settings["windows"]["supervised_hours"]),
            seed=int(settings["seed"]),
        )),
    }
    results["figures"] = figures
    written = pilot_report.write_report(results, directory)
    logger.info(f"report: {written}")
    return {"report": str(written), "figures": figures}


#: Stage name -> the function that runs it. A dictionary rather than a chain of ``if`` branches, so
#: that the runner, the ``all`` sequence and the smoke dispatcher all read one registry; a stage
#: absent from it refuses by name rather than doing nothing. A literal rather than a registration
#: call, so that importing this module still does no work at all.
STAGE_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "tests": stage_tests,
    "smoke": stage_smoke,
    "preflight": stage_preflight,
    "extract": stage_extract,
    "baseline": stage_baseline,
    "finetune": stage_finetune,
    "control": stage_control,
    "evaluate": stage_evaluate,
    "report": stage_report,
}


def run_pipeline(
    config_path: Any,
    *,
    device: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    stages: Sequence[str] = tuple(PIPELINE_STAGES),
) -> Dict[str, Any]:
    """Run the production stages of one configuration, in order, in one run directory.

    The first stage opens a new run; every stage after it resumes that directory, which is exactly
    how an operator runs the pipeline in pieces. It never runs ``tests`` or ``smoke``, so the smoke
    stage can dispatch it without recursing into itself.

    Args:
        config_path: The configuration to run.
        device: Device override, or ``None`` for the configuration's own.
        overrides: Settings overrides, or ``None``.
        stages: The stages to run, in order.

    Returns:
        The run directory and each stage's result.
    """
    ordered = list(stages)
    logger.info(f"pipeline stage 1/{len(ordered)}: {ordered[0]}")
    context = main(
        config_path=config_path, stage=ordered[0], device=device, overrides=overrides
    )
    directory = context["run_dir"]
    results = dict(context["results"])
    for position, stage in enumerate(ordered[1:], start=2):
        logger.info(f"pipeline stage {position}/{len(ordered)}: {stage}")
        finished = main(
            config_path=config_path, stage=stage, device=device, overrides=overrides,
            run_dir=directory, resume=True,
        )
        results.update(finished["results"])
    return {"run_dir": directory, "results": results}


def main(
    *,
    config_path: Any = DEFAULT_CONFIG_PATH,
    stage: str = pilot_config.ALL_STAGE,
    device: Optional[str] = None,
    run_dir: Optional[Any] = None,
    resume: bool = False,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve settings and run the requested stage. Never parses ``sys.argv``.

    Args:
        config_path: The pilot YAML, relative to the repository root or absolute.
        stage: One stage, or ``'all'`` for the whole ordered sequence.
        device: Torch device, overriding the config. ``None`` leaves the configured value, which
            may itself be ``None`` for "choose at runtime".
        run_dir: An existing run directory to resume or re-report from. ``None`` names a new one.
        resume: Continue ``run_dir`` rather than starting fresh.
        overrides: Pilot settings overriding the YAML, in the same nested shape.

    Returns:
        The run context: the resolved settings, the stage plan, the run directory, the persisted
        stage state, the frozen protocol record, and each stage's result under ``results``.

    Raises:
        PilotConfigError: If a setting or run argument is unknown, ill-typed, out of range, or
            jointly incoherent with another; if the run directory encloses one of the run's own
            inputs; or if
            a requested stage's inputs are unset or missing.
        RunStateError: If the run directory disagrees with the request -- a resume under changed
            settings, or a finished fitting stage asked to run again in place.
        FileNotFoundError: If the config file, or a resumed run directory, is not there.
        NotImplementedError: If a requested stage has no registered handler.
    """
    arguments = _validate_run_args(
        {
            "config_path": config_path,
            "stage": stage,
            "device": device,
            "run_dir": run_dir,
            "resume": resume,
            "overrides": overrides,
        },
        source="main()",
    )

    settings = pilot_config.resolve_settings(
        arguments["config_path"],
        overrides=arguments["overrides"],
        device=arguments["device"],
    )
    stages = pilot_config.stage_plan(arguments["stage"])
    # Before any directory is created and before any model is built: an unset checkpoint should
    # cost a message, not a first pass over the shards.
    pilot_config.require_inputs(settings, stages)

    # The run directory, the protocol and the resume semantics in one place: creating a fresh run,
    # continuing one, and re-entering a finished one to re-report are three different things, and
    # each of them has exactly one refusal.
    opened = pilot_config.open_run(
        settings,
        run_args=arguments,
        stages=stages,
        run_dir=arguments["run_dir"],
        resume=arguments["resume"],
    )
    directory = opened["run_dir"]
    context: Dict[str, Any] = {
        "settings": settings,
        "stages": list(opened["stages"]),
        "requested_stages": list(stages),
        "skipped": list(opened["skipped"]),
        "run_id": opened["run_id"],
        "run_dir": directory,
        "resume": bool(opened["resumed"]),
        "state": opened["state"],
        "protocol": opened["protocol"],
        "results": {},
    }

    logger.info(
        f"latent pilot: fold={settings['fold']} seed={settings['seed']} "
        f"stages={', '.join(context['stages']) or '(none, all complete)'} run_dir={directory}"
    )

    # Counted, because the stages are wildly uneven -- `extract` and `finetune` are most of a run
    # and the rest are seconds -- so "which of how many" is the only cheap answer to where a run
    # has got to. The per-stage bars inside answer the rest.
    #
    # Only when this call was handed more than one stage, which is the ``--stage all`` shape.
    # ``run_pipeline`` calls this function once per stage and counts the sequence itself, and a
    # "1/1" printed seven times underneath that would say the opposite of the truth.
    total_stages = len(context["stages"])
    for position, name in enumerate(context["stages"], start=1):
        counted = f"stage {position}/{total_stages}" if total_stages > 1 else f"stage {name}"
        handler = STAGE_HANDLERS.get(name)
        if handler is None:
            raise NotImplementedError(
                f"stage {name!r} has no registered handler. Registered stages: "
                f"{', '.join(sorted(STAGE_HANDLERS)) or '(none)'}."
            )
        logger.info(f"{counted}: {name} starting" if total_stages > 1 else f"{counted} starting")
        started = time.perf_counter()
        # Each stage reads what it needs from the context and writes its own result back, so the
        # sequence has one shared record rather than a chain of positional hand-offs. The state is
        # persisted as each one finishes: a run interrupted after the fit must not look, to the
        # stage that resumes it, like a run that never fitted anything.
        try:
            context["results"][name] = handler(context)
        except BaseException as error:  # noqa: BLE001 - recorded, then re-raised unchanged
            pilot_config.mark_failed(
                context["state"], name, directory, reason=f"{type(error).__name__}: {error}"
            )
            logger.info(f"{counted}: FAILED after {time.perf_counter() - started:.1f}s")
            raise
        pilot_config.mark_completed(context["state"], name, directory)
        logger.info(f"{counted}: finished in {time.perf_counter() - started:.1f}s")

    return context


#: Arguments used when the module is launched with no command line -- i.e. an IDE's Run button.
#:
#: Edit this and hit Run. Every key is optional; an entry left ``None`` takes the built-in default,
#: and an explicit command-line argument always wins over what is written here. A key that is not a
#: run argument raises at startup rather than being silently ignored.
#:
#: The machine-specific values -- the checkpoint, the statistics file and the three shard lists --
#: belong either in the YAML named by ``config_path`` or in the ``overrides`` block below. Putting
#: them here rather than in the YAML is supported and equivalent; both are recorded in the run's
#: protocol, so neither hides what the run read.
RUN_ARGS: Dict[str, Any] = {
    # The pilot YAML. Relative to the repository root, so this works from any working directory.
    "config_path": DEFAULT_CONFIG_PATH,
    # One of: tests, smoke, preflight, extract, baseline, finetune, control, evaluate, report, all.
    # Start with "tests", then "smoke", then "preflight" on real data, then "all".
    "stage": "all",
    # None uses the config's device; e.g. "cuda:0" or "cpu".
    "device": None,
    # None creates a new run under the configured run root. Name an existing directory to resume it
    # or to regenerate its report from saved artifacts.
    "run_dir": None,
    # Only meaningful together with run_dir.
    "resume": False,
    "overrides": {
        # Optional pilot-config overrides, in the YAML's own nested shape. Edit these or the YAML.
        # "paths": {
        #     "checkpoint": "/absolute/path/to/pretrained.ckpt",
        #     "statistics": "/absolute/path/to/matching_stats.hdf5",
        #     "train_shards": ["/absolute/path/to/fold_1/train/healthy_bg_no_cs.hdf5"],
        #     "val_shards": ["/absolute/path/to/fold_1/val/healthy_bg_no_cs.hdf5"],
        #     "test_shards": ["/absolute/path/to/fold_1/test/healthy_bg_no_cs.hdf5"],
        # },
    },
}


if __name__ == "__main__":
    # No CLI arguments: RUN_ARGS as edited above. Explicit CLI arguments override the corresponding
    # entries, key by key.
    _argv: List[str] = sys.argv[1:]
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # Repository-root-relative paths appear in the shipped configs and in RUN_ARGS, and under a
        # Run button the working directory is whatever the IDE chose. The resolver already resolves
        # against the root; moving there as well keeps every relative path a reader types on the
        # command line meaning the same thing.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    main(**resolve_run_args(RUN_ARGS, argv=_argv))
