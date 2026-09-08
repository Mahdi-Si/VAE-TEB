r"""Pilot settings: one strict schema, one resolver, one notion of where a path points.

**A separate namespace, deliberately.** Every pilot key lives under the single top-level
:data:`PILOT_KEY` block and is never merged into ``model_config.VAE_model``: the model
constructor's keyword sweep silently ignores unrelated leaves, so a pilot key that landed there
would be accepted, discarded and never reported. For the same reason no pilot setting reaches
architecture or preprocessing -- the checkpoint's own ``model_kwargs`` and the resolved config
beside it are the authority on both, and a pilot that could edit them would rebuild a model that
was never trained. There is no key here for $d_z$, the horizon, the trim, the channel widths, the
anchor stride or the forecast clock, and their absence is the mechanism: they are read from the
loaded checkpoint at runtime and cannot be contradicted from this file.

Resolution order, applied once by :func:`resolve_settings` and shared by every launch mode:

1. :data:`DEFAULTS`;
2. the ``latent_pilot`` block of the YAML named by ``config_path``;
3. the ``overrides`` mapping (from the run dictionary or ``--set``);
4. explicit top-level arguments such as ``device``.

Merging is :func:`teb_vae.lag_attn.config._deep_merge` -- the repository's one merge rule, bound
rather than restated, because the sibling evaluation packages already bind it and a second
implementation of "dicts merge, lists replace" would be free to drift from the one every other
config in this tree is resolved under. Lists replacing is not incidental here: a shard list is a
*split*, and a merge that appended would build a split nobody declared.

Nothing is mutated in place. Every entry point deep-copies before merging, so a caller's
``RUN_ARGS`` dictionary and its nested mappings are the same objects after a call as before it.

Paths resolve against the **repository root** rather than the working directory, because an IDE's
Run button chooses that directory and a relative shard path resolved against it surfaces much later
as an empty dataset. Absolute paths pass through untouched. The run destination is free -- a scratch
disk or a results share is the normal case on an execution machine -- with one refusal:
:func:`run_directory` will not write under a root that *encloses* one of the run's own inputs, so a
mistyped destination cannot write into the checkpoint's own tree.

Unknown keys are refused at every depth, and every declared tolerance in :data:`DEFAULTS` is an
engineering threshold rather than a clinically validated one -- the protocol record written at run
start says so beside the value. Production placeholders (checkpoint, statistics, shard lists) are
``None``/empty here on purpose and are demanded only by :func:`require_inputs`, and only for the
stages that actually read them: ``tests`` and ``smoke`` resolve on a checkout carrying no clinical
data at all.

Importing this module reads nothing, creates nothing and parses no arguments.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from loguru import logger

from teb_vae.lag_attn.config import _deep_merge, load_config

#: The one top-level YAML block pilot settings live under.
PILOT_KEY = "latent_pilot"

#: Repository root: this file is ``teb_vae/lag_attn_transformer_cfs/latent_pilot/config.py``, so
#: the root is four directories up. Resolved from ``__file__`` rather than the working directory,
#: which an IDE chooses and a shell does not have to agree with.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: This package's own directory. Used to locate the package's own test suite and its smoke fixture
#: cache; it does NOT constrain where runs are written -- see :func:`_check_run_root`, which allows
#: any writable destination and refuses only a root that encloses one of the run's own inputs.
PILOT_ROOT = Path(__file__).resolve().parent

#: The stages, in the order ``all`` runs them. Selection is locked between ``control`` and
#: ``evaluate``, which is what keeps the test split out of every fitting and selection decision.
STAGES: Tuple[str, ...] = (
    "tests",
    "smoke",
    "preflight",
    "extract",
    "baseline",
    "finetune",
    "control",
    "evaluate",
    "report",
)

#: The pseudo-stage that expands to all of :data:`STAGES`.
ALL_STAGE = "all"

#: Filename of the per-run stage record. Written by the runner; constructed here so the schema has
#: one definition.
STAGE_STATE_FILENAME = "stage_state.json"

#: Filename of the run's frozen protocol record.
PROTOCOL_FILENAME = "protocol.yaml"

#: Settings a stage cannot run without, as dotted paths into the resolved block.
#:
#: ``tests`` and ``smoke`` require nothing: the first touches no data at all and the second runs
#: against its own fixture configuration, whose paths are its own settings and are demanded by this
#: same table when that configuration is resolved. ``report`` requires nothing either -- it
#: regenerates presentation from a finished run directory, which is exactly the case where the
#: production shards may no longer be mounted.
_FITTING_INPUTS: Tuple[str, ...] = (
    "paths.checkpoint",
    "paths.statistics",
    "paths.train_shards",
    "paths.val_shards",
)
STAGE_INPUTS: Dict[str, Tuple[str, ...]] = {
    "tests": (),
    "smoke": (),
    "preflight": _FITTING_INPUTS,
    "extract": _FITTING_INPUTS,
    "baseline": _FITTING_INPUTS,
    "finetune": _FITTING_INPUTS,
    "control": _FITTING_INPUTS,
    "evaluate": _FITTING_INPUTS + ("paths.test_shards",),
    "report": (),
}

#: The full settings schema and its defaults. The tree is the schema: a key absent from here is
#: unknown, and a nested mapping here demands a mapping there.
#:
#: The numbers are the pilot's declared protocol, fixed before extraction and not swept in the
#: first run. Each is an engineering choice with a stated reason, and none is a clinically
#: validated threshold.
DEFAULTS: Dict[str, Any] = {
    # Declared before extraction, and part of the run directory's identity so two folds or two
    # seeds cannot land in one place.
    "fold": "fold_1",
    "seed": 42,
    # ``None`` resolves at runtime to cuda:0 when available, else cpu. A single device is enough:
    # the trainable set is two small heads.
    "device": None,
    # The format every figure of the run is written in. Any filetype the installed matplotlib can
    # write; the figure module refuses an unsupported one by name, which is where a typo belongs
    # rather than here -- validating it would mean importing a plotting backend to read a config.
    "figure_format": "pdf",
    "paths": {
        # The four production placeholders. ``None``/empty here is the shipped state -- the
        # execution machine fills them -- and :func:`require_inputs` is what turns an unfilled one
        # into a refusal, for the stages that read it and no others.
        "checkpoint": None,
        "statistics": None,
        "train_shards": [],
        "val_shards": [],
        "test_shards": [],
        # Runtime output. Free to point anywhere the machine can write -- a scratch disk or a
        # results share is the normal case on an execution machine. The one rule, enforced by
        # :func:`_check_run_root`, is that it must not enclose the run's own inputs.
        "run_root": "teb_vae/lag_attn_transformer_cfs/latent_pilot/runs",
    },
    "provenance": {
        # Optional GUID lists recording which recordings the checkpoint was pretrained on and
        # which it was selected on. Absent, exposure stays *unknown* -- which is not the same
        # statement as "disjoint", and the report is required to say so.
        "pretraining_guids": None,
        "selection_guids": None,
        # Optional GUID -> patient/delivery mapping. Absent, grouping is GUID-only and the
        # bootstrap resamples GUIDs, which the report discloses.
        "patient_map": None,
        # Free text naming the population the input statistics were fitted on, or ``None`` for
        # unknown. Recorded, never inferred.
        "statistics_population": None,
        # Free text naming the builder mode the shards were written in -- "holdout" or
        # "augmented". Section 4.1 requires it because the two partition differently: holdout
        # fixes one test set across every fold, augmented gives each fold its own partition and
        # adds extra healthy recordings to it, so what a held-out result generalizes to differs.
        # ``None`` records it as unknown, which is what it is; it is never guessed from a path.
        "dataset_build_mode": None,
    },
    "windows": {
        # The supervised bag: the final hour before delivery.
        "supervised_hours": 1.0,
        # Teacher preservation: the window the fine-tuning objective holds the latent still over,
        # and the window the forecast gate is measured on. This is a TRAINING setting.
        "preservation_hours": 3.0,
        # How far back the analysis reaches: the window latents are extracted over, the trajectory
        # bins tile, and every figure axis spans. ``None`` means "the same as preservation_hours",
        # which is what every run before this key existed did, so leaving it unset changes nothing.
        #
        # Setting it LARGER separates the two: the objective still supervises the final
        # ``supervised_hours`` and still preserves ``preservation_hours``, while the trajectories,
        # the per-bin discrimination and the figures reach further back. It may not be smaller --
        # the teacher term is computed on extracted anchors, so an analysis window inside the
        # preservation window would leave the loss without the support it is defined on.
        #
        # It is not free: the extraction is the long pass of a run and it also runs once per epoch
        # on validation, so doubling this roughly doubles the per-epoch cost of the fit.
        "analysis_hours": None,
        # Trajectory bin width; must divide the analysis window exactly, so the bins tile it with
        # no partial trailing bin drawn beside full ones.
        "bin_hours": 0.5,
        # The early window of the paired within-recording temporal comparison, in hours before
        # delivery. The late window is the supervised bag itself and is therefore not a separate
        # setting -- two keys that had to agree would eventually not.
        "early_window_hours": [2.0, 3.0],
        # Stored segment span in seconds, used to widen the coarse loader filter so a segment
        # crossing the three-hour boundary is not dropped before its eligible anchors are seen.
        # ``None`` derives it from the shards, which is the correct source; a number overrides for
        # a dataset that cannot be asked.
        "segment_span_seconds": None,
    },
    "eligibility": {
        # Late-bag eligibility, fixed before any outcome comparison and applied identically to
        # both classes. Pragmatic coverage rules, not clinical onset criteria.
        "min_late_segments": 2,
        "final_anchor_within_minutes": 30.0,
    },
    "bag": {
        # Recency half-life inside the supervised hour, in hours. A weighting preference, applied
        # identically to healthy and adverse recordings; it is not a severity target.
        "halflife_hours": 0.5,
    },
    "optim": {
        "mean_head_lr": 1.0e-4,
        "classifier_lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "grad_clip": 1.0,
        # Recordings per binary class per batch; the batch is twice this, by construction, so no
        # second key can disagree with it.
        "recordings_per_class": 4,
        "max_epochs": 10,
        "patience": 3,
        "classification_weight": 1.0,
        "preservation_weight": 0.1,
    },
    "baseline": {
        # The frozen baseline fits on cached vectors, so its budget is cheap -- but finite and
        # logged, rather than "until it stops improving" with no bound.
        "lr": 1.0e-3,
        "max_steps": 2000,
        "patience": 20,
    },
    "gates": {
        # Declared before training: a candidate that fails either is not eligible for selection.
        # Both are engineering tolerances.
        "forecast_mse_max_increase": 0.10,
        "saturation_max_increase_pp": 5.0,
        # How many validation recordings the preservation gate is measured on. Drawn once, before
        # any candidate is fitted, round-robin across adverse / healthy-BG / healthy-no-BG so both
        # classes are present; every candidate is then measured on exactly those recordings. A
        # subset rather than the whole split because this runs once per candidate epoch, and a
        # subset re-drawn per candidate would report two populations as two models.
        "subset_recordings": 24,
    },
    "bootstrap": {
        # Outcome-stratified resamples of GUIDs (or patients, where a mapping exists).
        "resamples": 1000,
    },
    "mc": {
        # Common random draws for the matched-policy NLL comparison.
        "draws": 8,
        # Optional convergence check at a larger draw count. ``None`` skips it.
        "large_draws": None,
    },
    # The frozen ``mu_prior`` probe. Required before any claim that the combined branch helps, so
    # switching it off also switches off that claim -- which the report states rather than omits.
    "prior_probe": True,
}

#: Value type for settings whose default is ``None`` and therefore carries no type of its own.
_NULLABLE_TYPES: Dict[str, Any] = {
    "device": str,
    "paths.checkpoint": str,
    "paths.statistics": str,
    "provenance.pretraining_guids": str,
    "provenance.selection_guids": str,
    "provenance.patient_map": str,
    "provenance.statistics_population": str,
    "windows.analysis_hours": (int, float),
    "windows.segment_span_seconds": (int, float),
    "mc.large_draws": int,
}

#: Element type for the list-valued settings. A list default carries no element type, and a shard
#: list holding a nested list is a mistake worth naming rather than discovering in the loader.
_LIST_ITEM_TYPES: Dict[str, Any] = {
    "paths.train_shards": str,
    "paths.val_shards": str,
    "paths.test_shards": str,
    "windows.early_window_hours": (int, float),
}

#: Settings that must be strictly greater than zero. A zero learning rate, window or weight is not
#: a configuration of this experiment; it is a run that silently measures nothing.
_POSITIVE: frozenset = frozenset({
    "windows.supervised_hours",
    "windows.preservation_hours",
    "windows.analysis_hours",
    "windows.bin_hours",
    "windows.segment_span_seconds",
    "eligibility.final_anchor_within_minutes",
    "bag.halflife_hours",
    "optim.mean_head_lr",
    "optim.classifier_lr",
    "optim.grad_clip",
    "optim.classification_weight",
    "baseline.lr",
})

#: Settings that may be zero but not negative.
_NON_NEGATIVE: frozenset = frozenset({
    "seed",
    "optim.weight_decay",
    "optim.preservation_weight",
    "gates.forecast_mse_max_increase",
    "gates.saturation_max_increase_pp",
})

#: Integer settings with a floor, and why the floor is where it is.
_MINIMUM: Dict[str, int] = {
    # One recording per class is a batch that cannot balance anything.
    "optim.recordings_per_class": 2,
    "optim.max_epochs": 1,
    "optim.patience": 1,
    "baseline.max_steps": 1,
    "baseline.patience": 1,
    "eligibility.min_late_segments": 1,
    # One recording cannot carry both classes, and a gate decided on a single recording is a gate
    # decided on whichever recording the seed happened to draw.
    "gates.subset_recordings": 2,
    "mc.draws": 1,
    "mc.large_draws": 1,
    # Below a hundred draws the percentile bounds are decided by two order statistics of a tiny
    # sample; printed beside a headline number they would read as an uncertainty statement that is
    # not one.
    "bootstrap.resamples": 100,
}

#: Settings with a ceiling.
_MAXIMUM: Dict[str, float] = {
    # numpy's seeding bound is the binding one of the three seeding APIs used.
    "seed": 2**32 - 1,
    # A saturation tolerance above a hundred percentage points can never fail.
    "gates.saturation_max_increase_pp": 100.0,
}

#: Dotted paths whose values are filesystem paths and are resolved against the repository root.
_PATH_SETTINGS: Tuple[str, ...] = (
    "paths.checkpoint",
    "paths.statistics",
    "paths.run_root",
    "provenance.pretraining_guids",
    "provenance.selection_guids",
    "provenance.patient_map",
)

#: Dotted paths holding lists of filesystem paths.
_PATH_LIST_SETTINGS: Tuple[str, ...] = (
    "paths.train_shards",
    "paths.val_shards",
    "paths.test_shards",
)


class PilotConfigError(ValueError):
    """A pilot setting, run argument or path is missing, unknown, or out of range.

    Distinct from a failure during a stage: nothing has run, nothing has been written, and the
    message names the setting and where its value came from.
    """


# =============================================================================
# Paths
# =============================================================================
def resolve_path(value: Any, *, root: Optional[Path] = None) -> Path:
    """Resolve one configured path against the repository root.

    Args:
        value: The path as configured. Absolute paths pass through untouched.
        root: The base for a relative path. Defaults to :data:`REPO_ROOT`.

    Returns:
        An absolute, normalised path. The target need not exist: existence is
        :func:`require_inputs`' question, and answering it here would refuse a ``report`` run whose
        shards have since been unmounted.

    Raises:
        PilotConfigError: If ``value`` is not a string or path.
    """
    if isinstance(value, Path):
        candidate = value
    elif isinstance(value, str):
        candidate = Path(value)
    else:
        raise PilotConfigError(
            f"expected a filesystem path, got {type(value).__name__} ({value!r})."
        )
    base = REPO_ROOT if root is None else Path(root)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _contained(path: Path, container: Path) -> bool:
    """Whether ``path`` lies inside ``container``."""
    return path == container or container in path.parents


# =============================================================================
# Schema validation
# =============================================================================
def _dotted(prefix: str, key: str) -> str:
    """Join a dotted settings path."""
    return f"{prefix}.{key}" if prefix else str(key)


def _check_leaf(path: str, value: Any, default: Any, source: str) -> Any:
    """Type- and range-check one leaf, returning the value to store.

    Args:
        path: The dotted settings path, used in every message.
        value: The configured value.
        default: The schema default at that path, whose type defines the expected one unless the
            default is ``None`` or a list.
        source: Where the value came from, for the message.

    Returns:
        The value, with an integer widened to ``float`` where the schema declares a float.

    Raises:
        PilotConfigError: On a wrong type, a value outside its declared range, or a malformed list.
    """
    if value is None:
        if default is None:
            return None
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) may not be null; it defaults to {default!r}. "
            f"Null is a value here, not an omission -- omit the key to take the default."
        )

    if isinstance(default, list):
        if not isinstance(value, list):
            raise PilotConfigError(
                f"{PILOT_KEY}.{path} (from {source}) must be a list, got "
                f"{type(value).__name__}. Lists replace rather than extend, so a scalar here "
                f"would silently become a one-element split."
            )
        item_type = _LIST_ITEM_TYPES.get(path, (str, int, float))
        bad = [item for item in value if not isinstance(item, item_type)]
        if bad:
            raise PilotConfigError(
                f"{PILOT_KEY}.{path} (from {source}) holds {bad[0]!r} of type "
                f"{type(bad[0]).__name__}; every entry must be "
                f"{getattr(item_type, '__name__', item_type)}."
            )
        return list(value)

    expected = _NULLABLE_TYPES.get(path) if default is None else type(default)
    if expected is None:
        expected = object
    # bool is a subclass of int, so an unguarded isinstance would admit ``true`` wherever a count
    # is expected -- and ``True`` would then be used as 1 with nothing raising.
    if expected in (int, float, (int, float)) and isinstance(value, bool):
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) must be a number, got the boolean {value!r}."
        )
    if expected is float:
        expected = (int, float)
    if expected is not object and not isinstance(value, expected):
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) must be "
            f"{getattr(expected, '__name__', expected)}, got {type(value).__name__} ({value!r})."
        )
    if isinstance(default, float) and isinstance(value, int):
        value = float(value)

    if path in _POSITIVE and not value > 0:
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) must be greater than zero, got {value!r}. "
            f"Zero here does not disable the setting; it makes the run measure nothing."
        )
    if path in _NON_NEGATIVE and value < 0:
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) must not be negative, got {value!r}."
        )
    if path in _MINIMUM and value < _MINIMUM[path]:
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) must be at least {_MINIMUM[path]}, got {value!r}."
        )
    if path in _MAXIMUM and value > _MAXIMUM[path]:
        raise PilotConfigError(
            f"{PILOT_KEY}.{path} (from {source}) must be at most {_MAXIMUM[path]}, got {value!r}."
        )
    return value


def _validate_tree(
    block: Mapping[str, Any], defaults: Mapping[str, Any], *, prefix: str, source: str
) -> Dict[str, Any]:
    """Validate one level of the settings tree against the schema and recurse.

    Args:
        block: The merged settings at this level.
        defaults: The schema at this level.
        prefix: The dotted path of this level, empty at the root.
        source: Where the settings came from, for every message.

    Returns:
        The validated level.

    Raises:
        PilotConfigError: On an unknown key, a block/leaf mismatch, or any leaf's own refusal.
    """
    unknown = sorted(set(block) - set(defaults))
    if unknown:
        known = ", ".join(sorted(defaults))
        where = f"{PILOT_KEY}.{prefix}" if prefix else PILOT_KEY
        raise PilotConfigError(
            f"unknown setting(s) under {where} (from {source}): "
            f"{', '.join(repr(key) for key in unknown)}. Known keys here: {known}. Nothing reads "
            f"an unrecognised key, so a misspelling would silently disable what it was meant to set."
        )

    resolved: Dict[str, Any] = {}
    for key, default in defaults.items():
        path = _dotted(prefix, key)
        value = block.get(key, default)
        if isinstance(default, dict):
            if not isinstance(value, Mapping):
                raise PilotConfigError(
                    f"{PILOT_KEY}.{path} (from {source}) must be a mapping of settings, got "
                    f"{type(value).__name__}."
                )
            resolved[key] = _validate_tree(value, default, prefix=path, source=source)
        else:
            resolved[key] = _check_leaf(path, value, default, source)
    return resolved


def analysis_hours(settings: Mapping[str, Any]) -> float:
    """How far back this run analyses, in hours before delivery.

    The fallback lives here and nowhere else: ``windows.analysis_hours`` is ``None`` by default and
    means "the same as ``windows.preservation_hours``", and a call site repeating that ``or`` would
    eventually be the one call site that forgot it.

    Args:
        settings: The resolved settings.

    Returns:
        ``windows.analysis_hours`` when set, otherwise ``windows.preservation_hours``.
    """
    windows = settings["windows"]
    configured = windows.get("analysis_hours")
    if configured is None:
        return float(windows["preservation_hours"])
    return float(configured)


def _cross_check(settings: Mapping[str, Any], source: str) -> None:
    """Refuse combinations that are individually valid and jointly incoherent.

    Args:
        settings: The validated settings.
        source: Where they came from, for every message.

    Raises:
        PilotConfigError: If the windows do not nest, the bins do not divide the preservation
            window, the early window falls outside it, the final-anchor rule falls outside the
            supervised hour, the larger draw count is smaller than the ordinary one, or a shard
            appears in more than one split.
    """
    windows = settings["windows"]
    supervised, preservation = windows["supervised_hours"], windows["preservation_hours"]
    if supervised > preservation:
        raise PilotConfigError(
            f"{PILOT_KEY}.windows.supervised_hours ({supervised}) exceeds preservation_hours "
            f"({preservation}) (from {source}): the supervised bag must sit inside the window the "
            f"teacher preserves, or the classification loss would move anchors nothing anchors."
        )
    analysis = analysis_hours(settings)
    if analysis < preservation:
        raise PilotConfigError(
            f"{PILOT_KEY}.windows.analysis_hours ({analysis}) is inside preservation_hours "
            f"({preservation}) (from {source}): the teacher term is computed on the anchors the "
            f"extraction kept, so a narrower analysis window would leave the preservation loss "
            f"without the support it is defined on. Widen it, or lower preservation_hours."
        )
    bins = analysis / windows["bin_hours"]
    if abs(bins - round(bins)) > 1e-9:
        raise PilotConfigError(
            f"{PILOT_KEY}.windows.bin_hours ({windows['bin_hours']}) does not divide the analysis "
            f"window ({analysis}) a whole number of times (from {source}); a partial trailing bin "
            f"would be drawn beside full ones and read as a real difference."
        )

    early = windows["early_window_hours"]
    if len(early) != 2 or not early[0] < early[1]:
        raise PilotConfigError(
            f"{PILOT_KEY}.windows.early_window_hours (from {source}) must be an increasing pair "
            f"[low, high] in hours before delivery, got {early!r}."
        )
    if early[0] < 0 or early[1] > analysis:
        raise PilotConfigError(
            f"{PILOT_KEY}.windows.early_window_hours {early!r} (from {source}) falls outside the "
            f"analysis window (0, {analysis}] h, where no latents are extracted."
        )
    if early[0] < supervised:
        raise PilotConfigError(
            f"{PILOT_KEY}.windows.early_window_hours {early!r} (from {source}) overlaps the "
            f"supervised window (0, {supervised}] h; the paired temporal comparison would then "
            f"contrast the supervised bag against part of itself."
        )

    limit = settings["eligibility"]["final_anchor_within_minutes"]
    if limit > supervised * 60.0:
        raise PilotConfigError(
            f"{PILOT_KEY}.eligibility.final_anchor_within_minutes ({limit}) exceeds the supervised "
            f"window of {supervised * 60.0:g} minutes (from {source}), so the rule would admit "
            f"every eligible recording and select nothing."
        )

    draws, large = settings["mc"]["draws"], settings["mc"]["large_draws"]
    if large is not None and large < draws:
        raise PilotConfigError(
            f"{PILOT_KEY}.mc.large_draws ({large}) is below mc.draws ({draws}) (from {source}); "
            f"the convergence check exists to add draws, not remove them."
        )

    seen: Dict[str, str] = {}
    for name in _PATH_LIST_SETTINGS:
        split = name.split(".")[-1]
        for shard in _dig(settings, name):
            if shard in seen:
                raise PilotConfigError(
                    f"shard {shard!r} appears in both {PILOT_KEY}.paths.{seen[shard]} and "
                    f"{PILOT_KEY}.paths.{split} (from {source}). The splits are fixed and must be "
                    f"disjoint; a shard on both sides would fit and evaluate on the same segments."
                )
            seen[shard] = split


def _dig(settings: Mapping[str, Any], path: str) -> Any:
    """Read one dotted settings path.

    Args:
        settings: The resolved settings.
        path: The dotted path.

    Returns:
        The value at that path.

    Raises:
        KeyError: If the path does not exist, which is a programming error rather than a
            configuration one -- every path used here is a literal from this module.
    """
    node: Any = settings
    for part in path.split("."):
        node = node[part]
    return node


# =============================================================================
# Resolution
# =============================================================================
def resolve_settings(
    config_path: Any,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve, validate and path-normalise the pilot settings.

    Args:
        config_path: The pilot YAML. Relative paths resolve against the repository root.
        overrides: Settings overriding the file, in the same nested shape. Deep-merged, so a
            partial block edits one leaf without dropping its siblings.
        device: An explicit device, overriding both the file and ``overrides``. ``None`` leaves
            the configured value, which itself may be ``None`` for "choose at runtime".

    Returns:
        The validated settings: every declared key present, every filesystem path absolute, and
        nothing shared with the caller's inputs.

    Raises:
        PilotConfigError: If the file carries no ``latent_pilot`` block, carries anything else at
            its top level, or any setting is unknown, ill-typed, out of range or jointly
            incoherent with another.
        FileNotFoundError: If the config file does not exist.
    """
    path = resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pilot config {str(path)!r} does not exist. Relative paths resolve against the "
            f"repository root ({REPO_ROOT}), never the working directory."
        )
    # The repository's one config reader, so the ``base:`` chain behaves here exactly as it does
    # for every other config in this tree.
    document = load_config(str(path))

    extra = sorted(set(document) - {PILOT_KEY})
    if extra:
        raise PilotConfigError(
            f"pilot config {str(path)!r} carries top-level key(s) "
            f"{', '.join(repr(key) for key in extra)}. Everything this pilot configures lives "
            f"under {PILOT_KEY!r}: a model or dataset block here would look like it reconfigured "
            f"the checkpoint, which no pilot setting is allowed to do."
        )
    block = document.get(PILOT_KEY)
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise PilotConfigError(
            f"{PILOT_KEY} in {str(path)!r} must be a mapping, got {type(block).__name__}."
        )

    merged = _deep_merge(DEFAULTS, dict(block))
    source = str(path)
    if overrides:
        if not isinstance(overrides, Mapping):
            raise PilotConfigError(
                f"overrides must be a mapping of settings, got {type(overrides).__name__}."
            )
        merged = _deep_merge(merged, deepcopy(dict(overrides)))
        source = f"{path} + overrides"
    if device is not None:
        merged["device"] = device

    settings = _validate_tree(merged, DEFAULTS, prefix="", source=source)

    for name in _PATH_SETTINGS:
        value = _dig(settings, name)
        if value is not None:
            _assign(settings, name, str(resolve_path(value)))
    for name in _PATH_LIST_SETTINGS:
        _assign(settings, name, [str(resolve_path(item)) for item in _dig(settings, name)])

    _cross_check(settings, source)

    _check_run_root(Path(_dig(settings, "paths.run_root")), settings)
    # Not a setting -- it is absent from the schema and would be refused as unknown on the way
    # in. It rides out with the resolved block so the protocol record and the settings digest both
    # say which file this run was configured from, which a directory name does not.
    settings["config_path"] = str(path)
    return settings


def _check_run_root(root: Path, settings: Mapping[str, Any]) -> None:
    """Refuse a run destination that would write over the run's own inputs.

    The destination itself is unconstrained: runs may be written to a scratch disk, a results
    share, or anywhere else the machine can write. What is refused is a root that *encloses* an
    input this run reads -- the checkpoint, the statistics file or a shard -- because a run
    creates, and on resume rewrites, directories underneath it, and the checkpoint is opened
    read-only precisely so that a pilot run cannot damage the model it is adapting.

    Args:
        root: The resolved, absolute run root (or an explicit run directory).
        settings: The resolved settings, read for the input paths.

    Raises:
        PilotConfigError: If any configured input lies inside ``root``. The message names the
            input, not only the root, because that is the half the operator has to move.
    """
    inputs: List[Tuple[str, str]] = []
    for name in ("paths.checkpoint", "paths.statistics"):
        value = _dig(settings, name)
        if value is not None:
            inputs.append((name, value))
    for name in _PATH_LIST_SETTINGS:
        for item in _dig(settings, name):
            inputs.append((name, item))
    for name, value in inputs:
        if _contained(Path(value), root):
            raise PilotConfigError(
                f"{PILOT_KEY}.paths.run_root resolves to {str(root)!r}, which contains the "
                f"input {PILOT_KEY}.{name} ({value!r}). A run writes and, on resume, rewrites "
                f"directories under its root; it must not be pointed at the tree holding the "
                f"checkpoint, the statistics file or the shards."
            )


def _assign(settings: Dict[str, Any], path: str, value: Any) -> None:
    """Write one dotted settings path in place."""
    parts = path.split(".")
    node: Any = settings
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def missing_inputs(settings: Mapping[str, Any], stages: Sequence[str]) -> List[str]:
    """Which of a stage set's inputs are unset or absent, described one per line.

    Split out from :func:`require_inputs` so a caller that can *fix* the absence -- the smoke
    stage, whose fixtures it writes itself -- can ask the question without catching the refusal.

    Args:
        settings: The resolved settings.
        stages: The stages about to run.

    Returns:
        One line per problem, each naming the dotted setting. Empty when every input is present.
    """
    needed: List[str] = []
    for stage in stages:
        for name in STAGE_INPUTS.get(stage, ()):
            if name not in needed:
                needed.append(name)

    problems: List[str] = []
    for name in needed:
        value = _dig(settings, name)
        if value is None or (isinstance(value, list) and not value):
            problems.append(
                f"  {PILOT_KEY}.{name} is unset. Fill it in the pilot YAML or in the run "
                f"dictionary's 'overrides' block."
            )
            continue
        for item in value if isinstance(value, list) else [value]:
            if not Path(item).exists():
                problems.append(f"  {PILOT_KEY}.{name} names {item!r}, which does not exist.")
    return problems


def require_inputs(settings: Mapping[str, Any], stages: Sequence[str]) -> None:
    """Refuse before anything runs if a requested stage's inputs are missing.

    Every missing setting is collected and reported together: an operator filling in four paths
    should learn that in one message rather than in four runs.

    Args:
        settings: The resolved settings.
        stages: The stages about to run.

    Raises:
        PilotConfigError: If a stage needs a setting that is unset, empty, or names a file that is
            not there. The message names the dotted setting, never only the file.
    """
    problems = missing_inputs(settings, stages)
    if problems:
        raise PilotConfigError(
            "the requested stage(s) "
            f"{', '.join(repr(stage) for stage in stages)} cannot run:\n"
            + "\n".join(problems)
            + "\n'tests' needs none of these, and 'smoke' writes its own non-clinical fixtures, "
            "so both run on a checkout carrying no clinical data at all."
        )


# =============================================================================
# Stages
# =============================================================================
def stage_plan(stage: Any) -> Tuple[str, ...]:
    """Expand a requested stage into the stages that will run, in run order.

    Args:
        stage: One of :data:`STAGES`, or ``'all'``.

    Returns:
        The ordered stages. ``'all'`` expands to every one of them, tests and smoke first and the
        report last; any other name is itself.

    Raises:
        PilotConfigError: If the name is not a stage.
    """
    if stage == ALL_STAGE:
        return STAGES
    if stage not in STAGES:
        raise PilotConfigError(
            f"unknown stage {stage!r}. Choose one of: {', '.join(STAGES)}, or {ALL_STAGE!r} for "
            f"the whole ordered sequence."
        )
    return (str(stage),)


def new_stage_state(stages: Sequence[str]) -> Dict[str, Any]:
    """Build the initial stage record for a run directory.

    The record is what makes a later stage able to refuse rather than silently redo work: it says
    which stages have completed, and whether selection has been locked -- the point after which
    the test split may be read and before which it may not.

    Args:
        stages: The stages this run intends to execute, in run order.

    Returns:
        A fresh record. Nothing is written here; persistence belongs to the runner.
    """
    return {
        "version": 1,
        "planned": list(stages),
        "completed": [],
        "failed": None,
        # Flipped once models, thresholds, controls, gates, projection procedure and seeds are
        # frozen. ``evaluate`` refuses to read the test split until it is true.
        "selection_locked": False,
    }


# =============================================================================
# Run identity and protocol
# =============================================================================
def settings_digest(settings: Mapping[str, Any]) -> str:
    """Return a short, stable digest of the resolved settings.

    Args:
        settings: The resolved settings.

    Returns:
        The first sixteen hex characters of the SHA-256 of the canonical JSON form. Stable across
        processes -- keys are sorted -- so the same settings always name the same run.
    """
    canonical = json.dumps(settings, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def run_id(settings: Mapping[str, Any], *, now: Optional[datetime] = None) -> str:
    """Build a run identifier: a timestamp and the settings digest.

    The timestamp is what an operator sorts by; the digest is what tells two runs of the same
    minute apart and what makes "did I change anything?" answerable from the directory name.

    Args:
        settings: The resolved settings.
        now: The timestamp to stamp, for a deterministic test. Defaults to the current local time.

    Returns:
        The identifier, in the stamp format the surrounding package's runs already use.
    """
    stamp = (now or datetime.now()).strftime("%Y-%m-%d--[%H-%M-%S]")
    return f"{stamp}-{settings_digest(settings)[:8]}"


def run_directory(
    settings: Mapping[str, Any],
    *,
    run_dir: Optional[Any] = None,
    identifier: Optional[str] = None,
) -> Path:
    """Resolve the run directory, without creating it.

    Args:
        settings: The resolved settings.
        run_dir: An explicit directory -- a resumed or re-reported run. Relative paths resolve
            against the repository root; absolute ones are taken as given.
        identifier: The run identifier for a new run. Defaults to a fresh :func:`run_id`.

    Returns:
        ``<run_root>/<fold>/seed_<seed>/<run_id>`` for a new run, or the resolved explicit
        directory.

    Raises:
        PilotConfigError: If an explicit directory encloses one of the run's own inputs.
    """
    if run_dir is not None:
        resolved = resolve_path(run_dir)
        _check_run_root(resolved, settings)
        return resolved
    root = Path(_dig(settings, "paths.run_root"))
    return root / str(settings["fold"]) / f"seed_{int(settings['seed'])}" / (
        identifier or run_id(settings)
    )


def software_record() -> Dict[str, Any]:
    """Describe the software this run is executing, as far as it can be established.

    Returns:
        The Python version, the repository revision and whether the working tree was dirty.
        Revision and dirtiness are ``None`` when git cannot answer -- an exported tree, or no git
        on the machine -- because unknown provenance is recorded as unknown rather than as clean.
    """
    revision: Optional[str] = None
    dirty: Optional[bool] = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version.split()[0],
        "repository_root": str(REPO_ROOT),
        "revision": revision,
        "working_tree_dirty": dirty,
    }


def protocol_record(
    settings: Mapping[str, Any],
    *,
    run_args: Mapping[str, Any],
    identifier: str,
    directory: Any,
    stages: Sequence[str],
) -> Dict[str, Any]:
    """Build the record frozen at the start of a run.

    Everything a later reader needs to say what was run and under which choices, and nothing that
    is only knowable afterwards: the checkpoint's digest, the cohort counts and the selected epoch
    are added by the stages that establish them.

    Args:
        settings: The resolved settings.
        run_args: The effective run arguments, as resolved from the dictionary and command line.
        identifier: The run identifier.
        directory: The run directory.
        stages: The stages this run will execute, in run order.

    Returns:
        The protocol record, deep-copied so a later mutation of the settings cannot rewrite it.
    """
    return {
        "run_id": identifier,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_directory": str(directory),
        "fold": settings["fold"],
        "seed": settings["seed"],
        "device": settings["device"],
        "stages": list(stages),
        "settings": deepcopy(dict(settings)),
        "settings_digest": settings_digest(settings),
        "run_args": deepcopy(dict(run_args)),
        "software": software_record(),
        "tolerances_note": (
            "the gate thresholds and eligibility rules in 'settings' are declared engineering "
            "tolerances, not clinically validated criteria"
        ),
    }



# =============================================================================
# The run directory: what is on disk, what may be re-run, and what is locked
# =============================================================================
#: Filename of the record that fixes every choice before the test split is read.
SELECTION_LOCK_FILENAME = "selection_lock.json"

#: Stages that may run again in an existing run directory **without** resuming.
#:
#: Each of them either touches no artifact (``tests``, ``smoke`` runs in its own subtree) or reads
#: finished ones and rewrites presentation (``preflight``, ``report``). Every other stage produces a
#: fitted artifact that a later stage was selected against, so re-running one in place would leave a
#: directory whose report describes a model it no longer holds.
RERUNNABLE_STAGES: Tuple[str, ...] = ("tests", "smoke", "preflight", "report")


class RunStateError(PilotConfigError):
    """A run directory cannot be continued as asked.

    Distinct from a bad setting and from a stage failure: the settings are valid and nothing has
    run, but the directory on disk and the request disagree -- a finished run being written into, a
    resume under changed settings, or a stage whose input was never produced.
    """


def _flatten(settings: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a settings tree into dotted paths, for naming what differs."""
    flat: Dict[str, Any] = {}
    for key, value in settings.items():
        path = _dotted(prefix, key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def settings_differences(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> Dict[str, Tuple[Any, Any]]:
    """Which settings a resumed run would change, as dotted paths.

    **A key this schema has gained since the run was written is not a difference**, provided the
    current value is that key's own default. The stored record describes the choices its artifacts
    were made under, and a setting that did not exist then was not a choice anyone made; refusing
    on it would make every finished run directory unreadable the first time a key was added --
    including for ``report``, which rewrites presentation and fits nothing. A path present in both
    with different values, and a path added and then *set* to something other than its default,
    both still refuse: those are choices, and continuing under them would leave a directory whose
    protocol describes a run that never happened.

    Args:
        stored: The settings the run directory was created under.
        current: The settings this invocation resolved.

    Returns:
        Path -> ``(stored, current)`` for every leaf that differs. Empty when they agree.
    """
    left, right = _flatten(stored), _flatten(current)
    defaults = _flatten(DEFAULTS)
    added_at_default = {
        path for path in set(right) - set(left)
        if path in defaults and right[path] == defaults[path]
    }
    if added_at_default:
        logger.info(
            f"settings key(s) {sorted(added_at_default)} are absent from the stored protocol and "
            f"carry their default value; treated as a schema addition rather than a changed choice"
        )
    return {
        path: (left.get(path), right.get(path))
        for path in sorted(set(left) | set(right))
        if path not in added_at_default and left.get(path) != right.get(path)
    }


def write_protocol(record: Mapping[str, Any], directory: Any) -> Path:
    """Write the frozen protocol record into a run directory.

    Args:
        record: The record from :func:`protocol_record`.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / PROTOCOL_FILENAME
    target.write_text(
        yaml.safe_dump(json.loads(json.dumps(dict(record), default=str)), sort_keys=True),
        encoding="utf-8",
    )
    return target


def read_protocol(directory: Any) -> Dict[str, Any]:
    """Read a run's frozen protocol record.

    Args:
        directory: The run directory.

    Returns:
        The record.

    Raises:
        RunStateError: If the directory holds none. A directory without one is not a pilot run,
            and continuing into it would mix this run's artifacts with whatever is there.
    """
    target = Path(directory) / PROTOCOL_FILENAME
    if not target.is_file():
        raise RunStateError(
            f"{target} is missing, so {Path(directory)} is not a pilot run directory. A new run "
            f"leaves run_dir unset and gets a fresh one; naming a directory continues a run that "
            f"already recorded what it was launched with."
        )
    return dict(yaml.safe_load(target.read_text(encoding="utf-8")) or {})


def write_stage_state(state: Mapping[str, Any], directory: Any) -> Path:
    """Persist the stage record after every stage, so an interruption is legible afterwards.

    Args:
        state: The stage record.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / STAGE_STATE_FILENAME
    target.write_text(json.dumps(dict(state), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return target


def read_stage_state(directory: Any) -> Dict[str, Any]:
    """Read a run's stage record, or an empty one for a directory that has none yet.

    Args:
        directory: The run directory.

    Returns:
        The record, with the keys :func:`new_stage_state` creates.
    """
    target = Path(directory) / STAGE_STATE_FILENAME
    if not target.is_file():
        return new_stage_state(())
    state = dict(json.loads(target.read_text(encoding="utf-8")))
    state.setdefault("completed", [])
    state.setdefault("failed", None)
    state.setdefault("selection_locked", False)
    return state


def _carry_forward_flags(state: Dict[str, Any], directory: Any) -> Dict[str, Any]:
    """Refresh the flags a *different* writer may have set since this record was read.

    ``stage_state.json`` has two writers. The runner holds one record for the length of a stage and
    persists it when the stage ends; :func:`lock_selection` writes ``selection_locked`` into the
    file in the middle of the evaluation stage, from a record it read itself. Without this, the
    runner's copy -- read before the lock existed -- is written back over it, and the run directory
    ends up claiming the selection was never locked while ``selection_lock.json`` sits beside it.

    Only ``selection_locked`` is carried, and only from false to true: it is the one field this
    module sets outside the runner's own record, and a lock is never withdrawn.

    Args:
        state: The runner's record, updated in place.
        directory: The run directory.

    Returns:
        The updated record.
    """
    if not state.get("selection_locked"):
        state["selection_locked"] = bool(
            read_stage_state(directory).get("selection_locked", False)
        )
    return state


def mark_completed(state: Dict[str, Any], stage: str, directory: Any) -> Dict[str, Any]:
    """Record that a stage finished, and persist the record.

    Args:
        state: The stage record, updated in place.
        stage: The stage that finished.
        directory: The run directory.

    Returns:
        The updated record.
    """
    if stage not in state["completed"]:
        state["completed"].append(stage)
    state["failed"] = None
    write_stage_state(_carry_forward_flags(state, directory), directory)
    return state


def mark_failed(state: Dict[str, Any], stage: str, directory: Any, *, reason: str) -> Dict[str, Any]:
    """Record that a stage failed, and persist the record.

    A failed stage is written down rather than merely raised, because the next invocation's resume
    has to know where the sequence stopped -- and because a traceback in a terminal is not an
    artifact anyone can send back with a bug report.

    Args:
        state: The stage record, updated in place.
        stage: The stage that failed.
        directory: The run directory.
        reason: A short description, typically the exception's message.

    Returns:
        The updated record.
    """
    state["failed"] = {"stage": stage, "reason": str(reason)}
    write_stage_state(_carry_forward_flags(state, directory), directory)
    return state


def require_completed(state: Mapping[str, Any], stage: str, *, needed_by: str) -> None:
    """Refuse a stage whose input was never produced.

    Args:
        state: The stage record.
        stage: The prerequisite.
        needed_by: The stage that needs it, for the message.

    Raises:
        RunStateError: Naming both. The runner never silently trains a missing model on the way to
            a report: a report that quietly refitted its own subject would be describing a
            different run from the one its directory records.
    """
    if stage not in list(state.get("completed") or []):
        raise RunStateError(
            f"stage {needed_by!r} needs {stage!r}, which this run directory has not completed "
            f"(completed: {list(state.get('completed') or []) or 'none'}). Run {stage!r} first, or "
            f"resume the run that produced it. Nothing here refits a missing prerequisite on its "
            f"own."
        )


def open_run(
    settings: Mapping[str, Any],
    *,
    run_args: Mapping[str, Any],
    stages: Sequence[str],
    run_dir: Optional[Any] = None,
    resume: bool = False,
    identifier: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve, create or continue a run directory, and say which stages will actually run.

    The three cases, and the one refusal each:

    * **A new run.** No ``run_dir``: a fresh directory named by :func:`run_id`, the protocol record
      written into it, every requested stage to run.
    * **A resume.** ``run_dir`` with ``resume``: the stored protocol is read back and its settings
      digest must match this invocation's, because continuing a run under changed settings would
      produce a directory whose protocol describes choices its artifacts were not made under.
      Completed stages are skipped and reported as skipped.
    * **A re-entry without resume.** ``run_dir`` alone: only :data:`RERUNNABLE_STAGES` may run.
      They touch no fitted artifact -- ``report`` rewrites presentation from what is already there,
      which is exactly what §12 asks for -- and any other stage is refused rather than allowed to
      overwrite an artifact a later stage was selected against.

    **Resume is stage-level, deliberately.** A fit interrupted halfway restarts from its own
    beginning rather than from its last epoch: exact mid-fit resume would have to persist the
    optimizer's moments, the sampler's stream and the RNG state every epoch, and would still not
    reproduce a run bit for bit across a device change -- so it would be a promise this package
    could not keep. The fits here are at most ``optim.max_epochs`` epochs over two small heads --
    short relative to the machinery exact resume would need at the protocol's own budget, though a
    large ``max_epochs`` makes a stage-level restart correspondingly expensive; restarting one is
    cheaper than the machinery that would avoid it, and the semantics are stated rather than
    implied.

    Args:
        settings: The resolved settings.
        run_args: The effective run arguments, recorded in a new run's protocol.
        stages: The stages requested, in run order.
        run_dir: An existing directory to continue or re-report from, or ``None`` for a new run.
        resume: Continue ``run_dir`` rather than re-entering it read-mostly.
        identifier: The run identifier for a new run. Defaults to a fresh :func:`run_id`.

    Returns:
        ``{'run_dir', 'run_id', 'protocol', 'state', 'stages', 'skipped', 'resumed'}``, where
        ``stages`` is what will run and ``skipped`` what was already complete.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist.
        RunStateError: On a settings mismatch, a directory that is not a pilot run, or a fitting
            stage requested in a finished directory without resuming.
    """
    if run_dir is None:
        chosen = identifier or run_id(settings)
        directory = run_directory(settings, identifier=chosen)
        directory.mkdir(parents=True, exist_ok=True)
        record = protocol_record(
            settings, run_args=run_args, identifier=chosen, directory=directory, stages=stages
        )
        write_protocol(record, directory)
        state = new_stage_state(stages)
        write_stage_state(state, directory)
        return {
            "run_dir": directory,
            "run_id": chosen,
            "protocol": record,
            "state": state,
            "stages": list(stages),
            "skipped": [],
            "resumed": False,
        }

    directory = run_directory(settings, run_dir=run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"run_dir {str(directory)!r} does not exist. A new run leaves run_dir unset and gets a "
            f"fresh directory; naming one is for resuming or re-reporting a finished run."
        )
    record = read_protocol(directory)
    stored = dict(record.get("settings") or {})
    differences = settings_differences(stored, settings)
    if differences:
        raise RunStateError(
            f"the settings differ from the ones {directory} was created under: {differences}. A "
            f"run's protocol record is what says which choices produced its artifacts; continuing "
            f"it under different ones would leave a directory describing a run that never "
            f"happened. Start a new run, or restore these settings."
        )
    state = read_stage_state(directory)
    completed = list(state.get("completed") or [])

    if resume:
        remaining = [stage for stage in stages if stage not in completed]
        skipped = [stage for stage in stages if stage in completed]
        state["planned"] = list(stages)
        write_stage_state(state, directory)
        if skipped:
            logger.info(f"resuming {directory}: skipping completed stage(s) {', '.join(skipped)}")
        return {
            "run_dir": directory,
            "run_id": record.get("run_id", directory.name),
            "protocol": record,
            "state": state,
            "stages": remaining,
            "skipped": skipped,
            "resumed": True,
        }

    blocked = [
        stage for stage in stages
        if stage not in RERUNNABLE_STAGES and stage in completed
    ]
    if blocked:
        raise RunStateError(
            f"stage(s) {blocked} already completed in {directory}, and re-running them in place "
            f"would overwrite artifacts that later stages were selected against. Pass resume=True "
            f"to continue this run from where it stopped, start a new run for a new experiment, or "
            f"request only {list(RERUNNABLE_STAGES)}, which rewrite nothing that was fitted."
        )
    state["planned"] = list(stages)
    write_stage_state(state, directory)
    return {
        "run_dir": directory,
        "run_id": record.get("run_id", directory.name),
        "protocol": record,
        "state": state,
        "stages": list(stages),
        "skipped": [],
        "resumed": False,
    }


# =============================================================================
# The selection lock
# =============================================================================
def lock_selection(directory: Any, record: Mapping[str, Any]) -> Path:
    """Freeze every choice the held-out comparison will be read under, before it is read.

    What the lock is for: the test split answers one prespecified question, and it stops answering
    it the moment anything is chosen after seeing it. So the models, the classifier, the threshold,
    the gate tolerances and the subset they were measured on, the control, the projection procedure
    and the seeds are all written down here -- with a timestamp -- and the evaluation stage refuses
    to open the test split until this file exists.

    It is a record rather than a promise: a reader can compare it against the report afterwards and
    see that the threshold quoted there is the one chosen on validation.

    Args:
        directory: The run directory.
        record: What is being locked. Serialised as given, so a caller adds a field by passing it.

    Returns:
        The written path.

    Raises:
        RunStateError: If the run is already locked. Re-locking after a look at the test split is
            exactly what the lock exists to prevent, so it is refused rather than overwritten.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / SELECTION_LOCK_FILENAME
    if target.is_file():
        raise RunStateError(
            f"{target} already exists: this run's selection is locked and cannot be re-locked. A "
            f"second lock would be a choice made after the held-out split was available, which is "
            f"the one thing the lock exists to make impossible."
        )
    payload = {
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **dict(record),
        "note": (
            "everything the held-out comparison is read under, fixed before the test split was "
            "opened; the evaluation stage refuses to run without this record"
        ),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    state = read_stage_state(path)
    state["selection_locked"] = True
    write_stage_state(state, path)
    logger.info(f"selection locked: {target}")
    return target


def read_selection_lock(directory: Any) -> Dict[str, Any]:
    """Read a run's selection lock.

    Args:
        directory: The run directory.

    Returns:
        The locked record.

    Raises:
        RunStateError: If the run is not locked.
    """
    target = Path(directory) / SELECTION_LOCK_FILENAME
    if not target.is_file():
        raise RunStateError(
            f"{target} is missing: this run has not locked its selection, so the held-out split "
            f"may not be read. The lock is written once the baseline, the adapted checkpoint, the "
            f"threshold, the controls and the analysis settings are fixed."
        )
    return dict(json.loads(target.read_text(encoding="utf-8")))


def require_selection_locked(directory: Any) -> bool:
    """Refuse to open the test split until the run's selection is locked.

    Written to be used as the argument it guards -- ``allow_test=require_selection_locked(run_dir)``
    -- so the only ordinary way to reach the held-out extraction passes through the check.

    Args:
        directory: The run directory.

    Returns:
        ``True``, always. The value exists so the call sits where the permission is granted.

    Raises:
        RunStateError: If the run is not locked.
    """
    read_selection_lock(directory)
    return True


__all__ = [
    "ALL_STAGE",
    "DEFAULTS",
    "PILOT_KEY",
    "PILOT_ROOT",
    "PROTOCOL_FILENAME",
    "REPO_ROOT",
    "RERUNNABLE_STAGES",
    "SELECTION_LOCK_FILENAME",
    "analysis_hours",
    "STAGES",
    "STAGE_INPUTS",
    "STAGE_STATE_FILENAME",
    "PilotConfigError",
    "RunStateError",
    "lock_selection",
    "mark_completed",
    "mark_failed",
    "new_stage_state",
    "open_run",
    "protocol_record",
    "read_protocol",
    "read_selection_lock",
    "read_stage_state",
    "require_completed",
    "require_inputs",
    "require_selection_locked",
    "resolve_path",
    "resolve_settings",
    "run_directory",
    "run_id",
    "settings_differences",
    "settings_digest",
    "software_record",
    "stage_plan",
    "write_protocol",
    "write_stage_state",
]
