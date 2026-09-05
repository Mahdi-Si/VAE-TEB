r"""The evaluation's configuration surface: the override merge, the forced loader, the schema.

Everything that shapes a run before a model, a loader or an output directory exists lives here,
and it is deliberately cheap to import: stdlib, ``yaml``, ``loguru``, and the sibling's two
validators. A misconfigured run must cost a parse, not a checkpoint load and a first pass over
the shards.

**The override delta, not a second config.** A training run writes its fully resolved
configuration beside its checkpoints, and that file -- not a committed YAML -- is the record of
what the model was trained on. But an evaluation genuinely needs a handful of different values:
the held-out k-fold shards rather than the healthy-only pretraining split, five extra
``load_fields`` the clinical questions are asked in, and an ``eval_config`` block. Those arrive
as a small committed delta deep-merged **over** the run's own resolved config, so the run's
contract stays authoritative and the divergence is a reviewable file rather than a pile of
command-line flags.

A ``base:`` chain cannot do this. :func:`teb_vae.lag_attn.config.load_config` resolves ``base:``
relative to the file that names it, and the checkpoint's ``resolved_config.yaml`` path is a
runtime value, so no committed file can reference it. Pointing ``base:`` at the shipped
``default.yaml`` instead would evaluate against what a config file currently says rather than
what the run trained under -- which is the drift the whole arrangement exists to prevent, so an
overrides file carrying a ``base:`` key is refused rather than honoured.

**The ``eval_config`` block is validated, not absorbed.** YAML has no schema and a mapping
silently accepts whatever it is given: ``max_sample`` instead of ``max_samples`` parses cleanly,
is never read, and means "no cap" -- a run that took hours instead of minutes and reported
nothing about why. So an unknown key raises and names the valid set.

What is *not* validated here, deliberately: the names inside ``caps``. That set grows with every
analysis, and a table of analyses that do not exist yet would be wrong more often than right. Cap
*values* are checked; an unread cap name is inert, because a cap only ever narrows what an
analysis retains.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml
from loguru import logger

from teb_vae.lag_attn.config import BASE_KEY, _deep_merge

# The sibling's validators, imported rather than restated: ``bool`` is an ``int`` subclass in
# Python, and the two places that trap has to be caught must not be able to disagree about it.
from teb_vae.lag_attn.eval.config_schema import _require_int, _validate_caps

#: The committed override delta, merged over a checkpoint's own resolved config.
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "configs" / "eval_overrides.yaml"

#: Every key the ``eval_config`` block accepts. Anything else raises -- see the module docstring.
#:
#: Two names are deliberately **absent**. ``alpha``: an operator who could widen the significance
#: level could make a difference appear or disappear from a config file, which is not a setting.
#: ``trajectory_bin_hours``: the same argument, applied to the bin width that decides whether a
#: trend against time before delivery is visible.
VALID_KEYS = frozenset(
    {
        "seed",
        "num_mc_samples",
        "max_samples",
        "caps",
        "prior_shuffle_min_nats",
        "min_active_dims",
        "event_lag_window_s",
        "bootstrap_resamples",
        # The image format every figure of the run is written in. ``None`` -- the shipped
        # setting -- leaves ``figures.DEFAULT_FIGURE_FORMAT`` standing, which is what every
        # committed figure manifest records.
        "figure_format",
        # How far before delivery a segment may be recorded and still be evaluated. ``None`` --
        # the shipped setting -- evaluates every segment the split carries.
        "max_hours_before_delivery",
    }
)

#: Values used when a key is absent. A partial block is legitimate -- a variant overriding one cap
#: should not have to restate the rest -- but a *misspelled* key is not, which is why absence is
#: defaulted and an unknown key raises.
#:
#: ``num_mc_samples``, ``prior_shuffle_min_nats`` and ``min_active_dims`` restate the readout
#: module's own defaults rather than importing them: that module pulls in ``torch`` and the whole
#: network, and this one must stay a stdlib parse. ``tests/test_eval_config_schema.py`` pins each
#: of the three equal to its counterpart, so a change there fails a test instead of drifting.
DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "num_mc_samples": 8,
    "max_samples": None,
    "caps": {},
    "prior_shuffle_min_nats": 1.0,
    "min_active_dims": 2,
    "event_lag_window_s": 120.0,
    "bootstrap_resamples": 2000,
    # Nullable on purpose: ``None`` means "whatever ``figures.DEFAULT_FIGURE_FORMAT`` is",
    # so this file names no format of its own and the default lives in exactly one place.
    "figure_format": None,
    # Nullable for the same reason ``max_samples`` is: absence means "no bound", which is a
    # different statement from any number and is the one the shipped runs make.
    "max_hours_before_delivery": None,
}

#: Upper bound on the seed: ``numpy.random.seed`` rejects anything outside $[0, 2^{32})$, and
#: torch's own seeding is unbounded, so this is the binding constraint of the three.
_MAX_SEED = 2**32

#: Fewest bootstrap resamples that produce an interval rather than a point. A handful of draws
#: yields percentile bounds decided by two order statistics of a tiny sample -- reported beside
#: every headline number, it would read as an active uncertainty statement that is not one.
_MIN_BOOTSTRAP_RESAMPLES = 100


# =============================================================================
# The override merge
# =============================================================================
def load_eval_overrides(overrides_path: Optional[Any] = None) -> Dict[str, Any]:
    """Read the override delta.

    Args:
        overrides_path: Path to the delta. Defaults to :data:`DEFAULT_OVERRIDES_PATH`.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If the file is not there.
        ValueError: If its top level is not a mapping, or if it carries a ``base:`` key -- see
            the module docstring for why that key cannot mean what it looks like it means here.
    """
    path = Path(DEFAULT_OVERRIDES_PATH if overrides_path is None else overrides_path)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation overrides not found: {path}")

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"evaluation overrides {str(path)!r} must contain a top-level YAML mapping, got "
            f"{type(loaded).__name__}."
        )
    if BASE_KEY in loaded:
        raise ValueError(
            f"evaluation overrides {str(path)!r} carries a '{BASE_KEY}:' key. The delta is merged "
            f"over the checkpoint's own resolved config, which is the record of what was "
            f"trained; a '{BASE_KEY}:' chain would instead inherit from whatever a committed "
            f"config file currently says, which is the drift this arrangement exists to prevent."
        )
    return loaded


def merge_eval_overrides(
    config: Mapping[str, Any], overrides_path: Optional[Any] = None
) -> Dict[str, Any]:
    """Deep-merge the evaluation override delta over a run's resolved config.

    Args:
        config: The checkpoint's own resolved config.
        overrides_path: Path to the delta. Defaults to :data:`DEFAULT_OVERRIDES_PATH`.

    Returns:
        A new merged config. Neither input is mutated. Nested mappings merge key by key, so
        setting one ``dataset_kwargs`` entry does not drop the rest of the block; lists replace
        wholesale, because a list is a value here -- ``load_fields`` must *replace* the training
        set, not extend it.
    """
    return merge_eval_overrides_with_provenance(config, overrides_path)[0]


def merge_eval_overrides_with_provenance(
    config: Mapping[str, Any], overrides_path: Optional[Any] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Merge the delta and record, per key, what the run trained under and what it is evaluated
    under.

    The record is the point. Merging a delta over a run's own configuration is a divergence from
    what was trained, and a divergence that appears nowhere in the output is indistinguishable
    from an accident: a reader of ``summary.json`` cannot otherwise tell that the shard list was
    repointed at the holdout split, or that ``load_fields`` grew by five names, or -- worse --
    that something *else* was overridden too.

    Args:
        config: The checkpoint's own resolved config.
        overrides_path: Path to the delta. Defaults to :data:`DEFAULT_OVERRIDES_PATH`.

    Returns:
        ``(merged, provenance)``; see :func:`override_provenance` for the record's shape.
    """
    overrides = load_eval_overrides(overrides_path)
    merged = _deep_merge(dict(config), overrides)
    return merged, override_provenance(config, merged, overrides)


def override_provenance(
    original: Mapping[str, Any], merged: Mapping[str, Any], overrides: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Return one record per leaf the delta set, with the run's value beside the merged one.

    Walked from the *delta*, so the record has one entry per thing the evaluation deliberately
    changed rather than one per config key. The merged value is read back out of the merged
    config rather than echoed from the delta, so a merge rule that did not do what it looks like
    it does -- a list extending instead of replacing, say -- shows up here as a disagreement
    rather than being papered over.

    Args:
        original: The run's own resolved config.
        merged: The result of merging the delta over it.
        overrides: The delta itself.

    Returns:
        Records ``{'path', 'run_value', 'eval_value', 'in_run_config'}``, ordered by path.
        ``in_run_config`` distinguishes an override that *changed* a trained value from one that
        added a key the run did not carry -- ``eval_config`` being the whole-block case.
    """
    records: List[Dict[str, Any]] = []

    def _walk(delta: Mapping[str, Any], prefix: str) -> None:
        for key, value in delta.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            run_value, present = _dig(original, path)
            if isinstance(value, Mapping) and isinstance(run_value, Mapping):
                # Both sides are mappings, so the merge recursed and the leaves below are what
                # actually differ; recording the whole block here would bury them.
                _walk(value, path)
                continue
            records.append(
                {
                    "path": path,
                    "run_value": run_value,
                    "eval_value": _dig(merged, path)[0],
                    "in_run_config": present,
                }
            )

    _walk(overrides, "")
    return sorted(records, key=lambda record: record["path"])


def _dig(config: Mapping[str, Any], path: str) -> Tuple[Any, bool]:
    """Resolve a dotted path through nested mappings.

    Args:
        config: The mapping to read.
        path: A dotted key path.

    Returns:
        ``(value, present)``. ``present`` is ``False`` when any component is missing, which is
        not the same as a value of ``None`` -- a config may legitimately set a key to null.
    """
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None, False
        current = current[part]
    return current, True


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


# =============================================================================
# The eval_config block
# =============================================================================
#: Floor on ``max_hours_before_delivery``, in hours: one trajectory bin width. Below it a run has
#: no whole window to draw, so the bound would empty every clock rather than narrow it. Kept equal
#: to the binning module's own width by test rather than by import -- this module validates a run's
#: settings before anything is built and stays a stdlib parse.
_MIN_HORIZON_HOURS = 0.5


def _require_positive_float(value: Any, name: str, *, minimum: float) -> float:
    """Return ``value`` as a finite float, raising if it is not one or is below ``minimum``.

    ``bool`` is rejected for the same reason :func:`_require_int` rejects it: it is a numeric
    subclass, so a stray ``true`` would validate and then mean $1$.

    Args:
        value: The configured value.
        name: Dotted config path, used in the error message.
        minimum: Smallest acceptable value, inclusive.

    Returns:
        The value as a ``float``.

    Raises:
        ValueError: If the value is not a finite number or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"eval_config.{name} must be a number, got {value!r} ({type(value).__name__})."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"eval_config.{name} must be finite, got {value!r}.")
    if number < minimum:
        raise ValueError(f"eval_config.{name} must be >= {minimum}, got {number}.")
    return number


def _require_figure_format(value: Any) -> str:
    """Return ``value`` as a matplotlib filetype, raising if this build cannot write it.

    The supported set is read from the **installed** matplotlib rather than written out here, so
    the check cannot go stale against the backend that will actually do the writing. The import is
    function-local to keep importing this module -- which ``probe.py`` does before it builds
    anything -- free of matplotlib.

    Args:
        value: The configured value.

    Returns:
        The format without its leading dot, lowercased.

    Raises:
        ValueError: If the value is not a string, or names a format matplotlib cannot write. The
            message lists the supported set, because the realistic error here is a typo.
    """
    from teb_vae.lag_attn.eval.figures import SUPPORTED_FIGURE_FORMATS

    if not isinstance(value, str):
        raise ValueError(
            f"eval_config.figure_format must be a string, got {value!r} "
            f"({type(value).__name__})."
        )
    normalised = value.strip().lstrip(".").lower()
    if normalised not in SUPPORTED_FIGURE_FORMATS:
        raise ValueError(
            f"eval_config.figure_format must be one of {sorted(SUPPORTED_FIGURE_FORMATS)}, "
            f"got {value!r}."
        )
    return normalised


def validate_eval_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate ``config['eval_config']`` and return it with defaults filled in.

    Call this immediately after the override merge and before any model, loader or output
    directory is built.

    Args:
        config: The merged run config.

    Returns:
        The validated block. Every cap is ``None`` or a positive int.

    Raises:
        ValueError: If the block is not a mapping, carries an unknown key, or any value fails its
            type or range check.
    """
    block = config.get("eval_config")
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise ValueError(f"eval_config must be a mapping, got {type(block).__name__}.")

    unknown = sorted(set(block) - VALID_KEYS)
    if unknown:
        raise ValueError(
            f"unknown eval_config key(s): {', '.join(repr(key) for key in unknown)}. "
            f"Valid keys are: {', '.join(sorted(VALID_KEYS))}. Nothing reads an unrecognised "
            f"key, so a misspelling here would silently disable whatever it was meant to set."
        )

    resolved: Dict[str, Any] = dict(DEFAULTS)
    resolved["caps"] = dict(DEFAULTS["caps"])
    resolved.update(block)

    resolved["seed"] = _require_int(resolved["seed"], "seed", minimum=0)
    if resolved["seed"] >= _MAX_SEED:
        raise ValueError(
            f"eval_config.seed must be < {_MAX_SEED} (numpy's bound), got {resolved['seed']}."
        )

    # K = 1 is admitted: one draw of the marginalised estimator. It is NOT the training-path score
    # under base_decode: mean, where the training path decodes the base branch at the prior mean.
    resolved["num_mc_samples"] = _require_int(resolved["num_mc_samples"], "num_mc_samples", minimum=1)

    if resolved["max_samples"] is not None:
        resolved["max_samples"] = _require_int(resolved["max_samples"], "max_samples", minimum=1)

    resolved["caps"] = _validate_caps(resolved["caps"])

    # A negative margin can never fail, so it would read as an active verdict that is not one.
    resolved["prior_shuffle_min_nats"] = _require_positive_float(
        resolved["prior_shuffle_min_nats"], "prior_shuffle_min_nats", minimum=0.0
    )
    resolved["min_active_dims"] = _require_int(
        resolved["min_active_dims"], "min_active_dims", minimum=1
    )
    # Zero would restrict the event-conditioned readouts to anchors exactly on a contraction.
    resolved["event_lag_window_s"] = _require_positive_float(
        resolved["event_lag_window_s"], "event_lag_window_s", minimum=1.0
    )
    resolved["bootstrap_resamples"] = _require_int(
        resolved["bootstrap_resamples"], "bootstrap_resamples", minimum=_MIN_BOOTSTRAP_RESAMPLES
    )

    # Nullable, and validated only when set -- the ``max_samples`` shape. A run that names no
    # format keeps the default, and the manifests, which record ``.pdf`` names, stay correct.
    if resolved["figure_format"] is not None:
        resolved["figure_format"] = _require_figure_format(resolved["figure_format"])

    # Nullable, and validated only when set -- the ``max_samples`` shape. It bounds the population
    # every clock is computed over, so it is a config key and not a call-site default: a bounded
    # run's trajectory is not comparable with an unbounded one's, and the dumped config is the
    # only durable record of which this run was.
    if resolved["max_hours_before_delivery"] is not None:
        resolved["max_hours_before_delivery"] = _require_positive_float(
            resolved["max_hours_before_delivery"],
            "max_hours_before_delivery",
            minimum=_MIN_HORIZON_HOURS,
        )

    return resolved
