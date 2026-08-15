r"""Everything that must hold before a single number is computed, in the causal feature domain.

The expensive failures in this pipeline are the silent ones. A checkpoint that did not load, a
config whose geometry contradicts the one the weights were trained under, an evaluation pointed at
the healthy-only pretraining split instead of the holdout -- each of them produces a full set of
plausible numbers and no error. The guards here run before the loader is built and before any
analysis writes anything, so a rejected run costs a checkpoint load and a handful of HDF5 attribute
reads.

**A refusal is not a crash, and the type says so.** Every guard raises
:class:`EvalPreconditionUnmet`, including the three reused from the training entry point, whose own
``ValueError`` is re-raised from the original so the trainer's long actionable message survives
verbatim. The runner calls this *outside* its fail-soft step wrapper, so a rejected run leaves its
inputs and its log and never a ``summary.json`` that reads like a result.

**Load is verified in weight space, not in behaviour space.** At construction this model is
*exactly* zero-KL: the posterior delta heads and every FiLM generator are zeroed after the generic
initialisation, so $q \equiv p$, the shared $\epsilon$ makes $z^q = z^p$ sample by sample, and the
two forecasts are bitwise identical. A behavioural probe therefore cannot separate "the checkpoint
never loaded" from "a real model whose source pathway collapsed" -- both read zero, and hard-failing
on the second would destroy the single most important finding a run can produce. So what raises is
that the zeroed tensors are no longer zero, which only a real load (or a deliberate perturbation)
can produce. That half is the sibling's, unchanged, because the initialisation it reads is.

**What this cell's guard set adds, and why each one is silent without it.**

``transform == 'causal'``
    The two dataset variants share every field name and every dtype. Only the root ``transform``
    attribute and the stored widths ($36/66/36/15$ against $43/66/43/15$) tell them apart, so a
    two-sided shard evaluated here would report a causal model on coefficients that contain their
    own future, with every shape correct. Checked on **every** configured shard, not the first.

``causal_reach_budget_s is None``
    The reach quantile is measured on the two-sided Morlet bank, which did not produce these
    coefficients. Set alongside a warm-up budget it is not a stricter run but an incoherent one,
    and the delay it resolves is a *shift* applied on top of a *mask*.

the warm-up budget against the checkpoint's stamped tuples
    ``causal_warmup_budget_steps`` is a config key and not a constructor parameter: the driver
    resolves it against the shards into ``target_keep_index``, ``target_warmup_steps``,
    ``source_keep_index`` and ``source_warmup_steps``, and those four are what land in
    ``model_kwargs``. They are in turn not config keys. So neither side of that pair can be
    reconciled by :func:`reconcile_with_checkpoint`, whose comparison silently skips a key absent
    from either -- and the comparison that *can* fail is this one: re-resolve the budget from the
    shards this run is about to read, and compare the result with what the checkpoint was built
    with. Two arms at two budgets have mutually unloadable checkpoints and the class stamp cannot
    separate them.

``fhr_st`` / ``fhr_ph`` normalised
    They are this model's **target**. Un-z-scored, the Gaussian NLL is computed against a variance
    model on another scale and the loader raises nothing.

**The causality disclosure is not the sibling's, and copying it would have been a false statement
rather than a conservative one.** The raw pipeline's sentence says the inputs read their own future,
because its features come from two-sided filters. Here they do not: the bank is strictly one-sided,
so a coefficient at step $t$ is a function of $\{x(s) : s \le t\}$ and the forecast claim is exact.
What survives is the narrower refusal -- the readout is still not a transfer entropy, because the
lag map is an attribution over *stored-coefficient* time, uncorrected for a composed group delay of
the same order as the lag search itself. The per-block group delays are read off the shards and
recorded beside the sentence, so ``summary.json`` states what its lag numbers are lags *in* without
any other file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from loguru import logger

from teb_vae.lag_attn.nets.decoders import HORIZON_ATTENTION_GAIN_INIT
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.causal_warmup import BUDGET_KEY, REACH_KEY, resolve_warmup_budget
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    COEFFICIENT_LAG_AXIS_LABEL,
    GROUP_DELAY_CAVEAT,
    MAX_MEASURED_GROUP_DELAY_SECONDS,
    SHIPPED_LAG_SPAN_SECONDS,
)
from teb_vae.lag_attn_cfs.model_kwargs import WARMUP_MODEL_KWARGS, warmup_model_kwargs

# The training entry point's own guards, imported rather than copied. Their messages name the exact
# command that regenerates a stats file and the exact reason a width mismatch must not be "fixed" by
# reverting the channel counts; a second copy could only ever drift from them. Two of the sibling's
# four are gone: ``_check_causal_budget_resolves`` guards the two-sided *reach* budget this variant
# refuses outright, and ``_check_raw_target_normalized`` is reused with this cell's own target
# fields rather than the raw ``'fhr'`` it defaults to.
from teb_vae.lag_attn_rws.trainer import (
    _check_declared_widths_against_shard,
    _check_raw_target_normalized,
    _check_stat_path,
)

if TYPE_CHECKING:  # pragma: no cover - a type name, never a runtime import
    # Imported for the annotation alone. At runtime ``binding`` imports *this* module, to wire
    # ``cfs_encoder_disclosure`` into ``CFS_BINDING``, so a runtime import here would be a cycle.
    from teb_vae.lag_attn_cfs.eval.binding import ModelBinding

#: File written into the results directory recording every check and its verdict.
PREFLIGHT_FILENAME = "preflight.json"

#: The placeholder both shipped configs carry instead of a real dataset root, so a launch fails on a
#: missing file rather than on a ``transform`` refusal someone might "fix" by dropping the warm-up
#: budget. Caught by name, and caught first, so the failure says what to do about it.
REPOINT_MARKER = "REPOINT_ME"

#: The trimmed grid this cell's whole warm-up geometry assumes. The stored per-channel warm-up
#: vectors are rebased by exactly this trim, so it places every channel's validity boundary; a
#: uniformly wrong rebase moves the anchor floor and the boundary together and every warm-fraction
#: readout still reports $1.0$.
REQUIRED_TRIM_MINUTES = 1.0

#: The dataset variant this pipeline reads, as the shards' own root attribute records it.
REQUIRED_TRANSFORM = "causal"

#: The stored blocks a causal shard carries, in the order the two streams concatenate them.
STORED_BLOCKS: Tuple[str, ...] = ("fhr_st", "fhr_ph", "up_st", "up_ph")

#: The fields the clinical questions are asked in, over and above the model's own data contract.
#: The loader *skips* a field a shard does not carry, silently, so a missing ``target`` would present
#: as "no classes found" and a missing ``epoch`` as "no trajectory data".
#:
#: ``guid`` and ``epoch`` are on this list where the sibling's carries only ``epoch``, and on this
#: cell they are load-bearing twice over: they key the per-recording aggregation and the trajectory
#: axis as they do everywhere, **and** they key the anchor tiling's per-segment phase. ``load_fields``
#: is honoured literally with no forced additions, so dropping either leaves every segment on one
#: tile grid with no shape, no count and no metric differing.
REQUIRED_EVAL_LOAD_FIELDS: Tuple[str, ...] = (
    "target",
    "epoch",
    "guid",
    "cs_label",
    "bg_label",
    "time_from_labor_onset",
)

#: The loader fields this model's reconstruction target is built from. Both blocks, because the
#: target is their concatenation and a config carrying one of them is a target with a hole in it.
#: The same tuple the experiment driver declares as its ``TARGET_FIELDS``.
TARGET_FIELDS: Tuple[str, ...] = ("fhr_st", "fhr_ph")

#: Objective keys reconciled against the checkpoint's own ``hyper_parameters``. These four decide
#: which loss the readouts are, so a disagreement means the run would report one objective's numbers
#: under another's name.
OBJECTIVE_KEYS: Tuple[str, ...] = ("likelihood", "free_bits", "lambda_full", "lambda_base")

#: Recorded from the checkpoint but never compared: $\beta$ and its ramp weight the *training* total
#: and enter none of the evaluated readouts, so a config that changed the schedule after the fit is
#: not evaluating the wrong thing.
SCHEDULE_KEYS: Tuple[str, ...] = ("kld_beta", "beta_schedule")

#: The sentence every run's summary carries, and it is **not** the sibling's.
#:
#: The raw pipeline's statement refuses causality outright, because its features come from two-sided
#: filters and a feature at step $t$ carries energy from its own future. That sentence is false here
#: and a copied refusal would be a false disclosure rather than a conservative one: this variant's
#: bank is strictly one-sided, so the forecast claim is exact.
#:
#: What survives is the narrower refusal, and it is why the string "transfer entropy" appears here
#: exactly once -- in the clause that refuses the name. The two figures are drawn from
#: :mod:`~teb_vae.lag_attn_cfs.eval.lag_axis` rather than restated, so the caveat and the axis it
#: qualifies cannot come to quote different numbers; the run's **own** per-block delays are recorded
#: beside this sentence in :func:`group_delay_summary`.
CAUSALITY_STATEMENT = (
    "The stored coefficients are produced by a strictly one-sided filter bank, so a coefficient at "
    "step t is a function of {x(s) : s <= t} alone and a forecast of step t + 1 + tau is a genuine "
    "forecast: this run's inputs do not contain their own future. The coupling readout is still "
    "named source_conditioned_kl_raw and no number in this run may be labelled a transfer entropy: "
    f"the lag map is an attribution over stored-coefficient time, uncorrected for a composed "
    f"one-sided group delay reaching {MAX_MEASURED_GROUP_DELAY_SECONDS:g} s on the committed causal "
    f"fixture -- the same order as the {SHIPPED_LAG_SPAN_SECONDS:g} s lag search itself. "
    "causal_delay_s records this run's own per-block delays."
)

#: The causality-record keys the *shared* half owns, which no encoder disclosure may return.
#:
#: Wider than the sibling's, and deliberately so. The warm-up budget, the anchor floor, the lag floor
#: and the measured lag support are properties of the **target domain** rather than of either
#: encoder: both cfs cells resolve the same budget against the same shards and decode the same anchor
#: set, so a reader comparing two models' ``preflight.json`` files compares them down these exact
#: names. Putting them in an encoder's half would mean the conv-Transformer cell either restated them
#: or silently stopped reporting them.
SHARED_CAUSALITY_KEYS: frozenset = frozenset({
    "one_sided_inputs",
    "statement",
    "transform",
    "causal_reach_budget_s",
    "group_delay_seconds",
    "warmup_budget",
    "anchor_geometry",
    "lag_support",
    "lag_axis",
    "source_delay_steps",
    "source_delay_seconds",
    "source_delay_is_max_over_channels",
    "horizon_seconds",
})


class EvalPreconditionUnmet(RuntimeError):
    """A run was refused before it computed anything, because an input could not be trusted.

    A distinct type rather than a bare ``RuntimeError``: the runner catches nothing around
    preflight, and a reader seeing ``EvalPreconditionUnmet`` in the traceback is looking at a refusal
    with an actionable message, not at a crash inside an analysis.
    """


# =================================================================================================
# Config readers
# =================================================================================================
def _dataset_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``dataset_config`` block, or an empty one."""
    return dict(config.get("dataset_config") or {})


def _dataloader_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``dataloader_config`` block, or an empty one."""
    return dict(_dataset_config(config).get("dataloader_config") or {})


def _dataset_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``dataset_kwargs`` block, or an empty one."""
    return dict(_dataloader_config(config).get("dataset_kwargs") or {})


def _vae_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``model_config.VAE_model`` block, or an empty one."""
    return dict((config.get("model_config") or {}).get("VAE_model") or {})


def _test_shards(config: Mapping[str, Any]) -> List[str]:
    """Return the configured evaluation shard paths."""
    return [str(path) for path in (_dataset_config(config).get("vae_test_datasets") or [])]


# =================================================================================================
# Config guards
# =================================================================================================
def check_repointed(config: Mapping[str, Any]) -> None:
    """Refuse a config still carrying the placeholder dataset root.

    Runs before every existence check, so the message names the real cause instead of reporting a
    missing file the operator would then go looking for.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: If any shard path or the statistics path carries the placeholder.
    """
    paths = _test_shards(config)
    stat_path = _dataset_config(config).get("stat_path")
    if stat_path is not None:
        paths.append(str(stat_path))

    offenders = [path for path in paths if REPOINT_MARKER in path]
    if offenders:
        raise EvalPreconditionUnmet(
            f"dataset_config still carries the {REPOINT_MARKER} placeholder:\n  "
            + "\n  ".join(offenders)
            + f"\nThese are deliberate non-paths, not typos. Point them at the CAUSAL k-fold "
            f"holdout split -- one HDF5 per canonical subgroup under "
            f"k_fold_cross_validation_dataset/test/ -- and at the statistics file regenerated from "
            f"those same shards at trim_minutes={REQUIRED_TRIM_MINUTES:g}."
        )


def check_test_shards_exist(config: Mapping[str, Any]) -> None:
    """Refuse a run whose evaluation shards are not on disk.

    The message names the two dataset build modes when the missing shard sits in a ``test/``
    directory that is not there, because that is the likely cause and it is invisible from the
    config: the dataset pipeline's default build mode is ``augmented``, which writes per-fold test
    splits and no shared holdout directory at all.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: If no shard is configured, or any configured shard is absent.
    """
    shards = _test_shards(config)
    if not shards:
        raise EvalPreconditionUnmet(
            "dataset_config.vae_test_datasets is empty, so there is nothing to evaluate. The "
            "evaluation reads the shared causal k-fold holdout split: one HDF5 per canonical "
            "subgroup, all eight, all three clinical classes."
        )

    missing = [path for path in shards if not Path(path).is_file()]
    if not missing:
        return

    absent_dirs = sorted(
        {str(Path(path).parent) for path in missing if not Path(path).parent.is_dir()}
    )
    detail = ""
    if absent_dirs:
        detail = (
            "\nThe directory itself is absent: "
            + ", ".join(absent_dirs)
            + "\nThe dataset pipeline builds in one of two modes. 'augmented' is the default and "
            "writes a per-fold test split inside each fold's directory; only the 'holdout' mode "
            "writes the shared test/ directory this evaluation reads. One pool, no fold loop, no "
            "double counting -- so a per-fold split is not a substitute for it."
        )
    raise EvalPreconditionUnmet(
        "dataset_config.vae_test_datasets names shard(s) that do not exist:\n  "
        + "\n  ".join(missing)
        + detail
    )


def check_trim_minutes(config: Mapping[str, Any]) -> None:
    r"""Refuse a run on a grid this cell's warm-up geometry is not valid on.

    The stored per-channel warm-up vectors are rebased by exactly this trim, so it is the trim that
    places every channel's validity boundary. A mismatch against the statistics file only
    ``warnings.warn``, and a uniformly wrong rebase is the one failure no metric can see: it moves
    the anchor floor and the validity boundary together, so ``target_warm_frac`` still reports
    exactly $1.0$ while the model scores pad.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: If ``trim_minutes`` is absent or is not $1.0$.
    """
    trim = _dataset_kwargs(config).get("trim_minutes")
    if trim is None or float(trim) != REQUIRED_TRIM_MINUTES:
        raise EvalPreconditionUnmet(
            f"dataset_config.dataloader_config.dataset_kwargs.trim_minutes must be "
            f"{REQUIRED_TRIM_MINUTES:g}, got {trim!r}. The stored causal warm-up vectors are "
            f"rebased by exactly this trim, so it places every channel's validity boundary and the "
            f"anchor floor that pairs with it. A wrong rebase moves both together and every "
            f"warm-fraction readout still reports 1.0."
        )


def check_causal_transform(config: Mapping[str, Any]) -> None:
    """Refuse a shard that is not the causal dataset variant.

    Every configured shard is read, not the first: a two-sided file beside causal ones would
    otherwise pass and be evaluated against a boundary its own coefficients do not have.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: Naming every shard whose root ``transform`` is not ``'causal'``,
            including one that declares none at all.
    """
    import h5py

    offenders: List[str] = []
    for path in _test_shards(config):
        with h5py.File(path, "r") as handle:
            declared = handle.attrs.get("transform")
        if isinstance(declared, bytes):
            declared = declared.decode("utf-8")
        if declared is None or str(declared) != REQUIRED_TRANSFORM:
            offenders.append(f"{path}: transform={declared!r}")

    if offenders:
        raise EvalPreconditionUnmet(
            f"dataset_config.vae_test_datasets names shard(s) that are not the "
            f"'{REQUIRED_TRANSFORM}' dataset variant:\n  "
            + "\n  ".join(offenders)
            + "\nThe two variants share every field name and every dtype, so only this attribute "
            "and the stored widths (36/66/36/15 against 43/66/43/15) tell them apart. A two-sided "
            "shard evaluated here would report a causal model on coefficients that contain their "
            "own future, with every shape correct and nothing raising. Repoint "
            "dataset_config.vae_test_datasets at the causal build."
        )


def check_load_fields(config: Mapping[str, Any]) -> None:
    """Refuse a config that does not ask the loader for the fields the analyses read.

    Asserted from the config rather than from a batch because the loader **skips** a field it was
    asked for and the shard does not carry, with no error at all -- so an absent field presents
    downstream as an empty result rather than as a data problem. The batch-side half of this check is
    the loader probe's, which sees what actually arrived.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: Naming every missing field.
    """
    load_fields = list(_dataset_kwargs(config).get("load_fields") or [])
    missing = [name for name in REQUIRED_EVAL_LOAD_FIELDS if name not in load_fields]
    if missing:
        raise EvalPreconditionUnmet(
            f"dataset_config.dataloader_config.dataset_kwargs.load_fields is missing {missing}. "
            f"Each is what a readout is asked in: 'target' recovers the class code (as a ratio "
            f"against 'weight'), 'epoch' is time before delivery, 'cs_label'/'bg_label' are the "
            f"subgroup cuts, and 'time_from_labor_onset' is NaN wherever the recording is absent "
            f"from the labour-onset table. 'guid' and 'epoch' additionally key the anchor tiling's "
            f"per-segment phase, and load_fields is honoured literally with no forced additions, so "
            f"dropping either leaves every segment on one tile grid with no shape, no count and no "
            f"metric differing. The evaluation override delta lists all of them; a config merged "
            f"without it carries only the fields the model itself consumes."
        )


def check_no_reach_budget(config: Mapping[str, Any]) -> None:
    """Refuse a two-sided forward-reach budget on one-sided features.

    The forward reach $L_{95}$ is an energy quantile of a *two-sided* filter, measured on the
    production Morlet bank -- which did not produce these coefficients. Set alongside the warm-up
    budget it is not a stricter run but an incoherent one, and what it resolves to is a *delay*: a
    shift, applied on top of a mask, that would make the model read every gated channel late.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: Naming both keys and the value to set.
    """
    reach = _vae_config(config).get("causal_reach_budget_s")
    if reach is not None:
        raise EvalPreconditionUnmet(
            f"{REACH_KEY}={reach!r} is set. The forward reach L95 is an energy quantile of a "
            f"two-sided filter and is undefined on one-sided features: it is measured on the "
            f"production Morlet bank, which did not produce these coefficients, and what it "
            f"resolves to is a channel *delay* applied on top of the warm-up *mask*. Set "
            f"{REACH_KEY}: null; the guard this family runs is {BUDGET_KEY}."
        )


def config_view_for_shard_guards(config: Mapping[str, Any], model: Any) -> Dict[str, Any]:
    """Build the config view the reused width guard expects, with the model's own widths.

    Two remappings, and without either the guard is a silent no-op on an evaluation run.

    ``_check_declared_widths_against_shard`` reads ``vae_train_datasets`` and returns early when the
    key is absent, so on a config that sets only ``vae_test_datasets`` it checks nothing -- and what
    it would have checked is the wrong population anyway.

    The widths compared must be the **model's**, not the config's. The evaluation rebuilds from the
    checkpoint's ``model_kwargs``, so a checkpoint whose geometry differs from the config's would
    pass a config-versus-shard comparison and then fail inside the forward with a channel error
    naming neither the checkpoint nor the config.

    Args:
        config: The merged run config. Not mutated.
        model: The rebuilt net.

    Returns:
        A shallow rebuild carrying the test shards under the training key and the model's widths.
    """
    dataset_config = dict(_dataset_config(config))
    dataset_config["vae_train_datasets"] = list(dataset_config.get("vae_test_datasets") or [])
    return {
        "dataset_config": dataset_config,
        "model_config": {
            "VAE_model": {
                "c_y": int(model.c_y),
                "c_u": int(model.c_u),
                "use_up_st": bool(model.use_up_st),
            }
        },
    }


def config_view_for_budget(config: Mapping[str, Any]) -> Dict[str, Any]:
    r"""Build the config view the warm-up resolver is re-run against: the **evaluation** shards.

    ``resolve_warmup_budget`` reads ``vae_train_datasets`` *and* ``vae_test_datasets``, which is
    right for a training driver resolving one boundary over one dataset. Here the training shards are
    whatever the checkpoint's resolved config named, and on the box an evaluation runs they may not
    exist at all -- while the boundary that matters is the one belonging to the shards this run is
    about to read. The training list is therefore emptied rather than left standing.

    Args:
        config: The merged run config. Not mutated.

    Returns:
        A deep-enough rebuild: the model block as-is, and a dataset block whose only shards are the
        evaluation ones.
    """
    dataset_config = dict(_dataset_config(config))
    dataset_config["vae_train_datasets"] = []
    return {
        "model_config": {"VAE_model": _vae_config(config)},
        "dataset_config": dataset_config,
    }


def _trainer_guard_refusal(guard: Any, config: Mapping[str, Any]) -> Optional[ValueError]:
    """Run one of the training entry point's guards and return its refusal, if any.

    The original is returned rather than swallowed so each caller can chain it: the trainer's
    messages name the exact command that regenerates a statistics file and the exact reason a width
    mismatch must not be "fixed" by reverting the channel counts, and none of that is worth
    restating here.

    Each caller raises its own :class:`EvalPreconditionUnmet` rather than sharing one wrapper,
    because :data:`GUARD_RECOVERY` is keyed on the function a refusal came from -- a single generic
    re-raiser would collapse three different causes into one row that could only name a fix in the
    abstract.

    Args:
        guard: The trainer guard to call.
        config: The config view it expects.

    Returns:
        The guard's own ``ValueError``, or ``None`` when it passed.
    """
    try:
        guard(dict(config))
    except ValueError as exc:
        return exc
    return None


def check_stat_path(config: Mapping[str, Any]) -> None:
    """Refuse a run whose normalization statistics file is unset or absent.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: Carrying the trainer guard's own message.
    """
    refusal = _trainer_guard_refusal(_check_stat_path, config)
    if refusal is not None:
        raise EvalPreconditionUnmet(str(refusal)) from refusal


def check_target_normalized(config: Mapping[str, Any]) -> None:
    """Refuse a config whose target blocks are not both loaded and normalized.

    The reused guard takes the field names as an argument precisely because which fields carry the
    target is the one thing a sibling forecasting another domain changes about it. Here they are
    ``fhr_st`` and ``fhr_ph`` rather than the raw ``'fhr'``: un-z-scored, this model's Gaussian NLL
    is computed against a variance model on another scale and nothing fails.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: Carrying the trainer guard's own message, naming the offending field
            and the offending list.
    """
    refusal = _trainer_guard_refusal(
        lambda checked: _check_raw_target_normalized(checked, fields=TARGET_FIELDS), config
    )
    if refusal is not None:
        raise EvalPreconditionUnmet(str(refusal)) from refusal


def check_declared_widths(config: Mapping[str, Any], model: Any) -> None:
    """Refuse a model whose declared widths disagree with the shards it is about to read.

    Args:
        config: The merged run config.
        model: The rebuilt net, which is the authority on the widths that will be forwarded.

    Raises:
        EvalPreconditionUnmet: Carrying the trainer guard's own message.
    """
    refusal = _trainer_guard_refusal(
        _check_declared_widths_against_shard, config_view_for_shard_guards(config, model)
    )
    if refusal is not None:
        raise EvalPreconditionUnmet(str(refusal)) from refusal


# =================================================================================================
# The warm-up budget, re-resolved against the shards this run reads
# =================================================================================================
def _as_int_tuple(values: Any) -> Optional[Tuple[int, ...]]:
    """Return a stamped channel tuple as ``tuple[int]``, or ``None`` when it is absent.

    A checkpoint's ``model_kwargs`` survives a ``torch.save``/``torch.load`` round trip as whatever
    sequence it was written as -- a tuple here, a list after a YAML round trip elsewhere -- so the
    comparison is made on values rather than on containers.

    Args:
        values: The stamped value, possibly ``None``.

    Returns:
        The tuple, or ``None``.
    """
    if values is None:
        return None
    return tuple(int(value) for value in values)


def check_warmup_budget_matches_checkpoint(
    config: Mapping[str, Any], *, model_kwargs: Mapping[str, Any], model_cls: type
) -> Dict[str, Any]:
    r"""Refuse a config whose warm-up budget does not resolve to the checkpoint's stamped tuples.

    The comparison :func:`reconcile_with_checkpoint` structurally cannot make.
    ``causal_warmup_budget_steps`` is a ``model_config.VAE_model`` key and **not** a constructor
    parameter, and the four tuples it resolves to are constructor parameters and **not** config keys
    -- so on both sides the reconciliation's ``if key not in vae_config or key not in model_kwargs``
    skips silently and passes every run. What can fail is this: re-resolve the budget against the
    shards this run is about to read, and compare it with what the checkpoint was actually built
    with.

    Two arms at two budgets have mutually unloadable checkpoints and the class stamp cannot separate
    them, so the failure this refuses is a run that reports one budget's channel axis under
    another's.

    Args:
        config: The merged run config.
        model_kwargs: The checkpoint's own constructor kwargs.
        model_cls: The network class, for the translation from a resolved budget to constructor
            keywords -- reused rather than restated, so the two cannot disagree about which keyword
            carries which tuple.

    Returns:
        The record: the threshold, the resolved widths, and the realised maximum warm-up.

    Raises:
        EvalPreconditionUnmet: If the budget cannot be resolved at all, if one side has a budget and
            the other does not, or if any of the four tuples disagrees -- naming both widths.
    """
    view = config_view_for_budget(config)
    try:
        resolved = resolve_warmup_budget(view)
    except ValueError as exc:
        raise EvalPreconditionUnmet(
            f"the causal warm-up budget could not be resolved against "
            f"dataset_config.vae_test_datasets: {exc}"
        ) from exc

    stamped = {name: _as_int_tuple(model_kwargs.get(name)) for name in WARMUP_MODEL_KWARGS}
    stamped_present = [name for name, value in stamped.items() if value is not None]

    if resolved is None:
        if stamped_present:
            raise EvalPreconditionUnmet(
                f"{BUDGET_KEY} is unset in this config, but the checkpoint was built with "
                f"{sorted(stamped_present)}. The model this run rebuilds therefore gates and masks "
                f"its channels while the config claims it does not, and every warm-fraction readout "
                f"would describe a guard the config does not know about. Evaluate the checkpoint "
                f"against its own resolved_config.yaml, which the training run writes beside it."
            )
        logger.warning(
            f"{BUDGET_KEY} is unset and the checkpoint carries no warm-up tuples: this is an "
            f"UNGATED run, reading every stored channel at every step -- including the leading "
            f"region where a one-sided filter's output is a function of assumed pre-recording "
            f"history, on coefficients whose normalisation constants excluded exactly that region."
        )
        return {"passed": True, "budget_steps": None, "gated": False}

    expected = {
        name: _as_int_tuple(value)
        for name, value in warmup_model_kwargs(resolved, model_cls).items()
    }
    disagreements = [
        f"{name}: the shards resolve to {len(expected[name] or ())} entries, the checkpoint was "
        f"built with "
        + ("none at all" if stamped[name] is None else f"{len(stamped[name] or ())}")
        for name in WARMUP_MODEL_KWARGS
        if expected[name] != stamped[name]
    ]
    if disagreements:
        raise EvalPreconditionUnmet(
            f"{BUDGET_KEY}={resolved.budget_steps} resolved against "
            f"dataset_config.vae_test_datasets disagrees with the checkpoint's own stamped "
            f"channel tuples:\n  " + "\n  ".join(disagreements)
            + f"\n{resolved.summary()}\nThe four tuples are what the constructor takes, and the "
            f"threshold that produced them is not: two arms at two budgets have mutually "
            f"unloadable checkpoints and the class stamp cannot separate them. Either repoint the "
            f"shards at the dataset this checkpoint was trained on, or set {BUDGET_KEY} to the "
            f"value that run used."
        )

    return {
        "passed": True,
        "gated": True,
        "budget_steps": int(resolved.budget_steps),
        "realised_max_warmup_steps": int(resolved.target.max_warmup),
        "target_declared_width": int(resolved.target.declared_width),
        "target_kept_width": int(resolved.target.kept_width),
        "source_declared_width": int(resolved.source.declared_width),
        "source_kept_width": int(resolved.source.kept_width),
        "target_dropped_index": list(resolved.target.dropped_index),
        "quantile": resolved.quantile,
        "summary": resolved.summary(),
    }


# =================================================================================================
# Checkpoint reconciliation
# =================================================================================================
def reconcile_with_checkpoint(
    config: Mapping[str, Any],
    *,
    model_kwargs: Mapping[str, Any],
    hyper_parameters: Mapping[str, Any],
    geometry_keys: Tuple[str, ...],
) -> Dict[str, Any]:
    """Refuse a config that contradicts the checkpoint it is being used to evaluate.

    This is what replaces the configuration inheritance the merge cannot have. The evaluation
    rebuilds the architecture from the checkpoint's own ``model_kwargs`` and the objective from its
    ``hyper_parameters``, so a config disagreeing with either is silently ignored today -- and the
    run then reports numbers from one model under another model's stated geometry.

    Only keys the config actually declares are compared: the constructor owns every default, and a
    key the config leaves out is deferring to it rather than contradicting it.

    :data:`SCHEDULE_KEYS` are recorded and **not** compared. $\\beta$ and its ramp weight the
    training total only; no evaluated readout applies them, so a schedule edited after the fit is not
    a reason to refuse the run.

    Args:
        config: The merged run config.
        model_kwargs: The checkpoint's own constructor kwargs.
        hyper_parameters: The checkpoint's own task hyperparameters.
        geometry_keys: The constructor keys to compare, from the binding of the model being
            evaluated. A key absent from the tuple is not compared and does not appear in the
            ``compared`` record, so a narrowed set is visible in the artifact rather than silently
            passing a config the run never checked.

    Returns:
        The reconciliation record: what was compared and what the checkpoint carries.

    Raises:
        EvalPreconditionUnmet: Naming every disagreeing key with both values.
    """
    vae_config = _vae_config(config)
    compared: Dict[str, Any] = {}
    disagreements: List[str] = []

    for key in geometry_keys:
        if key not in vae_config or key not in model_kwargs:
            continue
        config_value, checkpoint_value = vae_config[key], model_kwargs[key]
        compared[key] = {"config": config_value, "checkpoint": checkpoint_value}
        if config_value != checkpoint_value:
            disagreements.append(
                f"model_config.VAE_model.{key}: config says {config_value!r}, the checkpoint was "
                f"built with {checkpoint_value!r}"
            )

    for key in OBJECTIVE_KEYS:
        if key not in vae_config or key not in hyper_parameters:
            continue
        config_value, checkpoint_value = vae_config[key], hyper_parameters[key]
        compared[key] = {"config": config_value, "checkpoint": checkpoint_value}
        if config_value != checkpoint_value:
            disagreements.append(
                f"model_config.VAE_model.{key}: config says {config_value!r}, the run trained with "
                f"{checkpoint_value!r}"
            )

    if disagreements:
        raise EvalPreconditionUnmet(
            "the config disagrees with the checkpoint it is evaluating:\n  "
            + "\n  ".join(disagreements)
            + "\nThe architecture is rebuilt from the checkpoint's model_kwargs and the objective "
            "from its hyper_parameters, so the checkpoint always wins and the config's values would "
            "be reported beside numbers they did not produce. Evaluate the checkpoint against its "
            "own resolved_config.yaml, which the training run writes beside it."
        )

    return {
        "passed": True,
        "compared": compared,
        "not_compared": {
            key: hyper_parameters.get(key) for key in SCHEDULE_KEYS if key in hyper_parameters
        },
        "not_compared_reason": (
            "beta and its ramp weight the training total only; no evaluated readout applies them, "
            "so a schedule that changed after the fit does not invalidate this run. The warm-up "
            "budget is absent for a different reason and is checked by its own guard: it is a "
            "config key that names no constructor parameter, and the four tuples it resolves to are "
            "constructor parameters that name no config key, so neither side could be compared here"
        ),
    }


# =================================================================================================
# Weight-space load verification
# =================================================================================================
def load_witnesses(model: Any) -> Dict[str, List[torch.Tensor]]:
    r"""Return this model's construction-constant tensors as **deviations** from their constant.

    Unchanged from the sibling's, because the initialisation it reads is unchanged: this cell
    composes two mixins over the same network, and neither touches the posterior delta heads, the
    FiLM generators or the horizon attention's residual gains.

    Every entry is a tensor whose value at construction is known exactly, expressed here as
    ``value - constant`` -- so ``max|.|`` over a group is "how far training moved it", and the
    zero-initialised groups, whose constant is $0$, are returned unchanged.

    The groups are independent, because training moves them independently:

    ``delta_heads``
        ``posterior_head.delta_mu_head`` and ``delta_logvar_head``, zeroed so that $q \equiv p$ at
        step $0$ and any coupling the model reports had to be learned against that null. Under
        ``posterior_logvar_mode='independent'`` there is no ``delta_logvar_head``; the mean delta is
        zeroed in both modes, so the witness never becomes empty.

    ``film_generators``
        Every FiLM generator in the horizon core, zeroed *after* the generic initialisation
        xavier-refills them.

    ``horizon_attention_gains``
        The per-block residual gain of each horizon-attention block, which the constructor sets to
        exactly $10^{-2}$ and the generic initialisation does not touch. Reported as the deviation
        from that constant, so a checkpoint that never loaded reads $0$ here exactly as it does on
        the other two groups. Present only when the model was built with attention blocks.

    Args:
        model: The rebuilt net, after the checkpoint load.

    Returns:
        Witness name to that witness's deviations from its construction constant.
    """
    def _tensors(module: Any) -> List[torch.Tensor]:
        layers = list(module) if isinstance(module, torch.nn.ModuleList) else [module]
        found: List[torch.Tensor] = []
        for layer in layers:
            found.append(cast(torch.Tensor, layer.weight))
            if getattr(layer, "bias", None) is not None:
                found.append(cast(torch.Tensor, layer.bias))
        return found

    delta_heads: List[torch.Tensor] = []
    for head_name in ("delta_mu_head", "delta_logvar_head"):
        head = getattr(model.posterior_head, head_name, None)
        # None under posterior_logvar_mode='independent'; see the docstring.
        if head is not None:
            delta_heads.extend(_tensors(head))

    core = model.horizon_core
    film: List[torch.Tensor] = []
    if core.film_gen is not None:
        film.extend(_tensors(core.film_gen))
    if core.refine.film is not None:
        film.extend(_tensors(core.refine.film))

    witnesses = {"delta_heads": delta_heads, "film_generators": film}
    if core.attention is not None:
        witnesses["horizon_attention_gains"] = [
            block.residual_gain.detach() - HORIZON_ATTENTION_GAIN_INIT
            for block in core.attention
        ]
    return witnesses


def verify_weights_loaded(model: Any) -> Dict[str, Any]:
    """Verify in weight space that a checkpoint actually reached the model.

    Any-of, not all-of: the witness groups receive gradient independently, so a real checkpoint whose
    FiLM path never left zero is an ordinary model rather than a failed load. Each group's largest
    deviation from its construction constant is reported under ``max_abs_weight`` -- for the
    zero-initialised groups that *is* $\\max|w|$, which is why the field keeps its name -- so which
    witness carried the evidence is visible rather than inferred.

    Args:
        model: The rebuilt net, after the load.

    Returns:
        The per-witness magnitudes, which witnesses carried evidence, and the verdict.

    Raises:
        EvalPreconditionUnmet: If every witness tensor is still exactly at its construction constant,
            which no trained model produces and every failed load does.
    """
    magnitudes = {
        name: max((float(tensor.detach().abs().max()) for tensor in tensors), default=0.0)
        for name, tensors in load_witnesses(model).items()
    }
    carried = sorted(name for name, value in magnitudes.items() if value > 0.0)
    if not carried:
        raise EvalPreconditionUnmet(
            "every witness tensor in this model is still exactly at the value the constructor gave "
            "it, so no checkpoint weights reached it. The likeliest causes are a state dict whose "
            "keys did not align (load_checkpoint_strict returns None rather than raising) and a "
            "path naming a freshly constructed but untrained checkpoint. This is a weight-space "
            "check, not a behavioural one: a genuinely trained model whose source pathway collapsed "
            "still has nonzero weights here and passes, because that finding must be reported "
            "rather than refused.\n  "
            + "\n  ".join(f"{name}: max|w| = {value:.3e}" for name, value in magnitudes.items())
        )
    logger.info(f"checkpoint load verified in weight space; evidence from: {', '.join(carried)}")
    return {"passed": True, "max_abs_weight": magnitudes, "witnesses_with_evidence": carried}


# =================================================================================================
# The measured geometry: anchors, and the lag support the simplifications rest on
# =================================================================================================
def anchor_geometry(model: Any) -> Dict[str, Any]:
    r"""What the dense evaluation forward decodes, and at what stride the run was trained.

    The evaluation always decodes **densely**: ``resolve_anchor_geometry('test', batch)`` returns
    $(\varphi, S) = (0, 1)$, which is the geometry ``val`` and ``test`` already use. The training
    stride is recorded beside it because a figure or a table that did not say which geometry it was
    produced at would be unreadable against the training CSV -- $A_{\max}$ differs by a factor of
    $S$ between them.

    Args:
        model: The rebuilt net.

    Returns:
        The floor, the valid span, both anchor counts and the training stride.
    """
    floor = int(model.warmup_period)
    t_valid = int(model.geometry.t_valid)
    stride = int(model.anchor_stride)
    return {
        "anchor_floor": floor,
        "t_valid": t_valid,
        "horizon": int(model.horizon),
        "evaluation_stride": 1,
        "anchors_per_sample": t_valid - floor,
        "training_stride": stride,
        "training_anchors_per_sample_max": -(-(t_valid - floor) // stride),
        "target_kept_width": int(model.decoder_out_channels),
        "block_width": int(model.horizon) * int(model.decoder_out_channels),
    }


def lag_support(model: Any) -> Dict[str, Any]:
    r"""Measure how much lag support the earliest decoded anchor has, rather than assuming it.

    $$\texttt{lag\_support\_margin\_steps} = \min_t \mathcal A - (L - 1) - F_u,$$

    where $\min_t \mathcal A = F$ is the earliest decoded anchor, $L - 1 = \texttt{max\_lag}$ is the
    furthest searched lag and $F_u$ is the lag floor. At the shipped configuration it is
    $133 - 90 - 0 = 43$.

    Everything the per-lag analyses simplify away holds exactly when this is $\ge 0$: the raw
    pipeline's per-lag support correction, its untruncated recomputation over the anchors where every
    lag exists, and the truncation-aware entropy ceiling $\operatorname{mean}_t \log \min(t+1, L)$
    all collapse to no-ops, and the ceiling is exactly $\log L$. The three quantities behind it move
    independently -- a lower ``warmup_period`` arm, a wider ``max_lag``, or a non-zero ``lag_floor``
    each breaks it on its own -- so it is **measured and recorded** rather than assumed, and the two
    analyses read the number instead of asserting a simplification the geometry no longer supports.

    A negative margin is **not** a refusal. A truncated-support run is legitimate and the ported
    support corrections handle it; what would not be legitimate is an analysis claiming the
    simplification while the geometry had stopped supporting it.

    Args:
        model: The rebuilt net.

    Returns:
        The three quantities, the margin and whether every lag exists at every scored anchor.
    """
    floor = int(model.warmup_period)
    max_lag = int(model.max_lag)
    lag_floor = int(model.lag_floor)
    margin = floor - max_lag - lag_floor
    record = {
        "min_decoded_anchor": floor,
        "max_lag": max_lag,
        "n_lags": max_lag + 1,
        "lag_floor": lag_floor,
        "lag_support_margin_steps": margin,
        "every_lag_valid_at_every_anchor": bool(margin >= 0),
    }
    if margin < 0:
        logger.warning(
            f"lag support margin {margin} < 0: the earliest decoded anchor {floor} does not admit "
            f"every lag (max_lag={max_lag}, lag_floor={lag_floor}), so the per-lag readouts are "
            f"truncated at the leading anchors and the entropy ceiling is below log L. This is a "
            f"legitimate geometry, not a refusal -- the support corrections handle it -- but no "
            f"analysis may report the untruncated simplification for this run."
        )
    return record


# =================================================================================================
# Causality and group-delay disclosure
# =================================================================================================
def group_delay_summary(paths: Sequence[str]) -> Dict[str, Any]:
    r"""Reduce the shards' per-channel ``causal_delay_s`` to a per-block minimum, median and maximum.

    The composed one-sided group delay is what makes the lag axis stored-coefficient time rather than
    physical time, so its summary belongs beside the claim it qualifies. Left only in
    ``band_channel_map.csv``, a reader of ``summary.json`` would have the lag numbers and no statement
    of what they are lags *in* -- and the summary is the artifact that gets quoted.

    Read from the first shard rather than from all of them: the delay is a constant of the filter
    bank, and :func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget` has already refused a
    shard set that does not agree about the bank. The path it was read from travels in the record, so
    the number is attributable.

    Args:
        paths: The configured evaluation shards.

    Returns:
        ``{'source': <path>, <block>: {'min', 'median', 'max', 'n_channels'}}``, or an empty dict
        when no path was given. A block the shard does not carry is simply absent.
    """
    import h5py

    if not paths:
        return {}

    path = str(paths[0])
    record: Dict[str, Any] = {"source": path}
    with h5py.File(path, "r") as handle:
        for block in STORED_BLOCKS:
            if block not in handle or "causal_delay_s" not in handle[block].attrs:
                continue
            delays = np.asarray(handle[block].attrs["causal_delay_s"], dtype=np.float64)
            if delays.size == 0:
                continue
            record[block] = {
                "min": float(delays.min()),
                "median": float(np.median(delays)),
                "max": float(delays.max()),
                "n_channels": int(delays.size),
            }
    return record


def disclosed_attribute(model: Any, name: str) -> Any:
    """Read a model attribute a causality disclosure reports, or raise naming it.

    A ``getattr(model, name, default)`` would report a model that stopped exposing something as a
    model with nothing to report, and the disclosure would go quiet in exactly the case a reader most
    needs to be told.

    Public rather than private because the conv-Transformer cell's binding discloses a different set
    of attributes through the same contract, and two copies of this refusal could only ever come to
    word it differently.

    Args:
        model: The rebuilt net.
        name: The attribute to read.

    Returns:
        The attribute's value.

    Raises:
        AttributeError: Naming both the attribute and the class that does not carry it.
    """
    if not hasattr(model, name):
        raise AttributeError(
            f"{type(model).__name__} carries no {name!r}, which the causality disclosure reports. "
            f"An evaluation cannot describe an encoder it cannot read; if the attribute was "
            f"renamed, this disclosure is what has to follow it."
        )
    return getattr(model, name)


def cfs_encoder_disclosure(model: Any) -> Dict[str, Any]:
    """Return the encoder-specific half of the causality record for the recurrent encoder.

    Only what is a property of the **encoder**. The one-sidedness, the group delays, the warm-up
    budget, the anchor geometry and the lag support are properties of the target domain and of the
    data, true of both cfs cells, so they live in the shared half of
    :func:`causality_disclosure` and are compared between the two models down one set of key names.
    What is left here is ``causal_norm``, which is a key one architecture has and the
    conv-Transformer cell has no equivalent for at all.

    Takes ``model: Any`` and imports no model class, so the module stays free of the network it
    describes. Both attributes are read through :func:`disclosed_attribute`, which raises naming the
    class: a model that stopped exposing one must be reported as such rather than disclosed as a
    model with nothing to say.

    Args:
        model: The rebuilt net, which is what was trained and is therefore the authority on the guard
            it actually carries.

    Returns:
        The encoder's own keys, merged into the causality record in this order.

    Raises:
        AttributeError: If the model carries neither attribute, naming it and the class.
    """
    record: Dict[str, Any] = {
        "causal_norm": bool(disclosed_attribute(model, "causal_norm")),
        "n_causalized_norms": int(disclosed_attribute(model, "n_causalized_norms")),
    }
    if not record["causal_norm"]:
        # A second, independent failure of the same interpretation, and the more serious one: the
        # encoders' GroupNorm would pool across time, so the prior itself conditions on the future --
        # which is exactly the property the causal dataset variant exists to remove upstream.
        record["causal_norm_consequence"] = (
            "causal_norm=False: the encoders' GroupNorm pools statistics across the whole time "
            "axis, so every history state carries an image of its own future and the prior "
            "p(z_t | Y_<=t) conditions on Y_>t. The one-sidedness of the stored coefficients does "
            "not survive an encoder that pools over time, so the forecast claim above holds of the "
            "DATA and not of this run's model."
        )
        logger.warning(record["causal_norm_consequence"])
    return record


def causality_disclosure(
    config: Mapping[str, Any],
    model: Any,
    encoder_disclosure: Callable[[Any], Dict[str, Any]] = cfs_encoder_disclosure,
    *,
    warmup: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record what this run's causal standing actually is, in the run's own artifacts.

    Unconditional, including the statement. What is *not* unconditional is the encoder half: a model
    whose normalisers pool over time carries the one-sidedness of the data and not of its own prior,
    and :func:`cfs_encoder_disclosure` says so where that is the case.

    Args:
        config: The merged run config, for the configured reach budget and the shards the group
            delays are read from.
        model: The rebuilt net, which is what was trained and is therefore the authority on the
            geometry it actually carries.
        encoder_disclosure: Returns the encoder-specific keys, merged in at the position they occupy
            in the record. Defaults to this cell's, so a caller with one model to evaluate says
            nothing extra; a run of the conv-Transformer cell passes its own through the binding.
        warmup: The record :func:`check_warmup_budget_matches_checkpoint` returned, so the resolved
            budget travels with the claim it qualifies. ``None`` omits it, which is what a caller
            disclosing a model without a config's shards has.

    Returns:
        The disclosure record, promoted into ``summary.json`` as well as written to
        ``preflight.json``.

    Raises:
        ValueError: If the encoder disclosure returns a key the shared record owns.
    """
    horizon_seconds = float(model.horizon) * SECONDS_PER_STEP
    delay_steps = int(model.source_delay_steps)
    support = lag_support(model)
    # The encoder's own block, whatever it turns out to be for this architecture. Checked against the
    # keys the shared record owns before it is merged: the splat sits mid-literal, so a reused name
    # would either replace a shared key -- including ``statement`` -- or be dropped by a key below
    # it, and both would be silent in an artifact whose whole purpose is to be read literally.
    disclosed = encoder_disclosure(model)
    reserved = sorted(set(disclosed) & SHARED_CAUSALITY_KEYS)
    if reserved:
        raise ValueError(
            f"the encoder disclosure returned key(s) the shared causality record already owns: "
            f"{reserved}. An encoder discloses what is true of *it*; the shared half is not "
            f"overridable, because a reader compares two models' records down these key names -- "
            f"and on this cell the shared half includes the target domain's own geometry, which "
            f"both encoders share."
        )
    return {
        "one_sided_inputs": True,
        "statement": CAUSALITY_STATEMENT,
        "transform": REQUIRED_TRANSFORM,
        # Refused by its own guard, and recorded so the artifact states it rather than implying it.
        "causal_reach_budget_s": _vae_config(config).get("causal_reach_budget_s"),
        "group_delay_seconds": group_delay_summary(_test_shards(config)),
        "warmup_budget": dict(warmup) if warmup is not None else None,
        "anchor_geometry": anchor_geometry(model),
        "lag_support": support,
        "lag_axis": {"label": COEFFICIENT_LAG_AXIS_LABEL, "caveat": GROUP_DELAY_CAVEAT},
        **disclosed,
        "source_delay_steps": delay_steps,
        "source_delay_seconds": delay_steps * SECONDS_PER_STEP,
        # Per-channel delays have no single representative, so the maximum is used and every lag
        # computed from it is an upper bound. Recorded beside the number so the choice travels.
        "source_delay_is_max_over_channels": True,
        "horizon_seconds": horizon_seconds,
    }


# =================================================================================================
# The guard recovery table
# =================================================================================================
#: One row per function carrying a ``raise EvalPreconditionUnmet``, naming what to change.
#:
#: A mapping rather than prose so the evaluation record can render it and so
#: ``tests/test_eval_preflight.py`` can walk this module's AST and assert that every raise site has a
#: row -- the sibling's table is hand-kept in a document, and a table nothing checks is a table that
#: goes stale the first time a guard is added.
#:
#: Each ``recovery`` names a config key or a command. "The shards are wrong" is a description of the
#: problem; "repoint dataset_config.vae_test_datasets" is a recovery.
GUARD_RECOVERY: Dict[str, Dict[str, str]] = {
    "check_repointed": {
        "cause": "a shard path or stat_path still carries the REPOINT_ME placeholder",
        "recovery": (
            "edit dataset_config.vae_test_datasets and dataset_config.stat_path in "
            "eval/configs/eval_overrides.yaml to name the causal holdout split and its statistics "
            "file"
        ),
    },
    "check_test_shards_exist": {
        "cause": "a configured evaluation shard is not on disk, or none is configured",
        "recovery": (
            "point dataset_config.vae_test_datasets at an existing build, and rebuild the dataset "
            "in 'holdout' mode if the test/ directory itself is absent -- the pipeline's default "
            "'augmented' mode writes per-fold splits instead"
        ),
    },
    "check_stat_path": {
        "cause": "dataset_config.stat_path is unset or names a file that is not there",
        "recovery": (
            "regenerate it with hdf5_dataset/calculate_dataset_stats.py from the configured causal "
            "shards at trim_minutes=1.0, and set dataset_config.stat_path to the result"
        ),
    },
    "check_trim_minutes": {
        "cause": "the loader's trim is not the one the stored warm-up vectors were rebased at",
        "recovery": (
            "set dataset_config.dataloader_config.dataset_kwargs.trim_minutes: 1.0"
        ),
    },
    "check_causal_transform": {
        "cause": "a configured shard is the two-sided dataset variant, or declares no transform",
        "recovery": (
            "repoint dataset_config.vae_test_datasets at the causal build "
            "(hdf5_dataset/new_pipeline, causal variant, 'holdout' mode)"
        ),
    },
    "check_load_fields": {
        "cause": "dataset_config.dataloader_config.dataset_kwargs.load_fields omits a field a "
                 "readout is asked in, or a key the anchor tiling's phase is derived from",
        "recovery": (
            "add the named field to dataset_config.dataloader_config.dataset_kwargs.load_fields; "
            "the committed eval/configs/eval_overrides.yaml lists the full set"
        ),
    },
    "check_target_normalized": {
        "cause": "fhr_st or fhr_ph is missing from load_fields or from normalize_fields",
        "recovery": (
            "add both to dataset_config.dataloader_config.normalize_fields and to "
            "dataset_config.dataloader_config.dataset_kwargs.load_fields"
        ),
    },
    "check_no_reach_budget": {
        "cause": "model_config.VAE_model.causal_reach_budget_s is set on one-sided features",
        "recovery": "set model_config.VAE_model.causal_reach_budget_s: null",
    },
    "check_declared_widths": {
        "cause": "the model's c_y / c_u disagree with the configured shards' stored widths",
        "recovery": (
            "point dataset_config.vae_test_datasets at the shards this checkpoint was trained on; "
            "do not change model_config.VAE_model.c_y / c_u, which the checkpoint overrules"
        ),
    },
    "check_warmup_budget_matches_checkpoint": {
        "cause": "the warm-up budget re-resolved against the configured shards does not produce the "
                 "checkpoint's own stamped channel tuples",
        "recovery": (
            "set model_config.VAE_model.causal_warmup_budget_steps to the value the training run "
            "used, or repoint dataset_config.vae_test_datasets at that run's dataset"
        ),
    },
    "reconcile_with_checkpoint": {
        "cause": "a declared geometry or objective key contradicts the checkpoint",
        "recovery": (
            "evaluate the checkpoint against its own model_checkpoints/resolved_config.yaml, which "
            "the training run writes beside it"
        ),
    },
    "verify_weights_loaded": {
        "cause": "every witness tensor is still at its construction constant, so no weights loaded",
        "recovery": (
            "pass --checkpoint a trained .ckpt whose model_class matches this package's model; a "
            "freshly constructed checkpoint and a state dict whose keys did not align both land here"
        ),
    },
}


# =================================================================================================
# The run
# =================================================================================================
def run_preflight(
    *,
    config: Mapping[str, Any],
    model: Any,
    checkpoint_path: Any,
    model_kwargs: Mapping[str, Any],
    hyper_parameters: Mapping[str, Any],
    binding: "ModelBinding",
) -> Dict[str, Any]:
    """Run every guard, then record the run's causal standing.

    Order is deliberate. The placeholder check is first so its message is not pre-empted by a
    missing-file error someone would then go looking for; the shard, statistics, trim, transform,
    field and reach-budget guards follow, because each is a cheap config or attribute read; the width
    comparison and the reconciliation come next, because each needs the rebuilt model; the warm-up
    budget is re-resolved after them, because it is the expensive one -- it opens every configured
    shard -- and because a width or geometry disagreement gives a clearer message than a tuple-length
    mismatch would; and the weight-space load verification is last, because it is the check whose
    failure means nothing else mattered.

    Args:
        config: The merged run config.
        model: The rebuilt net, after the checkpoint load.
        checkpoint_path: The checkpoint being evaluated, recorded in the output.
        model_kwargs: The checkpoint's own constructor kwargs.
        hyper_parameters: The checkpoint's own task hyperparameters.
        binding: The binding of the model being evaluated, for the constructor keys the
            reconciliation compares, the class the warm-up tuples are translated for, and the encoder
            half of the causality record. Passed in rather than imported, so this module still names
            no model class.

    Returns:
        The preflight record, ready for :func:`write_preflight`.

    Raises:
        EvalPreconditionUnmet: On any failed guard, naming the fix.
    """
    check_repointed(config)
    check_test_shards_exist(config)
    check_stat_path(config)
    check_trim_minutes(config)
    check_causal_transform(config)
    check_load_fields(config)
    check_target_normalized(config)
    check_no_reach_budget(config)
    check_declared_widths(config, model)

    reconciliation = reconcile_with_checkpoint(
        config,
        model_kwargs=model_kwargs,
        hyper_parameters=hyper_parameters,
        geometry_keys=binding.geometry_keys,
    )
    warmup = check_warmup_budget_matches_checkpoint(
        config, model_kwargs=model_kwargs, model_cls=binding.model_cls
    )
    load_check = verify_weights_loaded(model)

    logger.info(
        "preflight passed: the shards are the causal variant, the statistics path, the trimmed "
        "grid, the loaded and normalised fields and the declared widths all hold, the config agrees "
        "with the checkpoint, the warm-up budget re-resolves to the checkpoint's own channel "
        "tuples, and the zero-initialised tensors carry loaded weights"
    )
    return {
        "checkpoint": str(checkpoint_path),
        "dataset_paths": _test_shards(config),
        "checks": {
            "repoint_placeholder": {"passed": True},
            "test_shards_exist": {"passed": True, "n_shards": len(_test_shards(config))},
            "stat_path": {"passed": True, "path": _dataset_config(config).get("stat_path")},
            "trim_minutes": {"passed": True, "value": REQUIRED_TRIM_MINUTES},
            "causal_transform": {
                "passed": True,
                "transform": REQUIRED_TRANSFORM,
                "n_shards_checked": len(_test_shards(config)),
            },
            "load_fields": {"passed": True, "required": list(REQUIRED_EVAL_LOAD_FIELDS)},
            "target_normalized": {"passed": True, "fields": list(TARGET_FIELDS)},
            "no_reach_budget": {"passed": True, "causal_reach_budget_s": None},
            "declared_widths": {
                "passed": True,
                "compared_against": "the model's own c_y / c_u, from the checkpoint's model_kwargs",
                "shards": _test_shards(config)[:1],
            },
            "config_matches_checkpoint": reconciliation,
            "warmup_budget_matches_checkpoint": warmup,
            "weights_loaded": load_check,
        },
        "causality": causality_disclosure(config, model, binding.encoder_disclosure, warmup=warmup),
    }


def write_preflight(record: Mapping[str, Any], output_dir: Any) -> Path:
    """Write the preflight record into the results directory.

    Args:
        record: The record from :func:`run_preflight`.
        output_dir: The results directory, created if absent.

    Returns:
        The path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PREFLIGHT_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(record), handle, indent=2, default=str)
    logger.info(f"wrote {path}")
    return path
