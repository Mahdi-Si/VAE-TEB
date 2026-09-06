r"""Everything that must hold before a single number is computed.

The expensive failures in this pipeline are the silent ones. A checkpoint that did not load, a
config whose geometry contradicts the one the weights were trained under, a raw target that
arrives at $140$ bpm while the decoder's log-variance models a $z$-scale, an evaluation pointed
at the healthy-only pretraining split instead of the holdout -- each of them produces a full set
of plausible numbers and no error. The guards here run before the loader is built and before any
analysis writes anything, so a rejected run costs a checkpoint load and two HDF5 shape reads.

**A refusal is not a crash, and the type says so.** Every guard raises
:class:`EvalPreconditionUnmet`, including the four reused from the training entry point, whose
own ``ValueError`` is re-raised from the original so the trainer's long actionable message
survives verbatim. The runner calls this *outside* its fail-soft step wrapper, so a rejected run
leaves its inputs and its log and never a ``summary.json`` that reads like a result.

**Load is verified in weight space, not in behaviour space.** At construction this model is
*exactly* zero-KL: the posterior delta heads and every FiLM generator are zeroed after the
generic initialisation, so $q \equiv p$, the shared $\epsilon$ makes $z^q = z^p$ sample by
sample, and the two forecasts are bitwise identical. A behavioural probe therefore cannot
separate "the checkpoint never loaded" from "a real model whose source pathway collapsed" -- both
read zero, and hard-failing on the second would destroy the single most important finding a run
can produce. So what raises is that the zeroed tensors are no longer zero, which only a real load
(or a deliberate perturbation) can produce.

The witness rule is **any-of**, and that is deliberate rather than lenient. Training moves the
delta heads, the FiLM generators and the horizon attention's residual gains independently, so a
checkpoint whose FiLM path never left zero is an ordinary model, not a failed load; an all-of rule
would refuse it, and it would refuse this repository's own perturbed-init test fixture as well.
Every witness's largest deviation from its construction constant is reported beside the verdict,
so which one carried the evidence is visible rather than inferred.

**The coupling readout is not causal, and every artifact says so.** Under the shipped
``causal_reach_budget_s: null`` an input feature at step $t$ reads well over a quarter of an hour
into its own future. Even a finite budget does not remove that: the reach it prunes on is a
$95\%$-energy quantile rather than a hard support, measured at roughly $20\times$ suppression at
$120$ s rather than removal. So the disclosure is unconditional, the budget, the surviving
channel counts and the bank's own longest reach are recomputed and recorded beside it, and no
artifact this pipeline writes calls the readout a transfer entropy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple, cast

import torch
from loguru import logger

from teb_vae.lag_attn import channel_reach
from teb_vae.lag_attn_rws.eval.binding import ModelBinding
from teb_vae.lag_attn.nets.decoders import HORIZON_ATTENTION_GAIN_INIT
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

# The training entry point's own guards, imported rather than copied. Their messages name the
# exact command that regenerates a stats file and the exact reason a width mismatch must not be
# "fixed" by reverting the channel counts; a second copy could only ever drift from them.
from teb_vae.lag_attn_rws.trainer import (
    _check_causal_budget_resolves,
    _check_declared_widths_against_shard,
    _check_raw_target_normalized,
    _check_stat_path,
)

#: File written into the results directory recording every check and its verdict.
PREFLIGHT_FILENAME = "preflight.json"

#: The placeholder both shipped configs carry instead of a real dataset root, so a launch fails
#: on a missing file rather than on a width mismatch someone might "fix" by reverting $c_y$ /
#: $c_u$. Caught by name, and caught first, so the failure says what to do about it.
REPOINT_MARKER = "REPOINT_ME"

#: The trimmed grid the whole raw-index geometry assumes. A mismatch against the statistics file
#: only ``warnings.warn``, and on the untrimmed grid anchor $0$'s forecast starts at raw sample
#: $256$ rather than $16$ -- off by exactly one minute, with nothing failing loudly.
REQUIRED_TRIM_MINUTES = 1.0

#: The fields the clinical questions are asked in, over and above the model's own data contract.
#: The loader *skips* a field a shard does not carry, silently, so a missing ``target`` would
#: present as "no classes found" and a missing ``epoch`` as "no trajectory data".
REQUIRED_EVAL_LOAD_FIELDS: Tuple[str, ...] = (
    "target",
    "epoch",
    "cs_label",
    "bg_label",
    "time_from_labor_onset",
    "second_stage_onset",
)

#: Constructor keys reconciled against the checkpoint's own ``model_kwargs``. Every one of them
#: changes what the numbers mean: the geometry decides which raw samples an anchor forecasts, and
#: the widths decide which channels feed which stream.
#:
#: This is the *rws* model's set, and it is what ``RWS_BINDING.geometry_keys`` is built from. The
#: reconciliation itself takes the tuple as an argument, because a second architecture reconciles
#: a different set -- its own encoder's keys, and not ``causal_norm``, which its constructor
#: refuses outright.
#:
#: The objective weights are deliberately absent, ``beta_prior`` and the three shape lambdas
#: alike: they weight the *training* criterion and enter none of the evaluated readouts, so a
#: config that changed one after the fit is not evaluating the wrong thing.
GEOMETRY_KEYS: Tuple[str, ...] = (
    "sequence_length",
    "d_model",
    "d_z",
    "horizon",
    "raw_per_step",
    "warmup_period",
    "c_y",
    "c_u",
    "use_up_st",
    "max_lag",
    "num_heads",
    "d_head",
    "horizon_attention_blocks",
    "causal_norm",
)

#: Objective keys reconciled against the checkpoint's own ``hyper_parameters``. These four decide
#: which loss the readouts are, so a disagreement means the run would report one objective's
#: numbers under another's name.
OBJECTIVE_KEYS: Tuple[str, ...] = ("likelihood", "free_bits", "lambda_full", "lambda_base")

#: Recorded from the checkpoint but never compared: $\beta$ and its ramp weight the *training*
#: total and enter none of the evaluated readouts, so a config that changed the schedule after
#: the fit is not evaluating the wrong thing.
SCHEDULE_KEYS: Tuple[str, ...] = ("kld_beta", "beta_schedule")

#: The sentence every run's summary carries. Verbatim rather than assembled, because it is the
#: one statement that stops the readout being read as the quantity it is named after.
#:
#: It deliberately carries **no** number. How far the longest-reaching channel looks ahead is a
#: property of the production filter bank, it is recomputed into ``max_channel_reach_s`` on every
#: run, and the figure quoted elsewhere in this repository is already about $9$ s stale against
#: the bank that actually ships -- so a literal here would be one more copy to go out of date.
NOT_CAUSAL_STATEMENT = (
    "The source-conditioned KL is a coupling readout, NOT a transfer entropy and NOT causal. "
    "The input features come from two-sided filters whose forward reach is bounded by a "
    "95%-energy quantile rather than by a hard support, so a feature at step t carries energy "
    "from its own future under every configured budget -- max_channel_reach_s records how far "
    "the longest-reaching channel of this feature bank looks ahead, and under the shipped "
    "causal_reach_budget_s: null no channel is pruned and no delay is applied at all. No number "
    "in this run may be labelled a transfer entropy."
)


class EvalPreconditionUnmet(RuntimeError):
    """A run was refused before it computed anything, because an input could not be trusted.

    A distinct type rather than a bare ``RuntimeError``: the runner catches nothing around
    preflight, and a reader seeing ``EvalPreconditionUnmet`` in the traceback is looking at a
    refusal with an actionable message, not at a crash inside an analysis.
    """


# =============================================================================
# Config guards
# =============================================================================
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


def check_repointed(config: Mapping[str, Any]) -> None:
    """Refuse a config still carrying the placeholder dataset root.

    Runs before every existence check, so the message names the real cause instead of reporting
    a missing file the operator would then go looking for.

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
            + f"\nThese are deliberate non-paths, not typos. Point them at the shared k-fold "
            f"holdout split -- one HDF5 per canonical subgroup under "
            f"k_fold_cross_validation_dataset/test/ -- and at the statistics file regenerated "
            f"from the same dataset at trim_minutes={REQUIRED_TRIM_MINUTES:g}."
        )


def check_test_shards_exist(config: Mapping[str, Any]) -> None:
    """Refuse a run whose evaluation shards are not on disk.

    The message names the two dataset build modes when the missing shard sits in a ``test/``
    directory that is not there, because that is the likely cause and it is invisible from the
    config: the dataset pipeline's default build mode is ``augmented``, which writes per-fold
    test splits and no shared holdout directory at all.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: If no shard is configured, or any configured shard is absent.
    """
    shards = _test_shards(config)
    if not shards:
        raise EvalPreconditionUnmet(
            "dataset_config.vae_test_datasets is empty, so there is nothing to evaluate. The "
            "evaluation reads the shared k-fold holdout split: one HDF5 per canonical subgroup, "
            "all eight, all three clinical classes."
        )

    missing = [path for path in shards if not Path(path).is_file()]
    if not missing:
        return

    absent_dirs = sorted({str(Path(path).parent) for path in missing if not Path(path).parent.is_dir()})
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
    r"""Refuse a run on a grid the raw-index geometry is not valid on.

    The forecast of anchor $t$ starts at raw sample $16(t+1)$ *on the trimmed grid*. Untrimmed it
    starts at $16(t+16)$, so anchor $0$'s $H$-step block begins one full minute later than
    every mask, every event index and every bpm overlay assumes. A mismatch against the
    statistics file's own trim only ``warnings.warn``.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: If ``trim_minutes`` is absent or is not $1.0$.
    """
    trim = _dataset_kwargs(config).get("trim_minutes")
    if trim is None or float(trim) != REQUIRED_TRIM_MINUTES:
        raise EvalPreconditionUnmet(
            f"dataset_config.dataloader_config.dataset_kwargs.trim_minutes must be "
            f"{REQUIRED_TRIM_MINUTES:g}, got {trim!r}. The whole raw-index geometry -- the "
            f"forecast of anchor t starting at raw sample 16*(t+1) -- assumes the trimmed grid, "
            f"and the statistics file was computed on it. On the untrimmed grid every forecast "
            f"window is shifted by one minute and nothing fails loudly."
        )


def check_load_fields(config: Mapping[str, Any]) -> None:
    """Refuse a config that does not ask the loader for the fields the analyses read.

    Asserted from the config rather than from a batch because the loader **skips** a field it was
    asked for and the shard does not carry, with no error at all -- so an absent field presents
    downstream as an empty result rather than as a data problem. The batch-side half of this
    check is the loader probe's, which sees what actually arrived.

    Args:
        config: The merged run config.

    Raises:
        EvalPreconditionUnmet: Naming every missing field.
    """
    load_fields = list(_dataset_kwargs(config).get("load_fields") or [])
    missing = [name for name in REQUIRED_EVAL_LOAD_FIELDS if name not in load_fields]
    if missing:
        raise EvalPreconditionUnmet(
            f"dataset_config.dataloader_config.dataset_kwargs.load_fields is missing "
            f"{missing}. Each is what a clinical readout is asked in: 'target' recovers the class "
            f"code (as a ratio against 'weight'), 'epoch' is time before delivery, "
            f"'cs_label'/'bg_label' are the subgroup cuts, 'time_from_labor_onset' is NaN "
            f"wherever the recording is absent from the labour-onset table, and "
            f"'second_stage_onset' is the signed seconds from second-stage onset that the second "
            f"clinical clock is resolved against. The evaluation override delta adds all six; a "
            f"config merged without it carries only the eight fields the model itself consumes."
        )


def config_view_for_shard_guards(
    config: Mapping[str, Any], model: Any
) -> Dict[str, Any]:
    """Build the config view the reused width guard expects, with the model's own widths.

    Two remappings, and without either the guard is a silent no-op on an evaluation run.

    ``_check_declared_widths_against_shard`` reads ``vae_train_datasets`` and returns early when
    the key is absent, so on a config that sets only ``vae_test_datasets`` it checks nothing --
    and what it would have checked is the wrong population anyway.

    The widths compared must be the **model's**, not the config's. The evaluation rebuilds from
    the checkpoint's ``model_kwargs``, so a checkpoint whose geometry differs from the config's
    would pass a config-versus-shard comparison and then fail inside the forward with a channel
    error naming neither the checkpoint nor the config.

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


def _reuse_trainer_guard(guard: Any, config: Mapping[str, Any], name: str) -> None:
    """Run one of the training entry point's guards, re-raising its refusal as this module's.

    The original is chained rather than replaced: the trainer's messages name the exact command
    that regenerates a statistics file and the exact reason a width mismatch must not be "fixed"
    by reverting the channel counts, and none of that is worth restating here.

    Args:
        guard: The trainer guard to call.
        config: The config view it expects.
        name: The check's name, for the log line.

    Raises:
        EvalPreconditionUnmet: Carrying the guard's own message.
    """
    try:
        guard(dict(config))
    except ValueError as exc:
        raise EvalPreconditionUnmet(str(exc)) from exc
    logger.debug(f"preflight check passed: {name}")


# =============================================================================
# Checkpoint reconciliation
# =============================================================================
def reconcile_with_checkpoint(
    config: Mapping[str, Any],
    *,
    model_kwargs: Mapping[str, Any],
    hyper_parameters: Mapping[str, Any],
    geometry_keys: Tuple[str, ...],
) -> Dict[str, Any]:
    """Refuse a config that contradicts the checkpoint it is being used to evaluate.

    This is what replaces the configuration inheritance the merge cannot have. The evaluation
    rebuilds the architecture from the checkpoint's own ``model_kwargs`` and the objective from
    its ``hyper_parameters``, so a config disagreeing with either is silently ignored today --
    and the run then reports numbers from one model under another model's stated geometry.

    Only keys the config actually declares are compared: the constructor owns every default, and
    a key the config leaves out is deferring to it rather than contradicting it.

    :data:`SCHEDULE_KEYS` are recorded and **not** compared. $\\beta$ and its ramp weight the
    training total only; no evaluated readout applies them, so a schedule edited after the fit is
    not a reason to refuse the run.

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
                f"model_config.VAE_model.{key}: config says {config_value!r}, the checkpoint "
                f"was built with {checkpoint_value!r}"
            )

    for key in OBJECTIVE_KEYS:
        if key not in vae_config or key not in hyper_parameters:
            continue
        config_value, checkpoint_value = vae_config[key], hyper_parameters[key]
        compared[key] = {"config": config_value, "checkpoint": checkpoint_value}
        if config_value != checkpoint_value:
            disagreements.append(
                f"model_config.VAE_model.{key}: config says {config_value!r}, the run trained "
                f"with {checkpoint_value!r}"
            )

    if disagreements:
        raise EvalPreconditionUnmet(
            "the config disagrees with the checkpoint it is evaluating:\n  "
            + "\n  ".join(disagreements)
            + "\nThe architecture is rebuilt from the checkpoint's model_kwargs and the "
            "objective from its hyper_parameters, so the checkpoint always wins and the config's "
            "values would be reported beside numbers they did not produce. Evaluate the "
            "checkpoint against its own resolved_config.yaml, which the training run writes "
            "beside it."
        )

    return {
        "passed": True,
        "compared": compared,
        "not_compared": {
            key: hyper_parameters.get(key)
            for key in SCHEDULE_KEYS
            if key in hyper_parameters
        },
        "not_compared_reason": (
            "beta and its ramp weight the training total only; no evaluated readout applies "
            "them, so a schedule that changed after the fit does not invalidate this run"
        ),
    }


# =============================================================================
# Weight-space load verification
# =============================================================================
def load_witnesses(model: Any) -> Dict[str, List[torch.Tensor]]:
    r"""Return this model's construction-constant tensors as **deviations** from their constant.

    Every entry is a tensor whose value at construction is known exactly, expressed here as
    ``value - constant`` -- so ``max|.|`` over a group is "how far training moved it", and the
    zero-initialised groups, whose constant is $0$, are returned unchanged. Expressing the
    deviation here rather than in :func:`verify_weights_loaded` is what lets a witness whose
    constant is *not* zero join the set without turning the check into one that can never fail.

    The groups are independent, because training moves them independently:

    ``delta_heads``
        ``posterior_head.delta_mu_head`` and ``delta_logvar_head``, zeroed so that $q \equiv p$
        at step $0$ and any coupling the model reports had to be learned against that null.

        Under ``posterior_logvar_mode='independent'`` there is no ``delta_logvar_head`` -- the
        posterior's log-variance is its own head, seeded at the pre-image of unit scale rather
        than at zero, so it is **not** a zero-witness and is deliberately not collected here. The
        mean delta is still zeroed in both modes, so the witness never becomes empty; an empty
        witness would make the load check pass on a checkpoint that loaded nothing.

    ``film_generators``
        Every FiLM generator in the horizon core, zeroed *after* the generic initialisation
        xavier-refills them, so the per-block-FiLM decoder starts bitwise identical to the
        FiLM-free one and consults $z$ through that path only as training drives it off zero.

    ``horizon_attention_gains``
        The per-block residual gain of each horizon-attention block, which the constructor sets to
        exactly $10^{-2}$ and the generic initialisation does not touch. Reported as the deviation
        from that constant, so a checkpoint that never loaded reads $0$ here exactly as it does on
        the other two groups. Present only when the model was built with attention blocks: an
        empty group would add a permanent ``0.0`` to every run's record, and a group that is not
        there says more than one that is always zero.

    Both heads are ``ModuleList``s under the head-structured posterior, so each is flattened
    rather than assumed to be a single layer, and biases are included: ``_zero_linear`` clears
    them too, so a checkpoint that moved only a bias is still evidence of a load.

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

    Any-of, not all-of: the witness groups receive gradient independently, so a real checkpoint
    whose FiLM path never left zero is an ordinary model rather than a failed load. Each group's
    largest deviation from its construction constant is reported under ``max_abs_weight`` -- for
    the zero-initialised groups that *is* $\\max|w|$, which is why the field keeps its name -- so
    which witness carried the evidence is visible rather than inferred.

    Args:
        model: The rebuilt net, after the load.

    Returns:
        The per-witness magnitudes, which witnesses carried evidence, and the verdict.

    Raises:
        EvalPreconditionUnmet: If every witness tensor is still exactly at its construction
            constant, which no trained model produces and every failed load does.
    """
    magnitudes = {
        name: max((float(tensor.detach().abs().max()) for tensor in tensors), default=0.0)
        for name, tensors in load_witnesses(model).items()
    }
    carried = sorted(name for name, value in magnitudes.items() if value > 0.0)
    if not carried:
        raise EvalPreconditionUnmet(
            "every witness tensor in this model is still exactly at the value the constructor "
            "gave it, so no checkpoint weights reached it. The likeliest causes are a state dict "
            "whose keys did not align "
            "(load_checkpoint_strict returns None rather than raising) and a path naming a "
            "freshly constructed but untrained checkpoint. This is a weight-space check, not a "
            "behavioural one: a genuinely trained model whose source pathway collapsed still has "
            "nonzero weights here and passes, because that finding must be reported rather than "
            "refused.\n  "
            + "\n  ".join(f"{name}: max|w| = {value:.3e}" for name, value in magnitudes.items())
        )
    logger.info(f"checkpoint load verified in weight space; evidence from: {', '.join(carried)}")
    return {"passed": True, "max_abs_weight": magnitudes, "witnesses_with_evidence": carried}


# =============================================================================
# Causality and reach disclosure
# =============================================================================
def channels_reading_past_the_horizon(horizon_seconds: float) -> Dict[str, Any]:
    r"""Count, per stored feature block, the channels whose forward reach exceeds the horizon.

    The horizon is the natural yardstick: a channel at step $t$ whose reach exceeds $H\Delta$
    carries energy from beyond the last sample the model is asked to forecast, so its "history"
    state has already seen the answer.

    Recomputed from :func:`~teb_vae.lag_attn.channel_reach.block_reach_seconds` on every run
    rather than stored as constants, because the reaches are a property of the production filter
    bank and a bank change must move these numbers.

    Args:
        horizon_seconds: The forecast horizon in seconds, $H\Delta$.

    Returns:
        Per block, the count over the horizon, the block width, and the largest reach seen.
    """
    return {
        name: {
            "n_over_horizon": int(sum(1 for reach in reaches if reach > horizon_seconds)),
            "n_channels": len(reaches),
            "max_reach_s": float(max(reaches)) if reaches else 0.0,
        }
        for name, reaches in channel_reach.block_reach_seconds().items()
    }


#: The causality-record keys the *shared* half owns, which no encoder disclosure may return. They
#: are properties of the feature bank and the geometry rather than of either encoder, and a reader
#: comparing two models' ``preflight.json`` files compares them down these exact names.
SHARED_CAUSALITY_KEYS: frozenset = frozenset({
    "not_causal",
    "statement",
    "max_channel_reach_s",
    "causal_reach_budget_s",
    "source_delay_steps",
    "source_delay_seconds",
    "source_delay_is_max_over_channels",
    "horizon_seconds",
    "channels_reading_past_the_horizon",
})


def rws_encoder_disclosure(model: Any) -> Dict[str, Any]:
    """Return the encoder-specific half of the causality record for the recurrent encoder.

    Split out of :func:`causality_disclosure` because it is the one part of that record that is a
    property of the *encoder* rather than of the feature bank: the refusal sentence, the channel
    reaches, the source delay and the horizon are true of any model reading this dataset, while
    ``causal_norm`` is a key one architecture has and another has no equivalent for. Each model
    discloses what is true of it, and a shared key that means nothing in one of them would be
    worse than two honest blocks.

    Takes ``model: Any`` and imports no model class, so the module stays free of the network it
    describes.

    Args:
        model: The rebuilt net, which is what was trained and is therefore the authority on the
            guard it actually carries.

    Returns:
        The encoder's own keys, merged into the causality record in this order.
    """
    record: Dict[str, Any] = {
        "causal_norm": bool(model.causal_norm),
        "n_causalized_norms": int(model.n_causalized_norms),
    }
    if not record["causal_norm"]:
        # A second, independent failure of the same interpretation, and the more serious one: the
        # encoders' GroupNorm would pool across time, so the prior itself conditions on the future.
        record["causal_norm_consequence"] = (
            "causal_norm=False: the encoders' GroupNorm pools statistics across the whole time "
            "axis, so every history state carries an image of its own future and the prior "
            "p(z_t | Y_<=t) conditions on Y_>t. The readout is not a coupling readout at all "
            "under this configuration."
        )
        logger.warning(record["causal_norm_consequence"])
    return record


def causality_disclosure(
    config: Mapping[str, Any],
    model: Any,
    encoder_disclosure: Callable[[Any], Dict[str, Any]] = rws_encoder_disclosure,
) -> Dict[str, Any]:
    """Record what this run's causal standing actually is, in the run's own artifacts.

    Unconditional, including the refusal sentence. A finite reach budget prunes channels on a
    $95\\%$-energy quantile and delays the survivors; it narrows the leak, it does not close it,
    and no budget currently trains. So the statement stands at every budget and the budget's
    effect is recorded beside it rather than in place of it.

    Args:
        config: The merged run config, for the configured budget.
        model: The rebuilt net, which is what was trained and is therefore the authority on the
            guard it actually carries.
        encoder_disclosure: Returns the encoder-specific keys, merged in at the position they
            occupy in the record. Defaults to this model's, so a caller with one model to
            evaluate says nothing extra; a run of another architecture passes its own through
            the binding.

    Returns:
        The disclosure record, promoted into ``summary.json`` as well as written to
        ``preflight.json``.
    """
    horizon_seconds = float(model.horizon) * SECONDS_PER_STEP
    delay_steps = int(model.source_delay_steps)
    per_block = channels_reading_past_the_horizon(horizon_seconds)
    # The encoder's own block, whatever it turns out to be for this architecture. Checked against
    # the keys the shared record owns before it is merged: the splat sits mid-literal, so a reused
    # name would either replace a shared key -- including ``statement``, the refusal sentence this
    # function documents as unconditional -- or be dropped by a key below it, and both would be
    # silent in an artifact whose whole purpose is to be read literally.
    disclosed = encoder_disclosure(model)
    reserved = sorted(set(disclosed) & SHARED_CAUSALITY_KEYS)
    if reserved:
        raise ValueError(
            f"the encoder disclosure returned key(s) the shared causality record already owns: "
            f"{reserved}. An encoder discloses what is true of *it*; the shared half is not "
            f"overridable, because a reader compares two models' records down these key names."
        )
    return {
        "not_causal": True,
        "statement": NOT_CAUSAL_STATEMENT,
        # The number the statement points at, recomputed from the bank rather than restated.
        "max_channel_reach_s": max(
            (entry["max_reach_s"] for entry in per_block.values()), default=0.0
        ),
        "causal_reach_budget_s": _vae_config(config).get("causal_reach_budget_s"),
        **disclosed,
        "source_delay_steps": delay_steps,
        "source_delay_seconds": delay_steps * SECONDS_PER_STEP,
        # Per-channel delays have no single representative, so the maximum is used and every lag
        # computed from it is an upper bound. Recorded beside the number so the choice travels.
        "source_delay_is_max_over_channels": True,
        "horizon_seconds": horizon_seconds,
        "channels_reading_past_the_horizon": per_block,
    }


# =============================================================================
# The run
# =============================================================================
def run_preflight(
    *,
    config: Mapping[str, Any],
    model: Any,
    checkpoint_path: Any,
    model_kwargs: Mapping[str, Any],
    hyper_parameters: Mapping[str, Any],
    binding: ModelBinding,
) -> Dict[str, Any]:
    """Run every guard, then record the run's causal standing.

    Order is deliberate. The placeholder check is first so its message is not pre-empted by a
    missing-file error someone would then go looking for; the shard, statistics, trim,
    normalisation and field guards follow, because each is a config read; the width comparison
    and the reconciliation come next, because each needs the rebuilt model; and the weight-space
    load verification is last, because it is the check whose failure means nothing else mattered.

    Args:
        config: The merged run config.
        model: The rebuilt net, after the checkpoint load.
        checkpoint_path: The checkpoint being evaluated, recorded in the output.
        model_kwargs: The checkpoint's own constructor kwargs.
        hyper_parameters: The checkpoint's own task hyperparameters.
        binding: The binding of the model being evaluated, for the constructor keys the
            reconciliation compares and the encoder half of the causality record. Passed in
            rather than imported, so this module still names no model class.

    Returns:
        The preflight record, ready for :func:`write_preflight`.

    Raises:
        EvalPreconditionUnmet: On any failed guard, naming the fix.
    """
    check_repointed(config)
    check_test_shards_exist(config)
    _reuse_trainer_guard(_check_stat_path, config, "stat_path")
    check_trim_minutes(config)
    _reuse_trainer_guard(_check_raw_target_normalized, config, "raw_target_normalized")
    check_load_fields(config)
    _reuse_trainer_guard(_check_causal_budget_resolves, config, "causal_budget_resolves")
    _reuse_trainer_guard(
        _check_declared_widths_against_shard,
        config_view_for_shard_guards(config, model),
        "declared_widths",
    )

    reconciliation = reconcile_with_checkpoint(
        config,
        model_kwargs=model_kwargs,
        hyper_parameters=hyper_parameters,
        geometry_keys=binding.geometry_keys,
    )
    load_check = verify_weights_loaded(model)

    logger.info(
        "preflight passed: the shards, the statistics path, the trimmed grid, the loaded fields "
        "and the declared widths all hold, the config agrees with the checkpoint, and the "
        "zero-initialised tensors carry loaded weights"
    )
    return {
        "checkpoint": str(checkpoint_path),
        "dataset_paths": _test_shards(config),
        "checks": {
            "repoint_placeholder": {"passed": True},
            "test_shards_exist": {"passed": True, "n_shards": len(_test_shards(config))},
            "stat_path": {"passed": True, "path": _dataset_config(config).get("stat_path")},
            "trim_minutes": {"passed": True, "value": REQUIRED_TRIM_MINUTES},
            "raw_target_normalized": {"passed": True},
            "load_fields": {"passed": True, "required": list(REQUIRED_EVAL_LOAD_FIELDS)},
            "causal_budget_resolves": {"passed": True},
            "declared_widths": {
                "passed": True,
                "compared_against": "the model's own c_y / c_u, from the checkpoint's model_kwargs",
                "shards": _test_shards(config)[:1],
            },
            "config_matches_checkpoint": reconciliation,
            "weights_loaded": load_check,
        },
        "causality": causality_disclosure(config, model, binding.encoder_disclosure),
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
