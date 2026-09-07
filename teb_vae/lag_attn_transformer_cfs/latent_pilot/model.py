r"""Strict checkpoint reconstruction, the classifier wrapper, and the gradient allowlist.

Written in LP-05: the loading path that is reused, and the exact freeze set the adaptation is
defined by.

Reconstruction
--------------

``teb_vae.lag_attn_transformer_cfs.eval.binding.TRF_CFS_BINDING`` already names this model's
``model_cls`` (``SeqVaeLagAttnTrfCfs``), its ``task_cls`` (``SeqVaeLagAttnTrfCfsTask``), the
constructor keys reconciled against a checkpoint and the committed evaluation override delta. The
pilot rebuilds through that binding and through ``teb_vae.lag_attn_cfs.eval.probe.load_task``
(``checkpoint_path, device, blob=, binding=``), which is strict in the way that matters:
``train.graph_models_utils.load_checkpoint_strict`` returns ``None`` rather than raising when no
module matches, so the caller must check it -- an unchecked call evaluates randomly initialised
weights and reports the result as a measurement. ``check_model_class`` runs *before* construction,
because the constructor is keyword-only and another architecture's ``model_kwargs`` would otherwise
surface as a ``TypeError`` naming a parameter instead of a class.

The checkpoint is the authority on everything it stamps: ``model_kwargs`` (architecture and the
whole channel/clock/trim contract), ``hyper_parameters`` (the objective the run trained under),
``model_class`` and ``epoch``. The resolved config written beside it supplies the dataset and
statistics contract. Neither is rebuilt from today's package defaults, and the pilot never writes to
either -- the source checkpoint is opened read-only and its digest goes into the run's protocol
record.

Trainable set
-------------

Exactly two things receive gradients:

1. ``model.posterior_head.delta_mu_head``. Head-structured -- the shipped geometry -- it is an
   ``nn.ModuleList`` of ``nn.Linear(fuse_out, d_z // num_heads)`` with
   ``fuse_out = max(2 * (d_z // num_heads), 16)``; flat, it is a single ``nn.Linear(d_model, d_z)``.
   All of its per-head modules train together. The parameter count is **derived from the loaded
   checkpoint and logged**, never assumed: at $d_z = 64$ over four heads it is
   $4 \times (32 \times 16 + 16) = 2112$, but a checkpoint at another geometry gives another number
   and a hard-coded one would misreport the size of the change.
2. A new ``Linear(d_z, 1)`` classifier on the pooled, standardized posterior mean -- 65 parameters at
   $d_z = 64$.

Everything else is frozen: both encoders, the input adapters, the prior head, the lag attention and
its query projection, the posterior fusion and norms, every variance head, and the decoder. The
optimizer receives an **explicit allowlist**, not ``model.parameters()``, so a module that acquires a
parameter later cannot silently join the fit.

Updating ``delta_mu_head`` is what makes this an adaptation of the representation rather than a
readout on top of it: $\mu^q = \mu^p + \Delta\mu$, so these are the coordinates the decoder actually
consumes. Fitting a classifier alone, or a separate projection with all of ``mu_post`` frozen, would
not change the model's latent at all. Equally, this narrow update reaches no attention pattern and no
history encoder, so a negative result bounds this adaptation and not the architecture.

Invariants the adaptation must leave standing, checked in deterministic evaluation against the keys
``CausalWarmupInputs.forward`` returns: ``mu_prior``, ``logvar_prior``, ``logvar_post`` and
``attn_weights`` are unchanged before and after; ``mu_post``, the source-conditioned KL and the
full-branch forecasts may move. A changed KL is a changed number, not evidence of changed
physiological coupling.

Mode and gradients
------------------

The backbone stays in ``eval()`` for the whole head-only fit, which disables dropout -- including the
posterior head's own ``a_dropout`` on the attended source summary -- and makes the frozen forward
deterministic. ``eval()`` does not disable autograd, so the trainable heads still receive gradients
in that mode; the student's heads are never placed under ``no_grad()``. The teacher is the same
architecture at the original weights with its outputs detached.

The optional frozen-fusion cache is a speed optimisation and nothing else: it is valid only with
every upstream parameter fixed and dropout disabled, it must be shown to reproduce the full forward's
means before it is allowed to back a fit, and the learned weights are inserted into the real model
for the final evaluation. Caching is not a different representation, and if the equivalence check is
not implemented the cache is omitted entirely.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from loguru import logger
from torch import nn

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.eval.collect import file_digest
from teb_vae.lag_attn_cfs.eval.probe import (
    load_task,
    read_checkpoint,
    resolve_device,
    resolved_config_for,
)
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: The one module inside the net this pilot trains. Head-structured -- the shipped geometry -- it is
#: an ``nn.ModuleList`` of ``Linear(fuse_out, d_z // num_heads)``; flat, a single
#: ``Linear(d_model, d_z)``. Both are handled, and the parameter count is derived at runtime rather
#: than written down, because a checkpoint at another geometry has a different one.
MEAN_HEAD_PATH = "posterior_head.delta_mu_head"

#: Forward outputs that must be **identical** before and after the adaptation, in deterministic
#: evaluation. Each is produced by a path this pilot freezes: the prior head, the prior's own
#: log-variance, the posterior log-variance (whose head reads the frozen fusion and the frozen
#: prior, never ``delta_mu_head``), and the lag attention's weights.
INVARIANT_OUTPUTS: Tuple[str, ...] = (
    "mu_prior",
    "logvar_prior",
    "logvar_post",
    "attn_weights",
)

#: Forward outputs the adaptation is allowed to move. ``mu_post`` is the point of the exercise;
#: the two full-branch decoder outputs and the source-conditioned KL move because they read it.
#: A changed KL is a changed number and not evidence of changed physiological coupling.
ADAPTED_OUTPUTS: Tuple[str, ...] = ("mu_post", "mu_full", "logvar_full", "kld_per_t")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A rebuilt checkpoint and everything needed to say which one it is.

    Attributes:
        task: The task wrapping the net. The **task** rather than the net alone, because the batch
            reaches the five-argument forward through its own input seam and a second
            implementation of that seam is exactly the drift this pilot must not introduce.
        model: ``task.orig_model``, bound for convenience.
        blob: The checkpoint's own dictionary, read once.
        config: The resolved configuration written beside it, which is the authority on the
            dataset, statistics and preprocessing contract.
        config_path: Where that configuration was found.
        checkpoint_path: The checkpoint itself.
        digest: Its content digest, for the run's protocol record.
        device: Where the model was placed.
        geometry: The derived geometry facts, logged and recorded rather than assumed.
    """

    task: Any
    model: Any
    blob: Dict[str, Any]
    config: Dict[str, Any]
    config_path: Path
    checkpoint_path: Path
    digest: str
    device: torch.device
    geometry: Dict[str, Any]


def geometry_record(model: Any, blob: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the geometry this checkpoint actually carries.

    Declared values come from the checkpoint's own ``model_kwargs`` and resolved ones from the
    rebuilt net, because those are the two places they are true. Nothing here is defaulted from
    this package: a promoted representation and an older ratio-power one disagree about the channel
    contract, and a constant written here would describe whichever configuration happened to ship
    when it was typed.

    Args:
        model: The rebuilt net.
        blob: The checkpoint dictionary.
        config: The resolved configuration beside it.

    Returns:
        The facts every later stage derives its windows, masks and reductions from.
    """
    kwargs = dict(blob.get("model_kwargs") or {})
    dataset = dict(config.get("dataset_config") or {})
    loader = dict(dataset.get("dataloader_config") or {})
    dataset_kwargs = dict(loader.get("dataset_kwargs") or {})
    shift = getattr(model, "target_forecast_shift", None)
    return {
        "model_class": blob.get("model_class"),
        "train_epoch": blob.get("epoch"),
        # Declared by the run that trained it.
        "d_z": kwargs.get("d_z"),
        "num_heads": kwargs.get("num_heads"),
        "d_head": kwargs.get("d_head"),
        "d_model": kwargs.get("d_model"),
        "c_y": kwargs.get("c_y"),
        "c_u": kwargs.get("c_u"),
        "use_up_st": kwargs.get("use_up_st"),
        "max_lag": kwargs.get("max_lag"),
        "lag_kv_source": kwargs.get("lag_kv_source"),
        "persistence_residual": kwargs.get("persistence_residual"),
        "delta_mu_scale": kwargs.get("delta_mu_scale"),
        "mu_scale": kwargs.get("mu_scale"),
        # Resolved by the net that was built.
        "sequence_length": int(model.geometry.t),
        "horizon": int(model.geometry.horizon),
        "warmup": int(model.geometry.warmup),
        "anchor_stride": int(getattr(model, "anchor_stride", 1)),
        "coverage_floor": float(getattr(model, "coverage_floor", 0.0)),
        "target_forecast_shift": None if shift is None else [int(value) for value in shift],
        # The loader contract the run trained under. ``trim_minutes`` places every channel's
        # validity boundary and every anchor timestamp, so it travels with the geometry.
        "trim_minutes": dataset_kwargs.get("trim_minutes"),
        "checkpoint_stat_path": dataset.get("stat_path"),
    }


def statistics_record(
    statistics_path: Any, *, trim_minutes: Optional[float], checkpoint_stat_path: Optional[str]
) -> Dict[str, Any]:
    """Check the statistics file against the loader contract, and record its provenance.

    The statistics decide the scale every input arrives on. On this target domain they do more
    than that: they are accumulated **excluding** each channel's warm-up region, so zero is the
    channel mean over the region the model reads -- which is what makes the feature-space
    baselines meaningful at all. A file built at a different trim, or from different shards,
    breaks that silently: every shape is right and every number is wrong.

    The loader only *warns* on a trim mismatch. This refuses, because a warning inside a
    multi-hour extraction is a warning nobody reads.

    Args:
        statistics_path: The statistics file this run will use.
        trim_minutes: The loader trim the checkpoint was trained under.
        checkpoint_stat_path: The statistics path recorded in the checkpoint's own configuration.

    Returns:
        The provenance record: path, digest, the file's own trim, and whether it is the same file
        the checkpoint trained under. Repointing is legitimate -- a fold's statistics are not the
        pretraining split's -- so a difference is **disclosed**, not refused.

    Raises:
        FileNotFoundError: If the file is not there.
        PilotConfigError: If it was built at a different trim from the one the loader will run at.
    """
    import h5py

    path = Path(statistics_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"statistics file {str(path)!r} does not exist. It must be the file generated from the "
            f"shards this run reads, at the checkpoint's own trim_minutes."
        )
    with h5py.File(str(path), "r") as handle:
        stored = handle.attrs.get("trim_minutes", None)
    stored_trim = None if stored is None else float(stored)
    if (
        stored_trim is not None
        and trim_minutes is not None
        and abs(stored_trim - float(trim_minutes)) > 1e-9
    ):
        raise PilotConfigError(
            f"statistics {str(path)!r} were accumulated at trim_minutes={stored_trim} but the "
            f"loader runs at {float(trim_minutes)}. The trim rebases every channel's validity "
            f"boundary, so the normalisation would be computed over a different region than the "
            f"model reads -- with every shape correct and every number wrong."
        )
    same_file = (
        checkpoint_stat_path is not None
        and Path(checkpoint_stat_path).resolve() == path.resolve()
    )
    return {
        "path": str(path),
        "digest": file_digest(path),
        "trim_minutes": stored_trim,
        "trim_minutes_recorded": stored_trim is not None,
        "same_as_checkpoint": bool(same_file),
        "checkpoint_stat_path": checkpoint_stat_path,
        "note": (
            "the statistics file the checkpoint recorded"
            if same_file else
            "repointed away from the checkpoint's own statistics; legitimate for a fold split, and "
            "disclosed here because the fitting population of these constants is part of what any "
            "held-out claim rests on"
        ),
    }


def load_pilot_checkpoint(
    checkpoint_path: Any,
    *,
    device: Optional[str] = None,
    binding: Any = TRF_CFS_BINDING,
) -> LoadedCheckpoint:
    """Rebuild the pretrained model strictly, from its own stamps.

    Every refusal on this path is the evaluation pipeline's, reused rather than restated: the class
    guard runs before construction, the architecture comes from the checkpoint's ``model_kwargs``
    and the objective from its ``hyper_parameters``, and ``load_checkpoint_strict`` -- which returns
    ``None`` instead of raising -- is checked, because an unchecked call evaluates randomly
    initialised weights and reports the result as a measurement.

    The source checkpoint is opened read-only and is never written to. Its digest goes into the
    run's protocol so a later reader can say which file produced these numbers.

    Args:
        checkpoint_path: The pretrained checkpoint.
        device: Torch device string, or ``None`` to choose cuda:0 when available.
        binding: The model to rebuild. This package's by default; an override is for a test that
            means to load something else through this path.

    Returns:
        The loaded bundle, in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint or its resolved configuration is missing.
        RuntimeError: If the checkpoint carries no ``model_kwargs`` or ``hyper_parameters``, or if
            its state dict does not align into the rebuilt model.
    """
    path = Path(checkpoint_path)
    resolved_device = resolve_device(device)
    config_path = resolved_config_for(path)
    config = load_config(str(config_path))
    blob = read_checkpoint(path)
    task = load_task(path, resolved_device, blob=blob, binding=binding)
    model = task.orig_model
    geometry = geometry_record(model, blob, config)

    logger.info(
        f"loaded {geometry['model_class']} from {path} (epoch {geometry['train_epoch']}): "
        f"d_z={geometry['d_z']} horizon={geometry['horizon']} T={geometry['sequence_length']} "
        f"trim_minutes={geometry['trim_minutes']} coverage_floor={geometry['coverage_floor']} "
        f"forecast_shift={'stored clock' if not geometry['target_forecast_shift'] else 'shifted'}"
    )
    return LoadedCheckpoint(
        task=task,
        model=model,
        blob=blob,
        config=config,
        config_path=Path(config_path),
        checkpoint_path=path,
        digest=file_digest(path),
        device=resolved_device,
        geometry=geometry,
    )


# =============================================================================
# The classifier
# =============================================================================
class LatentClassifier(nn.Module):
    r"""One linear logit on the standardized pooled posterior mean.

    $$\ell_i = w^\top S(v_i) + b, \qquad S(x) = (x - m) / s.$$

    The scaler travels **inside** the module as buffers rather than beside it, for one reason: it
    must be impossible to score a recording under a scaler other than the one the classifier was
    fitted with. Saving the classifier saves the constants; loading it restores them; refitting
    them would require constructing a new object, which is visible in a way that a mutated
    attribute is not.

    The standardizing constants are fitted once on pretrained **training** anchors and then frozen
    for every comparison -- the frozen model, the adapted one, the control, and every split.

    Args:
        d_z: The latent width.
        center: Per-coordinate means $m$. Defaults to zeros, which is the identity.
        scale: Per-coordinate scales $s$, already floored by the fitting code. Defaults to ones.

    Raises:
        ValueError: If a supplied constant has the wrong width, or if any scale is not positive --
            a zero would divide the run into infinities and report them as a latent.
    """

    def __init__(
        self,
        d_z: int,
        *,
        center: Optional[Any] = None,
        scale: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.d_z = int(d_z)
        self.linear = nn.Linear(self.d_z, 1)
        center_tensor = (
            torch.zeros(self.d_z) if center is None
            else torch.as_tensor(center, dtype=torch.float32).reshape(-1)
        )
        scale_tensor = (
            torch.ones(self.d_z) if scale is None
            else torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        )
        for name, tensor in (("center", center_tensor), ("scale", scale_tensor)):
            if tensor.numel() != self.d_z:
                raise ValueError(
                    f"{name} has {tensor.numel()} entries but the latent is {self.d_z}-dimensional."
                )
        if not bool(torch.all(scale_tensor > 0)):
            raise ValueError(
                "every scale must be positive; the fitting code floors them and reports latent "
                "collapse rather than emitting a zero."
            )
        self.register_buffer("center", center_tensor)
        self.register_buffer("scale", scale_tensor)

    def standardize(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the frozen scaler.

        Args:
            vectors: Pooled latent vectors ``(N, d_z)``.

        Returns:
            The standardized vectors, same shape.
        """
        return (vectors - self.center) / self.scale

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Return one logit per recording.

        Args:
            vectors: Pooled latent vectors ``(N, d_z)``, **unstandardized** -- the scaler is applied
                here so no call site can forget it.

        Returns:
            Logits ``(N,)``. Logits, not probabilities: the fit is class-balanced, so the sigmoid
            of this is an association score and not a calibrated risk.
        """
        return self.linear(self.standardize(vectors)).squeeze(-1)


# =============================================================================
# The gradient allowlist
# =============================================================================
def mean_head_module(model: Any) -> nn.Module:
    """Return the posterior mean-output module this pilot trains.

    Args:
        model: The rebuilt net.

    Returns:
        The ``delta_mu_head``: a ``ModuleList`` of per-head linears under the head-structured
        posterior, or a single linear otherwise.

    Raises:
        PilotConfigError: If the checkpoint has no such module. A pilot that silently trained
            nothing would report the frozen model's numbers as the adapted model's.
    """
    node: Any = model
    for part in MEAN_HEAD_PATH.split("."):
        node = getattr(node, part, None)
        if node is None:
            raise PilotConfigError(
                f"the rebuilt model has no {MEAN_HEAD_PATH!r}, so there is nothing for this pilot "
                f"to adapt. Fitting only a classifier would leave every latent coordinate exactly "
                f"as pretrained, which is not the experiment."
            )
    return node


def mean_head_parameters(model: Any) -> List[Tuple[str, nn.Parameter]]:
    """The named parameters of the mean-output module, in a stable order.

    Args:
        model: The rebuilt net.

    Returns:
        ``(qualified name, parameter)`` pairs. Qualified from the model root, so a name here is a
        key of the model's own state dict and the allowlist can be checked against it.
    """
    module = mean_head_module(model)
    return [
        (f"{MEAN_HEAD_PATH}.{name}", parameter)
        for name, parameter in module.named_parameters()
    ]


def freeze_for_pilot(model: Any) -> Dict[str, Any]:
    """Freeze everything except the posterior mean output, and put the net in evaluation mode.

    Two separate things, and both are needed:

    * **Gradients.** Every parameter is switched off and the mean-output parameters back on, so the
      optimizer's allowlist and the autograd graph agree by construction. The encoders, the input
      adapters, the prior head, the lag attention and its query projection, the posterior fusion and
      norms, every variance head and the decoder all stay exactly as pretrained.
    * **Mode.** ``eval()`` disables dropout -- including the posterior head's own dropout on the
      attended source summary -- which is what makes the frozen forward deterministic and the
      teacher comparison meaningful. It does **not** disable autograd, so the mean heads still
      receive gradients in this mode; they are never placed under ``no_grad``.

    Args:
        model: The rebuilt net. Modified in place.

    Returns:
        The record: the trainable parameter names and their count, derived from this checkpoint
        rather than assumed, and the frozen count beside it.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = mean_head_parameters(model)
    for _name, parameter in trainable:
        parameter.requires_grad_(True)
    model.eval()

    n_trainable = sum(parameter.numel() for _name, parameter in trainable)
    n_total = sum(parameter.numel() for parameter in model.parameters())
    record = {
        "trainable_names": [name for name, _parameter in trainable],
        "n_trainable_parameters": int(n_trainable),
        "n_frozen_parameters": int(n_total - n_trainable),
        "n_model_parameters": int(n_total),
        "training_mode": bool(model.training),
    }
    logger.info(
        f"pilot freeze: {n_trainable} trainable parameter(s) in "
        f"{len(record['trainable_names'])} tensor(s) under {MEAN_HEAD_PATH}, "
        f"{record['n_frozen_parameters']} frozen; model.training={model.training}"
    )
    return record


def check_pilot_mode(model: Any) -> None:
    """Refuse a model that has drifted out of the pilot's freeze.

    Cheap enough to call before every optimizer step, and it catches the two failures that produce
    a plausible-looking wrong result rather than an exception: a stray ``train()`` puts dropout back
    on -- changing both the student's forward and its comparison against the teacher -- and a
    parameter that regained ``requires_grad`` widens the adaptation past what the run reports.

    Args:
        model: The net.

    Raises:
        PilotConfigError: Naming the offending parameters, or the mode.
    """
    if model.training:
        raise PilotConfigError(
            "the backbone is in training mode during a head-only fit, so dropout is active: the "
            "student's forward and the teacher's are no longer comparable, and neither is "
            "reproducible. Call freeze_for_pilot before fitting."
        )
    allowed = {name for name, _parameter in mean_head_parameters(model)}
    leaked = sorted(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in allowed
    )
    if leaked:
        raise PilotConfigError(
            f"{len(leaked)} parameter(s) outside {MEAN_HEAD_PATH} require gradients, e.g. "
            f"{leaked[:5]}. The adaptation this run reports is the mean-output layers alone."
        )
    missing = sorted(
        name for name, parameter in mean_head_parameters(model) if not parameter.requires_grad
    )
    if missing:
        raise PilotConfigError(
            f"the mean-output parameter(s) {missing} do not require gradients, so this fit would "
            f"train the classifier alone and leave every latent coordinate as pretrained."
        )


def parameter_groups(
    model: Any,
    classifier: nn.Module,
    *,
    mean_head_lr: float,
    classifier_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Build the optimizer's parameter groups as an explicit allowlist.

    An allowlist rather than ``model.parameters()`` filtered by ``requires_grad``: the two agree
    today, and a module that acquires a parameter later would silently join the fit under the
    filter and could not under this.

    Args:
        model: The frozen net.
        classifier: The linear classifier.
        mean_head_lr: Learning rate for the posterior mean-output layers.
        classifier_lr: Learning rate for the classifier.
        weight_decay: Applied to both groups.

    Returns:
        Two groups, named so the optimizer state and the logs say which is which.
    """
    return [
        {
            "name": "mean_head",
            "params": [parameter for _name, parameter in mean_head_parameters(model)],
            "lr": float(mean_head_lr),
            "weight_decay": float(weight_decay),
        },
        {
            "name": "classifier",
            # The buffers -- the frozen scaler -- are deliberately not parameters and so cannot
            # appear here: standardizing constants that drifted during a fit would rescale the very
            # movement the comparison measures.
            "params": [parameter for parameter in classifier.parameters()],
            "lr": float(classifier_lr),
            "weight_decay": float(weight_decay),
        },
    ]


def describe_trainable(model: Any, classifier: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Report exactly what this run will update, with counts derived from the loaded checkpoint.

    Args:
        model: The net.
        classifier: The classifier, if one exists yet.

    Returns:
        Per-group names and parameter counts, and their total.
    """
    mean_head = mean_head_parameters(model)
    record: Dict[str, Any] = {
        "mean_head": {
            "path": MEAN_HEAD_PATH,
            "names": [name for name, _parameter in mean_head],
            "n_parameters": int(sum(p.numel() for _name, p in mean_head)),
        }
    }
    if classifier is not None:
        record["classifier"] = {
            "names": [name for name, _parameter in classifier.named_parameters()],
            "n_parameters": int(sum(p.numel() for p in classifier.parameters())),
            "n_buffer_values": int(sum(b.numel() for b in classifier.buffers())),
        }
    record["n_trainable_parameters"] = int(
        record["mean_head"]["n_parameters"]
        + record.get("classifier", {}).get("n_parameters", 0)
    )
    return record


def frozen_teacher(model: Any) -> Any:
    """A detached copy of the model at its current weights.

    The teacher of the preservation term. A copy rather than a second load from disk, so it is the
    same weights by construction; frozen and in evaluation mode, so it contributes no gradient and
    no dropout noise to the target the student is held to.

    Args:
        model: The net, before any update.

    Returns:
        The teacher. Sharing nothing with ``model``: a later update to the student cannot move the
        target it is being compared against.
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


# =============================================================================
# Forward inputs and the invariant check
# =============================================================================
def forward_inputs(task: Any, batch: Any) -> Tuple[Any, ...]:
    """Assemble the five positional arguments the net's forward takes.

    Through the task's own seam rather than a second implementation: the two target blocks are
    width-checked against $c_y$ there, the source stream is assembled according to ``use_up_st``,
    and the anchor geometry is resolved by the same code a real run uses. The pilot always reads
    the **dense** anchor set, which is what that seam returns outside training, and this asserts it
    rather than assuming it -- an extraction at the training tiling would silently cover a fifth of
    the anchors.

    Args:
        task: The loaded task.
        batch: A batch from the data module.

    Returns:
        ``(y_st, y_ph, u_stream, anchor_phase, anchor_stride)``.

    Raises:
        PilotConfigError: If the resolved geometry is not the dense one.
    """
    inputs = task._build_forward_inputs(batch)
    phase, stride = inputs[3], inputs[4]
    phase_value = int(phase) if not torch.is_tensor(phase) else int(phase.reshape(-1)[0])
    if int(stride) != 1 or phase_value != 0:
        raise PilotConfigError(
            f"the task resolved anchor geometry (phase={phase_value}, stride={int(stride)}) rather "
            f"than the dense (0, 1) this pilot extracts at. A tiled anchor set would cover a "
            f"fraction of the timeline and the trajectory would be drawn over the gaps."
        )
    return inputs


def to_device(task: Any, batch: Any) -> Any:
    """Move one batch to the model's device, through the task's own transfer hook.

    Lightning moves batches for a run inside a ``Trainer``; every pass in this package runs its own
    loop, so each of them has to do it, and each of them does it here. The hook rather than a local
    walk over the batch: it is the one the training step used, it knows which fields are tensors and
    which are collated lists of identifiers, and a second implementation would be free to disagree
    about a field neither of them names.

    A batch that is not a recognised collection -- the test suite's ``SimpleNamespace`` stubs --
    comes back unchanged, which is correct on the CPU those stubs run on.

    Args:
        task: The loaded task, which carries the device.
        batch: A batch from a dataloader, or a collated one.

    Returns:
        The batch on ``task.device``.
    """
    return task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)


def deterministic_outputs(
    task: Any,
    batch: Any,
    *,
    keys: Sequence[str] = INVARIANT_OUTPUTS,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Run one forward under fixed randomness and return the requested outputs.

    The forward draws $z$, so two calls differ unless the draw is pinned. The global RNG state is
    saved and restored around the seeding, because a diagnostic that silently advanced the training
    stream would make a run non-reproducible in a way nothing reports.

    Args:
        task: The loaded task.
        batch: A batch from the data module.
        keys: Which forward outputs to return.
        seed: The draw to pin.

    Returns:
        Detached clones of the requested outputs.
    """
    model = task.orig_model
    was_training = model.training
    model.eval()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        with torch.no_grad():
            outputs = model(*forward_inputs(task, batch))
    finally:
        torch.random.set_rng_state(state)
        if was_training:
            model.train()
    return {key: outputs[key].detach().clone() for key in keys if key in outputs}


def compare_outputs(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> Dict[str, float]:
    """Largest absolute difference per output.

    Args:
        before: Outputs from the pretrained model.
        after: Outputs from the adapted model, on the same batch and the same seed.

    Returns:
        Key -> maximum absolute difference. A key present on one side only is reported as ``nan``,
        which is a mismatch to investigate rather than a zero to pass.
    """
    keys = sorted(set(before) | set(after))
    differences: Dict[str, float] = {}
    for key in keys:
        if key not in before or key not in after:
            differences[key] = float("nan")
            continue
        differences[key] = float((before[key] - after[key]).abs().max().item())
    return differences


def assert_invariants(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    keys: Sequence[str] = INVARIANT_OUTPUTS,
    tolerance: float = 0.0,
) -> Dict[str, float]:
    """Refuse an adaptation that moved something it was supposed to freeze.

    Args:
        before: Pretrained outputs.
        after: Adapted outputs, same batch, same seed.
        keys: The outputs that must not have moved.
        tolerance: Maximum accepted absolute difference. Zero on CPU, where a frozen path is
            bit-identical; a small positive value is for GPU kernels whose reductions are not
            deterministic across calls.

    Returns:
        The per-key differences, for the record.

    Raises:
        PilotConfigError: Naming the outputs that moved and by how much.
    """
    differences = compare_outputs(
        {key: before[key] for key in keys if key in before},
        {key: after[key] for key in keys if key in after},
    )
    moved = {
        key: value for key, value in differences.items()
        if not (value <= float(tolerance))
    }
    if moved:
        raise PilotConfigError(
            f"outputs that this adaptation must leave untouched have changed: {moved}. Only "
            f"{MEAN_HEAD_PATH} is trained, so mu_prior, both latent log-variances and the "
            f"attention weights are identical by construction -- a difference here means the "
            f"freeze did not hold, and every comparison in this run is between two models that "
            f"differ in more than the mean output."
        )
    return differences
