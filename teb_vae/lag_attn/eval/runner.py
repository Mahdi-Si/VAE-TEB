r"""Checkpoint to model, batch to tensors, tensors to forward outputs.

Everything an analysis needs to *get at* the model lives here, and nothing that interprets
what comes back. Three things in this file are load-bearing and none of them is obvious from
a call site.

**The rebuild is self-describing, and the guard runs first.** A checkpoint written by
``SeqVaeLagAttnTask.on_save_checkpoint`` carries both ``model_class`` and ``model_kwargs``, so
the architecture rebuilds with no config file. :func:`train.graph_models_utils.check_model_class`
must run *before* construction, because the constructor is keyword-only with no ``**kwargs``:
a blob from a different model version would otherwise fail as a cryptic ``TypeError`` deep in
the constructor rather than as a message naming both classes.

**The objective comes from the checkpoint, not from the config.** ``compute_loss``'s behaviour
is set by nine arguments and *none of them is a constructor argument*, so none appears in
``model_kwargs``. They live in ``checkpoint["hyper_parameters"]``. A pipeline that rebuilt only
from ``model_kwargs`` and took the objective from its own YAML would silently score under
``compute_loss``'s defaults -- ``likelihood='mse'``, ``sigma_obs=1.0`` -- rather than the
shipped ``gaussian_nll`` and ``'learned'``, and every loss, the uplift and the
source-specificity ordering would be computed under an objective the model was never trained
with. See :class:`Objective`.

**Only declared tensor fields move to device.** ``guid`` and ``source_file_basename`` are
``list[str]`` after collation, so a blanket ``.to(device)`` crashes -- and the collectors read
them as Python values anyway.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import torch
from loguru import logger

from teb_vae.lag_attn.figure_primitives import future_target, to_numpy
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from train.graph_models_utils import check_model_class, load_checkpoint_strict

# Re-exported so the collectors and the analyses import their batch and tensor helpers from one
# place. The two lifted here are shared with ``plotting.py`` and therefore live in
# ``figure_primitives``; the rest are defined below.
__all__ = [
    "EvalRunner",
    "Objective",
    "batch_size_of",
    "field_names",
    "future_target",
    "get_field",
    "guid_of",
    "to_numpy",
]

#: Batch fields moved to the compute device. Everything else -- ``guid``,
#: ``source_file_basename``, ``source_file``, ``source_file_index`` -- stays where it is: the
#: first two are ``list[str]`` and a blanket transfer raises on them.
TENSOR_FIELDS: Tuple[str, ...] = (
    "fhr_st",
    "fhr_ph",
    "up_st",
    "up_ph",
    "weight",
    "target",
    "epoch",
    "fhr",
    "up",
)

#: Batches between progress lines inside one analysis. Small enough that a stalled run is
#: visible within a minute or two at production batch sizes.
PROGRESS_EVERY_N_BATCHES = 20

#: The nine settings that decide what ``compute_loss`` computes. Kept as one tuple because the
#: point of :class:`Objective` is that they travel together; resolving eight of nine from the
#: checkpoint and one from a config is the failure this exists to prevent.
OBJECTIVE_FIELDS: Tuple[str, ...] = (
    "likelihood",
    "sigma_obs",
    "free_bits",
    "detach_baseline_in_full",
    "lambda_full",
    "lambda_base",
    "lambda_lag",
    "beta_schedule",
    "kld_beta",
)

#: Objective field -> the ``model_config.VAE_model`` key that config-side sets it. Only
#: ``lambda_lag`` differs in name, and it differs in a way that is easy to miss: the config
#: calls it ``lag_smoothness_lambda`` and ``trainer.py`` does the translation.
_CONFIG_KEY_FOR_OBJECTIVE: Dict[str, str] = {
    "likelihood": "likelihood",
    "sigma_obs": "sigma_obs",
    "free_bits": "free_bits",
    "detach_baseline_in_full": "detach_baseline_in_full",
    "lambda_full": "lambda_full",
    "lambda_base": "lambda_base",
    "lambda_lag": "lag_smoothness_lambda",
    "beta_schedule": "beta_schedule",
    "kld_beta": "kld_beta",
}


def _values_agree(left: Any, right: Any) -> bool:
    """Compare two objective values, tolerating float representation.

    Both sides originate from the same YAML in the intended case, so exact equality would
    almost always hold -- but a value that made a round trip through a checkpoint should not
    be able to fail on a last-bit difference.

    Args:
        left: The checkpoint's value.
        right: The config's value.

    Returns:
        Whether the two describe the same setting.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


@dataclass(frozen=True)
class Objective:
    r"""The nine ``compute_loss`` settings a run was trained under.

    Resolved from ``checkpoint["hyper_parameters"]``, which the task's ``save_hyperparameters``
    writes for exactly this purpose: a run's objective is recoverable from its checkpoint
    alone. This is the single source for every ``compute_loss`` call in the pipeline.
    """

    likelihood: str
    sigma_obs: Any
    free_bits: float
    detach_baseline_in_full: bool
    lambda_full: float
    lambda_base: float
    lambda_lag: float
    beta_schedule: Optional[Dict[str, Any]]
    kld_beta: float
    #: The training epoch the checkpoint was written at, read from the blob's own ``epoch`` key.
    #: Not one of the nine -- it is not an hparam and has no config key -- but the schedule in
    #: ``beta_schedule`` is a function of exactly this, so the effective $\beta$ is unknowable
    #: without it. Defaulted so a hand-built objective (the tests, and any caller stating the
    #: nine settings directly) stays constructible; ``None`` means "the blob carried no epoch",
    #: which :meth:`effective_beta` refuses to guess past rather than silently reading as $0$.
    train_epoch: Optional[int] = None

    @classmethod
    def from_checkpoint(cls, blob: Dict[str, Any], checkpoint_path: Any = "<checkpoint>") -> "Objective":
        """Read the objective out of a loaded checkpoint blob.

        Args:
            blob: The deserialised checkpoint.
            checkpoint_path: Path, for the error message only.

        Returns:
            The resolved objective.

        Raises:
            RuntimeError: If the blob carries no ``hyper_parameters``, or is missing one of the
                nine settings. Both are unrecoverable: falling back to ``compute_loss``'s
                defaults would score the run under an objective it was never trained with.
        """
        hparams = blob.get("hyper_parameters") if isinstance(blob, dict) else None
        if not isinstance(hparams, dict):
            raise RuntimeError(
                f"checkpoint {checkpoint_path!r} carries no 'hyper_parameters', so the "
                f"objective it was trained under is unknown. Every loss, the uplift and the "
                f"source-specificity ordering depend on it, and compute_loss's own defaults "
                f"(likelihood='mse', sigma_obs=1.0) are not what this model ships. Use a "
                f"checkpoint written by SeqVaeLagAttnTask, which saves them."
            )
        missing = [name for name in OBJECTIVE_FIELDS if name not in hparams]
        if missing:
            raise RuntimeError(
                f"checkpoint {checkpoint_path!r} is missing objective setting(s) "
                f"{missing} from its 'hyper_parameters'. Scoring under compute_loss's defaults "
                f"instead would silently produce plausible numbers under the wrong objective."
            )
        # Lightning writes the training epoch at the top level of every checkpoint it saves, and
        # that is the counter ``task.py::_resolve_beta`` reads as ``self.current_epoch``. Taken
        # from the blob rather than from the eval config on purpose: a config key would be a
        # second, unverifiable claim about which epoch this checkpoint came from, and the whole
        # point of this class is that the run is recoverable from the checkpoint alone.
        epoch = blob.get("epoch") if isinstance(blob, dict) else None
        train_epoch = int(epoch) if isinstance(epoch, (int, float)) and not isinstance(epoch, bool) else None
        return cls(
            **{name: hparams[name] for name in OBJECTIVE_FIELDS}, train_epoch=train_epoch
        )

    def effective_beta(self, epoch: Optional[int] = None) -> float:
        r"""The KL weight this checkpoint was actually trained under, at ``epoch``.

        An exact mirror of ``task.py::_resolve_beta``, which is the only definition that matters:
        training logs *its* result as ``kld_beta`` and multiplies ``kld_loss`` by it inside
        ``total_loss``, so a table that reported the configured constant instead would disagree
        with every training row by $(\beta_{\mathrm{eff}} - \beta_{\mathrm{cfg}})\,L_{KL}$. The
        shipped config makes that gap two orders of magnitude: ``kld_beta: 0.001`` is documented
        as the fallback for ``kind == constant``, while the ``linear_warmup`` schedule it actually
        ships reaches $0.1$ past epoch $50$.

        The supported kinds, matching ``_resolve_beta`` term for term:

        * no schedule (not a dict) -- the ``kld_beta`` constant;
        * ``constant`` -- ``beta_schedule.value`` when present, else ``kld_beta``;
        * ``linear_warmup`` -- ramped over the first ``warmup_epochs`` epochs, then held:

          $$\beta(e) = \mathrm{start} + (\mathrm{end} - \mathrm{start})
                       \min\!\left(1, \frac{e}{\mathrm{warmup\_epochs}}\right).$$

        An unknown kind **raises**, as it does in training. Falling back to the constant would
        report a $\beta$ the run never used, which is the same defect in a quieter form.

        Args:
            epoch: The epoch to resolve at. ``None`` uses :attr:`train_epoch`, the epoch the
                checkpoint records -- which is what a reconciliation with training wants. Note
                the checkpoint's counter and ``self.current_epoch`` can differ by one at an epoch
                boundary depending on the Lightning version; past ``warmup_epochs`` the schedule
                is flat and that difference is exactly zero, and inside the ramp it is one step of
                $(\mathrm{end} - \mathrm{start}) / \mathrm{warmup\_epochs}$.

        Returns:
            The scalar $\beta$ weighting ``kld_loss``.

        Raises:
            ValueError: If ``beta_schedule.kind`` is not a supported value.
            RuntimeError: If the schedule needs an epoch and none is available. Reading a missing
                epoch as $0$ would report the *start* of the ramp for a checkpoint that may be
                hundreds of epochs past its end.
        """
        resolved_epoch = self.train_epoch if epoch is None else epoch
        schedule = self.beta_schedule

        # ``_resolve_beta`` falls back to ``hparams.get('kld_beta', 0.01)``; here ``kld_beta`` is a
        # required field -- ``from_checkpoint`` refuses a blob without it -- so its own default is
        # unreachable and the two agree wherever the task can actually run.
        if not isinstance(schedule, dict):
            return float(self.kld_beta)

        kind = str(schedule.get("kind", "constant"))
        if kind == "constant":
            value = schedule.get("value")
            return float(value) if value is not None else float(self.kld_beta)
        if kind == "linear_warmup":
            if resolved_epoch is None:
                raise RuntimeError(
                    "beta_schedule.kind='linear_warmup' makes the KL weight a function of the "
                    "training epoch, but this objective carries no epoch: the checkpoint blob "
                    "had no top-level 'epoch' key and none was passed. Reading it as 0 would "
                    "report the start of the ramp "
                    f"({float(schedule.get('start', 1.0e-4))}) for a checkpoint that may be far "
                    f"past its end ({float(schedule.get('end', 0.1))}). Use a checkpoint written "
                    "by Lightning, or pass the epoch explicitly."
                )
            start = float(schedule.get("start", 1.0e-4))
            end = float(schedule.get("end", 0.1))
            warmup_epochs = int(schedule.get("warmup_epochs", 50))
            if warmup_epochs <= 0:
                return end
            fraction = min(1.0, max(0.0, float(resolved_epoch) / float(warmup_epochs)))
            return start + (end - start) * fraction
        raise ValueError(
            f"unknown beta_schedule.kind={kind!r}; expected 'constant' or 'linear_warmup'."
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the settings as a plain dict, for ``preflight.json``."""
        return {name: getattr(self, name) for name in OBJECTIVE_FIELDS}

    def loss_kwargs(self, *, beta: float = 0.0) -> Dict[str, Any]:
        r"""Keyword arguments for ``model.compute_loss``, under this objective.

        ``beta`` defaults to $0$ rather than to the trained schedule. Eval is not training:
        every term is reported separately, and a $\beta$-weighted total would be the one number
        that silently depends on which epoch the checkpoint happens to be from. A caller that
        wants the training-scale total passes it explicitly.

        Args:
            beta: KL weight applied to the returned ``total_loss``.

        Returns:
            Keyword arguments for ``compute_loss``, excluding ``forward_outputs``, ``y_st``,
            ``y_ph`` and ``weight``.
        """
        sigma_obs = self.sigma_obs if isinstance(self.sigma_obs, str) else float(self.sigma_obs)
        return {
            "beta": float(beta),
            "lambda_full": float(self.lambda_full),
            "lambda_base": float(self.lambda_base),
            "likelihood": str(self.likelihood),
            "sigma_obs": sigma_obs,
            "free_bits": float(self.free_bits),
            "detach_baseline_in_full": bool(self.detach_baseline_in_full),
            "lambda_lag": float(self.lambda_lag),
        }

    def reconcile_with_config(self, config: Dict[str, Any]) -> None:
        """Raise if the eval config claims an objective the checkpoint disagrees with.

        Only keys the config actually sets are compared. The config is not the authority here
        -- the checkpoint is -- but a disagreement means the operator believes something false
        about the run, and silently preferring either side would leave that belief in place.

        Args:
            config: The merged run config.

        Raises:
            ValueError: If any setting disagrees, naming both values.
        """
        vae_config = (config.get("model_config") or {}).get("VAE_model") or {}
        problems = []
        for name in OBJECTIVE_FIELDS:
            config_key = _CONFIG_KEY_FOR_OBJECTIVE[name]
            if config_key not in vae_config:
                continue
            configured, trained = vae_config[config_key], getattr(self, name)
            if not _values_agree(trained, configured):
                problems.append(
                    f"{name}: checkpoint={trained!r} but "
                    f"model_config.VAE_model.{config_key}={configured!r}"
                )
        if problems:
            raise ValueError(
                "the eval config's objective disagrees with the checkpoint's own "
                "hyper_parameters:\n  "
                + "\n  ".join(problems)
                + "\nThe checkpoint is authoritative -- it records what was actually trained -- "
                "so fix the config to match it, or point the run at the checkpoint this config "
                "describes. Scoring under the config's objective would produce plausible "
                "numbers for a model that never optimised it."
            )


@dataclass(frozen=True)
class ForecastView:
    r"""One batch's forecast tensors, sliced to the valid anchors and aligned with $Y^{+}$.

    Every field spans the same $(B,\ T - H_d,\ H_d,\ c_y)$ grid, so a metric can multiply any of
    them by ``mask`` without a further slice. ``n_scattering`` travels with them because the
    scattering / phase-harmonic split comes from the *batch* -- the model stores only the
    combined $c_y$ and cannot supply it.
    """

    mu_full: torch.Tensor
    mu_base: torch.Tensor
    delta_mu_src: torch.Tensor
    logvar_full: torch.Tensor
    logvar_base: torch.Tensor
    y_plus: torch.Tensor
    mask: torch.Tensor
    n_scattering: int
    outputs: Dict[str, torch.Tensor]


@dataclass
class EvalRunner:
    """A loaded model plus the geometry and dispatch every analysis shares.

    Built by :meth:`from_checkpoint`; the fields are the rebuilt model, where it runs, where
    its outputs go, and the geometry read *off the model* rather than off a config -- eval
    builds from ``model_kwargs``, so the model is the only thing that knows its own widths.
    """

    model: SeqVaeLagAttn
    device: torch.device
    output_dir: Path
    objective: Objective
    checkpoint_path: Path
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Any,
        output_dir: Any,
        device: Optional[Any] = None,
    ) -> "EvalRunner":
        """Rebuild a model from a self-describing checkpoint and wrap it in a runner.

        Args:
            checkpoint_path: Path to a checkpoint written by ``SeqVaeLagAttnTask``.
            output_dir: The run directory. Created if absent.
            device: Torch device or device string. ``None`` selects ``cuda:0`` when available,
                else CPU.

        Returns:
            A runner holding the loaded model in ``eval()`` mode on ``device``.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            ValueError: If the checkpoint's ``model_class`` names a different model.
            RuntimeError: If ``model_kwargs`` is absent or empty, if the weights do not align,
                or if the objective cannot be resolved.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        # Loaded once so the class guard, the objective and the weight load all see the same
        # blob; reading it three times would be three chances to read three different files.
        blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        # Before construction, not after: the constructor is keyword-only with no **kwargs, so
        # another model's model_kwargs fails as a cryptic TypeError rather than as a message
        # naming both classes.
        check_model_class(blob, SeqVaeLagAttn.__name__)

        model_kwargs = blob.get("model_kwargs") if isinstance(blob, dict) else None
        if not model_kwargs:
            # Silent otherwise: SeqVaeLagAttn() is legal and builds the full production
            # geometry, so an empty dict yields a 300-step, 128-wide model that then fails to
            # align with the checkpoint's weights for reasons that look like corruption.
            raise RuntimeError(
                f"checkpoint {str(checkpoint_path)!r} carries no 'model_kwargs', so the "
                f"architecture cannot be rebuilt. SeqVaeLagAttn() with no arguments would "
                f"build the production geometry rather than raise. Use a checkpoint written "
                f"by SeqVaeLagAttnTask with model_kwargs= supplied."
            )

        model = SeqVaeLagAttn(**model_kwargs)
        objective = Objective.from_checkpoint(blob, checkpoint_path=str(checkpoint_path))

        if load_checkpoint_strict(model=model, checkpoint=blob) is None:
            # Returns None rather than raising, so an unchecked call evaluates a randomly
            # initialised model and reports nothing. Every number in the run would be
            # meaningless and none of them would look wrong.
            raise RuntimeError(
                f"could not align checkpoint {str(checkpoint_path)!r} into SeqVaeLagAttn: no "
                f"discovered module matched its state dict after stripping the known wrapper "
                f"prefixes ('model.', '_orig_model.', '_orig_mod.', 'module.', 'net.', ...). "
                f"Evaluating would otherwise proceed on randomly initialised weights. Check "
                f"the checkpoint was written by this architecture at this geometry."
            )

        resolved_device = cls.resolve_device(device)
        model.to(resolved_device)
        model.eval()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"loaded SeqVaeLagAttn from {checkpoint_path} onto {resolved_device} "
            f"(d_model={model.d_model}, d_z={model.d_z}, T={model.sequence_length}, "
            f"H_d={model.horizon}, L={model.lag_attn.L})"
        )
        return cls(
            model=model,
            device=resolved_device,
            output_dir=output_dir,
            objective=objective,
            checkpoint_path=checkpoint_path,
            model_kwargs=dict(model_kwargs),
        )

    @staticmethod
    def resolve_device(device: Optional[Any]) -> torch.device:
        """Resolve a device argument, defaulting to ``cuda:0`` when CUDA is available.

        Args:
            device: A device, a device string, or ``None``.

        Returns:
            The resolved device.
        """
        if device is not None:
            return torch.device(device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Geometry, read off the model
    # ------------------------------------------------------------------
    @property
    def num_lags(self) -> int:
        r"""Width of the attention window, $L = \mathrm{max\_lag} + 1$."""
        return int(self.model.lag_attn.L)

    @property
    def num_heads(self) -> int:
        """Attention head count. Not stored on the model -- read from the attention module."""
        return int(self.model.lag_attn.num_heads)

    @property
    def d_head(self) -> int:
        """Per-head width. Not stored on the model -- read from the attention module."""
        return int(self.model.lag_attn.d_head)

    def geometry(self) -> Dict[str, Any]:
        """Return the geometry and the interpretation-relevant flags, for recording."""
        model = self.model
        return {
            "sequence_length": int(model.sequence_length),
            "d_model": int(model.d_model),
            "d_z": int(model.d_z),
            "horizon": int(model.horizon),
            "warmup_period": int(model.warmup_period),
            "c_y": int(model.c_y),
            "c_u": int(model.c_u),
            "use_up_st": bool(model.use_up_st),
            "max_lag": int(model.max_lag),
            "num_lags": self.num_lags,
            "num_heads": self.num_heads,
            "d_head": self.d_head,
            "head_structured_latent": bool(model.head_structured_latent),
            "kld_support": str(model.kld_support),
            "causal_norm": bool(model.causal_norm),
            "n_causalized_norms": int(model.n_causalized_norms),
            "frozen_attn_proj": bool(model.frozen_attn_proj),
            "mu_scale": float(model.mu_scale),
            "delta_mu_scale": float(model.delta_mu_scale),
            "delta_logvar_scale": float(model.delta_logvar_scale),
        }

    # ------------------------------------------------------------------
    # Inference mode
    # ------------------------------------------------------------------
    @contextmanager
    def inference_mode(self) -> Iterator["EvalRunner"]:
        """Enter ``no_grad`` **and** ``eval()``, restoring the prior training flag on exit.

        Both halves matter and they fail differently. Without ``no_grad`` a multi-hour run
        retains a graph per batch and runs out of memory. Without ``eval()`` dropout is live,
        so the attention rows a ``train()``-mode pass produces do not sum to $1$ and the
        ``te_lag_map`` identity $K_t = \\sum_\\ell K_t^{(\\ell)}$ silently stops holding.

        The restore is in a ``finally`` so an analysis that raises mid-batch cannot leave the
        model in a different mode for every step that follows.
        """
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                yield self
        finally:
            self.model.train(was_training)

    # ------------------------------------------------------------------
    # Batch dispatch
    # ------------------------------------------------------------------
    def to_device(self, batch: Any) -> Any:
        """Move the declared tensor fields of ``batch`` to the compute device, in place.

        Args:
            batch: A collated batch from the loader.

        Returns:
            The same batch object, so this composes in a generator expression.
        """
        for name in TENSOR_FIELDS:
            value = get_field(batch, name)
            if not isinstance(value, torch.Tensor):
                continue
            moved = value.to(self.device, non_blocking=False)
            if isinstance(batch, dict):
                batch[name] = moved
            else:
                setattr(batch, name, moved)
        return batch

    def iter_batches(
        self,
        loader: Any,
        max_samples: Optional[int] = None,
        *,
        log_every: int = PROGRESS_EVERY_N_BATCHES,
    ) -> Iterator[Any]:
        r"""Iterate a loader under :meth:`inference_mode`, moving each batch to the device.

        ``max_samples`` caps by **sample** count, not batch count, and does not split a batch:
        iteration stops once the running total reaches the cap, so the total actually yielded
        can overshoot by up to ``batch_size - 1``. Splitting instead would make the last batch
        a different size from every other, which changes nothing numerically -- the model has
        no BatchNorm -- but does make a per-batch record read as a truncation bug.

        Note this is a *prefix* cap, and the test loader is built ``shuffle=False`` over eight
        concatenated per-subgroup files: a prefix is one subgroup and one class. It is correct
        only for a smoke run or a single-file split. Analyses that cap for memory take a seeded
        subsample over the full index space instead.

        Args:
            loader: The dataloader to iterate.
            max_samples: Sample cap, or ``None`` for the whole split.
            log_every: Emit a progress line every this many batches. A forty-minute analysis
                that logs one line at its start and one at its end is indistinguishable from a
                hang for thirty-nine of them.

        Yields:
            Batches, with the declared tensor fields on the compute device.
        """
        seen = 0
        index = 0
        with self.inference_mode():
            for batch in loader:
                yield self.to_device(batch)
                index += 1
                seen += int(batch_size_of(batch))
                if log_every > 0 and index % log_every == 0:
                    logger.info(f"  ... {index} batches, {seen} samples")
                if max_samples is not None and seen >= int(max_samples):
                    logger.info(
                        f"max_samples={max_samples} reached after {seen} samples; stopping "
                        f"iteration"
                    )
                    break

    # ------------------------------------------------------------------
    # Batch -> model inputs
    #
    # Copied from SeqVaeLagAttnTask rather than imported: importing the Lightning task into an
    # eval script would drag Lightning, the config and the optimizer into a path that needs
    # none of them, to reuse about twenty lines. A test pins the copy against the task's own
    # behaviour on the same batch, so the two cannot silently diverge.
    # ------------------------------------------------------------------
    def build_source_stream(self, batch: Any) -> torch.Tensor:
        r"""Assemble the source stream $u$, $(B, T, c_u)$.

        Args:
            batch: A batch from the data module.

        Returns:
            ``[up_st, up_ph]`` concatenated when ``use_up_st``, else ``up_ph`` alone.

        Raises:
            RuntimeError: If a required field is absent, or the assembled width disagrees with
                the model's ``c_u``.
        """
        up_ph = get_field(batch, "up_ph")
        if up_ph is None:
            raise RuntimeError(
                "batch has no `up_ph` field. Add 'up_ph' to dataset_kwargs.load_fields in the "
                "config, and check the HDF5 files were built by the pipeline that writes up_ph "
                "as a first-class dataset."
            )
        if not bool(self.model.use_up_st):
            return self._checked_source(up_ph, up_st=None, up_ph=up_ph)
        up_st = get_field(batch, "up_st")
        if up_st is None:
            raise RuntimeError(
                "the model was built with use_up_st=True but the batch has no `up_st` field. "
                "Either add 'up_st' to dataset_kwargs.load_fields, rebuild the HDF5 with "
                "up_st, or evaluate a checkpoint built with use_up_st=false."
            )
        return self._checked_source(torch.cat([up_st, up_ph], dim=-1), up_st=up_st, up_ph=up_ph)

    def _checked_source(
        self, stream: torch.Tensor, *, up_st: Optional[torch.Tensor], up_ph: torch.Tensor
    ) -> torch.Tensor:
        r"""Return ``stream`` having checked its width against the model's $c_u$.

        Checked on every batch, not only the first: a multi-file test split can concatenate
        shards of different vintages, and the mismatch would then appear partway through a run.

        Args:
            stream: The assembled source stream.
            up_st: The scattering block, or ``None`` under the ``use_up_st=False`` ablation.
            up_ph: The phase-harmonic block.

        Returns:
            ``stream`` unchanged.

        Raises:
            RuntimeError: If the stream's width disagrees with the model's ``c_u``.
        """
        expected, got = int(self.model.c_u), int(stream.shape[-1])
        if got == expected:
            return stream
        breakdown = (
            f"up_ph={int(up_ph.shape[-1])}"
            if up_st is None
            else f"up_st={int(up_st.shape[-1])} + up_ph={int(up_ph.shape[-1])}"
        )
        raise RuntimeError(
            f"source stream is {got} channels ({breakdown}) but the checkpoint's model was "
            f"built with c_u={expected} (use_up_st={bool(self.model.use_up_st)}). These widths "
            f"come from the HDF5, not from the model: point dataset_config at the shards this "
            f"checkpoint was trained on. Note 58 is both the current use_up_st=true width and "
            f"the old phase-only width, so decide from use_up_st before trusting the number."
        )

    def build_target_streams(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(fhr_st, fhr_ph)``, having checked their joint width against $c_y$.

        Args:
            batch: A batch from the data module.

        Returns:
            The target scattering and phase-harmonic blocks.

        Raises:
            RuntimeError: If their concatenated width disagrees with the model's ``c_y``.
        """
        y_st, y_ph = get_field(batch, "fhr_st"), get_field(batch, "fhr_ph")
        if y_st is None or y_ph is None:
            raise RuntimeError(
                "batch is missing `fhr_st` or `fhr_ph`. Both must appear in "
                "dataset_kwargs.load_fields."
            )
        expected = int(self.model.c_y)
        got = int(y_st.shape[-1]) + int(y_ph.shape[-1])
        if got != expected:
            raise RuntimeError(
                f"target stream is {got} channels (fhr_st={int(y_st.shape[-1])} + "
                f"fhr_ph={int(y_ph.shape[-1])}) but the checkpoint's model was built with "
                f"c_y={expected}. These widths come from the HDF5, not from the model: point "
                f"dataset_config at the shards this checkpoint was trained on."
            )
        return y_st, y_ph

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, batch: Any, *, lag_band_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Assemble the streams and run the model's forward.

        The returned dict is the model's, unmodified: an analysis that wants a subset takes it,
        and one that wants a derived quantity computes it, but nothing is dropped or renamed on
        the way through.

        Args:
            batch: A batch already on the compute device.
            lag_band_mask: Optional boolean lag keep-mask, ``(L,)`` or ``(T, L)``. ``None`` is
                a bit-exact no-op.

        Returns:
            The model's 24-key forward dict.
        """
        y_st, y_ph = self.build_target_streams(batch)
        u_stream = self.build_source_stream(batch)
        return self.model(y_st, y_ph, u_stream, lag_band_mask=lag_band_mask)

    def compute_loss(
        self, batch: Any, forward_outputs: Dict[str, torch.Tensor], *, beta: float = 0.0, **overrides
    ) -> Dict[str, torch.Tensor]:
        r"""Call ``model.compute_loss`` under the checkpoint's objective.

        The ``likelihood`` key is stripped from the result. ``compute_loss`` echoes the string
        it was given, and a metric logger or a DataFrame column that receives it coerces it to
        a clean $0.0$ rather than raising -- so a numeric consumer must never see it.

        Args:
            batch: The batch the forward was run on.
            forward_outputs: That forward's output dict.
            beta: KL weight for the returned ``total_loss``. See :meth:`Objective.loss_kwargs`.
            **overrides: Objective settings to override, for a controlled comparison.

        Returns:
            The loss dict, without ``likelihood``. Every value is a tensor.
        """
        y_st, y_ph = self.build_target_streams(batch)
        kwargs = dict(self.objective.loss_kwargs(beta=beta))
        kwargs.update(overrides)
        loss_dict = self.model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            weight=get_field(batch, "weight"),
            **kwargs,
        )
        return {key: value for key, value in loss_dict.items() if key != "likelihood"}

    def forecast_view(
        self, batch: Any, forward_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> "ForecastView":
        r"""Align a forward's forecast tensors with $Y^{+}$ and the feature mask.

        Three analyses -- forecast quality, uplift, residual activity -- all need the same four
        things: the forecast heads sliced to the valid anchors, the unfolded target, and the mask.
        Assembling them once here keeps the anchor slice and the mask construction in one place,
        rather than in three that must agree.

        Args:
            batch: A batch already on the compute device.
            forward_outputs: A completed forward, or ``None`` to run one.

        Returns:
            The aligned view.
        """
        outputs = self.forward(batch) if forward_outputs is None else forward_outputs
        y_st, y_ph = self.build_target_streams(batch)
        horizon = int(self.model.horizon)
        batch_size, seq_len = int(y_st.shape[0]), int(y_st.shape[1])
        anchors = seq_len - horizon

        target = future_target(y_st, y_ph, horizon)

        # Under detach_baseline_in_full, compute_loss scores mu_base + delta_mu_src rather than
        # the forward's own mu_full key. Detaching changes no value under no_grad, but the
        # recomposition is what was scored, so the pipeline recomposes too.
        if bool(self.objective.detach_baseline_in_full):
            mu_full = outputs["mu_base"] + outputs["delta_mu_src"]
        else:
            mu_full = outputs["mu_full"]

        from teb_vae.lag_attn.eval.masks import feature_mask

        return ForecastView(
            mu_full=mu_full[:, :anchors],
            mu_base=outputs["mu_base"][:, :anchors],
            delta_mu_src=outputs["delta_mu_src"][:, :anchors],
            logvar_full=outputs["logvar_full"][:, :anchors],
            logvar_base=outputs["logvar_base"][:, :anchors],
            y_plus=target,
            mask=feature_mask(
                self.model, get_field(batch, "weight"), batch_size, seq_len,
                device=target.device, dtype=target.dtype,
            ),
            n_scattering=int(y_st.shape[-1]),
            outputs=outputs,
        )

    def build_future_target(self, batch: Any) -> torch.Tensor:
        r"""Unfold $Y^{+}$: at anchor $t$, the window $Y[t+1 : t+1+H_d]$.

        Warm-up masking is deliberately *not* applied. Each caller uses its own window -- the
        feature loss masks to ``[warmup, T-H_d)``, the lag ablation to a common dead-anchor-safe
        support -- and a helper that pre-masked would force every one of them to undo it.

        Args:
            batch: A batch from the data module.

        Returns:
            The future target, $(B, T - H_d, H_d, c_y)$.
        """
        y_st, y_ph = self.build_target_streams(batch)
        return future_target(y_st, y_ph, int(self.model.horizon))


# ---------------------------------------------------------------------------
# Small batch/tensor helpers
#
# Here rather than in the figure module: the collectors need them before any figure exists.
# ---------------------------------------------------------------------------
def get_field(batch: Any, name: str) -> Any:
    """Read a batch field by name, tolerating both mapping and attribute access.

    The loader yields an ``AttributeDict``, which supports both; a test stub may be a plain
    ``SimpleNamespace``, which supports only the second.

    Args:
        batch: A batch or batch-like object.
        name: The field name.

    Returns:
        The field value, or ``None`` when the batch does not carry it.
    """
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def guid_of(batch: Any, index: int) -> str:
    """Return the GUID of one sample, as a string.

    ``guid`` survives collation as a ``list[str]``, never a tensor, which is why it is excluded
    from the device transfer.

    Args:
        batch: A batch from the data module.
        index: Position within the batch.

    Returns:
        The GUID, or ``"unknown"`` when the batch carries none.
    """
    guids = get_field(batch, "guid")
    if guids is None:
        return "unknown"
    if isinstance(guids, (list, tuple)):
        return str(guids[index])
    return str(guids)


def batch_size_of(batch: Any) -> int:
    """Return the number of samples in a batch, from the first tensor field present.

    Args:
        batch: A batch from the data module.

    Returns:
        The batch size.

    Raises:
        RuntimeError: If no declared tensor field is present, so the size cannot be determined.
    """
    for name in TENSOR_FIELDS:
        value = get_field(batch, name)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            return int(value.shape[0])
    raise RuntimeError(
        f"cannot determine the batch size: none of {TENSOR_FIELDS} is a tensor on this batch."
    )


def field_names(batch: Any) -> Sequence[str]:
    """Return the field names a batch carries, for logging and probe records.

    Args:
        batch: A batch from the data module.

    Returns:
        The field names, sorted.
    """
    if isinstance(batch, dict):
        return sorted(batch.keys())
    return sorted(name for name in vars(batch) if not name.startswith("_"))
