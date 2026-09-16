r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_slot_transformer_cfs.trainer \
        --config teb_vae/lag_slot_transformer_cfs/configs/tiny.yaml

    # Prod box. The rank count must equal the length of general_config.cuda_devices or Lightning
    # rejects the device/world-size mismatch at Trainer construction. Export the stamp so the
    # other ranks share rank zero's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_slot_transformer_cfs.trainer \
        --config teb_vae/lag_slot_transformer_cfs/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names the
config to use. A Run-button launch of the production config is a *single* process whose seven
devices make Lightning spawn distributed workers underneath it; for a single-device smoke run point
``RUN_CONFIG`` at the tiny config.

**Almost everything here is inherited, from two parents at once.** The warm-up budget resolution,
the pre-flight refusals, the callback assembly, the resolved-config persistence, the distributed
strategy selection and the step-granular learning-rate monitor all reach this class through the
causal parent and the conv-Transformer parent. Copies are how two models that must stay comparable
stop being comparable.

**What is written here is the warm start**, and it is written here because the inherited one cannot
do it. ``core_model_checkpoint`` goes through a strict load: every key must align, or the run
refuses. That is exactly right for resuming this model kind and exactly wrong for warm-starting from
a target-only forecaster, whose checkpoint has no source pathway at all -- the strict load would
refuse, and a load that merely tolerated the gap would leave the operator unable to say which
tensors arrived. :func:`transfer_target_weights` copies the target encoder, the prior and the
decoder, lists every tensor it transferred, could not find and left as constructed, and refuses a
tensor whose shape disagrees rather than skipping it.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so every absolute import below would fail before __main__ is
# reached. Launching as a module from the repo root sets __package__ and needs none of this, which
# is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.nets.model import (  # noqa: E402
    SeqVaeLagResidualTrfCfs,
)
from teb_vae.lag_slot_transformer_cfs.task import (  # noqa: E402
    TASK_METRIC_SUFFIXES,
    VALIDATION_MC_DRAWS_KEY,
    VALIDATION_MONITOR_SUFFIXES,
    SeqVaeLagResidualTrfCfsTask,
)

#: Module prefixes a target-only warm start transfers. Everything the base forecast is made of, and
#: nothing the source pathway touches.
#:
#: Written as prefixes rather than as exact names because the tensors underneath them are an
#: architecture detail that moves; what does not move is which *component* a target-only model and
#: this one have in common.
#:
#: ``clock_proj.`` and ``clock_norm.`` are on this list and were not always. The metadata clock is
#: a function of stored position and of nothing else, so it belongs to the **prior's** conditioning
#: rather than to the source pathway -- and the target-only arm of this same class trains both the
#: projection and the normaliser in front of it. Re-zeroing the projection on transfer would discard
#: a trained target-only component and start the candidate's prior from a state its baseline never
#: occupied; leaving the normaliser's affine at its constructed values while copying the projection
#: would pair a trained weight with an untrained input scale, and the base forecast would then not
#: equal the donor's. A checkpoint that carries neither tensor -- a target-only model of some other
#: architecture -- reports both as missing and leaves them as constructed, which is the right
#: starting point anyway.
TRANSFERABLE_PREFIXES: Tuple[str, ...] = (
    "target_gate.",
    "target_adapter.",
    "target_encoder.",
    "prior_head.",
    "clock_norm.",
    "clock_proj.",
    "horizon_core.",
    "decoder.",
)

#: Prefixes a transfer deliberately leaves at their constructed values, and re-zeroes afterwards.
#: The source pathway starts at the prior on every warm start that is not an exact resume of this
#: model kind, so a jointly trained run begins by asserting that the source says nothing.
SOURCE_PREFIXES: Tuple[str, ...] = (
    "proposal_head.",
    "source_encoder.",
)

#: The config key that asks for a partial, target-only transfer. Separate from the family's
#: ``core_model_checkpoint``, which means a strict load of this exact model kind: the two do
#: different things to the source pathway, and one key meaning both would make a run's starting
#: point unrecoverable from its config.
WARM_START_KEY = "target_warm_start_checkpoint"

#: The metric names the objective reports, in the order it assembles them. Derived from the
#: objective's own module rather than restated, so a metric that stops being produced fails the
#: driver's own test instead of leaving an empty column.
#: Columns produced on training batches alone. Two are the framework's, injected when the spike
#: breaker is enabled, and they are its whole diagnostic surface -- the per-step skip decision and
#: the running comparison it makes -- without which a run that trained normally and then skipped
#: every batch forever shows nothing. The other two are the pre-clip gradient norm and how often it
#: exceeded the threshold, which is exactly what the inherited clip has to be re-derived from: the
#: norm's epoch value is an aggregate, so "how often did the clip bind" is recoverable from no
#: other recorded quantity.
_TRAIN_ONLY_SUFFIXES: Tuple[str, ...] = (
    "spike_skipped",
    "spike_ema_loss",
    "grad_norm",
    "grad_clip_frac",
)

_OBJECTIVE_SUFFIXES: Tuple[str, ...] = (
    "total_loss",
    "nll_full_block",
    "nll_base_block",
    "nll_full_sample",
    "nll_base_sample",
    "pred_gap",
    "source_conditioned_kl_train",
    "source_conditioned_kl_raw",
    "prior_rate",
    "kld_active_frac",
    "scored_anchors",
    "scored_coefficients",
    "mask_coverage_frac",
    "anchor_coverage_frac",
    "mean_logvar_full",
    "mean_logvar_base",
    "logvar_full_floor_frac",
    "logvar_full_ceil_frac",
    "mean_logvar_prior",
    "mean_logvar_post",
    "logvar_prior_floor_frac",
    "delta_mu_rms",
    "kld_beta",
    "beta_prior",
)


#: The monitor's columns as the driver tracks them: on the validation stage alone, and only for a
#: run whose configuration set the draw count. Named here so the tracked surface, the callback
#: check below and the test that pins both read one tuple.
VALIDATION_MONITOR_COLUMNS: Tuple[str, ...] = tuple(
    f"val/{name}" for name in VALIDATION_MONITOR_SUFFIXES
)


def validation_monitor_draws(config: Mapping[str, Any]) -> Optional[int]:
    """Read the predictive monitor's draw count off a resolved configuration, and check it.

    Two refusals, both for a run that would otherwise fail late or not at all. A draw count that
    is not a positive integer is refused before a model is built. A checkpoint or early-stopping
    monitor that names one of the monitor's columns while the draw count is unset is refused
    here by name: left alone, the framework would raise at the first validation end that the
    monitored key does not exist, after the whole first epoch had been trained.

    Args:
        config: The resolved configuration.

    Returns:
        The draw count, or ``None`` when the monitor is off.

    Raises:
        ValueError: On a malformed draw count, or on a callback monitoring a column the run
            will not produce.
    """
    vae_config = (config.get("model_config") or {}).get("VAE_model") or {}
    draws = vae_config.get(VALIDATION_MC_DRAWS_KEY)
    if draws is not None:
        if isinstance(draws, bool) or int(draws) != draws or int(draws) < 1:
            raise ValueError(
                f"model_config.VAE_model.{VALIDATION_MC_DRAWS_KEY}={draws!r} must be a positive "
                f"integer or null: it is the number of latent draws the predictive validation "
                f"monitor marginalises over."
            )
        draws = int(draws)

    callbacks = (config.get("advanced_config") or {}).get("callbacks") or {}
    named: List[str] = []
    for block, keys in (
        ("early_stopping", ("monitor",)),
        ("model_checkpoint", ("monitor", "secondary_monitor")),
    ):
        section = callbacks.get(block) or {}
        if block == "early_stopping" and not section.get("enabled", False):
            continue
        for key in keys:
            value = section.get(key)
            if value in VALIDATION_MONITOR_COLUMNS:
                named.append(f"advanced_config.callbacks.{block}.{key}={value}")
    if named and draws is None:
        raise ValueError(
            f"{', '.join(named)} names a predictive-monitor column, but "
            f"model_config.VAE_model.{VALIDATION_MC_DRAWS_KEY} is null, so the run would never "
            f"log it and the callback would refuse at the end of the first validation epoch. "
            f"Set the draw count, or point the monitor at a legacy column."
        )
    return draws


#: Attribute names a task stores the net under, whose keys a checkpoint therefore carries as a
#: prefix. Written out because getting it wrong is not a failure: the transfer would find nothing
#: matching, refuse with "no tensor transferred", and send the operator looking at the checkpoint
#: rather than at the prefix.
TASK_MODULE_PREFIXES: Tuple[str, ...] = ("_orig_model.", "model.")


def strip_task_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a state mapping in the net's own key coordinates.

    A checkpoint written by the training task carries the net's keys under the attribute the task
    stores it as. The transfer takes a mapping already in the net's coordinates, so the prefix is
    removed here rather than inside it.

    Args:
        state: The checkpoint's state mapping, in whatever coordinates it was written.

    Returns:
        The mapping, with any known task prefix removed.
    """
    stripped: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        for prefix in TASK_MODULE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        stripped[name] = tensor
    return stripped


def transfer_target_weights(
    model: torch.nn.Module, checkpoint_state: Mapping[str, torch.Tensor]
) -> Dict[str, List[str]]:
    r"""Copy a target-only forecaster's weights into this model, and say exactly what happened.

    The target encoder, the prior and the decoder are the same components in both models, so a
    competitively trained target-only forecaster is the right starting point for the base branch --
    and the design's reason for wanting one is that a weak baseline makes the residual branch look
    informative by letting it correct a failure that had nothing to do with the source.

    **Three lists come back, and all three matter.** A tensor that transferred is one the run did
    not have to learn again. A tensor that was *missing* from the checkpoint is one the run starts
    randomly, which may be correct -- a source model built without the persistence residual has no
    such weight -- or may be a silent half-transfer. A tensor left as constructed is the source
    pathway, which is meant to start at zero. A transfer that reported only a success flag would
    make the second and third indistinguishable.

    **A shape disagreement refuses rather than skips.** A transferable tensor present at the wrong
    shape means the checkpoint was trained on a different task geometry -- another kept-channel set,
    another latent width, another horizon -- and skipping it would produce a model half-initialised
    from a task it will never be scored on.

    Args:
        model: The freshly constructed net.
        checkpoint_state: A state mapping from the target-only checkpoint, already unwrapped.

    Returns:
        ``{'transferred': ..., 'missing': ..., 'reinitialised': ..., 'unused': ...}``, each a
        sorted list of names.

    Raises:
        ValueError: On a transferable tensor whose shape disagrees, naming it and both shapes; or
            if nothing transferred at all, which means the checkpoint describes another
            architecture rather than a target-only version of this one.
    """
    own = dict(model.state_dict())
    # Buffers registered non-persistent are absent from both sides by design, so they never appear
    # in any of the four lists; their contents follow the resolved budget rather than the weights.
    transferred: List[str] = []
    missing: List[str] = []
    reinitialised: List[str] = []

    for name, tensor in own.items():
        if not name.startswith(TRANSFERABLE_PREFIXES):
            reinitialised.append(name)
            continue
        if name not in checkpoint_state:
            missing.append(name)
            continue
        incoming = checkpoint_state[name]
        if tuple(incoming.shape) != tuple(tensor.shape):
            raise ValueError(
                f"warm-start tensor {name!r} has shape {tuple(incoming.shape)} in the checkpoint "
                f"against {tuple(tensor.shape)} in this model. That is a different task geometry "
                f"-- another kept-channel set, latent width or horizon -- and transferring around "
                f"it would leave the model half-initialised from a task it will never be scored "
                f"on. Point the run at a checkpoint trained on this geometry."
            )
        tensor.copy_(incoming)
        transferred.append(name)

    if not transferred:
        raise ValueError(
            f"no tensor transferred from the warm-start checkpoint: none of its keys matched "
            f"{TRANSFERABLE_PREFIXES} in this model. That checkpoint describes another "
            f"architecture rather than a target-only version of this one, and continuing would "
            f"train from random weights while the config said otherwise."
        )

    # After the copy: the source pathway starts at the prior on every warm start, so a jointly
    # trained run begins by asserting that the source says nothing and earns every nat of coupling
    # it later reports. Re-zeroing here rather than trusting construction is what makes that true
    # even if a future transfer prefix grows to overlap the source pathway.
    #
    # Guarded, because the target-only arm builds no proposal head at all: warm-starting one
    # target-only model from another is a legitimate thing to do -- it is how a second seed of the
    # baseline is started from the first -- and it must not fail on a module that arm never has.
    if model.proposal_head is not None:
        model.proposal_head.zero_output()

    return {
        "transferred": sorted(transferred),
        "missing": sorted(missing),
        "reinitialised": sorted(reinitialised),
        "unused": sorted(set(checkpoint_state) - set(own)),
    }


class LagResidualTrfCfsTrainer(LagAttnCfsTrainer, LagAttnTrfRwsTrainer):
    """Experiment driver for
    :class:`~teb_vae.lag_slot_transformer_cfs.nets.model.SeqVaeLagResidualTrfCfs`.

    Three class attributes are re-pointed because all three collide: both parents set the model
    class, the task class and the checkpoint stem, so resolution order alone would take the causal
    side and every failure would be silent -- a run that looks like this package and builds another
    model, or one that interleaves two models' checkpoints in a shared output tree.

    ``PLOT_CONFIG_KEY`` deliberately stays the shared driver's literal. The callback assembly reads
    it, and a sibling that renames it to match its own package gets no figure, no error and nothing
    in the log saying why. What this class re-points instead is ``plot_callback_cls``, because the
    page the shared callback draws is not one this architecture can be drawn on; see
    :mod:`~teb_vae.lag_slot_transformer_cfs.sample_page`.
    """

    MODEL_CLS = SeqVaeLagResidualTrfCfs
    TASK_CLS = SeqVaeLagResidualTrfCfsTask
    CHECKPOINT_STEM = "lag-residual-trf-cfs"

    #: The metric surface this driver tracks, which is this package's rather than the family's.
    #:
    #: The inherited tuple names three groups this objective does not produce: the shape terms,
    #: which read a block's last axis as a trajectory where here it counts channels; the warm and
    #: novelty tertile splits of the forecast gap, which the shared objective merges and this one
    #: does not compute; and the source-null divergence, which needs a source pathway with
    #: parameters to encode a zeroed stream through. Tracking a name nothing produces is a column
    #: that is empty in every row of every run.
    TRACKED_METRICS: Tuple[str, ...] = (
        tuple(
            f"{stage}/{name}"
            for stage in ("train", "val")
            for name in _OBJECTIVE_SUFFIXES + TASK_METRIC_SUFFIXES
        )
        + tuple(f"train/{name}" for name in _TRAIN_ONLY_SUFFIXES)
        + ("lr",)
    )

    @classmethod
    def plot_callback_cls(cls) -> type:
        """Return this package's diagnostic-plot callback, importing it on the way.

        A method rather than a class attribute so the import stays **lazy**: the callback pulls
        matplotlib and the page module behind it, and a module-level attribute would import both
        in every run whether or not the config asked for a figure. The shared assembly calls this
        only inside its enabled branch.

        Re-pointed away from the family's callback because that one runs a forward without the
        per-lag proposals and hands the result to a page builder that reads two tensors this
        architecture does not produce; see :mod:`~teb_vae.lag_slot_transformer_cfs.plotting`.

        Returns:
            The callback class the shared ``train_model`` constructs.
        """
        from teb_vae.lag_slot_transformer_cfs.plotting import LagResidualTrfCfsPlotCallback

        return LagResidualTrfCfsPlotCallback

    def create_model(self) -> None:
        """Build the net, apply a target-only warm start if one is configured, and wrap it.

        Runs the inherited construction first, so the model, the causal standing message and the
        strict-load path are all the family's. The warm start is applied afterwards and refuses to
        run alongside the strict load: the two do different things to the source pathway, and a run
        that did both would have a starting point recoverable from neither key.

        Raises:
            ValueError: If both checkpoint keys are set, or if the transfer refuses.
        """
        model_config = self.config.get("model_config", {}) or {}
        warm_start = model_config.get(WARM_START_KEY)
        if warm_start is not None and model_config.get("core_model_checkpoint") is not None:
            raise ValueError(
                f"both core_model_checkpoint and {WARM_START_KEY} are set. The first is a strict "
                f"load of this exact model kind, which restores the source pathway too; the "
                f"second is a partial transfer that deliberately leaves it at zero. A run doing "
                f"both has a starting point that neither key describes. Set one."
            )

        super().create_model()
        self._log_lag_geometry()
        self._configure_validation_monitor()

        if warm_start is None:
            return

        blob = torch.load(str(warm_start), map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        stripped = strip_task_prefix(state)
        report = transfer_target_weights(self.pytorch_model, stripped)
        logger.info(f"target warm start from {warm_start}")
        for group in ("transferred", "missing", "reinitialised", "unused"):
            names = report[group]
            logger.info(f"  {group}: {len(names)} tensors" + (f" -- {names}" if names else ""))
        if report["missing"]:
            logger.warning(
                f"{len(report['missing'])} transferable tensors were absent from the warm-start "
                f"checkpoint and start randomly. That is correct when the source model was built "
                f"without them and a silent half-transfer otherwise; the names are above."
            )
        # No rebuild of the task. The transfer writes **in place** through ``Tensor.copy_``, and
        # the task the inherited construction built wraps this same module object, so it already
        # holds the transferred weights. Rebuilding it would be a second construction whose
        # keyword list is free to drift from the one the family's own path uses.

    def _configure_validation_monitor(self) -> None:
        """Hand the task the monitor's draw count and track its columns, when configured.

        The draw count reaches the task by the route the seed takes -- applied onto its saved
        hyperparameters after construction, so it is in the checkpoint -- rather than through the
        shared constructor call, whose keyword list is the family's. The three columns join the
        tracked surface **on this instance** only when the run will produce them: a legacy run
        keeps the class attribute and therefore exactly the columns it always had, and no column
        is tracked that is empty in every row.
        """
        draws = validation_monitor_draws(self.config)
        if draws is None:
            return
        self.apply_config_hyperparameters({VALIDATION_MC_DRAWS_KEY: draws}, self.pl_model)
        self.TRACKED_METRICS = tuple(self.TRACKED_METRICS) + VALIDATION_MONITOR_COLUMNS
        logger.info(
            f"predictive validation monitor on: K={draws} paired draws per dense validation "
            f"batch, unweighted mixture score of both branches under a fixed noise bank, "
            f"logged as {', '.join(VALIDATION_MONITOR_COLUMNS)}"
        )

    def _log_lag_geometry(self) -> None:
        r"""State the resolved lag bank in the run's first lines: its length, its scale, its depth.

        Three numbers that follow from one configured value and are each misread on their own. The
        bank length $L$ is one more than the furthest lag because the anchor's own step is a
        candidate; the summation scale $c_L$ follows from $L$ unless the configuration fixed it,
        and it changes the proposal amplitude, so two runs at two bank lengths differ in scale as
        well as in support; and the oldest centre in seconds is $(L - 1)$ steps back, not $L$,
        which is what "$L$ steps of history" is read as. The resolved configuration written
        beside the checkpoints carries the configured value; this line carries what it resolved
        to, on the model that was actually built.
        """
        model = self.pytorch_model
        n_lags = int(model.n_lags)
        oldest_seconds = float(n_lags - 1) * float(SECONDS_PER_STEP)
        if bool(getattr(model, "source_disabled", False)):
            logger.info(
                "resolved lag bank: none -- this arm builds no source pathway, so no lag is read"
            )
            return
        logger.info(
            f"resolved lag bank: L={n_lags} candidate lags (0..{n_lags - 1}, the anchor's own "
            f"step included), summation scale c_L={float(model.lag_scale):.6g}, oldest gathered "
            f"centre {oldest_seconds:g} s before the anchor in stored-coefficient time"
        )


def main(config_path: str) -> LagResidualTrfCfsTrainer:
    """Resolve the config, build everything, and run the fit.

    Delegates to the shared entry point with this package's driver. The pre-flight guards, the
    temporary resolved-config file and the resolved-config persistence are model-independent, and a
    copy of them here would be free to drift from the ones the comparison models run under.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.

    Returns:
        The driver, after the fit -- the only handle on where the run's checkpoints went.
    """
    return run_training(config_path, trainer_cls=LagResidualTrfCfsTrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an IDE's
    working directory is not something this module can rely on. Absolute paths pass through
    untouched.

    Args:
        config_path: The path as supplied on the command line or via :data:`RUN_CONFIG`.

    Returns:
        An absolute path.
    """
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(_REPO_ROOT, config_path)


#: Config used when the module is launched with no ``--config`` -- i.e. an IDE's Run button.
#: ``--config`` on the command line always wins over this value. A relative path is resolved against
#: the repository root, not the working directory.
#:
#: Point it at ``configs/tiny.yaml`` for a single-device smoke run. The production config's device
#: list makes a Run-button launch spawn distributed workers underneath one process.
RUN_CONFIG: str | None = "teb_vae/lag_slot_transformer_cfs/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. "
        "teb_vae/lag_slot_transformer_cfs/configs/default.yaml. Run from the repo root. Optional "
        "only if RUN_CONFIG is set in this file (for an IDE Run button).",
    )
    _args = parser.parse_args()

    _config_path = _args.config or RUN_CONFIG
    if _config_path is None:
        parser.error(
            "--config is required. To launch from an IDE Run button instead, set RUN_CONFIG "
            "near the bottom of this file to a config path."
        )

    _config_path = _resolve_cli_config_path(_config_path)

    # The paths *inside* a config are repo-root-relative too, and under an IDE Run button the
    # working directory is whatever the IDE chose -- a relative shard path then resolves to nothing
    # and the loader dies as "no samples match the specified filters" with no mention of the cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
