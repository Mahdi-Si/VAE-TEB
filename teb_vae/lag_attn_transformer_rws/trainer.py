r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn_transformer_rws.trainer \
        --config teb_vae/lag_attn_transformer_rws/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships
    # 7 -- or Lightning rejects the device/world-size mismatch at Trainer construction. Export
    # the stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_transformer_rws.trainer \
        --config teb_vae/lag_attn_transformer_rws/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names
the config to use. Note a Run-button launch of ``default.yaml`` is a *single* process whose seven
``cuda_devices`` make Lightning spawn DDP workers underneath it; for a single-device smoke run
point ``RUN_CONFIG`` at ``configs/tiny.yaml``.

Almost everything here is inherited. The config-to-constructor sweep, the causal-reach-budget
resolution, the four pre-flight guards, the callback assembly, the resolved-config persistence and
the DDP strategy selection are all model-independent and are reused rather than copied -- copies
are how two models that must stay comparable stop being comparable. What this module supplies is
the three class attributes naming this architecture, the step-granular learning-rate monitor its
step-granular warm-up needs to be observable at all, and the entry point.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

#: Repository root: ``teb_vae/lag_attn_transformer_rws/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` and `train.` imports below would fail
# with ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn_transformer_rws.trainer` from the repo root sets __package__ and
# needs none of this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lightning.pytorch.callbacks import LearningRateMonitor  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws  # noqa: E402
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask  # noqa: E402

#: Config key holding the step-granular warm-up length, read here and forwarded onto the task's
#: hyperparameters. Beside ``lr`` and ``lr_milestone`` in ``general_config`` because it is the
#: third term of the same schedule, not a property of the network.
LR_WARMUP_STEPS_KEY = "lr_warmup_steps"

#: ``VAE_model`` keys for which ``null`` is a *value* rather than "leave the constructor default".
#: The inherited sweep drops every null, which is right for a key whose null means "unset" -- the
#: reach budget is the model case -- and wrong for the one key of this architecture whose null
#: means something: ``source_attention_window: null`` is the unbounded source encoder, the arm the
#: whole locality sweep is measured against. Dropped, it would silently rebuild the shipped
#: 16-step window and the arm would report the baseline's numbers under the unbounded arm's name.
NULLABLE_MODEL_KEYS = frozenset({"source_attention_window"})


class LagAttnTrfRwsTrainer(LagAttnRwsTrainer):
    """Experiment driver for
    :class:`~teb_vae.lag_attn_transformer_rws.nets.model.SeqVaeLagAttnTrfRws`."""

    MODEL_CLS = SeqVaeLagAttnTrfRws
    TASK_CLS = SeqVaeLagAttnTrfRwsTask
    CHECKPOINT_STEM = "lag-attn-trf-rws"

    def compile_model_requested(self) -> bool:
        """Honour ``advanced_config.trainer.compile``, which a conv-Transformer net can serve.

        The inherited refusal is a fact about the *raw-signal* net, not about the objective or the
        task: its LSTM encoders defeat TorchInductor unconditionally. This architecture replaced
        those encoders, so the blocker does not exist here and the key becomes live -- for this
        driver and for every driver below it, which is why the implementation lives here rather
        than being restated in each.

        Of the three blockers once recorded against this family, one is gone with the recurrence,
        one is refused below, and the third never applied to the compiled region at all:

        * **LSTM encoders** -- replaced by the causal conv-Transformer.
        * **A checkpointed attention region** -- still reachable, because
          ``attention_grad_checkpoint`` is a live config key. Refused explicitly rather than
          silently ignored: a run that quietly dropped either the checkpointing or the compilation
          is a run the operator did not configure.
        * **The data-dependent mask indexing behind** ``kld_active_frac`` -- lives in
          ``compute_loss``, which the task reaches through ``orig_model``. Only the forward is
          compiled, so that indexing never enters the graph.

        **Compilation is not numerically free, and that is why every config here ships it off.**
        Inductor may reassociate float arithmetic, and this family's headline readout is
        ``pred_gap`` -- a difference of order $10^{-1}$ between two block NLLs of order $10^{2}$,
        a relative scale of about $10^{-4}$. Before adopting a compiled run, compare ``pred_gap``
        on one fixed batch against the eager value; a difference there is a difference in the
        number the model exists to produce, not a rounding detail.

        Returns:
            ``True`` when the config asks for compilation.

        Raises:
            ValueError: If compilation is requested together with ``attention_grad_checkpoint``,
                naming both keys.
        """
        trainer_config = (self.config.get("advanced_config", {}) or {}).get("trainer", {}) or {}
        if not bool(trainer_config.get("compile", False)):
            return False

        vae_config = (self.config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        if bool(vae_config.get("attention_grad_checkpoint", False)):
            raise ValueError(
                "advanced_config.trainer.compile is true and "
                "model_config.VAE_model.attention_grad_checkpoint is true, and the two cannot "
                "both hold: the recomputed lag-attention region defeats TorchInductor, so one "
                "of them would be silently dropped and the run would be neither the compiled "
                "one nor the checkpointed one. Turn off whichever the run does not need -- the "
                "shipped configs set attention_grad_checkpoint: false, so compilation is the "
                "one to keep unless memory is the binding constraint."
            )
        logger.info(
            "torch.compile is ON for the net's forward; the objective stays eager through "
            "orig_model. Inductor may reassociate float arithmetic -- verify `pred_gap` against "
            "an eager run on one fixed batch before reading this run's coupling numbers."
        )
        return True

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Sweep the config onto the constructor, re-admitting the keys whose ``null`` is a value.

        The inherited sweep forwards every ``VAE_model`` key naming a real constructor argument,
        except those set to ``null`` -- a null there means "use the constructor's own default",
        which is the right reading for every key of the architecture this driver was written for.
        This architecture has one key where it is not: an unbounded source encoder *is*
        ``source_attention_window: null``, and dropping it rebuilds the shipped window instead,
        with the arm still reporting under its own name. See :data:`NULLABLE_MODEL_KEYS`.

        Returns:
            Constructor kwargs for the net.
        """
        model_kwargs = super()._build_model_kwargs()
        vae_config = (self.config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        model_kwargs.update(
            {
                name: None
                for name in NULLABLE_MODEL_KEYS
                if name in vae_config and vae_config[name] is None
            }
        )
        return model_kwargs

    def create_model(self) -> None:
        """Build the net and its task through the inherited path, then attach the step warm-up.

        The inherited ``create_model`` forwards ``lr`` and ``lr_milestones`` onto the task's
        hyperparameters and knows nothing about a step-granular ramp, so the third term of the
        same schedule is applied here -- through the same mechanism, so it lands in
        ``self.hparams`` and therefore in every checkpoint, rather than in a second place the
        task would have to look. Absent from the config it resolves to $0$, which is exactly the
        value at which the task delegates back to the framework's epoch-granularity schedule.
        """
        super().create_model()
        general_config = self.config.get("general_config", {}) or {}
        self.apply_config_hyperparameters(
            {LR_WARMUP_STEPS_KEY: int(general_config.get(LR_WARMUP_STEPS_KEY, 0) or 0)},
            self.pl_model,
        )

    def _build_trainer_kwargs(self, callbacks, model=None) -> Dict[str, Any]:
        """Assemble the ``Trainer`` kwargs, with the learning-rate monitor moved to step
        granularity.

        The framework attaches ``LearningRateMonitor(logging_interval='epoch')`` to every run it
        builds. A warm-up measured in optimizer steps completes well inside the first epochs, so an
        epoch-granular monitor cannot show it at all -- and a schedule nobody can observe is a
        schedule nobody can tell has silently done nothing. The monitor is *replaced* rather than
        supplemented so exactly one of them logs, under one name, at one resolution.

        Args:
            callbacks: This model's callbacks, from ``train_model``.
            model: The Lightning module, forwarded to the DDP strategy selection.

        Returns:
            The kwargs dict the framework's ``build_trainer`` constructs a ``Trainer`` from.
        """
        kwargs = super()._build_trainer_kwargs(callbacks, model=model)
        kwargs["callbacks"] = [
            LearningRateMonitor(logging_interval="step")
            if isinstance(callback, LearningRateMonitor)
            else callback
            for callback in kwargs["callbacks"]
        ]
        return kwargs


def main(config_path: str) -> None:
    """Resolve the config, build everything, and run the fit.

    Delegates to the shared entry point with this package's driver. The pre-flight guards, the
    temporary resolved-config file, ``setup_config``'s ordering constraints and the
    resolved-config persistence are all model-independent, and a copy of them here would be free
    to drift from the ones the comparison model actually runs under.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
    """
    run_training(config_path, trainer_cls=LagAttnTrfRwsTrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an
    IDE's working directory is not something this module can rely on. Absolute paths pass
    through untouched.

    Args:
        config_path: The path as supplied on the command line or via ``RUN_CONFIG``.

    Returns:
        An absolute path.
    """
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(_REPO_ROOT, config_path)


#: Config used when the module is launched with no ``--config`` -- i.e. an IDE's Run button.
#: ``--config`` on the command line always wins over this value. A relative path is resolved
#: against the repository root, not the working directory.
RUN_CONFIG: str | None = "teb_vae/lag_attn_transformer_rws/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. "
        "teb_vae/lag_attn_transformer_rws/configs/default.yaml. Run from the repo root. "
        "Optional only if RUN_CONFIG is set in this file (for an IDE Run button).",
    )
    _args = parser.parse_args()

    _config_path = _args.config or RUN_CONFIG
    if _config_path is None:
        parser.error(
            "--config is required. To launch from an IDE Run button instead, set RUN_CONFIG "
            "near the bottom of this file to a config path."
        )

    _config_path = _resolve_cli_config_path(_config_path)

    # The paths *inside* a config are repo-root-relative too (see configs/tiny.yaml), and under
    # an IDE Run button the working directory is whatever the IDE chose -- a relative shard path
    # then resolves to nothing and the loader dies as "No samples match the specified filters"
    # with no mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
