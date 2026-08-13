r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn_transformer_cfs.trainer \
        --config teb_vae/lag_attn_transformer_cfs/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships
    # 7 -- or Lightning rejects the device/world-size mismatch at Trainer construction. Export
    # the stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_transformer_cfs.trainer \
        --config teb_vae/lag_attn_transformer_cfs/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names the
config to use. Note a Run-button launch of ``default.yaml`` is a *single* process whose seven
``cuda_devices`` make Lightning spawn DDP workers underneath it; for a single-device smoke run point
``RUN_CONFIG`` at ``configs/tiny.yaml``.

**Everything here is inherited, from two parents at once, and the three class attributes are the
entire difference between building this architecture and building either model it is compared
against.** The warm-up budget resolution, the five pre-flight refusals, the callback assembly, the
resolved-config persistence, the DDP strategy selection, the step-granular learning-rate monitor and
the diagnostic-plot wiring all reach this class through
:class:`~teb_vae.lag_attn_cfs.trainer.LagAttnCfsTrainer` and
:class:`~teb_vae.lag_attn_transformer_rws.trainer.LagAttnTrfRwsTrainer` -- copies are how two models
that must stay comparable stop being comparable.

Three consequences of the resolution order are worth reading twice, because none is written anywhere
else:

* ``TARGET_FIELDS``, ``TRACKED_METRICS`` and ``preflight`` come from the **causal** parent, which is
  what the target domain decides. Without ``fhr_st`` and ``fhr_ph`` in ``normalize_fields`` the
  target arrives at its stored scale, the Gaussian NLL is computed against a z-scale variance model,
  and the run trains a meaningless objective to completion with nothing raising; the guard that
  catches it lives in the shared entry point and reads the driver it was handed, so a
  ``trainer_cls=`` wiring mistake would leave it checking the *raw* model's field -- which these
  configs satisfy.
* ``compile_model_requested`` and ``_build_trainer_kwargs`` come from the **conv-Transformer**
  parent. The causal parent defines neither, so lookup passes through, and ``torch.compile`` becomes
  permitted on a model whose causal ancestor never exercised it. That is the right outcome -- it is
  the transformer encoder that makes compilation worth having -- but it arrives by resolution order
  rather than by anything written down, so ``tests/test_trainer.py`` asserts it explicitly. Shipped
  configs keep ``compile: false`` regardless.
* ``_build_model_kwargs`` and ``create_model`` are defined on **both** parents, and both run: each
  calls ``super()``, so the linearisation threads the conv-Transformer's contributions (re-admitting
  ``source_attention_window: null``, applying ``lr_warmup_steps``) underneath the causal one's (the
  four resolved warm-up tuples, the geometry log line, the seed, the budget handed to the task). A
  reader who assumes "resolves to the causal side" also loses the transformer half is wrong, and
  ``tests/test_trainer.py`` asserts both halves fire rather than only the outermost.
"""
from __future__ import annotations

import argparse
import os
import sys

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path rather
# than the repository root -- so the `teb_vae.` and `train.` imports below would fail with
# ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn_transformer_cfs.trainer` from the repo root sets __package__ and needs
# none of this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask  # noqa: E402
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer  # noqa: E402


class LagAttnTrfCfsTrainer(LagAttnCfsTrainer, LagAttnTrfRwsTrainer):
    """Experiment driver for
    :class:`~teb_vae.lag_attn_transformer_cfs.nets.model.SeqVaeLagAttnTrfCfs`.

    Three class attributes and no method. **All three are re-pointed because all three collide**:
    both parents set ``MODEL_CLS``, ``TASK_CLS`` and ``CHECKPOINT_STEM``, so resolution order alone
    would take the causal side and each failure would be silent --

    * omit ``MODEL_CLS`` and the driver builds a conv-LSTM model with no error anywhere: a run that
      looks like this package and is not;
    * omit ``TASK_CLS`` and the same, one layer up;
    * omit ``CHECKPOINT_STEM`` and it writes ``lag-attn-cfs-*.ckpt``, interleaving two models'
      checkpoints in whichever output tree they share.

    ``PLOT_CONFIG_KEY`` deliberately stays ``"lag_attn_rws_plotting"``, inherited from the shared
    driver: a sibling that renames it to match its own package gets no figure, no error and nothing
    in the log saying why.
    """

    MODEL_CLS = SeqVaeLagAttnTrfCfs
    TASK_CLS = SeqVaeLagAttnTrfCfsTask
    CHECKPOINT_STEM = "lag-attn-trf-cfs"


def main(config_path: str) -> None:
    """Resolve the config, build everything, and run the fit.

    Delegates to the shared entry point with this package's driver. The pre-flight guards, the
    temporary resolved-config file, ``setup_config``'s ordering constraints and the resolved-config
    persistence are all model-independent, and a copy of them here would be free to drift from the
    ones the comparison models actually run under.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
    """
    run_training(config_path, trainer_cls=LagAttnTrfCfsTrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an IDE's
    working directory is not something this module can rely on. Absolute paths pass through
    untouched.

    Args:
        config_path: The path as supplied on the command line or via ``RUN_CONFIG``.

    Returns:
        An absolute path.
    """
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(_REPO_ROOT, config_path)


#: Config used when the module is launched with no ``--config`` -- i.e. an IDE's Run button.
#: ``--config`` on the command line always wins over this value. A relative path is resolved against
#: the repository root, not the working directory.
RUN_CONFIG: str | None = "teb_vae/lag_attn_transformer_cfs/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. "
        "teb_vae/lag_attn_transformer_cfs/configs/default.yaml. Run from the repo root. Optional "
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

    # The paths *inside* a config are repo-root-relative too (see configs/tiny.yaml), and under an
    # IDE Run button the working directory is whatever the IDE chose -- a relative shard path then
    # resolves to nothing and the loader dies as "No samples match the specified filters" with no
    # mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
