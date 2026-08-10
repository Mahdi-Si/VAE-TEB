r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships
    # 7 -- or Lightning rejects the device/world-size mismatch at Trainer construction. Export
    # the stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_fs.trainer --config teb_vae/lag_attn_fs/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names
the config to use. Note a Run-button launch of ``default.yaml`` is a *single* process whose seven
``cuda_devices`` make Lightning spawn DDP workers underneath it; for a single-device smoke run
point ``RUN_CONFIG`` at ``configs/tiny.yaml``.

Everything here is inherited. The config-to-constructor sweep, the causal-reach-budget resolution,
the four pre-flight guards, the callback assembly, the resolved-config persistence, the DDP
strategy selection and the diagnostic-plot wiring are all model-independent and are reused rather
than copied -- copies are how two models that must stay comparable stop being comparable. What this
module supplies is five class attributes and an entry point.

Two of the five are worth reading twice. ``TARGET_FIELDS`` is what the shared entry point's
normalisation guard is run against, and it is the one thing a target-domain change alters about
that check: without ``fhr_st`` and ``fhr_ph`` in ``normalize_fields`` the target arrives at its
stored scale, the Gaussian NLL is computed against a z-scale variance model, and the run trains a
meaningless objective to completion with nothing raising. ``TRACKED_METRICS`` adds the four
forecast-gap columns this model reports and the raw-target siblings do not.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

#: Repository root: ``teb_vae/lag_attn_fs/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` and `train.` imports below would fail with
# ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn_fs.trainer` from the repo root sets __package__ and needs none of
# this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs  # noqa: E402
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask  # noqa: E402
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer  # noqa: E402
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402

#: Metric suffixes this model emits and the raw-signal siblings do not: the forecast gap resolved
#: by horizon step and split by stored feature block.
#:
#: They exist because with no evaluation pipeline every other readout is a scalar summed over
#: $H \cdot C_{\mathrm{keep}} = 2340$ coefficients, and one scalar cannot separate a model that
#: forecasts from one that reconstructs the part of the target its own history already determines.
#: A stored coefficient is a two-sided average, so at $\tau = 0$ half of the target's support lies
#: in observed history and by $\tau = 29$ none of it does; a gap that is real forecasting survives
#: to the far step, and one that is not, does not. The block split is the same question along the
#: other axis: the two stored blocks' filters have different reaches, so their blends differ.
#:
#: Emitted on both stages, like every other term of the objective.
_FORECAST_GAP_SUFFIXES: Tuple[str, ...] = (
    "pred_gap_tau_first",
    "pred_gap_tau_last",
    "pred_gap_st",
    "pred_gap_ph",
)


class LagAttnFsTrainer(LagAttnRwsTrainer):
    """Experiment driver for :class:`~teb_vae.lag_attn_fs.nets.model.SeqVaeLagAttnFs`."""

    MODEL_CLS = SeqVaeLagAttnFs
    TASK_CLS = SeqVaeLagAttnFsTask
    CHECKPOINT_STEM = "lag-attn-fs"

    #: The loader fields this model's reconstruction target is built from, checked by the shared
    #: entry point's normalisation guard. Both blocks, because the target is their concatenation
    #: and a config carrying one of them is a target with a hole in it.
    TARGET_FIELDS: Tuple[str, ...] = ("fhr_st", "fhr_ph")

    #: The inherited metric surface plus this model's four forecast-gap columns, on both stages.
    #: An attribute rather than an override of ``train_model``, which is 75 lines of callback
    #: assembly whose copy would be free to drift from the one the comparison model runs under.
    TRACKED_METRICS: Tuple[str, ...] = _TRACKED_METRICS + tuple(
        f"{stage}/{name}" for stage in ("train", "val") for name in _FORECAST_GAP_SUFFIXES
    )


def main(config_path: str) -> None:
    """Resolve the config, build everything, and run the fit.

    Delegates to the shared entry point with this package's driver. The pre-flight guards, the
    temporary resolved-config file, ``setup_config``'s ordering constraints and the resolved-config
    persistence are all model-independent, and a copy of them here would be free to drift from the
    ones the comparison model actually runs under.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
    """
    run_training(config_path, trainer_cls=LagAttnFsTrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an
    IDE's working directory is not something this module can rely on. Absolute paths pass through
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
#: ``--config`` on the command line always wins over this value. A relative path is resolved
#: against the repository root, not the working directory.
RUN_CONFIG: str | None = "teb_vae/lag_attn_fs/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. teb_vae/lag_attn_fs/configs/default.yaml. Run from "
        "the repo root. Optional only if RUN_CONFIG is set in this file (for an IDE Run button).",
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
