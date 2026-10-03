r"""Experiment driver for :class:`~teb_vae.lag_attn_transformer_patch.nets.model.SeqVaeLagAttnTrfPatch`.

The conv-Transformer driver with the patch model and task. What differs from the parent, and why:

* ``create_model`` hands the task ``general_config.seed`` (the anchor phase keys on it; the parent
  never passes it) and logs the anchor geometry.
* ``TRACKED_METRICS`` adds ``anchors_per_sample`` and ``val/kld_source_null``, so both reach
  ``metrics_history.csv``.
* ``causal_standing_message`` replaces the parent's sentence, which is false for patch inputs.
* ``preflight`` refuses the ST/PH feature keys the constructor sweep would drop in silence, then
  runs the shared raw-signal guards (``fhr``/``up`` loaded and normalized; ``weight``, ``guid``,
  ``epoch`` loaded; raw length against the shard; ``lambda_boundary`` off).

Run from the repository root::

    python -m teb_vae.lag_attn_transformer_patch.trainer \
        --config teb_vae/lag_attn_transformer_patch/configs/tiny.yaml

    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_transformer_patch.trainer \
        --config teb_vae/lag_attn_transformer_patch/configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple

#: Repository root: ``teb_vae/lag_attn_transformer_patch/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button puts this directory, not the repo root, on sys.path.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loguru import logger  # noqa: E402

from teb_vae.lag_attn_crws.trainer import (  # noqa: E402
    _check_boundary_term_is_off,
    _check_phase_key_fields,
    _check_raw_target_fields,
)
from teb_vae.lag_attn_rws.trainer import main as run_training  # noqa: E402
from teb_vae.lag_attn_transformer_e2e.trainer import (  # noqa: E402
    _check_raw_length_against_shard,
    _check_raw_source_normalized,
)
from teb_vae.lag_attn_transformer_patch.nets.model import SeqVaeLagAttnTrfPatch  # noqa: E402
from teb_vae.lag_attn_transformer_patch.task import SeqVaeLagAttnTrfPatchTask  # noqa: E402
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer  # noqa: E402

#: ``VAE_model`` keys of the ST/PH representation this model replaces. None reaches the patch
#: constructor, so the signature sweep would drop each one silently.
REMOVED_MODEL_KEYS: Tuple[str, ...] = (
    "c_y", "c_u", "use_up_st",
    "target_keep_index", "target_warmup_steps", "source_keep_index", "source_warmup_steps",
    "target_align_delays", "source_align_delays", "target_delays", "source_delays",
    "target_scored_horizon", "target_forecast_shift", "target_novelty_frac",
)
REMOVED_MODEL_PREFIXES: Tuple[str, ...] = ("causal_", "target_weight_", "target_phase_fast_")


class LagAttnTrfPatchTrainer(LagAttnTrfRwsTrainer):
    """Experiment driver for the patch-token cell."""

    MODEL_CLS = SeqVaeLagAttnTrfPatch
    TASK_CLS = SeqVaeLagAttnTrfPatchTask
    CHECKPOINT_STEM = "lag-attn-trf-patch"

    TRACKED_METRICS: Tuple[str, ...] = LagAttnTrfRwsTrainer.TRACKED_METRICS + (
        "train/anchors_per_sample",
        "val/anchors_per_sample",
        "val/kld_source_null",
    )

    def causal_standing_message(self) -> str:
        r = int(self.config["model_config"]["VAE_model"].get("raw_per_step", 16))
        return (
            f"causal standing: patch token t reads raw samples {r}t..{r}t+{r - 1} only, so the "
            f"inputs carry no group delay and the warm-up covers only the encoder's conv stem."
        )

    def create_model(self) -> None:
        super().create_model()
        model = self.pytorch_model
        floor, stride = int(model.warmup_period), int(model.anchor_stride)
        a_max = -(-(int(model.anchor_ceiling) - floor) // stride)
        logger.info(f"anchor geometry: H={model.horizon}, S={stride}, F={floor}, A_max={a_max}")
        self.apply_config_hyperparameters(
            {"seed": (self.config.get("general_config") or {}).get("seed")}, self.pl_model
        )

    @classmethod
    def preflight(cls, config: Dict[str, Any]) -> None:
        vae_config = (config.get("model_config") or {}).get("VAE_model") or {}
        offenders = [
            k for k in vae_config
            if k in REMOVED_MODEL_KEYS or k.startswith(REMOVED_MODEL_PREFIXES)
        ]
        if offenders:
            raise ValueError(
                f"model_config.VAE_model carries {offenders}: they configure the ST/PH feature "
                f"representation this patch model replaces and would be dropped in silence. "
                f"Remove them."
            )
        _check_raw_source_normalized(config)
        _check_raw_length_against_shard(config)
        _check_phase_key_fields(config)
        _check_raw_target_fields(config, fields=cls.TARGET_FIELDS)
        _check_boundary_term_is_off(config)


def main(config_path: str) -> None:
    """Run the shared entry point with this package's driver."""
    run_training(config_path, trainer_cls=LagAttnTrfPatchTrainer)


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root."""
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(_REPO_ROOT, config_path)


#: Config used when launched with no ``--config`` (an IDE Run button). Repo-root relative.
RUN_CONFIG: str | None = "teb_vae/lag_attn_transformer_patch/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. "
        "teb_vae/lag_attn_transformer_patch/configs/tiny.yaml. Run from the repo root. Optional "
        "only if RUN_CONFIG is set in this file.",
    )
    _args = parser.parse_args()

    _config_path = _args.config or RUN_CONFIG
    if _config_path is None:
        parser.error("--config is required (or set RUN_CONFIG in this file).")

    _config_path = _resolve_cli_config_path(_config_path)

    # Paths inside a config are repo-root-relative too.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
