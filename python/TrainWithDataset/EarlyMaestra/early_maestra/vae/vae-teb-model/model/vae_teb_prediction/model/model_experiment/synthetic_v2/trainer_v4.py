r"""S4-T01/T03: the synthetic training driver for ``synthetic_v4``.

Trains :class:`SeqVaeRawV4` (or the leaky negative-control subclass) on the synthetic raw cache by
reusing :class:`GraphModelVaeTebRawV4Trainer` **byte-unchanged** through its public interface: the
base ``train_model`` *receives* its loaders as arguments, so :func:`run_synthetic_training` mirrors
the model_raw ``main()`` and simply substitutes :class:`SyntheticRawDataModuleV4` loaders for the
HDF5 ones. Two seams the base does not expose are overridden here:

* :class:`SyntheticSeqVaeRawV4Pl` stamps the per-arm ``arm`` / ``render_mode`` provenance onto every
  checkpoint (the base already stamps ``model_class`` + ``model_kwargs``);
* :meth:`SyntheticGraphModelVaeTebRawV4Trainer.create_model` mirrors the base body but selects the
  leaky model class for the ``frontend_noncausal`` arm and wraps in the provenance-stamping Pl class.

The trainer reads its YAML from disk (in ``GraphModelBase.__init__``), so the driver writes the
arm-resolved (and optionally pilotized) config to ``resolved_config.yaml`` under the run dir and
constructs the trainer from that path.
"""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.arms_v4 import (
    arm_uses_leaky_frontend,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v4 import (
    SyntheticRawDataModuleV4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
    LeakyRawFrontendSeqVaeRawV4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    GraphModelVaeTebRawV4Trainer,
    SeqVaeRawV4,
    SeqVaeRawV4Pl,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)
from model.vae_teb_prediction.model.model_raw.trainer_raw_v4 import warm_start_from_v3

logger = logging.getLogger(__name__)


class SyntheticSeqVaeRawV4Pl(SeqVaeRawV4Pl):
    r"""``SeqVaeRawV4Pl`` that additionally stamps ``arm`` / ``render_mode`` onto each checkpoint."""

    def __init__(self, base_model, *, arm: Optional[str] = None,
                 render_mode: Optional[str] = None, **kwargs) -> None:
        r"""Store the per-arm provenance and defer to the base Lightning wrapper.

        Args:
            base_model: The wrapped :class:`SeqVaeRawV4` (or leaky subclass).
            arm: The arm name to stamp into checkpoints.
            render_mode: The render mode (``direct`` / ``am_carrier``) to stamp.
            **kwargs: Forwarded to :class:`SeqVaeRawV4Pl` (``lr``, ``lr_milestones``,
                ``model_kwargs``, ...).
        """
        self._arm = arm
        self._render_mode = render_mode
        super().__init__(base_model, **kwargs)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        r"""Stamp ``model_class`` / ``model_kwargs`` (base) plus ``arm`` / ``render_mode``."""
        super().on_save_checkpoint(checkpoint)
        checkpoint["arm"] = self._arm
        checkpoint["render_mode"] = self._render_mode


class SyntheticGraphModelVaeTebRawV4Trainer(GraphModelVaeTebRawV4Trainer):
    r"""Reused raw trainer with per-arm model-class selection + provenance-stamping Pl wrapper.

    Only :meth:`create_model` is overridden (the base exposes no model-class / Pl-class hook); every
    other trainer method -- ``_build_model_kwargs``, ``train_model``, ``_resolve_beta``, the spike
    breaker, ``_select_ddp_strategy``, all callbacks -- is inherited unchanged.
    """

    def __init__(self, config_file_path: Optional[str] = None, *,
                 arm: Optional[str] = None, render_mode: Optional[str] = None) -> None:
        r"""Store the arm / render mode, then defer to the base (which reads the YAML from disk)."""
        self._arm = arm
        self._render_mode = render_mode
        super().__init__(config_file_path=config_file_path)

    def _select_model_class(self) -> Type[SeqVaeRawV4]:
        r"""``LeakyRawFrontendSeqVaeRawV4`` for the ``frontend_noncausal`` arm, else ``SeqVaeRawV4``."""
        if self._arm is not None and arm_uses_leaky_frontend(self.config, self._arm):
            return LeakyRawFrontendSeqVaeRawV4
        return SeqVaeRawV4

    def create_model(self) -> None:
        r"""Mirror :meth:`GraphModelVaeTebRawV4Trainer.create_model`, swapping the model + Pl class.

        The two changed lines vs. the base are the ``model_cls`` construction and the
        :class:`SyntheticSeqVaeRawV4Pl` wrap (which carries ``arm`` / ``render_mode``); everything
        else -- warm-start, core-checkpoint reload, ``trainer_hparams``,
        ``apply_config_hyperparameters`` -- is identical to the base.
        """
        model_kwargs = self._build_model_kwargs()
        model_cls = self._select_model_class()
        _fe = model_kwargs.get("frontend", {}) or {}
        logger.info(
            "Building %s with kwargs: %s, frontend=<%d keys>",
            model_cls.__name__,
            ", ".join(f"{k}={v}" for k, v in model_kwargs.items() if k != "frontend"),
            len(_fe),
        )
        self.pytorch_model = model_cls(**model_kwargs)
        if not self.pytorch_model.causal_norm:
            logger.warning(
                "causal_norm=False: the encoders' GroupNorm pools statistics across time, so "
                "H_y[t] depends on Y[>t] and kld_raw is NOT a transfer-entropy surrogate."
            )

        model_cfg = self.config.get("model_config", {}) or {}
        vae_cfg = model_cfg.get("VAE_model", {}) or {}

        warm_start = model_cfg.get("warm_start_from")
        if warm_start:
            warm_start_from_v3(self.pytorch_model, str(warm_start))

        self.checkpoint = model_cfg.get("core_model_checkpoint")
        if self.checkpoint is not None:
            from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import check_model_class
            from train.graph_models_utils import load_checkpoint_strict

            blob = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            check_model_class(blob, type(self.pytorch_model).__name__)
            if load_checkpoint_strict(model=self.pytorch_model, checkpoint=blob) is None:
                raise RuntimeError(
                    f"could not align core_model_checkpoint {self.checkpoint!r} into "
                    f"{type(self.pytorch_model).__name__} (no matching module keys)."
                )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "kld_beta": vae_cfg.get("kld_beta", 0.01),
            "lambda_full": vae_cfg.get("lambda_full", 1.0),
            "lambda_base": vae_cfg.get("lambda_base", 0.5),
            "beta_schedule": vae_cfg.get("beta_schedule"),
            "likelihood": vae_cfg.get("likelihood", "gaussian_nll"),
            "sigma_obs": vae_cfg.get("sigma_obs", "learned"),
            "free_bits": vae_cfg.get("free_bits", 0.0),
            "detach_baseline_in_full": vae_cfg.get("detach_baseline_in_full", True),
            "lambda_lag": vae_cfg.get("lag_smoothness_lambda", 0.0),
            "lambda_lp": vae_cfg.get("lambda_lp", 0.5),
            "lambda_smooth": vae_cfg.get("lambda_smooth", 0.1),
            "lowpass_scales": vae_cfg.get("lowpass_scales", [4, 16, 32, 60]),
            "loss_spike_skip": vae_cfg.get("loss_spike_skip", {}) or {},
        }
        self.pl_model = SyntheticSeqVaeRawV4Pl(
            self.pytorch_model,
            arm=self._arm,
            render_mode=self._render_mode,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            model_kwargs=model_kwargs,
        )
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)


# ===========================================================================
# Config transforms + driver
# ===========================================================================
def _pilotize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    r"""Shrink a v4 config to a small-but-production-geometry pilot (fast CPU/1-GPU smoke).

    Sets the model widths to the ``SMALL_PROD_*`` values (full $5280$ geometry, ~a few hundred k
    params), a couple of epochs, a small batch, and disables MLflow + raw plotting for headless runs.
    Returns a deep copy; the input is untouched.
    """
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        SMALL_PROD_FRONTEND,
        SMALL_PROD_V3_KWARGS,
    )

    cfg = copy.deepcopy(config)
    vae = cfg.setdefault("model_config", {}).setdefault("VAE_model", {})
    # Preserve the arm's frontend-defining deltas (single_stride -> stages:[16], no_antialias ->
    # antialias:false, no_gated -> gated:false, norm_kind) before shrinking to the small_prod
    # widths -- a wholesale replace would silently erase those ablations in a --pilot sweep, making
    # the ablation arms architecturally identical to prod. Mirrors test_arm_build_v4._small_model_kwargs.
    arm_frontend = dict(vae.get("frontend") or {})
    vae.update(SMALL_PROD_V3_KWARGS)
    frontend = dict(SMALL_PROD_FRONTEND)
    for key in ("stages", "antialias", "gated", "norm_kind"):
        if key in arm_frontend:
            frontend[key] = arm_frontend[key]
    if frontend.get("stages") == [16]:
        frontend.pop("channels", None)  # single_stride: let the len(stages)==1 branch pick channels
    vae["frontend"] = frontend

    gen = cfg.setdefault("general_config", {})
    gen["epochs"] = 2
    gen["batch_size"] = {"train": 4, "test": 4}

    # Single-process loading for the pilot smoke (avoids fragile DataLoader-worker spawn under
    # Windows/pytest; the headline prod run keeps the config's worker count).
    cfg.setdefault("dataset_config", {}).setdefault("dataloader_config", {})["num_workers"] = 0

    adv = cfg.setdefault("advanced_config", {})
    adv.setdefault("tracking", {}).setdefault("mlflow", {})["enabled"] = False
    adv.setdefault("callbacks", {}).setdefault("raw_plotting", {})["enabled"] = False
    return cfg


def run_synthetic_training(
    config: Dict[str, Any],
    *,
    benchmark: str,
    arm: Optional[str],
    render_mode: str,
    run_dir: Path,
    cache_dir: Optional[Path] = None,
    pilot: bool = False,
) -> Dict[str, Any]:
    r"""Train one arm on the synthetic raw cache and write ``final.ckpt`` (+ ``best.ckpt``).

    Mirrors ``model_raw.trainer_raw_v4.main`` but substitutes :class:`SyntheticRawDataModuleV4`
    loaders (the base ``train_model`` receives its loaders as arguments).

    Args:
        config: The (arm-resolved) config tree.
        benchmark: Active benchmark key under ``benchmarks``.
        arm: The arm name (selects the model class; stamped into checkpoints).
        render_mode: The render mode (stamped into checkpoints).
        run_dir: The per-arm results dir (``results/<tag>/<arm>/``); receives the checkpoints.
        cache_dir: Optional explicit cache dir (defaults to the config-resolved cache).
        pilot: Shrink the model + epochs for a fast smoke run.

    Returns:
        ``{"final_ckpt": Path, "best_ckpt": Optional[Path], "train_results_dir": str}``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _pilotize_config(config) if pilot else copy.deepcopy(config)
    # Route the trainer's internal artifacts under the per-arm run dir.
    cfg.setdefault("general_config", {}).setdefault("folders_config", {})["out_dir_base"] = str(
        run_dir / "_train"
    )
    resolved_path = run_dir / "resolved_config.yaml"
    with open(resolved_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    trainer = SyntheticGraphModelVaeTebRawV4Trainer(
        config_file_path=str(resolved_path), arm=arm, render_mode=render_mode,
    )
    trainer.setup_config()
    trainer.create_model()

    batch_size = int(cfg["general_config"]["batch_size"]["train"])
    dm = SyntheticRawDataModuleV4(cfg, batch_size=batch_size, benchmark=benchmark, cache_dir=cache_dir)
    dm.setup("fit")
    pl_trainer = trainer.train_model(dm.train_dataloader(), dm.val_dataloader())

    final_ckpt = run_dir / "final.ckpt"
    pl_trainer.save_checkpoint(str(final_ckpt))

    best_ckpt: Optional[Path] = None
    best_path = getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", "")
    if best_path and Path(best_path).is_file():
        best_ckpt = run_dir / "best.ckpt"
        shutil.copy(str(best_path), str(best_ckpt))

    logger.info("run_synthetic_training[%s]: wrote %s", arm, final_ckpt)
    return {
        "final_ckpt": final_ckpt,
        "best_ckpt": best_ckpt,
        "train_results_dir": trainer.train_results_dir,
    }


def run_train_v4(ctx: StageContextV4) -> int:
    r"""``train`` stage: train the active arm on the synthetic raw cache."""
    config = ctx.config
    benchmark = ctx.benchmark
    render_mode = str(config["benchmarks"][benchmark]["raw"].get("render_mode", "direct"))
    out = run_synthetic_training(
        config, benchmark=benchmark, arm=ctx.arm, render_mode=render_mode,
        run_dir=ctx.run_dir(), pilot=ctx.pilot,
    )
    print(f"[train] arm={ctx.arm} render_mode={render_mode} -> {out['final_ckpt']}")
    if out["best_ckpt"] is not None:
        print(f"[train] best -> {out['best_ckpt']}")
    return 0


register_stage_v4(StageSpecV4(
    name="train",
    run=run_train_v4,
    order=40,
    model_dependent=True,
    fatal=True,
    help="train SeqVaeRawV4 on the synthetic raw cache (per arm); writes final.ckpt",
))
