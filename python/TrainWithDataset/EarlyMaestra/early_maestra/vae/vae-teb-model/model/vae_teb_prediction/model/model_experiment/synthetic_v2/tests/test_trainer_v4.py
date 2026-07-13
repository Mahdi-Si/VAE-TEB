r"""S4-T01 tests: the synthetic trainer builds the model + emits the full v4 metric dict."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import pytest

pytestmark = pytest.mark.v4


def _build_trainer(config: Dict[str, Any], cache_dir: Path, tmp_path: Path,
                   *, arm: Optional[str] = "prod", render_mode: str = "direct"):
    r"""Pilotize + write the config, then construct a synthetic trainer with a built model."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        SyntheticGraphModelVaeTebRawV4Trainer,
        _pilotize_config,
    )

    cfg = _pilotize_config(config)
    cfg["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "_train")
    cfg_path = tmp_path / "resolved.yaml"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    trainer = SyntheticGraphModelVaeTebRawV4Trainer(
        config_file_path=str(cfg_path), arm=arm, render_mode=render_mode,
    )
    trainer.setup_config()
    trainer.create_model()
    return trainer, cfg


def test_trainer_v4_step_emits_finite_metrics_and_anneals_beta(tiny_cache_v4, tmp_path):
    r"""One `compute_loss_and_metrics` emits the full finite v4 metric dict; beta anneals up."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v4 import (
        SyntheticRawDataModuleV4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import SeqVaeRawV4

    trainer, cfg = _build_trainer(tiny_cache_v4["config"], tiny_cache_v4["cache_dir"], tmp_path)
    assert isinstance(trainer.pytorch_model, SeqVaeRawV4)

    dm = SyntheticRawDataModuleV4(cfg, batch_size=4, cache_dir=tiny_cache_v4["cache_dir"])
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    trainer.pl_model.eval()
    with torch.no_grad():
        loss, metrics = trainer.pl_model.compute_loss_and_metrics(batch, 0, "train")

    assert math.isfinite(float(loss))
    for key in ("total_loss", "feat_loss", "base_loss", "kld_raw", "lowpass_loss",
                "smooth_loss", "mean_logvar_full", "mean_logvar_base", "pred_gap"):
        assert key in metrics, key
        assert math.isfinite(float(metrics[key])), key

    # linear_warmup with start<end -> non-decreasing beta.
    betas = [trainer.pl_model._resolve_beta(e) for e in (0, 25, 50)]
    assert betas[0] <= betas[1] <= betas[2]
    assert betas[0] < betas[2]  # genuinely anneals up
