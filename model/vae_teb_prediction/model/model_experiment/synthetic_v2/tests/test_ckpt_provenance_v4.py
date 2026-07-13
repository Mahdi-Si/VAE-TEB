r"""S4-T03 tests: per-arm checkpoints round-trip with arm / render_mode provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import pytest

pytestmark = pytest.mark.v4


def _build_trainer(config: Dict[str, Any], out_base: Path,
                   *, arm: Optional[str], render_mode: str):
    r"""Pilotize + write the config, then construct a synthetic trainer with a built model."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        SyntheticGraphModelVaeTebRawV4Trainer,
        _pilotize_config,
    )

    cfg = _pilotize_config(config)
    cfg["general_config"]["folders_config"]["out_dir_base"] = str(out_base / "_train")
    cfg_path = out_base / "resolved.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    trainer = SyntheticGraphModelVaeTebRawV4Trainer(
        config_file_path=str(cfg_path), arm=arm, render_mode=render_mode,
    )
    trainer.setup_config()
    trainer.create_model()
    return trainer


def _save_checkpoint(trainer, path: Path) -> None:
    r"""Save a checkpoint via the Pl subclass's ``on_save_checkpoint`` (no full fit needed)."""
    import torch

    checkpoint: Dict[str, Any] = {"state_dict": trainer.pl_model.state_dict()}
    trainer.pl_model.on_save_checkpoint(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))


def test_ckpt_provenance_roundtrip_two_arms(tiny_cache_v4, tmp_path):
    r"""Two arms save to distinct dirs; each checkpoint carries arm/render_mode + rebuilds."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.arms_v4 import (
        arm_uses_leaky_frontend,
        resolve_arm_v4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
        LeakyRawFrontendSeqVaeRawV4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import SeqVaeRawV4

    base_config = tiny_cache_v4["config"]
    results = {}
    for arm, render_mode in (("prod", "direct"), ("disable_source", "direct")):
        arm_cfg = resolve_arm_v4(base_config, arm)
        run_dir = tmp_path / arm
        trainer = _build_trainer(arm_cfg, run_dir, arm=arm, render_mode=render_mode)
        ckpt_path = run_dir / "final.ckpt"
        _save_checkpoint(trainer, ckpt_path)
        results[arm] = ckpt_path

    # Distinct dirs.
    assert results["prod"].parent != results["disable_source"].parent

    for arm in ("prod", "disable_source"):
        blob = torch.load(str(results[arm]), map_location="cpu", weights_only=False)
        assert blob["model_class"] == "SeqVaeRawV4"
        assert blob["arm"] == arm
        assert blob["render_mode"] == "direct"
        # model_kwargs rebuilds the model.
        cfg = resolve_arm_v4(base_config, arm)
        cls = LeakyRawFrontendSeqVaeRawV4 if arm_uses_leaky_frontend(cfg, arm) else SeqVaeRawV4
        model = cls(**blob["model_kwargs"])
        assert model is not None
    # disable_source arm actually set the flag through to the model kwargs.
    ds_blob = torch.load(str(results["disable_source"]), map_location="cpu", weights_only=False)
    assert ds_blob["model_kwargs"].get("disable_source") is True
