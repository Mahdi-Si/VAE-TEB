r"""S4-T05 tests (slow): a pilot end-to-end fit writes final.ckpt; one batch overfits."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

import pytest

pytestmark = [pytest.mark.v4, pytest.mark.slow]


def test_pilot_train_writes_final_ckpt_and_metrics(tiny_cache_v4, tmp_path):
    r"""`run_synthetic_training(pilot=True)` writes an arm-stamped final.ckpt + a metrics CSV."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        run_synthetic_training,
    )

    run_dir = tmp_path / "prod"
    out = run_synthetic_training(
        tiny_cache_v4["config"], benchmark="G1_raw_v4", arm="prod", render_mode="direct",
        run_dir=run_dir, cache_dir=tiny_cache_v4["cache_dir"], pilot=True,
    )
    final = out["final_ckpt"]
    assert Path(final).is_file()

    blob = torch.load(str(final), map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeRawV4"
    assert blob["arm"] == "prod"
    assert blob["render_mode"] == "direct"

    # The metrics CSV surface carries the v4 metric set.
    csvs = list(Path(out["train_results_dir"]).rglob("*.csv"))
    assert csvs, "no metrics CSV was written"
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in csvs)
    for token in ("feat_loss", "base_loss", "kld_raw", "mean_logvar_full"):
        assert token in text, token


def test_overfit_one_batch_decreases(tiny_cache_v4, tmp_path):
    r"""A manual AdamW loop over one batch drives total_loss well below its initial value."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v4 import (
        SyntheticRawDataModuleV4,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        SyntheticGraphModelVaeTebRawV4Trainer,
        _pilotize_config,
    )

    cfg = _pilotize_config(tiny_cache_v4["config"])
    cfg["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "_train")
    cfg_path = tmp_path / "resolved.yaml"
    with open(cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    trainer = SyntheticGraphModelVaeTebRawV4Trainer(
        config_file_path=str(cfg_path), arm="prod", render_mode="direct",
    )
    trainer.setup_config()
    trainer.create_model()

    dm = SyntheticRawDataModuleV4(cfg, batch_size=4, cache_dir=tiny_cache_v4["cache_dir"])
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    pl_model = trainer.pl_model
    pl_model.train()
    opt = torch.optim.AdamW(pl_model.parameters(), lr=3e-3)

    initial = float(pl_model.compute_loss_and_metrics(batch, 0, "train")[1]["total_loss"])
    for _ in range(40):
        opt.zero_grad()
        loss, _m = pl_model.compute_loss_and_metrics(batch, 0, "train")
        loss.backward()
        opt.step()
    final = float(pl_model.compute_loss_and_metrics(batch, 0, "train")[1]["total_loss"])
    assert final < 0.7 * initial, (initial, final)
