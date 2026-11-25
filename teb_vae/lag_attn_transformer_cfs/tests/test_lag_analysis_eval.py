"""The lag-analysis pipeline (``eval/lag_analysis``) end to end on a tiny model and the committed fixture.

One run of four steps on the CPU, then a second call that must skip them. It checks what a reader
of a results folder relies on: the five outputs exist, every statistic carries its definition, unit,
step and arm under the key ``<id>@<block>``, the JSON is strict, and a finished step is not redone.
Then the same run stopped after its first batch and resumed: every estimate, interval and curve must
equal the uninterrupted run's.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis.__main__ import main

from .conftest import CAUSAL_SHARD, absolutize_dataset_paths, make_task, tiny_warmup_kwargs

STEPS = ("gain", "shift", "clamp", "raw")


def test_the_pipeline_writes_a_self_describing_folder_and_resumes(tmp_path, capsys):
    # A tiny-width model at the fixture's real length (300 steps), saved as the trainer saves it,
    # with the resolved config beside it so the loader settings are found without --config.
    module = make_task(model_kwargs=tiny_warmup_kwargs(sequence_length=300))
    blob = {"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}
    module.on_save_checkpoint(blob)
    checkpoint = tmp_path / "run" / "model_checkpoints" / "tiny.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(blob, checkpoint)
    tiny = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    (checkpoint.parent / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["--checkpoint", str(checkpoint), "--shards", str(CAUSAL_SHARD),
            "--stats", str(Path(CAUSAL_SHARD).parent / "tiny_stats_causal.hdf5"), "--out", str(out),
            "--steps", ",".join(STEPS), "--k-samples", "2", "--num-workers", "0", "--device", "cpu", "--batch-size", "2"]

    assert main(argv) == 0

    for name in ("run_info.json", "lag_analysis.json", "LAG_ANALYSIS.md", "guid_table.csv", "figures"):
        assert (out / name).exists(), name
    def refuse(token):
        raise AssertionError(f"lag_analysis.json is not strict JSON: {token}")
    result = json.loads((out / "lag_analysis.json").read_text(), parse_constant=refuse)
    stats = result["statistics"]
    assert {"gain.sg_rw@all", "shift.W50@all", "clamp.pos@all"} <= set(stats)
    for key, row in stats.items():
        assert key == f"{row['id']}@{row['block']}" and row["step"] in STEPS
        assert row["definition"] and row["unit"] and row["arm"] and "n_guids" in row and "ci95" in row
    assert sorted(result["steps_run"]) == sorted(STEPS) and "heads" in result["steps_skipped"]
    info = json.loads((out / "run_info.json").read_text())
    assert info["n_guids"] == 4 and info["n_segments"] == 4 and len(info["checkpoint"]["sha256"]) == 64
    assert set(STEPS) <= set(info["seconds_per_step"])
    text = (out / "LAG_ANALYSIS.md").read_text(encoding="utf-8")
    assert text.index("## How to read this folder") < text.index("## Headline numbers") and "recorded = stored - 20 s" in text
    table = (out / "guid_table.csv").read_text().splitlines()
    assert len(table) == 5 and table[0].startswith("guid,subgroup") and "gain.g_all" in table[0]
    with np.load(out / "steps" / "gain.npz") as step:
        assert step["g_all"].dtype == np.float32 and step["g_all"].shape == (4,) and np.isfinite(step["g_all"]).all()

    # A second call finds every step file and runs no batch.
    stamps = {p.name: p.stat().st_mtime_ns for p in (out / "steps").iterdir()}
    capsys.readouterr()
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert all(f"[{step}] done before, skipped" in printed for step in STEPS) and "batch " not in printed
    assert stamps == {p.name: p.stat().st_mtime_ns for p in (out / "steps").iterdir()}

    # Stopped after the first of two batches, as a kill would stop it, then resumed from the progress file.
    resumed = argv[:7] + [str(tmp_path / "resumed")] + argv[8:] + ["--save-every", "1"]
    with pytest.raises(SystemExit, match="stopped after batch 1"):
        main(resumed + ["--stop-after-batches", "1"])
    progress = tmp_path / "resumed" / "steps" / "progress.state"
    assert progress.exists() and not (tmp_path / "resumed" / "steps" / "gain.npz").exists()
    capsys.readouterr()
    assert main(resumed) == 0
    assert "resuming after batch 1/2" in capsys.readouterr().out and not progress.exists()
    again = json.loads((tmp_path / "resumed" / "lag_analysis.json").read_text())
    assert again["statistics"] == stats and again["curves"] == result["curves"]
