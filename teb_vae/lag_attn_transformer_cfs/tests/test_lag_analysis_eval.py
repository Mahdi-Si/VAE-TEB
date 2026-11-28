"""The lag-analysis pipeline (``eval/lag_analysis``) end to end on a tiny model and the committed fixture.

One run of four steps on the CPU, then a second call that must skip them. It checks what a reader
of a results folder relies on: the five outputs exist, every statistic carries its definition, unit,
step and arm under the key ``<id>@<block>``, the JSON is strict, and a finished step is not redone.
Then the same run stopped after its first batch and resumed: every estimate, interval and curve must
equal the uninterrupted run's.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis import report
from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis.__main__ import main
from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis.steps import first_opening

from .conftest import CAUSAL_SHARD, absolutize_dataset_paths, make_task, tiny_warmup_kwargs

STEPS = ("gain", "shift", "clamp", "raw")
STATS = str(Path(CAUSAL_SHARD).parent / "tiny_stats_causal.hdf5")


def _checkpoint(tmp_path):
    """A tiny-width model at the fixture's real length (300 steps), saved as the trainer saves it, with
    the resolved config beside it so the loader settings are found without --config."""
    module = make_task(model_kwargs=tiny_warmup_kwargs(sequence_length=300))
    blob = {"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}
    module.on_save_checkpoint(blob)
    checkpoint = tmp_path / "run" / "model_checkpoints" / "tiny.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(blob, checkpoint)
    tiny = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    (checkpoint.parent / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return checkpoint


def _argv(checkpoint, shard, out, steps):
    return ["--checkpoint", str(checkpoint), "--shards", str(shard), "--stats", STATS, "--out", str(out),
            "--steps", ",".join(steps), "--k-samples", "2", "--num-workers", "0", "--device", "cpu", "--batch-size", "2"]


def test_the_pipeline_writes_a_self_describing_folder_and_resumes(tmp_path, capsys):
    checkpoint = _checkpoint(tmp_path)
    out = tmp_path / "out"
    argv = _argv(checkpoint, CAUSAL_SHARD, out, STEPS)

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
    assert text.index("## How to read this folder") < text.index("## Headline numbers") and "Seconds are on the stored clock" in text
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


def test_run_args_become_the_command_line_the_parser_reads():
    """The Run-button dict: lists expand, flags appear only when True, out defaults beside the run."""
    from teb_vae.lag_attn_transformer_cfs.eval.lag_analysis import __main__ as cli

    run_args = dict(cli.RUN_ARGS, checkpoint="/runs/r1/model_checkpoints/m-epoch=3.ckpt",
                    shards=["/a/*.hdf5", "/b.hdf5"], max_guids=5, force=True)
    args = cli.parse(cli.run_args_argv(run_args))

    assert Path(args.out) == Path("/runs/r1/lag_analysis/m-epoch=3")
    assert args.shards == ["/a/*.hdf5", "/b.hdf5"] and args.max_guids == 5 and args.force
    assert args.batch_size == 16 and args.config is None


ALL_STEPS = ("gain", "shift", "heads", "clamp", "clock", "events", "groups", "raw")


def test_a_healthy_only_run_heads_on_all_reads_past_the_reward_window_and_fabricates_nothing(tmp_path):
    """Named subgroup shards with healthy recordings only (the training run's validation set): every
    non-IRF step runs, the empty `adverse` block never becomes the headline, an empty class gives no
    estimate, the all-anchor and 0-180 s rows exist, the folder refuses a step file of another run, and
    the copy for sending is written."""
    checkpoint = _checkpoint(tmp_path)
    shard = tmp_path / "shards" / "healthy_bg_cs" / "shard_000.hdf5"
    shard.parent.mkdir(parents=True)
    shutil.copy(CAUSAL_SHARD, shard)
    out = tmp_path / "out"
    argv = _argv(checkpoint, shard, out, ALL_STEPS)

    assert main(argv) == 0

    result = json.loads((out / "lag_analysis.json").read_text())
    stats = result["statistics"]
    assert result["headline_block"] == "all"
    assert {row["block"] for row in stats.values()} >= {"all", "healthy"}
    assert not any(row["block"] == "adverse" for row in stats.values())
    for key in ("gain.sg_all", "gain.sg_outside_rw", "gain.sg_post_60_180", "gain.sg_quiet", "gain.gain_matched_all",
                "gain.gain_surA_all", "gain.gain_surB_all", "gain.gain_null_all", "events.peak_0_180",
                "events.peak_specific_0_180", "events.peak_height_over_null_0_180", "shift.W50_all", "clamp.pos_all"):
        assert f"{key}@all" in stats, key
    assert stats["gain.sg_outside_rw@all"]["n_guids"] > 0 and stats["gain.sg_outside_rw@all"]["estimate"] is not None
    for c in ("acidosis", "hie", "healthy_no_bg"):          # classes without a recording
        for sid in (f"shift.W50.{c}", f"shift.delta0.{c}"):
            row = stats[f"{sid}@{c}"]
            assert row["n_guids"] == 0 and row["estimate"] is None and row["ci95"] is None, sid
    for key, row in stats.items():                          # no estimate is made up from no recording
        if row["n_guids"] == 0:
            assert row["estimate"] is None, key
    text = (out / "LAG_ANALYSIS.md").read_text(encoding="utf-8")
    assert "Headline block of this run: `all`" in text and "`gain.sg_outside_rw`" in text
    send = out.with_name(out.name + "_send")
    assert (send / "lag_analysis.json").exists() and not list(send.rglob("*.png"))

    # A step file of another run in the folder: a run of another step must not merge it into the report.
    with np.load(out / "steps" / "clamp.npz") as z:
        arrays = dict(z)
    np.savez_compressed(out / "steps" / "clamp.npz", **{**arrays, "key": "another-run"})
    with pytest.raises(SystemExit, match="one report cannot mix them"):
        main(_argv(checkpoint, shard, out, ("gain",)))


def test_the_weighted_median_is_the_midpoint_and_empty_draws_give_nan():
    v = np.array([4.0, 1.0, 3.0, 2.0, np.nan])
    W = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 1], [1, 1, 1, 0, 0]], float)

    got = report.wmedian(v, W)

    assert got[0] == 2.5 and got[1] == 1.5 and np.isnan(got[2]) and got[3] == 3.0


def test_peak_and_w50_give_nan_for_a_draw_without_data_and_no_nan_offset():
    x = np.arange(5.0)
    Y = np.array([[np.nan] * 5, [0.0, 1.0, 3.0, 1.0, 0.0], [np.nan, 1.0, 3.0, np.nan, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0]])

    loc, height = report.parabolic_peak(Y, x)

    assert np.isnan(loc[0]) and np.isnan(height[0])
    assert loc[1] == 2.0 and height[1] == 3.0
    assert loc[2] == 2.0 and height[2] == 3.0                # a NaN neighbour: the bin itself, not NaN
    assert loc[3] == 3.0 and height[3] == 3.0                # maximum at the edge: no parabola opens downwards
    w = report.w50(np.array([[np.nan, np.nan], [0.8, 0.2], [0.9, 0.8]]), np.array([4.0, 8.0]))
    assert np.isnan(w[0]) and w[1] == 6.0 and w[2] == 8.0


def test_figures_draw_an_interval_that_misses_its_estimate_and_skip_a_missing_one(tmp_path):
    def row(sid, est, ci):
        return {"id": sid, "step": sid.split(".")[0], "block": "all", "estimate": est, "ci95": ci, "reference": None}

    rows = [row("clamp.pos", 1.0, [1.5, 2.0]), row("clamp.static", float("nan"), None), row("clamp.rev", 0.2, [-0.5, 0.1])]

    made = report.figures({}, rows, tmp_path, "all")

    assert "clamp.png" in made and (tmp_path / "figures" / "clamp.png").exists()


def test_a_first_opening_needs_a_full_follow_up_opened_or_not():
    shut, keys = np.zeros(40), np.ones(40, bool)
    opened = shut.copy()
    opened[14] = 1.0

    assert first_opening([10], shut[:20], keys[:20])[1].tolist() == [0, 0, 0]     # 9 anchors of follow-up: censored
    assert first_opening([10], opened[:20], keys[:20])[1].tolist() == [0, 0, 0]   # opened inside it: still censored
    assert first_opening([10], shut, keys)[1].tolist() == [1, 0, 1]
    hist, cnt = first_opening([10], opened, keys)
    assert cnt.tolist() == [1, 1, 0] and hist[3] == 1


def test_stats_gain_keeps_surb_rw_distinct_from_surb_all():
    r"""``gain.sg_surB_rw`` reports ``bm_rw - bs_rw`` even when ``am_out`` / ``bm_all`` are present."""
    n = 4
    z = np.zeros(n, dtype=np.float32)
    d = {
        "guid": np.array([f"g{i}" for i in range(n)]),
        "subgroup": np.array(["healthy_bg_cs"] * n),
        "am_rw": np.full(n, 0.40, dtype=np.float32),
        "as_rw": np.full(n, 0.10, dtype=np.float32),
        "am_all": np.full(n, 0.25, dtype=np.float32),
        "as_all": np.full(n, 0.05, dtype=np.float32),
        "am_out": np.full(n, 0.12, dtype=np.float32),
        "as_out": np.full(n, 0.02, dtype=np.float32),
        "am_post": np.full(n, 0.14, dtype=np.float32),
        "as_post": np.full(n, 0.04, dtype=np.float32),
        "am_quiet": np.full(n, 0.11, dtype=np.float32),
        "as_quiet": np.full(n, 0.03, dtype=np.float32),
        "bm_rw": np.full(n, 0.60, dtype=np.float32),
        "bs_rw": np.full(n, 0.10, dtype=np.float32),
        "bm_all": np.full(n, 0.17, dtype=np.float32),
        "bs_all": np.full(n, 0.10, dtype=np.float32),
        "g0_rw": z,
        "g0_all": z,
        "am_blk": np.zeros((n, 2), dtype=np.float32),
        "as_blk": np.zeros((n, 2), dtype=np.float32),
        "am_tau": np.zeros((n, 4), dtype=np.float32),
        "as_tau": np.zeros((n, 4), dtype=np.float32),
    }
    rows, curves = report.Rows(), {}
    report.stats_gain(d, rows, curves)
    by_id = {(r["id"], r["block"]): r for r in rows}

    assert by_id[("gain.sg_surB_rw", "all")]["estimate"] == pytest.approx(0.50)
    assert by_id[("gain.sg_surB_all", "all")]["estimate"] == pytest.approx(0.07)
    assert by_id[("gain.sg_surB_rw", "healthy")]["estimate"] == pytest.approx(0.50)
    assert by_id[("gain.sg_surB_all", "healthy")]["estimate"] == pytest.approx(0.07)
