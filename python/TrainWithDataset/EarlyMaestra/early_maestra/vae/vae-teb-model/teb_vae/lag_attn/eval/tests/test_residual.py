r"""Tests for the residual-usage analysis.

The two cases mirror each other and both are needed.

A ``perturb_full_pathway`` model must **not** fire the collapse flag. That is the case that
catches an analysis hardcoded to report collapse, or one reading the wrong tensor -- and it is
the one an untouched-model test cannot catch, because on an untouched model
``residual_decoder.mean_head`` is zero and every residual is legitimately zero.

An untouched model must fire it, which is what proves the flag is wired to anything at all.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval.analyses import residual as residual_analysis


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return ``(summary, per-sample frame, per-anchor frame)``."""
    torch.manual_seed(11)
    summary = residual_analysis.run_residual_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir,
        probe={"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4},
    )
    directory = Path(output_dir) / residual_analysis.ANALYSIS_DIRNAME
    return (
        summary,
        pd.read_csv(directory / "per_sample.csv"),
        pd.read_csv(directory / "per_anchor.csv"),
    )


def test_a_live_pathway_does_not_fire_the_collapse_flag(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The case that catches an analysis hardcoded to report collapse."""
    runner = make_eval_runner(perturb=True, output_dir=tmp_path / "runner")
    summary, frame, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "live"
    )

    assert summary["mean_residual_ratio"] > summary["collapse_threshold"]
    assert summary["collapsed"] is False
    assert frame["residual_ratio"].to_numpy().min() > 0.0


def test_an_untouched_model_fires_the_collapse_flag(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""``_zero_init_delta_heads`` zeroes the residual mean head, so $\delta\mu_{src} \equiv 0$.

    Which is exactly what a collapsed source pathway looks like, and what the flag must catch.
    """
    runner = make_eval_runner(perturb=False, output_dir=tmp_path / "runner")
    summary, frame, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "dead"
    )

    assert frame["residual_ratio"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    assert summary["collapsed"] is True


def test_the_per_sample_and_per_anchor_files_are_both_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Per-anchor is not redundant: it localises a pathway that dies partway through."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _, frame, per_anchor = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "files"
    )

    assert {"residual_rms", "forecast_rms", "residual_ratio"} <= set(frame.columns)
    # The per-anchor columns are melted out of the per-sample frame rather than left in it.
    assert not [name for name in frame.columns if name.startswith("a") and name[1:].isdigit()]

    anchors = int(300 - runner.model.horizon)
    assert set(per_anchor["anchor"]) == set(range(anchors))
    assert len(per_anchor) == 4 * anchors


def test_the_per_anchor_trace_is_nan_across_the_warmup(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """A zero there would read as a genuinely inactive prefix rather than as masked-out data."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _, _, per_anchor = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "warmup"
    )

    warmup = int(runner.model.warmup_period)
    early = per_anchor[per_anchor["anchor"] < warmup]["residual_rms"]
    assert len(early) > 0
    assert bool(early.isna().all())
    assert bool(per_anchor[per_anchor["anchor"] >= warmup]["residual_rms"].notna().any())


def test_the_collapse_threshold_comes_from_the_config(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Shared with the run-level health probe, so the two cannot disagree about "collapsed"."""
    runner = make_eval_runner(perturb=True, output_dir=tmp_path / "runner")
    config = dict(tiny_eval_config["eval_config"], health_probe_floor=1e9)
    summary, _, _ = _run(runner, tiny_loader, config, tmp_path / "threshold")

    assert summary["collapse_threshold"] == 1e9
    assert summary["collapsed"] is True, "an absurd floor must make even a live pathway fire"


def test_the_figure_is_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary, _, _ = _run(
        runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figure"
    )
    assert Path(summary["figure"]).suffix == ".pdf"
    assert Path(summary["figure"]).stat().st_size > 0
