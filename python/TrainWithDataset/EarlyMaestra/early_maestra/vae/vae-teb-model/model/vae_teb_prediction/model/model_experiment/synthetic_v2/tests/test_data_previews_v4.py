r"""S1-T05: the model-free data-preview stage (headless) writes per-cell overlays + a summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    data_previews_v4,
    run_pipeline_v4,
)

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


def _config(tmp_results: Path) -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["benchmarks"]["G1_raw_v4"]["mix"]["inverter"]["n_samples"] = 2000
    cfg["paths"]["results_dir"] = str(tmp_results)
    return cfg


@pytest.fixture(scope="module")
def previews(tmp_path_factory):
    r"""Run the preview stage once into a temp results dir (pilot grid, small n)."""
    tmp = tmp_path_factory.mktemp("previews_results")
    cfg = _config(tmp)
    ctx = run_pipeline_v4.StageContextV4(config=cfg, benchmark="G1_raw_v4", pilot=True)
    rc = data_previews_v4.run_data_previews_v4(ctx)
    out_dir = ctx.results_dir() / "data_previews"
    return {"rc": rc, "out_dir": out_dir}


def test_stage_returns_zero(previews) -> None:
    r"""The stage runs headless and returns 0."""
    assert previews["rc"] == 0


def test_one_figure_per_cell_level(previews) -> None:
    r"""At least one preview figure is written per cell level (pilot grid has 4 levels)."""
    figs = sorted(previews["out_dir"].glob("preview_cell*.png"))
    assert len(figs) == 4
    for f in figs:
        assert f.stat().st_size > 0


def test_summary_written_with_coupling_scores(previews) -> None:
    r"""``previews_summary.json`` carries a coupling score per cell."""
    summary = json.loads((previews["out_dir"] / "previews_summary.json").read_text("utf-8"))
    assert len(summary["rows"]) == 4
    for r in summary["rows"]:
        assert "coupling_score" in r


def test_null_and_strong_te_are_distinguishable(previews) -> None:
    r"""A null cell and the strongest-TE cell are separable by the coupling-score statistic."""
    summary = json.loads((previews["out_dir"] / "previews_summary.json").read_text("utf-8"))
    null_rows = [r for r in summary["rows"] if r["is_null"]]
    signal_rows = [r for r in summary["rows"] if not r["is_null"]]
    assert null_rows and signal_rows
    strongest = max(signal_rows, key=lambda r: r["te_inj"])
    assert strongest["coupling_score"] > 3.0 * null_rows[0]["coupling_score"]


def test_stage_is_registered() -> None:
    r"""Importing the module registers the ``data_previews`` stage."""
    assert "data_previews" in run_pipeline_v4.stage_names_v4()
