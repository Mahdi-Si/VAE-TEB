r"""S1-T03: the realizability pre-flight writes a per-cell table and a passing loose gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    realizability_v4,
    run_pipeline_v4,
)

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


def _config(tmp_results: Path | None = None) -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["benchmarks"]["G1_raw_v4"]["mix"]["inverter"]["n_samples"] = 2000
    if tmp_results is not None:
        cfg["paths"]["results_dir"] = str(tmp_results)
    return cfg


@pytest.fixture(scope="module")
def report() -> dict:
    r"""Run the realizability probe over the pilot grid once (reduced MC)."""
    return realizability_v4.compute_realizability(_config(), benchmark="G1_raw_v4", pilot=True)


def test_one_row_per_cell(report: dict) -> None:
    r"""The pilot grid ([0,1,2,3] x D=8) yields one row per cell with a boolean null flag."""
    assert len(report["rows"]) == 4
    assert sum(1 for r in report["rows"] if r["is_null"]) == 1
    for r in report["rows"]:
        assert set(r) >= {"cell_id", "te_inj", "D", "te_raw", "frac", "is_null"}


def test_loose_gate_passes_on_concentrated_cells(report: dict) -> None:
    r"""The loose gate passes: signal $>0$, monotone across the ladder, null below ceiling."""
    gate = report["gate"]
    assert gate["passed"] is True
    assert gate["signal_positive"] is True
    assert gate["null_below_ceiling"] is True
    assert gate["monotone"] is True


def test_constants_recorded_as_s6_seed(report: dict) -> None:
    r"""The observed frac range + null ceiling are persisted to seed the tight Sprint-6 gate."""
    const = report["constants"]
    for key in ("seed_frac_lo", "seed_frac_hi", "seed_null_ceiling",
                "observed_frac_lo", "observed_frac_hi", "observed_null_te_max"):
        assert key in const
    assert const["observed_frac_hi"] >= const["observed_frac_lo"]


def test_stage_writes_realizability_json(tmp_path: Path) -> None:
    r"""The registered stage writes a parseable ``realizability.json`` and returns 0."""
    cfg = _config(tmp_results=tmp_path)
    ctx = run_pipeline_v4.StageContextV4(config=cfg, benchmark="G1_raw_v4", pilot=True)
    rc = realizability_v4.run_realizability_v4(ctx)
    assert rc == 0
    out = ctx.results_dir() / "realizability.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["render_mode"] == "direct"
    assert data["gate"]["passed"] is True
    assert len(data["rows"]) == 4


def test_stage_is_registered() -> None:
    r"""Importing the stage module registers it into the v4 registry."""
    assert "realizability" in run_pipeline_v4.stage_names_v4()
