r"""S7-T06: the cross-arm gate table ``arms_report_v4``.

The table must render one row per configured arm with the v4 gate columns, degrade a missing per-arm
``metrics.json`` to ``n/a`` (never a crash), compute the derived ``pred_gain`` difference, and be
written once at the tag root by the model-free ``arms_report`` stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_report_v4 as ar
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v4 as rp

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


def _metrics(*, gamma: float, null_pass: bool, ordering: bool,
             model_class: str = "SeqVaeRawV4") -> Dict[str, Any]:
    r"""A minimal but schema-faithful v4 metrics dict carrying every tabulated column."""
    return {
        "model_class": model_class, "arm": "x", "render_mode": "direct",
        "calibration": {"gamma": gamma, "r2": 0.9, "spearman": 1.0},
        "null_cell_gate": {"mean": 0.01 if null_pass else 0.2, "ceiling": 0.05, "pass": null_pass},
        "prediction_controls": {"overall": {
            "feat_loss": 0.5, "base_loss": 0.7,
            "shuffle_penalty_shuffled": 0.15, "ordering_pass_shuffled": ordering}},
        "lag_recovery": {"mean_lag_mass": 0.9, "mean_lag_mass_pass": True},
        "te_raw_gate": {"gate": {"passed": True}},
    }


def _write_arm(tag_root: Path, arm: str, split: str, metrics: Dict[str, Any]) -> None:
    d = tag_root / arm / split
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle)


def test_dig_dotted_and_derived_difference() -> None:
    r"""``_dig`` resolves a dotted path and an ``"a - b"`` derived difference; missing -> None."""
    obj = {"a": {"b": 2.0}, "c": {"d": 0.5}}
    assert ar._dig(obj, "a.b") == 2.0
    assert ar._dig(obj, "a.b - c.d") == pytest.approx(1.5)
    assert ar._dig(obj, "a.missing") is None
    assert ar._dig(None, "a.b") is None


def test_render_value_verdicts() -> None:
    r"""Gate columns render pass/FAIL; numeric columns use the n/a-safe formatter."""
    assert ar._render_value("null_gate", True) == "pass"
    assert ar._render_value("null_gate", False) == "**FAIL**"
    assert ar._render_value("gamma", None) == "n/a"
    assert ar._render_value("gamma", 0.812345).startswith("0.812")


def test_build_table_has_row_per_arm_and_columns(tmp_path) -> None:
    r"""Two graded arms -> a header with every column + one row each; pred_gain is the derived diff."""
    tag_root = tmp_path / "G1_raw_v4"
    _write_arm(tag_root, "prod", "val", _metrics(gamma=0.8, null_pass=True, ordering=True))
    _write_arm(tag_root, "frontend_noncausal", "val",
               _metrics(gamma=0.1, null_pass=False, ordering=False,
                        model_class="LeakyRawFrontendSeqVaeRawV4"))

    md = ar.build_arms_report_v4(["prod", "frontend_noncausal"], tag_root, split="val", tag="G1_raw_v4")
    # Header carries the constant column labels.
    for label, _ in ar.ARMS_REPORT_COLUMNS:
        assert label in md
    # One row per arm (linked to its report).
    assert "[`prod`]" in md and "[`frontend_noncausal`]" in md
    # The G0 control fails the null gate; prod passes.
    assert "**FAIL**" in md and "pass" in md
    # pred_gain = base_loss - feat_loss = 0.7 - 0.5 = 0.2 rendered.
    assert "0.2" in md


def test_missing_arm_renders_na(tmp_path) -> None:
    r"""An arm with no metrics.json degrades to an n/a row + a 'not graded' note, not a crash."""
    tag_root = tmp_path / "G1_raw_v4"
    _write_arm(tag_root, "prod", "val", _metrics(gamma=0.8, null_pass=True, ordering=True))
    md = ar.build_arms_report_v4(["prod", "single_stride"], tag_root, split="val")
    assert "not graded" in md
    assert "n/a" in md
    assert "[`prod`]" in md  # prod still tabulated


def test_no_arm_graded_note(tmp_path) -> None:
    r"""When no arm is graded the table degrades to a single note."""
    md = ar.build_arms_report_v4(["prod", "single_stride"], tmp_path / "G1_raw_v4", split="test")
    assert "no arm has been graded" in md


def test_stage_registered_model_free_nonfatal() -> None:
    r"""The ``arms_report`` stage is registered, model-free (no --arm), non-fatal."""
    assert "arms_report" in rp._STAGE_REGISTRY_V4
    spec = rp._STAGE_REGISTRY_V4["arms_report"]
    assert spec.model_dependent is False
    assert spec.fatal is False


def test_stage_writes_report_at_tag_root(tmp_path) -> None:
    r"""The stage entry point writes ``arms_report_v4.md`` once at the tag root."""
    config = rp.load_config(str(_CONFIG_PATH))
    config = {**config, "paths": {**config.get("paths", {}), "results_dir": str(tmp_path)},
              "experiment": {**config.get("experiment", {}), "tag": "G1_raw_v4"}}
    tag_root = tmp_path / "G1_raw_v4"
    _write_arm(tag_root, "prod", "val", _metrics(gamma=0.8, null_pass=True, ordering=True))

    ctx = rp.StageContextV4(config=config, benchmark="G1_raw_v4", arm=None, split="val")
    rc = ar.run_arms_report_v4(ctx)
    assert rc == 0
    assert (tag_root / "arms_report_v4.md").is_file()
