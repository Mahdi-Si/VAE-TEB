r"""S7-T05: the per-arm/per-split ``final_report_v4`` report assembly.

The report must render from a fabricated ``metrics.json`` (the ``synth_metrics_v4`` fixture),
auto-link the ``visualize_v4`` gallery, degrade a broken section to an ``n/a`` note rather than
aborting, and emit an explicit "not graded" report when ``metrics.json`` is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v4 as fr
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v4 as rp

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


def _config() -> dict:
    return rp.load_config(str(_CONFIG_PATH))


def test_report_renders_from_metrics(synth_metrics_v4) -> None:
    r"""The report renders every headline gate + the figure gallery from a fixture metrics.json."""
    run_dir = synth_metrics_v4["run_dir"]
    report = fr.final_report_v4(_config(), benchmark="G1_raw_v4", arm="prod",
                                out_dir=run_dir, split="val")
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    # Headline sections present.
    for heading in ("## Provenance", "## Headline gates", "## Calibration by lag",
                    "## Figure gallery"):
        assert heading in text, f"missing heading {heading}"
    # Gate values rendered (γ from the fixture, null verdict, ordering verdict).
    assert "calibration slope" in text
    assert "pass" in text.lower()
    # Figures actually rendered and linked.
    figs = list((run_dir / "figures").glob("*.pdf"))
    assert len(figs) == 5, f"expected 5 figures, got {[f.name for f in figs]}"
    assert "kbar_vs_te" in text


def test_report_missing_metrics_is_not_graded(tmp_path) -> None:
    r"""With no metrics.json the report renders an explicit 'not graded' note (never raises)."""
    report = fr.final_report_v4(_config(), benchmark="G1_raw_v4", arm="prod",
                                out_dir=tmp_path / "prod" / "val", split="val")
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "not been graded" in text
    # No figures dir needed when ungraded.
    assert not (tmp_path / "prod" / "val" / "figures").exists()


def test_broken_section_degrades_to_na(synth_metrics_v4) -> None:
    r"""A section that raises degrades to an n/a note rather than aborting the report."""
    # Corrupt the calibration block so the by-lag section's int() coercion raises.
    metrics = dict(synth_metrics_v4["metrics"])
    metrics["calibration"] = dict(metrics["calibration"])
    metrics["calibration"]["by_lag"] = {"not-an-int": {"gamma": 1.0}}
    run_dir = synth_metrics_v4["run_dir"]
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle)

    report = fr.final_report_v4(_config(), benchmark="G1_raw_v4", arm="prod",
                                out_dir=run_dir, split="val")
    text = report.read_text(encoding="utf-8")
    # The report still assembles, and the broken section carries an n/a note.
    assert "## Headline gates" in text
    assert "n/a" in text


def test_report_stage_registered() -> None:
    r"""The ``report`` stage is registered, per-arm, non-fatal."""
    assert "report" in rp._STAGE_REGISTRY_V4
    spec = rp._STAGE_REGISTRY_V4["report"]
    assert spec.model_dependent is True
    assert spec.fatal is False


def test_report_stage_runs_off_context(synth_metrics_v4) -> None:
    r"""The stage entry point writes report.md into the context's split-scoped output dir."""
    run_dir = synth_metrics_v4["run_dir"]  # already <.../prod>
    # Build a context whose output_dir() resolves to the fixture's run_dir (arm=prod, split=None).
    config = _config()
    config = {**config, "paths": {**config.get("paths", {}),
                                   "results_dir": str(run_dir.parent.parent)}}
    # results_dir/<tag>/prod == run_dir requires tag == run_dir.parent.name; align via experiment.
    config = {**config, "experiment": {**config.get("experiment", {}),
                                       "tag": run_dir.parent.name}}
    ctx = rp.StageContextV4(config=config, benchmark="G1_raw_v4", arm="prod", split=None)
    assert ctx.output_dir() == run_dir
    rc = fr.run_report_v4(ctx)
    assert rc == 0
    assert (run_dir / "report.md").is_file()
