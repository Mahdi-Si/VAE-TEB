r"""One full pipeline run, end to end, against the fitted checkpoint with retention caps on.

Everything else in the suite drives one seam at a time; this file is the pass an operator makes
-- the fitted checkpoint, the committed override delta repointed at generated shards, every
analysis selected, retention caps on so the opt-in figures render -- and it asserts the *shape*
of what a run leaves behind: an ``ok`` step record from every registered analysis, the complete
artifact layout, and the opt-in figure families actually rendered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from teb_vae.lag_attn_rws.eval import collect, preflight, probe as probe_module
from teb_vae.lag_attn_rws.eval import run as run_module
from teb_vae.lag_attn_rws.tests.conftest import write_repointed_overrides

pytestmark = pytest.mark.slow

#: Grouped-variant figures are a *family*, not fixed filenames: the runner fans one violin per
#: cohort axis over whatever each analysis declared, so the set grows with the analyses.
GROUPED_SUFFIXES = ("_by_clinical_class.pdf", "_by_subgroup.pdf")

#: Retention caps for this run, all small: the opt-in figures -- the forecast overlay, the lag
#: heatmap, the per-sample pages -- render only where something was retained.
SMOKE_CAPS = {"waveforms": 4, "attention": 2, "pages": 4}


@pytest.fixture(scope="session")
def smoke_run(fitted_run, forecastable_shards, tmp_path_factory) -> Dict[str, Any]:
    """The full pipeline against the fitted checkpoint, with retention caps on."""
    overrides = write_repointed_overrides(
        tmp_path_factory.mktemp("smoke_overrides"), forecastable_shards
    )
    delta = yaml.safe_load(overrides.read_text(encoding="utf-8"))
    delta["eval_config"]["caps"] = dict(SMOKE_CAPS)
    overrides.write_text(yaml.safe_dump(delta, sort_keys=False), encoding="utf-8")

    output_dir = tmp_path_factory.mktemp("smoke_eval")
    exit_code = run_module.main(
        fitted_run,
        output_dir,
        overrides=overrides,
        device="cpu",
        num_samples=2,
    )
    results_dir = Path(output_dir) / run_module.RESULTS_DIRNAME
    summary = json.loads(
        (results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    return {"exit_code": exit_code, "results_dir": results_dir, "summary": summary}


# =============================================================================
# The run itself
# =============================================================================
def test_every_registered_analysis_completes_with_exit_code_zero(smoke_run):
    """Every selectable analysis, the unskippable channel map, and the loader probe: a step each,
    every one ok. A registry entry with no step record is an analysis the run silently lost."""
    assert smoke_run["exit_code"] == 0
    assert smoke_run["summary"]["failed"] == []
    steps = {record["name"]: record["status"] for record in smoke_run["summary"]["steps"]}

    expected = {"probe", *run_module.UNSKIPPABLE_ANALYSES, *run_module.ANALYSIS_FUNCTIONS}
    assert expected <= set(steps), sorted(expected - set(steps))
    assert all(status == "ok" for status in steps.values()), steps


def test_the_complete_artifact_layout_is_present(smoke_run):
    """The durable artifact set, by name: the summary and its heartbeat, the two preflight-side
    records, the dumped config and the log, the two durable tables with their sidecars, and the
    unskippable channel map's two files."""
    results_dir = smoke_run["results_dir"]

    for name in (
        run_module.SUMMARY_FILENAME,
        run_module.STEPS_FILENAME,
        preflight.PREFLIGHT_FILENAME,
        probe_module.PROBE_FILENAME,
        "resolved_config.yaml",
        run_module.LOG_FILENAME,
        "per_sample.csv",
        "per_anchor.parquet",
        collect.COLLECTION_FILENAME,
        "band_partition.json",
        "band_channel_map.csv",
    ):
        assert (results_dir / name).is_file(), f"the run left no {name}"

    subdirectories = {path.name for path in results_dir.iterdir() if path.is_dir()}
    missing = set(run_module.ANALYSIS_FUNCTIONS) - subdirectories
    assert missing == set(), f"no artifact subdirectory for {sorted(missing)}"


def test_the_opt_in_families_actually_rendered(smoke_run):
    """The caps were set, so the run must contain what they buy: an opt-in figure that silently
    stopped rendering is otherwise noticed only by the operator who needed it."""
    results_dir = smoke_run["results_dir"]

    assert list(results_dir.glob("samples/*/*.pdf")), "no per-sample pages rendered"
    grouped = [
        path for path in results_dir.rglob("*.pdf") if path.name.endswith(GROUPED_SUFFIXES)
    ]
    assert grouped, "no grouped variants rendered against a multi-class split"
    assert (results_dir / "forecast" / "forecast_overlay.pdf").is_file()
    assert (results_dir / "attention" / "lag_heatmap.pdf").is_file()

