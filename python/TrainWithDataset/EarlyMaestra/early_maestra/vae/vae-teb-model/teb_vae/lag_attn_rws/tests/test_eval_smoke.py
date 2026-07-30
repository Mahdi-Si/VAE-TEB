r"""One full pipeline run, end to end, and the committed figure manifest it must equal.

Everything else in the suite drives one seam at a time; this file is the pass an operator makes
-- the fitted checkpoint, the committed override delta repointed at generated shards, every
analysis selected, retention caps on so the opt-in figures render -- and it asserts the *shape*
of what a run leaves behind: the complete artifact layout, a step record from every registered
analysis, and a figure tree equal to the committed ``eval/figure_manifest.json``.

The manifest is the bridge into the fast gate. The documentation tests bind every figure to a
``FIGURE_GUIDE.md`` entry, but they run under ``-m "not slow"`` and cannot afford this run -- so
this test keeps the committed manifest equal to what a real run produces, and the fast tests
read the manifest. Drift in either direction fails here: a figure a run stopped emitting leaves
a stale manifest row, and a new figure is missing from it.

Regenerate after a deliberate figure change by deleting ``eval/figure_manifest.json`` and
running this file once: the test seeds the manifest from the run it just made and fails asking
for a review, and the next run passes against the committed copy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from teb_vae.lag_attn_rws.eval import collect, preflight, probe as probe_module
from teb_vae.lag_attn_rws.eval import run as run_module
from teb_vae.lag_attn_rws.tests.conftest import write_repointed_overrides

pytestmark = pytest.mark.slow

#: The committed manifest this run must equal.
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "eval" / "figure_manifest.json"

#: Grouped-variant figures are a *family*, not fixed filenames: the runner fans one violin per
#: cohort axis over whatever each analysis declared, so the set grows with the analyses and the
#: guide documents them as a family. Normalised out of the per-analysis lists by suffix.
GROUPED_SUFFIXES = ("_by_clinical_class.pdf", "_by_subgroup.pdf")

#: The families the manifest records instead of filenames, each with the marker string its
#: ``FIGURE_GUIDE.md`` entry must contain -- which is what the fast documentation test checks.
FAMILIES: Dict[str, Dict[str, str]] = {
    "grouped_variants": {
        "pattern": "*_by_clinical_class.pdf and *_by_subgroup.pdf, beside the table each resolves",
        "guide_marker": "_by_clinical_class.pdf",
    },
    "sample_pages": {
        "pattern": "samples/<selection>/sample<index>_<guid>_epoch<epoch>.pdf",
        "guide_marker": "The per-sample pages",
    },
}

#: Retention caps for this run, all small: the opt-in figures -- the forecast overlay, the lag
#: heatmap -- render only where something was retained, and a manifest built from a run without
#: them would exempt exactly the figures whose axes are easiest to misread.
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


def observed_figures(results_dir: Path) -> Dict[str, List[str]]:
    """Every figure the run emitted, grouped by analysis directory, families normalised out."""
    figures: Dict[str, List[str]] = {}
    for pdf in sorted(results_dir.rglob("*.pdf")):
        relative = pdf.relative_to(results_dir).as_posix()
        if relative.startswith("samples/"):
            continue
        if pdf.name.endswith(GROUPED_SUFFIXES):
            continue
        parts = relative.split("/")
        analysis = parts[0] if len(parts) > 1 else "."
        figures.setdefault(analysis, []).append(pdf.name)
    return {analysis: sorted(names) for analysis, names in sorted(figures.items())}


# =============================================================================
# The run itself
# =============================================================================
def test_the_full_run_completes_with_exit_code_zero(smoke_run):
    assert smoke_run["exit_code"] == 0
    assert smoke_run["summary"]["failed"] == []


def test_every_registered_analysis_contributes_a_step_record(smoke_run):
    """Every selectable analysis, the unskippable channel map, and the loader probe: a step each,
    every one ok. A registry entry with no step record is an analysis the run silently lost."""
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
    """The caps were set, so the run must contain what they buy -- otherwise the manifest this
    run seeds would silently exempt the opt-in figures from the documentation contract."""
    results_dir = smoke_run["results_dir"]

    assert list(results_dir.glob("samples/*/*.pdf")), "no per-sample pages rendered"
    grouped = [
        path for path in results_dir.rglob("*.pdf") if path.name.endswith(GROUPED_SUFFIXES)
    ]
    assert grouped, "no grouped variants rendered against a multi-class split"
    assert (results_dir / "forecast" / "forecast_overlay.pdf").is_file()
    assert (results_dir / "attention" / "lag_heatmap.pdf").is_file()


# =============================================================================
# The committed manifest
# =============================================================================
def test_the_committed_manifest_equals_what_the_run_produced(smoke_run):
    """Both directions at once: a stale row and a missing row are the same failure. When the
    manifest does not exist yet, it is seeded from this run and the test fails asking for a
    review -- committing a file no one has read is not a contract."""
    observed = observed_figures(smoke_run["results_dir"])

    if not MANIFEST_PATH.is_file():
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Every figure a full evaluation run emits, by analysis directory, "
                        "with dynamically-named figure families recorded as families. Kept "
                        "equal to a real run by tests/test_eval_smoke.py; read by the fast "
                        "documentation tests in tests/test_eval_docs.py. Regenerate by "
                        "deleting this file and running the smoke suite once."
                    ),
                    "figures": observed,
                    "families": FAMILIES,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pytest.fail(
            f"seeded {MANIFEST_PATH} from this run; review it, commit it, and re-run."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["figures"] == observed, (
        "the committed figure manifest disagrees with what a real run produces; if the change "
        "is deliberate, delete eval/figure_manifest.json and re-run this suite to reseed it"
    )
    assert manifest["families"] == FAMILIES
