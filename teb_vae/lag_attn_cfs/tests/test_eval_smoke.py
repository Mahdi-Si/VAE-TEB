r"""One full pipeline run, end to end: the pass an operator makes, and the figure manifest it fixes.

Everything else in this suite drives one seam at a time. This file asserts the **shape** of what a
run leaves behind -- the complete artifact layout, a step record from every registered analysis, an
exit code, a coverage block that says which population each analysis actually saw, and a figure tree
equal to the committed ``eval/figure_manifest.json``.

It is the test that catches an analysis which passes its own unit test and fails inside a full run,
and the one that catches an analysis which runs, returns a block, and writes nothing to disk.

**The manifest is the bridge into the fast gate.** ``tests/test_eval_docs.py`` binds every figure to
a ``FIGURE_GUIDE.md`` entry, but it runs under ``-m "not slow"`` and cannot afford this run -- so
this file keeps the committed manifest equal to what a real run produces and the fast tests read the
manifest. Drift fails here in both directions: a figure a run stopped emitting leaves a stale row,
and a new figure is missing from it. Regenerate after a deliberate figure change by deleting
``eval/figure_manifest.json`` and running this file once; it seeds the manifest from the run it just
made and fails asking for a review, and the next run passes against the committed copy.

**It starts no run of its own.** The session-scoped ``collected_run`` fixture is the suite's one
end-to-end pass -- every artifact-level assertion in this package reads that same run -- and a
second one here would double the most expensive thing the suite does to assert the same shapes.
:func:`test_this_file_starts_no_run_of_its_own` is what keeps that from quietly changing.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from teb_vae.lag_attn_cfs.eval import collect, preflight, probe as probe_module
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING

pytestmark = pytest.mark.slow

#: The committed manifest this run must equal.
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "eval" / "figure_manifest.json"

#: Grouped-variant figures are a *family*, not fixed filenames: the runner fans one violin per cohort
#: axis over whatever each analysis declared, so the set grows with the analyses and the guide
#: documents them as a family. Normalised out of the per-analysis lists by suffix.
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
        "guide_marker": "The per-recording pages",
    },
}

#: The durable artifact set, by name: the summary and its heartbeat, the two preflight-side
#: records, the dumped config and the log, the two durable tables with their sidecar, and the
#: unskippable channel map's three files -- the declared-axis map, the kept-axis map that
#: ``spectral_skill`` joins through, and the partition record itself.
DURABLE_ARTIFACTS = (
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
    "band_channel_map_kept.csv",
)


def _registry() -> Dict[str, Any]:
    """This cell's analyses: the shared registry with the binding's own merged in."""
    return run_module.merged_analysis_functions(CFS_BINDING)


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


# =================================================================================================
# The run itself
# =================================================================================================
def test_the_full_run_completes_with_exit_code_zero(collected_run) -> None:
    """The failed steps are named with their errors rather than left to a bare ``1 == 0``: this run
    is the most expensive thing the suite does, so a failure that does not say which analysis
    raised buys a second one."""
    failed = [
        f"{record['name']}: {record.get('error')}"
        for record in collected_run["summary"]["steps"]
        if record["status"] != "ok"
    ]

    assert failed == [], failed
    assert collected_run["exit_code"] == 0


def test_every_registered_analysis_contributes_a_step_record(collected_run) -> None:
    """Every selectable analysis, the unskippable channel map, and the loader probe: a step each,
    every one ok. A registry entry with no step record is an analysis the run silently lost.

    The three this cell alone has -- ``warmup``, ``source_null`` and ``spectral_skill`` -- are in
    the expected set by being registered on the binding, not by being named here, so registering a
    fourth reaches this assertion from one place.
    """
    steps = {record["name"]: record["status"] for record in collected_run["summary"]["steps"]}

    expected = {"probe", *run_module.UNSKIPPABLE_ANALYSES, *_registry()}
    assert expected <= set(steps), sorted(expected - set(steps))
    assert {"warmup", "source_null", "spectral_skill"} <= set(steps)
    assert all(status == "ok" for status in steps.values()), steps


def test_the_complete_artifact_layout_is_present(collected_run) -> None:
    """The durable artifact set by name, and one subdirectory per analysis that did not skip.

    "Did not skip" rather than "every analysis", because a skip is a legitimate outcome that the
    fixture deliberately provokes: ``tiny.yaml`` trains under ``likelihood: mse``, whose decoder
    log-variance head is never fitted, so ``calibration`` records a skip and writes nothing. What
    must hold either way is the pair -- an analysis wrote artifacts, or it said in its own block
    why it did not. An analysis that did neither is one the run silently lost.
    """
    results_dir = Path(collected_run["results_dir"])
    results = collected_run["summary"]["results"]

    for name in DURABLE_ARTIFACTS:
        assert (results_dir / name).is_file(), f"the run left no {name}"

    subdirectories = {path.name for path in results_dir.iterdir() if path.is_dir()}
    silent: Set[str] = {
        name for name in _registry()
        if name not in subdirectories
        and not (results.get(name) or {}).get("skipped")
    }
    assert silent == set(), f"no artifact subdirectory and no recorded skip for {sorted(silent)}"
    # Non-vacuity: the assertion above holds over a run where every analysis skipped, which is not
    # a run worth asserting anything about.
    wrote = subdirectories & set(_registry())
    assert len(wrote) > len(set(_registry()) - wrote), (
        f"only {sorted(wrote)} wrote artifacts; the rest recorded skips, so this run demonstrates "
        f"the skip path rather than the pipeline"
    )


def test_no_directory_is_left_by_an_analysis_this_package_does_not_have(collected_run) -> None:
    """``coherence`` is not ported at all -- a stored scattering coefficient is a modulus, so phase
    agreement and group delay have no analogue at any window length -- and ``spectral_skill`` is
    what this domain has instead. A directory by the other name would mean a reader could carry the
    raw pipeline's contract across."""
    subdirectories = {
        path.name for path in Path(collected_run["results_dir"]).iterdir() if path.is_dir()
    }

    assert "coherence" not in subdirectories
    assert "spectral_skill" in subdirectories


# =================================================================================================
# The coverage block
# =================================================================================================
def test_the_coverage_block_reports_a_population_per_uncapped_analysis(collected_run) -> None:
    """Effective-$n$ per analysis is what reveals two analyses having run on different populations
    -- a forecast scored over the whole split beside an uplift scored over a capped draw of one
    shard reconcile with each other only by coincidence, and nothing else in the output shows it.

    ``None`` is the third state and it is load-bearing: an analysis that scored **no** population
    -- the data-describing channel map, or a ``calibration`` that skipped because this checkpoint
    was trained under ``mse`` -- reports ``None`` rather than ``0``, and the warning below compares
    only the ones that reported a number. A zero there would read as a population of zero and make
    every scoring analysis look like a disagreement with it.
    """
    per_analysis = collected_run["summary"]["results"]["coverage"]["per_analysis"]

    assert per_analysis, "the coverage block recorded no analysis at all"
    assert all(
        {"n_samples", "composition", "capped"} <= set(record) for record in per_analysis.values()
    ), per_analysis

    scored = {
        name: record["n_samples"]
        for name, record in per_analysis.items()
        if not record["capped"] and record["n_samples"] is not None
    }
    assert scored, "no uncapped analysis reported a population, so the block tests nothing"
    assert all(isinstance(value, int) and value > 0 for value in scored.values()), scored
    # Never zero: a skip and a data-describing step report None, and the two states must stay
    # distinguishable in the artifact.
    assert 0 not in {record["n_samples"] for record in per_analysis.values()}


def test_a_population_disagreement_is_a_warning_and_not_a_failure(collected_run) -> None:
    """The exit code is non-zero **if and only if a step raised**. Two analyses over two
    populations is a reading hazard rather than a broken run, so it is recorded and logged and
    deliberately does not move the code -- which is exactly why the offline gate exists separately
    and refuses on the sanity block."""
    coverage = collected_run["summary"]["results"]["coverage"]

    assert isinstance(coverage["warnings"], list)
    assert collected_run["exit_code"] == 0
    assert collected_run["summary"]["failed"] == []


# =================================================================================================
# The one pass
# =================================================================================================
def test_this_file_starts_no_run_of_its_own() -> None:
    """The suite performs exactly one end-to-end pass and every artifact assertion reads it. A
    second ``main`` call here would double the most expensive thing this suite does in order to
    assert the same shapes -- and would do it invisibly, because both would be green."""
    source = Path(__file__).read_text(encoding="utf-8")

    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "main"
    ]

    assert calls == [], "this file calls main(); read the session-scoped run instead"


# =================================================================================================
# The committed figure manifest
# =================================================================================================
def test_the_opt_in_families_actually_rendered(collected_run) -> None:
    """The shipped caps are on in this run, so it must contain what they buy -- otherwise the
    manifest it seeds would silently exempt exactly the opt-in figures whose axes are easiest to
    misread, and the documentation contract would never reach them."""
    results_dir = Path(collected_run["results_dir"])

    assert list(results_dir.glob("samples/*/*.pdf")), "no per-recording pages rendered"
    grouped = [
        path for path in results_dir.rglob("*.pdf") if path.name.endswith(GROUPED_SUFFIXES)
    ]
    assert grouped, "no grouped variants rendered against a multi-cohort split"


def test_the_committed_manifest_equals_what_the_run_produced(collected_run) -> None:
    """Both directions at once: a stale row and a missing row are the same failure. When the manifest
    does not exist yet it is seeded from this run and the test fails asking for a review -- committing
    a file nobody has read is not a contract."""
    observed = observed_figures(Path(collected_run["results_dir"]))

    if not MANIFEST_PATH.is_file():
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Every figure a full evaluation run of this cell emits, by analysis "
                        "directory, with dynamically-named figure families recorded as families. "
                        "Kept equal to a real run by tests/test_eval_smoke.py; read by the fast "
                        "documentation tests in tests/test_eval_docs.py. Regenerate by deleting "
                        "this file and running the smoke suite once."
                    ),
                    "figures": observed,
                    "families": FAMILIES,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        pytest.fail(f"seeded {MANIFEST_PATH} from this run; review it, commit it, and re-run.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["figures"] == observed, (
        "the committed figure manifest disagrees with what a real run produces; if the change is "
        "deliberate, delete eval/figure_manifest.json and re-run this suite to reseed it"
    )
    assert manifest["families"] == FAMILIES


def test_the_manifest_names_no_analysis_this_package_does_not_have() -> None:
    """Read from the committed file rather than from the run, so it holds in the fast gate too:
    ``coherence`` is not ported at all and a manifest row under that name would put a figure into
    the documentation contract that nothing here can draw."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert "coherence" not in manifest["figures"]
    assert set(manifest["figures"]) <= {".", *run_module.UNSKIPPABLE_ANALYSES, *_registry()}
