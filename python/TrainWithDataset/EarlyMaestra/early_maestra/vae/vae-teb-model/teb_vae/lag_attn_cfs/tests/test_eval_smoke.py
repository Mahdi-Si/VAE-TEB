r"""One full pipeline run, end to end: the pass an operator makes.

Everything else in this suite drives one seam at a time. This file asserts the **shape** of what a
run leaves behind -- the complete artifact layout, a step record from every registered analysis, an
exit code, and a coverage block that says which population each analysis actually saw.

It is the test that catches an analysis which passes its own unit test and fails inside a full run,
and the one that catches an analysis which runs, returns a block, and writes nothing to disk.

It starts no run of its own: the session-scoped ``collected_run`` fixture is the suite's one
end-to-end pass, and every artifact-level assertion in this package reads that same run.
:data:`DURABLE_ARTIFACTS` is imported by the transformer and slot cells' smoke suites.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

import pytest

from teb_vae.lag_attn_cfs.eval import collect, preflight, probe as probe_module
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING

pytestmark = pytest.mark.slow


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
    every one ok. A registry entry with no step record is an analysis the run silently lost. The
    binding's own analyses are in the expected set by being registered on it, not by being named
    here.
    """
    steps = {record["name"]: record["status"] for record in collected_run["summary"]["steps"]}

    expected = {"probe", *run_module.UNSKIPPABLE_ANALYSES, *_registry()}
    assert expected <= set(steps), sorted(expected - set(steps))
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
