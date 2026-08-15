r"""One full pipeline run of this cell, end to end: the pass an operator makes.

The causal cell's smoke test with one substitution -- the run is this cell's -- and that
substitution is the whole point. Everything the pipeline does is the cfs cell's implementation
reached through :data:`~teb_vae.lag_attn_transformer_cfs.eval.binding.TRF_CFS_BINDING`, so what
this file catches is an analysis that runs on one architecture and fails on the other: a readout
that reached for an attribute only the conv-LSTM encoder has, a retention path whose tensor shapes
differ, a disclosure key that resolves to nothing here.

It asserts the **shape** of what a run leaves behind -- the complete artifact layout, a step record
from every registered analysis, an exit code, and a coverage block that says which population each
analysis actually saw -- and it asserts that the layout is the *same* one, because the cross-cell
comparison reads two directories down one set of names.

**It starts no run of its own.** The session-scoped ``trf_collected_run`` fixture is this suite's
one end-to-end pass, and :func:`test_this_file_starts_no_run_of_its_own` keeps that from quietly
changing.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from teb_vae.lag_attn_cfs.eval import collect, preflight, probe as probe_module
from teb_vae.lag_attn_cfs.tests.test_eval_smoke import DURABLE_ARTIFACTS
from teb_vae.lag_attn_transformer_cfs.eval import run as run_module

pytestmark = pytest.mark.slow


def _registry() -> Dict[str, Any]:
    """This cell's analyses, resolved through its binding on every call."""
    return run_module.analysis_registry()


# =================================================================================================
# The run itself
# =================================================================================================
def test_the_full_run_completes_with_exit_code_zero(trf_collected_run) -> None:
    """The failed steps are named with their errors rather than left to a bare ``1 == 0``: this run
    is the most expensive thing the suite does, so a failure that does not say which analysis
    raised buys a second one."""
    failed = [
        f"{record['name']}: {record.get('error')}"
        for record in trf_collected_run["summary"]["steps"]
        if record["status"] != "ok"
    ]

    assert failed == [], failed
    assert trf_collected_run["exit_code"] == 0


def test_every_registered_analysis_contributes_a_step_record(trf_collected_run) -> None:
    """Every selectable analysis, the unskippable channel map, and the loader probe: a step each,
    every one ok. The registry is the causal cell's, reached through the binding -- so this is also
    the assertion that the encoder replacement did not silently drop a question."""
    steps = {
        record["name"]: record["status"] for record in trf_collected_run["summary"]["steps"]
    }

    expected = {"probe", *run_module.UNSKIPPABLE_ANALYSES, *_registry()}
    assert expected <= set(steps), sorted(expected - set(steps))
    assert {"warmup", "source_null", "spectral_skill"} <= set(steps)
    assert all(status == "ok" for status in steps.values()), steps


def test_the_artifact_layout_is_the_causal_cells_own(trf_collected_run) -> None:
    """The same names, imported from the sibling suite rather than restated: the cross-cell
    comparison reads two run directories down one set of names, and a layout that diverged would
    make that comparison a directory-shape exercise."""
    results_dir = Path(trf_collected_run["results_dir"])
    results = trf_collected_run["summary"]["results"]

    assert preflight.PREFLIGHT_FILENAME in DURABLE_ARTIFACTS
    assert probe_module.PROBE_FILENAME in DURABLE_ARTIFACTS
    assert collect.COLLECTION_FILENAME in DURABLE_ARTIFACTS
    for name in DURABLE_ARTIFACTS:
        assert (results_dir / name).is_file(), f"the run left no {name}"

    # An analysis wrote artifacts, or it said in its own block why it did not. This cell's
    # ``tiny.yaml`` trains under ``likelihood: mse`` exactly as the causal cell's does, so
    # ``calibration``'s skip is the standing case rather than an anomaly.
    subdirectories = {path.name for path in results_dir.iterdir() if path.is_dir()}
    silent: Set[str] = {
        name for name in _registry()
        if name not in subdirectories
        and not (results.get(name) or {}).get("skipped")
    }
    assert silent == set(), f"no artifact subdirectory and no recorded skip for {sorted(silent)}"
    wrote = subdirectories & set(_registry())
    assert len(wrote) > len(set(_registry()) - wrote), (
        f"only {sorted(wrote)} wrote artifacts; the rest recorded skips, so this run demonstrates "
        f"the skip path rather than the pipeline"
    )


def test_no_directory_is_left_by_an_analysis_this_pipeline_does_not_have(
    trf_collected_run,
) -> None:
    """``coherence`` is not ported at all, for a reason that is a property of the *target domain*
    and therefore true of both cfs cells: a stored scattering coefficient is a modulus, so phase
    agreement and group delay have no analogue at any window length."""
    subdirectories = {
        path.name for path in Path(trf_collected_run["results_dir"]).iterdir() if path.is_dir()
    }

    assert "coherence" not in subdirectories
    assert "spectral_skill" in subdirectories


def test_every_registered_headline_path_resolves_on_this_cells_run(trf_collected_run) -> None:
    """The headline block is the one surface the reporting layer promises to keep resolvable, and
    it is the only surface the arm tables and the cross-cell table read. A path that resolves on
    the causal cell's run and not on this one would empty a column of the very comparison this
    package exists to make -- silently, because an unresolved path is written as ``null`` rather
    than raised."""
    from teb_vae.lag_attn_cfs.eval.binding import HEADLINE_SCALARS as EXTRA_SCALARS
    from teb_vae.lag_attn_cfs.eval.report_seam import HEADLINE_SCALARS as SHARED_SCALARS

    headline = trf_collected_run["summary"]["results"]["headline"]

    registered = [name for name, _ in SHARED_SCALARS] + [name for name, _ in EXTRA_SCALARS]
    assert set(registered) <= set(headline), sorted(set(registered) - set(headline))
    # The three cfs-only analyses' scalars in particular: they are registered through the binding,
    # which is the object this cell shares with the comparison cell by identity.
    unresolved = [name for name, _ in EXTRA_SCALARS if headline.get(name) is None]
    assert unresolved == [], unresolved


# =================================================================================================
# The coverage block
# =================================================================================================
def test_the_coverage_block_reports_a_population_per_uncapped_analysis(trf_collected_run) -> None:
    """Effective-$n$ per analysis is what reveals two analyses having run on different
    populations, which nothing else in the output shows.

    ``None`` is the third state and it is load-bearing: an analysis that scored **no** population
    reports it rather than ``0``, so a skip does not read as a population of zero and make every
    scoring analysis look like a disagreement with it.
    """
    per_analysis = trf_collected_run["summary"]["results"]["coverage"]["per_analysis"]

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
    assert 0 not in {record["n_samples"] for record in per_analysis.values()}


def test_a_population_disagreement_is_a_warning_and_not_a_failure(trf_collected_run) -> None:
    """The exit code is non-zero **if and only if a step raised**, which is why the offline gate
    exists separately and refuses on the sanity block."""
    coverage = trf_collected_run["summary"]["results"]["coverage"]

    assert isinstance(coverage["warnings"], list)
    assert trf_collected_run["exit_code"] == 0
    assert trf_collected_run["summary"]["failed"] == []


# =================================================================================================
# The one pass
# =================================================================================================
def test_this_file_starts_no_run_of_its_own() -> None:
    """This suite performs exactly one end-to-end pass and every artifact assertion reads it."""
    source = Path(__file__).read_text(encoding="utf-8")

    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "main"
    ]

    assert calls == [], "this file calls main(); read the session-scoped run instead"
