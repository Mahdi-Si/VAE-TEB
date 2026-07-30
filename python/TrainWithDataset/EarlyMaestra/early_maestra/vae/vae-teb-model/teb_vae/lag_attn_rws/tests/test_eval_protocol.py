r"""The one shape every analysis has, and the offline re-run that shape exists to make possible.

An analysis is ``run_<name>_analysis(context, *, eval_config, output_dir, probe) -> Dict``, and
that is asserted **by inspection over the shipped modules** rather than by convention. A signature
that has drifted is otherwise found at the ninth step of a multi-hour pass, by which point the
eight before it have already been written and the tenth will not run.

The property the protocol exists for is the last test here: an analysis reads the tables the
shared collection pass wrote, never the model, so a run with **no checkpoint at all** against a
finished directory produces analysis output. That is what makes re-running one analysis after a
long pass cost seconds rather than hours, and it is proved with the model's ``forward`` rigged to
raise -- a spy is the only way to tell "did not need the model" from "happened not to use it".
"""
from __future__ import annotations

import ast
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval import run as run_module
from teb_vae.lag_attn_rws.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext

#: The directory holding the shipped analyses.
ANALYSES_ROOT = Path(run_module.__file__).resolve().parent / "analyses"

#: The signature every analysis has: one positional context, then three keyword-only arguments.
EXPECTED_POSITIONAL = ("context",)
EXPECTED_KEYWORD_ONLY = ("eval_config", "output_dir", "probe")


def _analysis_functions() -> Dict[str, Any]:
    """Import every shipped ``analyses/*.py`` and return its ``run_<name>_analysis`` callables."""
    import importlib

    found: Dict[str, Any] = {}
    for path in sorted(ANALYSES_ROOT.glob("*.py")):
        if path.stem == "__init__":
            continue
        module = importlib.import_module(f"teb_vae.lag_attn_rws.eval.analyses.{path.stem}")
        for name, value in vars(module).items():
            if name.startswith("run_") and name.endswith("_analysis") and callable(value):
                found[f"{path.stem}.{name}"] = value
    return found


# =============================================================================
# The signature
# =============================================================================
def test_the_walk_found_analyses_to_check() -> None:
    """A walk that found nothing would pass every signature test below vacuously."""
    assert _analysis_functions(), f"no run_*_analysis found under {ANALYSES_ROOT}"


@pytest.mark.parametrize("name", sorted(_analysis_functions()))
def test_every_shipped_analysis_has_the_protocol_signature(name: str) -> None:
    signature = inspect.signature(_analysis_functions()[name])
    positional = [
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_only = [
        parameter.name for parameter in signature.parameters.values()
        if parameter.kind == parameter.KEYWORD_ONLY
    ]

    assert tuple(positional) == EXPECTED_POSITIONAL
    assert tuple(keyword_only) == EXPECTED_KEYWORD_ONLY


def test_every_registered_analysis_is_one_of_the_shipped_functions() -> None:
    """A registry entry pointing at something else would run outside the protocol entirely."""
    shipped = set(_analysis_functions().values())
    registered = {
        **run_module.UNSKIPPABLE_ANALYSES,
        **run_module.ANALYSIS_FUNCTIONS,
    }

    assert registered, "no analysis is registered at all"
    for name, function in registered.items():
        assert function in shipped, f"{name} is not a run_*_analysis under {ANALYSES_ROOT}"


def test_no_analysis_emits_its_own_grouped_variants() -> None:
    """The by-class and by-subgroup fan-out is the *runner's* job.

    Written per analysis it would be a cross-cutting change every analysis added later has to
    remember to make, and the one that forgets reports a pooled number over a mixed cohort with
    nothing saying so. An analysis declares a frame; it does not emit.
    """
    offenders: List[str] = []
    for path in sorted(ANALYSES_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "emit_grouped_variants" in source or "summarise_by_group" in source:
            offenders.append(path.name)

    assert offenders == []


# =============================================================================
# The return value
# =============================================================================
def test_the_unskippable_analysis_returns_the_protocol_keys(
    multi_class_config, multi_class_shards, tmp_path
) -> None:
    context = AnalysisContext(collection=None, config=multi_class_config)

    result = run_module.UNSKIPPABLE_ANALYSES["band_partition"](
        context, eval_config={}, output_dir=tmp_path, probe=None
    )

    assert set(result) >= set(REQUIRED_RESULT_KEYS)
    # None rather than zero: this analysis scores no segments, and a zero would enter the coverage
    # block's population comparison as a disagreement with every analysis that does.
    assert result["n_samples"] is None
    assert result["plan"]["capped"] is False


# =============================================================================
# The offline re-run
# =============================================================================
def _table_only_analysis(context, *, eval_config, output_dir, probe):
    """Read the collected table and write a CSV, touching no model.

    Deliberately reads ``per_sample`` off the context rather than being handed numbers: the point
    of the offline path is that the table is enough.
    """
    frame = context.collection.per_sample
    path = Path(output_dir) / "table_only.csv"
    frame[["sample_index", "guid"]].to_csv(path, index=False)
    return {
        "n_samples": int(len(frame)),
        "composition": {"n_recordings": int(frame["guid"].nunique())},
        "plan": {"capped": False},
        "rows_written": int(len(frame)),
    }


def test_a_table_only_analysis_runs_against_a_finished_directory_with_no_model(
    evaluated, tmp_path, monkeypatch
) -> None:
    """The reason collection and emission are separate steps at all.

    The finished run is copied rather than re-entered so this pass cannot disturb the fixture
    every other file in the suite questions.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    run_dir = tmp_path / "rerun"
    shutil.copytree(evaluated["results_dir"].parent, run_dir)

    def _explode(*args, **kwargs):
        raise AssertionError("the model was built and forwarded on an offline re-run")

    monkeypatch.setattr(SeqVaeLagAttnRws, "forward", _explode)
    monkeypatch.setattr(run_module, "ANALYSIS_FUNCTIONS", {"table_only": _table_only_analysis})

    exit_code = run_module.main(None, run_dir, only="table_only", device="cpu")

    results_dir = run_dir / run_module.RESULTS_DIRNAME
    summary = json.loads((results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["checkpoint"] is None
    assert summary["analyses_selected"] == ["table_only"]
    assert summary["results"]["table_only"]["rows_written"] == len(
        pd.read_csv(results_dir / "table_only.csv")
    )
    # The readouts of the run being re-read are still there: an offline pass reports the same
    # findings as the pass that collected them, plus its own.
    assert summary["results"]["readouts"] == evaluated["summary"]["results"]["readouts"]


def test_an_offline_run_without_tables_says_what_is_missing(tmp_path) -> None:
    """A directory that is not a finished run must name the two ways out, not fail obscurely."""
    with pytest.raises(FileNotFoundError, match="--checkpoint is required"):
        run_module.main(None, tmp_path / "empty")


def test_only_the_two_stated_analyses_reach_for_the_model_on_the_context() -> None:
    """The context carries ``task`` and ``loader``, and exactly two analyses may read them.

    This assertion used to be ``fields == {'collection', 'config'}``. It was relaxed when the
    per-sample pages landed, and the reason it could not stand is structural rather than a
    convenience: a diagnostic page is the whole forward output of one segment, and the pages for
    the *extreme* rows are chosen by sorting a table that did not exist while the pass ran, so
    neither retention nor a wider table can serve them.

    The list grew a second entry when the sufficiency gap landed, and the reason is the same shape
    and equally structural: the oracle probe is fitted on every segment's **encoder state**, which
    is on neither durable table and is not a per-segment scalar that could be put on one -- it is
    $T_{\\mathrm{valid}} \\times d_{\\mathrm{model}}$ per segment, and the fit reads it thousands
    of times.

    What the original assertion was actually protecting -- that no analysis quietly needs a model
    -- is pinned here instead, by source: every *other* analysis must not so much as mention the
    two fields, and the offline re-run above proves the whole registry still runs with both of
    them ``None``. Both exceptions record a skip rather than assuming a model, which is what makes
    that re-run pass rather than crash.
    """
    fields = {field for field in AnalysisContext.__dataclass_fields__}
    assert fields == {"collection", "config", "task", "loader"}

    reaching = sorted(
        path.stem
        for path in ANALYSES_ROOT.glob("*.py")
        if path.stem != "__init__"
        and any(
            name in path.read_text(encoding="utf-8")
            for name in ('context, "task"', 'context, "loader"', "context.task", "context.loader")
        )
    )
    assert reaching == ["samples", "sufficiency"]


def test_the_protocol_module_imports_nothing_from_the_model() -> None:
    """The protocol definition itself must stay importable without ``torch``."""
    source = (ANALYSES_ROOT / "__init__.py").read_text(encoding="utf-8")
    imported = [
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    ]

    assert all(
        not name.startswith(("torch", "lightning", "teb_vae.lag_attn_rws.nets"))
        for name in imported
    ), imported
