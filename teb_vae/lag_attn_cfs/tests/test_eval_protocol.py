r"""The one shape every analysis has, and the offline re-run that shape exists to make possible.

An analysis is ``run_<name>_analysis(context, *, eval_config, output_dir, probe) -> Dict``, and
that is asserted **by inspection over the shipped modules** rather than by convention. A signature
that has drifted is otherwise found at the ninth step of a multi-hour pass, by which point the
eight before it have already been written and the tenth will not run.

The inspection walks the *merged* registry -- the shared one plus whatever this cell's binding
registers -- so the two analyses only this cell has are covered by the same rule as the copied
ones rather than being exempt from it. That matters more here than in the sibling: they are the
analyses written from scratch, so they are the ones whose signature could plausibly be wrong.

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

from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING

#: The directory holding the shipped analyses.
ANALYSES_ROOT = Path(run_module.__file__).resolve().parent / "analyses"

#: The signature every analysis has: one positional context, then three keyword-only arguments.
EXPECTED_POSITIONAL = ("context",)
EXPECTED_KEYWORD_ONLY = ("eval_config", "output_dir", "probe")

#: The only analyses permitted to read ``task`` or ``loader`` off the context, each for a
#: structural reason the docstring of ``test_only_the_stated_analyses_reach_for_the_model_on_the_context``
#: gives. Three rather than two, and the third is the strongest case of the same rule: an
#: INTERVENTION on the model's input cannot be served by any table a forward already wrote, because
#: the forward it needs is one that never happened.
MODEL_READING_ANALYSES = frozenset({"samples", "sufficiency", "occlusion"})


def _analysis_functions() -> Dict[str, Any]:
    """Import every shipped ``analyses/*.py`` and return its ``run_<name>_analysis`` callables."""
    import importlib

    found: Dict[str, Any] = {}
    for path in sorted(ANALYSES_ROOT.glob("*.py")):
        if path.stem == "__init__":
            continue
        module = importlib.import_module(f"teb_vae.lag_attn_cfs.eval.analyses.{path.stem}")
        for name, value in vars(module).items():
            if name.startswith("run_") and name.endswith("_analysis") and callable(value):
                found[f"{path.stem}.{name}"] = value
    return found


def _merged_registry() -> Dict[str, Any]:
    """Every analysis a run of this cell can select or is given, in run order.

    The binding's extras are included because they are exactly the analyses this package wrote
    rather than copied; excluding them would leave the new code the least checked.
    """
    return {
        **run_module.UNSKIPPABLE_ANALYSES,
        **run_module.merged_analysis_functions(CFS_BINDING),
    }


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
    """A registry entry pointing at something else would run outside the protocol entirely. The
    merged registry rather than the shared one: this cell's own analyses are registered on the
    binding, and inspecting only the shared registry would exempt exactly the new code."""
    shipped = set(_analysis_functions().values())
    registered = _merged_registry()

    assert registered, "no analysis is registered at all"
    for name, function in registered.items():
        assert function in shipped, f"{name} is not a run_*_analysis under {ANALYSES_ROOT}"


def test_the_bindings_extras_are_inspected_by_the_same_rule_as_the_shared_registry() -> None:
    """Non-vacuity for the merge above: whatever the binding registers has to appear in the set
    the signature parametrisation walks, so an extra analysis cannot land unchecked."""
    merged = _merged_registry()

    assert set(run_module.ANALYSIS_FUNCTIONS) <= set(merged)
    for name, function in CFS_BINDING.extra_analyses.items():
        assert merged.get(name) is function
        signature = inspect.signature(function)
        assert tuple(
            parameter.name for parameter in signature.parameters.values()
            if parameter.kind == parameter.KEYWORD_ONLY
        ) == EXPECTED_KEYWORD_ONLY


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


def test_no_analysis_imports_another() -> None:
    """Anything two analyses share moves one layer down -- into ``metrics``, ``events`` or the
    reuse seam -- rather than being reached for sideways. Asserted here as well as in the layering
    walk because the shape a sideways import takes here is the one that looks harmless: a
    band-resolved readout reaching into the channel map that produced it, rather than reading the
    CSV that map was persisted to."""
    offenders: List[str] = []
    for path in sorted(ANALYSES_ROOT.glob("*.py")):
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = (
                node.module if isinstance(node, ast.ImportFrom)
                else None
            )
            names = (
                [alias.name for alias in node.names] if isinstance(node, (ast.Import,))
                else []
            )
            if module and "eval.analyses" in module:
                offenders.append(f"{path.name}: from {module}")
            for name in names:
                if "eval.analyses" in name:
                    offenders.append(f"{path.name}: import {name}")

    assert offenders == [], offenders


# =============================================================================
# The return value
# =============================================================================
def test_the_unskippable_step_is_inspected_by_the_same_rule_as_the_selectable_ones() -> None:
    """The channel map is its only member, and it obeys the protocol like everything else.

    Asserted through the merge rather than beside it, because the merge silently tolerates an empty
    half: a step that stopped being registered would leave every parametrised signature check above
    passing with one fewer subject and nothing saying so.
    """
    assert set(run_module.UNSKIPPABLE_ANALYSES) == {"band_partition"}
    assert set(_merged_registry()) == {"band_partition"} | set(
        run_module.merged_analysis_functions(CFS_BINDING)
    )


@pytest.mark.slow
def test_every_analysis_returns_the_protocol_keys_on_a_real_run(collected_run) -> None:
    """The return half of the protocol, read off a pass that actually ran every registered
    analysis. Asserted on the run rather than on hand-built contexts because the keys have to be
    there for the *runner* -- it reads them without knowing what the analysis did, and an analysis
    tested in isolation can return them while failing to on the inputs a real pass hands it."""
    results = collected_run["summary"]["results"]

    for name in _merged_registry():
        block = results.get(name)
        assert isinstance(block, dict), f"{name} produced no block"
        assert set(block) >= set(REQUIRED_RESULT_KEYS), sorted(set(REQUIRED_RESULT_KEYS) - set(block))
        assert "capped" in block["plan"]
        # ``None`` rather than zero for an analysis that scores no segments: a zero would enter the
        # coverage block's population comparison as a disagreement with every analysis that does.
        assert block["n_samples"] is None or isinstance(block["n_samples"], int)
        assert isinstance(block["composition"], dict)


@pytest.mark.slow
def test_every_declared_grouped_frame_names_a_file_that_exists(collected_run) -> None:
    """The declaration is a *path on disk plus the columns to resolve by*, and the runner reads it
    after the analysis returned. An analysis that declared a frame it did not write would have its
    fan-out skipped with a reason that reads like a cohort problem rather than like a bug."""
    results = collected_run["summary"]["results"]
    declared = 0

    for name in _merged_registry():
        for entry in results.get(name, {}).get("grouped_frames", []) or []:
            declared += 1
            path = Path(entry["path"])
            assert not path.is_absolute(), entry["path"]
            assert (collected_run["results_dir"] / path).is_file(), entry["path"]
            assert entry["value_columns"], name

    assert declared >= 2, "no analysis declared a grouped frame, so this proves nothing"


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


@pytest.mark.slow
def test_a_table_only_analysis_runs_against_a_finished_directory_with_no_model(
    collected_run, tmp_path, monkeypatch
) -> None:
    """The reason collection and emission are separate steps at all.

    The finished run is copied rather than re-entered so this pass cannot disturb the fixture
    every other file in the suite questions.
    """
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    run_dir = tmp_path / "rerun"
    shutil.copytree(collected_run["results_dir"].parent, run_dir)

    def _explode(*args, **kwargs):
        raise AssertionError("the model was built and forwarded on an offline re-run")

    monkeypatch.setattr(SeqVaeLagAttnCfs, "forward", _explode)
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
    assert summary["results"]["readouts"] == collected_run["summary"]["results"]["readouts"]


def test_an_offline_run_without_tables_says_what_is_missing(tmp_path) -> None:
    """A directory that is not a finished run must name the two ways out, not fail obscurely."""
    with pytest.raises(FileNotFoundError, match="--checkpoint is required"):
        run_module.main(None, tmp_path / "empty")


def test_only_the_stated_analyses_reach_for_the_model_on_the_context() -> None:
    """The context carries ``task`` and ``loader``, and exactly two analyses may read them.

    Every reason is structural rather than a convenience. A diagnostic page is the whole forward
    output of one segment, and the pages for the *extreme* rows are chosen by sorting a table that
    did not exist while the pass ran, so neither retention nor a wider table can serve them. The
    oracle probe is fitted on every segment's **encoder state**, which is on neither durable table
    and is not a per-segment scalar that could be put on one. And the occlusion readout scores a
    forward that never ran: it zeroes a band of the source's own values and re-encodes, so what it
    needs is not a number some pass forgot to record but a model to ask a counterfactual of.

    Everything else must not so much as mention the two fields. In this target domain that rule
    has one consequence worth naming: the band-resolved skill readout has an obvious reason to
    want ``model.target_gate`` -- it joins a per-channel vector on the kept axis to a channel map
    on the declared one -- and it may not have it, because ``context.task`` is ``None`` on exactly
    the path that join has to work on. The map is persisted and read off disk instead.
    """
    fields = {field for field in AnalysisContext.__dataclass_fields__}
    assert fields == {"collection", "config", "task", "loader"}

    modules = [path for path in sorted(ANALYSES_ROOT.glob("*.py")) if path.stem != "__init__"]
    reaching = sorted(
        path.stem
        for path in modules
        if any(
            name in path.read_text(encoding="utf-8")
            for name in ('context, "task"', 'context, "loader"', "context.task", "context.loader")
        )
    )

    assert modules, "the walk found no analyses, so this proves nothing"
    assert set(reaching) <= MODEL_READING_ANALYSES, sorted(
        set(reaching) - MODEL_READING_ANALYSES
    )


def test_no_analysis_reaches_the_target_gate_for_the_kept_channel_axis() -> None:
    """The join this target domain adds, and the one way it would be written wrongly. The
    per-channel readouts are indexed on the kept channels while a channel-to-band map is over the
    declared ones; asking the model which is which works in a pass that built one and returns
    ``None`` on the offline re-run the split exists for."""
    # ``__init__.py`` is excluded because it is where the prohibition is *written down*: it names
    # the attribute in prose, which is the opposite of reaching for it.
    offenders = [
        path.name
        for path in sorted(ANALYSES_ROOT.glob("*.py"))
        if path.stem != "__init__" and "target_gate" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


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
        not name.startswith(("torch", "lightning", "teb_vae.lag_attn_cfs.nets"))
        for name in imported
    ), imported
