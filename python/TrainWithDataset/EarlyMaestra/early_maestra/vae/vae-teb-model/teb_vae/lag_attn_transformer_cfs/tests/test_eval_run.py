r"""The two thin entry points, and the one property that makes them worth having.

``run.py`` and ``verify.py`` exist so that this cell can be launched and gated by name. What they
must **not** become is a second implementation: the two cfs cells exist to be compared, and a second
copy of the runner or of the gate's criteria is how two things that must stay comparable stop being
comparable -- the first fix to an analysis or a threshold lands on one side, and the two
``summary.json`` files quietly stop meaning the same thing.

So the assertions here are about delegation rather than about numbers. Every public callable is
either the cfs cell's object bound under this name, or a wrapper whose body reaches into it; the
registry is re-derived on every call rather than frozen at import; and neither module carries a
numeric stack at all, which is the mechanical form of "defines no numeric function".
"""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from teb_vae.lag_attn_cfs.eval import run as shared_run
from teb_vae.lag_attn_cfs.eval import verify as shared_verify
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_transformer_cfs.eval import run as run_module
from teb_vae.lag_attn_transformer_cfs.eval import verify as verify_module
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

from .conftest import _REPO_ROOT

#: The two modules under test, and the name each binds the cfs cell's module under. The delegation
#: check below reads that name out of the source, so an alias rename fails here rather than turning
#: the check vacuous.
WRAPPERS: Dict[Any, str] = {run_module: "shared_run", verify_module: "shared"}

#: The one public callable allowed to carry a body of its own without reaching into the cfs cell:
#: the argparse surface, which exists precisely so the usage line and the ``--only`` help name
#: *this* package. Everything else must delegate.
LOCAL_BY_DESIGN: Set[str] = {"build_parser"}


def _functions(module: Any) -> Dict[str, ast.FunctionDef]:
    """Return every **public** function defined at module level, by name.

    Public because that is the surface the wrapper rule is about: ``_cli`` is argparse dispatch,
    local for the same reason ``build_parser`` is, and private by the same convention that keeps
    it out of the module's contract.
    """
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def _reaches_into(node: ast.AST, alias: str) -> bool:
    """Whether a function body reads any attribute of ``alias``."""
    return any(
        isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == alias
        for child in ast.walk(node)
    )


# =================================================================================================
# Delegation
# =================================================================================================
@pytest.mark.parametrize("module", list(WRAPPERS), ids=lambda module: module.__name__)
def test_every_callable_is_a_wrapper_rather_than_a_second_implementation(module: Any) -> None:
    """Read off the source rather than trusted: a function that quietly grew a body of its own is
    exactly the drift these two modules exist to avoid, and it would look like a bug fix."""
    alias = WRAPPERS[module]

    local = [
        name for name, node in _functions(module).items()
        if name not in LOCAL_BY_DESIGN and not _reaches_into(node, alias)
    ]

    assert local == [], (
        f"{module.__name__}: {local} carry a body that never reaches into {alias}. Both entry "
        f"points are wrappers; a readout, a threshold or a criterion implemented here is one the "
        f"comparison cell does not have."
    )


@pytest.mark.parametrize("module", list(WRAPPERS), ids=lambda module: module.__name__)
def test_neither_module_carries_a_numeric_stack(module: Any) -> None:
    """The mechanical form of "defines no numeric function": neither module imports ``torch``,
    ``numpy`` or ``pandas``, so neither *can* compute a readout. ``verify`` additionally must stay
    importable where none of them is installed, which the subprocess check below proves."""
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert imported & {"torch", "numpy", "pandas", "scipy", "matplotlib"} == set()


def test_the_gate_criteria_are_the_cfs_cells_own_objects() -> None:
    """Not equal copies: the ten acceptance verdicts, the criteria registry and the named
    ``pred_gap`` column are properties of the shared objective and the shared readout registry, so
    a second set here would be a second set of thresholds for two cells compared under one."""
    assert verify_module.PRED_GAP_COLUMN is shared_verify.PRED_GAP_COLUMN
    assert verify_module.SUMMARY_FILENAME == shared_verify.SUMMARY_FILENAME
    assert verify_module.SWEPT_ANCHOR_STRIDE is shared_verify.SWEPT_ANCHOR_STRIDE
    # No criterion registry, no verdict list and no threshold of its own: this module defines none
    # of the three names the gate's decisions are made from.
    for name in ("CRITERIA", "CFS_VERDICTS", "verify"):
        assert name not in vars(verify_module), (
            f"{name} is defined here as well as in the cfs cell; the gate's criteria are the "
            f"shared objective's and two copies could only ever disagree"
        )


def test_the_runner_supplies_this_cells_binding_and_delegates_the_rest(monkeypatch) -> None:
    """``main`` adds one keyword and hands everything else on. Asserted by intercepting the shared
    runner rather than by running one, because what is being checked is which binding arrives."""
    seen: Dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> int:
        seen.update(kwargs)
        seen["args"] = args
        return 0

    monkeypatch.setattr(shared_run, "main", _capture)

    assert run_module.main("ckpt.ckpt", "out", device="cpu") == 0
    assert seen["binding"] is TRF_CFS_BINDING
    assert seen["args"] == ("ckpt.ckpt", "out")
    assert seen["device"] == "cpu"


def test_a_caller_may_override_the_binding(monkeypatch) -> None:
    """``setdefault`` rather than an assignment: the offline re-run tests drive this entry point
    with another binding to prove no model is built, and an assignment would silently ignore
    them."""
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(shared_run, "main", lambda *a, **k: seen.update(k) or 0)

    run_module.main("ckpt.ckpt", binding=CFS_BINDING)

    assert seen["binding"] is CFS_BINDING


# =================================================================================================
# The registry
# =================================================================================================
def test_the_registry_is_derived_on_every_call_rather_than_frozen_at_import(monkeypatch) -> None:
    """The help text, the selection ``main`` makes and the ``summary.json`` record must all read
    one mapping. Frozen at import, an analysis registered on the binding would reach the run and
    not the help text, or the reverse -- and nothing in the artifact would say which."""
    extended = dict(shared_run.ANALYSIS_FUNCTIONS)
    extended["a_new_shared_analysis"] = lambda *a, **k: None
    monkeypatch.setattr(shared_run, "ANALYSIS_FUNCTIONS", extended)

    assert "a_new_shared_analysis" in run_module.analysis_registry()


def test_the_registry_is_the_cfs_cells_in_the_cfs_cells_order() -> None:
    """The encoder edge must not change which questions are asked, nor the order they are asked
    in: two ``steps.json`` files read side by side are the point of the cross-cell table."""
    assert list(run_module.analysis_registry()) == list(
        shared_run.merged_analysis_functions(CFS_BINDING)
    )
    assert run_module.ANALYSES == tuple(run_module.analysis_registry())
    # Non-vacuity: the three cfs-only analyses are reached through the binding rather than by
    # being registered here.
    assert {"warmup", "source_null", "spectral_skill"} <= set(run_module.ANALYSES)


def test_the_unskippable_step_is_the_cfs_cells_and_is_not_selectable() -> None:
    """It describes the *data* -- the shards' own provenance and causal attributes -- so a
    model-specific addition here would be a category error."""
    assert run_module.UNSKIPPABLE_ANALYSES is shared_run.UNSKIPPABLE_ANALYSES
    assert set(run_module.UNSKIPPABLE_ANALYSES) & set(run_module.ANALYSES) == set()


def test_the_output_names_are_the_cfs_cells_own() -> None:
    """A caller of this entry point reads its artifacts by these names, and two cells writing two
    layouts would make the cross-cell comparison a directory-shape exercise."""
    assert run_module.RESULTS_DIRNAME == shared_run.RESULTS_DIRNAME
    assert run_module.SUMMARY_FILENAME == shared_run.SUMMARY_FILENAME
    assert run_module.STEPS_FILENAME == shared_run.STEPS_FILENAME
    assert run_module.LOG_FILENAME == shared_run.LOG_FILENAME


# =================================================================================================
# The gate's one non-negotiable property
# =================================================================================================
def test_importing_this_cells_gate_pulls_in_no_numeric_stack() -> None:
    """Run in a subprocess: this session has already imported ``torch``, so an in-process check
    would pass no matter what the module does. A summary produced on the production box has to be
    checkable on a machine that has never had a deep-learning stack on it, and that has to hold for
    the entry point an operator actually types."""
    source = (
        "import sys\n"
        "import teb_vae.lag_attn_transformer_cfs.eval.verify as gate\n"
        "leaked = sorted(name for name in sys.modules if name.split('.')[0] "
        "in {'torch', 'lightning', 'numpy', 'scipy', 'h5py', 'pandas', 'matplotlib'})\n"
        "assert gate.PRED_GAP_COLUMN\n"
        "print(','.join(leaked))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"importing this cell's gate pulled in {completed.stdout.strip()}"
    )


# =================================================================================================
# The tables
# =================================================================================================
def _write_arm(root: Path, name: str, *, model_class: str, anchor_stride: int = 15) -> None:
    """Write one finished-run shape under ``root``, through the cfs suite's own writer."""
    from teb_vae.lag_attn_cfs.tests.test_eval_verify import write_arm

    write_arm(root, name, model_class=model_class, anchor_stride=anchor_stride)


def test_the_tables_carry_this_cells_one_axis_and_the_cross_cell_comparison(tmp_path) -> None:
    """One sweep section rather than the cfs cell's four: this package ships one ``sweep_*.yaml``
    arm, and a section whose every row read ``(absent)`` would print a sweep nobody ran as though
    somebody had. The cross-cell table is the shared one, so the two cells' rows are assembled the
    same way."""
    _write_arm(tmp_path, "trf_dense", model_class="SeqVaeLagAttnTrfCfs", anchor_stride=1)
    _write_arm(tmp_path, "trf_tiled", model_class="SeqVaeLagAttnTrfCfs", anchor_stride=15)
    _write_arm(tmp_path, "cfs_tiled", model_class="SeqVaeLagAttnCfs", anchor_stride=15)
    out = tmp_path / "arms.md"

    assert verify_module.compare_arms(tmp_path, out) == 0
    document = out.read_text(encoding="utf-8")

    headings: List[str] = [line for line in document.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Arm inventory",
        "## Anchor tiling sweep (`anchor_stride`)",
        "## Cross-cell comparison",
    ], headings
    # Both cells' rows in the cross-cell table, keyed on the class each run recorded.
    assert document.count("SeqVaeLagAttnTrfCfs") >= 2
    assert "SeqVaeLagAttnCfs" in document
    assert shared_verify.SELECTION_RULE in document


def test_the_gate_and_the_tables_dispatch_from_one_command_line(tmp_path) -> None:
    import json

    from teb_vae.lag_attn_cfs.tests.test_eval_verify import clean_summary

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(clean_summary()), encoding="utf-8")

    assert verify_module._cli([str(summary)]) == 0
    assert verify_module._cli([str(summary), "--runs", str(tmp_path)]) == 2  # both is a usage error
    assert verify_module._cli([]) == 2  # neither is too
