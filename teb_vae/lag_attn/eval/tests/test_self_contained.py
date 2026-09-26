r"""The evaluation package must not depend on the tree it supersedes, or on Lightning.

``train/tests/test_layering.py`` already forbids ``teb_vae`` importing ``model``, recursively and
AST-walked. This file pins the *stronger* property the spec's goals promise -- no ``model``, no
Lightning, and no reach into the model's own ``task.py`` -- and it keeps holding as the package
grows, because it walks every ``.py`` under ``eval/`` rather than a list written today.

The AST walk is what makes it real. A lazy in-function ``import model.something`` is invisible
to a module-level check and is exactly the shortcut a future change would reach for; it is
counted here.

``trainer.py`` carries one narrow, deliberate exemption: ``preflight.py`` imports its two
preflight guards rather than copying ninety lines, so their long actionable error messages can
never drift. The exemption is per-module and per-target, not a blanket allowance.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

#: The package under scrutiny.
EVAL_ROOT = Path(__file__).resolve().parents[1]

#: Import prefixes no module under ``eval/`` may reach for.
FORBIDDEN_PREFIXES: Tuple[str, ...] = (
    "model",
    "lightning",
    "pytorch_lightning",
    "teb_vae.lag_attn.task",
    "teb_vae.lag_attn.trainer",
    "teb_vae.lag_attn.plotting",
)

#: ``module stem -> prefixes it may import anyway``. Narrow by construction: one module, one
#: target, for one stated reason.
EXEMPTIONS: Dict[str, Set[str]] = {
    "preflight": {"teb_vae.lag_attn.trainer"},
}

#: The test package itself is excluded. Its whole job is to build tasks and stub batches, so it
#: legitimately imports ``task.py`` and Lightning; the shipped package is what must stay clean.
EXCLUDED_DIRS = ("tests",)


def _shipped_modules() -> List[Path]:
    """Every ``.py`` under ``eval/`` that ships, i.e. excluding the test package."""
    return sorted(
        path
        for path in EVAL_ROOT.rglob("*.py")
        if not any(part in EXCLUDED_DIRS for part in path.relative_to(EVAL_ROOT).parts)
    )


def imported_names(source: str) -> List[str]:
    """Return every module name imported anywhere in ``source``, lazy imports included.

    Args:
        source: Python source text.

    Returns:
        The imported module names, in encounter order.
    """
    names: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which cannot reach outside the package.
            if node.level == 0 and node.module:
                names.append(node.module)
    return names


def _violations(path: Path) -> List[str]:
    """Return the forbidden imports in one module, honouring its exemptions."""
    allowed = EXEMPTIONS.get(path.stem, set())
    found = []
    for name in imported_names(path.read_text(encoding="utf-8")):
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                if any(name == ok or name.startswith(ok + ".") for ok in allowed):
                    continue
                found.append(name)
    return found


def test_no_shipped_module_imports_the_superseded_tree_or_lightning() -> None:
    """One walk over every shipped module; a failure names each offending file and its imports."""
    modules = _shipped_modules()
    # A walk that found nothing would pass vacuously.
    assert len(modules) >= 5, f"only found {[path.name for path in modules]}"

    violations = {
        str(path.relative_to(EVAL_ROOT)): found
        for path in modules
        if (found := _violations(path))
    }
    assert not violations, (
        f"{violations}: the eval package is forked from the predecessor rather than chained to "
        f"it, and it must stay runnable without Lightning. If an import is genuinely necessary, "
        f"add it to EXEMPTIONS with a reason."
    )


def test_the_check_catches_a_lazy_in_function_import(tmp_path: Path) -> None:
    """Non-vacuity: a module-level-only check would miss exactly this, and it is the likely shape.

    A future change needing "just one thing" from the old tree reaches for a lazy import inside
    the function that needs it, precisely because it looks less like a dependency.
    """
    scratch = tmp_path / "scratch.py"
    scratch.write_text(
        "def analyse():\n"
        "    from model.vae_teb_prediction.testing import metrics\n"
        "    return metrics\n",
        encoding="utf-8",
    )
    assert _violations(scratch) == ["model.vae_teb_prediction.testing"]


def test_the_check_catches_a_module_level_import(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch.py"
    scratch.write_text("import lightning.pytorch as pl\n", encoding="utf-8")
    assert _violations(scratch) == ["lightning.pytorch"]


def test_the_exemption_is_narrow_rather_than_a_blanket_allowance(tmp_path: Path) -> None:
    """``preflight`` may import ``trainer``; it may not therefore import anything at all."""
    scratch = tmp_path / "preflight.py"
    scratch.write_text(
        "from teb_vae.lag_attn.trainer import _check_stat_path\n"
        "from teb_vae.lag_attn.task import SeqVaeLagAttnTask\n",
        encoding="utf-8",
    )
    assert _violations(scratch) == ["teb_vae.lag_attn.task"]
