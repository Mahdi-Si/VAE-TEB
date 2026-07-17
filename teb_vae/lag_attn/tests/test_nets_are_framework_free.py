"""``nets/`` may import torch, the standard library and ``entmax``. Nothing else.

The rule exists so a network component can be constructed, forwarded and asserted without a
config file, a logger, a run directory, a process group, or a GPU. Every import that creeps in
past this line takes one of those away, and it is never obvious at the call site which one.

The specific inversions this catches, all of which were real in the tree this replaces:

* ``loguru`` -- a net that logs needs logging configured before it can be built.
* ``torch.distributed`` -- a net that reduces across ranks needs a process group to run at all,
  which turns a unit test into an integration test. Its import root is ``torch``, so a
  root-name allowlist alone would wave it straight through; it is checked by dotted path.
* ``numpy`` -- not wrong so much as a signal: it means a computation left the autograd graph.
* ``lightning`` / ``train`` / ``utils`` -- the framework. A net that reaches into it stops being
  reusable outside it.
* ``model`` -- the tree this package replaces, checked here as well as in the layering suite.

The batch-field check is the same rule from the other side. A net that reads ``batch.fhr_st`` has
learned the dataset's schema, and every such name is a place the data layer and the model layer
have to agree without anything checking that they do. Tensors go in as arguments; what they were
called on disk is the caller's business.

That check matches on word boundaries, which is not incidental. ``use_up_st`` is a constructor
argument -- an ablation toggle the model legitimately owns, and part of its public config
contract -- while ``batch.up_st`` is a schema read. A bare substring search cannot tell them
apart and would fail on the former, so the guard would get loosened or deleted, which is how
guards die.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"

# The rule, exactly: torch, the standard library, entmax -- plus this package itself, since the
# modules import each other. Deriving the stdlib set rather than listing it by hand keeps the
# guard honest: a hand-written list gets extended every time it fires, which is backwards.
_ALLOWED_ROOTS = {"torch", "entmax", "teb_vae"} | set(sys.stdlib_module_names)

# Forbidden by full dotted path, for cases an allowed root would otherwise admit.
_FORBIDDEN_PREFIXES = ("torch.distributed",)

# Field names from the HDF5 batch contract. A net must not know these exist.
_BATCH_FIELD_NAMES = ("fhr_st", "fhr_ph", "up_st", "up_ph", "fhr_up_ph", "cs_label", "bg_label")


def _net_modules() -> list[Path]:
    return sorted(_NETS_DIR.glob("*.py"))


def _imported_names(path: Path) -> set[str]:
    """Return the full dotted module names ``path`` imports.

    Walks the whole tree, so a lazy import inside a function counts too -- deferring an import
    does not undo the dependency, it only hides it until runtime.

    Args:
        path: The module to scan.

    Returns:
        Every absolute imported module name. Relative imports are skipped: they cannot leave
        this package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_there_are_net_modules_to_check():
    """A silently-empty glob would make every test below vacuous."""
    assert _net_modules(), f"no modules found under {_NETS_DIR}"


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_imports_only_torch_stdlib_and_entmax(path):
    offenders = sorted(
        name for name in _imported_names(path) if name.split(".")[0] not in _ALLOWED_ROOTS
    )
    assert not offenders, (
        f"nets/{path.name} imports {offenders} -- nets/ may import only torch, the standard "
        f"library and entmax, so that a network can be built without the framework around it"
    )


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_avoids_forbidden_submodules(path):
    offenders = sorted(
        name
        for name in _imported_names(path)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_PREFIXES)
    )
    assert not offenders, (
        f"nets/{path.name} imports {offenders} -- a net must not need a process group to run"
    )


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_names_no_batch_fields(path):
    source = path.read_text(encoding="utf-8")
    offenders = sorted(
        name for name in _BATCH_FIELD_NAMES if re.search(rf"\b{name}\b", source)
    )
    assert not offenders, (
        f"nets/{path.name} names the batch fields {offenders} -- a net takes tensors as "
        f"arguments and does not know what they were called on disk"
    )


def test_the_batch_field_guard_fires(tmp_path):
    """It must catch a schema read without catching a kwarg that merely contains the name."""
    reader = tmp_path / "reader.py"
    reader.write_text("def f(batch):\n    return batch.fhr_st\n", encoding="utf-8")
    assert [n for n in _BATCH_FIELD_NAMES if re.search(rf"\b{n}\b", reader.read_text())] == [
        "fhr_st"
    ]

    toggle = tmp_path / "toggle.py"
    toggle.write_text("def f(use_up_st=True):\n    return use_up_st\n", encoding="utf-8")
    assert not [
        n for n in _BATCH_FIELD_NAMES if re.search(rf"\b{n}\b", toggle.read_text())
    ]


def test_the_import_guard_fires(tmp_path):
    """A guard that cannot fail is not a guard."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "import torch\n"
        "from loguru import logger\n"
        "def f():\n"
        "    import lightning  # lazy imports count too\n",
        encoding="utf-8",
    )
    roots = {name.split(".")[0] for name in _imported_names(offender)}
    assert roots - _ALLOWED_ROOTS == {"loguru", "lightning"}


def test_the_dotted_guard_fires(tmp_path):
    """``torch.distributed`` has an allowed root, so only the dotted check can catch it."""
    offender = tmp_path / "offender.py"
    offender.write_text("import torch.distributed as dist\n", encoding="utf-8")

    names = _imported_names(offender)
    assert {name.split(".")[0] for name in names} <= _ALLOWED_ROOTS, "root check should pass here"
    assert any(
        name.startswith(_FORBIDDEN_PREFIXES[0]) for name in names
    ), "the dotted check must catch what the root check waves through"
