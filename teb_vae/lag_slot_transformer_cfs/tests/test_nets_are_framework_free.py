r"""The network modules import no training framework.

The separation is what lets a module under ``nets/`` be exercised from a plain script and a unit
test with no trainer, no configuration and no data module -- which is how every algebraic check in
this suite runs in under a second. It is also what keeps the objective, which is the thing two
architectures must share, free of anything that would tie it to one training loop.

Enforced by reading the import statements rather than by importing and inspecting: an import that
happens inside a function would be invisible to the second method and is exactly the way this
property erodes.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest

#: The directory whose modules must stay framework-free.
NETS_ROOT = Path(__file__).resolve().parents[1] / "nets"

#: Top-level packages a network module may never import, at module scope or inside a function.
FORBIDDEN_ROOTS = frozenset(
    {
        "lightning",
        "pytorch_lightning",
        "mlflow",
        "matplotlib",
        "pandas",
        "h5py",
        "yaml",
    }
)


def net_modules() -> List[Path]:
    """Every module under the network directory.

    Returns:
        The paths, sorted.
    """
    return sorted(
        path for path in NETS_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    )


def imported_roots(path: Path) -> List[str]:
    """Top-level package names one module imports, wherever the import appears.

    Args:
        path: The module to read.

    Returns:
        The root names, in source order.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.append(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", net_modules(), ids=lambda p: p.name)
def test_a_network_module_imports_no_training_framework(path: Path) -> None:
    """At module scope and inside functions alike.

    Args:
        path: The module to check.
    """
    offending = sorted(FORBIDDEN_ROOTS.intersection(imported_roots(path)))
    assert offending == [], f"{path.name} imports {offending}"


def test_the_directory_is_not_empty() -> None:
    """So a rename that emptied it would fail here rather than pass every check above."""
    assert len(net_modules()) >= 5
