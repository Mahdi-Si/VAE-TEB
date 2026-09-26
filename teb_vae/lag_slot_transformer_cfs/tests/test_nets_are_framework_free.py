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


def test_no_network_module_imports_a_training_framework() -> None:
    """At module scope and inside functions alike, in every module under ``nets/``."""
    modules = net_modules()
    assert modules, f"no module found under {NETS_ROOT}; the check would pass vacuously"
    offending = {}
    for path in modules:
        roots = sorted(FORBIDDEN_ROOTS.intersection(imported_roots(path)))
        if roots:
            offending[path.relative_to(NETS_ROOT).as_posix()] = roots
    assert offending == {}
