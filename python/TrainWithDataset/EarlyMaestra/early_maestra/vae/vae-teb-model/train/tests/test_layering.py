"""Package layering is one-way: ``utils/`` <- ``train/`` <- the model layer.

The layers, each allowed to depend only on the ones below it:

- ``utils/`` -- leaf helpers. Depends on no other first-party package.
- ``train/`` -- the training framework. May use ``utils/``, never a model.
- ``model/`` and ``teb_vae/`` -- the model layer. May use both.

This is not a new rule; it is what the tree already does. The guard exists because the
direction is invisible at a call site and a single upward import is enough to invert it:
``train/callbacks.py`` is model-agnostic only for as long as nothing teaches it a batch
field name, and it cannot learn one without first importing a model.

The concrete near-miss this was written for: the SeqVAE plotters were moved out of
``train/callbacks.py`` into ``utils/seqvae_plot_callbacks.py``, and they needed the
``log_artifact_to_mlflow`` seam that stayed behind -- which would have made ``utils/``
import ``train/``. The fix was to move the seam down to ``utils/mlflow_utils.py``, where
both layers can reach it. Without this test that inversion is a code-review catch, and
code review had already missed it once.

One rule here is not about layering. ``teb_vae/`` is the tree that replaces ``model/``, and the
two are forked rather than chained: the replacement must never depend on what it supersedes, or
retiring ``model/`` would mean untangling it again. ``teb_vae`` sits at the top of the stack, so
it has nothing above it to import and needs no upward rule of its own -- the rule it needs is
sideways, against ``model``.

Scope is per-rule. ``utils/`` and ``train/`` are checked at their top level only
(``glob("*.py")``, matching ``test_migration_guide_citations.py``); recursing would sweep
``train/tests/*.py``, which import ``model/`` deliberately. ``teb_vae/`` is checked recursively
(``rglob``) because it is nested and its rule must hold at every depth, tests included. ``ast``
sees lazy imports inside functions and ``TYPE_CHECKING`` blocks too, which is deliberate -- a
deferred import is still a dependency.
"""
import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# (package, packages it must not import from, scan nested subpackages too)
_RULES = [
    ("utils", ("train", "model", "teb_vae"), False),
    ("train", ("model", "teb_vae"), False),
    ("teb_vae", ("model",), True),
]


def _imported_roots(path):
    """Return the set of top-level package names ``path`` imports.

    Relative imports (``level > 0``) are skipped: they cannot leave their own package,
    so they can never cross a layer boundary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("package,forbidden,recursive", _RULES)
def test_layer_does_not_import_forbidden_packages(package, forbidden, recursive):
    root = _REPO_ROOT / package
    modules = sorted(root.rglob("*.py") if recursive else root.glob("*.py"))
    # A silently-empty glob would make this test vacuous.
    assert modules, f"no modules found under {package}/"

    # Report the path relative to the repo root, not the bare basename: under a recursive scan
    # two modules at different depths can share a name, and a bare name would not say which.
    violations = [
        f"{path.relative_to(_REPO_ROOT).as_posix()} imports "
        f"{sorted(_imported_roots(path) & set(forbidden))}"
        for path in modules
        if _imported_roots(path) & set(forbidden)
    ]
    assert not violations, (
        f"{package}/ must not import from {', '.join(f'{p}/' for p in forbidden)} -- "
        f"dependencies run one way, utils/ <- train/ <- the model layer, and teb_vae/ is forked "
        f"from the model/ tree it replaces rather than built on it:\n  " + "\n  ".join(violations)
    )


def test_detects_a_forbidden_import(tmp_path):
    """The check above is only worth having if it actually fires."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "import os\n"
        "from train.callbacks import LossPlotCallback\n"
        "def f():\n"
        "    import model.vae_teb_small.trainer  # lazy imports count too\n",
        encoding="utf-8",
    )
    assert _imported_roots(offender) & {"train", "model"} == {"train", "model"}


def test_teb_vae_rule_reaches_nested_modules():
    """The fork boundary is only real if the scan descends past ``teb_vae/__init__.py``.

    ``teb_vae`` is the one package checked recursively, and it is also the one whose modules all
    live several levels down. A rule that stopped at the top level would scan one docstring-only
    file and pass forever.
    """
    package, _, recursive = next(rule for rule in _RULES if rule[0] == "teb_vae")
    assert recursive, "the teb_vae rule must scan nested subpackages"

    scanned = sorted((_REPO_ROOT / package).rglob("*.py"))
    assert any(
        path.parent != _REPO_ROOT / package for path in scanned
    ), "the teb_vae scan found no module below the package root"
