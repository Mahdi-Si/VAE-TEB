r"""What each module of this evaluation package is allowed to import, enforced by an AST walk.

The rules are the sibling's with one deliberate inversion. There, ``teb_vae.lag_attn_rws.eval`` is
the package being protected; here it is a **permitted dependency at every layer**, because reusing
it is the entire design: this package is nine modules against thirty-four precisely because every
analysis, statistic, identity check and figure primitive is imported rather than owned. A rule
forbidding that reach would be a rule against the architecture.

Everything else carries over, and each rule still has teeth:

* **``lag_attn_rws.nets`` is permitted.** This model imports the shared geometry, raw targets and
  masks, objective, prior and posterior heads, lag cross-attention and decoder unchanged, so the
  evaluation reads the same modules the model does.
* **``task`` and ``trainer`` only through :data:`EXEMPTIONS`**, named module by named target, and
  the table is asserted minimal -- a permission that outlives its use makes the next reach for
  that name unreported.
* **``model/*`` is forbidden everywhere.** It is the tree both of these packages supersede.
* **No analysis imports another analysis.** Anything two of them share moves one layer down.
* **``verify`` may not import ``torch``**, so a summary produced on the production box can be
  gated on a machine with nothing installed.

The walk resolves aliased, lazy and relative imports before applying any rule, because all three
are shapes a name-based scan waves through: ``import lightning.pytorch as pl`` mentions no
forbidden string under its alias, a lazy ``import model.x`` inside the one function that needs it
is exactly what a future change reaches for, and ``from . import <sibling>`` between two analyses
names nothing at all.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import pytest

#: The package under scrutiny, and its directory.
PACKAGE = "teb_vae.lag_attn_transformer_rws.eval"
EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"

#: The sibling *evaluation* package, permitted everywhere. This is the inversion: the sibling's
#: own walk allows its neighbour only through a narrow allow-list of model-free modules, while
#: here the whole package is the dependency the design rests on.
SIBLING_EVAL = "teb_vae.lag_attn_rws.eval"

#: The sibling's shared network modules, permitted everywhere for the same reason: this model
#: imports them unchanged, so an evaluation that read a second copy would be measuring something
#: the model does not compute.
SIBLING_NETS = "teb_vae.lag_attn_rws.nets"

#: Import prefixes no module under ``eval/`` may reach for, whatever its layer.
FORBIDDEN_PREFIXES: Tuple[str, ...] = (
    "model",
    "lightning",
    "pytorch_lightning",
    "teb_vae.lag_attn_rws.task",
    "teb_vae.lag_attn_rws.trainer",
    "teb_vae.lag_attn_rws.plotting",
    # This package's own training path. Forbidden by default and reachable only from ``binding``,
    # which is what keeps the coupling to the architecture in one file: the task is how a
    # checkpoint becomes a scored model, and a second module reaching for it is a second place
    # that decides what is being evaluated. The trainer has no exemption at all -- an evaluation
    # that reached the training driver would be building an experiment rather than reading one.
    "teb_vae.lag_attn_transformer_rws.task",
    "teb_vae.lag_attn_transformer_rws.trainer",
)

#: Module stems at layers 1 and 3 -- the ones that touch the checkpoint, the model or the run.
#: Only these may carry an entry in :data:`EXEMPTIONS`; a layer-0 or layer-2 module reaching for
#: ``task`` or ``trainer`` is reported no matter what the table says.
MODEL_TOUCHING: FrozenSet[str] = frozenset({"binding", "run"})

#: Modules that must stay importable with **no** numeric stack installed, so ``torch`` joins their
#: forbidden list on top of the layer rules. Proved on the import graph rather than by an
#: uninstalled-torch harness, which could only ever prove it for one environment.
NO_TORCH_MODULES: FrozenSet[str] = frozenset({f"{PACKAGE}.verify"})

#: ``module stem -> the forbidden prefixes it may import anyway``. Narrow by construction: named
#: modules, named targets, one stated reason each.
#:
#: ``binding`` names the two classes a checkpoint is rebuilt into, which is the whole content of a
#: model binding and the reason a concrete one cannot live in the shared layer-0 module.
#:
#: The table is asserted **minimal**.
EXEMPTIONS: Dict[str, Set[str]] = {
    "binding": {"teb_vae.lag_attn_transformer_rws.task"},
}


# =============================================================================
# The walker
# =============================================================================
def _module_name_for(path: Path) -> str:
    """Return the dotted module name a file under ``eval/`` has.

    A package's ``__init__.py`` keeps its ``__init__`` suffix rather than collapsing to the
    package name: relative-import levels count from the *module*, so ``from . import x`` inside a
    package's ``__init__`` resolves to that package, which is what dropping one component from
    ``<package>.__init__`` gives.

    Args:
        path: Path to a ``.py`` file under :data:`EVAL_ROOT`.

    Returns:
        The dotted module name.
    """
    parts = path.relative_to(EVAL_ROOT).with_suffix("").parts
    return ".".join((PACKAGE, *parts))


def _resolve_relative(module_name: str, level: int, module: str | None) -> str:
    """Resolve a relative ``from ... import`` to an absolute dotted name.

    Args:
        module_name: The importing module's own dotted name.
        level: The number of leading dots.
        module: The dotted name after the dots, or ``None`` for ``from . import x``.

    Returns:
        The absolute package the import reads from. An over-long level walks off the top and
        yields what is left, which cannot match any rule and is a syntax error at runtime anyway.
    """
    parts = module_name.split(".")
    base = parts[: max(len(parts) - level, 0)]
    return ".".join(base + ([module] if module else []))


def imported_names(source: str, module_name: str) -> List[str]:
    """Return every module name ``source`` imports, resolved to absolute dotted form.

    Lazy in-function imports are walked exactly like module-level ones, and relative imports are
    resolved against ``module_name``. For ``from <package> import a, b`` both the package and
    ``<package>.a`` / ``<package>.b`` are returned, because an imported name may be a submodule
    and the caller cannot tell which from the syntax alone.

    Args:
        source: Python source text.
        module_name: The importing module's own dotted name.

    Returns:
        The candidate imported module names, in encounter order and possibly with duplicates.
    """
    names: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            package = (
                node.module or ""
                if node.level == 0
                else _resolve_relative(module_name, node.level, node.module)
            )
            if not package:
                continue
            names.append(package)
            names.extend(f"{package}.{alias.name}" for alias in node.names)
    return names


def _matches(name: str, prefix: str) -> bool:
    """Whether ``name`` is ``prefix`` or a module beneath it."""
    return name == prefix or name.startswith(prefix + ".")


def forbidden_imports(source: str, module_name: str) -> List[str]:
    """Return the imports ``module_name`` is not allowed to make.

    Args:
        source: The module's source text.
        module_name: Its dotted name, which decides both its layer and how its relative imports
            resolve.

    Returns:
        The offending absolute names. A name whose own prefix is already reported is dropped, so
        ``from ..trainer import X`` is reported once rather than twice.
    """
    stem = module_name.rsplit(".", 1)[-1]
    under_analyses = module_name.startswith(f"{PACKAGE}.analyses.")
    # An exemption is a property of a layer-1/3 module. A stem collision with an analysis must
    # not hand that analysis the same permission.
    allowed = (
        EXEMPTIONS.get(stem, set()) if stem in MODEL_TOUCHING and not under_analyses else set()
    )

    violations: List[str] = []
    for name in imported_names(source, module_name):
        if module_name in NO_TORCH_MODULES and _matches(name, "torch"):
            violations.append(name)
            continue

        # Checked before the forbidden list: the sibling's eval package is permitted at every
        # layer even though its own package prefix sits under a forbidden one's neighbourhood.
        if _matches(name, SIBLING_EVAL) or _matches(name, SIBLING_NETS):
            continue

        if any(_matches(name, prefix) for prefix in FORBIDDEN_PREFIXES):
            if not any(_matches(name, permitted) for permitted in allowed):
                violations.append(name)
            continue

        if under_analyses and name.startswith(f"{PACKAGE}.analyses."):
            other = name[len(f"{PACKAGE}.analyses.") :].split(".")[0]
            if other != stem:
                violations.append(name)

    unique = list(dict.fromkeys(violations))
    return [
        name
        for name in unique
        if not any(_matches(name, other) for other in unique if other != name)
    ]


# =============================================================================
# The shipped package
# =============================================================================
def _shipped_modules() -> List[Path]:
    """Every ``.py`` that ships under ``eval/``. The tests live one directory up, not here."""
    return sorted(EVAL_ROOT.rglob("*.py"))


def test_the_walk_found_modules_to_check() -> None:
    """A walk that found nothing would pass every other test in this file vacuously."""
    modules = _shipped_modules()
    assert len(modules) >= 4, f"only found {[path.name for path in modules]}"


@pytest.mark.parametrize(
    "module", _shipped_modules(), ids=lambda path: str(path.relative_to(EVAL_ROOT))
)
def test_every_shipped_module_stays_inside_its_layer(module: Path) -> None:
    """Walked per module so a failure names the file rather than the package."""
    module_name = _module_name_for(module)
    violations = forbidden_imports(module.read_text(encoding="utf-8"), module_name)
    assert not violations, (
        f"{module.relative_to(EVAL_ROOT)} imports {violations}. No Lightning and no model/ at any "
        f"layer; task and trainer only through EXEMPTIONS; no analysis imports another. The "
        f"sibling's eval package and nets are permitted everywhere -- that is the design. If the "
        f"import is genuinely necessary, add it to EXEMPTIONS with a reason."
    )


def test_the_binding_is_the_only_module_naming_a_model_class() -> None:
    """The whole coupling to this architecture is four facts in one file. A second module reaching
    for the net or the task would put the fork this package exists to avoid back in the tree."""
    reaching = [
        module.stem
        for module in _shipped_modules()
        if any(
            _matches(name, "teb_vae.lag_attn_transformer_rws.nets")
            or _matches(name, "teb_vae.lag_attn_transformer_rws.task")
            for name in imported_names(module.read_text(encoding="utf-8"), _module_name_for(module))
        )
    ]
    assert reaching == ["binding"]


def test_every_torch_free_module_actually_ships() -> None:
    """The rule is only enforced on files the walk finds. A ``NO_TORCH_MODULES`` entry naming a
    module that does not exist -- a rename, a module planned and never written -- would leave the
    property asserted and unchecked, which is the failure this file exists to make impossible."""
    shipped = {_module_name_for(module) for module in _shipped_modules()}

    assert NO_TORCH_MODULES <= shipped, f"missing: {sorted(NO_TORCH_MODULES - shipped)}"


def test_no_module_holds_a_permission_it_does_not_use() -> None:
    """The guard with teeth: a permission outlives its use silently, and the next reach for that
    name is then unreported. Every exempted name must appear in the module's actual imports."""
    by_stem = {module.stem: module for module in _shipped_modules()}

    unused = []
    for stem, permitted in EXEMPTIONS.items():
        module = by_stem.get(stem)
        assert module is not None, f"EXEMPTIONS names {stem!r}, which ships no module"
        names = set(imported_names(module.read_text(encoding="utf-8"), _module_name_for(module)))
        unused.extend(
            f"{stem} -> {target}"
            for target in sorted(permitted)
            if not any(_matches(name, target) for name in names)
        )

    assert unused == [], f"EXEMPTIONS grants imports that no longer happen: {unused}"


# =============================================================================
# Non-vacuity: the shapes a name-based check would miss or wave through
# =============================================================================
def test_an_aliased_lightning_import_is_reported() -> None:
    assert forbidden_imports("import lightning.pytorch as pl\n", f"{PACKAGE}.verify") == [
        "lightning.pytorch"
    ]


def test_a_lazy_in_function_import_is_reported() -> None:
    """A module-level-only check misses exactly this, and it is the likely shape: a change needing
    "just one thing" reaches for a lazy import inside the function that needs it."""
    source = (
        "def analyse():\n"
        "    from model.lstm_cnn_vae_teb.testing import metrics\n"
        "    return metrics\n"
    )
    assert forbidden_imports(source, f"{PACKAGE}.analyses.encoder_attention") == [
        "model.lstm_cnn_vae_teb.testing"
    ]


def test_a_relative_sibling_import_between_analyses_is_reported() -> None:
    """Analyses never import one another: anything two of them share moves one layer down."""
    assert forbidden_imports("from . import forecast\n", f"{PACKAGE}.analyses.coupling") == [
        f"{PACKAGE}.analyses.forecast"
    ]
    absolute = f"from {PACKAGE}.analyses.forecast import baseline\n"
    assert forbidden_imports(absolute, f"{PACKAGE}.analyses.coupling") == [
        f"{PACKAGE}.analyses.forecast"
    ]


def test_a_relative_parent_import_of_the_trainer_is_reported() -> None:
    """``from ..trainer import x`` names no forbidden string; it has to be resolved first."""
    assert forbidden_imports("from ..trainer import LagAttnTrfRwsTrainer\n", f"{PACKAGE}.run") == [
        "teb_vae.lag_attn_transformer_rws.trainer"
    ]


def test_the_exemption_permits_only_its_named_target() -> None:
    """``binding`` may take this package's ``task``; it does not thereby take anything at all."""
    source = (
        "from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask\n"
        "import lightning as L\n"
    )
    assert forbidden_imports(source, f"{PACKAGE}.binding") == ["lightning"]


def test_a_layer_two_module_gets_no_exemption_from_a_stem_collision() -> None:
    """``analyses/binding.py`` would share ``binding``'s stem; it must not share its permission."""
    source = "from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask\n"
    assert forbidden_imports(source, f"{PACKAGE}.analyses.binding") == [
        "teb_vae.lag_attn_transformer_rws.task"
    ]


def test_the_siblings_evaluation_package_is_permitted_at_every_layer() -> None:
    """The inversion, asserted rather than left implicit: reusing that package *is* the design."""
    source = (
        "from teb_vae.lag_attn_rws.eval import run as shared_run\n"
        "from teb_vae.lag_attn_rws.eval.analyses import coupling\n"
        "from teb_vae.lag_attn_rws.eval.binding import ModelBinding\n"
        "from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP\n"
    )
    assert forbidden_imports(source, f"{PACKAGE}.analyses.encoder_attention") == []
    assert forbidden_imports(source, f"{PACKAGE}.run") == []


def test_the_siblings_task_and_trainer_are_still_forbidden() -> None:
    """Its *evaluation* package is the dependency, not its training path. Reaching the sibling's
    task from here would rebuild the wrong model."""
    assert forbidden_imports(
        "from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask\n", f"{PACKAGE}.binding"
    ) == ["teb_vae.lag_attn_rws.task"]
    assert forbidden_imports(
        "from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME\n", f"{PACKAGE}.run"
    ) == ["teb_vae.lag_attn_rws.trainer"]


def test_the_acceptance_gate_may_not_import_torch() -> None:
    """Its one non-negotiable property: a summary produced on the production box can be checked on
    a machine with nothing installed."""
    assert forbidden_imports("import torch\n", f"{PACKAGE}.verify") == ["torch"]
    assert forbidden_imports("from torch import Tensor\n", f"{PACKAGE}.verify") == ["torch"]
    assert forbidden_imports(
        "def check():\n    import torch.nn as nn\n", f"{PACKAGE}.verify"
    ) == ["torch.nn"]
    # The rule is the gate's own, not the package's: the binding names a torch module by design.
    assert forbidden_imports("import torch\n", f"{PACKAGE}.binding") == []
