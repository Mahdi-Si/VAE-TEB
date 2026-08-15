r"""What each layer of the evaluation package is allowed to import, enforced by an AST walk.

The package is layered, and the layering is the thing that keeps an analysis from quietly becoming
a second training loop:

* **Layer 0 and layer 2** -- the pure modules and ``analyses/*`` -- import no Lightning, no
  ``model.*``, and none of this package's ``task``/``trainer``. An analysis receives tables and
  writes files; it does not rebuild a model.
* **Layer 1 and layer 3** -- the model-touching modules and the runner -- may import ``task`` and
  ``trainer``, but only where :data:`EXEMPTIONS` says so and only those. That exemption is
  deliberate rather than reluctant: the readout module assembles the model's inputs through the
  task's own builders precisely so an evaluation cannot feed the model a differently assembled
  stream than training did, and re-implementing those builders to win a layering rule would
  reintroduce the exact drift they exist to prevent.
* **No analysis imports another analysis.** Anything two of them share moves one layer down.
* **The shared evaluation package is reachable only through its model-free modules**
  (:data:`ALLOWED_SIBLING_EVAL_MODULES`), and only from the one seam that names it.
* **``model/*`` is forbidden everywhere, at every layer.** It is the tree this one supersedes.

**And the rule this package has that the raw pipeline does not: the pipeline it was forked from is
forbidden outright.** ``teb_vae.lag_attn_rws.eval`` is on the forbidden list in every form, because
a package that copied the analyses and then reached back into the sibling for one helper would have
two implementations *and* a dependency -- a half-fork, which is worse than either whole. The
exemption is a small, named set of **test** files, and :func:`test_the_sibling_eval_package_is_
reachable_only_from_the_named_test_files` is what keeps that set from growing quietly.

**One module sits at a different layer here than in the sibling**, and it is named rather than left
to be discovered: ``binding`` is layer 0 there, where it holds only the dataclass, and layer 1
here, where it also holds this cell's concrete ``CFS_BINDING`` and therefore names a model class.
The one consequence that matters is enforced below: nothing which must import without ``torch``
may import it.

The walk is what makes every rule real, and it resolves more than a name-based scan can. A lazy
``import model.x`` inside the one function that needs it is exactly the shortcut a future change
reaches for, precisely because it looks less like a dependency; a relative ``from ..trainer import
x`` and a relative ``from . import <sibling>`` never mention a forbidden string at all. All three
are resolved to absolute dotted names before any rule is applied.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import pytest

#: The package under scrutiny, and its directory.
PACKAGE = "teb_vae.lag_attn_cfs.eval"
EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"

#: The pipeline this package is a fork of. Forbidden everywhere under ``eval/``.
FORKED_FROM = "teb_vae.lag_attn_rws.eval"

#: Import prefixes no module under ``eval/`` may reach for, whatever its layer.
#:
#: ``teb_vae.lag_attn_rws.plotting`` is a Lightning callback, so it is forbidden for the same
#: reason ``lightning`` is; the rest of that package is **not** forbidden, because this cell's nets
#: are built out of its ``nets/`` layer and the acceptance gate deliberately imports its
#: ``collapse`` module, which is stdlib-only.
#:
#: ``teb_vae.lag_attn_transformer_cfs`` is forbidden so the dependency between the two cfs cells
#: runs one way only: that package binds this pipeline, and a reach back would be a cycle in which
#: importing either costs both.
FORBIDDEN_PREFIXES: Tuple[str, ...] = (
    "model",
    "lightning",
    "pytorch_lightning",
    "teb_vae.lag_attn_cfs.task",
    "teb_vae.lag_attn_cfs.trainer",
    "teb_vae.lag_attn_rws.plotting",
    "teb_vae.lag_attn_transformer_cfs",
    FORKED_FROM,
)

#: The shared evaluation package, and the modules within it this one may bind. Every entry is
#: model-free in the sense that matters: none of them constructs, loads or forwards a network.
SIBLING_EVAL = "teb_vae.lag_attn.eval"
ALLOWED_SIBLING_EVAL_MODULES: FrozenSet[str] = frozenset(
    {
        "band_partition",
        "collectors",
        "config_schema",
        "figures",
        "labels",
        "masks",
        "numerics",
        "report",
        "stats",
    }
)

#: Module stems at layers 1 and 3 -- the ones that touch the checkpoint, the model or the run.
#: Only these may carry an entry in :data:`EXEMPTIONS`; a layer-0 or layer-2 module reaching for
#: ``task`` or ``trainer`` is reported no matter what the table says.
#:
#: Two are here and are not on the sibling's list, which is the whole of the layering difference
#: between the two packages. ``binding`` holds this cell's concrete instance and therefore names a
#: model class. ``probe`` loads a checkpoint: this cell's forward takes five arguments and raises
#: without a phase above stride $1$, so its contract is *measured* against a rebuilt model rather
#: than read, where the sibling's probe never leaves the loader.
#:
#: Neither carries an exemption for ``probe``: it reaches the task through ``binding.task_cls``
#: rather than by importing it, so the one import site stays the binding's.
MODEL_TOUCHING: FrozenSet[str] = frozenset(
    {"binding", "metrics", "collect", "preflight", "probe", "oracle", "run"}
)

#: Modules that must stay importable with **no** numeric stack installed, so ``torch`` joins their
#: forbidden list on top of the layer rules -- and so does this package's ``binding``, which names
#: a model class and would pull one in transitively. The acceptance gate reads a finished run's
#: ``summary.json`` on whatever box the file was copied to, and proving the property here -- on the
#: import graph -- replaces an awkward uninstalled-torch harness that could only ever prove it for
#: one environment.
NO_TORCH_MODULES: FrozenSet[str] = frozenset({f"{PACKAGE}.verify"})

#: What a no-torch module may not import beyond ``torch`` itself.
NO_TORCH_FORBIDDEN: Tuple[str, ...] = ("torch", f"{PACKAGE}.binding")

#: ``module stem -> the forbidden prefixes it may import anyway``. Narrow by construction: named
#: modules, named targets, one stated reason each.
#:
#: ``binding`` names the task the checkpoint is scored through, which is the fact the pipeline
#: cannot derive and the reason the module exists.
#:
#: The table is asserted **minimal**: a module listing a name it does not import is a permission
#: that outlived its use, and the next reach for that name would go unreported.
EXEMPTIONS: Dict[str, Set[str]] = {
    "binding": {"teb_vae.lag_attn_cfs.task"},
}

#: The test files allowed to import the forked-from pipeline, each for a stated reason. Every
#: other file in this suite -- and every module under ``eval/`` -- is refused.
SIBLING_EVAL_TEST_EXEMPTIONS: Dict[str, str] = {
    "test_eval_sibling_agreement.py":
        "re-derives the shared arithmetic through both packages and asserts equality; that is the "
        "fork's anti-drift measure and it cannot be written without importing both",
    "test_eval_reuse.py":
        "pins the shared model-free primitives to the same objects both packages bind, by "
        "identity rather than by value",
    "test_eval_config_schema.py":
        "pins this package's eval_config key set against the sibling's, so the one added key is "
        "the only difference",
}


# =============================================================================
# The walker
# =============================================================================
def _module_name_for(path: Path) -> str:
    """Return the dotted module name a file under ``eval/`` has.

    A package's ``__init__.py`` keeps its ``__init__`` suffix rather than collapsing to the
    package name. That is not cosmetic: relative-import levels count from the *module*, so
    ``from . import x`` inside a package's ``__init__`` resolves to that package, which is what
    dropping one component from ``<package>.__init__`` gives.

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


def _sibling_eval_module(name: str) -> str | None:
    """Return the shared-eval submodule ``name`` reaches into, if any.

    Args:
        name: An absolute dotted import name.

    Returns:
        The first component below ``teb_vae.lag_attn.eval``, or ``None`` when the name is not
        under it. Importing the shared eval *package itself* yields ``None``: its ``__init__`` is
        a docstring, so the package name alone reaches nothing.
    """
    if not name.startswith(SIBLING_EVAL + "."):
        return None
    return name[len(SIBLING_EVAL) + 1 :].split(".")[0]


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
    exempt = stem in MODEL_TOUCHING and not under_analyses
    allowed = EXEMPTIONS.get(stem, set()) if exempt else set()

    violations: List[str] = []
    for name in imported_names(source, module_name):
        if module_name in NO_TORCH_MODULES and any(
            _matches(name, prefix) for prefix in NO_TORCH_FORBIDDEN
        ):
            violations.append(name)
            continue

        if any(_matches(name, prefix) for prefix in FORBIDDEN_PREFIXES):
            if not any(_matches(name, permitted) for permitted in allowed):
                violations.append(name)
            continue

        sibling = _sibling_eval_module(name)
        if sibling is not None and sibling not in ALLOWED_SIBLING_EVAL_MODULES:
            violations.append(name)
            continue

        if under_analyses and name.startswith(f"{PACKAGE}.analyses."):
            other = name[len(f"{PACKAGE}.analyses.") :].split(".")[0]
            if other != stem:
                violations.append(name)

    unique = list(dict.fromkeys(violations))
    return [
        name for name in unique
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
        f"{module.relative_to(EVAL_ROOT)} imports {violations}. Layers 0 and 2 take no Lightning, "
        f"no model/ and no task/trainer; layers 1 and 3 take task and trainer only through "
        f"EXEMPTIONS; the shared evaluation package is reachable only through "
        f"{sorted(ALLOWED_SIBLING_EVAL_MODULES)}; and {FORKED_FROM} is forbidden outright. If the "
        f"import is genuinely necessary, add it to EXEMPTIONS with a reason."
    )


def test_the_model_binding_is_walked_and_is_this_packages_one_layer_difference() -> None:
    """The walk is directory-driven, so this is what says the file was actually picked up rather
    than that a rule happened to hold over a set it was not in.

    ``binding`` holds the facts the pipeline cannot derive about the model it evaluates, and here
    it also holds the concrete instance -- so it names a model class, sits at layer 1, and carries
    exactly one exemption. The sibling keeps its instance in the runner and its ``binding`` at
    layer 0; that difference is stated here rather than left for a reader to infer from a
    permission table.
    """
    stems = {module.stem for module in _shipped_modules()}
    assert "binding" in stems

    assert "binding" in MODEL_TOUCHING
    assert EXEMPTIONS["binding"] == {"teb_vae.lag_attn_cfs.task"}

    binding = next(module for module in _shipped_modules() if module.stem == "binding")
    source = binding.read_text(encoding="utf-8")
    assert forbidden_imports(source, _module_name_for(binding)) == []
    # The exemption is for the task alone: naming the model class needs no further permission,
    # and Lightning is still refused here as everywhere.
    names = imported_names(source, _module_name_for(binding))
    assert not any(_matches(name, "lightning") for name in names)


def test_the_reuse_seam_is_the_only_module_naming_the_shared_evaluation_package() -> None:
    """The seam exists so the coupling is visible in one file; a second reach would hide it.

    ``config_schema`` is the stated exception: it takes the shared validators directly so that
    validating a run's settings costs a stdlib parse rather than a matplotlib import.
    """
    reaching = []
    for module in _shipped_modules():
        names = imported_names(module.read_text(encoding="utf-8"), _module_name_for(module))
        if any(_sibling_eval_module(name) is not None for name in names):
            reaching.append(module.stem)
    assert sorted(reaching) == ["_reuse", "config_schema"]


# =============================================================================
# The forked-from pipeline, in every form a reach could take
# =============================================================================
@pytest.mark.parametrize(
    "source, expected",
    [
        (f"from {FORKED_FROM} import frames\n", FORKED_FROM),
        (f"import {FORKED_FROM}.frames as sibling_frames\n", f"{FORKED_FROM}.frames"),
        (
            f"def f():\n    from {FORKED_FROM}.metrics import evaluate_batch\n",
            f"{FORKED_FROM}.metrics",
        ),
        # Four levels up from an analysis is ``teb_vae``, which is where the forked-from package
        # sits: the shortest relative path to it, and one that mentions no forbidden string.
        ("from ....lag_attn_rws.eval import cohort\n", FORKED_FROM),
    ],
    ids=["absolute", "aliased", "lazy", "relative"],
)
def test_a_reach_into_the_forked_from_pipeline_is_reported(source: str, expected: str) -> None:
    """Four forms, because a name-based scan sees only the first two and the relative one mentions
    no forbidden string at all until it is resolved."""
    assert forbidden_imports(source, f"{PACKAGE}.analyses.coupling") == [expected]


def test_the_rest_of_the_forked_from_package_is_still_reachable() -> None:
    """The ban is on the *evaluation* package, not on the model package around it: this cell's
    objective, masks and controls live in that package's ``nets/``, and the acceptance gate reads
    its stdlib-only collapse criterion rather than owning a second copy."""
    source = (
        "from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask\n"
        "from teb_vae.lag_attn_rws.collapse import is_collapsed\n"
        "from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME\n"
    )

    assert forbidden_imports(source, f"{PACKAGE}.preflight") == []


def test_the_forked_from_pipelines_lightning_callback_is_not() -> None:
    assert forbidden_imports(
        "from teb_vae.lag_attn_rws.plotting import DiagnosticPageCallback\n", f"{PACKAGE}.run"
    ) == ["teb_vae.lag_attn_rws.plotting"]


def test_the_sibling_eval_package_is_reachable_only_from_the_named_test_files() -> None:
    """The exemption the walk above cannot express, since it walks ``eval/`` and these are tests.

    Both directions: a fourth test file importing the forked-from pipeline fails here, and so does
    an exemption for a file that no longer imports it -- a permission that outlived its use is how
    the next reach goes unreported.
    """
    tests_root = Path(__file__).resolve().parent
    reaching = set()
    for path in sorted(tests_root.glob("*.py")):
        names = imported_names(path.read_text(encoding="utf-8"), f"tests.{path.stem}")
        if any(_matches(name, FORKED_FROM) for name in names):
            reaching.add(path.name)

    assert reaching == set(SIBLING_EVAL_TEST_EXEMPTIONS), (
        f"only in the exemption table: {sorted(set(SIBLING_EVAL_TEST_EXEMPTIONS) - reaching)}; "
        f"only in the suite: {sorted(reaching - set(SIBLING_EVAL_TEST_EXEMPTIONS))}"
    )
    assert all(reason.strip() for reason in SIBLING_EVAL_TEST_EXEMPTIONS.values())


# =============================================================================
# Non-vacuity: the shapes a name-based check would miss or wave through
# =============================================================================
def test_an_aliased_lightning_import_is_reported() -> None:
    assert forbidden_imports("import lightning.pytorch as pl\n", f"{PACKAGE}.frames") == [
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
    assert forbidden_imports(source, f"{PACKAGE}.analyses.forecast") == [
        "model.lstm_cnn_vae_teb.testing"
    ]


def test_a_relative_parent_import_is_reported() -> None:
    """``from ..trainer import x`` names no forbidden string; it has to be resolved first."""
    assert forbidden_imports(
        "from ..trainer import LagAttnCfsTrainer\n", f"{PACKAGE}.verify"
    ) == ["teb_vae.lag_attn_cfs.trainer"]


def test_a_relative_sibling_import_between_analyses_is_reported() -> None:
    """The rule with no counterpart in the shared package: analyses never import one another."""
    assert forbidden_imports("from . import forecast\n", f"{PACKAGE}.analyses.coupling") == [
        f"{PACKAGE}.analyses.forecast"
    ]
    absolute = f"from {PACKAGE}.analyses.forecast import baseline\n"
    assert forbidden_imports(absolute, f"{PACKAGE}.analyses.coupling") == [
        f"{PACKAGE}.analyses.forecast"
    ]


def test_an_analysis_may_import_its_own_module_and_the_layers_below_it() -> None:
    source = (
        f"from {PACKAGE} import config_schema, events, frames, lag_axis\n"
        f"from {PACKAGE}._reuse import stats\n"
        "from teb_vae.lag_attn.nets.lag_report import lag_compensated_seconds\n"
        "import numpy as np\n"
    )
    assert forbidden_imports(source, f"{PACKAGE}.analyses.coupling") == []


# =============================================================================
# The exemptions are narrow, and so is the sibling allow-list
# =============================================================================
def test_the_exemption_permits_only_its_named_targets() -> None:
    """``binding`` may take the task; it does not thereby take anything at all."""
    source = (
        "from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask\n"
        "from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer\n"
        "import lightning as L\n"
    )
    assert forbidden_imports(source, f"{PACKAGE}.binding") == [
        "teb_vae.lag_attn_cfs.trainer", "lightning",
    ]


def test_a_layer_two_module_gets_no_exemption_from_a_stem_collision() -> None:
    """``analyses/binding.py`` would share ``binding``'s stem; it must not share its permission."""
    source = "from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask\n"
    assert forbidden_imports(source, f"{PACKAGE}.analyses.binding") == [
        "teb_vae.lag_attn_cfs.task"
    ]


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


def test_the_acceptance_gate_may_not_import_torch_or_the_binding_that_names_a_model() -> None:
    """The gate's one non-negotiable property is that a summary produced on the production box can
    be checked on a machine with nothing installed. ``torch`` is the direct way to break it and
    this package's ``binding`` is the indirect one -- it names the model class, so importing it
    costs the whole numeric stack. The gate reads a finished ``summary.json`` and needs neither."""
    assert forbidden_imports("import torch\n", f"{PACKAGE}.verify") == ["torch"]
    assert forbidden_imports("from torch import Tensor\n", f"{PACKAGE}.verify") == ["torch"]
    assert forbidden_imports("def check():\n    import torch.nn as nn\n", f"{PACKAGE}.verify") == [
        "torch.nn"
    ]
    reach = f"from {PACKAGE}.binding import CFS_BINDING\n"
    assert forbidden_imports(reach, f"{PACKAGE}.verify") == [f"{PACKAGE}.binding"]
    # The rule is the gate's own, not the package's: the readout module is built on torch, and the
    # binding is what the runner reads the model out of.
    assert forbidden_imports("import torch\n", f"{PACKAGE}.metrics") == []
    assert forbidden_imports(reach, f"{PACKAGE}.run") == []


def test_a_shared_eval_module_outside_the_allow_list_is_reported() -> None:
    """``runner`` builds the shared package's own network; ``metrics`` assumes another target."""
    assert forbidden_imports(
        "from teb_vae.lag_attn.eval import runner\n", f"{PACKAGE}._reuse"
    ) == [f"{SIBLING_EVAL}.runner"]
    lazy = "def f():\n    from teb_vae.lag_attn.eval.analyses import probe\n"
    assert forbidden_imports(lazy, f"{PACKAGE}.probe") == [f"{SIBLING_EVAL}.analyses"]


def test_the_allowed_sibling_modules_all_exist() -> None:
    """An allow-list entry naming a module that is not there is permission for nothing."""
    shared_root = Path(__file__).resolve().parents[2] / "lag_attn" / "eval"
    missing = [
        name for name in sorted(ALLOWED_SIBLING_EVAL_MODULES)
        if not (shared_root / f"{name}.py").is_file()
    ]
    assert missing == []


def test_importing_the_shared_eval_package_itself_reaches_nothing() -> None:
    """Its ``__init__`` is a docstring, so the bare package name is not a reach into anything."""
    source = "from teb_vae.lag_attn.eval import labels, stats\n"
    assert forbidden_imports(source, f"{PACKAGE}._reuse") == []
