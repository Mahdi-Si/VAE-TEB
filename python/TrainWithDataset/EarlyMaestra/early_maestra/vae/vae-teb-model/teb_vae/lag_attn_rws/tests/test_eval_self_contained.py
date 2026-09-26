r"""What each layer of the evaluation package is allowed to import, enforced by an AST walk.

The package is layered, and the layering is the thing that keeps an analysis from quietly
becoming a second training loop:

* **Layer 0 and layer 2** -- the pure modules and ``analyses/*`` -- import no Lightning, no
  ``model.*``, and none of this module's ``task``/``trainer``/``plotting``. An analysis receives
  tables and writes files; it does not rebuild a model.
* **Layer 1 and layer 3** -- the model-touching modules and the runner -- may import ``task`` and
  ``trainer``, but only where :data:`EXEMPTIONS` says so and only those. That exemption is
  deliberate rather than reluctant: ``metrics`` assembles the model's inputs through the task's
  own builders precisely so an evaluation cannot feed the model a differently assembled stream
  than training did, and re-implementing those builders to win a layering rule would reintroduce
  the exact drift they exist to prevent.
* **No analysis imports another analysis.** Anything two of them share moves one layer down.
* **The sibling's evaluation package is reachable only through its model-free modules**
  (:data:`ALLOWED_SIBLING_EVAL_MODULES`). Its ``runner``, ``metrics`` and ``analyses`` assume a
  feature-space target with a channel axis this model does not have, and its ``runner``
  constructs the sibling's own network.
* **``model/*`` is forbidden everywhere, at every layer.** It is the tree this one supersedes.

The walk is what makes the rule real, and it resolves more than a name-based scan can. A lazy
``import model.x`` inside the one function that needs it is exactly the shortcut a future change
reaches for, precisely because it looks less like a dependency; a relative ``from ..trainer
import x`` and a relative ``from . import <sibling>`` never mention a forbidden string at all.
All three are resolved to absolute dotted names before any rule is applied.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Tuple

import pytest

#: The package under scrutiny, and its directory.
PACKAGE = "teb_vae.lag_attn_rws.eval"
EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"

#: Import prefixes no module under ``eval/`` may reach for, whatever its layer.
FORBIDDEN_PREFIXES: Tuple[str, ...] = (
    "model",
    "lightning",
    "pytorch_lightning",
    "teb_vae.lag_attn_rws.task",
    "teb_vae.lag_attn_rws.trainer",
    "teb_vae.lag_attn_rws.plotting",
)

#: The sibling evaluation package, and the modules within it this one may bind. Every entry is
#: model-free in the sense that matters: none of them constructs, loads or forwards a network.
#:
#: ``band_partition`` joined the list when the input channel map landed: it reads a shard's own
#: ``sel_*`` attributes and imports nothing but ``numpy``, so the arithmetic it owns -- the
#: harmonic grid, the descending filter bank, the clinical band edges -- describes the dataset
#: pipeline rather than either model, and it takes the phase block it partitions as an argument.
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
MODEL_TOUCHING: FrozenSet[str] = frozenset({"metrics", "collect", "preflight", "oracle", "run"})

#: Modules that must stay importable with **no** numeric stack installed, so ``torch`` joins
#: their forbidden list on top of the layer rules. The acceptance gate reads a finished run's
#: ``summary.json`` on whatever box the file was copied to, and proving the property here -- on
#: the import graph -- replaces an awkward uninstalled-torch harness that could only ever prove
#: it for one environment.
NO_TORCH_MODULES: FrozenSet[str] = frozenset({f"{PACKAGE}.verify"})

#: ``module stem -> the forbidden prefixes it may import anyway``. Narrow by construction: named
#: modules, named targets, one stated reason each.
#:
#: ``run`` rebuilds the task from the checkpoint's own ``model_kwargs`` and reads ``trainer``'s
#: resolved-config filename, so that an evaluation reconstructs what was trained rather than what
#: a config file currently says. ``preflight`` reuses the four guards the training entry point
#: already owns -- the statistics path, the declared channel widths, the normalized raw target
#: and the reach-budget resolution -- rather than copying ninety lines whose long, actionable
#: messages would then have two places to drift apart in.
#:
#: The table is asserted **minimal**: a module listing a name it does not import is a permission
#: that outlived its use, and the next reach for that name would go unreported.
EXEMPTIONS: Dict[str, Set[str]] = {
    "run": {"teb_vae.lag_attn_rws.task", "teb_vae.lag_attn_rws.trainer"},
    "preflight": {"teb_vae.lag_attn_rws.trainer"},
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
    """Return the sibling-eval submodule ``name`` reaches into, if any.

    Args:
        name: An absolute dotted import name.

    Returns:
        The first component below ``teb_vae.lag_attn.eval``, or ``None`` when the name is not
        under it. Importing the sibling eval *package itself* yields ``None``: its ``__init__``
        is a docstring, so the package name alone reaches nothing.
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
        ``from ..trainer import X`` is reported once as ``teb_vae.lag_attn_rws.trainer`` rather
        than twice.
    """
    stem = module_name.rsplit(".", 1)[-1]
    under_analyses = module_name.startswith(f"{PACKAGE}.analyses.")
    # An exemption is a property of a layer-1/3 module. A stem collision with an analysis must
    # not hand that analysis the same permission.
    allowed = EXEMPTIONS.get(stem, set()) if stem in MODEL_TOUCHING and not under_analyses else set()

    violations: List[str] = []
    for name in imported_names(source, module_name):
        if module_name in NO_TORCH_MODULES and _matches(name, "torch"):
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
    return [name for name in unique if not any(_matches(name, other) for other in unique if other != name)]


# =============================================================================
# The shipped package
# =============================================================================
def _shipped_modules() -> List[Path]:
    """Every ``.py`` that ships under ``eval/``. The tests live one directory up, not here."""
    return sorted(EVAL_ROOT.rglob("*.py"))


def test_the_model_binding_is_walked_and_sits_at_layer_zero() -> None:
    """The walk is directory-driven, so this is what says the file was actually picked up rather
    than that a rule happened to hold over a set it was not in.

    ``binding`` holds the facts the pipeline cannot derive about the model it is evaluating, and
    it stays at layer 0 precisely so that naming those facts costs no ``torch`` import: the
    acceptance gate and the documentation tests read the type without a numeric stack. It carries
    no exemption, because a module that named a model class would not be at layer 0 at all.
    """
    stems = {module.stem for module in _shipped_modules()}
    assert "binding" in stems

    binding = next(module for module in _shipped_modules() if module.stem == "binding")
    names = imported_names(binding.read_text(encoding="utf-8"), _module_name_for(binding))

    assert forbidden_imports(binding.read_text(encoding="utf-8"), _module_name_for(binding)) == []
    # Stdlib only, so importing the seam costs nothing: no torch, no numpy, no matplotlib, and
    # nothing from this repository -- which is what keeps a concrete binding out of this module.
    assert all(not name.startswith(("torch", "numpy", "matplotlib", "teb_vae")) for name in names)


@pytest.mark.parametrize(
    "module", _shipped_modules(), ids=lambda path: str(path.relative_to(EVAL_ROOT))
)
def test_every_shipped_module_stays_inside_its_layer(module: Path) -> None:
    """Walked per module so a failure names the file rather than the package."""
    module_name = _module_name_for(module)
    violations = forbidden_imports(module.read_text(encoding="utf-8"), module_name)
    assert not violations, (
        f"{module.relative_to(EVAL_ROOT)} imports {violations}. Layers 0 and 2 take no Lightning, "
        f"no model/ and no task/trainer/plotting; layers 1 and 3 take task and trainer only "
        f"through EXEMPTIONS; the sibling evaluation package is reachable only through "
        f"{sorted(ALLOWED_SIBLING_EVAL_MODULES)}. If the import is genuinely necessary, add it to "
        f"EXEMPTIONS with a reason."
    )


def test_the_reuse_seam_is_the_only_module_naming_the_sibling_evaluation_package() -> None:
    """The seam exists so the coupling is visible in one file; a second reach would hide it.

    ``config_schema`` is the stated exception: it takes the sibling's two validators directly so
    that validating a run's settings costs a stdlib parse rather than a matplotlib import.
    """
    reaching = []
    for module in _shipped_modules():
        names = imported_names(module.read_text(encoding="utf-8"), _module_name_for(module))
        if any(_sibling_eval_module(name) is not None for name in names):
            reaching.append(module.stem)
    assert sorted(reaching) == ["_reuse", "config_schema"]


# =============================================================================
# Non-vacuity: the shapes a name-based check would miss or wave through
# =============================================================================
_V = PACKAGE


@pytest.mark.parametrize(
    "source, module_name, expected",
    [
        # An aliased import still names its module.
        ("import lightning.pytorch as pl\n", f"{_V}.verify", ["lightning.pytorch"]),
        # A lazy in-function import: the likely shape of a change that needs "just one thing".
        (
            "def analyse():\n    from model.lstm_cnn_vae_teb.testing import metrics\n",
            f"{_V}.analyses.forecast",
            ["model.lstm_cnn_vae_teb.testing"],
        ),
        # A relative parent import names no forbidden string; it has to be resolved first.
        (
            "from ..trainer import RESOLVED_CONFIG_FILENAME\n",
            f"{_V}.verify",
            ["teb_vae.lag_attn_rws.trainer"],
        ),
        # Analyses never import one another.
        ("from . import forecast\n", f"{_V}.analyses.coupling", [f"{_V}.analyses.forecast"]),
        # An exemption permits only its named targets.
        (
            "from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask\n"
            "from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME\n"
            "import lightning as L\n",
            f"{_V}.run",
            ["lightning"],
        ),
        # A stem collision does not hand a layer-two module the runner's permission.
        (
            "from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask\n",
            f"{_V}.analyses.run",
            ["teb_vae.lag_attn_rws.task"],
        ),
        # Preflight reuses the trainer's guards and builds nothing, so it takes no task.
        (
            "from teb_vae.lag_attn_rws.trainer import _check_stat_path\n"
            "from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask\n",
            f"{_V}.preflight",
            ["teb_vae.lag_attn_rws.task"],
        ),
        # The acceptance gate must stay checkable on a box with no numeric stack.
        ("import torch\n", f"{_V}.verify", ["torch"]),
        # The sibling's runner builds the sibling's own network.
        ("from teb_vae.lag_attn.eval import runner\n", f"{_V}._reuse", [f"{SIBLING_EVAL}.runner"]),
    ],
)
def test_the_walker_reports_each_forbidden_shape(source, module_name, expected) -> None:
    """The walker is what makes the per-module rule real, so it must see each shape."""
    assert forbidden_imports(source, module_name) == expected


# =============================================================================
# The exemption table is minimal
# =============================================================================
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
# The page builder at the package root
#
# The evaluation's per-sample pages and the training callback draw the same figure, and they sit
# on opposite sides of this layering: the callback is a Lightning callback, and `analyses/*` may
# import neither Lightning nor `plotting`. The builder therefore lives at the package root, where
# both reach it and neither reaches the other -- so the rule that keeps it there is that the
# module itself stays framework-free.
# =============================================================================
#: The relocated builder, and what it may not reach for.
SAMPLE_PAGE = Path(__file__).resolve().parents[1] / "sample_page.py"
SAMPLE_PAGE_FORBIDDEN: Tuple[str, ...] = (
    "lightning",
    "pytorch_lightning",
    "model",
    "utils.mlflow_utils",
    "teb_vae.lag_attn_rws.plotting",
    "teb_vae.lag_attn_rws.task",
    "teb_vae.lag_attn_rws.trainer",
    PACKAGE,
)


def test_the_page_builder_is_framework_free() -> None:
    """A page builder importing Lightning would put it back in the evaluation's import graph,
    which is the whole reason it moved out of ``plotting.py``. Importing the eval package would
    invert the dependency the other way, and make the *training* path import the evaluation."""
    names = imported_names(SAMPLE_PAGE.read_text(encoding="utf-8"), "teb_vae.lag_attn_rws.sample_page")

    offending = [
        name for name in names
        if any(_matches(name, prefix) for prefix in SAMPLE_PAGE_FORBIDDEN)
    ]
    assert offending == []


