r"""Two standing rules, enforced mechanically because both are invisible at review time.

**The stored source timeline is canonical.** The dataset builder shifts the uterine-activity channel
when it writes the shards, and the stored signals are treated as if they were recorded that way. Any
term that adds that shift back, subtracts it, budgets it, audits it or simulates it -- under any of
its names -- is a bug rather than a parameterisation, and one whose effect is a plausible number on
a lag axis rather than a failure.

**Geometry belongs in the configuration, not in prose.** A horizon, an anchor count, a block width
or a channel count written as a literal in a comment or a report string is correct until the
configuration moves, and then it is a confident false statement that no test and no reader catches.
The configuration files are where those numbers live; everything else reads them from the model.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

import pytest

#: The package this file guards.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Every term that would reintroduce a downstream correction for the dataset's own source shift.
#: Matched case-insensitively across the whole package, source and configuration alike.
FORBIDDEN_TIMELINE_TERMS: Tuple[str, ...] = (
    "up_shift_secs",
    "tau_pre",
    "mechanical shift",
    "mechanical delay",
    "sensor delay",
    "sensor timeline",
    "acquisition shift",
    "acquisition delay",
)

#: Files whose *comments* may carry geometry literals, because they are where the geometry is
#: declared. A configuration that could not state its own horizon would state nothing.
GEOMETRY_LITERAL_ALLOWLIST: Tuple[str, ...] = (
    "configs/default.yaml",
    "configs/tiny.yaml",
    # The evaluation override delta, on the same ground as the two above: it is a configuration
    # file, and the lag bands it declares ARE the geometry of the readout. Their comment restates
    # each band's own edges beside it, which is what a reader of a partition needs and what the
    # rule exists to keep out of code.
    "eval/configs/eval_overrides.yaml",
)

#: Numbers that are geometry when they appear beside a geometry word. Small integers and ratios are
#: excluded: a kernel width of three and a probability of one half are not task geometry, and
#: forbidding them would make the rule unusable rather than strict.
GEOMETRY_PATTERN = re.compile(
    r"\b(?:horizon|anchors?|block width|channels?|lags?|stride)\b[^.\n]{0,40}?\b(\d{2,})\b",
    re.IGNORECASE,
)


def package_files(suffix: str) -> List[Path]:
    """Every file of one kind in the package, tests included, except this one.

    This module is excluded from its own scans, and the exclusion is not a convenience: the
    forbidden terms and the document-reference pattern are written out here as constants, so a
    gate that scanned itself would report itself and could never pass. Every other file in the
    package is scanned, including the other tests.

    Args:
        suffix: The extension to collect, with its dot.

    Returns:
        The paths, sorted.
    """
    own = Path(__file__).resolve()
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts and path.resolve() != own
    )


def test_no_source_timeline_correction_appears_anywhere_in_the_package() -> None:
    """Source, configuration and test alike.

    The rule is not that such a term is discouraged: it is that the stored timeline is the signal,
    so a downstream correction describes a recording that does not exist. Removing the term is the
    fix, never parameterising it.
    """
    offences: List[str] = []
    for path in package_files(".py") + package_files(".yaml"):
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_TIMELINE_TERMS:
            if term in text:
                offences.append(f"{path.relative_to(PACKAGE_ROOT)}: {term!r}")
    assert offences == [], offences


def test_no_module_references_a_markdown_document() -> None:
    """A design note moves, is superseded, or is renamed, and the reference outlives it.

    What a module needs to say about why it exists belongs in the module. A pointer to a document
    is a claim that ages without anything failing.
    """
    offences = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in package_files(".py")
        if re.search(r"\.md\b", path.read_text(encoding="utf-8"))
    ]
    assert offences == []


def test_no_comment_or_docstring_hard_codes_the_task_geometry() -> None:
    """Outside the configuration files, which are where the geometry is declared.

    A horizon or a channel count written into prose is correct until the configuration moves. The
    pattern is deliberately narrow -- a two-digit-or-longer number within a few words of a geometry
    term -- so that kernel widths, dilations and probabilities are not swept up with it.
    """
    offences: List[str] = []
    for path in package_files(".py") + package_files(".yaml"):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if relative in GEOMETRY_LITERAL_ALLOWLIST:
            continue
        for number, line in _prose_lines(path):
            match = GEOMETRY_PATTERN.search(line)
            if match:
                offences.append(f"{relative}:{number}: {line.strip()}")
    assert offences == [], offences


def _prose_lines(path: Path) -> List[Tuple[int, str]]:
    """The comment and docstring lines of one file, with their line numbers.

    Executable code is excluded on purpose: a literal in an expression is a value the program uses
    and a test can check, while a literal in prose is a claim nothing verifies.

    Args:
        path: The file to read.

    Returns:
        ``(line number, text)`` per prose line.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if path.suffix == ".yaml":
        return [
            (number, line)
            for number, line in enumerate(lines, start=1)
            if line.lstrip().startswith("#")
        ]

    prose: List[Tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            prose.append((number, line))

    # Docstrings, taken from the parsed tree rather than by pattern, so a string that merely looks
    # like one is not swept up.
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        start = 1 if isinstance(node, ast.Module) else node.body[0].lineno
        prose.extend(
            (start + offset, doc_line) for offset, doc_line in enumerate(doc.splitlines())
        )
    return prose


@pytest.mark.parametrize("path", package_files(".py"), ids=lambda p: p.name)
def test_every_module_carries_a_docstring(path: Path) -> None:
    """Including the test modules, which is where the reason for a check is recorded.

    Args:
        path: The module to check.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{path.name} has no module docstring"


def test_every_public_function_and_class_carries_a_docstring() -> None:
    """A public name with no docstring is a contract with no statement of what it is."""
    offences: List[str] = []
    for path in package_files(".py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_") and not node.name.startswith("__"):
                continue
            if ast.get_docstring(node) is None:
                offences.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: {node.name}")
    assert offences == []


def test_the_page_is_wired_by_the_shared_key_and_this_packages_own_callback() -> None:
    """The one seam whose failure is a missing figure rather than an error.

    The block name is the shared driver's literal because the callback assembly reads that key: a
    name matching this package would get no figure, no error and nothing in the log. The callback
    class is this package's because the family's runs a forward without the per-lag proposals and
    hands the result to a builder that reads two tensors this architecture does not produce.

    Both halves are asserted together because getting either one right on its own still leaves a
    run with no page.
    """
    from teb_vae.lag_slot_transformer_cfs.plotting import LagResidualTrfCfsPlotCallback
    from teb_vae.lag_slot_transformer_cfs.trainer import LagResidualTrfCfsTrainer

    assert LagResidualTrfCfsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert LagResidualTrfCfsTrainer.plot_callback_cls() is LagResidualTrfCfsPlotCallback

    trainer_source = (PACKAGE_ROOT / "trainer.py").read_text(encoding="utf-8")
    assert "`PLOT_CONFIG_KEY`` deliberately stays the shared driver's literal" in trainer_source
