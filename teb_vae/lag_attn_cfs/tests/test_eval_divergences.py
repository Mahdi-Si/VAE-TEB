r"""The divergence manifest: the fork's third anti-drift measure, and the one with teeth.

This package is a copy of ``teb_vae/lag_attn_rws/eval``, and a copy needs a register of what was
changed and why. Prose is not a control: the failure it has to catch is an **omission**, and nobody
notices a paragraph that was never written. So the register is committed data --
``divergences.json`` beside the code -- with one entry per module of the forked-from package,
classified into exactly three states, and this file is what makes each classification cost
something:

``equivalent``
    Must stay behaviour-equivalent. Once the module exists here, its entry has to **name** the
    assertions that exercise it, and those assertions have to exist. An "equivalent" claim nothing
    checks is the weakest thing in the file, and it is also the easiest one to leave behind.

``divergent``
    Deliberately differs, and carries a non-empty reason. The reason is what the evaluation record
    renders, so an empty one is a divergence a reader is never told about.

``absent``
    Not ported at all, and the file is asserted **not** to exist. That is what makes a later copy
    of it fail here rather than quietly reintroducing a readout this target domain cannot support.

**Both directions are checked**, because the drift this exists for runs both ways: a module of this
package that nobody classified, and a module the sibling *gained* that nobody noticed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Set, Tuple

import pytest

#: The manifest, and the two package directories it relates.
MANIFEST = Path(__file__).resolve().parents[1] / "eval" / "divergences.json"
EVAL_ROOT = Path(__file__).resolve().parents[1] / "eval"
FORKED_FROM_ROOT = Path(__file__).resolve().parents[2] / "lag_attn_rws" / "eval"

#: The three states, written out rather than read from the file: the manifest declaring its own
#: valid values would make any new state valid by declaring it.
STATES = frozenset({"equivalent", "divergent", "absent"})

#: Modules this package has that the forked-from one does not, and which therefore carry no entry.
#: Written out so that adding one is a decision recorded here rather than a silent gap in the
#: register -- the manifest is keyed by the *sibling's* modules by construction, so a cell-specific
#: addition can only be tracked from this side.
#:
#: The analyses this target domain has and the raw one cannot: where in the warm-up staircase
#: the forecast gap lives and whether the run decoded the population its configuration describes
#: (``warmup``); how much of the coupling readout survives zeroing the source, which is the
#: availability-clock hazard no permutation control can see (``source_null``); what the forecast
#: loses when the source's own values are removed from a band of the lag window, which is the
#: interventional half of the same question (``occlusion``); and the forecast resolved by the
#: frequency band of the target coefficient, which is what this domain's channel axis can answer in
#: place of the sibling's phase-domain pair (``spectral_skill``).
CELL_SPECIFIC_MODULES: Tuple[str, ...] = (
    "analyses/warmup.py",
    "analyses/source_null.py",
    # Cell-specific rather than divergent for a reason worth separating from the others: the
    # sibling has a lag window too, but its source stream is present from the first step, so a
    # band of it removed is not a band of a *delayed* channel set and the readout would answer a
    # different question there. The intervention is the causal cells' alone.
    "analyses/occlusion.py",
    "analyses/spectral_skill.py",
    "analyses/lag_clocks.py",
    # The lag structure of the anchors selected by their own pooled-KL quantile band. Cell
    # specific for lag_clocks' reason, and because it reads the per-anchor vector sidecar this
    # cell's collection pass writes and the sibling's does not.
    "analyses/lag_high_kl.py",
    # The KLD-scaled, band-restricted and per-head reading of the same lag structure. Cell
    # specific for lag_clocks' reason and one of its own: the sibling has no availability
    # clock, so it has no clock-excess profile to weight or to select on.
    "analyses/lag_kld_scaled.py",
    # The profile-shape reducer the clocks analysis is built on. Cell-specific rather than
    # divergent: the sibling has no counterpart to classify it against, because the one-sided
    # bank's compensated lag axis is this cell's alone.
    "lag_shape.py",
)

#: Where this suite's tests live, for resolving the named assertions.
TESTS_ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def manifest() -> Dict:
    """The parsed manifest."""
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(manifest) -> Dict[str, Dict]:
    """``module path -> entry``."""
    return manifest["modules"]


def _sibling_modules() -> Set[str]:
    """Every ``.py`` of the forked-from package, as a path relative to its ``eval/``."""
    return {
        path.relative_to(FORKED_FROM_ROOT).as_posix()
        for path in FORKED_FROM_ROOT.rglob("*.py")
    }


def _own_modules() -> Set[str]:
    """Every ``.py`` of this package, as a path relative to its ``eval/``."""
    return {path.relative_to(EVAL_ROOT).as_posix() for path in EVAL_ROOT.rglob("*.py")}


def _test_functions(filename: str) -> Set[str]:
    """The test function names defined in one file of this suite.

    Args:
        filename: A file name inside this suite's directory.

    Returns:
        The names, or an empty set when the file does not exist.
    """
    path = TESTS_ROOT / filename
    if not path.is_file():
        return set()
    return {
        node.name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# =================================================================================================
# The register covers the package it is a register of
# =================================================================================================
def test_every_module_of_the_forked_from_package_has_exactly_one_entry(entries) -> None:
    """The omission this file exists for. A module with no entry is a module nobody decided about,
    and it is invisible in every other artifact."""
    missing = sorted(_sibling_modules() - set(entries))

    assert missing == [], (
        f"{missing} are modules of the forked-from package with no entry. Classify each as "
        f"equivalent, divergent or absent -- a module nobody decided about is the case a prose "
        f"register cannot catch."
    )


def test_no_entry_names_a_module_the_forked_from_package_does_not_have(entries) -> None:
    """The other direction, and the one that catches the sibling *deleting* a module: an entry for
    a file that is not there any more classifies nothing and reads as though it did."""
    unknown = sorted(set(entries) - _sibling_modules())

    assert unknown == [], f"{unknown} name no module of the forked-from package"


def test_the_register_is_not_empty_or_truncated(entries) -> None:
    """A manifest reduced to a handful of entries would satisfy every rule above vacuously."""
    assert len(entries) == len(_sibling_modules())
    assert len(entries) >= 30


def test_every_module_of_this_package_is_either_classified_or_declared_cell_specific() -> None:
    """The third direction, which the manifest's own key set cannot express: a module *this*
    package has and the sibling does not carries no entry by construction, so it has to be
    declared here instead of silently escaping the register."""
    unclassified = sorted(_own_modules() - _sibling_modules() - set(CELL_SPECIFIC_MODULES))

    assert unclassified == [], (
        f"{unclassified} exist in this package, have no counterpart to be classified against, and "
        f"are not declared cell-specific"
    )
    # And the declaration does not outlive its module.
    assert all((EVAL_ROOT / name).is_file() for name in CELL_SPECIFIC_MODULES)


# =================================================================================================
# What each state costs
# =================================================================================================
def test_every_entry_carries_one_of_the_three_states(entries) -> None:
    wrong = sorted(
        f"{name}: {entry.get('state')!r}" for name, entry in entries.items()
        if entry.get("state") not in STATES
    )

    assert wrong == [], f"entries with no valid state: {wrong}"


def test_a_divergent_entry_carries_the_reason_the_record_renders(entries) -> None:
    """An empty reason is a divergence a reader is never told about, and the register is what the
    evaluation record's divergence section is built from."""
    silent = sorted(
        name for name, entry in entries.items()
        if entry["state"] == "divergent" and not str(entry.get("reason", "")).strip()
    )

    assert silent == [], f"divergent entries with no reason: {silent}"
    # Long enough to be a reason rather than a label. The shortest one here is a sentence.
    assert all(
        len(entry["reason"]) > 40
        for entry in entries.values()
        if entry["state"] == "divergent"
    )


def test_an_equivalent_entry_carries_no_reason(entries) -> None:
    """The two states are distinguished by whether anything changed, so a reason on an
    ``equivalent`` entry means one of the two fields is wrong and a reader cannot tell which."""
    contradictory = sorted(
        name for name, entry in entries.items()
        if entry["state"] == "equivalent" and str(entry.get("reason", "")).strip()
    )

    assert contradictory == []


def test_an_absent_entrys_file_is_absent(entries) -> None:
    """The check that makes ``absent`` a decision rather than a note: a later copy of the module
    fails here rather than quietly reintroducing a readout this target domain cannot support."""
    present = sorted(
        name for name, entry in entries.items()
        if entry["state"] == "absent" and (EVAL_ROOT / name).is_file()
    )

    assert present == [], f"{present} are classified absent but exist in this package"


def test_the_two_modules_that_are_not_ported_at_all_are_the_expected_ones(entries) -> None:
    """Named rather than counted: a third module quietly reclassified as absent would be a readout
    dropped from the pipeline with nothing but a state change to show for it."""
    absent = sorted(name for name, entry in entries.items() if entry["state"] == "absent")

    assert absent == ["analyses/coherence.py", "spectra.py"]


def test_an_equivalent_entry_names_assertions_that_exist(entries) -> None:
    """The rule that makes an ``equivalent`` claim cost something. Checked only once the module is
    here: a classification is a target while the file is still to be written, and requiring an
    assertion against a module that does not exist would only invite a placeholder.

    Both halves are checked -- the file and the function -- because a renamed test is exactly how
    the claim goes quiet while the entry still reads as though something checked it.
    """
    problems = []
    for name, entry in entries.items():
        if entry["state"] != "equivalent" or not (EVAL_ROOT / name).is_file():
            continue
        named = list(entry.get("exercised_by") or [])
        if not named:
            problems.append(f"{name}: exists here and names no assertion")
            continue
        for reference in named:
            filename, _, function = reference.partition("::")
            if not function:
                problems.append(f"{name}: {reference!r} is not '<file>::<test>'")
            elif function not in _test_functions(filename):
                problems.append(f"{name}: {reference} does not exist")

    assert problems == [], problems


def test_the_modules_that_have_landed_are_classified_against_what_is_on_disk(entries) -> None:
    """Non-vacuity for the rule above: it is silent about modules not yet ported, so this is what
    says the check is currently exercising something rather than skipping everything."""
    landed = {name for name in entries if (EVAL_ROOT / name).is_file()}
    equivalent_and_landed = {
        name for name in landed if entries[name]["state"] == "equivalent"
    }

    assert len(landed) >= 10
    assert len(equivalent_and_landed) >= 4


# =================================================================================================
# It reads without this document
# =================================================================================================
def test_the_manifest_states_what_its_three_states_mean(manifest) -> None:
    """It is committed data that outlives the conversation that produced it, so it has to be
    readable on its own -- including by whoever renders the evaluation record from it."""
    comment = manifest["_comment"]

    assert STATES <= set(comment)
    assert all(len(str(comment[state])) > 40 for state in STATES)
    assert "what_this_is" in comment


def test_the_manifest_is_the_only_place_the_register_is_kept() -> None:
    """A second, hand-kept copy is what the register exists to replace: the evaluation record's
    section is *rendered* from this file, so a module classified twice could be classified twice
    differently and only one of the two would be checked.

    Checked by **shape** rather than by counting the directory's JSON files. The eval package also
    commits ``figure_manifest.json``, which is a different kind of record entirely -- what a run
    emits, kept equal to a real run by the smoke suite -- and a rule that forbade a second JSON
    would forbid it for no reason. What must not exist twice is a *classification of the sibling's
    modules*, and that is what is looked for.
    """
    assert MANIFEST.is_file()
    duplicates = []
    for path in EVAL_ROOT.glob("*.json"):
        if path.name == MANIFEST.name:
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        modules = blob.get("modules") if isinstance(blob, dict) else None
        if isinstance(modules, dict) and any(
            isinstance(entry, dict) and "state" in entry for entry in modules.values()
        ):
            duplicates.append(path.name)

    assert duplicates == [], (
        f"{duplicates} also classifies the sibling's modules; the register is one file, and "
        f"EVAL.md's section is rendered from it rather than kept beside it"
    )
