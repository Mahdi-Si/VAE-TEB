r"""The fixtures this package builds on, the splice they arrive through, and the sibling it may not edit.

Nothing here is committed: the causal shard and its statistics belong to ``lag_attn``, because every
model in the family reads the same shards through the same loader, and a second copy would be a
second dataset that could come to disagree.

The splice is the subject of the rest of this file. The causal-input half of the fixture surface is
the causal-feature cell's, imported by name; the constructor keyword sets are local. Two things have
to hold for that to be sound, and neither is visible from either side alone:

* every imported name must still be exported by that suite -- an ``ImportError`` in whichever test
  happened to collect first names a symbol rather than the splice, and the obvious fix for it is an
  edit to a package this one may not touch;
* nothing under that package may change **for this package**. It is reached by reference precisely
  so that it does not have to, and the only registered exception is the family's hand-kept import
  guard, which builds forbidden dotted-prefix strings and imports nothing. Its own modules do move
  -- they are the family's causal propagation channel, and a change to the source pathway or the
  objective is meant to land there and reach every causal cell at once -- so what is checked is
  whether a changed module names *this* package, and, on a clean tree where that scan sees nothing,
  that no module there imports back.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List

import pytest

from teb_vae.lag_attn_cfs.tests import conftest as causal_conftest

from . import conftest as local

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: The sibling this package reaches into, and the only file under it that may name this package
#: back. The family's import guard must name every package a net could reach into, so a new package
#: has to be registered in it; the constant it is registered in builds forbidden dotted-prefix
#: **strings** and imports nothing, which is what makes registering a name long before the package
#: exists inert rather than merely harmless.
#:
#: **What the guard asks has been narrowed, and the narrowing is a correction rather than a
#: relaxation.** It used to ask that nothing under the sibling move at all, which was true of this
#: package's own arrival and is not a property of the sibling: the causal mixin there is the
#: propagation channel every causal cell in the family is built through, so a family-wide change to
#: the source pathway, the objective or the alignment lands in those files *by design* and reaches
#: this package through them. Stillness was never the invariant. The invariant is the **direction**
#: -- a member needed from that package is bound by reference or imported by name, and nothing
#: there moves *for* this package -- and what a sibling edit made for this package would carry, in
#: an import or a branch or a comment, is this package's own name.
_SIBLING = "teb_vae/lag_attn_cfs"
_SIBLING_ALLOWED_CHANGES = ("teb_vae/lag_attn_cfs/tests/test_nets_are_framework_free.py",)

#: The token a sibling module edited to accommodate this package would carry. Searched in source
#: only: the sibling's own records cite the whole grid by name, as records should, and a guard that
#: read a cross-reference in a design document as a code dependency would be turned off rather than
#: fixed.
_THIS_PACKAGE = "lag_attn_crws"

#: The production geometry this package declares, as literals. Every one of them is a *choice*
#: rather than a constraint -- a raw sample is honest at every step, so nothing about the target ties
#: the floor to the budget -- and holding them at the causal-feature cell's values is what leaves
#: exactly one variable between the two: what the decoder emits. A silent change to any of them
#: moves what every number this package reports is produced at.
_SHIPPED_GEOMETRY = (
    ("horizon", 30),
    ("warmup_period", 134),
    ("anchor_stride", 30),
    ("c_y", 102),
    ("c_u", 51),
)


#: Porcelain status letters that mean a **tracked** file moved: added, modified, deleted, renamed,
#: copied, unmerged. Staged and unstaged columns are both examined, so a change that is only in the
#: index counts -- which matters here, because this repository's workflow stages without committing
#: and an ``A`` entry is otherwise indistinguishable from a clean tree.
_TRACKED_CHANGE_LETTERS = "AMDRCU"


def _is_sibling_edit(code: str, path: str) -> bool:
    """Whether one ``git status --porcelain`` entry is an edit to a package reached by reference.

    A tracked file that moved counts, and so does a new ``.py`` file: both are edits to a package
    this design is supposed to leave alone. An untracked non-source file does not -- the tree
    already carries planning documents and crash dumps under every sibling, and a guard that failed
    on those would be turned off rather than fixed.

    Args:
        code: The two-character status field.
        path: The entry's path, already normalised to forward slashes.

    Returns:
        ``True`` if the entry must be accounted for by the allow-list.
    """
    if code == "??":
        return path.endswith(".py")
    return any(letter in _TRACKED_CHANGE_LETTERS for letter in code)


def _names_this_package(path: str) -> bool:
    """Whether a changed sibling module carries this package's name.

    A file that no longer exists counts, and deliberately: a deleted sibling module cannot be
    cleared by reading it, and the safe reading of an unreadable change is that it has to be looked
    at rather than that it is fine.

    Args:
        path: A repo-root-relative path with forward slashes.

    Returns:
        ``True`` if the file names this package, or could not be read to find out.
    """
    source = _REPO_ROOT / path
    if not source.is_file():
        return True
    return _THIS_PACKAGE in source.read_text(encoding="utf-8", errors="ignore")


def _sibling_changes(package: str) -> List[str]:
    """Return the paths under ``package`` that this working tree changes.

    Args:
        package: A repo-root-relative directory to scope ``git status`` to.

    Returns:
        Repo-root-relative paths with forward slashes, one per offending entry.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", package],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    changed: List[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        # A rename reports ``old -> new``; the new path is the one that exists to be judged.
        path = path.split(" -> ")[-1].strip().strip('"').replace("\\", "/")
        if _is_sibling_edit(code, path):
            changed.append(path)
    return changed


# =================================================================================================
# The committed fixtures
# =================================================================================================
def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble derives the repo root from this file's own depth; a wrong depth
    would resolve some unrelated directory without ever raising."""
    for package in ("teb_vae", "train", "utils"):
        assert (_REPO_ROOT / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {_REPO_ROOT}"
        )


def test_no_fixture_files_live_in_this_package():
    """The committed shard and stats are ``lag_attn``'s; this package references them by path.

    The two-sided file beside them is what makes "a causal shard is required" a comparison rather
    than an assertion about one file, and the statistics file matters as much as the shard: the
    dataset reader silently disables normalization on a stats-schema mismatch, so a missing one is
    every shape right and every number wrong.
    """
    assert not (_PACKAGE_DIR / "tests" / "fixtures").exists()
    shared = _REPO_ROOT / "teb_vae" / "lag_attn" / "tests" / "fixtures"

    assert (shared / "tiny_shard_causal.hdf5").is_file()
    assert (shared / "tiny_stats_causal.hdf5").is_file()
    assert local.CAUSAL_SHARD.is_file() and local.TWO_SIDED_SHARD.is_file()


# =================================================================================================
# The splice
# =================================================================================================
def test_the_imported_name_list_is_non_empty_and_every_name_is_still_exported():
    """A silently-empty list would make the identity check below vacuous, and would be the exact
    shape of the mistake: a name that stopped being shared without anyone noticing."""
    assert len(local.IMPORTED_FROM_CAUSAL) > 0

    missing = [name for name in local.IMPORTED_FROM_CAUSAL if not hasattr(causal_conftest, name)]
    assert missing == [], missing


@pytest.mark.parametrize("name", local.IMPORTED_FROM_CAUSAL)
def test_each_imported_name_is_the_causal_cells_own_object(name):
    """Identity rather than equality. These describe the committed shard and the boundary the
    resolver reads off it; a local copy would be free to describe a boundary the data no longer has,
    which is what reading the shards rather than declaring the vectors exists to prevent."""
    assert getattr(local, name) is getattr(causal_conftest, name)


def test_the_wrapped_config_delegates_to_the_siblings_own_builder():
    """The one imported name this package wraps rather than binds, and the property that keeps the
    wrap honest.

    ``causal_config`` is absent from :data:`IMPORTED_FROM_CAUSAL` because this row's alignment
    reference is $42.21$ s where the sibling's is ``target_max`` -- a raw target leaves the source
    reference uncancelled in the physical-lag identity, so the two rows cannot share a default.
    What must NOT happen is a second builder: the wrapper has to call the sibling's object, so a
    change to the shard, the widths or the schema still reaches this suite. Asserted by both
    halves -- the wrapper is a different object, and it produces the sibling's own output when
    handed the sibling's reference.
    """
    assert local.causal_config is not causal_conftest.causal_config

    mine = local.causal_config()
    theirs = causal_conftest.causal_config()
    vae = "model_config"

    # Differs in exactly one leaf: the reference.
    assert mine[vae]["VAE_model"]["causal_align_reference"] == local.SHIPPED_ALIGN_REFERENCE
    assert theirs[vae]["VAE_model"]["causal_align_reference"] == "target_max"

    # And in nothing else: handed the sibling's reference, the wrapper reproduces it exactly.
    assert local.causal_config(causal_align_reference="target_max") == theirs


def test_the_keyword_sets_are_local_objects_rather_than_the_siblings():
    """The other half of the splice. The geometry below is a choice this package makes -- nothing
    about a raw target forces it -- so an arm moving one leaf of it must not move the sibling's."""
    assert local.TINY_KWARGS is not causal_conftest.TINY_KWARGS
    assert local.SHIPPED_KWARGS is not causal_conftest.TINY_KWARGS


@pytest.mark.parametrize("key,value", _SHIPPED_GEOMETRY, ids=[key for key, _ in _SHIPPED_GEOMETRY])
def test_the_shipped_set_declares_the_production_geometry(key, value):
    """The configuration every number this package reports is produced at."""
    assert local.SHIPPED_KWARGS[key] == value


def test_the_shipped_set_leaves_the_decoder_width_to_the_architecture():
    r"""No ``decoder_out_channels``. The raw block is $R$ samples per horizon token, so the width
    follows ``raw_per_step`` and no configuration can put the decoder and the target on different
    widths -- which is the whole reason the feature-target mixin stays out of this cell's bases."""
    assert "decoder_out_channels" not in local.SHIPPED_KWARGS
    assert "decoder_out_channels" not in local.TINY_KWARGS
    assert local.SHIPPED_KWARGS["raw_per_step"] == local.TINY_KWARGS["raw_per_step"] == 16


# =================================================================================================
# The sibling is reached by reference, not edited
# =================================================================================================
def test_no_changed_module_under_the_causal_sibling_names_this_package():
    """The standing rule of this package's arrival, checked from the working tree.

    A member needed from that package is bound by reference in a class body or imported by name, so
    that nothing there has to move; the failure this catches is the tempting one, where a missing
    export or an inconvenient signature is "fixed" by editing a package six shipped cells score
    through.

    Read as *changed files that name this package* rather than as *changed files*. The sibling's
    modules do move -- they are the family's causal propagation channel, and a change to the source
    pathway or the objective is supposed to land there and reach every causal cell at once -- so
    their stillness was never the property worth guarding. What an edit made **for this package**
    carries is this package's name, in an import, a branch or a comment, and that is what is
    refused. One file is registered as an exception because it must name every package in the
    family: the import guard's forbidden-prefix list.
    """
    unexpected = [
        path
        for path in _sibling_changes(_SIBLING)
        if path.endswith(".py")
        and path not in _SIBLING_ALLOWED_CHANGES
        and _names_this_package(path)
    ]

    assert unexpected == [], unexpected


def test_the_causal_sibling_never_imports_this_package():
    """The other half of the same rule, and the half that does not need a dirty tree to be true.

    The scan above sees only what this working tree changed, so on a clean checkout it asserts
    nothing at all. This one is a property of the code as it stands: the dependency runs one way,
    from here into the sibling, and a sibling module importing back would make the two packages one
    -- a change on either side could then break the other, and the sibling is scored through by six
    shipped cells.

    ``tests/`` is excluded on the sibling's side for the reason the allow-list exists: its import
    guard has to enumerate every package in the family by name, and a test that reads its own
    subject as a violation is a test nobody keeps.
    """
    importers = [
        str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
        for path in sorted((_REPO_ROOT / _SIBLING).rglob("*.py"))
        if "tests" not in path.parts
        and re.search(rf"^\s*(from|import)\s+.*{_THIS_PACKAGE}", path.read_text(encoding="utf-8"),
                      re.MULTILINE)
    ]

    assert importers == [], importers


@pytest.mark.parametrize(
    "code,path,expected",
    [
        (" M", "teb_vae/lag_attn_cfs/task.py", True),
        ("M ", "teb_vae/lag_attn_cfs/task.py", True),
        ("A ", "teb_vae/lag_attn_cfs/nets/extra.py", True),
        ("D ", "teb_vae/lag_attn_cfs/nets/causal_inputs.py", True),
        ("R ", "teb_vae/lag_attn_cfs/nets/renamed.py", True),
        ("??", "teb_vae/lag_attn_cfs/nets/new_module.py", True),
        ("??", "teb_vae/lag_attn_cfs/DESIGN.md", False),
        ("??", "teb_vae/lag_attn_cfs/eval/bash.exe.stackdump", False),
    ],
)
def test_the_guard_classifies_an_entry_by_what_it_would_mean(code, path, expected):
    """The guard passing on a clean tree proves nothing about a dirty one, so the classifier is
    exercised directly.

    ``A`` is the entry worth naming: this repository's workflow stages without committing, so a new
    module added under the sibling and staged shows as ``A`` and nothing else -- neither a modified
    tracked file nor an untracked one -- and a guard reading only ``M`` and ``??`` would call that
    tree clean.
    """
    assert _is_sibling_edit(code, path) is expected


def test_the_registered_exception_names_a_file_that_exists():
    """An allow-list entry naming nothing would widen the guard silently rather than loudly."""
    for path in _SIBLING_ALLOWED_CHANGES:
        assert (_REPO_ROOT / path).is_file(), path


# =================================================================================================
# The suite's own surface
# =================================================================================================
def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use, and ``-m
    slow`` silently selects nothing."""
    markers = request.config.getini("markers")

    assert any(str(marker).startswith("slow") for marker in markers)


def test_the_invocation_lines_are_recorded_for_this_package():
    """``tests/__init__.py`` records how this suite is run, in both tiers, naming *this* package --
    a copy naming another one is a line nobody can paste."""
    recorded = (Path(__file__).resolve().parent / "__init__.py").read_text(encoding="utf-8")

    assert 'teb_vae/lag_attn_crws/tests -q -m "not slow"' in recorded
    assert "teb_vae/lag_attn_crws/tests -q -m slow" in recorded
    assert "lag_attn_cfs/tests" not in recorded
