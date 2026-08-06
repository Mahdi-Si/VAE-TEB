"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as both sibling packages' own copies do. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package reuses net layers from three sibling packages -- and
that would wave through an import of any of their Lightning tasks, trainers, plotting, config
loaders, evaluation packages or test helpers. Those are forbidden by dotted prefix instead, on all
**four** packages, so a net stays constructible without the framework around it.

This package's own ``nets`` is included in the ban even though nothing under it could import its own
task today: the task does not exist yet, and a guard whose coverage lags the package it guards is a
guard that goes stale exactly when the code it protects gets written.

One import here would look like a violation and is not. ``nets/frontend.py`` reaches into
``teb_vae.lag_attn_rws.nets.raw_masks`` for the validity threshold. That module is inside ``nets/``,
carries no framework, and is the repository's single definition of what makes a decimated step
valid -- which is the point of importing it rather than restating a float comparison that could
drift from the mask the objective scores against.

The sibling's "the guard fires" self-test is deliberately not ported, for the reason its own copy
records: it is proven there against the same machinery, and repeating it here would test the import
rather than this package.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from teb_vae.lag_attn.tests.test_nets_are_framework_free import (
    _ALLOWED_ROOTS,
    _BATCH_FIELD_NAMES,
    _FORBIDDEN_PREFIXES,
    _imported_names,
)

_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"

#: Every package whose framework layer a net here could reach into, this one included.
_PACKAGES = (
    "lag_attn",
    "lag_attn_rws",
    "lag_attn_transformer_rws",
    "lag_attn_transformer_e2e",
)

#: Everything under a ``teb_vae`` package that is not a net layer.
_FRAMEWORK_MODULES = ("task", "trainer", "plotting", "config", "eval", "tests")

_FRAMEWORK_PREFIXES = tuple(
    f"teb_vae.{package}.{module}" for package in _PACKAGES for module in _FRAMEWORK_MODULES
)
_LOCAL_FORBIDDEN_PREFIXES = _FORBIDDEN_PREFIXES + _FRAMEWORK_PREFIXES


def _net_modules() -> list[Path]:
    return sorted(_NETS_DIR.glob("*.py"))


def test_there_are_net_modules_to_check():
    """A silently-empty glob would make every test below vacuous."""
    assert _net_modules(), f"no modules found under {_NETS_DIR}"


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_imports_only_torch_stdlib_entmax_and_teb_vae(path):
    offenders = sorted(
        name for name in _imported_names(path) if name.split(".")[0] not in _ALLOWED_ROOTS
    )
    assert not offenders, (
        f"nets/{path.name} imports {offenders} -- nets/ may import only torch, the standard "
        f"library, entmax and the teb_vae net layers, so that a network can be built without "
        f"the framework around it"
    )


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_avoids_forbidden_submodules(path):
    offenders = sorted(
        name
        for name in _imported_names(path)
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _LOCAL_FORBIDDEN_PREFIXES
        )
    )
    assert not offenders, (
        f"nets/{path.name} imports {offenders} -- a net must not need a process group, a "
        f"config file or a Lightning module to run"
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


def test_the_dotted_ban_covers_all_four_packages():
    """The extension is only worth having if it names every package a net could reach into, and
    this package is now the fourth."""
    for package in _PACKAGES:
        for module in _FRAMEWORK_MODULES:
            assert f"teb_vae.{package}.{module}" in _LOCAL_FORBIDDEN_PREFIXES


def test_the_front_end_may_reach_the_shared_validity_threshold():
    """The one import that looks like a violation. ``raw_masks`` is inside ``nets/``, so it is legal
    -- and importing it is what keeps the front end's notion of a valid step identical to the mask
    the objective scores against."""
    imported = _imported_names(_NETS_DIR / "frontend.py")

    assert "teb_vae.lag_attn_rws.nets.raw_masks" in imported
    assert not any(
        name.startswith(prefix) for name in imported for prefix in _FRAMEWORK_PREFIXES
    )
