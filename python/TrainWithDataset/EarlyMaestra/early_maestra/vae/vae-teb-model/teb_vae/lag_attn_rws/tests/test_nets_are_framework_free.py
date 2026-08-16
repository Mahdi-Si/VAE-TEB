"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layer.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``,
whose machinery is imported rather than restated. One extension is needed here: the shared
``_ALLOWED_ROOTS`` admits ``teb_vae`` wholesale -- necessarily, since this package reuses
``teb_vae.lag_attn.nets`` -- and that would wave through an import of any sibling's Lightning
task, trainer, plotting, config loader, evaluation package or test helpers. Those are forbidden
by dotted prefix instead, on every package in the family, so a net stays constructible without the
framework around it.

The downstream packages are in the ban although nothing here could import them today -- they sit
*below* this one, composing or subclassing its model. A guard whose coverage lags the family it
guards goes stale exactly when the code it protects gets written.
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

#: Every package whose framework layer a net here could reach into, this one included. A tuple
#: rather than an inline literal so the meta-test below can hold it to the whole family.
_PACKAGES = (
    "lag_attn",
    "lag_attn_rws",
    "lag_attn_transformer_rws",
    "lag_attn_transformer_e2e",
    "lag_attn_fs",
    "lag_attn_transformer_fs",
    "lag_attn_cfs",
    "lag_attn_transformer_cfs",
    "lag_attn_crws",
    "lag_attn_transformer_crws",
)

#: Everything under a ``teb_vae`` package that is not a net layer.
#:
#: ``sample_page`` is here for the same reason ``plotting`` is, and is the easier one to forget:
#: both modules import matplotlib and ``utils.style``, and a net that reached one for a row builder
#: would need a figure backend to construct. The page is reached through the task, which is where
#: the batch field names and the drawing both belong.
_FRAMEWORK_MODULES = ("task", "trainer", "plotting", "sample_page", "config", "eval", "tests")

# The dotted-prefix extension: everything under teb_vae that is not a net layer. Every package in
# the family is covered so a future edit cannot route a Lightning import through a sibling either.
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


def test_the_dotted_ban_covers_every_package_in_the_family():
    """The extension is only worth having if it names every package a net could reach into, and a
    new sibling arriving is exactly the event that makes a hand-kept list go stale."""
    for package in _PACKAGES:
        for module in _FRAMEWORK_MODULES:
            assert f"teb_vae.{package}.{module}" in _LOCAL_FORBIDDEN_PREFIXES


def test_the_framework_prefix_guard_fires(tmp_path):
    """The extension must catch what the shared root allowlist waves through.

    ``teb_vae.lag_attn.config`` has an allowed root, so only the dotted check can reject it --
    exactly the gap the extension exists to close.
    """
    offender = tmp_path / "offender.py"
    offender.write_text(
        "from teb_vae.lag_attn.config import load_config\n"
        "def f():\n"
        "    from teb_vae.lag_attn_rws.task import anything  # lazy imports count too\n",
        encoding="utf-8",
    )

    names = _imported_names(offender)
    assert {name.split(".")[0] for name in names} <= _ALLOWED_ROOTS, "root check passes here"
    caught = {
        name
        for name in names
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _LOCAL_FORBIDDEN_PREFIXES
        )
    }
    assert caught == {"teb_vae.lag_attn.config", "teb_vae.lag_attn_rws.task"}
