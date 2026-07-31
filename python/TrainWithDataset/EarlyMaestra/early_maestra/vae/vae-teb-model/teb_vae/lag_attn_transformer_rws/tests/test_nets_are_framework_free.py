"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``,
whose machinery is imported rather than restated, as the sibling model's own copy does. One
extension is needed: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae`` wholesale -- necessarily,
since this package reuses ``teb_vae.lag_attn.nets`` and ``teb_vae.lag_attn_rws.nets`` -- and that
would wave through an import of any of the three packages' Lightning tasks, trainers, plotting,
config loaders, evaluation packages or test helpers. Those are forbidden by dotted prefix instead,
on all three, so a net stays constructible without the framework around it.

The sibling's "the guard fires" self-test is deliberately not ported. It is proven there against
the same machinery; repeating it here would test the import, not this package.
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

# The dotted-prefix extension: everything under teb_vae that is not a net layer. All three packages
# are covered so a future edit cannot route a Lightning import through either sibling.
_FRAMEWORK_PREFIXES = tuple(
    f"teb_vae.{package}.{module}"
    for package in ("lag_attn", "lag_attn_rws", "lag_attn_transformer_rws")
    for module in ("task", "trainer", "plotting", "config", "eval", "tests")
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


def test_the_dotted_ban_covers_all_three_packages():
    """The extension is only worth having if it names every package a net could reach into."""
    for package in ("lag_attn", "lag_attn_rws", "lag_attn_transformer_rws"):
        for module in ("task", "trainer", "plotting", "config", "eval", "tests"):
            assert f"teb_vae.{package}.{module}" in _LOCAL_FORBIDDEN_PREFIXES
