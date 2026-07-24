"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layer.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``,
whose machinery is imported rather than restated. One extension is needed here: the shared
``_ALLOWED_ROOTS`` admits ``teb_vae`` wholesale -- necessarily, since this package reuses
``teb_vae.lag_attn.nets`` -- and that would wave through an import of either model's Lightning
task, trainer, plotting, config loader, evaluation package or test helpers. Those are forbidden
by dotted prefix instead, on both packages, so a net stays constructible without the framework
around it.
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

# The dotted-prefix extension: everything under teb_vae that is not a net layer. Both packages
# are covered so a future edit cannot route a Lightning import through the sibling either.
_FRAMEWORK_PREFIXES = tuple(
    f"teb_vae.{package}.{module}"
    for package in ("lag_attn", "lag_attn_rws")
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
