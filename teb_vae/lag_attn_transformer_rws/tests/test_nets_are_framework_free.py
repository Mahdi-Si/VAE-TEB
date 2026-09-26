"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``,
whose machinery is imported rather than restated, as the sibling model's own copy does. One
extension is needed: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae`` wholesale -- necessarily,
since this package reuses ``teb_vae.lag_attn.nets`` and ``teb_vae.lag_attn_rws.nets`` -- and that
would wave through an import of any package's Lightning tasks, trainers, plotting, config loaders,
evaluation packages or test helpers. Those are forbidden by dotted prefix instead, on every package
in the family, so a net stays constructible without the framework around it.

The sibling's "the guard fires" self-test is deliberately not ported. It is proven there against
the same machinery; repeating it here would test the import, not this package.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from teb_vae.lag_attn.tests.test_nets_are_framework_free import (
    _ALLOWED_ROOTS,
    _FORBIDDEN_PREFIXES,
    _imported_names,
)

_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"

# The dotted-prefix extension: everything under teb_vae that is not a net layer. Every package in
# the family is covered so a future edit cannot route a Lightning import through a sibling.
#: Every package whose framework layer a net here could reach into, this one included.
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

_FRAMEWORK_PREFIXES = tuple(
    f"teb_vae.{package}.{module}" for package in _PACKAGES for module in _FRAMEWORK_MODULES
)
_LOCAL_FORBIDDEN_PREFIXES = _FORBIDDEN_PREFIXES + _FRAMEWORK_PREFIXES


def _net_modules() -> list[Path]:
    return sorted(_NETS_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_imports_stay_inside_the_net_layer(path):
    """Only torch, the standard library, ``entmax`` and ``teb_vae`` roots, and within ``teb_vae``
    no framework module of any package in the family."""
    names = _imported_names(path)
    outside = sorted(name for name in names if name.split(".")[0] not in _ALLOWED_ROOTS)
    forbidden = sorted(
        name
        for name in names
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _LOCAL_FORBIDDEN_PREFIXES
        )
    )
    assert not outside, (
        f"nets/{path.name} imports {outside} -- nets/ may import only torch, the standard "
        f"library, entmax and the teb_vae net layers, so that a network can be built without "
        f"the framework around it"
    )
    assert not forbidden, (
        f"nets/{path.name} imports {forbidden} -- a net must not need a process group, a "
        f"config file or a Lightning module to run"
    )
