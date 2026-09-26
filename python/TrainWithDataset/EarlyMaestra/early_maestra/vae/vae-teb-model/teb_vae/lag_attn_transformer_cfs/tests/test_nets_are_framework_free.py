"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as every sibling package's copy does. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package's model is nothing but three sibling imports -- and that
would wave through an import of any package's Lightning task, trainer, plotting, diagnostic page,
config loader, evaluation package or test helpers. Those are forbidden by dotted prefix instead, on
every package of the family, so a net stays constructible without the framework around it.

Two bans are inherited from the conv-LSTM causal cell and neither is hypothetical. Its
``causal_warmup.py`` opens HDF5 files, its ``model_kwargs.py`` reads a constructor signature and its
``warmup_budget.py`` draws matplotlib figures; all three sit outside ``nets/`` for exactly that
reason, and a net here reaching any of them would take ``h5py``, a filesystem and a figure backend
into a layer whose whole contract is that it can be constructed from integers.
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

#: Everything under a ``teb_vae`` package that is not a net layer. ``causal_warmup``,
#: ``model_kwargs`` and ``warmup_budget`` are the causal cell's additions to the list: the first
#: opens files, the second introspects a constructor, the third draws matplotlib figures, and none
#: of them belongs behind a layer that must build from integers. ``sample_page`` is here for the
#: same reason ``plotting`` is, and is the easier one to forget: both import matplotlib and
#: ``utils.style``, so a net that reached one for a row builder would need a figure backend to
#: construct.
_FRAMEWORK_MODULES = (
    "task",
    "trainer",
    "plotting",
    "sample_page",
    "config",
    "eval",
    "tests",
    "causal_warmup",
    "model_kwargs",
    "warmup_budget",
)

_FRAMEWORK_PREFIXES = tuple(
    f"teb_vae.{package}.{module}" for package in _PACKAGES for module in _FRAMEWORK_MODULES
)
_LOCAL_FORBIDDEN_PREFIXES = _FORBIDDEN_PREFIXES + _FRAMEWORK_PREFIXES


def _net_modules() -> list[Path]:
    return sorted(_NETS_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_imports_only_allowed_roots_and_no_framework_module(path):
    """Two halves of one boundary: every import root is torch, the standard library, ``entmax`` or
    ``teb_vae``, and no ``teb_vae`` import reaches a process group, a config file, a Lightning
    module or an HDF5 shard."""
    imported = _imported_names(path)

    outside = sorted(name for name in imported if name.split(".")[0] not in _ALLOWED_ROOTS)
    assert not outside, (
        f"nets/{path.name} imports {outside} -- nets/ may import only torch, the standard "
        f"library, entmax and the teb_vae net layers, so that a network can be built without "
        f"the framework around it"
    )
    forbidden = sorted(
        name
        for name in imported
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _LOCAL_FORBIDDEN_PREFIXES
        )
    )
    assert not forbidden, (
        f"nets/{path.name} imports {forbidden} -- a net must not need a process group, a config "
        f"file, a Lightning module or an HDF5 shard to run"
    )
