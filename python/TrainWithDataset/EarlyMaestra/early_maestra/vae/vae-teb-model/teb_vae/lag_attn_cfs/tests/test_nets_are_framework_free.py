"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as every sibling package's copy does. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package's net layer is built almost entirely out of sibling
imports -- and that would wave through an import of any package's Lightning task, trainer, plotting,
diagnostic page, config loader, evaluation package or test helpers. Those are forbidden by dotted
prefix instead, on every package of the family.

**Its own top-level modules are banned too.** ``causal_warmup.py`` opens HDF5 files,
``model_kwargs.py`` reads a constructor signature and ``warmup_budget.py`` draws figures; a net
reaching any of them would take ``h5py``, a filesystem or matplotlib into a layer whose whole
contract is that it can be constructed from integers.

And the two mixins must not reach an encoder module, which is what lets a second architecture
compose the identical pair.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from teb_vae.lag_attn.tests.test_nets_are_framework_free import (
    _ALLOWED_ROOTS,
    _FORBIDDEN_PREFIXES,
    _imported_names,
)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_NETS_DIR = _PACKAGE_DIR / "nets"

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
#: ``model_kwargs`` and ``warmup_budget`` are this package's additions to the list: the first opens
#: files, the second introspects a constructor, the third draws matplotlib figures, and none of
#: them belongs behind a layer that must build from integers.
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
        f"nets/{path.name} imports {offenders} -- a net must not need a process group, a config "
        f"file, a Lightning module or an HDF5 shard to run"
    )


def test_neither_mixin_reaches_an_encoder_module():
    """What makes both composable over a second architecture. Each may name the shared *primitives*
    -- the availability adapter and the channel gate are ``lag_attn``'s, not either architecture's --
    but neither may reach a model module, which is where an encoder is chosen."""
    for name in ("causal_inputs.py", "causal_feature_target.py"):
        imported = _imported_names(_NETS_DIR / name)
        assert "teb_vae.lag_attn_rws.nets.encoders" not in imported, name
        assert "teb_vae.lag_attn_transformer_rws.nets.encoders" not in imported, name
        assert "teb_vae.lag_attn_transformer_rws.nets.model" not in imported, name
