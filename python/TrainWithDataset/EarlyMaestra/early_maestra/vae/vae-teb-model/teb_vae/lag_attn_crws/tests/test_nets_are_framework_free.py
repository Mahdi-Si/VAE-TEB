"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as every sibling package's copy does. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package's net layer is built almost entirely out of sibling
imports -- and that would wave through an import of any package's Lightning task, trainer, plotting,
diagnostic page, config loader, evaluation package or test helpers. Those are forbidden by dotted
prefix instead, on every package of the family.

**This package's own version of the ban is about a sibling's modules rather than its own.** It ships
no ``causal_warmup.py`` and no ``model_kwargs.py``: the warm-up resolution opens HDF5 files and the
kwargs mapping introspects a constructor, and both already exist one package over, reached by
reference. So the dotted ban still names them -- a net here reaching
``teb_vae.lag_attn_cfs.causal_warmup`` would take ``h5py``, a filesystem and a dataset stack into a
layer whose whole contract is that it can be built from integers -- while the "these live above the
net layer" assertion becomes the statement that this package does not ship them at all.

The **batch-field** half bites differently here than on the feature-target cells. This model's
reconstruction target is a raw signal, and the raw arrays are batch fields as much as the stored
blocks are: a net that named one would have learned the dataset's schema for the one tensor it is
most tempting to fetch rather than be handed.
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

from .conftest import hand_seeding_offenders

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
#: ``model_kwargs`` and ``warmup_budget`` are the causal-feature cell's top-level modules and are
#: banned by prefix on every package: the first opens files, the second introspects a constructor,
#: the third draws matplotlib figures, and this package reaches the first two by reference from
#: above the net layer rather than from inside it.
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


def _net_modules() -> list:
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


@pytest.mark.parametrize("path", _net_modules(), ids=lambda p: p.name)
def test_module_names_no_batch_fields(path):
    source = path.read_text(encoding="utf-8")
    offenders = sorted(name for name in _BATCH_FIELD_NAMES if re.search(rf"\b{name}\b", source))
    assert not offenders, (
        f"nets/{path.name} names the batch fields {offenders} -- a net takes tensors as "
        f"arguments and does not know what they were called on disk"
    )


def test_the_dotted_ban_covers_every_package_in_the_family():
    """The extension is only worth having if it names every package a net could reach into, and
    this package's arrival is exactly the event that makes a hand-kept list go stale."""
    for package in _PACKAGES:
        for module in _FRAMEWORK_MODULES:
            assert f"teb_vae.{package}.{module}" in _LOCAL_FORBIDDEN_PREFIXES


def test_the_net_layer_reaches_both_halves_of_its_design():
    """The positive direction, which a guard that only forbade things could not give.

    The architecture comes from the raw-signal model and the whole causal input half from
    ``lag_attn_cfs``: the tiled forward, the warm-up adapter and the floored lag mask through
    ``causal_inputs``, and the five bound source-side members through ``causal_feature_target``. A
    refactor that quietly stopped importing one -- inlining the tiled forward back into the model,
    or copying the two readouts instead of binding them -- fails here.
    """
    model = _imported_names(_NETS_DIR / "model.py")
    assert "teb_vae.lag_attn_rws.nets.model" in model
    assert "teb_vae.lag_attn_crws.nets.causal_raw_inputs" in model

    mixin = _imported_names(_NETS_DIR / "causal_raw_inputs.py")
    assert "teb_vae.lag_attn_cfs.nets.causal_inputs" in mixin
    assert "teb_vae.lag_attn_cfs.nets.causal_feature_target" in mixin
    assert "teb_vae.lag_attn_rws.nets.raw_targets" in mixin
    assert "teb_vae.lag_attn_rws.nets.losses" in mixin


def test_the_mixin_reaches_no_encoder_module():
    """What makes it composable over a second architecture. It may name the shared *primitives* --
    the availability adapter and the channel gate are ``lag_attn``'s, not either architecture's --
    and it may name the raw-signal model, which is where the objective's own delegation lives; what
    it may not do is choose an encoder."""
    imported = _imported_names(_NETS_DIR / "causal_raw_inputs.py")

    for banned in (
        "teb_vae.lag_attn_rws.nets.encoders",
        "teb_vae.lag_attn_transformer_rws.nets.encoders",
        "teb_vae.lag_attn_transformer_rws.nets.model",
        "teb_vae.lag_attn_rws.nets.model",
    ):
        assert banned not in imported, banned


def test_the_model_module_writes_nothing_but_a_constructor():
    """The other side of the positive direction. Everything else is encoder-agnostic and lives on
    the mixin the conv-Transformer cell composes too, so a member appearing here is one that cell
    silently does not get."""
    source = (_NETS_DIR / "model.py").read_text(encoding="utf-8")

    assert source.count("\nclass ") == 1
    assert source.count("def __init__") == 1
    for banned in ("def forward", "def compute_loss", "nn.Linear", "nn.Conv1d", "register_buffer"):
        assert banned not in source, f"nets/model.py writes {banned!r}"


def test_this_package_ships_neither_of_the_two_modules_it_reaches_by_reference():
    """``causal_warmup`` opens shards and ``model_kwargs`` reads a signature, so both sit above any
    net layer -- and both already exist one package over.

    A copy here would be a second translation of one threshold into four channel tuples, free to
    describe a boundary the data no longer has, which is exactly what reading the shards rather than
    declaring the vectors exists to prevent. The assertion is therefore that this package ships
    neither, anywhere, while the sibling still does.
    """
    sibling = _PACKAGE_DIR.parent / "lag_attn_cfs"
    for name in ("causal_warmup.py", "model_kwargs.py"):
        assert not list(_PACKAGE_DIR.rglob(name)), name
        assert (sibling / name).is_file(), name


def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route; a stray global seed would silently override it while looking like diligence -- and here
    it would additionally move every tile phase, since the seed is one of the four halves of the
    phase key.

    The scan itself is :func:`~teb_vae.lag_attn_cfs.tests.conftest.hand_seeding_offenders`, reached
    by reference so this package and its sibling check one rule.
    """
    assert hand_seeding_offenders(_PACKAGE_DIR) == []
