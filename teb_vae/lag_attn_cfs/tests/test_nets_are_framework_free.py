"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as every sibling package's copy does. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package's net layer is built almost entirely out of sibling
imports -- and that would wave through an import of any package's Lightning task, trainer, plotting,
diagnostic page, config loader, evaluation package or test helpers. Those are forbidden by dotted
prefix instead, on all **seven** packages of the family.

Two bans here are this package's own, and neither is hypothetical.

**Its own top-level modules.** ``causal_warmup.py`` opens HDF5 files and ``model_kwargs.py`` reads a
constructor signature; both sit outside ``nets/`` for exactly that reason. A net reaching either
would take ``h5py``, ``torch``'s dataset stack and a filesystem into a layer whose whole contract is
that it can be constructed from integers.

**The batch-field half bites harder here than anywhere.** This model's reconstruction target *is*
two named stored blocks, and its warm-up boundary arrives per block, so the temptation to name them
inside the net is real twice over. The resolver names them, on purpose, and it is not a net.
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
    """The extension is only worth having if it names every package a net could reach into, and
    this package's arrival is exactly the event that makes a hand-kept list go stale."""
    for package in _PACKAGES:
        for module in _FRAMEWORK_MODULES:
            assert f"teb_vae.{package}.{module}" in _LOCAL_FORBIDDEN_PREFIXES


def test_the_net_layer_reaches_all_three_halves_of_its_design():
    """The positive direction, which a guard that only forbade things could not give.

    The architecture comes from the raw-signal model, the input-side half of causality from
    ``causal_inputs`` and the target domain from ``causal_feature_target``, which is itself the
    two-sided mixin extended. A refactor that quietly stopped importing one -- inlining the tiled
    forward back into the model, or reaching the two-sided mixin directly and losing the block split
    -- fails here.

    Both mixins name no encoder, which is what lets a second architecture compose the identical
    pair; that is asserted structurally in the two tests below rather than by import.
    """
    imported = _imported_names(_NETS_DIR / "model.py")
    assert "teb_vae.lag_attn_rws.nets.model" in imported
    assert "teb_vae.lag_attn_cfs.nets.causal_inputs" in imported
    assert "teb_vae.lag_attn_cfs.nets.causal_feature_target" in imported
    assert "teb_vae.lag_attn_fs.nets.feature_target" not in imported

    target = _imported_names(_NETS_DIR / "causal_feature_target.py")
    assert "teb_vae.lag_attn_fs.nets.feature_target" in target


def test_neither_mixin_reaches_an_encoder_module():
    """What makes both composable over a second architecture. Each may name the shared *primitives*
    -- the availability adapter and the channel gate are ``lag_attn``'s, not either architecture's --
    but neither may reach a model module, which is where an encoder is chosen."""
    for name in ("causal_inputs.py", "causal_feature_target.py"):
        imported = _imported_names(_NETS_DIR / name)
        assert "teb_vae.lag_attn_rws.nets.encoders" not in imported, name
        assert "teb_vae.lag_attn_transformer_rws.nets.encoders" not in imported, name
        assert "teb_vae.lag_attn_transformer_rws.nets.model" not in imported, name


def test_the_model_module_writes_nothing_but_a_constructor():
    """The other side of the positive direction. Everything else is encoder-agnostic and lives on a
    mixin the conv-Transformer cell composes too, so a member appearing here is one that cell
    silently does not get."""
    source = (_NETS_DIR / "model.py").read_text(encoding="utf-8")

    assert source.count("\nclass ") == 1
    assert source.count("def __init__") == 1
    for banned in ("def forward", "nn.Linear", "nn.Conv1d", "register_buffer"):
        assert banned not in source, f"nets/model.py writes {banned!r}"


def test_the_top_level_modules_are_outside_the_net_layer():
    """``causal_warmup`` opens shards, ``model_kwargs`` reads a signature and the two figure
    modules import matplotlib, so all four are above the net layer by construction rather than by
    convention -- and a move into ``nets/`` would be caught by the import guard above rather than
    going unseen."""
    for name in ("causal_warmup.py", "model_kwargs.py", "warmup_budget.py", "sample_page.py"):
        assert not (_NETS_DIR / name).exists()
        assert (_PACKAGE_DIR / name).exists()


def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route; a stray global seed would silently override it while looking like diligence."""
    offenders = []
    for path in _PACKAGE_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue  # tests seed themselves for reproducibility, legitimately
        source = path.read_text(encoding="utf-8")
        # The CALL, not the name: this package's task takes the run seed as a keyword and has to
        # name the framework's own seeding route to explain where it comes from, and a bare
        # substring scan would read that sentence as a hand seed.
        for pattern in ("torch.manual_seed(", "seed_everything(", "np.random.seed("):
            if pattern in source:
                offenders.append(f"{path.name}: {pattern}")
    assert offenders == []
