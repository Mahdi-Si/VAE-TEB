"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as every sibling package's copy does. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package's model is nothing but three sibling imports -- and that
would wave through an import of any package's Lightning task, trainer, plotting, diagnostic page,
config loader, evaluation package or test helpers. Those are forbidden by dotted prefix instead, on
all **seven** packages of the family, so a net stays constructible without the framework around it.

Two bans are inherited from the conv-LSTM causal cell and neither is hypothetical. Its
``causal_warmup.py`` opens HDF5 files, its ``model_kwargs.py`` reads a constructor signature and its
``warmup_budget.py`` draws matplotlib figures; all three sit outside ``nets/`` for exactly that
reason, and a net here reaching any of them would take ``h5py``, a filesystem and a figure backend
into a layer whose whole contract is that it can be constructed from integers.

The batch-field half of the rule bites harder here than anywhere: this model's reconstruction target
*is* two named stored blocks and its warm-up boundary arrives per block, so the temptation to name
them inside the net is real twice over. The task concatenates, the resolver names the blocks, and
what arrives at the net is one ``(B, T, c_y)`` tensor whose origin this layer does not know.

The **positive** direction is what this copy adds over the siblings'. This package writes no network
code at all, so a guard that only forbade things would pass on an empty directory: the net layer has
to be shown reaching *all three* imports, one for the architecture and two for the target domain.

The sibling's "the guard fires" self-tests are deliberately not ported, for the reason their own
copies record: they are proven there against the same machinery, and repeating them here would test
the import rather than this package.
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
    "lag_attn_fs",
    "lag_attn_transformer_fs",
    "lag_attn_cfs",
    "lag_attn_transformer_cfs",
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


def test_the_net_layer_reaches_all_three_of_its_halves():
    """The positive direction, and the one this copy exists for.

    This package writes no network code, so a guard that only forbade things would be satisfied by
    an empty directory. All three imports must be reached by name: the architecture from
    ``lag_attn_transformer_rws.nets.model``, the input-side half of causality from
    ``lag_attn_cfs.nets.causal_inputs`` and the target domain from
    ``lag_attn_cfs.nets.causal_feature_target``. A refactor that quietly stopped importing one --
    inlining the tiled forward back into the model, or reaching the two-sided mixin directly and
    losing the block split -- fails here.
    """
    imported = _imported_names(_NETS_DIR / "model.py")

    assert "teb_vae.lag_attn_transformer_rws.nets.model" in imported
    assert "teb_vae.lag_attn_cfs.nets.causal_inputs" in imported
    assert "teb_vae.lag_attn_cfs.nets.causal_feature_target" in imported
    # And *not* either conv-LSTM model, which is the specific wrong parent: inheriting from one
    # would linearise through its constructor and build a conv-LSTM model with nothing raising.
    assert "teb_vae.lag_attn_rws.nets.model" not in imported
    assert "teb_vae.lag_attn_cfs.nets.model" not in imported
    # Nor the two-sided mixin directly, which would restore the two-sided block split of 43 and
    # mislabel both reported block gaps against a stored block that is 36 wide here.
    assert "teb_vae.lag_attn_fs.nets.feature_target" not in imported


def test_the_net_layer_writes_no_architecture_of_its_own():
    """The other side of the positive direction: exactly one module, holding one class body.

    The encoder primitives are reached through the architecture parent and the warm-up mask, the
    lag floor and the anchor tiling through the causal one, rather than copied. A second module
    appearing under ``nets/`` is the event that should make someone re-read this.

    ``def __init__`` is the one exception the guard has to admit, and the reason is recorded where
    it is written: the experiment driver builds a run's kwargs by sweeping
    ``inspect.signature(MODEL_CLS.__init__)``, so a cell whose architecture has its own keyword
    schema has to state that schema. ``def forward`` is *not* admitted -- a forward here would be a
    second copy of the tiled one, free to disagree with the objective's anchor set.
    """
    modules = [path.name for path in _net_modules()]

    assert modules == ["__init__.py", "model.py"], modules
    source = (_NETS_DIR / "model.py").read_text(encoding="utf-8")
    assert source.count("\nclass ") == 1
    assert source.count("def __init__") == 1
    for banned in ("def forward", "nn.Linear", "nn.Conv1d", "register_buffer"):
        assert banned not in source, f"nets/model.py writes {banned!r}; it is meant to write nothing"
