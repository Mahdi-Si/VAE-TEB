"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``, whose
machinery is imported rather than restated, exactly as every sibling package's copy does. One
extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits ``teb_vae``
wholesale -- necessarily, since this package's model is nothing but two sibling imports -- and that
would wave through an import of any package's Lightning task, trainer, plotting, diagnostic page,
config loader, evaluation package or test helpers. Those are forbidden by dotted prefix instead, on
every package of the family, so a net stays constructible without the framework around it.

Three bans are inherited from the causal-feature cell and none is hypothetical. Its
``causal_warmup.py`` opens HDF5 files, its ``model_kwargs.py`` reads a constructor signature and its
``warmup_budget.py`` draws matplotlib figures; all three sit outside ``nets/`` for exactly that
reason, and a net here reaching any of them would take ``h5py``, a filesystem and a figure backend
into a layer whose whole contract is that it can be constructed from integers.

The **batch-field** half bites the way it does on the conv-LSTM cell of this row: the reconstruction
target is a raw signal, and the raw arrays are batch fields as much as the stored blocks are, so a
net that named one would have learned the dataset's schema for the one tensor it is most tempting to
fetch rather than be handed.

The **positive** direction is what this copy adds over the siblings'. This package writes no network
code at all, so a guard that only forbade things would pass on an empty directory: the net layer has
to be shown reaching *both* imports, one for the architecture and one for the input domain.

The sibling packages' "the guard fires" self-tests are deliberately not ported, for the reason their
own copies record: they are proven there against the same machinery, and repeating them here would
test the import rather than this package.
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
#: the third draws matplotlib figures. ``sample_page`` is here for the same reason ``plotting`` is,
#: and is the easier one to forget: both import matplotlib and ``utils.style``, so a net that
#: reached one for a row builder would need a figure backend to construct.
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


def test_the_net_layer_reaches_both_of_its_halves():
    """The positive direction, and the one this copy exists for.

    This package writes no network code, so a guard that only forbade things would be satisfied by
    an empty directory. Both imports must be reached by name: the architecture from
    ``lag_attn_transformer_rws.nets.model`` and the whole input domain -- the warm-up mask, the lag
    floor, the anchor tiling, the anchored raw gather and the three readouts -- from
    ``lag_attn_crws.nets.causal_raw_inputs``.
    """
    imported = _imported_names(_NETS_DIR / "model.py")

    assert "teb_vae.lag_attn_transformer_rws.nets.model" in imported
    assert "teb_vae.lag_attn_crws.nets.causal_raw_inputs" in imported
    # And *not* either conv-LSTM model, which is the specific wrong parent: inheriting from one
    # would linearise through its constructor and build a conv-LSTM model with nothing raising.
    assert "teb_vae.lag_attn_rws.nets.model" not in imported
    assert "teb_vae.lag_attn_crws.nets.model" not in imported
    # Nor the feature-target mixin, which would move the decoder to $C_{\mathrm{keep}} = 98$ against
    # a $(B, A, H, 16)$ raw target and fail three frames below the decision that caused it.
    assert "teb_vae.lag_attn_cfs.nets.causal_feature_target" not in imported


def test_the_net_layer_writes_no_architecture_of_its_own():
    """The other side of the positive direction: exactly one module, holding one class body.

    The encoder primitives are reached through the architecture parent and the warm-up mask, the
    lag floor, the anchor tiling and the anchored objective through the causal-input one, rather
    than copied. A second module appearing under ``nets/`` is the event that should make someone
    re-read this.

    ``def __init__`` is the one exception the guard has to admit, and the reason is recorded where
    it is written: the experiment driver builds a run's kwargs by sweeping
    ``inspect.signature(MODEL_CLS.__init__)``, so a cell whose architecture has its own keyword
    schema has to state that schema. ``def forward`` and ``def compute_loss`` are *not* admitted --
    either here would be a second copy of a member the conv-LSTM cell of this row already owns,
    free to disagree with it about which anchors a step scored.
    """
    modules = [path.name for path in _net_modules()]

    assert modules == ["__init__.py", "model.py"], modules
    source = (_NETS_DIR / "model.py").read_text(encoding="utf-8")
    assert source.count("\nclass ") == 1
    assert source.count("def __init__") == 1
    for banned in ("def forward", "def compute_loss", "nn.Linear", "nn.Conv1d", "register_buffer"):
        assert banned not in source, f"nets/model.py writes {banned!r}; it is meant to write nothing"


def test_this_package_ships_none_of_the_modules_it_reaches_by_reference():
    """The warm-up resolution, the kwargs mapping and the budget figure all live one or two packages
    over, and the diagnostic page lives in the conv-LSTM cell of this row.

    A copy of any of them here would be a second definition of a quantity two cells must agree on:
    one threshold's translation into four channel tuples, or one answer to which anchors a run
    decoded. The assertion is therefore that this package ships none of them, anywhere, while the
    packages that own them still do.
    """
    causal_feature = _PACKAGE_DIR.parent / "lag_attn_cfs"
    for name in ("causal_warmup.py", "model_kwargs.py", "warmup_budget.py"):
        assert not list(_PACKAGE_DIR.rglob(name)), name
        assert (causal_feature / name).is_file(), name

    conv_lstm = _PACKAGE_DIR.parent / "lag_attn_crws"
    assert not list(_PACKAGE_DIR.rglob("sample_page.py"))
    assert (conv_lstm / "sample_page.py").is_file()


def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route; a stray global seed would silently override it while looking like diligence -- and here
    it would additionally move every tile phase, since the seed is one of the four halves of the
    phase key.

    The scan itself is :func:`~teb_vae.lag_attn_cfs.tests.conftest.hand_seeding_offenders`, reached
    by reference so this package and its siblings check one rule.
    """
    assert hand_seeding_offenders(_PACKAGE_DIR) == []
