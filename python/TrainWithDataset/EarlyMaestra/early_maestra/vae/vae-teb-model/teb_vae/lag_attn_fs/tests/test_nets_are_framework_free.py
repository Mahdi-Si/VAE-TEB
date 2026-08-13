"""``nets/`` may import torch, the standard library, ``entmax`` and the sibling net layers.

The rule and its rationale live in ``teb_vae/lag_attn/tests/test_nets_are_framework_free.py``,
whose machinery is imported rather than restated, exactly as all three sibling packages' own
copies do. One extension is needed and it is the same one: the shared ``_ALLOWED_ROOTS`` admits
``teb_vae`` wholesale -- necessarily, since this package's model composes a sibling's and reuses
its objective outright -- and that would wave through an import of any package's Lightning task,
trainer, plotting, diagnostic page, config loader, evaluation package or test helpers. Those are
forbidden by dotted prefix instead, on all **six** packages of the family, so a net stays
constructible without the framework around it.

The batch-field half of the rule bites harder here than anywhere else in the family. This model's
reconstruction target *is* two named stored blocks, so the temptation to concatenate them inside
the net is real and the guard is what refuses it: the task concatenates, and what arrives here is
one ``(B, T, c_y)`` tensor whose origin this layer does not know. It is also why the shared
``figure_primitives.future_target`` -- whose signature is those two blocks -- is not called from
``nets/``: taking that signature would mean holding the schema the guard forbids. The three-token
unfold is written here instead, and ``test_objective.py`` pins it equal to the shared helper.

The same rule is why ``sample_page`` joins the dotted ban rather than being covered by ``plotting``
alone. This package's forecast rows live in ``lag_attn_fs/sample_page.py``, they are named after the
two stored blocks, and they import matplotlib -- so a net reaching for them would take both the
schema and a figure backend. They are reached through the task's ``forecast_rows`` seam and nowhere
else.

The sibling's "the guard fires" self-tests are deliberately not ported, for the reason their own
copies record: they are proven there against the same machinery, and repeating them here would
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
    """The extension is only worth having if it names every package a net could reach into, and
    this package's arrival is exactly the event that makes a hand-kept list go stale."""
    for package in _PACKAGES:
        for module in _FRAMEWORK_MODULES:
            assert f"teb_vae.{package}.{module}" in _LOCAL_FORBIDDEN_PREFIXES


def test_the_net_layer_reaches_the_sibling_it_composes_with():
    """The positive direction, and the reason the root allowlist cannot simply exclude
    ``teb_vae``: this model's net *must* import the sibling's model and objective, and a guard
    that forbade it would forbid the architecture.

    Asserted per module rather than over the union, because which module reaches what is the
    package's shape. ``feature_target.py`` is the target domain and reaches the objective; nothing
    else. ``model.py`` is the composition and reaches exactly its two halves -- an encoder model and
    that target -- so a refactor that inlined the target back into the model, or reached an encoder
    from the target, fails here rather than passing on a union that still contains both names.
    """
    target_domain = _imported_names(_NETS_DIR / "feature_target.py")
    composition = _imported_names(_NETS_DIR / "model.py")

    assert "teb_vae.lag_attn_rws.nets.losses" in target_domain
    assert "teb_vae.lag_attn_rws.nets.raw_masks" in target_domain
    assert "teb_vae.lag_attn_rws.nets.model" not in target_domain, (
        "the target domain must not name an encoder model, or it could not be mixed into a second"
    )

    assert "teb_vae.lag_attn_rws.nets.model" in composition
    assert "teb_vae.lag_attn_fs.nets.feature_target" in composition
