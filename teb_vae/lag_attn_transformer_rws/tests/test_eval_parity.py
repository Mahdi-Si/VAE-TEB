r"""This model's analysis registry against the sibling's, in both directions.

The two models are evaluated by one pipeline so that their runs are readable side by side: the
same ``summary.json`` blocks, the same per-recording CSVs, the same figure families. That holds
only while the two registries agree, and the way it stops holding is silent -- an analysis added to
the sibling reaches this package automatically *if* nothing here pins the set, and an analysis
added here would not reach the sibling at all.

So the expected set is computed from the registries themselves rather than written down:
``set(sibling.ANALYSIS_FUNCTIONS) | set(TRF_BINDING.extra_analyses)``. A hand-written list would
have to be edited every time either side gained an entry, which is the same as not having the test
-- the edit and the change would arrive in one commit and the test would confirm what it was just
told. Computed, it fails on any *unexpected* difference and passes on an intended one without
being touched.

``UNSKIPPABLE_ANALYSES`` is compared with no local additions permitted at all. A step that always
runs describes the **data** rather than the model -- the input channel map reads the shards' own
provenance and needs neither a forward pass nor a table -- so a model-specific entry there would
be a category error rather than a divergence.
"""
from __future__ import annotations

from typing import Dict, Set

import pytest

from teb_vae.lag_attn_rws.eval import run as shared_run
from teb_vae.lag_attn_rws.eval.binding import ModelBinding
from teb_vae.lag_attn_transformer_rws.eval import run as trf_run
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING


def _difference_message(mine: Set[str], expected: Set[str]) -> str:
    """Name which analyses are only in one place, and which set each belongs in."""
    return (
        f"only in this package: {sorted(mine - expected)}; "
        f"only in the sibling registry: {sorted(expected - mine)}. "
        f"A shared analysis belongs in teb_vae.lag_attn_rws.eval.run.ANALYSIS_FUNCTIONS, where "
        f"both models get it; one only this architecture can have belongs in "
        f"TRF_BINDING.extra_analyses."
    )


# =============================================================================
# The selectable registry
# =============================================================================
def test_the_registry_is_the_siblings_plus_exactly_this_models_additions() -> None:
    mine = set(trf_run.analysis_registry())
    expected = set(shared_run.ANALYSIS_FUNCTIONS) | set(TRF_BINDING.extra_analyses)

    assert mine == expected, _difference_message(mine, expected)


def test_the_shared_analyses_keep_the_siblings_run_order() -> None:
    """Order is a reading order and in one case load-bearing: ``cross_subgroup`` runs last because
    it reads the per-recording CSVs the analyses above it write. Extras are appended after, so
    reordering the shared registry for one model's addition cannot happen by accident."""
    mine = list(trf_run.analysis_registry())
    shared = list(shared_run.ANALYSIS_FUNCTIONS)

    assert mine[: len(shared)] == shared
    assert mine[len(shared) :] == list(TRF_BINDING.extra_analyses)


def test_each_registered_name_maps_to_the_same_implementation_as_the_siblings() -> None:
    """Equal *names* would still permit two implementations. Identity is the claim: one fix to an
    analysis reaches both models because there is one function."""
    mine = trf_run.analysis_registry()

    for name, function in shared_run.ANALYSIS_FUNCTIONS.items():
        assert mine[name] is function


def test_the_shared_registry_carries_none_of_this_packages_additions() -> None:
    """The other direction of the same guarantee. An analysis registered on the shared registry
    reaches every model that uses this pipeline, whether or not the question it answers means
    anything for them."""
    intrusions = set(shared_run.ANALYSIS_FUNCTIONS) & set(TRF_BINDING.extra_analyses)

    assert intrusions == set(), (
        f"{sorted(intrusions)} is registered both as a shared analysis and as this model's own; "
        f"the merge would refuse it, and the shared copy would reach the sibling as well"
    )


def test_the_published_analysis_names_are_the_registrys() -> None:
    """``ANALYSES`` feeds the ``--only`` / ``--skip`` help text. Derived from the registry, so a
    name in the help text that is not registered -- or the reverse -- cannot happen."""
    assert trf_run.ANALYSES == tuple(trf_run.analysis_registry())


# =============================================================================
# The unskippable steps
# =============================================================================
def test_the_unskippable_steps_are_the_siblings_with_no_local_additions() -> None:
    assert dict(trf_run.UNSKIPPABLE_ANALYSES) == dict(shared_run.UNSKIPPABLE_ANALYSES)
    assert trf_run.UNSKIPPABLE_ANALYSES is shared_run.UNSKIPPABLE_ANALYSES


def test_an_unskippable_step_is_never_also_selectable() -> None:
    """``--only band_partition`` must be an error naming the reason, not a silent no-op."""
    overlap = set(trf_run.UNSKIPPABLE_ANALYSES) & set(trf_run.analysis_registry())

    assert overlap == set()


# =============================================================================
# Not vacuous: the comparison has to fail when the registries actually diverge
# =============================================================================
def _binding_with(extra: Dict[str, object]) -> ModelBinding:
    return ModelBinding(
        model_cls=TRF_BINDING.model_cls,
        task_cls=TRF_BINDING.task_cls,
        tag=TRF_BINDING.tag,
        geometry_keys=TRF_BINDING.geometry_keys,
        encoder_disclosure=TRF_BINDING.encoder_disclosure,
        overrides_path=TRF_BINDING.overrides_path,
        extra_analyses=extra,
    )


def test_an_analysis_this_package_added_shows_up_as_a_difference() -> None:
    """The perturbation the real test would catch: a local addition that was never declared."""
    perturbed = set(
        shared_run.merged_analysis_functions(_binding_with({"undeclared": lambda ctx, **kw: {}}))
    )
    expected = set(shared_run.ANALYSIS_FUNCTIONS) | set(TRF_BINDING.extra_analyses)

    assert perturbed != expected
    assert "undeclared" in _difference_message(perturbed, expected)


def test_an_analysis_removed_from_the_sibling_shows_up_as_a_difference(monkeypatch) -> None:
    """And the direction that matters more, because it happens without anyone touching this
    package: an analysis dropped from the shared registry silently stops being measured here."""
    reduced = dict(shared_run.ANALYSIS_FUNCTIONS)
    dropped = reduced.popitem()[0]
    monkeypatch.setattr(shared_run, "ANALYSIS_FUNCTIONS", reduced)

    mine = set(trf_run.analysis_registry())
    expected_before_the_drop = set(reduced) | {dropped} | set(TRF_BINDING.extra_analyses)

    assert mine != expected_before_the_drop
    assert dropped in _difference_message(mine, expected_before_the_drop)


def test_the_registry_is_rebuilt_on_every_call_rather_than_frozen_at_import(monkeypatch) -> None:
    """Which is what makes the case above reach anything: a registry captured at import time
    would report the set as it was when this module was first loaded."""

    def _extra(context, **kwargs):
        return {}

    monkeypatch.setitem(shared_run.ANALYSIS_FUNCTIONS, "added_later", _extra)

    assert "added_later" in trf_run.analysis_registry()


def test_a_name_collision_between_the_two_registries_raises(monkeypatch) -> None:
    """Silently replacing a shared implementation would leave the two models reporting different
    things under one name -- indistinguishable, in the output, from them agreeing."""
    shared_name = next(iter(shared_run.ANALYSIS_FUNCTIONS))

    with pytest.raises(ValueError, match=shared_name):
        shared_run.merged_analysis_functions(
            _binding_with({shared_name: lambda ctx, **kw: {}})
        )
