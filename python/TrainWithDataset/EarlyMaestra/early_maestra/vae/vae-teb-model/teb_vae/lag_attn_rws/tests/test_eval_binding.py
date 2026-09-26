r"""The model-binding seam: what a binding may add, and that adding nothing changes nothing.

The evaluation package takes a :class:`~teb_vae.lag_attn_rws.eval.binding.ModelBinding` rather
than naming one model, so a second architecture can reuse it instead of forking it. That is only
safe if a binding that registers nothing produces exactly what the pipeline produced before the
seam existed, and if a binding that registers something can never silently replace a shared name:
an analysis, a headline scalar or a causality-record key.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from teb_vae.lag_attn_rws.eval import preflight, report_seam, run as run_module
from teb_vae.lag_attn_rws.eval.binding import ModelBinding


# =============================================================================
# The headline block
# =============================================================================
def test_the_headline_block_is_unchanged_when_a_binding_registers_nothing() -> None:
    """The neutrality claim behind the field, stated where the shared default lives: the extras are
    appended, so a binding that adds none produces the block this pipeline produced before the
    parameter existed -- key for key and in the same order."""
    results = {"readouts": {"mc_pred_gap": 1.5}, "verdicts": []}
    without = report_seam.build_headline(results)
    with_empty = report_seam.build_headline(results, run_module.RWS_BINDING.headline_scalars)

    assert with_empty == without
    assert list(with_empty) == list(without)
    # And not vacuous: a registered entry does reach the block.
    extended = report_seam.build_headline(results, (("added", ("readouts", "mc_pred_gap")),))
    assert extended["added"] == 1.5


# =============================================================================
# The merged registry
# =============================================================================
def test_the_merged_registry_is_the_shared_one_when_nothing_is_added() -> None:
    merged = run_module.merged_analysis_functions(run_module.RWS_BINDING)

    assert merged == dict(run_module.ANALYSIS_FUNCTIONS)
    assert list(merged) == list(run_module.ANALYSIS_FUNCTIONS)
    # A copy, so a caller that edits what it was handed cannot reach the shared registry.
    assert merged is not run_module.ANALYSIS_FUNCTIONS


def test_extra_analyses_are_appended_after_the_shared_ones_in_declaration_order() -> None:
    """Appended rather than interleaved: reordering the shared registry to place one model's
    addition would change the *sibling's* run order too."""
    def _first(context, **kwargs):
        return {}

    def _second(context, **kwargs):
        return {}

    merged = run_module.merged_analysis_functions(
        _binding_with({"aaa_first": _first, "zzz_second": _second})
    )

    assert list(merged)[-2:] == ["aaa_first", "zzz_second"]
    assert list(merged)[: len(run_module.ANALYSIS_FUNCTIONS)] == list(run_module.ANALYSIS_FUNCTIONS)


def test_an_extra_analysis_may_not_take_a_shared_name() -> None:
    """Silently replacing a shared implementation would leave two models reporting different
    things under one name, which is indistinguishable in the output from them agreeing."""
    shared = next(iter(run_module.ANALYSIS_FUNCTIONS))

    with pytest.raises(ValueError, match=shared):
        run_module.merged_analysis_functions(_binding_with({shared: lambda context, **kw: {}}))


def test_an_extra_headline_scalar_may_not_take_a_shared_name() -> None:
    """The same rule as the registry's, one seam over. The extras resolve last, so a reused name
    would silently replace a shared reading -- and the headline block is the only thing every arm
    table, the acceptance gate and the cross-model row read, so the substitution would be invisible
    everywhere it mattered."""
    shared_name = report_seam.HEADLINE_SCALARS[0][0]

    with pytest.raises(ValueError, match=shared_name):
        report_seam.build_headline({}, ((shared_name, ("anything", "at", "all")),))


def test_the_encoder_disclosure_may_not_take_a_shared_causality_key() -> None:
    """The disclosure is merged into the middle of the causality record, so a reused key would
    either replace a shared one -- including ``statement``, the refusal sentence that record exists
    to carry -- or be dropped by a key below it. Both are silent in an artifact whose whole purpose
    is to be read literally."""
    model = _DisclosureProbeModel()

    with pytest.raises(ValueError, match="statement"):
        preflight.causality_disclosure(
            {}, model, lambda _model: {"statement": "an encoder's own sentence"}
        )


def test_a_disclosure_of_this_encoders_own_keys_is_still_accepted() -> None:
    """Not vacuous: the guard must refuse only the shared names, not every key a disclosure adds."""
    record = preflight.causality_disclosure(
        {}, _DisclosureProbeModel(), lambda _model: {"an_encoder_only_key": 1}
    )

    assert record["an_encoder_only_key"] == 1
    assert record["statement"] == preflight.NOT_CAUSAL_STATEMENT


class _DisclosureProbeModel:
    """The two attributes :func:`causality_disclosure` reads off a net, and nothing else."""

    horizon = 30
    source_delay_steps = 0


def _binding_with(extra: Dict[str, Any]) -> ModelBinding:
    """The rws binding with a different ``extra_analyses``, for the merge cases above."""
    return ModelBinding(
        model_cls=run_module.RWS_BINDING.model_cls,
        task_cls=run_module.RWS_BINDING.task_cls,
        tag=run_module.RWS_BINDING.tag,
        geometry_keys=run_module.RWS_BINDING.geometry_keys,
        encoder_disclosure=run_module.RWS_BINDING.encoder_disclosure,
        overrides_path=run_module.RWS_BINDING.overrides_path,
        extra_analyses=extra,
    )
