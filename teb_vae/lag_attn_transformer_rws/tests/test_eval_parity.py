r"""This model's analysis registry against the sibling's.

The two models are evaluated by one pipeline so that their runs are readable side by side: the
same ``summary.json`` blocks, the same per-recording CSVs, the same figure families. That holds
only while the two registries agree, so the expected registry is computed from the registries
themselves rather than written down: the sibling's ``ANALYSIS_FUNCTIONS``, in order and by
identity, followed by ``TRF_BINDING.extra_analyses``.

``UNSKIPPABLE_ANALYSES`` describes the **data** rather than the model, so no entry there may also
be selectable. And the IDE launch dict ``RUN_ARGS`` offers exactly the command line's own settings.
"""
from __future__ import annotations

from teb_vae.lag_attn_rws.eval import run as shared_run
from teb_vae.lag_attn_transformer_rws.eval import run as trf_run
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING


def test_the_registry_is_the_siblings_in_order_then_this_models_additions() -> None:
    """Names, run order and implementations at once: ``cross_subgroup`` runs last because it reads
    the per-recording CSVs the analyses above it write, and one fix to a shared analysis reaches
    both models only because there is one function."""
    mine = list(trf_run.analysis_registry().items())
    expected = list(shared_run.ANALYSIS_FUNCTIONS.items()) + list(
        TRF_BINDING.extra_analyses.items()
    )

    assert [name for name, _ in mine] == [name for name, _ in expected]
    for (name, function), (_, reference) in zip(mine, expected):
        assert function is reference, name


def test_an_unskippable_step_is_never_also_selectable() -> None:
    """``--only band_partition`` must be an error naming the reason, not a silent no-op."""
    overlap = set(trf_run.UNSKIPPABLE_ANALYSES) & set(trf_run.analysis_registry())

    assert overlap == set()


def test_the_launch_dict_offers_exactly_the_command_lines_own_settings() -> None:
    """``RUN_ARGS`` is what the IDE Run button launches with. A key here that is not a flag is a
    setting that would appear in no artifact, and a flag missing here cannot be set from the IDE."""
    dests = {
        action.dest for action in trf_run.build_parser()._actions if action.dest != "help"
    }

    assert set(trf_run.RUN_ARGS) == dests
