r"""What this cell's binding declares, and the two ways a declaration can be wrong in silence.

``CFS_BINDING`` is what every call site of this pipeline gets by omission, so an edit to it
changes what a run *means* -- which class is rebuilt, which constructor keys are reconciled
against the checkpoint, which holdout delta is merged. Each field is pinned here against the
literal the code declares, so a change is a decision rather than a diff nobody read.

**The interesting failure is not a wrong value; it is a key that names nothing.**
``preflight.reconcile`` compares ``model_config.VAE_model[key]`` against ``model_kwargs[key]`` and
**silently skips any key absent from either side**. So a ``geometry_keys`` entry that is not both a
constructor parameter *and* a config key is a reconciliation that never happens and never says so
-- the run passes, the config and the checkpoint are free to disagree about that key, and the
symptom appears later as numbers computed at a geometry nobody chose. Both halves are therefore
asserted against the class and against the shipped ``configs/default.yaml`` rather than against a
second hand-kept list.

The four resolved warm-up tuples are the mirror image and are asserted **absent** for it: they are
constructor parameters of no config, so listing them would compare a checkpoint value against
nothing and pass every run. Their guard is a different one, and it re-resolves the budget from the
configured shards instead.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from teb_vae.lag_attn_cfs.eval import binding as binding_module, preflight, report_seam
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING, GEOMETRY_KEYS, ModelBinding
from teb_vae.lag_attn_cfs.eval.config_schema import DEFAULT_OVERRIDES_PATH
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

from .conftest import _REPO_ROOT

#: The constructor keys this cell reconciles against a checkpoint, written out rather than
#: imported: importing :data:`GEOMETRY_KEYS` and comparing it to itself would pass on any edit. An
#: **ordered** sequence, not a set -- the order is what the reconciliation record is built in and
#: what a reader of two runs' preflight files compares down.
#:
#: Fourteen of the nineteen are the raw cells'. Two this cell adds are the ones that decide the
#: population and the lag support: ``anchor_stride`` sets how many anchors a forward scores, and
#: ``lag_floor`` is one of the three quantities the lag-support margin is made of.
#:
#: The last three are architecture switches, and they are here for the same reason the widths are:
#: the evaluation rebuilds the model from the **checkpoint's** own ``model_kwargs``, so a config
#: disagreeing about one of them would not fail -- it would report one architecture's numbers under
#: another's stated description. ``lag_kv_source`` changes what the lag attention reads and
#: therefore what every lag readout means; ``prior_availability_input`` changes what the KL is a
#: divergence between; ``persistence_residual`` changes the predictor every ``nll_*`` and every
#: skill comparison is measured on.
#:
#: ``horizon_weight_halflife_steps`` is deliberately NOT here, on the same ground as the objective
#: weights: it re-weights the *training* criterion's horizon axis and this pipeline scores every
#: block unweighted, so a half-life edited after the fit contradicts no number the run reports.
CFS_GEOMETRY_KEYS = (
    "sequence_length",
    "d_model",
    "d_z",
    "horizon",
    "raw_per_step",
    "warmup_period",
    "c_y",
    "c_u",
    "use_up_st",
    "max_lag",
    "num_heads",
    "d_head",
    "horizon_attention_blocks",
    "causal_norm",
    "anchor_stride",
    "lag_floor",
    "prior_availability_input",
    "lag_kv_source",
    "persistence_residual",
)

#: What the driver resolves the budget into and stamps on the checkpoint. Constructor parameters,
#: none of them a config key, so none of them may appear above.
RESOLVED_WARMUP_KWARGS = (
    "target_keep_index",
    "target_warmup_steps",
    "source_keep_index",
    "source_warmup_steps",
)

#: The analyses this cell registers on its binding, in run order. Written out for the same reason
#: the geometry keys are: importing the registry and comparing it to itself would pass on any edit.
#: ``occlusion`` sits between the two cheap causal readouts and the two that only read tables,
#: which is a cost ordering rather than a dependency: it re-encodes and decodes once per band
#: per batch, and it is the second analysis in this package permitted to reach for the model on
#: the context, because an intervention on the model's INPUT cannot be served by any table a
#: forward already wrote.
CELL_SPECIFIC_ANALYSES = (
    "warmup",
    "source_null",
    "occlusion",
    "lag_clocks",
    "lag_kld_scaled",
    "spectral_skill",
)

#: The headline scalars those analyses add, in registration order. An arm table reads this block by
#: name, so the set of names is a contract with every future comparison rather than a detail of
#: whichever analysis happened to produce them.
CFS_HEADLINE_SCALARS = [
    "kld_source_null_nats",
    "coupling_minus_clock_nats",
    "coupling_minus_clock_ci_lo",
    "coupling_minus_clock_ci_hi",
    "clock_excess_argmax_lag_step",
    "clock_excess_peak_share",
    "clock_excess_degenerate",
    "clock_excess_rectified_frac",
    "pred_gap_warm_lo_nats",
    "pred_gap_warm_mid_nats",
    "pred_gap_warm_hi_nats",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
    "anchors_per_sample",
    "target_warm_frac",
    "spectral_gap_slow_baseline_nats",
    "spectral_gap_deceleration_nats",
    "spectral_gap_variability_nats",
    "spectral_gap_beat_to_beat_nats",
    "spectral_gap_unknown_nats",
    # The interventional readout's own four, and the shape of the block is the finding: a band NAME
    # first, because what the analysis reports is which lag range mattered most rather than a
    # magnitude on a fixed axis; then the delta it cost, the horizon step it peaked at, and the
    # fraction of that band's positions that were live at all. The last is what separates "the
    # source did not matter at those lags" from "there was less source there to remove", and an arm
    # table carrying the delta without it would confuse the two.
    "occlusion_peak_band",
    "occlusion_peak_band_delta_nats",
    "occlusion_peak_band_horizon_step",
    "occlusion_peak_band_live_fraction",
]


@pytest.fixture(scope="module")
def shipped_vae_config() -> Dict[str, Any]:
    """The shipped ``model_config.VAE_model`` block, read off the committed file."""
    shipped = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_cfs" / "configs" / "default.yaml"
    return yaml.safe_load(shipped.read_text(encoding="utf-8"))["model_config"]["VAE_model"]


@pytest.fixture(scope="module")
def constructor_parameters() -> frozenset:
    """Every keyword this cell's constructor accepts."""
    return frozenset(inspect.signature(SeqVaeLagAttnCfs.__init__).parameters)


# =================================================================================================
# The binding's fields
# =================================================================================================
def test_the_binding_names_this_packages_model_and_task() -> None:
    """By name rather than by identity: the assertion should read as the contract does, and an
    import swapped for another class with the same name is a different failure entirely."""
    assert CFS_BINDING.model_cls.__name__ == "SeqVaeLagAttnCfs"
    assert CFS_BINDING.task_cls.__name__ == "SeqVaeLagAttnCfsTask"


def test_the_tag_is_this_cells_own() -> None:
    """``<tag>-eval`` is where a run with no configured tag lands. A tag shared with another cell
    would put two models' runs in one directory, told apart by timestamp alone."""
    assert CFS_BINDING.tag == "lag_attn_cfs"


def test_the_geometry_keys_are_exactly_these_in_this_order() -> None:
    assert CFS_BINDING.geometry_keys == CFS_GEOMETRY_KEYS
    # One constant, one value: the module keeps the tuple and the binding is what carries it into
    # the reconciliation, so the two cannot come apart.
    assert CFS_BINDING.geometry_keys is GEOMETRY_KEYS


def test_every_geometry_key_is_a_parameter_of_this_constructor(constructor_parameters) -> None:
    """A key the constructor does not accept can never match a stamped ``model_kwargs`` entry."""
    unknown = [key for key in GEOMETRY_KEYS if key not in constructor_parameters]

    assert unknown == [], (
        f"{unknown} are not parameters of SeqVaeLagAttnCfs.__init__, so they cannot appear in a "
        f"checkpoint's model_kwargs and the reconciliation would skip them forever"
    )


def test_every_geometry_key_is_also_a_shipped_config_key(shipped_vae_config) -> None:
    """The half that a constructor check alone would miss, and the one that fails silently: the
    reconciliation skips a key absent from *either* side, so a constructor parameter that no config
    names is a key that could only ever be skipped."""
    unconfigured = [key for key in GEOMETRY_KEYS if key not in shipped_vae_config]

    assert unconfigured == [], (
        f"{unconfigured} are not keys of configs/default.yaml's model_config.VAE_model block, so "
        f"reconcile() would skip them on every run and the config could contradict the checkpoint "
        f"about them without a word"
    )


def test_the_warm_up_budget_and_its_four_tuples_are_absent_from_the_geometry_keys(
    shipped_vae_config, constructor_parameters
) -> None:
    r"""The mirror-image rule, and the reason the tuple looks incomplete to a first reader.

    ``causal_warmup_budget_steps`` is a config key and **not** a constructor parameter: the driver
    resolves it against the shards into the four tuples below, and *those* are what land in
    ``model_kwargs``. They are in turn not config keys. So neither half can be reconciled here --
    one would compare against a missing ``model_kwargs`` entry and the other against a missing
    config entry, and both would be skipped in silence. The guard that can actually fail
    re-resolves the budget from the configured shards and compares it with the checkpoint's stamped
    tuples, and it lives in ``preflight``.
    """
    assert "causal_warmup_budget_steps" not in GEOMETRY_KEYS
    assert "causal_warmup_budget_steps" in shipped_vae_config
    assert "causal_warmup_budget_steps" not in constructor_parameters

    for key in RESOLVED_WARMUP_KWARGS:
        assert key not in GEOMETRY_KEYS, key
        assert key in constructor_parameters, key
        assert key not in shipped_vae_config, key

    # And the width the four resolve to, which is a parameter of this constructor and of no config
    # -- and is recoverable from the stamped target_keep_index anyway.
    assert "decoder_out_channels" not in GEOMETRY_KEYS
    assert "decoder_out_channels" in constructor_parameters
    assert "decoder_out_channels" not in shipped_vae_config


def test_causal_norm_is_reconciled_here_and_must_be_dropped_by_the_transformer_cell(
    constructor_parameters,
) -> None:
    """The conv-LSTM encoder accepts ``causal_norm`` and the conv-Transformer one does not, so the
    second cell's binding carries fifteen of these sixteen. Recorded here rather than only there:
    a key copied across that a constructor does not accept is a ``TypeError`` at rebuild, and a key
    quietly dropped is a normaliser nobody reconciles."""
    assert "causal_norm" in GEOMETRY_KEYS
    assert "causal_norm" in constructor_parameters

    transformer = pytest.importorskip("teb_vae.lag_attn_transformer_cfs.nets.model")
    assert "causal_norm" not in inspect.signature(
        transformer.SeqVaeLagAttnTrfCfs.__init__
    ).parameters


def test_the_overrides_path_is_the_committed_delta_and_it_is_there() -> None:
    path = CFS_BINDING.overrides_path

    assert path is DEFAULT_OVERRIDES_PATH
    assert Path(path).is_file()
    assert Path(path).name == "eval_overrides.yaml"
    # Inside *this* package's eval directory, not a sibling's: a binding pointing at another
    # package's delta would evaluate the right checkpoint against the wrong holdout split.
    assert Path(path).parent.parent.name == "eval"
    assert Path(path).parent.parent.parent.name == "lag_attn_cfs"


def test_the_encoder_disclosure_is_this_cells_and_refuses_a_model_it_cannot_read() -> None:
    """Wired to ``preflight.cfs_encoder_disclosure``, which reports the recurrent encoder's own
    guard and nothing that belongs to the target domain -- the one-sidedness, the group delays, the
    warm-up budget, the anchor geometry and the lag support are shared by both cfs cells and are
    owned by the shared half of the causality record, so both models report them down one set of key
    names.

    An object carrying neither attribute raises naming the class rather than disclosing an empty
    block: a causality record that was complete-looking and empty is the one outcome worse than no
    record at all."""
    assert CFS_BINDING.encoder_disclosure is preflight.cfs_encoder_disclosure

    with pytest.raises(AttributeError, match="causal_norm"):
        CFS_BINDING.encoder_disclosure(object())


def test_this_cell_registers_exactly_its_four_own_analyses() -> None:
    """Pinned by name rather than counted, so an analysis that stopped being registered is a
    failure here rather than one that silently never runs -- which is indistinguishable in a
    ``summary.json`` from one that ran and found nothing."""
    assert list(CFS_BINDING.extra_analyses) == list(CELL_SPECIFIC_ANALYSES)
    assert CFS_BINDING.extra_analyses is binding_module.EXTRA_ANALYSES
    assert all(callable(function) for function in CFS_BINDING.extra_analyses.values())


def test_the_four_registered_analyses_are_the_functions_their_modules_define() -> None:
    """A registry entry pointing at something else would run outside the analysis protocol."""
    from teb_vae.lag_attn_cfs.eval.analyses import (
        lag_clocks, source_null, spectral_skill, warmup,
    )

    assert CFS_BINDING.extra_analyses["warmup"] is warmup.run_warmup_analysis
    assert CFS_BINDING.extra_analyses["source_null"] is source_null.run_source_null_analysis
    assert CFS_BINDING.extra_analyses["lag_clocks"] is lag_clocks.run_lag_clocks_analysis
    assert (
        CFS_BINDING.extra_analyses["spectral_skill"]
        is spectral_skill.run_spectral_skill_analysis
    )


def test_the_extras_are_selectable_and_run_before_the_analysis_that_reads_their_tables() -> None:
    """Two properties in one registry, and the second is why the merge is not a plain update.

    ``cross_subgroup`` reads per-recording CSVs off disk and its source table names three of this
    cell's own analyses, so an extra appended *after* it would be tested on every full run and
    found absent every time -- recorded as a partial directory rather than as a run order that
    cannot work.
    """
    registry = run_module.merged_analysis_functions(CFS_BINDING)

    assert set(CELL_SPECIFIC_ANALYSES) <= set(registry)
    assert list(registry)[-1] == "cross_subgroup"
    positions = {name: index for index, name in enumerate(registry)}
    for name in CELL_SPECIFIC_ANALYSES:
        assert positions[name] < positions["cross_subgroup"], name
    # And the shared order above them is untouched, so two cells' summaries still line up.
    shared = [name for name in registry if name in run_module.ANALYSIS_FUNCTIONS]
    assert shared == list(run_module.ANALYSIS_FUNCTIONS)


def test_an_extra_analysis_may_not_take_a_shared_name() -> None:
    """An extra is an addition, never an override: silently replacing a shared implementation
    would leave two models reporting different things under one name."""
    shared_name = next(iter(run_module.ANALYSIS_FUNCTIONS))
    clashing = ModelBinding(
        model_cls=CFS_BINDING.model_cls,
        task_cls=CFS_BINDING.task_cls,
        tag=CFS_BINDING.tag,
        geometry_keys=CFS_BINDING.geometry_keys,
        encoder_disclosure=CFS_BINDING.encoder_disclosure,
        overrides_path=CFS_BINDING.overrides_path,
        extra_analyses={shared_name: lambda *args, **kwargs: {}},
    )

    with pytest.raises(ValueError, match=shared_name):
        run_module.merged_analysis_functions(clashing)


def test_this_cells_headline_scalars_are_the_ones_its_analyses_produce() -> None:
    """Every path in the *shared* headline registry must resolve on a run of every model that uses
    this pipeline, so a scalar produced by an analysis only this cell has cannot go there and has
    to come through here. A number that stays out of the headline stays out of every arm table."""
    assert CFS_BINDING.headline_scalars is binding_module.HEADLINE_SCALARS
    names = [name for name, _ in CFS_BINDING.headline_scalars]

    assert names == CFS_HEADLINE_SCALARS
    assert len(set(names)) == len(names), "a duplicated name would silently shadow itself"


def test_every_registered_headline_path_is_keyed_all_the_way_down() -> None:
    """A path whose last step is a list index resolves to the wrong row the day a metric is added
    above it, and nothing in the artifact would say so. Each of the three analyses assembles a flat
    block of scalars for exactly this reason."""
    for name, path in CFS_BINDING.headline_scalars:
        assert path, name
        assert all(isinstance(step, str) for step in path), (name, path)
        assert path[0] in CELL_SPECIFIC_ANALYSES, (name, path)


def test_every_registered_headline_path_resolves_against_the_blocks_the_analyses_return() -> None:
    """Verifiable before any run exists, on a stub shaped like what the three analyses return. The
    end-to-end fixture asserts the same paths against a real run's results, which is where a path
    that resolves on a stub and not in reality would fail."""
    results = {
        "warmup": {
            "headline": {
                "pred_gap_warm_lo_nats": 0.1,
                "pred_gap_warm_mid_nats": 0.2,
                "pred_gap_warm_hi_nats": 0.3,
                "source_lag_warmth_frac_st": 0.4,
                "source_lag_warmth_frac_ph": 0.05,
            },
            "geometry_guards": {"anchors_per_sample": 152.0, "target_warm_frac": 1.0},
        },
        "source_null": {
            "difference": {
                "kld_source_null_nats": 1.25,
                "coupling_minus_clock_nats": 1.75,
                "ci_lo": 1.1,
                "ci_hi": 2.4,
            },
            # The same difference resolved by lag. ``clock_excess_degenerate`` is a BOOL and the
            # stub carries it as one deliberately: the headline finiteness check exempts bools
            # explicitly, so a builder that coerced it to a float would turn "this profile has no
            # readable shape" into a 0.0 that reads as a measured share.
            "lag": {
                "clock_excess_argmax_lag_step": 33,
                "clock_excess_peak_share": 0.42,
                "clock_excess_degenerate": False,
                "clock_excess_rectified_frac": 0.07,
            },
        },
        "spectral_skill": {
            "headline": {
                f"pred_gap_{band}_nats": 0.01
                for band in ("slow_baseline", "deceleration", "variability",
                             "beat_to_beat", "unknown")
            }
        },
        # The interventional readout's block. Its first entry is a band NAME rather than a number,
        # which the stub carries deliberately: the headline builder must pass a string through
        # untouched, and a builder that coerced every value to a float would turn the one entry
        # saying WHICH lag range mattered into a NaN and resolve it as absent.
        "occlusion": {
            "headline": {
                "band": "near",
                "delta_total_nats": 14.94,
                "peak_horizon_step": 5,
                "live_fraction": 1.0,
                "n_bands": 4,
            }
        },
        "verdicts": [],
    }

    headline = report_seam.build_headline(results, CFS_BINDING.headline_scalars)

    unresolved = sorted(
        name for name, _ in CFS_BINDING.headline_scalars if headline.get(name) is None
    )
    assert unresolved == []
    assert headline["coupling_minus_clock_nats"] == 1.75
    assert headline["anchors_per_sample"] == 152.0
    assert headline["occlusion_peak_band"] == "near"
    assert headline["clock_excess_argmax_lag_step"] == 33
    # Passed through as a bool rather than coerced: the finiteness check exempts bools, and a
    # degenerate profile reported as 0.0 would read as a share that was measured.
    assert headline["clock_excess_degenerate"] is False


def test_the_headline_block_is_unchanged_when_a_binding_registers_nothing() -> None:
    """The neutrality claim behind the field: the extras are appended, so a binding that adds none
    produces the block this pipeline produces without them -- key for key and in the same order.

    Checked against an empty tuple rather than against this cell's binding, which registers
    sixteen: what is being asserted is that the *mechanism* adds nothing of its own, and the test
    below asserts that this cell's sixteen are appended after the shared block rather than mixed
    into it.
    """
    results = {"readouts": {"mc_pred_gap": 1.5}, "verdicts": []}

    without = report_seam.build_headline(results)
    with_empty = report_seam.build_headline(results, ())

    assert with_empty == without
    assert list(with_empty) == list(without)
    # And not vacuous: a registered entry does reach the block.
    extended = report_seam.build_headline(results, (("added", ("readouts", "mc_pred_gap")),))
    assert extended["added"] == 1.5


def test_this_cells_scalars_are_appended_after_the_shared_block_rather_than_mixed_in() -> None:
    """The shared block's key order is what two cells' summaries are diffed down, so an extra that
    landed inside it would shift every row below it in a comparison that is read by position."""
    results = {"readouts": {"mc_pred_gap": 1.5}, "verdicts": []}

    headline = list(report_seam.build_headline(results, CFS_BINDING.headline_scalars))
    shared = [name for name, _ in report_seam.HEADLINE_SCALARS]

    assert headline[: len(shared)] == shared
    assert headline[len(shared) : len(shared) + len(CFS_HEADLINE_SCALARS)] == CFS_HEADLINE_SCALARS


def test_an_extra_headline_scalar_may_not_take_a_shared_name() -> None:
    """The extras resolve last, so a reused name would replace a shared reading with a
    cell-specific one under the shared name -- and every arm table, the acceptance gate and every
    cross-cell row reads this block *by name*, so the substitution would be invisible in the
    artifact."""
    shared_name = report_seam.HEADLINE_SCALARS[0][0]

    with pytest.raises(ValueError, match=shared_name):
        report_seam.build_headline({}, ((shared_name, ("anything", "at", "all")),))


def test_the_binding_cannot_be_edited_after_it_is_declared() -> None:
    """A binding is a declaration, not a setting: a field mutated mid-run would leave a summary
    describing a contract that was not in force when the tables were collected."""
    with pytest.raises(Exception):
        CFS_BINDING.tag = "something_else"  # type: ignore[misc]


def test_the_dataclass_itself_still_names_no_model() -> None:
    """The type is the sibling's, field for field. This package adds a concrete instance beside it
    and changes nothing about the seam, so a third cell can be bound by declaring one more."""
    fields = {name: parameter.annotation for name, parameter in
              inspect.signature(ModelBinding.__init__).parameters.items() if name != "self"}

    assert list(fields) == [
        "model_cls", "task_cls", "tag", "geometry_keys", "encoder_disclosure", "overrides_path",
        "extra_analyses", "headline_scalars",
    ]
