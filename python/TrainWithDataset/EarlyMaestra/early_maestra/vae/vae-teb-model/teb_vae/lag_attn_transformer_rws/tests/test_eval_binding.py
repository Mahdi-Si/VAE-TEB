r"""What this model's binding declares, and what the shared pipeline does with each field.

The binding is the whole of this package's coupling to the shared evaluation pipeline: four facts
that pipeline cannot derive, and nothing else. Each field decides what a run's numbers *mean* --
which class is rebuilt from a checkpoint, which constructor keys are reconciled against it, what
the encoder discloses about its own causal standing, which holdout split is merged in -- so each is
pinned here rather than left to be read off the code it configures.

Two of them get more than a pin.

**The geometry keys**, because reconciliation is the only guard between a config that contradicts
the weights and a run that reports one model's geometry beside another's numbers. The architecture
is rebuilt from the checkpoint's own ``model_kwargs``, so the checkpoint always wins; a key missing
from this tuple is a key the config may contradict in silence. Every one of the seven encoder keys
therefore has its own refusal case.

**``source_attention_window``**, because its ``null`` is a *value*. An unbounded source encoder
**is** ``source_attention_window: null`` -- it is the arm the whole locality sweep is measured
against -- rather than "use the constructor default", and a reconciliation that skipped it as
absent would pass an unbounded checkpoint against a config declaring a 16-step window and report
the sweep's baseline under the unbounded arm's name.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from teb_vae.lag_attn_rws.eval import preflight, run as shared_run
from teb_vae.lag_attn_rws.eval.binding import ModelBinding
from teb_vae.lag_attn_transformer_rws.eval import binding as binding_module
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING, trf_encoder_disclosure
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import SHIPPED_KWARGS, TINY_KWARGS

#: The constructor keys this model reconciles, written out rather than imported: comparing the
#: module's tuple against itself would pass on any edit. An **ordered** sequence, because that is
#: the order the ``compared`` record is built in and the order two runs' preflight files are read
#: down.
TRF_GEOMETRY_KEYS = (
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
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: The seven this architecture adds. Each changes what the numbers mean: the stem schedule and the
#: block counts set how much history a state summarises, the head count and feed-forward width the
#: capacity behind it, and the window the source encoder's reach.
ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: A disagreeing value per encoder key, of the right kind: a tuple key needs a tuple, or the
#: refusal would be a type error dressed up as a geometry disagreement.
DISAGREEING_VALUES: Dict[str, Any] = {
    "encoder_conv_kernels": (7, 7),
    "encoder_conv_dilations": (1, 4),
    "encoder_num_heads": 2,
    "encoder_d_ff": 128,
    "target_attention_blocks": 5,
    "source_attention_blocks": 1,
    "source_attention_window": 32,
}


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfRws(**TINY_KWARGS)


def _config(**vae_overrides: Any) -> Dict[str, Any]:
    """A merged-config shape carrying only what the reconciliation reads."""
    return {"model_config": {"VAE_model": dict(vae_overrides)}}


def _reconcile(config: Dict[str, Any], model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Reconcile through this model's binding, as a run does."""
    return preflight.reconcile_with_checkpoint(
        config,
        model_kwargs=model_kwargs,
        hyper_parameters={},
        geometry_keys=TRF_BINDING.geometry_keys,
    )


# =============================================================================
# The binding's fields
# =============================================================================
def test_the_binding_names_this_packages_model_and_task() -> None:
    assert TRF_BINDING.model_cls.__name__ == "SeqVaeLagAttnTrfRws"
    assert TRF_BINDING.task_cls.__name__ == "SeqVaeLagAttnTrfRwsTask"


def test_the_tag_is_this_models_own() -> None:
    """``<tag>-eval`` is where a run with no configured tag lands. Sharing the sibling's would put
    two models' runs in one directory, told apart only by timestamp."""
    assert TRF_BINDING.tag == "lag_attn_trf_rws"
    assert TRF_BINDING.tag != shared_run.RWS_BINDING.tag


def test_the_geometry_keys_are_exactly_these_in_this_order() -> None:
    assert TRF_BINDING.geometry_keys == TRF_GEOMETRY_KEYS


def test_the_geometry_keys_drop_causal_norm_and_add_the_seven_encoder_keys() -> None:
    """Stated as a difference against the sibling's, because that is the claim: everything the two
    models share is reconciled the same way, and the divergence is exactly the encoders."""
    sibling = set(shared_run.RWS_BINDING.geometry_keys)
    mine = set(TRF_BINDING.geometry_keys)

    assert sibling - mine == {"causal_norm"}
    assert mine - sibling == set(ENCODER_KEYS)


def test_a_checkpoint_predating_the_horizon_attention_still_reconciles() -> None:
    """The reconciliation skips a key either side lacks, which is what lets a run trained before
    a knob existed stay evaluable against a config that now carries it. Asserted rather than
    inferred: the alternative -- refusing on a key the checkpoint could not have recorded -- would
    strand every checkpoint written before this revision.
    """
    older = {key: value for key, value in TINY_KWARGS.items()}
    older.pop("horizon_attention_blocks", None)

    record = _reconcile(_config(horizon_attention_blocks=2), older)

    assert record["passed"] is True
    assert "horizon_attention_blocks" not in record["compared"]


def test_a_disagreeing_horizon_attention_depth_is_refused() -> None:
    """The other direction, and the reason the key is in the tuple at all: two blocks of decoder
    attention are two blocks of capacity, so a config claiming them over a checkpoint trained
    without them describes a model that was never fitted."""
    with pytest.raises(preflight.EvalPreconditionUnmet, match="horizon_attention_blocks"):
        _reconcile(
            _config(horizon_attention_blocks=2),
            dict(TINY_KWARGS, horizon_attention_blocks=0),
        )


def test_causal_norm_is_not_merely_irrelevant_here_but_unconstructable() -> None:
    """Why it is dropped rather than reconciled against a constant: the constructor refuses it, so
    a config carrying it fails at rebuild and reconciling it could only compare against nothing."""
    with pytest.raises(TypeError, match="causal_norm"):
        SeqVaeLagAttnTrfRws(**dict(TINY_KWARGS, causal_norm=True))


def test_the_overrides_path_is_this_packages_delta_and_it_is_there() -> None:
    path = Path(TRF_BINDING.overrides_path)

    assert path.is_file()
    assert path.name == "eval_overrides.yaml"
    assert path.parents[1].name == "eval"
    assert path.parents[2].name == "lag_attn_transformer_rws"
    assert path != Path(shared_run.RWS_BINDING.overrides_path)


def test_this_model_registers_exactly_one_analysis_of_its_own() -> None:
    """One, and it is the one the encoder replacement makes askable. Every other readout comes
    from the shared registry, so this model's ``summary.json`` and the sibling's carry the same
    blocks and are readable side by side -- and an addition here is a *strict* addition rather
    than a second implementation of something shared, which is what the merge refuses."""
    from teb_vae.lag_attn_transformer_rws.eval.analyses import encoder_attention

    assert set(TRF_BINDING.extra_analyses) == {"encoder_attention"}
    assert (
        TRF_BINDING.extra_analyses["encoder_attention"]
        is encoder_attention.run_encoder_attention_analysis
    )


def test_the_headline_scalars_are_this_analysis_own_and_collide_with_no_shared_name() -> None:
    """The six the arm tables read. Appended to the shared registry rather than added to it: every
    path in *that* tuple has to resolve on a run of every model, so an entry there would read as a
    number the sibling failed to produce rather than as one it cannot have."""
    from teb_vae.lag_attn_rws.eval import report_seam
    from teb_vae.lag_attn_transformer_rws.eval.analyses import encoder_attention

    registered = dict(TRF_BINDING.headline_scalars)

    assert set(registered) == {
        f"encoder_attention_{name}" for name in encoder_attention.HEADLINE_KEYS
    }
    assert set(registered) & {name for name, _ in report_seam.HEADLINE_SCALARS} == set()
    for name, path in registered.items():
        assert path[0] == encoder_attention.ANALYSIS_DIRNAME
        assert path[1] == "headline"
        assert name.endswith(path[2])


def test_the_binding_is_a_declaration_rather_than_a_setting() -> None:
    with pytest.raises(Exception):
        TRF_BINDING.tag = "something_else"  # type: ignore[misc]


def test_the_binding_is_the_shared_type_rather_than_a_look_alike() -> None:
    """A structurally-similar local class would drift from the shared one silently the first time
    a field was added."""
    assert isinstance(TRF_BINDING, ModelBinding)


# =============================================================================
# The encoder disclosure
# =============================================================================
def test_the_disclosure_carries_no_key_that_means_nothing_here(tiny_model) -> None:
    """``causal_norm`` and ``n_causalized_norms`` describe a time-pooling ``GroupNorm`` this
    architecture bans structurally. Reported anyway they would read as a setting someone could
    change, and a reader comparing two models' records would compare a real number against a
    placeholder."""
    record = trf_encoder_disclosure(tiny_model)

    assert "causal_norm" not in record
    assert "n_causalized_norms" not in record
    assert "causal_norm_consequence" not in record


def test_the_disclosure_reports_what_is_true_of_these_encoders(tiny_model) -> None:
    record = trf_encoder_disclosure(tiny_model)

    assert record["time_pooling_normalisers"] == 0
    assert record["time_pooling_normalisers_are_structural"] is True
    assert "test_no_time_pooling_normaliser_on_either_history_path" in record[
        "time_pooling_normalisers_proved_by"
    ]
    assert record["n_depthwise_init"] == int(tiny_model.n_depthwise_init)
    assert record["target_attention_blocks"] == TINY_KWARGS["target_attention_blocks"]
    assert record["source_attention_blocks"] == TINY_KWARGS["source_attention_blocks"]
    assert record["source_attention_window"] == TINY_KWARGS["source_attention_window"]


def test_the_named_test_exists_and_is_about_what_the_record_says_it_is() -> None:
    """A record pointing at a test nobody can find is a claim with no evidence behind it."""
    reference = trf_encoder_disclosure.__doc__ or ""
    assert reference  # the docstring is where the reasoning lives; a bare dict is not a disclosure

    record_path, _, test_name = (
        binding_module.trf_encoder_disclosure(SeqVaeLagAttnTrfRws(**TINY_KWARGS))[
            "time_pooling_normalisers_proved_by"
        ]
    ).partition("::")
    source = (Path(__file__).resolve().parents[3] / record_path).read_text(encoding="utf-8")

    assert f"def {test_name}(" in source


def test_the_source_reach_is_stated_against_the_lag_range_with_which_is_larger(tiny_model) -> None:
    r"""The comparison is the point rather than the two numbers. An encoder whose reach exceeded
    the lag range would already be doing the alignment the lag cross-attention exists to do, and a
    reader should not have to divide by $\Delta$ to find that out."""
    record = trf_encoder_disclosure(tiny_model)
    reach = record["source_receptive_field_steps"]

    # Tiny geometry: stem reach 1 + (3-1)*1 + (3-1)*2 = 7, plus 2 blocks * (4-1) = 13 steps.
    assert reach == 13
    assert record["source_receptive_field_seconds"] == reach * 4.0
    assert record["lag_range_max_steps"] == TINY_KWARGS["max_lag"]
    assert record["lag_range_max_seconds"] == TINY_KWARGS["max_lag"] * 4.0
    assert record["n_lags"] == TINY_KWARGS["max_lag"] + 1
    assert "is larger" in record["source_reach_vs_lag_range"]
    assert str(reach) in record["source_reach_vs_lag_range"]
    assert str(TINY_KWARGS["max_lag"]) in record["source_reach_vs_lag_range"]


def test_the_shipped_geometry_keeps_the_reach_inside_the_lag_range() -> None:
    r"""The architectural claim, at the geometry that actually trains: $R_U = 66$ steps against a
    furthest searched lag of $90$. A source encoder reaching past the lag range would make the lag
    attention's job redundant, and the sweep would be measuring nothing."""
    torch.manual_seed(0)
    record = trf_encoder_disclosure(SeqVaeLagAttnTrfRws(**SHIPPED_KWARGS))

    assert record["source_receptive_field_steps"] == 66
    assert record["source_receptive_field_seconds"] == 264.0
    assert record["lag_range_max_steps"] == 90
    assert record["source_reach_is_inside_the_lag_range"] is True
    assert "the lag range is larger" in record["source_reach_vs_lag_range"]


def test_a_reach_equal_to_the_lag_range_says_so_rather_than_claiming_one_is_larger() -> None:
    r"""The reach and the lag range are configured independently, so a sweep arm can land them on
    the same number. A two-way comparison would then print two identical figures and assert that
    one of them is larger, which a reader has to disbelieve to read the record correctly."""
    torch.manual_seed(0)
    # The tiny geometry's reach is 13 steps; ask for exactly that many lags.
    matched = SeqVaeLagAttnTrfRws(**dict(TINY_KWARGS, max_lag=13))

    record = trf_encoder_disclosure(matched)

    assert record["source_receptive_field_steps"] == record["lag_range_max_steps"] == 13
    assert "they are equal" in record["source_reach_vs_lag_range"]
    assert "is larger" not in record["source_reach_vs_lag_range"]
    # Equal is not inside: a reach that matches the furthest searched lag is not bounded below it.
    assert record["source_reach_is_inside_the_lag_range"] is False


def test_the_unbounded_arm_reports_an_absent_bound_rather_than_the_sequence_length() -> None:
    """"No bound" and "a bound that happens to equal $T$" are different statements, and the arm
    the locality sweep is measured against is the first one."""
    torch.manual_seed(0)
    unbounded = SeqVaeLagAttnTrfRws(**dict(TINY_KWARGS, source_attention_window=None))

    record = trf_encoder_disclosure(unbounded)

    assert record["source_attention_window"] is None
    assert record["source_receptive_field_steps"] is None
    assert record["source_receptive_field_seconds"] is None
    assert "no window" in record["source_reach_vs_lag_range"]
    assert record["source_receptive_field_steps"] != TINY_KWARGS["sequence_length"]
    # No verdict either: there is no bound, so there is nothing for it to be inside of.
    assert "source_reach_is_inside_the_lag_range" not in record


def test_a_missing_attribute_raises_naming_it_rather_than_reporting_nothing() -> None:
    """A silent ``getattr`` default would report a model that stopped exposing something as a
    model with nothing to report -- and the disclosure would go quiet in exactly the case a reader
    most needs to be told."""

    class _Renamed:
        pass

    with pytest.raises(AttributeError, match="target_encoder"):
        trf_encoder_disclosure(_Renamed())


def test_the_disclosure_is_what_the_binding_carries() -> None:
    assert TRF_BINDING.encoder_disclosure is trf_encoder_disclosure


def test_the_shared_half_of_the_record_is_unchanged(tiny_model) -> None:
    """The bank-side half -- the refusal sentence, the channel reaches, the source delay, the
    horizon -- describes the *dataset*, so it is identical for both models and comes from the
    shared function. Only the encoder block differs."""
    record = preflight.causality_disclosure(
        _config(causal_reach_budget_s=None), tiny_model, trf_encoder_disclosure
    )

    assert record["statement"] == preflight.NOT_CAUSAL_STATEMENT
    assert record["not_causal"] is True
    assert record["channels_reading_past_the_horizon"]
    assert record["time_pooling_normalisers"] == 0


# =============================================================================
# Reconciliation: every encoder key, and the one whose null is a value
# =============================================================================
@pytest.mark.parametrize("key", ENCODER_KEYS)
def test_a_config_contradicting_the_checkpoint_on_an_encoder_key_is_refused(key) -> None:
    """Each of the seven, individually. The architecture is rebuilt from the checkpoint's own
    ``model_kwargs``, so a disagreement the reconciliation missed would report the config's value
    beside numbers the checkpoint's value produced."""
    checkpoint_kwargs = dict(TINY_KWARGS)
    config = _config(**{key: DISAGREEING_VALUES[key]})

    with pytest.raises(preflight.EvalPreconditionUnmet) as excinfo:
        _reconcile(config, checkpoint_kwargs)

    message = str(excinfo.value)
    assert key in message
    assert repr(DISAGREEING_VALUES[key]) in message, "the config's value must be named"
    assert repr(checkpoint_kwargs[key]) in message, "the checkpoint's value must be named"


@pytest.mark.parametrize("key", ENCODER_KEYS)
def test_an_agreeing_encoder_key_is_compared_rather_than_skipped(key) -> None:
    """The other half: a key that passes must appear in the ``compared`` record, or "reconciled"
    and "never looked at" would be indistinguishable in the artifact."""
    record = _reconcile(_config(**{key: TINY_KWARGS[key]}), dict(TINY_KWARGS))

    assert record["passed"] is True
    assert key in record["compared"]
    assert record["compared"][key]["config"] == TINY_KWARGS[key]


def test_an_unbounded_checkpoint_passes_a_config_that_declares_null() -> None:
    """``null`` is a value, and this is the direction that must not be skipped as absent."""
    record = _reconcile(
        _config(source_attention_window=None),
        dict(TINY_KWARGS, source_attention_window=None),
    )

    assert record["passed"] is True
    assert "source_attention_window" in record["compared"]
    assert record["compared"]["source_attention_window"]["checkpoint"] is None


def test_an_unbounded_checkpoint_refuses_a_config_that_declares_a_window() -> None:
    """The failure the whole ``NULLABLE_MODEL_KEYS`` mechanism exists to make visible: the
    unbounded arm evaluated under the baseline's configured window."""
    with pytest.raises(preflight.EvalPreconditionUnmet, match="source_attention_window"):
        _reconcile(
            _config(source_attention_window=16),
            dict(TINY_KWARGS, source_attention_window=None),
        )


def test_a_windowed_checkpoint_refuses_a_config_that_declares_null() -> None:
    """And the reverse, because a null that meant "unset" would silently pass here."""
    with pytest.raises(preflight.EvalPreconditionUnmet, match="source_attention_window"):
        _reconcile(
            _config(source_attention_window=None),
            dict(TINY_KWARGS, source_attention_window=16),
        )


def test_the_trainer_keeps_the_null_window_in_model_kwargs(tmp_path) -> None:
    """The reconciliation can only compare a key the checkpoint carries. The inherited config
    sweep drops every ``null``; this architecture's driver re-admits this one, and without that
    the two cases above would both pass by the key being absent."""
    import yaml

    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_transformer_rws.tests.conftest import absolutize_dataset_paths
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    repo_root = Path(__file__).resolve().parents[3]
    tiny = repo_root / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "tiny.yaml"
    config = absolutize_dataset_paths(load_config(str(tiny)))
    config = copy.deepcopy(config)
    config["model_config"]["VAE_model"]["source_attention_window"] = None
    config_path = tmp_path / "unbounded.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    model_kwargs = LagAttnTrfRwsTrainer(config_file_path=str(config_path))._build_model_kwargs()

    assert "source_attention_window" in model_kwargs
    assert model_kwargs["source_attention_window"] is None
