r"""What this model's binding declares, and what the shared pipeline does with each field.

The binding is the whole of this package's coupling to the shared evaluation pipeline. What is
tested here is what it changes about a run: which constructor keys are reconciled against a
checkpoint, and what the encoder discloses about its own causal standing.

**The geometry keys**, because reconciliation is the only guard between a config that contradicts
the weights and a run that reports one model's geometry beside another's numbers. The architecture
is rebuilt from the checkpoint's own ``model_kwargs``, so the checkpoint always wins; a key missing
from this tuple is a key the config may contradict in silence. The set is checked against the
sibling's -- the divergence is exactly the encoders -- and every encoder key has its own refusal
case.

**The encoder disclosure**, because it states the source encoder's reach against the lag range,
and the arm the locality sweep is measured against is the one with no bound at all.

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
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING, trf_encoder_disclosure
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import SHIPPED_KWARGS, TINY_KWARGS

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
def test_the_tag_is_this_models_own() -> None:
    """``<tag>-eval`` is where a run with no configured tag lands. Sharing the sibling's would put
    two models' runs in one directory, told apart only by timestamp."""
    assert TRF_BINDING.tag != shared_run.RWS_BINDING.tag


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


# =============================================================================
# The encoder disclosure
# =============================================================================
def test_the_disclosure_reports_what_is_true_of_these_encoders(tiny_model) -> None:
    """Read off the built model. ``causal_norm`` and ``n_causalized_norms`` describe a time-pooling
    ``GroupNorm`` this architecture bans structurally, so reported anyway they would read as a
    setting someone could change."""
    record = trf_encoder_disclosure(tiny_model)

    assert "causal_norm" not in record and "n_causalized_norms" not in record
    assert record["time_pooling_normalisers"] == 0
    assert record["n_depthwise_init"] == int(tiny_model.n_depthwise_init)
    assert record["target_attention_blocks"] == TINY_KWARGS["target_attention_blocks"]
    assert record["source_attention_blocks"] == TINY_KWARGS["source_attention_blocks"]
    assert record["source_attention_window"] == TINY_KWARGS["source_attention_window"]


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
    r"""The architectural claim, at the geometry that actually trains: $R_U$ below the furthest
    searched lag. A source encoder reaching past the lag range would make the lag attention's job
    redundant, and the sweep would be measuring nothing."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**SHIPPED_KWARGS)
    record = trf_encoder_disclosure(model)

    assert record["source_receptive_field_steps"] == model.source_encoder.receptive_field
    assert record["lag_range_max_steps"] == SHIPPED_KWARGS["max_lag"]
    assert record["source_receptive_field_steps"] < record["lag_range_max_steps"]
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


def test_an_unbounded_checkpoint_passes_a_config_that_declares_null() -> None:
    """``null`` is a value, and this is the direction that must not be skipped as absent."""
    record = _reconcile(
        _config(source_attention_window=None),
        dict(TINY_KWARGS, source_attention_window=None),
    )

    assert record["passed"] is True
    assert "source_attention_window" in record["compared"]
    assert record["compared"]["source_attention_window"]["checkpoint"] is None


@pytest.mark.parametrize(
    "config_window, checkpoint_window",
    [(16, None), (None, 16)],
    ids=["unbounded-checkpoint", "windowed-checkpoint"],
)
def test_a_null_window_disagreeing_with_the_checkpoint_is_refused(
    config_window, checkpoint_window
) -> None:
    """Both directions. The unbounded arm evaluated under the baseline's configured window is the
    failure ``NULLABLE_MODEL_KEYS`` exists to make visible, and the reverse is what a null that
    meant "unset" would silently pass."""
    with pytest.raises(preflight.EvalPreconditionUnmet, match="source_attention_window"):
        _reconcile(
            _config(source_attention_window=config_window),
            dict(TINY_KWARGS, source_attention_window=checkpoint_window),
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
