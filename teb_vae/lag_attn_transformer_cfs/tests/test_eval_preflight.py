r"""The causal-feature guards, proved against **this** cell before anything else is built on them.

The guards themselves belong to :mod:`teb_vae.lag_attn_cfs.eval.preflight` and are tested there
against the conv-LSTM cell. What is tested here is the one thing that test cannot reach: that they
hold when driven through *this* binding, on *this* architecture.

That is worth a file of its own for a precise reason. Everything the causal-feature evaluation
pipeline does is validated against one cell, and this binding is a set of declarations that can be
wrong in exactly one interesting way -- a key that names nothing, or an encoder disclosure that reads
an attribute this net does not carry. Both fail here, cheaply, rather than after the whole pipeline
has been validated against the other cell alone.

Two refusals and one pass, all against the committed causal fixture: a two-sided shard is refused, a
well-formed run passes every check, and the causality record carries this encoder's own disclosure
and none of the conv-LSTM cell's keys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval.preflight import EvalPreconditionUnmet
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

from .conftest import (
    CAUSAL_SHARD,
    SHIPPED_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    causal_config,
    shipped_warmup_kwargs,
)

#: The committed statistics file accumulated from the causal fixture, at ``trim_minutes: 1.0``.
CAUSAL_STATS = Path(CAUSAL_SHARD).parent / "tiny_stats_causal.hdf5"

#: What the shipped delta asks the loader for, and the objective a checkpoint would stamp. Both are
#: the conv-LSTM cell's exactly, which is the point: the two cells are compared against each other,
#: so a guard that passed on one configuration and not the other would be a difference of protocol
#: rather than of architecture.
EVAL_LOAD_FIELDS = [
    "fhr", "up", "fhr_st", "fhr_ph", "up_ph", "up_st", "weight", "guid", "epoch", "target",
    "cs_label", "bg_label", "time_from_labor_onset",
]
HYPER_PARAMETERS: Dict[str, Any] = {
    "likelihood": "gaussian_nll",
    "free_bits": 0.0,
    "lambda_full": 1.0,
    "lambda_base": 1.0,
    "kld_beta": 1.0,
    "beta_schedule": None,
}


def eval_config() -> Dict[str, Any]:
    """Build the config an evaluation of this cell actually preflights.

    Returns:
        A fresh config dict, safe to mutate.
    """
    config = causal_config()
    config["model_config"]["VAE_model"].update(HYPER_PARAMETERS)
    dataset = config["dataset_config"]
    dataset["stat_path"] = str(CAUSAL_STATS)
    dataloader = dataset["dataloader_config"]
    dataloader["normalize_fields"] = ["fhr_st", "fhr_ph", "up_st", "up_ph"]
    dataloader["dataset_kwargs"]["load_fields"] = list(EVAL_LOAD_FIELDS)
    return config


@pytest.fixture(scope="module")
def model_kwargs() -> Dict[str, Any]:
    """The constructor kwargs a checkpoint of this cell trained on this fixture would stamp."""
    return shipped_warmup_kwargs()


@pytest.fixture(scope="module")
def model(model_kwargs) -> Any:
    """A conv-Transformer cfs net at the shipped geometry, perturbed so the load witness passes.

    Perturbing the posterior head stands in for a checkpoint load in the one respect this file
    checks: the delta heads are zero at construction, so a model whose weights never moved is exactly
    what the weight-space check refuses. The refusal itself is exercised in the conv-LSTM cell's own
    suite, against the same witness set -- this architecture shares the posterior head, the FiLM
    generators and the horizon attention with it.
    """
    torch.manual_seed(0)
    built = SeqVaeLagAttnTrfCfs(**model_kwargs)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in built.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return built


@pytest.fixture
def config() -> Dict[str, Any]:
    """A fresh evaluation config over the committed causal fixture (safe to mutate)."""
    return eval_config()


def _run(config: Dict[str, Any], model: Any, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Run preflight over one config against one model, through **this** cell's binding."""
    return preflight.run_preflight(
        config=config,
        model=model,
        checkpoint_path="<in-memory>",
        model_kwargs=model_kwargs,
        hyper_parameters=HYPER_PARAMETERS,
        binding=TRF_CFS_BINDING,
    )


def test_a_well_formed_run_of_this_cell_passes_every_check(config, model, model_kwargs) -> None:
    """The guards are the conv-LSTM cell's; what this asserts is that driving them through this
    binding -- its geometry keys, its model class, its encoder disclosure -- refuses nothing."""
    record = _run(config, model, model_kwargs)

    assert all(check["passed"] for check in record["checks"].values())
    assert record["checks"]["warmup_budget_matches_checkpoint"]["target_kept_width"] == 98


def test_a_two_sided_shard_is_refused_for_this_cell_too(config, model, model_kwargs) -> None:
    """The refusal is the target domain's rather than the encoder's, so it must hold identically on
    both cells -- and a binding is exactly the sort of seam through which a guard quietly stops
    running for one of them."""
    config["dataset_config"]["vae_test_datasets"] = [str(TWO_SIDED_SHARD)]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    assert "share every field name and every dtype" in str(excinfo.value)


def test_this_bindings_geometry_keys_are_the_ones_actually_reconciled(
    config, model, model_kwargs
) -> None:
    """The reconciliation record is built from the binding's tuple, so a key it does not carry is a
    key this run never checked -- and the artifact says which, rather than passing
    indistinguishably from a run that checked everything."""
    compared = _run(config, model, model_kwargs)["checks"]["config_matches_checkpoint"]["compared"]

    assert "causal_norm" not in compared, "not a keyword of this constructor at all"
    assert "anchor_stride" in compared and "warmup_period" in compared
    assert set(compared) <= set(TRF_CFS_BINDING.geometry_keys) | set(preflight.OBJECTIVE_KEYS)


def test_a_config_contradicting_an_encoder_key_is_refused(config, model, model_kwargs) -> None:
    """Non-vacuity for the seven keys this architecture adds: each changes what the numbers mean, and
    a tuple that merely listed them without them being config keys would reconcile nothing."""
    config["model_config"]["VAE_model"]["source_attention_window"] = (
        int(model_kwargs["source_attention_window"]) + 1
    )

    with pytest.raises(EvalPreconditionUnmet, match="source_attention_window"):
        _run(config, model, model_kwargs)


def test_the_causality_record_carries_this_encoders_disclosure_and_not_the_other_cells(
    config, model, model_kwargs
) -> None:
    """Two honest blocks rather than one shared key that means nothing in one of them -- while
    everything about the target domain stays identical, because both cells compose the same two
    mixins over the same data."""
    causality = _run(config, model, model_kwargs)["causality"]

    assert causality["time_pooling_normalisers"] == 0
    assert causality["time_pooling_normalisers_are_structural"] is True
    assert "causal_norm" not in causality
    assert "n_causalized_norms" not in causality

    # The shared half, identical to the conv-LSTM cell's because it describes the data and the
    # target-domain mixins rather than either encoder.
    assert causality["statement"] == preflight.CAUSALITY_STATEMENT
    assert causality["one_sided_inputs"] is True
    assert causality["transform"] == "causal"
    assert causality["anchor_geometry"]["anchors_per_sample"] == 137
    assert causality["lag_support"]["min_decoded_anchor"] == SHIPPED_WARMUP_PERIOD
    assert causality["lag_support"]["lag_support_margin_steps"] == SHIPPED_WARMUP_PERIOD - 90
    assert causality["warmup_budget"]["target_kept_width"] == 98
    assert causality["group_delay_seconds"]["fhr_st"]["max"] == pytest.approx(791.0, abs=0.05)
