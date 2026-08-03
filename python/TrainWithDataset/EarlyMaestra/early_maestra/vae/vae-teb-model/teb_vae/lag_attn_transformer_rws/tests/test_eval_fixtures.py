r"""The evaluation fixtures this package's suite is built on, and what each one has to carry.

The shards are the sibling's, imported rather than copied, and they are not re-tested here: what
they contain is a property of the dataset and is pinned in the suite that owns the writer. What is
tested here is what this package added on top -- that those shards reach *this* model through the
real loader, and that the throwaway checkpoint is one the evaluation will accept.

The checkpoint's acceptance is the load-bearing case. Preflight verifies a checkpoint load in
weight space rather than behaviourally, because this model is exactly zero-KL at construction and a
behavioural probe cannot separate "the checkpoint never loaded" from "a real model whose source
pathway collapsed". So it requires the zero-initialised tensors to have moved -- and a fixture that
skipped the optimizer steps would be refused by every test that drives a run, with a message about
the checkpoint rather than about the fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_rws.eval import preflight, run as shared_run
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from teb_vae.lag_attn_transformer_rws.eval.binding import TRF_BINDING

from .conftest import MULTI_CLASS_GUIDS_PER_SHARD, MULTI_CLASS_SUBGROUPS

pytestmark = pytest.mark.slow

#: The five fields the delta adds over the model's own data contract; every clinical question in
#: the pipeline is asked in one of them.
_CLINICAL_FIELDS = ("target", "epoch", "cs_label", "bg_label", "time_from_labor_onset")


@pytest.fixture(scope="module")
def batch(multi_class_shards, repointed_overrides):
    """One batch off the generated shards, through the real loader and the committed delta."""
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_rws.eval.config_schema import (
        force_single_process_loader,
        merge_eval_overrides,
    )
    from teb_vae.lag_attn_transformer_rws.tests.conftest import absolutize_dataset_paths
    from train.data_module import GraphDataModule

    repo_root = Path(__file__).resolve().parents[3]
    tiny = repo_root / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "tiny.yaml"
    config = merge_eval_overrides(
        absolutize_dataset_paths(load_config(str(tiny))), repointed_overrides
    )
    force_single_process_loader(config)
    return next(iter(GraphDataModule(config).test_dataloader()))


# =============================================================================
# The shards, as this model sees them
# =============================================================================
def test_the_shards_reach_this_model_through_the_real_loader(batch) -> None:
    """Through ``GraphDataModule`` and this package's own delta, not a stub: the fields, the
    widths and the trimmed length are what the model's data contract is written against."""
    assert batch.fhr_st.shape[-1] == 43
    assert batch.fhr_ph.shape[-1] == 66
    assert batch.up_st.shape[-1] == 43
    assert batch.up_ph.shape[-1] == 15
    # trim_minutes: 1.0 removes 15 decimated steps from each end of the stored 330.
    assert batch.fhr_st.shape[1] == 300


def test_all_five_clinical_fields_arrive(batch) -> None:
    """The loader *skips* a field a shard does not carry, silently, so a missing one presents
    downstream as "no classes found" rather than as an error."""
    missing = [field for field in _CLINICAL_FIELDS if not hasattr(batch, field)]
    assert missing == []


def test_enough_recordings_per_shard_for_a_testable_cohort(multi_class_shards) -> None:
    """The shared rank tests exclude any group with fewer than three finite values, so at two
    recordings per shard every by-subgroup and by-class comparison could only be a skip."""
    assert MULTI_CLASS_GUIDS_PER_SHARD >= 3
    assert len(multi_class_shards) == len(MULTI_CLASS_SUBGROUPS)


def test_three_clinical_classes_over_more_shards_than_classes(multi_class_shards) -> None:
    """With one class every by-class table has one group; with as many shards as classes the two
    groupings coincide and a bug that swapped them would be invisible."""
    assert len(set(MULTI_CLASS_SUBGROUPS.values())) == 3
    assert len(multi_class_shards) > 3


# =============================================================================
# The checkpoint
# =============================================================================
def test_the_checkpoint_sits_where_a_run_leaves_one(trained_run) -> None:
    """The evaluation finds a run's config by walking up from the checkpoint, so the fixture has
    to reproduce the layout rather than merely produce a blob."""
    assert Path(trained_run).is_file()
    assert Path(trained_run).parent.name == "model_checkpoints"
    assert (Path(trained_run).parent / RESOLVED_CONFIG_FILENAME).is_file()
    assert shared_run.resolved_config_for(Path(trained_run)).is_file()


def test_the_checkpoint_carries_the_two_records_a_rebuild_needs(trained_run) -> None:
    """Real ones, from the trainer's own kwargs builder: the architecture is rebuilt from
    ``model_kwargs`` and the objective from ``hyper_parameters``, so a fixture that faked either
    would test a model the trainer would never construct."""
    blob = shared_run.read_checkpoint(trained_run)

    assert blob["model_kwargs"]
    assert blob["hyper_parameters"]["likelihood"] == "gaussian_nll"
    assert blob["model_class"] == TRF_BINDING.model_cls.__name__
    # Every reconciled key is present, so the geometry comparison is not vacuous.
    missing = [key for key in TRF_BINDING.geometry_keys if key not in blob["model_kwargs"]]
    assert missing == []


def test_the_checkpoint_rebuilds_into_this_models_task(trained_run) -> None:
    task = shared_run.load_task(trained_run, torch.device("cpu"), binding=TRF_BINDING)

    assert type(task).__name__ == TRF_BINDING.task_cls.__name__
    assert type(task.orig_model).__name__ == TRF_BINDING.model_cls.__name__
    assert task.training is False


def test_the_checkpoint_passes_the_weight_space_load_check(trained_run) -> None:
    """The one that makes the fixture usable at all. The delta heads and FiLM generators are
    zeroed at construction, so preflight refuses a checkpoint whose weights never moved -- which
    is what the optimizer steps in the fixture exist to prevent."""
    task = shared_run.load_task(trained_run, torch.device("cpu"), binding=TRF_BINDING)

    check = preflight.verify_weights_loaded(task.orig_model)

    assert check["passed"] is True
