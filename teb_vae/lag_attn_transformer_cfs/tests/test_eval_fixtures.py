r"""The evaluation fixtures this suite is built on, and what each one has to carry.

The shards are the causal sibling's, imported rather than copied, and they are not re-tested here:
what they contain is a property of the *dataset* and is pinned in the suite that owns the writer.
What is tested here is what this package added on top -- that those shards reach *this* model
through the real driver, that the checkpoint the fit leaves behind is one the evaluation will
accept, and that a full pass through the causal pipeline records this cell rather than the other.

Two of the three assertions are load-bearing in a way that is easy to lose sight of.

**The checkpoint's acceptance.** Preflight verifies a checkpoint load in *weight space* rather than
behaviourally, because this model is exactly zero-KL at construction and a behavioural probe cannot
separate "the checkpoint never loaded" from "a real model whose source pathway collapsed". So it
requires the zero-initialised tensors to have moved -- and a fixture that skipped the optimizer
steps would be refused by every test that drives a run, with a message about the checkpoint rather
than about the fixture.

**The ``model_class`` stamp.** It is written in exactly one place a finished run keeps, and the
cross-cell table keys its rows on it: the dumped config carries every constructor keyword and not
the class they build, so a run whose stamp is wrong or missing is a row this package's whole
comparison is unable to place.

Beside them, the run itself: it completes with every step ok, records this cell's class, was
scored at the dense anchor set, and was launched with every clinical field its readouts are asked in.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval import run as shared_run
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING

pytestmark = pytest.mark.slow

#: The five fields the delta adds over the model's own data contract; every clinical question in
#: the pipeline is asked in one of them.
_CLINICAL_FIELDS = ("target", "epoch", "cs_label", "bg_label", "time_from_labor_onset")


# =================================================================================================
# The fit
# =================================================================================================
def test_the_checkpoint_sits_where_a_run_leaves_one(trf_cohort_run) -> None:
    """The evaluation finds a run's config by walking up from the checkpoint, so the fixture has
    to reproduce the layout rather than merely produce a blob."""
    checkpoints = sorted((Path(trf_cohort_run) / "model_checkpoints").glob("*.ckpt"))

    assert checkpoints, "the fit left no checkpoint"
    assert (Path(trf_cohort_run) / "model_checkpoints" / RESOLVED_CONFIG_FILENAME).is_file()
    assert shared_run.resolved_config_for(checkpoints[0]).is_file()


def test_the_checkpoint_carries_the_records_a_rebuild_and_a_cross_cell_row_need(
    trf_cohort_run,
) -> None:
    """Real ones, from the driver's own kwargs builder: the architecture is rebuilt from
    ``model_kwargs`` and the objective from ``hyper_parameters``, so a fixture that faked either
    would test a model the driver would never construct -- and ``model_class`` is what the
    cross-cell table keys a row on."""
    checkpoint = sorted((Path(trf_cohort_run) / "model_checkpoints").glob("*.ckpt"))[0]
    blob = shared_run.read_checkpoint(checkpoint)

    assert blob["model_kwargs"]
    assert blob["hyper_parameters"]
    assert blob["model_class"] == TRF_CFS_BINDING.model_cls.__name__ == "SeqVaeLagAttnTrfCfs"
    # Every reconciled key is present, so preflight's geometry comparison is not vacuous.
    missing = [key for key in TRF_CFS_BINDING.geometry_keys if key not in blob["model_kwargs"]]
    assert missing == []
    # And the four warm-up tuples the budget resolved against these shards, which decide which
    # target channels exist at all.
    for name in ("target_keep_index", "target_warmup_steps",
                 "source_keep_index", "source_warmup_steps"):
        assert blob["model_kwargs"].get(name), name


def test_the_checkpoint_rebuilds_into_this_cells_task_and_passes_the_load_check(
    trf_cohort_run,
) -> None:
    """The load check is the one that makes the fixture usable at all. The delta heads and FiLM
    generators are zeroed at construction, so preflight refuses a checkpoint whose weights never
    moved -- which is what the optimizer steps of a real one-epoch fit exist to prevent here."""
    checkpoint = sorted((Path(trf_cohort_run) / "model_checkpoints").glob("*.ckpt"))[0]

    task = shared_run.load_task(checkpoint, torch.device("cpu"), binding=TRF_CFS_BINDING)

    assert type(task).__name__ == TRF_CFS_BINDING.task_cls.__name__
    assert type(task.orig_model).__name__ == TRF_CFS_BINDING.model_cls.__name__
    assert task.training is False
    assert preflight.verify_weights_loaded(task.orig_model)["passed"] is True


# =================================================================================================
# The evaluation run
# =================================================================================================
def test_the_run_completes_and_records_this_cell(trf_collected_run) -> None:
    """The stamp travels from the checkpoint into ``run_context``, which is the only place a
    finished run says which architecture produced it.

    The failed steps are named with their errors rather than left to a bare ``1 == 0``: this run
    costs a quarter of an hour to reproduce, so a failure that does not say which analysis raised
    buys a second one.
    """
    failed = [
        f"{record['name']}: {record.get('error')}"
        for record in trf_collected_run["summary"]["steps"]
        if record["status"] != "ok"
    ]

    assert failed == [], failed
    assert trf_collected_run["exit_code"] == 0
    assert trf_collected_run["summary"]["run_context"]["model_class"] == "SeqVaeLagAttnTrfCfs"


def test_the_run_was_scored_at_the_dense_anchor_set(trf_collected_run) -> None:
    """The evaluation decodes densely whatever tiling the run trained at, and records both -- a
    figure or a table that did not say which geometry it was produced at would be unreadable
    against the training CSV."""
    geometry = trf_collected_run["summary"]["run_context"]["anchor_geometry"]

    assert geometry["anchor_phase"] == 0
    assert geometry["anchor_stride"] == 1
    assert geometry["training_anchor_stride"] is not None


def test_all_five_clinical_fields_reached_the_run(trf_collected_run) -> None:
    """The loader *skips* a field a shard was not asked for, silently, so a missing one presents
    downstream as "no classes found" rather than as an error. Read off the run's own dumped config
    rather than off the committed delta, which is the file that could be right while the run was
    launched against something else."""
    import yaml

    dumped = Path(trf_collected_run["results_dir"]) / RESOLVED_CONFIG_FILENAME
    config = yaml.safe_load(dumped.read_text(encoding="utf-8"))
    load_fields = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]

    missing = [field for field in _CLINICAL_FIELDS if field not in load_fields]
    assert missing == []
    # ``guid`` and ``epoch`` are load-bearing here rather than leftovers: the per-recording chain
    # and the tile phase are keyed on the pair.
    assert {"guid", "epoch"} <= set(load_fields)
