r"""Preflight refuses exactly the runs that would produce plausible numbers and mean nothing.

Every guard here defends against a failure with no symptom. A checkpoint that never loaded still
forwards; a config whose geometry contradicts the weights is silently overruled by the weights and
then reported beside them; a ``load_fields`` list missing ``target`` presents downstream as "no
classes found". Each test therefore asserts the *message*, not merely the raise -- a guard that fires
with the wrong explanation sends an operator to fix the wrong thing.

**Three of them are this cell's own, and each is silent in a way the raw pipeline's guards are not.**

*A two-sided shard.* The two dataset variants share every field name and every dtype, so only the
root ``transform`` attribute and the stored widths tell them apart. Evaluated here, one would report
a causal model on coefficients containing their own future with every shape correct.

*A warm-up budget that does not re-resolve to the checkpoint's own channel tuples.* Neither side of
that pair can be reconciled: the threshold is a config key that names no constructor parameter, and
the four tuples are constructor parameters that name no config key -- so
:func:`~teb_vae.lag_attn_cfs.eval.preflight.reconcile_with_checkpoint` skips both silently. The
comparison that *can* fail is re-resolution against the shards this run is about to read.

*The lag-support margin.* Not a refusal at all, and that is the point: it is a **measurement** the
per-lag analyses read, so neither of them asserts a simplification the geometry has stopped
supporting. Three independently configurable quantities go into it and any one of them can break it.

The load check keeps the sibling's three-way outcome, because the initialisation it reads is
unchanged: a fresh construction fails, a fully perturbed model passes, and a model perturbed only
through its posterior passes on the delta-head witness alone. Without that third case the any-of rule
could quietly become all-of and nothing would notice.

**What the committed fixture may be asserted on.** These fast tests run against
``tiny_shard_causal.hdf5`` and a model at the shipped geometry, so they are evidence about schema,
shapes, denominators, counts and refusals -- and about nothing else. Where a refusal needs a
condition, the test *constructs* it.
"""
from __future__ import annotations

import ast
import copy
import json
import inspect
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING
from teb_vae.lag_attn_cfs.eval.preflight import EvalPreconditionUnmet
from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

from .conftest import (
    CAUSAL_SHARD,
    FIXTURES,
    SHIPPED_HORIZON,
    SHIPPED_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    causal_config,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: The committed statistics file accumulated from the causal fixture, at ``trim_minutes: 1.0``.
CAUSAL_STATS = FIXTURES / "tiny_stats_causal.hdf5"

#: What the shipped delta asks the loader for. Restated here rather than read out of the YAML: this
#: is the set the guard is asserted against, and reading it from the file the guard also reads would
#: make the assertion a comparison of one value with itself.
EVAL_LOAD_FIELDS = [
    "fhr", "up", "fhr_st", "fhr_ph", "up_ph", "up_st", "weight", "guid", "epoch", "target",
    "cs_label", "bg_label", "time_from_labor_onset", "second_stage_onset",
]

#: The objective the tiny fixture's task carries, standing in for a checkpoint's own hyperparameters.
HYPER_PARAMETERS: Dict[str, Any] = {
    "likelihood": "gaussian_nll",
    "free_bits": 0.0,
    "lambda_full": 1.0,
    "lambda_base": 1.0,
    "kld_beta": 1.0,
    "beta_schedule": None,
}

#: The lag-support margin the shipped geometry produces: the earliest decoded anchor $F = 133$, the
#: furthest searched lag $L - 1 = 90$, and a lag floor of $0$. Written out because it is the number
#: the whole simplification in the per-lag analyses rests on.
SHIPPED_LAG_SUPPORT_MARGIN = SHIPPED_WARMUP_PERIOD - 90 - 0


def eval_config() -> Dict[str, Any]:
    """Build the config an evaluation actually preflights, over the committed causal fixture.

    The shipped geometry from :func:`~.conftest.causal_config`, plus the four things the evaluation
    delta adds and every guard below reads: the statistics file, the loaded fields, the normalised
    fields, and the objective keys the reconciliation compares.

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
    """The constructor kwargs a checkpoint trained on this fixture would stamp."""
    return shipped_warmup_kwargs()


@pytest.fixture(scope="module")
def model(model_kwargs) -> Any:
    """A model at the shipped geometry, perturbed so the weight-space load witness passes.

    Perturbing the posterior head stands in for a checkpoint load in the one respect this module
    checks: the delta heads are zero at construction, so a model whose weights never moved is exactly
    what :func:`~teb_vae.lag_attn_cfs.eval.preflight.verify_weights_loaded` refuses. The three
    load-check tests below build their own unperturbed models, so the refusal itself is still
    exercised.
    """
    torch.manual_seed(0)
    built = SeqVaeLagAttnCfs(**model_kwargs)
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
    """Run preflight over one config against one model, through this cell's binding."""
    return preflight.run_preflight(
        config=config,
        model=model,
        checkpoint_path="<in-memory>",
        model_kwargs=model_kwargs,
        hyper_parameters=HYPER_PARAMETERS,
        binding=CFS_BINDING,
    )


# =================================================================================================
# The well-formed run
# =================================================================================================
def test_a_well_formed_run_passes_every_check(config, model, model_kwargs) -> None:
    record = _run(config, model, model_kwargs)

    assert all(check["passed"] for check in record["checks"].values())
    assert record["dataset_paths"] == [str(CAUSAL_SHARD)]
    # Non-vacuity: the twelve checks are named, so a guard silently dropped from the run is a
    # failure here rather than a check nobody notices stopped happening.
    assert set(record["checks"]) == {
        "repoint_placeholder",
        "test_shards_exist",
        "stat_path",
        "trim_minutes",
        "causal_transform",
        "load_fields",
        "target_normalized",
        "no_reach_budget",
        "declared_widths",
        "config_matches_checkpoint",
        "warmup_budget_matches_checkpoint",
        "weights_loaded",
    }


def test_the_record_is_written_and_is_readable_json(config, model, model_kwargs, tmp_path) -> None:
    """And ``run_preflight`` writes nothing itself: the refusal must be able to happen before a
    results directory exists at all."""
    record = _run(config, model, model_kwargs)
    assert list(tmp_path.iterdir()) == []

    path = preflight.write_preflight(record, tmp_path)

    written = json.loads(Path(path).read_text(encoding="utf-8"))
    assert path.name == preflight.PREFLIGHT_FILENAME
    assert written["checks"]["weights_loaded"]["passed"] is True
    assert written["causality"]["statement"] == preflight.CAUSALITY_STATEMENT


# =================================================================================================
# The config guards
# =================================================================================================
def test_the_placeholder_is_refused_before_any_existence_check(config, model, model_kwargs) -> None:
    """Otherwise the failure reads as a missing file and the operator goes looking for one."""
    config["dataset_config"]["vae_test_datasets"] = [
        "/data1/fetal-heart-tracing/HDF5_Datasets/REPOINT_ME_causal/"
        "k_fold_cross_validation_dataset/test/hie_cs.hdf5"
    ]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    message = str(excinfo.value)
    assert preflight.REPOINT_MARKER in message
    assert "deliberate non-paths" in message
    # Not "no such file": the placeholder check must pre-empt the existence check.
    assert "do not exist" not in message


def test_a_missing_holdout_directory_names_both_dataset_build_modes(
    config, model, model_kwargs
) -> None:
    """The likely cause, and invisible from the config: the pipeline's default build mode writes
    per-fold test splits and no shared holdout directory at all."""
    config["dataset_config"]["vae_test_datasets"] = [
        "/nowhere/k_fold_cross_validation_dataset/test/hie_cs.hdf5"
    ]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    message = str(excinfo.value)
    assert "do not exist" in message
    assert "augmented" in message and "holdout" in message


def test_an_empty_shard_list_is_refused(config, model, model_kwargs) -> None:
    config["dataset_config"]["vae_test_datasets"] = []

    with pytest.raises(EvalPreconditionUnmet, match="nothing to evaluate"):
        _run(config, model, model_kwargs)


def test_a_missing_statistics_file_raises_with_the_trainers_own_message(
    config, model, model_kwargs
) -> None:
    """Reused, not copied: the actionable text names the generator command and its trim, and it must
    never drift from the training entry point's."""
    config["dataset_config"]["stat_path"] = None

    with pytest.raises(EvalPreconditionUnmet, match="normalization is silently"):
        _run(config, model, model_kwargs)


def test_an_untrimmed_grid_is_refused(config, model, model_kwargs) -> None:
    """On this cell the trim decides more than normalisation: the stored warm-up vectors are rebased
    by exactly it, so a wrong rebase moves the anchor floor and the validity boundary together and
    every warm-fraction readout still reports $1.0$."""
    config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"] = 0.0

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    message = str(excinfo.value)
    assert "trim_minutes" in message
    assert "rebased" in message and "1.0" in message


# =================================================================================================
# The causal variant, which is this cell's own refusal
# =================================================================================================
def test_a_two_sided_shard_is_refused_by_name(config, model, model_kwargs) -> None:
    """The failure with no symptom: the variants share every field name and every dtype, so a
    two-sided shard would be scored as though its coefficients were one-sided and every number would
    report a causal model."""
    config["dataset_config"]["vae_test_datasets"] = [str(TWO_SIDED_SHARD)]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    message = str(excinfo.value)
    assert str(TWO_SIDED_SHARD) in message
    assert "share every field name and every dtype" in message
    assert "36/66/36/15" in message and "43/66/43/15" in message


def test_every_configured_shard_is_read_not_only_the_first(config, model, model_kwargs) -> None:
    """A two-sided file *beside* causal ones is the arrangement a first-file check waves through,
    and it is the likelier accident than an all-two-sided list."""
    config["dataset_config"]["vae_test_datasets"] = [str(CAUSAL_SHARD), str(TWO_SIDED_SHARD)]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    assert str(TWO_SIDED_SHARD) in str(excinfo.value)
    assert str(CAUSAL_SHARD) not in str(excinfo.value)


def test_a_shard_declaring_no_transform_is_refused_too(
    config, model, model_kwargs, tmp_path
) -> None:
    """Absence is not permission. A shard predating the attribute cannot claim to be causal, and
    treating a missing declaration as a pass is how the guard becomes a no-op on old data."""
    from .conftest import write_variant

    stripped = write_variant(
        CAUSAL_SHARD, tmp_path / "no_transform.hdf5",
        lambda handle: handle.attrs.__delitem__("transform"),
    )
    config["dataset_config"]["vae_test_datasets"] = [str(stripped)]

    with pytest.raises(EvalPreconditionUnmet, match="transform=None"):
        _run(config, model, model_kwargs)


def test_a_two_sided_reach_budget_is_refused_naming_both_keys(config, model, model_kwargs) -> None:
    """The forward reach is an energy quantile of a two-sided filter, measured on a bank that did not
    produce these coefficients -- and what it resolves to is a *shift* applied on top of a mask."""
    config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 120.0

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    message = str(excinfo.value)
    assert "causal_reach_budget_s" in message
    assert "causal_warmup_budget_steps" in message


# =================================================================================================
# Fields
# =================================================================================================
def test_every_clinical_load_field_is_required_by_name(config, model, model_kwargs) -> None:
    """The loader *skips* a requested field the shard does not carry, silently, so an absent
    ``target`` presents as "no classes found" rather than as a data problem."""
    fields = config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    assert set(preflight.REQUIRED_EVAL_LOAD_FIELDS) <= set(fields)

    for field in preflight.REQUIRED_EVAL_LOAD_FIELDS:
        broken = eval_config()
        broken["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"].remove(field)

        with pytest.raises(EvalPreconditionUnmet) as excinfo:
            _run(broken, model, model_kwargs)

        assert field in str(excinfo.value)


def test_the_two_phase_key_fields_are_required_and_the_message_says_why(
    config, model, model_kwargs
) -> None:
    """The one the sibling's list does not carry. ``guid`` and ``epoch`` key the anchor tiling's
    per-segment phase, and ``load_fields`` is honoured literally with no forced additions -- so
    dropping either leaves every segment on one tile grid with no shape, no count and no metric
    differing."""
    assert "guid" in preflight.REQUIRED_EVAL_LOAD_FIELDS
    assert "epoch" in preflight.REQUIRED_EVAL_LOAD_FIELDS

    config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"].remove("guid")

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    assert "tile grid" in str(excinfo.value)


@pytest.mark.parametrize("field", ["fhr_st", "fhr_ph"])
@pytest.mark.parametrize("dropped_from", ["load_fields", "normalize_fields"])
def test_an_unnormalized_target_block_is_refused(
    config, model, model_kwargs, field: str, dropped_from: str
) -> None:
    """Both blocks, and both lists. The target is their concatenation, so a config carrying one of
    them is a target with a hole in it -- and without a block in ``normalize_fields`` nothing fails
    at all: the Gaussian NLL is computed against a variance model on another scale."""
    dataloader = config["dataset_config"]["dataloader_config"]
    if dropped_from == "load_fields":
        dataloader["dataset_kwargs"]["load_fields"].remove(field)
    else:
        dataloader["normalize_fields"].remove(field)

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    assert field in str(excinfo.value)
    assert dropped_from in str(excinfo.value)


def test_a_width_mismatch_against_the_shard_is_refused(config, model_kwargs) -> None:
    """A model whose declared widths disagree with the shards fails here rather than inside the
    forward, where the channel error names neither the checkpoint nor the config."""
    class _Narrow:
        c_y, c_u, use_up_st = 7, 5, True

    with pytest.raises(EvalPreconditionUnmet, match="channel widths disagree"):
        _run(config, _Narrow(), model_kwargs)


def test_the_width_check_reads_the_test_shards_and_the_models_own_widths(config, model) -> None:
    """The reused guard reads ``vae_train_datasets`` and returns silently when the key is absent, so
    on an eval config it would check nothing at all -- and what it would have checked is the wrong
    population anyway."""
    view = preflight.config_view_for_shard_guards(config, model)

    assert view["dataset_config"]["vae_train_datasets"] == config["dataset_config"][
        "vae_test_datasets"
    ]
    assert view["model_config"]["VAE_model"]["c_y"] == int(model.c_y)
    assert view["model_config"]["VAE_model"]["c_u"] == int(model.c_u)


# =================================================================================================
# Reconciliation against the checkpoint
# =================================================================================================
def test_a_config_contradicting_the_checkpoints_geometry_is_refused(
    config, model, model_kwargs
) -> None:
    """The architecture is rebuilt from the checkpoint's own ``model_kwargs``, so the checkpoint
    always wins -- and the config's number would be reported beside weights it did not produce."""
    config["model_config"]["VAE_model"]["d_z"] = int(model_kwargs["d_z"]) + 1

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    message = str(excinfo.value)
    assert "d_z" in message
    assert str(model_kwargs["d_z"]) in message


def test_a_config_contradicting_the_trained_objective_is_refused(
    config, model, model_kwargs
) -> None:
    """Scoring an ``mse`` run under a Gaussian NLL reports a different objective's numbers."""
    config["model_config"]["VAE_model"]["likelihood"] = "mse"

    with pytest.raises(EvalPreconditionUnmet, match="likelihood"):
        _run(config, model, model_kwargs)


def test_the_beta_schedule_is_recorded_but_not_reconciled(config, model, model_kwargs) -> None:
    r"""$\beta$ and its ramp weight the training total only; no evaluated readout applies them, so a
    schedule edited after the fit is not a reason to refuse the run."""
    config["model_config"]["VAE_model"]["beta_schedule"] = {"kind": "constant"}

    record = _run(config, model, model_kwargs)["checks"]["config_matches_checkpoint"]

    assert "beta_schedule" not in record["compared"]
    assert "beta_schedule" in record["not_compared"]
    # And the record says why the warm-up budget is absent from both halves, because "why is the
    # budget not reconciled" is the first question a reader of this block has.
    assert "warm-up budget" in record["not_compared_reason"]


def test_a_key_the_config_omits_defers_to_the_constructor(config, model, model_kwargs) -> None:
    """The constructor owns every default, so an omitted key is deference, not contradiction."""
    config["model_config"]["VAE_model"].pop("d_z", None)

    record = _run(config, model, model_kwargs)["checks"]["config_matches_checkpoint"]

    assert "d_z" not in record["compared"]
    assert record["passed"] is True


def test_the_compared_set_is_the_bindings_and_a_narrower_one_is_visible(
    config, model_kwargs
) -> None:
    """Which keys are reconciled is the binding's, because a second architecture reconciles a
    different set -- its own encoder's, and not ``causal_norm``.

    A narrowed tuple must *narrow the comparison* rather than raise: the record then shows the
    dropped key absent from ``compared``, so a run that quietly stopped checking something is legible
    in its own artifact instead of passing indistinguishably from one that checked it.
    """
    full = preflight.reconcile_with_checkpoint(
        config,
        model_kwargs=model_kwargs,
        hyper_parameters=HYPER_PARAMETERS,
        geometry_keys=CFS_BINDING.geometry_keys,
    )
    narrowed = preflight.reconcile_with_checkpoint(
        config,
        model_kwargs=model_kwargs,
        hyper_parameters=HYPER_PARAMETERS,
        geometry_keys=tuple(key for key in CFS_BINDING.geometry_keys if key != "anchor_stride"),
    )

    assert "anchor_stride" in full["compared"]
    assert narrowed["passed"] is True
    assert set(full["compared"]) - set(narrowed["compared"]) == {"anchor_stride"}


# =================================================================================================
# The warm-up budget against the checkpoint's stamped tuples
# =================================================================================================
def test_the_resolved_budget_is_recorded_with_both_streams_widths(
    config, model, model_kwargs
) -> None:
    """Recorded rather than merely checked: which channels survived is what every per-channel
    denominator in the run is taken over."""
    record = _run(config, model, model_kwargs)["checks"]["warmup_budget_matches_checkpoint"]

    assert record["gated"] is True
    assert record["budget_steps"] == config["model_config"]["VAE_model"][
        "causal_warmup_budget_steps"
    ]
    assert record["target_declared_width"] == 102
    assert record["target_kept_width"] == len(model_kwargs["target_keep_index"])
    assert record["source_kept_width"] == record["source_declared_width"] == 51
    # The realised maximum is not the configured threshold, and the distinction is load-bearing: a
    # threshold of 151 keeps the identical channels whose slowest still waits 134 steps.
    assert record["realised_max_warmup_steps"] <= record["budget_steps"]


def test_a_budget_resolving_to_other_tuples_than_the_checkpoints_is_refused(
    config, model, model_kwargs
) -> None:
    """The comparison the reconciliation structurally cannot make. Two arms at two budgets have
    mutually unloadable checkpoints and the class stamp cannot separate them, so a run that read one
    budget's channel axis under another's name would report every per-channel denominator wrong."""
    stamped = dict(model_kwargs)
    stamped["target_keep_index"] = tuple(stamped["target_keep_index"])[:-1]
    stamped["target_warmup_steps"] = tuple(stamped["target_warmup_steps"])[:-1]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, stamped)

    message = str(excinfo.value)
    assert "target_keep_index" in message
    # Both widths, so a reader can see which side is which without re-resolving anything.
    assert str(len(model_kwargs["target_keep_index"])) in message
    assert str(len(stamped["target_keep_index"])) in message


def test_a_config_with_no_budget_against_a_gated_checkpoint_is_refused(
    config, model, model_kwargs
) -> None:
    """The other direction, and the silent one: the rebuilt model gates and masks its channels while
    the config claims it does not, so every warm-fraction readout would describe a guard the config
    does not know about."""
    config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(config, model, model_kwargs)

    assert "target_keep_index" in str(excinfo.value)


def test_an_ungated_pair_passes_and_says_so(config) -> None:
    """A run with no budget on either side is legitimate -- it is what every two-sided cell of the
    family does -- so it passes, warns, and records ``gated: False`` rather than being refused."""
    config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None

    record = preflight.check_warmup_budget_matches_checkpoint(
        config, model_kwargs={}, model_cls=SeqVaeLagAttnCfs
    )

    assert record == {"passed": True, "budget_steps": None, "gated": False}


def test_the_budget_is_re_resolved_against_the_evaluation_shards_alone(config) -> None:
    """``resolve_warmup_budget`` reads both shard lists, which is right for a training driver over
    one dataset and wrong here: the training shards are whatever the checkpoint's resolved config
    named, and on the box an evaluation runs they may not exist at all."""
    config["dataset_config"]["vae_train_datasets"] = ["/nowhere/not_a_shard.hdf5"]

    view = preflight.config_view_for_budget(config)

    assert view["dataset_config"]["vae_train_datasets"] == []
    assert view["dataset_config"]["vae_test_datasets"] == [str(CAUSAL_SHARD)]


def test_a_budget_that_cannot_be_resolved_at_all_is_refused_carrying_the_reason(
    config, model_kwargs
) -> None:
    """The resolver's own refusals -- disagreeing shards, a trim that does not produce the declared
    window -- are the ones a cheap check cannot reach, so its message travels rather than being
    replaced.

    Reached directly rather than through :func:`_run`, because on a full pass the reconciliation
    fires first on the same edit and reports the clearer message; this guard is what catches it on a
    config that moved the checkpoint's side too.
    """
    config["model_config"]["VAE_model"]["sequence_length"] = 299

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.check_warmup_budget_matches_checkpoint(
            config, model_kwargs=model_kwargs, model_cls=SeqVaeLagAttnCfs
        )

    message = str(excinfo.value)
    assert "could not be resolved" in message
    # The resolver's own sentence, not a replacement for it.
    assert "sequence_length" in message and "299" in message


def test_the_unaligned_record_says_so_rather_than_omitting_it(config, model, model_kwargs) -> None:
    """A run reading a shard variant is a fact about the run, so it is recorded even when it is the
    legacy one -- an absent key would read as an older artifact rather than as an unaligned run."""
    record = _run(config, model, model_kwargs)["checks"]["warmup_budget_matches_checkpoint"]

    assert record["reference_delay_s"] is None
    assert record["leg_alignment"] == "none"
    assert record["source_dropped_index"] == []
    assert record["target_max_align_delay"] == record["source_max_align_delay"] == 0


def test_the_alignment_record_names_the_reference_and_the_channels_it_dropped(config) -> None:
    """The alignment removes channels for a reason the widths alone cannot state -- they are above
    the reference, not slow -- and it is not recoverable from any metric the run emits."""
    config["model_config"]["VAE_model"]["causal_align_reference"] = "target_max"
    config["model_config"]["VAE_model"]["warmup_period"] = 134
    resolved = resolve_warmup_budget(preflight.config_view_for_budget(config))
    assert resolved is not None

    record = preflight.check_warmup_budget_matches_checkpoint(
        config,
        model_kwargs=warmup_model_kwargs(resolved, SeqVaeLagAttnCfs),
        model_cls=SeqVaeLagAttnCfs,
    )

    assert record["reference_delay_s"] == pytest.approx(402.1604, abs=5e-4)
    assert record["source_dropped_index"] == [32, 33, 34, 35]
    assert record["source_kept_width"] == 47
    assert record["target_kept_width"] == 98
    assert record["target_max_align_delay"] == record["source_max_align_delay"] == 97


def test_a_checkpoint_built_at_another_reference_is_refused(config) -> None:
    """The failure the warm-up comparison alone would miss.

    Two arms at two *references* keep the identical target channels, so their checkpoints have the
    same target width and load cleanly into each other -- only every lag moves. The alignment
    tuples are therefore compared beside the warm-up ones rather than trusted to fall out of a
    width check.
    """
    config["model_config"]["VAE_model"]["causal_align_reference"] = "target_max"
    config["model_config"]["VAE_model"]["warmup_period"] = 134
    resolved = resolve_warmup_budget(preflight.config_view_for_budget(config))
    assert resolved is not None
    stamped = dict(warmup_model_kwargs(resolved, SeqVaeLagAttnCfs))
    # The unaligned arm's stamp: same four warm-up tuples, no shifts at all.
    stamped["target_align_delays"] = None
    stamped["source_align_delays"] = None

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.check_warmup_budget_matches_checkpoint(
            config, model_kwargs=stamped, model_cls=SeqVaeLagAttnCfs
        )

    message = str(excinfo.value)
    assert "target_align_delays" in message
    assert "none at all" in message
    assert "causal_align_reference" in message


def test_the_disclosure_carries_the_reference_beside_the_stale_step_count(config) -> None:
    r"""Two numbers that are both nonzero under an alignment and mean different things.

    ``source_delay_steps`` is a count of *stored steps*, attained by the fastest channel -- the one
    furthest from the reference. ``source_reference_delay_s`` is the physical instant every aligned
    source channel reports at a step. A lag in seconds is computed from the second; a
    stored-coefficient axis is indexed by the first. They travel together so a reader does not have
    to guess which is which.
    """
    config["model_config"]["VAE_model"]["causal_align_reference"] = "target_max"
    config["model_config"]["VAE_model"]["warmup_period"] = 134
    resolved = resolve_warmup_budget(preflight.config_view_for_budget(config))
    assert resolved is not None
    aligned_model = SeqVaeLagAttnCfs(
        **dict(
            shipped_warmup_kwargs(warmup_period=134),
            **warmup_model_kwargs(resolved, SeqVaeLagAttnCfs),
        )
    )
    warmup = preflight.check_warmup_budget_matches_checkpoint(
        config,
        model_kwargs=warmup_model_kwargs(resolved, SeqVaeLagAttnCfs),
        model_cls=SeqVaeLagAttnCfs,
    )

    record = preflight.causality_disclosure(config, aligned_model, warmup=warmup)

    assert record["source_delay_steps"] == 97
    assert record["source_reference_delay_s"] == pytest.approx(402.1604, abs=5e-4)
    assert record["source_delay_seconds"] == pytest.approx(97 * 4.0)
    assert record["source_reference_delay_s"] != record["source_delay_seconds"]


# =================================================================================================
# The measured geometry
# =================================================================================================
def test_the_lag_support_margin_is_measured_and_is_the_shipped_forty_three(model) -> None:
    r"""$133 - 90 - 0 = 43$. Every per-lag simplification -- the absent support correction, the
    untruncated recomputation, the $\log L$ entropy ceiling -- holds exactly when this is $\ge 0$."""
    record = preflight.lag_support(model)

    assert record["min_decoded_anchor"] == SHIPPED_WARMUP_PERIOD
    assert record["max_lag"] == 90
    assert record["lag_floor"] == 0
    assert record["lag_support_margin_steps"] == SHIPPED_LAG_SUPPORT_MARGIN == 43
    assert record["every_lag_valid_at_every_anchor"] is True


def test_each_of_the_three_quantities_can_break_the_margin_on_its_own() -> None:
    """Measured rather than assumed, and this is what says the measurement is not a constant: a lower
    floor, a wider lag range and a non-zero lag floor each move it independently, and any one of them
    silently reintroduces the truncation the shipped geometry does not have."""
    torch.manual_seed(0)
    baseline = preflight.lag_support(SeqVaeLagAttnCfs(**tiny_warmup_kwargs()))
    assert baseline["every_lag_valid_at_every_anchor"] is False or baseline[
        "lag_support_margin_steps"
    ] >= 0

    wider = preflight.lag_support(SeqVaeLagAttnCfs(**tiny_warmup_kwargs(max_lag=12)))
    floored = preflight.lag_support(SeqVaeLagAttnCfs(**tiny_warmup_kwargs(lag_floor=3)))

    assert wider["lag_support_margin_steps"] < baseline["lag_support_margin_steps"]
    assert floored["lag_support_margin_steps"] == baseline["lag_support_margin_steps"] - 3
    # And a negative margin is a warning, not a refusal: a truncated-support run is legitimate.
    assert wider["every_lag_valid_at_every_anchor"] is False


def test_the_anchor_geometry_records_both_strides(model) -> None:
    """The evaluation always decodes densely; the training stride is recorded beside it because a
    figure that did not say which geometry it was produced at would be unreadable against the
    training CSV -- $A_{\\max}$ differs by a factor of $S$ between them."""
    record = preflight.anchor_geometry(model)

    assert record["evaluation_stride"] == 1
    assert record["anchors_per_sample"] == record["t_valid"] - record["anchor_floor"] == 137
    assert record["training_stride"] == SHIPPED_HORIZON
    assert record["training_anchors_per_sample_max"] == -(-137 // SHIPPED_HORIZON)
    assert record["block_width"] == SHIPPED_HORIZON * record["target_kept_width"]


# =================================================================================================
# Weight-space load verification
# =================================================================================================
def test_a_freshly_constructed_model_is_refused() -> None:
    """Every witness tensor is still at the value the constructor gave it, which no trained model
    produces."""
    torch.manual_seed(0)
    fresh = SeqVaeLagAttnCfs(**tiny_warmup_kwargs())

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.verify_weights_loaded(fresh)

    message = str(excinfo.value)
    assert "still exactly at the value the constructor gave it" in message
    assert "delta_heads" in message and "film_generators" in message


def test_a_freshly_constructed_model_with_horizon_attention_is_refused_too() -> None:
    """The refusal must survive the witness set growing. The attention's residual gains start at a
    *nonzero* constant, so a witness that reported $\\max|w|$ rather than the deviation from that
    constant would carry "evidence" on every model ever built."""
    torch.manual_seed(0)
    fresh = SeqVaeLagAttnCfs(**tiny_warmup_kwargs(horizon_attention_blocks=2))

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.verify_weights_loaded(fresh)

    assert "horizon_attention_gains" in str(excinfo.value)


def test_a_model_perturbed_only_through_its_posterior_still_passes() -> None:
    """The reason the rule is any-of. Training moves the delta heads and the FiLM generators
    independently, so a real checkpoint whose FiLM path never left zero is an ordinary model -- and
    an all-of rule would refuse it, along with this repository's own test fixtures."""
    torch.manual_seed(0)
    built = SeqVaeLagAttnCfs(**tiny_warmup_kwargs())
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in built.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)

    record = preflight.verify_weights_loaded(built)

    assert record["witnesses_with_evidence"] == ["delta_heads"]
    assert record["max_abs_weight"]["film_generators"] == 0.0


# =================================================================================================
# The causality disclosure
# =================================================================================================
def test_the_statement_states_one_sidedness_and_refuses_the_name_exactly_once() -> None:
    """The sibling's sentence says the inputs read their own future. Here they do not, so a copied
    refusal would be a false disclosure rather than a conservative one. What survives is the
    narrower refusal, and the artifact scan is written to allow exactly one occurrence of the name
    it refuses."""
    statement = preflight.CAUSALITY_STATEMENT

    assert "one-sided" in statement
    assert "genuine forecast" in statement
    assert statement.count("transfer entropy") == 1
    assert "may be labelled a transfer entropy" in statement
    # The sibling's claim, and the one that must not survive the copy.
    assert "NOT causal" not in statement
    assert "95%-energy quantile" not in statement


def test_the_disclosure_is_assembled_exactly_as_it_is_written_down(config, model) -> None:
    """Key for key and in order. The encoder's half arrives through a callable rather than being read
    inline, and an extraction that changed *what* the record says -- or where a key sits in it --
    would be invisible to every assertion that reads one key at a time."""
    record = preflight.causality_disclosure(config, model)

    assert list(record) == [
        "one_sided_inputs",
        "statement",
        "transform",
        "causal_reach_budget_s",
        "group_delay_seconds",
        "warmup_budget",
        "anchor_geometry",
        "lag_support",
        "lag_axis",
        "causal_norm",
        "n_causalized_norms",
        "source_delay_steps",
        "source_delay_seconds",
        "source_delay_is_max_over_channels",
        # The alignment reference, beside the stored-step maximum and never merged into it: under a
        # channel alignment both are nonzero and they are different quantities, so a reader who
        # took one for the other would state a lag wrong by minutes with nothing failing.
        "source_reference_delay_s",
        "horizon_seconds",
    ]
    assert record["one_sided_inputs"] is True
    assert record["transform"] == "causal"
    assert record["causal_reach_budget_s"] is None
    assert record["warmup_budget"] is None
    # ``None`` rather than zero on a record with no resolved budget beside it: there is no common
    # clock to name, and zero would read as "aligned to the anchor itself".
    assert record["source_reference_delay_s"] is None
    assert record["horizon_seconds"] == float(model.horizon) * 4.0
    # And the same record when the callable is passed explicitly, which is how a run reaches it.
    assert preflight.causality_disclosure(config, model, preflight.cfs_encoder_disclosure) == record


def test_the_record_carries_the_group_delay_the_lag_axis_is_measured_in(config, model) -> None:
    """Left only in ``band_channel_map.csv``, a reader of ``summary.json`` would have the lag numbers
    and no statement of what they are lags *in* -- and the summary is the artifact that gets quoted.

    The values are the committed fixture's own, read off its block attributes.
    """
    delays = preflight.causality_disclosure(config, model)["group_delay_seconds"]

    assert delays["source"] == str(CAUSAL_SHARD)
    assert set(delays) == {"source", "fhr_st", "fhr_ph", "up_st", "up_ph"}
    assert delays["fhr_st"]["min"] == pytest.approx(13.3, abs=0.05)
    assert delays["fhr_st"]["max"] == pytest.approx(791.0, abs=0.05)
    assert delays["fhr_st"]["n_channels"] == 36
    assert delays["fhr_ph"]["n_channels"] == 66
    for entry in (delays[name] for name in ("fhr_st", "fhr_ph", "up_st", "up_ph")):
        assert entry["min"] <= entry["median"] <= entry["max"]


def test_the_completed_record_carries_the_resolved_budget_beside_the_statement(
    config, model, model_kwargs
) -> None:
    """So ``summary.json`` states the caveat and the guard that shaped the channel axis without any
    other file."""
    causality = _run(config, model, model_kwargs)["causality"]

    assert causality["warmup_budget"]["budget_steps"] == 134
    assert causality["warmup_budget"]["target_kept_width"] == 98
    assert causality["lag_support"]["lag_support_margin_steps"] == SHIPPED_LAG_SUPPORT_MARGIN
    assert causality["anchor_geometry"]["anchors_per_sample"] == 137
    assert "stored-coefficient time" in causality["lag_axis"]["label"]
    assert "not a transfer entropy" in causality["lag_axis"]["caveat"]


def test_the_encoder_half_is_this_encoders_and_carries_its_consequence(model) -> None:
    """``causal_norm`` is a property of the recurrent encoder rather than of the target domain, so it
    and its consequence sentence live in the callable the binding supplies. Everything the two cfs
    cells share is in the record's shared half instead."""

    class _Pooling:
        causal_norm, n_causalized_norms = False, 0

    assert preflight.cfs_encoder_disclosure(model) == {
        "causal_norm": bool(model.causal_norm),
        "n_causalized_norms": int(model.n_causalized_norms),
    }

    pooling = preflight.cfs_encoder_disclosure(_Pooling())
    assert pooling["causal_norm"] is False
    assert "p(z_t | Y_<=t) conditions on Y_>t" in pooling["causal_norm_consequence"]
    # The consequence this cell has that the raw one does not: one-sided data does not survive an
    # encoder that pools over time, so the forecast claim would hold of the data and not of the run.
    assert "holds of the DATA" in pooling["causal_norm_consequence"]


def test_the_consequence_is_logged_and_not_only_recorded() -> None:
    """An operator reading the console must be told; a sentence only in ``preflight.json`` is a
    sentence nobody sees until after the run."""

    class _Pooling:
        causal_norm, n_causalized_norms = False, 0

    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        preflight.cfs_encoder_disclosure(_Pooling())
    finally:
        logger.remove(sink)

    assert any("causal_norm=False" in message for message in warnings)


def test_the_disclosure_refuses_a_model_it_cannot_read_naming_both() -> None:
    """Rather than reporting a model that stopped exposing something as a model with nothing to
    report -- the disclosure would go quiet in exactly the case a reader most needs to be told."""

    class _Renamed:
        n_causalized_norms = 0

    with pytest.raises(AttributeError) as excinfo:
        preflight.cfs_encoder_disclosure(_Renamed())

    message = str(excinfo.value)
    assert "causal_norm" in message
    assert "_Renamed" in message


def test_a_binding_may_disclose_something_else_entirely(config, model) -> None:
    """What makes the seam a seam: an encoder with no ``causal_norm`` discloses its own facts and the
    record carries no key that means nothing for it. The shared half is unchanged, because it
    describes the target domain rather than the encoder."""
    record = preflight.causality_disclosure(
        config, model, lambda built: {"time_pooling_normalisers": 0}
    )

    assert record["time_pooling_normalisers"] == 0
    assert "causal_norm" not in record
    assert record["statement"] == preflight.CAUSALITY_STATEMENT
    assert record["lag_support"]["lag_support_margin_steps"] == SHIPPED_LAG_SUPPORT_MARGIN


@pytest.mark.parametrize("reserved", ["statement", "lag_support", "warmup_budget"])
def test_an_encoder_disclosure_may_not_overwrite_a_shared_key(config, model, reserved: str) -> None:
    """The splat sits mid-literal, so a reused name would either replace a shared key -- including
    the statement itself -- or be dropped by a key below it, and both would be silent in an artifact
    whose whole purpose is to be read literally."""
    with pytest.raises(ValueError, match=reserved):
        preflight.causality_disclosure(config, model, lambda built: {reserved: "anything"})


# =================================================================================================
# The guard recovery table
# =================================================================================================
def _functions_that_refuse() -> set:
    """Return the name of every function in ``preflight`` carrying a ``raise EvalPreconditionUnmet``.

    Walked from the AST rather than from a hand-kept list, which is the whole point: the sibling's
    recovery table lives in a document and nothing checks it, so a guard added without a row is a
    refusal an operator meets with no stated fix.

    Returns:
        The enclosing function names.
    """
    tree = ast.parse(Path(inspect.getfile(preflight)).read_text(encoding="utf-8"))
    refusing = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Raise)
                and isinstance(inner.exc, ast.Call)
                and isinstance(inner.exc.func, ast.Name)
                and inner.exc.func.id == "EvalPreconditionUnmet"
            ):
                refusing.add(node.name)
    return refusing


def test_every_raise_site_has_a_recovery_row() -> None:
    """Both directions: a guard with no row is a refusal with no stated fix, and a row for a function
    that no longer refuses is advice about something that cannot happen."""
    refusing = _functions_that_refuse()

    assert refusing, "the AST walk found no refusals at all, so this check is vacuous"
    assert refusing == set(preflight.GUARD_RECOVERY), (
        f"no recovery row: {sorted(refusing - set(preflight.GUARD_RECOVERY))}; "
        f"row for a function that does not refuse: "
        f"{sorted(set(preflight.GUARD_RECOVERY) - refusing)}"
    )


def test_every_recovery_names_a_config_key_or_a_command() -> None:
    """"The shards are wrong" is a description of the problem. "Repoint
    dataset_config.vae_test_datasets" is a recovery, and the difference is whether an operator knows
    what to edit."""
    actionable = []
    for name, row in preflight.GUARD_RECOVERY.items():
        assert row["cause"].strip(), name
        recovery = row["recovery"]
        assert recovery.strip(), name
        actionable.append(
            name if ("_config." in recovery or ".py" in recovery or "--" in recovery) else ""
        )

    assert all(actionable), (
        f"recovery text naming neither a config key nor a command: "
        f"{sorted(name for name, ok in zip(preflight.GUARD_RECOVERY, actionable) if not ok)}"
    )


def test_the_recovery_table_covers_the_causal_guards_by_name() -> None:
    """Named rather than counted: these three are what this cell adds, and a table that lost one
    would still pass a count."""
    assert {
        "check_causal_transform",
        "check_no_reach_budget",
        "check_warmup_budget_matches_checkpoint",
    } <= set(preflight.GUARD_RECOVERY)


# =================================================================================================
# Against a real checkpoint
# =================================================================================================
@pytest.mark.slow
def test_a_real_run_preflights_against_the_shards_it_was_trained_on(
    cohort_run, cohort_shards, cohort_stats
) -> None:
    """The one test that proves the guards against a checkpoint the driver actually wrote, rather
    than against constructor kwargs assembled here.

    What only a real run can supply is the pairing this module's own guard exists for: the four
    warm-up tuples the driver resolved against these shards and stamped into ``model_kwargs``, beside
    the ``resolved_config.yaml`` recording the configuration that produced them.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.eval import probe
    from teb_vae.lag_attn_cfs.eval.config_schema import merge_eval_overrides
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

    checkpoint = sorted((cohort_run / "model_checkpoints").glob("*.ckpt"))[0]
    merged = merge_eval_overrides(
        load_config(str(cohort_run / "model_checkpoints" / RESOLVED_CONFIG_FILENAME))
    )
    merged["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    merged["dataset_config"]["stat_path"] = cohort_stats

    blob = probe.read_checkpoint(checkpoint)
    task = probe.load_task(checkpoint, torch.device("cpu"), blob=blob)
    record = preflight.run_preflight(
        config=merged,
        model=task.orig_model,
        checkpoint_path=checkpoint,
        model_kwargs=blob["model_kwargs"],
        hyper_parameters=dict(task.hparams),
        binding=CFS_BINDING,
    )

    assert all(check["passed"] for check in record["checks"].values())
    assert record["checks"]["warmup_budget_matches_checkpoint"]["gated"] is True
    assert record["causality"]["statement"] == preflight.CAUSALITY_STATEMENT
    assert record["causality"]["group_delay_seconds"]["fhr_st"]["max"] > 0.0


@pytest.mark.slow
def test_a_real_checkpoint_against_another_budget_is_refused(
    cohort_run, cohort_shards, cohort_stats
) -> None:
    """Constructed rather than hoped for: the fixture's own budget agrees with its own checkpoint, so
    the refusal is only non-vacuous against a budget deliberately moved."""
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.eval import probe
    from teb_vae.lag_attn_cfs.eval.config_schema import merge_eval_overrides
    from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

    checkpoint = sorted((cohort_run / "model_checkpoints").glob("*.ckpt"))[0]
    merged = merge_eval_overrides(
        load_config(str(cohort_run / "model_checkpoints" / RESOLVED_CONFIG_FILENAME))
    )
    merged["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    merged["dataset_config"]["stat_path"] = cohort_stats
    blob = probe.read_checkpoint(checkpoint)
    resolved = int(merged["model_config"]["VAE_model"]["causal_warmup_budget_steps"])

    moved = copy.deepcopy(merged)
    moved["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = resolved - 1

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.check_warmup_budget_matches_checkpoint(
            moved, model_kwargs=blob["model_kwargs"], model_cls=CFS_BINDING.model_cls
        )

    assert "target_keep_index" in str(excinfo.value)
