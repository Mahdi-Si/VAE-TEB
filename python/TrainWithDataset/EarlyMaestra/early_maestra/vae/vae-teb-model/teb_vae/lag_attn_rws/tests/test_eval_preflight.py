r"""Preflight refuses exactly the runs that would produce plausible numbers and mean nothing.

Every guard here defends against a failure with no symptom. A checkpoint that never loaded still
forwards; a config whose geometry contradicts the weights is silently overruled by the weights
and then reported beside them; an untrimmed grid shifts every forecast window by a minute and
only ``warnings.warn``; a ``load_fields`` list missing ``target`` presents downstream as "no
classes found". Each test therefore asserts the *message*, not merely the raise -- a guard that
fires with the wrong explanation sends an operator to fix the wrong thing.

The load check is the one worth spelling out. This model is exactly zero-KL at construction, so a
behavioural probe cannot separate "the checkpoint never loaded" from "a real model whose source
pathway collapsed": both read zero, and refusing the second would destroy the most important
finding a run can produce. So the check that raises reads *weight space*, and the last three
tests pin its three-way outcome -- a fresh construction fails, a fully perturbed model passes,
and a model perturbed only through its posterior passes on the delta-head witness alone. Without
that third case the any-of rule could quietly become all-of and nothing would notice.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn import channel_reach
from teb_vae.lag_attn_rws.eval import preflight, run as run_module
from teb_vae.lag_attn_rws.eval.preflight import EvalPreconditionUnmet
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME

from .conftest import TINY_KWARGS


@pytest.fixture(scope="module")
def loaded(trained_run) -> dict:
    """The checkpoint under test, rebuilt exactly as a run rebuilds it."""
    blob = run_module.read_checkpoint(trained_run)
    task = run_module.load_task(trained_run, torch.device("cpu"), blob=blob)
    return {
        "model": task.orig_model,
        "model_kwargs": blob["model_kwargs"],
        "hyper_parameters": dict(task.hparams),
        "checkpoint": trained_run,
    }


@pytest.fixture
def config(trained_run, repointed_overrides) -> dict:
    """The merged config a run actually preflights: the checkpoint's own, plus the delta."""
    from teb_vae.lag_attn_rws.eval.config_schema import merge_eval_overrides

    resolved = load_config(str(trained_run.parent / RESOLVED_CONFIG_FILENAME))
    return merge_eval_overrides(resolved, repointed_overrides)


def _run(config: dict, loaded: dict) -> dict:
    """Run preflight over one config against the loaded checkpoint."""
    return preflight.run_preflight(
        config=config,
        model=loaded["model"],
        checkpoint_path=loaded["checkpoint"],
        model_kwargs=loaded["model_kwargs"],
        hyper_parameters=loaded["hyper_parameters"],
        binding=run_module.RWS_BINDING,
    )


# =============================================================================
# The well-formed run
# =============================================================================
def test_a_well_formed_run_passes_every_check(config, loaded) -> None:
    record = _run(config, loaded)

    assert all(check["passed"] for check in record["checks"].values())
    assert record["checkpoint"] == str(loaded["checkpoint"])
    assert record["dataset_paths"], "the shard list must be recorded, not merely checked"


# =============================================================================
# The config guards
# =============================================================================
def test_the_placeholder_is_refused_before_any_existence_check(config, loaded) -> None:
    """Otherwise the failure reads as a missing file and the operator goes looking for one."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["vae_test_datasets"] = [
        "/data1/REPOINT_ME_new_channel_selection/k_fold_cross_validation_dataset/test/hie_cs.hdf5"
    ]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(broken, loaded)

    message = str(excinfo.value)
    assert preflight.REPOINT_MARKER in message
    assert "deliberate non-paths" in message
    # Not "no such file": the placeholder check must pre-empt the existence check.
    assert "do not exist" not in message


def test_a_missing_holdout_directory_names_both_dataset_build_modes(config, loaded) -> None:
    """The likely cause, and invisible from the config: the pipeline's default build mode writes
    per-fold test splits and no shared holdout directory at all."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["vae_test_datasets"] = [
        "/nowhere/k_fold_cross_validation_dataset/test/hie_cs.hdf5"
    ]

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(broken, loaded)

    message = str(excinfo.value)
    assert "do not exist" in message
    assert "augmented" in message and "holdout" in message


def test_an_empty_shard_list_is_refused(config, loaded) -> None:
    broken = copy.deepcopy(config)
    broken["dataset_config"]["vae_test_datasets"] = []

    with pytest.raises(EvalPreconditionUnmet, match="nothing to evaluate"):
        _run(broken, loaded)


def test_a_missing_statistics_file_raises_with_the_trainers_own_message(config, loaded) -> None:
    """Reused, not copied: the actionable text names the generator command and its trim, and it
    must never drift from the training entry point's."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["stat_path"] = None

    with pytest.raises(EvalPreconditionUnmet, match="normalization is silently"):
        _run(broken, loaded)


def test_an_untrimmed_grid_is_refused(config, loaded) -> None:
    r"""The forecast of anchor $t$ starts at raw sample $16(t+1)$ only on the trimmed grid; on
    the untrimmed one it starts at $16(t+16)$, one full minute later, and nothing fails."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"] = 0.0

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(broken, loaded)

    assert "trim_minutes" in str(excinfo.value)
    assert "16*(t+1)" in str(excinfo.value)


def test_an_unnormalized_raw_target_is_refused(config, loaded) -> None:
    """Without ``'fhr'`` in ``normalize_fields`` nothing fails at all: the target arrives at
    ~140 bpm and the Gaussian NLL is computed against a z-scale variance model."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["dataloader_config"]["normalize_fields"] = ["up", "fhr_st"]

    with pytest.raises(EvalPreconditionUnmet, match="normalize_fields"):
        _run(broken, loaded)


def test_every_clinical_load_field_is_required_by_name(config, loaded) -> None:
    """The loader *skips* a requested field the shard does not carry, silently, so an absent
    ``target`` presents as "no classes found" rather than as a data problem."""
    kwargs = copy.deepcopy(config)["dataset_config"]["dataloader_config"]["dataset_kwargs"]
    assert set(preflight.REQUIRED_EVAL_LOAD_FIELDS) <= set(kwargs["load_fields"])

    for field in preflight.REQUIRED_EVAL_LOAD_FIELDS:
        broken = copy.deepcopy(config)
        fields = broken["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
        fields.remove(field)

        with pytest.raises(EvalPreconditionUnmet) as excinfo:
            _run(broken, loaded)

        assert field in str(excinfo.value)


def test_the_width_check_reads_the_test_shards_not_the_training_ones(config, loaded) -> None:
    """The reused guard reads ``vae_train_datasets`` and returns silently when the key is absent,
    so on an eval config it would check nothing at all -- and what it would have checked is the
    wrong population anyway."""
    view = preflight.config_view_for_shard_guards(config, loaded["model"])

    assert view["dataset_config"]["vae_train_datasets"] == config["dataset_config"][
        "vae_test_datasets"
    ]
    # The model's widths, not the config's: eval rebuilds from the checkpoint's model_kwargs.
    assert view["model_config"]["VAE_model"]["c_y"] == int(loaded["model"].c_y)
    assert view["model_config"]["VAE_model"]["c_u"] == int(loaded["model"].c_u)


def test_a_width_mismatch_against_the_shard_is_refused(config, loaded, monkeypatch) -> None:
    """A model whose declared widths disagree with the shards fails here rather than inside the
    forward, where the channel error names neither the checkpoint nor the config."""
    class _Narrow:
        c_y, c_u, use_up_st = 7, 5, True

    with pytest.raises(EvalPreconditionUnmet, match="channel widths disagree"):
        preflight.run_preflight(
            config=config,
            model=_Narrow(),
            checkpoint_path=loaded["checkpoint"],
            model_kwargs=loaded["model_kwargs"],
            hyper_parameters=loaded["hyper_parameters"],
            binding=run_module.RWS_BINDING,
        )


# =============================================================================
# Reconciliation against the checkpoint
# =============================================================================
def test_a_config_contradicting_the_checkpoints_geometry_is_refused(config, loaded) -> None:
    """The architecture is rebuilt from the checkpoint's own ``model_kwargs``, so the checkpoint
    always wins -- and the config's number would be reported beside weights it did not produce."""
    broken = copy.deepcopy(config)
    broken["model_config"]["VAE_model"]["d_z"] = int(loaded["model_kwargs"]["d_z"]) + 1

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        _run(broken, loaded)

    message = str(excinfo.value)
    assert "d_z" in message
    assert str(loaded["model_kwargs"]["d_z"]) in message


def test_a_config_contradicting_the_trained_objective_is_refused(config, loaded) -> None:
    """Scoring an ``mse`` run under a Gaussian NLL reports a different objective's numbers."""
    broken = copy.deepcopy(config)
    trained = loaded["hyper_parameters"]["likelihood"]
    broken["model_config"]["VAE_model"]["likelihood"] = (
        "gaussian_nll" if trained == "mse" else "mse"
    )

    with pytest.raises(EvalPreconditionUnmet, match="likelihood"):
        _run(broken, loaded)


def test_the_beta_schedule_is_recorded_but_not_reconciled(config, loaded) -> None:
    r"""$\beta$ and its ramp weight the training total only; no evaluated readout applies them,
    so a schedule edited after the fit is not a reason to refuse the run. Recorded so a reader
    can still see what the run trained under."""
    edited = copy.deepcopy(config)
    edited["model_config"]["VAE_model"]["beta_schedule"] = {"kind": "constant"}

    record = _run(edited, loaded)["checks"]["config_matches_checkpoint"]

    assert "beta_schedule" not in record["compared"]
    assert "beta_schedule" in record["not_compared"]


def test_a_key_the_config_omits_defers_to_the_constructor(config, loaded) -> None:
    """The constructor owns every default, so an omitted key is deference, not contradiction."""
    trimmed = copy.deepcopy(config)
    trimmed["model_config"]["VAE_model"].pop("d_z", None)

    record = _run(trimmed, loaded)["checks"]["config_matches_checkpoint"]

    assert "d_z" not in record["compared"]
    assert record["passed"] is True


def test_the_compared_set_is_the_bindings_and_a_narrower_one_is_visible(config, loaded) -> None:
    """Which keys are reconciled is the binding's, not this module's, because a second
    architecture reconciles a different set -- its own encoder's, and not ``causal_norm``.

    A narrowed tuple must *narrow the comparison* rather than raise: the record then shows the
    dropped key absent from ``compared``, so a run that quietly stopped checking something is
    legible in its own artifact instead of passing indistinguishably from one that checked it.
    """
    full = preflight.reconcile_with_checkpoint(
        config,
        model_kwargs=loaded["model_kwargs"],
        hyper_parameters=loaded["hyper_parameters"],
        geometry_keys=run_module.RWS_BINDING.geometry_keys,
    )
    narrowed = preflight.reconcile_with_checkpoint(
        config,
        model_kwargs=loaded["model_kwargs"],
        hyper_parameters=loaded["hyper_parameters"],
        geometry_keys=tuple(
            key for key in run_module.RWS_BINDING.geometry_keys if key != "d_z"
        ),
    )

    assert "d_z" in full["compared"]
    assert narrowed["passed"] is True
    assert "d_z" not in narrowed["compared"]
    # Only that key moved: the objective keys are reconciled from their own tuple and are
    # untouched by a change to the geometry set.
    assert set(full["compared"]) - set(narrowed["compared"]) == {"d_z"}


def test_a_key_the_binding_drops_is_no_longer_refused(config, loaded) -> None:
    """The other direction, and the one that matters for a second model: a config contradicting
    a key the binding does not carry must pass, because that key is not part of this
    architecture's geometry at all."""
    broken = copy.deepcopy(config)
    broken["model_config"]["VAE_model"]["d_z"] = int(loaded["model_kwargs"]["d_z"]) + 1

    with pytest.raises(EvalPreconditionUnmet, match="d_z"):
        preflight.reconcile_with_checkpoint(
            broken,
            model_kwargs=loaded["model_kwargs"],
            hyper_parameters=loaded["hyper_parameters"],
            geometry_keys=run_module.RWS_BINDING.geometry_keys,
        )

    assert preflight.reconcile_with_checkpoint(
        broken,
        model_kwargs=loaded["model_kwargs"],
        hyper_parameters=loaded["hyper_parameters"],
        geometry_keys=tuple(
            key for key in run_module.RWS_BINDING.geometry_keys if key != "d_z"
        ),
    )["passed"] is True


# =============================================================================
# Weight-space load verification
# =============================================================================
def test_a_freshly_constructed_model_is_refused(loaded) -> None:
    """Every witness tensor is still at the value the constructor gave it, which no trained model
    produces."""
    torch.manual_seed(0)
    fresh = SeqVaeLagAttnRws(**TINY_KWARGS)

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.verify_weights_loaded(fresh)

    message = str(excinfo.value)
    assert "still exactly at the value the constructor gave it" in message
    # Both witnesses reported, so a reader can see the check was not vacuous.
    assert "delta_heads" in message and "film_generators" in message


def test_a_freshly_constructed_model_with_horizon_attention_is_refused_too() -> None:
    """The refusal must survive the witness set growing. The attention's residual gains start at a
    *nonzero* constant, so a witness that reported $\\max|w|$ rather than the deviation from that
    constant would carry "evidence" on every model ever built -- and this check could then never
    fire again.
    """
    torch.manual_seed(0)
    fresh = SeqVaeLagAttnRws(**dict(TINY_KWARGS, horizon_attention_blocks=2))

    with pytest.raises(EvalPreconditionUnmet) as excinfo:
        preflight.verify_weights_loaded(fresh)

    assert "horizon_attention_gains" in str(excinfo.value)


def test_the_loaded_checkpoint_passes_and_names_the_witness(loaded) -> None:
    """The shipped configuration builds the horizon attention, so the record carries all three
    groups. The third one is conditional on the model rather than always present: a blockless core
    reports two, because an always-zero column in every such run's record would say less than its
    absence does. Both halves are asserted here so the conditionality is pinned in one place."""
    record = preflight.verify_weights_loaded(loaded["model"])

    assert record["passed"] is True
    assert record["witnesses_with_evidence"], "no witness carried evidence"
    assert loaded["model"].horizon_core.attention is not None
    assert set(record["max_abs_weight"]) == {
        "delta_heads",
        "film_generators",
        "horizon_attention_gains",
    }

    torch.manual_seed(0)
    blockless = SeqVaeLagAttnRws(**TINY_KWARGS)
    assert blockless.horizon_core.attention is None
    assert set(preflight.load_witnesses(blockless)) == {"delta_heads", "film_generators"}


def test_a_trained_attention_gain_is_itself_evidence_of_a_load() -> None:
    """The positive half. A gain that has moved off its construction constant can only have been
    moved by training, so it is a load witness in its own right -- the one the two existing groups
    cannot supply for a run whose posterior and FiLM path both stayed put."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(TINY_KWARGS, horizon_attention_blocks=2))
    assert model.horizon_core.attention is not None
    with torch.no_grad():
        for block in model.horizon_core.attention:
            block.residual_gain.add_(0.25)

    record = preflight.verify_weights_loaded(model)

    assert record["witnesses_with_evidence"] == ["horizon_attention_gains"]
    assert record["max_abs_weight"]["delta_heads"] == 0.0
    assert record["max_abs_weight"]["horizon_attention_gains"] == pytest.approx(0.25)


def test_a_model_perturbed_only_through_its_posterior_still_passes(loaded) -> None:
    """The reason the rule is any-of. Training moves the delta heads and the FiLM generators
    independently, so a real checkpoint whose FiLM path never left zero is an ordinary model --
    and an all-of rule would refuse it, along with this repository's own test fixture."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**TINY_KWARGS)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)

    record = preflight.verify_weights_loaded(model)

    assert record["witnesses_with_evidence"] == ["delta_heads"]
    assert record["max_abs_weight"]["film_generators"] == 0.0


# =============================================================================
# Causality and reach disclosure
# =============================================================================
def test_the_disclosure_is_assembled_exactly_as_it_is_written_down(config, loaded) -> None:
    """The whole record, against a dict written out by hand: key for key, value for value and in
    order. The encoder's half arrives through a callable rather than being read inline, and an
    extraction that changed *what* the record says -- or where a key sits in it -- would be
    invisible to every assertion that reads one key at a time.
    """
    model = loaded["model"]
    horizon_seconds = float(model.horizon) * 4.0
    per_block = preflight.channels_reading_past_the_horizon(horizon_seconds)
    expected = {
        "not_causal": True,
        "statement": preflight.NOT_CAUSAL_STATEMENT,
        "max_channel_reach_s": max(entry["max_reach_s"] for entry in per_block.values()),
        "causal_reach_budget_s": config["model_config"]["VAE_model"]["causal_reach_budget_s"],
        "causal_norm": bool(model.causal_norm),
        "n_causalized_norms": int(model.n_causalized_norms),
        "source_delay_steps": int(model.source_delay_steps),
        "source_delay_seconds": int(model.source_delay_steps) * 4.0,
        "source_delay_is_max_over_channels": True,
        "horizon_seconds": horizon_seconds,
        "channels_reading_past_the_horizon": per_block,
    }

    record = preflight.causality_disclosure(config, model)

    assert record == expected
    assert list(record) == list(expected)
    # And the same record when the callable is passed explicitly, which is how a run reaches it.
    assert (
        preflight.causality_disclosure(config, model, preflight.rws_encoder_disclosure) == expected
    )


def test_the_encoder_half_is_this_encoders_and_carries_its_consequence(loaded) -> None:
    """``causal_norm`` is a property of the recurrent encoder rather than of the feature bank, so
    it and its consequence sentence live in the callable the binding supplies. The warning is
    part of the disclosure: a run whose prior conditions on its own future must say so in the log
    as well as in the record."""

    class _Pooling:
        causal_norm, n_causalized_norms = False, 0

    assert preflight.rws_encoder_disclosure(loaded["model"]) == {
        "causal_norm": bool(loaded["model"].causal_norm),
        "n_causalized_norms": int(loaded["model"].n_causalized_norms),
    }

    pooling = preflight.rws_encoder_disclosure(_Pooling())
    assert pooling["causal_norm"] is False
    assert "p(z_t | Y_<=t) conditions on Y_>t" in pooling["causal_norm_consequence"]


def test_the_consequence_is_logged_and_not_only_recorded(loaded) -> None:
    """An operator reading the console must be told; a sentence only in ``preflight.json`` is a
    sentence nobody sees until after the run."""

    class _Pooling:
        causal_norm, n_causalized_norms = False, 0

    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        preflight.rws_encoder_disclosure(_Pooling())
    finally:
        logger.remove(sink)

    assert any("causal_norm=False" in message for message in warnings)


def test_a_binding_may_disclose_something_else_entirely(config, loaded) -> None:
    """What makes the seam a seam: an encoder with no ``causal_norm`` discloses its own facts and
    the record carries no key that means nothing for it. The bank-side half is unchanged, because
    it describes the dataset rather than the encoder."""
    record = preflight.causality_disclosure(
        config, loaded["model"], lambda model: {"time_pooling_norms": 0}
    )

    assert record["time_pooling_norms"] == 0
    assert "causal_norm" not in record
    assert "n_causalized_norms" not in record
    assert record["statement"] == preflight.NOT_CAUSAL_STATEMENT
    assert record["channels_reading_past_the_horizon"]


def test_the_disclosure_records_the_guard_the_model_actually_carries(config, loaded) -> None:
    record = _run(config, loaded)["causality"]
    model = loaded["model"]

    assert record["not_causal"] is True
    assert record["causal_norm"] == bool(model.causal_norm)
    assert record["n_causalized_norms"] == int(model.n_causalized_norms)
    assert record["source_delay_steps"] == int(model.source_delay_steps)
    assert record["source_delay_is_max_over_channels"] is True
    assert record["causal_reach_budget_s"] == config["model_config"]["VAE_model"][
        "causal_reach_budget_s"
    ]


def test_the_refusal_sentence_names_the_thing_the_readout_is_not(config, loaded) -> None:
    record = _run(config, loaded)["causality"]
    blocks = channel_reach.block_reach_seconds()

    assert "NOT a transfer entropy" in record["statement"]
    # The statement points at a recorded number rather than restating one. The figure quoted
    # elsewhere in this repository is already stale against the bank that ships, which is why.
    assert "max_channel_reach_s" in record["statement"]
    assert record["max_channel_reach_s"] == pytest.approx(
        max(max(reaches) for reaches in blocks.values())
    )
    assert record["max_channel_reach_s"] > record["horizon_seconds"]


def test_the_channel_counts_are_recomputed_rather_than_stored(config, loaded) -> None:
    """Recomputed from the production filter bank on every run, so a bank change moves them. A
    stored constant would keep reporting the old bank's answer indefinitely."""
    record = _run(config, loaded)["causality"]
    horizon_seconds = record["horizon_seconds"]
    blocks = channel_reach.block_reach_seconds()

    assert set(record["channels_reading_past_the_horizon"]) == set(blocks)
    for name, reaches in blocks.items():
        counted = record["channels_reading_past_the_horizon"][name]
        assert counted["n_over_horizon"] == sum(1 for r in reaches if r > horizon_seconds)
        assert counted["n_channels"] == len(reaches)
    # Non-vacuity: on the shipped bank some channels genuinely do read past the horizon.
    assert any(
        entry["n_over_horizon"] > 0
        for entry in record["channels_reading_past_the_horizon"].values()
    )


def test_a_finite_budget_records_its_delay_and_the_surviving_channel_counts(config) -> None:
    """The disclosure stands at every budget -- the reach it prunes on is a 95%-energy quantile
    rather than a hard support -- so what a budget changes is recorded, not the verdict."""
    from teb_vae.lag_attn.channel_reach import resolve_stream_budgets

    budget = resolve_stream_budgets(
        {"causal_reach_budget_s": 120.0, "use_up_st": True, "warmup_period": 30,
         "c_y": 109, "c_u": 58}
    )
    torch.manual_seed(0)
    guarded = SeqVaeLagAttnRws(
        **dict(
            TINY_KWARGS,
            sequence_length=64,
            warmup_period=30,
            target_keep_index=budget.target_keep_index,
            target_delays=budget.target_delays,
            source_keep_index=budget.source_keep_index,
            source_delays=budget.source_delays,
        )
    )
    budgeted = copy.deepcopy(config)
    budgeted["model_config"]["VAE_model"]["causal_reach_budget_s"] = 120.0

    record = preflight.causality_disclosure(budgeted, guarded)

    assert record["not_causal"] is True
    assert record["causal_reach_budget_s"] == 120.0
    assert record["source_delay_steps"] == budget.max_delay > 0
    assert record["source_delay_seconds"] == pytest.approx(budget.max_delay * 4.0)


# =============================================================================
# The record on disk
# =============================================================================
def test_the_record_is_written_and_is_readable_json(config, loaded, tmp_path) -> None:
    path = preflight.write_preflight(_run(config, loaded), tmp_path)

    written = json.loads(Path(path).read_text(encoding="utf-8"))
    assert path.name == preflight.PREFLIGHT_FILENAME
    assert written["checks"]["weights_loaded"]["passed"] is True
    assert written["causality"]["statement"] == preflight.NOT_CAUSAL_STATEMENT


# =============================================================================
# Wired into the run
# =============================================================================
def test_a_refused_run_leaves_no_summary(trained_run, tmp_path) -> None:
    """The property that makes the refusal worth having: a rejected input must not produce a
    file that reads like a result. Preflight therefore runs outside every failure-isolating
    wrapper, and before anything is scored.

    The run directory itself *is* created first, holding the merged config and the log carrying
    the refusal -- a refused run leaves its inputs, so an operator can read afterwards what was
    rejected and why. What it must not leave is a summary.
    """
    overrides = tmp_path / "still_placeholders.yaml"
    overrides.write_text(
        yaml.safe_dump(
            {
                "dataset_config": {
                    "vae_test_datasets": [
                        "/data1/REPOINT_ME_new_channel_selection/"
                        "k_fold_cross_validation_dataset/test/hie_cs.hdf5"
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    with pytest.raises(EvalPreconditionUnmet, match=preflight.REPOINT_MARKER):
        run_module.main(trained_run, output_dir, overrides=overrides, device="cpu")

    assert list(output_dir.rglob(run_module.SUMMARY_FILENAME)) == []
    assert list(output_dir.rglob(run_module.LOG_FILENAME)), "the refusal is not readable anywhere"


def test_a_completed_run_carries_the_preflight_and_the_disclosure(evaluated) -> None:
    summary = evaluated["summary"]

    assert (evaluated["results_dir"] / preflight.PREFLIGHT_FILENAME).is_file()
    assert summary["preflight"]["checks"]["weights_loaded"]["passed"] is True
    # Promoted out of the preflight block as well: a reader who opens only the summary must see
    # what the readout is not.
    assert summary["causality"]["statement"] == preflight.NOT_CAUSAL_STATEMENT
    assert summary["eval_config"]["seed"] == 42
