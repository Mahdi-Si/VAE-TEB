r"""The pre-flight refusals, and the failure each one moves out of a multi-day run.

Every refusal here guards a launch whose symptom is a **number** rather than an exception. That is
what distinguishes them from the shared entry point's four guards, which catch missing files and
mismatched widths -- things that fail loudly and early on their own. These do not:

* a two-sided shard shares every field name and every dtype with a causal one, so a run against it
  produces a complete metrics history describing a model whose inputs contain their own future;
* an anchor floor one step below the pairing scores the assumed pre-recording history of the
  slowest kept channel as signal, with every shape correct and ``target_warm_frac`` still $1.0$;
* ``lambda_boundary`` over a tiled anchor set joins two windows a whole horizon apart and calls the
  discontinuity between them an error;
* a ``load_fields`` without ``guid`` or ``epoch`` leaves every segment on one tile grid forever,
  and $A_{\max}$ is a geometry constant either way so no shape, no count and no metric differs;
* ``use_up_st: false`` flips the availability adapter's start embedding into existence, which is a
  construction-time change and therefore a DDP hazard rather than a width change.

They live in ``preflight`` rather than in the trainer's body so they run **before** ``setup_config``
-- before the run directory, the log sinks and the MLflow run exist on every rank.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

from .conftest import TWO_SIDED_SHARD, absolutize_dataset_paths

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


@pytest.fixture(scope="module")
def shipped() -> dict:
    """The tiny config, path-absolutised, at the shipped geometry. Treat as read-only."""
    return absolutize_dataset_paths(load_config(str(_TINY)))


@pytest.fixture
def config(shipped) -> dict:
    """A fresh mutable copy of it."""
    return copy.deepcopy(shipped)


def _vae(config: dict) -> dict:
    return config["model_config"]["VAE_model"]


def _loader(config: dict) -> dict:
    return config["dataset_config"]["dataloader_config"]


# --------------------------------------------------------------------------------------
# The shipped configuration passes
# --------------------------------------------------------------------------------------
def test_the_shipped_configuration_is_admitted(config):
    """The other direction of every refusal below: a guard that refused everything would pass each
    of them and leave the package unlaunchable."""
    LagAttnCfsTrainer.preflight(config)


# --------------------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------------------
def test_a_two_sided_shard_is_refused(config):
    """The load-bearing refusal. The two dataset variants share every field name and every dtype;
    only the root ``transform`` attribute and the channel counts tell them apart, and a rebuilt
    two-sided dataset at these widths would pass the width check and train to completion on
    coefficients that contain their own future."""
    config["dataset_config"]["vae_train_datasets"] = [str(TWO_SIDED_SHARD)]

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    assert "causal" in str(excinfo.value)


def test_a_reach_budget_beside_the_warm_up_budget_is_refused_naming_both_keys(config):
    """Not a stricter run but an incoherent one: the forward reach $L_{95}$ is an energy quantile of
    a two-sided kernel, measured on a bank that did not produce these coefficients -- and a reach
    budget applies a *shift* on top of the warm-up."""
    _vae(config)["causal_reach_budget_s"] = 120.0

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "causal_warmup_budget_steps" in message
    assert "causal_reach_budget_s" in message


def test_a_trim_that_does_not_produce_the_declared_window_is_refused_naming_both_paths(config):
    """The one failure mode ``target_warm_frac`` is blind to: the warm-up vectors are rebased by the
    trim, so a uniformly wrong rebase moves the anchor floor and the validity boundary **together**
    and the metric still reads exactly $1.0$."""
    _loader(config)["dataset_kwargs"]["trim_minutes"] = 2.0

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "trim_minutes" in message
    assert "sequence_length" in message


# --------------------------------------------------------------------------------------
# The geometry
# --------------------------------------------------------------------------------------
def test_a_floor_that_does_not_pair_with_the_budget_is_refused_naming_both_numbers(config):
    r"""$F \ge \max(B - 1,\; \max_c(W'_c + d_c))$ over the **survivors**, checked here so a mis-paired
    configuration fails before a run directory exists rather than inside the constructor after every
    rank initialised.

    Which of the two halves binds is itself part of the message, and at the shipped configuration it
    is the second: the aligned inputs are honest at the anchor only from $B = 134$, so the floor is
    $134$ rather than the $133$ the scored-target half alone would admit. Asserted on the number
    *and* on the requirement that produced it, because the two halves differ by one step and a
    message naming the wrong one would still name a plausible integer."""
    _vae(config)["warmup_period"] = 132

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "warmup_period=132" in message
    assert "134" in message
    assert "shifted inputs" in message


def test_a_higher_floor_than_the_pairing_requires_is_admitted(config):
    """The pairing is an inequality, not an equality, which is what makes the ten-minute policy arm
    a one-key change rather than a code change."""
    _vae(config)["warmup_period"] = 150

    LagAttnCfsTrainer.preflight(config)


def test_a_stride_that_leaves_a_phase_with_no_anchor_is_refused(config):
    """At the last phase the first anchor would be $F + S - 1$; if that anchor does not exist there
    is a phase at which the sample contributes no forecast at all, and its share of the epoch is
    silently dropped."""
    _vae(config)["warmup_period"] = 280
    _vae(config)["anchor_stride"] = 15

    with pytest.raises(ValueError, match="phase"):
        LagAttnCfsTrainer.preflight(config)


@pytest.mark.parametrize("weight", [0.1, 1.0])
def test_a_non_zero_boundary_weight_is_refused_unconditionally(config, weight):
    """The term is a slicing identity over ADJACENT anchors. This family always supplies an anchor
    set whose entries are $S$ apart, so at any non-zero weight it would score the gap between two
    windows a whole horizon apart as an error -- and the shared objective's own refusal fires inside
    the first training step, after the run directory exists."""
    _vae(config)["lambda_boundary"] = weight

    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnCfsTrainer.preflight(config)


def test_dropping_the_first_source_block_beside_a_budget_is_refused_naming_both_keys(config):
    """``use_up_st: false`` is not a width change here. ``up_st`` is the block that reaches warm-up
    zero, so without it the source's minimum wait is $41$ and the availability adapter's start
    embedding comes into existence -- a parameter reached only by the leading steps of a segment,
    which under ``find_unused_parameters=False`` is a DDP hazard."""
    _vae(config)["use_up_st"] = False
    _vae(config)["c_u"] = 15

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "use_up_st" in message
    assert "causal_warmup_budget_steps" in message


# --------------------------------------------------------------------------------------
# The loader lists
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["guid", "epoch"])
def test_a_load_fields_missing_a_phase_key_is_refused_naming_the_field(config, field):
    """``load_fields`` is honoured literally with no forced additions. Without both keys the tile
    phase has nothing per-segment to key on and every segment is decoded at one grid forever."""
    _loader(config)["dataset_kwargs"]["load_fields"].remove(field)

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    assert field in str(excinfo.value)
    assert "load_fields" in str(excinfo.value)


@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_the_cross_channel_block_is_refused_in_either_list(config, list_key):
    """A ``fhr_up_ph`` coefficient mixes both signals, so it would put the source's own signal into
    the forecast target -- and the causal variant does not store it at all. In ``load_fields`` the
    loader would raise, but only after every rank initialised; in ``normalize_fields`` it is
    silently ignored and reads as though the block were being handled."""
    fields = (
        _loader(config)["dataset_kwargs"]["load_fields"]
        if list_key == "load_fields"
        else _loader(config)["normalize_fields"]
    )
    fields.append("fhr_up_ph")

    with pytest.raises(ValueError) as excinfo:
        LagAttnCfsTrainer.preflight(config)

    assert "fhr_up_ph" in str(excinfo.value)
    assert list_key in str(excinfo.value)


def test_the_target_normalisation_guard_is_the_shared_one_rather_than_a_copy(config):
    """The refusal ``fhr_st`` / ``fhr_ph`` missing from either list needs is already the shared entry
    point's, parameterised on this driver's ``TARGET_FIELDS`` and running *before* this hook. A
    second copy here would be a second rule that could come to disagree with it, so what is asserted
    is that the shared one covers the case rather than that this one does."""
    from teb_vae.lag_attn_rws.trainer import _check_raw_target_normalized

    _loader(config)["normalize_fields"].remove("fhr_st")

    LagAttnCfsTrainer.preflight(config)  # not this hook's job
    with pytest.raises(ValueError, match=r"'fhr_st'"):
        _check_raw_target_normalized(config, fields=LagAttnCfsTrainer.TARGET_FIELDS)


# --------------------------------------------------------------------------------------
# Nothing is written before a refusal
# --------------------------------------------------------------------------------------
def test_a_refusal_leaves_no_run_directory(tmp_path, config):
    """The whole point of the hook's position in the entry point: on a multi-rank launch a failure
    after ``setup_config`` has already created directories, opened log sinks and started an MLflow
    run on every rank."""
    from teb_vae.lag_attn_cfs import trainer as trainer_module

    runs = tmp_path / "runs"
    config["general_config"]["folders_config"]["out_dir_base"] = str(runs)
    _vae(config)["lambda_boundary"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="lambda_boundary"):
        trainer_module.main(str(path))

    assert not runs.exists()
