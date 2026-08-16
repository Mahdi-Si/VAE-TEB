r"""The pre-flight refusals, and the failure each one moves out of a multi-day run.

Every refusal here guards a launch whose symptom is a **number** rather than an exception. That is
what distinguishes them from the shared entry point's four guards, which catch missing files and
mismatched widths -- things that fail loudly and early on their own. These do not:

* a two-sided shard shares every field name and every dtype with a causal one, so a run against it
  produces a complete metrics history describing a model whose inputs contain their own future --
  which is the *only* thing separating this cell from the model it is compared against;
* an anchor floor below the pairing decodes anchors whose inputs are still partly assumed
  pre-recording history, with every shape correct and every number finite;
* ``lambda_boundary`` over a tiled anchor set joins two windows a whole horizon apart and calls the
  discontinuity between them an error;
* a ``load_fields`` without ``guid`` or ``epoch`` leaves every segment on one tile grid forever, and
  $A_{\max}$ is a geometry constant either way so no shape, no count and no metric differs;
* a ``load_fields`` without ``weight`` leaves the raw target's gaps -- stored as $0$ bpm, about
  $-11\sigma$ once z-scored -- scored as signal at full weight;
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
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

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
    LagAttnCrwsTrainer.preflight(config)


# --------------------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------------------
def test_a_two_sided_shard_is_refused(config):
    """The load-bearing refusal of this cell, because one-sidedness of the inputs is the ONLY thing
    that distinguishes it from the model it is compared against. The two dataset variants share
    every field name and every dtype; only the root ``transform`` attribute and the channel counts
    tell them apart, and a rebuilt two-sided dataset at these widths would pass the width check and
    train to completion on inputs that contain the answer."""
    config["dataset_config"]["vae_train_datasets"] = [str(TWO_SIDED_SHARD)]

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

    assert "causal" in str(excinfo.value)


def test_a_reach_budget_beside_the_warm_up_budget_is_refused_naming_both_keys(config):
    """Not a stricter run but an incoherent one: the forward reach $L_{95}$ is an energy quantile of
    a two-sided kernel, measured on a bank that did not produce these coefficients -- and a reach
    budget applies a *shift* on top of the warm-up."""
    _vae(config)["causal_reach_budget_s"] = 120.0

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "causal_warmup_budget_steps" in message
    assert "causal_reach_budget_s" in message


def test_a_trim_that_does_not_produce_the_declared_window_is_refused_naming_both_paths(config):
    """The warm-up vectors are rebased by the trim, so a uniformly wrong rebase moves the anchor
    floor and the input-validity boundary **together** -- the declared policy would still read as
    satisfied while every kept channel was cold across the scored window."""
    _loader(config)["dataset_kwargs"]["trim_minutes"] = 2.0

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "trim_minutes" in message
    assert "sequence_length" in message


# --------------------------------------------------------------------------------------
# The geometry
# --------------------------------------------------------------------------------------
def test_a_floor_that_does_not_pair_with_the_budget_is_refused_naming_both_numbers(config):
    r"""$F \ge B - 1$ over the **survivors**, checked here so a mis-paired configuration fails before
    a run directory exists rather than inside the constructor after every rank initialised."""
    _vae(config)["warmup_period"] = 132

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

    message = str(excinfo.value)
    assert "warmup_period=132" in message
    assert "134" in message and "133" in message


def test_the_floor_refusal_is_the_constructors_own_and_states_the_input_warmth_policy(config):
    """Delegated rather than restated, so the pre-flight and the constructor cannot come to disagree
    about a policy that has exactly one expression. What the message must say is what this cell
    actually enforces: the raw target is honest at every step, so a lower floor would not corrupt the
    objective -- it would decode anchors whose *inputs* are still partly pre-recording history, which
    is a different claim about the run rather than a wrong number in it."""
    _vae(config)["warmup_period"] = 100

    with pytest.raises(ValueError) as preflight_error:
        LagAttnCrwsTrainer.preflight(config)
    with pytest.raises(ValueError) as constructor_error:
        CausalRawInputs._check_anchor_floor(100, (0, 134))

    assert str(preflight_error.value) == str(constructor_error.value)
    assert "input-warmth" in str(preflight_error.value)
    assert "raw target is honest at every step" in str(preflight_error.value)


def test_a_higher_floor_than_the_pairing_requires_is_admitted(config):
    """The pairing is an inequality, not an equality, which is what would make a stricter policy arm
    a one-key change rather than a code change."""
    _vae(config)["warmup_period"] = 150

    LagAttnCrwsTrainer.preflight(config)


def test_a_stride_that_leaves_a_phase_with_no_anchor_is_refused(config):
    """At the last phase the first anchor would be $F + S - 1$; if that anchor does not exist there
    is a phase at which the sample contributes no forecast at all, and its share of the epoch is
    silently dropped."""
    _vae(config)["warmup_period"] = 280
    _vae(config)["anchor_stride"] = 15

    with pytest.raises(ValueError, match="phase"):
        LagAttnCrwsTrainer.preflight(config)


@pytest.mark.parametrize("weight", [0.1, 1.0])
def test_a_non_zero_boundary_weight_is_refused_unconditionally(config, weight):
    """The term is a slicing identity over ADJACENT anchors. This cell always supplies an anchor set
    whose entries are $S$ apart, so at any non-zero weight it would score the gap between two windows
    a whole horizon apart as an error -- and the shared objective's own refusal fires inside the
    first training step, after the run directory exists.

    Worth reading twice here rather than in the feature-target cells: the term is *meaningful* on a
    raw target, and it ships at $0.05$ on the model this one is compared against. It is off here
    because of the anchor axis alone."""
    _vae(config)["lambda_boundary"] = weight

    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnCrwsTrainer.preflight(config)


def test_the_boundary_refusal_does_not_depend_on_the_stride(config):
    """Unconditional rather than conditional on the tiling. At ``anchor_stride: 1`` the slicing
    identity is legal again, so a conditional refusal would make the term's meaning a function of
    another key -- and a run that later moved the stride would silently start scoring the gap
    between two windows a horizon apart."""
    _vae(config)["anchor_stride"] = 1
    _vae(config)["lambda_boundary"] = 0.05

    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnCrwsTrainer.preflight(config)


def test_the_two_shape_terms_that_do_transfer_are_not_refused(config):
    """The other direction, and it is the reason the boundary refusal is stated per key rather than
    over the group: the multiscale $L_1$ and the derivative Huber are raw-waveform quantities over
    one anchor's own block, so a tiled anchor axis does not touch them and they ship at the
    comparison model's weights."""
    _vae(config)["lambda_ms"] = 0.1
    _vae(config)["lambda_deriv"] = 0.1

    LagAttnCrwsTrainer.preflight(config)


def test_dropping_the_first_source_block_beside_a_budget_is_refused_naming_both_keys(config):
    """``use_up_st: false`` is not a width change here. ``up_st`` is the block that reaches warm-up
    zero, so without it the source's minimum wait is $41$ and the availability adapter's start
    embedding comes into existence -- a parameter reached only by the leading steps of a segment,
    which under ``find_unused_parameters=False`` is a DDP hazard."""
    _vae(config)["use_up_st"] = False
    _vae(config)["c_u"] = 15

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

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
        LagAttnCrwsTrainer.preflight(config)

    assert field in str(excinfo.value)
    assert "load_fields" in str(excinfo.value)


@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_the_cross_channel_block_is_refused_in_either_list(config, list_key):
    """A ``fhr_up_ph`` coefficient mixes both signals, so it would put the source's own signal into
    the target-only branch's inputs -- and the causal variant does not store it at all. In
    ``load_fields`` the loader would raise, but only after every rank initialised; in
    ``normalize_fields`` it is silently ignored and reads as though the block were being handled."""
    fields = (
        _loader(config)["dataset_kwargs"]["load_fields"]
        if list_key == "load_fields"
        else _loader(config)["normalize_fields"]
    )
    fields.append("fhr_up_ph")

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

    assert "fhr_up_ph" in str(excinfo.value)
    assert list_key in str(excinfo.value)


@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_the_target_field_is_refused_missing_from_either_list(config, list_key):
    """Parameterised on this driver's ``TARGET_FIELDS`` rather than on the literal ``'fhr'``, and the
    rule itself is the shared entry point's -- **called** here rather than copied, so an unloaded
    target field and an unnormalised one have one expression in the repository. The pre-flight runs
    it too so that ``preflight`` is a complete statement of what this cell cannot serve, rather than
    a list whose gaps happen to be filled by the caller's own ordering."""
    for field in LagAttnCrwsTrainer.TARGET_FIELDS:
        broken = copy.deepcopy(config)
        fields = (
            _loader(broken)["dataset_kwargs"]["load_fields"]
            if list_key == "load_fields"
            else _loader(broken)[list_key]
        )
        fields.remove(field)

        with pytest.raises(ValueError, match=rf"'{field}' in .*{list_key}"):
            LagAttnCrwsTrainer.preflight(broken)


def test_a_load_fields_without_the_validity_signal_is_refused(config):
    """The half no shared guard covers, because ``weight`` is not a field the target is *built*
    from. It is the only trustworthy gap signal for a raw target -- gaps are stored as $0$ bpm, which
    after z-scoring is about $-11\\sigma$ rather than a detectable sentinel -- so a run without it
    scores pad as signal at full weight and every mask in the objective silently reads all-valid."""
    _loader(config)["dataset_kwargs"]["load_fields"].remove("weight")

    with pytest.raises(ValueError) as excinfo:
        LagAttnCrwsTrainer.preflight(config)

    assert "weight" in str(excinfo.value)
    assert "load_fields" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Nothing is written before a refusal
# --------------------------------------------------------------------------------------
def test_a_refusal_leaves_no_run_directory(tmp_path, config):
    """The whole point of the hook's position in the entry point: on a multi-rank launch a failure
    after ``setup_config`` has already created directories, opened log sinks and started an MLflow
    run on every rank."""
    from teb_vae.lag_attn_crws import trainer as trainer_module

    runs = tmp_path / "runs"
    config["general_config"]["folders_config"]["out_dir_base"] = str(runs)
    _vae(config)["lambda_boundary"] = 1.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="lambda_boundary"):
        trainer_module.main(str(path))

    assert not runs.exists()
