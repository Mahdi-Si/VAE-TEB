"""Preflight refuses exactly the runs that would produce plausible numbers and mean nothing.

The three-way distinction in the last section is the point of the file: an unloaded model must
raise, a fully perturbed model must pass, and a model perturbed *only* through its posterior
must pass the weight-space check while still emitting the behavioural warning. Without the
third case the two checks could collapse into one and nothing would notice.
"""
from __future__ import annotations

import copy
import json

import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn.eval import preflight
from teb_vae.lag_attn.eval.runner import EvalRunner
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from train.data_module import GraphDataModule


@pytest.fixture
def config(repo_root):
    """The merged test-suite eval config."""
    return load_config(str(repo_root / EVAL_TINY_CONFIG))


@pytest.fixture
def runner(tiny_checkpoint, tmp_path) -> EvalRunner:
    return EvalRunner.from_checkpoint(tiny_checkpoint, tmp_path / "run", device="cpu")


@pytest.fixture
def batch(config, runner, monkeypatch, repo_root):
    """One real batch off the committed shard, on the runner's device."""
    monkeypatch.chdir(repo_root)
    loader = GraphDataModule(config).test_dataloader()
    return runner.to_device(preflight.first_batch(loader))


# ---------------------------------------------------------------------------
# S1-T06: the hard guards
# ---------------------------------------------------------------------------
def test_passes_on_a_well_formed_run(config, runner, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    record = preflight.run_preflight(config=config, runner=runner)
    assert all(check["passed"] for check in record["checks"].values())


def test_repoint_placeholder_raises_before_the_missing_file_does(config, runner):
    """Otherwise the failure reads as a missing file and the operator goes looking for one."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["vae_test_datasets"] = [
        "/data1/.../REPOINT_ME_new_channel_selection/k_fold/test/hie_cs.hdf5"
    ]
    with pytest.raises(ValueError, match="REPOINT_ME"):
        preflight.run_preflight(config=broken, runner=runner)


def test_missing_stat_path_raises_with_the_trainer_message(config, runner):
    """Reused, not copied: the actionable text must never drift from the trainer's."""
    broken = copy.deepcopy(config)
    broken["dataset_config"]["stat_path"] = None
    with pytest.raises(ValueError, match="normalization is silently disabled"):
        preflight.run_preflight(config=broken, runner=runner)


def test_nonexistent_stat_path_raises(config, runner):
    broken = copy.deepcopy(config)
    broken["dataset_config"]["stat_path"] = "teb_vae/lag_attn/tests/fixtures/not_there.hdf5"
    with pytest.raises(ValueError, match="does not exist"):
        preflight.run_preflight(config=broken, runner=runner)


def test_widths_are_compared_against_the_model_not_the_config(
    config, tmp_path, monkeypatch, repo_root
):
    """A checkpoint whose geometry differs from the config must fail here, not in the forward.

    The config's own ``c_y`` is left correct on purpose: what is wrong is the *checkpoint*, and
    a guard that compared the config against the shard would happily pass this.
    """
    monkeypatch.chdir(repo_root)
    from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_KWARGS, build_tiny_checkpoint_blob

    blob = build_tiny_checkpoint_blob(dict(SHIPPED_KWARGS, c_y=44))
    path = tmp_path / "narrow.ckpt"
    torch.save(blob, path)
    narrow_runner = EvalRunner.from_checkpoint(path, tmp_path / "run", device="cpu")

    assert config["model_config"]["VAE_model"]["c_y"] == 109
    with pytest.raises(ValueError, match="c_y=44"):
        preflight.run_preflight(config=config, runner=narrow_runner)


def test_objective_disagreement_raises(config, runner, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    broken = copy.deepcopy(config)
    broken["model_config"]["VAE_model"]["free_bits"] = 0.75
    with pytest.raises(ValueError, match="free_bits"):
        preflight.run_preflight(config=broken, runner=runner)


def test_preconditions_are_recorded_with_their_consequences(config, runner, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    record = preflight.run_preflight(config=config, runner=runner)
    preconditions = record["preconditions"]
    assert preconditions["causal_norm"]["value"] is True
    assert preconditions["causal_norm"]["blocks"] == []
    assert preconditions["head_structured_latent"]["value"] is True
    assert preconditions["kld_support"]["value"] == "anchor"


def test_causal_norm_false_blocks_te_readouts_without_failing_the_run(tmp_path):
    """Recorded, not raised: the forecast and calibration analyses stay perfectly valid."""
    from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_KWARGS, build_tiny_checkpoint_blob

    blob = build_tiny_checkpoint_blob(dict(SHIPPED_KWARGS, causal_norm=False))
    path = tmp_path / "acausal.ckpt"
    torch.save(blob, path)
    runner = EvalRunner.from_checkpoint(path, tmp_path / "run", device="cpu")

    preconditions = preflight.interpretation_preconditions(runner)
    assert preconditions["causal_norm"]["value"] is False
    assert "te_lag_map" in preconditions["causal_norm"]["blocks"]
    assert "transfer-entropy surrogate" in preconditions["causal_norm"]["consequence"]


def test_head_structured_latent_false_blocks_only_the_per_head_decomposition(tmp_path):
    from teb_vae.lag_attn.eval.tests.conftest import PROD_KWARGS, build_tiny_checkpoint_blob

    blob = build_tiny_checkpoint_blob(dict(PROD_KWARGS, head_structured_latent=False))
    path = tmp_path / "flat.ckpt"
    torch.save(blob, path)
    runner = EvalRunner.from_checkpoint(path, tmp_path / "run", device="cpu")

    preconditions = preflight.interpretation_preconditions(runner)
    assert preconditions["head_structured_latent"]["blocks"] == ["per_head_kl_decomposition"]
    # The TE readouts themselves are untouched: causal_norm is what gates those.
    assert preconditions["causal_norm"]["blocks"] == []


def test_preflight_json_lists_every_check(config, runner, tmp_path, monkeypatch, repo_root):
    monkeypatch.chdir(repo_root)
    record = preflight.run_preflight(config=config, runner=runner)
    path = preflight.write_preflight(record, tmp_path / "out")

    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written["checks"]) == {
        "repoint_placeholder",
        "stat_path",
        "declared_widths",
        "objective_matches_config",
        "weights_loaded",
    }
    assert written["geometry"]["c_y"] == 109
    assert written["objective"]["likelihood"] == "gaussian_nll"
    assert written["model_kwargs"]


# ---------------------------------------------------------------------------
# S1-T07: weight-space verification versus the behavioural probe
# ---------------------------------------------------------------------------
def test_a_fresh_model_fails_the_weight_space_check(shipped_kwargs):
    """The only signal that separates "never loaded" from "loaded and collapsed"."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**shipped_kwargs)
    with pytest.raises(RuntimeError, match="still exactly zero") as excinfo:
        preflight.verify_weights_loaded(model)
    assert "load_checkpoint_strict returns None" in str(excinfo.value)


def test_a_loaded_model_passes_the_weight_space_check(runner):
    record = preflight.verify_weights_loaded(runner.model)
    assert record["passed"] is True
    assert any(value > 0.0 for value in record["max_abs_weight"].values())


def test_a_posterior_only_model_passes_the_weight_check_but_warns_behaviourally(
    shipped_kwargs, perturb_posterior, batch, tmp_path
):
    """The case that proves the two checks are genuinely distinct.

    Perturbing the posterior alone leaves ``residual_decoder.mean_head`` at zero, so
    ``delta_mu_src`` is identically zero regardless of $z$ -- the behavioural reading is a hard
    zero while the weight-space check passes on the strength of the posterior deltas.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**shipped_kwargs)
    perturb_posterior(model)

    assert preflight.verify_weights_loaded(model)["passed"] is True

    from teb_vae.lag_attn.eval.runner import EvalRunner as Runner

    posterior_runner = Runner(
        model=model,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        objective=_prod_objective(),
        checkpoint_path=tmp_path / "none.ckpt",
    )
    reading = preflight.probe_load_health(posterior_runner, batch, floor=0.01)
    assert reading["residual_ratio"] == 0.0
    assert reading["raised"] is False
    assert "warning" in reading


def test_health_probe_on_a_live_model_records_without_warning(runner, batch):
    reading = preflight.probe_load_health(runner, batch, floor=0.0)
    assert reading["residual_ratio"] > 0.0
    assert "warning" not in reading
    # Recorded but never gated on: under gaussian_nll with a learned sigma the full and
    # baseline losses use different variance heads and differ even at delta_mu_src == 0.
    assert "uplift_rel" in reading


def test_health_probe_floor_is_config_driven(runner, batch):
    lenient = preflight.probe_load_health(runner, batch, floor=0.0)
    strict = preflight.probe_load_health(runner, batch, floor=1.0e9)
    assert "warning" not in lenient
    assert "warning" in strict
    assert strict["raised"] is False, "the behavioural reading must never raise"


def _prod_objective():
    """The objective the suite's checkpoints are built with."""
    from teb_vae.lag_attn.eval.runner import Objective
    from teb_vae.lag_attn.tests.conftest import PROD_HPARAMS

    return Objective(**PROD_HPARAMS)
