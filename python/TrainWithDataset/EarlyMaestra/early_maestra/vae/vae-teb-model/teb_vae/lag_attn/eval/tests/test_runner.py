"""The runner rebuilds the right model, under the right objective, and refuses everything else.

The negative cases carry most of the weight here. Each of them is a failure that would
otherwise produce a complete set of plausible numbers: a checkpoint that did not load, an
objective taken from the wrong place, a batch whose widths moved. None of them looks wrong in
the output.
"""
from __future__ import annotations

import copy

import pytest
import torch

from teb_vae.lag_attn.eval.runner import EvalRunner, Objective, future_target
from teb_vae.lag_attn.eval.tests.conftest import build_tiny_checkpoint_blob
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn


@pytest.fixture
def runner(tiny_checkpoint, tmp_path) -> EvalRunner:
    """A runner rebuilt from the session checkpoint, on CPU."""
    return EvalRunner.from_checkpoint(tiny_checkpoint, tmp_path / "run", device="cpu")


def _save(blob, tmp_path, name="mutated.ckpt"):
    """Write a checkpoint blob and return its path."""
    path = tmp_path / name
    torch.save(blob, path)
    return path


# ---------------------------------------------------------------------------
# S1-T01: rebuild and inference mode
# ---------------------------------------------------------------------------
def test_rebuilds_from_the_checkpoint_alone(runner, tiny_checkpoint):
    """No config file involved: the architecture comes out of ``model_kwargs``."""
    blob = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    reference = SeqVaeLagAttn(**blob["model_kwargs"])

    assert runner.model.d_model == reference.d_model
    assert runner.model.c_y == reference.c_y == 109
    assert runner.geometry()["num_heads"] == reference.lag_attn.num_heads
    # num_heads / d_head are NOT model attributes -- they must come from the attention module.
    assert not hasattr(runner.model, "num_heads")


def test_loaded_parameters_match_the_checkpoint(runner, tiny_checkpoint):
    """Every parameter, not merely a shape-compatible model."""
    blob = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    saved = {
        key[len("model."):]: value
        for key, value in blob["state_dict"].items()
        if key.startswith("model.")
    }
    for name, parameter in runner.model.state_dict().items():
        assert torch.equal(parameter, saved[name]), f"{name} did not load"


def test_mutated_model_class_raises_before_construction(tiny_checkpoint, tmp_path):
    """The guard must fire on ``model_class``, not on a downstream ``TypeError``."""
    blob = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    blob["model_class"] = "SomeOtherModel"
    with pytest.raises(ValueError, match="model_class"):
        EvalRunner.from_checkpoint(_save(blob, tmp_path), tmp_path / "run", device="cpu")


def test_empty_model_kwargs_raises(tmp_path):
    """SeqVaeLagAttn() is legal and builds production geometry, so this cannot be left silent."""
    blob = build_tiny_checkpoint_blob()
    blob["model_kwargs"] = {}
    with pytest.raises(RuntimeError, match="no 'model_kwargs'"):
        EvalRunner.from_checkpoint(_save(blob, tmp_path), tmp_path / "run", device="cpu")


def test_misaligned_state_dict_raises_rather_than_proceeding(tmp_path):
    """``load_checkpoint_strict`` returns None on failure; an unchecked call evaluates noise."""
    blob = build_tiny_checkpoint_blob()
    # A geometry the saved weights cannot fit: the rebuild succeeds, the load cannot.
    blob["model_kwargs"] = dict(blob["model_kwargs"], d_model=64, d_head=16)
    with pytest.raises(RuntimeError, match="could not align checkpoint"):
        EvalRunner.from_checkpoint(_save(blob, tmp_path), tmp_path / "run", device="cpu")


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EvalRunner.from_checkpoint(tmp_path / "nope.ckpt", tmp_path / "run", device="cpu")


def test_inference_mode_enters_no_grad_and_eval(runner):
    """Both halves: a live dropout breaks the attention row sums the TE readout depends on."""
    runner.model.train()
    with runner.inference_mode():
        assert not runner.model.training
        assert not torch.is_grad_enabled()
    assert runner.model.training, "the prior mode must be restored"


def test_inference_mode_restores_the_mode_on_exception(runner):
    """A mid-batch failure must not leave the model in eval for every step that follows."""
    runner.model.train()
    with pytest.raises(ValueError):
        with runner.inference_mode():
            raise ValueError("boom")
    assert runner.model.training
    assert torch.is_grad_enabled()


def test_geometry_comes_from_the_model_not_a_config(runner):
    geometry = runner.geometry()
    assert geometry["d_z"] == runner.model.d_z
    assert geometry["num_lags"] == runner.model.max_lag + 1
    assert geometry["causal_norm"] is True
    assert geometry["head_structured_latent"] is True


# ---------------------------------------------------------------------------
# S1-T02: the objective comes from the checkpoint
# ---------------------------------------------------------------------------
def test_objective_resolves_from_hyper_parameters(runner):
    """Under compute_loss's own defaults these would read 'mse' and 1.0."""
    from teb_vae.lag_attn.tests.conftest import PROD_HPARAMS

    objective = runner.objective
    assert objective.likelihood == PROD_HPARAMS["likelihood"] == "gaussian_nll"
    assert objective.sigma_obs == PROD_HPARAMS["sigma_obs"] == "learned"
    assert objective.free_bits == PROD_HPARAMS["free_bits"]
    assert objective.detach_baseline_in_full is PROD_HPARAMS["detach_baseline_in_full"]
    assert objective.lambda_lag == PROD_HPARAMS["lambda_lag"]


def test_checkpoint_without_hyper_parameters_raises(tmp_path):
    blob = build_tiny_checkpoint_blob()
    del blob["hyper_parameters"]
    with pytest.raises(RuntimeError, match="no 'hyper_parameters'"):
        EvalRunner.from_checkpoint(_save(blob, tmp_path), tmp_path / "run", device="cpu")


def test_checkpoint_missing_one_objective_setting_raises(tmp_path):
    blob = build_tiny_checkpoint_blob()
    del blob["hyper_parameters"]["sigma_obs"]
    with pytest.raises(RuntimeError, match="missing objective setting"):
        EvalRunner.from_checkpoint(_save(blob, tmp_path), tmp_path / "run", device="cpu")


def test_objective_disagreeing_with_the_config_raises_naming_both(runner, repo_root):
    """Neither side is silently preferred: the operator believes something false either way."""
    from teb_vae.lag_attn.config import load_config

    config = load_config(str(repo_root / "teb_vae/lag_attn/eval/tests/fixtures/eval_tiny.yaml"))
    # As shipped the two agree, which is what makes the suite's end-to-end run possible.
    runner.objective.reconcile_with_config(config)

    mismatched = copy.deepcopy(config)
    mismatched["model_config"]["VAE_model"]["likelihood"] = "mse"
    with pytest.raises(ValueError) as excinfo:
        runner.objective.reconcile_with_config(mismatched)
    message = str(excinfo.value)
    assert "gaussian_nll" in message and "'mse'" in message


def test_objective_reconciliation_knows_the_lambda_lag_rename(runner):
    """The config calls it ``lag_smoothness_lambda``; only ``trainer.py`` does the translation."""
    config = {"model_config": {"VAE_model": {"lag_smoothness_lambda": 999.0}}}
    with pytest.raises(ValueError, match="lambda_lag"):
        runner.objective.reconcile_with_config(config)


def test_loss_kwargs_carry_the_trained_objective(runner):
    kwargs = runner.objective.loss_kwargs()
    assert kwargs["likelihood"] == "gaussian_nll"
    assert kwargs["sigma_obs"] == "learned"
    # beta defaults to 0: eval reports every term separately, and a beta-weighted total would
    # be the one number that silently depends on the checkpoint's epoch.
    assert kwargs["beta"] == 0.0


def test_objective_float_comparison_tolerates_representation():
    objective = Objective(
        likelihood="gaussian_nll",
        sigma_obs="learned",
        free_bits=0.1,
        detach_baseline_in_full=True,
        lambda_full=1.0,
        lambda_base=0.5,
        lambda_lag=1.0e-3,
        beta_schedule=None,
        kld_beta=0.01,
    )
    objective.reconcile_with_config(
        {"model_config": {"VAE_model": {"free_bits": 0.1 + 1e-16, "lambda_base": 0.5}}}
    )


# ---------------------------------------------------------------------------
# S1-T03/T04/T05: dispatch, streams, forward
# ---------------------------------------------------------------------------
def test_only_tensor_fields_move_to_device(runner, stub_batch):
    """``guid`` is a ``list[str]`` after collation; a blanket transfer crashes on it.

    The target is the ``meta`` device rather than the runner's own CPU, so the move is
    observable on a CPU-only machine -- against ``cpu`` every assertion here would hold
    whether or not the transfer ran at all.
    """
    runner.device = torch.device("meta")
    batch = dict(vars(stub_batch))
    batch["guid"] = ["a", "b", "c", "d"]
    batch["source_file_basename"] = ["tiny_shard.hdf5"] * 4

    moved = runner.to_device(batch)
    assert moved["fhr_st"].device.type == "meta", "declared tensor fields must move"
    assert moved["weight"].device.type == "meta"
    assert moved["guid"] == ["a", "b", "c", "d"], "guid must be untouched and still strings"
    assert isinstance(moved["source_file_basename"], list)


def test_missing_optional_field_is_skipped_not_raised(runner, stub_batch):
    """``target`` and ``epoch`` are optional; the model's five fields are not."""
    batch = dict(vars(stub_batch))
    assert "target" not in batch
    runner.to_device(batch)  # must not raise


def test_max_samples_caps_by_sample_count(runner):
    """And overshoots at the batch boundary rather than splitting a batch."""
    batches = [
        {"fhr_st": torch.zeros(3, 4, 43), "fhr_ph": torch.zeros(3, 4, 66)} for _ in range(5)
    ]
    drawn = list(runner.iter_batches(batches, max_samples=4))
    # Two batches: the first takes the count to 3, the second to 6 -- past the cap, which is
    # the documented overshoot.
    assert len(drawn) == 2


def test_iter_batches_runs_under_inference_mode(runner):
    runner.model.train()
    for _ in runner.iter_batches([{"fhr_st": torch.zeros(1, 4, 43)}]):
        assert not runner.model.training
        assert not torch.is_grad_enabled()
    assert runner.model.training


def test_source_stream_matches_the_task_elementwise(runner, stub_batch, task):
    """The divergence test: the copy in the runner must equal the task's own assembly.

    This is the assertion that stops the two drifting apart. About twenty lines are duplicated
    deliberately so the eval path does not import Lightning; nothing else keeps them honest.
    """
    from teb_vae.lag_attn.tests.conftest import SHIPPED_KWARGS

    lightning_task = task(model_kwargs=dict(SHIPPED_KWARGS))
    assert torch.equal(
        runner.build_source_stream(stub_batch),
        lightning_task._build_source_stream(stub_batch),
    )
    runner_st, runner_ph = runner.build_target_streams(stub_batch)
    task_st, task_ph = lightning_task._build_target_streams(stub_batch)
    assert torch.equal(runner_st, task_st) and torch.equal(runner_ph, task_ph)


def test_source_stream_is_up_st_then_up_ph(runner, stub_batch):
    """Order matters: the channel map downstream assumes [up_st(43), up_ph(15)]."""
    stream = runner.build_source_stream(stub_batch)
    assert stream.shape[-1] == 58
    assert torch.equal(stream[..., :43], stub_batch.up_st)
    assert torch.equal(stream[..., 43:], stub_batch.up_ph)


def test_source_stream_is_up_ph_alone_under_the_ablation(tmp_path, stub_batch):
    """``use_up_st=False`` feeds the phase-harmonic channels alone, at $c_u = 15$."""
    from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_KWARGS, build_tiny_checkpoint_blob

    blob = build_tiny_checkpoint_blob(dict(SHIPPED_KWARGS, use_up_st=False, c_u=15))
    path = tmp_path / "no_up_st.ckpt"
    torch.save(blob, path)
    ablated = EvalRunner.from_checkpoint(path, tmp_path / "run", device="cpu")

    stream = ablated.build_source_stream(stub_batch)
    assert torch.equal(stream, stub_batch.up_ph)
    # And it must not silently accept the with-scattering width: 58 is both the current
    # use_up_st=true c_u and the old phase-only one.
    stub_batch.up_ph = torch.zeros(4, 16, 58)
    with pytest.raises(RuntimeError, match="c_u=15"):
        ablated.build_source_stream(stub_batch)


def test_source_width_mismatch_raises_naming_both_widths(runner, stub_batch):
    stub_batch.up_ph = torch.zeros(4, 16, 14)
    with pytest.raises(RuntimeError, match="c_u=58"):
        runner.build_source_stream(stub_batch)


def test_width_check_runs_on_every_batch(runner, stub_batch, make_stub_batch_fn):
    """A multi-file split can concatenate shards of different vintages."""
    runner.build_source_stream(stub_batch)
    later = make_stub_batch_fn()
    later.up_st = torch.zeros(4, 16, 40)
    with pytest.raises(RuntimeError, match="source stream is 55 channels"):
        runner.build_source_stream(later)


def test_missing_up_st_under_use_up_st_raises_naming_the_field(runner, stub_batch):
    del stub_batch.up_st
    with pytest.raises(RuntimeError, match="up_st"):
        runner.build_source_stream(stub_batch)


def test_target_width_mismatch_raises(runner, stub_batch):
    stub_batch.fhr_ph = torch.zeros(4, 16, 44)  # the OLD phase-harmonic selection
    with pytest.raises(RuntimeError, match="c_y=109"):
        runner.build_target_streams(stub_batch)


def test_forward_returns_the_full_key_set_unmodified(runner, stub_batch):
    with runner.inference_mode():
        outputs = runner.forward(stub_batch)
    assert len(outputs) == 24
    for key in ("mu_full", "logvar_full", "te_lag_map", "kld_per_t_per_head", "kld_active_frac"):
        assert key in outputs


def test_build_future_target_equals_the_one_compute_loss_builds(runner, stub_batch):
    r"""$Y^{+}$ must be elementwise identical, or every forecast metric is off by an anchor."""
    y_plus = runner.build_future_target(stub_batch)
    horizon = int(runner.model.horizon)
    batch_size, seq_len, _ = stub_batch.fhr_st.shape
    assert y_plus.shape == (batch_size, seq_len - horizon, horizon, 109)

    # The construction compute_loss performs internally, spelled out.
    target = torch.cat([stub_batch.fhr_st, stub_batch.fhr_ph], dim=-1)
    expected = target[:, 1:, :].unfold(dimension=1, size=horizon, step=1).permute(0, 1, 3, 2)
    assert torch.equal(y_plus, expected)
    assert torch.equal(y_plus, future_target(stub_batch.fhr_st, stub_batch.fhr_ph, horizon))


def test_future_target_is_not_warmup_masked(runner, stub_batch):
    """Masking is the caller's job: each one uses a different window."""
    y_plus = runner.build_future_target(stub_batch)
    warmup = int(runner.model.warmup_period)
    assert torch.count_nonzero(y_plus[:, :warmup]) > 0


def test_compute_loss_drops_the_likelihood_string(runner, stub_batch):
    """A metric logger coerces a non-numeric value to a clean 0.0 rather than raising."""
    with runner.inference_mode():
        outputs = runner.forward(stub_batch)
        losses = runner.compute_loss(stub_batch, outputs)
    assert "likelihood" not in losses
    assert all(isinstance(value, torch.Tensor) for value in losses.values())


def test_compute_loss_uses_the_checkpoint_objective_not_the_defaults(runner, stub_batch):
    r"""Under ``mse`` the feature loss is a plain squared error and cannot go negative.

    A pipeline that took the objective from its own YAML would silently produce the second
    number while reporting the first's units.
    """
    with runner.inference_mode():
        outputs = runner.forward(stub_batch)
        trained = float(runner.compute_loss(stub_batch, outputs)["feat_loss"])
        as_mse = float(runner.compute_loss(stub_batch, outputs, likelihood="mse")["feat_loss"])
    assert trained != pytest.approx(as_mse)
