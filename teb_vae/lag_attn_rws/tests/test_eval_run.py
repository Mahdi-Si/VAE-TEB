r"""The evaluation entry point, driven end to end against the committed tiny shard.

This is the only place the evaluation package meets a real loader, and therefore the only place
three things can be checked at all: that a checkpoint's own resolved config is found and used,
that recording identifiers survive collation into the output as real strings rather than as
``'unknown'``, and that the summary is JSON a non-Python reader can parse.

The negative cases around checkpoint loading carry most of the remaining weight. Each of them --
a blob from another model, a blob with no architecture, a state dict that does not align -- would
otherwise produce a complete set of entirely plausible numbers from a randomly initialised model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.eval import run as run_module
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME, LagAttnRwsTrainer

from .conftest import TASK_HPARAMS, TINY_KWARGS, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"

#: Monte Carlo draws used by the end-to-end run. Two rather than the shipped eight: this test is
#: about the plumbing, and each draw decodes every branch over 270 anchors.
_TEST_DRAWS = 2


def _perturb_posterior(model: SeqVaeLagAttnRws, seed: int = 3, scale: float = 0.1) -> None:
    """Move the posterior off the prior, so the checkpoint describes a model that learned.

    Load-bearing rather than cosmetic. The delta heads are zero-initialised, so an unperturbed
    checkpoint is indistinguishable *in weight space* from one that never loaded -- and every
    KL-shaped assertion below would hold on a model whose weights were discarded.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)


def _absolute_tiny_config() -> dict:
    """The tiny config with every dataset path made absolute, so no chdir is needed."""
    return absolutize_dataset_paths(load_config(str(_TINY)))


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> Path:
    """A checkpoint written into a run-shaped directory, with its resolved config beside it.

    Mirrors what the training entry point leaves behind -- ``model_checkpoints/`` holding the
    blob and the resolved config -- without spending a fit to produce it.
    """
    run_dir = tmp_path_factory.mktemp("run")
    checkpoint_dir = run_dir / "model_checkpoints"
    checkpoint_dir.mkdir()

    config = _absolute_tiny_config()
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    driver = LagAttnRwsTrainer(config_file_path=str(config_path))
    model_kwargs = driver._build_model_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**model_kwargs)
    _perturb_posterior(model)
    task = SeqVaeLagAttnRwsTask(
        model, lr=1e-3, model_kwargs=model_kwargs,
        **dict(TASK_HPARAMS, likelihood=config["model_config"]["VAE_model"]["likelihood"]),
    )
    blob = {"state_dict": task.state_dict(), "epoch": 0, "global_step": 0,
            "hyper_parameters": dict(task.hparams)}
    task.on_save_checkpoint(blob)
    torch.save(blob, checkpoint_dir / "lag-attn-rws-epoch=00.ckpt")

    (checkpoint_dir / RESOLVED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_dir / "lag-attn-rws-epoch=00.ckpt"


@pytest.fixture(scope="module")
def evaluated(trained_run, tmp_path_factory) -> dict:
    """One real evaluation run; every assertion below is a question about the same run."""
    output_dir = tmp_path_factory.mktemp("eval")
    summary_path = run_module.main(
        trained_run, output_dir, device="cpu", num_samples=_TEST_DRAWS
    )
    return {
        "summary_path": summary_path,
        "text": summary_path.read_text(encoding="utf-8"),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "results_dir": summary_path.parent,
    }


# =============================================================================
# End to end
# =============================================================================
def test_the_run_writes_a_summary_and_the_config_it_used(evaluated):
    assert evaluated["summary_path"].name == run_module.SUMMARY_FILENAME
    assert (evaluated["results_dir"] / RESOLVED_CONFIG_FILENAME).is_file()
    assert evaluated["results_dir"].name == run_module.RESULTS_DIRNAME


def test_the_summary_reports_every_section_and_all_four_verdicts(evaluated):
    results = evaluated["summary"]["results"]

    assert set(results) >= {"readouts", "latent_health", "lag", "per_recording", "verdicts"}
    assert [verdict["name"] for verdict in results["verdicts"]] == [
        "predictive_improvement",
        "source_specificity",
        "prior_carries_target_state",
        "latent_not_collapsed",
    ]
    for verdict in results["verdicts"]:
        assert verdict["status"] in {"PASS", "FAIL", "INCONCLUSIVE"}


def test_the_summary_carries_the_checkpoint_and_config_it_evaluated(evaluated, trained_run):
    summary = evaluated["summary"]

    assert Path(summary["checkpoint"]) == trained_run
    assert Path(summary["config"]).name == RESOLVED_CONFIG_FILENAME
    assert summary["device"] == "cpu"


def test_the_summary_is_json_a_non_python_reader_can_parse(evaluated):
    """``json.dump`` emits the bare tokens ``NaN`` and ``Infinity`` for non-finite floats, which
    round-trip through Python and are rejected by every other parser."""
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in evaluated["text"]


def test_the_summary_holds_no_tensors_or_numpy_scalars(evaluated):
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert isinstance(value, (str, int, float, bool, type(None))), type(value)

    walk(evaluated["summary"])


def test_recording_identifiers_reach_the_output_as_real_guids(evaluated):
    """The one thing a stub batch cannot check: ``guid`` survives collation as a ``list[str]``,
    is never moved to a device, and must arrive as the shard's own identifiers rather than as the
    ``'unknown'`` fallback."""
    per_recording = evaluated["summary"]["results"]["per_recording"]

    assert per_recording, "no recordings were aggregated"
    assert "unknown" not in per_recording


def test_the_readouts_are_finite_and_the_gap_is_the_difference(evaluated):
    readouts = evaluated["summary"]["results"]["readouts"]

    for name, value in readouts.items():
        assert value is not None and math.isfinite(value), f"{name} is {value}"
    assert readouts["pred_gap"] == pytest.approx(
        readouts["nll_base_block"] - readouts["nll_full_block"], rel=1e-5
    )


def test_the_run_used_a_single_process_loader(evaluated):
    """Spawn workers over a multi-file HDF5 dataset silently truncate every pass after the
    first, and an evaluation makes many passes."""
    written = yaml.safe_load(
        (evaluated["results_dir"] / RESOLVED_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    loader_config = written["dataset_config"]["dataloader_config"]

    assert loader_config["num_workers"] == 0
    assert loader_config["persistent_workers"] is False


def test_an_unguarded_run_reports_no_input_delay(evaluated):
    summary = evaluated["summary"]

    assert summary["source_delay_steps"] == 0
    assert summary["results"]["lag"]["delay_steps"] == 0


def test_the_lag_report_adds_back_the_causal_input_delay():
    r"""A checkpoint trained under a reach budget has a stale source memory, so a peak at lag
    $\ell$ refers to content $\ell + \delta$ steps back. Reporting it with $\delta = 0$
    understates the physiological delay by up to two minutes at the $120$ s budget, with nothing
    failing -- so the delay is read off the *model*, which is what was trained.
    """
    from teb_vae.lag_attn_rws.channel_reach import resolve_stream_budgets

    budget = resolve_stream_budgets(
        {"causal_reach_budget_s": 120.0, "use_up_st": True, "warmup_period": 30,
         "c_y": 109, "c_u": 58}
    )
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(
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

    assert model.source_delay_steps == budget.max_delay == 30


# =============================================================================
# Finding the run's own configuration
# =============================================================================
def test_the_config_is_found_beside_the_checkpoint(trained_run):
    found = run_module.resolved_config_for(trained_run)

    assert found == trained_run.parent / RESOLVED_CONFIG_FILENAME


def test_a_checkpoint_without_its_config_names_every_path_tried(tmp_path):
    """A checkpoint copied out of its run directory has lost the record of what it trained on,
    and evaluating it against a guessed configuration is worse than not evaluating it."""
    orphan = tmp_path / "model_checkpoints" / "lonely.ckpt"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"")

    with pytest.raises(FileNotFoundError) as excinfo:
        run_module.resolved_config_for(orphan)

    message = str(excinfo.value)
    assert RESOLVED_CONFIG_FILENAME in message
    assert str(orphan.parent) in message


def test_the_output_directory_is_timestamped_with_a_collision_guard(tmp_path):
    config = {"general_config": {"tag": "rws", "folders_config": {"out_dir_base": str(tmp_path)}}}

    first = run_module.make_output_dir(config)
    second = run_module.make_output_dir(config)

    assert first != second
    assert first.name == second.name == run_module.RESULTS_DIRNAME


def test_an_explicit_output_directory_is_used_as_given(tmp_path):
    result = run_module.make_output_dir({}, tmp_path / "here")

    assert result == tmp_path / "here" / run_module.RESULTS_DIRNAME
    assert result.is_dir()


# =============================================================================
# Checkpoint loading
# =============================================================================
def _mutated(trained_run: Path, tmp_path: Path, mutate) -> Path:
    """Save a copy of the checkpoint with one key changed."""
    blob = torch.load(trained_run, map_location="cpu", weights_only=False)
    mutate(blob)
    path = tmp_path / "mutated.ckpt"
    torch.save(blob, path)
    return path


def test_a_checkpoint_from_another_model_is_refused(trained_run, tmp_path):
    def _rename(blob):
        blob["model_class"] = "SomeOtherModel"

    with pytest.raises(ValueError, match="model_class"):
        run_module.load_task(_mutated(trained_run, tmp_path, _rename), torch.device("cpu"))


def test_a_checkpoint_without_model_kwargs_is_refused(trained_run, tmp_path):
    """``SeqVaeLagAttnRws()`` with no arguments builds the *production* geometry rather than
    raising, so guessing would silently evaluate a different model."""

    def _drop(blob):
        blob["model_kwargs"] = {}

    with pytest.raises(RuntimeError, match="no 'model_kwargs'"):
        run_module.load_task(_mutated(trained_run, tmp_path, _drop), torch.device("cpu"))


def test_a_checkpoint_without_hyperparameters_is_refused(trained_run, tmp_path):
    """The likelihood is a checkpoint fact: scoring an ``mse`` run under a Gaussian NLL would
    report a different objective's numbers with nothing raising."""

    def _drop(blob):
        del blob["hyper_parameters"]

    with pytest.raises(RuntimeError, match="no 'hyper_parameters'"):
        run_module.load_task(_mutated(trained_run, tmp_path, _drop), torch.device("cpu"))


def test_a_state_dict_that_does_not_align_is_refused(trained_run, tmp_path):
    """``load_checkpoint_strict`` returns ``None`` rather than raising, so an unchecked call
    evaluates randomly initialised weights and reports the result as a measurement."""

    def _widen(blob):
        blob["model_kwargs"] = dict(blob["model_kwargs"], d_model=64, d_head=16)

    with pytest.raises(RuntimeError, match="could not align checkpoint"):
        run_module.load_task(_mutated(trained_run, tmp_path, _widen), torch.device("cpu"))


def test_a_missing_checkpoint_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_module.load_task(tmp_path / "absent.ckpt", torch.device("cpu"))


def test_the_loaded_weights_are_the_checkpoints_own(trained_run):
    """Every parameter, not merely a shape-compatible model."""
    task = run_module.load_task(trained_run, torch.device("cpu"))
    blob = torch.load(trained_run, map_location="cpu", weights_only=False)
    saved = {
        key[len("_orig_model.") :]: value
        for key, value in blob["state_dict"].items()
        if key.startswith("_orig_model.")
    }

    assert saved, "the checkpoint's state dict is not wrapper-prefixed as expected"
    for name, parameter in task.orig_model.state_dict().items():
        assert torch.equal(parameter, saved[name]), f"{name} did not load"


def test_the_loaded_task_is_in_evaluation_mode(trained_run):
    """Dropout live during evaluation would leave the attention rows not summing to one, so the
    lag attribution would not be a decomposition of anything."""
    assert run_module.load_task(trained_run, torch.device("cpu")).training is False


# =============================================================================
# JSON safety
# =============================================================================
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_become_null(value):
    assert run_module.json_safe(value) is None


def test_numpy_and_torch_values_become_plain_python():
    converted = run_module.json_safe(
        {
            "flag": np.bool_(True),
            "count": np.int64(3),
            "value": np.float32(1.5),
            "array": np.array([1.0, 2.0]),
            "tensor": torch.tensor([3.0, 4.0]),
            "path": Path("a") / "b",
        }
    )

    # np.bool_ is checked before the int branch; otherwise True would serialise as 1.
    assert converted["flag"] is True
    assert converted["count"] == 3 and isinstance(converted["count"], int)
    assert converted["value"] == pytest.approx(1.5)
    assert converted["array"] == [1.0, 2.0]
    assert converted["tensor"] == [3.0, 4.0]
    assert isinstance(converted["path"], str)


def test_an_unexpected_type_is_recorded_rather_than_dropped():
    """A stray object lands as its repr instead of killing the write at the end of a long run."""
    assert run_module.json_safe(torch.device("cpu")) == "cpu"
