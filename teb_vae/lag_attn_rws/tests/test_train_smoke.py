r"""One real fit, through the real framework, against the committed shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
data module -> model -> ``build_trainer`` -> ``fit`` -> checkpoint, on a CPU, in seconds. It is
the only test that can catch the failures that live *between* the pieces -- a metric key no
callback collects, a ``None`` in a metrics dict, a checkpoint that will not reload, a callback
constructed with the wrong keyword.

The invariant worth naming: at step 0 the KL is exactly zero, because the posterior's delta
heads are zero-initialised and the posterior therefore *is* the prior. Asserting it here rather
than on a bare model is the point -- it is the one place the property is checked after config
resolution, kwarg sweeping, data normalization and the framework's own seeding have all had a
chance to break it.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"
_PACKAGE_DIR = _REPO_ROOT / "teb_vae" / "lag_attn_rws"


def _run_fit(tmp_path, *, causal_reach_budget_s=None):
    """Run one real fit against the committed shard and return the driver and its trainer.

    Args:
        tmp_path: Directory to run in.
        causal_reach_budget_s: The reach budget, in seconds, or ``None`` for the shipped
            unguarded default.

    Returns:
        ``(driver, trainer)``.
    """
    config = load_config(str(_TINY))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    config["general_config"]["epochs"] = 2
    config["model_config"]["VAE_model"]["causal_reach_budget_s"] = causal_reach_budget_s
    absolutize_dataset_paths(config)
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from train.data_module import GraphDataModule

    driver = LagAttnRwsTrainer(config_file_path=str(config_path))
    driver.setup_config()
    data_module = GraphDataModule(driver.config)
    driver.create_model()
    trainer = driver.train_model(data_module.train_dataloader(), data_module.val_dataloader())
    return driver, trainer


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """Run one real fit at the shipped (unguarded) configuration.

    Module-scoped: this is the expensive test in the suite, and every assertion below is a
    different question about the same run. Two epochs, not the config's one: ``lr`` is logged
    at train-epoch *start* with ``on_epoch=True``, so its first CSV cell is always NaN, and the
    second epoch is also what exercises the LR scheduler stepping at all.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke"))


@pytest.fixture(scope="module")
def guarded_fit(tmp_path_factory):
    """The same fit with the causal input guard on, at the $120$ s reach budget.

    A second full fit rather than an assertion on the first, because what it exercises is only
    reachable end to end: the budget resolving from config, the four channel tuples surviving
    the kwarg sweep into narrower input adapters, the delayed stream reaching the encoders, and
    the resolved delay vector being written into the run's own configuration.
    """
    return _run_fit(tmp_path_factory.mktemp("smoke_guarded"), causal_reach_budget_s=120.0)


def test_the_fit_completes(fit):
    driver, trainer = fit

    assert trainer.current_epoch == 2
    assert trainer.state.finished


def test_the_losses_stay_finite(fit):
    driver, trainer = fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_zero_kl_init_invariant_survives_the_whole_stack(fit):
    r"""At initialisation $q(z_t \mid Y, U) = p(z_t \mid Y)$ exactly, so $K = 0$.

    Re-derived from a freshly built model rather than read off the trained run: after an epoch
    the KL is legitimately nonzero, and the question is whether the model the config builds
    starts at zero.
    """
    driver, _ = fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    model = SeqVaeLagAttnRws(**driver._build_model_kwargs()).eval()
    batch_size, seq_len = 2, model.sequence_length
    generator = torch.Generator().manual_seed(0)
    outputs = model(
        torch.randn(batch_size, seq_len, 43, generator=generator),
        torch.randn(batch_size, seq_len, 66, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
    )

    assert float(outputs["kld_per_t"].abs().max()) < 1e-6
    assert torch.equal(outputs["mu_base"], outputs["mu_full"])


def test_every_declared_metric_reaches_the_logger(fit):
    """The gap between "the task emits it" and "a callback collected it" is silent otherwise.
    The shuffled readouts are the ones only a real validation loop can prove wired."""
    driver, trainer = fit

    for name in (
        "train/total_loss",
        "train/main_loss",
        "train/source_conditioned_kl_raw",
        "val/total_loss",
        "val/nll_shuffled_block",
        "val/kld_shuffled",
        "val/shuffle_penalty",
    ):
        assert name in trainer.callback_metrics, f"{name} never reached callback_metrics"


def test_the_metrics_history_csv_has_no_all_nan_column(fit):
    """The check that catches a tracked name the framework never emits: the column appears in
    every run's CSV, NaN in every row, and nothing anywhere reports it."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_scheduled_beta_reaches_the_csv(fit):
    """The resolved schedule value, which starts at exactly zero -- the posterior-collapse
    guard the config documents."""
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    assert "train/kld_beta" in frame.columns
    assert float(frame["train/kld_beta"].iloc[0]) == pytest.approx(0.0)


def test_the_checkpoint_is_written_to_the_run_checkpoint_directory(fit):
    driver, _ = fit

    checkpoints = list(Path(driver.model_checkpoint_dir).glob("*.ckpt"))

    assert checkpoints, "no checkpoint was written; Lightning's default would have gone elsewhere"


def test_the_checkpoint_carries_its_contract_and_reloads(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file,
    through the repository's own loading helpers."""
    driver, _ = fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttnRws"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttnRws")
    rebuilt = SeqVaeLagAttnRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )


def test_the_run_directory_holds_the_logs(fit):
    driver, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()


def test_the_validation_figure_is_written_by_a_real_fit(fit):
    """The plotting callback driven by a real trainer and a real loader rather than by a fake.

    Everything the unit tests cannot reach meets here: the batch actually coming off the HDF5
    loader, the normalization statistics actually being reachable through it, and the callback
    surviving a Lightning validation epoch. The callback swallows its own exceptions by design,
    so a broken figure is silent everywhere except in this file count.
    """
    driver, _ = fit

    figures = list(
        (Path(driver.train_results_dir) / "lag_attn_rws_diagnostics").glob("*.pdf")
    )

    assert figures, "the enabled plotting callback wrote no figure"


def test_a_fit_completes_under_the_causal_reach_budget(guarded_fit):
    """The whole guard, end to end: config to filter bank to channel tuples to a trained model.

    A unit test can check each link; only a fit can check that they are connected -- most
    concretely that the narrowed adapters and the full declared ``c_y``/``c_u`` coexist, since
    the data boundary validates the batch against the declared widths while the model reads only
    the survivors.
    """
    driver, trainer = guarded_fit

    assert trainer.current_epoch == 2
    assert trainer.state.finished
    assert driver.pytorch_model.target_adapter.linear.in_features == 78
    assert driver.pytorch_model.source_adapter.linear.in_features == 29
    assert driver.pytorch_model.target_gate.max_delay == 30
    assert driver.pytorch_model.source_delay_steps == 30
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (driver.pytorch_model.c_y, driver.pytorch_model.c_u) == (109, 58)


def test_the_guarded_runs_losses_stay_finite(guarded_fit):
    """A delayed stream is zero for its first max(delta) steps; those steps must fall inside the
    warm-up rather than reaching the loss as a block of zeros."""
    _, trainer = guarded_fit

    for name, value in trainer.callback_metrics.items():
        assert math.isfinite(float(value)), f"{name} is {float(value)}"


def test_the_guarded_checkpoint_rebuilds_at_its_own_channel_widths(guarded_fit):
    """The adapters' widths depend on the resolved budget, so a checkpoint that recorded only
    the budget in seconds could not be rebuilt without re-running the resolution. The four
    channel tuples are therefore in ``model_kwargs``."""
    driver, _ = guarded_fit
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    path = next(iter(Path(driver.model_checkpoint_dir).glob("*.ckpt")))
    blob = torch.load(path, map_location="cpu", weights_only=False)

    assert len(blob["model_kwargs"]["target_keep_index"]) == 78
    assert len(blob["model_kwargs"]["source_delays"]) == 29
    rebuilt = SeqVaeLagAttnRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None


def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only
    seeding route; a stray global seed would silently override it while looking like
    diligence. The permutation generator's own rank-derived seed is a *local*
    ``torch.Generator``, deliberately different per rank, and leaves the global RNG untouched
    -- which is why the patterns below name the global calls specifically.
    """
    offenders = []
    for path in _PACKAGE_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue  # tests seed themselves for reproducibility, legitimately
        source = path.read_text(encoding="utf-8")
        for pattern in ("torch.manual_seed", "seed_everything", "np.random.seed"):
            if pattern in source:
                offenders.append(f"{path.name}: {pattern}")
    assert offenders == []
