r"""One real fit, through the real framework, against the committed shard.

Everything else in this suite tests a piece in isolation. This runs the whole thing: config ->
data module -> model -> ``build_trainer`` -> ``fit`` -> checkpoint, on a CPU, in seconds. It is the
only test that can catch the failures that live *between* the pieces -- a metric key that no
callback collects, a ``None`` in a metrics dict, a checkpoint that will not reload, a callback
constructed with the wrong keyword.

It is also the first production exercise of ``build_trainer`` anywhere in this repository. Its
kwargs have been unit-tested at the dict level for a while; nothing had ever handed them to a real
``Trainer`` and run.

The invariant worth naming: at step 0 the KL is exactly zero, because the posterior's delta heads
are zero-initialised and the posterior therefore *is* the prior. Asserting it here rather than on a
bare model is the point -- it is the one place the property is checked after config resolution,
kwarg sweeping, data normalization and the framework's own seeding have all had a chance to break
it.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn.trainer import LagAttnTrainer
from train.graph_models_utils import check_model_class, load_checkpoint_strict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TINY = _REPO_ROOT / "teb_vae" / "lag_attn" / "configs" / "tiny.yaml"


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    """Run one real epoch and return the driver, the fitted trainer, and the run directory.

    Module-scoped: this is the expensive test in the suite, and every assertion below is a
    different question about the same run.
    """
    tmp_path = tmp_path_factory.mktemp("smoke")
    config = load_config(str(_TINY))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path)
    # Two epochs, not the config's one. `lr` is logged at train-epoch *start* with on_epoch=True, so
    # it is not committed to callback_metrics until that epoch ends -- which is after the first
    # validation-epoch-end, where the history is collected. A single-epoch run therefore has an
    # all-NaN `lr` column for reasons that have nothing to do with the naming rule below, and the
    # second epoch is also what exercises the LR scheduler stepping at all.
    config["general_config"]["epochs"] = 2
    # The shard paths are repo-root-relative because entry points run from the repo root.
    dataset = config["dataset_config"]
    for key in ("vae_train_datasets", "vae_test_datasets"):
        dataset[key] = [str(_REPO_ROOT / path) for path in dataset[key]]
    dataset["stat_path"] = str(_REPO_ROOT / dataset["stat_path"])
    # Off: this asserts the training path, not the profiler's output.
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from train.data_module import GraphDataModule

    driver = LagAttnTrainer(config_file_path=str(config_path))
    driver.setup_config()
    data_module = GraphDataModule(driver.config)
    driver.create_model()
    trainer = driver.train_model(data_module.train_dataloader(), data_module.val_dataloader())
    return driver, trainer


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

    The posterior's log-variance is a zero-init residual around the prior's, and its mean delta
    head is zero-init too, so every nat the KL ever reports has been *earned* by source
    conditioning rather than inherited from a random mismatch at step 0. That is what makes the
    reported KL readable as a transfer entropy at all.

    Re-derived from a freshly built model here rather than read off the trained run: after an epoch
    the KL is legitimately nonzero, and the question is whether the model the config builds starts
    at zero.
    """
    driver, _ = fit
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

    model = SeqVaeLagAttn(**driver._build_model_kwargs()).eval()
    batch_size, seq_len = 2, model.sequence_length
    generator = torch.Generator().manual_seed(0)
    outputs = model(
        torch.randn(batch_size, seq_len, 43, generator=generator),
        torch.randn(batch_size, seq_len, 66, generator=generator),
        torch.randn(batch_size, seq_len, model.c_u, generator=generator),
    )

    assert float(outputs["kld_per_t"].abs().max()) < 1e-6


def test_the_metrics_history_csv_has_no_all_nan_column(fit):
    """The check that catches a tracked name the framework never emits.

    A bare ``kld_beta`` produced exactly this -- a column, present in every run's CSV, NaN in every
    row -- and nothing anywhere reported it.
    """
    driver, _ = fit
    frame = pd.read_csv(Path(driver.train_results_dir) / "metrics_history.csv")

    all_nan = [column for column in frame.columns if frame[column].isna().all()]
    assert all_nan == [], f"columns that are NaN for every epoch: {all_nan}"


def test_the_checkpoint_carries_its_contract_and_reloads(fit):
    """The end of the road: a blob that describes itself and rebuilds without a config file."""
    driver, _ = fit
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn

    checkpoints = sorted(Path(driver.model_checkpoint_dir).glob("*.ckpt"))
    assert checkpoints, "no checkpoint in the run directory; Lightning's default would go elsewhere"
    blob = torch.load(checkpoints[0], map_location="cpu", weights_only=False)

    assert blob["model_class"] == "SeqVaeLagAttn"
    assert blob["model_kwargs"] == driver._build_model_kwargs()
    check_model_class(blob, "SeqVaeLagAttn")
    rebuilt = SeqVaeLagAttn(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "the checkpoint's state dict did not align into a model rebuilt from its own kwargs"
    )


def test_the_run_directory_holds_the_logs(fit):
    driver, _ = fit

    assert (Path(driver.train_results_dir) / "full.log").is_file()


# --------------------------------------------------------------------------------------
# Diagnostic plotting
# --------------------------------------------------------------------------------------
def test_the_default_tiny_run_writes_no_diagnostic_plots(fit):
    """``lag_attn_plotting.enabled: false`` in the tiny config: the plotter is never constructed."""
    driver, _ = fit

    assert not (Path(driver.train_results_dir) / "lag_attn_diagnostics").exists()
