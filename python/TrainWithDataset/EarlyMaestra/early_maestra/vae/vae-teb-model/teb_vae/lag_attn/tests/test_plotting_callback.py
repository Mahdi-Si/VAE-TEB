r"""The plotting callback: its guards, its scheduled-beta fix, and its MLflow upload seam.

Three things this callback must get right beyond drawing a figure. It must plot the *scheduled*
$\beta$ for the epoch, not the raw hyperparameter -- the old callback read the hparam and so drew a
flat line under any schedule. It must route every saved file to MLflow through the rank-0 artifact
seam, which the old callback never did. And it must stay silent off rank 0, during the sanity pass,
and between plot epochs -- writing nothing and uploading nothing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn import plotting  # noqa: E402
from teb_vae.lag_attn.plotting import LagAttnPlotCallback  # noqa: E402
from train.test_utils import FakeMLflowLogger, FakeTrainer  # noqa: E402


def _pdfs(directory: Path):
    """Sorted names of the PDF files under ``directory`` (empty list if it has none)."""
    return sorted(p.name for p in directory.glob("*.pdf"))


def test_generate_plots_writes_a_diagnostic_and_a_control_into_the_named_subdir(
    tmp_path, task, stub_batch
):
    """Both figures land, and in the subdirectory the constructor was given -- no mkdir/rmdir dance."""
    cb = LagAttnPlotCallback(tmp_path, num_examples=1, subdir="custom_diag")
    cb._generate_plots(FakeTrainer(is_global_zero=True), stub_batch, task(), epoch=0)

    subdir = tmp_path / "custom_diag"
    names = _pdfs(subdir)
    assert len(names) == 2
    assert sum(name.endswith("_control.pdf") for name in names) == 1


def test_the_plotted_beta_is_the_scheduled_value_not_the_raw_hparam(
    tmp_path, task, stub_batch, monkeypatch
):
    """The bug this callback fixes: read the resolved schedule, never ``hparams['kld_beta']``."""
    captured = {}

    def _spy(**kwargs):
        captured["beta"] = kwargs["beta"]
        return plt.figure()

    monkeypatch.setattr(plotting, "_build_diagnostic_figure", _spy)
    monkeypatch.setattr(plotting, "_build_companion_figure", lambda **kwargs: plt.figure())

    module = task(hparams={
        "beta_schedule": {"kind": "linear_warmup", "start": 1.0e-4, "end": 0.5, "warmup_epochs": 10},
    })
    cb = LagAttnPlotCallback(tmp_path, num_examples=1)
    cb._generate_plots(FakeTrainer(is_global_zero=True), stub_batch, module, epoch=0)

    scheduled = module._resolve_beta(module.current_epoch)
    assert captured["beta"] == pytest.approx(scheduled)
    # The schedule's value genuinely differs from the raw hparam, so this proves which was read.
    assert captured["beta"] != pytest.approx(float(module.hparams["kld_beta"]))


def test_every_saved_file_uploads_once_through_the_rank_zero_seam(tmp_path, task, stub_batch):
    """Each written PDF is registered exactly once via the run-bound MLflow client."""
    logger = FakeMLflowLogger(run_id="run-7")
    cb = LagAttnPlotCallback(tmp_path, num_examples=1, mlflow_logger=logger)
    cb._generate_plots(FakeTrainer(is_global_zero=True), stub_batch, task(), epoch=0)

    written = {str(p) for p in (tmp_path / "lag_attn_diagnostics").glob("*.pdf")}
    assert len(written) == 2
    assert all(method == "log_artifact" and run_id == "run-7"
               for method, run_id, _ in logger.experiment.calls)
    assert {payload[0] for _, _, payload in logger.experiment.calls} == written


def _trainer_with_batch(stub_batch, **kwargs):
    """A FakeTrainer that also serves ``stub_batch`` as its first validation batch."""
    trainer = FakeTrainer(**kwargs)
    trainer.val_dataloaders = [[stub_batch]]
    return trainer


def test_off_rank_zero_it_writes_nothing_and_uploads_nothing(tmp_path, task, stub_batch):
    logger = FakeMLflowLogger()
    cb = LagAttnPlotCallback(tmp_path, num_examples=1, mlflow_logger=logger)
    cb.on_validation_epoch_end(_trainer_with_batch(stub_batch, is_global_zero=False), task())

    assert _pdfs(tmp_path / "lag_attn_diagnostics") == []
    assert logger.experiment.calls == []


def test_the_sanity_pass_is_skipped(tmp_path, task, stub_batch):
    cb = LagAttnPlotCallback(tmp_path, num_examples=1)
    cb.on_validation_epoch_end(
        _trainer_with_batch(stub_batch, is_global_zero=True, sanity_checking=True), task()
    )

    assert _pdfs(tmp_path / "lag_attn_diagnostics") == []


def test_it_only_fires_on_multiples_of_plot_frequency(tmp_path, task, stub_batch):
    cb = LagAttnPlotCallback(tmp_path, num_examples=1, plot_frequency=5)
    module = task()

    # Epoch 0: (0 + 1) % 5 != 0 -> nothing.
    cb.on_validation_epoch_end(_trainer_with_batch(stub_batch, current_epoch=0), module)
    assert _pdfs(tmp_path / "lag_attn_diagnostics") == []

    # Epoch 4: (4 + 1) % 5 == 0 -> it fires.
    cb.on_validation_epoch_end(_trainer_with_batch(stub_batch, current_epoch=4), module)
    assert _pdfs(tmp_path / "lag_attn_diagnostics") != []
