r"""The validation diagnostic figure, and the guards that keep it out of the training loop's way.

Two kinds of test here. The builder is driven directly from a real forward pass, which is what
pins the *content* -- that the forecast is drawn in bpm and that a lag axis says which of the two
lag quantities it shows. The callback is driven through its real hook with a fake trainer, which
is what pins the *behaviour* -- silent off rank zero, silent during the sanity pass, silent
between plot epochs, and never raising into a fit.

Both directions are asserted wherever a single direction would pass vacuously: the bpm test also
checks the no-statistics fallback, and the frequency test also checks the epoch that should fire.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.nets.lag_report import COMPENSATED_LAG_AXIS_LABEL  # noqa: E402
from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback  # noqa: E402
from train.test_utils import FakeMLflowLogger, FakeTrainer  # noqa: E402

#: A plausible target scale: the loader z-scores the raw FHR by one global mean and standard
#: deviation, so inverting it is an affine map with these two numbers.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}}


def _forward(task_module: Any, batch: Any) -> dict:
    """Run the net once on a stub batch and return everything the builder needs."""
    model = task_module.orig_model
    with torch.no_grad():
        outs = model(*task_module._build_target_streams(batch), task_module._build_source_stream(batch))
        kld_per_dim = model.kld_tensor(
            mu_prior=outs["mu_prior"],
            logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"],
            logvar_post=outs["logvar_post"],
        )
    return {"outs": outs, "kld_per_dim": kld_per_dim, "geometry": model.geometry}


def _build(task_module: Any, batch: Any, **overrides) -> Any:
    """Build the figure for sample 0 with sensible defaults, overridable per test."""
    pieces = _forward(task_module, batch)
    kwargs = dict(
        outs=pieces["outs"],
        kld_per_dim=pieces["kld_per_dim"],
        fhr_raw=batch.fhr,
        geometry=pieces["geometry"],
        sample_index=0,
        epoch=3,
        guid="rec-0001",
        beta=0.25,
        scalars={"nll_base_block": 1.0, "nll_full_block": 0.5, "pred_gap": 0.5},
        normalization_stats=_STATS,
    )
    kwargs.update(overrides)
    return plotting.build_diagnostic_figure(**kwargs)


def _axes_titled(figure: Any, prefix: str) -> Any:
    """Return the single axes whose title starts with ``prefix``."""
    matches = [ax for ax in figure.axes if ax.get_title().startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} panel, found {len(matches)}"
    return matches[0]


def _trainer_with_batch(batch: Any, **kwargs) -> FakeTrainer:
    """A fake trainer whose validation loader yields ``batch`` once."""
    trainer = FakeTrainer(**kwargs)
    trainer.val_dataloaders = [[batch]]  # type: ignore[attr-defined]
    return trainer


# =============================================================================
# The figure itself
# =============================================================================
def test_the_figure_builds_from_one_validation_batch(task, stub_batch):
    figure = _build(task(), stub_batch)
    try:
        assert len(figure.axes) >= 7, "one main axes per row, plus the reserved colorbar axes"
        for ax in figure.axes:
            if ax.get_title():
                assert ax.has_data(), f"panel {ax.get_title()!r} drew nothing"
    finally:
        plt.close(figure)


def test_the_forecast_panel_is_drawn_in_bpm(task, stub_batch):
    """z-units are the model's coordinate system, not a clinician's: a forecast that cannot be
    read against physiology is a forecast nobody can check."""
    module = task()
    figure = _build(module, stub_batch)
    try:
        ax = _axes_titled(figure, "Forecast at anchor")
        assert "bpm" in ax.get_ylabel()
    finally:
        plt.close(figure)


def test_without_statistics_the_forecast_panel_says_normalised_instead_of_lying(task, stub_batch):
    """The other direction: no statistics means z-units, labelled as such, never mislabelled
    bpm -- so the test above is about the conversion and not about the word."""
    figure = _build(task(), stub_batch, normalization_stats=None)
    try:
        ax = _axes_titled(figure, "Forecast at anchor")
        assert "normalised" in ax.get_ylabel()
        assert "bpm" not in ax.get_ylabel()
    finally:
        plt.close(figure)


def test_the_bpm_conversion_is_the_loaders_inverse(task, stub_batch):
    r"""$\mathrm{bpm} = z\,(\sigma + 10^{-8}) + \mu$, applied to the truth and to both forecasts
    alike, so the three curves in the panel stay comparable."""
    module = task()
    figure = _build(module, stub_batch)
    try:
        ax = _axes_titled(figure, "Forecast at anchor")
        drawn = ax.lines[0].get_ydata()  # the true future, plotted first
        geometry = module.orig_model.geometry
        start = geometry.future_block_start(
            int(round(0.6 * (geometry.t_valid - 1 - geometry.warmup))) + geometry.warmup
        )
        expected = (
            stub_batch.fhr[0, start : start + len(drawn)] * (_STATS["fhr"]["std"] + 1e-8)
            + _STATS["fhr"]["mean"]
        )
        assert torch.allclose(torch.as_tensor(drawn).float(), expected, atol=1e-3)
    finally:
        plt.close(figure)


def test_both_lag_panels_name_the_compensated_lag_quantity(task, stub_batch):
    """A lag axis reading "Lag (s)" is ambiguous between the physiological lag and the
    uncorrected sensor one, which differ by the 20 s the pipeline already removed."""
    figure = _build(task(), stub_batch)
    try:
        for prefix in ("Lag attention", "$\\widetilde K"):
            ax = _axes_titled(figure, prefix)
            labels = [child.get_ylabel() for child in ax.child_axes]
            assert COMPENSATED_LAG_AXIS_LABEL in labels, f"{prefix}: secondary axis labels {labels}"
    finally:
        plt.close(figure)


def test_the_title_carries_the_beta_it_was_given(task, stub_batch):
    figure = _build(task(), stub_batch, beta=0.125)
    try:
        assert "beta=0.125" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


# =============================================================================
# The callback
# =============================================================================
def test_it_writes_one_figure_per_requested_sample_and_logs_each(tmp_path, task, stub_batch):
    logger = FakeMLflowLogger(run_id="run-7")
    callback = LagAttnRwsPlotCallback(tmp_path, num_examples=2, mlflow_logger=logger)
    trainer = _trainer_with_batch(stub_batch)

    callback.on_validation_epoch_end(trainer, task())

    written = sorted(callback.output_dir.glob("*.pdf"))
    assert len(written) == 2
    assert [call[0] for call in logger.experiment.calls] == ["log_artifact", "log_artifact"]
    assert sorted(Path(call[2][0]) for call in logger.experiment.calls) == written


def test_it_plots_the_scheduled_beta_not_the_raw_hyperparameter(
    tmp_path, task, stub_batch, monkeypatch
):
    """Under any warm-up the raw ``kld_beta`` is the schedule's *endpoint*, so a figure reading
    it would report a constant on every epoch of every run."""
    captured: dict = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(plotting, "build_diagnostic_figure", spy)
    module = task(
        hparams={
            "kld_beta": 1.0,
            "beta_schedule": {
                "kind": "linear_warmup", "start": 1e-4, "end": 0.5, "warmup_epochs": 10,
            },
        }
    )
    callback = LagAttnRwsPlotCallback(tmp_path, num_examples=1)

    callback.on_validation_epoch_end(_trainer_with_batch(stub_batch), module)

    assert captured["beta"] == pytest.approx(module._resolve_beta(module.current_epoch))
    assert captured["beta"] != pytest.approx(float(module.hparams["kld_beta"]))


def test_it_is_silent_off_rank_zero(tmp_path, task, stub_batch):
    logger = FakeMLflowLogger()
    callback = LagAttnRwsPlotCallback(tmp_path, mlflow_logger=logger)

    callback.on_validation_epoch_end(
        _trainer_with_batch(stub_batch, is_global_zero=False), task()
    )

    assert list(callback.output_dir.glob("*.pdf")) == []
    assert logger.experiment.calls == []


def test_it_is_silent_during_the_sanity_pass(tmp_path, task, stub_batch):
    """The sanity pass runs before epoch 0 and would write a figure stamped like a real one."""
    callback = LagAttnRwsPlotCallback(tmp_path)

    callback.on_validation_epoch_end(
        _trainer_with_batch(stub_batch, sanity_checking=True), task()
    )

    assert list(callback.output_dir.glob("*.pdf")) == []


def test_it_honours_the_plot_frequency_in_both_directions(tmp_path, task, stub_batch):
    module = task()
    callback = LagAttnRwsPlotCallback(tmp_path, plot_frequency=5, num_examples=1)

    callback.on_validation_epoch_end(_trainer_with_batch(stub_batch, current_epoch=0), module)
    assert list(callback.output_dir.glob("*.pdf")) == []

    callback.on_validation_epoch_end(_trainer_with_batch(stub_batch, current_epoch=4), module)
    assert len(list(callback.output_dir.glob("*.pdf"))) == 1


def test_a_failure_inside_the_figure_never_reaches_the_training_loop(
    tmp_path, task, stub_batch, monkeypatch
):
    """A diagnostic figure is not worth a failed multi-day fit."""

    def explode(**_kwargs):
        raise RuntimeError("deliberate plotting failure")

    monkeypatch.setattr(plotting, "build_diagnostic_figure", explode)
    callback = LagAttnRwsPlotCallback(tmp_path)

    callback.on_validation_epoch_end(_trainer_with_batch(stub_batch), task())

    assert list(callback.output_dir.glob("*.pdf")) == []


def test_a_failing_validation_loader_never_reaches_the_training_loop(tmp_path, task):
    """The batch fetch builds a fresh iterator over the validation loader every plot epoch.
    Worker crashes and HDF5 errors surface there, not inside the figure, so the guard has to
    cover the fetch as well -- otherwise a figure aborts a multi-day fit.
    """

    class ExplodingLoader:
        def __iter__(self):
            raise RuntimeError("DataLoader worker (pid 1234) exited unexpectedly")

    callback = LagAttnRwsPlotCallback(tmp_path)
    trainer = FakeTrainer()
    trainer.val_dataloaders = [ExplodingLoader()]  # type: ignore[attr-defined]

    callback.on_validation_epoch_end(trainer, task())

    assert list(callback.output_dir.glob("*.pdf")) == []


def test_a_missing_validation_loader_is_not_an_error(tmp_path, task):
    callback = LagAttnRwsPlotCallback(tmp_path)
    trainer = FakeTrainer()
    trainer.val_dataloaders = None  # type: ignore[attr-defined]

    callback.on_validation_epoch_end(trainer, task())

    assert list(callback.output_dir.glob("*.pdf")) == []
