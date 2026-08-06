r"""The validation diagnostic figure, and the guards that keep it out of the training loop's way.

Two kinds of test here. The builder is driven directly from a real forward pass, which is what
pins the *content* -- that the forecast is drawn in bpm, that its windows tile the recording
without overlapping, that the untrained anchors are gone from the maps rather than merely shaded,
and that a lag axis says which of the two lag quantities it shows. The callback is driven through
its real hook with a fake trainer, which is what pins the *behaviour* -- silent off rank zero,
silent during the sanity pass, silent between plot epochs, and never raising into a fit.

Both directions are asserted wherever a single direction would pass vacuously: the bpm test also
checks the no-statistics fallback, the source-trace test also checks the batch that carries no
``up``, and the frequency test also checks the epoch that should fire.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.nets.lag_report import COMPENSATED_LAG_AXIS_LABEL  # noqa: E402
from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback  # noqa: E402
from train.test_utils import FakeMLflowLogger, FakeTrainer  # noqa: E402

#: Plausible raw-signal scales: the loader z-scores each by one global mean and standard
#: deviation, so inverting either is an affine map with its two numbers.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: Raw sampling rate, restated rather than imported: the page's second-axis arithmetic is what is
#: under test, and borrowing its own constant would make these assertions circular.
_FS_RAW = 4.0


def _forward(task_module: Any, batch: Any) -> dict:
    """Run the net once on a stub batch and return everything the builder needs.

    Through ``_build_forward_inputs``, which is the seam the callback itself now uses: a helper
    that assembled the inputs its own way could go on passing after the callback's route to them
    had changed.
    """
    model = task_module.orig_model
    with torch.no_grad():
        outs = model(*task_module._build_forward_inputs(batch))
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
        up_raw=batch.up,
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
        ax = _axes_titled(figure, "Forecast")
        assert "bpm" in ax.get_ylabel()
    finally:
        plt.close(figure)


def test_without_statistics_the_forecast_panel_says_normalised_instead_of_lying(task, stub_batch):
    """The other direction: no statistics means z-units, labelled as such, never mislabelled
    bpm -- so the test above is about the conversion and not about the word."""
    figure = _build(task(), stub_batch, normalization_stats=None)
    try:
        ax = _axes_titled(figure, "Forecast")
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
        ax = _axes_titled(figure, "Forecast")
        drawn = np.asarray(ax.lines[0].get_ydata(), dtype=float)  # the truth, plotted first
        expected = (
            stub_batch.fhr[0].numpy() * (_STATS["fhr"]["std"] + 1e-8) + _STATS["fhr"]["mean"]
        )
        # The truth is drawn only where a window covers it, so the comparison is on the covered
        # positions -- and asserting there are some is what stops an all-NaN curve passing.
        covered = np.isfinite(drawn)
        assert covered.any(), "the truth was drawn nowhere"
        assert np.allclose(drawn[covered], expected[covered], atol=1e-3)
    finally:
        plt.close(figure)


def test_the_forecast_windows_tile_the_recording_without_overlapping(task, stub_batch):
    """One anchor forecasts $H \\cdot R$ raw samples, so the anchors whose windows abut are spaced
    exactly $H$ apart. Drawn any other way the panel would show the same instant twice, from two
    different latents, with nothing saying which curve belonged to which."""
    module = task()
    figure = _build(module, stub_batch)
    try:
        geometry = module.orig_model.geometry
        anchors = list(range(geometry.warmup, geometry.t_valid, geometry.horizon))
        assert anchors, "the tiny geometry must still fit at least one window"
        edges = [geometry.future_block_start(anchor) / _FS_RAW for anchor in anchors]
        edges.append(
            (geometry.future_block_start(anchors[-1]) + geometry.horizon * geometry.r) / _FS_RAW
        )

        ax = _axes_titled(figure, "Forecast")
        # truth, base mean, full mean, then one vertical rule per window boundary
        drawn_edges = [float(line.get_xdata()[0]) for line in ax.lines[3:]]
        assert drawn_edges == pytest.approx(edges)

        # The support of the base mean: exactly the tiled samples, in one unbroken run -- which is
        # what "consecutive and non-overlapping" means once the NaN gaps are accounted for.
        base = np.asarray(ax.lines[1].get_ydata(), dtype=float)
        covered = np.flatnonzero(np.isfinite(base))
        assert covered.size == len(anchors) * geometry.horizon * geometry.r
        assert np.array_equal(covered, np.arange(covered[0], covered[-1] + 1))
        assert covered[0] == geometry.future_block_start(anchors[0])
    finally:
        plt.close(figure)


def test_the_untrained_anchors_are_cut_from_the_maps_rather_than_shaded_over(task, stub_batch):
    """Shading leaves the columns in the array, where they still set the colour scale: a warm-up
    transient then compresses every trained anchor into the bottom of the colormap. The axes still
    span the whole recording, so the rows stay column-aligned with the two raw ones above."""
    module = task()
    figure = _build(module, stub_batch)
    try:
        geometry = module.orig_model.geometry
        t_max = geometry.raw_len / _FS_RAW
        step = t_max / geometry.t
        warmup_sec, tail_sec = geometry.warmup * step, geometry.t_valid * step
        trained = geometry.t_valid - geometry.warmup

        expected = {
            "Target-only latent state": (warmup_sec, tail_sec, trained),
            "Per-dimension source-conditioned KL": (warmup_sec, tail_sec, trained),
            "$\\widetilde K": (warmup_sec, tail_sec, trained),
            # Attention is a property of the source stream and is defined at every step; only the
            # two KL maps are identically zero in the tail by construction of the mask.
            "Lag attention": (warmup_sec, t_max, geometry.t - geometry.warmup),
        }
        for prefix, (left, right, columns) in expected.items():
            ax = _axes_titled(figure, prefix)
            image = ax.images[0]
            assert image.get_extent()[:2] == pytest.approx((left, right)), prefix
            assert image.get_array().shape[1] == columns, prefix
            # One grey span per margin the row leaves empty, so a blank strip reads as "cut
            # deliberately" rather than as a panel that failed to draw.
            assert len(ax.patches) == (1 if right == pytest.approx(t_max) else 2), prefix

        for ax in figure.axes:
            if ax.get_title():
                assert ax.get_xlim() == pytest.approx((0.0, t_max)), ax.get_title()
    finally:
        plt.close(figure)


def test_the_first_row_draws_the_source_trace_beside_the_target(task, stub_batch):
    """The lag-attention and lag-KL rows are statements about UP, and a reader cannot check one
    against a trace that is not on the page. Both directions: a batch without ``up`` still builds,
    with no orphaned second axis claiming a signal that was never drawn."""
    figure = _build(task(), stub_batch)
    try:
        ax = _axes_titled(figure, "Raw target FHR")
        assert "bpm" in ax.get_ylabel()
        twins = [child for child in figure.axes if child.get_ylabel().startswith("UP")]
        assert len(twins) == 1 and twins[0].get_ylabel() == "UP (mmHg)"
        drawn = np.asarray(twins[0].lines[0].get_ydata(), dtype=float)
        expected = stub_batch.up[0].numpy() * (_STATS["up"]["std"] + 1e-8) + _STATS["up"]["mean"]
        assert np.allclose(drawn, expected, atol=1e-3)
    finally:
        plt.close(figure)

    figure = _build(task(), stub_batch, up_raw=None)
    try:
        assert not [child for child in figure.axes if child.get_ylabel().startswith("UP")]
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
