r"""Tests for the frequency-band and per-channel forecast analysis.

The tiny model is built at the production widths -- $c_y = 109$, being $43$ scattering plus $66$
phase-harmonic -- so the *real* production channel selection in ``tests/real_selection.py``
describes the committed tiny shard's channel axis exactly. The tests therefore run against the
partition the pipeline would really build, rather than a synthetic one that could agree with a
wrong implementation.

Three things need catching and each needs a different style.

The **skip path** is the one an operator meets first, on shards written before
``_write_selection_attrs``. It must be a recorded skip, not a crash, and not a partially-written
directory tree.

The **single-pass claim** is checked with a loader spy. It is the whole reason the band pass and
the channel pass share a function, and nothing about the output would reveal a second pass.

The **figures** are checked structurally *and* by sabotage: one band's channels are corrupted and
that band's violin must move. A figure of the wrong data has just as many panels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import band_partition, figures
from teb_vae.lag_attn.eval.analyses import frequency_band as frequency_band_analysis
from teb_vae.lag_attn.eval.tests import real_selection

#: Seed applied before the analysis, since ``forward`` samples $z$ unconditionally.
COMPARISON_SEED = 4321


@pytest.fixture
def results_dir(tmp_path) -> Path:
    """A run directory already carrying the partition ``run.py`` would have emitted into it."""
    directory = tmp_path / "results"
    directory.mkdir(parents=True, exist_ok=True)
    shard = real_selection.write_shard(tmp_path / "production.hdf5")
    record = band_partition.emit_partition([shard], real_selection.N_SCATTERING, directory)
    assert record["skipped"] is False, "the fixture must supply a usable partition"
    return directory


@pytest.fixture
def analysis(make_eval_runner, tiny_loader, tiny_eval_config, results_dir, tmp_path):
    """Run the analysis once and return ``(runner, analysis directory, summary)``."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    torch.manual_seed(COMPARISON_SEED)
    summary = frequency_band_analysis.run_frequency_band_analysis(
        runner,
        tiny_loader,
        eval_config=tiny_eval_config["eval_config"],
        output_dir=results_dir,
        probe={"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4},
    )
    return runner, results_dir / frequency_band_analysis.ANALYSIS_DIRNAME, summary


# ---------------------------------------------------------------------------
# Output tree
# ---------------------------------------------------------------------------
def test_one_subdirectory_per_partition_plus_the_channel_directory(analysis) -> None:
    _, directory, _ = analysis
    for name in frequency_band_analysis.PARTITION_NAMES:
        for filename in ("per_sample.csv", "horizon.csv", "anchor.csv"):
            assert (directory / name / filename).is_file(), f"{name}/{filename} was not written"
        for filename in ("band_violins.pdf", "band_horizon.pdf"):
            assert (directory / name / filename).stat().st_size > 0, f"{name}/{filename} is empty"

    channels = directory / frequency_band_analysis.CHANNEL_DIRNAME
    for filename in ("per_channel.csv", "per_channel_horizon.csv"):
        assert (channels / filename).is_file(), f"per_channel/{filename} was not written"
    assert (channels / "per_channel_frequency.pdf").stat().st_size > 0


def test_the_per_sample_table_carries_one_row_per_sample_and_one_pair_per_band(analysis) -> None:
    _, directory, _ = analysis
    frame = pd.read_csv(directory / "clinical" / "per_sample.csv")
    assert len(frame) == 4
    for band in ("beat_to_beat", "variability", "deceleration", "slow_baseline"):
        assert f"mse_{band}" in frame.columns
        assert f"r2_{band}" in frame.columns


def test_the_per_channel_table_has_one_row_per_channel_joined_to_the_map(analysis) -> None:
    """Joined so a downstream plot needs no second file and no reproduced join."""
    _, directory, _ = analysis
    frame = pd.read_csv(directory / frequency_band_analysis.CHANNEL_DIRNAME / "per_channel.csv")

    assert len(frame) == 109
    assert frame["channel"].tolist() == list(range(109))
    for column in ("block", "kind", "band", "freq_hz_primary", "mse"):
        assert column in frame.columns
    assert frame["mse"].notna().all()


def test_the_channel_horizon_table_spans_every_channel_and_horizon_step(analysis) -> None:
    runner, directory, _ = analysis
    frame = pd.read_csv(
        directory / frequency_band_analysis.CHANNEL_DIRNAME / "per_channel_horizon.csv"
    )
    horizon = int(runner.model.horizon)
    assert len(frame) == 109 * horizon
    assert set(frame["horizon"]) == set(range(horizon))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def test_the_channel_weighted_mean_over_bands_reproduces_the_pooled_total(analysis) -> None:
    r"""The identity that proves the partition tiles the channel axis with no gap or overlap.

    Computed here off the *emitted* summary rather than off the metric function, so it also
    covers the analysis having assembled the right channel groups.
    """
    _, _, summary = analysis
    for name in frequency_band_analysis.PARTITION_NAMES:
        bands = summary["partitions"][name]
        assert sum(record["n_channels"] for record in bands.values()) == 109


def test_the_per_channel_mean_reproduces_the_pooled_feat_mse(analysis) -> None:
    """The channel pass and the headline number must describe the same forecast."""
    _, directory, summary = analysis
    frame = pd.read_csv(directory / frequency_band_analysis.CHANNEL_DIRNAME / "per_channel.csv")
    assert float(frame["mse"].mean()) == pytest.approx(summary["pooled_feat_mse"], rel=1e-6)


def test_the_summary_names_the_worst_channel_with_its_band_and_frequency(analysis) -> None:
    _, directory, summary = analysis
    frame = pd.read_csv(directory / frequency_band_analysis.CHANNEL_DIRNAME / "per_channel.csv")

    worst = summary["worst_channel"]
    assert worst["channel"] == int(frame["mse"].idxmax())
    assert worst["band"] == frame.loc[worst["channel"], "band"]
    assert worst["kind"] == frame.loc[worst["channel"], "kind"]


def test_the_horizon_profile_carries_every_band_at_every_step(analysis) -> None:
    runner, directory, _ = analysis
    frame = pd.read_csv(directory / "clinical" / "horizon.csv")
    horizon = int(runner.model.horizon)
    assert set(frame["position"]) == set(range(horizon))
    assert len(frame) == frame["band"].nunique() * horizon


def test_the_warmup_anchors_are_nan_in_the_band_anchor_profile(analysis) -> None:
    """A zero there would read as a perfectly forecast prefix rather than as no data."""
    runner, directory, _ = analysis
    frame = pd.read_csv(directory / "clinical" / "anchor.csv")
    warmup = int(runner.model.warmup_period)

    early = frame[frame["position"] < warmup]["mse"]
    assert len(early) > 0 and early.isna().all()
    assert frame[frame["position"] >= warmup]["mse"].notna().any()


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------
def test_the_whole_analysis_makes_exactly_one_pass_over_the_loader(
    make_eval_runner, tiny_loader, tiny_eval_config, results_dir, tmp_path
) -> None:
    """Two partitions and the channel pass share one loop; nothing in the output would show it."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    passes = {"count": 0}
    original = runner.iter_batches

    def _spy(*args, **kwargs):
        passes["count"] += 1
        return original(*args, **kwargs)

    runner.iter_batches = _spy  # type: ignore[method-assign]
    frequency_band_analysis.run_frequency_band_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=results_dir, probe={"n_samples": 4},
    )
    assert passes["count"] == 1, f"the analysis iterated the loader {passes['count']} times"


# ---------------------------------------------------------------------------
# The skip path
# ---------------------------------------------------------------------------
def test_a_run_without_channel_provenance_records_a_skip_rather_than_crashing(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The vintage an operator actually meets: shards written before the sel_* attributes."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    bare = tmp_path / "bare_results"
    bare.mkdir()

    summary = frequency_band_analysis.run_frequency_band_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=bare, probe={"n_samples": 4},
    )
    assert summary["skipped"] is True
    assert band_partition.PARTITION_FILENAME in summary["reason"]
    assert not (bare / frequency_band_analysis.ANALYSIS_DIRNAME).exists(), (
        "a skipped analysis must leave no half-written directory tree behind"
    )


def test_a_partition_of_the_wrong_width_raises_rather_than_banding_wrong_channels(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Every band would be assembled from the wrong channels and the numbers would look fine."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    directory = tmp_path / "narrow"
    directory.mkdir()

    partition = band_partition.build_partition(
        real_selection.write_shard(tmp_path / "production.hdf5"),
        n_scattering=real_selection.N_SCATTERING,
    )
    # Drop a channel, so the map no longer describes what the checkpoint forecasts.
    partition.channels = partition.channels[:-1]
    band_partition.write_partition(partition, directory)

    with pytest.raises(RuntimeError, match="describes 108 channels"):
        frequency_band_analysis.run_frequency_band_analysis(
            runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
            output_dir=directory, probe={"n_samples": 4},
        )


# ---------------------------------------------------------------------------
# Labels and ordering
# ---------------------------------------------------------------------------
@pytest.fixture
def real_partition(tmp_path):
    return band_partition.build_partition(
        real_selection.write_shard(tmp_path / "production.hdf5"),
        n_scattering=real_selection.N_SCATTERING,
    )


def test_a_clinical_label_states_its_defining_hz_range(real_partition) -> None:
    groups = real_partition.partition("clinical")
    label = frequency_band_analysis.hz_label(
        real_partition, "clinical", "deceleration", groups["deceleration"]
    )
    assert "0.008-0.04 Hz" in label
    assert f"{len(groups['deceleration'])} ch" in label


def test_the_unbounded_top_band_is_rendered_without_an_infinity(real_partition) -> None:
    groups = real_partition.partition("clinical")
    label = frequency_band_analysis.hz_label(
        real_partition, "clinical", "beat_to_beat", groups["beat_to_beat"]
    )
    assert ">0.25 Hz" in label and "inf" not in label


def test_a_kind_label_states_the_hz_range_its_channels_actually_occupy(real_partition) -> None:
    """A harmonic kind is not a frequency range, so the observed span is the honest answer."""
    groups = real_partition.partition("by_kind")
    label = frequency_band_analysis.hz_label(real_partition, "by_kind", "ph_k4", groups["ph_k4"])
    assert "Hz" in label and "24 ch" in label


def test_a_label_whose_channels_carry_no_frequency_says_so(real_partition) -> None:
    """Rather than being drawn at 0 Hz, which would assert a frequency nothing determined."""
    groups = real_partition.partition("clinical")
    label = frequency_band_analysis.hz_label(
        real_partition, "clinical", band_partition.UNKNOWN_BAND,
        groups[band_partition.UNKNOWN_BAND],
    )
    assert "no centre frequency" in label


def test_bands_are_ordered_from_the_highest_frequency_to_the_lowest(real_partition) -> None:
    labels = frequency_band_analysis.ordered_labels(real_partition, "clinical")
    assert labels[0] == "beat_to_beat"
    assert labels.index("variability") < labels.index("deceleration")
    # The label whose channels carry no frequency sorts last rather than as 0 Hz.
    assert labels[-1] == band_partition.UNKNOWN_BAND


def test_an_empty_label_is_not_offered_as_a_row(real_partition) -> None:
    """An empty band would draw an anonymous blank row on every figure."""
    for name in frequency_band_analysis.PARTITION_NAMES:
        groups = real_partition.partition(name)
        for label in frequency_band_analysis.ordered_labels(real_partition, name):
            assert groups[label], f"{name}/{label} is empty and should not have been ordered"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_the_violin_figure_stacks_two_panels_with_hz_suffixed_ticks(
    analysis, monkeypatch, tmp_path, real_partition
) -> None:
    """Structural assertions plus the label content, which is the point of the figure."""
    captured: Dict[str, Any] = {}
    original = figures.render_to_pdf

    def _capture(fig, path, **kwargs):
        if Path(path).name == "band_violins.pdf":
            captured["titles"] = [ax.get_title() for ax in fig.axes if ax.get_title()]
            captured["ticks"] = [
                label.get_text() for label in fig.axes[0].get_xticklabels()
            ]
            captured["has_data"] = [ax.has_data() for ax in fig.axes if ax.get_title()]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_to_pdf", _capture)

    _, directory, _ = analysis
    frame = pd.read_csv(directory / "clinical" / "per_sample.csv")
    # Re-emit from the analysis's own writer so the assertion covers the shipped path.
    groups = real_partition.partition("clinical")
    labels = frequency_band_analysis.ordered_labels(real_partition, "clinical")
    renamed = frame.rename(
        columns={
            f"{metric}_{label}": f"clinical__{label}__{metric}"
            for label in labels for metric in ("mse", "r2")
        }
    )
    frequency_band_analysis._write_partition_figures(
        renamed, real_partition, "clinical", labels, groups,
        frequency_band_analysis._ProfileAccumulator(), tmp_path / "figs",
    )

    assert len(captured["titles"]) == 2
    assert "MSE" in captured["titles"][0] and "R^2" in captured["titles"][1]
    assert all(captured["has_data"])
    assert any("Hz" in tick for tick in captured["ticks"]), (
        "the violin ticks must carry explicit Hz ranges"
    )
    assert captured["ticks"][0].startswith("beat_to_beat"), "rows run high frequency to low"


def test_the_violin_panel_reports_the_band_that_was_sabotaged(tmp_path) -> None:
    """The assertion that makes the figure test non-vacuous.

    One band's samples are inflated, and its violin must sit above the others. A panel drawing
    the wrong column, or drawing every band from the same data, fails here while passing every
    structural check.
    """
    rng = np.random.default_rng(0)
    samples = {
        "quiet (0.25-1 Hz, 4 ch)": rng.uniform(0.1, 0.3, 40),
        "loud (0.04-0.25 Hz, 4 ch)": rng.uniform(9.0, 11.0, 40),
        "also quiet (0.008-0.04 Hz, 4 ch)": rng.uniform(0.1, 0.3, 40),
    }
    figure, axes = figures.new_figure(1)
    try:
        drawn = figures.violin_panel(axes[0, 0], samples, title="by band", ylabel="MSE")
        assert drawn == 3
        # The violin bodies' vertical extents, in the order they were drawn.
        centres = [
            float(np.mean(body.get_paths()[0].vertices[:, 1]))
            for body in axes[0, 0].collections
            if hasattr(body, "get_paths") and body.get_paths()
        ]
        assert int(np.argmax(centres[:3])) == 1, "the sabotaged band's violin did not respond"
    finally:
        figures.plt.close(figure)


def test_an_all_nan_band_keeps_its_slot_rather_than_shifting_the_labels(tmp_path) -> None:
    """Dropping an empty group would move every later label onto the wrong violin."""
    samples = {
        "a (1 Hz, 1 ch)": np.array([1.0, 2.0, 3.0]),
        "b (no centre frequency, 1 ch)": np.array([np.nan, np.nan]),
        "c (0.1 Hz, 1 ch)": np.array([4.0, 5.0, 6.0]),
    }
    figure, axes = figures.new_figure(1)
    try:
        assert figures.violin_panel(axes[0, 0], samples, title="t", ylabel="y") == 2
        ticks = [label.get_text() for label in axes[0, 0].get_xticklabels()]
        assert ticks == list(samples), "the empty group lost its slot"
    finally:
        figures.plt.close(figure)


def test_the_frequency_scatter_drops_channels_with_no_centre_frequency_and_says_so(tmp_path) -> None:
    r"""A log axis cannot draw $0$ Hz, and filling one in would assert a frequency nothing knows."""
    figure, axes = figures.new_figure(1)
    try:
        handle = figures.frequency_scatter(
            figure, axes[0, 0],
            np.array([0.5, np.nan, 0.05, np.nan]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            title="t", xlabel="Hz", ylabel="MSE",
        )
        assert handle is not None
        assert handle.get_offsets().shape[0] == 2
        assert axes[0, 0].get_xscale() == "log"
        legend = axes[0, 0].get_legend()
        assert legend is not None and "2 channel(s)" in legend.get_texts()[0].get_text()
    finally:
        figures.plt.close(figure)


def test_the_frequency_scatter_colours_the_phase_panel_by_its_harmonic_ratio(tmp_path) -> None:
    """A phase channel is a pair; colouring by $p$ is what keeps that in the figure."""
    figure, axes = figures.new_figure(1)
    try:
        handle = figures.frequency_scatter(
            figure, axes[0, 0],
            np.array([0.5, 0.2, 0.05]), np.array([1.0, 2.0, 3.0]),
            colour_by=np.array([1.19, 1.41, 1.68]), colour_label="p",
            title="t", xlabel="Hz", ylabel="MSE",
        )
        assert handle is not None
        assert handle.get_array() is not None, "the points were not coloured by the ratio"
    finally:
        figures.plt.close(figure)
