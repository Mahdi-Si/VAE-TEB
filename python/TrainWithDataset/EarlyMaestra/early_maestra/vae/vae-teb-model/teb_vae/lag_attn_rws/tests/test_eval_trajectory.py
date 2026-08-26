r"""Assembling a recording's segments into one trajectory, and the three ways that goes wrong.

The per-anchor table's first general consumer, and every assertion here is about the assembly
rather than about the numbers it carries.

**An overlap is averaged, not duplicated.** Consecutive segments of one recording overlap by
construction, so the same absolute second is scored twice. Concatenating puts that second on the
trajectory twice at two different values, and every downstream mean weights it double -- while the
figure looks entirely normal.

**A gap is a break.** A recording missing an hour has no trajectory across that hour, and a line
drawn straight through it reads as a slow trend. The break is carried in the data (``gap_before_s``)
rather than only in the drawing, so an analysis reading the table sees it too.

**A join is recorded.** A step at a segment boundary is an artifact of assembly rather than a
physiological event, and the boundary indices are what let a reader tell the two apart. They are
asserted to land at the joins rather than merely to exist.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval.figures_seam import figure_filename
from teb_vae.lag_attn_rws.eval.analyses import trajectory as analysis
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP


def _anchors(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """A per-anchor table carrying only what this analysis reads."""
    frame = pd.DataFrame(rows)
    for column in ("kld_per_t", "mc_pred_gap"):
        if column not in frame.columns:
            frame[column] = np.arange(len(frame), dtype=np.float64)
    return frame


def _segment(guid: str, epoch: float, anchors: range, *, value: float) -> List[Dict[str, Any]]:
    """One segment's rows: a constant value, so an averaged overlap has a known answer."""
    return [
        {"guid": guid, "epoch": epoch, "anchor": anchor, "kld_per_t": value,
         "mc_pred_gap": value}
        for anchor in anchors
    ]


def _context(per_anchor: pd.DataFrame) -> Any:
    """An analysis context built by hand, with no model and no collection pass."""
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=pd.DataFrame(), per_anchor=per_anchor, record={}, retained={}, results={},
        vectors={},
    )
    return AnalysisContext(collection=collection, config={})


# =============================================================================
# The overlap
# =============================================================================
def test_overlapping_timesteps_from_adjacent_segments_are_averaged_not_duplicated() -> None:
    r"""Two segments of one recording, ten anchors each, offset by five anchors: five absolute
    seconds are scored twice. The assembled trajectory must hold fifteen rows, not twenty, and the
    five shared ones must carry the mean of the two values rather than either of them."""
    step = float(SECONDS_PER_STEP)
    per_anchor = _anchors(
        _segment("a", -100.0 * step, range(10), value=1.0)
        + _segment("a", -95.0 * step, range(10), value=3.0)
    )

    trajectory, _ = analysis.whole_delivery(per_anchor)

    assert len(trajectory) == 15, "the five shared seconds must be one row each, not two"
    overlapping = trajectory[trajectory["n_contributing"] > 1]
    assert len(overlapping) == 5
    assert list(overlapping["kld_per_t"]) == [pytest.approx(2.0)] * 5
    # And the non-overlapping ends keep their own segment's value.
    assert float(trajectory["kld_per_t"].iloc[0]) == pytest.approx(1.0)
    assert float(trajectory["kld_per_t"].iloc[-1]) == pytest.approx(3.0)


def test_the_absolute_coordinate_is_the_epoch_plus_four_seconds_per_anchor() -> None:
    per_anchor = _anchors(_segment("a", -1000.0, range(3), value=1.0))

    trajectory, _ = analysis.whole_delivery(per_anchor)

    assert list(trajectory["t_abs_sec"]) == [
        -1000.0, -1000.0 + SECONDS_PER_STEP, -1000.0 + 2 * SECONDS_PER_STEP
    ]
    # Negative before delivery, so hours before delivery is the sign-flipped figure over 3600.
    assert float(trajectory["hours_before_delivery"].iloc[0]) == pytest.approx(1000.0 / 3600.0)


# =============================================================================
# The gap
# =============================================================================
def test_a_gap_produces_a_break_rather_than_an_interpolation() -> None:
    """Two segments an hour apart. The trajectory must carry the distance across the gap so a
    reader -- and the figure -- can leave it as a break instead of drawing a line through it."""
    per_anchor = _anchors(
        _segment("a", -7200.0, range(5), value=1.0) + _segment("a", -3600.0, range(5), value=2.0)
    )

    trajectory, _ = analysis.whole_delivery(per_anchor)

    gaps = np.asarray(trajectory["gap_before_s"], dtype=np.float64)
    breaks = np.isfinite(gaps) & (gaps > analysis.BREAK_TOLERANCE_S)
    assert int(breaks.sum()) == 1
    assert float(gaps[breaks][0]) == pytest.approx(7200.0 - 3600.0 - 4 * SECONDS_PER_STEP)
    # Contiguous anchors are exactly one step apart and are not breaks.
    assert float(gaps[1]) == pytest.approx(SECONDS_PER_STEP)


def test_the_figure_leaves_the_gap_as_a_gap() -> None:
    """Matplotlib joins consecutive points whatever their spacing, so the break has to be put into
    the drawn data as ``NaN`` -- otherwise an hour of missing recording is drawn as a straight
    interpolation and reads as a slow trend."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    per_anchor = _anchors(
        _segment("a", -7200.0, range(5), value=1.0) + _segment("a", -3600.0, range(5), value=2.0)
    )
    trajectory, _ = analysis.whole_delivery(per_anchor)

    figure = analysis.build_profile_figure(
        analysis.within_segment_profile(per_anchor), trajectory, analysis.READOUTS[0]
    )
    try:
        drawn = [
            np.asarray(line.get_ydata(), dtype=np.float64) for line in figure.axes[-1].lines
        ]
    finally:
        shared_figures.plt.close(figure)

    assert drawn, "the whole-delivery panel drew nothing"
    assert all(int(np.isnan(values).sum()) == 1 for values in drawn)


# =============================================================================
# The joins
# =============================================================================
def test_the_boundary_indices_land_at_the_segment_joins() -> None:
    """One row per segment, each naming the position in the recording's own row order at which
    that segment's first anchor landed."""
    step = float(SECONDS_PER_STEP)
    per_anchor = _anchors(
        _segment("a", -100.0 * step, range(10), value=1.0)
        + _segment("a", -95.0 * step, range(10), value=3.0)
    )

    trajectory, boundaries = analysis.whole_delivery(per_anchor)

    assert len(boundaries) == 2
    assert list(boundaries["index"]) == [0, 5]
    times = np.asarray(trajectory["t_abs_sec"], dtype=np.float64)
    for row in boundaries.itertuples():
        assert times[row.index] == pytest.approx(row.t_abs_sec)


def test_two_recordings_are_assembled_independently() -> None:
    """A trajectory is per recording; merging two would put one delivery's anchors on another's
    axis wherever their epochs happened to coincide."""
    per_anchor = _anchors(
        _segment("a", -1000.0, range(4), value=1.0) + _segment("b", -1000.0, range(4), value=2.0)
    )

    trajectory, boundaries = analysis.whole_delivery(per_anchor)

    assert len(trajectory) == 8
    assert set(trajectory["guid"]) == {"a", "b"}
    assert list(trajectory[trajectory["guid"] == "a"]["kld_per_t"]) == [1.0] * 4
    assert set(boundaries["guid"]) == {"a", "b"}


# =============================================================================
# Within a segment
# =============================================================================
def test_the_within_segment_profile_reduces_per_recording_before_across_them() -> None:
    """One recording contributing three segments must count once per anchor, at their mean --
    otherwise a long recording decides the shape of the profile."""
    per_anchor = _anchors(
        _segment("a", -1000.0, range(3), value=1.0)
        + _segment("a", -2000.0, range(3), value=3.0)
        + _segment("b", -3000.0, range(3), value=10.0)
    )

    profile = analysis.within_segment_profile(per_anchor)
    anchor_zero = profile[(profile["anchor"] == 0) & (profile["metric"] == "kld_per_t")].iloc[0]

    # Two recordings, at 2.0 (the mean of 1 and 3) and 10.0 -- not three segments at 1, 3 and 10.
    assert int(anchor_zero["n_recordings"]) == 2
    assert float(anchor_zero["mean"]) == pytest.approx(6.0)
    assert float(anchor_zero["seconds_in_segment"]) == pytest.approx(0.0)


def test_the_profile_carries_both_readouts_against_seconds_in_segment() -> None:
    profile = analysis.within_segment_profile(_anchors(_segment("a", -1000.0, range(4), value=1.0)))

    assert set(profile["metric"]) == {"kld_per_t", "pred_gap_mc_nats"}
    assert list(profile[profile["metric"] == "kld_per_t"]["seconds_in_segment"]) == [
        0.0, SECONDS_PER_STEP, 2 * SECONDS_PER_STEP, 3 * SECONDS_PER_STEP
    ]


# =============================================================================
# What it writes
# =============================================================================
def test_the_analysis_writes_both_tables_the_boundaries_and_the_figure(tmp_path) -> None:
    step = float(SECONDS_PER_STEP)
    per_anchor = _anchors(
        _segment("a", -100.0 * step, range(10), value=1.0)
        + _segment("a", -95.0 * step, range(10), value=3.0)
        + _segment("b", -200.0 * step, range(10), value=2.0)
    )
    per_anchor["sample_index"] = np.repeat([0, 1, 2], 10)

    result = analysis.run_trajectory_analysis(
        _context(per_anchor), eval_config={}, output_dir=tmp_path, probe=None
    )

    directory = tmp_path / analysis.ANALYSIS_DIRNAME
    for name in (
        analysis.WITHIN_SEGMENT_FILENAME, analysis.WHOLE_DELIVERY_FILENAME,
        analysis.BOUNDARIES_FILENAME, analysis.SUMMARY_FILENAME,
        # One page per readout, so neither is drawn on the other's scale.
        *(
            figure_filename(f"{analysis.PROFILE_FIGURE}_{readout.slug}")
            for readout in analysis.READOUTS
        ),
    ):
        assert (directory / name).is_file(), name
    assert result["whole_delivery"]["n_timesteps"] == 25
    assert result["whole_delivery"]["n_overlapping_timesteps"] == 5
    assert result["whole_delivery"]["n_boundaries"] == 3
    assert result["composition"]["n_recordings"] == 2
    assert result["n_samples"] == 3
    # The parquet round-trips, which is why it is the format the heavier table uses.
    assert len(pd.read_parquet(directory / analysis.WHOLE_DELIVERY_FILENAME)) == 25


def test_an_empty_per_anchor_table_is_a_recorded_skip(tmp_path) -> None:
    result = analysis.run_trajectory_analysis(
        _context(pd.DataFrame()), eval_config={}, output_dir=tmp_path, probe=None
    )

    assert result["skipped"] is True
    assert result["n_samples"] is None
    assert not (tmp_path / analysis.ANALYSIS_DIRNAME).exists()


# =============================================================================
# End to end, on a real run
# =============================================================================
def test_the_real_run_assembles_every_recording(evaluated) -> None:
    """The fixture gives each recording two segments at different epochs, so the assembly has
    something to assemble and the summary's counts are checkable against the tables."""
    block = evaluated["summary"]["results"]["trajectory"]
    directory = evaluated["results_dir"] / analysis.ANALYSIS_DIRNAME
    trajectory = pd.read_parquet(directory / analysis.WHOLE_DELIVERY_FILENAME)
    summary = pd.read_csv(directory / analysis.SUMMARY_FILENAME)

    assert block.get("skipped") is not True
    assert block["composition"]["n_recordings"] == evaluated["summary"]["results"]["n_recordings"]
    assert block["whole_delivery"]["n_timesteps"] == len(trajectory)
    assert int(summary["n_segments"].sum()) == evaluated["summary"]["results"]["n_samples"]
    # One row per (recording, absolute second): the key the averaging exists to make unique.
    assert not trajectory.duplicated(subset=["guid", "t_abs_sec"]).any()
