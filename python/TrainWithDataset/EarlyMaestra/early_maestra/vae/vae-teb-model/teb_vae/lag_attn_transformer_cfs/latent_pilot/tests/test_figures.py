r"""The three figures actually render, on hand-built frames.

**Execution-machine test**, because it draws: it needs matplotlib and writes files, which the
minimal logic subset does neither of. It needs no model, no checkpoint, no fixtures and no GPU --
every input here is a small table built in the file::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_figures.py -q

What it establishes is that the panels survive the shapes a real run hands them, which is where a
figure generator fails in practice rather than in principle: a group with no finite value, a bin no
recording occupied, a recording present in one model's table and not the other's, and a projection
whose explained variance is not a number. Every one of those is a legitimate state of a small
cohort, and a run must not lose a completed pipeline at its final step because of one.

What it does not establish is that any figure is *right*: the numbers are invented and nothing here
reads a value off a rendered panel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze, data
from teb_vae.lag_attn_transformer_cfs.latent_pilot import report as pilot_report
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

D_Z = 4
BIN_HOURS = 0.5
PRESERVATION_HOURS = 3.0
SUPERVISED_HOURS = 1.0


@pytest.fixture(autouse=True)
def png_figures():
    """Render PNG rather than the run default, so a test writes one small file per figure.

    Restored afterwards: the format is process-wide state, and leaving it changed would make a
    later test in the same session write something it did not ask for.
    """
    from teb_vae.lag_attn.eval import figures

    previous = figures.active_figure_format()
    pilot_report.configure_figures("png")
    yield
    figures.set_figure_format(previous)


def _bags(n_healthy: int = 4, n_adverse: int = 3, *, seed: int = 0):
    """A final-hour bag table and its standardized vectors."""
    generator = np.random.default_rng(seed)
    rows = []
    for index in range(n_healthy):
        rows.append({data.GUID_COLUMN: f"H{index}", labels.CLASS_COLUMN: "healthy",
                     data.OUTCOME_COLUMN: 0})
    for index in range(n_adverse):
        name = "acidosis" if index % 2 == 0 else "hie"
        rows.append({data.GUID_COLUMN: f"A{index}", labels.CLASS_COLUMN: name,
                     data.OUTCOME_COLUMN: 1})
    frame = pd.DataFrame(rows)
    return frame, generator.normal(size=(len(frame), D_Z))


def _projection(seed: int = 0, *, ratios=(0.41, 0.19)) -> analyze.Projection:
    """A two-component map, built directly rather than fitted: the fit is tested elsewhere."""
    generator = np.random.default_rng(seed)
    axes, _ = np.linalg.qr(generator.normal(size=(D_Z, 2)))
    return analyze.Projection(
        mean=np.zeros(D_Z),
        components=axes.T.copy(),
        explained_variance_ratio=np.asarray(ratios, dtype=np.float64),
        record={"population": "train", "versions": ["pretrained", "adapted"]},
    )


def _bins(frame: pd.DataFrame, *, occupied=(0, 1, 4), seed: int = 0) -> pd.DataFrame:
    """A scored bin table that occupies some bins and not others."""
    generator = np.random.default_rng(seed)
    rows = []
    for _index, row in frame.iterrows():
        for position in occupied:
            rows.append({
                data.GUID_COLUMN: row[data.GUID_COLUMN],
                labels.CLASS_COLUMN: row[labels.CLASS_COLUMN],
                data.OUTCOME_COLUMN: row[data.OUTCOME_COLUMN],
                data.BIN_COLUMN: int(position),
                data.BIN_LABEL_COLUMN: f"bin {position}",
                analyze.SCORE_COLUMN: float(generator.normal()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Figure 1
# =============================================================================
def test_the_latent_space_figure_renders_both_versions(tmp_path):
    frame, values = _bags()
    written = pilot_report.figure_latent_space(
        {"pretrained": (frame, values), "adapted": (frame, values + 0.3)},
        _projection(), tmp_path, n_arrows=3, seed=0,
    )
    assert Path(written).is_file() and Path(written).stat().st_size > 0
    assert Path(written).parent.name == pilot_report.FIGURE_DIRNAME
    assert Path(written).suffix == ".png"


def test_the_latent_space_figure_refuses_misaligned_values(tmp_path):
    frame, values = _bags()
    with pytest.raises(PilotConfigError):
        pilot_report.figure_latent_space(
            {"pretrained": (frame, values[:-1])}, _projection(), tmp_path
        )


def test_a_projection_without_finite_variance_still_renders(tmp_path):
    """A degenerate map is a legitimate outcome on a tiny cohort; the label says so instead of
    printing a percentage that was never computed."""
    frame, values = _bags()
    written = pilot_report.figure_latent_space(
        {"pretrained": (frame, values)},
        _projection(ratios=(np.nan, np.nan)), tmp_path,
    )
    assert Path(written).is_file()


def test_the_coverage_view_renders_on_the_same_map(tmp_path):
    """The control §7.3 asks for: the same coordinates, coloured by ascertainment and by how late
    each recording was still observed."""
    frame, values = _bags()
    frame["bg_label"] = [True, False, None, True, True, False, None]
    frame["cs_label"] = [False, False, True, True, None, True, False]
    frame["last_anchor_hours"] = [0.1, 0.2, 0.4, 0.5, 0.7, 0.9, float("nan")]
    written = pilot_report.figure_coverage_space(frame, values, _projection(), tmp_path)
    assert Path(written).is_file() and Path(written).stat().st_size > 0
    assert Path(written).name.startswith(pilot_report.FIGURE_COVERAGE_SPACE)


def test_the_coverage_view_refuses_misaligned_values(tmp_path):
    frame, values = _bags()
    with pytest.raises(PilotConfigError):
        pilot_report.figure_coverage_space(frame, values[:-1], _projection(), tmp_path)


# =============================================================================
# Figure 2
# =============================================================================
def test_the_supervised_axis_figure_renders_one_panel_per_model(tmp_path):
    frame, _values = _bags()
    scores = np.linspace(-2.0, 2.0, len(frame))
    written = pilot_report.figure_supervised_axis(
        frame,
        {"frozen": scores, "adapted": scores * 1.5},
        tmp_path,
        metrics={"frozen": {"auroc": 0.6, "average_precision": 0.3, "prevalence": 0.43,
                            "n_recordings": len(frame)}},
        thresholds={"frozen": 0.0, "adapted": 0.1},
    )
    assert Path(written).is_file() and Path(written).stat().st_size > 0


def test_the_supervised_axis_figure_refuses_scores_that_do_not_align(tmp_path):
    frame, _values = _bags()
    with pytest.raises(PilotConfigError):
        pilot_report.figure_supervised_axis(
            frame, {"frozen": np.zeros(len(frame) - 1)}, tmp_path
        )


def test_a_class_with_no_recording_leaves_its_slot_rather_than_shifting_the_others(tmp_path):
    """A cohort with no HIE case must not silently relabel the acidosis violin."""
    frame, _values = _bags(n_healthy=3, n_adverse=0)
    written = pilot_report.figure_supervised_axis(
        frame, {"frozen": np.zeros(len(frame))}, tmp_path
    )
    assert Path(written).is_file()


# =============================================================================
# Figure 3
# =============================================================================
def test_the_trajectory_figure_renders_with_gaps_and_counts(tmp_path):
    frame, _values = _bags()
    scored = _bins(frame)
    bands = analyze.group_bands(
        scored, group_column=labels.CLASS_COLUMN, resamples=100, seed=0,
        supervised=analyze.supervised_bins(
            bin_hours=BIN_HOURS, supervised_hours=SUPERVISED_HOURS
        ),
    )
    written = pilot_report.figure_trajectories(
        {"frozen": (bands, scored)},
        tmp_path,
        bin_hours=BIN_HOURS,
        preservation_hours=PRESERVATION_HOURS,
        supervised_hours=SUPERVISED_HOURS,
        n_traces=3,
        seed=0,
    )
    assert Path(written).is_file() and Path(written).stat().st_size > 0


def test_the_trajectory_figure_draws_a_raw_excerpt_when_one_is_supplied(tmp_path):
    frame, _values = _bags()
    scored = _bins(frame)
    bands = analyze.group_bands(scored, resamples=100, seed=0)
    guid = pilot_report.select_traces(
        scored.drop_duplicates(subset=[data.GUID_COLUMN]), n=1, seed=0
    )[0]
    hours = np.linspace(-PRESERVATION_HOURS, -0.1, 64)
    written = pilot_report.figure_trajectories(
        {"frozen": (bands, scored)},
        tmp_path,
        bin_hours=BIN_HOURS,
        preservation_hours=PRESERVATION_HOURS,
        supervised_hours=SUPERVISED_HOURS,
        n_traces=2,
        seed=0,
        excerpts={guid: {"hours": hours, "fhr": np.sin(hours), "up": np.cos(hours)}},
    )
    assert Path(written).is_file()


def test_an_empty_band_table_still_produces_a_figure(tmp_path):
    """A split that produced no estimable bin is a result to look at, not a crash at the final
    step of a multi-stage run."""
    written = pilot_report.figure_trajectories(
        {"frozen": (pd.DataFrame(columns=["group", data.BIN_COLUMN, "mean", "lo", "hi",
                                          "n_recordings"]),
                    pd.DataFrame(columns=[data.GUID_COLUMN, data.BIN_COLUMN,
                                          analyze.SCORE_COLUMN]))},
        tmp_path,
        bin_hours=BIN_HOURS,
        preservation_hours=PRESERVATION_HOURS,
        supervised_hours=SUPERVISED_HOURS,
        n_traces=0,
    )
    assert Path(written).is_file()


# =============================================================================
# The report beside them
# =============================================================================
def test_the_report_is_written_beside_the_figures(tmp_path):
    frame, values = _bags()
    figure = pilot_report.figure_latent_space(
        {"pretrained": (frame, values)}, _projection(), tmp_path
    )
    written = pilot_report.write_report(
        {"figures": {pilot_report.FIGURE_LATENT_SPACE: str(figure)}}, tmp_path
    )
    assert written.name == pilot_report.REPORT_FILENAME
    text = written.read_text(encoding="utf-8")
    assert str(figure) in text
    assert pilot_report.MISSING in text
