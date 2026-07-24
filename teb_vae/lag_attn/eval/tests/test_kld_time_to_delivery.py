r"""Tests for the KLD-trajectory-to-delivery analysis.

Driven by hand-built per-sample frames, exactly as ``test_cross_subgroup`` is: the collection step
needs a model, but the binning, the trajectory summary and the per-window statistics do not, and
building the inputs from a model would test something else. Every statistic is checked against a
direct ``scipy`` / ``pandas`` computation on the same numbers rather than a recorded constant, so a
regression that reordered or dropped a group is caught rather than frozen in.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval import run as run_module
from teb_vae.lag_attn.eval.analyses import kld_time_to_delivery as kd
from teb_vae.lag_attn.eval.report import json_safe

#: A per-class, per-window sample count comfortably above ``MIN_GROUP_SIZE``.
PER_CELL = 12

#: Class base levels; ``hie`` is shifted up so a genuine separation exists to be found.
CLASS_LEVELS = {"healthy": 0.5, "acidosis": 0.7, "hie": 1.1}

#: Subgroup a class maps to, so a frame carries both axes.
CLASS_SUBGROUP = {"healthy": "healthy_no_bg_no_cs", "acidosis": "acidosis_cs", "hie": "hie_cs"}


def _frame(
    *,
    classes=("healthy", "acidosis", "hie"),
    hours=(0.25, 0.75, 1.25, 1.75, 2.25),
    separated: bool = True,
    with_epoch: bool = True,
    with_subgroup: bool = True,
    per_cell: int = PER_CELL,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic per-sample frame with a controlled class separation across time windows.

    Args:
        classes: Clinical classes to populate.
        hours: Distinct hours-before-delivery, one per $0.5$ h window.
        separated: Whether the classes are drawn at different levels or the same one.
        with_epoch: Whether to attach the ``epoch`` column at all.
        with_subgroup: Whether to attach a subgroup column.
        per_cell: Samples per (class, window) cell.
        seed: Seed, so the fixture is reproducible.

    Returns:
        The frame, shaped exactly as the collector would hand :func:`kd.emit_analysis`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    for cls in classes:
        level = CLASS_LEVELS[cls] if separated else 0.6
        for hours_before in hours:
            for _ in range(per_cell):
                row = {
                    "sample_index": index,
                    "guid": f"g{index:04d}",
                    "source_file": f"{CLASS_SUBGROUP[cls]}.hdf5",
                    labels.CLASS_COLUMN: cls,
                    labels.SUBGROUP_COLUMN: CLASS_SUBGROUP[cls] if with_subgroup else None,
                    kd.KLD_COLUMN: float(level + rng.normal(0.0, 0.03)),
                    "n_support_steps": 200,
                }
                if with_epoch:
                    row["epoch"] = -float(hours_before) * 3600.0
                rows.append(row)
                index += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def test_binning_places_a_segment_in_its_thirty_minute_window() -> None:
    r"""``epoch`` seconds before delivery map to a $0.5$ h window and its centre."""
    frame = pd.DataFrame({
        kd.KLD_COLUMN: [0.5, 0.5, 0.5],
        # 0.25 h, 0.75 h and 1.10 h before delivery -> windows 0, 1, 2.
        "epoch": [-0.25 * 3600.0, -0.75 * 3600.0, -1.10 * 3600.0],
        labels.CLASS_COLUMN: ["healthy", "healthy", "healthy"],
        labels.SUBGROUP_COLUMN: [None, None, None],
    })
    binned = kd.bin_samples(frame, width=0.5)
    assert list(binned["bin"]) == [0, 1, 2]
    assert list(binned["bin_center_h"]) == pytest.approx([0.25, 0.75, 1.25])
    assert list(binned["time_to_delivery_h"]) == pytest.approx([0.25, 0.75, 1.10])


def test_binning_drops_nonfinite_rows() -> None:
    """A sample with no finite KL or no finite epoch has no place on the trajectory."""
    frame = pd.DataFrame({
        kd.KLD_COLUMN: [0.5, float("nan"), 0.7],
        "epoch": [-3600.0, -3600.0, float("nan")],
        labels.CLASS_COLUMN: ["healthy", "healthy", "healthy"],
        labels.SUBGROUP_COLUMN: [None, None, None],
    })
    binned = kd.bin_samples(frame, width=0.5)
    assert len(binned) == 1


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def test_trajectory_quartiles_match_a_direct_pandas_reduction() -> None:
    """The long-form (group, window) summary must equal a direct groupby on the same rows."""
    binned = kd.bin_samples(_frame(), width=0.5)
    trajectory = kd.build_trajectory(binned, labels.CLASS_COLUMN, width=0.5)

    for row in trajectory.itertuples():
        cell = binned[
            (binned[labels.CLASS_COLUMN] == row.group) & (binned["bin"] == row.bin)
        ][kd.KLD_COLUMN].to_numpy(dtype=np.float64)
        assert row.n == cell.size
        assert row.median == pytest.approx(float(np.percentile(cell, 50)))
        assert row.q25 == pytest.approx(float(np.percentile(cell, 25)))
        assert row.q75 == pytest.approx(float(np.percentile(cell, 75)))


def test_trajectory_excludes_rows_with_no_group() -> None:
    """A segment with no class belongs to no trajectory and must not become a phantom cohort."""
    frame = _frame(classes=("healthy", "acidosis"))
    # Blank out one class label; those rows must vanish from the class trajectory.
    frame.loc[frame[labels.CLASS_COLUMN] == "acidosis", labels.CLASS_COLUMN] = None
    binned = kd.bin_samples(frame, width=0.5)
    trajectory = kd.build_trajectory(binned, labels.CLASS_COLUMN, width=0.5)
    assert set(trajectory["group"]) == {"healthy"}


# ---------------------------------------------------------------------------
# Per-window significance
# ---------------------------------------------------------------------------
def test_per_window_kruskal_matches_a_direct_scipy_call() -> None:
    """Recomputed rather than recorded: a recorded constant survives a group being dropped."""
    from scipy import stats as sp

    binned = kd.bin_samples(_frame(), width=0.5)
    record = kd.analyse_class_trajectories(binned, width=0.5)
    assert record["tested"] is True

    for row in record["per_bin"]:
        cell = binned[binned["bin"] == row["bin"]]
        groups = [
            cell.loc[cell[labels.CLASS_COLUMN] == name, kd.KLD_COLUMN].to_numpy(dtype=np.float64)
            for name in sorted(cell[labels.CLASS_COLUMN].dropna().unique())
        ]
        statistic, p_value = sp.kruskal(*groups)
        assert row["statistic"] == pytest.approx(float(statistic))
        assert row["p_value"] == pytest.approx(float(p_value))


def test_a_genuine_separation_is_found_and_a_flat_one_is_not() -> None:
    """The non-vacuity check: the procedure must find a real difference and invent no absent one."""
    separated = kd.analyse_class_trajectories(kd.bin_samples(_frame(separated=True)))
    flat = kd.analyse_class_trajectories(kd.bin_samples(_frame(separated=False, seed=7)))
    assert separated["n_significant_bins"] >= 1
    assert flat["n_significant_bins"] == 0
    assert flat["pairwise"] == {}


def test_holm_is_applied_across_the_windows() -> None:
    """The family is the set of per-window omnibus tests; the adjusted p is never looser than raw."""
    record = kd.analyse_class_trajectories(kd.bin_samples(_frame()))
    tested = [row for row in record["per_bin"] if np.isfinite(row["p_value"])]
    assert record["n_bins_tested"] == len(tested)
    for row in tested:
        assert row["p_holm"] >= row["p_value"] - 1e-12
        assert row["n_windows_in_family"] == len(tested)


def test_pairwise_runs_only_for_the_windows_that_survived_holm() -> None:
    """The ordering is the multiple-comparison argument, not an implementation detail."""
    record = kd.analyse_class_trajectories(kd.bin_samples(_frame()))
    significant = {str(row["bin"]) for row in record["per_bin"] if row["significant"]}
    assert set(record["pairwise"]) == significant
    for comparisons in record["pairwise"].values():
        assert len(comparisons) == 3  # C(3, 2) classes


def test_a_class_below_the_minimum_size_is_excluded_from_a_window_and_recorded() -> None:
    """A rank test on two values has no power; its p describes the group size."""
    frame = _frame(hours=(0.25,))  # a single window
    # Leave 'hie' with two samples in that window.
    keep = frame[frame[labels.CLASS_COLUMN] != "hie"]
    trimmed = pd.concat([keep, frame[frame[labels.CLASS_COLUMN] == "hie"].head(2)])
    record = kd.analyse_class_trajectories(kd.bin_samples(trimmed), width=0.5)
    window = record["per_bin"][0]
    assert window["groups_excluded_as_too_small"] == {"hie": 2}
    assert window["n_groups"] == 2


def test_the_pooled_context_test_is_computed_and_flagged() -> None:
    """The pooled reading ignores time and must never be mistaken for the trajectory answer."""
    record = kd.analyse_class_trajectories(kd.bin_samples(_frame()))
    assert record["pooled"]["confounded_by_time"] is True
    assert "artifact" in record["pooled"]["note"]
    assert np.isfinite(record["pooled"]["p_value"])


# ---------------------------------------------------------------------------
# emit_analysis: skips, files and summary
# ---------------------------------------------------------------------------
def test_below_two_classes_the_test_is_skipped_but_the_trajectory_is_still_drawn(tmp_path) -> None:
    """A single-class split still has a trajectory; only the between-class test cannot run."""
    summary = kd.emit_analysis(_frame(classes=("healthy",)), tmp_path)
    assert summary["skipped"] is False
    assert summary["significance"]["tested"] is False
    assert "fewer than two" in summary["significance"]["reason"]
    assert (tmp_path / kd.ANALYSIS_DIRNAME / "trajectory.pdf").stat().st_size > 0


def test_a_missing_epoch_column_is_a_clean_skip(tmp_path) -> None:
    """A split without the epoch field has no time axis; skip, and leave no directory."""
    summary = kd.emit_analysis(_frame(with_epoch=False), tmp_path)
    assert summary["skipped"] is True
    assert "epoch" in summary["reason"]
    assert not (tmp_path / kd.ANALYSIS_DIRNAME).exists()


def test_no_labels_at_all_is_a_clean_skip(tmp_path) -> None:
    """The label-less tiny smoke shard: nothing to plot, nothing to test, no directory."""
    frame = _frame(with_subgroup=False)
    frame[labels.CLASS_COLUMN] = None
    summary = kd.emit_analysis(frame, tmp_path)
    assert summary["skipped"] is True
    assert not (tmp_path / kd.ANALYSIS_DIRNAME).exists()


def test_the_expected_files_are_written(tmp_path) -> None:
    summary = kd.emit_analysis(_frame(), tmp_path)
    directory = tmp_path / kd.ANALYSIS_DIRNAME
    for name in (
        "per_sample.csv", "trajectory_by_class.csv", "trajectory_by_subgroup.csv",
        "significance.csv", "pairwise.csv", f"{kd.ANALYSIS_DIRNAME}.json",
    ):
        assert (directory / name).is_file(), f"{name} was not written"
    assert (directory / "trajectory.pdf").stat().st_size > 0
    assert (directory / "significance.pdf").stat().st_size > 0

    per_sample = pd.read_csv(directory / "per_sample.csv")
    for column in ("kld_mean", "epoch", "time_to_delivery_h", "bin", "bin_center_h"):
        assert column in per_sample.columns
    assert summary["n_dropped_nonfinite"] == 0


def test_the_summary_is_json_safe(tmp_path) -> None:
    """It lands in ``summary.json``, written with ``allow_nan=False``."""
    summary = kd.emit_analysis(_frame(), tmp_path)
    json.dumps(json_safe(summary), allow_nan=False)


def test_the_written_record_is_json_safe(tmp_path) -> None:
    kd.emit_analysis(_frame(), tmp_path)
    blob = json.loads(
        (tmp_path / kd.ANALYSIS_DIRNAME / f"{kd.ANALYSIS_DIRNAME}.json").read_text(encoding="utf-8")
    )
    assert blob["tested"] is True
    assert "Kruskal-Wallis" in blob["method"] and "Holm" in blob["method"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_it_is_registered_and_runs_after_latent() -> None:
    """On by default via ``run``, and ordered after ``latent`` whose CSV it mirrors in definition."""
    assert "kld_time_to_delivery" in run_module.ANALYSES
    order = list(run_module.ANALYSES)
    assert order.index("kld_time_to_delivery") == order.index("latent") + 1
    assert (
        run_module.ANALYSIS_FUNCTIONS["kld_time_to_delivery"].__name__
        == "run_kld_time_to_delivery_analysis"
    )
