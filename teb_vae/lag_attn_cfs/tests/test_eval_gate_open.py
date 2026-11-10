r"""Known answers for the ``gate_open`` analysis: the populations, their readouts and their lags.

The analysis must keep the gate-on and the gate-off anchors apart and must make their
contributions add up to the all-anchor mean. A test that checked the all-anchor mean alone would
pass on an analysis that put every anchor in one population.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.analyses import gate_open

#: The gate of the eight anchors of the one synthetic segment: 3 off, 3 partly open, 2 fully open.
GATE = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.1, 0.5])


def _context(with_maps: bool = True) -> types.SimpleNamespace:
    """One recording, one segment, eight anchors and three lags.

    The KL of anchor $i$ is $i$ nats. A gate-off anchor splits it evenly over lags $0$ and $1$, and
    a gate-on anchor puts all of it on lag $2$, so the two populations have different lag profiles.
    """
    kl = np.arange(8, dtype=float)
    per_anchor = pd.DataFrame({
        "guid": "G0", "epoch": -3600.0, "anchor": np.arange(10, 18), "event_gate": GATE,
        "kld_per_t": kl, "mc_pred_gap": 2.0 * kl, "argmax_lag": np.where(GATE >= 0.2, 2, 0),
        "seconds_since_contraction": 10.0 * np.arange(8),
    })
    on = (GATE >= 0.2)[:, None]
    kl_map = np.where(on, kl[:, None] * np.array([0.0, 0.0, 1.0]), kl[:, None] * np.array([0.5, 0.5, 0.0]))
    attention = np.tile(np.array([0.25, 0.25, 0.5]), (8, 1))
    per_sample = pd.DataFrame({"guid": ["G0"], "clinical_class": ["healthy"]})
    collection = types.SimpleNamespace(
        per_anchor=per_anchor, per_sample=per_sample,
        anchor_vectors={"kl_lag_map": kl_map, "attention_lag_map": attention} if with_maps else {},
        results={"lag": {"n_lags": 3, "delay_steps": 0}},
    )
    return types.SimpleNamespace(collection=collection)


def test_the_populations_partition_the_anchors() -> None:
    """Gate on and gate off cover every anchor once, and fully open lies inside gate on."""
    masks = gate_open.population_masks(np.append(GATE, np.nan))
    assert masks["all"].sum() == 8
    assert not (masks["gate_on"] & masks["gate_off"]).any()
    assert (masks["gate_on"] | masks["gate_off"]).sum() == 8
    assert (masks["fully_open"] & ~masks["gate_on"]).sum() == 0
    assert masks["gate_on"].sum() == 5 and masks["fully_open"].sum() == 2


def test_group_sums_adds_the_rows_of_each_group_and_skips_a_negative_code() -> None:
    """Rows 0 and 2 belong to group 1, row 1 to group 0, and row 3 to no group."""
    values = np.arange(8, dtype=float).reshape(4, 2)
    sums = gate_open.group_sums(values, np.array([1, 0, 1, -1]), 2)
    assert sums.tolist() == [[2.0, 3.0], [4.0, 6.0]]


def test_the_analysis_reports_each_population_and_its_lag_structure(tmp_path) -> None:
    """The shares, the means, the additive contributions, the centroids and the runs."""
    result = gate_open.run_gate_open_analysis(
        _context(), eval_config={"bootstrap_resamples": 20}, output_dir=tmp_path
    )
    headline = result["headline"]
    assert headline["gate_on_frac"] == pytest.approx(5 / 8)
    assert headline["fully_open_frac"] == pytest.approx(2 / 8)
    assert headline["gate_on_kl_nats"] == pytest.approx(4.2)
    assert headline["gate_off_kl_nats"] == pytest.approx(7 / 3)
    assert headline["gate_on_gain_mc_nats"] == pytest.approx(8.4)
    # 21 of the run's 28 nats sit on the gate-on anchors, over 8 scored anchors.
    assert headline["gate_on_kl_share_of_run"] == pytest.approx(21 / 28)
    assert headline["gate_on_kl_contribution_nats"] == pytest.approx(21 / 8)
    assert headline["gate_off_gain_mc_contribution_nats"] == pytest.approx(14 / 8)
    # Every gate-on nat sits on lag 2, which is 8 s on a 4 s step with no input delay.
    assert headline["gate_on_kl_centroid_s"] == pytest.approx(8.0)
    assert headline["gate_on_kl_lag0_share"] == pytest.approx(0.0)

    summary = pd.read_csv(tmp_path / gate_open.ANALYSIS_DIRNAME / gate_open.SUMMARY_FILENAME)
    pooled = summary[summary["cohort"] == "all"].set_index("population")
    assert pooled.loc["gate_off", "kl_centroid_s"] == pytest.approx(2.0)
    assert pooled.loc["gate_off", "kl_lag0_share"] == pytest.approx(0.5)
    assert (
        pooled.loc["gate_on", "kl_contribution_nats"] + pooled.loc["gate_off", "kl_contribution_nats"]
        == pytest.approx(pooled.loc["all", "kl_nats"])
    )

    # Gate on at anchors 2 to 5 and 7: one run of four steps and one of one step, which ends at
    # the segment edge. Gate off at anchors 0, 1 and 6: the first run starts at the segment edge.
    assert result["runs"]["gate_on"]["n"] == 2
    assert result["runs"]["gate_on"]["median_s"] == pytest.approx(2.5 * gate_open.SECONDS_PER_STEP)
    assert result["runs"]["gate_on"]["n_touching_segment_edge"] == 1
    assert result["runs"]["gate_off"]["n_touching_segment_edge"] == 1
    for name in result["files"]:
        assert (tmp_path / gate_open.ANALYSIS_DIRNAME / name).is_file(), name


def test_the_lag_profile_of_a_population_sums_to_its_kl(tmp_path) -> None:
    """The per-lag attribution of a population sums over lags to that population's mean KL."""
    gate_open.run_gate_open_analysis(
        _context(), eval_config={"bootstrap_resamples": 20}, output_dir=tmp_path
    )
    directory = tmp_path / gate_open.ANALYSIS_DIRNAME
    profile = pd.read_csv(directory / gate_open.LAG_PROFILE_FILENAME)
    summary = pd.read_csv(directory / gate_open.SUMMARY_FILENAME)
    for population in ("all", "gate_on", "fully_open", "gate_off"):
        rows = profile[(profile["population"] == population) & (profile["cohort"] == "all")]
        expected = summary[(summary["population"] == population) & (summary["cohort"] == "all")]
        assert rows["kl_nats"].sum() == pytest.approx(float(expected["kl_nats"].iloc[0])), population


def test_a_clock_bin_with_no_anchors_keeps_its_row_in_the_figure_field() -> None:
    """Bins at 0, 1 and 3 give a four-row field whose third row is empty, so no row is shifted."""
    table = pd.DataFrame({
        "population": "gate_on", "bin_center": np.repeat([0.0, 1.0, 3.0], 2),
        "lag_s": np.tile([0.0, 4.0], 3), "kl_share": [0.9, 0.1, 0.8, 0.2, 0.6, 0.4],
    })
    field, centres, lags = gate_open._lag_field(table, "gate_on", "kl_share")
    assert centres.tolist() == [0.0, 1.0, 2.0, 3.0] and lags.tolist() == [0.0, 4.0]
    assert np.isnan(field[2]).all()
    assert field[3].tolist() == [0.6, 0.4]


def test_the_gate_table_reads_one_population_and_one_cohort() -> None:
    """The gate bins are the selection, so the table has one row per bin and no cohort column."""
    context = _context()
    data = gate_open.AnchorData(
        context.collection.per_anchor, GATE, context.collection.anchor_vectors,
        np.array([0.0, 4.0, 8.0]), pd.DataFrame({"clinical_class": ["healthy"]}, index=["G0"]),
    )
    scalars, lags = gate_open.gate_tables(data)
    assert "cohort" not in scalars.columns and "population" not in lags.columns
    # Gates 0.1, 0.5 and 0.9 fall in three bins, holding 3, 3 and 2 anchors.
    assert scalars["n_anchors"].tolist() == [3, 3, 2]
    assert scalars["share_of_anchors"].sum() == pytest.approx(1.0)


def test_the_class_trajectory_counts_a_recording_only_with_enough_anchors(tmp_path) -> None:
    """Twelve anchors in one window pass the floor of ten for the pooled population only."""
    context = _context()
    # Twelve anchors of one recording in one window: twelve in ``all``, and fewer than ten in
    # every gate population, so only ``all`` yields a trajectory value.
    frame = pd.concat([context.collection.per_anchor, context.collection.per_anchor.iloc[:4].assign(
        anchor=np.arange(18, 22))], ignore_index=True)
    context.collection.per_anchor = frame
    for name, array in list(context.collection.anchor_vectors.items()):
        context.collection.anchor_vectors[name] = np.vstack([array, array[:4]])
    gate_open.run_gate_open_analysis(context, eval_config={"bootstrap_resamples": 20}, output_dir=tmp_path)
    trajectory = pd.read_csv(tmp_path / gate_open.ANALYSIS_DIRNAME / gate_open.TRAJECTORY_FILENAME)
    assert set(trajectory.loc[trajectory["metric"] == "kl_nats", "population"]) == {"all"}
    pooled = trajectory[(trajectory["metric"] == "kl_nats") & (trajectory["cohort"] == "all")]
    # The mean KL of anchors 0 to 7 and again 0 to 3: (28 + 6) / 12.
    assert pooled["median"].iloc[0] == pytest.approx(34 / 12)
    assert pooled["n_recordings"].iloc[0] == 1


def test_a_run_without_a_gate_or_without_the_lag_maps_records_a_skip(tmp_path) -> None:
    """Nothing to split, or no lag map to read, is a recorded skip and not a table of zeros."""
    context = _context()
    context.collection.per_anchor = context.collection.per_anchor.drop(columns=["event_gate"])
    no_gate = gate_open.run_gate_open_analysis(context, eval_config={}, output_dir=tmp_path / "a")
    assert no_gate["skipped"] is True and no_gate["n_samples"] is None
    no_maps = gate_open.run_gate_open_analysis(
        _context(with_maps=False), eval_config={}, output_dir=tmp_path / "b"
    )
    assert no_maps["skipped"] is True and "kl_lag_map" in no_maps["reason"]
