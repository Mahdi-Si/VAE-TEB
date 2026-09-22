r"""The lag-structure layer and the four analyses built on it, against a hand-written collection.

Every analysis here reads the two sidecars the collection pass writes -- the per-segment
profiles and the per-anchor maps -- plus the per-sample and per-anchor tables, and nothing else.
So a collection assembled by hand at a small geometry exercises the whole of each one with no
fit, no shard and no checkpoint, which is what lets the arithmetic be pinned: a centroid computed
of a known profile, a band mass summed over known bins, a selection cut at a known quantile.

The failures worth naming are the quiet ones. A profile reduced on the wrong axis still yields a
centroid; a band mass summed over the wrong bins is still a number; a selection cut per class
rather than pooled still fills every table. Each is checked against a value computed a second way
here, and every analysis is also run against a collection that carries no sidecar, which must
record a skip rather than raise.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

from teb_vae.lag_slot_transformer_cfs.eval import lag_structure  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval.analyses import (  # noqa: E402
    band_clocks,
    high_kl_anchors,
    proposal_clocks,
    proposal_profile,
)
from teb_vae.lag_slot_transformer_cfs.eval.figures import configure_figure_style  # noqa: E402

#: The hand-written geometry: candidate lags, segments, anchors per segment, recordings.
N_LAGS = 6
N_SEGMENTS = 12
ANCHORS_PER_SEGMENT = 5
N_RECORDINGS = 6

#: The declared bands of the hand-written run, in stored steps.
BANDS = {"near": [0, 2], "far": [3, N_LAGS - 1]}

#: Seconds per stored step and the input delay the hand-written lag axis carries.
STEP_SECONDS = 4.0
DELAY_STEPS = 0

#: One recording per class pair, so every class has two recordings and a paired test can run.
CLASSES = ("hie", "acidosis", "healthy")


def _results() -> Dict[str, Any]:
    """The results block the analyses read: the lag axis and the declared bands."""
    return {
        "lag_readouts": {
            "lag_axis": {
                "n_lags": N_LAGS, "seconds_per_step": STEP_SECONDS, "delay_steps": DELAY_STEPS,
            },
            "band_edges": {"none": [0, -1], **BANDS, "all": [0, N_LAGS - 1]},
            "band_suppression": {"none": {}, "near": {}, "far": {}, "all": {}},
        },
    }


def _per_sample() -> pd.DataFrame:
    """One row per segment with every identity column and the margins the clock page bins."""
    rows = []
    for index in range(N_SEGMENTS):
        recording = index % N_RECORDINGS
        rows.append({
            "sample_index": index,
            "guid": f"R{recording}",
            # Two segments per recording, an hour apart, spread over twelve hours before delivery.
            "epoch": -3600.0 * (2.0 + recording * 1.5 + (index // N_RECORDINGS)),
            "clinical_class": CLASSES[recording % len(CLASSES)],
            "subgroup": f"sub{recording % 2}",
            "time_from_labor_onset": 100.0,
            "second_stage_onset": -1800.0 * (1.0 + recording) + 600.0 * (index // N_RECORDINGS),
            "margin_suppress_none": 0.0,
            "margin_suppress_near": 0.1 + 0.01 * index,
            "margin_suppress_far": 0.2 - 0.01 * index,
            "margin_suppress_all": 0.3,
            "margin_silence": 0.3,
            "margin_replace_zeros": 0.05,
            "margin_replace_constant": 0.04,
            "margin_permute": 0.03,
        })
    return pd.DataFrame(rows)


def _profiles(generator: np.random.Generator) -> Dict[str, np.ndarray]:
    """Per-segment profiles: a proposal norm peaked at lag 3, and a signed divergence drop."""
    lags = np.arange(N_LAGS, dtype=np.float64)
    base = np.exp(-0.5 * ((lags - 3.0) / 1.0) ** 2)
    proposal = base[None, :] * (1.0 + 0.1 * generator.standard_normal((N_SEGMENTS, N_LAGS)))
    drop = proposal * 0.2
    # One negative bin, so the rectification path is exercised rather than skipped.
    drop[0, 0] = -0.05
    return {"proposal_lag_profile": np.abs(proposal), "divergence_drop_lag_profile": drop}


def _per_anchor(generator: np.random.Generator) -> pd.DataFrame:
    """One row per anchor: the divergence, the gain, the argmax and the contraction age."""
    rows = []
    for index in range(N_SEGMENTS):
        for anchor in range(ANCHORS_PER_SEGMENT):
            kl = float(generator.uniform(0.05, 1.0))
            rows.append({
                "sample_index": index, "guid": f"R{index % N_RECORDINGS}", "epoch": -3600.0,
                "anchor": 10 + anchor, "kld_per_t": kl,
                # The gain grows with the divergence, so the usefulness reading is positive.
                "mean_pred_gap": kl * 0.5 + float(generator.normal(0.0, 0.05)),
                "proposal_argmax_lag": 3,
                "seconds_since_contraction": float(generator.uniform(0.0, 400.0)),
            })
    return pd.DataFrame(rows)


def _anchor_map(per_anchor: pd.DataFrame, generator: np.random.Generator) -> np.ndarray:
    """A per-anchor proposal map peaked at lag 3, larger where the divergence is larger."""
    lags = np.arange(N_LAGS, dtype=np.float64)
    base = np.exp(-0.5 * ((lags - 3.0) / 1.0) ** 2)
    scale = per_anchor["kld_per_t"].to_numpy()[:, None]
    return base[None, :] * (0.5 + scale) * (1.0 + 0.05 * generator.standard_normal((len(per_anchor), N_LAGS)))


def collection(*, with_sidecars: bool = True) -> Any:
    """A hand-written collection with, or without, the lag sidecars.

    Args:
        with_sidecars: Whether the per-sample profiles and the per-anchor map are present.

    Returns:
        An object with the attributes the analyses read off a collection.
    """
    generator = np.random.default_rng(7)
    per_sample = _per_sample()
    per_anchor = _per_anchor(generator)
    vectors = _profiles(generator) if with_sidecars else {}
    anchor_vectors = (
        {"proposal_lag_map": _anchor_map(per_anchor, generator)} if with_sidecars else {}
    )
    return SimpleNamespace(
        per_sample=per_sample, per_anchor=per_anchor, vectors=vectors,
        anchor_vectors=anchor_vectors, results=_results(), record={},
    )


def context(**kwargs: Any) -> Any:
    """The analysis context around :func:`collection`."""
    return SimpleNamespace(collection=collection(**kwargs), config={}, task=None, loader=None)


EVAL_CONFIG = {"seed": 0, "bootstrap_resamples": 50, "event_lag_window_s": 120.0}


@pytest.fixture(autouse=True)
def _style() -> None:
    """Apply the publication style once, as the runner does."""
    configure_figure_style()


# =============================================================================
# The layer
# =============================================================================
def test_the_segment_table_carries_both_profiles_shape_and_band_masses() -> None:
    """One row per segment, every statistic of both profiles, and the band masses summed over
    exactly the declared bins."""
    coll = collection()
    table, record = lag_structure.segment_table(coll, coll.results)
    proposal, drop = lag_structure.PROFILE_SOURCES

    assert len(table) == N_SEGMENTS
    assert record["sources"]["proposal"]["present"] and record["sources"]["drop"]["present"]
    assert record["sources"]["proposal"]["n_usable"] == N_SEGMENTS
    matrix = coll.vectors["proposal_lag_profile"]
    near = lag_structure.band_column("near", proposal)
    assert np.allclose(table[near].to_numpy(), matrix[:, 0:3].sum(axis=1))
    # The centroid of a profile peaked at lag 3 sits near 3 stored steps, in seconds.
    centroid = table[lag_structure.statistic_column("centroid", proposal)].to_numpy()
    assert np.all(np.abs(centroid - 3.0 * STEP_SECONDS) < STEP_SECONDS)
    # The signed profile's discarded mass is reported per row, and only the row with a negative
    # bin carries any.
    negative = table[f"negative_{drop.key}"].to_numpy()
    assert negative[0] == pytest.approx(-0.05)
    assert np.all(negative[1:] == 0.0)
    # The band mass of the signed profile is summed signed, not rectified.
    assert table[lag_structure.band_column("near", drop)].iloc[0] == pytest.approx(
        float(coll.vectors["divergence_drop_lag_profile"][0, 0:3].sum())
    )


def test_a_collection_without_the_sidecar_yields_an_empty_table_with_a_reason() -> None:
    """Absent rather than raised, so an older directory records a skip."""
    coll = collection(with_sidecars=False)
    table, record = lag_structure.segment_table(coll, coll.results)
    assert table.empty
    assert "sidecar" in record["reason"]
    assert lag_structure.present_sources(record) == []


def test_pooled_profiles_weight_each_recording_once() -> None:
    """A recording contributing two segments enters the pooled mean as one profile."""
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pooled = lag_structure.pooled_profile(matrix, ["A", "A", "B"])
    assert pooled["n_recordings"] == 2
    assert np.allclose(pooled["mean"], [0.5, 0.5])


def test_the_clocks_bin_the_table_and_the_second_clock_admits_eligible_recordings_only() -> None:
    """The delivery clock places every segment; the second-stage clock keeps only recordings
    with an onset, which here is all of them."""
    coll = collection()
    table, _record = lag_structure.segment_table(coll, coll.results)
    for clock in lag_structure.CLOCKS:
        binned, population = lag_structure.clock_rows(table, clock)
        assert len(binned) == N_SEGMENTS, clock.name
        assert population["n_recordings"] == N_RECORDINGS
        assert clock.bin_column in binned.columns


def test_windowed_tests_record_untestable_windows_rather_than_raising() -> None:
    """Two recordings per class is below the floor, so every window is recorded untestable."""
    coll = collection()
    table, _record = lag_structure.segment_table(coll, coll.results)
    clock = lag_structure.CLOCKS[0]
    binned, _population = lag_structure.clock_rows(table, clock)
    column = lag_structure.statistic_column("centroid", lag_structure.PROFILE_SOURCES[0])
    frames = lag_structure.per_recording_by_axis(binned, [column], clock)
    record = lag_structure.windowed_tests(frames["clinical_class"], column, clock)
    assert record["tested"] is True
    assert record["n_windows"] > 0
    assert record["n_significant_windows"] == 0
    assert all(np.isnan(window["p_value"]) for window in record["per_window"])


# =============================================================================
# The analyses
# =============================================================================
def _assert_protocol(block: Dict[str, Any], directory: Path) -> None:
    """Every analysis returns the protocol's keys and every file it names exists."""
    assert {"n_samples", "composition", "plan"} <= set(block)
    for name in block["files"]:
        assert (directory / name).is_file(), name


def test_proposal_profile_writes_its_tables_figures_and_grouped_frame(tmp_path: Path) -> None:
    """The four tables, the two figures, the guarded peaks and the grouped-frame declaration."""
    block = proposal_profile.run_proposal_profile_analysis(
        context(), eval_config=EVAL_CONFIG, output_dir=tmp_path
    )
    directory = tmp_path / proposal_profile.ANALYSIS_DIRNAME
    _assert_protocol(block, directory)
    assert block["n_samples"] == N_SEGMENTS
    peaks = {row["profile"]: row for row in block["peaks"]}
    assert peaks["proposal"]["argmax_lag_step"] == 3
    assert peaks["proposal"]["degenerate"] is False
    grouped = block["grouped_frames"][0]
    assert (tmp_path / grouped["path"]).is_file()
    per_recording = pd.read_csv(tmp_path / grouped["path"])
    assert len(per_recording) == N_RECORDINGS
    assert set(grouped["value_columns"]) <= set(per_recording.columns)
    stratified = pd.read_csv(directory / proposal_profile.STRATIFIED_FILENAME)
    assert set(stratified["group_column"]) == {"all", "clinical_class", "subgroup"}
    # Every cohort's share sums to one over the lags.
    shares = stratified.groupby(["group_column", "group", "profile"])["share"].sum()
    assert np.allclose(shares.to_numpy(), 1.0)


def test_proposal_clocks_writes_both_clocks_and_tests_the_centroids(tmp_path: Path) -> None:
    """Three figures per clock, the four tables, and a test record per tested centroid."""
    block = proposal_clocks.run_proposal_clocks_analysis(
        context(), eval_config=EVAL_CONFIG, output_dir=tmp_path
    )
    directory = tmp_path / proposal_clocks.ANALYSIS_DIRNAME
    _assert_protocol(block, directory)
    assert set(block["composition"]) == {clock.name for clock in lag_structure.CLOCKS}
    assert len(block["significance"]) == 2 * len(lag_structure.CLOCKS)
    trajectory = pd.read_csv(directory / proposal_clocks.TRAJECTORY_FILENAME)
    assert set(trajectory["clock"]) == {clock.name for clock in lag_structure.CLOCKS}
    band = lag_structure.band_column("near", lag_structure.PROFILE_SOURCES[0])
    assert band in set(trajectory["metric"])


def test_band_clocks_bins_every_margin_and_is_descriptive(tmp_path: Path) -> None:
    """The bands and the controls, on both clocks, with no test."""
    block = band_clocks.run_band_clocks_analysis(
        context(), eval_config=EVAL_CONFIG, output_dir=tmp_path
    )
    directory = tmp_path / band_clocks.ANALYSIS_DIRNAME
    _assert_protocol(block, directory)
    # The two reference identities are not bands: ``none`` is zero and ``all`` is the gap.
    assert block["bands"] == ["near", "far"]
    assert block["controls"] == ["replace_zeros", "replace_constant", "permute"]
    assert block["descriptive_only"] is True
    trajectory = pd.read_csv(directory / band_clocks.TRAJECTORY_FILENAME)
    assert "margin_suppress_near" in set(trajectory["metric"])
    assert "all" in set(trajectory["group_column"])


def test_high_kl_anchors_selects_pooled_and_reads_the_gain(tmp_path: Path) -> None:
    """One pooled threshold, the four bands, a positive usefulness reading on a gain that grows
    with the divergence, and every table and figure written."""
    block = high_kl_anchors.run_high_kl_anchors_analysis(
        context(), eval_config=EVAL_CONFIG, output_dir=tmp_path
    )
    directory = tmp_path / high_kl_anchors.ANALYSIS_DIRNAME
    _assert_protocol(block, directory)
    n_anchors = N_SEGMENTS * ANCHORS_PER_SEGMENT
    assert block["composition"]["n_anchors"] == n_anchors
    assert block["bands"]["high"] + block["bands"]["rest"] == n_anchors
    assert block["bands"]["top"] <= block["bands"]["high"]
    assert block["proposal_map_present"] is True
    # The threshold is the pooled quantile, recomputed here.
    coll = collection()
    expected = float(np.quantile(coll.per_anchor["kld_per_t"], high_kl_anchors.HIGH_QUANTILE))
    assert block["thresholds"]["high_nats"] == pytest.approx(expected)
    usefulness = block["usefulness"]
    assert usefulness["high_minus_rest_mean_interval"]["point"] > 0.0
    assert usefulness["overlap"]["n_high"] == block["bands"]["high"]
    recordings = pd.read_csv(directory / high_kl_anchors.RECORDINGS_FILENAME)
    assert len(recordings) == N_RECORDINGS
    assert 3 in block["hot_lags"]["lag_steps"]


@pytest.mark.parametrize(
    "run",
    [
        proposal_profile.run_proposal_profile_analysis,
        proposal_clocks.run_proposal_clocks_analysis,
        high_kl_anchors.run_high_kl_anchors_analysis,
    ],
)
def test_the_sidecar_analyses_record_a_skip_without_the_sidecar(run, tmp_path: Path) -> None:
    """A directory collected before the sidecars existed records a skip rather than raising."""
    block = run(context(with_sidecars=False), eval_config=EVAL_CONFIG, output_dir=tmp_path)
    if run is high_kl_anchors.run_high_kl_anchors_analysis:
        # The selection needs the per-anchor table only; without the map it runs with the
        # profile panels marked absent.
        assert block["proposal_map_present"] is False
        assert block["n_samples"] == N_SEGMENTS
    else:
        assert block["skipped"] is True
        assert block["n_samples"] is None
        assert block["files"] == []
