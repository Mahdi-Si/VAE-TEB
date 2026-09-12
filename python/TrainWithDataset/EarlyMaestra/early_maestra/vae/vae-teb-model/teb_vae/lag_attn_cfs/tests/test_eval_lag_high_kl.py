r"""The high-KL anchor selection and the lag structure read on it.

``lag_high_kl`` is the one analysis that reads the per-anchor vector sidecar, and each of its
decisions is a way it could be silently wrong:

**The threshold is pooled, and it is one number per run.** A quantile taken per class would let
every cohort select its own anchors and a class contrast on the selection would compare two
selections; a quantile taken per segment would let every segment choose its own. The threshold is
asserted equal to the quantile of the *pooled* per-anchor KL, and the same number reaches the
thresholds table, the record and the headline.

**The bands recompose.** ``high`` and ``rest`` partition the anchors, so their per-segment counts
sum to the segment's anchor count exactly; a band that double-counted or dropped an anchor would
still produce a plausible share.

**The restricted profile is the selected anchors' own.** The fixture puts the high anchors' mass
in one known lag range and the low anchors' at lag $0$, so the high band's centroid must land in
that range and the rest band's below it -- a reduction that averaged every anchor would put both
in the middle.

**What is tested is exactly the two readouts.** The significance table carries the high band's
centroid and share and nothing else, on both clocks; an analysis that quietly widened its family
would multiply the corrections a reader is holding.

**A directory collected before the sidecar existed is a named skip, not a crash**, and so is a
sidecar that is not row-aligned with the table.

Following the fixture rule this suite is bound by, everything below is evidence about schema,
shape, denominators, membership and refusals -- never about the sign, magnitude or significance of
any effect on real data.
"""
from __future__ import annotations

import types
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import lag_high_kl as analysis

#: A tiny lag window and a small anchor set per segment.
N_LAGS = 11
N_SEGMENTS = 12
ANCHORS_PER_SEGMENT = 10

#: The lag range the fixture's HIGH anchors concentrate their attribution in, in lag steps.
HIGH_LAGS = (3, 6)

EVAL_CONFIG: Dict[str, Any] = {"seed": 0, "event_lag_window_s": 120.0}


def _fixture(seed: int = 0):
    """Per-sample and per-anchor tables plus the sidecar, with two anchor populations.

    Every segment holds ``ANCHORS_PER_SEGMENT`` anchors; a known subset of them carries a large
    KL with its attribution concentrated in :data:`HIGH_LAGS`, the rest a small KL concentrated
    at lag $0$. Segments of the more severe class carry more high anchors, so a per-class share
    has something to differ on.
    """
    rng = np.random.default_rng(seed)
    per_sample = pd.DataFrame(
        {
            "sample_index": np.arange(N_SEGMENTS, dtype=np.int64),
            "guid": [f"REC{index // 2:02d}" for index in range(N_SEGMENTS)],
            "epoch": [-(3600.0 * (1 + index % 3)) for index in range(N_SEGMENTS)],
            "clinical_class": ["hie" if index < 6 else "healthy" for index in range(N_SEGMENTS)],
            "subgroup": [
                "hie_cs" if index < 6 else "healthy_bg_no_cs" for index in range(N_SEGMENTS)
            ],
            "second_stage_onset": [-1800.0] * N_SEGMENTS,
            "source_conditioned_kl_raw": np.linspace(0.5, 1.5, N_SEGMENTS),
        }
    )
    rows = []
    kl_map = []
    attn_map = []
    for segment in range(N_SEGMENTS):
        # 36 of 120 anchors are high -- exactly 30% -- so the pooled 0.7 quantile falls in the
        # gap between the two KL populations and the ``high`` band is the constructed set.
        n_high = 4 if segment < 6 else 2
        for anchor in range(ANCHORS_PER_SEGMENT):
            is_high = anchor < n_high
            kl = rng.uniform(2.0, 3.0) if is_high else rng.uniform(0.05, 0.2)
            profile = rng.uniform(0.001, 0.01, N_LAGS)
            if is_high:
                profile[HIGH_LAGS[0] : HIGH_LAGS[1] + 1] += 1.0
            else:
                profile[0] += 1.0
            profile = profile / profile.sum() * kl
            attention = profile / kl
            rows.append(
                {
                    "guid": per_sample.loc[segment, "guid"],
                    "epoch": per_sample.loc[segment, "epoch"],
                    "anchor": 100 + anchor,
                    "sample_index": segment,
                    "kld_per_t": kl,
                    # The forecast gain the source bought at the anchor: positive where the KL is
                    # high, around zero elsewhere, so usefulness and coupling coincide here.
                    "mc_pred_gap": rng.uniform(1.0, 2.0) if is_high else rng.uniform(-0.2, 0.2),
                    "argmax_lag": int(np.argmax(profile)),
                    # Half the anchors sit within the contraction window, half outside it.
                    "seconds_since_contraction": 30.0 if anchor % 2 == 0 else 600.0,
                }
            )
            kl_map.append(profile)
            attn_map.append(attention)
    per_anchor = pd.DataFrame(rows)
    vectors = {
        "kl_lag_map": np.asarray(kl_map, dtype=np.float16),
        "attention_lag_map": np.asarray(attn_map, dtype=np.float16),
    }
    return per_sample, per_anchor, vectors


def _context(
    *,
    anchor_vectors: Optional[Dict[str, np.ndarray]] = None,
    lag: Optional[Dict[str, Any]] = None,
    per_anchor: Optional[pd.DataFrame] = None,
) -> AnalysisContext:
    """An analysis context with no task and no loader, as an offline re-run has."""
    per_sample, anchors, vectors = _fixture()
    collection = types.SimpleNamespace(
        per_sample=per_sample,
        per_anchor=anchors if per_anchor is None else per_anchor,
        record={},
        retained={},
        vectors={},
        anchor_vectors=vectors if anchor_vectors is None else anchor_vectors,
        results={"lag": {"n_lags": N_LAGS, "delay_steps": 0} if lag is None else lag},
    )
    return AnalysisContext(collection=collection, config={})


def _run(context: AnalysisContext, tmp_path, config: Optional[Dict[str, Any]] = None):
    """Run the analysis and return ``(record, directory)``."""
    record = analysis.run_lag_high_kl_analysis(
        context, eval_config=config or EVAL_CONFIG, output_dir=tmp_path
    )
    return record, tmp_path / analysis.ANALYSIS_DIRNAME


# =================================================================================================
# The selection
# =================================================================================================
def test_the_threshold_is_the_pooled_quantile_and_reaches_every_artifact(tmp_path) -> None:
    """One number per run: the ``0.7`` quantile of every anchor's KL, pooled over both classes."""
    _, per_anchor, _ = _fixture()
    expected = float(np.quantile(per_anchor["kld_per_t"], 0.7))

    record, directory = _run(_context(), tmp_path)

    assert record["thresholds"]["high"]["lo_nats"] == pytest.approx(expected)
    assert record["headline"]["high_kl_threshold_nats"] == pytest.approx(expected)
    thresholds = pd.read_csv(directory / analysis.THRESHOLDS_FILENAME).set_index("band")
    assert thresholds.loc["high", "lo_nats"] == pytest.approx(expected)
    assert thresholds.loc["high", "q_lo"] == pytest.approx(0.7)
    assert set(thresholds.index) == {band.key for band in analysis.ANCHOR_BANDS}
    # Pooled: the population the quantile was taken over is every anchor of every class.
    assert record["population"]["n_anchors"] == N_SEGMENTS * ANCHORS_PER_SEGMENT
    assert record["bands"]["high"]["n_anchors_selected"] == int(
        (per_anchor["kld_per_t"] >= expected).sum()
    )


def test_the_high_and_rest_bands_partition_every_segments_anchors(tmp_path) -> None:
    """``high`` plus ``rest`` is every anchor, segment by segment, and the shares sum to one."""
    _, directory = _run(_context(), tmp_path)

    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)
    delivery = per_recording[
        (per_recording["clock"] == "time_to_delivery")
        & (per_recording["group_column"] == "clinical_class")
    ]
    assert len(delivery)
    total = delivery["high_n_anchors"] + delivery["rest_n_anchors"]
    assert (total == ANCHORS_PER_SEGMENT).all()
    assert (delivery["high_anchor_frac"] + delivery["rest_anchor_frac"]).to_numpy() == pytest.approx(1.0)
    # And the more severe class, which the fixture gave more high anchors, has the larger share.
    by_class = delivery.groupby("group")["high_anchor_frac"].mean()
    assert by_class["hie"] > by_class["healthy"]


def test_the_restricted_centroid_is_the_selected_anchors_own(tmp_path) -> None:
    """Known answer: the high anchors' attribution lives in :data:`HIGH_LAGS`, the rest's at lag 0.

    A reduction that averaged every anchor together would put both centroids between the two.
    """
    _, directory = _run(_context(), tmp_path)

    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)
    delivery = per_recording[per_recording["clock"] == "time_to_delivery"]
    high = delivery["high_lag_centroid_kl_s"].dropna().to_numpy()
    rest = delivery["rest_lag_centroid_kl_s"].dropna().to_numpy()
    assert len(high) and len(rest)
    assert (high >= HIGH_LAGS[0] * SECONDS_PER_STEP).all()
    assert (high <= HIGH_LAGS[1] * SECONDS_PER_STEP).all()
    assert (rest < 1.0 * SECONDS_PER_STEP).all()
    # The attention profile of the same anchors reads the same way.
    high_attn = delivery["high_lag_centroid_attn_s"].dropna().to_numpy()
    assert len(high_attn)
    assert (high_attn >= HIGH_LAGS[0] * SECONDS_PER_STEP).all()


def test_the_hot_lag_set_is_written_lag_by_lag_and_agrees_with_the_record(tmp_path) -> None:
    """The run's durable record of which lags were selected, and by what pooled attribution."""
    record, directory = _run(_context(), tmp_path)

    selection = pd.read_csv(directory / analysis.SELECTION_FILENAME)
    assert len(selection) == N_LAGS
    assert int(selection["hot"].sum()) == record["hot_lags"]["n_lags"]
    assert sorted(selection.loc[selection["hot"], "lag_step"]) == record["hot_lags"]["lag_steps"]
    pooled = selection["pooled_attribution_nats"].to_numpy()
    threshold = float(np.quantile(pooled, analysis.HOT_LAG_QUANTILE))
    assert set(selection.loc[selection["hot"], "lag_step"]) == set(
        np.nonzero(pooled >= threshold)[0]
    )
    # The pooled band profiles beside it, one column per band.
    for band in analysis.ANCHOR_BANDS:
        assert f"pooled_{band.key}_kl_nats" in selection.columns
    # The circularity is on the record, not left for a reader to notice.
    assert "selected FROM the KL" in record["selection_note"]


# =================================================================================================
# What is tested, and what is not
# =================================================================================================
def test_exactly_the_two_readouts_are_tested_on_both_clocks(tmp_path) -> None:
    """Four Holm families on the main table: the high band's centroid and share, on two clocks.

    The histogram's own families live on their own table and in their own block of the record, so
    a consumer of ``significance`` still meets exactly the four it always did.
    """
    record, directory = _run(_context(), tmp_path)

    assert list(record["readouts"]) == ["high_lag_centroid_kl_s", "high_anchor_frac"]
    significance = pd.read_csv(directory / analysis.SIGNIFICANCE_FILENAME)
    assert set(significance["metric_column"]) <= set(analysis.READOUTS)
    assert set(significance["clock"]) == {"time_to_delivery", "second_stage"}
    families = {(r["clock"], r["metric_column"]) for r in record["significance"]}
    assert len(families) == 4
    assert record["plan"]["tested_features"] == 2 + len(analysis.HISTOGRAM_READOUTS)
    assert "no p-value" in record["untested_note"]
    assert "four Holm families" in record["method"]
    assert "six further families" in record["method"]


def test_the_second_clock_is_a_subset_and_the_analysis_declares_itself_capped(tmp_path) -> None:
    """A recording with no onset cannot be placed on the second axis; the analysis says so."""
    record, _ = _run(_context(), tmp_path)

    assert record["plan"]["capped"] is True
    names = {entry["clock"] for entry in record["clocks"]}
    assert names == {"time_to_delivery", "second_stage"}
    second = next(entry for entry in record["clocks"] if entry["clock"] == "second_stage")
    assert "n_recordings_eligible" in second


# =================================================================================================
# The three further readings
# =================================================================================================
def test_the_argmax_by_decile_table_is_a_distribution_over_lags_per_bin(tmp_path) -> None:
    """Every populated (cohort, bin) row set sums to one over the lags."""
    record, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.ARGMAX_FILENAME)
    assert record["argmax_by_quantile"]["n_bins"] == analysis.N_KL_QUANTILE_BINS
    assert set(table["group"]) >= {"all", "hie", "healthy"}
    populated = table[table["n_anchors"] > 0]
    sums = populated.groupby(["group", "kl_bin"])["argmax_share"].sum()
    assert sums.to_numpy() == pytest.approx(1.0)
    # The fixture's high anchors peak inside HIGH_LAGS, so the top bin's mass sits there.
    top = table[(table["group"] == "all") & (table["kl_bin"] == analysis.N_KL_QUANTILE_BINS - 1)]
    inside = top[(top["lag_step"] >= HIGH_LAGS[0]) & (top["lag_step"] <= HIGH_LAGS[1])]
    assert inside["argmax_share"].sum() == pytest.approx(1.0)


def test_the_contraction_enrichment_is_per_recording_and_guarded_by_arm_size(tmp_path) -> None:
    """One row per recording; a row is reportable only with enough anchors in BOTH arms."""
    record, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.CONTRACTION_FILENAME)
    assert len(table) == N_SEGMENTS // 2
    assert set(table.columns) >= {
        "guid", "clinical_class", "n_event_anchors", "n_control_anchors",
        "high_share_event", "high_share_control", "enrichment", "reportable",
    }
    # Two segments per recording, five event anchors each: both arms clear the floor.
    assert table["reportable"].all()
    assert record["contraction_enrichment"]["n_reportable"] == len(table)
    assert record["contraction_enrichment"]["tested"] is False
    assert set(record["contraction_enrichment"]["by_class"]) == {"hie", "healthy"}

    # Below the floor the difference is withheld rather than reported.
    _, per_anchor, vectors = _fixture()
    thinned = per_anchor[per_anchor["anchor"] < 100 + analysis.MIN_ENRICHMENT_ANCHORS - 1].copy()
    keep = thinned.index.to_numpy()
    context = _context(
        per_anchor=thinned.reset_index(drop=True),
        anchor_vectors={name: array[keep] for name, array in vectors.items()},
    )
    _, directory = _run(context, tmp_path / "thinned")
    thin = pd.read_csv(directory / analysis.CONTRACTION_FILENAME)
    assert not thin["reportable"].any()
    assert thin["enrichment"].isna().all()


def test_the_recordings_table_is_the_cross_subgroup_source(tmp_path) -> None:
    """One row per recording, whole recording, carrying the column ``cross_subgroup`` reads."""
    from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup

    _, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.RECORDINGS_FILENAME)
    assert len(table) == N_SEGMENTS // 2
    assert table["guid"].is_unique
    source = next(
        s for s in cross_subgroup.METRIC_SOURCES if s.analysis == analysis.ANALYSIS_DIRNAME
    )
    assert source.filename == analysis.RECORDINGS_FILENAME
    assert source.column in table.columns
    assert table[source.column].notna().all()


def test_the_headline_block_resolves_on_the_fixture(tmp_path) -> None:
    """Every scalar the binding registers is present and finite here, and never NaN."""
    record, _ = _run(_context(), tmp_path)

    headline = record["headline"]
    assert set(headline) == {
        "high_kl_threshold_nats", "high_kl_centroid_kl_s", "high_kl_total_nats",
        "hot_lag_count", "hot_lag_share_kl", "high_kl_pred_gap_nats",
        "high_minus_rest_pred_gap_nats", "high_gain_overlap_share",
    }
    for name, value in headline.items():
        assert value is not None, name
        assert np.isfinite(float(value)), name


# =================================================================================================
# The artifacts, and the refusals
# =================================================================================================
def test_every_table_and_figure_is_written(tmp_path) -> None:
    """Twenty tables and twelve figures, each named in the record's ``files``."""
    record, directory = _run(_context(), tmp_path)

    tables = {
        analysis.THRESHOLDS_FILENAME, analysis.SELECTION_FILENAME, analysis.RECORDINGS_FILENAME,
        analysis.PER_RECORDING_FILENAME, analysis.TRAJECTORY_FILENAME, analysis.PROFILE_FILENAME,
        analysis.SIGNIFICANCE_FILENAME, analysis.PAIRWISE_FILENAME, analysis.ARGMAX_FILENAME,
        analysis.CONTRACTION_FILENAME, analysis.GAIN_BY_QUANTILE_FILENAME,
        analysis.GAIN_BY_ARGMAX_FILENAME, analysis.OCCLUSION_CONSISTENCY_FILENAME,
        analysis.HISTOGRAM_FILENAME, analysis.HISTOGRAM_FEATURES_FILENAME,
        analysis.HISTOGRAM_DISTANCE_FILENAME, analysis.HISTOGRAM_SIGNIFICANCE_FILENAME,
        analysis.HISTOGRAM_PAIRWISE_FILENAME, analysis.HISTOGRAM_DRIFT_FILENAME,
        analysis.HISTOGRAM_DRIFT_SUMMARY_FILENAME,
    }
    figures = {
        f"{analysis.SELECTION_FIGURE}.pdf",
        f"{analysis.USEFULNESS_FIGURE}.pdf",
        *(f"{clock.figure}.pdf" for clock in analysis.CLOCKS),
        *(f"{clock.windows_figure}.pdf" for clock in analysis.CLOCKS),
        *(f"{clock.histogram_figure}.pdf" for clock in analysis.CLOCKS),
        *(f"{clock.features_figure}.pdf" for clock in analysis.CLOCKS),
        *(f"{clock.drift_figure}.pdf" for clock in analysis.CLOCKS),
    }
    assert set(record["files"]) == tables | figures
    for name in tables | figures:
        assert (directory / name).is_file(), name
    assert set(REQUIRED_RESULT_KEYS) <= set(record)
    assert record["n_samples"] == N_SEGMENTS


def test_a_run_without_the_sidecar_records_a_named_skip(tmp_path) -> None:
    """A directory collected before the per-anchor vector sidecar existed is a partial input."""
    record, _ = _run(_context(anchor_vectors={}), tmp_path)

    assert record["skipped"] is True
    assert "sidecar" in record["reason"]
    assert set(REQUIRED_RESULT_KEYS) <= set(record)
    assert record["n_samples"] is None


def test_a_sidecar_misaligned_with_the_table_records_a_named_skip(tmp_path) -> None:
    """The sidecar is aligned by row position alone, so a length mismatch is refused."""
    _, _, vectors = _fixture()
    short = {name: array[:-3] for name, array in vectors.items()}

    record, _ = _run(_context(anchor_vectors=short), tmp_path)

    assert record["skipped"] is True
    assert "aligned" in record["reason"]


def test_a_run_with_no_lag_geometry_records_a_named_skip(tmp_path) -> None:
    """No axis to resolve a profile against is a partial input, and the skip names it."""
    record, _ = _run(_context(lag={}), tmp_path)

    assert record["skipped"] is True
    assert "lag geometry" in record["reason"]


def test_a_missing_attention_map_blanks_its_columns_rather_than_dropping_them(tmp_path) -> None:
    """The table's schema is the same on every run; a source the sidecar lacks is NaN there."""
    _, _, vectors = _fixture()

    record, directory = _run(_context(anchor_vectors={"kl_lag_map": vectors["kl_lag_map"]}), tmp_path)

    assert record.get("skipped") is not True
    assert record["sources"]["attn"]["available"] is False
    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)
    assert "high_lag_centroid_attn_s" in per_recording.columns
    assert per_recording["high_lag_centroid_attn_s"].isna().all()
    assert per_recording["high_lag_centroid_kl_s"].notna().any()


# =================================================================================================
# The usefulness half: is the coupling where the forecast gain is?
# =================================================================================================
def test_the_gain_band_and_the_high_band_name_the_same_anchors_on_this_fixture(tmp_path) -> None:
    """The fixture gives the high-KL anchors the forecast gain, so the two selections coincide and
    the overlap reads 100% against the 30% independence would give. A real run is under no such
    obligation, which is what the number is for."""
    record, _ = _run(_context(), tmp_path)

    overlap = record["usefulness"]["overlap"]
    assert overlap["available"] is True
    assert overlap["share_of_high_in_gain"] == pytest.approx(1.0)
    assert overlap["share_expected_if_independent"] == pytest.approx(0.3)
    assert record["thresholds"]["gain"]["on"] == "gain"
    assert record["thresholds"]["high"]["on"] == "kl"
    assert record["headline"]["high_gain_overlap_share"] == pytest.approx(1.0)


def test_the_high_bands_gain_is_tested_against_the_rests_within_recording(tmp_path) -> None:
    """One paired test over recordings, its own family; here the difference is positive in every
    recording by construction, and the record says which column supplied the gain."""
    record, directory = _run(_context(), tmp_path)

    usefulness = record["usefulness"]
    assert usefulness["tested"] is True
    assert usefulness["gain_column"] == "mc_pred_gap"
    assert usefulness["n_pairs"] == N_SEGMENTS // 2
    assert usefulness["positive_fraction"] == pytest.approx(1.0)
    assert usefulness["mean_difference_nats"] > 0.5
    assert "family of one" in usefulness["family"]
    assert record["headline"]["high_minus_rest_pred_gap_nats"] == pytest.approx(
        usefulness["mean_difference_nats"]
    )
    recordings = pd.read_csv(directory / analysis.RECORDINGS_FILENAME)
    assert (recordings["high_pred_gap_nats"] > recordings["rest_pred_gap_nats"]).all()


def test_the_gain_by_kl_decile_table_rises_with_the_decile_on_this_fixture(tmp_path) -> None:
    """Per class and pooled, one row per decile over recordings; the top deciles are the high
    anchors, whose gain the fixture made positive."""
    _, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.GAIN_BY_QUANTILE_FILENAME)
    pooled = table[table["group"] == "all"].set_index("kl_bin")
    assert len(pooled) == analysis.N_KL_QUANTILE_BINS
    assert pooled.loc[analysis.N_KL_QUANTILE_BINS - 1, "median"] > pooled.loc[0, "median"]
    assert pooled.loc[analysis.N_KL_QUANTILE_BINS - 1, "positive_anchor_share"] == pytest.approx(1.0)
    assert {"hie", "healthy"} <= set(table["group"])


def test_the_gain_by_argmax_table_covers_every_lag_for_every_selection(tmp_path) -> None:
    """Pooled plus the high, rest and gain bands, each on the full lag axis."""
    _, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.GAIN_BY_ARGMAX_FILENAME)
    assert set(table["selection"]) == {"all", "high", "rest", "gain"}
    assert (table.groupby("selection").size() == N_LAGS).all()
    high = table[table["selection"] == "high"]
    assert int(high["n_anchors"].sum()) == 36


def test_the_occlusion_join_is_a_named_skip_without_the_interventional_pass(tmp_path) -> None:
    """Read off disk; a directory whose occlusion pass never ran records why, and the table is
    written empty with its columns rather than not at all."""
    record, directory = _run(
        _context(), tmp_path,
        config={**EVAL_CONFIG, "occlusion_bands": {"near": [0, 5], "far": [6, 10]}},
    )

    consistency = record["usefulness"]["occlusion_consistency"]
    assert consistency["available"] is False
    assert "did not run" in consistency["reason"]
    table = pd.read_csv(directory / analysis.OCCLUSION_CONSISTENCY_FILENAME)
    assert len(table) == 0
    assert "occlusion_delta_nats" in table.columns


def test_the_occlusion_join_lands_per_recording_and_per_band_when_the_pass_ran(tmp_path) -> None:
    """With the interventional table present: one row per (recording, band), both shares and the
    delta, and a descriptive rho per band."""
    guids = [f"REC{index:02d}" for index in range(N_SEGMENTS // 2)]
    occlusion_dir = tmp_path / "occlusion"
    occlusion_dir.mkdir()
    pd.DataFrame(
        {
            "guid": guids,
            "occlusion_delta_near_nats": np.linspace(0.1, 1.0, len(guids)),
            "occlusion_delta_far_nats": np.linspace(1.0, 0.1, len(guids)),
        }
    ).to_csv(occlusion_dir / "occlusion_per_recording.csv", index=False)

    record, directory = _run(
        _context(), tmp_path,
        config={**EVAL_CONFIG, "occlusion_bands": {"near": [0, 5], "far": [6, 10]}},
    )

    consistency = record["usefulness"]["occlusion_consistency"]
    assert consistency["available"] is True
    assert set(consistency["bands"]) == {"near", "far"}
    assert consistency["bands"]["near"]["n_recordings"] == len(guids)
    assert consistency["tested"] is False
    table = pd.read_csv(directory / analysis.OCCLUSION_CONSISTENCY_FILENAME)
    assert len(table) == 2 * len(guids)
    assert table["attribution_share_all"].between(0.0, 1.0).all()
    # The two bands partition the axis, so the shares sum to one per recording.
    sums = table.groupby("guid")["attribution_share_all"].sum()
    assert sums.to_numpy() == pytest.approx(1.0)


def test_a_table_without_a_gain_column_leaves_the_gain_band_empty_and_the_test_unrun(
    tmp_path,
) -> None:
    """A per-anchor table from a pass that scored no forecast gain: the KL half is untouched and
    the usefulness half says why it could not run rather than reporting zeros."""
    _, per_anchor, _ = _fixture()
    record, directory = _run(
        _context(per_anchor=per_anchor.drop(columns=["mc_pred_gap"])), tmp_path
    )

    assert record.get("skipped") is not True
    assert record["usefulness"]["tested"] is False
    assert record["usefulness"]["gain_column"] is None
    assert record["usefulness"]["overlap"]["available"] is False
    assert record["bands"]["gain"]["n_anchors_selected"] == 0
    assert record["headline"]["high_kl_pred_gap_nats"] is None
    assert record["headline"]["high_kl_threshold_nats"] is not None
    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)
    assert per_recording["high_pred_gap_nats"].isna().all()
    assert per_recording["high_lag_centroid_kl_s"].notna().any()


# =================================================================================================
# The selection read as a distribution
# =================================================================================================
#: A second fixture, and it exists for one reason the first cannot serve: the shipped one puts two
#: recordings in each (class, window) cell, which is below ``stats.MIN_GROUP_SIZE``, so every
#: distance on it is a withheld ``NaN`` by design. This one puts six recordings and twelve segments
#: in each cell -- deliberately different numbers, so a per-segment denominator cannot pass a test
#: written for a per-recording one.
_WIDE_GUIDS = 12
_WIDE_HOURS = (1.0, 2.0, 3.0)
_WIDE_SEGMENTS_PER_CELL = 2
_WIDE_RECORDINGS_PER_CELL = _WIDE_GUIDS // 2


def _wide_fixture(shift: int = 0, seed: int = 1):
    """Two classes whose high anchors sit at the same lags, or a rigid ``shift`` apart.

    Args:
        shift: Lag steps by which the *healthy* class's high anchors are moved to a longer lag.
            Zero makes the two classes' distributions the same construction, so their distance is
            a known near-zero; a positive value makes the transport distance a known answer in
            seconds.
        seed: The generator seed.

    Returns:
        ``(per_sample, per_anchor, vectors)``, as :func:`_fixture` returns them.
    """
    rng = np.random.default_rng(seed)
    rows, kl_map, attn_map, samples = [], [], [], []
    segment = 0
    for guid_index in range(_WIDE_GUIDS):
        group = "hie" if guid_index < _WIDE_GUIDS // 2 else "healthy"
        low, high = (HIGH_LAGS if group == "hie" else (HIGH_LAGS[0] + shift, HIGH_LAGS[1] + shift))
        for hours in _WIDE_HOURS:
            for _ in range(_WIDE_SEGMENTS_PER_CELL):
                samples.append(
                    {
                        "sample_index": segment,
                        "guid": f"WIDE{guid_index:02d}",
                        "epoch": -(3600.0 * hours),
                        "clinical_class": group,
                        "subgroup": "hie_cs" if group == "hie" else "healthy_bg_no_cs",
                        "second_stage_onset": -1800.0,
                        "source_conditioned_kl_raw": 1.0,
                    }
                )
                for anchor in range(ANCHORS_PER_SEGMENT):
                    # Three of ten, so the pooled 0.7 quantile falls in the gap between the two
                    # KL populations and the ``high`` band is exactly the constructed set.
                    is_high = anchor < 3
                    kl = rng.uniform(2.0, 3.0) if is_high else rng.uniform(0.05, 0.2)
                    profile = rng.uniform(0.001, 0.01, N_LAGS)
                    profile[low : high + 1] += 1.0 if is_high else 0.0
                    profile[0] += 0.0 if is_high else 1.0
                    profile = profile / profile.sum() * kl
                    rows.append(
                        {
                            "guid": f"WIDE{guid_index:02d}",
                            "epoch": -(3600.0 * hours),
                            "anchor": 100 + anchor,
                            "sample_index": segment,
                            "kld_per_t": kl,
                            "mc_pred_gap": rng.uniform(1.0, 2.0) if is_high else 0.0,
                            "argmax_lag": int(np.argmax(profile)),
                            "seconds_since_contraction": 30.0 if anchor % 2 == 0 else 600.0,
                        }
                    )
                    kl_map.append(profile)
                    attn_map.append(profile / kl)
                segment += 1
    vectors = {
        "kl_lag_map": np.asarray(kl_map, dtype=np.float16),
        "attention_lag_map": np.asarray(attn_map, dtype=np.float16),
    }
    return pd.DataFrame(samples), pd.DataFrame(rows), vectors


def _wide_run(tmp_path_factory, shift: int):
    """Run the analysis once on the wide fixture and return ``(record, directory)``."""
    per_sample, per_anchor, vectors = _wide_fixture(shift=shift)
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=per_sample,
            per_anchor=per_anchor,
            record={},
            retained={},
            vectors={},
            anchor_vectors=vectors,
            results={"lag": {"n_lags": N_LAGS, "delay_steps": 0}},
        ),
        config={},
    )
    directory = tmp_path_factory.mktemp(f"wide{shift}")
    record = analysis.run_lag_high_kl_analysis(
        context, eval_config=EVAL_CONFIG, output_dir=directory
    )
    return record, directory / analysis.ANALYSIS_DIRNAME


@pytest.fixture(scope="module")
def aligned_run(tmp_path_factory):
    """The wide fixture with both classes' high anchors at the same lags."""
    return _wide_run(tmp_path_factory, shift=0)


@pytest.fixture(scope="module")
def shifted_run(tmp_path_factory):
    """The same, with the healthy class's high anchors moved two lag steps later."""
    return _wide_run(tmp_path_factory, shift=2)


def test_every_histogram_cell_is_a_distribution_over_the_lags(tmp_path) -> None:
    """The property that makes the table a histogram: each cell sums to one across the lags.

    A cell built by averaging un-normalised profiles would sum to a coupling magnitude instead, and
    would still draw as a plausible shape and still move across the classes.
    """
    _, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.HISTOGRAM_FILENAME)
    assert len(table)
    totals = table.groupby(["clock", "band", "source", "group", "time_bin"])["density"].sum()
    assert totals.to_numpy() == pytest.approx(1.0)
    # Every lag of the axis is present in every cell, so a gap is a blank rather than a shift.
    counts = table.groupby(["clock", "band", "source", "group", "time_bin"])["lag_step"].nunique()
    assert set(counts.to_numpy().tolist()) == {N_LAGS}


def test_both_bands_and_both_sources_reach_the_histogram_tables(tmp_path) -> None:
    """The two selections and the two readings are one file set, filtered rather than joined."""
    _, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.HISTOGRAM_FILENAME)
    assert set(table["band"]) == set(analysis.HISTOGRAM_BANDS)
    assert set(table["source"]) == {key for key, _, _ in analysis.PROFILE_SOURCES}
    assert set(table["clock"]) == {clock.name for clock in analysis.CLOCKS}


def test_the_high_band_histogram_sits_where_the_selected_anchors_mass_is(tmp_path) -> None:
    """Known answer: the fixture's high anchors carry their mass in :data:`HIGH_LAGS`.

    A histogram built over every anchor rather than the selected ones would put most of its
    density at lag $0$, where the fixture's low-KL anchors sit.
    """
    _, directory = _run(_context(), tmp_path)

    table = pd.read_csv(directory / analysis.HISTOGRAM_FILENAME)
    high = table[(table["band"] == "high") & (table["source"] == "attn")]
    inside = high[high["lag_step"].between(*HIGH_LAGS)]
    share = (
        inside.groupby(["clock", "group", "time_bin"])["density"].sum().to_numpy()
    )
    assert len(share)
    assert (share > 0.9).all()
    # And the lag the unselected anchors sit at carries almost nothing.
    at_zero = high[high["lag_step"] == 0]["density"].to_numpy()
    assert (at_zero < 0.05).all()


def test_the_feature_table_carries_every_statistic_and_agrees_on_the_denominator(tmp_path) -> None:
    """The features describe the drawn histogram, and both tables count the same recordings."""
    _, directory = _run(_context(), tmp_path)

    features = pd.read_csv(directory / analysis.HISTOGRAM_FEATURES_FILENAME)
    table = pd.read_csv(directory / analysis.HISTOGRAM_FILENAME)
    assert set(features["metric"]) == set(analysis.HISTOGRAM_FEATURE_COLUMNS)
    # The two magnitude-carrying statistics are omitted rather than reported meaninglessly.
    assert not any("total_nats" in metric for metric in set(features["metric"]))
    keys = ["clock", "band", "source", "group", "time_bin"]
    from_features = features.groupby(keys)["n_recordings"].max()
    from_table = table.groupby(keys)["n_recordings"].max()
    joined = from_features.to_frame("features").join(from_table.to_frame("table"), how="inner")
    assert len(joined)
    assert (joined["features"] == joined["table"]).all()


def test_a_cell_below_the_minimum_reports_its_counts_and_withholds_the_distance(tmp_path) -> None:
    """The shipped fixture puts two recordings in each cell, one below the floor.

    The row is emitted with its counts rather than dropped, so a reader meets the pair and can see
    why there is no number, and the distance is NaN rather than the zero that would read as
    agreement.
    """
    _, directory = _run(_context(), tmp_path)

    distances = pd.read_csv(directory / analysis.HISTOGRAM_DISTANCE_FILENAME)
    assert len(distances)
    assert set(distances["comparison"]) == {
        analysis.CLASS_PAIR_COMPARISON, analysis.WINDOW_DRIFT_COMPARISON
    }
    thin = distances[distances["n_left"] < 3]
    assert len(thin)
    assert thin["jensen_shannon"].isna().all()
    assert thin["wasserstein_s"].isna().all()


def test_the_histogram_counts_recordings_rather_than_segments(aligned_run) -> None:
    """The pseudo-replication guard, on a fixture where the two numbers differ.

    Each cell holds six recordings and twelve segments, so a denominator that counted segments
    would report twelve and would let a chatty recording outvote a quiet one in every cell.
    """
    _, directory = aligned_run

    table = pd.read_csv(directory / analysis.HISTOGRAM_FILENAME)
    delivery = table[(table["clock"] == "time_to_delivery") & (table["band"] == "high")]
    counts = set(delivery["n_recordings"].to_numpy().tolist())
    assert counts == {_WIDE_RECORDINGS_PER_CELL}
    assert _WIDE_RECORDINGS_PER_CELL * _WIDE_SEGMENTS_PER_CELL not in counts


def test_two_classes_with_the_same_lag_placement_are_a_near_zero_distance(aligned_run) -> None:
    """Both classes built the same way: the distance is the noise floor, not a separation."""
    _, directory = aligned_run

    distances = pd.read_csv(directory / analysis.HISTOGRAM_DISTANCE_FILENAME)
    pairs = distances[
        (distances["comparison"] == analysis.CLASS_PAIR_COMPARISON)
        & (distances["band"] == "high")
        & (distances["source"] == "attn")
        & (distances["clock"] == "time_to_delivery")
    ]
    assert len(pairs)
    assert pairs["wasserstein_s"].notna().all()
    assert (pairs["wasserstein_s"].to_numpy() < SECONDS_PER_STEP / 2.0).all()
    assert (pairs["jensen_shannon"].to_numpy() < 0.2).all()


def test_a_rigid_shift_of_one_class_is_reported_in_seconds(shifted_run) -> None:
    """The known answer that pins the unit end to end: two lag steps apart is $2\\Delta$ seconds.

    A pipeline transporting over lag indices would report a quarter of this and would still
    separate the classes in the right direction, in the right windows.
    """
    _, directory = shifted_run

    distances = pd.read_csv(directory / analysis.HISTOGRAM_DISTANCE_FILENAME)
    pairs = distances[
        (distances["comparison"] == analysis.CLASS_PAIR_COMPARISON)
        & (distances["band"] == "high")
        & (distances["source"] == "attn")
        & (distances["clock"] == "time_to_delivery")
    ]
    assert len(pairs)
    moved = pairs["wasserstein_s"].to_numpy()
    assert moved == pytest.approx(2.0 * SECONDS_PER_STEP, abs=1.0)
    # Oriented worst first, and it is the healthy class that was moved to the longer lag, so the
    # severe class sits at the shorter one and the signed difference is negative.
    assert (pairs["group_left"] == "hie").all()
    assert (pairs["centroid_delta_s"].to_numpy() < 0.0).all()


def test_the_record_keeps_the_distances_untested_and_the_main_families_at_four(tmp_path) -> None:
    """The histogram's families are their own block: the four the analysis defends are still four."""
    record, _ = _run(_context(), tmp_path)

    assert record["lag_histogram"]["distance"]["tested"] is False
    assert set(record["lag_histogram"]["bands"]) == set(analysis.HISTOGRAM_BANDS)
    assert "histogram" in record["untested_note"]
    assert len(record["significance"]) == len(analysis.CLOCKS) * len(analysis.READOUTS)


# =================================================================================================
# The tested half of the histogram: shape features per window, and their drift within recordings
# =================================================================================================
def test_exactly_the_three_shape_features_are_tested_on_the_high_kl_histogram(tmp_path) -> None:
    """Six families -- three features on two clocks -- on the high band's KL source, and the
    centroid, which the main table already tests, is not among them."""
    record, directory = _run(_context(), tmp_path)

    tested = record["lag_histogram"]["tested"]
    assert tested["band"] == "high" and tested["source"] == "kl"
    assert list(tested["readouts"]) == ["hist_median_s", "hist_iqr_s", "hist_entropy_nats"]
    assert "hist_centroid_s" not in tested["readouts"]
    families = {(r["clock"], r["metric_column"]) for r in tested["significance"]}
    assert len(families) == len(analysis.CLOCKS) * len(analysis.HISTOGRAM_READOUTS) == 6
    assert tested["n_holm_families"] == 6
    significance = pd.read_csv(directory / analysis.HISTOGRAM_SIGNIFICANCE_FILENAME)
    assert set(significance["metric_column"]) <= set(analysis.HISTOGRAM_READOUTS)
    assert set(significance["band"]) == {"high"} and set(significance["source"]) == {"kl"}
    assert list(significance.columns[:3]) == ["clock", "band", "source"]


def test_a_rigid_shift_between_the_classes_is_found_by_the_median_family_alone(shifted_run) -> None:
    """Known answer: the shifted fixture moves one class's block two lags later and changes nothing
    else, so the median lag separates the classes in every window of the delivery clock while the
    width and the entropy -- which a rigid shift leaves alone -- do not."""
    _, directory = shifted_run

    significance = pd.read_csv(directory / analysis.HISTOGRAM_SIGNIFICANCE_FILENAME)
    delivery = significance[significance["clock"] == "time_to_delivery"]
    median = delivery[delivery["metric_column"] == "hist_median_s"]
    assert len(median) and median["significant"].all()
    others = delivery[delivery["metric_column"] != "hist_median_s"]
    assert len(others) and not others["significant"].any()
    pairwise = pd.read_csv(directory / analysis.HISTOGRAM_PAIRWISE_FILENAME)
    moved = pairwise[pairwise["metric_column"] == "hist_median_s"]
    # Worst first, and the healthy class was moved to the longer lag, so the severe class sits at
    # the shorter one on every surviving window.
    assert len(moved) and (moved["left"] == "hie").all()
    assert (moved["cliffs_delta"].to_numpy() < 0.0).all()


def test_the_same_construction_in_both_classes_survives_no_histogram_family(aligned_run) -> None:
    """The negative control of the test above: the families find nothing when there is nothing."""
    _, directory = aligned_run

    significance = pd.read_csv(directory / analysis.HISTOGRAM_SIGNIFICANCE_FILENAME)
    assert len(significance) and not significance["significant"].any()


def test_the_drift_is_the_slope_against_forward_labour_time_on_both_clocks() -> None:
    """Known answer for the fit and for its sign convention.

    A feature rising by one unit per hour *toward delivery* is a negative slope against hours before
    delivery and a positive one against hours from onset; the drift reports it as $+1$ on both, and
    a recording scored in fewer than the minimum windows is absent rather than at zero.
    """
    delivery, second = analysis.CLOCKS
    for clock, sign in ((delivery, -1.0), (second, 1.0)):
        recordings = {"A": (3.0, 2.0, 1.0), "B": (2.5, 1.5, 0.5, -0.5), "SHORT": (0.5, 1.0)}
        rows = [
            {
                "group": "hie", "guid": guid,
                clock.bin_column: int(centre * 2), clock.center_column: centre,
                # value = 10 + 1 x forward time, forward time being the negated centre on the
                # delivery clock and the centre itself on the second-stage clock.
                "hist_median_s": 10.0 + sign * centre,
            }
            for guid, centres in recordings.items()
            for centre in centres
        ]
        drift = analysis.recording_drift(clock, "high", "kl", pd.DataFrame(rows), "hist_median_s")

        assert set(drift["guid"]) == {"A", "B"}
        assert drift["slope_per_h"].to_numpy() == pytest.approx(1.0)
        assert (drift["delta_last_minus_first"].to_numpy() > 0.0).all()
        assert set(drift["n_windows"]) == {3, 4}


def test_the_drift_summary_carries_every_class_and_feature_and_declares_its_families(
    aligned_run,
) -> None:
    """Every (feature, class) has a row against zero and every feature an omnibus row; the wide
    fixture's lag placement is constant in time, so nothing drifts and nothing survives.

    The wide fixture rather than the shipped one because only it scores a recording in three
    windows, the fewest a drift can be fitted over; on the shipped fixture every recording spans
    two and the drift table is legitimately empty.
    """
    record, directory = aligned_run

    summary = pd.read_csv(directory / analysis.HISTOGRAM_DRIFT_SUMMARY_FILENAME)
    delivery = summary[summary["clock"] == "time_to_delivery"]
    against_zero = delivery[delivery["row_kind"] == "class_vs_zero"]
    assert set(against_zero["metric"]) == set(analysis.HISTOGRAM_READOUTS)
    assert set(against_zero["group"]) == {"hie", "healthy"}
    omnibus = delivery[delivery["row_kind"] == "across_classes"]
    assert set(omnibus["metric"]) == set(analysis.HISTOGRAM_READOUTS)
    assert not summary["significant"].fillna(False).astype(bool).any()
    drift = pd.read_csv(directory / analysis.HISTOGRAM_DRIFT_FILENAME)
    assert (drift["n_windows"] >= analysis.MIN_DRIFT_WINDOWS).all()
    assert record["lag_histogram"]["tested"]["drift"]["min_windows"] == analysis.MIN_DRIFT_WINDOWS
    assert "positive means" in record["lag_histogram"]["tested"]["drift"]["forward_time"]
