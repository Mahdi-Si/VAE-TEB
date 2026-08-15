r"""The band-resolved readout, and the one join that would be wrong in silence.

The per-channel readouts are positional against the $C_{\mathrm{keep}}$ channels the warm-up budget
left standing; the channel-to-band map is over the $c_y$ **declared** ones. On the shipped dataset
the four dropped channels happen to be the trailing four, so a positional join is **accidentally
correct here** and wrong on any dataset whose survivors are not a prefix -- which is precisely the
failure no test over this fixture alone can catch. So the non-prefix case is *constructed* below,
and it is the most important test in this file.

The rest holds the record to what it may claim. Five coverage counts rather than one ratio, because
the declared and scored numerators coincide at $95$ here by arithmetic accident ($102 - 7 = 98 - 3$)
and "95 of 102" would imply channels the decoder never emitted. The unbanded channels as their own
row rather than folded into a neighbour whose skill they do not share. The per-band gaps summing
back to ``pred_gap``, without which they are five unrelated numbers rather than a decomposition.
And the three limits in the emitted record, because a reader who knows the raw pipeline's
frequency-domain analysis must not carry its contract across: a scattering coefficient is a
modulus, so nothing here can say whether a forecast is mistimed rather than mis-scaled.

**No assertion here is about which band forecasts best.** That is a finding about a model and a
population, and this fixture is neither.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval._reuse import band_partition as shared
from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import band_partition as band_partition_analysis
from teb_vae.lag_attn_cfs.eval.analyses import spectral_skill as analysis

from .conftest import CAUSAL_C_Y, causal_config

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}

#: The five coverage counts of section 5.8, measured on the generated causal shards. Emitted as
#: five numbers rather than one ratio: the declared and scored numerators coincide at 95 by
#: arithmetic accident, and quoting "95 of 102" would imply channels the decoder never emitted.
COVERAGE = {
    "declared_total": 102,
    "dropped_declared": 4,
    "kept_total": 98,
    "known_kept": 95,
    "unknown_kept": 3,
}


# =================================================================================================
# Stubs
# =================================================================================================
def _kept_map(entries: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """A kept-axis channel map, as the channel-map step persists one.

    Args:
        entries: One ``{'kept_channel': ..., 'channel': ..., 'band': ...}`` per kept channel.

    Returns:
        The frame.
    """
    return pd.DataFrame(list(entries))


def _per_sample(guids: Sequence[str], totals: Sequence[float]) -> pd.DataFrame:
    """A per-sample table carrying the recording, the cohorts and ``pred_gap``."""
    return pd.DataFrame(
        {
            "guid": list(guids),
            "clinical_class": ["healthy"] * len(guids),
            "subgroup": ["healthy_bg"] * len(guids),
            "pred_gap": [float(value) for value in totals],
        }
    )


def _context(per_sample: pd.DataFrame, vectors: Dict[str, np.ndarray]) -> AnalysisContext:
    """An analysis context with no task and no loader, as an offline re-run has."""
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={},
        results={}, vectors=vectors,
    )
    return AnalysisContext(collection=collection, config={})


# =================================================================================================
# The join
# =================================================================================================
def test_the_join_follows_the_kept_axis_rather_than_the_declared_index() -> None:
    """**The test this file exists for.**

    The kept channels here are $0, 2, 5$ of a declared width of six -- deliberately not a prefix,
    because on the shipped dataset they are one and a positional join is therefore accidentally
    correct there. Joining by the declared index would read position $5$ of a three-wide vector,
    which is an ``IndexError`` on a good day and one channel's skill attributed to another on a
    bad one.
    """
    kept = _kept_map(
        [
            {"kept_channel": 0, "channel": 0, "band": "deceleration"},
            {"kept_channel": 1, "channel": 2, "band": "variability"},
            {"kept_channel": 2, "channel": 5, "band": "variability"},
        ]
    )

    groups = analysis.band_positions(kept, width=3)

    assert groups["deceleration"].tolist() == [0]
    assert groups["variability"].tolist() == [1, 2]
    assert max(int(value) for positions in groups.values() for value in positions) == 2


def test_the_positions_tile_the_scored_axis_exactly() -> None:
    """Every kept channel is in exactly one band, so the bands are a partition of the axis every
    per-channel readout is indexed on -- which is what makes the per-band sums a decomposition."""
    kept = _kept_map(
        [
            {"kept_channel": index, "channel": index, "band": band}
            for index, band in enumerate(
                ["deceleration", "variability", "variability", shared.UNKNOWN_BAND]
            )
        ]
    )

    groups = analysis.band_positions(kept, width=4)

    covered = sorted(int(value) for positions in groups.values() for value in positions)
    assert covered == [0, 1, 2, 3]


def test_a_map_whose_length_disagrees_with_the_vectors_raises() -> None:
    """A truncating join would give every band a silently wrong denominator; a raise names the two
    widths so an operator can see which artifact is stale."""
    kept = _kept_map(
        [{"kept_channel": index, "channel": index, "band": "variability"} for index in range(3)]
    )

    with pytest.raises(ValueError, match="different channel axes"):
        analysis.band_positions(kept, width=5)


def test_a_map_whose_positions_are_not_a_permutation_of_the_axis_raises() -> None:
    """The failure a length check alone would miss: the right number of rows, indexed wrongly."""
    kept = _kept_map(
        [
            {"kept_channel": 0, "channel": 0, "band": "variability"},
            {"kept_channel": 0, "channel": 1, "band": "variability"},
            {"kept_channel": 2, "channel": 2, "band": "variability"},
        ]
    )

    with pytest.raises(ValueError, match="permutation"):
        analysis.band_positions(kept, width=3)


def test_a_band_the_shared_table_does_not_declare_raises_rather_than_vanishing() -> None:
    """Silently dropped, its channels would leave the per-band sums short of ``pred_gap`` with
    nothing in the record saying which channels went missing."""
    kept = _kept_map(
        [
            {"kept_channel": 0, "channel": 0, "band": "variability"},
            {"kept_channel": 1, "channel": 1, "band": "invented_band"},
        ]
    )

    with pytest.raises(ValueError, match="invented_band"):
        analysis.band_positions(kept, width=2)


def test_a_missing_kept_axis_map_is_a_skip_rather_than_a_raise(tmp_path) -> None:
    """A shard carrying no channel provenance is a property of the dataset, not a fault of the run,
    and the channel-map step already records its own reason."""
    kept, declared, reason = analysis.read_channel_maps(tmp_path)

    assert kept is None and declared is None
    assert analysis.KEPT_CHANNEL_MAP_FILENAME in reason


# =================================================================================================
# The decomposition
# =================================================================================================
def _three_band_run(tmp_path) -> Dict[str, Any]:
    """One run of the analysis over a four-channel axis spanning three bands."""
    (tmp_path / analysis.KEPT_CHANNEL_MAP_FILENAME).write_text(
        _kept_map(
            [
                {"kept_channel": 0, "channel": 0, "band": "deceleration"},
                {"kept_channel": 1, "channel": 2, "band": "variability"},
                {"kept_channel": 2, "channel": 5, "band": "variability"},
                {"kept_channel": 3, "channel": 7, "band": shared.UNKNOWN_BAND},
            ]
        ).to_csv(index=False),
        encoding="utf-8",
    )
    gap = np.array(
        [
            [0.10, 0.20, 0.30, 0.05],
            [0.20, 0.10, 0.10, 0.05],
            [0.05, 0.05, 0.40, 0.10],
            [0.30, 0.10, 0.10, 0.00],
            [0.10, 0.10, 0.10, 0.10],
            [0.20, 0.20, 0.05, 0.05],
        ]
    )
    vectors = {
        analysis.GAP_VECTOR: gap,
        "sq_error_per_channel_base": np.full_like(gap, 2.0),
        "sq_error_per_channel_full": np.full_like(gap, 1.0),
    }
    per_sample = _per_sample(
        ["REC00", "REC00", "REC01", "REC01", "REC02", "REC02"], gap.sum(axis=1)
    )
    result = analysis.run_spectral_skill_analysis(
        _context(per_sample, vectors), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )
    return {"result": result, "gap": gap, "dir": Path(tmp_path)}


def test_the_per_band_gaps_sum_back_to_pred_gap(tmp_path) -> None:
    """Without this they are unrelated numbers rather than a decomposition of one, and every other
    channel-axis split in this package has the same property for the same reason."""
    result = _three_band_run(tmp_path)["result"]

    assert result["recomposition"]["holds"] is True
    assert result["recomposition"]["max_rel_residual"] < 1e-6
    assert result["recomposition"]["n_recordings"] == 3


def test_the_unknown_band_is_its_own_row_and_is_never_folded_into_a_neighbour(tmp_path) -> None:
    """A band whose membership quietly absorbed the unbanded channels would misattribute their
    skill to a frequency they do not have."""
    result = _three_band_run(tmp_path)["result"]
    by_band = {row["band"]: row for row in result["bands"]}

    assert shared.UNKNOWN_BAND in by_band
    assert by_band[shared.UNKNOWN_BAND]["n_channels"] == 1
    assert result["coverage"]["unknown_kept"] == 1
    assert result["coverage"]["kept_total"] == 4
    assert result["coverage"]["known_kept"] == 3


def test_every_band_row_carries_the_channel_count_it_was_measured_over(tmp_path) -> None:
    """A band carried by one channel and one carried by forty are different findings, and the row
    alone cannot say which without its width."""
    result = _three_band_run(tmp_path)["result"]

    counts = {row["band"]: row["n_channels"] for row in result["bands"]}
    assert counts == {"deceleration": 1, "variability": 2, shared.UNKNOWN_BAND: 1}
    assert sum(counts.values()) == result["coverage"]["kept_total"]


def test_the_gap_is_summed_over_a_band_while_the_error_is_averaged(tmp_path) -> None:
    """The asymmetry is the arithmetic rather than a preference: the gap's channel decomposition is
    additive and must sum back, while a squared error summed over channels would make a band's
    error a statement about how many channels it holds."""
    run = _three_band_run(tmp_path)
    by_band = {row["band"]: row for row in run["result"]["bands"]}

    # Two channels at a constant 2.0 average to 2.0, not 4.0.
    assert by_band["variability"]["sq_error_base"] == pytest.approx(2.0)
    assert by_band["variability"]["sq_error_full"] == pytest.approx(1.0)
    assert by_band["variability"]["mse_skill"] == pytest.approx(0.5)
    # And the gap over the same two channels is their sum, per recording then averaged.
    expected = float(run["gap"][:, [1, 2]].sum(axis=1).reshape(3, 2).mean(axis=1).mean())
    assert by_band["variability"]["pred_gap_nats"] == pytest.approx(expected)


def test_the_channel_profile_reconciles_the_two_axes_row_by_row(tmp_path) -> None:
    """A reader going from a band statement to the channels behind it needs both indices, and the
    declared one is the only way to look a channel up in the shard that produced it."""
    run = _three_band_run(tmp_path)
    channels = pd.read_csv(run["dir"] / analysis.ANALYSIS_DIRNAME / analysis.CHANNEL_FILENAME)

    assert list(channels["kept_channel"]) == [0, 1, 2, 3]
    assert list(channels["declared_channel"]) == [0, 2, 5, 7]
    assert len(channels) == run["gap"].shape[1]
    # Each channel's gap is its own column, chained per recording then averaged -- so the profile
    # and the band rows are two reductions of one vector rather than two vectors.
    assert channels["pred_gap"].sum() == pytest.approx(
        sum(row["pred_gap_nats"] for row in run["result"]["bands"])
    )


def test_the_record_states_the_three_limits_and_declares_its_grouped_frame(tmp_path) -> None:
    """A reader of ``summary.json`` has the record and not the module, so the limits travel in it:
    the modulus, the analysing filter rather than the forecast's own spectrum, and the convention
    a phase channel's pair of frequencies is banded by."""
    run = _three_band_run(tmp_path)
    result = run["result"]

    assert set(REQUIRED_RESULT_KEYS) <= set(result)
    assert len(result["limits"]) == 3
    joined = " ".join(result["limits"])
    assert "moduli" in joined
    assert "ANALYSING FILTER" in joined
    assert "freq_hz_primary" in joined
    # Not coherence, said in the record rather than implied by the name alone.
    assert "coherence" in joined

    entry = result["grouped_frames"][0]
    assert entry["path"] == (
        f"{analysis.ANALYSIS_DIRNAME}/{analysis.PER_RECORDING_FILENAME}"
    )
    assert set(entry["value_columns"]) == {
        "pred_gap_deceleration", "pred_gap_variability", f"pred_gap_{shared.UNKNOWN_BAND}"
    }


def test_the_headline_block_carries_every_declared_band_whether_or_not_it_was_scored(
    tmp_path,
) -> None:
    """An arm table's column set must not change with the dataset, and an absent band is ``None``
    rather than ``NaN``: the headline's finiteness check reads a number that is not finite as a
    broken readout, while ``None`` correctly means "this run did not report it"."""
    result = _three_band_run(tmp_path)["result"]

    headline = result["headline"]
    assert set(headline) == {
        f"pred_gap_{band}_nats" for band in analysis.BAND_ORDER
    }
    assert headline["pred_gap_slow_baseline_nats"] is None, "no channel was in that band"
    assert isinstance(headline["pred_gap_variability_nats"], float)


def test_the_per_recording_frame_is_the_one_the_cross_cohort_analysis_reads(tmp_path) -> None:
    """The filename and the ``pred_gap_variability`` column are a contract with that analysis's
    source table, not a local choice: it reads this frame off disk by name."""
    from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup

    run = _three_band_run(tmp_path)
    written = pd.read_csv(
        run["dir"] / analysis.ANALYSIS_DIRNAME / analysis.PER_RECORDING_FILENAME
    )

    wanted = [
        source for source in cross_subgroup.METRIC_SOURCES
        if source.analysis == analysis.ANALYSIS_DIRNAME
    ]
    assert wanted, "the cross-cohort analysis registers no metric from this one"
    for source in wanted:
        assert source.filename == analysis.PER_RECORDING_FILENAME
        assert source.column in written.columns
    assert {"clinical_class", "subgroup"} <= set(written.columns)


def test_a_collection_carrying_no_per_channel_vectors_records_a_skip(tmp_path) -> None:
    """An older run directory whose sidecar predates the per-channel readouts reaches here, and a
    skip with a reason is the honest outcome rather than a raise."""
    (tmp_path / analysis.KEPT_CHANNEL_MAP_FILENAME).write_text(
        _kept_map([{"kept_channel": 0, "channel": 0, "band": "variability"}]).to_csv(index=False),
        encoding="utf-8",
    )

    result = analysis.run_spectral_skill_analysis(
        _context(_per_sample(["REC00"], [1.0]), {}),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    assert result["skipped"] is True
    assert analysis.GAP_VECTOR in result["reason"]
    assert result["n_samples"] is None


# =================================================================================================
# Against the real channel maps
# =================================================================================================
@pytest.fixture(scope="module")
def emitted_maps(cohort_shards, tmp_path_factory):
    """Both channel maps, written by the step that owns them, over the generated causal shards."""
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    config = causal_config()
    config["dataset_config"]["vae_train_datasets"] = list(cohort_shards)
    config["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    resolved = resolve_warmup_budget(config)
    assert resolved is not None

    output_dir = tmp_path_factory.mktemp("spectral_skill_maps")
    record = band_partition_analysis.emit_partition(
        list(cohort_shards),
        output_dir,
        declared={"target": CAUSAL_C_Y, "source": None},
        trim_steps=15,
        keep_index=resolved.target.keep_index,
        kept_width=resolved.target.kept_width,
    )
    assert record["skipped"] is False, record
    return Path(output_dir)


def test_the_five_coverage_counts_are_read_off_the_persisted_maps(emitted_maps) -> None:
    r"""102 declared, 4 dropped, 98 scored, 95 of them banded, 3 not.

    Five counts rather than one ratio: $102 - 7 = 98 - 3$, so the declared and scored numerators
    coincide at 95 by arithmetic accident and "95 of 102" would imply the analysis banded channels
    the decoder never emitted.
    """
    kept, declared, reason = analysis.read_channel_maps(emitted_maps)
    assert reason == ""
    assert kept is not None and declared is not None

    groups = analysis.band_positions(kept, width=COVERAGE["kept_total"])
    counts = analysis.coverage_counts(kept, declared, groups)

    for name, value in COVERAGE.items():
        assert counts[name] == value, name


def test_all_four_dropped_channels_are_unbanded(emitted_maps) -> None:
    """They sit below the phase selection's 0.008 Hz floor, which is the same property that makes
    their warm-up the longest -- so the budget removes exactly the low end of the frequency axis."""
    kept, declared, _ = analysis.read_channel_maps(emitted_maps)
    groups = analysis.band_positions(kept, width=COVERAGE["kept_total"])

    counts = analysis.coverage_counts(kept, declared, groups)

    assert counts["dropped_bands"] == {shared.UNKNOWN_BAND: COVERAGE["dropped_declared"]}


def test_the_scored_axis_is_the_kept_one_and_no_dropped_channel_appears_on_it(
    emitted_maps,
) -> None:
    """No emitted number may imply a dropped channel was scored, so the four the budget removed
    are absent from the axis rather than present at zero."""
    kept, _, _ = analysis.read_channel_maps(emitted_maps)
    groups = analysis.band_positions(kept, width=COVERAGE["kept_total"])

    assert sum(int(positions.size) for positions in groups.values()) == COVERAGE["kept_total"]
    assert set(int(value) for value in kept["channel"]).isdisjoint({32, 33, 34, 35})
    assert len(kept) == COVERAGE["kept_total"]


def test_the_analysis_runs_end_to_end_against_the_real_maps(emitted_maps, tmp_path) -> None:
    """The two halves joined: the real 98-wide band axis and a per-channel vector of that width."""
    import shutil

    for name in (analysis.KEPT_CHANNEL_MAP_FILENAME, analysis.DECLARED_CHANNEL_MAP_FILENAME):
        shutil.copy(emitted_maps / name, tmp_path / name)
    generator = np.random.default_rng(0)
    gap = generator.normal(size=(6, COVERAGE["kept_total"]))
    vectors = {
        analysis.GAP_VECTOR: gap,
        "sq_error_per_channel_base": np.abs(generator.normal(size=gap.shape)) + 1.0,
        "sq_error_per_channel_full": np.abs(generator.normal(size=gap.shape)) + 1.0,
    }
    per_sample = _per_sample(
        ["REC00", "REC00", "REC01", "REC01", "REC02", "REC02"], gap.sum(axis=1)
    )

    result = analysis.run_spectral_skill_analysis(
        _context(per_sample, vectors), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    assert result["skipped"] is False
    assert result["recomposition"]["holds"] is True
    for name, value in COVERAGE.items():
        assert result["coverage"][name] == value, name
    for name in result["files"]:
        assert (tmp_path / analysis.ANALYSIS_DIRNAME / name).is_file(), name
    # Every band the map actually carries has a row, and none of them is the whole axis.
    assert 1 < len(result["bands"]) <= len(analysis.BAND_ORDER)


# =================================================================================================
# On a real run, and offline against a finished one
# =================================================================================================
@pytest.mark.slow
def test_the_analysis_runs_and_recomposes_on_a_real_run(collected_run) -> None:
    """Read off a pass that actually ran it: an analysis can satisfy every stub above and emit
    nothing on the inputs a real pass hands it."""
    summary = collected_run["summary"]
    block = summary["results"][analysis.ANALYSIS_DIRNAME]

    assert block["skipped"] is False, block.get("reason")
    assert block["n_samples"] == summary["results"]["n_samples"]
    assert block["recomposition"]["holds"] is True, block["recomposition"]
    for name, value in COVERAGE.items():
        assert block["coverage"][name] == value, name
    assert block["coverage"]["kept_total"] == sum(
        row["n_channels"] for row in block["bands"]
    )
    for name in block["files"]:
        assert (collected_run["results_dir"] / analysis.ANALYSIS_DIRNAME / name).is_file(), name


@pytest.mark.slow
def test_every_registered_headline_path_resolves_on_a_real_runs_results(collected_run) -> None:
    """The stub check in the binding suite proves the paths are well formed; only a real run
    proves they resolve against what the three analyses actually returned."""
    from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING

    headline = collected_run["summary"]["results"]["headline"]

    for name, path in CFS_BINDING.headline_scalars:
        assert name in headline, name
        # ``pred_gap_slow_baseline_nats`` is legitimately ``None`` when no kept channel is in that
        # band, which is the case on this dataset; every other entry must be a number.
        if path[-1] != "pred_gap_slow_baseline_nats":
            assert headline[name] is not None, (name, path)


@pytest.mark.slow
def test_it_runs_against_a_finished_directory_with_no_model(
    collected_run, tmp_path, monkeypatch
) -> None:
    """The property the persisted map exists for, proved with ``forward`` rigged to raise.

    The band axis has to be resolvable with no checkpoint, no model and no GPU, because that is
    exactly the path ``--only spectral_skill --output-dir <a finished run>`` takes -- and it is
    why the join goes through a CSV on disk rather than through the model's own channel gate. A
    spy is the only way to tell "did not need the model" from "happened not to use it".
    """
    import json
    import shutil

    from teb_vae.lag_attn_cfs.eval import run as run_module
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    run_dir = tmp_path / "rerun"
    shutil.copytree(collected_run["results_dir"].parent, run_dir)

    def _explode(*args, **kwargs):
        raise AssertionError("the model was built and forwarded on an offline re-run")

    monkeypatch.setattr(SeqVaeLagAttnCfs, "forward", _explode)

    exit_code = run_module.main(
        None, run_dir, only=analysis.ANALYSIS_DIRNAME, device="cpu"
    )

    results_dir = run_dir / run_module.RESULTS_DIRNAME
    summary = json.loads((results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    block = summary["results"][analysis.ANALYSIS_DIRNAME]
    assert exit_code == 0
    assert summary["checkpoint"] is None
    assert summary["analyses_selected"] == [analysis.ANALYSIS_DIRNAME]
    assert block["skipped"] is False, block.get("reason")
    # The same band axis the collected pass resolved, from the same file on disk.
    assert block["coverage"] == collected_run["summary"]["results"][
        analysis.ANALYSIS_DIRNAME
    ]["coverage"]
