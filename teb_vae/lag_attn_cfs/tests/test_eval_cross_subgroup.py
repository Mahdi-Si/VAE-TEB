r"""Which cohort separations survive being asked properly, and on which unit they are asked.

Three properties are pinned here, and each catches a different way a by-cohort table is turned
into a false finding.

**The unit is the recording.** Every vector entering a test has one value per recording, and the
assertion is on the *length of the arrays the tests consume* rather than on a docstring: one
recording contributes tens of overlapping segments, so a test over segments is pseudo-replicated
by that factor and its $p$-values are anticonservative by an amount nothing in the output shows.

**A missing source is recorded, never raised.** The analysis is built to run against a partial run
directory -- that is what makes ``--only cross_subgroup`` against a finished run work at all -- so
a source whose analysis was skipped has to be information rather than an error. In this package
that path is exercised by construction rather than by contrivance: :data:`METRIC_SOURCES` names
tables written by analyses that have not landed yet, and every one of them is missing today.

**It needs no model.** The offline re-run is asserted with the model's ``forward`` rigged to
raise, because a spy is the only way to tell "did not need the model" from "happened not to use
it".

**And this cell's own metrics are in the family.** Three warm-up tertiles, the availability-clock
difference, and one band of the band-resolved skill -- the readouts no other cell in the grid has,
each named by the per-recording table its analysis writes rather than reached for through it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.figures_seam import figure_filename
from teb_vae.lag_attn_cfs.eval._reuse import band_partition, labels
from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup as analysis

#: A source pointing at the frame the fixtures below write.
_SOURCE = analysis.MetricSource("coupling", "coupling_per_recording.csv", "mc_pred_gap",
                                higher_is_better=True)


def _write_frame(root: Path, source: analysis.MetricSource, values: Dict[str, List[float]]) -> Path:
    """Write a per-recording frame with one row per recording, cohort-labelled."""
    rows: List[Dict[str, Any]] = []
    for group, numbers in values.items():
        for index, number in enumerate(numbers):
            rows.append(
                {
                    "guid": f"{group}_{index:03d}",
                    labels.SUBGROUP_COLUMN: group,
                    labels.CLASS_COLUMN: "healthy" if group.startswith("healthy") else "acidosis",
                    source.column: number,
                }
            )
    directory = root / source.analysis
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / source.filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =============================================================================
# The unit, and the three layers
# =============================================================================
def test_every_tested_vector_holds_one_value_per_recording(tmp_path) -> None:
    """The assertion with teeth: the arrays entering the omnibus test are counted, and their
    lengths must be the recording counts rather than the segment counts they were reduced from."""
    _write_frame(
        tmp_path,
        _SOURCE,
        {"healthy_bg_cs": [1.0, 2.0, 3.0, 4.0], "acidosis_cs": [5.0, 6.0, 7.0]},
    )

    frame = pd.read_csv(tmp_path / _SOURCE.analysis / _SOURCE.filename)
    usable, excluded = analysis.usable_groups(frame, _SOURCE.column, labels.SUBGROUP_COLUMN)

    assert {name: values.size for name, values in usable.items()} == {
        "healthy_bg_cs": 4, "acidosis_cs": 3
    }
    assert excluded == {}
    # The frame holds one row per recording, so the two counts agree here by construction -- and
    # that is the property: a segment-level source would make them disagree.
    assert sum(values.size for values in usable.values()) == frame["guid"].nunique()


def test_a_cohort_below_the_minimum_is_excluded_and_counted(tmp_path) -> None:
    """"This subgroup had two recordings" is the explanation for a missing comparison, and a
    reader who cannot see it will assume the comparison was made."""
    _write_frame(
        tmp_path,
        _SOURCE,
        {"healthy_bg_cs": [1.0, 2.0, 3.0, 4.0], "acidosis_cs": [5.0, 6.0]},
    )

    frame = pd.read_csv(tmp_path / _SOURCE.analysis / _SOURCE.filename)
    usable, excluded = analysis.usable_groups(frame, _SOURCE.column, labels.SUBGROUP_COLUMN)

    assert list(usable) == ["healthy_bg_cs"]
    assert excluded == {"acidosis_cs": 2}


def test_a_separated_pair_is_significant_and_an_overlapping_one_is_not(tmp_path) -> None:
    """Known answers on both sides. Two cohorts drawn far apart must survive Holm; two drawn from
    the same values must not -- otherwise the procedure is reporting its own arithmetic."""
    separated = tmp_path / "separated"
    overlapping = tmp_path / "overlapping"
    _write_frame(
        separated, _SOURCE,
        {"healthy_bg_cs": [1.0, 1.1, 1.2, 1.3, 1.4], "acidosis_cs": [9.0, 9.1, 9.2, 9.3, 9.4]},
    )
    _write_frame(
        overlapping, _SOURCE,
        {"healthy_bg_cs": [1.0, 1.1, 1.2, 1.3, 1.4], "acidosis_cs": [1.05, 1.15, 1.25, 1.35, 1.45]},
    )

    strong = analysis.analyse_metrics(separated, sources=(_SOURCE,))
    weak = analysis.analyse_metrics(overlapping, sources=(_SOURCE,))

    assert strong["significant_metrics"] == [_SOURCE.name]
    assert weak["significant_metrics"] == []
    # The pairwise layer runs only for what survived the omnibus: running it otherwise is the
    # multiple-comparison problem with extra steps.
    assert set(strong["pairwise"]) == {_SOURCE.name}
    assert weak["pairwise"] == {}
    assert abs(strong["pairwise"][_SOURCE.name][0]["cliffs_delta"]) == pytest.approx(1.0)


def test_the_holm_correction_is_applied_across_the_metrics_as_one_family(tmp_path) -> None:
    """Each metric's raw p is corrected against the number of tests that ran, not reported bare."""
    second = analysis.MetricSource("latent", "latent_per_recording.csv", "source_conditioned_kl_raw")
    _write_frame(tmp_path, _SOURCE, {"healthy_bg_cs": [1.0] * 5, "acidosis_cs": [9.0] * 5})
    _write_frame(tmp_path, second, {"healthy_bg_cs": [1.0] * 5, "acidosis_cs": [9.0] * 5})

    record = analysis.analyse_metrics(tmp_path, sources=(_SOURCE, second))

    assert record["n_metrics_tested"] == 2
    for item in record["omnibus"]:
        assert item["correction"] == "holm"
        assert item["n_tests_in_family"] == 2
        assert item["p_holm"] >= item["p_value"]
        assert item["unit"] == "recording"


# =============================================================================
# Missing sources are recorded, not raised
# =============================================================================
def test_a_missing_file_is_recorded_rather_than_raised(tmp_path) -> None:
    record = analysis.analyse_metrics(tmp_path, sources=(_SOURCE,))

    assert record["n_metrics_tested"] == 0
    assert [item["reason"] for item in record["missing_sources"]] == [
        "coupling/coupling_per_recording.csv was not written"
    ]


def test_a_missing_column_names_the_column(tmp_path) -> None:
    directory = tmp_path / _SOURCE.analysis
    directory.mkdir(parents=True)
    pd.DataFrame({"guid": ["a"], labels.SUBGROUP_COLUMN: ["healthy_bg_cs"]}).to_csv(
        directory / _SOURCE.filename, index=False
    )

    record = analysis.analyse_metrics(tmp_path, sources=(_SOURCE,))

    assert "carries no 'mc_pred_gap' column" in record["missing_sources"][0]["reason"]


def test_a_frame_without_the_cohort_column_is_recorded(tmp_path) -> None:
    """The column the collection pass attaches. A frame without it is a frame from an older run,
    and testing it on an axis it does not carry would silently compare nothing."""
    directory = tmp_path / _SOURCE.analysis
    directory.mkdir(parents=True)
    pd.DataFrame({"guid": ["a", "b"], _SOURCE.column: [1.0, 2.0]}).to_csv(
        directory / _SOURCE.filename, index=False
    )

    record = analysis.analyse_metrics(tmp_path, sources=(_SOURCE,))

    assert record["n_metrics_tested"] == 0
    assert "carries no 'subgroup' column" in record["missing_sources"][0]["reason"]


def test_a_single_cohort_split_records_a_skip_and_writes_nothing(tmp_path) -> None:
    """The ordinary outcome on the healthy-only pretraining split, and not an error."""
    _write_frame(tmp_path, _SOURCE, {"healthy_bg_cs": [1.0, 2.0, 3.0, 4.0]})

    result = analysis.run_cross_subgroup_analysis(
        None, eval_config={}, output_dir=tmp_path, probe=None
    )

    assert result["skipped"] is True
    assert result["n_samples"] is None
    assert not (tmp_path / analysis.ANALYSIS_DIRNAME).exists()


# =============================================================================
# What it writes
# =============================================================================
def test_the_analysis_writes_its_tables_the_record_and_the_figure(tmp_path) -> None:
    _write_frame(
        tmp_path, _SOURCE,
        {"healthy_bg_cs": [1.0, 1.1, 1.2, 1.3, 1.4], "acidosis_cs": [9.0, 9.1, 9.2, 9.3, 9.4]},
    )

    result = analysis.run_cross_subgroup_analysis(
        None, eval_config={}, output_dir=tmp_path, probe=None
    )

    directory = tmp_path / analysis.ANALYSIS_DIRNAME
    for name in (
        analysis.SIGNIFICANCE_FILENAME, analysis.PAIRWISE_FILENAME, analysis.RESULT_FILENAME,
        figure_filename(analysis.HEATMAP_FIGURE),
    ):
        assert (directory / name).is_file(), name
    significance = pd.read_csv(directory / analysis.SIGNIFICANCE_FILENAME)
    assert list(significance["unit"]) == ["recording"]
    # The inference path travels with the coefficient: which analysis, which file, which column,
    # and which direction is good.
    assert list(significance["file"]) == ["coupling/coupling_per_recording.csv"]
    assert bool(significance["higher_is_better"].iloc[0]) is True
    assert result["unit"] == "recording"
    assert result["largest_effects"][0]["magnitude"]
    # Every path in the returned record is a bare filename: the summary is a block two runs of one
    # checkpoint must compare equal, and an absolute path differs between them for no reason.
    assert all("/" not in name and "\\" not in name for name in result["files"])


def test_the_written_record_round_trips_as_json(tmp_path) -> None:
    """Non-finite statistics are ordinary here -- a degenerate cohort produces them -- and a file
    only Python can read back is not a record."""
    _write_frame(tmp_path, _SOURCE, {"healthy_bg_cs": [1.0] * 5, "acidosis_cs": [9.0] * 5})

    analysis.run_cross_subgroup_analysis(None, eval_config={}, output_dir=tmp_path, probe=None)

    path = tmp_path / analysis.ANALYSIS_DIRNAME / analysis.RESULT_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["group_column"] == labels.SUBGROUP_COLUMN
    assert record["alpha"] == pytest.approx(analysis.DEFAULT_ALPHA)


def test_neither_the_alpha_nor_the_minimum_cohort_size_is_configurable() -> None:
    """An operator who could lower either could make any metric significant, which is why they are
    module constants and why the analysis ignores ``eval_config`` entirely."""
    from teb_vae.lag_attn_cfs.eval import config_schema

    assert "alpha" not in config_schema.VALID_KEYS
    assert "min_group_size" not in config_schema.VALID_KEYS


def test_the_metric_sources_name_per_recording_tables_only() -> None:
    """A source pointing at ``per_sample.csv`` would test segments and read as though it tested
    recordings -- the exact pseudo-replication this analysis's unit exists to avoid."""
    assert analysis.METRIC_SOURCES
    for source in analysis.METRIC_SOURCES:
        assert source.filename.endswith(".csv")
        assert "per_sample" not in source.filename


# =============================================================================
# This cell's own metrics in the family
# =============================================================================
def test_the_cells_own_readouts_are_registered_and_named_by_their_own_tables() -> None:
    """The three the sibling cannot have. Each is named by the per-recording CSV its analysis
    writes rather than reached for through the analysis, which is what keeps this analysis able to
    run against a directory where that analysis was skipped."""
    registered = {source.name for source in analysis.METRIC_SOURCES}

    assert {
        "warmup.pred_gap_warm_lo",
        "warmup.pred_gap_warm_mid",
        "warmup.pred_gap_warm_hi",
        "source_null.coupling_minus_clock",
    } <= registered
    assert any(source.analysis == "spectral_skill" for source in analysis.METRIC_SOURCES)


def test_the_frequency_domain_pair_the_sibling_registers_is_gone() -> None:
    """``coherence`` is not ported at all -- a scattering coefficient is a modulus, so phase
    agreement has no analogue here at any window length. A source still naming it would look for a
    file this pipeline can never write and report it missing on every run, forever."""
    assert not any(source.analysis == "coherence" for source in analysis.METRIC_SOURCES)


def test_the_band_named_here_is_a_real_clinical_band() -> None:
    """The band-resolved entry names one column of a table this package's own analysis writes, and
    the band in that column name has to be one the shared partition actually produces -- otherwise
    the source is permanently missing and nothing says why."""
    assert analysis.SPECTRAL_SKILL_BAND in band_partition.CLINICAL_BANDS
    assert any(
        source.column.endswith(analysis.SPECTRAL_SKILL_BAND)
        for source in analysis.METRIC_SOURCES
        if source.analysis == "spectral_skill"
    )


def test_the_directions_registered_are_the_ones_a_larger_value_is_better_in() -> None:
    """``higher_is_better`` is a direction this table publishes beside a signed effect size, so a
    reader never has to go back to the analysis that produced the column. The gap-like readouts
    are the ones it is true for; the error-like ones are the ones it is false for."""
    directions = {source.name: source.higher_is_better for source in analysis.METRIC_SOURCES}

    assert directions["coupling.mc_pred_gap"] is True
    assert directions["source_null.coupling_minus_clock"] is True
    assert directions["warmup.pred_gap_warm_lo"] is True
    assert directions["forecast.sq_error_full"] is False
    assert directions["latent.logvar_prior_floor_frac"] is False


def test_every_source_that_cannot_be_satisfied_yet_is_recorded_rather_than_raised(tmp_path) -> None:
    """The partial-directory case, at its widest: most of :data:`METRIC_SOURCES` names a table an
    analysis that has not landed will write. Running against a directory holding only the coupling
    table must therefore test one metric and record every other source as missing -- which is the
    behaviour that keeps ``--only cross_subgroup`` usable, and is not a special case of it."""
    _write_frame(
        tmp_path, _SOURCE,
        {"healthy_bg_cs": [1.0, 1.1, 1.2, 1.3, 1.4], "acidosis_cs": [9.0, 9.1, 9.2, 9.3, 9.4]},
    )

    record = analysis.analyse_metrics(tmp_path)

    # Every source but the one column actually on disk, recorded rather than raised -- and every
    # one of them named, so the set is a statement about which sources went unsatisfied rather
    # than a count.
    assert record["n_metrics_tested"] == 1
    assert {item["column"] for item in record["missing_sources"]} == {
        source.column for source in analysis.METRIC_SOURCES if source.column != _SOURCE.column
    }
    # The two ways a source goes missing are distinguished, because they are fixed differently: a
    # file that was never written means its analysis did not run, and a file present without the
    # column means the analysis that wrote it did not produce that readout.
    assert any("was not written" in item["reason"] for item in record["missing_sources"])
    assert any(
        item["column"] == "source_conditioned_kl_raw" and "carries no" in item["reason"]
        for item in record["missing_sources"]
    )


# =============================================================================
# Offline, against a finished run, with no model
# =============================================================================
@pytest.mark.slow
def test_it_runs_against_a_finished_directory_with_no_checkpoint(
    collected_run, tmp_path, monkeypatch
) -> None:
    """The property the whole collect/emit split exists for, proved with ``forward`` rigged to
    raise: a spy is the only way to tell "did not need the model" from "happened not to use it"."""
    from teb_vae.lag_attn_cfs.eval import run as run_module
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs

    run_dir = tmp_path / "rerun"
    shutil.copytree(collected_run["results_dir"].parent, run_dir)

    def _explode(*args, **kwargs):
        raise AssertionError("the model was built and forwarded on an offline re-run")

    monkeypatch.setattr(SeqVaeLagAttnCfs, "forward", _explode)

    exit_code = run_module.main(None, run_dir, only="cross_subgroup", device="cpu")

    results_dir = run_dir / run_module.RESULTS_DIRNAME
    summary = json.loads((results_dir / run_module.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    block = summary["results"]["cross_subgroup"]
    assert exit_code == 0
    assert summary["analyses_selected"] == ["cross_subgroup"]
    assert block["skipped"] is False
    # It read the tables the earlier pass wrote rather than recomputing them, and every source it
    # could not satisfy is *recorded* rather than raised. Here that is every analysis that has
    # not landed yet -- the widest form of the partial-directory case this analysis has to
    # tolerate, and the reason the assertion is a subset rather than a fixed list.
    missing = {item["column"] for item in block["missing_sources"]}
    landed = {
        source.column for source in analysis.METRIC_SOURCES
        if source.analysis in ("coupling", "latent")
    }
    assert not (missing & landed), sorted(missing & landed)
    assert missing <= {source.column for source in analysis.METRIC_SOURCES} - landed
    assert all(item["reason"] for item in block["missing_sources"])


@pytest.mark.slow
def test_the_real_run_tests_the_cohorts_on_per_guid_vectors(collected_run) -> None:
    """End to end on the generated multi-subgroup shards: the count behind every test is the run's
    **recording** count, and the fixture is built so the two counts differ."""
    from scripts.make_tiny_shard import COHORT_SUBGROUPS

    block = collected_run["summary"]["results"]["cross_subgroup"]
    significance = pd.read_csv(
        collected_run["results_dir"] / analysis.ANALYSIS_DIRNAME / analysis.SIGNIFICANCE_FILENAME
    )

    n_recordings = collected_run["summary"]["results"]["n_recordings"]
    n_samples = collected_run["summary"]["results"]["n_samples"]
    assert block["skipped"] is False
    assert n_recordings < n_samples, "the fixture must hold more segments than recordings"
    # The assertion that pins the unit: had the sources been the per-sample table, this would be
    # the segment count and nothing in the output would say which was tested.
    assert set(significance["n_recordings"]) == {n_recordings}
    assert set(np.asarray(significance["n_groups"])) == {len(COHORT_SUBGROUPS)}
