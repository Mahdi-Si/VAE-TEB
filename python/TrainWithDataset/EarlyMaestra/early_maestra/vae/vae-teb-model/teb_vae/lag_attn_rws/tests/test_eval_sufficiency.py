r"""The sufficiency analysis: one population, two gaps, and the caveats travelling with them.

What can go wrong here is not that the arithmetic is wrong but that the three numbers describe
three different sets of recordings -- $D_{\mathrm{oracle}}$ the held-out half, $D_{\mathrm{base}}$
the whole split, $\Delta_{\mathrm{suff}}$ their difference -- at which point the gap is a
comparison of populations wearing the name of a measurement. So the assertions here are mostly
about *who* was measured: that the join landed, that the per-recording frame holds exactly the
held-out recordings, and that both gaps come off the same frame.

The second thing that can go wrong is silent: the two bias directions and the "estimate, not a
bound" sentence live in the emitted JSON rather than only in a docstring, because a reader meets
the number in ``summary.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval.figures_seam import figure_filename
from teb_vae.lag_attn_rws.eval import oracle
from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext
from teb_vae.lag_attn_rws.eval.analyses import sufficiency


@pytest.fixture(scope="module")
def result(evaluated) -> dict:
    """The sufficiency block of the shared evaluation run."""
    block = evaluated["summary"]["results"].get("sufficiency")
    assert block is not None, "the run produced no sufficiency block at all"
    return block


@pytest.fixture(scope="module")
def directory(evaluated) -> Path:
    """The analysis's own output directory in that run."""
    return Path(evaluated["results_dir"]) / sufficiency.ANALYSIS_DIRNAME


# =============================================================================
# The measurement happened, and on one population
# =============================================================================
def test_the_analysis_ran_rather_than_recording_a_skip(result) -> None:
    """A skip here is legitimate on a pass with no model; this fixture has one, so a skip would
    mean the guards fired on a population that can carry the measurement."""
    assert not result.get("skipped"), result.get("reason")
    assert result["n_samples"] > 0


def test_the_split_covers_every_recording_and_shares_none(result, evaluated) -> None:
    """The property the whole number rests on: a probe scored on recordings it was fitted to
    reports the bottleneck as costing less than it does."""
    split = result["split"]

    assert split["recordings_disjoint"] is True
    assert split["n_fit_recordings"] > 0 and split["n_held_out_recordings"] > 0
    assert (
        split["n_fit_recordings"] + split["n_held_out_recordings"]
        == evaluated["summary"]["results"]["n_recordings"]
    )
    # Recoverable from the summary alone, which is what makes the split reproducible.
    assert isinstance(split["seed"], int)


def test_the_reported_recordings_are_the_held_out_ones(result, directory) -> None:
    """The frame every statistic is computed on holds one row per held-out recording -- not per
    segment, and not per recording of the whole split."""
    frame = pd.read_csv(directory / sufficiency.PER_RECORDING_FILENAME)

    assert len(frame) == result["composition"]["n_held_out_recordings"]
    assert len(frame) == result["split"]["n_held_out_recordings"]
    assert frame["guid"].nunique() == len(frame)


def test_both_gaps_are_differences_over_the_same_frame(directory) -> None:
    r"""$\Delta_{\mathrm{suff}}$ and ``pred_gap`` are read side by side off one figure, so they
    have to be differences of the same per-recording means rather than three separate headlines."""
    frame = pd.read_csv(directory / sufficiency.PER_RECORDING_FILENAME)

    for name, left, right, _meaning in sufficiency.GAP_METRICS:
        assert np.allclose(
            np.asarray(frame[name], dtype=np.float64),
            np.asarray(frame[left], dtype=np.float64)
            - np.asarray(frame[right], dtype=np.float64),
            equal_nan=True,
        )


def test_the_summary_gap_matches_the_two_scores_it_is_a_difference_of(result) -> None:
    """The headline arithmetic, checked rather than assumed: a gap computed on a different
    reduction from its two terms would be off by an amount nothing else would show."""
    gap = result["gap"]

    assert gap["delta_suff_nats"] == pytest.approx(
        gap["d_base_mc_nats"] - gap["d_oracle_nats"], rel=1e-6, abs=1e-9
    )


# =============================================================================
# The caveats are in the artifact, not only in the documentation
# =============================================================================
def test_both_bias_directions_reach_the_emitted_record(result) -> None:
    """They oppose, and neither is measured; a reader who sees only one would read the number as
    a one-sided bound in whichever direction happened to be written down."""
    directions = {entry["direction"] for entry in result["bias_directions"]}

    assert directions == {"understates", "overstates"}
    for entry in result["bias_directions"]:
        assert "target_state" in entry["cause"] or "pretraining" in entry["cause"]
    assert "estimate, not a bound" in result["estimate_not_a_bound"]


def test_the_extra_encoder_pass_is_recorded_as_a_number(result) -> None:
    """The rest of the pipeline holds that the collection pass is the only model-touching cost.
    This analysis amends that, and the amendment is a measured size rather than a footnote."""
    extra = result["plan"]["extra_encoder_pass"]

    assert extra["n_segments"] > 0
    assert extra["n_bytes"] > 0
    assert "thousands of passes" in extra["reason"]


# =============================================================================
# Convergence and capacity accompany the number
# =============================================================================
def test_the_probe_reports_its_own_convergence(result) -> None:
    """An under-trained probe understates the gap. Whether it converged is therefore part of the
    result rather than something a reader has to infer from the curve."""
    convergence = result["convergence"]

    assert isinstance(convergence["converged"], bool)
    assert convergence["detail"]
    assert np.isfinite(float(convergence["final_held_out_nats"]))


def test_the_capacity_check_ran_and_says_which_way_it_went(result) -> None:
    capacity = result["capacity"]

    assert capacity["checked"] is True
    assert isinstance(capacity["capacity_bound"], bool)
    assert capacity["n_parameters_wide"] > capacity["n_parameters"]
    assert capacity["margin_nats"] == oracle.CAPACITY_MARGIN_NATS


def test_the_training_curve_is_written_for_both_widths(directory) -> None:
    """The evidence behind the convergence flag, on disk, so it can be looked at rather than
    trusted."""
    curve = pd.read_csv(directory / sufficiency.CURVE_FILENAME)

    assert set(curve.columns) == {"width_multiplier", "step", "held_out_nats", "fit_nats"}
    assert set(curve["width_multiplier"]) == {1, oracle.CAPACITY_WIDTH_MULTIPLIER}
    assert len(curve) > 0


# =============================================================================
# The artifacts
# =============================================================================
def test_every_declared_file_was_written(result, directory) -> None:
    missing = [name for name in result["files"] if not (directory / name).is_file()]

    assert missing == []
    assert figure_filename(sufficiency.SUFFICIENCY_FIGURE) in result["files"]


def test_the_grouped_variants_were_fanned_out_by_the_runner(result) -> None:
    """The analysis declares a frame; the runner emits the by-class and by-subgroup cuts. A
    declaration nothing acts on would be an analysis quietly reporting one pooled number."""
    declared = {entry["stem"] for entry in result["grouped_frames"]}

    assert declared
    assert set(result.get("grouped", {})) == declared


# =============================================================================
# The offline path, and the join it depends on
# =============================================================================
def test_a_pass_with_no_model_records_a_skip_rather_than_a_number(evaluated, tmp_path) -> None:
    """Every other analysis runs offline off the tables. This one cannot -- the encoder state it
    reads is on neither -- so it must say so instead of reporting an empty measurement."""
    from teb_vae.lag_attn_rws.eval import collect

    collection = collect.load_collection(evaluated["results_dir"])
    context = AnalysisContext(collection=collection, config={}, task=None, loader=None)

    outcome = sufficiency.run_sufficiency_analysis(
        context, eval_config={"seed": 0}, output_dir=tmp_path, probe=None
    )

    assert outcome["skipped"] is True
    assert outcome["n_samples"] is None
    assert "loader" in outcome["reason"]
    assert outcome["files"] == []


def test_the_join_keys_on_the_rounded_epoch_and_drops_what_does_not_match() -> None:
    """The join is the only thing making $D_{\\mathrm{oracle}}$ and $D_{\\mathrm{base}}$
    describe the same segments; a key that silently matched nothing would report an empty
    analysis rather than a broken key."""
    per_sample = pd.DataFrame(
        {
            "guid": ["A", "A", "B"],
            "epoch": [-600.0000001, -1200.0, -600.0],
            "mc_nll_base_block": [1.0, 2.0, 3.0],
        }
    )
    per_segment = {
        "guid": ["A", "B", "C"],
        "epoch": np.asarray([-600.0, -600.4, -600.0]),
        "nll_oracle_block": np.asarray([10.0, 20.0, 30.0]),
        "oracle_n_anchors": np.asarray([5.0, 5.0, 5.0]),
    }

    merged = sufficiency.join_oracle_scores(per_sample, per_segment)

    # A float that survived a CSV round trip still joins; a recording absent from the table does
    # not appear at all.
    assert sorted(merged["guid"]) == ["A", "B"]
    assert list(merged["nll_oracle_block"]) == [10.0, 20.0]


def test_a_join_that_matches_nothing_is_reported_rather_than_divided_by(evaluated) -> None:
    """An empty join is the shape a mis-keyed merge takes, and an analysis that returned a mean
    over zero recordings would report NaN as a finding."""
    from teb_vae.lag_attn_rws.eval import collect

    collection = collect.load_collection(evaluated["results_dir"])
    empty = sufficiency.join_oracle_scores(
        collection.per_sample,
        {
            "guid": ["not-a-recording"],
            "epoch": np.asarray([-1.0]),
            "nll_oracle_block": np.asarray([1.0]),
            "oracle_n_anchors": np.asarray([1.0]),
        },
    )

    assert empty.empty
    assert "nll_oracle_block" in empty.columns


# =============================================================================
# The run-level record
# =============================================================================
def test_the_step_record_marks_the_analysis_as_having_succeeded(evaluated) -> None:
    """A failure inside the wrapper is isolated and reported; this checks it was not one."""
    steps = json.loads(
        (Path(evaluated["results_dir"]) / "steps.json").read_text(encoding="utf-8")
    )
    record = next(step for step in steps if step["name"] == "sufficiency")

    assert record["ok"] is True, record.get("error")
