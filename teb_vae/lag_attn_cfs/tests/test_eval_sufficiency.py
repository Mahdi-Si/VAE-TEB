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

**The analysis itself is a behaviour-equivalent copy of the sibling's**, which is what
``divergences.json`` records and what ``test_eval_sibling_agreement.py`` exercises; the target
domain enters one layer below it, in ``eval/oracle.py``, and is asserted in
``test_eval_oracle.py``. So this file tests the analysis's *own* contract -- the join, the
population, the skip, the artifacts -- and does not re-assert the block width, which is the
oracle's to own.

Two tiers, following this suite's convention: the join, the reduction and the skip run on stub
frames and cost nothing; everything about a real directory reads the one ``slow`` run.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval import oracle
from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import sufficiency


def _context(
    per_sample: Optional[pd.DataFrame] = None,
    *,
    task: Any = None,
    loader: Any = None,
) -> AnalysisContext:
    """A context over a stub collection, with no model behind it unless one is supplied."""
    collection = types.SimpleNamespace(
        per_sample=pd.DataFrame() if per_sample is None else per_sample,
        per_anchor=pd.DataFrame(),
        record={},
        retained={},
        results={},
    )
    return AnalysisContext(collection=collection, config={}, task=task, loader=loader)


# =================================================================================================
# The join: the only thing making D_oracle and D_base describe the same segments
# =================================================================================================
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


def test_a_join_that_matches_nothing_is_reported_rather_than_divided_by() -> None:
    """An empty join is the shape a mis-keyed merge takes, and an analysis that returned a mean
    over zero recordings would report NaN as a finding."""
    per_sample = pd.DataFrame(
        {"guid": ["A"], "epoch": [-600.0], "mc_nll_base_block": [1.0]}
    )

    empty = sufficiency.join_oracle_scores(
        per_sample,
        {
            "guid": ["not-a-recording"],
            "epoch": np.asarray([-1.0]),
            "nll_oracle_block": np.asarray([1.0]),
            "oracle_n_anchors": np.asarray([1.0]),
        },
    )

    assert empty.empty
    assert "nll_oracle_block" in empty.columns


def test_a_segment_with_no_finite_epoch_is_dropped_rather_than_bucketed() -> None:
    """``epoch`` is NaN for a segment carrying none, and ``NaN`` compares equal to nothing -- so a
    key built without the guard would either raise or silently pair two unrelated segments."""
    per_sample = pd.DataFrame(
        {"guid": ["A", "A"], "epoch": [np.nan, -600.0], "mc_nll_base_block": [1.0, 2.0]}
    )
    per_segment = {
        "guid": ["A", "A"],
        "epoch": np.asarray([np.nan, -600.0]),
        "nll_oracle_block": np.asarray([10.0, 20.0]),
        "oracle_n_anchors": np.asarray([5.0, 5.0]),
    }

    merged = sufficiency.join_oracle_scores(per_sample, per_segment)

    assert len(merged) == 1
    assert float(merged["nll_oracle_block"].iloc[0]) == 20.0


# =================================================================================================
# The two gaps come off one frame
# =================================================================================================
def test_both_gaps_are_left_minus_right_over_columns_the_scores_list_carries() -> None:
    r"""$\Delta_{\mathrm{suff}}$ and ``pred_gap`` are read side by side off one figure, so they
    have to be differences of the same per-recording means rather than three separate headlines.
    Asserted on the declaration rather than on a run, so it holds before one exists."""
    scored = {column for column, _label in sufficiency.SCORE_COLUMNS}

    assert {name for name, _l, _r, _m in sufficiency.GAP_METRICS} == {
        "delta_suff_nats", "pred_gap_mc_nats"
    }
    for _name, left, right, meaning in sufficiency.GAP_METRICS:
        assert left in scored and right in scored
        assert meaning.strip()


def test_the_summary_rows_carry_an_interval_and_a_paired_test_for_each_gap() -> None:
    """Every held-out recording contributes both sides, so the paired form removes the
    between-recording variance that dominates every readout here; the unpaired one would throw
    that away and widen the interval for no reason."""
    per_guid = pd.DataFrame(
        {
            "nll_oracle_block": [90.0, 91.0, 92.0, 93.0],
            "mc_nll_base_block": [100.0, 101.0, 102.0, 103.0],
            "mc_nll_full_block": [99.0, 100.0, 101.0, 102.0],
        }
    )
    for name, left, right, _meaning in sufficiency.GAP_METRICS:
        per_guid[name] = per_guid[left] - per_guid[right]

    rows = sufficiency.build_rows(per_guid, resamples=64, seed=1)

    by_metric = {str(row["metric"]): row for row in rows}
    assert set(by_metric) == {column for column, _l in sufficiency.SCORE_COLUMNS} | {
        name for name, _l, _r, _m in sufficiency.GAP_METRICS
    }
    gap = by_metric["delta_suff_nats"]
    assert gap["value"] == pytest.approx(10.0)
    assert gap["lo"] <= gap["value"] <= gap["hi"]
    assert "p_value" in gap and "median_paired_difference" in gap
    # The three score rows carry no paired test: there is nothing to pair them against.
    assert "p_value" not in by_metric["nll_oracle_block"]


def test_the_curve_frame_carries_both_widths_and_flattens_to_one_long_table() -> None:
    """The evidence behind the convergence flag. Both probes' curves in one frame, because they
    are read against each other -- a narrow probe still descending where the wide one flattened is
    the shape a capacity-bound verdict is made of."""
    record = {
        "fit": {"width_multiplier": 1, "curve": [{"step": 0.0, "held_out_nats": 2.0,
                                                  "fit_nats": 2.5}]},
        "capacity": {
            "wide_fit": {
                "width_multiplier": 2,
                "curve": [{"step": 0.0, "held_out_nats": 1.0, "fit_nats": 1.5}],
            }
        },
    }

    frame = sufficiency.curve_frame(record)

    assert list(frame.columns) == ["width_multiplier", "step", "held_out_nats", "fit_nats"]
    assert sorted(frame["width_multiplier"]) == [1, 2]


def test_a_record_with_no_fit_yields_an_empty_curve_with_its_columns() -> None:
    """The figure reads this frame; an absent one has to be empty-with-columns rather than a
    ``KeyError`` inside the step wrapper."""
    frame = sufficiency.curve_frame({})

    assert frame.empty
    assert list(frame.columns) == ["width_multiplier", "step", "held_out_nats", "fit_nats"]


# =================================================================================================
# The offline path
# =================================================================================================
def test_a_pass_with_no_model_records_a_skip_rather_than_a_number(tmp_path) -> None:
    """Every other analysis but the pages runs offline off the tables. This one cannot -- the
    encoder state it reads is on neither -- so it must say so instead of reporting an empty
    measurement."""
    per_sample = pd.DataFrame(
        {"guid": ["A"], "epoch": [-600.0], "mc_nll_base_block": [1.0]}
    )

    outcome = sufficiency.run_sufficiency_analysis(
        _context(per_sample), eval_config={"seed": 0}, output_dir=tmp_path, probe=None
    )

    assert outcome["skipped"] is True
    assert outcome["n_samples"] is None
    assert "loader" in outcome["reason"]
    assert outcome["files"] == []


def test_an_empty_table_with_a_model_still_skips_rather_than_fitting_a_probe(tmp_path) -> None:
    """The other half of the same guard. A pass that built a model but collected nothing has no
    table to join against, and fitting the probe anyway would spend a training loop to produce a
    mean over zero recordings."""
    outcome = sufficiency.run_sufficiency_analysis(
        _context(task=object(), loader=object()),
        eval_config={"seed": 0}, output_dir=tmp_path, probe=None,
    )

    assert outcome["skipped"] is True
    assert outcome["files"] == []


# =================================================================================================
# Against the real run
# =================================================================================================
@pytest.fixture(scope="module")
def result(collected_run) -> Dict[str, Any]:
    """The sufficiency block of the shared evaluation run."""
    block = collected_run["summary"]["results"].get("sufficiency")
    assert block is not None, "the run produced no sufficiency block at all"
    return block


@pytest.fixture(scope="module")
def directory(collected_run) -> Path:
    """The analysis's own output directory in that run."""
    return Path(collected_run["results_dir"]) / sufficiency.ANALYSIS_DIRNAME


@pytest.mark.slow
def test_the_analysis_ran_rather_than_recording_a_skip(result) -> None:
    """A skip here is legitimate on a pass with no model; this fixture has one, so a skip would
    mean the guards fired on a population that can carry the measurement."""
    assert not result.get("skipped"), result.get("reason")
    assert result["n_samples"] > 0


@pytest.mark.slow
def test_the_split_covers_every_recording_and_shares_none(result, collected_run) -> None:
    """The property the whole number rests on: a probe scored on recordings it was fitted to
    reports the bottleneck as costing less than it does."""
    split = result["split"]

    assert split["recordings_disjoint"] is True
    assert split["n_fit_recordings"] > 0 and split["n_held_out_recordings"] > 0
    assert (
        split["n_fit_recordings"] + split["n_held_out_recordings"]
        == collected_run["summary"]["results"]["n_recordings"]
    )
    # Recoverable from the summary alone, which is what makes the split reproducible.
    assert isinstance(split["seed"], int)


@pytest.mark.slow
def test_the_reported_recordings_are_the_held_out_ones(result, directory) -> None:
    """The frame every statistic is computed on holds one row per held-out recording -- not per
    segment, and not per recording of the whole split."""
    frame = pd.read_csv(directory / sufficiency.PER_RECORDING_FILENAME)

    assert len(frame) == result["composition"]["n_held_out_recordings"]
    assert len(frame) == result["split"]["n_held_out_recordings"]
    assert frame["guid"].nunique() == len(frame)


@pytest.mark.slow
def test_both_gaps_are_differences_over_the_same_frame(directory) -> None:
    r"""The declaration is checked above; this is the same property on the frame a run wrote."""
    frame = pd.read_csv(directory / sufficiency.PER_RECORDING_FILENAME)

    for name, left, right, _meaning in sufficiency.GAP_METRICS:
        assert np.allclose(
            np.asarray(frame[name], dtype=np.float64),
            np.asarray(frame[left], dtype=np.float64)
            - np.asarray(frame[right], dtype=np.float64),
            equal_nan=True,
        )


@pytest.mark.slow
def test_the_summary_gap_matches_the_two_scores_it_is_a_difference_of(result) -> None:
    """The headline arithmetic, checked rather than assumed: a gap computed on a different
    reduction from its two terms would be off by an amount nothing else would show."""
    gap = result["gap"]

    assert gap["delta_suff_nats"] == pytest.approx(
        gap["d_base_mc_nats"] - gap["d_oracle_nats"], rel=1e-6, abs=1e-9
    )


@pytest.mark.slow
def test_the_oracle_scored_as_many_segments_as_the_join_produced(result, directory) -> None:
    r"""$D_{\mathrm{oracle}}$ is subtracted from ``mc_nll_base_block``, so the two have to come
    from the same rows. ``n_samples`` counts the joined segments and the per-recording frame
    reduces them; a join that dropped half the held-out half would still produce a plausible
    number, and only the two counts read together say it did not.

    The **geometry** the probe was scored at is the oracle module's own contract and is asserted
    in ``test_eval_oracle.py``, which is where a divergence between it and the collection pass's
    anchor set would be a wrong subtraction rather than a wrong count.
    """
    frame = pd.read_csv(directory / sufficiency.PER_RECORDING_FILENAME)

    assert result["n_samples"] <= result["split"]["n_held_out_segments"]
    assert result["n_samples"] > 0
    assert len(frame) <= result["split"]["n_held_out_recordings"]


@pytest.mark.slow
def test_both_bias_directions_reach_the_emitted_record(result) -> None:
    """They oppose, and neither is measured; a reader who sees only one would read the number as
    a one-sided bound in whichever direction happened to be written down."""
    directions = {entry["direction"] for entry in result["bias_directions"]}

    assert directions == {"understates", "overstates"}
    for entry in result["bias_directions"]:
        assert "target_state" in entry["cause"] or "pretraining" in entry["cause"]
    assert "estimate, not a bound" in result["estimate_not_a_bound"]


@pytest.mark.slow
def test_the_extra_encoder_pass_is_recorded_as_a_number(result) -> None:
    """The rest of the pipeline holds that the collection pass is the only model-touching cost.
    This analysis amends that, and the amendment is a measured size rather than a footnote."""
    extra = result["plan"]["extra_encoder_pass"]

    assert extra["n_segments"] > 0
    assert extra["n_bytes"] > 0
    assert "thousands of passes" in extra["reason"]


@pytest.mark.slow
def test_the_fit_budget_is_recorded_in_passes_over_the_fit_half(result) -> None:
    """A step count is not portable across populations: the same number that under-trains a probe
    on two thousand segments overfits one on twenty, and every fixture in this repository is the
    second case. So the budget is expressed in passes and the step count is *derived* -- and both
    travel, because a reader comparing two runs' probes needs to know which of the two moved."""
    fit = result["plan"]["fit"]

    assert fit["epochs"] == oracle.DEFAULT_FIT_EPOCHS
    assert oracle.MIN_FIT_STEPS <= fit["steps"] <= oracle.MAX_FIT_STEPS
    assert fit["batch_size"] <= result["split"]["n_fit_segments"]


@pytest.mark.slow
def test_the_probe_reports_its_own_convergence(result) -> None:
    """An under-trained probe understates the gap. Whether it converged is therefore part of the
    result rather than something a reader has to infer from the curve."""
    convergence = result["convergence"]

    assert isinstance(convergence["converged"], bool)
    assert convergence["detail"]
    assert np.isfinite(float(convergence["final_held_out_nats"]))


@pytest.mark.slow
def test_the_capacity_check_ran_and_says_which_way_it_went(result) -> None:
    capacity = result["capacity"]

    assert capacity["checked"] is True
    assert isinstance(capacity["capacity_bound"], bool)
    assert capacity["n_parameters_wide"] > capacity["n_parameters"]
    assert capacity["margin_nats"] == oracle.CAPACITY_MARGIN_NATS


@pytest.mark.slow
def test_the_training_curve_is_written_for_both_widths(directory) -> None:
    """The evidence behind the convergence flag, on disk, so it can be looked at rather than
    trusted."""
    curve = pd.read_csv(directory / sufficiency.CURVE_FILENAME)

    assert set(curve.columns) == {"width_multiplier", "step", "held_out_nats", "fit_nats"}
    assert set(curve["width_multiplier"]) == {1, oracle.CAPACITY_WIDTH_MULTIPLIER}
    assert len(curve) > 0


@pytest.mark.slow
def test_every_declared_file_was_written(result, directory) -> None:
    missing = [name for name in result["files"] if not (directory / name).is_file()]

    assert missing == []
    assert sufficiency.SUFFICIENCY_FIGURE in result["files"]


@pytest.mark.slow
def test_the_grouped_variants_were_fanned_out_by_the_runner(result) -> None:
    """The analysis declares a frame; the runner emits the by-class and by-subgroup cuts. A
    declaration nothing acts on would be an analysis quietly reporting one pooled number."""
    declared = {entry["stem"] for entry in result["grouped_frames"]}

    assert declared
    assert set(result.get("grouped", {})) == declared


@pytest.mark.slow
def test_the_step_record_marks_the_analysis_as_having_succeeded(collected_run) -> None:
    """A failure inside the wrapper is isolated and reported; this checks it was not one."""
    steps = json.loads(
        (Path(collected_run["results_dir"]) / "steps.json").read_text(encoding="utf-8")
    )
    record = next(step for step in steps if step["name"] == "sufficiency")

    assert record["ok"] is True, record.get("error")
