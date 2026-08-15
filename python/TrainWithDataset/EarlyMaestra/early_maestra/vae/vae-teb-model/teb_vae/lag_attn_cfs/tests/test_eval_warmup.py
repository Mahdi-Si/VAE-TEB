r"""The warm-up staircase read as a decomposition, and the two guards that are FAIL-able.

Three properties, and each is here because it is the one thing that makes its readout mean
something:

* **The three tertiles are a decomposition.** They are only three parts of ``pred_gap`` if they
  sum back to it, and an implementation whose tertile assignment stopped tiling the kept channel
  axis would emit three plausible numbers that no other test in this suite could tell from three
  correct ones. The check is on the **worst** recording rather than on the mean, because the
  failure moves gap *between* the parts and is therefore zero-mean across them by construction.

* **The two geometry guards' expectations come from the run's own geometry.** $A_{\max}$ is
  $T_{\mathrm{valid}} - F$ arithmetic on numbers the checkpoint produced, so a legitimate arm --
  a longer horizon, a higher floor -- moves the expectation with the model rather than failing a
  guard written against the shipped $152$. Both arms are constructed here rather than assumed to
  behave.

* **A small source-lag warmth fraction is the expected finding.** The record says so, in the
  output rather than only in a docstring, because a reader who found a value near zero and no
  statement beside it would read a designed property of this cell as a defect.
"""
from __future__ import annotations

import types
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import warmup as warmup_analysis

from .conftest import causal_config

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}

#: The shipped geometry, restated rather than imported from the analysis: the point of the guard is
#: that the analysis derives these from a run's own record, so an assertion that read them from the
#: same place would compare a number with itself.
SHIPPED_T_VALID = 285
SHIPPED_FLOOR = 133
SHIPPED_ANCHORS = 152


def _per_sample(
    tertiles: Sequence[Sequence[float]],
    *,
    guids: Optional[Sequence[str]] = None,
    anchors: float = float(SHIPPED_ANCHORS),
    warm_frac: float = 1.0,
    total: Optional[Sequence[float]] = None,
    block_score: float = 1.0,
) -> pd.DataFrame:
    """A per-sample table carrying the three tertiles, their total and both guards.

    Args:
        tertiles: One ``(lo, mid, hi)`` triple per segment.
        guids: The recording each segment belongs to; two segments per recording by default.
        anchors: The decoded anchor count every segment reports.
        warm_frac: The warm target fraction every segment reports.
        total: ``pred_gap`` per segment. The sum of the triple by default, which is what a healthy
            collection pass produces.
        block_score: The block score magnitude the recomposition tolerance is scaled by. Unity by
            default so a stub's tolerance is the plain relative one; a real run's is of order
            $10^{3}$.

    Returns:
        The frame, with the cohort columns every per-recording reduction carries.
    """
    rows = [tuple(float(value) for value in triple) for triple in tertiles]
    identifiers = list(guids) if guids is not None else [
        f"REC{index // 2:02d}" for index in range(len(rows))
    ]
    return pd.DataFrame(
        {
            "guid": identifiers,
            "clinical_class": ["healthy"] * len(rows),
            "subgroup": ["healthy_bg"] * len(rows),
            "nll_base_block": [float(block_score)] * len(rows),
            "pred_gap": [sum(row) for row in rows] if total is None else list(total),
            "pred_gap_warm_lo": [row[0] for row in rows],
            "pred_gap_warm_mid": [row[1] for row in rows],
            "pred_gap_warm_hi": [row[2] for row in rows],
            "source_lag_warmth_frac_st": [0.08] * len(rows),
            "source_lag_warmth_frac_ph": [0.01] * len(rows),
            "anchors_per_sample": [float(anchors)] * len(rows),
            "target_warm_frac": [float(warm_frac)] * len(rows),
        }
    )


def _record(t_valid: int = SHIPPED_T_VALID, floor: int = SHIPPED_FLOOR) -> Dict[str, Any]:
    """The collection record's geometry block, as the pass writes it."""
    return {
        "geometry": {
            "t_valid": int(t_valid),
            "anchor_floor": int(floor),
            "anchors_per_sample": int(t_valid) - int(floor),
            "anchor_stride": 1,
            "horizon": 15,
        }
    }


def _context(per_sample: pd.DataFrame, record: Optional[Dict[str, Any]] = None,
             config: Optional[Dict[str, Any]] = None) -> AnalysisContext:
    """An analysis context with no task and no loader, as an offline re-run has."""
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record=record or _record(),
        retained={}, results={}, vectors={},
    )
    return AnalysisContext(collection=collection, config=config or {})


# =================================================================================================
# The decomposition
# =================================================================================================
def test_the_three_tertiles_recompose_to_pred_gap_per_recording() -> None:
    """The property that makes them three parts of one number rather than three readouts."""
    per_guid = warmup_analysis.per_recording_means(
        _per_sample([(0.1, 0.2, 0.3), (0.4, 0.0, -0.1), (0.2, 0.2, 0.2), (0.0, 0.1, 0.9)]),
        warmup_analysis.VALUE_COLUMNS,
    )

    record = warmup_analysis.recomposition(per_guid)

    assert record["holds"] is True
    assert record["max_rel_residual"] < 1e-6
    assert record["n_recordings"] == 2
    assert "pred_gap" in record["identity"]


def test_a_tertile_that_stopped_tiling_the_channel_axis_is_reported() -> None:
    """Non-vacuity for the check above. A gap of $1.0$ against a triple summing to $0.6$ is what a
    partition that dropped a third of the channels would produce, and every shape stays correct."""
    per_guid = warmup_analysis.per_recording_means(
        _per_sample([(0.1, 0.2, 0.3)] * 4, total=[1.0] * 4),
        warmup_analysis.VALUE_COLUMNS,
    )

    record = warmup_analysis.recomposition(per_guid)

    assert record["holds"] is False
    assert record["max_abs_residual"] == pytest.approx(0.4)


def test_the_tolerance_is_scaled_by_the_block_score_rather_than_by_the_gap() -> None:
    r"""**The property that keeps this check honest as a model improves.**

    ``pred_gap`` is a difference of two block scores of order $10^{3}$, so the float32 accumulation
    error it inherits belongs to *those* rather than to the small number between them. A tolerance
    relative to the gap would tighten without limit as the gap approached zero -- reporting a
    healthy decomposition as broken on exactly the runs that matter -- so the same residual is
    accepted against a real block score and refused against a unit one.
    """
    residual = 2.5e-4
    broken = _per_sample(
        [(0.1, 0.2, 0.3)] * 4, total=[0.6 - residual] * 4, block_score=1360.0
    )
    at_unit_scale = _per_sample([(0.1, 0.2, 0.3)] * 4, total=[0.6 - residual] * 4)

    accepted = warmup_analysis.recomposition(
        warmup_analysis.per_recording_means(broken, warmup_analysis.VALUE_COLUMNS)
    )
    refused = warmup_analysis.recomposition(
        warmup_analysis.per_recording_means(at_unit_scale, warmup_analysis.VALUE_COLUMNS)
    )

    assert accepted["holds"] is True
    assert refused["holds"] is False
    assert accepted["max_abs_residual"] == pytest.approx(refused["max_abs_residual"])
    assert accepted["scale_column"] == "nll_base_block"


def test_a_frame_that_measured_nothing_reports_the_identity_as_unchecked() -> None:
    """``None`` rather than ``True``: an identity nothing was available to check is not one that
    held, and the two are indistinguishable in a summary unless the record says which."""
    empty = _per_sample([(np.nan, np.nan, np.nan)] * 2, total=[np.nan] * 2)
    per_guid = warmup_analysis.per_recording_means(empty, warmup_analysis.VALUE_COLUMNS)

    record = warmup_analysis.recomposition(per_guid)

    assert record["holds"] is None
    assert record["n_recordings"] == 0


# =================================================================================================
# The two geometry guards
# =================================================================================================
def test_the_expected_anchor_count_is_derived_from_the_runs_own_geometry() -> None:
    r"""$A_{\max} = T_{\mathrm{valid}} - F$, computed from the two numbers rather than read out of
    the pass's own copy of the answer -- so the two derivations can disagree and the disagreement
    is reported instead of one number checking itself."""
    expected = warmup_analysis.expected_geometry(_record())

    assert expected["expected_anchors_per_sample"] == SHIPPED_ANCHORS
    assert expected["recorded_anchors_per_sample"] == SHIPPED_ANCHORS
    assert expected["geometry_self_consistent"] is True
    assert expected["expected_target_warm_frac"] == 1.0


@pytest.mark.parametrize(
    "t_valid, floor, anchors",
    [
        (SHIPPED_T_VALID, SHIPPED_FLOOR, SHIPPED_ANCHORS),
        # sweep_horizon_30: T - H = 300 - 30, the same floor.
        (270, SHIPPED_FLOOR, 137),
        # sweep_floor_150: the shipped horizon, a higher floor.
        (SHIPPED_T_VALID, 150, 135),
    ],
    ids=["shipped", "horizon_30", "floor_150"],
)
def test_a_legitimate_arm_passes_the_guard_without_an_edit(
    t_valid: int, floor: int, anchors: int
) -> None:
    """The reason the expectation is arithmetic rather than a literal: both shipped sweep arms move
    the anchor count, and a guard written against $152$ would fail exactly the runs it should
    pass."""
    per_guid = warmup_analysis.per_recording_means(
        _per_sample([(0.1, 0.1, 0.1)] * 4, anchors=float(anchors)),
        warmup_analysis.VALUE_COLUMNS,
    )

    guards = warmup_analysis.guard_record(
        per_guid, warmup_analysis.expected_geometry(_record(t_valid, floor))
    )

    assert guards["expected_anchors_per_sample"] == anchors
    assert guards["anchors_per_sample"] == pytest.approx(float(anchors))
    assert guards["anchors_per_sample_ok"] is True
    assert guards["target_warm_frac_ok"] is True


def test_a_run_at_the_training_tiling_fails_the_anchor_guard() -> None:
    r"""$152/15$ is what the *training* stride decodes per sample, so a pass that resolved its
    anchor geometry from the wrong stage reports it here rather than nowhere."""
    per_guid = warmup_analysis.per_recording_means(
        _per_sample([(0.1, 0.1, 0.1)] * 4, anchors=11.0), warmup_analysis.VALUE_COLUMNS
    )

    guards = warmup_analysis.guard_record(per_guid, warmup_analysis.expected_geometry(_record()))

    assert guards["anchors_per_sample_ok"] is False
    assert guards["anchors_per_sample"] == pytest.approx(11.0)
    assert guards["expected_anchors_per_sample"] == SHIPPED_ANCHORS


def test_a_target_axis_that_was_not_fully_warm_fails_its_guard() -> None:
    """Below $1.0$ the objective scored assumed pre-recording history as signal, on coefficients
    normalised with constants that excluded exactly that region -- with every shape correct."""
    per_guid = warmup_analysis.per_recording_means(
        _per_sample([(0.1, 0.1, 0.1)] * 4, warm_frac=0.97), warmup_analysis.VALUE_COLUMNS
    )

    guards = warmup_analysis.guard_record(per_guid, warmup_analysis.expected_geometry(_record()))

    assert guards["target_warm_frac_ok"] is False
    assert guards["anchors_per_sample_ok"] is True


def test_a_record_with_no_geometry_leaves_the_count_unevaluated_rather_than_satisfied() -> None:
    """An offline re-run against a directory whose record predates the geometry block reaches here,
    and an unevaluated criterion reported as satisfied is worse than one not evaluated."""
    per_guid = warmup_analysis.per_recording_means(
        _per_sample([(0.1, 0.1, 0.1)] * 4), warmup_analysis.VALUE_COLUMNS
    )

    guards = warmup_analysis.guard_record(per_guid, warmup_analysis.expected_geometry({}))

    assert guards["anchors_per_sample_ok"] is None
    assert guards["expected_anchors_per_sample"] is None
    # The warm fraction half stands on its own: it is checked against an exact constant rather
    # than against anything the record has to supply.
    assert guards["target_warm_frac_ok"] is True


def test_the_extremes_travel_beside_the_mean_of_each_guard() -> None:
    """Both quantities are identical on every sample of a healthy run, so a mean sitting on its
    expectation while the minimum does not is a subset of the pass having decoded a different
    anchor set -- which a mean alone averages away."""
    frame = _per_sample([(0.1, 0.1, 0.1)] * 4)
    frame.loc[0, "anchors_per_sample"] = 100.0
    per_guid = warmup_analysis.per_recording_means(frame, warmup_analysis.VALUE_COLUMNS)

    guards = warmup_analysis.guard_record(per_guid, warmup_analysis.expected_geometry(_record()))

    assert guards["anchors_per_sample_min"] < guards["anchors_per_sample_max"]
    assert guards["anchors_per_sample_ok"] is False


# =================================================================================================
# The analysis
# =================================================================================================
def test_the_analysis_writes_both_tables_and_declares_its_grouped_frame(tmp_path) -> None:
    result = warmup_analysis.run_warmup_analysis(
        _context(_per_sample([(0.1, 0.2, 0.3)] * 4)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / warmup_analysis.ANALYSIS_DIRNAME
    assert (directory / warmup_analysis.PER_RECORDING_FILENAME).is_file()
    assert (directory / warmup_analysis.SUMMARY_FILENAME).is_file()
    assert set(REQUIRED_RESULT_KEYS) <= set(result)
    assert result["n_samples"] == 4
    assert result["composition"]["n_recordings"] == 2
    assert result["recomposition"]["holds"] is True

    entry = result["grouped_frames"][0]
    assert entry["path"] == (
        f"{warmup_analysis.ANALYSIS_DIRNAME}/{warmup_analysis.PER_RECORDING_FILENAME}"
    )
    # The three tertiles and the two warmth fractions, and not the two guards: a grouped figure of
    # a quantity that is constant on every recording is a figure of nothing.
    assert set(entry["value_columns"]) == set(warmup_analysis.GROUPED_METRICS)
    assert "anchors_per_sample" not in entry["value_columns"]


def test_the_per_recording_frame_is_the_one_the_cross_cohort_analysis_reads(tmp_path) -> None:
    """The filename and the three column names are a contract with that analysis's source table,
    not a local choice: it reads this frame off disk by name."""
    from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup

    warmup_analysis.run_warmup_analysis(
        _context(_per_sample([(0.1, 0.2, 0.3)] * 4)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )
    written = pd.read_csv(
        tmp_path / warmup_analysis.ANALYSIS_DIRNAME / warmup_analysis.PER_RECORDING_FILENAME
    )

    wanted = [
        source for source in cross_subgroup.METRIC_SOURCES
        if source.analysis == warmup_analysis.ANALYSIS_DIRNAME
    ]
    assert wanted, "the cross-cohort analysis registers no metric from this one"
    for source in wanted:
        assert source.filename == warmup_analysis.PER_RECORDING_FILENAME
        assert source.column in written.columns
    # And the cohort axes it resolves on.
    assert {"clinical_class", "subgroup"} <= set(written.columns)


def test_the_headline_block_is_flat_and_carries_the_five_comparable_readouts(tmp_path) -> None:
    """Flat because the binding's headline registry digs into it by key; five because the two
    geometry guards travel in their own block, beside the expectation they are judged against."""
    result = warmup_analysis.run_warmup_analysis(
        _context(_per_sample([(0.1, 0.2, 0.3)] * 4)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    headline = result["headline"]
    assert set(headline) == {
        "pred_gap_warm_lo_nats", "pred_gap_warm_mid_nats", "pred_gap_warm_hi_nats",
        "source_lag_warmth_frac_st", "source_lag_warmth_frac_ph",
    }
    assert headline["pred_gap_warm_lo_nats"] == pytest.approx(0.1)
    assert all(isinstance(value, float) for value in headline.values())


def test_the_record_states_that_a_small_warmth_fraction_is_expected(tmp_path) -> None:
    """In the output rather than only in a docstring: a reader who found a value near zero and no
    statement beside it would read a designed property of this cell as a defect."""
    result = warmup_analysis.run_warmup_analysis(
        _context(_per_sample([(0.1, 0.2, 0.3)] * 4)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    note = result["source_lag_warmth_note"]
    assert "expected" in note
    assert "DESIGN.md" in note, "the argument is cited rather than restated"


def test_every_metric_row_names_its_unit_and_what_it_means(tmp_path) -> None:
    """Two of the eight are fractions and one is a count; a table of eight means in one unnamed
    unit is a table a reader has to guess at."""
    result = warmup_analysis.run_warmup_analysis(
        _context(_per_sample([(0.1, 0.2, 0.3)] * 4)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    units = {row["metric"]: row["unit"] for row in result["metrics"]}
    assert units["pred_gap_warm_lo"] == "nats per anchor"
    assert units["source_lag_warmth_frac_st"] == "fraction of attention mass"
    assert units["anchors_per_sample"] == "anchors"
    assert all(row["meaning"].strip() for row in result["metrics"])


# =================================================================================================
# The two figures, and the skip that is not a failure
# =================================================================================================
def test_a_configuration_with_no_budget_records_a_skip_rather_than_raising(tmp_path) -> None:
    """A run whose every readout succeeded must not be reported as failed because a figure's input
    was unavailable -- and every two-sided run in this family sets no budget at all."""
    record = warmup_analysis.write_budget_figures({"model_config": {"VAE_model": {}}}, tmp_path)

    assert record["skipped"] is True
    assert "causal_warmup_budget_steps" in record["reason"]


def test_an_unreadable_shard_records_the_reason_rather_than_raising(tmp_path) -> None:
    """The offline re-run case: a finished directory copied to another machine still re-runs every
    analysis, and the shards it was collected from are not necessarily there."""
    config = causal_config()
    config["dataset_config"]["vae_train_datasets"] = [str(tmp_path / "absent.hdf5")]
    config["dataset_config"]["vae_test_datasets"] = [str(tmp_path / "absent.hdf5")]

    record = warmup_analysis.write_budget_figures(config, tmp_path)

    assert record["skipped"] is True
    assert record["reason"]


@pytest.mark.slow
def test_the_analysis_runs_and_its_guards_pass_on_a_real_run(collected_run) -> None:
    """Read off a pass that actually ran it: an analysis can satisfy every stub above and emit
    nothing on the inputs a real pass hands it.

    The guards are asserted to **pass** rather than merely to be reported, because they are exact
    structural numbers: the fixture's fit and its evaluation resolve their anchor geometry from one
    configuration, so anything but a pass means the two disagreed.
    """
    block = collected_run["summary"]["results"]["warmup"]

    assert block["n_samples"] == collected_run["summary"]["results"]["n_samples"]
    assert block["recomposition"]["holds"] is True, block["recomposition"]
    guards = block["geometry_guards"]
    assert guards["geometry_self_consistent"] is True
    assert guards["anchors_per_sample_ok"] is True, guards
    assert guards["target_warm_frac_ok"] is True, guards
    assert guards["target_warm_frac"] == pytest.approx(1.0)
    # Derived rather than compared against a literal: the fixture trains at the tiny geometry.
    assert guards["anchors_per_sample"] == pytest.approx(
        float(guards["expected_anchors_per_sample"])
    )
    assert all(value is not None for value in block["headline"].values())


@pytest.mark.slow
def test_the_grouped_variants_and_the_budget_figures_land_on_a_real_run(collected_run) -> None:
    """The runner's fan-out over the declared frame, and the two figures the modules that already
    draw them produce -- both reached only on a pass with a real configuration behind it."""
    results_dir = collected_run["results_dir"]
    block = collected_run["summary"]["results"]["warmup"]

    for name in block["files"]:
        assert (results_dir / warmup_analysis.ANALYSIS_DIRNAME / name).is_file(), name
    assert block["budget"]["skipped"] is False, block["budget"]
    grouped = block.get("grouped") or {}
    assert grouped, "the runner emitted no by-cohort variant over the declared frame"


def test_the_budget_figures_are_drawn_from_the_resolver_the_driver_itself_calls(tmp_path) -> None:
    """Against the committed causal fixture, so the numbers in the record are the ones the training
    driver resolves rather than a second derivation of them."""
    record = warmup_analysis.write_budget_figures(causal_config(), tmp_path)

    assert record["skipped"] is False, record
    assert record["budget_steps"] == 134
    assert record["target_kept_width"] == 98
    assert record["target_declared_width"] == 102
    assert record["target_dropped_index"] == [32, 33, 34, 35]
    assert record["target_dropped_warmup_steps"] == [162, 194, 233, 278]
    # The survivors' own maximum, which is what the anchor floor must clear -- not the threshold.
    assert record["realised_max_warmup_steps"] == 134
    assert record["source_kept_width"] == record["source_declared_width"] == 51
    for name in record["files"]:
        assert (tmp_path / name).is_file(), name
