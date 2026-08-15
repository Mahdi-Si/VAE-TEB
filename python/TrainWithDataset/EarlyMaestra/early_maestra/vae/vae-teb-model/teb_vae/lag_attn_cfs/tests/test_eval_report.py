r"""The reporting core: one analysis raising must not discard the ten that already succeeded.

Three properties carry the mechanism, and each of them is one edit away from being silently lost:

**Ctrl-C must still work.** ``except Exception`` lets ``KeyboardInterrupt`` through because it
derives from ``BaseException``; a well-meant widening to ``except BaseException`` would turn an
interrupt into a "failed step" that the run then continues past, and nothing else would notice.

**The traceback, not ``str(exc)``.** On an unattended multi-hour run the traceback is the entire
debugging surface -- ``KeyError: 'mu_full'`` names none of the call sites that could produce it.

**The summary must be JSON a non-Python reader can parse.** ``json.dump`` emits the bare token
``NaN`` for a non-finite float, which round-trips through Python and is rejected by every strict
parser -- and NaN is an entirely ordinary result for a fully masked sample.

The mechanism is the shared one and is asserted by *identity* rather than re-tested: two copies of
it would be two chances for a value that survives a round trip in one package to fail the write in
the other.

What this package owns is the **content** of the three blocks, and that is where the rest of this
file looks. The headline registry is a promise that every path in it resolves on a run of this
model -- a number that is not registered is invisible to the acceptance gate and to the arm tables,
which read this block and nothing else -- so every path is walked here against a constructed
results dict, long before a real run exists to walk it against.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval import report as shared_report
from teb_vae.lag_attn_cfs.eval import report_seam

#: The top-level results blocks the headline registry reads. Written out, so a registered path
#: reaching into a block no analysis produces fails here rather than resolving to ``None`` on
#: every run -- which is indistinguishable in the artifact from an analysis that did not run.
EXPECTED_HEADLINE_BLOCKS = {
    "calibration",
    "controls",
    "coupling",
    "lag",
    "latent_health",
    "n_recordings",
    "n_samples",
    "perm_control",
    "readouts",
}

#: The two verdicts only this cell can have, and the eight it shares with the raw cells.
CELL_SPECIFIC_VERDICTS = ("coupling_exceeds_availability_clock", "anchor_geometry_intact")


def _stub_results() -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Build a results dict carrying a distinct value at every registered headline path.

    Returns:
        ``(results, expected)`` -- the nested block a run would produce, and the flat name-to-value
        mapping :func:`build_headline` must reproduce from it. Constructed from the registry rather
        than written out, so adding an entry does not need an edit here; what it proves is that
        every path is *reachable* -- that none is empty, none tries to index through a scalar
        another path already claimed, and no two names collide.
    """
    results: Dict[str, Any] = {}
    expected: Dict[str, float] = {}
    for index, (name, path) in enumerate(report_seam.HEADLINE_SCALARS):
        assert path, f"{name} has an empty path"
        assert name not in expected, f"{name} is registered twice"
        value = float(index + 1)
        node = results
        for key in path[:-1]:
            node = node.setdefault(key, {})
            assert isinstance(node, dict), f"{name}: {path} indexes through a non-block"
        assert path[-1] not in node, f"{name}: {path} collides with another entry"
        node[path[-1]] = value
        expected[name] = value
    return results, expected


# =================================================================================================
# The seam binds, and does not fork
# =================================================================================================
def test_the_seam_binds_the_shared_implementations_rather_than_copies() -> None:
    """Identity, not equality: a fork would pass every behavioural test below and still drift."""
    assert report_seam.json_safe is shared_report.json_safe
    assert report_seam.Report is shared_report.Report
    assert report_seam.StepRecord is shared_report.StepRecord
    assert report_seam.summarise_by_group is shared_report.summarise_by_group


def test_the_grouped_emitter_delegates_rather_than_reimplementing(monkeypatch) -> None:
    """The one seam entry that is not a bare binding, and the reason it must still not be a fork.

    It adds this package's cohort order and palette -- two presentation decisions -- and nothing
    else. Asserted by intercepting the shared function: what reaches it is the caller's own
    arguments plus exactly those two, so the skip rules, the counts and the record's shape stay the
    shared ones.
    """
    from teb_vae.lag_attn_cfs.eval import cohort, figures_seam

    seen = {}

    def _spy(frame, directory, **kwargs):
        seen.update({"frame": frame, "directory": directory, **kwargs})
        return {"intercepted": True}

    monkeypatch.setattr(shared_report, "emit_grouped_variants", _spy)
    result = report_seam.emit_grouped_variants("frame", "dir", value_columns=["pred_gap"])

    assert result == {"intercepted": True}
    assert (seen["frame"], seen["directory"]) == ("frame", "dir")
    assert seen["value_columns"] == ["pred_gap"]
    assert seen["group_palette"] is figures_seam.group_colors
    assert seen["order_groups"](["hie", "acidosis", "healthy"], "clinical_class") == (
        cohort.ordered_groups(["hie", "acidosis", "healthy"], "clinical_class")
    ) == ["healthy", "acidosis", "hie"]
    assert set(seen) == {"frame", "directory", "value_columns", "order_groups", "group_palette"}


# =================================================================================================
# Failure isolation
# =================================================================================================
def test_a_raising_step_is_captured_with_its_full_traceback() -> None:
    report = report_seam.Report()

    def failing() -> None:
        raise KeyError("mu_full")

    assert report.step("coupling", failing) is None

    record = report.steps[0]
    assert record.ok is False
    assert "KeyError" in (record.error or "")
    # The frame name, which only a formatted traceback carries -- str(exc) is just "'mu_full'".
    assert "in failing" in (record.traceback or "")


def test_a_failure_sets_the_exit_code_and_does_not_stop_later_steps() -> None:
    """The whole reason the wrapper exists: an eleventh analysis raising must not lose ten."""
    report = report_seam.Report()

    report.step("forecast", lambda: "fine")
    report.step("coupling", lambda: 1 / 0)
    report.step("lag_kl", lambda: "also fine")

    assert [record.ok for record in report.steps] == [True, False, True]
    assert report.exit_code() == 1
    assert [record.name for record in report.failed_steps] == ["coupling"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_propagates_rather_than_being_recorded_as_a_failed_step(interrupt) -> None:
    report = report_seam.Report()

    def interrupted() -> None:
        raise interrupt()

    with pytest.raises(interrupt):
        report.step("coupling", interrupted)

    assert report.steps == []


# =================================================================================================
# Serialisation
# =================================================================================================
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_float_becomes_null(value: float) -> None:
    assert report_seam.json_safe(value) is None


def test_a_summary_carrying_a_nan_is_strict_json(tmp_path) -> None:
    report = report_seam.Report()
    report.set("readouts", {"pred_gap": float("nan"), "kl_total": np.float32(0.5)})

    path = report.write(tmp_path)

    def _reject(name: str) -> None:
        raise AssertionError(f"summary.json carries the non-standard constant {name!r}")

    written = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    assert written["results"]["readouts"]["pred_gap"] is None
    assert written["results"]["readouts"]["kl_total"] == pytest.approx(0.5)


def test_the_torch_and_numpy_types_this_package_produces_become_plain_python() -> None:
    """Every readout starts life as a tensor, so the tensor branch is not an edge case here."""
    converted = report_seam.json_safe(
        {
            "lag_profile": torch.tensor([3.0, 4.0]),
            "kl_total": torch.tensor(2.5),
            "flag": np.bool_(True),
            "count": np.int64(3),
            "path": Path("a") / "b",
        }
    )

    assert converted["lag_profile"] == [3.0, 4.0]
    assert converted["kl_total"] == pytest.approx(2.5)
    # np.bool_ is checked before the int branch; otherwise True would serialise as 1.
    assert converted["flag"] is True
    assert converted["count"] == 3 and isinstance(converted["count"], int)
    assert isinstance(converted["path"], str)


def test_the_steps_heartbeat_is_rewritten_as_each_step_finishes(tmp_path) -> None:
    """A run killed outright leaves no summary at all, and on a multi-hour pass the question
    afterwards is which step it was inside."""
    report = report_seam.Report()
    report.step("forecast", lambda: "fine")
    report.step("coupling", lambda: 1 / 0)

    written = json.loads(
        report_seam.write_steps(report.steps, tmp_path).read_text(encoding="utf-8")
    )

    assert [record["name"] for record in written] == ["forecast", "coupling"]
    assert [record["status"] for record in written] == ["ok", "failed"]


# =================================================================================================
# The headline block
# =================================================================================================
def test_every_registered_headline_path_resolves() -> None:
    """A registry entry whose path never resolves is a number the acceptance gate silently reads as
    absent, which is indistinguishable from an analysis that did not run. Walked against a
    constructed results dict, so the registry is checkable before a run exists; a real run re-walks
    the same paths, which is where an entry that resolves only on a stub would fail."""
    results, expected = _stub_results()

    headline = report_seam.build_headline(results)

    for name, value in expected.items():
        assert headline[name] == value, name


def test_the_registry_reads_only_blocks_a_run_actually_produces() -> None:
    """The half a constructed stub cannot check by itself: the stub is *built* from the paths, so
    it satisfies any path at all. Pinning the top-level blocks is what makes a path into a block no
    analysis writes fail here rather than resolve to ``None`` on every run."""
    blocks = {path[0] for _, path in report_seam.HEADLINE_SCALARS}

    assert blocks == EXPECTED_HEADLINE_BLOCKS


def test_an_unresolved_headline_path_yields_none_rather_than_raising() -> None:
    """An analysis that failed or was skipped legitimately has no headline, and losing the whole
    block to it would be losing the numbers that did resolve."""
    headline = report_seam.build_headline({"readouts": {"mc_pred_gap": 2.0}})

    assert headline["pred_gap_mc_nats"] == pytest.approx(2.0)
    assert headline["kl_argmax_lag_step"] is None
    assert headline["verdict_source_specificity"] is None


def test_the_headline_carries_both_pred_gap_estimators_under_names_that_say_which() -> None:
    """The Monte Carlo marginalised score and the training-path single-draw score are different
    estimators of the same quantity, and a bare ``pred_gap`` leaves a reader to guess."""
    names = {name for name, _ in report_seam.HEADLINE_SCALARS}

    assert {"pred_gap_mc_nats", "pred_gap_train_path_nats"} <= names
    assert "pred_gap" not in names


def test_only_the_unfloored_kl_may_be_read_as_a_rate() -> None:
    """``source_conditioned_kl_train`` has free bits applied per dimension per step before summing,
    so it exceeds the raw value by construction and hides a collapsed source pathway. The shipped
    ``free_bits: 0.0`` makes the two coincide today, which is exactly why the distinction lives in
    code rather than in an observation."""
    leaves = {path[-1] for _, path in report_seam.HEADLINE_SCALARS}

    assert "source_conditioned_kl_raw" in leaves
    assert "source_conditioned_kl_train" not in leaves


def test_no_frequency_domain_entry_survived_the_fork() -> None:
    """``coherence`` is not ported: a stored coefficient is a modulus, so the phase the estimator
    needs was discarded before the value was written. An entry left behind would resolve to
    ``None`` on every run of this cell and read as an analysis that failed."""
    names = {name for name, _ in report_seam.HEADLINE_SCALARS}
    blocks = {path[0] for _, path in report_seam.HEADLINE_SCALARS}

    assert not any("coherence" in name for name in names)
    assert "coherence" not in blocks


def test_the_calibration_gain_is_registered_per_coefficient_rather_than_per_raw_sample() -> None:
    """There is no raw sample in this pipeline for a gain to be per, and the two denominators
    differ by a factor of three at the shipped geometry -- so a column carried across under the
    sibling's name would be silently non-comparable with the sibling's number."""
    names = {name for name, _ in report_seam.HEADLINE_SCALARS}

    assert "calibration_nll_gain_per_coefficient" in names
    assert "calibration_nll_gain_per_raw_sample" not in names
    assert not any("raw_sample" in name for name in names)


def test_the_pred_gap_convention_states_the_block_this_cell_actually_scores() -> None:
    """A reader of two runs has to know what a nat is per. Here a block is $15 \\times 98 = 1470$
    coefficients rather than 480 raw samples, and the percentage that divides by it is therefore
    budget-local -- two arms at two warm-up budgets divide by two different numbers."""
    convention = report_seam.PRED_GAP_CONVENTION

    assert "H*C_keep" in convention
    assert "1470" in convention
    assert "BUDGET-LOCAL" in convention
    assert "bpm" in convention and "no bpm anywhere in this pipeline" in convention
    # The raw cells' block size appears once, in the clause that says this is not it: a reader
    # arriving from a raw run has that number in mind and needs it contradicted, not omitted.
    assert convention.count("480") == 1
    assert "not a 480-sample raw window" in convention


def test_the_convention_travels_inside_the_artifact_rather_than_beside_it() -> None:
    assert report_seam.build_headline({})["pred_gap_convention"] == report_seam.PRED_GAP_CONVENTION


# =================================================================================================
# The verdict registry
# =================================================================================================
def test_the_two_verdicts_only_this_cell_can_have_are_promoted() -> None:
    """A verdict that is not promoted is one the acceptance gate and every arm table cannot see."""
    assert len(report_seam.HEADLINE_VERDICTS) == 10
    for name in CELL_SPECIFIC_VERDICTS:
        assert name in report_seam.HEADLINE_VERDICTS


def test_every_promoted_verdict_reaches_the_headline_under_its_own_key() -> None:
    verdicts = [
        {"name": name, "status": "PASS"} for name in report_seam.HEADLINE_VERDICTS
    ]

    headline = report_seam.build_headline({"verdicts": verdicts})

    for name in report_seam.HEADLINE_VERDICTS:
        assert headline[f"verdict_{name}"] == "PASS"


def test_the_promotion_list_is_the_readout_modules_registry() -> None:
    """``report_seam`` restates the names rather than importing them -- it must stay importable
    without ``torch`` -- so the two are pinned equal here instead of drifting apart quietly."""
    metrics = pytest.importorskip(
        "teb_vae.lag_attn_cfs.eval.metrics",
        reason="the readout module does not exist yet; this pin activates with it",
    )

    assert report_seam.HEADLINE_VERDICTS == metrics.PROMOTED_VERDICTS


# =================================================================================================
# The sanity block
# =================================================================================================
def test_the_sanity_block_carries_the_checks_this_cell_can_actually_evaluate() -> None:
    """The two cross-spectral checks are gone with the estimator they describe. Left in place they
    would be permanently INCONCLUSIVE, which reads as an analysis that failed rather than one that
    does not exist."""
    sanity = report_seam.build_sanity({}, {})

    assert set(sanity["checks"]) == {
        "kl_identity",
        "per_anchor_recombines",
        "argmax_lag",
        "lag_map_sums_to_kl",
        "per_head_kl_sums_to_kl",
        "per_file_counts",
        "classes_present",
        "target_not_truncated",
        "headline_finite",
    }
    for record in sanity["checks"].values():
        assert record["verdict"] in {"pass", "fail", report_seam.INCONCLUSIVE}


def test_a_violated_identity_is_recorded_as_failed() -> None:
    record = report_seam.check_kl_identity(
        {"latent_health": {"kl_total_nats": 1.0}, "readouts": {"source_conditioned_kl_raw": 2.0}}
    )

    assert record["verdict"] == "fail"
    assert record["abs_difference"] == pytest.approx(1.0)


def test_a_violated_check_warns_without_changing_the_exit_code(tmp_path) -> None:
    """The asymmetry is deliberate, and it is why an offline acceptance gate exists separately: a
    run whose every step succeeded can still be one nobody should quote a number from."""
    report = report_seam.Report()
    report.step("forecast", lambda: "fine")
    report.results.update(
        {
            "latent_health": {"kl_total_nats": 1.0},
            "readouts": {"source_conditioned_kl_raw": 2.0},
        }
    )

    report_seam.finalise(
        report, output_dir=tmp_path, analyses=["forecast"], eval_config={"caps": {}}
    )

    assert report.results["sanity"]["checks"]["kl_identity"]["verdict"] == "fail"
    assert report.results["sanity"]["warning"] is True
    assert report.exit_code() == 0


def test_a_per_anchor_table_that_does_not_recombine_is_caught() -> None:
    import pandas as pd

    per_sample = pd.DataFrame({"sample_index": [0, 1], "nll_full_block": [10.0, 20.0]})
    per_anchor = pd.DataFrame(
        # Sample 0's anchors average to 10.0 as its row says; sample 1's average to 15.0, not 20.
        {"sample_index": [0, 0, 1, 1], "nll_full_block": [9.0, 11.0, 10.0, 20.0]}
    )

    record = report_seam.check_per_anchor_recombines(per_sample, per_anchor)

    assert record["verdict"] == "fail"
    assert record["max_abs_difference"]["nll_full_block"] == pytest.approx(5.0)


def test_a_zero_anchor_segment_is_not_a_recombination_failure() -> None:
    """It is NaN on the sample table and absent from the anchor table -- the same exclusion seen
    from both sides, not a disagreement."""
    import pandas as pd

    per_sample = pd.DataFrame({"sample_index": [0, 1], "nll_full_block": [10.0, float("nan")]})
    per_anchor = pd.DataFrame({"sample_index": [0, 0], "nll_full_block": [9.0, 11.0]})

    assert report_seam.check_per_anchor_recombines(per_sample, per_anchor)["verdict"] == "pass"


def test_the_three_warm_up_tertiles_are_on_the_recombination_list() -> None:
    """They are a decomposition of ``pred_gap`` over the kept channels, so each has to average back
    per anchor like any other per-anchor column -- and their *sum* identity is the ``warmup``
    analysis's to check. A column the pass has not produced is skipped rather than raising."""
    assert {"pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi"} <= set(
        report_seam.RECOMBINED_COLUMNS
    )
    assert report_seam.RECOMBINED_COLUMNS["kld_per_t"] == "source_conditioned_kl_raw"


@pytest.mark.parametrize(
    "argmax, expected, reason",
    [
        (0, "fail", "the attribution never looks back and the lag window is inert"),
        (5, "fail", "the peak is against the window edge"),
        (3, "pass", ""),
    ],
)
def test_the_argmax_lag_is_judged_against_the_attainable_ceiling(argmax, expected, reason) -> None:
    r"""The ceiling is read from the per-lag anchor counts rather than taken as $L - 1$: a lag no
    anchor contributes to is not attainable. At this cell's anchor floor every lag is attainable at
    every scored anchor, which makes the correction inert -- but inert because the geometry says
    so, measured per run, rather than because this check stopped looking."""
    record = report_seam.check_argmax_lag(
        {
            "lag": {
                "kl_argmax_lag_step": argmax,
                # Lags 6 and 7 exist in the window but no anchor reaches them.
                "kl_lag_anchor_counts": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 0.0, 0.0],
            }
        }
    )

    assert record["verdict"] == expected
    assert record["attainable_lag_ceiling"] == 5
    if reason:
        assert reason in record["detail"]


def test_a_run_with_no_lag_summary_is_inconclusive_rather_than_failed() -> None:
    assert report_seam.check_argmax_lag({})["verdict"] == report_seam.INCONCLUSIVE


def test_the_derived_blocks_survive_a_builder_that_raises(tmp_path, monkeypatch) -> None:
    """``finalise`` runs after every analysis, so anything raising here would lose the entire run
    -- every result *and* every captured traceback -- to a failure in the bookkeeping."""

    def _explode(*args, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(report_seam, "build_headline", _explode)
    report = report_seam.Report()
    report.set("readouts", {"mc_pred_gap": 1.0})

    report_seam.finalise(report, output_dir=tmp_path, analyses=[], eval_config={"caps": {}})

    assert "error" in report.results["headline"]
    assert report.results["readouts"] == {"mc_pred_gap": 1.0}
    assert "sanity" in report.results
