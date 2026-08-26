r"""The reporting core: one analysis raising must not discard the ten that already succeeded.

Three properties carry this file, and each of them is one edit away from being silently lost.

**Ctrl-C must still work.** ``except Exception`` lets ``KeyboardInterrupt`` through because it
derives from ``BaseException``; a well-meant widening to ``except BaseException`` would turn an
interrupt into a "failed step" that the run then continues past, and nothing else would notice.

**The traceback, not ``str(exc)``.** On an unattended multi-hour run the traceback is the entire
debugging surface -- ``KeyError: 'mu_full'`` names none of the call sites that could produce it.

**The summary must be JSON a non-Python reader can parse.** ``json.dump`` emits the bare token
``NaN`` for a non-finite float, which round-trips through Python and is rejected by every strict
parser -- and NaN is an entirely ordinary result for a fully masked sample.

The serialiser is the shared one, and that is asserted by identity rather than by re-testing its
arithmetic: two copies of it would be two chances for a value that survives a round trip in one
package to fail the write in the other.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval import report as shared_report
from teb_vae.lag_attn_rws.eval import report_seam, run as run_module


# =============================================================================
# The seam binds, and does not fork
# =============================================================================
def test_the_seam_binds_the_shared_implementations_rather_than_copies() -> None:
    """Identity, not equality: a fork would pass every behavioural test below and still drift."""
    assert report_seam.json_safe is shared_report.json_safe
    assert report_seam.Report is shared_report.Report
    assert report_seam.StepRecord is shared_report.StepRecord
    assert report_seam.summarise_by_group is shared_report.summarise_by_group


def test_the_grouped_emitter_delegates_rather_than_reimplementing(monkeypatch) -> None:
    """The one seam entry that is not a bare binding, and the reason it must still not be a fork.

    It adds this package's cohort order and palette -- two presentation decisions the sibling does
    not make -- and nothing else. Asserted by intercepting the shared function: what reaches it is
    the caller's own arguments plus exactly those two, so the skip rules, the counts and the
    record's shape stay the shared ones. A reimplementation would pass every behavioural test in
    ``test_eval_grouped.py`` and drift from the sibling's on the first change to either.
    """
    from teb_vae.lag_attn_rws.eval import cohort, figures_seam

    seen = {}

    def _spy(frame, directory, **kwargs):
        seen.update({"frame": frame, "directory": directory, **kwargs})
        return {"intercepted": True}

    monkeypatch.setattr(shared_report, "emit_grouped_variants", _spy)
    result = report_seam.emit_grouped_variants("frame", "dir", value_columns=["pred_gap"])

    assert result == {"intercepted": True}
    assert (seen["frame"], seen["directory"]) == ("frame", "dir")
    assert seen["value_columns"] == ["pred_gap"]
    # The two additions, checked by what they *do* rather than by identity: the ordering is a
    # lambda over ``cohort.ordered_groups``, so only its result is comparable.
    assert seen["group_palette"] is figures_seam.group_colors
    assert seen["order_groups"](["healthy", "acidosis", "hie"], "clinical_class") == (
        cohort.ordered_groups(["healthy", "acidosis", "hie"], "clinical_class")
    ) == ["hie", "acidosis", "healthy"]
    assert set(seen) == {"frame", "directory", "value_columns", "order_groups", "group_palette"}


def test_the_runner_writes_its_summary_through_the_same_serialiser() -> None:
    """``run.py`` owned a second copy of this before the seam existed."""
    assert run_module.json_safe is report_seam.json_safe
    assert run_module.SUMMARY_FILENAME == report_seam.SUMMARY_FILENAME == "summary.json"


# =============================================================================
# Failure isolation
# =============================================================================
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
    report.step("lag", lambda: "also fine")

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


def test_a_successful_step_returns_its_value_and_its_elapsed_time() -> None:
    report = report_seam.Report()

    assert report.step("forecast", lambda value: value * 2, 21) == 42
    assert report.steps[0].ok is True
    assert report.steps[0].elapsed_s >= 0.0
    assert report.exit_code() == 0


# =============================================================================
# Serialisation
# =============================================================================
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_float_becomes_null(value: float) -> None:
    assert report_seam.json_safe(value) is None


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


def test_a_summary_carrying_a_nan_is_strict_json(tmp_path) -> None:
    report = report_seam.Report()
    report.set("readouts", {"pred_gap": float("nan"), "kl_total": np.float32(0.5)})

    path = report.write(tmp_path)

    def _reject(name: str) -> None:
        raise AssertionError(f"summary.json carries the non-standard constant {name!r}")

    written = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    assert written["results"]["readouts"]["pred_gap"] is None
    assert written["results"]["readouts"]["kl_total"] == pytest.approx(0.5)


def test_an_unexpected_type_is_recorded_rather_than_failing_the_write(tmp_path) -> None:
    """A write that raises at the end of a multi-hour run has produced nothing."""
    report = report_seam.Report()
    report.set("device", torch.device("cpu"))

    written = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))

    assert written["results"]["device"] == "cpu"


# =============================================================================
# The headline block
# =============================================================================
def test_an_unresolved_headline_path_yields_none_rather_than_raising() -> None:
    """An analysis that failed or was skipped legitimately has no headline, and losing the whole
    block to it would be losing the ten numbers that did resolve."""
    headline = report_seam.build_headline({"readouts": {"mc_pred_gap": 2.0}})

    assert headline["pred_gap_mc_nats"] == pytest.approx(2.0)
    assert headline["kl_argmax_lag_step"] is None
    assert headline["verdict_source_specificity"] is None


#: Headline names that exist only under a likelihood with a predictive distribution, for two
#: unrelated reasons that happen to share a precondition.
#:
#: The calibration census: under ``'mse'`` the decoder's log-variance head is never fitted, so it
#: is not computed at all -- and a number invented for it would be arithmetic over an untrained
#: tensor. The likelihood-space percentage: under ``'mse'`` a block score is a sum of squared
#: errors rather than a log-density, so exponentiating it yields no density ratio; the two
#: error-space percentages beside it have no such precondition and must still resolve.
_GAUSSIAN_NLL_ONLY_HEADLINE_NAMES = (
    "calibration_mean_standardised_sq",
    "calibration_pit_max_cdf_deviation",
    "calibration_nll_gain_per_raw_sample",
    "pred_gap_mc_likelihood_pct",
)


def test_every_headline_name_resolves_on_a_real_run(evaluated) -> None:
    """A registry entry whose path never resolves is a number the acceptance gate silently reads
    as absent, which is indistinguishable from an analysis that did not run.

    The likelihood-conditional entries are the one legitimate exception, and the exception is
    *conditional* rather than a hole in the guard: they resolve under ``gaussian_nll`` and cannot
    exist under ``mse``, which is what this fixture's checkpoint was trained under. Any other
    unresolved name still fails here.
    """
    summary = evaluated["summary"]
    headline = summary["results"]["headline"]
    likelihood = str(summary["results"].get("likelihood") or "")

    unresolved = sorted(
        name for name, _ in report_seam.HEADLINE_SCALARS if headline.get(name) is None
    )
    expected = (
        [] if likelihood == "gaussian_nll" else sorted(_GAUSSIAN_NLL_ONLY_HEADLINE_NAMES)
    )
    assert unresolved == expected


def test_every_promoted_verdict_reaches_the_headline(evaluated) -> None:
    headline = evaluated["summary"]["results"]["headline"]

    for name in report_seam.HEADLINE_VERDICTS:
        assert headline[f"verdict_{name}"] in {"PASS", "FAIL", "INCONCLUSIVE"}


def test_the_promotion_list_is_the_readout_modules_registry() -> None:
    """``report_seam`` restates the names rather than importing them -- it must stay importable
    without ``torch`` -- so the two are pinned equal here instead of drifting apart quietly."""
    from teb_vae.lag_attn_rws.eval import metrics

    assert report_seam.HEADLINE_VERDICTS == metrics.PROMOTED_VERDICTS


def test_the_headline_carries_both_pred_gap_estimators_under_names_that_say_which() -> None:
    """The Monte Carlo marginalised score and the training-path single-draw score are different
    estimators of the same quantity, and a bare ``pred_gap`` leaves a reader to guess."""
    names = {name for name, _ in report_seam.HEADLINE_SCALARS}

    assert {"pred_gap_mc_nats", "pred_gap_train_path_nats"} <= names
    assert "pred_gap" not in names


# =============================================================================
# The sanity block
# =============================================================================
def test_the_kl_identity_holds_on_a_real_run(evaluated) -> None:
    """The per-dimension spectrum decomposes the raw KL, so it must sum to it. Reduced per batch
    rather than per recording it does not, and nothing else in the output moves."""
    check = evaluated["summary"]["results"]["sanity"]["checks"]["kl_identity"]

    assert check["verdict"] == "pass", check["detail"]


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


def test_the_sanity_block_appears_in_every_summary(evaluated) -> None:
    sanity = evaluated["summary"]["results"]["sanity"]

    assert set(sanity) >= {"checks", "failed", "n_failed", "n_inconclusive", "warning"}
    assert set(sanity["checks"]) == {
        "kl_identity",
        "per_anchor_recombines",
        "argmax_lag",
        # The two structural lag identities, re-measured per run rather than inherited from the
        # model tests: a lag profile that does not sum to the KL decomposes nothing.
        "lag_map_sums_to_kl",
        "per_head_kl_sums_to_kl",
        # The cross-spectral estimator's own: an exact Parseval identity between the FFT and the
        # time domain, and a loose magnitude check that the spectral residual is the same size as
        # the forecast error it describes.
        "coherence_parseval",
        "coherence_detrended_share",
        "per_file_counts",
        "classes_present",
        "target_not_truncated",
        "headline_finite",
    }
    for record in sanity["checks"].values():
        assert record["verdict"] in {"pass", "fail", report_seam.INCONCLUSIVE}


def test_the_two_tables_recombine_on_a_real_run(evaluated) -> None:
    """The per-anchor rows must be the rows the per-sample columns were reduced over. If they are
    not, every analysis reading one table while quoting a headline from the other is describing a
    different population."""
    check = evaluated["summary"]["results"]["sanity"]["checks"]["per_anchor_recombines"]

    assert check["verdict"] == "pass", check["detail"]
    assert set(check["columns_checked"]) >= {"nll_full_block", "source_conditioned_kl_raw"}


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


@pytest.mark.parametrize(
    "argmax, expected, reason",
    [
        (0, "fail", "the attribution never looks back and the lag window is inert"),
        (5, "fail", "the peak is against the window edge"),
        (3, "pass", ""),
    ],
)
def test_the_argmax_lag_is_judged_against_the_attainable_ceiling(
    argmax, expected, reason
) -> None:
    r"""The ceiling is read from the per-lag anchor counts rather than taken as $L - 1$: a lag no
    anchor contributes to is not attainable, which at short sequences removes the window's top."""
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
