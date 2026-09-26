r"""This package's headline and sanity blocks, and the bookkeeping that must survive a failure.

The fail-soft step wrapper and the JSON serialiser are the shared ones and are tested by their
owner; what is checked here is what this package builds on them:

**The headline.** A registry entry that never resolves is a number the acceptance gate silently
reads as absent, so every name must resolve on a real run (the likelihood-conditional ones
excepted, conditionally); every promoted verdict must reach it; and no headline path may read the
floored KL, which exceeds the raw value by construction and hides a collapsed source pathway.

**The sanity block.** The KL identity, the per-anchor recombination and the argmax-lag ceiling are
each measured on a real run and caught on a planted violation. A violated check warns without
changing the exit code, and a builder that raises inside ``finalise`` must not lose the run.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn_rws.eval import report_seam


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


def test_no_headline_path_resolves_to_a_floored_kl() -> None:
    """Only the unfloored KL may be read as a rate: free bits are applied per dimension per step
    before summing, so the floored value exceeds the raw one by construction. The shipped
    ``free_bits: 0.0`` makes the two coincide today, which is why this is checked on the registry
    rather than observed on a run."""

    def floored(scalars):
        return [
            (name, path) for name, path in scalars
            if any("kl_train" in part or part.endswith("_train") for part in path)
        ]

    assert floored(report_seam.HEADLINE_SCALARS) == []
    # Non-vacuity: an entry pointed at the floored readout is caught.
    assert floored((("kl", ("readouts", "source_conditioned_kl_train")),))


def test_the_promotion_list_is_the_readout_modules_registry() -> None:
    """``report_seam`` restates the names rather than importing them -- it must stay importable
    without ``torch`` -- so the two are pinned equal here instead of drifting apart quietly."""
    from teb_vae.lag_attn_rws.eval import metrics

    assert report_seam.HEADLINE_VERDICTS == metrics.PROMOTED_VERDICTS


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
