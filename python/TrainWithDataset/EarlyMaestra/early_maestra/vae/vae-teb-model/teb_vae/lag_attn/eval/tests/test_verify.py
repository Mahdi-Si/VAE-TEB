r"""The run-acceptance checker: each criterion's pass, fail and inconclusive path.

Every criterion is exercised in all three states, because ``INCONCLUSIVE`` is the one that
matters most and is the easiest to get wrong. A criterion that silently *passes* when the run
did not carry what it needs turns the whole verification into a formality -- and that is exactly
the failure mode a pre-registered checklist exists to prevent.

The last test drives the checker against a real smoke run, so the key paths the criteria dig for
are the ones the pipeline actually writes rather than the ones this file assumes.
"""
from __future__ import annotations

import json

import pytest

from teb_vae.lag_attn.eval import run as run_module
from teb_vae.lag_attn.eval import verify as verify_module
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG


def _summary(**results) -> dict:
    """A summary blob carrying whatever the test needs and nothing else."""
    return {"exit_code": 0, "failed": [], "results": results}


# ---------------------------------------------------------------------------
# Individual criteria
# ---------------------------------------------------------------------------
def test_exit_code():
    assert verify_module.check_exit_code(
        {"exit_code": 0, "failed": []}
    )["verdict"] == verify_module.PASS
    assert verify_module.check_exit_code(
        {"exit_code": 1, "failed": ["te_lag"]}
    )["verdict"] == verify_module.FAIL
    assert verify_module.check_exit_code({})["verdict"] == verify_module.INCONCLUSIVE


def test_per_file_counts():
    assert verify_module.check_per_file_counts(
        _summary(probe={"per_file": {"a.hdf5": 10, "b.hdf5": 3}})
    )["verdict"] == verify_module.PASS
    failed = verify_module.check_per_file_counts(
        _summary(probe={"per_file": {"a.hdf5": 10, "b.hdf5": 0}})
    )
    assert failed["verdict"] == verify_module.FAIL and "b.hdf5" in failed["detail"]
    assert verify_module.check_per_file_counts(
        _summary()
    )["verdict"] == verify_module.INCONCLUSIVE


def test_weights_loaded():
    assert verify_module.check_weights_loaded(
        _summary(preflight={"checks": {"weights_loaded": {"passed": True}}})
    )["verdict"] == verify_module.PASS
    assert verify_module.check_weights_loaded(
        _summary(preflight={"checks": {"weights_loaded": {"passed": False}}})
    )["verdict"] == verify_module.FAIL
    assert verify_module.check_weights_loaded(
        _summary()
    )["verdict"] == verify_module.INCONCLUSIVE


def test_uplift_needs_a_majority_and_a_positive_mean():
    assert verify_module.check_uplift(
        _summary(uplift={"positive_fraction": 0.81}, headline={"uplift_rel": 0.04})
    )["verdict"] == verify_module.PASS
    # A majority that helps, but a mean that does not: one large regression can do this.
    assert verify_module.check_uplift(
        _summary(uplift={"positive_fraction": 0.6}, headline={"uplift_rel": -0.01})
    )["verdict"] == verify_module.FAIL
    assert verify_module.check_uplift(
        _summary(uplift={"positive_fraction": 0.5}, headline={"uplift_rel": 0.01})
    )["verdict"] == verify_module.FAIL
    assert verify_module.check_uplift(_summary())["verdict"] == verify_module.INCONCLUSIVE


@pytest.mark.parametrize(
    "value, expected",
    [(0.5, verify_module.PASS), (1.0, verify_module.PASS),
     (0.0, verify_module.FAIL), (1.5, verify_module.FAIL)],
)
def test_kld_active_frac_bounds(value, expected):
    assert verify_module.check_kld_active_frac(
        _summary(headline={"kld_active_frac": value})
    )["verdict"] == expected


def test_kld_active_frac_is_inconclusive_when_unreported():
    assert verify_module.check_kld_active_frac(
        _summary(headline={})
    )["verdict"] == verify_module.INCONCLUSIVE


def test_specificity_must_resolve_but_need_not_be_source_specific():
    """``influential_not_specific`` is a real finding, not a pipeline failure.

    Treating it as one would be exactly the mistake the prediction-space criterion exists to
    prevent -- and would make the verification reject healthy checkpoints.
    """
    for verdict in ("source_specific", "influential_not_specific", "no_uplift"):
        assert verify_module.check_specificity_resolves(
            _summary(source_specificity={"verdict": verdict})
        )["verdict"] == verify_module.PASS, verdict

    assert verify_module.check_specificity_resolves(
        _summary(source_specificity={"verdict": "undetermined"})
    )["verdict"] == verify_module.FAIL
    assert verify_module.check_specificity_resolves(
        _summary()
    )["verdict"] == verify_module.INCONCLUSIVE


def test_coverage_is_checked_against_the_runs_own_nominal():
    """0.9545 is the two-sided mass of a 2-sigma band; comparing against 0.95 reads as error."""
    inside = _summary(calibration={"coverage": {
        "2sigma": {"nominal": 0.9545, "observed": 0.94, "gap": -0.0145}
    }})
    assert verify_module.check_coverage(inside)["verdict"] == verify_module.PASS

    outside = _summary(calibration={"coverage": {
        "2sigma": {"nominal": 0.9545, "observed": 0.72, "gap": -0.2345}
    }})
    assert verify_module.check_coverage(outside)["verdict"] == verify_module.FAIL


def test_a_skipped_calibration_is_inconclusive_not_a_failure():
    """Another objective has no learned predictive variance, so there is nothing to calibrate."""
    record = verify_module.check_coverage(
        _summary(calibration={"skipped": True, "reason": "likelihood is 'mse'"})
    )
    assert record["verdict"] == verify_module.INCONCLUSIVE
    assert "gaussian_nll" in record["detail"]


def test_headline_finite_and_sanity_block_delegate_to_the_run():
    passing = _summary(sanity={
        "checks": {"headline_finite": {"verdict": "pass", "detail": "all finite"}},
        "failed": [], "n_inconclusive": 1,
    })
    assert verify_module.check_headline_finite(passing)["verdict"] == verify_module.PASS
    assert verify_module.check_sanity_block(passing)["verdict"] == verify_module.PASS

    failing = _summary(sanity={
        "checks": {"headline_finite": {"verdict": "fail", "detail": "nan", "non_finite": ["crps"]}},
        "failed": ["headline_finite"], "n_inconclusive": 0,
    })
    assert verify_module.check_headline_finite(failing)["verdict"] == verify_module.FAIL
    assert verify_module.check_sanity_block(failing)["verdict"] == verify_module.FAIL


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------
def test_an_inconclusive_criterion_does_not_count_as_a_pass():
    """A partial verification must never read as a complete one."""
    report = verify_module.verify(_summary())
    assert report["passed"] is True, "nothing failed, so the run is not rejected"
    assert report["inconclusive"], "but the gaps must be named"
    assert report["n_passed"] < len(verify_module.CRITERIA)
    assert "partial" in verify_module.format_report(report)


def test_any_failure_fails_the_verification():
    report = verify_module.verify({"exit_code": 1, "failed": ["te_lag"], "results": {}})
    assert report["passed"] is False and "exit_code" in report["failed"]
    assert "VERDICT: FAIL" in verify_module.format_report(report)


def test_main_returns_a_nonzero_exit_code_on_failure(tmp_path, capsys):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"exit_code": 1, "failed": ["forecast"], "results": {}}),
                    encoding="utf-8")
    assert verify_module.main(path) == 1
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_main_writes_the_machine_readable_report(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary()), encoding="utf-8")
    out = tmp_path / "verification.json"
    verify_module.main(path, out)

    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["criteria"]) == {name for name, _ in verify_module.CRITERIA}
    assert report["summary_path"] == str(path)


# ---------------------------------------------------------------------------
# Against a real run
# ---------------------------------------------------------------------------
def test_the_criteria_read_the_keys_a_real_run_actually_writes(
    tiny_checkpoint, tmp_path, monkeypatch, repo_root
):
    """The load-bearing test: a criterion digging for a key the pipeline never writes would be
    permanently inconclusive, and would look like a cautious check rather than a broken one.

    The tiny fixture is four samples of an untrained model, so *which* verdicts come back is not
    the point -- that every criterion finds its data is.
    """
    monkeypatch.chdir(repo_root)
    output_dir = tmp_path / "run"
    run_module.main(
        config=str(repo_root / EVAL_TINY_CONFIG),
        checkpoint=str(tiny_checkpoint),
        output_dir=str(output_dir),
        device="cpu",
    )
    summary = json.loads(
        (output_dir / run_module.RESULTS_DIRNAME / "summary.json").read_text(encoding="utf-8")
    )
    report = verify_module.verify(summary)

    # Every criterion that can be evaluated without a trained checkpoint on real data must have
    # found its data. Only these two genuinely depend on the split carrying more than one shard
    # or on the model having learned something.
    structural = {
        "exit_code", "per_file_counts", "weights_loaded", "kld_active_frac",
        "specificity_resolves", "headline_finite", "sanity_block",
    }
    stuck = structural & set(report["inconclusive"])
    assert not stuck, (
        f"criteria {sorted(stuck)} could not find their data in a real summary -- they dig for "
        f"keys the pipeline does not write"
    )
