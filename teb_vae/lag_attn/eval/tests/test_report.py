"""A failing step is captured, an interrupt is not, and the summary is valid JSON either way.

The serialisation half matters more than it looks. A run that computes everything correctly
and then dies writing its summary because one metric came back NaN has produced nothing, and
NaN is a perfectly ordinary result for a fully-masked sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from teb_vae.lag_attn.eval import report as report_module
from teb_vae.lag_attn.eval.report import Report, json_safe


def test_a_raising_step_is_captured_with_its_traceback():
    """``str(exc)`` alone is not enough: for an unattended run the traceback is the whole surface."""
    report = Report()

    def failing():
        raise KeyError("mu_full")

    assert report.step("forecast", failing) is None
    record = report.steps[0]
    assert record.ok is False
    assert "KeyError" in record.error
    assert "in failing" in record.traceback
    assert record.elapsed_s >= 0.0


def test_a_successful_step_returns_its_value_and_its_elapsed_time():
    report = Report()
    assert report.step("forecast", lambda x: x * 2, 21) == 42
    assert report.steps[0].ok is True


def test_later_steps_still_run_after_a_failure():
    """The whole reason the wrapper exists: one analysis must not discard the other ten."""
    report = Report()
    report.step("first", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    report.step("second", lambda: "fine")
    assert [record.ok for record in report.steps] == [False, True]


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_are_not_swallowed(exception):
    """Catching these would turn Ctrl-C into a 'failed step' the run then continues past."""
    report = Report()

    def interrupted():
        raise exception()

    with pytest.raises(exception):
        report.step("forecast", interrupted)


def test_exit_code_is_nonzero_when_any_step_failed():
    report = Report()
    report.step("ok", lambda: None)
    assert report.exit_code() == 0
    report.step("bad", lambda: 1 / 0)
    assert report.exit_code() == 1


def test_summary_json_records_the_failure(tmp_path):
    report = Report()
    report.set("checkpoint", "/path/to.ckpt")
    report.step("bad", lambda: 1 / 0)
    path = report.write(tmp_path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["n_failed"] == 1
    assert written["failed"] == ["bad"]
    assert written["exit_code"] == 1
    assert "ZeroDivisionError" in written["steps"][0]["error"]
    assert written["results"]["checkpoint"] == "/path/to.ckpt"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_json_safe_handles_numpy_paths_and_non_finite_floats():
    value = {
        "int": np.int64(3),
        "float": np.float32(1.5),
        "bool": np.bool_(True),
        "array": np.array([1.0, 2.0]),
        "path": Path("a/b"),
        "nan": float("nan"),
        "inf": float("-inf"),
        "nested": [{"x": np.float64(0.25)}],
    }
    safe = json_safe(value)

    assert safe["int"] == 3 and isinstance(safe["int"], int)
    assert safe["float"] == pytest.approx(1.5) and isinstance(safe["float"], float)
    assert safe["bool"] is True
    assert safe["array"] == [1.0, 2.0]
    assert safe["path"] == str(Path("a/b"))  # str(), so the separator is the platform's
    # null, not the bare NaN token json.dump would otherwise emit: that is not valid JSON and
    # every strict parser rejects it.
    assert safe["nan"] is None and safe["inf"] is None
    assert safe["nested"][0]["x"] == pytest.approx(0.25)


def test_summary_with_non_finite_metrics_is_valid_strict_json(tmp_path):
    report = Report()
    report.set("forecast", {"r2": float("nan"), "mse": np.float32(0.5), "counts": np.arange(3)})
    path = report.write(tmp_path)

    # json.loads accepts NaN by default, so parse strictly to prove the file is portable.
    written = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    assert written["results"]["forecast"]["r2"] is None
    assert written["results"]["forecast"]["counts"] == [0, 1, 2]


def _reject(name):
    """Fail on any JSON constant a strict parser would refuse."""
    raise AssertionError(f"summary.json contains the non-standard constant {name!r}")


def test_a_tensor_is_converted_through_its_own_tolist():
    """Without the branch a tensor falls through to the repr catch-all and lands in the summary
    as the string ``'tensor([3., 4.])'`` -- readable, unparseable, and silently not a number.

    ``torch`` is reached for through ``sys.modules``, so the branch is live exactly when a tensor
    could exist. Importing it here is what puts it there.
    """
    import torch

    safe = json_safe({"block": torch.tensor([3.0, 4.0]), "scalar": torch.tensor(2.5)})

    assert safe["block"] == [3.0, 4.0]
    assert safe["scalar"] == pytest.approx(2.5)
    json.dumps(safe, allow_nan=False)


def test_an_unserialisable_value_is_recorded_rather_than_failing_the_write(tmp_path):
    """A write that raises at the end of a multi-hour run loses everything."""
    report = Report()
    report.set("device", object())
    written = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))
    assert isinstance(written["results"]["device"], str)


# ---------------------------------------------------------------------------
# Headline scalars
# ---------------------------------------------------------------------------
def test_headline_scalars_are_flattened_out_of_their_analysis_blocks():
    """A reader should not have to know which analysis produced which number."""
    headline = report_module.build_headline({
        "forecast": {"mean_feat_mse_total": 0.25, "mean_feat_r2_total": 0.8},
        "latent": {"diagnostics": {"kld_active_frac_masked": 0.5}},
        "collapse": {"verdict": "healthy"},
    })
    assert headline["feat_mse"] == pytest.approx(0.25)
    assert headline["kld_active_frac"] == pytest.approx(0.5), "nested paths must resolve"
    assert headline["collapse"] == "healthy"


def test_a_missing_analysis_yields_none_rather_than_raising():
    """An analysis that failed has no headline, and that must not lose the ones that succeeded."""
    headline = report_module.build_headline({"forecast": {"mean_feat_mse_total": 1.0}})
    assert headline["feat_mse"] == pytest.approx(1.0)
    assert headline["uplift_rel"] is None and headline["source_specificity"] is None


# ---------------------------------------------------------------------------
# Artifact manifest
# ---------------------------------------------------------------------------
def test_the_manifest_lists_every_emitted_file_with_its_size(tmp_path):
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "per_sample.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "forecast" / "heatmaps.pdf").write_bytes(b"%PDF-1.4 stub")
    (tmp_path / "preflight.json").write_text("{}", encoding="utf-8")

    manifest = report_module.build_manifest(tmp_path)
    assert set(manifest["files"]) == {
        "forecast/per_sample.csv", "forecast/heatmaps.pdf", "preflight.json"
    }
    assert manifest["n_files"] == 3
    assert manifest["files"]["preflight.json"] == 2
    # The figure subset is what FIGURE_GUIDE.md's test reads, so it must be exactly the PDFs.
    assert manifest["figures"] == ["forecast/heatmaps.pdf"] and manifest["n_figures"] == 1


def test_the_manifest_excludes_the_summary_it_will_be_written_into(tmp_path):
    (tmp_path / report_module.SUMMARY_FILENAME).write_text("{}", encoding="utf-8")
    (tmp_path / "loader_probe.json").write_text("{}", encoding="utf-8")
    assert list(report_module.build_manifest(tmp_path)["files"]) == ["loader_probe.json"]


# ---------------------------------------------------------------------------
# Coverage and inert caps
# ---------------------------------------------------------------------------
def test_analyses_on_different_populations_raise_a_warning():
    """Two metrics that describe different sets of recordings reconcile only by coincidence."""
    coverage = report_module.build_coverage(
        {
            "forecast": {"n_samples": 500, "composition": {"a.hdf5": 500}},
            "uplift": {"n_samples": 120, "composition": {"a.hdf5": 120}},
        },
        ["forecast", "uplift"],
    )
    assert coverage["warnings"], "a population mismatch must be surfaced"
    assert coverage["per_analysis"]["uplift"]["n_samples"] == 120


def test_a_capped_analysis_is_not_counted_as_a_population_mismatch():
    """A cap is a deliberate, recorded narrowing; flagging it would make the warning noise."""
    coverage = report_module.build_coverage(
        {
            "forecast": {"n_samples": 500},
            "attention": {"n_samples": 100, "plan": {"capped": True}},
        },
        ["forecast", "attention"],
    )
    assert coverage["warnings"] == []


def test_a_cap_above_max_samples_is_reported_as_inert():
    """The operator tuning caps.attention has in fact been bounded by max_samples all along."""
    warnings = report_module.check_inert_caps(
        {"max_samples": 100, "caps": {"attention": 2000, "samples": 8}}
    )
    assert len(warnings) == 1 and "caps.attention" in warnings[0]


def test_no_cap_is_inert_when_max_samples_is_unset():
    assert report_module.check_inert_caps({"max_samples": None, "caps": {"a": 5}}) == []


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def test_per_file_counts_pass_and_fail():
    assert report_module.check_per_file_counts(
        {"per_file": {"a.hdf5": 10, "b.hdf5": 4}}
    )["verdict"] == "pass"
    failed = report_module.check_per_file_counts({"per_file": {"a.hdf5": 10, "b.hdf5": 0}})
    assert failed["verdict"] == "fail" and failed["empty_files"] == ["b.hdf5"]
    assert report_module.check_per_file_counts(None)["verdict"] == report_module.INCONCLUSIVE


def test_classes_present_pass_fail_and_inconclusive():
    assert report_module.check_classes_present(
        {"per_target_class": {"1.0": 30, "2.0": 12}}
    )["verdict"] == "pass"
    assert report_module.check_classes_present(
        {"per_target_class": {"1.0": 30}}
    )["verdict"] == "fail"
    # The healthy-only pretraining split is uniformly zero, and that is correct there.
    assert report_module.check_classes_present(
        {"per_target_class": {"None": 40}}
    )["verdict"] == report_module.INCONCLUSIVE


@pytest.mark.parametrize(
    "attention, expected",
    [
        ({"median_argmax_lag": 12.0, "mean_entropy_nats": 1.0,
          "mean_attainable_entropy_nats": 4.5, "max_possible_entropy_nats": 4.5}, "pass"),
        # Pinned at lag 0: the attention never looks back and the lag window is inert.
        ({"median_argmax_lag": 0.0, "mean_entropy_nats": 1.0,
          "mean_attainable_entropy_nats": 4.5, "max_possible_entropy_nats": 4.5}, "fail"),
        # Uniform weights: the argmax is whichever lag won a rounding contest.
        ({"median_argmax_lag": 12.0, "mean_entropy_nats": 4.49,
          "mean_attainable_entropy_nats": 4.5, "max_possible_entropy_nats": 4.5}, "fail"),
    ],
)
def test_argmax_lag_catches_both_degenerate_readings(attention, expected):
    assert report_module.check_argmax_lag({"attention": attention})["verdict"] == expected


def test_argmax_lag_measures_uniformity_against_the_attainable_ceiling_not_log_l():
    """Divided by $\\log L$ the uniformity branch could never fire at production geometry.

    Causal masking gives anchor $t$ only $\\min(t + 1, L)$ valid lags, so at $L = 91$ with a
    warmup of $30$, attention that is exactly uniform over every causally available lag -- the
    degenerate case this check exists to catch -- scores $4.398$ against $\\log 91 = 4.511$: a
    ratio of $0.9749$, comfortably under the $0.99$ threshold. The 1% margin was justified as
    floating-point slack and is 24x smaller than that systematic gap.
    """
    uniform_over_valid_lags = {
        "median_argmax_lag": 12.0,
        "mean_entropy_nats": 4.397705,
        "mean_attainable_entropy_nats": 4.397705,
        "max_possible_entropy_nats": 4.510860,
    }
    verdict = report_module.check_argmax_lag({"attention": uniform_over_valid_lags})
    assert verdict["verdict"] == "fail", (
        "a structureless attention passed the degeneracy check: the uniformity branch is dead"
    )
    assert verdict["entropy_ratio"] == pytest.approx(1.0, rel=1e-6)

    # The same reading against log L is what the check used to divide by, and it passes.
    assert 4.397705 < 0.99 * 4.510860


def test_argmax_lag_is_inconclusive_when_the_attention_analysis_did_not_report():
    assert report_module.check_argmax_lag({})["verdict"] == report_module.INCONCLUSIVE


def test_a_non_finite_headline_scalar_is_a_failure_not_a_silent_null():
    """A summary of nothing but nulls that exits 0 is the quietest failure the pipeline has."""
    assert report_module.check_headline_finite({"feat_mse": 0.5, "crps": None})["verdict"] == "pass"
    failed = report_module.check_headline_finite({"feat_mse": float("nan"), "crps": 0.1})
    assert failed["verdict"] == "fail" and failed["non_finite"] == ["feat_mse"]
    assert report_module.check_headline_finite(
        {"feat_mse": float("inf")}
    )["verdict"] == "fail"


def test_a_none_headline_is_not_treated_as_non_finite():
    """None means the analysis was skipped, which the step record already covers."""
    assert report_module.check_headline_finite({"a": None, "b": None})["verdict"] == "pass"


def _target_probe(*, n_fractional, n_nonzero=100, n_non_finite=0, binary=False):
    """A probe record shaped as ``probe.run_probe`` produces one."""
    return {
        "target_values": {
            "n_values": 200, "n_nonzero": n_nonzero, "n_fractional": n_fractional,
            "n_non_finite": n_non_finite,
            "any_fractional": bool(n_fractional), "any_non_finite": bool(n_non_finite),
        },
        "weight": {"binary": binary},
    }


def test_target_truncation_is_detected_only_where_it_is_observable():
    # Fractional weight but not one fractional target anywhere: written through an integer dtype.
    assert report_module.check_target_not_truncated(
        _target_probe(n_fractional=0)
    )["verdict"] == "fail"

    assert report_module.check_target_not_truncated(
        _target_probe(n_fractional=12)
    )["verdict"] == "pass"

    # Binary weight makes the two indistinguishable, so the check must say so.
    assert report_module.check_target_not_truncated(
        _target_probe(n_fractional=0, binary=True)
    )["verdict"] == report_module.INCONCLUSIVE


def test_truncation_reads_every_step_not_one_value_per_recording():
    """The regression this check was rewritten for.

    ``per_target_class`` records each recording's *first nonzero* step, which sits in a
    full-weight region on almost every recording -- so it reports "all integers" on perfectly
    healthy fractional-weight data. Reading it would fail every such run.
    """
    healthy_but_integer_at_first_step = {
        "per_target_class": {"1.0": 40, "2.0": 10},   # what the old check read: all integers
        "target_values": {                            # what it must read: fractional elsewhere
            "n_values": 15000, "n_nonzero": 12000, "n_fractional": 340, "n_non_finite": 0,
            "any_fractional": True, "any_non_finite": False,
        },
        "weight": {"binary": False},
    }
    assert report_module.check_target_not_truncated(
        healthy_but_integer_at_first_step
    )["verdict"] == "pass"


def test_a_non_finite_target_is_its_own_failure_not_a_pass():
    """NaN != round(NaN), so a field full of NaN would otherwise count as 'fractional'."""
    assert report_module.check_target_not_truncated(
        _target_probe(n_fractional=5, n_non_finite=3)
    )["verdict"] == "fail"


def test_an_all_zero_target_is_inconclusive_rather_than_failing():
    """The healthy-only pretraining split, where a uniformly zero target is correct."""
    assert report_module.check_target_not_truncated(
        _target_probe(n_fractional=0, n_nonzero=0)
    )["verdict"] == report_module.INCONCLUSIVE


def test_truncation_is_inconclusive_without_the_raw_target_record():
    assert report_module.check_target_not_truncated(
        {"per_target_class": {"1.0": 3}}
    )["verdict"] == report_module.INCONCLUSIVE


def test_the_sanity_block_flags_a_warning_without_touching_the_exit_code():
    """A run can complete every step cleanly and still be one nobody should quote."""
    results = {
        "probe": {"per_file": {"a.hdf5": 0}, "per_target_class": {}, "weight": {}},
        "attention": {},
    }
    sanity = report_module.build_sanity(results, {"feat_mse": 1.0})
    assert sanity["warning"] is True
    assert "per_file_counts" in sanity["failed"] and sanity["n_failed"] >= 1
    assert sanity["n_inconclusive"] >= 1


def test_a_clean_sanity_block_raises_no_warning():
    results = {
        "probe": {
            "per_file": {"a.hdf5": 10},
            # Keyed by clinical class name, as run_probe writes it via labels.clinical_class_code.
            "per_target_class": {"healthy": 6, "acidosis": 4},
            "weight": {"binary": True},
        },
        "attention": {
            "median_argmax_lag": 9.0, "mean_entropy_nats": 1.2,
            "mean_attainable_entropy_nats": 4.4, "max_possible_entropy_nats": 4.5,
        },
    }
    sanity = report_module.build_sanity(results, {"feat_mse": 0.5})
    assert sanity["warning"] is False and sanity["failed"] == []


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------
def test_finalise_adds_every_required_block_and_the_table_prints(tmp_path):
    report = Report()
    report.set("eval_config", {"max_samples": None, "caps": {}})
    report.set("forecast", {"n_samples": 4, "mean_feat_mse_total": 0.5})
    report.set("analyses_selected", ["forecast"])
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "heatmaps.pdf").write_bytes(b"stub")

    report.finalise(tmp_path)
    for key in ("headline", "coverage", "sanity", "artifacts", "config_warnings"):
        assert key in report.results, f"finalise did not produce {key}"
    assert report.results["artifacts"]["figures"] == ["forecast/heatmaps.pdf"]

    table = report.console_table()
    assert "feat_mse" in table and "eval summary" in table


def test_a_block_that_cannot_be_assembled_does_not_lose_the_summary(tmp_path, monkeypatch):
    """``finalise`` runs after every analysis; a failure here would discard the whole run.

    That is exactly what ``step`` exists to prevent, and it would be perverse for the summariser
    to be the one place that does not honour it.
    """
    report = Report()
    report.set("forecast", {"n_samples": 4})
    report.step("forecast", lambda: None)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic manifest failure")

    monkeypatch.setattr(report_module, "build_manifest", boom)
    report.finalise(tmp_path)

    written = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))
    assert "synthetic manifest failure" in written["results"]["artifacts"]["error"]
    # The results and the step records survived, which is the whole point.
    assert written["results"]["forecast"]["n_samples"] == 4
    assert written["steps"][0]["ok"] is True


def test_the_console_table_never_raises_out_of_the_run(tmp_path, monkeypatch):
    report = Report()
    monkeypatch.setattr(
        report_module, "format_console_table",
        lambda *args: (_ for _ in ()).throw(ValueError("synthetic")),
    )
    assert "could not render" in report.console_table()


def test_a_headline_path_resolving_to_a_dict_does_not_break_the_table():
    """A bug in HEADLINE_SCALARS must read as an odd row, not a TypeError that costs the run."""
    table = report_module.format_console_table({"headline": {"feat_mse": {"nested": 1}}}, [])
    assert "feat_mse" in table


def test_the_manifest_excludes_a_previous_runs_files_when_the_directory_is_reused(tmp_path):
    """``--output-dir`` can name the same path twice; the default timestamped one cannot."""
    import os
    import time as _time

    stale = tmp_path / "old.pdf"
    stale.write_bytes(b"stale")
    old = _time.time() - 3600
    os.utime(stale, (old, old))

    boundary = _time.time()
    fresh = tmp_path / "new.pdf"
    fresh.write_bytes(b"fresh")

    manifest = report_module.build_manifest(tmp_path, since=boundary)
    assert manifest["figures"] == ["new.pdf"]
    assert manifest["n_excluded_stale"] == 1
    # Without the boundary it lists both, which is the behaviour a caller with no start time gets.
    assert report_module.build_manifest(tmp_path)["n_figures"] == 2


def test_the_cuda_field_is_absent_rather_than_zero_when_there_is_no_gpu(tmp_path):
    """A 0.00 GB peak reads as a measurement, and on a CPU box it is not one."""
    report = Report()
    report.finalise(tmp_path)
    written = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))
    assert "max_memory_allocated_gb" not in written["results"]
