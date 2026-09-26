r"""The content of this cell's report blocks: the headline, the sanity checks and the step record.

The step-isolation and serialisation mechanism (``Report``, ``json_safe``) is the shared one in
``teb_vae.lag_attn.eval.report`` and is tested there. What this package owns is the **content**
of the three blocks, and that is where this file looks. The headline registry is a promise that
every path in it resolves on a run of this model -- a number that is not registered is invisible
to the acceptance gate and to the arm tables, which read this block and nothing else -- so every
path is walked here against a constructed results dict, long before a real run exists to walk it
against. The sanity checks are exercised on constructed violations, and the argmax-lag check on
both attainable edges of the lag window.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import pytest

from teb_vae.lag_attn.eval import report as shared_report
from teb_vae.lag_attn_cfs.eval import report_seam


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
# The grouped emitter delegates
# =================================================================================================
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
    assert seen["order_groups"](["healthy", "acidosis", "hie"], "clinical_class") == (
        cohort.ordered_groups(["healthy", "acidosis", "hie"], "clinical_class")
    ) == ["hie", "acidosis", "healthy"]
    assert set(seen) == {"frame", "directory", "value_columns", "order_groups", "group_palette"}


# =================================================================================================
# The step record
# =================================================================================================
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


def test_an_unresolved_headline_path_yields_none_rather_than_raising() -> None:
    """An analysis that failed or was skipped legitimately has no headline, and losing the whole
    block to it would be losing the numbers that did resolve."""
    headline = report_seam.build_headline({"readouts": {"mc_pred_gap": 2.0}})

    assert headline["pred_gap_mc_nats"] == pytest.approx(2.0)
    assert headline["kl_argmax_lag_step"] is None
    assert headline["verdict_source_specificity"] is None


def test_only_the_unfloored_kl_may_be_read_as_a_rate() -> None:
    """``source_conditioned_kl_train`` has free bits applied per dimension per step before summing,
    so it exceeds the raw value by construction and hides a collapsed source pathway. The shipped
    ``free_bits: 0.0`` makes the two coincide today, which is exactly why the distinction lives in
    code rather than in an observation."""
    leaves = {path[-1] for _, path in report_seam.HEADLINE_SCALARS}

    assert "source_conditioned_kl_raw" in leaves
    assert "source_conditioned_kl_train" not in leaves


# =================================================================================================
# The verdict registry
# =================================================================================================
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
def test_every_sanity_check_yields_a_verdict_on_an_empty_run() -> None:
    """A run that produced nothing must still get a readable sanity block rather than a raise."""
    sanity = report_seam.build_sanity({}, {})

    assert sanity["checks"]
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


#: A profile with a readable shape: one clear peak and a bulk well below it. Used wherever the
#: check's *edge* logic is what is under test, so that degeneracy -- which is judged first -- cannot
#: be what produced the verdict.
_SHAPED_PROFILE = [1.0, 1.0, 1.0, 9.0, 1.0, 1.0]

#: Per-lag anchor counts where the top two lags of an eight-bin window are attainable by no anchor.
#: The ceiling is therefore 5 rather than 7, which is the correction the check exists to make: a lag
#: no anchor contributed to is not a lag the peak could have sat at.
_COUNTS = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 0.0, 0.0]


def _lag_summary(argmax, profile=None, counts=None):
    """A ``lag`` block shaped like the one the collection pass writes."""
    return {
        "lag": {
            "kl_argmax_lag_step": argmax,
            "kl_lag_anchor_counts": list(_COUNTS if counts is None else counts),
            "kl_lag_profile_support_corrected": list(
                _SHAPED_PROFILE if profile is None else profile
            ),
            "attention_entropy_per_head_nats": [1.3, 2.0, 2.3, 2.5],
        }
    }


@pytest.mark.parametrize(
    "argmax, expected, reason",
    [
        (0, report_seam.INCONCLUSIVE, "CENSORING"),
        (5, "fail", "the peak is against the window edge"),
        (3, "pass", "strictly inside"),
    ],
)
def test_the_argmax_lag_is_judged_against_both_attainable_edges(argmax, expected, reason) -> None:
    r"""Both ends of the window censor, and the check is symmetric in them.

    The **far** edge has always been read as censoring: a peak at the largest attainable lag means
    the true maximum may lie beyond $L$. The **near** edge is the mirror image and used to be read
    as inertness -- "the attribution never looks back" -- which is a conclusion the geometry does
    not support. By the identity
    $\tau^{\mathrm{phys}}_{\ell,h} = \Delta(\ell + 1 + h) + (\tau^u_{\mathrm{ref}}
    - \tau^y_{\mathrm{ref}})$, every physical delay shorter than the one lag
    $0$ encodes is reported *at* lag $0$, because the window carries no bin below it. On this
    family's geometry a $20$-$60$ s physiological delay is below lag $0$ at most horizon steps, so
    a pin at the floor is the readout hitting a wall.

    The verdicts differ in kind for that reason. A far-edge pin still FAILs, because a peak against
    the far edge is a model whose readout the window truncated and the run could have been
    configured otherwise. A near-edge pin is INCONCLUSIVE: the measurement happened and the answer
    is outside what this geometry can express.

    Both edges are read from the per-lag anchor counts rather than taken as $0$ and $L - 1$: a lag
    no anchor contributed to is not attainable at all.
    """
    record = report_seam.check_argmax_lag(_lag_summary(argmax))

    assert record["verdict"] == expected
    assert record["attainable_lag_ceiling"] == 5
    assert record["attainable_lag_floor"] == 0
    assert reason in record["detail"]


def test_a_near_edge_pin_carries_the_evidence_the_machinery_is_alive() -> None:
    """What the INCONCLUSIVE verdict has to carry, because it is not a pass and must not read as
    one: the shape statistics that say the measurement was real.

    Without them a reader cannot tell a censored answer from a model that reported nothing --
    which is precisely the confusion the old FAIL made, in the other direction.
    """
    record = report_seam.check_argmax_lag(_lag_summary(0))

    assert record["verdict"] == report_seam.INCONCLUSIVE
    assert record["peak_degenerate"] is False
    assert record["mass_above_half_peak"] is not None
    assert record["attention_entropy_per_head_nats"] == [1.3, 2.0, 2.3, 2.5]


@pytest.mark.parametrize("argmax", [0, 3, 5])
def test_a_degenerate_profile_fails_at_either_edge_and_in_the_middle(argmax) -> None:
    """Degeneracy is judged **first**, and that ordering is the whole reason the near edge can be
    reported as censoring at all.

    A profile whose peak is not distinguishable from its bulk has an argmax that names a bin rather
    than a lag: there is no measurement to censor, so a censoring verdict on it would report a
    geometry limit where the readout is simply absent. Asserted at all three positions, because a
    check that only guarded the middle would let a flat profile pass as censored at the edge that
    every arm of this family pins at.
    """
    record = report_seam.check_argmax_lag(_lag_summary(argmax, profile=[1.0] * 6))

    assert record["verdict"] == "fail"
    assert record["peak_degenerate"] is True
    assert "names a bin rather than a lag" in record["detail"]


def test_the_floor_lifts_off_zero_when_the_lowest_lags_carry_no_anchor() -> None:
    """The near edge is ``min(attainable)`` and not the literal $0$, which is what makes it
    symmetric with the far edge.

    At a short sequence length the lowest bins can carry no anchor at all, and a check reading the
    floor as $0$ would then call a pin at the real floor an interior peak -- reporting a censored
    readout as a clean pass, which is the failure this symmetry removes.
    """
    counts = [0.0, 0.0, 6.0, 5.0, 4.0, 3.0, 0.0, 0.0]
    record = report_seam.check_argmax_lag(_lag_summary(2, counts=counts))

    assert record["attainable_lag_floor"] == 2
    assert record["verdict"] == report_seam.INCONCLUSIVE


def test_a_run_with_no_per_lag_support_is_inconclusive_rather_than_judged() -> None:
    """No anchor counts means no attainable set, so neither edge is defined. Reported as
    unevaluated rather than as an interior peak, which is what an empty ``max`` would otherwise
    have to be defended against."""
    record = report_seam.check_argmax_lag(
        {"lag": {"kl_argmax_lag_step": 0, "kl_lag_anchor_counts": []}}
    )

    assert record["verdict"] == report_seam.INCONCLUSIVE
    assert "support" in record["detail"]


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
