r"""One real fit, then one real evaluation of it through the family's runner, then both gates.

Everything else in this suite tests a piece against constructed tensors. This runs the whole
evaluation path against a checkpoint a fit actually produced, on the generated integer-operator
cohort: the override merge over that run's own resolved configuration, the preflight guards, the
loader probe, the rebuild through the checkpoint contract, this cell's own collection pass with
every intervened arm scored in one draw loop, the family's durable tables, every analysis the
binding resolves to with its by-class and by-subgroup variants, the family's headline and sanity
block, and this cell's own gate over the summary.

It is the only place the failures that live *between* those pieces can surface, and the ones worth
naming are all of the same kind: a seam that is individually correct and jointly wrong. A checkpoint
whose keys carry a task prefix the rebuild does not strip. A band read from a configuration whose
lag window the fit never had. A margin taken against an arm that was scored under different noise. A
column the family's analyses read under a name this cell's pass never wrote.

**The reference identities are the assertions to read first.** Suppressing an empty band must
reproduce the matched arm exactly, and suppressing every band must reproduce the target-only prior
exactly. They hold here on real weights and a real split, which is what makes every other margin in
the file a difference of predictions rather than a difference between two code paths.

**It starts no run of its own.** The session-scoped ``slot_collected_run`` fixture is this suite's
one end-to-end pass, and :func:`test_this_file_starts_no_run_of_its_own` keeps that from quietly
changing.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any, Dict, Set

import pytest

from teb_vae.lag_attn_cfs.eval import collect, preflight, probe as probe_module
from teb_vae.lag_attn_cfs.tests.test_eval_smoke import DURABLE_ARTIFACTS
from teb_vae.lag_slot_transformer_cfs.eval import run as run_module
from teb_vae.lag_slot_transformer_cfs.eval import verify as eval_verify
from teb_vae.lag_slot_transformer_cfs.eval.analyses import arms, lag_suppression, resolved_axes

from .conftest import COHORT_PROFILE_SEGMENTS

pytestmark = pytest.mark.slow


def _registry() -> Dict[str, Any]:
    """This cell's analyses, resolved through its binding on every call."""
    return run_module.analysis_registry()


def _results(run: Dict[str, Any]) -> Dict[str, Any]:
    """This cell's own blocks, under the family's ``results`` key."""
    return run["summary"]["results"]


# =================================================================================================
# The run itself
# =================================================================================================
def test_the_full_run_completes_with_exit_code_zero(slot_collected_run) -> None:
    """The failed steps are named with their errors rather than left to a bare ``1 == 0``: this run
    is the most expensive thing the suite does, so a failure that does not say which analysis
    raised buys a second one."""
    failed = [
        f"{record['name']}: {record.get('error')}"
        for record in slot_collected_run["summary"]["steps"]
        if record["status"] != "ok"
    ]

    assert failed == [], failed
    assert slot_collected_run["exit_code"] == 0


def test_every_registered_analysis_contributes_a_step_record(slot_collected_run) -> None:
    """Every selectable analysis, the unskippable channel map, and the loader probe: a step each,
    every one ok. The registry is the family's less the two this architecture cannot produce, plus
    this cell's own six -- so this is also the assertion that binding the family's runner did not
    silently drop a question."""
    steps = {record["name"]: record["status"] for record in slot_collected_run["summary"]["steps"]}

    expected = {"probe", *run_module.UNSKIPPABLE_ANALYSES, *_registry()}
    assert expected <= set(steps), sorted(expected - set(steps))
    assert {"arms", "lag_suppression", "resolved_axes", "samples", "recording_traces",
            "attribution"} <= set(steps)
    assert not {"attention", "lag_kl", "source_null", "occlusion", "lag_clocks",
                "lag_kld_scaled", "lag_high_kl"} & set(steps)
    assert all(status == "ok" for status in steps.values()), steps


def test_the_artifact_layout_is_the_familys_own(slot_collected_run) -> None:
    """The same durable names, imported from the causal cell's suite rather than restated: a
    directory of this cell is read down the same layout as a lag-attentive cell's, and every
    analysis that did not record a skip left its own subdirectory."""
    results_dir = Path(slot_collected_run["results_dir"])
    results = _results(slot_collected_run)

    assert preflight.PREFLIGHT_FILENAME in DURABLE_ARTIFACTS
    assert probe_module.PROBE_FILENAME in DURABLE_ARTIFACTS
    assert collect.COLLECTION_FILENAME in DURABLE_ARTIFACTS
    for name in DURABLE_ARTIFACTS:
        assert (results_dir / name).is_file(), f"the run left no {name}"

    subdirectories = {path.name for path in results_dir.iterdir() if path.is_dir()}
    silent: Set[str] = {
        name for name in _registry()
        if name not in subdirectories and not (results.get(name) or {}).get("skipped")
    }
    assert silent == set(), f"no artifact subdirectory and no recorded skip for {sorted(silent)}"
    wrote = subdirectories & set(_registry())
    assert len(wrote) > len(set(_registry()) - wrote), (
        f"only {sorted(wrote)} wrote artifacts; the rest recorded skips, so this run demonstrates "
        f"the skip path rather than the pipeline"
    )
    # This cell's own three readout analyses, each with its table and its figures.
    extension = slot_collected_run["summary"]["eval_config"]["figure_format"]
    assert (results_dir / arms.ANALYSIS_DIRNAME / arms.PER_RECORDING_FILENAME).is_file()
    assert (results_dir / arms.ANALYSIS_DIRNAME / f"{arms.HEADLINE_FIGURE}.{extension}").is_file()
    assert (results_dir / lag_suppression.ANALYSIS_DIRNAME / lag_suppression.LAG_PROFILE_FILENAME).is_file()
    assert (results_dir / lag_suppression.ANALYSIS_DIRNAME / f"{lag_suppression.PROFILE_FIGURE}.{extension}").is_file()
    assert (results_dir / resolved_axes.ANALYSIS_DIRNAME / resolved_axes.HORIZON_FILENAME).is_file()
    # And the family's cohort-aware families rendered against a multi-cohort split.
    assert list(results_dir.glob("samples/*/*.pdf")), "no per-recording pages rendered"
    assert [path for path in results_dir.rglob("*_by_clinical_class.pdf")], "no grouped variants"


def test_the_shared_tables_carry_the_familys_columns_and_none_of_the_attention_ones(
    slot_collected_run,
) -> None:
    """The per-sample table the family's analyses read, under the family's names, written by this
    cell's own pass -- and no column that names a tensor this architecture does not compute."""
    results_dir = Path(slot_collected_run["results_dir"])
    with open(results_dir / collect.PER_SAMPLE_FILENAME, encoding="utf-8") as handle:
        columns = set(next(csv.reader(handle)))

    assert set(collect.IDENTITY_COLUMNS) <= columns
    assert {
        "nll_base_block", "nll_full_block", "pred_gap", "mc_nll_base_block", "mc_nll_full_block",
        "mc_pred_gap", "mean_pred_gap", "source_conditioned_kl_raw",
        "source_conditioned_kl_shuffled_raw", "mc_nll_shuffled_block",
        "mc_nll_base_shuffled_mu_block", "nll_persistence_block", "nll_climatology_block",
        "nll_segment_mean_block", "sq_error_base", "sq_error_full", "pred_gap_st", "pred_gap_ph",
        "pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi", "anchors_per_sample",
        "target_warm_frac", "mean_logvar_prior", "logvar_prior_floor_frac", "mean_logvar_full",
        "logvar_full_floor_frac", "logvar_full_ceil_frac", "delta_mu_sat_frac_masked",
    } <= columns, sorted(columns)
    assert not columns & {
        "attention_entropy_nats", "source_lag_warmth_frac_st", "kld_source_null",
        "coupling_minus_clock", "lag_map_identity_max_abs",
    }


def test_the_sanity_block_holds_the_identities_this_cell_can_measure(slot_collected_run) -> None:
    """The per-dimension divergence sums to the raw divergence and the per-anchor table
    recombines into the per-sample one; the lag identities the family measures on an attention are
    inconclusive here rather than failed, because there is no map to sum."""
    sanity = _results(slot_collected_run)["sanity"]

    assert sanity["failed"] == [], sanity["failed"]
    assert sanity["checks"]["kl_identity"]["verdict"] == "pass"
    assert sanity["checks"]["per_anchor_recombines"]["verdict"] == "pass"
    for name in ("argmax_lag", "lag_map_sums_to_kl", "per_head_kl_sums_to_kl"):
        assert sanity["checks"][name]["verdict"] == "inconclusive", name


def test_the_familys_verdicts_are_decided_and_the_clock_one_is_inconclusive(
    slot_collected_run,
) -> None:
    """The family's registry, decided over this cell's own overall means: the predictive and the
    latent criteria read quantities this pass produces, and the availability-clock criterion is
    INCONCLUSIVE by construction because no source-null arm exists here."""
    verdicts = {record["name"]: record["status"] for record in _results(slot_collected_run)["verdicts"]}

    assert verdicts["coupling_exceeds_availability_clock"] == "INCONCLUSIVE"
    for name in ("predictive_improvement", "source_specificity", "prior_carries_target_state",
                 "anchor_geometry_intact"):
        assert verdicts[name] in {"PASS", "FAIL", "INCONCLUSIVE"}, name
    assert verdicts["anchor_geometry_intact"] == "PASS"
    headline = _results(slot_collected_run)["headline"]
    assert headline["pred_gap_mc_nats"] is not None
    assert headline["pred_gap_mc_ci_lo"] <= headline["pred_gap_mc_nats"] <= headline["pred_gap_mc_ci_hi"]


# =================================================================================================
# This cell's own blocks
# =================================================================================================
def test_the_matched_gap_carries_a_recording_level_interval(slot_collected_run) -> None:
    """Recordings, not anchors: consecutive anchors' forecast windows overlap in all but one of
    their steps at the dense geometry, so an interval resampled over anchors would be narrower than
    the data supports."""
    results = _results(slot_collected_run)
    record = results["arm_scores"]["pred_gap"]

    assert record["method"] == "percentile bootstrap over recordings"
    assert record["n"] == results["n_recordings"]
    assert record["lo"] <= record["point"] <= record["hi"]
    # The same number under the family's name.
    assert results["readouts"]["mc_pred_gap"] == pytest.approx(record["point"])


def test_both_summaries_are_reported_separately(slot_collected_run) -> None:
    """They differ whenever recordings contribute unequal anchor counts, which is always, and a
    run carrying one of them cannot separate a real effect from a length effect."""
    results = _results(slot_collected_run)

    assert "pred_gap" in results["arm_scores"]
    assert "pred_gap" in results["anchor_weighted"]


def test_the_reference_arms_are_exact_on_real_weights(slot_collected_run) -> None:
    """The two identities that make every other margin in this file a measurement."""
    results = _results(slot_collected_run)
    bands = results["lag_readouts"]["band_suppression"]
    gap = results["arm_scores"]["pred_gap"]["point"]

    assert abs(bands["none"]["margin_nats"]) < eval_verify.EXACT_MARGIN_TOLERANCE
    assert abs(bands["all"]["margin_nats"] - gap) < eval_verify.EXACT_MARGIN_TOLERANCE
    assert (
        abs(results["source_controls"]["silence_margin_nats"] - gap)
        < eval_verify.EXACT_MARGIN_TOLERANCE
    )


def test_every_band_reports_its_usable_counts_beside_its_margin(slot_collected_run) -> None:
    """A margin without its exposure is a number a reader cannot weigh."""
    results = _results(slot_collected_run)
    bands = results["lag_readouts"]["band_suppression"]
    declared = list(results["lag_readouts"]["band_edges"])

    assert list(bands) == ["none", *declared, "all"]
    for name in declared:
        assert bands[name]["band_anchors"] > 0.0
        assert bands[name]["margin_nats"] is not None
        interval = bands[name]["margin_interval"]
        assert interval["n_paired"] == results["n_recordings"]


def test_the_three_source_controls_and_the_prior_shuffle_are_reported(slot_collected_run) -> None:
    """The permutation arm's counts are what say it is still a control, and the prior-shuffle
    control is scored under the same pairing so the two name one stranger."""
    results = _results(slot_collected_run)
    controls = results["source_controls"]

    for margin in ("replace_zeros_margin_nats", "replace_constant_margin_nats", "permute_margin_nats"):
        assert controls[margin] is not None, margin
    assert controls["n_control_pairs"] > 0
    assert controls["n_same_recording_pairs"] == 0
    assert "nll_base_shuffled_mu" in results["arm_scores"]
    assert results["readouts"]["mc_nll_base_shuffled_mu_block"] is not None


def test_the_lag_profile_reads_every_lag_in_latent_space_and_the_capped_ones_predictively(
    slot_collected_run,
) -> None:
    """The latent profile over the whole split; the predictive one over the cap, paired."""
    results = _results(slot_collected_run)
    profile = results["lag_readouts"]["lag_profile"]
    n_lags = results["lag_readouts"]["lag_axis"]["n_lags"]

    latent = profile["latent"]
    for name in ("proposal_norm", "update_shift", "divergence_drop"):
        assert len(latent[name]) == n_lags
    predictive = profile["predictive"]
    assert predictive["status"] == "READ"
    assert predictive["cap"] == COHORT_PROFILE_SEGMENTS
    assert 0 < predictive["n_segments"] <= COHORT_PROFILE_SEGMENTS + 4
    assert len(predictive["margin_nats"]["point"]) == n_lags


def test_the_resolved_axes_and_the_mixture_calibration_are_carried(slot_collected_run) -> None:
    """One curve per scored arm on the horizon axis, both stored blocks on the block axis, and the
    mixture calibration of both branches beside the family's observation-model census."""
    results = _results(slot_collected_run)
    horizon = results["horizon_resolved"]
    steps = horizon["positions"]

    assert steps == list(range(1, len(steps) + 1))
    assert {"base", "full", "suppress:none", "suppress:all", "silence"} <= set(horizon["nll"])
    assert results["block_resolved"]["positions"] == ["st", "ph"]
    for branch in ("base", "full"):
        assert results["mixture_calibration"][branch]["n_coefficients"] > 0
    assert results["calibration"]["n_coefficients"] > 0


def test_the_traces_and_attributions_ran_under_the_familys_names(slot_collected_run) -> None:
    """This cell's own stages, registered under the family's three names, drew a class-balanced
    selection on the multi-cohort split rather than recording a skip."""
    results = _results(slot_collected_run)

    assert results["recording_traces"]["status"] == "TRACED"
    assert results["recording_traces"]["n_recordings"] > 0
    assert results["attribution"]["status"] == "ATTRIBUTED"
    assert results["attribution"]["failures"] == []
    assert results["samples"]["n_samples"] > 0


def test_the_summary_names_the_analyses_this_architecture_cannot_produce(slot_collected_run) -> None:
    """A reader finding fewer columns than a sibling should not have to work out why."""
    results = _results(slot_collected_run)
    excluded = results["excluded_analyses"]

    assert "attention" in excluded and "lag_kl" in excluded
    assert set(results["excluded_analyses_mechanism"]["removed_from_shared_registry"]) == {
        "attention", "lag_kl"
    }
    assert not set(slot_collected_run["summary"]["analyses_selected"]) & set(excluded)


def test_no_attention_shaped_key_appears_anywhere_in_the_output(slot_collected_run) -> None:
    """A proposal norm under an attention name is the one claim the design proves cannot be made."""
    summary = slot_collected_run["summary"]
    keys = set(eval_verify._walk_keys(summary)) - set(_results(slot_collected_run)["excluded_analyses"])

    assert not keys & set(eval_verify.FORBIDDEN_KEYS)


def test_the_acceptance_gate_passes_on_what_the_run_wrote(slot_collected_run) -> None:
    """The gate reads the artifact rather than the objects that produced it."""
    summary, summary_path = slot_collected_run["summary"], slot_collected_run["summary_path"]
    result = eval_verify.verify(summary)

    assert result["failed"] == [], result["failed"]
    assert eval_verify.main(summary=str(summary_path)) == 0
    gap_verdict = next(
        record for record in result["verdicts"] if record["name"] == "predictive_gap_measured"
    )
    assert gap_verdict["status"] == "INCONCLUSIVE"
    assert gap_verdict["pred_gap_nats"] is not None


def test_the_per_recording_table_carries_what_the_intervals_were_built_from(slot_collected_run) -> None:
    """One row per recording under this cell's own names, where the acceptance pass reads it."""
    results = _results(slot_collected_run)
    table_path = Path(slot_collected_run["results_dir"]) / arms.PER_RECORDING_TABLE
    rows = list(csv.DictReader(table_path.read_text(encoding="utf-8").splitlines()))

    assert len(rows) == results["n_recordings"]
    assert {"guid", "nll_base", "nll_full", "pred_gap", "n_scored_anchors"} <= set(rows[0])
    gaps = [float(row["pred_gap"]) for row in rows]
    assert sum(gaps) / len(gaps) == pytest.approx(results["arm_scores"]["pred_gap"]["point"])
    split = results["arms"]["scored_split"]
    assert split["per_recording_table"] == arms.PER_RECORDING_TABLE
    assert split["n_recordings"] == results["n_recordings"]
    assert len(split["recording_digest"]) == 16


def test_the_run_records_its_provenance_where_the_family_records_it(slot_collected_run) -> None:
    """The training seed and tag in the runner's context block, the evaluation seed in the dumped
    settings, the argument sources beside them: nothing recoverable only from a shell history."""
    summary = slot_collected_run["summary"]

    assert summary["checkpoint"].endswith(".ckpt")
    assert summary["run_context"]["training_seed"] is not None
    assert summary["run_context"]["training_tag"]
    assert summary["run_context"]["model_class"] == "SeqVaeLagResidualTrfCfs"
    assert summary["arguments"]["sources"]["checkpoint"] == "cli"
    assert summary["causality"]["source_disabled"] is False
    assert summary["causality"]["source_receptive_field_steps"] == 1
    assert _results(slot_collected_run)["arm"]["model_kind"] == "fhr_lag_residual_cfs_v1"
    assert (slot_collected_run["summary_path"].parent / "resolved_config.yaml").is_file()


# =================================================================================================
# The one pass
# =================================================================================================
def test_this_file_starts_no_run_of_its_own() -> None:
    """The suite performs exactly one end-to-end pass and every artifact assertion reads it."""
    source = Path(__file__).read_text(encoding="utf-8")

    calls = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "main"
    ]

    assert calls == [], "this file calls main(); read the session-scoped run instead"
