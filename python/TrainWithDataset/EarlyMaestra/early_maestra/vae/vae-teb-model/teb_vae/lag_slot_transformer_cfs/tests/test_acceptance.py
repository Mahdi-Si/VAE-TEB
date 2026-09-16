r"""The acceptance protocol: the predeclaration, the seeds, the pairing and the reserved partition.

Every failure this file is written against is one where the arithmetic is right and the statement
is wrong:

* a comparison, a band or a seed minimum edited after a result was seen, which changes no number
  and no reader can detect;
* three scorings of one checkpoint counted as three training seeds;
* two arms compared across two draw counts or two splits, so a difference of estimators reads as a
  difference of models;
* a band chosen for being the largest and then reported at the interval of a band chosen in
  advance;
* a confirmation run whose partition the selection has already been scored on, which is a
  development number wearing a confirmation's name.

The integration fixture at the bottom runs the real thing: several fits at several seeds, each
scored through the real pass, assembled by the real protocol. It is the only place a defect
*between* the scoring pass and the protocol can show up -- a column renamed, a provenance field
never written, a table whose recordings do not line up across runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytest
import yaml

from teb_vae.lag_slot_transformer_cfs.eval import acceptance

from .test_arms import fit_arm, score_arm

#: The committed plan's digest, pinned.
#:
#: This assertion exists to FAIL when the predeclaration is edited. That is not a nuisance: the
#: plan's whole value is that it was written before any arm was trained, and an edit made after a
#: result has been seen is invisible in every number it produces. A deliberate revision updates
#: this literal and the plan's own revision counter together, and the diff is then the record that
#: it happened.
COMMITTED_PLAN_DIGEST = "5653fb573d2961c1"

#: The saved production export of the 91-entry development run, read here for its recorded
#: lag window alone: the acceptance pass must read it under the family declared for that window.
SAVED_EXPORT_SUMMARY = (
    Path(__file__).resolve().parents[3] / "output" / "lag_slot_summary" / "summary.json"
)

#: Draws the integration fixture scores at, and the seeds it fits. Small: this asserts that the
#: protocol reads what the pass writes, not the convergence of an estimator.
ARM_SEEDS = (11, 12, 13)
FIXTURE_DRAWS = 3
FIXTURE_RESAMPLES = 100


def plan_with(tmp_path: Path, **protocol: Any) -> Dict[str, Any]:
    """A plan file for a test, differing from the committed one only where a test needs it.

    Args:
        tmp_path: Where to write it.
        **protocol: Protocol settings to replace.

    Returns:
        The loaded plan.
    """
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    declaration["protocol"].update(protocol)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False), encoding="utf-8")
    return acceptance.load_plan(str(path))


def fake_run(
    arm: str,
    *,
    training_seed: int,
    values: Mapping[str, Mapping[str, float]],
    draws: int = 32,
    eval_seed: int = 42,
    label: str = "/data/fold_1/test",
    searched_lag_steps: Optional[int] = None,
    input_policy: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """One run record in the shape :func:`acceptance.discover_runs` returns.

    Built rather than read, so a test about the arithmetic does not need a fit behind it.

    Args:
        arm: The arm's name, written into the leaves the resolver reads.
        training_seed: The training seed.
        values: ``{recording: {column: value}}``.
        draws: The Monte Carlo draw count the run was scored at.
        eval_seed: The evaluation seed.
        label: The split's label.
        searched_lag_steps: The lag window the run's causality record names, or ``None`` for a
            summary that records none, as every summary written before the record carried it.
        input_policy: The two ablation flags the arm record carries, or ``None`` for both off.

    Returns:
        The record.
    """
    leaves = {
        "candidate": {"source_stem": "pointwise", "lag_fusion": "local"},
        "mean_only": {
            "source_stem": "pointwise",
            "lag_fusion": "local",
            "mean_only_residual": True,
        },
        "target_only": {"source_disabled": True},
    }[arm]
    # The family's summary layout: this cell's blocks under ``results``, the training identity
    # under the runner's ``run_context``, the evaluation seed under ``eval_config``, the searched
    # window under the promoted ``causality`` record.
    summary = {
        "checkpoint": f"{arm}.ckpt",
        "eval_config": {"seed": eval_seed},
        "run_context": {"training_seed": training_seed, "training_tag": f"{arm}_{training_seed}"},
        "results": {
            "arm": {"source_disabled": False, **leaves, **dict(input_policy or {})},
            "num_mc_samples": draws,
            "arms": {"scored_split": {"label": label, "n_recordings": len(values)}},
            "mixture_calibration": {},
        },
    }
    if searched_lag_steps is not None:
        summary["causality"] = {"searched_lag_steps": searched_lag_steps}
    return {
        **acceptance.run_identity(f"/runs/{arm}-{training_seed}", summary),
        "table": {guid: dict(row) for guid, row in values.items()},
        "probes": None,
    }


def straight_line(offset: float, *, n: int = 8) -> Dict[str, Dict[str, float]]:
    """A per-recording table whose gap is a known constant.

    Args:
        offset: The value ``nll_full`` sits below ``nll_base`` by.
        n: Recordings.

    Returns:
        ``{recording: {column: value}}``.
    """
    return {
        f"REC{index:03d}": {
            "nll_base": 100.0 + index,
            "nll_full": 100.0 + index - offset,
            "pred_gap": offset,
            "kld_per_anchor": 0.5,
            "nll_silence": 100.0 + index,
            "nll_replace:zeros": 100.5 + index,
            "nll_suppress:near": 100.0 + index - offset + 0.2 * index,
            "nll_suppress:far": 100.0 + index - offset + 0.05,
            "nll_suppress:none": 100.0 + index - offset,
            "nll_suppress:all": 100.0 + index,
        }
        for index in range(n)
    }


# =============================================================================
# The predeclaration
# =============================================================================
def test_the_committed_plan_loads_and_declares_only_arms_this_package_builds() -> None:
    """A comparison against an arm no configuration constructs is a comparison of nothing."""
    plan = acceptance.load_plan()
    known = set(acceptance.ARM_LEAVES.values()) | {"target_only"}

    assert set(plan["protocol"]) == acceptance.PROTOCOL_KEYS
    assert plan["primary_comparisons"]
    for entry in plan["primary_comparisons"]:
        assert entry["left"] in known and entry["right"] in known, entry["name"]


def test_the_committed_plans_digest_is_pinned() -> None:
    """The pin is the mechanism, and it is meant to fail when the declaration moves.

    Nothing stops the plan being edited. What this makes impossible is editing it *quietly*: a
    comparison added, a band appended or a seed minimum lowered after a result has been seen
    changes no number in any record and changes this literal.
    """
    assert acceptance.load_plan()["digest"] == COMMITTED_PLAN_DIGEST


def test_an_unknown_plan_key_is_refused_by_name(tmp_path: Path) -> None:
    """Nothing reads a misspelled key, so a plan carrying one would silently mean 'no minimum'."""
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    declaration["minimum_seeds"] = 3
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(declaration), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_seeds"):
        acceptance.load_plan(str(path))


def test_a_comparison_against_an_unbuilt_arm_is_refused(tmp_path: Path) -> None:
    """A declared comparison whose arm does not exist would report NO_EVIDENCE forever."""
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    declaration["primary_comparisons"][0]["right"] = "slot_transformer"
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(declaration), encoding="utf-8")

    with pytest.raises(ValueError, match="slot_transformer"):
        acceptance.load_plan(str(path))


def test_a_protocol_missing_a_setting_is_refused(tmp_path: Path) -> None:
    """A verdict reached under a value nobody wrote down is not a declared verdict."""
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    declaration["protocol"].pop("minimum_training_seeds")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(declaration), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_training_seeds"):
        acceptance.load_plan(str(path))


# =============================================================================
# Which arm produced a summary
# =============================================================================
@pytest.mark.parametrize(
    "leaves,expected",
    [
        ({"source_disabled": True}, "target_only"),
        ({"source_stem": "pointwise", "lag_fusion": "local"}, "candidate"),
        (
            {"source_stem": "pointwise", "lag_fusion": "local", "mean_only_residual": True},
            "mean_only",
        ),
        (
            {"source_stem": "pointwise", "lag_fusion": "local", "source_values_withheld": True},
            "capacity_control",
        ),
        ({"source_stem": "pointwise", "lag_fusion": "attention"}, "pointwise_attention"),
        ({"source_stem": "conv", "lag_fusion": "attention"}, "attention_reference"),
    ],
)
def test_the_arm_is_resolved_from_the_leaves(leaves: Dict[str, Any], expected: str) -> None:
    """From what the model was constructed with, not from what a directory was called."""
    summary = {"results": {"arm": {"source_disabled": False, **leaves}}}
    assert acceptance.arm_of(summary) == expected


def test_an_unrecognised_combination_is_reported_rather_than_guessed() -> None:
    """A scalar-lift arm is legitimate and is not in the chain; naming it 'candidate' would put a
    different model into a comparison of one declared change."""
    resolved = acceptance.arm_of(
        {
            "arm": {
                "source_disabled": False,
                "source_stem": "pointwise",
                "lag_fusion": "local",
                "source_scalar_lift": True,
            }
        }
    )

    assert resolved.startswith("unrecognised:")


# =============================================================================
# The arithmetic
# =============================================================================
def test_seed_averaging_uses_only_recordings_every_seed_scored() -> None:
    """A recording one seed missed would be represented by the seeds that saw it, and the vector
    would then average a different number of fits in different rows."""
    first = fake_run("candidate", training_seed=1, values=straight_line(1.0, n=4))
    second = fake_run("candidate", training_seed=2, values=straight_line(3.0, n=4))
    second["table"].pop("REC003")

    averaged = acceptance.seed_average([first, second], "pred_gap")

    assert set(averaged) == {"REC000", "REC001", "REC002"}
    assert all(value == pytest.approx(2.0) for value in averaged.values())


def test_three_scorings_of_one_checkpoint_are_not_three_seeds(tmp_path: Path) -> None:
    """The seed minimum is about repeated FITS. Counting one fit scored three times would let a
    draw-count sweep stand in for a repeated experiment."""
    plan = plan_with(tmp_path, primary_draws=32)
    runs = [
        fake_run("candidate", training_seed=7, values=straight_line(1.0), eval_seed=seed)
        for seed in (1, 2, 3)
    ]

    evidence = acceptance.arm_evidence(runs, plan=plan)

    assert evidence["candidate"]["n_training_seeds"] == 1
    assert not evidence["candidate"]["meets_minimum"]


def test_a_comparison_across_two_draw_counts_is_refused_as_unmatched(tmp_path: Path) -> None:
    """The negative logarithm of an average likelihood is upward biased at finite draws and two
    arms' biases need not cancel, so this would be a difference of estimators."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1)
    left = fake_run("candidate", training_seed=1, values=straight_line(1.0))
    right = fake_run("mean_only", training_seed=1, values=straight_line(0.5))
    # Same declared draw count in the grouping, a different evaluation seed in the run: two draw
    # sets, and the pairing would carry the difference between them.
    right["eval_seed"] = 99

    block = acceptance.primary_comparison_block(
        acceptance.arm_evidence([left, right], plan=plan), plan=plan
    )

    assert block["the_variance_update"]["status"] == "UNMATCHED"
    assert "eval_seed" in block["the_variance_update"]["detail"]


def test_two_arms_on_two_partitions_have_no_paired_difference(tmp_path: Path) -> None:
    """The safety net under the reserved partition: a development run and a reserved one share no
    recording, so pairing them produces nothing rather than a number built on an empty overlap."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1)
    left = fake_run("candidate", training_seed=1, values=straight_line(1.0))
    right = fake_run("mean_only", training_seed=1, values=straight_line(0.5))
    right["table"] = {f"OTHER{index}": row for index, row in enumerate(right["table"].values())}

    block = acceptance.primary_comparison_block(
        acceptance.arm_evidence([left, right], plan=plan), plan=plan
    )

    assert block["the_variance_update"]["status"] == "NO_SHARED_RECORDINGS"


def test_the_comparison_reads_the_difference_the_tables_carry(tmp_path: Path) -> None:
    """The one arithmetic assertion: a planted constant difference comes back as that constant."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    left = [
        fake_run("candidate", training_seed=seed, values=straight_line(2.0)) for seed in (1, 2, 3)
    ]
    right = [
        fake_run("mean_only", training_seed=seed, values=straight_line(0.5)) for seed in (1, 2, 3)
    ]

    block = acceptance.primary_comparison_block(
        acceptance.arm_evidence(left + right, plan=plan), plan=plan
    )
    record = block["the_variance_update"]

    assert record["status"] == "READ"
    # nll_full is lower on the arm with the larger gap, so the difference is negative and the
    # left arm is the better one -- which is the sign convention the plan declares.
    assert record["difference_nats"]["point"] == pytest.approx(-1.5)
    assert record["difference_nats"]["method"] == "percentile bootstrap over recordings"


def test_the_spread_travels_beside_the_interval(tmp_path: Path) -> None:
    """An interval on a mean can exclude zero on a population a third of whose recordings have the
    opposite sign, and a record carrying only the interval would describe that as uniform."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    runs = [fake_run("candidate", training_seed=seed, values=straight_line(2.0)) for seed in (1, 2)]

    record = acceptance.internal_and_reference_block(
        acceptance.arm_evidence(runs, plan=plan), None, plan=plan
    )["candidate"]

    assert set(record["internal_gap_nats"]["spread"]) >= {"min", "median", "max"}
    assert record["against_reference"]["status"] == "NO_REFERENCE"


# =============================================================================
# The band search
# =============================================================================
def test_the_family_adjusted_interval_is_wider_than_the_nominal_one(tmp_path: Path) -> None:
    """The peak band is chosen on the same recordings its interval is built from, so a nominal
    interval on the winner covers less than it claims."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=200)
    runs = [fake_run("candidate", training_seed=seed, values=straight_line(2.0)) for seed in (1, 2)]

    block = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)["candidate"]
    near = block["bands"]["near"]

    assert block["searched_bands"] == ["far", "near"]
    assert block["family_size"] == 2
    assert block["family_adjusted_confidence"] > block["nominal_confidence"]
    assert near["margin_nats_family_adjusted"]["hi"] >= near["margin_nats"]["hi"]
    assert near["margin_nats_family_adjusted"]["lo"] <= near["margin_nats"]["lo"]


def test_the_identity_arms_are_not_members_of_the_search(tmp_path: Path) -> None:
    """The empty band removes nothing and the full band removes everything: neither is a lag the
    search ranges over, and including them would widen every other band's interval for a constant.
    """
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    runs = [fake_run("candidate", training_seed=1, values=straight_line(2.0))]

    block = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)["candidate"]

    assert "none" not in block["searched_bands"] and "all" not in block["searched_bands"]


def test_the_peak_is_the_largest_margin(tmp_path: Path) -> None:
    """The fixture plants a margin that grows with the recording index in one band and a constant
    in the other, so which band peaks is known in advance."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    runs = [fake_run("candidate", training_seed=1, values=straight_line(2.0))]

    block = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)["candidate"]

    assert block["peak_band"] == "near"


def test_a_band_outside_the_declaration_fails_the_gate(tmp_path: Path) -> None:
    """A search whose family grew after its peak was seen has no multiplicity at all."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    values = straight_line(2.0)
    for row in values.values():
        row["nll_suppress:undeclared"] = row["nll_full"] + 0.4
    runs = [fake_run("candidate", training_seed=1, values=values)]

    bands = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)
    verdict = acceptance.check_declared_bands(bands)

    assert verdict["status"] == "FAIL"
    assert verdict["undeclared"]["candidate"] == ["undeclared"]


# =============================================================================
# The band family is the window's
# =============================================================================
def short_window_values(offset: float, *, bands: Mapping[str, Any], n: int = 8) -> Dict[str, Dict[str, float]]:
    """A per-recording table whose suppression columns are one window's family.

    Args:
        offset: The value ``nll_full`` sits below ``nll_base`` by.
        bands: The family's band names, each getting a suppressed column.
        n: Recordings.

    Returns:
        ``{recording: {column: value}}``.
    """
    table = straight_line(offset, n=n)
    for row in table.values():
        for name in list(row):
            if name.startswith("nll_suppress:") and name not in ("nll_suppress:none", "nll_suppress:all"):
                del row[name]
        for position, name in enumerate(bands):
            row[f"nll_suppress:{name}"] = row["nll_full"] + 0.1 * (position + 1)
    return table


def test_a_run_is_read_under_the_family_declared_for_its_own_window(tmp_path: Path) -> None:
    """The committed plan declares one family per window; a run over the short window is read
    under that window's family and is not failed for searching bands the wide family lacks."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    families = plan["exploratory_band_families"]
    assert len(families) >= 2
    wide, short = max(families), min(families)
    runs = [
        fake_run(
            "candidate", training_seed=1, searched_lag_steps=short,
            values=short_window_values(2.0, bands=families[short]),
        )
    ]

    block = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)["candidate"]
    verdict = acceptance.check_declared_bands({"candidate": block})

    assert block["status"] == "READ"
    assert block["lag_window"] == short and block["lag_window_note"] is None
    assert block["declared_bands"] == families[short]
    assert block["searched_bands"] == sorted(families[short])
    assert block["undeclared_bands"] == []
    assert block["family_size"] == len(families[short])
    assert verdict["status"] == "PASS"
    assert verdict["windows"] == {"candidate": short}
    # And the wide family is a different list, so the same bands under the wide window would fail.
    assert set(families[short]) != set(families[wide])


def test_a_run_over_a_window_the_plan_declares_no_family_for_is_refused(tmp_path: Path) -> None:
    """A search over an undeclared window has no fixed family, which is the same failure as an
    undeclared band by another route."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    undeclared = max(plan["exploratory_band_families"]) + 7
    runs = [
        fake_run("candidate", training_seed=1, searched_lag_steps=undeclared, values=straight_line(2.0))
    ]

    bands = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)
    verdict = acceptance.check_declared_bands(bands)

    assert bands["candidate"]["status"] == "UNDECLARED_FAMILY"
    assert bands["candidate"]["lag_window"] == undeclared
    assert verdict["status"] == "FAIL"
    assert verdict["undeclared_windows"]["candidate"]["lag_window"] == undeclared


def test_an_arm_whose_runs_span_two_windows_is_refused_by_name(tmp_path: Path) -> None:
    """Two windows are two searches, and one family cannot cover both."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    families = plan["exploratory_band_families"]
    wide, short = max(families), min(families)
    runs = [
        fake_run("candidate", training_seed=1, searched_lag_steps=wide, values=straight_line(2.0)),
        fake_run(
            "candidate", training_seed=2, searched_lag_steps=short,
            values=short_window_values(2.0, bands=families[short]),
        ),
    ]

    bands = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)

    assert bands["candidate"]["status"] == "MIXED_LAG_WINDOWS"
    assert acceptance.check_declared_bands(bands)["status"] == "FAIL"


def test_a_summary_that_records_no_window_is_read_under_the_shipped_one(tmp_path: Path) -> None:
    """Every summary written before the causality record carried the window searched the shipped
    production window, whose family is the revision-1 list; the record says the reading was
    indirect."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    runs = [fake_run("candidate", training_seed=1, values=straight_line(2.0))]

    block = acceptance.band_block(acceptance.arm_evidence(runs, plan=plan), plan=plan)["candidate"]

    assert block["status"] == "READ"
    assert block["lag_window"] == acceptance.shipped_lag_window()
    assert "shipped production window" in block["lag_window_note"]
    assert block["declared_bands"] == plan["exploratory_band_families"][acceptance.shipped_lag_window()]


def test_a_plan_carrying_only_the_legacy_band_key_is_read_as_the_shipped_windows_family(tmp_path: Path) -> None:
    """The revision-1 key named one family without saying which window; it was the shipped one."""
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    families = declaration.pop("exploratory_band_families")
    shipped = acceptance.shipped_lag_window()
    declaration["exploratory_bands"] = list(families[shipped])
    path = tmp_path / "legacy_plan.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="revision-1"):
        plan = acceptance.load_plan(str(path))

    assert plan["exploratory_band_families"] == {shipped: list(families[shipped])}
    assert plan["band_family_source"].startswith("legacy")
    assert "exploratory_bands" not in plan


def test_a_plan_carrying_both_band_keys_is_refused(tmp_path: Path) -> None:
    """Two declarations of one window cannot both be the predeclaration."""
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    declaration["exploratory_bands"] = ["anchor"]
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(declaration, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exploratory_bands"):
        acceptance.load_plan(str(path))


@pytest.mark.skipif(not SAVED_EXPORT_SUMMARY.is_file(), reason="the saved export is not on this machine")
def test_the_saved_wide_window_export_is_read_under_the_wide_family() -> None:
    """The development run's own causality record names its window, and the committed plan
    declares that window's family: the revision-1 bands unchanged."""
    summary = json.loads(SAVED_EXPORT_SUMMARY.read_text(encoding="utf-8"))
    plan = acceptance.load_plan()

    run = acceptance.run_identity(str(SAVED_EXPORT_SUMMARY.parent), summary)
    window, note = acceptance.arm_lag_window([run])

    assert run["searched_lag_steps"] == window == acceptance.shipped_lag_window()
    assert note is None
    assert plan["exploratory_band_families"][window] == ["anchor", "near", "mid", "far"]
    assert run["input_policy"] == {"zero_fhr_scattering_s0": False, "zero_up_scattering_s0": False}


def test_two_windows_or_two_input_policies_do_not_pair(tmp_path: Path) -> None:
    """The bank length and the ablation are leaves of their own: a pair that differs in either
    would carry that difference under whichever leaf the comparison names."""
    plan = plan_with(tmp_path, primary_draws=32, minimum_training_seeds=1, bootstrap_resamples=50)
    families = plan["exploratory_band_families"]
    wide, short = max(families), min(families)
    left = fake_run("candidate", training_seed=1, searched_lag_steps=wide, values=straight_line(1.0))
    right = fake_run("mean_only", training_seed=1, searched_lag_steps=short, values=straight_line(0.5))
    ablated = fake_run(
        "mean_only", training_seed=1, searched_lag_steps=wide, values=straight_line(0.5),
        input_policy={"zero_fhr_scattering_s0": True},
    )
    reference = fake_run("target_only", training_seed=9, values=straight_line(0.0))

    windows = acceptance.primary_comparison_block(
        acceptance.arm_evidence([left, right], plan=plan), plan=plan
    )["the_variance_update"]
    policies = acceptance.primary_comparison_block(
        acceptance.arm_evidence([left, ablated], plan=plan), plan=plan
    )["the_variance_update"]

    assert windows["status"] == "UNMATCHED" and "lag window" in windows["detail"]
    assert policies["status"] == "UNMATCHED" and "input policy" in policies["detail"]
    # A target-only run searches no window, so it pairs with either window.
    assert acceptance.scoring_mismatch([left, reference]) is None
    descriptor = acceptance.run_descriptor(ablated)
    assert descriptor["zero_fhr_scattering_s0"] is True and descriptor["searched_lag_steps"] == wide


# =============================================================================
# The verdicts
# =============================================================================
def test_a_confidently_degraded_base_fails_and_an_unclear_one_does_not() -> None:
    """The one verdict that can fail on a measurement rather than on a defect, and the reason it
    is not a threshold: a base branch behind the frozen reference makes every gap measured against
    it a measurement of that."""
    confident = {
        "candidate": {
            "against_reference": {"base_minus_reference_nats": {"point": 3.0, "lo": 1.0, "hi": 5.0}}
        }
    }
    unclear = {
        "candidate": {
            "against_reference": {
                "base_minus_reference_nats": {"point": 0.4, "lo": -2.0, "hi": 2.8}
            }
        }
    }
    improved = {
        "candidate": {
            "against_reference": {
                "base_minus_reference_nats": {"point": -1.0, "lo": -3.0, "hi": 0.5}
            }
        }
    }

    assert acceptance.check_base_not_degraded(confident)["status"] == "FAIL"
    assert acceptance.check_base_not_degraded(unclear)["status"] == "INCONCLUSIVE"
    assert acceptance.check_base_not_degraded(improved)["status"] == "PASS"


def test_the_confirmation_verdict_decides_on_recordings_rather_than_on_paths() -> None:
    """Two evaluation directories can name different shards and still overlap: one fold's train
    partition holds another fold's test recordings."""
    selection = [fake_run("candidate", training_seed=1, values=straight_line(1.0, n=6))]
    overlapping = [
        fake_run(
            "candidate",
            training_seed=1,
            values=straight_line(1.0, n=6),
            label="/data/fold_2/test",
        )
    ]
    reserved = [
        fake_run(
            "candidate",
            training_seed=1,
            values={
                f"HELD{index}": row
                for index, row in enumerate(straight_line(1.0, n=6).values())
            },
            label="/data/fold_2/test",
        )
    ]

    assert acceptance.check_confirmation_partition(selection, [])["status"] == "INCONCLUSIVE"
    assert acceptance.check_confirmation_partition(selection, overlapping)["status"] == "FAIL"
    assert acceptance.check_confirmation_partition(selection, reserved)["status"] == "PASS"


def test_a_source_conditioned_run_is_refused_as_the_reference(tmp_path: Path) -> None:
    """A model cannot serve as the baseline its own architecture is measured against."""
    directory = tmp_path / "eval_results"
    directory.mkdir(parents=True)
    (directory / acceptance.SUMMARY_FILENAME).write_text(
        json.dumps({"results": {"arm": {"source_disabled": False}}}), encoding="utf-8"
    )
    table = directory / acceptance.PER_RECORDING_TABLE
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("guid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target-only"):
        acceptance.read_reference(str(directory / acceptance.SUMMARY_FILENAME))


def test_the_protocol_reads_the_filenames_the_passes_write() -> None:
    """The names are pinned here rather than imported, so that a rename in either pass shows up as
    a failure rather than as a protocol that quietly finds nothing."""
    from teb_vae.lag_slot_transformer_cfs.eval import latent_probes
    from teb_vae.lag_slot_transformer_cfs.eval.analyses import arms

    assert acceptance.PER_RECORDING_TABLE == arms.PER_RECORDING_TABLE
    assert acceptance.PROBE_FILENAME == latent_probes.PROBE_FILENAME


# =============================================================================
# The whole protocol, on runs a fit actually produced
# =============================================================================
@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """Fit one arm at three seeds and a second at one, score them all, and score a reference.

    Module-scoped: the five fits are the expensive part of this file.

    Args:
        tmp_path_factory: pytest's directory factory.

    Returns:
        ``(runs root, reference summary path, work directory)``.
    """
    # A short root: the fits write deeply nested run directories, and a long temporary prefix
    # pushes the checkpoint paths past what the platform will rename.
    work = Path(tmp_path_factory.mktemp("p"))
    fits, runs, reference_root = work / "f", work / "d", work / "r"
    for directory in (fits, runs, reference_root):
        directory.mkdir(parents=True, exist_ok=True)

    for seed in ARM_SEEDS:
        checkpoint = fit_arm(fits, f"c{seed}", {"general_config.seed": seed})
        score_arm(runs, f"c{seed}", checkpoint)
    mean_only = fit_arm(fits, "mo", {"model_config.VAE_model.mean_only_residual": True})
    score_arm(runs, "mo", mean_only)

    target_only = fit_arm(
        fits, "ref", {"model_config.VAE_model.source_disabled": True, "general_config.seed": 99}
    )
    score_arm(reference_root, "ref", target_only)
    return runs, reference_root / "eval_ref" / "eval_results" / acceptance.SUMMARY_FILENAME, work


@pytest.mark.slow
def test_the_protocol_reads_runs_the_scoring_pass_wrote(evidence, tmp_path) -> None:
    """The seam. Five runs on disk, discovered, grouped by arm and training seed, and compared.

    A defect between the two passes shows up here and nowhere else: a renamed column, a provenance
    field never written, or a table whose recordings do not line up across runs.
    """
    runs_root, reference, _work = evidence
    plan = plan_with(tmp_path, primary_draws=FIXTURE_DRAWS, bootstrap_resamples=FIXTURE_RESAMPLES)

    discovered = acceptance.discover_runs(str(runs_root))
    record = acceptance.assess(
        discovered, plan=plan, reference=acceptance.read_reference(str(reference))
    )

    assert len(discovered) == len(ARM_SEEDS) + 1
    assert record["selection"]["arms"]["candidate"]["n_training_seeds"] == len(ARM_SEEDS)
    assert record["selection"]["arms"]["candidate"]["meets_minimum"]
    assert record["selection"]["arms"]["mean_only"]["n_training_seeds"] == 1
    assert not record["selection"]["arms"]["mean_only"]["meets_minimum"]


@pytest.mark.slow
def test_every_run_carries_the_provenance_the_protocol_groups_on(evidence) -> None:
    """A run whose training seed or split is recoverable only from a shell history cannot be
    grouped, and the protocol's first act is to group."""
    runs_root, _reference, _work = evidence

    for run in acceptance.discover_runs(str(runs_root)):
        descriptor = acceptance.run_descriptor(run)
        assert descriptor["training_seed"] is not None
        assert descriptor["split_label"]
        assert descriptor["recording_digest"]
        assert descriptor["n_recordings"] > 0
        assert descriptor["has_per_recording_table"]


@pytest.mark.slow
def test_the_declared_comparison_is_read_against_the_frozen_reference(evidence, tmp_path) -> None:
    """The comparison the internal gap cannot make, and the two seed counts that qualify it."""
    runs_root, reference, _work = evidence
    plan = plan_with(tmp_path, primary_draws=FIXTURE_DRAWS, bootstrap_resamples=FIXTURE_RESAMPLES)

    record = acceptance.assess(
        acceptance.discover_runs(str(runs_root)),
        plan=plan,
        reference=acceptance.read_reference(str(reference)),
    )
    comparison = record["selection"]["primary_comparisons"]["the_variance_update"]
    against = record["selection"]["per_arm"]["candidate"]["against_reference"]

    assert comparison["status"] == "BELOW_SEED_MINIMUM"
    assert comparison["difference_nats"]["n"] > 0
    assert against["status"] == "READ"
    for name in ("full_minus_reference_nats", "base_minus_reference_nats"):
        assert against[name]["lo"] <= against[name]["point"] <= against[name]["hi"]


@pytest.mark.slow
def test_the_record_says_what_it_has_not_measured(evidence, tmp_path) -> None:
    """Absence and zero must not read the same. No confirmation run and no probe artifact are both
    present as statements rather than as missing blocks."""
    runs_root, reference, _work = evidence
    plan = plan_with(tmp_path, primary_draws=FIXTURE_DRAWS, bootstrap_resamples=FIXTURE_RESAMPLES)

    record = acceptance.assess(
        acceptance.discover_runs(str(runs_root)),
        plan=plan,
        reference=acceptance.read_reference(str(reference)),
    )
    confirmation = next(
        entry for entry in record["verdicts"] if entry["name"] == "confirmation_partition"
    )

    assert confirmation["status"] == "INCONCLUSIVE"
    assert "confirms nothing" in confirmation["detail"]
    assert record["selection"]["latent_probes"]["candidate"]["status"] == "NOT_RUN"
    assert "confirmation" not in record


@pytest.mark.slow
def test_the_entry_point_writes_a_record_and_reports_its_verdicts(evidence, tmp_path) -> None:
    """Through ``main``, as an operator runs it: a plan, a root, a reference and a file."""
    runs_root, reference, _work = evidence
    declaration = yaml.safe_load(Path(acceptance.DEFAULT_PLAN_PATH).read_text(encoding="utf-8"))
    declaration["protocol"]["primary_draws"] = FIXTURE_DRAWS
    declaration["protocol"]["bootstrap_resamples"] = FIXTURE_RESAMPLES
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(declaration, sort_keys=False), encoding="utf-8")
    output = tmp_path / "acceptance.json"

    code = acceptance.main(
        runs=str(runs_root),
        reference=str(reference),
        plan=str(plan_path),
        output=str(output),
        sources={"runs": "cli"},
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert code == 0
    assert written["plan"]["digest"] and written["plan"]["digest"] != COMMITTED_PLAN_DIGEST
    assert written["run"]["argument_sources"]["runs"] == "cli"
    assert [entry["name"] for entry in written["verdicts"]]
    assert written["passed"]


@pytest.mark.slow
def test_a_root_with_no_evaluation_directory_is_refused(tmp_path: Path) -> None:
    """This pass reads finished evaluation directories, and an empty root is an operator error
    rather than an empty result."""
    assert acceptance.main(runs=str(tmp_path)) == 2
    assert acceptance.main() == 2
