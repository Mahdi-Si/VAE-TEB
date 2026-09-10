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
from typing import Any, Dict, Mapping

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
COMMITTED_PLAN_DIGEST = "a48059133e35b9ca"

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
    summary = {
        "arm": {"source_disabled": False, **leaves},
        "draws": {"num_mc_samples": draws},
        "run": {"seed": eval_seed, "training_seed": training_seed, "checkpoint": f"{arm}.ckpt"},
        "scored_split": {"label": label, "n_recordings": len(values)},
        "calibration": {},
    }
    return {
        "directory": f"/runs/{arm}-{training_seed}",
        "summary": summary,
        "table": {guid: dict(row) for guid, row in values.items()},
        "probes": None,
        "arm": acceptance.arm_of(summary),
        "training_seed": training_seed,
        "training_tag": f"{arm}_{training_seed}",
        "eval_seed": eval_seed,
        "draws": draws,
        "split": summary["scored_split"],
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
    assert acceptance.arm_of({"arm": {"source_disabled": False, **leaves}}) == expected


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
        json.dumps({"arm": {"source_disabled": False}}), encoding="utf-8"
    )
    (directory / acceptance.PER_RECORDING_FILENAME).write_text("guid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target-only"):
        acceptance.read_reference(str(directory / acceptance.SUMMARY_FILENAME))


def test_the_protocol_reads_the_filenames_the_passes_write() -> None:
    """The names are pinned here rather than imported, so that a rename in either pass shows up as
    a failure rather than as a protocol that quietly finds nothing."""
    from teb_vae.lag_slot_transformer_cfs.eval import latent_probes, run as eval_run

    assert acceptance.PER_RECORDING_FILENAME == eval_run.PER_RECORDING_FILENAME
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
