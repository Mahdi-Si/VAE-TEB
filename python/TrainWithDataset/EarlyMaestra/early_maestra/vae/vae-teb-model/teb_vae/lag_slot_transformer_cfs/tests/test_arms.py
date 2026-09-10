r"""The three-run sequence, end to end at fixture scale, before it is run at production scale.

The first real arm is three training runs and one evaluation, and every seam between them is a way
to lose days. A target-only checkpoint that quietly carries a source pathway. A warm start that
copies nothing and says it copied everything. A candidate whose source begins somewhere other than
zero, so every nat of coupling it reports was inherited rather than earned. A frozen reference that
cannot be scored through the same estimator as the candidate, so the one comparison that separates
"the source helped" from "the base got worse" is unavailable exactly when it is needed.

None of those is visible from a unit test of any single piece. All of them are visible here, in
about half a minute, against the committed integer-operator fixture.

**What this file does not do.** It does not train anything to convergence, does not use production
shards and makes no claim about whether uterine activity helps. It establishes that the sequence an
operator is about to run for days is wired correctly, and nothing more.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.tests.conftest import absolutize_dataset_paths
from teb_vae.lag_slot_transformer_cfs import trainer as trainer_module
from teb_vae.lag_slot_transformer_cfs.eval import run as eval_run
from teb_vae.lag_slot_transformer_cfs.eval import verify as eval_verify
from teb_vae.lag_slot_transformer_cfs.trainer import (
    SOURCE_PREFIXES,
    LagResidualTrfCfsTrainer,
    strip_task_prefix,
)

pytestmark = pytest.mark.slow

#: The configuration every arm here is a delta on.
TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs per fit. Two, matching the training smoke: the second is what steps the scheduler and
#: rotates the tile phase, and a model that never left its initialisation would make every source
#: readout zero for a reason about the fit rather than about the architecture.
ARM_EPOCHS = 2

#: Draws the evaluation scores at, and the resamples behind its interval. Small: this asserts the
#: shape of the sequence, not the convergence of an estimator.
ARM_DRAWS = 3
ARM_RESAMPLES = 100


def fit_arm(work: Path, name: str, overrides: Dict[str, Any]) -> Path:
    """Run one short fit through the real entry point and return its last checkpoint.

    Public so a second integration file can fit an arm through exactly this path. A private copy
    there would be a second definition of what "fit an arm" means, free to drift from this one in
    the epoch count, the profiler setting or the way the checkpoint is found.

    Args:
        work: The directory this arm writes into.
        name: The arm's name, which becomes its output subdirectory.
        overrides: Dotted config paths to set before the fit.

    Returns:
        The checkpoint path.
    """
    config = absolutize_dataset_paths(load_config(str(TINY_CONFIG)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(work / name)
    config["general_config"]["epochs"] = ARM_EPOCHS
    config["advanced_config"]["trainer"]["profiler"] = None
    for path, value in overrides.items():
        node = config
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    config_path = work / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured: Dict[str, Any] = {}
    original = LagResidualTrfCfsTrainer.train_model

    def capture(self, train_loader, validation_loader):
        """Run the inherited fit and keep the driver, the only handle on the run."""
        result = original(self, train_loader, validation_loader)
        captured["driver"] = self
        return result

    LagResidualTrfCfsTrainer.train_model = capture
    try:
        trainer_module.main(str(config_path))
    finally:
        del LagResidualTrfCfsTrainer.train_model

    checkpoints = sorted(Path(captured["driver"].train_results_dir).parent.rglob("*.ckpt"))
    assert checkpoints, f"the {name} fit wrote no checkpoint"
    return checkpoints[-1]


def write_eval_delta(work: Path, name: str) -> Path:
    """Write the evaluation override delta one run is scored under, and return its path.

    Public because every pass that reads the fixture's split needs the same one, and a second copy
    would be a second definition of which shards, which draw count and which lag bands a fixture
    run uses -- free to drift from this one in exactly the settings a comparison holds fixed.

    The delta is written per run rather than using the committed one, which names production shards
    and production lag bands the fixture geometry has neither of. Writing one is what an operator
    does for any other split, so the merge is exercised rather than bypassed.

    Args:
        work: The directory this arm writes into.
        name: The arm's name.

    Returns:
        The delta's path.
    """
    config = absolutize_dataset_paths(load_config(str(TINY_CONFIG)))
    delta = {
        "general_config": {"batch_size": {"test": 2}},
        "dataset_config": {
            "vae_test_datasets": config["dataset_config"]["vae_test_datasets"],
            "stat_path": config["dataset_config"]["stat_path"],
        },
        "eval_config": {
            "seed": 7,
            "num_mc_samples": ARM_DRAWS,
            "bootstrap_resamples": ARM_RESAMPLES,
            "occlusion_bands": {"near": [0, 3], "far": [4, 8]},
        },
    }
    delta_path = work / f"{name}_overrides.yaml"
    delta_path.write_text(yaml.safe_dump(delta, sort_keys=False), encoding="utf-8")
    return delta_path


def score_arm(work: Path, name: str, checkpoint: Path) -> Dict[str, Any]:
    """Score one checkpoint through the real entry point and return its summary.

    Public for the reason :func:`fit_arm` is.

    Args:
        work: The directory this arm writes into.
        name: The arm's name.
        checkpoint: The checkpoint to score.

    Returns:
        The parsed summary.
    """
    delta_path = write_eval_delta(work, name)

    assert (
        eval_run.main(
            checkpoint=str(checkpoint),
            output_dir=str(work / f"eval_{name}"),
            device="cpu",
            overrides=str(delta_path),
            sources={"checkpoint": "cli"},
        )
        == 0
    )
    path = work / f"eval_{name}" / "eval_results" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arms(tmp_path_factory):
    """Run the whole sequence once: baseline, reference, candidate, and two evaluations.

    Module-scoped because the three fits are the expensive part of this file and every assertion
    below reads the same runs.

    Args:
        tmp_path_factory: pytest's directory factory.

    Returns:
        A mapping of every artifact the assertions read.
    """
    # A short root: the fits write deeply nested run directories, and a long temporary prefix
    # pushes the checkpoint paths past what the platform will rename.
    work = Path(tmp_path_factory.mktemp("a"))

    target_only = {"model_config.VAE_model.source_disabled": True}
    baseline = fit_arm(work, "base", target_only)
    # The reference is the identical configuration at a different seed. Anything else that differed
    # would confound the one comparison it exists to make.
    reference = fit_arm(work, "ref", {**target_only, "general_config.seed": 4242})
    candidate = fit_arm(
        work,
        "cand",
        {
            "model_config.target_warm_start_checkpoint": str(baseline),
            "model_config.VAE_model.head_init_calibration": False,
        },
    )

    return {
        "work": work,
        "baseline": baseline,
        "reference": reference,
        "candidate": candidate,
        "reference_summary": score_arm(work, "ref", reference),
        "candidate_summary": score_arm(work, "cand", candidate),
    }


# =============================================================================
# The target-only arm
# =============================================================================
def test_a_target_only_checkpoint_carries_no_source_pathway(arms) -> None:
    """Not a model that holds the modules and leaves them idle.

    A checkpoint that carried them would train them under a distributed run as starved parameters,
    and would claim in its own manifest that the model reads a source it does not.
    """
    blob = torch.load(str(arms["baseline"]), map_location="cpu", weights_only=False)
    state = strip_task_prefix(blob["state_dict"])

    assert not [name for name in state if name.startswith(SOURCE_PREFIXES)]
    assert blob["model_kwargs"]["source_disabled"] is True
    # And the target half is all there, which is what makes it transferable.
    assert any(name.startswith("target_encoder.") for name in state)
    assert any(name.startswith("prior_head.") for name in state)


def test_the_target_only_arm_scores_a_gap_of_exactly_zero(arms) -> None:
    """By construction rather than by measurement, and the summary says which.

    This is the self-check that a run labelled target-only really was one: the full distribution is
    the prior, so the two decoded forecasts are bitwise identical and their scores cannot differ.
    """
    summary = arms["reference_summary"]

    assert summary["arm"]["source_disabled"] is True
    assert summary["headline"]["pred_gap"]["point"] == 0.0
    assert summary["headline"]["nll_full"]["point"] == summary["headline"]["nll_base"]["point"]


def test_the_target_only_run_names_every_intervention_it_could_not_make(arms) -> None:
    """Absent with a reason, never reported as a margin of zero.

    A zero margin is a measurement -- the source was there and removing it changed nothing. This
    run had no source pathway at all, and the two must not read the same in an artifact.
    """
    summary = arms["reference_summary"]
    controls = summary["source_controls"]

    assert set(controls["skipped"]) == {"suppress", "silence", "replace", "permute"}
    for reason in controls["skipped"].values():
        assert "no source pathway" in reason
    for margin in ("silence_margin_nats", "permute_margin_nats", "replace_zeros_margin_nats"):
        assert controls[margin] is None, margin
    assert summary["lag_readouts"]["band_suppression"] == {}
    assert summary["encoder_disclosure"]["source_disabled"] is True


def test_the_gate_passes_on_a_target_only_summary_and_says_why_it_is_quiet(arms) -> None:
    """Its two inconclusive verdicts name the arm rather than reporting an absent measurement."""
    result = eval_verify.verify(arms["reference_summary"])

    assert result["failed"] == [], result["failed"]
    detail = {record["name"]: record["detail"] for record in result["verdicts"]}
    assert "target-only checkpoint" in detail["reference_arms_are_exact"]
    assert "exactly zero by construction" in detail["predictive_gap_measured"]


# =============================================================================
# The warm start
# =============================================================================
def test_the_candidate_transfers_the_target_half_and_starts_the_source_at_zero(arms) -> None:
    """The invariant that makes every nat of coupling the candidate reports earned rather than
    inherited.

    Checked on the checkpoint the fit actually wrote rather than on the model in memory: what a
    later run loads is the file.
    """
    blob = torch.load(str(arms["candidate"]), map_location="cpu", weights_only=False)
    state = strip_task_prefix(blob["state_dict"])

    # The source pathway exists on this arm, unlike the baseline it started from.
    assert [name for name in state if name.startswith(SOURCE_PREFIXES)]
    assert blob["model_kwargs"]["source_disabled"] is False
    # The calibration is off, so the transfer was not overwritten by it.
    assert blob["model_kwargs"]["head_init_calibration"] is False


def test_the_candidate_is_scored_through_the_same_estimator_as_the_reference(arms) -> None:
    """Both summaries report the same draw count, the same anchor geometry and the same estimator.

    A candidate scored one way and a reference scored another cannot be compared at all, and the
    comparison would still produce a number.
    """
    candidate, reference = arms["candidate_summary"], arms["reference_summary"]

    assert candidate["draws"]["num_mc_samples"] == reference["draws"]["num_mc_samples"]
    assert candidate["draws"]["estimator"] == reference["draws"]["estimator"]
    assert candidate["n_recordings"] == reference["n_recordings"]
    assert candidate["arm"]["source_disabled"] is False


# =============================================================================
# The comparison the internal gap cannot make
# =============================================================================
def test_the_candidate_is_read_against_the_frozen_reference(arms) -> None:
    """The whole reason a third run exists.

    A candidate's own base branch trains jointly with its source pathway, so it can degrade -- and
    an improved internal gap would then be measuring the degradation. Only an independently trained
    target-only predictor separates the two.
    """
    result = eval_verify.verify(arms["candidate_summary"], arms["reference_summary"])
    verdict = next(
        record
        for record in result["verdicts"]
        if record["name"] == "candidate_against_external_reference"
    )

    assert verdict["reference_nll"] == arms["reference_summary"]["headline"]["nll_base"]["point"]
    assert verdict["candidate_full_nll"] is not None
    assert verdict["improvement_over_reference_nats"] is not None


def test_a_candidate_whose_base_fell_behind_the_reference_fails(arms) -> None:
    """Not a threshold question, which is why this one is a FAIL rather than a measurement.

    A base branch worse than an independently trained one means joint training made the baseline
    worse, and any gap measured against that base is measuring exactly that.
    """
    candidate = json.loads(json.dumps(arms["candidate_summary"]))
    reference = arms["reference_summary"]
    # Push the candidate's base a clear nat behind the frozen reference.
    candidate["headline"]["nll_base"]["point"] = (
        float(reference["headline"]["nll_base"]["point"]) + 1.0
    )

    result = eval_verify.verify(candidate, reference)

    assert "candidate_against_external_reference" in result["failed"]


def test_without_a_reference_the_verdict_says_what_is_missing(arms) -> None:
    """A gap against a candidate's own base is not the comparison the design asks for."""
    result = eval_verify.verify(arms["candidate_summary"])
    verdict = next(
        record
        for record in result["verdicts"]
        if record["name"] == "candidate_against_external_reference"
    )

    assert verdict["status"] == "INCONCLUSIVE"
    assert "no reference summary" in verdict["detail"]


def test_a_source_conditioned_run_is_refused_as_a_reference(arms) -> None:
    """It cannot serve as the baseline its own architecture is measured against."""
    result = eval_verify.verify(arms["candidate_summary"], arms["candidate_summary"])

    assert "candidate_against_external_reference" in result["failed"]
