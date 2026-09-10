r"""One real fit, then one real evaluation of it, then the gate that reads what it wrote.

Everything else in this suite tests a piece against constructed tensors. This runs the whole
evaluation path against a checkpoint a fit actually produced: the override merge over that run's own
resolved configuration, the loader, the rebuild through the checkpoint contract, the dense forward
with its proposals retained, every intervened arm, one draw loop, the recording-level bootstrap, the
summary, and the acceptance gate.

It is the only place the failures that live *between* those pieces can surface, and the ones worth
naming are all of the same kind: a seam that is individually correct and jointly wrong. A checkpoint
whose keys carry a task prefix the rebuild does not strip. A band read from a configuration whose
lag window the fit never had. A margin taken against an arm that was scored under different noise. A
summary whose numbers are fine and whose provenance block says nothing.

**The reference identities are the assertions to read first.** Suppressing an empty band must
reproduce the matched arm exactly, and suppressing every band must reproduce the target-only prior
exactly. They hold here on real weights and a real split, which is what makes every other margin in
the file a difference of predictions rather than a difference between two code paths.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.tests.conftest import absolutize_dataset_paths
from teb_vae.lag_slot_transformer_cfs import trainer as trainer_module
from teb_vae.lag_slot_transformer_cfs.eval import run as eval_run
from teb_vae.lag_slot_transformer_cfs.eval import verify as eval_verify
from teb_vae.lag_slot_transformer_cfs.trainer import LagResidualTrfCfsTrainer

pytestmark = pytest.mark.slow

#: The configuration the fit runs.
TINY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Epochs the fit runs. Two rather than one, matching the training smoke: the second is what steps
#: the scheduler and rotates the tile phase, and a checkpoint from a model that never left its
#: initialisation would make every source margin identically zero for a reason about the fit.
SMOKE_EPOCHS = 2

#: Draws the evaluation scores at. Small: this asserts the shape of what the pass produces, not the
#: convergence of an estimator, and the draw count is recorded in the summary either way.
SMOKE_DRAWS = 3

#: Bootstrap resamples. The schema's floor, because the fixture holds a handful of recordings and a
#: wider interval is not made narrower by resampling it more.
SMOKE_RESAMPLES = 100


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """Fit the tiny configuration, evaluate the checkpoint, and hand back the summary.

    Module-scoped because the fit and the pass are the expensive part of this file and every
    assertion below reads the same run.

    Args:
        tmp_path_factory: pytest's directory factory.

    Returns:
        ``(summary, summary_path, exit_code)``.
    """
    tmp_path = tmp_path_factory.mktemp("eval_smoke")
    config = absolutize_dataset_paths(load_config(str(TINY_CONFIG)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "fit")
    config["general_config"]["epochs"] = SMOKE_EPOCHS
    config["advanced_config"]["trainer"]["profiler"] = None

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    captured = {}
    original = LagResidualTrfCfsTrainer.train_model

    def capture(self, train_loader, validation_loader):
        """Run the inherited fit and keep the driver, which is the only handle on the run."""
        result = original(self, train_loader, validation_loader)
        captured["driver"] = self
        return result

    LagResidualTrfCfsTrainer.train_model = capture
    try:
        trainer_module.main(str(config_path))
    finally:
        del LagResidualTrfCfsTrainer.train_model

    checkpoints = sorted(Path(captured["driver"].train_results_dir).parent.rglob("*.ckpt"))
    assert checkpoints, "the fit wrote no checkpoint"

    # The committed delta names the production shards and the production lag bands, neither of
    # which the tiny geometry has. A run-specific delta is what an operator would write for any
    # other split, so writing one here exercises the merge rather than bypassing it.
    delta = {
        "general_config": {"batch_size": {"test": 2}},
        "dataset_config": {
            "vae_test_datasets": config["dataset_config"]["vae_test_datasets"],
            "stat_path": config["dataset_config"]["stat_path"],
        },
        "eval_config": {
            "seed": 7,
            "num_mc_samples": SMOKE_DRAWS,
            "bootstrap_resamples": SMOKE_RESAMPLES,
            "occlusion_bands": {"near": [0, 3], "far": [4, 8]},
        },
    }
    delta_path = tmp_path / "eval_overrides.yaml"
    delta_path.write_text(yaml.safe_dump(delta, sort_keys=False), encoding="utf-8")

    exit_code = eval_run.main(
        checkpoint=str(checkpoints[-1]),
        output_dir=str(tmp_path / "eval"),
        device="cpu",
        overrides=str(delta_path),
        sources={"checkpoint": "cli", "overrides": "cli"},
    )
    summary_path = tmp_path / "eval" / "eval_results" / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8")), summary_path, exit_code


def test_the_pass_completes_and_writes_a_summary(evaluated) -> None:
    """The whole path, from a checkpoint on disk to a summary a reader can open."""
    summary, summary_path, exit_code = evaluated

    assert exit_code == 0
    assert summary_path.is_file()
    assert summary["n_recordings"] > 0
    assert summary["n_segments"] > 0


def test_the_matched_gap_carries_a_recording_level_interval(evaluated) -> None:
    """Recordings, not anchors.

    Consecutive anchors' forecast windows overlap in all but one of their steps at the dense
    geometry, so an interval resampled over anchors would be narrower than the data supports.
    """
    summary, _path, _code = evaluated
    record = summary["headline"]["pred_gap"]

    assert record["method"] == "percentile bootstrap over recordings"
    assert record["n"] == summary["n_recordings"]
    assert record["lo"] <= record["point"] <= record["hi"]
    assert record["resamples"] == SMOKE_RESAMPLES


def test_both_summaries_are_reported_separately(evaluated) -> None:
    """They differ whenever recordings contribute unequal anchor counts, which is always, and a
    run carrying one of them cannot separate a real effect from a length effect."""
    summary, _path, _code = evaluated

    assert "pred_gap" in summary["headline"]
    assert "pred_gap" in summary["anchor_weighted"]


def test_the_reference_arms_are_exact_on_real_weights(evaluated) -> None:
    """The two identities that make every other margin in this file a measurement.

    An empty band removes nothing, so its arm is the matched forward; every band removes every lag,
    so its arm is the target-only prior. Both hold to the floating-point allowance the gate uses.
    """
    summary, _path, _code = evaluated
    bands = summary["lag_readouts"]["band_suppression"]
    gap = summary["headline"]["pred_gap"]["point"]

    assert abs(bands["none"]["margin_nats"]) < eval_verify.EXACT_MARGIN_TOLERANCE
    assert abs(bands["all"]["margin_nats"] - gap) < eval_verify.EXACT_MARGIN_TOLERANCE
    assert (
        abs(summary["source_controls"]["silence_margin_nats"] - gap)
        < eval_verify.EXACT_MARGIN_TOLERANCE
    )


def test_every_band_reports_its_usable_counts_beside_its_margin(evaluated) -> None:
    """A margin without its exposure is a number a reader cannot weigh: a band with nothing to
    remove scores near zero for a reason that is about the availability schedule."""
    summary, _path, _code = evaluated
    bands = summary["lag_readouts"]["band_suppression"]

    assert set(bands) == {"none", "near", "far", "all"}
    for name in ("near", "far"):
        assert bands[name]["band_anchors"] > 0.0
        assert bands[name]["band_channels"] > 0.0
        assert bands[name]["margin_nats"] is not None


def test_the_cancellation_ratio_travels_with_both_of_its_parts(evaluated) -> None:
    """A ratio near zero means either cancellation or a pathway near zero everywhere, and only the
    denominator separates them."""
    summary, _path, _code = evaluated
    cancellation = summary["lag_readouts"]["cancellation"]

    assert set(cancellation) == {"mean", "scale"}
    for channel in cancellation.values():
        assert channel["scored_anchors"] > 0.0
        for part in ("ratio", "numerator", "denominator"):
            assert channel[part] is not None, part


def test_the_exposure_separates_index_support_from_feature_warm_up(evaluated) -> None:
    """Two counts on the lag axis and one on the channel axis.

    A lag can be in range at every anchor and carry a fraction of its channels, and the anchor
    count alone would report it as fully exposed.
    """
    summary, _path, _code = evaluated
    exposure = summary["lag_readouts"]["exposure"]

    assert len(exposure["per_lag_anchors"]) == len(exposure["per_lag_channels"])
    assert all(count > 0 for count in exposure["per_lag_anchors"])
    assert len(exposure["per_source_channel"]) > 0


def test_the_three_source_controls_are_reported_with_their_pairing_counts(evaluated) -> None:
    """The permutation arm's counts are what say it is still a control.

    Counted off the permutation that ran rather than asserted from the way it was drawn: a grouped
    draw that silently stopped grouping looks exactly like one that works.
    """
    summary, _path, _code = evaluated
    controls = summary["source_controls"]

    for margin in (
        "replace_zeros_margin_nats",
        "replace_constant_margin_nats",
        "permute_margin_nats",
    ):
        assert controls[margin] is not None, margin
    assert controls["n_control_pairs"] > 0
    assert controls["n_same_recording_pairs"] == 0


def test_the_calibration_comes_from_the_mixture_for_both_branches(evaluated) -> None:
    """Both, because a calibration statement about the source-conditioned branch alone cannot say
    whether the source improved it or whether the observation model was already miscalibrated."""
    summary, _path, _code = evaluated

    for branch in ("base", "full"):
        block = summary["calibration"][branch]
        assert block["n_coefficients"] > 0
        assert 0.0 <= block["pit_mean"] <= 1.0
        assert set(block["coverage"]) == {"0.5", "0.9", "0.99"}
        # The reference a reader compares against travels with the measurement.
        assert block["uniform_pit_mean"] == 0.5


def test_the_summary_names_the_analyses_this_architecture_cannot_produce(evaluated) -> None:
    """A reader finding fewer columns than a sibling should not have to work out why, nor which
    mechanism left each one out."""
    summary, _path, _code = evaluated
    excluded = summary["excluded_analyses"]
    mechanism = summary["excluded_analyses_mechanism"]

    assert "attention" in excluded and "lag_kl" in excluded
    assert set(mechanism["removed_from_shared_registry"]) == {"attention", "lag_kl"}
    assert mechanism["never_registered_here"]
    for reason in excluded.values():
        assert reason.strip()


def test_no_attention_shaped_key_appears_anywhere_in_the_output(evaluated) -> None:
    """A proposal norm under an attention name is the one claim the design proves cannot be made."""
    summary, _path, _code = evaluated
    keys = set(eval_verify._walk_keys(summary)) - set(summary["excluded_analyses"])

    assert not keys & set(eval_verify.FORBIDDEN_KEYS)


def test_the_run_records_its_own_provenance(evaluated) -> None:
    """A run whose settings are recoverable only from a shell history is a run nobody can repeat."""
    summary, summary_path, _code = evaluated
    record = summary["run"]

    assert record["checkpoint"].endswith(".ckpt")
    assert record["model_kind"] == "fhr_lag_residual_cfs_v1"
    assert record["seed"] == 7
    assert record["argument_sources"]["checkpoint"] == "cli"
    assert record["geometry_keys"]
    # The merged configuration, beside the summary, is what the run actually ran under.
    assert (summary_path.parent / "resolved_config.yaml").is_file()


def test_the_per_recording_table_carries_what_the_intervals_were_built_from(evaluated) -> None:
    """One row per recording, beside the summary.

    The summary carries each column's interval; this carries the values behind it. A protocol
    reading several runs together resamples the recordings ONCE and averages the seeds inside each
    resample, and no interval can be taken apart into the vector that produced it.
    """
    summary, summary_path, _code = evaluated
    table_path = summary_path.parent / eval_run.PER_RECORDING_FILENAME
    rows = list(csv.DictReader(table_path.read_text(encoding="utf-8").splitlines()))

    assert len(rows) == summary["n_recordings"]
    assert {"guid", "nll_base", "nll_full", "pred_gap", "n_scored_anchors"} <= set(rows[0])
    # The equal-recording mean of the table is the point the summary reports, because the summary's
    # point IS that mean: two ways to the same number, and a table built from a different
    # population would be the one place that could go unnoticed.
    gaps = [float(row["pred_gap"]) for row in rows]
    assert sum(gaps) / len(gaps) == pytest.approx(summary["headline"]["pred_gap"]["point"])


def test_the_run_records_which_recordings_it_scored_and_where_they_came_from(evaluated) -> None:
    """The block the reserved-partition question is answered from.

    A confirmation run has to be shown to have scored recordings no run that chose an architecture
    ever saw, and neither a checkpoint nor a metric can say that: the files opened and the
    recordings returned are the whole of the evidence.
    """
    summary, _path, _code = evaluated
    split = summary["scored_split"]

    assert split["shards"] and all(str(path).endswith(".hdf5") for path in split["shards"])
    assert split["stat_path"].endswith(".hdf5")
    assert split["label"]
    assert split["n_recordings"] == summary["n_recordings"]
    assert len(split["recording_digest"]) == 16
    assert split["per_recording_table"] == eval_run.PER_RECORDING_FILENAME
    # Both seeds, because they answer different questions: several runs of one arm are several
    # training seeds scored under one evaluation seed.
    assert summary["run"]["training_seed"] >= 0
    assert summary["run"]["training_tag"]


def test_the_disclosure_states_the_source_reach_and_its_qualification(evaluated) -> None:
    """One stored sample is the additional NEURAL receptive field, and the record says so: the
    feature pipeline upstream still mixes raw history inside every coefficient."""
    summary, _path, _code = evaluated
    disclosure = summary["encoder_disclosure"]

    assert disclosure["source_receptive_field_steps"] == 1
    assert disclosure["searched_lag_steps"] > 1
    assert disclosure["source_encoder_parameters"] == 0
    assert "feature extraction" in disclosure["qualification"]


def test_the_acceptance_gate_passes_on_what_the_run_wrote(evaluated) -> None:
    """The gate reads the artifact rather than the objects that produced it, which is what makes it
    runnable against a summary copied off the box."""
    summary, summary_path, _code = evaluated
    result = eval_verify.verify(summary)

    assert result["failed"] == [], result["failed"]
    assert eval_verify.main(summary=str(summary_path)) == 0
    # And the gap is reported rather than gated: where an acceptable boundary sits is what the
    # first real runs measure, and a guessed threshold would decide it on the run meant to supply
    # the answer.
    gap_verdict = next(
        record for record in result["verdicts"] if record["name"] == "predictive_gap_measured"
    )
    assert gap_verdict["status"] == "INCONCLUSIVE"
    assert gap_verdict["pred_gap_nats"] is not None
