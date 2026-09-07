r"""The run directory: what it records, what may be re-run into it, and what the lock forbids.

Small files and dictionaries throughout. Nothing here builds a model, opens a checkpoint or reads a
shard: a stage record is JSON, a protocol record is YAML, and the resume decision is a comparison
between the two and a request. Run directories are created under this package's own ``runs/`` tree,
which is what :func:`config.run_directory` requires and what ``.gitignore`` covers, and each test
removes its own.

The pilot checkpoint and the base-model export write tensors and are checked separately, on the
execution machine.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import config as pilot_config
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import (
    PilotConfigError,
    RunStateError,
)

#: Where these tests put their run directories. Inside the package, because run containment is one
#: of the rules under test; under ``runs/``, because that is the ignored tree.
SCRATCH = pilot_config.PILOT_ROOT / "runs" / "logic_tests"


@pytest.fixture
def scratch():
    """A run root of this test's own, removed afterwards."""
    directory = SCRATCH / uuid.uuid4().hex[:8]
    directory.mkdir(parents=True, exist_ok=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        # Left exactly as found: the two directories above are removed when this test was the only
        # thing in them.
        for parent in (SCRATCH, SCRATCH.parent):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()


#: The shipped production template. Resolved here for its defaults only -- these tests never call
#: ``require_inputs``, so its unfilled checkpoint and shard lists are exactly right.
TEMPLATE = pilot_config.PILOT_ROOT / "configs" / "pilot.yaml"


def _settings(scratch, **overrides):
    """Resolved settings pointing their run root at the scratch directory."""
    return pilot_config.resolve_settings(
        TEMPLATE, overrides={"paths": {"run_root": str(scratch)}, **overrides}
    )


def _open(scratch, stages=("preflight", "extract"), **kwargs):
    """Open a run at this file's settings."""
    settings = kwargs.pop("settings", None) or _settings(scratch)
    return pilot_config.open_run(
        settings, run_args={"stage": "all"}, stages=list(stages), **kwargs
    )


# =============================================================================
# A new run
# =============================================================================
def test_a_new_run_creates_its_directory_and_writes_what_it_was_launched_with(scratch):
    opened = _open(scratch)

    assert opened["run_dir"].is_dir()
    assert not opened["resumed"]
    assert opened["stages"] == ["preflight", "extract"]
    assert opened["skipped"] == []

    protocol = pilot_config.read_protocol(opened["run_dir"])
    assert protocol["run_id"] == opened["run_id"]
    assert protocol["settings_digest"] == pilot_config.settings_digest(
        _settings(scratch)
    )
    assert pilot_config.read_stage_state(opened["run_dir"])["completed"] == []


def test_a_run_directory_outside_the_package_is_refused(scratch):
    """Runtime output stays in the pilot folder whatever the settings say."""
    with pytest.raises(PilotConfigError, match="outside this package"):
        _open(scratch, run_dir="/tmp/not-in-the-pilot-folder")


def test_naming_a_directory_that_does_not_exist_is_not_a_new_run(scratch):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _open(scratch, run_dir=scratch / "never-created")


def test_a_directory_without_a_protocol_is_not_a_pilot_run(scratch):
    (scratch / "stray").mkdir()

    with pytest.raises(RunStateError, match="not a pilot run directory"):
        _open(scratch, run_dir=scratch / "stray")


# =============================================================================
# Stage state
# =============================================================================
def test_completed_and_failed_stages_are_persisted_as_they_happen(scratch):
    opened = _open(scratch)
    directory, state = opened["run_dir"], opened["state"]

    pilot_config.mark_completed(state, "preflight", directory)
    assert pilot_config.read_stage_state(directory)["completed"] == ["preflight"]

    pilot_config.mark_failed(state, "extract", directory, reason="shards unmounted")
    stored = pilot_config.read_stage_state(directory)
    assert stored["failed"]["stage"] == "extract"
    assert "unmounted" in stored["failed"]["reason"]

    # Completing it clears the failure rather than leaving a stale one behind.
    pilot_config.mark_completed(state, "extract", directory)
    assert pilot_config.read_stage_state(directory)["failed"] is None


def test_a_stage_whose_prerequisite_never_ran_is_refused():
    """The runner never silently refits a missing input on the way to a report."""
    state = pilot_config.new_stage_state(pilot_config.STAGES)
    state["completed"] = ["extract"]

    pilot_config.require_completed(state, "extract", needed_by="baseline")
    with pytest.raises(RunStateError, match="needs 'finetune'"):
        pilot_config.require_completed(state, "finetune", needed_by="evaluate")


# =============================================================================
# Resume
# =============================================================================
def test_resuming_skips_what_is_finished_and_runs_the_rest(scratch):
    opened = _open(scratch, stages=("preflight", "extract", "baseline"))
    pilot_config.mark_completed(opened["state"], "preflight", opened["run_dir"])
    pilot_config.mark_completed(opened["state"], "extract", opened["run_dir"])

    resumed = _open(
        scratch,
        stages=("preflight", "extract", "baseline"),
        run_dir=opened["run_dir"],
        resume=True,
    )

    assert resumed["resumed"]
    assert resumed["skipped"] == ["preflight", "extract"]
    assert resumed["stages"] == ["baseline"]
    assert resumed["run_id"] == opened["run_id"]


def test_resuming_under_changed_settings_is_refused_and_names_what_changed(scratch):
    opened = _open(scratch)

    with pytest.raises(RunStateError, match="optim.max_epochs"):
        _open(
            scratch,
            settings=_settings(scratch, optim={"max_epochs": 3}),
            run_dir=opened["run_dir"],
            resume=True,
        )


def test_the_settings_difference_names_the_dotted_path_and_both_values():
    left = {"optim": {"max_epochs": 10, "patience": 3}, "seed": 42}
    right = {"optim": {"max_epochs": 3, "patience": 3}, "seed": 42}

    assert pilot_config.settings_differences(left, right) == {
        "optim.max_epochs": (10, 3)
    }
    assert pilot_config.settings_differences(left, left) == {}


# =============================================================================
# Re-entering a finished run without resuming
# =============================================================================
def test_a_finished_fitting_stage_is_not_overwritten_in_place(scratch):
    """Re-running it would leave a directory whose report describes a model it no longer holds."""
    opened = _open(scratch, stages=("extract", "baseline"))
    pilot_config.mark_completed(opened["state"], "baseline", opened["run_dir"])

    with pytest.raises(RunStateError, match="resume=True"):
        _open(scratch, stages=("baseline",), run_dir=opened["run_dir"])


def test_a_finished_run_can_still_be_re_reported(scratch):
    """§12's own instruction: regenerate presentation from saved artifacts without refitting."""
    opened = _open(scratch, stages=("extract", "report"))
    for stage in ("extract", "report"):
        pilot_config.mark_completed(opened["state"], stage, opened["run_dir"])

    again = _open(scratch, stages=("report",), run_dir=opened["run_dir"])

    assert again["stages"] == ["report"]
    assert not again["resumed"]


def test_every_rerunnable_stage_leaves_fitted_artifacts_alone():
    """The list is a claim about what those stages write, so it is stated once and read here."""
    assert set(pilot_config.RERUNNABLE_STAGES) <= set(pilot_config.STAGES)
    for stage in ("extract", "baseline", "finetune", "control", "evaluate"):
        assert stage not in pilot_config.RERUNNABLE_STAGES


# =============================================================================
# The selection lock
# =============================================================================
def test_the_test_split_cannot_be_opened_before_the_selection_is_locked(scratch):
    opened = _open(scratch)

    with pytest.raises(RunStateError, match="has not locked its selection"):
        pilot_config.require_selection_locked(opened["run_dir"])

    pilot_config.lock_selection(opened["run_dir"], {"selected_epoch": 2, "threshold": 0.3})

    assert pilot_config.require_selection_locked(opened["run_dir"]) is True


def test_the_lock_records_the_choices_and_when_they_were_fixed(scratch):
    opened = _open(scratch)
    pilot_config.lock_selection(
        opened["run_dir"],
        {"selected_epoch": 2, "threshold": 0.3, "seeds": {"run": 42}, "gates": {"mse": 0.1}},
    )

    locked = pilot_config.read_selection_lock(opened["run_dir"])

    assert locked["selected_epoch"] == 2
    assert locked["threshold"] == 0.3
    assert locked["seeds"] == {"run": 42}
    assert "locked_utc" in locked
    assert pilot_config.read_stage_state(opened["run_dir"])["selection_locked"] is True


def test_a_run_cannot_be_locked_twice(scratch):
    """A second lock is a choice made after the held-out split was available."""
    opened = _open(scratch)
    pilot_config.lock_selection(opened["run_dir"], {"selected_epoch": 0})

    with pytest.raises(RunStateError, match="cannot be re-locked"):
        pilot_config.lock_selection(opened["run_dir"], {"selected_epoch": 5})

    # And the first record is the one that stands.
    assert pilot_config.read_selection_lock(opened["run_dir"])["selected_epoch"] == 0
