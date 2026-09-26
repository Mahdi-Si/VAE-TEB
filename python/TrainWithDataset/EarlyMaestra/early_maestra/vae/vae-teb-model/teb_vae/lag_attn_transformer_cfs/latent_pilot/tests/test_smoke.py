r"""The whole pipeline, end to end, on artificial data.

**Execution-machine test, and the slowest one here.** It needs the fixtures generated first::

    python -m teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.fixtures.generate
    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_smoke.py -q

**What a green run here proves, and the whole of it:** the stages connect, each one consumes what
the one before it wrote, the artifacts round-trip through their own loaders, the selection lock
comes before the test split is read, and the report renders. **What it does not prove:** anything at
all about latents, outcomes or separation. The identities, the times and the labels are invented;
six recordings per split with three per class is a wiring fixture, not a cohort. No number this run
produces may be quoted, and no test here asserts the sign, magnitude or significance of any effect.

The pipeline is driven one stage at a time through the same ``main`` an operator uses, rather than
through ``all``: ``all`` begins with ``tests`` and ``smoke``, and a smoke scenario that invoked it
would recurse into itself. That is also what makes the resume path load-bearing here -- every stage
after the first continues an existing run directory, which is exactly how an operator runs the
production pipeline in pieces.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import config as pilot_config
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, extract
from teb_vae.lag_attn_transformer_cfs.latent_pilot import report as pilot_report
from teb_vae.lag_attn_transformer_cfs.latent_pilot import run as pilot_run
from teb_vae.lag_attn_transformer_cfs.latent_pilot import train as pilot_train
from teb_vae.lag_attn_transformer_cfs.latent_pilot.tests.conftest import SMOKE_CONFIG

#: The production stages, in the order they must run. ``tests`` and ``smoke`` are deliberately
#: absent: this file *is* the smoke scenario.
PIPELINE_STAGES = (
    "extract", "baseline", "finetune", "control", "evaluate", "report",
)


def _digest(path) -> str:
    """A file's SHA-256, for the source-integrity check."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def source_digest(smoke_fixtures) -> str:
    """The pretrained checkpoint's digest **before** anything runs.

    Taken through a fixture the run depends on, so it is computed first: a digest read afterwards
    would compare the file against itself and pass whatever happened to it.
    """
    return _digest(smoke_fixtures["checkpoint"])


@pytest.fixture(scope="module")
def smoke_run(smoke_fixtures, source_digest):
    """One complete pipeline run over the fixtures; returns its directory.

    Module-scoped: the stages depend on each other's artifacts, and running them twice would
    compare two runs.
    """
    context = pilot_run.main(config_path=str(SMOKE_CONFIG), stage="preflight")
    directory = Path(context["run_dir"])
    for stage in PIPELINE_STAGES:
        pilot_run.main(
            config_path=str(SMOKE_CONFIG), stage=stage, run_dir=str(directory), resume=True
        )
    return directory


# =============================================================================
# Isolation from anything real
# =============================================================================
def test_the_source_checkpoint_survives_the_run_unchanged(
    smoke_fixtures, source_digest, smoke_run
):
    """The adaptation is saved separately; the source checkpoint is opened read-only and stays
    byte-identical. A digest rather than a timestamp, because a rewrite that preserved the mtime
    would still change the bytes -- and the run's own protocol quotes this digest."""
    assert _digest(smoke_fixtures["checkpoint"]) == source_digest


# =============================================================================
# What each stage left behind
# =============================================================================
def test_every_stage_left_its_artifacts(smoke_run):
    """The cohort tables, keyed latents and their scaler, the three fits, the report, and one
    non-empty file for every captioned figure."""
    assert (smoke_run / data.MANIFEST_FILENAME).is_file()
    assert (smoke_run / data.COVERAGE_FILENAME).is_file()

    arrays = sorted(smoke_run.glob(f"*_{extract.LATENT_ARRAYS_FILENAME}"))
    index = sorted(smoke_run.glob(f"*_{extract.LATENT_INDEX_FILENAME}"))
    assert arrays and len(arrays) == len(index)
    assert (smoke_run / extract.SCALER_FILENAME).is_file()

    assert (smoke_run / f"{pilot_train.BASELINE_NAME}_{pilot_train.FIT_FILENAME}").is_file()
    assert (smoke_run / pilot_train.PILOT_CHECKPOINT_FILENAME).is_file()
    assert (smoke_run / f"{pilot_train.CONTROL_NAME}_{pilot_train.FIT_FILENAME}").is_file()

    assert (smoke_run / pilot_report.REPORT_FILENAME).is_file()
    figures = sorted((smoke_run / pilot_report.FIGURE_DIRNAME).glob("*"))
    assert {path.stem for path in figures} == set(pilot_report.CAPTIONS)
    assert all(path.stat().st_size > 0 for path in figures)


# =============================================================================
# Order, and the lock that enforces it
# =============================================================================
def test_every_stage_is_recorded_as_completed_in_run_order(smoke_run):
    state = pilot_config.read_stage_state(smoke_run)
    completed = list(state["completed"])
    assert "preflight" in completed
    for stage in PIPELINE_STAGES:
        assert stage in completed
    positions = [completed.index(stage) for stage in PIPELINE_STAGES]
    assert positions == sorted(positions)
    assert state.get("failed") in (None, "")


def test_the_selection_was_locked_before_the_test_split_was_read(smoke_run):
    """The lock is the mechanism, and its timestamp on disk is the evidence: every test artifact
    must be younger than the lock file."""
    lock = smoke_run / pilot_config.SELECTION_LOCK_FILENAME
    assert lock.is_file()
    assert pilot_config.read_stage_state(smoke_run)["selection_locked"] is True

    record = pilot_config.read_selection_lock(smoke_run)
    assert record["locked_utc"]

    locked_at = lock.stat().st_mtime
    test_artifacts = sorted(smoke_run.glob(f"test*_{extract.LATENT_ARRAYS_FILENAME}"))
    assert test_artifacts, "the evaluation stage extracted no test latents"
    for path in test_artifacts:
        assert path.stat().st_mtime >= locked_at


def test_a_stage_whose_prerequisite_never_ran_is_refused(smoke_fixtures):
    """Asking for a report never silently trains a model, and asking to evaluate never silently
    extracts."""
    context = pilot_run.main(config_path=str(SMOKE_CONFIG), stage="preflight")
    directory = Path(context["run_dir"])
    with pytest.raises(pilot_config.RunStateError):
        pilot_run.main(
            config_path=str(SMOKE_CONFIG), stage="evaluate",
            run_dir=str(directory), resume=True,
        )


def test_resuming_does_not_refit_what_is_already_finished(smoke_run):
    """The report is regenerated; the fitted artifacts are not touched. A rerun that refitted them
    would leave a directory whose report describes a model it no longer holds."""
    checkpoint = smoke_run / pilot_train.PILOT_CHECKPOINT_FILENAME
    report_path = smoke_run / pilot_report.REPORT_FILENAME
    before = _digest(checkpoint)

    pilot_run.main(config_path=str(SMOKE_CONFIG), stage="report", run_dir=str(smoke_run))

    assert _digest(checkpoint) == before
    assert report_path.is_file()


def test_a_finished_fitting_stage_cannot_be_rerun_in_place(smoke_run):
    """Re-running one would leave the later, already-selected artifacts describing a model that no
    longer exists, so it is refused rather than silently redone."""
    with pytest.raises(pilot_config.RunStateError):
        pilot_run.main(
            config_path=str(SMOKE_CONFIG), stage="finetune", run_dir=str(smoke_run)
        )
