"""Config schema, ``base:`` chains, ``--set`` overrides and digest (SPEC §13; the schema half of T-F1)."""
from __future__ import annotations

import json
import sys

import pytest

from teb_vae.classifier import config

DEFAULT = config.DEFAULT_CONFIG


def test_digest_is_stable_ignores_execution_knobs_and_tracks_settings():
    base = config.digest(config.load(DEFAULT))
    assert len(base) == 64 and base == config.digest(config.load(DEFAULT))
    assert base == config.digest(config.load(
        DEFAULT, ["classifier.run.device=cpu", "classifier.run.num_workers=0", "classifier.run.report_workers=0"]))
    assert base != config.digest(config.load(DEFAULT, ["classifier.labels.horizon_h=2"]))
    assert base != config.digest(config.load(DEFAULT, ["advanced_config.trainer.precision=bf16"]))


def test_a_run_dir_of_another_schema_version_is_refused(tmp_path, monkeypatch):
    """A run dir written under an older run-dir schema (e.g. no ``patient`` column) never resumes."""
    from teb_vae.classifier import run

    cfg = config.load(DEFAULT)
    monkeypatch.setattr(config, "SCHEMA_VERSION", config.SCHEMA_VERSION - 1)
    old = config.digest(cfg)
    monkeypatch.undo()
    assert old != config.digest(cfg)
    (tmp_path / "manifest.json").write_text(json.dumps({"config_digest": old}))
    try:
        with pytest.raises(ValueError, match="different config"):
            run.main(config=str(DEFAULT), stage="cohort", run_dir=str(tmp_path))
    finally:  # run.main pointed loguru's sinks at tmp_path
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr)


def test_landmark_axis_is_evaluation_only_and_left_out_of_the_digest():
    """``rel_second_stage_norm`` (§11.5) is no threshold basis; ``eval.norm_bin`` must divide 1; neither enters the
    digest, so a run dir made before the axis existed still resumes and re-evaluates."""
    with pytest.raises(ValueError, match="evaluation only"):
        config.load(DEFAULT, ["classifier.eval.thresholds=[{id: np30, policy: fpr_cap, alpha: 0.3, method: empirical, "
                              "basis: committed_overall, axis: rel_second_stage_norm, at: 0.5}]"])
    with pytest.raises(ValueError, match="norm_bin must divide 1"):
        config.load(DEFAULT, ["classifier.eval.norm_bin=0.3"])
    assert config.load(DEFAULT, ["classifier.eval.norm_bin=0.2"]).classifier.eval.norm_bin == 0.2
    assert config.digest(config.load(DEFAULT)) == config.digest(config.load(DEFAULT, [
        "classifier.eval.norm_bin=0.5",
        "classifier.eval.time_axes=[to_delivery, from_onset, rel_second_stage, position, elapsed]"]))
