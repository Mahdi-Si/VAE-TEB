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
