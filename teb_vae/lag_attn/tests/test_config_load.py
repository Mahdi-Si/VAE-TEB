r"""The shipped configs resolve and pass the framework's validator with no unknown or dead key.

A config key is only real if some code reads it. The tree this model was ported from accumulated
keys that nothing consumed -- a plotting block whose ``enabled: false`` disabled nothing, a
``checkpoint_frequency`` no module read -- and each one reads to a maintainer as a control that
exists. ``validate_config`` is not a schema (only a handful of keys are required, unknown keys merely
warn), so the assertion is on its warning list, not on the absence of an exception.

What each config *builds* is checked elsewhere: the model kwargs in ``test_trainer.py`` and one real
fit of the smoke variant in ``test_train_smoke.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from teb_vae.lag_attn.config import resolve_config_file
from train.test_utils import make_graph_model

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def loguru_warnings():
    """Collect the validator's warnings.

    ``validate_config`` reports an unknown or dead key through loguru, not the stdlib ``warnings``
    module, so a ``pytest.warns`` or ``caplog`` assertion against it would pass no matter what the
    config contained.
    """
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.mark.parametrize("name", ["default.yaml", "tiny.yaml"])
def test_every_shipped_config_validates_with_no_unknown_or_dead_key_warnings(
    name, tmp_path, loguru_warnings
):
    """Drives the framework's real validator, not a copy of its rules.

    Resolved first, which is the only way a config ever reaches the experiment driver: the driver
    reads a path and does not know about ``base:``, so handed the raw smoke variant it would see a
    config missing almost every required key. Building the driver also reads the keys the
    framework indexes bare in its constructor, which the validator itself never checks.
    """
    resolved = resolve_config_file(str(_CONFIG_DIR / name), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [m for m in loguru_warnings if "config:" in m] == []
