r"""The shipped configs load and validate cleanly, and keep the causal guards on.

``validate_config`` is not a schema (only nine keys are required, unknown keys merely warn), so a
green validator is necessary and not sufficient; the warning-list assertion is what closes the gap
for unknown or dead keys. Each config is resolved through its ``base:`` chain first, which is the
only way it ever reaches the experiment driver.

Two values are pinned because a silent revert of either removes the model's causal standing with
nothing else failing: ``causal_norm`` (off, the prior conditions on the target's future) and
``causal_reach_budget_s`` (``null``, the stored two-sided features leak the future into the target
branch). The loader's field list is pinned as a data contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from train.test_utils import make_graph_model

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"


@pytest.fixture
def loguru_warnings():
    """Collect the validator's warnings.

    ``validate_config`` reports an unknown or dead key through loguru, not the stdlib
    ``warnings`` module, so a ``pytest.warns`` assertion against it would pass no matter what
    the config contained.
    """
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def shipped():
    return load_config(str(_CONFIG))


@pytest.mark.parametrize("config_path", [_CONFIG, _TINY], ids=["default", "tiny"])
def test_the_resolved_config_validates_with_no_unknown_or_dead_key_warnings(
    config_path, tmp_path, loguru_warnings
):
    """Drives the framework's real validator, not a copy of its rules. Building the graph model
    also reads the keys ``GraphModelBase.__init__`` indexes before the validator runs."""
    resolved = resolve_config_file(str(config_path), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [m for m in loguru_warnings if "config:" in m] == []


def test_the_shipped_config_keeps_the_causal_guards_on(shipped):
    r"""With ``causal_norm`` off the prior conditions on the future and the KL is not a coupling
    readout; with ``causal_reach_budget_s`` at ``null`` the target branch reads its own future
    through the two-sided features and $D_{\mathrm{base}}$ is measured through that leak. Presence
    is asserted because ``null`` and *absent* look identical to a ``.get``."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["causal_norm"] is True
    assert "causal_reach_budget_s" in vae
    assert vae["causal_reach_budget_s"] is not None


def test_load_fields_covers_what_the_model_and_plots_read(shipped):
    load_fields = set(
        shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    )
    # The six the model consumes: the raw target, two target streams, two source streams, and
    # the validity mask.
    assert {"fhr", "fhr_st", "fhr_ph", "up_st", "up_ph", "weight"} <= load_fields
    # The two the diagnostic plots need.
    assert {"up", "guid"} <= load_fields
    # Classifier-era fields this model never reads, and the cross-channel block, which mixes both
    # signals in one coefficient and would destroy the target-only prior's separation.
    assert load_fields.isdisjoint({"target", "epoch", "cs_label", "bg_label", "fhr_up_ph"})
