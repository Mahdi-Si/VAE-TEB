r"""S8-T05: the README documents the v3 surface a reader would otherwise have to grep for.

A grep test, deliberately. Prose quality is a separate documentation review; what a test can pin is
that every flag, stage and config key the v3 path introduces is *named* somewhere in the README, so
nobody has to read `run_pipeline_v2.py` to discover that ``--stage cmi`` exists.

It also guards the two claims a reader can get catastrophically wrong: that the KL-space null
control should approach zero (it should not -- Finding F2), and that ``gamma_scat`` is a calibrated
slope (it is not -- S1-T05).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SV2 = Path(__file__).resolve().parents[1]
_README = _SV2 / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    assert _README.is_file(), f"missing {_README}"
    return _README.read_text(encoding="utf-8")


#: The seven tokens S8-T05 requires, plus the config file that carries them.
_REQUIRED_TOKENS = (
    "--arm",
    "--max-samples",
    "calibration",
    "lag_intervention",
    "cmi",
    "arms_report",
    "data_tag",
    "config_synth_v3.yaml",
)


@pytest.mark.parametrize("token", _REQUIRED_TOKENS)
def test_readme_names_the_v3_surface(readme: str, token: str) -> None:
    assert token in readme, f"README does not mention {token!r}"


def test_readme_names_all_three_arms(readme: str) -> None:
    for arm in ("parity", "v3_noncausal", "v3_prod"):
        assert f"`{arm}`" in readme, arm
    assert "causal_norm" in readme


def test_readme_describes_the_arm_scoped_output_tree(readme: str) -> None:
    r"""The arm level sits between the tag and the split, and only for model-dependent artifacts."""
    assert "results/<tag>/<arm>/<split>/" in readme
    assert "arms_report.md" in readme
    # The model-free artifacts stay at the tag root.
    assert "realizability.json" in readme


def test_readme_says_the_kl_null_control_is_a_readout_not_a_gate(readme: str) -> None:
    r"""The one caption a reader could invert the conclusion from."""
    lowered = readme.lower()
    assert "readout, not a gate" in lowered
    assert "finding f2" in lowered
    assert "not**\n> expected to approach `0`" in readme or "expected to approach `0`" in readme
    # ..and the gate that replaces it is stated.
    assert "L_feat < L_base < L_feat^π(U)" in readme


def test_readme_says_te_scat_is_ordinal_within_a_fixed_lag(readme: str) -> None:
    r"""S1-T05: `gamma_scat` is neither a calibrated slope nor a gate."""
    assert "ordinal within a fixed lag only" in readme
    assert "S1-T05" in readme
    assert "not** a calibrated slope" in readme


def test_readme_does_not_promise_a_stage_the_driver_lacks(readme: str) -> None:
    r"""Every ``--stage X`` the README names must be a registered stage."""
    import re

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    rp._load_stage_plugins()
    registered = set(rp.stage_names())
    named = set(re.findall(r"--stage ([a-z_0-9]+)", readme))
    assert named, "the README names no stages at all"
    assert named <= registered, f"README names unregistered stages: {sorted(named - registered)}"


def test_readme_names_every_registered_cli_stage(readme: str) -> None:
    r"""The converse: a stage a user can run must be discoverable from the README."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    rp._load_stage_plugins()
    for name in rp.stage_names():
        assert f"`{name}`" in readme, f"stage {name!r} is undocumented"
