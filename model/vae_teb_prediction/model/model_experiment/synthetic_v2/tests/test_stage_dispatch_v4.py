r"""S0-T05: ``run_pipeline_v4`` skeleton -- parser, registry, single-stage dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v4 as rp

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"

#: Records whether the stub stage ran, and with what context, across a dispatch.
_CALLS: list = []


def _stub_run(ctx) -> int:
    _CALLS.append(ctx)
    return 0


@pytest.fixture
def stub_stage() -> Iterator[str]:
    r"""Register a throwaway stub stage and remove it afterwards (registry is module-global)."""
    name = "_stub_v4_test"
    _CALLS.clear()
    rp.register_stage_v4(rp.StageSpecV4(name=name, run=_stub_run, order=999, help="test stub"))
    try:
        yield name
    finally:
        rp._STAGE_REGISTRY_V4.pop(name, None)


def test_help_lists_stage_option() -> None:
    r"""``--help`` runs and advertises the ``--stage`` option."""
    parser = rp.build_parser()
    help_text = parser.format_help()
    assert "--stage" in help_text


def test_no_stage_prints_help_and_returns_zero(capsys) -> None:
    r"""With no ``--stage`` the driver prints help and exits 0 (the runnable default)."""
    rc = rp.main(["--config", str(_CONFIG_PATH)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--stage" in out


def test_registered_stub_stage_is_called(stub_stage: str) -> None:
    r"""Dispatching a registered stage invokes its runner with a resolved context."""
    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", stub_stage])
    assert rc == 0
    assert len(_CALLS) == 1
    ctx = _CALLS[0]
    assert ctx.benchmark == "G1_raw_v4"
    assert isinstance(ctx.config, dict)


def test_unregistered_stage_errors_cleanly() -> None:
    r"""An unregistered ``--stage`` is rejected by argparse (clean non-zero exit)."""
    with pytest.raises(SystemExit) as excinfo:
        rp.main(["--config", str(_CONFIG_PATH), "--stage", "definitely_not_a_stage"])
    assert excinfo.value.code != 0


def test_stage_registry_is_forked_from_v2() -> None:
    r"""The v4 registry is a SEPARATE object from the v2 one (no cross-contamination)."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2

    assert rp._STAGE_REGISTRY_V4 is not run_pipeline_v2._STAGE_REGISTRY


def test_duplicate_registration_raises(stub_stage: str) -> None:
    r"""Registering the same stage name twice raises (mirrors the v2 registry contract)."""
    with pytest.raises(ValueError):
        rp.register_stage_v4(rp.StageSpecV4(name=stub_stage, run=_stub_run))
