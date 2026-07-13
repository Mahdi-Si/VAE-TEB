r"""S0-T05: ``run_pipeline_v4`` skeleton -- parser, registry, single-stage dispatch."""

from __future__ import annotations

import sys
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


# ===========================================================================
# S7-T01: arm-sweep driver.
# ===========================================================================
@pytest.fixture
def sweep_stub_stage() -> Iterator[str]:
    r"""Register a *model-dependent* stub stage so a no-``--arm`` invocation sweeps the arms."""
    name = "_sweep_stub_v4"
    _CALLS.clear()
    rp.register_stage_v4(rp.StageSpecV4(
        name=name, run=_stub_run, order=998, model_dependent=True, help="sweep stub"))
    try:
        yield name
    finally:
        rp._STAGE_REGISTRY_V4.pop(name, None)


def test_sweep_loops_every_arm(sweep_stub_stage: str) -> None:
    r"""A model-dependent stage with no ``--arm`` dispatches once per configured arm (S7-T01)."""
    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", sweep_stub_stage])
    assert rc == 0
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_v4
    config = rp.load_config(str(_CONFIG_PATH))
    expected = arms_v4.list_arms(config)
    assert len(expected) >= 7  # prod + 6 ablations (incl. am_carrier_prod)
    dispatched = sorted(ctx.arm for ctx in _CALLS)
    assert dispatched == sorted(expected)


def test_sweep_order_is_config_order_prod_first(sweep_stub_stage: str) -> None:
    r"""The sweep iterates arms in config-declared order: direct ``prod`` first, am probe last."""
    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", sweep_stub_stage])
    assert rc == 0
    order = [ctx.arm for ctx in _CALLS]
    assert order[0] == "prod", f"first swept arm must be the direct headline arm, got {order[0]!r}"
    assert order[-1] == "am_carrier_prod", (
        f"the am_carrier probe (separate cache) must sweep last, got {order[-1]!r}")


def test_explicit_arm_dispatches_once(sweep_stub_stage: str) -> None:
    r"""An explicit ``--arm`` bypasses the sweep and dispatches that single arm (subprocess path)."""
    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", sweep_stub_stage, "--arm", "prod"])
    assert rc == 0
    assert [ctx.arm for ctx in _CALLS] == ["prod"]


def test_dry_run_lists_arms_and_dispatches_nothing(sweep_stub_stage: str, capsys) -> None:
    r"""``--dry-run`` prints the per-arm plan (with run dirs) and dispatches no stage (S7-T01)."""
    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", sweep_stub_stage, "--dry-run"])
    assert rc == 0
    assert _CALLS == []  # nothing dispatched
    out = capsys.readouterr().out
    assert "prod" in out and "disable_source" in out
    assert "[plan]" in out


def test_model_free_stage_runs_once_at_root(stub_stage: str) -> None:
    r"""A model-free stage (default ``model_dependent=False``) dispatches once, arm-less."""
    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", stub_stage])
    assert rc == 0
    assert len(_CALLS) == 1
    assert _CALLS[0].arm is None


# ===========================================================================
# S7-T02: DDP-safe subprocess dispatch for ``train``.
# ===========================================================================
def test_subprocess_cmd_is_well_formed() -> None:
    r"""The train subprocess argv re-enters this driver with ``--arm/--config/--stage train``."""
    cmd = rp._stage_subprocess_cmd_v4(_CONFIG_PATH, "train", arm="prod", pilot=True, split="val")
    assert cmd[0] == sys.executable
    assert cmd[1] == "-m" and cmd[2] == rp._QUALNAME
    assert "--stage" in cmd and cmd[cmd.index("--stage") + 1] == "train"
    assert "--config" in cmd and cmd[cmd.index("--config") + 1] == str(_CONFIG_PATH)
    assert "--arm" in cmd and cmd[cmd.index("--arm") + 1] == "prod"
    assert "--pilot" in cmd
    assert "--split" in cmd and cmd[cmd.index("--split") + 1] == "val"


def test_subprocess_cmd_omits_unset_flags() -> None:
    r"""No ``--arm/--pilot/--split`` appears when those are unset (a clean minimal vector)."""
    cmd = rp._stage_subprocess_cmd_v4(_CONFIG_PATH, "train")
    assert "--arm" not in cmd and "--pilot" not in cmd and "--split" not in cmd


def test_train_sweep_dispatches_subprocess_per_arm(monkeypatch) -> None:
    r"""The ``train`` sweep routes each arm through a subprocess (never in-process), one per arm.

    ``train`` is the already-registered model-dependent stage (via the ``trainer_v4`` plugin);
    stubbing :func:`_run_subprocess` records the argv without launching Lightning/DDP.
    """
    if "train" not in rp._STAGE_REGISTRY_V4:
        pytest.skip("train stage not registered (trainer_v4 unavailable)")
    calls: list = []
    monkeypatch.setattr(rp, "_run_subprocess", lambda cmd, **kw: calls.append(list(cmd)) or 0)

    rc = rp.main(["--config", str(_CONFIG_PATH), "--stage", "train", "--pilot"])
    assert rc == 0
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_v4
    n_arms = len(arms_v4.list_arms(rp.load_config(str(_CONFIG_PATH))))
    assert len(calls) == n_arms
    assert all(c[c.index("--stage") + 1] == "train" and "--pilot" in c for c in calls)
    dispatched_arms = sorted(c[c.index("--arm") + 1] for c in calls)
    assert dispatched_arms == sorted(arms_v4.list_arms(rp.load_config(str(_CONFIG_PATH))))
