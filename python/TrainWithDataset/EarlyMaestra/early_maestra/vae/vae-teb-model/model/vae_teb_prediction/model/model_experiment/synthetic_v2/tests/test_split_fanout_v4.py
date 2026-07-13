r"""S7-T03: split fan-out + split-scoped output dirs for the ``synthetic_v4`` driver.

A split-scoped stage (``eval`` / ``test_plots`` / ``report``) must fan out over every cached split
and write each split's artifacts into a distinct ``results/<tag>/<arm>/<split>/`` directory, while a
split-independent stage (``train`` / model-free) dispatches once. These tests exercise the driver's
fan-out mechanism directly with a marker-writing stub -- no trained model needed -- so they prove
the "distinct per-split, no collision" contract cheaply; the full eval-over-splits path is covered
by the Sprint-7 pipeline smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v4 as rp

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


def _args(**over) -> argparse.Namespace:
    r"""A CLI-equivalent namespace with the driver's expected fields."""
    base = dict(config=str(_CONFIG_PATH), stage=None, arm=None, pilot=True,
                split=None, dry_run=False)
    base.update(over)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# _resolve_splits
# ---------------------------------------------------------------------------
def test_resolve_splits_discovers_cached(tiny_cache_v4) -> None:
    r"""``split=None`` returns exactly the splits whose ``.npz`` exists, in canonical order."""
    config = tiny_cache_v4["config"]
    got = rp._resolve_splits(config, "G1_raw_v4", None)
    cache_dir = tiny_cache_v4["cache_dir"]
    expected = [s for s in ("train", "val", "test") if (cache_dir / f"{s}.npz").is_file()]
    assert got == expected
    assert len(got) >= 2  # the tiny cache builds train/val/test


def test_resolve_splits_explicit_restricts(tiny_cache_v4) -> None:
    r"""An explicit split restricts to that one (even if others are present)."""
    assert rp._resolve_splits(tiny_cache_v4["config"], "G1_raw_v4", "test") == ["test"]


def test_resolve_splits_fallback_when_no_cache() -> None:
    r"""With no resolvable cache the fan-out falls back to ``["val"]`` (the runner default)."""
    empty = {"experiment": {"tag": "nope", "data_tag": "nope"}, "paths": {"data_dir": "/nonexistent"}}
    assert rp._resolve_splits(empty, "G1_raw_v4", None) == ["val"]


# ---------------------------------------------------------------------------
# StageContextV4.output_dir() split-scoping
# ---------------------------------------------------------------------------
def test_output_dir_split_scoped() -> None:
    r"""``output_dir`` appends the split; ``run_dir`` (checkpoint root) never does."""
    config = rp.load_config(str(_CONFIG_PATH))
    ctx_val = rp.StageContextV4(config=config, benchmark="G1_raw_v4", arm="prod", split="val")
    ctx_none = rp.StageContextV4(config=config, benchmark="G1_raw_v4", arm="prod", split=None)
    assert ctx_val.output_dir() == ctx_val.run_dir() / "val"
    assert ctx_none.output_dir() == ctx_none.run_dir()
    # The checkpoint root is split-independent under both.
    assert ctx_val.run_dir() == ctx_none.run_dir()


# ---------------------------------------------------------------------------
# Driver fan-out with a marker-writing split-scoped stub
# ---------------------------------------------------------------------------
_DISPATCHED: List[str] = []


def _marker_run(ctx: rp.StageContextV4) -> int:
    out = ctx.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "marker.txt").write_text(str(ctx.split), encoding="utf-8")
    _DISPATCHED.append(str(ctx.split))
    return 0


@pytest.fixture
def split_scoped_stub(monkeypatch) -> Iterator[str]:
    r"""Register a marker-writing stub and mark it split-scoped for the duration of the test."""
    name = "_splitscoped_stub_v4"
    _DISPATCHED.clear()
    rp.register_stage_v4(rp.StageSpecV4(
        name=name, run=_marker_run, order=997, model_dependent=True, help="split stub"))
    monkeypatch.setattr(rp, "_SPLIT_SCOPED_STAGES", frozenset({name}))
    try:
        yield name
    finally:
        rp._STAGE_REGISTRY_V4.pop(name, None)


def test_fan_out_writes_distinct_per_split_dirs(tiny_cache_v4, split_scoped_stub) -> None:
    r"""A split-scoped stage writes one marker per cached split, each in its own dir (no collision)."""
    config = tiny_cache_v4["config"]
    spec = rp._STAGE_REGISTRY_V4[split_scoped_stub]
    rc = rp._dispatch_for_arm(config, "G1_raw_v4", spec, "prod", _args())
    assert rc == 0

    splits = rp._resolve_splits(config, "G1_raw_v4", None)
    assert sorted(_DISPATCHED) == sorted(splits)
    run_dir = rp._run_dir(config, "G1_raw_v4", "prod")
    for s in splits:
        marker = run_dir / s / "marker.txt"
        assert marker.is_file(), f"missing per-split marker for {s}"
        assert marker.read_text(encoding="utf-8") == s


def test_split_independent_stage_dispatches_once(tiny_cache_v4) -> None:
    r"""A non-split-scoped stage dispatches once at the arm root, even with splits present."""
    name = "_notsplit_stub_v4"
    seen: List[str] = []
    rp.register_stage_v4(rp.StageSpecV4(
        name=name, run=lambda ctx: seen.append(str(ctx.split)) or 0,
        order=996, model_dependent=True, help="non-split stub"))
    try:
        spec = rp._STAGE_REGISTRY_V4[name]
        rc = rp._dispatch_for_arm(tiny_cache_v4["config"], "G1_raw_v4", spec, "prod", _args())
    finally:
        rp._STAGE_REGISTRY_V4.pop(name, None)
    assert rc == 0
    assert seen == ["None"]  # single dispatch, split not fanned out
