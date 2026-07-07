r"""Regression tests for per-split checkpoint resolution (run-root, not ``out_dir``).

The train stage writes ``best.ckpt`` / ``final.ckpt`` into the **run root**
``results/<tag>/``, while :func:`run_pipeline_v2._eval_splits` and
:func:`run_pipeline_v2._test_plots_splits` grade each split into its **own**
``results/<tag>/<split>/`` ``out_dir``. A prior bug forwarded that per-split ``out_dir``
as the checkpoint-discovery root, so eval raised ``no best.ckpt / final.ckpt under
.../train``. These tests pin the fix: the checkpoint is resolved once from the run root
and handed to every split as an explicit path.

The heavyweight ``run_eval`` / ``run_test_plots`` bodies are stubbed (monkeypatched) so
the test exercises only the orchestration/resolution logic on CPU with no model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    eval_v2,
    run_pipeline_v2 as rp,
)


def _touch_ckpt(run_root: Path, name: str) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    p = run_root / name
    p.write_bytes(b"")  # _resolve_run_ckpt only checks .is_file()
    return p


def test_eval_splits_resolves_ckpt_from_run_root(tmp_path, monkeypatch):
    r"""``_eval_splits`` hands each split the run-root ``best.ckpt`` and a per-split out_dir."""
    run_root = tmp_path / "results" / "G1_raw_v2_notch"
    ckpt = _touch_ckpt(run_root, "best.ckpt")

    calls: List[Tuple[str, str, Path]] = []

    def _fake_run_eval(config, *, benchmark, ckpt, split, out_dir, **_):
        calls.append((ckpt, split, Path(out_dir)))
        return {}

    # ``_eval_splits`` does a local ``from ...eval_v2 import run_eval`` at call time, so
    # patching the attribute on the eval_v2 module is what the local import picks up.
    monkeypatch.setattr(eval_v2, "run_eval", _fake_run_eval)

    rp._eval_splits(
        {}, "G1_raw_v2_notch", ckpt=None, splits=["train", "val", "test"],
        results_dir=run_root,
    )

    assert [c[1] for c in calls] == ["train", "val", "test"]
    for got_ckpt, split, got_out in calls:
        assert Path(got_ckpt) == ckpt              # run-root checkpoint, not None
        assert got_out == run_root / split          # per-split output dir preserved


def test_eval_splits_prefers_best_over_final(tmp_path, monkeypatch):
    r"""When both exist, ``best.ckpt`` wins over ``final.ckpt`` (resolver order)."""
    run_root = tmp_path / "results" / "tag"
    _touch_ckpt(run_root, "final.ckpt")
    best = _touch_ckpt(run_root, "best.ckpt")

    seen: Dict[str, str] = {}

    def _fake_run_eval(config, *, benchmark, ckpt, split, out_dir, **_):
        seen["ckpt"] = ckpt
        return {}

    monkeypatch.setattr(eval_v2, "run_eval", _fake_run_eval)
    rp._eval_splits({}, "tag", ckpt=None, splits=["test"], results_dir=run_root)
    assert Path(seen["ckpt"]) == best


def test_eval_splits_missing_ckpt_raises_pointing_at_run_root(tmp_path, monkeypatch):
    r"""No checkpoint anywhere -> fatal FileNotFoundError naming the run root."""
    run_root = tmp_path / "results" / "tag"
    run_root.mkdir(parents=True)
    monkeypatch.setattr(
        eval_v2, "run_eval",
        lambda *a, **k: pytest.fail("run_eval must not be reached without a checkpoint"),
    )
    with pytest.raises(FileNotFoundError) as exc:
        rp._eval_splits({}, "tag", ckpt=None, splits=["test"], results_dir=run_root)
    assert str(run_root) in str(exc.value)


def test_test_plots_splits_resolves_ckpt_from_run_root(tmp_path, monkeypatch):
    r"""``_test_plots_splits`` passes the run-root checkpoint + per-split out_dir."""
    run_root = tmp_path / "results" / "tag"
    ckpt = _touch_ckpt(run_root, "final.ckpt")

    calls: List[Tuple[str, str, Path]] = []
    monkeypatch.setattr(
        rp, "run_test_plots",
        lambda config, *, benchmark, ckpt, split, analysis_samples, out_dir, **_:
            calls.append((ckpt, split, Path(out_dir))),
    )
    rp._test_plots_splits(
        {}, "tag", ckpt=None, analysis_samples=2, splits=["train", "test"],
        results_dir=run_root,
    )
    assert [c[1] for c in calls] == ["train", "test"]
    for got_ckpt, split, got_out in calls:
        assert Path(got_ckpt) == ckpt
        assert got_out == run_root / split


def test_test_plots_splits_missing_ckpt_is_nonfatal(tmp_path, monkeypatch):
    r"""A missing checkpoint only warns for the diagnostics stage (never raises)."""
    run_root = tmp_path / "results" / "tag"
    run_root.mkdir(parents=True)
    monkeypatch.setattr(
        rp, "run_test_plots",
        lambda *a, **k: pytest.fail("run_test_plots must not run without a checkpoint"),
    )
    # Must return without raising even though no checkpoint exists.
    rp._test_plots_splits(
        {}, "tag", ckpt=None, analysis_samples=1, splits=["test"], results_dir=run_root,
    )
