r"""S7-T04: headless render tests for the ``visualize_v4`` ground-truth-grading figures.

Every figure must render from a fabricated ``metrics.json`` / ``per_sample_eval.npz`` fixture
(``synth_metrics_v4``) without error and write the requested formats, and must degrade a
missing/empty artifact to a placeholder panel rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v4 as viz

pytestmark = pytest.mark.v4


def _assert_written(paths, tmp_path: Path) -> None:
    assert paths, "no figure files written"
    for p in paths:
        assert Path(p).is_file(), f"declared but missing: {p}"
        assert Path(p).suffix in (".pdf", ".png")


def test_kbar_vs_te_renders(synth_metrics_v4, tmp_path) -> None:
    r"""The headline scatter renders from per-sample arrays + calibration."""
    out = viz.plot_kbar_vs_te_v4(
        synth_metrics_v4["per_sample"], synth_metrics_v4["metrics"], tmp_path / "kbar_vs_te")
    _assert_written(out, tmp_path)


def test_calibration_by_lag_renders(synth_metrics_v4, tmp_path) -> None:
    out = viz.plot_calibration_by_lag_v4(synth_metrics_v4["metrics"], tmp_path / "cal_by_lag")
    _assert_written(out, tmp_path)


def test_lag_recovery_renders(synth_metrics_v4, tmp_path) -> None:
    out = viz.plot_lag_recovery_v4(synth_metrics_v4["metrics"], tmp_path / "lag_recovery")
    _assert_written(out, tmp_path)


def test_pred_control_renders(synth_metrics_v4, tmp_path) -> None:
    out = viz.plot_pred_control_v4(synth_metrics_v4["metrics"], tmp_path / "pred_control")
    _assert_written(out, tmp_path)


def test_kbar_null_bar_renders(synth_metrics_v4, tmp_path) -> None:
    out = viz.plot_kbar_null_bar_v4(synth_metrics_v4["metrics"], tmp_path / "kbar_null")
    _assert_written(out, tmp_path)


def test_figure_specs_render_all(synth_metrics_v4, tmp_path) -> None:
    r"""The ``figure_specs`` registry (driven by the report) renders every figure."""
    specs = viz.figure_specs()
    assert len(specs) == 5
    for stem, render in specs:
        out = render(synth_metrics_v4["per_sample"], synth_metrics_v4["metrics"], tmp_path / stem)
        _assert_written(out, tmp_path)


# ---------------------------------------------------------------------------
# Warn-don't-gate: empty artifacts degrade to a placeholder, never raise.
# ---------------------------------------------------------------------------
def test_empty_metrics_degrade_to_placeholder(tmp_path) -> None:
    r"""Every figure renders a placeholder (not an exception) from an empty metrics dict."""
    empty: dict = {}
    assert viz.plot_kbar_vs_te_v4(None, empty, tmp_path / "a")
    assert viz.plot_calibration_by_lag_v4(empty, tmp_path / "b")
    assert viz.plot_lag_recovery_v4(empty, tmp_path / "c")
    assert viz.plot_pred_control_v4(empty, tmp_path / "d")
    assert viz.plot_kbar_null_bar_v4(empty, tmp_path / "e")


def test_png_only_format(synth_metrics_v4, tmp_path) -> None:
    r"""A single-format request writes exactly that format."""
    out = viz.plot_kbar_vs_te_v4(
        synth_metrics_v4["per_sample"], synth_metrics_v4["metrics"], tmp_path / "png_only",
        formats=("png",))
    assert len(out) == 1 and Path(out[0]).suffix == ".png"
