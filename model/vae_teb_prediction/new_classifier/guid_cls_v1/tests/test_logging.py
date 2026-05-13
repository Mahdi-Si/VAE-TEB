"""Smoke tests for the per-fold logging infrastructure.

Covers the modules added in plan
``read-model-dataset-explained-research-md-wobbly-salamander``:

  * :mod:`logging_utils` — loguru file sink + stdout/stderr tee
    capture both formatted and raw output to ``fold.log``;
    :func:`dump_json` / :func:`append_jsonl` produce JSON-safe payloads
    even when the input contains ``NaN`` / numpy / torch scalars.
  * :mod:`diagnostics_callback` — :class:`TrainingDiagnosticsCallback`
    computes a global grad norm without touching trainer internals.
  * :mod:`lightning_module` helpers — :func:`_per_class_prf1`,
    :func:`_binary_brier`, :func:`_expected_calibration_error` produce
    the expected values on synthetic data.

These tests intentionally exercise the helpers in isolation rather
than spinning up a full Lightning trainer — that keeps the test
fast and stable, while the end-to-end manual verification at the
plan's §"Verification" §2 covers the integration path.
"""

from __future__ import annotations

import json
import sys

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from model.vae_teb_prediction.new_classifier.guid_cls_v1 import logging_utils  # noqa: E402
from model.vae_teb_prediction.new_classifier.guid_cls_v1.diagnostics_callback import (  # noqa: E402
    _global_grad_norm,
    _global_weight_norm,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.lightning_module import (  # noqa: E402
    _binary_brier,
    _expected_calibration_error,
    _per_class_prf1,
)


# ---------------------------------------------------------------------------
# logging_utils.attach_fold_log_sinks / detach
# ---------------------------------------------------------------------------


def test_attach_detach_sink_captures_loguru_and_stdout(tmp_path) -> None:
    """Loguru lines and ``print`` output both land in ``fold.log``."""
    from loguru import logger

    handle = logging_utils.attach_fold_log_sinks(
        tmp_path, log_level="INFO", capture_stdout_stderr=True
    )
    try:
        logger.info("LOGURU_PROBE_LINE_42")
        print("STDOUT_PROBE_LINE_42")
        print("STDERR_PROBE_LINE_42", file=sys.stderr)
    finally:
        logging_utils.detach_fold_log_sinks(handle)

    log_path = tmp_path / "logs" / "fold.log"
    assert log_path.exists(), "fold.log not created"
    text = log_path.read_text(encoding="utf-8")
    assert "LOGURU_PROBE_LINE_42" in text
    assert "STDOUT_PROBE_LINE_42" in text
    assert "STDERR_PROBE_LINE_42" in text


def test_attach_detach_restores_streams(tmp_path) -> None:
    """``sys.stdout`` / ``sys.stderr`` are restored after detach."""
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    handle = logging_utils.attach_fold_log_sinks(tmp_path)
    assert sys.stdout is not orig_stdout
    assert sys.stderr is not orig_stderr
    logging_utils.detach_fold_log_sinks(handle)
    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr


def test_attach_without_stream_capture(tmp_path) -> None:
    """``capture_stdout_stderr=False`` leaves the streams alone."""
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    handle = logging_utils.attach_fold_log_sinks(
        tmp_path, capture_stdout_stderr=False
    )
    try:
        assert sys.stdout is orig_stdout
        assert sys.stderr is orig_stderr
    finally:
        logging_utils.detach_fold_log_sinks(handle)


# ---------------------------------------------------------------------------
# logging_utils.dump_json / append_jsonl / to_json_safe
# ---------------------------------------------------------------------------


def test_to_json_safe_handles_numpy_torch_nan() -> None:
    """``NaN`` / ``Inf`` coerced to ``None``; tensors/arrays to plain lists."""
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "np_float": np.float32(1.5),
        "np_int": np.int64(7),
        "np_array": np.array([1.0, 2.0, float("nan")]),
        "torch_scalar": torch.tensor(3.25),
        "torch_vec": torch.tensor([0.1, 0.2, 0.3]),
        "list": [1, 2.0, np.float64(3.0)],
    }
    safe = logging_utils.to_json_safe(payload)
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["np_float"] == pytest.approx(1.5)
    assert safe["np_int"] == 7
    assert safe["np_array"][0] == pytest.approx(1.0)
    assert safe["np_array"][2] is None
    assert safe["torch_scalar"] == pytest.approx(3.25)
    assert safe["torch_vec"] == pytest.approx([0.1, 0.2, 0.3])
    assert safe["list"][2] == pytest.approx(3.0)
    # Round-trip through ``json.dumps`` must succeed.
    json.dumps(safe)


def test_dump_json_writes_pretty(tmp_path) -> None:
    """``dump_json`` writes valid JSON and creates parent dirs."""
    target = tmp_path / "deeper" / "setup.json"
    logging_utils.dump_json(target, {"k": 1, "v": [1.0, 2.0]})
    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == {"k": 1, "v": [1.0, 2.0]}


def test_append_jsonl_appends_one_line_per_call(tmp_path) -> None:
    """Two calls produce two separate JSON objects on separate lines."""
    target = tmp_path / "summary.jsonl"
    logging_utils.append_jsonl(target, {"event": "a", "n": 1})
    logging_utils.append_jsonl(target, {"event": "b", "n": 2})
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "a"
    assert json.loads(lines[1])["event"] == "b"


# ---------------------------------------------------------------------------
# lightning_module: per-class metrics + calibration
# ---------------------------------------------------------------------------


def test_per_class_prf1_perfect_predictions() -> None:
    """Perfect predictions -> precision = recall = F1 = 1 for every class."""
    # 6 examples, 2 per class, every prediction is the true class.
    probs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    res = _per_class_prf1(probs, target)
    assert torch.allclose(res["precision"], torch.ones(3))
    assert torch.allclose(res["recall"], torch.ones(3))
    assert torch.allclose(res["f1"], torch.ones(3))
    assert float(res["macro_f1"]) == pytest.approx(1.0)
    # Confusion matrix is diagonal.
    assert torch.equal(res["confusion"], torch.diag(torch.tensor([2, 2, 2])))
    assert torch.equal(res["support"], torch.tensor([2, 2, 2]))


def test_per_class_prf1_handles_absent_class() -> None:
    """A class with zero support yields zero recall but no NaNs."""
    probs = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
        ]
    )
    target = torch.tensor([0, 1])
    res = _per_class_prf1(probs, target)
    # Class 2 has zero support and zero predictions.
    assert float(res["recall"][2]) == 0.0
    assert float(res["precision"][2]) == 0.0
    assert float(res["f1"][2]) == 0.0
    assert torch.isfinite(res["macro_f1"]).item()


def test_binary_brier_and_ece_on_perfect_calibration() -> None:
    """Perfectly calibrated probabilities have low ECE; opposite labels high Brier."""
    # Perfect predictions: prob = label.
    probs = torch.tensor([0.0, 0.0, 1.0, 1.0])
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert float(_binary_brier(probs, target)) == pytest.approx(0.0)
    assert float(_expected_calibration_error(probs, target, n_bins=10)) == pytest.approx(0.0)

    # Worst case: prob = 1 - label.
    probs_bad = torch.tensor([1.0, 1.0, 0.0, 0.0])
    target_bad = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert float(_binary_brier(probs_bad, target_bad)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# diagnostics_callback helpers
# ---------------------------------------------------------------------------


def test_global_grad_norm_matches_manual() -> None:
    """``_global_grad_norm`` matches the closed-form L2 norm over grads."""
    a = torch.tensor([3.0, 4.0], requires_grad=True)
    b = torch.tensor([0.0, -12.0], requires_grad=True)
    # Make grads non-None: 3^2 + 4^2 = 25 -> ||a.grad||_2 = 5.
    # 0^2 + 12^2 = 144 -> ||b.grad||_2 = 12.
    a.grad = torch.tensor([3.0, 4.0])
    b.grad = torch.tensor([0.0, -12.0])
    grad_norm = _global_grad_norm([a, b])
    # Combined L2: sqrt(25 + 144) = 13.
    assert grad_norm == pytest.approx(13.0)


def test_global_grad_norm_zero_when_no_grads() -> None:
    """No ``.grad`` set -> grad norm 0."""
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    assert _global_grad_norm([a]) == pytest.approx(0.0)


def test_global_weight_norm_skips_frozen_params() -> None:
    """``requires_grad=False`` params don't contribute."""
    trainable = torch.nn.Parameter(torch.tensor([3.0, 4.0]))      # ||.||=5
    frozen = torch.nn.Parameter(torch.tensor([10.0, 10.0]), requires_grad=False)
    norm = _global_weight_norm([trainable, frozen])
    assert norm == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# kfold_trainer: subprocess detection guard
# ---------------------------------------------------------------------------


def test_attach_subprocess_raw_log_skips_in_main_process(tmp_path) -> None:
    """In sequential mode (main process), the raw-log dup2 must NOT fire.

    The fixture deliberately calls ``_attach_subprocess_raw_log`` from
    the main process; the guard at the top should detect
    ``MainProcess`` and return ``None`` without touching
    ``sys.stdout`` / ``sys.stderr``. Regression test for the bug
    where the helper hijacked the user's terminal during
    ``--sequential`` k-fold runs.
    """
    import multiprocessing

    from model.vae_teb_prediction.new_classifier.guid_cls_v1.kfold_trainer import (
        _attach_subprocess_raw_log,
    )

    # Sanity check: we really are in the main process.
    assert multiprocessing.current_process().name == "MainProcess"

    # Provide a minimal config so the function's later branches (which
    # we never reach due to the guard) would have valid inputs.
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "general_config:\n"
        "  tag: test_run\n"
        "  folders_config:\n"
        f"    out_dir_base: {tmp_path}\n",
        encoding="utf-8",
    )

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    handle = _attach_subprocess_raw_log(
        config_path=str(cfg_path),
        fold_id=1,
        output_dir_override=None,
    )
    # Guard returns None, streams remain untouched.
    assert handle is None
    assert sys.stdout is orig_stdout
    assert sys.stderr is orig_stderr
