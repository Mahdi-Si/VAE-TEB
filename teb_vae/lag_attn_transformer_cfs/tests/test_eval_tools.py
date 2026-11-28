r"""The small offline tools beside the evaluation: the linear baselines and the quality-arm runner.

Each test drives one pure function on hand-built inputs, with no checkpoint and no shard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from teb_vae.lag_attn_transformer_cfs.eval import linear_baselines, run_quality_arms


def test_the_ridge_shrinks_the_weights_but_not_the_intercept() -> None:
    r"""With $\lambda N I_0$ the intercept is free, so a huge ridge still predicts the fit mean
    of a constant target $y = 5$ rather than pulling the prediction towards $0$."""
    features = torch.randn(64, 3, dtype=torch.float64)
    targets = torch.full((64, 1), 5.0, dtype=torch.float64)

    prediction = linear_baselines._ridge(features, targets, features[:4], ridge=1e6)

    assert prediction.numpy() == pytest.approx(np.full((4, 1), 5.0), abs=1e-6)


def test_the_per_recording_mean_weights_each_recording_once() -> None:
    r"""Recording ``a`` has three anchors at error $1$, recording ``b`` one at error $9$: the flat
    anchor mean is $3$, the per-recording mean $\frac{1}{2}(1 + 9) = 5$."""
    target = torch.zeros(4, 1, 1, dtype=torch.float64)
    prediction = torch.tensor([1.0, 1.0, 1.0, 3.0], dtype=torch.float64).reshape(4, 1, 1)
    mask, scattering = np.ones((1, 1), dtype=bool), np.ones(1, dtype=bool)

    table = linear_baselines.score({"zero": prediction}, target, mask, scattering, ["a", "a", "a", "b"])

    assert table["zero"]["mse_all"] == pytest.approx(3.0)
    assert table["zero"]["mse_all_per_recording"] == pytest.approx(5.0)


def _checkpoint(tmp_path: Path, name: str, **switches: bool) -> Path:
    """A checkpoint file with its run's resolved config beside it."""
    directory = tmp_path / name / "model_checkpoints"
    directory.mkdir(parents=True)
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump({"model_config": {"VAE_model": dict(switches)}}), encoding="utf-8"
    )
    checkpoint = directory / "last.ckpt"
    checkpoint.write_bytes(b"")
    return checkpoint


def test_a_masked_arm_trained_without_ua_valid_event_weights_is_accepted(tmp_path) -> None:
    """``run_s8_source_only`` keeps both masks with ``event_weight_ua_valid`` off; the arm is the
    masks, so the switch is reported and not enforced."""
    checkpoint = _checkpoint(
        tmp_path, "masked", source_quality_mask=True, target_quality_mask=True, event_weight_ua_valid=False
    )

    switches = run_quality_arms.check_arm("masked", checkpoint)

    assert switches["event_weight_ua_valid"] is False


def test_a_swapped_arm_is_still_refused(tmp_path) -> None:
    """Non-vacuity: a control checkpoint given as the masked arm fails on the masks."""
    checkpoint = _checkpoint(tmp_path, "control", source_quality_mask=False, target_quality_mask=False)

    with pytest.raises(SystemExit, match="did not train as the masked arm"):
        run_quality_arms.check_arm("masked", checkpoint)


def test_no_run_arg_path_starts_with_a_double_slash() -> None:
    """``//data/...`` is a UNC share on Windows, not the POSIX path the other entries name."""
    paths = [value for value in run_quality_arms.RUN_ARGS.values() if isinstance(value, str)]

    assert not [path for path in paths if path.startswith("//")]
