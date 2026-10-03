"""The planted-delay check runs end to end. Its band reading is a measurement and is not asserted."""
from __future__ import annotations

import math

import pytest

from teb_vae.lag_attn_transformer_patch import lag_recovery_check


@pytest.mark.slow
def test_the_check_runs_end_to_end_and_reports_finite_numbers(tmp_path) -> None:
    record = lag_recovery_check.main(
        config="teb_vae/lag_attn_transformer_patch/configs/planted.yaml",
        override=[
            "general_config.epochs=2",
            f"general_config.folders_config.out_dir_base={tmp_path}",
        ],
    )

    assert record["epochs"] == 2
    for name in ("raw", "corrected"):
        assert 0.0 <= record[f"{name}_band_share"] <= 1.0
        assert 0 <= record[f"{name}_argmax"] <= record["max_lag"]
        assert all(math.isfinite(v) for v in record[f"{name}_profile"])
