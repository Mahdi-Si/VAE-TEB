r"""S1-T02: the re-targeted raw-TE probe dispatches by render mode and recovers the coupling."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import te_raw_v4
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    generate_cell_raw,
    solve_cell_coupling,
)

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"
_N = 384
_D = 8


def _config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["benchmarks"]["G1_raw_v4"]["mix"]["inverter"]["n_samples"] = 2000
    return cfg


@pytest.fixture(scope="module")
def probe_data():
    r"""One strong-TE (direct) cell and one null cell, generated once (fixed seeds)."""
    cfg = _config()
    sol = solve_cell_coupling(cfg, 3.0, _D, benchmark="G1_raw_v4")
    sig = generate_cell_raw(_N, B=float(sol["B_y_scalar"]), D=_D, config=cfg,
                            benchmark="G1_raw_v4", seed=11, render_mode="direct")
    null = generate_cell_raw(_N, B=0.0, D=_D, config=cfg,
                             benchmark="G1_raw_v4", seed=12, render_mode="direct")
    return {"config": cfg, "sig": sig, "null": null}


def test_direct_signal_te_raw_is_positive(probe_data) -> None:
    r"""On a $B>0$ direct cell the direct probe recovers a positive $\mathrm{TE}_{\mathrm{raw}}$."""
    res = te_raw_v4.measure_te_raw_v4(
        probe_data["sig"]["fhr_raw"], probe_data["sig"]["up_raw"],
        D=_D, render_mode="direct", config=probe_data["config"], benchmark="G1_raw_v4",
    )
    assert res["render_mode"] == "direct"
    assert res["te_raw"] > 0.0


def test_direct_null_te_raw_carries_no_positive_info(probe_data) -> None:
    r"""On a null ($B=0$) cell the direct probe carries no positive TE (sits below the ceiling)."""
    res = te_raw_v4.measure_te_raw_v4(
        probe_data["null"]["fhr_raw"], probe_data["null"]["up_raw"],
        D=_D, render_mode="direct", config=probe_data["config"], benchmark="G1_raw_v4",
    )
    assert res["te_raw"] <= 0.1


def test_direct_exceeds_bandpass_on_a_direct_cell(probe_data) -> None:
    r"""The band-pass path filters away the low-freq coupling, so direct > am on a direct cell."""
    kw = dict(D=_D, config=probe_data["config"], benchmark="G1_raw_v4")
    direct = te_raw_v4.measure_te_raw_v4(
        probe_data["sig"]["fhr_raw"], probe_data["sig"]["up_raw"], render_mode="direct", **kw)
    am = te_raw_v4.measure_te_raw_v4(
        probe_data["sig"]["fhr_raw"], probe_data["sig"]["up_raw"], render_mode="am_carrier", **kw)
    assert direct["te_raw"] > am["te_raw"]
    assert am["render_mode"] == "am_carrier"


def test_direct_path_uses_decimated_shape_and_explicit_kh(probe_data, monkeypatch) -> None:
    r"""The direct path passes $(n,330,1)$ decimated arrays and explicit $K$/$H$ to the estimator."""
    captured = {}

    def _spy(Y, U, *, K, H, delay_max, ridge, n_anchors, n_seeds):
        captured.update(Y_shape=Y.shape, U_shape=U.shape, K=K, H=H, delay_max=delay_max)
        return {"gain": 0.5, "n_used": 6, "n_total": 6, "ill_fraction": 0.0}

    monkeypatch.setattr(te_raw_v4, "_r0_gain_over_anchors", _spy)
    te_raw_v4.measure_te_raw_v4(
        probe_data["sig"]["fhr_raw"], probe_data["sig"]["up_raw"],
        D=_D, render_mode="direct", config=probe_data["config"], benchmark="G1_raw_v4",
    )
    assert captured["Y_shape"] == (_N, 330, 1)
    assert captured["U_shape"] == (_N, 330, 1)
    assert captured["K"] == 80        # data.K_history
    assert captured["H"] == 30        # data.horizon
    assert captured["delay_max"] == _D


def test_unknown_render_mode_raises(probe_data) -> None:
    r"""An unrecognised render mode raises ``ValueError``."""
    with pytest.raises(ValueError):
        te_raw_v4.measure_te_raw_v4(
            probe_data["sig"]["fhr_raw"], probe_data["sig"]["up_raw"],
            D=_D, render_mode="nonsense", config=probe_data["config"], benchmark="G1_raw_v4",
        )


def test_non_2d_input_raises(probe_data) -> None:
    r"""A non-2-D waveform input raises ``ValueError``."""
    bad = probe_data["sig"]["fhr_raw"][None]  # (1, n, 5280)
    with pytest.raises(ValueError):
        te_raw_v4.measure_te_raw_v4(
            bad, bad, D=_D, render_mode="direct", config=probe_data["config"],
            benchmark="G1_raw_v4",
        )
