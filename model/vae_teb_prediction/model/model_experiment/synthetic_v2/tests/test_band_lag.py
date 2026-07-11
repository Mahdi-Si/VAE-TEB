r"""Tests for the ``lag_mode: band`` long, per-sample-variable lag path.

Covers the raw-generator band helpers (:func:`raw_generators._draw_per_sample_delays`,
:func:`raw_generators.simulate_latent_pair_band`,
:func:`raw_generators.true_lag_trajectory_per_sample`), the per-sample delay / true-lag
wiring through :func:`raw_generators.generate_cell_raw` and
:func:`build_dataset_v2.generate_pilot_samples`, and the ``K_history`` correctness gate
(``K >= D_max``) that keeps the injected-TE label exact at long lags. Fixed-mode behaviour
is asserted byte-unchanged. No scattering transform is run, so these stay fast; the
end-to-end label wiring + frac_Phi at long lags is exercised by the pilot build.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    analytic_te as ate,
    build_dataset_v2 as bd,
    eval_v2 as ev,
    raw_generators as rg,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"


@pytest.fixture(scope="module")
def base_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def band_config(base_config) -> dict:
    r"""A trimmed band-mode config: one tight long window + cheap inverter + K_history=140."""
    cfg = copy.deepcopy(base_config)
    mix = cfg["benchmarks"]["G1_raw"]["mix"]
    mix["target_te_grid"] = [0.0, 2.0]
    mix["lag_mode"] = "band"
    mix["lag_bands"] = [[45, 52]]
    mix["lag_band_units"] = "steps"
    mix["inverter"]["n_samples"] = 4000
    cfg["benchmarks"]["G1_raw"]["data"]["K_history"] = 140
    return cfg


def _latent_kwargs(cfg: dict) -> dict:
    data = cfg["benchmarks"]["G1_raw"]["data"]
    r, w = data["oscillators"][0]
    return dict(
        r=float(r), w=float(w), target_ar=float(data["target_ar"]),
        sigma2_y=float(data["sigma2_y"]), sigma2_eta=float(data["sigma2_eta"]),
    )


# ---------------------------------------------------------------------------
# Per-sample delay draw
# ---------------------------------------------------------------------------


def test_draw_per_sample_delays_uniform_and_deterministic() -> None:
    d = rg._draw_per_sample_delays(5000, 45, 52, seed=7)
    assert d.min() == 45 and d.max() == 52
    assert set(int(x) for x in d) == set(range(45, 53))
    # Roughly uniform over the 8 values (each ~1/8 of the mass).
    counts = np.bincount(d, minlength=53)[45:53]
    assert counts.min() > 0.5 * counts.mean()
    # Deterministic for a fixed seed; different for a different seed.
    assert np.array_equal(d, rg._draw_per_sample_delays(5000, 45, 52, seed=7))
    assert not np.array_equal(d, rg._draw_per_sample_delays(5000, 45, 52, seed=8))


def test_draw_per_sample_delays_rejects_bad_range() -> None:
    with pytest.raises(ValueError):
        rg._draw_per_sample_delays(10, 52, 45, seed=0)  # lo > hi
    with pytest.raises(ValueError):
        rg._draw_per_sample_delays(10, 0, 5, seed=0)     # lo < 1


# ---------------------------------------------------------------------------
# Band latent simulation
# ---------------------------------------------------------------------------


def test_simulate_latent_pair_band_shapes_and_delays(band_config) -> None:
    lk = _latent_kwargs(band_config)
    c, d, delays = rg.simulate_latent_pair_band(
        300, 330, B=2.0, delay_min=45, delay_max=52, seed=1, delay_seed=2, **lk
    )
    assert c.shape == (300, 330) and d.shape == (300, 330)
    assert delays.shape == (300,)
    assert 45 <= int(delays.min()) and int(delays.max()) <= 52


def test_band_degenerate_matches_scalar_stats(band_config) -> None:
    r"""A degenerate band ``[D, D]`` reproduces the scalar simulator's marginal statistics."""
    lk = _latent_kwargs(band_config)
    cb, db, delays = rg.simulate_latent_pair_band(
        3000, 330, B=1.5, delay_min=20, delay_max=20, seed=5, delay_seed=6, **lk
    )
    cs, ds = rg.simulate_latent_pair(3000, 330, B=1.5, D=20, seed=5, **lk)
    assert (delays == 20).all()
    # Seeds are re-derived per group, so this is a statistical (not bit) match.
    assert abs(float(db.std()) - float(ds.std())) / float(ds.std()) < 0.05
    assert abs(float(cb.std()) - float(cs.std())) / float(cs.std()) < 0.05


def test_true_lag_trajectory_per_sample() -> None:
    delays = np.array([45, 60, 75], dtype=np.int64)
    tt = rg.true_lag_trajectory_per_sample(delays, 300)
    assert tt.shape == (3, 300)
    assert tt.dtype == np.int16
    for i, d in enumerate(delays):
        assert (tt[i] == d).all()


# ---------------------------------------------------------------------------
# generate_cell_raw / generate_pilot_samples wiring
# ---------------------------------------------------------------------------


def test_generate_cell_raw_band_wires_per_sample_delay(band_config) -> None:
    cells, _ = bd.enumerate_cells_v2(band_config)
    signal = [c for c in cells if c.target_te > 0.0][0]
    raw = bd.generate_pilot_samples(signal, 48, "train", band_config)
    sd = raw["sample_delay"]
    assert sd is not None
    assert 45 <= int(sd.min()) and int(sd.max()) <= 52
    # true_lag_tt is per-sample flat at each row's own drawn lag.
    assert (raw["true_lag_tt"][:, 0] == sd).all()
    assert raw["meta"]["lag_mode"] == "band"
    assert raw["meta"]["delay_min"] == 45 and raw["meta"]["delay_max"] == 52


def test_generate_cell_raw_fixed_is_unchanged(base_config) -> None:
    r"""Fixed mode returns no ``sample_delay`` and a single flat lag (byte-compatible)."""
    cfg = copy.deepcopy(base_config)
    cfg["benchmarks"]["G1_raw"]["mix"]["target_te_grid"] = [0.0, 2.0]
    cfg["benchmarks"]["G1_raw"]["mix"]["lag_grid"] = [8]
    cfg["benchmarks"]["G1_raw"]["mix"]["inverter"]["n_samples"] = 4000
    cells, _ = bd.enumerate_cells_v2(cfg)
    fc = [c for c in cells if c.target_te > 0.0][0]
    assert fc.lag_mode == "fixed"
    raw = bd.generate_pilot_samples(fc, 16, "train", cfg)
    assert raw["sample_delay"] is None
    assert (raw["true_lag_tt"] == 8).all()
    assert raw["meta"]["lag_mode"] == "fixed"


# ---------------------------------------------------------------------------
# K_history correctness gate
# ---------------------------------------------------------------------------


def test_k_history_gate_at_top_of_band(band_config) -> None:
    r"""The injected-TE label needs ``K >= D_max``: at D=75, K=60 under-measures; K>=75 agrees."""
    lk = _latent_kwargs(band_config)
    sol = bd.solve_cell_coupling(band_config, 2.0, 67, 75)
    B = float(sol["B_y_scalar"])

    def te_at(kh: int) -> float:
        return ate.te_block_state_space_gaussian(
            oscillators=[(lk["r"], lk["w"])], target_ar=lk["target_ar"], delays=[75],
            B_y=[B], sigma2_y=lk["sigma2_y"], sigma2_eta=lk["sigma2_eta"], H=30,
            K_history=kh, n_samples=8000, seed=1,
        )

    te140, te80, te60 = te_at(140), te_at(80), te_at(60)
    assert te60 < te140 * 0.85, "K=60 (< D=75) should under-measure the coupling"
    assert abs(te80 - te140) / te140 < 0.1, "K>=D_max should recover the coupling"


def test_band_te_by_delay_mean_hits_target(band_config) -> None:
    r"""The per-delay TE map averages to the requested target within tolerance (K=140)."""
    cells, dropped = bd.enumerate_cells_v2(band_config)
    assert dropped == []
    signal = [c for c in cells if c.target_te > 0.0][0]
    mean_te = float(np.mean(list(signal.te_by_delay.values())))
    assert abs(mean_te - signal.target_te) / signal.target_te < 0.05


@pytest.mark.slow
def test_realizability_preflight_band_scopes_to_pilot_window(band_config) -> None:
    r"""The pilot pre-flight must enumerate its own band window, not the full mix grid.

    Regression for a bug where ``run_realizability_preflight`` forwarded only ``lag_grid``
    to :func:`enumerate_cells_v2`, so under ``lag_mode: band`` it fell back to the full
    ``mix.lag_bands`` grid and silently un-scoped the pilot. Runs the real transform on a
    tiny grid, so it only asserts the enumeration scoping / grid header (not frac values).
    """
    cfg = copy.deepcopy(band_config)
    ev_cfg = cfg["benchmarks"]["G1_raw"]["eval"]["realizability"]
    ev_cfg["pilot"] = {
        "target_te_grid": [0.0, 2.0],
        "lag_bands": [[45, 52]],       # pilot window is ONE band...
        "lag_grid": [4, 8],
        "n_per_cell": 32,
    }
    # ...distinct from the full mix.lag_bands grid (which would give more cells).
    cfg["benchmarks"]["G1_raw"]["mix"]["lag_bands"] = [[45, 52], [52, 60], [60, 67]]

    res = ev.run_realizability_preflight(cfg, pilot=True, print_table=False)
    assert res["grid"]["lag_mode"] == "band"
    assert res["grid"]["lag_bands"] == [[45, 52]]
    # 2 target_te x 1 pilot window = 2 cells (NOT 2 x 3 = 6 from the full mix grid).
    assert len(res["per_cell"]) == 2
    assert {c["D"] for c in res["per_cell"].values()} == {45}
