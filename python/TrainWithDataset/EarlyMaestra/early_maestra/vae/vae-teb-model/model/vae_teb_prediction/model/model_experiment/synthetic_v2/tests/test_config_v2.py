r"""Tests for ``config_synth_v2.yaml`` (S0-T01).

Asserts the config parses and carries every EXPLAINED §16 key plus the S0-required
extras (``scattering.batch_size``, the ``seeds`` block, ``eval.realizability.fatal``,
``mix.inverter``), that ``render_mode`` is ``am_carrier`` and ``norm_stats_source``
is ``synthetic_pool``, that the lag mode is ``fixed``, that the inherited
model/loss/optim/dataset blocks exist, and that there is **no** ``m_grid`` anywhere
(single pathway).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_synth_v2.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    r"""Load the parsed ``config_synth_v2.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _iter_keys(obj: Any) -> Iterator[str]:
    r"""Yield every mapping key found recursively in ``obj``."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _iter_keys(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_keys(item)


def test_config_parses(config: dict) -> None:
    assert isinstance(config, dict) and config


def test_top_level_blocks_present(config: dict) -> None:
    for block in [
        "experiment", "runtime", "paths", "seeds", "benchmarks",
        "model", "loss", "optim", "dataset", "ddp",
    ]:
        assert block in config, f"missing top-level block: {block}"


def test_seeds_block(config: dict) -> None:
    seeds = config["seeds"]
    for key in ["base_seed", "dgp", "inverter_mc", "shuffle"]:
        assert key in seeds, f"missing seeds.{key}"


def test_g1_raw_data_block(config: dict) -> None:
    data = config["benchmarks"]["G1_raw"]["data"]
    for key in [
        "sequence_length", "horizon", "K_history",
        "oscillators", "target_ar", "sigma2_y", "sigma2_eta",
    ]:
        assert key in data, f"missing data.{key}"
    # Single pathway: exactly one oscillator spec.
    assert len(data["oscillators"]) == 1
    assert data["oscillators"][0] == [0.80, 0.10]


def test_g1_raw_raw_block(config: dict) -> None:
    raw = config["benchmarks"]["G1_raw"]["raw"]
    assert raw["render_mode"] == "am_carrier"
    for key in [
        "fs", "n_raw", "f_pulse", "fhrv_notch_enabled", "am_offset_ratio",
        "mu_fhr_bpm", "mu_up", "baseline_wander_std", "decel_depth_bpm",
        "contraction_mmHg", "fhrv_band_power", "accel", "noise_std",
    ]:
        assert key in raw, f"missing raw.{key}"
    # The FHRV notch (§7.3, §19) defends the coupled decel channel from LF dressing.
    assert raw["fhrv_notch_enabled"] is True


def test_g1_raw_scattering_block(config: dict) -> None:
    scattering = config["benchmarks"]["G1_raw"]["scattering"]
    for key in [
        "J", "Q", "T", "max_order", "batch_size",
        "phase_min_freq", "norm_stats_source", "real_fold_stats_path",
    ]:
        assert key in scattering, f"missing scattering.{key}"
    assert scattering["norm_stats_source"] == "synthetic_pool"
    assert isinstance(scattering["batch_size"], int) and scattering["batch_size"] > 0
    assert set(scattering["phase_min_freq"]) == {"fhr", "up"}


def test_g1_raw_mix_block(config: dict) -> None:
    mix = config["benchmarks"]["G1_raw"]["mix"]
    assert mix["lag_mode"] == "fixed"
    for key in [
        "target_te_grid", "lag_grid",
        "n_per_cell_train", "n_per_cell_val", "n_per_cell_test", "inverter",
    ]:
        assert key in mix, f"missing mix.{key}"
    assert 0.0 in mix["target_te_grid"], "mix.target_te_grid must include a 0.0 null anchor"
    inverter = mix["inverter"]
    for key in ["n_samples", "lo", "hi", "tol", "max_iter"]:
        assert key in inverter, f"missing mix.inverter.{key}"


def test_g1_raw_eval_block(config: dict) -> None:
    ev = config["benchmarks"]["G1_raw"]["eval"]
    assert "realizability" in ev
    assert ev["realizability"]["fatal"] is False
    for key in ["frac_threshold", "frac_upper", "ridge"]:
        assert key in ev["realizability"], f"missing eval.realizability.{key}"
    # Two-sided frac_Phi gate: lower bound < upper bound, and the upper bound sits above
    # 1 (frac_Phi is biased high, so 1.0 must remain inside the pass band).
    assert ev["realizability"]["frac_threshold"] < ev["realizability"]["frac_upper"]
    assert ev["realizability"]["frac_upper"] >= 1.0


def test_model_contract(config: dict) -> None:
    model = config["model"]
    # The unchanged VAE-TEB model consumes fhr(87) / up(101).
    assert model["c_y"] == 87
    assert model["c_u"] == 101


def test_no_m_grid_anywhere(config: dict) -> None:
    keys = set(_iter_keys(config))
    assert "m_grid" not in keys, "v2 is single-pathway: no m_grid allowed"
    assert "M" not in config["benchmarks"]["G1_raw"]["data"]


def test_full_grid_locked(config: dict) -> None:
    r"""S4-T01: the locked headline grid + render knobs are present and consistent.

    Sprint 3's de-risk fixed the recovered recipe: carrier ``f_pulse = 0.06`` Hz and a
    ``lag_grid`` restricted to $D \ge 8$ (short lag $D = 4$ is under-preserved). This
    asserts the config reflects that lock and stays single-pathway (no ``m_grid``,
    fixed lag), so a build reads a coherent grid.
    """
    bench = config["benchmarks"]["G1_raw"]
    mix = bench["mix"]
    # Locked lag grid: fixed per-cell lag, every lag D >= 8 (D=4 dropped).
    assert mix["lag_mode"] == "fixed"
    assert mix["lag_grid"] == [8, 12, 20], mix["lag_grid"]
    assert all(int(d) >= 8 for d in mix["lag_grid"]), "locked lags must be D >= 8"
    # Target-TE grid keeps the 0.0 null anchor.
    assert 0.0 in mix["target_te_grid"], "grid must include a 0.0 null anchor"
    # Per-cell sample counts present and positive for every split.
    for key in ("n_per_cell_train", "n_per_cell_val", "n_per_cell_test"):
        assert isinstance(mix[key], int) and mix[key] > 0, f"mix.{key} must be > 0"
    # Locked render knobs: recovered carrier; am_carrier mode.
    raw = bench["raw"]
    assert raw["render_mode"] == "am_carrier"
    assert float(raw["f_pulse"]) == 0.06, raw["f_pulse"]
    # Still single-pathway: exactly one oscillator, no m_grid.
    assert len(bench["data"]["oscillators"]) == 1
    assert "m_grid" not in _iter_keys(mix)
