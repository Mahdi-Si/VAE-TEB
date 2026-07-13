r"""S1-T01: the concentrated cell grid solves a known, monotone, null-anchored coupling."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cells_v4

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


def _fast_config() -> dict:
    r"""The real config with the MC inverter shrunk so the solve is quick but still accurate."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    inv = cfg["benchmarks"]["G1_raw_v4"]["mix"]["inverter"]
    inv["n_samples"] = 1200
    inv["max_iter"] = 20
    return cfg


@pytest.fixture(scope="module")
def solved_grid():
    r"""Enumerate + solve the concentrated grid once (reduced MC) for the whole module."""
    cfg = _fast_config()
    cells, dropped = cells_v4.enumerate_cells_v4(cfg, benchmark="G1_raw_v4")
    return {"cells": cells, "dropped": dropped, "config": cfg}


def test_grid_covers_the_ladder(solved_grid) -> None:
    r"""One cell per ladder level (``cells_per_level=1``), nothing dropped."""
    cells = solved_grid["cells"]
    assert solved_grid["dropped"] == []
    assert [c.target_te for c in cells] == [0.0, 0.5, 1.0, 2.0, 3.0]
    assert all(c.D == 8 for c in cells)


def test_null_cell_has_zero_coupling(solved_grid) -> None:
    r"""The null cell ($\mathrm{TE}_{\mathrm{inj}}=0$) has $B=0$ and zero realised TE."""
    nulls = [c for c in solved_grid["cells"] if c.is_null]
    assert len(nulls) == 1
    assert nulls[0].B_y_scalar == 0.0
    assert nulls[0].te_block_realised == 0.0


def test_coupling_and_realised_te_are_monotone(solved_grid) -> None:
    r"""$B$ and the realised block TE are non-decreasing across the TE ladder."""
    cells = sorted(solved_grid["cells"], key=lambda c: c.target_te)
    b_vals = [c.B_y_scalar for c in cells]
    te_vals = [c.te_block_realised for c in cells]
    assert b_vals == sorted(b_vals)
    assert te_vals == sorted(te_vals)


def test_realised_te_matches_ladder_target(solved_grid) -> None:
    r"""Each non-null cell's realised block TE is within tolerance of its ladder target."""
    for c in solved_grid["cells"]:
        if c.is_null:
            continue
        assert abs(c.te_block_realised - c.target_te) <= 0.2 + 0.1 * c.target_te


def test_cells_per_level_replication() -> None:
    r"""``cells_per_level`` replicates each level into i.i.d. cells (unique ids, shared coupling)."""
    cfg = _fast_config()
    cfg["benchmarks"]["G1_raw_v4"]["mix"]["target_te_grid"] = [0.0, 1.0]
    cells, _ = cells_v4.enumerate_cells_v4(cfg, benchmark="G1_raw_v4", cells_per_level=3)
    assert len(cells) == 2 * 3
    ids = [c.cell_id for c in cells]
    assert ids == list(range(6))  # contiguous
    # The three replicas of the signal level share one solved coupling.
    signal_bs = {c.B_y_scalar for c in cells if not c.is_null}
    assert len(signal_bs) == 1


def test_band_lag_mode_is_rejected() -> None:
    r"""Band lag is deferred: ``lag_mode='band'`` raises in the first cut."""
    cfg = _fast_config()
    cfg["benchmarks"]["G1_raw_v4"]["mix"]["lag_mode"] = "band"
    with pytest.raises(ValueError):
        cells_v4.enumerate_cells_v4(cfg, benchmark="G1_raw_v4")
