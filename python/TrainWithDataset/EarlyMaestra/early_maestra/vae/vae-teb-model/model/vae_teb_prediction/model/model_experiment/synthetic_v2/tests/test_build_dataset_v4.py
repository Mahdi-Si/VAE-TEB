r"""Sprint 2 tests for ``build_dataset_v4`` -- the raw ``.npz`` cache build (no scattering).

Selections mirror the task list:
``-k generate`` (S2-T01), ``-k weight`` (S2-T02), ``-k assemble`` (S2-T03a),
``-k meta`` (S2-T03b), ``-k resume`` (S2-T04), ``-k build_all`` (S2-T05),
``-k am_coexist`` (S2-T06). All are marked ``v4``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import build_dataset_v4 as bd
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.arms_v4 import resolve_arm_v4
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cells_v4 import enumerate_cells_v4
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.data_previews_v4 import (
    coupling_score,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    frontend_mask,
    load_config,
)

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_synth_v4.yaml"


@pytest.fixture(scope="module")
def v4_config() -> Dict[str, Any]:
    r"""The parsed ``config_synth_v4.yaml`` (loaded once per module)."""
    return load_config(str(_CONFIG_PATH))


def _two_cells(config: Dict[str, Any]):
    r"""One null + one signal cell (target_te 0.0 and 2.0, D=8)."""
    cells, _ = enumerate_cells_v4(
        config, benchmark="G1_raw_v4", target_te_grid=[0.0, 2.0], lag_grid=[8],
    )
    null = next(c for c in cells if c.is_null)
    signal = next(c for c in cells if not c.is_null)
    return null, signal


# ===========================================================================
# S2-T01: generate_split_raw_v4
# ===========================================================================
def test_generate_split_raw_v4_shapes(v4_config):
    r"""One cell generates raw pairs of the documented shapes."""
    _, signal = _two_cells(v4_config)
    out = bd.generate_split_raw_v4(signal, 4, "train", v4_config, benchmark="G1_raw_v4")
    assert out["fhr_raw"].shape == (4, bd.RAW_LEN)
    assert out["up_raw"].shape == (4, bd.RAW_LEN)
    assert out["true_lag_tt"].shape == (4, bd.T_TILDE)
    assert out["latents"]["c"].shape == (4, bd.T_TILDE)
    assert out["latents"]["d"].shape == (4, bd.T_TILDE)


def test_generate_split_raw_v4_coupled_band_contrast(v4_config):
    r"""Source coupling reaches the raw waveform only for the ``B>0`` cell."""
    null, signal = _two_cells(v4_config)
    out_null = bd.generate_split_raw_v4(null, 48, "train", v4_config, benchmark="G1_raw_v4")
    out_sig = bd.generate_split_raw_v4(signal, 48, "train", v4_config, benchmark="G1_raw_v4")
    score_null = coupling_score(out_null["fhr_raw"], out_null["up_raw"], null.D)
    score_sig = coupling_score(out_sig["fhr_raw"], out_sig["up_raw"], signal.D)
    assert score_sig > score_null + 0.02, (score_sig, score_null)


# ===========================================================================
# S2-T02: synth_weight
# ===========================================================================
def test_synth_weight_shape_allones():
    r"""Default synth_weight is an all-ones ``(n, 330)`` float32 array."""
    w = bd.synth_weight(5, rng=np.random.default_rng(0))
    assert w.shape == (5, bd.T_TILDE)
    assert w.dtype == np.float32
    assert np.all(w == 1.0)


def test_synth_weight_frontend_mask_roundtrip():
    r"""A synthesized weight is accepted by the model's ``frontend_mask`` (no raise)."""
    import torch

    w = bd.synth_weight(3, rng=np.random.default_rng(1))
    mask = frontend_mask(torch.from_numpy(w), bd.RAW_LEN, bd.DECIMATION)
    assert tuple(mask.shape) == (3, bd.RAW_LEN)
    assert set(np.unique(mask.numpy()).tolist()) <= {0.0, 1.0}


def test_synth_weight_gap_upsamples_to_raw_zero_run():
    r"""A planted decimated gap upsamples to a $16\times$-wide raw zero-run under frontend_mask."""
    import torch

    w = bd.synth_weight(20, rng=np.random.default_rng(2), gap_frac=1.0)
    gapped_rows = np.where((w == 0).any(axis=1))[0]
    assert gapped_rows.size > 0
    mask = frontend_mask(torch.from_numpy(w), bd.RAW_LEN, bd.DECIMATION).numpy()
    r = gapped_rows[0]
    n_zero_dec = int((w[r] == 0).sum())
    n_zero_raw = int((mask[r] == 0).sum())
    assert n_zero_raw == n_zero_dec * bd.DECIMATION


# ===========================================================================
# S2-T03a: assemble_split_v4
# ===========================================================================
def test_assemble_split_v4_arrays(v4_config, tmp_path):
    r"""Assembled split has the raw fields + weight/target + all provenance, row-aligned."""
    cells, _ = enumerate_cells_v4(
        v4_config, benchmark="G1_raw_v4", target_te_grid=[0.0, 2.0], lag_grid=[8],
    )
    parts_dir = tmp_path / "_parts"
    bd.build_split_parts_v4(
        cells, "train", 5, config=v4_config, benchmark="G1_raw_v4",
        render_mode="direct", base_seed=0, parts_dir=parts_dir,
    )
    stats = bd._fit_norm_stats_v4(cells, "train", parts_dir=parts_dir)
    arrays = bd.assemble_split_v4(
        cells, "train", parts_dir=parts_dir, stats=stats, config=v4_config, benchmark="G1_raw_v4",
    )
    n = len(cells) * 5
    assert arrays["fhr"].shape == (n, bd.RAW_LEN)
    assert arrays["up"].shape == (n, bd.RAW_LEN)
    assert arrays["weight"].shape == (n, bd.T_TILDE)
    assert arrays["target"].shape == (n, bd.T_TILDE)
    for key in ("sample_te_true", "sample_delay", "sample_cell_id",
                "sample_held_out", "sample_raw_index", "true_lag_tt"):
        assert arrays[key].shape[0] == n, key
    # Normalisation: pooled train fhr/up are ~zero-mean unit-std.
    assert abs(float(arrays["fhr"].mean())) < 0.2
    assert abs(float(arrays["fhr"].std()) - 1.0) < 0.2


# ===========================================================================
# S2-T03b: norm_stats + meta (assert on the session tiny cache)
# ===========================================================================
def test_meta_json_and_norm_stats_v4(tiny_cache_v4):
    r"""The built cache carries a scalar norm_stats.npz and a complete meta.json."""
    cache_dir = tiny_cache_v4["cache_dir"]
    with np.load(cache_dir / "norm_stats.npz") as z:
        for key in ("fhr_mean", "fhr_std", "up_mean", "up_std"):
            assert key in z.files
            assert np.isfinite(float(z[key]))
        assert float(z["fhr_std"]) > 0.0 and float(z["up_std"]) > 0.0

    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    for key in ("sequence_length", "render_mode", "raw_len", "decimation", "cells"):
        assert key in meta
    assert meta["raw_len"] == bd.RAW_LEN
    assert meta["decimation"] == bd.DECIMATION
    assert meta["sequence_length"] == bd.T_TILDE
    assert meta["render_mode"] == "direct"
    assert len(meta["cells"]) == len(tiny_cache_v4["cells"])
    manifest0 = meta["cells"][0]
    for field in ("cell_id", "target_te", "D", "B_y_scalar", "te_block_realised", "is_null"):
        assert field in manifest0


# ===========================================================================
# S2-T04: fingerprint resume
# ===========================================================================
def test_fingerprint_resume_v4(v4_config, tmp_path):
    r"""Unchanged rebuild regenerates nothing; a D change invalidates + regenerates."""
    cells, _ = enumerate_cells_v4(
        v4_config, benchmark="G1_raw_v4", target_te_grid=[0.0, 2.0], lag_grid=[8],
    )
    parts_dir = tmp_path / "_parts"
    paths = bd.build_split_parts_v4(
        cells, "train", 4, config=v4_config, benchmark="G1_raw_v4",
        render_mode="direct", base_seed=0, parts_dir=parts_dir,
    )
    mtimes1 = {p.name: p.stat().st_mtime_ns for p in paths}

    # Re-run identical: no regeneration (fingerprints match).
    bd.build_split_parts_v4(
        cells, "train", 4, config=v4_config, benchmark="G1_raw_v4",
        render_mode="direct", base_seed=0, parts_dir=parts_dir,
    )
    mtimes2 = {p.name: p.stat().st_mtime_ns for p in paths}
    assert mtimes1 == mtimes2

    # Change D -> fingerprint mismatch -> regeneration.
    cells_d6, _ = enumerate_cells_v4(
        v4_config, benchmark="G1_raw_v4", target_te_grid=[0.0, 2.0], lag_grid=[6],
    )
    bd.build_split_parts_v4(
        cells_d6, "train", 4, config=v4_config, benchmark="G1_raw_v4",
        render_mode="direct", base_seed=0, parts_dir=parts_dir,
    )
    mtimes3 = {p.name: p.stat().st_mtime_ns for p in paths}
    assert all(mtimes3[k] != mtimes2[k] for k in mtimes2)


# ===========================================================================
# S2-T05: build_all driver writes a complete cache
# ===========================================================================
def test_build_all_v4_writes_complete_cache(tiny_cache_v4):
    r"""The tiny build produced a complete cache (splits + meta + norm_stats)."""
    cache_dir = tiny_cache_v4["cache_dir"]
    for split in ("train", "val", "test"):
        assert (cache_dir / f"{split}.npz").is_file()
    assert (cache_dir / "meta.json").is_file()
    assert (cache_dir / "norm_stats.npz").is_file()

    n_by_split = tiny_cache_v4["n_override"]
    n_cells = len(tiny_cache_v4["cells"])
    for split, n_per_cell in n_by_split.items():
        with np.load(cache_dir / f"{split}.npz") as z:
            assert z["fhr"].shape == (n_cells * n_per_cell, bd.RAW_LEN)
            assert z["weight"].shape == (n_cells * n_per_cell, bd.T_TILDE)


# ===========================================================================
# S2-T06: am_carrier cache coexists with the direct cache
# ===========================================================================
def test_am_coexist_v4(v4_config, tmp_path):
    r"""The am_carrier arm builds a second cache under its own data_tag, both intact."""
    grid = {"target_te_grid": [0.0, 2.0], "lag_grid": [8]}
    nov = {"train": 4, "val": 2, "test": 2}

    direct_dir = tmp_path / "direct"
    bd.build_all_v4(v4_config, benchmark="G1_raw_v4", out_dir=direct_dir,
                    grid_override=grid, n_override=nov)

    am_config = resolve_arm_v4(v4_config, "am_carrier_prod")
    am_dir = tmp_path / "am"
    bd.build_all_v4(am_config, benchmark="G1_raw_v4", out_dir=am_dir,
                    grid_override=grid, n_override=nov)

    assert direct_dir != am_dir
    assert (direct_dir / "train.npz").is_file()
    assert (am_dir / "train.npz").is_file()
    direct_meta = json.loads((direct_dir / "meta.json").read_text(encoding="utf-8"))
    am_meta = json.loads((am_dir / "meta.json").read_text(encoding="utf-8"))
    assert direct_meta["render_mode"] == "direct"
    assert am_meta["render_mode"] == "am_carrier"
    # The arm re-keys the data_tag, so the default resolved cache dirs differ too.
    assert am_meta["data_tag"] == "G1_raw_v4_am"
