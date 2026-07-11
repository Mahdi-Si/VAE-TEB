r"""Tests for the Sprint 3 cell enumeration + pilot generation (S3-T01).

Covers :func:`build_dataset_v2.enumerate_cells_v2` (null cell kept with $B = 0$,
signal cells solved within the inverter tolerance, unsolvable cells logged and
dropped rather than crashing, and ``band`` lag mode enumerating per-sample lag
windows), the deterministic per-(cell, split) seed, and
:func:`build_dataset_v2.generate_pilot_samples`.

The inverter Monte-Carlo is the slow part, so a trimmed grid + smaller ``n_samples``
is used throughout. See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 3, S3-T01.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    build_dataset_v2 as bd,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"


@pytest.fixture(scope="module")
def base_config() -> dict:
    r"""Load the parsed ``config_synth_v2.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _fast_config(base: dict) -> dict:
    r"""A trimmed copy: a small grid and a cheaper inverter Monte-Carlo."""
    cfg = copy.deepcopy(base)
    mix = cfg["benchmarks"]["G1_raw"]["mix"]
    mix["target_te_grid"] = [0.0, 1.0, 2.0]
    mix["lag_grid"] = [8]
    mix["inverter"]["n_samples"] = 4000
    return cfg


@pytest.fixture(scope="module")
def fast_config(base_config) -> dict:
    return _fast_config(base_config)


@pytest.fixture(scope="module")
def enumerated(fast_config):
    r"""Enumerate the trimmed grid once (the inverter solve is the slow part)."""
    return bd.enumerate_cells_v2(fast_config)


# ---------------------------------------------------------------------------
# S3-T01: enumeration
# ---------------------------------------------------------------------------


def test_enumerate_null_and_signal_cells(enumerated) -> None:
    r"""Null cell has ``B=0, te=0``; signal cells solve within tol; ids are contiguous."""
    cells, dropped = enumerated
    assert dropped == []
    # 3 target_te x 1 lag = 3 kept cells.
    assert len(cells) == 3
    assert [c.cell_id for c in cells] == [0, 1, 2]

    null = [c for c in cells if c.target_te == 0.0]
    assert len(null) == 1
    assert null[0].B_y_scalar == 0.0
    assert null[0].te_block_realised == 0.0

    for cell in cells:
        if cell.target_te == 0.0:
            continue
        assert cell.B_y_scalar > 0.0
        # achieved block TE lands on the requested target within a few percent.
        assert abs(cell.te_block_realised - cell.target_te) / cell.target_te < 0.05


def test_enumerate_dedup_and_types(enumerated) -> None:
    r"""Every kept cell is a :class:`CellV2`; the fixed lag is stamped on each."""
    cells, _ = enumerated
    assert all(isinstance(c, bd.CellV2) for c in cells)
    assert {c.D for c in cells} == {8}


def _band_config(base: dict) -> dict:
    r"""A trimmed band-mode copy: one tight long window + a cheap inverter Monte-Carlo."""
    cfg = copy.deepcopy(base)
    mix = cfg["benchmarks"]["G1_raw"]["mix"]
    mix["target_te_grid"] = [0.0, 2.0]
    mix["lag_mode"] = "band"
    mix["lag_bands"] = [[45, 52]]
    mix["lag_band_units"] = "steps"
    mix["inverter"]["n_samples"] = 4000
    # K_history must span the longest lag (>= 52); the fixed default (80) already does, but
    # the long-lag regime uses 140 for margin.
    cfg["benchmarks"]["G1_raw"]["data"]["K_history"] = 140
    return cfg


def test_enumerate_band_mode(base_config) -> None:
    r"""``lag_mode: band`` enumerates ``(te, [lo, hi])`` cells with a per-delay TE map."""
    cfg = _band_config(base_config)
    cells, dropped = bd.enumerate_cells_v2(cfg)
    assert dropped == []
    # 2 target_te x 1 band = 2 cells.
    assert len(cells) == 2
    for cell in cells:
        assert cell.lag_mode == "band"
        assert (cell.delay_min, cell.delay_max) == (45, 52)
        # te_by_delay covers the full inclusive window.
        assert set(cell.te_by_delay.keys()) == set(range(45, 53))

    null = [c for c in cells if c.target_te == 0.0][0]
    assert null.B_y_scalar == 0.0
    assert all(v == 0.0 for v in null.te_by_delay.values())

    signal = [c for c in cells if c.target_te > 0.0][0]
    assert signal.B_y_scalar > 0.0
    # The mean block TE over the window lands on the requested target within a few percent.
    mean_te = float(np.mean(list(signal.te_by_delay.values())))
    assert abs(mean_te - signal.target_te) / signal.target_te < 0.05


def test_band_and_fixed_solve_coincide_at_degenerate_window(base_config) -> None:
    r"""A degenerate band ``[D, D]`` solves the same coupling as fixed lag ``D``."""
    cfg = copy.deepcopy(base_config)
    cfg["benchmarks"]["G1_raw"]["mix"]["inverter"]["n_samples"] = 4000
    fixed = bd.solve_cell_coupling(cfg, 2.0, 8)            # fixed: delay_max defaults to None
    band = bd.solve_cell_coupling(cfg, 2.0, 8, 8)         # degenerate band [8, 8]
    assert abs(fixed["B_y_scalar"] - band["B_y_scalar"]) < 1e-9
    assert "te_by_delay" not in fixed  # fixed cells carry no per-delay map


def test_enumerate_drops_unsolvable(fast_config) -> None:
    r"""An out-of-bracket target is logged + dropped (not raised); the null survives."""
    cfg = copy.deepcopy(fast_config)
    # A tiny hi bracket cannot reach TE=1 or 2 nats -> the inverter raises -> drop.
    cfg["benchmarks"]["G1_raw"]["mix"]["inverter"]["hi"] = 1.0e-3
    cells, dropped = bd.enumerate_cells_v2(cfg)
    # Only the null (target_te=0, solved without the bracket) survives.
    assert [c.target_te for c in cells] == [0.0]
    assert len(dropped) == 2
    assert {d["target_te"] for d in dropped} == {1.0, 2.0}
    assert all("reason" in d for d in dropped)


def test_enumerate_grid_override(fast_config) -> None:
    r"""Explicit grid overrides bypass the config grid (used by the pilot pre-flight)."""
    cells, _ = bd.enumerate_cells_v2(
        fast_config, target_te_grid=[0.0, 2.0], lag_grid=[4, 8]
    )
    # 2 target_te x 2 lags = 4 cells.
    assert len(cells) == 4
    assert {(c.target_te, c.D) for c in cells} == {
        (0.0, 4), (0.0, 8), (2.0, 4), (2.0, 8)
    }


# ---------------------------------------------------------------------------
# S3-T01: seeds + pilot generation
# ---------------------------------------------------------------------------


def test_cell_seed_distinct_across_splits_and_cells() -> None:
    r"""No two (cell, split) units share a seed; a bad split name raises."""
    seeds = {
        (cid, split): bd.cell_seed(0, cid, split)
        for cid in range(3)
        for split in ("train", "val", "test")
    }
    assert len(set(seeds.values())) == len(seeds)
    with pytest.raises(ValueError, match="unknown split"):
        bd.cell_seed(0, 0, "holdout")


def test_generate_pilot_samples_shapes(enumerated, fast_config) -> None:
    r"""Pilot generation returns raw pairs of the right shape and stamps the cell id."""
    cells, _ = enumerated
    signal = next(c for c in cells if c.target_te > 0.0)
    raw = bd.generate_pilot_samples(signal, 4, "train", fast_config)
    assert raw["fhr_raw"].shape == (4, 5280)
    assert raw["up_raw"].shape == (4, 5280)
    assert raw["meta"]["cell_id"] == signal.cell_id
    assert raw["meta"]["D"] == signal.D
    assert raw["true_lag_tt"].shape[0] == 4


def test_generate_pilot_samples_deterministic(enumerated, fast_config) -> None:
    r"""Same (cell, split, base_seed) -> identical raw arrays; different splits differ."""
    cells, _ = enumerated
    cell = next(c for c in cells if c.target_te > 0.0)
    a = bd.generate_pilot_samples(cell, 3, "train", fast_config, base_seed=0)
    b = bd.generate_pilot_samples(cell, 3, "train", fast_config, base_seed=0)
    c_val = bd.generate_pilot_samples(cell, 3, "val", fast_config, base_seed=0)
    assert np.array_equal(a["fhr_raw"], b["fhr_raw"])
    assert np.array_equal(a["up_raw"], b["up_raw"])
    assert not np.array_equal(a["fhr_raw"], c_val["fhr_raw"])


# ---------------------------------------------------------------------------
# S4-T02/T03: full build -> cache -> loaders (needs the real scattering transform)
# ---------------------------------------------------------------------------
#
# These build a *tiny* real cache (2 cells, small N) at the production shape 5280 so the
# 43/44/43/58 channel counts are genuine, then exercise the cache schema, the shared
# row-aligned shuffle, determinism, resume, the loader field mapping, and a real
# ``model.forward`` on a built batch. The scattering adapter (an expensive filter bank)
# is built once per module and reused across builds.

# v2 provenance keys the cache MUST expose and the v1 keys it MUST NOT.
_V2_PROVENANCE = (
    "sample_te_true", "sample_te_scat", "sample_frac_phi",
    "sample_delay", "sample_cell_id", "sample_held_out", "sample_raw_index",
)
_V1_ONLY_PROVENANCE = (
    "sample_M", "sample_delay_min", "sample_delay_max", "sample_band_id",
)


def _tiny_build_config(base: dict) -> dict:
    r"""A trimmed build config: 2 cells (1 null + 1 signal), tiny N, cheaper inverter."""
    cfg = copy.deepcopy(base)
    mix = cfg["benchmarks"]["G1_raw"]["mix"]
    mix["target_te_grid"] = [0.0, 2.0]
    mix["lag_grid"] = [8]
    mix["inverter"]["n_samples"] = 4000
    mix["n_per_cell_train"] = 8
    mix["n_per_cell_val"] = 6
    mix["n_per_cell_test"] = 6
    return cfg


@pytest.fixture(scope="module")
def tiny_cfg(base_config) -> dict:
    return _tiny_build_config(base_config)


@pytest.fixture(scope="module")
def adapter(base_config):
    r"""One production-shape scattering adapter, reused across every build in this module."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.scattering_adapter import (
        ScatteringAdapter,
    )
    return ScatteringAdapter(base_config)


@pytest.fixture(scope="module")
def built_cache(tiny_cfg, adapter, tmp_path_factory):
    r"""Build the tiny cache once; return ``(out_dir, cells)``."""
    out_dir = tmp_path_factory.mktemp("v2_cache")
    bd.build_all(tiny_cfg, out_dir=out_dir, adapter=adapter)
    cells, _ = bd.enumerate_cells_v2(tiny_cfg)
    return out_dir, cells


def test_build_cache_provenance_and_schema(built_cache) -> None:
    r"""S4-T02a/b: the cache carries §17 shapes + v2 provenance, no M/band fields."""
    out_dir, _cells = built_cache
    n_expected = {"train": 2 * 8, "val": 2 * 6, "test": 2 * 6}
    for split, n in n_expected.items():
        with np.load(out_dir / f"{split}.npz") as npz:
            keys = set(npz.files)
            # Native model-facing fields at the §17 shapes.
            assert npz["fhr_st"].shape == (n, 300, 43)
            assert npz["fhr_ph"].shape == (n, 300, 44)
            assert npz["up_st"].shape == (n, 300, 43)
            assert npz["up_ph"].shape == (n, 300, 58)
            assert npz["weight"].shape == (n, 300)
            assert npz["true_lag_tt"].shape == (n, 300)
            # v2 provenance present and correctly shaped/typed; v1-only keys absent.
            for key in _V2_PROVENANCE:
                assert key in keys, f"{split}: missing {key}"
                assert npz[key].shape == (n,)
            for key in _V1_ONLY_PROVENANCE:
                assert key not in keys, f"{split}: v1-only key {key} leaked into v2 cache"
            assert npz["sample_delay"].dtype == np.int16
            assert npz["sample_cell_id"].dtype == np.int16
            assert npz["sample_held_out"].dtype == np.int8
            assert npz["sample_raw_index"].dtype == np.int32
            # Each cell contributes rows 0..n_per_cell-1; after the shared shuffle the
            # index still spans exactly [0, n_per_cell) (per-cell count = n / 2 cells).
            n_per_cell = n // 2
            assert npz["sample_raw_index"].min() == 0
            assert npz["sample_raw_index"].max() == n_per_cell - 1
            for key in ("sample_te_true", "sample_te_scat", "sample_frac_phi"):
                assert npz[key].dtype == np.float32
            # true_lag_tt is the flat fixed lag D=8 everywhere (fixed-lag mode).
            assert np.all(npz["true_lag_tt"] == 8)


def test_build_meta_manifest(built_cache) -> None:
    r"""S4-T02b: meta.json carries the v2 per-cell manifest + channel/coupled maps."""
    import json

    out_dir, cells = built_cache
    with open(out_dir / "meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["c_y"] == 87 and meta["c_u"] == 101
    assert meta["sequence_length"] == 300
    assert meta["channel_map"]["fhr_st"] == [0, 43]
    assert meta["channel_map"]["up_ph"] == [43, 101]
    assert set(meta["coupled_channel"]) == {"up_st", "fhr_st", "hz", "xi"}
    assert "m_grid" not in json.dumps(meta), "no m_grid anywhere in v2 meta"
    # Per-cell manifest: one entry per kept cell, each with the v2 measured fields.
    assert len(meta["cells"]) == len(cells)
    for entry in meta["cells"]:
        for key in ("cell_id", "target_te", "D", "B_y_scalar",
                    "te_block_realised", "te_scat_measured", "frac_phi"):
            assert key in entry, f"cell manifest missing {key}"
    # norm_stats.npz was persisted with <field>_<stat> keys.
    with np.load(out_dir / "norm_stats.npz") as stats:
        for field in ("fhr_st", "fhr_ph", "up_st", "up_ph"):
            assert f"{field}_mean" in stats.files
            assert f"{field}_std" in stats.files


def test_build_cache_shuffle_row_alignment(built_cache) -> None:
    r"""S4-T02b: the shared permutation shuffles the pool but keeps rows aligned."""
    out_dir, cells = built_cache
    te_by_cell = {int(c.cell_id): float(c.te_block_realised) for c in cells}
    with np.load(out_dir / "train.npz") as npz:
        cell_ids = np.asarray(npz["sample_cell_id"])
        te_true = np.asarray(npz["sample_te_true"])
        delay = np.asarray(npz["sample_delay"])
    # Row-alignment: each row's stamped TE / delay still matches its cell after shuffle.
    for i in range(cell_ids.shape[0]):
        assert te_true[i] == pytest.approx(te_by_cell[int(cell_ids[i])])
        assert int(delay[i]) == 8
    # The pool is genuinely shuffled (not left in per-cell blocks).
    assert not np.array_equal(cell_ids, np.sort(cell_ids))


def test_raw_provider_roundtrip(built_cache, tiny_cfg) -> None:
    r"""``make_raw_provider`` regenerates the exact raw window for a shuffled cache row.

    The cache stores only features, but each row's ``(sample_cell_id, sample_raw_index)`` is a
    deterministic regeneration key. The provider must return the same analysis-window slice as
    a direct :func:`generate_pilot_samples` for that cell/row -- so the samples_diag first panel
    shows the raw waveform that actually produced the row's features.
    """
    out_dir, cells = built_cache
    cells_by_id = {int(c.cell_id): c for c in cells}
    with np.load(out_dir / "test.npz") as npz:
        cid = np.asarray(npz["sample_cell_id"])
        ridx = np.asarray(npz["sample_raw_index"])

    provider = bd.make_raw_provider(tiny_cfg, "test", cache_dir=out_dir)
    win = slice(240, 5040)                      # TRIM_STEPS(15) * DECIMATION(16) each end
    # Check one row per cell (covers null + signal).
    for target_cid in sorted(cells_by_id):
        row = int(np.flatnonzero(cid == target_cid)[0])
        fhr_win, up_win = provider(int(cid[row]), int(ridx[row]))
        assert fhr_win.shape == (4800,) and up_win.shape == (4800,)
        raw = bd.generate_pilot_samples(cells_by_id[target_cid], 6, "test", tiny_cfg)
        exp_fhr = raw["fhr_raw"][int(ridx[row]), win].astype(np.float32)
        exp_up = raw["up_raw"][int(ridx[row]), win].astype(np.float32)
        assert np.allclose(fhr_win, exp_fhr, atol=1e-5)
        assert np.allclose(up_win, exp_up, atol=1e-5)


def test_raw_provider_feeds_dataset(built_cache, tiny_cfg) -> None:
    r"""A dataset with a ``raw_provider`` exposes ``fhr`` / ``up`` on each item; default omits."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        dataset_v2 as ds2,
    )

    out_dir, _ = built_cache
    provider = bd.make_raw_provider(tiny_cfg, "test", cache_dir=out_dir)
    with_raw = ds2.SyntheticTEDatasetV2(out_dir / "test.npz", raw_provider=provider)
    item = with_raw[0]
    assert "raw_index" in item
    assert item["fhr"].shape == (4800,) and item["up"].shape == (4800,)
    # Default (no provider): raw-free, so training / eval loaders never regenerate.
    plain = ds2.SyntheticTEDatasetV2(out_dir / "test.npz")
    assert "fhr" not in plain[0] and "up" not in plain[0]


def test_build_cache_deterministic(tiny_cfg, adapter, tmp_path_factory) -> None:
    r"""S4-T02b: rebuilding from the stored seeds yields byte-identical arrays."""
    dir_a = tmp_path_factory.mktemp("det_a")
    dir_b = tmp_path_factory.mktemp("det_b")
    bd.build_all(tiny_cfg, out_dir=dir_a, adapter=adapter)
    bd.build_all(tiny_cfg, out_dir=dir_b, adapter=adapter)
    for split in ("train", "val", "test"):
        with np.load(dir_a / f"{split}.npz") as a, np.load(dir_b / f"{split}.npz") as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                assert np.array_equal(a[key], b[key]), f"{split}:{key} not deterministic"


def test_build_cache_resume(tiny_cfg, adapter, tmp_path_factory) -> None:
    r"""S4-T02c: a re-run skips completed Stage-1 parts and reproduces the cache."""
    out_dir = tmp_path_factory.mktemp("resume")
    bd.build_all(tiny_cfg, out_dir=out_dir, adapter=adapter)
    part = out_dir / "_parts" / "train_cell000.npz"
    assert part.is_file()
    mtime_before = part.stat().st_mtime_ns
    with np.load(out_dir / "train.npz") as npz:
        first = {k: np.asarray(npz[k]).copy() for k in npz.files}
    # Re-run: the existing part must be skipped (mtime unchanged) and the final cache
    # must be identical.
    bd.build_all(tiny_cfg, out_dir=out_dir, adapter=adapter, resume=True)
    assert part.stat().st_mtime_ns == mtime_before, "resume rewrote a completed part"
    with np.load(out_dir / "train.npz") as npz:
        for key in first:
            assert np.array_equal(first[key], np.asarray(npz[key])), f"{key} changed on resume"


def test_build_resume_regenerates_on_n_change(tiny_cfg, adapter, tmp_path_factory) -> None:
    r"""S4-T02c: a changed n_per_cell invalidates the part fingerprint -> regenerate.

    Guards the config's documented 'scale up by re-running the build with larger counts'
    workflow: re-running into the same cache dir with a larger train N must NOT silently
    reuse the smaller stale parts.
    """
    out_dir = tmp_path_factory.mktemp("resume_n")
    bd.build_all(tiny_cfg, out_dir=out_dir, adapter=adapter,
                 n_override={"train": 8, "val": 6, "test": 6})
    with np.load(out_dir / "train.npz") as npz:
        assert npz["fhr_st"].shape[0] == 2 * 8
    # Re-run into the SAME dir with a larger train N: the 8-sample parts must be
    # regenerated (fingerprint mismatch), so the cache grows to the new count.
    bd.build_all(tiny_cfg, out_dir=out_dir, adapter=adapter,
                 n_override={"train": 10, "val": 6, "test": 6})
    with np.load(out_dir / "train.npz") as npz:
        assert npz["fhr_st"].shape[0] == 2 * 10, "stale part reused after n bump"
        assert np.all(npz["sample_cell_id"] < 2)


def test_loader_fields_and_no_dropped_keys(built_cache) -> None:
    r"""S4-T03: a v2 batch exposes the fields + v2 provenance and NO M/band keys."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        dataset_v2 as ds2,
    )

    out_dir, _ = built_cache
    dataset = ds2.SyntheticTEDatasetV2(out_dir / "train.npz")
    loader = ds2.make_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.fhr_st.shape == (4, 300, 43)
    assert batch.fhr_ph.shape == (4, 300, 44)
    assert batch.up_st.shape == (4, 300, 43)
    assert batch.up_ph.shape == (4, 300, 58)
    assert batch.weight.shape == (4, 300)
    # v2 per-sample metadata is present.
    for key in ("te_true", "te_scat", "frac_phi", "delay", "cell_id", "held_out", "guid"):
        assert key in batch, f"batch missing {key}"
    # v1 grouping keys are gone.
    for key in ("M", "band_id", "delay_min", "delay_max"):
        assert key not in batch, f"v1-only key {key} leaked into a v2 batch"
    # build_u_stream concatenates up_st(43) + up_ph(58) = 101.
    assert ds2.build_u_stream(batch).shape == (4, 300, 101)
    del batch, loader, dataset


def test_model_forward_compat(built_cache, base_config) -> None:
    r"""S4-T03: a built batch runs through the unchanged model.forward without errors."""
    import inspect

    import torch

    import sys

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        dataset_v2 as ds2,
    )

    # The repo has an irregular package layout: ``model/vae_teb_prediction/utils`` is a
    # regular package WITHOUT ``custom_logger``, and pytest inserts
    # ``model/vae_teb_prediction`` onto ``sys.path`` ahead of the repo root, shadowing
    # the real repo-root ``utils`` the model imports. Force the repo root to the front so
    # ``utils.custom_logger`` resolves before instantiating the model.
    _root = str(Path(__file__).resolve().parents[6])
    if _root in sys.path:
        sys.path.remove(_root)
    sys.path.insert(0, _root)
    from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1

    out_dir, _ = built_cache
    dataset = ds2.SyntheticTEDatasetV2(out_dir / "train.npz")
    loader = ds2.make_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    # Instantiate the model from the config's model block (filtered to constructor kwargs).
    model_cfg = dict(base_config["model"])
    accepted = set(inspect.signature(SeqVaeLagAttnV1.__init__).parameters)
    kwargs = {k: v for k, v in model_cfg.items() if k in accepted}
    model = SeqVaeLagAttnV1(**kwargs)
    model.eval()

    with torch.no_grad():
        out = model.forward(
            y_st=batch.fhr_st, y_ph=batch.fhr_ph, u_stream=ds2.build_u_stream(batch)
        )
    assert out["kld_per_t"].shape == (4, 300)
    num_heads = int(kwargs.get("num_heads", 4))
    max_lag = int(kwargs.get("max_lag", 90))
    assert out["attn_weights"].shape == (4, 300, num_heads, max_lag + 1)
    del batch, loader, dataset, model
