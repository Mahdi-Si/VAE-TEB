r"""Unit tests for the ``mixed_dataset`` builder.

Covers the pure enumeration / holdout / memoisation logic, the channel-decomp
closure across $M$, and a tiny real build round-trip (per-sample provenance,
deterministic shuffle, collate of mixed ``delay_max``, and backward-compatible
loading of a homogeneous cache). The Monte-Carlo budgets are shrunk so the
build round-trip stays in the low-seconds range on CPU.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    mixed_dataset as MD,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    SyntheticTEDataset,
    make_dataloader,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    load_config,
    resolve_active_benchmark,
)

_CONFIG = Path(__file__).resolve().parent / "config_synth.yaml"


def _tiny_config(
    *,
    m_grid=(8, 16),
    target_te_grid=(1.5, 3.0),
    lag_bands=None,
    holdout=None,
    n_train=4,
    n_val=2,
    n_test=3,
    te_n_samples=900,
):
    """Build a benchmark-resolved G1_mix config with a shrunk grid + MC budget."""
    cfg = load_config(_CONFIG)
    cfg["experiment"]["benchmark"] = "G1_mix"
    resolve_active_benchmark(cfg)
    mix = cfg["benchmarks"]["G1_mix"]["mix"]
    mix["m_grid"] = list(m_grid)
    mix["target_te_grid"] = list(target_te_grid)
    mix["lag_bands"] = lag_bands or {"mid": [1, 15]}
    mix["holdout"] = holdout if holdout is not None else []
    mix["n_per_cell_train"] = n_train
    mix["n_per_cell_val"] = n_val
    mix["n_per_cell_test"] = n_test
    mix["inverter"] = {"n_samples": te_n_samples, "lo": 1e-4, "hi": 10.0,
                       "tol": 0.05, "max_iter": 8}
    cfg["data"]["te_n_samples"] = te_n_samples
    cfg["data"]["sequence_length"] = 120
    cfg["model"]["sequence_length"] = 120
    cfg["model"]["horizon"] = 30
    return cfg


# ---------------------------------------------------------------------------
# verify_holdout_marginals
# ---------------------------------------------------------------------------

def test_verify_holdout_marginals_passes_default():
    """The shipped default holdout leaves every marginal in a trained cell."""
    cfg = load_config(_CONFIG)
    mix = cfg["benchmarks"]["G1_mix"]["mix"]
    holdout = [(int(m), float(t), str(b)) for m, t, b in mix["holdout"]]
    MD.verify_holdout_marginals(
        holdout, mix["m_grid"], mix["target_te_grid"], list(mix["lag_bands"]),
    )


def test_verify_holdout_rejects_off_grid_value():
    with pytest.raises(ValueError, match="not in m_grid"):
        MD.verify_holdout_marginals(
            [(99, 1.5, "mid")], [8, 16, 32], [0.1, 1.5], ["mid"],
        )


def test_verify_holdout_rejects_full_marginal_removal():
    """Holding out the only cell carrying an M removes that whole marginal."""
    with pytest.raises(ValueError, match="every trained cell with M=16"):
        MD.verify_holdout_marginals(
            [(16, 1.5, "mid")], [8, 16], [1.5], ["mid"],
        )


# ---------------------------------------------------------------------------
# _solve_cell_b_y memoisation
# ---------------------------------------------------------------------------

def test_solve_cell_b_y_memoised(monkeypatch):
    r"""Equal $(d_{\min}, d_{\max}, \mathrm{TE}/M)$ keys hit the inverter once."""
    calls = {"n": 0}

    def _fake_inverter(*, target_te_block, **_kw):
        calls["n"] += 1
        return {"B_y_scalar": 0.01 * float(target_te_block), "te_block": float(target_te_block)}

    monkeypatch.setattr(MD, "B_y_for_mean_te_block_state_space", _fake_inverter)
    cache = {}
    common = dict(
        oscillators=[(0.99, 0.05)], target_ar=0.95, sigma2_y=1.0,
        sigma2_eta=0.01, horizon=30, K_history=160,
        inverter_cfg={"n_samples": 100}, cache=cache,
    )
    # (1, 15, 0.05) twice -> one inverter call; a distinct target -> a second.
    a = MD._solve_cell_b_y(1, 15, 0.05, **common)
    b = MD._solve_cell_b_y(1, 15, 0.05, **common)
    c = MD._solve_cell_b_y(1, 15, 0.10, **common)
    assert calls["n"] == 2
    assert a == b
    assert c["te_block"] == pytest.approx(0.10)


def test_enumerate_cells_uses_per_channel_target(monkeypatch):
    """The inverter is called with TE/M and the cell TE is the solve x M."""
    seen = []

    def _fake_inverter(*, target_te_block, **_kw):
        seen.append(float(target_te_block))
        return {"B_y_scalar": 0.5, "te_block": float(target_te_block)}

    monkeypatch.setattr(MD, "B_y_for_mean_te_block_state_space", _fake_inverter)
    # Force the floor low so nothing is trimmed.
    monkeypatch.setattr(MD, "mean_te_block_state_space_over_delays",
                        lambda **_kw: 1e-6)
    cfg = _tiny_config(m_grid=(8, 16), target_te_grid=(1.5,),
                       lag_bands={"mid": [1, 15]})
    cells, dropped = MD.enumerate_mix_cells(cfg)
    assert dropped == []
    by_M = {c.M: c for c in cells}
    assert set(by_M) == {8, 16}
    assert by_M[8].per_channel_target == pytest.approx(1.5 / 8)
    assert by_M[16].per_channel_target == pytest.approx(1.5 / 16)
    # cell TE = per-channel solve x M = target, held fixed across M.
    assert by_M[8].te_cell_realised == pytest.approx(1.5)
    assert by_M[16].te_cell_realised == pytest.approx(1.5)


def test_enumerate_trims_below_floor_but_protects_holdout(monkeypatch):
    """A high floor trims the low-TE/high-M cell; a trimmed holdout is fatal."""
    monkeypatch.setattr(MD, "B_y_for_mean_te_block_state_space",
                        lambda *, target_te_block, **_kw: {
                            "B_y_scalar": 0.5, "te_block": float(target_te_block)})
    # floor*margin = 0.08*1.5 = 0.12 trims (16, 1.5) [1.5/16=0.094] but keeps
    # (16, 3.0) [3/16=0.188] and both M=8 cells, so M=16 still has a trained
    # cell and verify_holdout_marginals passes.
    monkeypatch.setattr(MD, "mean_te_block_state_space_over_delays",
                        lambda **_kw: 0.08)
    cfg = _tiny_config(m_grid=(8, 16), target_te_grid=(1.5, 3.0),
                       lag_bands={"mid": [1, 15]})
    cells, dropped = MD.enumerate_mix_cells(cfg)
    triples = {(c.M, c.target_te) for c in cells}
    assert (16, 1.5) not in triples               # trimmed below the floor
    assert (16, 3.0) in triples and (8, 1.5) in triples
    assert any(d["M"] == 16 and d["target_te"] == 1.5 for d in dropped)
    # Now make that trimmed cell a holdout -> hard error (not a silent drop).
    cfg["benchmarks"]["G1_mix"]["mix"]["holdout"] = [[16, 1.5, "mid"]]
    with pytest.raises(ValueError, match="held-out cell.*unreachable"):
        MD.enumerate_mix_cells(cfg)


# ---------------------------------------------------------------------------
# channel-decomp closure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M", [8, 16, 32])
def test_channel_decomp_closes_for_all_M(M):
    """n_self / n_dist stay non-negative for every informative width."""
    from model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset import (
        _resolve_channel_decomp,
    )
    cfg = load_config(_CONFIG)
    data = cfg["benchmarks"]["G1_mix"]["data"]
    decomp = _resolve_channel_decomp({**data, "M": M}, 87, 101, "G1")
    assert decomp["m"] == M
    assert decomp["n_self"] >= 0
    assert decomp["n_dist"] >= 0
    assert decomp["m"] + decomp["n_self"] + decomp["n_smallnoise"] == 87
    assert decomp["m_source"] + decomp["n_dist"] + decomp["n_noise"] == 101


# ---------------------------------------------------------------------------
# build round-trip
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    """Build a 2-cell mixed cache (in-mix + holdout) once for the module."""
    tmp = tmp_path_factory.mktemp("g1mix")
    # te_n_samples=1500 keeps the MC floor low enough that the interior
    # held-out (16, 3.0, mid) cell stays reachable; (16, 3.0) has the largest
    # M=16 per-channel target so it is the safest holdout for a tiny build.
    cfg = _tiny_config(
        m_grid=(8, 16), target_te_grid=(1.5, 3.0),
        lag_bands={"short": [1, 8], "mid": [1, 15]},
        holdout=[[16, 3.0, "mid"]], n_train=4, n_val=2, n_test=3,
        te_n_samples=1500,
    )
    cfg["paths"]["data_dir"] = str(tmp)
    cfg["experiment"]["tag"] = "tiny"
    in_dir = MD.build_g1_mix(copy.deepcopy(cfg), force=True, holdout=False)
    ho_dir = MD.build_g1_mix(copy.deepcopy(cfg), force=True, holdout=True)
    return {"in": in_dir, "ho": ho_dir, "cfg": cfg}


def test_build_writes_splits_and_manifest(tiny_cache):
    in_dir = tiny_cache["in"]
    for f in ("train.npz", "val.npz", "test.npz", "meta.json"):
        assert (in_dir / f).is_file()
    meta = json.loads((in_dir / "meta.json").read_text())
    assert meta["benchmark"] == "G1_mix"
    assert meta["mixture"]["standardize_mode"] == "per_cell"
    assert meta["mixture"]["cells"], "manifest carries the per-cell records"
    # In-mix excludes the held-out (16, 3.0, mid) cell.
    triples = {(c["M"], c["target_te"], c["band"]) for c in meta["mixture"]["cells"]}
    assert (16, 3.0, "mid") not in triples


def test_npz_has_provenance_arrays(tiny_cache):
    with np.load(tiny_cache["in"] / "train.npz") as npz:
        for key in ("sample_te_true", "sample_M", "sample_delay_min",
                    "sample_delay_max", "sample_band_id", "sample_cell_id",
                    "sample_held_out"):
            assert key in npz.files
        n = npz["fhr_st"].shape[0]
        assert npz["sample_M"].shape == (n,)
        assert set(np.unique(npz["sample_held_out"]).tolist()) == {0}


def test_dataset_exposes_per_sample_fields(tiny_cache):
    ds = SyntheticTEDataset(tiny_cache["in"] / "train.npz")
    s = ds[0]
    for key in ("te_true", "M", "delay_min", "delay_max", "band_id",
                "cell_id", "held_out"):
        assert key in s
    assert s["M"] in (8, 16)
    assert s["held_out"] == 0
    # te_true is the per-sample cell TE, not a single dataset scalar.
    te_vals = {float(ds[i]["te_true"]) for i in range(len(ds))}
    assert len(te_vals) >= 2


def test_mixed_delay_max_collates(tiny_cache):
    """A batch mixing different delay_max collates cleanly to (B,)."""
    ds = SyntheticTEDataset(tiny_cache["in"] / "train.npz")
    loader = make_dataloader(ds, batch_size=len(ds), shuffle=False)
    batch = next(iter(loader))
    assert batch["delay_max"].shape == (len(ds),)
    assert batch["M"].shape == (len(ds),)
    assert batch["te_true"].shape == (len(ds),)


def test_holdout_cache_is_test_only_and_flagged(tiny_cache):
    ho_dir = tiny_cache["ho"]
    assert (ho_dir / "test.npz").is_file()
    assert not (ho_dir / "train.npz").is_file()
    ds = SyntheticTEDataset(ho_dir / "test.npz")
    assert all(int(ds[i]["held_out"]) == 1 for i in range(len(ds)))
    meta = json.loads((ho_dir / "meta.json").read_text())
    triples = {(c["M"], c["target_te"], c["band"]) for c in meta["mixture"]["cells"]}
    assert triples == {(16, 3.0, "mid")}


def test_deterministic_build(tmp_path):
    """Same build_seed -> identical pooled arrays (row-aligned shuffle)."""
    cfg = _tiny_config(m_grid=(8,), target_te_grid=(1.5,),
                       lag_bands={"mid": [1, 15]}, n_train=4, n_val=2, n_test=2)
    cfg["paths"]["data_dir"] = str(tmp_path / "a")
    cfg["experiment"]["tag"] = "det"
    d1 = MD.build_g1_mix(copy.deepcopy(cfg), force=True)
    cfg2 = copy.deepcopy(cfg)
    cfg2["paths"]["data_dir"] = str(tmp_path / "b")
    d2 = MD.build_g1_mix(cfg2, force=True)
    with np.load(d1 / "train.npz") as a, np.load(d2 / "train.npz") as b:
        assert np.array_equal(a["fhr_st"], b["fhr_st"])
        assert np.array_equal(a["sample_cell_id"], b["sample_cell_id"])


def test_homogeneous_cache_loads_without_provenance(tmp_path):
    """A cache lacking sample_* keys loads with provenance disabled (BC)."""
    T = 40
    np.savez(
        str(tmp_path / "train.npz"),
        fhr_st=np.zeros((3, T, 43), np.float32),
        fhr_ph=np.zeros((3, T, 44), np.float32),
        up_st=np.zeros((3, T, 43), np.float32),
        up_ph=np.zeros((3, T, 58), np.float32),
        weight=np.ones((3, T), np.float32),
    )
    (tmp_path / "meta.json").write_text(json.dumps({
        "te_true": 1.23, "true_lag_band": [0, 1, 2], "tag": "homo",
        "benchmark": "G1",
    }))
    ds = SyntheticTEDataset(tmp_path / "train.npz")
    s = ds[0]
    assert float(s["te_true"]) == pytest.approx(1.23)
    assert "M" not in s and "cell_id" not in s
    assert ds._provenance is None
