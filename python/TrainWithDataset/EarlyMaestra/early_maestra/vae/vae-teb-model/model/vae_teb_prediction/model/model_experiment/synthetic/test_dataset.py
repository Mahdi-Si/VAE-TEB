"""Pytest checks for ``dataset`` / ``build_dataset`` -- Phase 2 of the plan.

Verifies the cached-dataset round trip: :func:`build_dataset.build_dataset`
writes ``{train,val,test}.npz`` + ``meta.json`` + ``preview.pdf``,
:class:`SyntheticTEDataset` loads native-shaped samples, the collate path
produces ``AttributeDict`` batches, ``build_u_stream`` rebuilds the
101-channel source, and one batch flows cleanly through
``SeqVaeLagAttnV1.forward`` (Task 2.5). Run from the repo root with
``python -m pytest``.
"""

from pathlib import Path

import pytest
import torch
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    te_block_gaussian,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset import (
    build_dataset,
    load_config,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    AttributeDict,
    SyntheticTEDataset,
    build_u_stream,
    make_dataloader,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_active_benchmark,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1

_CONFIG_PATH = Path(__file__).resolve().parent / "config_synth.yaml"
_T = 300
_N_TRAIN, _N_VAL, _N_TEST = 6, 4, 4


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory) -> Path:
    """Build a tiny on-disk Benchmark-A cache once for the whole module.

    Returns:
        The cache directory ``<tmp>/A/test_tiny`` holding the three ``.npz``
        splits, ``meta.json`` and ``preview.pdf``.
    """
    tmp = tmp_path_factory.mktemp("synth_cache")
    config = load_config(_CONFIG_PATH)
    config["paths"]["data_dir"] = str(tmp)  # absolute -> overrides ./data
    config["experiment"]["tag"] = "test_tiny"
    config["data"]["n_train"] = _N_TRAIN
    config["data"]["n_val"] = _N_VAL
    config["data"]["n_test"] = _N_TEST
    out_dir = build_dataset(config, force=True)
    return out_dir


def test_build_writes_all_artifacts(tiny_cache: Path):
    for fname in ("train.npz", "val.npz", "test.npz", "meta.json", "preview.pdf"):
        assert (tiny_cache / fname).is_file(), fname
    assert (tiny_cache / "preview.pdf").stat().st_size > 0


def test_dataset_shapes_and_dtype(tiny_cache: Path):
    ds = SyntheticTEDataset(tiny_cache / "train.npz")
    assert len(ds) == _N_TRAIN
    sample = ds[0]
    assert isinstance(sample, AttributeDict)
    assert sample.fhr_st.shape == (_T, 43)
    assert sample.fhr_ph.shape == (_T, 44)
    assert sample.up_st.shape == (_T, 43)
    assert sample.up_ph.shape == (_T, 58)
    assert sample.weight.shape == (_T,)
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        assert sample[field].dtype == torch.float32
    assert torch.all(sample.weight == 1.0)
    assert isinstance(sample.guid, str) and "test_tiny_train" in sample.guid
    assert isinstance(sample.te_true, float)
    assert sample.true_lag_band.dtype == torch.long
    assert sample.true_lag_band.shape == (30,)


def test_collate_batch(tiny_cache: Path):
    ds = SyntheticTEDataset(tiny_cache / "train.npz")
    loader = make_dataloader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, AttributeDict)
    assert batch.fhr_st.shape == (3, _T, 43)
    assert batch.fhr_ph.shape == (3, _T, 44)
    assert batch.up_st.shape == (3, _T, 43)
    assert batch.up_ph.shape == (3, _T, 58)
    assert batch.weight.shape == (3, _T)
    assert batch.true_lag_band.shape == (3, 30)
    assert len(batch.guid) == 3 and all(isinstance(g, str) for g in batch.guid)


def test_build_u_stream(tiny_cache: Path):
    ds = SyntheticTEDataset(tiny_cache / "train.npz")
    loader = make_dataloader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    u_stream = build_u_stream(batch)
    assert u_stream.shape == (3, _T, 101)
    # The first 43 channels must be exactly up_st, the remaining 58 up_ph.
    assert torch.equal(u_stream[..., :43], batch.up_st)
    assert torch.equal(u_stream[..., 43:], batch.up_ph)


def test_meta_json_roundtrip(tiny_cache: Path):
    ds = SyntheticTEDataset(tiny_cache / "train.npz")
    meta = ds.meta
    assert meta["benchmark"] == "A"
    assert meta["tag"] == "test_tiny"
    expected_te = te_block_gaussian(
        meta["a"], meta["sigma2"], meta["horizon"], meta["M"]
    )
    assert meta["te_true"] == pytest.approx(expected_te, abs=1e-6)
    assert meta["split_sizes"] == {
        "train": _N_TRAIN, "val": _N_VAL, "test": _N_TEST
    }
    assert set(meta["split_seeds"]) == {"train", "val", "test"}
    assert meta["channel_map"]["fhr_st"] == [0, 43]
    assert meta["channel_map"]["up_ph"] == [43, 101]


@pytest.mark.parametrize("benchmark", ["B", "C", "E", "G"])
def test_build_dataset_dispatch(tmp_path, benchmark):
    """Phase 7: build_dataset dispatches to the B / C / E / G generators."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["experiment"]["benchmark"] = benchmark
    raw["experiment"]["tag"] = f"test_{benchmark}"
    config = resolve_active_benchmark(raw)
    config["paths"]["data_dir"] = str(tmp_path)  # absolute -> overrides ./data
    config["data"]["n_train"] = 6
    config["data"]["n_val"] = 4
    config["data"]["n_test"] = 4

    out_dir = build_dataset(config, force=True)
    for fname in ("train.npz", "val.npz", "test.npz", "meta.json", "preview.pdf"):
        assert (out_dir / fname).is_file(), fname

    ds = SyntheticTEDataset(out_dir / "train.npz")
    assert ds.meta["benchmark"] == benchmark
    sample = ds[0]
    assert sample.fhr_st.shape == (_T, 43)
    assert sample.fhr_ph.shape == (_T, 44)
    assert sample.up_st.shape == (_T, 43)
    assert sample.up_ph.shape == (_T, 58)

    # Benchmark-specific metadata.
    if benchmark == "B":
        assert "rho" in ds.meta and "burn_in" in ds.meta
        assert ds.meta["te_true"] > 0.0
    elif benchmark == "C":
        assert "q" in ds.meta and "obs_noise" in ds.meta
        assert "a" not in ds.meta and "sigma2" not in ds.meta
    elif benchmark == "E":
        assert "lag_band_1" in ds.meta and "lag_band_2" in ds.meta
        assert "te_true_1" in ds.meta and "te_true_2" in ds.meta
    elif benchmark == "G":
        assert ds.meta["te_true"] == 0.0
        assert ds.meta["reverse_roles"] is True
        assert ds.meta["true_lag_band"] == []


def test_forward_pass_smoke(tiny_cache: Path):
    """Task 2.5: one cached batch flows through ``SeqVaeLagAttnV1.forward``."""
    torch.manual_seed(0)
    ds = SyntheticTEDataset(tiny_cache / "train.npz")
    loader = make_dataloader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = SeqVaeLagAttnV1()  # defaults already equal the native shapes.
    model.eval()
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, build_u_stream(batch))

    expected_keys = {
        "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z",
        "target_state", "source_state", "decoder_state", "attended_source",
        "attn_weights", "mu_base", "logvar_base", "delta_mu_src", "mu_full",
        "logvar_full", "raw_future_pred", "kld_per_t", "te_lag_map",
        "warmup_mask", "mu_prior_sat_frac", "delta_mu_sat_frac",
    }
    assert set(out.keys()) == expected_keys
    assert out["kld_per_t"].shape == (2, _T)
    assert out["mu_full"].shape == (2, _T, 30, 87)
    assert out["te_lag_map"].shape == (2, _T, 91)
    assert out["attn_weights"].shape == (2, _T, 4, 91)
    assert torch.isfinite(out["kld_per_t"]).all()
    assert torch.isfinite(out["mu_full"]).all()
