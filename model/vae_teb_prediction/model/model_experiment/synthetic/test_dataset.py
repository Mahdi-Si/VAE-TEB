"""Pytest checks for ``dataset`` / ``build_dataset`` (v2 benchmarks).

Verifies the cached-dataset round trip end-to-end:
:func:`build_dataset.build_dataset` writes ``{train,val,test}.npz`` +
``meta.json`` + ``preview.pdf`` for every v2 benchmark,
:class:`SyntheticTEDataset` loads native-shaped samples, the collate path
produces ``AttributeDict`` batches, ``build_u_stream`` rebuilds the
101-channel source, and one batch flows cleanly through
``SeqVaeLagAttnV1.forward``.

Sprint 3 rewrite: the v1 fixture / dispatch tests were retired alongside the
v1 generators; the scaffolding tests (shapes / collate / build_u_stream /
meta round-trip) are kept and re-parametrised over the v2 benchmarks.
Run from the repo root with ``python -m pytest``.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset import (
    _GENERATORS,
    _build_gen_kwargs,
    build_dataset,
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
_EXPECTED_FORWARD_KEYS = {
    "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z",
    "target_state", "source_state", "decoder_state", "attended_source",
    # C7: per-head attended summaries a^(m) and the additive per-head KL.
    "attended_source_heads",
    "attn_weights", "mu_base", "logvar_base", "delta_mu_src", "mu_full",
    "logvar_full", "raw_future_pred", "kld_per_t", "kld_per_t_per_head",
    "te_lag_map", "warmup_mask", "mu_prior_sat_frac", "delta_mu_sat_frac",
}


def _tiny_v2_config(benchmark: str, data_dir: Path) -> Dict[str, Any]:
    """Load ``config_synth.yaml`` with a tiny-cache override for ``benchmark``.

    The cache is rooted at ``data_dir`` (a per-test ``tmp_path``), the active
    benchmark is set to ``benchmark`` and the per-split counts are tiny so
    the build finishes in a few seconds.

    Args:
        benchmark: The v2 benchmark id (``"G1"`` / ``"G1-rev"`` / ``"G2"`` /
            ``"G3"``).
        data_dir: Absolute path used as ``paths.data_dir`` (overrides
            ``./data``).

    Returns:
        The benchmark-resolved config dict ready for :func:`build_dataset`.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["experiment"]["benchmark"] = benchmark
    raw["experiment"]["tag"] = f"test_{benchmark}"
    config = resolve_active_benchmark(raw)
    config["paths"]["data_dir"] = str(data_dir)  # absolute -> overrides ./data
    config["data"]["n_train"] = _N_TRAIN
    config["data"]["n_val"] = _N_VAL
    config["data"]["n_test"] = _N_TEST
    # G1 / G1-rev MC TE estimate is the slowest step at build time; shrink it
    # for the tiny smoke caches.
    if benchmark in ("G1", "G1-rev"):
        config["data"]["te_n_samples"] = 2_000
    return config


@pytest.fixture(scope="module")
def tiny_g1_cache(tmp_path_factory) -> Path:
    """Build a tiny on-disk G1 cache once for the whole module.

    Returns:
        The cache directory ``<tmp>/G1/test_G1`` holding the three ``.npz``
        splits, ``meta.json`` and ``preview.pdf``.
    """
    tmp = tmp_path_factory.mktemp("synth_cache_g1")
    config = _tiny_v2_config("G1", tmp)
    return build_dataset(config, force=True)


# --- Build / artifact tests --------------------------------------------------

@pytest.mark.parametrize("benchmark", ["G1", "G2", "G3"])
def test_build_writes_all_artifacts_v2(tmp_path, benchmark):
    """Each v2 generator emits the full cache directory contents."""
    config = _tiny_v2_config(benchmark, tmp_path)
    out_dir = build_dataset(config, force=True)
    for fname in ("train.npz", "val.npz", "test.npz", "meta.json", "preview.pdf"):
        assert (out_dir / fname).is_file(), fname
    assert (out_dir / "preview.pdf").stat().st_size > 0
    ds = SyntheticTEDataset(out_dir / "train.npz")
    assert ds.meta["benchmark"] == benchmark
    assert isinstance(ds.meta["te_true"], float)
    assert ds.meta["te_true"] >= 0.0  # G1/G2/G3 all carry non-negative TE


@pytest.mark.parametrize("benchmark", ["G1", "G1-rev", "G2", "G3"])
def test_build_dataset_dispatch_v2(benchmark):
    """``_build_gen_kwargs`` returns a kwargs dict valid for the target generator."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["experiment"]["benchmark"] = benchmark
    config = resolve_active_benchmark(raw)
    data, model = config["data"], config["model"]
    c_y = int(data["c_y_st"]) + int(data["c_y_ph"])
    c_u = int(data["c_u_st"]) + int(data["c_u_ph"])

    gen = _GENERATORS[benchmark]
    kwargs = _build_gen_kwargs(benchmark, data, model, c_y, c_u)

    import inspect
    sig = inspect.signature(gen)
    valid = set(sig.parameters.keys())
    extras = set(kwargs) - valid
    assert not extras, (
        f"_build_gen_kwargs[{benchmark!r}] emits unknown kwargs {extras}; "
        f"generator signature is {sorted(valid)}"
    )
    assert "T" in kwargs and kwargs["T"] == int(data["sequence_length"])
    assert kwargs["c_y"] == c_y and kwargs["c_u"] == c_u


def test_g1_rev_te_zero(tmp_path):
    """The G1-rev directionality variant has ``te_true == 0`` and an empty band."""
    config = _tiny_v2_config("G1-rev", tmp_path)
    out_dir = build_dataset(config, force=True)
    ds = SyntheticTEDataset(out_dir / "train.npz")
    assert ds.meta["benchmark"] == "G1-rev"
    assert ds.meta["te_true"] == 0.0
    assert ds.meta["true_lag_band"] == []
    assert ds.meta["reverse_roles"] is True
    assert ds.meta["direction"] == "Y_to_X"


# --- Forward-pass smoke (one batch through SeqVaeLagAttnV1) ------------------

@pytest.mark.parametrize("benchmark", ["G1", "G2", "G3"])
def test_forward_pass_smoke_v2(tmp_path, benchmark):
    """One cached batch flows through ``SeqVaeLagAttnV1.forward`` for each v2 benchmark."""
    config = _tiny_v2_config(benchmark, tmp_path)
    out_dir = build_dataset(config, force=True)

    torch.manual_seed(0)
    ds = SyntheticTEDataset(out_dir / "train.npz")
    loader = make_dataloader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = SeqVaeLagAttnV1()  # defaults match the native shapes (V2-D1)
    model.eval()
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, build_u_stream(batch))

    assert set(out.keys()) == _EXPECTED_FORWARD_KEYS
    assert out["kld_per_t"].shape == (2, _T)
    assert out["mu_full"].shape == (2, _T, 30, 87)
    assert out["te_lag_map"].shape == (2, _T, 91)
    assert out["attn_weights"].shape == (2, _T, 4, 91)
    assert torch.isfinite(out["kld_per_t"]).all()
    assert torch.isfinite(out["mu_full"]).all()


# --- Dataset / collate / build_u_stream / meta round-trip (kept from v1) -----

def test_dataset_shapes_and_dtype(tiny_g1_cache: Path):
    """``SyntheticTEDataset`` exposes per-sample native-shape tensors + metadata."""
    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
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
    assert isinstance(sample.guid, str) and "test_G1_train" in sample.guid
    assert isinstance(sample.te_true, float)
    assert sample.true_lag_band.dtype == torch.long
    # G1 within-signal random-walk lag: the union band is {0, ..., max visited
    # lag - 1}; the realised lags live in meta["delay_histogram"] (the per-sample
    # trajectory is not stored).
    d_max = max(int(k) for k in ds.meta["delay_histogram"])
    assert sample.true_lag_band.tolist() == list(range(0, d_max))


def test_collate_batch(tiny_g1_cache: Path):
    """The :func:`make_dataloader` collate emits a batched :class:`AttributeDict`."""
    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
    loader = make_dataloader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, AttributeDict)
    assert batch.fhr_st.shape == (3, _T, 43)
    assert batch.fhr_ph.shape == (3, _T, 44)
    assert batch.up_st.shape == (3, _T, 43)
    assert batch.up_ph.shape == (3, _T, 58)
    assert batch.weight.shape == (3, _T)
    assert batch.true_lag_band.shape == (3, len(ds.true_lag_band))
    assert len(batch.guid) == 3 and all(isinstance(g, str) for g in batch.guid)


def test_build_u_stream(tiny_g1_cache: Path):
    """``build_u_stream`` concatenates ``up_st`` + ``up_ph`` to the 101-ch input."""
    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
    loader = make_dataloader(ds, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    u_stream = build_u_stream(batch)
    assert u_stream.shape == (3, _T, 101)
    assert torch.equal(u_stream[..., :43], batch.up_st)
    assert torch.equal(u_stream[..., 43:], batch.up_ph)


def test_meta_json_roundtrip(tiny_g1_cache: Path):
    """The cached ``meta.json`` carries the analytic TE, splits and channel map."""
    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
    meta = ds.meta
    assert meta["benchmark"] == "G1"
    assert meta["tag"] == "test_G1"
    assert isinstance(meta["te_true"], float) and meta["te_true"] >= 0.0
    assert meta["split_sizes"] == {
        "train": _N_TRAIN, "val": _N_VAL, "test": _N_TEST
    }
    assert set(meta["split_seeds"]) == {"train", "val", "test"}
    assert meta["channel_map"]["fhr_st"] == [0, 43]
    assert meta["channel_map"]["up_ph"] == [43, 101]
    # G1-specific fields the generator emits. _build_gen_kwargs auto-tiles a
    # length-1 oscillator/delay/B_y spec to M copies (here M=16 default), so the
    # meta records one delay per informative channel.
    assert meta["target_ar"] == pytest.approx(0.95)
    assert meta["M"] == 16
    # G1 within-signal random-walk lag (real-data regime): the lag drifts as a
    # bounded reflecting walk in {delay_min..delay_max}; the per-sample trajectory
    # is summarised by delay_histogram, and the union band is {0, ..., max-1}.
    assert meta["variable_delay"] is True
    assert meta["delay_walk"] is True
    assert meta["delay_min"] == 1 and meta["delay_max"] == 15
    assert meta["delays_per_sample"] is None
    hist = {int(k): int(v) for k, v in meta["delay_histogram"].items()}
    assert min(hist) >= 1 and max(hist) <= 15
    assert meta["true_lag_band"] == list(range(0, max(hist)))


# --- Channel decomposition v2 (structured distractors) -----------------------


@pytest.mark.parametrize("benchmark", ["G1", "G2", "G3"])
def test_meta_records_channel_decomp(tmp_path, benchmark):
    r"""``meta.json`` carries the resolved ``channel_decomp`` + ``channel_layout``."""
    config = _tiny_v2_config(benchmark, tmp_path)
    out_dir = build_dataset(config, force=True)
    ds = SyntheticTEDataset(out_dir / "train.npz")
    assert "channel_decomp" in ds.meta
    assert "channel_layout" in ds.meta
    decomp = ds.meta["channel_decomp"]
    layout = ds.meta["channel_layout"]
    assert decomp["m"] == ds.meta["M"]
    # Target budget closes against c_y; source budget against c_u.
    c_y, c_u = ds.meta["c_y"], ds.meta["c_u"]
    assert decomp["m"] + decomp["n_self"] + decomp["n_smallnoise"] == c_y
    assert decomp["m_source"] + decomp["n_dist"] + decomp["n_noise"] == c_u
    # Layout reports the absolute index lists.
    assert layout["Y"]["te"][0] == 0
    assert layout["Y"]["smallnoise"][-1] == c_y - 1
    assert layout["U"]["te"][0] == 0
    assert layout["U"]["noise"][-1] == c_u - 1
    # G3 source TE width is M*K_classes; G1/G2 is M.
    if benchmark == "G3":
        assert decomp["m_source"] == ds.meta["M"] * ds.meta["K_classes"]
    else:
        assert decomp["m_source"] == ds.meta["M"]


def test_dataset_attaches_layout_attrs(tiny_g1_cache: Path):
    """``SyntheticTEDataset.channel_decomp`` / ``.channel_layout`` are populated."""
    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
    assert ds.channel_decomp is not None
    assert ds.channel_layout is not None
    assert ds.channel_decomp["m"] == ds.meta["M"]
    # Indices into the model's native (B, T, c_y) / (B, T, c_u) buffers.
    layout = ds.channel_layout
    assert layout["Y"]["te"][0] == 0
    assert max(layout["Y"]["smallnoise"]) == ds.meta["c_y"] - 1
    assert max(layout["U"]["noise"]) == ds.meta["c_u"] - 1


def test_mmap_dataset_matches_eager(tiny_g1_cache: Path):
    """The memmap-backed loader returns samples identical to the eager loader.

    ``np.savez`` caches are uncompressed, so the default ``mmap='auto'`` must
    engage; every tensor field, dtype and metadata scalar must match the
    ``mmap=False`` (eager) reference bit-for-bit.
    """
    ds_eager = SyntheticTEDataset(tiny_g1_cache / "train.npz", mmap=False)
    ds_mmap = SyntheticTEDataset(tiny_g1_cache / "train.npz")  # default auto
    assert ds_eager.mmap_active is False
    assert ds_mmap.mmap_active is True
    assert isinstance(ds_mmap._arrays["fhr_st"], np.memmap)
    assert len(ds_mmap) == len(ds_eager)
    for idx in (0, len(ds_eager) - 1):
        a, b = ds_eager[idx], ds_mmap[idx]
        for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
            assert b[field].dtype == torch.float32
            assert torch.equal(a[field], b[field]), field
        assert a.guid == b.guid
        assert a.te_true == b.te_true
        assert torch.equal(a.true_lag_band, b.true_lag_band)
        if "true_lag_tt" in a:
            assert torch.equal(a.true_lag_tt, b.true_lag_tt)
    # The returned tensors are writable copies, not views into the read-only
    # mapping (an in-place op must not raise / touch the cache).
    sample = ds_mmap[0]
    sample.fhr_st += 1.0


def test_mmap_falls_back_on_compressed_npz(tiny_g1_cache: Path, tmp_path):
    """A ``savez_compressed`` archive silently falls back to the eager loader."""
    import shutil

    with np.load(tiny_g1_cache / "train.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    np.savez_compressed(tmp_path / "train.npz", **arrays)
    shutil.copy(tiny_g1_cache / "meta.json", tmp_path / "meta.json")

    ds = SyntheticTEDataset(tmp_path / "train.npz")  # auto -> fallback
    assert ds.mmap_active is False
    ref = SyntheticTEDataset(tiny_g1_cache / "train.npz", mmap=False)
    a, b = ref[0], ds[0]
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        assert torch.equal(a[field], b[field]), field


def test_mmap_dataset_pickles_without_data(tiny_g1_cache: Path):
    """Pickling a memmap dataset ships specs, not arrays (spawn-safe workers).

    A spawn-start DataLoader worker receives the dataset via pickle; the
    payload must stay far below the mapped data size (otherwise every worker
    would materialise a private copy -- the exact regression the mapping
    exists to prevent), and the unpickled dataset must serve identical
    samples through the re-opened mappings.
    """
    import pickle

    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
    assert ds.mmap_active is True
    payload = pickle.dumps(ds)
    mapped_bytes = sum(ds._arrays[f].nbytes for f in ds._arrays)
    assert len(payload) < mapped_bytes / 4, (
        f"pickle payload {len(payload)} B suspiciously large vs "
        f"{mapped_bytes} B mapped -- did the memmaps get serialised?"
    )
    ds2 = pickle.loads(payload)
    assert ds2.mmap_active is True
    a, b = ds[1], ds2[1]
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight"):
        assert torch.equal(a[field], b[field]), field
    assert a.guid == b.guid


def test_smallnoise_tail_variance_in_cached_split(tiny_g1_cache: Path):
    r"""The cached Y small-noise tail keeps $\sigma^2_{\text{smallnoise}}$ on disk."""
    ds = SyntheticTEDataset(tiny_g1_cache / "train.npz")
    layout = ds.channel_layout
    decomp = ds.channel_decomp
    sigma = float(decomp["sigma_smallnoise"])
    # Build the full Y buffer from the cached fhr_st / fhr_ph fields.
    fhr_st = ds._arrays["fhr_st"]                  # (n, T, 43)
    fhr_ph = ds._arrays["fhr_ph"]                  # (n, T, 44)
    Y = torch.from_numpy(np.concatenate([fhr_st, fhr_ph], axis=-1))
    sn_idx = layout["Y"]["smallnoise"]
    Ysn = Y[..., sn_idx]
    var = Ysn.var(dim=(0, 1))
    # Tiny tmp cache (n=6); use loose 60% relative slack.
    assert ((var - sigma ** 2).abs() / (sigma ** 2)).max().item() < 0.6, (
        f"smallnoise tail variance deviates from sigma^2={sigma**2:.6f}; "
        f"per-channel var = {var.tolist()}"
    )
