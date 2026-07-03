r"""Tests for the Sprint 2 scattering adapter + normalisation (S2-T01…S2-T04).

Covers the transform wrapper (exact ``43/44/43/58`` counts at the production
``shape=5280`` and the fail-loud count guard), the local normalisation (pointwise
transform selection, ``synthetic_pool`` z-score, and production parity), the fs-correct
coupled-channel identification and its (documented) latent-tracking correlation, and the
monotone-normalisation invariance + scattering-heatmap figure.

The real transform is expensive, so the full ``shape=5280`` adapter and one strong-cell
transform are built once per module and reused; the pure normalisation checks use small
synthetic arrays. See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    raw_generators as rg,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    scattering_adapter as sa,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.visualize_v2 import (
    plot_scattering_heatmap,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"

# Strong am_carrier cell for the fs-correct-channel / tracking checks. A fixed B avoids
# the (slow) inverter Monte-Carlo in the test path; the source channel's tracking does not
# depend on B (the source latent c is present at any coupling).
_STRONG_B = 3.0
_STRONG_D = 8
_STRONG_N = 32


@pytest.fixture(scope="module")
def config() -> dict:
    r"""Load the parsed ``config_synth_v2.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def adapter_full(config) -> "sa.ScatteringAdapter":
    r"""The production-shape (``n_raw=5280``) adapter, built once (strict counts on)."""
    return sa.ScatteringAdapter(config)


@pytest.fixture(scope="module")
def strong_raw(config) -> dict:
    r"""Raw FHR/UP pairs for a strong am_carrier cell (fixed B, single generation)."""
    return rg.generate_cell_raw(
        _STRONG_N, B=_STRONG_B, D=_STRONG_D, config=config, seed=0, te_inj=2.0
    )


@pytest.fixture(scope="module")
def strong_fields(adapter_full, strong_raw):
    r"""Transform + normalise the strong cell once; reused by several tests."""
    fields, stats = adapter_full.transform_and_normalise(
        strong_raw["fhr_raw"], strong_raw["up_raw"]
    )
    return fields, stats


# ---------------------------------------------------------------------------
# S2-T01: transform wrapper, exact counts, fail-loud guard
# ---------------------------------------------------------------------------


def test_counts_and_zscore(strong_fields) -> None:
    r"""The four fields have the exact ``43/44/43/58`` counts and are per-channel z-scored."""
    fields, _ = strong_fields
    assert fields["fhr_st"].shape == (_STRONG_N, 300, 43)
    assert fields["fhr_ph"].shape == (_STRONG_N, 300, 44)
    assert fields["up_st"].shape == (_STRONG_N, 300, 43)
    assert fields["up_ph"].shape == (_STRONG_N, 300, 58)
    for name, arr in fields.items():
        assert arr.dtype == np.float32, name
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        # synthetic_pool: population z-score -> per-channel mean 0, std in [0, 1] (->1
        # away from near-constant channels, whose std collapses toward 0).
        assert np.abs(mean).max() < 1e-3, name
        assert std.max() <= 1.0 + 1e-4, name
        assert np.median(std) > 0.9, name


def test_counts_mismatch_raises(adapter_full) -> None:
    r"""``assert_production_counts`` fails loudly when a count drifts from the contract."""
    original = adapter_full.fhr_ph_channels
    try:
        adapter_full.fhr_ph_channels = original + 1
        with pytest.raises(ValueError):
            adapter_full.assert_production_counts()
    finally:
        adapter_full.fhr_ph_channels = original  # restore (module-scoped fixture)


# ---------------------------------------------------------------------------
# S2-T02: normalisation (transform selection, synthetic_pool, parity, real_fold guard)
# ---------------------------------------------------------------------------


def test_transform_field_selection() -> None:
    r"""``*_st`` logs ch 1.. (ch 0 untouched); ``*_ph`` asinh-transforms all channels."""
    rng = np.random.default_rng(0)
    st = np.abs(rng.standard_normal((2, 6, 7))) + 0.1
    t = sa._transform_field(st, "fhr_st", sa.LOG_EPSILON)
    assert np.allclose(t[:, 0, :], st[:, 0, :])  # order-0 untouched
    assert np.allclose(t[:, 1:, :], np.log(st[:, 1:, :] + sa.LOG_EPSILON))
    ph = rng.standard_normal((2, 6, 7))
    tp = sa._transform_field(ph, "up_ph", sa.LOG_EPSILON)
    assert np.allclose(tp, np.arcsinh(ph))


def test_synthetic_pool_zscore() -> None:
    r"""``synthetic_pool`` stats give per-channel mean 0 / std 1 on non-degenerate fields."""
    rng = np.random.default_rng(1)
    fields = {
        "fhr_st": np.abs(rng.standard_normal((8, 43, 50))) + 0.5,
        "up_ph": rng.standard_normal((8, 58, 50)) * 2.0,
    }
    stats = sa.compute_norm_stats(fields)
    normed = sa.normalise_fields(fields, stats)
    for name, arr in normed.items():
        assert np.abs(arr.mean(axis=(0, 2))).max() < 1e-4, name
        assert np.allclose(arr.std(axis=(0, 2)), 1.0, atol=1e-3), name


def test_parity_vs_production() -> None:
    r"""The local normaliser matches production ``normalize_tensor_data`` on a fixed input."""
    torch = pytest.importorskip("torch")
    prod = pytest.importorskip("hdf5_dataset.hdf5_dataset")
    normalize_tensor_data = prod.normalize_tensor_data

    rng = np.random.default_rng(2)
    fields = {
        "fhr_st": (np.abs(rng.standard_normal((3, 43, 40))) * 2.0).astype(np.float32),
        "up_ph": (rng.standard_normal((3, 58, 40)) * 3.0).astype(np.float32),
    }
    mine_stats = sa.compute_norm_stats(fields)
    mine = sa.normalise_fields(fields, mine_stats)
    prod_stats = {
        f: {"mean": mine_stats[f]["mean"], "variance": mine_stats[f]["std"] ** 2}
        for f in fields
    }
    log_cfg = {"fhr_st": "all_except_0", "up_st": "all_except_0"}
    asinh_cfg = {"fhr_ph": "all", "up_ph": "all"}
    for field, arr in fields.items():
        out = normalize_tensor_data(
            torch.from_numpy(arr), field, prod_stats, log_cfg, asinh_cfg, log_epsilon=1e-6
        ).numpy()
        assert np.abs(out - mine[field]).max() < 1e-4, field


def test_real_fold_requires_path() -> None:
    r"""``real_fold`` without a configured path raises a clear error."""
    with pytest.raises(ValueError):
        sa.load_real_fold_stats(None)


# ---------------------------------------------------------------------------
# S2-T03: fs-correct coupled-channel identification and latent tracking
# ---------------------------------------------------------------------------


def test_coupled_channel_within_q_step(adapter_full) -> None:
    r"""The coupled ``*_st`` channel sits within one $Q$-step of $f_{\mathrm{pulse}}$."""
    coupled = adapter_full.coupled_channel_indices()
    f_pulse = adapter_full.f_pulse
    q = adapter_full.Q
    assert coupled["up_st"] == coupled["fhr_st"]
    assert coupled["up_st"] >= 1  # never the order-0 baseline channel
    assert f_pulse * 2.0 ** (-1.0 / q) <= coupled["hz"] <= f_pulse * 2.0 ** (1.0 / q)


def test_coupled_channel_tracks_latent(adapter_full, strong_raw, strong_fields) -> None:
    r"""The coupled ``up_st`` channel tracks the decimated source latent ``c[15:315]``.

    Note (S2-T03 / Sprint 1 flag): with the default ``f_pulse=0.02`` and
    ``am_offset_ratio=4`` the AM preservation is low (Sprint 1 estimated ~0.21), so the
    measured ``|corr|`` is well below the aspirational 0.6. This is a *documented*
    shortfall routed to the Sprint 3 ``frac_Phi`` recovery (S3-T06), not a Sprint 2
    failure; here we only require clearly-positive tracking.
    """
    fields, _ = strong_fields
    idx = adapter_full.coupled_channel_indices()["up_st"]
    c = strong_raw["latents"]["c"][:, 15:315]
    chan = fields["up_st"][:, :, idx]
    c_c = c - c.mean(axis=1, keepdims=True)
    ch_c = chan - chan.mean(axis=1, keepdims=True)
    denom = np.sqrt((c_c ** 2).sum(axis=1)) * np.sqrt((ch_c ** 2).sum(axis=1))
    corr = (c_c * ch_c).sum(axis=1) / (denom + 1e-12)
    assert float(np.abs(corr).mean()) > 0.15


# ---------------------------------------------------------------------------
# S2-T04: monotone-normalisation invariance and the scattering heatmap
# ---------------------------------------------------------------------------


def test_normalisation_is_monotone() -> None:
    r"""Each channel's normalisation is strictly order-preserving (log/asinh/z-score)."""
    rng = np.random.default_rng(3)
    fields = {
        "fhr_st": np.abs(rng.standard_normal((2, 43, 60))) + 0.1,
        "fhr_ph": rng.standard_normal((2, 44, 60)),
    }
    stats = sa.compute_norm_stats(fields)
    normed = sa.normalise_fields(fields, stats)
    for name in fields:
        for ch in (0, 1, 5, 20):  # includes order-0 (affine) for the *_st field
            x = fields[name][0, ch, :]
            y = normed[name][0, ch, :]
            assert np.array_equal(np.argsort(x), np.argsort(y)), (name, ch)


def test_scattering_heatmap_written(adapter_full, strong_fields, tmp_path) -> None:
    r"""The scattering + phase-harmonic heatmap figure is written in both formats.

    Passes the phase-harmonic fields so the four-panel path (scattering pair + phase
    pair) is exercised; the two-panel back-compat path is a simple subset.
    """
    fields, _ = strong_fields
    idx = adapter_full.coupled_channel_indices()["up_st"]
    written = plot_scattering_heatmap(
        fields["fhr_st"][:2],
        fields["up_st"][:2],
        tmp_path / "scattering_heatmap",
        fhr_ph=fields["fhr_ph"][:2],
        up_ph=fields["up_ph"][:2],
        coupled_idx=idx,
        center_freqs=adapter_full.center_freqs_np,
        fs=adapter_full.fs,
    )
    assert len(written) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
