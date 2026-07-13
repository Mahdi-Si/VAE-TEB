r"""S6-T04: raw-adapted collectors + the forward-dict contract test.

Runs the tiny fixture model through the adapted collectors on a small in-memory loader and asserts:
(a) the raw forward dict keeps the v3 attention / TE / KL shapes and a 4-D $(B, T, H, R)$ ``mu_full``;
(b) ``collect_metrics`` emits raw forecast columns and **no** scattering ``feat_mse_st`` / ``feat_mse_ph``;
(c) ``collect_forecast_errors_per_horizon`` emits ``[guid, epoch, label, h, mse_step]`` with no st/ph;
(d) the domain-agnostic latent / KL / attention / TE-lag collectors run on the raw dict unchanged.

Tiny geometry: $T=28, H=4, R=16$, ``max_lag=8`` -> $9$ lag taps, ``num_heads=4``.
"""
from __future__ import annotations

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import (
    collect_attention_maps,
    collect_forecast_errors_per_horizon,
    collect_latents,
    collect_metrics,
    collect_te_lag_maps,
)
from model.vae_teb_prediction.model.model_raw.testing.conftest import make_raw_stub_batch
from model.vae_teb_prediction.model.model_raw.testing.metrics import (
    aggregate_te_lag_map,
    compute_attention_diagnostics,
)

_B, _T, _H, _R = 4, 28, 4, 16
_HEADS, _LAGS = 4, 9  # max_lag=8 -> 9 taps


def _runner(tiny_checkpoint, tmp_path) -> TestRunner:
    ckpt_path, _ = tiny_checkpoint
    return TestRunner.from_checkpoint(ckpt_path, tmp_path / "out")


def _loader():
    return [make_raw_stub_batch(batch_size=_B, raw_len=512, seed=s) for s in range(2)]


def test_forward_dict_shapes_contract(tiny_checkpoint, tmp_path) -> None:
    """The raw forward dict keeps the v3 attention/TE/KL shapes and a 4-D mu_full."""
    runner = _runner(tiny_checkpoint, tmp_path)
    batch = make_raw_stub_batch(batch_size=_B, raw_len=512)
    with runner.inference_mode():
        b = next(iter(runner.iter_batches([batch])))
        out = runner.forward(b)

    assert tuple(out["attn_weights"].shape) == (_B, _T, _HEADS, _LAGS)
    assert tuple(out["te_lag_map"].shape) == (_B, _T, _LAGS)
    assert tuple(out["kld_per_t"].shape) == (_B, _T)
    # mu_full unpacks as (B, T, H, R).
    assert out["mu_full"].dim() == 4
    b_, t_, h_, r_ = out["mu_full"].shape
    assert (b_, t_, h_, r_) == (_B, _T, _H, _R)

    # Agnostic metric functions consume the raw dict unchanged.
    agg = aggregate_te_lag_map(out["te_lag_map"], runner.warmup_steps)
    assert tuple(agg["te_lag_mean"].shape) == (_B, _LAGS)
    assert tuple(agg["te_lag_argmax"].shape) == (_B,)
    diag = compute_attention_diagnostics(out["attn_weights"], runner.warmup_steps)
    assert tuple(diag["entropy"].shape) == (_B, _T, _HEADS)


def test_collect_metrics_has_raw_columns_no_channel_split(tiny_checkpoint, tmp_path) -> None:
    """``collect_metrics`` returns raw forecast columns and drops the scattering st/ph split."""
    runner = _runner(tiny_checkpoint, tmp_path)
    df = collect_metrics(runner, _loader(), max_samples=None)
    assert len(df) == 2 * _B
    for col in ("raw_mse", "raw_vaf", "raw_r2", "raw_lowpass_mse", "feat_mse_total", "kld_mean"):
        assert col in df.columns
    assert "feat_mse_st" not in df.columns
    assert "feat_mse_ph" not in df.columns


def test_collect_forecast_errors_per_horizon_no_st_ph(tiny_checkpoint, tmp_path) -> None:
    """Per-horizon error table carries ``mse_step`` only (no ``mse_st`` / ``mse_ph``)."""
    runner = _runner(tiny_checkpoint, tmp_path)
    df = collect_forecast_errors_per_horizon(runner, _loader(), max_samples=None)
    assert set(df.columns) == {"guid", "epoch", "label", "h", "mse_step"}
    # One row per (sample, horizon step).
    assert len(df) == 2 * _B * _H
    assert set(df["h"].unique()) == set(range(_H))


def test_agnostic_collectors_run_on_raw(tiny_checkpoint, tmp_path) -> None:
    """The latent / attention / TE-lag collectors run unchanged on the raw forward dict."""
    runner = _runner(tiny_checkpoint, tmp_path)
    latents = collect_latents(runner, _loader(), max_samples=None)
    # (N * T, d_z) flattened latent trajectory: N = 2 * B samples, T = 28, d_z = 24.
    assert latents.shape == (2 * _B * _T, 24)
    attn = collect_attention_maps(runner, _loader(), max_samples=None)
    assert len(attn) == 2 * _B
    te = collect_te_lag_maps(runner, _loader(), max_samples=None)
    assert len(te) >= 1
