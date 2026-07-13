r"""S4-T06 tests: the shared eval runner builds a live TestRunner + synthetic loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.v4

_FORWARD_KEYS = {
    "mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post", "z",
    "target_state", "source_state", "decoder_state", "attended_source", "attended_source_heads",
    "attn_weights", "mu_base", "logvar_base", "delta_mu_src", "mu_full", "logvar_full",
    "raw_future_pred", "kld_per_t", "kld_per_t_per_head", "te_lag_map", "warmup_mask",
    "mu_prior_sat_frac", "delta_mu_sat_frac", "kld_active_frac",
}


def _small_prod_model_kwargs() -> Dict[str, Any]:
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        SMALL_PROD_FRONTEND,
        SMALL_PROD_V3_KWARGS,
    )

    return dict(frontend=dict(SMALL_PROD_FRONTEND), raw_len=5280, decimation=16,
                **SMALL_PROD_V3_KWARGS)


def _save_small_prod_checkpoint(path: Path) -> None:
    r"""Build a small-prod `SeqVaeRawV4`, stamp provenance via the Pl subclass, and save it."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import SeqVaeRawV4
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        SyntheticSeqVaeRawV4Pl,
    )

    kwargs = _small_prod_model_kwargs()
    model = SeqVaeRawV4(**kwargs)
    pl_module = SyntheticSeqVaeRawV4Pl(
        model, arm="prod", render_mode="direct", lr=1e-3, model_kwargs=kwargs,
    )
    checkpoint: Dict[str, Any] = {"state_dict": pl_module.state_dict()}
    pl_module.on_save_checkpoint(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))


def test_eval_runner_forward_returns_25_keys(tiny_cache_v4, tmp_path):
    r"""`runner.forward` on a synthetic batch returns the full 25-key dict."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        _build_runner_and_loader_v4,
    )

    ckpt = tmp_path / "prod" / "final.ckpt"
    _save_small_prod_checkpoint(ckpt)

    runner, loader = _build_runner_and_loader_v4(
        ckpt, tiny_cache_v4["config"], benchmark="G1_raw_v4",
        cache_dir=tiny_cache_v4["cache_dir"], batch_size=2, split="val",
        device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    out = runner.forward(batch)
    assert set(out.keys()) == _FORWARD_KEYS
    assert out["mu_full"].shape == (2, 300, 30, 16)
    assert out["te_lag_map"].shape == (2, 300, 91)


def test_eval_runner_model_class_mismatch_raises(tiny_cache_v4, tmp_path):
    r"""A checkpoint with an unknown `model_class` raises before evaluation."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        _build_runner_and_loader_v4,
    )

    bad = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {}, "model_class": "SomethingElse", "model_kwargs": {}}, str(bad))
    with pytest.raises(ValueError):
        _build_runner_and_loader_v4(
            bad, tiny_cache_v4["config"], cache_dir=tiny_cache_v4["cache_dir"],
            device=torch.device("cpu"),
        )


def _runner_and_loader(tiny_cache_v4, tmp_path, *, subdir: str):
    r"""Build a small-prod checkpoint + a live runner/loader routed under ``tmp_path/subdir``."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        _build_runner_and_loader_v4,
    )

    ckpt = tmp_path / "prod" / "final.ckpt"
    _save_small_prod_checkpoint(ckpt)
    return _build_runner_and_loader_v4(
        ckpt, tiny_cache_v4["config"], benchmark="G1_raw_v4",
        cache_dir=tiny_cache_v4["cache_dir"], output_dir=tmp_path / subdir,
        batch_size=2, split="val", device=torch.device("cpu"),
    )


def test_raw_forecast_metrics_finite(tiny_cache_v4, tmp_path):
    r"""S5-T01: ``compute_raw_forecast_metrics`` yields finite per-horizon VAF/MSE/SNR/R^2."""
    import numpy as np
    import torch

    from model.vae_teb_prediction.model.model_raw.testing.metrics import (
        compute_raw_forecast_metrics,
    )

    runner, loader = _runner_and_loader(tiny_cache_v4, tmp_path, subdir="metrics")
    batch = next(iter(loader))
    with runner.inference_mode():
        out = runner.forward(batch)
        x_plus = runner.build_future_target(batch)
        m = compute_raw_forecast_metrics(
            out["mu_full"], x_plus, int(runner.warmup_steps), int(runner.horizon),
        )
    for key in ("raw_vaf", "raw_mse", "raw_snr", "raw_r2"):
        assert torch.isfinite(m[key]).all(), f"{key} not finite"
    assert m["raw_mse_per_horizon"].shape == (x_plus.shape[0], int(runner.horizon))
    assert np.isfinite(m["raw_mse_per_horizon"].detach().cpu().numpy()).all()


def test_g10_calibration_returns_report(tiny_cache_v4, tmp_path):
    r"""S5-T01: G10 calibration returns a coverage/NLL/CRPS report (or a logged skip)."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        run_raw_metrics_v4,
    )

    runner, loader = _runner_and_loader(tiny_cache_v4, tmp_path, subdir="metrics2")
    res = run_raw_metrics_v4(runner, loader, max_samples_forecast=4, max_samples_calib=4)
    assert "raw_forecast" in res and "calibration" in res
    assert isinstance(res["calibration"], dict)
    assert (runner.output_dir / "raw_forecast").is_dir()


def test_overlays_written(tiny_cache_v4, tmp_path):
    r"""S5-T02: at least one qualitative overlay artefact is written headlessly."""
    import matplotlib

    matplotlib.use("Agg")

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        run_overlays_v4,
    )

    runner, loader = _runner_and_loader(tiny_cache_v4, tmp_path, subdir="overlays")
    res = run_overlays_v4(runner, loader, max_samples=2)
    assert int(res.get("n_plotted", 0)) >= 1
    samples_dir = runner.output_dir / "samples_diag"
    assert samples_dir.is_dir() and any(samples_dir.iterdir())
    # The raw forecast horizon spans H*R = 30*16 = 480 raw samples (2 min at 4 Hz).
    assert int(runner.horizon) * int(getattr(runner.model, "decimation", 16)) == 480


def test_agnostic_analyses_run(tiny_cache_v4, tmp_path):
    r"""S5-T03: the domain-agnostic latent/KL/attention/TE analyses run without raising."""
    import matplotlib

    matplotlib.use("Agg")

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        run_agnostic_analyses_v4,
    )

    import numpy as np

    runner, loader = _runner_and_loader(tiny_cache_v4, tmp_path, subdir="agnostic")
    caps = {"latent": 4, "kld_lag": 2, "attention": 4, "te_lag": 4}
    res = run_agnostic_analyses_v4(runner, loader, caps=caps)
    assert set(res) == {"latent", "kld_lag", "attention", "te_lag"}
    # Each analysis returns *some* artefact (dict / ndarray / list) rather than aborting the stage;
    # ``run_latent_distribution_analysis`` returns an ndarray of pooled latents.
    assert all(isinstance(v, (dict, list, np.ndarray)) or v is None for v in res.values())
    # Attention diagnostics (the attention-entropy output over attn_weights) must run cleanly.
    assert isinstance(res["attention"], dict) and "error" not in res["attention"]
    # The TE-lag analysis consumes te_lag_map without raising.
    assert isinstance(res["te_lag"], dict) and "error" not in res["te_lag"]
