r"""S7-T03: end-to-end v3 checkpoint round-trip + additive-contract regression.

Trains a tiny v3 model a few steps, saves a checkpoint carrying the version-agnostic
``model_class``/``model_kwargs`` contract, reloads it through ``TestRunner.from_checkpoint``
with the pipeline's model-class alias flipped to v3, and then drives the analysis suite.

Three things are pinned:

* **Checkpoint round-trip.** ``from_checkpoint`` rebuilds v3 from its stamped kwargs, the
  ``check_model_class`` guard accepts it, and ``load_checkpoint_strict`` aligns key-for-key.
* **Additive forward contract (G9).** The reloaded model still emits every v1 forward key with
  its original shape, plus the two additive v3 keys (``kld_active_frac``, ``raw_logvar_prior``).
* **The suite still runs on v3.** ``run_all_analyses`` completes; the new ``calibration`` (G10)
  and ``cmi_comparison`` (G11) steps produce artefacts; and no contract-sensitive analysis
  fails. Every analysis's pass/error status is enumerated so a regression is never silent.

The full HDF5-backed subgroup pipeline (``run_full_test_pipeline`` over real data) is the
manual complement to this scripted, tiny-data gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import model.vae_teb_prediction.testing.base as base_module
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from model.vae_teb_prediction.testing.analyses import run_all_analyses
from model.vae_teb_prediction.testing.base import TestRunner

# Tiny production-shaped v3 (head-structured + frozen W_o + all v3 flags).
_KWARGS = dict(
    sequence_length=40, d_model=32, d_z=8, horizon=6, warmup_period=4,
    c_y=87, c_u=101, use_up_st=True, max_lag=8, num_heads=4, d_head=8, dropout=0.0,
    causal_norm=True, posterior_logvar="residual", logvar_bound="smooth",
    kld_support="anchor", lag_bias_init="alibi_decay",
    head_structured_latent=True, freeze_unused_attn_proj=True,
)
_WARMUP, _HORIZON = 4, 6

# The v1 forward keys the collectors depend on, plus the two additive v3 keys and the decoder
# observation logvar the calibration report reads.
_CONTRACT_KEYS = {
    "mu_prior": 3, "logvar_prior": 3, "mu_post": 3, "logvar_post": 3, "z": 3,
    "target_state": 3, "source_state": 3, "decoder_state": 3, "attended_source": 3,
    "attn_weights": 4, "mu_base": 4, "mu_full": 4, "delta_mu_src": 4,
    "kld_per_t": 2, "te_lag_map": 3, "warmup_mask": 1,
    # additive v3 keys + learned-variance heads
    "kld_active_frac": 0, "raw_logvar_prior": 3, "logvar_full": 4, "logvar_base": 4,
}
_MUST_PASS = ("histogram", "forecast_quality", "uplift", "residual_usage",
              "calibration", "cmi_comparison")


class _Batch:
    """Batch with the fields the runner + analyses read (guid/target/epoch for labels)."""

    def __init__(self, n: int = 6, T: int = 40, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.fhr_st = torch.randn(n, T, 43, generator=g)
        self.fhr_ph = torch.randn(n, T, 44, generator=g)
        self.up_st = torch.randn(n, T, 43, generator=g)
        self.up_ph = torch.randn(n, T, 58, generator=g)
        self.weight = torch.ones(n, T)
        # Two outcome classes so label-conditioned analyses have >= 2 groups.
        classes = [1 + (seed * n + i) % 2 for i in range(n)]
        self.target = torch.stack([float(c) * torch.ones(T) for c in classes])
        self.epoch = torch.tensor([-3600.0 - 600.0 * i for i in range(n)])
        self.guid = [f"e{seed}_{i}" for i in range(n)]


def _make_checkpoint(path: Path) -> None:
    """Train a few steps on a source-coupled batch and save the version-agnostic blob."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV3(**_KWARGS).train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    batch = _Batch(seed=0)
    u_stream = torch.cat([batch.up_st, batch.up_ph], dim=-1)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        outputs = model(y_st=batch.fhr_st, y_ph=batch.fhr_ph, u_stream=u_stream)
        loss = model.compute_loss(
            forward_outputs=outputs, y_st=batch.fhr_st, y_ph=batch.fhr_ph,
            beta=1.0, likelihood="gaussian_nll", sigma_obs="learned",
        )["total_loss"]
        loss.backward()
        opt.step()

    blob = {
        "model_class": "SeqVaeLagAttnV3",
        "model_kwargs": dict(_KWARGS),
        "state_dict": model.state_dict(),
    }
    torch.save(blob, str(path))


@pytest.fixture(autouse=True)
def _v3_alias(monkeypatch):
    monkeypatch.setattr(base_module, "SeqVaeLagAttn", SeqVaeLagAttnV3)


@pytest.fixture
def runner(tmp_path) -> TestRunner:
    ckpt = tmp_path / "lag-attn-v3-epoch=000.ckpt"
    _make_checkpoint(ckpt)
    runner = TestRunner.from_checkpoint(
        checkpoint_path=ckpt, output_dir=tmp_path / "results",
        config_path=None, device=torch.device("cpu"),
    )
    return runner


def test_checkpoint_round_trips_through_from_checkpoint(runner):
    assert isinstance(runner.model, SeqVaeLagAttnV3)
    assert runner.warmup_steps == _WARMUP and runner.horizon == _HORIZON
    # The frozen attention projection survived the reload (head-structured production path).
    assert runner.model.frozen_attn_proj is True


def test_reloaded_model_preserves_the_additive_forward_contract(runner):
    batch = _Batch(seed=1)
    with runner.inference_mode():
        outputs = runner.forward(batch)
    for key, ndim in _CONTRACT_KEYS.items():
        assert key in outputs, f"forward contract broke: '{key}' missing"
        if ndim > 0:
            assert outputs[key].dim() == ndim, (
                f"'{key}' rank {outputs[key].dim()} != expected {ndim}"
            )


def test_full_analysis_suite_runs_and_new_steps_emit_artifacts(runner, tmp_path):
    loader = [_Batch(seed=0), _Batch(seed=1)]
    results = run_all_analyses(
        runner, loader, max_samples=12,
        skip_trajectory=True, skip_forecast_heatmaps=True, skip_per_class_breakdown=True,
    )

    # Enumerate every analysis's status so a regression is never silent.
    status = {
        name: ("error: " + res["error"]) if isinstance(res, dict) and "error" in res else "ok"
        for name, res in results.items()
    }
    for name, state in sorted(status.items()):
        print(f"[e2e] {name}: {state}")

    # The additive-contract gate: the contract-sensitive core analyses and both new steps must
    # succeed on a valid v3 forward dict.
    for name in _MUST_PASS:
        assert name in results, f"{name} did not run"
        assert not (isinstance(results[name], dict) and "error" in results[name]), (
            f"{name} failed on v3: {results[name].get('error')}"
        )

    # New steps write their artefacts.
    calib_dir = runner.output_dir / "calibration"
    cmi_dir = runner.output_dir / "cmi_comparison"
    assert (calib_dir / "summary.json").is_file()
    assert (cmi_dir / "comparison_table.csv").is_file()
    assert (cmi_dir / "summary.json").is_file()
    cmi_summary = json.loads((cmi_dir / "summary.json").read_text(encoding="utf-8"))
    assert "spearman_kraw_infonce" in cmi_summary and "spearman_kraw_mine" in cmi_summary
