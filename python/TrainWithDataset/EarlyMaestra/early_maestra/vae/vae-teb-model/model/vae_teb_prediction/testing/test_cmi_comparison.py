r"""S6-T02/T03: the CMI comparison analysis, driven through a real ``TestRunner``.

These pin that the analysis turns the per-sample :math:`(u, y, c, K_{\mathrm{raw}})` features
into a comparison table with rank correlations and patient-level bootstrap CIs, that it
degrades gracefully on a model that emits no encoder states, and that the optional empirical-TE
CSV is joined at the patient level when supplied.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import model.vae_teb_prediction.testing.base as base_module
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from model.vae_teb_prediction.testing.analyses.cmi_comparison import run_cmi_comparison
from model.vae_teb_prediction.testing.base import TestRunner

_KWARGS = dict(
    sequence_length=40, d_model=32, d_z=8, horizon=6, warmup_period=4,
    c_y=87, c_u=101, use_up_st=True, max_lag=8, num_heads=4, d_head=8, dropout=0.0,
    causal_norm=True, posterior_logvar="residual", logvar_bound="smooth",
    kld_support="anchor",
)
_WARMUP, _HORIZON = 4, 6
# Small critic + few folds/iters: the analysis wiring is under test, not estimator quality.
_FIT = dict(bounds=("infonce", "mine"), n_folds=2, n_iters=40, hidden=16, embed=8, n_boot=200)


class _Batch:
    """The batch fields the runner reads, plus per-sample ``guid`` for patient grouping."""

    def __init__(self, n: int = 4, T: int = 40, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.fhr_st = torch.randn(n, T, 43, generator=g)
        self.fhr_ph = torch.randn(n, T, 44, generator=g)
        self.up_st = torch.randn(n, T, 43, generator=g)
        self.up_ph = torch.randn(n, T, 58, generator=g)
        self.weight = torch.ones(n, T)
        self.guid = [f"s{seed}_{i}" for i in range(n)]


@pytest.fixture(autouse=True)
def _v3_alias(monkeypatch):
    monkeypatch.setattr(base_module, "SeqVaeLagAttn", SeqVaeLagAttnV3)


@pytest.fixture
def runner(tmp_path) -> TestRunner:
    torch.manual_seed(0)
    model = SeqVaeLagAttnV3(**_KWARGS).eval()
    return TestRunner(
        model=model, device=torch.device("cpu"), output_dir=tmp_path,
        warmup_steps=_WARMUP, horizon=_HORIZON, max_lag=8, use_up_st=True,
    )


@pytest.fixture
def loader():
    return [_Batch(seed=0), _Batch(seed=1)]


def test_analysis_writes_the_comparison_table_and_rank_correlations(runner, loader, tmp_path):
    summary = run_cmi_comparison(runner, loader, max_samples=8, **_FIT)

    out = tmp_path / "cmi_comparison"
    for name in ("per_sample.csv", "comparison_table.csv", "summary.json"):
        assert (out / name).is_file(), f"{name} was not written"
    assert (out / "cmi_comparison.pdf").stat().st_size > 0

    assert summary["n_samples"] == 8
    for key in ("spearman_kraw_infonce", "spearman_kraw_mine", "spearman_infonce_mine"):
        assert key in summary and -1.0 <= summary[key] <= 1.0

    per_sample = pd.read_csv(out / "per_sample.csv")
    for col in ("guid", "k_raw", "cmi_infonce", "cmi_mine"):
        assert col in per_sample.columns
    assert len(per_sample) == 8

    table = pd.read_csv(out / "comparison_table.csv")
    for col in ("quantity", "mean", "ci_low", "ci_high", "spearman_vs_kraw", "n"):
        assert col in table.columns
    assert set(table["quantity"]) >= {"k_raw", "cmi_infonce", "cmi_mine"}
    # Patient-level bootstrap CIs are computable (>= 2 distinct guids) and finite here.
    kraw_row = table[table["quantity"] == "k_raw"].iloc[0]
    assert kraw_row["ci_low"] <= kraw_row["mean"] <= kraw_row["ci_high"]

    on_disk = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["n_samples"] == 8
    assert "capacity" in on_disk


def test_analysis_joins_empirical_te_when_supplied(runner, loader, tmp_path):
    te_csv = tmp_path / "empirical_te.csv"
    rows = [{"guid": f"s{s}_{i}", "ite_valid": 0.1 * (i + 1) + s}
            for s in (0, 1) for i in range(4)]
    pd.DataFrame(rows).to_csv(te_csv, index=False)

    summary = run_cmi_comparison(runner, loader, max_samples=8, empirical_te_csv=te_csv, **_FIT)

    assert summary["has_empirical_te"] is True
    assert "spearman_kraw_empirical" in summary
    per_sample = pd.read_csv(tmp_path / "cmi_comparison" / "per_sample.csv")
    assert "ite_valid" in per_sample.columns
    assert per_sample["ite_valid"].notna().all()


def test_analysis_skips_when_te_csv_is_absent(runner, loader):
    summary = run_cmi_comparison(runner, loader, max_samples=8,
                                 empirical_te_csv="does_not_exist.csv", **_FIT)
    assert summary["has_empirical_te"] is False


def test_analysis_skips_cleanly_when_the_model_emits_no_states(tmp_path, loader):
    class _NoStates(SeqVaeLagAttnV3):
        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            outputs.pop("target_state")
            return outputs

    torch.manual_seed(0)
    runner = TestRunner(
        model=_NoStates(**_KWARGS).eval(), device=torch.device("cpu"), output_dir=tmp_path,
        warmup_steps=_WARMUP, horizon=_HORIZON, max_lag=8, use_up_st=True,
    )
    result = run_cmi_comparison(runner, loader, max_samples=8, **_FIT)
    assert "error" in result and "target_state" in result["error"]
    assert not (Path(tmp_path) / "cmi_comparison" / "per_sample.csv").exists()


def test_analysis_honours_the_skip_switch(runner, loader):
    assert run_cmi_comparison(runner, loader, max_samples=0) == {}
