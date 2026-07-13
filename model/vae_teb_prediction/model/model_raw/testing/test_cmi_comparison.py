r"""S7-T03: neural-CMI corroboration (G11) of the raw model's ``kld_raw`` surrogate.

``run_cmi_comparison`` is domain-agnostic -- it consumes ``target_state`` / ``source_state`` / the
future-target summary / ``kld_per_t`` from the forward dict, all of which the raw ``SeqVaeRawV4``
emits unchanged. This test confirms it runs on the tiny fixture, returns a signed-Spearman alignment
between each neural-CMI bound and ``kld_raw``, and degrades gracefully (no empirical column) when no
empirical-TE CSV is supplied.
"""
from __future__ import annotations

from model.vae_teb_prediction.model.model_raw.testing.analyses.cmi_comparison import (
    run_cmi_comparison,
)
from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.conftest import make_tiny_eval_loader


def _runner(tiny_checkpoint, tmp_path) -> TestRunner:
    ckpt_path, _ = tiny_checkpoint
    return TestRunner.from_checkpoint(ckpt_path, tmp_path / "out")


def test_cmi_comparison_runs_and_reports_signed_spearman(tiny_checkpoint, tmp_path) -> None:
    """Returns a signed-Spearman between each neural-CMI bound and ``kld_raw`` (empirical absent)."""
    runner = _runner(tiny_checkpoint, tmp_path)
    loader = make_tiny_eval_loader(n_batches=4, batch_size=4)  # 16 samples >= 2 * n_folds
    res = run_cmi_comparison(
        runner, loader, max_samples=16, output_dir=tmp_path / "cmi",
        bounds=("infonce", "mine"), n_folds=2, n_iters=25, n_boot=50,
    )

    assert "error" not in res
    assert res["n_samples"] == 16
    # A signed-Spearman between each neural-CMI bound and the raw KL surrogate.
    assert "spearman_kraw_infonce" in res
    assert "spearman_kraw_mine" in res
    assert isinstance(res["spearman_kraw_infonce"], float)
    assert -1.0 <= res["spearman_kraw_infonce"] <= 1.0
    # No empirical-TE CSV supplied -> the empirical column is skipped, not fabricated.
    assert "spearman_kraw_empirical" not in res


def test_cmi_comparison_skips_gracefully_on_too_few_samples(tiny_checkpoint, tmp_path) -> None:
    """With fewer samples than folds require, it returns an ``error`` entry instead of raising."""
    runner = _runner(tiny_checkpoint, tmp_path)
    loader = make_tiny_eval_loader(n_batches=1, batch_size=2)  # 2 samples < 2 * 3 folds
    res = run_cmi_comparison(
        runner, loader, max_samples=2, output_dir=tmp_path / "cmi2",
        bounds=("infonce",), n_folds=3, n_iters=10, n_boot=10,
    )
    assert "error" in res
