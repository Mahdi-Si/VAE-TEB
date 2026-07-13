r"""S7-T02: end-to-end ``run_tests`` (quick) on the tiny fixture via an in-memory loader.

Proves the raw eval pipeline produces the expected output tree -- raw forecast metrics + plots, the
G10 calibration report, and the domain-agnostic latent / KL / attention / TE folders -- with **no**
scattering ``frequency_band_forecast`` step, and that it runs with no HDF5 dependency
(``loader_override`` + the tiny checkpoint). Content is meaningless on the untrained fixture; this
locks the *tree* and the *no-scattering-step* contract (Sprint 8 adds trained known-answer checks).
"""
from __future__ import annotations

from pathlib import Path

from model.vae_teb_prediction.model.model_raw.testing.conftest import make_tiny_eval_loader
from model.vae_teb_prediction.model.model_raw.testing.run_tests import run_full_test_pipeline

_CONFIG = Path(__file__).with_name("config_raw_v4_testing.yaml")


def test_pipeline_produces_raw_output_tree_no_band_step(tiny_checkpoint, tmp_path) -> None:
    ckpt_path, _ = tiny_checkpoint
    out = tmp_path / "eval_out"
    loader = make_tiny_eval_loader(n_batches=3, batch_size=4)

    results = run_full_test_pipeline(
        checkpoint_path=str(ckpt_path),
        data_path=None,
        output_dir=str(out),
        config_path=str(_CONFIG),
        loader_override=loader,
        max_samples=12,
        skip_trajectory=True,        # no per-GUID loader in-memory
        skip_forecast_heatmaps=True, # per-sample diagnostics covered by test_visualizers
        skip_interactive=True,
        analysis_samples=0,
    )

    # The raw forecast step and the calibration step ran and wrote their directories.
    assert (out / "raw_forecast").is_dir()
    assert (out / "raw_forecast" / "raw_metrics.csv").exists()
    assert (out / "calibration" / "per_sample.csv").exists()

    # No scattering frequency-band step anywhere in the tree or the results dict.
    assert not (out / "frequency_band_forecast").exists()
    assert "frequency_band_forecast" not in results
    assert "raw_forecast" in results

    # A couple of the domain-agnostic steps are present in the results dict (they may carry an
    # ``error`` entry on the untrained tiny fixture -- we only assert they were attempted).
    for step in ("histogram", "calibration", "cmi_comparison", "te_lag", "attention"):
        assert step in results
