r"""Tests for the Sprint 7 journal figures and TE-aware plotting (S7-T01…T08).

Every figure test writes to ``tmp_path`` under the headless ``Agg`` backend (selected
by :mod:`visualize_v2` at import) and asserts the PDF + PNG artifacts exist and are
non-empty. Figures are exercised on tiny synthetic arrays / metrics dicts so the suite
stays fast; the heavy real-transform path is covered by the ``slow`` integration test.

Test selection (``-k``):
    * ``raw``              -- annotated raw-signal preview (S7-T01)
    * ``paired``           -- raw + scattering paired preview (S7-T02)
    * ``decomp`` / ``diag``-- latent/AM decomposition + diagnostics panel (S7-T03)
    * ``metadata_bridge`` / ``sample_te`` -- standard-testing TE bridge (S7-T06)
    * ``aggregate`` / ``te_columns`` -- TE-aware aggregate figures (S7-T08)
    * ``report``           -- final report assembly (S7-T05)
    * ``spectra`` / ``authoring`` / ``coupling`` / ``separation`` -- data-generation
      story figures (band recipe §4-5, TE authoring §9, latent coupling §6, AM separation §7)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# The ``testing`` stack (used by the S7-T06/T07 bridge tests) pulls the *old* model, which
# does a bare ``from utils.custom_logger import ...``. That resolves to the **repo-root**
# ``utils`` package (which has ``custom_logger``); the sibling ``model/vae_teb_prediction/
# utils`` package has no ``custom_logger`` and shadows it if it is on ``sys.path`` ahead of
# the repo root. So drop the sibling dir and put the repo root first -- mirrors the guard in
# test_eval_v2 / test_train_v2 (which only need the repo root for the *new* model).
_REPO_ROOT = str(Path(__file__).resolve().parents[6])
_VAE_TEB_DIR = str(Path(__file__).resolve().parents[4])
sys.path[:] = [p for p in sys.path if p not in (_REPO_ROOT, _VAE_TEB_DIR)]
sys.path.insert(0, _REPO_ROOT)


def _ensure_repo_root_utils() -> None:
    r"""Make the repo-root ``utils`` (with ``custom_logger``) win over the sibling package.

    pytest keeps re-inserting ``model/vae_teb_prediction`` on ``sys.path`` as the test
    package's rootpath (its dir has no ``__init__.py``); that dir's ``utils`` subpackage
    lacks ``custom_logger`` and shadows the repo-root ``utils`` the *old* model imports.
    Call this at the start of any test that pulls the ``testing`` stack: it drops the
    sibling dir and any mis-resolved cached ``utils`` module so the import resolves to the
    repo-root package. (The real ``run_pipeline_v2 --stage test_plots`` path is unaffected:
    it runs with the repo root -- not the sibling -- on ``sys.path``.)
    """
    import importlib

    while _VAE_TEB_DIR in sys.path:
        sys.path.remove(_VAE_TEB_DIR)
    if not sys.path or sys.path[0] != _REPO_ROOT:
        if _REPO_ROOT in sys.path:
            sys.path.remove(_REPO_ROOT)
        sys.path.insert(0, _REPO_ROOT)
    for name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
        mod = sys.modules.get(name)
        if (getattr(mod, "__file__", None) or "").startswith(_VAE_TEB_DIR):
            del sys.modules[name]
    importlib.invalidate_caches()


from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v2 as viz


def _assert_written(paths) -> None:
    r"""Assert every returned figure path exists and is non-empty."""
    assert paths, "no figure files were written"
    for path in paths:
        p = Path(path)
        assert p.is_file(), f"missing figure file: {p}"
        assert p.stat().st_size > 0, f"empty figure file: {p}"


# ---------------------------------------------------------------------------
# S7-T01: plot_style_v2 + annotated raw-signal figure
# ---------------------------------------------------------------------------


def test_plot_style_v2_public_api() -> None:
    r"""``plot_style_v2`` exposes the house-style API the gallery depends on."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        plot_style_v2 as ps,
    )

    for name in ("apply_style", "style_axes", "stacked_figure", "attach_colorbar",
                 "save_figure", "add_caption", "COLOR_BLUE", "SAVE_DPI"):
        assert hasattr(ps, name), f"plot_style_v2 missing {name}"
    ps.apply_style()  # idempotent; must not raise
    assert ps.SAVE_DPI == 600


def test_raw_preview_writes_pdf_and_png(tmp_path: Path) -> None:
    r"""[raw] The annotated raw preview writes a PDF and a 600-dpi PNG."""
    rng = np.random.default_rng(0)
    n_raw = 1200  # 5 min at 4 Hz -- enough to render, small enough to be fast
    fhr = 140.0 + 5.0 * rng.standard_normal(n_raw).cumsum() / np.sqrt(n_raw)
    up = 15.0 + 3.0 * rng.standard_normal(n_raw).cumsum() / np.sqrt(n_raw)
    out = viz.plot_raw_preview(
        fhr, up, tmp_path / "raw_preview",
        meta={"te_inj": 2.0, "D": 8, "B": 1.23, "f_pulse": 0.06},
    )
    _assert_written(out)
    assert {p.suffix for p in map(Path, out)} == {".pdf", ".png"}


def test_raw_preview_with_phase_harmonics(tmp_path: Path) -> None:
    r"""[raw] Supplying the phase-harmonic fields adds the heatmap panels and still writes.

    Exercises the four-panel path (two raw traces + two ``bwr`` phase-harmonic heatmaps);
    the fields are tiny random z-scored stand-ins (the real transform is covered by the
    slow scattering-adapter test).
    """
    rng = np.random.default_rng(3)
    n_raw, t_dec = 1200, 300
    fhr = 140.0 + 5.0 * rng.standard_normal(n_raw)
    up = 15.0 + 3.0 * rng.standard_normal(n_raw)
    fhr_ph = rng.standard_normal((t_dec, 44)).astype(np.float32)
    up_ph = rng.standard_normal((t_dec, 58)).astype(np.float32)
    out = viz.plot_raw_preview(
        fhr, up, tmp_path / "raw_preview_ph",
        meta={"te_inj": 2.0, "D": 8, "B": 1.23, "f_pulse": 0.06},
        fhr_ph=fhr_ph, up_ph=up_ph,
    )
    _assert_written(out)
    assert {p.suffix for p in map(Path, out)} == {".pdf", ".png"}


# ---------------------------------------------------------------------------
# S7-T02: raw + scattering paired preview
# ---------------------------------------------------------------------------


def test_raw_scatter_paired_writes_and_tracks(tmp_path: Path) -> None:
    r"""[paired] The paired preview writes files and the coupled channel tracks the latent.

    Fabricates a coupled scattering channel that is a noisy copy of the decimated latent
    (the physics of real tracking is proven by the S2-T03 scattering test); this asserts
    the plotting path and that the fed coupled row correlates with the overlaid latent.
    """
    rng = np.random.default_rng(1)
    n_raw, t_dec, n_ch, t_tot, trim = 1200, 300, 43, 330, 15
    coupled_idx = 20
    fhr_raw = 140.0 + rng.standard_normal(n_raw)
    up_raw = 15.0 + rng.standard_normal(n_raw)
    latent_c = rng.standard_normal(t_tot)
    latent_d = rng.standard_normal(t_tot)

    fhr_st = rng.standard_normal((t_dec, n_ch)).astype(np.float32)
    up_st = rng.standard_normal((t_dec, n_ch)).astype(np.float32)
    # Plant a tracking coupled channel = latent[trim:trim+T] + small noise.
    fhr_st[:, coupled_idx] = (latent_d[trim:trim + t_dec]
                              + 0.1 * rng.standard_normal(t_dec)).astype(np.float32)
    up_st[:, coupled_idx] = (latent_c[trim:trim + t_dec]
                             + 0.1 * rng.standard_normal(t_dec)).astype(np.float32)
    # Phase-harmonic fields (the other half of the model input); exercise the 6-panel path.
    fhr_ph = rng.standard_normal((t_dec, 44)).astype(np.float32)
    up_ph = rng.standard_normal((t_dec, 58)).astype(np.float32)

    out = viz.plot_raw_scatter_paired(
        fhr_raw, up_raw, fhr_st, up_st, tmp_path / "paired",
        coupled_idx=coupled_idx, latent_c=latent_c, latent_d=latent_d,
        fhr_ph=fhr_ph, up_ph=up_ph,
        center_freqs=np.linspace(0.001, 0.37, n_ch - 1), trim=trim,
        meta={"te_inj": 3.0, "D": 8},
    )
    _assert_written(out)
    corr = np.corrcoef(up_st[:, coupled_idx], latent_c[trim:trim + t_dec])[0, 1]
    assert np.isfinite(corr) and corr > 0.5


# ---------------------------------------------------------------------------
# S7-T03: latent/AM decomposition + diagnostics panel
# ---------------------------------------------------------------------------


def _tiny_config() -> dict:
    r"""Load the real ``config_synth_v2.yaml`` (the generators read it directly)."""
    import yaml
    cfg_path = (Path(__file__).resolve().parent.parent / "config_synth_v2.yaml")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("render_mode", ["am_carrier", "pulse_train"])
def test_latent_am_decomposition_writes(tmp_path: Path, render_mode: str) -> None:
    r"""[decomp] The AM decomposition figure renders for both render modes."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        raw_generators as rg,
    )
    cfg = _tiny_config()
    out_gen = rg.generate_cell_raw(2, B=1.5, D=8, config=cfg, seed=0, te_inj=2.0,
                                   render_mode=render_mode)
    out = viz.plot_latent_am_decomposition(
        out_gen["latents"], tmp_path / f"decomp_{render_mode}",
        f_pulse=out_gen["meta"]["f_pulse"], meta=out_gen["meta"],
    )
    _assert_written(out)


# ---------------------------------------------------------------------------
# Data-generation story figures (the controls): band recipe / TE authoring /
# latent coupling / AM separation
# ---------------------------------------------------------------------------


def test_band_spectra_writes(tmp_path: Path) -> None:
    r"""[spectra] The band-recipe Welch-PSD figure writes a PDF and a 600-dpi PNG."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        raw_generators as rg,
    )
    gen = rg.generate_cell_raw(2, B=1.5, D=8, config=_tiny_config(), seed=0, te_inj=2.0)
    out = viz.plot_band_spectra(
        gen["fhr_raw"], gen["up_raw"], tmp_path / "band_spectra",
        fs=gen["meta"]["fs"], meta=gen["meta"],
    )
    _assert_written(out)
    assert {p.suffix for p in map(Path, out)} == {".pdf", ".png"}


def test_latent_coupling_writes(tmp_path: Path) -> None:
    r"""[coupling] The coupling-pathway + lag figure writes a PDF and a PNG."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        raw_generators as rg,
    )
    gen = rg.generate_cell_raw(3, B=1.5, D=8, config=_tiny_config(), seed=0, te_inj=2.0)
    out = viz.plot_latent_coupling(
        gen["latents"], tmp_path / "latent_coupling", D=8, horizon=30, fs=gen["meta"]["fs"],
    )
    _assert_written(out)


def test_am_separation_writes(tmp_path: Path) -> None:
    r"""[separation] The AM-separation carrier de-risk figure writes a PDF and a PNG (analytic, fast)."""
    out = viz.plot_am_separation(_tiny_config(), tmp_path / "am_separation")
    _assert_written(out)


def test_te_authoring_writes(tmp_path: Path) -> None:
    r"""[authoring] The TE-authoring figure writes a PDF and a PNG (tiny B-sweep + MC to stay fast)."""
    out = viz.plot_te_authoring(
        _tiny_config(), tmp_path / "te_authoring", delay=8, n_b=3, n_samples=400,
        target_te_grid=[0.0, 1.0],
    )
    _assert_written(out)


def _fake_metrics() -> dict:
    r"""Build a small metrics.json-shaped dict spanning a few cells (signal + null).

    Includes the S7 prediction-gap + profile fields: per-cell ``pred_gain``/``uplift_rel``,
    both null controls, ``warmup``/``horizon``, and a **string-keyed** ``per_cell_profiles``
    (mixed lags D in {8, 12} + a null cell) mimicking the JSON round-trip so the
    ``int()``-coercion in :func:`visualize_v2._per_cell_profiles` is exercised.
    """
    cells = [(0.0, 8), (1.0, 8), (2.0, 12), (3.0, 8)]
    per_cell = []
    for cid, (te, D) in enumerate(cells):
        per_cell.append({
            "cell_id": cid, "te_inj": te, "te_scat": te * 0.9, "D": D,
            "kbar_mean": 0.2 + 0.4 * te, "n": 300, "frac_phi": None if te == 0 else 1.1,
            "pred_gain": 0.0 if te == 0 else 0.02 * te,
            "uplift_rel": 0.0 if te == 0 else 0.05 * te,
            "lag_mass": 0.0 if te == 0 else 0.85, "peak_lag_err": 0.0,
            "null_shuffle_ratio": 1.0 if te == 0 else 0.1,
            "null_reverse_ratio": 0.98 if te == 0 else 0.12,
        })
    rng = np.random.default_rng(0)
    profiles = {}
    for cid, (te, D) in enumerate(cells):
        a = np.full(91, 0.3)
        if te > 0:  # a bump inside the true band for signal cells
            a = a + 3.0 * np.exp(-0.5 * ((np.arange(91) - (D - 1.5)) / 2.0) ** 2)
        profiles[str(cid)] = {   # string keys mimic the JSON round-trip
            "lag_profile": list(a),
            "kbar_over_time": list(0.2 + 0.4 * te + 0.05 * rng.standard_normal(300)),
            "lag_count": 300,
        }
    return {
        "run_tag": "unit", "split": "test", "warmup": 30, "horizon": 30,
        "calibration": {"gamma_inj": 0.98, "alpha_inj": 0.2, "r2_inj": 0.99,
                        "gamma_scat": 1.05, "alpha_scat": 0.2, "r2_scat": 0.98,
                        "n_cells": len(per_cell)},
        "lag_recovery": {"mean_lag_mass": 0.85, "lag_mass_threshold": 0.8},
        "per_cell": per_cell,
        "per_cell_profiles": profiles,
    }


def _fake_realizability() -> dict:
    r"""A ``realizability.json``-shaped dict giving each cell a ``te_raw`` (for three-TE)."""
    return {"per_cell": [
        {"cell_id": cid, "te_raw": 0.0 if te == 0 else 0.45 * te}
        for cid, (te, _D) in enumerate([(0.0, 8), (1.0, 8), (2.0, 12), (3.0, 8)])
    ]}


def _fake_per_sample(n_per_cell: int = 60) -> dict:
    r"""A ``per_sample_eval.npz``-shaped dict of length-N arrays over the same cells.

    Mirrors the four ``_fake_metrics`` cells (TE $\in \{0,1,2,3\}$, lags $\{8, 12\}$) with a
    per-sample $\bar K$ scatter around the cell mean, so the per-sample scatter / per-lag
    calibration figures have a real cloud to draw.
    """
    cells = [(0.0, 8), (1.0, 8), (2.0, 12), (3.0, 8)]
    rng = np.random.default_rng(1)
    kbar, te_inj, te_scat, cell_id, delay = [], [], [], [], []
    for cid, (te, D) in enumerate(cells):
        kbar.append(0.2 + 0.4 * te + 0.05 * rng.standard_normal(n_per_cell))
        te_inj.append(np.full(n_per_cell, te))
        te_scat.append(np.full(n_per_cell, te * 0.9))
        cell_id.append(np.full(n_per_cell, cid))
        delay.append(np.full(n_per_cell, D))
    return {
        "kbar": np.concatenate(kbar),
        "te_inj": np.concatenate(te_inj),
        "te_scat": np.concatenate(te_scat),
        "cell_id": np.concatenate(cell_id),
        "delay": np.concatenate(delay),
        "split": np.asarray("test"),
    }


def test_diagnostics_panel_writes(tmp_path: Path) -> None:
    r"""[diag] The 2x2 diagnostics panel renders from a metrics dict."""
    out = viz.plot_diagnostics_panel(_fake_metrics(), tmp_path / "diagnostics")
    _assert_written(out)


# ---------------------------------------------------------------------------
# S7-T08: TE-aware aggregate figures
# ---------------------------------------------------------------------------


def test_aggregate_figures_write(tmp_path: Path) -> None:
    r"""[aggregate] The grouped calibration / preservation / lag-mass figures render."""
    metrics = _fake_metrics()
    _assert_written(viz.plot_calibration_by_lag(metrics, tmp_path / "calib_by_lag"))
    _assert_written(viz.plot_frac_phi_distribution(metrics, tmp_path / "frac_dist",
                                                   frac_threshold=0.7))
    _assert_written(viz.plot_lag_mass_summary(metrics, tmp_path / "lag_summary"))


def test_te_kld_scatter_writes(tmp_path: Path) -> None:
    r"""[aggregate] The per-sample TE-vs-K̄ scatter renders (and the no-data fallback)."""
    metrics = _fake_metrics()
    metrics["calibration"].update({
        "gamma_inj_sample": 0.4, "alpha_inj_sample": 0.2, "r2_inj_sample": 0.9,
        "gamma_scat_sample": 0.44, "alpha_scat_sample": 0.2, "r2_scat_sample": 0.88,
        "n_samples": 240,
    })
    per_sample = _fake_per_sample()
    _assert_written(viz.plot_te_kld_scatter(per_sample, metrics, tmp_path / "te_kld_scatter"))
    # Graceful placeholder when no per-sample arrays are available.
    _assert_written(viz.plot_te_kld_scatter(None, metrics, tmp_path / "te_kld_empty"))


def test_calibration_by_lag_per_sample(tmp_path: Path) -> None:
    r"""[aggregate] calibration_by_lag draws per-lag small multiples from per-sample data."""
    metrics = _fake_metrics()
    metrics["calibration"]["by_lag"] = {
        "8": {"gamma_inj": 0.40, "alpha_inj": 0.20, "r2_inj": 0.90,
              "gamma_scat": 0.44, "alpha_scat": 0.20, "r2_scat": 0.88, "n": 180},
        "12": {"gamma_inj": 0.41, "alpha_inj": 0.19, "r2_inj": 0.90,
               "gamma_scat": 0.45, "alpha_scat": 0.20, "r2_scat": 0.88, "n": 60},
    }
    per_sample = _fake_per_sample()
    _assert_written(viz.plot_calibration_by_lag(metrics, tmp_path / "calib_by_lag_ps",
                                                per_sample=per_sample))


# ---------------------------------------------------------------------------
# S8: KLD-summary family vs TE figures (§14.5)
# ---------------------------------------------------------------------------


def _fake_per_sample_full(n_per_cell: int = 60) -> dict:
    r"""Augment :func:`_fake_per_sample` with the KLD summary family + per-head columns.

    Adds the ``kbar_sum`` / ``kbar_max`` / ``kbar_median`` / ``kbar_p90`` / ``kbar_full`` /
    ``kbar_postwarm`` / ``kbar_inband`` / ``kbar_outband`` scalar summaries and the per-head
    ``kbar_head{m}`` columns (head 0 carries the coupling, the rest are flat), so the S8
    figures have a full family to draw.
    """
    ps = _fake_per_sample(n_per_cell)
    base = ps["kbar"]
    ps["kbar_sum"] = base * 100.0
    ps["kbar_max"] = base * 2.0
    ps["kbar_median"] = base.copy()
    ps["kbar_p90"] = base * 1.5
    ps["kbar_full"] = base * 0.9
    ps["kbar_postwarm"] = base * 0.95
    inb = base * 0.6
    ps["kbar_inband"] = inb
    ps["kbar_outband"] = base - inb
    rng = np.random.default_rng(2)
    for m in range(4):
        ps[f"kbar_head{m}"] = (base * 0.6 if m == 0 else 0.1 + 0.02 * rng.standard_normal(base.size))
    ps["kbar_head"] = np.stack([ps[f"kbar_head{m}"] for m in range(4)], axis=1)
    return ps


def _fake_kld_variants() -> dict:
    r"""A ``calibration.kld_variants``-shaped dict spanning the summary family + two heads."""
    names = ["kbar", "kbar_sum", "kbar_max", "kbar_median", "kbar_p90", "kbar_full",
             "kbar_postwarm", "kbar_inband", "kbar_outband", "kbar_head0", "kbar_head1"]
    kv = {}
    for i, v in enumerate(names):
        kv[v] = {"gamma_inj": 0.4 - 0.02 * i, "alpha_inj": 0.2, "r2_inj": 0.9,
                 "pearson_inj": 0.95 - 0.06 * i, "spearman_inj": 0.93 - 0.06 * i,
                 "gamma_scat": 0.44, "alpha_scat": 0.2, "r2_scat": 0.88,
                 "pearson_scat": 0.9, "spearman_scat": 0.9, "n": 240}
    return kv


def test_kld_variant_figures_write(tmp_path: Path) -> None:
    r"""[kld] The S8 KLD-summary-family figures render from per-sample + kld_variants data."""
    metrics = _fake_metrics()
    metrics["calibration"]["kld_variants"] = _fake_kld_variants()
    metrics["calibration"].update({
        "gamma_inj_sample": 0.4, "alpha_inj_sample": 0.2, "r2_inj_sample": 0.9,
        "gamma_scat_sample": 0.44, "alpha_scat_sample": 0.2, "r2_scat_sample": 0.88,
        "n_samples": 240,
    })
    ps = _fake_per_sample_full()
    _assert_written(viz.plot_kld_variants_vs_te(ps, metrics, tmp_path / "kld_variants"))
    _assert_written(viz.plot_kld_variants_vs_te(ps, metrics, tmp_path / "kld_variants_scat",
                                                te_axis="scat"))
    _assert_written(viz.plot_kld_te_correlation(metrics, tmp_path / "kld_corr"))
    _assert_written(viz.plot_kld_te_density(ps, metrics, tmp_path / "kld_density"))
    _assert_written(viz.plot_kld_te_density(ps, metrics, tmp_path / "kld_density_inband",
                                            variant="kbar_inband"))
    _assert_written(viz.plot_kld_distribution_by_te(ps, metrics, tmp_path / "kld_dist"))
    _assert_written(viz.plot_per_head_kld_vs_te(ps, metrics, tmp_path / "per_head_kld"))


def test_kld_variant_figures_degrade_on_empty(tmp_path: Path) -> None:
    r"""[kld] The S8 figures degrade gracefully with no per-sample / no kld_variants data."""
    metrics = {"run_tag": "empty", "split": "test", "calibration": {}}
    _assert_written(viz.plot_kld_variants_vs_te({}, metrics, tmp_path / "kv_empty"))
    _assert_written(viz.plot_kld_variants_vs_te(None, metrics, tmp_path / "kv_none"))
    _assert_written(viz.plot_kld_te_correlation(metrics, tmp_path / "kc_empty"))
    _assert_written(viz.plot_kld_te_density({}, metrics, tmp_path / "kd_empty"))
    _assert_written(viz.plot_kld_distribution_by_te({}, metrics, tmp_path / "kdist_empty"))
    _assert_written(viz.plot_per_head_kld_vs_te({}, metrics, tmp_path / "ph_empty"))


# ---------------------------------------------------------------------------
# S7 extension: prediction-gap + previously-unplotted diagnostics
# ---------------------------------------------------------------------------


def test_pred_gain_figures_write(tmp_path: Path) -> None:
    r"""[pred_gain] The pred-gain-vs-TE / vs-Kbar and three-TE figures render."""
    metrics = _fake_metrics()
    realiz = _fake_realizability()
    _assert_written(viz.plot_pred_gain_vs_te(metrics, tmp_path / "pred_gain_vs_te",
                                             realizability=realiz))
    _assert_written(viz.plot_pred_gain_vs_kbar(metrics, tmp_path / "pred_gain_vs_kbar"))
    _assert_written(viz.plot_three_te(metrics, tmp_path / "three_te", realizability=realiz))
    # three-TE degrades to two series when realizability (te_raw) is absent.
    _assert_written(viz.plot_three_te(metrics, tmp_path / "three_te_none",
                                      realizability=None))


def test_profile_figures_write(tmp_path: Path) -> None:
    r"""[profile] The per-cell lag-profile / KLD-over-time / null-control figures render."""
    metrics = _fake_metrics()
    _assert_written(viz.plot_lag_profiles(metrics, tmp_path / "lag_profiles"))
    _assert_written(viz.plot_kld_vs_time(metrics, tmp_path / "kld_vs_time"))
    _assert_written(viz.plot_null_controls(metrics, tmp_path / "null_controls"))


def test_pred_gain_and_profiles_degrade_on_empty(tmp_path: Path) -> None:
    r"""[pred_gain] The new plotters degrade gracefully on a metrics dict with no data."""
    empty = {"run_tag": "empty", "split": "test", "per_cell": [], "per_cell_profiles": {}}
    _assert_written(viz.plot_pred_gain_vs_te(empty, tmp_path / "pg_te_empty"))
    _assert_written(viz.plot_pred_gain_vs_kbar(empty, tmp_path / "pg_kbar_empty"))
    _assert_written(viz.plot_lag_profiles(empty, tmp_path / "lagprof_empty"))
    _assert_written(viz.plot_kld_vs_time(empty, tmp_path / "kldt_empty"))
    _assert_written(viz.plot_null_controls(empty, tmp_path / "null_empty"))


def test_report_assembles_gallery_and_gates(tmp_path: Path) -> None:
    r"""[report] final_report_v2 writes report.md referencing the figures + metrics gates."""
    import copy

    import yaml

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    cfg_path = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))
    cfg["experiment"]["tag"] = "report_unit"
    cfg["experiment"]["benchmark"] = "G1_raw"
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")

    results_dir = tmp_path / "results" / "report_unit"
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    # A metrics.json (so the headline figure + gates render) and a couple of gallery files.
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(_fake_metrics(), handle)
    (results_dir / "figures" / "raw_preview.pdf").write_bytes(b"%PDF-1.4 stub")
    samples = results_dir / "test_plots" / "samples_diag"
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "s0.pdf").write_bytes(b"%PDF-1.4 stub")
    with open(samples / "sample_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        w = __import__("csv").writer(handle)
        w.writerow(["guid", "te_true", "te_scat", "frac_phi", "sample_delay", "kld_mean",
                    "out_path"])
        w.writerow(["report_unit_test_000000", "2.0", "1.8", "0.9", "8", "1.05", "s0.pdf"])

    report_path = fr.final_report_v2(cfg, benchmark="G1_raw")
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    for token in ("final report", "Headline gates", "gamma", "frac_", "Figure gallery",
                  "raw_preview", "TE_inj", "Representative sample"):
        assert token in text or token.lower() in text.lower(), token
    # the headline diagnostics figure was rendered from the metrics.
    assert (results_dir / "figures" / "headline_diagnostics.pdf").is_file()


# ---------------------------------------------------------------------------
# S7-T06: standard-testing metadata bridge (additive & guarded)
# ---------------------------------------------------------------------------

_CACHE_CHANNELS = {"fhr_st": 43, "fhr_ph": 44, "up_st": 43, "up_ph": 58}


def _write_tiny_v2_cache(cache_dir: Path, *, T: int = 32, per_cell: int = 4,
                         delay: int = 4, horizon: int = 8) -> Path:
    r"""Write a tiny v2 ``test.npz`` + ``meta.json`` (2 cells) with full provenance.

    Mirrors the fixture cache used by ``test_eval_v2`` so the standard-testing TE bridge
    can be exercised without the real transform. Returns the ``test.npz`` path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cells = [
        {"cell_id": 0, "te_inj": 0.0, "te_scat": -0.1, "frac_phi": float("nan"), "delay": delay},
        {"cell_id": 1, "te_inj": 2.0, "te_scat": 1.8, "frac_phi": 0.9, "delay": delay},
    ]
    n = per_cell * len(cells)
    arrays = {f: rng.standard_normal((n, T, c)).astype(np.float32)
              for f, c in _CACHE_CHANNELS.items()}
    arrays["weight"] = np.ones((n, T), np.float32)
    arrays["true_lag_tt"] = np.zeros((n, T), np.float32)
    te_true = np.empty(n, np.float32); te_scat = np.empty(n, np.float32)
    frac = np.empty(n, np.float32); delay_a = np.empty(n, np.int16)
    cid = np.empty(n, np.int16); held = np.zeros(n, np.int8)
    for i, cell in enumerate(cells):
        sl = slice(i * per_cell, (i + 1) * per_cell)
        te_true[sl] = cell["te_inj"]; te_scat[sl] = cell["te_scat"]
        frac[sl] = cell["frac_phi"]; delay_a[sl] = cell["delay"]; cid[sl] = cell["cell_id"]
        arrays["true_lag_tt"][sl] = cell["delay"]
    arrays.update({
        "sample_te_true": te_true, "sample_te_scat": te_scat, "sample_frac_phi": frac,
        "sample_delay": delay_a, "sample_cell_id": cid, "sample_held_out": held,
    })
    np.savez(cache_dir / "test.npz", **arrays)
    meta = {"te_true": 1.0, "tag": "unit", "benchmark": "G1_raw",
            "true_lag_band": list(range(max(0, delay - horizon), delay))}
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return cache_dir / "test.npz"


def test_metadata_bridge_extractors_carry_te(tmp_path: Path) -> None:
    r"""[metadata_bridge] A collated v2 batch exposes the TE/lag fields to the extractors."""
    _ensure_repo_root_utils()
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import dataset_v2 as ds2
    from model.vae_teb_prediction.testing import collectors

    npz = _write_tiny_v2_cache(tmp_path / "cache")
    dataset = ds2.SyntheticTEDatasetV2(npz)
    loader = ds2.make_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert collectors._extract_te_true(batch, 0) is not None
    assert collectors._extract_scalar_field(batch, "te_scat", 0) is not None
    assert collectors._extract_scalar_field(batch, "frac_phi", 0) is not None
    assert collectors._extract_delay(batch, 0) == 4
    assert collectors._extract_int_field(batch, "cell_id", 0) is not None
    lag_tt = collectors._extract_array_field(batch, "true_lag_tt", 0)
    assert lag_tt is not None and lag_tt.shape[0] == 32
    assert collectors._extract_array_field(batch, "true_lag_band", 0) is not None


def test_metadata_bridge_extractors_none_on_real_batch() -> None:
    r"""[metadata_bridge] The guarded extractors return None when the field is absent (real data)."""
    _ensure_repo_root_utils()
    from model.vae_teb_prediction.testing import collectors

    class _Bare:
        pass

    bare = _Bare()
    assert collectors._extract_scalar_field(bare, "te_scat", 0) is None
    assert collectors._extract_delay(bare, 0) is None
    assert collectors._extract_int_field(bare, "cell_id", 0) is None
    assert collectors._extract_array_field(bare, "true_lag_tt", 0) is None


def test_sample_te_diagnostic_aware(tmp_path: Path) -> None:
    r"""[sample_te] The one-sample diagnostic renders the TE metadata + true-lag band."""
    _ensure_repo_root_utils()
    from model.vae_teb_prediction.testing.plot_single_samples import (
        plot_sample_lag_attn_diagnostic,
    )
    T, H, C, L, dz = 20, 4, 87, 8, 4
    rng = np.random.default_rng(0)
    sample = {
        "mu_full": rng.standard_normal((T, H, C)).astype(np.float32),
        "mu_base": rng.standard_normal((T, H, C)).astype(np.float32),
        "delta_src": rng.standard_normal((T, H, C)).astype(np.float32),
        "y_plus": rng.standard_normal((T - H, H, C)).astype(np.float32),
        "z": rng.standard_normal((T, dz)).astype(np.float32),
        "attn": rng.random((T, 2, L)).astype(np.float32),
        "te_lag": rng.random((T, L)).astype(np.float32),
        "kld_t": rng.random(T).astype(np.float32),
        "kld_sum_t": rng.random(T).astype(np.float32),
        "kld_l2_t": rng.random(T).astype(np.float32),
        "kld_per_dim": rng.random((T, dz)).astype(np.float32),
        "fhr": None, "up": None, "guid": "unit_000", "epoch": None, "label": None,
        "metrics": {"kld_mean": 1.2},
    }
    out = tmp_path / "sample_diag.pdf"
    plot_sample_lag_attn_diagnostic(
        sample, out, warmup=2, horizon=H,
        true_te=2.0, te_scat=1.8, te_raw=1.5, frac_phi=0.9, delay=4, kld_value=1.2,
        true_lag_tt=np.full(T, 4.0), true_lag_band=np.array([0, 1, 2, 3]),
    )
    assert out.is_file() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# S7-T07: run_tests.py bridge smoke (real testing pipeline, tiny cache)
# ---------------------------------------------------------------------------

_T07_T, _T07_H, _T07_WARM, _T07_LAG, _T07_DELAY = 32, 4, 2, 8, 4
_T07_MODEL = {
    "sequence_length": _T07_T, "d_model": 16, "d_z": 4, "horizon": _T07_H,
    "warmup_period": _T07_WARM, "c_y": 87, "c_u": 101, "use_up_st": True,
    "max_lag": _T07_LAG, "num_heads": 2, "d_head": 8, "lstm_layers": 1, "dropout": 0.0,
    "decoder_hidden": 16, "logvar_clamp": [-5.0, 3.0], "mu_scale": 5.0,
    "delta_mu_scale": 3.0, "latent_stats_momentum": 0.01, "use_entmax": False,
    "attention_grad_checkpoint": False, "head_structured_latent": False,
    "lag_bias_init": "normal", "horizon_depth": 1, "horizon_kernel": 3,
    "horizon_film": False, "encoder_extra_dilations": [],
}


@pytest.mark.slow
def test_test_plots_bridge_writes_te_annotated_diagnostics(tmp_path: Path) -> None:
    r"""[bridge] ``run_test_plots`` drives run_tests.py -> TE-annotated sample PDFs + CSV.

    Builds a tiny v2 cache + a tiny checkpoint, runs the standard testing pipeline through
    ``loader_override`` (no HDF5 / stats), and asserts the ``samples_diag`` PDFs and
    ``sample_metrics.csv`` (with the TE provenance columns) are produced.
    """
    _ensure_repo_root_utils()
    import copy
    import csv as _csv

    import torch
    import yaml

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import pl_module_v2 as plm
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
        run_pipeline_v2 as rp,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        resolve_cache_dir,
    )

    cfg_path = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))
    cfg["model"] = copy.deepcopy(_T07_MODEL)
    cfg["experiment"]["tag"] = "test_plots_unit"
    cfg["experiment"]["benchmark"] = "G1_raw"
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg.setdefault("dataset", {}).update({"num_workers": 0, "pin_memory": False,
                                          "persistent_workers": False, "mmap": "auto"})
    cfg.setdefault("optim", {})["batch_size"] = 4

    cache_dir = resolve_cache_dir(cfg, benchmark="G1_raw")
    _write_tiny_v2_cache(cache_dir, T=_T07_T, per_cell=4, delay=_T07_DELAY, horizon=_T07_H)

    results_dir = Path(cfg["paths"]["results_dir"]) / "test_plots_unit"
    results_dir.mkdir(parents=True, exist_ok=True)
    model, kwargs = plm.build_model(cfg["model"], torch.device("cpu"))
    plm.save_checkpoint_v2(
        results_dir / "final.ckpt", model=model, model_kwargs=kwargs, config=cfg,
        data_meta={}, epoch=1, val_loss=float("nan"),
        loss_settings={"beta": 1e-3}, latent_stats_fitted=False,
    )

    out = rp.run_test_plots(cfg, benchmark="G1_raw", split="test", analysis_samples=2,
                            out_dir=results_dir)
    samples_dir = out["out_dir"] / "samples_diag"
    pdfs = list(samples_dir.glob("*.pdf"))
    assert pdfs, f"no sample PDFs under {samples_dir}"
    csv_path = samples_dir / "sample_metrics.csv"
    assert csv_path.is_file()
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        cols = set(next(_csv.reader(handle)))
    for col in ("te_true", "te_scat", "frac_phi", "sample_delay", "cell_id", "kld_mean"):
        assert col in cols, f"missing TE column {col!r} in sample_metrics.csv ({cols})"


# ---------------------------------------------------------------------------
# Interactive Plotly HTML loss curve (S5-T04 live callback backend)
# ---------------------------------------------------------------------------
def _write_lightning_metrics_csv(path: Path, *, n_epochs: int = 3) -> None:
    r"""Write a realistic Lightning-style ``metrics.csv``.

    Mirrors the real v2 log: every metric is forked into a ``_step`` and an ``_epoch``
    column, ``train`` / ``val`` are separate keys, and the bookkeeping ``epoch`` /
    ``step`` columns plus the ``LearningRateMonitor`` ``lr-AdamW`` duplicate are present
    -- so the HTML enumeration's suffix-collapse and exclusion logic is exercised.
    """
    import csv as _csv

    train_metrics = [
        "total_loss", "feat_loss", "base_loss", "kld_loss", "kld_nats",
        "pred_gap", "mu_prior_sat_frac", "delta_mu_sat_frac", "kld_beta",
        "spike_ema_loss", "spike_skips_total",
    ]
    val_metrics = [
        "total_loss", "feat_loss", "base_loss", "kld_loss", "kld_nats",
        "pred_gap", "mu_prior_sat_frac", "delta_mu_sat_frac", "kld_beta",
    ]
    fieldnames = ["epoch", "step"]
    for m in train_metrics:
        fieldnames += [f"train/{m}_step", f"train/{m}_epoch"]
    for m in val_metrics:
        fieldnames.append(f"val/{m}_epoch")
    fieldnames += ["lr", "lr-AdamW"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(n_epochs):
            scale = 1.0 / (epoch + 1)
            row = {
                "epoch": epoch,
                "step": (epoch + 1) * 10,
                "lr": 1e-3 * scale,
                "lr-AdamW": 1e-3 * scale,
            }
            for i, m in enumerate(train_metrics):
                row[f"train/{m}_step"] = ""  # step rows blank at the epoch aggregate
                row[f"train/{m}_epoch"] = round(scale * (i + 1), 4)
            for i, m in enumerate(val_metrics):
                row[f"val/{m}_epoch"] = round(scale * (i + 1) * 1.1, 4)
            writer.writerow(row)


def test_loss_curves_html_written(tmp_path) -> None:
    r"""``plot_loss_curves_html`` overlays every logged metric as its own distinct trace."""
    pytest.importorskip("plotly")
    metrics_csv = tmp_path / "metrics.csv"
    _write_lightning_metrics_csv(metrics_csv)

    out = viz.plot_loss_curves_html(
        metrics_csv, tmp_path / "figures" / "training_curves"
    )
    assert out is not None and len(out) == 1
    html_path = out[0]
    assert html_path.suffix == ".html"
    assert html_path.is_file() and html_path.stat().st_size > 0
    # ``include_plotlyjs=True`` embeds the library, so the file is self-contained.
    assert "plotly" in html_path.read_text(encoding="utf-8").lower()

    # Every logged metric becomes its own trace -- far more than the old 6-trace curve --
    # and no two traces share a colour. Assert on the enumeration + colour helpers.
    import csv as _csv

    with open(metrics_csv, newline="", encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    triples = viz._enumerate_html_metrics(rows)
    keys = [k for _, k, _ in triples]
    labels = [lbl for lbl, _, _ in triples]
    assert len(triples) > 6
    # Bookkeeping / duplicate columns are excluded; the bare ``lr`` trace is kept.
    assert not any(k in ("epoch", "step", "lr-AdamW") for k in keys)
    assert "lr" in keys
    # A metric's train series is drawn immediately before its val twin.
    assert labels.index("val total_loss") == labels.index("train total_loss") + 1
    # Distinct colour per trace.
    colors = viz._html_trace_colors(len(triples))
    assert len(set(colors)) == len(colors) == len(triples)


def test_loss_curves_html_missing_csv_is_noop(tmp_path) -> None:
    r"""A missing ``metrics.csv`` is non-fatal: returns ``None`` and writes nothing."""
    pytest.importorskip("plotly")
    out = viz.plot_loss_curves_html(
        tmp_path / "does_not_exist.csv", tmp_path / "figures" / "curve"
    )
    assert out is None
    assert not (tmp_path / "figures" / "curve.html").exists()
