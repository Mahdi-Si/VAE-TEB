r"""Sprint 7 (G-G): tests for the ground-truth latent provider and the neural CMI estimator.

Two tiers, matching the Sprint 5/6 convention:

* **Fast** (no cache, no checkpoint): the InfoNCE arithmetic, the ceiling, determinism, the
  synthetic-recovery thresholds, the per-cell aggregation schema, the ``bias`` identity, the
  graceful-degradation path, and the report section's ``n/a``.
* **Slow** (``@pytest.mark.slow``, real cache): the latent provider's grid alignment and its
  agreement with the build's own generation path.

Two acceptance criteria here are **stronger than the ones S7-T01 specified**, because the specified
ones do not test what they claim (see :func:`test_latent_crop_aligns_with_the_cached_feature` and
:func:`test_coupling_lag_recovers_D_exactly`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from .conftest import find_cache_dir

_SV2 = Path(__file__).resolve().parents[1]


# ===========================================================================
# Helpers
# ===========================================================================
def _load_config():
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
        load_config,
    )

    return load_config(_SV2 / "config_synth_v3.yaml")


def _xcorr_peak(a_rows: List[np.ndarray], b_rows: List[np.ndarray],
                lags: np.ndarray) -> Tuple[int, float]:
    r"""``argmax`` over ``lags`` of the mean correlation of ``a[t]`` with ``b[t + lag]``."""
    scores = np.zeros(len(lags), dtype=np.float64)
    for a, b in zip(a_rows, b_rows):
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        for i, lag in enumerate(lags):
            if lag >= 0:
                scores[i] += float(np.dot(a[: len(a) - lag], b[lag:])) / (len(a) - lag)
            else:
                m = -lag
                scores[i] += float(np.dot(a[m:], b[: len(b) - m])) / (len(b) - m)
    scores /= max(len(a_rows), 1)
    k = int(np.argmax(scores))
    return int(lags[k]), float(scores[k])


def _joint_regression_lag(c_rows, d_rows, lmax: int) -> Tuple[int, np.ndarray]:
    r"""``argmax_l |beta_l|`` from the OLS fit $d_k \sim [d_{k-1},\, c_{k-0}, \dots, c_{k-l_{max}}]$.

    Because the data-generating process is exactly
    $d_k = A_y d_{k-1} + B\,c_{k-D} + \varepsilon_k$, conditioning on $d_{k-1}$ and regressing on
    the *whole* source lag window removes both the AR(1) memory and the source's own
    autocorrelation, so the surviving coefficient is non-zero only at $\ell = D$. A raw
    cross-correlation of $c$ against $d$ does **not** have this property: the AR(1) target smears
    the source response over lags $D, D+1, \dots$ with geometric weights $A_y^j$, so its peak sits
    at $D$ or $D+1$ depending on the cell.
    """
    designs, targets = [], []
    for c, d in zip(c_rows, d_rows):
        k = np.arange(lmax + 1, len(d))
        cols = [d[k - 1]] + [c[k - l] for l in range(lmax + 1)]
        designs.append(np.stack(cols, axis=1))
        targets.append(d[k])
    X = np.concatenate(designs, axis=0)
    y = np.concatenate(targets, axis=0)
    X = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    source_coefs = beta[1: lmax + 2]
    return int(np.argmax(np.abs(source_coefs))), source_coefs


@pytest.fixture(scope="module")
def latent_fixture():
    r"""``(config, cache_dir, provider, npz)`` on the ``val`` split, or ``None`` without a cache."""
    cache_dir = find_cache_dir()
    if cache_dir is None:
        return None
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        make_latent_provider,
    )

    config = _load_config()
    provider = make_latent_provider(config, "val", cache_dir=cache_dir)
    npz = np.load(cache_dir / "val.npz", mmap_mode="r")
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    return {"config": config, "cache_dir": cache_dir, "provider": provider,
            "npz": npz, "meta": meta}


# ===========================================================================
# S7-T01: the latent provider
# ===========================================================================
@pytest.mark.slow
def test_latent_provider_window_length_and_finiteness(latent_fixture) -> None:
    r"""Rows come back at ``sequence_length`` and are finite."""
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    fx = latent_fixture
    seq_len = int(fx["meta"]["sequence_length"])
    assert fx["provider"].window_length == seq_len

    cid = np.asarray(fx["npz"]["sample_cell_id"])
    ridx = np.asarray(fx["npz"]["sample_raw_index"])
    c, d = fx["provider"](int(cid[0]), int(ridx[0]))
    assert c.shape == (seq_len,) and d.shape == (seq_len,)
    assert c.dtype == np.float64 and d.dtype == np.float64
    assert np.isfinite(c).all() and np.isfinite(d).all()


@pytest.mark.slow
def test_latent_provider_matches_the_builds_own_generation(latent_fixture) -> None:
    r"""The provider's rows equal ``generate_pilot_samples``' latents under the same crop.

    This is the S7-T01 "regenerated raw matches ``make_raw_provider`` to 1e-6" criterion, asserted
    one level closer to the thing that matters: the provider must not merely produce *a* latent, it
    must produce the latent belonging to the cached row.
    """
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        CellV2,
        generate_pilot_samples,
    )

    fx = latent_fixture
    meta, npz = fx["meta"], fx["npz"]
    cid = np.asarray(npz["sample_cell_id"])
    ridx = np.asarray(npz["sample_raw_index"])
    cell_id = int(cid[0])
    row = int(ridx[0])

    spec = next(c for c in meta["cells"] if int(c["cell_id"]) == cell_id)
    cell = CellV2(cell_id=cell_id, target_te=float(spec["target_te"]), D=int(spec["D"]),
                  B_y_scalar=float(spec["B_y_scalar"]),
                  te_block_realised=float(spec["te_block_realised"]))
    n_cell = int(np.count_nonzero(cid == cell_id))
    raw = generate_pilot_samples(
        cell, n_cell, "val", fx["config"],
        base_seed=int(meta["seeds"].get("dgp", meta["seeds"].get("base_seed", 0))),
        render_mode=meta.get("render_mode"),
    )
    trim = (int(meta["raw"]["n_raw"]) // 16 - int(meta["sequence_length"])) // 2
    crop = slice(trim, trim + int(meta["sequence_length"]))

    c_got, d_got = fx["provider"](cell_id, row)
    np.testing.assert_allclose(c_got, raw["latents"]["c"][row, crop], atol=1e-6)
    np.testing.assert_allclose(d_got, raw["latents"]["d"][row, crop], atol=1e-6)


@pytest.mark.slow
def test_trim_steps_invariant_holds() -> None:
    r"""``TRIM_STEPS == (T_tot - sequence_length) // 2``: the crop that makes $t_m = t_l - 15$."""
    cache_dir = find_cache_dir()
    if cache_dir is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import (
        DECIMATION,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.scattering_adapter import (
        TRIM_STEPS,
    )

    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    t_tot = int(meta["raw"]["n_raw"]) // int(DECIMATION)
    seq_len = int(meta["sequence_length"])
    assert t_tot == 330 and seq_len == 300
    assert int(TRIM_STEPS) == (t_tot - seq_len) // 2 == 15


@pytest.mark.slow
def test_latent_crop_aligns_with_the_cached_feature(latent_fixture) -> None:
    r"""The cropped latents align at **lag 0** with the cached features of the same row.

    S7-T01 specifies "cross-correlating the cropped ``c`` against the cropped ``d`` peaks at exactly
    the cell's ``delay``". That criterion **cannot detect a crop bug**: cropping $c$ and $d$ by the
    same amount leaves their relative lag unchanged, so it passes for ``[0:300]``, ``[15:315]`` and
    ``[30:330]`` alike. What pins the crop is the latent's alignment against the *model-facing*
    cached feature, which lives on the trimmed grid. Measured here: the correct crop peaks at
    $0$; ``[0:300]`` peaks at $-15$ and ``[30:330]`` at $+15$.
    """
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    fx = latent_fixture
    npz, meta = fx["npz"], fx["meta"]
    cid = np.asarray(npz["sample_cell_id"])
    ridx = np.asarray(npz["sample_raw_index"])
    ch_fhr = int(meta["coupled_channel"]["fhr_st"])
    ch_up = int(meta["coupled_channel"]["up_st"])
    lags = np.arange(-25, 26)

    # A strongly-coupled cell, so the target channel carries a legible envelope.
    cell_id = max(
        (int(c["cell_id"]) for c in meta["cells"] if float(c["te_block_realised"]) > 1.5),
        default=None,
    )
    assert cell_id is not None, "the grid must contain a strongly-coupled cell"

    rows = np.where(cid == cell_id)[0][:48]
    c_rows, d_rows, up_rows, fhr_rows = [], [], [], []
    for i in rows:
        c, d = fx["provider"](cell_id, int(ridx[i]))
        c_rows.append(c)
        d_rows.append(d)
        up_rows.append(np.asarray(npz["up_st"][i, :, ch_up], dtype=np.float64))
        fhr_rows.append(np.asarray(npz["fhr_st"][i, :, ch_fhr], dtype=np.float64))

    # The render is affine-positive AM (A ~ affine(latent)), NOT a modulus, so the latent
    # correlates with the feature directly -- do not take an absolute value here.
    lag_src, r_src = _xcorr_peak(c_rows, up_rows, lags)
    lag_tgt, r_tgt = _xcorr_peak(d_rows, fhr_rows, lags)
    assert lag_src == 0, f"source latent misaligned with up_st by {lag_src} steps (r={r_src:.3f})"
    assert lag_tgt == 0, f"target latent misaligned with fhr_st by {lag_tgt} steps (r={r_tgt:.3f})"
    assert r_src > 0.5 and r_tgt > 0.3


@pytest.mark.slow
def test_a_deliberate_crop_error_is_caught(latent_fixture) -> None:
    r"""A $\pm 15$-step crop error shifts the latent-vs-feature peak by exactly $\mp 15$.

    The guard that :func:`test_latent_crop_aligns_with_the_cached_feature` would be useless without:
    it proves the statistic actually *moves* when the crop is wrong.
    """
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
        CellV2,
        generate_pilot_samples,
    )

    fx = latent_fixture
    npz, meta = fx["npz"], fx["meta"]
    cid = np.asarray(npz["sample_cell_id"])
    ridx = np.asarray(npz["sample_raw_index"])
    ch_up = int(meta["coupled_channel"]["up_st"])
    seq = int(meta["sequence_length"])

    cell_id = max(int(c["cell_id"]) for c in meta["cells"]
                  if float(c["te_block_realised"]) > 1.5)
    spec = next(c for c in meta["cells"] if int(c["cell_id"]) == cell_id)
    cell = CellV2(cell_id=cell_id, target_te=float(spec["target_te"]), D=int(spec["D"]),
                  B_y_scalar=float(spec["B_y_scalar"]),
                  te_block_realised=float(spec["te_block_realised"]))
    raw = generate_pilot_samples(
        cell, int(np.count_nonzero(cid == cell_id)), "val", fx["config"],
        base_seed=int(meta["seeds"].get("dgp", 0)), render_mode=meta.get("render_mode"),
    )
    c_full = raw["latents"]["c"]
    rows = np.where(cid == cell_id)[0][:48]
    up_rows = [np.asarray(npz["up_st"][i, :, ch_up], dtype=np.float64) for i in rows]
    lags = np.arange(-25, 26)

    for offset, expected in ((0, -15), (15, 0), (30, +15)):
        c_rows = [c_full[int(ridx[i]), offset: offset + seq] for i in rows]
        lag, _ = _xcorr_peak(c_rows, up_rows, lags)
        assert lag == expected, f"crop [{offset}:{offset + seq}] gave lag {lag}, expected {expected}"


@pytest.mark.slow
def test_coupling_lag_recovers_D_exactly(latent_fixture) -> None:
    r"""Joint regression recovers the true lag $D$ **exactly**, on every signal cell.

    The S7-T01 criterion says a raw cross-correlation of $c$ against $d$ "peaks at exactly the
    cell's ``delay``". It does not: since $d_k = A_y d_{k-1} + B c_{k-D} + \varepsilon_k$, the AR(1)
    target smears the source response over $\ell \ge D$ with geometric weights, and the measured
    peak lands on $D$ or $D+1$ depending on the cell. Partialling out $d_{k-1}$ and the rest of the
    source lag window (:func:`_joint_regression_lag`) restores an exact recovery -- and it is the
    same reason the model's ``te_lag_map`` spreads mass over $\ell \ge D$ rather than spiking at
    $D$.

    Null cells ($B = 0$) have no lag to recover and are excluded.
    """
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    fx = latent_fixture
    npz, meta = fx["npz"], fx["meta"]
    cid = np.asarray(npz["sample_cell_id"])
    ridx = np.asarray(npz["sample_raw_index"])
    delay = np.asarray(npz["sample_delay"])
    lmax = 30

    signal = [int(c["cell_id"]) for c in meta["cells"] if float(c["te_block_realised"]) > 0.0]
    assert len(signal) >= 8, f"expected >= 8 signal cells, got {len(signal)}"

    for cell_id in signal:
        rows = np.where(cid == cell_id)[0][:64]
        D = int(delay[rows[0]])
        pairs = [fx["provider"](cell_id, int(ridx[i])) for i in rows]
        peak, coefs = _joint_regression_lag([p[0] for p in pairs], [p[1] for p in pairs], lmax)
        assert peak == D, (
            f"cell {cell_id}: joint-regression lag {peak} != true D {D} "
            f"(beta[D]={coefs[D]:.4f})"
        )


@pytest.mark.slow
def test_latent_provider_writes_nothing(latent_fixture) -> None:
    r"""No cache file is created or modified. The ``.npz`` schema stays frozen."""
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        make_latent_provider,
    )

    cache_dir = latent_fixture["cache_dir"]
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in cache_dir.iterdir() if p.is_file()}
    provider = make_latent_provider(latent_fixture["config"], "test", cache_dir=cache_dir)
    provider(0, 0)
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
             for p in cache_dir.iterdir() if p.is_file()}
    assert before == after


@pytest.mark.slow
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_latent_provider_works_for_every_split(split: str) -> None:
    r"""All three splits resolve, and their seed offsets give *different* latents."""
    cache_dir = find_cache_dir()
    if cache_dir is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        make_latent_provider,
    )

    config = _load_config()
    provider = make_latent_provider(config, split, cache_dir=cache_dir)
    npz = np.load(cache_dir / f"{split}.npz", mmap_mode="r")
    cid = np.asarray(npz["sample_cell_id"])
    ridx = np.asarray(npz["sample_raw_index"])
    c, d = provider(int(cid[0]), int(ridx[0]))
    assert np.isfinite(c).all() and np.isfinite(d).all()
    assert c.shape == d.shape


def test_latent_provider_is_total_on_a_bad_row(tmp_path) -> None:
    r"""An unknown cell / out-of-range row yields a NaN window rather than a raise."""
    cache_dir = find_cache_dir()
    if cache_dir is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        make_latent_provider,
    )

    provider = make_latent_provider(_load_config(), "val", cache_dir=cache_dir)
    c, d = provider(9_999, 0)
    assert c.shape == (provider.window_length,)
    assert np.isnan(c).all() and np.isnan(d).all()


def test_latent_provider_raises_without_a_cache(tmp_path) -> None:
    r"""Construction fails loudly on a missing ``meta.json``; the stage catches and skips."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        make_latent_provider,
    )

    with pytest.raises(FileNotFoundError):
        make_latent_provider({}, "val", cache_dir=tmp_path)


# ===========================================================================
# S7-T02: the InfoNCE critic and the residualised estimator
# ===========================================================================
def _fast_cfg(**over):
    r"""A small, deterministic estimator config at the shipped capacity, on a short budget."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import CMIConfig

    base = dict(critic_width=64, critic_depth=1, embed_dim=16, contrastive_batch=32,
                n_iters=1500, eval_every=50, patience=10, lr=1e-3, fit_frac=0.5,
                n_boot=200, seed=0, device="cpu")
    base.update(over)
    return CMIConfig(**base)


#: The recovery draw pinned by S7-T03. `B = 0.15` gives an analytic block TE of 0.8234 nats.
_REC = dict(D=3, horizon=5, k_history=10, n=200, t_tot=120)


def test_infonce_bound_matches_the_closed_form_and_respects_log_k() -> None:
    r"""On fixed logits the bound equals its definition, and $\hat I \le \log K$ always."""
    import torch

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        infonce_lower_bound,
    )

    torch.manual_seed(0)
    k = 8
    scores = torch.randn(k, k, dtype=torch.float64)
    got = float(infonce_lower_bound(scores))
    expected = float(
        (scores.diagonal() - (torch.logsumexp(scores, dim=0) - math.log(k))).mean()
    )
    assert got == pytest.approx(expected, abs=1e-12)
    assert got <= math.log(k) + 1e-9

    # A perfectly-separating critic saturates exactly at log K; a blind one sits at 0.
    saturated = torch.full((k, k), -50.0, dtype=torch.float64)
    saturated.fill_diagonal_(50.0)
    assert float(infonce_lower_bound(saturated)) == pytest.approx(math.log(k), abs=1e-6)
    assert float(infonce_lower_bound(torch.zeros(k, k, dtype=torch.float64))) == pytest.approx(0.0)


def test_estimator_exposes_the_ceiling_and_the_near_ceiling_rule() -> None:
    r"""$\log 32 = 3.47$ nats brackets the $\mathrm{TE}_{\mathrm{inj}} = 3.0$ cell, and only it."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        NeuralCMIEstimator,
    )

    est = NeuralCMIEstimator(4, 4, 4, _fast_cfg())
    assert est.ceiling_nats == pytest.approx(math.log(32))
    assert est.ceiling_nats == pytest.approx(3.4657, abs=1e-3)
    # ceiling_claim_frac = 0.8 -> threshold 2.7726 nats.
    assert est.near_ceiling(3.0) is True
    assert est.near_ceiling(2.0) is False
    assert est.near_ceiling(0.5) is False
    assert est.near_ceiling(float("nan")) is False


def test_estimator_gradients_flow_and_the_bound_improves() -> None:
    r"""Fitting raises the InfoNCE trace on a linearly dependent pair."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        NeuralCMIEstimator,
    )

    rng = np.random.default_rng(0)
    n_groups, per_group = 40, 20
    groups = np.repeat(np.arange(n_groups), per_group)
    u = rng.normal(size=(n_groups * per_group, 3))
    v = u @ rng.normal(size=(3, 3)) + 0.3 * rng.normal(size=(n_groups * per_group, 3))
    c = rng.normal(size=(n_groups * per_group, 2))  # uninformative conditioning

    est = NeuralCMIEstimator(3, 3, 2, _fast_cfg(n_iters=300))
    out = est.fit_estimate(u, v, c, groups)
    assert 0 < len(est.trace) <= 300
    assert np.mean(est.trace[-30:]) > np.mean(est.trace[:30])
    assert out["converged"] is True
    assert out["estimate"] > 0.5
    assert out["estimate"] <= out["ceiling_nats"] + 1e-6


def test_early_stopping_restores_the_best_validation_critic() -> None:
    r"""The fit stops on a validation slice carved from the fit samples, and reports where.

    Without this, an InfoNCE critic run to convergence on a few thousand correlated anchors
    memorises the pairing and its held-out bound collapses -- measured at $-22$ nats on a
    $\mathrm{TE}_{\mathrm{inj}} = 0$ cell after 3000 unstopped steps. Early stopping is what makes
    the null cells read $\approx 0$ instead.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        NeuralCMIEstimator,
    )

    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(24), 24)
    n = 24 * 24
    # Pure noise: nothing to learn, so the validation score peaks early and the fit stops.
    u, v, c = rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, 2))

    est = NeuralCMIEstimator(3, 3, 2, _fast_cfg(n_iters=4000, patience=5))
    out = est.fit_estimate(u, v, c, groups)
    assert out["stopped_iter"] < 4000, "patience must terminate a hopeless fit"
    assert 0 < out["best_iter"] <= out["stopped_iter"]
    assert len(est.val_trace) >= 1
    # An honest estimator reports ~0 where there is no information, not a large negative bound.
    assert abs(out["estimate"]) < 0.2
    # The three partitions are disjoint by construction.
    assert out["n_groups_fit"] + out["n_groups_val"] + out["n_groups_eval"] == 24


def test_estimator_is_deterministic_under_a_fixed_seed() -> None:
    r"""Two runs at one seed agree bitwise, estimate and interval alike."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        NeuralCMIEstimator,
    )

    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(30), 16)
    u = rng.normal(size=(480, 3))
    v = 0.8 * u + 0.5 * rng.normal(size=(480, 3))
    c = rng.normal(size=(480, 2))

    a = NeuralCMIEstimator(3, 3, 2, _fast_cfg(n_iters=200)).fit_estimate(u, v, c, groups)
    b = NeuralCMIEstimator(3, 3, 2, _fast_cfg(n_iters=200)).fit_estimate(u, v, c, groups)
    assert a["estimate"] == b["estimate"]
    assert (a["ci_lo"], a["ci_hi"]) == (b["ci_lo"], b["ci_hi"])


def test_estimator_never_lets_a_sample_straddle_fit_and_eval() -> None:
    r"""The fit/eval split is by *sample*, so an anchor's siblings never leak across it."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        NeuralCMIEstimator,
    )

    groups = np.repeat(np.arange(10), 40)
    rng = np.random.default_rng(2)
    fit_idx, eval_idx = NeuralCMIEstimator._split_by_group(groups, 0.5, rng)
    assert set(groups[fit_idx]).isdisjoint(set(groups[eval_idx]))
    assert len(fit_idx) + len(eval_idx) == len(groups)


def test_estimator_rejects_too_few_samples() -> None:
    r"""Three partitions -- fit, early-stopping validation, held-out estimate -- need 3 samples."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        NeuralCMIEstimator,
    )

    z = np.zeros((64, 2))
    with pytest.raises(ValueError, match="3 distinct samples"):
        NeuralCMIEstimator(2, 2, 2, _fast_cfg()).fit_estimate(z, z, z, np.zeros(64, dtype=int))
    with pytest.raises(ValueError, match="share a length"):
        NeuralCMIEstimator(2, 2, 2, _fast_cfg()).fit_estimate(
            z, z[:10], z, np.arange(64) % 4
        )


def test_gaussian_closed_form_recovers_a_known_partial_information() -> None:
    r"""The closed form matches the analytic partial information of a Gaussian triple."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cmi_v3 import (
        gaussian_cmi_closed_form,
    )

    rng = np.random.default_rng(3)
    n = 200_000
    c = rng.normal(size=(n, 1))
    e_u, e_v = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    rho = 0.6
    u = c + e_u
    v = c + rho * e_u + math.sqrt(1 - rho**2) * e_v
    # Conditional on c, corr(u, v) = rho exactly, so I(U;V|C) = -0.5*log(1 - rho^2).
    expected = -0.5 * math.log(1 - rho**2)
    assert gaussian_cmi_closed_form(u, v, c) == pytest.approx(expected, rel=0.02)


# ===========================================================================
# S7-T03: synthetic recovery, with the thresholds pinned as named constants
# ===========================================================================
def test_recovery_is_positive_on_dependent_draws() -> None:
    r"""A coupled Gaussian pair yields an estimate above ``_RECOVER_MIN_DEPENDENT``."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    out = cmi_v3.recover_synthetic(B=0.15, cfg=_fast_cfg(), seed=0, **_REC)
    assert out["te_analytic"] > 0.5, "the draw must actually carry TE"
    assert out["estimate"] > cmi_v3._RECOVER_MIN_DEPENDENT
    assert out["estimate"] <= out["ceiling_nats"] + 1e-6


def test_recovery_is_near_zero_on_independent_draws() -> None:
    r"""With $B = 0$ the source is disconnected; the estimate sits inside the null band.

    It is not asserted **non-negative**: :math:`\hat I_{\mathrm{NCE}}` on residualised finite data
    fluctuates about zero and typically lands slightly below it.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    out = cmi_v3.recover_synthetic(B=0.0, cfg=_fast_cfg(), seed=0, **_REC)
    assert out["te_analytic"] < 1e-3
    assert abs(out["estimate"]) < cmi_v3._RECOVER_MAX_INDEPENDENT


def test_recovery_lands_within_a_factor_of_two_of_the_analytic_te() -> None:
    r"""Below the ceiling the InfoNCE bound is loose but not by more than ``_RECOVER_FACTOR``."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    out = cmi_v3.recover_synthetic(B=0.15, cfg=_fast_cfg(), seed=0, **_REC)
    te = out["te_analytic"]
    assert te <= cmi_v3._RECOVER_TE_MAX_FOR_ABSOLUTE, "absolute claims only below this TE"
    ratio = out["estimate"] / te
    assert 1.0 / cmi_v3._RECOVER_FACTOR <= ratio <= cmi_v3._RECOVER_FACTOR


def test_the_anchor_windowing_reproduces_the_block_te_estimand() -> None:
    r"""The exact closed form on the estimator's own anchors matches the independent block TE.

    This is the check that :func:`cmi_v3.latent_anchor_rows` slices the *right* windows. The two
    references are computed by completely different routes -- a residual-covariance determinant on
    the anchors, and a Monte-Carlo determinant ratio over fresh simulations -- so agreement is
    evidence about the windowing, not a tautology.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    out = cmi_v3.recover_synthetic(B=0.15, cfg=_fast_cfg(n_iters=1), seed=0, **_REC)
    rel_err = abs(out["cmi_gauss_exact"] - out["te_analytic"]) / out["te_analytic"]
    assert rel_err < 0.05, f"closed form {out['cmi_gauss_exact']} vs MC {out['te_analytic']}"


@pytest.mark.slow
def test_recovery_is_monotone_across_the_coupling_ladder() -> None:
    r"""The estimate rises with the coupling, and the bound loosens toward the ceiling.

    Measured (``n_iters = 600``, $n = 300$): ratio to the analytic TE falls $0.89 \to 0.75$ as the
    TE climbs $0.82 \to 2.56$ nats against a $\log 32 = 3.47$ ceiling -- the signature of an
    InfoNCE bound, not of a broken estimator.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    cfg = _fast_cfg(n_iters=600)
    runs = [cmi_v3.recover_synthetic(B=b, cfg=cfg, seed=0, D=3, horizon=5, k_history=10,
                                     n=300, t_tot=120)
            for b in (0.0, 0.15, 0.5, 1.2)]
    estimates = [r["estimate"] for r in runs]
    assert estimates == sorted(estimates), f"not monotone in the coupling: {estimates}"
    for r in runs:
        assert r["estimate"] <= r["ceiling_nats"] + 1e-6
    # The bound loosens as the target approaches the ceiling.
    ratios = [r["estimate"] / r["te_analytic"] for r in runs[1:]]
    assert ratios[0] > ratios[-1], f"expected the bound to loosen toward the ceiling: {ratios}"


@pytest.mark.slow
def test_recovery_rank_correlates_with_te_on_the_real_cells(latent_fixture) -> None:
    r"""Under ground-truth (``latent``) conditioning, the CMI ranks the real cells by their TE.

    Runs the estimator model-free on the cached cells' regenerated latents, so this holds
    regardless of how well any checkpoint was trained. Asserts its own $N$ rather than assuming it.
    """
    if latent_fixture is None:
        pytest.skip("no synthetic_v2 cache built; run --stage build first")
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
        _spearman_finite,
    )

    fx = latent_fixture
    npz, meta = fx["npz"], fx["meta"]
    cid = np.asarray(npz["sample_cell_id"])
    ridx = np.asarray(npz["sample_raw_index"])
    horizon = int(meta["horizon"])
    k_hist = 24  # >= max D = 20; see the S7-T03 lookback resolution in Section 11
    cfg = _fast_cfg(n_iters=3000, patience=12)

    te_inj, cmi, gauss = [], [], []
    for spec in meta["cells"]:
        cell_id = int(spec["cell_id"])
        rows = np.where(cid == cell_id)[0]
        anchors = list(cmi_v3.valid_anchor_range(int(meta["sequence_length"]),
                                                 k_hist, k_hist, horizon))[::3]
        u_rows, v_rows, c_rows, groups = [], [], [], []
        for g, i in enumerate(rows):
            c_lat, d_lat = fx["provider"](cell_id, int(ridx[i]))
            u, v, cond = cmi_v3.latent_anchor_rows(
                c_lat, d_lat, anchors, u_lookback=k_hist, c_lookback=k_hist, horizon=horizon
            )
            u_rows.append(u)
            v_rows.append(v)
            c_rows.append(cond)
            groups.append(np.full(len(anchors), g))
        u_all, v_all, c_all = map(np.concatenate, (u_rows, v_rows, c_rows))
        est = cmi_v3.NeuralCMIEstimator(k_hist, horizon, k_hist, cfg)
        out = est.fit_estimate(u_all, v_all, c_all, np.concatenate(groups))
        te_inj.append(float(spec["te_block_realised"]))
        cmi.append(out["estimate"])
        gauss.append(cmi_v3.gaussian_cmi_closed_form(u_all, v_all, c_all))

    te_a, cmi_a = np.asarray(te_inj), np.asarray(cmi)
    n_cells = len(te_a)
    assert n_cells >= 8, f"expected >= 8 cells for a rank correlation, got {n_cells}"
    rho = _spearman_finite(cmi_a, te_a)
    assert rho is not None and rho > cmi_v3._RECOVER_MIN_SPEARMAN, f"rho={rho} over {n_cells} cells"

    # The exact closed form on the same anchors reproduces the injected TE: the lookback of 24
    # spans the coupling support (max D = 20) without accruing the log-det bias a longer one does.
    # The residual +2..5% is that bias, and it shrinks with the anchor count.
    signal = te_a > 0
    np.testing.assert_allclose(np.asarray(gauss)[signal], te_a[signal], rtol=0.08)

    # Null cells: no source, so no information -- and no large negative bound either.
    assert np.all(np.abs(cmi_a[~signal]) < cmi_v3._RECOVER_MAX_INDEPENDENT)

    # Below the ceiling the bound is loose but stays within a factor of two.
    low = signal & (te_a <= cmi_v3._RECOVER_TE_MAX_FOR_ABSOLUTE)
    ratios = cmi_a[low] / te_a[low]
    assert np.all(ratios >= 1.0 / cmi_v3._RECOVER_FACTOR), f"ratios={np.round(ratios, 3)}"
    assert np.all(ratios <= cmi_v3._RECOVER_FACTOR)

    # The TE = 3.0 cells sit above 0.8 * log(32); no absolute-nats claim is made for them.
    est = cmi_v3.NeuralCMIEstimator(1, 1, 1, cfg)
    assert any(est.near_ceiling(t) for t in te_a), "the ceiling check must actually fire"


# ===========================================================================
# S7-T04a/b: the comparison, the bias, and the stage's graceful degradation
# ===========================================================================
#: Horizon of the synthetic anchor pools below. Only sizes the estimator; nothing depends on it.
_HORIZON = 5


def _fake_cells(n_cells: int = 4, n_rows: int = 12, n_anchors: int = 40, seed: int = 0):
    r"""Per-cell anchor pools with a coupling strength that grows with the cell's TE."""
    rng = np.random.default_rng(seed)
    cells = {}
    for cid in range(n_cells):
        te = float(cid) * 0.5
        n = n_rows * n_anchors
        groups = np.repeat(np.arange(n_rows), n_anchors)
        c = rng.normal(size=(n, 3))
        u = c[:, :2] @ rng.normal(size=(2, 4)) + rng.normal(size=(n, 4))
        noise = rng.normal(size=(n, 3))
        v = te * (u[:, :3]) + c @ rng.normal(size=(3, 3)) + noise
        entry = {"cell_id": cid, "n_rows": n_rows, "n_anchors": n, "te_inj": te,
                 "te_scat": te * 2.0, "delay": 8, "kbar": 0.1 * te, "groups": groups}
        for name in ("latent", "feature_gt", "feature_model"):
            entry[name] = {"u": u.copy(), "v": v.copy(), "c": c.copy()}
        cells[cid] = entry
    return cells


def _cmp_cfg(**over):
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    cfg = dict(cmi_v3._DEFAULT_CMI)
    cfg.update(critic_width=32, critic_depth=1, embed_dim=8, n_iters=200, eval_every=25,
               patience=4, n_boot=100, min_cells_for_rho=8)
    cfg.update(over)
    return cfg


def test_comparison_emits_the_documented_schema() -> None:
    r"""``cmi.json``'s top-level and per-cell keys, and the estimator provenance block."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    cfg = _cmp_cfg()
    configs = list(cmi_v3._ALL_CONFIGS)
    summary = cmi_v3.run_cmi_comparison(_fake_cells(), cfg=cfg, configs=configs,
                                        horizon=_HORIZON, seed=0)
    for key in ("per_cell", "overall", "recovery", "ceiling_nats", "ceiling_claim_frac",
                "configs", "estimator", "horizon"):
        assert key in summary, key
    assert summary["ceiling_nats"] == pytest.approx(math.log(cfg["contrastive_batch"]))

    cell = summary["per_cell"]["1"]
    for key in ("cell_id", "n_rows", "n_anchors", "te_inj", "te_scat", "delay", "kbar",
                "cmi_latent", "cmi_feature_gt", "cmi_feature_model", "bias",
                "cmi_latent_gauss_exact"):
        assert key in cell, key
    for key in ("estimate", "ci_lo", "ci_hi", "ceiling_nats", "near_ceiling", "converged",
                "cond_r2_u", "cond_r2_v", "best_iter", "stopped_iter"):
        assert key in cell["cmi_latent"], key

    overall = summary["overall"]
    for name in configs:
        entry = overall[f"rho_kbar_cmi_{name}"]
        assert set(entry) == {"rho", "ci", "n_cells", "gated"}
        assert entry["n_cells"] == 4          # its N is reported, not assumed
    assert overall["rho_reported_not_gated"] is True   # 4 cells < min_cells_for_rho = 8
    assert "rho_cmi_te_inj_latent" in overall


def test_bias_is_feature_model_minus_feature_gt() -> None:
    r"""``bias`` is exactly the difference of the two feature-config estimates."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    summary = cmi_v3.run_cmi_comparison(
        _fake_cells(), cfg=_cmp_cfg(), configs=list(cmi_v3._ALL_CONFIGS),
        horizon=_HORIZON, seed=0,
    )
    for cell in summary["per_cell"].values():
        expected = (cell["cmi_feature_model"]["estimate"]
                    - cell["cmi_feature_gt"]["estimate"])
        assert cell["bias"]["estimate"] == pytest.approx(expected, abs=1e-12)
        assert cell["bias"]["ci_lo"] <= cell["bias"]["estimate"] <= cell["bias"]["ci_hi"]
    bias = summary["overall"]["cmi_bias"]
    assert bias["n_cells"] == 4
    assert bias["ci_lo"] <= bias["estimate"] <= bias["ci_hi"]


def test_bias_is_flagged_unreliable_on_a_non_transferable_conditioning() -> None:
    r"""A negative held-out ``cond_r2_v`` invalidates the bias, and says so.

    Measured on the pilot, the separation is total: the two ``causal_norm: false`` arms have a
    negative ``cond_r2_v`` on **15 of 15** cells, ``v3_prod`` a positive one on **15 of 15**.
    A time-pooling ``GroupNorm`` makes ``target_state[b, t]`` carry per-sample statistics that a
    regression fitted on the fit samples cannot transfer, so their ``bias`` of $-0.66$ / $-0.54$
    is an artefact rather than a measurement of a worse summary.

    ``cond_r2_u`` is *not* gated on: on a $\mathrm{TE}_{\mathrm{inj}} = 0$ cell the target's past
    genuinely cannot predict the disconnected source, so a small negative value there is correct.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    good = {"cond_r2_u": 0.19, "cond_r2_v": 0.07}
    assert cmi_v3._bias_reliability(good) == {"reliable": True, "reason": None}

    # A null cell on a causal arm: cond_r2_u < 0 is expected, and must NOT flip the flag.
    null_cell = {"cond_r2_u": -0.053, "cond_r2_v": 0.013}
    assert cmi_v3._bias_reliability(null_cell)["reliable"] is True

    leaky = {"cond_r2_u": -0.18, "cond_r2_v": -0.32}
    verdict = cmi_v3._bias_reliability(leaky)
    assert verdict["reliable"] is False
    assert "cond_r2_v" in verdict["reason"] and "generalise" in verdict["reason"]

    # ..and it reaches `overall`, so `arms_report` can gate on it.
    summary = cmi_v3.run_cmi_comparison(
        _fake_cells(), cfg=_cmp_cfg(), configs=list(cmi_v3._ALL_CONFIGS),
        horizon=_HORIZON, seed=0,
    )
    bias = summary["overall"]["cmi_bias"]
    assert set(bias) >= {"estimate", "ci_lo", "ci_hi", "n_cells", "reliable",
                         "n_cells_unreliable", "reason"}
    assert "cond_r2_feature_model" in summary["overall"]
    assert set(summary["overall"]["cond_r2_feature_model"]) == {"u", "v"}


def test_comparison_is_seeded_and_reproducible() -> None:
    r"""Two runs at one seed give identical estimates and identical bootstrap intervals."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    kwargs = dict(cfg=_cmp_cfg(), configs=[cmi_v3.CONFIG_LATENT], horizon=_HORIZON, seed=3)
    a = cmi_v3.run_cmi_comparison(_fake_cells(), **kwargs)
    b = cmi_v3.run_cmi_comparison(_fake_cells(), **kwargs)
    for cid in a["per_cell"]:
        pa, pb = a["per_cell"][cid]["cmi_latent"], b["per_cell"][cid]["cmi_latent"]
        assert (pa["estimate"], pa["ci_lo"], pa["ci_hi"]) == (pb["estimate"], pb["ci_lo"],
                                                              pb["ci_hi"])
    assert a["overall"]["rho_kbar_cmi_latent"]["ci"] == b["overall"]["rho_kbar_cmi_latent"]["ci"]
    assert cmi_v3._N_BOOT_CMI == 2000  # the named constant the default config carries


def test_comparison_drops_latent_configs_when_the_pool_is_absent() -> None:
    r"""Without the latent provider the two GT configs vanish; ``feature_model`` survives."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    cells = _fake_cells()
    for entry in cells.values():
        entry.pop("latent")
        entry.pop("feature_gt")
    summary = cmi_v3.run_cmi_comparison(
        cells, cfg=_cmp_cfg(), configs=[cmi_v3.CONFIG_FEATURE_MODEL], horizon=_HORIZON, seed=0,
    )
    cell = summary["per_cell"]["1"]
    assert "cmi_feature_model" in cell
    assert "cmi_latent" not in cell and "bias" not in cell
    assert summary["recovery"] == {"available": False}
    assert summary["overall"]["cmi_bias"] is None


def test_comparison_degrades_on_an_empty_cell_set() -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    out = cmi_v3.run_cmi_comparison({}, cfg=_cmp_cfg(), configs=["latent"], horizon=5)
    assert "error" in out


_T, _C_Y, _C_U, _D_H = 60, 4, 3, 6


class _StubModel:
    r"""Emits the three keys the collector reads: ``target_state``, ``kld_per_t``, ``mu_full``."""

    def __init__(self, t: int, d_h: int) -> None:
        self.t, self.d_h = t, d_h

    def eval(self):
        return self

    def __call__(self, y_st, y_ph, u_stream):
        import torch

        b = y_st.shape[0]
        g = torch.Generator().manual_seed(0)
        return {
            "target_state": torch.randn(b, self.t, self.d_h, generator=g),
            "kld_per_t": torch.ones(b, self.t),
        }


class _StubRunner:
    def __init__(self, batches, model, warmup, horizon) -> None:
        self._batches, self.model = batches, model
        self.warmup_steps, self.horizon = warmup, horizon

    def inference_mode(self):
        import contextlib

        return contextlib.nullcontext()

    def iter_batches(self, loader, max_samples=None):
        seen = 0
        for b in self._batches:
            if max_samples is not None and seen >= max_samples:
                break
            yield b
            seen += int(b.fhr_st.shape[0])

    def build_future_target(self, batch):
        return batch.y_plus


def _stub_batch(n: int, horizon: int):
    import torch
    import types

    g = torch.Generator().manual_seed(1)
    return types.SimpleNamespace(
        fhr_st=torch.zeros(n, _T, 2), fhr_ph=torch.zeros(n, _T, 2),
        up_st=torch.randn(n, _T, 2, generator=g), up_ph=torch.randn(n, _T, 1, generator=g),
        y_plus=torch.randn(n, _T - horizon, horizon, _C_Y, generator=g),
        delay=torch.full((n,), 8, dtype=torch.long),
        te_true=torch.full((n,), 2.0),
        te_scat=torch.full((n,), 5.0),
        cell_id=torch.zeros(n, dtype=torch.long),
        raw_index=torch.arange(n, dtype=torch.long),
    )


def test_collector_builds_clean_window_anchors_for_every_config() -> None:
    r"""Anchors sit inside ``eval``'s clean window, and every config gets matching pools.

    ``rho(kbar, CMI)`` only compares like with like if the CMI's anchors are the ones $\bar K$ was
    averaged over -- so the collector must reuse ``eval_v2._clean_window_mean``'s mask, not invent
    its own window.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    horizon, warmup = 5, 4
    cfg = _cmp_cfg(latent_lookback=6, u_lookback=6, c_lookback=6, anchor_stride=1,
                   u_channels="all", y_channels="all", max_anchors_per_cell=10_000)
    meta = {"coupled_channel": {"up_st": 0, "fhr_st": 0}}
    batches = [_stub_batch(4, horizon), _stub_batch(4, horizon)]
    runner = _StubRunner(batches, _StubModel(_T, _D_H), warmup, horizon)

    rng = np.random.default_rng(0)

    def latent_provider(cell_id, raw_index):
        return rng.normal(size=_T), rng.normal(size=_T)

    cells, n_seen = cmi_v3.collect_cmi_anchors(
        runner.model, runner, None, latent_provider,
        cfg=cfg, configs=list(cmi_v3._ALL_CONFIGS), meta=meta, seed=0,
    )
    assert n_seen == 8
    assert set(cells) == {0}
    entry = cells[0]
    assert entry["n_rows"] == 8
    assert entry["te_inj"] == 2.0 and entry["delay"] == 8 and entry["kbar"] == 1.0

    # Anchors: inside [max(warmup, D-1), T-H) AND with a full lookback. delay=8 -> lo = max(5, 7) = 7.
    n_anchors_per_row = entry["n_anchors"] // entry["n_rows"]
    lo = max(warmup, 8 - 1, 6 - 1)
    assert n_anchors_per_row == len(range(lo, _T - horizon))

    for name in cmi_v3._ALL_CONFIGS:
        pool = entry[name]
        assert pool["u"].shape[0] == pool["v"].shape[0] == pool["c"].shape[0] == entry["n_anchors"]
    # latent: (Lu,), (H,), (Lc,) -- the raw DGP coordinates.
    assert entry["latent"]["u"].shape[1] == 6
    assert entry["latent"]["v"].shape[1] == horizon
    # feature: flattened (Lu x C_u) and (H x C_y); the two feature configs share U and Y+ exactly.
    assert entry["feature_gt"]["u"].shape[1] == 6 * _C_U
    assert entry["feature_gt"]["v"].shape[1] == horizon * _C_Y
    np.testing.assert_array_equal(entry["feature_gt"]["u"], entry["feature_model"]["u"])
    np.testing.assert_array_equal(entry["feature_gt"]["v"], entry["feature_model"]["v"])
    # ..and differ only in the conditioning: GT latent history vs the model's target_state.
    assert entry["feature_gt"]["c"].shape[1] == 6
    assert entry["feature_model"]["c"].shape[1] == _D_H

    # Groups index the SAMPLE, so no sample straddles the estimator's fit/eval split.
    assert len(np.unique(entry["groups"])) == 8


def test_collector_refuses_to_guess_a_missing_raw_index() -> None:
    r"""Without ``raw_index`` the latents cannot be matched to their cached rows.

    Defaulting to zeros would regenerate cell ``cid``'s *first* row for every sample -- a silent,
    plausible-looking wrong answer, and exactly the failure mode this pipeline exists to catch.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    horizon, warmup = 5, 4
    batch = _stub_batch(4, horizon)
    del batch.raw_index
    runner = _StubRunner([batch], _StubModel(_T, _D_H), warmup, horizon)
    cfg = _cmp_cfg(latent_lookback=6, u_lookback=6, c_lookback=6, anchor_stride=1)

    with pytest.raises(KeyError, match="raw_index"):
        cmi_v3.collect_cmi_anchors(
            runner.model, runner, None, lambda c, r: (np.zeros(_T), np.zeros(_T)),
            cfg=cfg, configs=[cmi_v3.CONFIG_LATENT], meta={}, seed=0,
        )

    # ..but a feature-only run never needs it, and proceeds.
    cells, _ = cmi_v3.collect_cmi_anchors(
        runner.model, runner, None, None,
        cfg=dict(cfg, u_channels="all", y_channels="all"),
        configs=[cmi_v3.CONFIG_FEATURE_MODEL], meta={}, seed=0,
    )
    assert cells[0]["n_rows"] == 4


def test_collector_skips_a_row_whose_latents_are_nan() -> None:
    r"""A NaN window from the total provider drops that row, keeping every column aligned."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    horizon, warmup = 5, 4
    runner = _StubRunner([_stub_batch(8, horizon)], _StubModel(_T, _D_H), warmup, horizon)
    cfg = _cmp_cfg(latent_lookback=6, u_lookback=6, c_lookback=6, anchor_stride=1,
                   u_channels="all", y_channels="all")

    def flaky(cell_id, raw_index):
        if raw_index in (1, 2):
            return np.full(_T, np.nan), np.full(_T, np.nan)
        return np.arange(_T, dtype=float), np.arange(_T, dtype=float)

    cells, n_seen = cmi_v3.collect_cmi_anchors(
        runner.model, runner, None, flaky,
        cfg=cfg, configs=list(cmi_v3._ALL_CONFIGS), meta={"coupled_channel": {}}, seed=0,
    )
    assert n_seen == 8                       # consumed 8
    entry = cells[0]
    assert entry["n_rows"] == 6              # kept 6; rows 1 and 2 had NaN latents
    for name in cmi_v3._ALL_CONFIGS:
        assert entry[name]["u"].shape[0] == entry["n_anchors"]
    assert len(entry["groups"]) == entry["n_anchors"]
    assert len(np.unique(entry["groups"])) == 6


def test_collector_drops_a_cell_with_too_few_usable_rows() -> None:
    r"""Fewer than three samples cannot be split into fit / validation / held-out partitions."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    horizon, warmup = 5, 4
    runner = _StubRunner([_stub_batch(4, horizon)], _StubModel(_T, _D_H), warmup, horizon)
    cfg = _cmp_cfg(latent_lookback=6, u_lookback=6, c_lookback=6, anchor_stride=1,
                   u_channels="all", y_channels="all")

    def mostly_nan(cell_id, raw_index):
        if raw_index >= 2:
            return np.full(_T, np.nan), np.full(_T, np.nan)
        return np.arange(_T, dtype=float), np.arange(_T, dtype=float)

    cells, n_seen = cmi_v3.collect_cmi_anchors(
        runner.model, runner, None, mostly_nan,
        cfg=cfg, configs=list(cmi_v3._ALL_CONFIGS), meta={"coupled_channel": {}}, seed=0,
    )
    assert n_seen == 4
    assert cells == {}, "a 2-row cell must be dropped, not fitted"


def test_collector_refuses_a_latent_config_without_a_provider() -> None:
    r"""The stage drops the latent configs when the provider fails; a direct caller must too.

    Without this the collector would die on ``None(cell_id, raw_index)`` deep inside the batch loop,
    long after the useful context is gone.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    for configs in ([cmi_v3.CONFIG_LATENT], [cmi_v3.CONFIG_FEATURE_GT],
                    list(cmi_v3._ALL_CONFIGS)):
        with pytest.raises(ValueError, match="ground-truth latent provider"):
            cmi_v3.collect_cmi_anchors(
                None, None, None, None, cfg=_cmp_cfg(), configs=configs, meta={},
            )


def test_recovery_block_flags_near_ceiling_cells() -> None:
    r"""Cells above ``ceiling_claim_frac * log K`` are excluded from the absolute-nats claim."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    # te_inj runs 0.0, 0.5, ... ; push one cell over 0.8 * log(32) = 2.77 nats.
    cells = _fake_cells(n_cells=8)
    summary = cmi_v3.run_cmi_comparison(
        cells, cfg=_cmp_cfg(), configs=[cmi_v3.CONFIG_LATENT], horizon=_HORIZON, seed=0,
    )
    rec = summary["recovery"]
    assert rec["available"] is True
    assert rec["n_cells"] == 8
    # te_inj = 3.0 and 3.5 exceed 2.77; te_inj = 0 is null; 5 cells carry an absolute claim.
    assert rec["n_absolute_claim_cells"] == 5
    assert set(rec["near_ceiling_cells"]) == {6, 7}
    assert rec["max_abs_null_cmi"] is not None


def test_channel_index_reduction() -> None:
    r"""``coupled`` selects the single injected channel; ``all`` keeps the stream."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    meta = {"coupled_channel": {"up_st": 20, "fhr_st": 20}}
    assert cmi_v3._channel_index(meta, "all", 101, "up_st").tolist() == list(range(101))
    assert cmi_v3._channel_index(meta, "coupled", 101, "up_st").tolist() == [20]
    assert cmi_v3._channel_index(meta, "coupled", 87, "fhr_st").tolist() == [20]
    with pytest.raises(ValueError, match="all.*coupled"):
        cmi_v3._channel_index(meta, "pca", 101, "up_st")


def test_cmi_cfg_merges_over_the_documented_defaults() -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    cfg = cmi_v3._cmi_cfg({}, "G1_raw")
    assert cfg == cmi_v3._DEFAULT_CMI
    over = {"benchmarks": {"G1_raw": {"eval": {"cmi": {"n_iters": 7, "u_channels": "all"}}}}}
    cfg = cmi_v3._cmi_cfg(over, "G1_raw")
    assert cfg["n_iters"] == 7 and cfg["u_channels"] == "all"
    assert cfg["critic_width"] == cmi_v3._DEFAULT_CMI["critic_width"]
    bad = {"benchmarks": {"G1_raw": {"eval": {"cmi": {"configs": ["latent", "nope"]}}}}}
    with pytest.raises(ValueError, match="unknown entries"):
        cmi_v3._cmi_cfg(bad, "G1_raw")


def test_stage_and_section_are_registered_correctly() -> None:
    r"""``cmi`` is opt-in, arm-scoped, and non-fatal; the section slots in after Sprint 6's."""
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import run_pipeline_v2 as rp

    rp._load_stage_plugins()
    spec = rp._STAGE_REGISTRY["cmi"]
    assert spec.run is cmi_v3.run_cmi_stage
    assert spec.model_dependent is True
    assert spec.fatal is False
    assert spec.default_on is False        # fits one critic per (cell x config)
    assert "cmi" in rp.stage_names()

    names = [s.name for s in fr._SECTION_REGISTRY]
    assert "Neural CMI" in names
    orders = {s.name: s.order for s in fr._SECTION_REGISTRY}
    assert orders["Neural CMI"] > orders["Interventional lag attribution"]


def test_cmi_section_renders_na_without_the_json(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    ctx = fr.SectionContext(config={}, benchmark="G1_raw", results_dir=tmp_path, metrics=None)
    lines = cmi_v3._render_cmi_section(ctx)
    assert any("n/a" in ln for ln in lines)
    assert any("Neural CMI" in ln for ln in lines)

    (tmp_path / "cmi.json").write_text(json.dumps({"error": "latents unavailable"}), "utf-8")
    lines = cmi_v3._render_cmi_section(ctx)
    assert any("latents unavailable" in ln for ln in lines)


def test_cmi_section_renders_the_table_and_the_ceiling(tmp_path) -> None:
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import final_report_v2 as fr

    summary = cmi_v3.run_cmi_comparison(
        _fake_cells(), cfg=_cmp_cfg(), configs=list(cmi_v3._ALL_CONFIGS),
        horizon=_HORIZON, seed=0,
    )
    summary["arm"] = "v3_prod"
    summary["split"] = "val"
    summary["n_samples"] = 48
    (tmp_path / "cmi.json").write_text(json.dumps(summary), "utf-8")

    ctx = fr.SectionContext(config={}, benchmark="G1_raw", results_dir=tmp_path, metrics=None)
    text = "\n".join(cmi_v3._render_cmi_section(ctx))
    assert "Neural CMI" in text
    assert "InfoNCE ceiling" in text and "3.4657" in text
    assert "model-coupling bias" in text
    assert "target_state" in text
    assert "n/a" not in text.split("| cell |")[0].split("## Neural CMI")[1][:200]


def test_cmi_table_csv_header_and_rows(tmp_path) -> None:
    r"""One row per cell, one column group per configuration, plus the bias."""
    import csv as _csv

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3

    summary = cmi_v3.run_cmi_comparison(
        _fake_cells(n_cells=3), cfg=_cmp_cfg(), configs=list(cmi_v3._ALL_CONFIGS),
        horizon=_HORIZON, seed=0,
    )
    path = cmi_v3.write_cmi_table(summary, tmp_path / "cmi_table.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(_csv.reader(handle))
    header, body = rows[0], rows[1:]
    assert header[: len(cmi_v3._CSV_BASE)] == list(cmi_v3._CSV_BASE)
    for name in cmi_v3._ALL_CONFIGS:
        assert f"cmi_{name}" in header and f"cmi_{name}_lo" in header
    assert "cmi_latent_gauss_exact" in header
    assert header[-3:] == ["bias", "bias_lo", "bias_hi"]
    assert len(body) == 3
    assert [r[0] for r in body] == ["0", "1", "2"]


def test_a_real_summary_renders_through_the_visualizer(tmp_path) -> None:
    r"""End-to-end: a summary this module actually produces reaches a figure.

    The figure's own contract (ceiling drawn and labelled, empty-input guard, missing-bias panel)
    is pinned in ``test_visualize_v2.py -k cmi`` against a hand-written payload, so this only has
    to prove the two halves fit together.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import cmi_v3
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v2 as viz

    summary = cmi_v3.run_cmi_comparison(
        _fake_cells(), cfg=_cmp_cfg(), configs=list(cmi_v3._ALL_CONFIGS),
        horizon=_HORIZON, seed=0,
    )
    summary["arm"] = "v3_prod"
    summary["split"] = "val"
    written = viz.plot_cmi_comparison(summary, tmp_path / "cmi_comparison")
    assert {p.suffix for p in written} == {".pdf", ".png"}
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)
