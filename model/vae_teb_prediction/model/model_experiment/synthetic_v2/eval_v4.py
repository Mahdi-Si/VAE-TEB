r"""Sprint 6: the headline falsifiable grader for ``synthetic_v4`` (raw-domain ground truth).

Grades the raw model's per-step KL surrogate $\bar K$ against the **known** injected transfer
entropy $\mathrm{TE}_{\mathrm{inj}}$, with four core gates:

* **calibration** -- OLS $\bar K = \alpha + \gamma\,\mathrm{TE}_{\mathrm{inj}}$ (slope $\gamma$,
  intercept $\alpha$, $R^2$, Spearman $\rho$) plus a per-lag table and the KLD-summary family
  (:func:`fit_calibration_v4`);
* **null-cell gate** -- $\bar K\big|_{\mathrm{TE}_{\mathrm{inj}}=0}\to 0$ below a loose decidable
  ceiling (:func:`null_cell_gate_v4`);
* **prediction-space source control** -- $\mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}}
  < \mathcal L_{\mathrm{feat}}^{\pi(U)}$ with ``shuffle_penalty > 0`` on the **raw** forecast MSE,
  where $\mathcal L_{\mathrm{feat}}^{\pi(U)}$ is scored through the model-native permutation
  ``model.perm_forward_outputs`` (:func:`prediction_controls_v4`);
* **lag recovery** -- $\operatorname{argmax}_\ell \bar\alpha_{t,\ell} \approx D$ against the planted
  lag $D$ (:func:`recover_lag_v4`, via the ``planted_lag_to_model_lag`` identity).

The reduction / calibration / lag *skeleton* is reused from :mod:`eval_v2` (``_masked_forecast_loss``,
``_clean_window_mean``, ``fit_calibration_slope``, ``recover_lags`` ...); the per-sample collection is
re-implemented against the **raw** forward keys (4-D ``mu_full``/``mu_base`` $(B,T,H,R)$ and the raw
future target from ``build_future_target``). There is no ``te_scat`` / ``frac_phi``:
$\mathrm{TE}_{\mathrm{inj}}$ (``sample_te_true``) is the sole ground-truth axis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import geometry_align_v4
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
    _bootstrap_ci,
    _clean_window_mean,
    _group_per_cell,
    _masked_forecast_loss,
    _row_window_reductions,
    _spearman_finite,
    _spearman_sign,
    KLD_SCALAR_VARIANTS,
    _OUT_OF_SUPPORT_UNDER_ANCHOR,
    fit_calibration_slope,
    prediction_controls,
    recover_lags,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
    _build_runner_and_loader_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)

logger = logging.getLogger(__name__)

#: Loose decidable ceiling on the mean $\bar K$ over null cells (nats); the tight ceiling is the
#: S1-T03 empirical constant.
_DEFAULT_NULL_KBAR_CEILING: float = 0.5
#: Default LagMass pass threshold and argmax tolerance for lag recovery.
_DEFAULT_LAG_MASS_THRESHOLD: float = 0.5
_DEFAULT_LAG_TOLERANCE: int = 1
#: The name of the model-native prediction-space control ($\pi(U)$-permuted source).
_SHUFFLE_CONTROL: str = "shuffled"


def _prov(batch: Any, name: str, default: float, dtype: Any, n: int) -> np.ndarray:
    r"""Read a per-sample provenance field off the batch as a length-``n`` numpy array."""
    val = getattr(batch, name, None)
    if val is None:
        return np.full(n, default, dtype=dtype)
    arr = np.asarray(val.detach().cpu().numpy() if hasattr(val, "detach") else val)
    return arr.reshape(-1).astype(dtype)


# ===========================================================================
# S6-T01: per-sample K-bar collection against the RAW forward keys.
# ===========================================================================
def collect_per_sample_kbar_v4(
    runner: Any,
    loader: Any,
    *,
    warmup: int,
    horizon: int,
    max_samples: Optional[int] = None,
    control: str = _SHUFFLE_CONTROL,
) -> Dict[str, Any]:
    r"""Reduce each sample to $\bar K$ + raw forecast losses from the raw forward dict (S6-T01).

    For every batch it runs ``runner.forward`` (the 25-key raw dict), reduces the per-step KL
    ``kld_per_t`` $(B,T)$ over each sample's clean window (:func:`_clean_window_mean`), scores the
    4-D ``mu_full``/``mu_base`` $(B,T,H,R)$ against the raw future target
    ``runner.build_future_target(batch)`` with :func:`_masked_forecast_loss`, and produces the
    model-native shuffled control ``feat_loss_<control>`` via ``model.perm_forward_outputs`` (the
    $\pi(U)$-permuted source scored against the *same* raw target). The KLD-summary family
    (:data:`KLD_SCALAR_VARIANTS`), the per-head split, and per-cell lag profiles are accumulated
    exactly as in :func:`eval_v2.collect_per_sample_kbar`, but there is no ``te_scat`` / ``frac_phi``.

    Args:
        runner: A live :class:`TestRunner` around a synthetic-v4 checkpoint.
        loader: A synthetic-v4 dataloader (batches expose ``fhr`` / ``up`` / ``weight`` and the
            provenance ``te_true`` / ``delay`` / ``cell_id`` / ``held_out``).
        warmup: Encoder warm-up $w$ (excluded from the clean window).
        horizon: Forecast horizon $H$ (the final $H$ steps are excluded).
        max_samples: Optional cap on samples consumed.
        control: Name for the model-native prediction-space control (default ``"shuffled"``).

    Returns:
        A dict of per-sample $(N,)$ arrays (``kbar`` + the :data:`KLD_SCALAR_VARIANTS` family,
        ``feat_loss`` / ``base_loss`` / ``pred_gain`` / ``feat_loss_<control>``, provenance
        ``te_inj`` / ``te_scat`` (NaN) / ``delay`` / ``cell_id`` / ``held_out``), the per-head
        ``kbar_head`` $(N, M)$ + expanded columns, and the aggregates ``lag_profiles``
        (``{cell_id: (L,)}``), ``kbar_over_time`` (``{cell_id: (T,)}``), ``n`` and ``T``.
    """
    import torch

    model = runner.model
    gen = torch.Generator().manual_seed(0)

    cols: Dict[str, List[np.ndarray]] = {
        k: [] for k in (
            list(KLD_SCALAR_VARIANTS)
            + ["cell_id", "delay", "te_inj", "te_scat", "held_out",
               "feat_loss", "base_loss", "pred_gain", f"feat_loss_{control}"]
        )
    }
    head_chunks: List[np.ndarray] = []
    lag_sum: Dict[int, np.ndarray] = {}
    lag_cnt: Dict[int, int] = {}
    kot_sum: Dict[int, np.ndarray] = {}
    kot_cnt: Dict[int, int] = {}
    T_seen = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            out = runner.forward(batch)
            kld_bt = out["kld_per_t"]                                # (B, T)
            bsz, T = int(kld_bt.shape[0]), int(kld_bt.shape[1])
            T_seen = T
            device = kld_bt.device

            delay_np = _prov(batch, "delay", 0, np.int64, bsz)
            delay_t = torch.as_tensor(delay_np, device=device)

            kbar, valid = _clean_window_mean(kld_bt, delay_t, warmup=warmup, horizon=horizon)
            vmask = valid.to(kld_bt.dtype)
            vdenom = vmask.sum(dim=1).clamp(min=1.0)
            kld_np = kld_bt.detach().cpu().numpy().astype(np.float64)
            valid_np = valid.detach().cpu().numpy()

            cols["kbar"].append(kbar.detach().cpu().numpy())
            red = _row_window_reductions(kld_np, valid_np)
            cols["kbar_sum"].append(red["sum"])
            cols["kbar_max"].append(red["max"])
            cols["kbar_median"].append(red["median"])
            cols["kbar_p90"].append(red["p90"])
            cols["kbar_full"].append(kld_np.mean(axis=1))
            hi_pw = int(T - horizon)
            cols["kbar_postwarm"].append(
                kld_np[:, int(warmup):hi_pw].mean(axis=1) if hi_pw > int(warmup)
                else np.full(bsz, np.nan, dtype=np.float64))

            # Directed-KL split over the true lag band L* = {max(0,D-H)..D-1}; te_lag_map sums to
            # kld_per_t over lags, so in-band + out-band == kbar exactly.
            te_map = out.get("te_lag_map", None)
            if te_map is not None:
                Lc = int(te_map.shape[-1])
                l_idx = torch.arange(Lc, device=device).unsqueeze(0)
                lo_b = torch.clamp(delay_t.long() - int(horizon), min=0).unsqueeze(1)
                hi_b = delay_t.long().unsqueeze(1)
                band = ((l_idx >= lo_b) & (l_idx < hi_b)).to(te_map.dtype)
                inband_t = (te_map * band.unsqueeze(1)).sum(dim=-1)     # (B, T)
                kbar_inb = (inband_t * vmask).sum(dim=1) / vdenom
                cols["kbar_inband"].append(kbar_inb.detach().cpu().numpy())
                cols["kbar_outband"].append((kbar - kbar_inb).detach().cpu().numpy())
            else:
                cols["kbar_inband"].append(np.full(bsz, np.nan, dtype=np.float64))
                cols["kbar_outband"].append(np.full(bsz, np.nan, dtype=np.float64))

            kph = out.get("kld_per_t_per_head", None)
            if kph is not None:
                head_kbar = (kph * vmask.unsqueeze(-1)).sum(dim=1) / vdenom.unsqueeze(-1)
                head_chunks.append(head_kbar.detach().cpu().numpy())

            cid = _prov(batch, "cell_id", 0, np.int64, bsz)
            cols["cell_id"].append(cid)
            cols["delay"].append(delay_np)
            cols["te_inj"].append(_prov(batch, "te_true", np.nan, np.float64, bsz))
            # te_scat has no meaning in the raw pipeline; stamped NaN so the reused per-cell
            # grouping (_group_per_cell) and calibration table stay structurally compatible.
            cols["te_scat"].append(np.full(bsz, np.nan, dtype=np.float64))
            cols["held_out"].append(_prov(batch, "held_out", 0, np.int64, bsz))

            # -- Raw forecast losses: 4-D mu_full/mu_base vs the raw future target ---------------
            mu_full = out["mu_full"]
            mu_base = out["mu_base"]
            x_plus = runner.build_future_target(batch)                  # (B, T_valid, H, R)
            l_full = _masked_forecast_loss(mu_full, x_plus, valid)
            l_base = _masked_forecast_loss(mu_base, x_plus, valid)
            cols["feat_loss"].append(l_full.detach().cpu().numpy())
            cols["base_loss"].append(l_base.detach().cpu().numpy())
            cols["pred_gain"].append((l_base - l_full).detach().cpu().numpy())

            # Model-native prediction-space control: the pi(U)-permuted source's forecast scored
            # against the SAME raw target (a wrong source must be worse than no source).
            if bsz >= 2:
                perm_out = model.perm_forward_outputs(out, generator=gen)
                l_shuf = _masked_forecast_loss(perm_out["mu_full"], x_plus, valid)
                cols[f"feat_loss_{control}"].append(l_shuf.detach().cpu().numpy())
            else:
                cols[f"feat_loss_{control}"].append(np.full(bsz, np.nan, dtype=np.float64))

            # -- Per-cell lag profile + K-bar-over-time (clean-window means, summed per cell) ----
            lag_map = te_map if te_map is not None else out["attn_weights"].mean(dim=2)
            valid_lag = valid.to(lag_map.dtype).unsqueeze(-1)
            prof = (lag_map * valid_lag).sum(dim=1) / valid_lag.sum(dim=1).clamp(min=1.0)
            prof_np = prof.detach().cpu().numpy().astype(np.float64)
            for i in range(bsz):
                c = int(cid[i])
                lag_sum[c] = lag_sum.get(c, np.zeros(prof_np.shape[1])) + prof_np[i]
                lag_cnt[c] = lag_cnt.get(c, 0) + 1
                kot_sum[c] = kot_sum.get(c, np.zeros(T)) + kld_np[i]
                kot_cnt[c] = kot_cnt.get(c, 0) + 1

    arrs: Dict[str, Any] = {k: np.concatenate(v) for k, v in cols.items() if v}
    if head_chunks:
        kbar_head = np.concatenate(head_chunks, axis=0)                 # (N, M)
        arrs["kbar_head"] = kbar_head
        for m in range(kbar_head.shape[1]):
            arrs[f"kbar_head{m}"] = kbar_head[:, m]
    arrs["lag_profiles"] = {c: lag_sum[c] / max(lag_cnt[c], 1) for c in lag_sum}
    arrs["kbar_over_time"] = {c: kot_sum[c] / max(kot_cnt[c], 1) for c in kot_sum}
    arrs["n"] = int(arrs["kbar"].size) if "kbar" in arrs else 0
    arrs["T"] = int(T_seen)
    return arrs


# ===========================================================================
# S6-T02: K-bar-vs-TE_inj calibration (single ground-truth axis).
# ===========================================================================
def _safe_slope(points):
    try:
        return fit_calibration_slope(points)
    except ValueError:
        return None


def fit_calibration_v4(arrs: Dict[str, Any], *, kld_support: str = "anchor") -> Dict[str, Any]:
    r"""Fit $\bar K = \alpha + \gamma\,\mathrm{TE}_{\mathrm{inj}}$ on the single ground-truth axis (S6-T02).

    Reuses the :mod:`eval_v2` OLS / grouping / rank kernels but drops the scattering axis: the raw
    pipeline has only $\mathrm{TE}_{\mathrm{inj}}$ (``te_inj``). Reports the per-cell and pooled
    per-sample slope $\gamma$, intercept $\alpha$, $R^2$ and Spearman $\rho$; a per-lag table; and
    the KLD-summary family, with ``kbar_postwarm`` the anchor comparator and ``kbar_full`` flagged
    ``out_of_support`` when ``kld_support == "anchor"``.

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar_v4`.
        kld_support: The evaluated model's ``kld_support`` (``"anchor"`` / ``"full"``).

    Returns:
        A dict with ``gamma`` / ``alpha`` / ``r2`` / ``spearman`` / ``monotonic`` / ``n_cells``
        (per-cell), the pooled ``gamma_sample`` / ``alpha_sample`` / ``r2_sample`` / ``n_samples``,
        a ``by_lag`` table, a ``kld_variants`` table (each with ``out_of_support``), and a
        ``per_cell`` summary list.
    """
    per_cell = _group_per_cell(arrs)
    cells = list(per_cell.values())
    inj_points = [(c["te_inj"], c["kbar"]) for c in cells]
    fit = _safe_slope(inj_points)
    rho = _spearman_sign(inj_points)

    kbar_a = np.asarray(arrs["kbar"], dtype=np.float64)
    te_a = np.asarray(arrs["te_inj"], dtype=np.float64)
    fit_s = _safe_slope(list(zip(te_a, kbar_a)))

    delay_a = np.asarray(arrs.get("delay", np.zeros_like(kbar_a)))
    by_lag: Dict[int, Dict[str, Any]] = {}
    finite_delays = np.unique(delay_a[np.isfinite(delay_a.astype(np.float64))]) \
        if delay_a.size else []
    for d in finite_delays:
        sel = delay_a == d
        f = _safe_slope(list(zip(te_a[sel], kbar_a[sel])))
        by_lag[int(d)] = {
            "gamma": f["gamma"] if f else None,
            "alpha": f["alpha"] if f else None,
            "r2": f["r2"] if f else None,
            "n": int(np.sum(sel)),
        }

    variant_names = [v for v in KLD_SCALAR_VARIANTS if v in arrs]
    variant_names += sorted(
        k for k in arrs
        if str(k).startswith("kbar_head") and np.asarray(arrs.get(k)).ndim == 1
    )
    kld_variants: Dict[str, Any] = {}
    for v in variant_names:
        y = np.asarray(arrs[v], dtype=np.float64)
        if y.shape != kbar_a.shape:
            continue
        f = _safe_slope(list(zip(te_a, y)))
        kld_variants[v] = {
            "n": int(np.isfinite(y).sum()),
            "out_of_support": bool(v in _OUT_OF_SUPPORT_UNDER_ANCHOR
                                   and str(kld_support) == "anchor"),
            "gamma": f["gamma"] if f else None,
            "alpha": f["alpha"] if f else None,
            "r2": f["r2"] if f else None,
            "spearman": _spearman_finite(te_a, y),
        }

    return {
        "kld_support": str(kld_support),
        "gamma": fit["gamma"] if fit else None,
        "alpha": fit["alpha"] if fit else None,
        "r2": fit["r2"] if fit else None,
        "spearman": rho,
        "monotonic": bool(rho is not None and rho > 0),
        "n_cells": len(cells),
        "n_samples": int(kbar_a.size),
        "gamma_sample": fit_s["gamma"] if fit_s else None,
        "alpha_sample": fit_s["alpha"] if fit_s else None,
        "r2_sample": fit_s["r2"] if fit_s else None,
        "by_lag": by_lag,
        "kld_variants": kld_variants,
        "per_cell": [
            {k: c[k] for k in ("cell_id", "te_inj", "kbar", "delay", "n")} for c in cells
        ],
    }


# ===========================================================================
# S6-T03: null-cell gate.
# ===========================================================================
def null_cell_gate_v4(
    arrs: Dict[str, Any], *, ceiling: float = _DEFAULT_NULL_KBAR_CEILING, boot_seed: int = 0,
) -> Dict[str, Any]:
    r"""Gate $\bar K\big|_{\mathrm{TE}_{\mathrm{inj}}=0}\to 0$ against a loose ceiling (S6-T03).

    Bootstraps a CI on the per-cell mean $\bar K$ over the null cells (``te_inj == 0``) and passes
    when the mean sits below ``ceiling``. The loose ceiling is decidable now; the tight ceiling is
    the S1-T03 empirical constant.

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar_v4`.
        ceiling: Loose decidable ceiling on the mean null $\bar K$ (nats).
        boot_seed: Seed for the bootstrap CI.

    Returns:
        ``{mean, std, ci_lo, ci_hi, n_cells, ceiling, pass}`` or ``None`` when the grid has no null
        cell.
    """
    per_cell = _group_per_cell(arrs)
    null_kbars = np.asarray(
        [c["kbar"] for c in per_cell.values() if float(c["te_inj"]) == 0.0], dtype=np.float64)
    if not null_kbars.size:
        return {"pass": None, "n_cells": 0, "ceiling": float(ceiling),
                "note": "no null (te_inj==0) cell in the grid"}
    mean_null, ci_lo, ci_hi = _bootstrap_ci(null_kbars, rng=np.random.default_rng(int(boot_seed)))
    return {
        "mean": mean_null,
        "std": float(np.std(null_kbars)) if null_kbars.size > 1 else 0.0,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_cells": int(null_kbars.size),
        "ceiling": float(ceiling),
        "pass": bool(np.isfinite(mean_null) and mean_null < float(ceiling)),
    }


# ===========================================================================
# S6-T04: raw prediction-space source control (model-native permutation).
# ===========================================================================
def prediction_controls_v4(
    arrs: Dict[str, Any], *, control: str = _SHUFFLE_CONTROL, null_tol: float = 0.05,
) -> Dict[str, Any]:
    r"""The raw prediction-space source control $\mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}}
    < \mathcal L_{\mathrm{feat}}^{\pi(U)}$ (S6-T04).

    A thin adapter over :func:`eval_v2.prediction_controls`: the ``feat_loss_<control>`` column was
    produced by :func:`collect_per_sample_kbar_v4` through the **model-native** permutation
    (``model.perm_forward_outputs``) scored on the raw forecast MSE, so the gate reads exactly as in
    v2 (``ordering_pass`` on signal cells, ``shuffle_penalty > 0``).

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar_v4` (needs ``feat_loss`` / ``base_loss``
            / ``feat_loss_<control>`` / ``cell_id`` / ``te_inj``).
        control: The control column name (default ``"shuffled"``).
        null_tol: Relative tolerance for the null-cell coincidence check.

    Returns:
        The :func:`eval_v2.prediction_controls` result (``controls`` / ``n_signal_cells`` /
        ``overall`` / ``per_cell``).
    """
    return prediction_controls(arrs, [control], null_tol=null_tol)


# ===========================================================================
# S6-T05: lag recovery vs the planted D.
# ===========================================================================
def recover_lag_v4(
    arrs: Dict[str, Any], *,
    horizon: int,
    tolerance: int = _DEFAULT_LAG_TOLERANCE,
    threshold: float = _DEFAULT_LAG_MASS_THRESHOLD,
) -> Dict[str, Any]:
    r"""Recover the planted lag $D$ from the accumulated attention lag profiles (S6-T05).

    Reuses :func:`eval_v2.recover_lags` on the per-cell ``lag_profiles`` and the per-cell metadata
    from :func:`_group_per_cell`. The planted lag maps to the model lag index unchanged
    (``geometry_align_v4.planted_lag_to_model_lag(D) == D``), so the recovered
    $\operatorname{argmax}_\ell \bar\alpha_{t,\ell}$ is compared directly to $D$ within ``tolerance``.

    Args:
        arrs: The dict from :func:`collect_per_sample_kbar_v4` (needs ``lag_profiles``).
        horizon: The forecast horizon $H$ (sets the true band $\{\max(0,D-H)..D-1\}$).
        tolerance: The $\pm$ step tolerance for the argmax match.
        threshold: The LagMass pass threshold.

    Returns:
        The :func:`eval_v2.recover_lags` result (per-cell table + ``mean_lag_mass`` /
        ``frac_within_tol`` / ``mean_lag_mass_pass``), with the planted-lag identity asserted.
    """
    # Guard the alignment identity so a geometry drift fails loudly rather than mis-scoring lag.
    assert geometry_align_v4.planted_lag_to_model_lag(int(horizon)) == int(horizon)
    cells_by_id = _group_per_cell(arrs)
    lag_profiles = {int(c): np.asarray(p, dtype=np.float64)
                    for c, p in arrs.get("lag_profiles", {}).items()}
    return recover_lags(
        lag_profiles, cells_by_id, horizon=int(horizon),
        tolerance=int(tolerance), threshold=float(threshold),
    )


# ===========================================================================
# S6-T06: assembly + the ``eval`` stage (machine artifacts only).
# ===========================================================================
def _eval_knob(config: Dict[str, Any], benchmark: str, key: str, default: Any) -> Any:
    r"""Read a grading knob from ``benchmarks.<b>.eval.grading.<key>`` with a fallback default."""
    grading = (config.get("benchmarks", {}).get(benchmark, {})
               .get("eval", {}).get("grading", {}))
    return grading.get(key, default)


def _te_raw_gate(config: Dict[str, Any], benchmark: str, results_dir: Path, *, pilot: bool,
                 render_mode: Optional[str] = None) -> Dict[str, Any]:
    r"""Fold in the Sprint-1 model-free ``te_raw`` realizability gate.

    Reuses a pre-written ``realizability.json`` (from the ``realizability`` stage) when present, else
    recomputes it via :func:`realizability_v4.compute_realizability`. The cached file lives at the
    render-mode-agnostic tag root (``results/<tag>/``), so it is reused **only** when its recorded
    ``render_mode`` matches ``render_mode`` (the arm's); otherwise it belongs to a different render
    mode (e.g. the ``direct`` pre-flight while grading ``am_carrier_prod``) and is recomputed, so the
    model-free gate is never mis-attributed across render modes.
    """
    cached = results_dir / "realizability.json"
    if cached.is_file():
        try:
            with open(cached, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            cached_rm = report.get("render_mode")
            if render_mode is None or cached_rm is None or str(cached_rm) == str(render_mode):
                return {"gate": report.get("gate"), "constants": report.get("constants"),
                        "source": "realizability.json"}
            logger.warning("realizability.json render_mode=%r != arm render_mode=%r; recomputing",
                           cached_rm, render_mode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", cached, exc)
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.realizability_v4 import (
        compute_realizability,
    )
    report = compute_realizability(config, benchmark=benchmark, pilot=pilot)
    return {"gate": report.get("gate"), "constants": report.get("constants"), "source": "recomputed"}


def _write_per_sample_eval(arrs: Dict[str, Any], path: Path) -> None:
    r"""Persist the per-sample arrays (1-D/2-D numpy only) to ``per_sample_eval.npz``."""
    payload = {k: np.asarray(v) for k, v in arrs.items()
               if isinstance(v, np.ndarray) and v.ndim in (1, 2)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


def run_eval_v4(ctx: StageContextV4) -> int:
    r"""``eval`` stage: assemble the four core gates + the ``te_raw`` gate into ``metrics.json``.

    Builds the runner/loader off the arm's checkpoint, collects per-sample $\bar K$, fits the
    calibration, runs the null gate, the raw prediction-space control and lag recovery, folds in the
    ``te_raw`` realizability gate, and writes ``metrics.json`` + ``per_sample_eval.npz`` under the
    arm's run dir. **No human-readable report** (that is S7-T05).
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_runner_v4 import (
        _resolve_eval_checkpoint,
    )

    config = ctx.config
    benchmark = ctx.benchmark
    # Artifacts are split-scoped (results/<tag>/<arm>/<split>/) when the driver fans out over
    # splits (S7-T03); with split=None this collapses to the arm root the Sprint-6 stage assumed.
    # The checkpoint itself is discovered from the arm root by _resolve_eval_checkpoint.
    run_dir = ctx.output_dir()
    ckpt = _resolve_eval_checkpoint(ctx)
    runner, loader = _build_runner_and_loader_v4(
        ckpt, config, benchmark=benchmark, output_dir=run_dir / "_eval",
        batch_size=2, split=ctx.split or "val",
    )
    model = runner.model
    warmup, horizon = int(runner.warmup_steps), int(runner.horizon)
    kld_support = str(getattr(model, "kld_support", "anchor"))
    render_mode = str(config["benchmarks"][benchmark]["raw"].get("render_mode", "direct"))

    max_samples = 16 if ctx.pilot else None
    arrs = collect_per_sample_kbar_v4(
        runner, loader, warmup=warmup, horizon=horizon, max_samples=max_samples)

    null_ceiling = float(_eval_knob(config, benchmark, "null_cell_kbar_ceiling",
                                    _DEFAULT_NULL_KBAR_CEILING))
    lag_thr = float(_eval_knob(config, benchmark, "lag_mass_threshold",
                               _DEFAULT_LAG_MASS_THRESHOLD))

    metrics: Dict[str, Any] = {
        "model_class": type(model).__name__,
        "arm": ctx.arm,
        "render_mode": render_mode,
        "kld_support": kld_support,
        "n_samples": int(arrs.get("n", 0)),
        "calibration": fit_calibration_v4(arrs, kld_support=kld_support),
        "null_cell_gate": null_cell_gate_v4(arrs, ceiling=null_ceiling),
        "prediction_controls": prediction_controls_v4(arrs),
        "lag_recovery": recover_lag_v4(arrs, horizon=horizon, threshold=lag_thr),
        "te_raw_gate": _te_raw_gate(config, benchmark, ctx.results_dir(), pilot=ctx.pilot,
                                    render_mode=render_mode),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    tmp = metrics_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    tmp.replace(metrics_path)
    _write_per_sample_eval(arrs, run_dir / "per_sample_eval.npz")

    cal = metrics["calibration"]
    print(f"[eval] arm={ctx.arm} gamma={cal.get('gamma')} r2={cal.get('r2')} "
          f"null_pass={metrics['null_cell_gate'].get('pass')} -> {metrics_path}")
    return 0


def _json_default(obj: Any) -> Any:
    r"""JSON-serialise numpy scalars / arrays that slip into the metrics tree."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")


register_stage_v4(StageSpecV4(
    name="eval",
    run=run_eval_v4,
    order=50,
    model_dependent=True,
    fatal=True,
    help="ground-truth grading: K-bar-vs-te_inj calibration, null gate, pred control, lag recovery",
))
