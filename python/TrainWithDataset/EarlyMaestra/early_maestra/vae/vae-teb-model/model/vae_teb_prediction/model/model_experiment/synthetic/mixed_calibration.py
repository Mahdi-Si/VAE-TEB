r"""``mixed_calibration`` -- $\beta$-sweep + calibration-based $\beta$ selection
for the ``G1_mix`` mixed-population model.

The mixed-population experiment trains **one** model on a heterogeneous pool
(:mod:`mixed_dataset`) and reports the latent-KL surrogate
$\bar K = \mathrm{mean}\,\texttt{kld\_per\_t}$ as a transfer-entropy estimator.
The headline claim is *calibration*,

$$
\bar K \;=\; \alpha \;+\; \gamma\,\mathrm{TE}^{(H)}_{U\to Y},
\qquad \gamma \to 1 .
$$

Calibration is **$\beta$-dependent**: the KL weight $\beta$ controls how much
source information the bottleneck passes, so a single fixed $\beta$ cannot be
assumed to calibrate. At a too-small $\beta$ the KL is nearly free and
$I_q(Z;U\mid Y) \gg \mathrm{TE}$ (so $\gamma \gg 1$); at a too-large $\beta$ the
latent collapses ($\gamma \to 0$). This module therefore trains the pooled model
**once per $\beta$**, evaluates each checkpoint with :func:`mixed_eval.evaluate_mixed`,
fits the **per-$M$** calibration slope at each $\beta$, and selects

$$
\beta^\star \;=\; \arg\min_\beta\;
    \operatorname{mean}_M |\gamma_M(\beta) - 1|
    \;+\; \lambda_\alpha \operatorname{mean}_M |\alpha_M(\beta)| ,
$$

the per-$M$ generalisation of :func:`calibration.select_beta_by_calibration`.

Compute model (Decision: 8x A6000 box): the **default** is *task-parallel
single-GPU* -- one independent pooled training per $\beta$, fanned across the
GPUs by :func:`gpu_pool.run_gpu_pool` (``mode="mix_beta"``). A sequential
multi-GPU DDP fallback (:func:`train_ddp.train_ddp` per $\beta$) is available via
``mode="ddp"`` for boxes where a single pooled model does not saturate one card.

Run modes (Decision V2-D8): both a CLI and an edit-and-run ``RUN_CONFIG``.

    # Build the pool + sweep + select (task-parallel across 8 GPUs):
    python -m ...synthetic.mixed_calibration --gpus 0,1,2,3,4,5,6,7
    # Evaluate + select only (checkpoints already trained by gpu_pool):
    python -m ...synthetic.mixed_calibration --no-build --no-train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    calibration as cal,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    mixed_dataset as md,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    mixed_eval as me,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    plot_style as ps,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    apply_path_overrides,
    load_config,
    resolve_active_benchmark,
    resolve_user_path,
)

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_BENCHMARK = "G1_mix"
_OUT_SUBDIR = "mixed_calibration"


# =============================================================================
# Grid + tag helpers (shared with gpu_pool._cells_mix_beta -- one source of truth)
# =============================================================================

def _resolve_beta_grid(config: Dict[str, Any]) -> List[float]:
    r"""Resolve the $\beta$ grid for the mixed sweep.

    Fallback chain (first non-empty wins): ``mix_calibration.beta_grid`` ->
    ``calibration.beta_grid`` -> ``beta_sweep.grid``. This is the single source
    of truth shared with :func:`gpu_pool._cells_mix_beta`, so the trained $\beta$
    set and the evaluated $\beta$ set can never diverge.

    Args:
        config: The parsed config.

    Returns:
        The (possibly empty) list of $\beta$ values as floats.
    """
    for block, key in (
        ("mix_calibration", "beta_grid"),
        ("calibration", "beta_grid"),
        ("beta_sweep", "grid"),
    ):
        grid = (config.get(block, {}) or {}).get(key)
        if grid:
            return [float(b) for b in grid]
    return []


def _beta_run_tag(beta: float) -> str:
    r"""Results subdirectory for one $\beta$: ``mixed_calibration/beta_<token>``.

    Reuses :func:`calibration._beta_token` so the checkpoint lands where
    :func:`evaluate_betas` later looks
    (``results/G1_mix/mixed_calibration/beta_<token>/final.ckpt``).

    Args:
        beta: The KL weight.

    Returns:
        The run-tag string.
    """
    return f"{_OUT_SUBDIR}/{cal._beta_token(float(beta))}"


def _alpha_penalty(config: Dict[str, Any], override: Optional[float]) -> float:
    """Resolve the intercept penalty $\\lambda_\\alpha$ for the selector."""
    if override is not None:
        return float(override)
    mc_cfg = config.get("mix_calibration", {}) or {}
    cal_cfg = config.get("calibration", {}) or {}
    return float(mc_cfg.get("alpha_penalty", cal_cfg.get("alpha_penalty", 0.05)))


# =============================================================================
# Build
# =============================================================================

def build_pool_caches(
    config: Dict[str, Any], *, force: bool = False, holdout: bool = True,
) -> Tuple[Path, Optional[Path]]:
    r"""Build the in-mix pool once (and optionally the held-out pool).

    Reuses :func:`mixed_dataset.build_g1_mix`; the build runs serially in this
    process (never inside a GPU worker), and ``mix.build_workers: auto`` fans the
    per-cell coupling solves + sample generation across the CPU cores.

    Args:
        config: The parsed, ``G1_mix``-resolved config.
        force: Regenerate even when a complete cache exists.
        holdout: Also build the test-only held-out extrapolation cache.

    Returns:
        ``(in_mix_dir, holdout_dir)`` -- ``holdout_dir`` is ``None`` when
        ``holdout=False``.
    """
    in_mix_dir = md.build_g1_mix(config, force=force, holdout=False)
    holdout_dir = (
        md.build_g1_mix(config, force=force, holdout=True) if holdout else None
    )
    return in_mix_dir, holdout_dir


def _ckpt_exists(config: Dict[str, Any], run_tag: str, ckpt_name: str) -> bool:
    """Whether ``results/G1_mix/<run_tag>/<ckpt_name>`` is already on disk."""
    root = resolve_user_path(config["paths"]["results_dir"])
    return (root / _BENCHMARK / run_tag / ckpt_name).is_file()


# =============================================================================
# Train (two strategies)
# =============================================================================

def train_betas_task_parallel(
    config: Dict[str, Any],
    config_path: Path,
    *,
    betas: Sequence[float],
    gpus: Sequence[int],
    build: bool = True,
    force: bool = False,
    epochs: Optional[int] = None,
    seed: Optional[int] = None,
    data_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    r"""Default path: one single-GPU pooled training per $\beta$, fanned across GPUs.

    Delegates to :func:`gpu_pool.run_gpu_pool` with ``mode="mix_beta"`` so the
    cell enumeration (and the in-mix + held-out cache build) live in exactly one
    place (:func:`gpu_pool._cells_mix_beta`). ``betas`` is accepted for signature
    symmetry with :func:`train_betas_ddp`; the grid the pool trains is resolved
    identically via :func:`_resolve_beta_grid`.

    Args:
        config: The parsed, ``G1_mix``-resolved config.
        config_path: Path to ``config_synth.yaml`` (workers reload it).
        betas: The $\beta$ grid (informational; the pool re-resolves it).
        gpus: GPU slot list.
        build: Build the pools before dispatch (idempotent).
        force: Retrain cells whose ``final.ckpt`` exists.
        epochs, seed: Optional global per-cell overrides.
        data_dir, results_dir: Optional path overrides.
        dry_run: Enumerate cells and return without training.

    Returns:
        The :func:`gpu_pool.run_gpu_pool` result dict.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        gpu_pool as gp,
    )

    return gp.run_gpu_pool(
        config, config_path, mode="mix_beta", gpus=gpus,
        build=build, force=force, epochs=epochs, seed=seed,
        data_dir=data_dir, results_dir=results_dir, dry_run=dry_run,
    )


def train_betas_ddp(
    config: Dict[str, Any],
    *,
    betas: Sequence[float],
    devices: Any,
    in_mix_tag: str,
    force: bool = False,
    epochs: Optional[int] = None,
    seed: Optional[int] = None,
    ckpt_name: str = "final.ckpt",
) -> List[Dict[str, Any]]:
    r"""Fallback path: sequential multi-GPU DDP training, one $\beta$ at a time.

    Each $\beta$ is a full DDP run via :func:`train_ddp.train_ddp` (which forces
    ``benchmark=G1_mix`` and overlays the Gaussian-NLL loss); the ``beta``
    override routes through ``_OVERRIDE_MAP`` to ``loss.kld_beta``. Cells whose
    checkpoint already exists are skipped unless ``force``.

    Args:
        config: The parsed config (any active benchmark; DDP re-forces G1_mix).
        betas: The $\beta$ grid.
        devices: Device spec forwarded to ``train_ddp`` (int / list / "0,1,...").
        in_mix_tag: The pooled cache tag (``experiment.tag``).
        force: Retrain even when the checkpoint exists.
        epochs, seed: Optional overrides.
        ckpt_name: Checkpoint file used for the skip check.

    Returns:
        One result dict per trained $\beta$ (skips report ``status="skipped"``).
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        train_ddp as tddp,
    )

    results: List[Dict[str, Any]] = []
    for beta in betas:
        beta_f = float(beta)
        run_tag = _beta_run_tag(beta_f)
        if not force and _ckpt_exists(config, run_tag, ckpt_name):
            print(f"[mix-cal][ddp] skip beta={beta_f:g}: {run_tag}/{ckpt_name} exists")
            results.append({"beta": beta_f, "run_tag": run_tag, "status": "skipped"})
            continue
        print(f"[mix-cal][ddp] train beta={beta_f:g} -> {run_tag}")
        res = tddp.train_ddp(config, overrides={
            "beta": beta_f, "devices": devices, "data_tag": in_mix_tag,
            "run_tag": run_tag, "epochs": epochs, "seed": seed,
        })
        res = {**res, "beta": beta_f, "status": "ok"}
        results.append(res)
    return results


# =============================================================================
# Evaluate + per-beta calibration fit + selection
# =============================================================================

def evaluate_betas(
    config: Dict[str, Any],
    *,
    betas: Sequence[float],
    in_mix_tag: str,
    holdout_tag: Optional[str] = None,
    ckpt_name: str = "final.ckpt",
) -> Dict[float, Dict[str, Any]]:
    r"""Evaluate every trained $\beta$ checkpoint with :func:`mixed_eval.evaluate_mixed`.

    Per-$\beta$ evaluation is run **without** the held-out cache by default (the
    expensive extrapolation pass is only needed once, at $\beta^\star$); pass a
    ``holdout_tag`` to include it. Missing checkpoints are skipped with a notice.

    Args:
        config: The parsed config.
        betas: The $\beta$ grid.
        in_mix_tag: In-mix pooled cache tag.
        holdout_tag: Optional held-out cache tag (``None`` -> skip for speed).
        ckpt_name: Checkpoint to evaluate (``final.ckpt`` / ``best.ckpt``).

    Returns:
        ``{beta: metrics_dict}`` for every $\beta$ whose checkpoint was found.
    """
    out: Dict[float, Dict[str, Any]] = {}
    for beta in betas:
        beta_f = float(beta)
        run_tag = _beta_run_tag(beta_f)
        if not _ckpt_exists(config, run_tag, ckpt_name):
            print(f"[mix-cal] eval skip beta={beta_f:g}: no {run_tag}/{ckpt_name}")
            continue
        print(f"[mix-cal] evaluate beta={beta_f:g} ({run_tag}) ...")
        metrics = me.evaluate_mixed(
            config, run_tag=run_tag, in_mix_tag=in_mix_tag,
            holdout_tag=holdout_tag, ckpt_name=ckpt_name,
        )
        out[beta_f] = metrics
    return out


def _slice_fit(slices: Dict[str, Any], key: str, sub: Optional[str] = None
               ) -> Optional[Dict[str, float]]:
    """Return one ``{alpha,gamma,r2}`` fit from a calibration slices dict, or None."""
    block = slices.get(key)
    if not isinstance(block, dict):
        return None
    if sub is None:
        return block if block else None
    val = block.get(sub)
    return val if isinstance(val, dict) and val else None


def fit_per_beta_calibration(
    metrics_by_beta: Dict[float, Dict[str, Any]],
    *,
    use_nullsub: bool = False,
) -> Dict[str, Any]:
    r"""Collate the per-$\beta$ calibration slopes that ``mixed_eval`` already fit.

    No re-fitting: :func:`mixed_eval.evaluate_mixed` already fits
    $\bar K = \alpha + \gamma\,\mathrm{TE}$ overall, per-$M$ and per-band. This
    transposes those into per-$\beta$ tables and builds the **primary** per-$M$
    selection table whose score is the per-$M$-robust

    $$
    s(\beta) \;=\; \operatorname{mean}_M |\gamma_M - 1|
        \;+\; \lambda_\alpha \operatorname{mean}_M |\alpha_M| ,
    $$

    which (unlike a single pooled slope) cannot hide an $M$ with $\gamma\!\ll\!1$
    behind another with $\gamma\!\gg\!1$.

    Args:
        metrics_by_beta: ``{beta: metrics}`` from :func:`evaluate_betas`.
        use_nullsub: Read the null-subtracted slices
            (``calibration.in_mix_nullsub``) instead of the raw ``in_mix`` ones.

    Returns:
        ``{"per_M": {M: [rows]}, "overall": [rows], "by_band": {band: [rows]},
        "pooled_M": [rows]}``; each ``row`` carries ``beta`` plus the fit keys.
    """
    in_key = "in_mix_nullsub" if use_nullsub else "in_mix"
    per_M: Dict[str, List[Dict[str, Any]]] = {}
    overall: List[Dict[str, Any]] = []
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    pooled_M: List[Dict[str, Any]] = []

    for beta in sorted(metrics_by_beta):
        slices = (metrics_by_beta[beta].get("calibration", {}) or {}).get(in_key, {})
        if not isinstance(slices, dict):
            continue
        ov = _slice_fit(slices, "overall")
        if ov is not None:
            overall.append({"beta": beta, **ov})
        by_M = slices.get("by_M", {}) or {}
        gamma_devs: List[float] = []
        alpha_abs: List[float] = []
        gammas: List[float] = []
        r2s: List[float] = []
        for m_str, fit in by_M.items():
            if not isinstance(fit, dict) or not fit:
                continue
            per_M.setdefault(str(m_str), []).append({"beta": beta, **fit})
            g, a = float(fit.get("gamma", np.nan)), float(fit.get("alpha", np.nan))
            if math.isfinite(g) and math.isfinite(a):
                gamma_devs.append(abs(g - 1.0))
                alpha_abs.append(abs(a))
                gammas.append(g)
                r2s.append(float(fit.get("r2", np.nan)))
        for b_name, fit in (slices.get("by_band", {}) or {}).items():
            if isinstance(fit, dict) and fit:
                by_band.setdefault(str(b_name), []).append({"beta": beta, **fit})
        if gamma_devs:
            pooled_M.append({
                "beta": beta,
                "mean_abs_gamma_dev": float(np.mean(gamma_devs)),
                "mean_abs_alpha": float(np.mean(alpha_abs)),
                "mean_gamma": float(np.mean(gammas)),
                "mean_r2": float(np.nanmean(r2s)) if r2s else float("nan"),
                "n_M": int(len(gamma_devs)),
            })
    return {
        "per_M": per_M, "overall": overall,
        "by_band": by_band, "pooled_M": pooled_M,
    }


def select_beta(
    tables: Dict[str, Any], *, alpha_penalty: float = 0.05,
) -> Dict[str, Any]:
    r"""Select $\beta^\star$ from the per-$M$-robust score (primary) + report slices.

    Primary: $\beta^\star = \arg\min_\beta s(\beta)$ with
    $s(\beta) = \operatorname{mean}_M|\gamma_M-1| + \lambda_\alpha
    \operatorname{mean}_M|\alpha_M|$ (tie-break by higher mean $R^2$). The overall
    and per-band selections (via :func:`calibration.select_beta_by_calibration`)
    are reported alongside for context.

    Args:
        tables: The output of :func:`fit_per_beta_calibration`.
        alpha_penalty: $\lambda_\alpha$ in the score.

    Returns:
        ``{"beta_star", "primary", "selected_pooled_M", "selected_overall",
        "selected_by_M": {M: sel}, "rationale"}``.
    """
    pooled = tables.get("pooled_M", [])
    best: Optional[Dict[str, Any]] = None
    for row in pooled:
        gdev = float(row.get("mean_abs_gamma_dev", float("nan")))
        aabs = float(row.get("mean_abs_alpha", float("nan")))
        if not (math.isfinite(gdev) and math.isfinite(aabs)):
            continue
        score = gdev + float(alpha_penalty) * aabs
        r2 = float(row.get("mean_r2", float("nan")))
        if (
            best is None
            or score < best["score"] - 1e-12
            or (abs(score - best["score"]) < 1e-12
                and math.isfinite(r2)
                and (not math.isfinite(best.get("mean_r2", float("nan")))
                     or r2 > best["mean_r2"]))
        ):
            best = {**row, "score": score}

    selected_overall = cal.select_beta_by_calibration(
        tables.get("overall", []), alpha_penalty=alpha_penalty)
    selected_by_M = {
        m: cal.select_beta_by_calibration(rows, alpha_penalty=alpha_penalty)
        for m, rows in (tables.get("per_M", {}) or {}).items()
    }
    beta_star = float(best["beta"]) if best else float("nan")
    return {
        "beta_star": beta_star,
        "primary": "pooled_M",
        "selected_pooled_M": best or {},
        "selected_overall": selected_overall,
        "selected_by_M": selected_by_M,
        "rationale": (
            f"argmin_beta mean_M|gamma_M-1| + {alpha_penalty:g}*mean_M|alpha_M| "
            f"(tie-break by mean R^2); primary axis = per-M calibration."
        ),
    }


# =============================================================================
# Artifacts (CSV + JSON + figures)
# =============================================================================

_SUMMARY_FIELDS = [
    "beta", "slice", "alpha", "gamma", "r2", "n", "selected",
]


def _write_summary_csv(tables: Dict[str, Any], selected: Dict[str, Any], path: Path
                       ) -> None:
    """Write one row per (β, slice) with a ``selected`` flag on $\\beta^\\star$."""
    path.parent.mkdir(parents=True, exist_ok=True)
    beta_star = float(selected.get("beta_star", float("nan")))

    def _is_star(b: float) -> bool:
        return math.isfinite(beta_star) and abs(float(b) - beta_star) < 1e-12

    rows: List[Dict[str, Any]] = []
    for r in tables.get("overall", []):
        rows.append({"beta": r["beta"], "slice": "overall", "alpha": r.get("alpha"),
                     "gamma": r.get("gamma"), "r2": r.get("r2"), "n": r.get("n"),
                     "selected": _is_star(r["beta"])})
    for m, rs in (tables.get("per_M", {}) or {}).items():
        for r in rs:
            rows.append({"beta": r["beta"], "slice": f"M={m}", "alpha": r.get("alpha"),
                         "gamma": r.get("gamma"), "r2": r.get("r2"), "n": r.get("n"),
                         "selected": _is_star(r["beta"])})
    for b, rs in (tables.get("by_band", {}) or {}).items():
        for r in rs:
            rows.append({"beta": r["beta"], "slice": f"band={b}", "alpha": r.get("alpha"),
                         "gamma": r.get("gamma"), "r2": r.get("r2"), "n": r.get("n"),
                         "selected": _is_star(r["beta"])})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SUMMARY_FIELDS})


def _fig_gamma_vs_beta(tables: Dict[str, Any], selected: Dict[str, Any], out_dir: Path
                       ) -> None:
    r"""$\gamma(\beta)$ -- one line per $M$ (primary) + overall, with $\gamma=1$ and $\beta^\star$."""
    import matplotlib.pyplot as plt

    ps.apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    per_M = tables.get("per_M", {}) or {}
    palette = ps.PALETTE_EXTENDED
    for i, m in enumerate(sorted(per_M, key=lambda s: int(s))):
        rows = sorted(per_M[m], key=lambda r: float(r["beta"]))
        bs = [float(r["beta"]) for r in rows]
        gs = [float(r.get("gamma", np.nan)) for r in rows]
        ax.plot(bs, gs, marker="o", markersize=4, linewidth=1.3,
                color=palette[i % len(palette)], label=rf"$M={m}$")
    ov = sorted(tables.get("overall", []), key=lambda r: float(r["beta"]))
    if ov:
        ax.plot([float(r["beta"]) for r in ov],
                [float(r.get("gamma", np.nan)) for r in ov],
                marker="s", markersize=3, linewidth=1.0, linestyle="--",
                color=ps.COLOR_GRAY, label="overall")
    ax.axhline(1.0, color=ps.COLOR_BLACK, linestyle=":", linewidth=0.9)
    beta_star = float(selected.get("beta_star", float("nan")))
    if math.isfinite(beta_star):
        ax.axvline(beta_star, color=ps.COLOR_VERMILLION, linestyle="--",
                   linewidth=1.1, label=rf"$\beta^\star={beta_star:.1e}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"KL weight $\beta$")
    ax.set_ylabel(r"calibration slope $\gamma$")
    ax.set_title(r"per-$M$ calibration slope $\gamma(\beta)$ "
                 r"($\gamma\to1$ is the target)")
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=ps.FONT_LEGEND)
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "gamma_vs_beta")


def _fig_selection_score(
    tables: Dict[str, Any], selected: Dict[str, Any], out_dir: Path,
    *, alpha_penalty: float,
) -> None:
    r"""Selection score $s(\beta)$ and its two terms vs $\beta$, with $\beta^\star$ marked."""
    import matplotlib.pyplot as plt

    ps.apply_style()
    pooled = sorted(tables.get("pooled_M", []), key=lambda r: float(r["beta"]))
    if not pooled:
        return
    ap = float(alpha_penalty)
    bs = [float(r["beta"]) for r in pooled]
    gdev = [float(r.get("mean_abs_gamma_dev", np.nan)) for r in pooled]
    aabs = [float(r.get("mean_abs_alpha", np.nan)) for r in pooled]
    score = [g + ap * a for g, a in zip(gdev, aabs)]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(bs, gdev, marker="o", markersize=4, color=ps.COLOR_BLUE,
            label=r"$\operatorname{mean}_M|\gamma_M-1|$")
    ax.plot(bs, aabs, marker="^", markersize=4, color=ps.COLOR_GREEN,
            label=r"$\operatorname{mean}_M|\alpha_M|$")
    ax.plot(bs, score, marker="s", markersize=4, color=ps.COLOR_VERMILLION,
            linewidth=1.6, label=r"score $s(\beta)$")
    beta_star = float(selected.get("beta_star", float("nan")))
    if math.isfinite(beta_star):
        ax.axvline(beta_star, color=ps.COLOR_BLACK, linestyle="--",
                   linewidth=1.0, label=rf"$\beta^\star={beta_star:.1e}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"KL weight $\beta$")
    ax.set_ylabel("selection score (lower is better)")
    ax.set_title(r"$\beta$ selection: per-$M$ calibration score")
    ax.legend(loc="best", frameon=False, fontsize=ps.FONT_LEGEND)
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "selection_score_vs_beta")


def _render_figures(
    tables: Dict[str, Any], selected: Dict[str, Any], out_dir: Path,
    *, alpha_penalty: float,
) -> None:
    """Render the $\\beta$-selection figures (best-effort -- never aborts)."""
    try:
        _fig_gamma_vs_beta(tables, selected, out_dir)
    except Exception as exc:  # noqa: BLE001 -- a plot must never gate selection
        print(f"[mix-cal] figure gamma_vs_beta skipped: {exc}")
    try:
        _fig_selection_score(tables, selected, out_dir, alpha_penalty=alpha_penalty)
    except Exception as exc:  # noqa: BLE001
        print(f"[mix-cal] figure selection_score skipped: {exc}")


# =============================================================================
# Orchestrator
# =============================================================================

def run_mixed_calibration(
    config: Dict[str, Any],
    config_path: Path,
    *,
    gpus: Optional[Sequence[int]] = None,
    devices: Any = 8,
    mode: str = "task_parallel",
    betas: Optional[Sequence[float]] = None,
    build: bool = True,
    train: bool = True,
    force: bool = False,
    epochs: Optional[int] = None,
    seed: Optional[int] = None,
    alpha_penalty: Optional[float] = None,
    ckpt_name: str = "final.ckpt",
    dry_run: bool = False,
    data_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Sweep $\beta$, fit per-$M$ calibration, and select $\beta^\star$.

    Args:
        config: The parsed config (any active benchmark; this forces ``G1_mix``).
        config_path: Path to ``config_synth.yaml`` (workers reload it).
        gpus: GPU slot list for the task-parallel path.
        devices: Device spec for the DDP fallback.
        mode: ``"task_parallel"`` (default) or ``"ddp"``.
        betas: Explicit $\beta$ grid; ``None`` -> :func:`_resolve_beta_grid`.
        build: Build the in-mix + held-out pools before training.
        train: Run training; ``False`` evaluates already-trained checkpoints.
        force: Retrain even when a checkpoint exists.
        epochs, seed: Optional global overrides.
        alpha_penalty: $\lambda_\alpha$ override for the selector.
        ckpt_name: Checkpoint to evaluate.
        dry_run: Enumerate + print, then return without training/eval.
        data_dir, results_dir: Optional path overrides.

    Returns:
        ``{"betas", "selected", "tables", "out_dir", "in_mix_tag", "holdout_tag"}``.
    """
    config["experiment"]["benchmark"] = _BENCHMARK
    apply_path_overrides(config, {"data_dir": data_dir, "results_dir": results_dir})
    resolve_active_benchmark(config)

    betas = [float(b) for b in (betas if betas else _resolve_beta_grid(config))]
    if not betas:
        raise ValueError(
            "mixed_calibration: empty beta grid -- set mix_calibration.beta_grid, "
            "calibration.beta_grid, or beta_sweep.grid."
        )
    # Pin the resolved grid into the config so gpu_pool's mix_beta enumerator
    # (which re-resolves from config) trains EXACTLY the betas we evaluate -- this
    # is what makes an explicit ``--betas`` override consistent across train+eval.
    config.setdefault("mix_calibration", {})["beta_grid"] = list(betas)
    lam = _alpha_penalty(config, alpha_penalty)
    in_mix_tag = str(config["experiment"]["tag"])
    holdout_tag = in_mix_tag + md._HOLDOUT_SUFFIX
    out_dir = resolve_user_path(config["paths"]["results_dir"]) / _BENCHMARK / _OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[mix-cal] benchmark={_BENCHMARK} in_mix='{in_mix_tag}' "
        f"holdout='{holdout_tag}' mode={mode}\n"
        f"          betas={[float(b) for b in betas]} "
        f"alpha_penalty={lam:g} out_dir={out_dir}"
    )

    if build and not dry_run:
        build_pool_caches(config, force=force, holdout=True)

    if train:
        if mode == "task_parallel":
            if gpus is None:
                raise ValueError("mode='task_parallel' requires gpus=...")
            train_betas_task_parallel(
                config, config_path, betas=betas, gpus=gpus,
                build=False,  # pools already built above
                force=force, epochs=epochs, seed=seed,
                data_dir=data_dir, results_dir=results_dir, dry_run=dry_run,
            )
        elif mode == "ddp":
            if not dry_run:
                train_betas_ddp(
                    config, betas=betas, devices=devices, in_mix_tag=in_mix_tag,
                    force=force, epochs=epochs, seed=seed, ckpt_name=ckpt_name,
                )
        else:
            raise ValueError(f"unknown mode {mode!r}; expected task_parallel|ddp")

    if dry_run:
        print("[mix-cal] dry run -- no evaluation / selection performed.")
        return {"betas": [float(b) for b in betas], "selected": {},
                "tables": {}, "out_dir": str(out_dir),
                "in_mix_tag": in_mix_tag, "holdout_tag": holdout_tag}

    # Evaluate every trained checkpoint (in-mix only -- fast; the held-out
    # extrapolation pass is run once at beta* via mixed_eval directly).
    metrics_by_beta = evaluate_betas(
        config, betas=betas, in_mix_tag=in_mix_tag, holdout_tag=None,
        ckpt_name=ckpt_name,
    )
    if not metrics_by_beta:
        raise RuntimeError(
            "mixed_calibration: no beta checkpoints found to evaluate -- train "
            "first (gpu_pool --mode mix_beta or this runner with --train)."
        )

    tables = fit_per_beta_calibration(metrics_by_beta, use_nullsub=False)
    tables_nullsub = fit_per_beta_calibration(metrics_by_beta, use_nullsub=True)
    selected = select_beta(tables, alpha_penalty=lam)

    _write_summary_csv(tables, selected, out_dir / "summary.csv")
    payload = {
        "benchmark": _BENCHMARK,
        "in_mix_tag": in_mix_tag,
        "holdout_tag": holdout_tag,
        "betas": [float(b) for b in betas],
        "alpha_penalty": lam,
        "train_mode": mode,
        "ckpt_name": ckpt_name,
        "tables": tables,
        "tables_nullsub": tables_nullsub,
        "selected": selected,
        "evaluated_betas": sorted(metrics_by_beta),
    }
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _render_figures(tables, selected, out_dir, alpha_penalty=lam)

    bstar = selected.get("beta_star", float("nan"))
    sel_pool = selected.get("selected_pooled_M", {}) or {}
    print(
        f"[mix-cal] done -> {out_dir}\n"
        f"  beta* = {bstar:.3e}  "
        f"mean_M|gamma-1|={sel_pool.get('mean_abs_gamma_dev', float('nan')):.3f}  "
        f"mean_M|alpha|={sel_pool.get('mean_abs_alpha', float('nan')):.3f}\n"
        f"  next: full extrapolation eval at beta* via\n"
        f"    python -m ...mixed_eval --run-tag {_beta_run_tag(bstar)} "
        f"--in-mix-tag {in_mix_tag} --holdout-tag {holdout_tag}"
    )
    return {"betas": [float(b) for b in betas], "selected": selected,
            "tables": tables, "out_dir": str(out_dir),
            "in_mix_tag": in_mix_tag, "holdout_tag": holdout_tag}


# =============================================================================
# CLI / edit-and-run
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (every override defaults to ``None``)."""
    p = argparse.ArgumentParser(
        description="Beta-sweep + calibration-based beta selection for the "
                    "G1_mix mixed-population model."
    )
    p.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    p.add_argument("--gpus", type=str, default=None,
                   help="comma-separated GPU slot list for task-parallel mode")
    p.add_argument("--devices", type=str, default="8",
                   help="DDP device spec (mode=ddp): int / list / '0,1,...'")
    p.add_argument("--mode", type=str, default="task_parallel",
                   choices=["task_parallel", "ddp"])
    p.add_argument("--betas", type=str, default=None,
                   help="explicit beta grid, e.g. '1e-3,1e-2,1e-1,1,3'")
    p.add_argument("--no-build", action="store_true", dest="no_build",
                   help="assume the pools are already cached")
    p.add_argument("--no-train", action="store_true", dest="no_train",
                   help="evaluate already-trained checkpoints; skip training")
    p.add_argument("--force", action="store_true",
                   help="retrain cells whose checkpoint exists")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--alpha-penalty", type=float, default=None, dest="alpha_penalty")
    p.add_argument("--ckpt-name", type=str, default="final.ckpt", dest="ckpt_name")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--data-dir", type=str, default=None, dest="data_dir")
    p.add_argument("--results-dir", type=str, default=None, dest="results_dir")
    return p.parse_args(argv)


def _parse_gpus(spec: Optional[str]) -> Optional[List[int]]:
    """Parse a ``--gpus`` string into a slot list (``None`` if unset)."""
    if not spec:
        return None
    return [int(x) for x in str(spec).split(",") if str(x).strip() != ""]


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    config = load_config(args.config)
    betas = (
        [float(b) for b in str(args.betas).split(",") if b.strip() != ""]
        if args.betas else None
    )
    run_mixed_calibration(
        config, args.config,
        gpus=_parse_gpus(args.gpus), devices=args.devices, mode=args.mode,
        betas=betas, build=not args.no_build, train=not args.no_train,
        force=args.force, epochs=args.epochs, seed=args.seed,
        alpha_penalty=args.alpha_penalty, ckpt_name=args.ckpt_name,
        dry_run=args.dry_run, data_dir=args.data_dir, results_dir=args.results_dir,
    )


if __name__ == "__main__":
    CONFIG_PATH = _DEFAULT_CONFIG
    RUN_CONFIG = {
        "gpus": [0, 1, 2, 3, 4, 5, 6, 7],   # task-parallel slot list
        "devices": "8",                      # DDP fallback device spec
        "mode": "task_parallel",             # or "ddp"
        "betas": None,                       # None -> config mix_calibration.beta_grid
        "build": True,
        "train": True,
        "force": False,
        "epochs": None,
        "seed": None,
        "alpha_penalty": None,
        "ckpt_name": "final.ckpt",
        "dry_run": False,
        "data_dir": None,
        "results_dir": None,
    }

    if len(sys.argv) > 1:
        main()
    else:
        cfg = load_config(CONFIG_PATH)
        run_mixed_calibration(cfg, CONFIG_PATH, **RUN_CONFIG)
