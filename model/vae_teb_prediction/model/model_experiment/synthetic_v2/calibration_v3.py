r"""Sprint 5 (goal G-E): predictive calibration of the learned observation variance.

Under ``sigma_obs: learned`` the decoder emits ``logvar_full``, so the forecast is a per-element
Gaussian $\mathcal{N}(\mu_{\mathrm{full}}, \sigma^2_{\mathrm{full}})$ rather than a bare point
estimate. That distribution can be scored *as a distribution*: negative log-likelihood, CRPS,
central-interval coverage, and probability-integral-transform reliability by lead time.

Why this belongs beside the $\bar K$ calibration. Nothing in the transfer-entropy surrogate
$\bar K$ checks whether the model's **uncertainty** is honest. A variance-collapsed decoder can
post an excellent MSE and a confident-looking $K$ while its $95\%$ intervals cover $60\%$ of the
truth. Coverage and reliability are the diagnostics that catch that; $\bar K$ never will.

Two things make this module small. First, every kernel already ships in
``model/vae_teb_prediction/testing/`` and is consumed **read-only** -- this is a bridge, not a
build. Second, the stage/report-section registries landed in Sprint 4, so nothing here edits
``run_pipeline_v2.main()`` or ``final_report_v2``'s section order.

What *is* new is the synthetic-only extension in :func:`collect_calibration_by_te`: because the
injected transfer entropy is known per cell, calibration can be **stratified by
$\mathrm{TE}_{\mathrm{inj}}$**. If the learned variance is honest, nominal coverage holds across
TE levels. Systematic over-confidence exactly where the coupling is strong would be a finding --
and on the $\mathrm{TE}_{\mathrm{inj}} = 0$ null cells, where the model is a pure baseline
forecaster, coverage must sit at nominal or the variance head is simply miscalibrated.

Note on the key name. The split's ``metrics.json`` already carries a ``calibration`` block: the
$\bar K = \alpha + \gamma\,\mathrm{TE}$ fit. This module's scalar summary is folded in under
``calibration_predictive``. The collision is a real trap, and the two blocks measure entirely
different things.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import _jsonable
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
    SectionContext,
    SectionSpec,
    _fmt,
    register_section,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
    StageContext,
    StageSpec,
    _build_runner_and_loader,
    _split_dir,
    register_stage,
)

#: Registry order. Sequences this stage among the *other plugins* only; the dict driver pins
#: every plugin between ``test_plots`` and ``report``.
_STAGE_ORDER = 12
_SECTION_ORDER = 20

_DEFAULT_LEVELS: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
_DEFAULT_N_BINS = 20
_DEFAULT_MAX_SAMPLES = 2000
#: The nominal level whose coverage is gated on the ``te_inj = 0`` cells, and its tolerance.
_DEFAULT_TE_COVERAGE_LEVEL = 0.9
_DEFAULT_TE_COVERAGE_TOL = 0.10
#: Fallback rounding when no nominal TE grid is available (see :func:`_snap_to_grid`).
_TE_DECIMALS = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _calibration_cfg(config: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    r"""Read ``benchmarks.<benchmark>.eval.calibration``, falling back to the defaults."""
    bench = (config.get("benchmarks") or {}).get(benchmark) or {}
    cfg = ((bench.get("eval") or {}).get("calibration")) or {}
    return {
        "levels": tuple(float(x) for x in cfg.get("levels", _DEFAULT_LEVELS)),
        "n_bins": int(cfg.get("n_bins", _DEFAULT_N_BINS)),
        "max_samples": cfg.get("max_samples", _DEFAULT_MAX_SAMPLES),
        "te_coverage_level": float(
            cfg.get("te_coverage_level", _DEFAULT_TE_COVERAGE_LEVEL)
        ),
        "te_coverage_tol": float(cfg.get("te_coverage_tol", _DEFAULT_TE_COVERAGE_TOL)),
    }


def _te_grid(config: Dict[str, Any], benchmark: str) -> Optional[List[float]]:
    r"""The nominal injected-TE grid ``benchmarks.<benchmark>.mix.target_te_grid``, or ``None``."""
    bench = (config.get("benchmarks") or {}).get(benchmark) or {}
    grid = (bench.get("mix") or {}).get("target_te_grid")
    return [float(x) for x in grid] if grid else None


def _snap_to_grid(te: float, grid: Optional[Sequence[float]]) -> float:
    r"""Snap a *realised* per-cell TE onto the nominal grid level it was solved for.

    ``build_dataset_v2`` stamps ``te_true`` with ``te_block_realised``, the TE the solved
    coupling actually achieves -- not the grid target. Three cells nominally at
    :math:`\mathrm{TE}_{\mathrm{inj}} = 0.5` (one per lag :math:`D \in \{8, 12, 20\}`) therefore
    carry ``0.499121``, ``0.499343`` and ``0.500882``. Grouping on the raw float would split
    every level into one row per lag and defeat the stratification, which is meant to pool the
    lags and vary only the coupling strength.

    Args:
        te: The realised per-sample TE.
        grid: The nominal grid, or ``None`` to fall back to rounding.

    Returns:
        The nearest nominal level, or ``round(te, _TE_DECIMALS)`` when no grid is configured.
    """
    if not grid:
        return round(float(te), _TE_DECIMALS)
    return float(min(grid, key=lambda g: abs(float(te) - g)))


def _fold_metrics_json(split_dir: Path, key: str, block: Dict[str, Any]) -> bool:
    r"""Read-modify-write ``metrics.json``, adding ``block`` under ``key``.

    Every other top-level key is preserved untouched -- in particular the existing
    ``calibration`` block, which is the $\gamma$-versus-TE fit and must not be shadowed.

    Args:
        split_dir: The per-split output directory.
        key: Top-level key to write (here, ``calibration_predictive``).
        block: The JSON-able scalar summary.

    Returns:
        ``True`` when an existing ``metrics.json`` was extended, ``False`` when a new one was
        created (i.e. the split had not been graded by ``--stage eval`` yet).
    """
    path = split_dir / "metrics.json"
    existed = path.is_file()
    data: Dict[str, Any] = {}
    if existed:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    data[key] = _jsonable(block)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    if not existed:
        logger.warning(
            "calibration: {} did not exist; created it with only the {!r} block. Run "
            "--stage eval first for the full grading.", path, key
        )
    return existed


# ---------------------------------------------------------------------------
# The one kernel testing/ does not ship: coverage resolved by horizon
# ---------------------------------------------------------------------------
def coverage_by_horizon(
    mu_full: torch.Tensor,
    logvar_full: torch.Tensor,
    y_plus: torch.Tensor,
    warmup: int,
    horizon: int,
    *,
    levels: Sequence[float] = _DEFAULT_LEVELS,
) -> torch.Tensor:
    r"""Empirical central-interval coverage per sample, per horizon step, per nominal level.

    ``testing.metrics.compute_interval_coverage`` pools coverage over the horizon axis and
    returns $(B, n_{\mathrm{levels}})$. The TE-stratified table is keyed
    ``(te_level, horizon, level)``, so the horizon axis has to survive. This reduction is the
    same one, stopped one axis early:

    $$c_{b,h,j} \;=\; \frac{1}{|\mathcal{T}|\,C}\sum_{t \in \mathcal{T}}\sum_{c=1}^{C}
      \mathbb{1}\!\left[\,\left|\frac{y_{b,t,h,c} - \mu_{b,t,h,c}}{\sigma_{b,t,h,c}}\right|
      \;\le\; \Phi^{-1}\!\left(\tfrac{1 + p_j}{2}\right)\right],$$

    with $\mathcal{T} = [\text{warmup},\, T - H_d)$ from
    :meth:`TestRunner.valid_anchor_range` -- the *public* accessor for the same slice
    ``testing.metrics._valid_triplet`` applies internally. Averaging the result over $h$
    reproduces the shipped ``coverage`` exactly, which is what ``test_calibration_v3`` asserts;
    that identity is the only thing pinning this derivation to the shipped kernel.

    Args:
        mu_full: Forecast mean $(B, T, H_d, C)$.
        logvar_full: Forecast observation log-variance $(B, T, H_d, C)$.
        y_plus: Unfolded future target $(B, T - H_d, H_d, C)$.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon $H_d$.
        levels: Nominal central-interval levels, each strictly inside $(0, 1)$.

    Returns:
        $(B, H_d, n_{\mathrm{levels}})$ coverage tensor.

    Raises:
        ValueError: If any level lies outside $(0, 1)$.
    """
    if any(not (0.0 < float(p) < 1.0) for p in levels):
        raise ValueError(f"levels must lie strictly inside (0, 1), got {tuple(levels)}")

    T = int(mu_full.shape[1])
    H_d = int(horizon)
    start = min(int(warmup), max(T - H_d, 0))
    end = max(T - H_d, 0)

    mu = mu_full[:, start:end]
    sigma = torch.exp(0.5 * logvar_full[:, start:end])
    y = y_plus[:, start:end]

    B = int(mu_full.shape[0])
    nominal = torch.tensor(
        [float(p) for p in levels], device=mu_full.device, dtype=mu_full.dtype
    )
    if mu.shape[1] == 0:
        return torch.zeros(B, H_d, len(levels), device=mu_full.device, dtype=mu_full.dtype)

    abs_z = ((y - mu) / sigma).abs()                                # (B, T_v, H_d, C)
    z_crit = 1.4142135623730951 * torch.erfinv(nominal)             # (n_levels,)
    inside = abs_z.unsqueeze(-1) <= z_crit.view(1, 1, 1, 1, -1)     # (B,T_v,H_d,C,n_levels)
    return inside.to(mu.dtype).mean(dim=(1, 3))                     # (B, H_d, n_levels)


# ---------------------------------------------------------------------------
# S5-T02: the TE-stratified pass
# ---------------------------------------------------------------------------
def collect_calibration_by_te(
    runner: Any,
    loader: Any,
    max_samples: Optional[int] = None,
    *,
    levels: Sequence[float] = _DEFAULT_LEVELS,
    te_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    r"""Score NLL / CRPS / sharpness / coverage, grouped by the injected TE level.

    Groups on ``batch.te_true`` -- the per-sample cell $\mathrm{TE}_{\mathrm{inj}}$ stamped by
    ``build_dataset_v2`` -- snapped onto ``te_grid`` by :func:`_snap_to_grid`, so the three lags
    sharing a nominal level pool into one row. NLL, CRPS and sharpness come straight from the
    shipped kernels; coverage comes from :func:`coverage_by_horizon`, which is pinned to the
    shipped kernel by test.

    **Clean-window convention.** The shipped kernels score anchors
    $[\text{warmup},\, T - H_d)$, whereas ``eval_v2.collect_per_sample_kbar`` scores
    $[\max(\text{warmup},\, D - 1),\, T - H_d)$. The two coincide whenever
    $\text{warmup} \ge \max_b D_b - 1$, which holds for this benchmark
    ($\text{warmup} = 30$, $D \le 20$). That precondition is checked at run time and warned
    about rather than assumed, so the conventions can never silently diverge.

    Args:
        runner: The configured ``TestRunner``.
        loader: Dataloader over the evaluation subset.
        max_samples: Cap on samples consumed; ``None`` consumes all.
        levels: Nominal central-interval levels.
        te_grid: The nominal injected-TE grid to snap onto; ``None`` falls back to rounding.

    Returns:
        A dict with ``by_te`` (long-format :class:`pandas.DataFrame` keyed
        ``(te_level, horizon, level)``), ``per_te_summary`` (horizon-pooled scalars per TE
        level), ``n_samples``, ``levels`` and ``warmup_ok``.

    Raises:
        RuntimeError: If the model emits no ``logvar_full`` (a fixed-``sigma_obs`` checkpoint).
    """
    from model.vae_teb_prediction.testing.metrics import compute_crps, compute_nll

    levels = tuple(float(p) for p in levels)
    n_lv = len(levels)

    # Per-TE-level accumulators, all keyed by the rounded te_true.
    counts: Dict[float, int] = {}
    nll_sum: Dict[float, np.ndarray] = {}
    crps_sum: Dict[float, np.ndarray] = {}
    sharp_sum: Dict[float, np.ndarray] = {}
    cov_sum: Dict[float, np.ndarray] = {}
    cells: Dict[float, set] = {}
    te_realised: Dict[float, float] = {}

    max_delay = 0
    n_total = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            logvar_full = outputs.get("logvar_full")
            if logvar_full is None:
                raise RuntimeError(
                    "model emitted no 'logvar_full'; TE-stratified calibration needs the "
                    "decoder's observation log-variance head."
                )
            mu_full = outputs["mu_full"]
            y_plus = runner.build_future_target(batch)
            warmup, horizon = runner.warmup_steps, runner.horizon

            nll = compute_nll(mu_full, logvar_full, y_plus, warmup, horizon)
            crps = compute_crps(mu_full, logvar_full, y_plus, warmup, horizon)
            cov = coverage_by_horizon(
                mu_full, logvar_full, y_plus, warmup, horizon, levels=levels
            )                                                       # (B, H_d, n_levels)
            sharp = _sharpness_by_horizon(logvar_full, warmup, horizon)  # (B, H_d)

            te = _to_numpy_1d(getattr(batch, "te_true", None), len(mu_full))
            cid = _to_numpy_1d(getattr(batch, "cell_id", None), len(mu_full))
            delay = getattr(batch, "delay", None)
            if delay is not None:
                max_delay = max(max_delay, int(torch.as_tensor(delay).max()))

            nll_h = nll["nll_per_horizon"].detach().cpu().numpy()
            crps_h = crps["crps_per_horizon"].detach().cpu().numpy()
            cov_h = cov.detach().cpu().numpy()
            sharp_h = sharp.detach().cpu().numpy()

            for i in range(int(mu_full.shape[0])):
                key = _snap_to_grid(float(te[i]), te_grid)
                if key not in counts:
                    H_d = int(nll_h.shape[1])
                    counts[key] = 0
                    nll_sum[key] = np.zeros(H_d, dtype=np.float64)
                    crps_sum[key] = np.zeros(H_d, dtype=np.float64)
                    sharp_sum[key] = np.zeros(H_d, dtype=np.float64)
                    cov_sum[key] = np.zeros((H_d, n_lv), dtype=np.float64)
                    cells[key] = set()
                    te_realised[key] = 0.0
                counts[key] += 1
                nll_sum[key] += nll_h[i]
                crps_sum[key] += crps_h[i]
                sharp_sum[key] += sharp_h[i]
                cov_sum[key] += cov_h[i]
                te_realised[key] += float(te[i])
                if cid is not None:
                    cells[key].add(int(cid[i]))
                n_total += 1

    warmup_ok = int(runner.warmup_steps) >= max_delay - 1
    if not warmup_ok:
        logger.warning(
            "calibration: warmup={} < max(delay)-1={}; the shipped kernels' anchor window "
            "[warmup, T-H) is WIDER than eval_v2's [max(warmup, D-1), T-H), so these numbers "
            "are not directly comparable to per_sample_eval.npz.",
            int(runner.warmup_steps), max_delay - 1,
        )

    rows: List[Dict[str, Any]] = []
    per_te: Dict[str, Dict[str, Any]] = {}
    for key in sorted(counts):
        n = float(counts[key])
        nll_m, crps_m = nll_sum[key] / n, crps_sum[key] / n
        sharp_m, cov_m = sharp_sum[key] / n, cov_sum[key] / n
        H_d = nll_m.shape[0]
        for h in range(H_d):
            for j, level in enumerate(levels):
                rows.append({
                    "te_level": key,
                    "horizon": h,
                    "level": float(level),
                    "coverage": float(cov_m[h, j]),
                    "coverage_error": float(cov_m[h, j] - level),
                    "nll": float(nll_m[h]),
                    "crps": float(crps_m[h]),
                    "sharpness": float(sharp_m[h]),
                    "n_samples": int(counts[key]),
                })
        entry: Dict[str, Any] = {
            "n_samples": int(counts[key]),
            "n_cells": len(cells[key]),
            # The nominal key is the grouping level; this is the mean TE the solved coupling
            # actually realised across the cells pooled into it.
            "te_realised_mean": float(te_realised[key] / n),
            "nll_mean": float(nll_m.mean()),
            "crps_mean": float(crps_m.mean()),
            "sharpness_mean": float(sharp_m.mean()),
        }
        for j, level in enumerate(levels):
            tag = f"coverage_{int(round(level * 100))}"
            entry[tag] = float(cov_m[:, j].mean())
            entry[f"{tag}_error"] = entry[tag] - float(level)
        per_te[f"{key:g}"] = entry

    return {
        "by_te": pd.DataFrame(rows),
        "per_te_summary": per_te,
        "n_samples": n_total,
        "levels": list(levels),
        "warmup_ok": bool(warmup_ok),
    }


def _sharpness_by_horizon(
    logvar_full: torch.Tensor, warmup: int, horizon: int
) -> torch.Tensor:
    r"""Mean predictive $\sigma$ per sample per horizon step, over the valid anchors."""
    T, H_d = int(logvar_full.shape[1]), int(horizon)
    start, end = min(int(warmup), max(T - H_d, 0)), max(T - H_d, 0)
    sigma = torch.exp(0.5 * logvar_full[:, start:end])
    if sigma.shape[1] == 0:
        return torch.zeros(
            logvar_full.shape[0], H_d, device=logvar_full.device, dtype=logvar_full.dtype
        )
    return sigma.mean(dim=(1, 3))


def _to_numpy_1d(value: Any, batch_size: int) -> Optional[np.ndarray]:
    r"""Coerce a per-sample batch field to a length-``batch_size`` float array, or ``None``."""
    if value is None:
        return None
    arr = torch.as_tensor(value).detach().cpu().numpy().reshape(-1)
    if arr.size != batch_size:
        return None
    return arr.astype(np.float64)


def _te_null_coverage_gate(
    result: Dict[str, Any], level: float, tol: float
) -> Optional[Dict[str, Any]]:
    r"""Coverage at ``level`` on the $\mathrm{TE}_{\mathrm{inj}} = 0$ cells, against ``tol``.

    On the null cells the model is a pure baseline forecaster -- there is no source signal to
    exploit -- so an honest learned variance must cover at nominal there whatever it does
    elsewhere. Returns ``None`` when the grid carries no null cell.
    """
    per_te = result.get("per_te_summary") or {}
    null_key = next((k for k in per_te if abs(float(k)) < 1e-12), None)
    if null_key is None:
        return None
    tag = f"coverage_{int(round(level * 100))}"
    empirical = per_te[null_key].get(tag)
    if empirical is None or not math.isfinite(empirical):
        return None
    return {
        "nominal": float(level),
        "empirical": float(empirical),
        "abs_error": abs(float(empirical) - float(level)),
        "tolerance": float(tol),
        "pass": bool(abs(float(empirical) - float(level)) <= float(tol)),
        "n_samples": int(per_te[null_key].get("n_samples", 0)),
    }


# ---------------------------------------------------------------------------
# S5-T01/T03: the stage
# ---------------------------------------------------------------------------
def _render_figures(
    calib_dir: Path, figures_dir: Path, summary: Dict[str, Any], level: float
) -> None:
    r"""Re-render the shipped ``testing/`` calibration plots into the split's gallery.

    ``run_calibration_analysis`` writes them under ``calibration/``; ``final_report_v2``'s
    gallery only globs ``figures/``. The visualizers are **imported, never copied**, and
    re-driven from the CSVs they just wrote. The TE-stratified heatmap is the one figure this
    pipeline adds. A plotting failure loses figures, never data.
    """
    from model.vae_teb_prediction.testing.visualizers import (
        plot_coverage_vs_nominal,
        plot_reliability_curve,
        plot_sharpness_by_horizon,
    )

    from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v2 as viz

    figures_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("per_sample.csv", plot_coverage_vs_nominal, "calibration_coverage.pdf", {}),
        ("reliability.csv", plot_reliability_curve, "calibration_reliability.pdf", {}),
        (
            "per_horizon.csv",
            plot_sharpness_by_horizon,
            "calibration_sharpness_by_horizon.pdf",
            {"constant_sigma": summary.get("constant_sigma")},
        ),
        (
            "calibration_by_te.csv",
            lambda df, path: viz.plot_calibration_by_te(df, path, level=level),
            "calibration_by_te",
            {},
        ),
    )
    for csv_name, fn, out_name, kwargs in jobs:
        src = calib_dir / csv_name
        if not src.is_file():
            continue
        try:
            fn(pd.read_csv(src), figures_dir / out_name, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a bad figure must not lose the CSVs
            logger.warning("calibration: {} failed ({})", out_name, exc)


def run_calibration_stage(ctx: StageContext) -> int:
    r"""Score the predictive distribution for one arm, on every requested split.

    Per split: build the runner and loader from the checkpoint's own ``model_kwargs``
    (S4-T04), call the shipped ``run_calibration_analysis``, run the TE-stratified pass, write
    ``calibration_by_te.csv``, render the figures into the gallery, and fold the scalar summary
    into ``metrics.json`` under ``calibration_predictive``.

    Args:
        ctx: The arm-resolved stage context.

    Returns:
        ``0`` on success. A checkpoint trained with a fixed ``sigma_obs`` yields an ``error``
        entry rather than a raise, matching the shipped graceful-skip contract.
    """
    from model.vae_teb_prediction.testing.analyses.calibration import (
        run_calibration_analysis,
    )

    cfg = _calibration_cfg(ctx.config, ctx.benchmark)
    max_samples = ctx.max_samples if ctx.max_samples is not None else cfg["max_samples"]
    run_dir = ctx.run_dir()

    for split in ctx.splits():
        runner, loader, used_split, ckpt_path = _build_runner_and_loader(
            ctx.config, benchmark=ctx.benchmark, arm=ctx.arm, ckpt=ctx.ckpt, split=split,
        )
        split_dir = _split_dir(run_dir, used_split)
        calib_dir = split_dir / "calibration"
        logger.info(
            "calibration[{}/{}]: grading {} (max_samples={})",
            ctx.arm or "-", used_split, ckpt_path.name, max_samples,
        )

        summary = run_calibration_analysis(
            runner, loader, max_samples, output_dir=calib_dir,
            levels=cfg["levels"], n_bins=cfg["n_bins"],
        )
        if "error" in summary:
            logger.warning(
                "calibration[{}/{}]: skipped -- {}", ctx.arm or "-", used_split,
                summary["error"],
            )
            _fold_metrics_json(split_dir, "calibration_predictive", summary)
            continue
        if not summary:
            # ``max_samples <= 0``: the shipped analysis declines and returns {}. Nothing to
            # stratify and nothing worth folding.
            logger.warning(
                "calibration[{}/{}]: no samples scored (max_samples={})",
                ctx.arm or "-", used_split, max_samples,
            )
            continue

        # S5-T02: the synthetic-only stratification. A second forward pass (no backward),
        # bounded by max_samples; the shipped summary above stays the canonical global one.
        strat = collect_calibration_by_te(
            runner, loader, max_samples, levels=cfg["levels"],
            te_grid=_te_grid(ctx.config, ctx.benchmark),
        )
        by_te: pd.DataFrame = strat.pop("by_te")
        calib_dir.mkdir(parents=True, exist_ok=True)
        by_te.to_csv(calib_dir / "calibration_by_te.csv", index=False)

        gate = _te_null_coverage_gate(
            strat, cfg["te_coverage_level"], cfg["te_coverage_tol"]
        )
        summary = dict(summary)
        summary["by_te"] = strat["per_te_summary"]
        summary["by_te_csv"] = str(calib_dir / "calibration_by_te.csv")
        summary["warmup_ok"] = strat["warmup_ok"]
        summary["null_cell_coverage"] = gate
        summary["n_samples_by_te"] = strat["n_samples"]

        _render_figures(
            calib_dir, split_dir / "figures", summary, cfg["te_coverage_level"]
        )
        _fold_metrics_json(split_dir, "calibration_predictive", summary)
        logger.info(
            "calibration[{}/{}]: NLL {:.4f}, CRPS {:.4f}, coverage_90 {:.3f}",
            ctx.arm or "-", used_split, summary.get("nll_mean", float("nan")),
            summary.get("crps_mean", float("nan")),
            summary.get("coverage_90", float("nan")),
        )
    return 0


# ---------------------------------------------------------------------------
# Report section
# ---------------------------------------------------------------------------
def _render_calibration_section(ctx: SectionContext) -> List[str]:
    r"""Render the "Predictive calibration" section, or ``n/a`` when the stage has not run."""
    lines = ["## Predictive calibration (G-E)", ""]
    block = (ctx.metrics or {}).get("calibration_predictive")
    if not block:
        lines += ["> n/a — `--stage calibration` has not been run for this split.", ""]
        return lines
    if "error" in block:
        lines += [f"> n/a — {block['error']}", ""]
        return lines

    lines += [
        "Scores the learned Gaussian $\\mathcal{N}(\\mu_{\\mathrm{full}}, "
        "\\sigma^2_{\\mathrm{full}})$ as a distribution. `nll_gain_over_constant` > 0 means "
        "the heteroscedastic head beats a single global $\\hat\\sigma$.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| samples | {block.get('n_samples', 'n/a')} |",
        f"| NLL (mean) | {_fmt(block.get('nll_mean'))} |",
        f"| CRPS (mean) | {_fmt(block.get('crps_mean'))} |",
        f"| sharpness $\\bar\\sigma$ | {_fmt(block.get('sharpness_mean'))} |",
        f"| constant-$\\sigma$ reference | {_fmt(block.get('constant_sigma'))} |",
        f"| `nll_gain_over_constant` | {_fmt(block.get('nll_gain_over_constant'))} |",
        f"| reliability max deviation | {_fmt(block.get('reliability_max_deviation'))} |",
    ]
    for level in (50, 80, 90, 95):
        key = f"coverage_{level}"
        if key in block:
            lines.append(
                f"| coverage @ {level}% (error) | {_fmt(block[key], '.3f')} "
                f"({_fmt(block.get(f'{key}_error'), '+.3f')}) |"
            )

    gate = block.get("null_cell_coverage")
    if gate:
        mark = "pass" if gate.get("pass") else "**FAIL**"
        lines.append(
            f"| coverage @ {gate['nominal']:.0%} on the $\\mathrm{{TE}}_{{\\mathrm{{inj}}}}=0$"
            f" cells | {_fmt(gate.get('empirical'), '.3f')} "
            f"(tol {_fmt(gate.get('tolerance'), '.2f')}) — {mark} |"
        )
    if block.get("warmup_ok") is False:
        lines.append("| clean-window convention | **mismatched** (warmup < max(D)-1) |")
    lines.append("")

    by_te = block.get("by_te") or {}
    if by_te:
        lines += [
            "Stratified by injected TE. An honest variance holds nominal coverage across "
            "levels; over-confidence concentrated where the coupling is strong is a finding.",
            "",
            "| $\\mathrm{TE}_{\\mathrm{inj}}$ | n | NLL | CRPS | $\\bar\\sigma$ | cov@90% |",
            "|---|---|---|---|---|---|",
        ]
        for key in sorted(by_te, key=lambda k: float(k)):
            row = by_te[key]
            lines.append(
                f"| {key} | {row.get('n_samples', '?')} | {_fmt(row.get('nll_mean'))} | "
                f"{_fmt(row.get('crps_mean'))} | {_fmt(row.get('sharpness_mean'))} | "
                f"{_fmt(row.get('coverage_90'), '.3f')} |"
            )
        lines.append("")
    return lines


register_stage(
    StageSpec(
        "calibration", _STAGE_ORDER, True, True, run_calibration_stage,
        fatal=False,
        help="predictive calibration: NLL / CRPS / coverage / reliability, stratified by TE",
    )
)
register_section(SectionSpec("Predictive calibration", _SECTION_ORDER, _render_calibration_section))
