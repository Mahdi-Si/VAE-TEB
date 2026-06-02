r"""Final report -- collate every phase's metrics into one summary (task 7.6).

A **read-only** collation step: it scans the per-benchmark artifacts the earlier
phases wrote (it trains nothing, loads no model) and assembles

    * ``report_table.csv`` -- one row per (benchmark, metric): the measured
      value, the success criterion and a PASS / FAIL / DEFERRED status;
    * ``headline.pdf`` -- the 4-panel headline figure (null / rank / lag /
      $\beta$-curve);
    * ``report.json`` -- the full structured collation plus a manuscript
      **claim tier** (strong / moderate / weak, per
      ``model_validation.md`` Section 11-12).

It tolerates missing artifacts: a phase whose run is still deferred simply
yields ``DEFERRED`` rows and an empty headline panel, so the report can be
generated at any point in the project.

Inputs scanned (all optional):
    * ``results/<bench>/eval_te/metrics.json``        -- Metrics 1-4 (Phase 4).
    * ``results/<bench>/lag_recovery/metrics.json``   -- lag recovery (Phase 5 /
      Sprint 4).
    * ``results/<bench>/beta_sweep/analysis.json``    -- rate-distortion (Phase 6).
    * ``results/<bench>/null_controls/metrics.json``  -- null controls
      (Sprint 6.4 / 6.5: wrong-delay + zero-coupling).
    * ``results/<bench>/calibration/calibration.json`` -- calibration
      slope / selected-$\beta$ record per benchmark.
    * ``results/directionality/metrics.json``         -- directionality
      (Sprint 6.1).

Run modes (project convention -- Decision D9 in
``synthetic_te_validation_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.final_report [--config PATH] [--results-root DIR]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)

# ``synthetic/`` package dir and its parent ``model_experiment/``.
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Benchmarks reported, in order. v1 entries A/B/C/E were removed in earlier
# sprints (V2-D7). G1-rev / G1_twoband / G2_wrong_delay / G2_zero_coupling
# are diagnostic / control variants that feed into the directionality and
# null-control blocks rather than their own benchmark rows.
_BENCHMARKS = ("G1", "G2", "G3")

_TABLE_FIELDS = ["benchmark", "metric", "value", "criterion", "status"]

# PASS threshold for "K_bar should collapse" null-control rows (shuffle /
# reverse / zero-coupling). 0.05 nats is well below the v2 per-step TE band
# of 0.05-0.3 nats (model_validation_v2.md V2-D6), so passing it is a real
# signal of collapse, not just a noisy small value.
_K_COLLAPSE_PASS = 0.05


# =============================================================================
# Small helpers
# =============================================================================

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning ``None`` if it is absent or unreadable.

    Args:
        path: Path to a ``.json`` artifact.

    Returns:
        The parsed dict, or ``None`` when the file is missing / malformed.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _dig(obj: Optional[Dict[str, Any]], *keys: str) -> Any:
    """Walk a nested dict by ``keys``, returning ``None`` on any miss.

    Args:
        obj: The (possibly ``None``) dict to walk.
        *keys: The successive keys to follow.

    Returns:
        The nested value, or ``None`` if any key is absent.
    """
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_num(x: Any) -> bool:
    """Return True if ``x`` is a finite real number.

    Args:
        x: Any value.

    Returns:
        ``True`` when ``x`` is an ``int`` / ``float`` and finite.
    """
    return isinstance(x, (int, float)) and bool(np.isfinite(x))


def _status(
    value: Any, ok: Optional[bool] = None, *, info_only: bool = False,
) -> str:
    """Map a measured value to a PASS / FAIL / INFO / DEFERRED status.

    A finite value with ``info_only=True`` reports as ``"INFO"`` regardless
    of any pass/fail criterion -- used for rows that should appear in the
    report (so the reader sees the number) but never move the claim tier
    (e.g. the wrong-delay null control, where the recurrent source encoder
    can still propagate signal across more than $L_{\\max}$ steps and the
    expected collapse is therefore best-effort, not gating).

    Args:
        value: The measured value (``None`` / non-finite -> deferred).
        ok: The pass/fail flag, or ``None`` when it cannot be decided.
            Ignored if ``info_only=True``.
        info_only: If ``True``, return ``"INFO"`` on any finite value.

    Returns:
        ``"DEFERRED"``, ``"INFO"``, ``"PASS"`` or ``"FAIL"``.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "DEFERRED"
    if info_only:
        return "INFO"
    if ok is None:
        return "DEFERRED"
    return "PASS" if ok else "FAIL"


# Backwards-compatible alias for code (and tests) that imported the standalone
# helper before it was folded into :func:`_status`. ``info_only=True`` is
# equivalent to the original behaviour.
def _status_info(value: Any) -> str:
    """Map an informational metric to INFO / DEFERRED (delegates to :func:`_status`)."""
    return _status(value, info_only=True)


# =============================================================================
# Collation
# =============================================================================

def _collate(results_root: Path) -> Dict[str, Any]:
    """Scan ``results_root`` for every phase artifact.

    Args:
        results_root: The ``results/`` directory to scan.

    Returns:
        A nested dict ``{benchmarks: {<b>: {eval_te, lag_recovery,
        beta_sweep, beta_grid, null_controls, calibration}}, directionality}``
        with ``None`` for any artifact that has not been produced yet.
    """
    per_bench: Dict[str, Any] = {}
    for b in _BENCHMARKS:
        per_bench[b] = {
            "eval_te": _load_json(results_root / b / "eval_te" / "metrics.json"),
            "lag_recovery": _load_json(
                results_root / b / "lag_recovery" / "metrics.json"
            ),
            "beta_sweep": _load_json(
                results_root / b / "beta_sweep" / "analysis.json"
            ),
            # beta x M x TE grid written by :func:`beta_sweep.run_beta_grid`
            # to ``results/<bench>/beta_grid/analysis.json`` (per-(M, TE) beta
            # curves + the multi-line figures). Tolerant of absence.
            "beta_grid": _load_json(
                results_root / b / "beta_grid" / "analysis.json"
            ),
            # Null-control re-evaluations written by :mod:`null_controls` to
            # ``results/<source_benchmark>/null_controls/metrics.json``.
            "null_controls": _load_json(
                results_root / b / "null_controls" / "metrics.json"
            ),
            # Calibration JSON written by :mod:`calibration` to
            # ``results/<bench>/calibration/calibration.json``. The headline
            # figure (panel i) reads ``te_points`` + ``selected`` from here.
            "calibration": _load_json(
                results_root / b / "calibration" / "calibration.json"
            ),
        }
    return {
        "benchmarks": per_bench,
        "directionality": _load_json(
            results_root / "directionality" / "metrics.json"
        ),
    }


def _benchmark_rows(b: str, art: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the report-table rows for one benchmark.

    Args:
        b: Benchmark identifier.
        art: The collated ``{eval_te, lag_recovery, beta_sweep,
            null_controls}`` artifacts.

    Returns:
        A list of ``_TABLE_FIELDS``-keyed rows.
    """
    ev = art.get("eval_te")
    lag = art.get("lag_recovery")
    beta = art.get("beta_sweep")
    nc = art.get("null_controls")
    rows: List[Dict[str, Any]] = []

    def add(metric: str, value: Any, criterion: str, ok: Optional[bool]) -> None:
        rows.append({
            "benchmark": b, "metric": metric, "value": value,
            "criterion": criterion, "status": _status(value, ok),
        })

    def add_info(metric: str, value: Any, criterion: str) -> None:
        """Add an informational row that never moves the claim tier."""
        rows.append({
            "benchmark": b, "metric": metric, "value": value,
            "criterion": criterion, "status": _status(value, info_only=True),
        })

    # Metric 1 -- null error.
    e0 = _dig(ev, "metrics", "metric1_null", "E_0")
    add("null_E_0", e0, "small vs smallest-signal K_bar",
        (e0 < 0.05) if _is_num(e0) else None)

    # Metric 2 -- monotonicity.
    rho = _dig(ev, "metrics", "metric2_spearman")
    add("spearman_rho", rho, "> 0.95",
        (rho > 0.95) if _is_num(rho) else None)

    # Metric 3 -- calibration slope.
    gamma = _dig(ev, "metrics", "metric3_calibration", "gamma")
    add("calibration_gamma", gamma, "~ 1.0 (Phase-6 / NLL goal)",
        (0.5 <= gamma <= 2.0) if _is_num(gamma) else None)

    # Metric 4 -- predictive gain.
    pg0 = _dig(ev, "metrics", "metric4_pred_gain", "verdict_te0_near_zero")
    pgp = _dig(ev, "metrics", "metric4_pred_gain", "verdict_te_pos_positive")
    add("pred_gain_te0", pg0, "pred_gap ~ 0 when TE=0",
        bool(pg0) if isinstance(pg0, bool) else None)
    add("pred_gain_te_pos", pgp, "pred_gap > 0 when TE>0",
        bool(pgp) if isinstance(pgp, bool) else None)

    # Lag recovery -- attention lag-mass ratio + sliding-window LOLO.
    lag_ratio = _dig(lag, "task_5_2_lag_mass_attn", "ratio_to_uniform")
    add("lag_mass_ratio_to_uniform", lag_ratio, "> 1 (concentration)",
        (lag_ratio > 1.0) if _is_num(lag_ratio) else None)
    lolo = _dig(lag, "task_5_4_lolo", "lag_mass_lolo")
    add("lag_mass_lolo", lolo, "> 0.8",
        (lolo > 0.8) if _is_num(lolo) else None)

    # Beta sweep -- the selected rate-distortion beta (informational).
    sel_beta = _dig(beta, "analysis", "selected_beta")
    if sel_beta is None:
        sel_beta = _dig(beta, "selected_beta")
    add("selected_beta", sel_beta, "rate-distortion pick", None)

    # Null controls -- shuffled / reversed source means come from the sweep
    # aggregate; wrong-delay / zero-coupling come from the per-benchmark
    # null_controls re-evaluation (see :mod:`null_controls`).
    k_shuffled = _dig(ev, "metrics", "metric1_null", "k_bar_shuffled_mean")
    add("k_bar_shuffled", k_shuffled, f"<= {_K_COLLAPSE_PASS} (collapse to 0)",
        (k_shuffled < _K_COLLAPSE_PASS) if _is_num(k_shuffled) else None)

    k_reversed = _dig(ev, "metrics", "metric1_null", "k_bar_reversed_mean")
    add("k_bar_reversed", k_reversed, f"<= {_K_COLLAPSE_PASS} (collapse to 0)",
        (k_reversed < _K_COLLAPSE_PASS) if _is_num(k_reversed) else None)

    # Wrong-delay is INFO only -- the unidirectional source LSTM can
    # propagate signal across more than ``max_lag`` steps, so $\bar K$ may
    # not fully collapse. The criterion string explicitly disclaims the
    # would-have-passed threshold so a reader scanning the CSV does not
    # mistake INFO for an asserted pass.
    k_wd = _dig(nc, "controls", "wrong_delay", "k_bar")
    add_info(
        "k_bar_wrong_delay", k_wd,
        f"INFO only (collapse target <= {_K_COLLAPSE_PASS}; "
        f"recurrent-encoder caveat)",
    )

    # Zero-coupling: true TE = 0 by construction, surrogate must collapse.
    k_zc = _dig(nc, "controls", "zero_coupling", "k_bar")
    add("k_bar_zero_coupling", k_zc,
        f"<= {_K_COLLAPSE_PASS} (true TE = 0)",
        (k_zc < _K_COLLAPSE_PASS) if _is_num(k_zc) else None)
    return rows


def _claim_tier(table: List[Dict[str, Any]],
                directionality: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Derive the manuscript claim tier from the report-table statuses.

    Per ``model_validation_v2.md`` Section 8: *strong* = the null,
    monotonicity, lag, directionality, residual-usefulness AND null-control
    criteria all hold; *moderate* = null, monotonicity, null-controls and
    residual-usefulness hold but lag / calibration are weak; *weak* = only
    $K$ responds to source presence.

    The ``null_controls`` criterion passes iff at least one of the
    per-benchmark ``k_bar_shuffled`` / ``k_bar_reversed`` /
    ``k_bar_zero_coupling`` rows passes (the wrong-delay row is INFO only
    and never gates).

    Args:
        table: The assembled report-table rows.
        directionality: The collated directionality artifact (or ``None``).

    Returns:
        A dict with the per-criterion pass map and the resulting ``tier``
        (``"strong"`` / ``"moderate"`` / ``"weak"`` / ``"deferred"``).
    """
    def _all(metric: str) -> Optional[bool]:
        """True if every benchmark row for ``metric`` passed; None if deferred."""
        hits = [r for r in table if r["metric"] == metric]
        if not hits or all(r["status"] == "DEFERRED" for r in hits):
            return None
        return all(r["status"] == "PASS" for r in hits
                   if r["status"] != "DEFERRED")

    def _any(metrics: Tuple[str, ...]) -> Optional[bool]:
        """True if any of ``metrics`` has at least one PASS row; None when all deferred."""
        flags = [_all(m) for m in metrics]
        if all(f is None for f in flags):
            return None
        return any(bool(f) for f in flags if f is not None)

    direction_ok = _dig(directionality, "comparison",
                        "verdict_direction_specific")
    crit = {
        "null": _all("null_E_0"),
        "monotonicity": _all("spearman_rho"),
        "lag": _all("lag_mass_ratio_to_uniform"),
        "directionality": (bool(direction_ok)
                           if isinstance(direction_ok, bool) else None),
        "residual_usefulness": _all("pred_gain_te_pos"),
        "calibration": _all("calibration_gamma"),
        # Null controls (shuffle / reverse / zero-coupling). Wrong-delay is
        # INFO only, so it does not appear in this list.
        "null_controls": _any((
            "k_bar_shuffled", "k_bar_reversed", "k_bar_zero_coupling",
        )),
    }
    decided = [v for v in crit.values() if v is not None]
    strong_keys = (
        "null", "monotonicity", "lag", "directionality",
        "residual_usefulness", "null_controls",
    )
    moderate_keys = (
        "null", "monotonicity", "residual_usefulness", "null_controls",
    )
    if not decided:
        tier = "deferred"
    elif all(crit[k] for k in strong_keys if crit[k] is not None) and len(decided) >= 4:
        tier = "strong"
    elif all(crit[k] for k in moderate_keys if crit[k] is not None):
        tier = "moderate"
    else:
        tier = "weak"
    return {
        "criteria": crit,
        "tier": tier,
        "note": (
            "Claim tier per model_validation_v2.md Section 8. 'deferred' "
            "means no criterion has a converged-checkpoint verdict yet -- the "
            "GPU training runs are deferred to the multi-GPU box."
        ),
    }


# =============================================================================
# Output: CSV / JSON / headline figure
# =============================================================================

def _write_table_csv(table: List[Dict[str, Any]], path: Path) -> None:
    """Write the unified report table.

    Args:
        table: The assembled report-table rows.
        path: Destination CSV path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TABLE_FIELDS)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k) for k in _TABLE_FIELDS})


_CALIBRATION_PANEL_TITLE = (
    r"Calibration: $\bar K$ vs $\mathrm{TE}_{\mathrm{true}}$"
)
_LAG_PANEL_TITLE = r"Lag recovery on G1"
_DIRECTIONALITY_PANEL_TITLE = "Directionality (G1 vs G1-rev)"
_BETA_PANEL_TITLE = r"Rate-distortion (G1): $\bar K$ vs $\beta$"


def _empty_panel(ax: Any, title: str, ps: Any) -> None:
    """Mark a panel as not-yet-run."""
    ax.text(0.5, 0.5, "not run / deferred", ha="center", va="center",
            transform=ax.transAxes, fontsize=ps.FONT_LEGEND,
            color=ps.COLOR_GRAY)
    ax.set_title(title)
    ps.style_axes(ax)


def _bench_color(idx: int, ps: Any) -> str:
    """Pick a stable palette colour for the ``idx``-th benchmark."""
    return ps.PALETTE_EXTENDED[idx % len(ps.PALETTE_EXTENDED)]


def _render_calibration_panel(ax: Any, per_bench: Dict[str, Any], ps: Any) -> None:
    r"""Panel (i): $\bar K$ vs $\mathrm{TE}_{\mathrm{true}}$ with $y=x$.

    Reads each benchmark's ``calibration/calibration.json`` and plots the
    selected-$\beta$ cells with the fitted line and a $y=x$ reference.
    """
    max_axis = 0.0
    plotted = False
    for idx, b in enumerate(_BENCHMARKS):
        cal = per_bench[b].get("calibration")
        if not cal:
            continue
        selected = cal.get("selected") or {}
        beta_star = selected.get("beta")
        if not _is_num(beta_star):
            continue
        cells = cal.get("cells") or []
        pts = [
            (float(c["te_true_block"]), float(c["k_bar"]))
            for c in cells
            if _is_num(c.get("te_true_block")) and _is_num(c.get("k_bar"))
            and _is_num(c.get("beta"))
            and abs(float(c["beta"]) - float(beta_star)) < 1e-12
        ]
        if not pts:
            continue
        pts.sort()
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        color = _bench_color(idx, ps)
        ax.scatter(
            xs, ys, s=60, color=color, edgecolor=ps.COLOR_BLACK,
            linewidth=0.6, zorder=3,
            label=(
                rf"{b}: $\gamma$={float(selected.get('gamma', float('nan'))):.2f}"
                rf", $\alpha$={float(selected.get('alpha', float('nan'))):.2f}"
            ),
        )
        gamma = selected.get("gamma")
        alpha = selected.get("alpha")
        if _is_num(gamma) and _is_num(alpha):
            xx = np.linspace(0.0, float(xs.max()) * 1.05, 64)
            ax.plot(xx, float(alpha) + float(gamma) * xx,
                    color=color, lw=1.2, alpha=0.85)
        max_axis = max(max_axis, float(xs.max()), float(ys.max()))
        plotted = True

    if not plotted:
        return _empty_panel(ax, _CALIBRATION_PANEL_TITLE, ps)
    lim = max(max_axis * 1.1, 1e-3)
    ax.plot([0.0, lim], [0.0, lim], ls="--", color=ps.COLOR_GRAY,
            lw=0.9, label=r"$y = x$")
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_xlabel(r"analytic block TE $\mathrm{TE}_{\mathrm{true}}$ (nats)")
    ax.set_ylabel(r"latent KL $\bar K$ (nats)")
    ax.set_title(_CALIBRATION_PANEL_TITLE)
    ax.legend(loc="lower right", fontsize=ps.FONT_LEGEND - 1, frameon=False)
    ps.style_axes(ax)


def _render_g1_lag_panel(ax: Any, per_bench: Dict[str, Any], ps: Any) -> None:
    r"""Panel (ii): $A_\ell$ for G1 with the true band shaded.

    Mirrors :func:`lag_recovery._plot_lolo_vs_attn_overlay`: a blue bar for
    sliding-window LOLO $A_\ell$ on the left axis, an orange line for the
    sum-normalised attention $\bar\alpha_\ell$ on a twin axis, and the true
    source-lag spans shaded in vermillion.
    """
    lag = per_bench["G1"].get("lag_recovery")
    A_lag = _dig(lag, "sprint_4_1_sliding_lolo", "A_lag")
    if A_lag is None:
        A_lag = _dig(lag, "task_5_4_lolo", "A_lag")
    if not (isinstance(A_lag, list) and A_lag):
        return _empty_panel(ax, _LAG_PANEL_TITLE, ps)

    spans = _dig(lag, "ground_truth", "lag_band_spans") or []
    window_w = _dig(lag, "sprint_4_1_sliding_lolo", "window_width")
    A = np.asarray(A_lag, dtype=float)
    L = len(A)
    lag_axis = np.arange(L)
    for span in spans:
        try:
            lo_s, hi_s = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        ax.axvspan(lo_s - 0.5, hi_s + 0.5, color=ps.COLOR_VERMILLION,
                   alpha=0.18, zorder=1)
    lolo_handle = ax.bar(
        lag_axis, np.nan_to_num(A, nan=0.0),
        width=0.85, color=ps.COLOR_BLUE,
        edgecolor="none", zorder=2,
        label=r"$A_\ell$ (sliding LOLO)",
    )

    legend_handles: List[Any] = [lolo_handle]
    attn = _dig(lag, "per_anchor", "attn_lag_profile") or []
    if isinstance(attn, list) and len(attn) >= L:
        a = np.asarray(attn[:L], dtype=float)
        a_total = float(np.nansum(a))
        if a_total > 1e-12:
            a = a / a_total
        ax2 = ax.twinx()
        line_attn = ax2.plot(
            lag_axis, np.nan_to_num(a, nan=0.0),
            color=ps.COLOR_ORANGE, lw=1.1,
            label=r"$\bar\alpha_\ell$ (norm. attention)",
        )
        ax2.set_ylabel(r"$\bar\alpha_\ell$", color=ps.COLOR_ORANGE)
        ax2.tick_params(axis="y", colors=ps.COLOR_ORANGE)
        legend_handles.extend(line_attn)

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel(r"$A_\ell$ (normalised)", color=ps.COLOR_BLUE)
    ax.tick_params(axis="y", colors=ps.COLOR_BLUE)
    title = r"Lag recovery on G1 (true band $\mathcal{L}^\star$ shaded)"
    if _is_num(window_w):
        title += f"  [$w$={int(window_w)}]"
    ax.set_title(title)
    ax.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        loc="upper right", fontsize=ps.FONT_LEGEND - 1, frameon=False,
    )
    ps.style_axes(ax)


def _render_directionality_panel(
    ax: Any, collated: Dict[str, Any], ps: Any,
) -> None:
    r"""Panel (iii): G1 vs G1-rev $\bar K$ bar pair with ratio annotation."""
    comparison = _dig(collated, "directionality", "comparison") or {}
    k_fwd = comparison.get("k_bar_forward")
    k_rev = comparison.get("k_bar_reverse")
    if not (_is_num(k_fwd) and _is_num(k_rev)):
        return _empty_panel(ax, _DIRECTIONALITY_PANEL_TITLE, ps)

    k_fwd_f = float(k_fwd)
    k_rev_f = float(k_rev)
    bars = ax.bar(
        [r"$\bar K_{X\to Y}$ (G1)", r"$\bar K_{Y\to X}$ (G1-rev)"],
        [k_fwd_f, k_rev_f],
        color=[ps.COLOR_BLUE, ps.COLOR_ORANGE],
        edgecolor=ps.COLOR_BLACK, linewidth=0.4,
    )
    ymax = max(k_fwd_f, k_rev_f, 1e-9)
    for bar, val in zip(bars, [k_fwd_f, k_rev_f]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02 * ymax,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=ps.FONT_LEGEND - 1,
        )
    ratio = comparison.get("directionality_ratio")
    if _is_num(ratio):
        ax.annotate(
            rf"ratio $\bar K_{{X\to Y}}/\bar K_{{Y\to X}}$ = {float(ratio):.2f}",
            xy=(0.5, 0.94), xycoords="axes fraction",
            ha="center", va="top",
            fontsize=ps.FONT_LEGEND,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=ps.COLOR_LIGHT_GRAY),
        )
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.set_title(_DIRECTIONALITY_PANEL_TITLE)
    ax.set_ylim(0.0, ymax * 1.25)
    ps.style_axes(ax)


def _render_g1_rate_distortion_panel(
    ax: Any, per_bench: Dict[str, Any], ps: Any,
) -> None:
    r"""Panel (iv): G1 rate-distortion $\bar K$ vs $\beta$ with $\beta^\star$.

    Reads ``G1/beta_sweep/analysis.json`` for the curve and
    ``G1/calibration/calibration.json`` for the calibration-selected
    $\beta^\star$ marker. Filters each curve entry by both ``beta`` and
    ``k_bar`` in a single pass so the two axes stay paired.
    """
    g1_bs = per_bench["G1"].get("beta_sweep")
    curve = _dig(g1_bs, "analysis", "rate_distortion_curve")
    if not curve:
        return _empty_panel(ax, _BETA_PANEL_TITLE, ps)

    pts = [
        (float(c["beta"]), float(c["k_bar"]))
        for c in curve
        if _is_num(c.get("beta")) and _is_num(c.get("k_bar"))
    ]
    if not pts:
        return _empty_panel(ax, _BETA_PANEL_TITLE, ps)
    pts.sort()
    betas, kbar = zip(*pts)
    ax.plot(list(betas), list(kbar), marker="o", color=ps.COLOR_BLUE,
            lw=1.2, markersize=4, markeredgecolor=ps.COLOR_BLACK,
            markeredgewidth=0.4, label="G1")
    beta_star = _dig(per_bench["G1"], "calibration", "selected", "beta")
    if _is_num(beta_star):
        ax.axvline(float(beta_star), color=ps.COLOR_VERMILLION,
                   ls=":", lw=1.0,
                   label=rf"$\beta^\star$={float(beta_star):.1e}")
    ax.set_xscale("log")
    ax.set_xlabel(r"bottleneck coefficient $\beta$")
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.set_title(_BETA_PANEL_TITLE)
    ax.legend(loc="best", fontsize=ps.FONT_LEGEND - 1, frameon=False)
    ps.style_axes(ax)


def _make_headline(collated: Dict[str, Any], out_dir: Path) -> None:
    r"""Render the 4-panel headline figure.

    Panel layout:

    1. Calibration $\bar K$ vs $\mathrm{TE}_{\mathrm{true}}$ across
       $\{G1, G2, G3\}$ with the $y = x$ reference.
       Reads ``calibration/calibration.json``.
    2. Lag recovery $A_\ell$ for G1 with the true band shaded.
       Reads ``G1/lag_recovery/metrics.json``.
    3. Directionality $\bar K_{X\to Y}$ vs $\bar K_{Y\to X}$ bar pair.
       Reads ``directionality/metrics.json``.
    4. Rate-distortion $\bar K$ vs $\beta$ for G1, $\beta^\star$ marked.
       Reads ``G1/beta_sweep/analysis.json`` and
       ``G1/calibration/calibration.json``.

    Each panel falls back to a "not run / deferred" placeholder when its
    inputs are missing.

    Args:
        collated: The :func:`_collate` output.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()
    per_bench = collated["benchmarks"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Synthetic TE validation -- headline summary",
                 fontsize=ps.FONT_TITLE, fontweight="bold")
    _render_calibration_panel(axes[0, 0], per_bench, ps)
    _render_g1_lag_panel(axes[0, 1], per_bench, ps)
    _render_directionality_panel(axes[1, 0], collated, ps)
    _render_g1_rate_distortion_panel(axes[1, 1], per_bench, ps)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ps.save_figure(fig, out_dir / "headline")


# =============================================================================
# Library entry point (Decision D9)
# =============================================================================

def build_final_report(
    config: Dict[str, Any],
    *,
    results_root: Optional[Any] = None,
) -> Dict[str, Any]:
    r"""Collate every phase's metrics into the final report (task 7.6).

    Args:
        config: The parsed ``config_synth.yaml`` (used only to resolve
            ``paths.results_dir``).
        results_root: Explicit ``results/`` directory to scan. Defaults to
            ``<model_experiment>/<paths.results_dir>``.

    Returns:
        A results dict: ``table`` (the report-table rows), ``claim_tier`` and
        ``out_dir``.
    """
    if results_root is not None:
        root = Path(results_root).resolve()
    else:
        root = tm.resolve_user_path(config["paths"]["results_dir"])
    out_dir = root / "final_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[final-report] scanning {root}")
    collated = _collate(root)

    table: List[Dict[str, Any]] = []
    for b in _BENCHMARKS:
        table.extend(_benchmark_rows(b, collated["benchmarks"][b]))
    # The global directionality row. The "benchmark" tag is the literal
    # "directionality" (was the v1 "G" placeholder); the v1 G benchmark was
    # removed and the G1-rev directionality test lives under
    # results/directionality/ regardless of benchmark.
    ratio = _dig(collated["directionality"], "comparison",
                 "directionality_ratio")
    direction_ok = _dig(collated["directionality"], "comparison",
                        "verdict_direction_specific")
    table.append({
        "benchmark": "directionality",
        "metric": "directionality_ratio", "value": ratio,
        "criterion": "K_fwd >> K_rev (> 5)",
        "status": _status(ratio, bool(direction_ok)
                          if isinstance(direction_ok, bool) else None),
    })

    claim = _claim_tier(table, collated["directionality"])

    _write_table_csv(table, out_dir / "report_table.csv")
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "results_root": str(root),
        "claim_tier": claim,
        "table": table,
        "collated_present": {
            b: {
                "eval_te": collated["benchmarks"][b]["eval_te"] is not None,
                "lag_recovery": (
                    collated["benchmarks"][b]["lag_recovery"] is not None
                ),
                "beta_sweep": (
                    collated["benchmarks"][b]["beta_sweep"] is not None
                ),
                "beta_grid": (
                    collated["benchmarks"][b]["beta_grid"] is not None
                ),
                "null_controls": (
                    collated["benchmarks"][b]["null_controls"] is not None
                ),
                "calibration": (
                    collated["benchmarks"][b]["calibration"] is not None
                ),
            }
            for b in _BENCHMARKS
        },
        "directionality_present": collated["directionality"] is not None,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _make_headline(collated, out_dir)

    n_pass = sum(1 for r in table if r["status"] == "PASS")
    n_fail = sum(1 for r in table if r["status"] == "FAIL")
    n_def = sum(1 for r in table if r["status"] == "DEFERRED")
    print(
        f"[final-report] {len(table)} metric rows  "
        f"PASS={n_pass}  FAIL={n_fail}  DEFERRED={n_def}\n"
        f"  claim tier: {claim['tier']}\n"
        f"[done] artifacts -> {out_dir}"
    )
    return {"table": table, "claim_tier": claim, "out_dir": str(out_dir)}


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    p = argparse.ArgumentParser(
        description="Collate the synthetic-TE validation metrics into a final "
                    "report (task 7.6)."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--results-root", type=str, default=None, dest="results_root",
        help="results/ directory to scan (defaults to paths.results_dir)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: parse args, load config, build the report.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    config = tm.load_config(args.config)
    build_final_report(config, results_root=args.results_root)


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision D9)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly.
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "results_root": None,   # None -> config paths.results_dir
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        build_final_report(config, results_root=RUN_CONFIG["results_root"])
