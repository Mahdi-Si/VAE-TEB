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
    * ``results/<bench>/eval_te/metrics.json``      -- Metrics 1-4 (Phase 4).
    * ``results/<bench>/lag_recovery/metrics.json`` -- lag recovery (Phase 5 /
      task 7.3).
    * ``results/<bench>/beta_sweep/analysis.json``  -- rate-distortion (Phase 6).
    * ``results/directionality/metrics.json``       -- directionality (task 7.4).
    * ``results/B/rho_null/rho_null.json``          -- Benchmark-B rho-null.

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
from typing import Any, Dict, List, Optional

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)

# ``synthetic/`` package dir and its parent ``model_experiment/``.
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Benchmarks reported, in order. (G feeds the directionality block instead of
# getting its own metric column; D is out of Phase-7 scope.)
_BENCHMARKS = ("A", "B", "C", "E")

_TABLE_FIELDS = ["benchmark", "metric", "value", "criterion", "status"]


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


def _status(value: Any, ok: Optional[bool]) -> str:
    """Map a measured value + pass flag to a PASS / FAIL / DEFERRED status.

    Args:
        value: The measured value (``None`` / non-finite -> the run is deferred).
        ok: The pass/fail flag, or ``None`` when it cannot be decided.

    Returns:
        ``"DEFERRED"``, ``"PASS"`` or ``"FAIL"``.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "DEFERRED"
    if ok is None:
        return "DEFERRED"
    return "PASS" if ok else "FAIL"


# =============================================================================
# Collation
# =============================================================================

def _collate(results_root: Path) -> Dict[str, Any]:
    """Scan ``results_root`` for every phase artifact.

    Args:
        results_root: The ``results/`` directory to scan.

    Returns:
        A nested dict ``{benchmarks: {<b>: {eval_te, lag_recovery,
        beta_sweep}}, directionality, rho_null}`` with ``None`` for any
        artifact that has not been produced yet.
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
        }
    return {
        "benchmarks": per_bench,
        "directionality": _load_json(
            results_root / "directionality" / "metrics.json"
        ),
        "rho_null": _load_json(
            results_root / "B" / "rho_null" / "rho_null.json"
        ),
    }


def _benchmark_rows(b: str, art: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the report-table rows for one benchmark.

    Args:
        b: Benchmark identifier.
        art: The collated ``{eval_te, lag_recovery, beta_sweep}`` artifacts.

    Returns:
        A list of ``_TABLE_FIELDS``-keyed rows.
    """
    ev = art.get("eval_te")
    lag = art.get("lag_recovery")
    beta = art.get("beta_sweep")
    rows: List[Dict[str, Any]] = []

    def add(metric: str, value: Any, criterion: str, ok: Optional[bool]) -> None:
        rows.append({
            "benchmark": b, "metric": metric, "value": value,
            "criterion": criterion, "status": _status(value, ok),
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

    # Lag recovery -- attention lag-mass ratio + LOLO / two-band.
    lag_ratio = _dig(lag, "task_5_2_lag_mass_attn", "ratio_to_uniform")
    add("lag_mass_ratio_to_uniform", lag_ratio, "> 1 (concentration)",
        (lag_ratio > 1.0) if _is_num(lag_ratio) else None)
    if b == "E":
        tb = _dig(lag, "task_7_3_two_band", "ratio_error")
        add("two_band_ratio_error", tb, "small (mass ratio ~ TE ratio)",
            (tb < 0.5) if _is_num(tb) else None)
    else:
        lolo = _dig(lag, "task_5_4_lolo", "lag_mass_lolo")
        add("lag_mass_lolo", lolo, "> 0.8",
            (lolo > 0.8) if _is_num(lolo) else None)

    # Beta sweep -- the selected rate-distortion beta (informational).
    sel_beta = _dig(beta, "analysis", "selected_beta")
    if sel_beta is None:
        sel_beta = _dig(beta, "selected_beta")
    add("selected_beta", sel_beta, "rate-distortion pick", None)
    return rows


def _claim_tier(table: List[Dict[str, Any]],
                directionality: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Derive the manuscript claim tier from the report-table statuses.

    Per ``model_validation.md`` Section 11-12: *strong* = the null,
    monotonicity, lag, directionality and residual-usefulness criteria all
    hold; *moderate* = null, monotonicity and residual-usefulness hold but lag
    / calibration are weak; *weak* = only $K$ responds to source presence.

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
    }
    decided = [v for v in crit.values() if v is not None]
    if not decided:
        tier = "deferred"
    elif all(crit[k] for k in
             ("null", "monotonicity", "lag", "directionality",
              "residual_usefulness") if crit[k] is not None) and len(decided) >= 4:
        tier = "strong"
    elif all(crit[k] for k in ("null", "monotonicity", "residual_usefulness")
             if crit[k] is not None):
        tier = "moderate"
    else:
        tier = "weak"
    return {
        "criteria": crit,
        "tier": tier,
        "note": (
            "Claim tier per model_validation.md Section 11-12. 'deferred' "
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


def _make_headline(collated: Dict[str, Any], out_dir: Path) -> None:
    r"""Render the 4-panel headline figure (null / rank / lag / $\beta$-curve).

    Every panel tolerates missing inputs -- a deferred phase yields a panel with
    a "not run" placeholder rather than a crash. The figure uses the shared
    publication style in :mod:`plot_style`.

    Args:
        collated: The :func:`_collate` output.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()

    def _bench_color(idx: int) -> str:
        """Pick a stable palette colour for the ``idx``-th benchmark."""
        return ps.PALETTE_EXTENDED[idx % len(ps.PALETTE_EXTENDED)]

    per_bench = collated["benchmarks"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Synthetic TE validation -- headline summary",
                 fontsize=ps.FONT_TITLE, fontweight="bold")

    def _empty(ax, title: str) -> None:
        """Mark a panel as not-yet-run."""
        ax.text(0.5, 0.5, "not run / deferred", ha="center", va="center",
                transform=ax.transAxes, fontsize=ps.FONT_LEGEND,
                color=ps.COLOR_GRAY)
        ax.set_title(title)
        ps.style_axes(ax)

    # --- Panel 1: null -- E_0 and shuffled-source K_bar per benchmark ------
    ax = axes[0, 0]
    labels, e0s, shuf = [], [], []
    for b in _BENCHMARKS:
        m1 = _dig(per_bench[b], "eval_te", "metrics", "metric1_null")
        if m1:
            labels.append(b)
            e0s.append(m1.get("E_0", np.nan))
            shuf.append(m1.get("k_bar_shuffled_mean", np.nan))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.2, e0s, width=0.4, color=ps.COLOR_BLUE,
               label=r"$E_0$ (null $\bar K$)")
        ax.bar(x + 0.2, shuf, width=0.4, color=ps.COLOR_ORANGE,
               label=r"shuffled-source $\bar K$")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"$\bar K$ (nats)")
        ax.set_title("Null controls")
        ax.legend()
        ps.style_axes(ax)
    else:
        _empty(ax, "Null controls")

    # --- Panel 2: rank -- K_bar vs te_true across benchmarks ---------------
    ax = axes[0, 1]
    plotted = False
    for idx, b in enumerate(_BENCHMARKS):
        rows = _dig(per_bench[b], "eval_te", "rows") or []
        te = [r.get("te_true") for r in rows if _is_num(r.get("te_true"))]
        kb = [r.get("k_bar") for r in rows if _is_num(r.get("te_true"))]
        if te:
            ax.scatter(te, kb, s=38, color=_bench_color(idx),
                       edgecolors=ps.COLOR_BLACK, linewidths=0.4,
                       zorder=3, label=f"{b}")
            plotted = True
    if plotted:
        ax.set_xlabel("analytic block TE (nats)")
        ax.set_ylabel(r"$\bar K$ (nats)")
        ax.set_title(r"Monotonicity: $\bar K$ vs true TE")
        ax.legend()
        ps.style_axes(ax)
    else:
        _empty(ax, r"Monotonicity: $\bar K$ vs true TE")

    # --- Panel 3: lag -- attention lag-mass ratio per benchmark ------------
    ax = axes[1, 0]
    labels, ratios = [], []
    for b in _BENCHMARKS:
        r = _dig(per_bench[b], "lag_recovery", "task_5_2_lag_mass_attn",
                 "ratio_to_uniform")
        if _is_num(r):
            labels.append(b)
            ratios.append(r)
    if labels:
        ax.bar(labels, ratios, color=ps.COLOR_GREEN)
        ax.axhline(1.0, color=ps.COLOR_GRAY, ls="--", lw=0.9,
                   label="uniform baseline")
        ax.set_ylabel("lag-mass ratio to uniform")
        ax.set_title("Lag concentration")
        ax.legend()
        ps.style_axes(ax)
    else:
        _empty(ax, "Lag concentration")

    # --- Panel 4: beta-curve -- rate-distortion from the beta sweep --------
    ax = axes[1, 1]
    plotted = False
    for idx, b in enumerate(_BENCHMARKS):
        curve = _dig(per_bench[b], "beta_sweep", "analysis",
                     "rate_distortion_curve")
        if curve:
            betas = [c["beta"] for c in curve]
            kbar = [c["k_bar"] for c in curve]
            ax.plot(betas, kbar, marker="o", color=_bench_color(idx),
                    label=f"{b}")
            plotted = True
    if plotted:
        ax.set_xscale("log")
        ax.set_xlabel(r"bottleneck coefficient $\beta$")
        ax.set_ylabel(r"$\bar K$ (nats)")
        ax.set_title(r"Rate-distortion: $\bar K$ vs $\beta$")
        ax.legend()
        ps.style_axes(ax)
    else:
        _empty(ax, r"Rate-distortion: $\bar K$ vs $\beta$")

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
        root = (_EXPERIMENT_DIR / str(config["paths"]["results_dir"])).resolve()
    out_dir = root / "final_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[final-report] scanning {root}")
    collated = _collate(root)

    table: List[Dict[str, Any]] = []
    for b in _BENCHMARKS:
        table.extend(_benchmark_rows(b, collated["benchmarks"][b]))
    # The global directionality row.
    ratio = _dig(collated["directionality"], "comparison",
                 "directionality_ratio")
    direction_ok = _dig(collated["directionality"], "comparison",
                        "verdict_direction_specific")
    table.append({
        "benchmark": "G", "metric": "directionality_ratio", "value": ratio,
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
            }
            for b in _BENCHMARKS
        },
        "directionality_present": collated["directionality"] is not None,
        "rho_null_present": collated["rho_null"] is not None,
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
