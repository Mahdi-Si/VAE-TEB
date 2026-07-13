r"""S1-T03: the model-free raw-TE realizability pre-flight stage for ``synthetic_v4``.

Before committing the full cache build (Sprint 2) or blaming the model for a low $\bar K$, this
stage answers "is the injected TE even *present* in the raw waveform?". Per cell it generates a
pilot batch in the configured render mode, runs the re-targeted probe
(:func:`measure_te_raw_v4`), and writes ``realizability.json`` with a per-cell
$\mathrm{TE}_{\mathrm{inj}} / \mathrm{TE}_{\mathrm{raw}} / \mathrm{frac}$ table plus a **loose,
decidable** pass/fail gate:

* signal cells ($\mathrm{TE}_{\mathrm{inj}} > 0$): $\mathrm{TE}_{\mathrm{raw}} > 0$ and the
  per-level mean $\mathrm{TE}_{\mathrm{raw}}$ is non-decreasing across the TE ladder;
* null cells ($\mathrm{TE}_{\mathrm{inj}} = 0$): $\mathrm{TE}_{\mathrm{raw}} \le$ ``null_ceiling``.

Per the S1 decision, the tight $\mathrm{frac} \in [\mathrm{lo}, \mathrm{hi}]$ band is **not** a
Sprint-1 blocker: the *observed* frac range and null ceiling are recorded into the JSON as the seed
for the Sprint-6 gate, to be tightened on the prod headline run.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cells_v4 import (
    enumerate_cells_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    generate_cell_raw,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.te_raw_v4 import (
    measure_te_raw_v4,
)

logger = logging.getLogger(__name__)

#: Default pilot batch size per cell when the config does not specify one.
_DEFAULT_N_PER_CELL = 384


def compute_realizability(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    pilot: bool = True,
    n_per_cell: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""Generate a pilot batch per cell, probe $\mathrm{TE}_{\mathrm{raw}}$, and grade the loose gate.

    Args:
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        pilot: Use the ``eval.realizability.pilot`` sub-grid (default); otherwise the full
            ``mix`` grid.
        n_per_cell: Override the per-cell sample count (else the pilot/config value).
        seed: Base generation seed (each cell offsets by its ``cell_id``).

    Returns:
        A JSON-serialisable dict ``{benchmark, render_mode, gate, constants, rows}``.
    """
    bench = config["benchmarks"][benchmark]
    render_mode = str(bench["raw"].get("render_mode", "direct"))
    ev = bench.get("eval", {}).get("realizability", {})
    null_ceiling = float(ev.get("null_ceiling", 0.05))
    seed_frac_lo = float(ev.get("frac_threshold", 0.30))
    seed_frac_hi = float(ev.get("frac_upper", 3.0))

    pilot_cfg = ev.get("pilot", {}) if pilot else {}
    te_grid = pilot_cfg.get("target_te_grid") if pilot else None
    lag_grid = pilot_cfg.get("lag_grid") if pilot else None
    n = int(n_per_cell if n_per_cell is not None
            else pilot_cfg.get("n_per_cell", _DEFAULT_N_PER_CELL))

    cells, dropped = enumerate_cells_v4(
        config, benchmark=benchmark, target_te_grid=te_grid, lag_grid=lag_grid,
    )

    rows: List[Dict[str, Any]] = []
    for cell in cells:
        out = generate_cell_raw(
            n, B=cell.B_y_scalar, D=cell.D, config=config, benchmark=benchmark,
            seed=seed + cell.cell_id, render_mode=render_mode, te_inj=cell.te_block_realised,
        )
        probe = measure_te_raw_v4(
            out["fhr_raw"], out["up_raw"], D=cell.D, render_mode=render_mode,
            config=config, benchmark=benchmark,
        )
        te_raw = float(probe["te_raw"])
        te_inj = float(cell.te_block_realised)
        frac = (te_raw / te_inj) if te_inj > 0.0 else None
        rows.append({
            "cell_id": cell.cell_id,
            "target_te": float(cell.target_te),
            "te_inj": te_inj,
            "D": int(cell.D),
            "is_null": bool(cell.is_null),
            "te_raw": te_raw if np.isfinite(te_raw) else None,
            "frac": frac,
            "snr_per_step": float(probe["snr_per_step"]),
            "n_used": int(probe["n_used"]),
            "ill_fraction": float(probe["ill_fraction"]),
        })

    gate = _grade_gate(rows, null_ceiling=null_ceiling)
    constants = _observed_constants(
        rows, seed_frac_lo=seed_frac_lo, seed_frac_hi=seed_frac_hi, null_ceiling=null_ceiling,
    )
    return {
        "benchmark": benchmark,
        "render_mode": render_mode,
        "n_per_cell": n,
        "dropped": dropped,
        "gate": gate,
        "constants": constants,
        "rows": rows,
    }


def _grade_gate(rows: List[Dict[str, Any]], *, null_ceiling: float) -> Dict[str, Any]:
    r"""Grade the loose decidable gate: signal $>0$ + monotone, null $\le$ ceiling."""
    signal = [r for r in rows if not r["is_null"]]
    nulls = [r for r in rows if r["is_null"]]

    signal_positive = all((r["te_raw"] is not None and r["te_raw"] > 0.0) for r in signal)
    null_ok = all((r["te_raw"] is None or r["te_raw"] <= null_ceiling) for r in nulls)

    # Monotone in the per-level mean te_raw across the ascending TE ladder.
    by_level: Dict[float, List[float]] = {}
    for r in signal:
        if r["te_raw"] is not None:
            by_level.setdefault(r["target_te"], []).append(r["te_raw"])
    level_means = [float(np.mean(by_level[te])) for te in sorted(by_level)]
    monotone = level_means == sorted(level_means)

    passed = bool(signal_positive and null_ok and monotone)
    return {
        "passed": passed,
        "signal_positive": bool(signal_positive),
        "null_below_ceiling": bool(null_ok),
        "monotone": bool(monotone),
        "n_signal": len(signal),
        "n_null": len(nulls),
        "level_means": level_means,
        "null_ceiling": null_ceiling,
    }


def _observed_constants(
    rows: List[Dict[str, Any]], *, seed_frac_lo: float, seed_frac_hi: float, null_ceiling: float,
) -> Dict[str, Any]:
    r"""Record the *observed* frac range and null ceiling (the seed for the S6 tight gate)."""
    fracs = [r["frac"] for r in rows if r["frac"] is not None]
    null_te = [r["te_raw"] for r in rows if r["is_null"] and r["te_raw"] is not None]
    return {
        "seed_frac_lo": seed_frac_lo,
        "seed_frac_hi": seed_frac_hi,
        "seed_null_ceiling": null_ceiling,
        "observed_frac_lo": (float(min(fracs)) if fracs else None),
        "observed_frac_hi": (float(max(fracs)) if fracs else None),
        "observed_null_te_max": (float(max(null_te)) if null_te else None),
        "note": ("Loose Sprint-1 gate; observed_* seed the tight Sprint-6 frac gate, "
                 "retuned on the prod headline run."),
    }


def run_realizability_v4(ctx: StageContextV4) -> int:
    r"""``realizability`` stage: probe every cell, print the table, write ``realizability.json``.

    Returns:
        ``0`` on a passing gate; ``1`` when the gate fails and ``eval.realizability.fatal`` is set.
    """
    config = ctx.config
    benchmark = ctx.benchmark
    report = compute_realizability(config, benchmark=benchmark, pilot=ctx.pilot)

    out_dir = ctx.results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "realizability.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    tmp_path.replace(out_path)

    _print_table(report)
    print(f"[realizability] wrote {out_path}")

    fatal = bool(config["benchmarks"][benchmark].get("eval", {})
                 .get("realizability", {}).get("fatal", False))
    if not report["gate"]["passed"] and fatal:
        print("[realizability] GATE FAILED (fatal=true)")
        return 1
    return 0


def _print_table(report: Dict[str, Any]) -> None:
    r"""Print a compact per-cell te_inj / te_raw / frac table plus the gate verdict."""
    print(f"\n[realizability] benchmark={report['benchmark']} render_mode={report['render_mode']} "
          f"n_per_cell={report['n_per_cell']}")
    print(f"{'cell':>4} {'te_inj':>8} {'D':>3} {'te_raw':>9} {'frac':>7} {'snr/step':>9} {'null':>5}")
    for r in report["rows"]:
        frac = "-" if r["frac"] is None else f"{r['frac']:.3f}"
        te_raw = "nan" if r["te_raw"] is None else f"{r['te_raw']:.4f}"
        print(f"{r['cell_id']:>4} {r['te_inj']:>8.3f} {r['D']:>3} {te_raw:>9} {frac:>7} "
              f"{r['snr_per_step']:>9.4f} {str(r['is_null']):>5}")
    g = report["gate"]
    print(f"[realizability] GATE {'PASS' if g['passed'] else 'FAIL'} "
          f"(signal>0={g['signal_positive']}, null<=ceil={g['null_below_ceiling']}, "
          f"monotone={g['monotone']})")


register_stage_v4(StageSpecV4(
    name="realizability",
    run=run_realizability_v4,
    order=10,
    model_dependent=False,
    fatal=True,
    help="model-free raw-TE realizability pre-flight (writes realizability.json)",
))
