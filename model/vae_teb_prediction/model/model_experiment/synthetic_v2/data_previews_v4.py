r"""S1-T05: model-free raw data-preview stage for ``synthetic_v4``.

Renders one raw FHR/UP overlay per cell level -- FHR in bpm (top), UP in mmHg (bottom), with a
planted-lag $D$ annotation -- so the generated waveforms can be eyeballed before any model runs.
Alongside the figures it writes ``previews_summary.json`` carrying a **coupling score** per cell:
the magnitude of the lag-$D$ cross-correlation between the decimated, z-scored UP source and FHR
target. A null cell ($B=0$) scores near zero while a strong-TE cell scores high, so null vs
strong-TE are distinguishable by a summary statistic, not just by eye.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import plot_style_v2 as ps
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.cells_v4 import (
    enumerate_cells_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.eval_v2 import (
    _fourier_decimate,
    _zscore_channel,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.raw_generators import DECIMATION
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    generate_cell_raw,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)

logger = logging.getLogger(__name__)

#: Colours (memory: blue = FHR, teal = UP); fall back gracefully if a name is absent.
_FHR_COLOR = getattr(ps, "COLOR_BLUE", "C0")
_UP_COLOR = getattr(ps, "TEAL_DARK", getattr(ps, "COLOR_TEAL_DARK", getattr(ps, "COLOR_ORANGE", "C1")))


def coupling_score(fhr_raw: np.ndarray, up_raw: np.ndarray, D: int, *, decimation: int = DECIMATION) -> float:
    r"""Mean magnitude of the lag-$D$ cross-correlation between decimated UP and FHR.

    Decimates both raw signals to the $330$-step grid, per-channel z-scores, and correlates
    $\mathrm{FHR}[t]$ against $\mathrm{UP}[t-D]$ over the valid window -- a cheap, raw-domain proxy
    for the planted coupling that is $\approx 0$ for a null cell and large for a strong-TE cell.

    Args:
        fhr_raw: FHR waveform(s) $(n, N)$.
        up_raw: UP waveform(s) $(n, N)$.
        D: The planted lag in decimated steps.
        decimation: Decimation factor (default :data:`raw_generators.DECIMATION`).

    Returns:
        The mean absolute lag-$D$ correlation over samples (in $[0, 1]$).
    """
    n_dec = fhr_raw.shape[1] // int(decimation)
    fd = _zscore_channel(_fourier_decimate(np.asarray(fhr_raw, float), n_dec))
    ud = _zscore_channel(_fourier_decimate(np.asarray(up_raw, float), n_dec))
    a = fd[:, D:]
    b = ud[:, : ud.shape[1] - D]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).mean(axis=1)
    den = np.sqrt((a * a).mean(axis=1) * (b * b).mean(axis=1)) + 1e-9
    return float(np.abs(num / den).mean())


def _plot_cell_preview(fhr: np.ndarray, up: np.ndarray, *, cell, fs: float, decimation: int,
                       out_path: Path) -> None:
    r"""Render a single FHR/UP raw overlay (bpm / mmHg) with a planted-lag annotation."""
    ps.apply_style()
    import matplotlib.pyplot as plt

    t_min = np.arange(fhr.shape[0]) / float(fs) / 60.0
    lag_s = cell.D * (decimation / float(fs))

    fig, (ax_f, ax_u) = plt.subplots(2, 1, sharex=True, figsize=(9.0, 5.0))
    ax_f.plot(t_min, fhr, color=_FHR_COLOR, lw=0.6)
    ax_f.set_ylabel("FHR (bpm)")
    ax_u.plot(t_min, up, color=_UP_COLOR, lw=0.6)
    ax_u.set_ylabel("UP (mmHg)")
    ax_u.set_xlabel("time (min)")

    kind = "NULL" if cell.is_null else f"TE_inj={cell.te_block_realised:.2f}"
    fig.suptitle(f"cell {cell.cell_id} | {kind} | planted lag D={cell.D} steps ({lag_s:.0f} s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=getattr(ps, "SAVE_DPI", 150))
    plt.close(fig)


def compute_previews(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    out_dir: Path,
    pilot: bool = True,
    n_per_cell: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""Render one preview per cell and return the per-cell coupling-score summary.

    Args:
        config: The parsed ``config_synth_v4.yaml`` tree.
        benchmark: Active benchmark key.
        out_dir: Directory to write the figures + ``previews_summary.json`` into.
        pilot: Use the ``eval.realizability.pilot`` grid (default).
        n_per_cell: Samples generated per cell (one is plotted; all feed the coupling score).
        seed: Base generation seed.

    Returns:
        ``{benchmark, render_mode, out_dir, rows}`` where each row carries ``coupling_score``.
    """
    bench = config["benchmarks"][benchmark]
    render_mode = str(bench["raw"].get("render_mode", "direct"))
    fs = float(bench["raw"]["fs"])

    ev = bench.get("eval", {}).get("realizability", {})
    pilot_cfg = ev.get("pilot", {}) if pilot else {}
    cells, _ = enumerate_cells_v4(
        config, benchmark=benchmark,
        target_te_grid=pilot_cfg.get("target_te_grid") if pilot else None,
        lag_grid=pilot_cfg.get("lag_grid") if pilot else None,
    )

    out_dir = Path(out_dir)
    rows: List[Dict[str, Any]] = []
    for cell in cells:
        out = generate_cell_raw(
            n_per_cell, B=cell.B_y_scalar, D=cell.D, config=config, benchmark=benchmark,
            seed=seed + cell.cell_id, render_mode=render_mode, te_inj=cell.te_block_realised,
        )
        fig_path = out_dir / f"preview_cell{cell.cell_id:02d}_te{cell.target_te:.1f}.png"
        _plot_cell_preview(out["fhr_raw"][0], out["up_raw"][0], cell=cell, fs=fs,
                           decimation=DECIMATION, out_path=fig_path)
        score = coupling_score(out["fhr_raw"], out["up_raw"], cell.D)
        rows.append({
            "cell_id": cell.cell_id,
            "target_te": float(cell.target_te),
            "te_inj": float(cell.te_block_realised),
            "D": int(cell.D),
            "is_null": bool(cell.is_null),
            "coupling_score": score,
            "figure": fig_path.name,
        })

    summary = {"benchmark": benchmark, "render_mode": render_mode,
               "out_dir": str(out_dir), "rows": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "previews_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_data_previews_v4(ctx: StageContextV4) -> int:
    r"""``data_previews`` stage: render per-cell raw overlays + the coupling-score summary."""
    out_dir = ctx.results_dir() / "data_previews"
    summary = compute_previews(ctx.config, benchmark=ctx.benchmark, out_dir=out_dir, pilot=ctx.pilot)
    print(f"[data_previews] wrote {len(summary['rows'])} previews to {out_dir}")
    for r in summary["rows"]:
        print(f"  cell {r['cell_id']:>2} te_inj={r['te_inj']:.2f} "
              f"coupling_score={r['coupling_score']:.4f} null={r['is_null']}")
    return 0


register_stage_v4(StageSpecV4(
    name="data_previews",
    run=run_data_previews_v4,
    order=20,
    model_dependent=False,
    fatal=True,
    help="model-free raw FHR/UP preview overlays + coupling-score summary",
))
