r"""S7-T05: the per-arm / per-split human-readable report for ``synthetic_v4``.

:func:`final_report_v4` collates the raw-domain ground-truth-grading artifacts a v4 run produces --
the evaluation gates (``metrics.json``), the per-sample arrays (``per_sample_eval.npz``), and the
model-free realizability probe (``realizability.json``) -- into a single ``report.md`` under the
arm's (split-scoped) run dir, alongside the :mod:`visualize_v4` figure gallery. The Sprint-6 ``eval``
stage emits only machine artifacts; this stage owns **all** human-readable rendering.

The report is deliberately *warn-don't-gate*: every section reads its keys defensively and a section
that raises (a missing/renamed key, a malformed artifact) degrades to an ``n/a`` note rather than
losing the whole report -- a diagnostic surface is more useful partial than absent. It mirrors the
scattering-domain :mod:`final_report_v2` structure but reads the **single-axis** v4 schema
(un-suffixed ``calibration.gamma``, top-level ``null_cell_gate`` with a ``ceiling``,
``prediction_controls.overall`` with the ``shuffled`` control, ``lag_recovery``, ``te_raw_gate``);
there is no ``te_scat`` / ``frac_Phi``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Reuse the v2 report helpers verbatim (schema-agnostic loaders + the n/a-safe formatter).
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
    _fmt,
    _gather_figures,
    _load_json,
    _load_per_sample,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    register_stage_v4,
)

logger = logging.getLogger(__name__)

_STAGE_ORDER = 55


# ---------------------------------------------------------------------------
# Section helpers (each returns markdown lines; a raiser degrades to n/a).
# ---------------------------------------------------------------------------
def _safe_section(name: str, render: Callable[[], List[str]]) -> List[str]:
    r"""Render one section, degrading a raising one to an ``n/a`` note (warn-don't-gate)."""
    try:
        return list(render())
    except Exception as exc:  # noqa: BLE001 -- one bad section never loses the report
        logger.warning("final_report_v4: section %s failed (%s)", name, exc)
        return [f"### {name}", "", f"> ⚠ n/a — section failed: {type(exc).__name__}: {exc}", ""]


def _provenance_lines(config: Dict[str, Any], benchmark: str, arm: Optional[str],
                      split: Optional[str], metrics: Optional[Dict[str, Any]]) -> List[str]:
    r"""The run-provenance bullet list."""
    m = metrics or {}
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    return [
        f"- **experiment tag**: `{tag}`",
        f"- **benchmark**: `{benchmark}`",
        f"- **arm**: `{arm if arm is not None else 'n/a'}`",
        f"- **split**: `{split if split is not None else 'n/a'}`",
        f"- **model class**: `{m.get('model_class', 'n/a')}`",
        f"- **render mode**: `{m.get('render_mode', 'n/a')}`",
        f"- **KL support**: `{m.get('kld_support', 'n/a')}`",
        f"- **samples graded**: {m.get('n_samples', 'n/a')}",
        "",
    ]


def _headline_gates_lines(metrics: Optional[Dict[str, Any]]) -> List[str]:
    r"""The four core gates + the te_raw gate as one markdown table."""
    m = metrics or {}
    cal = m.get("calibration") or {}
    null = m.get("null_cell_gate") or {}
    pred = (m.get("prediction_controls") or {}).get("overall") or {}
    lag = m.get("lag_recovery") or {}
    te_raw = (m.get("te_raw_gate") or {}).get("gate") or {}

    def _verdict(x: Any) -> str:
        return "n/a" if x is None else ("✅ pass" if bool(x) else "❌ FAIL")

    te_raw_pass = te_raw.get("passed", te_raw.get("pass")) if isinstance(te_raw, dict) else None
    rows = [
        ("calibration slope γ", _fmt(cal.get("gamma")), "γ > 0 (K̄ tracks TE_inj)"),
        ("calibration R²", _fmt(cal.get("r2")), "per-cell OLS"),
        ("calibration Spearman ρ", _fmt(cal.get("spearman")), "rank monotonicity"),
        ("null-cell K̄", _fmt(null.get("mean")),
         f"ceiling {_fmt(null.get('ceiling'))} → {_verdict(null.get('pass'))}"),
        ("pred-space ordering", _verdict(pred.get("ordering_pass_shuffled",
                                                  pred.get("ordering_pass"))),
         f"L_feat < L_base < L_feat^π(U); penalty {_fmt(pred.get('shuffle_penalty_shuffled'))}"),
        ("lag mass (signal mean)", _fmt(lag.get("mean_lag_mass")),
         f"thr {_fmt(lag.get('lag_mass_threshold'))} → {_verdict(lag.get('mean_lag_mass_pass'))}"),
        ("te_raw realizability", _verdict(te_raw_pass), "model-free preflight (S1)"),
    ]
    out = ["| gate | value | note |", "|---|---|---|"]
    out += [f"| {label} | {val} | {note} |" for label, val, note in rows]
    out.append("")
    return out


def _calibration_by_lag_lines(metrics: Optional[Dict[str, Any]]) -> List[str]:
    r"""The per-lag calibration table (``calibration.by_lag``)."""
    by_lag = ((metrics or {}).get("calibration") or {}).get("by_lag") or {}
    if not by_lag:
        return ["> n/a — no `calibration.by_lag` (run `--stage eval`).", ""]
    out = ["| lag D | γ | α | R² | n |", "|---|---|---|---|---|"]
    for d in sorted(by_lag, key=lambda k: int(float(k))):
        v = by_lag[d] or {}
        out.append(f"| {int(float(d))} | {_fmt(v.get('gamma'))} | {_fmt(v.get('alpha'))} "
                   f"| {_fmt(v.get('r2'))} | {v.get('n', 'n/a')} |")
    out.append("")
    return out


def _per_cell_lines(metrics: Optional[Dict[str, Any]]) -> List[str]:
    r"""The per-cell K̄ / TE_inj table (``calibration.per_cell``)."""
    rows = ((metrics or {}).get("calibration") or {}).get("per_cell") or []
    if not rows:
        return ["> n/a — no `calibration.per_cell`.", ""]
    out = ["| cell | TE_inj | K̄ | delay | n |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.get('cell_id', '?')} | {_fmt(r.get('te_inj'))} | {_fmt(r.get('kbar'))} "
                   f"| {r.get('delay', '?')} | {r.get('n', '?')} |")
    out.append("")
    return out


def _figure_gallery_lines(figures: List[Path], results_dir: Path) -> List[str]:
    r"""Auto-link every rendered figure (PDF preferred) relative to the report."""
    if not figures:
        return ["> n/a — no figures rendered.", ""]
    out: List[str] = []
    for fig in figures:
        try:
            rel = fig.relative_to(results_dir)
        except ValueError:
            rel = fig
        out.append(f"- [{fig.name}]({rel.as_posix()})")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _render_markdown_v4(config: Dict[str, Any], benchmark: str, *, arm: Optional[str],
                        split: Optional[str], results_dir: Path,
                        metrics: Optional[Dict[str, Any]], figures: List[Path]) -> List[str]:
    r"""Render the full v4 report markdown (every section wrapped warn-don't-gate)."""
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    title = f"# synthetic_v4 report — `{tag}` / arm `{arm or 'n/a'}` / split `{split or 'n/a'}`"
    lines: List[str] = [title, ""]
    if metrics is None:
        lines += ["> ⚠ This arm/split has not been graded — no `metrics.json` found. "
                  "Run `--stage eval` first.", ""]
    lines += ["## Provenance", ""]
    lines += _safe_section("Provenance",
                           lambda: _provenance_lines(config, benchmark, arm, split, metrics))
    lines += ["## Headline gates", ""]
    lines += _safe_section("Headline gates", lambda: _headline_gates_lines(metrics))
    lines += ["## Calibration by lag", ""]
    lines += _safe_section("Calibration by lag", lambda: _calibration_by_lag_lines(metrics))
    lines += ["## Per-cell K̄ vs TE_inj", ""]
    lines += _safe_section("Per-cell", lambda: _per_cell_lines(metrics))
    lines += ["## Figure gallery", ""]
    lines += _safe_section("Figure gallery",
                           lambda: _figure_gallery_lines(figures, results_dir))
    return lines


def final_report_v4(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw_v4",
    arm: Optional[str] = None,
    out_dir: Path,
    split: Optional[str] = None,
    render_figures: bool = True,
) -> Path:
    r"""Assemble the per-arm/per-split ``report.md`` (+ the :mod:`visualize_v4` gallery) (S7-T05).

    Reads ``metrics.json`` / ``per_sample_eval.npz`` from ``out_dir`` (the split-scoped run dir the
    ``eval`` stage wrote), renders every :func:`visualize_v4.figure_specs` figure into
    ``<out_dir>/figures/`` (each guarded independently), and writes ``<out_dir>/report.md``. The
    split-independent ``realizability.json`` is read from ``out_dir`` if present, else from the tag
    root. A missing ``metrics.json`` yields an explicit "not graded" report rather than an error.

    Args:
        config: The (arm-resolved) config tree.
        benchmark: Active benchmark key under ``benchmarks``.
        arm: The resolved arm name (for the header/provenance), or ``None``.
        out_dir: The run dir the report + figures are written into (split-scoped by the driver).
        split: The split label for the header, or ``None``.
        render_figures: When ``True`` and ``metrics.json`` is present, render the figure gallery.

    Returns:
        The written ``report.md`` :class:`Path`.
    """
    results_dir = Path(out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"

    metrics = _load_json(results_dir / "metrics.json")

    if render_figures and metrics is not None:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import visualize_v4 as viz
        figures_dir.mkdir(parents=True, exist_ok=True)
        per_sample = _load_per_sample(results_dir)
        for stem, render in viz.figure_specs():
            try:
                render(per_sample, metrics, figures_dir / stem)
            except Exception as exc:  # noqa: BLE001 -- one bad figure never suppresses the rest
                logger.warning("final_report_v4: figure %s skipped (%s)", stem, exc)

    figures = _gather_figures(figures_dir)
    lines = _render_markdown_v4(config, benchmark, arm=arm, split=split, results_dir=results_dir,
                                metrics=metrics, figures=figures)
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("final_report_v4: wrote %s", report_path)
    return report_path


def run_report_v4(ctx: StageContextV4) -> int:
    r"""``report`` stage: render the per-arm/per-split ``report.md`` + figure gallery (S7-T05).

    Split-scoped (``results/<tag>/<arm>/<split>/``) under the driver's fan-out; owns all
    human-readable rendering (the Sprint-6 ``eval`` stage emits only machine artifacts). Non-fatal:
    a report failure warns and never gates a sweep.
    """
    report_path = final_report_v4(
        ctx.config, benchmark=ctx.benchmark, arm=ctx.arm,
        out_dir=ctx.output_dir(), split=ctx.split,
    )
    print(f"[report] arm={ctx.arm} split={ctx.split} -> {report_path}")
    return 0


register_stage_v4(StageSpecV4(
    name="report",
    run=run_report_v4,
    order=_STAGE_ORDER,
    model_dependent=True,
    fatal=False,
    help="assemble the per-arm/per-split report.md + visualize_v4 figure gallery",
))
