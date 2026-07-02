r"""Final report assembly for ``synthetic_v2`` (Sprint 7, S7-T05).

:func:`final_report_v2` collates every artifact a ``synthetic_v2`` run produces -- the
build manifest (``meta.json``), the evaluation gates (``metrics.json``), the model-free
realizability probe (``realizability.json``), the journal figure gallery, and the standard
testing per-sample diagnostics (``sample_metrics.csv`` + a representative TE-annotated
sample PDF) -- into a single markdown report under ``results/<tag>/``, plus a rendered
headline diagnostics figure. It supersedes the minimal Sprint-6 :func:`eval_v2.write_report`
seam (which remains as an internal fallback for the ``metrics``-only summary).

The report is the delivery surface of the pipeline: it presents the three transfer
entropies ($\mathrm{TE}_{\mathrm{inj}}$, $\mathrm{TE}_{\mathrm{raw}}$,
$\mathrm{TE}_{\mathrm{scat}}$) and the preservation fraction
$\mathrm{frac}_\Phi = \mathrm{TE}_{\mathrm{scat}} / \mathrm{TE}_{\mathrm{inj}}$, the
$\gamma$-calibration of the model surrogate $\bar K$ against both $\mathrm{TE}_{\mathrm{inj}}$
and $\mathrm{TE}_{\mathrm{scat}}$, lag recovery, and the null controls -- with every figure
referenced and at least one representative sample showing the actual TE values.

See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 7.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    r"""Load a JSON file, returning ``None`` when it is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("final_report_v2: could not read %s (%s)", path, exc)
        return None


def _load_per_sample(results_dir: Path) -> Optional[Dict[str, Any]]:
    r"""Load the length-$N$ per-sample eval arrays from ``per_sample_eval.npz``.

    Written by :func:`eval_v2.run_eval`; backs the per-sample TE-vs-$\bar K$ scatter and the
    per-lag calibration small-multiples. Returns ``None`` when absent (older runs) so the
    figures fall back gracefully.

    Args:
        results_dir: The run directory holding ``per_sample_eval.npz``.

    Returns:
        A dict of arrays keyed by field name, or ``None``.
    """
    path = results_dir / "per_sample_eval.npz"
    if not path.is_file():
        return None
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}
    except Exception as exc:  # noqa: BLE001  (report must not fail on a bad side-car)
        logger.warning("final_report_v2: could not read %s (%s)", path, exc)
        return None


def _fmt(x: Any, spec: str = ".4g") -> str:
    r"""Format a scalar for markdown (``n/a`` for ``None`` / non-finite)."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf or xf in (float("inf"), float("-inf")):  # NaN / inf
        return "n/a"
    return format(xf, spec)


def _gather_figures(*dirs: Path) -> List[Path]:
    r"""Collect every ``.pdf`` / ``.png`` under the given directories (sorted, de-duped)."""
    seen: Dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in ("*.pdf", "*.png"):
            for p in sorted(d.glob(pattern)):
                # Prefer the PDF when both exist for the same stem.
                key = str(p.with_suffix(""))
                if key not in seen or p.suffix == ".pdf":
                    seen[key] = p
    return sorted(seen.values(), key=lambda p: p.name)


def _resolve_results_and_cache(
    config: Dict[str, Any], benchmark: str, out_dir: Optional[Path]
) -> tuple[Path, Optional[Path]]:
    r"""Resolve the ``results/<tag>`` dir and the cache dir (for ``meta.json``)."""
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    if out_dir is not None:
        results_dir = Path(out_dir)
    else:
        results_root = Path(config.get("paths", {}).get("results_dir", "./results"))
        if not results_root.is_absolute():
            results_root = _MODULE_DIR / results_root
        results_dir = results_root / tag
    cache_dir: Optional[Path] = None
    try:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (
            resolve_cache_dir,
        )
        cache_dir = resolve_cache_dir(config, benchmark=benchmark)
    except Exception:  # noqa: BLE001  (report must not fail on a missing cache)
        cache_dir = None
    return results_dir, cache_dir


def _representative_sample(samples_csv: Path) -> Optional[Dict[str, str]]:
    r"""Pick the highest-TE row from a testing ``sample_metrics.csv`` (or the first row)."""
    if not samples_csv.is_file():
        return None
    try:
        with open(samples_csv, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    if not rows:
        return None
    if "te_true" in rows[0]:
        def _te(r: Dict[str, str]) -> float:
            try:
                return float(r.get("te_true") or "nan")
            except ValueError:
                return float("nan")
        finite = [r for r in rows if _te(r) == _te(r)]
        if finite:
            return max(finite, key=_te)
    return rows[0]


def final_report_v2(
    config: Dict[str, Any],
    *,
    benchmark: str = "G1_raw",
    out_dir: Optional[Path] = None,
    render_headline: bool = True,
) -> Path:
    r"""Assemble the full ``synthetic_v2`` markdown report + headline figure (S7-T05).

    Collates ``meta.json`` (build manifest / three-TE-per-cell table), ``metrics.json``
    (calibration / lag / null gates), ``realizability.json`` (model-free preservation),
    the figure gallery, and the standard-testing ``sample_metrics.csv`` + a representative
    TE-annotated sample PDF into ``results/<tag>/report.md``. Missing artifacts degrade to
    an explicit "not available" note rather than failing.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        out_dir: Optional override for the run directory (defaults to ``results/<tag>/``).
        render_headline: When ``True`` and ``metrics.json`` is present, render the headline
            diagnostics figure into ``figures/headline_diagnostics.{pdf,png}``.

    Returns:
        The written ``report.md`` :class:`Path`.
    """
    results_dir, cache_dir = _resolve_results_and_cache(config, benchmark, out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"

    metrics = _load_json(results_dir / "metrics.json")
    realizability = _load_json(results_dir / "realizability.json")
    meta = _load_json(cache_dir / "meta.json") if cache_dir is not None else None

    # Render the headline diagnostics figure plus the full aggregate + prediction-gap
    # gallery from the metrics (and realizability, for TE_raw). Each plot is guarded
    # independently so one failing figure never suppresses the rest; every figure written
    # into ``figures_dir`` before ``_gather_figures`` (below) is auto-listed in the gallery.
    if render_headline and metrics is not None:
        from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
            visualize_v2 as viz,
        )
        figures_dir.mkdir(parents=True, exist_ok=True)
        per_sample = _load_per_sample(results_dir)
        frac_thr = None
        try:
            frac_thr = (config.get("benchmarks", {}).get(benchmark, {})
                        .get("eval", {}).get("realizability", {}).get("frac_threshold"))
        except AttributeError:
            frac_thr = None
        plot_specs = [
            ("headline_diagnostics",
             lambda p: viz.plot_diagnostics_panel(metrics, p, realizability=realizability)),
            ("te_kld_scatter",
             lambda p: viz.plot_te_kld_scatter(per_sample, metrics, p)),
            ("calibration_by_lag",
             lambda p: viz.plot_calibration_by_lag(metrics, p, per_sample=per_sample)),
            ("frac_phi_distribution",
             lambda p: viz.plot_frac_phi_distribution(metrics, p, frac_threshold=frac_thr)),
            ("lag_mass_summary", lambda p: viz.plot_lag_mass_summary(metrics, p)),
            ("pred_gain_vs_te",
             lambda p: viz.plot_pred_gain_vs_te(metrics, p, realizability=realizability)),
            ("pred_gain_vs_kbar", lambda p: viz.plot_pred_gain_vs_kbar(metrics, p)),
            ("three_te", lambda p: viz.plot_three_te(metrics, p, realizability=realizability)),
            ("lag_profiles", lambda p: viz.plot_lag_profiles(metrics, p)),
            ("kld_vs_time", lambda p: viz.plot_kld_vs_time(metrics, p)),
            ("null_controls", lambda p: viz.plot_null_controls(metrics, p)),
        ]
        for name, fn in plot_specs:
            try:
                fn(figures_dir / name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("final_report_v2: figure %s skipped (%s)", name, exc)

    samples_dir = results_dir / "test_plots" / "samples_diag"
    rep = _representative_sample(samples_dir / "sample_metrics.csv")
    figures = _gather_figures(figures_dir, samples_dir)

    lines = _render_markdown(
        config, benchmark, results_dir, metrics, meta, realizability, rep, figures,
    )
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("final_report_v2: wrote %s", report_path)
    return report_path


def _render_markdown(
    config: Dict[str, Any],
    benchmark: str,
    results_dir: Path,
    metrics: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
    realizability: Optional[Dict[str, Any]],
    rep: Optional[Dict[str, str]],
    figures: List[Path],
) -> List[str]:
    r"""Build the report markdown lines from the collated artifacts."""
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    cal = (metrics or {}).get("calibration", {}) or {}
    lag = (metrics or {}).get("lag_recovery", {}) or {}
    nul = (metrics or {}).get("null_controls", {}) or {}
    frac = (metrics or {}).get("frac_phi", {}) or {}

    lines: List[str] = [
        f"# synthetic_v2 final report — `{tag}`",
        "",
        f"- benchmark: `{benchmark}`",
        f"- render mode: `{(meta or {}).get('render_mode', (config.get('benchmarks', {}).get(benchmark, {}).get('raw', {}) or {}).get('render_mode', 'am_carrier'))}`",
        f"- checkpoint: `{(metrics or {}).get('ckpt', 'n/a')}`",
        f"- split: `{(metrics or {}).get('split', 'n/a')}`"
        f" (n_samples={(metrics or {}).get('n_samples', 'n/a')}, "
        f"n_cells={(metrics or {}).get('n_cells', (len((meta or {}).get('cells', [])) or 'n/a'))})",
        f"- pooled TE_inj: `{_fmt((meta or {}).get('te_true'))}` nats",
        "",
        "> Companion docs: `SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md` (math/design) and",
        "> `SYNTHETIC_V2_SPEC_AND_SPRINTS.md` (build roadmap).",
        "",
        "## Headline gates",
        "",
        "| gate | value |",
        "|---|---|",
        f"| $\\gamma_{{\\mathrm{{inj}}}}$ (K̄ vs TE_inj, per-cell) | {_fmt(cal.get('gamma_inj'))} |",
        f"| $\\gamma_{{\\mathrm{{scat}}}}$ (K̄ vs TE_scat, per-cell) | {_fmt(cal.get('gamma_scat'))} |",
        f"| $\\gamma_{{\\mathrm{{inj}}}}$ per-sample (n={(metrics or {}).get('n_samples', '?')}) | "
        f"{_fmt(cal.get('gamma_inj_sample'))} ($R^2$={_fmt(cal.get('r2_inj_sample'), '.2f')}) |",
        f"| $\\gamma_{{\\mathrm{{scat}}}}$ per-sample | "
        f"{_fmt(cal.get('gamma_scat_sample'))} ($R^2$={_fmt(cal.get('r2_scat_sample'), '.2f')}) |",
        f"| mean frac_Φ (signal cells) | {_fmt(frac.get('mean'))} "
        f"[{_fmt(frac.get('min'))}, {_fmt(frac.get('max'))}] |",
        f"| mean LagMass | {_fmt(lag.get('mean_lag_mass'))} "
        f"(thr {_fmt(lag.get('lag_mass_threshold'))}) |",
    ]
    for ctrl, res in nul.items():
        lines.append(f"| null_ratio ({ctrl}) → 0 | {_fmt((res or {}).get('mean_ratio'))} |")
    lines.append("")

    # --- per-lag per-sample calibration table (Enhancement C/D) ------------------------
    by_lag = (cal or {}).get("by_lag") or {}
    if by_lag:
        lines += [
            "### Calibration by lag (per-sample fit)", "",
            "| D | $\\gamma_{\\mathrm{inj}}$ | $R^2_{\\mathrm{inj}}$ "
            "| $\\gamma_{\\mathrm{scat}}$ | $R^2_{\\mathrm{scat}}$ | n |",
            "|---|---|---|---|---|---|",
        ]
        for d in sorted(by_lag, key=lambda k: int(k)):
            e = by_lag[d] or {}
            lines.append(
                f"| {d} | {_fmt(e.get('gamma_inj'))} | {_fmt(e.get('r2_inj'), '.2f')} "
                f"| {_fmt(e.get('gamma_scat'))} | {_fmt(e.get('r2_scat'), '.2f')} "
                f"| {e.get('n', '?')} |"
            )
        lines.append("")

    if metrics is None:
        lines += [
            "> ⚠ `metrics.json` not found — run `--stage eval` (after the headline "
            "`--stage train`) to populate the calibration / lag / null gates.",
            "",
        ]

    # --- three-TE per-cell table (from the build manifest + realizability probe) ------
    lines += ["## Three transfer entropies per cell", "",
              "The injected latent TE (`TE_inj`, exact), the raw-domain TE (`TE_raw`, "
              "measured), the scattering-realizable TE (`TE_scat`, measured on the "
              "model-facing features), and the preservation fraction "
              "`frac_Φ = TE_scat / TE_inj`.", ""]
    cells = (meta or {}).get("cells")
    te_raw_by_cell = _te_raw_by_cell(realizability)
    if cells:
        lines += [
            "| cell | target_TE | D | B | TE_inj (realised) | TE_raw | TE_scat | frac_Φ |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in cells:
            cid = c.get("cell_id")
            lines.append(
                f"| {cid} | {_fmt(c.get('target_te'))} | {c.get('D')} "
                f"| {_fmt(c.get('B_y_scalar'))} | {_fmt(c.get('te_block_realised'))} "
                f"| {_fmt(te_raw_by_cell.get(cid))} | {_fmt(c.get('te_scat_measured'))} "
                f"| {_fmt(c.get('frac_phi'))} |"
            )
        lines.append("")
    else:
        lines += ["> ⚠ `meta.json` not found — build the cache (`--stage build`) to populate "
                  "the per-cell manifest.", ""]

    # --- representative sample ---------------------------------------------------------
    lines += ["## Representative sample (standard testing diagnostic)", ""]
    if rep is not None:
        pdf = rep.get("out_path")
        lines += [
            "One per-sample diagnostic from the standard `run_tests.py` path, showing the "
            "actual TE provenance the model was graded against:",
            "",
            "| field | value |",
            "|---|---|",
            f"| guid | `{rep.get('guid', 'n/a')}` |",
            f"| TE_inj | {_fmt(rep.get('te_true'))} |",
            f"| TE_scat | {_fmt(rep.get('te_scat'))} |",
            f"| frac_Φ | {_fmt(rep.get('frac_phi'))} |",
            f"| model K̄ (kld_mean) | {_fmt(rep.get('kld_mean'))} |",
            f"| lag D | {rep.get('sample_delay', 'n/a')} |",
        ]
        if pdf:
            rel = _rel(results_dir / "test_plots" / "samples_diag" / pdf, results_dir)
            lines.append(f"| diagnostic PDF | [`{pdf}`]({rel}) |")
        lines.append("")
    else:
        lines += ["> ⚠ no `sample_metrics.csv` — run `--stage test_plots` to generate the "
                  "TE-annotated per-sample diagnostics.", ""]

    # --- figure gallery ----------------------------------------------------------------
    # The interactive training curve is an HTML (not a PDF/PNG), so ``_gather_figures``
    # never picks it up; link it explicitly at the top of the gallery when present.
    lines += ["## Figure gallery", ""]
    tc_html = results_dir / "figures" / "training_curves.html"
    if tc_html.is_file():
        lines.append(
            f"- [`training_curves.html`]({_rel(tc_html, results_dir)}) "
            "— interactive training curves (every logged metric; open in a browser)"
        )
    for fig in figures:
        lines.append(f"- [`{fig.name}`]({_rel(fig, results_dir)})")
    if not figures and not tc_html.is_file():
        lines.append("> ⚠ no figures found under `figures/` — run the preview / eval / "
                     "report stages to populate the gallery.")
    lines.append("")

    lines += [
        "---",
        "",
        f"_Report assembled by `final_report_v2` for run `{tag}`._",
        "",
    ]
    return lines


def _te_raw_by_cell(realizability: Optional[Dict[str, Any]]) -> Dict[Any, Any]:
    r"""Map ``cell_id -> te_raw`` from a ``realizability.json`` (keyed by cell)."""
    out: Dict[Any, Any] = {}
    if not realizability:
        return out
    per_cell = realizability.get("per_cell", {})
    entries = per_cell.values() if isinstance(per_cell, dict) else per_cell
    for entry in entries or []:
        if isinstance(entry, dict) and "cell_id" in entry:
            out[entry["cell_id"]] = entry.get("te_raw")
    return out


def _rel(path: Path, base: Path) -> str:
    r"""Best-effort relative path (POSIX) of ``path`` against ``base`` for markdown links."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
