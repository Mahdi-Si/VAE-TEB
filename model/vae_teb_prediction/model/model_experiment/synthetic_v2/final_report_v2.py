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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent


# =============================================================================
# Report-section registry (S4-T05)
# =============================================================================
@dataclass(frozen=True)
class SectionContext:
    r"""Everything a report section may read.

    Attributes:
        config: The parsed config tree.
        benchmark: Active benchmark key.
        results_dir: The **per-split** directory the report is being written into.
        metrics: The split's ``metrics.json``, or ``None`` when it has not been graded.
        meta: The cache's ``meta.json``, or ``None``.
        realizability: The model-free ``realizability.json``, or ``None``.
        split: The split label, or ``None``.
    """

    config: Dict[str, Any]
    benchmark: str
    results_dir: Path
    metrics: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    realizability: Optional[Dict[str, Any]] = None
    split: Optional[str] = None


@dataclass(frozen=True)
class SectionSpec:
    r"""One registered report section.

    Attributes:
        name: Short identifier, used in the ``n/a`` note when the section raises.
        order: Render order among registered sections (lower renders first).
        render: ``(SectionContext) -> List[str]`` returning markdown lines.
    """

    name: str
    order: int
    render: Callable[[SectionContext], List[str]] = field(repr=False)


_SECTION_REGISTRY: List[SectionSpec] = []


def register_section(spec: SectionSpec) -> None:
    r"""Register an append-only report section, keeping the list sorted by ``spec.order``.

    Sprints 5/6/7 each contribute one section from their own module, so the fixed section
    order in :func:`_render_markdown` never has to be edited (and three analyses can be built
    in parallel without touching the same block).

    Args:
        spec: The section to register.

    Raises:
        ValueError: When ``spec.name`` is already registered.
    """
    if any(s.name == spec.name for s in _SECTION_REGISTRY):
        raise ValueError(f"report section {spec.name!r} is already registered")
    _SECTION_REGISTRY.append(spec)
    _SECTION_REGISTRY.sort(key=lambda s: s.order)


def _render_registered_sections(ctx: SectionContext) -> List[str]:
    r"""Render every registered section, degrading a raising one to an ``n/a`` note.

    A report is a diagnostic artifact: losing the whole thing because one experimental
    section hit a missing key is a worse failure than printing that one section as ``n/a``.

    Args:
        ctx: The section context.

    Returns:
        The concatenated markdown lines of every registered section.
    """
    lines: List[str] = []
    for spec in _SECTION_REGISTRY:
        try:
            lines += list(spec.render(ctx))
        except Exception as exc:  # noqa: BLE001 -- one bad section never loses the report
            logger.warning("final_report_v2: section %s failed (%s)", spec.name, exc)
            lines += [f"## {spec.name}", "", f"> ⚠ n/a — section failed: "
                      f"{type(exc).__name__}: {exc}", ""]
    return lines


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


def tag_root(config: Dict[str, Any], benchmark: str) -> Path:
    r"""The arm-**independent** run root, ``results/<tag>/`` (S4-T06).

    Resolved absolutely from the config rather than by walking ``.parent`` up from the report's
    output directory, because that walk has a different depth in each layout:

    ==============================  ========================  ============================
    layout                          report ``results_dir``    depth to ``results/<tag>/``
    ==============================  ========================  ============================
    v1 / v2 (arm-less)              ``<tag>/<split>/``        ``.parent``
    v3 (arm-scoped)                 ``<tag>/<arm>/<split>/``  ``.parents[1]``
    ==============================  ========================  ============================

    The split-independent data-generation gallery (``figures/``) and ``realizability.json``
    live here; the per-arm ``training_curves.html`` does not.

    Args:
        config: The parsed config tree.
        benchmark: Active benchmark key, used as the tag fallback.

    Returns:
        The ``results/<tag>/`` :class:`Path` (not created).
    """
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    results_root = Path(config.get("paths", {}).get("results_dir", "./results"))
    if not results_root.is_absolute():
        results_root = _MODULE_DIR / results_root
    return results_root / tag


def _resolve_results_and_cache(
    config: Dict[str, Any], benchmark: str, out_dir: Optional[Path]
) -> tuple[Path, Optional[Path]]:
    r"""Resolve the report's output dir and the cache dir (for ``meta.json``)."""
    if out_dir is not None:
        results_dir = Path(out_dir)
    else:
        results_dir = tag_root(config, benchmark)
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
    split: Optional[str] = None,
    render_headline: bool = True,
) -> Path:
    r"""Assemble the full ``synthetic_v2`` markdown report + headline figure (S7-T05).

    Collates ``meta.json`` (build manifest / three-TE-per-cell table), ``metrics.json``
    (calibration / lag / null gates), ``realizability.json`` (model-free preservation),
    the figure gallery, and the standard-testing ``sample_metrics.csv`` + a representative
    TE-annotated sample PDF into ``<out_dir>/report.md``. Missing artifacts degrade to
    an explicit "not available" note rather than failing.

    When ``out_dir`` is a **per-split** directory (``results/<tag>/<split>/``, as the driver
    passes for each of train / val / test), the split-independent ``realizability.json`` is
    read from the parent ``results/<tag>/`` if not present locally, and the report links the
    shared split-independent figure gallery (``../figures/``).

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        benchmark: Active benchmark key under ``benchmarks``.
        out_dir: Optional override for the run directory (defaults to ``results/<tag>/``).
        split: Optional split label (``train`` / ``val`` / ``test``) for the report header
            when ``out_dir`` is a per-split directory.
        render_headline: When ``True`` and ``metrics.json`` is present, render the headline
            diagnostics figure into ``figures/headline_diagnostics.{pdf,png}``.

    Returns:
        The written ``report.md`` :class:`Path`.
    """
    results_dir, cache_dir = _resolve_results_and_cache(config, benchmark, out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"

    # S4-T06: three distinct anchors, because the arm layout inserts one directory level.
    #
    #   tag_root    results/<tag>/            data-story figures/, realizability.json
    #   run_root    results/<tag>/<arm>/      figures/training_curves.html   (== tag_root when
    #                                                                         arm-less)
    #   results_dir results/<tag>/<arm>/<sp>/ this split's figures/
    #
    # Resolving the first by ``.parent`` (as this did) silently pointed at the ARM directory
    # under a v3 config, so a per-split report linked the arm's training curves as if they were
    # the shared data-generation gallery -- and the shared gallery was never linked at all.
    root = tag_root(config, benchmark)
    run_root = results_dir.parent

    metrics = _load_json(results_dir / "metrics.json")
    # realizability.json is split-independent (model-free preflight): read it locally, else
    # from the tag root.
    realizability = (_load_json(results_dir / "realizability.json")
                     or _load_json(root / "realizability.json"))
    meta = _load_json(cache_dir / "meta.json") if cache_dir is not None else None
    shared_figures = root / "figures"
    have_shared = shared_figures.is_dir() and shared_figures != figures_dir

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
        # Which KLD summary the single-variant figures (density / violins) feature; the full
        # family is always computed in per_sample_eval.npz (§14.5).
        density_variant = "kbar"
        try:
            density_variant = str((config.get("benchmarks", {}).get(benchmark, {})
                                   .get("eval", {}).get("kld_analysis", {}) or {})
                                  .get("density_variant", "kbar"))
        except AttributeError:
            density_variant = "kbar"
        plot_specs = [
            ("headline_diagnostics",
             lambda p: viz.plot_diagnostics_panel(metrics, p, realizability=realizability)),
            ("te_kld_scatter",
             lambda p: viz.plot_te_kld_scatter(per_sample, metrics, p)),
            ("calibration_by_lag",
             lambda p: viz.plot_calibration_by_lag(metrics, p, per_sample=per_sample)),
            # KLD-summary family vs TE (§14.5): different KLD definitions + summarisations.
            ("kld_variants_vs_te",
             lambda p: viz.plot_kld_variants_vs_te(per_sample, metrics, p)),
            ("kld_variants_vs_te_scat",
             lambda p: viz.plot_kld_variants_vs_te(per_sample, metrics, p, te_axis="scat")),
            ("kld_te_correlation",
             lambda p: viz.plot_kld_te_correlation(metrics, p)),
            ("kld_te_density",
             lambda p: viz.plot_kld_te_density(per_sample, metrics, p, variant=density_variant)),
            ("kld_distribution_by_te",
             lambda p: viz.plot_kld_distribution_by_te(per_sample, metrics, p,
                                                       variant=density_variant)),
            ("per_head_kld_vs_te",
             lambda p: viz.plot_per_head_kld_vs_te(per_sample, metrics, p)),
            ("frac_phi_distribution",
             lambda p: viz.plot_frac_phi_distribution(metrics, p, frac_threshold=frac_thr)),
            ("lag_mass_summary", lambda p: viz.plot_lag_mass_summary(metrics, p)),
            ("pred_gain_vs_te",
             lambda p: viz.plot_pred_gain_vs_te(metrics, p, realizability=realizability)),
            ("pred_gain_vs_kbar", lambda p: viz.plot_pred_gain_vs_kbar(metrics, p)),
            ("three_te", lambda p: viz.plot_three_te(metrics, p, realizability=realizability)),
            ("lag_profiles", lambda p: viz.plot_lag_profiles(metrics, p)),
            ("kld_vs_time", lambda p: viz.plot_kld_vs_time(metrics, p)),
            # The discriminating gate (prediction space) and, beside it, the demoted KL-space
            # ratio under an honest caption -- the negative result is shown, not hidden.
            ("null_controls", lambda p: viz.plot_null_controls(metrics, p)),
            ("kl_shuffle_readout", lambda p: viz.plot_kl_shuffle_readout(metrics, p)),
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
        split=split, shared_figures=(shared_figures if have_shared else None),
        run_root=run_root,
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
    *,
    split: Optional[str] = None,
    shared_figures: Optional[Path] = None,
    run_root: Optional[Path] = None,
) -> List[str]:
    r"""Build the report markdown lines from the collated artifacts.

    Args:
        run_root: The **arm** run dir (``results/<tag>/<arm>/``, or the tag root when
            arm-less), where training writes ``figures/training_curves.html``. Defaults to
            ``results_dir.parent``.
    """
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    run_root = run_root if run_root is not None else results_dir.parent
    split_label = split or (metrics or {}).get("split")
    cal = (metrics or {}).get("calibration", {}) or {}
    lag = (metrics or {}).get("lag_recovery", {}) or {}
    nul = (metrics or {}).get("null_controls", {}) or {}
    pred = (metrics or {}).get("prediction_controls", {}) or {}
    frac = (metrics or {}).get("frac_phi", {}) or {}

    title_split = f" — split `{split_label}`" if split_label else ""
    lines: List[str] = [
        f"# synthetic_v2 final report — `{tag}`{title_split}",
        "",
        f"- benchmark: `{benchmark}`",
        # Provenance (S4-T03). Three structurally-identical arms otherwise produce three
        # indistinguishable reports; the arm and class come from the GRADED checkpoint.
        f"- arm: `{(metrics or {}).get('arm') or 'n/a'}`   "
        f"model class: `{(metrics or {}).get('model_class', 'n/a')}`",
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
    lines += _null_cell_rows(cal)
    lines += _prediction_controls_rows(pred)
    lines.append("")
    lines += _readouts_section(nul)

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

    # --- KLD-summary family vs TE (§14.5) --------------------------------------------------
    lines += _kld_variants_section(cal)

    # --- append-only sections contributed by the analysis modules (S4-T05) -----------------
    # Each registered section is a pure ``(SectionContext) -> List[str]`` and is rendered
    # inside its own guard, so a Sprint 5/6/7 analysis can add a section from its own file
    # and a raising section degrades to an "n/a" note rather than losing the whole report.
    lines += _render_registered_sections(
        SectionContext(config=config, benchmark=benchmark, results_dir=results_dir,
                       metrics=metrics, meta=meta, realizability=realizability,
                       split=split_label)
    )

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
    # Training writes the interactive curve into the RUN root's figures dir
    # (``results/<tag>/<arm>/figures/``), never into a per-split subfolder. Looking for it
    # under ``results_dir`` -- as this did -- produced a link that was dead in every layout.
    tc_html = run_root / "figures" / "training_curves.html"
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
    # Per-split reports link the shared, split-independent data-generation gallery.
    if shared_figures is not None:
        lines.append(
            f"- _(split-independent data-generation figures: "
            f"[`{_rel(shared_figures, results_dir)}/`]({_rel(shared_figures, results_dir)}/))_"
        )
    lines.append("")

    lines += [
        "---",
        "",
        f"_Report assembled by `final_report_v2` for run `{tag}`._",
        "",
    ]
    return lines


def _null_cell_rows(cal: Dict[str, Any]) -> List[str]:
    r"""Headline-gate rows for the null-cell intercept (S4-T02).

    v3 initialises with $K \equiv 0$, so the calibration intercept $\alpha$ is no longer
    absorbing a random log-variance-head floor. That makes
    $\bar K \big|_{\mathrm{TE}_{\mathrm{inj}} = 0} \to 0$ a claim the model can be held to --
    one v1 structurally could not satisfy -- and it is the cleanest single number separating
    the ``parity`` arm from the v3 arms. The two are rendered adjacently because a calibrated
    surrogate has $\alpha \approx \bar K \big|_{\mathrm{TE} = 0}$: a large gap between them
    means the affine fit is being dragged by the signal cells.

    Args:
        cal: The ``metrics['calibration']`` dict (possibly empty or pre-S4).

    Returns:
        Markdown table rows to append to the headline-gates table.
    """
    knc = (cal or {}).get("kbar_at_null_cells")
    if not knc:
        return [r"| $\bar K$ at null cells (TE_inj = 0) | n/a |",
                f"| $\\alpha_{{\\mathrm{{inj}}}}$ (calibration intercept) "
                f"| {_fmt((cal or {}).get('alpha_inj'))} |"]
    verdict = "pass" if knc.get("pass") else "**FAIL**"
    return [
        f"| $\\bar K$ at null cells (TE_inj = 0) | {_fmt(knc.get('mean'))} "
        f"[{_fmt(knc.get('ci_lo'))}, {_fmt(knc.get('ci_hi'))}] nats, {verdict} "
        f"(thr {_fmt(knc.get('threshold'))}, n={knc.get('n_cells', '?')} cells) |",
        f"| $\\alpha_{{\\mathrm{{inj}}}}$ (calibration intercept) "
        f"| {_fmt((cal or {}).get('alpha_inj'))} |",
    ]


def _prediction_controls_rows(pred: Dict[str, Any]) -> List[str]:
    r"""Headline-gate rows for the prediction-space control (S3-T05).

    Renders, per input-stream corruption, the mean shuffle penalty
    $\mathcal L_{\mathrm{feat}}^{\pi(U)} - \mathcal L_{\mathrm{feat}}$ over the signal cells
    and the verdict on the ordering
    $\mathcal L_{\mathrm{feat}} < \mathcal L_{\mathrm{base}} < \mathcal L_{\mathrm{feat}}^{\pi(U)}$.
    A pre-S3 ``metrics.json`` has no ``prediction_controls`` block and renders ``n/a`` rather
    than raising.

    Args:
        pred: The ``metrics['prediction_controls']`` block (possibly empty).

    Returns:
        Markdown table rows to append to the headline-gates table.
    """
    overall = (pred or {}).get("overall") or {}
    controls = (pred or {}).get("controls") or []
    if not controls:
        return ["| prediction control (feat < base < feat_corrupted) | n/a |"]
    rows = [
        f"| $\\mathcal{{L}}_{{\\mathrm{{feat}}}}$ / $\\mathcal{{L}}_{{\\mathrm{{base}}}}$ "
        f"(signal cells) | {_fmt(overall.get('feat_loss'))} / {_fmt(overall.get('base_loss'))} |",
    ]
    for ctrl in controls:
        verdict = overall.get(f"ordering_pass_{ctrl}")
        mark = "n/a" if verdict is None else ("pass" if verdict else "**FAIL**")
        rows.append(
            f"| ordering gate ({ctrl}): "
            f"$\\mathcal{{L}}_{{\\mathrm{{feat}}}} < \\mathcal{{L}}_{{\\mathrm{{base}}}} "
            f"< \\mathcal{{L}}_{{\\mathrm{{feat}}}}^{{\\pi(U)}}$ | {mark} "
            f"(penalty {_fmt(overall.get(f'shuffle_penalty_{ctrl}'))}) |"
        )
    frac = overall.get("ordering_pass_frac")
    rows.append(
        f"| signal cells passing the ordering gate | {_fmt(frac, '.0%')} "
        f"of {(pred or {}).get('n_signal_cells', '?')} |"
    )
    return rows


def _readouts_section(nul: Dict[str, Any]) -> List[str]:
    r"""Render the demoted KL-space null ratio in its own table, with an honest caption.

    This used to be a headline gate captioned ``null_ratio -> 0``. It is not a gate:
    $\mathrm{KL}(q \,\|\, p)$ measures that the source moved the posterior, not that it moved
    it *correctly*, and a deranged source is out of distribution for a posterior trained on
    matched pairs -- so it usually moves the belief *more*. v3 measured the ratio in
    $[1.02, 1.10]$ on a model that was demonstrably exploiting the source (Finding F2).

    Note:
        The eval-time ``feat_loss_<control>`` corrupts the **input stream**; the training-time
        ``feat_loss_perm`` deranges the already-encoded ``source_state`` along the batch axis.
        They are different corruptions, so they are never rendered in the same table.

    Args:
        nul: The ``metrics['null_controls']`` block (possibly empty).

    Returns:
        Markdown lines for the ``## Readouts`` section.
    """
    lines = [
        "## Readouts (not gates)",
        "",
        "| readout | value |",
        "|---|---|",
    ]
    if not nul:
        lines += ["| $\\bar K_{\\mathrm{shuffled}} / \\bar K_{\\mathrm{signal}}$ | n/a |", ""]
        return lines
    for ctrl, res in nul.items():
        lines.append(
            f"| $\\bar K_{{\\mathrm{{{ctrl}}}}} / \\bar K_{{\\mathrm{{signal}}}}$ "
            f"| {_fmt((res or {}).get('mean_ratio'))} |"
        )
    lines += [
        "",
        "> The KL-space null ratio is **expected to sit near 1.0**, even on a model that "
        "demonstrably exploits the source. $\\mathrm{KL}(q\\,\\|\\,p)$ measures that the "
        "source moved the belief, not that it moved it correctly; a deranged source is still "
        "a source, and it is out of distribution for a posterior trained on matched pairs "
        "(v3 Finding F2). The control that discriminates is the prediction-space ordering in "
        "the headline gates above.",
        "",
    ]
    return lines


# Human-readable labels + display order for the KLD summary family (mirrors
# ``visualize_v2.KLD_VARIANT_LABELS``; kept local so the report needs no matplotlib import).
_KLD_VARIANT_LABELS = {
    "kbar": "mean K_t (clean window)",
    "kbar_sum": "sum K_t (integrated)",
    "kbar_max": "max K_t (peak)",
    "kbar_median": "median K_t",
    "kbar_p90": "p90 K_t",
    "kbar_full": "mean K_t (full seq.)",
    "kbar_postwarm": "mean K_t (post-warmup)",
    "kbar_inband": "in-band KL (L*)",
    "kbar_outband": "out-band KL",
}


def _kld_variant_label(key: str) -> str:
    r"""Human-readable label for a KLD summary key (falls back for per-head columns)."""
    if key in _KLD_VARIANT_LABELS:
        return _KLD_VARIANT_LABELS[key]
    if str(key).startswith("kbar_head"):
        return f"head {str(key).replace('kbar_head', '')} KL"
    return str(key)


def _kld_variants_section(cal: Dict[str, Any]) -> List[str]:
    r"""Render the KLD-summary-family vs TE correlation table (§14.5) for the report.

    Reads the ``calibration.kld_variants`` block written by :func:`eval_v2.fit_calibration`
    and tabulates, per KLD summary, its per-sample slope $\gamma$, Pearson $r$ and Spearman
    $\rho$ against both $\mathrm{TE}_{\mathrm{inj}}$ and $\mathrm{TE}_{\mathrm{scat}}$, ranked
    by $|\rho_{\mathrm{inj}}|$ so the best-tracking summary is first. Returns an explicit
    "not available" note when the block is absent (older runs).

    A variant stamped ``out_of_support`` (S4-T01: ``kbar_full`` under anchor-aligned KL
    support) is marked with a dagger and **sorted below every in-support variant**, so the
    table's first row -- the one a reader takes as "the best surrogate" -- can never be a
    summary that is partly measuring an untrained region of the sequence.

    Args:
        cal: The ``metrics['calibration']`` dict.

    Returns:
        The markdown lines for the section.
    """
    kv = (cal or {}).get("kld_variants") or {}
    lines: List[str] = ["## KLD summary vs TE (which flavour tracks TE)", ""]
    if not kv:
        lines += ["> ⚠ no `calibration.kld_variants` — re-run `--stage eval` to populate the "
                  "KLD-summary family (mean/sum/max/median/p90, window, in-band/out-band, "
                  "per-head).", ""]
        return lines
    lines += [
        "Every model-free KLD summary (§14.5) regressed on the transfer entropy: the "
        "per-sample slope $\\gamma$, Pearson $r$ and Spearman $\\rho$ vs both TEs, ranked by "
        "$|\\rho_{\\mathrm{inj}}|$ (best tracker first). See `kld_variants_vs_te`, "
        "`kld_te_correlation`, `kld_te_density` in the gallery.",
        "",
        "| KLD summary | $\\gamma_{\\mathrm{inj}}$ | $r_{\\mathrm{inj}}$ "
        "| $\\rho_{\\mathrm{inj}}$ | $\\gamma_{\\mathrm{scat}}$ | $\\rho_{\\mathrm{scat}}$ | n |",
        "|---|---|---|---|---|---|---|",
    ]

    def _oos(item: Any) -> bool:
        return bool((kv.get(item) or {}).get("out_of_support", False))

    def _absrho(item: Any) -> float:
        v = (kv.get(item) or {}).get("spearman_inj")
        try:
            return abs(float(v))
        except (TypeError, ValueError):
            return -1.0

    # In-support variants first (by |rho|), then the flagged ones. An out-of-support summary
    # with a high correlation is not a better surrogate; it is a measurement of a region the
    # model was never trained to shape.
    for v in sorted(kv, key=lambda k: (not _oos(k), _absrho(k)), reverse=True):
        e = kv[v] or {}
        label = _kld_variant_label(v) + (" †" if _oos(v) else "")
        lines.append(
            f"| {label} | {_fmt(e.get('gamma_inj'))} "
            f"| {_fmt(e.get('pearson_inj'), '.2f')} | {_fmt(e.get('spearman_inj'), '.2f')} "
            f"| {_fmt(e.get('gamma_scat'))} | {_fmt(e.get('spearman_scat'), '.2f')} "
            f"| {e.get('n', '?')} |"
        )
    if any(_oos(v) for v in kv):
        lines += [
            "",
            "> † **Out of support.** This summary averages $K_t$ over steps the model was "
            "never trained to shape (the warm-up prefix and the untrained final-$H$ tail), "
            "because this checkpoint uses anchor-aligned KL support (`kld_support: anchor`). "
            "It is reported as evidence but excluded from best-variant selection; "
            "`kbar_postwarm` is the exact anchor support and is the correct comparator.",
        ]
    lines.append("")
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
    r"""Relative path (POSIX) of ``path`` against ``base``, for markdown links.

    Uses :func:`os.path.relpath`, not :meth:`Path.relative_to`, because the latter cannot walk
    upwards: it raises whenever ``path`` is not *under* ``base``, which is exactly the case for
    the shared data-generation gallery (``results/<tag>/figures/``) linked from a per-split
    report (``results/<tag>/<arm>/<split>/``). The previous fallback emitted an **absolute
    filesystem path**, producing a link that broke the moment the report left this machine.

    Falls back to the POSIX absolute path only when the two live on different drives, where no
    relative path exists.

    Args:
        path: The link target.
        base: The directory the markdown file lives in.

    Returns:
        A POSIX-style relative link (possibly containing ``..``).
    """
    import os
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # different drives on Windows -- no relative path exists
        return path.as_posix()
