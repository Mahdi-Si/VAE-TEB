r"""Sprint 8 (S8-T01): the cross-arm index that answers the G0 question with a table.

The three arms differ only in their latent/encoder machinery, and each writes its own artifact
tree. Reading them side by side is the whole point of the ladder, and prose cannot do it. This
module assembles ``results/<tag>/arms_report.md``: one row per arm, one column per gate, every
value traced to a named key in a named artifact.

Two contrasts are encoded in the row order:

* ``parity`` $\to$ ``v3_noncausal`` isolates **G1/G2/G3/G5** -- residual posterior variance, smooth
  log-variance bounds, anchor-aligned KL support, ALiBi lag decay, sparse attention.
* ``v3_noncausal`` $\to$ ``v3_prod`` isolates **G0** alone, the causal-normalisation fix, measured
  against a known ground-truth transfer entropy. This is the claim the whole effort exists to test.

Four of the column sources named in the S8-T01 task card do not exist
------------------------------------------------------------------------

They were verified against the code and against the artifacts on disk. :data:`ARMS_REPORT_COLUMNS`
is the corrected mapping, and a test asserts the emitted header against it.

============================  ==========================================================
spec says                     reality
============================  ==========================================================
``prediction_controls``       No ``pred_gap`` key. ``pred_gap`` is a *training* metric
``.pred_gap``                 (``pl_module_v2``), logged to ``logs/*/metrics.csv``. The eval
                              analogue is ``base_loss - feat_loss``, emitted here as ``pred_gain``.
``null_controls``             No ``kld_shuffled_ratio`` key -- also a training metric. The eval
``.kld_shuffled_ratio``       readout is ``null_controls.shuffle.mean_ratio`` (``eval_v2``).
``calibration_predictive``    Actually ``nll_mean`` / ``crps_mean`` / ``coverage_90``
``.{nll, crps,               (``calibration_v3``; integer-percent tag, not a float).
  coverage_0.9}``
``lag_intervention``          No such key. ``rho_by_band`` is a dict over bands; S8-T01's
``.rho_deltaL_attn``          extension to ``summarise_lag_intervention`` adds the per-cell true
                              band, so ``rho_by_band["inband"].rho`` is the column's source.
============================  ==========================================================

Reading the table
-----------------

``gamma_scat`` carries a dagger. S1-T05 established that $\mathrm{TE}_{\mathrm{scat}}$ is ordinal
**within a fixed lag only** -- exactly monotone in $\mathrm{TE}_{\mathrm{inj}}$ at each $D$, but
with $18.8\%$ of cross-$D$ cell pairs inverted, and an absolute scale set by the probe's free ridge
($\pm 60\%$). A pooled $\gamma_{\mathrm{scat}}$ therefore mixes model behaviour with a
$D$-dependent probe bias in a way $\gamma_{\mathrm{inj}}$ does not. $\mathrm{TE}_{\mathrm{inj}}$ is
the primary axis and ``spearman_inj`` the primary rank statistic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
    _fmt,
    _load_json,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (
    StageContext,
    StageSpec,
    register_stage,
)

_STAGE_ORDER = 15

#: Artifact keys. ``metrics`` is the per-split ``metrics.json`` (with ``calibration_predictive``
#: folded in by Sprint 5); the other two are side-car JSONs written by Sprints 6 and 7.
ARTIFACT_METRICS = "metrics"
ARTIFACT_LAG = "lag_intervention"
ARTIFACT_CMI = "cmi"

_ARTIFACT_FILES: Dict[str, str] = {
    ARTIFACT_METRICS: "metrics.json",
    ARTIFACT_LAG: "lag_intervention.json",
    ARTIFACT_CMI: "cmi.json",
}

#: ``(column_label, artifact, dotted_key)``. A ``dotted_key`` of the form ``"a - b"`` is a derived
#: difference of two dotted paths. Asserted against the emitted header by
#: ``test_final_report_v3.py::test_arms_report_header_matches_the_constant``.
ARMS_REPORT_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("gamma_inj", ARTIFACT_METRICS, "calibration.gamma_inj"),
    ("gamma_scat †", ARTIFACT_METRICS, "calibration.gamma_scat"),
    ("alpha_inj", ARTIFACT_METRICS, "calibration.alpha_inj"),
    ("r2_inj", ARTIFACT_METRICS, "calibration.r2_inj"),
    ("spearman_inj", ARTIFACT_METRICS, "calibration.spearman_inj"),
    ("kbar_at_null", ARTIFACT_METRICS, "calibration.kbar_at_null_cells.mean"),
    ("null_gate", ARTIFACT_METRICS, "calibration.kbar_at_null_cells.pass"),
    # CORRECTION 1: prediction_controls carries no `pred_gap`; derive the eval-time gain.
    ("pred_gain", ARTIFACT_METRICS,
     "prediction_controls.overall.base_loss - prediction_controls.overall.feat_loss"),
    ("ordering_pass", ARTIFACT_METRICS, "prediction_controls.overall.ordering_pass"),
    ("shuffle_penalty", ARTIFACT_METRICS,
     "prediction_controls.overall.shuffle_penalty_shuffle"),
    # CORRECTION 2: null_controls carries no `kld_shuffled_ratio`; it is a per-control mean_ratio.
    ("kld_shuffle_ratio", ARTIFACT_METRICS, "null_controls.shuffle.mean_ratio"),
    ("mean_lag_mass", ARTIFACT_METRICS, "lag_recovery.mean_lag_mass"),
    ("inband_gate_pass", ARTIFACT_LAG, "overall.inband_gate_pass"),
    # CORRECTION 4: `rho_deltaL_attn` exists only once `rho_by_band` covers the true band.
    ("rho_deltaL_attn", ARTIFACT_LAG, "rho_by_band.inband.rho"),
    # CORRECTION 3: the shipped keys are nll_mean / crps_mean / coverage_90.
    ("nll", ARTIFACT_METRICS, "calibration_predictive.nll_mean"),
    ("crps", ARTIFACT_METRICS, "calibration_predictive.crps_mean"),
    ("coverage_0.9", ARTIFACT_METRICS, "calibration_predictive.coverage_90"),
    ("rho_kbar_cmi", ARTIFACT_CMI, "overall.rho_kbar_cmi_feature_model.rho"),
    ("cmi_bias", ARTIFACT_CMI, "overall.cmi_bias.estimate"),
    # Whether that bias may be read at all: a negative held-out R^2 of `target_state` on Y+ means
    # the conditioning does not transfer across samples. See the `cond_r2` footnote.
    ("cmi_bias_ok", ARTIFACT_CMI, "overall.cmi_bias.reliable"),
    ("cond_r2_target_state", ARTIFACT_CMI, "overall.cond_r2_feature_model.v"),
    ("cmi_recovery_rho", ARTIFACT_CMI, "recovery.spearman_cmi_te_inj"),
)

#: Columns whose value is a pass/fail verdict rather than a number.
_BOOL_COLUMNS = frozenset({"null_gate", "ordering_pass", "inband_gate_pass", "cmi_bias_ok"})

_PREAMBLE = """The three arms share one cache, one seed set, one objective, and one set of splits.
Only `model_kwargs` differ, so every difference below is attributable to the model.

| contrast | isolates |
|---|---|
| `parity` → `v3_noncausal` | G1/G2/G3/G5: residual posterior variance, smooth log-variance bounds, anchor-aligned KL support, ALiBi lag decay, sparse attention |
| `v3_noncausal` → `v3_prod` | **G0 alone**: the causal-normalisation fix, measured against a known ground-truth transfer entropy |

`parity` is `SeqVaeLagAttnV3` under v1's latent machinery, and serves as the baseline.
"""

_FOOTNOTES = """
† `gamma_scat` is **ordinal within a fixed lag only** (S1-T05): $\\mathrm{TE}_{\\mathrm{scat}}$ is
exactly monotone in $\\mathrm{TE}_{\\mathrm{inj}}$ at each $D$, but 18.8% of cross-$D$ cell pairs
invert, and its absolute scale swings $\\pm 60\\%$ with the probe's free ridge. A pooled slope mixes
model behaviour with a $D$-dependent probe bias. $\\mathrm{TE}_{\\mathrm{inj}}$ is the primary axis
and `spearman_inj` the primary rank statistic.

`kld_shuffle_ratio` is a **readout, not a gate** (Finding F2): $\\mathrm{KL}(q \\,\\|\\, p)$ measures
"the source moved my belief", not "...correctly", so a deranged source still moves it. It is not
expected to approach 0. The headline source-usage gate is the prediction-space ordering
$\\mathcal{L}_{\\mathrm{feat}} < \\mathcal{L}_{\\mathrm{base}} < \\mathcal{L}_{\\mathrm{feat}}^{\\pi(U)}$,
reported as `ordering_pass` / `shuffle_penalty`.

`pred_gain` is $\\mathcal{L}_{\\mathrm{base}} - \\mathcal{L}_{\\mathrm{feat}}$ from `eval`, **not** the
training-time `pred_gap` (which lives in `logs/*/metrics.csv`).

`cmi_recovery_rho` is $\\rho(\\mathrm{CMI}_{\\mathrm{latent}}, \\mathrm{TE}_{\\mathrm{inj}})$ and is
**model-free** — it reads only regenerated ground-truth latents, so it is identical across arms up
to the anchor subsample. It is a check on the CMI estimator, not on the model. `rho_kbar_cmi` and
`cmi_bias` are the model-dependent CMI readouts, and only rank-level claims are made for them.

`cond_r2_target_state` is the **held-out** $R^2$ of a linear fit of $Y^+$ on the model's
`target_state`. A negative value means that conditioning does not transfer across samples, and
`cmi_bias_ok` then reads `FAIL`: the arm's `cmi_bias` measures a non-transferable summary rather
than a worse one, and must not be compared with an arm whose conditioning does transfer. This is
what a time-pooling `GroupNorm` does to `target_state` — the very leak `causal_norm: true` closes.
"""


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
def _dig(obj: Optional[Dict[str, Any]], dotted: str) -> Any:
    r"""Resolve a dotted path, or an ``"a - b"`` difference of two paths, to a scalar.

    Args:
        obj: The parsed artifact, or ``None`` when it is absent.
        dotted: ``"calibration.gamma_inj"``, or ``"a.b - c.d"`` for a derived difference.

    Returns:
        The value, or ``None`` when any component of the path is missing. ``None`` renders as
        ``n/a``, which is how a whole missing artifact degrades one column at a time.
    """
    if obj is None:
        return None
    if " - " in dotted:
        left, right = (_dig(obj, part.strip()) for part in dotted.split(" - ", 1))
        if left is None or right is None:
            return None
        return float(left) - float(right)

    cursor: Any = obj
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _render_value(column: str, value: Any) -> str:
    r"""Format one cell: a verdict for the gate columns, ``_fmt`` for everything else."""
    if value is None:
        return "n/a"
    if column in _BOOL_COLUMNS:
        return "pass" if bool(value) else "**FAIL**"
    return _fmt(value, ".4g")


def _load_arm(tag_root: Path, arm: str, split: str) -> Dict[str, Optional[Dict[str, Any]]]:
    r"""Load the three source artifacts for one arm's split; each may be absent."""
    split_dir = Path(tag_root) / arm / split
    return {key: _load_json(split_dir / name) for key, name in _ARTIFACT_FILES.items()}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_arms_report(
    arms: Sequence[str],
    tag_root: Path,
    *,
    split: str,
    tag: str = "",
) -> str:
    r"""Assemble the cross-arm markdown table.

    Renders with one, two or three arms present. Every missing artifact renders ``n/a`` for the
    columns it sources, independently of the others, so a run that skipped ``--stage cmi`` still
    produces a complete calibration table.

    Args:
        arms: Arm names, in ladder order.
        tag_root: ``results/<tag>/``; each arm's artifacts live under ``<arm>/<split>/``.
        split: The split to tabulate.
        tag: The experiment tag, for the heading.

    Returns:
        The markdown document.
    """
    tag_root = Path(tag_root)
    lines: List[str] = [
        f"# Cross-arm report — `{tag or tag_root.name}` (split `{split}`)",
        "",
        _PREAMBLE,
        "",
    ]

    loaded = {arm: _load_arm(tag_root, arm, split) for arm in arms}
    present = [a for a in arms if any(v is not None for v in loaded[a].values())]
    if not present:
        lines += [
            f"> n/a — no arm has been graded on split `{split}`. Run `--stage eval` first.",
            "",
        ]
        return "\n".join(lines)

    for arm in arms:
        if arm not in present:
            lines.append(f"> `{arm}`: not graded on split `{split}`; its row renders `n/a`.")
    if len(present) != len(arms):
        lines.append("")

    header = ["arm", "model_class"] + [c[0] for c in ARMS_REPORT_COLUMNS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for arm in arms:
        artifacts = loaded[arm]
        metrics = artifacts.get(ARTIFACT_METRICS)
        report_rel = os.path.relpath(tag_root / arm / split / "report.md", tag_root)
        model_class = (metrics or {}).get("model_class") or "n/a"
        row = [f"[`{arm}`]({Path(report_rel).as_posix()})", f"`{model_class}`"]
        for column, artifact, dotted in ARMS_REPORT_COLUMNS:
            row.append(_render_value(column, _dig(artifacts.get(artifact), dotted)))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", _FOOTNOTES.strip(), ""]

    missing = sorted({
        _ARTIFACT_FILES[key]
        for arm in present for key, value in loaded[arm].items() if value is None
    })
    if missing:
        lines += [
            "Sources not found for at least one arm (their columns read `n/a`): "
            + ", ".join(f"`{m}`" for m in missing),
            "",
        ]
    return "\n".join(lines)


def run_arms_report_stage(ctx: StageContext) -> int:
    r"""Write ``results/<tag>/arms_report.md``, once, across every configured arm.

    Model-free and cross-arm: it reads each arm's artifacts and writes at the tag root, so it is
    registered ``model_dependent=False`` (``--stage arms_report`` needs no ``--arm``) and the
    driver dispatches it once, after the per-arm sweep.

    Args:
        ctx: The stage context. ``ctx.arm`` is always ``None`` here.

    Returns:
        ``0``. Registered ``fatal=False``.
    """
    arms = list((ctx.config.get("arms") or {}).keys())
    tag_root = ctx.tag_root()
    tag = str((ctx.config.get("experiment") or {}).get("tag", ctx.benchmark))

    if not arms:
        logger.warning("arms_report: config has no `arms` block; nothing to tabulate.")
        return 0

    for split in ctx.splits():
        # A split with no artifacts at all is skipped rather than tabulated as an empty table.
        if not any((tag_root / arm / split).is_dir() for arm in arms):
            logger.info("arms_report: no arm has a `{}` directory; skipping.", split)
            continue
        text = build_arms_report(arms, tag_root, split=split, tag=tag)
        out = tag_root / ("arms_report.md" if len(ctx.splits()) == 1
                          else f"arms_report_{split}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("arms_report: wrote {} ({} arms, split {})", out, len(arms), split)
    return 0


register_stage(
    StageSpec(
        "arms_report", _STAGE_ORDER, True, False, run_arms_report_stage,
        fatal=False,
        help="cross-arm gate index -> results/<tag>/arms_report.md",
    )
)
