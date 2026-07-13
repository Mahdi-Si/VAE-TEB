r"""S7-T06: the cross-arm gate index for ``synthetic_v4`` (the raw-domain G0 answer, in one table).

The ~8 arms share one cache, one seed set, one objective, and one set of splits; only their
front-end / encoder machinery differs, so every difference in the table below is attributable to the
model. This module assembles ``results/<tag>/arms_report_v4.md``: one row per arm, one column per
gate, every value traced to a named dotted key in that arm's ``metrics.json``. Reading the arms side
by side is the whole point of the ladder -- ``frontend_noncausal`` (the G0-in-front-end negative
control) is expected to inflate the null and degrade the calibration relative to ``prod``.

It mirrors the scattering-domain :mod:`arms_report_v3` (self-registered ``model_dependent=False``
stage; ``_dig`` resolves a dotted path or an ``"a - b"`` derived difference to ``None`` -> ``n/a`` so
a missing artifact degrades one column at a time), but the columns read the **single-axis** v4
schema (un-suffixed ``calibration.gamma``, top-level ``null_cell_gate``, ``prediction_controls``
with the ``shuffled`` control, ``lag_recovery``, ``te_raw_gate``); there is no ``te_scat`` /
``cmi`` / ``lag_intervention`` artifact in the first cut.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.final_report_v2 import (
    _fmt,
    _load_json,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v4 import (
    StageContextV4,
    StageSpecV4,
    _resolve_splits,
    register_stage_v4,
)

logger = logging.getLogger(__name__)

_STAGE_ORDER = 60

#: The single per-arm artifact the v4 first cut tabulates (per split).
_METRICS_FILE = "metrics.json"

#: ``(column_label, dotted_key)``. A ``dotted_key`` of the form ``"a - b"`` is a derived difference
#: of two dotted paths (``None`` when either is missing). Asserted against the emitted header by
#: ``test_arms_report_v4.py``.
ARMS_REPORT_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("gamma", "calibration.gamma"),
    ("r2", "calibration.r2"),
    ("spearman", "calibration.spearman"),
    ("null_kbar", "null_cell_gate.mean"),
    ("null_gate", "null_cell_gate.pass"),
    ("pred_gain",
     "prediction_controls.overall.base_loss - prediction_controls.overall.feat_loss"),
    ("ordering_pass", "prediction_controls.overall.ordering_pass_shuffled"),
    ("shuffle_penalty", "prediction_controls.overall.shuffle_penalty_shuffled"),
    ("mean_lag_mass", "lag_recovery.mean_lag_mass"),
    ("lag_gate", "lag_recovery.mean_lag_mass_pass"),
    ("te_raw_gate", "te_raw_gate.gate.passed"),
)

#: Columns whose value is a pass/fail verdict rather than a number.
_BOOL_COLUMNS = frozenset({"null_gate", "ordering_pass", "lag_gate", "te_raw_gate"})

_PREAMBLE = """The arms share one cache, one seed set, one objective, and one set of splits; only the
front-end / encoder `model_kwargs` differ, so every difference below is attributable to the model.

| contrast | isolates |
|---|---|
| `prod` → `frontend_noncausal` | **G0-in-front-end**: a time-pooling (leaky) front-end norm makes a token a function of the future, inflating the null and degrading the calibration against a known ground-truth TE |
| `prod` → `disable_source` | the source pathway itself: with no UP the posterior collapses to the prior, K̄ ≈ 0 |
| `prod` → `single_stride` / `no_antialias` / `no_gated` | front-end architecture ablations (stride factorisation, anti-alias, gating) |
| `prod` → `am_carrier_prod` | whether the learned causal front end recovers an AM-modulated coupling (trained on the am_carrier cache) |

`prod` is the causal-front-end headline arm; `frontend_noncausal` is the G0 negative control.
"""

_FOOTNOTES = """
`pred_gain` is L_base − L_feat from `eval` (the source-usefulness margin), not the training-time
`pred_gap`. `ordering_pass` is the discriminating source-usage gate
(L_feat < L_base < L_feat^π(U)); `shuffle_penalty` = L_feat^π(U) − L_feat should be > 0 on a model
that uses the true source. `te_raw_gate` is the model-free Sprint-1 realizability preflight: it
confirms the injected TE is present in the raw waveform, so a low K̄ is attributable to the model,
not the data. TE_inj is the sole calibration axis (there is no TE_scat in the raw pipeline).
"""


def _dig(obj: Optional[Dict[str, Any]], dotted: str) -> Any:
    r"""Resolve a dotted path, or an ``"a - b"`` difference of two paths, to a scalar.

    Args:
        obj: The parsed artifact, or ``None`` when absent.
        dotted: ``"calibration.gamma"``, or ``"a.b - c.d"`` for a derived difference.

    Returns:
        The value, or ``None`` when any path component is missing (``None`` renders ``n/a``, which
        is how a whole missing artifact degrades one column at a time).
    """
    if obj is None:
        return None
    if " - " in dotted:
        left, right = (_dig(obj, part.strip()) for part in dotted.split(" - ", 1))
        if left is None or right is None:
            return None
        try:
            return float(left) - float(right)
        except (TypeError, ValueError):
            return None
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


def build_arms_report_v4(arms: Sequence[str], tag_root: Path, *, split: str,
                         tag: str = "") -> str:
    r"""Assemble the cross-arm markdown gate table for one split.

    Renders with any number of arms present; every missing ``metrics.json`` renders ``n/a`` for the
    columns it sources, independently of the others, so a sweep that graded only some arms still
    produces a complete table.

    Args:
        arms: Arm names, in ladder order.
        tag_root: ``results/<tag>/``; each arm's metrics live under ``<arm>/<split>/metrics.json``.
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

    loaded = {arm: _load_json(tag_root / arm / split / _METRICS_FILE) for arm in arms}
    present = [a for a in arms if loaded[a] is not None]
    if not present:
        lines += [f"> n/a — no arm has been graded on split `{split}`. Run `--stage eval` first.",
                  ""]
        return "\n".join(lines)

    for arm in arms:
        if arm not in present:
            lines.append(f"> `{arm}`: not graded on split `{split}`; its row renders `n/a`.")
    if len(present) != len(arms):
        lines.append("")

    header = ["arm", "model_class"] + [c[0] for c in ARMS_REPORT_COLUMNS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for arm in arms:
        metrics = loaded[arm]
        report_rel = os.path.relpath(tag_root / arm / split / "report.md", tag_root)
        model_class = (metrics or {}).get("model_class") or "n/a"
        row = [f"[`{arm}`]({Path(report_rel).as_posix()})", f"`{model_class}`"]
        for column, dotted in ARMS_REPORT_COLUMNS:
            row.append(_render_value(column, _dig(metrics, dotted)))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", _FOOTNOTES.strip(), ""]

    missing = [a for a in arms if loaded[a] is None]
    if missing:
        lines += ["Arms not graded on this split (their rows read `n/a`): "
                  + ", ".join(f"`{m}`" for m in missing), ""]
    return "\n".join(lines)


def run_arms_report_v4(ctx: StageContextV4) -> int:
    r"""``arms_report`` stage: write ``results/<tag>/arms_report_v4.md``, once, across every arm.

    Model-free and cross-arm: it reads each arm's per-split ``metrics.json`` and writes at the tag
    root, so it is registered ``model_dependent=False`` (``--stage arms_report`` needs no ``--arm``)
    and the driver dispatches it once, after the per-arm sweep. Non-fatal.

    Args:
        ctx: The stage context (``ctx.arm`` is always ``None`` here).

    Returns:
        ``0``.
    """
    arms = list((ctx.config.get("arms") or {}).keys())
    tag_root = ctx.results_dir()
    tag = str((ctx.config.get("experiment") or {}).get("tag", ctx.benchmark))
    if not arms:
        logger.warning("arms_report_v4: config has no `arms` block; nothing to tabulate.")
        return 0

    splits = _resolve_splits(ctx.config, ctx.benchmark, ctx.split)
    for split in splits:
        # Skip a split with no graded arm rather than tabulate an empty table.
        if not any((tag_root / arm / split).is_dir() for arm in arms):
            logger.info("arms_report_v4: no arm has a `%s` directory; skipping.", split)
            continue
        text = build_arms_report_v4(arms, tag_root, split=split, tag=tag)
        out = tag_root / ("arms_report_v4.md" if len(splits) == 1
                          else f"arms_report_v4_{split}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[arms_report] wrote {out} ({len(arms)} arms, split {split})")
    return 0


register_stage_v4(StageSpecV4(
    name="arms_report",
    run=run_arms_report_v4,
    order=_STAGE_ORDER,
    model_dependent=False,
    fatal=False,
    help="cross-arm gate index -> results/<tag>/arms_report_v4.md",
))
