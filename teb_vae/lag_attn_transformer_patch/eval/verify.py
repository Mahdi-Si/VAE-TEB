r"""The acceptance gate for one patch-cell run, and its arm-leaves table. Reads files only, no torch.

**The gate**::

    python -m teb_vae.lag_attn_transformer_patch.eval.verify <run>/eval_results/summary.json

Delegated in full to :mod:`teb_vae.lag_attn_cfs.eval.verify`: the exit code, the weight-load check,
the ``pred_gap`` column, the verdicts and the sanity block are properties of the shared objective and
registry. A ``pred_gap`` here is in nats over the patch block (H x 2 = 60 cells per anchor), so it is
**not** comparable with a CFS cell's H x C_keep block; no cross-cell table is emitted.

**The table**::

    python -m teb_vae.lag_attn_transformer_patch.eval.verify --runs <dir> --out RESULTS_arms.md

The shared arm inventory plus one column per leaf a shipped ``configs/sweep_*.yaml`` changes, read
from each run's own resolved config. The CFS tiling and cross-cell tables are not shipped here.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from teb_vae.lag_attn_cfs.eval import launch, verify as shared

DEFAULT_ARMS_OUT = "RESULTS_arms.md"

#: The leaf each shipped patch ``sweep_*.yaml`` changes.
SWEPT_ARM_LEAVES: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
    (leaf, ("model_config", "VAE_model", leaf))
    for leaf in ("lag_kv_source", "persistence_residual", "source_validity", "warmup_period")
)


def main(summary_path: Any, json_out: Optional[Any] = None) -> int:
    """Verify one run against the shared criteria; non-zero when any criterion failed."""
    return shared.main(summary_path, json_out)


def build_arm_tables(arms: Sequence[Dict[str, Any]]) -> str:
    """The arm inventory and the arm-leaves table, as one markdown document."""
    lines: List[str] = [
        "# Arm comparison (patch cell)",
        "",
        f"From {len(arms)} finished run(s); every value is read from the run's own resolved config "
        f"and headline block. `pred_gap` is `{shared.PRED_GAP_COLUMN}` (nats per anchor over the "
        "H x 2 patch block).",
        "",
        "## Arm inventory",
        "",
    ]
    lines += shared.build_arm_inventory(arms)
    lines += ["", "## Arm leaves", ""]
    lines += shared._table(
        ["Run", *(f"`{leaf}`" for leaf, _ in SWEPT_ARM_LEAVES), "`pred_gap`", f"`{shared.KL_COLUMN}`"],
        [
            [arm["run"],
             *(shared._render(shared._dig_config(arm["config"], path)) for _, path in SWEPT_ARM_LEAVES),
             shared._headline_cell(arm, shared.PRED_GAP_COLUMN),
             shared._headline_cell(arm, shared.KL_COLUMN)]
            for arm in sorted(arms, key=lambda record: record["run"])
        ],
    )
    lines.append("")
    return "\n".join(lines)


def compare_arms(runs_dir: Any, out_path: Any) -> int:
    """Scan ``runs_dir`` for summaries and write :func:`build_arm_tables` to ``out_path``."""
    return shared.compare_arms(runs_dir, out_path, build_arm_tables)


def build_parser() -> argparse.ArgumentParser:
    """Both entry points' parser."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_patch.eval.verify",
        description="Check a finished patch eval run, or tabulate a directory of runs.",
    )
    parser.add_argument("summary", nargs="?", default=None, help="A run's summary.json (the gate).")
    parser.add_argument("--json-out", dest="json_out", default=None, help="Gate only: report path.")
    parser.add_argument("--runs", default=None, help="Directory of finished runs: emit the table.")
    parser.add_argument("--out", default=None, help=f"Table only. Default: {DEFAULT_ARMS_OUT}.")
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Dispatch between the gate and the table. Returns the process exit code."""
    values, _sources = launch.resolve_launch_args(build_parser(), RUN_ARGS, argv)
    if values["runs"] is not None:
        if values["summary"] is not None:
            print("give either a summary path or --runs, not both.")
            return 2
        return compare_arms(values["runs"], values["out"] or DEFAULT_ARMS_OUT)
    if values["summary"] is None:
        print("a summary path is required unless --runs names a directory (or set RUN_ARGS).")
        return 2
    return main(values["summary"], values["json_out"])


RUN_ARGS: Dict[str, Any] = {"summary": None, "json_out": None, "runs": None, "out": None}


if __name__ == "__main__":
    sys.exit(_cli())
