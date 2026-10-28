r"""Print what each pair of healthy fractions gives, and recommend one pair.

``create_new_pipeline.select_classification_guids`` takes ``bg_cls_fraction`` $f_{BG}$ and
``no_bg_cls_fraction`` $f_{noBG}$: the fraction of each eligible healthy subgroup that goes to the
classification pool. This script runs that same function on the screening CSV for every pair on
a grid, so every number it prints is the number the real selection produces. It writes no
dataset file.

For each pair it prints:

* the pool: BG and no-BG GUIDs, the ``healthy_no_bg_cs`` GUIDs, and healthy : unhealthy;
* one fold: ``healthy_no_bg_cs`` train GUIDs (80 % of the pool), healthy test GUIDs, and the
  unhealthy prevalence in test;
* the extra test GUIDs, and the eligible healthy GUIDs left for pretraining.

**The recommendation rule.** Only equal fractions are candidates: with $f_{BG} = f_{noBG}$ the
train/val healthy mix equals the test mix and the population mix, so the classifier sees no
healthy distribution shift between validation and test. From these, the script takes the largest
$f$ whose healthy : unhealthy ratio is at most ``max_healthy_per_unhealthy`` and which leaves at
least ``min_pretrain_share`` of the eligible healthy GUIDs for pretraining. Both limits are
judgment calls; change them in ``RUN_ARGS``.

Needs ``create_new_pipeline.py`` importable (``torch`` and the production packages), as on the
production box. On that box this file may sit in the same folder as ``create_new_pipeline.py``.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

#: Repository root: ``hdf5_dataset/new_pipeline/choose_selection_fractions.py`` -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the launch helper nor the pipeline can be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # Repository layout, and a box with the teb_vae package.
    from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402
except ModuleNotFoundError as _exc:
    if not str(_exc.name).startswith("teb_vae"):
        raise
    from launch import missing_required, resolve_launch_args  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _pipeline() -> Any:
    """Import ``create_new_pipeline``: flat first (production), then from the repository package.

    Returns:
        The module.
    """
    try:
        return importlib.import_module("create_new_pipeline")
    except ModuleNotFoundError as exc:
        if exc.name != "create_new_pipeline":
            raise
    return importlib.import_module("hdf5_dataset.new_pipeline.create_new_pipeline")


def evaluate_pair(cnp: Any, screening: pd.DataFrame, bg_fraction: float,
                  no_bg_fraction: float, test_mode: str,
                  healthy_eligibility: str = "eligible_3h") -> Dict[str, Any]:
    """Run the real selection for one pair and summarize it.

    Args:
        cnp: The ``create_new_pipeline`` module.
        screening: The screening CSV.
        bg_fraction: $f_{BG}$.
        no_bg_fraction: $f_{noBG}$.
        test_mode: ``'augmented'`` or ``'holdout'``.
        healthy_eligibility: ``'eligible_3h'`` or ``'eligible_2h'``.

    Returns:
        One row of the table.
    """
    result = cnp.select_classification_guids(
        screening, test_mode=test_mode, bg_cls_fraction=bg_fraction,
        no_bg_cls_fraction=no_bg_fraction, verbose=False,
        healthy_eligibility=healthy_eligibility)
    stats, pool = result["stats"], result["trainval"]
    k = cnp.N_FOLDS
    n_unhealthy = sum(len(pool.get(sg, [])) for sg in cnp.UNHEALTHY_SUBGROUPS)
    n_bg = sum(len(pool.get(sg, [])) for sg in cnp.BG_SUBGROUPS)
    n_no_bg = sum(len(pool.get(sg, [])) for sg in cnp.NO_BG_SUBGROUPS)
    healthy_test = sum(stats["healthy_test_per_fold"].values())
    unhealthy_test = (sum(len(result["test"].get(sg, [])) for sg in cnp.UNHEALTHY_SUBGROUPS)
                      if test_mode == "holdout" else n_unhealthy / k)
    n_pretrain = sum(stats["n_pretraining"].values())
    n_eligible = sum(stats["eligible_healthy"].values())
    # Augmented: KFold leaves (K-1)/K for train+val, and the inner split takes 1/(K-1) of it.
    train_share = (k - 2) / k if test_mode == "augmented" else (k - 1) / k
    return {
        "f_bg": bg_fraction, "f_no_bg": no_bg_fraction,
        "pool_unhealthy": n_unhealthy, "pool_bg": n_bg, "pool_no_bg": n_no_bg,
        "pool_no_bg_cs": len(pool.get("healthy_no_bg_cs", [])),
        "healthy_per_unhealthy": (n_bg + n_no_bg) / max(n_unhealthy, 1),
        "train_no_bg_cs_per_fold": len(pool.get("healthy_no_bg_cs", [])) * train_share,
        "healthy_test_per_fold": healthy_test,
        "test_prevalence": unhealthy_test / max(unhealthy_test + healthy_test, 1e-9),
        "extra_test": sum(stats["n_extra_test"].values()),
        "pretrain_left": n_pretrain,
        "pretrain_share": n_pretrain / max(n_eligible, 1),
    }


def recommend(table: pd.DataFrame, max_healthy_per_unhealthy: float,
              min_pretrain_share: float) -> Optional[pd.Series]:
    """Apply the rule of the module docstring.

    Args:
        table: Output of :func:`build_table`.
        max_healthy_per_unhealthy: The largest healthy : unhealthy ratio of the pool.
        min_pretrain_share: The smallest share of eligible healthy GUIDs left for pretraining.

    Returns:
        The chosen row, or ``None`` if no equal pair meets both limits.
    """
    ok = table[(table["f_bg"] == table["f_no_bg"])
               & (table["healthy_per_unhealthy"] <= max_healthy_per_unhealthy)
               & (table["pretrain_share"] >= min_pretrain_share)]
    return None if ok.empty else ok.sort_values("f_bg").iloc[-1]


def build_table(cnp: Any, screening: pd.DataFrame, fractions: Sequence[float],
                test_mode: str, equal_only: bool,
                healthy_eligibility: str = "eligible_3h") -> pd.DataFrame:
    """Evaluate every pair of the grid.

    Args:
        cnp: The ``create_new_pipeline`` module.
        screening: The screening CSV.
        fractions: The grid values, used for both fractions.
        test_mode: ``'augmented'`` or ``'holdout'``.
        equal_only: Evaluate the pairs with $f_{BG} = f_{noBG}$ only.
        healthy_eligibility: ``'eligible_3h'`` or ``'eligible_2h'``.

    Returns:
        One row per pair.
    """
    rows: List[Dict[str, Any]] = []
    for f_bg in fractions:
        for f_no_bg in fractions:
            if equal_only and f_bg != f_no_bg:
                continue
            rows.append(evaluate_pair(cnp, screening, f_bg, f_no_bg, test_mode,
                                      healthy_eligibility))
            logger.info(f"evaluated f_bg={f_bg}, f_no_bg={f_no_bg}")
    return pd.DataFrame(rows)


def main(screening_csv_path: Optional[str] = None, fractions: Optional[Sequence[float]] = None,
         test_mode: Optional[str] = None, equal_only: Optional[bool] = None,
         max_healthy_per_unhealthy: Optional[float] = None,
         min_pretrain_share: Optional[float] = None,
         output_csv_path: Optional[str] = None,
         healthy_eligibility: Optional[str] = None) -> int:
    """Print the eligible counts, the table and the recommendation.

    Args:
        screening_csv_path: ``guid_screening_results.csv``.
        fractions: Grid values. Default: :data:`DEFAULT_FRACTIONS`.
        test_mode: ``'augmented'`` or ``'holdout'``. Default: ``'augmented'``.
        equal_only: Evaluate equal pairs only. Default: off.
        max_healthy_per_unhealthy: Limit of the rule. Default: 7.0.
        min_pretrain_share: Limit of the rule. Default: 0.8.
        output_csv_path: Also write the table here. Default: no file.
        healthy_eligibility: ``'eligible_3h'`` or ``'eligible_2h'``. Default: ``'eligible_3h'``.

    Returns:
        The process exit code.
    """
    cnp = _pipeline()
    fractions = list(fractions or DEFAULT_FRACTIONS)
    test_mode = cnp.validate_test_mode(test_mode or cnp.AUGMENTED)
    max_ratio = 7.0 if max_healthy_per_unhealthy is None else float(max_healthy_per_unhealthy)
    min_share = 0.8 if min_pretrain_share is None else float(min_pretrain_share)

    screening = pd.read_csv(screening_csv_path)
    usable = screening[~screening["error"].astype(bool)]
    counts = pd.DataFrame({
        "rows": screening.groupby("subgroup").size(),
        "errors": screening.groupby("subgroup")["error"].sum(),
        "eligible_2h": usable.groupby("subgroup")["eligible_2h"].sum(),
        "eligible_3h": usable.groupby("subgroup")["eligible_3h"].sum(),
    }).fillna(0).astype(int)
    print(f"Screening CSV, per subgroup (unhealthy use eligible_2h, healthy "
          f"{healthy_eligibility or cnp.HEALTHY_ELIGIBILITY}):")
    print(counts.to_string())
    print()

    table = build_table(cnp, screening, fractions, test_mode, bool(equal_only),
                        healthy_eligibility or cnp.HEALTHY_ELIGIBILITY)
    if output_csv_path:
        table.to_csv(output_csv_path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print()

    choice = recommend(table, max_ratio, min_share)
    if choice is None:
        print(f"No equal pair has healthy : unhealthy <= {max_ratio} and leaves >= "
              f"{min_share:.0%} for pretraining. Relax a limit or extend the grid.")
        return 1
    print(f"Recommended: bg_cls_fraction = no_bg_cls_fraction = {choice['f_bg']}")
    print(f"  pool healthy : unhealthy = {choice['healthy_per_unhealthy']:.2f} "
          f"(limit {max_ratio}); healthy_no_bg_cs in train per fold = "
          f"{choice['train_no_bg_cs_per_fold']:.0f}")
    print(f"  per fold: {choice['healthy_test_per_fold']:.0f} healthy test GUIDs, unhealthy "
          f"prevalence {choice['test_prevalence']:.1%}")
    print(f"  pretraining keeps {choice['pretrain_left']} GUIDs "
          f"({choice['pretrain_share']:.1%} of eligible healthy; limit {min_share:.0%})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. No defaults: :data:`RUN_ARGS` fills every gap.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/new_pipeline/choose_selection_fractions.py",
        description="Print what each pair of healthy fractions gives, and recommend one.")
    p.add_argument("--screening-csv-path", dest="screening_csv_path")
    p.add_argument("--fractions", dest="fractions", nargs="+", type=float)
    p.add_argument("--test-mode", dest="test_mode", choices=("augmented", "holdout"))
    p.add_argument("--equal-only", dest="equal_only", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--max-healthy-per-unhealthy", dest="max_healthy_per_unhealthy", type=float)
    p.add_argument("--min-pretrain-share", dest="min_pretrain_share", type=float)
    p.add_argument("--output-csv-path", dest="output_csv_path")
    p.add_argument("--healthy-eligibility", dest="healthy_eligibility",
                   choices=("eligible_2h", "eligible_3h"))
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If the screening CSV was supplied by neither route.
    """
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("screening_csv_path",))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # str -- guid_screening_results.csv: one row per .mat file of every subgroup folder. Required.
    "screening_csv_path": r"/data/deid/isilon/MS_model/new_ph_dataset/guid_screening_results.csv",

    # list[float] | None -- grid values, used for both fractions.
    #        (None -> 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    "fractions": None,

    # "augmented" | "holdout"   (None -> "augmented")
    "test_mode": None,

    # bool | None -- evaluate equal pairs only: 6 selections instead of 36.  (None -> False)
    "equal_only": None,

    # float | None -- the rule's largest healthy : unhealthy ratio of the pool.  (None -> 7.0)
    "max_healthy_per_unhealthy": None,

    # float | None -- the rule's smallest share of eligible healthy GUIDs left for pretraining.
    #        (None -> 0.8)
    "min_pretrain_share": None,

    # str | None -- also write the table as a CSV.  (None -> no file)
    "output_csv_path": None,

    # "eligible_2h" | "eligible_3h" | None -- the healthy classification rule; use the value you
    #        will give create_new_pipeline.py.  (None -> "eligible_3h")
    "healthy_eligibility": "eligible_2h",
}


if __name__ == "__main__":
    sys.exit(_cli())
