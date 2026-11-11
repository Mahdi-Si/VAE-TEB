r"""Summarise the pilot fine-tunes: every arm against a reference arm of the same seeds.

For each arm the script averages the per-anchor scores over the seeds, joins the anchor
covariates of the Phase 2 tolerance pass (reward with and without the UA rule, invalid raw time
in the lag window, running-gap flag) and the linear baselines, and reports paired differences
against the reference with recording-level bootstrap intervals.

**Scopes.** ``all`` anchors; ``onset60``, anchors within 60 s after any detected onset (the
reward without the UA rule), so every arm is read on the same anchors; ``dropped_ended``,
anchors of contractions that the UA rule drops, with invalid UA in the lag window and no gap
running at the anchor; ``dropped_inside``, anchors of dropped contractions inside a running gap;
``clean_rewarded``, rewarded anchors with a fully valid lag window. Each scope is also split
into the unseen cohorts (acidosis, HIE) and the possibly-seen cohort (healthy).

**Skill against the baselines.** The squared error of the full branch per block, divided by the
squared error of the own-channel AR(6) and of the UA-lag ridge on the same anchors and cells.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {
    "tag": "",                       # folder suffix: "" for the select pilots, "_test" for the final pass
    "reference": "control",
    "arms": None,                    # None: every arm found in the folder
    "seeds": [0, 1],
    "covariates": "tolerance_anchors.parquet",
    "baselines": "baselines_select.parquet",
}
KEYS = ["guid", "epoch", "anchor"]


def load_arm(folder: Path, arm: str, seeds: List[int]) -> Optional[pd.DataFrame]:
    """Average one arm's per-anchor scores over its seeds.

    Args:
        folder: The pilot output folder.
        arm: The arm name.
        seeds: The seeds to average; a missing seed is skipped.

    Returns:
        One row per anchor with the seed-mean of every score column and ``n_seeds``, or ``None``
        when no seed exists.
    """
    frames = [pd.read_parquet(folder / f"{arm}_s{seed}.parquet") for seed in seeds
              if (folder / f"{arm}_s{seed}.parquet").exists()]
    if not frames:
        return None
    stacked = pd.concat(frames, ignore_index=True)
    stacked["epoch"] = stacked["epoch"].round(0)
    out = stacked.groupby(KEYS, as_index=False).mean(numeric_only=True)
    out["n_seeds"] = len(frames)
    return out


def scopes_of(table: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Boolean masks of the reporting scopes.

    Args:
        table: The joined per-anchor table.

    Returns:
        ``{scope: mask}``.
    """
    dropped = (table["omega_trained"] <= 0.5) & (table["omega_free"] > 0.5)
    gap = table["raw_invalid_s_window"] > 0
    base = {
        "all": np.ones(len(table), dtype=bool),
        "onset60": (table["omega_free"] > 0.5).to_numpy(),
        "clean_rewarded": ((table["omega_trained"] > 0.5) & (table["keys_invalid_cov"] == 0)).to_numpy(),
        "dropped_ended": (dropped & gap & (table["inside_gap"] == 0)).to_numpy(),
        "dropped_inside": (dropped & (table["inside_gap"] == 1)).to_numpy(),
        "not_onset60": (table["omega_free"] <= 0.5).to_numpy(),
    }
    unseen = (table["cohort"] != "healthy").to_numpy()
    out = dict(base)
    for name in ("all", "onset60", "dropped_ended"):
        out[f"{name}/unseen"] = base[name] & unseen
        out[f"{name}/possibly_seen"] = base[name] & ~unseen
    return out


def calibration_of(table: pd.DataFrame) -> Dict[str, float]:
    """Coverage, floor share and the official tail statistic from pooled cell counts.

    Args:
        table: A per-anchor score table.

    Returns:
        A dictionary of coverage values per block and the worst relative tail error.
    """
    nominal = {1: 0.6826895, 2: 0.9544997, 3: 0.9973002}
    out: Dict[str, float] = {}
    cells = table["n_st"].sum() + table["n_ph"].sum()
    worst = 0.0
    for k, target in nominal.items():
        for block in ("st", "ph"):
            out[f"cov{k}_{block}"] = float(table[f"in{k}_{block}"].sum() / table[f"n_{block}"].sum())
        tail = 1.0 - float((table[f"in{k}_st"].sum() + table[f"in{k}_ph"].sum()) / cells)
        worst = max(worst, abs(tail - (1 - target)) / (1 - target))
    out["worst_relative_tail_error"] = worst
    for block in ("st", "ph"):
        out[f"floor_{block}"] = float(table[f"floor_{block}"].sum() / table[f"n_{block}"].sum())
    return out


def skill_of(table: pd.DataFrame) -> Dict[str, float]:
    """Mean squared error of the full branch and its ratio to the linear baselines.

    Args:
        table: The joined table with the baseline columns.

    Returns:
        MSE per block and horizon range, and the ratios to ``own_ar`` and ``ridge_yu``.
    """
    out: Dict[str, float] = {}
    for prefix, count in (("sse", "n"), ("sse0", "n0"), ("sse5", "n5")):
        for block in ("st", "ph"):
            cells = table[f"{count}_{block}"].sum()
            model = table[f"{prefix}_{block}_full"].sum() / cells
            out[f"mse_{prefix}_{block}"] = float(model)
            for name in ("own_ar", "ridge_yu"):
                column = f"{prefix}_{block}_{name}"
                if column in table:
                    base_cells = table[f"{count}_{block}_bl"].sum()
                    out[f"ratio_{prefix}_{block}_{name}"] = float(model / (table[column].sum() / base_cells))
    return out


def main(tag: str, reference: str, arms: Optional[List[str]], seeds: List[int], covariates: str,
         baselines: str) -> Dict[str, Any]:
    """Build and print the pilot summary.

    Args:
        tag: Suffix of the pilot folder.
        reference: The reference arm.
        arms: Arms to report; ``None`` reports every arm in the folder.
        seeds: Seeds to average.
        covariates: File name of the anchor covariates under ``out/phase2``.
        baselines: File name of the baseline table under ``out/phase4``.

    Returns:
        The summary dictionary.
    """
    folder = S.OUT / "phase4" / f"pilots{tag}"
    if arms is None:
        arms = sorted({p.stem.rsplit("_s", 1)[0] for p in folder.glob("*_s*.parquet")})
    cov = pd.read_parquet(S.OUT / "phase2" / covariates,
                          columns=KEYS + ["omega_trained", "omega_free", "raw_invalid_s_window", "inside_gap",
                                          "keys_invalid", "cohort", "code"])
    cov = cov.rename(columns={"keys_invalid": "keys_invalid_cov"})
    cov["epoch"] = cov["epoch"].round(0)
    base_path = S.OUT / "phase4" / baselines
    base = None
    if base_path.exists():
        base = pd.read_parquet(base_path)
        base["epoch"] = base["epoch"].round(0)
        base = base.rename(columns={c: f"{c}_bl" for c in ("n_st", "n_ph", "n0_st", "n0_ph", "n5_st", "n5_ph")})
        base = base.drop(columns=[c for c in ("cohort", "code") if c in base])
    ref = load_arm(folder, reference, seeds)
    if ref is None:
        raise FileNotFoundError(f"no output of the reference arm {reference!r} in {folder}")
    summary: Dict[str, Any] = {"reference": reference, "seeds": seeds, "arms": {}}
    for arm in arms:
        table = load_arm(folder, arm, seeds)
        if table is None:
            continue
        joined = table.merge(ref, on=KEYS, suffixes=("", "_ref")).merge(cov, on=KEYS, how="inner")
        if base is not None:
            joined = joined.merge(base, on=KEYS, how="left")
        entry: Dict[str, Any] = {"n_seeds": int(table["n_seeds"].iloc[0]), "n_anchors": int(len(joined)),
                                 "calibration": calibration_of(table), "skill": skill_of(joined) if base is not None else {}}
        for scope, mask in scopes_of(joined).items():
            rows = joined[mask]
            if rows["guid"].nunique() < 5:
                continue
            guid = rows["guid"].to_numpy()
            block = {"n": int(len(rows)), "n_rec": int(rows["guid"].nunique())}
            for column in ("nll_full", "nll_base", "gain", "kl", "kl_null", "g", "nll_full_st", "nll_full_ph"):
                block[column] = float(rows[column].mean())
                block[f"d_{column}"] = S.paired_bootstrap(guid, (rows[column] - rows[f"{column}_ref"]).to_numpy())
            entry[scope] = block
        summary["arms"][arm] = entry
    (folder / f"summary_vs_{reference}.json").write_text(json.dumps(summary, indent=1))

    def ci(d: Dict[str, float], digits: int = 2) -> str:
        star = "*" if (d["lo"] > 0 or d["hi"] < 0) else " "
        return f"{d['mean']:+.{digits}f} [{d['lo']:+.{digits}f},{d['hi']:+.{digits}f}]{star}"

    for scope in ("all", "all/unseen", "all/possibly_seen", "onset60", "clean_rewarded", "dropped_ended",
                  "dropped_ended/unseen", "dropped_inside", "not_onset60"):
        print(f"\n== scope {scope}  (differences: arm - {reference}; * = interval excludes 0)")
        print(f"{'arm':34s} {'dNLL_full':>26s} {'dGain':>26s} {'dKL':>24s} {'dKLnull':>24s}  G     KL   KLnull g")
        for arm, entry in summary["arms"].items():
            if scope not in entry:
                continue
            b = entry[scope]
            print(f"{arm:34s} {ci(b['d_nll_full']):>26s} {ci(b['d_gain']):>26s} {ci(b['d_kl']):>24s} "
                  f"{ci(b['d_kl_null']):>24s}  {b['gain']:5.2f} {b['kl']:5.2f} {b['kl_null']:5.2f} {b['g']:.2f}")
    print("\n== calibration and skill (select anchors, seed mean)")
    for arm, entry in summary["arms"].items():
        c, k = entry["calibration"], entry["skill"]
        print(f"{arm:34s} cov1 st/ph {c['cov1_st']:.3f}/{c['cov1_ph']:.3f} cov3 {c['cov3_st']:.4f}/{c['cov3_ph']:.4f} "
              f"tailerr {c['worst_relative_tail_error']:.2f} floor st/ph {c['floor_st']:.2f}/{c['floor_ph']:.2f} | "
              + " ".join(f"{name.replace('ratio_', '')}={value:.3f}" for name, value in k.items() if name.startswith("ratio_")))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=RUN_ARGS["tag"])
    parser.add_argument("--reference", default=RUN_ARGS["reference"])
    parser.add_argument("--arms", default=None if RUN_ARGS["arms"] is None else ",".join(RUN_ARGS["arms"]))
    parser.add_argument("--covariates", default=RUN_ARGS["covariates"])
    parser.add_argument("--baselines", default=RUN_ARGS["baselines"])
    args = parser.parse_args()
    main(args.tag, args.reference, None if args.arms is None else args.arms.split(","),
         list(RUN_ARGS["seeds"]), args.covariates, args.baselines)
