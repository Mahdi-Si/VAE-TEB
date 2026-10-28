r"""Compare the two arms of the quality-mask experiment on one set of anchors, aggregates only.

The experiment trains the model twice on the same quality-version-4 shards: with the quality masks
(``configs/prod_q4_masked.yaml``) and without them (``configs/prod_q4_no_quality_mask.yaml``). Each
checkpoint is evaluated by ``eval/run.py`` under ``eval/configs/eval_overrides_prod_q4.yaml``, so
both per-anchor tables carry the quality labels of
:func:`~teb_vae.lag_attn_cfs.eval.metrics.anchor_quality_labels` on the same dense anchors. This
module joins the two tables on ``(guid, epoch, anchor)`` and reports, per stratum:

* ``clean`` -- the masked arm's ``ua_key_open == 1`` and ``target_bad_steps == 0``: both arms see
  valid UA keys and score the same target cells, so every difference is the model's;
* ``bad_ua`` -- ``ua_key_open == 0``: the masked arm's posterior is its prior there (its KL must be
  exactly $0$), and the control's KL is what it took from UA that the mask calls bad;
* ``all`` -- every joined anchor;

each split into ``event`` (a contraction onset at most ``event_window_s`` before the anchor, from
the masked arm's detector, which reads ``up_valid``), ``quiet`` and ``any``.

Every statistic is a mean over recordings of the recording's own anchor mean, with a bootstrap
interval over recordings, because anchors of one recording are not independent. The paired
difference masked $-$ control is taken per recording before the bootstrap. The output holds
counts, means and intervals only: no GUID, no path below the run directories, no signal, so it can
leave the production box.

Run it from the PyCharm Run button after filling in :data:`RUN_ARGS`, or:

.. code-block:: bash

    python -m teb_vae.lag_attn_transformer_cfs.eval.compare_quality_arms \
        --masked-dir <masked eval output dir> --control-dir <control eval output dir> \
        --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/eval/compare_quality_arms.py`` -> up four.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and the launch helper cannot be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402

RESULTS_DIRNAME = "eval_results"
PER_ANCHOR_FILENAME = "per_anchor.parquet"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
KEY = ("guid", "epoch", "anchor")
RECORDING = "guid"
AGE_COLUMN = "seconds_since_contraction"
CLASS_COLUMN = "clinical_class"
LABELS = ("ua_key_open", "target_bad_steps")
#: Per-anchor readouts compared between the arms, all in nats per anchor.
METRICS = ("kld_per_t", "mc_pred_gap", "mean_pred_gap", "mc_nll_full_block", "mc_nll_base_block")
#: The quality switches each arm's resolved config is expected to show.
SWITCHES = ("source_quality_mask", "event_weight_ua_valid", "target_quality_mask",
            "source_event_mask")
OUTPUT_JSON = "quality_arms_comparison.json"
OUTPUT_MD = "quality_arms_comparison.md"


def _results_dir(run_dir: str) -> Path:
    """Return the ``eval_results`` folder of an evaluation output directory, or the folder itself."""
    path = Path(run_dir)
    return path / RESULTS_DIRNAME if (path / RESULTS_DIRNAME).is_dir() else path


def read_arm(run_dir: str) -> Dict[str, Any]:
    """Read one arm's per-anchor table and its quality switches.

    Args:
        run_dir: The evaluation output directory (``--output-dir`` of ``eval/run.py``) or its
            ``eval_results`` folder.

    Returns:
        ``{'table': DataFrame, 'switches': {name: value}}``.

    Raises:
        FileNotFoundError: If the per-anchor table is absent.
        ValueError: If the table lacks the join key or a readout.
    """
    results = _results_dir(run_dir)
    table = pd.read_parquet(results / PER_ANCHOR_FILENAME)
    missing = [name for name in (*KEY, *METRICS) if name not in table.columns]
    if missing:
        raise ValueError(f"{results / PER_ANCHOR_FILENAME} lacks the columns {missing}")
    switches: Dict[str, Any] = {}
    config_path = results / RESOLVED_CONFIG_FILENAME
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        vae = (config.get("model_config") or {}).get("VAE_model") or {}
        switches = {name: vae.get(name, "(absent)") for name in SWITCHES}
    return {"table": table, "switches": switches}


def join_arms(masked: pd.DataFrame, control: pd.DataFrame, event_window_s: float) -> pd.DataFrame:
    """Join the two per-anchor tables on the anchors both scored, and assign the strata.

    The labels and the contraction age come from the masked arm: its detector reads ``up_valid``,
    so one stratum means one set of anchors for both arms.

    Args:
        masked: The masked arm's per-anchor table.
        control: The control's per-anchor table.
        event_window_s: The most seconds after a contraction onset at which an anchor is ``event``.

    Returns:
        One row per joined anchor, the readouts suffixed ``_masked`` and ``_control``, with the
        columns ``quality`` and ``timing``.

    Raises:
        ValueError: If the masked table lacks the quality labels, or no anchor joins.
    """
    absent = [name for name in LABELS if name not in masked.columns]
    if absent:
        raise ValueError(
            f"the masked arm's per-anchor table lacks {absent}. Evaluate both checkpoints with "
            f"eval/configs/eval_overrides_prod_q4.yaml, which sets the loader's quality masks."
        )
    keep = [*KEY, *METRICS]
    extra = [name for name in (*LABELS, AGE_COLUMN, CLASS_COLUMN) if name in masked.columns]
    joined = masked[keep + extra].merge(
        control[keep], on=list(KEY), suffixes=("_masked", "_control"), validate="one_to_one"
    )
    if joined.empty:
        raise ValueError("no anchor is in both tables; were both arms evaluated on the same files?")
    clean = (joined["ua_key_open"] > 0.5) & (joined["target_bad_steps"] < 0.5)
    joined["quality"] = np.where(clean, "clean", np.where(joined["ua_key_open"] < 0.5, "bad_ua",
                                                          "other"))
    age = joined[AGE_COLUMN] if AGE_COLUMN in joined.columns else pd.Series(np.nan, joined.index)
    joined["timing"] = np.where(age.notna() & (age <= event_window_s), "event", "quiet")
    return joined


def _bootstrap(values: np.ndarray, resamples: int, rng: np.random.Generator) -> Dict[str, float]:
    """Mean and 95% percentile bootstrap interval of per-recording values."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)), "n": int(values.size)}


def summarise(joined: pd.DataFrame, resamples: int, seed: int) -> List[Dict[str, Any]]:
    """Recording-level means of every readout, per arm and stratum, and the paired difference.

    Args:
        joined: Output of :func:`join_arms`.
        resamples: Bootstrap resamples over recordings.
        seed: The bootstrap seed.

    Returns:
        One row per (quality, timing) stratum.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for quality in ("clean", "bad_ua", "all"):
        for timing in ("event", "quiet", "any"):
            part = joined
            if quality != "all":
                part = part[part["quality"] == quality]
            if timing != "any":
                part = part[part["timing"] == timing]
            row: Dict[str, Any] = {"quality": quality, "timing": timing,
                                   "n_anchors": int(len(part)),
                                   "n_recordings": int(part[RECORDING].nunique())}
            if len(part):
                means = part.groupby(RECORDING)[
                    [f"{m}_{arm}" for m in METRICS for arm in ("masked", "control")]].mean()
                for metric in METRICS:
                    masked, control = means[f"{metric}_masked"], means[f"{metric}_control"]
                    row[f"{metric}_masked"] = _bootstrap(masked.to_numpy(), resamples, rng)
                    row[f"{metric}_control"] = _bootstrap(control.to_numpy(), resamples, rng)
                    diff = (masked - control).to_numpy()
                    row[f"{metric}_diff"] = dict(
                        _bootstrap(diff, resamples, rng),
                        positive_share=float(np.mean(diff > 0)) if diff.size else float("nan"))
                row["kld_per_t_masked_max_abs"] = float(part["kld_per_t_masked"].abs().max())
            rows.append(row)
    return rows


def _by_class(joined: pd.DataFrame, resamples: int, seed: int) -> List[Dict[str, Any]]:
    """The paired ``mc_pred_gap`` and KL differences on clean anchors, per clinical class."""
    if CLASS_COLUMN not in joined.columns:
        return []
    rng = np.random.default_rng(seed + 1)
    rows = []
    clean = joined[joined["quality"] == "clean"]
    for name, part in clean.groupby(CLASS_COLUMN):
        means = part.groupby(RECORDING)[[f"{m}_{a}" for m in ("mc_pred_gap", "kld_per_t")
                                         for a in ("masked", "control")]].mean()
        row = {CLASS_COLUMN: str(name), "n_recordings": int(len(means))}
        for metric in ("mc_pred_gap", "kld_per_t"):
            row[f"{metric}_diff"] = _bootstrap(
                (means[f"{metric}_masked"] - means[f"{metric}_control"]).to_numpy(), resamples, rng)
        rows.append(row)
    return rows


def _fmt(stat: Mapping[str, Any]) -> str:
    """``mean [lo, hi]`` with three significant decimals."""
    return f"{stat['mean']:.3f} [{stat['ci_lo']:.3f}, {stat['ci_hi']:.3f}]"


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    """Write the strata as a Markdown table, the arms' switches above it."""
    lines = ["# Quality-mask arms: masked against control", "",
             f"Joined anchors: {report['n_joined_anchors']}; recordings: "
             f"{report['n_joined_recordings']}. Means over recordings, 95% bootstrap intervals.", ""]
    for arm in ("masked", "control"):
        lines.append(f"- {arm} switches: {report['switches'][arm]}")
    lines += ["", "| quality | timing | anchors | recordings | mc_pred_gap masked | "
              "mc_pred_gap control | diff (share > 0) | KL masked | KL control | KL diff |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["strata"]:
        if not row["n_anchors"]:
            lines.append(f"| {row['quality']} | {row['timing']} | 0 | 0 | | | | | | |")
            continue
        diff = row["mc_pred_gap_diff"]
        lines.append(
            f"| {row['quality']} | {row['timing']} | {row['n_anchors']} | {row['n_recordings']} | "
            f"{_fmt(row['mc_pred_gap_masked'])} | {_fmt(row['mc_pred_gap_control'])} | "
            f"{_fmt(diff)} ({diff['positive_share']:.2f}) | {_fmt(row['kld_per_t_masked'])} | "
            f"{_fmt(row['kld_per_t_control'])} | {_fmt(row['kld_per_t_diff'])} |")
    if report["by_class"]:
        lines += ["", "## Clean anchors by clinical class", "",
                  "| class | recordings | mc_pred_gap diff | KL diff |", "|---|---|---|---|"]
        for row in report["by_class"]:
            lines.append(f"| {row[CLASS_COLUMN]} | {row['n_recordings']} | "
                         f"{_fmt(row['mc_pred_gap_diff'])} | {_fmt(row['kld_per_t_diff'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(masked_dir: Optional[str] = None, control_dir: Optional[str] = None,
         output_dir: Optional[str] = None, event_window_s: Optional[float] = None,
         resamples: Optional[int] = None, seed: Optional[int] = None) -> int:
    """Join the arms, summarise the strata, and write the JSON and Markdown reports.

    Args:
        masked_dir: The masked arm's evaluation output directory.
        control_dir: The control's evaluation output directory.
        output_dir: Where the two reports go; created if absent.
        event_window_s: Seconds after an onset that count as ``event``. Default: 220, the
            training contraction weight's reach (55 steps of 4 s).
        resamples: Bootstrap resamples over recordings. Default: 2000.
        seed: Bootstrap seed. Default: 42.

    Returns:
        The process exit code.
    """
    window = 220.0 if event_window_s is None else float(event_window_s)
    resamples = 2000 if resamples is None else int(resamples)
    seed = 42 if seed is None else int(seed)
    masked, control = read_arm(masked_dir), read_arm(control_dir)
    joined = join_arms(masked["table"], control["table"], window)
    report = {
        "event_window_s": window, "bootstrap_resamples": resamples, "seed": seed,
        "n_masked_anchors": int(len(masked["table"])),
        "n_control_anchors": int(len(control["table"])),
        "n_joined_anchors": int(len(joined)),
        "n_joined_recordings": int(joined[RECORDING].nunique()),
        "quality_shares": {k: float(v) for k, v in
                           joined["quality"].value_counts(normalize=True).items()},
        "switches": {"masked": masked["switches"], "control": control["switches"]},
        "strata": summarise(joined, resamples, seed),
        "by_class": _by_class(joined, resamples, seed),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / OUTPUT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(out / OUTPUT_MD, report)
    print(f"wrote {out / OUTPUT_JSON} and {out / OUTPUT_MD}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; every ``dest`` is also a :data:`RUN_ARGS` key."""
    p = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_transformer_cfs.eval.compare_quality_arms",
        description="Compare the masked and control arms on one set of anchors.")
    p.add_argument("--masked-dir", dest="masked_dir")
    p.add_argument("--control-dir", dest="control_dir")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--event-window-s", dest="event_window_s", type=float)
    p.add_argument("--resamples", dest="resamples", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    return p


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run."""
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("masked_dir", "control_dir", "output_dir"))
    if refusal:
        raise SystemExit(refusal)
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # ================================ FILL IN: the paths ================================
    # str -- the masked arm's eval output directory (the --output-dir given to eval/run.py).
    "masked_dir": r"/data/deid/isilon/MS_model/new_vae_teb_cross_attention/lag_attn_transformer_cfs/q4_eval_masked",
    # str -- the control arm's eval output directory.
    "control_dir": r"/data/deid/isilon/MS_model/new_vae_teb_cross_attention/lag_attn_transformer_cfs/q4_eval_no_quality_mask",
    # str -- where quality_arms_comparison.json and .md go. Aggregates only: safe to send.
    "output_dir": r"/data/deid/isilon/MS_model/new_vae_teb_cross_attention/lag_attn_transformer_cfs/q4_quality_arms_comparison",
    # ========================= Everything below has a default =========================
    # float | None -- seconds after a contraction onset that count as `event`.  (None -> 220)
    "event_window_s": None,
    # int | None -- bootstrap resamples over recordings.  (None -> 2000)
    "resamples": None,
    # int | None -- bootstrap seed.  (None -> 42)
    "seed": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
