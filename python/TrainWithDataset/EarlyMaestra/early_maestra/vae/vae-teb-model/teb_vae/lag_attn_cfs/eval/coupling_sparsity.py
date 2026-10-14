r"""Is the coupling sparse in time? Bin a finished evaluation's per-anchor readouts by event age.

The source-conditioned branch exists to measure what the uterine activity tells the model about
the FHR future that the FHR's own past does not. If that information arrives only around
contractions, then two per-anchor quantities of a finished evaluation should concentrate at small
``seconds_since_contraction``: the KL $K_t$, which is what the model *spent*, and the forecast
gaps $D_0 - D_1$, which is what the source *bought*. The two can disagree, and the disagreement is
the diagnosis:

* KL flat across event age while the gap concentrates near contractions: the model pays for the
  source at every anchor and is repaid at a few. That is the leak the posterior event gate and
  the contraction weighting of the full branch exist to close.
* Both concentrating: the model has learned the sparsity on its own, and only $\beta$'s level is
  in question.
* Neither concentrating: the premise is wrong for this representation, or the detector is.

Everything here reads the tables the collection pass already wrote (``per_anchor.parquet`` and
``per_sample.csv``), so the question costs no training run and no model. The event age column is
the one the collection pass derived from the raw UP with the evaluation's own detector, so a bin
here and a training-time contraction weight name the same event.

A second table bins the **base** branch's NLL by hours since labour onset, demeaned per recording,
to ask whether the forecast still depends on where the segment sits in labour once the model has
seen its own twenty minutes of history. A flat profile says the window already carries the stage;
a trend says a labour clock has information the window lacks.

Run from an IDE's Run button by editing :data:`RUN_ARGS`, or from the command line::

    python -m teb_vae.lag_attn_cfs.eval.coupling_sparsity --eval-dir <run>/eval_results
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Upper edges of the event-age bins in seconds; the last open bin and the "no contraction behind
#: this anchor" bin are added by :func:`event_age_bins`. $30$ s is the shortest published
#: UA-to-FHR delay, $120$ s the longest contraction, $300$ s a typical inter-contraction interval.
EVENT_AGE_EDGES_S: Sequence[float] = (30.0, 60.0, 120.0, 180.0, 300.0)

#: Upper edges of the hours-since-onset bins; negative is before onset, NaN is unknown.
ONSET_HOUR_EDGES: Sequence[float] = (0.0, 2.0, 4.0, 6.0, 9.0, 12.0)

#: The per-anchor columns binned, in the order the table reports them. Columns absent from an
#: older evaluation are skipped rather than failing the run.
ANCHOR_COLUMNS: Sequence[str] = (
    "kld_per_t",
    "pred_gap",
    "mean_pred_gap",
    "mc_pred_gap",
    "nll_base_block",
    "nll_full_block",
)

EVENT_AGE_COLUMN = "seconds_since_contraction"
ONSET_COLUMN = "time_from_labor_onset"
SECONDS_PER_STEP = 4.0

#: The IDE entry: edit these and press Run. ``eval_dir`` is a run's ``eval_results`` directory.
#: ``trim_minutes`` ``None`` reads the loader's trim from the run's ``resolved_config.yaml`` beside
#: that directory and falls back to the production $1.0$ when the file is absent.
RUN_ARGS: Dict[str, Any] = dict(
    eval_dir="REPOINT_ME/eval_results",
    output_dir=None,
    bootstrap=200,
    seed=0,
    trim_minutes=None,
)

#: The production trim, used only when no resolved config is found beside the evaluation.
DEFAULT_TRIM_MINUTES = 1.0


def resolve_trim_minutes(eval_dir: Path, trim_minutes: Optional[float]) -> float:
    r"""The loader's trim for this run: the argument, else the run's resolved config, else $1.0$.

    Args:
        eval_dir: The ``eval_results`` directory; the resolved config is looked for one level up.
        trim_minutes: An explicit value, or ``None`` to read it.

    Returns:
        The trim in minutes.
    """
    if trim_minutes is not None:
        return float(trim_minutes)
    resolved = eval_dir.parent / "resolved_config.yaml"
    if resolved.is_file():
        import yaml

        config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        value = (
            ((config.get("dataset_config") or {}).get("dataloader_config") or {})
            .get("dataset_kwargs", {})
            .get("trim_minutes")
        )
        if value is not None:
            return float(value)
    print(f"no trim_minutes in a resolved config beside {eval_dir}; using {DEFAULT_TRIM_MINUTES}")
    return DEFAULT_TRIM_MINUTES


def _event_age_labels() -> List[str]:
    """The event-age bin labels in order, ``[lo, hi)`` in seconds then ``>= last``."""
    edges = [0.0, *EVENT_AGE_EDGES_S]
    labels = [f"[{int(lo)}, {int(hi)})" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">= {int(edges[-1])}")
    return labels


def event_age_bins(ages: pd.Series) -> pd.Series:
    r"""Label every anchor with its event-age bin.

    Args:
        ages: ``seconds_since_contraction`` per anchor, NaN where no contraction precedes the
            anchor in its segment.

    Returns:
        A string label per anchor: ``[lo, hi)`` in seconds, ``>= last`` or ``none``.
    """
    binned = pd.cut(
        ages, bins=[0.0, *EVENT_AGE_EDGES_S, np.inf], right=False, labels=_event_age_labels()
    )
    return binned.astype(object).where(ages.notna(), "none")


def onset_hour_bins(hours: pd.Series) -> pd.Series:
    r"""Label every anchor with its hours-since-onset bin, ``pre-onset`` below $0$, ``unknown`` on NaN.

    Args:
        hours: Hours since labour onset at the anchor.

    Returns:
        A string label per anchor.
    """
    edges = list(ONSET_HOUR_EDGES)
    labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">= {edges[-1]:g}")
    binned = pd.cut(hours, bins=[*edges, np.inf], right=False, labels=labels).astype(object)
    binned = binned.where(~(hours < 0.0), "pre-onset")
    return binned.where(hours.notna(), "unknown")


def recording_bootstrap_ci(
    frame: pd.DataFrame, column: str, *, n_boot: int, seed: int
) -> Dict[str, float]:
    r"""Percentile interval of a column's mean under resampling of **recordings**, not anchors.

    Anchors of one recording are near-duplicates, so an anchor-level interval would be far too
    narrow. Recordings are drawn with replacement; every anchor of a drawn recording travels with
    it, and the statistic is the anchor-weighted mean of the resample.

    Args:
        frame: Rows of one bin, carrying ``guid`` and ``column``.
        column: The column whose mean is bootstrapped.
        n_boot: Number of resamples; $0$ reports NaN bounds.
        seed: Generator seed.

    Returns:
        ``{'lo', 'hi'}`` at the $2.5$ and $97.5$ percentiles.
    """
    # NaN cells (a gap unpaired on this row, say) leave both the sum and the count, so a bin whose
    # column is all NaN reports NaN bounds rather than a mean of zero.
    by_guid = frame.groupby("guid", sort=False)[column].agg(["sum", "count"])
    sums, counts = by_guid["sum"].to_numpy(float), by_guid["count"].to_numpy(float)
    if n_boot <= 0 or len(sums) < 2 or counts.sum() == 0:
        return {"lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(int(n_boot), len(sums)))
    drawn_counts = counts[draws].sum(axis=1)
    means = np.where(drawn_counts > 0, sums[draws].sum(axis=1) / np.maximum(drawn_counts, 1.0), np.nan)
    if np.isnan(means).all():
        return {"lo": float("nan"), "hi": float("nan")}
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def bin_by_event_age(
    per_anchor: pd.DataFrame, *, n_boot: int, seed: int
) -> pd.DataFrame:
    r"""One row per event-age bin: anchor and recording counts, means with intervals, KL mass share.

    Args:
        per_anchor: The evaluation's per-anchor table.
        n_boot: Bootstrap resamples for the intervals.
        seed: Generator seed.

    Returns:
        The table, ordered by bin.
    """
    if EVENT_AGE_COLUMN not in per_anchor.columns:
        raise ValueError(
            f"per_anchor.parquet carries no '{EVENT_AGE_COLUMN}'; the collection pass needs the raw "
            f"'up' trace in load_fields to derive it"
        )
    frame = per_anchor.assign(_bin=event_age_bins(per_anchor[EVENT_AGE_COLUMN]))
    columns = [name for name in ANCHOR_COLUMNS if name in frame.columns]
    total_kl = float(frame["kld_per_t"].sum()) if "kld_per_t" in columns else float("nan")
    rows: List[Dict[str, Any]] = []
    for label in [*_event_age_labels(), "none"]:
        rows_in = frame[frame["_bin"] == label]
        row: Dict[str, Any] = {
            "event_age_bin": label,
            "n_anchors": int(len(rows_in)),
            "anchor_share": float(len(rows_in) / max(len(frame), 1)),
            "n_recordings": int(rows_in["guid"].nunique()) if len(rows_in) else 0,
        }
        if "kld_per_t" in columns:
            row["kl_mass_share"] = (
                float(rows_in["kld_per_t"].sum() / total_kl) if total_kl > 0 else float("nan")
            )
        for index, name in enumerate(columns):
            if len(rows_in) == 0:
                row[f"{name}_mean"] = float("nan")
                row[f"{name}_lo"] = row[f"{name}_hi"] = float("nan")
                continue
            interval = recording_bootstrap_ci(rows_in, name, n_boot=n_boot, seed=seed + index)
            row[f"{name}_mean"] = float(rows_in[name].mean())
            row[f"{name}_lo"], row[f"{name}_hi"] = interval["lo"], interval["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def bin_base_nll_by_onset_hours(
    per_anchor: pd.DataFrame, per_sample: pd.DataFrame, *, trim_minutes: float
) -> pd.DataFrame:
    r"""One row per hours-since-onset bin: the recording-demeaned base NLL and its support.

    The anchor's clock is the segment's ``time_from_labor_onset`` (its untrimmed start) plus the
    trim lead plus $4 t$ seconds. The base NLL is demeaned **within each recording** before
    binning, so a recording that is hard to forecast everywhere does not read as a late-labour
    effect through where its segments happen to fall.

    Args:
        per_anchor: The evaluation's per-anchor table.
        per_sample: The evaluation's per-sample table, for ``time_from_labor_onset``.
        trim_minutes: The loader's trim, which offsets the segment start to the first step.

    Returns:
        The table, ordered by bin; empty if the per-anchor table has no ``nll_base_block``.
    """
    if "nll_base_block" not in per_anchor.columns or ONSET_COLUMN not in per_sample.columns:
        return pd.DataFrame()
    onset = per_sample.set_index("sample_index")[ONSET_COLUMN]
    frame = per_anchor.join(onset, on="sample_index")
    lead = 60.0 * float(trim_minutes)
    hours = (frame[ONSET_COLUMN] + lead + SECONDS_PER_STEP * frame["anchor"]) / 3600.0
    frame = frame.assign(
        _hours=hours,
        _bin=onset_hour_bins(hours),
        _demeaned=frame["nll_base_block"]
        - frame.groupby("guid")["nll_base_block"].transform("mean"),
    )
    labels = [
        *[f"[{lo:g}, {hi:g})" for lo, hi in zip(ONSET_HOUR_EDGES[:-1], ONSET_HOUR_EDGES[1:])],
        f">= {ONSET_HOUR_EDGES[-1]:g}",
    ]
    rows: List[Dict[str, Any]] = []
    for label in ["pre-onset", *labels, "unknown"]:
        rows_in = frame[frame["_bin"] == label]
        rows.append(
            {
                "onset_hours_bin": label,
                "n_anchors": int(len(rows_in)),
                "n_recordings": int(rows_in["guid"].nunique()) if len(rows_in) else 0,
                "nll_base_demeaned_mean": float(rows_in["_demeaned"].mean()) if len(rows_in) else float("nan"),
                "nll_base_mean": float(rows_in["nll_base_block"].mean()) if len(rows_in) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def run(
    eval_dir: str,
    *,
    output_dir: Optional[str] = None,
    bootstrap: int = 200,
    seed: int = 0,
    trim_minutes: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    r"""Read the two tables, write the two binned tables, print both.

    Args:
        eval_dir: A run's ``eval_results`` directory.
        output_dir: Where the CSVs go; defaults to ``<eval_dir>/coupling_sparsity``.
        bootstrap: Recording-level bootstrap resamples for the intervals.
        seed: Generator seed.
        trim_minutes: The loader's trim, for the onset clock; ``None`` resolves it from the run's
            ``resolved_config.yaml`` beside the evaluation, with $1.0$ as the fallback.

    Returns:
        ``{'event_age': ..., 'onset_hours': ...}``.

    Raises:
        FileNotFoundError: If either table is missing from ``eval_dir``.
        ValueError: If the per-anchor table carries no ``seconds_since_contraction`` column.
    """
    eval_path = Path(eval_dir)
    trim = resolve_trim_minutes(eval_path, trim_minutes)
    per_anchor = pd.read_parquet(eval_path / "per_anchor.parquet")
    per_sample = pd.read_csv(eval_path / "per_sample.csv")
    out = Path(output_dir) if output_dir else eval_path / "coupling_sparsity"
    out.mkdir(parents=True, exist_ok=True)

    event_age = bin_by_event_age(per_anchor, n_boot=int(bootstrap), seed=int(seed))
    event_age.to_csv(out / "kl_and_gap_by_event_age.csv", index=False)
    onset_hours = bin_base_nll_by_onset_hours(per_anchor, per_sample, trim_minutes=trim)
    onset_hours.to_csv(out / "base_nll_by_onset_hours.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("KL and forecast gap by seconds since the last contraction")
        print(event_age.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()
        print("Recording-demeaned base NLL by hours since labour onset")
        print(onset_hours.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwritten to {out}")
    return {"event_age": event_age, "onset_hours": onset_hours}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry; with no arguments, :data:`RUN_ARGS` is the configuration.

    Args:
        argv: Command-line arguments without the program name; ``None`` reads ``sys.argv``.

    Raises:
        SystemExit: If the evaluation directory does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", default=RUN_ARGS["eval_dir"])
    parser.add_argument("--output-dir", default=RUN_ARGS["output_dir"])
    parser.add_argument("--bootstrap", type=int, default=RUN_ARGS["bootstrap"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    parser.add_argument(
        "--trim-minutes", type=float, default=RUN_ARGS["trim_minutes"],
        help="loader trim in minutes; omitted, it is read from the run's resolved_config.yaml",
    )
    args = parser.parse_args(argv)
    if not os.path.isdir(args.eval_dir):
        raise SystemExit(f"eval dir not found: {args.eval_dir} (edit RUN_ARGS or pass --eval-dir)")
    run(
        args.eval_dir,
        output_dir=args.output_dir,
        bootstrap=args.bootstrap,
        seed=args.seed,
        trim_minutes=args.trim_minutes,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
