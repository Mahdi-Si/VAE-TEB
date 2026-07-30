r"""How far apart the two forecasts are, and how far the source moved the belief behind them.

The sibling pipeline's ``residual`` analysis has no direct analogue here, and the reason is
structural rather than incidental: that model has a ``delta_mu_src`` and a base-plus-residual
decoder, so its residual *is* a tensor. This one has neither. ``mu_base`` and ``mu_full`` are two
passes of **one shared decoder** on two latents, and the only residual that exists is the
difference between what those two passes produced. So what is reported here is that difference,
plus the latent-side movement that caused it:

* **The forecast difference**, $\lVert \mu_{\mathrm{full}} - \mu_{\mathrm{base}} \rVert$ per
  scored raw sample, in bpm. Distinct from ``pred_gap``, which is a difference of *scores*: two
  forecasts can differ everywhere and score identically, and a source that moves the forecast
  without improving it is a different finding from one that does neither.
* **The latent drift**, in the two forms that are not the same quantity. ``delta_mu_rms`` is the
  per-**element** RMS of $\mu^q - \mu^p$; ``mu_post_prior_gap_rms`` sums over $d_z$ **first**, so
  it is the size of the belief shift per step rather than a per-coordinate figure. At equal
  support they differ by exactly $\sqrt{d_z}$, and an implementation that conflated them would be
  wrong by that factor with nothing looking odd.

**Every RMS accumulates unrooted and roots once.** An RMS is the square root of a mean, and by
Jensen $\operatorname{mean}(\sqrt{x}) \le \sqrt{\operatorname{mean}(x)}$ -- so averaging finished
per-segment roots is biased **low**, in the direction that flatters the model. The collection pass
therefore carries the squares, and the root is taken here, once, at the end of the chain. The
biased form is computed beside it and reported as such, because a caveat nobody can check is a
caveat nobody believes.

**One caveat weakens this in the model's favour, and it is stated rather than assumed away.** Both
branches share a single log-variance head applied to two different $z$, rather than reading two
separate variance heads -- so the two forecasts' *uncertainties* are not independent estimates,
and a difference between them understates how differently the two latents are being read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    scored_sample_count,
)
from teb_vae.lag_attn_rws.eval.metrics import NORMALISED_UNIT, sigma_to_bpm

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "residual"

#: What it writes.
PER_RECORDING_FILENAME = "residual_per_recording.csv"
SUMMARY_FILENAME = "residual_summary.csv"

#: The caveat that travels in the output rather than only in this docstring.
SHARED_VARIANCE_CAVEAT = (
    "both forecasts come from one shared decoder and one shared log-variance head applied to two "
    "different latents, not from two separate variance heads -- so the difference between them "
    "understates how differently the two latents are read, in the model's favour"
)

#: The unrooted per-sample columns, the name each roots into, whether the root converts to bpm,
#: and what it means. Every one is a mean of squares on the per-sample table; the root happens
#: here, after the per-recording chain, exactly once.
RMS_METRICS: Tuple[Tuple[str, str, bool, str], ...] = (
    (
        "forecast_difference_sq",
        "forecast_difference_rms",
        True,
        "RMS of mu_full - mu_base per scored raw sample: how far the source moved the forecast, "
        "which is not the same as how much it improved it",
    ),
    (
        "delta_mu_sq",
        "delta_mu_rms",
        False,
        "per-element RMS of mu_post - mu_prior over the KL's anchor support",
    ),
    (
        "mu_post_prior_gap_sq",
        "mu_post_prior_gap_rms",
        False,
        "L2 norm of mu_post - mu_prior per step, summed over d_z before the mean -- the size of "
        "the belief shift, larger than the per-element figure by sqrt(d_z)",
    ),
)

#: The rooted column the collection pass also carries, kept so the Jensen bias is a measured
#: number rather than a claim: it is the mean of per-segment roots, which is what this analysis
#: does *not* report as the RMS.
_ROOTED_PER_SEGMENT = {"delta_mu_sq": "delta_mu_rms"}

#: The metrics resolved by cohort. The **unrooted** squares, which is what the frame carries: a
#: grouped variant of the mean square and one of the RMS answer the same question, and rooting
#: after a per-cohort mean would be a third reduction with its own Jensen bias.
GROUPED_METRICS: Tuple[str, ...] = tuple(square for square, _, _, _ in RMS_METRICS)


def build_rows(
    per_guid: pd.DataFrame,
    normalization: Optional[Dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    r"""Root each accumulated mean square once, after the per-recording chain.

    The interval's bounds are rooted rather than the interval being rebuilt: a monotone transform
    of a percentile interval **is** the percentile interval of the transform, so $\sqrt{\cdot}$ of
    the bounds on the mean square is the interval on the RMS.

    Args:
        per_guid: Per-recording means of the unrooted squares.
        normalization: The loader's FHR statistics, for the bpm column.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per metric, carrying the correctly-rooted value with its interval, the biased
        average-of-roots beside it where the pass carried one, and the unit the bpm column really
        is.
    """
    rows: List[Dict[str, Any]] = []
    for square_column, name, to_bpm, meaning in RMS_METRICS:
        squares = finite_column(per_guid, square_column)
        interval = shared_stats.bootstrap_ci(squares, resamples=resamples, seed=seed)
        rooted = (
            float(np.sqrt(interval["point"]))
            if np.isfinite(interval["point"]) and interval["point"] >= 0.0
            else float("nan")
        )
        converted, unit = (
            sigma_to_bpm([rooted], normalization) if to_bpm else ([rooted], NORMALISED_UNIT)
        )
        row: Dict[str, Any] = {
            "metric": name,
            "meaning": meaning,
            "source_column": square_column,
            "n": int(interval["n"]),
            "mean_square": interval["point"],
            "rms_normalised": rooted,
            "rms_lo_normalised": float(np.sqrt(max(interval["lo"], 0.0)))
            if np.isfinite(interval["lo"]) else float("nan"),
            "rms_hi_normalised": float(np.sqrt(max(interval["hi"], 0.0)))
            if np.isfinite(interval["hi"]) else float("nan"),
            "rms": float(converted[0]),
            "unit": unit,
        }
        biased_column = _ROOTED_PER_SEGMENT.get(square_column)
        if biased_column is not None:
            biased = finite_column(per_guid, biased_column)
            row["mean_of_per_segment_rms"] = (
                float(np.nanmean(biased)) if np.isfinite(biased).any() else float("nan")
            )
            # Reported, not corrected: this is the number a naive implementation would call the
            # RMS, and by Jensen it can only sit at or below the rooted-once one.
            row["jensen_bias"] = row["mean_of_per_segment_rms"] - rooted
        rows.append(row)
    return rows


def run_residual_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the forecast difference and the latent drift, rooted once, per recording.

    Args:
        context: The analysis context, read for the per-sample table and the loader statistics
            the bpm column needs.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the metric table, the unit, and the shared-variance caveat.
    """
    collection = context.collection
    per_sample = collection.per_sample
    record = dict(getattr(collection, "record", None) or {})
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    columns = [square for square, _, _, _ in RMS_METRICS] + list(_ROOTED_PER_SEGMENT.values())
    per_guid = per_recording_means(per_sample, columns)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    rows = build_rows(
        per_guid, dict(record.get("normalization") or {}), resamples=resamples, seed=seed
    )
    pd.DataFrame(rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    distribution = {
        name: describe(finite_column(per_guid, square), name=name)
        for square, name, _, _ in RMS_METRICS
    }
    return {
        "n_samples": scored_sample_count(per_sample, "forecast_difference_sq"),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "metrics": rows,
        "mean_square_distribution": list(distribution.values()),
        "unit": next((row["unit"] for row in rows if row["unit"] != NORMALISED_UNIT),
                     NORMALISED_UNIT),
        "caveat": SHARED_VARIANCE_CAVEAT,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [PER_RECORDING_FILENAME, SUMMARY_FILENAME],
    }
