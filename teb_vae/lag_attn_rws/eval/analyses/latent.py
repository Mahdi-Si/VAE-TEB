r"""How much of the latent carries source information, and whether its variance is fitted or bound.

Two questions, and the second is the one nothing else in the pipeline asks.

**Is the KL spread over the latent, or collapsed onto a corner of it?** The per-dimension spectrum
$\bar K_d$ answers it directly: how many dimensions clear the activity threshold, and what share of
the total the single largest holds. Both travel the same aggregation chain as every scalar --
support-weighted within a segment, unweighted over a recording's segments, unweighted across
recordings -- which is what makes $\sum_d \bar K_d$ equal the headline $\bar K$ it decomposes.

**Is the prior variance pinned on its clamp?** This is the failure that hides. The KL carries

$$\mathrm{KL} \supset \frac{(\mu^q - \mu^p)^2}{\sigma_p^2},$$

so a prior variance sitting on its lower clamp divides by a bound rather than by a fitted quantity
and inflates every coupling number by an arbitrary factor -- while ``mean_logvar_full`` and
``mean_logvar_base``, which are *decoder* variances and are what a reader looks at, stay perfectly
healthy. The model already computes the detectors and the trainer logs them every epoch; what was
missing was an evaluation that reads them and a criterion that can fail on them. Both are here:
``prior_variance_not_pinned`` is a failable verdict, and the margin it uses is the model's own
:data:`~teb_vae.lag_attn_rws.nets.model.LOGVAR_FLOOR_MARGIN_FRAC` rather than a second copy of
$0.05$.

The bound is a **sigmoid**, so an exact-equality test against the asymptote reads $0.0$ forever
while the variance sits pinned against it. Everything here is measured against the margin instead:
$5\%$ of the clamp's range, which on the shipped $[-5, 3]$ is $0.4$ nats.

**The two saturation fractions need a masked recomputation and the log-variance fractions do
not.** That is the opposite of the sibling pipeline's framing and is worth stating: in this model
``mean_logvar_*`` and both clamp fractions are already computed over their masked supports, while
the model's own ``mu_prior_sat_frac`` and ``delta_mu_sat_frac`` are flat means over every element
-- warm-up prefix and untrained tail included. Both framings are emitted, per recording, and they
may legitimately disagree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    scored_sample_count,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "latent"

#: What it writes.
SPECTRUM_FILENAME = "latent_spectrum.csv"
DIAGNOSTICS_FILENAME = "latent_diagnostics.csv"
PER_RECORDING_FILENAME = "latent_per_recording.csv"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
SPECTRUM_FIGURE = "kl_spectrum.pdf"

#: The per-sample diagnostics reduced per recording, each with what it is for. Ordered so the
#: table reads prior first, then posterior, then the two bounds' saturation.
DIAGNOSTIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("mean_logvar_prior", "mean prior log-variance over the KL's anchor support"),
    (
        "logvar_prior_floor_frac",
        "fraction of the prior log-variance within the margin of its lower clamp -- the "
        "detector for an inflated KL denominator",
    ),
    ("mean_logvar_post", "mean posterior log-variance over the same support"),
    ("mean_logvar_full", "mean decoder log-variance over the scored raw samples"),
    (
        "logvar_full_floor_frac",
        "fraction of the decoder log-variance on its floor: over-confident, and where a loss "
        "spike comes from",
    ),
    (
        "logvar_full_ceil_frac",
        "fraction on its ceiling: the decoder has given up and is predicting noise, which reads "
        "as a healthy falling NLL while pred_gap goes to zero",
    ),
    ("mu_prior_rms", "RMS of the prior mean over its masked support"),
    (
        "mu_prior_sat_frac_raw",
        "prior mean against its tanh bound, over every element -- the model's own framing, which "
        "includes the warm-up prefix and the untrained tail",
    ),
    (
        "mu_prior_sat_frac_masked",
        "the same fraction over the KL's anchor support only; it may legitimately disagree with "
        "the raw one",
    ),
    ("delta_mu_sat_frac_raw", "posterior residual against its bound, over every element"),
    ("delta_mu_sat_frac_masked", "the same over the KL's anchor support only"),
)

#: The KL columns reduced beside them, so the spectrum's total has its scalar next to it.
_KL_COLUMNS: Tuple[str, ...] = ("source_conditioned_kl_raw",)

#: The diagnostics worth resolving by cohort: the KL itself and the two clamp fractions that
#: decide whether it is a rate at all. Not all thirteen -- a grouped figure with thirteen panels is
#: one nobody reads, and the rest are read pooled because they describe the checkpoint rather than
#: the cohort.
GROUPED_METRICS: Tuple[str, ...] = (
    "source_conditioned_kl_raw",
    "logvar_prior_floor_frac",
    "mean_logvar_prior",
)


def spectrum_frame(
    kld_per_dimension: Sequence[float], *, threshold: float = KLD_ACTIVE_EPS
) -> pd.DataFrame:
    """Lay the per-dimension KL out as a table, sorted by how much each dimension carries.

    Args:
        kld_per_dimension: The chained per-dimension KL, in latent-dimension order.
        threshold: The activity threshold a dimension must clear to count as carrying anything.

    Returns:
        One row per latent dimension -- its index, its KL, its share of the total, whether it is
        active, and its rank. Sorted descending, because the question the spectrum answers is
        "how many dimensions carry this" and that is read off the head of a sorted list.
    """
    values = np.asarray(list(kld_per_dimension), dtype=np.float64)
    if values.size == 0:
        return pd.DataFrame(columns=["dimension", "kl_nats", "share", "active", "rank"])
    total = float(values.sum())
    order = np.argsort(-values)
    return pd.DataFrame(
        {
            "dimension": order.astype(int),
            "kl_nats": values[order],
            "share": values[order] / total if total > 0.0 else np.full(values.size, np.nan),
            "active": values[order] > float(threshold),
            "rank": np.arange(values.size, dtype=int),
        }
    )


def build_diagnostic_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise each latent diagnostic over the recordings, with its interval.

    Args:
        per_guid: Per-recording means of the per-sample table.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per diagnostic, each carrying what it is for -- a fraction whose meaning lives in
        a docstring is a fraction that gets read as its own opposite.
    """
    rows: List[Dict[str, Any]] = []
    for column, meaning in DIAGNOSTIC_COLUMNS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        rows.append(
            {
                "metric": column,
                "meaning": meaning,
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
            }
        )
    return rows


def build_spectrum_figure(
    spectrum: pd.DataFrame, *, threshold: float = KLD_ACTIVE_EPS
) -> Any:
    """Draw the per-dimension KL, sorted, with the activity threshold marked.

    Sorted descending rather than in dimension order: a latent index means nothing, and the shape
    of the sorted spectrum is the entire finding -- a few tall bars is a collapsed latent whatever
    the indices happen to be. The threshold is drawn because "active dimensions" is a count
    against a line, and a bar chart without that line invites a reader to pick their own.

    Args:
        spectrum: The spectrum table, as :func:`spectrum_frame` returns it.
        threshold: The activity threshold to mark.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(1)
    axis = axes[0, 0]
    values = finite_column(spectrum, "kl_nats")
    if values.size and np.isfinite(values).any():
        positions = np.arange(values.size, dtype=np.float64)
        axis.bar(positions, values, color=figures.COLOR_BLUE, alpha=0.85, width=0.8)
        axis.axhline(
            float(threshold), color=figures.COLOR_VERMILLION, linestyle="--", linewidth=figures.LINE_REGULAR,
            label=f"activity threshold {float(threshold):g} nats",
        )
        axis.legend(fontsize=figures.FONT_LABEL, loc="best")
    else:
        axis.text(
            0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes
        )
    axis.set_title("Per-dimension KL, sorted")
    axis.set_xlabel("latent dimension, ordered by the KL it carries")
    axis.set_ylabel("nats per anchor")
    figures.style_axes(axis)
    return figure


def run_latent_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the KL spectrum and the bound-variance diagnostics, per recording.

    Args:
        context: The analysis context, read for the per-sample table and the pass's own latent
            health block.
        eval_config: The validated block, for the bootstrap settings and the collapse threshold.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the spectrum, the diagnostics, and the run's verdict on whether
        the prior variance is pinned -- surfaced here beside the number that decided it.
    """
    collection = context.collection
    per_sample = collection.per_sample
    results = dict(getattr(collection, "results", None) or {})
    record = dict(getattr(collection, "record", None) or {})
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    health = dict(results.get("latent_health") or {})
    spectrum = spectrum_frame(health.get("kld_per_dimension") or [])
    spectrum.to_csv(directory / SPECTRUM_FILENAME, index=False)

    columns = [name for name, _ in DIAGNOSTIC_COLUMNS] + list(_KL_COLUMNS)
    per_guid = per_recording_means(per_sample, columns)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    diagnostic_rows = build_diagnostic_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(diagnostic_rows).to_csv(directory / DIAGNOSTICS_FILENAME, index=False)

    figure_name = str(
        figures.render_to_pdf(
            build_spectrum_figure(spectrum), directory / SPECTRUM_FIGURE
        ).name
    )
    verdicts = {
        str(verdict.get("name")): verdict
        for verdict in (results.get("verdicts") or [])
        if isinstance(verdict, dict)
    }
    return {
        "n_samples": scored_sample_count(per_sample, "source_conditioned_kl_raw"),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "health": health,
        "activity_threshold_nats": float(KLD_ACTIVE_EPS),
        "diagnostics": diagnostic_rows,
        # The clamp and the margin the fractions above were measured against, read from the model
        # that produced them rather than from a config file that may since have changed.
        "bounds": dict(record.get("bounds") or {}),
        "prior_variance_verdict": verdicts.get("prior_variance_not_pinned"),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            SPECTRUM_FILENAME, DIAGNOSTICS_FILENAME, PER_RECORDING_FILENAME, figure_name
        ],
    }
