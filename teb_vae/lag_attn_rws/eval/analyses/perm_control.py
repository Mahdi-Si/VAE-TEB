r"""Does the model use *this* recording's source, or react to any source at all?

A nonzero source-conditioned KL proves nothing on its own. The posterior sees the source, so it
reacts to whatever it is given -- and a stranger's source is out of distribution for a posterior
trained only on matched pairs, which routinely moves it **more**. So the discriminating comparison
lives in prediction space:

$$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}.$$

**The verdict takes three losses and nothing else.** That is the content of the criterion rather
than a simplification of it: a criterion that also read the KL would fail exactly the healthy
models it should pass, because $K_{\mathrm{shuffled}} > K_{\mathrm{true}}$ is what a healthy model
looks like. The KL-space reading is emitted here under an explicit label, and
``shuffled_exceeds_true`` is recorded as a **description** consumed by nothing.

**``influential_not_specific`` is a real finding, not a failure mode of this analysis.** A model
whose forecast improves under any source it is handed has learned that the source stream *exists*;
it has not learned to read this recording's. Naming that outcome mechanically is what stops it
being read as either success or breakage.

**The shuffled branch is scored on its own.** ``perm_forward_outputs`` draws a fresh $\epsilon$ for
the permuted latent, so a sample-by-sample difference against the matched branch would carry an
independent sampling term that the common-random-numbers pairing removes from every other
comparison in this pipeline. And only ``controls.RECOMPUTED_KEYS`` describe the permuted pairing:
the result is a shallow copy, so its ``kld_per_t`` is the *matched* value, and the shuffled KL this
analysis reads is recomputed from the permuted distribution parameters in the collection pass
rather than read off that key.
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
    positive_fraction,
    scored_sample_count,
)
from teb_vae.lag_attn_rws.eval.metrics import source_specificity_verdict

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "perm_control"

#: What it writes.
PER_RECORDING_FILENAME = "perm_control_per_recording.csv"
SUMMARY_FILENAME = "perm_control_summary.csv"

#: The four branch scores the ordering is read from, marginalised, in reporting order.
BRANCH_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("base", "mc_nll_base_block"),
    ("full", "mc_nll_full_block"),
    ("shuffled", "mc_nll_shuffled_block"),
    ("base_shuffled_mu", "mc_nll_base_shuffled_mu_block"),
)

#: The two KL columns, matched and control.
_KL_TRUE = "source_conditioned_kl_raw"
_KL_SHUFFLED = "source_conditioned_kl_shuffled_raw"

#: Every per-sample column this analysis reduces per recording.
VALUE_COLUMNS: Tuple[str, ...] = tuple(
    column for _, column in BRANCH_COLUMNS
) + (_KL_TRUE, _KL_SHUFFLED)

#: The metrics resolved by cohort: the three block scores the specificity ordering is read from.
#: The KL columns are left pooled -- the ordering is a statement about prediction space, and a
#: by-cohort KL is the latent analysis's subject rather than this one's.
GROUPED_METRICS: Tuple[str, ...] = tuple(
    column for name, column in BRANCH_COLUMNS if name in ("base", "full", "shuffled")
)

#: The mechanically defined outcomes. Each is a *description* of what the three losses did; only
#: ``specific`` is what the architecture claims.
OUTCOMES: Tuple[str, ...] = (
    "specific",
    "influential_not_specific",
    "no_improvement",
    "inconclusive",
)


def classify_outcome(
    d_base: Optional[float], d_full: Optional[float], d_shuffled: Optional[float]
) -> str:
    """Name what the three losses did, in one of :data:`OUTCOMES`.

    Args:
        d_base: The target-only branch's marginalised block score.
        d_full: The source-conditioned branch's.
        d_shuffled: The stranger's-source branch's.

    Returns:
        ``specific`` when the full ordering holds; ``influential_not_specific`` when the source
        helps but a stranger's helps too, which says the model reads *a* source rather than this
        one; ``no_improvement`` when the source added nothing; ``inconclusive`` when a branch is
        missing.
    """
    values = (d_base, d_full, d_shuffled)
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return "inconclusive"
    base, full, shuffled = (float(value) for value in values)  # type: ignore[arg-type]
    if full >= base:
        return "no_improvement"
    return "specific" if shuffled > base else "influential_not_specific"


def build_branch_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise each branch's per-recording block score with its interval.

    Args:
        per_guid: Per-recording means.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per branch, in reporting order.
    """
    rows: List[Dict[str, Any]] = []
    for branch, column in BRANCH_COLUMNS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        rows.append(
            {
                "branch": branch,
                "source_column": column,
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
            }
        )
    return rows


def build_penalty_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Score the two control penalties per recording: a stranger's source, a stranger's prior.

    Each is a *paired* quantity -- both branches are scored on the same recording -- so the
    interval is over the per-recording differences and the test is the signed-rank one. Positive
    means the control is worse than the branch it degrades, which is what a working control does.

    Args:
        per_guid: Per-recording means.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per penalty, each carrying the fraction of recordings on which the control cost
        something, with its denominator.
    """
    rows: List[Dict[str, Any]] = []
    for name, control_column, reference_column, meaning in (
        (
            "shuffle_penalty",
            "mc_nll_shuffled_block",
            "mc_nll_base_block",
            "D_shuffled - D_base: what a stranger's source costs against no source at all",
        ),
        (
            "prior_shuffle_penalty",
            "mc_nll_base_shuffled_mu_block",
            "mc_nll_base_block",
            "D_base(shuffled mu_p) - D_base: what a stranger's prior latent costs",
        ),
    ):
        control = finite_column(per_guid, control_column)
        reference = finite_column(per_guid, reference_column)
        difference = control - reference
        interval = shared_stats.bootstrap_ci(difference, resamples=resamples, seed=seed)
        positive = positive_fraction(difference)
        paired = shared_stats.wilcoxon_paired(
            control, reference, label_left=control_column, label_right=reference_column
        )
        rows.append(
            {
                "penalty": name,
                "meaning": meaning,
                **{key: value for key, value in describe(difference).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "positive_fraction": positive["fraction"],
                "n_positive": positive["n_positive"],
                "n_recordings_scored": positive["n"],
                "n_recordings_dropped_not_finite": positive["n_dropped_not_finite"],
                "wilcoxon_p_value": paired["p_value"],
                "wilcoxon_n_pairs": paired["n_pairs"],
            }
        )
    return rows


def build_kl_description(per_guid: pd.DataFrame) -> Dict[str, Any]:
    r"""Report the KL-space reading, labelled as a description and consumed by nothing.

    ``shuffled_exceeds_true`` sits **true** on a healthy model: a mismatched source is out of
    distribution for a posterior trained on matched pairs, so it moves the posterior further from
    the prior, not less. It is reported because a reader will otherwise compute it themselves and
    read it backwards.

    Args:
        per_guid: Per-recording means.

    Returns:
        Both KLs, their difference, and the flag -- under a key that says what it is not.
    """
    true_kl = finite_column(per_guid, _KL_TRUE)
    shuffled_kl = finite_column(per_guid, _KL_SHUFFLED)
    mean_true = float(np.nanmean(true_kl)) if np.isfinite(true_kl).any() else float("nan")
    mean_shuffled = (
        float(np.nanmean(shuffled_kl)) if np.isfinite(shuffled_kl).any() else float("nan")
    )
    comparable = np.isfinite(mean_true) and np.isfinite(mean_shuffled)
    return {
        "source_conditioned_kl_raw": mean_true,
        "source_conditioned_kl_shuffled_raw": mean_shuffled,
        "difference": (mean_shuffled - mean_true) if comparable else float("nan"),
        # A description. The verdict below cannot see it, deliberately.
        "shuffled_exceeds_true": bool(mean_shuffled > mean_true) if comparable else None,
        "note": (
            "descriptive only: a stranger's source is out of distribution for a posterior "
            "trained on matched pairs, so a healthy model moves the posterior *more* under the "
            "control. The specificity criterion is decided in prediction space and cannot read "
            "this."
        ),
    }


def run_perm_control_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the two negative controls per recording and classify what they showed.

    Args:
        context: The analysis context, read for the per-sample table and the pass's own control
            accounting -- how many pairings were drawn, and how many of them landed inside their
            own recording.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the branch table, the two penalties, the KL description, the
        specificity verdict and the outcome classification.
    """
    collection = context.collection
    per_sample = collection.per_sample
    results = dict(getattr(collection, "results", None) or {})
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    per_guid = per_recording_means(per_sample, VALUE_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    branch_rows = build_branch_rows(per_guid, resamples=resamples, seed=seed)
    penalty_rows = build_penalty_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(branch_rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    scores = {row["branch"]: row["mean"] for row in branch_rows}

    def _score(branch: str) -> Optional[float]:
        value = scores.get(branch)
        return None if value is None or not np.isfinite(float(value)) else float(value)

    verdict = source_specificity_verdict(
        _score("base"), _score("full"), _score("shuffled")
    )
    return {
        "n_samples": scored_sample_count(per_sample, "mc_nll_shuffled_block"),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "branches": branch_rows,
        "penalties": penalty_rows,
        "kl_space": build_kl_description(per_guid),
        # The same criterion the run's own verdict list carries, applied to the same
        # per-recording means: one implementation, read here rather than restated.
        "specificity_verdict": verdict.as_dict(),
        "outcome": classify_outcome(_score("base"), _score("full"), _score("shuffled")),
        "outcomes_available": list(OUTCOMES),
        # What the pairing actually did. A control that has silently stopped being a control --
        # every sample paired inside its own recording -- looks exactly like one that works.
        "pairing": dict(results.get("controls") or {}),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [PER_RECORDING_FILENAME, SUMMARY_FILENAME],
    }
