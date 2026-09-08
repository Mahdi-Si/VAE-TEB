r"""Temporal summaries, full-space geometry, and the one shared projection.

Written in LP-12: the aggregation order and the fitting discipline, both implemented below.

Trajectories
------------

Each model's frozen final-hour classifier is applied to every available recording-bin summary over
the six half-hour bins of $(0, 3]$ hours. Aggregation is always anchors within segments, then
segments within recordings, then recordings within a group, so a densely sampled segment cannot
outvote a sparse one. Group means carry GUID-bootstrap bands and per-bin counts; missing observations
stay missing rather than being filled or interpolated. Bins outside the supervised final hour are
labelled as exploratory applications outside the window the loss was defined on.

The paired temporal question uses each recording's mean score in an early window $(2, 3]$ h and a
late window $(0, 1]$ h, and compares $\Delta_i = \mathrm{late}_i - \mathrm{early}_i$ between adverse
and healthy groups among recordings observed in **both** windows, with acidosis and HIE shown
separately and the same calculation repeated for the frozen baseline. Restricting to paired coverage
removes the composition change from different patients contributing to different bins; it does not
remove coverage selection or confounding. An increase toward delivery is consistent with changing
outcome-associated signal. It is not evidence of physiological worsening, and no increase is equally
plausible.

Geometry, in the real latent space
----------------------------------

Class centroids are fitted on **training** recordings in the same fixed standardized $d_z$ space, and
nearest-centroid classification is reported as a head-independent check: linear discrimination can
improve while centroid separation does not, and that combination is stated rather than smoothed over.
Movement, covariance and effective rank are computed in full latent space, never on projection
coordinates. Effective rank is $\exp(-\sum_j p_j \log p_j)$ over the normalized non-negative
covariance eigenvalues, with an explicit convention for the zero-variance case.

Projection
----------

**One** two-component PCA, fitted without labels on the concatenation of the pretrained and adapted
**training** recording-bin vectors after the fixed scaling, with contributions controlled so that
recordings, occupied bins and the two model versions weigh equally. Its map and explained variance
are persisted and then applied unchanged to validation and test and to both models. Refitting per
model, per split or per time bin and reading the axis change as latent motion is the specific mistake
this discipline exists to prevent. A pretrained-only PCA is available as a sensitivity view when
movement into new directions would otherwise be hidden.

No nonlinear embedding is part of the minimum deliverable. If one is added later it is an
unsupervised, seed-fixed appendix with its settings disclosed, and no claim is made from distances
between separate fits: a neighbourhood embedding is a visualization, not a held-out discrimination
test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval.stats import MIN_GROUP_SIZE, bootstrap_ci
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: Column the trajectory and window tables carry each recording's classifier score in.
SCORE_COLUMN = "score"

#: On-disk name of the persisted projection.
PROJECTION_FILENAME = "projection.json"

#: Confidence level of every band this module draws, matching the held-out intervals.
CONFIDENCE = evaluate.CONFIDENCE


# =============================================================================
# Trajectories over the six fixed bins
# =============================================================================
def bin_summaries(
    extraction: Any,
    recordings: pd.DataFrame,
    *,
    split: str,
    bin_hours: float,
    preservation_hours: float,
    key: str = "mu_post",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """One vector per recording per **occupied** trajectory bin.

    Unoccupied bins are absent rather than filled: a recording with no retained anchor in a bin has
    no observation there, and drawing one would invent a measurement. Every consumer below carries
    that absence through instead of interpolating over it.

    Args:
        extraction: The split's extraction.
        recordings: The recording table, with outcomes and eligibility.
        split: The split, checked against the extraction.
        bin_hours: Bin width; the protocol's half hour.
        preservation_hours: The window's upper edge.
        key: Which latent quantity to pool.

    Returns:
        ``(frame, values)``: one row per ``(guid, bin)`` with its label, outcome, class name and
        counts, and the aligned matrix.

    Raises:
        PilotConfigError: If the extraction is another split's, or if nothing survives.
    """
    frame = extraction.retained
    present = sorted({str(value) for value in frame[data.SPLIT_COLUMN].tolist()})
    if present != [str(split)]:
        raise PilotConfigError(
            f"the extraction carries split(s) {present} but bin summaries were requested for "
            f"{split!r}."
        )
    eligible, outcomes = data.eligible_anchors(frame, recordings, split=split)
    if eligible.empty:
        raise PilotConfigError(
            f"no eligible {split!r} recording contributed a retained anchor, so there is no "
            f"trajectory to summarise."
        )
    bins, values = data.recording_bin_means(
        eligible,
        extraction.arrays[key],
        bin_hours=bin_hours,
        preservation_hours=preservation_hours,
    )
    bins = bins.copy()
    bins[data.SPLIT_COLUMN] = str(split)
    bins[data.OUTCOME_COLUMN] = [
        int(outcomes[str(guid)]) for guid in bins[data.GUID_COLUMN].tolist()
    ]
    names = {
        str(row[data.GUID_COLUMN]): row.get(labels.CLASS_COLUMN)
        for _index, row in recordings.iterrows()
    }
    bins[labels.CLASS_COLUMN] = [
        names.get(str(guid)) for guid in bins[data.GUID_COLUMN].tolist()
    ]
    return bins, values


def score_frame(frame: pd.DataFrame, values: np.ndarray, classifier: Any) -> pd.DataFrame:
    r"""Apply one model's **frozen final-hour classifier** to every summary in a table.

    $$s = w^\top S(v) + b,$$

    the same head, unchanged, at every time bin. Bins outside the supervised final hour are marked
    as such on the row: the loss was defined on the last hour, so a score two hours out is an
    application of that head outside the window it was fitted on, and the figure caption has to say
    so. The head is not refitted per bin, which would make each bin's score a different quantity.

    Args:
        frame: Any table of recording summaries, from :func:`bin_summaries` or
            :func:`window_scores`.
        values: The aligned matrix, **unstandardized** -- the classifier holds the scaler.
        classifier: The model's fitted classifier.

    Returns:
        A copy carrying :data:`SCORE_COLUMN`.
    """
    import torch

    classifier.eval()
    with torch.no_grad():
        scores = classifier(
            torch.as_tensor(np.asarray(values, dtype=np.float32))
        ).detach().cpu().numpy()
    out = frame.copy()
    out[SCORE_COLUMN] = np.asarray(scores, dtype=np.float64).reshape(-1)
    return out


def supervised_bins(*, bin_hours: float, supervised_hours: float) -> List[int]:
    """Which trajectory bins lie inside the window the classification loss was defined on.

    Args:
        bin_hours: Bin width.
        supervised_hours: The supervised window's upper edge.

    Returns:
        The bin indices, nearest delivery first. Every other bin is an exploratory application of
        the head outside its own window, and is labelled that way wherever it is drawn.
    """
    return [
        index
        for index, (_low, high) in enumerate(
            data.bin_edges(bin_hours=bin_hours, preservation_hours=supervised_hours)
        )
    ]


def group_bands(
    scored: pd.DataFrame,
    *,
    group_column: str = data.OUTCOME_COLUMN,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
    supervised: Sequence[int] = (),
) -> pd.DataFrame:
    """Group means and GUID-bootstrap bands per bin, with the counts under each.

    The bootstrap is the repository's own :func:`~teb_vae.lag_attn.eval.stats.bootstrap_ci` over
    **recordings**, so a band here means what a band means everywhere else in this tree. A bin with
    too few recordings to resample reports its count and no band rather than an interval decided by
    two order statistics.

    Missing observations stay missing: a recording absent from a bin contributes nothing to that
    bin, and the count printed under the bin is what the band actually rests on -- not the cohort
    size, which would imply a coverage the data does not have.

    Args:
        scored: A bin table carrying :data:`SCORE_COLUMN`, from :func:`score_frame`.
        group_column: What to group by; the binary outcome by default, the class name for the
            acidosis/HIE split.
        resamples: Bootstrap draws.
        seed: Seeds every band, so the figure is reproducible from the record.
        confidence: Band coverage.
        supervised: The bins inside the supervised window, marked on each row.

    Returns:
        One row per ``(group, bin)``: the mean, the band, the recording count and whether the bin
        lies inside the supervised window. ``group`` is text in every case, because one table holds
        the bands of both groupings and the column has to have a single type.
    """
    inside = set(int(value) for value in supervised)
    rows: List[Dict[str, Any]] = []
    for (group, index), block in scored.groupby(
        [group_column, data.BIN_COLUMN], sort=True
    ):
        values = np.asarray(block[SCORE_COLUMN], dtype=np.float64)
        band = bootstrap_ci(
            values, confidence=confidence, resamples=resamples, seed=seed
        )
        rows.append({
            # As text, whatever the grouping column holds. ``stage_evaluate`` stacks the outcome
            # grouping (0/1) and the class grouping ("healthy", "acidosis", "hie") into one table
            # under one ``group`` column, and a column holding both is an object column that
            # parquet refuses to write. The reader already reads it as text.
            "group": str(group),
            data.BIN_COLUMN: int(index),
            data.BIN_LABEL_COLUMN: (
                block[data.BIN_LABEL_COLUMN].iloc[0]
                if data.BIN_LABEL_COLUMN in block.columns else ""
            ),
            "n_recordings": int(len(block)),
            "mean": band["point"],
            "lo": band["lo"],
            "hi": band["hi"],
            "supervised_window": int(index) in inside,
            "band_note": band.get("note", ""),
        })
    table = pd.DataFrame(rows)
    logger.info(
        f"trajectory bands: {len(table)} (group, bin) cell(s) over "
        f"{scored[data.GUID_COLUMN].nunique()} recording(s); "
        f"{int((table['n_recordings'] < MIN_GROUP_SIZE).sum())} cell(s) too small for a band"
    )
    return table


# =============================================================================
# The paired early/late comparison
# =============================================================================
def window_scores(
    extraction: Any,
    recordings: pd.DataFrame,
    classifier: Any,
    *,
    split: str,
    early: Sequence[float],
    late: Sequence[float],
    key: str = "mu_post",
) -> pd.DataFrame:
    r"""Each recording's score in an early and a late window, and their difference.

    $$\Delta_i = \mathrm{late}_i - \mathrm{early}_i,$$

    over the recordings observed in **both** windows. The pairing is the point: different patients
    contribute to different bins, so an unpaired early-versus-late contrast partly measures who was
    recorded when. Pairing removes that composition change. It does **not** remove coverage
    selection -- the recordings observed in both windows are not a random subset -- and it does not
    remove confounding.

    Both windows are reduced by the same anchors-then-segments-then-recordings order, so the
    difference is not an artefact of two aggregation rules.

    Args:
        extraction: The split's extraction.
        recordings: The recording table.
        classifier: The model's frozen final-hour classifier.
        split: The split, checked against the extraction.
        early: ``(low, high)`` of the early window in hours before delivery.
        late: ``(low, high)`` of the late window -- the supervised bag's own window.
        key: Which latent quantity to pool.

    Returns:
        One row per recording observed in both windows: its outcome, class, the two scores, their
        difference and the counts behind each. Recordings observed in one window only are absent,
        and the counts of what was dropped are logged.
    """
    frame = extraction.retained
    present = sorted({str(value) for value in frame[data.SPLIT_COLUMN].tolist()})
    if present != [str(split)]:
        raise PilotConfigError(
            f"the extraction carries split(s) {present} but window scores were requested for "
            f"{split!r}."
        )
    eligible, outcomes = data.eligible_anchors(frame, recordings, split=split)
    names = {
        str(row[data.GUID_COLUMN]): row.get(labels.CLASS_COLUMN)
        for _index, row in recordings.iterrows()
    }

    scored: Dict[str, pd.DataFrame] = {}
    for name, window in (("early", early), ("late", late)):
        rows, values = data.window_means(
            eligible, extraction.arrays[key], low=float(window[0]), high=float(window[1])
        )
        scored[name] = score_frame(rows, values, classifier)

    paired = scored["early"].merge(
        scored["late"], on=data.GUID_COLUMN, suffixes=("_early", "_late")
    )
    paired = paired[[
        data.GUID_COLUMN,
        f"{SCORE_COLUMN}_early",
        f"{SCORE_COLUMN}_late",
        "n_segments_early",
        "n_segments_late",
        "n_anchors_early",
        "n_anchors_late",
    ]].copy()
    paired["delta"] = paired[f"{SCORE_COLUMN}_late"] - paired[f"{SCORE_COLUMN}_early"]
    paired[data.OUTCOME_COLUMN] = [
        int(outcomes[str(guid)]) for guid in paired[data.GUID_COLUMN].tolist()
    ]
    paired[labels.CLASS_COLUMN] = [
        names.get(str(guid)) for guid in paired[data.GUID_COLUMN].tolist()
    ]
    logger.info(
        f"paired windows: {len(paired)} recording(s) observed in both "
        f"({len(scored['early'])} early, {len(scored['late'])} late); the difference is read only "
        f"over the paired subset, which is not a random subset of the cohort"
    )
    return paired


def _difference_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> Dict[str, Any]:
    """Percentile interval on the difference of two independent group means.

    Each group is resampled with replacement on its own, which is what makes this a two-sample
    statement rather than a paired one: the recordings in the two groups are different recordings.

    Args:
        left: One group's values.
        right: The other's.
        resamples: Draws.
        seed: Seeds both.
        confidence: Coverage.

    Returns:
        The point difference, its interval and both counts; the bounds are ``nan`` with a note when
        either group is too small to resample.
    """
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    record: Dict[str, Any] = {
        "point": float(a.mean() - b.mean()) if a.size and b.size else float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "n_left": int(a.size),
        "n_right": int(b.size),
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "method": "two-sample percentile bootstrap over recordings",
    }
    if a.size < MIN_GROUP_SIZE or b.size < MIN_GROUP_SIZE:
        record["note"] = (
            f"groups of {a.size} and {b.size}; below {MIN_GROUP_SIZE} a bootstrap reproduces the "
            f"sample rather than estimating its spread"
        )
        return record
    generator = np.random.default_rng(int(seed))
    draws = (
        a[generator.integers(0, a.size, size=(int(resamples), a.size))].mean(axis=1)
        - b[generator.integers(0, b.size, size=(int(resamples), b.size))].mean(axis=1)
    )
    alpha = 1.0 - float(confidence)
    record["lo"] = float(np.quantile(draws, alpha / 2.0))
    record["hi"] = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return record


def paired_contrast(
    paired: pd.DataFrame,
    *,
    group_column: str = data.OUTCOME_COLUMN,
    adverse: Any = 1,
    healthy: Any = 0,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> Dict[str, Any]:
    """Compare the within-recording change between the two groups.

    Args:
        paired: The table from :func:`window_scores`.
        group_column: What to group by; the binary outcome, or the class name for the
            acidosis/HIE split.
        adverse: The value naming the adverse group in ``group_column``.
        healthy: The value naming the controls.
        resamples: Bootstrap draws.
        seed: Seeds every interval here.
        confidence: Coverage.

    Returns:
        Each group's mean change with its own interval, the between-group difference with its own,
        and the counts. Plus the sentence this result is allowed to support and the two it is not.
    """
    groups = {
        "adverse": np.asarray(
            paired[paired[group_column] == adverse]["delta"], dtype=np.float64
        ),
        "healthy": np.asarray(
            paired[paired[group_column] == healthy]["delta"], dtype=np.float64
        ),
    }
    record: Dict[str, Any] = {
        "group_column": str(group_column),
        "n_paired_recordings": int(len(paired)),
        "groups": {
            name: {
                "n_recordings": int(values.size),
                **bootstrap_ci(
                    values, confidence=confidence, resamples=resamples, seed=seed
                ),
            }
            for name, values in groups.items()
        },
        "difference": _difference_ci(
            groups["adverse"], groups["healthy"],
            resamples=resamples, seed=seed, confidence=confidence,
        ),
        "interpretation": (
            "an increase toward delivery is CONSISTENT WITH changing outcome-associated signal; it "
            "is not evidence of physiological worsening, and no increase is equally plausible"
        ),
        "limitation": (
            "pairing removes the composition change from different patients contributing to "
            "different windows; it removes neither coverage selection nor confounding"
        ),
    }
    logger.info(
        f"paired early/late: adverse {record['groups']['adverse']['point']:.4f} "
        f"(n={record['groups']['adverse']['n_recordings']}), healthy "
        f"{record['groups']['healthy']['point']:.4f} "
        f"(n={record['groups']['healthy']['n_recordings']}); difference "
        f"{record['difference']['point']:.4f}"
    )
    return record


# =============================================================================
# Geometry, in the real latent space
# =============================================================================
def class_centroids(values: Any, outcomes: Any) -> Dict[int, np.ndarray]:
    """Class means in the fixed standardized latent space, fitted on **training** recordings.

    The head-independent half of the geometry check: linear discrimination can improve while the
    classes' centres stay exactly where they were, and only a statement that does not read through
    the fitted head can say so.

    Args:
        values: Standardized recording vectors, ``(N, d_z)``.
        outcomes: Their binary outcomes.

    Returns:
        Outcome -> centroid.

    Raises:
        PilotConfigError: If a class is absent, where its centroid is not a small measurement but
            an undefined one.
    """
    matrix = np.asarray(values, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.int64).reshape(-1)
    centroids: Dict[int, np.ndarray] = {}
    for outcome in (0, 1):
        rows = y == outcome
        if not rows.any():
            raise PilotConfigError(
                f"no training recording carries outcome {outcome}, so its centroid is undefined "
                f"and nearest-centroid classification has nothing to measure against."
            )
        centroids[outcome] = matrix[rows].mean(axis=0)
    return centroids


def nearest_centroid_scores(values: Any, centroids: Mapping[int, Any]) -> np.ndarray:
    r"""A higher-is-more-adverse score from the distance to each training centroid.

    $$s_i = \lVert x_i - c_0 \rVert_2 - \lVert x_i - c_1 \rVert_2 ,$$

    so the sign says which centroid is nearer and the magnitude how much. Reported through the same
    :func:`~latent_pilot.evaluate.recording_metrics` as the fitted head, on the same recordings, so
    the two are comparable -- and an improvement in one without the other is a statement this
    package can make rather than smooth over.

    **In full latent space**, never on projection coordinates: two components of a map fitted for a
    picture are not the space the model works in.

    Args:
        values: Standardized recording vectors ``(N, d_z)``.
        centroids: The training centroids.

    Returns:
        One score per row.
    """
    matrix = np.asarray(values, dtype=np.float64)
    healthy = np.linalg.norm(matrix - np.asarray(centroids[0], dtype=np.float64), axis=1)
    adverse = np.linalg.norm(matrix - np.asarray(centroids[1], dtype=np.float64), axis=1)
    return healthy - adverse


def effective_rank(eigenvalues: Any) -> float:
    r"""The exponential of the eigenvalue-spectrum entropy.

    $$\mathrm{erank} = \exp\left(-\sum_j p_j \log p_j\right), \qquad
    p_j = \frac{\lambda_j}{\sum_k \lambda_k},$$

    over the **non-negative** eigenvalues -- a covariance's true spectrum is non-negative, and the
    small negatives an eigendecomposition returns are numerical rather than directions.

    **The zero-variance convention is explicit**: when nothing varies the eigenvalues sum to zero,
    $p$ is undefined, and this returns $0.0$ -- no direction carries anything. It is not $1$, which
    would say one direction does.

    Args:
        eigenvalues: The covariance's eigenvalues.

    Returns:
        The effective rank, in $[0, d]$.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    positive = values[values > 0.0]
    total = float(positive.sum())
    if positive.size == 0 or total <= 0.0:
        return 0.0
    shares = positive / total
    return float(np.exp(-np.sum(shares * np.log(shares))))


def covariance_summary(values: Any) -> Dict[str, Any]:
    """The spread of a set of standardized recording vectors, in full latent space.

    Args:
        values: Standardized recording vectors ``(N, d_z)``.

    Returns:
        The total variance, the eigenvalue spectrum, the effective rank, and the share the leading
        direction carries -- the pair that says whether a representation is using its width or one
        direction of it.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape[0] < 2:
        return {
            "n_recordings": int(matrix.shape[0]),
            "d_z": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "total_variance": float("nan"),
            "effective_rank": float("nan"),
            "leading_share": float("nan"),
            "eigenvalues": [],
            "note": "fewer than two recordings; a covariance is undefined rather than zero",
        }
    covariance = np.cov(matrix, rowvar=False)
    spectrum = np.sort(np.linalg.eigvalsh(np.atleast_2d(covariance)))[::-1]
    total = float(spectrum[spectrum > 0.0].sum())
    return {
        "n_recordings": int(matrix.shape[0]),
        "d_z": int(matrix.shape[1]),
        "total_variance": total,
        "effective_rank": effective_rank(spectrum),
        "leading_share": float(spectrum[0] / total) if total > 0.0 else float("nan"),
        "eigenvalues": [float(value) for value in spectrum],
    }


def movement_summary(before: Any, after: Any, *, scale: Any) -> Dict[str, Any]:
    r"""How far the adaptation moved each recording, in standardized latent units.

    $$m_i = \left\lVert \frac{v_i^{\rm after} - v_i^{\rm before}}{s} \right\rVert_2 .$$

    Standardized by the **frozen training scaler**, the same constants every other number in this
    run is expressed in, so a movement of one is a movement of one training standard deviation
    rather than of whatever the coordinate happened to be scaled by.

    Args:
        before: The frozen model's recording vectors, unstandardized.
        after: The adapted model's, on the same recordings in the same order.
        scale: The frozen scaler's per-coordinate scales.

    Returns:
        The distribution of the movement over recordings and the mean per-coordinate movement.

    Raises:
        PilotConfigError: If the two are not the same recordings in the same order.
    """
    left = np.asarray(before, dtype=np.float64)
    right = np.asarray(after, dtype=np.float64)
    if left.shape != right.shape:
        raise PilotConfigError(
            f"movement was asked for between {left.shape} and {right.shape} vectors. The two "
            f"models are read on the same recordings, in the same order; a shape mismatch means a "
            f"change of population is about to be reported as a change of representation."
        )
    residual = (right - left) / np.asarray(scale, dtype=np.float64)
    distances = np.linalg.norm(residual, axis=1)
    return {
        "n_recordings": int(distances.size),
        "d_z": int(left.shape[1]) if left.ndim == 2 else 0,
        "mean": float(distances.mean()) if distances.size else float("nan"),
        "median": float(np.median(distances)) if distances.size else float("nan"),
        "q25": float(np.quantile(distances, 0.25)) if distances.size else float("nan"),
        "q75": float(np.quantile(distances, 0.75)) if distances.size else float("nan"),
        "max": float(distances.max()) if distances.size else float("nan"),
        "mean_per_coordinate": (
            [float(value) for value in np.abs(residual).mean(axis=0)] if distances.size else []
        ),
        "units": "training standard deviations of the frozen scaler",
    }


# =============================================================================
# The one shared projection
# =============================================================================
@dataclass(frozen=True)
class Projection:
    """A two-component map, fitted once and then applied unchanged.

    Attributes:
        mean: The weighted centre it subtracts, ``(d_z,)``.
        components: Its axes, ``(n_components, d_z)``, orthonormal and sign-fixed.
        explained_variance_ratio: What share of the weighted total variance each axis carries.
        record: The population it was fitted on and the weighting that made every recording, bin
            and model version count equally.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    record: Dict[str, Any]

    def transform(self, values: Any) -> np.ndarray:
        """Project standardized recording vectors onto the fitted axes.

        Args:
            values: ``(N, d_z)`` standardized vectors, from any split and any model version.

        Returns:
            ``(N, n_components)`` coordinates. The same map every time: that is the whole
            discipline, and it is why this class holds no fitting method.
        """
        matrix = np.asarray(values, dtype=np.float64)
        return (matrix - self.mean) @ self.components.T


def projection_weights(frame: pd.DataFrame) -> np.ndarray:
    r"""Per-row weights giving every recording, and every occupied bin within it, an equal share.

    $$w_{ib} = \frac{1}{N \cdot B_i},$$

    with $N$ recordings and $B_i$ the bins recording $i$ actually occupies. Without this a
    well-covered recording contributing six bins would weigh six times a sparse one contributing
    one, and the map's leading direction would describe coverage as much as latent structure.

    Args:
        frame: A bin table carrying :data:`~latent_pilot.data.GUID_COLUMN`.

    Returns:
        Weights summing to one.
    """
    guids = [str(value) for value in frame[data.GUID_COLUMN].tolist()]
    occupied: Dict[str, int] = {}
    for guid in guids:
        occupied[guid] = occupied.get(guid, 0) + 1
    n_recordings = len(occupied)
    return np.asarray(
        [1.0 / (n_recordings * occupied[guid]) for guid in guids], dtype=np.float64
    )


def fit_projection(
    versions: Mapping[str, Tuple[pd.DataFrame, Any]],
    *,
    n_components: int = 2,
) -> Projection:
    r"""Fit **one** label-free PCA on the training bin vectors of every model version at once.

    Three things it does not do, each of which is a way of reading an artefact as a result:

    * **It sees no labels.** A supervised axis exists already -- the classifier's -- and is drawn
      separately and named as such. A projection fitted with labels would put the separation into
      the picture and then report the picture as evidence of it.
    * **It is fitted once, on training data.** Validation and test are transformed by the saved map.
      A map refitted per split would move its own axes between panels.
    * **It is fitted on both versions together.** Refitting per model and reading the axis change as
      latent motion is the specific mistake this discipline exists to prevent: the two panels of the
      before/after figure share axes because they share this map.

    Weighting: every model version gets $1/V$, every recording within a version $1/N$, and every
    occupied bin within a recording $1/B_i$. So the map describes the cohort rather than whoever was
    recorded longest.

    Signs are fixed deterministically -- each axis is oriented so its largest-magnitude coordinate
    is positive -- because an eigendecomposition is free to return either sign and a flipped panel
    between two runs would read as movement.

    Args:
        versions: ``{version name: (training bin frame, standardized values)}``. One entry gives
            the pretrained-only sensitivity map; two give the shared map the main figure uses.
        n_components: Axes to keep.

    Returns:
        The projection.

    Raises:
        PilotConfigError: If a frame carries any split other than training, if the versions
            disagree on width, or if there is too little to fit.
    """
    if not versions:
        raise PilotConfigError("a projection needs at least one model version to fit on.")

    blocks: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    counts: Dict[str, Any] = {}
    width: Optional[int] = None
    for name, (frame, values) in versions.items():
        splits = sorted({str(value) for value in frame.get(data.SPLIT_COLUMN, [])})
        if splits and splits != ["train"]:
            raise PilotConfigError(
                f"version {name!r} carries split(s) {splits}. The projection is fitted on training "
                f"data alone and applied unchanged to the rest; fitting it on validation or test "
                f"would put those populations into the axes every panel is drawn on."
            )
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(frame):
            raise PilotConfigError(
                f"version {name!r} supplies {matrix.shape} values for {len(frame)} row(s)."
            )
        if width is None:
            width = int(matrix.shape[1])
        elif int(matrix.shape[1]) != width:
            raise PilotConfigError(
                f"version {name!r} is {matrix.shape[1]}-dimensional and an earlier one is {width}; "
                f"one map cannot span two latent widths."
            )
        blocks.append(matrix)
        weights.append(projection_weights(frame) / len(versions))
        counts[name] = {
            "n_rows": int(len(frame)),
            "n_recordings": int(frame[data.GUID_COLUMN].nunique()),
        }

    matrix = np.concatenate(blocks, axis=0)
    weight = np.concatenate(weights, axis=0)
    weight = weight / weight.sum()
    if matrix.shape[0] < n_components + 1:
        raise PilotConfigError(
            f"{matrix.shape[0]} row(s) cannot support a {n_components}-component projection."
        )

    mean = weight @ matrix
    centred = matrix - mean
    covariance = (centred * weight[:, None]).T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: int(n_components)]
    components = eigenvectors[:, order].T.copy()
    spectrum = eigenvalues[order]

    # Deterministic orientation: an eigendecomposition may return either sign, and a panel that
    # flipped between two runs would read as movement rather than as a convention.
    for index, axis in enumerate(components):
        if axis[int(np.argmax(np.abs(axis)))] < 0.0:
            components[index] = -axis

    total = float(eigenvalues[eigenvalues > 0.0].sum())
    ratio = np.asarray(
        [float(value / total) if total > 0.0 else float("nan") for value in spectrum],
        dtype=np.float64,
    )
    record = {
        "population": "train",
        "versions": sorted(versions),
        "counts": counts,
        "n_components": int(n_components),
        "d_z": int(width or 0),
        "n_rows": int(matrix.shape[0]),
        "total_variance": total,
        "explained_variance_ratio": [float(value) for value in ratio],
        "weighting": (
            "each model version 1/V, each recording within a version 1/N, each occupied bin within "
            "a recording 1/B_i"
        ),
        "labels_used": False,
        "note": (
            "one map for every version and every split; refitting it per model, per split or per "
            "bin and reading the axis change as latent motion is the mistake it exists to prevent"
        ),
    }
    logger.info(
        f"projection fitted on {matrix.shape[0]} training row(s) from "
        f"{', '.join(sorted(versions))}: explained variance "
        f"{', '.join(f'{value:.3f}' for value in ratio)}"
    )
    return Projection(
        mean=mean,
        components=components,
        explained_variance_ratio=ratio,
        record=record,
    )


def save_projection(projection: Projection, directory: Any) -> Path:
    """Persist the map so every later panel is drawn on the axes this one fitted.

    Args:
        projection: The fitted map.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / PROJECTION_FILENAME
    target.write_text(
        json.dumps(
            {
                "mean": projection.mean.tolist(),
                "components": projection.components.tolist(),
                "explained_variance_ratio": projection.explained_variance_ratio.tolist(),
                "record": projection.record,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


def load_projection(directory: Any) -> Projection:
    """Read the saved map back.

    Args:
        directory: The run directory.

    Returns:
        The projection.

    Raises:
        FileNotFoundError: If it was never written. The map is fitted once, during analysis, and
            every later figure reads that one file rather than refitting.
    """
    target = Path(directory) / PROJECTION_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} is missing. The projection is fitted once, on training data, and every "
            f"panel is drawn on that one map; a figure that refitted it would move its own axes."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    return Projection(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        components=np.asarray(payload["components"], dtype=np.float64),
        explained_variance_ratio=np.asarray(
            payload["explained_variance_ratio"], dtype=np.float64
        ),
        record=dict(payload.get("record") or {}),
    )


__all__ = [
    "CONFIDENCE",
    "PROJECTION_FILENAME",
    "SCORE_COLUMN",
    "Projection",
    "bin_summaries",
    "class_centroids",
    "covariance_summary",
    "effective_rank",
    "fit_projection",
    "group_bands",
    "load_projection",
    "movement_summary",
    "nearest_centroid_scores",
    "paired_contrast",
    "projection_weights",
    "save_projection",
    "score_frame",
    "supervised_bins",
    "window_scores",
]
