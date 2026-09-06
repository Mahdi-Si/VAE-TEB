r"""Physiological events: does the forecast see a deceleration coming, and do contractions matter?

This is the analysis the raw target exists for. Everything else in this pipeline scores the
forecast as a number; here it is scored as a *waveform*, against the two events a clinician reads
a trace for.

**Deceleration forecasting skill.** The detector runs on the true raw FHR and on each branch's
forecast, both in bpm, block by block, and the two detections are matched. Two things about the
geometry have to be stated rather than discovered:

* The ported detector needs the event to sit at least ``edge_seconds`` from either end of what it
  is given, so an $H \cdot R$-sample block ($4H$ s) leaves an interior two ``edge_seconds`` shorter.
  Part of the forecast horizon is not searchable at all, and every rate here is a rate over that interior.
* Consecutive anchors' blocks overlap in $H - 1$ of their $H$ steps, so one physiological
  deceleration is re-detected once per anchor -- roughly fifteen times over the usable interior.
  An anchor-level hit rate is pseudo-replicated by that factor. **Fixing the horizon step $\tau$
  removes it exactly**: for a given $\tau$ there is exactly one anchor whose block places a given
  absolute raw sample at that step, so a rate computed per $\tau$ counts each event once, by
  construction rather than by a de-duplication rule bolted on afterwards. Both counts are emitted
  so the factor is a measured number.

**Contraction-triggered response.** The forecast, the truth and the difference $\mu^q - \mu^p$,
averaged over the blocks that follow a detected contraction, against a **count-matched
per-recording random-trigger null**. The null matters more than it looks: the reported statistic
is the deepest point of the average inside a response window, and a minimum over a window is
negative on *any* data. Passing the null through the identical operator is what turns that
selection bias from an assumption into a measurement.

**Contraction-conditioned coupling.** ``pred_gap`` and $K_t$ restricted to anchors within
``event_lag_window_s`` of a contraction, against count-matched control anchors from the same
recordings. This is the per-anchor table's load-bearing consumer: a per-segment mean cannot
express "the anchors near a contraction", and it is the one half of this analysis that needs no
retained waveform at all -- the contraction timing already travels on that table.

**Two halves, two gates.** The two waveform readouts need actual forecast blocks, which are
retained only under ``eval_config.caps.waveforms`` and are two megabytes a sample; the conditioned
coupling reads the anchor table and runs on every anchor of the split. So they gate separately,
and a run that retained nothing records a skip naming the key to set rather than reporting rates
over no data. The cost of the first half is worth stating: it is three detector calls per anchor
per retained sample, so ``caps.waveforms`` is the control on it as well as on the memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_rws.eval import cohort, events
from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_rws.eval.collect import CONTRACTION_AGE_COLUMN
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
)
from teb_vae.lag_attn_rws.eval.metrics import to_bpm
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "events"

#: What it writes.
DECELERATION_FILENAME = "deceleration_skill.csv"
TRIGGERED_FILENAME = "contraction_triggered.csv"
CONDITIONED_FILENAME = "conditioned_coupling.csv"
CONDITIONED_PER_RECORDING_FILENAME = "conditioned_coupling_per_recording.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them.
DECELERATION_FIGURE = "deceleration_skill"
TRIGGERED_FIGURE = "contraction_triggered"
CONDITIONED_FIGURE = "conditioned_coupling"

#: The two model branches scored against the truth, and the difference reported beside them.
BRANCHES: Tuple[str, ...] = ("base", "full")

#: How close a forecast deceleration must land to a true one to count as the same event, in
#: seconds. Strictly below **half** the detector's own ``min_distance``, and that bound is the
#: point rather than the value: two detections are at least ``min_distance`` apart, so both can be
#: within this of one truth event only if it sits less than twice this from each. Below the bound
#: the greedy matching and a nearest-neighbour lookup provably agree and neither can become
#: order-dependent; above it, they can disagree and the answer would depend on iteration order.
MATCH_TOLERANCE_S = 0.5 * events.DECELERATION_MIN_DISTANCE_S - 1.5

#: Seconds before a contraction onset the triggered average is baselined against. The correction
#: is taken from the **truth** and subtracted from all four curves alike, so the difference curve
#: is exactly $\mu^q - \mu^p$ rather than a difference of two separately baselined things.
TRIGGER_BASELINE_S = 30.0

#: Where the triggered response is looked for, in seconds after the contraction onset. The
#: physiological contraction-to-deceleration delay is $20$-$120\,$s, and the preprocessing has
#: already advanced UP by $20\,$s, so this is that interval on the aligned timeline rather than on
#: the sensor's.
RESPONSE_WINDOW_S: Tuple[float, float] = (20.0, 120.0)

#: Random-trigger draws behind the null band and the null dip distribution. A module constant
#: rather than an ``eval_config`` key, for the reason ``alpha`` is not one: an operator who could
#: raise it could make a band tighten until a finding appeared.
NULL_DRAWS = 200

#: Half-width of the drawn null band, in standard deviations of the null draws. Two, so the band
#: brackets its own mean and a curve leaving it is a two-sigma statement rather than a decoration.
NULL_BAND_SIGMAS = 2.0

#: Guards. Below either of these the conditioned comparison is a recorded skip rather than a
#: number: a rate over a handful of anchors from one or two recordings is dominated by which
#: recordings they were.
MIN_EVENT_ANCHORS = 200
MIN_EVENT_RECORDINGS = 4

#: The same guard for the triggered average, in triggers rather than anchors.
MIN_TRIGGERS = 20

#: The two coupling readouts the conditioned comparison cuts, as ``(reported name, column)``.
CONDITIONED_READOUTS: Tuple[Tuple[str, str], ...] = (
    ("pred_gap_mc_nats", "mc_pred_gap"),
    ("source_conditioned_kl_raw", "kld_per_t"),
)


# =============================================================================
# Reaching the retained waveforms
# =============================================================================
def retained_frame(collection: Any) -> pd.DataFrame:
    """Return the per-sample rows the waveform retention kept, in retained-array row order.

    Args:
        collection: What the shared pass produced.

    Returns:
        One row per retained sample, carrying ``row`` (its index into the retained arrays) beside
        the identity columns. Empty when nothing was retained.
    """
    retained = dict(getattr(collection, "retained", None) or {})
    indices = retained.get("waveforms_sample_index")
    per_sample = getattr(collection, "per_sample", None)
    if indices is None or per_sample is None or per_sample.empty:
        return pd.DataFrame(columns=["row", "sample_index", "guid", labels.CLASS_COLUMN])
    frame = per_sample.set_index("sample_index")
    keep = [int(value) for value in np.asarray(indices).tolist() if int(value) in frame.index]
    rows = frame.loc[keep].reset_index()
    rows.insert(0, "row", np.arange(len(rows), dtype=np.int64))
    return rows


def _has_waveforms(collection: Any) -> bool:
    """Whether the pass retained everything the waveform readouts need."""
    retained = dict(getattr(collection, "retained", None) or {})
    needed = ("target", "mu_base", "mu_full", "up_raw", "weight", "waveforms_sample_index")
    return all(name in retained for name in needed) and len(retained["target"]) > 0


def _blocks(array: Any, row: int) -> np.ndarray:
    """Return one retained sample's per-anchor blocks, flattened to $(T, H \\cdot R)$."""
    values = np.asarray(array[int(row)], dtype=np.float64)
    return values.reshape(values.shape[0], -1)


def _block_validity(weight: Any, row: int, *, decimation: int, raw_len: int) -> np.ndarray:
    """Return the raw-sample validity of one retained sample's whole trimmed trace."""
    return events.raw_validity(
        np.asarray(weight[int(row)], dtype=np.float64), decimation=decimation, raw_len=raw_len
    )


# =============================================================================
# Deceleration forecasting skill
# =============================================================================
def deceleration_detections(
    blocks_bpm: np.ndarray,
    validity: np.ndarray,
    *,
    decimation: int,
    raw_per_step: int,
    usable_tau: np.ndarray,
) -> Dict[int, List[int]]:
    r"""Detect decelerations block by block and index them by horizon step.

    Args:
        blocks_bpm: One sample's per-anchor forecast or truth blocks, $(T, H \cdot R)$, in bpm.
        validity: The sample's raw-sample validity over the whole trimmed trace.
        decimation: Raw samples per decimated step $D$, which is also where anchor $t$'s block
            starts: $D(t + 1)$.
        raw_per_step: Raw samples per horizon step $R$.
        usable_tau: The horizon steps a detection may be counted at.

    Returns:
        ``{tau: [absolute raw index, ...]}``, one entry per usable $\tau$. Absolute rather than
        block-relative, because that is the frame in which a forecast detection and a true one are
        the same event.
    """
    found: Dict[int, List[int]] = {int(tau): [] for tau in usable_tau}
    allowed = set(found)
    n_anchors, block_len = blocks_bpm.shape
    for anchor in range(n_anchors):
        start = decimation * (anchor + 1)
        window = validity[start : start + block_len]
        if window.size < block_len or not window.any():
            continue
        nadirs = events.detect_decelerations(blocks_bpm[anchor], valid=window)["nadir_raw"]
        for nadir in nadirs.tolist():
            tau = int(nadir) // int(raw_per_step)
            if tau in allowed:
                found[tau].append(start + int(nadir))
    return found


def skill_rows(
    per_sample_detections: Sequence[Dict[str, Any]],
    *,
    usable_tau: np.ndarray,
    raw_per_step: int,
    resamples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    r"""Turn per-sample detections into per-horizon-step rates on per-recording units.

    Within a recording the hits and the events are **summed** across its segments before the rate
    is formed, so a segment carrying one event does not weigh as much as one carrying ten; across
    recordings the rate is averaged, which is the unit every clinical question in this pipeline is
    asked in.

    Args:
        per_sample_detections: One entry per retained sample, carrying its ``guid`` and, per
            branch, the truth and branch detections indexed by $\tau$.
        usable_tau: The horizon steps rates are reported at.
        raw_per_step: Raw samples per horizon step, for the lead-time axis.
        resamples: Bootstrap resamples over recordings.
        seed: Bootstrap seed.

    Returns:
        One row per ``(branch, tau)``: the hit rate and its interval, the false-alarm rate, the
        absolute lead-time error over matched pairs, and every count behind them.
    """
    tolerance = int(round(MATCH_TOLERANCE_S * events.FS_RAW))
    rows: List[Dict[str, Any]] = []
    for branch in BRANCHES:
        for tau in usable_tau.tolist():
            per_guid: Dict[str, Dict[str, float]] = {}
            errors: List[float] = []
            for entry in per_sample_detections:
                truth = sorted(entry["truth"].get(int(tau), ()))
                found = sorted(entry[branch].get(int(tau), ()))
                matched_truth, matched_found = events.match_events(
                    truth, found, tolerance=tolerance
                )
                cell = per_guid.setdefault(
                    str(entry["guid"]), {"hits": 0.0, "truth": 0.0, "false": 0.0, "found": 0.0}
                )
                cell["hits"] += float(matched_truth.sum())
                cell["truth"] += float(len(truth))
                cell["false"] += float((~matched_found).sum())
                cell["found"] += float(len(found))
                for position, index in enumerate(truth):
                    if not matched_truth[position]:
                        continue
                    partner = min(found, key=lambda value: abs(value - index))
                    errors.append(abs(partner - index) / events.FS_RAW)
            rows.append(
                _skill_row(
                    branch=branch, tau=int(tau), per_guid=per_guid, errors=errors,
                    raw_per_step=raw_per_step, resamples=resamples, seed=seed,
                )
            )
    return rows


def _skill_row(
    *,
    branch: str,
    tau: int,
    per_guid: Dict[str, Dict[str, float]],
    errors: Sequence[float],
    raw_per_step: int,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Assemble one ``(branch, tau)`` row from its per-recording numerators and denominators."""
    hit_rates = np.array(
        [cell["hits"] / cell["truth"] for cell in per_guid.values() if cell["truth"] > 0],
        dtype=np.float64,
    )
    false_rates = np.array(
        [cell["false"] / cell["found"] for cell in per_guid.values() if cell["found"] > 0],
        dtype=np.float64,
    )
    interval = shared_stats.bootstrap_ci(hit_rates, resamples=resamples, seed=seed)
    error_array = np.asarray(list(errors), dtype=np.float64)
    return {
        "branch": branch,
        "horizon_step": int(tau),
        # The block starts one raw sample past the anchor's causal endpoint, so step tau is
        # (tau + 1) forecast steps ahead of it.
        "lead_time_s": float(tau + 1) * SECONDS_PER_STEP,
        "hit_rate": float(hit_rates.mean()) if hit_rates.size else float("nan"),
        "hit_rate_ci_lo": interval["lo"],
        "hit_rate_ci_hi": interval["hi"],
        "false_alarm_rate": float(false_rates.mean()) if false_rates.size else float("nan"),
        "lead_time_abs_error_s": float(error_array.mean()) if error_array.size else float("nan"),
        "n_recordings_with_events": int(hit_rates.size),
        "n_true_events": int(sum(cell["truth"] for cell in per_guid.values())),
        "n_forecast_events": int(sum(cell["found"] for cell in per_guid.values())),
        "n_matched": int(error_array.size),
        "match_tolerance_s": float(MATCH_TOLERANCE_S),
        "raw_per_step": int(raw_per_step),
    }


# =============================================================================
# Contraction-triggered response
# =============================================================================
def triggered_average(
    samples: Sequence[Dict[str, Any]], *, rng: np.random.Generator
) -> Dict[str, Any]:
    r"""Average each curve over the blocks following a contraction, and over random triggers.

    Args:
        samples: One entry per retained sample, carrying its ``guid``, the per-anchor blocks of
            each curve, its valid trigger range and its detected contraction onsets.
        rng: Generator for the null triggers, seeded by the caller.

    Returns:
        The observed curves, the null band, the dip statistic per curve with its null, and the
        counts behind them. Every array is a list so the record serialises.
    """
    curve_names = ("truth", *BRANCHES, "difference")
    real = [np.asarray(entry["onsets"], dtype=np.int64) for entry in samples]
    observed: Dict[str, np.ndarray] = {}
    n_triggers = 0
    for name in curve_names:
        observed[name], n_triggers = _average(samples, real, name)
    null_curves: Dict[str, List[np.ndarray]] = {name: [] for name in curve_names}
    for _draw in range(NULL_DRAWS):
        # Count-matched per recording: every sample contributes exactly as many random triggers as
        # it has real contractions, so the null differs from the observation only in that the
        # triggers are decoupled from the contractions.
        drawn = [
            _random_triggers(entry, rng, int(onsets.size))
            for entry, onsets in zip(samples, real)
        ]
        for name in curve_names:
            null_curves[name].append(_average(samples, drawn, name)[0])
    return _triggered_record(observed, null_curves, n_triggers=n_triggers, n_samples=len(samples))


def _average(
    samples: Sequence[Dict[str, Any]], triggers: Sequence[np.ndarray], name: str
) -> Tuple[np.ndarray, int]:
    """Average one curve's baseline-corrected snippets over every sample's triggers.

    Args:
        samples: The retained samples.
        triggers: Each sample's trigger indices, parallel to ``samples``.
        name: Which curve.

    Returns:
        ``(mean snippet, contributing trigger count)``. All-NaN when nothing contributed.
    """
    total: Optional[np.ndarray] = None
    count = 0
    for entry, drawn in zip(samples, triggers):
        if np.asarray(drawn).size == 0:
            continue
        snippets = _snippets(entry, name, np.asarray(drawn, dtype=np.int64))
        if snippets.shape[0] == 0:
            continue
        total = snippets.sum(axis=0) if total is None else total + snippets.sum(axis=0)
        count += int(snippets.shape[0])
    if total is None or count == 0:
        width = int(samples[0]["blocks"]["truth"].shape[1]) if samples else 0
        return np.full(width, np.nan, dtype=np.float64), 0
    return total / float(count), count


def _snippets(entry: Dict[str, Any], name: str, triggers: np.ndarray) -> np.ndarray:
    """Return one sample's baseline-corrected block snippets at the given triggers.

    The baseline is the **truth**'s mean over the seconds before the trigger, subtracted from every
    curve alike. Baselining each curve against its own pre-window would make the difference curve
    a difference of two separately shifted things rather than $\\mu^q - \\mu^p$.
    """
    decimation = int(entry["decimation"])
    blocks = entry["blocks"][name]
    truth_trace = entry["truth_trace"]
    pre = int(round(TRIGGER_BASELINE_S * events.FS_RAW))
    usable = triggers[(triggers >= entry["first_trigger"]) & (triggers <= entry["last_trigger"])]
    if usable.size == 0:
        return np.zeros((0, blocks.shape[1]), dtype=np.float64)
    anchors = np.ceil(usable / float(decimation)).astype(np.int64) - 1
    keep = (anchors >= 0) & (anchors < blocks.shape[0])
    usable, anchors = usable[keep], anchors[keep]
    if usable.size == 0:
        return np.zeros((0, blocks.shape[1]), dtype=np.float64)
    # The trace starts at raw index `decimation`, so a raw index maps to it by that offset.
    offsets = np.arange(-pre, 0, dtype=np.int64)
    baseline_index = usable[:, None] - decimation + offsets[None, :]
    valid = (baseline_index >= 0) & (baseline_index < truth_trace.size)
    baseline = np.where(valid, truth_trace[np.clip(baseline_index, 0, truth_trace.size - 1)],
                        np.nan)
    correction = np.nanmean(baseline, axis=1, keepdims=True)
    # The difference curve is a difference of two forecasts, so the truth's level cancels out of
    # it exactly -- subtracting the correction would shift it away from zero for no reason.
    if name == "difference":
        return blocks[anchors]
    return blocks[anchors] - correction


def _random_triggers(
    entry: Dict[str, Any], rng: np.random.Generator, count: int
) -> np.ndarray:
    """Draw ``count`` uniform triggers from one sample's valid range -- the count-matched null."""
    if count <= 0 or entry["last_trigger"] < entry["first_trigger"]:
        return np.empty(0, dtype=np.int64)
    return rng.integers(entry["first_trigger"], entry["last_trigger"] + 1, size=count)


def _triggered_record(
    observed: Dict[str, np.ndarray],
    null_curves: Dict[str, List[np.ndarray]],
    *,
    n_triggers: int,
    n_samples: int,
) -> Dict[str, Any]:
    """Assemble the triggered-average record, including the dip statistic and its null."""
    record: Dict[str, Any] = {
        "n_triggers": int(n_triggers),
        "n_samples": int(n_samples),
        "n_null_draws": int(NULL_DRAWS),
        "response_window_s": list(RESPONSE_WINDOW_S),
        "baseline_window_s": float(TRIGGER_BASELINE_S),
        "null_band_sigmas": float(NULL_BAND_SIGMAS),
        # Every trigger here is a contraction onset, and this pipeline's onset is not the one the
        # sibling's published stage-2 numbers were computed with. A lag read off this figure and
        # a lag read off those is not the same measurement, and the divergence travels with the
        # number rather than living only in a docstring.
        "onset_convention": events.ONSET_WALK_BACK_NOTE,
        "curves": {},
    }
    for name, values in observed.items():
        stack = np.vstack(null_curves[name]) if null_curves[name] else np.zeros((0, values.size))
        null_mean = stack.mean(axis=0) if stack.size else np.full_like(values, np.nan)
        null_sd = stack.std(axis=0) if stack.size else np.full_like(values, np.nan)
        observed_dip, observed_lag = _dip(values)
        null_dips = np.array([_dip(row)[0] for row in stack], dtype=np.float64)
        spread = float(np.nanstd(null_dips)) if null_dips.size else float("nan")
        record["curves"][name] = {
            "mean": values.tolist(),
            "null_mean": null_mean.tolist(),
            "null_lo": (null_mean - NULL_BAND_SIGMAS * null_sd).tolist(),
            "null_hi": (null_mean + NULL_BAND_SIGMAS * null_sd).tolist(),
            "dip": observed_dip,
            "dip_lag_s": observed_lag,
            "null_dip_mean": float(np.nanmean(null_dips)) if null_dips.size else float("nan"),
            "null_dip_sd": spread,
            # The whole point of passing the null through the identical minimum: a dip inside the
            # null band is the operator's own selection bias, not a response.
            "dip_z": (
                float((observed_dip - float(np.nanmean(null_dips))) / spread)
                if spread and np.isfinite(spread) and spread > 0.0 else float("nan")
            ),
        }
    return record


def _dip(curve: np.ndarray) -> Tuple[float, float]:
    """Return the deepest point of a curve inside the response window and its lag in seconds."""
    lo = int(round(RESPONSE_WINDOW_S[0] * events.FS_RAW))
    hi = min(int(round(RESPONSE_WINDOW_S[1] * events.FS_RAW)), int(curve.size))
    if hi <= lo or not np.isfinite(curve[lo:hi]).any():
        return float("nan"), float("nan")
    window = curve[lo:hi]
    position = int(np.nanargmin(window))
    return float(window[position]), float(lo + position) / events.FS_RAW


# =============================================================================
# Contraction-conditioned coupling
# =============================================================================
def conditioned_anchors(
    per_anchor: pd.DataFrame, *, window_s: float, seed: int
) -> pd.DataFrame:
    r"""Split the anchor table into event anchors and count-matched control anchors.

    Args:
        per_anchor: The per-anchor table, carrying :data:`~teb_vae.lag_attn_rws.eval.collect.
            CONTRACTION_AGE_COLUMN`.
        window_s: Seconds after a contraction onset an anchor still counts as conditioned.
        seed: Seed for the control draw, so the comparison is reproducible from the summary.

    Returns:
        The event and control rows with a ``condition`` column. Controls are drawn **within each
        recording** and matched to that recording's event count, so the comparison cannot be a
        comparison between recordings wearing two labels. Empty when the column is absent.
    """
    if per_anchor.empty or CONTRACTION_AGE_COLUMN not in per_anchor.columns:
        return per_anchor.head(0).assign(condition=pd.Series(dtype=object))
    age = np.asarray(per_anchor[CONTRACTION_AGE_COLUMN], dtype=np.float64)
    near = np.isfinite(age) & (age >= 0.0) & (age <= float(window_s))
    events_frame = per_anchor[near]
    # "Not near" is both far and never: NaN means no contraction preceded this anchor at all,
    # which is as much a control as one an hour past its contraction.
    far_frame = per_anchor[~near]

    rng = np.random.default_rng(int(seed))
    picked: List[pd.DataFrame] = []
    counts = events_frame.groupby("guid").size() if len(events_frame) else pd.Series(dtype=int)
    for guid, cell in far_frame.groupby("guid", sort=True):
        wanted = int(counts.get(guid, 0))
        if wanted <= 0:
            continue
        take = min(wanted, len(cell))
        chosen = rng.choice(len(cell), size=take, replace=False)
        picked.append(cell.iloc[np.sort(chosen)])
    controls = pd.concat(picked, ignore_index=True) if picked else far_frame.head(0)
    return pd.concat(
        [events_frame.assign(condition="event"), controls.assign(condition="control")],
        ignore_index=True,
    )


def conditioned_rows(
    split: pd.DataFrame, labels_by_guid: pd.Series, *, resamples: int, seed: int
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Compare each readout between event and control anchors, per recording and per class.

    Args:
        split: The event/control anchors.
        labels_by_guid: Each recording's clinical class, for the per-class cut.
        resamples: Bootstrap resamples over recordings.
        seed: Bootstrap seed.

    Returns:
        ``(rows, per_recording)``. Each row carries the paired difference with its interval and a
        signed-rank test; the frame carries one row per recording per readout, which is what the
        grouped fan-out and the figure both read.
    """
    rows: List[Dict[str, Any]] = []
    pieces: List[pd.DataFrame] = []
    for name, column in CONDITIONED_READOUTS:
        if column not in split.columns or not len(split):
            continue
        # Per recording *within* each condition first: a recording contributing forty event
        # anchors and one contributing four then count the same, which is the unit every other
        # comparison in this pipeline is made on.
        pivot = split.groupby(["guid", "condition"])[column].mean().unstack("condition")
        for side in ("event", "control"):
            if side not in pivot.columns:
                pivot[side] = np.nan
        cut = pd.DataFrame(
            {
                "guid": [str(value) for value in pivot.index],
                "metric": name,
                labels.CLASS_COLUMN: [labels_by_guid.get(value) for value in pivot.index],
                "event": np.asarray(pivot["event"], dtype=np.float64),
                "control": np.asarray(pivot["control"], dtype=np.float64),
            }
        )
        cut["difference"] = cut["event"] - cut["control"]
        pieces.append(cut)
        rows.append(_conditioned_row(name, column, cut, resamples=resamples, seed=seed))
        # Named ``class_name`` rather than ``cohort``, which is the module holding the canonical
        # order two lines below. The per-class rows follow that order rather than the alphabetical
        # one ``groupby`` produces, so the CSV reads HIE, acidosis, healthy like every figure.
        by_class = {
            str(class_name): cell
            for class_name, cell in cut.groupby(labels.CLASS_COLUMN, dropna=True)
        }
        for class_name in cohort.ordered_groups(list(by_class), labels.CLASS_COLUMN):
            rows.append(
                _conditioned_row(
                    name, column, by_class[class_name],
                    resamples=resamples, seed=seed, cohort=class_name,
                )
            )
    per_recording = (
        pd.concat(pieces, ignore_index=True) if pieces
        else pd.DataFrame(columns=["guid", "metric", labels.CLASS_COLUMN, "event", "control",
                                   "difference"])
    )
    return rows, per_recording


def _conditioned_row(
    name: str,
    column: str,
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    cohort: str = "pooled",
) -> Dict[str, Any]:
    """Summarise one readout's event-minus-control difference over recordings."""
    differences = finite_column(frame, "difference")
    interval = shared_stats.bootstrap_ci(differences, resamples=resamples, seed=seed)
    paired = shared_stats.wilcoxon_paired(
        finite_column(frame, "event"),
        finite_column(frame, "control"),
        label_left="anchors near a contraction",
        label_right="count-matched control anchors",
    )
    return {
        "metric": name,
        "source_column": column,
        "cohort": cohort,
        **{key: value for key, value in describe(differences).items() if key != "metric"},
        "event_mean": float(np.nanmean(finite_column(frame, "event")))
        if len(frame) else float("nan"),
        "control_mean": float(np.nanmean(finite_column(frame, "control")))
        if len(frame) else float("nan"),
        "ci_lo": interval["lo"],
        "ci_hi": interval["hi"],
        "wilcoxon_p_value": paired["p_value"],
        "wilcoxon_n_pairs": paired["n_pairs"],
    }


# =============================================================================
# Figures
# =============================================================================
def build_deceleration_figure(skill: pd.DataFrame) -> Any:
    """Draw hit rate, false-alarm rate and lead-time error against lead time in seconds."""
    figure, axes = figures.new_figure(3)
    panels = (
        ("hit_rate", "Deceleration hit rate by lead time", "fraction of true events found"),
        ("false_alarm_rate", "False-alarm rate by lead time", "fraction of forecast events unmatched"),
        ("lead_time_abs_error_s", "Timing error of matched events", "seconds"),
    )
    for index, (column, title, ylabel) in enumerate(panels):
        axis = axes[index, 0]
        if not len(skill) or column not in skill.columns:
            figures.multi_line_panel(axis, np.zeros(0), np.zeros((0, 0)), [], title=title,
                                     xlabel="lead time (s)", ylabel=ylabel)
            continue
        ordered = skill.sort_values("lead_time_s")
        lead = np.asarray(ordered[ordered["branch"] == BRANCHES[0]]["lead_time_s"],
                          dtype=np.float64)
        series = np.vstack([
            np.asarray(ordered[ordered["branch"] == branch][column], dtype=np.float64)
            for branch in BRANCHES
        ])
        figures.multi_line_panel(axis, lead, series, list(BRANCHES), title=title,
                                 xlabel="lead time (s)", ylabel=ylabel)
    return figure


def build_triggered_figure(record: Dict[str, Any]) -> Any:
    """Draw each triggered-average curve over its null band.

    The band is the null's mean $\\pm$ :data:`NULL_BAND_SIGMAS` standard deviations, so it
    brackets the null mean at every point by construction -- which is what makes "the observed
    curve leaves the band" a statement rather than an impression.
    """
    curves = list((record.get("curves") or {}).items())
    figure, axes = figures.new_figure(max(len(curves), 1), height_per_row=2.4)
    for index, (name, block) in enumerate(curves):
        axis = axes[index, 0]
        mean = np.asarray(block["mean"], dtype=np.float64)
        seconds = np.arange(mean.size, dtype=np.float64) / events.FS_RAW
        low = np.asarray(block["null_lo"], dtype=np.float64)
        high = np.asarray(block["null_hi"], dtype=np.float64)
        axis.fill_between(seconds, low, high, color=figures.COLOR_GRAY, alpha=0.3, linewidth=0,
                          label=f"random-trigger null ({NULL_BAND_SIGMAS:g} sd)")
        axis.plot(seconds, np.asarray(block["null_mean"], dtype=np.float64),
                  color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN, linestyle="--", label="null mean")
        axis.plot(seconds, mean, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_REGULAR, label=name)
        axis.axvspan(*RESPONSE_WINDOW_S, color=figures.COLOR_ORANGE, alpha=0.12, zorder=0)
        axis.set_title(f"Contraction-triggered {name}, n = {int(record.get('n_triggers') or 0)}")
        axis.set_xlabel("time after contraction onset (s)")
        axis.set_ylabel("bpm" if name != "difference" else "bpm (full - base)")
        axis.legend(fontsize=figures.FONT_LABEL, loc="best")
        figures.style_axes(axis)
    if not curves:
        axes[0, 0].text(0.5, 0.5, figures.EMPTY_NOTE, transform=axes[0, 0].transAxes,
                        ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY)
        figures.style_axes(axes[0, 0])
    return figure


def build_conditioned_figure(per_recording: pd.DataFrame) -> Any:
    """Draw the event-versus-control comparison per readout, split by clinical class.

    The classes run HIE, acidosis, healthy and are coloured red, amber, green -- the same order
    and palette every other cohort figure in this evaluation uses, so two of them can be read
    side by side. ``groupby`` alone would order them alphabetically, putting acidosis first.
    """
    metrics = [name for name, _ in CONDITIONED_READOUTS]
    figure, axes = figures.new_figure(len(metrics))
    for index, metric in enumerate(metrics):
        axis = axes[index, 0]
        cut = per_recording[per_recording["metric"] == metric] if len(per_recording) else (
            per_recording
        )
        values: Dict[str, np.ndarray] = {}
        if len(cut):
            # ``dropna=False`` keeps the unlabelled recordings as their own violin rather than
            # dropping them silently; the canonical order does not know that label, so it lands
            # after the three classes it does know.
            by_class = {
                f"{name}": finite_column(cell, "difference")
                for name, cell in cut.groupby(labels.CLASS_COLUMN, dropna=False)
            }
            values = {
                name: by_class[name]
                for name in cohort.ordered_groups(list(by_class), labels.CLASS_COLUMN)
            }
        figures.violin_panel(
            axis, values or {"all": np.zeros(0)},
            title=f"{metric}: anchors near a contraction minus control anchors",
            ylabel="difference (nats per anchor)",
            colors=figures.group_colors(list(values)),
            reference=0.0, reference_label="no conditioning effect",
        )
    return figure


# =============================================================================
# The registry entry point
# =============================================================================
def run_events_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score deceleration forecasting, the contraction-triggered response, and both conditioned.

    Args:
        context: The analysis context, read for the retained waveforms and both tables.
        eval_config: The validated block, for the event window, the bootstrap settings and the
            seed the null and the control draw follow from.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the three readouts, each carrying its own skip reason when its
        inputs were not retained or its guards did not clear.
    """
    collection = context.collection
    record = dict(getattr(collection, "record", None) or {})
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    window_s = float(eval_config.get("event_lag_window_s", 120.0))

    waveforms = _waveform_readouts(collection, record, resamples=resamples, seed=seed)
    conditioned = _conditioned_readout(
        collection, directory, window_s=window_s, resamples=resamples, seed=seed
    )

    pd.DataFrame(waveforms["skill"]).to_csv(directory / DECELERATION_FILENAME, index=False)
    pd.DataFrame(conditioned["rows"]).to_csv(directory / CONDITIONED_FILENAME, index=False)
    written = [DECELERATION_FILENAME, CONDITIONED_FILENAME]
    triggered = waveforms["triggered"]
    if triggered.get("curves"):
        pd.DataFrame(
            {name: block["mean"] for name, block in triggered["curves"].items()}
        ).to_csv(directory / TRIGGERED_FILENAME, index=False)
        written.append(TRIGGERED_FILENAME)

    figure_names = [
        str(figures.render_figure(
            build_deceleration_figure(pd.DataFrame(waveforms["skill"])),
            directory / DECELERATION_FIGURE,
        ).name),
        str(figures.render_figure(
            build_triggered_figure(triggered), directory / TRIGGERED_FIGURE
        ).name),
        str(figures.render_figure(
            build_conditioned_figure(conditioned["per_recording"]), directory / CONDITIONED_FIGURE
        ).name),
    ]

    result: Dict[str, Any] = {
        "n_samples": waveforms["n_samples"],
        "composition": {
            "n_waveform_samples": waveforms["n_samples"] or 0,
            "n_event_anchors": conditioned["n_event_anchors"],
            "n_recordings": conditioned["n_recordings"],
        },
        "plan": {
            "capped": True,
            "cap_source": "eval_config.caps.waveforms",
            "bootstrap_resamples": resamples,
            "seed": seed,
            "event_lag_window_s": window_s,
        },
        "deceleration": waveforms["deceleration"],
        "triggered": {key: value for key, value in triggered.items() if key != "curves"},
        "conditioned": conditioned["record"],
        "files": written + figure_names,
    }
    if len(conditioned["per_recording"]):
        result["grouped_frames"] = [
            grouped_frame_entry(
                ANALYSIS_DIRNAME, CONDITIONED_PER_RECORDING_FILENAME, ("difference",)
            )
        ]
    return result


def _waveform_readouts(
    collection: Any, record: Dict[str, Any], *, resamples: int, seed: int
) -> Dict[str, Any]:
    """Run the two readouts that need retained forecast blocks, or record why they did not.

    Args:
        collection: What the shared pass produced.
        record: Its provenance record, for the geometry and the FHR statistics.
        resamples: Bootstrap resamples.
        seed: Seed for the null triggers.

    Returns:
        ``{'skill': rows, 'deceleration': record, 'triggered': record, 'n_samples': int|None}``.
    """
    empty = {
        "skill": [],
        "triggered": {"skipped": True},
        "deceleration": {},
        "n_samples": None,
    }
    if not _has_waveforms(collection):
        reason = (
            "no forecast waveform was retained, so there is nothing to run a detector on. Set "
            "eval_config.caps.waveforms to retain some; retention is opt-in because the blocks "
            "are about two megabytes per sample."
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: waveform readouts skipped -- {reason}")
        empty["deceleration"] = {"skipped": True, "reason": reason}
        empty["triggered"] = {"skipped": True, "reason": reason}
        return empty

    geometry = dict(record.get("geometry") or {})
    normalization = dict(record.get("normalization") or {})
    decimation = int(geometry.get("decimation", 16))
    raw_per_step = int(geometry.get("raw_per_step", decimation))
    raw_len = int(geometry.get("raw_len", 0))
    retained = dict(collection.retained)
    rows = retained_frame(collection)
    block_len = int(np.asarray(retained["target"][0]).reshape(
        np.asarray(retained["target"][0]).shape[0], -1
    ).shape[1])
    usable_tau = events.usable_horizon_steps(block_len, raw_per_step=raw_per_step)
    unit = to_bpm(np.zeros(1), normalization)[1]

    detections: List[Dict[str, Any]] = []
    triggers: List[Dict[str, Any]] = []
    n_detector_calls = 0
    for _, row in rows.iterrows():
        index = int(row["row"])
        validity = _block_validity(
            retained["weight"], index, decimation=decimation, raw_len=raw_len
        )
        blocks = {
            "truth": to_bpm(_blocks(retained["target"], index), normalization)[0],
            "base": to_bpm(_blocks(retained["mu_base"], index), normalization)[0],
            "full": to_bpm(_blocks(retained["mu_full"], index), normalization)[0],
        }
        blocks["difference"] = blocks["full"] - blocks["base"]
        entry: Dict[str, Any] = {"guid": str(row["guid"]), "truth": {}, "base": {}, "full": {}}
        if usable_tau.size:
            for name in ("truth", *BRANCHES):
                entry[name] = deceleration_detections(
                    blocks[name], validity, decimation=decimation,
                    raw_per_step=raw_per_step, usable_tau=usable_tau,
                )
                n_detector_calls += int(blocks[name].shape[0])
            detections.append(entry)
        triggers.append(
            _trigger_entry(retained, index, row, blocks, validity, decimation=decimation)
        )

    skill = skill_rows(
        detections, usable_tau=usable_tau, raw_per_step=raw_per_step,
        resamples=resamples, seed=seed,
    ) if detections else []
    usable_lo, usable_hi = events.usable_interval(block_len)
    per_anchor_count = sum(
        len(indices) for entry in detections for indices in entry["truth"].values()
    )
    return {
        "skill": skill,
        "deceleration": {
            "unit": unit,
            "block_samples": block_len,
            "usable_interval_samples": [int(usable_lo), int(usable_hi)],
            "usable_horizon_steps": [int(value) for value in usable_tau.tolist()],
            "edge_seconds": float(events.DECELERATION_EDGE_S),
            "prominence_bpm": float(events.DECELERATION_PROMINENCE_BPM),
            "match_tolerance_s": float(MATCH_TOLERANCE_S),
            # The de-duplication rule, stated as the arithmetic it is: one anchor per (event,
            # horizon step), so the per-event count is the per-anchor count divided by the number
            # of usable steps rather than by the horizon.
            "deduplication": (
                "rates are computed per (event, horizon step); for a fixed step exactly one "
                "anchor places a given raw sample there, so each physiological event is counted "
                "once per step rather than once per anchor"
            ),
            "n_true_detections_per_anchor": int(per_anchor_count),
            "n_true_events_per_step": int(
                per_anchor_count / max(len(usable_tau), 1)
            ),
            "pseudo_replication_factor": int(len(usable_tau)),
            "n_detector_calls": int(n_detector_calls),
        },
        "triggered": _triggered_or_skip(triggers, seed=seed),
        "n_samples": int(len(rows)),
    }


def _trigger_entry(
    retained: Dict[str, Any],
    index: int,
    row: Any,
    blocks: Dict[str, np.ndarray],
    validity: np.ndarray,
    *,
    decimation: int,
) -> Dict[str, Any]:
    """Assemble one retained sample's contraction triggers and the curves they index into."""
    up_raw = np.asarray(retained["up_raw"][index], dtype=np.float64).reshape(-1)
    onsets = events.detect_contractions(up_raw[: validity.size], valid=validity)["onset_raw"]
    # The truth trace, reassembled from the blocks themselves: horizon step 0 of anchor t is raw
    # samples [D(t+1), D(t+2)), so those steps laid end to end are the contiguous trace starting
    # at raw index D. Taken from the already-converted truth blocks rather than from the retained
    # tensor, so the baseline a snippet is corrected by is in the same units as the snippet.
    truth_trace = blocks["truth"][:, :decimation].reshape(-1)
    pre = int(round(TRIGGER_BASELINE_S * events.FS_RAW))
    # A trigger needs a full baseline behind it, and an anchor whose block is carried. Anchor $t$'s
    # block is held whole in `blocks`, so the last trigger is the last anchor's own block start:
    # subtracting the block width instead would discard real contractions in the final horizon
    # whose blocks are entirely in range.
    first_trigger = int(decimation + pre)
    last_trigger = int(decimation * blocks["truth"].shape[0])
    detected = np.asarray(onsets, dtype=np.int64)
    usable = detected[(detected >= first_trigger) & (detected <= last_trigger)]
    return {
        "guid": str(row["guid"]),
        "blocks": blocks,
        "truth_trace": np.asarray(truth_trace, dtype=np.float64),
        "decimation": int(decimation),
        # Only the onsets that contribute a snippet. The null draws its triggers *inside* this
        # range, so every drawn trigger contributes; count-matching against the detected onsets
        # instead would average the null over more triggers than the observation, tightening its
        # band by roughly $\sqrt{N_{\mathrm{null}}/N_{\mathrm{obs}}}$ and inflating `dip_z` in the
        # one statistic that decides whether a dip is a response.
        "onsets": usable,
        "n_onsets_detected": int(detected.size),
        "first_trigger": first_trigger,
        "last_trigger": last_trigger,
    }


def _triggered_or_skip(triggers: Sequence[Dict[str, Any]], *, seed: int) -> Dict[str, Any]:
    """Run the triggered average when the guards clear, and record a skip when they do not."""
    # Counted over the triggers that contribute, not the ones detected: a guard that certifies
    # more triggers than the average is built from is not guarding the average.
    n_triggers = sum(int(len(entry["onsets"])) for entry in triggers)
    n_detected = sum(int(entry.get("n_onsets_detected", len(entry["onsets"]))) for entry in triggers)
    n_recordings = len({entry["guid"] for entry in triggers if len(entry["onsets"])})
    if n_triggers < MIN_TRIGGERS or n_recordings < MIN_EVENT_RECORDINGS:
        reason = (
            f"{n_triggers} usable contraction(s) of {n_detected} detected over {n_recordings} "
            f"recording(s); the triggered average needs at least {MIN_TRIGGERS} over "
            f"{MIN_EVENT_RECORDINGS}, below which the curve is a description of whichever "
            f"recordings happened to contribute"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: triggered average skipped -- {reason}")
        return {"skipped": True, "reason": reason, "n_triggers": n_triggers,
                "n_onsets_detected": n_detected, "n_recordings": n_recordings}
    record = triggered_average(triggers, rng=np.random.default_rng(seed))
    # The honest denominator beside the count that was used: a trigger too close to either end of
    # the recording to carry a baseline or an anchor is dropped, and the run says how many.
    record["n_onsets_detected"] = n_detected
    return record


def _conditioned_readout(
    collection: Any, directory: Path, *, window_s: float, resamples: int, seed: int
) -> Dict[str, Any]:
    """Restrict both coupling readouts to anchors near a contraction, or record why not."""
    per_anchor = getattr(collection, "per_anchor", None)
    per_sample = getattr(collection, "per_sample", None)
    blank = pd.DataFrame(columns=["guid", "metric", labels.CLASS_COLUMN, "event", "control",
                                  "difference"])
    if per_anchor is None or per_anchor.empty or CONTRACTION_AGE_COLUMN not in per_anchor.columns:
        reason = (
            "the per-anchor table carries no contraction timing, so there are no anchors to "
            "condition on"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: conditioned coupling skipped -- {reason}")
        return {"rows": [], "per_recording": blank, "n_event_anchors": 0, "n_recordings": 0,
                "record": {"skipped": True, "reason": reason}}

    split = conditioned_anchors(per_anchor, window_s=window_s, seed=seed)
    events_only = split[split["condition"] == "event"] if len(split) else split
    n_event_anchors = int(len(events_only))
    n_recordings = int(events_only["guid"].nunique()) if n_event_anchors else 0
    if n_event_anchors < MIN_EVENT_ANCHORS or n_recordings < MIN_EVENT_RECORDINGS:
        reason = (
            f"{n_event_anchors} anchor(s) within {window_s:g} s of a contraction over "
            f"{n_recordings} recording(s); the comparison needs at least {MIN_EVENT_ANCHORS} over "
            f"{MIN_EVENT_RECORDINGS}"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: conditioned coupling skipped -- {reason}")
        return {"rows": [], "per_recording": blank, "n_event_anchors": n_event_anchors,
                "n_recordings": n_recordings,
                "record": {"skipped": True, "reason": reason}}

    labels_by_guid = (
        per_sample.groupby("guid")[labels.CLASS_COLUMN].first()
        if per_sample is not None and labels.CLASS_COLUMN in per_sample.columns
        else pd.Series(dtype=object)
    )
    rows, per_recording = conditioned_rows(
        split, labels_by_guid, resamples=resamples, seed=seed
    )
    per_recording.to_csv(directory / CONDITIONED_PER_RECORDING_FILENAME, index=False)
    return {
        "rows": rows,
        "per_recording": per_recording,
        "n_event_anchors": n_event_anchors,
        "n_recordings": n_recordings,
        "record": {
            "window_s": float(window_s),
            "n_event_anchors": n_event_anchors,
            "n_control_anchors": int((split["condition"] == "control").sum()),
            "n_recordings": n_recordings,
            # What "within the window of a contraction" is measured from -- see the triggered
            # record for why the convention has to travel with the number.
            "onset_convention": events.ONSET_WALK_BACK_NOTE,
            "rows": rows,
        },
    }
