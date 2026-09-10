r"""The decision rules, declared before any fit, and the rates they aggregate to.

An instrument is only an instrument if what counts as a detection was fixed before the data was
seen. This module holds those rules and nothing else: the campaign imports them, applies them, and
records the settings that produced every verdict beside the verdict itself, so a reader can tell a
rule that was declared from one that was chosen afterwards.

**Two rules, and they answer different questions.**

*Relevance* asks whether the fitted model's source-conditioned branch is a better predictive density
than its own target-only branch, over the scored anchors, at a recording-level interval. It is the
rule every generator is scored under, including the ones declared to carry no source information --
where firing is a false positive rather than a success.

*Recovery* asks whether the lag readout puts its largest suppression margin where the plant
actually is. It applies only where a generator plants a dependence at declared source times and
where those times are readable at all inside the searched window; the campaign refuses to score it
otherwise rather than recording a failure that is a property of the geometry.

**Why a one-sided interval and not a threshold on the point estimate.** The point estimate of a gap
is positive about half the time under no effect, so a rule reading it alone would report a
false-positive rate near one half and say nothing. Requiring the lower interval end to clear zero is
what makes the false-positive rate a measurement of the readout rather than of the sign of noise.

**Why the lag axis is partitioned into windows rather than read lag by lag.** A per-lag suppression
would score one arm per lag, which at a production lag window is a hundred extra forwards per batch
for a resolution the representation does not have: two lags closer together than the source
encoder's own reach are summaries of overlapping windows. The window width is therefore a declared
setting, recorded with every verdict, and a campaign that changed it changed its instrument.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from teb_vae.lag_attn.eval.stats import bootstrap_ci


@dataclass(frozen=True)
class Criteria:
    r"""Everything that decides a verdict, fixed before the campaign runs.

    Attributes:
        confidence: Coverage of every interval, in $(0, 1)$.
        resamples: Bootstrap resamples behind each interval, drawn over segments.
        window_width: Lags per window in the declared partition of the lag axis.
        seed: Seed for the resampling, so an interval is reproducible from the record alone.
        num_mc_samples: Monte Carlo draws $K$ every branch is scored at.
    """

    confidence: float = 0.9
    resamples: int = 400
    window_width: int = 4
    seed: int = 20260909
    num_mc_samples: int = 8

    def as_record(self) -> Dict[str, Any]:
        """The settings, for the campaign's own artifact.

        Returns:
            The declared settings as plain values.
        """
        return {
            "confidence": float(self.confidence),
            "resamples": int(self.resamples),
            "window_width": int(self.window_width),
            "seed": int(self.seed),
            "num_mc_samples": int(self.num_mc_samples),
            "relevance_rule": (
                "the lower end of the interval on the per-segment predictive gap clears zero"
            ),
            "recovery_rule": (
                "the window with the largest point margin intersects the declared direct support, "
                "and that window's own interval clears zero"
            ),
        }


def lag_windows(n_lags: int, width: int) -> Dict[str, Tuple[int, int]]:
    r"""Partition the lag axis into contiguous inclusive windows of a declared width.

    The last window absorbs the remainder rather than being dropped, because a lag left out of the
    partition is a lag no suppression arm ever removes -- and the readout would then be blind to a
    plant sitting in it without anything saying so.

    Args:
        n_lags: $L$, the candidate lag count.
        width: Lags per window.

    Returns:
        ``{name: (lo, hi)}``, inclusive, in ascending order.

    Raises:
        ValueError: If either argument is not positive, or if the width exceeds the lag axis --
            a single window over the whole axis is a suppression of everything, which is the
            silence arm under another name.
    """
    if int(n_lags) < 1 or int(width) < 1:
        raise ValueError(f"n_lags and width must be >= 1, got {n_lags} and {width}")
    if int(width) >= int(n_lags):
        raise ValueError(
            f"a window width of {width} over {n_lags} lags gives one window covering the whole "
            f"axis, whose suppression is the silence arm rather than a lag readout."
        )
    windows: Dict[str, Tuple[int, int]] = {}
    for low in range(0, int(n_lags), int(width)):
        high = min(low + int(width), int(n_lags)) - 1
        if high >= int(n_lags) - int(width):
            high = int(n_lags) - 1
        windows[f"{low:03d}_{high:03d}"] = (low, high)
        if high == int(n_lags) - 1:
            break
    return windows


def interval(values: Sequence[float], criteria: Criteria) -> Dict[str, Any]:
    """One recording-level interval, at the declared settings.

    Args:
        values: One value per segment. Each segment is one recording here, which is what the
            generators build and what makes the resample unit the right one.
        criteria: The declared settings.

    Returns:
        The point estimate, the interval and the honest count.
    """
    return bootstrap_ci(
        values,
        confidence=float(criteria.confidence),
        resamples=int(criteria.resamples),
        seed=int(criteria.seed),
    )


def _clears_zero(block: Mapping[str, Any]) -> bool:
    """Whether an interval's lower end is a finite number above zero.

    The shared bootstrap reports a non-finite bound rather than raising when it has too few finite
    values to estimate a spread, and that state must read as "no detection" rather than propagate:
    an interval nobody could compute does not support an effect.

    Args:
        block: What :func:`interval` returned.

    Returns:
        ``True`` when the interval supports a positive effect.
    """
    low = block.get("lo")
    return low is not None and float(low) == float(low) and float(low) > 0.0


def relevance_verdict(
    per_segment_gap: Sequence[float], criteria: Criteria
) -> Dict[str, Any]:
    """Did this fit report the source as helping, at the declared interval?

    Args:
        per_segment_gap: The per-segment predictive gap, base minus full in nats per anchor, so a
            positive value means the source-conditioned branch scored better.
        criteria: The declared settings.

    Returns:
        The verdict and the interval it was reached from.
    """
    block = interval(per_segment_gap, criteria)
    return {"detected": _clears_zero(block), "gap": block}


def recovery_verdict(
    window_margins: Mapping[str, Sequence[float]],
    windows: Mapping[str, Tuple[int, int]],
    support: Optional[Tuple[int, int]],
    criteria: Criteria,
    *,
    graded: bool = True,
) -> Dict[str, Any]:
    """Did the lag readout put its largest suppression margin where the plant is?

    A margin is the suppressed arm's predictive score less the matched one's, so a **positive**
    margin means the fitted model predicts worse without those lags. The largest is the window the
    readout says it relies on most.

    Args:
        window_margins: ``{window: per-segment margins}``, one entry per declared window.
        windows: The declared partition, so a verdict names the lags it is about.
        support: The generator's declared direct support, or ``None`` where it plants none.
        criteria: The declared settings.
        graded: Whether the band is a criterion this readout may be failed against. ``False`` on
            the instrument whose plant reaches the stored grid through the real feature operator:
            the peak, the band and the distance between them are all still reported -- that
            distance is the delay spread and is the whole measurement -- but no pass or fail is
            recorded, because the operator's own spread is not something the readout could undo.

    Returns:
        The verdict, the peak window, its interval, and every window's point margin so a reader
        sees the profile rather than only the winner. ``scored`` is ``False`` where the generator
        declared no support, where the band is not graded, or where no window carries a finite
        margin.
    """
    profile = {name: interval(values, criteria) for name, values in window_margins.items()}
    finite = {
        name: block
        for name, block in profile.items()
        if block.get("point") is not None and float(block["point"]) == float(block["point"])
    }
    if support is None or not finite:
        return {
            "scored": False,
            "recovered": False,
            "peak_window": None,
            "support": None if support is None else [int(support[0]), int(support[1])],
            "profile": profile,
        }

    peak = max(finite, key=lambda name: float(finite[name]["point"]))
    low, high = windows[peak]
    inside = not (high < int(support[0]) or low > int(support[1]))
    return {
        "scored": bool(graded),
        "recovered": bool(graded and inside and _clears_zero(profile[peak])),
        "peak_window": {"name": peak, "lags": [int(low), int(high)]},
        "support": [int(support[0]), int(support[1])],
        # How far the peak sits from the plant, in lags, and zero when it overlaps. On the
        # instrument whose plant reaches the stored grid through the feature operator this is the
        # measurement rather than a miss: the operator spreads a raw instant across many stored
        # steps, and the distance is that spread.
        "peak_distance_lags": 0
        if inside
        else int(min(abs(low - int(support[1])), abs(int(support[0]) - high))),
        "profile": profile,
    }


def campaign_rates(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate every run into the two rates the campaign exists to report.

    **Power** is measured on the generators declared to carry source information and **the
    false-positive rate** on those declared not to. They are reported separately per generator and
    never pooled into one number: a generator whose source is a deterministic function of target
    history and one whose source is independent noise fail differently, and an average over them
    would hide which.

    A run that did not complete is counted in the denominator, not dropped. A campaign that
    silently excluded its failures would report a rate over the runs that worked.

    Args:
        records: One record per (generator, seed) run.

    Returns:
        Per-generator rates and the two pooled summaries, each with its own denominator.
    """
    by_generator: Dict[str, list] = defaultdict(list)
    for record in records:
        by_generator[str(record["generator"])].append(record)

    per_generator: Dict[str, Any] = {}
    for name, runs in sorted(by_generator.items()):
        informative = bool(runs[0]["truth"]["source_informative"])
        detected = sum(1 for run in runs if bool(run.get("relevance", {}).get("detected")))
        scored = [run for run in runs if bool(run.get("recovery", {}).get("scored"))]
        recovered = sum(1 for run in scored if bool(run["recovery"]["recovered"]))
        per_generator[name] = {
            "runs": len(runs),
            "source_informative": informative,
            "detections": detected,
            "detection_rate": detected / len(runs),
            # Named for what it is on this generator, so a reader never has to remember which
            # column a control's number belongs in.
            "rate_is": "power" if informative else "false_positive_rate",
            "recovery_scored": len(scored),
            "recoveries": recovered,
            "recovery_rate": (recovered / len(scored)) if scored else None,
            "causal": bool(runs[0]["truth"]["causal"]),
            "note": str(runs[0]["truth"]["note"]),
        }

    informative = [row for row in per_generator.values() if row["source_informative"]]
    controls = [row for row in per_generator.values() if not row["source_informative"]]
    return {
        "per_generator": per_generator,
        "power": _pooled(informative),
        "false_positive_rate": _pooled(controls),
        "note": (
            "Power is measured on the generators that carry source information and the "
            "false-positive rate on those that do not; the two never share a denominator. A rate "
            "here is a rate for this campaign's reduced fit, at these settings, and is not a "
            "statement about a production run's sensitivity."
        ),
    }


def _pooled(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool one family of generators into a single rate with its own denominator.

    Args:
        rows: The per-generator rows of one family.

    Returns:
        The pooled rate, or an empty denominator when the family has no generator.
    """
    runs = sum(int(row["runs"]) for row in rows)
    detections = sum(int(row["detections"]) for row in rows)
    return {
        "generators": len(rows),
        "runs": runs,
        "detections": detections,
        "rate": (detections / runs) if runs else None,
    }


__all__ = [
    "Criteria",
    "campaign_rates",
    "interval",
    "lag_windows",
    "recovery_verdict",
    "relevance_verdict",
]
