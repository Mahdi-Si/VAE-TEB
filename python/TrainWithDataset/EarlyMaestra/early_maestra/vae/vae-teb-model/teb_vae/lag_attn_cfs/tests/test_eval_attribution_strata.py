r"""The late-window strata pass: its anchor rules, its matched band, its onset alignment, its tests.

Model-free and on planted tables, so each check fails only when the logic it names breaks:
the per-segment KL tiers, the rule that a segment without a contraction gives no "between
contractions" anchors, the matched band choosing the narrowest best-covered band in nats, the
onset re-indexing $x = a - 4\ell$, and the paired test finding a planted within-recording shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import attribution_strata as strata
from teb_vae.lag_attn_cfs.eval._reuse import labels


def _late(kl, *, guid="r0", epoch=-3600.0, age=None, cls="healthy") -> pd.DataFrame:
    """One late segment's anchors, labelled, as :func:`late_anchors` and :func:`attach_class` leave them."""
    n = len(kl)
    frame = pd.DataFrame({
        "guid": [guid] * n, "epoch": [epoch] * n, "anchor": np.arange(n) + 134, "kld_per_t": kl, "coverage": 1.0,
        strata.CONTRACTION_AGE_COLUMN: np.full(n, np.nan) if age is None else age,
    })
    late = strata.late_anchors(frame)
    late[labels.CLASS_COLUMN] = cls
    return late


def test_the_kl_tiers_are_cut_within_each_segment() -> None:
    late = pd.concat([_late(np.arange(10.0), guid="a"), _late(100.0 + np.arange(10.0), guid="b")], ignore_index=True)
    tier = strata.kl_tier(late)
    # Each segment has its own top and bottom, whatever the other segment's level.
    for guid in ("a", "b"):
        mine = tier[(late["guid"] == guid).to_numpy()]
        assert (mine[:3] == strata.KL_LOW).all() and (mine[-3:] == strata.KL_HIGH).all()


def test_a_segment_without_a_contraction_gives_no_quiet_anchors() -> None:
    flat = _late(np.ones(6), guid="flat")
    active = _late(np.ones(6), guid="active", age=np.array([10.0, 50.0, 130.0, 200.0, np.nan, 5.0]))
    late = pd.concat([flat, active], ignore_index=True)
    phase = strata.ua_phase(late, window_s=120.0)
    assert (phase[:6] == "").all()
    assert list(phase[6:]) == [strata.UA_CONTRACTION, strata.UA_CONTRACTION, strata.UA_QUIET,
                               strata.UA_QUIET, strata.UA_QUIET, strata.UA_CONTRACTION]


def test_the_matched_band_is_the_narrowest_band_every_class_reaches() -> None:
    # Every class reaches every band, so the tie goes to the narrowest band in nats: the bottom one,
    # since the planted K_t is skewed with a long upper tail.
    values = np.concatenate([np.linspace(0.0, 1.0, 40), np.linspace(1.0, 50.0, 10)])
    late = pd.concat([_late(values, guid=f"{cls}{r}", cls=cls) for cls in ("hie", "healthy") for r in range(3)], ignore_index=True)
    low, high, record = strata.matched_band(late, np.ones(len(late), bool))
    widths = np.diff(record["edges_nats"])
    assert record["chosen_band"] == int(np.argmin(widths)) and np.isclose(high - low, widths.min())


def test_onset_alignment_puts_the_step_at_the_onset_at_zero() -> None:
    # Row i carries its mass at lag i and has its onset i steps back: the mass lands at x = 0.
    aligned, x = strata.onset_locked(np.eye(4), np.array([0.0, 4.0, 8.0, np.nan]))
    zero = int(np.flatnonzero(x == 0.0)[0])
    assert (aligned[:3, zero] == 1.0).all() and np.isnan(aligned[3]).all()
    # The anchor's own step (lag 0) sits a seconds after the onset.
    assert aligned[2, int(np.flatnonzero(x == 8.0)[0])] == 0.0


def test_the_paired_test_finds_a_planted_within_recording_shift() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for r in range(12):
        level = rng.normal(0.0, 5.0)                         # a large between-recording offset
        for stratum, shift in ((strata.KL_HIGH, 1.0), (strata.KL_LOW, 0.0)):
            rows.append({"guid": f"g{r}", "stratum": stratum, labels.CLASS_COLUMN: "healthy",
                         "anchor:gain": level + shift + rng.normal(0.0, 0.1), "n_anchors": 4})
    stats = strata.paired_tests(pd.DataFrame(rows), seed=0)
    hit = stats[(stats["contrast"] == "kl") & (stats["metric"] == "anchor:gain") & (stats["class"] == "healthy")].iloc[0]
    assert hit["n_pairs"] == 12 and np.isclose(hit["mean_diff"], 1.0, atol=0.1) and hit["significant"]
