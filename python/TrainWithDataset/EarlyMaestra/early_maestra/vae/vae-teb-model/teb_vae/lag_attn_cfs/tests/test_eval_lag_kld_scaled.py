r"""The KLD-scaled, band-restricted and per-head reading of the lag structure.

``lag_clocks`` reduces the whole lag window through statistics that divide the magnitude out. This
analysis is the three things that leaves unanswered, and each of its decisions is a way it could be
silently wrong:

**The selection must not be estimated from the quantity it selects on.** The bands come from
``eval_config.occlusion_bands``, which is geometry-fixed and shared with the interventional
readout; the soft weight comes from the *pooled* clock-excess profile and is applied identically to
every segment. A per-segment weight would let each segment choose its own lag axis, and a
comparison across segments would then be comparing different axes rather than different segments.

**Two statistics change meaning on a restricted support.** ``near_mass`` and ``far_mass`` are
measured from the axis's own start, so on a band they re-base onto the band's start and
``far_mass`` goes identically zero on any band narrower than ``FAR_SECONDS``. They are omitted from
banded sources, and their absence is asserted rather than left to inspection.

**Nothing here is tested.** No Holm family, no significance table. Asserted, because an analysis
that quietly began emitting $p$-values would multiply the corrections a reader is holding with
nothing on the page saying so.

Following the fixture rule this suite is bound by, everything below is evidence about schema,
shape, denominators, membership and refusals -- never about the sign, magnitude or significance of
any effect.
"""
from __future__ import annotations

import types
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import lag_kld_scaled as analysis

#: A tiny lag window, and a four-band partition covering it exactly once -- the shape the shipped
#: delta has at the production window. The widths are a reading convenience; nothing asserts them.
N_LAGS = 11
N_HEADS = 2
BANDS: Dict[str, list] = {"anchor": [0, 2], "near": [3, 6], "mid": [7, 8], "far": [9, 10]}

#: Segments, at two per recording so a per-recording reduction has something to reduce.
N_SEGMENTS = 12

EVAL_CONFIG: Dict[str, Any] = {"seed": 0, "occlusion_bands": BANDS}


def _profiles(seed: int = 0):
    """Per-segment profiles with mass in the ``near`` band for half the segments, lag 0 for the rest.

    Two populations rather than one, so a band-restricted statistic has something to distinguish
    and an all-identical fixture cannot make a broken reduction look correct.
    """
    rng = np.random.default_rng(seed)
    kl = rng.uniform(0.01, 0.05, (N_SEGMENTS, N_LAGS))
    attn = rng.uniform(0.01, 0.05, (N_SEGMENTS, N_LAGS))
    for index in range(N_SEGMENTS):
        if index % 2 == 0:
            kl[index, 3:7] += 0.5
            attn[index, 3:7] += 0.3
        else:
            kl[index, 0] += 0.8
            attn[index, 0] += 0.5
    return kl, attn


def _lag_block(excess: Optional[list] = None, num_heads: int = N_HEADS) -> Dict[str, Any]:
    """The collection's lag block, carrying the geometry and the pooled clock-excess profile."""
    kl, _ = _profiles()
    matched = list(kl.mean(axis=0))
    if excess is None:
        # A clean excess concentrated in the `near` band, so the weight is not degenerate.
        excess = [0.30 if 3 <= index <= 6 else 0.002 for index in range(N_LAGS)]
    return {
        "n_lags": N_LAGS,
        "delay_steps": 0,
        "num_heads": num_heads,
        "kl_lag_profile": matched,
        "kl_lag_profile_clock_excess": list(excess),
    }


def _context(
    *,
    lag: Optional[Dict[str, Any]] = None,
    per_head: Optional[np.ndarray] = None,
    vectors: Optional[Dict[str, np.ndarray]] = None,
) -> AnalysisContext:
    """An analysis context with no task and no loader, as an offline re-run has."""
    kl, attn = _profiles()
    if per_head is None:
        # Head-major and summing to the pooled attribution, which is what the readout guarantees.
        per_head = np.concatenate([kl * 0.6, kl * 0.4], axis=1)
    resolved = vectors if vectors is not None else {
        "lag_profile_untruncated": kl,
        "attention_profile_support_corrected": attn,
        "lag_profile_per_head": per_head,
    }
    per_sample = pd.DataFrame(
        {
            "guid": [f"REC{index // 2:02d}" for index in range(N_SEGMENTS)],
            "epoch": [-(3600.0 * (1 + index % 3)) for index in range(N_SEGMENTS)],
            "clinical_class": ["hie" if index < 6 else "healthy" for index in range(N_SEGMENTS)],
            "subgroup": [
                "hie_severe" if index < 6 else "healthy_bg_no_cs" for index in range(N_SEGMENTS)
            ],
            "second_stage_onset": [-1800.0] * N_SEGMENTS,
            "source_conditioned_kl_raw": np.linspace(0.5, 1.5, N_SEGMENTS),
        }
    )
    collection = types.SimpleNamespace(
        per_sample=per_sample,
        per_anchor=pd.DataFrame(),
        record={},
        retained={},
        vectors=resolved,
        results={"lag": _lag_block() if lag is None else lag},
    )
    return AnalysisContext(collection=collection, config={})


def _run(context: AnalysisContext, tmp_path, config: Optional[Dict[str, Any]] = None):
    """Run the analysis and return ``(record, directory)``."""
    record = analysis.run_lag_kld_scaled_analysis(
        context, eval_config=config or EVAL_CONFIG, output_dir=tmp_path
    )
    return record, tmp_path / analysis.ANALYSIS_DIRNAME


# =================================================================================================
# The four families of source
# =================================================================================================
def test_every_family_of_source_is_resolved_at_the_runs_own_geometry(tmp_path) -> None:
    """Two base profiles times (full + four bands + one weighted), plus one source per head.

    Built at run time rather than declared, because two of the families are run-dimensioned: the
    bands come from the config and the heads from ``num_heads``, which is a geometry key an arm can
    change. A module-level cross product could carry neither, and that -- not table width -- is
    why this is its own analysis rather than more columns on ``lag_clocks``.
    """
    record, _ = _run(_context(), tmp_path)

    keys = set(record["features"])
    assert {"kl", "attn"} <= keys
    assert {f"kl_{name}" for name in BANDS} <= keys
    assert {f"attn_{name}" for name in BANDS} <= keys
    assert {"kl_dw", "attn_dw"} <= keys
    assert {f"kl_h{head}" for head in range(N_HEADS)} <= keys
    assert set(record["sources"]["families"]) == {"full", "band", "weighted", "head"}


def test_a_full_support_source_carries_only_the_statistics_lag_clocks_does_not(tmp_path) -> None:
    """The two that keep the scale, and nothing else.

    The twelve scale-free statistics on the full support are ``lag_clocks``' own columns. Restating
    them here would put one quantity on two pages under two names, which is the failure a shared
    vocabulary exists to prevent.
    """
    _, directory = _run(_context(), tmp_path)

    trajectory = pd.read_csv(directory / analysis.TRAJECTORY_FILENAME)
    on_full = set(trajectory.loc[trajectory["source"] == "kl", "statistic"])

    assert on_full == set(analysis.FULL_SUPPORT_STATISTICS)


def test_no_banded_source_carries_a_statistic_that_re_bases_onto_the_band(tmp_path) -> None:
    """``near_mass`` and ``far_mass`` are measured from ``seconds[0]``.

    On a band that start is the *band's*, so the two would silently answer a different question --
    and ``far_mass`` would be identically zero on any band narrower than ``FAR_SECONDS``, which is
    three of these four. Four columns of structural zeros presented as measurements is worse than
    four absent ones, so they are omitted and their absence is checked here.
    """
    _, directory = _run(_context(), tmp_path)

    trajectory = pd.read_csv(directory / analysis.TRAJECTORY_FILENAME)
    banded = trajectory[trajectory["source"].isin([f"kl_{name}" for name in BANDS])]
    assert len(banded), "the fixture produced no banded rows to check"
    assert not set(analysis.NON_RESTRICTABLE) & set(banded["statistic"])

    # ... and they ARE carried where the support is the whole window, so the omission above is a
    # restriction rule rather than the statistics having been dropped everywhere.
    weighted = trajectory[trajectory["source"] == "kl_dw"]
    assert set(analysis.NON_RESTRICTABLE) <= set(weighted["statistic"])


def test_the_per_head_sources_sum_to_the_pooled_one(tmp_path) -> None:
    r"""$\sum_m K^{(m)}\alpha^{(m)}_\ell$ is the pooled attribution, so the head sources are a
    refinement of it rather than a second quantity.

    Checked on ``total_nats``, which is the one statistic where the identity survives the
    reduction: it is a sum over lags, so the sum over heads of the per-head totals is the pooled
    total. No scale-free statistic would show this, which is part of why the nats-scale pair
    exists.
    """
    _, directory = _run(_context(), tmp_path)

    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)
    totals = per_recording[
        (per_recording["statistic"] == "total_nats")
        & (per_recording["group_column"] == "clinical_class")
        & (per_recording["clock"] == "time_to_delivery")
    ]
    keys = ["group", "guid", "time_bin"]
    pooled = totals[totals["source"] == "kl"].set_index(keys)["value"]
    heads = (
        totals[totals["source"].isin([f"kl_h{head}" for head in range(N_HEADS)])]
        .groupby(keys)["value"]
        .sum()
    )
    assert len(pooled)
    for key, value in pooled.items():
        assert heads.loc[key] == pytest.approx(value, rel=1e-6), key


# =================================================================================================
# The soft weight
# =================================================================================================
def test_the_weight_is_peak_normalised_and_drags_the_centroid_onto_the_lags_it_keeps(
    tmp_path,
) -> None:
    r"""Known answer on the weight, and its consequence on the statistic.

    The weight itself is exact -- $\omega_\ell = \Delta^+_\ell / \max_k \Delta^+_k$, so it is
    $1$ at the excess peak -- and the selection table is where a reader checks it. The centroid is
    then asserted as a consequence rather than as a second exact number: it is a mass-weighted mean
    of the *underlying* profile after weighting, so pinning it exactly would pin the fixture's
    profile rather than the weighting.
    """
    # Concentrated on lags 4-6 but NOT one-hot: a single live bin out of eleven is 91% exact zeros,
    # which the degeneracy guard correctly refuses -- see the test below, which pins that.
    excess = [0.30 if 4 <= index <= 6 else 0.01 for index in range(N_LAGS)]
    _, directory = _run(_context(lag=_lag_block(excess=excess)), tmp_path)

    selection = pd.read_csv(directory / analysis.SELECTION_FILENAME)
    weights = selection.set_index("lag_step")["soft_weight"]
    assert weights.loc[4:6].to_numpy() == pytest.approx(1.0)
    assert weights.loc[0] == pytest.approx(0.01 / 0.30)

    per_recording = pd.read_csv(directory / analysis.PER_RECORDING_FILENAME)

    def _centroids(source: str):
        rows = per_recording[
            (per_recording["source"] == source) & (per_recording["statistic"] == "centroid")
        ]
        return rows["value"].dropna().to_numpy()

    from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

    weighted = _centroids("kl_dw")
    assert len(weighted)
    # Every weighted centroid sits inside the window the weight kept, which the unweighted one
    # need not: the fixture puts half its segments' mass at lag 0.
    assert (weighted >= 3.0 * SECONDS_PER_STEP).all()
    assert (weighted <= 7.0 * SECONDS_PER_STEP).all()


def test_a_clock_excess_with_one_live_bin_is_refused_as_too_sparse_to_weight_with(
    tmp_path,
) -> None:
    r"""A one-hot excess is $91\%$ exact zeros at this width, which is above
    ``DEGENERATE_ZERO_FRACTION``.

    That refusal is correct and worth pinning rather than working around: ``entmax15`` sparsifies,
    so a profile reduced to a handful of live bins has a shape set by which of them survived
    rather than by where the source informed -- and a weight built from it would carry that
    accident into every window and every class at once.
    """
    one_hot = [1.0 if index == 5 else 0.0 for index in range(N_LAGS)]

    record, _ = _run(_context(lag=_lag_block(excess=one_hot)), tmp_path)

    assert record["sources"]["weight"]["available"] is False
    assert "kl_dw" not in record["features"]
    assert record["sources"]["weight"]["zero_fraction"] > 0.9


def test_a_degenerate_clock_excess_yields_no_weighted_source_at_all(tmp_path) -> None:
    """The guard that is expected to fire on the runs measured so far.

    A weight built from a flat profile is a near-uniform vector dressed as a selection: the
    weighted sources would look like independent evidence while being the unweighted ones. So they
    are not emitted, and the record says why rather than shipping a weight nobody should read.
    """
    record, directory = _run(
        _context(lag=_lag_block(excess=[0.1] * N_LAGS)), tmp_path
    )

    assert "kl_dw" not in record["features"]
    assert "attn_dw" not in record["features"]
    assert record["sources"]["weight"]["available"] is False
    assert "no readable shape" in record["sources"]["weight"]["reason"]

    trajectory = pd.read_csv(directory / analysis.TRAJECTORY_FILENAME)
    assert not len(trajectory[trajectory["source"].str.endswith("_dw")])
    # The geometry-fixed half is unaffected -- which is the point of it needing no estimate.
    assert {f"kl_{name}" for name in BANDS} <= set(record["features"])


def test_a_run_without_a_clock_excess_profile_keeps_the_bands(tmp_path) -> None:
    """The weight is the only thing a missing clock-excess profile costs. A directory collected
    before the lag-resolved source-null arm existed is a partial input, not a broken one."""
    lag = _lag_block()
    del lag["kl_lag_profile_clock_excess"]

    record, _ = _run(_context(lag=lag), tmp_path)

    assert record["sources"]["weight"]["available"] is False
    assert "predates" in record["sources"]["weight"]["reason"]
    assert {f"kl_{name}" for name in BANDS} <= set(record["features"])


# =================================================================================================
# The per-head reshape
# =================================================================================================
def test_a_per_head_vector_that_does_not_factor_is_dropped_whole(tmp_path) -> None:
    """Head-major and $M \\cdot L$ wide, or nothing.

    A flat vector whose length does not factor into ``num_heads * n_lags`` is a mis-assembled
    profile rather than a short one, and reshaping it into a plausible wrong answer is what this
    refuses -- the same guard ``metrics.lag_summary`` and ``lag_kl`` carry at the other two sites.
    """
    ragged = np.zeros((N_SEGMENTS, N_HEADS * N_LAGS - 1))

    record, _ = _run(_context(per_head=ragged), tmp_path)

    assert not [key for key in record["features"] if key.startswith("kl_h")]
    reasons = [entry["source"] for entry in record["sources"]["skipped"]]
    assert "kl_h*" in reasons


# =================================================================================================
# What this analysis deliberately does not do
# =================================================================================================
def test_no_significance_table_is_written_and_the_record_says_so(tmp_path) -> None:
    """The counterpart of ``lag_clocks``' "no untested statistic reaches the significance tables".

    There, the invariant keeps a Holm family at two. Here it is stronger: nothing is tested at all,
    so this analysis adds no family. An analysis that quietly began emitting $p$-values would
    multiply the corrections a reader is holding with nothing on the page saying so.
    """
    record, directory = _run(_context(), tmp_path)

    assert record["tested"] is False
    assert record["plan"]["tested_features"] == 0
    assert "NO Holm family" in record["no_inference_note"]
    assert [name for name in record["files"] if "significance" in name or "pairwise" in name] == []
    assert not list(directory.glob("*significance*"))
    assert not list(directory.glob("*pairwise*"))


def test_the_selection_is_written_to_disk_rather_than_left_to_be_reconstructed(tmp_path) -> None:
    """A selection reconstructed later from a re-run is not the selection the numbers beside it
    were chosen with: the bands come from a config that can be edited and the weight from a profile
    a different checkpoint would move."""
    _, directory = _run(_context(), tmp_path)

    selection = pd.read_csv(directory / analysis.SELECTION_FILENAME)

    assert len(selection) == N_LAGS
    assert set(selection.columns) == {
        "lag_step", "compensated_seconds", "band", "clock_excess_nats", "soft_weight",
    }
    assert set(selection["band"]) == set(BANDS)
    # The weight is peak-normalised, so it tops out at exactly one.
    assert selection["soft_weight"].max() == pytest.approx(1.0)


def test_every_emitted_row_states_the_unit_its_value_is_in(tmp_path) -> None:
    """One ``value`` column carrying seconds, nats, shares and a dimensionless moment. A number
    whose unit a reader has to infer from a statistic name is one they will infer wrongly."""
    _, directory = _run(_context(), tmp_path)

    for name in (analysis.PER_RECORDING_FILENAME, analysis.TRAJECTORY_FILENAME):
        frame = pd.read_csv(directory / name)
        assert "unit" in frame.columns, name
        assert frame["unit"].notna().all(), name
        assert (frame["unit"].astype(str) != "").all(), name


# =================================================================================================
# The two clocks, and the refusals
# =================================================================================================
def test_the_second_clock_is_a_subset_and_the_analysis_declares_itself_capped(tmp_path) -> None:
    """A recording with no recorded onset cannot be placed on the second axis at all, so the two
    clocks answer for different populations by design. Declaring ``capped`` is what makes the
    coverage block read that as a design decision rather than as a disagreement."""
    record, _ = _run(_context(), tmp_path)

    assert record["plan"]["capped"] is True
    names = {entry["clock"] for entry in record["clocks"]}
    assert names == {"time_to_delivery", "second_stage"}
    second = next(entry for entry in record["clocks"] if entry["clock"] == "second_stage")
    assert "n_recordings_eligible" in second


def test_a_run_with_no_configured_band_records_a_named_skip(tmp_path) -> None:
    """``occlusion_bands`` is read by two analyses now. An operator emptying it to skip the
    expensive interventional pass also removes this analysis's selection -- so it skips, naming
    that, rather than emitting its unrestricted half under a page whose selection is missing."""
    record = analysis.run_lag_kld_scaled_analysis(
        _context(), eval_config={"seed": 0, "occlusion_bands": {}}, output_dir=tmp_path
    )

    assert record["skipped"] is True
    assert "occlusion_bands" in record["reason"]
    assert "two analyses" in record["reason"]
    assert set(REQUIRED_RESULT_KEYS) <= set(record)


def test_a_run_with_no_lag_geometry_records_a_named_skip(tmp_path) -> None:
    """No axis to resolve a profile against is a partial input, and the skip names it."""
    record = analysis.run_lag_kld_scaled_analysis(
        _context(lag={}), eval_config=EVAL_CONFIG, output_dir=tmp_path
    )

    assert record["skipped"] is True
    assert "lag geometry" in record["reason"]


def test_a_sidecar_with_no_usable_profile_records_a_named_skip(tmp_path) -> None:
    """A vector of the wrong width is a mis-assembled profile rather than a short one."""
    record = analysis.run_lag_kld_scaled_analysis(
        _context(vectors={"lag_profile_untruncated": np.zeros((N_SEGMENTS, N_LAGS - 3))}),
        eval_config=EVAL_CONFIG,
        output_dir=tmp_path,
    )

    assert record["skipped"] is True
    assert "sidecar" in record["reason"] or "vector" in record["reason"]
