r"""The one control that can separate a coupling readout from a deterministic availability clock.

The source availability pattern is a function of $t$ alone and is identical in every row of a
batch, and it enters $q(z \mid Y, U)$ but not $p(z \mid Y)$ -- so the posterior can be pushed off
the prior by the clock and nothing else, and the coupling readout would report that as coupling.
No permutation of rows removes what every row shares, which is why the permutation control is
structurally blind to this rather than merely weaker at it.

**The load-bearing property is that the two readouts are reduced on one support**, and it is
proved here rather than claimed in a docstring: a batch whose source stream is *already* zero must
produce a null KL bit-identical to the matched one, because the null is the same computation on
the same inputs over the same mask. Any difference in the mask, in the sum over $d_z$, or in the
contributing-anchor denominator shows up as a non-zero difference on that batch, and nothing else
would.

The rest is the shape of the emitted record: an interval the acceptance verdict is decided on, a
positive fraction that reports its denominator, and the caveat that weakens the claim in the
model's favour -- zeroing floors the source's *variation* and is not literally the availability
pattern acting alone.
"""
from __future__ import annotations

import types
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.analyses import REQUIRED_RESULT_KEYS, AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import source_null as source_null_analysis
from teb_vae.lag_attn_cfs.eval.metrics import evaluate_batch

from .conftest import make_stub_batch

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


def _per_sample(
    coupling: Sequence[float],
    clock: Sequence[float],
    *,
    guids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """A per-sample table carrying both readouts and the difference the pass computes per sample.

    Args:
        coupling: The matched coupling readout per segment.
        clock: The source-null readout per segment.
        guids: The recording each segment belongs to; two segments per recording by default.

    Returns:
        The frame, with the cohort columns every per-recording reduction carries.
    """
    identifiers = list(guids) if guids is not None else [
        f"REC{index // 2:02d}" for index in range(len(coupling))
    ]
    return pd.DataFrame(
        {
            "guid": identifiers,
            "clinical_class": ["healthy"] * len(coupling),
            "subgroup": ["healthy_bg"] * len(coupling),
            "source_conditioned_kl_raw": [float(value) for value in coupling],
            "kld_source_null": [float(value) for value in clock],
            # As the collection pass writes it: the difference is taken per sample, not
            # differenced from two per-recording means afterwards.
            "coupling_minus_clock": [
                float(left) - float(right) for left, right in zip(coupling, clock)
            ],
        }
    )


def _context(per_sample: pd.DataFrame, lag: Optional[dict] = None) -> AnalysisContext:
    """An analysis context with no task and no loader, as an offline re-run has.

    ``lag`` populates ``collection.results['lag']``, which is where the lag-resolved half reads
    the two profiles from -- that block round-trips through ``collection.json``, so the lag half
    works offline exactly as the scalar half does.
    """
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={},
        results={"lag": dict(lag)} if lag else {}, vectors={},
    )
    return AnalysisContext(collection=collection, config={})


#: A lag block shaped like the finding the lag-resolved null exists to expose: the matched
#: attribution pinned at the near edge, and a null arm accounting for nearly all of it there, so
#: the CLOCK-EXCESS profile peaks inside [3, 6] where the matched one does not. Lag 7 is the one
#: bin where the null exceeds the matched arm, which is what makes the profile signed.
_MATCHED = [5.0, 1.0, 0.5, 0.6, 0.7, 0.6, 0.5, 0.20, 0.1, 0.1, 0.1]
_NULL = [4.9, 0.9, 0.4, 0.1, 0.1, 0.1, 0.1, 0.25, 0.1, 0.1, 0.1]

#: The geometry-fixed partition, at the tiny width. The same key the interventional readout reads,
#: because one partition per run is what makes ``near`` name one lag range in both tables.
_BANDS = {"anchor": [0, 2], "near": [3, 6], "mid": [7, 8], "far": [9, 10]}


def _lag_block(matched=None, null=None) -> dict:
    """A collection lag block carrying the three profiles the lag half reads."""
    matched = list(_MATCHED if matched is None else matched)
    null = list(_NULL if null is None else null)
    return {
        "n_lags": len(matched),
        "delay_steps": 0,
        "kl_lag_profile": matched,
        "kl_lag_profile_null": null,
        "kl_lag_profile_clock_excess": [m - n for m, n in zip(matched, null)],
    }


def _LAG_PER_SAMPLE() -> pd.DataFrame:
    """Six segments over three recordings, enough for a bootstrap interval to be real.

    The scalar half is not what these tests are about, so the values only have to be finite and
    to give a positive difference; the lag half reads the collection block, not this table.
    """
    # The differences must VARY: the difference figure histograms them, and a constant column
    # cannot be binned.
    return _per_sample(
        [1.0, 1.1, 0.9, 1.2, 1.05, 0.95], [0.60, 0.75, 0.45, 0.70, 0.60, 0.50]
    )


def _lag_config() -> dict:
    """The eval block the lag half reads: the bootstrap settings plus the band partition."""
    return {"bootstrap_resamples": 200, "seed": 0, "occlusion_bands": _BANDS}


# =================================================================================================
# The lag-resolved half
# =================================================================================================
def test_the_lag_table_decomposes_the_gated_scalar_over_the_lags(tmp_path) -> None:
    r"""The identity the lag-resolved null exists for, at the table a reader actually opens.

    $\sum_\ell \Delta_\ell = \texttt{coupling\_minus\_clock}$, because the matched profile sums
    to ``source_conditioned_kl_raw`` and the null one to ``kld_source_null``. That is what makes
    the per-lag clock-excess a decomposition of the very scalar ``clock_margin_min_nats`` gates,
    rather than a second lag reading that happens to have a clock subtracted from it.
    """
    lag = _lag_block()
    record = source_null_analysis.run_source_null_analysis(
        _context(_LAG_PER_SAMPLE(), lag),
        eval_config=_lag_config(),
        output_dir=tmp_path,
    )

    frame = pd.read_csv(
        tmp_path / source_null_analysis.ANALYSIS_DIRNAME
        / source_null_analysis.LAG_PROFILE_FILENAME
    )
    expected = sum(lag["kl_lag_profile"]) - sum(lag["kl_lag_profile_null"])
    assert frame["clock_excess_nats"].sum() == pytest.approx(expected, abs=1e-6)
    assert record["lag"]["net_nats"] == pytest.approx(expected, abs=1e-6)

    # The rectified column is a rectification: never negative, and never larger than the signed
    # one at any bin.
    assert (frame["clock_excess_positive_nats"] >= 0.0).all()
    assert (frame["clock_excess_positive_nats"] >= frame["clock_excess_nats"] - 1e-12).all()
    # ... and its total is an UPPER bound on the gated scalar, by exactly the discarded mass.
    assert record["lag"]["positive_nats"] > record["lag"]["net_nats"]
    assert record["lag"]["positive_nats"] + record["lag"]["negative_nats"] == pytest.approx(
        record["lag"]["net_nats"]
    )


def test_every_lag_is_labelled_with_the_band_the_intervention_removes_it_in(tmp_path) -> None:
    """One partition per run. The observational table names the same four bands the occlusion
    readout scores, so the two pages are read against each other by filtering rather than by
    aligning two four-way splits by eye."""
    record = source_null_analysis.run_source_null_analysis(
        _context(_LAG_PER_SAMPLE(), _lag_block()),
        eval_config=_lag_config(),
        output_dir=tmp_path,
    )

    frame = pd.read_csv(
        tmp_path / source_null_analysis.ANALYSIS_DIRNAME
        / source_null_analysis.LAG_PROFILE_FILENAME
    )
    assert set(frame["band"]) == set(_BANDS)
    for name, span in _BANDS.items():
        assert list(frame.loc[frame["band"] == name, "lag_step"]) == list(
            range(span[0], span[1] + 1)
        )
    # The shares are of the POSITIVE mass, so they partition it.
    assert sum(record["lag"]["band_shares"].values()) == pytest.approx(1.0)


def test_a_readable_clock_excess_profile_nominates_the_band_around_its_peak(tmp_path) -> None:
    """The secondary selection: the contiguous run at or above half the peak, which is
    ``lag_shape.peak_width``'s own definition rather than a threshold invented here."""
    record = source_null_analysis.run_source_null_analysis(
        _context(_LAG_PER_SAMPLE(), _lag_block()),
        eval_config=_lag_config(),
        output_dir=tmp_path,
    )

    assert record["lag"]["clock_excess_degenerate"] is False
    assert record["lag"]["delta_mask"] == [3, 6]
    assert record["lag"]["delta_mask_reason"] == ""


def test_a_degenerate_clock_excess_profile_nominates_nothing_and_says_why(tmp_path) -> None:
    """The guard that matters most, and the one expected to fire on the runs measured so far.

    The diagnosed run put 67.5% of its KL in an availability clock and 0.160 nats of source
    content across 91 lags. ``entmax15`` assigns lags exactly zero, so a profile that flat still
    has a perfectly confident argmax -- and a mask cut from it would name a band on arithmetic
    accident while looking exactly like a finding. A withheld mask is the measurement.
    """
    flat = [1.0] * 11
    record = source_null_analysis.run_source_null_analysis(
        # A null arm a hair below the matched one at every lag: a positive, perfectly flat excess.
        _context(_LAG_PER_SAMPLE(), _lag_block(matched=flat, null=[0.9] * 11)),
        eval_config=_lag_config(),
        output_dir=tmp_path,
    )

    assert record["lag"]["clock_excess_degenerate"] is True
    assert record["lag"]["delta_mask"] is None
    assert "degenerate" in record["lag"]["delta_mask_reason"]
    # The refusal names the alternative rather than leaving a reader with nothing.
    assert "geometry-fixed bands" in record["lag"]["delta_mask_reason"]


def test_a_run_predating_the_lag_resolved_null_reports_absent_rather_than_zero(tmp_path) -> None:
    """A finished directory collected before the null arm was lag-resolved is a partial input.

    Every key is still present, because the headline paths must resolve on every run -- and the
    unmeasured ones are ``None`` rather than ``NaN``, because the headline finiteness check reads
    a non-finite NUMBER as a broken run and a ``None`` as an analysis that did not report.
    """
    record = source_null_analysis.run_source_null_analysis(
        _context(_LAG_PER_SAMPLE()), eval_config=_lag_config(), output_dir=tmp_path
    )

    block = record["lag"]
    assert block["measured"] is False
    for key in (
        "clock_excess_argmax_lag_step",
        "clock_excess_peak_share",
        "clock_excess_degenerate",
        "clock_excess_rectified_frac",
    ):
        assert block[key] is None, key
    # No table and no figure are written for a half that was not measured.
    assert source_null_analysis.LAG_PROFILE_FILENAME not in record["files"]


def test_the_rectification_caveat_travels_with_the_shares_it_qualifies(tmp_path) -> None:
    """The one thing a reader can get wrong here that the arithmetic does not announce: the
    positive total is an upper bound on the gated scalar, and the band shares partition the
    positive mass rather than the scalar."""
    record = source_null_analysis.run_source_null_analysis(
        _context(_LAG_PER_SAMPLE(), _lag_block()),
        eval_config=_lag_config(),
        output_dir=tmp_path,
    )

    caveat = record["lag"]["rectification_caveat"]
    assert "UPPER BOUND" in caveat
    assert "coupling_minus_clock" in caveat
    assert "signed" in caveat.lower()


# =================================================================================================
# The support, measured rather than claimed
# =================================================================================================
def _readouts(task, perturb_posterior, *, zero_source: bool):
    """One real forward's per-sample columns, on a model whose KL is not structurally zero.

    The perturbation is load-bearing rather than incidental: the posterior delta heads are
    zero-initialised, so at init the posterior equals the prior exactly and **every** KL is $0$ --
    under which the equality this file is about would hold on a model that was completely wrong.

    Args:
        task: The task factory fixture.
        perturb_posterior: The factory that breaks the zero-init.
        zero_source: Whether to hand the forward a source stream that is already zero.

    Returns:
        The per-sample columns.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    batch = make_stub_batch(seed=5)
    if zero_source:
        batch.up_st = torch.zeros_like(batch.up_st)
        batch.up_ph = torch.zeros_like(batch.up_ph)
    torch.manual_seed(0)
    return evaluate_batch(module, batch, num_samples=1).columns


def test_a_source_that_is_already_zero_makes_the_null_and_the_matched_readout_identical(
    task, perturb_posterior
) -> None:
    """The proof that both are reduced on one support, in one expression.

    The null re-encodes a **zeroed** source stream. Handed a batch whose source is already zero it
    is therefore the same computation on the same inputs over the same mask, so the two columns
    must agree to float32 accumulation -- and any divergence in the mask, in the sum over $d_z$ or
    in the contributing-anchor denominator would show up here as a difference of the KL's own
    order, and in no other test in this suite.

    To accumulation rather than bit-exactly: the null is a **second** pass through the recurrent
    source encoder, so its reductions happen in their own order and the last bits of a float32
    accumulation over $T$ steps are not reproducible between two such passes.
    """
    columns = _readouts(task, perturb_posterior, zero_source=True)
    coupling = columns["source_conditioned_kl_raw"]

    assert float(coupling.abs().min()) > 0.0, (
        "a structurally zero KL would satisfy the equality below on any implementation"
    )
    assert torch.allclose(columns["kld_source_null"], coupling, rtol=1e-5, atol=1e-6)
    # The difference is accumulation noise rather than a measurement: orders of magnitude below
    # the quantity it is a difference of, which is what separates it from a wrong support.
    assert float(columns["coupling_minus_clock"].abs().max()) < 1e-4 * float(
        coupling.abs().min()
    )


def test_a_real_source_moves_the_two_apart_so_the_test_above_is_not_vacuous(
    task, perturb_posterior
) -> None:
    """With a source that carries variation the null is a different re-encode, so the difference
    is a measurement rather than a structural zero."""
    columns = _readouts(task, perturb_posterior, zero_source=False)

    assert not torch.allclose(
        columns["kld_source_null"], columns["source_conditioned_kl_raw"], atol=1e-9
    )
    assert torch.allclose(
        columns["coupling_minus_clock"],
        columns["source_conditioned_kl_raw"] - columns["kld_source_null"],
    )


# =================================================================================================
# The summary rows
# =================================================================================================
def test_the_difference_row_carries_its_interval_the_positive_fraction_and_a_paired_test() -> None:
    """A status without the numbers behind it is a status a reader cannot check, and a fraction
    without its denominator is one a coverage collapse would silently move."""
    per_guid = source_null_analysis.per_recording_means(
        _per_sample([3.0, 3.2, 2.8, 3.1, 3.4, 2.9], [1.0, 1.4, 1.2, 1.1, 1.3, 1.0]),
        source_null_analysis.GROUPED_METRICS,
    )

    rows = source_null_analysis.build_rows(per_guid, resamples=200, seed=0)
    by_metric = {row["metric"]: row for row in rows}
    difference = by_metric[source_null_analysis.DIFFERENCE_COLUMN]

    assert [row["metric"] for row in rows] == list(source_null_analysis.GROUPED_METRICS)
    assert difference["mean"] == pytest.approx(
        by_metric["source_conditioned_kl_raw"]["mean"] - by_metric["kld_source_null"]["mean"]
    )
    assert np.isfinite(difference["ci_lo"]) and np.isfinite(difference["ci_hi"])
    assert difference["positive_fraction"] == pytest.approx(1.0)
    assert difference["n_recordings_scored"] == 3
    assert difference["n_positive"] == 3
    assert difference["wilcoxon_n_pairs"] == 3


def test_the_positive_fraction_reports_the_denominator_it_was_taken_over() -> None:
    r"""``np.nan > 0`` is ``False``, so a recording that scored no anchor would otherwise be
    counted silently as evidence *against* and a coverage collapse would read as a falling
    fraction rather than as a falling $n$."""
    frame = _per_sample([3.0, 3.0, np.nan, np.nan, 1.0, 1.0], [1.0, 1.0, np.nan, np.nan, 2.0, 2.0])
    per_guid = source_null_analysis.per_recording_means(
        frame, source_null_analysis.GROUPED_METRICS
    )

    rows = source_null_analysis.build_rows(per_guid, resamples=200, seed=0)
    difference = {row["metric"]: row for row in rows}[
        source_null_analysis.DIFFERENCE_COLUMN
    ]

    assert difference["n_recordings_scored"] == 2
    assert difference["n_recordings_dropped_not_finite"] == 1
    assert difference["positive_fraction"] == pytest.approx(0.5)


def test_the_difference_block_is_flat_and_carries_both_halves_beside_the_difference() -> None:
    """The headline registry digs into this block by key, and a difference quoted without the two
    numbers it came from cannot be sanity-checked: $0.1$ out of $0.2$ and $0.1$ out of $20$ are the
    same difference and opposite findings."""
    per_guid = source_null_analysis.per_recording_means(
        _per_sample([3.0] * 6, [1.0] * 6),
        source_null_analysis.GROUPED_METRICS,
    )
    rows = source_null_analysis.build_rows(per_guid, resamples=200, seed=0)

    block = source_null_analysis.difference_record(rows)

    assert block["coupling_minus_clock_nats"] == pytest.approx(2.0)
    assert block["source_conditioned_kl_raw_nats"] == pytest.approx(3.0)
    assert block["kld_source_null_nats"] == pytest.approx(1.0)
    assert block["ci_lo"] == pytest.approx(2.0)
    assert block["ci_hi"] == pytest.approx(2.0)
    assert "DESIGN.md" in block["caveat"]


def test_a_run_that_measured_nothing_reports_the_difference_as_unmeasured() -> None:
    """``NaN`` rather than a fabricated zero: the verdict reads an absent measurement as
    unevaluated, and a zero would read as "the clock accounts for all of it"."""
    per_guid = source_null_analysis.per_recording_means(
        _per_sample([np.nan] * 6, [np.nan] * 6), source_null_analysis.GROUPED_METRICS
    )
    rows = source_null_analysis.build_rows(per_guid, resamples=200, seed=0)

    block = source_null_analysis.difference_record(rows)

    assert not np.isfinite(block["coupling_minus_clock_nats"])
    assert block["n_recordings"] == 0


# =================================================================================================
# The analysis
# =================================================================================================
def test_the_analysis_writes_both_tables_and_declares_its_grouped_frame(tmp_path) -> None:
    result = source_null_analysis.run_source_null_analysis(
        _context(_per_sample([3.0, 3.2, 2.8, 3.1, 3.4, 2.9], [1.0, 1.4, 1.2, 1.1, 1.3, 1.0])),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / source_null_analysis.ANALYSIS_DIRNAME
    assert set(REQUIRED_RESULT_KEYS) <= set(result)
    assert (directory / source_null_analysis.PER_RECORDING_FILENAME).is_file()
    assert (directory / source_null_analysis.SUMMARY_FILENAME).is_file()
    assert result["n_samples"] == 6
    assert result["composition"]["n_recordings"] == 3

    entry = result["grouped_frames"][0]
    assert entry["path"] == (
        f"{source_null_analysis.ANALYSIS_DIRNAME}/"
        f"{source_null_analysis.PER_RECORDING_FILENAME}"
    )
    # All three, because a cohort whose coupling differs may differ in the clock rather than in
    # the source, and only both halves beside the difference let a reader tell which.
    assert set(entry["value_columns"]) == set(source_null_analysis.GROUPED_METRICS)


def test_the_record_cites_the_design_rather_than_restating_its_argument(tmp_path) -> None:
    """Two copies of an argument are two chances for one of them to stop being true."""
    result = source_null_analysis.run_source_null_analysis(
        _context(_per_sample([3.0] * 6, [1.0] * 6)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    assert "lag_attn_cfs/DESIGN.md section 8" in result["caveat"]
    assert "weaker" in result["caveat"]
    assert "nonlinear" in result["caveat"]
    # And the reason the permutation control is not a substitute, beside the number rather than
    # left for a reader holding both to work out.
    assert "every row" in result["perm_control_note"]


def test_the_per_recording_frame_is_the_one_the_cross_cohort_analysis_reads(tmp_path) -> None:
    """The filename and the column name are a contract with that analysis's source table, not a
    local choice: it reads this frame off disk by name."""
    from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup

    source_null_analysis.run_source_null_analysis(
        _context(_per_sample([3.0, 3.2, 2.8, 3.1, 3.4, 2.9], [1.0, 1.4, 1.2, 1.1, 1.3, 1.0])),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )
    written = pd.read_csv(
        tmp_path
        / source_null_analysis.ANALYSIS_DIRNAME
        / source_null_analysis.PER_RECORDING_FILENAME
    )

    wanted = [
        source for source in cross_subgroup.METRIC_SOURCES
        if source.analysis == source_null_analysis.ANALYSIS_DIRNAME
    ]
    assert wanted, "the cross-cohort analysis registers no metric from this one"
    for source in wanted:
        assert source.filename == source_null_analysis.PER_RECORDING_FILENAME
        assert source.column in written.columns
        # Higher is better: more of the coupling is source variation rather than a clock.
        assert source.higher_is_better is True
    assert {"clinical_class", "subgroup"} <= set(written.columns)


def test_every_row_states_the_unit_and_what_it_measures(tmp_path) -> None:
    result = source_null_analysis.run_source_null_analysis(
        _context(_per_sample([3.0] * 6, [1.0] * 6)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    assert result["unit"] == "nats per anchor"
    for row in result["metrics"]:
        assert row["unit"] == result["unit"]
        assert row["meaning"].strip()


# =================================================================================================
# On a real run
# =================================================================================================
@pytest.mark.slow
def test_the_analysis_runs_and_its_interval_reaches_the_verdict(collected_run) -> None:
    """The wiring end to end: the analysis produces the interval, and the criterion that is stated
    on that interval is re-decided from it rather than left at the collection pass's placeholder.

    Under the shipped unset threshold the status stays INCONCLUSIVE, which is the point -- what
    must have changed is that the *measurement* is now attached to it.
    """
    summary = collected_run["summary"]
    block = summary["results"]["source_null"]
    difference = block["difference"]

    assert block["n_samples"] == summary["results"]["n_samples"]
    assert np.isfinite(difference["coupling_minus_clock_nats"])
    assert np.isfinite(difference["ci_lo"]) and np.isfinite(difference["ci_hi"])
    assert difference["ci_lo"] <= difference["coupling_minus_clock_nats"] <= difference["ci_hi"]

    verdict = next(
        entry for entry in summary["results"]["verdicts"]
        if entry["name"] == "coupling_exceeds_availability_clock"
    )
    assert summary["eval_config"]["clock_margin_min_nats"] is None
    assert verdict["status"] == "INCONCLUSIVE"
    assert verdict["values"]["interval_lo"] == pytest.approx(difference["ci_lo"])
    assert verdict["values"]["coupling_minus_clock_nats"] == pytest.approx(
        difference["coupling_minus_clock_nats"]
    )
    # And the number is in the headline whatever the status said, which is what lets the threshold
    # be set from the observed spread rather than guessed.
    assert summary["results"]["headline"]["coupling_minus_clock_nats"] == pytest.approx(
        difference["coupling_minus_clock_nats"]
    )


@pytest.mark.slow
def test_every_file_the_analysis_declared_is_on_disk(collected_run) -> None:
    """A declared artifact that was never written reads in the summary exactly like one that was."""
    directory = collected_run["results_dir"] / source_null_analysis.ANALYSIS_DIRNAME
    block = collected_run["summary"]["results"]["source_null"]

    for name in block["files"]:
        assert (directory / name).is_file(), name
    assert block.get("grouped"), "the runner emitted no by-cohort variant over the declared frame"
