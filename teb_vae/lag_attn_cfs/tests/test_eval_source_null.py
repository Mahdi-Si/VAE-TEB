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


def _context(per_sample: pd.DataFrame) -> AnalysisContext:
    """An analysis context with no task and no loader, as an offline re-run has."""
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={},
        results={}, vectors={},
    )
    return AnalysisContext(collection=collection, config={})


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
