r"""The per-lag KL attribution: the identity it rests on, the support it assumes, and the reading.

Three things are checked here and they fail in different ways.

**The identity.** $\sum_\ell \widetilde K_{t,\ell} = K_t$ holds anchor by anchor, and the one
mechanism that breaks it is dropout on the attention probabilities: dropout rescales them by
$1/(1-p)$ and zeroes a random subset, so each head's weights no longer sum to one and the
attribution holds only in expectation. Every number the analysis reports would still look
entirely reasonable. So the run measures the residual on its own numbers, and the checks below
prove the measurement has teeth.

**The support.** At this cell's shipped geometry the anchor floor $F = 133$ exceeds the furthest
searched lag $L - 1 = 90$, so every lag is causally valid at every scored anchor and all three
profiles coincide. That is a property of $F \ge L - 1$ rather than of the domain: the floor,
``max_lag`` and ``lag_floor`` move independently, and a ``sweep_floor_*`` arm reintroduces
truncation. So the analysis reads preflight's measured margin, measures the per-lag anchor counts,
and records whether the two agree -- and the tests below assert the agreement **conditionally on
that margin**, with a constructed truncated case where the three must differ.

**The reading.** An argmax is not a reading of a profile. Ties resolve to the lowest index, and
``entmax15`` assigns lags exactly zero, so a profile that is flat or nearly empty still has a
perfectly confident argmax. The peak description and the mechanical degeneracy criterion are what
stop that from being reported as a finding, and they are tested on synthetic profiles of known
answer rather than on whatever the fixture happened to produce.

And throughout: this axis is **stored-coefficient time**. The caveat that says so travels on every
artifact that states a lag position, and its absence from any of them is a test failure here.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.figures_seam import figure_filename
from teb_vae.lag_attn_cfs.eval import lag_axis, preflight, report_seam
from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import lag_kl
from teb_vae.lag_attn.nets.lag_report import lag_compensated_seconds

#: The identity key the lag map's residual is reported under.
_MAP_RESIDUAL = "lag_map_sums_to_kl_max_abs_nats"

#: The shipped geometry's measured margin: $F - \max_{\rm lag} - F_u = 133 - 90 - 0$.
_SHIPPED_MARGIN = 43


# =================================================================================================
# The measured lag support: read, not assumed
# =================================================================================================
def _write_preflight(directory: Path, *, margin: Optional[int]) -> Path:
    """Write a preflight record carrying a lag-support block, or one carrying none."""
    causality: Dict[str, Any] = {}
    if margin is not None:
        causality["lag_support"] = {
            "min_decoded_anchor": 133,
            "max_lag": 90,
            "n_lags": 91,
            "lag_floor": 0,
            "lag_support_margin_steps": int(margin),
            "every_lag_valid_at_every_anchor": bool(margin >= 0),
        }
    path = directory / lag_axis.PREFLIGHT_FILENAME
    path.write_text(json.dumps({"causality": causality}), encoding="utf-8")
    return path


def test_the_reader_and_the_writer_name_the_same_file() -> None:
    """The reader restates the filename rather than importing it, because ``preflight`` rebuilds a
    checkpoint and the analyses that read the number run offline. Pinned equal so a rename cannot
    leave one side reading a file the other stopped writing."""
    assert lag_axis.PREFLIGHT_FILENAME == preflight.PREFLIGHT_FILENAME == "preflight.json"


def test_the_margin_is_read_off_the_runs_own_preflight_record(tmp_path) -> None:
    _write_preflight(tmp_path, margin=_SHIPPED_MARGIN)

    support = lag_axis.read_lag_support(tmp_path)

    assert support["measured"] is True
    assert support["lag_support_margin_steps"] == _SHIPPED_MARGIN
    assert support["every_lag_valid_at_every_anchor"] is True


def test_an_absent_record_reports_unmeasured_rather_than_a_default(tmp_path) -> None:
    """``None`` rather than a default margin, and the flag ``None`` rather than ``False``: an
    analysis must be able to tell "the geometry is truncated" from "nobody measured", and a
    ``False`` here would make an unmeasured run report a truncation it never had."""
    support = lag_axis.read_lag_support(tmp_path)

    assert support["measured"] is False
    assert support["lag_support_margin_steps"] is None
    assert support["every_lag_valid_at_every_anchor"] is None
    assert lag_axis.PREFLIGHT_FILENAME in support["reason"]


def test_a_record_with_no_support_block_is_also_unmeasured(tmp_path) -> None:
    _write_preflight(tmp_path, margin=None)

    support = lag_axis.read_lag_support(tmp_path)

    assert support["measured"] is False
    assert "lag_support" in support["reason"]


def _lag_block(*, truncated: bool, n_lags: int = 9) -> Dict[str, Any]:
    """A pass's lag block, either at a geometry that admits every lag or at one that does not.

    Untruncated: one profile, uniform anchor counts, all three readings identical. Truncated: the
    long lags are averaged over fewer anchors, so the corrected and untruncated profiles part
    company with the raw one -- which is exactly the bias the corrections exist to remove.
    """
    raw = np.linspace(1.0, 0.2, n_lags)
    counts = np.full(n_lags, 20.0)
    corrected = raw.copy()
    untruncated = raw.copy()
    if truncated:
        counts = np.linspace(20.0, 4.0, n_lags)
        corrected = raw * 20.0 / counts
        untruncated = corrected * 1.05
    return {
        "n_lags": n_lags,
        "delay_steps": 0,
        "kl_lag_profile": raw.tolist(),
        "kl_lag_profile_support_corrected": corrected.tolist(),
        "kl_lag_profile_untruncated": untruncated.tolist(),
        "kl_lag_anchor_counts": counts.tolist(),
        "kl_argmax_lag_step": 0,
    }


def test_at_a_non_negative_margin_the_three_profiles_are_measured_to_coincide() -> None:
    r"""The simplification $F \ge L - 1$ buys, asserted as a **measurement** conditional on the
    margin rather than as a property of the domain."""
    recorded = {
        "measured": True,
        "lag_support_margin_steps": _SHIPPED_MARGIN,
        "every_lag_valid_at_every_anchor": True,
    }

    support = lag_kl.measured_lag_support(_lag_block(truncated=False), recorded)

    assert support["lag_support_margin_steps"] >= 0
    assert support["anchor_counts_uniform"] is True
    assert support["profiles_agree"] is True
    assert support["max_abs_profile_difference"] <= lag_kl.PROFILE_AGREEMENT_TOLERANCE
    assert support["computed_and_observed_agree"] is True


def test_at_a_negative_margin_the_three_profiles_are_measured_to_differ() -> None:
    """Non-vacuity for the case above. A test that only ever saw the shipped geometry would pass
    on an implementation that returned ``True`` unconditionally."""
    recorded = {
        "measured": True,
        "lag_support_margin_steps": -5,
        "every_lag_valid_at_every_anchor": False,
    }

    support = lag_kl.measured_lag_support(_lag_block(truncated=True), recorded)

    assert support["lag_support_margin_steps"] < 0
    assert support["anchor_counts_uniform"] is False
    assert support["profiles_agree"] is False
    assert support["max_abs_profile_difference"] > lag_kl.PROFILE_AGREEMENT_TOLERANCE
    # The computed and observed readings still agree with each other: both say truncated.
    assert support["computed_and_observed_agree"] is True


def test_a_geometry_preflight_and_the_pass_disagree_about_is_reported_as_such() -> None:
    """The failure no other number would show: preflight described one geometry and the pass
    decoded at another. Recorded rather than raised -- an analysis that refused here would take
    down a run whose tables are perfectly readable."""
    recorded = {
        "measured": True,
        "lag_support_margin_steps": _SHIPPED_MARGIN,
        "every_lag_valid_at_every_anchor": True,
    }

    support = lag_kl.measured_lag_support(_lag_block(truncated=True), recorded)

    assert support["computed_and_observed_agree"] is False


def test_an_unmeasured_margin_leaves_the_comparison_undecided() -> None:
    """A run whose margin nobody measured must not report an agreement it cannot have checked."""
    support = lag_kl.measured_lag_support(
        _lag_block(truncated=False), dict(lag_axis.UNMEASURED_LAG_SUPPORT)
    )

    assert support["measured"] is False
    assert support["computed_and_observed_agree"] is None
    # What the pass itself observed is still reported: that half needs no preflight record.
    assert support["anchor_counts_uniform"] is True


# =================================================================================================
# The three profiles
# =================================================================================================
def test_all_three_profiles_are_kept_and_named() -> None:
    """The untruncated recomputation is retained rather than folded into the corrected one: whether
    the three coincide is the measurement that says this geometry admits every lag."""
    assert [name for name, _, _, _ in lag_kl.PROFILES] == [
        "raw", "support_corrected", "untruncated"
    ]


def test_the_profile_table_carries_a_column_per_profile() -> None:
    lag = _lag_block(truncated=True)
    seconds = lag_kl.compensated_seconds_axis(9, delay_steps=0)

    frame = lag_kl.profile_frame(lag, seconds)

    assert len(frame) == 9
    for _, key, column, _ in lag_kl.PROFILES:
        assert column in frame.columns
        assert list(frame[column]) == pytest.approx(list(lag[key]))
    assert list(frame["anchor_count"]) == pytest.approx(lag["kl_lag_anchor_counts"])


def test_an_empty_lag_block_yields_the_columns_and_no_rows() -> None:
    """A run that scored nothing reaches this analysis like any other, and a consumer reading the
    CSV must find the columns rather than an empty file with no header."""
    frame = lag_kl.profile_frame({}, lag_kl.compensated_seconds_axis(9, delay_steps=0))

    assert len(frame) == 0
    for _, _, column, _ in lag_kl.PROFILES:
        assert column in frame.columns


def test_the_summary_describes_every_profile() -> None:
    rows = lag_kl.build_summary_rows(_lag_block(truncated=True), 0)

    assert [row["profile"] for row in rows] == ["raw", "support_corrected", "untruncated"]
    for row in rows:
        assert row["argmax_lag_step"] is not None
        assert row["axis_caveat"] == lag_axis.GROUP_DELAY_CAVEAT


# =================================================================================================
# The identity
# =================================================================================================
@pytest.mark.parametrize("identity", sorted(report_seam.LAG_IDENTITIES))
def test_a_violated_identity_names_dropout_as_the_likely_cause(identity: str) -> None:
    """A residual with no explanation beside it costs an afternoon of reading the wrong module.

    Parametrised over the registry rather than written for one identity: the set grows -- the
    source-null arm's attribution is the third -- and an identity added without a case here would
    reach the sanity block with nothing checking that its failure message is readable.
    """
    residual_key = report_seam.LAG_IDENTITIES[identity][0]
    record = report_seam.check_lag_identity(
        {
            "lag": {"identity_residuals": {residual_key: 0.5}},
            "readouts": {"source_conditioned_kl_raw": 1.0},
        },
        identity,
    )

    assert record["verdict"] == "fail"
    assert "dropout" in record["detail"]
    assert record["max_abs_residual_nats"] == pytest.approx(0.5)


def test_an_unmeasured_identity_is_inconclusive_rather_than_passing() -> None:
    """"The run did not check this" and "this held" are different statements."""
    record = report_seam.check_lag_identity({"lag": {}}, "lag_map_sums_to_kl")

    assert record["verdict"] == report_seam.INCONCLUSIVE


def test_the_tolerance_floor_lifts_with_the_quantity_it_bounds() -> None:
    """Both identities are exact, so their residual is float32 accumulation -- which grows with
    the KL rather than staying under a fixed number of nats. A flat bound would fail a healthy
    production run for no reason other than that its KL is large."""
    floor = report_seam.identity_tolerance_for(1.0)
    lifted = report_seam.identity_tolerance_for(1e4)

    assert floor == report_seam.IDENTITY_TOLERANCE
    assert lifted > floor
    assert report_seam.identity_tolerance_for(None) == report_seam.IDENTITY_TOLERANCE


# =================================================================================================
# Reading a profile
# =================================================================================================
def test_a_bimodal_profile_reports_both_peaks() -> None:
    """An argmax reports one of two delays and a mean reports neither."""
    profile = [0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.1, 0.8, 0.1]

    peaks = lag_kl.secondary_peaks(profile)

    assert [record["lag_step"] for record in peaks] == [7]
    assert peaks[0]["share_of_peak"] == pytest.approx(0.8)
    assert lag_kl.peak_width(profile)["argmax"] == 2


def test_a_flat_profile_is_degenerate_by_the_stated_criterion() -> None:
    """Flat, not "looks flat": the peak-to-median ratio is 1.0 against a threshold of 1.1, and the
    reason says so in the output rather than in a docstring."""
    verdict = lag_kl.degeneracy([0.2] * 9)

    assert verdict["degenerate"] is True
    assert verdict["peak_to_median"] == pytest.approx(1.0)
    assert verdict["peak_to_median"] < lag_kl.DEGENERATE_PEAK_TO_MEDIAN
    assert any("peak-to-median" in reason for reason in verdict["reasons"])


def test_a_nearly_all_zero_profile_is_degenerate_for_the_other_reason() -> None:
    """``entmax15`` assigns exact zeros, so sparsity is legitimate -- but a profile with one live
    bin in twenty has a shape set by which bins survived rather than by where the source informed.
    """
    verdict = lag_kl.degeneracy([0.0] * 19 + [1.0])

    assert verdict["degenerate"] is True
    assert verdict["zero_fraction"] == pytest.approx(0.95)
    assert any("exactly zero" in reason for reason in verdict["reasons"])


def test_a_peaked_profile_is_not_degenerate() -> None:
    """The criterion has to admit the ordinary case, or it reports every run as unreadable."""
    verdict = lag_kl.degeneracy([0.05, 0.1, 0.4, 1.0, 0.4, 0.1, 0.05, 0.02, 0.01])

    assert verdict["degenerate"] is False
    assert verdict["reasons"] == []


def test_the_peak_width_is_the_contiguous_run_around_the_argmax() -> None:
    """Contiguity is what makes it a width: counting every bin above the threshold anywhere would
    report a bimodal profile as one very wide peak, which is the opposite of what it is."""
    width = lag_kl.peak_width([0.0, 0.6, 1.0, 0.6, 0.0, 0.9, 0.0])

    assert width["argmax"] == 2
    assert (width["lo"], width["hi"]) == (1, 3)
    assert width["width_bins"] == 3


def test_the_mass_above_half_the_peak_carries_its_bin_count() -> None:
    """A peak holding four fifths of the attribution and one holding a twentieth have the same
    argmax and are different findings."""
    concentration = lag_kl.mass_above([1.0, 0.0, 0.0, 0.0])

    assert concentration["share"] == pytest.approx(1.0)
    assert concentration["n_bins"] == 1


def test_an_empty_profile_reports_nothing_rather_than_raising() -> None:
    """A run that scored nothing reaches this analysis like any other."""
    assert lag_kl.peak_width([])["argmax"] is None
    assert lag_kl.secondary_peaks([]) == []
    assert lag_kl.degeneracy([])["degenerate"] is True
    assert np.isnan(lag_kl.mass_above([])["share"])


# =================================================================================================
# The lag axis
# =================================================================================================
def test_the_seconds_axis_is_the_compensated_one_elementwise() -> None:
    r"""Not $4\ell$, and not $4\ell + 20$: the compensated figure $4(\ell + \delta)$, built
    through the shared converter so a second implementation cannot drift from it."""
    axis = lag_kl.compensated_seconds_axis(9, delay_steps=0)

    assert axis.tolist() == [
        float(lag_compensated_seconds(lag, delay_steps=0)) for lag in range(9)
    ]


def test_a_nonzero_input_delay_shifts_every_second_by_four_delta() -> None:
    """The historical bug, in the shape it took: one consumer read the delay and the other read
    zero, and the two reports of one run disagreed by a whole horizon with nothing raising."""
    delay = 30
    base = lag_kl.compensated_seconds_axis(9, delay_steps=0)

    shifted = lag_kl.compensated_seconds_axis(9, delay_steps=delay)

    assert np.allclose(shifted - base, 4.0 * delay)


# =================================================================================================
# The figure
# =================================================================================================
def _figure_lines(figure):
    """The drawn curves of the top panel, by label."""
    return {line.get_label(): line for line in figure.axes[0].get_lines()}


def test_the_profile_figure_draws_against_the_coefficient_time_axis() -> None:
    """The axis label names the quantity; these assertions are about the *numbers*, which is what
    was historically wrong -- and about the label saying what the axis is time **in**."""
    import matplotlib.pyplot as plt

    lag = _lag_block(truncated=False)
    seconds = lag_kl.compensated_seconds_axis(9, delay_steps=0)
    profile = lag_kl.profile_frame(lag, seconds)

    figure = lag_kl.build_profile_figure(profile, lag, delay_steps=0, n_lags=9)
    try:
        drawn = _figure_lines(figure)["raw attribution (sums to the KL)"]
        x_values = np.asarray(drawn.get_xdata(), dtype=float)
        label = figure.axes[0].get_xlabel()
        n_curves = len(
            [line for line in figure.axes[0].get_lines() if line.get_linestyle() == "-"]
        )
    finally:
        plt.close(figure)

    assert np.allclose(x_values, seconds)
    assert label == lag_axis.COEFFICIENT_LAG_AXIS_LABEL
    assert "stored-coefficient time" in label
    assert "bpm" not in label and "physiological" not in label
    # All three profiles are drawn, not two: at this geometry they lie on top of one another and
    # that coincidence is the reading.
    assert n_curves == len(lag_kl.PROFILES)


def test_the_figure_carries_the_group_delay_caveat_on_its_face() -> None:
    """A figure is the artifact most likely to be lifted out of a run directory and shown alone,
    and a peak read off one without this beside it is read as a physiological latency."""
    import matplotlib.pyplot as plt

    lag = _lag_block(truncated=False)
    seconds = lag_kl.compensated_seconds_axis(9, delay_steps=0)

    figure = lag_kl.build_profile_figure(
        lag_kl.profile_frame(lag, seconds), lag, delay_steps=0, n_lags=9
    )
    try:
        texts = [artist.get_text() for artist in figure.texts]
    finally:
        plt.close(figure)

    assert lag_axis.GROUP_DELAY_CAVEAT in texts
    assert "not a transfer entropy" in lag_axis.GROUP_DELAY_CAVEAT


def test_rebuilding_at_a_nonzero_delay_shifts_every_drawn_second() -> None:
    """The property that catches a figure reading its own guess of the delay."""
    import matplotlib.pyplot as plt

    lag = {"kl_lag_profile": [0.1] * 9, "kl_lag_anchor_counts": [9.0] * 9}
    delay = 30

    drawn = []
    for delay_steps in (0, delay):
        seconds = lag_kl.compensated_seconds_axis(9, delay_steps=delay_steps)
        figure = lag_kl.build_profile_figure(
            lag_kl.profile_frame(lag, seconds), lag, delay_steps=delay_steps, n_lags=9
        )
        try:
            drawn.append(
                np.asarray(
                    _figure_lines(figure)["raw attribution (sums to the KL)"].get_xdata(),
                    dtype=float,
                )
            )
        finally:
            plt.close(figure)

    assert np.allclose(drawn[1] - drawn[0], 4.0 * delay)


# =================================================================================================
# The analysis end to end, on stub tables
# =================================================================================================
def _context(lag: Dict[str, Any], per_sample: Optional[pd.DataFrame] = None) -> AnalysisContext:
    """An analysis context carrying a finished lag block and a minimal per-sample table."""
    frame = per_sample if per_sample is not None else pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b"],
            "source_conditioned_kl_raw": [1.0, 1.1, 0.9, 1.0],
            "lag_map_identity_max_abs": [0.0] * 4,
            "head_kl_identity_max_abs": [0.0] * 4,
        }
    )
    collection = types.SimpleNamespace(
        per_sample=frame,
        per_anchor=pd.DataFrame(),
        record={},
        retained={},
        vectors={},
        results={"lag": lag, "readouts": {"source_conditioned_kl_raw": 1.0}},
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_records_the_measured_support_it_read(tmp_path) -> None:
    """Read from the run's own preflight record, so a geometry arm is a number in the output
    rather than a simplification nobody rechecked."""
    _write_preflight(tmp_path, margin=_SHIPPED_MARGIN)

    result = lag_kl.run_lag_kl_analysis(
        _context(_lag_block(truncated=False)), eval_config={}, output_dir=tmp_path, probe=None
    )

    support = result["lag_support"]
    assert support["measured"] is True
    assert support["lag_support_margin_steps"] == _SHIPPED_MARGIN
    assert support["profiles_agree"] is True
    assert support["computed_and_observed_agree"] is True


def test_the_analysis_runs_without_a_preflight_record(tmp_path) -> None:
    """``--only lag_kl --output-dir <a finished run>`` on a directory whose preflight record was
    never written is a re-run with less information, not a failure."""
    result = lag_kl.run_lag_kl_analysis(
        _context(_lag_block(truncated=False)), eval_config={}, output_dir=tmp_path, probe=None
    )

    assert result["lag_support"]["measured"] is False
    assert (tmp_path / lag_kl.ANALYSIS_DIRNAME / lag_kl.PROFILE_FILENAME).is_file()


def test_the_analysis_writes_every_table_and_carries_the_caveat(tmp_path) -> None:
    _write_preflight(tmp_path, margin=_SHIPPED_MARGIN)

    result = lag_kl.run_lag_kl_analysis(
        _context(_lag_block(truncated=False)), eval_config={}, output_dir=tmp_path, probe=None
    )

    directory = tmp_path / lag_kl.ANALYSIS_DIRNAME
    for name in (
        lag_kl.PROFILE_FILENAME, lag_kl.SUMMARY_FILENAME, lag_kl.PER_RECORDING_FILENAME,
        lag_kl.STRATIFIED_PROFILE_FILENAME, lag_kl.STRATIFIED_PEAKS_FILENAME,
        figure_filename(lag_kl.PROFILE_FIGURE),
    ):
        assert (directory / name).is_file(), name
    assert result["axis_caveat"] == lag_axis.GROUP_DELAY_CAVEAT
    summary = pd.read_csv(directory / lag_kl.SUMMARY_FILENAME)
    assert set(summary["axis_caveat"]) == {lag_axis.GROUP_DELAY_CAVEAT}
    assert {row["profile"] for row in result["peaks"]} == {
        name for name, _, _, _ in lag_kl.PROFILES
    }


# =================================================================================================
# The stratified reading
# =================================================================================================
def _stratified_inputs(n_lags: int = 6):
    """A per-sample table and its vector sidecar: two classes, two recordings each, two segments.

    The two classes are given profiles peaking at different lags, so a per-cohort argmax has a
    known answer and an implementation that pooled the cohorts would report neither.
    """
    from teb_vae.lag_attn_cfs.eval._reuse import labels

    rows, vectors = [], []
    for name, peak in (("healthy", 1), ("acidosis", 4)):
        for recording in range(2):
            for _segment in range(2):
                rows.append(
                    {
                        "guid": f"{name}_{recording}",
                        "epoch": -3600.0 * (1 + recording),
                        labels.CLASS_COLUMN: name,
                        labels.SUBGROUP_COLUMN: f"shard_{name}",
                        "source_conditioned_kl_raw": 1.0,
                    }
                )
                profile = np.full(n_lags, 0.1)
                profile[peak] = 1.0
                vectors.append(profile)
    stacked = np.asarray(vectors, dtype=np.float64)
    # Keyed by the *attribute* the collection pass writes into the vector sidecar, which is what
    # the analysis reads; the reported profile name is the other half of the same tuple.
    return pd.DataFrame(rows), {
        attribute: stacked.copy() for _, attribute, _ in lag_kl.STRATIFIED_PROFILES
    }


def test_the_per_cohort_profiles_are_emitted_at_full_lag_resolution() -> None:
    """The whole profile per cohort, not an argmax: two cohorts whose peaks coincide can still
    put very different amounts of mass near them, and only the profile says so."""
    per_sample, vectors = _stratified_inputs(n_lags=6)

    frame, skipped = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )

    class_rows = frame[frame["group_column"] == "clinical_class"]
    for name, _, _ in lag_kl.STRATIFIED_PROFILES:
        for group in ("healthy", "acidosis"):
            cell = class_rows[(class_rows["profile"] == name) & (class_rows["group"] == group)]
            assert len(cell) == 6, f"{name}/{group} was reduced rather than emitted"
    assert skipped == {}


def test_the_per_cohort_argmax_recovers_each_cohorts_own_peak() -> None:
    """Non-vacuity for the shape assertion above: a pooled implementation reports one peak for
    both cohorts, and this is where that shows."""
    per_sample, vectors = _stratified_inputs(n_lags=6)

    frame, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )
    peaks = {
        (row["group"], row["profile"]): row["argmax_lag_step"]
        for row in lag_kl.stratified_peak_rows(frame, delay_steps=0)
        if row["group_column"] == "clinical_class"
    }

    assert peaks[("healthy", "kl_untruncated")] == 1
    assert peaks[("acidosis", "kl_untruncated")] == 4


def test_each_cohort_row_counts_recordings_rather_than_segments() -> None:
    """Two recordings of two segments each must count as two, or a long recording decides the
    cohort's profile."""
    per_sample, vectors = _stratified_inputs()

    frame, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )

    assert set(frame["n_recordings"]) == {2}
    assert len(per_sample) == 8


def test_an_unlabelled_recording_does_not_become_a_cohort_named_after_the_absence() -> None:
    """The table is read back from CSV on every re-run, and that is where an absent class arrives
    as ``NaN`` rather than as ``None``. Stringifying before the null test turns it into a cohort
    literally called ``"nan"``, which also clears the single-cohort skip guard -- so a split with
    one real class reports a two-cohort comparison against a cohort that does not exist.
    """
    per_sample, vectors = _stratified_inputs()
    per_sample["clinical_class"] = ["healthy"] * 4 + [float("nan")] * 4

    frame, skipped = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )
    peaks = lag_kl.stratified_peak_rows(frame, delay_steps=0)

    assert "nan" not in set(frame["group"]), "the absence was emitted as a cohort"
    assert "nan" not in {row["group"] for row in peaks}
    assert "clinical_class" in skipped, "one real class is nothing to compare"
    assert "clinical_class" not in set(frame["group_column"])


def test_a_cohort_counts_only_the_recordings_that_measured_something() -> None:
    """A recording whose every segment scored no anchors is an all-``NaN`` row. The means skip it,
    so counting it in ``n_recordings`` labels the profile with evidence that never entered it.
    """
    per_sample, vectors = _stratified_inputs()
    unscored = per_sample["guid"] == "healthy_1"
    for attribute in vectors:
        vectors[attribute][np.asarray(unscored)] = np.nan

    frame, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )
    healthy = frame[(frame["group_column"] == "clinical_class") & (frame["group"] == "healthy")]
    acidosis = frame[(frame["group_column"] == "clinical_class") & (frame["group"] == "acidosis")]

    assert set(healthy["n_recordings"]) == {1}, "the unscored recording measured nothing"
    assert set(acidosis["n_recordings"]) == {2}, "and the untouched cohort is unaffected"


def test_the_time_axis_is_cut_on_the_same_grid_the_trajectory_uses() -> None:
    """One grid, defined one layer down, so two analyses cannot report windows that do not line
    up. The fixture's two epochs are an hour apart, which is two windows at 0.5 h."""
    from teb_vae.lag_attn_cfs.eval import cohort

    per_sample, vectors = _stratified_inputs()

    frame, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )
    windows = frame[frame["group_column"] == lag_kl.TIME_AXIS]

    assert sorted(set(windows["bin_center_h"])) == [1.25, 2.25]
    assert set(windows["group"]) == {"1.25 h", "2.25 h"}
    assert cohort.TRAJECTORY_BIN_HOURS == 0.5


def test_a_single_cohort_population_records_a_skip_on_that_axis() -> None:
    """One cohort is the pooled profile under another name, and on the healthy-only pretraining
    split that is the ordinary outcome rather than an error."""
    per_sample, vectors = _stratified_inputs()
    per_sample["clinical_class"] = "healthy"

    frame, skipped = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )

    assert "clinical_class" in skipped
    assert "nothing to compare" in skipped["clinical_class"]
    assert "clinical_class" not in set(frame["group_column"])


def test_the_per_head_vector_is_reshaped_head_major_and_dropped_when_it_does_not_factor() -> None:
    r"""The flattened vector is $M \cdot L$ head-major. A vector whose length does not factor is a
    mis-assembled profile rather than a short one, and reshaping it would produce a plausible
    wrong answer."""
    per_sample, vectors = _stratified_inputs(n_lags=6)
    heads = np.concatenate(
        [np.tile(np.eye(6)[2], (8, 1)), np.tile(np.eye(6)[5], (8, 1))], axis=1
    )
    vectors[lag_kl.PER_HEAD_ATTRIBUTE] = heads

    frame, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=2
    )
    per_head = frame[frame["profile"].str.startswith("attention_head_")]
    peaks = {
        (row["group"], row["profile"]): row["argmax_lag_step"]
        for row in lag_kl.stratified_peak_rows(frame, delay_steps=0)
        if row["group_column"] == "clinical_class"
    }

    assert set(per_head["head"]) == {0, 1}
    assert peaks[("healthy", "attention_head_0")] == 2
    assert peaks[("healthy", "attention_head_1")] == 5

    vectors[lag_kl.PER_HEAD_ATTRIBUTE] = heads[:, :-1]
    dropped, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=2
    )
    assert dropped[dropped["profile"].str.startswith("attention_head_")].empty


def test_the_stratified_peaks_carry_the_caveat_too() -> None:
    """The other artifact that states a lag position, and therefore the other one a reader could
    quote a physiological latency out of."""
    per_sample, vectors = _stratified_inputs()

    frame, _ = lag_kl.stratified_profiles(
        per_sample, vectors, delay_steps=0, n_lags=6, num_heads=0
    )
    peaks = lag_kl.stratified_peak_rows(frame, delay_steps=0)

    assert peaks
    assert {row["axis_caveat"] for row in peaks} == {lag_axis.GROUP_DELAY_CAVEAT}


# =================================================================================================
# On a real run
# =================================================================================================
@pytest.mark.slow
def test_the_identity_holds_on_a_real_run(collected_run) -> None:
    """End to end, on the run every other file in this suite questions."""
    checks = collected_run["summary"]["results"]["sanity"]["checks"]

    assert checks["lag_map_sums_to_kl"]["verdict"] == "pass", checks["lag_map_sums_to_kl"]
    assert checks["per_head_kl_sums_to_kl"]["verdict"] == "pass"
    assert checks["lag_map_sums_to_kl"]["max_abs_residual_nats"] >= 0.0


@pytest.mark.slow
def test_the_analysis_reports_the_measured_support_on_a_real_run(collected_run) -> None:
    r"""The whole point of the measurement: preflight's margin and the pass's own anchor counts,
    compared rather than assumed. The tiny fixture's geometry is not the shipped one, so what is
    asserted is the *agreement of the two readings*, never a particular margin."""
    block = collected_run["summary"]["results"]["lag_kl"]
    support = block["lag_support"]

    assert support["measured"] is True
    assert support["computed_and_observed_agree"] is True
    # Conditional, exactly as the analysis records it: the equality of the three profiles is a
    # consequence of a non-negative margin, not a property of this target domain.
    if support["lag_support_margin_steps"] >= 0:
        assert support["anchor_counts_uniform"] is True
        assert support["profiles_agree"] is True
    else:
        assert support["anchor_counts_uniform"] is False


@pytest.mark.slow
def test_the_real_run_writes_the_three_profiles_and_the_caveat(collected_run) -> None:
    directory = Path(collected_run["results_dir"]) / lag_kl.ANALYSIS_DIRNAME
    block = collected_run["summary"]["results"]["lag_kl"]
    profile = pd.read_csv(directory / lag_kl.PROFILE_FILENAME)

    assert (directory / figure_filename(lag_kl.PROFILE_FIGURE)).is_file()
    assert len(profile) == block["composition"]["n_lags"]
    assert profile["compensated_seconds"].tolist() == [
        float(lag_compensated_seconds(lag, delay_steps=block["delay_steps"]))
        for lag in range(len(profile))
    ]
    for _, _, column, _ in lag_kl.PROFILES:
        assert column in profile.columns
    assert block["identity"][_MAP_RESIDUAL] <= block["identity"]["tolerance_nats"]
    assert block["source_delay_is_max_over_channels"] is True
    assert block["axis_caveat"] == lag_axis.GROUP_DELAY_CAVEAT


@pytest.mark.slow
def test_the_real_run_stratifies_the_lag_readout_and_records_the_restriction(collected_run) -> None:
    """End to end: the fixture carries three classes and eight subgroups, so both cohort axes are
    cut, and the anchor restriction the untruncated profiles rest on travels with them."""
    block = collected_run["summary"]["results"]["lag_kl"]["stratified"]
    frame = pd.read_csv(
        Path(collected_run["results_dir"])
        / lag_kl.ANALYSIS_DIRNAME
        / lag_kl.STRATIFIED_PROFILE_FILENAME
    )

    assert set(block["axes"]) == {"clinical_class", "subgroup", lag_kl.TIME_AXIS}
    assert block["restricted_to_anchors_from"] == block["n_lags"] - 1
    # Full resolution per (axis, cohort, profile), never an argmax standing in for a profile.
    for _, cell in frame.groupby(["group_column", "group", "profile"]):
        assert len(cell) == block["n_lags"]
    assert {row["profile"] for row in block["peaks"]} >= {
        name for name, _, _ in lag_kl.STRATIFIED_PROFILES
    }
