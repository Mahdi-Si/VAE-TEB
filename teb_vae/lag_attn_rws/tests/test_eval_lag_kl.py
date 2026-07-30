r"""The per-lag KL attribution: the identity it rests on, and reading a profile without an argmax.

Two things are checked here and they fail in different ways.

**The identity.** $\sum_\ell \widetilde K_{t,\ell} = K_t$ holds anchor by anchor, and the one
mechanism that breaks it is dropout on the attention probabilities: dropout rescales them by
$1/(1-p)$ and zeroes a random subset, so each head's weights no longer sum to one and the
attribution holds only in expectation. Every number the analysis reports would still look
entirely reasonable. So the run measures the residual on its own numbers rather than inheriting
the property from a model test, and the tests below prove the measurement has teeth.

**The reading.** An argmax is not a reading of a profile. Ties resolve to the lowest index, and
``entmax15`` assigns lags exactly zero, so a profile that is flat or nearly empty still has a
perfectly confident argmax. The peak description and the mechanical degeneracy criterion are what
stop that from being reported as a finding, and they are tested on synthetic profiles of known
answer rather than on whatever the fixture happened to produce.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_rws.eval import report_seam
from teb_vae.lag_attn_rws.eval.analyses import lag_kl
from teb_vae.lag_attn_rws.eval.metrics import identity_residual_per_sample
from teb_vae.lag_attn_rws.nets.lag_report import lag_compensated_seconds
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import SHIPPED_KWARGS, TINY_KWARGS

#: The identity key the lag map's residual is reported under.
_MAP_RESIDUAL = "lag_map_sums_to_kl_max_abs_nats"


def _model_at_shipped_dropout(perturb_posterior) -> SeqVaeLagAttnRws:
    """A tiny model built at the **shipped** dropout, with its posterior moved off the prior.

    The shipped value is read from ``SHIPPED_KWARGS`` rather than written down, and the geometry
    stays tiny: what the tests below turn on is whether dropout is active anywhere near the
    attention, not how many anchors there are. At ``dropout=0.0`` -- the tiny default -- every
    assertion here would pass whether or not the identity is fragile at all, which is the trap
    this fixture exists to avoid.

    The perturbation is load-bearing: the delta heads are zero-initialised, so an unperturbed
    model has an identically zero KL and an identically zero lag map, and the identity holds
    vacuously on any model at all.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(
        **dict(TINY_KWARGS, dropout=SHIPPED_KWARGS["dropout"])
    ).eval()
    perturb_posterior(model)
    return model


def _worst_residual(model: SeqVaeLagAttnRws, inputs) -> float:
    """The worst per-anchor lag-map residual over a forward, on every anchor."""
    with torch.no_grad():
        outputs = model(*inputs)
    assert float(outputs["kld_per_t"].abs().max()) > 0.0, "a zero KL would agree vacuously"
    support = torch.ones_like(outputs["kld_per_t"])
    return float(
        identity_residual_per_sample(
            outputs["source_kl_lag_map"], outputs["kld_per_t"], support
        ).max()
    )


# =============================================================================
# The identity, at the shipped dropout
# =============================================================================
def test_the_attention_carries_no_dropout_even_at_the_shipped_setting(perturb_posterior) -> None:
    """The structural reason the identity holds, pinned at the line that provides it.

    The obvious expectation reads the other way -- identity under ``eval()``, violation under
    ``train()`` at ``dropout: 0.1`` -- and it is wrong about this model. ``SeqVaeLagAttnRws``
    builds ``LagCrossAttention`` at ``dropout=0.0`` unconditionally, so the model's ``dropout``
    kwarg never reaches the attention probabilities and the identity holds in both modes. That is
    the stronger property, and this is the assertion that keeps it true: a change to that
    constructor argument fails here rather than silently making every lag number hold only in
    expectation.
    """
    model = _model_at_shipped_dropout(perturb_posterior)

    assert float(SHIPPED_KWARGS["dropout"]) > 0.0, "a zero shipped dropout makes this vacuous"
    assert model.lag_attn.attn_dropout.p == 0.0


def test_the_identity_holds_in_both_modes_at_the_shipped_dropout(
    perturb_posterior, inputs
) -> None:
    """Both modes, because the eval path's correctness must not depend on remembering ``eval()``.

    It does not, for the reason the test above pins -- and asserting it here is what would catch
    the day it starts to.
    """
    model = _model_at_shipped_dropout(perturb_posterior)

    evaluating = _worst_residual(model, inputs)
    model.train()
    training = _worst_residual(model, inputs)

    assert evaluating <= report_seam.IDENTITY_TOLERANCE
    assert training <= report_seam.IDENTITY_TOLERANCE


def test_dropout_on_the_attention_probabilities_breaks_the_identity(
    perturb_posterior, inputs
) -> None:
    """The measurement's teeth, produced the only way this model permits.

    The attention module is constructed at zero dropout, so the failure mode has to be introduced
    on the built module: this is precisely the regression a future edit to that constructor
    argument would ship, and the check must see it. Dropout applies only in training mode, so the
    module is put there while the rest of the model stays in ``eval()`` -- isolating the cause to
    the attention probabilities rather than to a change of mode.
    """
    model = _model_at_shipped_dropout(perturb_posterior)
    model.lag_attn.attn_dropout.p = float(SHIPPED_KWARGS["dropout"])
    model.lag_attn.train()

    residual = _worst_residual(model, inputs)

    assert residual > report_seam.IDENTITY_TOLERANCE * 100.0, (
        "attention dropout must show up as a gross violation, not a marginal one"
    )


def test_a_violated_identity_names_dropout_as_the_likely_cause() -> None:
    """A residual with no explanation beside it costs an afternoon of reading the wrong module."""
    record = report_seam.check_lag_identity(
        {
            "lag": {"identity_residuals": {_MAP_RESIDUAL: 0.5}},
            "readouts": {"source_conditioned_kl_raw": 1.0},
        },
        "lag_map_sums_to_kl",
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


def test_the_identity_holds_on_a_real_run(evaluated) -> None:
    """End to end, on the run every other file in this suite questions."""
    checks = evaluated["summary"]["results"]["sanity"]["checks"]

    assert checks["lag_map_sums_to_kl"]["verdict"] == "pass", checks["lag_map_sums_to_kl"]
    assert checks["per_head_kl_sums_to_kl"]["verdict"] == "pass"
    assert checks["lag_map_sums_to_kl"]["max_abs_residual_nats"] >= 0.0


# =============================================================================
# Reading a profile
# =============================================================================
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
    profile = [0.0] * 19 + [1.0]

    verdict = lag_kl.degeneracy(profile)

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


# =============================================================================
# The lag axis
# =============================================================================
def test_the_seconds_axis_is_the_compensated_one_elementwise() -> None:
    r"""Not $4\ell$, and not $4\ell + 20$: the compensated figure $4(\ell + \delta)$, built
    through the shared converter so a second implementation cannot drift from it."""
    axis = lag_kl.compensated_seconds_axis(9, delay_steps=0)

    assert axis.tolist() == [
        float(lag_compensated_seconds(lag, delay_steps=0)) for lag in range(9)
    ]


def test_a_nonzero_input_delay_shifts_every_second_by_four_delta() -> None:
    """The historical bug, in the shape it took: one consumer read the delay and the other read
    zero, and the two reports of one run disagreed by two minutes with nothing raising."""
    delay = 30
    base = lag_kl.compensated_seconds_axis(9, delay_steps=0)

    shifted = lag_kl.compensated_seconds_axis(9, delay_steps=delay)

    assert np.allclose(shifted - base, 4.0 * delay)


# =============================================================================
# The figure
# =============================================================================
def _figure_lines(figure):
    """The drawn curves of the top panel, by label."""
    axis = figure.axes[0]
    return {line.get_label(): line for line in axis.get_lines()}


def test_the_profile_figure_draws_against_the_compensated_seconds_axis() -> None:
    """The axis label names the quantity; these assertions are about the *numbers*, which is what
    was historically wrong."""
    import matplotlib.pyplot as plt

    lag = {
        "kl_lag_profile": [0.1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0],
        "kl_lag_profile_support_corrected": [0.1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0],
        "kl_lag_anchor_counts": [9.0] * 9,
        "kl_argmax_lag_step": 1,
    }
    seconds = lag_kl.compensated_seconds_axis(9, delay_steps=0)
    profile = lag_kl.profile_frame(lag, seconds)

    figure = lag_kl.build_profile_figure(profile, lag, delay_steps=0, n_lags=9)
    try:
        drawn = _figure_lines(figure)["raw attribution (sums to the KL)"]
        x_values = np.asarray(drawn.get_xdata(), dtype=float)
        label = figure.axes[0].get_xlabel()
    finally:
        plt.close(figure)

    assert np.allclose(x_values, seconds)
    assert "compensated" in label


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


# =============================================================================
# The analysis, on a real run
# =============================================================================
def test_the_analysis_writes_its_tables_and_reports_the_identity(evaluated) -> None:
    """Its output on the run every other file here questions."""
    import pandas as pd

    block = evaluated["summary"]["results"]["lag_kl"]
    directory = evaluated["results_dir"] / lag_kl.ANALYSIS_DIRNAME
    profile = pd.read_csv(directory / lag_kl.PROFILE_FILENAME)

    assert (directory / lag_kl.PROFILE_FIGURE).is_file()
    assert len(profile) == block["composition"]["n_lags"]
    assert profile["compensated_seconds"].tolist() == [
        float(lag_compensated_seconds(lag, delay_steps=block["delay_steps"]))
        for lag in range(len(profile))
    ]
    assert block["identity"][_MAP_RESIDUAL] <= block["identity"]["tolerance_nats"]
    # Every reported lag says what it was compensated by, and that the delay is an upper bound.
    assert block["source_delay_is_max_over_channels"] is True
    assert {row["profile"] for row in block["peaks"]} == {"raw", "support_corrected"}


# =============================================================================
# The stratified reading
#
# No pipeline before this one cut the lag readout by anything at all. What is asserted here is
# that the cut keeps the whole profile rather than reducing it to an argmax, that it travels the
# per-recording chain, and that a population with one cohort is a recorded skip rather than a
# one-cohort "comparison".
# =============================================================================
def _stratified_inputs(n_lags: int = 6):
    """A per-sample table and its vector sidecar: two classes, two recordings each, two segments.

    The two classes are given profiles peaking at different lags, so a per-cohort argmax has a
    known answer and an implementation that pooled the cohorts would report neither.
    """
    import pandas as pd

    from teb_vae.lag_attn_rws.eval._reuse import labels

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
        attribute: stacked.copy()
        for _, attribute, _ in lag_kl.STRATIFIED_PROFILES
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
    from teb_vae.lag_attn_rws.eval import cohort

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


def test_the_real_run_stratifies_the_lag_readout_and_records_the_restriction(evaluated) -> None:
    """End to end: the fixture carries three classes and four subgroups, so both cohort axes are
    cut, and the anchor restriction the untruncated profiles rest on travels with them."""
    import pandas as pd

    block = evaluated["summary"]["results"]["lag_kl"]["stratified"]
    frame = pd.read_csv(
        evaluated["results_dir"] / lag_kl.ANALYSIS_DIRNAME / lag_kl.STRATIFIED_PROFILE_FILENAME
    )

    assert set(block["axes"]) == {"clinical_class", "subgroup", lag_kl.TIME_AXIS}
    assert block["restricted_to_anchors_from"] == block["n_lags"] - 1
    # Full resolution per (axis, cohort, profile), never an argmax standing in for a profile.
    for _, cell in frame.groupby(["group_column", "group", "profile"]):
        assert len(cell) == block["n_lags"]
    assert {row["profile"] for row in block["peaks"]} >= {
        name for name, _, _ in lag_kl.STRATIFIED_PROFILES
    }
