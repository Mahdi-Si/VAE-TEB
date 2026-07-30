r"""Whether the forecast is any good: the baselines, the skill arithmetic, and the horizon axis.

Four kinds of assertion, in the order they matter.

**Known answers.** Persistence on a constant signal has exactly zero error; climatology on
standard-normal input scores the analytic Gaussian value; a forecast equal to the truth scores
skill $1$; one equal to the baseline scores exactly $0$. These pin the arithmetic itself, without
a model and without a fixture, and they are what would catch a skill score computed the wrong way
round.

**Exactness of the decomposition.** The horizon-resolved score must sum over $\tau$ back to the
per-anchor block score the objective uses. It does so by construction -- both reduce the same
elementwise term -- and the test is what keeps that true if either reduction is rewritten. In
float64, because in float32 the two reduction orders differ in the last bits and a tolerance loose
enough to admit that is loose enough to admit a real error.

**The denominators.** The per-$\tau$ score divides by the per-$\tau$ masked anchor count and not
by the per-anchor contributing indicator, which is an ``amax`` over $\tau$. The difference only
shows up near gaps, in the direction that flatters late horizons, so the test builds a gap.

**Signs, on real checkpoints.** At a perturbed random init the model forecasts what climatology
does, so its skill is indistinguishable from zero; after a short fit on data with something to
learn, it is not. Those two are the `slow` end of the file and they are what says the whole
apparatus measures a model rather than a convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval.analyses import forecast as forecast_analysis
from teb_vae.lag_attn_rws.eval.metrics import (
    BASELINE_LOGVAR,
    BASELINE_NAMES,
    baseline_forecasts,
    horizon_block_sums,
    masked_raw_block_per_horizon_step,
    masked_raw_error_sums,
)
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

#: The tiny geometry, matching ``TINY_KWARGS``: $T = 16$, $H = 4$, $R = 16$, so $T_{valid} = 12$
#: and a block is $64$ raw samples. Built directly rather than through a model: none of the
#: arithmetic under test needs one, and constructing one would make every case below slower and
#: none of them stronger.
GEOMETRY = TrimmedRawGeometry(raw_len=256, decimation=16, horizon=4, warmup=2)

#: $\tfrac{1}{2}\log 2\pi$ -- half the Gaussian constant, so the analytic climatology score below
#: is written the way it is derived.
HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _fixture(fhr_raw: torch.Tensor, weight: torch.Tensor):
    """Return ``(target, mask)`` for a raw signal and its validity, at :data:`GEOMETRY`."""
    target = build_future_target(fhr_raw, GEOMETRY)
    mask, _coverage = forecast_mask(weight, GEOMETRY)
    return target, mask


def _constant_signal(value: float = 137.0, batch: int = 2):
    """A perfectly flat raw signal with every step valid."""
    fhr_raw = torch.full((batch, GEOMETRY.raw_len), float(value), dtype=torch.float64)
    weight = torch.ones(batch, GEOMETRY.t, dtype=torch.float64)
    return fhr_raw, weight


# =============================================================================
# The baselines, against known answers
# =============================================================================
def test_persistence_on_a_constant_signal_forecasts_it_exactly() -> None:
    """The whole point of a persistence baseline: on a signal that does not move it is right."""
    fhr_raw, weight = _constant_signal()
    target, mask = _fixture(fhr_raw, weight)

    persistence = baseline_forecasts(fhr_raw, weight, GEOMETRY)["persistence"]
    sums = masked_raw_error_sums(persistence, target, mask)

    assert float(sums["sum_sq"].sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(sums["n_raw"].sum()) > 0.0, "an all-masked fixture would agree vacuously"


def test_climatology_on_standard_normal_input_scores_the_analytic_value() -> None:
    r"""With $\mu = 0$ and $\sigma = 1$ the per-sample score is
    $\tfrac{1}{2}(\log 2\pi + x^2)$, so a block of $H R$ standard-normal samples scores
    $H R \cdot \tfrac{1}{2}(\log 2\pi + 1)$ in expectation. Asserted loosely because it is an
    expectation over a finite draw -- but far tighter than the difference between this and any
    other plausible convention (a mean instead of a sum is $64\times$ smaller; dropping the
    constant moves it by $59$)."""
    generator = torch.Generator().manual_seed(7)
    fhr_raw = torch.randn(16, GEOMETRY.raw_len, generator=generator, dtype=torch.float64)
    weight = torch.ones(16, GEOMETRY.t, dtype=torch.float64)
    target, mask = _fixture(fhr_raw, weight)

    climatology = baseline_forecasts(fhr_raw, weight, GEOMETRY)["climatology"]
    block, contributing = masked_raw_block_per_anchor(
        climatology, target, mask,
        likelihood="gaussian_nll", logvar=torch.tensor(BASELINE_LOGVAR, dtype=torch.float64),
    )
    scored = float(block.sum() / contributing.sum())
    block_size = GEOMETRY.horizon * GEOMETRY.r

    assert scored == pytest.approx(block_size * (HALF_LOG_2PI + 0.5), rel=0.05)


def test_the_segment_mean_baseline_is_the_mean_of_the_observed_samples() -> None:
    """It reads the whole segment, including its future -- which is why it is the strong baseline
    and why the docstring says so rather than leaving a reader to assume it is causal."""
    fhr_raw = torch.arange(GEOMETRY.raw_len, dtype=torch.float64)[None, :].repeat(2, 1)
    weight = torch.ones(2, GEOMETRY.t, dtype=torch.float64)

    segment_mean = baseline_forecasts(fhr_raw, weight, GEOMETRY)["segment_mean"]

    assert float(segment_mean.reshape(-1)[0]) == pytest.approx(float(fhr_raw[0].mean()))


def test_persistence_carries_the_last_observed_sample_across_a_gap() -> None:
    r"""A gap is stored as $0$ bpm, which after z-scoring is roughly $-11\sigma$ and is not a
    detectable sentinel. Carrying *that* forward would not measure persistence, it would measure
    the gap -- and it would make the baseline arbitrarily bad wherever the signal is worst."""
    fhr_raw, weight = _constant_signal(batch=1)
    gap_step = 5
    fhr_raw[:, gap_step * GEOMETRY.decimation : (gap_step + 1) * GEOMETRY.decimation] = 0.0
    weight[:, gap_step] = 0.0

    persistence = baseline_forecasts(fhr_raw, weight, GEOMETRY)["persistence"].reshape(-1)

    # The anchor *after* the gap must reuse the last valid step before it, not the gap's zeros.
    assert float(persistence[gap_step + 1]) == pytest.approx(137.0)
    assert float(persistence[gap_step]) == pytest.approx(137.0)


def test_every_named_baseline_is_produced() -> None:
    """A name in the registry with no forecast behind it would report a skill against nothing."""
    fhr_raw, weight = _constant_signal()

    assert set(baseline_forecasts(fhr_raw, weight, GEOMETRY)) == set(BASELINE_NAMES)


# =============================================================================
# The skill arithmetic
# =============================================================================
def test_a_forecast_equal_to_the_truth_scores_skill_one() -> None:
    model = np.array([0.0, 0.0, 0.0, 0.0])
    baseline = np.array([1.0, 4.0, 0.25, 9.0])

    assert forecast_analysis.skill_against(model, baseline) == pytest.approx([1.0] * 4)


def test_a_forecast_equal_to_the_baseline_scores_skill_exactly_zero() -> None:
    values = np.array([1.0, 4.0, 0.25])

    assert forecast_analysis.skill_against(values, values) == pytest.approx([0.0] * 3)


def test_a_zero_error_baseline_yields_no_skill_rather_than_an_infinity() -> None:
    """A baseline that is exactly right on a degenerate recording would otherwise report an
    infinite skill as evidence."""
    skill = forecast_analysis.skill_against(np.array([1.0, 0.5]), np.array([0.0, 1.0]))

    assert np.isnan(skill[0]) and skill[1] == pytest.approx(0.5)


def _per_guid_frame(model_sq: float, model_block: float, *, n: int = 6) -> pd.DataFrame:
    """A per-recording frame where every baseline scores 1.0 and the model scores what is asked."""
    columns = {
        "sq_error_base": model_sq, "sq_error_full": model_sq,
        "nll_base_block": model_block, "nll_full_block": model_block,
        "abs_error_base": 0.0, "abs_error_full": 0.0,
        "signed_error_base": 0.0, "signed_error_full": 0.0,
    }
    for name in BASELINE_NAMES:
        columns[f"sq_error_{name}"] = 1.0
        columns[f"nll_{name}_block"] = 10.0
    return pd.DataFrame({name: [value] * n for name, value in columns.items()})


def test_the_skill_table_reports_one_row_per_branch_and_baseline_with_its_n() -> None:
    rows = forecast_analysis.build_skill_rows(
        _per_guid_frame(0.0, 0.0), resamples=200, seed=0
    )

    assert len(rows) == len(forecast_analysis.MODEL_BRANCHES) * len(BASELINE_NAMES)
    assert {row["baseline"] for row in rows} == set(BASELINE_NAMES)
    assert all(row["n_recordings"] == 6 for row in rows)


def test_a_perfect_forecast_scores_one_against_every_baseline_with_a_finite_interval() -> None:
    rows = forecast_analysis.build_skill_rows(
        _per_guid_frame(0.0, 0.0), resamples=200, seed=0
    )

    assert all(row["mse_skill"] == pytest.approx(1.0) for row in rows)
    assert all(
        np.isfinite(row["mse_skill_lo"]) and np.isfinite(row["mse_skill_hi"]) for row in rows
    )
    # And the log-score column is a difference, so it is the baseline's score, not a ratio.
    assert all(row["advantage_nats_per_anchor"] == pytest.approx(10.0) for row in rows)


def test_a_forecast_equal_to_the_baselines_scores_zero_in_both_spaces() -> None:
    """Both columns at once: it is the case where the ratio and the difference agree, and a skill
    computed as ``ratio - 1`` rather than ``1 - ratio`` would pass one and fail the other."""
    rows = forecast_analysis.build_skill_rows(
        _per_guid_frame(1.0, 10.0), resamples=200, seed=0
    )

    assert all(row["mse_skill"] == pytest.approx(0.0) for row in rows)
    assert all(row["advantage_nats_per_anchor"] == pytest.approx(0.0) for row in rows)


def test_the_r2_reference_is_one_of_a_closed_set() -> None:
    """An $R^2$ whose reference is implicit is a claim a reader cannot check: against the segment
    mean and against climatology it is a different number."""
    assert forecast_analysis.R2_REFERENCE in forecast_analysis.R2_REFERENCES
    assert set(forecast_analysis.R2_REFERENCES) <= set(BASELINE_NAMES)


def test_the_rmse_roots_once_rather_than_averaging_finished_roots() -> None:
    r"""Jensen: $\operatorname{mean}(\sqrt{x}) \le \sqrt{\operatorname{mean}(x)}$, so averaging
    per-segment RMSEs is biased **low** -- in the direction that flatters the model. The frame
    below has widely different per-recording squared errors, which is where the two differ; it
    carries four recordings because a bootstrap over fewer than three reports a note instead of an
    interval."""
    frame = pd.DataFrame(
        {
            "sq_error_base": [1.0, 9.0, 1.0, 9.0], "sq_error_full": [1.0, 9.0, 1.0, 9.0],
            "abs_error_base": [1.0, 3.0, 1.0, 3.0], "abs_error_full": [1.0, 3.0, 1.0, 3.0],
            "signed_error_base": [0.0] * 4, "signed_error_full": [0.0] * 4,
        }
    )

    rows = forecast_analysis.build_error_rows(frame, None, resamples=200, seed=0)

    # sqrt(mean(1, 9)) = sqrt(5) = 2.236..., not mean(sqrt(1), sqrt(9)) = 2.0.
    assert rows[0]["rmse_normalised"] == pytest.approx(float(np.sqrt(5.0)))
    assert rows[0]["rmse_normalised"] > 2.0
    assert rows[0]["unit"] == "normalised"


# =============================================================================
# The horizon axis
# =============================================================================
def test_the_per_horizon_score_sums_over_tau_to_the_per_anchor_block_score() -> None:
    """The identity that makes the horizon curve a decomposition of the headline rather than a
    second quantity. In float64: the two reduction orders differ in float32's last bits, and a
    tolerance loose enough for that is loose enough to hide a real disagreement."""
    generator = torch.Generator().manual_seed(3)
    fhr_raw = torch.randn(3, GEOMETRY.raw_len, generator=generator, dtype=torch.float64)
    weight = torch.ones(3, GEOMETRY.t, dtype=torch.float64)
    weight[:, 7] = 0.0
    target, mask = _fixture(fhr_raw, weight)
    mu = torch.randn(target.shape, generator=generator, dtype=torch.float64) * 0.1
    logvar = torch.randn(target.shape, generator=generator, dtype=torch.float64) * 0.1

    per_tau = masked_raw_block_per_horizon_step(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar
    )
    block, _contributing = masked_raw_block_per_anchor(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar
    )

    assert per_tau.shape == (3, GEOMETRY.t_valid, GEOMETRY.horizon)
    assert torch.allclose(per_tau.sum(dim=2), block, rtol=1e-12, atol=1e-12)
    assert float(block.abs().sum()) > 0.0, "an all-zero fixture would agree vacuously"


def test_the_horizon_denominator_is_per_tau_and_not_the_per_anchor_indicator() -> None:
    r"""The per-anchor indicator is an ``amax`` over $\tau$, so an anchor whose late steps fall in
    a gap still counts as contributing. Dividing by it would average those steps' structural zeros
    into the late horizons and make them read better exactly where the signal is worst."""
    fhr_raw, weight = _constant_signal(batch=1)
    # One gapped decimated step: it is the forecast step of several anchors, at different tau.
    weight[:, 9] = 0.0
    target, mask = _fixture(fhr_raw, weight)
    mu = torch.zeros_like(target)
    logvar = torch.zeros_like(target)

    sums = horizon_block_sums(mu, logvar, target, mask, likelihood="gaussian_nll")
    per_tau_counts = np.asarray(sums["n_anchors"])
    contributing = float((mask.amax(dim=-1) > 0).sum())

    assert per_tau_counts.shape == (GEOMETRY.horizon,)
    assert per_tau_counts.min() < contributing, (
        "the per-tau counts must be strictly smaller than the per-anchor count wherever the gap "
        "removed a forecast step; equality means the wrong denominator is in use"
    )
    assert np.asarray(sums["sum_block"]).shape == (GEOMETRY.horizon,)


def _horizon_record(n_steps: int = 4) -> dict:
    """A hand-built accumulator record with distinct, checkable per-step values."""
    return {
        "base_sum_block": [10.0, 20.0, 30.0, 40.0][:n_steps],
        "base_n_anchors": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "full_sum_block": [8.0, 18.0, 30.0, 44.0][:n_steps],
        "full_n_anchors": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "base_sum_sq": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "base_count": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "full_sum_sq": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "full_count": [5.0, 5.0, 4.0, 2.0][:n_steps],
    }


def test_the_horizon_curve_divides_each_step_by_its_own_denominator() -> None:
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["d_base_nats"]) == pytest.approx([2.0, 4.0, 7.5, 20.0])
    assert list(curves["d_full_nats"]) == pytest.approx([1.6, 3.6, 7.5, 22.0])


def test_the_horizon_gap_is_the_difference_of_the_two_scores() -> None:
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["gap_nats"]) == pytest.approx(
        list(np.asarray(curves["d_base_nats"]) - np.asarray(curves["d_full_nats"])), abs=1e-12
    )


def test_the_horizon_axis_is_lead_time_in_seconds_not_step_index() -> None:
    r"""Horizon step $\tau$ reads decimated step $t + 1 + \tau$, so its lead time ends at
    $4(\tau + 1)$ seconds: step $0$ is four seconds ahead, not zero."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["lead_seconds"]) == pytest.approx([4.0, 8.0, 12.0, 16.0])


def test_the_horizon_curve_names_the_path_it_was_computed_on() -> None:
    """The marginalisation does not commute with the sum over $\\tau$, so the curve is the
    single-draw one and has to say so rather than leaving it to be inferred."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert set(curves["score_path"]) == {"single-draw (training path)"}


def test_an_absent_horizon_block_yields_no_curve_rather_than_an_invented_one() -> None:
    assert forecast_analysis.horizon_curves({}).empty
    assert forecast_analysis.horizon_curves({"base_sum_block": [1.0]}).empty


def test_the_horizon_rmse_converts_through_the_spread_path() -> None:
    """An RMSE is a difference of levels, so the mean does not enter it. Through the level map a
    unit RMSE would come back as 140 bpm rather than as 10."""
    curves = forecast_analysis.horizon_curves(
        _horizon_record(), normalization={"fhr": {"mean": 140.0, "std": 10.0}}
    )

    assert list(curves["rmse_base"]) == pytest.approx([10.0] * 4, rel=1e-6)
    assert set(curves["rmse_unit"]) == {"bpm"}


# =============================================================================
# The figures
# =============================================================================
def test_the_horizon_figure_spans_the_whole_forecast_window_in_seconds(shipped_kwargs) -> None:
    """$[0, 120]$ seconds rather than $[0, 30]$ steps, on every panel: a reader who has to
    multiply by four is a reader who will eventually forget to."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    horizon = int(shipped_kwargs["horizon"])
    curves = forecast_analysis.horizon_curves(_horizon_record())

    figure = forecast_analysis.build_horizon_figure(curves, horizon_steps=horizon)
    try:
        limits = [axis.get_xlim() for axis in figure.axes]
    finally:
        shared_figures.plt.close(figure)

    assert limits == [(0.0, 120.0)] * len(limits)
    assert len(limits) == 3


def test_the_horizon_figures_gap_line_is_the_recomputed_difference() -> None:
    """Asserted on the drawn data, not on the frame it came from: a panel plotting the wrong
    column would pass every table assertion above."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    curves = forecast_analysis.horizon_curves(_horizon_record())
    expected = np.asarray(curves["d_base_nats"]) - np.asarray(curves["d_full_nats"])

    figure = forecast_analysis.build_horizon_figure(curves, horizon_steps=4)
    try:
        drawn = np.asarray(figure.axes[1].lines[0].get_ydata(), dtype=np.float64)
    finally:
        shared_figures.plt.close(figure)

    assert drawn == pytest.approx(expected, abs=1e-6)


def test_the_anchor_profile_shades_the_two_regions_the_geometry_defines(tiny_kwargs) -> None:
    r"""The warm-up prefix $[0, w)$ carries no loss term and the tail $[T - H, T)$ holds anchors
    whose forecast window runs past the end of the segment. Both bounds come from the model's own
    geometry, so a geometry change fails this test instead of silently mis-shading."""
    from teb_vae.lag_attn.eval import figures as shared_figures
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    geometry = SeqVaeLagAttnRws(**tiny_kwargs).geometry
    record = {
        "warmup": geometry.warmup, "t_valid": geometry.t_valid, "t": geometry.t,
    }
    profile = pd.DataFrame(
        {
            "anchor": np.arange(geometry.t_valid),
            "nll_base_block": np.linspace(1.0, 2.0, geometry.t_valid),
            "nll_full_block": np.linspace(1.0, 2.0, geometry.t_valid),
            "pred_gap": np.zeros(geometry.t_valid),
        }
    )

    figure, spans = forecast_analysis.build_anchor_profile_figure(profile, record)
    try:
        # Read off the rectangle's own x extent rather than off its path: ``axvspan`` draws under
        # a blended transform, so the path is a unit square and carries none of the data
        # coordinates the shading actually covers.
        drawn = [
            (float(patch.get_x()), float(patch.get_x() + patch.get_width()))
            for patch in figure.axes[0].patches
        ]
    finally:
        shared_figures.plt.close(figure)

    assert spans["warmup"] == (0.0, float(geometry.warmup))
    assert spans["untrained_tail"] == (float(geometry.t_valid), float(geometry.t))
    assert drawn == [spans["warmup"], spans["untrained_tail"]]


# =============================================================================
# The overlay, whose retention is opt-in
# =============================================================================
def _synthetic_context(*, retained=None, normalization=None):
    """An analysis context built by hand, with no model and no collection pass.

    Waveform retention is opt-in -- ``eval_config.caps.waveforms`` -- so a run that did not ask
    for it retains nothing and the overlay is never reached. Driving the analysis directly is what
    exercises both branches without paying for two more end-to-end evaluations.
    """
    import types

    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    n_recordings, n_anchors = 4, GEOMETRY.t_valid
    per_sample = pd.DataFrame({"guid": [f"g{index}" for index in range(n_recordings)]})
    for branch in ("base", "full", *BASELINE_NAMES):
        per_sample[f"nll_{branch}_block"] = np.linspace(10.0, 13.0, n_recordings)
        per_sample[f"sq_error_{branch}"] = np.linspace(1.0, 4.0, n_recordings)
    for branch in ("base", "full"):
        per_sample[f"abs_error_{branch}"] = 0.5
        per_sample[f"signed_error_{branch}"] = 0.0
    per_anchor = pd.DataFrame(
        {
            "anchor": np.tile(np.arange(n_anchors), n_recordings),
            "nll_base_block": 1.0, "nll_full_block": 0.9, "pred_gap": 0.1,
        }
    )
    record = {
        "geometry": {
            "raw_len": GEOMETRY.raw_len, "decimation": GEOMETRY.decimation,
            "horizon": GEOMETRY.horizon, "warmup": GEOMETRY.warmup,
            "t": GEOMETRY.t, "t_valid": GEOMETRY.t_valid, "raw_per_step": GEOMETRY.r,
        },
        "normalization": normalization or {},
        "horizon": _horizon_record(GEOMETRY.horizon),
        "likelihood": "gaussian_nll",
    }
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=per_anchor, record=record, retained=retained or {}
    )
    return AnalysisContext(collection=collection, config={})


def _retained_waveforms(rows: int = 2) -> dict:
    """Retained forecast blocks shaped as the collection pass keeps them."""
    shape = (rows, GEOMETRY.t_valid, GEOMETRY.horizon, GEOMETRY.r)
    return {
        "target": np.zeros(shape),
        "mu_base": np.full(shape, 0.5),
        "mu_full": np.full(shape, -0.5),
        "waveforms_sample_index": np.arange(rows),
    }


def test_no_waveform_retained_means_no_overlay_rather_than_an_empty_page(tmp_path) -> None:
    """Retention is opt-in because the three tensors this figure needs are two megabytes per
    sample. A run that did not ask for them has not failed."""
    result = forecast_analysis.run_forecast_analysis(
        _synthetic_context(), eval_config={"bootstrap_resamples": 200, "seed": 0},
        output_dir=tmp_path, probe=None,
    )

    assert forecast_analysis.OVERLAY_FIGURE not in result["files"]
    directory = tmp_path / forecast_analysis.ANALYSIS_DIRNAME
    assert not (directory / forecast_analysis.OVERLAY_FIGURE).exists()


def test_a_retained_waveform_is_drawn_and_recorded(tmp_path) -> None:
    result = forecast_analysis.run_forecast_analysis(
        _synthetic_context(retained=_retained_waveforms()),
        eval_config={"bootstrap_resamples": 200, "seed": 0},
        output_dir=tmp_path, probe=None,
    )

    assert forecast_analysis.OVERLAY_FIGURE in result["files"]
    assert (
        tmp_path / forecast_analysis.ANALYSIS_DIRNAME / forecast_analysis.OVERLAY_FIGURE
    ).is_file()


def test_the_overlay_draws_the_forecast_in_bpm_against_lead_time_in_seconds() -> None:
    """The one figure that shows a *level*, so it is the one that takes the affine conversion --
    and the only one where the spread conversion would be the error."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure = forecast_analysis.build_overlay_figure(
        _retained_waveforms(),
        row=0, anchor=GEOMETRY.warmup,
        geometry={"decimation": GEOMETRY.decimation},
        normalization={"fhr": {"mean": 140.0, "std": 10.0}},
    )
    try:
        axis = figure.axes[0]
        lines = {line.get_label(): np.asarray(line.get_ydata()) for line in axis.lines}
        x = np.asarray(axis.lines[0].get_xdata(), dtype=np.float64)
    finally:
        shared_figures.plt.close(figure)

    # target 0 -> 140 bpm, mu_base +0.5 -> 145, mu_full -0.5 -> 135.
    assert lines["truth"] == pytest.approx(np.full(x.size, 140.0), rel=1e-6)
    assert lines["target-only (base)"] == pytest.approx(np.full(x.size, 145.0), rel=1e-6)
    assert lines["source-conditioned (full)"] == pytest.approx(np.full(x.size, 135.0), rel=1e-6)
    # 480 raw samples at 4 Hz on the production grid; H * R samples of 0.25 s here.
    assert x.size == GEOMETRY.horizon * GEOMETRY.r
    assert float(x[-1]) == pytest.approx(GEOMETRY.horizon * 4.0)


# =============================================================================
# On a real run
# =============================================================================
def test_the_analysis_emits_both_skill_spaces_and_records_the_baseline_sigma(evaluated) -> None:
    """A learned-variance model beats a fixed-variance baseline partly on variance modelling
    alone, so the NLL-space number alone cannot be read; and which $\\sigma$ the baselines were
    given decides the whole of it, so it travels in the output rather than in a docstring."""
    block = evaluated["summary"]["results"]["forecast"]

    assert block["baseline_logvar"] == pytest.approx(BASELINE_LOGVAR)
    assert block["baselines"] == list(BASELINE_NAMES)
    for row in block["skill"]:
        assert "mse_skill" in row and "advantage_nats_per_anchor" in row


def test_the_analysis_writes_its_tables_and_figures(evaluated) -> None:
    directory = evaluated["results_dir"] / forecast_analysis.ANALYSIS_DIRNAME

    for name in (
        forecast_analysis.SCORES_FILENAME,
        forecast_analysis.SKILL_FILENAME,
        forecast_analysis.HORIZON_FILENAME,
        forecast_analysis.ANCHOR_FILENAME,
        forecast_analysis.BASELINE_FIGURE,
        forecast_analysis.ANCHOR_FIGURE,
        forecast_analysis.HORIZON_FIGURE,
    ):
        assert (directory / name).is_file(), name
    # Every row is a recording, not a segment: the fixture has more segments than recordings.
    scores = pd.read_csv(directory / forecast_analysis.SCORES_FILENAME)
    assert len(scores) == evaluated["summary"]["results"]["n_recordings"]
    assert len(scores) < evaluated["summary"]["results"]["n_samples"]


def test_the_horizon_curve_has_one_point_per_forecast_step(evaluated) -> None:
    curves = pd.read_csv(
        evaluated["results_dir"]
        / forecast_analysis.ANALYSIS_DIRNAME
        / forecast_analysis.HORIZON_FILENAME
    )
    geometry = evaluated["summary"]["collection"]["geometry"]

    assert len(curves) == int(geometry["horizon"])
    assert curves["lead_seconds"].max() == pytest.approx(4.0 * int(geometry["horizon"]))


def test_at_a_random_init_the_skill_against_climatology_is_indistinguishable_from_zero(
    evaluated,
) -> None:
    """The other half of the sign check. The ``trained_run`` fixture is a perturbed random init
    whose decoder head is calibrated to predict the normalisation's own centre -- which *is*
    climatology -- so a pipeline reporting a large skill here would be reporting its own bug."""
    r2 = evaluated["summary"]["results"]["forecast"]["r2"]

    assert r2["reference"] == forecast_analysis.R2_REFERENCE
    assert abs(float(r2["base"])) < 0.05
    assert abs(float(r2["full"])) < 0.05


@pytest.mark.slow
def test_a_fitted_checkpoint_beats_climatology_by_a_wide_margin(fitted_evaluated) -> None:
    """The criterion every later "the model learned something" acceptance test rests on.

    Fitted on one draw of the forecastable shards and scored on another, so this is out-of-sample
    skill rather than memorisation -- which, with far more parameters than segments, is what an
    in-sample number would be measuring.
    """
    summary = fitted_evaluated["summary"]
    r2 = summary["results"]["forecast"]["r2"]

    assert fitted_evaluated["exit_code"] == 0
    assert float(r2["base"]) > 0.5
    assert float(r2["full"]) > 0.5


@pytest.mark.slow
def test_a_fitted_checkpoints_forecast_error_is_reported_in_bpm(fitted_evaluated) -> None:
    """A forecast of a $140 \\pm 15$ bpm signal must land within a few bpm, and the conversion
    that says so is the spread one -- the affine map would report about $140$ bpm of error for a
    model that is nearly right."""
    errors = {row["branch"]: row for row in summary_forecast(fitted_evaluated)["error"]}

    for branch in ("base", "full"):
        assert errors[branch]["unit"] == "bpm"
        assert 0.0 < float(errors[branch]["rmse"]) < 20.0
        assert abs(float(errors[branch]["bias"])) < 5.0


def summary_forecast(evaluated_fixture) -> dict:
    """Return the forecast block of a finished run's summary."""
    return evaluated_fixture["summary"]["results"]["forecast"]
