r"""The frozen latent probes: the split, the streaming fit, and what the number is scored against.

Three failures this file exists to detect, none of them visible from the output:

**A probe fitted on what it is scored on.** A $64$-dimensional design over millions of anchors will
fit a recording it has already seen, so an in-sample coefficient of determination measures the fit.
The split is by recording and the two sides must stay disjoint however the batches arrive.

**A streaming solve that is not the solve.** The pass accumulates second moments and never holds a
row, which is what lets it use every anchor rather than a subsample. That is only worth doing if it
agrees with the fit it stands in for, and nothing about the output would look wrong if it did not.

**A null that flatters the probe.** The coefficient is taken against the fitting split's own mean,
which is the prediction a model that had seen only those recordings would make. Scored against the
scoring split's own mean instead, every probe would gain the difference between two cohorts' means
for free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from teb_vae.lag_slot_transformer_cfs.eval import latent_probes

from .test_arms import fit_arm, write_eval_delta

#: Rows, features and targets in the constructed checks. Small and not square: a design whose
#: shapes all match hides an axis swapped for another.
N_ROWS, N_FEATURES, N_TARGETS = 240, 6, 8


def moments_of(features: np.ndarray, targets: np.ndarray) -> latent_probes.DesignMoments:
    """Accumulate one array pair in two unequal blocks.

    Two blocks rather than one, because a streaming accumulator that only ever sees a single call
    is not being tested as one.

    Args:
        features: $(n, d)$ rows.
        targets: $(n, m)$ rows.

    Returns:
        The accumulated moments.
    """
    moments = latent_probes.DesignMoments(features.shape[1], targets.shape[1])
    cut = features.shape[0] // 3
    moments.add(features[:cut], targets[:cut])
    moments.add(features[cut:], targets[cut:])
    return moments


def direct_ridge(features: np.ndarray, targets: np.ndarray, alpha: float):
    """The same ridge, solved from the rows, as the reference the streaming solve is checked on.

    Args:
        features: $(n, d)$ rows.
        targets: $(n, m)$ rows.
        alpha: Ridge penalty in standardised units.

    Returns:
        ``(coefficients, intercept)``.
    """
    mean_x, mean_y = features.mean(axis=0), targets.mean(axis=0)
    centred = features - mean_x
    scale = centred.std(axis=0)
    standardised = centred / scale
    gram = standardised.T @ standardised / features.shape[0]
    cross = standardised.T @ (targets - mean_y) / features.shape[0]
    beta = np.linalg.solve(gram + alpha * np.eye(features.shape[1]), cross) / scale[:, None]
    return beta, mean_y - beta.T @ mean_x


# =============================================================================
# The split
# =============================================================================
def test_the_split_is_a_function_of_the_recording_and_nothing_else() -> None:
    """Two arms' probes have to be fitted on the same cohort, or their difference carries the
    difference between two cohorts as well as the difference between two latents."""
    assignments = [latent_probes.split_of("GUID-0007") for _ in range(5)]

    assert len(set(assignments)) == 1
    assert set(assignments) <= {"fit", "score"}


def test_the_split_puts_recordings_on_both_sides() -> None:
    """A rule that sent every recording one way would report a probe fitted on everything and
    scored on nothing, and the pass would say so only in its counts."""
    sides = {latent_probes.split_of(f"GUID-{index:04d}") for index in range(200)}

    assert sides == {"fit", "score"}


def test_the_two_sides_are_disjoint() -> None:
    """One digest of one identifier decides the side, so a recording cannot be on both."""
    fitted = {
        guid
        for guid in (f"GUID-{index:04d}" for index in range(200))
        if latent_probes.split_of(guid) == "fit"
    }
    scored = {
        guid
        for guid in (f"GUID-{index:04d}" for index in range(200))
        if latent_probes.split_of(guid) == "score"
    }

    assert not fitted & scored
    assert len(fitted) + len(scored) == 200


# =============================================================================
# The streaming solve
# =============================================================================
def test_the_streaming_solve_is_the_solve_it_stands_in_for() -> None:
    """Accumulating moments is only worth doing if it agrees with the fit from the rows, and
    nothing in the output would look wrong if it did not."""
    generator = np.random.default_rng(20260909)
    features = generator.normal(size=(N_ROWS, N_FEATURES))
    targets = features @ generator.normal(size=(N_FEATURES, N_TARGETS)) + generator.normal(
        scale=0.1, size=(N_ROWS, N_TARGETS)
    )

    beta, intercept = latent_probes.ridge_from_moments(moments_of(features, targets))
    expected_beta, expected_intercept = direct_ridge(features, targets, latent_probes.RIDGE_ALPHA)

    assert np.allclose(beta, expected_beta, atol=1e-10, rtol=1e-8)
    assert np.allclose(intercept, expected_intercept, atol=1e-10, rtol=1e-8)


def test_the_sums_of_squares_are_the_ones_the_rows_give() -> None:
    """The residual sum of squares is expanded into second moments rather than accumulated from
    residuals, and the expansion is where a sign or a cross term goes wrong silently."""
    generator = np.random.default_rng(1)
    features = generator.normal(size=(N_ROWS, N_FEATURES))
    targets = features @ generator.normal(size=(N_FEATURES, N_TARGETS)) + generator.normal(
        scale=0.5, size=(N_ROWS, N_TARGETS)
    )
    held = generator.normal(size=(N_ROWS, N_FEATURES))
    held_targets = held @ generator.normal(size=(N_FEATURES, N_TARGETS))

    fit = moments_of(features, targets)
    score = moments_of(held, held_targets)
    beta, intercept = latent_probes.ridge_from_moments(fit)
    sums = latent_probes.explained_variance(fit, score, beta, intercept)

    predicted = held @ beta + intercept
    null = targets.mean(axis=0)
    assert np.allclose(
        sums["ss_res"], ((held_targets - predicted) ** 2).sum(axis=0), atol=1e-8, rtol=1e-8
    )
    assert np.allclose(
        sums["ss_tot"], ((held_targets - null) ** 2).sum(axis=0), atol=1e-8, rtol=1e-8
    )


def test_a_readable_target_scores_near_one_and_noise_does_not() -> None:
    """The two ends the coefficient has to reach, so a probe that always reports the same number
    is not mistaken for a measurement."""
    generator = np.random.default_rng(7)
    weights = generator.normal(size=(N_FEATURES, N_TARGETS))
    features = generator.normal(size=(N_ROWS, N_FEATURES))
    held = generator.normal(size=(N_ROWS, N_FEATURES))

    readable = latent_probes.probe_report(
        latent_probes.explained_variance(
            moments_of(features, features @ weights),
            moments_of(held, held @ weights),
            *latent_probes.ridge_from_moments(moments_of(features, features @ weights)),
        ),
        horizon=2,
    )
    noise = latent_probes.probe_report(
        latent_probes.explained_variance(
            moments_of(features, generator.normal(size=(N_ROWS, N_TARGETS))),
            moments_of(held, generator.normal(size=(N_ROWS, N_TARGETS))),
            *latent_probes.ridge_from_moments(
                moments_of(features, generator.normal(size=(N_ROWS, N_TARGETS)))
            ),
        ),
        horizon=2,
    )

    assert readable["r2"] > 0.99
    assert noise["r2"] < 0.2
    assert len(readable["r2_per_horizon_step"]) == 2


def test_a_coordinate_with_no_variance_contributes_nothing_rather_than_dividing_by_zero() -> None:
    """A latent coordinate the prior holds constant is a real state, and it is the one case where
    standardising the design would divide by zero."""
    generator = np.random.default_rng(3)
    features = generator.normal(size=(N_ROWS, N_FEATURES))
    features[:, 2] = 1.5
    targets = generator.normal(size=(N_ROWS, N_TARGETS))

    beta, _intercept = latent_probes.ridge_from_moments(moments_of(features, targets))

    assert np.all(np.isfinite(beta))
    assert np.allclose(beta[2, :], 0.0)


def test_a_negative_coefficient_is_reported_as_it_comes() -> None:
    """Clamping at zero would hide the one outcome that says the latent carries nothing: a probe
    fitted on other recordings CAN predict worse than their mean."""
    report = latent_probes.probe_report(
        {"ss_res": np.array([4.0, 4.0]), "ss_tot": np.array([1.0, 1.0])}, horizon=1
    )

    assert report["r2"] == pytest.approx(-3.0)


# =============================================================================
# The pass, on a checkpoint a fit produced
# =============================================================================
@pytest.mark.slow
def test_the_pass_probes_a_real_checkpoint_and_records_what_it_probed(tmp_path_factory) -> None:
    """The seam: a fit, then the pass, then the artifact the protocol reads.

    Everything above is arithmetic on constructed arrays. This is the only place the gather, the
    complete-coverage rule, the anchor-relative target and the artifact's provenance are exercised
    against a model that was actually trained.
    """
    # A short root: the fits write deeply nested run directories, and a long temporary prefix
    # pushes the checkpoint paths past what the platform will rename.
    work = Path(tmp_path_factory.mktemp("q"))
    checkpoint = fit_arm(work, "probe", {})
    delta = write_eval_delta(work, "probe")

    code = latent_probes.main(
        checkpoint=str(checkpoint),
        output_dir=str(work / "probe_eval"),
        device="cpu",
        overrides=str(delta),
        sources={"checkpoint": "cli"},
    )
    artifact = json.loads(
        (work / "probe_eval" / "eval_results" / latent_probes.PROBE_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    assert code == 0
    assert set(artifact["probes"]) == set(latent_probes.FEATURE_SETS)
    assert artifact["counts"]["sides_are_disjoint"]
    assert artifact["counts"]["fit_anchors"] > 0 and artifact["counts"]["score_anchors"] > 0
    assert artifact["counts"]["block_coefficients"] > 0
    # The provenance the protocol attaches the artifact to its run by.
    assert artifact["run"]["checkpoint"] == str(checkpoint)
    assert artifact["run"]["training_seed"] >= 0
    assert artifact["settings"]["fit_percent"] == latent_probes.PROBE_FIT_PERCENT
    assert "persistence" in artifact["settings"]["target"]
