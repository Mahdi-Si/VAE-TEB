r"""Availability of the builder's remaining repair operations on the canonical dataset timeline.

CFS-04 asks which of the operations between the adaptor's output and the causal transform are
retrospective (read samples after an issue time) and which are online (a prefix of the record
produces a prefix of the output). The dataset-creation UP shift is the canonical timeline and is
outside this audit by the project rule; what is left in :mod:`create_new_pipeline` between the
adaptor and :func:`transform_sample` is ``_sanitize_signals``, which is
``interpolate_bad_values`` followed by a pointwise clip and a denormal flush.

What these tests establish, on synthetic rows through the executable functions:

* **Interior non-finite runs are repaired retrospectively.** ``np.interp`` bridges an interior gap
  linearly between its two valid knots, and the right knot lies after every sample inside the gap.
  A run that stops at an issue time inside the gap holds the last valid value instead, so the two
  disagree on every repaired sample before the right knot.
* **A leading non-finite run reads the first valid sample** -- a future read confined to the record
  head -- and **a trailing run holds the last valid sample**, which is prefix-equivalent.
* **Everything else in the sanitiser is pointwise** and therefore prefix-equivalent, and a
  perturbation after an issue time reaches no earlier sample while changing later ones, so the
  prefix claim on the finite path is not vacuous.

What they do not establish, and why it is recorded here rather than tested: whether the interior
branch is ever *reached* on a real record. By reading the (untracked, production-only) adaptor,
``read_input_common_2d`` zero-fills any segdat segment that contains a NaN and marks its targets as
pad, so no NaN survives the adaptor and ``weight`` is $0$ where the raw trace was zero; the
sanitiser then sees finite rows and the interior branch is dead on that path. The adaptor's
whole-block Fourier ``scipy.signal.resample`` runs only when a record's own sampling rate exceeds
the $4$ Hz base rate, and the adaptor retains neither the raw rate nor a flag, so whether it ran
for a given shard is not recoverable from the shard and needs the raw records (open in CFS-04).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

N_SAMPLES = 400


def _ramp(n_samples: int = N_SAMPLES) -> np.ndarray:
    """One row of a strictly increasing ramp, offset so it is never confused with zero padding."""
    return np.arange(n_samples, dtype=np.float64)[None, :] + 100.0


def test_an_interior_non_finite_run_is_repaired_from_a_future_knot(pipeline: Any) -> None:
    """The retrospective branch: the bridge across an interior gap ends at a knot that does not
    exist at an issue time inside the gap, so the full-record repair and the prefix repair disagree
    on every sample from the start of the gap to the issue time."""
    full = _ramp()
    full[0, 100:110] = np.nan
    issue = 105
    prefix = full[:, :issue].copy()

    repaired_full = pipeline.interpolate_bad_values(full.copy())
    repaired_prefix = pipeline.interpolate_bad_values(prefix)

    # The full-record bridge is the ramp itself (linear between the knots at 99 and 110)...
    assert np.allclose(repaired_full[0, 100:105], np.arange(100, 105) + 100.0)
    # ...while the prefix, whose gap runs to its end, holds the last valid sample flat.
    assert np.all(repaired_prefix[0, 100:105] == repaired_prefix[0, 99])
    assert not np.allclose(repaired_full[0, :issue], repaired_prefix[0])
    # The disagreement is local to the run: everything before the gap is identical.
    assert np.array_equal(repaired_full[0, :100], repaired_prefix[0, :100])


def test_a_trailing_non_finite_run_is_held_flat_and_is_prefix_equivalent(pipeline: Any) -> None:
    full = _ramp()
    full[0, 390:] = np.nan

    repaired_full = pipeline.interpolate_bad_values(full.copy())

    assert np.all(repaired_full[0, 390:] == repaired_full[0, 389])
    for issue in (391, 395, N_SAMPLES):
        repaired_prefix = pipeline.interpolate_bad_values(full[:, :issue].copy())
        assert np.array_equal(repaired_full[0, :issue], repaired_prefix[0]), issue


def test_a_leading_non_finite_run_reads_the_first_valid_sample(pipeline: Any) -> None:
    """A future read, confined to the record head: the first ten samples take the value at sample
    ten. Recorded as a fact of the operation rather than as a defect -- nothing before the first
    valid sample is an issue time anyone forecasts from."""
    full = _ramp()
    full[0, :10] = np.nan

    repaired = pipeline.interpolate_bad_values(full.copy())

    assert np.all(repaired[0, :10] == full[0, 10])
    assert np.array_equal(repaired[0, 10:], full[0, 10:])


def test_the_sanitiser_is_pointwise_and_therefore_prefix_equivalent_on_finite_input(
    pipeline: Any,
) -> None:
    rng = np.random.default_rng(0)
    fhr = rng.normal(140.0, 60.0, size=(2, N_SAMPLES))
    up = rng.normal(20.0, 80.0, size=(2, N_SAMPLES))
    # Subnormals are flushed and the ranges are clipped, both per sample.
    fhr[0, 5] = 1e-45
    up[1, 7] = -1e-41

    full_fhr, full_up = pipeline._sanitize_signals(fhr.copy(), up.copy())

    assert full_fhr.min() >= 0.0 and full_fhr.max() <= 500.0
    assert full_up.min() >= -50.0 and full_up.max() <= 500.0
    assert full_fhr[0, 5] == 0.0 and full_up[1, 7] == 0.0
    for issue in (1, 37, 250, N_SAMPLES):
        prefix_fhr, prefix_up = pipeline._sanitize_signals(
            fhr[:, :issue].copy(), up[:, :issue].copy()
        )
        assert np.array_equal(full_fhr[:, :issue], prefix_fhr), issue
        assert np.array_equal(full_up[:, :issue], prefix_up), issue


def test_a_perturbation_after_the_issue_time_reaches_no_earlier_sample_but_changes_later_ones(
    pipeline: Any,
) -> None:
    """The non-vacuous half of the prefix claim on the finite path."""
    base = _ramp()
    perturbed = base.copy()
    perturbed[0, 300:] += 50.0

    clean_fhr, _ = pipeline._sanitize_signals(base.copy(), base.copy())
    moved_fhr, _ = pipeline._sanitize_signals(perturbed.copy(), perturbed.copy())

    assert np.array_equal(clean_fhr[0, :300], moved_fhr[0, :300])
    assert not np.array_equal(clean_fhr[0, 300:], moved_fhr[0, 300:])


def test_the_interior_repair_is_the_only_non_pointwise_step(pipeline: Any) -> None:
    """A NaN gap that straddles the issue time is the one way the sanitiser reads the future: with
    the gap removed, the same two rows are prefix-equivalent through the whole sanitiser."""
    with_gap = _ramp()
    with_gap[0, 100:110] = np.nan
    issue = 105

    full_fhr, _ = pipeline._sanitize_signals(with_gap.copy(), with_gap.copy())
    prefix_fhr, _ = pipeline._sanitize_signals(
        with_gap[:, :issue].copy(), with_gap[:, :issue].copy()
    )
    assert not np.array_equal(full_fhr[0, :issue], prefix_fhr[0])

    without_gap = _ramp()
    full_fhr, _ = pipeline._sanitize_signals(without_gap.copy(), without_gap.copy())
    prefix_fhr, _ = pipeline._sanitize_signals(
        without_gap[:, :issue].copy(), without_gap[:, :issue].copy()
    )
    assert np.array_equal(full_fhr[0, :issue], prefix_fhr[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_every_non_finite_value_is_repaired_the_same_way(pipeline: Any, bad: float) -> None:
    full = _ramp()
    full[0, 200] = bad

    repaired = pipeline.interpolate_bad_values(full.copy())

    assert np.isfinite(repaired).all()
    assert repaired[0, 200] == pytest.approx(300.0)
