r"""S0-T03: the synthetic<->model grid-alignment contract."""

from __future__ import annotations

import types

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import geometry_align_v4 as ga

pytestmark = pytest.mark.v4


def test_step_conversions() -> None:
    r"""Cropped anchor $t=0$ is uncropped token $15$, and the round-trip is exact."""
    assert ga.latent_to_model_step(15) == 0
    assert ga.model_to_latent_step(0) == 15
    for t in (0, 15, 100, 269):
        assert ga.model_to_latent_step(ga.latent_to_model_step(t)) == t


def test_planted_lag_is_relative_and_unchanged() -> None:
    r"""A planted lag maps to the model lag index unchanged (relative-offset invariance)."""
    for D in (0, 1, 8, 30, 90):
        assert ga.planted_lag_to_model_lag(D) == D


def test_assert_alignment_passes_against_real_geometry() -> None:
    r"""``assert_alignment`` holds against the reused production ``model_raw`` geometry."""
    assert ga.assert_alignment() is True


def test_assert_alignment_raises_on_wrong_crop() -> None:
    r"""A grid that is not a pure $\pm 15$ shift (wrong crop) fails loudly."""
    wrong = types.SimpleNamespace(crop=10, t_valid=270)
    with pytest.raises(AssertionError):
        ga.assert_alignment(wrong)


def test_assert_alignment_raises_on_wrong_t_valid() -> None:
    r"""A geometry with the right crop but a wrong ``t_valid`` still fails."""
    wrong = types.SimpleNamespace(crop=15, t_valid=260)
    with pytest.raises(AssertionError):
        ga.assert_alignment(wrong)
