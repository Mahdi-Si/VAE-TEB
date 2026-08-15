r"""There is no unit conversion here, and the absence is the design rather than an omission.

**The inverted port of the sibling's file.** ``teb_vae/lag_attn_rws/tests/test_eval_units.py`` proves
that the two z-to-bpm conversions are not interchangeable -- a *level* is affine,
$x(s + \varepsilon) + m$, while a *spread* is a difference of levels and scales only -- because
putting a spread through the level map turns an RMSE of $0.1$ z-units into $141$ bpm, a
physiologically reasonable fetal heart rate and therefore a number nobody questions.

That question does not arise in this target domain, and this file proves it cannot. The forecast
here is $98$ wavelet-modulus and phase-harmonic coefficients, and a coefficient has **no clinical
unit at all**: there is no bpm for it to be converted into, no $\sigma$ for a band to be drawn in,
and inverting the loader's per-channel statistics would put the $98$ channels on scales spanning
orders of magnitude -- which destroys every pooled statistic, every shared colour bar and the
warm-up tertile split that reads across them.

So ``BPM_UNIT``, ``to_bpm``, ``sigma_to_bpm``, ``fhr_normalization`` and ``_DENORMALIZE_EPSILON``
are **deleted rather than repointed**, and that distinction is what these tests defend: a
repointed conversion is one that can be called, and a conversion that can be called will be. What
is left is one label, ``normalised``, and the assertions below say that every emitted unit is it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from teb_vae.lag_attn_cfs.eval import metrics

#: Every name the sibling exports for the conversion this package does not have.
REMOVED_SYMBOLS = (
    "BPM_UNIT",
    "to_bpm",
    "sigma_to_bpm",
    "fhr_normalization",
    "_DENORMALIZE_EPSILON",
)

#: The module's own source, for the scans that a namespace check cannot do -- a helper defined
#: inside a function is not an attribute of the module.
SOURCE = Path(metrics.__file__).read_text(encoding="utf-8")


# =================================================================================================
# Nothing converts
# =================================================================================================
@pytest.mark.parametrize("name", REMOVED_SYMBOLS)
def test_the_conversion_the_sibling_needs_does_not_exist_here(name: str) -> None:
    """Named one by one rather than counted, so a symbol reintroduced under its own name fails
    with that name in the message."""
    assert not hasattr(metrics, name), (
        f"{name} exists in this package's readout module. A coefficient has no clinical unit, so "
        f"a conversion here would be arithmetic with no meaning -- and one that can be called "
        f"will be."
    )


def test_the_siblings_module_does_have_them() -> None:
    """Non-vacuity. Without this the four assertions above would pass against a typo in every
    name, and against a sibling that had itself dropped them."""
    sibling = pytest.importorskip("teb_vae.lag_attn_rws.eval.metrics")

    assert all(hasattr(sibling, name) for name in REMOVED_SYMBOLS)


def test_no_attribute_of_the_module_mentions_the_clinical_unit() -> None:
    """Broader than the five names above: *any* public or private attribute of the module carrying
    the substring, so a conversion reintroduced under a new name is caught too."""
    offending = sorted(name for name in dir(metrics) if "bpm" in name.lower())

    assert offending == [], f"{offending} name a clinical unit this target domain does not have"


def test_no_function_or_constant_defined_in_the_module_mentions_it_either() -> None:
    """An attribute scan sees the module namespace; a lazily-imported or nested definition is not
    in it. This walks every definition and every assignment target in the source instead."""
    tree = ast.parse(SOURCE)
    defined = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.append(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.append(node.id)

    assert [name for name in defined if "bpm" in name.lower()] == []


def test_nothing_reaches_for_the_repositorys_denormaliser() -> None:
    """``denormalize_signal_data`` is the one supported z-to-bpm path in this repository, and the
    sibling calls it. Reaching it from here would be the same failure under another name."""
    assert "denormalize_signal_data" not in SOURCE
    assert "get_normalization_stats" not in SOURCE


# =================================================================================================
# One label, and it is on the artifacts
# =================================================================================================
def test_the_only_unit_label_is_the_normalised_one() -> None:
    assert metrics.NORMALISED_UNIT == "normalised"


def test_the_calibration_record_states_the_unit_its_numbers_are_in() -> None:
    """A calibration census over coefficients in $z$ units reads exactly like one over bpm unless
    something says which. The label is in the emitted record rather than in a document beside it.
    """
    report = metrics.calibration_report(
        {
            "count": 100.0,
            "sum_residual_sq": 100.0,
            "sum_standardised_sq": 100.0,
            "sum_logvar": 0.0,
            "crps_sum": 50.0,
            "pit_histogram": [5.0] * metrics.PIT_BINS,
            "logvar_histogram": [0.0] * metrics.LOGVAR_BINS,
            "within_1_sigma": 68.0,
            "within_2_sigma": 95.0,
            "within_3_sigma": 100.0,
        },
        logvar_clamp=(-5.0, 3.0),
    )

    assert report["unit"] == metrics.NORMALISED_UNIT


def test_the_calibration_gain_is_per_coefficient_and_names_no_raw_sample() -> None:
    r"""The scored unit here is one of the $H \cdot C_{\mathrm{keep}}$ coefficients of a forecast
    block, and there is no raw sample anywhere in this pipeline for a gain to be per. The rename is
    not cosmetic: the two denominators differ by a factor of three at the shipped geometry, so a
    column carried across under the sibling's name would be silently non-comparable with the
    sibling's number."""
    report = metrics.calibration_report(
        {
            "count": 10.0,
            "sum_residual_sq": 10.0,
            "sum_standardised_sq": 10.0,
            "sum_logvar": 0.0,
            "crps_sum": 1.0,
            "pit_histogram": [0.5] * metrics.PIT_BINS,
            "logvar_histogram": [0.0] * metrics.LOGVAR_BINS,
        },
        logvar_clamp=(-5.0, 3.0),
    )

    assert set(report["nll"]) == {
        "model_per_coefficient",
        "homoscedastic_per_coefficient",
        "gain_per_coefficient",
        "homoscedastic_sigma",
    }
    assert "n_coefficients" in report
    assert all("raw_sample" not in key for key in report)


def test_the_calibration_records_weighting_sentence_names_coefficients() -> None:
    """The sentence is copied into the summary verbatim, and ``tests/test_eval_naming.py`` scans
    every artifact for the phrase 'raw sample'. It has to be right at the source."""
    report = metrics.calibration_report(
        {
            "count": 10.0,
            "sum_residual_sq": 10.0,
            "sum_standardised_sq": 10.0,
            "sum_logvar": 0.0,
            "crps_sum": 1.0,
            "pit_histogram": [0.5] * metrics.PIT_BINS,
            "logvar_histogram": [0.0] * metrics.LOGVAR_BINS,
        }
    )

    assert "coefficient" in report["weighting"]
    assert "raw sample" not in report["weighting"]


def test_a_census_over_nothing_is_a_skip_rather_than_a_number() -> None:
    """An empty report, not a report full of NaNs: the calibration verdict distinguishes "could
    not be evaluated" from "evaluated and failed", and only an empty block reaches the first."""
    assert metrics.calibration_report({}) == {}
    assert metrics.calibration_report({"count": 0.0}) == {}
