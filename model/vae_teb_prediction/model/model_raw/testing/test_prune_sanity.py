"""S6-T01c: assert the scattering-specific modules are pruned and the agnostic ones survive.

The raw port copies the scattering-domain testing suite into ``model_raw/testing/`` but drops every
module that only has meaning for the 87-channel scattering/phase feature future -- the kymatio
``band_partition`` map, the frequency-band forecast analysis, its unit-test, and the band-uplift
regression (Test 3). This test locks that prune in: the pruned modules must be **absent** from the
raw testing package, while the kept domain-agnostic analyses and core modules must import cleanly.
"""
from __future__ import annotations

import importlib
import importlib.util

import pytest

_PKG = "model.vae_teb_prediction.model.model_raw.testing"

#: Modules that must NOT exist under the raw testing package (scattering-band specific).
PRUNED_MODULES = [
    f"{_PKG}.band_partition",
    f"{_PKG}.test_band_partition_extensions",
    f"{_PKG}.analyses.frequency_band_forecast",
    f"{_PKG}.causal_te_validation.test_03_band_uplift_regression",
]

#: Core + agnostic analysis modules that must import cleanly after the copy/prune.
KEPT_MODULES = [
    f"{_PKG}.base",
    f"{_PKG}.metrics",
    f"{_PKG}.collectors",
    f"{_PKG}.visualizers",
    f"{_PKG}.run_tests",
    f"{_PKG}.analyses",
    f"{_PKG}.analyses.calibration",
    f"{_PKG}.analyses.cmi_comparison",
    f"{_PKG}.analyses.latent",
    f"{_PKG}.analyses.attention_diagnostics",
    f"{_PKG}.analyses.te_lag_analysis",
    f"{_PKG}.causal_te_validation.runner",
    f"{_PKG}.TE_Calculated.cmi_adapter",
]


@pytest.mark.parametrize("module", PRUNED_MODULES)
def test_scattering_module_is_pruned(module: str) -> None:
    """``find_spec`` returns ``None`` for every pruned scattering module."""
    assert importlib.util.find_spec(module) is None, f"{module} should be pruned"


@pytest.mark.parametrize("module", KEPT_MODULES)
def test_kept_module_imports(module: str) -> None:
    """Every kept core/agnostic module imports without error."""
    importlib.import_module(module)


def test_analyses_package_drops_frequency_band_export() -> None:
    """The scattering ``run_frequency_band_forecast_analysis`` is no longer exported."""
    analyses = importlib.import_module(f"{_PKG}.analyses")
    assert not hasattr(analyses, "run_frequency_band_forecast_analysis")
    assert "run_frequency_band_forecast_analysis" not in getattr(analyses, "__all__", [])


def test_kept_agnostic_analyses_are_exported() -> None:
    """The domain-agnostic analyses the raw pipeline relies on are still exported."""
    analyses = importlib.import_module(f"{_PKG}.analyses")
    for name in (
        "run_calibration_analysis",
        "run_cmi_comparison",
        "run_latent_distribution_analysis",
        "run_attention_diagnostics",
        "run_te_lag_class_analysis",
        "run_causal_te_validation",
    ):
        assert hasattr(analyses, name), f"{name} should still be exported"
