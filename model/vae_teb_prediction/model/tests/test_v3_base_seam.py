"""S0-T04: testing alias seam + model-class guard recognise SeqVaeLagAttnV3.

The alias in ``testing/base.py`` is a one-line comment-toggle (v3 line added commented, v1
active). This test verifies the *mechanism*: ``check_model_class`` accepts a v3-labelled
checkpoint, and ``_lag_attn_kwargs_from_config`` surfaces v3's new constructor flags via
``inspect.signature`` when v3 is the active alias.

Lives in ``model/tests/`` (not ``testing/``) so it is collectable in isolation: this
package's ``conftest.py`` normalises ``sys.path`` before the ``testing`` package -- whose
``__init__.py`` does absolute ``model.vae_teb_prediction`` imports -- is loaded. A conftest
inside the ``testing`` package cannot do that (it would run ``testing/__init__.py`` first).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import model.vae_teb_prediction.testing.base as base
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from model.vae_teb_prediction.testing.base import (
    _lag_attn_kwargs_from_config,
    check_model_class,
)

# tests/ -> model/  (config_lag_attn_v3.yaml lives beside the model module)
_V3_CONFIG = Path(__file__).resolve().parents[1] / "config_lag_attn_v3.yaml"


def test_check_model_class_accepts_v3():
    # Matching label passes silently.
    check_model_class({"model_class": "SeqVaeLagAttnV3"}, "SeqVaeLagAttnV3")
    assert SeqVaeLagAttnV3.__name__ == "SeqVaeLagAttnV3"


def test_check_model_class_rejects_mismatch():
    with pytest.raises(ValueError):
        check_model_class({"model_class": "SeqVaeLagAttnV1"}, "SeqVaeLagAttnV3")


def test_kwargs_from_config_surfaces_v3_flags(monkeypatch):
    # Point the seam at v3 (as the commented import line does when toggled).
    monkeypatch.setattr(base, "SeqVaeLagAttn", SeqVaeLagAttnV3)

    with open(_V3_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    kwargs = base._lag_attn_kwargs_from_config(cfg)
    for flag in ("logvar_bound", "posterior_logvar", "delta_logvar_scale", "kld_support"):
        assert flag in kwargs, f"v3 flag {flag} not surfaced by _lag_attn_kwargs_from_config"
    # A representative v1 backbone arg is still discovered (explicit signature, not **kwargs).
    assert "d_model" in kwargs and "head_structured_latent" in kwargs
    assert kwargs["posterior_logvar"] == "residual"
    assert kwargs["kld_support"] == "anchor"


def test_module_imported_symbol_exists():
    # The imported names are the ones the seam comment references.
    assert callable(check_model_class)
    assert callable(_lag_attn_kwargs_from_config)
