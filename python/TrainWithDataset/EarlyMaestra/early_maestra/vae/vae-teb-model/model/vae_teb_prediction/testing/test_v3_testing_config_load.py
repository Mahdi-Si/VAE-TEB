r"""S7-T02: ``testing/config_lag_attn_v3.yaml`` loads and rebuilds a valid ``SeqVaeLagAttnV3``.

The testing pipeline rebuilds the model from the config's ``model_config.VAE_model`` via
``_lag_attn_kwargs_from_config`` (``inspect.signature`` discovery) whenever a checkpoint carries
no stamped ``model_kwargs``. This pins that the shipped v3 testing config surfaces every v3
architecture flag through that path, and that the optional empirical-TE key is present.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import model.vae_teb_prediction.testing.base as base_module
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from model.vae_teb_prediction.testing.base import _lag_attn_kwargs_from_config

_CONFIG_PATH = Path(__file__).resolve().parent / "config_lag_attn_v3.yaml"


@pytest.fixture(autouse=True)
def _v3_alias(monkeypatch):
    """Point the discovery helper's class alias at v3."""
    monkeypatch.setattr(base_module, "SeqVaeLagAttn", SeqVaeLagAttnV3)


def _load_cfg() -> dict:
    assert _CONFIG_PATH.is_file(), f"missing testing config: {_CONFIG_PATH}"
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_testing_config_rebuilds_v3():
    cfg = _load_cfg()
    kwargs = _lag_attn_kwargs_from_config(cfg)
    for flag in ("causal_norm", "logvar_bound", "posterior_logvar", "kld_support",
                 "lag_bias_init", "head_structured_latent", "freeze_unused_attn_proj"):
        assert flag in kwargs, f"v3 flag {flag} not surfaced from the testing config"

    model = SeqVaeLagAttnV3(**kwargs)
    assert model.causal_norm is True
    assert model.posterior_logvar == "residual"
    assert model.logvar_bound == "smooth"
    assert model.kld_support == "anchor"
    assert model.lag_attn.lag_score_bias is not None  # alibi_decay
    assert model.frozen_attn_proj is True             # head-structured + freeze


def test_empirical_te_csv_key_is_present_and_defaults_null():
    cfg = _load_cfg()
    ds = cfg["dataset_config"]
    assert "empirical_te_csv" in ds, "the optional G11 empirical-TE key must be documented"
    assert ds["empirical_te_csv"] is None
