"""S0-T03: config_lag_attn_v3.yaml loads and constructs a valid SeqVaeLagAttnV3.

Mirrors the testing pipeline's ``_lag_attn_kwargs_from_config`` discovery: forward only the
``VAE_model`` keys whose names match a real constructor parameter (via ``inspect.signature``),
so nested trainer-side groups (beta_schedule, horizon_refine, encoder, loss_spike_skip) are
ignored and the constructor defaults fill the rest.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_lag_attn_v3.yaml"


def _kwargs_from_config(cfg: dict) -> dict:
    vae_cfg = cfg["model_config"]["VAE_model"]
    params = inspect.signature(SeqVaeLagAttnV3.__init__).parameters
    kwargs = {}
    for name, param in params.items():
        if name in ("self", "init_weights"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in vae_cfg and vae_cfg[name] is not None:
            kwargs[name] = vae_cfg[name]
    # logvar_clamp arrives as a YAML list; coerce to a float tuple.
    if "logvar_clamp" in kwargs:
        lc = kwargs["logvar_clamp"]
        kwargs["logvar_clamp"] = (float(lc[0]), float(lc[1]))
    return kwargs


def test_config_loads_and_constructs():
    assert _CONFIG_PATH.is_file(), f"missing config: {_CONFIG_PATH}"
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    kwargs = _kwargs_from_config(cfg)
    # The new v3 flags must be discoverable from the config.
    for flag in (
        "posterior_logvar", "logvar_bound", "kld_support", "delta_logvar_scale",
        "causal_norm", "lambda_perm", "perm_every_n_batches",
    ):
        assert flag in kwargs, f"v3 flag {flag} not surfaced from config"

    model = SeqVaeLagAttnV3(**kwargs)
    assert isinstance(model, SeqVaeLagAttnV3)
    # Production settings from the YAML.
    assert model.causal_norm is True and model.n_causalized_norms == 10  # G0
    assert model.posterior_logvar == "residual"                          # G1
    assert model.logvar_bound == "smooth"                                # G2
    assert model.kld_support == "anchor"                                 # G3
    assert model.lag_attn.lag_score_bias is not None                     # G5 (alibi_decay)
    # G6 ships as a READOUT: K_shuffled is logged every step, but it must not enter the loss.
    # KL(q||p) rises under a deranged source rather than falling, and a positive lambda_perm
    # collapsed the source pathway in half of the seeds tried -- see the config comment.
    assert model.lambda_perm == 0.0
    assert model.perm_every_n_batches >= 1
    # G7 + head-structured production: W_o is dead, so it must be frozen for plain 'ddp'.
    assert model.frozen_attn_proj is True
