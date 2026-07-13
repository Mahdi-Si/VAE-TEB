r"""S3-T03 batch-contract test: a v4 datamodule batch drives ``SeqVaeRawV4.forward``.

Uses ``make_small_prod_raw_model`` (full $5280$ geometry, shrunk widths) -- NOT the
``TINY_RAW_LEN=512`` toy fixtures, whose $32$-length weight would make ``frontend_mask`` raise
against the $330$-length cache weight.
"""

from __future__ import annotations

import copy

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.datamodule_v4 import (
    SyntheticRawDataModuleV4,
)

pytestmark = pytest.mark.v4

# The exact 25 keys of the raw forward dict.
_FORWARD_KEYS = {
    "mu_prior", "logvar_prior", "raw_logvar_prior", "mu_post", "logvar_post", "z",
    "target_state", "source_state", "decoder_state", "attended_source", "attended_source_heads",
    "attn_weights", "mu_base", "logvar_base", "delta_mu_src", "mu_full", "logvar_full",
    "raw_future_pred", "kld_per_t", "kld_per_t_per_head", "te_lag_map", "warmup_mask",
    "mu_prior_sat_frac", "delta_mu_sat_frac", "kld_active_frac",
}


def test_batch_contract_v4_forward(tiny_cache_v4):
    r"""``_default_batch_to_inputs`` + ``forward`` accept a v4 batch and return the 25-key dict."""
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        make_small_prod_raw_model,
    )

    cfg = copy.deepcopy(tiny_cache_v4["config"])
    cfg.setdefault("dataset_config", {}).setdefault("dataloader_config", {})["num_workers"] = 0

    dm = SyntheticRawDataModuleV4(
        cfg, batch_size=2, benchmark="G1_raw_v4", cache_dir=tiny_cache_v4["cache_dir"],
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    model = make_small_prod_raw_model()
    model.eval()

    fhr_raw, up_raw, mask = model._default_batch_to_inputs(batch)
    assert fhr_raw.shape == (2, 5280)
    assert up_raw.shape == (2, 5280)
    assert mask.shape == (2, 5280)
    uniq = set(mask.unique().tolist())
    assert uniq <= {0.0, 1.0}

    import torch

    with torch.no_grad():
        out = model.forward(fhr_raw, up_raw, mask)

    assert set(out.keys()) == _FORWARD_KEYS
    assert out["mu_full"].shape == (2, 300, 30, 16)
    assert out["attn_weights"].shape == (2, 300, 4, 91)
    assert out["te_lag_map"].shape == (2, 300, 91)
    assert out["kld_per_t"].shape == (2, 300)
    assert out["raw_future_pred"] is not None
    assert out["raw_future_pred"].shape == (2, 300, 30, 16)
