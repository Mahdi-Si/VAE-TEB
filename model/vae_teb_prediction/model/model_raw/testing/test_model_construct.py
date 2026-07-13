"""S2-T01: ``SeqVaeRawV4`` construction (input-pathway swap, in-place decoders, G0 guard)."""
from __future__ import annotations

import pytest
import torch.nn as nn

from model.vae_teb_prediction.model.model_raw.raw_frontend import (
    CausalGroupNorm,
    CausalRawFrontend,
    assert_no_time_pooling_norm,
)
from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    TINY_FRONTEND,
    make_tiny_raw_model,
)
from model.vae_teb_prediction.model.model_raw.vae_teb_raw_v4 import (
    RawBaselineFutureDecoderV4,
    RawResidualFutureDecoderV4,
    SeqVaeRawV4,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
    PosteriorHeadV3,
    PriorHeadV3,
    SeqVaeLagAttnV3,
)


def test_construct_isinstance_and_heads():
    m = make_tiny_raw_model()
    assert isinstance(m, SeqVaeLagAttnV3)
    assert m.model_class == "SeqVaeRawV4"
    # The four inherited v3 heads are present (prior + posterior are the v3 variants; the two
    # decoders are replaced in place by the raw decoders).
    assert isinstance(m.prior_head, PriorHeadV3)
    assert isinstance(m.posterior_head, PosteriorHeadV3)
    assert isinstance(m.baseline_decoder, RawBaselineFutureDecoderV4)
    assert isinstance(m.residual_decoder, RawResidualFutureDecoderV4)


def test_front_ends_distinct_and_typed():
    m = make_tiny_raw_model()
    assert isinstance(m.frontend_y, CausalRawFrontend)
    assert isinstance(m.frontend_u, CausalRawFrontend)
    # Distinct objects, never weight-shared.
    assert m.frontend_y is not m.frontend_u
    fy = {id(p) for p in m.frontend_y.parameters()}
    fu = {id(p) for p in m.frontend_u.parameters()}
    assert fy.isdisjoint(fu)


def test_feature_adapters_removed():
    m = make_tiny_raw_model()
    # The feature adapters must be gone (else they starve DDP with find_unused_parameters=False).
    assert not hasattr(m, "target_adapter")
    assert not hasattr(m, "source_adapter")


def test_sequence_length_matches_front_end_tokens():
    m = make_tiny_raw_model().eval()
    # The inherited core must be sized by the front-end token count T; a forward emits exactly T
    # tokens on the anchor axis (and R substeps on the raw axis).
    assert m.geometry.t == 28
    from model.vae_teb_prediction.model.model_raw.testing.conftest import make_raw_batch

    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    assert out["mu_full"].shape == (2, m.geometry.t, m.geometry.horizon, m.geometry.r)


def test_decoders_share_horizon_core():
    m = make_tiny_raw_model()
    # Both raw decoders reuse the single shared horizon core (warm-startable, memory-cheap).
    assert m.baseline_decoder.core is m.horizon_core
    assert m.residual_decoder.core is m.horizon_core


def test_g0_guard_passes_on_input_pathway():
    m = make_tiny_raw_model()
    # Front ends + encoders must be free of time-pooling / batch-coupling norms.
    for sub in (m.frontend_y, m.frontend_u, m.target_encoder, m.source_encoder):
        assert_no_time_pooling_norm(sub)  # must not raise
    # And the encoders were causalised (CausalGroupNorm present, no plain GroupNorm).
    enc_norms = [type(mod) for mod in m.target_encoder.modules()]
    assert CausalGroupNorm in enc_norms
    assert nn.GroupNorm not in enc_norms


def test_construct_production_geometry():
    """Constructs at the production geometry (raw_len=5280 -> T=300) with the config front end."""
    prod_frontend = {
        "stages": [2, 2, 2, 2],
        "channels": [32, 64, 96, 128],
        "d_raw": 128,
        "antialias": True,
        "antialias_kernel": "binomial5",
        "gated": True,
        "norm_kind": "causal_group_norm",
        "norm_num_groups": 8,
        "first_kernels_fhr": [7, 31, 65],
        "first_kernels_up": [15, 65, 129],
        "decoder_head": "learned_basis",
        "basis_size": 8,
        "dropout": 0.05,
    }
    m = SeqVaeRawV4(
        frontend=prod_frontend,
        raw_len=5280,
        decimation=16,
        d_model=128,
        d_z=24,
        horizon=30,
        warmup_period=30,
        max_lag=90,
        num_heads=4,
        d_head=32,
        logvar_bound="smooth",
        posterior_logvar="residual",
        kld_support="anchor",
        head_structured_latent=True,
        freeze_unused_attn_proj=True,
        causal_norm=True,
        lag_bias_init="alibi_decay",
    )
    assert m.geometry.t == 300
    assert m.geometry.r == 16
    assert m.sequence_length == 300


@pytest.mark.parametrize("decoder_head", ["learned_basis", "linear"])
def test_construct_both_decoder_heads(decoder_head):
    m = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": decoder_head})
    assert m.baseline_decoder.decoder_head == decoder_head
    assert m.residual_decoder.decoder_head == decoder_head
