r"""S0-T02: the reuse surface re-exports the *identical* origin objects.

If any symbol were accidentally re-implemented or shadowed, ``is`` identity would break and a
warm-start state_dict / ``isinstance`` check downstream would silently mismatch.
"""
from __future__ import annotations

from model.vae_teb_prediction.model.model_raw import reuse

from model.vae_teb_prediction.model import vae_teb_lag_attn_v1 as v1
from model.vae_teb_prediction.model import vae_teb_lag_attn_v3 as v3
from model.vae_teb_prediction.model import vae_teb_model_prediction as prim


def test_v3_symbols_are_identical() -> None:
    assert reuse.CausalGroupNorm is v3.CausalGroupNorm
    assert reuse.causalize_norms is v3.causalize_norms
    assert reuse.smooth_bound is v3.smooth_bound
    assert reuse.PriorHeadV3 is v3.PriorHeadV3
    assert reuse.PosteriorHeadV3 is v3.PosteriorHeadV3
    assert reuse.BaselineFutureDecoderV3 is v3.BaselineFutureDecoderV3
    assert reuse.ResidualFutureDecoderV3 is v3.ResidualFutureDecoderV3
    assert reuse.SeqVaeLagAttnV3 is v3.SeqVaeLagAttnV3


def test_v1_symbols_are_identical() -> None:
    assert reuse.HorizonDecoderCore is v1.HorizonDecoderCore
    assert reuse.TEAnalysisHead is v1.TEAnalysisHead
    # Explicitly the v1 encoders (NOT the same-named ones in vae_teb_model_prediction).
    assert reuse.TargetEncoder is v1.TargetEncoder
    assert reuse.SourceEncoder is v1.SourceEncoder


def test_primitive_symbols_are_identical() -> None:
    assert reuse.CausalConv1d is prim.CausalConv1d
    assert reuse.CausalMultiChannelConvBlock is prim.CausalMultiChannelConvBlock
    assert reuse.ResidualMLP is prim.ResidualMLP
    assert reuse.geometric_schedule is prim.geometric_schedule
    assert reuse.initialization is prim.initialization


def test_encoder_disambiguation() -> None:
    # The v1 encoders are the ones re-exported; if vae_teb_model_prediction also defines
    # TargetEncoder/SourceEncoder they must be DIFFERENT objects (documented hazard).
    prim_target = getattr(prim, "TargetEncoder", None)
    if prim_target is not None:
        assert reuse.TargetEncoder is not prim_target


def test_all_exports_present() -> None:
    for name in reuse.__all__:
        assert hasattr(reuse, name), f"missing re-export: {name}"
