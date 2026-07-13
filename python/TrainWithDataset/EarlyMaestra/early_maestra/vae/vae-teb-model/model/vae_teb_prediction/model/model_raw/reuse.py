r"""Single import surface for the v3/v1 symbols the raw v4 model builds on.

`SeqVaeRawV4` inherits the entire v3 scientific-cleanliness core, so v4 code imports the shared
pieces from **here** rather than reaching into the origin modules directly. Re-exporting through
one module makes the reuse boundary explicit and gives the Sprint-0 contract test a single place
to assert that every symbol is the *identical* object as its origin (no accidental shadow/copy).

Disambiguation note: two distinct ``TargetEncoder``/``SourceEncoder`` classes exist in the
codebase -- the v1 ones (:mod:`vae_teb_lag_attn_v1`, single already-projected ``(B, T, d_model)``
input) and a differently-typed pair in :mod:`vae_teb_model_prediction`. v4 wires the front-end
tokens into the **v1** encoders, so those are the ones re-exported here.
"""
from __future__ import annotations

# -- v3 scientific-cleanliness core (heads, causal norm, bound helper, base model) -----------
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
    BaselineFutureDecoderV3,
    CausalGroupNorm,
    PosteriorHeadV3,
    PriorHeadV3,
    ResidualFutureDecoderV3,
    SeqVaeLagAttnV3,
    causalize_norms,
    smooth_bound,
)

# -- v1 shared modules (decoder core, TE head, causal encoders) ------------------------------
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (
    HorizonDecoderCore,
    SourceEncoder,
    TargetEncoder,
    TEAnalysisHead,
)

# -- Shared low-level primitives (causal conv, MLP, init, schedule) --------------------------
from model.vae_teb_prediction.model.vae_teb_model_prediction import (
    CausalConv1d,
    CausalMultiChannelConvBlock,
    ResidualMLP,
    geometric_schedule,
    initialization,
)

__all__ = [
    # v3
    "BaselineFutureDecoderV3",
    "CausalGroupNorm",
    "PosteriorHeadV3",
    "PriorHeadV3",
    "ResidualFutureDecoderV3",
    "SeqVaeLagAttnV3",
    "causalize_norms",
    "smooth_bound",
    # v1
    "HorizonDecoderCore",
    "SourceEncoder",
    "TargetEncoder",
    "TEAnalysisHead",
    # primitives
    "CausalConv1d",
    "CausalMultiChannelConvBlock",
    "ResidualMLP",
    "geometric_schedule",
    "initialization",
]
