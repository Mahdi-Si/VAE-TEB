r"""Step-wise causality and prefix equivalence through the **whole model**, unconditionally.

This is the one claim in the package that is genuinely new rather than inherited, and the word doing
the work is *unconditionally*.

The conv-LSTM feature model makes the same claim with a qualifier. Its
``lag_attn_fs/tests/test_smear.py`` asserts step-wise causality under ``causal_norm: true`` and
records, in a paired test, that **without** the flag the claim is false: the conv-LSTM encoders carry
a time-pooling normaliser whose statistics run over the whole sequence, so anchor $t$'s state moves
when step $t + 1$ moves. The flag exists to causalise those normalisers after the fact.

These encoders have nothing to causalise. ``RMSNorm`` reduces over channels only, the convolutions
pad left, and attention is causal by kernel flag or by an explicit band mask -- so **``causal_norm``
is not a constructor keyword of this model at all**, and there is no flag beside these assertions for
a reader to go looking for. That is why the tiny fixture is asserted here as well as the shipped
budget: the tiny half is exactly the half that fails on the sibling, and passing it is what makes the
claim unconditional rather than configured.

Two separately falsifiable properties, both through the assembled model:

1. **Step-wise causality.** Resampling every input stream strictly after $t$ leaves the forward at
   $t$ bitwise unchanged -- the prior, the encoder states, and both forecasts, which is the part the
   encoder-level tests cannot reach because they stop at the encoder.
2. **Prefix equivalence**, $\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$. Strictly stronger:
   causality can hold while this fails, if positions were computed relative to the sequence end or
   the rotary table were indexed from a length-dependent offset.

**What is deliberately not ported.** The encoder-level versions of these properties, and every
primitive they rest on, are owned by seven modules in ``lag_attn_transformer_rws/tests``:
``test_attention_block.py``, ``test_blocks.py``, ``test_encoders.py``, ``test_encoder_causality.py``,
``test_rope.py``, ``test_source_window.py`` and ``test_prefix_equivalence.py``. This package *imports*
those objects rather than copying them, so a copy of their tests would be roughly $1{,}600$ lines
asserting properties of the same import. What is new is the composition -- the channel gate, the
availability-aware adapter, the prior and posterior heads, the lag cross-attention and the shared
decoder stacked on top -- and the negative controls, including the length-normalised rotary encoding
that proves prefix equivalence is not vacuous, live in the last of those modules.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    BATCH,
    CAUSALITY_TOL,
    MOVEMENT_TOL,
    SEQ_LEN,
    SHIPPED_KWARGS,
    relative_change,
    resample_after,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_transformer_rws.tests.test_prefix_equivalence import _PREFIX_TOL

#: Everything the forward returns per step or per anchor that must be bit-stable at the cut. The two
#: forecasts are the half no encoder-level test can reach, and the reason this file exists.
_PER_ANCHOR_KEYS = ("mu_base", "logvar_base", "mu_full", "logvar_full")
_PER_STEP_KEYS = ("mu_prior", "logvar_prior", "target_state", "source_state", "mu_post", "kld_per_t")


def _streams(length: int, seed: int = 0):
    """Seeded ``(y_st, y_ph, u_stream)`` at a chosen decimated length."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(BATCH, length, 43, generator=generator),
        torch.randn(BATCH, length, 66, generator=generator),
        torch.randn(BATCH, length, 58, generator=generator),
    )


def _built(kwargs) -> SeqVaeLagAttnTrfFs:
    """A model in ``eval()`` at zero dropout, so two forwards are the same computation twice."""
    torch.manual_seed(0)
    return SeqVaeLagAttnTrfFs(**dict(kwargs, dropout=0.0)).eval()


def _forward(model, streams):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams)


@pytest.fixture(scope="module")
def tiny_model() -> SeqVaeLagAttnTrfFs:
    """The tiny guarded model, built once. The half of the claim the conv-LSTM sibling fails."""
    return _built(tiny_gated_kwargs())


@pytest.fixture(scope="module")
def shipped_model() -> SeqVaeLagAttnTrfFs:
    """The production model at the shipped $120$ s reach budget, built once: $T = 300$ steps and a
    $78$-wide decoder head make each forward expensive enough to be worth sharing."""
    return _built(shipped_gated_kwargs())


# ---------------------------------------------------------------------------------------
# 1. Step-wise causality, through the whole model
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cut", [0, 1, 8])
def test_the_whole_model_reads_no_step_after_the_anchor_at_the_tiny_fixture(tiny_model, cut):
    r"""The half that fails on the conv-LSTM sibling without ``causal_norm: true``, and what makes
    the claim here unconditional.

    ``causal_norm`` is **not a constructor keyword of this model**, so there is no flag to qualify
    this with and nothing for a reader who knows the sibling's copy to go looking for: the property
    is a consequence of the encoders' construction -- ``RMSNorm`` reduces over channels only, the
    convolutions pad left, attention is causal by kernel flag or band mask -- rather than of a
    post-hoc repair.

    Every stream is resampled, not only the target: ``mu_full`` reads the source through the lag
    attention, whose window runs into the strict past, so a source-side leak would land there and
    nowhere else.
    """
    streams = _streams(SEQ_LEN)
    reference = _forward(tiny_model, streams)
    perturbed = _forward(
        tiny_model,
        tuple(resample_after(x, cut, seed=5 + index) for index, x in enumerate(streams)),
    )

    assert not hasattr(tiny_model, "causal_norm")
    for key in _PER_STEP_KEYS + _PER_ANCHOR_KEYS:
        assert torch.equal(reference[key][:, : cut + 1], perturbed[key][:, : cut + 1]), key
    # The paired control: the perturbation did reach the model, so the bit-stability is a statement
    # about causality rather than about a dead pathway.
    at_end = relative_change(
        reference["target_state"][:, -1], perturbed["target_state"][:, -1]
    )
    assert at_end > MOVEMENT_TOL, (
        f"the target state at the last step moved by only {at_end:.3e} -- the perturbation never "
        f"reached the model, so the bit-stability above proves nothing"
    )


@pytest.mark.parametrize("cut", [0, 150, 268])
def test_the_whole_model_reads_no_step_after_the_anchor_at_the_shipped_budget(shipped_model, cut):
    """The same claim at the production geometry and the shipped reach budget, where the gate gathers
    $78$ of $109$ target channels and delays every survivor.

    The gate can only move a channel's reach *earlier*, so it cannot break causality -- but it is an
    index operation along a chosen axis, and a delay applied along the wrong one is invisible to every
    shape check. Asserted rather than argued.
    """
    streams = _streams(int(SHIPPED_KWARGS["sequence_length"]), seed=1)
    reference = _forward(shipped_model, streams)
    perturbed = _forward(
        shipped_model,
        tuple(resample_after(x, cut, seed=11 + index) for index, x in enumerate(streams)),
    )

    assert shipped_model.decoder_out_channels == 78
    for key in _PER_STEP_KEYS + _PER_ANCHOR_KEYS:
        assert torch.equal(reference[key][:, : cut + 1], perturbed[key][:, : cut + 1]), key
    at_end = relative_change(
        reference["target_state"][:, -1], perturbed["target_state"][:, -1]
    )
    assert at_end > MOVEMENT_TOL


def test_the_forecast_at_an_anchor_is_a_function_of_that_anchors_history_alone(shipped_model):
    r"""The claim stated the way the objective reads it, which is what makes it worth having beyond
    the encoder-level tests.

    Anchor $t$'s forecast covers decimated steps $t+1 \ldots t+H$, and the target block it is scored
    against is gathered from exactly those steps. So the whole point is that $\mu[:, t]$ is
    computable from $y_{\le t}$ while the *target* comes from $y_{>t}$: if the two ever met, the
    reported ``pred_gap`` would be a measurement of leakage.
    """
    streams = _streams(int(SHIPPED_KWARGS["sequence_length"]), seed=2)
    anchor = 100
    reference = _forward(shipped_model, streams)
    # Perturb exactly the window the forecast at ``anchor`` is scored against.
    horizon_only = list(streams)
    generator = torch.Generator().manual_seed(21)
    for index, stream in enumerate(horizon_only):
        moved = stream.clone()
        window = moved[:, anchor + 1 : anchor + 1 + shipped_model.horizon]
        moved[:, anchor + 1 : anchor + 1 + shipped_model.horizon] = torch.randn(
            window.shape, generator=generator
        )
        horizon_only[index] = moved
    perturbed = _forward(shipped_model, tuple(horizon_only))

    for key in _PER_ANCHOR_KEYS:
        assert torch.equal(reference[key][:, anchor], perturbed[key][:, anchor]), key
    # And the target block *did* change, which is the other half of the statement.
    built_before = shipped_model._build_forecast_target(
        torch.cat(streams[:2], dim=-1)
    )
    built_after = shipped_model._build_forecast_target(
        torch.cat(tuple(horizon_only)[:2], dim=-1)
    )
    assert not torch.equal(built_before[:, anchor], built_after[:, anchor])


# ---------------------------------------------------------------------------------------
# 2. Prefix equivalence, through the whole model
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("anchor", [1, 5, SEQ_LEN - 5])
def test_encoding_a_prefix_reproduces_the_full_run_at_that_step_at_the_tiny_fixture(
    tiny_model, anchor
):
    r"""$\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$ through the assembled model.

    Strictly stronger than causality and separately falsifiable: causality can hold while this fails
    -- if positions were computed relative to the sequence end, if the rotary table were indexed from
    a length-dependent offset, or if a convolution padded on the right and the padding happened to be
    masked out downstream. The sibling's own copy of this property is encoder-level, over
    ``build_stream_encoder``; here it runs through the gate, the adapters, both heads, the lag
    attention and the shared decoder.

    A tolerance rather than a bitwise assertion, and the sibling's recorded one: the full and prefix
    runs compute the same quantity by different reduction orders -- a masked key contributes an exact
    zero in one and is simply absent in the other -- and ``scaled_dot_product_attention`` may pick a
    different kernel at a different sequence length.
    """
    streams = _streams(SEQ_LEN, seed=3)
    full = _forward(tiny_model, streams)
    prefix = _forward(tiny_model, tuple(x[:, : anchor + 1] for x in streams))

    for key in _PER_STEP_KEYS:
        movement = relative_change(full[key][:, anchor], prefix[key][:, -1])
        assert movement < _PREFIX_TOL, (
            f"{key}: reading step {anchor} of the full sequence differs from encoding "
            f"X[0:{anchor + 1}] and reading its last step, by {movement:.3e} relative"
        )


@pytest.mark.parametrize("anchor", [1, 150])
def test_encoding_a_prefix_reproduces_the_full_run_at_the_shipped_budget(shipped_model, anchor):
    """The same property at $T = 300$ with the shipped reach budget, where the attention context is
    two orders of magnitude longer and a length-dependent position would have far more room to
    show."""
    streams = _streams(int(SHIPPED_KWARGS["sequence_length"]), seed=4)
    full = _forward(shipped_model, streams)
    prefix = _forward(shipped_model, tuple(x[:, : anchor + 1] for x in streams))

    for key in _PER_STEP_KEYS:
        movement = relative_change(full[key][:, anchor], prefix[key][:, -1])
        assert movement < _PREFIX_TOL, f"{key} moved by {movement:.3e}"


def test_the_base_forecast_head_is_prefix_equivalent_under_the_shipped_base_decode():
    """The decoder is a per-anchor map on $z$, so prefix equivalence of the latent should carry to the
    forecast -- but the decoder is invoked on a *slice* ``z[:, :t_valid]`` whose length depends on the
    input, and a slice is exactly the kind of thing that shifts by one under a shorter input.

    Asserted under the shipped ``base_decode: mean``, and that is not a convenience. Under the
    constructor default ``'sample'`` the base branch decodes a *draw*, $z^p = \\mu^p + \\sigma^p
    \\epsilon$, and ``randn_like`` fills row-major over $(B, T, d_z)$ -- so the $\\epsilon$ at anchor
    $t$ of sample $b > 0$ sits at a different offset in the flat draw when $T$ changes, and the two
    runs differ by sampling noise rather than by anything about positions. That is a property of the
    harness, not of the model, and it is why prefix equivalence is a question about the latent and the
    *mean*-decoded branch. The next test records it rather than leaving it to be rediscovered.
    """
    model = _built(dict(tiny_gated_kwargs(), base_decode="mean"))
    streams = _streams(SEQ_LEN, seed=5)
    anchor = 5
    full = _forward(model, streams)
    prefix = _forward(model, tuple(x[:, : anchor + 1] for x in streams))

    assert model.base_decode == "mean"
    for key in ("mu_base", "logvar_base"):
        assert prefix[key].shape[1] == anchor + 1, key
        movement = relative_change(full[key][:, anchor], prefix[key][:, -1])
        assert movement < _PREFIX_TOL, f"{key} moved by {movement:.3e}"


def test_the_sampled_branch_differs_across_input_lengths_by_noise_and_not_by_position(tiny_model):
    r"""Recorded so the limit above reads as measured rather than as a weakened assertion.

    ``randn_like`` over $(B, T, d_z)$ is filled row-major, so at a fixed seed sample $0$'s
    $\epsilon$ at anchor $t$ is the same in a prefix run and a full run -- the offset $t\,d_z$ is
    unchanged -- while sample $1$'s sits at $T d_z + t d_z$ and moves with $T$. So the sampled branch
    is prefix-equivalent in the first batch element and not in the others, which is exactly the
    signature of a shape-dependent draw and not of a length-dependent position.
    """
    streams = _streams(SEQ_LEN, seed=7)
    anchor = 5
    full = _forward(tiny_model, streams)
    prefix = _forward(tiny_model, tuple(x[:, : anchor + 1] for x in streams))

    assert tiny_model.base_decode == "sample"
    assert BATCH > 1, "the whole distinction below needs a second batch element"
    assert relative_change(full["mu_base"][0, anchor], prefix["mu_base"][0, -1]) < _PREFIX_TOL
    assert relative_change(full["mu_base"][1, anchor], prefix["mu_base"][1, -1]) > MOVEMENT_TOL
    # And the latent's *mean* is prefix-equivalent in every element, which locates the difference in
    # the draw rather than in the encoder.
    for element in range(BATCH):
        assert (
            relative_change(full["mu_prior"][element, anchor], prefix["mu_prior"][element, -1])
            < _PREFIX_TOL
        )


def test_the_probe_is_not_vacuous_at_the_last_step(tiny_model):
    """A self-check on the harness: at $t = T - 1$ the prefix *is* the full sequence, so the
    assertion is trivially true no matter how positions are indexed -- which is why every anchor
    above is away from that end."""
    streams = _streams(SEQ_LEN, seed=6)
    full = _forward(tiny_model, streams)
    prefix = _forward(tiny_model, tuple(x[:, :SEQ_LEN] for x in streams))

    assert relative_change(full["mu_prior"][:, -1], prefix["mu_prior"][:, -1]) == 0.0
    # And a genuinely different prefix does move it, so the tolerance is not admitting everything.
    shorter = _forward(tiny_model, tuple(x[:, : SEQ_LEN - 1] for x in streams))
    assert relative_change(full["mu_prior"][:, -1], shorter["mu_prior"][:, -1]) > CAUSALITY_TOL
