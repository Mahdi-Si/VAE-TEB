r"""The lag embeddings, the shared proposal head and its selector.

Three properties here have no other proof in the repository, and each fails silently:

* the final projection must still be **exactly zero** after the family's generic initialisation
  pass, which xavier-fills every linear layer built before it. The same repair exists twice already
  elsewhere in this family, which is why it is checked rather than assumed;
* a proposal must read **one stored source time**, which is the architectural claim and is measured
  by a Jacobian rather than argued from the absence of a temporal operator;
* the split input projection must equal the **concatenated** form it stands in for, because the
  slice order is the one thing a later edit could reverse without any shape changing.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.blocks import initialization
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import (
    LAG_EMBED_STD,
    LagProposalHead,
)
from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import (
    IDENTITY_WIDTH,
    PointwiseSourceEncoder,
    gather_lag_window,
    lag_validity,
)
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_BATCH,
    TINY_C_U,
    TINY_D_MODEL,
    TINY_D_Z,
    TINY_N_LAGS,
    TINY_SOURCE_WARMUP,
)

SOURCE_DIM = TINY_C_U * IDENTITY_WIDTH


def build_head(**overrides) -> LagProposalHead:
    """Build a proposal head at the tiny geometry, with any keyword replaced.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The head.
    """
    kwargs = dict(
        d_model=TINY_D_MODEL,
        d_z=TINY_D_Z,
        n_lags=TINY_N_LAGS,
        source_dim=SOURCE_DIM,
    )
    kwargs.update(overrides)
    return LagProposalHead(**kwargs)


def trained_head(seed: int = 7, **overrides) -> LagProposalHead:
    """A head whose output projection has been moved off zero, as training would move it.

    Every assertion about a *nonzero* proposal is vacuous at initialisation, because the head is
    built to emit exactly zero. This stands in for the first optimizer step.

    Args:
        seed: Seed for the draw.
        **overrides: Constructor keywords to replace.

    Returns:
        The head, with a nonzero final projection.
    """
    head = build_head(**overrides)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.output_proj.weight.normal_(0.0, 0.3, generator=generator)
        head.output_proj.bias.normal_(0.0, 0.3, generator=generator)
    return head


def make_window(seed: int = 11, n_anchors: int = 4) -> torch.Tensor:
    """A seeded source window at the tiny geometry, in the gather's own five-axis shape.

    Args:
        seed: Seed for the draw.
        n_anchors: Anchors on the second axis.

    Returns:
        A $(B, A, L, C_U, W)$ float tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        TINY_BATCH,
        n_anchors,
        TINY_N_LAGS,
        TINY_C_U,
        IDENTITY_WIDTH,
        generator=generator,
    )


def make_state(seed: int = 13, n_anchors: int = 4) -> torch.Tensor:
    """A seeded target state matching :func:`make_window`.

    Args:
        seed: Seed for the draw.
        n_anchors: Anchors on the second axis.

    Returns:
        A $(B, A, d_h)$ float tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(TINY_BATCH, n_anchors, TINY_D_MODEL, generator=generator)


# =================================================================================================
# Construction and initialisation
# =================================================================================================
def test_the_output_projection_starts_at_exactly_zero() -> None:
    """So the full distribution is the prior at initialisation, for every input."""
    head = build_head()
    assert torch.all(head.output_proj.weight == 0.0)
    assert torch.all(head.output_proj.bias == 0.0)

    mean, scale = head(make_state(), make_window())
    assert torch.all(mean == 0.0)
    assert scale is not None and torch.all(scale == 0.0)


def test_the_zeroing_survives_the_generic_initialisation_pass() -> None:
    """The pass xavier-fills every linear layer, including this one, and undoes the constructor.

    The repair is a second call after the pass, which is exactly what a composing model's
    post-initialisation block does. Without it the model starts with a nonzero source correction and
    the exact zero-update start silently does not hold.
    """
    head = build_head()
    initialization(head)
    assert bool((head.output_proj.weight != 0.0).any()), "the pass did not refill; test is vacuous"

    head.zero_output()
    assert torch.all(head.output_proj.weight == 0.0)
    assert torch.all(head.output_proj.bias == 0.0)
    # Idempotent, because the composing model calls it from two places.
    head.zero_output()
    assert torch.all(head.output_proj.weight == 0.0)


def test_the_lag_embedding_survives_the_generic_initialisation_pass() -> None:
    """It carries no repair hook, and this is why: the pass leaves an embedding alone."""
    head = build_head()
    before = head.lag_embedding.weight.detach().clone()
    assert float(before.std()) == pytest.approx(LAG_EMBED_STD, rel=0.6)

    initialization(head)
    assert torch.equal(head.lag_embedding.weight, before)


def test_the_mean_only_arm_builds_no_scale_parameters() -> None:
    """A different module tree, not a flag consulted in the forward.

    A scale head that exists but is never read is a starved parameter block under a distributed run
    and a claim in the checkpoint that the model updates a variance it does not.
    """
    full = build_head()
    lean = build_head(mean_only=True)

    assert full.output_proj.out_features == 2 * TINY_D_Z
    assert lean.output_proj.out_features == TINY_D_Z
    assert lean.output_proj.weight.numel() < full.output_proj.weight.numel()

    mean, scale = lean(make_state(), make_window())
    assert scale is None
    assert mean.shape == (TINY_BATCH, 4, TINY_N_LAGS, TINY_D_Z)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(d_model=0), "d_model must be > 0"),
        (dict(d_z=0), "d_z must be > 0"),
        (dict(n_lags=0), "n_lags must be > 0"),
        (dict(source_dim=0), "source_dim must be > 0"),
        (dict(lag_embed_dim=0), "lag_embed_dim must be > 0"),
        (dict(hidden=0), "hidden must be > 0"),
    ],
)
def test_the_constructor_refuses_a_degenerate_width(kwargs, message: str) -> None:
    """A zero width builds a head that cannot be reached, which is a distributed-run hazard."""
    with pytest.raises(ValueError, match=message):
        build_head(**kwargs)


# =================================================================================================
# The forward
# =================================================================================================
def test_the_split_input_projection_equals_the_concatenated_form() -> None:
    r"""The slice order **is** the documented concatenation order $[h \Vert E \Vert \zeta]$.

    Reversing two slices would leave every shape correct and train a head reading the source vector
    through the target state's weights. Nothing else in the suite would notice.
    """
    head = trained_head()
    state, window = make_state(), make_window()
    batch, n_anchors = state.shape[0], state.shape[1]

    flat_source = window.reshape(batch, n_anchors, TINY_N_LAGS, -1)
    lags = torch.arange(TINY_N_LAGS)
    embedded = head.lag_embedding(lags)[None, None, :, :].expand(
        batch, n_anchors, -1, -1
    )
    concatenated = torch.cat(
        [
            state.unsqueeze(2).expand(-1, -1, TINY_N_LAGS, -1),
            flat_source,
            embedded,
        ],
        dim=-1,
    )
    reference = torch.nn.functional.gelu(head.input_proj(concatenated))
    reference = torch.nn.functional.gelu(head.hidden_proj(reference))
    expected_mean, expected_scale = head.output_proj(reference).split(TINY_D_Z, dim=-1)

    mean, scale = head(state, window)
    assert torch.allclose(mean, expected_mean, atol=1e-6, rtol=1e-6)
    assert scale is not None
    assert torch.allclose(scale, expected_scale, atol=1e-6, rtol=1e-6)


def test_a_proposal_reads_exactly_one_stored_source_time() -> None:
    r"""$\partial r_{t,\ell} / \partial E_{t,k} = 0$ for $k \neq \ell$, by Jacobian.

    The locality claim. A temporal operator added to the source pathway, or a head that pooled over
    the lag axis, would leave the shapes and every other test in this file intact.
    """
    head = trained_head()
    state = make_state(n_anchors=1)
    window = make_window(n_anchors=1).requires_grad_(True)

    def propose(source: torch.Tensor) -> torch.Tensor:
        """Mean proposals for one batch element, so the Jacobian stays small."""
        return head(state, source)[0][0]

    jacobian = torch.autograd.functional.jacobian(propose, window)
    # (A_out, L_out, d_z) against (B_in, A_in, L_in, C_U, W). Reduce every axis but the two lag
    # axes, so what is left is how much each output lag depends on each input lag.
    reduced = jacobian.abs().sum(dim=2)  # (A_out, L_out, B_in, A_in, L_in, C_U, W)
    reduced = reduced.sum(dim=(2, 3, 5, 6))  # (A_out, L_out, L_in)
    for lag_out in range(TINY_N_LAGS):
        for lag_in in range(TINY_N_LAGS):
            total = float(reduced[0, lag_out, lag_in])
            if lag_out == lag_in:
                assert total > 0.0, f"lag {lag_out} does not read its own source time"
            else:
                assert total == 0.0, f"lag {lag_out} reads lag {lag_in}"


def test_a_proposal_may_combine_channels_at_its_own_source_time() -> None:
    """Locality is over *times*, not over channels: the head is allowed to read the whole vector."""
    head = trained_head()
    state, window = make_state(n_anchors=1), make_window(n_anchors=1)

    baseline, _ = head(state, window)
    moved = window.clone()
    moved[0, 0, 2, 0, 0] += 1.0  # one channel of one lag
    perturbed, _ = head(state, moved)

    difference = (perturbed - baseline).abs()
    assert float(difference[0, 0, 2].max()) > 0.0
    other_lags = [lag for lag in range(TINY_N_LAGS) if lag != 2]
    assert float(difference[0, 0, other_lags].max()) == 0.0


def test_lag_identity_makes_the_head_sensitive_to_lag_order() -> None:
    """Without it, summing a shared lag-blind map would be invariant to permuting source times."""
    head = trained_head()
    state, window = make_state(), make_window()

    straight, _ = head(state, window)
    swapped_window = window.clone()
    swapped_window[:, :, [0, 1]] = swapped_window[:, :, [1, 0]]
    swapped, _ = head(state, swapped_window)

    # A lag-blind head would satisfy swapped[:, :, 0] == straight[:, :, 1] exactly.
    assert not torch.allclose(swapped[:, :, 0], straight[:, :, 1])


def test_a_zero_selector_silences_one_lag_and_leaves_the_others_untouched() -> None:
    """The intervention contract: suppression is exact, and it is local to the band it names."""
    head = trained_head()
    state, window = make_state(), make_window()

    baseline_mean, baseline_scale = head(state, window)
    selector = torch.ones(TINY_BATCH, state.shape[1], TINY_N_LAGS)
    selector[:, :, 3] = 0.0
    suppressed_mean, suppressed_scale = head(state, window, selector=selector)

    assert torch.all(suppressed_mean[:, :, 3] == 0.0)
    assert suppressed_scale is not None and torch.all(suppressed_scale[:, :, 3] == 0.0)
    kept = [lag for lag in range(TINY_N_LAGS) if lag != 3]
    assert torch.equal(suppressed_mean[:, :, kept], baseline_mean[:, :, kept])
    assert baseline_scale is not None
    assert torch.equal(suppressed_scale[:, :, kept], baseline_scale[:, :, kept])


def test_an_all_zero_selector_silences_every_proposal() -> None:
    """The invariant every source control is read against."""
    head = trained_head()
    mean, scale = head(
        make_state(),
        make_window(),
        selector=torch.zeros(TINY_BATCH, 4, TINY_N_LAGS),
    )
    assert torch.all(mean == 0.0)
    assert scale is not None and torch.all(scale == 0.0)


def test_an_unavailable_lag_contributes_an_exact_zero_not_a_learned_constant() -> None:
    """The head's response to an all-zero source vector is a constant, and it is not nothing.

    Without the validity gate, a lag whose every channel is out of range or still cold would push
    the head's bias term into the sum once per empty lag, which is a source correction produced by
    the absence of a source.
    """
    head = trained_head()
    state, window = make_state(), make_window()

    empty = torch.ones(TINY_BATCH, state.shape[1], TINY_N_LAGS, dtype=torch.bool)
    empty[:, :, 4] = False
    gated_mean, _ = head(state, window, lag_valid=empty)
    assert torch.all(gated_mean[:, :, 4] == 0.0)

    # And the constant it would otherwise have contributed is genuinely nonzero, so the assertion
    # above is a measurement rather than a tautology.
    zeroed_window = window.clone()
    zeroed_window[:, :, 4] = 0.0
    ungated_mean, _ = head(state, zeroed_window)
    assert float(ungated_mean[:, :, 4].abs().max()) > 0.0


def test_the_validity_gate_and_the_selector_compose() -> None:
    """Either one alone silences a lag, and both together still do."""
    head = trained_head()
    state, window = make_state(), make_window()
    n_anchors = state.shape[1]

    valid = torch.ones(TINY_BATCH, n_anchors, TINY_N_LAGS, dtype=torch.bool)
    valid[:, :, 1] = False
    selector = torch.ones(TINY_BATCH, n_anchors, TINY_N_LAGS)
    selector[:, :, 2] = 0.0

    mean, _ = head(state, window, lag_valid=valid, selector=selector)
    assert torch.all(mean[:, :, 1] == 0.0)
    assert torch.all(mean[:, :, 2] == 0.0)
    assert float(mean[:, :, 0].abs().max()) > 0.0


def test_the_head_consumes_the_gather_output_directly() -> None:
    """The two modules meet on one width, reported by the encoder rather than restated here."""
    encoder = PointwiseSourceEncoder(c_u=TINY_C_U, warmup_steps=TINY_SOURCE_WARMUP)
    head = trained_head(source_dim=encoder.source_dim)

    generator = torch.Generator().manual_seed(3)
    stream = torch.randn(TINY_BATCH, 20, TINY_C_U, generator=generator)
    anchors = torch.arange(12, 16, dtype=torch.long)[None, :].expand(TINY_BATCH, -1)

    encoded, mask = encoder(stream)
    window, window_mask = gather_lag_window(encoded, mask, anchors, n_lags=TINY_N_LAGS)
    mean, scale = head(
        make_state(n_anchors=4), window, lag_valid=lag_validity(window_mask)
    )

    assert mean.shape == (TINY_BATCH, 4, TINY_N_LAGS, TINY_D_Z)
    assert scale is not None and scale.shape == mean.shape


@pytest.mark.parametrize(
    "state_shape, window_shape, message",
    [
        ((TINY_BATCH, 4), (TINY_BATCH, 4, TINY_N_LAGS, TINY_C_U, IDENTITY_WIDTH),
         "target_state must be 3-D"),
        ((TINY_BATCH, 4, TINY_D_MODEL + 1),
         (TINY_BATCH, 4, TINY_N_LAGS, TINY_C_U, IDENTITY_WIDTH),
         "against d_model"),
        ((TINY_BATCH, 4, TINY_D_MODEL), (TINY_BATCH, 4, TINY_N_LAGS),
         "at least 4-D"),
        ((TINY_BATCH, 4, TINY_D_MODEL),
         (TINY_BATCH, 5, TINY_N_LAGS, TINY_C_U, IDENTITY_WIDTH),
         "disagree with the target state"),
        ((TINY_BATCH, 4, TINY_D_MODEL),
         (TINY_BATCH, 4, TINY_N_LAGS + 1, TINY_C_U, IDENTITY_WIDTH),
         "requested lag slots"),
        ((TINY_BATCH, 4, TINY_D_MODEL),
         (TINY_BATCH, 4, TINY_N_LAGS, TINY_C_U + 1, IDENTITY_WIDTH),
         "against source_dim"),
    ],
)
def test_the_forward_refuses_a_mismatched_pair(
    state_shape, window_shape, message: str
) -> None:
    """Each mismatch would otherwise identify the wrong lag slot or read the wrong weights."""
    head = build_head()
    with pytest.raises(ValueError, match=message):
        head(torch.zeros(state_shape), torch.zeros(window_shape))
