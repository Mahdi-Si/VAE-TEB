r"""What the forward returns, on which axis, and what it deliberately does not return.

Two failures this file exists to catch, both of which produce a plausible number rather than an
exception:

* **a fabricated attention-shaped key.** The evaluation machinery this family shares reads
  ``attn_weights`` and ``source_kl_lag_map`` and turns them into a per-lag attribution of the
  divergence. This architecture computes neither, no such attribution exists for it, and satisfying
  the old contract with a tensor built from proposal norms would put a quantity in a column whose
  name misdescribes it.
* **an anchor-indexed tensor read as a time-indexed one.** Every latent tensor's second axis is a
  position in the decoded anchor set. It is a different length and a different order from the
  stored grid, and a reader that assumes otherwise gets a well-shaped wrong answer.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import bound_update
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_BATCH,
    TINY_D_MODEL,
    TINY_D_Z,
    TINY_FLOOR,
    TINY_MODEL_HORIZON,
    TINY_MODEL_STRIDE,
    TINY_N_LAGS,
    TINY_SEQ_LEN,
    TINY_TARGET_KEEP,
    build_tiny_model,
    tiny_model_kwargs,
    tiny_streams,
)

#: Keys every forward returns on the shipped arm, which builds both residual channels and the
#: persistence input. Written out rather than derived, so a key that appears or disappears fails
#: here instead of quietly changing what a downstream reader finds.
CONTRACT_KEYS = frozenset(
    {
        "anchor_index",
        "anchor_valid",
        "mu_prior",
        "logvar_prior",
        "raw_logvar_prior",
        "mu_post",
        "logvar_post",
        "z_prior",
        "z_post",
        "mu_base",
        "logvar_base",
        "mu_full",
        "logvar_full",
        "kld_per_anchor_dim",
        "kld_per_anchor",
        "update_mean",
        "raw_update_mean",
        "update_logsigma",
        "raw_update_logsigma",
        "lag_valid",
        "target_state",
        "conditioning_state",
        "persistence",
        "mu_prior_sat_frac",
        "residual_mu_sat_frac",
        "residual_logsigma_sat_frac",
        "cancellation_ratio_mean",
        "cancellation_numerator_mean",
        "cancellation_denominator_mean",
        "cancellation_ratio_scale",
        "cancellation_numerator_scale",
        "cancellation_denominator_scale",
    }
)

#: The keys the lag-attentive family emits that this architecture must not, under any arm.
FORBIDDEN_KEYS = (
    "attn_weights",
    "attended_source_heads",
    "source_kl_lag_map",
    "kld_per_t_per_head",
    "kld_per_t",
    "source_state",
)


def build_model(**overrides) -> SeqVaeLagResidualTrfCfs:
    """Build the tiny model with any keyword replaced.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model, in evaluation mode so a forward is reproducible.
    """
    return build_tiny_model(**overrides).eval()


def run(model: SeqVaeLagResidualTrfCfs, *, seed: int = 0, **kwargs):
    """Run one dense forward under a fixed noise seed.

    Args:
        model: The model.
        seed: Seed for the reparameterisation draw, so two runs are comparable.
        **kwargs: Extra forward keywords.

    Returns:
        The forward's dict.
    """
    y_st, y_ph, u_stream = tiny_streams()
    torch.manual_seed(seed)
    return model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, **kwargs)


# =================================================================================================
# The key set
# =================================================================================================
def test_the_forward_returns_exactly_the_contract() -> None:
    """No key more and no key fewer, on the shipped arm."""
    assert set(run(build_model())) == set(CONTRACT_KEYS)


@pytest.mark.parametrize("key", FORBIDDEN_KEYS)
def test_no_attention_shaped_key_is_emitted(key: str) -> None:
    """Under every arm, including the one that retains the proposals for diagnostics.

    This architecture computes no attention distribution and no per-lag allocation of the
    divergence, and none exists for it: the cross terms in the summed update can reinforce or
    cancel, so a nonnegative per-lag decomposition is not merely unavailable but absent.
    """
    for extra in ({}, {"return_proposals": True}):
        assert key not in run(build_model(), **extra)
        assert key not in run(build_model(mean_only_residual=True), **extra)


def test_the_mean_only_arm_omits_the_scale_keys_rather_than_zeroing_them() -> None:
    """A zero tensor would let a reader report a scale update the model cannot make."""
    outputs = run(build_model(mean_only_residual=True))
    for absent in (
        "update_logsigma",
        "raw_update_logsigma",
        "cancellation_ratio_scale",
        "cancellation_numerator_scale",
        "cancellation_denominator_scale",
    ):
        assert absent not in outputs
    # And the full log-variance is the prior's, exactly.
    assert torch.equal(outputs["logvar_post"], outputs["logvar_prior"])
    assert "residual_logsigma_sat_frac" in outputs  # reported as zero, so the surface is stable


def test_the_persistence_key_follows_the_decoder() -> None:
    """Present only where the decoder was built with the residual, as the family's convention is."""
    assert "persistence" in run(build_model())
    assert "persistence" not in run(build_model(persistence_residual=False))


# =================================================================================================
# The shapes, and which axis they carry
# =================================================================================================
def test_every_tensor_matches_the_declared_contract_densely() -> None:
    """At stride one, where the anchor set is the dense range from the floor to the ceiling."""
    model = build_model()
    outputs = run(model, return_proposals=True)

    n_anchors = TINY_SEQ_LEN - TINY_MODEL_HORIZON - TINY_FLOOR
    kept = len(TINY_TARGET_KEEP)
    expected = {
        "anchor_index": (TINY_BATCH, n_anchors),
        "anchor_valid": (TINY_BATCH, n_anchors),
        "mu_prior": (TINY_BATCH, n_anchors, TINY_D_Z),
        "mu_post": (TINY_BATCH, n_anchors, TINY_D_Z),
        "z_prior": (TINY_BATCH, n_anchors, TINY_D_Z),
        "z_post": (TINY_BATCH, n_anchors, TINY_D_Z),
        "kld_per_anchor_dim": (TINY_BATCH, n_anchors, TINY_D_Z),
        "kld_per_anchor": (TINY_BATCH, n_anchors),
        "update_mean": (TINY_BATCH, n_anchors, TINY_D_Z),
        "mu_base": (TINY_BATCH, n_anchors, TINY_MODEL_HORIZON, kept),
        "mu_full": (TINY_BATCH, n_anchors, TINY_MODEL_HORIZON, kept),
        "logvar_full": (TINY_BATCH, n_anchors, TINY_MODEL_HORIZON, kept),
        "persistence": (TINY_BATCH, n_anchors, kept),
        "lag_valid": (TINY_BATCH, n_anchors, TINY_N_LAGS),
        "conditioning_state": (TINY_BATCH, n_anchors, TINY_D_MODEL),
        "mean_proposals": (TINY_BATCH, n_anchors, TINY_N_LAGS, TINY_D_Z),
        "scale_proposals": (TINY_BATCH, n_anchors, TINY_N_LAGS, TINY_D_Z),
    }
    for name, shape in expected.items():
        assert tuple(outputs[name].shape) == shape, name

    # The one tensor that keeps the STORED grid, and it is the encoder's output rather than
    # anything the latent touches.
    assert tuple(outputs["target_state"].shape) == (TINY_BATCH, TINY_SEQ_LEN, TINY_D_MODEL)


def test_the_anchor_axis_is_not_the_time_axis() -> None:
    """At the training stride they are different lengths and the anchors are not consecutive."""
    model = build_model()
    y_st, y_ph, u_stream = tiny_streams()
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0)

    anchors = outputs["anchor_index"]
    assert anchors.shape[1] < TINY_SEQ_LEN
    assert outputs["mu_prior"].shape[1] == anchors.shape[1]
    steps = anchors[0].tolist()
    assert steps == list(range(TINY_FLOOR, TINY_SEQ_LEN - TINY_MODEL_HORIZON, TINY_MODEL_STRIDE))


def test_the_last_label_lies_on_the_last_stored_step() -> None:
    """The anchor ceiling is what keeps the furthest forecast window inside the record."""
    model = build_model()
    outputs = run(model)
    last_anchor = int(outputs["anchor_index"].max())
    assert last_anchor + TINY_MODEL_HORIZON == TINY_SEQ_LEN - 1


def test_padded_anchors_repeat_a_legal_index_and_are_marked_invalid() -> None:
    """A padded slot holding a distinct legal anchor would be gathered and scored twice."""
    model = build_model()
    y_st, y_ph, u_stream = tiny_streams()
    # Phases 2 and 3 are the short rows at this stride, so padding actually exists.
    phase = torch.tensor([0, 2, 3], dtype=torch.long)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=phase)

    anchors, valid = outputs["anchor_index"], outputs["anchor_valid"]
    assert not bool(valid.all()), "no padding in the fixture; the check is vacuous"
    for row in range(anchors.shape[0]):
        real = anchors[row][valid[row]]
        padded = anchors[row][~valid[row]]
        assert bool((anchors[row] < TINY_SEQ_LEN - TINY_MODEL_HORIZON).all())
        if padded.numel():
            assert bool((padded == real[-1]).all())


# =================================================================================================
# The fusion, end to end
# =================================================================================================
def test_the_returned_proposals_reproduce_the_returned_update() -> None:
    """The retained array and the accumulated total are the same computation, not two."""
    model = build_model()
    outputs = run(model, return_proposals=True)

    expected_raw = model.lag_scale * outputs["mean_proposals"].sum(dim=2)
    assert torch.allclose(outputs["raw_update_mean"], expected_raw, atol=1e-6)
    assert torch.allclose(
        outputs["update_mean"],
        bound_update(expected_raw, model.residual_mu_scale),
        atol=1e-6,
    )


def test_the_full_parameters_are_the_prior_plus_the_bounded_residual() -> None:
    r"""$\mu^q = \mu^p + \sigma^p a$ and $\lambda^q = \lambda^p + 2b$, as returned."""
    model = build_model()
    outputs = run(model)
    sigma_prior = torch.exp(0.5 * outputs["logvar_prior"])

    assert torch.allclose(
        outputs["mu_post"],
        outputs["mu_prior"] + sigma_prior * outputs["update_mean"],
        atol=1e-6,
    )
    assert torch.allclose(
        outputs["logvar_post"],
        outputs["logvar_prior"] + 2.0 * outputs["update_logsigma"],
        atol=1e-6,
    )


def test_the_returned_divergence_agrees_with_the_family_formula() -> None:
    """Measured on the model's own outputs, not on a re-derivation of its parameters."""
    model = build_model()
    outputs = run(model)
    general = model.kld_tensor(
        outputs["mu_prior"],
        outputs["logvar_prior"],
        outputs["mu_post"],
        outputs["logvar_post"],
    )
    assert torch.allclose(outputs["kld_per_anchor_dim"], general, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        outputs["kld_per_anchor"], outputs["kld_per_anchor_dim"].sum(dim=-1), atol=1e-6
    )


def test_the_cancellation_parts_come_back_beside_the_ratio() -> None:
    """A near-zero ratio from cancellation and from silence are not the same finding."""
    model = build_model()
    outputs = run(model, return_proposals=True)
    proposals = outputs["mean_proposals"]

    assert torch.allclose(
        outputs["cancellation_numerator_mean"], proposals.sum(dim=2).norm(dim=-1), atol=1e-6
    )
    assert torch.allclose(
        outputs["cancellation_denominator_mean"],
        proposals.norm(dim=-1).sum(dim=2),
        atol=1e-6,
    )


# Chunking has its own file: the agreement checks, the measured reassociation tolerance and the
# detachment check all belong together, and a second copy here would be the same comparison run
# twice at two tolerances.
