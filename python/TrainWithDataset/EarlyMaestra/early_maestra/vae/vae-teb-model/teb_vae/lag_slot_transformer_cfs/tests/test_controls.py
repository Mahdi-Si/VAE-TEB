r"""The four interventions, and the identities that make every margin they produce a measurement.

Two of the checks here are cheap and load-bearing out of all proportion to their size. Suppressing
an **empty** band must reproduce the matched forward bitwise, and suppressing **every** band must
reproduce the prior branch bitwise. Together they pin the intervention path to the forward path at
both ends, so a margin measured between them is a difference of predictions rather than a
difference between two implementations of the same arithmetic that happen to disagree in the third
decimal. Neither identity is expensive and neither holds by accident: the first fails the moment
the suppressed arm re-derives the summed update instead of subtracting from it, and the second
fails the moment that subtraction is used where a sum over no lags is meant.

The rest are about what an intervention must *not* move. A replacement that shifted the
availability announcement, or the metadata clock, would be reported as a source-value effect while
carrying a schedule effect; and a permutation that paired a segment with its own recording's
neighbour is not a stranger's source at all.

**Every arm is exercised on a model whose source pathway has been woken.** A freshly constructed
model has an exactly zero output projection, so every proposal is zero, every intervention is a
no-op, and every assertion below would pass on a control that does nothing whatsoever.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets import controls

from .conftest import build_tiny_model, tiny_streams


def woken_model(**overrides):
    """Build the tiny model and give its proposal head weights that are not zero.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model, in evaluation mode, with a source pathway that moves the full branch.
    """
    model = build_tiny_model(**overrides).eval()
    torch.manual_seed(20260909)
    torch.nn.init.normal_(model.proposal_head.output_proj.weight, std=0.3)
    torch.nn.init.normal_(model.proposal_head.output_proj.bias, std=0.3)
    return model


@pytest.fixture
def matched():
    """A woken model and one matched forward carrying its per-lag proposals.

    Returns:
        ``(model, streams, outputs)``.
    """
    model = woken_model()
    streams = tiny_streams()
    with torch.no_grad():
        outputs = model(
            *streams, anchor_phase=0, anchor_stride=1, return_proposals=True
        )
    # The pathway actually moves the branch, without which every assertion below is vacuous.
    assert float(outputs["update_mean"].abs().max()) > 0.0
    return model, streams, outputs


# =============================================================================
# Proposal suppression
# =============================================================================
def test_suppressing_an_empty_band_reproduces_the_matched_forward_bitwise(matched) -> None:
    """The reference arm. A margin against it is a measurement only if it is exactly zero."""
    model, _streams, outputs = matched
    empty = torch.zeros(model.n_lags, dtype=torch.bool)

    result = controls.suppressed_parameters(model, outputs, empty)

    assert torch.equal(result["mu_post"], outputs["mu_post"])
    assert torch.equal(result["logvar_post"], outputs["logvar_post"])
    assert torch.equal(result["kld_per_anchor"], outputs["kld_per_anchor"])


def test_suppressing_every_lag_reproduces_the_prior_branch_bitwise(matched) -> None:
    """A sum over no lags is zero by definition, and the arm is written so that it is.

    Reaching it by subtracting the whole chunk-accumulated total from itself would leave a residue
    of a few units in the last place -- a full branch that merely nearly equals the prior, and a
    "source removed" arm whose divergence is a small positive number rather than zero.
    """
    model, _streams, outputs = matched
    every = torch.ones(model.n_lags, dtype=torch.bool)

    result = controls.suppressed_parameters(model, outputs, every)

    assert torch.equal(result["mu_post"], outputs["mu_prior"])
    assert torch.equal(result["logvar_post"], outputs["logvar_prior"])
    assert float(result["kld_per_anchor"].abs().max()) == 0.0


def test_suppressing_a_band_moves_the_full_branch_and_its_divergence(matched) -> None:
    """A partial removal is a real intervention, which the two identities above cannot show.

    Both endpoints hold for an arm that ignores its band argument entirely, so this is what
    separates a working suppression from one that always returns the matched or the prior value.
    """
    model, _streams, outputs = matched
    band = controls.band_lag_mask(model.n_lags, 1, 2)

    result = controls.suppressed_parameters(model, outputs, band)

    assert float((result["mu_post"] - outputs["mu_post"]).abs().max()) > 0.0
    assert float((result["kld_per_anchor"] - outputs["kld_per_anchor"]).abs().max()) > 0.0
    assert result["mu_post"].shape == outputs["mu_post"].shape


def test_the_bounds_still_hold_under_suppression(matched) -> None:
    r"""$|\mu^q_d - \mu^p_d| \le a_{\max}\sigma^p_d$ on the intervened arm too.

    The limiter is re-applied after the suppressed sum rather than the matched bounded update
    being scaled down, which is what keeps the guarantee. A suppressed arm that rescaled the
    already-bounded value would satisfy the bound and be the wrong number.
    """
    model, _streams, outputs = matched
    band = controls.band_lag_mask(model.n_lags, 0, 1)

    result = controls.suppressed_parameters(model, outputs, band)

    sigma_prior = torch.exp(0.5 * outputs["logvar_prior"])
    gap = (result["mu_post"] - outputs["mu_prior"]).abs()
    assert bool((gap <= model.residual_mu_scale * sigma_prior + 1e-6).all())
    ratio = torch.exp(0.5 * (result["logvar_post"] - outputs["logvar_prior"]))
    limit = float(torch.exp(torch.tensor(model.residual_logsigma_scale)))
    assert bool(((ratio <= limit + 1e-6) & (ratio >= 1.0 / limit - 1e-6)).all())


def test_a_suppression_without_the_cached_proposals_refuses(matched) -> None:
    """The proposals are the whole input; without them the arm would suppress nothing and say so
    nowhere."""
    model, streams, _outputs = matched
    with torch.no_grad():
        thin = model(*streams, anchor_phase=0, anchor_stride=1)

    with pytest.raises(KeyError, match="return_proposals"):
        controls.suppressed_parameters(
            model, thin, torch.zeros(model.n_lags, dtype=torch.bool)
        )


def test_a_band_past_the_last_candidate_lag_refuses(matched) -> None:
    """It would be reported under a name that overstates what was removed."""
    model, _streams, _outputs = matched
    with pytest.raises(ValueError, match="past the model's last candidate lag"):
        controls.band_lag_mask(model.n_lags, 0, model.n_lags)


def test_a_band_is_inclusive_at_both_ends(matched) -> None:
    """The configuration states bands as inclusive pairs, so a half-open reading would silently
    remove one lag fewer than the name says."""
    model, _streams, _outputs = matched
    mask = controls.band_lag_mask(model.n_lags, 1, 3)
    assert mask.tolist() == [False, True, True, True] + [False] * (model.n_lags - 4)


# =============================================================================
# Source-value replacement
# =============================================================================
@pytest.mark.parametrize("mode", controls.REPLACEMENT_MODES)
def test_a_replacement_leaves_the_clock_and_the_availability_announcement_untouched(
    matched, mode
) -> None:
    """The arm changes what the source said and not when it arrived.

    Both halves are asserted rather than argued. The metadata clock is a function of stored
    position, so a bitwise-equal prior is the evidence that no source value reached it; and the
    per-anchor per-lag validity is the availability announcement, which must be the matched one
    because the replacement preserves the stream's shape and its finiteness.
    """
    model, streams, outputs = matched
    y_st, y_ph, u_stream = streams
    substituted = controls.replaced_source_stream(u_stream, mode)

    with torch.no_grad():
        replaced = model(y_st, y_ph, substituted, anchor_phase=0, anchor_stride=1)

    assert torch.equal(replaced["mu_prior"], outputs["mu_prior"])
    assert torch.equal(replaced["logvar_prior"], outputs["logvar_prior"])
    assert torch.equal(replaced["lag_valid"], outputs["lag_valid"])
    assert bool(torch.isfinite(substituted).all())


def test_a_replacement_moves_the_full_branch_with_the_selectors_still_enabled(matched) -> None:
    """That is what makes it a different question from suppression: the head still runs.

    A valid standardized zero is an observation rather than an absence, so this arm's full branch
    is not required to equal the prior -- and no code path here forces it to.
    """
    model, streams, outputs = matched
    y_st, y_ph, u_stream = streams
    substituted = controls.replaced_source_stream(u_stream, "zeros")

    with torch.no_grad():
        replaced = model(y_st, y_ph, substituted, anchor_phase=0, anchor_stride=1)

    assert float((replaced["mu_post"] - outputs["mu_post"]).abs().max()) > 0.0
    assert float((replaced["mu_post"] - replaced["mu_prior"]).abs().max()) > 0.0


def test_the_constant_arm_removes_temporal_variation_and_keeps_the_level(matched) -> None:
    """Which is the arm that separates temporal source content from a recording's own level."""
    _model, streams, _outputs = matched
    _y_st, _y_ph, u_stream = streams

    constant = controls.replaced_source_stream(u_stream, "constant")

    # Identical at every stored step, per sample and per channel...
    assert torch.allclose(constant, constant[:, :1].expand_as(constant))
    # ... and on the recording's own level rather than an arbitrary one.
    assert torch.allclose(constant[:, 0], u_stream.mean(dim=1), atol=1e-6)


def test_mask_only_and_zeros_are_one_intervention_on_the_identity_encoder(matched) -> None:
    """By construction rather than by coincidence, which is why the two names both exist.

    The encoding is the coefficient beside its mask, so a zeroed stream leaves exactly the mask.
    Asserting the identity here is what stops a later reader from believing the run measured two
    different things -- and what makes the refusal on the lifted arm read as a real distinction
    rather than as caution.
    """
    _model, streams, _outputs = matched
    _y_st, _y_ph, u_stream = streams
    assert torch.equal(
        controls.replaced_source_stream(u_stream, "mask_only"),
        controls.replaced_source_stream(u_stream, "zeros"),
    )


def test_mask_only_refuses_on_a_lifted_encoder_rather_than_reporting_the_zeros_arm() -> None:
    """The lift of a zero is a learned constant, so the identity above stops holding there."""
    streams = tiny_streams()
    with pytest.raises(ValueError, match="mask_only"):
        controls.replaced_source_stream(streams[2], "mask_only", scalar_lift=True)


def test_an_unknown_replacement_mode_names_the_arms_that_exist() -> None:
    """A misspelt mode would otherwise fall through to whichever branch happened to be last."""
    streams = tiny_streams()
    with pytest.raises(ValueError, match="unknown replacement mode"):
        controls.replaced_source_stream(streams[2], "zeroes")


# =============================================================================
# Cross-recording pairing
# =============================================================================
def test_the_pairing_is_cross_recording_and_preserves_within_source_time_order() -> None:
    """Both halves of the control, and the second is the one an index-level shuffle would break.

    Permuting rows moves whole recordings and leaves each one's stored axis intact. Shuffling
    individual source steps would destroy the autocorrelation that makes a lag window mean
    anything, and would test a claim nobody made.
    """
    recordings = ["a", "a", "b", "b", "c", "c"]
    generator = torch.Generator().manual_seed(20260909)

    index = controls.cross_recording_index(recordings, generator=generator)

    assert controls.same_recording_pairs(recordings, index) == 0
    stream = torch.randn(len(recordings), 12, 3)
    permuted = stream[index]
    for position, partner in enumerate(index.tolist()):
        assert torch.equal(permuted[position], stream[partner])


def test_a_batch_one_recording_dominates_refuses_rather_than_pairing_it_with_itself() -> None:
    """A control that has silently stopped being a control looks exactly like one that works."""
    recordings = ["a", "a", "a", "b"]
    assert not controls.groups_can_derange(recordings)
    with pytest.raises(controls.NoCrossGroupPartner):
        controls.cross_recording_index(recordings)


# =============================================================================
# The selectors-off arm
# =============================================================================
def test_the_silenced_arm_reproduces_the_prior_and_measures_nothing_else(matched) -> None:
    """It verifies the equality invariant, and the module docstring says so rather than letting a
    reader take its zero margin for a finding."""
    model, streams, outputs = matched
    y_st, y_ph, u_stream = streams
    silent = torch.zeros(
        u_stream.shape[0], outputs["mu_prior"].shape[1], model.n_lags
    )

    with torch.no_grad():
        silenced = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, selector=silent)

    assert torch.equal(silenced["mu_post"], silenced["mu_prior"])
    assert torch.equal(silenced["logvar_post"], silenced["logvar_prior"])
    assert float(silenced["kld_per_anchor"].abs().max()) == 0.0
    assert "verifies the equality invariant" in controls.__doc__


def test_the_qualification_text_states_all_three_limits() -> None:
    """It travels into the written artifact, so it is pinned where it is defined."""
    text = controls.SUPPRESSION_QUALIFICATION
    assert "which stored source time" in text
    assert "fitted computation" in text
    assert "physiological delay" in text
