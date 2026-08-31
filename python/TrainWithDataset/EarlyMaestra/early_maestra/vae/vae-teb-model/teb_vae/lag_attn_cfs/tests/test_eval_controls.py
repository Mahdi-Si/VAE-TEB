r"""The two controls, and the hazard only one of them can see.

**The permutation control** must pair a target with a *stranger's* source, not a neighbour's.
$\pi(i) \neq i$ is the wrong guarantee on this loader: ``test_dataloader`` is unshuffled over eight
concatenated per-subgroup shards and one delivery contributes tens of consecutive 20-minute
segments, so a batch routinely holds several segments of the same recording -- and Sattolo's
algorithm will happily pair one with another. Those two share a mother, a sensor placement and a
labour; the "shuffled" forecast built from one is not the out-of-recording control the ordering
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ is read against.

**The source-null control** answers a question no permutation can. The source availability pattern
$m^u_{t,c}$ is a deterministic function of $t$, **identical in every row of a batch**, and it
enters $q(z \mid Y, U)$ but not $p(z \mid Y)$ -- so it can push the posterior off the prior and
inflate the coupling readout with no source information in it at all. Deranging rows cannot remove
something every row shares. The null re-encodes a **zeroed** source stream instead, and

$$\Delta_{\mathrm{clock}} = \texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null}$$

is the part of the coupling readout attributable to source *variation*. If the two are equal, the
readout is measuring a clock.

Four properties of that arm are load-bearing and each is asserted below: it draws no random
number, so it cannot move the reparameterisation stream for the rest of a multi-hour pass; it
reduces on the **same** anchor support the matched readout does, so the difference is defined; it
is exactly zero when the matched source carries no variation either; and the encode is one
broadcast row while the divergence still varies per sample -- the first is the saving, the second
is what makes the readout a measurement rather than a constant.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    batch_recordings,
    evaluate,
    evaluate_batch,
    model_inputs,
    source_null_kld_per_sample,
)
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

from .conftest import BATCH, STUB_GAP_STEP, make_stub_batch


@pytest.fixture
def trained_task(task, perturb_posterior):
    """A tiny task whose posterior has been moved off the prior.

    Load-bearing: at initialisation the delta heads are zero, so the posterior *is* the prior,
    every KL is exactly zero, and every assertion about a difference of KLs below would hold on a
    model that is completely wrong.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    return module


class _OneBatchLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


def _labelled(batch_size: int, guids, seed: int = 0):
    """A stub batch carrying the given recording identifiers."""
    batch = make_stub_batch(batch=batch_size, seed=seed)
    batch.guid = list(guids)
    return batch


def _dense_forward(module, batch):
    """Run the dense forward the evaluation runs, and return everything the null arm needs.

    Rebuilt through the same three calls ``evaluate_batch`` makes rather than reaching into it,
    so a test that pins the two supports equal is comparing two derivations rather than one.
    """
    model = module.orig_model
    y_st, y_ph, u_stream, _target_features, weight = model_inputs(module, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    mask, _coverage = forecast_mask(
        weight, model.geometry, coverage_floor=model.coverage_floor,
        anchors=anchors, anchor_valid=anchor_valid,
    )
    support = kl_mask(mask, model.geometry, anchors=anchors, anchor_valid=anchor_valid)
    return outputs, u_stream, weight, support


# =================================================================================================
# The source-null arm
# =================================================================================================
def test_the_null_arm_draws_no_random_number(trained_task, stub_batch) -> None:
    """It must not shift the reparameterisation stream for every subsequent step of a run: the
    Monte Carlo estimator draws from the same process, so one extra ``randn_like`` here would make
    every score after it a different sample of the same quantity."""
    outputs, u_stream, _weight, support = _dense_forward(trained_task, stub_batch)
    before = torch.random.get_rng_state()

    source_null_kld_per_sample(trained_task.orig_model, outputs, u_stream, support)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_per_sample_column_recombines_into_the_shared_functions_scalar(
    trained_task, stub_batch
) -> None:
    """``controls.source_null_kld`` is where this arm is *defined*, and it reduces a whole batch to
    one number. The per-sample column here is the same quantity with the batch axis open, so the
    anchor-weighted recombination of it must be that number -- which is the same treatment
    ``prior_rate`` gets, and for the same reason."""
    outputs, u_stream, weight, support = _dense_forward(trained_task, stub_batch)

    per_sample = source_null_kld_per_sample(
        trained_task.orig_model, outputs, u_stream, support
    )["kld_source_null"]
    shared = controls.source_null_kld(trained_task.orig_model, outputs, u_stream, weight)

    counts = support.sum(dim=1)
    recombined = float((per_sample * counts).sum() / counts.sum())
    assert recombined == pytest.approx(float(shared), rel=1e-6)


def test_the_null_and_the_matched_readout_share_one_anchor_support(
    trained_task
) -> None:
    """The difference is only defined if the two are reduced over the same set. Asserted on a batch
    whose support is *non-trivial* -- the stub carries a gap inside the trained anchor range, and
    the coverage floor drops the anchors whose window touches it -- because on a fully valid batch
    every support in the pipeline coincides and the assertion would be vacuous."""
    batch = make_stub_batch(seed=2)
    outputs, u_stream, _weight, support = _dense_forward(trained_task, batch)
    model = trained_task.orig_model

    readout = evaluate_batch(trained_task, batch, num_samples=1)

    counts = support.sum(dim=1)
    assert bool((counts > 0).all()), "every sample must score something or this proves nothing"
    assert float(counts.max()) < float(model.geometry.t_valid - model.warmup_period), (
        "the gap must actually remove anchors, otherwise every support coincides trivially"
    )
    # Both columns are the same weighted mean over the same weights, so the difference column is
    # exactly the difference of the two -- which is what "the same support" buys.
    assert torch.allclose(
        readout.columns["coupling_minus_clock"],
        readout.columns["source_conditioned_kl_raw"] - readout.columns["kld_source_null"],
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        readout.columns["kld_source_null"],
        source_null_kld_per_sample(model, outputs, u_stream, support)["kld_source_null"],
        rtol=1e-6,
    )


def test_the_null_profile_decomposes_the_null_scalar_over_the_lags(trained_task, stub_batch):
    r"""The identity that makes a per-lag clock-excess a decomposition rather than a second
    reading.

    $$\sum_\ell \widetilde K^{\mathrm{null}}_{b,\ell} = K^{\mathrm{null}}_b$$

    holds because each head's null attention sums to one over its valid lags and the latent groups
    are head-aligned, exactly as it does on the matched arm. Its consequence is the one that
    matters: since the matched profile sums to ``source_conditioned_kl_raw``, the difference of the
    two profiles sums to ``coupling_minus_clock`` -- so the clock-excess attribution is a
    decomposition of the very scalar ``clock_margin_min_nats`` gates, rather than a differently
    normalised quantity that happens to be lag-resolved.
    """
    outputs, u_stream, _weight, support = _dense_forward(trained_task, stub_batch)

    arm = source_null_kld_per_sample(trained_task.orig_model, outputs, u_stream, support)

    assert torch.allclose(
        arm["lag_profile_null"].sum(dim=-1), arm["kld_source_null"], rtol=1e-5, atol=1e-6
    )
    # The residual the sanity block reads, measured on this batch rather than assumed. Attention
    # dropout is the one mechanism that breaks it, and the model builds its attention at zero
    # dropout for exactly this reason.
    assert float(arm["lag_map_null_identity_max_abs"].max()) < 1e-4


def test_the_clock_excess_profile_sums_to_the_gated_scalar(trained_task, stub_batch):
    """The whole point of the lag-resolved null, asserted end to end through ``evaluate_batch``
    rather than through the arm alone: what a reader subtracts is the *emitted* pair of profiles,
    and a reduction that put the two on different anchor supports would leave both identities
    above intact while making their difference meaningless."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    excess = readout.lag_profile - readout.lag_profile_null
    assert torch.allclose(
        excess.sum(dim=-1),
        readout.columns["coupling_minus_clock"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_per_head_attribution_sums_over_heads_to_the_pooled_one(trained_task, stub_batch):
    r"""$\sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell} = \widetilde K_{t,\ell}$, at the segment level.

    This is the ``TEAnalysisHead`` identity with the head axis kept open, and it is what makes the
    per-head profile a refinement of the shipped decomposition rather than a second quantity
    wearing its name. It is also the check that the head-major flattening agrees with the reshape
    every consumer applies -- a transposed layout would still sum to the same total, so the sum is
    asserted per lag rather than in aggregate.
    """
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    n_lags = readout.lag_profile.shape[-1]
    per_head = readout.lag_profile_per_head
    num_heads = per_head.shape[-1] // n_lags
    assert per_head.shape[-1] == num_heads * n_lags

    stacked = per_head.reshape(per_head.shape[0], num_heads, n_lags)
    assert torch.allclose(stacked.sum(dim=1), readout.lag_profile, rtol=1e-5, atol=1e-6)


def test_a_source_that_already_carries_no_variation_leaves_no_difference(
    trained_task
) -> None:
    """The known-answer case, and the only one where the right answer is a constant. Handed a
    source stream that is already zero, the matched arm *is* the null arm, so
    ``coupling_minus_clock`` is zero to float tolerance -- and a non-zero value would mean the two
    arms differ by something other than the source they were given."""
    batch = make_stub_batch(seed=7)
    batch.up_st = torch.zeros_like(batch.up_st)
    batch.up_ph = torch.zeros_like(batch.up_ph)

    readout = evaluate_batch(trained_task, batch, num_samples=1)

    assert torch.allclose(
        readout.columns["coupling_minus_clock"],
        torch.zeros(BATCH),
        atol=1e-5,
    )
    assert float(readout.columns["kld_source_null"].min()) > 0.0, (
        "a zero KL on both sides would make the equality above vacuous"
    )


def test_the_null_encode_is_one_broadcast_row_and_the_divergence_still_varies(
    trained_task, stub_batch
) -> None:
    r"""Both halves, because they are two different claims.

    With $x \equiv 0$ the adapter's output is a function of the availability pattern alone, so it
    is identical in every batch element and the arm encodes **once**, at batch $1$, and expands --
    that is the saving, and a per-row encode would be paying for a broadcast. The divergence must
    nonetheless vary per sample, because the query is $\mu^p$ and the prior is the sample's own; a
    constant column here would be a readout that measures nothing and subtracts a constant from
    the coupling.

    The first half is asserted at a **tolerance rather than bitwise**, and the reason is worth
    recording where it would otherwise look like a weakened test: encoding the same row twice in
    one batched call does not reproduce it bit for bit, because the LSTM's and the convolutions'
    matrix products accumulate over a batch-dependent tiling. That is precisely why the arm encodes
    at batch $1$ and expands -- a view of one row, so the state it attends over is exactly one row
    rather than two that agree to rounding.
    """
    outputs, u_stream, _weight, support = _dense_forward(trained_task, stub_batch)
    model = trained_task.orig_model

    zeros = u_stream.new_zeros((u_stream.shape[0], *u_stream.shape[1:]))
    gated = zeros if model.source_gate is None else model.source_gate(zeros)
    with torch.no_grad():
        encoded = model.source_encoder(model.source_adapter(gated))
    per_sample = source_null_kld_per_sample(model, outputs, u_stream, support)[
        "kld_source_null"
    ]

    assert torch.allclose(encoded[0], encoded[1], rtol=0.0, atol=1e-5), (
        "a zeroed source must encode to one row up to batched-GEMM rounding; if it does not, the "
        "adapter is reading something other than the availability pattern and the one-encode "
        "saving is unsound"
    )
    assert float(per_sample[0]) != float(per_sample[1])


def test_the_two_columns_are_on_every_row_of_every_batch(trained_task) -> None:
    """Unlike the permutation controls the null needs no second sample -- it is a zeroed stream
    rather than a stranger's -- so a one-sample batch still carries both columns. A column that
    appeared only on derangeable batches would be missing from exactly the runs whose loader hands
    out degenerate ones."""
    readout = evaluate_batch(trained_task, make_stub_batch(batch=1), num_samples=1)

    assert readout.columns["kld_source_null"].shape == (1,)
    assert readout.columns["coupling_minus_clock"].shape == (1,)
    assert "mc_nll_shuffled_block" not in readout.columns


# =================================================================================================
# The permutation control's draw
# =================================================================================================
@pytest.mark.parametrize(
    "groups",
    [
        ["a", "a", "b", "b"],
        ["a", "b"],
        ["a", "a", "b", "c"],
        ["a"] * 4 + ["b"] * 3 + ["c"] * 1,
    ],
)
def test_a_grouped_derangement_never_pairs_within_a_group(groups) -> None:
    """The whole point, over compositions ranging from balanced to right at the feasibility
    boundary. Drawn repeatedly because a construction that is *usually* cross-group would pass a
    single draw."""
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        perm = controls.make_derangement(len(groups), generator=generator, groups=groups)

        assert torch.equal(perm.sort().values, torch.arange(len(groups))), "not a permutation"
        for position, partner in enumerate(perm.tolist()):
            assert groups[position] != groups[partner], f"{groups} paired within its own group"


@pytest.mark.parametrize(
    "groups,expected",
    [
        (["a", "b"], True),
        (["a", "a", "b", "b"], True),
        (["a", "a", "a", "b"], False),
        (["a", "a"], False),
        (["a"], False),
    ],
)
def test_feasibility_is_the_exact_half_batch_condition(groups, expected) -> None:
    r"""Hall's theorem gives $2\max_g |g| \le B$ exactly, so the predicate rejects only batches
    that genuinely have no valid pairing."""
    assert controls.groups_can_derange(groups) is expected


def test_the_forward_control_pairs_across_recordings_at_the_dense_geometry(
    trained_task, stub_batch
) -> None:
    """Threaded all the way through the rebuild, not merely available on the draw -- and at the
    anchor set the evaluation decodes at, because ``perm_forward_outputs`` gathers the permuted
    latent at ``anchor_index`` and a control that decoded a different set would be scored against
    this one's target."""
    outputs, _u_stream, _weight, _support = _dense_forward(trained_task, stub_batch)

    permuted = controls.perm_forward_outputs(
        trained_task.orig_model, outputs, groups=["a", "b"],
        anchors=outputs["anchor_index"],
    )

    assert permuted["perm_index"].tolist() == [1, 0]
    assert permuted["mu_full"].shape == outputs["mu_full"].shape


def test_the_control_decodes_the_prefix_when_the_anchor_set_is_withheld(
    trained_task, stub_batch
) -> None:
    """The negative control for the call site above, and the reason ``anchors=`` is passed there.

    ``perm_forward_outputs`` takes the anchor set as an argument and falls back to the contiguous
    prefix without it. That fallback is right for a model that decodes every anchor and wrong here,
    and it is **silent** at the call site: ``evaluate_batch`` reads only the permuted posterior's
    two distribution parameters, which are $(B, T, d_z)$ either way. So the wrong shape would
    surface nowhere until something read the control's forecast.
    """
    outputs, _u_stream, _weight, _support = _dense_forward(trained_task, stub_batch)

    withheld = controls.perm_forward_outputs(
        trained_task.orig_model, outputs, groups=["a", "b"]
    )

    assert withheld["mu_full"].shape[1] == trained_task.orig_model.geometry.t_valid
    assert withheld["mu_full"].shape[1] != outputs["mu_full"].shape[1]


# =================================================================================================
# The evaluation loop's accounting
# =================================================================================================
def test_a_batch_spanning_two_recordings_pairs_across_them(trained_task) -> None:
    readout = evaluate_batch(trained_task, _labelled(BATCH, ["a", "b"]), num_samples=1)

    assert readout.n_control_pairs == BATCH
    assert readout.n_same_recording_pairs == 0


def test_a_single_recording_batch_is_excluded_and_counted(trained_task) -> None:
    """Excluded whole rather than scored without its control: a partially scored batch produces
    a different column set, and averaging an inconsistent set together is how a control stops
    being reported with nothing failing. Counted because the batches one recording fills on its
    own are the *longest* recordings' -- dropping them silently removes a non-random slice."""
    loader = _OneBatchLoader(
        [_labelled(2, ["solo", "solo"]), _labelled(2, ["a", "b"], seed=1)]
    )

    results = evaluate(trained_task, loader, num_samples=1)

    assert results["n_batches"] == 1
    assert results["controls"]["n_batches_excluded_no_cross_recording_partner"] == 1
    assert results["controls"]["n_samples_excluded_no_cross_recording_partner"] == 2
    assert results["controls"]["same_recording_pairing_rate"] == 0.0


def test_the_control_statistics_are_reported_even_when_nothing_was_excluded(
    trained_task
) -> None:
    """Present at zero, not absent: a reader cannot tell an unreported exclusion count from a
    zero one, and the pairing rate is the only evidence the control is still a control."""
    loader = _OneBatchLoader([_labelled(4, ["a", "a", "b", "b"])])

    control_block = evaluate(trained_task, loader, num_samples=1)["controls"]

    assert control_block["n_batches_excluded_no_cross_recording_partner"] == 0
    assert control_block["n_control_pairs"] == 4
    assert control_block["n_same_recording_pairs"] == 0
    assert control_block["same_recording_pairing_rate"] == 0.0


def test_the_recording_accessor_distinguishes_absent_from_uniform() -> None:
    """An absent ``guid`` means the grouping is *unknown*, which is not the same as every sample
    belonging to one recording: the first calls for the ungrouped derangement, the second for
    excluding the batch entirely."""
    unlabelled = make_stub_batch()
    unlabelled.guid = None

    assert batch_recordings(unlabelled, BATCH) is None
    assert batch_recordings(_labelled(2, ["a", "a"]), 2) == ["a", "a"]


def test_the_gap_the_stub_batch_carries_is_inside_the_scored_range(trained_task) -> None:
    """Non-vacuity for every support assertion in this file: a gap outside $[F, T_{\\rm valid})$
    would leave every mask in the pipeline fully valid and each of them would agree trivially."""
    model = trained_task.orig_model

    assert model.warmup_period <= STUB_GAP_STEP < model.geometry.t_valid
