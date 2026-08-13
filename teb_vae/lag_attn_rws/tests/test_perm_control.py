r"""The validation-only permutation control: source-free branches untouched, no rank alone.

The control's scientific job is the acceptance ordering
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$, and the property that makes
that ordering readable is pinned first: a derangement of the source must leave every
target-only quantity -- the prior, the base forecast, the base score -- bitwise unchanged, or
the "no source" reference has quietly moved.

**Every KL assertion perturbs the posterior first.** At init the posterior *is* the prior, so a
deranged source moves nothing and every shuffled readout is 0 for reasons that have nothing to
do with being correct.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import TINY_KWARGS, make_stub_batch

_BATCH = 4


def _model(perturb_posterior=None, **overrides) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(TINY_KWARGS, **overrides)).eval()
    if perturb_posterior is not None:
        perturb_posterior(model)
    return model


def _forward(model, batch):
    torch.manual_seed(0)
    with torch.no_grad():
        return model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))


# --------------------------------------------------------------------------------------
# The rebuild itself
# --------------------------------------------------------------------------------------
def test_the_base_branch_and_prior_are_bitwise_identical_under_permutation(perturb_posterior):
    """The source-free reference must not move -- and not only by object identity: re-scoring
    the permuted dict must reproduce the base score exactly."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    for key in ("mu_prior", "logvar_prior", "mu_base", "logvar_base", "target_state"):
        assert torch.equal(permuted[key], out[key]), f"{key} moved under permutation"

    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]
    assert torch.equal(true["nll_base_block"], shuffled["nll_base_block"])


def test_the_source_driven_tensors_are_genuinely_rebuilt(perturb_posterior):
    """The mirror image, so the test above cannot pass on a control that rebuilds nothing."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    for key in ("mu_post", "z_post", "mu_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_scoring_the_permuted_dict_yields_the_shuffled_kl(perturb_posterior):
    """``compute_loss`` recomputes the KL from the distribution parameters, so the permuted
    posterior yields a genuinely different (and nonzero) KL against the same prior."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )
    true = model.compute_loss(out, batch.fhr, weight=batch.weight)["metrics"]
    shuffled = model.compute_loss(permuted, batch.fhr, weight=batch.weight)["metrics"]

    assert float(shuffled["source_conditioned_kl_raw"]) > 0.0
    assert float(shuffled["source_conditioned_kl_raw"]) != pytest.approx(
        float(true["source_conditioned_kl_raw"]), rel=1e-6
    )


def test_the_control_rebuilds_the_query_under_query_uses_logvar(perturb_posterior):
    """``query_uses_logvar`` sizes ``query_proj`` at ``2*d_z`` and the main forward feeds it
    ``[mu^p || logvar^p]``. The control must rebuild the query the same way; a ``mu^p``-only query
    is ``d_z``-wide and would raise a shape mismatch in the projection for that whole sweep arm."""
    model = _model(perturb_posterior, query_uses_logvar=True)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    # It ran without a width mismatch, and the source-driven tensors were genuinely rebuilt.
    for key in ("mu_post", "z_post", "mu_full"):
        assert not torch.equal(permuted[key], out[key]), f"{key} was not rebuilt"


def test_the_control_uses_a_real_derangement(perturb_posterior):
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))

    permuted = controls.perm_forward_outputs(model, out)

    perm = permuted["perm_index"]
    assert not bool((perm == torch.arange(_BATCH)).any()), "the derangement has a fixed point"


def test_a_degenerate_batch_cannot_be_deranged(perturb_posterior):
    """B < 2 has no derangement; the *task's* schedule is what must keep it away from here."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(1))

    with pytest.raises(ValueError, match="batch_size >= 2"):
        controls.perm_forward_outputs(model, out)


# --------------------------------------------------------------------------------------
# The anchor set and the lag mask
# --------------------------------------------------------------------------------------
def test_the_permuted_decode_lands_on_the_anchors_it_is_given(perturb_posterior):
    """The gathered decode, against the dense one it selects from.

    The control is scored against the *matched* forward's target and mask, so it must decode the
    matched forward's anchors. Under a tiled anchor set a decode of the contiguous prefix would
    be the wrong shape entirely -- and because the control is validation-only, that failure would
    first appear after a full training epoch.
    """
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))
    perm = controls.make_derangement(_BATCH)
    anchors = torch.tensor([[2, 6, 9]] * _BATCH)

    # Seeded per call: the permuted latent is drawn with a fresh epsilon by design, so without
    # this the two decodes would differ by the draw rather than by the anchor set.
    torch.manual_seed(0)
    dense = controls.perm_forward_outputs(model, out, perm_index=perm)
    torch.manual_seed(0)
    gathered = controls.perm_forward_outputs(model, out, perm_index=perm, anchors=anchors)

    assert gathered["mu_full"].shape[:2] == (_BATCH, anchors.shape[1])
    for slot, anchor in enumerate(anchors[0].tolist()):
        # Same z, same decoder, same seed: the gathered rows are the dense rows they name.
        assert torch.allclose(
            gathered["mu_full"][:, slot], dense["mu_full"][:, anchor], atol=1e-6
        )


def test_the_full_prefix_supplied_explicitly_is_the_dense_decode(perturb_posterior):
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))
    perm = controls.make_derangement(_BATCH)
    prefix = torch.arange(model.geometry.t_valid).expand(_BATCH, -1).contiguous()

    torch.manual_seed(0)
    dense = controls.perm_forward_outputs(model, out, perm_index=perm)
    torch.manual_seed(0)
    explicit = controls.perm_forward_outputs(model, out, perm_index=perm, anchors=prefix)

    assert torch.equal(explicit["mu_full"], dense["mu_full"])


def test_the_anchor_keys_travel_through_untouched_and_are_not_recomputed(perturb_posterior):
    """They describe the *pairing's geometry*, not the pairing, so a derangement cannot move
    them -- and a reader must not be told they were rebuilt."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))
    anchors = torch.tensor([[2, 6, 9]] * _BATCH)
    valid = torch.ones_like(anchors, dtype=torch.bool)
    out = dict(out, anchor_index=anchors, anchor_valid=valid)

    permuted = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH), anchors=anchors
    )

    assert "anchor_index" not in controls.RECOMPUTED_KEYS
    assert "anchor_valid" not in controls.RECOMPUTED_KEYS
    assert permuted["anchor_index"] is out["anchor_index"]
    assert permuted["anchor_valid"] is out["anchor_valid"]


def test_the_rebuilt_attention_runs_under_the_models_own_lag_mask(perturb_posterior):
    """Not ``lag_attn``'s default. A model that restricts which lags it may read would otherwise
    have that restriction bypassed by the control alone, and the only symptom would be a shuffled
    readout computed against more source history than the matched one."""
    model = _model(perturb_posterior)
    out = _forward(model, make_stub_batch(_BATCH))
    seen = []

    def _floored(seq_len, device=None):
        mask = model.lag_attn.build_lag_mask(seq_len, device=device)
        seen.append(mask)
        steps = torch.arange(seq_len, device=device)[:, None]
        lags = torch.arange(model.lag_attn.L, device=device)[None, :]
        return mask & (steps - lags >= 4)  # a floor the default mask does not have

    unfloored = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )
    model.build_lag_mask = _floored
    floored = controls.perm_forward_outputs(
        model, out, perm_index=controls.make_derangement(_BATCH)
    )

    assert seen, "the control never asked the model for its lag mask"
    assert not torch.equal(floored["attn_weights"], unfloored["attn_weights"])
    # And specifically: the floor bit, on the lags it excludes.
    assert float(floored["attn_weights"][:, 5, :, 2:].abs().max()) == 0.0


# --------------------------------------------------------------------------------------
# The source-null arm
#
# A second control, and the reason it exists is that the first cannot see one specific failure:
# the availability announcement is identical in every row of the batch, so no permutation of rows
# removes it. What is pinned here is the arm's *mechanics* -- what it rebuilds, what it reuses,
# what it costs and what it must not touch. What it measures is the causal package's to assert,
# because only there does a model carry an availability pattern worth flooring.
# --------------------------------------------------------------------------------------
def _null(model, out, u_stream):
    return controls.source_null_forward_outputs(model, out, u_stream)


def _u_stream(batch):
    return torch.cat([batch.up_st, batch.up_ph], -1)


def test_the_null_arm_replaces_exactly_the_three_keys_it_names(perturb_posterior):
    """Three, not the permutation arm's seven. The absences are the design: this is a KL readout,
    so it draws no latent sample and runs no decoder -- and the three keys it therefore leaves
    behind still hold the *matched* forward's values, which is what the named constant warns about
    and what a reader must not take for the null's."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    nulled = _null(model, out, _u_stream(batch))

    assert set(controls.SOURCE_NULL_KEYS) == {"mu_post", "logvar_post", "attn_weights"}
    changed = {key for key in out if not torch.equal(nulled[key], out[key])}
    assert changed <= set(controls.SOURCE_NULL_KEYS)
    assert changed, "the null moved nothing at all"
    for key in ("z_post", "mu_full", "logvar_full"):
        assert nulled[key] is out[key]


def test_the_source_free_branches_are_the_same_objects_as_the_matched_forwards(
    perturb_posterior,
):
    """As under permutation, and for the same reason: the prior, both encoder states and the base
    forecast cannot depend on the source, so the arm must reuse them rather than recompute
    something that ought to be equal to them."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    nulled = _null(model, out, _u_stream(batch))

    for key in ("mu_prior", "logvar_prior", "raw_logvar_prior", "mu_base", "logvar_base",
                "target_state", "source_state"):
        assert nulled[key] is out[key], key


def test_the_null_encode_is_one_row_broadcast_and_is_bitwise_the_full_batch_one(
    perturb_posterior,
):
    r"""With $x \equiv 0$ the adapter's output is a function of the availability pattern alone, so
    the source state is identical in every batch element and is encoded once. Bitwise, not
    approximately -- nothing in the source pathway couples batch elements, so a full-batch encode
    of zeros must reproduce it exactly, and any difference would mean the saving changed the
    answer."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)
    u_stream = _u_stream(batch)

    nulled = _null(model, out, u_stream)

    zeros = torch.zeros_like(u_stream)
    gated = zeros if model.source_gate is None else model.source_gate(zeros)
    with torch.no_grad():
        full_batch = model.source_encoder(model.source_adapter(gated))
        _, alpha, attended = model.lag_attn(
            model.query_proj(out["mu_prior"]),
            full_batch,
            model.build_lag_mask(full_batch.shape[1], full_batch.device),
        )
        mu_post, logvar_post = model.posterior_head(
            out["target_state"], attended, out["mu_prior"], out["raw_logvar_prior"]
        )

    assert torch.equal(full_batch[0], full_batch[-1]), "the encode is not row-invariant"
    assert torch.equal(nulled["mu_post"], mu_post)
    assert torch.equal(nulled["logvar_post"], logvar_post)
    assert torch.equal(nulled["attn_weights"], alpha)


def test_the_null_arm_draws_no_random_number(perturb_posterior):
    """A ``torch.randn_like`` here would shift the reparameterisation stream for every subsequent
    step of the run -- a readout changing the thing it reports on, and the one failure mode that
    would show up as every bitwise comparison in the family drifting rather than as anything
    naming this function."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)

    before = torch.random.get_rng_state()
    _null(model, out, _u_stream(batch))

    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_null_arm_ignores_the_stream_it_is_handed(perturb_posterior):
    """It reads shape, dtype and device and nothing else -- which is what makes it a floor rather
    than a second reading of the source. Also the property the permutation arm cannot have."""
    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)
    u_stream = _u_stream(batch)

    first = _null(model, out, u_stream)
    second = _null(model, out, torch.randn_like(u_stream))

    for key in controls.SOURCE_NULL_KEYS:
        assert torch.equal(first[key], second[key]), key


def test_the_null_kld_is_reduced_over_the_kl_masks_own_support(perturb_posterior):
    """The same support, the same denominator and the same summation as
    ``source_conditioned_kl_raw`` -- which is the only reason subtracting one from the other is a
    difference of one quantity rather than of two conventions."""
    from teb_vae.lag_attn_rws.nets.losses import kld_tensor, masked_source_kl
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

    model = _model(perturb_posterior)
    batch = make_stub_batch(_BATCH)
    out = _forward(model, batch)
    u_stream = _u_stream(batch)

    reported = controls.source_null_kld(model, out, u_stream, batch.weight)

    nulled = _null(model, out, u_stream)
    forecast, _coverage = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
    expected = masked_source_kl(
        kld_tensor(
            mu_prior=out["mu_prior"],
            logvar_prior=out["logvar_prior"],
            mu_post=nulled["mu_post"],
            logvar_post=nulled["logvar_post"],
        ),
        kl_mask(forecast, model.geometry),
    )["source_conditioned_kl_raw"]

    assert torch.equal(reported, expected)
    assert float(reported) >= 0.0


# --------------------------------------------------------------------------------------
# The task wiring
# --------------------------------------------------------------------------------------
_CONTROL_KEYS = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}


def test_the_control_never_runs_on_a_training_batch(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert _CONTROL_KEYS.isdisjoint(metrics)


def test_validation_runs_it_and_it_contributes_nothing_to_the_loss(
    task, stub_batch, perturb_posterior
):
    """The returned validation loss must be exactly the perm-free objective: a control that
    leaked into ``val/total_loss`` would steer the ``ModelCheckpoint`` selection."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert _CONTROL_KEYS <= set(metrics)
    assert float(metrics["kld_shuffled"]) > 0.0
    assert float(loss) == pytest.approx(float(metrics["main_loss"]), rel=1e-6)


def test_shuffle_penalty_is_the_shuffled_minus_matched_score(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    expected = float(metrics["nll_shuffled_block"]) - float(metrics["nll_full_block"])
    assert float(metrics["shuffle_penalty"]) == pytest.approx(expected, rel=1e-5)


def test_the_shuffled_readouts_build_no_autograd_graph(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    for key in _CONTROL_KEYS:
        assert not metrics[key].requires_grad, key


def test_a_degenerate_validation_batch_skips_with_metrics_absent_not_zero(
    task, make_stub_batch_fn, perturb_posterior
):
    """Zeros would be wrong numbers, not placeholders: the epoch aggregate is the mean over the
    steps that reported a metric, so zero-filling skipped steps scales the shuffled readouts
    toward nothing and inverts the ordering the control exists to check."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch_fn(batch_size=1), 0, "val")

    assert _CONTROL_KEYS.isdisjoint(metrics)
    assert torch.isfinite(loss)


def test_a_false_on_any_rank_wins_the_collective_decision(
    task, stub_batch, perturb_posterior, monkeypatch
):
    """The MIN reduce: a rank whose last uneven batch is degenerate cannot be outvoted, or the
    ranks that ran the control would hang the ``sync_dist`` on its metrics. Simulated by a fake
    collective whose peer always answers 'cannot', which must silence this rank too."""
    import teb_vae.lag_attn_rws.task as task_module

    module = task()
    perturb_posterior(module.orig_model)
    reduce_ops = []

    monkeypatch.setattr(task_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(task_module.dist, "is_initialized", lambda: True)

    def _peer_says_no(tensor, op=None):
        reduce_ops.append(op)
        tensor.copy_(torch.minimum(tensor, torch.zeros_like(tensor)))  # peer's flag is 0

    monkeypatch.setattr(task_module.dist, "all_reduce", _peer_says_no)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert _CONTROL_KEYS.isdisjoint(metrics), "a rank ran the control alone"
    assert reduce_ops == [task_module.dist.ReduceOp.MIN], (
        "the decision must be MIN-reduced exactly once: run only if every rank can"
    )


def test_a_true_from_every_rank_keeps_the_control_running(
    task, stub_batch, perturb_posterior, monkeypatch
):
    """The mirror image: the fake collective must be able to pass a decision through, or the
    test above passes on a wiring that simply never runs the control under DDP."""
    import teb_vae.lag_attn_rws.task as task_module

    module = task()
    perturb_posterior(module.orig_model)

    monkeypatch.setattr(task_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(task_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(task_module.dist, "all_reduce", lambda tensor, op=None: None)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert _CONTROL_KEYS <= set(metrics)
