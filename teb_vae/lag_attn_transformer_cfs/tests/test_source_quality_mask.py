r"""The UA quality mask: ``source_quality_mask`` and the forward's ``source_valid``.

A bad UA step must contribute nothing to the FHR latent. UA reaches the posterior at query $t$
through the lag keys at $t - \ell$, so the mask acts twice: it removes the bad steps from the keys,

$$K_{b,t,\ell} = \mathrm{lag\_mask}_{t,\ell} \wedge m_{b,\,t-\ell},$$

and it closes the posterior gate where no key is left, $g^{\mathrm{eff}}_t = g_t\,o_t$ with
$o_t = \exists \ell: K_{t\ell}$. Removing the keys alone is not enough: the attended summary is then
zero, but LayerNorm of zero is the norm's bias, so the gate would stay open and $q \ne p$.

What is asserted here:

* the off state is bitwise the model before the keyword, and the flag adds no parameter;
* the flag and the mask must travel together, and a mask of the wrong shape is refused;
* an all-valid mask is bitwise the unmasked forward, under both normalisers and with or without
  the learned gate;
* a fully masked sample has $q = p$ exactly and a zero KL, and leaves the other samples bitwise;
* a removed step gets zero attention weight at every query that could read it;
* a K/V body with a reach $R$ widens a bad step over the $R - 1$ keys after it, an unbounded body
  over the rest of the sequence, and the per-step adapter not at all;
* the controls re-attend under the same mask, and the permutation arm carries the donor's;
* the objective averages the gate over the open anchors and reports the closed share, charges
  the gate over the KL's own anchor count, and reports the KL over the open anchors;
* the task appends the mask as the sixth input and returns the contraction weight to its floor
  where an anchor is closed;
* the task zeroes the target steps the FHR step mask marks bad (``target_quality_mask``),
  refuses ``event_weight_ua_valid`` without ``up_valid``, and ANDs the contraction spans into the
  source mask (``source_event_mask``);
* the per-anchor readout carries ``source_open`` and ``event_gate``, and the quality labels
  ``ua_key_open`` and ``target_bad_steps`` whatever the model applies;
* the pre-flight refuses the flag without the loader rule, and the reverse; the three quality
  switches without the fields they read; and every shipped config passes both checks.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.eval.metrics import evaluate_batch
from teb_vae.lag_attn_cfs.trainer import _check_quality_switch_pairing, _check_source_quality_pairing
from teb_vae.lag_attn_rws.event_weight import contraction_source_mask
from teb_vae.lag_attn_rws.nets.controls import (
    perm_forward_outputs,
    source_null_forward_outputs,
    source_null_kld,
)

from .conftest import (
    CAUSAL_C_Y,
    CAUSAL_ST_WIDTH,
    TINY_TARGET_KEEP_INDEX,
    build,
    make_stub_batch,
    make_streams,
    make_task,
    tiny_warmup_kwargs,
)

#: Three samples, so the permutation control has a non-trivial derangement.
BATCH = 3
#: The step removed from sample 1 in the partial-mask tests.
REMOVED_STEP = 10


def _pair(perturb, **overrides):
    """The same weights built without and with the flag, both perturbed off their zero-init."""
    kwargs = tiny_warmup_kwargs(**overrides)
    off = build(kwargs).eval()
    on = build(dict(kwargs, source_quality_mask=True)).eval()
    perturb(off)
    perturb(on)
    return kwargs, off, on


def _masks(u_stream):
    """All valid; and sample 0 fully masked with sample 1 missing one step."""
    ones = torch.ones(u_stream.shape, dtype=torch.bool)
    partial = ones.clone()
    partial[0] = False
    partial[1, REMOVED_STEP] = False
    return ones, partial


# ---------------------------------------------------------------------------------------
# Construction and refusals
# ---------------------------------------------------------------------------------------
def test_the_flag_adds_no_parameter_and_no_buffer(tiny_warmup):
    without = build(tiny_warmup)
    explicit = build(dict(tiny_warmup, source_quality_mask=True))
    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))


def test_the_flag_and_the_mask_travel_together(tiny_warmup):
    off = build(tiny_warmup).eval()
    on = build(dict(tiny_warmup, source_quality_mask=True)).eval()
    y_st, y_ph, u = make_streams(tiny_warmup, batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    with torch.no_grad():
        with pytest.raises(ValueError, match="needs source_valid"):
            on(y_st, y_ph, u)
        with pytest.raises(ValueError, match="built without source_quality_mask"):
            off(y_st, y_ph, u, None, None, valid)
        with pytest.raises(ValueError, match="has shape"):
            on(y_st, y_ph, u, None, None, valid[..., :-1])


# ---------------------------------------------------------------------------------------
# The forward
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_entmax", [True, False])
@pytest.mark.parametrize("gate", [True, False])
def test_an_all_valid_mask_is_bitwise_the_unmasked_forward(perturb_posterior, gate, use_entmax):
    kwargs, off, on = _pair(perturb_posterior, posterior_event_gate=gate, use_entmax=use_entmax)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    ones, _ = _masks(u)
    with torch.no_grad():
        plain, masked = off(y_st, y_ph, u), on(y_st, y_ph, u, None, None, ones)
    for key in ("mu_post", "logvar_post", "attn_weights", "kld_per_t"):
        assert torch.equal(plain[key], masked[key]), key
    assert bool(masked["source_open"].all())
    assert "source_key_mask" not in plain and "source_open" not in plain


@pytest.mark.parametrize("use_entmax", [True, False])
@pytest.mark.parametrize("gate", [True, False])
def test_a_fully_masked_sample_is_the_prior_exactly(perturb_posterior, gate, use_entmax):
    kwargs, off, on = _pair(perturb_posterior, posterior_event_gate=gate, use_entmax=use_entmax)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    _, partial = _masks(u)
    with torch.no_grad():
        plain, masked = off(y_st, y_ph, u), on(y_st, y_ph, u, None, None, partial)

    assert not bool(masked["source_open"][0].any())
    assert torch.equal(masked["mu_post"][0], masked["mu_prior"][0])
    assert torch.allclose(masked["logvar_post"][0], masked["logvar_prior"][0], atol=1e-6)
    assert float(masked["kld_per_t"][0].abs().max()) < 1e-6
    # The other samples do not see sample 0's mask.
    assert torch.equal(masked["mu_post"][2], plain["mu_post"][2])
    if gate:
        assert float(masked["event_gate"][0].abs().max()) == 0.0


def test_a_removed_step_gets_no_weight_at_any_query_that_could_read_it(perturb_posterior):
    kwargs, _, on = _pair(perturb_posterior, posterior_event_gate=True)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    _, partial = _masks(u)
    with torch.no_grad():
        alpha = on(y_st, y_ph, u, None, None, partial)["attn_weights"]  # (B, T, H, L)
    lags = alpha.shape[-1]
    for t in range(REMOVED_STEP, min(alpha.shape[1], REMOVED_STEP + lags)):
        assert float(alpha[1, t, :, t - REMOVED_STEP].abs().max()) == 0.0


def test_one_bad_channel_makes_its_step_bad(perturb_posterior):
    """A step is usable only if every source channel the model reads is."""
    kwargs, _, on = _pair(perturb_posterior)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    per_channel = torch.ones(u.shape, dtype=torch.bool)
    per_channel[1, REMOVED_STEP, CAUSAL_ST_WIDTH] = False
    per_step = per_channel.all(dim=-1)
    with torch.no_grad():
        a = on(y_st, y_ph, u, None, None, per_channel)
        b = on(y_st, y_ph, u, None, None, per_step)
    assert torch.equal(a["source_key_mask"], b["source_key_mask"])
    assert torch.equal(a["mu_post"], b["mu_post"])
    assert not bool(a["source_key_mask"][1, REMOVED_STEP, 0])


@pytest.mark.parametrize("use_entmax", [True, False])
def test_gradients_stay_finite_through_a_fully_masked_sample(use_entmax):
    kwargs = tiny_warmup_kwargs(posterior_event_gate=True, use_entmax=use_entmax)
    model = build(dict(kwargs, source_quality_mask=True)).train()
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    _, partial = _masks(u)
    out = model(y_st, y_ph, u, None, None, partial)
    (out["kld_per_t"].sum() + out["mu_full"].sum()).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def _widened(valid, reach):
    r"""The reference widening: $m'_s = \bigwedge_{j=0}^{R-1} m_{s-j}$, with $m_{s<0}$ usable."""
    out = valid.clone()
    for shift in range(1, reach):
        out[:, shift:] &= valid[:, :-shift]
    return out


@pytest.mark.parametrize("arm", ["adapter", "conv_stem", "encoder"])
def test_the_key_mask_widens_by_the_kv_bodys_reach(arm):
    r"""A key reads $R$ steps of source, so a bad step closes itself and the $R - 1$ keys after it.
    The adapter reads one step: $R = 1$, and the mask is the per-step one."""
    kwargs = tiny_warmup_kwargs(lag_kv_source=arm)
    model = build(dict(kwargs, source_quality_mask=True)).eval()
    steps = int(kwargs["sequence_length"])
    body = model.source_kv_body()
    reach = model._source_kv_reach(steps)
    if arm == "adapter":
        assert body is None and reach == 1
    else:
        assert reach == min(int(body.receptive_field), steps) and reach > 1

    valid = torch.ones(BATCH, steps, dtype=torch.bool)
    valid[1, REMOVED_STEP] = False
    lag_mask = torch.ones(steps, int(model.max_lag) + 1, dtype=torch.bool)
    keys = model.source_key_mask(valid, lag_mask)

    # Lag 0 of query t is step t itself, so the lag-0 column is the widened per-step mask.
    assert torch.equal(keys[..., 0], _widened(valid, reach))
    last_bad = min(REMOVED_STEP + reach - 1, steps - 1)
    assert not bool(keys[1, last_bad, 0])
    if last_bad + 1 < steps:
        assert bool(keys[1, last_bad + 1, 0])
    assert bool(keys[1, :REMOVED_STEP, 0].all()) and bool(keys[0, :, 0].all())


def test_an_unbounded_body_closes_every_key_after_a_bad_step():
    r"""A full-prefix encoder (``source_attention_window`` ``None``) has no reach bound: every key
    from the bad step on reads it, so $R = T$."""
    kwargs = tiny_warmup_kwargs(lag_kv_source="encoder", source_attention_window=None)
    model = build(dict(kwargs, source_quality_mask=True)).eval()
    steps = int(kwargs["sequence_length"])
    assert model.source_kv_body().receptive_field is None
    assert model._source_kv_reach(steps) == steps

    valid = torch.ones(BATCH, steps, dtype=torch.bool)
    valid[1, REMOVED_STEP] = False
    keys = model.source_key_mask(valid, torch.ones(steps, int(model.max_lag) + 1, dtype=torch.bool))
    assert not bool(keys[1, REMOVED_STEP:, 0].any())
    assert bool(keys[1, :REMOVED_STEP, 0].all())


# ---------------------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------------------
def test_the_permutation_arm_attends_under_the_donors_mask(perturb_posterior):
    kwargs, _, on = _pair(perturb_posterior, posterior_event_gate=True)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    _, partial = _masks(u)
    with torch.no_grad():
        matched = on(y_st, y_ph, u, None, None, partial)
        shuffled = perm_forward_outputs(on, matched, perm_index=torch.tensor([1, 2, 0]))
    # Sample 2 received sample 0's source, which is fully masked: its posterior is its prior.
    assert torch.equal(shuffled["source_key_mask"][2], matched["source_key_mask"][0])
    assert torch.equal(shuffled["mu_post"][2], matched["mu_prior"][2])
    assert torch.equal(shuffled["source_open"], shuffled["source_key_mask"].any(dim=-1))


def test_the_source_null_arm_keeps_the_recordings_mask(perturb_posterior):
    kwargs, _, on = _pair(perturb_posterior, posterior_event_gate=True)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    _, partial = _masks(u)
    with torch.no_grad():
        matched = on(y_st, y_ph, u, None, None, partial)
        nulled = source_null_forward_outputs(on, matched, u)
        readout = source_null_kld(on, matched, u, torch.ones(u.shape[0], u.shape[1]))
    assert torch.equal(nulled["mu_post"][0], matched["mu_prior"][0])
    assert torch.isfinite(readout)


# ---------------------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------------------
def test_the_gate_mean_counts_only_open_anchors_and_the_closed_share_is_reported(
    perturb_posterior,
):
    kwargs, _, on = _pair(perturb_posterior, posterior_event_gate=True)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    ones, partial = _masks(u)
    weight = torch.ones(u.shape[0], u.shape[1])
    target = torch.cat([y_st, y_ph], dim=-1)
    with torch.no_grad():
        full = on.compute_loss(on(y_st, y_ph, u, None, None, ones), target, weight=weight)
        part = on.compute_loss(on(y_st, y_ph, u, None, None, partial), target, weight=weight)
    assert float(full["metrics"]["source_mask_fraction"]) == 0.0
    share = float(part["metrics"]["source_mask_fraction"])
    # Sample 0 of three is closed. The tiny `encoder` arm reads 13 steps per key, so the one bad
    # step of sample 1 also closes the few anchors whose every key reads it; far below a sample.
    assert 0.3 < share < 0.4
    # Closed anchors carry a zero gate. Averaged over every anchor, the mean would drop to about
    # (1 - share) of the unmasked mean; averaged over the open anchors only, it stays near it.
    full_mean = float(full["metrics"]["event_gate_mean"])
    part_mean = float(part["metrics"]["event_gate_mean"])
    assert abs(part_mean - full_mean) < 0.5 * share * full_mean


def test_the_gate_penalty_and_the_kl_share_one_denominator(perturb_posterior):
    """A closed anchor adds 0 to the gate's sum and to the KL's, and both divide by the KL's anchor
    count, so the mask cannot reweigh the gate against the KL. The open-anchor KL is the raw KL
    over the open share."""
    kwargs, _, on = _pair(perturb_posterior, posterior_event_gate=True)
    # Float64: the penalty is read as a difference of two total losses, and in float32 that
    # difference keeps only about 4 significant digits.
    on = on.double()
    y_st, y_ph, u = (x.double() for x in make_streams(kwargs, batch=BATCH))
    ones, partial = _masks(u)
    weight = torch.ones(u.shape[0], u.shape[1], dtype=u.dtype)
    target = torch.cat([y_st, y_ph], dim=-1)
    with torch.no_grad():
        full_out = on(y_st, y_ph, u, None, None, ones)
        part_out = on(y_st, y_ph, u, None, None, partial)
        results = {
            (name, lam): on.compute_loss(out, target, weight=weight, beta=1.0,
                                         lambda_event_gate=lam)["metrics"]
            for name, out in (("full", full_out), ("part", part_out)) for lam in (0.0, 0.5)
        }
    for name in ("full", "part"):
        charged, free = results[(name, 0.5)], results[(name, 0.0)]
        share = float(charged["source_mask_fraction"])
        penalty = float(charged["total_loss"] - free["total_loss"]) / 0.5
        assert penalty == pytest.approx(float(charged["event_gate_mean"]) * (1.0 - share),
                                        rel=1e-4, abs=1e-6)
        assert float(charged["source_conditioned_kl_open"]) * (1.0 - share) == pytest.approx(
            float(charged["source_conditioned_kl_raw"]), rel=1e-4, abs=1e-6)
    # Without a closed anchor the two KL readouts are one value.
    full = results[("full", 0.0)]
    assert float(full["source_conditioned_kl_open"]) == pytest.approx(
        float(full["source_conditioned_kl_raw"]), rel=1e-6, abs=1e-9)


def test_the_closed_share_is_zero_without_the_mask(perturb_posterior):
    kwargs, off, _ = _pair(perturb_posterior)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    with torch.no_grad():
        result = off.compute_loss(
            off(y_st, y_ph, u), torch.cat([y_st, y_ph], dim=-1), weight=torch.ones(BATCH, u.shape[1])
        )
    assert float(result["metrics"]["source_mask_fraction"]) == 0.0


# ---------------------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------------------
def _masked_batch():
    batch = make_stub_batch()
    steps = batch.up_st.shape[1]
    batch.up_st_step_valid = torch.ones(batch.up_st.shape, dtype=torch.bool)
    batch.up_ph_step_valid = torch.ones(batch.up_ph.shape, dtype=torch.bool)
    batch.up_ph_step_valid[0, : steps // 2] = False
    return batch


def test_the_task_appends_the_mask_as_the_sixth_input():
    module = make_task(dict(tiny_warmup_kwargs(), source_quality_mask=True))
    module._stage = "val"
    batch = _masked_batch()
    inputs = module._build_forward_inputs(batch)
    assert len(inputs) == 6
    expected = torch.cat([batch.up_st_step_valid, batch.up_ph_step_valid], dim=-1)
    assert torch.equal(inputs[5], expected)
    assert inputs[5].shape == inputs[2].shape


def test_the_task_refuses_a_batch_without_the_mask():
    module = make_task(dict(tiny_warmup_kwargs(), source_quality_mask=True))
    module._stage = "val"
    with pytest.raises(RuntimeError, match="up_st_step_valid"):
        module._build_forward_inputs(make_stub_batch())


def test_an_unmasked_task_keeps_five_inputs():
    module = make_task(tiny_warmup_kwargs())
    module._stage = "val"
    assert len(module._build_forward_inputs(_masked_batch())) == 5


def test_the_contraction_weight_returns_to_its_floor_at_closed_anchors():
    module = make_task(
        dict(tiny_warmup_kwargs(), source_quality_mask=True), hparams={"event_weight_floor": 0.25}
    )
    weight = torch.ones(2, 6)
    source_open = torch.tensor([[True] * 6, [True, True, False, False, True, True]])
    closed = module._close_event_weight(weight, {"source_open": source_open})
    assert torch.equal(closed[0], weight[0])
    assert torch.allclose(closed[1], torch.tensor([1.0, 1.0, 0.25, 0.25, 1.0, 1.0]))
    assert module._close_event_weight(None, {"source_open": source_open}) is None
    assert module._close_event_weight(weight, {}) is weight


def _fhr_masked_batch():
    """The stub batch with all-valid FHR step masks at the causal block widths."""
    batch = make_stub_batch()
    batch.fhr_st_step_valid = torch.ones(batch.fhr_st.shape, dtype=torch.bool)
    batch.fhr_ph_step_valid = torch.ones(batch.fhr_ph.shape, dtype=torch.bool)
    return batch


def test_the_target_quality_mask_zeroes_only_steps_a_forecast_channel_marks_bad():
    """A bad kept channel removes its step from the weight; a bad channel the target gate drops
    does not. Off, the weight is returned untouched."""
    module = make_task(hparams={"target_quality_mask": True})
    batch = _fhr_masked_batch()
    kept = TINY_TARGET_KEEP_INDEX[0]
    dropped = sorted(set(range(CAUSAL_C_Y)) - set(TINY_TARGET_KEEP_INDEX))[0]
    assert kept < CAUSAL_ST_WIDTH  # so the flag lands in fhr_st, the first block
    batch.fhr_st_step_valid[0, 5, kept] = False
    block, channel = (
        ("fhr_st_step_valid", dropped) if dropped < CAUSAL_ST_WIDTH
        else ("fhr_ph_step_valid", dropped - CAUSAL_ST_WIDTH)
    )
    getattr(batch, block)[1, 7, channel] = False

    weighted = module._target_quality_weight(batch, batch.weight)
    expected = batch.weight.clone()
    expected[0, 5] = 0.0
    assert torch.equal(weighted, expected)
    assert make_task()._target_quality_weight(batch, batch.weight) is batch.weight


def test_the_target_quality_mask_refuses_a_batch_without_it_or_at_the_wrong_width():
    module = make_task(hparams={"target_quality_mask": True})
    with pytest.raises(RuntimeError, match="fhr_st_step_valid"):
        module._target_quality_weight(make_stub_batch(), make_stub_batch().weight)
    batch = _fhr_masked_batch()
    batch.fhr_ph_step_valid = batch.fhr_ph_step_valid[..., :-1]
    with pytest.raises(RuntimeError, match="channels wide"):
        module._target_quality_weight(batch, batch.weight)


def test_the_detectors_ua_mask_needs_up_valid_and_all_valid_changes_nothing():
    """``event_weight_ua_valid`` reads the loader's ``up_valid``; without it the step raises,
    and an all-valid mask gives the weight of the run without the switch."""
    floor = {"event_weight_floor": 0.1}
    on = make_task(hparams=dict(floor, event_weight_ua_valid=True))
    off = make_task(hparams=floor)
    batch = make_stub_batch()
    assert off._event_ua_valid(batch) is None
    with pytest.raises(RuntimeError, match="up_valid"):
        on._event_anchor_weight(batch, batch.weight)
    batch.up_valid = torch.ones(batch.up.shape, dtype=torch.bool)
    assert torch.equal(
        on._event_anchor_weight(batch, batch.weight), off._event_anchor_weight(batch, batch.weight)
    )


def test_the_source_event_mask_ands_the_contraction_spans_into_the_source_mask(monkeypatch):
    """The spans come from the contraction weight's detector under the unmasked ``weight``, and
    a step outside them leaves every channel of the source mask."""
    model_kwargs = dict(tiny_warmup_kwargs(), source_quality_mask=True)
    module = make_task(model_kwargs, hparams={"event_weight_floor": 0.0, "source_event_mask": True})
    module._stage = "val"
    batch = _masked_batch()

    # The wiring: the task's spans are the shared helper's, with the task's own detector.
    spans = module._source_event_steps(batch, batch.weight)
    reference = contraction_source_mask(
        batch.up, batch.weight, decimation=int(module.orig_model.geometry.decimation),
        detector=module._event_detector(),
    )
    assert torch.equal(spans, reference)

    inside = torch.zeros(batch.weight.shape, dtype=torch.bool)
    inside[:, 4:9] = True
    seen = []

    def _steps(_batch, weight):
        seen.append(weight)
        return inside

    monkeypatch.setattr(module, "_source_event_steps", _steps)
    valid = module._build_forward_inputs(batch)[5]
    expected = torch.cat([batch.up_st_step_valid, batch.up_ph_step_valid], dim=-1) & inside[..., None]
    assert torch.equal(valid, expected)
    assert seen and seen[0] is batch.weight

    off = make_task(model_kwargs, hparams={"event_weight_floor": 0.0})
    off._stage = "val"
    assert off._source_event_steps(batch, batch.weight) is None
    assert torch.equal(
        off._build_forward_inputs(batch)[5],
        torch.cat([batch.up_st_step_valid, batch.up_ph_step_valid], dim=-1),
    )


def test_the_source_event_mask_is_refused_at_construction_without_a_source_mask():
    """A net without ``source_quality_mask`` has no source mask to narrow, so the task refuses the
    ablation when it is built rather than ignoring it at every batch."""
    with pytest.raises(ValueError, match="source_event_mask"):
        make_task(tiny_warmup_kwargs(), hparams={"event_weight_floor": 0.0, "source_event_mask": True})


def test_the_source_event_mask_refuses_a_batch_without_raw_up():
    module = make_task(
        dict(tiny_warmup_kwargs(), source_quality_mask=True),
        hparams={"event_weight_floor": 0.0, "source_event_mask": True},
    )
    batch = make_stub_batch()
    del batch.up
    with pytest.raises(RuntimeError, match="source_event_mask"):
        module._source_event_steps(batch, batch.weight)


# ---------------------------------------------------------------------------------------
# The per-anchor readout
# ---------------------------------------------------------------------------------------
def test_the_per_anchor_readout_carries_the_source_state_of_each_anchor(perturb_posterior):
    r"""``source_open`` is 0 at every anchor of a fully masked sample and 1 where nothing is
    masked, and the effective gate $g_t o_t$ is 0 wherever the anchor is closed."""
    module = make_task(dict(tiny_warmup_kwargs(posterior_event_gate=True), source_quality_mask=True))
    perturb_posterior(module.orig_model)
    module.eval()
    batch = make_stub_batch()
    batch.up_st_step_valid = torch.ones(batch.up_st.shape, dtype=torch.bool)
    batch.up_ph_step_valid = torch.ones(batch.up_ph.shape, dtype=torch.bool)
    batch.up_st_step_valid[0] = False
    torch.manual_seed(0)
    per_anchor = evaluate_batch(module, batch, num_samples=1).per_anchor

    assert per_anchor["source_open"].shape == per_anchor["anchor_index"].shape
    assert bool((per_anchor["source_open"][0] == 0.0).all())
    assert bool((per_anchor["source_open"][1] == 1.0).all())
    assert bool((per_anchor["event_gate"][0] == 0.0).all())

    plain = make_task(tiny_warmup_kwargs(posterior_event_gate=True))
    plain.eval()
    unmasked = evaluate_batch(plain, make_stub_batch(), num_samples=1).per_anchor
    assert "source_open" not in unmasked and "event_gate" in unmasked


def test_the_quality_labels_are_read_from_the_batch_whatever_the_model_applies(perturb_posterior):
    r"""``ua_key_open`` is the masked model's ``source_open`` on both a masked and an unmasked net,
    and ``target_bad_steps`` counts the forecast steps the FHR step mask removes. A control and a
    masked arm can then be compared on the same clean anchors."""
    batch = make_stub_batch()
    batch.up_st_step_valid = torch.ones(batch.up_st.shape, dtype=torch.bool)
    batch.up_ph_step_valid = torch.ones(batch.up_ph.shape, dtype=torch.bool)
    batch.up_st_step_valid[0] = False
    masked = make_task(dict(tiny_warmup_kwargs(posterior_event_gate=True), source_quality_mask=True))
    plain = make_task(tiny_warmup_kwargs(posterior_event_gate=True))
    for module in (masked, plain):
        perturb_posterior(module.orig_model)
        module.eval()
    torch.manual_seed(0)
    on = evaluate_batch(masked, batch, num_samples=1).per_anchor
    torch.manual_seed(0)
    off = evaluate_batch(plain, batch, num_samples=1).per_anchor
    assert torch.equal(on["ua_key_open"], on["source_open"])
    assert torch.equal(off["ua_key_open"], on["source_open"])
    assert "target_bad_steps" not in off

    y_width = batch.fhr_st.shape[-1]
    batch.fhr_st_step_valid = torch.ones(batch.fhr_st.shape, dtype=torch.bool)
    batch.fhr_ph_step_valid = torch.ones(batch.fhr_ph.shape, dtype=torch.bool)
    clean = evaluate_batch(plain, batch, num_samples=1).per_anchor
    assert bool((clean["target_bad_steps"] == 0.0).all())
    kept = int(TINY_TARGET_KEEP_INDEX[0])
    block, channel = ("fhr_st", kept) if kept < y_width else ("fhr_ph", kept - y_width)
    getattr(batch, f"{block}_step_valid")[1, :, channel] = False
    gapped = evaluate_batch(plain, batch, num_samples=1).per_anchor
    assert bool((gapped["target_bad_steps"][0] == 0.0).all())
    supported = gapped["contributing"][1] > 0
    assert bool((gapped["target_bad_steps"][1][supported] > 0.0).all())


# ---------------------------------------------------------------------------------------
# The pre-flight pairing
# ---------------------------------------------------------------------------------------
def _config(flag, spec):
    return {
        "model_config": {"VAE_model": {"source_quality_mask": flag}},
        "dataset_config": {"dataloader_config": {"dataset_kwargs": {"ua_step_mask": spec}}},
    }


@pytest.mark.parametrize(
    "flag, spec",
    [
        (True, {"bits": None, "rho_epsilon": 0.1, "hold_steps": 0}),
        (True, {"bits": ["TRANSIENT", "HOLD"]}),
        (False, None),
        (False, {"bits": None, "rho_epsilon": None, "hold_steps": 0}),
    ],
)
def test_the_preflight_accepts_a_paired_flag_and_rule(flag, spec):
    _check_source_quality_pairing(_config(flag, copy.deepcopy(spec)))


@pytest.mark.parametrize(
    "flag, spec",
    [
        (True, None),
        (True, {"bits": None, "rho_epsilon": None}),
        (False, {"bits": None, "rho_epsilon": 0.1}),
        (False, {"bits": ["invalid"]}),
    ],
)
def test_the_preflight_refuses_a_flag_without_its_rule_and_the_reverse(flag, spec):
    with pytest.raises(ValueError, match="source_quality_mask"):
        _check_source_quality_pairing(_config(flag, spec))


def _switch_config(vae, kwargs):
    return {
        "model_config": {"VAE_model": dict(vae)},
        "dataset_config": {"dataloader_config": {"dataset_kwargs": dict(kwargs)}},
    }


#: The shipped combination: the floor, both raw-grid masks, and the FHR step rule.
_SHIPPED_VAE = {
    "event_weight_floor": 0.0,
    "source_quality_mask": True,
    "event_weight_ua_valid": True,
    "target_quality_mask": True,
    "source_event_mask": False,
}
_SHIPPED_KWARGS = {"emit_quality_mask": True, "fhr_step_mask": {"bits": "invalid", "rho_epsilon": None}}


@pytest.mark.parametrize(
    "vae, kwargs",
    [
        ({}, {}),
        (_SHIPPED_VAE, _SHIPPED_KWARGS),
        (dict(_SHIPPED_VAE, source_event_mask=True), _SHIPPED_KWARGS),
        ({"target_quality_mask": False}, {"fhr_step_mask": {"bits": None, "rho_epsilon": None}}),
    ],
)
def test_the_preflight_accepts_the_quality_switches_with_their_fields(vae, kwargs):
    _check_quality_switch_pairing(_switch_config(vae, copy.deepcopy(kwargs)))


@pytest.mark.parametrize(
    "vae, kwargs, match",
    [
        # The detector's UA mask without the field, and without a detector to give it to.
        (dict(_SHIPPED_VAE), dict(_SHIPPED_KWARGS, emit_quality_mask=False), "event_weight_ua_valid"),
        (dict(_SHIPPED_VAE, event_weight_floor=None), _SHIPPED_KWARGS, "event_weight_ua_valid"),
        # The target mask without its rule, and a rule nothing reads.
        (dict(_SHIPPED_VAE), dict(_SHIPPED_KWARGS, fhr_step_mask=None), "target_quality_mask"),
        (dict(_SHIPPED_VAE, target_quality_mask=False), _SHIPPED_KWARGS, "target_quality_mask"),
        # The source ablation without the key mask it narrows, and without a detector.
        (dict(_SHIPPED_VAE, source_event_mask=True, source_quality_mask=False), _SHIPPED_KWARGS,
         "source_event_mask"),
        (dict(_SHIPPED_VAE, source_event_mask=True, event_weight_floor=None, event_weight_ua_valid=False),
         _SHIPPED_KWARGS, "source_event_mask"),
    ],
)
def test_the_preflight_refuses_a_quality_switch_without_its_field(vae, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _check_quality_switch_pairing(_switch_config(vae, copy.deepcopy(kwargs)))


_CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", _CONFIGS, ids=[path.stem for path in _CONFIGS])
def test_every_shipped_config_passes_both_quality_checks(path):
    """A shipped arm that the pre-flight refuses would fail on the cluster before its first step."""
    config = load_config(str(path))
    _check_source_quality_pairing(config)
    _check_quality_switch_pairing(config)
