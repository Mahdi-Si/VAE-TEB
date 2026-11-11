r"""The UA tolerance port: running-gap key removal, tolerant flags, tolerant reward, decoder bound.

Four mechanisms, each behind a key that is inert at its default:

* ``source_running_gap_keys`` (model): with ``'remove'``, the key at lag $\ell$ of anchor $t$
  leaves the lag attention when an invalid UA run $[a, b]$ satisfies $a < D(t - \ell + 1)$ and
  $b + g + 1 \ge D(t + 1)$, because the bridge across the run reads samples after the anchor;
* ``source_valid_tolerance_s`` (task): a flag turns valid where the channel's untrimmed causal
  window holds no sample of a long run;
* ``event_weight_ua_tolerance_s`` (task): the contraction detector counts short runs as valid;
* ``decoder_logvar_clamp`` (model): the decoder's own log-variance bound.

What is asserted here:

* the pure rules of :mod:`teb_vae.lag_attn_cfs.ua_tolerance` against hand-worked cases and a
  brute-force reference, including the guard, a run that reaches the segment end and the trap of
  a window counted on the trimmed grid;
* the two-way refusals of the forward's seventh argument and of the keys that need another key;
* default inertness: the state dict and the forward are bitwise the model before the keys, and
  an all-kept removal mask is bitwise the forward without removal;
* the gate closes, and $q = p$, at an anchor with no key left, in both mask modes;
* the controls re-attend with the removal, and the permutation arm carries the donor's;
* the task builds the seventh input, the tolerant flags and the tolerant detector validity;
* the decoder bound reaches the decoder alone, and the loss reads its floor fraction against it.
"""
from __future__ import annotations

import inspect
import math

import h5py
import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs import ua_tolerance as U
from teb_vae.lag_attn_cfs.trainer import _check_ua_tolerance_pairing
from teb_vae.lag_attn_rws.nets.controls import perm_forward_outputs, source_null_forward_outputs
from teb_vae.lag_attn_rws.task import checkpoint_task_kwargs

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_SHARD,
    build,
    causal_config,
    make_stub_batch,
    make_streams,
    make_task,
    tiny_warmup_kwargs,
)

#: Three samples, so the permutation control has a non-trivial derangement.
BATCH = 3
#: Raw samples per feature step in the hand-worked cases; the model's own is 16.
D = 4


def _valid_from_runs(length, runs, rows=1):
    """A raw validity ``(rows, length)`` with the inclusive ``(row, a, b)`` runs invalid."""
    valid = torch.ones(rows, length, dtype=torch.bool)
    for row, a, b in runs:
        valid[row, a:b + 1] = False
    return valid


def _brute_removed(up_valid, steps, lags, decimation, guard):
    """$R_{t\\ell}$ written out from its definition, run by run."""
    invalid = ~up_valid.numpy()
    removed = np.zeros((invalid.shape[0], steps, lags), dtype=bool)
    rows, starts, ends = U.run_table(invalid)
    for row, a, b in zip(rows.tolist(), starts.tolist(), ends.tolist()):
        for t in range(steps):
            for lag in range(min(lags, t + 1)):
                if a < decimation * (t - lag + 1) and b + guard + 1 >= decimation * (t + 1):
                    removed[row, t, lag] = True
    return torch.from_numpy(removed)


# ---------------------------------------------------------------------------------------
# The pure rules
# ---------------------------------------------------------------------------------------
def test_the_run_table_finds_every_run_and_no_run_crosses_a_row():
    invalid = np.array([[True, False, False, True, True], [True, True, False, False, True]])
    rows, starts, ends = U.run_table(invalid)
    assert rows.tolist() == [0, 0, 1, 1]
    assert starts.tolist() == [0, 3, 0, 4]
    assert ends.tolist() == [0, 4, 1, 4]


def test_a_run_at_most_the_tolerance_is_short_and_a_longer_one_is_long():
    # 1 s at 4 Hz is 4 samples: a 4-sample run is short, a 5-sample run is long.
    valid = _valid_from_runs(20, [(0, 2, 5), (0, 10, 14)])
    long_bad, short_bad = U.split_runs(valid, 1.0)
    assert short_bad[0].nonzero().flatten().tolist() == [2, 3, 4, 5]
    assert long_bad[0].nonzero().flatten().tolist() == [10, 11, 12, 13, 14]
    assert not bool((long_bad & short_bad).any())
    assert torch.equal(long_bad | short_bad, ~valid)


def test_tolerant_flags_clear_a_step_only_when_its_window_holds_no_long_run_sample():
    # T = 6 steps of D = 4 samples; one long run over samples 8..15, i.e. steps 2 and 3.
    valid = _valid_from_runs(6 * D, [(0, 8, 15)])
    flags = torch.zeros(1, 6, 2, dtype=torch.bool)
    windows = torch.tensor([1, 3])
    out = U.tolerant_flags(flags, valid, windows, tolerance_s=1.0, decimation=D)
    # W = 1 reads its own step: steps 2 and 3 stay invalid.
    assert out[0, :, 0].tolist() == [True, True, False, False, True, True]
    # W = 3 reads [D(s - 2), D(s + 1)): every step from 2 to 5 sees the run.
    assert out[0, :, 1].tolist() == [True, True, False, False, False, False]


def test_tolerant_flags_only_ever_turn_a_flag_valid():
    generator = torch.Generator().manual_seed(0)
    valid = torch.rand(3, 10 * D, generator=generator) > 0.2
    flags = torch.rand(3, 10, 4, generator=generator) > 0.5
    out = U.tolerant_flags(flags, valid, torch.tensor([1, 2, 5, 12]), 0.5, decimation=D)
    assert bool((out | ~flags).all())          # flags => out
    assert bool(out[flags].all())


def test_a_short_run_alone_turns_every_flag_valid_and_no_tolerance_changes_nothing():
    valid = _valid_from_runs(6 * D, [(0, 9, 10)])
    flags = torch.zeros(1, 6, 3, dtype=torch.bool)
    assert bool(U.tolerant_flags(flags, valid, torch.tensor([1, 4, 9]), 1.0, decimation=D).all())
    assert U.tolerant_flags(flags, valid, torch.tensor([1, 4, 9]), None, decimation=D) is flags
    assert U.tolerant_flags(flags, valid, torch.tensor([1, 4, 9]), 0.0, decimation=D) is flags


def test_the_trimmed_window_is_the_trap_the_untrimmed_window_avoids():
    """On the trimmed grid a fast channel reads 0 steps, clamped to 1: it then turns valid at a
    step whose untrimmed window still holds a long-run sample."""
    valid = _valid_from_runs(8 * D, [(0, 0, 7)])      # a long run over steps 0 and 1
    flags = torch.zeros(1, 8, 1, dtype=torch.bool)
    trimmed = U.tolerant_flags(flags, valid, torch.tensor([0]), 1.0, decimation=D)
    untrimmed = U.tolerant_flags(flags, valid, torch.tensor([5]), 1.0, decimation=D)
    assert bool(trimmed[0, 2, 0]) and not bool(untrimmed[0, 2, 0])
    assert untrimmed[0, :, 0].tolist() == [False] * 6 + [True, True]


def test_the_declared_windows_are_the_shards_untrimmed_warmup_not_the_trimmed_one():
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    windows = U.declared_source_windows([str(CAUSAL_SHARD)], use_up_st=True)
    with h5py.File(CAUSAL_SHARD, "r") as handle:
        stored = np.concatenate([np.asarray(handle[name].attrs["causal_warmup_steps"])
                                 for name in ("up_st", "up_ph")])
    assert list(windows) == [max(int(step), 1) for step in stored]
    assert len(windows) == CAUSAL_C_U
    trimmed = resolve_warmup_budget(causal_config()).source.declared_warmup_steps
    # The trap: the trim clips the fast channels to 0 where the stored window is not.
    assert any(t == 0 < w for t, w in zip(trimmed, windows))


def test_the_task_fallback_windows_are_never_shorter_than_the_stored_ones():
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    module = make_task()
    module.warmup_budget = resolve_warmup_budget(causal_config())
    fallback = module._source_valid_windows()
    stored = torch.tensor(U.declared_source_windows([str(CAUSAL_SHARD)]))
    assert bool((fallback >= stored).all())
    module.hparams["source_valid_windows"] = stored.tolist()
    assert torch.equal(module._source_valid_windows(), stored)


def test_removed_keys_match_the_definition_on_random_masks():
    generator = torch.Generator().manual_seed(1)
    valid = torch.rand(4, 12 * D, generator=generator) > 0.15
    for guard_s in (0.0, 1.0, 4.0):
        got = U.removed_keys(valid, num_steps=12, num_lags=5, decimation=D, guard_s=guard_s)
        want = _brute_removed(valid, 12, 5, D, U.seconds_to_samples(guard_s))
        assert torch.equal(got, want), guard_s


def test_the_guard_keeps_a_run_running_past_its_last_sample():
    # A one-sample run at sample 11, the last of step 2. Without a guard the bridge still reads
    # sample 12, after anchor 2, so the run is running at anchor 2 alone; a guard of 4 samples
    # makes it read up to sample 16, so it is running at anchor 3 (samples 12..15) too.
    valid = _valid_from_runs(8 * D, [(0, 11, 11)])
    bare = U.removed_keys(valid, num_steps=8, num_lags=4, decimation=D, guard_s=0.0)
    guarded = U.removed_keys(valid, num_steps=8, num_lags=4, decimation=D, guard_s=1.0)
    assert bare[0].nonzero().tolist() == [[2, 0]]
    assert guarded[0].nonzero().tolist() == [[2, 0], [3, 0], [3, 1]]


def test_a_run_reaching_the_segment_end_removes_keys_up_to_the_last_anchor():
    valid = _valid_from_runs(8 * D, [(0, 22, 8 * D - 1)])     # steps 5..7, to the end
    removed = U.removed_keys(valid, num_steps=8, num_lags=4, decimation=D, guard_s=4.0)
    assert not bool(removed[0, :5].any())
    for t in (5, 6, 7):
        assert removed[0, t].tolist() == [lag <= t - 5 for lag in range(4)]


def test_the_raw_grid_must_be_the_feature_grid():
    with pytest.raises(ValueError, match="trimmed raw grid"):
        U.removed_keys(torch.ones(1, 31, dtype=torch.bool), num_steps=8, num_lags=4, decimation=D)


def test_the_tolerant_reward_counts_short_runs_valid_and_keeps_long_ones():
    valid = _valid_from_runs(40, [(0, 2, 5), (0, 10, 30)])
    out = U.tolerant_reward_validity(valid, 1.0)
    assert bool(out[0, 2:6].all())
    assert not bool(out[0, 10:31].any())
    assert U.tolerant_reward_validity(valid, None) is valid


# ---------------------------------------------------------------------------------------
# Construction, refusals and inertness of the model keys
# ---------------------------------------------------------------------------------------
def _removal_kwargs(mode="hard", **overrides):
    return tiny_warmup_kwargs(
        source_quality_mask=True,
        source_mask_mode=mode,
        posterior_event_gate=True,
        **overrides,
    )


def test_the_removal_needs_the_quality_mask_and_a_known_mode():
    with pytest.raises(ValueError, match="needs source_quality_mask"):
        build(tiny_warmup_kwargs(source_running_gap_keys="remove"))
    with pytest.raises(ValueError, match="must be one of"):
        build(_removal_kwargs(source_running_gap_keys="drop"))


def test_the_removal_and_its_forward_argument_travel_together():
    keep = build(_removal_kwargs()).eval()
    remove = build(_removal_kwargs(source_running_gap_keys="remove")).eval()
    keep_kwargs = _removal_kwargs()
    y_st, y_ph, u = make_streams(keep_kwargs, batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    removed = torch.zeros(BATCH, u.shape[1], int(remove.lag_attn.L), dtype=torch.bool)
    with torch.no_grad():
        with pytest.raises(ValueError, match="needs source_key_removed"):
            remove(y_st, y_ph, u, None, None, valid)
        with pytest.raises(ValueError, match="source_running_gap_keys='remove' to apply"):
            keep(y_st, y_ph, u, None, None, valid, removed)
        with pytest.raises(ValueError, match="has shape"):
            remove(y_st, y_ph, u, None, None, valid, removed[..., :-1])
    assert keep_kwargs["source_mask_mode"] == "hard"


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_the_new_keys_add_nothing_to_the_state_dict(mode):
    plain = build(_removal_kwargs(mode))
    explicit = build(_removal_kwargs(mode, source_running_gap_keys="remove",
                                     decoder_logvar_clamp=(-9.0, 3.0)))
    assert list(plain.state_dict()) == list(explicit.state_dict())
    for name, tensor in plain.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name


def test_the_defaults_are_bitwise_the_model_before_the_keys(perturb_posterior):
    kwargs = _removal_kwargs("soft")
    before = build(kwargs).eval()
    after = build(dict(kwargs, source_running_gap_keys="keep", decoder_logvar_clamp=None)).eval()
    perturb_posterior(before)
    perturb_posterior(after)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    valid[0, 5:9] = False
    with torch.no_grad():
        torch.manual_seed(0)
        one = before(y_st, y_ph, u, None, None, valid)
        torch.manual_seed(0)
        two = after(y_st, y_ph, u, None, None, valid)
    assert sorted(one) == sorted(two)
    for key, value in one.items():
        assert torch.equal(value, two[key]), key


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_an_all_kept_removal_mask_is_bitwise_the_forward_without_removal(perturb_posterior, mode):
    keep = build(_removal_kwargs(mode)).eval()
    remove = build(_removal_kwargs(mode, source_running_gap_keys="remove")).eval()
    perturb_posterior(keep)
    perturb_posterior(remove)
    y_st, y_ph, u = make_streams(_removal_kwargs(mode), batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    valid[1, 7] = False
    removed = torch.zeros(BATCH, u.shape[1], int(remove.lag_attn.L), dtype=torch.bool)
    with torch.no_grad():
        torch.manual_seed(0)
        plain = keep(y_st, y_ph, u, None, None, valid)
        torch.manual_seed(0)
        masked = remove(y_st, y_ph, u, None, None, valid, removed)
    for key in ("mu_post", "logvar_post", "attn_weights", "event_gate", "kld_per_t"):
        assert torch.equal(plain[key], masked[key]), key
    assert torch.equal(masked["source_key_removed"], removed)


# ---------------------------------------------------------------------------------------
# The forward under removal
# ---------------------------------------------------------------------------------------
def _removed_mask(model, u, closed_steps=(), removed_lag=None):
    """Sample 0 has every key removed at ``closed_steps``; sample 1 loses lag ``removed_lag``."""
    removed = torch.zeros(u.shape[0], u.shape[1], int(model.lag_attn.L), dtype=torch.bool)
    for step in closed_steps:
        removed[0, step] = True
    if removed_lag is not None:
        removed[1, :, removed_lag] = True
    return removed


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_an_anchor_with_no_key_left_closes_its_gate_and_keeps_q_equal_p(perturb_posterior, mode):
    model = build(_removal_kwargs(mode, source_running_gap_keys="remove")).eval()
    perturb_posterior(model)
    y_st, y_ph, u = make_streams(_removal_kwargs(mode), batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    closed = (12, 13, 14)
    removed = _removed_mask(model, u, closed_steps=closed, removed_lag=0)
    with torch.no_grad():
        out = model(y_st, y_ph, u, None, None, valid, removed)
    for step in closed:
        assert float(out["event_gate"][0, step].abs().max()) == 0.0
        assert torch.equal(out["mu_post"][0, step], out["mu_prior"][0, step])
    # A removed lag gets no attention weight; the other samples keep their keys.
    assert float(out["attn_weights"][1, :, :, 0].abs().max()) == 0.0
    if mode == "hard":
        assert not bool(out["source_open"][0, list(closed)].any())
        assert torch.equal(out["source_key_mask"], model.source_key_mask(
            valid, model.build_lag_mask(u.shape[1], u.device), u.dtype))


# ---------------------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_the_permutation_arm_carries_the_donors_removal(perturb_posterior, mode):
    model = build(_removal_kwargs(mode, source_running_gap_keys="remove")).eval()
    perturb_posterior(model)
    y_st, y_ph, u = make_streams(_removal_kwargs(mode), batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    every_step = tuple(range(u.shape[1]))
    removed = _removed_mask(model, u, closed_steps=every_step)
    with torch.no_grad():
        matched = model(y_st, y_ph, u, None, None, valid, removed)
        shuffled = perm_forward_outputs(model, matched, perm_index=torch.tensor([1, 2, 0]))
    # Sample 2 received sample 0's source, every key of which is removed: its posterior is its prior.
    assert torch.equal(shuffled["source_key_removed"][2], removed[0])
    assert torch.equal(shuffled["mu_post"][2], matched["mu_prior"][2])
    if mode == "hard":
        assert torch.equal(
            shuffled["source_open"],
            (shuffled["source_key_mask"] & ~shuffled["source_key_removed"]).any(dim=-1),
        )


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_the_source_null_arm_keeps_the_recordings_removal(perturb_posterior, mode):
    model = build(_removal_kwargs(mode, source_running_gap_keys="remove")).eval()
    perturb_posterior(model)
    y_st, y_ph, u = make_streams(_removal_kwargs(mode), batch=BATCH)
    valid = torch.ones(u.shape, dtype=torch.bool)
    removed = _removed_mask(model, u, closed_steps=tuple(range(u.shape[1])))
    with torch.no_grad():
        matched = model(y_st, y_ph, u, None, None, valid, removed)
        nulled = source_null_forward_outputs(model, matched, u)
    assert torch.equal(nulled["mu_post"][0], matched["mu_prior"][0])
    assert float(nulled["event_gate"][0].abs().max()) == 0.0


# ---------------------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------------------
def _quality_batch(long_run=True):
    batch = make_stub_batch()
    batch.up_st_step_valid = torch.ones(batch.up_st.shape, dtype=torch.bool)
    batch.up_ph_step_valid = torch.ones(batch.up_ph.shape, dtype=torch.bool)
    batch.up_ph_step_valid[0, 6:10] = False
    batch.up_valid = torch.ones(batch.up.shape, dtype=torch.bool)
    batch.up_valid[0, 100:104] = False                    # 1 s: short at any tolerance >= 1 s
    if long_run:
        batch.up_valid[1, 200:280] = False                # 20 s: long at 16 s
    return batch


def _tolerance_task(**hparams):
    model_kwargs = dict(_removal_kwargs("soft"), source_running_gap_keys="remove")
    return make_task(model_kwargs, hparams=hparams)


def test_the_task_appends_the_removal_as_the_seventh_input():
    module = _tolerance_task()
    module._stage = "val"
    batch = _quality_batch()
    inputs = module._build_forward_inputs(batch)
    assert len(inputs) == 7
    model = module.orig_model
    expected = U.removed_keys(batch.up_valid, num_steps=inputs[2].shape[1],
                              num_lags=int(model.lag_attn.L), decimation=int(model.geometry.decimation),
                              guard_s=4.0)
    assert torch.equal(inputs[6], expected)
    assert bool(inputs[6].any())
    out = model(*inputs)
    assert torch.equal(out["source_key_removed"], expected)


def test_the_task_refuses_a_removal_batch_without_up_valid():
    module = _tolerance_task()
    module._stage = "val"
    batch = _quality_batch()
    del batch.up_valid
    with pytest.raises(RuntimeError, match="up_valid"):
        module._build_forward_inputs(batch)


def test_the_task_relaxes_the_flags_with_the_untrimmed_windows():
    windows = [3] * CAUSAL_C_U
    module = _tolerance_task(source_valid_tolerance_s=16.0, source_valid_windows=windows)
    module._stage = "val"
    batch = _quality_batch()
    source = module._build_source_stream(batch)
    got = module._build_source_valid(batch, source)
    dataset = torch.cat([batch.up_st_step_valid, batch.up_ph_step_valid], dim=-1)
    want = U.tolerant_flags(dataset, batch.up_valid, torch.tensor(windows), 16.0,
                            decimation=int(module.orig_model.geometry.decimation))
    assert torch.equal(got, want)
    assert bool((got & ~dataset).any())


def test_the_task_refuses_the_tolerant_flags_without_the_removal():
    with pytest.raises(ValueError, match="look ahead through the bridge"):
        make_task(dict(_removal_kwargs("soft")), hparams={"source_valid_tolerance_s": 16.0})


def test_the_detector_reads_short_runs_as_valid_under_the_reward_tolerance():
    hparams = {"event_weight_floor": 0.1, "event_weight_ua_valid": True}
    strict = make_task(hparams=hparams)
    tolerant = make_task(hparams=dict(hparams, event_weight_ua_tolerance_s=16.0))
    batch = _quality_batch()
    assert torch.equal(strict._event_ua_valid(batch), batch.up_valid)
    assert torch.equal(tolerant._event_ua_valid(batch), U.tolerant_reward_validity(batch.up_valid, 16.0))
    with pytest.raises(ValueError, match="event_weight_ua_valid is off"):
        make_task(hparams={"event_weight_floor": 0.1, "event_weight_ua_tolerance_s": 16.0})


def test_a_tolerance_must_be_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        make_task(hparams={"event_weight_floor": 0.1, "event_weight_ua_valid": True,
                           "event_weight_ua_tolerance_s": 0.0})


def test_the_window_vector_travels_in_the_checkpoint_hyperparameters():
    from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask

    windows = [3] * CAUSAL_C_U
    module = _tolerance_task(source_valid_tolerance_s=16.0, source_valid_windows=windows,
                             ua_bridge_guard_s=2.0)
    rebuilt = checkpoint_task_kwargs(SeqVaeLagAttnTrfCfsTask, dict(module.hparams))
    assert rebuilt["source_valid_windows"] == windows
    assert rebuilt["source_valid_tolerance_s"] == 16.0
    assert rebuilt["ua_bridge_guard_s"] == 2.0


def test_the_training_step_runs_and_back_propagates_with_every_key_on():
    model_kwargs = dict(_removal_kwargs("soft"), source_running_gap_keys="remove",
                        decoder_logvar_clamp=(-9.0, 3.0))
    module = make_task(model_kwargs, hparams={
        "event_weight_floor": 0.1, "event_weight_ua_valid": True,
        "event_weight_ua_tolerance_s": 16.0, "source_valid_tolerance_s": 16.0,
        "source_valid_windows": [3] * CAUSAL_C_U, "lambda_source_null": 1.0,
    })
    loss, _ = module.compute_loss_and_metrics(_quality_batch(), 0, "train")
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in module.orig_model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


# ---------------------------------------------------------------------------------------
# The pre-flight
# ---------------------------------------------------------------------------------------
_EMITS = {"emit_quality_mask": True}


@pytest.mark.parametrize(
    "vae, kwargs",
    [
        ({}, {}),
        ({"source_running_gap_keys": "keep"}, {}),
        ({"source_running_gap_keys": "remove", "source_quality_mask": True}, _EMITS),
        ({"source_running_gap_keys": "remove", "source_quality_mask": True,
          "source_valid_tolerance_s": 16.0}, _EMITS),
        ({"event_weight_ua_valid": True, "event_weight_ua_tolerance_s": 16.0}, _EMITS),
    ],
)
def test_the_preflight_accepts_the_tolerance_keys_with_what_they_need(vae, kwargs):
    _check_ua_tolerance_pairing(dict(vae), dict(kwargs))


@pytest.mark.parametrize(
    "vae, kwargs, match",
    [
        ({"source_running_gap_keys": "remove", "source_quality_mask": True}, {}, "emit_quality_mask"),
        ({"source_running_gap_keys": "remove"}, _EMITS, "source_quality_mask"),
        ({"source_valid_tolerance_s": 16.0, "source_quality_mask": True}, _EMITS, "through the bridge"),
        ({"event_weight_ua_tolerance_s": 16.0}, _EMITS, "event_weight_ua_valid"),
    ],
)
def test_the_preflight_refuses_a_tolerance_key_without_what_it_needs(vae, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _check_ua_tolerance_pairing(dict(vae), dict(kwargs))


# ---------------------------------------------------------------------------------------
# The decoder's own log-variance bound
# ---------------------------------------------------------------------------------------
def test_the_decoder_bound_reaches_the_decoder_alone(perturb_posterior):
    kwargs = tiny_warmup_kwargs()
    shared = build(kwargs).eval()
    own = build(dict(kwargs, decoder_logvar_clamp=(-9.0, 3.0))).eval()
    perturb_posterior(shared)
    perturb_posterior(own)
    assert own.decoder.logvar_clamp == (-9.0, 3.0)
    assert own.decoder_logvar_bound == (-9.0, 3.0)
    assert tuple(own.logvar_clamp) == tuple(shared.logvar_clamp)
    assert shared.decoder_logvar_bound == tuple(float(v) for v in shared.logvar_clamp)
    y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
    with torch.no_grad():
        torch.manual_seed(0)
        one = shared(y_st, y_ph, u)
        torch.manual_seed(0)
        two = own(y_st, y_ph, u)
    for key in ("mu_prior", "logvar_prior", "mu_post", "logvar_post"):
        assert torch.equal(one[key], two[key]), key
    for key in ("logvar_full", "logvar_base"):
        assert float(two[key].min()) >= -9.0 and float(two[key].max()) <= 3.0
    assert not torch.equal(one["logvar_full"], two["logvar_full"])


def test_the_decoder_bound_is_refused_unless_lo_is_below_hi():
    with pytest.raises(ValueError, match="lo < hi"):
        build(tiny_warmup_kwargs(decoder_logvar_clamp=(3.0, -9.0)))


def test_the_calibrated_init_seeds_the_decoder_at_unit_scale_under_its_own_bound():
    model = build(tiny_warmup_kwargs(head_init_calibration=True, decoder_logvar_clamp=(-9.0, 3.0)))
    bias = model.decoder.logvar_head.bias.detach()
    assert torch.allclose(bias, torch.full_like(bias, math.log(9.0 / 3.0)))


def test_the_loss_reads_the_forecast_floor_against_the_decoders_bound(perturb_posterior):
    from teb_vae.lag_attn_cfs.eval.metrics import decoder_logvar_bound

    kwargs = tiny_warmup_kwargs()
    floors = {}
    for name, clamp in (("shared", None), ("own", (-9.0, 3.0))):
        model = build(dict(kwargs, decoder_logvar_clamp=clamp)).eval()
        perturb_posterior(model)
        y_st, y_ph, u = make_streams(kwargs, batch=BATCH)
        with torch.no_grad():
            out = model(y_st, y_ph, u)
            # A forecast log-variance of -6: below the shared floor (-5), well above the decoder's.
            out = {**out, "logvar_full": torch.full_like(out["logvar_full"], -6.0)}
            metrics = model.compute_loss(
                out, torch.cat([y_st, y_ph], dim=-1), weight=torch.ones(BATCH, y_st.shape[1]),
                beta=0.0, beta_prior=0.0, lambda_full=1.0, lambda_base=1.0,
                likelihood="gaussian_nll", free_bits=0.0, lambda_ms=0.0, lambda_deriv=0.0,
                lambda_boundary=0.0,
            )["metrics"]
        floors[name] = float(metrics["logvar_full_floor_frac"])
        assert decoder_logvar_bound(model) == model.decoder_logvar_bound
    assert floors["shared"] == pytest.approx(1.0)
    assert floors["own"] == pytest.approx(0.0)


def test_the_calibration_analysis_reads_the_decoders_own_clamp():
    from teb_vae.lag_attn_cfs.eval.analyses.calibration import _decoder_clamp

    same = {"logvar_clamp": [-6.0, 3.0], "logvar_margin": 0.45,
            "decoder_logvar_clamp": [-6.0, 3.0], "decoder_logvar_margin": 0.45}
    own = dict(same, decoder_logvar_clamp=[-9.0, 3.0], decoder_logvar_margin=0.6)
    assert _decoder_clamp(same) == ([-6.0, 3.0], 0.45, "model_config.VAE_model.logvar_clamp")
    assert _decoder_clamp(own) == ([-9.0, 3.0], 0.6, "model_config.VAE_model.decoder_logvar_clamp")
    assert _decoder_clamp({"logvar_clamp": [-5.0, 3.0], "logvar_margin": 0.4})[0] == [-5.0, 3.0]


def test_both_causal_cells_take_the_two_model_keywords():
    from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
    from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs

    for cls in (SeqVaeLagAttnCfs, SeqVaeLagAttnTrfCfs):
        parameters = inspect.signature(cls.__init__).parameters
        assert parameters["source_running_gap_keys"].default == "keep"
        assert parameters["decoder_logvar_clamp"].default is None
