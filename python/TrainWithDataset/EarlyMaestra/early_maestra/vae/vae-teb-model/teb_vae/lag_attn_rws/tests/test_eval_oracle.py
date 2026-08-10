r"""The oracle probe: that it mirrors the decoder, reads the state, and cannot touch the model.

"The loss went down" is not evidence a probe measured anything -- it passes with the conditioning
wired to a constant, which is precisely the failure that would make $\Delta_{\mathrm{suff}}$ read
as zero and be reported as "the bottleneck costs nothing". So the fit is questioned three ways,
and each one fails on a different bug:

* **the conditioning ablation** -- a probe fitted on *another recording's* encoder state must end
  up worse on the matched held-out half. This is what separates "learned to read the state" from
  "learned the population mean";
* **the known answer** -- on a cache where the forecast block is a deterministic function of the
  encoder state, the probe must drive its held-out score below the trivial $\mu = 0, \sigma = 1$
  predictor's. A probe that cannot solve a solved problem cannot bound anything;
* **bypass isolation** -- after the probe's optimizer step, no production parameter carries a
  gradient. The layering test proves ``nets/`` does not import the oracle, which is a different
  and much weaker guarantee: it says the model cannot call the probe, not that the probe cannot
  move the model.

The known-answer cache is built by hand rather than drawn from a shard, because "the block is a
deterministic function of the state" is a property no generated recording has, and the whole point
of a known answer is that the answer is known.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_rws.eval import oracle
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

from .conftest import TINY_KWARGS

#: The model geometry the loader-driven tests build. The generated shards are $300$ decimated
#: steps at the production horizon, so the geometry has to be the real one; only the widths shrink,
#: which is what keeps a cache pass over twenty-four segments to a few seconds.
LOADER_MODEL_KWARGS: Dict[str, Any] = dict(
    TINY_KWARGS,
    sequence_length=300,
    horizon=30,
    warmup_period=30,
    decoder_hidden=16,
    horizon_depth=1,
)

#: The known-answer fit. Full-batch over the eight fit segments, because a four-of-eight draw
#: makes the held-out curve a picture of the draw rather than of the fit; the learning rate is the
#: largest that does not send the log-variance head into the divergence a summed 64-sample
#: Gaussian score rewards.
KNOWN_ANSWER_STEPS = 200
KNOWN_ANSWER_BATCH = 8
KNOWN_ANSWER_LR = 5e-4


# =============================================================================
# The known-answer cache
# =============================================================================
def known_answer_cache(geometry: Any, d_model: int, *, seed: int = 0) -> oracle.StateCache:
    r"""Build a cache whose forecast block is a deterministic function of the encoder state.

    Each segment carries a constant raw level $c_i$, and its state is $b + c_i v$ for one fixed
    base direction $b$ and one fixed sensitive direction $v$. The map the probe has to learn is
    exact and one-dimensional, which is what makes "the probe drives the score below the trivial
    predictor" a statement about the probe rather than about the data.

    **The level is carried by the state's direction, not by its magnitude, and that is not a
    stylistic choice.** The production decoder's projection begins with a ``LayerNorm`` over the
    state's channels, so an encoding that scaled one channel by $c_i$ would be normalised away
    entirely -- the probe would be asked to recover a number the architecture has already
    discarded, and would fail at a problem nothing in this pipeline actually poses. A real encoder
    state carries the level across many channels, which is exactly what $b + c_i v$ reproduces.

    Two segments per recording, so the recording-level split has something to split.

    Args:
        geometry: The model's trimmed-grid geometry.
        d_model: Width of the encoder state.
        seed: Seed for the levels and the two directions.

    Returns:
        The cache, with every step valid so no anchor is masked out for a reason unrelated to
        what is being tested.
    """
    rng = np.random.default_rng(seed)
    n_recordings, per_recording = 8, 2
    n_segments = n_recordings * per_recording

    # Bounded rather than Gaussian: the state-to-level map is smooth but not linear once the
    # LayerNorm is in front of it, and a long tail would ask the probe to extrapolate.
    levels = rng.uniform(-1.0, 1.0, size=n_segments).astype(np.float32)
    base = rng.normal(size=d_model).astype(np.float32)
    direction = rng.normal(size=d_model).astype(np.float32)
    state = (
        base[None, None, :] + levels[:, None, None] * direction[None, None, :]
    ).astype(np.float32)
    state = np.broadcast_to(state, (n_segments, geometry.t_valid, d_model)).copy()

    return oracle.StateCache(
        target_state=torch.from_numpy(state),
        fhr_raw=torch.from_numpy(
            np.repeat(levels[:, None], geometry.raw_len, axis=1).astype(np.float32)
        ),
        weight=torch.ones((n_segments, geometry.t), dtype=torch.float32),
        guid=[f"REC{index // per_recording:02d}" for index in range(n_segments)],
        epoch=np.arange(n_segments, dtype=np.float64) * -600.0,
    )


def trivial_predictor_nats(cache: oracle.StateCache, geometry: Any) -> float:
    r"""Score the $\mu = 0, \sigma = 1$ predictor on the cache, in nats per anchor.

    The stand-in for $D_{\mathrm{base}}$ in the known-answer test: a decoder whose latent carried
    nothing about the level could do no better than this, so a probe that does not beat it has not
    read the state.

    Args:
        cache: The cache.
        geometry: The model's trimmed-grid geometry.

    Returns:
        The pooled score.
    """
    target = build_future_target(cache.fhr_raw, geometry)
    mask, _coverage = forecast_mask(cache.weight, geometry, coverage_floor=0.0)
    block, contributing = masked_raw_block_per_anchor(
        torch.zeros(()), target, mask, likelihood="gaussian_nll", logvar=torch.zeros(())
    )
    return float((block * contributing).sum() / contributing.sum().clamp_min(1.0))


def _fit(
    model: Any,
    cache: oracle.StateCache,
    *,
    shuffle_conditioning: bool = False,
    steps: int = KNOWN_ANSWER_STEPS,
    seed: int = 0,
) -> oracle.FitResult:
    """Fit a fresh probe on the known-answer cache and return its fit.

    Takes the *model*, not the task: nothing in the fit reaches the task, which is itself part of
    what is being asserted -- the probe reads a cache and nothing else.
    """
    fit_rows, held_out_rows = oracle.guid_split(cache.guid, seed=seed)
    torch.manual_seed(0)
    probe = oracle.build_probe(model, width_multiplier=1)
    return oracle.fit_probe(
        probe, cache, fit_rows, held_out_rows,
        geometry=model.geometry,
        future_index=model.future_index,
        coverage_floor=0.0,
        likelihood="gaussian_nll",
        device=torch.device("cpu"),
        steps=steps,
        batch_size=KNOWN_ANSWER_BATCH,
        learning_rate=KNOWN_ANSWER_LR,
        eval_every=max(1, steps // 8),
        seed=seed,
        shuffle_conditioning=shuffle_conditioning,
    )


@pytest.fixture(scope="module")
def tiny_model():
    """The tiny net, built once for this file.

    A model rather than a task, and module-scoped rather than per test: every assertion here is
    about the probe, and rebuilding the net for each one would spend more time constructing the
    thing not under test than fitting the thing that is.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**TINY_KWARGS)


@pytest.fixture(scope="module")
def known_answer(tiny_model):
    """The known-answer cache and the trivial predictor's score on it."""
    cache = known_answer_cache(tiny_model.geometry, tiny_model.d_model)
    return cache, trivial_predictor_nats(cache, tiny_model.geometry)


@pytest.fixture(scope="module")
def known_answer_fits(tiny_model, known_answer):
    """The matched fit and its shuffled-conditioning control, computed once.

    Two questions are asked of the same pair -- does the probe beat the trivial predictor, and
    does it beat a probe fitted on a stranger's state -- and fitting twice per question would
    double the only expensive thing in this file.
    """
    cache, _trivial = known_answer
    return _fit(tiny_model, cache), _fit(tiny_model, cache, shuffle_conditioning=True)


# =============================================================================
# Capacity mirroring
# =============================================================================
def test_the_probe_mirrors_the_production_decoder_rather_than_restating_its_shape(
    tiny_model,
) -> None:
    """Everything that decides capacity is read off the loaded model. A probe built from a second
    copy of those numbers would measure a different decoder from the one being evaluated."""
    model = tiny_model
    probe = oracle.build_probe(model)

    assert probe.core.d_hidden == model.horizon_core.d_hidden
    assert probe.core.horizon == model.horizon_core.horizon
    assert probe.core.film == model.horizon_core.film
    assert probe.core.film_per_block == model.horizon_core.film_per_block
    assert len(probe.core.refine.blocks) == len(model.horizon_core.refine.blocks)
    assert (
        probe.core.refine.blocks[0]["conv"].kernel_size
        == model.horizon_core.refine.blocks[0]["conv"].kernel_size
    )
    assert probe.core.attention_blocks == model.horizon_core.attention_blocks
    assert probe.core.attention_heads == model.horizon_core.attention_heads
    assert probe.out_channels == model.decoder.out_channels
    assert probe.logvar_clamp == model.logvar_clamp


def test_the_probe_carries_the_models_horizon_attention_rather_than_none() -> None:
    """The mirroring above is asserted on a model whose attention is off, where "mirrored" and
    "hardcoded to zero" are the same number. Built with the blocks on, the probe must have them
    -- a blockless probe would bound a decoder nobody trained, and would make the oracle gap read
    as bottleneck cost when part of it was missing capacity."""
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(TINY_KWARGS, horizon_attention_blocks=2))

    probe = oracle.build_probe(model)

    assert probe.core.attention is not None
    assert len(probe.core.attention) == 2
    # A fresh stack, not the model's: sharing it would hand the probe trained horizon dynamics.
    model_parameters = {id(parameter) for parameter in model.parameters()}
    assert not model_parameters & {id(p) for p in probe.core.attention.parameters()}


def test_the_probe_reads_the_encoder_state_rather_than_the_latent(tiny_model) -> None:
    """The one deliberate difference. Its input width is $d_{model}$, not $d_z$ -- which is the
    whole experiment: same decoder, no bottleneck."""
    model = tiny_model
    probe = oracle.build_probe(model)

    states = torch.zeros((2, model.geometry.t_valid, model.d_model))
    mu, logvar = probe(states)

    assert model.d_model != model.d_z, "the tiny geometry must not make the two widths equal"
    assert mu.shape == (2, model.geometry.t_valid, model.horizon, model.raw_per_step)
    assert logvar.shape == mu.shape


def test_a_wider_probe_is_strictly_larger_and_shares_nothing_with_the_model(tiny_model) -> None:
    """The capacity check needs a probe that genuinely has more capacity, and neither probe may
    share a parameter with the checkpoint -- a shared core would hand it trained horizon dynamics
    for free."""
    model = tiny_model
    narrow = oracle.build_probe(model, width_multiplier=1)
    wide = oracle.build_probe(model, width_multiplier=oracle.CAPACITY_WIDTH_MULTIPLIER)

    assert oracle.parameter_count(wide) > oracle.parameter_count(narrow)
    model_parameters = {id(parameter) for parameter in model.parameters()}
    for probe in (narrow, wide):
        assert not model_parameters & {id(parameter) for parameter in probe.parameters()}


# =============================================================================
# The recording-level split
# =============================================================================
def test_no_recording_appears_on_both_sides_of_the_split() -> None:
    """A segment-level split is not a split: one delivery contributes tens of segments whose
    forecast windows overlap in 29 of their 30 steps."""
    guids = [f"REC{index // 3:02d}" for index in range(30)]

    fit_rows, held_out_rows = oracle.guid_split(guids, seed=7)

    fit_guids = {guids[index] for index in fit_rows}
    held_out_guids = {guids[index] for index in held_out_rows}
    assert fit_guids and held_out_guids
    assert not fit_guids & held_out_guids
    assert len(fit_rows) + len(held_out_rows) == len(guids)


def test_the_split_is_reproducible_from_its_seed_and_moves_with_it() -> None:
    """Recorded in the summary as a seed, so it has to be recoverable from one."""
    guids = [f"REC{index // 2:02d}" for index in range(24)]

    first = oracle.guid_split(guids, seed=1)
    again = oracle.guid_split(guids, seed=1)
    other = oracle.guid_split(guids, seed=2)

    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[1], other[1])


def test_a_two_recording_population_still_splits_into_two_sides() -> None:
    """Rounding must never put every recording on one side: the held-out score would then be
    fitted on itself and the gap would be reported as larger than it is."""
    fit_rows, held_out_rows = oracle.guid_split(["A", "A", "B", "B"], seed=0)

    assert len(fit_rows) == 2 and len(held_out_rows) == 2


# =============================================================================
# The three validations of the fit
# =============================================================================
def test_the_probe_beats_its_shuffled_conditioning_control(known_answer_fits) -> None:
    """Conditioning ablation. Fitted on another recording's state, the probe can only learn the
    population mean, and the matched held-out score says so."""
    real, shuffled = known_answer_fits

    assert shuffled.shuffled_conditioning and not real.shuffled_conditioning
    assert real.final_held_out_nats < shuffled.final_held_out_nats - 1.0, (
        f"real {real.final_held_out_nats:.4g} vs shuffled "
        f"{shuffled.final_held_out_nats:.4g} nats/anchor"
    )


def test_the_known_answer_fixture_drives_the_probe_below_the_trivial_predictor(
    known_answer, known_answer_fits
) -> None:
    """Known answer. The block is an exact function of the state, so a probe that reads the state
    must beat the predictor that does not."""
    _cache, trivial = known_answer
    real, _shuffled = known_answer_fits

    assert real.final_held_out_nats < trivial


def test_the_probe_step_leaves_no_gradient_on_the_production_model(
    tiny_model, known_answer
) -> None:
    """Bypass isolation. The cached states are detached, so the optimizer has no path back into
    the checkpoint -- asserted after a real step rather than argued from the import graph."""
    cache, _trivial = known_answer

    _fit(tiny_model, cache, steps=2)

    assert all(parameter.grad is None for parameter in tiny_model.parameters())


def test_the_cache_carries_no_graph_back_to_the_model(known_answer) -> None:
    """The property the assertion above depends on, stated directly: a cache that kept its graph
    would make the isolation a matter of luck about which tensors happened to be used."""
    cache, _trivial = known_answer

    assert cache.target_state.grad_fn is None
    assert not cache.target_state.requires_grad


# =============================================================================
# Convergence and capacity, both mechanical
# =============================================================================
def _curve(values: List[float]) -> List[Dict[str, float]]:
    """Wrap held-out values as the curve the convergence rule reads."""
    return [
        {"step": float(index), "held_out_nats": value, "fit_nats": value}
        for index, value in enumerate(values)
    ]


def test_a_flattened_curve_counts_as_converged() -> None:
    converged, detail = oracle.assess_convergence(
        _curve([100.0, 40.0, 12.0, 10.2, 10.05, 10.0, 10.0, 10.0])
    )

    assert converged
    assert "final quarter" in detail


def test_a_curve_still_descending_at_its_end_flags_itself() -> None:
    """An under-trained probe understates the gap, so it must say so rather than report a small
    number as a finding about the model."""
    converged, detail = oracle.assess_convergence(_curve([100.0, 80.0, 60.0, 40.0, 20.0, 10.0]))

    assert not converged
    assert "final quarter" in detail


def test_a_probe_that_never_improved_is_not_called_converged() -> None:
    """Flat from the first evaluation is a probe that failed to start, not one that finished."""
    converged, detail = oracle.assess_convergence(_curve([10.0, 10.0, 10.0, 10.0]))

    assert not converged
    assert "did not fit" in detail


def _fit_result(score: float, *, width: int = 1, parameters: int = 100) -> oracle.FitResult:
    """A minimal fit result carrying only what the capacity verdict reads."""
    return oracle.FitResult(
        width_multiplier=width,
        n_parameters=parameters,
        steps=1,
        curve=[],
        final_held_out_nats=score,
        best_held_out_nats=score,
        best_step=1,
        converged=True,
        convergence_detail="",
    )


def test_the_capacity_verdict_fires_only_past_its_margin() -> None:
    """A capacity gap smaller than the margin the coupling readout itself is judged against cannot
    change how the sufficiency number is read."""
    inside = oracle.capacity_verdict(_fit_result(100.0), _fit_result(99.5, width=2))
    outside = oracle.capacity_verdict(_fit_result(100.0), _fit_result(90.0, width=2))

    assert inside["checked"] and outside["checked"]
    assert inside["capacity_bound"] is False
    assert outside["capacity_bound"] is True
    assert outside["improvement_nats"] == pytest.approx(10.0)
    assert inside["margin_nats"] == oracle.CAPACITY_MARGIN_NATS


def test_an_unrun_capacity_check_is_unmeasured_rather_than_adequate() -> None:
    """``None`` is the third state, and it must not read as "the probe was big enough"."""
    verdict = oracle.capacity_verdict(_fit_result(100.0), None)

    assert verdict["checked"] is False
    assert verdict["capacity_bound"] is None


# =============================================================================
# The budget, derived from the population rather than written down
# =============================================================================
def test_the_step_count_scales_with_the_fit_population() -> None:
    """A fixed step count that trains a probe on two thousand segments overfits one on twenty, and
    every fixture in this repository is the second case."""
    small = oracle.resolve_budget(12, epochs=20, batch_size=16)
    large = oracle.resolve_budget(2000, epochs=20, batch_size=16)

    # The batch never exceeds the population: a larger one draws the same segments repeatedly and
    # pays for every draw.
    assert small["batch_size"] == 12
    assert large["batch_size"] == 16
    assert small["steps"] < large["steps"]
    assert large["steps"] == pytest.approx(20 * 2000 / 16, rel=0.01)


def test_the_budget_is_bounded_at_both_ends() -> None:
    assert oracle.resolve_budget(1, epochs=1)["steps"] == oracle.MIN_FIT_STEPS
    assert oracle.resolve_budget(10**6, epochs=1000)["steps"] == oracle.MAX_FIT_STEPS


def test_an_explicit_step_count_overrides_the_derivation() -> None:
    assert oracle.resolve_budget(2000, steps=7)["steps"] == 7


# =============================================================================
# The cache: one encoder pass, and the fit does not make it two
# =============================================================================
def test_the_fit_reads_the_cache_rather_than_re_running_the_encoder(
    task, multi_class_loader
) -> None:
    """The stated amendment to the one-model-touching-pass rule is that the encoder runs *once*
    more, not once per step. Counted through a forward hook, because "the code looks like it
    caches" is not the property."""
    built = task(model_kwargs=LOADER_MODEL_KWARGS)
    model = built.orig_model
    forwards: List[int] = []
    handle = model.register_forward_hook(lambda *_args: forwards.append(1))
    try:
        cache = oracle.cache_target_states(built, multi_class_loader, cap=None, seed=0)
        after_cache = len(forwards)

        fit_rows, held_out_rows = oracle.guid_split(cache.guid, seed=0)
        torch.manual_seed(0)
        oracle.fit_probe(
            oracle.build_probe(model), cache, fit_rows, held_out_rows,
            geometry=model.geometry, future_index=model.future_index,
            coverage_floor=float(model.coverage_floor), likelihood="gaussian_nll",
            device=torch.device("cpu"), steps=3, batch_size=2, eval_every=3, seed=0,
        )
    finally:
        handle.remove()

    assert len(cache) == len(multi_class_loader.dataset)
    assert after_cache == int(np.ceil(len(cache) / int(multi_class_loader.batch_size)))
    assert len(forwards) == after_cache, "the fit re-ran the encoder instead of reading the cache"
    assert cache.target_state.shape[1:] == (model.geometry.t_valid, model.d_model)
    assert len(cache.guid) == len(cache) and cache.epoch.shape == (len(cache),)


def test_the_fit_leaves_the_global_random_state_where_it_found_it(
    task, multi_class_loader
) -> None:
    """The behavioural half of the package-wide ban on seeding by hand.

    The probe's initialisation runs on the global generators -- ``nn.init`` takes no generator --
    so the fit seeds them, inside ``torch.random.fork_rng``. What must hold is that nothing
    downstream sees a stream it would not otherwise have seen: an analysis after this one that
    draws from the global RNG (the per-sample pages do, through the model's own
    reparameterisation) would otherwise render different figures depending on whether the oracle
    had run.

    The comparison is against the state the **cache pass alone** leaves, not against the state
    before it: caching is an ordinary model forward and legitimately advances the stream, exactly
    as every other analysis's forward does.
    """
    built = task(model_kwargs=LOADER_MODEL_KWARGS)
    settings = dict(eval_config={"seed": 5, "caps": {}}, steps=2, curve_points=2)

    torch.manual_seed(123)
    oracle.cache_target_states(
        built, multi_class_loader, cap=None, seed=5 + oracle._SEED_OFFSET_CACHE
    )
    after_cache_only = torch.random.get_rng_state()

    torch.manual_seed(123)
    record = oracle.run_oracle(built, multi_class_loader, **settings)
    after_the_whole_run = torch.random.get_rng_state()

    assert not record["skipped"], record.get("reason")
    assert torch.equal(after_the_whole_run, after_cache_only)


def test_run_oracle_reports_the_split_the_capacity_check_and_both_biases(
    task, multi_class_loader
) -> None:
    """The record a caller turns into a summary block, end to end over a real loader."""
    built = task(model_kwargs=LOADER_MODEL_KWARGS)

    record = oracle.run_oracle(
        built, multi_class_loader, eval_config={"seed": 5, "caps": {}}, steps=2, curve_points=2
    )

    split = record["split"]
    assert split["recordings_disjoint"] is True
    assert split["n_fit_recordings"] and split["n_held_out_recordings"]
    assert record["capacity"]["checked"] is True
    assert {entry["direction"] for entry in record["bias_directions"]} == {
        "understates", "overstates"
    }
    per_segment = record["per_segment"]
    assert len(per_segment["guid"]) == split["n_held_out_segments"]
    assert np.isfinite(per_segment["nll_oracle_block"]).any()


def test_a_capped_cache_draws_over_the_whole_split_rather_than_a_prefix(
    task, multi_class_loader
) -> None:
    """The loader is concatenated per-subgroup files, so a prefix cap is one subgroup and one
    clinical class -- the failure mode every cap in this pipeline is drawn to avoid."""
    built = task(model_kwargs=LOADER_MODEL_KWARGS)
    total = len(multi_class_loader.dataset)

    capped = oracle.cache_target_states(built, multi_class_loader, cap=8, seed=0)

    assert 8 < total, "the cap has to be below the population for this to test anything"
    assert len(capped) == 8
    # More than one recording, and not merely the first eight rows of the dataset.
    assert len(set(capped.guid)) > 1
