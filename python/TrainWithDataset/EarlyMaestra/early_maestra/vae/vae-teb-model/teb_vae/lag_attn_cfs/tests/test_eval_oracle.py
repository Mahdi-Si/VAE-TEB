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

**Three things this file asserts that the raw sibling's cannot**, because they are what the target
domain changed:

* the probe's output width is $C_{\mathrm{keep}}$ -- the surviving-channel count the warm-up budget
  resolved -- rather than $R$ raw samples per horizon step;
* the target is built by the **model's own** :meth:`_build_forecast_target` at the anchor set the
  forward returned, and the mask by the anchored ``forecast_mask``, so the probe is scored over the
  anchors the collection pass scored the two model branches over. ``model.future_index`` is not
  read at all;
* the cached anchor set is the dense one, and the conditioning state is gathered onto it -- so a
  fit at this cell's *training* tiling, which decodes a tenth as many anchors, would be a different
  population and is not what the cache holds.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import oracle
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

from .conftest import build, shipped_warmup_kwargs, tiny_warmup_kwargs

#: The model geometry the loader-driven tests build: the shipped window, floor, horizon and
#: warm-up budget with only the widths shrunk. The geometry has to be the real one -- the generated
#: shards are $300$ decimated steps and the decoder's width **is** the resolved budget's surviving
#: channel count -- so what shrinks is $d_{\mathrm{model}}$, the decoder hidden width and the
#: horizon depth, which is what keeps a cache pass over forty-eight segments to a few seconds.
def _loader_model_kwargs() -> Dict[str, Any]:
    """Return the loader-shaped constructor kwargs, resolved against the committed budget."""
    return shipped_warmup_kwargs(
        d_model=32,
        d_z=8,
        d_head=8,
        max_lag=8,
        lstm_layers=1,
        dropout=0.0,
        decoder_hidden=16,
        horizon_depth=1,
        horizon_attention_blocks=0,
        encoder_extra_dilations=(),
    )


#: The known-answer fit. Full-batch over the eight fit segments, because a four-of-eight draw
#: makes the held-out curve a picture of the draw rather than of the fit; the learning rate is the
#: largest that does not send the log-variance head into the divergence a summed block Gaussian
#: score rewards.
KNOWN_ANSWER_STEPS = 200
KNOWN_ANSWER_BATCH = 8
KNOWN_ANSWER_LR = 5e-4

#: How many segments the loader-driven ``run_oracle`` calls cache. Below the split's forty-eight,
#: so the cap is exercised, and comfortably above the four recordings the split needs.
LOADER_CACHE_CAP = 16


# =============================================================================
# The known-answer cache
# =============================================================================
def known_answer_cache(model: Any, *, seed: int = 0) -> oracle.StateCache:
    r"""Build a cache whose forecast block is a deterministic function of the encoder state.

    Each segment carries a constant coefficient level $c_i$ on every declared target channel, and
    its state is $b + c_i v$ for one fixed base direction $b$ and one fixed sensitive direction
    $v$. The map the probe has to learn is exact and one-dimensional, which is what makes "the
    probe drives the score below the trivial predictor" a statement about the probe rather than
    about the data.

    **The level is carried by the state's direction, not by its magnitude, and that is not a
    stylistic choice.** The production decoder's projection begins with a ``LayerNorm`` over the
    state's channels, so an encoding that scaled one channel by $c_i$ would be normalised away
    entirely -- the probe would be asked to recover a number the architecture has already
    discarded, and would fail at a problem nothing in this pipeline actually poses. A real encoder
    state carries the level across many channels, which is exactly what $b + c_i v$ reproduces.

    The anchor set is the model's own at :data:`DENSE_ANCHOR_GEOMETRY`, not an ``arange``: it is
    what the forward returns and what the cache is required to carry, and a hand-built one could
    disagree with it in exactly the way the cached field exists to prevent.

    Two segments per recording, so the recording-level split has something to split.

    Args:
        model: The net, for its geometry, its declared width and its anchor set.
        seed: Seed for the levels and the two directions.

    Returns:
        The cache, with every step valid so no anchor is masked out for a reason unrelated to
        what is being tested.
    """
    rng = np.random.default_rng(seed)
    n_recordings, per_recording = 8, 2
    n_segments = n_recordings * per_recording
    geometry, d_model = model.geometry, int(model.d_model)

    # Bounded rather than Gaussian: the state-to-level map is smooth but not linear once the
    # LayerNorm is in front of it, and a long tail would ask the probe to extrapolate.
    levels = rng.uniform(-1.0, 1.0, size=n_segments).astype(np.float32)
    base = rng.normal(size=d_model).astype(np.float32)
    direction = rng.normal(size=d_model).astype(np.float32)
    state = (
        base[None, None, :] + levels[:, None, None] * direction[None, None, :]
    ).astype(np.float32)
    state = np.broadcast_to(state, (n_segments, geometry.t_valid, d_model)).copy()

    anchor_phase, anchor_stride = DENSE_ANCHOR_GEOMETRY
    anchors, anchor_valid = model._build_anchor_index(
        batch=n_segments,
        device=torch.device("cpu"),
        anchor_phase=anchor_phase,
        anchor_stride=anchor_stride,
    )
    features = np.broadcast_to(
        levels[:, None, None], (n_segments, geometry.t, int(model.c_y))
    ).astype(np.float32)

    return oracle.StateCache(
        target_state=torch.from_numpy(state),
        target_features=torch.from_numpy(features.copy()),
        weight=torch.ones((n_segments, geometry.t), dtype=torch.float32),
        anchor_index=anchors,
        anchor_valid=anchor_valid,
        guid=[f"REC{index // per_recording:02d}" for index in range(n_segments)],
        epoch=np.arange(n_segments, dtype=np.float64) * -600.0,
    )


def trivial_predictor_nats(cache: oracle.StateCache, model: Any) -> float:
    r"""Score the $\mu = 0, \sigma = 1$ predictor on the cache, in nats per anchor.

    The stand-in for $D_{\mathrm{base}}$ in the known-answer test: a decoder whose latent carried
    nothing about the level could do no better than this, so a probe that does not beat it has not
    read the state.

    Built through the model's own gather and the anchored mask, which is what the probe is scored
    through -- a second construction of either would make the comparison a comparison of two
    denominators.

    Args:
        cache: The cache.
        model: The net.

    Returns:
        The pooled score.
    """
    target = model._build_forecast_target(cache.target_features, cache.anchor_index)
    mask, _coverage = forecast_mask(
        cache.weight,
        model.geometry,
        coverage_floor=float(model.coverage_floor),
        anchors=cache.anchor_index,
        anchor_valid=cache.anchor_valid,
    )
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
    what is being asserted -- the probe reads a cache and the model's own target gather, and
    nothing else.
    """
    fit_rows, held_out_rows = oracle.guid_split(cache.guid, seed=seed)
    torch.manual_seed(0)
    probe = oracle.build_probe(model, width_multiplier=1)
    return oracle.fit_probe(
        probe, cache, fit_rows, held_out_rows,
        model=model,
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
    """The tiny **guarded** net, built once for this file.

    Guarded rather than ungated, because the guard is what this cell's oracle is about: the probe's
    output width is the budget's surviving-channel count, and an ungated model would make that
    assertion a statement about $c_y$ instead.

    A model rather than a task, and module-scoped rather than per test: every assertion here is
    about the probe, and rebuilding the net for each one would spend more time constructing the
    thing not under test than fitting the thing that is.
    """
    return build(tiny_warmup_kwargs())


@pytest.fixture(scope="module")
def known_answer(tiny_model):
    """The known-answer cache and the trivial predictor's score on it."""
    cache = known_answer_cache(tiny_model)
    return cache, trivial_predictor_nats(cache, tiny_model)


@pytest.fixture(scope="module")
def known_answer_fits(tiny_model, known_answer):
    """The matched fit and its shuffled-conditioning control, computed once.

    Two questions are asked of the same pair -- does the probe beat the trivial predictor, and
    does it beat a probe fitted on a stranger's state -- and fitting twice per question would
    double the only expensive thing in this file.
    """
    cache, _trivial = known_answer
    return _fit(tiny_model, cache), _fit(tiny_model, cache, shuffle_conditioning=True)


@pytest.fixture(scope="module")
def loader_task():
    """A task at the loader-shaped geometry, built once for the four loader-driven tests."""
    from teb_vae.lag_attn_cfs.tests.conftest import make_task

    module = make_task(model_kwargs=_loader_model_kwargs())
    module.eval()
    return module


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


def test_the_probes_width_is_the_budgets_surviving_channel_count(tiny_model) -> None:
    """The target domain's own assertion, and the one a copy of the raw probe would fail silently.
    The decoder emits one value per **surviving** target channel, so a probe built at $c_y$ would
    forecast channels the budget dropped -- a wider block, a larger nats figure, and a
    $\\Delta_{\\mathrm{suff}}$ that is not a difference against ``nll_base_block`` at all."""
    model = tiny_model
    probe = oracle.build_probe(model)

    assert model.target_gate is not None, "an ungated model would make this vacuous"
    kept = int(model.target_gate.out_channels)
    assert kept < int(model.c_y), "the tiny budget must actually drop channels"
    assert probe.out_channels == kept


def test_the_probe_emits_the_anchor_axis_it_is_handed(tiny_model) -> None:
    r"""The one deliberate difference from the model's own decoder: its input width is
    $d_{\mathrm{model}}$, not $d_z$ -- same decoder, no bottleneck -- and its first axis is the
    **decoded anchor set** rather than the dense $[0, T_{\mathrm{valid}})$ prefix."""
    model = tiny_model
    probe = oracle.build_probe(model)
    n_anchors = int(model.geometry.t_valid - model.warmup_period)

    states = torch.zeros((2, n_anchors, model.d_model))
    mu, logvar = probe(states)

    assert model.d_model != model.d_z, "the tiny geometry must not make the two widths equal"
    assert mu.shape == (2, n_anchors, model.horizon, model.target_gate.out_channels)
    assert logvar.shape == mu.shape


def test_the_probe_carries_the_models_horizon_attention_rather_than_none() -> None:
    """The mirroring above is asserted on a model whose attention is off, where "mirrored" and
    "hardcoded to zero" are the same number. Built with the blocks on, the probe must have them
    -- a blockless probe would bound a decoder nobody trained, and would make the oracle gap read
    as bottleneck cost when part of it was missing capacity."""
    model = build(tiny_warmup_kwargs(horizon_attention_blocks=2))

    probe = oracle.build_probe(model)

    assert probe.core.attention is not None
    assert len(probe.core.attention) == 2
    # A fresh stack, not the model's: sharing it would hand the probe trained horizon dynamics.
    model_parameters = {id(parameter) for parameter in model.parameters()}
    assert not model_parameters & {id(p) for p in probe.core.attention.parameters()}


def test_the_probe_is_the_same_capacity_as_the_decoder_within_one_projection_layer(
    tiny_model,
) -> None:
    r""""The same decoder at the same capacity" is measured here rather than claimed.

    The one place the two genuinely differ is the input projection -- the production decoder reads
    $z$ at width $d_z$ and the probe reads the encoder state at $d_{\mathrm{model}}$, which *is*
    the experiment -- so the counts cannot be equal. What must hold is that the difference is
    exactly that layer and nothing else: a probe missing the horizon attention, or built at half
    the hidden width, would sit far outside this tolerance and would report the bottleneck's cost
    plus its own missing capacity as one number.

    The tolerance is stated as a fraction rather than derived, because deriving it would restate
    ``ResidualMLP``'s own layer schedule here and the two copies could disagree; a quarter is loose
    enough to admit the projection at every shipped width and far tighter than any missing block.
    """
    model = tiny_model
    probe = oracle.build_probe(model)

    probe_parameters = oracle.parameter_count(probe)
    decoder_parameters = oracle.parameter_count(model.decoder)

    assert model.d_model > model.d_z, "the probe must be the wider of the two, or this is vacuous"
    assert probe_parameters > decoder_parameters
    assert probe_parameters <= decoder_parameters * 1.25, (
        f"probe {probe_parameters} against decoder {decoder_parameters}: the two differ by more "
        f"than the input projection, so the probe is not the same decoder at the same capacity"
    )


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
# The target and the mask are the model's own, at the anchors it decoded
# =============================================================================
def test_the_batch_is_assembled_at_the_cached_anchors_through_the_models_own_gather(
    tiny_model, known_answer
) -> None:
    r"""The load-bearing difference from the sibling's oracle. Three shapes, one identity:

    the conditioning is gathered onto the anchor axis, the target is
    $(B, A_{\max}, H, C_{\mathrm{keep}})$ from the model's own ``_build_forecast_target``, and the
    mask is the anchored one. A target built over the dense prefix instead would have the right
    rank, the right dtype and the wrong rows -- and on this cell $A_{\max}$ and
    $T_{\mathrm{valid}}$ differ, which is what makes the shape check able to see it at all.
    """
    cache, _trivial = known_answer
    model = tiny_model
    rows = torch.arange(3, dtype=torch.long)

    states, target, mask = oracle._batch_tensors(
        model, cache, rows, rows, device=torch.device("cpu")
    )

    n_anchors = int(cache.anchor_index.shape[1])
    assert n_anchors < int(model.geometry.t_valid), "otherwise the shape proves nothing"
    assert states.shape == (3, n_anchors, int(model.d_model))
    assert target.shape == (
        3, n_anchors, int(model.horizon), int(model.target_gate.out_channels)
    )
    assert mask.shape == (3, n_anchors, int(model.horizon))
    # The state at anchor position $a$ is the state at decimated step ``anchor_index[a]``.
    expected = cache.target_state[rows].gather(
        1, cache.anchor_index[rows][:, :, None].expand(-1, -1, int(model.d_model))
    )
    assert torch.equal(states, expected)


def test_the_conditioning_ablation_moves_the_state_and_leaves_the_target_alone(
    tiny_model, known_answer
) -> None:
    """What makes the shuffled control a control. It pairs a segment's target with another row's
    state; if it also gathered that row's *anchors*, it would score one segment's coefficients
    against another segment's window and the comparison would be of two different things."""
    cache, _trivial = known_answer
    model = tiny_model
    targets = torch.tensor([0, 1], dtype=torch.long)
    strangers = torch.tensor([4, 5], dtype=torch.long)

    matched_states, matched_target, matched_mask = oracle._batch_tensors(
        model, cache, targets, targets, device=torch.device("cpu")
    )
    shuffled_states, shuffled_target, shuffled_mask = oracle._batch_tensors(
        model, cache, targets, strangers, device=torch.device("cpu")
    )

    assert torch.equal(matched_target, shuffled_target)
    assert torch.equal(matched_mask, shuffled_mask)
    assert not torch.equal(matched_states, shuffled_states)


def test_nothing_in_the_oracle_reads_the_raw_index_grid() -> None:
    """``model.future_index`` indexes a $4\\,$Hz raw grid this target does not have. It is the one
    symbol a mechanical copy of the sibling's module would have carried through, and it would have
    built a target of raw samples that this decoder never emits.

    Walked as an AST rather than searched as text: the module's own docstring names both symbols in
    prose -- which is the opposite of reaching for them -- and a substring scan cannot tell the
    two apart.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(oracle.__file__).read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "future_index" not in attributes | names
    assert "build_future_target" not in names | imported
    # And the gather it is replaced by is genuinely reached for, so this is not two absences.
    assert "_build_forecast_target" in attributes
    assert "forecast_mask" in imported


# =============================================================================
# The recording-level split
# =============================================================================
def test_no_recording_appears_on_both_sides_of_the_split() -> None:
    """A segment-level split is not a split: one delivery contributes tens of segments whose
    forecast windows overlap in $H - 1$ of their $H$ steps."""
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


def test_an_overlapping_split_is_refused_at_runtime_rather_than_only_in_a_test(
    loader_task, cohort_loader, monkeypatch
) -> None:
    """The split is the only thing standing between $\\Delta_{\\mathrm{suff}}$ and a probe scored
    on what it was fitted to, and it is drawn by an ordinary function. So the disjointness is
    checked inside the run, on the split it actually drew -- not merely asserted here on a
    synthetic one, which would prove nothing about a future edit to the draw."""
    def _overlapping(guids, *, seed, held_out_fraction=0.5):
        positions = np.arange(len(guids), dtype=np.int64)
        return positions, positions

    monkeypatch.setattr(oracle, "guid_split", _overlapping)

    with pytest.raises(ValueError, match="not a split"):
        oracle.run_oracle(
            loader_task, cohort_loader,
            eval_config={"seed": 5, "caps": {"oracle": LOADER_CACHE_CAP}},
            steps=1, curve_points=1, capacity_check=False,
        )


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
    """Bypass isolation. The cached states are detached and the target gather runs under
    ``no_grad``, so the optimizer has no path back into the checkpoint -- asserted after a real
    step rather than argued from the import graph.

    This matters more here than in the sibling: the target is built by a **method of the model**,
    so an implementation that built it inside the graph would give the probe's backward pass a
    route into the channel gate."""
    cache, _trivial = known_answer

    _fit(tiny_model, cache, steps=2)

    assert all(parameter.grad is None for parameter in tiny_model.parameters())


def test_the_reported_score_is_the_final_state_over_the_whole_held_out_half(
    tiny_model, known_answer, monkeypatch
) -> None:
    r"""$D_{\mathrm{oracle}}$ is measured once, at the end, over **everything** held out.

    The curve's points are a fixed *subsample* of each side, so they are comparable with each other
    and not with the number the summary reports. Reading the last one instead would put
    $D_{\mathrm{oracle}}$ on a different population from the ``mc_nll_base_block`` it is
    subtracted from -- and reading the *best* one would select the step by the held-out score it is
    about to report, fitting the held-out half through the back door.

    The subsample is shrunk here so the two populations genuinely differ: the known-answer cache
    holds eight held-out segments, fewer than :data:`CURVE_SAMPLE_SEGMENTS`, and at that size the
    curve is measured over the whole half and the distinction is invisible.
    """
    cache, _trivial = known_answer
    model = tiny_model
    monkeypatch.setattr(oracle, "CURVE_SAMPLE_SEGMENTS", 2)
    fit_rows, held_out_rows = oracle.guid_split(cache.guid, seed=0)

    torch.manual_seed(0)
    probe = oracle.build_probe(model, width_multiplier=1)
    fit = oracle.fit_probe(
        probe, cache, fit_rows, held_out_rows,
        model=model, likelihood="gaussian_nll", device=torch.device("cpu"),
        steps=12, batch_size=KNOWN_ANSWER_BATCH, learning_rate=KNOWN_ANSWER_LR,
        eval_every=3, seed=0,
    )

    scores, anchors = oracle.score_rows(
        probe, cache, held_out_rows, model=model, likelihood="gaussian_nll",
        device=torch.device("cpu"), batch_size=KNOWN_ANSWER_BATCH,
    )
    whole_half = float(np.sum(scores * anchors) / np.sum(anchors))

    assert len(oracle._curve_sample(held_out_rows, seed=1)) < len(held_out_rows)
    assert fit.final_held_out_nats == pytest.approx(whole_half, rel=1e-9)
    # The two are genuinely different numbers here, which is what makes the assertion above a
    # statement about which population was scored rather than a tautology.
    assert fit.final_held_out_nats != pytest.approx(
        float(fit.curve[-1]["held_out_nats"]), rel=1e-9
    )


def test_the_cache_carries_no_graph_back_to_the_model(known_answer) -> None:
    """The property the assertion above depends on, stated directly: a cache that kept its graph
    would make the isolation a matter of luck about which tensors happened to be used."""
    cache, _trivial = known_answer

    for tensor in (cache.target_state, cache.target_features, cache.weight):
        assert tensor.grad_fn is None
        assert not tensor.requires_grad


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
# The activation ceiling: the budget this cell needs and the raw one does not
# =============================================================================
def test_the_batch_falls_to_the_activation_ceiling_and_the_passes_do_not(tiny_model) -> None:
    r"""A fit step keeps a $(B \cdot A_{\max}, H, d_{\mathrm{hidden}})$ tensor alive per refine and
    attention block, so what a step costs is that product rather than the segment count the batch
    names -- and at this cell's shipped decoder it is large enough to matter.

    The lowered batch must cost **steps, not passes**: the budget is expressed in passes over the
    fit half, so a batch cut by four buys four times the steps and the probe sees the same data.
    An implementation that lowered the batch and left the step count alone would quietly convert a
    memory bound into an under-trained probe -- the one failure this module exists to report rather
    than absorb.
    """
    per_segment = oracle.activation_elements_per_segment(tiny_model)
    huge = oracle.MAX_FIT_ACTIVATION_ELEMENTS  # one segment fills the whole ceiling

    unbounded = oracle.resolve_budget(240, epochs=20, activation_per_segment=1)
    bounded = oracle.resolve_budget(240, epochs=20, activation_per_segment=huge)

    assert per_segment > 0
    assert unbounded["batch_size"] == oracle.DEFAULT_FIT_BATCH_SIZE
    assert unbounded["batch_size_source"] == "population"
    assert bounded["batch_size"] == 1
    assert bounded["batch_size_source"] == "activation ceiling"
    # Passes preserved: sixteen times the steps at a sixteenth of the batch.
    assert bounded["steps"] == unbounded["steps"] * oracle.DEFAULT_FIT_BATCH_SIZE
    assert bounded["activation_elements"] <= oracle.MAX_FIT_ACTIVATION_ELEMENTS


def test_the_step_ceiling_is_on_work_rather_than_on_iterations() -> None:
    """Stated as :data:`MAX_FIT_STEPS` steps *at the default batch* and applied as a bound on
    segment-visits. A ceiling on iterations alone would let the activation clamp above silently
    halve the passes a production run's probe gets."""
    at_default = oracle.resolve_budget(10**6, epochs=1000, activation_per_segment=1)
    at_a_quarter = oracle.resolve_budget(
        10**6, epochs=1000, activation_per_segment=oracle.MAX_FIT_ACTIVATION_ELEMENTS // 4
    )

    assert at_default["steps"] == oracle.MAX_FIT_STEPS
    assert at_a_quarter["batch_size"] == 4
    assert at_a_quarter["steps"] == oracle.MAX_FIT_SEGMENT_VISITS // 4
    # The same work either way, which is what makes the two ceilings one ceiling.
    for budget in (at_default, at_a_quarter):
        assert budget["steps"] * budget["batch_size"] == oracle.MAX_FIT_SEGMENT_VISITS


def test_the_widest_probe_is_what_the_batch_is_sized_for(tiny_model) -> None:
    """One batch for both fits, sized for the doubled-width refit -- so the capacity comparison
    stays a comparison of widths rather than of step counts, and the bound holds for the fit that
    actually allocates the most."""
    narrow = oracle.activation_elements_per_segment(tiny_model)
    wide = oracle.activation_elements_per_segment(
        tiny_model, width_multiplier=oracle.CAPACITY_WIDTH_MULTIPLIER
    )

    assert wide == narrow * oracle.CAPACITY_WIDTH_MULTIPLIER
    # And it is the model's own geometry, not a restatement: anchors times horizon times width.
    anchors = int(tiny_model.geometry.t_valid) - int(tiny_model.warmup_period)
    assert narrow == anchors * int(tiny_model.horizon) * int(tiny_model.horizon_core.d_hidden)


# =============================================================================
# The cache: one encoder pass, and the fit does not make it two
# =============================================================================
def test_the_fit_reads_the_cache_rather_than_re_running_the_encoder(
    loader_task, cohort_loader
) -> None:
    """The stated amendment to the one-model-touching-pass rule is that the encoder runs *once*
    more, not once per step. Counted through a forward hook, because "the code looks like it
    caches" is not the property."""
    model = loader_task.orig_model
    forwards: List[int] = []
    handle = model.register_forward_hook(lambda *_args: forwards.append(1))
    try:
        cache = oracle.cache_target_states(loader_task, cohort_loader, cap=None, seed=0)
        after_cache = len(forwards)

        fit_rows, held_out_rows = oracle.guid_split(cache.guid, seed=0)
        torch.manual_seed(0)
        oracle.fit_probe(
            oracle.build_probe(model), cache, fit_rows, held_out_rows,
            model=model, likelihood="gaussian_nll",
            device=torch.device("cpu"), steps=2, batch_size=2, eval_every=2, seed=0,
        )
    finally:
        handle.remove()

    assert len(cache) == len(cohort_loader.dataset)
    assert after_cache == int(np.ceil(len(cache) / int(cohort_loader.batch_size)))
    assert len(forwards) == after_cache, "the fit re-ran the encoder instead of reading the cache"
    assert cache.target_state.shape[1:] == (model.geometry.t_valid, model.d_model)
    assert cache.target_features.shape[1:] == (model.geometry.t, model.c_y)
    assert len(cache.guid) == len(cache) and cache.epoch.shape == (len(cache),)


def test_the_cached_anchor_set_is_the_dense_one_the_collection_pass_scored(
    loader_task, cohort_loader
) -> None:
    r"""The population the sufficiency gap is a gap over. This cell's model tiles at
    ``anchor_stride = H`` in training, so a cache built at the model's configured stride would
    hold a tenth of the anchors and $D_{\mathrm{oracle}}$ would be a score over a different set
    from the ``nll_base_block`` it is subtracted from."""
    model = loader_task.orig_model
    cache = oracle.cache_target_states(
        loader_task, cohort_loader, cap=LOADER_CACHE_CAP, seed=0
    )

    expected = int(model.geometry.t_valid - model.warmup_period)
    assert int(model.anchor_stride) > 1, "otherwise dense and tiled would be the same set"
    assert cache.anchor_index.shape == (len(cache), expected)
    assert bool(cache.anchor_valid.all()), "the dense set has no padding by construction"
    assert int(cache.anchor_index[0, 0]) == int(model.warmup_period)


def test_the_fit_leaves_the_global_random_state_where_it_found_it(
    loader_task, cohort_loader
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
    settings = dict(
        eval_config={"seed": 5, "caps": {"oracle": LOADER_CACHE_CAP}},
        steps=2,
        curve_points=2,
        capacity_check=False,
    )

    torch.manual_seed(123)
    oracle.cache_target_states(
        loader_task, cohort_loader,
        cap=LOADER_CACHE_CAP, seed=5 + oracle._SEED_OFFSET_CACHE,
    )
    after_cache_only = torch.random.get_rng_state()

    torch.manual_seed(123)
    record = oracle.run_oracle(loader_task, cohort_loader, **settings)
    after_the_whole_run = torch.random.get_rng_state()

    assert not record["skipped"], record.get("reason")
    assert torch.equal(after_the_whole_run, after_cache_only)


def test_run_oracle_reports_the_split_the_capacity_check_and_both_biases(
    loader_task, cohort_loader
) -> None:
    """The record a caller turns into a summary block, end to end over a real loader."""
    record = oracle.run_oracle(
        loader_task, cohort_loader,
        eval_config={"seed": 5, "caps": {"oracle": LOADER_CACHE_CAP}},
        steps=2, curve_points=2,
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


def test_the_record_states_the_geometry_and_the_block_it_scored_over(
    loader_task, cohort_loader
) -> None:
    """A nats-per-anchor figure from this cell is a sum over $H \\cdot C_{\\mathrm{keep}}$
    coefficients whose $C_{\\mathrm{keep}}$ the warm-up budget decided, at an anchor set that is
    not the one training tiles at. Both facts are budget- and arm-local, so a record that stated
    neither could not be read against another arm's -- or against the training CSV."""
    model = loader_task.orig_model

    record = oracle.run_oracle(
        loader_task, cohort_loader,
        eval_config={"seed": 5, "caps": {"oracle": LOADER_CACHE_CAP}},
        steps=2, curve_points=2, capacity_check=False,
    )

    geometry = record["anchor_geometry"]
    assert (geometry["anchor_phase"], geometry["anchor_stride"]) == DENSE_ANCHOR_GEOMETRY
    assert geometry["training_stride"] == int(model.anchor_stride)
    assert record["block_width"] == int(model.decoder.out_channels)
    assert "budget-local" in record["block_convention"]

    # And the capacity mirror as a measurement in the artifact, so a reader of ``summary.json``
    # can see how far "the same decoder at the same capacity" is from literal.
    mirror = record["capacity_mirror"]
    assert mirror["decoder_parameters"] == oracle.parameter_count(model.decoder)
    assert mirror["probe_parameters"] > mirror["decoder_parameters"]
    assert (mirror["d_model"], mirror["d_z"]) == (int(model.d_model), int(model.d_z))


def test_a_capped_cache_draws_over_the_whole_split_rather_than_a_prefix(
    loader_task, cohort_loader
) -> None:
    """The loader is concatenated per-subgroup files, so a prefix cap is one subgroup and one
    clinical class -- the failure mode every cap in this pipeline is drawn to avoid."""
    total = len(cohort_loader.dataset)

    capped = oracle.cache_target_states(loader_task, cohort_loader, cap=8, seed=0)

    assert 8 < total, "the cap has to be below the population for this to test anything"
    assert len(capped) == 8
    # More than one recording, and not merely the first eight rows of the dataset.
    assert len(set(capped.guid)) > 1
