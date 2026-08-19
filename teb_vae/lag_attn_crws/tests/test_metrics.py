r"""The three kept readouts, and the source-null floor read beside them.

The causal-feature cells report eight added readouts. **Five of them partition kept target
channels**, which this target does not have -- its block's last axis counts raw samples, which have
no warm-up to be past and no filter to rank -- so they are dropped rather than re-pointed, and
``test_raw_target.py`` asserts their absence. Three transfer unchanged because each is about the
*source* stream or the *anchor set*, and neither notion moves with the target:

* ``anchors_per_sample`` -- a **guard**, not a result. It must sit at its geometry-derived value, and
  a row outside that band means the tiling broke rather than that the model learned something.
* ``source_lag_warmth_frac_st`` / ``_ph`` -- the share of attention mass landing on lags where a
  source block is warm, per stored block. This sizes the compromise the design makes on the source:
  every channel is kept rather than gated, so the residual is measured instead of resolved and a
  *small* value is the expected finding rather than a failure.

The warmth columns are pinned at **both extremes**, because a range check alone passes on an
inverted metric: an all-zero warm-up must read exactly $1.0$ and a warm-up beyond every reachable lag
exactly $0.0$. Between them one fraction is recomputed here by an explicit sum over
$(b, a, m, \ell)$, which is what pins the per-block split rather than merely the pooled figure.

``kld_source_null`` is the fourth quantity read off this model, and it is a control rather than a
readout: it is not merged into ``compute_loss`` at all, because it costs a second encode and belongs
on the evaluation stages only. It is exercised here against the free function that computes it. The
property that makes it a *different* control from the permutation arm is asserted directly -- the
source availability pattern is a deterministic function of $t$ and identical in every row of the
batch, so no derangement of rows can remove it, and this arm does not move under one.

Every KL assertion runs after ``perturb_posterior``. The posterior delta heads are zero-initialised,
so on a fresh model the posterior *is* the prior, every divergence is identically zero, and both
directions of the non-vacuity pair below would hold on a model that ignores its source entirely.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective

from .conftest import (
    BATCH,
    CAUSAL_C_U,
    SHIPPED_HORIZON,
    TINY_SEQ_LEN,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    make_stub_batch,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: The two per-block source-warmth columns, named once.
_WARMTH_KEYS = ("source_lag_warmth_frac_st", "source_lag_warmth_frac_ph")

#: Everything ``compute_loss`` merges onto the shared objective's dict.
_ADDED_METRIC_KEYS = ("anchors_per_sample", *_WARMTH_KEYS)

#: The two phases the tiled fixtures run at: the second row is one anchor short of the first, which
#: is the only way a padded slot exists at all.
_PHASES = (0, TINY_STRIDE - 1)


def _weight(model, batch: int = BATCH, value: float = 1.0) -> torch.Tensor:
    """A uniform decimated weight at a model's own sequence length."""
    return torch.full((batch, model.geometry.t), float(value))


def _tiled(stride: int = TINY_STRIDE, **overrides):
    """The tiny model at a tiling, its three input tensors, and a seeded raw target signal."""
    kwargs = tiny_warmup_kwargs(anchor_stride=stride, **overrides)
    model = build(kwargs).eval()
    return model, make_streams(kwargs), make_raw_signal(kwargs)


def _metrics(model, streams, signal, phase, *, weight=None, likelihood="mse"):
    """One forward and its full metric dict."""
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*streams, phase)
    weight = _weight(model) if weight is None else weight
    return out, model.compute_loss(
        out, signal, weight=weight, likelihood=likelihood
    )["metrics"]


def _anchor_count(model, phase, stride, batch: int) -> float:
    r"""``anchors_per_sample`` alone, over a synthetic forward at a real geometry.

    The guard is a function of the decoded anchor set and of nothing the network computes, so it can
    be exercised at the **production** geometry without paying for a production forward -- which is
    what makes it testable at the batch width its shipped value is a mean over.

    Args:
        model: The model whose anchor geometry is read.
        phase: $\varphi$ per sample, or ``None`` at stride $1$.
        stride: $S$.
        batch: Samples the anchor set is built for.

    Returns:
        The reported count.
    """
    anchors, valid = model._build_anchor_index(batch, torch.device("cpu"), phase, stride)
    return float(
        model._anchors_per_sample(
            {"anchor_index": anchors, "anchor_valid": valid}, torch.zeros(1)
        )
    )


def _hand_warmth(model, out) -> dict:
    r"""The two warmth fractions, summed here term by term rather than reduced by the readout.

    $$\frac{\sum_{b,a,m,\ell} v_{b,a}\,\alpha_{b,\,t_{b,a},\,m,\,\ell}\;
             \mathrm{warm}\!\left(t_{b,a} - \ell\right)}
           {\sum_{b,a,m,\ell} v_{b,a}\,\alpha_{b,\,t_{b,a},\,m,\,\ell}}$$

    Written as four nested loops on purpose: the readout's own version gathers, expands and reduces,
    and a second vectorised expression of it would share the mistakes worth catching -- an anchor
    axis read as a step axis, a lag sign flipped, a denominator counting rows rather than mass.

    Args:
        model: The model whose two warmth patterns are read.
        out: A forward dict carrying ``attn_weights``, ``anchor_index`` and ``anchor_valid``.

    Returns:
        ``{'source_lag_warmth_frac_st', 'source_lag_warmth_frac_ph'}`` as floats.
    """
    alpha = out["attn_weights"]
    anchors, valid = out["anchor_index"], out["anchor_valid"]
    batch, _steps, heads, lags = alpha.shape

    total = 0.0
    warm = {name: 0.0 for name in _WARMTH_KEYS}
    patterns = {
        "source_lag_warmth_frac_st": model.source_block_warm_st.tolist(),
        "source_lag_warmth_frac_ph": model.source_block_warm_ph.tolist(),
    }
    for sample in range(batch):
        for slot in range(int(anchors.shape[1])):
            if not bool(valid[sample, slot]):
                continue
            anchor = int(anchors[sample, slot])
            for head in range(heads):
                for lag in range(lags):
                    mass = float(alpha[sample, anchor, head, lag])
                    total += mass
                    step = anchor - lag
                    if step < 0:
                        continue
                    for name, pattern in patterns.items():
                        if pattern[step]:
                            warm[name] += mass
    return {name: value / total for name, value in warm.items()}


# =================================================================================================
# anchors_per_sample: the tiling, actually firing
# =================================================================================================
def test_the_anchor_count_is_the_geometry_derived_tile_count_at_the_training_stride() -> None:
    r"""Every phase in $[0, S)$ at once, so the reported mean is the real one:
    $\lceil (T_{\mathrm{valid}} - F - \varphi)/S \rceil$, which at the shipped geometry is $11$ for
    $\varphi \le 16$ and $4$ otherwise, summing to $137$ and averaging to $137/30$.

    Re-derived here from the model's own resolved geometry rather than written down, so a horizon or
    budget change moves the expectation with the model instead of failing this test.
    """
    model = build(shipped_warmup_kwargs())
    stride = model.anchor_stride
    span = model.geometry.t_valid - model.warmup_period

    reported = _anchor_count(model, torch.arange(stride), stride, batch=stride)

    per_phase = [-(-(span - phase) // stride) for phase in range(stride)]
    assert stride == SHIPPED_HORIZON and span == 137
    assert min(per_phase) == 4 and max(per_phase) == 5
    assert sum(per_phase) == span
    assert reported == pytest.approx(span / stride, rel=1e-6)


def test_the_anchor_count_is_the_whole_valid_range_at_the_validation_stride() -> None:
    r"""Validation and test decode every valid anchor, so the count is $T_{\mathrm{valid}} - F = 137$
    exactly -- not a tile set at one fixed phase, which would sample the same $11$ positions of every
    segment forever."""
    model = build(shipped_warmup_kwargs())
    span = model.geometry.t_valid - model.warmup_period

    assert _anchor_count(model, None, 1, batch=2) == float(span)


def test_the_anchor_count_reports_the_decoded_set_on_a_fully_masked_batch() -> None:
    """Counted off ``anchor_valid``, not off the mask. A batch whose weight is entirely zero has no
    contributing anchor at all, and this column must still say which anchors the forward *built* --
    otherwise a gap-heavy validation batch would read as the geometry having collapsed."""
    model, streams, signal = _tiled()
    weight = _weight(model, value=0.0)

    out, metrics = _metrics(model, streams, signal, torch.tensor(_PHASES), weight=weight)

    assert float(metrics["anchors_per_sample"]) == float(out["anchor_valid"].sum()) / BATCH
    for name in _ADDED_METRIC_KEYS:
        assert math.isfinite(float(metrics[name])), name


# =================================================================================================
# source_lag_warmth_frac: the compromise, sized
# =================================================================================================
def test_the_two_warmth_fractions_are_the_hand_summed_shares_of_the_attention_mass() -> None:
    """The intermediate case, term by term, and with the two blocks apart.

    Both blocks are read off the *same* attention rows and differ only in which lagged source steps
    count as warm, so a split taken at the wrong boundary -- or applied to a pooled pattern -- would
    return two equal numbers with every shape correct. The assertion that they differ is what makes
    the per-block split checked rather than merely computed.
    """
    model, streams, signal = _tiled()
    out, metrics = _metrics(model, streams, signal, torch.tensor(_PHASES))

    expected = _hand_warmth(model, out)

    for name in _WARMTH_KEYS:
        value = float(metrics[name])
        assert 0.0 <= value <= 1.0, (name, value)
        assert value == pytest.approx(expected[name], rel=1e-5), name
    # The first stored source block is warm from step 4 and the second not until step 10, so a
    # pooled figure would let the first carry the fraction.
    assert float(metrics[_WARMTH_KEYS[0]]) != float(metrics[_WARMTH_KEYS[1]])


def test_a_source_warm_from_step_zero_reads_exactly_one() -> None:
    """The upper extreme, exactly rather than approximately: with every channel honest at step $0$
    both patterns are all-``True``, so every unit of attention mass lands on a warm lag and the
    fraction is the ratio of one sum to itself."""
    model, streams, signal = _tiled(
        source_warmup_steps=tuple(0 for _ in range(CAUSAL_C_U))
    )

    _out, metrics = _metrics(model, streams, signal, torch.tensor(_PHASES))

    assert bool(model.source_block_warm_st.all()) and bool(model.source_block_warm_ph.all())
    for name in _WARMTH_KEYS:
        assert float(metrics[name]) == 1.0, name


def test_a_source_warm_after_every_reachable_lag_reads_exactly_zero() -> None:
    """The lower extreme, which is what makes the metric orientable at all. A range check and a
    recomposition are both satisfied by an inverted fraction; only the two ends together fix which
    way it points."""
    model, streams, signal = _tiled(
        source_warmup_steps=tuple(TINY_SEQ_LEN for _ in range(CAUSAL_C_U))
    )

    _out, metrics = _metrics(model, streams, signal, torch.tensor(_PHASES))

    assert not bool(model.source_block_warm_st.any())
    assert not bool(model.source_block_warm_ph.any())
    for name in _WARMTH_KEYS:
        assert float(metrics[name]) == 0.0, name


def test_both_warmth_fractions_move_when_the_lag_floor_moves() -> None:
    r"""A non-zero ``lag_floor`` forbids the lags nearest the anchor, so the surviving mass shifts
    onto source steps that are further back -- and therefore, per block, differently warm. A column
    that did not move here would be reporting something other than where the attention landed."""
    model, streams, signal = _tiled()
    floored, floored_streams, floored_signal = _tiled(lag_floor=6)

    _out, unfloored_metrics = _metrics(model, streams, signal, torch.tensor(_PHASES))
    _out, floored_metrics = _metrics(
        floored, floored_streams, floored_signal, torch.tensor(_PHASES)
    )

    assert model.lag_floor == 0 and floored.lag_floor == 6
    for name in _WARMTH_KEYS:
        assert float(unfloored_metrics[name]) != float(floored_metrics[name]), name


def test_the_three_readouts_carry_no_gradient() -> None:
    """They are diagnostics. A term that reached the graph would be an objective term no weight in
    any config controls."""
    model, streams, signal = _tiled()
    model.train()

    out = model(*streams, torch.tensor(_PHASES))
    metrics = model.compute_loss(out, signal, weight=_weight(model))["metrics"]

    for name in _ADDED_METRIC_KEYS:
        assert not metrics[name].requires_grad, name


# =================================================================================================
# Where the three are merged, and what stops one from shadowing a column
# =================================================================================================
def test_the_three_readouts_are_merged_and_none_shadows_an_objective_metric() -> None:
    """Merged by assignment onto the shared objective's own dict, which is pinned bitwise for the
    four shipped forecasters and therefore may not gain them there.

    An assignment is silent on a collision, so what protects the column is that the three names are
    *new*: a readout reusing an objective name would replace it in ``metrics_history.csv`` and in
    MLflow with nothing raising, and the column is read across the whole family.
    """
    model, streams, signal = _tiled()
    out, metrics = _metrics(model, streams, signal, torch.tensor(_PHASES))

    objective = compute_shared_objective(
        out,
        gather_anchored_future_target(
            signal, model.geometry, out["anchor_index"], future_index=model.future_index
        ),
        weight=_weight(model),
        geometry=model.geometry,
        block_width=model.geometry.r,
        coverage_floor=model.coverage_floor,
        logvar_clamp=model.logvar_clamp,
        likelihood="mse",
    )["metrics"]

    assert set(_ADDED_METRIC_KEYS) <= set(metrics)
    assert set(_ADDED_METRIC_KEYS) & set(objective) == set()


def test_a_readout_reusing_an_objective_metric_name_is_refused(monkeypatch) -> None:
    """The refusal this cell's readouts will be merged through once the task exists, exercised on
    the shared task the package's own will inherit it from.

    The hook merges last, so a plain update would let a readout replace an objective metric *under
    the objective's own name*: ``pred_gap`` in ``metrics_history.csv`` would then stop meaning what
    it means in every other cell of the grid, with no error and nothing in the log.

    The model is built at the dense stride because the shared task has no phase seam -- deriving one
    per segment is what this package's own task adds -- and the guard being tested is downstream of
    both.
    """
    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask

    kwargs = tiny_warmup_kwargs()
    module = SeqVaeLagAttnRwsTask(build(kwargs), lr=1e-3, model_kwargs=kwargs, likelihood="mse")
    module.setup("fit")
    batch = make_stub_batch()

    monkeypatch.setattr(
        type(module),
        "_added_metrics",
        lambda self, inputs, outs, weight, stage: {"pred_gap": torch.zeros(())},
    )
    with pytest.raises(ValueError, match=r"pred_gap"):
        module.compute_loss_and_metrics(batch, 0, "val")

    # Not vacuous: a name of its own still merges, which is what the hook is for -- and the three
    # readouts this cell merges in ``compute_loss`` arrive on the same dict.
    monkeypatch.setattr(
        type(module),
        "_added_metrics",
        lambda self, inputs, outs, weight, stage: {"a_name_no_objective_uses": torch.zeros(())},
    )
    _loss, metrics = module.compute_loss_and_metrics(batch, 0, "val")
    assert "a_name_no_objective_uses" in metrics
    assert set(_ADDED_METRIC_KEYS) <= set(metrics)


# =================================================================================================
# kld_source_null: the floor the availability clock induces
# =================================================================================================
def _null_and_matched(model, streams, signal, phase):
    """``(forward, source_conditioned_kl_raw, kld_source_null)`` from one forward."""
    out, metrics = _metrics(model, streams, signal, phase)
    null = controls.source_null_kld(model, out, streams[2], _weight(model))
    return out, float(metrics["source_conditioned_kl_raw"]), float(null)


def test_the_null_readout_is_a_finite_nonnegative_rate(perturb_posterior) -> None:
    """A KL over the same support and in the same nats-per-anchor units as the coupling readout it
    is printed beside, which is the only reason subtracting one from the other means anything."""
    model, streams, signal = _tiled()
    perturb_posterior(model)

    _out, matched, null = _null_and_matched(model, streams, signal, torch.tensor(_PHASES))

    assert math.isfinite(null) and null >= 0.0
    assert math.isfinite(matched)


def test_the_null_is_read_over_the_tiled_anchor_support(perturb_posterior) -> None:
    """The same anchors the coupling readout is averaged over, so the difference of the two is a
    difference of one quantity rather than of two denominators. Two strides decode different sets, so
    a null computed over a fixed support would not move between them."""
    model, streams, signal = _tiled()
    perturb_posterior(model)
    weight = _weight(model)

    torch.manual_seed(0)
    with torch.no_grad():
        tiled = model(*streams, torch.tensor(_PHASES), TINY_STRIDE)
        dense = model(*streams, None, 1)

    assert tiled["anchor_index"].shape != dense["anchor_index"].shape
    assert float(controls.source_null_kld(model, tiled, streams[2], weight)) != pytest.approx(
        float(controls.source_null_kld(model, dense, streams[2], weight)), rel=1e-9
    )


def test_the_null_encode_is_one_row_broadcast_and_reproduces_the_full_batch_one(
    perturb_posterior,
) -> None:
    r"""With $x \equiv 0$ the adapter's output is a function of the availability pattern alone, so
    the source state is identical in every batch element and the arm encodes it once at batch $1$
    and expands. What that saving must not do is change the answer, and this is where that is
    checked: the same three tensors, rebuilt from a **full-batch** encode of zeros.

    **Not bitwise, and the reason is the kernel rather than the control.** The convolutional-LSTM
    encoder's batched kernels reduce in an order that depends on the batch extent, so a batch-$1$
    encode and a batch-$B$ one differ in the last bits -- and so do two *identical rows of one*
    full-batch encode, by the same order of magnitude. That second quantity is measured here rather
    than assumed, and the arm's own discrepancy is required to sit at it: a residual larger than the
    encoder's own row-to-row noise would be the broadcast changing the answer, which is exactly what
    this test is for. Asserting equality bit for bit would be asserting a property of the linear
    algebra backend.

    The lag mask is the model's own -- the floored one -- because a control that built its own would
    bypass a restriction the model was configured with.
    """
    model, streams, signal = _tiled()
    perturb_posterior(model)
    out, _metric_dict = _metrics(model, streams, signal, torch.tensor(_PHASES))
    u_stream = streams[2]

    nulled = controls.source_null_forward_outputs(model, out, u_stream)

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

    assert not model.query_uses_logvar, "this fixture poses a mu^p query; the reference assumes it"

    # The encoder's own row-to-row noise on an input that is identical in every row: the floor any
    # comparison against a batch-1 encode can be held to, and the reason this one is not bitwise.
    row_noise = float((full_batch[0] - full_batch[-1]).abs().max())
    bound = max(10.0 * row_noise, 1e-6)
    assert row_noise < 1e-5, row_noise
    for name, expected in (
        ("mu_post", mu_post),
        ("logvar_post", logvar_post),
        ("attn_weights", alpha),
    ):
        assert tuple(nulled[name].shape) == tuple(expected.shape), name
        assert float((nulled[name] - expected).abs().max()) < bound, name


def test_the_null_differs_from_the_coupling_readout_when_the_source_is_read(
    perturb_posterior,
) -> None:
    """The first direction of the non-vacuity pair: on a model whose posterior responds to the
    source, replacing that source with a flat trajectory must change the divergence. Equality here
    would mean the readout was measuring the availability clock all along -- which is exactly the
    finding this arm exists to be able to report, and therefore exactly what must not hold by
    construction."""
    model, streams, signal = _tiled()
    perturb_posterior(model)

    _out, matched, null = _null_and_matched(model, streams, signal, torch.tensor(_PHASES))

    assert matched != pytest.approx(null, rel=1e-6)


def test_the_null_equals_the_coupling_readout_when_the_posterior_ignores_the_source(
    perturb_posterior,
) -> None:
    r"""The second direction, and the sharper one. The attended source enters the posterior only
    through ``a_head_norm``, so zeroing that norm's affine parameters makes the fusion's source half
    identically zero whatever the source was -- a posterior that reads the target and nothing else.
    Both readouts must then agree exactly, because they differ only in the source.

    Without this direction the test above would pass for a null arm that computed *anything*
    different from the matched one.
    """
    model, streams, signal = _tiled()
    perturb_posterior(model)
    with torch.no_grad():
        model.posterior_head.a_head_norm.weight.zero_()
        model.posterior_head.a_head_norm.bias.zero_()

    _out, matched, null = _null_and_matched(model, streams, signal, torch.tensor(_PHASES))

    assert matched > 0.0, "the posterior collapsed onto the prior; the probe is vacuous"
    assert matched == pytest.approx(null, rel=1e-6)


def test_the_null_does_not_move_under_a_derangement_of_the_source(perturb_posterior) -> None:
    """The property that makes this a control the shuffle is not.

    The permutation arm deranges the encoded source across the batch, and every row carries the same
    availability pattern, so no permutation can remove it. This arm replaces the stream instead, so a
    derangement of that stream leaves it exactly where it was -- while the shuffled readout, by
    construction, does not.
    """
    model, streams, signal = _tiled()
    perturb_posterior(model)
    y_st, y_ph, u_stream = streams
    deranged = u_stream.flip(0)

    _out, _matched, null = _null_and_matched(model, streams, signal, torch.tensor(_PHASES))
    _out, _matched_perm, null_perm = _null_and_matched(
        model, (y_st, y_ph, deranged), signal, torch.tensor(_PHASES)
    )

    assert not torch.equal(u_stream, deranged), "the derangement was a no-op"
    assert null == pytest.approx(null_perm, rel=1e-6)
