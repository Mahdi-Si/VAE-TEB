r"""The kept source-warmth readouts, their merge onto the objective, and the source-null floor.

The causal-feature cells report readouts that partition kept target channels, which this target does
not have -- its block's last axis counts raw samples -- so only the source- and anchor-side readouts
transfer. ``source_lag_warmth_frac_st`` / ``_ph`` are the share of attention mass landing on lags
where a source block is warm, per stored block, computed against the two per-step warmth patterns
this cell resolves itself.

The warmth columns are pinned at **both extremes**, because a range check alone passes on an
inverted metric: an all-zero warm-up must read exactly $1.0$ and a warm-up beyond every reachable lag
exactly $0.0$. Between them one fraction is recomputed here by an explicit sum over
$(b, a, m, \ell)$, which is what pins the per-block split rather than merely the pooled figure.

``kld_source_null`` is exercised against the free function that computes it, as a non-vacuity pair
against ``source_conditioned_kl_raw``: it must differ while the posterior reads the source and equal
it exactly once the posterior cannot. Every KL assertion runs after ``perturb_posterior``: the
posterior delta heads are zero-initialised, so on a fresh model every divergence is identically zero.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective

from .conftest import (
    BATCH,
    CAUSAL_C_U,
    TINY_SEQ_LEN,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
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
    # The two stored source blocks warm at different steps, so a pooled figure would let the first
    # carry the fraction.
    assert float(metrics[_WARMTH_KEYS[0]]) != float(metrics[_WARMTH_KEYS[1]])


@pytest.mark.parametrize(
    ("wait", "expected"),
    [(0, 1.0), (TINY_SEQ_LEN, 0.0)],
    ids=["warm_from_step_zero", "warm_after_every_lag"],
)
def test_the_warmth_fractions_read_exactly_their_extremes(wait: int, expected: float) -> None:
    """Both ends, exactly rather than approximately, which is what makes the metric orientable: a
    range check and a recomposition are both satisfied by an inverted fraction. With every source
    channel honest at step $0$ every unit of attention mass lands on a warm lag; with every channel
    cold past the window none does."""
    model, streams, signal = _tiled(source_warmup_steps=tuple(wait for _ in range(CAUSAL_C_U)))

    _out, metrics = _metrics(model, streams, signal, torch.tensor(_PHASES))

    patterns = torch.cat([model.source_block_warm_st, model.source_block_warm_ph])
    assert bool((patterns == bool(expected)).all())
    for name in _WARMTH_KEYS:
        assert float(metrics[name]) == expected, name


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
    """Merged by assignment onto the shared objective's own dict, which is silent on a collision, so
    what protects the column is that the three names are *new*: a readout reusing an objective name
    would replace it in ``metrics_history.csv`` and in MLflow with nothing raising."""
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


# =================================================================================================
# kld_source_null: the floor the availability clock induces
# =================================================================================================
def _null_and_matched(model, streams, signal, phase):
    """``(forward, source_conditioned_kl_raw, kld_source_null)`` from one forward."""
    out, metrics = _metrics(model, streams, signal, phase)
    null = controls.source_null_kld(model, out, streams[2], _weight(model))
    return out, float(metrics["source_conditioned_kl_raw"]), float(null)


def test_the_null_differs_from_the_coupling_readout_when_the_source_is_read(
    perturb_posterior,
) -> None:
    """The first direction of the non-vacuity pair: on a model whose posterior responds to the
    source, replacing that source with a flat trajectory must change the divergence. Equality here
    would mean the readout was measuring the availability clock all along."""
    model, streams, signal = _tiled()
    perturb_posterior(model)

    _out, matched, null = _null_and_matched(model, streams, signal, torch.tensor(_PHASES))

    assert matched != pytest.approx(null, rel=1e-6)


def test_the_null_equals_the_coupling_readout_when_the_posterior_ignores_the_source(
    perturb_posterior,
) -> None:
    r"""The second direction, and the sharper one. The attended source enters the posterior only
    through ``a_head_norm``, so zeroing that norm's affine parameters makes the fusion's source half
    identically zero whatever the source was. Both readouts must then agree exactly -- which also
    requires them to be averaged over the same tiled anchor support and in the same units.
    """
    model, streams, signal = _tiled()
    perturb_posterior(model)
    with torch.no_grad():
        model.posterior_head.a_head_norm.weight.zero_()
        model.posterior_head.a_head_norm.bias.zero_()

    _out, matched, null = _null_and_matched(model, streams, signal, torch.tensor(_PHASES))

    assert matched > 0.0, "the posterior collapsed onto the prior; the probe is vacuous"
    assert matched == pytest.approx(null, rel=1e-6)
