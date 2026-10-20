r"""The two event mechanisms: the posterior event gate and the contraction weighting of the full branch.

Both are shared-parent mechanisms whose constructor and loss defaults are OFF, and whose off-state
is bitwise the model and the objective before the keys existed; `test_construct.py` pins the
construction half beside the other switches. This cell's `default.yaml` ships both ON. What is
asserted here is the *on*-state's contract:

* the gate exists only under the flag, starts open and input-independent, leaves the KL at its
  exact zero at initialisation, and at $g = 0$ makes the posterior the prior exactly in both
  log-variance modes;
* its sparsity term is refused on a model that built no gate, and rides $\beta$;
* the contraction weight marks the reach after each detected onset and nothing else;
* the weight moves only the full-branch term that enters the total, leaving ``nll_full_block``
  and ``pred_gap`` the unweighted quantities, and the task builds it from the batch's raw UP.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn.nets.heads import EVENT_GATE_BIAS_INIT
from teb_vae.lag_attn_rws.event_weight import contraction_anchor_weight
from teb_vae.lag_attn_rws.seed_event_gate_checkpoint import seed_event_gate
from train.graph_models_utils import load_checkpoint_strict

from .conftest import TINY_SEQ_LEN, build, make_streams, tiny_warmup_kwargs

DECIMATION = 16


def _forward(kwargs, seed: int = 0):
    """One model and one forward at a fixed seed; the stride is the inert 1, so no phase is needed."""
    model = build(kwargs, seed=seed).eval()
    streams = make_streams(kwargs)
    torch.manual_seed(seed)
    with torch.no_grad():
        return model, streams, model(*streams)


def _loss(model, streams, outputs, **overrides):
    weight = torch.ones(streams[0].shape[0], streams[0].shape[1])
    return model.compute_loss(
        outputs, torch.cat([streams[0], streams[1]], dim=-1), weight=weight, **overrides
    )["metrics"]


# ---------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------
def test_the_gate_is_absent_off_and_present_open_and_flat_on():
    """Off: no key, like every other switch. On: $(B, T, 1)$ in $(0, 1)$, and at init exactly
    $\\sigma(b_0)$ everywhere -- zero weight, so no input moves it before training does."""
    _, _, off = _forward(tiny_warmup_kwargs())
    assert "event_gate" not in off

    kwargs = tiny_warmup_kwargs(posterior_event_gate=True)
    _, streams, on = _forward(kwargs)
    gate = on["event_gate"]
    assert gate.shape == (streams[0].shape[0], streams[0].shape[1], 1)
    assert torch.allclose(gate, torch.full_like(gate, 1.0 / (1.0 + math.exp(-EVENT_GATE_BIAS_INIT))))


@pytest.mark.parametrize(
    "mode_kwargs, exact",
    [
        ({"posterior_logvar_mode": "residual"}, True),
        ({"posterior_logvar_mode": "independent", "head_init_calibration": True}, False),
    ],
)
def test_the_gated_model_keeps_the_zero_kl_start(mode_kwargs, exact):
    """The gate multiplies a residual that is zero at init, so it cannot move the KL off its exact
    zero under `residual`; under `independent` the interpolation towards the prior's raw value
    agrees to float rounding, as the ungated head does under calibration."""
    _, _, outputs = _forward(tiny_warmup_kwargs(posterior_event_gate=True, **mode_kwargs))
    kld = outputs["kld_per_t"]
    if exact:
        assert torch.all(kld == 0.0)
    else:
        assert torch.allclose(kld, torch.zeros_like(kld), atol=1e-5)


@pytest.mark.parametrize("mode", ["residual", "independent"])
def test_a_closed_gate_is_the_prior_exactly_in_both_modes(mode, perturb_posterior):
    """$g = 0$ must give $q = p$ in both moments, on a head whose deltas are NOT zero -- otherwise the
    sparsity term would be buying something other than silence."""
    kwargs = tiny_warmup_kwargs(posterior_event_gate=True, posterior_logvar_mode=mode)
    model, _, outputs = _forward(kwargs)
    perturb_posterior(model)
    head = model.posterior_head
    attended = outputs["attended_source_heads"]
    with torch.no_grad():
        mu_open, _ = head(
            outputs["target_state"], attended, outputs["mu_prior"], outputs["raw_logvar_prior"],
            gate=torch.ones_like(outputs["event_gate"]),
        )
        mu_closed, logvar_closed = head(
            outputs["target_state"], attended, outputs["mu_prior"], outputs["raw_logvar_prior"],
            gate=torch.zeros_like(outputs["event_gate"]),
        )
    assert not torch.allclose(mu_open, outputs["mu_prior"]), "the perturbation must have moved q"
    assert torch.equal(mu_closed, outputs["mu_prior"])
    assert torch.allclose(logvar_closed, outputs["logvar_prior"], atol=1e-6)


def test_a_pre_gate_checkpoint_loads_into_the_gated_model_only_once_seeded():
    """The strict loader refuses the old checkpoint as it is; the seeded copy loads, carries the
    old weights, opens the gate at its constructor seed and stamps the kwargs it now fits."""
    kwargs = tiny_warmup_kwargs()
    ungated = build(kwargs)
    blob = {
        "state_dict": {f"model.{name}": value for name, value in ungated.state_dict().items()},
        "model_kwargs": dict(kwargs),
        "model_class": type(ungated).__name__,
    }
    gated = build(tiny_warmup_kwargs(posterior_event_gate=True), seed=1)

    assert load_checkpoint_strict(gated, blob) is None
    seeded = seed_event_gate(blob)
    assert load_checkpoint_strict(gated, seeded) is gated

    assert seeded["model_kwargs"]["posterior_event_gate"] is True
    assert "model_kwargs" in blob and "posterior_event_gate" not in blob["model_kwargs"]
    head = gated.posterior_head.event_gate_head
    assert torch.all(head.weight == 0.0) and float(head.bias) == EVENT_GATE_BIAS_INIT
    assert torch.equal(gated.target_adapter.linear.weight, ungated.target_adapter.linear.weight)
    with pytest.raises(ValueError, match="already carries"):
        seed_event_gate(seeded)


def test_a_sparsity_weight_on_a_gateless_model_is_refused_by_name():
    model, streams, outputs = _forward(tiny_warmup_kwargs())
    with pytest.raises(ValueError, match="event_gate"):
        _loss(model, streams, outputs, lambda_event_gate=0.1)


def test_the_gate_term_rides_beta_and_reports_its_mean():
    r"""$\lambda_g \beta \, \mathbb E[g]$: at $\beta = 0$ the term is absent whatever $\lambda_g$,
    and at $\beta = 1$ the total moves by exactly $\lambda_g$ times the reported gate mean."""
    model, streams, outputs = _forward(tiny_warmup_kwargs(posterior_event_gate=True))
    base = _loss(model, streams, outputs, beta=1.0)
    charged = _loss(model, streams, outputs, beta=1.0, lambda_event_gate=0.5)
    cold = _loss(model, streams, outputs, beta=0.0, lambda_event_gate=0.5)
    cold_base = _loss(model, streams, outputs, beta=0.0)

    assert 0.0 < float(charged["event_gate_mean"]) < 1.0
    assert float(charged["lambda_event_gate"]) == 0.5
    # Compared as totals rather than as a difference of two float32 totals in the thousands, whose
    # cancellation error would swamp a relative tolerance on the 0.44-nat term.
    assert torch.allclose(
        charged["total_loss"], base["total_loss"] + 0.5 * charged["event_gate_mean"]
    )
    assert torch.equal(cold["total_loss"], cold_base["total_loss"])


# ---------------------------------------------------------------------------------------
# The contraction weight
# ---------------------------------------------------------------------------------------
def _bumped_up(batch: int, steps: int, onsets_steps, width_steps: int = 15) -> torch.Tensor:
    """A flat UP trace with one smooth contraction-shaped bump per onset, on the raw grid."""
    raw = torch.zeros(batch, steps * DECIMATION)
    t = torch.arange(steps * DECIMATION, dtype=torch.float32) / DECIMATION
    for onset in onsets_steps:
        centre = onset + width_steps  # the bump peaks a width after its onset
        raw += 30.0 * torch.exp(-0.5 * ((t - centre) / (0.6 * width_steps)) ** 2)
    return raw + 0.1 * torch.randn(batch, steps * DECIMATION, generator=torch.Generator().manual_seed(0))


def test_the_contraction_weight_marks_the_reach_after_each_onset_and_the_floor_elsewhere():
    """Two planted contractions on a quiet trace: weight 1 inside the reach after each detected
    onset, the floor far from both, and no third value anywhere."""
    steps, onsets, reach, floor = 300, (100, 200), 10, 0.1
    up = _bumped_up(2, steps, onsets)
    weight = torch.ones(2, steps)

    out = contraction_anchor_weight(up, weight, decimation=DECIMATION, reach_steps=reach, floor=floor)

    assert out.shape == (2, steps)
    # In-dtype comparisons: the weight is float32, where 0.1 is not the Python float 0.1.
    assert bool(((out == floor) | (out == 1.0)).all())
    for onset in onsets:
        window = out[0, onset - 6 : onset + reach + 6]
        assert float(window.max()) == 1.0, f"no active anchor near the planted onset at {onset}"
    assert bool((out[0, 20:60] == floor).all())
    assert bool((out[0, 260:] == floor).all())
    # The run of ones after an onset is exactly reach + 1 steps long.
    first_active = int(torch.nonzero(out[0] == 1.0)[0])
    assert torch.all(out[0, first_active : first_active + reach + 1] == 1.0)
    assert bool(out[0, first_active + reach + 1] == floor)

    # The onset cache: a keyed second call reads the detector's result back and agrees bitwise.
    cache: dict = {}
    first = contraction_anchor_weight(
        up, weight, decimation=DECIMATION, reach_steps=reach, floor=floor, keys=["a", "b"], cache=cache
    )
    second = contraction_anchor_weight(
        up, weight, decimation=DECIMATION, reach_steps=reach, floor=floor, keys=["a", "b"], cache=cache
    )
    assert set(cache) == {"a", "b"} and torch.equal(first, out) and torch.equal(second, out)


def test_the_contraction_weight_refuses_a_bad_floor_or_reach():
    up, weight = torch.zeros(1, 32 * DECIMATION), torch.ones(1, 32)
    with pytest.raises(ValueError, match="floor"):
        contraction_anchor_weight(up, weight, decimation=DECIMATION, reach_steps=1, floor=1.0)
    with pytest.raises(ValueError, match="reach"):
        contraction_anchor_weight(up, weight, decimation=DECIMATION, reach_steps=-1, floor=0.0)


def test_the_weight_moves_only_the_full_event_term():
    """A uniform weight is the unweighted objective; a non-uniform one leaves ``nll_full_block``
    and ``pred_gap`` untouched and moves ``nll_full_event_block`` alone."""
    model, streams, outputs = _forward(tiny_warmup_kwargs())
    batch, steps = streams[0].shape[0], streams[0].shape[1]
    plain = _loss(model, streams, outputs)
    uniform = _loss(model, streams, outputs, full_anchor_weight=torch.ones(batch, steps))
    skewed_weight = torch.full((batch, steps), 0.1)
    skewed_weight[:, : steps // 2] = 1.0
    skewed = _loss(model, streams, outputs, full_anchor_weight=skewed_weight)

    assert torch.equal(plain["nll_full_event_block"], plain["nll_full_block"])
    assert float(plain["event_anchor_frac"]) == 1.0
    assert float(uniform["nll_full_event_block"]) == pytest.approx(float(plain["nll_full_block"]), rel=1e-6)
    assert float(uniform["total_loss"]) == pytest.approx(float(plain["total_loss"]), rel=1e-6)

    assert torch.equal(skewed["nll_full_block"], plain["nll_full_block"])
    assert torch.equal(skewed["pred_gap"], plain["pred_gap"])
    assert not torch.equal(skewed["nll_full_event_block"], plain["nll_full_block"])
    assert 0.0 < float(skewed["event_anchor_frac"]) < 1.0


def test_the_task_builds_the_weight_from_the_batch_only_when_a_floor_is_set(task, stub_batch):
    """Off: the four columns report the unweighted objective. On: the full event term is what the
    total carries, and the weight came from the batch's raw ``up`` at the net's own reach."""
    off = task()
    _, off_metrics = off.compute_loss_and_metrics(stub_batch, 0, "train")
    assert torch.equal(off_metrics["nll_full_event_block"], off_metrics["nll_full_block"])
    assert float(off_metrics["event_anchor_frac"]) == 1.0
    assert float(off_metrics["event_gate_mean"]) == 1.0

    on = task(hparams={"event_weight_floor": 0.1})
    _, on_metrics = on.compute_loss_and_metrics(stub_batch, 0, "train")
    assert 0.0 <= float(on_metrics["event_anchor_frac"]) <= 1.0
    assert torch.isfinite(on_metrics["nll_full_event_block"])
    assert on.hparams["event_weight_floor"] == 0.1

    weight = on._event_anchor_weight(stub_batch, stub_batch.weight)
    assert weight is not None and weight.shape == (stub_batch.weight.shape[0], TINY_SEQ_LEN)
    assert off._event_anchor_weight(stub_batch, stub_batch.weight) is None

    # The weight runs THIS cell's evaluation detector, not the raw-signal cells' copy, so a training
    # anchor and a per-anchor table bin name one event.
    from teb_vae.lag_attn_cfs.eval import events as cfs_events
    from teb_vae.lag_attn_rws.eval import events as rws_events

    assert on._event_detector() is cfs_events.detect_contractions
    assert on._event_detector() is not rws_events.detect_contractions
