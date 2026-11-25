r"""The IRF head: in place of the lag attention (``source_path='irf'``) or beside it (``irf_readout``).

Ten checks. The first seven are of the head and of the replacement. The tiny model builds, runs both stages and starves no parameter. The step-grid UA
level sits on the loader's step grid. The head is causal. The head reads a planted delay and stays
mostly shut for another row's UA. ``irf_lr_scale`` puts the head in its own optimizer group.
``irf_channels`` shifts the listed channels only. With the option off the model is the one before the option existed: the reference values at the bottom were
recorded from the code at commit ``e6dae170``. The last three are of the head beside the lag
attention: the readout changes no output and its loss terms reach no other parameter; the gate's
evidence term is inert at zero weights and moves the gate otherwise; the wrong-UA full term sends
its gradient where the full term does.

The head's windows are fixed (resting tone and high-pass 75 steps, delays up to 30 steps), so the
head checks run on 300-step synthetic sequences, not on the 24-step tiny geometry, where the gate
cannot open.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from teb_vae.coupling_irf.data import TRIM, _steps
from teb_vae.coupling_irf.model import irf_delay, plant
from teb_vae.coupling_irf.tests.test_model import synthetic
from teb_vae.lag_attn_cfs.nets.irf_source import IrfSourceHead, ua_step_level

from .conftest import TINY_STRIDE, build, make_streams, make_stub_batch, make_task, tiny_warmup_kwargs

ANCHORS = torch.arange(134, 270)  # the shipped dense anchor range


def _batch():
    """Four stub rows with the UA validity the head reads."""
    batch = make_stub_batch(4)
    batch.up_valid = torch.ones_like(batch.up, dtype=torch.bool)
    return batch


def test_the_irf_model_builds_runs_and_starves_no_parameter():
    """One backward of the full objective, all three new weights on, reaches every trainable
    parameter: what ``find_unused_parameters=False`` needs. The validation step runs too, without
    the readouts that re-pose a posterior."""
    module = make_task(
        model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, source_path="irf"),
        hparams=dict(lambda_source_expl=1.0, lambda_source_wrong=1.0),
    )
    model = module.orig_model
    batch = _batch()

    loss, metrics = module.compute_loss_and_metrics(batch, 0, "train")
    loss.backward()
    with torch.no_grad():
        _loss, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    assert not [n for n in ("query_proj", "lag_attn", "posterior_head") if hasattr(model, n)]
    assert sorted(k for k in model.state_dict() if k.startswith("irf_head.")) == [
        "irf_head.eta_logit", "irf_head.log_lambda", "irf_head.rho_logit"
    ]
    starved = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not starved, starved
    assert torch.isfinite(loss) and float(metrics["source_conditioned_kl_raw"]) == 0.0
    added = {"pred_gap_expl", "pred_gap_wrong", "pred_gap_expl_wrong", "irf_gate_open", "irf_gate_open_wrong"}
    assert added <= set(metrics) and added <= set(val_metrics)
    assert not {"kld_shuffled", "kld_source_null", "pred_gap_mean"} & set(val_metrics)


def test_the_ua_level_is_on_the_loaders_step_grid():
    """Step t of the trimmed window ends at trimmed raw sample 16 t. Checked against the step cache
    of ``coupling_irf.data``, which reads the untrimmed row."""
    generator = np.random.default_rng(0)
    raw = generator.normal(size=(2, 5280)).astype(np.float32)
    valid = generator.random((2, 5280)) > 0.3
    mean, _sd, frac = _steps(raw, valid)

    level, ok = ua_step_level(torch.from_numpy(raw[:, TRIM:-TRIM]), torch.from_numpy(valid[:, TRIM:-TRIM]), 16, 300)

    # Step 0 has one sample inside the trimmed window, so it is never valid here.
    assert torch.allclose(level[:, 1:], torch.from_numpy(mean[:, 1:]), atol=1e-5)
    assert torch.equal(ok[:, 1:], torch.from_numpy(frac[:, 1:].astype(np.float32) >= 0.5))
    assert not ok[:, 0].any()


def _coupled(delay: float, seed: int = 0):
    """Eight 300-step rows. Channel 0 carries a response to its own UA at ``delay`` steps; channel 1
    is noise."""
    b = synthetic(T=300, seed=seed)
    u = b["ua"] - b["ua"].amin(-1, keepdim=True)
    level = (plant(b["fhr"], u, torch.full((8,), delay), depth=20.0) - 140.0) / 20.0
    noise = 0.1 * torch.randn(8, 300, generator=torch.Generator().manual_seed(seed + 1))
    ok = torch.ones(8, 300, dtype=torch.bool)
    return torch.stack([level, noise], -1), b["ua"] / 25.0, ok


def _head(y, ua, ok):
    head = IrfSourceHead(30, y.shape[-1]).eval()
    with torch.no_grad():
        return head(y, ok[..., None].expand_as(y), ok, ua, ok, ANCHORS[None].expand(len(y), -1))


def test_nothing_at_an_anchor_depends_on_the_future():
    y, ua, ok = _coupled(8.0, seed=1)
    cut = 200                                               # change everything after step 200
    y_later, ua_later = y.clone(), ua.clone()
    y_later[:, cut + 1:] += 1.5
    ua_later[:, cut + 1:] += 1.6
    own, later = _head(y, ua, ok), _head(y_later, ua_later, ok)
    past = ANCHORS <= cut

    assert own["a"][:, past].max() > 0.3                    # the head is switched on
    for key in ("delta", "a", "h"):
        assert torch.allclose(own[key][:, past], later[key][:, past], atol=1e-5), key
    # the explain mode reads UA up to the target, so future UA must change it, but future FHR must not
    assert not torch.allclose(own["delta_expl"][:, past], _head(y, ua_later, ok)["delta_expl"][:, past], atol=1e-5)
    assert torch.allclose(own["delta_expl"][:, past], _head(y_later, ua, ok)["delta_expl"][:, past], atol=1e-5)


@pytest.mark.parametrize("true_delay", [5.0, 9.0, 15.0])    # 20, 36 and 60 s
def test_a_planted_delay_is_read_and_another_rows_ua_opens_the_gate_much_less(true_delay):
    y, ua, ok = _coupled(true_delay)
    own, wrong = _head(y, ua, ok), _head(y, ua.roll(1, 0), ok)

    delay, depth = irf_delay(own["h"][:, -1, :, 0])         # IRF of the coupled channel, last anchor
    assert (delay - true_delay).abs().median() < 1.0, delay  # within one step (4 s)
    assert (depth < 0).all()
    # The synthetic contractions are near periodic, so a rolled UA still fits a little.
    assert own["a"][:, -1, 0].mean() > 0.7 > 2 * wrong["a"][:, -1, 0].mean(), (own["a"][:, -1], wrong["a"][:, -1])
    assert not own["a"][..., 1].any()                       # the noise channel stays shut


def test_irf_lr_scale_gives_the_head_its_own_optimizer_group():
    """Every trainable parameter is in exactly one group, the head's at the scaled rate. At the
    default 1.0 the optimizer is built from the flat list, as before the key."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, source_path="irf")
    scaled, plain = make_task(model_kwargs=kwargs, irf_lr_scale=30.0), make_task(model_kwargs=kwargs)

    rest, head = scaled.build_optimizer(scaled.configure_param_groups()).param_groups
    head_ids = {id(p) for p in scaled.orig_model.irf_head.parameters()}
    trainable = [id(p) for p in scaled.parameters() if p.requires_grad]

    assert {id(p) for p in head["params"]} == head_ids and head["lr"] == pytest.approx(30.0 * rest["lr"])
    assert sorted(id(p) for group in (rest, head) for p in group["params"]) == sorted(trainable)
    assert len(plain.build_optimizer(plain.configure_param_groups()).param_groups) == 1


def test_irf_channels_shifts_the_listed_channels_and_no_other():
    """The head runs on kept channels 0 and 5 only; the model returns every tensor at the full kept
    width, with an open gate on the coupled channel and exact zeros on the unlisted ones."""
    kwargs = tiny_warmup_kwargs(sequence_length=300, source_path="irf", irf_channels=[0, 5])
    model = build(kwargs).eval()
    y, ua, ok = _coupled(9.0)
    features = torch.zeros(8, 300, kwargs["c_y"])
    features[..., 0], features[..., 5] = y[..., 0], y[..., 1]   # declared = kept index for both

    with torch.no_grad():
        out = model.irf_source(features, ua, ok, ok, ANCHORS[None].expand(8, -1))

    unlisted = [c for c in range(model.decoder_out_channels) if c not in (0, 5)]
    assert model.irf_head.rho_logit.shape == (model.horizon, 2)
    assert out["delta"].shape == (8, len(ANCHORS), model.horizon, model.decoder_out_channels)
    assert out["a"][:, -1, 0].mean() > 0.7 and out["delta"][..., 0].abs().max() > 0
    for key in ("delta", "delta_expl", "h", "a", "z"):
        assert not out[key][..., unlisted].any(), key


def test_the_readout_changes_no_output_and_its_terms_reach_no_other_parameter(perturb_posterior):
    """``irf_readout``: same weights and same noise give every output of the plain model, and the
    head's three loss terms leave no gradient on any parameter outside the head."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    plain = make_task(model_kwargs=kwargs)
    hybrid = make_task(
        model_kwargs=dict(kwargs, irf_readout=True),
        hparams=dict(lambda_source_irf=1.0, lambda_source_expl=1.0, lambda_source_wrong=1.0),
    )
    batch, outs = _batch(), []
    for module in (plain, hybrid):
        perturb_posterior(module.orig_model)                # the same perturbation in both
        torch.manual_seed(1)
        outs.append(module.orig_model(*module._build_forward_inputs(batch)))

    assert all(torch.equal(value, outs[1][key]) for key, value in outs[0].items())
    assert {"irf_h", "irf_gate", "irf_z", "irf_ua_ok", "irf_delta", "irf_delta_expl"} == set(outs[1]) - set(outs[0])
    features, weight = hybrid._build_raw_target(batch)
    loss, metrics = hybrid._irf_source_terms(
        batch, hybrid._build_forward_inputs(batch), outs[1], features, weight, "gaussian_nll", "train"
    )
    loss.backward()
    model = hybrid.orig_model
    assert "pred_gap_irf" in metrics
    assert all(p.grad is not None for p in model.irf_head.parameters())
    assert not [n for n, p in model.named_parameters() if not n.startswith("irf_head.") and p.grad is not None]


def test_the_gate_evidence_is_inert_at_zero_weights_and_moves_the_gate_otherwise():
    """``irf_gate_evidence`` on 300-step rows, where the head has evidence to give."""
    kwargs = tiny_warmup_kwargs(sequence_length=300, posterior_event_gate=True, source_quality_mask=True,
                                source_mask_mode="soft", irf_readout=True)
    ok = torch.ones(2, 300, dtype=torch.bool)
    head_inputs = dict(ua_level=torch.randn(2, 300, generator=torch.Generator().manual_seed(2)),
                       ua_valid=ok, target_step_valid=ok)

    def forward(model):
        torch.manual_seed(1)
        with torch.no_grad():
            return model.eval()(*make_streams(kwargs), 0, 1, ok[..., None].expand(-1, -1, kwargs["c_u"]), **head_inputs)

    without, model = forward(build(kwargs)), build(dict(kwargs, irf_gate_evidence=True))
    at_zero = forward(model)
    with torch.no_grad():
        model.irf_evidence_gate.weight.fill_(1.0)
    moved = forward(model)

    assert at_zero["irf_evidence"].shape == (2, 300, 2) and at_zero["irf_evidence"].max() > 0
    assert all(torch.equal(value, at_zero[key]) for key, value in without.items())
    assert not torch.equal(moved["event_gate"], at_zero["event_gate"])


def test_the_wrong_ua_full_term_sends_its_gradient_where_the_full_term_does(perturb_posterior):
    """``lambda_full_wrong``: the term reaches the lag attention, and, because the full branch
    detaches nothing, the shared decoder and the target encoder as well."""
    module = make_task(
        model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE), hparams=dict(lambda_full_wrong=1.0)
    )
    model, batch = module.orig_model, _batch()
    perturb_posterior(model)
    _loss, metrics = module.compute_loss_and_metrics(batch, 0, "train")
    features, weight = module._build_raw_target(batch)
    out = model(*module._build_forward_inputs(batch))

    nll, _metrics = module._full_wrong_term(batch, out, features, weight, "gaussian_nll", "train")
    nll.backward()

    assert "pred_gap_full_wrong" in metrics and torch.isfinite(nll)
    for name in ("query_proj.weight", "lag_attn.W_k.weight", "decoder.mean_head.weight"):
        assert model.get_parameter(name).grad.abs().sum() > 0, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.target_encoder.parameters())


def test_with_the_option_off_the_model_is_the_one_before_the_option_existed():
    """Same state-dict keys and same forward as the code before ``source_path`` existed."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, posterior_event_gate=True, persistence_residual=True)
    model = build(kwargs).eval()
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():                                   # off the zero-KL start, so full != base
        for parameter in model.posterior_head.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)
    keys = sorted(model.state_dict())
    torch.manual_seed(1)
    with torch.no_grad():
        out = model(*make_streams(kwargs), 1, TINY_STRIDE)

    assert model.source_path == "lag_attn"
    assert (len(keys), hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]) == (316, "49ce312ef6f4bf6d")
    for name, reference in REFERENCE.items():
        assert float(out[name].double().abs().sum()) == pytest.approx(reference, rel=1e-5), name


#: Sum of absolute values of each forward output, recorded from the code at commit e6dae170.
REFERENCE = {
    "mu_base": 2045.2247626632452,
    "mu_full": 2092.5409951657057,
    "logvar_full": 3289.0107934474945,
    "kld_per_t": 899.0802245736122,
    "attn_weights": 192.00000000325963,
    "event_gate": 43.66530340909958,
}
