"""ClassifierNet (SPEC §9, §15 T-M1-T-M3) on synthetic contract batches."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from teb_vae.classifier.config import load
from teb_vae.classifier.model import ClassifierNet, Head, aggregate, n_params, running_aggregate

SMALL = {"model.step.d": 8, "model.pooling.d_attn": 8, "model.token.d": 16, "model.sequence.layers": 1,
         "model.sequence.heads": 2, "model.sequence.d_ff": 16, "model.head.hidden": 8}
C, CA, D, T = 5, 1, 3, 12
SEQ_KEYS = ("x", "step_mask", "attn", "ctx", "seg_mask", "t_h", "w", "row")
AGGS = ("causal_transformer", "gru", "attention_mil")


def cfg(small: bool = True, **over):
    """The classifier config: default.yaml + SMALL + overrides (``a__b=v`` means ``a.b=v``). ``labels__*`` and
    ``train__loss__*`` skip the root validator's head / loss / λ3 cross-checks, so any net-loss pairing builds."""
    items = {**(SMALL if small else {}), **{k.replace("__", "."): v for k, v in over.items()}}
    lab = {k.removeprefix("labels."): items.pop(k) for k in list(items) if k.startswith("labels.")}
    loss = {k.removeprefix("train.loss."): items.pop(k) for k in list(items) if k.startswith("train.loss.")}
    c = load(overrides=[f"classifier.{k}={v}" for k, v in items.items()]).classifier
    return c.model_copy(update={"labels": c.labels.model_copy(update=lab),
                                "train": c.train.model_copy(update={"loss": c.train.loss.model_copy(update=loss)})})


def net(c, n_values=C, n_attn=CA, n_ctx=D, **kw) -> ClassifierNet:
    torch.manual_seed(0)
    return ClassifierNet(n_values=n_values, n_attn=n_attn, n_ctx=n_ctx, model_cfg=c.model,
                         labels_cfg=c.labels, **kw)


def jitter(m: torch.nn.Module) -> torch.nn.Module:
    """Perturb every parameter so zero-initialised heads/biases give non-trivial outputs."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return m


def seg_batch(b: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(b, T, generator=g) > 0.3
    mask[:, -1] = True
    return {"x": torch.randn(b, T, C, generator=g) * mask[..., None], "step_mask": mask,
            "attn": torch.rand(b, T, CA, generator=g) * mask[..., None], "ctx": torch.randn(b, D, generator=g),
            "y": (torch.arange(b) % 2).float(), "y3": torch.arange(b) % 3, "w": torch.ones(b),
            "guid": torch.arange(b), "row": torch.arange(b)}


def seq_batch(lengths=(5, 3, 1, 0), seed: int = 0) -> dict:
    """Right-padded GUID batch; irregular t_h on a 1/16 h grid (exact float arithmetic)."""
    g = torch.Generator().manual_seed(seed)
    b, n = len(lengths), max(lengths)
    seg = torch.arange(n)[None] < torch.tensor(lengths)[:, None]
    mask = (torch.rand(b, n, T, generator=g) > 0.3)
    mask[..., -1] = True
    mask &= seg[..., None]
    t_h = torch.cumsum(torch.randint(1, 40, (b, n), generator=g) / 16, 1)
    return {"x": torch.randn(b, n, T, C, generator=g) * mask[..., None], "step_mask": mask,
            "attn": torch.rand(b, n, T, CA, generator=g) * mask[..., None],
            "ctx": torch.randn(b, n, D, generator=g) * seg[..., None], "seg_mask": seg,
            "t_h": (t_h - t_h[:, :1]) * seg, "y": (torch.arange(b) % 2).float(), "y3": torch.arange(b) % 3,
            "w": seg.float(), "guid": torch.arange(b), "row": torch.where(seg, torch.arange(n), -1)}


def _finite(out: dict, m: torch.nn.Module) -> None:
    assert all(torch.isfinite(v).all() for v in out.values())
    sum(v.sum() for k, v in out.items() if k.endswith("_score")).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


# ---- T-M1 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("pool", ["mean", "mean_max", "gated_attention", "query"])
@pytest.mark.parametrize("kind", AGGS)
def test_no_nan_with_padded_rows_and_single_segment(kind, pool):
    m = jitter(net(cfg(model__sequence__kind=kind, model__pooling__kind=pool)))
    out = m(seq_batch((5, 1, 0)))
    assert out["pos"].shape == (3, 5, 1)
    _finite(out, m)


@pytest.mark.parametrize("temporal", ["none", "causal_conv", "transformer"])
@pytest.mark.parametrize("pool", ["mean", "mean_max", "gated_attention", "query", "conjunctive"])
def test_segment_scope_finite(pool, temporal):
    m = jitter(net(cfg(model__scope="segment", model__pooling__kind=pool, model__step__temporal=temporal)))
    b = seg_batch()
    b["step_mask"][0, :-1] = False  # a single valid step
    out = m(b)
    assert out["seg"].shape == (4, 1) and out["step_attn"].shape == (4, T)
    assert ("step_seg" in out) == (pool == "conjunctive")
    _finite(out, m)


def test_fully_masked_segment_raises():
    b = seg_batch()
    b["step_mask"][1] = False
    with pytest.raises(ValueError, match="no valid step"):
        net(cfg(model__scope="segment"))(b)


# ---- T-M2 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("temporal", ["none", "transformer"])
@pytest.mark.parametrize("kind", AGGS)
def test_online_equals_prefix_sweep(kind, temporal):
    m = jitter(net(cfg(model__sequence__kind=kind, model__step__temporal=temporal))).eval()
    b = seq_batch((6, 4, 1, 0))
    with torch.no_grad():
        full = m(b)
        for n in range(6):
            pre = m({k: (v[:, : n + 1] if k in SEQ_KEYS else v) for k, v in b.items()})
            real = b["seg_mask"][:, n]
            for key in ("pos", "pos_score", "seg_score"):
                torch.testing.assert_close(pre[key][:, n][real], full[key][:, n][real], atol=1e-5, rtol=0)


@pytest.mark.parametrize("kind", AGGS)
def test_time_shift_invariance(kind):
    m = jitter(net(cfg(model__sequence__kind=kind))).eval()
    b = seq_batch((5, 3))
    real = b["seg_mask"]
    with torch.no_grad():
        a, s = m(b)["pos"][real], m({**b, "t_h": b["t_h"] + 8.0})["pos"][real]
    torch.testing.assert_close(a, s, atol=1e-5, rtol=0)


def test_time_bias_changes_output():
    m = jitter(net(cfg())).eval()
    b = seq_batch((5, 3))
    with torch.no_grad():
        assert not torch.allclose(m(b)["pos"], m({**b, "t_h": b["t_h"] * 4})["pos"])


# ---- T-M3 --------------------------------------------------------------------------------------
def test_coral_biases_stay_ordered_and_cumulative_probs_monotone():
    torch.manual_seed(0)
    head = Head(4, 8, 0.0, "ordinal", 5, [0.4, 0.3, 0.15, 0.1, 0.05])
    opt = torch.optim.SGD(head.parameters(), lr=1.0)
    for _ in range(50):  # random cumulative targets, inconsistent with any ordering
        o = head(torch.randn(32, 4))
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(o, torch.randint(0, 2, o.shape).float()).backward()
        opt.step()
        b = head.biases()
        assert (b[1:] <= b[:-1]).all()
    o = head(torch.randn(64, 4))
    p = torch.sigmoid(o)  # P(Y > k)
    assert (p[:, 1:] <= p[:, :-1] + 1e-7).all()
    torch.testing.assert_close(head.score(o), o[:, 0])  # the alarm logit is logit P(Y >= 1) = g + b_1


def test_ordinal_score_is_the_probability_of_adverse():
    """At init (zero last layer) σ(score) = P(Y >= 1) = 1 - π_0, what calibration, Brier/ECE and guid_logloss read."""
    head = Head(4, 8, 0.0, "ordinal", 3, [0.8, 0.15, 0.05])
    torch.testing.assert_close(torch.sigmoid(head.score(head(torch.randn(10, 4)))), torch.full((10,), 0.2))


# ---- heads, cues, keys, budget -----------------------------------------------------------------
def test_prior_bias_init():
    x = torch.randn(10, 4)
    pi3 = torch.tensor([0.5, 0.3, 0.2])
    close = torch.testing.assert_close
    close(torch.sigmoid(Head(4, 8, 0.0, "binary", 2, [0.7, 0.3])(x)), torch.full((10, 1), 0.3))
    close(Head(4, 8, 0.0, "multiclass", 3, pi3.tolist())(x).softmax(-1), pi3.expand(10, 3))
    close(torch.sigmoid(Head(4, 8, 0.0, "ordinal", 3, pi3.tolist())(x)), torch.tensor([0.5, 0.2]).expand(10, 2))
    m = net(cfg(model__scope="segment"), priors={"main": [0.7, 0.3], "aux3": pi3.tolist()}).eval()
    out = m(seg_batch())
    close(torch.sigmoid(out["seg_score"]), torch.full((4,), 0.3))
    close(out["aux3"].softmax(-1), pi3.expand(4, 3))
    close(out["aux3_score"], torch.full((4,), math.log(0.5 / 0.5)))


def test_attention_cues_shape_weights_but_are_not_pooled():
    m = jitter(net(cfg(model__scope="segment"))).eval()
    b = seg_batch()
    with torch.no_grad():
        h = m.step_encoder(b["x"], b["step_mask"])
        pooled = [m.pooling(h, b["step_mask"], cues) for cues in (b["attn"], b["attn"] + 3 * torch.randn(4, T, CA))]
    assert not torch.allclose(pooled[0][1], pooled[1][1])
    for e, a in pooled:
        assert e.shape == (4, 8)  # d_step: the cue channel is not in the pooled space
        torch.testing.assert_close(e, (a[..., None] * h).sum(1))
        torch.testing.assert_close(a.sum(-1), torch.ones(4))
        assert (a[~b["step_mask"]] == 0).all()


@pytest.mark.parametrize("head,task,k_out", [("binary", "adverse_vs_healthy", 1),
                                             ("multiclass", "three_class", 3), ("ordinal", "three_class", 2)])
@pytest.mark.parametrize("scope", ["segment", "sequence"])
def test_output_keys_and_shapes(scope, head, task, k_out):
    m = net(cfg(model__scope=scope, labels__head=head, labels__task=task))  # λ3 0.3, segment_head on
    if scope == "segment":
        out = m(seg_batch())
        shapes = {"seg": (4, k_out), "seg_score": (4,), "aux3": (4, 3), "aux3_score": (4,), "step_attn": (4, T)}
    else:
        out = m(seq_batch((4, 2)))
        shapes = {"pos": (2, 4, k_out), "pos_score": (2, 4), "seg": (2, 4, k_out), "seg_score": (2, 4),
                  "aux3_pos": (2, 4, 3), "aux3_pos_score": (2, 4), "aux3_seg": (2, 4, 3),
                  "aux3_seg_score": (2, 4), "step_attn": (2, 4, T)}
    assert {k: tuple(v.shape) for k, v in out.items()} == shapes


def test_optional_heads_off_and_mil_attention():
    m = net(cfg(labels__aux_3class_weight=0, model__segment_head=False, model__sequence__kind="attention_mil"))
    out = m(seq_batch((4, 2)))
    assert set(out) == {"pos", "pos_score", "step_attn", "seq_attn"}
    a = out["seq_attn"]
    assert (a.triu(1) == 0).all() and torch.allclose(a[1, :2].sum(-1), torch.ones(2))


@pytest.mark.parametrize("fusion", ["concat", "film", "token", "late"])
def test_covariate_fusion(fusion):
    m = jitter(net(cfg(), n_cov=2, fusion=fusion))
    b = seq_batch((4, 2))
    out = m({**b, "cov": torch.randn(2, 4, 2)})
    _finite(out, m)
    if fusion != "token":
        m = net(cfg(model__scope="segment"), n_cov=2, fusion=fusion)
        _finite(m({**seg_batch(), "cov": torch.randn(4, 2)}), m)


def test_non_causal_needs_final_only_or_mil():
    c = cfg()
    model_cfg = c.model.model_copy(update={"sequence": c.model.sequence.model_copy(update={"causal": False})})
    kw = dict(n_values=C, n_attn=CA, n_ctx=D, model_cfg=model_cfg)
    with pytest.raises(ValueError, match="causal"):
        ClassifierNet(**kw, labels_cfg=c.labels)
    m = ClassifierNet(**kw, labels_cfg=c.labels.model_copy(update={"strategy": "mil"}))
    assert torch.isfinite(m(seq_batch())["pos"]).all()


@pytest.mark.parametrize("head,task,loss", [("binary", "adverse_vs_healthy", "bce"),
                                            ("multiclass", "three_class", "ce"), ("ordinal", "three_class", "coral")])
@pytest.mark.parametrize("scope", ["segment", "sequence"])
def test_model_to_loss(scope, head, task, loss):
    from teb_vae.classifier.losses import compute_loss
    c = cfg(model__scope=scope, labels__head=head, labels__task=task, train__loss__name=loss)
    m = jitter(net(c))
    b = seq_batch((5, 1, 0)) if scope == "sequence" else seg_batch()
    b["y"] = b["y"] if head == "binary" else b["y3"]
    total, parts = compute_loss(m(b), b, scope=scope, labels_cfg=c.labels, train_cfg=c.train, lse_tau=c.model.lse_tau)
    total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_param_budget_default():
    assert n_params(net(cfg(small=False), n_values=129, n_attn=1, n_ctx=9)) <= 500_000
    assert n_params(net(cfg(small=False), n_values=126, n_attn=0, n_ctx=9)) <= 500_000


def test_no_forbidden_submodule_names():
    names = {part for name, _ in net(cfg()).named_modules() for part in name.split(".")}
    assert not names & {"model", "net", "network", "module"}


# ---- aggregators -------------------------------------------------------------------------------
def test_aggregate_known_answers():
    s = np.array([0.0, 2.0, -1.0, 1.0])
    expect = {"max": 2.0, "mean": 0.5, "last": 1.0, "topk_mean": 1.0,
              "lse": math.log(np.exp(s).mean())}
    for kind, value in expect.items():
        assert aggregate(s, kind) == pytest.approx(value)
        assert aggregate(torch.tensor(s), kind).item() == pytest.approx(value)
        run = running_aggregate(s, kind)
        np.testing.assert_allclose(run, [aggregate(s[: n + 1], kind) for n in range(4)])
    assert aggregate(s, "lse", tau=0.5) == pytest.approx(0.5 * math.log(np.exp(s / 0.5).mean()))
    np.testing.assert_allclose(running_aggregate(s, "max"), [0, 2, 2, 2])
    with pytest.raises(ValueError):
        aggregate(s, "noisy_or")
