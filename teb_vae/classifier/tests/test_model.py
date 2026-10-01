"""ClassifierNet (SPEC §9, §15 T-M1-T-M3) on synthetic contract batches."""
from __future__ import annotations


import pytest
import torch

from teb_vae.classifier.config import load
from teb_vae.classifier.model import ClassifierNet, Head

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


# ---- heads, cues, keys, budget -----------------------------------------------------------------
# ---- aggregators -------------------------------------------------------------------------------
