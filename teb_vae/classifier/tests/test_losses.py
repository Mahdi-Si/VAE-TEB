"""Losses (SPEC §10.2-§10.3): known answers, weighting schemes, GUID-equal composition, guards."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from teb_vae.classifier.config import load
from teb_vae.classifier.losses import (
    PARTS, class_weights, compute_loss, coral_log_probs, criterion, loss_fn, prior_offset,
)

BASE = load().classifier
L2, L3 = math.log(2), math.log(3)


def loss_cfg(**kw):
    return BASE.train.loss.model_copy(update=kw)


def train_cfg(loss=None, **weights):
    return BASE.train.model_copy(update={"loss": loss or BASE.train.loss,
                                         "loss_weights": BASE.train.loss_weights.model_copy(update=weights)})


def labels(**kw):
    return BASE.labels.model_copy(update=kw)


def close(a, b):
    torch.testing.assert_close(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32))


def test_known_answers():
    s, y = torch.tensor([[0.0], [2.0]]), torch.tensor([1.0, 0.0])
    sp2 = math.log1p(math.exp(2))
    close(loss_fn(loss_cfg(name="bce"), "binary")(s, y), [L2, sp2])
    p2 = 1 / (1 + math.exp(-2))
    close(loss_fn(loss_cfg(name="focal", focal_alpha=0.25, focal_gamma=2.0), "binary")(s, y),
          [0.25 * 0.25 * L2, 0.75 * p2 ** 2 * sp2])
    close(loss_fn(loss_cfg(name="weighted_bce"), "binary", class_weights=[0.5, 1.5])(s, y), [1.5 * L2, 0.5 * sp2])
    close(loss_fn(loss_cfg(name="logit_adjusted", logit_adjust_tau=1.0), "binary", priors=[0.8, 0.2])(s[:1], y[:1]),
          [math.log(5)])
    close(loss_fn(loss_cfg(name="bce", label_smoothing=0.2), "binary")(s[1:], torch.ones(1)),
          [-(0.9 * math.log(p2) + 0.1 * math.log(1 - p2))])
    o, y3 = torch.zeros(2, 3), torch.tensor([0, 2])
    close(loss_fn(loss_cfg(name="ce"), "multiclass")(o, y3), [L3, L3])
    close(loss_fn(loss_cfg(name="weighted_ce"), "multiclass", class_weights=[1, 2, 3])(o, y3), [L3, 3 * L3])
    close(loss_fn(loss_cfg(name="focal_ce", focal_gamma=2.0), "multiclass")(o, y3), [(2 / 3) ** 2 * L3] * 2)
    close(loss_fn(loss_cfg(name="ce"), "multiclass")(torch.zeros(2, 4, 3), torch.zeros(2, 4, dtype=torch.long)),
          torch.full((2, 4), L3))
    coral = loss_fn(loss_cfg(name="coral"), "ordinal")
    close(coral(torch.tensor([[1.0, -1.0]]), torch.tensor([1])), [2 * math.log1p(math.exp(-1))])
    close(coral(torch.zeros(1, 2), torch.tensor([2])), [2 * L2])


def test_loss_config_errors():
    with pytest.raises(ValueError, match="focal_alpha"):
        loss_fn(loss_cfg(name="focal", focal_alpha=None), "binary")
    for name in ("auc_margin", "pauc"):
        with pytest.raises(ValueError, match="batch-level"):  # no per-element form: criterion routes it
            loss_fn(loss_cfg(name=name), "binary")
    with pytest.raises(ValueError, match="needs alpha"):
        criterion(loss_cfg(name="pauc"), "binary")
    with pytest.raises(ValueError, match="binary only"):
        criterion(loss_cfg(name="auc_margin"), "multiclass")
    with pytest.raises(ValueError, match="does not fit"):
        loss_fn(loss_cfg(name="ce"), "binary")
    with pytest.raises(ValueError, match="does not fit"):
        loss_fn(loss_cfg(name="cumulative_link"), "multiclass")
    with pytest.raises(ValueError, match="class_weights"):
        loss_fn(loss_cfg(name="weighted_bce"), "binary")


def test_cumulative_link_is_the_proportional_odds_nll():
    """-log P(Y = y), P(Y > k) = σ(o_k): p = (σ(-o_1), σ(o_1) - σ(o_2), σ(o_2)); label smoothing as in CE."""
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    o = torch.tensor([[1.0, -1.0]] * 3, requires_grad=True)
    p = [sig(-1), sig(1) - sig(-1), sig(-1)]
    ell = loss_fn(loss_cfg(name="cumulative_link"), "ordinal")(o, torch.tensor([0, 1, 2]))
    close(ell, [-math.log(q) for q in p])
    ell.sum().backward()
    assert torch.isfinite(o.grad).all()
    close(coral_log_probs(torch.tensor([[3.0, 0.5, -2.0]])).exp().sum(), 1.0)  # K = 4 sums to one
    smooth = loss_fn(loss_cfg(name="cumulative_link", label_smoothing=0.3), "ordinal")(o[:1], torch.tensor([1]))
    close(smooth, [-(0.7 * math.log(p[1]) + 0.1 * sum(math.log(q) for q in p))])
    # coral and cumulative_link share the head output; they differ in the likelihood
    assert not torch.allclose(loss_fn(loss_cfg(name="coral"), "ordinal")(o, torch.tensor([0, 1, 2])), ell)


def test_class_weights():
    close(class_weights([30, 10], "none"), [1.0, 1.0])
    close(class_weights([30, 10], "inverse"), [0.5, 1.5])
    sq = torch.tensor([30 ** -0.5, 10 ** -0.5])
    close(class_weights([30, 10], "sqrt_inverse"), 2 * sq / sq.sum())
    en = torch.tensor([0.1 / (1 - 0.9), 0.1 / (1 - 0.81)])
    close(class_weights([1, 2], "effective_number", beta=0.9), 2 * en / en.sum())


def test_prior_offset():
    assert prior_offset([1.0, 3.0]) == pytest.approx(L3)
    assert prior_offset([1.0, 1.0], priors=[0.8, 0.2], tau=1.0) == pytest.approx(math.log(4))
    close(prior_offset([1.0, 2.0, 4.0]), torch.tensor([1.0, 2.0, 4.0]).log())
    # The weighted-BCE minimiser on prevalence 1/4 sits at logit(1/4) + offset: zero gradient there.
    w = class_weights([3, 1], "inverse")
    s = torch.full((4, 1), math.log(1 / 3) + prior_offset(w), requires_grad=True)
    loss_fn(loss_cfg(name="weighted_bce"), "binary", class_weights=w)(s, torch.tensor([1.0, 0, 0, 0])).mean().backward()
    assert s.grad.sum().abs() < 1e-6


def test_segment_scope_guid_equal_weighting():
    # GUID A has 3 segments, GUID B one; batch w is pre-divided by the GUID's Σω (contract).
    s = torch.tensor([[0.0], [1.0], [2.0], [-1.0]])
    batch = {"y": torch.tensor([1.0, 1, 1, 0]), "y3": torch.tensor([1, 1, 1, -1]),
             "w": torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0])}
    total, parts = compute_loss({"seg": s, "aux3": torch.zeros(4, 3)}, batch, scope="segment",
                                labels_cfg=BASE.labels, train_cfg=BASE.train)
    ell = F.binary_cross_entropy_with_logits(s[:, 0], batch["y"], reduction="none")
    assert tuple(parts) == PARTS
    close(parts["loss_positions"], (ell[:3].mean() + ell[3]) / 2)
    close(parts["loss_aux3"], L3)  # the y3 = -1 row is masked out
    close(total, parts["loss_positions"] + 0.3 * L3)
    assert parts["loss_final"] == parts["loss_bag"] == parts["loss_segment"] == 0


def _seq(lengths, w_rows, seed=0):
    b, n = len(lengths), max(lengths)
    g = torch.Generator().manual_seed(seed)
    mask = torch.arange(n)[None] < torch.tensor(lengths)[:, None]
    big = torch.full((b, n, 1), 1e3)  # pads hold junk that must be ignored
    out = {"pos": torch.where(mask[..., None], torch.randn(b, n, 1, generator=g), big),
           "seg": torch.where(mask[..., None], torch.randn(b, n, 1, generator=g), big),
           "aux3_pos": torch.randn(b, n, 3, generator=g), "aux3_seg": torch.randn(b, n, 3, generator=g)}
    w = torch.zeros(b, n)
    for i, row in enumerate(w_rows):
        w[i, : len(row)] = torch.tensor(row)
    batch = {"seg_mask": mask, "w": w, "y": (torch.arange(b) % 2 == 0).float(), "y3": torch.arange(b) % 3}
    return out, batch


def _bce(s, y):
    return F.binary_cross_entropy_with_logits(s, torch.as_tensor(y, dtype=torch.float32).expand_as(s), reduction="none")


def test_sequence_scope_guid_equal_and_zero_weight_guard():
    # GUID 2 has Σω = 0: it enters the final term only. GUID 3 is a fully padded row.
    out, batch = _seq((4, 2, 3, 0), [[1, 1, 0.5, 0.5], [1, 1], [0, 0, 0]])
    total, parts = compute_loss(out, batch, scope="sequence", labels_cfg=labels(aux_3class_weight=0.0),
                                train_cfg=BASE.train)
    y = batch["y"]
    final = torch.stack([_bce(out["pos"][g, n - 1, 0], y[g]) for g, n in enumerate((4, 2, 3))]).mean()

    def per_guid(key):
        a = (_bce(out[key][0, :, 0], y[0]) * batch["w"][0]).sum() / 3
        return (a + _bce(out[key][1, :2, 0], y[1]).mean()) / 2

    close(parts["loss_final"], final)
    close(parts["loss_positions"], per_guid("pos"))
    close(parts["loss_segment"], per_guid("seg"))
    assert parts["loss_bag"] == parts["loss_aux3"] == 0
    close(total, final + 0.5 * per_guid("pos") + 0.3 * per_guid("seg"))


def test_mil_bag():
    out, batch = _seq((3, 2, 0), [])  # mil: cohort ω = 0 everywhere
    total, parts = compute_loss(out, batch, scope="sequence", labels_cfg=labels(strategy="mil", aux_3class_weight=0.0),
                                train_cfg=BASE.train, lse_tau=0.5)
    y = batch["y"]
    bag = torch.stack([_bce(0.5 * torch.log(torch.exp(out["seg"][g, :n, 0] / 0.5).mean()), y[g])
                       for g, n in enumerate((3, 2))]).mean()
    close(parts["loss_bag"], bag)  # λ_bag = 0 in the config -> weight 1 under mil
    assert parts["loss_positions"] == parts["loss_segment"] == 0
    close(total, parts["loss_final"] + bag)


def test_sequence_aux_and_parts_finite():
    out, batch = _seq((4, 1, 0), [[1, 1, 1, 1], [1]])
    batch["y3"] = torch.tensor([2, -1, 0])
    total, parts = compute_loss(out, batch, scope="sequence", labels_cfg=BASE.labels,
                                train_cfg=train_cfg(bag=0.2), lse_tau=1.0)
    assert tuple(parts) == PARTS and torch.isfinite(total) and all(torch.isfinite(v) for v in parts.values())
    ce = lambda o, t: F.cross_entropy(o, torch.full((len(o),), t), reduction="none")
    only_g0 = (1.0 * ce(out["aux3_pos"][0, 3:], 2).mean() + 0.5 * ce(out["aux3_pos"][0], 2).mean()
               + 0.2 * ce(torch.logsumexp(out["aux3_seg"][0], 0, keepdim=True) - math.log(4), 2).mean()
               + 0.3 * ce(out["aux3_seg"][0], 2).mean())
    close(parts["loss_aux3"], only_g0)  # y3 = -1 GUID and the padded row are masked out


def _logit(h):
    return torch.logit(torch.as_tensor(h, dtype=torch.float32))[:, None]


def test_auc_losses_known_answers_gradients_and_single_class_batches():
    """AUC-M: E₊(h - a)² + E₋(h - b)² + (1 - a + b)₊² at a, b = the class means; one-way pAUC: the squared hinge over
    positives × the top ⌈α·|N|⌉ negatives. h = σ(s)."""
    h, y, w = [0.8, 0.6, 0.2, 0.4], torch.tensor([1.0, 1, 0, 0]), torch.ones(4)
    aucm, pauc = criterion(loss_cfg(name="auc_margin"), "binary"), criterion(loss_cfg(name="pauc"), "binary", alpha=0.5)
    close(aucm(_logit(h), y, w), 0.01 + 0.01 + 0.6 ** 2)
    close(pauc(_logit(h), y, w), (0.6 ** 2 + 0.8 ** 2) / 2)  # k = 1: the negative at 0.4 only
    close(criterion(loss_cfg(name="pauc"), "binary", alpha=1.0)(_logit(h), y, w),
          sum((1 - (p - n)) ** 2 for p in (0.8, 0.6) for n in (0.2, 0.4)) / 4)
    close(aucm(_logit(h), y, torch.tensor([1.0, 0, 1, 1])), 0.0 + 0.01 + (1 - (0.8 - 0.3)) ** 2)  # w = 0 drops a row
    for crit in (aucm, pauc):
        s = _logit(h).requires_grad_()
        crit(s, y, w).backward()
        assert torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
        for one in (torch.ones(4), torch.zeros(4)):  # a population without both classes: exactly 0, finite grads
            s = _logit(h).requires_grad_()
            loss = crit(s, one, w)
            loss.backward()
            assert loss.item() == 0.0 and torch.isfinite(s.grad).all()
        assert crit(_logit([0.9, 0.8, 0.1, 0.2]), y, w) < crit(_logit([0.1, 0.2, 0.9, 0.8]), y, w)  # ranked lower


@pytest.mark.parametrize("name", ["auc_margin", "pauc"])
def test_auc_losses_fit_a_ranking_through_compute_loss(name):
    """A short fit (segment scope, both classes per batch): a linear score learns the planted ranking."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 4, generator=g)
    y = (x[:, 0] + 0.3 * torch.randn(256, generator=g) > 0).float()
    lin = torch.nn.Linear(4, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    tr = BASE.train.model_copy(update={"loss": loss_cfg(name=name), "sampler": "class_balanced"})
    batch = {"y": y, "y3": torch.full((256,), -1), "w": torch.ones(256)}
    for _ in range(100):
        opt.zero_grad()
        total, _ = compute_loss({"seg": lin(x)}, batch, scope="segment", labels_cfg=labels(aux_3class_weight=0.0),
                                train_cfg=tr, alpha=0.3)
        total.backward()
        opt.step()
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y.numpy(), lin(x).detach()[:, 0].numpy()) > 0.9
