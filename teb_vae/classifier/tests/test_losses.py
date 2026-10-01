"""Losses (SPEC §10.2-§10.3): known answers, weighting schemes, GUID-equal composition, guards."""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.classifier.config import load
from teb_vae.classifier.losses import (
    class_weights, loss_fn, prior_offset,
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


def test_prior_offset():
    assert prior_offset([1.0, 3.0]) == pytest.approx(L3)
    assert prior_offset([1.0, 1.0], priors=[0.8, 0.2], tau=1.0) == pytest.approx(math.log(4))
    close(prior_offset([1.0, 2.0, 4.0]), torch.tensor([1.0, 2.0, 4.0]).log())
    # The weighted-BCE minimiser on prevalence 1/4 sits at logit(1/4) + offset: zero gradient there.
    w = class_weights([3, 1], "inverse")
    s = torch.full((4, 1), math.log(1 / 3) + prior_offset(w), requires_grad=True)
    loss_fn(loss_cfg(name="weighted_bce"), "binary", class_weights=w)(s, torch.tensor([1.0, 0, 0, 0])).mean().backward()
    assert s.grad.sum().abs() < 1e-6
