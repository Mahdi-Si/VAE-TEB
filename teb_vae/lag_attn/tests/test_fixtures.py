"""The one shared fixture that can make the rest of the suite lie silently.

``perturb_posterior`` is the only thing standing between a KL assertion and vacuous truth. A
version that perturbed nothing would leave every KL at $0$ and every KL test green.
"""
from __future__ import annotations

import torch
from torch import nn


def test_perturb_posterior_actually_changes_posterior_parameters(perturb_posterior):
    """The fixture is a factory; this asserts the factory's product does something."""

    class _StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.posterior_head = nn.Linear(4, 4)
            self.other_head = nn.Linear(4, 4)

    model = _StubModel()
    before = {name: p.clone() for name, p in model.named_parameters()}

    perturb_posterior(model)

    assert not torch.equal(model.posterior_head.weight, before["posterior_head.weight"])
    assert not torch.equal(model.posterior_head.bias, before["posterior_head.bias"])
    # Scoped to the posterior: perturbing the whole model would change what the KL tests mean.
    assert torch.equal(model.other_head.weight, before["other_head.weight"])


