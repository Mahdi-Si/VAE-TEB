"""The shared fixtures are themselves load-bearing, so they get their own tests.

Two of them can fail silently in a way that makes the rest of the suite lie:

* ``inputs`` feeds every forward in the suite. Wrong channel counts would not raise -- the
  adapters would happily build against them -- they would just stop testing the real contract.
* ``perturb_posterior`` is the only thing standing between a KL assertion and vacuous truth.
  A version that perturbed nothing would leave every KL at $0$ and every KL test green.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from teb_vae.lag_attn.tests.conftest import BATCH, PROD_KWARGS, SEQ_LEN, TINY_KWARGS


def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble must derive the repo root from this file's own depth.

    Hardcoding a nesting depth is how the equivalent shim in the tree this replaces broke: it
    assumed four levels, and a tree of a different shape would resolve it to some unrelated
    directory without ever raising.
    """
    repo_root = Path(__file__).resolve().parents[3]
    for package in ("teb_vae", "train", "utils"):
        assert (repo_root / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {repo_root}"
        )


def test_prod_kwargs_carries_no_retired_flags():
    """Smooth bounding and a residual posterior log-variance are the model, not options.

    They are not constructor arguments, so leaving them in the fixture would raise ``TypeError``
    on every construction in the suite.
    """
    for retired in ("posterior_logvar", "logvar_bound"):
        assert retired not in PROD_KWARGS


def test_prod_kwargs_extends_tiny_kwargs():
    assert all(PROD_KWARGS[key] == value for key, value in TINY_KWARGS.items())


def test_dropout_is_off():
    """Nonzero dropout would make every seeded comparison in the suite flaky."""
    assert TINY_KWARGS["dropout"] == 0.0
    assert PROD_KWARGS["dropout"] == 0.0


def test_inputs_match_the_model_input_contract(inputs):
    y_st, y_ph, u_stream = inputs
    assert y_st.shape == (BATCH, SEQ_LEN, 43)
    assert y_ph.shape == (BATCH, SEQ_LEN, 44)
    assert u_stream.shape == (BATCH, SEQ_LEN, 101)
    # c_y is the concatenated FHR stream; c_u the concatenated UP stream. Both are asserted by
    # the constructor, so a fixture that disagreed would fail every construction test with a
    # message pointing at the model rather than at here.
    assert y_st.shape[-1] + y_ph.shape[-1] == TINY_KWARGS["c_y"]
    assert u_stream.shape[-1] == TINY_KWARGS["c_u"]


def test_inputs_are_deterministic(inputs):
    generator = torch.Generator().manual_seed(0)
    expected = torch.randn(BATCH, SEQ_LEN, 43, generator=generator)
    assert torch.equal(inputs[0], expected)


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


def test_perturb_posterior_is_deterministic(perturb_posterior):
    class _StubModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.posterior_head = nn.Linear(4, 4)

    torch.manual_seed(0)
    first = _StubModel()
    torch.manual_seed(0)
    second = _StubModel()

    perturb_posterior(first)
    perturb_posterior(second)

    assert torch.equal(first.posterior_head.weight, second.posterior_head.weight)
