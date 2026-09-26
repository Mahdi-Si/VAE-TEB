r"""No forward decides graph membership by looking at a tensor, and only ``W_o`` is frozen.

Production trains under plain ``"ddp"`` with ``find_unused_parameters=False``, so a parameter left
out of the autograd graph makes the reducer wait forever for a gradient that never arrives. A
parameter multiplied by an identically-**zero** tensor *is* reachable -- it receives a zeros
gradient rather than ``None`` -- so what actually breaks the contract is a data-dependent
``if mask.any(): ...``, which drops parameters on some ranks and batches and not others and hangs
the run rather than failing it. The front end is where that temptation lives: its validity mask is
a tensor that is zero over whole windows on real data.

A backward on one batch cannot rule out a branch that only fires on another, so the forwards are
walked instead; the walk's machinery is imported from the sibling's copy, where it is self-tested.
The backward evidence on real batches -- with a gap, and fully masked -- is in
``test_ddp_strategy.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_rws.tests.test_ddp_reachability import (
    _forward_conditionals,
    _reads_a_tensor_value,
)

#: This package's two ``nets`` modules. Both forwards run per batch and per rank, so a branch in
#: either would be equally fatal and equally invisible.
_WALKED_MODULES = ("frontend.py", "model.py")

_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"


def test_the_frozen_output_projection_is_the_only_frozen_parameter(tiny_kwargs):
    """The lag attention's $W_o$ feeds nothing under the head-structured posterior and would be
    permanently unreachable; clearing ``requires_grad`` keeps it out of the reducer's expectation
    set. Nothing else may be frozen, or a parameter would silently stop training."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]

    assert frozen == ["lag_attn.W_o.weight", "lag_attn.W_o.bias"]


@pytest.mark.parametrize("filename", _WALKED_MODULES)
def test_no_forward_branches_on_a_tensor_value(filename):
    source = (_NETS_DIR / filename).read_text(encoding="utf-8")
    conditionals = _forward_conditionals(source)
    offenders = [
        f"{filename}:{line} in {name}(): {ast.unparse(test)}"
        for name, line, test in conditionals
        if _reads_a_tensor_value(test)
    ]

    assert not offenders, (
        "a forward branching on tensor content drops parameters from the graph on some ranks and "
        "not others, which hangs a run under find_unused_parameters=False rather than failing "
        f"it: {offenders}"
    )
