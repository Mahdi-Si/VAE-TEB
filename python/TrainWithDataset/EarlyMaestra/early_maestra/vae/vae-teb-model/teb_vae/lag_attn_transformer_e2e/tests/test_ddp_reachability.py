r"""Every parameter reaches the graph, and no forward decides that by looking at a tensor.

Production trains under plain ``"ddp"`` with ``find_unused_parameters=False``, so a parameter left
out of the autograd graph makes the reducer wait forever for a gradient that never arrives -- on the
production box, after the dev box passed.

The intuitive reading of what that requires is wrong, and this package has a fresh way to get it
wrong. A parameter multiplied by an identically-**zero** tensor *is* reachable: its
``AccumulateGrad`` node fires, it receives a zeros gradient rather than ``None``, and the reducer
marks it ready. What actually breaks the contract is a parameter left *out* of the graph, which is
what a data-dependent ``if mask.any(): ...`` does -- on some ranks and not others, on some batches
and not others, so the ranks disagree about which gradients to expect and the run hangs rather than
failing.

The front end is where that temptation lives here. Its validity mask is a *tensor*, it is zero over
whole windows on real data, and the obvious-looking optimisation -- skip the gap handling when the
window is fully valid -- is precisely the branch that hangs a run. The masking is multiplicative and
unconditional instead, and both halves of this file assert that rather than assume it: the backward
half proves the front-end parameters receive gradients on a batch carrying a planted gap, and the
AST half proves no forward could have branched on any batch at all.

The walk's machinery is **imported** from the sibling's copy rather than restated. Porting its own
self-tests would test the import; what is local is the non-vacuity check, which asks whether the walk
found this package's conditionals, and the parametrisation over this package's two ``nets`` modules.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    STUB_GAP_STEP,
    make_stub_batch,
)
from teb_vae.lag_attn_transformer_rws.tests.test_ddp_reachability import (
    _forward_conditionals,
    _reads_a_tensor_value,
)

#: This package's two ``nets`` modules. Both forwards run per batch and per rank, so a branch in
#: either would be equally fatal and equally invisible. ``frontend.py`` is the new one and the one
#: with a mask to be tempted by.
_WALKED_MODULES = ("frontend.py", "model.py")

_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"


# ---------------------------------------------------------------------------------------
# Half one: a real backward
# ---------------------------------------------------------------------------------------
def _loss(model: SeqVaeLagAttnTrfE2E, seq_len: int, beta_prior: float = 0.0) -> torch.Tensor:
    """One forward and the real objective over a stub batch of the model's own geometry.

    The stub batch carries a planted weight gap, so the front ends' mask channel is genuinely
    non-constant: a reachability claim measured on a uniformly valid batch would be a claim about the
    easy case only.

    Args:
        model: The net under test.
        seq_len: Sequence length of the stub batch.
        beta_prior: Weight of the prior scale rate, so the coverage claim is exercised under the
            anchored objective as well as the historical three-term one.

    Returns:
        The scalar ``total_loss``.
    """
    batch = make_stub_batch(BATCH, seq_len)
    assert float(batch.weight[:, STUB_GAP_STEP].max()) == 0.0, "the planted gap is gone"
    out = model(batch.fhr, batch.up, batch.weight)
    return model.compute_loss(out, batch.fhr, weight=batch.weight, beta_prior=beta_prior)[
        "metrics"
    ]["total_loss"]


def _unreached(model: SeqVaeLagAttnTrfE2E) -> List[str]:
    """Names of parameters that require a gradient and did not receive one."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_every_parameter_is_reachable(tiny_kwargs, beta_prior):
    """Both anchor weights: the prior scale rate is the one objective term a config can switch on,
    and the reachability claim must hold for the objective production actually optimises."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs)

    _loss(model, SEQ_LEN, beta_prior=beta_prior).backward()

    assert not _unreached(model), (
        f"unreachable under find_unused_parameters=False: {_unreached(model)}"
    )


def test_every_front_end_parameter_receives_a_gradient_rather_than_none(tiny_kwargs):
    """Named separately from the sweep above, because the front ends are what this package added and
    their masking is what a branchy implementation would have gated. A zeros gradient counts -- that
    is the whole point -- so the assertion is ``is not None``, not ``> 0``."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs)

    _loss(model, SEQ_LEN).backward()

    for stream in ("target_frontend", "source_frontend"):
        frontend = getattr(model, stream)
        missing = [
            name for name, parameter in frontend.named_parameters() if parameter.grad is None
        ]
        assert not missing, f"{stream}: {missing}"


def test_the_probe_catches_a_parameter_that_is_genuinely_dangling(tiny_kwargs):
    """The negative control. Without it the two tests above pass on any model whose parameters all
    happen to be used, including one whose reachability nothing enforces."""

    class _DanglingModel(SeqVaeLagAttnTrfE2E):
        """Deliberately broken: one parameter that no forward reads."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.orphan = nn.Parameter(torch.zeros(4))

    torch.manual_seed(0)
    model = _DanglingModel(**tiny_kwargs)

    _loss(model, SEQ_LEN).backward()

    assert _unreached(model) == ["orphan"]


def test_the_frozen_output_projection_is_excluded_rather_than_unreachable(tiny_kwargs):
    """The lag attention's $W_o$ feeds nothing and would be permanently unreachable; clearing
    ``requires_grad`` is what keeps it out of the reducer's expectation set instead."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]

    assert frozen == ["lag_attn.W_o.weight", "lag_attn.W_o.bias"]


def test_a_fully_masked_batch_still_reaches_every_front_end_parameter(tiny_kwargs):
    """The extreme case the multiplicative masking has to survive, and the one a branch would have
    been written for: every step invalid, so the mask channel is identically zero.

    The featurisation's value and delta channels are then zero too, and the stage projections'
    biases are the only thing keeping the tokens off an exact zero -- which is why the projections
    carry a bias at all. Every front-end parameter must still receive a gradient.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs)
    batch = make_stub_batch(BATCH, SEQ_LEN)

    out = model(batch.fhr, batch.up, torch.zeros_like(batch.weight))
    # A fully invalid batch admits no anchor, so the objective is not what carries gradient here;
    # the forward's own outputs are, which is the narrower claim this test needs.
    (out["target_state"].sum() + out["source_state"].sum()).backward()

    for stream in ("target_frontend", "source_frontend"):
        frontend = getattr(model, stream)
        missing = [
            name for name, parameter in frontend.named_parameters() if parameter.grad is None
        ]
        assert not missing, f"{stream} under a fully masked batch: {missing}"


# ---------------------------------------------------------------------------------------
# Half two: no forward may branch on tensor content
# ---------------------------------------------------------------------------------------
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


@pytest.mark.parametrize("filename", _WALKED_MODULES)
def test_the_walk_finds_the_conditionals_that_are_there(filename):
    """A silently empty walk would make the test above pass on any source at all -- including one
    whose ``forward`` was renamed, or one that grew a branch inside a helper the walk does not
    follow. Both modules are known to carry shape guards inside a ``forward``."""
    source = (_NETS_DIR / filename).read_text(encoding="utf-8")

    assert _forward_conditionals(source), f"nets/{filename}: the walk saw no forward conditional"


def test_the_front_ends_mask_is_applied_without_a_branch(tiny_kwargs, raw_inputs):
    """The behavioural counterpart of the walk, stated on the mask itself.

    Multiplicative and unconditional means the *same* operations run whatever the weight holds, so a
    fully valid batch and a fully invalid one take the identical path and differ only in values. If
    the gap handling were gated, these two would not merely differ -- one of them would skip a
    module, and the parameters inside it would leave the graph.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfE2E(**tiny_kwargs).eval()
    y_raw, _u_raw, weight = raw_inputs

    with torch.no_grad():
        valid = model.target_frontend(y_raw, torch.ones_like(weight))
        invalid = model.target_frontend(y_raw, torch.zeros_like(weight))

    assert valid.shape == invalid.shape
    assert bool(torch.isfinite(valid).all()) and bool(torch.isfinite(invalid).all())
    assert not torch.equal(valid, invalid), "the mask reached nothing; the claim is empty"
