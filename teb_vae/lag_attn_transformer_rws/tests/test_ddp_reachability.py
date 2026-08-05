r"""Every parameter reaches the graph, and no forward decides that by looking at a tensor.

Production trains under plain ``"ddp"`` with ``find_unused_parameters=False``, so a parameter left
out of the autograd graph makes the reducer wait forever for a gradient that never arrives -- on
the production box, after the dev box passed.

The intuitive reading of what that requires is wrong, and getting it wrong is how the availability
mechanism would have been built. A parameter multiplied by an identically-**zero** tensor *is*
reachable: its ``AccumulateGrad`` node fires, it receives a zeros gradient rather than ``None``,
and the reducer marks it ready. What actually breaks the contract is a parameter left *out* of the
graph, which is what a data-dependent ``if indicator.any(): e = e + e_start`` does -- on some
ranks and not others, on some batches and not others, so the ranks disagree about which gradients
to expect and the run hangs rather than failing.

So this file has two halves and the second is the load-bearing one. Running a real backward and
checking that every parameter has a gradient is necessary but weak: it would pass on exactly the
branchy implementation described above, on any batch where the branch happened to be taken. The
AST walk is what rules that implementation out for every batch, on every rank, without running
one.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from teb_vae.lag_attn_transformer_rws.tests.conftest import BATCH, SEQ_LEN, make_stub_batch
from teb_vae.lag_attn_transformer_rws.tests.test_forward_contract import guarded_kwargs

#: The two modules whose forwards run per batch and per rank. ``blocks.py`` is covered by the same
#: walk because a branch introduced there would be just as fatal and just as invisible.
_WALKED_MODULES = ("encoders.py", "model.py", "blocks.py")

_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"


# ---------------------------------------------------------------------------------------
# Half one: a real backward, under both configurations
# ---------------------------------------------------------------------------------------
def _loss(model: SeqVaeLagAttnTrfRws, seq_len: int, beta_prior: float = 0.0) -> torch.Tensor:
    """One forward and the real objective over a stub batch of the model's own geometry.

    Args:
        model: The net under test.
        seq_len: Sequence length of the stub batch.
        beta_prior: Weight of the prior scale rate, so the coverage claim is exercised under
            the anchored objective as well as the historical three-term one.

    Returns:
        The scalar ``total_loss``.
    """
    batch = make_stub_batch(BATCH, seq_len)
    out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    return model.compute_loss(out, batch.fhr, weight=batch.weight, beta_prior=beta_prior)[
        "metrics"
    ]["total_loss"]


def _unreached(model: SeqVaeLagAttnTrfRws) -> List[str]:
    """Names of parameters that require a gradient and did not receive one."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_every_parameter_is_reachable_under_the_unguarded_configuration(tiny_kwargs, beta_prior):
    """Both anchor weights: the prior scale rate is the one objective term a config can switch
    on, and the reachability claim must hold for the objective production actually optimises."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs)

    _loss(model, SEQ_LEN, beta_prior=beta_prior).backward()

    assert not _unreached(model), (
        f"unreachable under find_unused_parameters=False: {_unreached(model)}"
    )


@pytest.mark.parametrize("beta_prior", [0.0, 1.0e-2], ids=["unanchored", "anchored"])
def test_every_parameter_is_reachable_under_a_real_reach_budget(tiny_kwargs, beta_prior):
    """The guarded case, and it is only the guarded case if the guard is real.

    The channel tuples come from the same budget resolution the experiment driver runs, and the
    two assertions before the backward are what stop this silently becoming a second copy of the
    unguarded test: with no gate, or with every delay at zero, neither availability parameter is
    constructed and there is nothing here the test above did not already cover.
    """
    kwargs = guarded_kwargs(tiny_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**kwargs)

    assert model.source_gate is not None and model.target_gate is not None
    assert model.source_gate.max_delay > 0 and model.target_gate.max_delay > 0
    assert model.target_adapter.mask_proj is not None
    assert model.target_adapter.start_embed is not None

    _loss(model, int(kwargs["sequence_length"]), beta_prior=beta_prior).backward()

    assert not _unreached(model), (
        f"unreachable under find_unused_parameters=False: {_unreached(model)}"
    )


def test_the_availability_parameters_receive_a_gradient_rather_than_none(tiny_kwargs):
    """Named individually, because they are the two the guarded run adds and the two a branchy
    implementation would drop. A zeros gradient counts -- that is the whole point -- so the
    assertion is ``is not None``, not ``> 0``."""
    kwargs = guarded_kwargs(tiny_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfRws(**kwargs)

    _loss(model, int(kwargs["sequence_length"])).backward()

    for stream in ("target_adapter", "source_adapter"):
        adapter = getattr(model, stream)
        assert adapter.mask_proj.weight.grad is not None, f"{stream}.mask_proj"
        assert adapter.start_embed.grad is not None, f"{stream}.start_embed"


def test_the_probe_catches_a_parameter_that_is_genuinely_dangling(tiny_kwargs):
    """The negative control. Without it the two tests above pass on any model whose parameters all
    happen to be used, including one whose reachability nothing enforces."""

    class _DanglingModel(SeqVaeLagAttnTrfRws):
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
    model = SeqVaeLagAttnTrfRws(**tiny_kwargs)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]

    assert frozen == ["lag_attn.W_o.weight", "lag_attn.W_o.bias"]


# ---------------------------------------------------------------------------------------
# Half two: no forward may branch on tensor content
# ---------------------------------------------------------------------------------------
def _forward_conditionals(source: str) -> List[Tuple[str, int, ast.expr]]:
    """Every conditional test appearing inside a ``forward`` in ``source``.

    Args:
        source: Python source text.

    Returns:
        ``(function_name, line_number, test_expression)`` per ``if`` statement and per conditional
        expression lexically inside a function named ``forward``.
    """
    found: List[Tuple[str, int, ast.expr]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "forward":
            continue
        for inner in ast.walk(node):
            if isinstance(inner, (ast.If, ast.IfExp)):
                found.append((node.name, inner.lineno, inner.test))
    return found


def _reads_a_tensor_value(test: ast.expr) -> bool:
    r"""Whether a conditional test could be reading the *content* of a tensor.

    Getting a value out of a tensor takes a call -- ``.any()``, ``.item()``, ``.sum()``,
    ``bool()`` -- or an element subscript. What is left is module state fixed at construction
    (``self.mask_proj is not None``, ``self.query_uses_logvar``, ``self.left_padding > 0``) and
    shape metadata (``x.shape[-1] != self.d_head``), neither of which can differ between ranks in
    a way that changes which parameters enter the graph. A shape guard that raises fails the run;
    it does not hang it.

    So the rule is recursive and syntactic: an expression is admitted if it is built only from
    constants, names, attributes, ``.shape`` subscripts, comparisons and boolean operators. Any
    call, any other subscript, any arithmetic anywhere inside it is rejected.

    Its one deliberate gap, stated rather than left to be discovered: a bare attribute compared
    against a constant (``self.availability > 0``) is admitted, because in this package those
    attributes are constructor-fixed scalars -- and a tensor there raises "Boolean value of Tensor
    with more than one element is ambiguous" on the first forward rather than diverging silently,
    which is the failure mode this walk exists to prevent.

    Args:
        test: The conditional's test expression.

    Returns:
        ``True`` if the test is not one of the permitted forms.
    """
    return not _is_metadata_only(test)


def _is_metadata_only(node: ast.expr) -> bool:
    """Whether an expression reads only names, attributes, constants and ``.shape``."""
    if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.Subscript):
        value = node.value
        return isinstance(value, ast.Attribute) and value.attr == "shape"
    if isinstance(node, ast.Compare):
        return _is_metadata_only(node.left) and all(
            _is_metadata_only(comparator) for comparator in node.comparators
        )
    if isinstance(node, ast.BoolOp):
        return all(_is_metadata_only(value) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        return _is_metadata_only(node.operand)
    return False


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


def test_the_walk_finds_the_conditionals_that_are_there():
    """A silently empty walk would make the test above pass on any source at all -- including one
    whose ``forward`` was renamed. Both adapter conditionals are known to exist."""
    source = (_NETS_DIR / "encoders.py").read_text(encoding="utf-8")
    conditionals = _forward_conditionals(source)

    assert len(conditionals) >= 2
    assert all(not _reads_a_tensor_value(test) for _, _, test in conditionals)


@pytest.mark.parametrize(
    "body",
    [
        "if x.any():\n            x = x + 1",
        "if bool(mask.sum() == 0):\n            x = x + 1",
        "if x[0] > 0:\n            x = x + 1",
        "x = x + 1 if x.max() > 0 else x",
        "if self.buffer.sum() > 0:\n            x = x + 1",
        "if mask.numel() and x.shape[0] > 0:\n            x = x + 1",
    ],
    ids=["any", "bool-sum", "subscript", "conditional-expression", "buffer-content",
         "call-beside-a-shape-read"],
)
def test_the_walk_flags_a_tensor_valued_branch(body):
    """The guard fires. Each of these is a real way to drop a parameter from the graph, and the
    conditional expression is included because it is the form that reads least like a branch."""
    source = f"class M:\n    def forward(self, x, mask):\n        {body}\n        return x\n"

    conditionals = _forward_conditionals(source)

    assert conditionals, "the walk did not even see the conditional"
    assert any(_reads_a_tensor_value(test) for _, _, test in conditionals)


def test_the_walk_ignores_conditionals_outside_a_forward():
    """Construction-time branching is fine and this architecture uses it: the availability terms
    are *built* conditionally and *added* unconditionally."""
    source = (
        "class M:\n"
        "    def __init__(self, delays):\n"
        "        if max(delays) > 0:\n"
        "            self.proj = 1\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )

    assert _forward_conditionals(source) == []
