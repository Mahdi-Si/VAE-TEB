r"""Every parameter reaches the graph, at the net layer, under both channel guards.

Production trains under ``find_unused_parameters=False``, so a parameter left out of the autograd
graph makes the reducer wait forever for a gradient that never arrives -- on the production box, after
the dev box passed. ``tests/test_ddp_strategy.py`` asks the same question through the task, which is
where the objective's weights and the likelihood live; this file asks it of the net, which is where the
*availability* parameters live, and those exist only when a reach budget is configured.

The intuitive reading of what the contract requires is wrong, and getting it wrong is how the
availability mechanism would have been built. A parameter multiplied by an identically-**zero** tensor
*is* reachable: its ``AccumulateGrad`` node fires, it receives a zeros gradient rather than ``None``,
and the reducer marks it ready. What actually breaks the contract is a parameter left *out* of the
graph, which is what a data-dependent ``if indicator.any(): e = e + e_start`` does -- on some ranks and
not others, on some batches and not others, so the ranks disagree about which gradients to expect and
the run hangs rather than failing.

The AST walk that rules that implementation out for every batch is **deliberately not ported**. It is
a statement about the two ``forward`` methods that run per batch, and both belong to modules this
package imports rather than writes: the encoders and blocks are
``teb_vae/lag_attn_transformer_rws/nets/``, walked by that package's own copy, and the
``AvailabilityInputAdapter`` whose two conditionals the walk exists to police is in the shared
``teb_vae/lag_attn/nets/encoders.py``, walked there too. What this file asserts instead is the premise
that makes those walks sufficient here: **this package's ``nets/`` defines no ``forward`` at all**, so
there is no per-batch branch of its own for a walk to find.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs

from .conftest import BATCH, SEQ_LEN, make_patterned_batch, resolve_target_budget

#: Sequence and warm-up lengths the guarded probe below runs at. Longer than the tiny fixture's,
#: because the production budget's own resolution refuses a delay longer than the warm-up and its
#: worst delay is exactly $30$ steps.
_GUARDED_SEQ_LEN = 64
_GUARDED_WARMUP = 30

#: This package's net layer. Two files, and between them they define no ``forward``: the model class
#: is an empty body and the mixin owns the width hook, the target gather, the four gap splits and the
#: objective delegation, none of which is a forward.
_NETS_DIR = Path(__file__).resolve().parents[1] / "nets"

#: The mixin's own module, which lives in the feature-domain sibling's ``nets/`` because that is where
#: the feature target was defined, measured and tested. Named here because it is half of what this
#: package's net layer *is*, and a reader looking for the second file would otherwise not find it.
_MIXIN_MODULE = (
    Path(__file__).resolve().parents[2] / "lag_attn_fs" / "nets" / "feature_target.py"
)


def _guarded_kwargs(tiny_kwargs: dict) -> dict:
    """The tiny keyword set carrying the **production** reach budget's resolved channel tuples.

    Resolved rather than hand-written, following the conv-Transformer sibling's own probe. The tiny
    fixture's hand-made guard has a zero in its delay tuple, and a zero minimum is exactly the case
    in which the adapter builds **no start embedding** -- there is no zero-filled prefix for one to
    fill -- so a probe built on it would assert the availability mechanism while exercising half of
    it. The production budget's smallest delay is positive, which is what puts both availability
    parameters in the graph.

    Args:
        tiny_kwargs: The tiny constructor keyword set.

    Returns:
        A new keyword set with the four channel tuples merged in, at a sequence and warm-up length
        the budget's own resolution accepts.
    """
    budget = resolve_target_budget()
    assert budget is not None
    return dict(
        tiny_kwargs,
        sequence_length=_GUARDED_SEQ_LEN,
        warmup_period=_GUARDED_WARMUP,
        target_keep_index=budget.target_keep_index,
        target_delays=budget.target_delays,
        source_keep_index=budget.source_keep_index,
        source_delays=budget.source_delays,
    )


def _loss(model: SeqVaeLagAttnTrfFs, batch, beta_prior: float = 0.0) -> torch.Tensor:
    """One forward and the real objective over a patterned batch of the model's own geometry.

    Args:
        model: The net under test.
        batch: The batch to run on; its two target blocks are the forecast target.
        beta_prior: Weight of the prior scale rate, so the coverage claim is exercised under the
            anchored objective as well as the historical three-term one.

    Returns:
        The scalar ``total_loss``.
    """
    outs = model(
        batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
    )
    target = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
    return model.compute_loss(outs, target, weight=batch.weight, beta_prior=beta_prior)[
        "metrics"
    ]["total_loss"]


def _unreached(model: nn.Module) -> List[str]:
    """Names of parameters that require a gradient and did not receive one."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


# ---------------------------------------------------------------------------------------
# A real backward, at both channel guards and under both objectives
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("beta_prior", [0.0, 0.1], ids=["unanchored", "anchored"])
def test_every_parameter_is_reachable_under_the_unguarded_configuration(tiny_kwargs, beta_prior):
    """Both anchor weights: the prior scale rate is the one objective term a config can switch on, and
    the reachability claim must hold for the objective production actually optimises."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**tiny_kwargs)

    _loss(model, make_patterned_batch(BATCH, SEQ_LEN), beta_prior=beta_prior).backward()

    assert not _unreached(model), (
        f"unreachable under find_unused_parameters=False: {_unreached(model)}"
    )


@pytest.mark.parametrize("beta_prior", [0.0, 0.1], ids=["unanchored", "anchored"])
def test_every_parameter_is_reachable_under_a_real_channel_guard(tiny_kwargs, beta_prior):
    """The guarded case, and it is only the guarded case if the guard is real.

    The five assertions before the backward are what stop this silently becoming a second copy of the
    test above: with no gate, with every delay at zero, or with a *minimum* delay of zero, one or both
    availability parameters are never constructed and there is nothing here the unguarded test did not
    already cover. The guard also narrows the decoder to the budget's survivors, so this is the one arm
    where the availability adapters and a $C_{\\mathrm{keep}}$-wide output head are exercised together
    -- which is the pairing neither sibling has run.
    """
    kwargs = _guarded_kwargs(tiny_kwargs)
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**kwargs)

    assert model.source_gate is not None and model.target_gate is not None
    assert model.source_gate.max_delay > 0 and model.target_gate.max_delay > 0
    assert model.target_adapter.mask_proj is not None
    assert model.target_adapter.start_embed is not None
    assert model.decoder_out_channels == len(kwargs["target_keep_index"]) == 78

    batch = make_patterned_batch(BATCH, _GUARDED_SEQ_LEN)
    _loss(model, batch, beta_prior=beta_prior).backward()

    assert not _unreached(model), (
        f"unreachable under find_unused_parameters=False: {_unreached(model)}"
    )


def test_the_availability_parameters_receive_a_gradient_rather_than_none(tiny_kwargs):
    """Named individually, because they are the two the guarded run adds and the two a branchy
    implementation would drop. A zeros gradient counts -- that is the whole point -- so the assertion
    is ``is not None``, not ``> 0``."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**_guarded_kwargs(tiny_kwargs))

    _loss(model, make_patterned_batch(BATCH, _GUARDED_SEQ_LEN)).backward()

    for stream in ("target_adapter", "source_adapter"):
        adapter = getattr(model, stream)
        assert adapter.mask_proj.weight.grad is not None, f"{stream}.mask_proj"
        assert adapter.start_embed.grad is not None, f"{stream}.start_embed"


def test_the_widened_decoder_heads_receive_a_gradient(tiny_kwargs):
    """The tensors the target domain changed, named because they are the group the DDP strategy's one
    config-decided starvation axis is about. Under ``gaussian_nll`` -- the shipped likelihood -- both
    heads are in the graph; ``tests/test_ddp_strategy.py`` shows the ``mse`` mirror image."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**tiny_kwargs)

    _loss(model, make_patterned_batch(BATCH, SEQ_LEN)).backward()

    assert model.decoder.mean_head.weight.grad is not None
    assert model.decoder.logvar_head.weight.grad is not None
    assert model.decoder.mean_head.out_features == model.decoder_out_channels


def test_the_probe_catches_a_parameter_that_is_genuinely_dangling(tiny_kwargs):
    """The negative control. Without it the tests above pass on any model whose parameters all happen
    to be used, including one whose reachability nothing enforces."""

    class _DanglingModel(SeqVaeLagAttnTrfFs):
        """Deliberately broken: one parameter that no forward reads."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.orphan = nn.Parameter(torch.zeros(4))

    torch.manual_seed(0)
    model = _DanglingModel(**tiny_kwargs)

    _loss(model, make_patterned_batch(BATCH, SEQ_LEN)).backward()

    assert _unreached(model) == ["orphan"]


def test_the_frozen_output_projection_is_excluded_rather_than_unreachable(tiny_kwargs):
    """The lag attention's $W_o$ feeds nothing and would be permanently unreachable; clearing
    ``requires_grad`` is what keeps it out of the reducer's expectation set instead."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**tiny_kwargs)

    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]

    assert frozen == ["lag_attn.W_o.weight", "lag_attn.W_o.bias"]


# ---------------------------------------------------------------------------------------
# Why the sibling walks are sufficient here
# ---------------------------------------------------------------------------------------
def _defines_forward(source: str) -> bool:
    """Whether ``source`` defines a function named ``forward`` anywhere."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        for node in ast.walk(ast.parse(source))
    )


def test_this_packages_net_layer_defines_no_forward():
    """The premise that makes the siblings' AST walks sufficient rather than merely elsewhere.

    Those walks reject a ``forward`` that branches on tensor content, because such a branch drops
    parameters from the graph on some ranks and not others and hangs a run rather than failing it.
    Every ``forward`` that executes here belongs to a module this package imports -- the encoders and
    blocks from the conv-Transformer sibling, the availability adapter from the shared net layer -- and
    both are walked where they live. If a ``forward`` ever appears in this package's own ``nets/``,
    that reasoning stops holding and this test is what says so.
    """
    local = sorted(_NETS_DIR.glob("*.py"))

    assert local, "the net directory is empty; this glob is checking nothing"
    offenders = [
        path.name for path in local if _defines_forward(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        f"{offenders} define a forward, so the sibling packages' tensor-branch walks no longer cover "
        f"every per-batch code path this model runs"
    )
    # And the mixin, which is the other half of this model's net layer even though it lives in the
    # feature-domain sibling's package: it owns no forward either.
    assert not _defines_forward(_MIXIN_MODULE.read_text(encoding="utf-8"))


def test_the_walks_that_do_cover_this_model_exist_where_they_are_claimed_to():
    """A cross-reference asserted rather than written in prose: the two files whose walks this package
    leans on are real, and each really does run the check. A renamed or deleted module would leave the
    test above true and the coverage argument false."""
    walkers = (
        Path(__file__).resolve().parents[2]
        / "lag_attn_transformer_rws"
        / "tests"
        / "test_ddp_reachability.py"
    )

    source = walkers.read_text(encoding="utf-8")

    assert walkers.is_file()
    assert "test_no_forward_branches_on_a_tensor_value" in source
    # It must reach the SHARED net layer, where the availability adapter's two conditionals are: a
    # walk scoped to that package's own nets/ would pass by finding nothing.
    assert "_SHARED_NETS_DIR" in source
