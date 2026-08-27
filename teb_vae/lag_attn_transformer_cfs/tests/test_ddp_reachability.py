r"""Every parameter reaches the graph under both guard states, and the availability terms are
unconditional in the forward.

``find_unused_parameters=False`` is what the shipped DDP strategy claims, and the claim is a
statement about *this* composition rather than about either parent: the decoder head is the target
domain's width, the availability projections exist only because the warm-up brought them into
existence, and the encoder is the architecture parent's. A parameter starved on any of those three
makes the reducer raise on the first production step and on no development-box run.

The two guard states are both exercised because the guarded one is the only configuration in which
the adapters carry an availability projection at all: an ungated model builds none, so a starvation
introduced by that projection would be invisible in the arm every other suite defaults to.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import List

import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask

from .conftest import (
    TINY_KWARGS,
    TINY_STRIDE,
    make_stub_batch,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)


def _starved_parameters(module, batch_idx: int) -> List[str]:
    """Backward one training step and name the trainable parameters left without a gradient."""
    module.zero_grad(set_to_none=True)
    loss, _metrics = module.compute_loss_and_metrics(make_stub_batch(4), batch_idx, "train")
    loss.backward()
    return [
        name
        for name, parameter in module.orig_model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "ungated"])
@pytest.mark.parametrize("beta_prior", [0.0, 0.1], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, guarded, beta_prior
):
    """What actually licenses ``find_unused_parameters=False``, re-earned on this architecture's
    encoders, on the widened decoder head, and on the availability terms the warm-up brings into
    existence.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step.
    """
    kwargs = (
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
        if guarded
        else dict(TINY_KWARGS, anchor_stride=TINY_STRIDE)
    )
    module = task(
        model_kwargs=kwargs,
        hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior},
    )
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under "
        f"find_unused_parameters=False the reducer raises on exactly these."
    )


@pytest.mark.parametrize("guarded", [True, False], ids=["guarded", "ungated"])
def test_under_mse_the_starved_set_is_exactly_the_decoder_logvar_head(
    task, perturb_posterior, guarded
):
    """The mirror image, and the justification for the fallback strategy: with mse the decoder
    log-variance head is trainable and unused. **Exactly** that head and nothing else -- if some
    other parameter starved here, ``find_unused_parameters=True`` would be covering for a second
    defect rather than for a documented configuration choice."""
    kwargs = (
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
        if guarded
        else dict(TINY_KWARGS, anchor_stride=TINY_STRIDE)
    )
    module = task(model_kwargs=kwargs, hparams={"likelihood": "mse"})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert set(starved) == {"decoder.logvar_head.weight", "decoder.logvar_head.bias"}, starved
    # And it is the head whose width the budget decides.
    assert (
        module.orig_model.decoder.logvar_head.bias.numel()
        == module.orig_model.decoder_out_channels
    )


def test_the_attention_projection_is_frozen_out_of_the_expectation_set(task):
    """The mechanism that removes the second starvation axis: frozen means not expected, not merely
    unused."""
    module = task()

    assert not any(
        parameter.requires_grad for parameter in module.orig_model.lag_attn.W_o.parameters()
    )


# --------------------------------------------------------------------------------------
# The start embedding: a construction-time hazard rather than a width
# --------------------------------------------------------------------------------------
def test_the_shipped_budget_builds_a_start_embedding_on_both_streams():
    r"""Both adapters build the start indicator, and that is a DDP hazard the design carries.

    This asserted the OPPOSITE until the channel alignment shipped, on the reasoning that both
    streams reach warm-up zero -- ``fhr_st`` and ``up_st`` are both honest from step $0$ -- so no
    parameter would be reached only by the leading steps of a segment. The warm-up vectors still
    reach zero, which is why the first two assertions are unchanged; but the adapter is fed
    $W'_c + d_c$, not $W'_c$, and every channel but the reference carries a strictly positive
    shift. At this cell's ``target_max`` reference the minimum is $80$, so the first eighty steps
    of every segment have no available channel at all and both adapters build the parameter.

    The consequence is the one the rest of this module exists for: under
    ``find_unused_parameters=False`` a rank whose batch never reaches those leading steps marks the
    embedding unready and the reducer raises. It is a *construction-time* change, so no shape and
    no width anywhere says it happened.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCfs(**shipped_warmup_kwargs())

    assert min(model.target_warmup_steps) == 0
    assert min(model.source_warmup_steps) == 0
    for adapter in (model.target_adapter, model.source_adapter):
        assert getattr(adapter, "start_embed", None) is not None
        assert adapter.min_delay == 80


def test_dropping_the_first_source_block_would_build_one():
    """The negative control on the test above, and the reason the driver's pre-flight refuses the
    configuration that produces it.

    Without ``up_st`` the source's fastest surviving channel waits $41$ steps, so every step below
    that has no available channel at all and the adapter builds its start embedding -- a parameter
    reached only by the leading steps of a segment. Under ``find_unused_parameters=False`` a rank
    whose batch never reaches those steps marks it unready and the reducer raises; and it is a
    *construction-time* change, so no shape and no width anywhere says it happened.
    """
    torch.manual_seed(0)
    source_only = shipped_warmup_kwargs()
    keep = [
        index for index, step in enumerate(source_only["source_warmup_steps"]) if step > 0
    ]
    # The shift vector is positional -- one entry per SURVIVING channel -- so it has to be
    # narrowed alongside the warm-up. Leaving it at full width made `ChannelDelay` refuse by
    # length rather than letting the model build, which hid what this control is measuring.
    model = SeqVaeLagAttnTrfCfs(
        **dict(
            source_only,
            c_u=len(keep),
            use_up_st=False,
            source_keep_index=tuple(range(len(keep))),
            source_warmup_steps=tuple(
                source_only["source_warmup_steps"][index] for index in keep
            ),
            source_align_delays=tuple(
                source_only["source_align_delays"][index] for index in keep
            ),
        )
    )

    assert model.source_adapter.start_embed is not None


def test_the_availability_terms_are_unconditional_in_the_forward():
    """The DDP rule the whole availability mechanism is built under: every branch is a
    construction-time decision on whether a module exists, never a runtime test of a tensor value.
    A forward that skipped the projection on the steps where the mask is all-ones would leave its
    parameter unready on exactly the ranks whose batch happened not to need it.

    Checked by walking the adapter's ``forward`` for a conditional whose test reads a tensor rather
    than an attribute's existence -- and its input validator too, which the forward calls
    unconditionally, so that moving a branch behind a call cannot move it out of this rule."""
    from teb_vae.lag_attn.nets.encoders import AvailabilityInputAdapter

    tests = []
    for member in (
        AvailabilityInputAdapter.forward,
        AvailabilityInputAdapter._validate_stream,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(member)))
        tests.extend(
            node.test
            for node in ast.walk(tree)
            if isinstance(node, (ast.If, ast.IfExp))
            # A branch that does nothing but raise is admitted, and only that shape. The hazard
            # this rule exists for is a step that *skips a projection* and leaves its parameter
            # unready on the ranks whose batch did not need it; a validator that aborts the
            # process cannot produce it, and refusing one would mean the guarded adapters -- the
            # only ones whose mask broadcasts over a wrong width -- had to be the lenient ones.
            and not (
                isinstance(node, ast.If)
                and node.body
                and all(isinstance(statement, ast.Raise) for statement in node.body)
                and not node.orelse
            )
        )

    assert tests, "the adapter's forward has no conditional at all; this test checks nothing"
    for test in tests:
        is_none_check = (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Is, ast.IsNot))
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )
        assert is_none_check, (
            f"a conditional in the adapter's forward tests something other than whether a module "
            f"was built: {ast.dump(test)}"
        )


# --------------------------------------------------------------------------------------
# The tile phase introduces no collective, and no shape is a function of the data
# --------------------------------------------------------------------------------------
def test_the_phase_derivation_runs_no_collective():
    """Each rank hashes its own samples, so nothing has to be synchronised -- which is the whole
    reason the phase is *derived* rather than drawn. A draw would need either a shared generator
    seeded identically (and then every rank would tile its different data at the same grid) or an
    all-gather inside the input builder."""
    source = inspect.getsource(SeqVaeLagAttnTrfCfsTask.anchor_phase)
    source += inspect.getsource(SeqVaeLagAttnTrfCfsTask.resolve_anchor_geometry)
    source += inspect.getsource(SeqVaeLagAttnTrfCfsTask._build_forward_inputs)

    for collective in ("all_reduce", "all_gather", "broadcast", "barrier", "dist."):
        assert collective not in source, f"the phase derivation reaches {collective}"


@pytest.mark.parametrize("phase", range(TINY_STRIDE))
def test_the_anchor_tensor_shape_is_a_geometry_constant_at_every_phase(phase):
    r"""$A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$ does not vary with $\varphi$ or with
    the batch, so no rank can disagree about a shape and no shape is a function of the data. What
    varies is how many entries are *real*, which travels in ``anchor_valid``."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCfs(**tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()
    expected = -(-(model.geometry.t_valid - model.warmup_period) // TINY_STRIDE)

    index, valid = model._build_anchor_index(
        batch=3, device=torch.device("cpu"), anchor_phase=phase, anchor_stride=TINY_STRIDE
    )

    assert index.shape == valid.shape == (3, expected)
    # Padded slots repeat the row's last real anchor, so a padded index is never a distinct legal
    # anchor -- which is what keeps the KL and reconstruction denominators from diverging.
    for row in range(3):
        real = index[row][valid[row]]
        assert len(set(real.tolist())) == int(valid[row].sum())
        if not bool(valid[row].all()):
            assert int(index[row][~valid[row]][0]) == int(real[-1])
