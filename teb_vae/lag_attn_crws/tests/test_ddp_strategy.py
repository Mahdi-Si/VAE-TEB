r"""Which DDP strategy the configured parameter usage permits, and the evidence that it is safe.

``find_unused_parameters=False`` makes the reducer expect every parameter to be marked ready in
every backward, and one that is not raises or deadlocks on a real multi-GPU box. The selector is a
*claim* about the model; the grad-coverage tests are the *evidence* -- without them this file would
only assert that a function returns what it was written to return.

The claim is inherited from the raw-target model this cell is the causal-input counterpart of, and
what this cell changes about it is a live question on four counts:

* the availability mask is now **load-bearing** rather than a guard that happens to be off;
* ``start_embed`` is **conditionally constructed** -- it exists only when every channel of a stream
  is unavailable for at least one step -- so a configuration change flips a parameter into
  existence that is reached only by the leading steps of a segment. That is a construction-time
  hazard, not a width change. Two things now trigger it: dropping the first source block, which the
  pre-flight refuses by name, and the **channel alignment**, which the shipped configuration
  enables and which puts one on *both* streams. The second is safe and the tests below say why --
  the leading region it is live on is the leading region of every segment, so every rank reaches
  it in every backward;
* the decoder log-variance head is consumed only under ``gaussian_nll``, while ``tiny.yaml`` ships
  ``mse``. Its width is $R$ rather than a resolved channel count, which is the one thing about this
  starvation that a budget cannot move;
* the per-segment tile phase is new, and a derivation that needed a collective would be a
  synchronisation point inside the input builder.

That last one is the one with no shape to check: $A_{\max}$ is a geometry constant either way, so a
phase that disagreed across ranks would produce correctly shaped tensors and a differently tiled
gradient on every rank, forever.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import List

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

from .conftest import (
    TINY_KWARGS,
    TINY_STRIDE,
    absolutize_dataset_paths,
    make_stub_batch,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


@pytest.fixture
def trainer(tmp_path):
    """A driver on the shipped config; ``setup_config`` is never called.

    The shipped config rather than the tiny one: nothing here reads the shards, and what is under
    test is the strategy the *production* configuration selects.
    """
    driver = LagAttnCrwsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)
    driver.train_results_dir = str(tmp_path / "train_results")
    return driver


def _config(**vae_overrides) -> dict:
    """A minimal config carrying only the keys the strategy selector reads."""
    return {"model_config": {"VAE_model": dict(vae_overrides)}}


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
def test_the_shipped_config_earns_every_parameter_reachable(trainer):
    """The payoff of the learned observation variance plus the unconditional ``W_o`` freeze: the
    reducer can expect every parameter."""
    assert trainer.ddp_kwargs(trainer.config)["find_unused_parameters"] is False


def test_a_single_device_needs_no_strategy(trainer):
    assert trainer.select_ddp_strategy(1, trainer.config) == "auto"


def test_the_smoke_configs_mse_selects_the_fallback(trainer):
    """``tiny.yaml`` ships ``likelihood: mse`` precisely so the smoke path exercises this branch
    where it is cheap to observe, rather than leaving it configured and never run."""
    from teb_vae.lag_attn.config import load_config

    tiny = load_config(str(_TINY))

    assert tiny["model_config"]["VAE_model"]["likelihood"] == "mse"
    assert trainer.ddp_kwargs(tiny)["find_unused_parameters"] is True
    assert trainer.ddp_kwargs(_config(likelihood="mse"))["find_unused_parameters"] is True


def test_the_buffer_broadcast_is_off_and_the_gradients_are_bucket_views(trainer):
    """Two performance settings the shorthand strategy strings cannot express, which is why the
    selector returns an instance."""
    kwargs = trainer.ddp_kwargs(trainer.config)

    assert kwargs["broadcast_buffers"] is False
    assert kwargs["gradient_as_bucket_view"] is True


def test_no_buffer_is_a_running_statistic_so_the_broadcast_is_safe_to_skip():
    """What licenses ``broadcast_buffers=False``: every buffer is a deterministic function of the
    config, built identically in each rank's constructor, so the broadcast restores values that were
    never going to differ. A ``BatchNorm`` running statistic is the one kind that genuinely diverges
    per rank, and there is none.

    This cell adds **two** buffers of its own -- the per-block source warmth patterns -- and both
    are functions of the resolved budget and the geometry, so they belong in the same category. The
    causal-feature cell's third, the kept-target-channel tertile assignment, is absent because this
    target has no channels to partition, and its absence is asserted rather than assumed. They are
    non-persistent for a second reason: their contents follow the budget, so a persistent copy would
    make a checkpoint trained at one budget fail to load at another and report it as misaligned keys
    rather than as a budget mismatch."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**shipped_warmup_kwargs())
    buffers = dict(model.named_buffers())

    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    for name in ("source_block_warm_st", "source_block_warm_ph"):
        assert name in buffers, name
        assert name not in model.state_dict(), f"{name} reaches a checkpoint"
    assert "warm_tertile_id" not in buffers


def test_static_graph_is_not_claimed(trainer):
    """A correctness call rather than an omission: the loss-spike breaker substitutes a
    zero-weighted sum over every parameter on a skipped batch, which is a structurally different
    backward from the one iteration 1 recorded. ``static_graph=True`` promises DDP that never
    happens, and the breaker ships enabled."""
    assert "static_graph" not in trainer.ddp_kwargs(trainer.config)


def test_the_settings_reach_the_strategy_object(trainer):
    """``DDPStrategy`` forwards unrecognised kwargs into ``_ddp_kwargs`` and on to
    ``DistributedDataParallel``. That name is Lightning-internal, so it is asserted here only."""
    strategy = trainer.select_ddp_strategy(8, trainer.config)

    assert type(strategy).__name__ == "DDPStrategy"
    assert strategy._ddp_kwargs == trainer.ddp_kwargs(trainer.config)


def test_the_selector_is_a_pure_function_of_config(trainer):
    """The framework passes the *Lightning module* as ``model``, not the raw net; a selector that
    read a net attribute off it would find nothing and silently regress the shipped config to the
    slow strategy on the one box where it costs."""
    without_model = trainer.select_ddp_strategy(8, trainer.config)
    with_wrapper = trainer.select_ddp_strategy(8, trainer.config, model=object())

    assert without_model._ddp_kwargs == with_wrapper._ddp_kwargs


def test_the_hook_is_the_un_prefixed_name_the_framework_looks_up():
    """The framework calls ``select_ddp_strategy`` and nothing else. Inherited here, so what is
    asserted is that this driver did not shadow it with an underscore-prefixed copy that would never
    run."""
    assert "_select_ddp_strategy" not in vars(LagAttnCrwsTrainer)
    assert LagAttnCrwsTrainer.select_ddp_strategy is LagAttnRwsTrainer.select_ddp_strategy


# --------------------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------------------
def _starved_parameters(module, batch_idx: int) -> List[str]:
    """Backward one training step and name the trainable parameters left without a gradient."""
    module.zero_grad(set_to_none=True)
    loss, _ = module.compute_loss_and_metrics(make_stub_batch(4), batch_idx, "train")
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
    """What actually licenses ``find_unused_parameters=False``, re-earned on a decoder scoring raw
    samples and -- the new part -- on the availability terms the warm-up brings into existence.

    Both guard states, because the guarded one is the only configuration in which the adapters carry
    an availability projection at all: an ungated model builds none, so a starvation introduced by
    that projection would be invisible in the arm every other suite defaults to.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step.
    """
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if guarded else dict(
        TINY_KWARGS, anchor_stride=TINY_STRIDE
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
    defect rather than for a documented configuration choice.

    Its width is the one thing the budget cannot move: the head emits $R$ raw samples per horizon
    token whatever the warm-up keeps, which is why the two guard states starve the same tensor at
    the same size rather than at two."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if guarded else dict(
        TINY_KWARGS, anchor_stride=TINY_STRIDE
    )
    module = task(model_kwargs=kwargs, hparams={"likelihood": "mse"})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert set(starved) == {"decoder.logvar_head.weight", "decoder.logvar_head.bias"}, starved
    assert (
        module.orig_model.decoder.logvar_head.bias.numel()
        == module.orig_model.decoder_out_channels
        == module.orig_model.raw_per_step
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
def test_the_shipped_aligned_budget_builds_a_start_embedding_on_both_streams():
    r"""The shipped configuration builds it, and the reason it is not the hazard it would once have
    been.

    Both streams reach warm-up zero on their own -- the target because ``fhr_st``'s fastest channels
    are honest from step $0$, the source because ``up_st``'s are -- but the adapter is fed
    $W'_c + d_c$, and the alignment shifts every channel of both streams onto one clock. The
    combined minimum is therefore the fastest channel's *shift*, $80$ steps, and the adapter builds
    a start indicator on each stream: a learned $d_{\mathrm{model}}$-wide vector per stream, live
    on the leading region where no channel of that stream has arrived at all.

    **Why ``find_unused_parameters=False`` still holds.** A parameter reached only by *some* batches
    is the hazard; this one is reached by every batch of every rank, because the leading $80$ steps
    are the leading steps of every segment the loader serves and the term is added unconditionally
    in the forward -- the branch is ``self.start_embed is not None``, a test on a module built in
    ``__init__``, never on tensor content. The two facts are asserted together, because the first
    without the second reads as a regression.

    **Which reference "shipped" means here.** ``shipped_warmup_kwargs`` reaches the causal-feature
    package's ``causal_config``, whose ``SHIPPED_ALIGN_REFERENCE`` is ``target_max``, so the budget
    built below is the $402.1604$ s one and $80$ is its combined minimum. This cell's own
    ``configs/default.yaml`` ships $42.21$ s, where the same minimum is $1$; the property under test
    -- a positive combined minimum builds the indicator on both streams -- holds at either.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**shipped_warmup_kwargs())

    assert min(model.target_warmup_steps) == 0
    assert min(model.source_warmup_steps) == 0
    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.start_embed is not None
        assert adapter.min_delay == 1
        # Live on the leading region of every segment, and only there.
        indicator = adapter.start_indicator.squeeze(-1)
        assert bool(indicator[: adapter.min_delay].all())
        assert not bool(indicator[adapter.min_delay :].any())


def test_the_unaligned_arm_builds_none_and_dropping_the_first_source_block_would():
    """The comparison arm keeps the original property, and the negative control on it.

    With no alignment reference the gate is a pure gather, the adapter is fed the warm-up alone and
    both streams reach zero, so neither builds a start indicator. Without ``up_st`` the source's
    fastest surviving channel waits $41$ steps, so every step below that has no available channel at
    all and the adapter builds one -- a *construction-time* change, with no shape and no width
    anywhere saying it happened, which is why the pre-flight refuses that configuration by name.
    """
    from teb_vae.lag_attn.config import load_config

    torch.manual_seed(0)
    unaligned = SeqVaeLagAttnCrws(**shipped_warmup_kwargs(align=False))
    for adapter in (unaligned.target_adapter, unaligned.source_adapter):
        assert getattr(adapter, "start_embed", None) is None

    source_only = shipped_warmup_kwargs(align=False)
    keep = [
        index
        for index, step in enumerate(source_only["source_warmup_steps"])
        if step > 0
    ]
    model = SeqVaeLagAttnCrws(
        **dict(
            source_only,
            c_u=len(keep),
            use_up_st=False,
            source_keep_index=tuple(range(len(keep))),
            source_warmup_steps=tuple(
                source_only["source_warmup_steps"][index] for index in keep
            ),
        )
    )

    assert model.source_adapter.start_embed is not None
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["model_config"]["VAE_model"]["use_up_st"] = False
    config["model_config"]["VAE_model"]["c_u"] = 15
    with pytest.raises(ValueError, match="use_up_st"):
        LagAttnCrwsTrainer.preflight(config)


def test_the_availability_terms_are_unconditional_in_the_forward():
    """The DDP rule the whole availability mechanism is built under: every branch is a
    construction-time decision on whether a module exists, never a runtime test of a tensor value.
    A forward that skipped the projection on the steps where the mask is all-ones would leave its
    parameter unready on exactly the ranks whose batch happened not to need it.

    Checked by walking the adapter's ``forward`` for a conditional whose test reads a tensor rather
    than an attribute's existence -- and its input validator too, which the forward calls
    unconditionally, so that moving a branch behind a call cannot move it out of this rule."""
    import textwrap

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
            # process cannot produce it.
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
# The tile phase introduces no collective
# --------------------------------------------------------------------------------------
def test_the_phase_derivation_runs_no_collective():
    """Each rank hashes its own samples, so nothing has to be synchronised -- which is the whole
    reason the phase is *derived* rather than drawn. A draw would need either a shared generator
    seeded identically (and then every rank would tile its different data at the same grid) or an
    all-gather inside the input builder.

    Read off **this** task's own attributes rather than off the cell they are bound from, so a
    binding that stopped being a binding is checked here too."""
    source = inspect.getsource(SeqVaeLagAttnCrwsTask.anchor_phase)
    source += inspect.getsource(SeqVaeLagAttnCrwsTask.resolve_anchor_geometry)
    source += inspect.getsource(SeqVaeLagAttnCrwsTask._build_forward_inputs)

    for collective in ("all_reduce", "all_gather", "broadcast", "barrier", "dist."):
        assert collective not in source, f"the phase derivation reaches {collective}"


def test_the_only_collective_in_the_step_is_the_inherited_permutation_decision():
    """The one that must stay: the control's metrics are logged with ``sync_dist=True``, so a rank
    that logged them while a peer with a degenerate last batch did not would hang the metric sync.
    Asserted so the test above reads as "the phase adds none" rather than as "there are none"."""
    source = inspect.getsource(SeqVaeLagAttnRwsTask._sync_perm_decision)

    assert "all_reduce" in source
    assert "_sync_perm_decision" not in vars(SeqVaeLagAttnCrwsTask)


@pytest.mark.parametrize("phase", range(TINY_STRIDE))
def test_the_anchor_tensor_shape_is_a_geometry_constant_at_every_phase(phase):
    r"""$A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$ does not vary with $\varphi$ or with
    the batch, so no rank can disagree about a shape and no shape is a function of the data. What
    varies is how many entries are *real*, which travels in ``anchor_valid``.

    Without this the reducer would still be fine -- shapes do not enter it -- but every collective
    that reduces a per-anchor quantity would be reducing tensors of different lengths."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()
    expected = -(-(model.geometry.t_valid - model.warmup_period) // TINY_STRIDE)

    index, valid = model._build_anchor_index(
        batch=3, device=torch.device("cpu"), anchor_phase=phase
    )

    assert index.shape == valid.shape == (3, expected)
