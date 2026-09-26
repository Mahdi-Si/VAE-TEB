r"""The evidence that the DDP strategy the inherited selector picks is safe for this model.

``find_unused_parameters=False`` makes the reducer expect every parameter to be marked ready in
every backward, and one that is not raises or deadlocks on a real multi-GPU box. The selector itself
is the raw-signal driver's and is tested there; what this cell has to re-earn is the *evidence* the
selector's claim rests on, because it changes the model on three counts:

* the availability mask is **load-bearing** rather than a guard that happens to be off, so the
  grad-coverage tests run on both the guarded and the ungated model;
* ``start_embed`` is **conditionally constructed** -- it exists only when every channel of a stream
  is unavailable for at least one step. The channel alignment puts one on *both* streams and is
  safe, because the leading region it is live on is the leading region of every segment; dropping
  the first source block would put one on the source for a different reason, which the pre-flight
  refuses by name;
* the decoder log-variance head is $C_{\mathrm{keep}}$ wide and is consumed only under
  ``gaussian_nll``, so under ``mse`` it is exactly the starved set and nothing else is.

And the buffers this cell adds are deterministic functions of the budget, never a running
statistic, which is what licenses ``broadcast_buffers=False``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

from .conftest import (
    TINY_KWARGS,
    TINY_STRIDE,
    absolutize_dataset_paths,
    make_stub_batch,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


# --------------------------------------------------------------------------------------
# The buffers
# --------------------------------------------------------------------------------------
def test_no_buffer_is_a_running_statistic_so_the_broadcast_is_safe_to_skip():
    """What licenses ``broadcast_buffers=False``: every buffer is a deterministic function of the
    config, built identically in each rank's constructor, so the broadcast restores values that were
    never going to differ. A ``BatchNorm`` running statistic is the one kind that genuinely diverges
    per rank, and there is none.

    This cell adds three buffers of its own -- the warm-up tertile assignment and the two per-block
    source warmth patterns -- and every one is a function of the resolved budget and the geometry, so
    they belong in the same category. They are non-persistent for a second reason: their contents
    follow the budget, so a persistent copy would make a checkpoint trained at one budget fail to
    load at another and report it as misaligned keys rather than as a budget mismatch."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnCfs(**shipped_warmup_kwargs())
    buffers = dict(model.named_buffers())

    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    for name in (
        "warm_tertile_id",
        "novelty_tertile_id",
        "source_block_warm_st",
        "source_block_warm_ph",
    ):
        assert name in buffers, name
        assert name not in model.state_dict(), f"{name} reaches a checkpoint"


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
    """What actually licenses ``find_unused_parameters=False``, re-earned on the widened decoder
    head and -- the new part -- on the availability terms the warm-up brings into existence.

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
    defect rather than for a documented configuration choice."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE) if guarded else dict(
        TINY_KWARGS, anchor_stride=TINY_STRIDE
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


# --------------------------------------------------------------------------------------
# The start embedding: a construction-time hazard rather than a width
# --------------------------------------------------------------------------------------
def test_the_shipped_aligned_budget_builds_a_start_embedding_on_both_streams():
    r"""The shipped aligned configuration builds it, and the reason it is not a hazard.

    Both streams reach warm-up zero on their own -- the target because ``fhr_st``'s fastest channels
    are honest from step $0$, the source because ``up_st``'s are -- but the adapter is fed
    $W'_c + d_c$, and the alignment shifts every channel of both streams onto one clock. The
    combined minimum is therefore the fastest channel's *shift*, and the adapter builds
    a start indicator on each stream: a learned $d_{\mathrm{model}}$-wide vector per stream, live
    on the leading region where no channel of that stream has arrived at all.

    **Why ``find_unused_parameters=False`` still holds.** A parameter reached only by *some* batches
    is the hazard; this one is reached by every batch of every rank, because those leading steps
    are the leading steps of every segment the loader serves and the term is added unconditionally
    in the forward -- the branch is ``self.start_embed is not None``, a test on a module built in
    ``__init__``, never on tensor content. The two facts are asserted together, because the first
    without the second reads as a regression.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnCfs(**shipped_warmup_kwargs())

    assert min(model.target_warmup_steps) == 0
    assert min(model.source_warmup_steps) == 0
    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.start_embed is not None
        assert adapter.min_delay > 0
        # Live on the leading region of every segment, and only there.
        indicator = adapter.start_indicator.squeeze(-1)
        assert bool(indicator[: adapter.min_delay].all())
        assert not bool(indicator[adapter.min_delay :].any())


def test_the_preflight_refuses_dropping_the_first_source_block():
    """Without ``up_st`` the source's fastest surviving channel waits, so every step below it has no
    available channel at all and the adapter builds a start indicator -- a *construction-time*
    change, with no shape and no width anywhere saying it happened, which is why the pre-flight
    refuses that configuration by name."""
    from teb_vae.lag_attn.config import load_config

    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["model_config"]["VAE_model"]["use_up_st"] = False
    config["model_config"]["VAE_model"]["c_u"] = 15
    with pytest.raises(ValueError, match="use_up_st"):
        LagAttnCfsTrainer.preflight(config)
