r"""Which DDP strategy the shipped configuration selects, and the evidence that it is safe.

``find_unused_parameters=False`` makes the reducer expect every parameter to be marked ready in
every backward, and one that is not raises or deadlocks on a real multi-GPU box. The selector is the
raw-signal driver's and is tested there; what this cell has to re-earn is the *evidence* on its own
model: under ``gaussian_nll`` every trainable parameter receives a gradient -- on both guard states,
because only the guarded one carries the availability projections the warm-up brings into existence
-- and under ``mse`` the starved set is exactly the decoder log-variance head, which is what the
fallback strategy covers.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

from .conftest import (
    TINY_KWARGS,
    TINY_STRIDE,
    make_stub_batch,
    tiny_align_kwargs,
    tiny_warmup_kwargs,
)

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


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


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
def test_the_shipped_config_earns_every_parameter_reachable(trainer):
    """The payoff of the learned observation variance plus the unconditional ``W_o`` freeze: the
    reducer can expect every parameter."""
    assert trainer.ddp_kwargs(trainer.config)["find_unused_parameters"] is False


# --------------------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------------------
def _arm_kwargs(arm: str) -> dict:
    """The tiny keyword set at a real tiling, in one of the three guard states.

    ``aligned`` is the shipped state: the shift lifts every channel's combined wait off zero, so
    both adapters build a start-of-record embedding -- a parameter reached only on the leading
    steps of a segment, which every batch of every rank has.
    """
    if arm == "ungated":
        return dict(TINY_KWARGS, anchor_stride=TINY_STRIDE)
    if arm == "aligned":
        return tiny_align_kwargs(anchor_stride=TINY_STRIDE)
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


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


@pytest.mark.parametrize("arm", ["guarded", "ungated", "aligned"])
@pytest.mark.parametrize("beta_prior", [0.0, 0.1], ids=["unanchored", "anchored"])
def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(
    task, perturb_posterior, arm, beta_prior
):
    """What actually licenses ``find_unused_parameters=False``, re-earned on a decoder scoring raw
    samples and -- the new part -- on the availability terms the warm-up brings into existence.

    Every guard state, because the guarded one is the only configuration in which the adapters
    carry an availability projection at all, and the aligned one the only one with a start-of-record
    embedding: a starvation introduced by either would be invisible in the ungated arm.

    Perturbed first: at init the posterior deltas are zero, so the attention pathway carries no
    downstream weight and would read as starved for a reason that vanishes after one step.
    """
    module = task(
        model_kwargs=_arm_kwargs(arm),
        hparams={"likelihood": "gaussian_nll", "beta_prior": beta_prior},
    )
    if arm == "aligned":
        adapters = (module.orig_model.target_adapter, module.orig_model.source_adapter)
        assert all(adapter.start_embed is not None for adapter in adapters), "nothing to probe"
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under "
        f"find_unused_parameters=False the reducer raises on exactly these."
    )


@pytest.mark.parametrize("arm", ["guarded", "ungated"])
def test_under_mse_the_starved_set_is_exactly_the_decoder_logvar_head(
    task, perturb_posterior, arm
):
    """The mirror image, and the justification for the fallback strategy: with mse the decoder
    log-variance head is trainable and unused. **Exactly** that head and nothing else -- if some
    other parameter starved here, ``find_unused_parameters=True`` would be covering for a second
    defect rather than for a documented configuration choice.

    Its width is the one thing the budget cannot move: the head emits $R$ raw samples per horizon
    token whatever the warm-up keeps, which is why the two guard states starve the same tensor at
    the same size rather than at two."""
    module = task(model_kwargs=_arm_kwargs(arm), hparams={"likelihood": "mse"})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert set(starved) == {"decoder.logvar_head.weight", "decoder.logvar_head.bias"}, starved
    assert (
        module.orig_model.decoder.logvar_head.bias.numel()
        == module.orig_model.decoder_out_channels
        == module.orig_model.raw_per_step
    )
