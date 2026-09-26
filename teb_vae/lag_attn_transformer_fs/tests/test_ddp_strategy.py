r"""Which DDP strategy each shipped config earns, and the gradient-coverage evidence that it is safe.

``find_unused_parameters=False`` makes the reducer expect every parameter to be marked ready in every
backward, and one that is not raises or deadlocks on a real multi-GPU box -- on the production box,
after the dev box passed. The selector is a *claim* about the model; the backward passes below are
the *evidence*.

The claim is inherited twice over, and what this file asks is whether it survives the **pairing**:
the starved set is decided by config -- the decoder log-variance head is consumed only under
``likelihood: gaussian_nll`` -- and that head is the tensor the target domain *widened*, on a model
whose encoders are the ones the availability adapters were built for. The guarded backward is the one
arm where the availability adapters and a $C_{\mathrm{keep}}$-wide output head run together.

A parameter multiplied by an identically-zero tensor *is* reachable -- it receives a zeros gradient
rather than ``None`` -- so the probes ask for ``grad is None``, not for a non-zero gradient. Each
backward runs without the prior-scale anchor: adding an objective term can only add graph edges, so
the unanchored objective is the harder case.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer

from .conftest import BATCH, SEQ_LEN, make_patterned_batch, make_stub_batch, resolve_target_budget

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

#: Sequence and warm-up lengths the guarded probe runs at: the production budget's own resolution
#: refuses a delay longer than the warm-up, so the tiny fixture's lengths are too short for it.
_GUARDED_SEQ_LEN = 64
_GUARDED_WARMUP = 30


def _unreached(model: nn.Module) -> List[str]:
    """Names of parameters that require a gradient and did not receive one."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]


# --------------------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config_name, find_unused",
    [("default.yaml", False), ("tiny.yaml", True)],
)
def test_each_shipped_config_selects_the_strategy_its_likelihood_earns(config_name, find_unused):
    """The shipped ``gaussian_nll`` config earns an every-parameter-reachable reducer; the smoke
    config's ``mse`` selects the fallback, which is what the two evidence tests below justify."""
    driver = LagAttnTrfFsTrainer(config_file_path=str(_CONFIG_DIR / "default.yaml"))
    config = load_config(str(_CONFIG_DIR / config_name))

    assert driver.ddp_kwargs(config)["find_unused_parameters"] is find_unused


# --------------------------------------------------------------------------------------
# The evidence
# --------------------------------------------------------------------------------------
def _starved_parameters(module, batch_idx: int) -> list:
    """Backward one training step and name the trainable parameters left without a gradient."""
    module.zero_grad(set_to_none=True)
    loss, _metrics = module.compute_loss_and_metrics(
        make_stub_batch(4, SEQ_LEN), batch_idx, "train"
    )
    loss.backward()
    return _unreached(module.orig_model)


def test_under_gaussian_nll_no_parameter_is_left_without_a_gradient(task, perturb_posterior):
    """What licenses ``find_unused_parameters=False`` for the shipped config, re-earned on the
    pairing. Perturbed first: at init the posterior deltas are zero, so the attention pathway would
    read as starved for a reason that vanishes after one step."""
    module = task(hparams={"likelihood": "gaussian_nll", "beta_prior": 0.0})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert not starved, (
        f"parameters expecting a gradient but not receiving one: {starved}. Under "
        f"find_unused_parameters=False the reducer raises on exactly these."
    )


def test_under_mse_the_starved_set_is_exactly_the_decoder_logvar_head(task, perturb_posterior):
    """The mirror image, and the justification for the fallback strategy. **Exactly** that head and
    nothing else -- if some other parameter starved here, ``find_unused_parameters=True`` would be
    covering for a second defect rather than for a documented configuration choice."""
    module = task(hparams={"likelihood": "mse", "beta_prior": 0.0})
    perturb_posterior(module.orig_model)

    starved = _starved_parameters(module, 0)

    assert set(starved) == {"decoder.logvar_head.weight", "decoder.logvar_head.bias"}, starved


def test_every_parameter_is_reachable_under_a_real_channel_guard(tiny_kwargs):
    """The guarded case, at the **production** budget's resolved channel tuples, and it is only the
    guarded case if the guard is real.

    The tiny fixture's hand-made guard has a zero minimum delay, in which case the adapter builds no
    start embedding; the production budget's smallest delay is positive, which is what puts both
    availability parameters in the graph. The assertions before the backward stop this silently
    becoming a copy of the unguarded evidence above.
    """
    budget = resolve_target_budget()
    assert budget is not None
    kwargs = dict(
        tiny_kwargs,
        sequence_length=_GUARDED_SEQ_LEN,
        warmup_period=_GUARDED_WARMUP,
        target_keep_index=budget.target_keep_index,
        target_delays=budget.target_delays,
        source_keep_index=budget.source_keep_index,
        source_delays=budget.source_delays,
    )
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**kwargs)

    assert model.source_gate.max_delay > 0 and model.target_gate.max_delay > 0
    assert model.target_adapter.mask_proj is not None
    assert model.target_adapter.start_embed is not None
    assert model.decoder_out_channels == len(kwargs["target_keep_index"])

    batch = make_patterned_batch(BATCH, _GUARDED_SEQ_LEN)
    outs = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    target = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
    model.compute_loss(outs, target, weight=batch.weight, beta_prior=0.0)["metrics"][
        "total_loss"
    ].backward()

    assert not _unreached(model), (
        f"unreachable under find_unused_parameters=False: {_unreached(model)}"
    )
