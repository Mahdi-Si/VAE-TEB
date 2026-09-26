r"""The DDP strategy this driver selects for this cell's configs, and the model property behind it.

The selector is the shared driver's and is tested there, so what this file re-earns is not the code
but the *claim* on this composition: the shipped config selects ``find_unused_parameters=False``
(which ``test_ddp_reachability.py`` licenses by measuring every gradient), an ``mse`` config selects
the fallback, and no buffer is a running statistic, so skipping the buffer broadcast is safe.
"""
from __future__ import annotations

from pathlib import Path

import torch

from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer

from .conftest import shipped_warmup_kwargs

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_the_shipped_config_expects_every_parameter_and_mse_selects_the_fallback(tmp_path):
    """The payoff of the learned observation variance plus the unconditional ``W_o`` freeze: the
    reducer can expect every parameter. Under ``mse`` the decoder log-variance head is trainable
    and unused, so the same driver must fall back. ``setup_config`` is never called: nothing here
    reads the shards."""
    trainer = LagAttnTrfCfsTrainer(config_file_path=str(_CONFIG))
    trainer.output_base_dir = str(tmp_path)

    assert trainer.ddp_kwargs(trainer.config)["find_unused_parameters"] is False
    mse = {"model_config": {"VAE_model": {"likelihood": "mse"}}}
    assert trainer.ddp_kwargs(mse)["find_unused_parameters"] is True


def test_no_buffer_is_a_running_statistic_so_the_broadcast_is_safe_to_skip():
    """What licenses ``broadcast_buffers=False``: every buffer is a deterministic function of the
    config, built identically in each rank's constructor, so the broadcast restores values that were
    never going to differ. A ``BatchNorm`` running statistic is the one kind that genuinely diverges
    per rank, and there is none anywhere in the model at the shipped geometry and budget."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCfs(**shipped_warmup_kwargs())

    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
