r"""The DDP strategy this cell's configs select, and the model property that licenses it.

The selector is the shared driver's, reached through both parents, so what this file re-earns is not
the code but the *claim*: that each shipped config selects the strategy its likelihood can honour --
every parameter reachable under the learned observation variance, the fallback under ``mse``, whose
decoder log-variance head starves (``tests/test_ddp_reachability.py`` measures both) -- and that no
buffer of this composition is a running statistic, which is what licenses skipping the buffer
broadcast. Each would fail silently on a development box and loudly on the first production step.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

from .conftest import shipped_warmup_kwargs

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"


@pytest.mark.parametrize(
    "config_path, find_unused", [(_CONFIG, False), (_TINY, True)], ids=["shipped", "tiny"]
)
def test_each_config_selects_the_strategy_its_likelihood_can_honour(
    tmp_path, config_path, find_unused
) -> None:
    """The shipped config earns ``find_unused_parameters=False``; ``tiny.yaml`` ships ``likelihood:
    mse`` precisely so the smoke path exercises the fallback where it is cheap to observe, rather
    than leaving it configured and never run. Nothing here reads the shards."""
    driver = LagAttnTrfCrwsTrainer(config_file_path=str(_CONFIG))
    driver.output_base_dir = str(tmp_path)

    kwargs = driver.ddp_kwargs(load_config(str(config_path)))

    assert kwargs["find_unused_parameters"] is find_unused


def test_no_buffer_is_a_running_statistic_so_the_broadcast_is_safe_to_skip() -> None:
    """What licenses ``broadcast_buffers=False``: every buffer is a deterministic function of the
    config, built identically in each rank's constructor, so the broadcast restores values that were
    never going to differ. A ``BatchNorm`` running statistic is the one kind that genuinely diverges
    per rank, and there is none."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfCrws(**shipped_warmup_kwargs())

    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
