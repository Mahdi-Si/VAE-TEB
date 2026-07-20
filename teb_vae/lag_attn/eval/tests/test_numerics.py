"""``configure_numerics`` actually pins what it claims to pin, and pinning it makes runs repeat.

The returned dict is written into ``summary.json`` and is the only record of the numeric
environment a run executed under, so it has to be read back from global state rather than
echoed from the assignments -- which is exactly what the first test checks.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from teb_vae.lag_attn.eval.numerics import configure_numerics
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn


def test_reported_settings_match_global_state():
    """A recorded setting that does not match reality is worse than no record."""
    settings = configure_numerics(7)

    assert settings["seed"] == 7
    assert settings["default_dtype"] == str(torch.get_default_dtype()) == "torch.float32"
    assert settings["float32_matmul_precision"] == torch.get_float32_matmul_precision() == "highest"
    assert settings["cuda_matmul_allow_tf32"] is torch.backends.cuda.matmul.allow_tf32 is False
    assert settings["cudnn_allow_tf32"] is torch.backends.cudnn.allow_tf32 is False
    # default.yaml sets benchmark: true for training; nondeterministic algorithm selection
    # would break the rerun-reproducibility requirement, so eval turns it back off.
    assert settings["cudnn_benchmark"] is torch.backends.cudnn.benchmark is False
    for device_type, enabled in settings["autocast_enabled"].items():
        assert enabled is torch.is_autocast_enabled(device_type) is False


def test_is_idempotent_and_cpu_safe():
    """Called twice, on a machine with or without CUDA, it reports the same environment."""
    first = configure_numerics(11)
    second = configure_numerics(11)
    assert first == second


def test_seeds_every_generator_the_pipeline_draws_from():
    """torch, numpy and ``random`` -- caps subsample with numpy, the forward samples with torch."""
    configure_numerics(3)
    drawn = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    configure_numerics(3)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == drawn


def test_two_seeded_forwards_are_bit_identical(shipped_kwargs, inputs):
    """The end the settings exist for: a rerun at the same seed reproduces the numbers exactly.

    ``forward`` samples $z$ unconditionally, so this covers the stochastic path -- not merely a
    deterministic one that would have matched regardless.
    """
    outputs = []
    for _ in range(2):
        configure_numerics(5)
        model = SeqVaeLagAttn(**shipped_kwargs)
        model.eval()
        with torch.no_grad():
            outputs.append(model(*inputs))

    for key in ("z", "mu_full", "logvar_full", "kld_per_t", "attn_weights"):
        assert torch.equal(outputs[0][key], outputs[1][key]), f"{key} is not reproducible"
