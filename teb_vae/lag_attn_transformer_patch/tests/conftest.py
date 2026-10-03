"""Shared pytest configuration for the patch-token lag-attention VAE.

Puts the repository root on ``sys.path`` (with the sibling suites' ``utils`` pin), imports the
shared fixtures and helpers from the sibling suites rather than restating them, and defines the one
local thing: :data:`TINY_KWARGS` and :func:`build`. The CFS ``make_stub_batch`` already yields
``fhr``/``up`` at ``16 * seq_len`` plus ``weight``, ``guid`` and ``epoch``; call it with
``seq_len=300``. Its weight gap sits at step 12, inside the 30-step warm-up, so a test that needs a
scored gap plants its own.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# tests -> lag_attn_transformer_patch -> teb_vae -> repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# Pin the repo-root ``utils`` package before a nested one can shadow it (see the sibling suites).
try:
    importlib.import_module("utils")
except Exception:
    pass

from teb_vae.lag_attn_cfs.tests.conftest import (  # noqa: E402,F401
    BATCH,
    STUB_GAP_STEP,
    absolutize_dataset_paths,
    make_stub_batch,
    perturb_posterior,
    pytest_configure,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import (  # noqa: E402,F401
    MOVEMENT_TOL,
    relative_change,
)

#: Tiny widths at the SHIPPED geometry (T 300, R 16, H 30, F 30, S 15): A_max = 16 train, 240 dense.
#: ``horizon_film`` stays on: the horizon core hardcodes per-block FiLM.
TINY_KWARGS: Dict[str, Any] = dict(
    sequence_length=300,
    raw_per_step=16,
    horizon=30,
    warmup_period=30,
    anchor_stride=15,
    lag_kv_source="adapter",
    forecast_ar_residual=True,
    d_model=32,
    d_z=8,
    d_head=8,
    num_heads=4,
    encoder_num_heads=4,
    max_lag=8,
    encoder_d_ff=64,
    target_attention_blocks=2,
    source_attention_blocks=2,
    source_attention_window=8,
    dropout=0.0,
    decoder_hidden=16,
    horizon_film=True,
)


def build(*, seed: int = 0, **overrides: Any):
    """``SeqVaeLagAttnTrfPatch(**TINY_KWARGS | overrides)`` under a fixed seed, in eval mode."""
    from teb_vae.lag_attn_transformer_patch.nets.model import SeqVaeLagAttnTrfPatch

    torch.manual_seed(seed)
    return SeqVaeLagAttnTrfPatch(**{**TINY_KWARGS, **overrides}).eval()
