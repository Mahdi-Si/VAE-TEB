r"""Shared pytest configuration and fixtures for the lag-residual causal-feature forecaster.

Puts the repository root on ``sys.path`` so the absolute ``teb_vae.*`` imports resolve no matter
which directory pytest is invoked from, and pins the real ``utils`` package the same way the
sibling suites do.

**The fixtures here are deliberately self-contained**, and that is a decision rather than an
oversight. The sibling suites import their tiny geometry from the causal cell's conftest, which
carries the *legacy* phase operator's channel widths; this architecture's first task is the integer
operator's, and a suite that closed over the legacy widths would test a geometry no configuration
of this package describes. The modules under test here are geometry-agnostic -- they take widths as
arguments and hold no opinion about the dataset -- so seeded tensors at a small declared geometry
are the whole of what they need. The shard-backed fixtures arrive with the model, which is the first
thing in this package that has an opinion about the data.

Every constant below is small enough to keep a whole suite under a few seconds and large enough to
be structurally faithful: more than one anchor, more than one lag reaching before the record
begins, and a warm-up staircase whose slowest channel is still cold at the first anchor -- which is
the case that separates index support from feature warm-up.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch

# teb_vae/lag_slot_transformer_cfs/tests/conftest.py -> parents[0]=tests,
# [1]=lag_slot_transformer_cfs, [2]=teb_vae, [3]=repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# There are two ``utils`` packages in this repository: the real one at the root and a near-empty one
# deeper in the tree. On a repository-wide run another conftest can put the latter's parent first on
# ``sys.path``, shadowing the real one. Binding the root package now -- while the repo root is still
# first -- pins its ``__path__`` for every later ``utils.<submodule>`` import.
try:
    importlib.import_module("utils")
except Exception:
    pass

#: Samples per batch in every fixture here.
TINY_BATCH = 3

#: Stored steps $T$. Large enough that the earliest anchor still has lags reaching before the
#: record begins, which is the out-of-range branch of the gather.
TINY_SEQ_LEN = 24

#: Source channels $C_U$.
TINY_C_U = 6

#: Candidate lags $L$, so lags run over $0, \ldots, L-1$.
TINY_N_LAGS = 5

#: Target state width $d_h$ the proposal head conditions on.
TINY_D_MODEL = 16

#: Latent width $d_z$.
TINY_D_Z = 4

#: First eligible anchor $F$. Below the slowest channel's warm-up on purpose: at this anchor the
#: lag index is perfectly in range while that channel has not warmed up, which is exactly the pair
#: of conditions the availability rules must keep apart.
TINY_FLOOR = 6

#: $W'_j$ per source channel: a staircase from immediately warm to still cold at the first anchor.
TINY_SOURCE_WARMUP: Tuple[int, ...] = (0, 1, 3, 5, 8, 11)


#: Declared target channels of the integer-operator arm: the first stored block, then the second.
DECLARED_ST = 36
DECLARED_TARGET_PH = 44
DECLARED_C_Y = DECLARED_ST + DECLARED_TARGET_PH

#: Declared source channels of the same arm.
DECLARED_SOURCE_PH = 10
DECLARED_C_U = DECLARED_ST + DECLARED_SOURCE_PH

#: The tiny model's horizon and anchor stride. The stride must not exceed the horizon, and it is
#: above one on purpose: at stride one the tiling has no phase, and every padding assertion in the
#: suite would pass without a padded slot ever existing.
TINY_MODEL_HORIZON = 4
TINY_MODEL_STRIDE = 4

#: Surviving target channels: the last four of the first stored block are dropped, mirroring the
#: shipped budget, so the keep-index is non-contiguous and a positional block split would land in
#: the wrong place.
TINY_TARGET_KEEP = tuple(range(DECLARED_ST - 4)) + tuple(range(DECLARED_ST, DECLARED_C_Y))

#: $W'_c$ per surviving target channel. Bounded by the anchor floor, which the constructor refuses
#: to place below the slowest kept target channel's own warm-up.
TINY_TARGET_WARMUP = tuple(
    min(index // 16, TINY_FLOOR) for index in range(len(TINY_TARGET_KEEP))
)

#: Surviving source channels: all of them, as the shipped budget keeps them.
TINY_SOURCE_KEEP = tuple(range(DECLARED_C_U))

#: $W'_j$ per surviving source channel, rising past the anchor floor on purpose: the slowest source
#: channels are still cold at the first anchors, which is the condition the availability rules must
#: keep separate from index support.
TINY_MODEL_SOURCE_WARMUP = tuple(min(index // 4, 11) for index in range(DECLARED_C_U))


def tiny_model_kwargs(**overrides):
    """Constructor keywords for a structurally faithful miniature of the shipped model.

    Small in every width and every count, and faithful in the four things the suite asserts
    against: a non-contiguous target keep-index, a source warm-up that outlasts the anchor floor,
    a stride above one so the tiling has phases and padding, and an even model width so the
    metadata clock's frequency pairs are complete.

    Args:
        **overrides: Keywords to replace.

    Returns:
        The keyword dict.
    """
    kwargs = dict(
        sequence_length=TINY_SEQ_LEN,
        d_model=TINY_D_MODEL,
        d_z=TINY_D_Z,
        horizon=TINY_MODEL_HORIZON,
        raw_per_step=16,
        warmup_period=TINY_FLOOR,
        c_y=DECLARED_C_Y,
        c_u=DECLARED_C_U,
        use_up_st=True,
        max_lag=TINY_N_LAGS - 1,
        dropout=0.0,
        decoder_hidden=16,
        horizon_depth=2,
        horizon_kernel=3,
        horizon_film=True,
        horizon_attention_blocks=1,
        encoder_conv_kernels=(3, 3),
        encoder_conv_dilations=(1, 2),
        encoder_num_heads=4,
        encoder_d_ff=32,
        target_attention_blocks=2,
        anchor_stride=TINY_MODEL_STRIDE,
        target_keep_index=TINY_TARGET_KEEP,
        target_warmup_steps=TINY_TARGET_WARMUP,
        source_keep_index=TINY_SOURCE_KEEP,
        source_warmup_steps=TINY_MODEL_SOURCE_WARMUP,
        persistence_residual=True,
    )
    kwargs.update(overrides)
    return kwargs


def build_tiny_model(*, seed: int = 20260909, **overrides):
    """Construct the tiny model **deterministically**, so two builds are comparable.

    The generic initialisation pass draws from the global generator, so two constructions in one
    process produce two different models unless the generator is seeded first. Every comparison in
    this suite that builds a model twice -- chunked against unchunked, one share against another,
    an arm against its reference -- depends on that, and without the seed such a test does not fail:
    it passes or fails on whatever random state the tests before it happened to leave, which is the
    worst of both.

    Args:
        seed: Seed applied immediately before construction.
        **overrides: Constructor keywords to replace.

    Returns:
        The model, freshly constructed and in training mode.
    """
    from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs

    torch.manual_seed(seed)
    return SeqVaeLagResidualTrfCfs(**tiny_model_kwargs(**overrides))


def tiny_streams(batch: int = TINY_BATCH, seed: int = 0):
    """Seeded model inputs at the tiny declared widths.

    Args:
        batch: Samples in the batch.
        seed: Seed, so a batch is reproducible.

    Returns:
        ``(y_st, y_ph, u_stream)``.
    """
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, TINY_SEQ_LEN, DECLARED_ST, generator=generator),
        torch.randn(batch, TINY_SEQ_LEN, DECLARED_TARGET_PH, generator=generator),
        torch.randn(batch, TINY_SEQ_LEN, DECLARED_C_U, generator=generator),
    )


@pytest.fixture
def source_stream() -> torch.Tensor:
    """A seeded gated source stream at the tiny declared geometry.

    Returns:
        A $(B, T, C_U)$ float tensor.
    """
    generator = torch.Generator().manual_seed(20260909)
    return torch.randn(TINY_BATCH, TINY_SEQ_LEN, TINY_C_U, generator=generator)


@pytest.fixture
def anchors() -> torch.Tensor:
    """A dense anchor set from the tiny floor to the end of the record.

    Every sample gets the same anchors, which is what makes an assertion about one row an
    assertion about the geometry rather than about a draw.

    Returns:
        A $(B, A)$ ``long`` tensor.
    """
    row = torch.arange(TINY_FLOOR, TINY_SEQ_LEN, dtype=torch.long)
    return row[None, :].expand(TINY_BATCH, -1).contiguous()


@pytest.fixture
def target_state() -> torch.Tensor:
    """A seeded target state at the tiny anchor set, as the proposal head conditions on it.

    Returns:
        A $(B, A, d_h)$ float tensor, with $A$ matching the :func:`anchors` fixture.
    """
    generator = torch.Generator().manual_seed(20260910)
    n_anchors = TINY_SEQ_LEN - TINY_FLOOR
    return torch.randn(TINY_BATCH, n_anchors, TINY_D_MODEL, generator=generator)
