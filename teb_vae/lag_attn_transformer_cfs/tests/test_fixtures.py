r"""The splice: which half of the conftest comes from where, and that the two halves agree.

This suite's fixtures come from two siblings and neither choice is interchangeable. The data half --
the committed causal shard, the configuration builder, the tiny warm-up staircase, the stub batch
carrying the two phase-key fields, the seeded input streams -- is the conv-LSTM causal cell's,
because it describes the *dataset* and the *target domain*. The causality probe and its two
tolerances are the conv-Transformer cell's, because they describe the *architecture*. The
constructor keyword sets are written locally at the conv-Transformer schema, because neither
sibling's would construct: one carries five keywords this constructor refuses, the other carries the
two-sided widths and a two-minute horizon.

The two halves meet at :data:`SHARED_GEOMETRY_KEYS`. The imported batch machinery, the imported
budget resolution and every anchor count this suite asserts close over those values while every
model here is built from the local sets, so the splice is sound only while the two agree -- and a
disagreement builds a model neither parent's suite tests, with no shape differing, because
$A_{\max}$ and the block width are geometry constants either way.

This file is what turns that paragraph into a measurement.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.tests import conftest as causal_conftest
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_rws.tests import conftest as architecture_conftest

from . import conftest as local

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Names this conftest takes from the architecture sibling, and names it takes from the causal one.
#: Written out rather than derived, so a fixture that quietly moved from one half to the other is a
#: failure against a stated intention.
_ARCHITECTURE_HALF = (
    "CAUSALITY_TOL",
    "MOVEMENT_TOL",
    "assert_token_causal",
    "build_stream_encoder",
    "relative_change",
    "resample_after",
)
_TARGET_HALF = (
    "BATCH",
    "CAUSAL_C_U",
    "CAUSAL_C_Y",
    "CAUSAL_SHARD",
    "SHIPPED_BUDGET_STEPS",
    "SHIPPED_HORIZON",
    "SHIPPED_SEQUENCE_LENGTH",
    "SHIPPED_WARMUP_PERIOD",
    "STUB_GAP_STEP",
    "TASK_HPARAMS",
    "TINY_SEQ_LEN",
    "TINY_STRIDE",
    "TINY_TARGET_KEEP_INDEX",
    "TINY_TARGET_WARMUP_STEPS",
    "absolutize_dataset_paths",
    "causal_config",
    "make_stub_batch",
    "make_streams",
    "perturb_posterior",
)


# =================================================================================================
# Which half comes from where
# =================================================================================================
@pytest.mark.parametrize("name", _ARCHITECTURE_HALF)
def test_the_architecture_half_is_the_conv_transformer_siblings_own_object(name):
    """Identity. A local re-implementation of the causality probe would be a second definition of
    what "causal" means for this architecture, free to drift from the one that suite proves."""
    assert getattr(local, name) is getattr(architecture_conftest, name)


@pytest.mark.parametrize("name", _TARGET_HALF)
def test_the_target_half_is_the_causal_cells_own_object(name):
    """Identity again, and for a sharper reason: these describe the committed shard and the boundary
    the resolver reads off it. A second copy would be free to describe a boundary the data no longer
    has, which is exactly what reading the shards rather than declaring the vectors exists to
    prevent."""
    assert getattr(local, name) is getattr(causal_conftest, name)


def test_the_keyword_sets_are_local_rather_than_either_siblings():
    """Neither sibling's would construct, and the failure modes are different in each direction: the
    causal cell's carries ``lstm_layers`` and four more this constructor refuses by name, and the
    architecture sibling's carries the two-sided widths and a two-minute horizon, which would build
    a model describing a dataset that does not exist."""
    assert local.TINY_KWARGS is not causal_conftest.TINY_KWARGS
    assert local.SHIPPED_KWARGS is not architecture_conftest.SHIPPED_KWARGS
    assert local.TINY_KWARGS != causal_conftest.TINY_KWARGS
    assert local.SHIPPED_KWARGS != architecture_conftest.SHIPPED_KWARGS


# =================================================================================================
# The splice is sound only while the two halves agree
# =================================================================================================
def test_the_shared_geometry_keys_are_named_and_non_empty():
    """A silently-empty list would make every agreement check below vacuous, and would be the exact
    shape of the mistake: a key that stopped being shared without anyone noticing."""
    assert local.SHARED_GEOMETRY_KEYS
    assert set(local.SHARED_GEOMETRY_KEYS) <= set(local.SHIPPED_KWARGS)


@pytest.mark.parametrize("key", local.SHARED_GEOMETRY_KEYS)
def test_the_shipped_sets_agree_on_every_shared_geometry_key(key):
    """The measurement the splice rests on. Written out independently on both sides, so this is a
    real comparison rather than a tautology over one literal read twice."""
    theirs = causal_conftest.shipped_warmup_kwargs()

    assert local.SHIPPED_KWARGS[key] == theirs[key], key


@pytest.mark.parametrize("key", local.SHARED_GEOMETRY_KEYS)
def test_the_tiny_sets_agree_on_every_shared_geometry_key_that_both_declare(key):
    """The same check at the tiny geometry, where the imported ``make_stub_batch`` and
    ``make_streams`` close over the values directly."""
    theirs = causal_conftest.TINY_KWARGS
    if key not in theirs or key not in local.TINY_KWARGS:
        pytest.skip(f"{key} is not declared in both tiny sets")

    assert local.TINY_KWARGS[key] == theirs[key], key


def test_the_two_shipped_sets_differ_only_in_the_encoder():
    """The other direction: the splice would also be broken by two sets that agreed on *everything*,
    because then this package would be testing the conv-LSTM cell."""
    theirs = causal_conftest.shipped_warmup_kwargs()
    mine = local.shipped_warmup_kwargs()

    encoder_keys = {
        "encoder_conv_kernels",
        "encoder_conv_dilations",
        "encoder_num_heads",
        "encoder_d_ff",
        "target_attention_blocks",
        "source_attention_blocks",
        "source_attention_window",
    }
    # Directional rather than by set difference. Two keyword sets may also differ by a key one of
    # them states at the constructor's own default -- an inert declaration, not an architecture
    # difference -- and pinning the difference by equality would make this test fail on those.
    assert encoder_keys <= set(mine)
    assert encoder_keys.isdisjoint(theirs)
    assert set(local.CONV_LSTM_ONLY_KEYS).isdisjoint(mine)
    assert set(local.CONV_LSTM_ONLY_KEYS) & set(theirs)

    # And every key both declare must agree in value, or the encoder edge is not the only edge.
    differing = [key for key in set(mine) & set(theirs) if mine[key] != theirs[key]]
    assert differing == [], differing


# =================================================================================================
# The five conv-LSTM keys are absent from every kwargs dict this package builds
# =================================================================================================
@pytest.mark.parametrize(
    "builder",
    [
        lambda: dict(local.TINY_KWARGS),
        lambda: dict(local.SHIPPED_KWARGS),
        local.tiny_warmup_kwargs,
        local.shipped_warmup_kwargs,
    ],
    ids=["tiny", "shipped", "tiny_guarded", "shipped_guarded"],
)
def test_no_kwargs_dict_carries_a_conv_lstm_key(builder):
    """Each would raise ``TypeError`` at the constructor rather than being ignored, so the failure
    is loud -- but it would be loud in whichever test happened to build first, naming a keyword
    rather than the conftest."""
    kwargs = builder()

    present = [key for key in local.CONV_LSTM_ONLY_KEYS if key in kwargs]
    assert present == [], present


def test_every_local_kwargs_dict_actually_constructs():
    """The positive direction, and the one an absence check cannot give: a set missing an encoder
    key would satisfy every test above and build a model at the constructor's own defaults."""
    for builder in (local.tiny_warmup_kwargs, local.shipped_warmup_kwargs):
        torch.manual_seed(0)
        model = SeqVaeLagAttnTrfCfs(**builder())
        assert model.target_gate is not None


# =================================================================================================
# The fixture surface, and the invocation lines
# =================================================================================================
def test_the_committed_causal_fixture_and_its_statistics_are_present():
    """Every automated test in this package reads them; a missing one is the loader's opaque
    "No samples match the specified filters" rather than a path error."""
    assert local.CAUSAL_SHARD.exists()
    assert (local.CAUSAL_SHARD.parent / "tiny_stats_causal.hdf5").exists()
    # And the two-sided shard beside it, which is what makes "a causal shard is required" a
    # comparison rather than an assertion about one file.
    assert local.TWO_SIDED_SHARD.exists()


def test_the_stub_batch_carries_the_two_phase_key_fields():
    """``guid`` and ``epoch``, and ``epoch`` per **segment**: the tiling phase is keyed on the pair,
    and a batch without them would make every phase assertion a test of the refusal rather than of
    the derivation."""
    batch = local.make_stub_batch()

    assert len(batch.guid) == local.BATCH
    assert batch.epoch.shape == (local.BATCH,)
    assert len(set(batch.epoch.tolist())) == local.BATCH, "the segments share a start time"


def test_the_slow_marker_is_registered():
    """There is no repository-wide pytest configuration to declare it, so each package registers its
    own or ``-m slow`` silently selects nothing."""
    source = inspect.getsource(local.pytest_configure)

    assert "slow" in source


def test_the_package_invocation_lines_name_this_package():
    """The two lines are this package's own, with the package name substituted -- not copied
    verbatim from the conv-LSTM cell, which would send a reader to another suite."""
    doc = (Path(__file__).resolve().parents[0] / "__init__.py").read_text(encoding="utf-8")

    assert "teb_vae/lag_attn_transformer_cfs/tests -q -m \"not slow\"" in doc
    assert "teb_vae/lag_attn_transformer_cfs/tests -q -m slow" in doc
    assert "lag_attn_cfs/tests" not in doc
