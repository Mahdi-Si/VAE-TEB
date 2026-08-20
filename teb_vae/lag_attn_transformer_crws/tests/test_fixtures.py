r"""The splice: which half of the conftest comes from where, and that the two halves agree.

This suite's fixtures come from two siblings and neither choice is interchangeable. The input and
target half -- the committed causal shard, the configuration builder, the tiny warm-up staircase,
the stub batch carrying the two phase-key fields, the seeded input streams and the seeded raw
signal -- is the conv-LSTM causal-input cell's, because it describes the *dataset*, the *target
domain* and the *anchor geometry*. The causality probe and its two tolerances are the
conv-Transformer raw-signal cell's, because they describe the *architecture*. The constructor
keyword sets are written locally at the conv-Transformer schema, because neither sibling's would
construct: one carries five keywords this constructor refuses, the other carries the two-sided
widths and a two-minute horizon.

The two halves meet at :data:`SHARED_GEOMETRY_KEYS`, and here that seam is load-bearing rather than
decorative: the two conftests are independently maintained, so nothing but this file makes them
agree. The imported batch machinery, the imported budget resolution, the imported raw-signal builder
and every anchor count this suite asserts close over the conv-LSTM cell's values while every model
here is built from the local sets -- and a disagreement builds a model neither parent's suite tests,
with no shape differing, because $A_{\max}$ and the raw block width are geometry constants either
way.

This file is what turns that paragraph into a measurement.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_crws.tests import conftest as causal_conftest
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_rws.tests import conftest as architecture_conftest

from . import conftest as local

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: The seven keys that are the encoder edge between this cell and the conv-LSTM one. Written out
#: rather than derived from the difference between the two sets, because the difference is what the
#: test below is measuring.
_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: The production geometry this package declares, as literals. Held at the conv-LSTM cell's values
#: so that the two differ in exactly one variable -- which encoder reads the inputs -- and a silent
#: change to any of them moves what every number this package reports is produced at.
_SHIPPED_GEOMETRY = (
    ("horizon", 30),
    ("warmup_period", 134),
    ("anchor_stride", 30),
    ("c_y", 102),
    ("c_u", 51),
)


# =================================================================================================
# Which half comes from where
# =================================================================================================
def test_both_imported_name_lists_are_non_empty_and_still_exported():
    """A silently-empty list would make every identity check below vacuous, and would be the exact
    shape of the mistake: a name that stopped being shared without anyone noticing."""
    assert local.IMPORTED_FROM_ARCHITECTURE
    assert local.IMPORTED_FROM_CRWS

    missing = [
        name
        for name in local.IMPORTED_FROM_ARCHITECTURE
        if not hasattr(architecture_conftest, name)
    ] + [name for name in local.IMPORTED_FROM_CRWS if not hasattr(causal_conftest, name)]
    assert missing == [], missing


@pytest.mark.parametrize("name", local.IMPORTED_FROM_ARCHITECTURE)
def test_the_architecture_half_is_the_conv_transformer_siblings_own_object(name):
    """Identity. A local re-implementation of the causality probe would be a second definition of
    what "causal" means for this architecture, free to drift from the one that suite proves -- and
    this encoder stack has no time-pooling normaliser to flip, so its control is positional and
    shares nothing with the conv-LSTM cell's."""
    assert getattr(local, name) is getattr(architecture_conftest, name)


@pytest.mark.parametrize("name", local.IMPORTED_FROM_CRWS)
def test_the_input_and_target_half_is_the_conv_lstm_cells_own_object(name):
    """Identity again, and for a sharper reason: these describe the committed shard, the boundary
    the resolver reads off it and the anchor geometry the objective scores at. A second copy would
    be free to describe a boundary the data no longer has, which is exactly what reading the shards
    rather than declaring the vectors exists to prevent."""
    assert getattr(local, name) is getattr(causal_conftest, name)


def test_the_two_imported_halves_are_disjoint():
    """A name reachable from both lists would resolve by import order rather than by intention, and
    the second import statement would silently win."""
    overlap = set(local.IMPORTED_FROM_ARCHITECTURE) & set(local.IMPORTED_FROM_CRWS)

    assert overlap == set(), overlap


def test_the_keyword_sets_are_local_rather_than_either_siblings():
    """Neither sibling's would construct, and the failure modes differ in each direction: the
    conv-LSTM cell's carries ``lstm_layers`` and three more this constructor refuses by name, and
    the conv-Transformer raw-signal sibling's carries the two-sided widths and a two-minute horizon,
    which would build a model describing a dataset that does not exist."""
    assert local.TINY_KWARGS is not causal_conftest.TINY_KWARGS
    assert local.SHIPPED_KWARGS is not architecture_conftest.SHIPPED_KWARGS
    assert local.TINY_KWARGS != causal_conftest.TINY_KWARGS
    assert local.SHIPPED_KWARGS != architecture_conftest.SHIPPED_KWARGS


# =================================================================================================
# The splice is sound only while the two halves agree
# =================================================================================================
def test_the_shared_geometry_keys_are_named_and_non_empty():
    """A silently-empty list would make every agreement check below vacuous."""
    assert local.SHARED_GEOMETRY_KEYS
    assert set(local.SHARED_GEOMETRY_KEYS) <= set(local.SHIPPED_KWARGS)


def test_the_shared_geometry_keys_name_every_key_both_shipped_sets_declare():
    """The list is only worth having if it is complete. Anything both halves set and this list omits
    could drift with nothing failing, which is the whole failure mode the splice has."""
    both = set(local.SHIPPED_KWARGS) & set(causal_conftest.SHIPPED_KWARGS)
    # The encoder edge is the declared difference, so it is excluded by name rather than by
    # happening not to appear -- and it does not appear in the conv-LSTM set at all.
    shared = both - set(_ENCODER_KEYS)

    missing = sorted(shared - set(local.SHARED_GEOMETRY_KEYS))
    # The keys below are shared and deliberately outside the list: they are widths, weights and
    # numerical settings rather than the geometry the imported fixtures close over. Naming them
    # here is what makes the assertion above fail on the arrival of a *new* shared key rather than
    # pass on whatever the two sets happen to hold today.
    allowed = {
        "d_model",
        "d_z",
        "num_heads",
        "d_head",
        "dropout",
        "decoder_hidden",
        "horizon_depth",
        "horizon_kernel",
        "horizon_film",
        "horizon_attention_blocks",
        "horizon_embed_std",
        "head_init_calibration",
        "a_head_gain",
        "logvar_clamp",
        "mu_scale",
        "delta_mu_scale",
        "delta_logvar_scale",
        "use_entmax",
        "lag_bias_init",
        "coverage_floor",
    }
    assert set(missing) <= allowed, sorted(set(missing) - allowed)


@pytest.mark.parametrize("key", local.SHARED_GEOMETRY_KEYS)
def test_the_shipped_sets_agree_on_every_shared_geometry_key(key):
    """The measurement the splice rests on. Written out independently on both sides, so this is a
    real comparison rather than a tautology over one literal read twice."""
    theirs = causal_conftest.shipped_warmup_kwargs()

    assert local.SHIPPED_KWARGS[key] == theirs[key], key


@pytest.mark.parametrize("key", local.SHARED_GEOMETRY_KEYS)
def test_the_tiny_sets_agree_on_every_shared_geometry_key_that_both_declare(key):
    """The same check at the tiny geometry, where the imported ``make_stub_batch``,
    ``make_streams`` and ``make_raw_signal`` close over the values directly."""
    theirs = causal_conftest.TINY_KWARGS
    if key not in theirs or key not in local.TINY_KWARGS:
        pytest.skip(f"{key} is not declared in both tiny sets")

    assert local.TINY_KWARGS[key] == theirs[key], key


def test_the_two_shipped_sets_differ_only_in_the_encoder():
    """The other direction: the splice would also be broken by two sets that agreed on *everything*,
    because then this package would be testing the conv-LSTM cell."""
    theirs = causal_conftest.shipped_warmup_kwargs()
    mine = local.shipped_warmup_kwargs()

    encoder_keys = set(_ENCODER_KEYS)
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


@pytest.mark.parametrize("key,value", _SHIPPED_GEOMETRY, ids=[key for key, _ in _SHIPPED_GEOMETRY])
def test_the_shipped_set_declares_the_production_geometry(key, value):
    """The configuration every number this package reports is produced at."""
    assert local.SHIPPED_KWARGS[key] == value


def test_the_shipped_set_leaves_the_decoder_width_to_the_architecture():
    r"""No ``decoder_out_channels``. It is not a keyword of this constructor at all -- the
    architecture parent has none either -- so the raw block's width follows ``raw_per_step`` and no
    configuration can put the decoder and the target on different widths."""
    assert "decoder_out_channels" not in local.SHIPPED_KWARGS
    assert "decoder_out_channels" not in local.TINY_KWARGS
    assert local.SHIPPED_KWARGS["raw_per_step"] == local.TINY_KWARGS["raw_per_step"] == 16


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
        model = SeqVaeLagAttnTrfCrws(**builder())
        assert model.target_gate is not None


# =================================================================================================
# The fixture surface, and the invocation lines
# =================================================================================================
def test_no_fixture_files_live_in_this_package():
    """The committed shard and stats are ``lag_attn``'s; this package references them by path.

    The two-sided file beside them is what makes "a causal shard is required" a comparison rather
    than an assertion about one file, and the statistics file matters as much as the shard: the
    dataset reader silently disables normalization on a stats-schema mismatch, so a missing one is
    every shape right and every number wrong.
    """
    assert not (_PACKAGE_DIR / "tests" / "fixtures").exists()
    shared = _REPO_ROOT / "teb_vae" / "lag_attn" / "tests" / "fixtures"

    assert (shared / "tiny_shard_causal.hdf5").is_file()
    assert (shared / "tiny_stats_causal.hdf5").is_file()
    assert local.CAUSAL_SHARD.is_file() and local.TWO_SIDED_SHARD.is_file()


def test_the_stub_batch_carries_the_two_phase_key_fields():
    """``guid`` and ``epoch``, and ``epoch`` per **segment**: the tiling phase is keyed on the pair,
    and a batch without them would make every phase assertion a test of the refusal rather than of
    the derivation."""
    batch = local.make_stub_batch()

    assert len(batch.guid) == local.BATCH
    assert batch.epoch.shape == (local.BATCH,)
    assert len(set(batch.epoch.tolist())) == local.BATCH, "the segments share a start time"


def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use, and ``-m
    slow`` silently selects nothing."""
    markers = request.config.getini("markers")

    assert any(str(marker).startswith("slow") for marker in markers)
    assert "slow" in inspect.getsource(local.pytest_configure)


def test_the_invocation_lines_are_recorded_for_this_package():
    """``tests/__init__.py`` records how this suite is run, in both tiers, naming *this* package --
    a copy naming another one is a line nobody can paste."""
    recorded = (Path(__file__).resolve().parent / "__init__.py").read_text(encoding="utf-8")

    assert 'teb_vae/lag_attn_transformer_crws/tests -q -m "not slow"' in recorded
    assert "teb_vae/lag_attn_transformer_crws/tests -q -m slow" in recorded
    assert "lag_attn_crws/tests" not in recorded
