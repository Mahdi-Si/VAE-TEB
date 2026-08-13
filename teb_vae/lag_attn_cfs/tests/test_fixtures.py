r"""The fixtures this package builds on, and the two properties they must keep.

Nothing here is committed: the causal shard and its statistics belong to ``lag_attn``, because every
model in the family reads the same shards through the same loader, and a second copy would be a
second dataset that could come to disagree.

What is local is the *tiny geometry*, and it is deliberately larger than the two-sided cells' -- $24$
decimated steps against $16$. That is not a preference. A tiling needs a floor, a stride and room for
**more than one tile**, and at $T = 16$ with any usable floor there is exactly one anchor per phase:
every padding assertion in the suite would pass without a padded slot ever existing, and the whole
distinction between $A_{\max}$ and the number of *valid* entries would be untestable.

The stub batch is local for a different reason. The siblings' carries the two-sided widths, and this
package's model declares $36 + 66$ and $36 + 15$; it also carries ``guid`` and ``epoch``, which no
sibling needs, because the anchor tiling's phase is keyed on the pair.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .conftest import (
    BATCH,
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    CAUSAL_SHARD,
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
    STUB_GAP_STEP,
    TINY_HORIZON,
    TINY_SEQ_LEN,
    TINY_STRIDE,
    TINY_WARMUP_PERIOD,
    TWO_SIDED_SHARD,
    make_stub_batch,
)


def test_repo_root_resolves_to_the_directory_holding_the_packages():
    """The ``sys.path`` preamble derives the repo root from this file's own depth; a wrong depth
    would resolve some unrelated directory without ever raising."""
    repo_root = Path(__file__).resolve().parents[3]
    for package in ("teb_vae", "train", "utils"):
        assert (repo_root / package / "__init__.py").is_file(), (
            f"{package}/__init__.py not found under the resolved repo root {repo_root}"
        )


def test_no_fixture_files_live_in_this_package():
    """The committed shard and stats are ``lag_attn``'s; this package references them by path, and
    the CAUSAL pair is what it reads -- the two-sided file beside it is what makes "a causal shard
    is required" a comparison rather than an assertion about one file."""
    assert not (Path(__file__).resolve().parent / "fixtures").exists()
    shared = Path(__file__).resolve().parents[2] / "lag_attn" / "tests" / "fixtures"

    assert (shared / "tiny_shard_causal.hdf5").is_file()
    assert (shared / "tiny_stats_causal.hdf5").is_file()
    assert CAUSAL_SHARD.is_file() and TWO_SIDED_SHARD.is_file()


def test_the_committed_shard_is_the_causal_variant():
    """The one root attribute that separates the two dataset variants, which otherwise share every
    field name and every dtype."""
    import h5py

    with h5py.File(CAUSAL_SHARD, "r") as handle:
        assert handle.attrs["transform"] == "causal"
        assert "causal_warmup_quantile" in handle.attrs
        assert "fhr_up_ph" not in handle, (
            "the cross-signal block is present; the causal variant does not store it"
        )
        widths = {name: int(handle[name].shape[1]) for name in
                  ("fhr_st", "fhr_ph", "up_st", "up_ph")}

    assert widths == {
        "fhr_st": CAUSAL_ST_WIDTH,
        "fhr_ph": CAUSAL_PH_WIDTH,
        "up_st": CAUSAL_ST_WIDTH,
        "up_ph": CAUSAL_C_U - CAUSAL_ST_WIDTH,
    }


def test_the_shard_carries_the_two_fields_the_tile_phase_is_keyed_on():
    """``load_fields`` is honoured literally, so a shard that stored neither would make the phase
    unresolvable at the loader rather than at the refusal."""
    import h5py

    with h5py.File(CAUSAL_SHARD, "r") as handle:
        assert "guid" in handle
        assert "epoch" in handle


# --------------------------------------------------------------------------------------
# The tiny geometry
# --------------------------------------------------------------------------------------
def test_the_tiny_geometry_leaves_room_for_more_than_one_tile():
    """The reason this package's tiny window is longer than its siblings'. With one anchor per phase
    every ``anchor_valid`` assertion would hold vacuously and a padded slot would never exist."""
    t_valid = TINY_SEQ_LEN - TINY_HORIZON
    a_max = -(-(t_valid - TINY_WARMUP_PERIOD) // TINY_STRIDE)

    assert a_max > 1
    # And the last phase gets strictly fewer, which is what makes the padding path reachable.
    last_phase = -(-(t_valid - TINY_WARMUP_PERIOD - (TINY_STRIDE - 1)) // TINY_STRIDE)
    assert last_phase < a_max


def test_the_tiny_geometry_reproduces_the_shipped_pairing_in_miniature():
    """Not a shrunken copy but the same *shape* of problem: a floor above the model's own warm-up, a
    stride equal to the horizon, and a budget that drops the slowest channels."""
    assert TINY_STRIDE == TINY_HORIZON
    assert TINY_WARMUP_PERIOD < TINY_SEQ_LEN - TINY_HORIZON
    assert SHIPPED_WARMUP_PERIOD == SHIPPED_BUDGET_STEPS - 1
    assert SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON == 285


# --------------------------------------------------------------------------------------
# The stub batch
# --------------------------------------------------------------------------------------
def test_the_stub_batch_carries_the_causal_widths():
    """The two-sided siblings' batch declares $43 + 66$ and $43 + 15$; a model built at this
    package's widths would refuse it, which is the point of having a local one."""
    batch = make_stub_batch()

    assert batch.fhr_st.shape == (BATCH, TINY_SEQ_LEN, CAUSAL_ST_WIDTH)
    assert batch.fhr_ph.shape == (BATCH, TINY_SEQ_LEN, CAUSAL_PH_WIDTH)
    assert batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1] == CAUSAL_C_Y
    assert batch.up_st.shape[-1] + batch.up_ph.shape[-1] == CAUSAL_C_U
    assert batch.fhr.shape == (BATCH, 16 * TINY_SEQ_LEN)


def test_the_stub_batch_carries_the_deliberate_gap():
    """A uniformly valid weight would leave every mask assertion in the suite green whether or not
    the masks work, and the gap sits inside the trained anchor range so every mask sees it."""
    batch = make_stub_batch()

    assert float(batch.weight[:, STUB_GAP_STEP].max()) == 0.0
    assert TINY_WARMUP_PERIOD <= STUB_GAP_STEP < TINY_SEQ_LEN - TINY_HORIZON


def test_the_stub_batch_carries_a_per_segment_start_time_and_not_only_a_recording_id():
    """``guid`` identifies the recording; ``epoch`` is ``domain_start`` in seconds and is per
    segment. A batch whose start times were identical would make the phase a function of the
    recording alone and every "segments of one recording tile differently" assertion vacuous."""
    batch = make_stub_batch(4)

    assert len(batch.guid) == 4
    assert batch.epoch.shape == (4,)
    assert len(set(batch.epoch.tolist())) == 4


def test_the_slow_marker_is_registered(request):
    """Registered via ``addinivalue_line``; an unregistered marker warns on every use."""
    markers = request.config.getini("markers")

    assert any(str(marker).startswith("slow") for marker in markers)


def test_the_invocation_lines_are_recorded_for_this_package():
    """``tests/__init__.py`` records how this suite is run, in both tiers, naming *this* package --
    a copy naming another one is a line nobody can paste."""
    recorded = (Path(__file__).resolve().parent / "__init__.py").read_text(encoding="utf-8")

    assert "teb_vae/lag_attn_cfs/tests" in recorded
    assert '-m "not slow"' in recorded
    assert "-m slow" in recorded


def test_the_streams_helper_builds_at_the_declared_widths(tiny_kwargs):
    """The three forward inputs a bare-constructed model takes, at the widths its ``c_y`` and
    ``c_u`` declare rather than at the widths its budget keeps -- the gate is the model's own."""
    from .conftest import make_streams

    y_st, y_ph, u_stream = make_streams(tiny_kwargs)

    assert y_st.shape[-1] == CAUSAL_ST_WIDTH
    assert y_ph.shape[-1] == CAUSAL_PH_WIDTH
    assert u_stream.shape[-1] == int(tiny_kwargs["c_u"]) == CAUSAL_C_U
    assert torch.equal(y_st, make_streams(tiny_kwargs)[0])  # seeded, so a batch is reproducible
