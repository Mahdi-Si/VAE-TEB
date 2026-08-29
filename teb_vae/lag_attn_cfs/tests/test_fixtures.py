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
    stride equal to the horizon, and a budget that drops the slowest channels.

    The shipped floor is the **aligned** one, which is the budget itself rather than one below it:
    a shifted input channel is honest at $W'_c + d_c$, whose maximum the zero-marginal-warm-up
    lemma puts at exactly $B$. The scored-target half of the pairing still asks only for $B - 1$,
    and both are asserted so a reference removed without the floor following it fails here.
    """
    assert TINY_STRIDE == TINY_HORIZON
    assert TINY_WARMUP_PERIOD < TINY_SEQ_LEN - TINY_HORIZON
    assert SHIPPED_WARMUP_PERIOD == SHIPPED_BUDGET_STEPS
    assert SHIPPED_WARMUP_PERIOD >= SHIPPED_BUDGET_STEPS - 1
    assert SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON == 270


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


# =================================================================================================
# The planted-delay fixture, and the check that it carries what it claims
#
# The other committed shard is the real bank over committed raw segments: it is what a model is
# trained and evaluated on in miniature, and nothing about its content is known in advance. This one
# is the opposite kind of object -- an INSTRUMENT. A delay is planted at the raw level, the pair is
# pushed through the same bank, and the informative lags on the written coefficients are therefore
# known: a band around the plant, and nowhere else.
#
# That is what makes it usable as a gate on a lag readout, and it is also what makes it dangerous.
# If the plant did not survive the bank -- whose one-sided group delays reach the same order as the
# delay itself -- then a model failing to recover it would be reporting a property of the fixture,
# and the failure would read as a finding about the architecture. So the coupling is re-measured
# from the WRITTEN coefficients here, before any test that uses the shard runs.
# =================================================================================================
#: The check config's own lag geometry, written out rather than loaded. The plant has to sit
#: strictly inside $(H, L - 1)$ *at the geometry the check runs*, and reading both from the config
#: the check also reads would make the interval agree with itself; these are the two numbers
#: `configs/planted.yaml` pins and refuses to retune, so they are stated here as the claim.
_PLANTED_HORIZON = 30
_PLANTED_MAX_LAG = 90

#: Seconds per decimated step on this family's grid, so the stamped delay in seconds and the stamped
#: delay in steps have to agree rather than being two independent claims.
_STEP_SECONDS = 4.0

#: The planted pair, beside the two committed variants.
PLANTED_SHARD = CAUSAL_SHARD.parent / "tiny_shard_causal_planted.hdf5"
PLANTED_STATS = CAUSAL_SHARD.parent / "tiny_stats_causal_planted.hdf5"


def test_the_planted_pair_is_committed_beside_the_other_fixtures():
    """Both files, in the shared directory every cell's fixture tests already assert against. A
    stats file missing beside its shard is the failure that matters: zero is the channel mean only
    under statistics accumulated the same way, and the whole source-null control rests on that."""
    assert PLANTED_SHARD.is_file()
    assert PLANTED_STATS.is_file()


def test_the_planted_shard_is_a_real_causal_shard_and_not_a_hand_written_one():
    """The coefficients come from the production bank through the production writer, so the file
    carries the same self-describing attributes as the committed shard beside it.

    Asserted because the alternative was available and would have been easier: hand-writing blocks
    with a delay between them. That would have fabricated the warm-up boundary as well, and a
    fabricated boundary is a *second* boundary -- the anchor floor would then be resolved against a
    number no filter produced, and every warm-fraction readout would be measuring the fabrication.
    """
    import h5py

    with h5py.File(PLANTED_SHARD, "r") as handle:
        attributes = dict(handle.attrs)
        blocks = {name: handle[name].shape for name in ("fhr_st", "fhr_ph", "up_st", "up_ph")}

    assert attributes["transform"] == "causal"
    assert "causal_warmup_quantile" in attributes
    assert "causal_leg_alignment" in attributes
    # The same four blocks at the same widths as the committed causal shard: the fixture differs in
    # what the signals SAY, not in what shape they are, which is what lets one config read either.
    assert {name: shape[1] for name, shape in blocks.items()} == {
        "fhr_st": CAUSAL_ST_WIDTH,
        "fhr_ph": CAUSAL_PH_WIDTH,
        "up_st": CAUSAL_ST_WIDTH,
        "up_ph": CAUSAL_C_U - CAUSAL_ST_WIDTH,
    }


def test_the_planted_geometry_is_stamped_on_the_shard():
    r"""The check script reads the plant off the file rather than assuming it, so the file has to
    say it -- and every stamped number is one the generator **measured** on the written
    coefficients rather than one it declared.

    The delay is asserted to sit strictly inside $(H, L - 1)$ at the check's own geometry, which is
    the property that makes the instrument an instrument: below $H$ the informative band would fall
    off the near edge of the lag window at some horizon steps, and at or above $L - 1$ off the far
    edge. Both would be a fixture no model could pass, at which case a failure would say nothing.
    """
    import h5py

    with h5py.File(PLANTED_SHARD, "r") as handle:
        attributes = dict(handle.attrs)

    delay = int(attributes["planted_delay_steps"])
    assert _PLANTED_HORIZON < delay < _PLANTED_MAX_LAG
    assert float(attributes["planted_delay_seconds"]) == delay * _STEP_SECONDS
    assert len(attributes["planted_coupled_channels"]) > 0
    assert len(attributes["planted_control_channels"]) > 0
    assert attributes["planted_source_block"] == "up_st"
    assert attributes["planted_target_block"] == "fhr_st"


def test_the_planted_delay_is_recoverable_from_the_stored_coefficients_alone():
    """The instrument validated **without a model**, which is the whole reason it can gate one.

    Cross-correlation between the coupled source channel and the matched target channel, on the
    coefficients as written -- so what is measured is the file a model is handed, ``float32`` round
    trip included. Two halves, and the second is what makes the first mean something: the coupled
    channels peak inside a narrow band around the plant, and the control channels peak nowhere.

    The pairing is by matched index rather than by search. The two blocks are one bank over two
    signals, so channel $c$ of each carries the same composed group delay and the pair's delays
    cancel; a pair drawn across blocks would measure the plant plus a block offset and report the
    sum as though it were the plant.
    """
    from scripts.make_tiny_shard import self_check_planted_shard

    report, passed = self_check_planted_shard(str(PLANTED_SHARD))

    assert passed, report
    assert "PASS" in report
