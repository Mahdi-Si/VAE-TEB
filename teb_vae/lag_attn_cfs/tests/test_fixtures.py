r"""The fixtures this package builds on, and the ground truth the rest of the suite relies on.

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

The planted-delay shard is checked last: its delay is stamped on the file and recoverable from the
stored coefficients alone, which is what lets it gate a lag readout.
"""
from __future__ import annotations

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    CAUSAL_SHARD,
    STUB_GAP_STEP,
    TINY_HORIZON,
    TINY_SEQ_LEN,
    TINY_STRIDE,
    TINY_WARMUP_PERIOD,
    make_stub_batch,
)


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


# --------------------------------------------------------------------------------------
# The stub batch
# --------------------------------------------------------------------------------------
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

#: The planted shard, beside the two committed variants.
PLANTED_SHARD = CAUSAL_SHARD.parent / "tiny_shard_causal_planted.hdf5"


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
