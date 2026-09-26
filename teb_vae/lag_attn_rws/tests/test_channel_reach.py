r"""The forward-reach numbers, checked against something that is not themselves.

The whole causal guard rests on a per-channel reach vector, and a reach vector is not the kind
of thing a self-comparison can validate: recomputing it the same way and asserting it matches
would pass on any consistent mistake. So the primary check here is external -- the same numbers
are already pinned, independently, by ``teb_vae/lag_attn/eval/representation_capacity_probe.py``
against the analysis document they were published in.

A further property gets its own tests because it fails silently: **the phase-harmonic channel
order must match the shard's.** The reach vector is positional, so a reordering delays the wrong
channels and nothing anywhere fails. The stored order is ``KymatioPhaseScattering1D``'s pair
enumeration filtered by a boolean mask, and that is what is reproduced here; the shard's own
``sel_i`` / ``sel_j`` attributes are the authority, and the test against them runs when a shard
that carries them is available.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from teb_vae.lag_attn.eval.representation_capacity_probe import (
    Q,
    build_filter_bank,
    select_phase_pairs,
)
from teb_vae.lag_attn.channel_reach import (
    SOURCE_PHASE_BAND_HZ,
    TARGET_PHASE_BAND_HZ,
    block_reach_seconds,
    stream_reach_seconds,
)

#: A production shard carrying the phase-selection attributes, for the ordering cross-check. The
#: committed fixture does not carry them (it predates the shard writer that stores them), so the
#: check is opt-in through the environment rather than silently skipped forever.
_PRODUCTION_SHARD_ENV = "TEB_VAE_PRODUCTION_SHARD"

#: The forecast horizon the probe's own pinned numbers are stated against, in seconds.
_HORIZON_S = 120.0


# ---------------------------------------------------------------------------------------
# The external check: the numbers the probe already pins
# ---------------------------------------------------------------------------------------
def test_the_low_pass_reach_matches_the_published_figure():
    r"""$\phi$ is essentially causal at $8$--$9.5$ s, which is the probe's pinned range."""
    blocks = block_reach_seconds()

    # Channel 0 of the scattering block is S_0 = x * phi, stored first and unmasked.
    assert 8.0 <= blocks["fhr_st"][0] <= 9.5


def test_the_causal_scattering_channel_count_matches_the_published_figure():
    """27 of 43 ``fhr_st`` channels reach no further than the 120 s horizon. Published in the
    analysis document and asserted by the probe's own self-test, so this is an independent
    check rather than a restatement of this module's arithmetic."""
    blocks = block_reach_seconds()

    within = sum(1 for value in blocks["fhr_st"] if value <= _HORIZON_S)

    assert (within, len(blocks["fhr_st"])) == (27, 43)


def test_the_block_widths_are_the_stored_widths():
    """A reach vector of the wrong length would be gathered against the wrong channels."""
    blocks = block_reach_seconds()

    assert {name: len(values) for name, values in blocks.items()} == {
        "fhr_st": 43,
        "fhr_ph": 66,
        "up_st": 43,
        "up_ph": 15,
    }


def test_every_source_phase_channel_reaches_at_least_a_hundred_seconds():
    """A structural property of the source phase band, not an accident of one filter bank: its
    upper edge is 0.05 Hz, so both endpoints of every pair are slow. It is what makes budgets
    below 100 s drop the block entirely."""
    blocks = block_reach_seconds()

    assert min(blocks["up_ph"]) >= 100.0


# ---------------------------------------------------------------------------------------
# Channel ordering
# ---------------------------------------------------------------------------------------
def _pipeline_pair_order(band_hz):
    """The ``(i, j)`` pairs in the order the shard writer stores them, from the pipeline's own
    index convention.

    ``KymatioPhaseScattering1D._build_coupling_indices`` enumerates pairs with
    $\\xi_j \\ge \\xi_i$ in ascending ``(i, j)`` order, and the writer selects from that axis
    with a boolean mask, which preserves the order. Reproducing the enumeration here rather than
    importing the pipeline's selector is deliberate: that selector lives behind module-level
    imports that do not resolve outside the production environment.
    """
    bank = build_filter_bank()
    hz = bank.hz
    f_min, f_max = band_hz
    kept = []
    for i in range(bank.n_filters):
        for j in range(bank.n_filters):
            if hz[j] < hz[i]:
                continue  # the pipeline keeps only xi_j >= xi_i
            # The pipeline's band test applies to the pair's endpoints, and its power test
            # admits a harmonic ratio within a 5% relative tolerance.
            if hz[i] < f_min or hz[j] > f_max:
                continue
            ratio = hz[j] / hz[i]
            if any(abs(ratio - 2 ** (k / Q)) < 0.05 * 2 ** (k / Q) for k in (4, 6, 8)):
                kept.append((i, j))
    return kept


@pytest.mark.parametrize(
    "band, expected_width",
    [(TARGET_PHASE_BAND_HZ, 66), (SOURCE_PHASE_BAND_HZ, 15)],
    ids=["fhr_ph", "up_ph"],
)
def test_the_phase_pair_order_matches_the_pipelines_enumeration(band, expected_width):
    """Order, not just count. Counting alone is what the selection was previously validated by,
    and a permuted reach vector passes a count check while delaying the wrong channels."""
    ours = select_phase_pairs(build_filter_bank(), band[0], band[1])

    assert len(ours) == expected_width
    assert ours == _pipeline_pair_order(band)


@pytest.mark.parametrize("field", ["fhr_ph", "up_ph"])
def test_the_phase_pair_order_matches_a_production_shards_attributes(field):
    """The authority on the stored order. The committed fixture predates the writer that records
    these attributes, so this runs against a production shard when one is pointed at."""
    shard = os.environ.get(_PRODUCTION_SHARD_ENV)
    if not shard or not Path(shard).is_file():
        pytest.skip(
            f"no production shard at {_PRODUCTION_SHARD_ENV}={shard!r}. The committed fixture "
            f"carries no sel_i/sel_j attributes, so the stored phase-harmonic channel order "
            f"cannot be cross-checked here; point that variable at a shard written by "
            f"hdf5_dataset/new_pipeline/create_new_pipeline.py to enable it."
        )

    import h5py

    with h5py.File(shard, "r") as handle:
        attributes = dict(handle[field].attrs)
    if "sel_i" not in attributes:
        pytest.skip(f"{shard} carries no sel_i attribute on {field}; it predates the writer")

    band = TARGET_PHASE_BAND_HZ if field == "fhr_ph" else SOURCE_PHASE_BAND_HZ
    ours = select_phase_pairs(build_filter_bank(), band[0], band[1])

    assert [pair[0] for pair in ours] == np.asarray(attributes["sel_i"]).tolist()
    assert [pair[1] for pair in ours] == np.asarray(attributes["sel_j"]).tolist()


# ---------------------------------------------------------------------------------------
# The stream view the model consumes
# ---------------------------------------------------------------------------------------
def test_the_streams_concatenate_in_the_models_own_order():
    """The model builds its target stream as ``[scattering, phase]`` and gathers positionally
    into it, so the reach vector must be concatenated the same way."""
    blocks = block_reach_seconds()
    streams = stream_reach_seconds(use_up_st=True)

    assert streams["target"] == blocks["fhr_st"] + blocks["fhr_ph"]
    assert streams["source"] == blocks["up_st"] + blocks["up_ph"]
    assert (len(streams["target"]), len(streams["source"])) == (109, 58)


def test_the_source_stream_drops_its_scattering_block_under_the_ablation():
    streams = stream_reach_seconds(use_up_st=False)

    assert streams["source"] == block_reach_seconds()["up_ph"]
    assert len(streams["source"]) == 15
