r"""Does the per-channel delay actually stop a feature from seeing the future? Measured, not argued.

Everything else about the causal guard is analytic: a filter's forward reach $L_{95}$ is read off
its impulse response, and the delay $\delta_c = \lceil \mathrm{reach}_c / \Delta \rceil$ follows
by arithmetic. This turns that into an observation. A raw signal is perturbed strictly *after* an
anchor's causal endpoint, the features are recomputed through the production transform, and the
values the model would actually read are compared.

The two arms answer different questions and each would be worthless alone:

* **With no delay**, the retained features must *move* -- otherwise the whole guard is solving a
  problem that does not exist, and a test asserting the delayed features hold still would pass on
  any model. Restricted to channels whose analytic reach exceeds the perturbation offset, derived
  from the same reach computation the delays come from: a channel faster than the offset
  legitimately cannot move, and including it would make the arm flaky.
* **At the $120$ s budget**, the same measurement must be far smaller.

**What "far smaller" means here, and why it is not zero.** $L_{95}$ is an energy *quantile*, not a
support: $5\%$ of every filter's energy lies beyond its stated reach, so a channel delayed exactly
to its budget still sees a tail. Against a perturbation this severe -- half the record replaced
with fresh noise -- that tail is visible. The measured suppression is roughly a factor of $20$ on
the worst channel, not a reduction to numerical noise, and the thresholds below are set from that.
The residual is larger on the phase-harmonic block than on the scattering block, which is
consistent with a phase coefficient normalising by its own envelope and so amplifying exactly the
low-energy tail the quantile discounts. Neither the label nor the design claims more: the guard
bounds the leak, and only genuinely causal transforms remove it.

The transform is exercised at filter-bank level through ``KymatioPhaseScattering1D`` rather than
through the pipeline's ``compute_scattering_masks``, whose module-level imports do not resolve
outside the production environment.

Slow, and excluded from the default run:

.. code-block:: bash

    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m "not slow"
    .venv/Scripts/python.exe -m pytest teb_vae/lag_attn_rws/tests -q -m slow
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval.representation_capacity_probe import (
    N_RAW,
    build_filter_bank,
    select_phase_pairs,
)
from teb_vae.lag_attn.channel_reach import (
    SECONDS_PER_STEP,
    TARGET_PHASE_BAND_HZ,
    resolve_channel_budget,
    stream_reach_seconds,
)

pytestmark = pytest.mark.slow

#: Raw samples per decimated step, and the transform's parameters, as the pipeline sets them.
_DECIMATION = 16
_J, _Q, _T = 11, 4, 16

#: The anchor probed, on the *untrimmed* $330$-step grid the features are computed on. Mid-segment,
#: so neither the reflect-padded boundary nor the perturbation's own edge is nearby. The delay is a
#: shift along the decimated axis, so it is the same shift on the trimmed grid the model consumes
#: (where this step would be numbered $165 - 15$).
_ANCHOR = 165

#: Independent signals averaged over. A single draw is dominated by which channel happens to catch
#: the noise: the per-channel maximum swings by a factor of six across seeds at $B = 1$ and is
#: stable to a few percent at $B = 8$.
_BATCH = 8

#: The perturbation begins at the anchor's causal endpoint, so a channel read undelayed sees it
#: only if its reach exceeds one step.
_PERTURBATION_OFFSET_S = SECONDS_PER_STEP

#: Largest relative movement the guard may leave. Observed maximum at the $120$ s budget is
#: $0.5$--$0.8$ across seeds; this is set about $1.6$-fold above that.
_GUARDED_TOLERANCE = 1.25

#: Smallest relative movement the *unguarded* features must show. Observed maximum with no delay
#: is $12$--$17$; this is set roughly two-fold below that.
_LEAK_FLOOR = 6.0

# A margin, not a preference. Were the tolerance ever loosened toward the floor, both arms would
# pass on a model with no guard at all and the test would assert nothing.
assert _LEAK_FLOOR > 4.0 * _GUARDED_TOLERANCE


@pytest.fixture(scope="module")
def leak_measurement():
    r"""Perturb the raw future, recompute the features, and measure what each read position moves.

    Returns:
        A dict with ``'guarded'`` -- relative movement at each surviving channel's *delayed* read
        position under the $120$ s budget -- and ``'undelayed'`` -- the same at the anchor itself,
        for the channels whose reach exceeds the perturbation offset.
    """
    from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D

    transform = KymatioPhaseScattering1D(
        J=_J,
        Q=_Q,
        T=_T,
        shape=N_RAW,
        device=torch.device("cpu"),
        # The pipeline's settings for the stored blocks: first order only, no input window.
        max_order=1,
        tukey_alpha=None,
    )

    # The stored target block is [43 scattering, 66 phase-harmonic]; the phase channels are a
    # selection from the transform's full 903-pair axis, located by their (i, j) indices so the
    # test and the reach vector index the same channels.
    pair_column = {
        (int(i), int(j)): column
        for column, (i, j) in enumerate(
            zip(transform.i_idx.tolist(), transform.j_idx.tolist())
        )
    }
    phase_columns = [
        pair_column[pair]
        for pair in select_phase_pairs(build_filter_bank(), *TARGET_PHASE_BAND_HZ)
    ]

    generator = torch.Generator().manual_seed(0)
    clean = torch.randn(_BATCH, N_RAW, generator=generator)
    # Strictly after the anchor's causal endpoint: decimated step t is the filter output at raw
    # index 16t, and its own block ends at 16(t+1) - 1.
    cut = _DECIMATION * (_ANCHOR + 1)
    perturbed = clean.clone()
    perturbed[:, cut:] = torch.randn(
        _BATCH, N_RAW - cut, generator=torch.Generator().manual_seed(1)
    )

    with torch.no_grad():
        clean_out = transform(clean, compute_phase=True)
        perturbed_out = transform(perturbed, compute_phase=True)

    def stored_block(out):
        return np.concatenate(
            [out["scattering"].numpy(), out["phase_corr"].numpy()[:, phase_columns]], axis=1
        )

    before, after = stored_block(clean_out), stored_block(perturbed_out)
    # Each channel is measured against its own spread over the record, so channels on wildly
    # different scales contribute comparably.
    scale = before.std(axis=2).mean(axis=0) + 1e-12

    def movement(channel: int, step: int) -> float:
        return float(np.abs(before[:, channel, step] - after[:, channel, step]).mean() / scale[channel])

    reach = stream_reach_seconds()["target"]
    keep_index, delays = resolve_channel_budget(reach, 120.0, warmup_period=30)
    leaky = [c for c in range(len(reach)) if reach[c] > _PERTURBATION_OFFSET_S]

    return {
        "guarded": np.array(
            [movement(c, _ANCHOR - delay) for c, delay in zip(keep_index, delays)]
        ),
        "undelayed": np.array([movement(c, _ANCHOR) for c in leaky]),
        "guarded_channels": keep_index,
        "leaky_channels": leaky,
        "n_channels": len(reach),
    }


@pytest.mark.parametrize("arm", ["budget-120", "no-delay"])
def test_the_channel_delay_suppresses_the_measured_future_leak(leak_measurement, arm):
    """One direction per arm. Neither passes on a model where the guard does nothing: the
    unguarded arm requires a large measured leak, and the guarded arm requires a small one, with
    the two thresholds separated by the margin asserted at import."""
    if arm == "budget-120":
        movement = leak_measurement["guarded"]
        assert movement.max() < _GUARDED_TOLERANCE, (
            f"the delayed reads still move by {movement.max():.2f} of a channel spread; the "
            f"worst channel is "
            f"{leak_measurement['guarded_channels'][int(np.argmax(movement))]}"
        )
    else:
        movement = leak_measurement["undelayed"]
        assert movement.max() > _LEAK_FLOOR, (
            f"perturbing the raw future moved the undelayed features by at most "
            f"{movement.max():.2f}, below the floor -- the probe is not detecting the leak it "
            f"exists to measure, so the guarded arm proves nothing"
        )


def test_every_channel_slower_than_the_offset_moves_without_the_delay(leak_measurement):
    """The restriction that keeps the unguarded arm honest, checked rather than assumed: every
    channel that analytically *should* see past the anchor measurably does. A channel here that
    did not move would mean the reach vector and the transform disagree about which channel is
    which -- the mis-ordering failure that has no other signal."""
    movement = leak_measurement["undelayed"]

    assert float(movement.min()) > 0.1


def test_the_restriction_is_not_the_whole_block(leak_measurement):
    """If every channel were slower than the offset the restriction would be a no-op, and the
    arm above would silently stop being a restricted claim."""
    assert len(leak_measurement["leaky_channels"]) < leak_measurement["n_channels"]


def test_the_guard_prunes_as_well_as_delays(leak_measurement):
    """The budget removes channels it cannot delay within the warm-up, so the guarded arm is
    measured over fewer channels than the block has. Pinned so a resolution that silently kept
    everything would show up here rather than as a suspiciously good leak number."""
    assert len(leak_measurement["guarded_channels"]) == 78
    assert leak_measurement["n_channels"] == 109
