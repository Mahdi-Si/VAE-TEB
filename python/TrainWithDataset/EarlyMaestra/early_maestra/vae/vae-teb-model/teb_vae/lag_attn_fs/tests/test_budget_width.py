r"""The reach budget decides the target width, and therefore what the objective sums over.

In the raw sibling the budget shapes only what the model *reads*: the decoder emits $R = 16$ raw
samples per horizon token whatever survives. Here it also shapes what the model is *scored on* --
the decoder emits one coefficient per surviving target channel, so the budget sets the decoder
width, the block cardinality $H_d \cdot C_{\mathrm{keep}}$ the NLL sums over, and hence the scale
of every reconstruction number the run reports. Two arms at different budgets have non-comparable
``pred_gap`` and mutually unloadable checkpoints.

That binding is what this file pins: the resolved survivor count at the shipped budget and at the
unguarded arm, the reach statistics of the surviving set that make the far horizon clean, and --
against the committed shard rather than a stub -- that the declared widths the keep-index is
positional against are the widths the loader actually delivers.

The resolution itself is ``lag_attn``'s and is tested there; what is asserted here is the numbers
this model was costed against and their consequences for its own shapes.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.channel_reach import stream_reach_seconds
from teb_vae.lag_attn.figure_primitives import future_target
from teb_vae.lag_attn_fs.tests.conftest import (
    SHIPPED_KWARGS,
    SHIPPED_REACH_BUDGET_S,
    absolutize_dataset_paths,
    build_target_gate,
    resolve_target_budget,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The sibling's tiny config, resolved through its own ``base:`` chain. It is the loader
#: configuration this package inherits -- the same shards, the same trim, the same
#: ``normalize_fields`` -- and reading it here rather than writing a fourth copy is what keeps the
#: real-data check bound to the configuration a run uses.
_TINY_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "tiny.yaml"

#: Surviving channels at the shipped budget, and the full declared width.
_KEPT_CHANNELS = 78
_ALL_CHANNELS = 109

#: The horizon, and the coefficients the reconstruction sums over per contributing anchor at the
#: shipped budget: $30 \times 78 = 2340$, against the raw model's $30 \times 16 = 480$.
_HORIZON = SHIPPED_KWARGS["horizon"]
_BLOCK_WIDTH = _HORIZON * _KEPT_CHANNELS


# ---------------------------------------------------------------------------------------
# The resolved width
# ---------------------------------------------------------------------------------------
def test_the_shipped_budget_keeps_seventy_eight_target_channels():
    """The figure the decoder width, the block cardinality and the $\\beta$ recalibration were
    all costed against."""
    budget = resolve_target_budget(SHIPPED_REACH_BUDGET_S)

    assert budget is not None
    assert len(budget.target_keep_index) == _KEPT_CHANNELS
    assert len(budget.target_delays) == _KEPT_CHANNELS


def test_the_unguarded_arm_keeps_every_declared_channel():
    """``causal_reach_budget_s: null`` builds no gate at all, so the decoder width follows $c_y$
    and the unguarded arm is a well-defined configuration rather than an unhandled case."""
    assert resolve_target_budget(None) is None
    assert build_target_gate(None) is None
    assert SHIPPED_KWARGS["c_y"] == _ALL_CHANNELS


def test_the_decoder_width_follows_the_gate_not_the_raw_grid():
    """``raw_per_step`` stays in the configuration -- the geometry needs it and the diagnostic
    page's time axis is drawn on the raw grid -- but it stops being the decoder width."""
    gate = build_target_gate(SHIPPED_REACH_BUDGET_S)

    assert gate.out_channels == _KEPT_CHANNELS
    assert gate.out_channels != SHIPPED_KWARGS["raw_per_step"]
    assert _HORIZON * gate.out_channels == _BLOCK_WIDTH == 2340


def test_the_block_grows_by_the_factor_beta_is_recalibrated_for():
    r"""$H_d C / (H_d R) = 2340 / 480 \approx 4.9$ at the shipped budget, $6.8$ ungated. The
    reconstruction applies that much more pressure against a KL over the same $d_z$ the raw model
    uses -- the family moves that width together -- which is
    why $\beta$ is retuned rather than inherited, and why the direction is *wider*, not
    narrower: a larger reconstruction at fixed $\beta$ makes $\beta\,\mathrm{KL}$ relatively
    weaker."""
    raw_block = _HORIZON * SHIPPED_KWARGS["raw_per_step"]

    assert raw_block == 480
    assert _BLOCK_WIDTH / raw_block == pytest.approx(4.875, abs=1e-3)
    assert (_HORIZON * _ALL_CHANNELS) / raw_block == pytest.approx(6.8125, abs=1e-3)


# ---------------------------------------------------------------------------------------
# What the surviving set is made of
# ---------------------------------------------------------------------------------------
def test_the_surviving_sets_reach_is_short_enough_for_the_horizon_to_come_clean():
    r"""The measurement the target restriction rests on. A coefficient at horizon step $\tau$ is
    centred $\approx 4\tau$ s after the anchor, so its support stops overlapping observed history
    at step $\rho_c / 4$. The kept set's worst channel reaches $117.25$ s and comes clean at step
    $29.3$, just past the $H_d = 30$ horizon; the full set retains a channel at $965.5$ s and
    never reaches a fully clean step at all."""
    reach = stream_reach_seconds()["target"]
    keep_index = resolve_target_budget(SHIPPED_REACH_BUDGET_S).target_keep_index
    kept = [reach[index] for index in keep_index]

    assert len(reach) == _ALL_CHANNELS
    assert statistics.median(kept) == pytest.approx(28.0)
    assert max(kept) == pytest.approx(117.25)
    assert max(kept) <= SHIPPED_REACH_BUDGET_S
    # The full set, for the contrast the restriction is justified by.
    assert statistics.median(reach) == pytest.approx(47.25)
    assert max(reach) == pytest.approx(965.5)


def test_the_survivors_come_from_both_stored_blocks():
    """$27$ of $43$ scattering and $51$ of $66$ phase-harmonic. Both blocks surviving is what
    makes the ``fhr_st``-against-``fhr_ph`` split of the reported gap non-vacuous."""
    keep_index = resolve_target_budget(SHIPPED_REACH_BUDGET_S).target_keep_index
    scattering = sum(1 for index in keep_index if index < 43)

    assert scattering == 27
    assert len(keep_index) - scattering == 51


def test_the_keep_index_is_positional_into_the_declared_width():
    """Strictly ascending and inside $[0, c_y)$: the delay vector is positional against it, and a
    reordered or out-of-range index would gather the wrong channels with no other signal."""
    keep_index = resolve_target_budget(SHIPPED_REACH_BUDGET_S).target_keep_index

    assert all(later > earlier for earlier, later in zip(keep_index, keep_index[1:]))
    assert 0 <= min(keep_index) and max(keep_index) < _ALL_CHANNELS


# ---------------------------------------------------------------------------------------
# Against the committed shard
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_batch():
    """One batch from the committed shard, through the real loader and the shipped trim."""
    from teb_vae.lag_attn.config import load_config
    from train.data_module import GraphDataModule

    config = absolutize_dataset_paths(load_config(str(_TINY_CONFIG)))
    return next(iter(GraphDataModule(config).train_dataloader()))


@pytest.mark.slow
def test_the_loader_delivers_the_widths_the_keep_index_indexes_into(real_batch):
    """The keep-index is resolved from a filter bank and applied to a tensor from an HDF5 file.
    Nothing connects the two but the declared widths, and a shard written at other widths would
    gather the wrong channels rather than fail."""
    assert real_batch.fhr_st.shape[-1] == 43
    assert real_batch.fhr_ph.shape[-1] == 66
    assert real_batch.fhr_st.shape[-1] + real_batch.fhr_ph.shape[-1] == _ALL_CHANNELS
    assert real_batch.fhr_st.shape[1] == SHIPPED_KWARGS["sequence_length"]
    assert real_batch.weight.shape[1] == SHIPPED_KWARGS["sequence_length"]


@pytest.mark.slow
def test_the_target_block_builds_from_the_committed_shard(real_batch):
    """End to end on real data: concatenate the two stored blocks, unfold the future, gather the
    survivors, and land on the shape the decoder emits."""
    gate = build_target_gate(SHIPPED_REACH_BUDGET_S)
    block = torch.index_select(
        future_target(real_batch.fhr_st, real_batch.fhr_ph, _HORIZON), -1, gate.keep_index
    )

    samples = real_batch.fhr_st.shape[0]
    t_valid = SHIPPED_KWARGS["sequence_length"] - _HORIZON
    assert block.shape == (samples, t_valid, _HORIZON, _KEPT_CHANNELS)
    assert torch.isfinite(block).all()


@pytest.mark.slow
def test_the_index_identity_holds_on_real_data(real_batch):
    """The same identity the planted pattern pins, re-checked where the values are not chosen:
    anchor $t$, step $\\tau$ is stored step $t + 1 + \\tau$ of the surviving channels."""
    gate = build_target_gate(SHIPPED_REACH_BUDGET_S)
    stream = torch.cat([real_batch.fhr_st, real_batch.fhr_ph], dim=-1)
    kept = torch.index_select(stream, -1, gate.keep_index)
    block = torch.index_select(
        future_target(real_batch.fhr_st, real_batch.fhr_ph, _HORIZON), -1, gate.keep_index
    )

    for anchor, tau in ((0, 0), (137, 11), (269, 29)):
        assert torch.equal(block[:, anchor, tau, :], kept[:, anchor + 1 + tau, :])


@pytest.mark.slow
def test_the_target_blocks_are_normalized_by_the_loader(real_batch):
    """``fhr_st`` and ``fhr_ph`` are the reconstruction target here, not merely inputs. A stats
    file the loader rejects disables normalization with a warning and hands back correctly shaped,
    wrongly scaled tensors -- and a Gaussian NLL against those is meaningless with nothing
    raising anywhere."""
    for block in (real_batch.fhr_st, real_batch.fhr_ph):
        assert abs(float(block.mean())) < 5.0
        assert float(block.std()) < 20.0
