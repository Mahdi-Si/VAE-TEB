r"""The loader's ``ua_step_mask``: per-step UA validity on each UA block's own step grid.

``ua_step_mask`` turns the stored UA quality into ``up_st_step_valid`` / ``up_ph_step_valid``,
$(T, C)$ booleans the model's ``source_quality_mask`` reads. Two rules, OR-joined:

* **bits** -- stored step $k$ reads raw sample $16k$, so it is bad when any raw sample
  $16k - 15, \dots, 16k$ carries a named bit: the four seconds it reads, never a later sample. The
  synthetic shard of ``test_quality_loader.py`` flags raw samples $239 \dots 243$; with the first
  kept step at $k = 15$ (raw sample $240$), sample $239$ falls in kept step $0$ and samples
  $241 \dots 243$ in kept step $1$. A naive ``reshape(T, 16)`` of the trimmed row would put
  $240 \dots 243$ in step $0$ and drop $239$ altogether, so these placements tell the two apart.
  The version-4 flags sit at raw $280$ (FLAT, invalid) and $281$ (REST and PUSH, valid), both in
  kept step $3$.
* **rho_epsilon** -- a coefficient is bad when its stored $\rho > \varepsilon$, per channel.

What this file guards: the window placement, the OR over named bits and the ``"invalid"`` alias,
``hold_steps``, the $\rho$ rule and its OR with the bits, the off states, and every refusal.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from hdf5_dataset.tests.test_quality_loader import (
    KEPT_STEPS,
    LEN_SEQUENCE,
    TRIM_STEPS,
    WIDTHS,
    _dataset,
    _write_shard,
)

BLOCKS = ("up_st", "up_ph")
LOAD = ["up_st", "up_ph", "fhr_st", "weight", "guid"]


@pytest.fixture()
def quality_shard(tmp_path: Path) -> Path:
    """A quality shard with the flags of ``test_quality_loader.py``."""
    return _write_shard(tmp_path / "quality.hdf5")


def _bad_steps(shard: Path, **spec) -> np.ndarray:
    """The kept steps marked bad in segment 0's ``up_ph`` under a bits-only rule."""
    sample = _dataset(shard, load_fields=LOAD, ua_step_mask=spec)[0]
    valid = sample["up_ph_step_valid"]
    return np.flatnonzero(~valid[:, 0].numpy())


def _rho_bad(segment: int, block: str, epsilon: float) -> np.ndarray:
    """The synthetic shard's stored $\\rho$ rule, recomputed: ``(T, C)`` bool, ``True`` = bad."""
    width = WIDTHS[block]
    codes = (np.arange(width * LEN_SEQUENCE).reshape(width, LEN_SEQUENCE) + segment) % 256
    kept = codes[:, TRIM_STEPS:-TRIM_STEPS]
    return (kept.astype(np.float64) / 255.0 > epsilon).T


# =================================================================================================
# The bit rule
# =================================================================================================
def test_the_invalid_set_marks_the_last_four_seconds_each_step_reads(quality_shard: Path) -> None:
    """CODE at 239 lands in kept step 0, CODE and STEP at 241 and 243 in kept step 1, FLAT in 3."""
    assert _bad_steps(quality_shard, bits=["invalid"]).tolist() == [0, 1, 3]


@pytest.mark.parametrize(
    "bits, expected",
    [
        (["CENSORED"], [0]),            # raw 240 = 16 * 15, the last sample step 15 reads
        (["OSCILLATION"], [1]),         # raw 242 and 243
        (["STEP"], [1]),                # raw 243 only
        (["TRANSIENT"], []),            # not flagged anywhere
        (["FLAT"], [3]),                # raw 280
        (["REST"], [3]),                # raw 281, a valid bit a caller may still name
        (["PUSH"], [3]),                # raw 281
        (["CENSORED", "OSCILLATION"], [0, 1]),
        ("censored", [0]),              # a single name, any case
    ],
)
def test_named_bits_are_or_joined(quality_shard: Path, bits, expected) -> None:
    assert _bad_steps(quality_shard, bits=bits).tolist() == expected


def test_the_bit_rule_is_one_value_for_every_channel(quality_shard: Path) -> None:
    sample = _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"bits": ["invalid"]})[0]
    for block in BLOCKS:
        valid = sample[f"{block}_step_valid"]
        assert valid.shape == (KEPT_STEPS, WIDTHS[block])
        assert valid.dtype == torch.bool
        assert torch.equal(valid, valid[:, :1].expand_as(valid))


def test_hold_steps_extends_a_bad_step_forward_only(quality_shard: Path) -> None:
    assert _bad_steps(quality_shard, bits=["CENSORED"], hold_steps=2).tolist() == [0, 1, 2]
    assert _bad_steps(quality_shard, bits=["OSCILLATION"], hold_steps=1).tolist() == [1, 2]


def test_a_hold_that_starts_in_the_trimmed_lead_in_reaches_the_served_window(
    tmp_path: Path,
) -> None:
    """The rule runs on the untrimmed grid, so a flag 3 steps before the window still holds into it."""
    shard = _write_shard(tmp_path / "early.hdf5")
    with h5py.File(shard, "r+") as f:
        bits = f["up_quality"][:]
        bits[:] = 0
        bits[:, 16 * (TRIM_STEPS - 3)] = 8  # TRANSIENT, read by untrimmed step 12
        f["up_quality"][:] = bits
    assert _bad_steps(shard, bits=["TRANSIENT"]).tolist() == []
    assert _bad_steps(shard, bits=["TRANSIENT"], hold_steps=4).tolist() == [0, 1]


# =================================================================================================
# The rho rule
# =================================================================================================
@pytest.mark.parametrize("segment", [0, 1])
def test_rho_marks_each_coefficient_above_epsilon(quality_shard: Path, segment: int) -> None:
    sample = _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"rho_epsilon": 0.1})[segment]
    for block in BLOCKS:
        expected = ~_rho_bad(segment, block, 0.1)
        assert np.array_equal(sample[f"{block}_step_valid"].numpy(), expected), block


def test_the_two_rules_are_or_joined(quality_shard: Path) -> None:
    sample = _dataset(
        quality_shard, load_fields=LOAD, ua_step_mask={"bits": ["invalid"], "rho_epsilon": 0.1}
    )[0]
    for block in BLOCKS:
        bad = _rho_bad(0, block, 0.1)
        bad[[0, 1], :] = True
        assert np.array_equal(sample[f"{block}_step_valid"].numpy(), ~bad), block


def test_hold_steps_does_not_touch_the_rho_rule(quality_shard: Path) -> None:
    held = _dataset(
        quality_shard, load_fields=LOAD, ua_step_mask={"rho_epsilon": 0.1, "hold_steps": 5}
    )[0]
    plain = _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"rho_epsilon": 0.1})[0]
    for block in BLOCKS:
        assert torch.equal(held[f"{block}_step_valid"], plain[f"{block}_step_valid"])


# =================================================================================================
# Off states and scope
# =================================================================================================
@pytest.mark.parametrize("spec", [None, {"bits": None, "rho_epsilon": None}, {"bits": []}])
def test_no_rule_emits_nothing(quality_shard: Path, spec) -> None:
    """``None`` and a mapping that sets no rule are both off: that is how a variant turns it off."""
    sample = _dataset(quality_shard, load_fields=LOAD, ua_step_mask=spec)[0]
    assert not any(name.endswith("_step_valid") for name in sample)


def test_only_the_loaded_ua_blocks_get_a_mask(quality_shard: Path) -> None:
    sample = _dataset(
        quality_shard, load_fields=["up_ph", "weight"], ua_step_mask={"bits": ["invalid"]}
    )[0]
    assert "up_ph_step_valid" in sample
    assert "up_st_step_valid" not in sample


def test_the_masks_collate_as_booleans(quality_shard: Path) -> None:
    from torch.utils.data import DataLoader

    from hdf5_dataset.hdf5_dataset import attribute_dict_collate

    dataset = _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"rho_epsilon": 0.1})
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=attribute_dict_collate)))
    for block in BLOCKS:
        assert batch[f"{block}_step_valid"].shape == (2, KEPT_STEPS, WIDTHS[block])
        assert batch[f"{block}_step_valid"].dtype == torch.bool


# =================================================================================================
# Refusals
# =================================================================================================
def test_a_plain_file_is_refused(tmp_path: Path) -> None:
    plain = _write_shard(tmp_path / "plain.hdf5", quality=False)
    with pytest.raises(ValueError, match="signal-quality build"):
        _dataset(plain, load_fields=LOAD, ua_step_mask={"bits": ["invalid"]})


def test_an_unknown_bit_name_is_refused(quality_shard: Path) -> None:
    with pytest.raises(ValueError, match="does not define"):
        _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"bits": ["SPIKE"]})


def test_an_unknown_key_is_refused(quality_shard: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"bits": ["CODE"], "pool": "any"})


@pytest.mark.parametrize("epsilon", [-0.1, 1.0])
def test_an_epsilon_outside_the_unit_interval_is_refused(quality_shard: Path, epsilon) -> None:
    with pytest.raises(ValueError, match="rho_epsilon"):
        _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"rho_epsilon": epsilon})


def test_a_negative_hold_is_refused(quality_shard: Path) -> None:
    with pytest.raises(ValueError, match="hold_steps"):
        _dataset(quality_shard, load_fields=LOAD, ua_step_mask={"bits": ["CODE"], "hold_steps": -1})


def test_a_rho_rule_on_a_file_without_rho_is_refused(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "no_rho.hdf5", omit=("up_st_rho",))
    with pytest.raises(ValueError, match="up_st_rho"):
        _dataset(shard, load_fields=LOAD, ua_step_mask={"rho_epsilon": 0.1})
