r"""The loader against signal-quality shards: opt-in fields, their geometry, masks and pairing.

A file built with the signal-quality step (root attribute ``quality_version``) stores the bridged
signals in ``fhr``/``up`` and adds bit fields, pre-fill signals, the contamination $\rho$ of every
coefficient and per-segment scalars. The shards here are tiny synthetic causal files written with
h5py, carrying only the attributes the layout check reads, so every value a test compares against
is known by construction.

What this file guards
---------------------
* **The opt-in contract.** ``load_fields=None`` must serve a quality file exactly as it serves a
  file without the step: no quality field, so every old caller gets the same keys.
* **The geometry of a named field.** Bit fields and pre-fill signals are trimmed like ``fhr``/``up``;
  $\rho$ is trimmed and transposed like its block and divided by ``quality_rho_scale``; nothing is
  normalised, and the bit codes are never cast to float.
* **The decoded masks.** ``up_valid``/``fhr_valid`` follow the stored invalid-bit masks, with the
  FHR mask chosen per segment by ``fhr_repair_exact``.
* **The FHR step mask.** ``fhr_step_mask`` builds ``fhr_st_step_valid``/``fhr_ph_step_valid`` with
  the rules of ``ua_step_mask`` (``test_ua_step_mask.py``), and ``"invalid"`` resolves per segment
  by ``fhr_repair_exact``, as the raw FHR mask does.
* **Coherence.** A file list or a stats pairing that mixes quality and non-quality builds, or two
  quality versions, is refused at construction, and so is a step-mask $\varepsilon$ that differs
  from the statistics' ``quality_rho_epsilon``.
* **The writers and the GUID view.** The legacy append writers refuse a quality file, and the GUID
  collate pads every quality field with a value that reads as "not recorded".
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hdf5_dataset.guid_hdf5_dataset import sequence_collate_fn
from hdf5_dataset.hdf5_dataset import (
    AttributeDict,
    CombinedHDF5Dataset,
    _QUALITY_FIELDS,
    append_sample,
    append_samples_batch,
    attribute_dict_collate,
)

#: Raw samples and stored steps per segment, as in a production shard.
N_RAW = 5280
LEN_SEQUENCE = 330
#: ``trim_minutes=1.0`` discards $240$ raw samples, i.e. $15$ steps, from each end.
TRIM_MINUTES = 1.0
TRIM_RAW = 240
TRIM_STEPS = 15
KEPT_RAW = N_RAW - 2 * TRIM_RAW
KEPT_STEPS = LEN_SEQUENCE - 2 * TRIM_STEPS
#: Channels per causal block. Small, and different per block, so a transposed or swapped block
#: cannot pass a shape check.
WIDTHS = {"fhr_st": 3, "fhr_ph": 2, "up_st": 4, "up_ph": 1}
N_SEGMENTS = 4

#: The bit maps the builder writes (``hdf5_dataset/signal_quality.py``, quality version 4).
UA_BITS = {"MISSING": 1, "CODE": 2, "CENSORED": 4, "TRANSIENT": 8, "STEP": 16, "HOLD": 32,
           "FLOOR": 64, "ISLAND": 128, "OSCILLATION": 256, "REST": 512, "PUSH": 1024,
           "FLAT": 2048}
#: Every UA bit but the valid ones: CENSORED, OSCILLATION, REST and PUSH.
UA_INVALID = 1 | 2 | 8 | 16 | 32 | 64 | 128 | 2048
#: The root ``quality_version`` of the synthetic quality shard.
QUALITY_VERSION = 4
FHR_BITS = {"MISSING": 1, "REPAIRED": 2, "JUMP": 8, "HOLD": 32, "LINEAR": 64}
FHR_INVALID_EXACT = 1 | 2 | 8
FHR_INVALID_RULES = 1 | 2 | 8 | 32 | 64
#: ``fhr_repair_exact`` per segment: both FHR mask choices occur.
REPAIR_EXACT = (1, 0, 1, 0)

#: First kept raw sample at ``trim_minutes=1.0``, where the test flags are placed.
FIRST_KEPT = TRIM_RAW
#: Kept raw sample of the version-4 flags: FLAT (invalid), then REST and PUSH (valid). Raw samples
#: $280$ and $281$ are both read by untrimmed step $18$ ($273 \dots 288$), i.e. kept step $3$.
V4_FLAGS_AT = 40
#: ``fhr_raw_match`` per segment: the last segment did not read the raw FHR channel.
FHR_RAW_MATCH = (0.875, 0.875, 0.875, float("nan"))


# =================================================================================================
# Synthetic shards
# =================================================================================================
def _write_shard(
    path: Path,
    *,
    quality: bool = True,
    quality_version: Optional[int] = QUALITY_VERSION,
    kernel_digest: str = "digest-a",
    omit: Sequence[str] = (),
) -> Path:
    """Write a tiny causal shard, with or without the signal-quality datasets.

    Args:
        path: Where to write.
        quality: Add the quality datasets and root attributes.
        quality_version: The root ``quality_version`` of a quality shard; ``None`` omits it.
        kernel_digest: The root ``quality_kernel_digest`` of a quality shard.
        omit: Quality datasets to leave out.

    Returns:
        ``path``.
    """
    n = N_SEGMENTS
    rows = np.arange(n, dtype=np.float32)[:, None]
    with h5py.File(path, "w") as f:
        f.attrs["transform"] = "causal"
        f.attrs["causal_leg_alignment"] = "envelope"
        f.attrs["causal_phase_operator"] = "integer_harmonic_v1"
        f.attrs["causal_warmup_quantile"] = np.float32(0.95)
        f["fhr"] = (120.0 + rows + np.arange(N_RAW, dtype=np.float32) % 7).astype(np.float32)
        f["up"] = (10.0 + rows + np.arange(N_RAW, dtype=np.float32) % 11).astype(np.float32)
        for block, width in WIDTHS.items():
            values = rows[:, :, None] + np.arange(width * LEN_SEQUENCE, dtype=np.float32).reshape(
                1, width, LEN_SEQUENCE
            )
            f[block] = values.astype(np.float32)
            f[block].attrs["causal_warmup_steps"] = np.zeros(width, dtype=np.int64)
        f["target"] = np.zeros((n, LEN_SEQUENCE), dtype=np.float32)
        f["weight"] = np.ones((n, LEN_SEQUENCE), dtype=np.float32)
        f.create_dataset("guid", data=[f"g{i // 2}" for i in range(n)], dtype=h5py.string_dtype())
        f["epoch"] = np.arange(n, dtype=np.float32) * 1200.0
        f["cs_label"] = np.zeros(n, dtype=np.uint8)
        f["bg_label"] = np.zeros(n, dtype=np.uint8)
        if not quality:
            return path

        if quality_version is not None:
            f.attrs["quality_version"] = np.int64(quality_version)
        f.attrs["quality_lookahead_s"] = np.float64(122.0)
        f.attrs["quality_jump_units"] = np.float64(15.0)
        f.attrs["quality_ceiling_values"] = np.array([100.0, 127.0])
        f.attrs["quality_rho_scale"] = np.int32(255)
        f.attrs["quality_kernel_digest"] = kernel_digest

        up_bits = np.zeros((n, N_RAW), dtype=np.uint16)
        up_bits[:, FIRST_KEPT - 1] = UA_BITS["CODE"]              # trimmed away
        up_bits[:, FIRST_KEPT] = UA_BITS["CENSORED"]              # valid
        up_bits[:, FIRST_KEPT + 1] = UA_BITS["CODE"]              # invalid
        up_bits[:, FIRST_KEPT + 2] = UA_BITS["OSCILLATION"]       # valid, above uint8
        up_bits[:, FIRST_KEPT + 3] = UA_BITS["OSCILLATION"] | UA_BITS["STEP"]  # invalid
        up_bits[:, FIRST_KEPT + V4_FLAGS_AT] = UA_BITS["FLAT"]                 # invalid
        up_bits[:, FIRST_KEPT + V4_FLAGS_AT + 1] = UA_BITS["REST"] | UA_BITS["PUSH"]  # valid
        fhr_bits = np.zeros((n, N_RAW), dtype=np.uint8)
        fhr_bits[:, FIRST_KEPT] = FHR_BITS["HOLD"]                # invalid under the rules only
        fhr_bits[:, FIRST_KEPT + 1] = FHR_BITS["JUMP"]            # invalid under both
        fhr_bits[:, FIRST_KEPT + 2] = FHR_BITS["LINEAR"]          # invalid under the rules only
        signals: Dict[str, np.ndarray] = {
            "up_quality": up_bits,
            "fhr_quality": fhr_bits,
            "up_raw": ((np.arange(N_RAW) % 100) + rows).astype(np.float16),
            "fhr_raw": ((np.arange(N_RAW) % 50) + 100 + rows).astype(np.float16),
        }
        for name, values in signals.items():
            if name not in omit:
                f[name] = values
        if "up_quality" in f:
            f["up_quality"].attrs["bits"] = json.dumps(UA_BITS)
            f["up_quality"].attrs["invalid_bits"] = np.int32(UA_INVALID)
        if "fhr_quality" in f:
            f["fhr_quality"].attrs["bits"] = json.dumps(FHR_BITS)
            f["fhr_quality"].attrs["invalid_bits_exact"] = np.int32(FHR_INVALID_EXACT)
            f["fhr_quality"].attrs["invalid_bits_rules"] = np.int32(FHR_INVALID_RULES)
        for block, width in WIDTHS.items():
            if f"{block}_rho" in omit:
                continue
            codes = (np.arange(width * LEN_SEQUENCE).reshape(1, width, LEN_SEQUENCE) + rows[:, :, None])
            f[f"{block}_rho"] = (codes % 256).astype(np.uint8)
        segment = {
            "up_valid_frac": np.full(n, 0.75, dtype=np.float32),
            "up_censored_frac": np.full(n, 0.125, dtype=np.float32),
            "fhr_valid_frac": np.full(n, 0.5, dtype=np.float32),
            "fhr_repair_exact": np.asarray(REPAIR_EXACT, dtype=np.uint8),
            "fhr_raw_match": np.asarray(FHR_RAW_MATCH, dtype=np.float32),
        }
        for name, values in segment.items():
            if name not in omit:
                f[name] = values
    return path


def _write_stats(
    path: Path, *, quality_version: Optional[int], rho_epsilon: Optional[float] = None
) -> Path:
    """Write a minimal statistics file: the root attributes the pairing reads and one ``up`` group.

    A ``up_raw`` group is added as well, so a loader that normalised a quality field because the
    statistics name it would be caught.

    Args:
        path: Where to write.
        quality_version: The root ``quality_version``; ``None`` omits it.
        rho_epsilon: The root ``quality_rho_epsilon``; ``None`` omits it.

    Returns:
        ``path``.
    """
    with h5py.File(path, "w") as f:
        f.attrs["transform"] = "causal"
        f.attrs["causal_leg_alignment"] = "envelope"
        f.attrs["causal_phase_operator"] = "integer_harmonic_v1"
        f.attrs["trim_minutes"] = TRIM_MINUTES
        f.attrs["log_epsilon"] = 1e-6
        if quality_version is not None:
            f.attrs["quality_version"] = np.int64(quality_version)
        if rho_epsilon is not None:
            f.attrs["quality_rho_epsilon"] = np.float64(rho_epsilon)
        for name in ("up", "up_raw"):
            group = f.create_group(name)
            group.attrs["shape"] = (KEPT_RAW,)
            group.attrs["count"] = 1
            group.attrs["mean_scalar"] = 10.0
            group.attrs["std_scalar"] = 2.0
    return path


@pytest.fixture()
def quality_shard(tmp_path: Path) -> Path:
    """A quality shard."""
    return _write_shard(tmp_path / "quality.hdf5")


@pytest.fixture()
def plain_shard(tmp_path: Path) -> Path:
    """The same shard built without the signal-quality step."""
    return _write_shard(tmp_path / "plain.hdf5", quality=False)


def _dataset(paths: Any, **kwargs: Any) -> CombinedHDF5Dataset:
    """A dataset with the caches off, so each test reads what is on disk."""
    options: Dict[str, Any] = dict(
        paths=[str(p) for p in ([paths] if isinstance(paths, Path) else paths)],
        cache_size=0, pin_memory=False, trim_minutes=TRIM_MINUTES,
    )
    options.update(kwargs)
    return CombinedHDF5Dataset(**options)


#: Every quality field the synthetic shard stores.
STORED_QUALITY_FIELDS = sorted(
    ["up_quality", "fhr_quality", "up_raw", "fhr_raw", "up_valid_frac", "up_censored_frac",
     "fhr_valid_frac", "fhr_repair_exact", "fhr_raw_match"] + [f"{block}_rho" for block in WIDTHS]
)


# =================================================================================================
# The opt-in contract
# =================================================================================================
def test_load_fields_none_serves_no_quality_field(quality_shard: Path, plain_shard: Path) -> None:
    """An old caller gets the keys it got before the quality datasets existed."""
    sample = _dataset(quality_shard)[0]
    assert not set(sample) & _QUALITY_FIELDS
    assert set(sample) == set(_dataset(plain_shard)[0])


def test_the_field_groups_name_every_stored_quality_dataset() -> None:
    """The loader's groups cover exactly what the builder writes on a causal file."""
    assert set(STORED_QUALITY_FIELDS) <= _QUALITY_FIELDS
    assert _QUALITY_FIELDS - set(STORED_QUALITY_FIELDS) == {"fhr_up_ph_rho"}


# =================================================================================================
# Named fields: geometry and dtypes
# =================================================================================================
def test_bit_fields_are_trimmed_like_the_raw_signals_and_stay_integer(quality_shard: Path) -> None:
    """``up_quality`` widens uint16 to int32 losslessly; ``fhr_quality`` stays uint8."""
    sample = _dataset(quality_shard, load_fields=["up", "up_quality", "fhr_quality"])[0]
    assert sample["up_quality"].shape == sample["up"].shape == (KEPT_RAW,)
    assert sample["up_quality"].dtype == torch.int32
    assert sample["fhr_quality"].dtype == torch.uint8
    with h5py.File(quality_shard, "r") as f:
        stored_up = f["up_quality"][0][TRIM_RAW:-TRIM_RAW]
        stored_fhr = f["fhr_quality"][0][TRIM_RAW:-TRIM_RAW]
    assert np.array_equal(sample["up_quality"].numpy(), stored_up.astype(np.int32))
    assert np.array_equal(sample["fhr_quality"].numpy(), stored_fhr)
    # The trimmed-away flag is gone and the bit above uint8 survived.
    assert int(sample["up_quality"][2]) == UA_BITS["OSCILLATION"]
    assert int(sample["up_quality"][0]) == UA_BITS["CENSORED"]


def test_pre_fill_signals_are_raw_units_in_the_dataset_dtype(quality_shard: Path) -> None:
    """``*_raw`` is float in ``dtype`` and trimmed; its stored float16 values come back exactly."""
    sample = _dataset(quality_shard, load_fields=["up_raw", "fhr_raw"])[1]
    with h5py.File(quality_shard, "r") as f:
        for name in ("up_raw", "fhr_raw"):
            expected = f[name][1][TRIM_RAW:-TRIM_RAW].astype(np.float32)
            assert sample[name].dtype == torch.float32
            assert np.array_equal(sample[name].numpy(), expected), name


def test_rho_is_trimmed_transposed_and_scaled_like_its_block(quality_shard: Path) -> None:
    r"""$(T, C)$ like the block, $\rho = \text{stored} / 255$ in $[0, 1]$."""
    fields = list(WIDTHS) + [f"{block}_rho" for block in WIDTHS]
    sample = _dataset(quality_shard, load_fields=fields)[2]
    with h5py.File(quality_shard, "r") as f:
        for block, width in WIDTHS.items():
            rho = sample[f"{block}_rho"]
            assert rho.shape == sample[block].shape == (KEPT_STEPS, width), block
            assert rho.dtype == torch.float32
            expected = f[f"{block}_rho"][2][:, TRIM_STEPS:-TRIM_STEPS].T.astype(np.float32) / 255.0
            assert np.allclose(rho.numpy(), expected, rtol=0, atol=1e-7), block
            assert float(rho.min()) >= 0.0 and float(rho.max()) <= 1.0


def test_rho_follows_the_dataset_dtype(quality_shard: Path) -> None:
    """Under ``float16`` the float quality fields are half precision too."""
    sample = _dataset(quality_shard, load_fields=["fhr_st_rho", "up_raw"], dtype=torch.float16)[0]
    assert sample["fhr_st_rho"].dtype == torch.float16
    assert sample["up_raw"].dtype == torch.float16


def test_per_segment_scalars_are_plain_tensors(quality_shard: Path) -> None:
    """Fractions are 0-d floats; ``fhr_repair_exact`` is a 0-d bool."""
    dataset = _dataset(quality_shard, load_fields=["up_valid_frac", "up_censored_frac",
                                                  "fhr_valid_frac", "fhr_repair_exact",
                                                  "fhr_raw_match"])
    for index, exact in enumerate(REPAIR_EXACT):
        sample = dataset[index]
        assert sample["fhr_repair_exact"].dtype == torch.bool
        assert sample["fhr_repair_exact"].dim() == 0
        assert bool(sample["fhr_repair_exact"]) == bool(exact)
        assert float(sample["up_valid_frac"]) == 0.75
        assert float(sample["up_censored_frac"]) == 0.125
        if np.isnan(FHR_RAW_MATCH[index]):
            assert torch.isnan(sample["fhr_raw_match"])
        else:
            assert float(sample["fhr_raw_match"]) == FHR_RAW_MATCH[index]
        assert sample["fhr_valid_frac"].dim() == 0


def test_quality_fields_are_never_normalised(quality_shard: Path, tmp_path: Path) -> None:
    """Even when the stats file and ``normalize_fields`` both name one."""
    stats = _write_stats(tmp_path / "stats.hdf5", quality_version=QUALITY_VERSION)
    dataset = _dataset(quality_shard, load_fields=["up", "up_raw", "fhr_st_rho"],
                       stats_path=str(stats), normalize_fields=["up", "up_raw", "fhr_st_rho"])
    assert dataset.normalization_enabled
    assert "up_raw" not in dataset.normalization_stats
    sample = dataset[0]
    with h5py.File(quality_shard, "r") as f:
        stored_up = f["up"][0][TRIM_RAW:-TRIM_RAW]
        stored_raw = f["up_raw"][0][TRIM_RAW:-TRIM_RAW].astype(np.float32)
    assert np.allclose(sample["up"].numpy(), (stored_up - 10.0) / (2.0 + 1e-8))  # normalised
    assert np.array_equal(sample["up_raw"].numpy(), stored_raw)                 # untouched


# =================================================================================================
# Decoded masks
# =================================================================================================
def test_the_quality_masks_are_off_by_default(quality_shard: Path) -> None:
    sample = _dataset(quality_shard, load_fields=["up", "fhr"])[0]
    assert "up_valid" not in sample and "fhr_valid" not in sample


def test_the_ua_mask_follows_the_invalid_bits(quality_shard: Path) -> None:
    """CENSORED, OSCILLATION, REST and PUSH are valid; CODE, STEP and FLAT are not."""
    sample = _dataset(quality_shard, load_fields=["up"], emit_quality_mask=True)[0]
    mask = sample["up_valid"]
    assert mask.dtype == torch.bool and mask.shape == (KEPT_RAW,)
    assert mask[:4].tolist() == [True, False, True, False]
    assert mask[V4_FLAGS_AT:V4_FLAGS_AT + 2].tolist() == [False, True]
    assert int((~mask).sum()) == 3


def test_the_fhr_mask_is_chosen_per_segment_by_the_repair_source(quality_shard: Path) -> None:
    """HOLD and LINEAR invalidate only where the rules stood in for the repair mask."""
    dataset = _dataset(quality_shard, load_fields=["fhr"], emit_quality_mask=True)
    for index, exact in enumerate(REPAIR_EXACT):
        mask = dataset[index]["fhr_valid"]
        assert mask.shape == (KEPT_RAW,)
        if exact:
            assert mask[:3].tolist() == [True, False, True], index
        else:
            assert mask[:3].tolist() == [False, False, False], index
        assert bool(mask[3:].all()), index


def test_the_masks_need_the_quality_datasets(plain_shard: Path) -> None:
    with pytest.raises(ValueError, match="emit_quality_mask=True"):
        _dataset(plain_shard, emit_quality_mask=True)


# =================================================================================================
# The FHR step mask
# =================================================================================================
#: Fields of the FHR step-mask tests: both FHR blocks and one UA block, which must get no FHR mask.
FHR_LOAD = ["fhr_st", "fhr_ph", "up_ph", "weight", "guid"]


def _fhr_rho_bad(segment: int, block: str, epsilon: float) -> np.ndarray:
    r"""The synthetic shard's stored $\rho$ rule for an FHR block: ``(T, C)`` bool, ``True`` = bad."""
    width = WIDTHS[block]
    codes = (np.arange(width * LEN_SEQUENCE).reshape(width, LEN_SEQUENCE) + segment) % 256
    return (codes[:, TRIM_STEPS:-TRIM_STEPS].astype(np.float64) / 255.0 > epsilon).T


def _fhr_bad_steps(shard: Path, segment: int, **spec: Any) -> List[int]:
    """The kept steps marked bad in one segment's ``fhr_ph`` under a bits-only rule."""
    sample = _dataset(shard, load_fields=FHR_LOAD, fhr_step_mask=spec)[segment]
    return np.flatnonzero(~sample["fhr_ph_step_valid"][:, 0].numpy()).tolist()


@pytest.mark.parametrize("segment", range(N_SEGMENTS))
def test_the_fhr_invalid_alias_follows_the_repair_source(quality_shard: Path, segment: int) -> None:
    """HOLD at raw 240 (kept step 0) and LINEAR at 242 count only under the rules.

    JUMP at raw 241 is invalid under both sets, so kept step 1 is bad in every segment; kept step 0
    is bad only where ``fhr_repair_exact`` is 0 and the rules stood in for the repair mask.
    """
    expected = [1] if REPAIR_EXACT[segment] else [0, 1]
    assert _fhr_bad_steps(quality_shard, segment, bits=["invalid"]) == expected


@pytest.mark.parametrize("bits, expected", [(["HOLD"], [0]), (["JUMP", "LINEAR"], [1]),
                                            (["REPAIRED"], [])])
def test_named_fhr_bits_ignore_the_repair_source(quality_shard: Path, bits, expected) -> None:
    """A named bit is one value for every segment; only ``"invalid"`` reads ``fhr_repair_exact``."""
    for segment in range(N_SEGMENTS):
        assert _fhr_bad_steps(quality_shard, segment, bits=bits) == expected, segment


@pytest.mark.parametrize("segment", [0, 1])
def test_the_fhr_rho_rule_marks_each_coefficient_above_epsilon(
    quality_shard: Path, segment: int
) -> None:
    sample = _dataset(quality_shard, load_fields=FHR_LOAD, fhr_step_mask={"rho_epsilon": 0.1})[segment]
    for block in ("fhr_st", "fhr_ph"):
        valid = sample[f"{block}_step_valid"]
        assert valid.shape == (KEPT_STEPS, WIDTHS[block]) and valid.dtype == torch.bool, block
        assert np.array_equal(valid.numpy(), ~_fhr_rho_bad(segment, block, 0.1)), block


def test_the_fhr_and_ua_step_masks_are_independent(quality_shard: Path) -> None:
    """``fhr_step_mask`` alone emits no UA mask, and only for the FHR blocks the sample loaded."""
    sample = _dataset(quality_shard, load_fields=["fhr_ph", "up_ph", "weight"],
                      fhr_step_mask={"bits": ["invalid"]})[0]
    assert "fhr_ph_step_valid" in sample
    assert "fhr_st_step_valid" not in sample
    assert "up_ph_step_valid" not in sample
    both = _dataset(quality_shard, load_fields=FHR_LOAD, fhr_step_mask={"bits": ["invalid"]},
                    ua_step_mask={"bits": ["invalid"]})[0]
    assert {"fhr_st_step_valid", "fhr_ph_step_valid", "up_ph_step_valid"} <= set(both)


@pytest.mark.parametrize("spec", [None, {"bits": None, "rho_epsilon": None}])
def test_no_fhr_rule_emits_nothing(quality_shard: Path, spec) -> None:
    sample = _dataset(quality_shard, load_fields=FHR_LOAD, fhr_step_mask=spec)[0]
    assert not any(name.endswith("_step_valid") for name in sample)


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"bits": ["invalid"], "pool": "any"}, "unknown keys"),
        ({"bits": ["CODE"]}, "does not define"),       # a UA bit, not an FHR one
        ({"rho_epsilon": 1.0}, "rho_epsilon"),
        ({"rho_epsilon": -0.1}, "rho_epsilon"),
        ({"bits": ["HOLD"], "hold_steps": -1}, "hold_steps"),
    ],
)
def test_a_bad_fhr_step_mask_is_refused(quality_shard: Path, spec, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _dataset(quality_shard, load_fields=FHR_LOAD, fhr_step_mask=spec)


def test_the_fhr_invalid_alias_needs_the_repair_source(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path / "no_repair.hdf5", omit=("fhr_repair_exact",))
    with pytest.raises(ValueError, match="fhr_repair_exact"):
        _dataset(shard, load_fields=FHR_LOAD, fhr_step_mask={"bits": ["invalid"]})


def test_the_fhr_step_mask_needs_a_quality_build(plain_shard: Path) -> None:
    with pytest.raises(ValueError, match="signal-quality build"):
        _dataset(plain_shard, load_fields=FHR_LOAD, fhr_step_mask={"bits": ["invalid"]})


# =================================================================================================
# Coherence of a file list
# =================================================================================================
@pytest.mark.parametrize("quality_first", [True, False])
def test_mixing_quality_and_plain_files_is_refused(
    quality_shard: Path, plain_shard: Path, quality_first: bool
) -> None:
    paths = [quality_shard, plain_shard] if quality_first else [plain_shard, quality_shard]
    with pytest.raises(ValueError, match="Mixed quality and non-quality files") as error:
        _dataset(paths)
    assert str(quality_shard) in str(error.value) and str(plain_shard) in str(error.value)


def test_two_agreeing_quality_files_are_accepted(quality_shard: Path, tmp_path: Path) -> None:
    other = _write_shard(tmp_path / "quality_b.hdf5")
    dataset = _dataset([quality_shard, other], load_fields=["up_quality"])
    assert len(dataset) == 2 * N_SEGMENTS


def test_different_quality_versions_are_refused(quality_shard: Path, tmp_path: Path) -> None:
    other = _write_shard(tmp_path / "quality_v2.hdf5", quality_version=2)
    with pytest.raises(ValueError, match="Mixed quality versions"):
        _dataset([quality_shard, other])


def test_a_different_kernel_digest_is_refused(quality_shard: Path, tmp_path: Path) -> None:
    other = _write_shard(tmp_path / "quality_b.hdf5", kernel_digest="digest-b")
    with pytest.raises(ValueError, match="quality_kernel_digest"):
        _dataset([quality_shard, other])


def test_quality_datasets_without_a_version_are_refused(tmp_path: Path) -> None:
    path = _write_shard(tmp_path / "unversioned.hdf5", quality_version=None)
    with pytest.raises(ValueError, match="no quality_version"):
        _dataset(path)


# =================================================================================================
# Requested fields the files do not store
# =================================================================================================
def test_a_requested_quality_field_on_a_plain_file_is_refused(plain_shard: Path) -> None:
    with pytest.raises(ValueError, match=r"\['up_quality'\].*without the signal-quality step"):
        _dataset(plain_shard, load_fields=["fhr", "up_quality"])


def test_a_requested_quality_field_the_quality_file_lacks_is_refused(tmp_path: Path) -> None:
    path = _write_shard(tmp_path / "partial.hdf5", omit=["up_st_rho"])
    with pytest.raises(ValueError, match="up_st_rho"):
        _dataset(path, load_fields=["up_st", "up_st_rho"])


def test_a_missing_ordinary_field_is_still_skipped(quality_shard: Path) -> None:
    """``fhr_up_ph`` is absent on every causal file and is skipped, as before."""
    sample = _dataset(quality_shard, load_fields=["fhr_st", "fhr_up_ph"])[0]
    assert "fhr_st" in sample and "fhr_up_ph" not in sample


# =================================================================================================
# quality_info
# =================================================================================================
def test_quality_info_describes_the_bit_fields(quality_shard: Path, plain_shard: Path) -> None:
    info = _dataset(quality_shard).quality_info
    assert info is not None
    assert info.version == QUALITY_VERSION and info.rho_scale == 255 and info.lookahead_s == 122.0
    assert info.up_bits == UA_BITS and info.up_invalid_bits == UA_INVALID
    assert info.fhr_bits == FHR_BITS
    assert (info.fhr_invalid_bits_exact, info.fhr_invalid_bits_rules) == (
        FHR_INVALID_EXACT, FHR_INVALID_RULES
    )
    assert info.kernel_digest == "digest-a"
    assert _dataset(plain_shard).quality_info is None


def test_quality_info_cannot_be_changed_from_outside(quality_shard: Path) -> None:
    dataset = _dataset(quality_shard)
    dataset.quality_info.up_bits["MISSING"] = 99
    assert dataset.quality_info.up_bits["MISSING"] == 1


# =================================================================================================
# Statistics pairing
# =================================================================================================
def test_quality_data_with_non_quality_statistics_is_refused(
    quality_shard: Path, tmp_path: Path
) -> None:
    stats = _write_stats(tmp_path / "old_stats.hdf5", quality_version=None)
    with pytest.raises(ValueError, match="Recompute the statistics for this build"):
        _dataset(quality_shard, stats_path=str(stats))


def test_plain_data_with_quality_statistics_is_refused(plain_shard: Path, tmp_path: Path) -> None:
    stats = _write_stats(tmp_path / "quality_stats.hdf5", quality_version=QUALITY_VERSION)
    with pytest.raises(ValueError, match="quality mismatch"):
        _dataset(plain_shard, stats_path=str(stats))


def test_statistics_of_another_quality_version_are_refused(
    quality_shard: Path, tmp_path: Path
) -> None:
    stats = _write_stats(tmp_path / "v2_stats.hdf5", quality_version=2)
    with pytest.raises(ValueError, match="quality_version=2"):
        _dataset(quality_shard, stats_path=str(stats))


def test_matching_quality_statistics_pair_and_report_the_epsilon(
    quality_shard: Path, tmp_path: Path
) -> None:
    stats = _write_stats(tmp_path / "stats.hdf5", quality_version=QUALITY_VERSION, rho_epsilon=0.05)
    dataset = _dataset(quality_shard, stats_path=str(stats))
    assert dataset.normalization_enabled
    assert dataset.stats_quality_rho_epsilon == 0.05


def test_old_statistics_with_old_data_pair_silently(plain_shard: Path, tmp_path: Path) -> None:
    stats = _write_stats(tmp_path / "old_stats.hdf5", quality_version=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all fails this test
        dataset = _dataset(plain_shard, stats_path=str(stats))
    assert dataset.normalization_enabled
    assert dataset.stats_quality_rho_epsilon is None


@pytest.mark.parametrize("key", ["ua_step_mask", "fhr_step_mask"])
def test_a_step_mask_epsilon_other_than_the_statistics_is_refused(
    quality_shard: Path, tmp_path: Path, key: str
) -> None:
    r"""The mask would keep values the normalisation excluded: $\varepsilon = 0.1$ against $0.05$."""
    stats = _write_stats(tmp_path / "stats.hdf5", quality_version=QUALITY_VERSION, rho_epsilon=0.05)
    with pytest.raises(ValueError, match=rf"{key}\.rho_epsilon is 0\.1 .*quality_rho_epsilon=0\.05"):
        _dataset(quality_shard, stats_path=str(stats), **{key: {"rho_epsilon": 0.1}})


@pytest.mark.parametrize("key", ["ua_step_mask", "fhr_step_mask"])
def test_a_step_mask_epsilon_equal_to_the_statistics_pairs(
    quality_shard: Path, tmp_path: Path, key: str
) -> None:
    r"""Equal up to float rounding: $0.1 + 10^{-12}$ against the stats file's float64 $0.1$."""
    stats = _write_stats(tmp_path / "stats.hdf5", quality_version=QUALITY_VERSION, rho_epsilon=0.1)
    dataset = _dataset(quality_shard, stats_path=str(stats), **{key: {"rho_epsilon": 0.1 + 1e-12}})
    assert dataset.stats_quality_rho_epsilon == 0.1


@pytest.mark.parametrize(
    "spec, epsilon", [({"bits": ["invalid"]}, 0.05), ({"rho_epsilon": 0.2}, None)],
    ids=["bits_only", "statistics_without_epsilon"],
)
def test_a_step_mask_without_an_epsilon_pair_is_not_refused(
    quality_shard: Path, tmp_path: Path, spec: Dict[str, Any], epsilon: Optional[float]
) -> None:
    r"""A bits-only mask has no $\varepsilon$ to compare, and old statistics record none."""
    stats = _write_stats(tmp_path / "stats.hdf5", quality_version=QUALITY_VERSION,
                         rho_epsilon=epsilon)
    dataset = _dataset(quality_shard, stats_path=str(stats), ua_step_mask=spec, fhr_step_mask=spec)
    assert dataset.stats_quality_rho_epsilon == epsilon


# =================================================================================================
# Workers and collation
# =================================================================================================
def test_quality_fields_collate_with_their_dtypes(quality_shard: Path) -> None:
    dataset = _dataset(quality_shard, load_fields=["up_quality", "fhr_quality", "fhr_st_rho",
                                                   "fhr_repair_exact", "up_valid_frac"],
                       emit_quality_mask=True)
    batch = next(iter(DataLoader(dataset, batch_size=N_SEGMENTS, collate_fn=attribute_dict_collate)))
    assert batch["up_quality"].dtype == torch.int32
    assert batch["fhr_quality"].dtype == torch.uint8
    assert batch["up_valid"].dtype == torch.bool and batch["fhr_valid"].dtype == torch.bool
    assert batch["fhr_repair_exact"].dtype == torch.bool
    assert batch["fhr_repair_exact"].shape == (N_SEGMENTS,)
    assert batch["fhr_st_rho"].shape == (N_SEGMENTS, KEPT_STEPS, WIDTHS["fhr_st"])
    assert batch["up_valid_frac"].shape == (N_SEGMENTS,)


def test_the_quality_state_survives_pickling_to_a_worker(quality_shard: Path) -> None:
    dataset = _dataset(quality_shard, load_fields=["up_quality", "fhr_st_rho"],
                       emit_quality_mask=True)
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone.quality_info == dataset.quality_info
    for key, value in dataset[1].items():
        if torch.is_tensor(value):
            assert torch.equal(value, clone[1][key]), key


# =================================================================================================
# The legacy writers and the GUID view
# =================================================================================================
def _legacy_rows(k: int) -> Dict[str, Any]:
    """Arguments of the legacy append writers for ``k`` rows, shaped like the synthetic shard."""
    def block(width: int) -> np.ndarray:
        return np.zeros((k, width, LEN_SEQUENCE), dtype=np.float32)

    return dict(
        fhr_batch=np.zeros((k, N_RAW), dtype=np.float32),
        up_batch=np.zeros((k, N_RAW), dtype=np.float32),
        fhr_st_batch=block(WIDTHS["fhr_st"]),
        fhr_ph_batch=block(WIDTHS["fhr_ph"]),
        fhr_up_ph_batch=block(1),
        target_batch=np.zeros((k, LEN_SEQUENCE), dtype=np.float32),
        weight_batch=np.ones((k, LEN_SEQUENCE), dtype=np.float32),
        guid_batch=["g9"] * k,
        epoch_batch=np.zeros(k, dtype=np.float32),
        cs_label_batch=np.zeros(k, dtype=np.uint8),
        bg_label_batch=np.zeros(k, dtype=np.uint8),
        tlo_batch=np.full(k, np.nan, dtype=np.float32),
    )


def test_the_legacy_writers_refuse_a_quality_file(quality_shard: Path) -> None:
    """Their new rows would carry bit code 0, which reads as a clean sample; nothing is written."""
    rows = _legacy_rows(1)
    with pytest.raises(ValueError, match="signal-quality build"):
        append_samples_batch(str(quality_shard), **rows)
    with pytest.raises(ValueError, match="signal-quality build"):
        append_sample(
            str(quality_shard), rows["fhr_batch"][0], rows["up_batch"][0],
            rows["fhr_st_batch"][0], rows["fhr_ph_batch"][0], rows["fhr_up_ph_batch"][0],
            rows["target_batch"][0], rows["weight_batch"][0], "g9", 0.0, False, False,
        )
    with h5py.File(quality_shard, "r") as f:
        assert {name: f[name].shape[0] for name in f} == {name: N_SEGMENTS for name in f}


def test_the_guid_collate_pads_quality_fields_as_not_recorded() -> None:
    r"""A padded segment reads as missing: $\rho = 1$, the MISSING bit, False masks, NaN match."""
    def recording(n: int) -> AttributeDict:
        return AttributeDict(
            num_segments=n, guid=f"g{n}", cs_label=False, bg_label=False,
            fhr_raw_match=torch.full((n,), 0.875),
            up_st_rho=torch.zeros(n, KEPT_STEPS, WIDTHS["up_st"]),
            up_quality=torch.zeros(n, KEPT_RAW, dtype=torch.int32),
            up_valid=torch.ones(n, KEPT_RAW, dtype=torch.bool),
            fhr_st_step_valid=torch.ones(n, KEPT_STEPS, WIDTHS["fhr_st"], dtype=torch.bool),
        )

    batch = sequence_collate_fn([recording(2), recording(1)])
    assert batch["lengths"].tolist() == [2, 1]
    assert torch.isnan(batch["fhr_raw_match"][1, 1])
    assert float(batch["fhr_raw_match"][1, 0]) == 0.875
    assert bool((batch["up_st_rho"][1, 1] == 1.0).all())
    assert bool((batch["up_quality"][1, 1] == 1).all())
    assert batch["up_valid"].dtype == torch.bool and not batch["up_valid"][1, 1].any()
    assert not batch["fhr_st_step_valid"][1, 1].any()
