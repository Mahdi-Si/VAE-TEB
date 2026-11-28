"""Unit tests for ``hdf5_dataset.new_pipeline.build_fold_datasets.is_complete``."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hdf5_dataset.new_pipeline import build_fold_datasets as bfd


def _write_shard(path: Path, attrs: Mapping[str, Any]) -> str:
    """Create a minimal HDF5 file carrying only the given root attributes.

    Args:
        path: Output HDF5 file path.
        attrs: Mapping of root attribute names to values.

    Returns:
        String path of the written HDF5 file.
    """
    with h5py.File(path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
    return str(path)


def _causal_attrs(**overrides: Any) -> dict[str, Any]:
    """Return a baseline set of causal signal-quality root attributes.

    Args:
        **overrides: Attribute values replacing or deleting (when omitted via caller) defaults.

    Returns:
        Dictionary of root HDF5 attributes.
    """
    attrs: dict[str, Any] = {
        "source_guid_digest": "digest-abc",
        "source_pickle_path": "/tmp/fold_cv.pickle",
        "transform": "causal",
        "causal_leg_alignment": "envelope",
        "causal_phase_operator": "integer_harmonic_v1",
        "quality_version": 4,
        "quality_box_valid": np.asarray(False, dtype=np.float64),
    }
    attrs.update(overrides)
    return attrs


def test_is_complete_matches_identical_causal_quality_build(tmp_path: Path) -> None:
    """Matching GUID digest, pickle, transform, phase operator, leg alignment, and quality rules."""
    shard = _write_shard(tmp_path / "ok.hdf5", _causal_attrs())
    assert bfd.is_complete(
        shard,
        "digest-abc",
        "/tmp/fold_cv.pickle",
        quality_version=4,
        transform="causal",
        leg_alignment="envelope",
        phase_operator="integer_harmonic_v1",
        quality_valid_boxes=False,
    )


def test_is_complete_rejects_mismatched_transform_and_causal_options(tmp_path: Path) -> None:
    """Stale shards from another transform, leg alignment, or phase operator must be rebuilt."""
    shard = _write_shard(tmp_path / "causal.hdf5", _causal_attrs())
    base = dict(
        digest="digest-abc",
        source_pickle_path="/tmp/fold_cv.pickle",
        quality_version=4,
        transform="causal",
        leg_alignment="envelope",
        phase_operator="integer_harmonic_v1",
        quality_valid_boxes=False,
    )
    assert not bfd.is_complete(shard, **{**base, "transform": "two_sided"})
    assert not bfd.is_complete(shard, **{**base, "leg_alignment": "none"})
    assert not bfd.is_complete(shard, **{**base, "phase_operator": "ratio_power_v0"})
    assert not bfd.is_complete(shard, **{**base, "quality_valid_boxes": True})
    assert not bfd.is_complete(shard, **{**base, "quality_version": None})


def test_is_complete_ignores_causal_and_box_attrs_for_two_sided_no_quality(tmp_path: Path) -> None:
    """Two-sided shards without signal quality omit causal and quality attributes."""
    shard = _write_shard(
        tmp_path / "two_sided.hdf5",
        {
            "source_guid_digest": "digest-2s",
            "source_pickle_path": "/tmp/fold_cv.pickle",
            "transform": "two_sided",
        },
    )
    assert bfd.is_complete(
        shard,
        "digest-2s",
        "/tmp/fold_cv.pickle",
        quality_version=None,
        transform="two_sided",
        leg_alignment="none",
        phase_operator="ratio_power_v0",
        quality_valid_boxes=False,
    )
    assert not bfd.is_complete(
        shard,
        "digest-2s",
        "/tmp/fold_cv.pickle",
        quality_version=None,
        transform="causal",
        leg_alignment="none",
        phase_operator="ratio_power_v0",
    )


def test_is_complete_rejects_missing_causal_or_box_attrs_when_expected(tmp_path: Path) -> None:
    """Legacy shards missing ``causal_phase_operator`` or ``quality_box_valid`` are rebuilt."""
    attrs = _causal_attrs()
    del attrs["causal_phase_operator"]
    shard_no_op = _write_shard(tmp_path / "no_op.hdf5", attrs)
    assert not bfd.is_complete(
        shard_no_op,
        "digest-abc",
        "/tmp/fold_cv.pickle",
        quality_version=4,
        transform="causal",
        leg_alignment="envelope",
        phase_operator="integer_harmonic_v1",
        quality_valid_boxes=False,
    )

    attrs_no_box = _causal_attrs()
    del attrs_no_box["quality_box_valid"]
    shard_no_box = _write_shard(tmp_path / "no_box.hdf5", attrs_no_box)
    assert not bfd.is_complete(
        shard_no_box,
        "digest-abc",
        "/tmp/fold_cv.pickle",
        quality_version=4,
        transform="causal",
        leg_alignment="envelope",
        phase_operator="integer_harmonic_v1",
        quality_valid_boxes=False,
    )
