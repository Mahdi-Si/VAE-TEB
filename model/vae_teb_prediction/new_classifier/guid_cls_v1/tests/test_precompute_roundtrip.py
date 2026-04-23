"""Roundtrip test: synthetic cache HDF5 → GuidSequenceDataset → collate.

Bypasses the VAE forward pass (which requires a real checkpoint) and uses the
shared :func:`write_synthetic_cache` helper to exercise the dataset and
collate function end-to-end. Skipped automatically when torch isn't
installed (e.g. in lightweight editing envs).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (  # noqa: E402
    build_relative_time_bucket_index,
    guid_sequence_collate_fn,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (  # noqa: E402
    GuidSequenceDataset,
    estimate_inverse_frequency_weights,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.tests.synthetic_cache import (  # noqa: E402
    D_MODEL,
    D_Z,
    L,
    T,
    write_synthetic_cache,
)


@pytest.fixture()
def synthetic_cache(tmp_path: Path) -> Path:
    cache_path = tmp_path / "fold_1" / "train.hdf5"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_cache(cache_path, num_guids=6, segments_per_guid=4)
    return cache_path


def test_dataset_loads_synthetic_cache(synthetic_cache: Path) -> None:
    """GuidSequenceDataset must load every synthetic GUID and expose labels."""
    ds = GuidSequenceDataset(
        synthetic_cache,
        warmup_left=30,
        warmup_right=30,
        min_samples_per_guid=3,
        min_valid_weight_fraction=0.0,
        cross_delivery_censoring=True,
    )
    assert len(ds) == 6
    labels_3 = ds.get_guid_labels_3class()
    assert sorted(set(labels_3)) == [0, 1, 2]
    labels_bin = ds.get_guid_labels_binary()
    assert all(b == int(l != 0) for l, b in zip(labels_3, labels_bin))
    assert ds.guid_lengths == [4] * 6


def test_dataset_filters_too_short_guids(tmp_path: Path) -> None:
    """A GUID with fewer than min_samples valid segments must be dropped."""
    cache_path = tmp_path / "fold_1" / "train.hdf5"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_cache(cache_path, num_guids=4, segments_per_guid=2)
    ds = GuidSequenceDataset(
        cache_path,
        min_samples_per_guid=3,
    )
    assert len(ds) == 0 or all(s >= 3 for s in ds.guid_lengths)


def test_getitem_shapes(synthetic_cache: Path) -> None:
    """Sample tensors must match the schema declared by the dataset."""
    ds = GuidSequenceDataset(synthetic_cache, min_samples_per_guid=3)
    s = ds[0]
    assert s["h_y"].shape == (4, T, D_MODEL)
    assert s["mu_prior_norm"].shape == (4, T, D_Z)
    assert s["mu_post_norm"].shape == (4, T, D_Z)
    assert s["kld_per_t"].shape == (4, T)
    assert s["mean_alpha"].shape == (4, T, L)
    assert s["weight"].shape == (4, T)
    assert s["hat_w"].shape == (4, T)
    assert s["c_meta"].shape == (4, 10)
    assert s["cum_monitor_hours"].shape == (4,)
    assert s["delta_t_hours"].shape == (4,)
    assert s["target_per_t"].shape == (4, T)
    assert s["label_3"] in {0, 1, 2}
    assert s["label_bin"] in {0, 1}


def test_collate_pads_to_max_n(synthetic_cache: Path) -> None:
    """Mixing GUIDs of different lengths must produce a uniform batch."""
    ds = GuidSequenceDataset(synthetic_cache, min_samples_per_guid=3)
    s0 = ds[0]
    s1 = ds[1]
    # Truncate s1 to 3 segments to simulate a shorter GUID.
    short = {k: v for k, v in s1.items()}
    n_short = 3
    for key in [
        "h_y", "mu_prior_norm", "mu_post_norm", "kld_per_t", "mean_alpha", "weight",
        "hat_w", "target_per_t", "c_meta", "cum_monitor_hours", "gap_ratio",
        "delta_t_hours", "bar_w_segment", "f_valid_segment", "cs_label",
        "bg_label", "time_from_labor_onset", "second_stage_onset", "epoch",
    ]:
        short[key] = short[key][:n_short]
    short["num_segments"] = n_short

    batch = guid_sequence_collate_fn([s0, short])
    assert batch["h_y"].shape == (2, 4, T, D_MODEL)
    assert batch["segment_mask"].shape == (2, 4)
    assert batch["segment_mask"][0].sum().item() == 4
    assert batch["segment_mask"][1].sum().item() == 3
    assert batch["rel_bucket_idx"].shape == (2, 4, 4)
    # Padding-row tokens should not bleed any data into the model: the model
    # zeros them via segment_mask, but the collate already zero-fills.
    assert torch.all(batch["h_y"][1, 3] == 0)


def test_relative_time_bucket_known_pattern() -> None:
    """A constant cum_h yields bucket 0 everywhere."""
    cum_h = torch.zeros(2, 5)
    idx = build_relative_time_bucket_index(cum_h, num_buckets=32, d_max=40.0)
    assert torch.all(idx == 0)

    # A pair separated by Δt = d_max * 20min should saturate to last bucket.
    cum_h = torch.zeros(1, 2)
    cum_h[0, 1] = 40.0 * (1200.0 / 3600.0)  # 40 slots in hours
    idx = build_relative_time_bucket_index(cum_h, num_buckets=32, d_max=40.0)
    assert idx[0, 0, 1].item() == 31


def test_inverse_frequency_weights() -> None:
    """Inverse-frequency weights sum to num_classes when all classes seen."""
    weights = estimate_inverse_frequency_weights([0, 0, 1, 2, 2, 2], num_classes=3)
    assert pytest.approx(sum(weights), rel=1e-6) == 3.0


def test_cross_delivery_censoring_zeros_late_steps(tmp_path: Path) -> None:
    """Per-step ``hat_w`` must be zero past delivery within a kept segment."""
    cache_path = tmp_path / "fold_1" / "train.hdf5"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Choose epochs such that the last segment straddles delivery.
    write_synthetic_cache(
        cache_path,
        num_guids=2,
        segments_per_guid=3,
        base_epoch_seconds=-2520.0,  # so segments at -2520, -1320, -120 (last crosses)
        segment_stride_seconds=1200.0,
    )
    ds = GuidSequenceDataset(
        cache_path,
        cross_delivery_censoring=True,
        min_samples_per_guid=3,
    )
    if len(ds) == 0:
        pytest.skip("All GUIDs filtered out by precompute-time cross-delivery rule")
    s = ds[0]
    last_epoch = float(s["epoch"][-1].item())
    if last_epoch + 1260.0 <= 0:
        pytest.skip("This synthetic GUID does not actually straddle delivery")
    # Trailing decimated steps with absolute time > 0 must be zeroed.
    assert s["hat_w"][-1, -1].item() == 0.0


def test_mu_norm_delta_equals_raw_delta_over_std(synthetic_cache: Path) -> None:
    """``mu_post_norm - mu_prior_norm`` must equal ``(mu_post - mu_prior) / std``.

    This is the contract that ``segment_tokenizer`` relies on so that ``Δμ``
    has the description's intended semantics: the *raw* posterior delta
    scaled by the per-fold latent standard deviation.
    """
    ds = GuidSequenceDataset(synthetic_cache, min_samples_per_guid=3)
    s = ds[0]
    mu_post_norm = s["mu_post_norm"]
    mu_prior_norm = s["mu_prior_norm"]
    diff_norm = mu_post_norm - mu_prior_norm
    # Reconstruct the raw difference from the per-fold std.
    std = torch.from_numpy(ds._mu_post_std)            # (d_z,)
    # Read the raw values back through h5py to compare.
    with h5py.File(synthetic_cache, "r", libver="latest") as fh:
        raw_post = fh["guids"][ds.get_guid_list()[0]]["mu_post"][()]
        raw_prior = fh["guids"][ds.get_guid_list()[0]]["mu_prior"][()]
    raw_diff = torch.from_numpy(raw_post - raw_prior).float()
    expected = raw_diff / std.view(1, 1, -1)
    assert torch.allclose(diff_norm, expected, atol=1e-4)


def test_label_consistency_assertion(tmp_path: Path) -> None:
    """If a single GUID has mixed non-zero targets, the dataset must raise."""
    cache_path = tmp_path / "fold_1" / "train.hdf5"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_cache(cache_path, num_guids=1, segments_per_guid=3)
    # Manually introduce a mismatch.
    with h5py.File(cache_path, "r+", libver="latest") as fh:
        guid = list(fh["guids"].keys())[0]
        target = fh["guids"][guid]["target"][:]
        target[1, 60:200] = 3  # different from class assigned to other segments
        del fh["guids"][guid]["target"]
        fh["guids"][guid].create_dataset("target", data=target)

    with pytest.raises(RuntimeError, match="inconsistent segment targets"):
        GuidSequenceDataset(cache_path, min_samples_per_guid=3)
