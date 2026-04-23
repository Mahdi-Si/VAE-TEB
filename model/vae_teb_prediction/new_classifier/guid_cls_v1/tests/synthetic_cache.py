"""Synthetic-cache builder for unit tests.

Writes a minimal HDF5 matching the ``guid_cls_v1`` precompute schema using
only ``numpy`` and ``h5py`` — no torch dependency. Shared between torch-based
and torch-free test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


T = 300
D_MODEL = 128
D_Z = 24
L = 91


def write_synthetic_cache(
    path: Path,
    *,
    num_guids: int = 4,
    segments_per_guid: int = 5,
    base_epoch_seconds: float = -7200.0,
    segment_stride_seconds: float = 1200.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Write a small HDF5 cache mimicking :mod:`precompute_latents` output.

    Args:
        path: Output cache path.
        num_guids: Number of GUIDs to fabricate.
        segments_per_guid: Constant segment count per GUID.
        base_epoch_seconds: Earliest epoch value for the first segment of
            every GUID (negative = before delivery).
        segment_stride_seconds: Δt between consecutive segments.
        seed: RNG seed for reproducibility.

    Returns:
        A small manifest listing the class assignments so tests can assert
        against them.
    """
    rng = np.random.default_rng(seed)
    mu_post_mean = rng.normal(scale=0.1, size=(D_Z,)).astype(np.float32)
    mu_post_var = rng.uniform(0.5, 1.5, size=(D_Z,)).astype(np.float32)
    expected_class_assignments = []

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", libver="latest") as fh:
        fh.attrs["schema_version"] = "v1"
        fh.attrs["vae_checkpoint_sha256"] = "0" * 64
        fh.attrs["vae_checkpoint_path"] = "synthetic"
        fh.attrs["cache_input_signature"] = "synthetic-cache-signature"
        fh.attrs["cache_input_summary_json"] = "{}"
        fh.attrs["use_up_st"] = True
        fh.attrs["d_z"] = D_Z
        fh.attrs["d_model"] = D_MODEL
        fh.attrs["L"] = L
        fh.attrs["T"] = T
        fh.attrs["warmup_period"] = 30
        fh.attrs["partition"] = "train"
        fh.attrs["fold_id"] = 1
        fh.attrs["mu_post_mean"] = mu_post_mean
        fh.attrs["mu_post_var"] = mu_post_var
        fh.attrs["latent_stats_count"] = 1234

        guids_grp = fh.create_group("guids")
        for g in range(num_guids):
            cls = (g % 3) + 1
            expected_class_assignments.append(cls - 1)
            grp = guids_grp.create_group(f"GUID_{g:03d}")
            S = segments_per_guid
            grp.attrs["S"] = S
            grp.create_dataset(
                "h_y", data=rng.normal(size=(S, T, D_MODEL)).astype(np.float16)
            )
            grp.create_dataset(
                "mu_prior", data=rng.normal(size=(S, T, D_Z)).astype(np.float32)
            )
            grp.create_dataset(
                "mu_post",
                data=(
                    rng.normal(size=(S, T, D_Z)).astype(np.float32)
                    * mu_post_var.reshape(1, 1, -1)
                    + mu_post_mean.reshape(1, 1, -1)
                ),
            )
            grp.create_dataset(
                "kld_per_t",
                data=rng.uniform(0, 5, size=(S, T)).astype(np.float32),
            )
            grp.create_dataset(
                "mean_alpha",
                data=rng.dirichlet(np.ones(L), size=(S, T)).astype(np.float16),
            )
            grp.create_dataset(
                "weight",
                data=rng.uniform(0.5, 1.0, size=(S, T)).astype(np.float32),
            )
            target = np.zeros((S, T), dtype=np.int8)
            target[:, 60:200] = cls
            grp.create_dataset("target", data=target)
            epochs = (
                base_epoch_seconds
                + np.arange(S, dtype=np.float64) * segment_stride_seconds
            )
            grp.create_dataset("epoch", data=epochs)
            tlo = np.arange(S, dtype=np.float32) * 600.0
            grp.create_dataset("time_from_labor_onset", data=tlo)
            grp.create_dataset(
                "second_stage_onset",
                data=np.full((S,), np.nan, dtype=np.float32),
            )
            grp.create_dataset(
                "cs_label",
                data=np.full((S,), 1 if g % 2 == 0 else 0, dtype=np.uint8),
            )
            grp.create_dataset(
                "bg_label", data=np.full((S,), 1, dtype=np.uint8)
            )

    return {
        "expected_classes": expected_class_assignments,
        "num_guids": num_guids,
        "segments_per_guid": segments_per_guid,
    }


__all__ = [
    "write_synthetic_cache",
    "T",
    "D_MODEL",
    "D_Z",
    "L",
]
