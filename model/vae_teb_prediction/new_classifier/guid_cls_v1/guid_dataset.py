"""GUID-level dataset for ``guid_cls_v1``.

Reads the per-fold latent cache produced by :mod:`precompute_latents` and
returns one **whole GUID** per ``__getitem__``. Each GUID sample carries:

* per-segment VAE encodings (``h_y``, ``mu_prior_norm``, ``mu_post_norm``,
  ``kld_per_t``, ``mean_alpha``, ``weight``, ``target``);
* per-segment scalar metadata (``epoch``, ``time_from_labor_onset``,
  ``second_stage_onset``, ``cs_label``, ``bg_label``);
* per-segment causal metadata vector ``c_meta`` (5-d, computed in this
  dataset — never inside the model);
* GUID-level labels (3-class and binary).

Filters applied at construction time (PRD §4.3):

* ``epoch_min`` (default −23000 s) — already enforced at precompute time.
* Cross-delivery exclusion (``epoch + 1260 ≤ 0``) — already enforced.
* ``min_samples_per_guid`` (default 3) — drop GUIDs with fewer valid segments.
* ``min_valid_weight_fraction`` (default 0.1) — drop segments dominated by
  gaps within the central window. Applied per segment after computing the
  classifier-time mask.

The dataset never feeds raw ``epoch`` to the model. It is used only for
ordering segments and for retrospective epoch-binned evaluation.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset


SECONDS_PER_HOUR = 3600.0
SEGMENT_DURATION_SEC = 1200.0  # 20 min nominal stride between consecutive segments
SEGMENT_DECIMATED_STEP_SEC = 4.0


def _psi(x: np.ndarray | float) -> np.ndarray | float:
    """Signed log compression: ``sign(x) * log(1 + |x|)``."""
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.log1p(np.abs(x))
    return math.copysign(math.log1p(abs(x)), x) if x != 0 else 0.0


def _build_c_meta_arrays(
    *,
    epoch: np.ndarray,
    tlo: np.ndarray,
    sso: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Module-level c_meta builder shared by cached + live datasets.

    See :meth:`GuidSequenceDataset._build_c_meta` for the layout (5-d:
    ``[ψ(τ_lab), m_lab, ψ(τ_sso), m_sso, ι_sso]``) and rationale. The cum_h
    / gap_ratio / Δt spans returned alongside are consumed by the temporal
    transformer's relative-time bias only — they are not stacked into
    ``c_meta`` (PRD §4.4).

    Args:
        epoch: Per-segment epoch in seconds (sorted ascending).
        tlo: Time from labour onset in seconds (NaN allowed).
        sso: Time from second-stage onset in seconds (NaN allowed).

    Returns:
        ``(c_meta (S, 5), cum_h (S,), gap_ratio_h (S,), dt_h (S,))``.
    """
    S = len(epoch)
    dt_sec = np.zeros(S, dtype=np.float64)
    if S > 1:
        dt_sec[1:] = epoch[1:] - epoch[:-1]
    dt_h = (dt_sec / SECONDS_PER_HOUR).astype(np.float32)
    cum_h = np.cumsum(dt_h).astype(np.float32)
    rho = np.zeros(S, dtype=np.float32)
    rho[1:] = np.maximum(
        0.0, dt_sec[1:] / SEGMENT_DURATION_SEC - 1.0
    ).astype(np.float32)

    tlo_h = tlo / SECONDS_PER_HOUR
    m_tlo = np.isnan(tlo_h).astype(np.float32)
    tlo_h_clean = np.where(np.isnan(tlo_h), 0.0, tlo_h)

    sso_h = sso / SECONDS_PER_HOUR
    m_sso = np.isnan(sso_h).astype(np.float32)
    sso_h_clean = np.where(np.isnan(sso_h), 0.0, sso_h)
    iota_sso = ((m_sso == 0.0) & (sso_h_clean >= 0.0)).astype(np.float32)

    c_meta = np.stack(
        [
            _psi(tlo_h_clean.astype(np.float64)).astype(np.float32),
            m_tlo,
            _psi(sso_h_clean.astype(np.float64)).astype(np.float32),
            m_sso,
            iota_sso,
        ],
        axis=1,
    )
    return c_meta, cum_h, rho.astype(np.float32), dt_h


def _build_hat_w(
    *,
    weight: np.ndarray,
    epoch: np.ndarray,
    T: int,
    warmup_left: int,
    warmup_right: int,
    cross_delivery_censoring: bool,
) -> np.ndarray:
    """Build the classifier-time per-step mask ``hat_w`` shared by both datasets.

    Args:
        weight: ``(S, T)`` per-step weight in [0, 1].
        epoch: ``(S,)`` per-segment epoch in seconds (relative to delivery).
        T: Per-segment timestep count.
        warmup_left: Central-window left trim in steps.
        warmup_right: Central-window right trim in steps.
        cross_delivery_censoring: When True, zero per-step weight where the
            absolute time exceeds delivery (``epoch + 60 + 4·t > 0``).

    Returns:
        ``(S, T)`` float32 tensor of post-trim per-step weights.
    """
    t_arr = np.arange(T, dtype=np.float32)
    central = (t_arr >= warmup_left) & (t_arr < T - warmup_right)
    hat_w = weight * central[None, :]
    if cross_delivery_censoring:
        per_t_sec = (
            epoch[:, None] + 60.0 + SEGMENT_DECIMATED_STEP_SEC * t_arr[None, :]
        )
        hat_w = hat_w * (per_t_sec <= 0.0).astype(np.float32)
    return hat_w.astype(np.float32)


class GuidSequenceDataset(Dataset):
    """One GUID per item; reads the precomputed VAE-latent cache.

    Args:
        cache_path: Path to the per-fold partition HDF5 produced by
            :mod:`precompute_latents`.
        warmup_left: Per-segment central-window left trim in decimated steps
            (default 30, matches PRD §6.3).
        warmup_right: Per-segment central-window right trim in decimated
            steps (default 30).
        min_samples_per_guid: Drop GUIDs with fewer valid segments after
            filtering (default 3).
        min_valid_weight_fraction: Per-segment central-window minimum
            ``weight > 0`` fraction (default 0.1). Segments below this are
            removed from the GUID before the ``min_samples`` check.
        cross_delivery_censoring: When True, additionally zero per-step
            weight for time steps whose absolute time exceeds delivery
            (defence in depth — precompute already filters cross-delivery
            *segments*).
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        warmup_left: int = 30,
        warmup_right: int = 30,
        min_samples_per_guid: int = 3,
        min_valid_weight_fraction: float = 0.1,
        cross_delivery_censoring: bool = True,
    ) -> None:
        super().__init__()
        self.cache_path = str(cache_path)
        self.warmup_left = int(warmup_left)
        self.warmup_right = int(warmup_right)
        self.min_samples_per_guid = int(min_samples_per_guid)
        self.min_valid_weight_fraction = float(min_valid_weight_fraction)
        self.cross_delivery_censoring = bool(cross_delivery_censoring)

        # File handle is opened lazily per worker to keep the dataset
        # picklable for spawn-based multiprocessing.
        self._fh: Optional[h5py.File] = None
        self._fh_lock = threading.Lock()

        # Build the GUID index + per-segment filter map up front.
        with h5py.File(self.cache_path, "r", libver="latest") as fh:
            self.attrs: Dict[str, Any] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in fh.attrs.items()
            }
            self.d_z: int = int(self.attrs["d_z"])
            self.d_model_vae: int = int(self.attrs["d_model"])
            self.L: int = int(self.attrs["L"])
            self.T: int = int(self.attrs["T"])
            self.warmup_period: int = int(self.attrs["warmup_period"])
            mean = np.asarray(self.attrs["mu_post_mean"], dtype=np.float32)
            var = np.asarray(self.attrs["mu_post_var"], dtype=np.float32)
            self._mu_post_mean = mean
            self._mu_post_std = np.sqrt(var + 1e-5).astype(np.float32)

            guids_grp = fh["guids"]  # type: ignore[assignment]
            kept: List[Tuple[str, np.ndarray]] = []
            dropped_guids = 0
            dropped_low_weight = 0
            for guid in guids_grp.keys():  # type: ignore[union-attr]
                seg_grp = guids_grp[guid]  # type: ignore[index]
                weights = seg_grp["weight"][()]                  # (S, T)
                epochs = seg_grp["epoch"][()].astype(np.float64)  # (S,)
                S = weights.shape[0]
                # Per-segment valid fraction in the central window.
                center = slice(self.warmup_left, self.T - self.warmup_right)
                centre = weights[:, center]
                frac_valid = (centre > 0.5).mean(axis=1)
                seg_keep = frac_valid >= self.min_valid_weight_fraction
                kept_idx = np.nonzero(seg_keep)[0]
                dropped_low_weight += int(S - len(kept_idx))
                if len(kept_idx) < self.min_samples_per_guid:
                    dropped_guids += 1
                    continue
                # Sort by epoch (ascending = earliest first).
                kept_idx = kept_idx[np.argsort(epochs[kept_idx])]
                kept.append((guid, kept_idx))

        logger.info(
            f"GuidSequenceDataset[{self.cache_path}]: "
            f"kept {len(kept)} GUIDs, dropped {dropped_guids} too-short, "
            f"dropped {dropped_low_weight} low-weight segments"
        )
        if not kept:
            raise RuntimeError(
                f"GuidSequenceDataset built from {self.cache_path} is empty "
                f"after filters (min_samples={self.min_samples_per_guid}, "
                f"min_valid_weight_fraction={self.min_valid_weight_fraction})"
            )

        self._index: List[Tuple[str, np.ndarray]] = kept
        self._guid_lengths: List[int] = [int(len(idx)) for _, idx in kept]

        # GUID labels are pre-computed once so class weights and label
        # consistency assertions are cheap.
        self._guid_labels_3: List[int] = []
        self._guid_labels_bin: List[int] = []
        with h5py.File(self.cache_path, "r", libver="latest") as fh:
            guids_grp = fh["guids"]
            for guid, idx in kept:
                seg_grp = guids_grp[guid]
                tgt = seg_grp["target"][()]                       # (S, T) int8|float32
                # Round to int so partial-weight float targets
                # (``class_id * weight`` with weight < 1.0) still resolve to
                # the canonical class id at the max-over-T step. Older int8
                # caches round-trip unchanged.
                seg_targets = np.rint(
                    tgt.max(axis=1).astype(np.float64)
                ).astype(np.int64)                                # (S,)
                kept_targets = seg_targets[idx]
                # All non-zero entries should agree; zeros are gap-only segments
                # that should already be filtered out, but guard anyway.
                non_zero = kept_targets[kept_targets > 0]
                if len(non_zero) == 0:
                    raise RuntimeError(
                        f"GUID {guid!r} has no non-zero per-segment target — "
                        f"data pipeline upstream should not produce this"
                    )
                top = int(non_zero.max())
                # Consistency check: every non-zero segment must agree on the
                # GUID's outcome class.
                disagreement = non_zero[non_zero != top]
                if disagreement.size > 0:
                    raise RuntimeError(
                        f"GUID {guid!r} has inconsistent segment targets "
                        f"({np.unique(non_zero).tolist()}). "
                        f"This breaks the GUID-level label assumption."
                    )
                y3 = top - 1                                       # 0=H, 1=A, 2=HIE
                self._guid_labels_3.append(y3)
                self._guid_labels_bin.append(int(y3 != 0))

    # ------------------------------------------------------------------
    # Pickling helpers (file handle cannot survive spawn).
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Strip the open HDF5 handle and lock before pickling for workers."""
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_lock"] = None  # rebuilt in __setstate__
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and lazily reopen the HDF5 handle on first read."""
        self.__dict__.update(state)
        self._fh = None
        self._fh_lock = threading.Lock()

    def _open(self) -> h5py.File:
        """Open the cache file on demand (per worker process)."""
        if self._fh is None:
            with self._fh_lock:
                if self._fh is None:
                    self._fh = h5py.File(
                        self.cache_path, "r", libver="latest", swmr=True
                    )
        return self._fh

    def close(self) -> None:
        """Close the HDF5 handle (called on garbage collection)."""
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._fh = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        self.close()

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of GUIDs in the dataset."""
        return len(self._index)

    @property
    def guid_lengths(self) -> List[int]:
        """Per-GUID segment counts; consumed by ``LengthBucketSampler``."""
        return list(self._guid_lengths)

    def get_guid_list(self) -> List[str]:
        """Return the ordered list of GUID identifiers."""
        return [g for g, _ in self._index]

    def get_guid_labels_3class(self) -> List[int]:
        """0-based 3-class GUID labels (0=healthy, 1=acidosis, 2=HIE)."""
        return list(self._guid_labels_3)

    def get_guid_labels_binary(self) -> List[int]:
        """Binary GUID labels (0=healthy, 1=unhealthy)."""
        return list(self._guid_labels_bin)

    # ------------------------------------------------------------------
    # Per-GUID feature assembly
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one GUID's full feature bundle.

        Args:
            idx: GUID index in the dataset.

        Returns:
            Dict with the keys consumed by :func:`guid_sequence_collate_fn`.
            Tensors are float32 (model dtype); ``mu_post`` is *already*
            normalised by the per-fold stats baked into the cache.
        """
        guid, segs = self._index[idx]
        fh = self._open()
        seg_grp = fh["guids"][guid]
        sel = list(int(i) for i in segs)
        S = len(sel)

        # Helper: read + slice + cast to float32
        def _read(name: str) -> np.ndarray:
            return np.asarray(seg_grp[name][sel])

        h_y = _read("h_y").astype(np.float32)                  # (S, T, d_model_vae)
        mu_prior_raw = _read("mu_prior").astype(np.float32)     # (S, T, d_z)
        mu_post_raw = _read("mu_post").astype(np.float32)       # (S, T, d_z)
        kld_per_t = _read("kld_per_t").astype(np.float32)       # (S, T)
        mean_alpha = _read("mean_alpha").astype(np.float32)     # (S, T, L)
        weight = _read("weight").astype(np.float32)             # (S, T)
        # ``target`` is stored as float32 by the new precompute
        # (``class_id * weight``). Rounding via ``np.rint`` yields the
        # canonical integer class id; older int8 caches round-trip unchanged.
        target_per_t = np.rint(
            _read("target").astype(np.float64)
        ).astype(np.int64)                                      # (S, T)
        epoch = np.asarray(seg_grp["epoch"][sel], dtype=np.float64)
        tlo = np.asarray(seg_grp["time_from_labor_onset"][sel], dtype=np.float32)
        sso = np.asarray(seg_grp["second_stage_onset"][sel], dtype=np.float32)
        cs = np.asarray(seg_grp["cs_label"][sel], dtype=np.uint8)
        bg = np.asarray(seg_grp["bg_label"][sel], dtype=np.uint8)

        # Normalise BOTH mu_prior and mu_post with the same per-fold stats.
        # Applying identical (mean, std) to both keeps the difference
        # ``mu_post_norm - mu_prior_norm = (mu_post_raw - mu_prior_raw) / std``
        # which matches the description §5.2 semantics: the *raw* posterior
        # delta scaled by the latent std. Without this both terms would mix
        # scales (post is z-scored, prior was raw) and break Δμ.
        mu_post_norm = (mu_post_raw - self._mu_post_mean) / self._mu_post_std
        mu_prior_norm = (mu_prior_raw - self._mu_post_mean) / self._mu_post_std

        # Classifier-time mask (PRD §6.3): central window + per-step weight,
        # plus cross-delivery censoring. Computed per-step in (S, T).
        t_arr = np.arange(self.T, dtype=np.float32)
        # Spec §5.4: 1-indexed ``1[30 < t ≤ 270]`` excludes the first 30 and
        # last 30 steps, leaving the central 240. Translated to 0-indexed
        # numpy indexing this is ``{30, 31, …, 269}``.
        central = ((t_arr >= self.warmup_left) & (t_arr < self.T - self.warmup_right))
        hat_w = weight * central[None, :]
        if self.cross_delivery_censoring:
            # Approximate per-step absolute time in seconds (PRD §6.3 formula).
            per_t_sec = (
                epoch[:, None] + 60.0 + SEGMENT_DECIMATED_STEP_SEC * t_arr[None, :]
            )
            hat_w = hat_w * (per_t_sec <= 0.0).astype(np.float32)

        # Causally-available metadata vector c_meta (5-d) per PRD §4.4 / §6.7.
        # Built from epochs via the chronological order baked into ``segs``.
        # NOTE: per-segment signal-quality summaries (mean of ``hat_w`` and
        # valid-step fraction) are deliberately NOT part of c_meta or g_glob
        # — they describe sensor validity, not physiology, so they would
        # leak monitoring-quality artefacts into the classifier. ``hat_w``
        # itself is still consumed inside the segment tokenizer purely as a
        # masking signal.
        # Cumulative monitoring time, per-segment Δt, and the gap ratio are
        # likewise excluded from c_meta because they all derive from the
        # *spans* of observed segments, which are biased by the dataset's
        # quality filter (a noisier patient has more early segments rejected,
        # so their first surviving ``epoch[0]`` is later, which shrinks every
        # downstream cumulative/span statistic). They are still returned by
        # this method because the temporal transformer's relative-time
        # attention bias derives ``rel_bucket_idx`` from ``cum_h`` — that is
        # a structural pairwise distance, not a per-segment feature.
        c_meta, cum_h, gap_ratio_h, dt_h = self._build_c_meta(
            epoch=epoch,
            tlo=tlo,
            sso=sso,
        )

        sample: Dict[str, Any] = {
            "guid": guid,
            "h_y": torch.from_numpy(h_y),
            "mu_prior_norm": torch.from_numpy(mu_prior_norm),
            "mu_post_norm": torch.from_numpy(mu_post_norm),
            "kld_per_t": torch.from_numpy(kld_per_t),
            "mean_alpha": torch.from_numpy(mean_alpha),
            "weight": torch.from_numpy(weight),
            "hat_w": torch.from_numpy(hat_w.astype(np.float32)),
            "target_per_t": torch.from_numpy(target_per_t),
            "epoch": torch.from_numpy(epoch.astype(np.float64)),
            "time_from_labor_onset": torch.from_numpy(tlo),
            "second_stage_onset": torch.from_numpy(sso),
            "cs_label": torch.from_numpy(cs.astype(np.bool_)),
            "bg_label": torch.from_numpy(bg.astype(np.bool_)),
            "c_meta": torch.from_numpy(c_meta.astype(np.float32)),
            "cum_monitor_hours": torch.from_numpy(cum_h.astype(np.float32)),
            "gap_ratio": torch.from_numpy(gap_ratio_h.astype(np.float32)),
            "delta_t_hours": torch.from_numpy(dt_h.astype(np.float32)),
            "label_3": int(self._guid_labels_3[idx]),
            "label_bin": int(self._guid_labels_bin[idx]),
            "num_segments": S,
        }
        return sample

    def _build_c_meta(
        self,
        *,
        epoch: np.ndarray,
        tlo: np.ndarray,
        sso: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble the 5-d causal metadata vector per segment.

        Thin wrapper around :func:`_build_c_meta_arrays`; kept on the class
        for backward compatibility with code that calls
        ``GuidSequenceDataset._build_c_meta``. See the module-level helper
        for the layout and the rationale on why span-derived stats live
        outside ``c_meta``.
        """
        return _build_c_meta_arrays(epoch=epoch, tlo=tlo, sso=sso)


class LiveGuidSequenceDataset(Dataset):
    """One GUID per item; reads raw VAE inputs from per-fold partition HDF5s.

    Mirror of :class:`GuidSequenceDataset` for the live-VAE training path.
    Each sample carries the same per-segment metadata + ``c_meta`` /
    ``hat_w`` as the cached dataset, plus the four raw VAE input streams
    (``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``) needed by
    ``vae.encode_only``. The classifier's
    :meth:`GuidOutcomeClassifier.live_forward` runs the VAE per batch and
    z-scores the resulting ``mu_post`` / ``mu_prior`` with the VAE's
    running stats — equivalent to what the precompute path bakes into the
    cache.

    Filters mirror the cached path:

    * ``epoch_min`` (default −23000 s) and ``epoch_max`` (cross-delivery
      −1260 s) — applied at the segment level by
      :class:`CombinedHDF5Dataset`.
    * ``min_valid_weight_fraction`` — applied per segment before the
      ``min_samples_per_guid`` check, exactly like the cached dataset.
    * ``min_samples_per_guid`` — drop GUIDs with fewer surviving segments.

    Args:
        files: HDF5 partition file paths (e.g. from
            :func:`precompute_latents.get_fold_partition_files`).
        T: Per-segment timestep count after trim. Must match the VAE's
            ``sequence_length`` (default 300).
        warmup_left: Per-segment central-window left trim.
        warmup_right: Per-segment central-window right trim.
        min_samples_per_guid: Drop GUIDs with fewer valid segments.
        min_valid_weight_fraction: Per-segment central-window minimum
            ``weight > 0`` fraction.
        cross_delivery_censoring: Whether to zero per-step weight past
            delivery time (defence in depth).
        epoch_min: Optional ``epoch_min`` filter forwarded to
            :class:`CombinedHDF5Dataset`.
        trim_minutes: Trim before reaching the VAE; must match the
            precompute setting (default 1.0 min).
        stats_path: Stats HDF5 used to normalise ``fhr_st``/``fhr_ph``/
            ``up_st``/``up_ph`` before the VAE forward.
        normalize_fields: Fields to normalise (defaults to the four raw
            scattering streams).
    """

    DEFAULT_LOAD_FIELDS: Tuple[str, ...] = (
        "fhr_st",
        "fhr_ph",
        "up_st",
        "up_ph",
        "weight",
        "target",
        "epoch",
        "cs_label",
        "bg_label",
        "guid",
        "time_from_labor_onset",
        "second_stage_onset",
    )

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        T: int = 300,
        warmup_left: int = 30,
        warmup_right: int = 30,
        min_samples_per_guid: int = 3,
        min_valid_weight_fraction: float = 0.1,
        cross_delivery_censoring: bool = True,
        epoch_min: Optional[float] = None,
        trim_minutes: float = 1.0,
        stats_path: Optional[str] = None,
        normalize_fields: Optional[Sequence[str]] = None,
        # Cached-dataset-compatible attrs (filled at init from the first
        # underlying sample so the classifier's ``from_cache_attrs`` path
        # also works for the live dataset).
        d_model_vae: int = 128,
        d_z: int = 24,
    ) -> None:
        super().__init__()
        # Lazy import to avoid pulling the heavy HDF5 layer when the live
        # path is not used.
        from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset  # noqa: WPS433

        self.files = [str(p) for p in files]
        self.T = int(T)
        self.warmup_left = int(warmup_left)
        self.warmup_right = int(warmup_right)
        self.min_samples_per_guid = int(min_samples_per_guid)
        self.min_valid_weight_fraction = float(min_valid_weight_fraction)
        self.cross_delivery_censoring = bool(cross_delivery_censoring)

        # Reproduce the precompute filtering exactly so the live VAE sees
        # the same input distribution as the cached path used.
        if normalize_fields is None:
            normalize_fields = ["fhr_st", "fhr_ph", "up_st", "up_ph"]
        self._underlying = CombinedHDF5Dataset(
            paths=self.files,
            load_fields=list(self.DEFAULT_LOAD_FIELDS),
            epoch_min=epoch_min,
            epoch_max=-1260.0,        # cross-delivery cut
            cache_size=0,
            pin_memory=False,
            stats_path=stats_path,
            normalize_fields=list(normalize_fields),
            trim_minutes=float(trim_minutes),
        )

        # These dimensions are exposed for compatibility with
        # ``GuidOutcomeClassifier.from_cache_attrs`` and the trainer's
        # ``train_ds.d_model_vae`` / ``train_ds.d_z`` lookup; they default
        # to the standard VAE setting and can be overridden.
        self.d_model_vae: int = int(d_model_vae)
        self.d_z: int = int(d_z)

        # Build the per-GUID index by reading metadata directly from h5py
        # (one fancy-indexed read per file). Iterating the underlying
        # dataset's __getitem__ would be ~100× slower because each call
        # reads + normalises full FHR/UP tensors.
        self._index: List[Tuple[str, List[int]]] = []
        self._guid_lengths: List[int] = []
        self._guid_labels_3: List[int] = []
        self._guid_labels_bin: List[int] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Populate ``_index`` / ``_guid_lengths`` / labels by scanning HDF5."""
        # Group underlying-dataset row indices by source file to amortise
        # the h5py fancy-indexed reads.
        per_file: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for combined_idx, (file_idx, sample_idx) in enumerate(
            self._underlying.index_map
        ):
            per_file[file_idx].append((combined_idx, sample_idx))

        # Per-segment scalar metadata, keyed by combined dataset index.
        # We deliberately do NOT cache the per-segment ``weight`` / ``target``
        # arrays here — those are re-read on demand via the underlying
        # dataset in ``__getitem__``, so caching them would inflate the
        # pickled state crossing worker boundaries by 100s of MB.
        per_seg_meta: Dict[int, Dict[str, Any]] = {}
        # Class id derived per segment at init time (used only for label
        # derivation + consistency checks; thrown away once the GUID
        # label is computed).
        per_seg_top_class: Dict[int, int] = {}

        for file_idx, pairs in per_file.items():
            pairs.sort(key=lambda x: x[1])  # h5py prefers sorted reads
            sample_indices = [p[1] for p in pairs]
            combined_indices = [p[0] for p in pairs]
            path = self._underlying.paths[file_idx]
            with h5py.File(path, "r", libver="latest") as fh:
                guids_raw = fh["guid"][sample_indices]
                epochs = fh["epoch"][sample_indices].astype(np.float64)
                weights = fh["weight"][sample_indices]                # (N, T)
                targets = fh["target"][sample_indices]                 # (N, T)
                tlo = fh["time_from_labor_onset"][sample_indices].astype(
                    np.float32
                )
                if "second_stage_onset" in fh:
                    sso = fh["second_stage_onset"][sample_indices].astype(
                        np.float32
                    )
                else:
                    sso = np.full(len(sample_indices), np.nan, dtype=np.float32)
                cs = fh["cs_label"][sample_indices].astype(np.uint8)
                bg = fh["bg_label"][sample_indices].astype(np.uint8)

            T_local = int(weights.shape[-1])
            center = slice(self.warmup_left, T_local - self.warmup_right)
            frac_valid_arr = (weights[:, center] > 0.5).mean(axis=1)
            for k, combined_idx in enumerate(combined_indices):
                if float(frac_valid_arr[k]) < self.min_valid_weight_fraction:
                    continue
                guid = (
                    guids_raw[k].decode("utf-8")
                    if isinstance(guids_raw[k], bytes)
                    else str(guids_raw[k])
                )
                per_seg_meta[combined_idx] = {
                    "guid": guid,
                    "epoch": float(epochs[k]),
                    "tlo": float(tlo[k]),
                    "sso": float(sso[k]),
                    "cs": int(cs[k]),
                    "bg": int(bg[k]),
                }
                seg_targets = np.rint(
                    np.asarray(targets[k], dtype=np.float64)
                ).astype(np.int64)
                tops = seg_targets[seg_targets > 0]
                per_seg_top_class[combined_idx] = (
                    int(tops.max()) if tops.size > 0 else 0
                )

        # Group surviving segments by GUID, sort by epoch.
        guid_to_segs: Dict[str, List[int]] = defaultdict(list)
        for combined_idx, meta in per_seg_meta.items():
            guid_to_segs[meta["guid"]].append(combined_idx)

        dropped_guids = 0
        for guid, indices in guid_to_segs.items():
            if len(indices) < self.min_samples_per_guid:
                dropped_guids += 1
                continue
            indices.sort(key=lambda ci: per_seg_meta[ci]["epoch"])

            # Derive GUID-level label from per-segment top classes (same
            # rule as the cached dataset: every non-zero segment must
            # agree on the GUID's outcome class).
            non_zero_classes = [
                per_seg_top_class[ci]
                for ci in indices
                if per_seg_top_class[ci] > 0
            ]
            if not non_zero_classes:
                raise RuntimeError(
                    f"GUID {guid!r} has no non-zero per-segment target — "
                    f"data pipeline upstream should not produce this"
                )
            top = int(max(non_zero_classes))
            disagreements = [c for c in non_zero_classes if c != top]
            if disagreements:
                raise RuntimeError(
                    f"GUID {guid!r} has inconsistent segment targets "
                    f"({sorted(set(non_zero_classes))}). "
                    f"This breaks the GUID-level label assumption."
                )
            y3 = top - 1                                               # 0=H, 1=A, 2=HIE

            self._index.append((guid, list(indices)))
            self._guid_lengths.append(len(indices))
            self._guid_labels_3.append(y3)
            self._guid_labels_bin.append(int(y3 != 0))

        # Cache the trimmed per-segment scalar metadata for __getitem__.
        self._per_seg_meta = per_seg_meta

        logger.info(
            f"LiveGuidSequenceDataset[files={len(self.files)}]: kept "
            f"{len(self._index)} GUIDs, dropped {dropped_guids} too-short, "
            f"{len(per_seg_meta) - sum(self._guid_lengths)} segments not in any "
            f"surviving GUID"
        )
        if not self._index:
            raise RuntimeError(
                f"LiveGuidSequenceDataset built from {self.files} is empty "
                f"after filters (min_samples={self.min_samples_per_guid}, "
                f"min_valid_weight_fraction={self.min_valid_weight_fraction})"
            )

    # ------------------------------------------------------------------
    # Pickling helpers (the underlying dataset already strips file handles)
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    @property
    def guid_lengths(self) -> List[int]:
        """Per-GUID segment counts; consumed by ``LengthBucketSampler``."""
        return list(self._guid_lengths)

    def get_guid_list(self) -> List[str]:
        return [g for g, _ in self._index]

    def get_guid_labels_3class(self) -> List[int]:
        return list(self._guid_labels_3)

    def get_guid_labels_binary(self) -> List[int]:
        return list(self._guid_labels_bin)

    # ------------------------------------------------------------------
    # Per-GUID feature assembly
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one GUID's full raw-signal bundle for the live VAE path."""
        guid, indices = self._index[idx]
        S = len(indices)

        # Read each segment via the underlying dataset (handles
        # normalisation + trim + tensor creation). Each call returns an
        # AttributeDict.
        per_seg = [self._underlying[ci] for ci in indices]

        fhr_st = torch.stack([s.fhr_st for s in per_seg], dim=0)        # (S, T, 43)
        fhr_ph = torch.stack([s.fhr_ph for s in per_seg], dim=0)        # (S, T, 44)
        up_st = torch.stack([s.up_st for s in per_seg], dim=0)          # (S, T, 43)
        up_ph = torch.stack([s.up_ph for s in per_seg], dim=0)          # (S, T, 58)
        weight = torch.stack([s.weight for s in per_seg], dim=0).float()
        target_per_t = torch.stack(
            [s.target for s in per_seg], dim=0
        ).float()                                                       # (S, T)
        target_per_t = torch.round(target_per_t).long()

        epoch = np.asarray(
            [self._per_seg_meta[ci]["epoch"] for ci in indices], dtype=np.float64
        )
        tlo = np.asarray(
            [self._per_seg_meta[ci]["tlo"] for ci in indices], dtype=np.float32
        )
        sso = np.asarray(
            [self._per_seg_meta[ci]["sso"] for ci in indices], dtype=np.float32
        )
        cs = np.asarray(
            [self._per_seg_meta[ci]["cs"] for ci in indices], dtype=np.uint8
        )
        bg = np.asarray(
            [self._per_seg_meta[ci]["bg"] for ci in indices], dtype=np.uint8
        )

        T = int(weight.shape[-1])
        hat_w = _build_hat_w(
            weight=weight.numpy(),
            epoch=epoch,
            T=T,
            warmup_left=self.warmup_left,
            warmup_right=self.warmup_right,
            cross_delivery_censoring=self.cross_delivery_censoring,
        )                                                               # (S, T)

        c_meta, cum_h, gap_ratio_h, dt_h = _build_c_meta_arrays(
            epoch=epoch, tlo=tlo, sso=sso
        )

        return {
            "guid": guid,
            # Raw VAE inputs — pad-aware collate keys these as
            # (B, N_max, T, C). Live forward chunks them per batch.
            "fhr_st": fhr_st,
            "fhr_ph": fhr_ph,
            "up_st": up_st,
            "up_ph": up_ph,
            "weight": weight,
            "hat_w": torch.from_numpy(hat_w),
            "target_per_t": target_per_t,
            "epoch": torch.from_numpy(epoch.astype(np.float64)),
            "time_from_labor_onset": torch.from_numpy(tlo),
            "second_stage_onset": torch.from_numpy(sso),
            "cs_label": torch.from_numpy(cs.astype(np.bool_)),
            "bg_label": torch.from_numpy(bg.astype(np.bool_)),
            "c_meta": torch.from_numpy(c_meta.astype(np.float32)),
            "cum_monitor_hours": torch.from_numpy(cum_h.astype(np.float32)),
            "gap_ratio": torch.from_numpy(gap_ratio_h.astype(np.float32)),
            "delta_t_hours": torch.from_numpy(dt_h.astype(np.float32)),
            "label_3": int(self._guid_labels_3[idx]),
            "label_bin": int(self._guid_labels_bin[idx]),
            "num_segments": S,
        }


def estimate_inverse_frequency_weights(
    labels: Sequence[int], num_classes: int
) -> List[float]:
    """Inverse-frequency class weights at the GUID level.

    ``alpha_c = N_total / (num_classes * N_class_c)``. Classes absent from
    ``labels`` receive weight 1.0 to keep downstream code simple.

    Args:
        labels: Iterable of class-id integers in ``[0, num_classes)``.
        num_classes: Number of distinct classes.

    Returns:
        Length-``num_classes`` list of float weights.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = counts.sum()
    weights: List[float] = []
    for c in range(num_classes):
        n_c = int(counts[c])
        if n_c == 0:
            weights.append(1.0)
        else:
            weights.append(float(total) / float(num_classes * n_c))
    return weights


__all__ = [
    "GuidSequenceDataset",
    "estimate_inverse_frequency_weights",
]
