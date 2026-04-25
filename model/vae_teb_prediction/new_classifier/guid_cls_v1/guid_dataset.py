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

        Layout (per segment):
            ``c_meta = [ψ(τ_lab), m_lab, ψ(τ_sso), m_sso, ι_sso]``

        where ``τ_lab``/``τ_sso`` are TLO/SSO in hours, ``m_*`` is the
        NaN-mask flag, and ``ι_sso`` is 1 once second-stage has begun.

        Args:
            epoch: Per-segment epoch in seconds (sorted ascending = far →
                near delivery). Used only to compute ``cum_h`` / ``dt_h``
                for the *transformer's* relative-time bias — these spans
                are NOT stacked into ``c_meta``.
            tlo: Time from labour onset in seconds (NaN allowed).
            sso: Time from second-stage onset in seconds (NaN allowed).

        Returns:
            ``(c_meta (S, 5), cum_h (S,), gap_ratio_h (S,), dt_h (S,))``.
            ``cum_h``, ``gap_ratio_h`` and ``dt_h`` are returned alongside
            ``c_meta`` because the temporal transformer derives its
            relative-time bucket index from ``cum_h`` and ``gap_ratio`` is
            logged as a diagnostic — but none of these spans are exposed
            to the head as features (they are biased by the quality
            filter on ``epoch[0]``).
        """
        S = len(epoch)
        # Δt in hours from the previous observed segment.
        dt_sec = np.zeros(S, dtype=np.float64)
        if S > 1:
            dt_sec[1:] = epoch[1:] - epoch[:-1]
        dt_h = (dt_sec / SECONDS_PER_HOUR).astype(np.float32)
        # Cumulative monitoring time since first observed segment, in hours.
        cum_h = np.cumsum(dt_h).astype(np.float32)
        # Excess gap ratio (>= 0).
        rho = np.zeros(S, dtype=np.float32)
        rho[1:] = np.maximum(0.0, dt_sec[1:] / SEGMENT_DURATION_SEC - 1.0).astype(
            np.float32
        )

        # TLO and SSO in hours, NaN handling.
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
