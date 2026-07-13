r"""``SyntheticRawDatasetV4`` -- cached-dataset loader for the ``synthetic_v4`` raw cache.

A pure loader over one ``.npz`` written by :func:`build_dataset_v4.build_all_v4`, exposing each
sample as an :class:`AttributeDict` with the raw model's native field names so a batch is drop-in
for :meth:`SeqVaeRawV4._default_batch_to_inputs` (``.fhr`` / ``.up`` / ``.weight``) and the standard
``model_raw/testing`` surface:

    * ``fhr`` $(5280,)$, ``up`` $(5280,)$ -- the normalised raw $4\,\mathrm{Hz}$ waveforms.
    * ``weight`` $(330,)$ -- decimated validity (all-ones unless a gap was planted).
    * ``target`` $(330,)$ -- a label-only placeholder (synthetic data has no classifier label).
    * ``te_true``, ``delay``, ``cell_id``, ``held_out``, ``raw_index``, ``true_lag_tt``, ``guid``
      -- per-sample metadata / ground truth.

The v4 cache stores the **normalised** ``fhr``/``up`` (a single global scalar z-score per stream fit
on the train pool -- see :mod:`build_dataset_v4`), matching the raw model's loader-normalised input
contract, so the dataset applies no further normalisation. Unlike the v2 loader there is **no
``raw_provider``**: the raw waveforms are the primary cached fields, so nothing is regenerated per
item (the on-demand regenerator :func:`build_dataset_v4.make_raw_provider_v4` is a standalone
utility for the deferred CMI, not wired here).

:class:`AttributeDict`, :func:`attribute_dict_collate`, :func:`make_dataloader_v4` mirror the tiny
local copies in :mod:`dataset_v2` (kept local to keep the package standalone). The ZIP-header mmap
plumbing (``_read_npz_member_specs`` / ``_MmapUnsupported``) is pure-numpy with no scattering deps
and is imported from :mod:`dataset_v2` unchanged rather than recopied.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v2 import (
    _MemberSpec,
    _MmapUnsupported,
    _read_npz_member_specs,
)

#: Raw tensor fields stored in every split ``.npz`` (native raw-model field names).
_FIELDS_V4 = ("fhr", "up", "weight", "target")

#: Large per-timestep arrays worth memory-mapping ($O(n \cdot L)$ bytes).
_MMAP_FIELDS_V4 = _FIELDS_V4 + ("true_lag_tt",)

#: Per-sample $O(n)$ provenance arrays; always eagerly loaded (scalars).
_PROVENANCE_FIELDS_V4 = (
    "sample_te_true",    # float32 -- cell TE_inj (the canonical label)
    "sample_delay",      # int16   -- fixed source->target lag D
    "sample_cell_id",    # int16   -- index into meta['cells']
    "sample_held_out",   # int8    -- 0 (no held-out concept in the v4 grid)
    "sample_raw_index",  # int32   -- within-cell row index (deterministic raw regen key)
)


class AttributeDict(dict):
    r"""A ``dict`` that also supports attribute-style access (thin local copy, cf. ``dataset_v2``)."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'AttributeDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def attribute_dict_collate(batch: list) -> AttributeDict:
    r"""Collate a list of :class:`AttributeDict` samples into one batched :class:`AttributeDict`."""
    return AttributeDict(default_collate(batch))


class SyntheticRawDatasetV4(Dataset):
    r"""Dataset over a single cached ``synthetic_v4`` raw split (``train`` / ``val`` / ``test``).

    Two backing modes (mirroring :class:`dataset_v2.SyntheticTEDatasetV2`): memory-mapped (default,
    ``mmap='auto'`` -- the large per-timestep arrays are :class:`numpy.memmap` views into the
    uncompressed ``.npz``, so the process holds no private copy and pages are shared through the OS
    page cache) or eager (``mmap=False`` or an unmappable archive).

    Attributes:
        meta: The decoded ``meta.json`` dict (analytic ground truth + per-cell manifest).
        split: Split name derived from the ``.npz`` stem.
        te_true: Pooled block transfer entropy of this cache, in nats.
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None,
        *,
        mmap: Union[bool, str] = "auto",
    ) -> None:
        r"""Open a cached raw split.

        Args:
            npz_path: Path to a ``{train,val,test}.npz`` from :func:`build_dataset_v4.build_all_v4`.
            meta_path: Path to the shared ``meta.json`` (defaults to ``meta.json`` next to
                ``npz_path``).
            mmap: Backing mode. ``'auto'`` / ``True`` memory-maps and falls back to eager when the
                archive is not mappable; ``False`` forces eager.

        Raises:
            FileNotFoundError: If ``npz_path`` or the resolved ``meta.json`` is absent.
            KeyError: If the ``.npz`` is missing one of the raw fields.
        """
        self.npz_path = Path(npz_path)
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"cache split not found: {self.npz_path}")
        self.split = self.npz_path.stem

        meta_path = (
            Path(meta_path) if meta_path is not None else self.npz_path.parent / "meta.json"
        )
        if not meta_path.is_file():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta: Dict[str, Any] = json.load(fh)

        self.te_true: float = float(self.meta.get("te_true", 0.0))
        self._tag: str = str(self.meta.get("tag", self.meta.get("benchmark", "syn")))

        self._mmap_specs: Optional[Dict[str, _MemberSpec]] = None
        self.mmap_active: bool = False
        specs: Optional[Dict[str, _MemberSpec]] = None
        if mmap is not False:
            try:
                specs = _read_npz_member_specs(self.npz_path)
            except (_MmapUnsupported, zipfile.BadZipFile, OSError) as exc:
                print(
                    f"[dataset_v4][note] {self.npz_path.name}: memory-mapping "
                    f"unavailable ({exc}); loading eagerly into RAM."
                )
                specs = None

        if specs is not None:
            missing = [f for f in _FIELDS_V4 if f not in specs]
            if missing:
                raise KeyError(f"{self.npz_path} is missing fields: {missing}")
            self._mmap_specs = {name: specs[name] for name in _MMAP_FIELDS_V4 if name in specs}
            self._reopen_memmaps()
            present = [f for f in _PROVENANCE_FIELDS_V4 if f in specs]
            if present:
                with np.load(self.npz_path) as npz:
                    self._provenance: Optional[Dict[str, np.ndarray]] = {
                        f: np.asarray(npz[f]) for f in present
                    }
            else:
                self._provenance = None
            self.mmap_active = True
        else:
            with np.load(self.npz_path) as npz:
                missing = [f for f in _FIELDS_V4 if f not in npz.files]
                if missing:
                    raise KeyError(f"{self.npz_path} is missing fields: {missing}")
                self._arrays = {f: np.asarray(npz[f], dtype=np.float32) for f in _FIELDS_V4}
                self._true_lag_tt: Optional[np.ndarray] = (
                    np.asarray(npz["true_lag_tt"]) if "true_lag_tt" in npz.files else None
                )
                present = [f for f in _PROVENANCE_FIELDS_V4 if f in npz.files]
                self._provenance = (
                    {f: np.asarray(npz[f]) for f in present} if present else None
                )
        self._n = int(self._arrays["fhr"].shape[0])

    # ------------------------------------------------------------------
    # Memory-map plumbing (spawn-safe pickling)
    # ------------------------------------------------------------------
    def _reopen_memmaps(self) -> None:
        r"""(Re)open the :class:`numpy.memmap` views from ``_mmap_specs`` (worker-safe)."""
        assert self._mmap_specs is not None
        specs = self._mmap_specs

        def _open(name: str) -> np.memmap:
            offset, dtype, shape, order = specs[name]
            return np.memmap(
                self.npz_path, dtype=dtype, mode="r", offset=offset,
                shape=shape, order=("F" if order == "F" else "C"),
            )

        self._arrays = {f: _open(f) for f in _FIELDS_V4}
        self._true_lag_tt = _open("true_lag_tt") if "true_lag_tt" in self._mmap_specs else None

    def __getstate__(self) -> Dict[str, Any]:
        r"""Drop the (unpicklable-as-views) memmaps; keep the cheap specs."""
        state = dict(self.__dict__)
        if self._mmap_specs is not None:
            state["_arrays"] = None
            state["_true_lag_tt"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        r"""Restore state and re-map the array views in the new process."""
        self.__dict__.update(state)
        if self._mmap_specs is not None:
            self._reopen_memmaps()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> AttributeDict:
        r"""Return sample ``idx`` as an :class:`AttributeDict`.

        Returns:
            An :class:`AttributeDict` with the four raw tensor fields (``fhr`` $(5280,)$, ``up``
            $(5280,)$, ``weight`` $(330,)$, ``target`` $(330,)$), the per-step ground-truth lag
            ``true_lag_tt`` $(330,)$ when present, and the per-sample metadata: ``te_true``
            (this sample's cell $\mathrm{TE}_{\mathrm{inj}}$), ``delay`` (the fixed lag $D$),
            ``cell_id``, ``held_out``, ``raw_index`` and ``guid`` (str).
        """
        sample = AttributeDict()
        for field in _FIELDS_V4:
            arr = np.array(self._arrays[field][idx], dtype=np.float32)
            sample[field] = torch.from_numpy(arr)
        sample["te_true"] = self.te_true
        if self._true_lag_tt is not None:
            sample["true_lag_tt"] = torch.from_numpy(
                self._true_lag_tt[idx].astype(np.int64)
            )  # (T_tilde,) long
        if self._provenance is not None:
            prov = self._provenance
            sample["te_true"] = float(prov["sample_te_true"][idx])
            sample["delay"] = int(prov["sample_delay"][idx])
            sample["cell_id"] = int(prov["sample_cell_id"][idx])
            sample["held_out"] = int(prov["sample_held_out"][idx])
            if "sample_raw_index" in prov:
                sample["raw_index"] = int(prov["sample_raw_index"][idx])
        sample["guid"] = f"{self._tag}_{self.split}_{idx:06d}"
        return sample


def make_dataloader_v4(
    dataset: SyntheticRawDatasetV4,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    r"""Build a ``DataLoader`` wired with :func:`attribute_dict_collate` (cf. ``make_dataloader``).

    Args:
        dataset: A :class:`SyntheticRawDatasetV4`.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle every epoch (``True`` for training).
        num_workers: Worker processes (default ``0``).
        pin_memory: Request page-locked host memory (CUDA only).
        persistent_workers: Keep workers alive across epochs (coerced off when ``num_workers == 0``).
        drop_last: Whether to drop a trailing partial batch.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    persistent = bool(persistent_workers) and int(num_workers) > 0
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent,
        collate_fn=attribute_dict_collate,
    )
