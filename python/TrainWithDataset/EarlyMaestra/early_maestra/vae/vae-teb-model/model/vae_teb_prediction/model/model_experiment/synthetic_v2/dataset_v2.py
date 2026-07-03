r"""``SyntheticTEDatasetV2`` -- cached-dataset loader for the ``synthetic_v2`` cache.

Copied/adapted from ``synthetic/dataset.py`` (Decision D2: v2 owns the code, so the
loader is local rather than importing the v1 package). It is a pure loader over one
``.npz`` written by :func:`build_dataset_v2.build_all`, exposing each sample as an
:class:`AttributeDict` with the model's native field names so a batch is drop-in for
``vae_teb_lag_attn_v1.SeqVaeLagAttnV1.forward`` and the standard ``testing`` surface:

    * ``fhr_st`` $(T, 43)$, ``fhr_ph`` $(T, 44)$ -- target $Y$ split.
    * ``up_st``  $(T, 43)$, ``up_ph`` $(T, 58)$  -- source $U$ split.
    * ``weight`` $(T,)$ -- all-ones (synthetic data has no gaps).
    * ``te_true``, ``te_scat``, ``frac_phi``, ``delay``, ``cell_id``, ``held_out``,
      ``true_lag_band``, ``true_lag_tt``, ``guid`` -- per-sample metadata.

The v2 provenance differs from the v1 mixed cache: there is **no** ``sample_M`` /
``sample_delay_min`` / ``sample_delay_max`` / ``sample_band_id`` (single pathway, fixed
lag), and v2 adds ``sample_te_scat`` / ``sample_frac_phi``. The clean-window floor
$\max(w, D - 1)$ reads the fixed lag from ``delay`` (§14; S6 wires the eval).

The cache layout (see :mod:`build_dataset_v2`) is ``<data_dir>/<benchmark>/<tag>/
{train,val,test}.npz`` with one shared ``meta.json`` describing the analytic ground
truth and the per-cell manifest.

Public API:
    AttributeDict: attribute-accessible ``dict`` (thin local copy).
    attribute_dict_collate: collate a list of ``AttributeDict``s into one.
    SyntheticTEDatasetV2: ``torch.utils.data.Dataset`` over a cached v2 ``.npz``.
    build_u_stream: ``cat(up_st, up_ph)`` -> $(B, T, 101)$.
    make_dataloader: ``DataLoader`` wired with :func:`attribute_dict_collate`.

Note:
    :class:`AttributeDict` and :func:`attribute_dict_collate` are **thin local copies**
    of the originals in ``hdf5_dataset/hdf5_dataset.py`` (lines 384 and 973); importing
    that module drags in ``h5py`` and the whole HDF5 pipeline, so the ~8 trivial lines
    are copied to keep the synthetic_v2 package standalone.
"""

from __future__ import annotations

import json
import logging
import struct
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from numpy.lib import format as _npy_format
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)

# Tensor fields stored in every split ``.npz`` (native model channel layout).
_FIELDS = ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight")

# Large per-timestep arrays worth memory-mapping ($O(n \cdot T \cdot C)$ bytes).
# The per-sample provenance arrays are $O(n)$ scalars and stay eagerly loaded.
_MMAP_FIELDS = _FIELDS + ("true_lag_tt",)

# One member spec of an uncompressed ``.npz``: (absolute byte offset of the array
# data inside the archive file, dtype, shape, 'C'/'F' memory order).
_MemberSpec = Tuple[int, np.dtype, Tuple[int, ...], str]

# Per-sample provenance arrays written by :func:`build_dataset_v2.build_all`. Each is
# length ``n`` and row-aligned to ``fhr_st``. v2 drops the v1 ``sample_M`` /
# ``sample_delay_min`` / ``sample_delay_max`` / ``sample_band_id`` (single pathway,
# fixed lag) and adds ``sample_te_scat`` / ``sample_frac_phi``.
_PROVENANCE_FIELDS = (
    "sample_te_true",    # float32 -- cell TE_inj (the canonical label)
    "sample_te_scat",    # float32 -- measured TE_scat at build N (v2-only)
    "sample_frac_phi",   # float32 -- frac_Phi = TE_scat / TE_inj at build N (v2-only)
    "sample_te_raw",     # float32 -- OPTIONAL raw-domain TE (only when the build stamps it)
    "sample_delay",      # int16   -- fixed source->target lag D
    "sample_cell_id",    # int16   -- index into the meta['cells'] manifest
    "sample_held_out",   # int8    -- 0 in-mix, 1 held-out cache
    "sample_raw_index",  # int32   -- within-cell row index (deterministic raw regen key)
)


class AttributeDict(dict):
    """A ``dict`` that also supports attribute-style access.

    Thin local copy of ``hdf5_dataset.hdf5_dataset.AttributeDict`` so the synthetic_v2
    package does not import the HDF5 stack. ``batch.fhr_st`` is equivalent to
    ``batch["fhr_st"]``.
    """

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'AttributeDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def attribute_dict_collate(batch: list) -> AttributeDict:
    """Collate a list of :class:`AttributeDict` samples into a batched one.

    Delegates to :func:`torch.utils.data._utils.collate.default_collate` (which stacks
    tensors along a new batch axis and leaves strings / Python scalars as lists) and
    re-wraps the result so attribute access survives.

    Args:
        batch: List of per-sample :class:`AttributeDict` items.

    Returns:
        A single :class:`AttributeDict` of batched tensors / lists.
    """
    return AttributeDict(default_collate(batch))


class _MmapUnsupported(Exception):
    """Raised when a cache ``.npz`` cannot be memory-mapped (eager fallback)."""


def _read_npz_member_specs(npz_path: Path) -> Dict[str, _MemberSpec]:
    r"""Locate every array of an **uncompressed** ``.npz`` inside the archive.

    ``np.load(..., mmap_mode=...)`` ignores ``mmap_mode`` for ``.npz`` archives, but the
    v2 caches use ``np.savez`` (ZIP **stored**, not deflated), so each ``.npy`` member
    sits as a contiguous byte range inside the archive file. This resolves, per member,
    the absolute byte offset of the raw array data (local ZIP header -> ``.npy`` header
    -> data start) plus its dtype / shape / order, so the caller can hand each one to
    :class:`numpy.memmap` directly -- no decompression, no copy, and the OS page cache
    shares the bytes across every process that maps the same file (the DDP multi-rank
    RAM fix).

    Args:
        npz_path: Path to the ``.npz`` archive.

    Returns:
        Mapping ``{array_name: (data_offset, dtype, shape, order)}`` with the trailing
        ``.npy`` stripped from each member name.

    Raises:
        _MmapUnsupported: If any member is compressed, uses an unsupported ``.npy``
            format version, holds Python objects, or is empty -- callers fall back to
            the eager in-RAM loader.
    """
    specs: Dict[str, _MemberSpec] = {}
    with zipfile.ZipFile(npz_path) as zf, open(npz_path, "rb") as fh:
        for info in zf.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise _MmapUnsupported(f"member {info.filename!r} is compressed")
            fh.seek(info.header_offset)
            header = fh.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise _MmapUnsupported(
                    f"member {info.filename!r}: bad local file header"
                )
            n_name, n_extra = struct.unpack("<HH", header[26:30])
            fh.seek(info.header_offset + 30 + n_name + n_extra)
            try:
                version = _npy_format.read_magic(fh)
                if version == (1, 0):
                    shape, fortran, dtype = _npy_format.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, fortran, dtype = _npy_format.read_array_header_2_0(fh)
                else:
                    raise _MmapUnsupported(
                        f"member {info.filename!r}: npy format {version}"
                    )
            except ValueError as exc:
                raise _MmapUnsupported(f"member {info.filename!r}: {exc}") from exc
            if dtype.hasobject:
                raise _MmapUnsupported(f"member {info.filename!r}: object dtype")
            if int(np.prod(shape)) == 0:
                raise _MmapUnsupported(f"member {info.filename!r}: empty array")
            name = info.filename
            if name.endswith(".npy"):
                name = name[:-4]
            specs[name] = (fh.tell(), dtype, tuple(shape), "F" if fortran else "C")
    return specs


class SyntheticTEDatasetV2(Dataset):
    r"""Dataset over a single cached ``synthetic_v2`` split (``train``/``val``/``test``).

    A pure loader over one ``.npz`` written by :func:`build_dataset_v2.build_all`,
    exposing each sample as an :class:`AttributeDict`. Two backing modes exist:

    * **Memory-mapped** (default, ``mmap='auto'``): the large per-timestep arrays are
      :class:`numpy.memmap` views straight into the uncompressed ``.npz``, so the
      process holds **no private copy** of the split -- pages are demand-loaded and
      shared through the OS page cache (the DDP multi-rank RAM fix). ``__getitem__``
      copies each row into a fresh writable ``float32`` tensor.
    * **Eager** (``mmap=False`` or automatic fallback for a compressed / unmappable
      archive): read everything into RAM as ``float32`` and close the handle.

    Attributes:
        meta: The decoded ``meta.json`` dict (analytic ground truth + per-cell manifest).
        split: Split name derived from the ``.npz`` stem.
        te_true: Pooled block transfer entropy of this cache, in nats.
        true_lag_band: List of source lags carrying the transfer (the pooled union).
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None,
        *,
        mmap: Union[bool, str] = "auto",
        raw_provider: Optional[Callable[[int, int], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        r"""Open a cached split.

        Args:
            npz_path: Path to a ``{train,val,test}.npz`` produced by
                :func:`build_dataset_v2.build_all`.
            meta_path: Path to the shared ``meta.json``. Defaults to ``meta.json`` next
                to ``npz_path``.
            mmap: Backing mode for the large per-timestep arrays. ``'auto'`` (default)
                or ``True`` memory-maps them out of the uncompressed ``.npz`` and
                silently falls back to the eager loader when the archive is not
                mappable. ``False`` forces the eager loader. The realised mode is
                exposed as :attr:`mmap_active`.
            raw_provider: Optional ``(cell_id, raw_index) -> (fhr_win, up_win)`` callable
                (see :func:`build_dataset_v2.make_raw_provider`). When supplied,
                :meth:`__getitem__` deterministically regenerates each row's raw $4\,\mathrm{Hz}$
                FHR/UP waveform and exposes it as ``fhr`` / ``up`` so the per-sample diagnostic
                figure's first panel can draw it. Default ``None`` keeps the loader raw-free (no
                regeneration cost) for training / evaluation. Requires the ``sample_raw_index``
                provenance field in the cache.

        Raises:
            FileNotFoundError: If ``npz_path`` or the resolved ``meta.json`` is absent.
            KeyError: If the ``.npz`` is missing one of the native fields.
        """
        self._raw_provider = raw_provider
        self.npz_path = Path(npz_path)
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"cache split not found: {self.npz_path}")
        self.split = self.npz_path.stem

        meta_path = (
            Path(meta_path)
            if meta_path is not None
            else self.npz_path.parent / "meta.json"
        )
        if not meta_path.is_file():
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta: Dict[str, Any] = json.load(fh)

        self.te_true: float = float(self.meta["te_true"])
        self.true_lag_band = list(self.meta["true_lag_band"])
        self._tag: str = str(self.meta.get("tag", self.meta.get("benchmark", "syn")))
        self._lag_band_tensor = torch.tensor(self.true_lag_band, dtype=torch.long)

        # --- backing arrays: memory-mapped (default) or eager in-RAM ---------
        self._mmap_specs: Optional[Dict[str, _MemberSpec]] = None
        self.mmap_active: bool = False
        specs: Optional[Dict[str, _MemberSpec]] = None
        if mmap is not False:
            try:
                specs = _read_npz_member_specs(self.npz_path)
            except (_MmapUnsupported, zipfile.BadZipFile, OSError) as exc:
                print(
                    f"[dataset_v2][note] {self.npz_path.name}: memory-mapping "
                    f"unavailable ({exc}); loading eagerly into RAM."
                )
                specs = None

        if specs is not None:
            missing = [f for f in _FIELDS if f not in specs]
            if missing:
                raise KeyError(f"{self.npz_path} is missing fields: {missing}")
            self._mmap_specs = {
                name: specs[name] for name in _MMAP_FIELDS if name in specs
            }
            self._reopen_memmaps()
            present = [f for f in _PROVENANCE_FIELDS if f in specs]
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
                missing = [f for f in _FIELDS if f not in npz.files]
                if missing:
                    raise KeyError(f"{self.npz_path} is missing fields: {missing}")
                self._arrays = {
                    f: np.asarray(npz[f], dtype=np.float32) for f in _FIELDS
                }
                self._true_lag_tt: Optional[np.ndarray] = (
                    np.asarray(npz["true_lag_tt"])
                    if "true_lag_tt" in npz.files else None
                )
                present = [f for f in _PROVENANCE_FIELDS if f in npz.files]
                self._provenance = (
                    {f: np.asarray(npz[f]) for f in present} if present else None
                )
        self._n = int(self._arrays["fhr_st"].shape[0])

    # ------------------------------------------------------------------
    # Memory-map plumbing (spawn-safe pickling)
    # ------------------------------------------------------------------
    def _reopen_memmaps(self) -> None:
        """(Re)open the :class:`numpy.memmap` views from ``_mmap_specs``.

        Called from ``__init__`` and again from :meth:`__setstate__` after unpickling in
        a DataLoader worker -- every process maps the same file regions, so the OS page
        cache backs them all with one physical copy.
        """
        assert self._mmap_specs is not None
        specs = self._mmap_specs

        def _open(name: str) -> np.memmap:
            offset, dtype, shape, order = specs[name]
            return np.memmap(
                self.npz_path, dtype=dtype, mode="r", offset=offset,
                shape=shape, order=("F" if order == "F" else "C"),
            )

        self._arrays = {f: _open(f) for f in _FIELDS}
        self._true_lag_tt = (
            _open("true_lag_tt") if "true_lag_tt" in self._mmap_specs else None
        )

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the (unpicklable-as-views) memmaps; keep the cheap specs."""
        state = dict(self.__dict__)
        if self._mmap_specs is not None:
            state["_arrays"] = None
            state["_true_lag_tt"] = None
        # The raw provider is a closure (not picklable) and is only used on the
        # single-process plotting path; workers get a raw-free dataset.
        state["_raw_provider"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and re-map the array views in the new process."""
        self.__dict__.update(state)
        if self._mmap_specs is not None:
            self._reopen_memmaps()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> AttributeDict:
        r"""Return sample ``idx`` as an :class:`AttributeDict`.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            An :class:`AttributeDict` with the five native tensor fields (``fhr_st``,
            ``fhr_ph``, ``up_st``, ``up_ph``, ``weight``), the per-step ground-truth lag
            ``true_lag_tt`` $(T,)$ when present, and the per-sample metadata: ``te_true``
            (this sample's cell $\mathrm{TE}_{\mathrm{inj}}$), ``te_scat``, ``frac_phi``,
            ``delay`` (the fixed lag $D$), ``cell_id``, ``held_out``, ``raw_index``,
            ``true_lag_band`` (``long`` tensor) and ``guid`` (str). When a ``raw_provider`` is
            attached, also ``fhr`` / ``up`` (raw $4\,\mathrm{Hz}$ waveforms, physical units,
            $4800$ samples). No ``M`` / ``delay_min`` / ``delay_max`` / ``band_id`` keys (v2 is
            single-pathway, fixed lag).
        """
        sample = AttributeDict()
        for field in _FIELDS:
            arr = np.array(self._arrays[field][idx], dtype=np.float32)
            sample[field] = torch.from_numpy(arr)
        sample["te_true"] = self.te_true
        sample["true_lag_band"] = self._lag_band_tensor
        if self._true_lag_tt is not None:
            sample["true_lag_tt"] = torch.from_numpy(
                self._true_lag_tt[idx].astype(np.int64)
            )  # (T,) long
        if self._provenance is not None:
            # Override the dataset-level ``te_true`` with this sample's own cell TE and
            # attach the v2 per-sample scalars. Only plain Python scalars are added (no
            # per-sample band tensor) so batches mixing different ``delay`` collate
            # cleanly to ``(B,)``; the eval rebuilds each cell's band from ``delay``.
            prov = self._provenance
            sample["te_true"] = float(prov["sample_te_true"][idx])
            sample["te_scat"] = float(prov["sample_te_scat"][idx])
            sample["frac_phi"] = float(prov["sample_frac_phi"][idx])
            sample["delay"] = int(prov["sample_delay"][idx])
            sample["cell_id"] = int(prov["sample_cell_id"][idx])
            sample["held_out"] = int(prov["sample_held_out"][idx])
            # ``sample_te_raw`` is optional (only present when the build stamps it);
            # exposed guarded so the standard-testing TE bridge (S7-T06) can pick it up.
            if "sample_te_raw" in prov:
                sample["te_raw"] = float(prov["sample_te_raw"][idx])
            if "sample_raw_index" in prov:
                sample["raw_index"] = int(prov["sample_raw_index"][idx])
            # Deterministically regenerate this row's raw 4 Hz FHR/UP (physical units) when a
            # provider is attached, so the per-sample diagnostic's first panel can draw it.
            # ``collect_predictions`` reads ``batch.fhr`` / ``batch.up`` (a no-op denormalise
            # when the loader carries no fhr/up stats), so nothing else needs changing. The
            # keys are set for EVERY item (the provider is total: real window or a NaN window),
            # so a batch never collates with inconsistent keys -- a partial raw failure would
            # otherwise crash ``default_collate`` and abort the whole diagnostics stage.
            if self._raw_provider is not None and "sample_raw_index" in prov:
                win_len = int(getattr(self._raw_provider, "window_length", 0))
                try:
                    fhr_win, up_win = self._raw_provider(
                        int(prov["sample_cell_id"][idx]),
                        int(prov["sample_raw_index"][idx]),
                    )
                except Exception as exc:  # noqa: BLE001 -- raw is a plotting nicety, never fatal
                    logger.warning("dataset_v2: raw regeneration failed for idx %d (%s)",
                                   idx, exc)
                    fhr_win = up_win = (np.full(win_len, np.nan, np.float32)
                                        if win_len else None)
                if fhr_win is not None:
                    sample["fhr"] = torch.from_numpy(np.asarray(fhr_win, dtype=np.float32))
                    sample["up"] = torch.from_numpy(np.asarray(up_win, dtype=np.float32))
        sample["guid"] = f"{self._tag}_{self.split}_{idx:06d}"
        return sample


def build_u_stream(batch: Any) -> torch.Tensor:
    r"""Concatenate the source streams into the model's ``u_stream`` input.

    Mirrors ``SeqVaeLagAttnV1``'s ``use_up_st=True`` layout: ``up_st`` (43 channels)
    followed by ``up_ph`` (58 channels) along the channel axis, so
    $U \in \mathbb{R}^{B \times T \times 101}$.

    Args:
        batch: An :class:`AttributeDict` (batched) with ``up_st`` and ``up_ph``.

    Returns:
        The source stream tensor of shape $(B, T, 101)$.
    """
    return torch.cat([batch.up_st, batch.up_ph], dim=-1)


def make_dataloader(
    dataset: SyntheticTEDatasetV2,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a ``DataLoader`` wired with :func:`attribute_dict_collate`.

    Args:
        dataset: A :class:`SyntheticTEDatasetV2`.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle every epoch (``True`` for training).
        num_workers: Worker processes (default ``0``; the split is served from the
            shared page cache or RAM).
        pin_memory: Request page-locked host memory for the input batch (CUDA only).
        persistent_workers: Keep workers alive across epochs (only when
            ``num_workers > 0``; coerced off otherwise so the YAML default never
            crashes PyTorch).
        drop_last: Whether to drop a trailing partial batch.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    persistent = bool(persistent_workers) and int(num_workers) > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=persistent,
        drop_last=drop_last,
        collate_fn=attribute_dict_collate,
    )
