r"""``SyntheticTEDataset`` -- cached-dataset loader for the TE benchmarks.

Per Decisions D3 and D7, this dataset **loads a pre-generated, on-disk dataset**
(written by :mod:`build_dataset`) rather than generating samples on the fly.
Each ``__getitem__`` returns an :class:`AttributeDict` with the model's native
field names so the batch is drop-in compatible with both ``train_minimal`` and
the existing ``testing/base.py`` ``TestRunner``:

    * ``fhr_st`` $(T, 43)$, ``fhr_ph`` $(T, 44)$ -- target $Y$ split.
    * ``up_st``  $(T, 43)$, ``up_ph`` $(T, 58)$  -- source $U$ split.
    * ``weight`` $(T,)$ -- all-ones (synthetic data has no gaps).
    * ``te_true``, ``true_lag_band``, ``guid`` -- per-sample metadata.

The cache layout (see :mod:`build_dataset`) is
``<data_dir>/<benchmark>/<tag>/{train,val,test}.npz`` with one shared
``meta.json`` describing the analytic ground truth.

Public API:
    AttributeDict: attribute-accessible ``dict`` (thin local copy).
    attribute_dict_collate: collate a list of ``AttributeDict``s into one.
    SyntheticTEDataset: ``torch.utils.data.Dataset`` over a cached ``.npz``.
    build_u_stream: ``cat(up_st, up_ph)`` -> $(B, T, 101)$.
    make_dataloader: ``DataLoader`` wired with :func:`attribute_dict_collate`.

Note:
    :class:`AttributeDict` and :func:`attribute_dict_collate` are **thin local
    copies** of the originals in ``hdf5_dataset/hdf5_dataset.py`` (lines 384 and
    973). Importing that module would drag in ``h5py`` and the whole HDF5
    pipeline; copying the ~8 trivial lines keeps the synthetic package
    standalone (Decision D2). Task 2.2 explicitly permits "a thin copy".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

# Tensor fields stored in every split ``.npz`` (native model channel layout).
_FIELDS = ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight")


class AttributeDict(dict):
    """A ``dict`` that also supports attribute-style access.

    Thin local copy of ``hdf5_dataset.hdf5_dataset.AttributeDict`` so the
    synthetic package does not import the HDF5 stack. ``batch.fhr_st`` is
    equivalent to ``batch["fhr_st"]``.
    """

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(
            f"'AttributeDict' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def attribute_dict_collate(batch: list) -> AttributeDict:
    """Collate a list of :class:`AttributeDict` samples into a batched one.

    Thin local copy of ``hdf5_dataset.hdf5_dataset.attribute_dict_collate``.
    Delegates to :func:`torch.utils.data._utils.collate.default_collate` (which
    stacks tensors along a new batch axis and leaves strings as a list) and
    re-wraps the result so attribute access survives.

    Args:
        batch: List of per-sample :class:`AttributeDict` items.

    Returns:
        A single :class:`AttributeDict` of batched tensors / lists.
    """
    return AttributeDict(default_collate(batch))


class SyntheticTEDataset(Dataset):
    r"""Dataset over a single cached synthetic-TE split (``train``/``val``/``test``).

    The dataset is a pure loader: it reads one ``.npz`` written by
    :mod:`build_dataset` fully into RAM, closes the archive handle, and exposes
    each sample as an :class:`AttributeDict`. Closing the handle keeps the
    cache file unlocked (safe for temp-dir cleanup on Windows) and per-sample
    access needs no disk round trip. The analytic ground truth -- block
    transfer entropy ``te_true`` and the ``true_lag_band`` $\{D-H,\dots,D-1\}$
    -- lives in the shared ``meta.json`` and is attached to every sample so
    downstream evaluators (``evaluate_te``, ``lag_recovery``) can reach it from
    any batch.

    Attributes:
        meta: The decoded ``meta.json`` dict (analytic ground truth + config).
        split: Split name derived from the ``.npz`` stem (``train``/``val``/...).
        te_true: Block transfer entropy of this benchmark, in nats.
        true_lag_band: List of source lags carrying the transfer.
        channel_decomp: Resolved structured channel decomposition
            (``m``, ``n_self``, ``n_smallnoise``, ``m_source``, ``n_dist``,
            ``n_noise`` plus the AR(1) / oscillator ranges) or ``None`` if
            the cache pre-dates the v2 decomposition pipeline.
        channel_layout: Per-block absolute channel index lists
            ``{"Y": {"te", "self", "smallnoise"}, "U": {"te", "dist",
            "noise"}}`` or ``None`` for legacy caches. Downstream evaluators
            read this to colour-code or mask sub-blocks.
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Open a cached split.

        Args:
            npz_path: Path to a ``{train,val,test}.npz`` produced by
                :mod:`build_dataset`.
            meta_path: Path to the shared ``meta.json``. Defaults to
                ``meta.json`` next to ``npz_path``.

        Raises:
            FileNotFoundError: If ``npz_path`` or the resolved ``meta.json``
                does not exist.
            KeyError: If the ``.npz`` is missing one of the native fields.
        """
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
        # v2 structured channel decomposition. Legacy caches written before
        # the decomposition pipeline existed leave these as ``None``.
        self.channel_decomp: Optional[Dict[str, Any]] = self.meta.get(
            "channel_decomp"
        )
        self.channel_layout: Optional[Dict[str, Dict[str, list]]] = self.meta.get(
            "channel_layout"
        )

        # Read the split fully into RAM and close the archive handle inside
        # the ``with`` block, so the cache file is never left open.
        with np.load(self.npz_path) as npz:
            missing = [f for f in _FIELDS if f not in npz.files]
            if missing:
                raise KeyError(f"{self.npz_path} is missing fields: {missing}")
            self._arrays = {
                f: np.asarray(npz[f], dtype=np.float32) for f in _FIELDS
            }
            # Optional per-sample, per-step ground-truth lag $d_{i,t}$ of shape
            # $(n, T)$, written by ``build_dataset`` for the synthetic
            # benchmarks. Absent from legacy caches and from real (HDF5) data,
            # in which case the lag-attention overlay is simply skipped.
            self._true_lag_tt: Optional[np.ndarray] = (
                np.asarray(npz["true_lag_tt"]) if "true_lag_tt" in npz.files
                else None
            )
        self._n = int(self._arrays["fhr_st"].shape[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> AttributeDict:
        """Return sample ``idx`` as an :class:`AttributeDict`.

        Args:
            idx: Sample index in ``[0, len(self))``.

        Returns:
            An :class:`AttributeDict` with the five native tensor fields
            (``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``, ``weight``) plus the
            per-sample metadata ``te_true`` (float), ``true_lag_band``
            (``long`` tensor) and ``guid`` (str). When the cache carries the
            per-step ground-truth lag, ``true_lag_tt`` (``long`` tensor of shape
            $(T,)$) is also attached for the lag-attention overlay.
        """
        sample = AttributeDict()
        for field in _FIELDS:
            # np.array(...) forces a writable, contiguous float32 copy that the
            # collate stack then owns independently of the in-RAM cache.
            arr = np.array(self._arrays[field][idx], dtype=np.float32)
            sample[field] = torch.from_numpy(arr)
        sample["te_true"] = self.te_true
        sample["true_lag_band"] = self._lag_band_tensor
        if self._true_lag_tt is not None:
            sample["true_lag_tt"] = torch.from_numpy(
                self._true_lag_tt[idx].astype(np.int64)
            )                                                    # (T,) long
        sample["guid"] = f"{self._tag}_{self.split}_{idx:06d}"
        return sample


def build_u_stream(batch: Any) -> torch.Tensor:
    r"""Concatenate the source streams into the model's ``u_stream`` input.

    Mirrors ``SeqVaeLagAttnPl._build_source_stream`` exactly: ``up_st`` (43
    channels) followed by ``up_ph`` (58 channels) along the channel axis, so
    $U \in \mathbb{R}^{B \times T \times 101}$. This is the ``use_up_st=True``
    layout; the synthetic benchmarks always use the full 101-channel source.

    Args:
        batch: An :class:`AttributeDict` (batched) with ``up_st`` and ``up_ph``.

    Returns:
        The source stream tensor of shape $(B, T, 101)$.
    """
    return torch.cat([batch.up_st, batch.up_ph], dim=-1)


def make_dataloader(
    dataset: SyntheticTEDataset,
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
        dataset: A :class:`SyntheticTEDataset`.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle every epoch (``True`` for training).
        num_workers: Worker processes. Defaults to ``0``; the split is already
            held fully in RAM (see :class:`SyntheticTEDataset`), so workers
            would only add fork / IPC overhead on a single-GPU laptop. On
            a multi-GPU box ``num_workers >= 2`` lets host->device copies
            overlap with the forward pass; set via ``dataset.num_workers``
            in ``config_synth.yaml``.
        pin_memory: If True, request page-locked host memory for the input
            batch. Only useful when CUDA is in use; off by default for the
            same single-laptop reason. Set via ``dataset.pin_memory``.
        persistent_workers: If True, keep DataLoader worker processes alive
            across epochs. Only honoured when ``num_workers > 0``; otherwise
            forced off by PyTorch. Set via ``dataset.persistent_workers``.
        drop_last: Whether to drop a trailing partial batch.

    Returns:
        A configured :class:`torch.utils.data.DataLoader`.
    """
    # ``persistent_workers=True`` with ``num_workers=0`` raises in PyTorch;
    # silently coerce so the YAML default never triggers a hard crash.
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


if __name__ == "__main__":
    # Self-check: generate a tiny cache in a temp dir, load it, collate a batch.
    import tempfile

    from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
        gen_smooth_arx,
    )

    _T, _DELAY = 300, 60
    _Y, _U, _meta = gen_smooth_arx(
        n=6, T=_T, rho_u=0.99, rho_y=0.95, c=0.5,
        sigma2_eta=1.0, sigma2_eps=1.0, delay=_DELAY, M=4, seed=0,
    )
    with tempfile.TemporaryDirectory() as _tmp:
        _dir = Path(_tmp)
        # The generator stamps the per-step true lag into meta as a numpy
        # array; ``build_dataset`` pops it into the ``.npz``. Mirror that here
        # so the self-check exercises the ``true_lag_tt`` round-trip and keeps
        # ``meta.json`` JSON-serialisable.
        _true_lag_tt = _meta.pop("true_lag_tt", None)
        _extra: Dict[str, Any] = {}
        if _true_lag_tt is not None:
            _extra["true_lag_tt"] = np.asarray(_true_lag_tt, dtype=np.int16)
        np.savez(
            _dir / "train.npz",
            fhr_st=_Y[..., :43].numpy(),
            fhr_ph=_Y[..., 43:87].numpy(),
            up_st=_U[..., :43].numpy(),
            up_ph=_U[..., 43:101].numpy(),
            weight=np.ones((6, _T), dtype=np.float32),
            **_extra,
        )
        _meta_out = dict(_meta)
        _meta_out["tag"] = "smoke"
        with open(_dir / "meta.json", "w", encoding="utf-8") as _fh:
            json.dump(_meta_out, _fh, indent=2)

        _ds = SyntheticTEDataset(_dir / "train.npz")
        assert len(_ds) == 6, len(_ds)
        _s = _ds[0]
        assert _s.fhr_st.shape == (_T, 43), _s.fhr_st.shape
        assert _s.fhr_ph.shape == (_T, 44), _s.fhr_ph.shape
        assert _s.up_st.shape == (_T, 43), _s.up_st.shape
        assert _s.up_ph.shape == (_T, 58), _s.up_ph.shape
        assert _s.weight.shape == (_T,), _s.weight.shape
        if _true_lag_tt is not None:
            assert _s.true_lag_tt.shape == (_T,), _s.true_lag_tt.shape
        print(f"[sample] guid={_s.guid}  te_true={_s.te_true:.4f}")

        _loader = make_dataloader(_ds, batch_size=3, shuffle=False)
        _batch = next(iter(_loader))
        assert _batch.fhr_st.shape == (3, _T, 43), _batch.fhr_st.shape
        _u = build_u_stream(_batch)
        assert _u.shape == (3, _T, 101), _u.shape
        assert len(_batch.guid) == 3 and isinstance(_batch.guid[0], str)
        print(f"[batch]  u_stream={tuple(_u.shape)}  guids={list(_batch.guid)}")

    print("All dataset checks passed.")
