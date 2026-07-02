r"""Scattering + phase-harmonic transform and normalisation for ``synthetic_v2`` (Sprint 2).

This module is the deterministic encoder bridge (C)+(D)+(E) of the pipeline: it turns a
raw $4\,\mathrm{Hz}$ FHR/UP pair (from :func:`raw_generators.generate_cell_raw`) into the
model's four normalised feature fields

    ``fhr_st`` $(43)$ / ``fhr_ph`` $(44)$ / ``up_st`` $(43)$ / ``up_ph`` $(58)$

by calling the **production** :class:`KymatioPhaseScattering1D` transform unchanged and
applying a parity-tested local copy of the production normalisation (log / asinh +
per-channel z-score, §12). Only ``hdf5_dataset/kymatio_phase_scattering.py`` is imported;
``create_new_pipeline.py`` (which pulls ``early_maestra``) is never touched.

Design decisions realised here (see ``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` §11–§12
and ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 2):

* **Two forward passes** — one per signal. A single
  ``forward(x, compute_phase=True, scattering_channel=0, phase_channels=[0])`` on a
  single-channel raw waveform returns **both** ``'scattering'`` and ``'phase_corr'`` for
  that channel, so the four model fields need only two passes (no cross pass for training).
* **fs-correct coupled-channel identification** — the ``center_freqs`` buffer holds
  *normalised* $\xi$; the physical carrier channel is
  ``1 + argmin(|center_freqs * fs - f_pulse|)`` (the ``1 +`` skips the order-0 baseline
  channel), asserted to be within one $Q$-step of $f_{\mathrm{pulse}}$.
* **Axis convention** — the transform is channels-first $(n, C, 330)$; a symmetric
  $15$-step/end trim gives $(n, C, 300)$ (aligned to the decimated latent ``[15:315]``);
  normalisation runs channels-first (matching the production per-channel broadcast); the
  end-to-end :meth:`ScatteringAdapter.transform_and_normalise` transposes to the
  model-/cache-facing $(n, 300, C)$.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Make the repo root importable so the production transform resolves whether this module
# is imported by its package path or run from a driver. The repo root is six levels up:
# synthetic_v2 -> model_experiment -> model -> vae_teb_prediction -> model -> <repo root>.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hdf5_dataset.kymatio_phase_scattering import (  # noqa: E402  (after sys.path bootstrap)
    KymatioPhaseScattering1D,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Symmetric edge trim per end, in decimated steps ($=1\,\mathrm{min}$ at $0.25\,\mathrm{Hz}$).
#: Maps the transform's $330$ steps to the $T = 300$ analysis window; the surviving window
#: is the decimated latent slice ``[15:315]`` (§3).
TRIM_STEPS: int = 15

#: Production channel counts at ``shape = 5280`` (verified live; asserted when
#: ``strict_counts`` is on). ``up_st`` shares the scattering count with ``fhr_st``.
FHR_ST_CHANNELS: int = 43
FHR_PH_CHANNELS: int = 44
UP_ST_CHANNELS: int = 43
UP_PH_CHANNELS: int = 58

#: Production ``n_raw`` at which the four counts equal the ``43/44/43/58`` contract; a
#: :class:`ScatteringAdapter` built at this length defaults to ``strict_counts=True``.
PRODUCTION_N_RAW: int = 5280

#: Normalisation epsilons, matching the production ``normalize_tensor_data`` (§12).
LOG_EPSILON: float = 1e-6
Z_EPSILON: float = 1e-8

#: Which fields get which pointwise transform before the per-channel z-score. Mirrors the
#: production ``log_norm_channels_config`` (``all_except_0`` on ``*_st``) and
#: ``asinh_norm_channels_config`` (``all`` on ``*_ph``).
_LOG_FIELDS: Tuple[str, ...] = ("fhr_st", "up_st")
_ASINH_FIELDS: Tuple[str, ...] = ("fhr_ph", "up_ph")
_FIELD_NAMES: Tuple[str, ...] = ("fhr_st", "fhr_ph", "up_st", "up_ph")


# ---------------------------------------------------------------------------
# Normalisation (local, parity-tested copy of hdf5_dataset.normalize_tensor_data)
# ---------------------------------------------------------------------------


def _channel_axis(ndim: int) -> int:
    r"""Return the channel axis for a channels-first feature array.

    Args:
        ndim: Number of array dimensions (``2`` for $(C, T)$, ``3`` for $(n, C, T)$).

    Returns:
        The axis index of the channel dimension.
    """
    if ndim == 3:
        return 1
    if ndim == 2:
        return 0
    raise ValueError(f"expected a 2-D (C, T) or 3-D (n, C, T) array, got ndim={ndim}")


def _transform_field(arr: np.ndarray, field: str, log_epsilon: float) -> np.ndarray:
    r"""Apply the field's pointwise transform (log on ``*_st`` ch 1.., asinh on ``*_ph``).

    Order-0 (channel $0$) of the scattering fields is left untouched (it carries the
    physiological DC / baseline, §12); the log clamp is inert on the non-negative
    scattering magnitudes.

    Args:
        arr: Channels-first feature array $(n, C, T)$ or $(C, T)$.
        field: Field name (one of ``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph``).
        log_epsilon: Additive $\epsilon$ inside the log (production uses $10^{-6}$).

    Returns:
        The transformed array (same shape and layout as ``arr``).
    """
    out = np.asarray(arr, dtype=np.float64).copy()
    ch_axis = _channel_axis(out.ndim)
    if field in _LOG_FIELDS:
        # log(clamp(x, 0) + eps) on every channel except order-0 (channel index 0).
        idx = [slice(None)] * out.ndim
        idx[ch_axis] = slice(1, None)
        sel = tuple(idx)
        out[sel] = np.log(np.clip(out[sel], 0.0, None) + log_epsilon)
    elif field in _ASINH_FIELDS:
        out = np.arcsinh(out)
    return out


def _broadcast_stat(vec: np.ndarray, ndim: int) -> np.ndarray:
    r"""Reshape a per-channel stat vector for broadcasting over a channels-first array.

    Args:
        vec: Per-channel statistic, shape $(C,)$.
        ndim: Dimensionality of the target array (``2`` or ``3``).

    Returns:
        ``vec`` reshaped to $(1, C, 1)$ for ``ndim == 3`` or $(C, 1)$ for ``ndim == 2``.
    """
    if ndim == 3:
        return vec[None, :, None]
    if ndim == 2:
        return vec[:, None]
    raise ValueError(f"expected ndim 2 or 3, got {ndim}")


def compute_norm_stats(
    fields: Dict[str, np.ndarray], *, log_epsilon: float = LOG_EPSILON
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Compute per-channel normalisation stats from a synthetic pool (``synthetic_pool``).

    Stats are computed on the **transformed** features (post log / asinh), matching the
    production convention where the z-score follows the pointwise transform. For each field
    the mean and std are taken over the sample and time axes, giving one value per channel.

    Args:
        fields: Channels-first feature arrays keyed by field name, each $(n, C, T)$.
        log_epsilon: Additive $\epsilon$ inside the log.

    Returns:
        A dict ``{field: {'mean': (C,), 'std': (C,)}}`` (float32).
    """
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for field, arr in fields.items():
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(
                f"compute_norm_stats expects channels-first (n, C, T) arrays; "
                f"field {field!r} has shape {arr.shape}"
            )
        transformed = _transform_field(arr, field, log_epsilon)
        mean = transformed.mean(axis=(0, 2)).astype(np.float32)
        std = transformed.std(axis=(0, 2)).astype(np.float32)
        stats[field] = {"mean": mean, "std": std}
    return stats


def normalise_fields(
    fields: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, np.ndarray]],
    *,
    log_epsilon: float = LOG_EPSILON,
    z_eps: float = Z_EPSILON,
) -> Dict[str, np.ndarray]:
    r"""Normalise channels-first feature fields (log/asinh + per-channel z-score).

    This is a local, parity-tested reimplementation of
    ``hdf5_dataset.hdf5_dataset.normalize_tensor_data`` restricted to the four v2 fields:
    order-0 untouched, $\ln(\max(x, 0) + \epsilon)$ on ``*_st`` channels $1..$, $\operatorname{asinh}$
    on all ``*_ph`` channels, then $(x - \mu_c) / (\sigma_c + \epsilon_z)$ per channel.

    Args:
        fields: Channels-first feature arrays keyed by field name, $(n, C, T)$ or $(C, T)$.
        stats: Per-channel stats ``{field: {'mean': (C,), 'std': (C,)}}``.
        log_epsilon: Additive $\epsilon$ inside the log (production uses $10^{-6}$).
        z_eps: Additive $\epsilon$ on the std in the z-score (production uses $10^{-8}$).

    Returns:
        A dict of normalised arrays (float32), same keys / layout as ``fields``.
    """
    out: Dict[str, np.ndarray] = {}
    for field, arr in fields.items():
        arr = np.asarray(arr)
        if field not in stats:
            raise KeyError(f"normalise_fields: no stats for field {field!r}")
        transformed = _transform_field(arr, field, log_epsilon)
        mean = _broadcast_stat(np.asarray(stats[field]["mean"], dtype=np.float64), arr.ndim)
        std = _broadcast_stat(np.asarray(stats[field]["std"], dtype=np.float64), arr.ndim)
        out[field] = ((transformed - mean) / (std + z_eps)).astype(np.float32)
    return out


def load_real_fold_stats(
    path: Optional[str], *, fields: Tuple[str, ...] = _FIELD_NAMES
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Load per-channel normalisation stats from a real training fold's ``stats.hdf5``.

    Reads, per field, a ``mean`` dataset and either a ``std`` or a ``variance`` dataset
    (production stats store ``variance``; $\sigma = \sqrt{\mathrm{variance}}$). The default
    ``norm_stats_source`` is ``synthetic_pool``; ``real_fold`` is opt-in and is validated
    against a concrete fold when one is wired in a later sprint.

    Args:
        path: Path to the fold's ``stats.hdf5`` (must be set when ``real_fold`` is selected).
        fields: Field names to load stats for.

    Returns:
        A dict ``{field: {'mean': (C,), 'std': (C,)}}`` (float32).

    Raises:
        ValueError: If ``path`` is ``None`` (``real_fold`` selected without a path).
        KeyError: If a field's group / datasets are missing from the file.
    """
    if path is None:
        raise ValueError(
            "norm_stats_source='real_fold' requires 'real_fold_stats_path' to be set in "
            "the config; got None."
        )
    import h5py  # local import: only needed for the opt-in real_fold path

    stats: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        for field in fields:
            if field not in handle:
                raise KeyError(f"real-fold stats file {path!r} has no group for {field!r}")
            group = handle[field]
            if "mean" not in group:
                raise KeyError(f"real-fold stats {field!r} is missing a 'mean' dataset")
            mean = np.asarray(group["mean"], dtype=np.float32).reshape(-1)
            if "std" in group:
                std = np.asarray(group["std"], dtype=np.float32).reshape(-1)
            elif "variance" in group:
                std = np.sqrt(np.asarray(group["variance"], dtype=np.float32)).reshape(-1)
            else:
                raise KeyError(
                    f"real-fold stats {field!r} needs a 'std' or 'variance' dataset"
                )
            stats[field] = {"mean": mean, "std": std}
    return stats


# ---------------------------------------------------------------------------
# The transform adapter
# ---------------------------------------------------------------------------


class ScatteringAdapter:
    r"""Wrap the production scattering transform + normalisation for ``synthetic_v2``.

    Constructs the (expensive) :class:`KymatioPhaseScattering1D` filter bank and the two
    phase-selection masks once, then exposes batched transform / end-to-end methods. The
    channel counts are asserted against the ``43/44/43/58`` contract when built at the
    production ``n_raw`` (see ``strict_counts``).

    Attributes:
        transform: The wrapped :class:`KymatioPhaseScattering1D`.
        device: Torch device the transform runs on.
        fs: Raw sampling rate in Hz.
        f_pulse: Pulse-shape carrier frequency in Hz (true Hz).
        Q: Wavelets per octave (for the one-$Q$-step coupled-channel tolerance).
        trim: Symmetric edge trim per end (decimated steps).
        batch_size: Sample-batch size for the transform (bounds peak GPU memory).
        scattering_channels: Emitted scattering channel count ($= $ ``fhr_st`` / ``up_st``).
        fhr_ph_channels / up_ph_channels: Selected phase-coefficient counts.
        center_freqs_np: The transform's ``center_freqs`` (normalised $\xi$) as a numpy array.
        fhr_phase_mask / up_phase_mask: Boolean pair-selection masks (on ``device``).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        benchmark: str = "G1_raw",
        device: Optional[torch.device] = None,
        strict_counts: Optional[bool] = None,
    ) -> None:
        r"""Build the transform and phase masks from ``config_synth_v2.yaml``.

        Args:
            config: The parsed config tree (reads ``benchmarks.<benchmark>.scattering`` and
                ``.raw`` plus ``runtime.device``).
            benchmark: Active benchmark key under ``benchmarks``.
            device: Torch device override; when ``None`` it is resolved from
                ``runtime.device`` (``auto`` → CUDA if available else CPU).
            strict_counts: When ``True`` the constructor asserts the ``43/44/43/58`` counts;
                when ``None`` it defaults to ``True`` iff ``raw.n_raw == 5280`` (so tests at
                short shapes, whose filter banks yield other counts, skip the assertion).
        """
        bench = config["benchmarks"][benchmark]
        scat = bench["scattering"]
        raw = bench["raw"]

        self.J = int(scat["J"])
        self.Q = int(scat["Q"])
        self.T = int(scat["T"])
        self.max_order = int(scat["max_order"])
        self.batch_size = int(scat.get("batch_size", 32))
        self.norm_stats_source = str(scat.get("norm_stats_source", "synthetic_pool"))
        self.real_fold_stats_path = scat.get("real_fold_stats_path")
        self._fhr_min_freq = float(scat["phase_min_freq"]["fhr"])
        self._up_min_freq = float(scat["phase_min_freq"]["up"])

        self.fs = float(raw["fs"])
        self.n_raw = int(raw["n_raw"])
        self.f_pulse = float(raw["f_pulse"])
        self.trim = TRIM_STEPS

        if device is None:
            requested = str(config.get("runtime", {}).get("device", "auto"))
            if requested == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(requested)
        self.device = device

        self.transform = KymatioPhaseScattering1D(
            J=self.J,
            Q=self.Q,
            T=self.T,
            shape=self.n_raw,
            max_order=self.max_order,
            border_mode="reflect",
            device=self.device,
        )
        self.transform.eval()

        # Pair-selection masks (bool over the phase-pair dimension), computed once. The
        # selector is channel-independent: only the two min-freq thresholds differ.
        self.fhr_phase_mask = self.transform.select_fhr_phase_coefficients(
            min_freq=self._fhr_min_freq
        )["optimal_mask"]
        self.up_phase_mask = self.transform.select_fhr_phase_coefficients(
            min_freq=self._up_min_freq
        )["optimal_mask"]

        self.center_freqs_np = (
            self.transform.center_freqs.detach().cpu().numpy().astype(np.float64)
        )
        # Scattering emits one order-0 channel plus one per first-order wavelet (== the
        # number of center frequencies) for max_order=1; validated live in transform_raw.
        self.scattering_channels = 1 + int(self.center_freqs_np.shape[0])
        self.fhr_ph_channels = int(self.fhr_phase_mask.sum().item())
        self.up_ph_channels = int(self.up_phase_mask.sum().item())

        if strict_counts is None:
            strict_counts = self.n_raw == PRODUCTION_N_RAW
        self.strict_counts = bool(strict_counts)
        if self.strict_counts:
            self.assert_production_counts()

    def assert_production_counts(self) -> None:
        r"""Assert the emitted channel counts equal the ``43/44/43/58`` model contract.

        Raises:
            ValueError: If ``(scattering, fhr_ph, up_ph)`` is not ``(43, 44, 58)``. The
                scattering count is shared by ``fhr_st`` and ``up_st``.
        """
        actual = (self.scattering_channels, self.fhr_ph_channels, self.up_ph_channels)
        expected = (FHR_ST_CHANNELS, FHR_PH_CHANNELS, UP_PH_CHANNELS)
        if actual != expected:
            raise ValueError(
                "scattering channel counts do not match the model contract: got "
                f"(scattering/st={actual[0]}, fhr_ph={actual[1]}, up_ph={actual[2]}), "
                f"expected (st={expected[0]}, fhr_ph={expected[1]}, up_ph={expected[2]}); "
                "the local kymatio filter bank differs from the verified build."
            )

    def _time_length(self) -> int:
        r"""Return the trimmed decimated length ($T' - 2\,\mathrm{trim}$)."""
        t_full = self.n_raw // self.T
        t_out = t_full - 2 * self.trim
        if t_out <= 0:
            raise ValueError(
                f"trim ({self.trim}) too large for decimated length {t_full} "
                f"(n_raw={self.n_raw}, T={self.T})"
            )
        return t_out

    def _one_pass(self, signal: torch.Tensor, phase_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        r"""Run one self-phase pass and return trimmed (scattering, masked-phase) arrays.

        Args:
            signal: A single-channel raw batch $(b, N)$ on ``self.device``.
            phase_mask: Boolean pair-selection mask for this signal (on ``self.device``).

        Returns:
            ``(st, ph)`` channels-first numpy arrays $(b, C_{st}, T)$ and $(b, C_{ph}, T)$,
            edge-trimmed to the analysis window.
        """
        with torch.no_grad():
            out = self.transform(
                signal,
                compute_phase=True,
                compute_cross_phase=False,
                scattering_channel=0,
                phase_channels=[0],
            )
        lo, hi = self.trim, (self.n_raw // self.T) - self.trim
        st = out["scattering"][:, :, lo:hi]
        ph = out["phase_corr"][:, phase_mask, :][:, :, lo:hi]
        return st.detach().cpu().numpy(), ph.detach().cpu().numpy()

    def transform_raw(
        self, fhr_raw: np.ndarray, up_raw: np.ndarray
    ) -> Dict[str, np.ndarray]:
        r"""Transform raw FHR/UP pairs into the four **un-normalised** feature fields.

        Two passes per batch (one per signal). Batches over samples with ``batch_size`` to
        bound peak GPU memory. Output is channels-first and edge-trimmed to $T = 300$ (at
        the production ``n_raw``).

        Args:
            fhr_raw: FHR waveform(s), $(n, N)$ or $(N,)$.
            up_raw: UP waveform(s), $(n, N)$ or $(N,)$.

        Returns:
            A dict with channels-first arrays ``fhr_st`` $(n, 43, 300)$, ``fhr_ph``
            $(n, 44, 300)$, ``up_st`` $(n, 43, 300)$, ``up_ph`` $(n, 58, 300)$ (float32).

        Raises:
            ValueError: If the inputs disagree in shape/length, or the emitted counts /
                trimmed length do not match the built transform (fails loudly).
        """
        fhr = np.atleast_2d(np.asarray(fhr_raw, dtype=np.float32))
        up = np.atleast_2d(np.asarray(up_raw, dtype=np.float32))
        if fhr.shape != up.shape:
            raise ValueError(
                f"fhr_raw {fhr.shape} and up_raw {up.shape} must have the same shape"
            )
        if fhr.shape[1] != self.n_raw:
            raise ValueError(
                f"raw length {fhr.shape[1]} != adapter n_raw {self.n_raw}; build the "
                "adapter with a matching raw.n_raw"
            )

        n = fhr.shape[0]
        chunks: Dict[str, list] = {name: [] for name in _FIELD_NAMES}
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            fhr_dev = torch.from_numpy(fhr[start:stop]).to(self.device)
            up_dev = torch.from_numpy(up[start:stop]).to(self.device)
            fhr_st, fhr_ph = self._one_pass(fhr_dev, self.fhr_phase_mask)
            up_st, up_ph = self._one_pass(up_dev, self.up_phase_mask)
            chunks["fhr_st"].append(fhr_st)
            chunks["fhr_ph"].append(fhr_ph)
            chunks["up_st"].append(up_st)
            chunks["up_ph"].append(up_ph)

        fields = {name: np.concatenate(chunks[name], axis=0) for name in _FIELD_NAMES}

        # Fail loudly if the emitted layout drifts from what the transform reported.
        t_out = self._time_length()
        expected_channels = {
            "fhr_st": self.scattering_channels,
            "fhr_ph": self.fhr_ph_channels,
            "up_st": self.scattering_channels,
            "up_ph": self.up_ph_channels,
        }
        for name, arr in fields.items():
            if arr.shape != (n, expected_channels[name], t_out):
                raise ValueError(
                    f"field {name!r} has shape {arr.shape}, expected "
                    f"{(n, expected_channels[name], t_out)}"
                )
        return fields

    def transform_cross(self, fhr_raw: np.ndarray, up_raw: np.ndarray) -> np.ndarray:
        r"""Optional UP→FHR cross-phase field (probe path only; not fed to the model).

        Produces the cross-phase coefficients with a third forward pass
        (``compute_cross_phase=True``). Used by the §14.3 realizability probe to inspect
        cross-channel coupling; the four training fields never use this.

        Args:
            fhr_raw: FHR waveform(s), $(n, N)$ or $(N,)$.
            up_raw: UP waveform(s), $(n, N)$ or $(N,)$.

        Returns:
            The edge-trimmed cross-phase array $(n, C_{\mathrm{cross}}, T)$ (float32,
            channels-first). Channels are stacked as ``[up, fhr]`` (channel 0 = UP).
        """
        fhr = np.atleast_2d(np.asarray(fhr_raw, dtype=np.float32))
        up = np.atleast_2d(np.asarray(up_raw, dtype=np.float32))
        if fhr.shape != up.shape:
            raise ValueError("fhr_raw and up_raw must have the same shape")
        n = fhr.shape[0]
        lo, hi = self.trim, (self.n_raw // self.T) - self.trim
        chunks = []
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            stacked = np.stack([up[start:stop], fhr[start:stop]], axis=1)  # (b, 2, N)
            dev = torch.from_numpy(stacked).to(self.device)
            with torch.no_grad():
                out = self.transform(
                    dev,
                    compute_phase=False,
                    compute_cross_phase=True,
                    scattering_channel=0,
                    phase_channels=[0, 1],
                )
            cross = out["cross_phase_corr"][:, :, lo:hi]
            chunks.append(cross.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def coupled_channel_indices(
        self, fs: Optional[float] = None, f_pulse: Optional[float] = None
    ) -> Dict[str, Any]:
        r"""Identify the ``*_st`` first-order channel carrying the pulse-shape carrier.

        The scattering channel that tracks the coupled latent amplitude is the first-order
        channel whose centre frequency is closest to $f_{\mathrm{pulse}}$. ``center_freqs``
        holds normalised $\xi$, so physical Hz $= \xi\,f_s$; the ``1 +`` skips the order-0
        baseline channel.

        Args:
            fs: Sampling rate override in Hz (defaults to the config ``raw.fs``).
            f_pulse: Carrier override in Hz (defaults to the config ``raw.f_pulse``).

        Returns:
            A dict ``{'up_st': idx, 'fhr_st': idx, 'hz': chosen_hz, 'xi': chosen_xi}`` — the
            same first-order index for both signals (identical filter bank and carrier).

        Raises:
            ValueError: If the chosen channel's Hz is not within one $Q$-step
                ($f_{\mathrm{pulse}} \cdot 2^{\pm 1/Q}$) of $f_{\mathrm{pulse}}$.
        """
        fs = self.fs if fs is None else float(fs)
        f_pulse = self.f_pulse if f_pulse is None else float(f_pulse)
        hz = self.center_freqs_np * fs
        k = int(np.argmin(np.abs(hz - f_pulse)))
        chosen_hz = float(hz[k])
        st_idx = k + 1  # skip order-0 (channel 0)
        lo = f_pulse * 2.0 ** (-1.0 / self.Q)
        hi = f_pulse * 2.0 ** (1.0 / self.Q)
        if not (lo <= chosen_hz <= hi):
            raise ValueError(
                f"coupled channel {st_idx} at {chosen_hz:.5f} Hz is not within one "
                f"Q-step [{lo:.5f}, {hi:.5f}] Hz of f_pulse={f_pulse:.5f} Hz "
                f"(Q={self.Q})"
            )
        return {
            "up_st": st_idx,
            "fhr_st": st_idx,
            "hz": chosen_hz,
            "xi": float(self.center_freqs_np[k]),
        }

    def transform_and_normalise(
        self,
        fhr_raw: np.ndarray,
        up_raw: np.ndarray,
        *,
        stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        norm_stats_source: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        r"""End-to-end: transform → trim → normalise → transpose to model layout.

        Args:
            fhr_raw: FHR waveform(s), $(n, N)$ or $(N,)$.
            up_raw: UP waveform(s), $(n, N)$ or $(N,)$.
            stats: Precomputed per-channel stats to reuse (e.g. a locked pool / real fold);
                when ``None`` they are derived per ``norm_stats_source``.
            norm_stats_source: Override for the config's ``norm_stats_source``
                (``synthetic_pool`` computes stats from this pool; ``real_fold`` loads them).

        Returns:
            ``(fields, stats)`` where ``fields`` is a dict of **model-/cache-facing**
            $(n, 300, C)$ arrays ``fhr_st`` $(300, 43)$, ``fhr_ph`` $(300, 44)$, ``up_st``
            $(300, 43)$, ``up_ph`` $(300, 58)$ (float32), and ``stats`` are the per-channel
            stats used (so a build can reuse a single pooled normalisation).
        """
        source = self.norm_stats_source if norm_stats_source is None else norm_stats_source
        fields_cf = self.transform_raw(fhr_raw, up_raw)  # channels-first (n, C, T)

        if stats is None:
            if source == "synthetic_pool":
                stats = compute_norm_stats(fields_cf)
            elif source == "real_fold":
                stats = load_real_fold_stats(self.real_fold_stats_path)
            else:
                raise ValueError(
                    f"unknown norm_stats_source {source!r} (expected 'synthetic_pool' "
                    "or 'real_fold')"
                )

        normed_cf = normalise_fields(fields_cf, stats)
        fields = {
            name: np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))
            for name, arr in normed_cf.items()
        }
        return fields, stats
