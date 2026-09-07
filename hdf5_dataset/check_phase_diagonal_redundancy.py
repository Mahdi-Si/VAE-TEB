r"""Measure how redundant the diagonal phase-harmonic channels are.

Settles open item 1 of ``PHASE_HARMONIC_CHANNEL_SELECTION.md`` §12: the
recommendation to drop the diagonal ($k = 0$) rests on a correlation measured
on a *synthetic* FHR-like signal, and the document asks for it to be repeated
on a real shard before committing.

For $i = j$ and $p = 1$ the phase-harmonic operator is the identity, so

$$C_{i,i,1} = \phi * \left(z_i \overline{z_i}\right) = \phi * |z_i|^2$$

while the scattering channel stored alongside it is $\phi * |z_i|$. These are
different quantities, but if $|z_i|$ is roughly constant across the $T = 16$
(4 s) smoothing window then $\phi * |z_i|^2 \approx (\phi * |z_i|)^2$, and
after the pipeline's transforms ($\operatorname{asinh}$ on phase,
$\log(\cdot + \varepsilon)$ on scattering) plus per-channel standardisation the
two become near-collinear. This script measures exactly that collinearity,
under the transforms the pipeline actually applies.

**It re-runs the transform rather than reading the stored ``fhr_ph`` block.**
Recovering which of the stored channels are diagonal would mean reconstructing
the ordering of the very selector being replaced; indexing ``model.autoc_idx``
into the full pair axis is unambiguous and works against old and new shards
alike.

Usage on the production box::

    python hdf5_dataset/check_phase_diagonal_redundancy.py \
        --hdf5 /path/to/fold_1/train/healthy_bg_cs.hdf5 --n-samples 256

Reads only; writes nothing. Interpretation is printed at the end: if the
**in-band** median $|r|$ falls below ~0.9, keep the diagonal by setting
``PHASE_HARMONIC_K_STEPS = (0, 4, 6, 8)`` in ``create_new_pipeline.py``.
"""

import os
import sys
import argparse
from typing import Optional, Tuple

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D

# Must match the pipeline's stats epsilon so the comparison reflects the
# transform the model actually sees.
LOG_EPSILON = 1e-6

# Band edges mirroring create_new_pipeline's FHR_PHASE_BAND_HZ /
# UP_PHASE_BAND_HZ. Duplicated rather than imported so this script runs without
# the pipeline's production-only imports.
DEFAULT_FHR_BAND_HZ = (0.008, 1.00)
DEFAULT_UP_BAND_HZ = (0.008, 0.05)


class _PearsonAccumulator:
    r"""Streaming Pearson correlation for each of ``n_channels`` channel pairs.

    Accumulates the sufficient statistics $\sum a$, $\sum b$, $\sum a^2$,
    $\sum b^2$, $\sum ab$ and $n$ per channel, so memory stays flat regardless
    of how many samples are processed.

    Args:
        n_channels: Number of independent channel pairs to track.
    """

    def __init__(self, n_channels: int) -> None:
        self._n = np.zeros(n_channels, dtype=np.float64)
        self._sa = np.zeros(n_channels, dtype=np.float64)
        self._sb = np.zeros(n_channels, dtype=np.float64)
        self._saa = np.zeros(n_channels, dtype=np.float64)
        self._sbb = np.zeros(n_channels, dtype=np.float64)
        self._sab = np.zeros(n_channels, dtype=np.float64)

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        """Add a batch.

        Args:
            a: Shape ``(n_channels, n_points)``, first variable.
            b: Shape ``(n_channels, n_points)``, second variable.
        """
        finite = np.isfinite(a) & np.isfinite(b)
        a = np.where(finite, a, 0.0)
        b = np.where(finite, b, 0.0)
        self._n += finite.sum(axis=1)
        self._sa += a.sum(axis=1)
        self._sb += b.sum(axis=1)
        self._saa += (a * a).sum(axis=1)
        self._sbb += (b * b).sum(axis=1)
        self._sab += (a * b).sum(axis=1)

    def correlation(self) -> np.ndarray:
        r"""Finalise and return Pearson $r$ per channel.

        Returns:
            Shape ``(n_channels,)``. ``NaN`` where a channel had no finite
            points or zero variance.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            n = np.where(self._n > 1, self._n, np.nan)
            cov = self._sab / n - (self._sa / n) * (self._sb / n)
            var_a = self._saa / n - (self._sa / n) ** 2
            var_b = self._sbb / n - (self._sb / n) ** 2
            return cov / np.sqrt(var_a * var_b)


def _assert_scattering_channel_map(model: KymatioPhaseScattering1D) -> None:
    r"""Verify that wavelet filter $i$ maps to scattering channel $i + 1$.

    The whole comparison depends on pairing each diagonal phase channel with
    the correct scattering channel. Kymatio emits one order-0 channel followed
    by the order-1 channels in filter-bank order, so the map is $i \mapsto i+1$
    — but that is checked here rather than assumed.

    Args:
        model: Constructed transform.

    Raises:
        RuntimeError: If the layout differs from the assumed mapping.
    """
    meta = model.meta()
    order = np.asarray(meta["order"])
    xi = np.asarray(meta["xi"])[:, 0]
    center_freqs = model.center_freqs.cpu().numpy()

    n_order0 = int((order == 0).sum())
    if n_order0 != 1:
        raise RuntimeError(
            f"Expected exactly 1 order-0 scattering channel, found {n_order0}. "
            "The i -> i+1 channel map no longer holds."
        )
    order1_xi = xi[order == 1]
    if order1_xi.shape != center_freqs.shape or not np.allclose(
        order1_xi, center_freqs, atol=1e-7
    ):
        raise RuntimeError(
            "Order-1 scattering centre frequencies do not match "
            "`center_freqs` elementwise; the i -> i+1 channel map no longer "
            "holds and the diagonal/scattering pairing would be wrong."
        )


def _stream_correlations(
    model: KymatioPhaseScattering1D,
    hdf5_path: str,
    channel: int,
    n_samples: int,
    batch_size: int,
    trim_steps: int,
    device: str,
) -> np.ndarray:
    r"""Correlate every diagonal phase channel with its scattering counterpart.

    Args:
        model: Constructed transform.
        hdf5_path: Shard to read raw ``fhr`` / ``up`` from.
        channel: 0 for FHR, 1 for UP.
        n_samples: Maximum number of segments to read.
        batch_size: Segments per forward pass.
        trim_steps: Decimated steps dropped from each end, excluding the
            padding-contaminated segment edges.
        device: Torch device string.

    Returns:
        Pearson $r$ per wavelet filter, shape ``(n_filters,)``.
    """
    field = "fhr" if channel == 0 else "up"
    accumulator = _PearsonAccumulator(len(model.autoc_idx))
    autoc_idx = model.autoc_idx.to(device)

    with h5py.File(hdf5_path, "r") as f:
        total = min(n_samples, int(f[field].shape[0]))
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            raw = np.asarray(f[field][start:end], dtype=np.float32)
            # The transform expects (B, n_signal_channels, N); it only reads
            # the channel selected below, so a single-channel stack suffices.
            batch = torch.from_numpy(raw[:, None, :]).to(device)

            with torch.no_grad():
                out = model(
                    x=batch,
                    compute_phase=True,
                    compute_cross_phase=False,
                    scattering_channel=0,
                    phase_channels=[0],
                )

            # phase_corr: (B, n_pairs, T) -> diagonal only, (B, n_filters, T)
            diag = out["phase_corr"][:, autoc_idx, :]
            # scattering: (B, 1 + n_filters, T); channel 0 is order-0.
            scat = out["scattering"][:, 1:, :]

            if trim_steps > 0:
                diag = diag[:, :, trim_steps:-trim_steps]
                scat = scat[:, :, trim_steps:-trim_steps]

            # Apply the pipeline's own per-field transforms before correlating.
            a = torch.asinh(diag)
            b = torch.log(torch.clamp(scat, min=0.0) + LOG_EPSILON)

            # (B, C, T) -> (C, B*T)
            a = a.permute(1, 0, 2).reshape(a.shape[1], -1).cpu().numpy()
            b = b.permute(1, 0, 2).reshape(b.shape[1], -1).cpu().numpy()
            accumulator.update(a.astype(np.float64), b.astype(np.float64))

    return accumulator.correlation()


def _summarise(r: np.ndarray, label: str) -> Optional[float]:
    r"""Print median $|r|$ and the high-correlation fractions for a subset.

    Args:
        r: Pearson $r$ per filter, may contain ``NaN``.
        label: Row label.

    Returns:
        The median $|r|$, or ``None`` when the subset has no finite values.
    """
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        print(f"  {label:<28s} no finite correlations")
        return None
    abs_r = np.abs(finite)
    median = float(np.median(abs_r))
    print(
        f"  {label:<28s} n={finite.size:3d}  median|r|={median:.3f}  "
        f">0.95: {np.mean(abs_r > 0.95):.2f}  >0.99: {np.mean(abs_r > 0.99):.2f}"
    )
    return median


def analyse_stream(
    model: KymatioPhaseScattering1D,
    hdf5_path: str,
    channel: int,
    band_hz: Tuple[float, float],
    fs: float,
    args: argparse.Namespace,
) -> Optional[float]:
    r"""Run and report the redundancy measurement for one signal stream.

    Args:
        model: Constructed transform.
        hdf5_path: Shard to read.
        channel: 0 for FHR, 1 for UP.
        band_hz: The selection band; defines the decision-relevant subset.
        fs: Sampling rate in Hz.
        args: Parsed CLI arguments.

    Returns:
        In-band median $|r|$, or ``None`` when unavailable.
    """
    name = "FHR" if channel == 0 else "UP"
    print(f"\n{name}  (band {band_hz[0]}-{band_hz[1]} Hz)")

    r = _stream_correlations(
        model=model,
        hdf5_path=hdf5_path,
        channel=channel,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        trim_steps=args.trim_steps,
        device=args.device,
    )

    freqs_hz = model.center_freqs.cpu().numpy() * fs
    in_band = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1])

    _summarise(r, "all filters")
    median_in_band = _summarise(r[in_band], "in-band (decision-relevant)")

    if args.per_channel:
        print(f"\n  {'filter':>6s} {'xi (Hz)':>10s} {'r':>8s}  in-band")
        for idx in range(len(r)):
            flag = "yes" if in_band[idx] else ""
            print(
                f"  {idx:6d} {freqs_hz[idx]:10.4f} {r[idx]:8.3f}  {flag}"
            )

    return median_in_band


def main() -> int:
    """Parse arguments, run both streams, and print the verdict.

    Returns:
        Process exit code: 0 on success, 1 if the shard cannot be used.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure diagonal phase-harmonic redundancy against the "
            "scattering block on a real shard."
        )
    )
    parser.add_argument("--hdf5", required=True, help="Path to a dataset shard.")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Segments to read (default: 256).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Segments per forward pass (default: 16).")
    # Default 15 = int(4*60*1.0)//16, matching the shipped trim_minutes=1.0,
    # so the measurement covers the region the model is actually trained on.
    # The untrimmed edges are driven by reflection padding rather than signal,
    # which perturbs |r| in a direction that depends on the recording — on one
    # synthetic sample trimming moved the in-band UP median from 0.825 to
    # 0.878, i.e. upward. Since the verdict is a hard threshold, measure the
    # trimmed region rather than assuming the bias is benign in either
    # direction. Pass --trim-steps 0 to see the untrimmed figure.
    parser.add_argument("--trim-steps", type=int, default=15,
                        help="Decimated steps trimmed from each end "
                             "(default: 15, matching trim_minutes=1.0; "
                             "pass 0 to measure the untrimmed segment).")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available).")
    parser.add_argument("--fs", type=float, default=4.0,
                        help="Sampling rate in Hz (default: 4.0).")
    parser.add_argument("--per-channel", action="store_true",
                        help="Print the full per-filter table.")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.hdf5):
        print(f"error: shard not found: {args.hdf5}", file=sys.stderr)
        return 1

    with h5py.File(args.hdf5, "r") as f:
        for required in ("fhr", "up"):
            if required not in f:
                print(f"error: '{required}' missing from {args.hdf5}",
                      file=sys.stderr)
                return 1
        signal_length = int(f["fhr"].shape[1])
        available = int(f["fhr"].shape[0])

    print(f"shard:  {args.hdf5}")
    print(f"using:  {min(args.n_samples, available)} of {available} segments "
          f"(len {signal_length}), device={args.device}, "
          f"trim_steps={args.trim_steps}")

    model = KymatioPhaseScattering1D(
        J=11, Q=4, T=16, shape=signal_length,
        device=args.device, tukey_alpha=None, max_order=1,
    )
    _assert_scattering_channel_map(model)

    fhr_median = analyse_stream(
        model, args.hdf5, 0, DEFAULT_FHR_BAND_HZ, args.fs, args
    )
    up_median = analyse_stream(
        model, args.hdf5, 1, DEFAULT_UP_BAND_HZ, args.fs, args
    )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    # Plain ASCII: this is read on consoles whose encoding is not UTF-8.
    print("Synthetic reference (doc sec. 5): median|r| = 0.967, "
          ">0.95: 0.57, >0.99: 0.26")
    medians = [m for m in (fhr_median, up_median) if m is not None]
    if not medians:
        print("No usable correlations — cannot decide. Check the shard.")
        return 1

    worst = min(medians)
    if worst < 0.9:
        print(
            f"In-band median |r| drops to {worst:.3f} (< 0.90): the diagonal "
            f"carries enough unique\nvariance to keep. Set "
            f"PHASE_HARMONIC_K_STEPS = (0, 4, 6, 8) in "
            f"create_new_pipeline.py\n(-> fhr_ph 94, up_ph 26) and regenerate."
        )
    else:
        print(
            f"In-band median |r| is {worst:.3f} (>= 0.90): the diagonal is "
            f"largely redundant with\nthe scattering block stored alongside "
            f"it. Keep the shipped\nPHASE_HARMONIC_K_STEPS = (4, 6, 8) "
            f"(-> fhr_ph 66, up_ph 15)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
