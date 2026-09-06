r"""Cross-spectral arithmetic for the forecast, on the $\tau$-slice grid. One implementation, two
consumers.

``metrics`` computes the sufficient statistics on the device; ``analyses/coherence`` turns them
into readouts. An analysis may not import another and neither may reach into the other's
internals, so the arithmetic they share lives here -- the same reason ``frames``, ``lag_axis`` and
``events`` exist one layer down.

This module is **numpy only**. It holds no model, no torch and no I/O.

The $\tau$-slice
================

The construction the whole analysis rests on. Anchor $t$'s forecast of horizon step $\tau$, raw
sub-sample $r$, is raw index

$$n(t, \tau, r) = R\,(t + 1) + R\,\tau + r, \qquad \tau \in [0, H),\ r \in [0, R).$$

**Fix $\tau$ and concatenate over consecutive anchors.** Incrementing $t$ by one advances the
block start by exactly $R$, and each anchor contributes exactly $R$ consecutive samples, so the
concatenation is contiguous, gap-free and non-overlapping. Over the trained anchors
$t \in [w, T_{\mathrm{valid}})$ that is $A\,R$ samples of continuous $4\,$Hz signal -- at the
shipped geometry $240 \times 16 = 3840$ samples, $960$ s.

Three properties follow, and all three are why this module exists rather than a Welch call inside
one forecast block:

1. **The resolution is the slice's, not the block's.** A single $H \cdot R$-sample block supports
   ``nperseg = 64`` at best, which puts the whole $[0, 0.04)$ Hz deceleration span in the DC bin
   that the detrend has already removed -- the documented reason spectral analysis was deferred.
   A $960$ s slice supports ``nperseg = 512``, giving $\Delta f = 7.8125$ mHz and four bins below
   $0.03$ Hz.
2. **Lead time is an axis, not a trade.** Slice $\tau$ holds lead times
   $[(R\tau + 1)/f_s,\ R(\tau + 1)/f_s]$ -- at the shipped geometry $[4\tau + 0.25,\ 4\tau + 4]$ s.
   Thirty slices tile $0$--$120$ s, each at the *full* frequency resolution. There is no
   STFT-inside-a-block time/frequency compromise anywhere in this analysis.
3. **Validity stays exact.** The forecast mask is constant within a decimated step, so as long as
   ``nperseg`` and the hop are integer multiples of $R$ a Welch window spans a whole number of
   anchors and its validity is an exact ``all()`` over decimated mask entries. That is what lets
   the estimator *drop* a window touching a gap rather than interpolate across it. Hence
   :data:`NPERSEG_STEPS` and :data:`HOP_STEPS` are expressed in **decimated steps**, and the raw
   lengths are derived from them -- never the other way round.

The truth's $\tau$-slices are shifted copies of one signal, so $S_{xx}$ is $\tau$-invariant up to
the mask; every $\tau$-dependence in the coherence, the gain and the phase is the forecast's.

Sums, then ratio -- always
==========================

Nothing here forms a ratio the caller could have formed earlier. Every function takes accumulated
$S_{xx}$, $S_{yy}$ and $S_{xy}$ and ratios once, at the end, and that is load-bearing twice over:

* $S_{ee} = S_{xx} + S_{yy} - 2\,\mathrm{Re}\,S_{xy}$ is **linear** in the three spectra, so the
  three-way decomposition, the band totals and the Parseval reconciliation are exact
  *simultaneously* only when $\gamma$, $g$ and $\phi$ all come from one aggregated triple.
* Magnitude-squared coherence is upward-biased at low averaging depth,
  $\mathbb{E}[\hat\gamma^2] \approx \gamma^2 + (1 - \gamma^2)/n_d$. A single segment contributes at
  most $14$ windows per $\tau$ -- a bias of up to $7$ percentage points. Summing over windows,
  over a recording's segments and over recordings before ratioing drives $n_d$ into the thousands,
  where the bias is $10^{-4}$.

Scaling convention
==================

Periodic Hann, $U = \sum_n w_n^2$; each window's own mean removed from each series independently;
one-sided weights $c_0 = c_{N/2} = 1$ and $c_k = 2$ otherwise; and

$$P_k = \frac{c_k\,\overline{X_k}\,Y_k}{N\,U}
\qquad\Longrightarrow\qquad
\sum_k P_{xx,k} = \frac{1}{U}\sum_n w_n^2 \,(x_n - \bar x)^2 .$$

So a spectrum here is a **variance in $z^2$, not a density**: there is no $\Delta f$ divisor, which
is exactly what makes Parseval hold as a plain sum over bins and therefore as a plain sum over
bands. This is the single easiest thing in the module to get wrong, so it is asserted by test
rather than only stated.

Because each window's mean is removed, the spectrum carries **no** information about the forecast's
level. A baseline offset is invisible here and is ``forecast``'s mean-error column instead; the
Parseval identity is correspondingly against the *detrended* residual, never the raw one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from teb_vae.lag_attn_rws.eval import events

#: Raw sampling rate, in Hz. Bound from :mod:`events` rather than restated -- the two modules
#: describe the same grid, and a second literal is how they would come to disagree.
FS_RAW = events.FS_RAW

#: Welch window length and hop, in **decimated steps**. Expressed in steps rather than in raw
#: samples so the "a window is a whole number of anchors" property holds by construction at any
#: geometry, which is what makes the gap rule an exact ``all()`` instead of a tolerance.
#:
#: $32$ steps is $128$ s at the shipped geometry, giving $\Delta f = 7.8125$ mHz. The trade is
#: stated in ``EVAL.md``: a longer window resolves the deceleration band better and is far more
#: likely to straddle a dropout, since a window touching one is dropped whole.
NPERSEG_STEPS = 32
HOP_STEPS = 16

#: Fetal heart-rate variability bands, in Hz, half-open $[\mathrm{lo}, \mathrm{hi})$ except the
#: last, which includes the Nyquist bin so the partition is exhaustive.
#:
#: These are the literature's bands for *fetal* HRV -- VLF, LF, MF ("movement": fetal movement and
#: maternal breathing) and HF (fetal breathing) -- plus a band above physiology that is reported
#: and not interpreted.
#:
#: **They are deliberately not** ``band_partition.CLINICAL_BANDS``, and the column that carries
#: them is named ``hrv_band`` rather than ``band`` so the two cannot be confused in one results
#: directory. Two reasons, both specific to this analysis: that table's $0.25$ Hz edge is exactly
#: the decoder's token-seam frequency, and a band boundary is the worst place for an artifact to
#: sit; and it has no LF/MF split, which is the distinction a spectral statement about the forecast
#: most needs. ``EVAL.md`` carries the crosswalk between the two.
#:
#: **The VLF band starts at $0$, not at the literature's $0.003$ Hz, and that is a real choice.**
#: The per-window mean is removed, but the Hann taper leaves a residue -- $\sum_n w_n (x_n - \bar
#: x)$ is not zero for a non-constant window -- so the DC bin carries signal rather than the
#: segment's level. Excluding it would leave a bin in no band and break Parseval as a band sum;
#: including it puts the taper residue where it belongs, with the rest of the sub-$0.03$ Hz
#: content. At the shipped $\Delta f$ no non-DC bin falls below $0.003$ Hz anyway, so the
#: literature's floor is vacuous here rather than overridden.
BANDS_HZ: Dict[str, Tuple[float, float]] = {
    "vlf": (0.0, 0.03),
    "lf": (0.03, 0.15),
    "mf": (0.15, 0.50),
    "hf": (0.50, 1.00),
    "noise": (1.00, 2.00),
}

#: Harmonics of the token-seam frequency to test. The decoder emits each $R$-sample sub-block from
#: a single ``nn.Linear`` applied to one horizon token, so nothing couples the last sample of one
#: token to the first of the next: a discontinuity with period $R$ is architecturally plausible in
#: $\hat\mu$ and impossible in the truth, which is smooth across that boundary by construction.
#:
#: Period $R$ samples is $f_s/R = 0.25$ Hz at the shipped geometry, landing on bin
#: $\mathrm{nperseg}/R = 32$ exactly -- a third reason ``nperseg`` must be a multiple of $R$, since
#: a seam frequency between bins would smear across the neighbourhood it is being compared against.
SEAM_HARMONICS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

#: Which neighbouring bins a seam bin is compared against, as an inclusive offset range on each
#: side. It starts at $4$ rather than $1$ so the Hann mainlobe -- which is $2$ bins wide either
#: side, plus sidelobes -- does not contaminate the reference with the very peak being measured.
SEAM_NEIGHBOURHOOD_BINS: Tuple[int, int] = (4, 12)

#: Grid on which a group delay is searched, in seconds. One raw sample: the delay is a property of
#: the raw grid and cannot be resolved below it.
DELAY_GRID_STEP_S = 1.0 / FS_RAW


# =============================================================================
# Slice and window geometry
# =============================================================================
@dataclass(frozen=True)
class SliceGeometry:
    r"""The $\tau$-slice Welch layout, derived from the model's own geometry.

    Every field is derived by :func:`slice_geometry`; the dataclass is frozen so a layout cannot be
    edited into one whose windows no longer align with anchors, which is the property the gap rule
    depends on.

    Attributes:
        n_anchors: Trained anchors per segment $A = T_{\mathrm{valid}} - w$.
        n_samples: Raw samples in one $\tau$-slice, $A\,R$.
        raw_per_step: Raw samples per horizon token $R$.
        horizon: Forecast horizon $H$ in decimated steps -- the number of $\tau$-slices.
        warmup: Leading anchors excluded from every loss, $w$.
        nperseg_steps: Welch window length in decimated steps.
        hop_steps: Welch hop in decimated steps.
        nperseg: Welch window length in raw samples, $R \cdot \mathrm{nperseg\_steps}$.
        hop: Welch hop in raw samples. The overlap is ``nperseg - hop``.
        n_windows: Windows per $(\text{segment}, \tau)$. **Zero** when the slice cannot hold one,
            which is reported rather than raised -- a geometry too short to measure is a fact about
            the run, not a bug in it.
        n_freq: One-sided bin count, $\mathrm{nperseg}/2 + 1$.
        delta_f_hz: Bin spacing $f_s / \mathrm{nperseg}$.
        fs: Raw sampling rate in Hz.
    """

    n_anchors: int
    n_samples: int
    raw_per_step: int
    horizon: int
    warmup: int
    nperseg_steps: int
    hop_steps: int
    nperseg: int
    hop: int
    n_windows: int
    n_freq: int
    delta_f_hz: float
    fs: float

    def window_anchor_span(self, index: int) -> Tuple[int, int]:
        r"""Anchors covered by Welch window ``index``, as a half-open range.

        Args:
            index: Window position in $[0, \mathrm{n\_windows})$.

        Returns:
            ``(start, stop)`` in **absolute** anchor indices -- offset by the warm-up, because the
            slice begins at anchor $w$ and the forecast mask is indexed from anchor $0$.
        """
        start = self.warmup + index * self.hop_steps
        return start, start + self.nperseg_steps

    def lead_seconds(self, tau: int) -> Tuple[float, float]:
        r"""Lead-time span of $\tau$-slice ``tau``, in seconds.

        Raw sample $(t, \tau, r)$ sits $R\tau + r + 1$ samples beyond anchor $t$'s causal endpoint
        $R(t+1) - 1$, so the slice spans $[(R\tau + 1)/f_s,\ R(\tau + 1)/f_s]$. At the shipped
        geometry that is $[4\tau + 0.25,\ 4\tau + 4]$ s, and the thirty slices tile $0$--$120$ s
        without overlap or gap.

        Args:
            tau: Horizon step in $[0, H)$.

        Returns:
            ``(earliest, latest)`` lead time in seconds.
        """
        earliest = (self.raw_per_step * int(tau) + 1) / self.fs
        latest = self.raw_per_step * (int(tau) + 1) / self.fs
        return float(earliest), float(latest)

    def lead_center_seconds(self) -> np.ndarray:
        r"""The lead-time axis, one midpoint per $\tau$-slice, $(H,)$ in seconds."""
        return np.array(
            [sum(self.lead_seconds(tau)) / 2.0 for tau in range(self.horizon)], dtype=np.float64
        )

    def describe(self) -> Dict[str, Any]:
        """Return the layout as a JSON-safe record, for the collection pass's own dump.

        Every field :func:`layout_from_record` needs to rebuild the layout is present, so an
        offline re-run reconstructs it rather than re-deriving it from whatever else happens to be
        on the record.
        """
        return {
            "n_anchors": int(self.n_anchors),
            "n_samples": int(self.n_samples),
            "raw_per_step": int(self.raw_per_step),
            "horizon": int(self.horizon),
            "warmup": int(self.warmup),
            "nperseg": int(self.nperseg),
            "hop": int(self.hop),
            "noverlap": int(self.nperseg - self.hop),
            "nperseg_steps": int(self.nperseg_steps),
            "hop_steps": int(self.hop_steps),
            "n_windows": int(self.n_windows),
            "n_freq": int(self.n_freq),
            "delta_f_hz": float(self.delta_f_hz),
            "fs_hz": float(self.fs),
            "window": "periodic Hann",
            "detrend": "constant, per window, each series independently",
            "scaling": (
                "P_k = c_k * conj(X_k) * Y_k / (nperseg * U) with U = sum(w^2) and c_0 = c_Nyq = "
                "1, else 2. A spectrum here is a variance in z^2, NOT a density: there is no "
                "delta-f divisor, which is what makes Parseval a plain sum over bins and hence "
                "over bands. Because each window's mean is removed, the spectrum says nothing "
                "about the forecast's level."
            ),
        }


def slice_geometry(
    *,
    t_valid: int,
    warmup: int,
    horizon: int,
    raw_per_step: int,
    nperseg_steps: int = NPERSEG_STEPS,
    hop_steps: int = HOP_STEPS,
    fs: float = FS_RAW,
) -> SliceGeometry:
    r"""Derive the $\tau$-slice Welch layout from the model's geometry.

    The window and hop are given in decimated steps and multiplied up, never rounded down from a
    raw length. That is what guarantees a window spans a whole number of anchors, and therefore
    that window validity is an exact ``all()`` over the forecast mask rather than a judgement about
    a partially covered step.

    Args:
        t_valid: Anchors with a fully observed forecast window, $T_{\mathrm{valid}}$.
        warmup: Leading anchors excluded from every loss, $w$.
        horizon: Forecast horizon $H$ in decimated steps.
        raw_per_step: Raw samples per horizon token $R$.
        nperseg_steps: Welch window length in decimated steps.
        hop_steps: Welch hop in decimated steps.
        fs: Raw sampling rate in Hz.

    Returns:
        The layout. ``n_windows`` is $0$ when the trained-anchor span cannot hold one window --
        reported, not raised, so a tiny geometry produces a recorded skip rather than a traceback.

    Raises:
        ValueError: On a non-positive window or hop, or a hop longer than the window (which would
            leave un-analysed samples between consecutive windows).
    """
    if nperseg_steps < 1 or hop_steps < 1:
        raise ValueError(
            f"nperseg_steps ({nperseg_steps}) and hop_steps ({hop_steps}) must both be >= 1"
        )
    if hop_steps > nperseg_steps:
        raise ValueError(
            f"hop_steps ({hop_steps}) exceeds nperseg_steps ({nperseg_steps}); consecutive "
            f"windows would leave a gap of samples that no window analyses, so the spectrum "
            f"would describe a subsample of the slice while its window count says otherwise"
        )
    n_anchors = max(int(t_valid) - int(warmup), 0)
    n_samples = n_anchors * int(raw_per_step)
    nperseg = int(nperseg_steps) * int(raw_per_step)
    hop = int(hop_steps) * int(raw_per_step)
    n_windows = 0 if n_samples < nperseg else (n_samples - nperseg) // hop + 1
    return SliceGeometry(
        n_anchors=n_anchors,
        n_samples=n_samples,
        raw_per_step=int(raw_per_step),
        horizon=int(horizon),
        warmup=int(warmup),
        nperseg_steps=int(nperseg_steps),
        hop_steps=int(hop_steps),
        nperseg=nperseg,
        hop=hop,
        n_windows=int(n_windows),
        n_freq=nperseg // 2 + 1,
        delta_f_hz=float(fs) / float(nperseg),
        fs=float(fs),
    )


def layout_from_record(record: Optional[Mapping[str, Any]]) -> Optional[SliceGeometry]:
    """Rebuild the layout a finished run used, from the block it dumped.

    An offline re-run must read the geometry the pass actually ran under rather than re-derive one
    from the shipped constants: a run collected at a different window length is still readable, and
    a silently re-derived layout would relabel its every frequency and lead time.

    Args:
        record: The ``coherence`` block of ``collection.json``, or ``None``.

    Returns:
        The layout, or ``None`` when the block is absent or predates the fields this needs.
    """
    if not record:
        return None
    required = ("raw_per_step", "horizon", "warmup", "nperseg_steps", "hop_steps")
    if any(record.get(name) is None for name in required):
        return None
    warmup = int(record["warmup"])
    return slice_geometry(
        t_valid=warmup + int(record.get("n_anchors") or 0),
        warmup=warmup,
        horizon=int(record["horizon"]),
        raw_per_step=int(record["raw_per_step"]),
        nperseg_steps=int(record["nperseg_steps"]),
        hop_steps=int(record["hop_steps"]),
        fs=float(record.get("fs_hz") or FS_RAW),
    )


def welch_window(nperseg: int) -> np.ndarray:
    r"""The periodic Hann window, $(N,)$.

    Periodic ($w_n = \tfrac12 - \tfrac12\cos(2\pi n/N)$) rather than symmetric
    ($\ldots/(N-1)$), which is the convention for spectral estimation and the one ``scipy``'s
    ``get_window`` uses: the symmetric form repeats its endpoint under the DFT's implicit
    periodicity and leaks accordingly.

    Args:
        nperseg: Window length $N$ in raw samples.

    Returns:
        The window as ``float64``.
    """
    n = np.arange(int(nperseg), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / float(nperseg))


def one_sided_weights(nperseg: int) -> np.ndarray:
    r"""One-sided folding weights $c_k$, $(N/2 + 1,)$.

    Every bin but DC and Nyquist has a conjugate twin in the discarded half of the spectrum, so it
    carries twice its own magnitude; DC and Nyquist are their own twins and carry once. Getting
    this wrong is a factor of two in exactly the two bins a reader is least likely to check, and it
    breaks Parseval by a small amount that looks like accumulated round-off.

    Args:
        nperseg: Window length $N$ in raw samples.

    Returns:
        The weights as ``float64``.
    """
    n_freq = int(nperseg) // 2 + 1
    weights = np.full(n_freq, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if int(nperseg) % 2 == 0:
        # Only an even-length window has a Nyquist bin that is its own conjugate.
        weights[-1] = 1.0
    return weights


def frequency_axis(nperseg: int, *, fs: float = FS_RAW) -> np.ndarray:
    r"""The one-sided frequency axis in Hz, $(N/2 + 1,)$, from $0$ to $f_s/2$ inclusive."""
    return np.arange(int(nperseg) // 2 + 1, dtype=np.float64) * (float(fs) / float(nperseg))


# =============================================================================
# Bands
# =============================================================================
def band_names(bands: Optional[Mapping[str, Tuple[float, float]]] = None) -> Tuple[str, ...]:
    """Return the band names in their table order, which is ascending in frequency."""
    return tuple(BANDS_HZ if bands is None else bands)


def band_edges(
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return the band table as a plain dict, for a run to dump beside its numbers."""
    return {str(name): (float(lo), float(hi)) for name, (lo, hi) in (bands or BANDS_HZ).items()}


def band_index(
    frequencies: Any, bands: Optional[Mapping[str, Tuple[float, float]]] = None
) -> np.ndarray:
    r"""Map every frequency bin to its band, and assert the partition is exact.

    The assertion is the content. Parseval holds here as a *sum over bands* only if the bands
    partition the bins -- every bin in exactly one band, none left over. A band table whose top
    edge sits below Nyquist, or whose floor sits above $\Delta f$, silently drops bins, and the
    residual-spectrum decomposition would then fail to reconcile with the time domain by an amount
    that reads as a normalisation bug rather than as a missing band.

    Args:
        frequencies: The one-sided axis in Hz, $(F,)$, ascending from $0$.
        bands: The band table. Defaults to :data:`BANDS_HZ`.

    Returns:
        $(F,)$ of ``int64``, each entry the band's position in the table's order.

    Raises:
        ValueError: If any bin falls in no band or in more than one, naming the offending bin and
            its frequency.
    """
    axis = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    table = band_edges(bands)
    assigned = np.full(axis.size, -1, dtype=np.int64)
    for position, (name, (lo, hi)) in enumerate(table.items()):
        # Half-open on the left everywhere, and closed on the right for the top band alone, so the
        # Nyquist bin -- which sits exactly on the top edge -- lands somewhere.
        is_top = position == len(table) - 1
        inside = (axis >= lo) & ((axis <= hi) if is_top else (axis < hi))
        clash = inside & (assigned >= 0)
        if clash.any():
            first = int(np.argmax(clash))
            raise ValueError(
                f"band {name!r} overlaps an earlier band at bin {first} ({axis[first]:.6g} Hz); "
                f"the band table must partition the spectrum, because every band sum here is a "
                f"term of an exact Parseval identity"
            )
        assigned[inside] = position
    if (assigned < 0).any():
        first = int(np.argmax(assigned < 0))
        raise ValueError(
            f"bin {first} ({axis[first]:.6g} Hz) falls in no band of "
            f"{sorted(table)}. Every bin from DC to Nyquist must be covered: the residual "
            f"spectrum reconciles with the time domain only as a sum over all of them, so an "
            f"uncovered bin reads downstream as a normalisation error rather than as a gap in "
            f"this table"
        )
    return assigned


def band_bin_counts(
    frequencies: Any, bands: Optional[Mapping[str, Tuple[float, float]]] = None
) -> Dict[str, int]:
    """Count the bins in each band, so a band too thin to mean anything is visible as a number.

    Args:
        frequencies: The one-sided axis in Hz.
        bands: The band table. Defaults to :data:`BANDS_HZ`.

    Returns:
        Band name -> bin count, in table order.
    """
    assigned = band_index(frequencies, bands)
    names = band_names(bands)
    return {name: int((assigned == position).sum()) for position, name in enumerate(names)}


def collapse_to_bands(values: Any, assigned: Any, n_bands: int) -> np.ndarray:
    r"""Sum a per-bin array into per-band totals along its last axis.

    Args:
        values: $(\ldots, F)$, real or complex.
        assigned: The bin-to-band map from :func:`band_index`, $(F,)$.
        n_bands: How many bands the map runs over.

    Returns:
        $(\ldots, B)$ of the same dtype.
    """
    array = np.asarray(values)
    index = np.asarray(assigned, dtype=np.int64).reshape(-1)
    out = np.zeros(array.shape[:-1] + (int(n_bands),), dtype=array.dtype)
    # ``np.add.at`` would also work but is markedly slower; one masked sum per band is fine at
    # these sizes and reads more plainly than a scatter-add.
    for position in range(int(n_bands)):
        out[..., position] = array[..., index == position].sum(axis=-1)
    return out


def reshape_band_horizon(flat: Any, *, horizon: int, n_bands: int) -> np.ndarray:
    r"""Unflatten a stored $(\ldots, H \cdot B)$ vector back to $(\ldots, H, B)$.

    The per-segment statistics travel through ``per_sample_vectors.npz`` with a single trailing
    axis -- the convention ``attention_profile_per_head`` already follows -- so every consumer
    unflattens with this one function rather than with its own ``reshape`` and its own opinion
    about which index is major. The layout is $\tau$-major.

    Args:
        flat: $(\ldots, H \cdot B)$.
        horizon: $H$.
        n_bands: $B$.

    Returns:
        $(\ldots, H, B)$.

    Raises:
        ValueError: If the trailing axis is not $H \cdot B$ long.
    """
    array = np.asarray(flat, dtype=np.float64)
    expected = int(horizon) * int(n_bands)
    if array.shape[-1] != expected:
        raise ValueError(
            f"expected a trailing axis of {expected} = horizon {horizon} x bands {n_bands}, got "
            f"{array.shape[-1]}; the stored vector belongs to a different geometry or band table"
        )
    return array.reshape(array.shape[:-1] + (int(horizon), int(n_bands)))


# =============================================================================
# Derived readouts
# =============================================================================
def derive(sxx: Any, syy: Any, sxy: Any) -> Dict[str, np.ndarray]:
    r"""Turn accumulated cross-spectra into the coherence readouts and the exact error split.

    With $\gamma^2 = |S_{xy}|^2/(S_{xx}S_{yy})$, $g = \sqrt{S_{yy}/S_{xx}}$ and
    $\phi = \arg S_{xy}$, the normalised residual spectrum decomposes **exactly**:

    $$\frac{S_{ee}}{S_{xx}}
    \;=\; \underbrace{(1 - \gamma^2)}_{\text{irreducible}}
    \;+\; \underbrace{2 g \gamma\,(1 - \cos\phi)}_{\text{timing}}
    \;+\; \underbrace{(g - \gamma)^2}_{\text{amplitude}},$$

    since the right-hand side collapses to $1 + g^2 - 2g\gamma\cos\phi$ and
    $S_{ee} = S_{xx} + S_{yy} - 2\,\mathrm{Re}\,S_{xy}$. All three terms are non-negative for any
    admissible triple: $\gamma^2 \le 1$ by Cauchy-Schwarz, and $g, \gamma \ge 0$.

    That split is what makes a coherence number actionable:

    * ``irreducible`` $= 1 - \gamma^2$ is the floor. No per-frequency complex filter applied to the
      forecast could get the error below it -- this is genuine unpredictability at that frequency.
    * ``timing`` vanishes **exactly** when $\phi = 0$. This is what arriving at the wrong moment
      costs, and nothing else contributes to it.
    * ``amplitude`` vanishes **exactly** when $g = \gamma$, the amplitude a forecast has when its
      variance matches the share of the truth it can actually account for.

    **The obvious alternative split is wrong for this purpose and was tried first.** Writing the
    excess as $\gamma^2\sin^2\phi + (g - \gamma\cos\phi)^2$ is equally exact, but its amplitude term
    is $(1 - \cos\phi)^2$ for a *pure delay* -- so a perfectly-scaled forecast that merely arrives
    late reports amplitude error, which is the exact confusion this analysis exists to remove. The
    form above puts a pure delay entirely in ``timing`` and a pure attenuation entirely in
    ``amplitude``.

    **``amplitude`` and ``gain`` answer different questions and both travel.** $g = \gamma$ is the
    mean-square-optimal amplitude given the coherence, so an MSE-trained forecaster is *supposed* to
    shrink towards it; ``amplitude`` measures the distance from that. Whether the forecast has the
    truth's variance at all is $g$ against $1$, and $g < 1$ is the over-smoothing signature -- a
    forecast that tracks the truth's shape while flattening its excursions. A well-trained model can
    be near-zero on ``amplitude`` and still far below $1$ on ``gain``; that is not a contradiction,
    it is regression to the mean, and reading either number alone hides it.

    **The arithmetic goes through $\gamma\cos\phi = \mathrm{Re}\,S_{xy}/\sqrt{S_{xx}S_{yy}}$ and
    $\gamma\sin\phi$ rather than through a magnitude and an angle**, so the identity holds to
    machine precision with no trigonometry in the path at all -- ``timing`` is formed as
    $2g(\gamma - \gamma\cos\phi)$, which needs neither a division by $\gamma$ nor a cosine.
    Recovering $\phi$ with ``arctan2`` and re-applying ``cos`` would leave a residual that grows
    with $|\phi|$ and would make the emitted ``decomposition_residual`` a measure of this function
    rather than of the data.

    Args:
        sxx: Accumulated truth auto-spectrum, $(\ldots, F)$, real and non-negative.
        syy: Accumulated forecast auto-spectrum, same shape.
        sxy: Accumulated cross-spectrum $\overline{X}Y$, same shape, complex.

    Returns:
        ``coherence``, ``gain``, ``phase_rad``, ``irreducible``, ``timing``, ``amplitude``,
        ``residual_normalised`` and ``decomposition_residual`` -- the last being the three terms
        minus the residual, which is zero to round-off and is emitted so that it is *measured*
        rather than assumed. Every entry is ``NaN`` wherever a denominator is not strictly
        positive: a bin no window contributed to, or a constant signal, is unmeasured rather than
        perfectly coherent.
    """
    xx = np.asarray(sxx, dtype=np.float64)
    yy = np.asarray(syy, dtype=np.float64)
    xy = np.asarray(sxy, dtype=np.complex128)

    # Strictly positive, both of them. A zero here is an empty accumulator or a constant signal,
    # and either way the ratio is undefined rather than unity -- 0/0 reported as perfect coherence
    # is the single most flattering way this could fail.
    # One definedness rule for every readout, so a row is measured or it is not -- rather than
    # some columns reporting on a bin others call unmeasured.
    usable = (xx > 0.0) & (yy > 0.0)
    safe_xx = np.where(usable, xx, np.nan)
    scale = np.sqrt(safe_xx * np.where(usable, yy, np.nan))

    gamma_cos = xy.real / scale
    gamma_sin = xy.imag / scale
    coherence = gamma_cos**2 + gamma_sin**2
    gamma = np.sqrt(coherence)
    gain = np.sqrt(np.where(usable, yy, np.nan) / safe_xx)

    irreducible = 1.0 - coherence
    # $2 g \gamma (1 - \cos\phi)$, written as $2 g (\gamma - \gamma\cos\phi)$ so neither a cosine
    # nor a division by a possibly-zero $\gamma$ enters the path.
    timing = 2.0 * gain * (gamma - gamma_cos)
    amplitude = (gain - gamma) ** 2
    residual_normalised = (xx + yy - 2.0 * xy.real) / safe_xx
    return {
        "coherence": coherence,
        "gain": gain,
        "phase_rad": np.arctan2(gamma_sin, gamma_cos),
        "irreducible": irreducible,
        "timing": timing,
        "amplitude": amplitude,
        "residual_normalised": residual_normalised,
        "decomposition_residual": irreducible + timing + amplitude - residual_normalised,
    }


def delay_alias_period(frequencies: Any) -> float:
    r"""The period, in seconds, beyond which a group delay is not identifiable.

    A cross-spectrum sampled on a uniform grid of spacing $\Delta f$ cannot distinguish $d$ from
    $d + 1/\Delta f$: the rotation $e^{i 2\pi k \Delta f d}$ is identical at every bin for the two.
    At the shipped $\Delta f = 7.8125$ mHz that period is $128$ s, so the principal interval is
    $[-64, +64]$ s.

    This is a property of the grid, not of the estimator, and **phase unwrapping has exactly the
    same limit** -- it is not a cost of searching rather than fitting. What the search avoids is
    the unwrap heuristic itself, which mis-tracks at a low-coherence bin and propagates that error
    to every bin above it.

    Args:
        frequencies: The frequencies in Hz, ascending and uniformly spaced, $(F,)$.

    Returns:
        $1/\Delta f$ in seconds, or ``inf`` when fewer than two bins are supplied.
    """
    axis = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if axis.size < 2:
        return float("inf")
    return 1.0 / float(axis[1] - axis[0])


def estimate_delay(
    sxy: Any,
    frequencies: Any,
    *,
    max_seconds: Optional[float] = None,
    step_seconds: float = DELAY_GRID_STEP_S,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Recover a group delay by searching for the shift that aligns the cross-spectrum's phase.

    If $y(t) = x(t - d)$ then $S_{xy}(f) = |X(f)|^2 e^{-i 2\pi f d}$, so
    $\big|\sum_f S_{xy}(f)\,e^{i 2\pi f d'}\big|$ peaks at $d' = d$. The search maximises that over
    a grid of one raw sample.

    **The answer is a principal value.** The delay is identifiable only modulo
    :func:`delay_alias_period`, so the search runs over $[-P/2, +P/2]$ and a wider request is
    *refused* rather than served: outside that interval the grid holds exact ties, and which one an
    ``argmax`` returns is an implementation detail rather than a measurement. At the shipped
    geometry the interval is $\pm 64$ s, which is far wider than a horizon-long forecast can
    plausibly be mis-timed by and still be recognisable as the same event.

    **A search rather than a fit to the unwrapped phase.** Not for range -- unwrapping has the same
    alias limit -- but because unwrapping needs the phase to advance by less than $\pi$ between
    adjacent bins, and a single low-coherence bin where it does not shifts every bin above it. The
    search is magnitude-weighted instead, so the bins carrying cross-power decide, and it returns a
    concentration beside the delay so a meaningless answer says so.

    Args:
        sxy: Accumulated cross-spectrum over the bins to use, $(\ldots, F)$, complex. Pass a band's
            bins to get that band's delay.
        frequencies: The matching frequencies in Hz, $(F,)$, ascending and uniformly spaced.
        max_seconds: Half-width of the search. Defaults to half the alias period, which is the
            widest meaningful value.
        step_seconds: Grid spacing.

    Returns:
        ``(delay_seconds, concentration)``, both $(\ldots,)$. **A positive delay means the forecast
        lags the truth.** ``concentration`` is the aligned magnitude over $\sum_f |S_{xy}|$, in
        $[0, 1]$: it is $1$ when every bin agrees on the delay and near $0$ when the phase is
        incoherent, so a delay reported beside a low concentration is a number the data does not
        support. Both are ``NaN`` where the cross-spectrum carries no magnitude, and where fewer
        than two bins were supplied -- one bin's phase constrains a delay only modulo $1/f$.

    Raises:
        ValueError: If ``max_seconds`` exceeds half the alias period.
    """
    xy = np.asarray(sxy, dtype=np.complex128)
    axis = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if axis.size < 2:
        shape = xy.shape[:-1]
        return np.full(shape, np.nan), np.full(shape, np.nan)

    limit = 0.5 * delay_alias_period(axis)
    if max_seconds is None:
        max_seconds = limit
    elif float(max_seconds) > limit * (1.0 + 1e-9):
        raise ValueError(
            f"a delay search half-width of {float(max_seconds):g} s exceeds the alias limit of "
            f"{limit:g} s set by the bin spacing ({1.0 / delay_alias_period(axis):g} Hz). Beyond "
            f"it the grid holds exact ties -- d and d + {delay_alias_period(axis):g} s rotate "
            f"every bin identically -- so the reported delay would be whichever tie argmax "
            f"happened to reach first rather than a measurement"
        )
    grid = np.arange(
        -float(max_seconds), float(max_seconds) + 0.5 * float(step_seconds), float(step_seconds)
    )
    # (D, F) rotation matrix; D is a few hundred and F at most a few hundred, so the dense form is
    # cheaper and far plainer than an iterative search.
    rotation = np.exp(2j * np.pi * grid[:, None] * axis[None, :])
    scores = np.abs(np.tensordot(xy, rotation, axes=([-1], [1])))  # (..., D)
    total = np.abs(xy).sum(axis=-1)
    best = np.argmax(scores, axis=-1)
    aligned = np.take_along_axis(scores, best[..., None], axis=-1)[..., 0]

    usable = total > 0.0
    delay = np.where(usable, grid[best], np.nan)
    concentration = ratio_of_sums(aligned, total)
    return delay, concentration


def seam_bins(nperseg: int, raw_per_step: int) -> np.ndarray:
    r"""Bin indices of the token-seam frequency and its harmonics.

    The seam period is $R$ raw samples, so the fundamental is $f_s/R$ and harmonic $k$ sits at bin
    $k \cdot \mathrm{nperseg}/R$ -- an integer exactly when ``nperseg`` is a multiple of $R$, which
    :func:`slice_geometry` guarantees.

    Args:
        nperseg: Window length in raw samples.
        raw_per_step: Raw samples per horizon token $R$.

    Returns:
        The bins inside the one-sided spectrum, ascending. Harmonics beyond Nyquist are dropped.
    """
    stride = int(nperseg) // int(raw_per_step)
    n_freq = int(nperseg) // 2 + 1
    bins = np.array([k * stride for k in SEAM_HARMONICS], dtype=np.int64)
    return bins[bins < n_freq]


def seam_ratio(
    power: Any,
    bins: Any,
    *,
    neighbourhood: Tuple[int, int] = SEAM_NEIGHBOURHOOD_BINS,
) -> np.ndarray:
    r"""Power at each seam bin over the median of its neighbourhood.

    A ratio near $1$ means the seam frequency is unremarkable; a ratio far above it means power is
    concentrated there. The truth cannot produce one -- the raw trace knows nothing about the
    decoder's token boundary -- so the truth's own ratio is the control, and reading a branch's
    ratio without it would confuse an artifact with whatever the FHR happens to do at $0.25$ Hz.

    The median rather than the mean: a neighbourhood that happens to contain another harmonic
    would drag a mean upward and mask the very peak being measured.

    Args:
        power: An auto-spectrum, $(\ldots, F)$.
        bins: Seam bin indices, $(S,)$.
        neighbourhood: Inclusive offset range on each side, excluding offsets below the first so
            the Hann mainlobe does not contaminate its own reference.

    Returns:
        $(\ldots, S)$. ``NaN`` where the neighbourhood is empty or its median is not positive.
    """
    array = np.asarray(power, dtype=np.float64)
    n_freq = array.shape[-1]
    seams = np.asarray(bins, dtype=np.int64).reshape(-1)
    near, far = int(neighbourhood[0]), int(neighbourhood[1])
    seam_set = set(int(value) for value in seams.tolist())

    out = np.full(array.shape[:-1] + (seams.size,), np.nan, dtype=np.float64)
    for position, bin_index in enumerate(seams.tolist()):
        offsets = [offset for offset in range(near, far + 1)]
        candidates = [bin_index - offset for offset in offsets] + [
            bin_index + offset for offset in offsets
        ]
        # Drop neighbours that leave the spectrum or that are themselves seams, so a harmonic never
        # serves as the reference for another.
        reference = [
            index for index in candidates if 0 <= index < n_freq and index not in seam_set
        ]
        if not reference:
            continue
        median = np.median(array[..., reference], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[..., position] = np.where(median > 0.0, array[..., bin_index] / median, np.nan)
    return out


def ratio_of_sums(numerator: Any, denominator: Any) -> np.ndarray:
    """Divide two accumulators, yielding ``NaN`` rather than an infinity on an empty denominator.

    Args:
        numerator: Any accumulated quantity.
        denominator: Its denominator, of a broadcastable shape.

    Returns:
        The ratio, ``NaN`` wherever the denominator is not strictly positive. Never ``inf`` (which
        a finiteness check downstream would refuse) and never ``0.0`` (which reads as a measured
        zero rather than as nothing measured).
    """
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bottom > 0.0, top / np.where(bottom > 0.0, bottom, np.nan), np.nan)


def mean_over_horizon(values: Any) -> np.ndarray:
    r"""Average a $(\ldots, H, B)$ readout over the horizon axis, skipping unmeasured slices.

    A $\tau$-pooled number is the **mean over $\tau$ of the per-$\tau$ ratio**, never a ratio of
    $\tau$-summed spectra. The two differ and only the first is meaningful: $\phi$ depends on
    $\tau$, so a coherent sum of $S_{xy}$ across lead times cancels, and a forecast whose timing
    error grows with lead time would report as one with no timing error at all.

    Args:
        values: $(\ldots, H, B)$, possibly holding ``NaN`` where a slice measured nothing.

    Returns:
        $(\ldots, B)$.
    """
    array = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanmean(array, axis=-2)
