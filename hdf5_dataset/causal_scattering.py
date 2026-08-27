r"""One-sided (causal) scattering and phase harmonics, and an explicit two-sided reference.

Why this module exists
----------------------
The features stored by ``hdf5_dataset/new_pipeline/create_new_pipeline.py`` are **two-sided**
wavelet transforms. The value of a scattering or phase-harmonic channel at decimated step $t$ is a
weighted average over raw samples on *both* sides of $t$, so a model told it is conditioning on
"the past up to $t$" is in fact reading part of the interval it is being asked to forecast.
:mod:`teb_vae.lag_attn.channel_reach` measures that violation per channel as the forward reach
$L_{95}$ -- the slowest ``fhr_st`` channel reads $965$ s into its own future -- and
``teb_vae/lag_attn_rws/DESIGN.md`` S7 records both the current mitigation (drop channels past a
budget, read the survivors stale) and its limit: *"The guard bounds the leak; it does not remove
it... Only genuinely causal transforms remove the leak."*

This module builds those genuinely causal transforms, plus an explicit two-sided implementation of
the same chain so the causal effect can be separated from any reimplementation difference:

=====  ======================================================  ==============================
Arm    What it is                                              What it isolates
=====  ======================================================  ==============================
A      ``KymatioPhaseScattering1D`` as shipped                  what the dataset contains
B      this module's chain on the production Morlet bank        validates this code against A
C      this module's chain on :func:`build_causal_bank`         the effect under study
=====  ======================================================  ==============================

A-vs-B is the correctness gate; B-vs-C is the comparison. Without arm B every B-vs-C difference
would be confounded with "this file differs from kymatio's".

What causality actually costs
-----------------------------
The finding this module exists to support is **not** "the causal transform leaks nothing" -- that
is true by construction and uninformative. It is that a causal filter buys the removal of a
forward leak by paying a backward delay, and the exchange rate is worse than one:

$$\tau_g = \frac{n}{2\pi b_k} \approx 1.5 \times L_{95}, \qquad \text{median over the bank.}$$

So relative to :func:`~teb_vae.lag_attn.channel_reach.resolve_channel_budget` at the shipped
$120$ s budget, the causal transform is *strictly staler* on every channel that budget already
keeps, and buys back only the channels the budget has to drop -- at delays of hundreds of seconds.
Feed the $\tau_g$ vector through the **same** ``resolve_channel_budget`` as the reach vector and
the two arms land in one table under one predicate; that comparison is the deliverable.

Two further structural costs, both measured per channel rather than argued away:

1. **Analyticity.** By Paley--Wiener a filter cannot be simultaneously exactly causal and exactly
   analytic. The phase-harmonic operator reads $\arg(x \star \psi)$, so the causal bank is exactly
   causal and only approximately analytic. :func:`response_summary` reports the defect.
2. **Warm-up.** A causal channel is influenced by the assumed pre-recording history until its
   finite support has passed. The implemented 95%-energy diagnostic measures the practically
   dominant part of that influence; the slowest values exceed the $1320$ s segment, so those
   channels are dominated by the pad for most of it. :func:`causal_support_samples` reports it.

Filter design decisions that are correctness, not taste
--------------------------------------------------------
* **Bandwidth is matched at half power on both sides.** A Gaussian $e^{-\nu^2/2\sigma^2}$ reaches
  $|H| = 1/\sqrt{2}$ at $\nu = \sigma\sqrt{\ln 2}$ (**not** $\sigma\sqrt{2\ln 2}$, which is the
  half-*amplitude* point); a gamma envelope of order $n$ and rate $b$ reaches it at
  $b\sqrt{2^{1/n}-1}$. Mixing the two conventions makes the causal bank $41\%$ too wide and its
  delay $29.3\%$ too short; equivalently, the correct delay is $41.4\%$ longer -- flattering the
  causal arm on exactly the axis under study.
* **The causal wavelet is zero-mean.** An uncorrected $t^{n-1}e^{-2\pi b t}e^{i2\pi\xi t}u(t)$ has
  DC gain $(1 + \xi^2/b^2)^{-n/2}$ relative to its peak: $0.14\%$ on the constant-$Q$ ladder, but
  $11.9\%$ on the slowest filter, where kymatio pins $\sigma = \sigma_{\min}$ and $\xi/\sigma$
  falls to $2.64$. Against a $140$ bpm FHR baseline that swamps everything the channel is for.
  The fix is the one ``morlet_1d`` already uses: subtract a scaled copy of the same envelope so
  $\hat\psi(0) = 0$ exactly, which stays strictly causal.
* **Normalisation is kymatio's** ``normalize='l1'``: unit $L^1$ norm in time
  (``kymatio/scattering1d/filter_bank.py::get_normalizing_factor``). For an analytic filter with a
  non-negative envelope this makes the peak gain exactly $1$, so a dB overlay of the two banks
  compares stopbands directly. It also lets ``SCATTERING_PHASE_HARMONIC_MATH_Complete.md`` S19.3's
  argument -- a fixed per-channel affine factor is absorbed by the pipeline's ``asinh``/``log``
  statistics -- carry over to the causal bank unchanged.
* **Decimation is plain subsampling.** kymatio periodises the spectrum; mean-periodisation
  followed by an $M/d$-point inverse DFT is *exactly* time subsampling, and $1456/16 = 91$ lands
  the decimated grid on raw index $0$. Subsampling is used here because taking every $16$-th
  sample is visibly causal whereas a spectral fold is not.

The one deliberate deviation, and how it is pinned
---------------------------------------------------
``SCATTERING_PHASE_HARMONIC_MATH_Complete.md`` S15.3 documents that production's
``_apply_phi_filter`` **truncates** the spectrum to the first $M/d$ bins instead of periodising it,
so the stored phase block is $d$ times the *analytic projection* of the smoothed product rather
than $\Re\{(\cdot)\star\phi\}$. :func:`phase_block` therefore takes a ``phi_mode``:
``'kymatio_truncate'`` reproduces that operator step for step, so arm B can validate against arm A
on the phase blocks too, and ``'exact'`` applies the documented definition. The ratio between the
two modes of the *same* code is the measured S15.3 deviation -- an exact per-channel number rather
than an inference. On the diagonal ($i = j$, product real) it must land in $[d/2, d] = [8, 16]$.

What this module is for now
---------------------------
Its chain functions are numpy and take one segment per call, at $\approx 1.5$ s a segment. That is
deliberate and no longer a limitation: :mod:`hdf5_dataset.causal_scattering_torch` builds datasets,
and this is the **reference it is gated against**, plus the filter design both share -- one
definition, so the two cannot drift. Everything a dataset build needs from here (the bank, the
channel plan, the geometry) is torch-free and batch-free by design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.scattering1d.utils import compute_minimum_support_to_pad

#: Production geometry, as ``hdf5_dataset/new_pipeline/create_new_pipeline.py`` sets it: raw
#: sampling rate in Hz, the wavelet bank's octaves / wavelets-per-octave / low-pass width, the
#: stored segment length in raw samples, and the raw-to-decimated factor (one step is $4$ s).
FS = 4.0
J, Q, T = 11, 4, 16
N_RAW = 5280
DECIMATION = 16

#: Band edges $(f_{\min}, f_{\max})$ in Hz of the two stored phase-harmonic selections:
#: :data:`TARGET_PHASE_BAND_HZ` selects ``fhr_ph`` and :data:`SOURCE_PHASE_BAND_HZ` ``up_ph``.
#: The lower edge is the analysis-window floor -- below it a wavelet's envelope outruns the
#: trimmed segment and the coefficient measures padding rather than signal.
TARGET_PHASE_BAND_HZ = (0.008, 1.00)
SOURCE_PHASE_BAND_HZ = (0.008, 0.05)

#: Order $n$ of the gamma envelope $t^{\,n-1}e^{-2\pi b t}$. Four is the standard gammatone order:
#: concentrated enough to be a usable band-pass, low enough that $\tau_g = n/(2\pi b)$ stays finite.
GAMMATONE_ORDER = 4

#: Fraction of $\tau_g$ a channel's content actually sits at: $1 - 1/(2\gamma) = 0.875$ at
#: :data:`GAMMATONE_ORDER`. :attr:`CausalBank.group_delay_samples` ships the phase group delay
#: $\tau_g = \gamma/(2\pi b)$, the envelope's *mean*, which is the right number to REPORT as a
#: channel's staleness. It is not the right number to ALIGN on: the realised lag is the
#: spectrum-weighted average group delay, equivalently the impulse response's energy centroid
#: $(2\gamma-1)/(4\pi b)$, because $\tau_g(\nu) = \tau_g(\xi)\,b^2/(b^2+(\nu-\xi)^2)$ is *maximal*
#: at the centre frequency and a channel's own passband can therefore only pull the realised lag
#: down. The spread is one-sided and does not average out. Measured over 30 segments of the aligned
#: shard, causal ``fhr_st`` against the centred block: median realised/reported $0.903$ over all 30
#: resolved channels, $0.882$ over the nine slow ones where the $4$ s grid quantises by under
#: $2.5\%$ -- against $0.875$ predicted here and $1.000$ predicted by $\tau_g$.
ALIGNMENT_DELAY_FACTOR = 1.0 - 1.0 / (2.0 * GAMMATONE_ORDER)

#: Causal kernel length in taps. Measured requirement, not a guess: at the corrected rate
#: $b = 1.914\,\sigma$ the slowest kernel retains only $70.7\%$ of its $L^1$ mass inside $2^{13}$
#: taps and $98.6\%$ inside $2^{14}$, against $99.999\%$ at $2^{15}$. Enlarging the kernel is
#: legitimate precisely *because* it is causal -- the extra taps reach further into the past, and a
#: causal filter never reads forward.
CAUSAL_KERNEL_TAPS = 1 << 15

#: Harmonic steps and tolerance of the stored phase selections, as the shard writer sets them.
PHASE_K_STEPS = (4, 6, 8)
PHASE_REL_TOL = 0.05

#: Energy fraction a causal kernel's leading taps must enclose for its output to be counted as a
#: function of the recording rather than of the assumed pre-recording history. Defined once and
#: used as :func:`causal_support_samples`'s default, because every published causal figure, the
#: per-channel measurement CSV and the stored ``causal_warmup_steps`` must all mean the same thing
#: by "warm-up". Raising it to $0.99$ lengthens every warm-up by $\approx 18\%$ and would put the
#: dataset at odds with all three.
CAUSAL_WARMUP_QUANTILE = 0.95

#: Half-power ($-3$ dB) half-width of a Gaussian of unit $\sigma$, and of a gamma envelope of unit
#: rate at :data:`GAMMATONE_ORDER`. Named because getting these two onto the same convention is the
#: single easiest way to build a causal bank that silently flatters itself; see the module
#: docstring.
GAUSSIAN_HALF_POWER = math.sqrt(math.log(2.0))
GAMMA_HALF_POWER = math.sqrt(2.0 ** (1.0 / GAMMATONE_ORDER) - 1.0)


# =================================================================================================
# Production geometry
# =================================================================================================
def production_padding(n_signal: int = N_RAW) -> Tuple[int, int, int]:
    r"""The padding chain ``KymatioPhaseScattering1D`` uses, reproduced for the two-sided arm.

    ``build_filter_bank`` runs this chain to size its FFT but does not expose the result, and arm B
    must place its signal in the buffer exactly where production does or its decimated grid will be
    offset from the shard's by a fraction of a step -- a shift nothing downstream would flag.

    Args:
        n_signal: Signal length in raw samples.

    Returns:
        ``(pad_left, pad_right, n_padded)``; $(1456, 1456, 8192)$ at the production geometry.
    """
    min_to_pad = min(compute_minimum_support_to_pad(n_signal, J, Q, T), n_signal - 1)
    j_max = int(np.floor(np.log2(3 * n_signal - 2)))
    j_pad = min(int(np.ceil(np.log2(n_signal + 2 * min_to_pad))), j_max)
    n_padded = 2 ** j_pad
    # kymatio splits the surplus with the extra sample on the left; compute_padding's own rule.
    pad_right = (n_padded - n_signal) // 2
    pad_left = n_padded - n_signal - pad_right
    return pad_left, pad_right, n_padded


@lru_cache(maxsize=4)
def _scattering_filters(n_padded: int) -> Tuple[Any, Any]:
    """kymatio's realised filters at a padded length -- the one call site for the factory.

    Every consumer of the production bank in this module goes through here: :func:`build_filter_bank`
    for the filters themselves, :func:`_production_filter_levels` for the intermediate decimation
    levels the two-sided cascade needs. Realised filter *shape* depends on the padded length, so a
    second factory call sized differently would silently produce a different bank.

    Note:
        The returned structures are kymatio's own and are cached, so callers must treat them as
        read-only.

    Args:
        n_padded: FFT length the filters are realised on; a power of two, from
            :func:`production_padding`.

    Returns:
        ``(phi_f, psi1_f)`` exactly as ``scattering_filter_factory`` returns them.
    """
    phi_f, psi1_f, _, _ = scattering_filter_factory(
        J_support=int(round(math.log2(n_padded))), J_scattering=J, Q=Q, T=T
    )
    return phi_f, psi1_f


@dataclass(frozen=True)
class FilterBank:
    """The production first-order filter bank, rebuilt.

    Attributes:
        psi: First-order filters in the frequency domain, ``(n_filters, n_padded)``.
        phi: The low-pass filter in the frequency domain, ``(n_padded,)``.
        xi: Centre frequencies in cycles per sample (kymatio's unit), descending.
        sigma: Frequency widths in cycles per sample.
        taps: Filter-tap times in seconds, centred at $0$; negative is the past.
    """

    psi: np.ndarray
    phi: np.ndarray
    xi: np.ndarray
    sigma: np.ndarray
    taps: np.ndarray

    @property
    def hz(self) -> np.ndarray:
        """Centre frequencies in Hz."""
        return self.xi * FS

    @property
    def n_filters(self) -> int:
        """Number of first-order filters."""
        return int(self.psi.shape[0])


def build_filter_bank(n_signal: int = N_RAW) -> FilterBank:
    r"""Rebuild the bank exactly as ``KymatioPhaseScattering1D`` does.

    The causal bank is *matched* to this one -- centre frequency by centre frequency, bandwidth by
    bandwidth -- so this is the reference every measurement in this module is taken against. It
    keeps $\xi$ and $\sigma$, which is the whole reason it exists alongside
    :func:`_production_filter_levels`: the causal design needs the widths, the two-sided cascade
    needs the decimation levels, and neither carries the other's.

    Args:
        n_signal: Signal length in raw samples, which fixes the padded length and therefore the
            realised filter shapes.

    Returns:
        The populated :class:`FilterBank`.
    """
    _, _, n_padded = production_padding(n_signal)
    phi_f, psi1_f = _scattering_filters(n_padded)
    return FilterBank(
        psi=np.stack([spec["levels"][0] for spec in psi1_f], axis=0),
        phi=np.asarray(phi_f["levels"][0]),
        xi=np.array([spec["xi"] for spec in psi1_f]),
        sigma=np.array([spec["sigma"] for spec in psi1_f]),
        taps=two_sided_taps(n_padded) / FS,
    )


def select_phase_pairs(
    bank: FilterBank,
    f_min: float,
    f_max: float,
    k_steps: Tuple[int, ...] = PHASE_K_STEPS,
    rel_tol: float = PHASE_REL_TOL,
) -> list:
    r"""Rebuild a stored phase-harmonic channel selection from the documented rule.

    A pair $(i, j)$ with $\xi_i \le \xi_j$ is kept when both endpoints lie in the band and the
    ratio $p = \xi_j/\xi_i$ sits within a **relative** tolerance of some $2^{k/Q}$. Relative
    rather than absolute because the power grid is geometric. Reproducing the shipped counts
    ($66$ for ``fhr_ph``, $15$ for ``up_ph``) from the rule alone is what licenses treating
    channel $c$ here as channel $c$ on disk.

    Args:
        bank: The production filter bank.
        f_min: Lower band edge in Hz, applied to the slower wavelet $\xi_i$.
        f_max: Upper band edge in Hz, applied to the faster wavelet $\xi_j$.
        k_steps: Harmonic steps admitted.
        rel_tol: Relative tolerance on the $2^{k/Q}$ power grid.

    Returns:
        The kept ``(i, j)`` index pairs in stored channel order, ``i`` indexing the lower
        frequency.
    """
    hz = bank.hz
    n = bank.n_filters
    pairs = sorted({(a, b) if hz[a] <= hz[b] else (b, a)
                    for a in range(n) for b in range(a, n)})
    keep = []
    for i, j in pairs:
        if hz[i] < f_min or hz[j] > f_max:
            continue
        power = hz[j] / hz[i]
        if any(abs(power - 2 ** (k / Q)) < rel_tol * 2 ** (k / Q) for k in k_steps):
            keep.append((i, j))
    return keep


# =================================================================================================
# The causal bank
# =================================================================================================
@dataclass(frozen=True)
class CausalBank:
    r"""A strictly one-sided replacement for the production Morlet/Gaussian bank.

    Kernels are stored **causally indexed**: element $\tau$ is the weight on $x(t - \tau)$, so
    index $0$ is "now" and every element reads the past. There is no future half that could be
    accidentally retained, which is what lets the test assert one-sidedness *bitwise* rather than
    to a tolerance.

    Attributes:
        psi: ``(n_filters, n_taps)`` complex wavelet kernels, unit $L^1$ norm, zero mean.
        phi: ``(n_taps,)`` non-negative low-pass kernel summing to $1$, so its DC gain matches the
            production Gaussian $\phi$'s exactly and $S_0$ amplitudes are directly comparable.
        b: ``(n_filters,)`` gamma rate in cycles per sample.
        b_phi: Low-pass gamma rate in cycles per sample.
        xi: Centre frequencies in cycles per sample, descending -- the production bank's.
        sigma: The production bank's frequency widths, which ``b`` was matched to.
        order: Gamma order $n$.
        kind: ``'gammatone'``, ``'truncated_morlet'`` or ``'delayed_morlet'``.
    """

    psi: np.ndarray
    phi: np.ndarray
    b: np.ndarray
    b_phi: float
    xi: np.ndarray
    sigma: np.ndarray
    order: int
    kind: str

    @property
    def hz(self) -> np.ndarray:
        """Centre frequencies in Hz."""
        return self.xi * FS

    @property
    def n_filters(self) -> int:
        """Number of first-order filters."""
        return int(self.psi.shape[0])

    @property
    def n_taps(self) -> int:
        """Kernel length in samples."""
        return int(self.psi.shape[1])

    @property
    def group_delay_samples(self) -> np.ndarray:
        r"""Phase group delay $\tau_g = n/(2\pi b)$ at each centre frequency, in samples.

        This is the delay a cross-correlation against the two-sided arm recovers, and it is the
        envelope's *mean*. The envelope's *mode* is $(n-1)/(2\pi b)$ and its energy centroid is
        $(2n-1)/(4\pi b)$ -- three different numbers, $25\%$ apart at $n = 4$. Quoting the mode as
        "the group delay" understates the cost, so the phase definition is the one that ships.
        """
        return self.order / (2.0 * np.pi * self.b)

    @property
    def group_delay_s(self) -> np.ndarray:
        """Phase group delay in seconds."""
        return self.group_delay_samples / FS

    @property
    def phi_group_delay_s(self) -> float:
        """The low-pass filter's phase group delay in seconds."""
        return float(self.order / (2.0 * np.pi * self.b_phi) / FS)

    @property
    def label(self) -> str:
        """Short human-readable name for figure legends and CSV rows."""
        return {
            "gammatone": f"causal (gammatone n={self.order})",
            "truncated_morlet": "causal (truncated Morlet)",
            "delayed_morlet": "causal (delayed Morlet)",
        }[self.kind]


def gammatone_rate(sigma: np.ndarray | float, order: int = GAMMATONE_ORDER) -> np.ndarray | float:
    r"""Gamma rate $b$ whose $-3$ dB bandwidth matches a Gaussian of width $\sigma$.

    Both widths are taken at **half power** ($|H| = 1/\sqrt{2}$), which is what "$-3$ dB" means:

    $$e^{-\nu^2/2\sigma^2} = \tfrac{1}{\sqrt2} \;\Rightarrow\; \nu = \sigma\sqrt{\ln 2},
      \qquad
      \big(1 + (\nu/b)^2\big)^{-n/2} = \tfrac{1}{\sqrt2} \;\Rightarrow\; \nu = b\sqrt{2^{1/n}-1},$$

    so $b = \sigma\sqrt{\ln 2}\,/\,\sqrt{2^{1/n}-1}$, which is $1.9140\,\sigma$ at $n = 4$.

    The trap this docstring exists to close: $\sigma\sqrt{2\ln 2}$ is the Gaussian's half-*amplitude*
    half-width, and pairing it with the gamma's half-*power* width gives $b = 2.707\,\sigma$ -- a
    bank $41\%$ too wide whose group delay is $29.3\%$ too short. Equivalently, the correct delay
    is $41.4\%$ longer. Bandwidth is what is matched, and the delay is then whatever the uncertainty
    principle charges for it.

    Args:
        sigma: Gaussian width(s) in cycles per sample.
        order: Gamma order $n$.

    Returns:
        The matching rate(s) $b$, same shape as *sigma*.
    """
    return sigma * GAUSSIAN_HALF_POWER / math.sqrt(2.0 ** (1.0 / order) - 1.0)


def _gamma_envelope(taps: np.ndarray, order: int, rate: float) -> np.ndarray:
    r"""Causal gamma envelope $t^{\,n-1}e^{-2\pi b t}\,\mathbb{1}_{t>0}$, peak-normalised.

    Evaluated in logs and shifted by its own maximum before exponentiating. The direct product
    would compute $t^{n-1}$ up to $\approx 3.5\times10^{13}$ at $t = 2^{15}$ before the exponential
    pulls it back; in log space the two terms cancel at every tap, so the form is stable for any
    order and any rate.

    Args:
        taps: Tap times in samples, index $0$ being $t = 0$.
        order: Gamma order $n$.
        rate: Decay rate $b$ in cycles per sample.

    Returns:
        The envelope, exactly $0$ at $t = 0$ for $n \ge 2$.
    """
    envelope = np.zeros(taps.shape, dtype=np.float64)
    positive = taps > 0
    log_envelope = (order - 1) * np.log(taps[positive]) - 2.0 * np.pi * rate * taps[positive]
    envelope[positive] = np.exp(log_envelope - log_envelope.max())
    return envelope


def _zero_mean(kernel: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    r"""Subtract a scaled copy of the envelope so the kernel has exactly zero mean.

    The discrete analogue of ``morlet_1d``'s ``kappa = gabor_f[0] / low_pass_f[0]``:

    $$\psi(t) \leftarrow \psi(t) - \kappa\, a(t), \qquad
      \kappa = \frac{\sum_t \psi(t)}{\sum_t a(t)},$$

    which sets $\hat\psi(0) = 0$ exactly while leaving the kernel strictly causal, because the
    correction term shares the kernel's own support. Without it the slowest filters pass $11.9\%$
    of DC relative to their passband peak, and an FHR baseline near $140$ bpm would dominate the
    channel entirely.

    Args:
        kernel: The uncorrected kernel.
        envelope: The non-negative envelope to subtract a multiple of; same support as *kernel*.

    Returns:
        The zero-mean kernel.
    """
    return kernel - (kernel.sum() / envelope.sum()) * envelope


def _l1_normalise(kernel: np.ndarray) -> np.ndarray:
    r"""Scale a kernel to unit $L^1$ norm -- kymatio's ``normalize='l1'``.

    Args:
        kernel: Time-domain taps.

    Returns:
        The kernel scaled so $\sum_t |h(t)| = 1$.

    Raises:
        ValueError: If the kernel is numerically zero, which would otherwise emit a silent NaN
            filter. In practice this means the tap grid is far too short for the time constant.
    """
    total = np.abs(kernel).sum()
    if total < 1e-300:
        raise ValueError(
            "kernel has ~zero L1 norm; the tap grid is almost certainly too short for this "
            "filter's time constant"
        )
    return kernel / total


def build_causal_bank(
    reference: Optional[FilterBank] = None,
    *,
    order: int = GAMMATONE_ORDER,
    n_taps: int = CAUSAL_KERNEL_TAPS,
) -> CausalBank:
    r"""A strictly causal complex-gammatone bank matched to the production Morlets.

    Each filter is built **in the time domain**, which is what makes causality exact rather than
    asymptotic:

    $$\psi^{c}_k(t) = a_k(t)\big(e^{\,i 2\pi \xi_k t} - \kappa_k\big),
      \qquad a_k(t) = t^{\,n-1} e^{-2\pi b_k t}\,\mathbb{1}_{t>0},$$

    with $\xi_k$ taken from the production bank, $b_k$ from :func:`gammatone_rate` so the $-3$ dB
    bandwidths agree, and $\kappa_k$ from :func:`_zero_mean` so $\hat\psi(0) = 0$. Centre frequency
    and bandwidth therefore match the Morlet each filter replaces; the group delay
    $\tau_g = n/(2\pi b_k)$ is the price, and it is $\approx 1.5\times$ the forward reach removed.

    The low-pass is the same envelope matched against $\sigma_{\text{low}} = \sigma_0/T$, kept
    non-negative and normalised to sum $1$ so its DC gain equals the production Gaussian's.

    Args:
        reference: Production bank to match. Defaults to :func:`build_filter_bank`.
        order: Gamma order $n$.
        n_taps: Kernel length. The default is measured, not chosen -- see
            :data:`CAUSAL_KERNEL_TAPS`.

    Returns:
        The populated :class:`CausalBank`.
    """
    reference = reference if reference is not None else build_filter_bank()
    taps = np.arange(n_taps, dtype=np.float64)
    rate = np.asarray(gammatone_rate(reference.sigma, order), dtype=np.float64)

    kernels = np.zeros((reference.n_filters, n_taps), dtype=np.complex128)
    for index in range(reference.n_filters):
        envelope = _gamma_envelope(taps, order, float(rate[index]))
        gabor = envelope * np.exp(2j * np.pi * reference.xi[index] * taps)
        kernels[index] = _l1_normalise(_zero_mean(gabor, envelope))

    # sigma_low is read off the production phi rather than restated as 0.1/T, so a change to
    # kymatio's sigma0 cannot leave this bank quietly matched to the wrong low-pass.
    sigma_low = _gaussian_sigma_of(reference.phi)
    b_phi = float(gammatone_rate(sigma_low, order))
    phi_kernel = _gamma_envelope(taps, order, b_phi)
    phi_kernel = phi_kernel / phi_kernel.sum()

    return CausalBank(
        psi=kernels,
        phi=phi_kernel,
        b=rate,
        b_phi=b_phi,
        xi=reference.xi.copy(),
        sigma=reference.sigma.copy(),
        order=order,
        kind="gammatone",
    )


def build_truncated_morlet_bank(
    reference: Optional[FilterBank] = None,
    *,
    n_taps: int = CAUSAL_KERNEL_TAPS,
    delay_s: Optional[np.ndarray] = None,
) -> CausalBank:
    r"""The Morlet made causal by cutting it, in two variants -- both contrast arms, not candidates.

    With ``delay_s=None`` this is the **naive** variant: keep the Morlet's past half ($t \le 0$) and
    drop the rest. The cut lands exactly on the envelope's peak, so the kernel has a step
    discontinuity at $\tau = 0$ whose spectrum is a sinc -- broad sidelobes at roughly $-13$ dB, so
    the channel reports energy from bands it does not name. **No taper can fix this**: the
    discontinuity is at the maximum, so any window that smooths it also deletes the filter. That is
    the point of including the arm.

    With ``delay_s`` given this is the **honest** variant: keep taps out to $-D$ and re-index so the
    kernel starts at delay $0$, i.e. a legitimate causal FIR that reproduces the Morlet exactly,
    delayed by $D$. Setting $D = L_{95}$ shows that keeping Morlet selectivity causally costs the
    same order of delay the gammatone pays -- which reframes the whole comparison from filter shape
    to delay, where it belongs.

    Both variants re-apply the zero-mean correction and the $L^1$ normalisation afterwards, because
    truncation destroys both.

    Args:
        reference: Production bank to cut. Defaults to :func:`build_filter_bank`.
        n_taps: Kernel length.
        delay_s: Per-filter delay in seconds for the delayed variant, ``(n_filters,)``.

    Returns:
        The populated :class:`CausalBank`, ``kind`` reflecting the variant.
    """
    reference = reference if reference is not None else build_filter_bank()
    n_reference = int(reference.phi.size)
    index = np.arange(n_reference)
    # kymatio lays filters out centred at tap 0 and wrapping; negative taps are the past.
    reference_taps = np.where(index <= n_reference // 2, index, index - n_reference)

    kernels = np.zeros((reference.n_filters, n_taps), dtype=np.complex128)
    for k in range(reference.n_filters):
        morlet = np.fft.ifft(reference.psi[k])
        # Shift the Morlet right by D samples so its centre sits at delay D, then keep only the
        # taps that have become non-negative delays. D = 0 recovers the naive cut at the peak.
        shift = 0 if delay_s is None else int(round(float(delay_s[k]) * FS))
        delay_index = shift - reference_taps
        keep = (delay_index >= 0) & (delay_index < n_taps)
        kernels[k, delay_index[keep]] = morlet[keep]
        envelope = np.abs(kernels[k])
        if envelope.sum() == 0:
            raise ValueError(f"truncation left filter {k} empty; n_taps={n_taps} is too short")
        kernels[k] = _l1_normalise(_zero_mean(kernels[k], envelope))

    phi_time = np.fft.ifft(reference.phi).real
    phi_kernel = np.zeros(n_taps, dtype=np.float64)
    keep = reference_taps <= 0
    delay_index = (-reference_taps[keep]).astype(int)
    inside = delay_index < n_taps
    phi_kernel[delay_index[inside]] = phi_time[keep][inside]
    phi_kernel = np.abs(phi_kernel)
    phi_kernel = phi_kernel / phi_kernel.sum()

    sigma_low = _gaussian_sigma_of(reference.phi)
    return CausalBank(
        psi=kernels,
        phi=phi_kernel,
        b=np.asarray(gammatone_rate(reference.sigma, GAMMATONE_ORDER), dtype=np.float64),
        b_phi=float(gammatone_rate(sigma_low, GAMMATONE_ORDER)),
        xi=reference.xi.copy(),
        sigma=reference.sigma.copy(),
        order=GAMMATONE_ORDER,
        kind="truncated_morlet" if delay_s is None else "delayed_morlet",
    )


# =================================================================================================
# Filter measurements
# =================================================================================================
def half_power_half_width(spectrum: np.ndarray) -> float:
    r"""Measured $-3$ dB half-width of a filter, in cycles per sample.

    One implementation is used for both banks, so "matched bandwidth" is a measurement rather than
    an assertion.

    The crossing is found by **linear interpolation between adjacent bins**, not by zero-padding
    the kernel in time. Zero-padding would be the obvious way to refine the estimate and it is
    wrong here: a two-sided kernel is stored wrapped around tap $0$, so appending zeros relocates
    its entire negative-time half to positive time and destroys the filter. Interpolating the
    magnitude spectrum in place is immune to that and is accurate enough -- at the slow end the
    half-width still spans several bins, and the residual error is well inside the few-percent
    tolerance the bandwidth-match test uses.

    Args:
        spectrum: Filter in the frequency domain, ``(n,)``.

    Returns:
        The half-width in cycles per sample; ``nan`` if the crossing is not bracketed.
    """
    magnitude = np.abs(spectrum)
    order = np.argsort(np.fft.fftfreq(magnitude.size))
    frequencies = np.fft.fftfreq(magnitude.size)[order]
    magnitude = magnitude[order]

    peak = int(np.argmax(magnitude))
    threshold = magnitude[peak] / math.sqrt(2.0)

    def crossing(step: int) -> Optional[float]:
        """Interpolated frequency where the passband first falls through the threshold."""
        index = peak
        # Walk outward from the peak: the passband is the contiguous run containing the maximum,
        # so a sidelobe that climbs back over the threshold cannot widen the answer.
        while 0 <= index + step < magnitude.size and magnitude[index + step] >= threshold:
            index += step
        outer = index + step
        if not 0 <= outer < magnitude.size:
            return None
        span = magnitude[index] - magnitude[outer]
        if span <= 0:
            return float(frequencies[index])
        weight = (magnitude[index] - threshold) / span
        return float(frequencies[index] + weight * (frequencies[outer] - frequencies[index]))

    low, high = crossing(-1), crossing(+1)
    if low is None or high is None:
        return float("nan")
    return float(abs(high - low) / 2.0)


def _gaussian_sigma_of(phi_spectrum: np.ndarray) -> float:
    r"""Recover the Gaussian width $\sigma$ of the production low-pass from its own spectrum.

    Read rather than restated as $\sigma_0/T$ so that a change to kymatio's $\sigma_0$ cannot leave
    the causal low-pass quietly matched to a filter production no longer uses.

    Args:
        phi_spectrum: The low-pass in the frequency domain.

    Returns:
        $\sigma$ in cycles per sample.
    """
    return half_power_half_width(phi_spectrum) / GAUSSIAN_HALF_POWER


def two_sided_taps(n_padded: int) -> np.ndarray:
    """Tap times of a kernel stored centred at tap $0$ and wrapping -- kymatio's layout.

    Args:
        n_padded: Kernel length.

    Returns:
        Taps in $[-n/2, n/2)$ samples; negative is the past, positive the future.
    """
    index = np.arange(n_padded)
    return np.where(index <= n_padded // 2, index, index - n_padded).astype(np.float64)


def embed_on_two_sided_axis(bank: CausalBank) -> Tuple[np.ndarray, np.ndarray]:
    r"""Lay a causal bank's kernels on a centred tap axis, so both banks can be measured alike.

    A :class:`CausalBank` stores kernels indexed by **delay** $\tau \ge 0$, which is what makes it
    structurally incapable of representing a future tap -- there is no storage for one. That is
    the strongest possible causality guarantee, but it also means the delay axis and kymatio's
    signed tap axis are different objects, and measuring "future energy" against the delay index
    would call every tap a future tap.

    This maps delay $\tau$ to time $t = -\tau$ and pads the future half with exact zeros, putting
    the causal kernels on the same axis as the production ones. The future half being **bitwise**
    zero is then a checkable assertion rather than a claim about the construction.

    Args:
        bank: The causal bank.

    Returns:
        ``(kernels, taps)`` -- ``(n_filters, 2 * n_taps)`` complex and the matching tap times.
    """
    n_taps = bank.n_taps
    width = 2 * n_taps
    embedded = np.zeros((bank.n_filters, width), dtype=np.complex128)
    taps = two_sided_taps(width)
    # Delay tau lands at time -tau; index -tau wraps to width - tau, which is exactly where
    # two_sided_taps puts negative times.
    embedded[:, 0] = bank.psi[:, 0]
    embedded[:, width - n_taps + 1 :] = bank.psi[:, 1:][:, ::-1]
    return embedded, taps


def future_energy_fraction(kernel: np.ndarray, taps: np.ndarray) -> float:
    r"""Fraction of a filter's energy at taps strictly after $t = 0$.

    The measure that replaces :func:`~teb_vae.lag_attn.eval.representation_capacity_probe.forward_reach`
    when a causal bank is involved. ``forward_reach`` returns the $95\%$ *quantile* of future-tap
    energy, and for a causal kernel that energy is pure round-off -- so the quantile of it is an
    arbitrary number in $[0, n/2]$ samples, not $0$. This fraction is $\approx 0.5$ for a two-sided
    filter and exactly $0$ for a causal one, and is therefore comparable across both.

    Args:
        kernel: Time-domain taps.
        taps: Tap times matching *kernel*; positive means the future.

    Returns:
        The energy fraction in $[0, 1]$.
    """
    energy = np.abs(kernel) ** 2
    return float(energy[taps > 0].sum() / energy.sum())


def causal_support_samples(
    kernel: np.ndarray, quantile: float = CAUSAL_WARMUP_QUANTILE
) -> int:
    r"""Leading taps of a causal kernel enclosing *quantile* of its energy -- the warm-up.

    A causal channel needs this much history before its output is a function of the signal rather
    than of the pad. It is the mirror of forward reach, and the column without which the causal arm
    looks free: the slowest kernels here are longer than the $1320$ s segment.

    Args:
        kernel: Time-domain taps of a causal kernel, index $0$ being $t = 0$.
        quantile: Energy fraction to enclose.

    Returns:
        The support in samples.
    """
    energy = np.abs(kernel) ** 2
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), quantile)) + 1


def response_summary(spectrum: np.ndarray, xi: float) -> Dict[str, float]:
    r"""Per-filter frequency-response facts, for the comparison CSV.

    Args:
        spectrum: Filter in the frequency domain.
        xi: Nominal centre frequency in cycles per sample.

    Returns:
        ``peak_gain``, ``bw3db_hz``, ``dc_gain_rel`` ($|\hat H(0)|/|\hat H|_{\max}$) and
        ``neg_freq_gain_rel`` ($|\hat H(-\xi)|/|\hat H|_{\max}$, the analyticity defect).
    """
    magnitude = np.abs(spectrum)
    peak = float(magnitude.max())
    negative_bin = int(round(-xi * spectrum.size)) % spectrum.size
    return {
        "peak_gain": peak,
        "bw3db_hz": half_power_half_width(spectrum) * 2.0 * FS,
        "dc_gain_rel": float(magnitude[0] / peak),
        "neg_freq_gain_rel": float(magnitude[negative_bin] / peak),
    }


# =================================================================================================
# Convolution
# =================================================================================================
def _prepend_history(x: np.ndarray, history: int, pad: str) -> np.ndarray:
    """Prepend *history* samples of assumed past to each row of a signal.

    The pad mode is a correctness question, not a preference. ``'edge'`` replicates the first
    sample, asserting only that the recording was locally constant before it started -- and a
    zero-mean $\\psi$ annihilates a constant exactly, so in the passband the assertion costs
    nothing. Reflection is deliberately **not** offered: it mirrors the signal *forward* in time
    and would reintroduce exactly the future dependence a causal kernel exists to remove.

    Args:
        x: ``(..., n_signal)``.
        history: Samples to prepend.
        pad: ``'edge'`` or ``'zero'``.

    Returns:
        ``(..., history + n_signal)``.

    Raises:
        ValueError: On an unknown pad mode.
    """
    if pad == "edge":
        prefix = np.repeat(x[..., :1], history, axis=-1)
    elif pad == "zero":
        prefix = np.zeros(x.shape[:-1] + (history,), dtype=x.dtype)
    else:
        raise ValueError(f"unknown pad mode {pad!r}; use 'edge' or 'zero'")
    return np.concatenate([prefix, x], axis=-1)


def causal_convolve(x: np.ndarray, kernels: np.ndarray, *, pad: str = "edge") -> np.ndarray:
    r"""Linear convolution of a signal with causally-indexed kernels.

    $$y_c(t) = \sum_{\tau \ge 0} h_c(\tau)\, x(t - \tau),$$

    so output $t$ reads input $\le t$ and nothing else. The history before $t = 0$ is supplied by
    the pad, and the pad mode is a correctness question, not a preference: ``'edge'`` replicates
    $x(0)$, asserting only that the recording was locally constant before it started -- and a
    zero-mean $\psi$ annihilates a constant exactly, so the assertion costs nothing in the
    passband. Reflection is **not** offered, because it mirrors the signal *forward* in time and
    would reintroduce exactly the future dependence these kernels exist to remove.

    Args:
        x: Signal, ``(n_signal,)``, real or complex.
        kernels: ``(n_kernels, n_taps)``, index $0$ being zero delay.
        pad: ``'edge'`` or ``'zero'``.

    Returns:
        ``(n_kernels, n_signal)``, complex.
    """
    n_signal = int(x.shape[-1])
    history = int(kernels.shape[-1]) - 1
    padded = _prepend_history(np.asarray(x), history, pad)

    # Size the FFT so wraparound is impossible, then take the segment whose every tap was supplied
    # by real signal or by the assumed history -- i.e. a true linear convolution, not a circular one.
    n_fft = 1 << int(padded.size + kernels.shape[-1] - 1).bit_length()
    spectrum = np.fft.fft(padded, n_fft)
    filtered = np.fft.ifft(spectrum[None, :] * np.fft.fft(kernels, n_fft, axis=-1), axis=-1)
    return filtered[:, history : history + n_signal]


def causal_smooth(u: np.ndarray, phi: np.ndarray, *, pad: str = "edge") -> np.ndarray:
    r"""Causally low-pass every row of a real or complex array with one $\phi$ kernel.

    The mirror of :func:`causal_convolve` -- many signals through one filter rather than one signal
    through many -- so the whole phase block is smoothed in a single batched FFT.

    Args:
        u: ``(n_rows, n_signal)``, real or complex.
        phi: ``(n_taps,)`` causal low-pass kernel.
        pad: ``'edge'`` or ``'zero'``.

    Returns:
        ``(n_rows, n_signal)``, complex.
    """
    rows = np.atleast_2d(u)
    n_signal = int(rows.shape[-1])
    history = int(phi.shape[-1]) - 1
    padded = _prepend_history(rows, history, pad)

    n_fft = 1 << int(padded.shape[-1] + phi.shape[-1] - 1).bit_length()
    spectrum = np.fft.fft(padded, n_fft, axis=-1) * np.fft.fft(phi, n_fft)[None, :]
    smoothed = np.fft.ifft(spectrum, axis=-1)[:, history : history + n_signal]
    return smoothed.reshape(u.shape) if u.ndim > 1 else smoothed[0]


def _reflect_index(index: np.ndarray, n: int) -> np.ndarray:
    """Fold arbitrary integer indices into ``[0, n)`` by mirror reflection at both ends.

    ``np.pad(mode='reflect')`` refuses widths beyond ``n - 1``; production's ``_reflect_pad``
    handles that by reflecting iteratively, and this is the same sequence written as an index map.

    Args:
        index: Indices, possibly negative or beyond ``n``.
        n: Signal length.

    Returns:
        Indices in ``[0, n)``.
    """
    if n == 1:
        return np.zeros_like(index)
    period = 2 * (n - 1)
    folded = np.abs(index) % period
    return np.where(folded < n, folded, period - folded)


def reflect_pad(x: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
    """Reflection padding matching production's ``border_mode='reflect'``.

    Args:
        x: Signal, ``(..., n_signal)``.
        pad_left: Samples to prepend.
        pad_right: Samples to append.

    Returns:
        ``(..., pad_left + n_signal + pad_right)``.
    """
    n = x.shape[-1]
    index = _reflect_index(np.arange(-pad_left, n + pad_right), n)
    return x[..., index]


def two_sided_responses(x: np.ndarray, bank: FilterBank) -> np.ndarray:
    r"""Wavelet responses $y_k = x \star \psi_k$ on the production geometry, unpadded.

    Reproduces ``KymatioPhaseScattering1D._apply_filters``: reflect-pad to $8192$, multiply by the
    filter spectra, invert, crop back to the signal.

    Args:
        x: Signal, ``(n_signal,)``.
        bank: The production filter bank.

    Returns:
        ``(n_filters, n_signal)``, complex.
    """
    n_signal = int(x.shape[-1])
    pad_left, pad_right, _ = production_padding(n_signal)
    spectrum = np.fft.fft(reflect_pad(np.asarray(x, dtype=np.float64), pad_left, pad_right))
    filtered = np.fft.ifft(spectrum[None, :] * bank.psi, axis=-1)
    return filtered[:, pad_left : pad_left + n_signal]


# =================================================================================================
# Leg alignment
# =================================================================================================
def pair_leg_skew(bank: CausalBank, pairs: np.ndarray) -> np.ndarray:
    r"""How far apart in time the two legs of each phase-harmonic pair report the signal.

    A phase-harmonic coefficient multiplies two wavelet responses **at the same index**, but the
    two come from filters of different centre frequency and therefore different group delay, so
    they describe two different physical instants. The gap is

    $$\Delta_{ij} \;=\; \tau_i - \tau_j
      \;=\; \frac{\gamma}{2\pi}\Bigl(\frac{1}{b_i} - \frac{1}{b_j}\Bigr)
      \;=\; \tau_i\Bigl(1 - \frac{1}{p_{ij}}\Bigr),
      \qquad p_{ij} = \frac{\xi_j}{\xi_i} \ge 1,$$

    the last equality holding on a constant-$Q$ ladder, where $b \propto \sigma \propto \xi$ gives
    $\tau_i/\tau_j = p_{ij}$. So the skew is $\tfrac12\tau_i$, $0.646\,\tau_i$ and $\tfrac34\tau_i$
    for the three stored harmonic families $p \in \{2,\,2^{3/2},\,4\}$: **it grows with the
    harmonic ratio, which is the parameter the block exists to sweep.**

    Column $0$ of *pairs* indexes the lower frequency, so $\xi_i \le \xi_j$, so $\tau_i \ge \tau_j$
    and the result is non-negative by construction; a negative entry means the pair list was built
    or reordered by something other than :func:`select_phase_pairs`.

    The low-pass delay $\tau_\phi$ is common to both legs and cancels, so this is the difference of
    the **wavelet** delays and the composed one would give the same number.

    Args:
        bank: The causal bank the pairs index into.
        pairs: ``(n_pairs, 2)`` of $(i, j)$, column $0$ indexing the lower frequency.

    Returns:
        ``(n_pairs,)`` skew in seconds, non-negative.
    """
    index = np.asarray(pairs, dtype=int).reshape(-1, 2)
    delay = bank.group_delay_s
    return delay[index[:, 0]] - delay[index[:, 1]]


def leg_alignment_shift(
    bank: CausalBank, pairs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""The per-pair delay and de-rotation that put both legs of a pair on one clock.

    Leg $i$ is the slower one, so the pair is brought onto one clock by **delaying leg $j$**, which
    is a pure delay of an already-causal response and is therefore exactly causal:

    $$\tilde y_j^{\,\mathrm{al}}[t] \;=\; y_j[t - s_{ij}]\;e^{\,i2\pi\xi_j s_{ij}},
      \qquad s_{ij} = \operatorname{round}\bigl(\Delta_{ij} f_s\bigr) \ge 0 .$$

    **The phasor is not optional, and dropping it is worse than doing nothing.** A gammatone
    delays its *envelope* by $\tau_g$ and its *carrier* by nothing -- the corrected kernel's
    response at its own centre frequency is real and positive, so the phase delay there is exactly
    zero (measured $\le 0.3^{\circ}$ on the shipped bank). A plain time shift therefore moves the
    carrier as well as the envelope and injects a spurious rotation of $2\pi\xi_j s_{ij}$, which at
    $\xi_j = 0.033$ Hz and $\Delta_{ij} = 291.6$ s is $9.6$ whole turns. Measured, the shift
    without the phasor makes the block *worse* than leaving it alone: median correlation against
    the centred block at the predicted delay moves from $+0.049$ to $-0.432$ on ``fhr_ph``. The
    two outputs are returned together so a caller cannot take one and forget the other.

    The shift is per **pair**, not per filter: one fast filter serves up to three slow partners at
    three different harmonic ratios and needs a different $s_{ij}$ in each, so it must be applied
    after a per-pair gather rather than to the response array.

    Args:
        bank: The causal bank the pairs index into.
        pairs: ``(n_pairs, 2)`` of $(i, j)$, column $0$ indexing the lower frequency.

    Returns:
        ``(shift, phasor)``: ``(n_pairs,)`` non-negative integer raw-sample shifts, and
        ``(n_pairs,)`` unit-modulus complex de-rotations to multiply the delayed leg by.

    Raises:
        ValueError: If any pair's skew is negative, which would mean advancing a leg -- reading
            its own future -- rather than delaying it.
    """
    index = np.asarray(pairs, dtype=int).reshape(-1, 2)
    skew = pair_leg_skew(bank, index)
    negative = np.flatnonzero(skew < 0.0)
    if negative.size:
        row = int(negative[0])
        raise ValueError(
            f"phase pair {row} = ({int(index[row, 0])}, {int(index[row, 1])}) has skew "
            f"{float(skew[row]):.4f} s < 0: column 0 must index the lower frequency, so the "
            f"faster leg would have to be advanced rather than delayed, reading its own future."
        )
    shift = np.round(skew * FS).astype(np.int64)
    phasor = np.exp(2j * np.pi * bank.xi[index[:, 1]] * shift)
    return shift, phasor


#: The leg-alignment modes the causal phase block ships. ``'none'`` multiplies the two legs at one
#: stored index, as every shard on disk was built; ``'envelope'`` puts them on one clock through
#: :func:`leg_alignment_shift`. There is deliberately no delay-without-phasor mode: it has no
#: legitimate caller and scores *worse* than no alignment at all, so it exists only as a negative
#: control built locally in the test that proves the phasor is doing the work.
LEG_ALIGNMENT_MODES = ("none", "envelope")


def resolve_leg_alignment(
    bank: CausalBank, pairs: np.ndarray, leg_alignment: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Turn a leg-alignment mode into the per-pair shift :func:`phase_products` takes.

    One resolution shared by the numpy chain and its torch twin, so the two implementations of
    the alignment cannot drift: the twin reimplements the *convolution*, never the filter design
    or the quantities derived from it.

    Args:
        bank: The causal bank the pairs index into.
        pairs: ``(n_pairs, 2)`` of $(i, j)$, column $0$ indexing the lower frequency.
        leg_alignment: One of :data:`LEG_ALIGNMENT_MODES`.

    Returns:
        ``None`` for ``'none'``, or ``(shift, phasor)`` for ``'envelope'``.

    Raises:
        ValueError: On an unknown mode, naming it and the two that ship.
    """
    if leg_alignment == "none":
        return None
    if leg_alignment == "envelope":
        return leg_alignment_shift(bank, pairs)
    raise ValueError(
        f"unknown leg_alignment {leg_alignment!r}; use 'none' or 'envelope'"
    )


# =================================================================================================
# The chain
# =================================================================================================
@lru_cache(maxsize=4)
def _production_filter_levels(n_padded: int) -> Tuple[Tuple[np.ndarray, ...], Tuple[Any, ...]]:
    r"""The production filters *with all their decimation levels*, which ``FilterBank`` drops.

    :class:`FilterBank` keeps only ``levels[0]`` of each filter, which is all the reach arithmetic
    needs. Reproducing kymatio's cascade needs $\hat\phi$ pre-folded to each intermediate rate
    (``levels[k]``) and each wavelet's dyadic index $j$, so those are fetched here rather than
    re-derived -- from the same :func:`_scattering_filters` call :func:`build_filter_bank` uses, so
    the two views cannot come from differently-sized banks.

    Args:
        n_padded: FFT length the filters are realised on.

    Returns:
        ``(phi_levels, psi_j)`` -- the low-pass at each fold level, and each wavelet's $j$.
    """
    phi_f, psi1_f = _scattering_filters(n_padded)
    return (
        tuple(np.asarray(level) for level in phi_f["levels"]),
        tuple(int(spec["j"]) for spec in psi1_f),
    )


def _subsample_fourier(spectrum: np.ndarray, factor: int) -> np.ndarray:
    r"""Decimate by periodising the spectrum -- kymatio's ``subsample_fourier``.

    $$\widehat{y_{\downarrow 2^{k}}}[m] = \frac{1}{2^{k}}\sum_{r=0}^{2^{k}-1}
        \hat y\big[m + r\,M/2^{k}\big].$$

    Args:
        spectrum: ``(..., n)`` frequency-domain signal.
        factor: Decimation factor, a divisor of ``n``.

    Returns:
        ``(..., n // factor)``.
    """
    if factor == 1:
        return spectrum
    shape = spectrum.shape[:-1] + (factor, spectrum.shape[-1] // factor)
    return spectrum.reshape(shape).mean(axis=-2)


def scattering_block_two_sided(
    x: np.ndarray,
    bank: FilterBank,
    *,
    decimation: int = DECIMATION,
    decimation_mode: str = "full_rate",
) -> np.ndarray:
    r"""Order-$0$ and order-$1$ scattering on the production bank -- arm B.

    $$S_0 = x \star \phi, \qquad S_1^{(k)} = |x \star \psi_k| \star \phi.$$

    Channel $0$ is $S_0$ and channels $1 \ldots 42$ are $S_1$ in bank order (descending $\xi$) --
    the layout ``fhr_st`` and ``up_st`` are stored in.

    Two decimation conventions are offered for the same reason ``phi_mode`` exists on the phase
    block: one validates this code against the shard, the other is the convention the causal arm
    can actually share.

    * ``'kymatio'`` reproduces production's cascade, which decimates each band by $2^{k_1}$,
      $k_1 = \min(j_1, \log_2 T)$, **before** taking the modulus, then smooths with $\hat\phi$
      pre-folded to that rate. Sampling commutes with the modulus, but the modulus of an already
      sampled band creates frequencies above the reduced Nyquist, so this convention aliases.
      Measured against ``'full_rate'`` on a real segment that costs up to $2.6\times10^{-2}$
      relative on the $k_1 = 4$ bands, while $S_0$ and the $k_1 \le 2$ bands agree to
      $\approx 3\times10^{-9}$.
    * ``'full_rate'`` keeps the whole cascade at full rate and subsamples once at the end. It is
      the more accurate of the two, and it is what the causal arm does -- so B-vs-C compares two
      transforms rather than two decimation schedules.

    Args:
        x: Signal, ``(n_signal,)``.
        bank: The production filter bank.
        decimation: Total subsampling factor; $16$ gives one frame per $4$ s.
        decimation_mode: ``'full_rate'`` or ``'kymatio'``.

    Returns:
        ``(1 + n_filters, n_signal // decimation)``, real.

    Raises:
        ValueError: On an unknown *decimation_mode*.
    """
    if decimation_mode not in ("full_rate", "kymatio"):
        raise ValueError(
            f"unknown decimation_mode {decimation_mode!r}; use 'full_rate' or 'kymatio'"
        )

    n_signal = int(x.shape[-1])
    pad_left, pad_right, n_padded = production_padding(n_signal)
    padded = reflect_pad(np.asarray(x, dtype=np.float64), pad_left, pad_right)
    spectrum = np.fft.fft(padded)

    if decimation_mode == "full_rate":
        order_zero = np.fft.ifft(spectrum * bank.phi).real
        envelopes = np.abs(np.fft.ifft(spectrum[None, :] * bank.psi, axis=-1))
        order_one = np.fft.ifft(np.fft.fft(envelopes, axis=-1) * bank.phi[None, :], axis=-1).real
        stacked = np.concatenate([order_zero[None, :], order_one], axis=0)
        return stacked[:, pad_left : pad_left + n_signal][:, ::decimation]

    # kymatio's cascade. Every band ends at the same total decimation log2(T), so one shared
    # unpad index pair applies -- start = pad_left // decimation, length = n_signal // decimation.
    phi_levels, psi_j = _production_filter_levels(n_padded)
    log2_t = int(round(math.log2(decimation)))
    start, length = pad_left // decimation, n_signal // decimation

    rows = [np.fft.ifft(_subsample_fourier(spectrum * bank.phi, decimation)).real]
    for index in range(bank.n_filters):
        k1 = max(min(psi_j[index], log2_t), 0)
        envelope = np.abs(np.fft.ifft(_subsample_fourier(spectrum * bank.psi[index], 2**k1)))
        remaining = max(log2_t - k1, 0)
        smoothed = _subsample_fourier(np.fft.fft(envelope) * phi_levels[k1], 2**remaining)
        rows.append(np.fft.ifft(smoothed).real)

    return np.stack(rows)[:, start : start + length]


def scattering_block_causal(
    x: np.ndarray, bank: CausalBank, *, decimation: int = DECIMATION, pad: str = "edge"
) -> np.ndarray:
    """Order-$0$ and order-$1$ scattering on a causal bank -- arm C.

    Structurally identical to :func:`scattering_block_two_sided`; only the filters and the padding
    differ, which is what makes arm B a valid control for arm C.

    Args:
        x: Signal, ``(n_signal,)``.
        bank: The causal bank.
        decimation: Subsampling factor.
        pad: History supplied before $t = 0$; ``'edge'`` or ``'zero'``.

    Returns:
        ``(1 + n_filters, n_signal // decimation)``, real.
    """
    signal = np.asarray(x, dtype=np.float64)
    order_zero = causal_smooth(signal[None, :], bank.phi, pad=pad).real
    envelopes = np.abs(causal_convolve(signal, bank.psi, pad=pad))
    order_one = causal_smooth(envelopes, bank.phi, pad=pad).real
    stacked = np.concatenate([order_zero, order_one], axis=0)
    return stacked[:, ::decimation]


def phase_products(
    responses_low: np.ndarray,
    responses_high: np.ndarray,
    pairs: np.ndarray,
    xi: np.ndarray,
    *,
    leg_shift: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    r"""The un-smoothed phase-harmonic products $[y_i]^{p_{ij}}\,\overline{y_j}$.

    $$[y]^p = |y|\,e^{\,i p \arg y}, \qquad p_{ij} = \xi_j/\xi_i \ge 1,$$

    magnitude to the **first** power, so $\big|[y_i]^p \overline{y_j}\big| = |y_i||y_j|$. Formed in
    polar coordinates, exactly as ``_accelerate_phase`` does, to avoid the branch cut a complex
    power would cross.

    Split out from the smoothing so the two ``phi_mode`` variants operate on one identical product
    -- which is what makes their ratio a measurement of the smoothing operator alone.

    Args:
        responses_low: ``(n_filters, n_signal)``, supplying the accelerated leg $y_i$.
        responses_high: ``(n_filters, n_signal)``, supplying the conjugated leg $y_j$.
        pairs: ``(n_pairs, 2)`` of $(i, j)$, $i$ indexing the lower frequency.
        xi: Centre frequencies in cycles per sample.
        leg_shift: ``None`` for the unaligned product, or the ``(shift, phasor)`` pair
            :func:`leg_alignment_shift` returns, which delays the conjugated leg onto the slow
            leg's clock. Applied **after** the per-pair gather below, which is the only correct
            place: one fast filter serves up to three slow partners at three harmonic ratios, so a
            shift applied to *responses_high* before the gather could satisfy one of them and
            would be silently wrong for the other two, with every shape correct. Leading samples
            are filled by replicating the response's first sample, which is the same assumed
            history the causal convolution ahead of it already ran on.

    Returns:
        ``(n_pairs, n_signal)``, complex.

    Raises:
        ValueError: If *leg_shift*'s two vectors do not carry one entry per pair.
    """
    index_low, index_high = pairs[:, 0], pairs[:, 1]
    power = (xi[index_high] / xi[index_low])[:, None]
    y_low = responses_low[index_low]
    accelerated = np.abs(y_low) * np.exp(1j * power * np.angle(y_low))

    y_high = responses_high[index_high]
    if leg_shift is not None:
        shift, phasor = leg_shift
        n_pairs = int(index_high.size)
        if shift.shape != (n_pairs,) or phasor.shape != (n_pairs,):
            raise ValueError(
                f"leg_shift carries {shift.shape} shifts and {phasor.shape} phasors for "
                f"{n_pairs} pairs. The shift is indexed by pair, not by filter, so a vector of "
                f"another length would be gathered against the wrong pairs."
            )
        # Clipping the source index at zero *is* the edge replication: every tap the shift pushes
        # before the start reads the response's first sample.
        taps = np.arange(y_high.shape[-1])
        source = np.maximum(taps[None, :] - shift[:, None], 0)
        y_high = np.take_along_axis(y_high, source, axis=-1) * phasor[:, None]
    return accelerated * np.conj(y_high)


def smooth_products_kymatio(
    products: np.ndarray, phi_spectrum: np.ndarray, *, decimation: int = DECIMATION
) -> np.ndarray:
    r"""Production's ``_apply_phi_filter``, reproduced step for step.

    Reflect-pad the length-$N$ complex product back to $M$, multiply by $\hat\phi$, **truncate** to
    the first $M/d$ bins, invert at that length, and slice ``[pad_left//d : +N//d]``. The truncation
    -- rather than the periodisation kymatio's own scattering path uses -- is the deviation
    ``SCATTERING_PHASE_HARMONIC_MATH_Complete.md`` S15.3 documents: the result is $d$ times the
    analytic (positive-frequency) projection of the smoothed product, not
    $\Re\{(\cdot) \star \phi\}$.

    Reproducing it here rather than describing it is what lets arm B validate against arm A on the
    phase blocks, and turns the deviation into a per-channel measured ratio against
    :func:`smooth_products_exact` on the *same* product.

    Args:
        products: ``(n_pairs, n_signal)`` complex, from :func:`phase_products`.
        phi_spectrum: The production low-pass in the frequency domain.
        decimation: Subsampling factor $d$.

    Returns:
        ``(n_pairs, n_signal // decimation)``, real.
    """
    n_signal = int(products.shape[-1])
    pad_left, pad_right, n_padded = production_padding(n_signal)
    padded = reflect_pad(products, pad_left, pad_right)
    smoothed = np.fft.fft(padded, axis=-1) * phi_spectrum[None, :]
    smoothed = np.fft.ifft(smoothed[:, : n_padded // decimation], axis=-1)
    start = pad_left // decimation
    return smoothed[:, start : start + n_signal // decimation].real


def smooth_products_exact(
    products: np.ndarray, phi_spectrum: np.ndarray, *, decimation: int = DECIMATION
) -> np.ndarray:
    r"""The documented operator $\Re\{(\cdot) \star \phi\}$, decimated by subsampling.

    Identical to :func:`smooth_products_kymatio` except that the spectrum is inverted at full
    length and subsampled in time rather than truncated in frequency. The two therefore differ
    only in the smoothing operator, which is the point.

    Args:
        products: ``(n_pairs, n_signal)`` complex.
        phi_spectrum: The production low-pass in the frequency domain.
        decimation: Subsampling factor.

    Returns:
        ``(n_pairs, n_signal // decimation)``, real.
    """
    n_signal = int(products.shape[-1])
    pad_left, pad_right, _ = production_padding(n_signal)
    padded = reflect_pad(products, pad_left, pad_right)
    smoothed = np.fft.ifft(np.fft.fft(padded, axis=-1) * phi_spectrum[None, :], axis=-1)
    return smoothed[:, pad_left : pad_left + n_signal][:, ::decimation].real


def phase_block_two_sided(
    x_low: np.ndarray,
    x_high: np.ndarray,
    pairs: np.ndarray,
    bank: FilterBank,
    *,
    phi_mode: str = "exact",
    decimation: int = DECIMATION,
) -> np.ndarray:
    r"""Phase-harmonic correlations on the production bank -- arm B.

    $$\Phi_{ij}(t) = \Re\Big\{\big([y_i]^{p_{ij}}\,\overline{y_j}\big) \star \phi\Big\}(t).$$

    Passing the two channels separately covers both stored cases with one function: ``x_low is
    x_high`` gives the self-phase blocks (``fhr_ph``, ``up_ph``), and distinct signals would give
    the cross-channel block.

    Args:
        x_low: Signal supplying the accelerated leg, ``(n_signal,)``.
        x_high: Signal supplying the conjugated leg, ``(n_signal,)``.
        pairs: ``(n_pairs, 2)`` of $(i, j)$.
        bank: The production filter bank.
        phi_mode: ``'exact'`` for the documented operator, ``'kymatio_truncate'`` to reproduce
            production's deviation.
        decimation: Subsampling factor.

    Returns:
        ``(n_pairs, n_signal // decimation)``, real.

    Raises:
        ValueError: On an unknown *phi_mode*.
    """
    if len(pairs) == 0:
        return np.zeros((0, int(x_low.shape[-1]) // decimation))

    responses_low = two_sided_responses(x_low, bank)
    responses_high = responses_low if x_high is x_low else two_sided_responses(x_high, bank)
    products = phase_products(responses_low, responses_high, pairs, bank.xi)

    if phi_mode == "exact":
        return smooth_products_exact(products, bank.phi, decimation=decimation)
    if phi_mode == "kymatio_truncate":
        return smooth_products_kymatio(products, bank.phi, decimation=decimation)
    raise ValueError(f"unknown phi_mode {phi_mode!r}; use 'exact' or 'kymatio_truncate'")


def phase_block_causal(
    x_low: np.ndarray,
    x_high: np.ndarray,
    pairs: np.ndarray,
    bank: CausalBank,
    *,
    decimation: int = DECIMATION,
    pad: str = "edge",
    leg_alignment: str = "none",
) -> np.ndarray:
    r"""Phase-harmonic correlations on a causal bank -- arm C.

    Only the documented smoothing operator is available. ``'kymatio_truncate'`` has no causal
    counterpart: keeping the positive-frequency half of a spectrum is a non-causal projection, so
    transplanting production's deviation onto a causal chain would silently reintroduce future
    dependence.

    **The default is ``'none'``, and that is a compatibility decision rather than a preference.**
    :func:`transform_sample` and the torch twin's batch entry point call this with no mode
    argument, and the committed causal fixture's blocks are rebuilt through that path and diffed
    against the stored bytes. A default of ``'envelope'`` would therefore turn a shared contract
    test red, and would let an ad hoc build produce aligned data carrying no attribute that says
    so. The writer passes the mode explicitly.

    Note:
        A pair's warm-up is set by its **slower** leg, and its delay compounds both legs plus the
        low-pass -- so a phase channel is staler than either of the scattering channels it is
        built from. This is why the comparison reports the phase blocks separately. Under
        ``'none'`` the composed *delay* is a prediction the block does not obey; see
        :func:`_compose_phase`.

    Args:
        x_low: Signal supplying the accelerated leg, ``(n_signal,)``.
        x_high: Signal supplying the conjugated leg, ``(n_signal,)``.
        pairs: ``(n_pairs, 2)`` of $(i, j)$.
        bank: The causal bank.
        decimation: Subsampling factor.
        pad: History supplied before $t = 0$.
        leg_alignment: ``'none'`` to multiply the legs at one stored index, as every shard on disk
            was built, or ``'envelope'`` to put them on one clock first. See
            :data:`LEG_ALIGNMENT_MODES`.

    Returns:
        ``(n_pairs, n_signal // decimation)``, real.

    Raises:
        ValueError: On an unknown *leg_alignment*.
    """
    # Resolved before the early return and before any convolution, so an unknown mode is refused
    # whatever the pair list holds and without paying for the chain first.
    leg_shift = resolve_leg_alignment(bank, pairs, leg_alignment)
    if len(pairs) == 0:
        return np.zeros((0, int(x_low.shape[-1]) // decimation))

    responses_low = causal_convolve(x_low, bank.psi, pad=pad)
    responses_high = responses_low if x_high is x_low else causal_convolve(x_high, bank.psi, pad=pad)
    products = phase_products(
        responses_low, responses_high, pairs, bank.xi, leg_shift=leg_shift
    )
    smoothed = causal_smooth(products, bank.phi, pad=pad)
    return smoothed.real[:, ::decimation]


# =================================================================================================
# The stored channel plan
# =================================================================================================
#: The four coefficient blocks a causal file stores, in the order the model concatenates them:
#: target stream first, then source. ``fhr_up_ph`` is absent because the causal variant does not
#: produce it.
CAUSAL_BLOCKS = ("fhr_st", "fhr_ph", "up_st", "up_ph")


@dataclass(frozen=True)
class CausalChannelPlan:
    r"""One stored block's surviving channels, with the warm-up and delay each one carries.

    The single source of a causal file's widths and per-channel provenance. Whatever asks -- the
    HDF5 writer sizing a dataset, the transform gathering rows, the loader rebasing validity for a
    trim, the operator log -- reads it here, so a width, a warm-up vector and a stored channel
    order cannot disagree with one another.

    It **composes** measurements rather than re-deriving them: warm-up comes from
    :func:`causal_support_samples` at :data:`CAUSAL_WARMUP_QUANTILE`, delay from
    :attr:`CausalBank.group_delay_s` and :attr:`CausalBank.phi_group_delay_s`. No new
    energy-quantile or $n/(2\pi b)$ arithmetic lives here.

    Attributes:
        name: Block name, carried so a refusal can say which block it is about.
        kept: Channel indices retained, ascending, indexing the block *before* the drop. A
            scattering block stores $S_0$ at channel $0$, so channel $c$ is filter $c - 1$ there;
            a phase block's channel $c$ is pair $c$.
        warmup_steps: Warm-up per kept channel in **decimated steps**, ceiling-rounded, in
            untrimmed coordinates -- the storage geometry every other stored field uses. A loader
            reading the file at a different trim rebases it itself.
        delay_s: Composed group delay per kept channel in seconds. Recorded, never compensated;
            see :func:`build_channel_plan`.
    """

    name: str
    kept: np.ndarray
    warmup_steps: np.ndarray
    delay_s: np.ndarray

    @property
    def n_channels(self) -> int:
        """Stored width of the block -- derived from the kept channels, never a literal."""
        return int(self.kept.size)


def _compose_scattering(
    per_filter: np.ndarray, low_pass: float
) -> np.ndarray:
    r"""Compose a per-filter quantity along the scattering cascade, $S_0$ first.

    $$q(S_0) = q_\phi, \qquad q(S_1^{(k)}) = q_k + q_\phi,$$

    supports and delays both **add** through a cascade, and every stored channel ends in the
    low-pass -- which is exactly why $S_0$ is not free either.

    Args:
        per_filter: The quantity per first-order filter, ``(n_filters,)``.
        low_pass: The same quantity for $\phi$.

    Returns:
        ``(1 + n_filters,)`` in stored channel order.
    """
    return np.concatenate([[low_pass], per_filter + low_pass])


def _compose_phase(
    per_filter: np.ndarray, low_pass: float, pairs: np.ndarray
) -> np.ndarray:
    r"""Compose a per-filter quantity along a phase-harmonic pair.

    $$q(\Phi_{ij}) = \max(q_i,\ q_j) + q_\phi .$$

    The **maximum** rather than the sum: the product $[y_i]^{p}\overline{y_j}$ is formed pointwise
    from two responses at the same $t$, so it is usable once the slower leg is, and is only then
    smoothed by $\phi$. Summing here would overstate every phase channel.

    **For the warm-up this is true under either leg alignment.** Delaying the fast leg by
    $s_{ij}$ lengthens its own warm-up to $W_j + s_{ij}$, and that never overtakes $W_i$ on any
    stored pair -- asserted rather than assumed, pair by pair, by
    ``tests/test_causal_torch.py::test_the_leg_alignment_costs_no_warm_up_on_any_stored_pair``,
    which reports the tightest slack in its failure message. So the composed warm-up, the stored
    widths and the drop rule are the same numbers either way.

    **For the delay it is true only under ``leg_alignment='envelope'``.** Under the unaligned
    operator $\max(\tau_i, \tau_j) + \tau_\phi$ is a *prediction the block does not obey*: the two
    legs report the signal at $t - \tau_i$ and $t - \tau_j$, so the product is a cross-scale phase
    correlation evaluated across the skew rather than a delayed copy of the centred coefficient.
    Measured against the centred block over twelve segments, the ``fhr_ph`` cross-correlation's
    argmax overshoots this composition by a median of $60.5$ steps. Aligned, the same argmax lands
    on it -- median signed miss $0$ steps, and the correlation *at* the composed delay rises from
    $+0.07$ to $+0.80$. The number stored in ``causal_delay_s`` is therefore this one in both
    cases, and it becomes correct rather than merely recorded when the block is built aligned.

    Args:
        per_filter: The quantity per first-order filter, ``(n_filters,)``.
        low_pass: The same quantity for $\phi$.
        pairs: ``(n_pairs, 2)`` of $(i, j)$.

    Returns:
        ``(n_pairs,)`` in stored channel order.
    """
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=float)
    return np.maximum(per_filter[pairs[:, 0]], per_filter[pairs[:, 1]]) + low_pass


def build_channel_plan(
    bank: CausalBank,
    target_pairs: np.ndarray,
    source_pairs: np.ndarray,
    *,
    sequence_length: int = N_RAW // DECIMATION,
    decimation: int = DECIMATION,
) -> Dict[str, CausalChannelPlan]:
    r"""The stored plan for all four causal blocks: what survives, and how stale it is.

    **The drop rule.** A channel whose warm-up exceeds the stored segment never leaves the
    pad-dominated region -- its boundary never closes, so it carries no signal at *any* step of
    *any* segment, and storing it would store the assumed pre-recording history. Those channels
    are dropped here rather than masked later, because there is no trim, no model and no
    normalisation under which they become usable.

    At the production geometry this removes the seven slowest wavelets from each scattering block
    and nothing from either phase block: both phase selections are band-limited at $0.008$ Hz,
    which excludes the slowest filters entirely, so the drop is a clean channel-axis operation
    rather than a re-selection.

    **The delay is recorded, not compensated.** A causal channel is stale by its composed group
    delay -- hundreds of seconds on the slow channels -- and nothing in this pipeline shifts it
    back. :attr:`CausalChannelPlan.delay_s` exists so a future consumer that wants to align
    channels, or to refuse a channel that is too stale for a forecast horizon, has the number
    without re-deriving it from the bank.

    Args:
        bank: The causal bank the file will be built with.
        target_pairs: ``(n_pairs, 2)`` phase pairs for ``fhr_ph``, in stored channel order.
        source_pairs: ``(n_pairs, 2)`` phase pairs for ``up_ph``.
        sequence_length: Stored decimated length; a channel warming up past it is dropped.
        decimation: Raw samples per stored step.

    Returns:
        One :class:`CausalChannelPlan` per block, keyed by block name.
    """
    support = np.array(
        [causal_support_samples(bank.psi[index]) for index in range(bank.n_filters)],
        dtype=np.float64,
    )
    phi_support = float(causal_support_samples(bank.phi))
    delay = bank.group_delay_s
    phi_delay = bank.phi_group_delay_s

    target_pairs = np.asarray(target_pairs, dtype=int).reshape(-1, 2)
    source_pairs = np.asarray(source_pairs, dtype=int).reshape(-1, 2)

    scattering_support = _compose_scattering(support, phi_support)
    scattering_delay = _compose_scattering(delay, phi_delay)
    per_block = {
        "fhr_st": (scattering_support, scattering_delay),
        "up_st": (scattering_support, scattering_delay),
        "fhr_ph": (
            _compose_phase(support, phi_support, target_pairs),
            _compose_phase(delay, phi_delay, target_pairs),
        ),
        "up_ph": (
            _compose_phase(support, phi_support, source_pairs),
            _compose_phase(delay, phi_delay, source_pairs),
        ),
    }

    plan = {}
    for name in CAUSAL_BLOCKS:
        block_support, block_delay = per_block[name]
        # Ceiling, not rounding: a step is valid only once the warm-up has fully passed, and a
        # step that is 40% pad is not half valid.
        warmup = np.ceil(block_support / decimation).astype(np.int32)
        kept = np.flatnonzero(warmup <= sequence_length)
        plan[name] = CausalChannelPlan(
            name=name,
            kept=kept.astype(np.int32),
            warmup_steps=warmup[kept],
            delay_s=block_delay[kept],
        )
    return plan


def channel_alignment_delays(
    delay_s: np.ndarray, reference_s: float, step_s: float
) -> np.ndarray:
    r"""Per-channel step shift that brings a whole stored block onto one reference clock.

    $$d_c \;=\; \operatorname{round}\!\Bigl(\kappa\,\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr),
      \qquad \Delta = \texttt{step\_s},
      \qquad \kappa = 1 - \frac{1}{2\gamma} = 0.875 .$$

    **Why the difference is scaled.** *delay_s* is the reported $\tau_g$; a channel's content sits
    at $\kappa\tau_g$ (:data:`ALIGNMENT_DELAY_FACTOR`). Only the *difference* carries the factor, so
    the reference channel still takes shift $0$ and the drop rule is untouched -- $\tau_c \le
    \tau_{\mathrm{ref}}$ is scale-invariant -- while every survivor lands on the common effective
    clock $\kappa\tau_{\mathrm{ref}}$ rather than $\tau_{\mathrm{ref}}$.

    Reading channel $c$ at step $t - d_c$ instead of $t$ makes every entry of the vector describe
    one physical instant. It is a re-indexing of the lag origin, not a loss of information:
    delaying channel $c$ by $d_c$ and then reading lag $\ell$ is reading channel $c$ at lag
    $\ell + d_c$.

    **Rounding, not ceiling.** Both directions are causally safe here -- $d_c \ge 0$ only selects
    which *already-causal* stored step is read, and over-delaying is as non-anticipative as
    under-delaying -- so the only criterion is residual misalignment, which rounding minimises at
    $\lvert\tau_{\mathrm{ref}} - \tau_c - \Delta d_c\rvert \le \Delta/2$. (Contrast the warm-up
    ceiling in :func:`build_channel_plan`, where the ceiling *is* load-bearing, because a step
    that is $40\%$ pad is not $40\%$ valid.)

    **The reference must be the maximum of the channels it is applied to.** A channel above it
    would need $d_c < 0$, i.e. to be read from a *later* stored step, which reads raw signal after
    the anchor and destroys the one property the causal construction exists for. Such channels are
    dropped by the caller, not advanced -- this refuses them by name rather than silently
    producing a negative gather, the same asymmetry that makes
    :class:`~teb_vae.lag_attn.nets.delays.ChannelDelay` refuse a negative entry.

    The comparison is exact rather than toleranced, because *reference_s* is meant to be one of
    the entries of *delay_s* (or of a sibling block's, built by the same bank and stored at the
    same precision), so equality at the reference channel is bitwise rather than approximate.

    Args:
        delay_s: ``(C,)`` composed group delay per stored channel, in seconds -- the shard's
            ``causal_delay_s``.
        reference_s: The common reference delay in seconds.
        step_s: One stored step in seconds; $4$ s at the production geometry.

    Returns:
        ``(C,)`` non-negative integer step shifts, zero at the reference channel.

    Raises:
        ValueError: If any channel's delay exceeds the reference, naming the channel index and
            both delays.
    """
    delay = np.asarray(delay_s, dtype=np.float64)
    above = np.flatnonzero(delay > float(reference_s))
    if above.size:
        index = int(above[0])
        raise ValueError(
            f"channel {index} has delay {float(delay[index]):.4f} s, above the reference "
            f"{float(reference_s):.4f} s ({above.size} of {delay.size} channels are). Aligning it "
            f"would need a negative shift, which reads the channel's own future; a channel above "
            f"the reference must be dropped, not advanced."
        )
    return np.round(
        ALIGNMENT_DELAY_FACTOR * (float(reference_s) - delay) / float(step_s)
    ).astype(np.int64)


def novelty_fraction(
    bank: CausalBank,
    plan: Dict[str, CausalChannelPlan],
    target_pairs: np.ndarray,
    source_pairs: np.ndarray,
    horizon_steps: int,
    *,
    decimation: int = DECIMATION,
) -> Dict[str, np.ndarray]:
    r"""Share of each stored channel drawn from raw samples an anchor has not yet seen.

    A target coefficient stamped at $\Delta(t + 1 + h)$ is a weighted average over the past of
    *that* instant, and the group delay puts most of that weight before the anchor at $\Delta t$.
    With composed envelope $g_c = \lvert\psi_k\rvert \star \phi$,

    $$\nu_c(h) \;=\; \frac{\int_0^{\Delta(1+h)} g_c(\tau)\,\mathrm d\tau}
                          {\int_0^{\infty} g_c(\tau)\,\mathrm d\tau}$$

    is the share of it that is genuinely new. This is **not** a leak -- every one of those
    coefficients still depends on raw samples after the anchor, so the forecast claim is exact.
    What it says is that the effective forecast horizon is **per channel**: the slowest kept
    target channel draws $2.6\%$ of its value from the two minutes it is being asked to predict,
    while a fast channel draws all of it. A block score summed over both mixes two different
    claims, which is why the number is stored rather than assumed uniform.

    **Phase channels take the slower leg.** A phase coefficient's sensitivity is a mixture of both
    legs' envelopes, weighted by a product this function does not model; the slow leg's fraction
    is the smaller of the two, so reporting it is the conservative choice and cannot overstate how
    much of a channel is forecast. Column $0$ of a pair list is the slow leg by construction.

    $S_0$ is $\phi$ alone -- no wavelet, no modulus -- so its composed envelope is the low-pass
    itself. The scattering composition is written out here rather than delegated to
    :func:`_compose_scattering`, which *adds* the low-pass term and is right for a support or a
    delay but not for a fraction.

    The convolution is done by FFT: the two kernels are $2^{15}$ taps each and a direct
    convolution of $42$ such pairs takes minutes.

    Note:
        $\int_0^\infty$ is taken over the stored kernel, which holds $99.999\%$ of the true
        envelope's $L^1$ mass at the shipped tap count -- the same approximation
        :func:`causal_support_samples` makes, and for the same reason.

    Args:
        bank: The causal bank the plan was built from.
        plan: The stored channel plan, one entry per block of :data:`CAUSAL_BLOCKS`.
        target_pairs: ``(n_pairs, 2)`` phase pairs for ``fhr_ph``, in stored channel order.
        source_pairs: ``(n_pairs, 2)`` phase pairs for ``up_ph``.
        horizon_steps: Forecast horizon $H$ in decimated steps; the window is
            ``horizon_steps * decimation`` raw samples.
        decimation: Raw samples per stored step.

    Returns:
        ``{block: (C,) float64}`` in the unit interval, on the stored channel axis.

    Raises:
        ValueError: If *horizon_steps* is not positive, which would make the fraction meaningless
            rather than merely small.
    """
    if int(horizon_steps) <= 0:
        raise ValueError(
            f"horizon_steps must be positive, got {horizon_steps}. A zero-length horizon has no "
            f"novel samples in it by definition, which measures nothing about the bank."
        )
    horizon_samples = int(horizon_steps) * int(decimation)

    # Linear (not circular) convolution of every wavelet modulus with the low-pass: the FFT is
    # taken at the next power of two above the 2n-1 output length, so nothing wraps around into
    # the leading taps this measurement is entirely about.
    n_taps = bank.n_taps
    size = 1 << int(math.ceil(math.log2(2 * n_taps)))
    spectrum = np.fft.rfft(np.abs(bank.psi), n=size, axis=-1) * np.fft.rfft(bank.phi, n=size)
    composed = np.fft.irfft(spectrum, n=size, axis=-1)[:, : 2 * n_taps - 1]
    # Both factors are non-negative, so the true convolution is; anything below zero is FFT
    # round-off, and clipping it keeps the cumulative fraction inside [0, 1] and non-decreasing.
    composed = np.maximum(composed, 0.0)

    per_filter = composed[:, :horizon_samples].sum(axis=-1) / composed.sum(axis=-1)
    low_pass = float(bank.phi[:horizon_samples].sum() / bank.phi.sum())

    scattering = np.concatenate([[low_pass], per_filter])
    per_block = {
        "fhr_st": scattering,
        "up_st": scattering,
        "fhr_ph": per_filter[np.asarray(target_pairs, dtype=int).reshape(-1, 2)[:, 0]],
        "up_ph": per_filter[np.asarray(source_pairs, dtype=int).reshape(-1, 2)[:, 0]],
    }
    return {name: per_block[name][plan[name].kept] for name in CAUSAL_BLOCKS}


# =================================================================================================
# Channel identity
# =================================================================================================
def selected_pairs(
    band_hz: Tuple[float, float],
    reference: Optional[FilterBank] = None,
) -> np.ndarray:
    r"""Phase-harmonic pair selection for a stored band, in the stored channel order.

    Delegates to :func:`select_phase_pairs` rather than restating the rule, so the two callers of
    a selection -- the measurement code here and the shard verification in
    :func:`assert_matches_shard` -- cannot drift apart on what channel $c$ means.

    Args:
        band_hz: ``(f_min, f_max)`` in Hz -- :data:`TARGET_PHASE_BAND_HZ` for ``fhr_ph``,
            :data:`SOURCE_PHASE_BAND_HZ` for ``up_ph``.
        reference: The production bank. Defaults to :func:`build_filter_bank`.

    Returns:
        ``(n_pairs, 2)`` of $(i, j)$, column $0$ indexing the lower frequency.
    """
    reference = reference if reference is not None else build_filter_bank()
    pairs = select_phase_pairs(reference, band_hz[0], band_hz[1], PHASE_K_STEPS, PHASE_REL_TOL)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def assert_matches_shard(
    pairs: np.ndarray,
    reference: FilterBank,
    attrs: Mapping[str, Any],
    *,
    name: str = "block",
) -> None:
    r"""Refuse to proceed unless a rebuilt pair list is the one stored in the shard.

    This is the channel-correspondence guarantee, and it runs **before** any comparison rather than
    as an afterthought: arm C channel $c$ means arm A channel $c$ only because this passes. A
    silent misalignment would produce a plausible-looking but meaningless comparison, which is a
    worse outcome than a crash.

    Args:
        pairs: ``(n_pairs, 2)`` from :func:`selected_pairs`.
        reference: The production bank, for the centre frequencies.
        attrs: The stored dataset's HDF5 attributes.
        name: Block name, for the error message.

    Raises:
        ValueError: On any disagreement in count, index, frequency or harmonic power.
    """
    stored_i = np.asarray(attrs["sel_i"], dtype=int)
    stored_j = np.asarray(attrs["sel_j"], dtype=int)
    if pairs.shape[0] != stored_i.size:
        raise ValueError(
            f"{name}: rebuilt {pairs.shape[0]} phase pairs but the shard stores {stored_i.size}"
        )
    if not np.array_equal(pairs[:, 0], stored_i) or not np.array_equal(pairs[:, 1], stored_j):
        raise ValueError(
            f"{name}: rebuilt phase pairs differ from the shard's sel_i/sel_j; channel c here "
            f"would not be channel c on disk, so the comparison would be meaningless"
        )
    hz = reference.hz
    for stored_key, computed in (
        ("sel_xi_i_hz", hz[stored_i]),
        ("sel_xi_j_hz", hz[stored_j]),
    ):
        if stored_key in attrs:
            stored = np.asarray(attrs[stored_key], dtype=float)
            if not np.allclose(stored, computed, rtol=1e-5, atol=1e-9):
                raise ValueError(f"{name}: {stored_key} disagrees with the rebuilt bank")
    if "sel_power" in attrs:
        stored = np.asarray(attrs["sel_power"], dtype=float)
        if not np.allclose(stored, hz[stored_j] / hz[stored_i], rtol=1e-5):
            raise ValueError(f"{name}: sel_power disagrees with the rebuilt bank")


def transform_sample(
    fhr: np.ndarray,
    up: np.ndarray,
    bank: FilterBank | CausalBank,
    *,
    decimation: int = DECIMATION,
    phi_mode: str = "exact",
    decimation_mode: str = "full_rate",
    pad: str = "edge",
    reference: Optional[FilterBank] = None,
) -> Dict[str, np.ndarray]:
    r"""All four feature blocks the model consumes, for one segment, on either arm.

    Dispatches on the bank type so a caller can run arm B and arm C through one call site and be
    sure the only difference is the filters.

    ``fhr_up_ph`` is deliberately not produced: ``teb_vae/lag_attn_rws/DESIGN.md`` S2 records that
    it is never loaded, because a cross-channel coefficient mixes both signals in one number and
    would destroy the separation between the target-only prior and the source-conditioned
    posterior.

    Args:
        fhr: Raw fetal heart rate, ``(n_signal,)``.
        up: Raw uterine pressure, ``(n_signal,)``.
        bank: A :class:`FilterBank` for arm B or a :class:`CausalBank` for arm C.
        decimation: Subsampling factor.
        phi_mode: Two-sided arm only; see :func:`phase_block_two_sided`. Pair with
            ``decimation_mode='kymatio'`` and ``phi_mode='kymatio_truncate'`` to reproduce the
            shard, or leave both at their defaults for the convention the causal arm shares.
        decimation_mode: Two-sided arm only; see :func:`scattering_block_two_sided`.
        pad: Causal arm only; see :func:`causal_convolve`.
        reference: The production bank, for pair selection. Defaults to :func:`build_filter_bank`.

    Returns:
        ``{'fhr_st': (43, T), 'fhr_ph': (66, T), 'up_st': (43, T), 'up_ph': (15, T)}``.
    """
    reference = reference if reference is not None else build_filter_bank()
    target_pairs = selected_pairs(TARGET_PHASE_BAND_HZ, reference)
    source_pairs = selected_pairs(SOURCE_PHASE_BAND_HZ, reference)

    if isinstance(bank, CausalBank):
        return {
            "fhr_st": scattering_block_causal(fhr, bank, decimation=decimation, pad=pad),
            "fhr_ph": phase_block_causal(
                fhr, fhr, target_pairs, bank, decimation=decimation, pad=pad
            ),
            "up_st": scattering_block_causal(up, bank, decimation=decimation, pad=pad),
            "up_ph": phase_block_causal(up, up, source_pairs, bank, decimation=decimation, pad=pad),
        }

    return {
        "fhr_st": scattering_block_two_sided(
            fhr, bank, decimation=decimation, decimation_mode=decimation_mode
        ),
        "fhr_ph": phase_block_two_sided(
            fhr, fhr, target_pairs, bank, phi_mode=phi_mode, decimation=decimation
        ),
        "up_st": scattering_block_two_sided(
            up, bank, decimation=decimation, decimation_mode=decimation_mode
        ),
        "up_ph": phase_block_two_sided(
            up, up, source_pairs, bank, phi_mode=phi_mode, decimation=decimation
        ),
    }
