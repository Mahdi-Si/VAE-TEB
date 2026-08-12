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

    Returns:
        ``(n_pairs, n_signal)``, complex.
    """
    index_low, index_high = pairs[:, 0], pairs[:, 1]
    power = (xi[index_high] / xi[index_low])[:, None]
    y_low = responses_low[index_low]
    accelerated = np.abs(y_low) * np.exp(1j * power * np.angle(y_low))
    return accelerated * np.conj(responses_high[index_high])


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
) -> np.ndarray:
    r"""Phase-harmonic correlations on a causal bank -- arm C.

    Only the documented operator is available. ``'kymatio_truncate'`` has no causal counterpart:
    keeping the positive-frequency half of a spectrum is a non-causal projection, so transplanting
    production's deviation onto a causal chain would silently reintroduce future dependence.

    Note:
        A pair's warm-up is set by its **slower** leg, and its delay compounds both legs plus the
        low-pass -- so a phase channel is staler than either of the scattering channels it is
        built from. This is why the comparison reports the phase blocks separately.

    Args:
        x_low: Signal supplying the accelerated leg, ``(n_signal,)``.
        x_high: Signal supplying the conjugated leg, ``(n_signal,)``.
        pairs: ``(n_pairs, 2)`` of $(i, j)$.
        bank: The causal bank.
        decimation: Subsampling factor.
        pad: History supplied before $t = 0$.

    Returns:
        ``(n_pairs, n_signal // decimation)``, real.
    """
    if len(pairs) == 0:
        return np.zeros((0, int(x_low.shape[-1]) // decimation))

    responses_low = causal_convolve(x_low, bank.psi, pad=pad)
    responses_high = responses_low if x_high is x_low else causal_convolve(x_high, bank.psi, pad=pad)
    products = phase_products(responses_low, responses_high, pairs, bank.xi)
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
