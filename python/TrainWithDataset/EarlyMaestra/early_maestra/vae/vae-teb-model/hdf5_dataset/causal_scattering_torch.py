r"""The causal scattering and phase-harmonic chain, batched on a torch device.

Why this module exists
----------------------
:mod:`hdf5_dataset.causal_scattering` defines the transform and proves it correct, in numpy, one
segment per call. A full dataset build is on the order of $120{,}000$ segment-transforms, which at
$1$--$2$ s each is days of CPU -- so the measurement code cannot be the build code. This module is
the build path: the same chain, batched, on whichever device the writer is given.

It reimplements **only the convolution**. Every filter comes from the finished
:class:`~hdf5_dataset.causal_scattering.CausalBank` returned by the public
:func:`~hdf5_dataset.causal_scattering.build_causal_bank`, and no underscore-prefixed name is
imported from that module, so there is exactly one definition of the filter design and no way for
the two paths to drift. What is checked against numpy is therefore a convolution, not a filter
bank -- which is what makes the numerical gate in the tests interpretable.

Three things here are correctness, not performance
--------------------------------------------------
* **The FFT is complex on the wavelet path.** The gammatone kernels are complex and
  non-Hermitian, and the chain needs the complex response: the scattering block takes
  $|\cdot|$ and the phase block takes $\arg(\cdot)$. ``irfft`` would return a real signal,
  silently discarding the negative-frequency content that *is* the measured analyticity defect --
  the quantity that makes a causal filter only approximately analytic. ``rfft``/``irfft`` is used
  only for the two genuinely real-input $\phi$ smoothings, where it is exact.
* **The history pad is ``'edge'`` or ``'zero'``, never reflection.** Reflection mirrors the signal
  *forward* in time, which would reintroduce exactly the future dependence a causal kernel exists
  to remove. :meth:`CausalTorchBank.prepend_history` refuses it by name.
* **Decimation is plain subsampling at the very end.** The cascade stays at full rate throughout.
  Taking every $16$-th sample is visibly causal; kymatio's spectral periodisation before the
  modulus is not, and it aliases by up to $2.6\times10^{-2}$ relative on the fastest bands.

The batch axis is independent by construction -- every operation is elementwise or an FFT along
the time axis -- so a segment's coefficients do not depend on the batch it was computed in. The
writer's per-segment retry re-runs a failed segment **alone**, and stores the result beside its
peers, so that independence is a correctness requirement rather than a nicety; the tests assert it.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch

from .causal_scattering import (
    CAUSAL_BLOCKS,
    DECIMATION,
    N_RAW,
    CausalBank,
    CausalChannelPlan,
    resolve_leg_alignment,
)

#: Pad modes offering an assumed pre-recording history. ``'edge'`` replicates $x(0)$, asserting
#: only that the recording was locally constant before it started -- and a zero-mean $\psi$
#: annihilates a constant exactly, so the assertion costs nothing in the passband.
HISTORY_PAD_MODES = ("edge", "zero")


def _fft_length(history: int, n_signal: int) -> int:
    r"""The shortest FFT length at which the retained output slice is free of wraparound.

    A linear convolution of the padded signal (``history + n_signal`` samples) with a kernel of
    ``history + 1`` taps is ``2 * history + n_signal`` samples long, and a length-$F$ circular
    convolution aliases the tail back onto the head: output $n$ picks up the linear result at
    $n + F$, for every $n$ below ``2 * history + n_signal - F``. Only ``[history, history +
    n_signal)`` is kept, so the requirement is

    $$2H + N - F \le H \iff F \ge H + N,$$

    the padded length itself -- **not** the full linear length. At the production geometry that is
    $32767 + 5280 = 38047$, so $F = 2^{16}$, where the numpy reference sizes its transform at
    $2^{17}$ to make wraparound impossible everywhere rather than merely where it is read. Halving
    $F$ halves both the time and the memory of every stage, and the retained slice is bit-for-bit
    the same convolution.

    Args:
        history: Assumed-history samples prepended, i.e. ``n_taps - 1``.
        n_signal: Signal length in raw samples.

    Returns:
        The power of two at or above ``history + n_signal``.
    """
    padded = history + n_signal
    length = 1 << (padded - 1).bit_length()
    if length < padded:  # pragma: no cover - arithmetic guard, unreachable for positive inputs
        raise ValueError(f"fft length {length} is below the padded length {padded}")
    return length


class CausalTorchBank:
    r"""A causal bank realised on a device, with its kernel spectra cached.

    Construction is the expensive part and happens once per output file: the kernels are built in
    float64 on the CPU by :func:`~hdf5_dataset.causal_scattering.build_causal_bank`, transformed at
    :meth:`fft_length`, cast, and moved. Every batch then costs one forward FFT of the signal, a
    broadcast multiply against the cached spectra, and one inverse FFT.

    Args:
        bank: The finished causal bank. Its kernels are used as they are; no filter formula is
            restated here.
        device: Where the spectra live and the batches are transformed. **Explicit**: a build is a
            multi-hour commitment to one GPU on an eight-GPU box, so there is no implicit
            ``cuda`` default to pick the wrong one silently.
        n_signal: Raw segment length the bank is sized for. Fixes :meth:`fft_length`, so a batch
            of a different length is refused rather than silently transformed at the wrong size.
        dtype: ``torch.complex64`` for the build, ``torch.complex128`` for the numerical gate
            against the float64 numpy reference.

    Attributes:
        bank: The source :class:`~hdf5_dataset.causal_scattering.CausalBank`.
        device: The device the spectra live on.
        n_signal: Raw segment length.
        history: Assumed-history samples prepended before every convolution, ``n_taps - 1``.

    Raises:
        ValueError: On a dtype that is not one of the two complex types.
    """

    def __init__(
        self,
        bank: CausalBank,
        device: torch.device | str,
        *,
        n_signal: int = N_RAW,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError(
                f"dtype must be torch.complex64 or torch.complex128, got {dtype}. The chain is "
                f"complex throughout: the wavelet responses carry the phase the phase block reads."
            )
        self.bank = bank
        self.device = torch.device(device)
        self.n_signal = int(n_signal)
        self.history = bank.n_taps - 1
        self._dtype = dtype
        self._real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self._fft_length = _fft_length(self.history, self.n_signal)

        # Transform in float64 and cast afterwards, so the cached spectra are the correctly-rounded
        # single-precision values of the double-precision kernels rather than the transform of
        # already-rounded ones.
        psi = torch.from_numpy(np.ascontiguousarray(bank.psi))
        phi = torch.from_numpy(np.ascontiguousarray(bank.phi)).to(torch.complex128)
        self._psi_spectra = (
            torch.fft.fft(psi, n=self._fft_length, dim=-1).to(dtype).to(self.device)
        )
        self._phi_spectrum = (
            torch.fft.fft(phi, n=self._fft_length, dim=-1).to(dtype).to(self.device)
        )
        self._xi = torch.from_numpy(np.ascontiguousarray(bank.xi)).to(
            self._real_dtype
        ).to(self.device)

    # ---------------------------------------------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------------------------------------------
    @property
    def fft_length(self) -> int:
        """Transform length; see :func:`_fft_length` for why it is the padded length, not more."""
        return self._fft_length

    @property
    def n_filters(self) -> int:
        """Number of first-order wavelets."""
        return self.bank.n_filters

    @property
    def dtype(self) -> torch.dtype:
        """Complex dtype the chain runs in."""
        return self._dtype

    @property
    def spectra_bytes(self) -> int:
        """Device memory held by the cached spectra -- the one fixed cost of a build.

        $(K + 1)$ spectra of :meth:`fft_length` complex elements: $23$ MB at the production
        geometry in ``complex64``, built once per output file and reused across every batch.
        """
        return (self.n_filters + 1) * self._fft_length * self._psi_spectra.element_size()

    @property
    def psi_spectra(self) -> torch.Tensor:
        """The cached wavelet spectra, ``(n_filters, fft_length)``."""
        return self._psi_spectra

    @property
    def phi_spectrum(self) -> torch.Tensor:
        """The cached low-pass spectrum, ``(fft_length,)``."""
        return self._phi_spectrum

    # ---------------------------------------------------------------------------------------------
    # Convolution
    # ---------------------------------------------------------------------------------------------
    def prepend_history(self, x: torch.Tensor, pad: str = "edge") -> torch.Tensor:
        """Prepend :attr:`history` samples of assumed past to each row.

        Args:
            x: ``(..., n_signal)``.
            pad: ``'edge'`` or ``'zero'``.

        Returns:
            ``(..., history + n_signal)``.

        Raises:
            ValueError: On any other mode, and with a message for ``'reflect'`` in particular.
        """
        if pad == "reflect":
            raise ValueError(
                "pad='reflect' is refused on a causal chain: reflection mirrors the signal "
                "forward in time, so the assumed history would be a copy of the future the "
                "transform exists not to read. Use 'edge' or 'zero'."
            )
        if pad not in HISTORY_PAD_MODES:
            raise ValueError(f"unknown pad mode {pad!r}; use 'edge' or 'zero'")
        shape = x.shape[:-1] + (self.history,)
        prefix = (
            x[..., :1].expand(shape)
            if pad == "edge"
            else torch.zeros(shape, dtype=x.dtype, device=x.device)
        )
        return torch.cat([prefix, x], dim=-1)

    def _retained(self, filtered: torch.Tensor) -> torch.Tensor:
        """Slice the output samples whose every tap came from real signal or assumed history."""
        return filtered[..., self.history : self.history + self.n_signal]

    def wavelet_responses(self, x: torch.Tensor, pad: str = "edge") -> torch.Tensor:
        r"""$y_k = x \star \psi_k$ for every wavelet, over a batch.

        One forward FFT of the signal per batch, not one per filter: the transform is taken once
        and broadcast against the cached spectra.

        Args:
            x: Real signals, ``(B, n_signal)``.
            pad: History supplied before $t = 0$.

        Returns:
            ``(B, n_filters, n_signal)`` complex.
        """
        # Complex both ways, deliberately. rfft here would be cheaper and irfft would return a real
        # signal, discarding the negative-frequency content that *is* the measured analyticity
        # defect -- and with it the phase the phase block reads. The input is left real only
        # because torch.fft.fft promotes it itself, so casting first would merely double the memory
        # of the largest tensor in the chain.
        padded = self.prepend_history(self._as_real(x), pad)
        spectrum = torch.fft.fft(padded, n=self._fft_length, dim=-1)
        filtered = torch.fft.ifft(spectrum[:, None, :] * self._psi_spectra[None, :, :], dim=-1)
        return self._retained(filtered)

    def smooth_real(self, u: torch.Tensor, pad: str = "edge") -> torch.Tensor:
        r"""Low-pass real rows with $\phi$: $u \star \phi$, staying real throughout.

        The only place ``rfft``/``irfft`` is legitimate. $\phi$ is real and non-negative and the
        input here is a signal or a modulus envelope, so the product spectrum is Hermitian and the
        inverse real transform is exact -- it discards nothing but the redundant half.

        Args:
            u: ``(..., n_signal)`` real.
            pad: History supplied before $t = 0$.

        Returns:
            ``(..., n_signal)`` real.
        """
        padded = self.prepend_history(self._as_real(u), pad)
        spectrum = torch.fft.rfft(padded, n=self._fft_length, dim=-1)
        half = self._phi_spectrum[: self._fft_length // 2 + 1]
        smoothed = torch.fft.irfft(spectrum * half, n=self._fft_length, dim=-1)
        return self._retained(smoothed)

    def smooth_complex(self, u: torch.Tensor, pad: str = "edge") -> torch.Tensor:
        r"""Low-pass complex rows with $\phi$, keeping both halves of the spectrum.

        The phase-harmonic products are complex and not Hermitian, so this is the path they take.

        Args:
            u: ``(..., n_signal)`` complex.
            pad: History supplied before $t = 0$.

        Returns:
            ``(..., n_signal)`` complex.
        """
        padded = self.prepend_history(u.to(self._dtype), pad)
        spectrum = torch.fft.fft(padded, n=self._fft_length, dim=-1)
        smoothed = torch.fft.ifft(spectrum * self._phi_spectrum, dim=-1)
        return self._retained(smoothed)

    # ---------------------------------------------------------------------------------------------
    # The stored blocks
    # ---------------------------------------------------------------------------------------------
    def scattering_block(
        self, x: torch.Tensor, *, decimation: int = DECIMATION, pad: str = "edge"
    ) -> torch.Tensor:
        r"""Order-$0$ and order-$1$ scattering: $S_0 = x \star \phi$, $S_1^{(k)} = |x \star \psi_k| \star \phi$.

        Channel $0$ is $S_0$ and channels $1 \ldots K$ are $S_1$ in bank order (descending $\xi$),
        which is the layout ``fhr_st`` and ``up_st`` are stored in.

        Args:
            x: Real signals, ``(B, n_signal)``.
            decimation: Subsampling factor applied once, at the end.
            pad: History supplied before $t = 0$.

        Returns:
            ``(B, 1 + n_filters, n_signal // decimation)`` real.
        """
        signal = self._as_real(x)
        order_zero = self.smooth_real(signal, pad)[:, None, :]
        envelopes = torch.abs(self.wavelet_responses(signal, pad))
        order_one = self.smooth_real(envelopes, pad)
        return torch.cat([order_zero, order_one], dim=1)[..., ::decimation]

    def phase_block(
        self,
        x_low: torch.Tensor,
        x_high: torch.Tensor,
        pairs: np.ndarray | torch.Tensor,
        *,
        decimation: int = DECIMATION,
        pad: str = "edge",
        leg_alignment: str = "none",
    ) -> torch.Tensor:
        r"""Phase-harmonic correlations $\Phi_{ij} = \Re\{([y_i]^{p_{ij}}\overline{y_j}) \star \phi\}$.

        $[y]^p = |y| e^{\,i p \arg y}$ with $p_{ij} = \xi_j/\xi_i \ge 1$, formed in **polar
        coordinates** exactly as the numpy reference does -- a complex power would cross the branch
        cut and produce a different function. The magnitude is taken to the first power, so
        $\big|[y_i]^p \overline{y_j}\big| = |y_i||y_j|$.

        Only the documented smoothing operator is available. Production's spectral truncation is
        the analytic (positive-frequency) projection, which is a non-causal operation; transplanting
        it here would silently reintroduce future dependence, so there is no ``phi_mode``.

        Args:
            x_low: Signal supplying the accelerated leg $y_i$, ``(B, n_signal)``.
            x_high: Signal supplying the conjugated leg $y_j$; pass the same tensor for a
                self-phase block.
            pairs: ``(n_pairs, 2)`` of $(i, j)$ in stored channel order, $i$ the lower frequency.
                Supplied by the caller, because the writer must take its pairs from the same
                selection it writes the ``sel_*`` provenance from.
            decimation: Subsampling factor applied once, at the end.
            pad: History supplied before $t = 0$.
            leg_alignment: ``'none'`` or ``'envelope'``; see
                :data:`~hdf5_dataset.causal_scattering.LEG_ALIGNMENT_MODES`. The shift and its
                de-rotation phasor come from the shared numpy bank rather than being recomputed
                here, which is what keeps the two implementations from drifting -- and is not only
                a tidiness argument: the phasor's angle reaches $9.6$ turns, so evaluating
                $e^{\,i2\pi\xi_j s}$ in this module's single precision would lose four digits that
                a float64 evaluation rounded once does not.

        Returns:
            ``(B, n_pairs, n_signal // decimation)`` real.

        Raises:
            ValueError: On an unknown *leg_alignment*.
        """
        index = torch.as_tensor(np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                                device=self.device)
        # Resolved before the empty-pair return, so an unknown mode is refused either way.
        leg_shift = resolve_leg_alignment(self.bank, pairs, leg_alignment)
        if index.shape[0] == 0:
            return torch.zeros(
                (x_low.shape[0], 0, self.n_signal // decimation),
                dtype=self._real_dtype, device=self.device,
            )

        responses_low = self.wavelet_responses(x_low, pad)
        responses_high = (
            responses_low if x_high is x_low else self.wavelet_responses(x_high, pad)
        )
        low = responses_low[:, index[:, 0], :]
        high = responses_high[:, index[:, 1], :]
        if leg_shift is not None:
            high = self._align_leg(high, *leg_shift)
        power = (self._xi[index[:, 1]] / self._xi[index[:, 0]])[None, :, None]
        accelerated = torch.polar(torch.abs(low), power * torch.angle(low))
        smoothed = self.smooth_complex(accelerated * torch.conj(high), pad)
        return smoothed.real[..., ::decimation]

    def _align_leg(
        self, high: torch.Tensor, shift: np.ndarray, phasor: np.ndarray
    ) -> torch.Tensor:
        r"""Delay each **already-gathered** conjugated leg and de-rotate its carrier.

        No restructuring is needed here, unlike the numpy reference: the gather onto the pair axis
        has already happened above, so ``high`` is one row per pair and the per-pair shift lands
        directly on it.

        Args:
            high: ``(B, n_pairs, n_signal)`` complex, the gathered conjugated legs.
            shift: ``(n_pairs,)`` integer raw-sample shifts.
            phasor: ``(n_pairs,)`` complex de-rotations, from the numpy bank in float64.

        Returns:
            ``(B, n_pairs, n_signal)`` complex.
        """
        taps = torch.arange(high.shape[-1], device=self.device)
        # Clamping at zero is the edge replication: a tap the shift pushes before the start reads
        # the response's first sample, which is the same assumed history the convolution ran on.
        source = torch.clamp(
            taps[None, :] - torch.as_tensor(shift, device=self.device)[:, None], min=0
        )
        delayed = torch.gather(high, -1, source[None].expand(high.shape[0], -1, -1))
        rotation = torch.as_tensor(phasor, device=self.device).to(self._dtype)
        return delayed * rotation[None, :, None]

    def transform_batch(
        self,
        fhr: torch.Tensor,
        up: torch.Tensor,
        target_pairs: np.ndarray | torch.Tensor,
        source_pairs: np.ndarray | torch.Tensor,
        *,
        plan: Optional[Mapping[str, CausalChannelPlan]] = None,
        decimation: int = DECIMATION,
        pad: str = "edge",
        leg_alignment: str = "none",
    ) -> Dict[str, torch.Tensor]:
        r"""All four stored blocks for a batch of segments, dropped to the plan's channels.

        ``fhr_up_ph`` is not produced. The cross-phase block mixes both signals into one number,
        no model loads it, and it is the one block with no ``sel_*`` provenance to verify channel
        identity against.

        Args:
            fhr: Raw fetal heart rate, ``(B, n_signal)``.
            up: Raw uterine pressure, ``(B, n_signal)``.
            target_pairs: Phase pairs for ``fhr_ph``.
            source_pairs: Phase pairs for ``up_ph``.
            plan: The stored channel plan. ``None`` returns every channel undropped, which is what
                the drop rule is measured against.
            decimation: Subsampling factor.
            pad: History supplied before $t = 0$.
            leg_alignment: Applied to **both** phase blocks; it is a property of the transform a
                shard was built with, not of one block. Defaults to ``'none'``, so a caller that
                does not ask for the alignment gets bit-for-bit what every shard on disk holds.

        Returns:
            ``{'fhr_st', 'fhr_ph', 'up_st', 'up_ph'}``, each ``(B, C, n_signal // decimation)``.

        Raises:
            ValueError: If a signal's length is not the length this bank was sized for, or on an
                unknown *leg_alignment*.
        """
        for name, signal in (("fhr", fhr), ("up", up)):
            if int(signal.shape[-1]) != self.n_signal:
                raise ValueError(
                    f"{name} is {int(signal.shape[-1])} samples but this bank was built for "
                    f"{self.n_signal}; rebuild it, because the transform length is sized from it"
                )

        blocks = {
            "fhr_st": self.scattering_block(fhr, decimation=decimation, pad=pad),
            "fhr_ph": self.phase_block(fhr, fhr, target_pairs, decimation=decimation, pad=pad,
                                       leg_alignment=leg_alignment),
            "up_st": self.scattering_block(up, decimation=decimation, pad=pad),
            "up_ph": self.phase_block(up, up, source_pairs, decimation=decimation, pad=pad,
                                      leg_alignment=leg_alignment),
        }
        if plan is None:
            return blocks
        return {name: self._gather(blocks[name], plan[name]) for name in CAUSAL_BLOCKS}

    def _gather(self, block: torch.Tensor, plan: CausalChannelPlan) -> torch.Tensor:
        """Keep the plan's channels, in the plan's order, leaving their values untouched.

        Args:
            block: ``(B, C_full, T)``.
            plan: The block's plan.

        Returns:
            ``(B, C_kept, T)``.

        Raises:
            ValueError: If the plan does not describe a block of this width, which would otherwise
                surface as an out-of-range gather or, worse, as a silently wrong channel map.
        """
        kept = torch.as_tensor(plan.kept.astype(np.int64), device=block.device)
        if kept.numel() and int(kept.max()) >= block.shape[1]:
            raise ValueError(
                f"plan for '{plan.name}' indexes channel {int(kept.max())} of a block that is "
                f"{block.shape[1]} channels wide; the plan and the transform disagree about the "
                f"filter bank"
            )
        return block.index_select(1, kept)

    def _as_real(self, x: torch.Tensor) -> torch.Tensor:
        """Move a real input onto this bank's device and precision."""
        return x.to(device=self.device, dtype=self._real_dtype)


def transform_batch_numpy(
    torch_bank: CausalTorchBank,
    fhr: np.ndarray,
    up: np.ndarray,
    target_pairs: np.ndarray,
    source_pairs: np.ndarray,
    *,
    plan: Optional[Mapping[str, CausalChannelPlan]] = None,
    decimation: int = DECIMATION,
    pad: str = "edge",
    leg_alignment: str = "none",
) -> Dict[str, np.ndarray]:
    """Transform a numpy batch and return numpy, for callers that never see a tensor.

    The writer holds its segments as numpy and writes numpy, so this is the seam that keeps the
    device round trip in one place instead of at every call site.

    Args:
        torch_bank: The realised bank.
        fhr: ``(B, n_signal)``.
        up: ``(B, n_signal)``.
        target_pairs: Phase pairs for ``fhr_ph``.
        source_pairs: Phase pairs for ``up_ph``.
        plan: The stored channel plan; ``None`` returns every channel.
        decimation: Subsampling factor.
        pad: History supplied before $t = 0$.
        leg_alignment: Passed through to both phase blocks, defaulting to ``'none'``.

    Returns:
        The four blocks as numpy arrays, in the bank's real precision.
    """
    with torch.no_grad():
        blocks = torch_bank.transform_batch(
            torch.from_numpy(np.ascontiguousarray(fhr)),
            torch.from_numpy(np.ascontiguousarray(up)),
            target_pairs,
            source_pairs,
            plan=plan,
            decimation=decimation,
            pad=pad,
            leg_alignment=leg_alignment,
        )
    return {name: value.detach().cpu().numpy() for name, value in blocks.items()}


