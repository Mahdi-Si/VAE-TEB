r"""The pointwise source representation, and the per-anchor per-lag gather that reads it.

The whole of the source encoder, and the whole of what makes this architecture different from its
lag-attentive sibling on the input side: each standardized source coefficient becomes its own
two-number representation,

$$e_{s,j} = \bigl[x^{\mathrm{safe}}_{s,j},\; m_{s,j}\bigr],
\qquad
x^{\mathrm{safe}}_{s,j} = \begin{cases}\bar U_{s,j}, & m_{s,j} = 1,\\ 0, & m_{s,j} = 0,\end{cases}$$

and nothing mixes two stored times before the fusion boundary. Every available coefficient is
recoverable **exactly** from its own representation, which is the property the whole design rests
on: with masks and metadata held fixed,

$$\frac{\partial e_{s,j}}{\partial \bar U_{r,k}} = 0 \quad\text{for } (r,k) \neq (s,j).$$

The recommended configuration therefore has **no source encoder parameters at all**. The optional
scalar lift is the one arm that adds any, and it is per channel and per coefficient, so it does not
weaken the statement above -- it only widens each coefficient's own representation.

**What must never appear in this module.** No temporal convolution, no recurrence, no state-space
block, no attention, no pooling, no temporal normalisation, no source dropout and no learned
competition across lags. Joint interpretation of channels at one source time happens in the fusion
head that consumes this output, and joint interpretation of *times* happens only in the explicit sum
at the latent-parameter boundary.

**Availability is per channel, not per stream.** A slow source channel is unavailable long after a
fast one has settled, and at some anchors the source index is perfectly in range while the channel
that index would read has not warmed up yet. Those are two different conditions with two different
causes, and this module keeps them separate all the way through: index support is a property of
$t - \ell$, feature warm-up is a property of the channel, and conflating them reports a channel as
readable for as many steps as its own warm-up lasts, with no shape changing and nothing raising.

**Sanitisation is a replacement, never a multiplication.** A nonfinite value multiplied by a zero
mask is still nonfinite; the value is substituted first and the mask applied afterwards. And a
nonfinite value at a position the availability rules call valid is a data error rather than an
implicit zero observation: it is refused by name, because silently normalising it to zero would
feed the model a fabricated observation on a channel that reported none.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

#: Coordinates every encoded coefficient carries: its own standardized value, and its availability
#: bit. Named rather than written as a literal because the scalar-lift arm appends to it, and three
#: sites downstream size a linear layer from the total.
IDENTITY_WIDTH = 2

#: Coordinates the optional scalar lift appends per coefficient.
LIFT_WIDTH = 2

#: Hidden width of the per-channel lift, whose shape is $1 \rightarrow \text{hidden} \rightarrow 2$.
LIFT_HIDDEN = 8


class PointwiseSourceEncoder(nn.Module):
    r"""Encode each source coefficient on its own, with its availability bit beside it.

    An ``nn.Module`` rather than a free function for one reason only: the optional scalar lift
    carries parameters, and a module is what puts them in the tree, on the right device and in the
    state dict. In the recommended configuration the module holds **no parameters**, which
    :meth:`has_parameters` reports and a construction test asserts.

    The availability rule this module owns is the per-stored-step half,

    $$m_{s,j} = \mathbb 1\bigl[s \ge W'_j\bigr] \, v^U_{s,j},$$

    the channel's own warm-up against the accepted-observation indicator. The remaining factor --
    whether the *lagged* index $t - \ell$ exists at all -- belongs to the gather, because it is a
    property of the anchor and the lag rather than of the stored stream, and it is applied in
    :func:`gather_lag_window`.

    Attributes:
        c_u: Declared source channel count, as the gated stream presents it.
        warmup_steps: $W'_j$ per channel, positional against the stream's own channel order, or
            ``None`` for a stream with no warm-up at all -- where every step is warm and there is
            nothing to wait out.
        scalar_lift: Whether the per-channel lift was built.
        withhold_values: Whether the value coordinate is emitted as an exact zero, leaving the
            availability bit as the whole of what a lag says.
    """

    #: Declared so the registered buffer types as a tensor rather than as ``Tensor | Module``.
    warmup_vector: torch.Tensor

    def __init__(
        self,
        *,
        c_u: int,
        warmup_steps: Optional[Sequence[int]] = None,
        scalar_lift: bool = False,
        withhold_values: bool = False,
        lift_hidden: int = LIFT_HIDDEN,
    ) -> None:
        r"""Initialize the encoder.

        Args:
            c_u: Source channel count of the stream this encoder reads. This is the **gated**
                width when the model carries a source gate, because the gate runs first and the
                warm-up vector is positional against its survivors.
            warmup_steps: $W'_j$ per channel, in the stream's own channel order. ``None`` builds no
                warm-up condition, which is the ungated arm.
            scalar_lift: Build the optional per-channel lift, widening each coefficient's
                representation from :data:`IDENTITY_WIDTH` to
                ``IDENTITY_WIDTH + LIFT_WIDTH``. The identity and mask coordinates are retained
                either way, so the lift can never discard the original available coefficient.
            withhold_values: Emit $e_{s,j} = [0, m_{s,j}]$ instead of
                $[x^{\mathrm{safe}}_{s,j}, m_{s,j}]$, so the fusion head receives the availability
                announcement and the lag identity and **no source value at all**.

                This is the capacity control, and the reason it lives here rather than in an
                intervention is that it has to be a property of a *trained* model: the question it
                answers is whether an enabled residual head of the same trainable capacity can
                produce a predictive gain from lag identity and availability alone, which no
                intervention on a model trained with values could ask. The width is unchanged, so
                the head, the optimizer state and the checkpoint's key set are the candidate's
                exactly and the arm differs from it in one declared thing.
            lift_hidden: Hidden width of each channel's lift.

        Raises:
            ValueError: If ``c_u`` is not positive, if ``warmup_steps`` has a length other than
                ``c_u`` -- it is positional against the channel axis, so a mismatched vector would
                apply one channel's warm-up to another with every shape still correct -- if any
                warm-up entry is negative, if ``lift_hidden`` is not positive on a lifted arm, or
                if the lift is asked for alongside withheld values, where every lift output is the
                per-channel constant $\phi_j(0)$ and the arm would be carrying learned parameters
                that read nothing.
        """
        super().__init__()
        if int(c_u) <= 0:
            raise ValueError(f"c_u must be > 0, got {c_u}")
        if bool(scalar_lift) and bool(withhold_values):
            raise ValueError(
                "scalar_lift and withhold_values cannot both be set: the lift reads the value "
                "coordinate, and with the values withheld it maps an exact zero to a learned "
                "per-channel constant on every coefficient of every lag. The arm would hold "
                "parameters that read nothing and would still be reported as a capacity control."
            )
        self.c_u = int(c_u)
        self.withhold_values = bool(withhold_values)

        waits = None if warmup_steps is None else [int(step) for step in warmup_steps]
        if waits is not None:
            if len(waits) != self.c_u:
                raise ValueError(
                    f"warmup_steps has {len(waits)} entries against c_u={self.c_u}. The vector is "
                    f"positional against the source channel axis, so a mismatched length would "
                    f"wait out one channel's warm-up on another channel's coefficients."
                )
            if any(step < 0 for step in waits):
                raise ValueError(
                    f"warmup_steps has negative entries: "
                    f"{[step for step in waits if step < 0]}. A warm-up is a count of steps to "
                    f"wait, and a negative one would mark a channel available before the record "
                    f"begins."
                )
        self.warmup_steps: Optional[Tuple[int, ...]] = (
            None if waits is None else tuple(waits)
        )

        # Non-persistent for the reason every geometry-shaped tensor in this family is: its length
        # follows the resolved warm-up budget, so a persistent copy would make a checkpoint trained
        # at one budget fail to load at another and report it as a missing key rather than as a
        # budget mismatch. Registered at all -- rather than kept as a plain tuple -- so the
        # comparison against the step index happens on the module's own device.
        self.register_buffer(
            "warmup_vector",
            torch.zeros(self.c_u, dtype=torch.long)
            if waits is None
            else torch.tensor(waits, dtype=torch.long),
            persistent=False,
        )

        self.scalar_lift = bool(scalar_lift)
        if not self.scalar_lift:
            # No parameter at all, rather than an identity one: the recommended arm's claim is that
            # the source encoder has no learned parameters, and an inert tensor in the state dict
            # would make that claim unverifiable from the checkpoint.
            self.lift_in_weight = None
            self.lift_in_bias = None
            self.lift_out_weight = None
            self.lift_out_bias = None
            return

        if int(lift_hidden) <= 0:
            raise ValueError(f"lift_hidden must be > 0 on a lifted arm, got {lift_hidden}")
        hidden = int(lift_hidden)
        # A separate multilayer perceptron per channel, held as batched parameters rather than as
        # c_u separate modules: the arithmetic below is a per-channel matrix product with no
        # cross-channel term, which is what keeps the pointwise property true, and c_u small
        # modules would be the same parameters with a Python loop over them in every forward.
        self.lift_in_weight = nn.Parameter(torch.empty(self.c_u, hidden))
        self.lift_in_bias = nn.Parameter(torch.zeros(self.c_u, hidden))
        self.lift_out_weight = nn.Parameter(torch.empty(self.c_u, hidden, LIFT_WIDTH))
        self.lift_out_bias = nn.Parameter(torch.zeros(self.c_u, LIFT_WIDTH))
        self.reset_lift_parameters()

    # ------------------------------------------------------------------
    # Construction-time readouts
    # ------------------------------------------------------------------
    @property
    def out_width(self) -> int:
        r"""Coordinates per encoded coefficient: $2$, or $4$ on the lifted arm.

        Read by the fusion head to size its input projection, so the two cannot disagree about how
        wide a source vector is.

        Returns:
            The per-coefficient width.
        """
        return IDENTITY_WIDTH + (LIFT_WIDTH if self.scalar_lift else 0)

    @property
    def source_dim(self) -> int:
        r"""Width of one lag's flattened source vector: $C_U$ coefficients of :attr:`out_width`.

        Returns:
            The flattened per-lag width.
        """
        return self.c_u * self.out_width

    def has_parameters(self) -> bool:
        """Whether this encoder holds any learned parameter at all.

        Returns:
            ``False`` on the recommended arm, ``True`` on the scalar-lift arm.
        """
        return any(True for _ in self.parameters())

    def reset_lift_parameters(self) -> None:
        """Initialise the per-channel lift, or do nothing on an unlifted arm.

        Kept as a method, and called again from a composing model's post-initialisation block, for
        the reason every zeroing hook in this family is: the generic initialisation pass walks
        ``nn.Linear`` and ``nn.Conv1d`` and would leave these loose parameters wherever the
        constructor put them, so a model that *did* want them re-drawn has one call to make and a
        model that does not is unaffected either way.
        """
        if not self.scalar_lift:
            return
        assert self.lift_in_weight is not None  # narrowed by the flag
        assert self.lift_out_weight is not None
        assert self.lift_in_bias is not None
        assert self.lift_out_bias is not None
        # Fan-in of the first layer is one -- a scalar coefficient -- so a fan-based scheme
        # degenerates; a fixed unit scale is what a one-dimensional input wants, and the second
        # layer takes the usual fan-in scaling.
        nn.init.normal_(self.lift_in_weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.lift_in_bias)
        nn.init.normal_(
            self.lift_out_weight, mean=0.0, std=self.lift_out_weight.shape[1] ** -0.5
        )
        nn.init.zeros_(self.lift_out_bias)

    # ------------------------------------------------------------------
    # The encoding
    # ------------------------------------------------------------------
    def step_availability(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        r"""Which stored steps each channel has warmed up by: $\mathbb 1[s \ge W'_j]$.

        A function of the resolved warm-up vector and the step index alone -- no source value
        reaches it -- which is what lets the same tensor describe the availability of every
        recording in the dataset and lets a caller compute it once per batch.

        Args:
            sequence_length: $T$, the stored step count of the stream.
            device: Device to build the indicator on.

        Returns:
            A boolean $(T, C_U)$ indicator.
        """
        steps = torch.arange(int(sequence_length), device=device)[:, None]
        return steps >= self.warmup_vector.to(device)[None, :]

    def forward(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Encode a gated source stream coefficient by coefficient.

        Args:
            source: The gated source stream $(B, T, C_U)$, loader-normalized. This is the tensor
                the channel gate emitted, not the declared-width stream, because the warm-up vector
                is positional against the survivors.

        Returns:
            ``(encoded, mask)``: the per-coefficient representation
            $(B, T, C_U, \texttt{out\_width})$ and its boolean availability $(B, T, C_U)$. The
            mask travels separately as well as occupying a coordinate of the encoding, because the
            gather needs it as a mask and the fusion head needs it as a feature, and deriving one
            from the other at a call site is how the two come to disagree.

        Raises:
            ValueError: If the stream is not 3-D or its channel count is not the declared $C_U$;
                or if it carries a nonfinite value at a position the availability rules call
                valid, naming the first such batch element, step and channel. That is a data
                error: the alternative is feeding the model a fabricated zero observation on a
                channel that reported none, which no downstream readout could distinguish from a
                real standardized zero.
        """
        if source.dim() != 3:
            raise ValueError(
                f"source stream must be 3-D (B, T, c_u), got shape {tuple(source.shape)}"
            )
        if source.shape[-1] != self.c_u:
            raise ValueError(
                f"source stream has {source.shape[-1]} channels but this encoder declares "
                f"c_u={self.c_u}; the warm-up vector is positional against the channel axis, so a "
                f"mismatch would wait out the wrong channels rather than fail downstream."
            )

        available = self.step_availability(source.shape[1], source.device)  # (T, C_U)
        finite = torch.isfinite(source)

        # The refusal, before any arithmetic touches the values. Restricted to the region the
        # availability rules already call valid: the leading warm-up region legitimately holds
        # whatever the shard builder left there, and refusing on it would reject every real
        # recording.
        offending = available[None, :, :] & ~finite
        if bool(offending.any()):
            index = torch.nonzero(offending, as_tuple=False)[0].tolist()
            raise ValueError(
                f"source stream carries a nonfinite value at batch element {index[0]}, stored "
                f"step {index[1]}, channel {index[2]}, which the warm-up vector reports as "
                f"available. A nonfinite value in a declared-valid input is a data error, not an "
                f"implicit zero observation: normalising it to zero would present the model with "
                f"a standardized-zero coefficient that no readout could tell apart from a real "
                f"one. Reject the shard, or declare an explicit source-quality policy and version "
                f"the conditioning change it makes."
            )

        mask = available[None, :, :].expand_as(source)  # (B, T, C_U)
        # Substitution first, multiplication second. The unavailable region may hold nonfinite
        # values, and a nonfinite value multiplied by a zero mask is still nonfinite -- in the
        # forward and, worse, in every gradient that reaches it. ``torch.where`` replaces the value
        # rather than scaling it, so nothing nonfinite ever enters the graph.
        sanitised = torch.where(finite, source, torch.zeros_like(source))
        x_safe = sanitised * mask.to(source.dtype)
        # The capacity control, applied here rather than downstream so that no value ever enters
        # the graph: a head that never receives one cannot have learned from it, which is the whole
        # claim the arm is built to support.
        value = torch.zeros_like(x_safe) if self.withhold_values else x_safe

        parts = [value.unsqueeze(-1), mask.to(source.dtype).unsqueeze(-1)]
        if self.scalar_lift:
            parts.append(self._lift(x_safe))
        return torch.cat(parts, dim=-1), mask

    def gather(
        self,
        encoded: torch.Tensor,
        mask: torch.Tensor,
        anchors: torch.Tensor,
        *,
        n_lags: int,
        lag_floor: int = 0,
        lag_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Read the per-anchor per-lag window out of this encoder's own representation.

        A method rather than a free call at the composing model's site because the two source
        representations this package builds are gathered differently -- one carries a channel axis
        and one has mixed it away -- and the model must not be the place that knows which. It
        delegates to :func:`gather_lag_window`.

        Args:
            encoded: This encoder's own output.
            mask: The availability it returned beside it.
            anchors: The anchor index $(B, A)$.
            n_lags: How many lags this call gathers.
            lag_floor: $F_u$, the earliest stored step a lag may read.
            lag_offset: The first lag this call gathers.

        Returns:
            ``(window, window_mask)``.
        """
        return gather_lag_window(
            encoded,
            mask,
            anchors,
            n_lags=n_lags,
            lag_floor=lag_floor,
            lag_offset=lag_offset,
        )

    def _lift(self, x_safe: torch.Tensor) -> torch.Tensor:
        r"""The optional per-channel lift, $\phi_j(x^{\mathrm{safe}}_{s,j}) \in \mathbb R^2$.

        Each channel has its own two-layer map and reads its own coefficient only, so the lift
        leaves the pointwise property exactly where it was: no term here couples two channels or
        two stored times.

        Applied to $x^{\mathrm{safe}}$ rather than to the raw value, so an unavailable coefficient
        lifts $\phi_j(0)$ -- a per-channel constant, which the mask coordinate beside it is what
        distinguishes from the lift of a genuinely observed standardized zero. The gather zeroes
        every coordinate at a position it rules out, so that constant reaches the fusion head only
        where a real coefficient is present.

        Args:
            x_safe: The masked standardized coefficients $(B, T, C_U)$.

        Returns:
            The lift $(B, T, C_U, 2)$.
        """
        assert self.lift_in_weight is not None  # narrowed by the caller's flag check
        assert self.lift_in_bias is not None
        assert self.lift_out_weight is not None
        assert self.lift_out_bias is not None
        # (B, T, C_U, 1) * (C_U, hidden) broadcasts to (B, T, C_U, hidden): one scalar into each
        # channel's own hidden layer, with no sum over any axis, which is what makes this per
        # channel rather than a mixing layer wearing a per-channel name.
        hidden = torch.nn.functional.gelu(
            x_safe.unsqueeze(-1) * self.lift_in_weight + self.lift_in_bias
        )
        return (
            torch.einsum("btch,cho->btco", hidden, self.lift_out_weight)
            + self.lift_out_bias
        )


def lag_source_index(
    anchors: torch.Tensor,
    length: int,
    *,
    n_lags: int,
    lag_floor: int = 0,
    lag_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The stored step each anchor-lag pair reads, and whether that step exists.

    $$s_{t,\ell} = t - \ell, \qquad
      \mathbb 1\bigl[F_u \le s_{t,\ell} < T\bigr].$$

    Extracted so that the two gathers below decide range validity **once**, from one expression.
    Two copies of this arithmetic is how a per-channel gather and a per-state gather come to
    disagree about which lags exist, and the disagreement would be a wrong number rather than a
    failure: both shapes stay correct either way.

    The returned index is **clamped into the record**, so the gather it feeds is always legal, and
    every position the clamp rescued is ruled out by the validity flag beside it. A negative index
    left unclamped wraps to the end of the record and reads real future data.

    Args:
        anchors: The anchor index $(B, A)$, integer, in $[0, T)$.
        length: $T$, the stored length the index must stay inside.
        n_lags: How many lags this call covers.
        lag_floor: $F_u$, the earliest stored step a lag may read.
        lag_offset: The first lag this call covers, non-zero only under lag chunking.

    Returns:
        ``(safe_index, in_range)``: the clamped index $(B, A, L)$ and its boolean validity of the
        same shape.

    Raises:
        ValueError: If ``n_lags`` is not positive, if either offset is negative, if the anchor
            tensor is not $(B, A)$, or if an anchor lies outside $[0, T)$ -- which the clamp would
            otherwise turn into a legal read of the wrong stored step.
    """
    if int(n_lags) < 1:
        raise ValueError(f"n_lags must be >= 1, got {n_lags}")
    if int(lag_floor) < 0:
        raise ValueError(f"lag_floor must be >= 0, got {lag_floor}")
    if int(lag_offset) < 0:
        raise ValueError(f"lag_offset must be >= 0, got {lag_offset}")
    if anchors.dim() != 2:
        raise ValueError(f"anchors must be (B, A), got shape {tuple(anchors.shape)}")

    index = anchors.to(torch.long)
    if bool(((index < 0) | (index >= int(length))).any()):
        offending = int(index[(index < 0) | (index >= int(length))][0])
        raise ValueError(
            f"anchor {offending} is outside [0, T) = [0, {int(length)}). An anchor out of range "
            f"would be clamped by the surrogate index below and would then read a legal but wrong "
            f"stored step, which is a wrong number rather than a failure."
        )

    lags = torch.arange(
        int(lag_offset), int(lag_offset) + int(n_lags), device=anchors.device
    )
    source_index = index[:, :, None] - lags[None, None, :]  # (B, A, L)
    in_range = (source_index >= int(lag_floor)) & (source_index < int(length))
    return source_index.clamp(min=0, max=int(length) - 1), in_range


def gather_lag_window(
    encoded: torch.Tensor,
    mask: torch.Tensor,
    anchors: torch.Tensor,
    *,
    n_lags: int,
    lag_floor: int = 0,
    lag_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Read each anchor's lag window out of the encoded stream, safely.

    Anchor $t$ and lag $\ell$ read stored step $t - \ell$, and the full availability of that
    coefficient is

    $$m_{t,\ell,j}
      = \mathbb 1\bigl[F_u \le t - \ell < T\bigr]\;
        \mathbb 1\bigl[t - \ell \ge W'_j\bigr]\;
        v^U_{t-\ell,j},$$

    of which the encoder supplied the last two factors and this function supplies the first.

    **Validity is decided before the gather, and the gather then runs on a surrogate index.** An
    out-of-range source step has no legal index to read, and the two obvious ways to handle that
    both fail silently: a negative index wraps to the end of the record and gathers real *future*
    data with every shape correct, and a boolean-indexed gather makes the output shape a function
    of the data, which no distributed run survives. So the index is clamped into range, the gather
    always reads something legal, and every coordinate the combined mask rules out is zeroed
    afterwards -- including the mask coordinate, so that a bit set in the returned encoding always
    means a coefficient genuinely present at that anchor, lag and channel.

    **Index support and feature warm-up stay separate.** A lag can be perfectly in range while the
    channel it would read has not warmed up: those are different conditions, they fail at different
    anchors, and the per-lag exposure readout downstream has to be able to tell them apart. This
    function returns their conjunction, and the two factors that made it are recoverable because
    the caller holds the encoder's own mask.

    Args:
        encoded: The per-coefficient representation $(B, T, C_U, W)$ from
            :meth:`PointwiseSourceEncoder.forward`.
        mask: Its boolean availability $(B, T, C_U)$, from the same call.
        anchors: The anchor index $(B, A)$, integer, in $[0, T)$. Any anchor subset is admissible,
            which is what lets a caller chunk the anchor axis without this function knowing.
        n_lags: How many lags this call gathers, so they run over
            $\texttt{lag\_offset}, \ldots, \texttt{lag\_offset} + n\_lags - 1$. It is the full $L$
            for an unchunked call and the chunk width otherwise.
        lag_floor: $F_u$, the earliest stored step a lag may read. Ships at $0$, where the
            condition is the plain in-range one.
        lag_offset: The first lag this call gathers. Non-zero only when the caller is chunking the
            lag axis, and the reason this is an argument rather than a slice of a full gather: the
            point of chunking is never to build the full window at all.

    Returns:
        ``(window, window_mask)``: the gathered representation $(B, A, L, C_U, W)$ and its boolean
        availability $(B, A, L, C_U)$.

    Raises:
        ValueError: If the shapes disagree, if ``n_lags`` is not positive, if ``lag_floor`` is
            negative, or if an anchor lies outside $[0, T)$ -- naming the offending value, because
            an out-of-range anchor is the one input to this function whose consequence is a
            gathered wrong step rather than an exception.
    """
    if encoded.dim() != 4:
        raise ValueError(
            f"encoded must be 4-D (B, T, C_U, W), got shape {tuple(encoded.shape)}"
        )
    if mask.shape != encoded.shape[:3]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match the encoding's leading axes "
            f"{tuple(encoded.shape[:3])}; the two come from one call and a mismatch means one of "
            f"them describes a different batch."
        )
    batch, length = encoded.shape[0], encoded.shape[1]
    if anchors.shape[0] != batch:
        raise ValueError(
            f"anchors must be (B, A) with B={batch}, got shape {tuple(anchors.shape)}"
        )
    # The surrogate index and its validity, decided before anything is read.
    safe_index, in_range = lag_source_index(
        anchors, length, n_lags=n_lags, lag_floor=lag_floor, lag_offset=lag_offset
    )

    n_anchors = anchors.shape[1]
    channels, width = encoded.shape[2], encoded.shape[3]
    flat = safe_index.reshape(batch, -1)  # (B, A * L)
    window = encoded.gather(
        1, flat[:, :, None, None].expand(-1, -1, channels, width)
    ).reshape(batch, n_anchors, int(n_lags), channels, width)
    window_mask = mask.gather(
        1, flat[:, :, None].expand(-1, -1, channels)
    ).reshape(batch, n_anchors, int(n_lags), channels)

    window_mask = window_mask & in_range[..., None]
    # Every coordinate, not only the value coordinate: at a ruled-out position the surrogate index
    # read a real coefficient of a legal step, and its availability bit came back set. Leaving that
    # bit would tell the fusion head a coefficient is present where none is.
    return window * window_mask.unsqueeze(-1).to(window.dtype), window_mask


def gather_lag_state(
    state: torch.Tensor,
    mask: torch.Tensor,
    anchors: torch.Tensor,
    *,
    n_lags: int,
    lag_floor: int = 0,
    lag_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Read each anchor's lag window out of a **stream-wide state**, and its availability beside it.

    The counterpart of :func:`gather_lag_window` for a representation that has already mixed the
    channel axis away -- a convolution stem's output, whose width is the model's rather than the
    source's. It reads the same stored step under the same surrogate index, so the two gathers
    cannot disagree about which lags exist.

    **The availability travels separately and is still per channel**, which is the point. Whether a
    channel has warmed up is a property of that channel and that stored step, and it does not stop
    being one because an encoder mixed the values together: the per-lag exposure readout has to
    report the same counts whichever representation produced the window, or two arms' exposure
    tables are not comparable. The state itself is zeroed wherever no channel is available at all,
    so a lag with nothing behind it contributes an exact zero rather than the stem's response to
    whatever the leading region of the record happened to hold.

    Args:
        state: The stream-wide representation $(B, T, d)$.
        mask: Per-channel availability $(B, T, C_U)$, from the pointwise encoder that resolved it.
        anchors: The anchor index $(B, A)$.
        n_lags: How many lags this call gathers.
        lag_floor: $F_u$, the earliest stored step a lag may read.
        lag_offset: The first lag this call gathers, non-zero only under lag chunking.

    Returns:
        ``(window, window_mask)``: the gathered state $(B, A, L, d)$ and the per-channel
        availability $(B, A, L, C_U)$ behind it.

    Raises:
        ValueError: If the state is not 3-D, if the mask's leading axes disagree with it, or on
            anything :func:`lag_source_index` refuses.
    """
    if state.dim() != 3:
        raise ValueError(f"state must be 3-D (B, T, d), got shape {tuple(state.shape)}")
    if mask.shape[:2] != state.shape[:2]:
        raise ValueError(
            f"mask leading axes {tuple(mask.shape[:2])} disagree with the state's "
            f"{tuple(state.shape[:2])}; the two describe one batch of one recording set."
        )

    batch, length, width = state.shape
    if anchors.shape[0] != batch:
        raise ValueError(
            f"anchors must be (B, A) with B={batch}, got shape {tuple(anchors.shape)}"
        )
    safe_index, in_range = lag_source_index(
        anchors, length, n_lags=n_lags, lag_floor=lag_floor, lag_offset=lag_offset
    )

    n_anchors, channels = anchors.shape[1], mask.shape[2]
    flat = safe_index.reshape(batch, -1)  # (B, A * L)
    window = state.gather(1, flat[:, :, None].expand(-1, -1, width)).reshape(
        batch, n_anchors, int(n_lags), width
    )
    window_mask = mask.gather(
        1, flat[:, :, None].expand(-1, -1, channels)
    ).reshape(batch, n_anchors, int(n_lags), channels)
    window_mask = window_mask & in_range[..., None]
    return window * lag_validity(window_mask).unsqueeze(-1).to(window.dtype), window_mask


def lag_validity(window_mask: torch.Tensor) -> torch.Tensor:
    r"""Whether each anchor-lag pair carries any available channel at all.

    $$v_{t,\ell} = \mathbb 1\Bigl[\textstyle\sum_j m_{t,\ell,j} > 0\Bigr].$$

    The factor the fusion head multiplies a proposal by, so a lag whose every channel is out of
    range or still cold contributes an exact zero rather than the head's response to an all-zero
    input vector -- which is a learned constant, not nothing.

    Args:
        window_mask: The gathered availability $(B, A, L, C_U)$.

    Returns:
        A boolean $(B, A, L)$ indicator.
    """
    return window_mask.any(dim=-1)


__all__ = [
    "IDENTITY_WIDTH",
    "LIFT_HIDDEN",
    "LIFT_WIDTH",
    "PointwiseSourceEncoder",
    "gather_lag_state",
    "gather_lag_window",
    "lag_source_index",
    "lag_validity",
]
