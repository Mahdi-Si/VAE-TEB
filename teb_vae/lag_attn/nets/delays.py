r"""Per-channel time shift: read channel $c$ at step $t - \delta_c$ instead of at step $t$.

The input features are two-sided wavelet transforms, so the value stored at decimated step $t$
already contains raw signal from *after* $t$ -- up to $974$ s after it, for the lowest-frequency
channel. Any predictive quantity read off a model fed those features is therefore partly a
measurement of the model reading its own answer. Delaying channel $c$ by
$\delta_c = \lceil \mathrm{reach}_c / \Delta \rceil$ steps ($\Delta = 4$ s per step) moves its
forward reach back behind the anchor's causal endpoint, so that

$$
\underbrace{\Delta\,(t - \delta_c + 1) - \Delta}_{\text{end of the delayed step}}
  + \mathrm{reach}_c \;\le\; \Delta\,(t+1) - \Delta ,
$$

which is exactly $\mathrm{reach}_c \le \Delta\,\delta_c$.

Per channel rather than one uniform guard band, because the two dominate very differently at
equal guarantee: at a $120$ s cap the fastest channels are only $1$ step stale while a uniform
band would make every channel $30$ steps stale. The cost is that the module needs a vector, not
a scalar, and that the first $\max_c \delta_c$ steps of the sequence are partly zeroed -- which
is why the guard requires $\max_c \delta_c \le w$, the loss warm-up.

Two classes, one wrapping the other. :class:`ChannelDelay` is the shift alone.
:class:`ChannelGate` is the whole guard as one object -- select the channels that survive the
budget, then delay each survivor -- so a consumer holds a single handle with a single width and a
single maximum delay, rather than a buffer and a submodule it must keep paired by hand.

Pure torch, and deliberately ignorant of where the numbers came from: this module receives
concrete integers and knows nothing about filter banks, reaches or budgets. That separation is
what lets the delay be tested as an index operation rather than as a claim about wavelets.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class ChannelDelay(nn.Module):
    r"""Shift each channel back in time by its own integer number of steps.

    Output step $t$ of channel $c$ is input step $t - \delta_c$ of channel $c$; steps before the
    channel's delay have no source and are emitted as zero. With every $\delta_c = 0$ the module
    is a bitwise identity, which is what makes the unguarded configuration exactly the model
    that has no delay at all.
    """

    #: Declared so the registered buffer types as a tensor rather than as ``Tensor | Module``.
    delay_steps: torch.Tensor

    def __init__(self, *, num_channels: int, delays: Sequence[int]) -> None:
        r"""Initialize the delay.

        Args:
            num_channels: Channel count $C$ of the tensors this will be applied to.
            delays: One non-negative delay $\delta_c$ per channel, in decimated steps.

        Raises:
            ValueError: If ``num_channels`` is not positive, if ``delays`` has a different
                length (which would silently delay the wrong channels -- the delay vector is
                positional, and nothing downstream would fail), or if any delay is negative (a
                negative delay reads the channel's *future*, the opposite of the guard's
                purpose).
        """
        super().__init__()

        num_channels = int(num_channels)
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")

        delay_values = [int(value) for value in delays]
        if len(delay_values) != num_channels:
            raise ValueError(
                f"delays has {len(delay_values)} entries but num_channels is {num_channels}. "
                f"The delay vector is positional -- one entry per surviving channel, in channel "
                f"order -- so a length mismatch would delay the wrong channels with no other "
                f"failure signal."
            )
        negative = [(index, value) for index, value in enumerate(delay_values) if value < 0]
        if negative:
            raise ValueError(
                f"delays must be >= 0; got negative entries at "
                f"{negative}. A negative delay reads a channel from its own future, which is "
                f"the leak this module exists to remove."
            )

        self.num_channels = num_channels
        # Non-empty by construction: num_channels >= 1 and the length check above agree.
        self.max_delay = max(delay_values)

        # Non-persistent: the vector's *length* is the surviving-channel count, which changes
        # with the configured reach budget. A persistent buffer would put a budget-shaped tensor
        # in every checkpoint, and a checkpoint trained at one budget would then fail to load at
        # another as "checkpoint keys did not align" rather than as a message about the budget.
        self.register_buffer(
            "delay_steps", torch.tensor(delay_values, dtype=torch.long), persistent=False
        )

    def extra_repr(self) -> str:
        """Summarise the delay for ``print(model)``."""
        return f"num_channels={self.num_channels}, max_delay={self.max_delay}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the per-channel delay.

        Args:
            x: Feature stream $(B, T, C)$.

        Returns:
            The delayed stream, same shape and dtype. Position $(b, t, c)$ holds
            $x[b,\, t - \delta_c,\, c]$, or $0$ when $t < \delta_c$.

        Raises:
            ValueError: If ``x`` is not $(B, T, C)$ with $C$ equal to ``num_channels``.
        """
        if x.dim() != 3 or x.shape[-1] != self.num_channels:
            raise ValueError(
                f"expected a (B, T, {self.num_channels}) stream, got {tuple(x.shape)}"
            )

        # (T, C) source index. Built per forward rather than cached: T is a property of the
        # batch, not of the module, and an arange is far cheaper than the gather it feeds.
        steps = torch.arange(x.shape[1], device=x.device).unsqueeze(-1)
        source_index = steps - self.delay_steps.unsqueeze(0)
        available = source_index >= 0

        # gather does not broadcast, so the index is expanded to the batch -- a view, not a copy.
        # Clamped because an out-of-range index is an error even where the result is discarded.
        gathered = x.gather(
            1, source_index.clamp_min(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        )
        return gathered * available.to(x.dtype)


class ChannelGate(nn.Module):
    r"""The causal input guard for one stream: keep the surviving channels, delay each one.

    One object rather than a keep-index beside a delay module, because the two are meaningless
    apart: the delay vector is positional *against* the keep-index, so anything holding one must
    hold the other, and a consumer that wants the stream's emitted width or its worst delay
    should not have to know which of the pair to ask.

    The gather runs before the delay, so the delays are indexed by surviving channel rather than
    by declared channel.
    """

    #: Declared so the registered buffer types as a tensor rather than as ``Tensor | Module``.
    keep_index: torch.Tensor

    def __init__(
        self,
        *,
        declared_width: int,
        keep_index: Optional[Sequence[int]] = None,
        delays: Optional[Sequence[int]] = None,
    ) -> None:
        r"""Initialize the gate.

        A missing ``keep_index`` becomes the identity and a missing ``delays`` becomes zeros, so
        either argument alone is enough to build a gate. Both missing is *not* handled here: that
        is the unguarded case, and it is represented by having no gate at all rather than by an
        identity one.

        Args:
            declared_width: The stream's full declared channel count, which the keep-index
                indexes into.
            keep_index: Surviving channel indices, strictly ascending. ``None`` keeps all.
            delays: One delay in decimated steps per survivor, in the same order. ``None``
                means no delay.

        Raises:
            ValueError: If the keep-index is empty, has entries outside
                $[0, \mathrm{declared\_width})$, or is not strictly ascending -- the delay vector
                is positional against it, so a reordered index would delay the wrong channels
                with no other failure signal. Also if the delay vector's length disagrees, which
                :class:`ChannelDelay` raises.
        """
        super().__init__()

        width = int(declared_width)
        indices = list(range(width)) if keep_index is None else [int(i) for i in keep_index]
        if not indices:
            raise ValueError(
                "keep_index is empty: the model would train to completion having never read "
                "this stream."
            )
        outside = [index for index in indices if index < 0 or index >= width]
        if outside:
            raise ValueError(f"keep_index has entries outside [0, {width}): {outside}")
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(
                "keep_index must be strictly ascending; the delay vector is positional against "
                "it, so a reordered index would delay the wrong channels."
            )

        self.declared_width = width
        # Non-persistent for the same reason ChannelDelay's buffer is: its length is the
        # surviving-channel count, so a persistent copy would make a checkpoint trained at one
        # reach budget fail to load at another, reported as misaligned keys rather than as a
        # budget mismatch.
        self.register_buffer(
            "keep_index", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.delay = ChannelDelay(
            num_channels=len(indices),
            delays=[0] * len(indices) if delays is None else delays,
        )

    @property
    def out_channels(self) -> int:
        """Channels this gate emits, which is what the downstream adapter must be built for."""
        return int(self.keep_index.numel())

    @property
    def max_delay(self) -> int:
        r"""The worst survivor's delay $\max_c \delta_c$, in decimated steps."""
        return self.delay.max_delay

    def extra_repr(self) -> str:
        """Summarise the gate for ``print(model)``."""
        return f"{self.declared_width} -> {self.out_channels}, max_delay={self.max_delay}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Gather the surviving channels and delay each by its own $\delta_c$.

        Args:
            x: The full-width stream $(B, T, C_{\mathrm{declared}})$.

        Returns:
            The gated stream $(B, T, C_{\mathrm{kept}})$.
        """
        return self.delay(torch.index_select(x, -1, self.keep_index))
