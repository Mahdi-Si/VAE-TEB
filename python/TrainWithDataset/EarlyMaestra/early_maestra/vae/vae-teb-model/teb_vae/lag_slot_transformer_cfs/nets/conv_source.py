r"""The convolutional source representation: the comparator arm's stem, and only that.

This module exists to be **compared against**, not to be recommended. The recommended source
representation is pointwise: each standardized coefficient becomes its own value-and-availability
pair and nothing mixes two stored times before the fusion boundary. The stem here is the thing that
representation replaced -- a gated causal convolution stack over the source stream, whose output at
$t$ is a summary of a bounded window ending at $t$ rather than of $t$ alone.

**Why a package that argues against it builds it anyway.** A predictive difference between two
trained models says nothing about *which* of their differences produced it. The evaluated
lag-attentive forecaster differed from the recommended architecture in the source stem, in the lag
fusion, in the residual parameterisation, in the metadata encoding and in the sampling policy all at
once, so a comparison against it measures that package. Holding four of those fixed and moving the
stem alone is what makes "removing the source temporal convolution changed the predictive gap by
this much" a statement about the stem. Everything downstream of this module -- the prior, the
residual equations and their bounds, the shared decoder, the paired noise, the objective and its
reduction -- is the same code the recommended arm runs.

**What the stem does and does not change about causality.** Its output at $t$ reaches back
$R_U = 1 + \sum_i (k_i - 1) d_i$ stored steps, itself included, and no further: it is causal, and
the additional neural source receptive field it introduces is that number rather than one. Two lags
closer together than $R_U$ are therefore summaries of overlapping windows, which is the resolution
floor of any lag readout taken over this representation and is reported as such. It does not reach
forward, and it does not undo the raw-to-feature mixing upstream of the model, which is far wider
than either arm's neural reach and which no model-side change touches.

**Availability stays per channel even though the values no longer are.** Whether a source channel
has warmed up at a stored step is a property of the channel and the step. The stem mixes the values;
it does not make the announcement disappear, and the exposure readout has to report the same counts
for both arms or two arms' exposure tables are not comparable. So this module holds a parameter-free
pointwise encoder purely for its availability resolution, feeds the stream through the family's
availability adapter -- which is what tells the stem *which* leading region is not yet signal -- and
carries the per-channel mask alongside the state it gathers.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.encoders import (
    AvailabilityInputAdapter,
    GatedCausalConvStem,
)
from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import (
    PointwiseSourceEncoder,
    gather_lag_state,
)


class ConvSourceStem(nn.Module):
    r"""Encode the source stream with a bounded causal convolution stack.

    The interface is deliberately the pointwise encoder's: ``forward`` returns
    ``(encoded, mask)`` and ``gather`` reads one anchor's lag window out of it, so the composing
    model calls the same two methods whichever representation it was configured with and holds no
    opinion about which. What differs is the shape of the encoding -- a model-width state per
    stored step rather than a per-coefficient pair -- and the fact that this one **has parameters**
    and mixes stored times, which is the whole declared difference between the two arms.

    Attributes:
        c_u: Source channel count of the stream this stem reads, as the gate presents it.
        d_model: Width of the state it emits, which is also its per-lag vector width.
        availability: The parameter-free encoder that resolves per-channel warm-up, held for its
            mask alone.
        adapter: The availability adapter that projects the stream and announces its warm-up.
        stem: The gated causal convolution stack.
    """

    def __init__(
        self,
        *,
        c_u: int,
        d_model: int,
        adapter: AvailabilityInputAdapter,
        conv_kernels: Sequence[int],
        conv_dilations: Sequence[int],
        warmup_steps: Optional[Sequence[int]] = None,
    ) -> None:
        r"""Hold the adapter, build the stem, and resolve the availability both are read against.

        The adapter is **handed in rather than built here**, because resolving which availability
        terms a stream announces is the composing model's job and it is not a simple one: the term
        is the channel's warm-up wait combined with any shift the gate applied, and a copy of that
        resolution living here is a copy that can fall out of step with the one the target stream
        uses.

        Args:
            c_u: Source channel count of the gated stream.
            d_model: Model width $d_h$, held end to end through the stem.
            adapter: The stream's availability adapter, already resolved at its own delays. It
                projects the stream to ``d_model`` and announces which leading region is not yet
                signal, which is what tells the stem what it is convolving over.
            conv_kernels: Kernel width per stem block.
            conv_dilations: Dilation per stem block, parallel to ``conv_kernels``.
            warmup_steps: $W'_j$ per channel, in the stream's own channel order, or ``None`` for a
                stream with no warm-up.

        Raises:
            ValueError: On any width or schedule the underlying modules refuse.
        """
        super().__init__()
        self.c_u = int(c_u)
        self.d_model = int(d_model)

        # Parameter-free, and held for the availability rule alone: the warm-up condition and the
        # accepted-finite refusal are properties of the stream rather than of an encoder, and
        # resolving them twice is how two arms come to disagree about which lags exist.
        self.availability = PointwiseSourceEncoder(
            c_u=self.c_u, warmup_steps=warmup_steps, scalar_lift=False
        )
        self.adapter = adapter
        # No dropout inside the stem, and not as an oversight: the source pathway carries none on
        # any arm here, so that invoking one decoder twice under one latent draw gives exactly
        # paired branches in training mode as well as in evaluation mode.
        self.stem = GatedCausalConvStem(
            d_model=self.d_model,
            conv_kernels=conv_kernels,
            conv_dilations=conv_dilations,
            dropout=0.0,
        )

    @property
    def source_dim(self) -> int:
        r"""Width of one lag's source vector: the model width $d_h$.

        Named to match the pointwise encoder's property of the same name, because the fusion head
        sizes its input projection from it and the two must not be able to disagree.

        Returns:
            The per-lag width.
        """
        return self.d_model

    @property
    def receptive_field(self) -> int:
        r"""Stored steps of source history the state at $t$ reaches, itself included.

        The resolution floor of a lag readout taken over this representation, and the number the
        recommended arm reduces to one. Read off the stem rather than recomputed, so a run at
        another schedule discloses its own.

        Returns:
            $R_U$, in stored steps.
        """
        return int(self.stem.receptive_field)

    def has_parameters(self) -> bool:
        """Whether this encoder holds any learned parameter.

        Always ``True`` here, unlike the recommended arm's, and that is the declared difference
        rather than an implementation detail.

        Returns:
            ``True``.
        """
        return any(True for _ in self.parameters())

    def forward(self, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Announce the stream's availability, project it, and run the causal stack.

        The availability pass runs first and is the same one the pointwise arm runs, so a nonfinite
        value at a position the warm-up rules call valid is refused here exactly as it is there --
        before any convolution has spread it across a whole window.

        Args:
            source: The gated source stream $(B, T, C_U)$, loader-normalized.

        Returns:
            ``(state, mask)``: the stem's output $(B, T, d_h)$ and the per-channel availability
            $(B, T, C_U)$ that was resolved before it ran.

        Raises:
            ValueError: On a stream the availability pass refuses -- a wrong channel count, or a
                nonfinite value where a channel reports itself available.
        """
        encoded, mask = self.availability(source)
        # The sanitised values rather than the raw stream: the availability pass has already
        # replaced everything the mask rules out, and a convolution over a nonfinite leading region
        # would spread it across every window that touches it. Coordinate zero is the value; the
        # coordinate beside it is the availability bit, which the adapter announces separately.
        return self.stem(self.adapter(encoded[..., 0])), mask

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
        r"""Read the per-anchor per-lag window out of the stem's state.

        Args:
            encoded: The stem's output $(B, T, d_h)$.
            mask: The per-channel availability beside it.
            anchors: The anchor index $(B, A)$.
            n_lags: How many lags this call gathers.
            lag_floor: $F_u$, the earliest stored step a lag may read.
            lag_offset: The first lag this call gathers.

        Returns:
            ``(window, window_mask)``, the state $(B, A, L, d_h)$ and the availability
            $(B, A, L, C_U)$ behind it.
        """
        return gather_lag_state(
            encoded,
            mask,
            anchors,
            n_lags=n_lags,
            lag_floor=lag_floor,
            lag_offset=lag_offset,
        )

    def extra_repr(self) -> str:
        """Report the widths and the reach, which is what tells the two arms apart."""
        return (
            f"c_u={self.c_u}, d_model={self.d_model}, "
            f"receptive_field={self.receptive_field} steps"
        )


__all__ = ["ConvSourceStem"]
