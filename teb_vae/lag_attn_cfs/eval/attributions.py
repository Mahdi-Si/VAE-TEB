r"""Captum attributions of what happens inside a causal-feature forecaster: the shared core.

Every other readout of the lag structure in this family is either **observational** -- the
attention over lags, the KL attribution built from it, the proposal norm -- or **interventional**
on one axis, the lag band a source stream is zeroed on. Neither says *which input coefficients, at
which stored steps and which channels, drove* a given per-anchor quantity. This module answers that
with gradient attribution: a thin :class:`torch.nn.Module` wrapper turns the model's dense forward
into one scalar per sample at one chosen anchor -- the divergence $K_t$, the mean-decoded block
score of either branch and their gap, one latent coordinate, or the model's own lag readout on a
band -- and Captum attributes that scalar back over the three input streams $(y^{st}, y^{ph}, u)$
at their **declared** widths on the stored grid.

Two cells go through this module, exactly as they go through
:mod:`~teb_vae.lag_attn_cfs.eval.traces`: the lag-attentive cells, whose latent tensors are dense
over $T$ and whose lag readout is an attention distribution, and the lag-residual cell, whose
latent tensors live on the anchor axis and whose only per-lag magnitude is a proposal norm. A
:class:`CellBinding` names which is which; the wrapper, the baselines, the reductions and the
figures are shared, and nothing here touches a loader or a table.

**Which quantities are attributed, and on which branch.** Means, never samples: the readouts are
functions of $(\mu^p, \ell^p, \mu^q, \ell^q)$ and of the decoder applied to $\mu$ -- the
mean-decoded block score the collection pass reports as ``mean_pred_gap`` -- so no
reparameterisation draw enters any attributed number and two runs agree bitwise. A readout defined
on the sampled branch would need a frozen $\epsilon$ and would attribute the noise as well as the
input, which no figure here could separate. The model's own forward is what is differentiated,
under :func:`attributed_forward`: it decodes each row's anchor alone and at the two latent means,
so its own forecasts are the mean-decoded ones, it draws no $\epsilon$ at all, and it no longer
spends most of every integration step decoding anchors no readout reads.

**What is differentiated, per baseline.** Under the source-null baseline the target streams sit
at their own values at both ends of the path, so only the source is handed to Captum and the
target maps are the exact zeros they would be anyway; the backward then never enters the target
encoder. The same holds for the lag-band ablation, which perturbs the source alone.

**The baselines are a modelling decision, and there are two.** Under the loader's z-scoring an
exact zero is the channel mean over the region the model reads -- the climatology baseline the
pipeline already uses -- and the input warm-up gate already multiplies every not-yet-warm step by
exactly zero, so a zero baseline changes nothing on the steps the model never read.
:data:`BASELINE_SOURCE_NULL` zeroes the source and holds both target streams fixed: it is the
exact null arm ``source_null`` measures, so an attribution of $K_t$ or ``pred_gap`` to the source
under it is an attribution to source *content* -- the availability announcement is a constant of
$t$, is identical on both ends of the path, and cannot be attributed to any input at all.
:data:`BASELINE_ALL_ZERO` zeroes every stream and is the reference on which the target streams'
own attribution is read.

**The path enters the baseline from the side, and the record says by how much.** Every encoder in
the family normalises per step -- a causal group norm on the conv-LSTM cell, a root-mean-square
norm on the transformer cells -- and an exactly-zero stream is a degenerate point of that
normalisation: on every tiny fixture the readout jumps by a finite amount between $\alpha = 0$ and
$\alpha = 10^{-6}$ along the straight path and the directional derivative at $\alpha = 0$ is of
order $10^{14}$ to $10^{22}$, so integrated gradients from the exact zero do not converge at any
step count. The path therefore starts at $x_0 = b + \alpha_0 (x - b)$ with
$\alpha_0 =$ :data:`BASELINE_ENTRY_FRACTION`, where the integral converges to a relative error
of order $10^{-3}$ at :data:`IG_STEPS` steps on the fixtures -- below it on most, but the
transformer ``entmax15`` fixture measured $1.3\times10^{-3}$, so the residual is reported per row
rather than assumed; the readout at the exact baseline, at
the entry point and at the input all travel on every row, so the **entry jump**
$f(x_0) - f(b)$ is a reported scalar rather than a hidden one. It belongs to no input step: it is
the normalisation snapping out of its degenerate state. The one measured exception is the
lag-residual model's source-null path under its default ``pointwise`` source stem, which carries no
temporal normalisation and was smooth at zero; its all-zero path still passes through the target
encoder's norm and is degenerate like every other. The entry point is applied to every path
regardless.

**Structural properties every attribution here is checked against**, and that the tests assert on
the tiny models: attribution to any stored step later than the anchor is exactly zero; attribution
to a source step a channel has not warmed up at is exactly zero; the source attribution of a
target-only readout ($\mu^p$, the base block score) is exactly zero; and the integrated-gradient
sum reproduces $f(x) - f(x_0)$ within tolerance. The lag axis of every profile here is
**stored-coefficient time**, and every figure prints :data:`ATTRIBUTION_NOTE`, the one-line form of
the caveat; the full sentences stay in the records and the guide.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from captum.attr import FeatureAblation, IntegratedGradients, LayerIntegratedGradients
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from torch import nn

from teb_vae.lag_attn.figure_primitives import sample_cell_edges
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import cohort, events, lag_hist, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import band_partition, labels
from teb_vae.lag_attn_cfs.eval.lag_axis import COEFFICIENT_LAG_AXIS_LABEL
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, forecast_likelihood_terms
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor, raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

#: This analysis's own subdirectory inside a results directory, and the files it writes.
ANALYSIS_DIRNAME = "attribution"
ROWS_FILENAME = "attribution_rows.csv"
VECTORS_FILENAME = "attribution_vectors.npz"
MAPS_FILENAME = "attribution_maps.npz"
RECORDINGS_FILENAME = "attribution_recordings.csv"
SUMMARY_FILENAME = "attribution_summary.csv"
BANDS_FILENAME = "attribution_bands.csv"
LAG_BANDS_FILENAME = "attribution_lag_bands.csv"
LAYER_FILENAME = "attribution_layer.csv"
NULL_FILENAME = "attribution_null.csv"
TRACE_DIRNAME = "traces"
TRACE_SUFFIX = "_attribution_trace"

#: The fixed-name figures, as stems.
MAP_FIGURE = "attribution_maps"
LAG_PROFILE_FIGURE = "attribution_lag_profile"
BAND_FIGURE = "attribution_bands"
LAYER_FIGURE = "attribution_layer"
NULL_FIGURE = "attribution_null"
CHANNEL_FIGURE = "attribution_channels"
LAG_CHANNEL_FIGURE = "attribution_lag_channel"
TIME_PROFILE_FIGURE = "attribution_time_profile"
CHECKS_FIGURE = "attribution_checks"
DELIVERY_FIGURE = "attribution_time_to_delivery"
BLOCK_FIGURE = "attribution_blocks"
HORIZON_FIGURE = "attribution_horizon"

#: The per-block table: per readout, baseline and input block, the recording-mean signed and
#: unsigned attribution and the unsigned share of the block in the row's total.
BLOCKS_FILENAME = "attribution_blocks.csv"

#: The population lag-by-channel maps, mean over attributed anchors of the signed and unsigned
#: attribution re-indexed by offset from the anchor, per main readout, baseline and stream.
LAG_CHANNEL_FILENAME = "attribution_lag_channel.npz"

#: ``eval_config.caps`` name bounding how many **segments** are attributed. Absent means
#: :data:`DEFAULT_SEGMENTS` rather than every segment: an attribution is tens of forwards and
#: backwards per anchor, so an uncapped pass over a split would cost more than the collection pass.
CAP_NAME = "attribution_segments"
DEFAULT_SEGMENTS = 24

#: Anchors attributed per segment, spread evenly over the segment's scored anchors so the maps
#: cover its early, middle and late phases rather than one draw of them.
ANCHORS_PER_SEGMENT = 4

#: The informative-anchor rule (``attribution_pass.informative_anchors``). An anchor qualifies when
#: its $K_t$ is in the upper $30\%$ of the KL **of its own segment**, and when its signal is clean:
#: the forecast coverage and the mean validity over the searched lag window both reach
#: :data:`CLEAN_COVERAGE`. The threshold is per segment rather than pooled over the cohort, so a
#: subgroup whose coupling is weak everywhere still contributes its most coupled moments instead of
#: no anchor at all; a pooled threshold dropped such a subgroup from every by-subgroup figure.
HIGH_KL_QUANTILE = 0.7
CLEAN_COVERAGE = 0.95

#: Config key of the main pass's segments per drawn recording, and its default. One keeps the
#: original draw; more spreads that many segments evenly over each recording's stored timeline, so
#: a recording's summary rests on several 20-minute windows rather than on one.
SEGMENTS_PER_RECORDING_CAP_NAME = "attribution_segments_per_recording"
DEFAULT_SEGMENTS_PER_RECORDING = 1

#: Example pages per class: the highest-KL clean anchors, one per recording.
EXAMPLES_PER_CLASS = 3

#: Recordings followed through every one of their segments **per class**, for the trace figure. One,
#: because each is a few hundred attributions; the one chosen is the class's most **complete**
#: recording over the window the run reads its clocks over -- the fewest missing segments -- so
#: the figure shows an evolution rather than the holes in it.
TRACE_RECORDINGS_PER_CLASS = 1

#: The readouts the per-recording trace attributes at its anchors: the latent change and the
#: forecast gain, so a recording's trace shows where the source moved the belief and where it
#: helped the forecast, on the same anchors.
TRACE_READOUTS: Tuple[str, ...] = ("kld", "pred_gap")

#: The readouts attributed at one **example** anchor per class, with their full maps kept for the
#: example pages: the divergence, the forecast gap and the full-branch block score -- the latent
#: change, the gain, and the output score itself -- beside the lag readout on every configured
#: band, which the pass adds. Attributed under both baselines at that one anchor, so the target
#: map (which only the all-zero path moves) and the source map (the source-null comparison) are
#: both on the page.
EXAMPLE_READOUTS: Tuple[str, ...] = ("kld", "pred_gap", "nll_full", "mse_full", "mse_gap")

#: Where the example pages go, under the analysis directory, and the tail of their names.
EXAMPLE_DIRNAME = "maps"
EXAMPLE_SUFFIX = "_attribution_maps"

#: Integrated-gradient steps and Captum's internal batch of interpolated inputs, in rows times
#: steps per forward. The step count is the one at which the completeness residual fell to order
#: $10^{-3}$ of the readout on the fixtures with the entry fraction below -- not below it on every
#: one: the transformer ``entmax15`` fixture measured $1.3\times10^{-3}$ (the conv-LSTM cell's
#: per-step group norm makes its path the roughest of the three); the residual is recorded per row
#: so a production run can say what it reached there.
IG_STEPS = 64
IG_INTERNAL_BATCH_SIZE = 64

#: Where along the straight path the integration starts; see the module docstring.
BASELINE_ENTRY_FRACTION = 1e-3

#: Offset applied to the run's seed for this analysis's segment draw, distinct from every other
#: draw's in the pipeline.
DRAW_SEED_OFFSET = 13

#: The two baselines.
BASELINE_SOURCE_NULL = "source_null"
BASELINE_ALL_ZERO = "all_zero"
BASELINES: Tuple[str, ...] = (BASELINE_SOURCE_NULL, BASELINE_ALL_ZERO)

#: The readouts the wrapper can turn into one scalar per anchor.
READOUT_KLD = "kld"
READOUT_KLD_DIM = "kld_dim"
READOUT_MU_POST_DIM = "mu_post_dim"
READOUT_MU_PRIOR_DIM = "mu_prior_dim"
READOUT_NLL_FULL = "nll_full"
READOUT_NLL_BASE = "nll_base"
READOUT_PRED_GAP = "pred_gap"
READOUT_LAG_BAND = "lag_band"
#: The full-branch block score at ONE horizon step -- the near and the far end of the forecast
#: block are different questions of the same inputs -- and the forecast FIDELITY readouts: the
#: masked squared error of the mean-decoded full forecast, which the learned variance cannot
#: trade against, and its base-minus-full gap. The block score is a log-density and a well
#: calibrated but wide forecast scores it well; the squared error is what a reader means by
#: "how close was the forecast". The block scores are taken under the model's own density -- its
#: scored-cell mask and its AR(1) coefficient -- and the two fidelity readouts under the cell mask
#: alone, since an error of the mean forecast has no innovation.
READOUT_NLL_HORIZON = "nll_horizon"
READOUT_MSE_FULL = "mse_full"
READOUT_MSE_GAP = "mse_gap"
READOUTS: Tuple[str, ...] = (
    READOUT_KLD, READOUT_KLD_DIM, READOUT_MU_POST_DIM, READOUT_MU_PRIOR_DIM,
    READOUT_NLL_FULL, READOUT_NLL_BASE, READOUT_PRED_GAP, READOUT_LAG_BAND,
    READOUT_NLL_HORIZON, READOUT_MSE_FULL, READOUT_MSE_GAP,
)
#: The block-score readouts, which decode a latent mean and score the forecast block.
BLOCK_SCORE_READOUTS: Tuple[str, ...] = (
    READOUT_NLL_FULL, READOUT_NLL_BASE, READOUT_PRED_GAP, READOUT_NLL_HORIZON,
    READOUT_MSE_FULL, READOUT_MSE_GAP,
)
#: The readouts whose value depends on the target streams alone, so their source attribution is
#: zero by construction and is asserted rather than assumed.
TARGET_ONLY_READOUTS: Tuple[str, ...] = (READOUT_MU_PRIOR_DIM, READOUT_NLL_BASE)
#: What a production pass attributes under both baselines: the latent change, the forecast gain,
#: the full-branch score itself and the full-branch fidelity, so which inputs move the score and
#: which move the error can be read against each other on every population figure.
MAIN_READOUTS: Tuple[str, ...] = (READOUT_KLD, READOUT_PRED_GAP, READOUT_NLL_FULL, READOUT_MSE_FULL)
#: The horizon steps the per-horizon block score is attributed at, as names of a position in the
#: block: the first step and the last, resolved against the model's horizon by
#: :func:`horizon_steps`. The ``band`` column of a row carries the step as ``h<step>``.
HORIZON_READOUT_STEPS: Tuple[str, ...] = ("first", "last")

#: The four input blocks every attribution map can be summed over: the two target blocks and
#: the two source blocks, in the order they are concatenated in.
BLOCK_TARGET_SCATTERING = "target_scattering"
BLOCK_TARGET_PHASE = "target_phase"
BLOCK_SOURCE_SCATTERING = "source_scattering"
BLOCK_SOURCE_PHASE = "source_phase"
BLOCKS: Tuple[str, ...] = (
    BLOCK_TARGET_SCATTERING, BLOCK_TARGET_PHASE, BLOCK_SOURCE_SCATTERING, BLOCK_SOURCE_PHASE,
)

#: The two input streams the attributions are reduced on. ``target`` is the declared
#: concatenation of the two target blocks, ``source`` the declared source stream.
STREAM_TARGET = "target"
STREAM_SOURCE = "source"
STREAMS: Tuple[str, ...] = (STREAM_TARGET, STREAM_SOURCE)

#: The sentence every artifact of this analysis carries.
ATTRIBUTION_CAVEAT = (
    "an attribution is a sensitivity of a fitted computation, not a causal claim about the "
    "physiology: it says which stored coefficients the model's readout responded to along one "
    "interpolation path from one baseline, under one trained parameterisation. The source "
    "attribution under the source-null baseline is to source CONTENT relative to the "
    "availability clock, which is a constant of the step index and cannot be attributed to any "
    "input. Every lag axis is stored-coefficient time. Every summary is over recordings"
)

#: The one line an attribution **figure** carries in place of :data:`ATTRIBUTION_CAVEAT`: the two
#: claims a reader must not make from a map or a lag profile. The argument for each stays in
#: :data:`ATTRIBUTION_CAVEAT` (``summary.json``, ``ATTRIBUTION.md``) and in
#: :data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT`.
ATTRIBUTION_NOTE = "Model sensitivity, not a causal effect or a physiological latency."

#: What was tried on the tiny fixtures and what came of it, recorded in every block so a reader
#: of a summary knows which methods were rejected and why rather than only which were kept.
METHOD_RECORD: Dict[str, Dict[str, str]] = {
    "IntegratedGradients": {
        "status": "shipped",
        "note": "the primary method; the path enters the baseline at the entry fraction because it "
                "does not converge from the exact zero, and this run's completeness residual is "
                "measured per row (checks.completeness_rel_*)",
    },
    "LayerIntegratedGradients": {
        "status": "shipped",
        "note": "on the per-head FUSION outputs of the head-structured posterior in the "
                "lag-attentive cells (the only route from the source into the posterior: a "
                "complete per-head split along the source-null path) and on the proposal head's "
                "OUTPUT in the lag-residual cell (a complete per-lag split through the limiter)",
    },
    "FeatureAblation": {
        "status": "shipped",
        "note": "model-agnostic; grouped by lag band of the source relative to the anchor, so it is "
                "the occlusion analysis's intervention read on this analysis's readouts and anchors",
    },
    "GradCAM": {
        "status": "shipped (cohort pass, lag-attentive cells)",
        "note": "computed in gradcam.py rather than through captum's LayerGradCam, which pools "
                "gradients over every axis after the second and so assumes channel-first maps, "
                "while these layers are time-major (B, T, D); three views: the input of the target "
                "encoder's last attention block, the source K/V stream, and the lag-attention "
                "weights scored as gradient-weighted attention; one forward and one backward per "
                "readout, no baseline and no completeness",
    },
    "InputXGradient": {
        "status": "evaluated, not shipped",
        "note": "runs and passes every structural check; a single-point gradient, no completeness, "
                "and it adds nothing the integrated form does not",
    },
    "Saliency": {
        "status": "evaluated, not shipped",
        "note": "the absolute gradient; unsigned, no completeness",
    },
    "GradientShap": {
        "status": "evaluated, not shipped",
        "note": "integrated gradients averaged over a random baseline distribution; its per-sample "
                "completeness residual is of the order of the readout by construction, and the "
                "randomness of the baseline would make two runs disagree",
    },
    "Occlusion": {
        "status": "evaluated, not shipped",
        "note": "a sliding window straddles the anchor and assigns a window's effect to steps after "
                "it, so its per-step output fails the causality check by construction; the grouped "
                "FeatureAblation above is the same intervention on the right groups",
    },
    "DeepLift": {
        "status": "evaluated, not usable",
        "note": "runs, but its rescale rule reaches only the nonlinearities it hooks (ReLU, Tanh, "
                "Sigmoid modules); the GELU and SiLU functionals, the smooth bounds and entmax15 "
                "pass through as plain gradients, so its completeness residual is of the order of "
                "the readout itself",
    },
    "LayerConductance": {
        "status": "evaluated, not shipped",
        "note": "sums differ from the readout difference on the layers tested, where the layer "
                "integrated gradient is exact; nothing it adds survives that",
    },
    "NeuronConductance": {
        "status": "evaluated, not shipped",
        "note": "the conductance of one latent coordinate is the input attribution of that "
                "coordinate's readout, which the readout registry already provides; and the heads "
                "return tuples the selector cannot index",
    },
}


# =============================================================================
# Which cell
# =============================================================================
@dataclass(frozen=True)
class CellBinding:
    """What differs between the two cells, as far as this module needs to know.

    Attributes:
        name: ``'attention'`` or ``'slot'``.
        dense_latent: Whether the latent tensors are dense over $T$ (gathered at the anchor) or
            already on the anchor axis (selected by column).
        lag_readout: What the cell's own per-lag magnitude is, for the agreement statistic and the
            figure legends.
        lag_qualification: The sentence that qualifies that readout on every artifact.
        layer_label: What the layer attribution is taken on.
        layer_axis: The name of the axis the layer attribution is resolved on.
        lag_band_unit: The unit of the lag readout attributed on a band.
    """

    name: str
    dense_latent: bool
    lag_readout: str
    lag_qualification: str
    layer_label: str
    layer_axis: str
    lag_band_unit: str


ATTENTION_CELL = CellBinding(
    name="attention",
    dense_latent=True,
    lag_readout="KL attribution over lags (K_t times the head-structured attention)",
    lag_qualification=(
        "the model's own lag readout here is the KL attribution K_t * alpha, summed over heads, "
        "which inherits the prior-variance inflation the attention is immune to"
    ),
    layer_label="the per-head fused features of the head-structured posterior",
    layer_axis="head",
    lag_band_unit="attention mass",
)

SLOT_CELL = CellBinding(
    name="slot",
    dense_latent=False,
    lag_readout="proposal norm ||r_{t,l}||_2 over lags",
    lag_qualification=(
        "the model's own lag readout here is the proposal NORM at every lag: an update magnitude "
        "before the sum and the limiter, not a distribution over lags and not an allocation of "
        "the divergence"
    ),
    layer_label="the per-lag proposals the fusion sums",
    layer_axis="lag",
    lag_band_unit="proposal norm",
)

#: The unit of every readout, which is also the unit of its attributions (an integrated-gradient
#: map sums to a difference of the readout). Every readout is per anchor; the block scores sum
#: over the scored cells (the per-step score over one horizon step's); the squared errors are in
#: the loader's standardised coefficient units, squared.
READOUT_UNITS: Mapping[str, str] = {
    READOUT_KLD: "nats",
    READOUT_KLD_DIM: "nats",
    READOUT_MU_POST_DIM: "latent units",
    READOUT_MU_PRIOR_DIM: "latent units",
    READOUT_NLL_FULL: "nats",
    READOUT_NLL_BASE: "nats",
    READOUT_PRED_GAP: "nats",
    READOUT_NLL_HORIZON: "nats",
    READOUT_MSE_FULL: "squared z",
    READOUT_MSE_GAP: "squared z",
}


def readout_unit(readout: str, cell: CellBinding) -> str:
    """The unit of a readout on a cell: :data:`READOUT_UNITS`, or the cell's own for the lag readout."""
    return cell.lag_band_unit if readout == READOUT_LAG_BAND else READOUT_UNITS.get(readout, "readout units")


# =============================================================================
# The wrapper
# =============================================================================
#: The model's reparameterisation seam, by cell: the lag-attentive cells' private one and the
#: lag-residual cell's public one. Both map $(\mu^p, \ell^p, \mu^q, \ell^q)$ to $(z^p, z^q)$.
REPARAMETERISATION_SEAMS: Tuple[str, ...] = ("_reparameterize_shared", "reparameterize_shared")


@contextmanager
def attributed_forward(model: Any, anchors: torch.Tensor) -> Iterator[None]:
    r"""The model's own forward, decoding each row's anchor alone and at its two latent means.

    Two of the model's seams are shadowed on the instance for the duration and restored after,
    and nothing else changes. The **anchor builder** returns each row's own anchor as a one-anchor
    set: the decoder is most of a dense forward's cost and every readout here reads one anchor,
    so decoding the $A$ anchors of the dense axis would spend the forward on forecasts nothing
    reads. The **reparameterisation** returns $(\mu^p, \mu^q)$, so the forward's own two
    forecasts are the mean-decoded ones the collection pass scores as ``mean_*`` -- through the
    same decoder call, with the same persistence input -- and no $\epsilon$ is drawn: an attributed
    readout is a deterministic function of its inputs and consumes no random number. Everything
    read at an anchor is a function of that anchor alone -- the latents are dense over $T$ or
    computed per anchor, and the decoder decodes each anchor independently -- so each readout
    equals the dense forward's, which the tests assert.

    Args:
        model: The rebuilt net.
        anchors: Each row's anchor as a stored step, $(B,)$ ``long``, of the batch the forward
            will be called with.

    Yields:
        Nothing; the model is restored on exit, whatever happens.
    """
    def one_anchor(batch: int, device: Any, anchor_phase: Any = None, anchor_stride: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        index = anchors.to(device=device, dtype=torch.long).reshape(int(batch), 1)
        return index, torch.ones_like(index, dtype=torch.bool)

    def at_means(mu_prior: torch.Tensor, logvar_prior: torch.Tensor, mu_post: torch.Tensor, logvar_post: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return mu_prior, mu_post

    shadows = {"_build_anchor_index": one_anchor}
    shadows.update({name: at_means for name in REPARAMETERISATION_SEAMS if hasattr(model, name)})
    for name, value in shadows.items():
        object.__setattr__(model, name, value)
    try:
        yield
    finally:
        for name in shadows:
            object.__delattr__(model, name)


class AnchorReadout(nn.Module):
    r"""The model's own forward, reduced to one scalar per sample at one anchor per sample.

    Captum attributes a function of its ``inputs`` tuple; this is that function. Each row names
    its anchor by its column on the dense evaluation axis (``columns`` is per row, so a batch of
    rows can attribute several anchors of one segment in one call); the real model is called
    under :func:`attributed_forward`, which decodes that one anchor at the two latent means, and
    the chosen readout is returned there. The block-score readouts score the forward's own two
    mean-decoded forecasts, rebuilding the forecast target and the forecast mask from the two
    extra tensors on every call, because Captum expands the batch along its interpolation axis
    and a target built once outside would no longer match.

    **Every input is tied into the graph with a zero coefficient**, so a readout that does not
    depend on the source -- the prior mean, the base block score -- still yields a gradient for
    it: exactly zero, which is the structural fact the tests assert, rather than an autograd
    error about an unused tensor.

    Attributes:
        model: The rebuilt net.
        cell: Which cell's forward and tensor layout this is.
        readout: One of :data:`READOUTS`.
        likelihood: The objective's likelihood, for the block-score readouts.
        lag_band: The inclusive lag band the lag readout sums over.
    """

    def __init__(
        self,
        model: nn.Module,
        cell: CellBinding,
        *,
        readout: str,
        likelihood: str = "gaussian_nll",
        lag_band: Tuple[int, int] = (0, 0),
        horizon: int = 0,
    ) -> None:
        """Bind the model and the readout.

        Args:
            model: The rebuilt net, in evaluation mode.
            cell: The cell binding.
            readout: One of :data:`READOUTS`.
            likelihood: ``'mse'`` or ``'gaussian_nll'``, for the block-score readouts; the
                fidelity readouts score under ``'mse'`` whatever this says.
            lag_band: The inclusive ``(lo, hi)`` lag pair the lag readout sums over.
            horizon: The horizon step the per-horizon readout scores, ``0 <= horizon < H``.

        Raises:
            ValueError: If ``readout`` is not one of :data:`READOUTS`.
        """
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.model = model
        self.cell = cell
        self.readout = str(readout)
        self.likelihood = str(likelihood)
        self.lag_band = (int(lag_band[0]), int(lag_band[1]))
        self.horizon = int(horizon)

    def anchor_steps(self, columns: torch.Tensor) -> torch.Tensor:
        """Each row's anchor as a stored step: its column on the dense evaluation axis, read through
        the model's own anchor builder at that geometry, with no forward.

        Args:
            columns: Each row's position on the dense anchor axis, $(B,)$ ``long``.

        Returns:
            The stored steps, $(B,)$ ``long``.
        """
        index, _valid = self.model._build_anchor_index(
            batch=int(columns.shape[0]), device=columns.device,
            anchor_phase=DENSE_ANCHOR_GEOMETRY[0], anchor_stride=DENSE_ANCHOR_GEOMETRY[1],
        )
        return index[torch.arange(columns.shape[0], device=columns.device), columns]

    def _at_anchor(self, dense: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Read a per-row tensor at each row's own anchor: a gather on a dense axis, the one decoded
        anchor on the anchor axis."""
        if self.cell.dense_latent:
            index = anchors.view(-1, 1, *([1] * (dense.dim() - 2))).expand(-1, 1, *dense.shape[2:])
            return dense.gather(1, index).squeeze(1)
        return dense[:, 0]

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        target_features: torch.Tensor,
        weight: torch.Tensor,
        columns: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        r"""Run the model and return the readout at each row's anchor.

        Args:
            y_st: Target scattering block $(B, T, \cdot)$, declared width.
            y_ph: Target phase-harmonic block $(B, T, \cdot)$, declared width.
            u_stream: Source stream $(B, T, c_u)$, declared width.
            target_features: The declared-width target stream the block scores are taken
                against, $(B, T, c_y)$.
            weight: The decimated validity signal $(B, T)$.
            columns: Each row's anchor as a position on the dense anchor axis, $(B,)$ ``long``.
            coordinates: Each row's latent coordinate for the per-coordinate readouts, $(B,)$
                ``long``; ignored by the others.

        Returns:
            The readout, $(B,)$.
        """
        model = self.model
        anchors = self.anchor_steps(columns)                            # (B,)
        with attributed_forward(model, anchors):
            outputs = model(y_st, y_ph, u_stream, **({} if self.cell.dense_latent else {"return_proposals": True}))
        rows = torch.arange(columns.shape[0], device=columns.device)
        anchor_valid = outputs["anchor_valid"][:, 0]                    # (B,)
        name = self.readout
        # The zero tie: see the class docstring. The two posterior parameters are tied in as
        # well, so a layer attribution on a module whose output reaches only the parameter a
        # readout does not read -- the scale proposals under the mean-decoded gap -- still finds
        # that output on the graph, with a gradient of exactly zero.
        tie = 0.0 * (
            y_st.sum() + y_ph.sum() + u_stream.sum()
            + outputs["mu_post"].sum() + outputs["logvar_post"].sum()
        )

        if name == READOUT_KLD:
            dense = outputs["kld_per_t"] if self.cell.dense_latent else outputs["kld_per_anchor"]
            return self._at_anchor(dense, anchors) + tie
        if name in (READOUT_KLD_DIM, READOUT_MU_POST_DIM, READOUT_MU_PRIOR_DIM):
            if name == READOUT_KLD_DIM:
                dense = (
                    model.kld_tensor(
                        mu_prior=outputs["mu_prior"], logvar_prior=outputs["logvar_prior"],
                        mu_post=outputs["mu_post"], logvar_post=outputs["logvar_post"],
                    )
                    if self.cell.dense_latent else outputs["kld_per_anchor_dim"]
                )
            else:
                dense = outputs["mu_post" if name == READOUT_MU_POST_DIM else "mu_prior"]
            vector = self._at_anchor(dense, anchors)                   # (B, d_z)
            return vector[rows, coordinates] + tie
        if name in BLOCK_SCORE_READOUTS:
            anchor_column = anchors[:, None]
            target = model._build_forecast_target(target_features, anchor_column)
            mask, _coverage = forecast_mask(
                model.scored_weight(weight), model.geometry,
                coverage_floor=model.coverage_floor, anchors=anchor_column,
                anchor_valid=anchor_valid[:, None],
            )
            fidelity = name in (READOUT_MSE_FULL, READOUT_MSE_GAP)
            likelihood = "mse" if fidelity else self.likelihood
            # The density the collection pass scores under, so a block-score readout is the
            # ``mean_*`` column's own number; the fidelity readouts count the scored cells only.
            density = forecast_likelihood_terms(model)
            if fidelity:
                density["ar_coef"] = None
            gap = name in (READOUT_PRED_GAP, READOUT_MSE_GAP)
            scores: Dict[str, torch.Tensor] = {}
            for branch in ("full", "base"):
                if not gap and not name.endswith(branch) and name != READOUT_NLL_HORIZON:
                    continue
                if name == READOUT_NLL_HORIZON and branch != "full":
                    continue
                # The forward's own forecast of this branch at the row's anchor, decoded at the
                # latent mean under ``attributed_forward``: (B, 1, H, C_keep).
                forecast_mu, forecast_logvar = outputs[f"mu_{branch}"], outputs[f"logvar_{branch}"]
                if name == READOUT_NLL_HORIZON:
                    # The block score resolved by horizon step: summed over the channels of one
                    # step rather than over the whole block. Summed over steps it is ``nll_full``.
                    score = raw_sample_score(
                        forecast_mu, target, likelihood=likelihood, logvar=forecast_logvar,
                        step_mask=mask, **density,
                    )
                    scores[branch] = (score * mask[..., None]).sum(dim=3)[:, 0, self.horizon]
                    continue
                block, _ = masked_raw_block_per_anchor(
                    forecast_mu, target, mask, likelihood=likelihood, logvar=forecast_logvar,
                    **density,
                )
                scores[branch] = block[:, 0]
            if gap:
                return scores["base"] - scores["full"] + tie
            return scores["base" if name == READOUT_NLL_BASE else "full"] + tie
        # The lag readout: the attention mass on the band in the attentive cells, the proposal
        # norm on the band in the residual cell.
        low, high = self.lag_band
        if self.cell.dense_latent:
            alpha = self._at_anchor(outputs["attn_weights"], anchors)          # (B, M, L)
            return alpha.mean(dim=1)[:, low:high + 1].sum(dim=-1) + tie
        proposals = outputs["mean_proposals"][:, 0]                            # (B, L, d_z)
        return proposals[:, low:high + 1].norm(dim=-1).sum(dim=-1) + tie


# =============================================================================
# Inputs, baselines, anchors
# =============================================================================
def baselines_for(name: str, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Build one named baseline tuple for an input tuple.

    Args:
        name: One of :data:`BASELINES`.
        inputs: ``(y_st, y_ph, u_stream)``.

    Returns:
        The baseline tuple, same shapes.

    Raises:
        ValueError: If the name is unknown.
    """
    y_st, y_ph, u_stream = inputs
    if name == BASELINE_SOURCE_NULL:
        return (y_st.detach().clone(), y_ph.detach().clone(), torch.zeros_like(u_stream))
    if name == BASELINE_ALL_ZERO:
        return tuple(torch.zeros_like(x) for x in inputs)
    raise ValueError(f"baseline must be one of {BASELINES}, got {name!r}")


def entry_point(
    inputs: Sequence[torch.Tensor],
    baselines: Sequence[torch.Tensor],
    fraction: float = BASELINE_ENTRY_FRACTION,
) -> Tuple[torch.Tensor, ...]:
    r"""The point the integration starts from: $x_0 = b + \alpha_0 (x - b)$.

    Args:
        inputs: The input tuple.
        baselines: The exact baseline tuple.
        fraction: $\alpha_0$.

    Returns:
        The entry tuple.
    """
    return tuple(b + float(fraction) * (x - b) for x, b in zip(inputs, baselines))


@torch.no_grad()
def contributing_columns(model: Any, weight: torch.Tensor, outputs: Mapping[str, torch.Tensor]) -> np.ndarray:
    """Which positions of the anchor axis are scored, per sample, as a boolean $(B, A)$ array.

    Built from the same masks the collection pass scores under, so an attributed anchor is one
    the tables carry a score for.

    Args:
        model: The rebuilt net.
        weight: The decimated validity signal $(B, T)$.
        outputs: The dense forward's dict.

    Returns:
        The indicator.
    """
    from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors

    mask, _coverage = forecast_mask(
        model.scored_weight(weight), model.geometry, coverage_floor=model.coverage_floor,
        anchors=outputs["anchor_index"], anchor_valid=outputs["anchor_valid"],
    )
    return (contributing_anchors(mask) > 0.0).detach().cpu().numpy()


def spread_columns(contributing: np.ndarray, per_segment: int = ANCHORS_PER_SEGMENT) -> List[np.ndarray]:
    """Choose evenly spaced scored anchors per sample.

    Evenly spaced over the scored set rather than drawn, so the attributed anchors of every
    segment span its early, middle and late phases, and two runs attribute the same ones.

    Args:
        contributing: The $(B, A)$ scored indicator.
        per_segment: How many anchors to choose per sample, as an upper bound.

    Returns:
        One ascending array of anchor-axis positions per sample; empty where none is scored.
    """
    chosen: List[np.ndarray] = []
    for row in np.asarray(contributing, dtype=bool):
        positions = np.flatnonzero(row)
        if positions.size == 0:
            chosen.append(np.zeros(0, dtype=np.int64))
            continue
        take = min(int(per_segment), int(positions.size))
        picks = np.unique(np.round(np.linspace(0, positions.size - 1, take)).astype(np.int64))
        chosen.append(positions[picks].astype(np.int64))
    return chosen


def informative_columns(
    anchor_index: np.ndarray,
    contributing: np.ndarray,
    candidates: Sequence[int],
    weight: np.ndarray,
    *,
    n_lags: int,
    per_segment: int = ANCHORS_PER_SEGMENT,
    spacing: int = 1,
) -> Tuple[np.ndarray, int]:
    r"""Choose one sample's highest-KL anchors whose source history is clean.

    The candidates come from the collection pass, already high-KL and with a clean forecast
    window, in descending $K_t$ (:func:`~.attribution_pass.informative_anchors`). This function
    adds the history check the per-anchor table cannot make: the mean validity
    ``weight`` over the searched lag window $[t_a - L + 1, t_a]$ must reach
    :data:`CLEAN_COVERAGE`. Anchors that pass come first, in $K_t$ order; anchors that fail follow,
    so a segment is never left empty. Chosen anchors are at least ``spacing`` steps apart, so two
    of them never share most of their forecast block.

    Args:
        anchor_index: The sample's anchor axis as stored steps, $(A,)$.
        contributing: The sample's scored indicator, $(A,)$.
        candidates: Candidate anchor steps, highest $K_t$ first.
        weight: The sample's decimated validity signal, $(T,)$.
        n_lags: $L$, the searched lag window.
        per_segment: How many anchors to choose, as an upper bound.
        spacing: The smallest distance between two chosen anchors, in stored steps.

    Returns:
        ``(columns, n_unclean)``: the chosen anchor-axis positions in $K_t$ order, and how many of
        them failed the history check.
    """
    position = {int(step): column for column, step in enumerate(anchor_index) if contributing[column]}
    ranked = []
    for order, step in enumerate(int(s) for s in candidates):
        column = position.get(step)
        if column is None:
            continue
        window = np.asarray(weight[max(0, step - int(n_lags) + 1):step + 1], dtype=np.float64)
        clean = bool(window.size) and float(window.mean()) >= CLEAN_COVERAGE - 1e-6
        ranked.append((not clean, order, column, step))
    ranked.sort()
    chosen: List[int] = []
    steps: List[int] = []
    unclean = 0
    for failed, _order, column, step in ranked:
        if len(chosen) == int(per_segment):
            break
        if all(abs(step - other) >= int(spacing) for other in steps):
            chosen.append(column)
            steps.append(step)
            unclean += int(failed)
    return np.asarray(chosen, dtype=np.int64), unclean


def _host(tensor: torch.Tensor) -> np.ndarray:
    """One tensor to a float64 array on the host."""
    return tensor.detach().cpu().to(torch.float64).numpy()


@torch.no_grad()
def model_lag_readout(
    model: Any,
    cell: CellBinding,
    outputs: Mapping[str, torch.Tensor],
    sample: torch.Tensor,
    columns: torch.Tensor,
) -> np.ndarray:
    r"""The model's own per-lag magnitude at each row's anchor, $(N, L)$ as float64.

    The KL attribution $\sum_m K^{(m)}_t \alpha^{(m)}_{t,\ell}$ in the attentive cells; the
    proposal norm $\lVert r^\mu_{t,\ell} \rVert_2$, ``NaN`` where the lag carried no available
    channel, in the residual cell -- the same two profiles the traces carry. Read off the
    **unexpanded** forward: a row names the batch element it came from and its anchor column.

    Args:
        model: The rebuilt net.
        cell: The cell binding.
        outputs: The dense forward's dict, taken with the proposals retained in the residual cell.
        sample: Each row's batch element, $(N,)$ ``long``.
        columns: Each row's anchor-axis position, $(N,)$ ``long``.

    Returns:
        The profile per row, or an all-``NaN`` $(N, L)$ array on an arm that produces none.
    """
    if cell.dense_latent:
        anchors = outputs["anchor_index"][sample, columns]
        dense = outputs["source_kl_lag_map"][sample]                      # (N, T, L)
        index = anchors[:, None, None].expand(-1, 1, dense.shape[-1])
        return _host(dense.gather(1, index).squeeze(1))
    n_lags = int(model.n_lags)
    proposals = outputs.get("mean_proposals")
    if proposals is None:
        return np.full((int(columns.shape[0]), n_lags), np.nan)
    norm = proposals[sample, columns].norm(dim=-1)
    valid = outputs["lag_valid"][sample, columns].to(torch.bool)
    return _host(torch.where(valid, norm, torch.full_like(norm, float("nan"))))


def warm_from_step(model: Any, stream: str) -> Optional[np.ndarray]:
    r"""The stored step each **declared** channel of a stream becomes live at, or ``None``.

    A gathered-and-shifted channel at encoder step $t$ reads stored step $t - d_c$ and is masked
    while $t < W'_c + d_c$, so in stored coordinates the channel is live from $W'_c$. A channel the
    gate dropped is live from nowhere and is marked with the sequence length.

    Args:
        model: The rebuilt net.
        stream: ``'target'`` or ``'source'``.

    Returns:
        One step per declared channel, or ``None`` for a stream with no warm-up at all.
    """
    waits = model.target_warmup_steps if stream == STREAM_TARGET else model.source_warmup_steps
    gate = model.target_gate if stream == STREAM_TARGET else model.source_gate
    width = int(model.c_y if stream == STREAM_TARGET else model.c_u)
    if waits is None:
        return None
    live = np.full(width, int(model.sequence_length), dtype=np.int64)
    keep = list(range(width)) if gate is None else [int(v) for v in gate.keep_index.tolist()]
    for channel, wait in zip(keep, waits):
        live[int(channel)] = int(wait)
    return live


def horizon_steps(model: Any, names: Sequence[str] = HORIZON_READOUT_STEPS) -> Dict[str, int]:
    """Resolve the named horizon positions against the model's own block length.

    Args:
        model: The rebuilt net, for ``horizon``.
        names: ``'first'``, ``'last'`` or a decimal step.

    Returns:
        ``{name: step}`` in the order given, each ``0 <= step < H``; a name resolving past the
        block is dropped rather than clipped, so a one-step block attributes one step once.
    """
    length = int(getattr(model, "horizon", 1))
    out: Dict[str, int] = {}
    for name in names:
        step = {"first": 0, "last": length - 1}.get(str(name))
        step = int(name) if step is None else step
        if 0 <= step < length and step not in out.values():
            out[str(name)] = step
    return out


def source_block_split(model: Any) -> int:
    """Where the declared source stream's scattering block ends and its phase block begins."""
    split = getattr(model, "SOURCE_BLOCK_SPLIT", None)
    if split is None or not bool(getattr(model, "use_up_st", True)):
        return 0
    return int(min(int(split), int(model.c_u)))


def block_sums(
    target: np.ndarray, source: np.ndarray, *, n_scattering: int, source_split: int
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    r"""Sum $(N, T, C)$ maps over each of the four input blocks: signed and unsigned.

    Args:
        target: The declared target maps, scattering block first.
        source: The declared source maps, scattering block first where the source carries one.
        n_scattering: Width of the target scattering block.
        source_split: Width of the source scattering block; $0$ on a phase-only source.

    Returns:
        ``{block: (signed (N,), unsigned (N,))}`` in :data:`BLOCKS` order.
    """
    target = np.asarray(target, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    spans = {
        BLOCK_TARGET_SCATTERING: target[:, :, :int(n_scattering)],
        BLOCK_TARGET_PHASE: target[:, :, int(n_scattering):],
        BLOCK_SOURCE_SCATTERING: source[:, :, :int(source_split)],
        BLOCK_SOURCE_PHASE: source[:, :, int(source_split):],
    }
    return {name: (field.sum(axis=(1, 2)), np.abs(field).sum(axis=(1, 2))) for name, field in spans.items()}


def masked_field(
    field: np.ndarray, live: Optional[np.ndarray], *, anchor: Optional[int] = None
) -> np.ndarray:
    r"""Blank the cells the model never read: cold steps per channel and, if asked, after the anchor.

    A $(T, C)$ map with ``NaN`` where step $t < $ ``live[c]`` -- the input gate multiplies those
    cells by zero, so what is stored there is not what the encoder saw and must not set a colour
    scale -- and, when ``anchor`` is given, at every step after it, where a causal attribution is
    exactly zero. Drawn as the bad colour, so a blank cell reads as "not read" rather than as a
    zero the model found.

    Args:
        field: $(T, C)$.
        live: Each channel's first live step, or ``None`` for no gate.
        anchor: The anchor's stored step, or ``None`` to keep the steps after it.

    Returns:
        A float64 copy with the blanked cells ``NaN``.
    """
    out = np.array(field, dtype=np.float64, copy=True)
    steps = np.arange(out.shape[0])
    if live is not None:
        out[steps[:, None] < np.asarray(live, dtype=np.int64)[None, :out.shape[1]]] = np.nan
    if anchor is not None:
        out[steps > int(anchor)] = np.nan
    return out


#: Decades of dynamic range every logarithmic colour scale and axis here keeps below its own
#: maximum -- the traces' constant, so a map here and a trace there stretch the same way.
LOG_DECADES = traces.LOG_PANEL_DECADES


def signed_log_norm(field: np.ndarray) -> Optional[Any]:
    r"""A symmetric-log colour scale about zero over a signed field, or ``None`` on an empty one.

    Linear inside $\max|a| \cdot 10^{-\mathrm{LOG\_DECADES}}$ and logarithmic beyond it, so a
    handful of dominant cells no longer flatten every other cell into white.
    """
    finite = np.asarray(field, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size or not np.abs(finite).max() > 0.0:
        return None
    limit = float(np.abs(finite).max())
    return mcolors.SymLogNorm(linthresh=limit * 10.0 ** (-LOG_DECADES), vmin=-limit, vmax=limit, base=10)


def unsigned_log_norm(field: np.ndarray) -> Optional[Any]:
    """A log colour scale over a non-negative field's positive mass, or ``None`` when it has none."""
    finite = np.asarray(field, dtype=np.float64)
    positive = finite[np.isfinite(finite) & (finite > 0.0)]
    if not positive.size:
        return None
    top = float(positive.max())
    return mcolors.LogNorm(vmin=top * 10.0 ** (-LOG_DECADES), vmax=top)


def symlog_axis(ax: Any, *values: Any, axis: str = "y", headroom: float = 0.0) -> None:
    """Put an axis on a symmetric-log scale floored :data:`LOG_DECADES` below the data's largest magnitude.

    The linear floor is derived from the data drawn rather than fixed, because a share per lag and
    a block score in nats are orders of magnitude apart and one threshold cannot serve both. A
    panel whose data has no magnitude stays linear. ``headroom`` is room for a legend, in
    **decades** above the data's top -- the linear headroom the shared legend helper makes is a
    sliver on a log axis.
    """
    stacked = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in values]) if values else np.zeros(0)
    finite = stacked[np.isfinite(stacked)]
    if not finite.size or not np.abs(finite).max() > 0.0:
        return
    threshold = float(np.abs(finite).max()) * 10.0 ** (-LOG_DECADES)
    (ax.set_yscale if axis == "y" else ax.set_xscale)("symlog", linthresh=threshold, base=10)
    if headroom > 0.0:
        low, high = ax.get_ylim() if axis == "y" else ax.get_xlim()
        high = max(float(high), threshold) * 10.0 ** float(headroom)
        (ax.set_ylim if axis == "y" else ax.set_xlim)(low, high)


def symlog_legend(
    ax: Any, *values: Any, ncol: int = 2, axis: str = "y", legend: bool = True, **kwargs: Any
) -> None:
    """A symmetric-log axis with a decade of headroom and the legend placed in it.

    ``legend=False`` keeps the axis and its headroom and draws no legend, for a panel whose keys
    a neighbouring panel of the same figure already carries.
    """
    symlog_axis(ax, *values, axis=axis, headroom=1.0)
    if legend:
        ax.legend(loc="upper left", ncol=int(ncol), fontsize=figures.FONT_TINY, **kwargs)


def _attach_colorbar(figure: Any, image: Any, *, ax: Any, cax: Any, label: str, norm: Any) -> Any:
    """One colourbar convention for every map here; a symmetric-log scale gets five readable ticks."""
    colorbar = figure.colorbar(image, cax=cax) if cax is not None else figure.colorbar(image, ax=ax, fraction=0.03, pad=0.015, aspect=30)
    if isinstance(norm, mcolors.SymLogNorm):
        top = float(norm.vmax)
        mid = top * 10.0 ** (-LOG_DECADES / 2.0)
        colorbar.set_ticks([-top, -mid, 0.0, mid, top])
        colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: "0" if v == 0.0 else f"{v:.0e}"))
        colorbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
    if label:
        colorbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"])
    colorbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], width=plt.rcParams["ytick.major.width"])
    colorbar.outline.set_linewidth(plt.rcParams["axes.linewidth"])
    return colorbar


# =============================================================================
# The attributions
# =============================================================================
@dataclass
class AttributionBatch:
    """One Captum call's worth of attributions, as arrays on the host.

    Attributes:
        readout: The readout attributed.
        baseline: The baseline name.
        sample: Which sample of the batch each row came from, $(N,)$.
        column: Each row's anchor-axis position, $(N,)$.
        anchor: Each row's anchor as a stored step, $(N,)$.
        coordinate: Each row's latent coordinate, $(N,)$; $-1$ where the readout has none.
        value_input: The readout at the input, $(N,)$.
        value_baseline: The readout at the exact baseline, $(N,)$.
        value_entry: The readout at the entry point, $(N,)$.
        delta: Captum's convergence delta, attribution sum minus the input-to-entry difference.
        target: Attribution over the declared target axis, $(N, T, c_y)$ float32.
        source: Attribution over the declared source axis, $(N, T, c_u)$ float32.
    """

    readout: str
    baseline: str
    sample: np.ndarray
    column: np.ndarray
    anchor: np.ndarray
    coordinate: np.ndarray
    value_input: np.ndarray
    value_baseline: np.ndarray
    value_entry: np.ndarray
    delta: np.ndarray
    target: np.ndarray
    source: np.ndarray


def expand_rows(
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    columns_per_sample: Sequence[np.ndarray],
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], torch.Tensor, np.ndarray]:
    """Repeat every sample once per chosen anchor, so one call attributes several anchors.

    Args:
        inputs: ``(y_st, y_ph, u_stream)``, $(B, \\ldots)$.
        extra: ``(target_features, weight)``, $(B, \\ldots)$.
        columns_per_sample: One array of anchor-axis positions per sample.

    Returns:
        ``(inputs, extra, columns, sample)``: the row-expanded tensors, the per-row column tensor
        and the per-row sample index.
    """
    counts = [int(len(c)) for c in columns_per_sample]
    repeats = torch.tensor(counts, dtype=torch.long, device=inputs[0].device)
    rows_inputs = tuple(x.repeat_interleave(repeats, dim=0) for x in inputs)
    rows_extra = tuple(x.repeat_interleave(repeats, dim=0) for x in extra)
    columns = torch.tensor(
        np.concatenate([np.asarray(c, dtype=np.int64) for c in columns_per_sample])
        if counts and sum(counts) else np.zeros(0, dtype=np.int64),
        dtype=torch.long, device=inputs[0].device,
    )
    sample = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(counts)]) if sum(counts) else np.zeros(0, dtype=np.int64)
    return rows_inputs, rows_extra, columns, sample


def integrated_gradients(
    wrapper: AnchorReadout,
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    columns: torch.Tensor,
    *,
    baseline: str,
    coordinates: Optional[torch.Tensor] = None,
    n_steps: int = IG_STEPS,
    internal_batch_size: int = IG_INTERNAL_BATCH_SIZE,
    entry_fraction: float = BASELINE_ENTRY_FRACTION,
) -> AttributionBatch:
    r"""Integrated gradients of one readout over the three streams, one row per anchor.

    Under :data:`BASELINE_SOURCE_NULL` the two target streams sit at their own values at both ends
    of the path, so $(x - x_0) = 0$ there and their attribution is exactly zero whatever the
    gradient is. They are then handed to the forward as fixed arguments rather than attributed,
    which returns the same zero maps without a backward pass through the target encoder at every
    one of the $n$ steps.

    Args:
        wrapper: The readout module.
        inputs: The row-expanded ``(y_st, y_ph, u_stream)``.
        extra: The row-expanded ``(target_features, weight)``.
        columns: Per-row anchor-axis positions, $(N,)$.
        baseline: One of :data:`BASELINES`.
        coordinates: Per-row latent coordinate, or ``None`` for readouts without one.
        n_steps: Integration steps.
        internal_batch_size: Captum's batch of interpolated inputs, in rows times steps; raised to
            the row count where it is smaller, which is what Captum does anyway.
        entry_fraction: Where the path enters the baseline.

    Returns:
        The batch of attributions, empty arrays when there is no row.
    """
    n_rows = int(columns.shape[0])
    if coordinates is None:
        coordinates = torch.zeros(n_rows, dtype=torch.long, device=columns.device)
    empty = np.zeros(0, dtype=np.float64)
    if n_rows == 0:
        return AttributionBatch(
            wrapper.readout, baseline, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), empty, empty, empty, empty,
            np.zeros((0, *inputs[0].shape[1:2], inputs[0].shape[2] + inputs[1].shape[2]), dtype=np.float32),
            np.zeros((0, *inputs[2].shape[1:]), dtype=np.float32),
        )
    exact = baselines_for(baseline, inputs)
    start = entry_point(inputs, exact, entry_fraction)
    forward_args = (extra[0], extra[1], columns, coordinates)
    with torch.no_grad():
        value_input = wrapper(*inputs, *forward_args)
        value_baseline = wrapper(*exact, *forward_args)
        value_entry = wrapper(*start, *forward_args)
    anchors = wrapper.anchor_steps(columns)
    settings = {
        "n_steps": int(n_steps), "internal_batch_size": max(int(internal_batch_size), n_rows),
        "return_convergence_delta": True,
    }
    if baseline == BASELINE_SOURCE_NULL:
        (source,), delta = IntegratedGradients(_source_only(wrapper)).attribute(
            (inputs[2].detach(),), baselines=(start[2].detach(),),
            additional_forward_args=(inputs[0].detach(), inputs[1].detach(), *forward_args), **settings,
        )
        target = torch.zeros(
            (*inputs[0].shape[:2], inputs[0].shape[2] + inputs[1].shape[2]), dtype=source.dtype, device=source.device
        )
    else:
        attributions, delta = IntegratedGradients(wrapper).attribute(
            tuple(x.detach() for x in inputs), baselines=tuple(x.detach() for x in start),
            additional_forward_args=forward_args, **settings,
        )
        target = torch.cat([attributions[0], attributions[1]], dim=-1)
        source = attributions[2]
    return AttributionBatch(
        readout=wrapper.readout,
        baseline=baseline,
        sample=np.zeros(n_rows, dtype=np.int64),
        column=columns.detach().cpu().numpy().astype(np.int64),
        anchor=anchors.detach().cpu().numpy().astype(np.int64),
        coordinate=(
            coordinates.detach().cpu().numpy().astype(np.int64)
            if wrapper.readout in (READOUT_KLD_DIM, READOUT_MU_POST_DIM, READOUT_MU_PRIOR_DIM)
            else np.full(n_rows, -1, dtype=np.int64)
        ),
        value_input=_host(value_input),
        value_baseline=_host(value_baseline),
        value_entry=_host(value_entry),
        delta=_host(delta),
        target=target.detach().cpu().to(torch.float32).numpy(),
        source=source.detach().cpu().to(torch.float32).numpy(),
    )


def _source_only(wrapper: AnchorReadout) -> Any:
    """The wrapper with the source stream first, so Captum attributes it alone.

    Captum attributes its ``inputs`` and passes ``additional_forward_args`` through unchanged
    (expanded along its interpolation axis), so the two target streams travel there when they are
    held fixed along the path.
    """
    def forward(u_stream: torch.Tensor, y_st: torch.Tensor, y_ph: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        return wrapper(y_st, y_ph, u_stream, *rest)

    return forward


def lag_band_feature_mask(
    anchors: torch.Tensor, bands: Mapping[str, Tuple[int, int]], sequence_length: int, channels: int
) -> Tuple[torch.Tensor, Dict[str, int]]:
    r"""Group the source stream's positions by lag band relative to each row's own anchor.

    Step $s$ of row $b$ belongs to band $[\ell_{lo}, \ell_{hi}]$ exactly when
    $\ell_{lo} \le t_b - s \le \ell_{hi}$, over every channel at once -- the same positions the
    occlusion analysis zeroes. Everything else -- steps after the anchor and beyond the furthest
    band -- is group $0$ and is ablated too, so the check that removing it changes nothing is on
    the table rather than assumed.

    Args:
        anchors: Each row's anchor as a stored step, $(N,)$.
        bands: ``{name: (lo, hi)}`` inclusive lag pairs.
        sequence_length: $T$.
        channels: The source width.

    Returns:
        ``(mask, group_of_band)``: the $(N, T, C)$ ``long`` mask and each band's group id.
    """
    steps = torch.arange(int(sequence_length), device=anchors.device)[None, :]
    lag = anchors.to(torch.long)[:, None] - steps                            # (N, T)
    mask = torch.zeros(anchors.shape[0], int(sequence_length), dtype=torch.long, device=anchors.device)
    groups: Dict[str, int] = {}
    for index, (name, (low, high)) in enumerate(bands.items(), start=1):
        mask = torch.where((lag >= int(low)) & (lag <= int(high)), torch.full_like(mask, index), mask)
        groups[str(name)] = index
    return mask[:, :, None].expand(-1, -1, int(channels)).contiguous(), groups


def ablate_lag_bands(
    wrapper: AnchorReadout,
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    columns: torch.Tensor,
    anchors: Sequence[int],
    bands: Mapping[str, Tuple[int, int]],
    *,
    coordinates: Optional[torch.Tensor] = None,
    perturbations_per_eval: int = 4,
) -> Dict[str, np.ndarray]:
    r"""Zero the source on each lag band relative to each row's anchor and re-read the readout.

    Captum's feature ablation with the band grouping of :func:`lag_band_feature_mask`; the
    reported number is $f(x^{\setminus \mathrm{band}}) - f(x)$, the **occlusion sign** -- positive
    means the readout rose without the band -- rather than Captum's own $f(x) - f(x^{\setminus})$.

    Args:
        wrapper: The readout module.
        inputs: The row-expanded input tuple.
        extra: The row-expanded extra tuple.
        columns: Per-row anchor-axis positions.
        anchors: Per-row anchor steps.
        bands: The lag bands.
        coordinates: Per-row coordinates, or ``None``.
        perturbations_per_eval: How many ablations Captum batches per forward.

    Returns:
        ``{band: (N,) delta}`` plus ``'rest'`` for everything outside the bands.
    """
    n_rows = int(columns.shape[0])
    if n_rows == 0:
        return {**{str(name): np.zeros(0) for name in bands}, "rest": np.zeros(0)}
    if coordinates is None:
        coordinates = torch.zeros(n_rows, dtype=torch.long, device=columns.device)
    anchor_tensor = torch.as_tensor(np.asarray(anchors, dtype=np.int64), device=columns.device)
    mask, groups = lag_band_feature_mask(anchor_tensor, bands, inputs[2].shape[1], inputs[2].shape[2])
    # Only the source is ablated: the target streams travel as fixed arguments, since the
    # source-null baseline holds them at their own values and ablating them would change nothing.
    source = FeatureAblation(_source_only(wrapper)).attribute(
        inputs[2].detach(),
        baselines=torch.zeros_like(inputs[2]),
        feature_mask=mask,
        additional_forward_args=(inputs[0].detach(), inputs[1].detach(), extra[0], extra[1], columns, coordinates),
        perturbations_per_eval=int(perturbations_per_eval),
    )
    source = source.detach().cpu().to(torch.float64).numpy()
    host_mask = mask.detach().cpu().numpy()
    deltas: Dict[str, np.ndarray] = {}
    for name, group in list(groups.items()) + [("rest", 0)]:
        values = np.full(n_rows, np.nan)
        for row in range(n_rows):
            hit = np.argwhere(host_mask[row] == group)
            if hit.size:
                # Captum writes one value on every position of a group; the negation is the sign
                # convention stated above.
                values[row] = -float(source[row, hit[0][0], hit[0][1]])
        deltas[name] = values
    return deltas


def layer_attribution(
    wrapper: AnchorReadout,
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    columns: torch.Tensor,
    *,
    baseline: str,
    coordinates: Optional[torch.Tensor] = None,
    n_steps: int = IG_STEPS,
    internal_batch_size: int = IG_INTERNAL_BATCH_SIZE,
    entry_fraction: float = BASELINE_ENTRY_FRACTION,
) -> Dict[str, np.ndarray]:
    r"""Integrated gradients on the layer where the source enters the latent, per row.

    In the attentive cells the layers are the posterior head's per-head **fusion** modules --
    one per head under the head-structured posterior, each called once per forward and each
    reading the target state beside its own head's attended summary -- so the per-head split is
    $\sum_d$ of the attribution over head $m$'s fused feature at the row's anchor. Under the
    source-null baseline the target state is fixed along the path, so what flows through the
    fusion is the source's effect alone and the split sums to the readout difference; under the
    all-zero baseline the prior's direct route into the posterior bypasses the fusion and the
    split is partial, which is why the pass takes it under the source-null baseline only. A flat
    (non-head-structured) posterior has one fusion and no split, and the record says so. In the
    residual cell the layer is the proposal head and the attribution is on its **output**, the
    per-lag mean and scale proposals, summed over the latent coordinates at the row's anchor: a
    per-lag split of the readout through the summation and the limiter, which the cell's own
    arithmetic cannot allocate.

    Args:
        wrapper: The readout module.
        inputs: The row-expanded input tuple.
        extra: The row-expanded extra tuple.
        columns: Per-row anchor-axis positions.
        baseline: One of :data:`BASELINES`.
        coordinates: Per-row coordinates, or ``None``.
        n_steps: Integration steps.
        internal_batch_size: Captum's batch of interpolated inputs.
        entry_fraction: Where the path enters the baseline.

    Returns:
        ``{'per_unit': (N, M or L), 'total': (N,)}`` -- the per-head or per-lag attribution and its
        sum. Empty when the cell's model does not build the layer.
    """
    n_rows = int(columns.shape[0])
    model = wrapper.model
    cell = wrapper.cell
    if coordinates is None:
        coordinates = torch.zeros(n_rows, dtype=torch.long, device=columns.device)
    if cell.dense_latent:
        head = model.posterior_head
        layer = list(head.fusion) if getattr(head, "head_structured", False) else None
    else:
        head = getattr(model, "proposal_head", None)
        # The head's own single-tensor output where it declares one: on a mean-only arm the head
        # returns ``(mean, None)``, and a layer hook cannot clone a tuple carrying ``None``.
        layer = getattr(head, "attribution_layer", head)
    if n_rows == 0 or layer is None:
        return {"per_unit": np.zeros((0, 0)), "total": np.zeros(0)}
    exact = baselines_for(baseline, inputs)
    start = entry_point(inputs, exact, entry_fraction)
    forward_args = (extra[0], extra[1], columns, coordinates)
    with warnings.catch_warnings():
        # Captum warns that a list of layers must not be a chain; the per-head fusion modules
        # each read the target state and their own head, so the warning does not apply.
        warnings.simplefilter("ignore", category=UserWarning)
        method = LayerIntegratedGradients(wrapper, layer)
    # The residual cell chunks its proposal pass in training; a chunked layer is called several
    # times per forward and a hook sees only the last call, so the pass runs unchunked here and
    # the setting is restored whatever happens.
    chunk_names = ("anchor_chunk", "lag_chunk")
    saved = {name: getattr(model, name, None) for name in chunk_names if hasattr(model, name)}
    for name in saved:
        setattr(model, name, None)
    try:
        out = method.attribute(
            tuple(x.detach() for x in inputs),
            baselines=tuple(x.detach() for x in start),
            additional_forward_args=forward_args,
            n_steps=int(n_steps),
            internal_batch_size=max(int(internal_batch_size), n_rows),
        )
    finally:
        for name, value in saved.items():
            setattr(model, name, value)
    parts = [part for part in (out if isinstance(out, (tuple, list)) else [out]) if part is not None]
    rows = torch.arange(n_rows, device=columns.device)
    if cell.dense_latent:
        # One fused feature per head, each (B, T, d): the split is their sums at the anchor.
        anchors = wrapper.anchor_steps(columns)
        per_unit = torch.stack([part[rows, anchors].sum(dim=-1) for part in parts], dim=1)   # (N, M)
    else:
        # The proposal head's outputs at the one anchor the attributed forward decodes: the mean
        # proposals, and the scale proposals where the arm has them. Both reach the readout -- the
        # divergence carries the scale update too -- so the per-lag split sums the two channels
        # over the latent coordinates.
        per_unit = torch.stack([part[:, 0].sum(dim=-1) for part in parts], dim=0).sum(dim=0)
    return {"per_unit": _host(per_unit), "total": _host(per_unit.sum(dim=-1))}


# =============================================================================
# Reductions (numpy only from here on)
# =============================================================================
def time_profile(maps: np.ndarray) -> np.ndarray:
    """Sum an $(N, T, C)$ attribution over its channels: $(N, T)$."""
    return np.asarray(maps, dtype=np.float64).sum(axis=-1)


def channel_profile(maps: np.ndarray) -> np.ndarray:
    """Sum an $(N, T, C)$ attribution over its steps: $(N, C)$."""
    return np.asarray(maps, dtype=np.float64).sum(axis=1)


def lag_profile(source_time: np.ndarray, anchors: Sequence[int], n_lags: int) -> np.ndarray:
    r"""Re-index a source time profile by offset from the anchor: $q_\ell = p_{t_a - \ell}$.

    ``NaN`` where $t_a - \ell$ falls before the record: a lag the anchor could not read is not a
    lag that was read and found empty.

    Args:
        source_time: $(N, T)$ per-step source attribution.
        anchors: Per-row anchor steps.
        n_lags: $L$.

    Returns:
        $(N, L)$.
    """
    profile = np.asarray(source_time, dtype=np.float64)
    out = np.full((profile.shape[0], int(n_lags)), np.nan)
    for row, anchor in enumerate(anchors):
        for lag in range(int(n_lags)):
            step = int(anchor) - lag
            if step >= 0:
                out[row, lag] = profile[row, step]
    return out


def band_sums(profile: np.ndarray, groups: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Sum a per-channel (or per-lag) profile over each group's positions.

    Args:
        profile: $(N, K)$.
        groups: ``{name: positions}``.

    Returns:
        ``{name: (N,)}``; a group with no position in range sums to ``NaN``.
    """
    values = np.asarray(profile, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for name, positions in groups.items():
        index = np.asarray(positions, dtype=np.int64)
        index = index[(index >= 0) & (index < values.shape[1])]
        out[str(name)] = np.nansum(values[:, index], axis=1) if index.size else np.full(values.shape[0], np.nan)
    return out


def lag_band_groups(bands: Mapping[str, Tuple[int, int]], n_lags: int) -> Dict[str, np.ndarray]:
    """The lag positions each inclusive band covers, clipped to the window."""
    return {
        str(name): np.arange(max(int(low), 0), min(int(high), int(n_lags) - 1) + 1)
        for name, (low, high) in bands.items()
    }


def agreement(lag_profiles: np.ndarray, model_profiles: np.ndarray) -> Dict[str, np.ndarray]:
    r"""How the lag-aligned attribution magnitude agrees with the model's own lag readout, per row.

    Two statistics, because each answers what the other cannot: the Pearson correlation between
    $|q_\ell|$ and the model's profile over the lags both carry, and the Jensen--Shannon distance
    between the two normalised to distributions over the same lags -- bounded, and blind to a
    scale the correlation would be sensitive to.

    Args:
        lag_profiles: $(N, L)$ attribution by lag.
        model_profiles: $(N, L)$ the model's own profile.

    Returns:
        ``{'lag_corr': (N,), 'lag_js': (N,)}``, ``NaN`` where fewer than three lags are shared or
        either profile carries no mass.
    """
    left = np.abs(np.asarray(lag_profiles, dtype=np.float64))
    right = np.asarray(model_profiles, dtype=np.float64)
    corr = np.full(left.shape[0], np.nan)
    js = np.full(left.shape[0], np.nan)
    for row in range(left.shape[0]):
        shared = np.isfinite(left[row]) & np.isfinite(right[row])
        if shared.sum() < 3:
            continue
        a, b = left[row][shared], right[row][shared]
        if a.std() > 0.0 and b.std() > 0.0:
            corr[row] = float(np.corrcoef(a, b)[0, 1])
        js[row] = lag_hist.jensen_shannon(a, b)
    return {"lag_corr": corr, "lag_js": js}


def channel_groups_from_map(channel_map: Optional[pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
    """Read the declared-axis channel map into ``{stream: {band: positions}}``.

    The attributions live on the model's **input** axis, which is the declared width of each
    stream, so the join goes through the declared-axis map rather than the kept-axis one the
    per-channel decoder readouts use.

    Args:
        channel_map: The ``band_channel_map.csv`` frame, or ``None``.

    Returns:
        The groups, empty when there is no map.
    """
    if channel_map is None or channel_map.empty:
        return {}
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for stream in STREAMS:
        rows = channel_map[channel_map["stream"].astype(str) == stream]
        if rows.empty:
            continue
        bands: Dict[str, List[int]] = {}
        for _, row in rows.iterrows():
            bands.setdefault(str(row["band"]), []).append(int(row["channel"]))
        groups[stream] = {name: np.asarray(sorted(index), dtype=np.int64) for name, index in bands.items()}
    return groups


# =============================================================================
# Figures
# =============================================================================
# =============================================================================
# The figures
# =============================================================================
#: The descriptive name of each readout, as the records carry it; a figure uses :data:`READOUT_SHORT`.
READOUT_TITLES: Mapping[str, str] = {
    READOUT_KLD: "divergence $K_t$",
    READOUT_PRED_GAP: "forecast gap (base $-$ full)",
    READOUT_NLL_FULL: "full-branch score",
    READOUT_NLL_BASE: "base-branch score",
    READOUT_MSE_FULL: "full-branch squared error",
    READOUT_MSE_GAP: "squared-error gap (base $-$ full)",
    READOUT_NLL_HORIZON: "full-branch score at one step",
    READOUT_KLD_DIM: "top coordinate $K_{t,d}$",
    READOUT_LAG_BAND: "lag readout on a band",
}

#: Colours of the readouts when several are overlaid on one lag axis: the main readouts in the
#: shared palette, the lag-band readouts in the band colours the rest of the family uses, in
#: declaration order.
READOUT_COLOURS: Mapping[str, str] = {
    READOUT_KLD: figures.COLOR_BLUE, READOUT_PRED_GAP: figures.COLOR_VERMILLION,
    READOUT_NLL_FULL: figures.COLOR_GREEN, READOUT_MSE_FULL: figures.COLOR_PURPLE,
    READOUT_MSE_GAP: figures.COLOR_ORANGE, READOUT_NLL_HORIZON: figures.COLOR_GRAY,
}
BAND_COLOURS: Tuple[str, ...] = (figures.COLOR_ORANGE, figures.COLOR_GREEN, figures.COLOR_PURPLE, "#56B4E9", "#999999")

#: The colour of a cell the model never read: a cold step of a channel, or a step after the anchor.
UNREAD_COLOUR = figures.COLOR_LIGHT_GRAY

#: The example page's geometry: the samples pages' width, and inches per unit of row height, so
#: a page here and a sample page there put one stored second at the same place on the sheet.
EXAMPLE_PAGE_WIDTH = 14.0
EXAMPLE_ROW_INCHES = 2.2
EXAMPLE_HEADER_INCHES = 0.75
#: Row heights on the example page, in units of :data:`EXAMPLE_ROW_INCHES`.
EXAMPLE_RAW_ROW = 0.7
EXAMPLE_INPUT_ROW = 1.25
EXAMPLE_MAP_ROW = 1.0
EXAMPLE_LATENT_ROW = 0.6
EXAMPLE_LAYER_ROW = 0.8
EXAMPLE_LAG_ROW = 1.1

#: The overview's width per class example, in inches.
OVERVIEW_COLUMN_WIDTH = 5.0


def example_variants(
    lag_bands: Mapping[str, Tuple[int, int]], horizons: Optional[Mapping[str, int]] = None
) -> List[Tuple[str, str, Tuple[int, int], int]]:
    """Every readout an example anchor is attributed for, as ``(readout, tag, lag_band, horizon)``.

    The plain example readouts first, then the per-horizon score at each named step, then the
    lag readout on each configured band. The tag is the row's ``band`` column: empty for a plain
    readout, ``h<step>`` for a horizon step, the band's name for a band.

    Args:
        lag_bands: The configured lag bands.
        horizons: ``{name: step}`` from :func:`horizon_steps`, or ``None`` for none.

    Returns:
        The variants, in page order.
    """
    plan: List[Tuple[str, str, Tuple[int, int], int]] = [(name, "", (0, 0), 0) for name in EXAMPLE_READOUTS]
    plan += [(READOUT_NLL_HORIZON, f"h{int(step)}", (0, 0), int(step)) for step in dict(horizons or {}).values()]
    plan += [(READOUT_LAG_BAND, str(name), (int(span[0]), int(span[1])), 0) for name, span in lag_bands.items()]
    return plan


def variant_label(readout: str, tag: str = "") -> str:
    """What a readout variant is called on a figure: the readout's title, qualified by its tag."""
    if readout == READOUT_LAG_BAND:
        return f"lag readout on band {tag!r}"
    if readout == READOUT_NLL_HORIZON:
        return f"full-branch score at horizon step {tag[1:] if tag.startswith('h') else tag}"
    return READOUT_TITLES.get(readout, readout)


#: What each readout is called on a **figure**: a short noun phrase for a panel title, a legend
#: entry or a tick label. :data:`READOUT_TITLES` keeps the descriptive names the records carry
#: (for example the ``label`` column of the block table); the sign convention of the gaps, base
#: minus full, is in the guide.
READOUT_SHORT: Mapping[str, str] = {
    READOUT_KLD: "Divergence $K_t$",
    READOUT_PRED_GAP: "Forecast gap",
    READOUT_NLL_FULL: "Full-branch score",
    READOUT_NLL_BASE: "Base-branch score",
    READOUT_MSE_FULL: "Full squared error",
    READOUT_MSE_GAP: "Squared-error gap",
    READOUT_NLL_HORIZON: "Score at one step",
    READOUT_KLD_DIM: "Top $K_{t,d}$",
    READOUT_LAG_BAND: "Lag band",
}


def variant_short(readout: str, tag: str = "") -> str:
    """What a readout variant is called on a figure: :data:`READOUT_SHORT`, qualified by its tag."""
    if readout == READOUT_LAG_BAND:
        return f"Lag band {tag}"
    if readout == READOUT_NLL_HORIZON:
        return f"Score, step {tag[1:] if tag.startswith('h') else tag}"
    return READOUT_SHORT.get(readout, readout)


def variant_colour(readout: str, tag: str, lag_bands: Sequence[str]) -> str:
    """The overlay colour of a readout variant: the readout's own, or its band's."""
    if readout == READOUT_LAG_BAND and tag in lag_bands:
        return BAND_COLOURS[list(lag_bands).index(tag) % len(BAND_COLOURS)]
    return READOUT_COLOURS.get(readout, figures.COLOR_GRAY)


def _empty(ax: Any, title: str = "") -> None:
    """Mark an axes as holding nothing."""
    ax.text(
        0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
        ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
    )
    ax.set_title(title)
    figures.style_axes(ax, grid="none")


def input_norm(values: np.ndarray) -> Optional[Any]:
    """A linear colour scale over the 1st to 99th percentile of a field's finite cells, or ``None``."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    low, high = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0))
    return mcolors.Normalize(vmin=low, vmax=high if high > low else low + 1.0)


#: How the example maps' colour scales are shared. Stated in ``ATTRIBUTION.md`` and the figure
#: guide; the figures that draw them no longer print it.
SHARED_SCALE_NOTE = (
    "colour scales are shared by every class example: the input coefficients one linear scale per "
    "stream (1st to 99th percentile of the read cells of all examples), each attribution map one "
    "symmetric-log scale per readout, baseline and stream at plus or minus its largest magnitude "
    "over all examples, so a colour means the same value on every page; the raw FHR and UP are on "
    "the fixed CTG scales"
)


def example_norms(examples: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, ...], Any]:
    r"""One colour scale per map kind, shared by every example drawn with it.

    Keys: ``('input', stream)``, the standardised coefficients over the read cells of every
    example (:func:`input_norm`); ``(readout, baseline, band, stream)``, an attribution map over
    the cells the model read up to each example's anchor (:func:`signed_log_norm`); and
    ``('latent',)`` and ``('layer',)`` for the two rows off the time axis. A kind whose every cell
    is zero or blank carries no key, and its map falls back to a scale of its own.

    Args:
        examples: The example records, as :func:`attribution_pass.attribute_example` builds them.

    Returns:
        The scales by key.
    """
    pooled: Dict[Tuple[str, ...], List[np.ndarray]] = {}
    for item in examples:
        anchor = int(item["anchor"])
        for stream in STREAMS:
            live = item.get(f"live_{stream}")
            field = (item.get("inputs") or {}).get(stream)
            if field is not None:
                pooled.setdefault(("input", stream), []).append(masked_field(field, live).ravel())
            for (readout, baseline, band), entry in (item.get("maps") or {}).items():
                pooled.setdefault((readout, baseline, band, stream), []).append(
                    masked_field(entry[stream], live, anchor=anchor).ravel()
                )
        pooled.setdefault(("latent",), []).extend(
            np.asarray(value, dtype=np.float64).ravel() for value in (item.get("latent") or {}).values()
        )
        pooled.setdefault(("layer",), []).extend(
            np.asarray(value, dtype=np.float64).ravel() for value in (item.get("layer") or {}).values()
        )
    norms: Dict[Tuple[str, ...], Any] = {}
    for key, pieces in pooled.items():
        values = np.concatenate(pieces) if pieces else np.zeros(0)
        norm = input_norm(values) if key[0] == "input" else signed_log_norm(values)
        if norm is not None:
            norms[key] = norm
    return norms


def _mark_anchor(ax: Any, anchor: int, horizon: int) -> None:
    """Rule the anchor's stored step and shade the forecast block it scores, on a seconds axis."""
    if horizon:
        ax.axvspan((anchor + 0.5) * SECONDS_PER_STEP, (anchor + 0.5 + int(horizon)) * SECONDS_PER_STEP,
                   color=figures.COLOR_VERMILLION, alpha=0.12, linewidth=0, zorder=1)
    ax.axvline(float(anchor) * SECONDS_PER_STEP, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_REGULAR, zorder=3)


def _raw_row(ax: Any, item: Mapping[str, Any], name: str, *, steps: int, titled: bool = True) -> None:
    r"""One raw signal of an example's segment, on the stored-time axis every map row uses.

    Raw sample $i$ sits at $i / f_s$ seconds and stored step $t$ at $t\,\Delta$, as on the samples
    pages, so the anchor rule and the forecast window fall on the same instant here and on the
    maps below; the signal is drawn on the fixed CTG scale of the traces, so two examples share
    one grid.

    Args:
        ax: The axes.
        item: The example record, read for ``raw``, ``raw_units``, ``anchor`` and ``horizon``.
        name: ``'fhr'`` or ``'up'``.
        steps: $T$, which fixes the axis to the whole segment.
        titled: Whether to title the panel; an overview titles its first column only.
    """
    title = traces.RAW_SIGNAL_TITLES[name] if titled else ""
    values = (item.get("raw") or {}).get(name)
    if values is None:
        _empty(ax, title)
    else:
        values = np.asarray(values, dtype=np.float64)
        traces.draw_raw_signal(
            ax, np.arange(values.size) / float(events.FS_RAW), values, name,
            str((item.get("raw_units") or {}).get(name, "normalised")),
        )
        ax.set_title(title)
        figures.style_axes(ax)
    _mark_anchor(ax, int(item["anchor"]), int(item.get("horizon", 0) or 0))
    ax.set_xlim(0.0, float(steps) * SECONDS_PER_STEP)
    ax.set_xlabel("stored time (s)")


def _stream_map(
    figure: Any, ax: Any, field: Optional[np.ndarray], *, title: str, anchor: int,
    live: Optional[np.ndarray], n_scattering: Optional[int] = None, colorbar_label: str = "",
    kind: str = "signed", cax: Any = None, horizon: int = 0, norm: Any = None, colorbar: bool = True,
) -> Any:
    r"""One (channel x stored second) map with the cells the model never read blanked.

    Two kinds. An **input** map shows the standardised coefficients on a sequential scale with
    robust (1st to 99th percentile) limits taken over the live cells only; the cold cells of every
    channel, which the input gate multiplies by zero, are blanked rather than drawn, so the
    warm-up region neither sets the colour scale nor reads as data. A **signed** map shows an
    attribution on a symmetric-log scale about zero, with the cold cells and every step after the
    anchor blanked, because both are exactly zero by construction and a zero the model could not
    have produced is not a finding. The anchor is ruled and the forecast block it scores shaded.
    The axis runs over the whole segment, $[0, T\Delta]$, as the raw rows and the samples pages do.

    Args:
        figure: The figure, for the colourbar.
        ax: The axes.
        field: $(T, C)$, or ``None`` for an absent map.
        title: The panel title.
        anchor: The anchor's stored step, ruled in vermilion.
        live: Each channel's first live step, or ``None``.
        n_scattering: For a target map, the width of the scattering block, so the boundary to the
            phase-harmonic block is ruled.
        colorbar_label: The colourbar label.
        kind: ``'input'`` or ``'signed'``.
        cax: An axes to draw the colourbar into, or ``None`` to steal room beside ``ax``.
        horizon: The block length $H$, for the shaded forecast window; $0$ shades nothing.
        norm: A colour scale shared with other maps (:func:`example_norms`), or ``None`` for this
            map's own.
        colorbar: Whether to draw the colourbar; a row of maps on one shared scale draws one.

    Returns:
        The image, or ``None`` when nothing was drawn.
    """
    if field is None:
        _empty(ax, title)
        if cax is not None:
            cax.set_axis_off()
        return None
    steps = int(np.asarray(field).shape[0])
    masked = masked_field(field, live, anchor=None if kind == "input" else int(anchor))
    if kind == "input":
        colormap = plt.get_cmap("viridis").with_extremes(bad=UNREAD_COLOUR)
        norm = norm or input_norm(masked) or mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        colormap = plt.get_cmap("RdBu_r").with_extremes(bad=UNREAD_COLOUR)
        norm = norm or signed_log_norm(masked) or mcolors.Normalize(vmin=-1.0, vmax=1.0)
    left, right = sample_cell_edges(steps, float(SECONDS_PER_STEP))
    image = ax.imshow(
        masked.T, aspect="auto", origin="upper", cmap=colormap, norm=norm, interpolation="none",
        extent=(left, right, masked.shape[1] - 0.5, -0.5),
    )
    if n_scattering and 0 < int(n_scattering) < masked.shape[1]:
        ax.axhline(int(n_scattering) - 0.5, color=figures.COLOR_BLACK, linewidth=plt.rcParams["axes.linewidth"])
    _mark_anchor(ax, int(anchor), int(horizon))
    ax.set_xlim(0.0, float(steps) * SECONDS_PER_STEP)
    ax.set_title(title)
    ax.set_xlabel("stored time (s)")
    ax.set_ylabel("declared channel")
    if colorbar:
        _attach_colorbar(figure, image, ax=ax, cax=cax, label=colorbar_label, norm=norm)
    elif cax is not None:
        cax.set_axis_off()
    figures.style_axes(ax, grid="none")
    return image


def _lag_view(
    ax: Any, profile: np.ndarray, model_profile: np.ndarray, *, lag_seconds: np.ndarray,
    cell: CellBinding, title: str, legend: bool = True,
) -> None:
    """Normalised $|q_\\ell|$ beside the normalised model lag readout, on one share axis.

    One axis rather than a twin: both are shares per lag once normalised, and two scales on one
    panel is the one layout a reader cannot check by eye. Symmetric-log, because one lag's share
    is often a hundred times another's and a linear axis shows only the one. The agreement of the
    two curves (``lag_corr``, ``lag_js``) is in the per-row table, not in the title. ``legend=False``
    leaves the keys to the panel of the figure that carries them.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if not np.isfinite(profile).any():
        _empty(ax, title)
        return
    share = lag_hist.normalise(np.abs(profile))[0]
    model_share = lag_hist.normalise(np.asarray(model_profile, dtype=np.float64))[0]
    ax.plot(lag_seconds, share, color=figures.COLOR_BLUE, linewidth=figures.LINE_REGULAR, label="|attribution|")
    if np.isfinite(model_share).any():
        ax.plot(lag_seconds, model_share, color=figures.COLOR_ORANGE, linewidth=figures.LINE_THIN,
                label="model lag readout")
    ax.set_title(title)
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("share per lag")
    symlog_legend(ax, share, model_share, ncol=2, legend=legend)
    figures.style_axes(ax)


def _example_title(item: Mapping[str, Any], separator: str = ", ") -> str:
    """The identity of one example anchor: the recording with its subgroup, then its class.

    The anchor and its epoch are in the examples manifest, not on the figure.
    """
    return separator.join((
        f"guid {item['guid']}, subgroup {item[labels.SUBGROUP_COLUMN]}",
        f"class {item[labels.CLASS_COLUMN]}",
    ))


#: The overview's rows, top to bottom, and their heights in units of :data:`EXAMPLE_ROW_INCHES`.
OVERVIEW_ROWS: Tuple[Tuple[str, float], ...] = (
    ("raw_fhr", EXAMPLE_RAW_ROW), ("raw_up", EXAMPLE_RAW_ROW),
    ("input_target", EXAMPLE_MAP_ROW), ("input_source", EXAMPLE_MAP_ROW),
    ("map_target", EXAMPLE_MAP_ROW), ("map_source", EXAMPLE_MAP_ROW), ("lags", EXAMPLE_LAG_ROW),
)


def build_map_figure(
    examples: Sequence[Mapping[str, Any]],
    *,
    lag_seconds: np.ndarray,
    cell: CellBinding,
    caveat: str,
    norms: Optional[Mapping[Tuple[str, ...], Any]] = None,
) -> Any:
    r"""One column per class example: the physiology, the inputs, and what $K_t$ was attributed to.

    Every column is one anchor of one segment, and every row but the last is on that segment's
    stored-time axis, $[0, T\Delta]$ seconds, identical in every column, with the anchor ruled and
    its forecast block shaded. Top to bottom: the raw FHR and the raw UP on the fixed CTG scales;
    the two **input** streams the encoders read, standardised coefficients with the cold cells
    blanked; the attribution of $K_t$ to the target under the **all-zero** baseline, the only
    baseline under which the target moves, and to the source under the **source-null** baseline,
    the primary comparison; and, off the time axis, the lag-aligned source attribution against the
    model's own lag readout at that anchor. Each map row is drawn on one colour scale shared by
    every column (:func:`example_norms`), so a colour means the same value for every class; the
    scale is stated in the guide, not under the figure. Panels are titled in the first column
    only and the legend is kept in the first lag panel; a column is named by its header.

    Args:
        examples: One per class, as :func:`attribution_pass.attribute_example` builds them.
        lag_seconds: The compensated lag axis.
        cell: The cell binding.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).
        norms: The shared colour scales, or ``None`` to derive them from ``examples``.

    Returns:
        The figure, laid out; the caller renders and closes it.
    """
    norms = example_norms(examples) if norms is None else norms
    n_cols = max(len(examples), 1)
    heights = [height for _name, height in OVERVIEW_ROWS]
    figure_height = sum(heights) * EXAMPLE_ROW_INCHES + EXAMPLE_HEADER_INCHES
    figure = plt.figure(figsize=(OVERVIEW_COLUMN_WIDTH * n_cols + 1.2, figure_height))
    bottom = max(0.02, 0.35 / figure_height) + figures.caveat_note(figure, caveat)
    grid = GridSpec(
        len(OVERVIEW_ROWS), n_cols + 1, figure=figure, height_ratios=heights,
        width_ratios=[1.0] * n_cols + [0.035], left=0.07, right=0.94,
        top=1.0 - EXAMPLE_HEADER_INCHES / figure_height, bottom=bottom, hspace=0.75, wspace=0.3,
    )
    shared: Optional[Any] = None
    last_time_row = max(row for row, (name, _height) in enumerate(OVERVIEW_ROWS) if name != "lags")
    for col in range(n_cols):
        item = examples[col] if col < len(examples) else None
        for row, (name, _height) in enumerate(OVERVIEW_ROWS):
            time_row = name != "lags"
            ax = figure.add_subplot(grid[row, col], sharex=shared if time_row else None)
            if time_row and shared is None:
                shared = ax
            last = col == n_cols - 1
            cax = figure.add_subplot(grid[row, n_cols]) if last else None
            if cax is not None:
                cax.set_label("<colorbar>")
            if item is None:
                _empty(ax)
                if cax is not None:
                    cax.set_axis_off()
                continue
            anchor, horizon = int(item["anchor"]), int(item.get("horizon", 0) or 0)
            steps = int(np.asarray(item["inputs"][STREAM_TARGET]).shape[0])
            first = col == 0
            if name.startswith("raw_"):
                _raw_row(ax, item, name[len("raw_"):], steps=steps, titled=first)
                if cax is not None:
                    cax.set_axis_off()
            elif name.startswith("input_"):
                stream = name[len("input_"):]
                _stream_map(
                    figure, ax, (item.get("inputs") or {}).get(stream),
                    title=f"{stream.capitalize()} input" if first else "", anchor=anchor,
                    live=item.get(f"live_{stream}"),
                    n_scattering=item.get("n_scattering") if stream == STREAM_TARGET else None,
                    kind="input", cax=cax, horizon=horizon, norm=norms.get(("input", stream)),
                    colorbar=last, colorbar_label="standardised",
                )
            elif name.startswith("map_"):
                stream = name[len("map_"):]
                baseline = BASELINE_ALL_ZERO if stream == STREAM_TARGET else BASELINE_SOURCE_NULL
                entry = (item.get("maps") or {}).get((READOUT_KLD, baseline, ""))
                _stream_map(
                    figure, ax, None if entry is None else entry[stream],
                    title=f"$K_t$ to {stream}" if first else "", anchor=anchor,
                    live=item.get(f"live_{stream}"),
                    n_scattering=item.get("n_scattering") if stream == STREAM_TARGET else None,
                    colorbar_label=READOUT_UNITS[READOUT_KLD], cax=cax, horizon=horizon,
                    norm=norms.get((READOUT_KLD, baseline, "", stream)), colorbar=last,
                )
            else:
                null = (item.get("maps") or {}).get((READOUT_KLD, BASELINE_SOURCE_NULL, ""))
                if null is None:
                    _empty(ax, "$K_t$ by lag" if first else "")
                else:
                    _lag_view(ax, null["lag_profile"], item["model_profile"], lag_seconds=lag_seconds,
                              cell=cell, title="$K_t$ by lag" if first else "", legend=first)
                if cax is not None:
                    cax.set_axis_off()
            if not first:
                ax.set_ylabel("")
            if time_row and row < last_time_row:
                ax.set_xlabel("")
            if row == 0:
                ax.annotate(
                    _example_title(item, separator="\n"), xy=(0.5, 1.0), xycoords="axes fraction",
                    xytext=(0.0, 18.0), textcoords="offset points", ha="center", va="bottom",
                    fontsize=figures.FONT_SMALL,
                )
    figures.mark_laid_out(figure)
    return figure


def build_example_figure(
    item: Mapping[str, Any],
    *,
    lag_seconds: np.ndarray,
    cell: CellBinding,
    caveat: str,
    lag_bands: Mapping[str, Tuple[int, int]],
    horizons: Optional[Mapping[str, int]] = None,
    norms: Optional[Mapping[Tuple[str, ...], Any]] = None,
) -> Any:
    r"""One anchor of one recording, every readout, on one stored-time axis: the sample page's layout.

    The page is a stack of full-width rows on one shared time axis, laid out as the samples pages
    are -- one data column and one colour-axis column, every row spanning the whole segment,
    $[0, T\Delta]$ seconds -- so a column of the page is the same stored second on every row and a
    reader compares maps by looking down rather than across. The rows: the raw FHR and the raw UP
    of the segment on the fixed CTG scales, the physiology every map below is read against; the
    two input streams the encoders read (cold cells blanked); then, per readout variant, its target
    attribution under the all-zero baseline and its source attribution under the source-null
    baseline, each on a symmetric-log scale, titled with the readout and the stream (the values at
    the input and at the exact null are in the per-row table); then three rows off the time axis -- the latent at the anchor (prior mean, source
    shift, per-coordinate divergence), the layer split per readout on the cell's own axis (head or
    lag), and every readout's lag-aligned source attribution overlaid on one lag axis against the
    model's lag readout. Every time row rules the anchor and shades its forecast block.

    Args:
        item: The example, as :func:`attribution_pass.attribute_example` builds it.
        lag_seconds: The compensated lag axis.
        cell: The cell binding.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).
        lag_bands: The configured lag bands, in the order their rows are drawn.
        horizons: The named horizon steps, in the order their rows are drawn, or ``None``.
        norms: The colour scales shared by every example page (:func:`example_norms`), or
            ``None`` for scales of this page's own.

    Returns:
        The figure, laid out; the caller renders and closes it.
    """
    maps = item.get("maps") or {}
    inputs = item.get("inputs") or {}
    anchor = int(item["anchor"])
    horizon = int(item.get("horizon", 0) or 0)
    variants = example_variants(lag_bands, horizons)
    latent = item.get("latent") or {}
    layer = item.get("layer") or {}
    norms = example_norms([item]) if norms is None else norms
    steps = int(np.asarray(inputs[STREAM_TARGET]).shape[0])

    rows: List[Tuple[str, float]] = [
        ("raw_fhr", EXAMPLE_RAW_ROW), ("raw_up", EXAMPLE_RAW_ROW),
        ("input_target", EXAMPLE_INPUT_ROW), ("input_source", EXAMPLE_INPUT_ROW),
    ]
    for readout, tag, _band, _step in variants:
        rows += [(f"target:{readout}:{tag}", EXAMPLE_MAP_ROW), (f"source:{readout}:{tag}", EXAMPLE_MAP_ROW)]
    n_time_rows = len(rows)
    if latent:
        rows.append(("latent", EXAMPLE_LATENT_ROW))
    if any(np.asarray(v).size for v in layer.values()):
        rows.append(("layer", EXAMPLE_LAYER_ROW))
    rows.append(("lags", EXAMPLE_LAG_ROW))

    heights = [height for _, height in rows]
    figure_height = sum(heights) * EXAMPLE_ROW_INCHES
    figure = plt.figure(figsize=(EXAMPLE_PAGE_WIDTH, figure_height))
    bottom = max(0.02, 0.35 / figure_height) + figures.caveat_note(figure, caveat)
    grid = GridSpec(
        len(rows), 2, figure=figure, height_ratios=heights, width_ratios=[1.0, 0.022],
        left=0.065, right=0.93, top=1.0 - EXAMPLE_HEADER_INCHES / figure_height, bottom=bottom,
        hspace=0.55, wspace=0.09,
    )
    time_axes: List[Any] = []

    def row_axes(position: int, *, shared: bool) -> Tuple[Any, Any]:
        ax = figure.add_subplot(grid[position, 0], sharex=time_axes[0] if (shared and time_axes) else None)
        cax = figure.add_subplot(grid[position, 1])
        cax.set_label("<colorbar>")
        if shared:
            time_axes.append(ax)
        return ax, cax

    for position, (name, _height) in enumerate(rows[:n_time_rows]):
        ax, cax = row_axes(position, shared=True)
        if name.startswith("raw_"):
            _raw_row(ax, item, name[len("raw_"):], steps=steps)
            cax.set_axis_off()
        elif name.startswith("input_"):
            stream = name[len("input_"):]
            _stream_map(figure, ax, inputs.get(stream), title=f"{stream.capitalize()} input",
                        anchor=anchor, live=item.get(f"live_{stream}"),
                        n_scattering=item.get("n_scattering") if stream == STREAM_TARGET else None,
                        kind="input", cax=cax, horizon=horizon, norm=norms.get(("input", stream)),
                        colorbar_label="standardised")
        else:
            stream, readout, tag = name.split(":", 2)
            baseline = BASELINE_ALL_ZERO if stream == STREAM_TARGET else BASELINE_SOURCE_NULL
            entry = maps.get((readout, baseline, tag))
            _stream_map(
                figure, ax, None if entry is None else entry[stream],
                title=f"{variant_short(readout, tag)}: {stream}", anchor=anchor,
                live=item.get(f"live_{stream}"), n_scattering=item.get("n_scattering") if stream == STREAM_TARGET else None,
                colorbar_label=readout_unit(readout, cell), kind="signed", cax=cax, horizon=horizon,
                norm=norms.get((readout, baseline, tag, stream)),
            )
        if position < n_time_rows - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")

    position = n_time_rows
    if latent:
        ax, cax = row_axes(position, shared=False)
        position += 1
        field = np.stack([np.asarray(latent[key], dtype=np.float64) for key in ("mu_prior", "shift", "kld_dim")], axis=0)
        norm = norms.get(("latent",)) or signed_log_norm(field) or mcolors.Normalize(vmin=-1.0, vmax=1.0)
        image = ax.imshow(field, aspect="auto", origin="upper", cmap="RdBu_r", norm=norm, interpolation="none",
                          extent=(-0.5, field.shape[1] - 0.5, 2.5, -0.5))
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["$\\mu^p$", "$\\mu^q - \\mu^p$", "$K_{t,d}$"])
        top = int(np.argmax(np.asarray(latent["kld_dim"], dtype=np.float64)))
        ax.plot([top], [2], marker="v", color=figures.COLOR_BLACK, markersize=4, linestyle="none")
        ax.set_title("Latent at the anchor")
        ax.set_xlabel("latent coordinate")
        _attach_colorbar(figure, image, ax=ax, cax=cax, label="latent units / nats", norm=norm)
        figures.style_axes(ax, grid="none")

    if any(np.asarray(v).size for v in layer.values()):
        ax, cax = row_axes(position, shared=False)
        position += 1
        keys = [(readout, tag) for readout, tag, _b, _s in variants if np.asarray(layer.get((readout, tag), ())).size]
        width = max(int(np.asarray(layer[key]).size) for key in keys)
        field = np.full((len(keys), width), np.nan)
        for index, key in enumerate(keys):
            values = np.asarray(layer[key], dtype=np.float64).reshape(-1)
            field[index, :values.size] = values
        norm = norms.get(("layer",)) or signed_log_norm(field) or mcolors.Normalize(vmin=-1.0, vmax=1.0)
        if cell.layer_axis == "head":
            extent = (-0.5, width - 0.5, len(keys) - 0.5, -0.5)
        else:
            half = 0.5 * float(SECONDS_PER_STEP)
            extent = (float(lag_seconds[0]) - half, float(lag_seconds[min(width, len(lag_seconds)) - 1]) + half, len(keys) - 0.5, -0.5)
        image = ax.imshow(field, aspect="auto", origin="upper", cmap=plt.get_cmap("RdBu_r").with_extremes(bad=UNREAD_COLOUR),
                          norm=norm, interpolation="none", extent=extent)
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels([variant_short(*key) for key in keys], fontsize=figures.FONT_TINY)
        ax.set_title(f"Attribution per {cell.layer_axis}")
        ax.set_xlabel("head" if cell.layer_axis == "head" else COEFFICIENT_LAG_AXIS_LABEL)
        if cell.layer_axis == "head":
            ax.set_xticks(np.arange(width))
        _attach_colorbar(figure, image, ax=ax, cax=cax, label="readout units", norm=norm)
        figures.style_axes(ax, grid="none")

    ax, cax = row_axes(position, shared=False)
    cax.set_axis_off()
    model_profile = np.asarray(item.get("model_profile", np.full(len(lag_seconds), np.nan)), dtype=np.float64)
    series: List[np.ndarray] = []
    for readout, tag, _band, _step in variants:
        entry = maps.get((readout, BASELINE_SOURCE_NULL, tag))
        if entry is None or not np.isfinite(entry["lag_profile"]).any():
            continue
        share = lag_hist.normalise(np.abs(np.asarray(entry["lag_profile"], dtype=np.float64)))[0]
        series.append(share)
        ax.plot(lag_seconds, share, color=variant_colour(readout, tag, list(lag_bands)),
                linewidth=figures.LINE_REGULAR, linestyle="--" if readout == READOUT_LAG_BAND else "-",
                label=variant_short(readout, tag))
    if np.isfinite(model_profile).any():
        model_share = lag_hist.normalise(model_profile)[0]
        series.append(model_share)
        ax.plot(lag_seconds, model_share, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle=":",
                label="model lag readout")
    if series:
        ax.set_title("Lag profile")
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share per lag")
        symlog_legend(ax, *series, ncol=3)
        figures.style_axes(ax)
    else:
        _empty(ax, "Lag profile")

    figure.suptitle(_example_title(item), fontsize=figures.FONT_NOTE, y=1.0 - 0.3 * EXAMPLE_HEADER_INCHES / figure_height)
    figures.mark_laid_out(figure)
    return figure


def build_block_figure(blocks: pd.DataFrame, *, caveat: str) -> Any:
    r"""Which input block each readout responded to: the four blocks' shares and signed sums.

    Left: per readout variant, the mean over recordings of the **unsigned** attribution's share
    in each of the four input blocks -- target scattering, target phase-harmonic, source
    scattering, source phase-harmonic -- under the all-zero baseline, the one path along which
    every stream moves, as a stacked bar summing to one. Right: the same rows' **signed** block
    sums, on a symmetric-log axis, so a block that raised a readout and one that lowered it are
    told apart. The block score and the squared error answer "which inputs drive the forecast";
    the divergence and the gap answer "which inputs drive the latent change and the gain".

    Args:
        blocks: The block table, one row per (readout, band, baseline, block).
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 2, height_per_row=3.4, width=13.0)
    subset = blocks[blocks["baseline"].astype(str) == BASELINE_ALL_ZERO] if len(blocks) else blocks
    if subset.empty:
        _empty(axes[0, 0], "Share by input block")
        _empty(axes[0, 1], "Signed block sums")
        figures.caveat_note(figure, caveat)
        return figure
    if {"readout", "band"} <= set(subset.columns):
        # The figure's own short names; the table's ``label`` column keeps the descriptive ones.
        subset = subset.assign(label=[
            variant_short(str(readout), "" if pd.isna(band) else str(band))
            for readout, band in zip(subset["readout"], subset["band"])
        ])
    labels_in_order = list(dict.fromkeys(subset["label"].astype(str)))
    y = np.arange(len(labels_in_order))
    colours = dict(zip(BLOCKS, (figures.COLOR_BLUE, "#56B4E9", figures.COLOR_ORANGE, figures.COLOR_VERMILLION)))
    ax = axes[0, 0]
    left = np.zeros(len(labels_in_order))
    for block in BLOCKS:
        part = subset[subset["block"].astype(str) == block].set_index("label")
        share = np.asarray([float(part["share_mean"].get(name, np.nan)) for name in labels_in_order])
        share = np.where(np.isfinite(share), share, 0.0)
        ax.barh(y, share, left=left, color=colours[block], label=block.replace("_", " "), height=0.7)
        left += share
    ax.set_yticks(y)
    ax.set_yticklabels(labels_in_order, fontsize=figures.FONT_TINY)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of |attribution|")
    ax.set_title("Share by input block", fontsize=figures.FONT_SMALL)
    ax.legend(fontsize=figures.FONT_TINY, loc="lower right", ncol=2)
    figures.style_axes(ax)
    ax = axes[0, 1]
    width = 0.8 / len(BLOCKS)
    drawn: List[np.ndarray] = []
    for offset, block in enumerate(BLOCKS):
        part = subset[subset["block"].astype(str) == block].set_index("label")
        values = np.asarray([float(part["signed_mean"].get(name, np.nan)) for name in labels_in_order])
        drawn.append(values)
        ax.barh(y + (offset - (len(BLOCKS) - 1) / 2) * width, values, height=width, color=colours[block], label=block.replace("_", " "))
    ax.set_yticks(y)
    ax.set_yticklabels(labels_in_order, fontsize=figures.FONT_TINY)
    ax.invert_yaxis()
    ax.axvline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
    symlog_axis(ax, *drawn, axis="x")
    ax.set_xlabel("signed attribution (symlog)")
    ax.set_title("Signed block sums", fontsize=figures.FONT_SMALL)
    figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def build_horizon_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    horizons: Mapping[str, int],
    caveat: str,
) -> Any:
    r"""How the near and the far end of the forecast block read the inputs.

    Three panels over the per-horizon score rows. Left: per named horizon step, the mean over
    recordings of the unsigned attribution total of the target stream (all-zero baseline) and of
    the source stream (source-null baseline), with the score itself at the input beside them.
    Middle: the source attribution by lag per horizon step, source-null baseline, magnitude,
    mean over recordings. Right: the target attribution by offset from the anchor per step,
    all-zero baseline. A far step that reads the source more than the near one is a source whose
    information pays later in the block.

    Args:
        rows: The per-row table.
        vectors: The row-aligned arrays.
        lag_seconds: The compensated lag axis.
        horizons: ``{name: step}`` from :func:`horizon_steps`.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 3, height_per_row=3.0, width=15.0)
    if not len(rows) or not horizons:
        for col, title in enumerate(("Stream totals", "Source by lag", "Target by lag")):
            _empty(axes[0, col], title)
        figures.caveat_note(figure, caveat)
        return figure
    readout = rows["readout"].astype(str).to_numpy()
    baseline = rows["baseline"].astype(str).to_numpy()
    tag = rows["band"].astype(str).to_numpy()
    guids = rows["guid"].astype(str).to_numpy()
    names = list(horizons)
    x = np.arange(len(names))
    ax = axes[0, 0]
    bars: List[np.ndarray] = []
    for offset, (stream, base, colour) in enumerate((
        (STREAM_TARGET, BASELINE_ALL_ZERO, figures.COLOR_BLUE), (STREAM_SOURCE, BASELINE_SOURCE_NULL, figures.COLOR_ORANGE),
    )):
        heights = []
        for name in names:
            keep = (readout == READOUT_NLL_HORIZON) & (baseline == base) & (tag == f"h{int(horizons[name])}")
            column = rows[f"{stream}_abs_total"].to_numpy(dtype=np.float64)
            heights.append(float(_per_recording_mean(column[:, None], guids, keep)[0]) if keep.any() else np.nan)
        bars.append(np.asarray(heights))
        ax.bar(x + (offset - 0.5) * 0.38, heights, width=0.38, color=colour, label=stream.capitalize())
    scores = []
    for name in names:
        keep = (readout == READOUT_NLL_HORIZON) & (baseline == BASELINE_SOURCE_NULL) & (tag == f"h{int(horizons[name])}")
        column = rows["value_input"].to_numpy(dtype=np.float64)
        scores.append(float(_per_recording_mean(column[:, None], guids, keep)[0]) if keep.any() else np.nan)
    ax.plot(x, scores, color=figures.COLOR_BLACK, marker="o", markersize=figures.MARKER_SMALL, linewidth=figures.LINE_THIN,
            label="Score at input")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name} (step {int(horizons[name])})" for name in names])
    ax.set_title("Stream totals", fontsize=figures.FONT_SMALL)
    ax.set_ylabel("nats per anchor (symlog)")
    symlog_legend(ax, *bars, np.asarray(scores), ncol=1)
    figures.style_axes(ax)
    keyed = False
    for col, (stream, base, vector, xlabel) in enumerate((
        (STREAM_SOURCE, BASELINE_SOURCE_NULL, "lag_profile", COEFFICIENT_LAG_AXIS_LABEL),
        (STREAM_TARGET, BASELINE_ALL_ZERO, "target_lag_profile", COEFFICIENT_LAG_AXIS_LABEL),
    ), start=1):
        ax = axes[0, col]
        drawn: List[np.ndarray] = []
        for index, name in enumerate(names):
            keep = (readout == READOUT_NLL_HORIZON) & (baseline == base) & (tag == f"h{int(horizons[name])}")
            if not keep.any() or vector not in vectors:
                continue
            profile = _per_recording_mean(np.abs(vectors[vector].astype(np.float64)), guids, keep)
            drawn.append(profile)
            ax.plot(lag_seconds, profile, color=(figures.COLOR_BLUE, figures.COLOR_VERMILLION, figures.COLOR_GREEN)[index % 3],
                    linewidth=figures.LINE_REGULAR, label=f"{name} (step {int(horizons[name])})")
        if not drawn:
            _empty(ax, f"{stream.capitalize()} by lag")
            continue
        ax.set_title(f"{stream.capitalize()} by lag", fontsize=figures.FONT_SMALL)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|attribution| (nats, symlog)")
        # Both panels key the same horizon steps to the same colours: one legend, in the first.
        symlog_legend(ax, *drawn, ncol=2, legend=not keyed)
        keyed = True
        figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def _class_colours(classes: Sequence[str]) -> Dict[str, str]:
    """The severity palette for a list of class names."""
    return figures.group_colors([str(name) for name in classes])


def _per_recording_mean(matrix: np.ndarray, guids: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mean over the rows of each recording, then over recordings, of the kept rows."""
    frames = []
    for guid in np.unique(guids[keep]):
        frames.append(np.nanmean(matrix[keep][guids[keep] == guid], axis=0))
    return np.nanmean(np.stack(frames, axis=0), axis=0) if frames else np.full(matrix.shape[1], np.nan)


def build_lag_profile_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    readouts: Sequence[str],
    cell: CellBinding,
    caveat: str,
    lag_bands: Optional[Mapping[str, Tuple[int, int]]] = None,
) -> Any:
    r"""The lag-aligned source attribution against the model's own lag readout, pooled and by class.

    One row per readout under the source-null baseline: the mean over recordings of the
    normalised $|q_\ell|$ beside the mean normalised model profile, pooled (left) and per class
    (right); the recording-mean agreement statistics (``lag_corr``, ``lag_js``) are in the
    per-row table. Normalised per row so a recording with a large readout does not decide the
    shape for every other. The legend is kept in the first populated row. A last row puts
    every readout on one axis: the divergence, the forecast gap and the lag readout on each band,
    so where the model is sensitive for its latent change, for its forecast and for each band's
    own lag readout can be read against each other and against the model's lag profile.

    Args:
        rows: The per-row table.
        vectors: The row-aligned arrays, read for ``lag_profile`` and ``model_profile``.
        lag_seconds: The compensated lag axis.
        readouts: The readouts to draw, one row each.
        cell: The cell binding.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).
        lag_bands: The configured lag bands, for the comparison row; ``None`` draws the main
            readouts alone there.

    Returns:
        The figure.
    """
    bands = dict(lag_bands or {})
    figure, axes = figures.new_figure(len(readouts) + 1, 2, height_per_row=2.4, width=12.0)
    if len(rows) and "lag_profile" in vectors:
        null = (rows["baseline"].astype(str) == BASELINE_SOURCE_NULL).to_numpy()
        readout_column = rows["readout"].astype(str).to_numpy()
        band_column = rows["band"].astype(str).to_numpy() if "band" in rows.columns else np.full(len(rows), "")
        guids = rows["guid"].astype(str).to_numpy()
        classes = rows[labels.CLASS_COLUMN].astype(object).to_numpy()
        attribution = lag_hist.normalise(np.abs(vectors["lag_profile"]))
        model_profile = lag_hist.normalise(vectors["model_profile"])
    else:
        null = np.zeros(0, dtype=bool)
        readout_column = band_column = guids = classes = np.zeros(0, dtype=object)
        attribution = model_profile = np.zeros((0, len(lag_seconds)))

    keyed = False
    for index, readout in enumerate(readouts):
        keep = null & (readout_column == readout) if null.size else null
        if not keep.any():
            _empty(axes[index, 0], READOUT_SHORT.get(readout, readout))
            _empty(axes[index, 1], f"{READOUT_SHORT.get(readout, readout)}, by class")
            continue
        ax = axes[index, 0]
        ax.plot(lag_seconds, _per_recording_mean(attribution, guids, keep), color=figures.COLOR_BLUE,
                linewidth=figures.LINE_REGULAR, label="|attribution|")
        ax.plot(lag_seconds, _per_recording_mean(model_profile, guids, keep), color=figures.COLOR_ORANGE,
                linewidth=figures.LINE_THIN, label="model lag readout")
        ax.set_title(READOUT_SHORT.get(readout, readout))
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share per lag")
        symlog_legend(ax, _per_recording_mean(attribution, guids, keep), _per_recording_mean(model_profile, guids, keep),
                      ncol=2, legend=not keyed)
        figures.style_axes(ax)

        ax = axes[index, 1]
        names = labels.ordered_groups(
            [c for c in pd.unique(classes[keep]) if c is not None and not pd.isna(c)], labels.CLASS_COLUMN
        )
        colours = _class_colours(names)
        drawn = 0
        for name in names:
            of_class = keep & np.asarray([c == name for c in classes], dtype=bool)
            if not of_class.any():
                continue
            colour = colours.get(name, figures.COLOR_GRAY)
            ax.plot(lag_seconds, _per_recording_mean(attribution, guids, of_class), color=colour,
                    linewidth=figures.LINE_REGULAR, label=str(name))
            ax.plot(lag_seconds, _per_recording_mean(model_profile, guids, of_class), color=colour,
                    linewidth=figures.LINE_THIN, linestyle="--")
            drawn += 1
        if drawn == 0:
            _empty(ax, f"{READOUT_SHORT.get(readout, readout)}, by class")
        else:
            ax.set_title(f"{READOUT_SHORT.get(readout, readout)}, by class")
            ax.plot([], [], color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="|attribution|")
            ax.plot([], [], color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle="--", label="model lag readout")
            ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
            ax.set_ylabel("share per lag")
            symlog_legend(ax, _per_recording_mean(attribution, guids, keep), ncol=min(max(drawn, 1), 3),
                          legend=not keyed)
            figures.style_axes(ax)
        keyed = True

    # The comparison row: every readout on one axis, then the band readouts against their bands.
    ax, ax_bands = axes[len(readouts), 0], axes[len(readouts), 1]
    drawn = 0
    series: List[Tuple[str, np.ndarray, str]] = []
    for readout in readouts:
        keep = null & (readout_column == readout) if null.size else null
        if keep.any():
            series.append((READOUT_SHORT.get(readout, readout), _per_recording_mean(attribution, guids, keep),
                           READOUT_COLOURS.get(readout, figures.COLOR_GRAY)))
    band_series: List[Tuple[str, np.ndarray, str, Tuple[int, int]]] = []
    for position, (name, span) in enumerate(bands.items()):
        keep = null & (readout_column == READOUT_LAG_BAND) & (band_column == str(name)) if null.size else null
        if keep.any():
            colour = BAND_COLOURS[position % len(BAND_COLOURS)]
            profile = _per_recording_mean(attribution, guids, keep)
            series.append((variant_short(READOUT_LAG_BAND, str(name)), profile, colour))
            band_series.append((str(name), profile, colour, (int(span[0]), int(span[1]))))
    for label, profile, colour in series:
        ax.plot(lag_seconds, profile, color=colour, linewidth=figures.LINE_REGULAR, label=label)
        drawn += 1
    if null.any():
        ax.plot(lag_seconds, _per_recording_mean(model_profile, guids, null), color=figures.COLOR_BLACK,
                linewidth=figures.LINE_THIN, linestyle="--", label="model lag readout")
    if drawn == 0:
        _empty(ax, "All readouts")
    else:
        ax.set_title("All readouts")
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share per lag")
        symlog_legend(ax, *[profile for _label, profile, _colour in series], ncol=2)
        figures.style_axes(ax)
    if not band_series:
        _empty(ax_bands, "Lag bands")
    else:
        for name, profile, colour, (low, high) in band_series:
            low, high = max(0, low), min(len(lag_seconds) - 1, high)
            if low <= high:
                ax_bands.axvspan(
                    lag_seconds[low] - 0.5 * SECONDS_PER_STEP, lag_seconds[high] + 0.5 * SECONDS_PER_STEP,
                    color=colour, alpha=0.12, linewidth=0,
                )
            ax_bands.plot(lag_seconds, profile, color=colour, linewidth=figures.LINE_REGULAR, label=name)
        ax_bands.set_title("Lag bands")
        ax_bands.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax_bands.set_ylabel("share per lag")
        symlog_legend(ax_bands, *[profile for _name, profile, _colour, _span in band_series], ncol=min(len(band_series), 4))
        figures.style_axes(ax_bands)
    figures.caveat_note(figure, caveat)
    return figure


def offset_channel_map(maps: np.ndarray, anchors: Sequence[int], n_lags: int) -> np.ndarray:
    r"""Re-index $(N, T, C)$ maps by offset from each row's anchor: $(N, L, C)$, ``NaN`` before the record.

    The per-channel form of :func:`lag_profile`: cell $(\ell, c)$ is the attribution of channel
    $c$ at stored step $t_a - \ell$, so a population mean over rows is a map of *which channel at
    which offset* the readout responded to.

    Args:
        maps: $(N, T, C)$ per-step, per-channel attribution.
        anchors: Per-row anchor steps.
        n_lags: $L$.

    Returns:
        $(N, L, C)$.
    """
    field = np.asarray(maps, dtype=np.float64)
    out = np.full((field.shape[0], int(n_lags), field.shape[2]), np.nan)
    for row, anchor in enumerate(anchors):
        high = min(int(anchor) + 1, int(n_lags))
        if high <= 0:
            continue
        # Offsets 0 .. high-1 read stored steps anchor .. anchor-high+1, in that order.
        out[row, :high, :] = field[row, int(anchor) - np.arange(high), :]
    return out


def _band_colour_of_channels(width: int, groups: Mapping[str, np.ndarray]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Colour every channel of a stream by its frequency band; grey where the map names none."""
    colours = [figures.COLOR_GRAY] * int(width)
    legend: List[Tuple[str, str]] = []
    for position, (band, channels) in enumerate(groups.items()):
        colour = BAND_COLOURS[position % len(BAND_COLOURS)]
        legend.append((band_partition.band_display_label(str(band)), colour))
        for channel in np.asarray(channels, dtype=np.int64):
            if 0 <= int(channel) < width:
                colours[int(channel)] = colour
    return colours, legend


def _stream_selection(rows: pd.DataFrame, readout: str, stream: str) -> np.ndarray:
    """The rows of one readout under the baseline a stream's attribution is read on.

    The target inputs move only along the all-zero path, so a target profile is read there; the
    source is read under the source-null baseline, the primary comparison.
    """
    if not len(rows):
        return np.zeros(0, dtype=bool)
    baseline = BASELINE_ALL_ZERO if stream == STREAM_TARGET else BASELINE_SOURCE_NULL
    return (
        (rows["readout"].astype(str) == readout).to_numpy()
        & (rows["baseline"].astype(str) == baseline).to_numpy()
    )


def build_channel_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    *,
    readouts: Sequence[str],
    channel_groups: Mapping[str, Mapping[str, np.ndarray]],
    n_scattering: Optional[int],
    caveat: str,
) -> Any:
    r"""Which declared input channels each readout responded to, summed over stored time.

    One row per readout, the target stream (all-zero baseline) left and the source stream
    (source-null baseline) right. Bars are the mean over recordings of the **signed** channel
    profile $\sum_t a_{t,c}$; the line is the mean of the unsigned one $\sum_t |a_{t,c}|$, which
    says how much the channel mattered when its contributions cancel over time. Bars are coloured
    by the channel's frequency band where the run wrote a channel map; on the target stream the
    scattering block sits left of the rule and the phase-harmonic block right of it.

    Args:
        rows: The per-row table.
        vectors: The row-aligned arrays, read for the signed and unsigned channel profiles.
        readouts: The readouts, one row each.
        channel_groups: ``{stream: {band: positions}}`` from the channel map, possibly empty.
        n_scattering: Width of the target scattering block, or ``None``.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure. The legend and the block label are kept in the first populated row of each
        stream's column.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.4, width=12.0)
    guids = rows["guid"].astype(str).to_numpy() if len(rows) else np.zeros(0, dtype=object)
    keyed = {stream: False for stream in STREAMS}
    for index, readout in enumerate(readouts):
        for col, stream in enumerate(STREAMS):
            ax = axes[index, col]
            title = f"{READOUT_SHORT.get(readout, readout)}: {stream}"
            keep = _stream_selection(rows, readout, stream)
            signed_name, unsigned_name = f"channel_profile_{stream}", f"channel_abs_profile_{stream}"
            if not keep.any() or signed_name not in vectors:
                _empty(ax, title)
                continue
            signed = _per_recording_mean(vectors[signed_name].astype(np.float64), guids, keep)
            width = int(signed.size)
            colours, legend = _band_colour_of_channels(width, channel_groups.get(stream, {}))
            x = np.arange(width)
            ax.bar(x, signed, width=0.85, color=colours, linewidth=0, label="Signed")
            if unsigned_name in vectors:
                unsigned = _per_recording_mean(vectors[unsigned_name].astype(np.float64), guids, keep)
                ax.plot(x, unsigned, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, label="Unsigned")
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            if stream == STREAM_TARGET and n_scattering and 0 < int(n_scattering) < width:
                ax.axvline(float(n_scattering) - 0.5, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE,
                           linestyle="--")
                if not keyed[stream]:
                    ax.text(float(n_scattering) - 0.5, 0.98, " phase-harmonic block", transform=ax.get_xaxis_transform(),
                            ha="left", va="top", fontsize=figures.FONT_TINY, color=figures.COLOR_GRAY)
            if not keyed[stream]:
                for label, colour in legend:
                    ax.plot([], [], marker="s", linestyle="none", color=colour, label=label)
            ax.set_title(title)
            ax.set_xlabel("declared channel")
            ax.set_ylabel("attribution (symlog)")
            ax.set_xlim(-0.5, width - 0.5)
            symlog_legend(ax, signed, unsigned if unsigned_name in vectors else signed, ncol=3,
                          legend=not keyed[stream])
            keyed[stream] = True
            figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def build_lag_channel_figure(
    lag_channel: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    *,
    lag_seconds: np.ndarray,
    readouts: Sequence[str],
    n_scattering: Optional[int],
    caveat: str,
) -> Any:
    r"""Which channel at which offset from the anchor: the population mean of the unsigned maps.

    One row per readout: the target stream re-indexed by offset from the anchor under the
    all-zero baseline (left) and the source stream by lag under the source-null baseline (right),
    each the mean over attributed anchors of $|a|$ at that offset and channel. The two marginals
    the other figures draw -- the lag profile and the channel profile -- are the row and column
    sums of these maps, and a peak here says both at once: which coefficient, how far back.

    Args:
        lag_channel: ``{(readout, baseline, stream): {'mean_abs': (L, C), 'mean': (L, C),
            'n_rows': int}}`` from the pass.
        lag_seconds: The compensated lag axis.
        readouts: The readouts, one row each.
        n_scattering: Width of the target scattering block, or ``None``.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.8, width=12.0)
    half = 0.5 * float(SECONDS_PER_STEP)
    for index, readout in enumerate(readouts):
        for col, stream in enumerate(STREAMS):
            ax = axes[index, col]
            baseline = BASELINE_ALL_ZERO if stream == STREAM_TARGET else BASELINE_SOURCE_NULL
            title = f"{READOUT_SHORT.get(readout, readout)}: {stream}"
            entry = lag_channel.get((readout, baseline, stream))
            if entry is None or not np.isfinite(np.asarray(entry["mean_abs"])).any():
                _empty(ax, title)
                continue
            field = np.asarray(entry["mean_abs"], dtype=np.float64).T   # (C, L)
            figures.heatmap_with_colorbar(
                figure, ax, field, symmetric=False, interpolation="none", norm=unsigned_log_norm(field),
                title=title, xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel="declared channel",
                colorbar_label="mean |attribution| (log)",
                extent=(float(lag_seconds[0]) - half, float(lag_seconds[-1]) + half, field.shape[0] - 0.5, -0.5),
                separator_row=(int(n_scattering) - 1) if (stream == STREAM_TARGET and n_scattering) else None,
            )
    figures.caveat_note(figure, caveat)
    return figure


def build_time_profile_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    readouts: Sequence[str],
    caveat: str,
) -> Any:
    r"""The signed attribution by offset from the anchor, for both streams.

    The lag-profile figure draws magnitudes, which say where the model was sensitive and not in
    which direction. Here, per readout, the source stream (source-null baseline, left) and the
    target stream (all-zero baseline, right) are drawn by offset from the anchor with the mean
    **positive** part above the axis and the mean **negative** part below it, the net mean as a
    line and the unsigned mean dashed. An offset whose positive and negative parts are both large
    with a small net is one where channels pull the readout both ways.

    Args:
        rows: The per-row table.
        vectors: The row-aligned arrays, read for ``lag_profile`` and ``target_lag_profile``.
        lag_seconds: The compensated lag axis.
        readouts: The readouts, one row each.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure. Every panel keys the same four curves, so the legend is kept in the first
        populated panel.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.4, width=12.0)
    guids = rows["guid"].astype(str).to_numpy() if len(rows) else np.zeros(0, dtype=object)
    keyed = False
    for index, readout in enumerate(readouts):
        for col, (stream, name) in enumerate(((STREAM_SOURCE, "lag_profile"), (STREAM_TARGET, "target_lag_profile"))):
            ax = axes[index, col]
            title = f"{READOUT_SHORT.get(readout, readout)}: {stream}"
            keep = _stream_selection(rows, readout, stream)
            if not keep.any() or name not in vectors:
                _empty(ax, title)
                continue
            profile = vectors[name].astype(np.float64)
            positive = _per_recording_mean(np.where(profile > 0.0, profile, 0.0), guids, keep)
            negative = _per_recording_mean(np.where(profile < 0.0, profile, 0.0), guids, keep)
            net = _per_recording_mean(profile, guids, keep)
            unsigned = _per_recording_mean(np.abs(profile), guids, keep)
            ax.fill_between(lag_seconds, 0.0, positive, color=figures.COLOR_VERMILLION, alpha=0.35, linewidth=0,
                            label="Raises readout")
            ax.fill_between(lag_seconds, negative, 0.0, color=figures.COLOR_BLUE, alpha=0.35, linewidth=0,
                            label="Lowers readout")
            ax.plot(lag_seconds, net, color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="Net")
            ax.plot(lag_seconds, unsigned, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN, linestyle="--",
                    label="Unsigned")
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title(title)
            ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
            ax.set_ylabel("attribution (symlog)")
            symlog_legend(ax, positive, negative, unsigned, ncol=2, legend=not keyed)
            keyed = True
            figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def build_checks_figure(rows: pd.DataFrame, *, tolerance: float, caveat: str) -> Any:
    r"""The numerical checks behind every row, so a map is read after its residual, not before.

    One horizontal strip per attributed variant -- readout, band and baseline -- so every row of
    the table is a point and nothing needs a legend. Left: the relative completeness residual on a
    logarithmic axis against the tolerance the pass counts against. Middle: the entry jump as a
    share of the readout's whole move along the path, $|f(x_0) - f(b)| / |f(x) - f(b)|$ -- the
    part of the move the excluded start of the path accounts for, unitless so every readout sits
    on one axis. Right: the two structural checks per readout, the largest attribution to a step
    after the anchor and to a gated-off source step, exactly zero on a causal model and printed
    beside their bars so a zero reads as a zero. The baseline of every strip is named on its tick
    label, so the points carry one colour per baseline and no key. A numerical check makes no
    claim about the physiology and carries no note under it.

    Args:
        rows: The per-row table.
        tolerance: The completeness tolerance the pass counts rows against.
        caveat: Kept for the signature shared by the builders; this figure prints no note.

    Returns:
        The figure.
    """
    needed = {"readout", "baseline", "completeness_rel", "entry_jump", "value_input", "value_baseline"}
    if not len(rows) or not needed <= set(rows.columns):
        figure, axes = figures.new_figure(1, 3, height_per_row=3.0, width=15.0)
        for col, title in enumerate(("Completeness residual", "Entry jump share", "Structural checks")):
            _empty(axes[0, col], title)
        return figure
    band = rows["band"].fillna("").astype(str).to_numpy() if "band" in rows.columns else np.full(len(rows), "")
    keys = list(zip(rows["readout"].astype(str), band, rows["baseline"].astype(str)))
    groups = list(dict.fromkeys(keys))
    group_of = np.asarray([groups.index(key) for key in keys])
    labels_of = [f"{variant_short(readout, tag)}, {baseline}" for readout, tag, baseline in groups]
    figure, axes = figures.new_figure(1, 3, height_per_row=max(3.0, 0.24 * len(groups) + 1.2), width=15.0)
    residual = np.asarray(rows["completeness_rel"], dtype=np.float64)
    move = np.abs(np.asarray(rows["value_input"], dtype=np.float64) - np.asarray(rows["value_baseline"], dtype=np.float64))
    share = np.divide(np.abs(np.asarray(rows["entry_jump"], dtype=np.float64)), move,
                      out=np.full(move.shape, np.nan), where=move > 0.0)
    for ax, values, title, xlabel in (
        (axes[0, 0], residual, "Completeness residual", "Relative residual"),
        (axes[0, 1], share, "Entry jump share", "Entry jump / path move"),
    ):
        finite = np.isfinite(values) & (values > 0.0)
        floor = float(values[finite].min()) if finite.any() else 1e-12
        # A row whose value is exactly zero is drawn at the axis floor rather than dropped.
        shown = np.where(finite, values, floor)
        for index in range(len(groups)):
            keep = (group_of == index) & np.isfinite(values)
            colour = figures.COLOR_BLUE if groups[index][2] == BASELINE_SOURCE_NULL else figures.COLOR_VERMILLION
            ax.scatter(shown[keep], np.full(int(keep.sum()), index), s=10, color=colour, alpha=0.7, linewidths=0)
            if keep.any():
                ax.plot([float(np.nanmedian(shown[keep]))] * 2, [index - 0.35, index + 0.35], color=figures.COLOR_BLACK,
                        linewidth=figures.LINE_REGULAR)
        ax.set_xscale("log")
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels(labels_of if ax is axes[0, 0] else [], fontsize=figures.FONT_TINY)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.set_title(title, fontsize=figures.FONT_SMALL)
        ax.set_xlabel(xlabel)
        figures.style_axes(ax)
    axes[0, 0].axvline(float(tolerance), color=figures.COLOR_BLACK, linestyle=":", linewidth=figures.LINE_REGULAR,
                       label="Tolerance")
    axes[0, 0].plot([], [], color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="Median")
    axes[0, 0].legend(fontsize=figures.FONT_TINY)

    ax = axes[0, 2]
    checks = [("after_anchor_max_abs", "After the anchor", figures.COLOR_BLUE),
              ("gated_off_max_abs", "Gated-off source", figures.COLOR_ORANGE)]
    if all(name in rows.columns for name, _label, _colour in checks):
        readouts = list(dict.fromkeys(rows["readout"].astype(str)))
        y = np.arange(len(readouts))
        largest = 0.0
        for offset, (name, label, colour) in enumerate(checks):
            values = [float(np.nanmax(rows.loc[rows["readout"].astype(str) == readout, name])) for readout in readouts]
            largest = max([largest, *values])
            positions = y + (offset - 0.5) * 0.38
            ax.barh(positions, values, height=0.38, color=colour, label=label)
            for position, value in zip(positions, values):
                ax.annotate(f"{value:.2g}", (value, position), textcoords="offset points", xytext=(3, 0),
                            va="center", fontsize=figures.FONT_TINY)
        # Linear from zero: the expected value is exactly zero, printed beside every bar.
        ax.set_xlim(0.0, 1.3 * largest if largest > 0.0 else 1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([READOUT_SHORT.get(readout, readout) for readout in readouts], fontsize=figures.FONT_TINY)
        ax.set_ylim(len(readouts) - 0.5, -0.5)
        ax.set_title("Structural checks", fontsize=figures.FONT_SMALL)
        ax.set_xlabel("|attribution|, readout units")
        ax.legend(fontsize=figures.FONT_TINY, loc="lower right")
        figures.style_axes(ax)
    else:
        _empty(ax, "Structural checks")
    return figure


def _lag_centroid(profile: np.ndarray, lag_seconds: np.ndarray) -> np.ndarray:
    r"""The $|q|$-weighted mean lag of each row, in seconds; ``NaN`` where a row carries no mass."""
    weights = np.abs(np.asarray(profile, dtype=np.float64))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    total = weights.sum(axis=1)
    return np.divide((weights * lag_seconds[None, :]).sum(axis=1), total, out=np.full(total.shape, np.nan), where=total > 0.0)


def build_delivery_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    readouts: Sequence[str],
    caveat: str,
) -> Any:
    r"""The attributed anchors on the clinical clock: source attribution against hours before delivery.

    Per readout under the source-null baseline: the source attribution total (left) and the
    $|q|$-weighted lag centroid of the source attribution (right) of every attributed anchor
    against its hours before delivery, one point per anchor in its class colour, with the class
    median per one-hour window drawn where at least three recordings fall in it. The anchors are a
    few per segment of one segment per recording, so this is a sparse view; the traces and the
    clock analyses are where the clinical clock is read densely.

    Args:
        rows: The per-row table.
        vectors: The row-aligned arrays, read for ``lag_profile``.
        lag_seconds: The compensated lag axis.
        readouts: The readouts, one row each.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure. Every panel keys the same classes, so the legend is kept in the first
        populated panel.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.4, width=12.0)
    window = 1.0
    keyed = False
    if len(rows) and "lag_profile" in vectors:
        hours = -traces.absolute_seconds(rows["epoch"], rows["anchor"]) / cohort.SECONDS_PER_HOUR
        centroid = _lag_centroid(vectors["lag_profile"], lag_seconds)
        classes = rows[labels.CLASS_COLUMN].astype(object).to_numpy()
        guids = rows["guid"].astype(str).to_numpy()
    else:
        hours = centroid = np.zeros(0)
        classes = guids = np.zeros(0, dtype=object)
    for index, readout in enumerate(readouts):
        keep = _stream_selection(rows, readout, STREAM_SOURCE)
        for col, (values, ylabel, what) in enumerate((
            (np.asarray(rows["source_total"], dtype=np.float64) if len(rows) else np.zeros(0),
             READOUT_UNITS.get(readout, "readout units"), "source total"),
            (centroid, "s (stored-coefficient time)", "lag centroid"),
        )):
            ax = axes[index, col]
            title = f"{READOUT_SHORT.get(readout, readout)}: {what}"
            usable = keep & np.isfinite(values) & np.isfinite(hours) if keep.size else keep
            if not usable.any():
                _empty(ax, title)
                continue
            names = labels.ordered_groups([c for c in pd.unique(classes[usable]) if c is not None and not pd.isna(c)], labels.CLASS_COLUMN)
            colours = _class_colours(names)
            for name in names:
                of_class = usable & np.asarray([c == name for c in classes], dtype=bool)
                colour = colours.get(name, figures.COLOR_GRAY)
                ax.scatter(hours[of_class], values[of_class], s=8, color=colour, alpha=0.6, linewidths=0,
                           label=str(name))
                bins = np.floor(hours[of_class] / window).astype(np.int64)
                table = pd.DataFrame({"bin": bins, "guid": guids[of_class], "value": values[of_class]})
                per_recording = table.groupby(["bin", "guid"])["value"].mean().reset_index()
                counts = per_recording.groupby("bin")["guid"].nunique()
                medians = per_recording.groupby("bin")["value"].median()
                enough = counts[counts >= 3].index
                if len(enough):
                    ax.plot((np.asarray(enough, dtype=np.float64) + 0.5) * window, medians.loc[enough].to_numpy(),
                            color=colour, linewidth=figures.LINE_EMPHASIS, marker="o", markersize=2.5)
            ax.set_title(title)
            ax.set_xlabel("hours before delivery")
            ax.set_ylabel(ylabel)
            ax.invert_xaxis()
            if not keyed:
                figures.legend_with_headroom(ax, ncol=3, headroom=0.35)
                keyed = True
            figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def build_band_figure(
    bands: pd.DataFrame,
    lag_bands: pd.DataFrame,
    *,
    readouts: Sequence[str],
    caveat: str,
) -> Any:
    """Frequency-band and lag-band attributions, beside the readouts they are read against.

    Top: per readout, the mean over recordings of the attribution summed over each frequency
    band, one bar group per stream -- the target read along the all-zero path, the source along
    the source-null one, the only paths each moves on -- with the spectral-skill gap per target
    band drawn on a twin axis where the run carries it. Bottom: per lag band of the source, what
    the band contributed to the readout, three ways on one sign convention (positive: the band
    raised the readout): the integrated-gradient sum over the band, and minus the change the
    readout makes when the band is removed, by this analysis's feature ablation and by the
    occlusion analysis where the run carries it (which scores its own anchors).

    Args:
        bands: The frequency-band table.
        lag_bands: The lag-band table.
        readouts: The readouts, one column each.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure. The legend of each row is kept in its first populated panel.
    """
    n_cols = max(len(readouts), 1)
    keyed = [False, False]
    figure, axes = figures.new_figure(2, n_cols, height_per_row=3.0, width=5.0 * n_cols)
    for col, readout in enumerate(readouts):
        ax = axes[0, col]
        subset = bands[bands["readout"].astype(str) == readout] if len(bands) else bands
        if subset.empty:
            _empty(ax, f"{READOUT_SHORT.get(readout, readout)}: frequency band")
        else:
            names = list(dict.fromkeys(subset["band"].astype(str)))
            x = np.arange(len(names))
            width = 0.38
            for offset, (stream, colour) in enumerate(((STREAM_TARGET, figures.COLOR_BLUE), (STREAM_SOURCE, figures.COLOR_ORANGE))):
                part = subset[subset["stream"].astype(str) == stream].set_index("band")
                heights = [float(part["attribution_mean"].get(name, np.nan)) for name in names]
                ax.bar(x + (offset - 0.5) * width, heights, width=width, color=colour, label=stream.capitalize())
            if "spectral_skill_pred_gap_nats" in subset.columns and subset["spectral_skill_pred_gap_nats"].notna().any():
                twin = ax.twinx()
                part = subset[subset["stream"].astype(str) == STREAM_TARGET].set_index("band")
                twin.plot(x, [float(part["spectral_skill_pred_gap_nats"].get(name, np.nan)) for name in names],
                          color=figures.COLOR_GREEN, marker="o", markersize=2.5, linewidth=figures.LINE_THIN,
                          label="Spectral skill")
                twin.set_ylabel("nats per anchor", fontsize=figures.FONT_LABEL)
                twin.tick_params(labelsize=figures.FONT_TINY)
                if not keyed[0]:
                    twin.legend(fontsize=figures.FONT_TINY, loc="lower right")
            ax.set_xticks(x)
            # Ticks name the frequency range (period in parentheses), not the clinical band key
            # the table is keyed by.
            ax.set_xticklabels(
                [band_partition.band_display_label(name) for name in names],
                rotation=30, ha="right", fontsize=figures.FONT_TINY,
            )
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title(f"{READOUT_SHORT.get(readout, readout)}: frequency band", fontsize=figures.FONT_SMALL)
            ax.set_ylabel(f"{READOUT_UNITS.get(readout, 'readout units')} (symlog)")
            symlog_axis(ax, subset["attribution_mean"].to_numpy(dtype=np.float64), headroom=1.0)
            if not keyed[0]:
                ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
                keyed[0] = True
            figures.style_axes(ax)

        ax = axes[1, col]
        subset = lag_bands[lag_bands["readout"].astype(str) == readout] if len(lag_bands) else lag_bands
        if subset.empty:
            _empty(ax, f"{READOUT_SHORT.get(readout, readout)}: lag band")
        else:
            names = list(dict.fromkeys(subset["band"].astype(str)))
            x = np.arange(len(names))
            # ``(column, sign, label, colour)``: the two removal deltas are negated onto the
            # integrated gradient's convention, so three readings that agree point the same way.
            series = [
                ("ig_attribution_mean", 1.0, "IG sum", figures.COLOR_BLUE),
                ("ablation_delta_mean", -1.0, "$-$ ablation", figures.COLOR_PURPLE),
                ("occlusion_delta_total_nats", -1.0, "$-$ occlusion", figures.COLOR_GREEN),
            ]
            present = [entry for entry in series if entry[0] in subset.columns and subset[entry[0]].notna().any()]
            width = 0.8 / max(len(present), 1)
            indexed = subset.set_index("band")
            drawn: List[np.ndarray] = []
            for offset, (column, sign, label, colour) in enumerate(present):
                values = sign * np.asarray([float(indexed[column].get(name, np.nan)) for name in names])
                drawn.append(values)
                ax.bar(x + (offset - (len(present) - 1) / 2) * width, values, width=width, color=colour, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=figures.FONT_TINY)
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title(f"{READOUT_SHORT.get(readout, readout)}: lag band", fontsize=figures.FONT_SMALL)
            ax.set_ylabel(f"{READOUT_UNITS.get(readout, 'readout units')} (symlog)")
            symlog_axis(ax, *drawn, headroom=1.0)
            if not keyed[1]:
                ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
                keyed[1] = True
            figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def build_layer_figure(
    layer: pd.DataFrame,
    top_coordinate: pd.DataFrame,
    *,
    cell: CellBinding,
    lag_seconds: np.ndarray,
    caveat: str,
) -> Any:
    """The layer split and how the top divergence coordinate is fed.

    Left: per readout, the mean over recordings of the layer attribution per unit -- per head in
    the attentive cells, per lag in the residual cell -- pooled; the per-class means are in the
    layer table. Right: the mean over recordings of the source attribution of the anchor's
    largest per-coordinate divergence by lag. That readout is attributed along the source-null
    path only, where the target does not move, so it has no target profile to draw.

    Args:
        layer: The layer table, long-form: ``readout, unit, clinical_class, n_recordings, mean``.
        top_coordinate: Rows of the per-row table for the top-coordinate readout, with their
            lag-aligned source profiles attached as the column ``lag_profile`` (arrays).
        cell: The cell binding.
        lag_seconds: The compensated lag axis.
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 2, height_per_row=3.2, width=12.0)
    ax = axes[0, 0]
    if layer.empty:
        _empty(ax, f"Layer split per {cell.layer_axis}")
    else:
        readouts = list(dict.fromkeys(layer["readout"].astype(str)))
        pooled = layer[layer[labels.CLASS_COLUMN].astype(str) == "pooled"]
        units = np.sort(pooled["unit"].unique()) if not pooled.empty else np.sort(layer["unit"].unique())
        x = np.asarray(units, dtype=np.float64) if cell.layer_axis == "head" else lag_seconds[np.asarray(units, dtype=np.int64)]
        for index, readout in enumerate(readouts):
            part = pooled[pooled["readout"].astype(str) == readout].set_index("unit")
            values = [float(part["mean"].get(unit, np.nan)) for unit in units]
            if cell.layer_axis == "head":
                ax.bar(x + (index - (len(readouts) - 1) / 2) * 0.8 / len(readouts), values,
                       width=0.8 / len(readouts), label=READOUT_SHORT.get(readout, readout),
                       color=READOUT_COLOURS.get(readout, figures.COLOR_GRAY))
            else:
                ax.plot(x, values, linewidth=figures.LINE_REGULAR, label=READOUT_SHORT.get(readout, readout),
                        color=READOUT_COLOURS.get(readout, figures.COLOR_GRAY))
        ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
        if cell.layer_axis == "head":
            # Heads are integers; a fractional tick between two heads names nothing.
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(unit)) for unit in units])
        ax.set_title(f"Layer split per {cell.layer_axis}", fontsize=figures.FONT_SMALL)
        ax.set_xlabel("head" if cell.layer_axis == "head" else COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("readout units (symlog)")
        symlog_axis(ax, pooled["mean"].to_numpy(dtype=np.float64) if not pooled.empty else layer["mean"].to_numpy(dtype=np.float64), headroom=1.0)
        ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
        figures.style_axes(ax)
    ax = axes[0, 1]
    if top_coordinate.empty or "lag_profile" not in top_coordinate.columns:
        _empty(ax, "Top $K_{t,d}$ by lag")
    else:
        source = np.abs(np.stack([np.asarray(v, dtype=np.float64) for v in top_coordinate["lag_profile"]], axis=0))
        guids = top_coordinate["guid"].astype(str).to_numpy()
        profile = _per_recording_mean(source, guids, np.ones(len(guids), dtype=bool))
        ax.plot(lag_seconds, profile, color=figures.COLOR_ORANGE, linewidth=figures.LINE_REGULAR)
        ax.set_title("Top $K_{t,d}$ by lag", fontsize=figures.FONT_SMALL)
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("|attribution| (nats, symlog)")
        symlog_axis(ax, profile, headroom=1.0)
        figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


def build_null_figure(null: pd.DataFrame, *, caveat: str) -> Any:
    r"""The null decomposition of $K_t$: clock, content, and the entry jump, by class.

    Left: for the divergence under the source-null baseline, per class, the mean over recordings
    of the readout at the input, at the exact null (the clock part), the attributed content
    (the integrated-gradient sum) and the entry jump. Right: under the all-zero baseline, the
    split of the attribution between the target and the source streams.

    Args:
        null: The null table, one row per (class, readout).
        caveat: The one-line note printed under the figure (:data:`ATTRIBUTION_NOTE`).

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 2, height_per_row=3.2, width=12.0)
    for col, (baseline, columns, title) in enumerate((
        (BASELINE_SOURCE_NULL,
         [("value_input_mean", "$K_t$ at input"), ("value_baseline_mean", "$K_t$ at null"),
          ("attributed_mean", "IG sum"), ("entry_jump_mean", "Entry jump")],
         "Source-null: clock and content"),
        (BASELINE_ALL_ZERO,
         [("target_total_mean", "Target"), ("source_total_mean", "Source"),
          ("entry_jump_mean", "Entry jump")],
         "All-zero: target and source"),
    )):
        ax = axes[0, col]
        subset = null[(null["baseline"].astype(str) == baseline) & (null["readout"].astype(str) == READOUT_KLD)] if len(null) else null
        if subset.empty:
            _empty(ax, title)
            continue
        classes = list(dict.fromkeys(subset[labels.CLASS_COLUMN].astype(str)))
        x = np.arange(len(classes))
        width = 0.8 / len(columns)
        indexed = subset.set_index(labels.CLASS_COLUMN)
        for offset, (column, label) in enumerate(columns):
            ax.bar(x + (offset - (len(columns) - 1) / 2) * width,
                   [float(indexed[column].get(name, np.nan)) for name in classes], width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{name} (n={int(indexed['n_recordings'].get(name, 0))})" for name in classes],
                           fontsize=figures.FONT_TINY)
        ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
        ax.set_title(title, fontsize=figures.FONT_SMALL)
        ax.set_ylabel("nats per anchor")
        ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
        figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


#: The rows of a recording's attribution trace figure: the lag-aligned attribution of the
#: divergence and the model's own lag readout as heatmaps on one lag axis, the agreement and the
#: source share as lines, and the readout values behind them.
TRACE_PANELS: Tuple[Any, ...] = (
    traces.HeatmapPanel("kld_attribution_lag_map", "$K_t$ attribution by lag", "", log=True, lag_axis=True),
    traces.HeatmapPanel("pred_gap_attribution_lag_map", "Forecast-gap attribution by lag", "", log=True,
                        lag_axis=True),
    traces.HeatmapPanel("model_lag_map", "Model lag readout", "", log=True, lag_axis=True),
    traces.LinePanel(("kld_lag_corr", "pred_gap_lag_corr"), "Agreement with lag readout", "Pearson $r$",
                     labels=("$K_t$", "Forecast gap")),
    traces.LinePanel(("kld_source_total", "pred_gap_source_total"), "Source attribution",
                     "nats", labels=("$K_t$", "Forecast gap")),
    traces.LinePanel(("kld_value_input", "kld_value_baseline"), "Divergence $K_t$",
                     "nats", labels=("input", "null")),
    traces.LinePanel(("pred_gap_value_input", "pred_gap_value_baseline"),
                     "Forecast gap", "nats", labels=("input", "null")),
)

#: The lag families a trace's shape statistics are taken of.
TRACE_LAG_PROFILES: Dict[str, str] = {
    "kld_attribution_lag_map": "kld_attr_lag",
    "pred_gap_attribution_lag_map": "gap_attr_lag",
    "model_lag_map": "model_lag",
}


# =============================================================================
# Cost
# =============================================================================
def cost_record(*, elapsed_s: float, n_segments: int, n_rows: int, n_forward_equivalents: int, device: Any) -> Dict[str, Any]:
    """What the pass cost, in the collection pass's key vocabulary plus this analysis's own rate.

    Args:
        elapsed_s: Wall-clock seconds of the attribution loop.
        n_segments: Segments attributed.
        n_rows: Attribution rows (one per anchor, readout and baseline).
        n_forward_equivalents: Forward-and-backward passes over one row the loop performed.
        device: The device, or ``None``.

    Returns:
        The record. ``peak_allocated_bytes`` is a CUDA figure and ``None`` elsewhere.
    """
    elapsed = max(float(elapsed_s), 1e-9)
    resolved = None if device is None else torch.device(device)
    peak: Optional[int] = None
    if resolved is not None and resolved.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(resolved))
    rate = float(n_segments) / elapsed
    return {
        "device": None if resolved is None else str(resolved),
        "n_samples": int(n_segments),
        "n_rows": int(n_rows),
        "n_forward_equivalents": int(n_forward_equivalents),
        "elapsed_s": float(elapsed),
        "samples_per_second": rate,
        "hours_per_1000_samples": (1000.0 / rate / 3600.0) if rate > 0.0 else None,
        "seconds_per_row": (elapsed / n_rows) if n_rows else None,
        "peak_allocated_bytes": peak,
        "note": (
            "measured on this pass at the anchor count, readout set, baseline set and step count "
            "recorded in the plan. A longer pass extrapolates as hours = (n_samples / 1000) * "
            "hours_per_1000_samples at those settings; halving IG_STEPS roughly halves "
            "seconds_per_row. peak_allocated_bytes is the CUDA allocator's process peak and is "
            "null on any other device."
        ),
    }


__all__ = [
    "ANALYSIS_DIRNAME", "ANCHORS_PER_SEGMENT", "ATTENTION_CELL", "ATTRIBUTION_CAVEAT", "ATTRIBUTION_NOTE",
    "AnchorReadout", "AttributionBatch", "BANDS_FILENAME", "BAND_FIGURE", "BASELINES",
    "BASELINE_ALL_ZERO", "BASELINE_ENTRY_FRACTION", "BASELINE_SOURCE_NULL", "CAP_NAME",
    "CellBinding", "DEFAULT_SEGMENTS", "DEFAULT_SEGMENTS_PER_RECORDING", "DRAW_SEED_OFFSET",
    "IG_INTERNAL_BATCH_SIZE", "IG_STEPS", "SEGMENTS_PER_RECORDING_CAP_NAME",
    "LAG_BANDS_FILENAME", "LAG_PROFILE_FIGURE", "LAYER_FIGURE", "LAYER_FILENAME", "MAIN_READOUTS",
    "MAPS_FILENAME", "MAP_FIGURE", "METHOD_RECORD", "NULL_FIGURE", "NULL_FILENAME", "READOUTS",
    "READOUT_KLD", "READOUT_KLD_DIM", "READOUT_LAG_BAND", "READOUT_MU_POST_DIM",
    "READOUT_MU_PRIOR_DIM", "READOUT_NLL_BASE", "READOUT_NLL_FULL", "READOUT_PRED_GAP",
    "RECORDINGS_FILENAME", "ROWS_FILENAME", "SLOT_CELL", "STREAMS", "STREAM_SOURCE",
    "STREAM_TARGET", "SUMMARY_FILENAME", "TARGET_ONLY_READOUTS", "TRACE_DIRNAME",
    "CHANNEL_FIGURE", "CHECKS_FIGURE", "DELIVERY_FIGURE", "EXAMPLE_DIRNAME", "EXAMPLE_READOUTS",
    "EXAMPLE_SUFFIX", "LAG_CHANNEL_FIGURE", "LAG_CHANNEL_FILENAME", "READOUT_TITLES", "TIME_PROFILE_FIGURE",
    "TRACE_LAG_PROFILES", "TRACE_READOUTS", "TRACE_PANELS", "TRACE_RECORDINGS_PER_CLASS", "TRACE_SUFFIX",
    "VECTORS_FILENAME", "ablate_lag_bands", "agreement", "band_sums", "baselines_for",
    "build_band_figure", "build_channel_figure", "build_checks_figure", "build_delivery_figure",
    "build_example_figure", "build_lag_channel_figure", "build_lag_profile_figure", "build_layer_figure",
    "build_map_figure", "build_time_profile_figure", "offset_channel_map",
    "build_null_figure", "channel_groups_from_map", "channel_profile", "contributing_columns",
    "cost_record", "entry_point", "expand_rows", "integrated_gradients", "lag_band_feature_mask",
    "lag_band_groups", "lag_profile", "layer_attribution", "model_lag_readout", "spread_columns",
    "time_profile", "warm_from_step",
    "BLOCKS", "BLOCKS_FILENAME", "BLOCK_FIGURE", "BLOCK_SCORE_READOUTS", "HORIZON_FIGURE",
    "HORIZON_READOUT_STEPS", "LOG_DECADES", "READOUT_MSE_FULL", "READOUT_MSE_GAP", "READOUT_NLL_HORIZON",
    "UNREAD_COLOUR", "block_sums", "build_block_figure", "build_horizon_figure", "example_variants",
    "horizon_steps", "masked_field", "signed_log_norm", "source_block_split", "symlog_axis",
    "unsigned_log_norm", "variant_label", "variant_short", "symlog_legend", "READOUT_SHORT",
    "READOUT_UNITS", "REPARAMETERISATION_SEAMS", "SHARED_SCALE_NOTE", "attributed_forward", "example_norms",
    "input_norm", "readout_unit",
]
