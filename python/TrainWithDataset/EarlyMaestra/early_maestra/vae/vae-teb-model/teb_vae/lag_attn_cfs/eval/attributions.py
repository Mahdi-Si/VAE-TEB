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
input, which no figure here could separate.

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
below $10^{-3}$ at :data:`IG_STEPS` steps on every fixture; the readout at the exact baseline, at
the entry point and at the input all travel on every row, so the **entry jump**
$f(x_0) - f(b)$ is a reported scalar rather than a hidden one. It belongs to no input step: it is
the normalisation snapping out of its degenerate state.

**Structural properties every attribution here is checked against**, and that the tests assert on
the tiny models: attribution to any stored step later than the anchor is exactly zero; attribution
to a source step a channel has not warmed up at is exactly zero; the source attribution of a
target-only readout ($\mu^p$, the base block score) is exactly zero; and the integrated-gradient
sum reproduces $f(x) - f(x_0)$ within tolerance. The lag axis of every profile here is
**stored-coefficient time**, and every lag-resolved figure prints the caveat.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from teb_vae.lag_attn_cfs.eval import cohort, lag_hist, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import band_partition, labels
from teb_vae.lag_attn_cfs.eval.lag_axis import COEFFICIENT_LAG_AXIS_LABEL, GROUP_DELAY_CAVEAT
from teb_vae.lag_attn_cfs.eval.metrics import forecast_likelihood_terms
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

#: Integrated-gradient steps and Captum's internal batch of interpolated inputs. The step count is
#: the one at which the completeness residual fell below $10^{-3}$ of the readout on every fixture
#: with the entry fraction below (the conv-LSTM cell's per-step group norm makes its path the
#: roughest of the three); the residual is recorded per row so a production run can say whether
#: it held there too.
IG_STEPS = 64
IG_INTERNAL_BATCH_SIZE = 16

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

#: What was tried on the tiny fixtures and what came of it, recorded in every block so a reader
#: of a summary knows which methods were rejected and why rather than only which were kept.
METHOD_RECORD: Dict[str, Dict[str, str]] = {
    "IntegratedGradients": {
        "status": "shipped",
        "note": "the primary method; completeness holds to 1e-3 at 32 steps once the path enters "
                "the baseline at the entry fraction, and fails at every step count from the exact "
                "zero (see BASELINE_ENTRY_FRACTION)",
    },
    "LayerIntegratedGradients": {
        "status": "shipped",
        "note": "on the posterior head's INPUTS in the lag-attentive cells (the attended per-head "
                "summaries: a complete per-head split) and on the proposal head's OUTPUT in the "
                "lag-residual cell (a complete per-lag split through the limiter)",
    },
    "FeatureAblation": {
        "status": "shipped",
        "note": "model-agnostic; grouped by lag band of the source relative to the anchor, so it is "
                "the occlusion analysis's intervention read on this analysis's readouts and anchors",
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
    """

    name: str
    dense_latent: bool
    lag_readout: str
    lag_qualification: str
    layer_label: str
    layer_axis: str


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
)


# =============================================================================
# The wrapper
# =============================================================================
class AnchorReadout(nn.Module):
    r"""The model's dense forward, reduced to one scalar per sample at one anchor per sample.

    Captum attributes a function of its ``inputs`` tuple; this is that function. It calls the
    real model at the dense evaluation geometry, picks each row's own anchor from the anchor axis
    (``columns`` is per row, so a batch of rows can attribute several anchors of one segment in
    one call), and returns the chosen readout there. The block-score readouts rebuild the
    forecast target and the forecast mask from the two extra tensors on every call, because
    Captum expands the batch along its interpolation axis and a target built once outside would
    no longer match.

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

    def _forward_kwargs(self) -> Dict[str, Any]:
        """The dense geometry, plus the retained proposals where the lag readout needs them."""
        kwargs: Dict[str, Any] = {"anchor_phase": 0, "anchor_stride": 1}
        if not self.cell.dense_latent:
            kwargs["return_proposals"] = True
        return kwargs

    def _at_anchor(self, dense: torch.Tensor, anchors: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        """Read a per-row tensor at each row's own anchor: a gather on a dense axis, a select on the anchor axis."""
        if self.cell.dense_latent:
            index = anchors.view(-1, 1, *([1] * (dense.dim() - 2))).expand(-1, 1, *dense.shape[2:])
            return dense.gather(1, index).squeeze(1)
        return dense[torch.arange(dense.shape[0], device=dense.device), columns]

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
            columns: Each row's anchor as a position on the anchor axis, $(B,)$ ``long``.
            coordinates: Each row's latent coordinate for the per-coordinate readouts, $(B,)$
                ``long``; ignored by the others.

        Returns:
            The readout, $(B,)$.
        """
        model = self.model
        outputs = model(y_st, y_ph, u_stream, **self._forward_kwargs())
        rows = torch.arange(columns.shape[0], device=columns.device)
        anchors = outputs["anchor_index"][rows, columns]                 # (B,)
        anchor_valid = outputs["anchor_valid"][rows, columns]           # (B,)
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
            return self._at_anchor(dense, anchors, columns) + tie
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
            vector = self._at_anchor(dense, anchors, columns)          # (B, d_z)
            return vector[rows, coordinates] + tie
        if name in BLOCK_SCORE_READOUTS:
            anchor_column = anchors[:, None]
            target = model._build_forecast_target(target_features, anchor_column)
            mask, _coverage = forecast_mask(
                model.scored_weight(weight), model.geometry,
                coverage_floor=model.coverage_floor, anchors=anchor_column,
                anchor_valid=anchor_valid[:, None],
            )
            persistence = outputs.get("persistence")
            if persistence is not None:
                persistence = persistence[rows, columns][:, None]
            fidelity = name in (READOUT_MSE_FULL, READOUT_MSE_GAP)
            likelihood = "mse" if fidelity else self.likelihood
            # The density the collection pass scores under, so a block-score readout is the
            # ``mean_*`` column's own number; the fidelity readouts count the scored cells only.
            density = forecast_likelihood_terms(model)
            if fidelity:
                density["ar_coef"] = None
            gap = name in (READOUT_PRED_GAP, READOUT_MSE_GAP)
            scores: Dict[str, torch.Tensor] = {}
            for branch, key in (("full", "mu_post"), ("base", "mu_prior")):
                if not gap and not name.endswith(branch) and name != READOUT_NLL_HORIZON:
                    continue
                if name == READOUT_NLL_HORIZON and branch != "full":
                    continue
                mu = self._at_anchor(outputs[key], anchors, columns)[:, None]   # (B, 1, d_z)
                forecast_mu, forecast_logvar = model.decoder(mu, persistence=persistence)
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
            alpha = self._at_anchor(outputs["attn_weights"], anchors, columns)  # (B, M, L)
            return alpha.mean(dim=1)[:, low:high + 1].sum(dim=-1) + tie
        proposals = outputs["mean_proposals"][rows, columns]                    # (B, L, d_z)
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


def symlog_legend(ax: Any, *values: Any, ncol: int = 2, axis: str = "y", **kwargs: Any) -> None:
    """A symmetric-log axis with a decade of headroom and the legend placed in it."""
    symlog_axis(ax, *values, axis=axis, headroom=1.0)
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
    """Integrated gradients of one readout over the three streams, one row per anchor.

    Args:
        wrapper: The readout module.
        inputs: The row-expanded ``(y_st, y_ph, u_stream)``.
        extra: The row-expanded ``(target_features, weight)``.
        columns: Per-row anchor-axis positions, $(N,)$.
        baseline: One of :data:`BASELINES`.
        coordinates: Per-row latent coordinate, or ``None`` for readouts without one.
        n_steps: Integration steps.
        internal_batch_size: Captum's batch of interpolated inputs.
        entry_fraction: Where the path enters the baseline.

    Returns:
        The batch of attributions, empty arrays when there is no row.
    """
    n_rows = int(columns.shape[0])
    model = wrapper.model
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
        anchors = model(*inputs, **wrapper._forward_kwargs())["anchor_index"][
            torch.arange(n_rows, device=columns.device), columns
        ]
    method = IntegratedGradients(wrapper)
    attributions, delta = method.attribute(
        tuple(x.detach() for x in inputs),
        baselines=tuple(x.detach() for x in start),
        additional_forward_args=forward_args,
        n_steps=int(n_steps),
        internal_batch_size=int(internal_batch_size),
        return_convergence_delta=True,
    )
    target = torch.cat([attributions[0], attributions[1]], dim=-1)
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
        source=attributions[2].detach().cpu().to(torch.float32).numpy(),
    )


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
    # The target streams are one group each and are never ablated: their ids sit above every
    # source id, and their baselines equal their inputs, so ablating them changes nothing.
    n_source_groups = len(groups) + 1
    masks = (
        torch.full_like(inputs[0], n_source_groups, dtype=torch.long)[:1],
        torch.full_like(inputs[1], n_source_groups + 1, dtype=torch.long)[:1],
        mask,
    )
    baselines = baselines_for(BASELINE_SOURCE_NULL, inputs)
    attributions = FeatureAblation(wrapper).attribute(
        tuple(x.detach() for x in inputs),
        baselines=tuple(x.detach() for x in baselines),
        feature_mask=masks,
        additional_forward_args=(extra[0], extra[1], columns, coordinates),
        perturbations_per_eval=int(perturbations_per_eval),
    )
    source = attributions[2].detach().cpu().to(torch.float64).numpy()
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
        ``{'per_unit': (N, M or L), 'total': (N,), 'off_axis_total': (N,)}`` -- the per-head or
        per-lag attribution, its sum, and a zero column kept for the row schema. Empty when the
        cell's model does not build the layer.
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
        layer = getattr(model, "proposal_head", None)
    if n_rows == 0 or layer is None:
        return {"per_unit": np.zeros((0, 0)), "total": np.zeros(0), "off_axis_total": np.zeros(0)}
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
        with torch.no_grad():
            anchors = model(*inputs, **wrapper._forward_kwargs())["anchor_index"][
                torch.arange(n_rows, device=columns.device), columns
            ]
        out = method.attribute(
            tuple(x.detach() for x in inputs),
            baselines=tuple(x.detach() for x in start),
            additional_forward_args=forward_args,
            n_steps=int(n_steps),
            internal_batch_size=int(internal_batch_size),
        )
    finally:
        for name, value in saved.items():
            setattr(model, name, value)
    parts = [part for part in (out if isinstance(out, (tuple, list)) else [out]) if part is not None]
    rows = torch.arange(n_rows, device=columns.device)
    if cell.dense_latent:
        # One fused feature per head, each (B, T, d): the split is their sums at the anchor.
        per_unit = torch.stack([part[rows, anchors].sum(dim=-1) for part in parts], dim=1)   # (N, M)
    else:
        # The proposal head's outputs: the mean proposals, and the scale proposals where the arm
        # has them. Both reach the readout -- the divergence carries the scale update too -- so the
        # per-lag split sums the two channels at the row's anchor over the latent coordinates.
        per_unit = torch.stack([part[rows, columns].sum(dim=-1) for part in parts], dim=0).sum(dim=0)
    off_axis = torch.zeros(n_rows, dtype=per_unit.dtype, device=per_unit.device)
    return {"per_unit": _host(per_unit), "total": _host(per_unit.sum(dim=-1)), "off_axis_total": _host(off_axis)}


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
#: What each readout is called on a figure.
READOUT_TITLES: Mapping[str, str] = {
    READOUT_KLD: "divergence $K_t$",
    READOUT_PRED_GAP: "forecast gap (base $-$ full)",
    READOUT_NLL_FULL: "full-branch block score",
    READOUT_NLL_BASE: "base-branch block score",
    READOUT_MSE_FULL: "full-branch squared error (fidelity)",
    READOUT_MSE_GAP: "squared-error gap (base $-$ full)",
    READOUT_NLL_HORIZON: "full-branch score at one horizon step",
    READOUT_KLD_DIM: "top divergence coordinate $K_{t,d}$",
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
EXAMPLE_INPUT_ROW = 1.25
EXAMPLE_MAP_ROW = 1.0
EXAMPLE_LATENT_ROW = 0.6
EXAMPLE_LAYER_ROW = 0.8
EXAMPLE_LAG_ROW = 1.1


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


def _stream_map(
    figure: Any, ax: Any, field: Optional[np.ndarray], *, title: str, anchor: int,
    live: Optional[np.ndarray], n_scattering: Optional[int] = None, colorbar_label: str = "",
    kind: str = "signed", cax: Any = None, horizon: int = 0,
) -> Any:
    r"""One (channel x stored second) map with the cells the model never read blanked.

    Two kinds. An **input** map shows the standardised coefficients on a sequential scale with
    robust (1st to 99th percentile) limits taken over the live cells only; the cold cells of every
    channel, which the input gate multiplies by zero, are blanked rather than drawn, so the
    warm-up region neither sets the colour scale nor reads as data. A **signed** map shows an
    attribution on a symmetric-log scale about zero, with the cold cells and every step after the
    anchor blanked, because both are exactly zero by construction and a zero the model could not
    have produced is not a finding. The anchor is ruled and the forecast block it scores shaded.

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
    finite = masked[np.isfinite(masked)]
    if kind == "input":
        colormap = plt.get_cmap("viridis").with_extremes(bad=UNREAD_COLOUR)
        low, high = (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0))) if finite.size else (0.0, 1.0)
        norm = mcolors.Normalize(vmin=low, vmax=high if high > low else low + 1.0)
    else:
        colormap = plt.get_cmap("RdBu_r").with_extremes(bad=UNREAD_COLOUR)
        norm = signed_log_norm(masked) or mcolors.Normalize(vmin=-1.0, vmax=1.0)
    left, right = sample_cell_edges(steps, float(SECONDS_PER_STEP))
    image = ax.imshow(
        masked.T, aspect="auto", origin="upper", cmap=colormap, norm=norm, interpolation="none",
        extent=(left, right, masked.shape[1] - 0.5, -0.5),
    )
    if n_scattering and 0 < int(n_scattering) < masked.shape[1]:
        ax.axhline(int(n_scattering) - 0.5, color=figures.COLOR_BLACK, linewidth=plt.rcParams["axes.linewidth"])
    if horizon:
        ax.axvspan((anchor + 0.5) * SECONDS_PER_STEP, (anchor + 0.5 + int(horizon)) * SECONDS_PER_STEP,
                   color=figures.COLOR_VERMILLION, alpha=0.12, linewidth=0, zorder=2)
    ax.axvline(float(anchor) * SECONDS_PER_STEP, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_REGULAR)
    ax.set_xlim(left, right)
    ax.set_title(title)
    ax.set_xlabel("stored time (s)")
    ax.set_ylabel("declared channel")
    _attach_colorbar(figure, image, ax=ax, cax=cax, label=colorbar_label, norm=norm)
    figures.style_axes(ax, grid="none")
    return image


def _lag_view(
    ax: Any, profile: np.ndarray, model_profile: np.ndarray, *, lag_seconds: np.ndarray,
    cell: CellBinding, title: str,
) -> None:
    """Normalised $|q_\\ell|$ beside the normalised model lag readout, on one share axis.

    One axis rather than a twin: both are shares per lag once normalised, and two scales on one
    panel is the one layout a reader cannot check by eye. Symmetric-log, because one lag's share
    is often a hundred times another's and a linear axis shows only the one.
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
    fit = agreement(profile[None, :], np.asarray(model_profile, dtype=np.float64)[None, :])
    ax.set_title(f"{title}; corr {fit['lag_corr'][0]:.2f}, JS {fit['lag_js'][0]:.2f}")
    ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
    ax.set_ylabel("share per lag")
    symlog_legend(ax, share, model_share, ncol=2)
    figures.style_axes(ax)


def _example_title(item: Mapping[str, Any]) -> str:
    """The identity line of one example anchor: who, where in the cohort, and when."""
    hours = -(float(item["epoch"]) + float(item["anchor"]) * SECONDS_PER_STEP) / cohort.SECONDS_PER_HOUR
    return (
        f"guid {item['guid']} — subgroup {item[labels.SUBGROUP_COLUMN]} — class {item[labels.CLASS_COLUMN]} "
        f"— anchor step {int(item['anchor'])}, {hours:.2f} h before delivery"
    )


def build_map_figure(
    examples: Sequence[Mapping[str, Any]],
    *,
    lag_seconds: np.ndarray,
    cell: CellBinding,
    caveat: str,
) -> Any:
    r"""One row per class example: the inputs the encoders read, and what $K_t$ was attributed to.

    Five columns. The two **input** maps first -- the declared target stream (scattering block
    above the phase-harmonic block) and the declared source stream, standardised coefficients over
    stored second and channel with the cold cells blanked -- because an attribution map is read
    against the coefficients it was taken on. Then the attribution of $K_t$: to the target under
    the **all-zero** baseline, which is the only baseline under which target inputs move, and to
    the source under the **source-null** baseline, which is the primary comparison, both on a
    symmetric-log scale. Last, the lag-aligned source attribution against the model's own lag
    readout at that anchor.

    Args:
        examples: One per class, as :func:`attribution_pass.attribute_example` builds them.
        lag_seconds: The compensated lag axis.
        cell: The cell binding, for the legend.
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    n_rows = max(len(examples), 1)
    figure, axes = figures.new_figure(n_rows, 5, height_per_row=2.5, width=17.0)
    for row in range(n_rows):
        if row >= len(examples):
            for col in range(5):
                _empty(axes[row, col])
            continue
        item = examples[row]
        anchor = int(item["anchor"])
        horizon = int(item.get("horizon", 0) or 0)
        inputs = item.get("inputs") or {}
        maps = item.get("maps") or {}
        _stream_map(
            figure, axes[row, 0], inputs.get(STREAM_TARGET), title="target input (standardised)",
            anchor=anchor, live=item.get("live_target"), n_scattering=item.get("n_scattering"), kind="input",
            horizon=horizon,
        )
        _stream_map(
            figure, axes[row, 1], inputs.get(STREAM_SOURCE), title="source input (standardised)",
            anchor=anchor, live=item.get("live_source"), kind="input", horizon=horizon,
        )
        all_zero = maps.get((READOUT_KLD, BASELINE_ALL_ZERO, ""))
        null = maps.get((READOUT_KLD, BASELINE_SOURCE_NULL, ""))
        _stream_map(
            figure, axes[row, 2], None if all_zero is None else all_zero[STREAM_TARGET],
            title="$K_t$ to the target (all-zero)", anchor=anchor,
            live=item.get("live_target"), n_scattering=item.get("n_scattering"), colorbar_label="nats",
            horizon=horizon,
        )
        _stream_map(
            figure, axes[row, 3], None if null is None else null[STREAM_SOURCE],
            title="$K_t$ to the source (source-null)", anchor=anchor,
            live=item.get("live_source"), colorbar_label="nats", horizon=horizon,
        )
        if null is None:
            _empty(axes[row, 4], "lag-aligned source attribution of $K_t$")
        else:
            _lag_view(
                axes[row, 4], null["lag_profile"], item["model_profile"], lag_seconds=lag_seconds,
                cell=cell, title="$K_t$ by lag",
            )
        axes[row, 0].text(
            -0.22, 0.5, _example_title(item).replace(" — ", "\n"), transform=axes[row, 0].transAxes,
            rotation=90, ha="center", va="center", fontsize=figures.FONT_TINY,
        )
    figures.caveat_note(figure, f"{cell.lag_qualification}. {caveat}. {GROUP_DELAY_CAVEAT}")
    return figure


def build_example_figure(
    item: Mapping[str, Any],
    *,
    lag_seconds: np.ndarray,
    cell: CellBinding,
    caveat: str,
    lag_bands: Mapping[str, Tuple[int, int]],
    horizons: Optional[Mapping[str, int]] = None,
) -> Any:
    r"""One anchor of one recording, every readout, on one stored-time axis: the sample page's layout.

    The page is a stack of full-width rows on one shared time axis, laid out as the samples pages
    are -- one data column and one colour-axis column, every row spanning the whole segment -- so
    a column of the page is the same stored second on every row and a reader compares maps by
    looking down rather than across. The rows: the two input streams the encoders read (cold
    cells blanked); then, per readout variant, its target attribution under the all-zero baseline
    and its source attribution under the source-null baseline, each on a symmetric-log scale
    with the value at the input and at the exact null in the title; then three rows off the time
    axis -- the latent at the anchor (prior mean, source shift, per-coordinate divergence), the
    layer split per readout on the cell's own axis (head or lag), and every readout's lag-aligned
    source attribution overlaid on one lag axis against the model's lag readout.

    Args:
        item: The example, as :func:`attribution_pass.attribute_example` builds it.
        lag_seconds: The compensated lag axis.
        cell: The cell binding.
        caveat: The sentence printed under the figure.
        lag_bands: The configured lag bands, in the order their rows are drawn.
        horizons: The named horizon steps, in the order their rows are drawn, or ``None``.

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

    rows: List[Tuple[str, float]] = [("input_target", EXAMPLE_INPUT_ROW), ("input_source", EXAMPLE_INPUT_ROW)]
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
    bottom = max(0.02, 0.35 / figure_height) + figures.caveat_note(
        figure, f"{cell.lag_qualification}. {caveat}. {GROUP_DELAY_CAVEAT}"
    )
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
        if name == "input_target":
            _stream_map(figure, ax, inputs.get(STREAM_TARGET), title="target input coefficients (standardised; cold cells blanked)",
                        anchor=anchor, live=item.get("live_target"), n_scattering=item.get("n_scattering"),
                        kind="input", cax=cax, horizon=horizon)
        elif name == "input_source":
            _stream_map(figure, ax, inputs.get(STREAM_SOURCE), title="source input coefficients (standardised; cold cells blanked)",
                        anchor=anchor, live=item.get("live_source"), kind="input", cax=cax, horizon=horizon)
        else:
            stream, readout, tag = name.split(":", 2)
            baseline = BASELINE_ALL_ZERO if stream == STREAM_TARGET else BASELINE_SOURCE_NULL
            entry = maps.get((readout, baseline, tag))
            label = variant_label(readout, tag)
            values = "" if entry is None else f" — at the input {entry['value_input']:.3g}, at the null {entry['value_baseline']:.3g}"
            _stream_map(
                figure, ax, None if entry is None else entry[stream],
                title=f"{stream} attribution of the {label} ({baseline} baseline){values}", anchor=anchor,
                live=item.get(f"live_{stream}"), n_scattering=item.get("n_scattering") if stream == STREAM_TARGET else None,
                colorbar_label="readout units", kind="signed", cax=cax, horizon=horizon,
            )
        if position < n_time_rows - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")

    position = n_time_rows
    if latent:
        ax, cax = row_axes(position, shared=False)
        position += 1
        field = np.stack([np.asarray(latent[key], dtype=np.float64) for key in ("mu_prior", "shift", "kld_dim")], axis=0)
        norm = signed_log_norm(field) or mcolors.Normalize(vmin=-1.0, vmax=1.0)
        image = ax.imshow(field, aspect="auto", origin="upper", cmap="RdBu_r", norm=norm, interpolation="none",
                          extent=(-0.5, field.shape[1] - 0.5, 2.5, -0.5))
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["$\\mu^p$", "$\\mu^q - \\mu^p$", "$K_{t,d}$"])
        top = int(np.argmax(np.asarray(latent["kld_dim"], dtype=np.float64)))
        ax.plot([top], [2], marker="v", color=figures.COLOR_BLACK, markersize=4, linestyle="none")
        ax.set_title(f"the latent at the anchor: prior mean, source shift and per-coordinate divergence "
                     f"(top coordinate {top}, $K_t$ = {float(np.sum(latent['kld_dim'])):.3g} nats)")
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
        norm = signed_log_norm(field) or mcolors.Normalize(vmin=-1.0, vmax=1.0)
        if cell.layer_axis == "head":
            extent = (-0.5, width - 0.5, len(keys) - 0.5, -0.5)
        else:
            half = 0.5 * float(SECONDS_PER_STEP)
            extent = (float(lag_seconds[0]) - half, float(lag_seconds[min(width, len(lag_seconds)) - 1]) + half, len(keys) - 0.5, -0.5)
        image = ax.imshow(field, aspect="auto", origin="upper", cmap=plt.get_cmap("RdBu_r").with_extremes(bad=UNREAD_COLOUR),
                          norm=norm, interpolation="none", extent=extent)
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels([variant_label(*key) for key in keys], fontsize=figures.FONT_TINY)
        ax.set_title(f"activations: attribution on {cell.layer_label}, per {cell.layer_axis}, of every readout (source-null baseline)")
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
                label=variant_label(readout, tag))
    if np.isfinite(model_profile).any():
        model_share = lag_hist.normalise(model_profile)[0]
        series.append(model_share)
        ax.plot(lag_seconds, model_share, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle=":",
                label=f"model lag readout ({cell.lag_readout})")
    if series:
        ax.set_title("every readout on one lag axis: normalised |source attribution| by offset from the anchor (source-null baseline)")
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share per lag")
        symlog_legend(ax, *series, ncol=3)
        figures.style_axes(ax)
    else:
        _empty(ax, "every readout on one lag axis")

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
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 2, height_per_row=3.4, width=13.0)
    subset = blocks[blocks["baseline"].astype(str) == BASELINE_ALL_ZERO] if len(blocks) else blocks
    if subset.empty:
        _empty(axes[0, 0], "unsigned attribution share by input block")
        _empty(axes[0, 1], "signed attribution by input block")
        figures.caveat_note(figure, caveat)
        return figure
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
    ax.set_xlabel("share of the unsigned attribution")
    ax.set_title("unsigned share by input block (all-zero baseline)", fontsize=figures.FONT_SMALL)
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
    ax.set_title("signed block sums (positive raised the readout)", fontsize=figures.FONT_SMALL)
    ax.legend(fontsize=figures.FONT_TINY, loc="lower right", ncol=2)
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
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 3, height_per_row=3.0, width=15.0)
    if not len(rows) or not horizons:
        for col, title in enumerate(("stream totals per horizon step", "source attribution by lag per horizon step",
                                     "target attribution by offset per horizon step")):
            _empty(axes[0, col], title)
        figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
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
        ax.bar(x + (offset - 0.5) * 0.38, heights, width=0.38, color=colour, label=f"{stream} |attribution| total ({base})")
    scores = []
    for name in names:
        keep = (readout == READOUT_NLL_HORIZON) & (baseline == BASELINE_SOURCE_NULL) & (tag == f"h{int(horizons[name])}")
        column = rows["value_input"].to_numpy(dtype=np.float64)
        scores.append(float(_per_recording_mean(column[:, None], guids, keep)[0]) if keep.any() else np.nan)
    ax.plot(x, scores, color=figures.COLOR_BLACK, marker="o", markersize=figures.MARKER_SMALL, linewidth=figures.LINE_THIN,
            label="score at the input (nats)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name} (step {int(horizons[name])})" for name in names])
    ax.set_title("per-step score: stream totals", fontsize=figures.FONT_SMALL)
    ax.set_ylabel("nats per anchor (symlog)")
    symlog_legend(ax, *bars, np.asarray(scores), ncol=1)
    figures.style_axes(ax)
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
            _empty(ax, f"{stream} |attribution| by offset per horizon step")
            continue
        ax.set_title(f"per-step score: {stream} |attribution| by offset ({base})", fontsize=figures.FONT_SMALL)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|attribution| (nats, symlog)")
        symlog_legend(ax, *drawn, ncol=2)
        figures.style_axes(ax)
    figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
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
    (right), with the recording-mean agreement statistics in the titles. Normalised per row so a
    recording with a large readout does not decide the shape for every other. A last row puts
    every readout on one axis: the divergence, the forecast gap and the lag readout on each band,
    so where the model is sensitive for its latent change, for its forecast and for each band's
    own lag readout can be read against each other and against the model's lag profile.

    Args:
        rows: The per-row table.
        vectors: The row-aligned arrays, read for ``lag_profile`` and ``model_profile``.
        lag_seconds: The compensated lag axis.
        readouts: The readouts to draw, one row each.
        cell: The cell binding.
        caveat: The sentence printed under the figure.
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

    for index, readout in enumerate(readouts):
        keep = null & (readout_column == readout) if null.size else null
        if not keep.any():
            _empty(axes[index, 0], f"{READOUT_TITLES.get(readout, readout)}: lag-aligned attribution")
            _empty(axes[index, 1], f"{READOUT_TITLES.get(readout, readout)}: by class")
            continue
        subset = rows[keep]
        ax = axes[index, 0]
        ax.plot(lag_seconds, _per_recording_mean(attribution, guids, keep), color=figures.COLOR_BLUE,
                linewidth=figures.LINE_REGULAR, label="|attribution|, normalised")
        ax.plot(lag_seconds, _per_recording_mean(model_profile, guids, keep), color=figures.COLOR_ORANGE,
                linewidth=figures.LINE_THIN, label="model lag readout, normalised")
        corr = float(np.nanmean(subset["lag_corr"])) if "lag_corr" in subset else float("nan")
        js = float(np.nanmean(subset["lag_js"])) if "lag_js" in subset else float("nan")
        ax.set_title(
            f"{READOUT_TITLES.get(readout, readout)}: pooled over {len(np.unique(guids[keep]))} "
            f"recording(s); corr {corr:.2f}, JS {js:.2f}",
        )
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share per lag")
        symlog_legend(ax, _per_recording_mean(attribution, guids, keep), _per_recording_mean(model_profile, guids, keep), ncol=2)
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
                    linewidth=figures.LINE_REGULAR, label=f"{name} (n={len(np.unique(guids[of_class]))})")
            ax.plot(lag_seconds, _per_recording_mean(model_profile, guids, of_class), color=colour,
                    linewidth=figures.LINE_THIN, linestyle="--")
            drawn += 1
        if drawn == 0:
            _empty(ax, f"{READOUT_TITLES.get(readout, readout)}: by class")
        else:
            ax.set_title(f"{READOUT_TITLES.get(readout, readout)}: by class (solid attribution, dashed model readout)")
            ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
            ax.set_ylabel("share per lag")
            symlog_legend(ax, _per_recording_mean(attribution, guids, keep), ncol=min(max(drawn, 1), 3))
            figures.style_axes(ax)

    # The comparison row: every readout on one axis, then the band readouts against their bands.
    ax, ax_bands = axes[len(readouts), 0], axes[len(readouts), 1]
    drawn = 0
    series: List[Tuple[str, np.ndarray, str]] = []
    for readout in readouts:
        keep = null & (readout_column == readout) if null.size else null
        if keep.any():
            series.append((READOUT_TITLES.get(readout, readout), _per_recording_mean(attribution, guids, keep),
                           READOUT_COLOURS.get(readout, figures.COLOR_GRAY)))
    band_series: List[Tuple[str, np.ndarray, str, Tuple[int, int]]] = []
    for position, (name, span) in enumerate(bands.items()):
        keep = null & (readout_column == READOUT_LAG_BAND) & (band_column == str(name)) if null.size else null
        if keep.any():
            colour = BAND_COLOURS[position % len(BAND_COLOURS)]
            profile = _per_recording_mean(attribution, guids, keep)
            series.append((f"lag readout on {name!r}", profile, colour))
            band_series.append((str(name), profile, colour, (int(span[0]), int(span[1]))))
    for label, profile, colour in series:
        ax.plot(lag_seconds, profile, color=colour, linewidth=figures.LINE_REGULAR, label=label)
        drawn += 1
    if null.any():
        ax.plot(lag_seconds, _per_recording_mean(model_profile, guids, null), color=figures.COLOR_BLACK,
                linewidth=figures.LINE_THIN, linestyle="--", label="model lag readout")
    if drawn == 0:
        _empty(ax, "every readout on one lag axis")
    else:
        ax.set_title("every readout on one lag axis: normalised |source attribution|, pooled")
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("share per lag")
        symlog_legend(ax, *[profile for _label, profile, _colour in series], ncol=2)
        figures.style_axes(ax)
    if not band_series:
        _empty(ax_bands, "the lag-band readouts against their own bands")
    else:
        for name, profile, colour, (low, high) in band_series:
            low, high = max(0, low), min(len(lag_seconds) - 1, high)
            if low <= high:
                ax_bands.axvspan(
                    lag_seconds[low] - 0.5 * SECONDS_PER_STEP, lag_seconds[high] + 0.5 * SECONDS_PER_STEP,
                    color=colour, alpha=0.12, linewidth=0,
                )
            ax_bands.plot(lag_seconds, profile, color=colour, linewidth=figures.LINE_REGULAR, label=name)
        ax_bands.set_title("the lag-band readouts against their own bands (shaded)")
        ax_bands.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax_bands.set_ylabel("share per lag")
        symlog_legend(ax_bands, *[profile for _name, profile, _colour, _span in band_series], ncol=min(len(band_series), 4))
        figures.style_axes(ax_bands)
    figures.caveat_note(figure, f"{cell.lag_qualification}. {caveat}. {GROUP_DELAY_CAVEAT}")
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
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.4, width=12.0)
    guids = rows["guid"].astype(str).to_numpy() if len(rows) else np.zeros(0, dtype=object)
    for index, readout in enumerate(readouts):
        for col, stream in enumerate(STREAMS):
            ax = axes[index, col]
            title = f"{READOUT_TITLES.get(readout, readout)}: {stream} stream by declared channel"
            keep = _stream_selection(rows, readout, stream)
            signed_name, unsigned_name = f"channel_profile_{stream}", f"channel_abs_profile_{stream}"
            if not keep.any() or signed_name not in vectors:
                _empty(ax, title)
                continue
            signed = _per_recording_mean(vectors[signed_name].astype(np.float64), guids, keep)
            width = int(signed.size)
            colours, legend = _band_colour_of_channels(width, channel_groups.get(stream, {}))
            x = np.arange(width)
            ax.bar(x, signed, width=0.85, color=colours, linewidth=0, label="signed, mean over recordings")
            if unsigned_name in vectors:
                unsigned = _per_recording_mean(vectors[unsigned_name].astype(np.float64), guids, keep)
                ax.plot(x, unsigned, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN,
                        label="unsigned, mean over recordings")
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            if stream == STREAM_TARGET and n_scattering and 0 < int(n_scattering) < width:
                ax.axvline(float(n_scattering) - 0.5, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE,
                           linestyle="--")
                ax.text(float(n_scattering) - 0.5, 0.98, " phase-harmonic block", transform=ax.get_xaxis_transform(),
                        ha="left", va="top", fontsize=figures.FONT_TINY, color=figures.COLOR_GRAY)
            for label, colour in legend:
                ax.plot([], [], marker="s", linestyle="none", color=colour, label=label)
            ax.set_title(f"{title} ({len(np.unique(guids[keep]))} recording(s))")
            ax.set_xlabel("declared channel")
            ax.set_ylabel("attribution (symlog)")
            ax.set_xlim(-0.5, width - 0.5)
            symlog_legend(ax, signed, unsigned if unsigned_name in vectors else signed, ncol=3)
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
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.8, width=12.0)
    half = 0.5 * float(SECONDS_PER_STEP)
    for index, readout in enumerate(readouts):
        for col, stream in enumerate(STREAMS):
            ax = axes[index, col]
            baseline = BASELINE_ALL_ZERO if stream == STREAM_TARGET else BASELINE_SOURCE_NULL
            title = f"{READOUT_TITLES.get(readout, readout)}: |attribution| of the {stream} stream by offset and channel"
            entry = lag_channel.get((readout, baseline, stream))
            if entry is None or not np.isfinite(np.asarray(entry["mean_abs"])).any():
                _empty(ax, title)
                continue
            field = np.asarray(entry["mean_abs"], dtype=np.float64).T   # (C, L)
            figures.heatmap_with_colorbar(
                figure, ax, field, symmetric=False, interpolation="none", norm=unsigned_log_norm(field),
                title=f"{title} ({int(entry['n_rows'])} anchor(s), {baseline} baseline)",
                xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel="declared channel", colorbar_label="|attribution| (log)",
                extent=(float(lag_seconds[0]) - half, float(lag_seconds[-1]) + half, field.shape[0] - 0.5, -0.5),
                separator_row=(int(n_scattering) - 1) if (stream == STREAM_TARGET and n_scattering) else None,
            )
    figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
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
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.4, width=12.0)
    guids = rows["guid"].astype(str).to_numpy() if len(rows) else np.zeros(0, dtype=object)
    for index, readout in enumerate(readouts):
        for col, (stream, name) in enumerate(((STREAM_SOURCE, "lag_profile"), (STREAM_TARGET, "target_lag_profile"))):
            ax = axes[index, col]
            title = f"{READOUT_TITLES.get(readout, readout)}: signed {stream} attribution by offset from the anchor"
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
                            label="positive part (raises the readout)")
            ax.fill_between(lag_seconds, negative, 0.0, color=figures.COLOR_BLUE, alpha=0.35, linewidth=0,
                            label="negative part (lowers the readout)")
            ax.plot(lag_seconds, net, color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="net mean")
            ax.plot(lag_seconds, unsigned, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN, linestyle="--",
                    label="unsigned mean")
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title(f"{title} ({len(np.unique(guids[keep]))} recording(s))")
            ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
            ax.set_ylabel("attribution (symlog)")
            symlog_legend(ax, positive, negative, unsigned, ncol=2)
            figures.style_axes(ax)
    figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
    return figure


def build_checks_figure(rows: pd.DataFrame, *, tolerance: float, caveat: str) -> Any:
    r"""The numerical checks behind every row, so a map is read after its residual, not before.

    Left: the relative completeness residual of every row per readout and baseline, on a
    logarithmic axis against the tolerance the pass counts against. Middle: the entry jump
    $f(x_0) - f(b)$ against the readout at the input, per baseline -- how much of the readout the
    excluded start of the path accounts for. Right: the two structural checks, the largest
    attribution to a step after the anchor and to a gated-off source step, per readout; exactly
    zero on a causal model, and drawn on a symmetric-log axis so a non-zero shows.

    Args:
        rows: The per-row table.
        tolerance: The completeness tolerance the pass counts rows against.
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 3, height_per_row=3.0, width=13.5)
    ax = axes[0, 0]
    if len(rows) and "completeness_rel" in rows.columns:
        residual = np.asarray(rows["completeness_rel"], dtype=np.float64)
        finite = residual[np.isfinite(residual) & (residual > 0.0)]
        if finite.size:
            edges = np.logspace(np.log10(max(finite.min(), 1e-12)), np.log10(max(finite.max(), tolerance * 10.0)), 30)
            groups = list(dict.fromkeys(zip(rows["readout"].astype(str), rows["baseline"].astype(str))))
            for position, (readout, baseline) in enumerate(groups):
                keep = ((rows["readout"].astype(str) == readout) & (rows["baseline"].astype(str) == baseline)).to_numpy()
                values = residual[keep]
                values = values[np.isfinite(values) & (values > 0.0)]
                if values.size:
                    ax.hist(values, bins=edges, histtype="step", linewidth=figures.LINE_REGULAR,
                            color=(list(READOUT_COLOURS.values()) + list(BAND_COLOURS))[position % 7],
                            linestyle="-" if baseline == BASELINE_SOURCE_NULL else "--",
                            label=f"{READOUT_TITLES.get(readout, readout)}, {baseline}")
            ax.axvline(float(tolerance), color=figures.COLOR_BLACK, linestyle=":", linewidth=figures.LINE_REGULAR,
                       label=f"tolerance {tolerance:g}")
            over = int((residual > tolerance).sum())
            ax.set_xscale("log")
            ax.set_title(f"completeness residual per row ({over} of {residual.size} over tolerance)")
            ax.set_xlabel("relative completeness residual")
            ax.set_ylabel("rows")
            figures.legend_with_headroom(ax, ncol=2, headroom=0.5)
            figures.style_axes(ax)
        else:
            _empty(ax, "completeness residual per row")
    else:
        _empty(ax, "completeness residual per row")

    ax = axes[0, 1]
    if len(rows) and {"entry_jump", "value_input"} <= set(rows.columns):
        drawn = 0
        for baseline, colour in ((BASELINE_SOURCE_NULL, figures.COLOR_BLUE), (BASELINE_ALL_ZERO, figures.COLOR_VERMILLION)):
            keep = (rows["baseline"].astype(str) == baseline).to_numpy()
            if keep.any():
                ax.scatter(rows.loc[keep, "value_input"], rows.loc[keep, "entry_jump"], s=9, color=colour,
                           alpha=0.7, linewidths=0, label=f"{baseline} baseline")
                drawn += 1
        if drawn:
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title("entry jump $f(x_0) - f(b)$ against the readout at the input")
            ax.set_xlabel("readout at the input")
            ax.set_ylabel("entry jump (readout units)")
            figures.legend_with_headroom(ax, ncol=2, headroom=0.3)
            figures.style_axes(ax)
        else:
            _empty(ax, "entry jump against the readout at the input")
    else:
        _empty(ax, "entry jump against the readout at the input")

    ax = axes[0, 2]
    checks = [("after_anchor_max_abs", "after the anchor"), ("gated_off_max_abs", "gated-off source step")]
    if len(rows) and all(name in rows.columns for name, _ in checks):
        readouts = list(dict.fromkeys(rows["readout"].astype(str)))
        x = np.arange(len(readouts))
        for offset, (name, label) in enumerate(checks):
            values = [float(np.nanmax(rows.loc[rows["readout"].astype(str) == readout, name])) for readout in readouts]
            ax.bar(x + (offset - 0.5) * 0.38, values, width=0.38, label=f"largest |attribution| {label}",
                   color=figures.COLOR_BLUE if offset == 0 else figures.COLOR_ORANGE)
            for position, value in zip(x + (offset - 0.5) * 0.38, values):
                ax.annotate(f"{value:.2g}", (position, value), textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=figures.FONT_TINY)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xticks(x)
        ax.set_xticklabels([READOUT_TITLES.get(readout, readout) for readout in readouts], fontsize=figures.FONT_TINY)
        ax.set_title("structural checks per readout (exactly zero on a causal model)")
        ax.set_ylabel("|attribution| (readout units), symmetric log")
        figures.legend_with_headroom(ax, ncol=1, headroom=0.4)
        figures.style_axes(ax)
    else:
        _empty(ax, "structural checks per readout")
    figures.caveat_note(figure, caveat)
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
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(max(len(readouts), 1), 2, height_per_row=2.4, width=12.0)
    window = 1.0
    if len(rows) and "lag_profile" in vectors:
        hours = -(np.asarray(rows["epoch"], dtype=np.float64) + np.asarray(rows["anchor"], dtype=np.float64) * SECONDS_PER_STEP) / cohort.SECONDS_PER_HOUR
        centroid = _lag_centroid(vectors["lag_profile"], lag_seconds)
        classes = rows[labels.CLASS_COLUMN].astype(object).to_numpy()
        guids = rows["guid"].astype(str).to_numpy()
    else:
        hours = centroid = np.zeros(0)
        classes = guids = np.zeros(0, dtype=object)
    for index, readout in enumerate(readouts):
        keep = _stream_selection(rows, readout, STREAM_SOURCE)
        for col, (values, ylabel, what) in enumerate((
            (np.asarray(rows["source_total"], dtype=np.float64) if len(rows) else np.zeros(0), "readout units", "source attribution total"),
            (centroid, "s (stored-coefficient time)", "lag centroid of |source attribution|"),
        )):
            ax = axes[index, col]
            title = f"{READOUT_TITLES.get(readout, readout)}: {what} against hours before delivery"
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
                           label=f"{name} (n={len(np.unique(guids[of_class]))})")
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
            figures.legend_with_headroom(ax, ncol=3, headroom=0.35)
            figures.style_axes(ax)
    figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
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
    band, one bar group per stream, with the spectral-skill gap per target band drawn on a twin
    axis where the run carries it. Bottom: per lag band of the source, the integrated-gradient
    share, the feature-ablation delta of this analysis, and the occlusion analysis's delta where
    the run carries it.

    Args:
        bands: The frequency-band table.
        lag_bands: The lag-band table.
        readouts: The readouts, one column each.
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    n_cols = max(len(readouts), 1)
    figure, axes = figures.new_figure(2, n_cols, height_per_row=3.0, width=5.0 * n_cols)
    for col, readout in enumerate(readouts):
        ax = axes[0, col]
        subset = bands[bands["readout"].astype(str) == readout] if len(bands) else bands
        if subset.empty:
            _empty(ax, f"{readout}: by frequency band")
        else:
            names = list(dict.fromkeys(subset["band"].astype(str)))
            x = np.arange(len(names))
            width = 0.38
            for offset, (stream, colour) in enumerate(((STREAM_TARGET, figures.COLOR_BLUE), (STREAM_SOURCE, figures.COLOR_ORANGE))):
                part = subset[subset["stream"].astype(str) == stream].set_index("band")
                heights = [float(part["attribution_mean"].get(name, np.nan)) for name in names]
                ax.bar(x + (offset - 0.5) * width, heights, width=width, color=colour, label=f"{stream} attribution")
            if "spectral_skill_pred_gap_nats" in subset.columns and subset["spectral_skill_pred_gap_nats"].notna().any():
                twin = ax.twinx()
                part = subset[subset["stream"].astype(str) == STREAM_TARGET].set_index("band")
                twin.plot(x, [float(part["spectral_skill_pred_gap_nats"].get(name, np.nan)) for name in names],
                          color=figures.COLOR_GREEN, marker="o", markersize=2.5, linewidth=figures.LINE_THIN,
                          label="spectral_skill pred_gap (target band)")
                twin.set_ylabel("nats per anchor", fontsize=figures.FONT_LABEL)
                twin.tick_params(labelsize=figures.FONT_TINY)
                twin.legend(fontsize=figures.FONT_TINY, loc="lower right")
            ax.set_xticks(x)
            # Ticks name the frequency range (period in parentheses), not the clinical band key
            # the table is keyed by.
            ax.set_xticklabels(
                [band_partition.band_display_label(name) for name in names],
                rotation=30, ha="right", fontsize=figures.FONT_TINY,
            )
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title(f"{readout}: attribution by frequency band (source-null baseline)", fontsize=figures.FONT_SMALL)
            ax.set_ylabel("attribution (symlog)")
            symlog_axis(ax, subset["attribution_mean"].to_numpy(dtype=np.float64), headroom=1.0)
            ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
            figures.style_axes(ax)

        ax = axes[1, col]
        subset = lag_bands[lag_bands["readout"].astype(str) == readout] if len(lag_bands) else lag_bands
        if subset.empty:
            _empty(ax, f"{readout}: by lag band")
        else:
            names = list(dict.fromkeys(subset["band"].astype(str)))
            x = np.arange(len(names))
            series = [
                ("ig_attribution_mean", "integrated gradients (sum over the band)", figures.COLOR_BLUE),
                ("ablation_delta_mean", "feature ablation delta (this analysis)", figures.COLOR_PURPLE),
                ("occlusion_delta_total_nats", "occlusion delta (its own pass)", figures.COLOR_GREEN),
            ]
            present = [(column, label, colour) for column, label, colour in series
                       if column in subset.columns and subset[column].notna().any()]
            width = 0.8 / max(len(present), 1)
            indexed = subset.set_index("band")
            for offset, (column, label, colour) in enumerate(present):
                ax.bar(x + (offset - (len(present) - 1) / 2) * width,
                       [float(indexed[column].get(name, np.nan)) for name in names],
                       width=width, color=colour, label=label)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=figures.FONT_TINY)
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_title(f"{readout}: source by lag band relative to the anchor", fontsize=figures.FONT_SMALL)
            ax.set_ylabel("readout units (symlog)")
            symlog_axis(ax, *[subset[column].to_numpy(dtype=np.float64) for column, _l, _c in present], headroom=1.0)
            ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
            figures.style_axes(ax)
    figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
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
    the attentive cells, per lag in the residual cell -- pooled and by class. Right: the mean
    time profile, per stream, of the attribution of the anchor's largest per-coordinate divergence
    re-indexed by offset from the anchor.

    Args:
        layer: The layer table, long-form: ``readout, unit, clinical_class, n_recordings, mean``.
        top_coordinate: Rows of the per-row table for the top-coordinate readout, with their
            lag-aligned source and target profiles attached as columns ``lag_profile`` and
            ``target_lag_profile`` (arrays).
        cell: The cell binding.
        lag_seconds: The compensated lag axis.
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 2, height_per_row=3.2, width=12.0)
    ax = axes[0, 0]
    if layer.empty:
        _empty(ax, f"layer attribution per {cell.layer_axis}")
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
                       width=0.8 / len(readouts), label=readout)
            else:
                ax.plot(x, values, linewidth=figures.LINE_REGULAR, label=readout)
        ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
        if cell.layer_axis == "head":
            # Heads are integers; a fractional tick between two heads names nothing.
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(unit)) for unit in units])
        ax.set_title(f"attribution on {cell.layer_label}, per {cell.layer_axis} (source-null baseline)", fontsize=figures.FONT_SMALL)
        ax.set_xlabel("head" if cell.layer_axis == "head" else COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("attribution, mean over recordings (symlog)")
        symlog_axis(ax, pooled["mean"].to_numpy(dtype=np.float64) if not pooled.empty else layer["mean"].to_numpy(dtype=np.float64), headroom=1.0)
        ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
        figures.style_axes(ax)
    ax = axes[0, 1]
    if top_coordinate.empty or "lag_profile" not in top_coordinate.columns:
        _empty(ax, "how the top divergence coordinate is fed")
    else:
        source = np.stack([np.asarray(v, dtype=np.float64) for v in top_coordinate["lag_profile"]], axis=0)
        target = np.stack([np.asarray(v, dtype=np.float64) for v in top_coordinate["target_lag_profile"]], axis=0)
        ax.plot(lag_seconds, np.nanmean(np.abs(source), axis=0), color=figures.COLOR_ORANGE,
                linewidth=figures.LINE_REGULAR, label="source |attribution| by lag")
        ax.plot(lag_seconds, np.nanmean(np.abs(target), axis=0), color=figures.COLOR_BLUE,
                linewidth=figures.LINE_REGULAR, label="target |attribution| by offset")
        ax.set_title(
            f"the anchor's largest K_(t,d) coordinate: |attribution| by offset from the anchor "
            f"({len(top_coordinate)} anchor(s), {top_coordinate['guid'].nunique()} recording(s))",
            fontsize=figures.FONT_SMALL,
        )
        ax.set_xlabel(COEFFICIENT_LAG_AXIS_LABEL)
        ax.set_ylabel("|attribution| (nats, symlog)")
        symlog_axis(ax, np.nanmean(np.abs(source), axis=0), np.nanmean(np.abs(target), axis=0), headroom=1.0)
        ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
        figures.style_axes(ax)
    figures.caveat_note(figure, f"{caveat}. {GROUP_DELAY_CAVEAT}")
    return figure


def build_null_figure(null: pd.DataFrame, *, caveat: str) -> Any:
    r"""The null decomposition of $K_t$: clock, content, and the entry jump, by class.

    Left: for the divergence under the source-null baseline, per class, the mean over recordings
    of the readout at the input, at the exact null (the clock part), the attributed content
    (the integrated-gradient sum) and the entry jump. Right: under the all-zero baseline, the
    split of the attribution between the target and the source streams.

    Args:
        null: The null table, one row per (class, readout).
        caveat: The sentence printed under the figure.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1, 2, height_per_row=3.2, width=12.0)
    for col, (baseline, columns, title) in enumerate((
        (BASELINE_SOURCE_NULL,
         [("value_input_mean", "K_t at the input"), ("value_baseline_mean", "K_t at the null (clock)"),
          ("attributed_mean", "attributed content (IG sum)"), ("entry_jump_mean", "entry jump")],
         "divergence, source-null baseline: clock and content"),
        (BASELINE_ALL_ZERO,
         [("target_total_mean", "target attribution"), ("source_total_mean", "source attribution"),
          ("entry_jump_mean", "entry jump")],
         "divergence, all-zero baseline: target vs source"),
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
        ax.set_ylabel("nats per anchor, mean over recordings")
        ax.legend(fontsize=figures.FONT_TINY, loc="upper right")
        figures.style_axes(ax)
    figures.caveat_note(figure, caveat)
    return figure


#: The rows of a recording's attribution trace figure: the lag-aligned attribution of the
#: divergence and the model's own lag readout as heatmaps on one lag axis, the agreement and the
#: source share as lines, and the readout values behind them.
TRACE_PANELS: Tuple[Any, ...] = (
    traces.HeatmapPanel("kld_attribution_lag_map", "|source attribution of $K_t$| by lag (source-null baseline, log)", "",
                        log=True, lag_axis=True),
    traces.HeatmapPanel("pred_gap_attribution_lag_map",
                        "|source attribution of the forecast gap| by lag (source-null baseline, log)", "", log=True,
                        lag_axis=True),
    traces.HeatmapPanel("model_lag_map", "The model's own lag readout (log)", "", log=True, lag_axis=True),
    traces.LinePanel(("kld_lag_corr", "pred_gap_lag_corr"),
                     "Correlation of the lag-aligned attribution with the model's lag readout", "Pearson $r$",
                     labels=("$K_t$", "forecast gap")),
    traces.LinePanel(("kld_source_total", "pred_gap_source_total"), "Source attribution totals",
                     "readout units", labels=("$K_t$", "forecast gap")),
    traces.LinePanel(("kld_value_input", "kld_value_baseline"), "$K_t$ at the input and at the exact null",
                     "nats", labels=("input", "null")),
    traces.LinePanel(("pred_gap_value_input", "pred_gap_value_baseline"),
                     "Forecast gap at the input and at the exact null", "nats", labels=("input", "null")),
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
    "ANALYSIS_DIRNAME", "ANCHORS_PER_SEGMENT", "ATTENTION_CELL", "ATTRIBUTION_CAVEAT",
    "AnchorReadout", "AttributionBatch", "BANDS_FILENAME", "BAND_FIGURE", "BASELINES",
    "BASELINE_ALL_ZERO", "BASELINE_ENTRY_FRACTION", "BASELINE_SOURCE_NULL", "CAP_NAME",
    "CellBinding", "DEFAULT_SEGMENTS", "DRAW_SEED_OFFSET", "IG_INTERNAL_BATCH_SIZE", "IG_STEPS",
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
    "unsigned_log_norm", "variant_label", "symlog_legend",
]
