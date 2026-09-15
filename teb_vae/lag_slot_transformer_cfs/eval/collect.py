r"""The lag-residual cell's own collection pass: one forward, every arm, the family's tables.

This is the pass :data:`~teb_vae.lag_slot_transformer_cfs.eval.binding.LAG_RESIDUAL_BINDING`
declares in place of the shared one, and it exists because the shared pass is written against the
lag-attention forward: it reads the attention weights, the per-lag divergence map and the
source-null arm, and it reduces a latent produced at every stored step. This architecture computes
none of those, and every latent tensor it produces is indexed by decoded anchor. The one thing it
must not do is fabricate the tensors the shared readout asks for -- a proposal norm reported under
an attention name is a per-lag attribution that does not exist -- so it produces the shared tables
from its own forward instead: the same identity columns, the same column names wherever the
quantity is the same one, and no name at all where it is not.

**What one pass does.** Per batch it runs **one** dense forward with the per-lag proposals retained,
builds every intervened arm the checkpoint admits -- the band suppressions from the cached
proposals, the two source replacements and the cross-recording permutation from a re-run forward,
a stranger's prior for the prior-shuffle control -- and scores all of them in a single draw loop
under one $\epsilon^{(k)}$ per replicate. Then it aggregates per recording, bootstraps over
recordings, and assembles the results block the summary carries.

**Two tables from one set of tensors.** The shared per-sample and per-anchor tables carry the
family's column names (``mc_nll_full_block``, ``pred_gap`` on the training path, the three trivial
baselines, the latent diagnostics, the per-channel vectors) so every table-driven analysis of the
family runs on them unchanged. This cell's own protocol -- the per-recording table its acceptance
pass reads, the arm intervals, the band margins, the resolved axes -- keeps its own names
(``nll_<arm>`` for the marginalised score of every arm, ``pred_gap`` for the marginalised gap) in
the results block and in ``per_recording.csv``. The two conventions are stated here once: on the
shared table ``pred_gap`` is the single-draw training-path gap and ``mc_pred_gap`` the
marginalised one; on this cell's own table ``pred_gap`` is the marginalised gap, because that is
the number its protocol has always read.

**Every arm of this architecture is scorable through this one pass**, which is the condition that
makes a comparison between two of them a comparison of models rather than of scoring paths. What
differs between arms is which interventions exist to run, and each one that does not is skipped by
name with its reason rather than reported as a margin of zero -- a zero is the finding that an
intervention changed nothing, and the two must not read the same. The run's ``arm`` block records
the mechanism, the parameter split and what the band margins mean.

**Why every arm is scored in one loop.** Common random numbers are what make a margin a difference
of predictions. Two arms with identical latent parameters produce bitwise identical scores here, so
the selectors-off arm's margin is exactly zero and the empty-band arm's is exactly zero -- and each
of those zeros is then evidence that the intervention path and the forward path are one computation
rather than two that nearly agree.

**Three axes beyond the block score, all under the same draws.** Every scored arm is additionally
resolved by horizon step and by stored target block, so a source that informs the first predicted
step and not the last, or the scattering block and not the phase-harmonic one, is a statement the
run can make. And on the recommended fusion the lag axis is read at every single lag: cheaply in
latent space for the whole split, and as a predictive margin on a capped number of segments, where
each lag's proposals are removed alone and the arm is scored under the same draws as every other.
The single-lag margins are read **after** the band and joint removals, which is the order the
design fixes: a single-lag peak read off a window whose joint removal does nothing is noise.

**Every margin travels with a paired interval.** Two arms scored on the same recordings under the
same draws differ per recording, so the interval on their margin is the interval of those
differences, drawn once over recordings, and not two overlapping intervals of the two arms.

**What a reader must not take from the output.** The band margins do not decompose the gap and are
not normalised to; the qualification travels in the artifact beside them. A band with no available
source is recorded as missing rather than as a measured zero. And a positive source margin is
necessary rather than sufficient: the whole reason this architecture exists is a checkpoint whose
source-conditioned branch was a **worse** predictive density than its target-only one, so the
headline is the gap against the internal base, and its sign is the first thing to read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval.stats import bootstrap_ci
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import collect as shared_collect
from teb_vae.lag_attn_cfs.eval.metrics import (
    BASELINE_LOGVAR,
    DENSE_ANCHOR_GEOMETRY,
    NORMALISED_UNIT,
    Aggregate,
    baseline_forecasts,
    batch_field,
    batch_guids,
    batch_recordings,
    branch_channel_scores,
    build_verdicts,
    calibration_report,
    calibration_sums,
    expected_anchors_per_sample,
    horizon_block_sums,
    horizon_residual_sums,
    latent_health,
    masked_raw_error_sums,
    model_inputs,
    target_block_membership,
)
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC, SATURATION_FRAC
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_slot_transformer_cfs.eval import lag_metrics
from teb_vae.lag_slot_transformer_cfs.eval.binding import (
    ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE,
    EXCLUDED_ANALYSES,
    MODEL_KIND,
    UNREGISTERED_ANALYSES,
)
from teb_vae.lag_slot_transformer_cfs.eval.predictive import (
    DEFAULT_COVERAGE_LEVELS,
    calibration_census,
    draw_concentration,
    finish_calibration,
    matched_predictive_scores,
    merge_calibration,
)
from teb_vae.lag_slot_transformer_cfs.nets import controls
from teb_vae.lag_slot_transformer_cfs.nets.core import pathway_parameter_counts

#: The two branches whose mixture calibration is accumulated. Both, because a calibration statement
#: about the source-conditioned branch alone cannot say whether the source improved it or whether
#: the observation model was already miscalibrated without it.
CALIBRATED_BRANCHES: Tuple[str, ...] = ("base", "full")

#: Prefix that marks a scored arm as a suppression of one lag band, so the summary's assembly can
#: tell the interventions apart from the two matched branches without a second list to keep aligned.
SUPPRESSION_PREFIX = lag_metrics.SUPPRESSION_PREFIX

#: Prefix that marks a scored arm as the suppression of one single lag. These arms are scored in the
#: same draw loop as every other on the segments the profile cap admits, and they reach the summary
#: as one per-lag curve rather than as one column each: a table with one column per candidate lag
#: would be unreadable and would put the fine profile on the same footing as the declared bands,
#: which the design reads first.
LAG_ARM_PREFIX = "lag:"

#: The cap name under ``eval_config.caps`` that bounds how many segments the single-lag predictive
#: profile is scored on. Absent means the profile is skipped and recorded as such, matching the
#: family's rule that retention is opt-in; the committed delta sets it.
LAG_PROFILE_CAP = "lag_profile"

#: The arm the cross-recording pairing produces, and the arm the same pairing produces on the
#: PRIOR: a stranger's prior latent decoded as the base forecast. The second is the family's
#: prior-shuffle control -- the check that the prior carries recording-specific target state
#: rather than a recording-independent average -- and it is scored here under the same pairing as
#: the source permutation so "a stranger's source" and "a stranger's prior" name the same
#: stranger.
PERMUTE_ARM = "permute"
PRIOR_SHUFFLE_ARM = "base_shuffled_mu"

#: The source controls, and the column each one's margin is taken from. One place, read by the
#: block that reports them, so the control names in the summary and the columns in the table
#: cannot drift apart. The prior-shuffle arm is deliberately not among them: it intervenes on the
#: prior, not on the source, and its reading is the family's ``prior_carries_target_state``
#: verdict rather than a source margin.
CONTROL_COLUMNS: Mapping[str, str] = {
    "silence": "nll_silence",
    "replace_zeros": "nll_replace:zeros",
    "replace_constant": "nll_replace:constant",
    PERMUTE_ARM: f"nll_{PERMUTE_ARM}",
}

#: The two stored target blocks, in the order the kept channel axis carries them: the scattering
#: coefficients first, the phase-harmonic coefficients after. Named as the configuration names
#: their weights, so a reader can match a block to the weight it was trained under.
TARGET_BLOCKS: Tuple[str, ...] = ("st", "ph")

#: Which of this cell's scored arms the SHARED table reports, and under which of the family's
#: names. The matched pair and the two pairing controls are the same quantity under both
#: conventions -- a marginalised block score of that arm -- so they travel under the family's
#: names as well as this cell's; every other arm is this cell's own and stays under its own.
SHARED_ARM_NAMES: Mapping[str, str] = {
    "base": "base",
    "full": "full",
    PERMUTE_ARM: "shuffled",
    PRIOR_SHUFFLE_ARM: "base_shuffled_mu",
}

#: The warm-up tertile names, in the order ``warm_tertile_id`` numbers them.
TERTILE_NAMES: Tuple[str, ...] = ("lo", "mid", "hi")

#: The sentence the two ``pred_gap`` conventions are stated in, written into the results block so
#: a reader of one table beside the other has it without this module.
PRED_GAP_CONVENTIONS = (
    "Two tables, two conventions. On the family's per_sample.csv, pred_gap is the single-draw "
    "training-path gap and mc_pred_gap the Monte Carlo marginalised one, as on every cell of the "
    "family; the marginalised score of an arm is mc_nll_<arm>_block there. On this cell's own "
    "per_recording.csv and in its arm_scores, band and control blocks, nll_<arm> is the "
    "marginalised score of that arm and pred_gap is the marginalised gap nll_base - nll_full, "
    "because that is the number this cell's acceptance protocol reads. The two tables carry the "
    "same numbers under the two names; neither is a second estimate of the other."
)


@dataclass
class SharedReadout:
    r"""One batch's per-sample readouts under the family's names, as the shared sink reads them.

    Duck-typed against :class:`~teb_vae.lag_attn_cfs.eval.metrics.BatchReadout`: the shared
    :class:`~teb_vae.lag_attn_cfs.eval.collect.Collector` reads these attributes by name and
    carries whichever of the family's vector readouts the readout has. Four of them exist here --
    the per-dimension divergence and the three channel vectors -- and none of the lag or
    attention vectors does, which is what keeps an attention-shaped name out of the sidecar.

    Attributes:
        guids: Recording identifier per sample.
        columns: Named per-sample values, each a $(B,)$ tensor, under the family's column names.
        n_anchors: Contributing anchors per sample, $(B,)$.
        kld_per_dim: Per-sample per-coordinate divergence over the scored anchors, $(B, d_z)$.
        gap_per_channel: The training-path forecast gap resolved per surviving target channel,
            $(B, C_{\mathrm{keep}})$; sums over channels to the sample's ``pred_gap``.
        sq_error_per_channel_base: Per-channel masked mean squared error of the target-only
            branch, $(B, C_{\mathrm{keep}})$.
        sq_error_per_channel_full: The same for the source-conditioned branch.
        per_anchor: The per-anchor tensors the anchor table is built from, each $(B, A)$, with
            ``anchor_index`` and ``contributing`` among them.
        retained: Whole tensors a retention plan asked to keep, by the family's names.
        horizon_sums: Residual, log-variance and block-score sums resolved by horizon step, per
            branch, each $(H,)$.
        calibration_sums: The observation model's calibration accumulators over the full
            branch's scored coefficients; empty under ``'mse'``.
        n_control_pairs: Samples paired against another recording by the permutation control.
        n_same_recording_pairs: How many of those pairs landed inside their own recording.
        per_anchor_vectors: Empty. No per-anchor lag map exists on this architecture.
    """

    guids: List[str]
    columns: Dict[str, torch.Tensor]
    n_anchors: torch.Tensor
    kld_per_dim: torch.Tensor
    gap_per_channel: torch.Tensor
    sq_error_per_channel_base: torch.Tensor
    sq_error_per_channel_full: torch.Tensor
    per_anchor: Dict[str, torch.Tensor]
    retained: Dict[str, torch.Tensor]
    horizon_sums: Dict[str, torch.Tensor]
    calibration_sums: Dict[str, torch.Tensor]
    n_control_pairs: int = 0
    n_same_recording_pairs: int = 0
    per_anchor_vectors: Dict[str, torch.Tensor] = field(default_factory=dict)


# =============================================================================
# Per-sample reductions
# =============================================================================
def _per_sample_mean(per_anchor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average a per-anchor quantity within each sample, over its weighted anchors.

    Args:
        per_anchor: $(B, A)$ values.
        weights: $(B, A)$ non-negative weights; zero anchors drop out entirely.

    Returns:
        $(B,)$ per-sample means, in float64.
    """
    weights = weights.to(torch.float64)
    return (per_anchor.to(torch.float64) * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _per_sample_vector_mean(per_anchor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average a per-anchor *vector* quantity within each sample, over its weighted anchors.

    Args:
        per_anchor: $(B, A, C)$ values.
        weights: $(B, A)$ non-negative weights.

    Returns:
        $(B, C)$ per-sample means, in float64.
    """
    weights = weights.to(torch.float64)
    numerator = (per_anchor.to(torch.float64) * weights.unsqueeze(-1)).sum(dim=1)
    return numerator / weights.sum(dim=1).clamp_min(1.0).unsqueeze(-1)


def _per_sample_element_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Average a per-coefficient quantity within each sample, over its scored coefficients.

    The denominator is $C_{\mathrm{keep}} \sum_{a,\tau} m_{a,\tau}$ -- the scored coefficient
    count -- and not $H \cdot C_{\mathrm{keep}}$, which over-states it on any anchor with masked
    forecast steps.

    Args:
        values: $(B, A, H, C_{\mathrm{keep}})$ values, or anything broadcastable to that shape.
        mask: The forecast mask $(B, A, H)$.

    Returns:
        $(B,)$ per-sample means, in float64.
    """
    weights = mask[..., None].to(torch.float64)
    channels = float(values.shape[-1])
    denominator = (mask.to(torch.float64).sum(dim=(1, 2)) * channels).clamp_min(1.0)
    return (values.to(torch.float64) * weights).sum(dim=(1, 2, 3)) / denominator


def kept_block_split(model: Any) -> Optional[int]:
    r"""Where the second stored target block begins on the **kept** channel axis.

    The keep-index is positional into the declared stream, and the survivors are not contiguous,
    so the boundary is counted rather than taken from the declared split: it is how many kept
    channels lie below the declared boundary. ``None`` on a model whose kept axis holds one block
    only, where a split would leave the other block's score as a row of zeros.

    Args:
        model: The rebuilt net.

    Returns:
        The kept-position boundary, or ``None``.
    """
    declared_split = getattr(model, "TARGET_BLOCK_SPLIT", None)
    if declared_split is None:
        return None
    gate = getattr(model, "target_gate", None)
    declared = (
        torch.arange(int(model.c_y)) if gate is None else gate.keep_index.detach().cpu()
    )
    first = int((declared < int(declared_split)).sum())
    if not 0 < first < int(declared.numel()):
        return None
    return first


# =============================================================================
# The arms
# =============================================================================
def intervened_branches(
    model: Any,
    outputs: Mapping[str, torch.Tensor],
    streams: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    recordings: Optional[Sequence[str]],
    perm_generator: Optional[torch.Generator],
    single_lags: bool = False,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    r"""Build every scored arm's latent parameters from one matched forward.

    On the recommended fusion the suppression arms come from the cached per-lag updates, so no
    encoder and no head runs again for them. On a comparator arm whose fusion normalises over the
    lag axis there is no such term to subtract -- removing a lag removes it from the denominator too
    -- so its band arms hand the forward a selector and let the distribution re-form over what is
    left. Both reduce to the same two reference identities, an empty band reproducing the matched
    arm and a full band reproducing the prior, which is what makes each arm's own margins
    measurements.

    The replacement and permutation arms re-run the forward under a substituted stream on every arm,
    because a proposal is a function of the source values it was given and no cached set can answer
    a question about different ones; every other argument of that forward is the matched one, and
    the prior coming back bitwise unchanged is the evidence that nothing but the source moved. Both
    are skipped whole on the arm that withholds its source values, where neither is an intervention
    at all. The permutation's pairing is reused for the prior-shuffle control, which decodes a
    stranger's **prior** under the same pairing.

    The single-lag arms are the band arms at the finest partition and come from the same cached
    subtraction. They exist only on the local fusion: on the normalised aggregation each would be
    one more forward per lag, and the reading would be a different quantity in any case.

    Args:
        model: The net.
        outputs: The matched forward's dict, taken with ``return_proposals=True``.
        streams: ``(y_st, y_ph, u_stream)`` as the matched forward received them.
        masks: ``{band: (L,) bool}`` removal masks, including the two reference arms.
        recordings: One recording identifier per sample, or ``None`` when the batch carries none.
        perm_generator: Generator for the cross-recording pairing.
        single_lags: Whether to add one suppression arm per candidate lag.

    Returns:
        ``(branches, record)``. *branches* maps an arm name to its ``(mu, logvar)``; *record* holds
        the control bookkeeping -- how many pairs the permutation arm drew, how many landed inside
        their own recording, the permuted forward's own per-anchor divergence, and why an arm was
        skipped when it was.
    """
    y_st, y_ph, u_stream = streams
    branches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "base": (outputs["mu_prior"], outputs["logvar_prior"]),
        "full": (outputs["mu_post"], outputs["logvar_post"]),
    }
    record: Dict[str, Any] = {
        "n_control_pairs": 0,
        "n_same_recording_pairs": 0,
        "skipped": {},
        "batch_without_partner": False,
        # The permuted forward's own divergence per anchor, for the shared table's
        # ``source_conditioned_kl_shuffled_raw``; ``None`` where the arm did not run.
        "shuffled_kld_per_anchor": None,
    }

    # A target-only checkpoint has no source pathway to intervene on, and that is a legitimate
    # subject for this pass rather than an error: the independently trained target-only predictor
    # is the external reference the candidate's gap has to be read against, and it has to be scored
    # through the SAME estimator, the same anchors, the same mask and the same draws, or the
    # comparison is between two scoring paths rather than two models.
    #
    # Every arm is skipped by name rather than silently absent, so a reader of two summaries side by
    # side sees why one has fewer columns.
    if getattr(model, "source_disabled", False):
        record["skipped"] = {
            arm: (
                "this checkpoint was built with no source pathway, so there is nothing to "
                "intervene on: the full distribution IS the prior and the divergence is exactly "
                "zero by construction."
            )
            for arm in ("suppress", "silence", "replace", PERMUTE_ARM, "lag_profile")
        }
        return branches, record

    batch_size, n_anchors = u_stream.shape[0], outputs["mu_prior"].shape[1]
    phase, stride = DENSE_ANCHOR_GEOMETRY
    local_fusion = str(getattr(model, "lag_fusion", "local")) == "local"

    # Two suppression paths, and which one runs is a property of the arm rather than a choice.
    # A local sum has a per-lag term to subtract, so its band arms are recomputed from the cached
    # proposals and nothing runs again. A normalised aggregation has none -- removing a lag from it
    # removes it from the denominator too -- so its band arms hand the forward a selector and let
    # the distribution re-form over what is left. Both reduce to the same two reference identities:
    # an empty band reproduces the matched arm and a full band reproduces the prior.
    for band, mask in masks.items():
        if local_fusion:
            suppressed = controls.suppressed_parameters(model, outputs, mask)
        else:
            suppressed = model(
                y_st,
                y_ph,
                u_stream,
                anchor_phase=phase,
                anchor_stride=stride,
                selector=controls.band_selector(
                    mask, batch_size, n_anchors, dtype=u_stream.dtype
                ),
            )
        branches[f"{SUPPRESSION_PREFIX}{band}"] = (
            suppressed["mu_post"],
            suppressed["logvar_post"],
        )

    # The finest partition of the same intervention, one lag at a time, from the same cached
    # subtraction. Read after the bands rather than instead of them.
    if single_lags:
        if local_fusion and "mean_proposals" in outputs:
            n_lags = int(model.n_lags)
            for lag in range(n_lags):
                removed = torch.zeros(n_lags, dtype=torch.bool, device=u_stream.device)
                removed[lag] = True
                single = controls.suppressed_parameters(model, outputs, removed)
                branches[f"{LAG_ARM_PREFIX}{lag}"] = (single["mu_post"], single["logvar_post"])
        else:
            record["skipped"]["lag_profile"] = (
                "this checkpoint aggregates its lags with a normalised distribution, which "
                "produces no per-lag update to remove alone; a single-lag profile there would be "
                "one re-run forward per lag and a different quantity from the local fusion's "
                "under the same name. The declared bands are the lag readout on this arm."
            )

    # The selectors-off arm. It verifies the equality invariant and nothing else, and it is scored
    # rather than asserted so that the invariant is checked on the same numbers a margin is read
    # from rather than on a separate forward nobody compares against.
    silent = torch.zeros(
        batch_size, n_anchors, model.n_lags, device=u_stream.device, dtype=u_stream.dtype
    )
    silenced = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride, selector=silent)
    branches["silence"] = (silenced["mu_post"], silenced["logvar_post"])

    # Two of the three replacement laws, and the third is absent on purpose rather than forgotten.
    # ``mask_only`` asks what the head does with the availability announcement and no value; on the
    # recommended encoder the value coordinate IS the coefficient, so a zeroed stream already
    # leaves exactly the mask and the arm is the ``zeros`` arm under a second name. Scoring it
    # would put one measurement on the table twice. It becomes a distinct question under the
    # scalar lift, where the lift of a zero is a learned constant -- and there the control refuses
    # rather than reporting the zeros arm, so it cannot be added here without an encoding-level
    # hook that drops the value coordinates.
    #
    # Both arms are skipped whole on the capacity control, where the source values are withheld at
    # the encoder and every substitution is therefore the same computation as the matched forward.
    # Reported as skipped rather than as two margins of exactly zero: a zero margin is the finding
    # that replacing the values changed nothing, and on this arm they were never read.
    if bool(getattr(model, "source_values_withheld", False)):
        record["skipped"]["replace"] = (
            "this checkpoint withholds the source values at the encoder, so a substituted stream "
            "reaches the fusion as the same all-zero value coordinates the matched forward did. "
            "The margin would be exactly zero for a reason about the arm rather than about the "
            "source, which is the reading a reported zero would invite."
        )
    else:
        for mode in ("zeros", "constant"):
            substituted = controls.replaced_source_stream(
                u_stream, mode, scalar_lift=bool(model.source_scalar_lift)
            )
            replaced = model(
                y_st, y_ph, substituted, anchor_phase=phase, anchor_stride=stride
            )
            branches[f"replace:{mode}"] = (replaced["mu_post"], replaced["logvar_post"])

    if bool(getattr(model, "source_values_withheld", False)):
        # For the reason the replacement arms are skipped above, and one step stronger: the
        # availability announcement is a function of stored position and the resolved warm-up
        # alone, so pairing a recording with another one changes nothing this arm reads. A
        # recording-specificity margin of zero here would say the model is not recording specific,
        # which is true of it by construction rather than by measurement.
        record["skipped"][PERMUTE_ARM] = (
            "this checkpoint withholds the source values, and what it does read -- lag identity "
            "and the availability announcement -- does not vary with which recording the stream "
            "came from, so a cross-recording pairing is not an intervention on this arm."
        )
    elif recordings is not None and controls.groups_can_derange(recordings):
        index = controls.cross_recording_index(
            recordings, generator=perm_generator, device=u_stream.device
        )
        permuted = model(
            y_st, y_ph, u_stream[index], anchor_phase=phase, anchor_stride=stride
        )
        branches[PERMUTE_ARM] = (permuted["mu_post"], permuted["logvar_post"])
        # The same pairing on the PRIOR: a stranger's target-only latent decoded as the base
        # forecast. The prior is target-only, so its parameters are simply re-indexed rather than
        # re-run, and the pairing being the permutation's is what makes the two controls name one
        # stranger.
        branches[PRIOR_SHUFFLE_ARM] = (
            outputs["mu_prior"][index],
            outputs["logvar_prior"][index],
        )
        record["shuffled_kld_per_anchor"] = permuted["kld_per_anchor"]
        record["n_control_pairs"] = int(u_stream.shape[0])
        record["n_same_recording_pairs"] = controls.same_recording_pairs(recordings, index)
    else:
        # A property of this batch, not of the arm: flagged so the pass counts it apart from the
        # arms that cannot run on the checkpoint at all, and only reports the arm as skipped when
        # no batch of the split could pair.
        record["batch_without_partner"] = True
        record["skipped"][PERMUTE_ARM] = (
            "no cross-recording pairing exists in this batch: it carries no recording "
            "identifiers, or one recording holds more than half of it. Counted rather than "
            "silently dropped -- a control that stopped being a control looks like one that works."
        )
    return branches, record


def mean_decoded_scores(
    model: Any,
    branches: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    persistence: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    r"""Score branches decoded at their latent **mean**, with no draw.

    The family's deterministic plug-in estimator, written here rather than imported because the
    shared one gathers the latent at the decoded anchors and every latent tensor of this
    architecture is already anchor-indexed. Each branch's $\mu$ is decoded once under the decoder's
    own predictive variance; two runs of a checkpoint agree bitwise, and two branches with equal
    means score identically.

    Args:
        model: The net, for its shared decoder.
        branches: ``{name: (mu, logvar)}``, each $(B, A, d_z)$. Only ``mu`` is read.
        target: The gathered forecast target $(B, A, H, C_{\mathrm{keep}})$.
        mask: The forecast mask $(B, A, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        persistence: The matched forward's own persistence input, or ``None``.

    Returns:
        ``{name: (B, A) per-anchor block score}``.
    """
    scores: Dict[str, torch.Tensor] = {}
    for name, (mu, _logvar) in branches.items():
        forecast_mu, forecast_logvar = model.decoder(mu, persistence=persistence)
        block, _contributing = masked_raw_block_per_anchor(
            forecast_mu, target, mask, likelihood=likelihood, logvar=forecast_logvar
        )
        scores[name] = block
    return scores


# =============================================================================
# One batch
# =============================================================================
def score_batch(
    task: Any,
    batch: Any,
    *,
    masks: Mapping[str, torch.Tensor],
    num_samples: int,
    mc_generator: Optional[torch.Generator],
    perm_generator: Optional[torch.Generator],
    lag_profile: bool = False,
    block_split: Optional[int] = None,
    retain: Sequence[str] = (),
) -> Tuple[Dict[str, Any], SharedReadout]:
    r"""Run one batch: one forward, every arm, one draw loop, both readouts.

    Args:
        task: The loaded task wrapping the net.
        batch: A batch already on the model's device.
        masks: ``{band: (L,) bool}`` removal masks, built once for the pass so every batch
            suppresses the same lags -- a per-batch rebuild would be the same arithmetic with one
            more place for the band definition to differ.
        num_samples: Monte Carlo draws $K$.
        mc_generator: Generator for the latent draws.
        perm_generator: Generator for the cross-recording pairing.
        lag_profile: Whether this batch also scores one suppression arm per candidate lag, under
            the same draws as every other arm.
        block_split: The kept-position boundary between the two stored target blocks, for the
            block-resolved scores; ``None`` resolves the horizon axis alone.
        retain: Forward-output names to carry back whole on the shared readout, plus ``'target'``,
            ``'weight'``, ``'up_raw'`` and ``'fhr_raw'`` for the tensors the forward does not
            return. Empty by default: a retained forecast set is several megabytes per sample.

    Returns:
        ``(record, readout)``. *record* is this cell's own: per-sample scores per arm under this
        cell's names, the per-sample curves the resolved axes are built from, the guids that
        weight them, and the exposure, cancellation, latent-profile and mixture-calibration
        **sums** the pass accumulates -- sums rather than means, so the reported figure is a mean
        over the whole split rather than a mean of per-batch means. *readout* carries the same
        forward under the family's names, for the shared tables.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, batch)

    phase, stride = DENSE_ANCHOR_GEOMETRY
    outputs = model(
        y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride, return_proposals=True
    )
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    # The forecast clock's pooled validity, so every readout here is scored under exactly the mask
    # the training objective used. The identity object under the stored clock.
    mask, coverage = forecast_mask(
        model.scored_weight(weight),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchors,
        anchor_valid=anchor_valid,
    )

    batch_size = int(y_st.shape[0])
    recordings = batch_recordings(batch, batch_size)
    branches, control_record = intervened_branches(
        model,
        outputs,
        (y_st, y_ph, u_stream),
        masks,
        recordings=recordings,
        perm_generator=perm_generator,
        single_lags=lag_profile,
    )

    # Every arm but the single-lag ones is resolved by horizon step and by block: a single-lag arm
    # is scored for one margin and nothing else, and resolving each of them would multiply the
    # cheapest part of the loop by the lag count for a curve nobody reads.
    resolved = tuple(name for name in branches if not name.startswith(LAG_ARM_PREFIX))
    persistence = outputs.get("persistence")
    scored = matched_predictive_scores(
        model,
        branches,
        target,
        mask,
        likelihood=likelihood,
        num_samples=num_samples,
        generator=mc_generator,
        # The forward's own tensor rather than a second gather of the same target: it is
        # target-only, identical across arms and draws, and both decoder calls of every arm receive
        # this same object.
        persistence=persistence,
        calibrate=CALIBRATED_BRANCHES,
        resolve=resolved,
        block_split=block_split,
    )
    contributing = scored["base"].contributing
    weights = contributing.to(torch.float64)
    per_sample_anchors = weights.sum(dim=1)

    def per_sample(values: torch.Tensor) -> torch.Tensor:
        """Average a per-anchor quantity within each sample, over its scored anchors.

        Works on a scalar per anchor and on a vector per anchor alike: the anchor axis is the
        second one and every trailing axis is carried through.
        """
        shaped = weights.view(weights.shape + (1,) * (values.dim() - 2))
        summed = (values.to(torch.float64) * shaped).sum(dim=1)
        count = per_sample_anchors.clamp_min(1.0).view((-1,) + (1,) * (summed.dim() - 1))
        return summed / count

    # ---------------------------------------------------------------------
    # This cell's own record
    # ---------------------------------------------------------------------
    columns: Dict[str, torch.Tensor] = {
        f"nll_{name}": per_sample(scored[name].marginal) for name in resolved
    }
    columns["pred_gap"] = columns["nll_base"] - columns["nll_full"]
    columns["kld_per_anchor"] = per_sample(outputs["kld_per_anchor"])
    columns["draw_concentration_full"] = per_sample(
        draw_concentration(scored["full"].per_draw)
    )

    # The per-sample curves: one vector per sample per resolved quantity. The horizon and block
    # axes for every resolved arm, the gap on both, and -- on the batches the profile cap admits --
    # the single-lag margins as one vector over lags.
    curves: Dict[str, torch.Tensor] = {}
    for name in resolved:
        branch = scored[name]
        if branch.per_horizon is not None:
            curves[f"nll_{name}_by_horizon"] = per_sample(branch.per_horizon)
        if branch.per_block is not None:
            curves[f"nll_{name}_by_block"] = per_sample(branch.per_block)
    for axis in ("by_horizon", "by_block"):
        if f"nll_base_{axis}" in curves:
            curves[f"pred_gap_{axis}"] = curves[f"nll_base_{axis}"] - curves[f"nll_full_{axis}"]
    lag_columns = [name for name in branches if name.startswith(LAG_ARM_PREFIX)]
    if lag_columns:
        matched = columns["nll_full"]
        curves["lag_margin"] = torch.stack(
            [per_sample(scored[name].marginal) - matched for name in lag_columns], dim=1
        )

    # Absent on a target-only checkpoint, where no window was ever gathered. Reported as an empty
    # exposure rather than as counts of zero: nothing was measured, which is a different statement
    # from a source that was available and carried nothing.
    channel_mask = outputs.get("source_channel_mask")
    exposure: Dict[str, torch.Tensor] = {}
    latent_profile: Dict[str, torch.Tensor] = {}
    if channel_mask is not None:
        exposure = lag_metrics.lag_exposure(
            outputs["lag_valid"], channel_mask, contributing
        )
        exposure["channels_per_source_channel"] = lag_metrics.channel_exposure(
            channel_mask, contributing
        )
    # The latent per-lag profile costs one pass of cheap arithmetic over the cached proposals, so
    # it is taken on every batch of the split rather than on the capped ones the predictive profile
    # is scored on. Present only where per-lag updates exist to remove.
    if "mean_proposals" in outputs and str(getattr(model, "lag_fusion", "local")) == "local":
        latent_profile = lag_metrics.per_lag_latent_totals(model, outputs, contributing)
    record = {
        "guids": batch_guids(batch, batch_size),
        "columns": {name: value.cpu() for name, value in columns.items()},
        "curves": {name: value.cpu() for name, value in curves.items()},
        "n_anchors": per_sample_anchors.cpu(),
        "exposure": {name: value.cpu() for name, value in exposure.items()},
        "latent_profile": {name: value.cpu() for name, value in latent_profile.items()},
        "cancellation": lag_metrics.cancellation_totals(outputs, contributing),
        "calibration": {
            name: calibration_census(
                scored[name].cdf_sum, mask, levels=DEFAULT_COVERAGE_LEVELS
            )
            for name in CALIBRATED_BRANCHES
            if scored[name].cdf_sum is not None
        },
        "control": control_record,
    }

    # ---------------------------------------------------------------------
    # The family's readout, from the same tensors
    # ---------------------------------------------------------------------
    readout = shared_readout(
        model,
        batch,
        outputs,
        scored=scored,
        branches=branches,
        target=target,
        target_features=target_features,
        weight=weight,
        mask=mask,
        coverage=coverage,
        contributing=contributing,
        likelihood=likelihood,
        persistence=persistence,
        control_record=control_record,
        guids=record["guids"],
        retain=retain,
    )
    return record, readout


def shared_readout(
    model: Any,
    batch: Any,
    outputs: Mapping[str, torch.Tensor],
    *,
    scored: Mapping[str, Any],
    branches: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    target_features: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    coverage: torch.Tensor,
    contributing: torch.Tensor,
    likelihood: str,
    persistence: Optional[torch.Tensor],
    control_record: Mapping[str, Any],
    guids: List[str],
    retain: Sequence[str],
) -> SharedReadout:
    r"""Reduce one scored batch to the family's per-sample and per-anchor readouts.

    Every column here is one the family's table-driven analyses read, under the name they read it
    by, and every one is the same quantity it is on the lag-attentive cells: a marginalised block
    score, a training-path block score, a trivial baseline scored through the same loss and mask,
    a latent diagnostic over the scored anchors. What the family computes from an attention is
    absent rather than substituted.

    The **KL support is the contributing-anchor set.** On the lag-attentive cells the divergence
    is produced at every stored step and reduced over a dense support; here it is produced at the
    decoded anchors only, so its support is the anchors the forecast scored, and
    ``source_conditioned_kl_raw`` averages the per-anchor divergence over exactly those. The
    per-anchor table's ``kld_per_t`` column is that same divergence at each anchor, so the two
    recombine as the family's sanity check requires.

    The **saturation fractions** are measured against this cell's own bounds. The prior mean is
    read against ``mu_scale`` as everywhere in the family; the residual is the bounded mean update
    $a_t$ read against $a_{\max}$, in prior standard deviations, because that is the bound this
    architecture places on the source correction and there is no separate posterior mean bound to
    read it against.

    Args:
        model: The net.
        batch: The batch, for the raw traces the retention keeps.
        outputs: The matched forward's dict.
        scored: The scored branches, from :func:`~predictive.matched_predictive_scores`.
        branches: ``{arm: (mu, logvar)}`` as scored.
        target: The gathered forecast target $(B, A, H, C_{\mathrm{keep}})$.
        target_features: The declared-width target stream $(B, T, c_y)$, for the baselines.
        weight: The decimated validity signal $(B, T)$.
        mask: The forecast mask $(B, A, H)$.
        coverage: Each anchor's coverage $(B, A)$.
        contributing: The $0/1$ scored-anchor indicator $(B, A)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        persistence: The matched forward's persistence input, or ``None``.
        control_record: The pairing bookkeeping :func:`intervened_branches` returned.
        guids: Recording identifier per sample.
        retain: Forward-output names to carry back whole.

    Returns:
        The readout the shared sink consumes.
    """
    batch_size = int(target.shape[0])
    device, dtype = target.device, outputs["mu_prior"].dtype
    weights = contributing.to(torch.float64)
    nan_column = torch.full((batch_size,), float("nan"), dtype=torch.float64)

    # The training-path scores: the forward's own decoded forecasts under the one shared draw,
    # exactly as the objective sees them.
    training_full, _ = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood=likelihood, logvar=outputs["logvar_full"]
    )
    training_base, _ = masked_raw_block_per_anchor(
        outputs["mu_base"], target, mask, likelihood=likelihood, logvar=outputs["logvar_base"]
    )
    # The mean-decoded pair, and the two pairing controls where they ran.
    shared_branches = {
        family: branches[own] for own, family in SHARED_ARM_NAMES.items() if own in branches
    }
    mean_scores = mean_decoded_scores(
        model, shared_branches, target, mask, likelihood=likelihood, persistence=persistence
    )

    mu_prior, logvar_prior = outputs["mu_prior"], outputs["logvar_prior"]
    mu_post, logvar_post = outputs["mu_post"], outputs["logvar_post"]
    delta_mu = mu_post - mu_prior
    kld_per_anchor = outputs["kld_per_anchor"]

    columns: Dict[str, torch.Tensor] = {
        "nll_base_block": _per_sample_mean(training_base, contributing),
        "nll_full_block": _per_sample_mean(training_full, contributing),
        "source_conditioned_kl_raw": _per_sample_mean(kld_per_anchor, contributing),
        # The prior's scale rate, on the same support and in the same nats-per-anchor units as
        # the divergence above it, so the two are addable exactly as they are in the objective.
        "prior_rate": _per_sample_mean(
            (0.5 * (logvar_prior.exp() - 1.0 - logvar_prior)).sum(dim=-1), contributing
        ),
        "mu_prior_rms": _per_sample_mean((mu_prior**2).mean(dim=-1), contributing).sqrt(),
        # Unrooted, and rooted beside it: averaging finished per-segment roots across a recording
        # is biased low by Jensen, so the square is what the aggregation chain carries.
        "delta_mu_sq": _per_sample_mean((delta_mu**2).mean(dim=-1), contributing),
        # Summed over $d_z$ before the mean: the size of the belief shift per anchor.
        "mu_post_prior_gap_sq": _per_sample_mean((delta_mu**2).sum(dim=-1), contributing),
    }
    columns["delta_mu_rms"] = columns["delta_mu_sq"].sqrt()
    columns["pred_gap"] = columns["nll_base_block"] - columns["nll_full_block"]
    for own, family in SHARED_ARM_NAMES.items():
        # NaN rather than absent on a batch where a pairing control did not run: the shared table
        # holds one row per segment with every column, and a column absent from one batch would
        # leave the table ragged. NaN is the family's own representation of "not measured".
        columns[f"mc_nll_{family}_block"] = (
            _per_sample_mean(scored[own].marginal, contributing) if own in scored else nan_column
        )
        columns[f"mean_nll_{family}_block"] = (
            _per_sample_mean(mean_scores[family], contributing)
            if family in mean_scores else nan_column
        )
    columns["mc_pred_gap"] = columns["mc_nll_base_block"] - columns["mc_nll_full_block"]
    columns["mean_pred_gap"] = columns["mean_nll_base_block"] - columns["mean_nll_full_block"]
    shuffled_kld = control_record.get("shuffled_kld_per_anchor")
    columns["source_conditioned_kl_shuffled_raw"] = (
        nan_column if shuffled_kld is None else _per_sample_mean(shuffled_kld, contributing)
    )

    # The three trivial forecasts, scored through the model's own loss function with the
    # identical mask at the identical anchors, so a skill score is a comparison of predictors
    # rather than of scoring conventions.
    baselines = baseline_forecasts(target_features, weight, model, outputs["anchor_index"])
    baseline_logvar = torch.full((), BASELINE_LOGVAR, dtype=target.dtype, device=device)
    for name, baseline_mu in baselines.items():
        baseline_block, _ = masked_raw_block_per_anchor(
            baseline_mu, target, mask, likelihood=likelihood, logvar=baseline_logvar
        )
        columns[f"nll_{name}_block"] = _per_sample_mean(baseline_block, contributing)

    # Point-forecast error, in the loader's z units and per scored coefficient rather than per
    # anchor. The squares stay unrooted here for the Jensen reason above.
    point_forecasts: Dict[str, torch.Tensor] = {
        "base": outputs["mu_base"], "full": outputs["mu_full"], **baselines
    }
    for name, point_mu in point_forecasts.items():
        sums = masked_raw_error_sums(point_mu, target, mask)
        scored_count = sums["n_coefficients"].to(torch.float64).clamp_min(1.0)
        columns[f"sq_error_{name}"] = sums["sum_sq"].to(torch.float64) / scored_count
        if name in ("base", "full"):
            columns[f"abs_error_{name}"] = sums["sum_abs"].to(torch.float64) / scored_count
            columns[f"signed_error_{name}"] = (
                sums["sum_residual"].to(torch.float64) / scored_count
            )
    columns["forecast_difference_sq"] = _per_sample_element_mean(
        (outputs["mu_full"] - outputs["mu_base"]) ** 2, mask
    )

    # The channel axis: both branches' masked scores per anchor and per surviving channel, and
    # every reduction of that one pair -- the per-channel gap vector, the two stored-block gaps
    # and the three warm-up tertile gaps -- so all of them are partial sums of the training-path
    # ``pred_gap`` they are read beside.
    base_by_channel = branch_channel_scores(
        outputs["mu_base"], outputs["logvar_base"], target, mask, likelihood=likelihood
    )
    full_by_channel = branch_channel_scores(
        outputs["mu_full"], outputs["logvar_full"], target, mask, likelihood=likelihood
    )
    gap_by_anchor_channel = base_by_channel - full_by_channel  # (B, A, C_keep)
    gap_per_channel = _per_sample_vector_mean(gap_by_anchor_channel, contributing)
    first_block = target_block_membership(model, device, gap_per_channel.dtype)
    columns["pred_gap_st"] = (gap_per_channel * first_block).sum(dim=1)
    columns["pred_gap_ph"] = (gap_per_channel * (1.0 - first_block)).sum(dim=1)
    tertile = model.warm_tertile_id.to(device)
    for group, name in enumerate(TERTILE_NAMES):
        selector = (tertile == group).to(gap_per_channel.dtype)
        columns[f"pred_gap_warm_{name}"] = (gap_per_channel * selector).sum(dim=1)

    # The two geometry guards. The anchor count is read off ``anchor_valid`` so a batch whose
    # validity is entirely zero still reports which anchors the forward built; the warm fraction
    # is resolved at construction and echoed.
    columns["anchors_per_sample"] = outputs["anchor_valid"].to(torch.float64).sum(dim=1)
    columns["target_warm_frac"] = torch.full(
        (batch_size,), float(model.target_warm_frac), dtype=torch.float64
    )

    # The bound-variance diagnostics. A prior variance pinned on its lower clamp inflates the
    # divergence by an arbitrary factor while the decoder variances look healthy, so both ends of
    # both clamps are counted separately rather than inferred from a mean.
    lo, hi = float(model.logvar_clamp[0]), float(model.logvar_clamp[1])
    floor_threshold = lo + LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    ceil_threshold = hi - LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    columns["mean_logvar_prior"] = _per_sample_mean(logvar_prior.mean(dim=-1), contributing)
    columns["mean_logvar_post"] = _per_sample_mean(logvar_post.mean(dim=-1), contributing)
    columns["logvar_prior_floor_frac"] = _per_sample_mean(
        (logvar_prior <= floor_threshold).to(dtype).mean(dim=-1), contributing
    )
    columns["mean_logvar_full"] = _per_sample_element_mean(outputs["logvar_full"], mask)
    columns["logvar_full_floor_frac"] = _per_sample_element_mean(
        (outputs["logvar_full"] <= floor_threshold).to(dtype), mask
    )
    columns["logvar_full_ceil_frac"] = _per_sample_element_mean(
        (outputs["logvar_full"] >= ceil_threshold).to(dtype), mask
    )
    # The saturation fractions in both framings: a flat mean over every element, and the same
    # over the scored anchors. The residual is the bounded update against its own bound, in
    # prior standard deviations, which is where this architecture bounds the source correction.
    saturated_mu = (mu_prior.abs() >= SATURATION_FRAC * model.mu_scale).to(dtype)
    saturated_update = (
        outputs["update_mean"].abs() >= SATURATION_FRAC * model.residual_mu_scale
    ).to(dtype)
    columns["mu_prior_sat_frac_raw"] = saturated_mu.mean(dim=(1, 2)).to(torch.float64)
    columns["mu_prior_sat_frac_masked"] = _per_sample_mean(saturated_mu.mean(dim=-1), contributing)
    columns["delta_mu_sat_frac_raw"] = saturated_update.mean(dim=(1, 2)).to(torch.float64)
    columns["delta_mu_sat_frac_masked"] = _per_sample_mean(
        saturated_update.mean(dim=-1), contributing
    )

    # The per-anchor table: the same numbers before the within-sample reduction, so the table
    # recombines into the per-sample one exactly, keyed on the forward's own anchor index.
    per_anchor: Dict[str, torch.Tensor] = {
        "anchor_index": outputs["anchor_index"],
        "contributing": contributing,
        "coverage": coverage,
        "kld_per_t": kld_per_anchor,
        "nll_base_block": training_base,
        "nll_full_block": training_full,
        "pred_gap": training_base - training_full,
        "mc_nll_base_block": scored["base"].marginal,
        "mc_nll_full_block": scored["full"].marginal,
        "mc_pred_gap": scored["base"].marginal - scored["full"].marginal,
        "mean_nll_base_block": mean_scores["base"],
        "mean_nll_full_block": mean_scores["full"],
        "mean_pred_gap": mean_scores["base"] - mean_scores["full"],
    }
    for group, name in enumerate(TERTILE_NAMES):
        selector = (tertile == group).to(gap_by_anchor_channel.dtype)
        per_anchor[f"pred_gap_warm_{name}"] = (gap_by_anchor_channel * selector).sum(dim=2)

    # The observation model's calibration census over the full branch's scored coefficients, as
    # the family accumulates it. Empty under ``'mse'``, where the log-variance head is untrained.
    calibration = (
        calibration_sums(
            outputs["mu_full"], outputs["logvar_full"], target, mask,
            logvar_clamp=model.logvar_clamp,
        )
        if likelihood == "gaussian_nll"
        else {}
    )

    retained: Dict[str, torch.Tensor] = {}
    if retain:
        available: Dict[str, torch.Tensor] = dict(outputs)
        available["target"] = target
        available["weight"] = weight
        up_raw = batch_field(batch, "up")
        if isinstance(up_raw, torch.Tensor):
            available["up_raw"] = up_raw
        fhr_raw = batch_field(batch, "fhr")
        if isinstance(fhr_raw, torch.Tensor):
            available["fhr_raw"] = fhr_raw
        retained = {name: available[name] for name in retain if name in available}

    return SharedReadout(
        guids=list(guids),
        columns=columns,
        n_anchors=weights.sum(dim=1),
        kld_per_dim=_per_sample_vector_mean(outputs["kld_per_anchor_dim"], contributing),
        gap_per_channel=gap_per_channel,
        sq_error_per_channel_base=(
            ((outputs["mu_base"] - target) ** 2 * mask[..., None]).sum(dim=(1, 2))
            / mask.sum(dim=(1, 2)).clamp_min(1.0)[:, None]
        ),
        sq_error_per_channel_full=(
            ((outputs["mu_full"] - target) ** 2 * mask[..., None]).sum(dim=(1, 2))
            / mask.sum(dim=(1, 2)).clamp_min(1.0)[:, None]
        ),
        per_anchor=per_anchor,
        retained=retained,
        horizon_sums={
            f"{branch}_{statistic}": value
            for branch, (branch_mu, branch_logvar) in (
                ("base", (outputs["mu_base"], outputs["logvar_base"])),
                ("full", (outputs["mu_full"], outputs["logvar_full"])),
            )
            for statistic, value in {
                **horizon_residual_sums(branch_mu, branch_logvar, target, mask),
                **horizon_block_sums(
                    branch_mu, branch_logvar, target, mask, likelihood=likelihood
                ),
            }.items()
        },
        calibration_sums=calibration,
        n_control_pairs=int(control_record["n_control_pairs"]),
        n_same_recording_pairs=int(control_record["n_same_recording_pairs"]),
    )


# =============================================================================
# This cell's aggregation
# =============================================================================
def aggregate_by_recording(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    """Average each column and each curve within a recording, then hand back the per-recording values.

    Not a flat mean over anchors or over segments. Consecutive anchors' forecast windows overlap in
    all but one of their steps at the dense geometry, so anchors within a recording are very far
    from independent; averaging over them and reporting the result as if it had that many samples
    behind it overstates the precision of every number and weights the headline toward whichever
    recordings happen to be longest.

    A segment that scored no anchor measured nothing and is excluded rather than averaged in as a
    zero: its per-sample mean divides by a denominator clamped to one, so an empty numerator reads
    as exactly zero and would pull a summed-block score of hundreds of nats toward it.

    The segment and anchor counts come back from this same walk rather than from a second one, and
    that is not tidiness: they are the weights the exported table is read with, and a count computed
    under a different empty-segment rule from the mean beside it would describe a different
    population from the number it qualifies.

    Columns and curves are both averaged per name, over the segments that carried that name,
    because neither is present on every batch. The single-lag margin curve exists only on the
    segments the profile cap admitted, and the permute column only on the batches where a
    cross-recording pairing existed at all -- a trailing batch that one recording holds more than
    half of records the arm as skipped and carries no such column. A recording none of whose
    segments carried a name is absent from it rather than present as a zero, and a recording some
    of whose segments did is averaged over those segments only. The count in the exposure table
    is the all-columns segment count, which is the population every column present on every batch
    is read with; the permute margin's own paired count is reported beside its interval.

    Args:
        records: The per-batch records :func:`score_batch` returned.

    Returns:
        ``({recording: {column: value}}, {recording: {'n_segments', 'n_scored_anchors'}},
        {curve: {recording: vector}})``.
    """
    sums: Dict[str, Dict[str, float]] = {}
    column_counts: Dict[str, Dict[str, int]] = {}
    counts: Dict[str, int] = {}
    anchors: Dict[str, float] = {}
    curve_sums: Dict[str, Dict[str, np.ndarray]] = {}
    curve_counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        names = list(record["columns"])
        curve_names = list(record.get("curves") or {})
        for position, guid in enumerate(record["guids"]):
            scored = float(record["n_anchors"][position])
            if scored <= 0.0:
                continue
            bucket = sums.setdefault(guid, {})
            per_column_count = column_counts.setdefault(guid, {})
            counts[guid] = counts.get(guid, 0) + 1
            anchors[guid] = anchors.get(guid, 0.0) + scored
            for name in names:
                bucket[name] = bucket.get(name, 0.0) + float(record["columns"][name][position])
                per_column_count[name] = per_column_count.get(name, 0) + 1
            for name in curve_names:
                vector = np.asarray(record["curves"][name][position], dtype=np.float64)
                per_curve = curve_sums.setdefault(name, {})
                per_count = curve_counts.setdefault(name, {})
                per_curve[guid] = per_curve.get(guid, 0.0) + vector
                per_count[guid] = per_count.get(guid, 0) + 1
    per_recording = {
        guid: {name: total / column_counts[guid][name] for name, total in bucket.items()}
        for guid, bucket in sums.items()
    }
    exposure = {
        guid: {
            "n_segments": float(counts[guid]),
            "n_scored_anchors": float(anchors[guid]),
        }
        for guid in per_recording
    }
    curves = {
        name: {guid: total / float(curve_counts[name][guid]) for guid, total in per_curve.items()}
        for name, per_curve in curve_sums.items()
    }
    return per_recording, exposure, curves


def arm_scores_block(
    per_recording: Mapping[str, Mapping[str, float]],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Every scored column's equal-recording mean, each with a recording-level bootstrap interval.

    Both summaries the design asks for are emitted separately rather than one standing for the
    other: the equal-recording mean is what the interval is built over, and the anchor-weighted mean
    is what an operator comparing against a training log expects. They differ whenever recordings
    contribute unequal numbers of anchors, which is always.

    Args:
        per_recording: The per-recording column means.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling, so the interval is reproducible from the summary alone.

    Returns:
        ``{column: bootstrap record}`` for every scored column. A column some recordings lack --
        the permute arm, on recordings every segment of which fell in a batch without a partner --
        is bootstrapped over the recordings that hold it, and the record's own ``n`` says how many.
    """
    if not per_recording:
        return {}
    names = sorted({name for values in per_recording.values() for name in values})
    return {
        name: bootstrap_ci(
            [values[name] for values in per_recording.values() if name in values],
            resamples=int(resamples),
            seed=int(seed),
        )
        for name in names
    }


def paired_margin_block(
    per_recording: Mapping[str, Mapping[str, float]],
    columns: Mapping[str, str],
    *,
    matched: str = "nll_full",
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """The paired interval of every named arm's margin against the matched branch.

    Args:
        per_recording: The per-recording column means.
        columns: ``{name: column}`` naming the intervened arms' columns.
        matched: The column the margins are taken against.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        ``{name: paired record}``, with :data:`~lag_metrics.MISSING` where an arm did not run.
    """
    return {
        name: lag_metrics.paired_margin(
            per_recording, column, matched, resamples=int(resamples), seed=int(seed)
        )
        for name, column in columns.items()
    }


def curve_difference(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """The per-recording difference of two curves, on the recordings both hold.

    Args:
        left: ``{recording: vector}``.
        right: ``{recording: vector}``.

    Returns:
        ``{recording: left - right}``.
    """
    return {guid: left[guid] - right[guid] for guid in left if guid in right}


def resolved_axis_block(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    *,
    axis: str,
    positions: Sequence[Any],
    unit: str,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    r"""One resolved axis of the summary: every arm's curve, the gap, and every margin, paired.

    Every entry is a :func:`~lag_metrics.bootstrap_curve` record -- ``point``, ``lo`` and ``hi``
    as lists over the axis, under one resampling of the recordings -- so a reader can follow an
    interval from one position to the next. The margins are intervals of per-recording
    **differences**, as the scalar margins are.

    Args:
        curves: The per-recording curves, keyed ``nll_<arm>_<axis>`` and ``pred_gap_<axis>``.
        axis: ``'by_horizon'`` or ``'by_block'``.
        positions: The axis labels, one per position.
        unit: The unit every value is in, recorded beside them.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        The block, empty when no curve of this axis was collected.
    """
    suffix = f"_{axis}"
    arms = {
        name[len("nll_"):-len(suffix)]: rows
        for name, rows in curves.items()
        if name.startswith("nll_") and name.endswith(suffix)
    }
    if not arms:
        return {}

    def interval(rows: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        """Bootstrap one per-recording curve."""
        return lag_metrics.bootstrap_curve(rows, resamples=int(resamples), seed=int(seed))

    matched = arms.get("full", {})
    return {
        "positions": list(positions),
        "unit": unit,
        "nll": {arm: interval(rows) for arm, rows in arms.items()},
        "pred_gap": interval(curves.get(f"pred_gap{suffix}", {})),
        "band_margins": {
            arm[len(SUPPRESSION_PREFIX):]: interval(curve_difference(rows, matched))
            for arm, rows in arms.items()
            if arm.startswith(SUPPRESSION_PREFIX)
        },
        "control_margins": {
            name: interval(curve_difference(arms[column[len("nll_"):]], matched))
            for name, column in CONTROL_COLUMNS.items()
            if column[len("nll_"):] in arms
        },
        "note": (
            "Each position is the marginal mixture of that subset's own likelihood factors under "
            "the shared draws, so the positions do not sum to the joint block score and are not "
            "made to. Intervals are percentile bootstraps over recordings under one resampling for "
            "every position; the margins are intervals of per-recording differences against the "
            "matched full branch."
        ),
    }


def arm_record(model: Any) -> Dict[str, Any]:
    """Which arm produced this summary, what it costs, and what its band margins mean.

    Written into every run rather than inferred by a reader from which columns are missing. A
    mechanism-separating comparison is a set of runs that each changed one declared thing, and the
    only way that reads as a comparison afterwards is if each run says which thing it changed.

    The parameter split is here for the same reason. A fusion that holds more weights than the one
    it is compared against can win by capacity, so the budgets belong beside the margins rather
    than in a launch log.

    The suppression note is the one entry that is not bookkeeping. Removing a lag from an explicit
    sum leaves every other term standing; removing it from a normalised distribution redistributes
    its weight over the survivors. Both answer "what does this model do when it cannot read these
    lags", and the two numbers are not on one scale.

    Args:
        model: The rebuilt net.

    Returns:
        The arm block of the summary.
    """
    source_disabled = bool(getattr(model, "source_disabled", False))
    fusion = str(getattr(model, "lag_fusion", "local"))
    return {
        # The architecture's own identity, beside the arm leaves: it is what says a checkpoint's
        # source fusion is not the lag-attentive one whatever the two constructors accept.
        "model_kind": MODEL_KIND,
        "source_disabled": source_disabled,
        "source_stem": None if source_disabled else str(getattr(model, "source_stem", "pointwise")),
        "lag_fusion": None if source_disabled else fusion,
        "mean_only_residual": bool(getattr(model, "mean_only_residual", False)),
        "source_scalar_lift": bool(getattr(model, "source_scalar_lift", False)),
        "source_values_withheld": bool(getattr(model, "source_values_withheld", False)),
        "lag_summation_scale": float(getattr(model, "lag_scale", 1.0)),
        "parameters": pathway_parameter_counts(model),
        "note": (
            "a target-only checkpoint: the full distribution is the prior, the divergence is "
            "exactly zero, and no source intervention exists to run"
            if source_disabled
            else "a source-conditioned checkpoint: every intervened arm below was scored under "
            "the same draws as the matched one"
        ),
        "suppression_semantics": (
            None
            if source_disabled
            else (
                "a band's proposals are removed from the explicit sum and every surviving "
                "proposal keeps the weight it had"
                if fusion == "local"
                else "a band is removed from the normalised distribution over lags, so the "
                "surviving weights grow to fill it. That is what an attention head cannot read a "
                "lag means, and it makes this margin a different quantity from a local fusion's "
                "under the same band name: compare each against its own matched branch, never "
                "against the other arm's margin."
            )
        ),
    }


def lag_axis_record(model: Any) -> Dict[str, Any]:
    """The axis every per-lag figure of this run is drawn against, read off the model.

    Stored-coefficient time: lag $\\ell$ names the source coefficient stored $\\ell$ steps before
    the anchor, and the seconds beside it are that many stored steps. The delay term is the
    model's own causal input delay in stored steps, which is what a lag axis compensates for, and
    nothing else is added to it.

    Args:
        model: The rebuilt net.

    Returns:
        ``{'n_lags', 'seconds_per_step', 'delay_steps', 'clock'}``.
    """
    return {
        "n_lags": int(getattr(model, "n_lags", 0) or 0),
        "seconds_per_step": float(SECONDS_PER_STEP),
        "delay_steps": int(model.source_delay_steps),
        "clock": "stored-coefficient time: stored steps back from the anchor",
    }


def lag_profile_block(
    *,
    latent_totals: Optional[Mapping[str, torch.Tensor]],
    exposure_totals: Optional[Mapping[str, torch.Tensor]],
    lag_margins: Mapping[str, np.ndarray],
    n_profiled_segments: int,
    cap: Optional[int],
    skipped: Optional[str],
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """The per-lag readouts: the latent profile over the split, the predictive one over the cap.

    Args:
        latent_totals: The accumulated :func:`~lag_metrics.per_lag_latent_totals`, or ``None``.
        exposure_totals: The accumulated per-lag exposure, or ``None``.
        lag_margins: ``{recording: (L,) margins}`` from the segments the cap admitted.
        n_profiled_segments: How many segments the predictive profile was scored on.
        cap: The configured cap, or ``None`` when the profile was not asked for.
        skipped: Why the predictive profile did not run on this arm, or ``None``.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        The block.
    """
    latent = lag_metrics.lag_profile_summary(
        latent_totals, None if not exposure_totals else exposure_totals["anchors_per_lag"]
    )
    if skipped is not None:
        predictive: Dict[str, Any] = {"status": "SKIPPED", "detail": skipped}
    elif cap is None:
        predictive = {
            "status": "NOT_REQUESTED",
            "detail": (
                f"eval_config.caps.{LAG_PROFILE_CAP} is absent, so no segment was scored with "
                f"single lags removed. Set it to the number of segments to profile."
            ),
        }
    elif not lag_margins:
        predictive = {
            "status": "EMPTY",
            "detail": "no segment reached the single-lag arms; nothing was scored.",
            "n_segments": int(n_profiled_segments),
            "cap": int(cap),
        }
    else:
        predictive = {
            "status": "READ",
            "n_segments": int(n_profiled_segments),
            "n_recordings": len(lag_margins),
            "cap": int(cap),
            "margin_nats": lag_metrics.bootstrap_curve(
                lag_margins, resamples=int(resamples), seed=int(seed)
            ),
            "detail": (
                "each lag's margin is the score with that lag's proposals alone removed less the "
                "matched score, in nats per anchor, paired per recording under the shared draws "
                "over the first segments of the split up to the cap. A positive value means the "
                "fitted model predicts worse without that lag."
            ),
        }
    return {
        "latent": latent,
        "predictive": predictive,
        "note": (
            "The latent profile removes one lag at a time from the cached proposals and reports "
            "the proposal norm, the shift of the bounded mean update and the drop of the "
            "divergence, each averaged over the scored anchors the lag was live at. None is an "
            "allocation over lags: an exactly zero-sum reallocation changes every one of them at "
            "every lag while changing no prediction. The single-lag margins are read after the "
            "band and joint removals, never instead of them."
        ),
    }


def _block_channel_counts(model: Any, block_split: Optional[int]) -> Dict[str, int]:
    """How many kept channels each stored target block holds, read off the model.

    Args:
        model: The rebuilt net.
        block_split: The kept-position boundary, or ``None``.

    Returns:
        ``{block: count}``, empty when no split exists.
    """
    if block_split is None:
        return {}
    width = int(getattr(model, "decoder_out_channels", 0) or 0)
    return {TARGET_BLOCKS[0]: int(block_split), TARGET_BLOCKS[1]: max(width - int(block_split), 0)}


def _margin(scores: Mapping[str, Any], arm: str) -> Any:
    """One arm's predictive margin against the matched full branch.

    Args:
        scores: The bootstrapped per-column block, keyed by column name.
        arm: The arm's name, as :func:`intervened_branches` keyed it.

    Returns:
        The margin in nats per anchor, or ``MISSING`` when the arm did not run.
    """
    matched = scores.get("nll_full", {}).get("point")
    intervened = scores.get(f"nll_{arm}", {}).get("point")
    if matched is None or intervened is None:
        return lag_metrics.MISSING
    return float(intervened) - float(matched)


def anchor_weighted(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """The same columns, weighted by each segment's scored-anchor count.

    Reported beside the equal-recording means rather than instead of them. The two differ whenever
    recordings contribute unequal numbers of anchors, which is always, and a summary carrying only
    one of them leaves a reader unable to tell a real effect from a length effect.

    Args:
        records: The per-batch records.

    Returns:
        ``{column: value}``, empty when nothing was scored. Each column is divided by the anchors
        of the batches that carried it, since the permute column is absent from a batch without a
        cross-recording partner; dividing it by every batch's anchors would shrink it toward zero.
    """
    totals: Dict[str, float] = {}
    denominators: Dict[str, float] = {}
    for record in records:
        anchors = record["n_anchors"].to(torch.float64)
        batch_anchors = float(anchors.sum())
        for name, values in record["columns"].items():
            totals[name] = totals.get(name, 0.0) + float(
                (values.to(torch.float64) * anchors).sum()
            )
            denominators[name] = denominators.get(name, 0.0) + batch_anchors
    return {
        name: total / denominators[name]
        for name, total in totals.items()
        if denominators[name] > 0.0
    }


# =============================================================================
# The family's readouts, off the shared table
# =============================================================================
def per_recording_chain(
    per_sample: pd.DataFrame, vectors: Mapping[str, np.ndarray]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, List[float]]]:
    r"""Reduce the shared per-sample table per recording, then across recordings.

    The family's aggregation chain on the family's table: each column is averaged within a
    recording over the segments that carry a finite value, then across recordings with equal
    weight, and the vector readouts travel the identical chain so a per-dimension divergence still
    sums to the scalar it decomposes. A segment the pass blanked -- no scored anchor, or a pairing
    control that had no partner in its batch -- is absent from the means rather than a zero.

    Args:
        per_sample: The shared per-sample table the sink assembled.
        vectors: The per-sample vector readouts, row-aligned with it.

    Returns:
        ``(per_recording, overall, overall_vectors)``: the per-recording column means keyed by
        recording, their mean across recordings, and the across-recording mean of each vector.
    """
    identity = set(shared_collect.IDENTITY_COLUMNS)
    value_columns = [
        name for name in per_sample.columns
        if name not in identity and pd.api.types.is_numeric_dtype(per_sample[name])
    ]
    if per_sample.empty or "guid" not in per_sample.columns or not value_columns:
        return {}, {}, {}
    grouped = per_sample.groupby("guid")[value_columns].mean()
    per_recording = {
        str(guid): {
            name: float(value)
            for name, value in row.items()
            if value is not None and np.isfinite(float(value))
        }
        for guid, row in grouped.iterrows()
    }
    overall = {
        name: float(value)
        for name, value in grouped.mean().items()
        if np.isfinite(float(value))
    }
    overall_vectors: Dict[str, List[float]] = {}
    guids = per_sample["guid"].astype(str).to_numpy()
    for name, rows in vectors.items():
        array = np.asarray(rows, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != len(per_sample):
            continue
        frame = pd.DataFrame(array)
        frame["guid"] = guids
        overall_vectors[name] = [
            float(value) for value in frame.groupby("guid").mean().mean(axis=0).to_numpy()
        ]
    return per_recording, overall, overall_vectors


def family_results(
    model: Any,
    *,
    per_sample: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    calibration_totals: Mapping[str, torch.Tensor],
    eval_config: Mapping[str, Any],
    num_samples: int,
    likelihood: str,
    n_batches: int,
    control_pairs: int,
    same_recording_pairs: int,
    batches_without_partner: int,
    segments_without_partner: int,
) -> Dict[str, Any]:
    r"""The results blocks the family's headline, sanity block and analyses read.

    Under the same keys the shared pass writes them -- ``readouts``, ``latent_health``,
    ``calibration``, ``controls``, ``verdicts`` and the population counts -- and built from the
    shared table rather than from this cell's own record, so the number the headline quotes is the
    number the table carries. The verdicts are the family's own registry, decided by the family's
    own function over the same overall means: the availability-clock criterion is INCONCLUSIVE on
    this architecture, which computes no source-null arm, and every other criterion reads a
    quantity this pass produces.

    Args:
        model: The rebuilt net.
        per_sample: The shared per-sample table.
        vectors: The per-sample vector readouts.
        calibration_totals: The accumulated calibration sums of the full branch.
        eval_config: The validated settings, for the two verdict thresholds.
        num_samples: Monte Carlo draws $K$.
        likelihood: The objective's likelihood.
        n_batches: Batches scored.
        control_pairs: Samples the permutation control paired.
        same_recording_pairs: Pairs that landed inside their own recording.
        batches_without_partner: Batches on which the pairing controls could not run.
        segments_without_partner: Segments of those batches.

    Returns:
        The blocks, ready to merge into the results.
    """
    per_recording, overall, overall_vectors = per_recording_chain(per_sample, vectors)
    unscored = int((per_sample["n_anchors"] <= 0).sum()) if "n_anchors" in per_sample else 0
    aggregate = Aggregate(
        per_recording=per_recording,
        overall=overall,
        n_samples=int(len(per_sample)),
        n_samples_without_anchors=unscored,
        kld_per_dim=list(overall_vectors.get("kld_per_dim", [])),
        gap_per_channel=list(overall_vectors.get("gap_per_channel", [])),
        sq_error_per_channel_base=list(overall_vectors.get("sq_error_per_channel_base", [])),
        sq_error_per_channel_full=list(overall_vectors.get("sq_error_per_channel_full", [])),
    )
    clamp = model.logvar_clamp
    calibration = calibration_report(
        {name: value.detach().cpu() for name, value in calibration_totals.items()},
        logvar_clamp=clamp,
    )
    verdicts = build_verdicts(
        aggregate,
        prior_shuffle_min_nats=float(eval_config["prior_shuffle_min_nats"]),
        min_active_dims=int(eval_config["min_active_dims"]),
        clock_margin_min_nats=eval_config.get("clock_margin_min_nats"),
        expected_anchors_per_sample=expected_anchors_per_sample(model),
        logvar_margin=LOGVAR_FLOOR_MARGIN_FRAC * (float(clamp[1]) - float(clamp[0])),
        calibration=calibration,
    )
    return {
        "n_batches": int(n_batches),
        "n_batches_skipped_too_small": 0,
        "n_samples": aggregate.n_samples,
        "n_samples_without_anchors": aggregate.n_samples_without_anchors,
        "n_recordings": aggregate.n_recordings,
        "num_mc_samples": int(num_samples),
        "likelihood": likelihood,
        "anchor_geometry": {
            "anchor_phase": DENSE_ANCHOR_GEOMETRY[0],
            "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
            "anchors_per_sample_expected": expected_anchors_per_sample(model),
            "training_stride": int(model.anchor_stride),
            "target_kept_width": int(model.decoder_out_channels),
            "block_width": int(model.horizon) * int(model.decoder_out_channels),
        },
        "units": NORMALISED_UNIT,
        "readouts": dict(overall),
        "latent_health": latent_health(aggregate),
        "calibration": calibration,
        "controls": {
            "same_recording_pairing_rate": (
                (same_recording_pairs / control_pairs) if control_pairs else None
            ),
            "n_control_pairs": int(control_pairs),
            "n_same_recording_pairs": int(same_recording_pairs),
            # Nothing is excluded here: a batch without a cross-recording partner is scored on
            # every arm but the two pairing controls, whose columns it carries as NaN. The two
            # family keys are kept at zero so a reader comparing against a lag-attentive run
            # sees the policy difference stated rather than a missing key.
            "n_batches_excluded_no_cross_recording_partner": 0,
            "n_samples_excluded_no_cross_recording_partner": 0,
            "n_batches_without_cross_recording_partner": int(batches_without_partner),
            "n_samples_without_cross_recording_partner": int(segments_without_partner),
        },
        "verdicts": [verdict.as_dict() for verdict in verdicts],
    }


# =============================================================================
# The pass
# =============================================================================
def collect_tables(
    task: Any,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    num_samples: int,
    n_total: Optional[int] = None,
    max_batches: Optional[int] = None,
    perm_generator: Optional[torch.Generator] = None,
    mc_generator: Optional[torch.Generator] = None,
    delay_steps: int = 0,
) -> shared_collect.Collection:
    """Walk the split once and assemble the family's tables and this cell's results.

    The signature is :func:`teb_vae.lag_attn_cfs.eval.collect.collect_tables`'s, because that is
    the seam the runner calls through; ``delay_steps`` is accepted for it and is read off the model
    instead, since this architecture stamps its own.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader.
        eval_config: The validated ``eval_config`` block.
        num_samples: Monte Carlo draws $K$.
        n_total: Samples the pass will see, for the retention draw. Read from the loader's
            dataset when omitted.
        max_batches: Stop after this many batches, or ``None`` for the whole split.
        perm_generator: Generator for the cross-recording pairing.
        mc_generator: Generator for the latent draws.
        delay_steps: Accepted for the shared signature; unused.

    Returns:
        The collection, with ``results`` carrying this cell's blocks beside the family's.
    """
    del delay_steps
    model = task.orig_model
    device = task.device
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    seed = int(eval_config["seed"])
    resamples = int(eval_config["bootstrap_resamples"])
    bands = dict(eval_config.get("occlusion_bands") or {})
    caps = dict(eval_config.get("caps") or {})
    profile_cap: Optional[int] = caps.get(LAG_PROFILE_CAP)
    block_split = kept_block_split(model)
    masks = lag_metrics.band_masks(bands, model.n_lags, device=device)

    plan = shared_collect.RetentionPlan.build(
        eval_config.get("caps"),
        n_total=shared_collect._loader_length(loader) if n_total is None else int(n_total),
        seed=seed,
    )
    collector = shared_collect.Collector(
        plan,
        model=model,
        num_mc_samples=int(num_samples),
        device=next(model.parameters()).device,
    )
    retain = plan.tensor_names()

    records: List[Dict[str, Any]] = []
    exposure_totals: Optional[Dict[str, torch.Tensor]] = None
    latent_totals: Optional[Dict[str, torch.Tensor]] = None
    cancellation_totals: Optional[Dict[str, Dict[str, float]]] = None
    mixture_totals: Dict[str, Any] = {}
    calibration_totals: Dict[str, torch.Tensor] = {}
    control_pairs = same_recording = 0
    profiled_segments = 0
    skipped: Dict[str, str] = {}
    # Batches the permute arm could not run on because one recording held more than half of
    # them, typically the trailing partial batch of the split. Counted apart from ``skipped``:
    # that dict says an arm did not run on this checkpoint at all, and a control that ran on
    # every batch but the last must not be reported as if it had never run.
    batches_without_partner = segments_without_partner = 0
    was_training = task.training
    task.eval()
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= int(max_batches):
                    break
                batch = task.transfer_batch_to_device(batch, device, dataloader_idx=0)
                # The profile is scored on the first segments of the split up to the cap: the
                # loader is fixed-seed shuffled, so the leading segments are a draw over the
                # split rather than one shard's prefix.
                profile_this_batch = (
                    profile_cap is not None and profiled_segments < int(profile_cap)
                )
                record, readout = score_batch(
                    task,
                    batch,
                    masks=masks,
                    num_samples=num_samples,
                    mc_generator=mc_generator,
                    perm_generator=perm_generator,
                    lag_profile=profile_this_batch,
                    block_split=block_split,
                    retain=retain,
                )
                collector.observe(batch, readout)
                for name, value in readout.calibration_sums.items():
                    calibration_totals[name] = (
                        value if name not in calibration_totals
                        else calibration_totals[name] + value
                    )
                if "lag_margin" in record["curves"]:
                    profiled_segments += int(record["curves"]["lag_margin"].shape[0])
                exposure_totals = lag_metrics.merge_counts(exposure_totals, record["exposure"])
                if record["latent_profile"]:
                    latent_totals = lag_metrics.merge_counts(
                        latent_totals, record["latent_profile"]
                    )
                cancellation_totals = lag_metrics.merge_cancellation(
                    cancellation_totals, record["cancellation"]
                )
                for branch, block in record["calibration"].items():
                    mixture_totals[branch] = merge_calibration(mixture_totals.get(branch), block)
                control_pairs += int(record["control"]["n_control_pairs"])
                same_recording += int(record["control"]["n_same_recording_pairs"])
                batch_skips = dict(record["control"]["skipped"])
                if record["control"]["batch_without_partner"]:
                    batches_without_partner += 1
                    segments_without_partner += len(record["guids"])
                    batch_skips.pop(PERMUTE_ARM)
                skipped.update(batch_skips)
                # Released once the sink and the accumulators have had them: the per-anchor and
                # retained tensors are orders of magnitude larger than the per-sample columns.
                readout.per_anchor = {}
                readout.retained = {}
                records.append(record)
                logger.info(f"scored batch {index + 1}")
    finally:
        task.train(was_training)

    # Only when no batch at all could pair is the arm itself unrun; then the per-batch reason is
    # the run's reason, and the margin below comes back MISSING with it.
    if batches_without_partner and not any(
        f"nll_{PERMUTE_ARM}" in record["columns"] for record in records
    ):
        skipped[PERMUTE_ARM] = (
            "no batch of the split held a cross-recording pairing: every batch carried no "
            "recording identifiers, or had one recording holding more than half of it."
        )

    per_recording, per_recording_exposure, curves = aggregate_by_recording(records)
    arm_scores = arm_scores_block(per_recording, resamples=resamples, seed=seed)
    band_intervals = paired_margin_block(
        per_recording,
        {band: f"nll_{SUPPRESSION_PREFIX}{band}" for band in masks},
        resamples=resamples,
        seed=seed,
    )
    control_intervals = paired_margin_block(
        per_recording, CONTROL_COLUMNS, resamples=resamples, seed=seed
    )
    exposure = (
        {}
        if not exposure_totals
        else lag_metrics.band_exposure(masks, exposure_totals)
    )
    horizon = int(model.horizon)
    block_channels = _block_channel_counts(model, block_split)

    collection = collector.finish()
    results: Dict[str, Any] = {
        # The first thing a reader of two summaries has to know, because it decides what every
        # other block in the file can possibly say. A target-only arm's gap is exactly zero by
        # construction and its lag readouts are empty; a candidate's are neither.
        "arm": arm_record(model),
        "arm_scores": arm_scores,
        "anchor_weighted": anchor_weighted(records),
        "n_segments": sum(len(record["guids"]) for record in records),
        # The values every interval above was built from, one row per recording, under this
        # cell's own names. Carried out of here rather than rebuilt by a reader from the batches,
        # which no longer exist by the time the summary is written.
        "per_recording": {
            guid: {**columns, **per_recording_exposure[guid]}
            for guid, columns in per_recording.items()
        },
        "lag_readouts": lag_metrics.qualified_report(
            {
                "band_suppression": lag_metrics.band_suppression_block(
                    arm_scores, exposure, intervals=band_intervals
                ),
                # The declared edges, so the lag figures can shade the bands where they sit.
                "band_edges": {name: [int(lo), int(hi)] for name, (lo, hi) in bands.items()},
                "cancellation": lag_metrics.cancellation_summary(cancellation_totals or {}),
                # Plain lists rather than tensors, because the tables and the figures read this
                # block before the summary is serialised.
                "exposure": {
                    "per_band": exposure,
                    "per_lag_anchors": (
                        [] if not exposure_totals else exposure_totals["anchors_per_lag"].tolist()
                    ),
                    "per_lag_channels": (
                        [] if not exposure_totals else exposure_totals["channels_per_lag"].tolist()
                    ),
                    "per_source_channel": (
                        []
                        if not exposure_totals
                        else exposure_totals["channels_per_source_channel"].tolist()
                    ),
                },
                "lag_profile": lag_profile_block(
                    latent_totals=latent_totals,
                    exposure_totals=exposure_totals,
                    lag_margins=curves.get("lag_margin", {}),
                    n_profiled_segments=profiled_segments,
                    cap=profile_cap,
                    skipped=skipped.get("lag_profile"),
                    resamples=resamples,
                    seed=seed,
                ),
                "lag_axis": lag_axis_record(model),
            }
        ),
        "horizon_resolved": resolved_axis_block(
            curves,
            axis="by_horizon",
            positions=list(range(1, horizon + 1)),
            unit="nats per anchor per horizon step",
            resamples=resamples,
            seed=seed,
        ),
        "block_resolved": {
            **resolved_axis_block(
                curves,
                axis="by_block",
                positions=list(TARGET_BLOCKS),
                unit="nats per anchor, summed over the block's own channels and the horizon",
                resamples=resamples,
                seed=seed,
            ),
            "channels_per_block": block_channels,
        },
        "source_controls": {
            "silence_margin_nats": _margin(arm_scores, "silence"),
            "replace_zeros_margin_nats": _margin(arm_scores, "replace:zeros"),
            "replace_constant_margin_nats": _margin(arm_scores, "replace:constant"),
            "permute_margin_nats": _margin(arm_scores, PERMUTE_ARM),
            "silence_margin_interval": control_intervals["silence"],
            "replace_zeros_margin_interval": control_intervals["replace_zeros"],
            "replace_constant_margin_interval": control_intervals["replace_constant"],
            "permute_margin_interval": control_intervals[PERMUTE_ARM],
            "n_control_pairs": control_pairs,
            "n_same_recording_pairs": same_recording,
            # Batches the permute arm sat out because one recording held more than half of them.
            # Their segments are still scored on every other column; the permute margin's own
            # ``n_paired`` says how many recordings it was read over.
            "n_batches_without_partner": batches_without_partner,
            "n_segments_without_partner": segments_without_partner,
            "skipped": skipped,
            "note": (
                "Each margin is the arm's own predictive score less the matched full branch's, in "
                "nats per anchor, so a positive value means the fitted model predicts worse under "
                "the intervention; the interval beside it is a percentile bootstrap over the "
                "per-recording differences. The silence arm verifies the equality invariant and "
                "nothing else: its margin equals the base-minus-full gap by construction, because "
                "every selector off reproduces the prior exactly."
            ),
        },
        # The predictive mixture's calibration for both branches, accumulated in the draw loop.
        # Distinct from the family's ``calibration`` block below, which is the observation
        # model's census over the full branch's single forward; the two answer different
        # questions and both travel.
        "mixture_calibration": {
            branch: finish_calibration(mixture_totals.get(branch))
            for branch in CALIBRATED_BRANCHES
        },
        "draws": {
            "num_mc_samples": int(num_samples),
            "estimator": "log mean likelihood, common random numbers across arms",
            "note": (
                "The likelihood average is unbiased for the model likelihood; its negative "
                "logarithm is upward biased for the negative log density at finite K, and two "
                "arms' biases need not cancel. The effective draw count beside every gap is what "
                "says whether K was large enough for these anchors."
            ),
        },
        "conventions": PRED_GAP_CONVENTIONS,
        # Every analysis this architecture cannot produce, with the tensor it would have needed
        # and how it came to be absent: removed from the shared registry by the binding, or never
        # registered here at all. A reader of a directory with fewer columns than a sibling's
        # should not have to work out which.
        "excluded_analyses": ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE,
        "excluded_analyses_mechanism": {
            "removed_from_shared_registry": list(EXCLUDED_ANALYSES),
            "never_registered_here": list(UNREGISTERED_ANALYSES),
        },
    }
    results.update(
        family_results(
            model,
            per_sample=collection.per_sample,
            vectors=collection.vectors,
            calibration_totals=calibration_totals,
            eval_config=eval_config,
            num_samples=num_samples,
            likelihood=likelihood,
            n_batches=len(records),
            control_pairs=control_pairs,
            same_recording_pairs=same_recording,
            batches_without_partner=batches_without_partner,
            segments_without_partner=segments_without_partner,
        )
    )
    collection.results = results
    # The three facts no later analysis can recover for itself, as the shared pass records them:
    # the anchor and channel geometry, the loader's z-scoring, and the model's own bounds. The
    # bounds are this architecture's: the source correction is bounded in prior standard
    # deviations rather than by a posterior mean bound, and the record names the bound it has.
    collection.record["geometry"] = shared_collect.geometry_record(model)
    collection.record["normalization"] = shared_collect.normalization_record(loader)
    collection.record["bounds"] = bounds_record(model)
    collection.record["likelihood"] = likelihood
    shared_collect.check_per_anchor_key(collection.per_anchor)
    if len(collection.per_sample) != int(results["n_samples"]):
        raise ValueError(
            f"the per-sample table holds {len(collection.per_sample)} row(s) but the pass scored "
            f"{results['n_samples']} sample(s). One row per scored segment is what makes every "
            f"table-driven analysis agree with the headline readouts."
        )
    logger.info(
        f"collected {len(collection.per_sample)} sample row(s) and "
        f"{len(collection.per_anchor)} anchor row(s) from "
        f"{collection.record['n_recordings']} recording(s); "
        f"{collection.record['n_segments_excluded_zero_anchors']} segment(s) scored no anchors"
    )
    return collection


def bounds_record(model: Any) -> Dict[str, Any]:
    r"""This architecture's own bound conventions, for the analyses that read them offline.

    The log-variance clamp and its margin are the family's and are read by the same names. The
    source correction is bounded by $a_{\max}$ in **prior standard deviations** rather than by a
    posterior mean bound in latent units, so the record carries ``residual_mu_scale`` and
    ``residual_logsigma_scale`` and no ``delta_mu_scale``: a reader of the saturation fractions
    finds the bound they were measured against under the name the model gives it.

    Args:
        model: The rebuilt net.

    Returns:
        The clamp, the margin fraction and the margin it works out to, the prior mean bound, the
        two residual bounds, and the saturation fraction the counts are taken at.
    """
    lo, hi = float(model.logvar_clamp[0]), float(model.logvar_clamp[1])
    return {
        "logvar_clamp": [lo, hi],
        "logvar_margin_frac": float(LOGVAR_FLOOR_MARGIN_FRAC),
        "logvar_margin": float(LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)),
        "mu_scale": float(model.mu_scale),
        "residual_mu_scale": float(model.residual_mu_scale),
        "residual_logsigma_scale": float(model.residual_logsigma_scale),
        "saturation_frac": float(SATURATION_FRAC),
        "note": (
            "delta_mu_sat_frac_* count the bounded mean update a_t against residual_mu_scale, in "
            "prior standard deviations; this architecture bounds the source correction there and "
            "has no separate posterior mean bound."
        ),
    }


__all__ = [
    "CALIBRATED_BRANCHES",
    "CONTROL_COLUMNS",
    "LAG_ARM_PREFIX",
    "LAG_PROFILE_CAP",
    "PERMUTE_ARM",
    "PRED_GAP_CONVENTIONS",
    "PRIOR_SHUFFLE_ARM",
    "SHARED_ARM_NAMES",
    "SUPPRESSION_PREFIX",
    "TARGET_BLOCKS",
    "SharedReadout",
    "aggregate_by_recording",
    "anchor_weighted",
    "arm_record",
    "arm_scores_block",
    "bounds_record",
    "collect_tables",
    "curve_difference",
    "family_results",
    "intervened_branches",
    "kept_block_split",
    "lag_axis_record",
    "lag_profile_block",
    "mean_decoded_scores",
    "paired_margin_block",
    "per_recording_chain",
    "resolved_axis_block",
    "score_batch",
    "shared_readout",
]
