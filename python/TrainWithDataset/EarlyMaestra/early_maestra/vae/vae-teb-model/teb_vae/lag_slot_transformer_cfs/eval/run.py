r"""Score one lag-residual checkpoint: the matched gap, the lag readouts, the source controls.

Run from the repository root:

.. code-block:: bash

    python -m teb_vae.lag_slot_transformer_cfs.eval.run \
        --checkpoint output/<run>/model_checkpoints/<name>.ckpt

From an IDE's Run button, with no command line: fill in ``RUN_ARGS`` at the bottom of this file.
``--checkpoint`` is the only value the pass cannot proceed without, and it is enforced **after** the
merge rather than by ``required=True`` -- which fires before a launch dict is ever consulted and
would make the Run button unusable no matter what the dict said.

**What one pass does.** It rebuilds the net and its task from a checkpoint, merges the committed
override delta over that run's own resolved configuration, and walks the evaluation split once. Per
batch it runs **one** dense forward retaining whatever per-lag arrays the arm produces, builds every
intervened arm the checkpoint admits, and scores them all in a single draw loop under one
$\epsilon^{(k)}$ per replicate. Then it aggregates per recording, bootstraps over recordings, and
writes ``summary.json`` with the per-recording table beside it -- the values every interval in the
summary was built from, which a protocol reading several runs together needs and cannot recover
from an interval.

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

**The figures are drawn from the summary, never from the tensors.** Every figure the pass writes
is built from the blocks the summary carries and the tables beside it, so a figure and the number
it illustrates cannot disagree, and the same figures can be redrawn from a finished directory on a
box with no checkpoint and no shard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_slot_transformer_cfs/eval/run.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script -- which is what an IDE's Run button does -- this file's own directory goes
# on sys.path instead of the repository root, and every absolute import below fails before
# ``__main__`` is reached. Launched as a module it sets ``__package__`` and needs none of this,
# which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn.eval.report import SUMMARY_FILENAME, json_safe  # noqa: E402
from teb_vae.lag_attn.eval.stats import bootstrap_ci  # noqa: E402
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides,
    validate_eval_config,
)
from teb_vae.lag_attn_cfs.eval.launch import missing_required, resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_cfs.eval.metrics import (  # noqa: E402
    batch_guids,
    batch_recordings,
    model_inputs,
)
from teb_vae.lag_attn_cfs.eval.probe import (  # noqa: E402
    load_task,
    read_checkpoint,
    resolve_device,
    resolved_config_for,
)
from teb_vae.lag_attn_cfs.eval.run import (  # noqa: E402
    dump_resolved_config,
    make_output_dir,
)
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn.eval.numerics import configure_numerics  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval import figures, lag_metrics  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval import attribution  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval import recording_traces  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.eval.binding import (  # noqa: E402
    ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE,
    EXCLUDED_ANALYSES,
    LAG_RESIDUAL_BINDING,
    MODEL_KIND,
    UNREGISTERED_ANALYSES,
)
from teb_vae.lag_slot_transformer_cfs.eval.predictive import (  # noqa: E402
    DEFAULT_COVERAGE_LEVELS,
    calibration_census,
    draw_concentration,
    finish_calibration,
    matched_predictive_scores,
    merge_calibration,
)
from teb_vae.lag_slot_transformer_cfs.nets import controls  # noqa: E402
from teb_vae.lag_slot_transformer_cfs.nets.core import (  # noqa: E402
    pathway_parameter_counts,
)
from train.data_module import GraphDataModule  # noqa: E402

#: The anchor geometry every evaluation forward runs at. Dense: phase zero, stride one. An
#: evaluation that decoded the training stride would report every number over a fifth of the
#: anchors and say so nowhere, which is why the stride is not a configurable evaluation setting.
DENSE_ANCHOR_GEOMETRY: Tuple[int, int] = (0, 1)

#: Offsets applied to the configured seed for the two explicit generators, so the latent draw and
#: the cross-recording pairing are independent streams and both still follow from one recorded
#: value. Distinct and non-zero: a generator seeded with the bare seed would replay the numbers the
#: global stream is already handing out.
_SEED_OFFSET_MC, _SEED_OFFSET_PERM = 3, 2

#: The two branches whose mixture calibration is accumulated. Both, because a calibration statement
#: about the source-conditioned branch alone cannot say whether the source improved it or whether
#: the observation model was already miscalibrated without it.
CALIBRATED_BRANCHES: Tuple[str, ...] = ("base", "full")

#: Prefix that marks a scored arm as a suppression of one lag band, so the summary's assembly can
#: tell the interventions apart from the two matched branches without a second list to keep aligned.
SUPPRESSION_PREFIX = "suppress:"

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

#: The source controls, and the column each one's margin is taken from. One place, read by the
#: block that reports them, so the control names in the summary and the columns in the table
#: cannot drift apart.
CONTROL_COLUMNS: Mapping[str, str] = {
    "silence": "nll_silence",
    "replace_zeros": "nll_replace:zeros",
    "replace_constant": "nll_replace:constant",
    "permute": "nll_permute",
}

#: The two stored target blocks, in the order the kept channel axis carries them: the scattering
#: coefficients first, the phase-harmonic coefficients after. Named as the configuration names
#: their weights, so a reader can match a block to the weight it was trained under.
TARGET_BLOCKS: Tuple[str, ...] = ("st", "ph")

#: The per-recording table, written beside the summary as the siblings' passes write theirs.
#:
#: The summary carries each column's interval; this carries the values those intervals were built
#: from, one row per recording. Two things need them and neither can be served by an interval: a
#: bootstrap over several seeds of one arm has to resample recordings ONCE and average the seeds
#: inside each resample, which needs every seed's per-recording vector rather than its summary; and
#: two runs can only be shown to have scored disjoint recordings by comparing the recordings they
#: scored.
PER_RECORDING_FILENAME = "per_recording.csv"

#: The per-lag table: one row per candidate lag, carrying the exposure, the latent profile and
#: the predictive margin with its interval. What the lag figures are drawn from.
LAG_PROFILE_FILENAME = "lag_profile.csv"

#: The horizon-resolved table: one row per scored arm and horizon step, with the interval.
HORIZON_FILENAME = "horizon_resolved.csv"

#: Key the assembled results carry the table under, popped before the summary is serialised. The
#: table is a file of its own because it grows with the split while everything else in the summary
#: is a fixed handful of blocks.
PER_RECORDING_KEY = "per_recording_table"

#: Key the assembled results carry the per-recording **curves** under -- one vector per recording
#: per resolved quantity. Popped with the table: the curves are what the resolved blocks and the
#: lag profile were bootstrapped from, and they are written out as the two tables above rather
#: than carried in the summary.
PER_RECORDING_CURVES_KEY = "per_recording_curves"

#: Key the assembled results carry the scored segments' identities under -- one row per segment
#: with its recording, epoch, class and subgroup. Popped before the summary is serialised, once the
#: per-recording traces have drawn their selection from it.
SEGMENT_IDENTITIES_KEY = "segment_identities"


def build_run_config(checkpoint: Any, overrides: Optional[Any] = None) -> Dict[str, Any]:
    """Merge the committed evaluation delta over the checkpoint's own resolved configuration.

    The training run's resolved config is the record of what the model was trained on, so it is the
    base and the delta is merged on top. The alternative -- loading the shipped ``default.yaml`` --
    would evaluate against what a config file says today rather than against what produced this
    checkpoint, which is the drift the arrangement exists to prevent.

    Args:
        checkpoint: The checkpoint being scored; its run's resolved config is found beside it.
        overrides: An override delta path, or ``None`` for this package's committed one.

    Returns:
        The merged, validated configuration, with the loader forced single-process.
    """
    base = load_config(str(resolved_config_for(checkpoint)))
    delta = LAG_RESIDUAL_BINDING.overrides_path if overrides is None else overrides
    merged = merge_eval_overrides(base, delta)
    merged["eval_config"] = validate_eval_config(merged)
    return force_single_process_loader(merged)


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
    at all.

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
        their own recording, and why an arm was skipped when it was.
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
            for arm in ("suppress", "silence", "replace", "permute", "lag_profile")
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
        record["skipped"]["permute"] = (
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
        branches["permute"] = (permuted["mu_post"], permuted["logvar_post"])
        record["n_control_pairs"] = int(u_stream.shape[0])
        record["n_same_recording_pairs"] = controls.same_recording_pairs(recordings, index)
    else:
        # A property of this batch, not of the arm: flagged so the pass counts it apart from the
        # arms that cannot run on the checkpoint at all, and only reports the arm as skipped when
        # no batch of the split could pair.
        record["batch_without_partner"] = True
        record["skipped"]["permute"] = (
            "no cross-recording pairing exists in this batch: it carries no recording "
            "identifiers, or one recording holds more than half of it. Counted rather than "
            "silently dropped -- a control that stopped being a control looks like one that works."
        )
    return branches, record


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
) -> Dict[str, Any]:
    r"""Run one batch: one forward, every arm, one draw loop.

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

    Returns:
        The batch's record: per-sample scores per arm, the per-sample curves the resolved axes are
        built from, the guids that weight them, and the exposure, cancellation, latent-profile and
        calibration **sums** the pass accumulates. Sums rather than means, so the reported figure
        is a mean over the whole split rather than a mean of per-batch means, which would weight a
        short batch equally with a full one.
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
    mask, _coverage = forecast_mask(
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
        persistence=outputs.get("persistence"),
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
    return {
        "guids": batch_guids(batch, batch_size),
        # Who each segment is, beyond the recording the columns are weighted by: its epoch and
        # its cohort labels, which the per-recording traces draw their class-balanced selection
        # from after the pass. Recorded here because this is the one place every scored segment
        # passes through with its batch fields in hand.
        "identity": recording_traces.batch_identity(batch, batch_size),
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


def headline_block(
    per_recording: Mapping[str, Mapping[str, float]],
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """The gap and every arm's margin, each with a recording-level bootstrap interval.

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
    block: Dict[str, Any] = {
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
    return block


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
        "delay_steps": int(getattr(model, "source_delay_steps", 0) or 0),
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


def run_pass(
    task: Any,
    loader: Any,
    *,
    eval_config: Mapping[str, Any],
    num_samples: int,
    max_batches: Optional[int],
) -> Dict[str, Any]:
    """Walk the split once and assemble everything the summary reports.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        eval_config: The validated evaluation settings.
        num_samples: Monte Carlo draws $K$.
        max_batches: Stop after this many batches, or ``None`` for the whole split.

    Returns:
        The assembled results, ready for :func:`~teb_vae.lag_attn.eval.report.json_safe`.
    """
    model = task.orig_model
    device = task.device
    seed = int(eval_config["seed"])
    resamples = int(eval_config["bootstrap_resamples"])
    mc_generator = torch.Generator(device=device).manual_seed(seed + _SEED_OFFSET_MC)
    perm_generator = torch.Generator().manual_seed(seed + _SEED_OFFSET_PERM)
    bands = dict(eval_config.get("occlusion_bands") or {})
    caps = dict(eval_config.get("caps") or {})
    profile_cap: Optional[int] = caps.get(LAG_PROFILE_CAP)
    block_split = kept_block_split(model)

    masks = lag_metrics.band_masks(bands, model.n_lags, device=device)

    records: List[Dict[str, Any]] = []
    exposure_totals: Optional[Dict[str, torch.Tensor]] = None
    latent_totals: Optional[Dict[str, torch.Tensor]] = None
    cancellation_totals: Optional[Dict[str, Dict[str, float]]] = None
    calibration_totals: Dict[str, Any] = {}
    control_pairs = same_recording = 0
    profiled_segments = 0
    skipped: Dict[str, str] = {}
    # Batches the permute arm could not run on because one recording held more than half of
    # them, typically the trailing partial batch of the split. Counted apart from ``skipped``:
    # that dict says an arm did not run on this checkpoint at all, and a control that ran on
    # every batch but the last must not be reported as if it had never run.
    batches_without_partner = segments_without_partner = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= int(max_batches):
                break
            batch = task.transfer_batch_to_device(batch, device, dataloader_idx=0)
            # The profile is scored on the first segments of the split up to the cap: the loader
            # is fixed-seed shuffled, so the leading segments are a draw over the split rather
            # than one shard's prefix.
            profile_this_batch = profile_cap is not None and profiled_segments < int(profile_cap)
            record = score_batch(
                task,
                batch,
                masks=masks,
                num_samples=num_samples,
                mc_generator=mc_generator,
                perm_generator=perm_generator,
                lag_profile=profile_this_batch,
                block_split=block_split,
            )
            if "lag_margin" in record["curves"]:
                profiled_segments += int(record["curves"]["lag_margin"].shape[0])
            exposure_totals = lag_metrics.merge_counts(exposure_totals, record["exposure"])
            if record["latent_profile"]:
                latent_totals = lag_metrics.merge_counts(latent_totals, record["latent_profile"])
            cancellation_totals = lag_metrics.merge_cancellation(
                cancellation_totals, record["cancellation"]
            )
            for branch, block in record["calibration"].items():
                calibration_totals[branch] = merge_calibration(
                    calibration_totals.get(branch), block
                )
            control_pairs += int(record["control"]["n_control_pairs"])
            same_recording += int(record["control"]["n_same_recording_pairs"])
            batch_skips = dict(record["control"]["skipped"])
            if record["control"]["batch_without_partner"]:
                batches_without_partner += 1
                segments_without_partner += len(record["guids"])
                batch_skips.pop("permute")
            skipped.update(batch_skips)
            records.append(record)
            logger.info(f"scored batch {index + 1}")

    # Only when no batch at all could pair is the arm itself unrun; then the per-batch reason is
    # the run's reason, and the margin below comes back MISSING with it.
    if batches_without_partner and not any("nll_permute" in r["columns"] for r in records):
        skipped["permute"] = (
            "no batch of the split held a cross-recording pairing: every batch carried no "
            "recording identifiers, or had one recording holding more than half of it."
        )

    per_recording, per_recording_exposure, curves = aggregate_by_recording(records)
    identities = recording_traces.identity_frame(records)
    headline = headline_block(per_recording, resamples=resamples, seed=seed)
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
    anchor_weighted = _anchor_weighted(records)
    horizon = int(model.horizon)
    block_channels = _block_channel_counts(model, block_split)
    return {
        SEGMENT_IDENTITIES_KEY: identities,
        "headline": headline,
        "anchor_weighted": anchor_weighted,
        "n_recordings": len(per_recording),
        "n_segments": sum(len(record["guids"]) for record in records),
        # The values every interval above was built from, one row per recording. Carried out of
        # here rather than rebuilt by a reader from the batches, which no longer exist by the time
        # the summary is written.
        PER_RECORDING_KEY: {
            guid: {**columns, **per_recording_exposure[guid]}
            for guid, columns in per_recording.items()
        },
        PER_RECORDING_CURVES_KEY: curves,
        # The first thing a reader of two summaries has to know, because it decides what every
        # other block in the file can possibly say. A target-only arm's gap is exactly zero by
        # construction and its lag readouts are empty; a candidate's are neither.
        #
        # The mechanism and the budgets travel with it, because a comparison of arms is not
        # readable without them: two arms differing by a fusion differ by however many parameters
        # that fusion holds, and a predictive difference is not attributable to a mechanism until
        # the budgets are on the table beside it.
        "arm": arm_record(model),
        "lag_readouts": lag_metrics.qualified_report(
            {
                "band_suppression": lag_metrics.band_suppression_block(
                    headline, exposure, intervals=band_intervals
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
            "silence_margin_nats": _margin(headline, "silence"),
            "replace_zeros_margin_nats": _margin(headline, "replace:zeros"),
            "replace_constant_margin_nats": _margin(headline, "replace:constant"),
            "permute_margin_nats": _margin(headline, "permute"),
            "silence_margin_interval": control_intervals["silence"],
            "replace_zeros_margin_interval": control_intervals["replace_zeros"],
            "replace_constant_margin_interval": control_intervals["replace_constant"],
            "permute_margin_interval": control_intervals["permute"],
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
        "calibration": {
            branch: finish_calibration(calibration_totals.get(branch))
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
        # Every analysis this architecture cannot produce, with the tensor it would have needed
        # and how it came to be absent: removed from the shared registry by the binding, or never
        # registered here at all. A reader of a directory with fewer columns than a sibling's
        # should not have to work out which.
        "excluded_analyses": ANALYSES_THIS_ARCHITECTURE_CANNOT_PRODUCE,
        "excluded_analyses_mechanism": {
            "removed_from_shared_registry": list(EXCLUDED_ANALYSES),
            "never_registered_here": list(UNREGISTERED_ANALYSES),
        },
        "encoder_disclosure": LAG_RESIDUAL_BINDING.encoder_disclosure(model),
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


def _margin(headline: Mapping[str, Any], arm: str) -> Any:
    """One arm's predictive margin against the matched full branch.

    Args:
        headline: The bootstrapped per-column block.
        arm: The arm's name, as :func:`intervened_branches` keyed it.

    Returns:
        The margin in nats per anchor, or ``MISSING`` when the arm did not run.
    """
    matched = headline.get("nll_full", {}).get("point")
    intervened = headline.get(f"nll_{arm}", {}).get("point")
    if matched is None or intervened is None:
        return lag_metrics.MISSING
    return float(intervened) - float(matched)


def _anchor_weighted(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
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


def write_per_recording_table(path: Any, table: Mapping[str, Mapping[str, float]]) -> None:
    """Write one row per recording, in a stable column and row order.

    Sorted by recording rather than by the order the loader happened to hand the batches out, so
    two runs of the same split produce two files a reader can diff line by line.

    Args:
        path: The file to write.
        table: ``{recording: {column: value}}``.
    """
    columns = sorted({name for row in table.values() for name in row})
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["guid", *columns])
        for guid in sorted(table):
            row = table[guid]
            writer.writerow([guid, *(row.get(name, "") for name in columns)])


def lag_profile_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per candidate lag, assembled from the summary's own lag blocks.

    Built from the summary rather than from the tensors, so the table and the figure drawn from
    it carry exactly the numbers the summary does.

    Args:
        results: The assembled results.

    Returns:
        The rows, in lag order; empty when the run read no lag.
    """
    readouts = results.get("lag_readouts") or {}
    axis = readouts.get("lag_axis") or {}
    exposure = readouts.get("exposure") or {}
    profile = readouts.get("lag_profile") or {}
    latent = profile.get("latent") or {}
    predictive = profile.get("predictive") or {}
    margin = predictive.get("margin_nats") or {}
    anchors = list(exposure.get("per_lag_anchors") or [])
    if not anchors:
        return []
    channels = list(exposure.get("per_lag_channels") or [])
    step = float(axis.get("seconds_per_step", SECONDS_PER_STEP))
    delay = int(axis.get("delay_steps", 0) or 0)
    rows: List[Dict[str, Any]] = []
    for lag in range(len(anchors)):
        row: Dict[str, Any] = {
            "lag": lag,
            "seconds": step * (lag + delay),
            "anchors": float(anchors[lag]),
            "channels": float(channels[lag]) if lag < len(channels) else "",
        }
        for name in ("proposal_norm", "update_shift", "divergence_drop", "scale_proposal_norm"):
            values = latent.get(name)
            row[name] = "" if not values or values[lag] is None else float(values[lag])
        for part in ("point", "lo", "hi"):
            values = margin.get(part)
            row[f"predictive_margin_{part}"] = (
                "" if not values or lag >= len(values) else float(values[lag])
            )
        rows.append(row)
    return rows


def horizon_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per arm and horizon step, assembled from the summary's horizon block.

    Args:
        results: The assembled results.

    Returns:
        The rows; empty when the run resolved no horizon axis.
    """
    block = results.get("horizon_resolved") or {}
    positions = list(block.get("positions") or [])
    rows: List[Dict[str, Any]] = []
    series: List[Tuple[str, Mapping[str, Any]]] = [
        (f"nll_{arm}", record) for arm, record in (block.get("nll") or {}).items()
    ]
    if block.get("pred_gap"):
        series.append(("pred_gap", block["pred_gap"]))
    series += [
        (f"margin_{SUPPRESSION_PREFIX}{band}", record)
        for band, record in (block.get("band_margins") or {}).items()
    ]
    series += [
        (f"margin_{name}", record)
        for name, record in (block.get("control_margins") or {}).items()
    ]
    for name, record in series:
        for position, step in enumerate(positions):
            rows.append(
                {
                    "series": name,
                    "horizon_step": step,
                    "point": record["point"][position],
                    "lo": record["lo"][position],
                    "hi": record["hi"][position],
                    "n": record.get("n"),
                }
            )
    return rows


def write_rows(path: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of homogeneous rows as a CSV, or nothing when there are none.

    Args:
        path: The file to write.
        rows: The rows, every one carrying the same keys.
    """
    if not rows:
        return
    columns = list(rows[0])
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scored_split_record(
    config: Mapping[str, Any], recordings: Sequence[str]
) -> Dict[str, Any]:
    """Which recordings this pass scored, and which files they came out of.

    **This is the block the reserved-partition question is answered from.** A confirmation run has
    to be shown to have scored a partition that no run used to choose an architecture, a
    hyperparameter or a threshold ever touched, and neither a checkpoint nor a metric can say that:
    the only evidence is which files were opened and which recordings came back. Both are recorded
    here so the question is settled from the artifacts rather than from a memory of which shards
    were pointed at in which week.

    The label is the common parent of the scored files rather than a parsed fold name. A parse
    would encode one dataset layout into a gate that has to keep working when the next build names
    its directories differently, and the common parent separates two partitions exactly as well.

    Args:
        config: The merged run configuration.
        recordings: The recordings the pass actually scored.

    Returns:
        The split block: the files, the statistics they were standardised with, the common parent,
        and a digest of the scored recording set.
    """
    shards = [str(path) for path in config.get("dataset_config", {}).get("vae_test_datasets", [])]
    try:
        common = os.path.commonpath([os.path.dirname(os.path.abspath(path)) for path in shards])
    except ValueError:
        # Paths on different drives have no common parent. A Windows-only condition, and a
        # split assembled from two drives is a legitimate arrangement rather than an error.
        common = ""
    ordered = sorted(str(guid) for guid in recordings)
    return {
        "shards": shards,
        "stat_path": str(config.get("dataset_config", {}).get("stat_path", "")),
        "label": common,
        "n_recordings": len(ordered),
        # A cheap identity for the scored recording set: two runs whose digests agree scored the
        # same recordings, and two whose digests differ need the tables themselves to say how far
        # apart they are. It is an identity check, not a privacy measure -- the table beside it
        # carries the recordings in the clear, as every per-recording export in this repository
        # does.
        "recording_digest": hashlib.sha256(
            "\n".join(ordered).encode("utf-8")
        ).hexdigest()[:16],
    }


def run_traces(
    task: Any,
    loader: Any,
    identities: Any,
    *,
    eval_config: Mapping[str, Any],
    results_dir: Any,
) -> Dict[str, Any]:
    """Run the per-recording traces inside a failure-isolating guard.

    The stage re-reads a handful of recordings through the loader after the pass has finished;
    a failure in it must not lose the pass, so it is recorded under ``recording_traces.error``
    and the summary is written regardless.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        identities: The scored segments' identities the pass recorded.
        eval_config: The validated evaluation settings.
        results_dir: The run's results directory.

    Returns:
        The stage's block, or the error.
    """
    try:
        return recording_traces.run_recording_traces(
            task, loader, identities,
            eval_config=dict(eval_config), results_dir=results_dir,
            geometry_record={"t": int(task.orig_model.geometry.t)},
        )
    except Exception as error:  # noqa: BLE001 - lost traces must not lose the run
        logger.exception("the per-recording traces failed; the summary is complete without them")
        return {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}


def run_attributions(
    task: Any,
    loader: Any,
    identities: Any,
    *,
    config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    results_dir: Any,
) -> Dict[str, Any]:
    """Run the Captum attributions inside a failure-isolating guard.

    The stage re-reads a class-balanced draw of segments through the loader after the pass has
    finished and attributes three per-anchor readouts over the input streams; a failure in it
    must not lose the pass, so it is recorded under ``attribution.error`` and the summary is
    written regardless.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        identities: The scored segments' identities the pass recorded.
        config: The merged run configuration, for the channel map the stage builds.
        eval_config: The validated evaluation settings.
        results_dir: The run's results directory.

    Returns:
        The stage's block, or the error.
    """
    try:
        return attribution.run_attribution(
            task, loader, identities,
            config=config, eval_config=dict(eval_config), results_dir=results_dir,
            geometry_record={"t": int(task.orig_model.geometry.t)},
        )
    except Exception as error:  # noqa: BLE001 - lost attributions must not lose the run
        logger.exception("the attributions failed; the summary is complete without them")
        return {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}


def render_figures(results: Mapping[str, Any], results_dir: Any) -> Dict[str, Any]:
    """Draw every figure of the run from the assembled summary, inside a failure-isolating guard.

    A figure that fails must not lose a multi-hour pass at its last step, so the failure is
    recorded in the summary under ``figures.error`` and the pass completes; the summary is then
    enough to redraw the figures afterwards.

    Args:
        results: The assembled results, with the per-recording table still attached.
        results_dir: The run's results directory.

    Returns:
        The figure manifest: ``{name: relative path}`` plus the format, or the error.
    """
    try:
        return figures.render_run_figures(
            results,
            results_dir,
            per_recording=results.get(PER_RECORDING_KEY) or {},
        )
    except Exception as error:  # noqa: BLE001 - a lost figure must not lose the run
        logger.exception("figure rendering failed; the summary is complete without the figures")
        return {"error": f"{type(error).__name__}: {error}"}


def main(
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    num_mc_samples: Optional[int] = None,
    max_batches: Optional[int] = None,
    overrides: Optional[str] = None,
    sources: Optional[Mapping[str, str]] = None,
) -> int:
    """Score one checkpoint and write ``summary.json``.

    Args:
        checkpoint: The ``.ckpt`` to score. Required, enforced here rather than by argparse.
        output_dir: An explicit run directory, or ``None`` for a timestamped one.
        device: Device string, or ``None`` to choose automatically.
        num_mc_samples: Draws $K$, or ``None`` for the configured value.
        max_batches: Stop after this many batches, or ``None`` for the whole split.
        overrides: An override delta path, or ``None`` for this package's committed one.
        sources: Where each launch value came from, recorded in the summary so a run's provenance
            is recoverable from its own output rather than from a shell history.

    Returns:
        The process exit code: ``0`` on success, ``2`` on a refusal.
    """
    refusal = missing_required({"checkpoint": checkpoint}, ("checkpoint",))
    if refusal is not None:
        logger.error(refusal)
        return 2

    config = build_run_config(checkpoint, overrides)
    eval_config = config["eval_config"]
    configure_numerics(int(eval_config["seed"]))
    resolved_device = resolve_device(device)
    draws = int(eval_config["num_mc_samples"] if num_mc_samples is None else num_mc_samples)

    results_dir = make_output_dir(config, output_dir, binding=LAG_RESIDUAL_BINDING)
    logger.info(f"writing results to {results_dir}")
    dump_resolved_config(config, results_dir)
    # Once per run, before any figure: the style mutates global rcParams and the format is a
    # property of the run, read from the delta so the dumped configuration records it.
    figures.configure_figure_style(eval_config.get("figure_format"))

    blob = read_checkpoint(checkpoint)
    task = load_task(checkpoint, resolved_device, blob=blob, binding=LAG_RESIDUAL_BINDING)
    loader = GraphDataModule(config).test_dataloader()

    results = run_pass(
        task,
        loader,
        eval_config=eval_config,
        num_samples=draws,
        max_batches=max_batches,
    )
    results.pop(PER_RECORDING_CURVES_KEY, None)
    # After the pass, from the identities it recorded: a class-balanced draw of recordings
    # re-read segment by segment. Its own directory beside the tables, its own block in the
    # summary, and a failure inside it costs the traces rather than the pass.
    identities = results.pop(SEGMENT_IDENTITIES_KEY)
    results["recording_traces"] = run_traces(
        task, loader, identities,
        eval_config=eval_config, results_dir=results_dir,
    )
    # The same identities, a second post-pass stage: gradient attributions of the divergence, the
    # gap and the proposal norm over the input streams, into their own directory and their own
    # block, and a failure inside it costs the attributions rather than the pass.
    results["attribution"] = run_attributions(
        task, loader, identities,
        config=config, eval_config=eval_config, results_dir=results_dir,
    )
    table = results[PER_RECORDING_KEY]
    write_per_recording_table(results_dir / PER_RECORDING_FILENAME, table)
    logger.info(f"wrote {results_dir / PER_RECORDING_FILENAME}")
    write_rows(results_dir / LAG_PROFILE_FILENAME, lag_profile_rows(results))
    write_rows(results_dir / HORIZON_FILENAME, horizon_rows(results))
    results["scored_split"] = {
        **scored_split_record(config, list(table)),
        "per_recording_table": PER_RECORDING_FILENAME,
        "lag_profile_table": LAG_PROFILE_FILENAME,
        "horizon_table": HORIZON_FILENAME,
    }
    results["run"] = {
        "checkpoint": str(checkpoint),
        # Both identities, because they answer different questions. ``model_class`` is what the
        # training run stamped and what the class guard compared against; ``model_kind`` is the
        # architecture's own name, and it is what says a checkpoint's source fusion is not the
        # lag-attentive one whatever the two constructors happen to accept.
        "model_class": str(blob.get("model_class", LAG_RESIDUAL_BINDING.model_cls.__name__)),
        "model_kind": MODEL_KIND,
        "device": str(resolved_device),
        "seed": int(eval_config["seed"]),
        # The TRAINING seed and tag, read out of the checkpoint's own resolved configuration. The
        # seed above is the evaluation's, and the two answer different questions: several runs of
        # one arm are several training seeds scored under one evaluation seed, so a protocol that
        # grouped runs by the evaluation seed would see one seed everywhere and a protocol that
        # compared two arms under different evaluation seeds would compare two draw sets.
        "training_seed": int(config.get("general_config", {}).get("seed", -1)),
        "training_tag": str(config.get("general_config", {}).get("tag", "")),
        "overrides_path": str(
            LAG_RESIDUAL_BINDING.overrides_path if overrides is None else overrides
        ),
        "geometry_keys": list(LAG_RESIDUAL_BINDING.geometry_keys),
        # Recorded rather than reconstructed from a shell history: a run whose draw count came from
        # a flag and whose seed came from the delta is a different run from one where both came
        # from the delta, and only this line says which it was.
        "argument_sources": dict(sources or {}),
    }
    # Drawn from the assembled summary and the table, before the table leaves the results: every
    # figure is a picture of a number the summary carries.
    results["figures"] = render_figures(results, results_dir)
    results.pop(PER_RECORDING_KEY)

    summary_path = results_dir / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(results), handle, indent=2)
    logger.info(f"wrote {summary_path}")
    return 0


#: Values used when the module is launched with no command line -- i.e. an IDE's Run button. Keyed
#: by argparse ``dest``. A flag on the command line always wins over the entry here, **per key**, so
#: varying one thing does not discard the rest of the dict.
#:
#: ``checkpoint`` MUST be filled in for this file to run at all; everything else may stay ``None``.
#:
#: This dict is a launch convenience and not a second configuration surface. Anything that shapes
#: what the run *measures* -- the seed, the bands, the bootstrap resamples, the profile cap -- belongs
#: in the override delta, which is dumped into the run directory and is therefore the durable
#: record; a value injected from Python appears in no artifact and cannot be recovered from the
#: output afterwards. ``num_mc_samples`` is the one borderline entry and it is here because a
#: draw-count sweep is exactly the iteration a flag exists for -- the resolved value reaches the
#: summary either way.
RUN_ARGS: Dict[str, Any] = {
    # REQUIRED. Path to the .ckpt to score, repo-root-relative or absolute.
    "checkpoint": None,
    # An explicit run directory, or None for a timestamped one under the config's out_dir_base.
    "output_dir": None,
    # 'cuda:0', 'cpu', or None to choose automatically.
    "device": None,
    # Monte Carlo draws K, or None for the override delta's num_mc_samples.
    "num_mc_samples": None,
    # Stop after this many batches. None scores the whole split; a small number is a smoke run.
    "max_batches": None,
    # An override delta path, or None for this package's committed eval_overrides.yaml.
    "overrides": None,
}


def build_parser() -> argparse.ArgumentParser:
    """Build this entry point's own parser.

    Enumerated here rather than borrowed from a sibling, and that is not tidiness: a borrowed
    parser prints the sibling's module path in the usage line of a command an operator ran against
    this one.

    **No ``required=True`` and no non-``None`` default anywhere below.** The first fires before the
    launch dict is ever consulted, so it makes the Run button unusable no matter what the dict
    says. The second is subtler: the merge treats any non-``None`` parsed value as having come from
    the command line, so an argparse default silently makes that key's ``RUN_ARGS`` entry
    unreachable -- the operator edits the dict, nothing changes, and nothing says why. Real
    defaults are applied after the merge.

    Returns:
        The parser, whose ``dest`` set is also the valid key set for :data:`RUN_ARGS`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_slot_transformer_cfs.eval.run",
        description="Score a lag-residual checkpoint.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to the .ckpt to score.")
    parser.add_argument("--output-dir", default=None, help="Explicit run directory.")
    parser.add_argument("--device", default=None, help="'cuda:0', 'cpu', or omit to choose.")
    parser.add_argument(
        "--num-mc-samples", type=int, default=None, help="Monte Carlo draws K."
    )
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Stop after this many batches."
    )
    parser.add_argument("--overrides", default=None, help="Override delta path.")
    return parser


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, merge with :data:`RUN_ARGS`, and run.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    # The paths inside a config are repo-root-relative, and under an IDE Run button the working
    # directory is whatever the IDE chose -- a relative shard path then resolves to nothing and the
    # loader dies as "no samples match the specified filters" with no mention of the cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(f"argument sources: {sources}")
    return main(**values, sources=sources)


if __name__ == "__main__":
    sys.exit(_cli())
