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

**What a reader must not take from the output.** The band margins do not decompose the gap and are
not normalised to; the qualification travels in the artifact beside them. A band with no available
source is recorded as missing rather than as a measured zero. And a positive source margin is
necessary rather than sufficient: the whole reason this architecture exists is a checkpoint whose
source-conditioned branch was a **worse** predictive density than its target-only one, so the
headline is the gap against the internal base, and its sign is the first thing to read.
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

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn.eval.report import SUMMARY_FILENAME, json_safe  # noqa: E402
from teb_vae.lag_attn.eval.stats import bootstrap_ci  # noqa: E402
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
from teb_vae.lag_slot_transformer_cfs.eval import lag_metrics  # noqa: E402
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

#: The per-recording table, written beside the summary as the siblings' passes write theirs.
#:
#: The summary carries each column's interval; this carries the values those intervals were built
#: from, one row per recording. Two things need them and neither can be served by an interval: a
#: bootstrap over several seeds of one arm has to resample recordings ONCE and average the seeds
#: inside each resample, which needs every seed's per-recording vector rather than its summary; and
#: two runs can only be shown to have scored disjoint recordings by comparing the recordings they
#: scored.
PER_RECORDING_FILENAME = "per_recording.csv"

#: Key the assembled results carry the table under, popped before the summary is serialised. The
#: table is a file of its own because it grows with the split while everything else in the summary
#: is a fixed handful of blocks.
PER_RECORDING_KEY = "per_recording_table"


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


def intervened_branches(
    model: Any,
    outputs: Mapping[str, torch.Tensor],
    streams: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    recordings: Optional[Sequence[str]],
    perm_generator: Optional[torch.Generator],
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

    Args:
        model: The net.
        outputs: The matched forward's dict, taken with ``return_proposals=True``.
        streams: ``(y_st, y_ph, u_stream)`` as the matched forward received them.
        masks: ``{band: (L,) bool}`` removal masks, including the two reference arms.
        recordings: One recording identifier per sample, or ``None`` when the batch carries none.
        perm_generator: Generator for the cross-recording pairing.

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
    record: Dict[str, Any] = {"n_control_pairs": 0, "n_same_recording_pairs": 0, "skipped": {}}

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
            for arm in ("suppress", "silence", "replace", "permute")
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

    Returns:
        The batch's record: per-sample scores per arm, the guids that weight them, and the
        exposure, cancellation and calibration **sums** the pass accumulates. Sums rather than
        means, so the reported figure is a mean over the whole split rather than a mean of
        per-batch means, which would weight a short batch equally with a full one.
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
    )

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
    )
    contributing = scored["base"].contributing
    weights = contributing.to(torch.float64)
    per_sample_anchors = weights.sum(dim=1)

    def per_sample(values: torch.Tensor) -> torch.Tensor:
        """Average a per-anchor quantity within each sample, over its scored anchors."""
        return (values.to(torch.float64) * weights).sum(dim=1) / per_sample_anchors.clamp_min(1.0)

    columns: Dict[str, torch.Tensor] = {
        f"nll_{name}": per_sample(branch.marginal) for name, branch in scored.items()
    }
    columns["pred_gap"] = columns["nll_base"] - columns["nll_full"]
    columns["kld_per_anchor"] = per_sample(outputs["kld_per_anchor"])
    columns["draw_concentration_full"] = per_sample(
        draw_concentration(scored["full"].per_draw)
    )

    # Absent on a target-only checkpoint, where no window was ever gathered. Reported as an empty
    # exposure rather than as counts of zero: nothing was measured, which is a different statement
    # from a source that was available and carried nothing.
    channel_mask = outputs.get("source_channel_mask")
    exposure: Dict[str, torch.Tensor] = {}
    if channel_mask is not None:
        exposure = lag_metrics.lag_exposure(
            outputs["lag_valid"], channel_mask, contributing
        )
        exposure["channels_per_source_channel"] = lag_metrics.channel_exposure(
            channel_mask, contributing
        )
    return {
        "guids": batch_guids(batch, batch_size),
        "columns": {name: value.cpu() for name, value in columns.items()},
        "n_anchors": per_sample_anchors.cpu(),
        "exposure": {name: value.cpu() for name, value in exposure.items()},
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
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Average each column within a recording, then hand back the per-recording values.

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

    Args:
        records: The per-batch records :func:`score_batch` returned.

    Returns:
        ``({recording: {column: value}}, {recording: {'n_segments', 'n_scored_anchors'}})``.
    """
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    anchors: Dict[str, float] = {}
    for record in records:
        names = list(record["columns"])
        for position, guid in enumerate(record["guids"]):
            scored = float(record["n_anchors"][position])
            if scored <= 0.0:
                continue
            bucket = sums.setdefault(guid, {name: 0.0 for name in names})
            counts[guid] = counts.get(guid, 0) + 1
            anchors[guid] = anchors.get(guid, 0.0) + scored
            for name in names:
                bucket[name] += float(record["columns"][name][position])
    per_recording = {
        guid: {name: total / counts[guid] for name, total in bucket.items()}
        for guid, bucket in sums.items()
    }
    exposure = {
        guid: {
            "n_segments": float(counts[guid]),
            "n_scored_anchors": float(anchors[guid]),
        }
        for guid in per_recording
    }
    return per_recording, exposure


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
        ``{column: bootstrap record}`` for every scored column.
    """
    if not per_recording:
        return {}
    names = sorted(next(iter(per_recording.values())))
    return {
        name: bootstrap_ci(
            [values[name] for values in per_recording.values()],
            resamples=int(resamples),
            seed=int(seed),
        )
        for name in names
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
    mc_generator = torch.Generator(device=device).manual_seed(seed + _SEED_OFFSET_MC)
    perm_generator = torch.Generator().manual_seed(seed + _SEED_OFFSET_PERM)
    bands = dict(eval_config.get("occlusion_bands") or {})

    masks = lag_metrics.band_masks(bands, model.n_lags, device=device)

    records: List[Dict[str, Any]] = []
    exposure_totals: Optional[Dict[str, torch.Tensor]] = None
    cancellation_totals: Optional[Dict[str, Dict[str, float]]] = None
    calibration_totals: Dict[str, Any] = {}
    control_pairs = same_recording = 0
    skipped: Dict[str, str] = {}
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= int(max_batches):
                break
            batch = task.transfer_batch_to_device(batch, device, dataloader_idx=0)
            record = score_batch(
                task,
                batch,
                masks=masks,
                num_samples=num_samples,
                mc_generator=mc_generator,
                perm_generator=perm_generator,
            )
            exposure_totals = lag_metrics.merge_counts(exposure_totals, record["exposure"])
            cancellation_totals = lag_metrics.merge_cancellation(
                cancellation_totals, record["cancellation"]
            )
            for branch, block in record["calibration"].items():
                calibration_totals[branch] = merge_calibration(
                    calibration_totals.get(branch), block
                )
            control_pairs += int(record["control"]["n_control_pairs"])
            same_recording += int(record["control"]["n_same_recording_pairs"])
            skipped.update(record["control"]["skipped"])
            records.append(record)
            logger.info(f"scored batch {index + 1}")

    per_recording, per_recording_exposure = aggregate_by_recording(records)
    headline = headline_block(
        per_recording,
        resamples=int(eval_config["bootstrap_resamples"]),
        seed=seed,
    )
    exposure = (
        {}
        if not exposure_totals
        else lag_metrics.band_exposure(masks, exposure_totals)
    )
    anchor_weighted = _anchor_weighted(records)
    return {
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
                "band_suppression": lag_metrics.band_suppression_block(headline, exposure),
                "cancellation": lag_metrics.cancellation_summary(cancellation_totals or {}),
                "exposure": {
                    "per_band": exposure,
                    "per_lag_anchors": (
                        [] if not exposure_totals else exposure_totals["anchors_per_lag"]
                    ),
                    "per_lag_channels": (
                        [] if not exposure_totals else exposure_totals["channels_per_lag"]
                    ),
                    "per_source_channel": (
                        []
                        if not exposure_totals
                        else exposure_totals["channels_per_source_channel"]
                    ),
                },
            }
        ),
        "source_controls": {
            "silence_margin_nats": _margin(headline, "silence"),
            "replace_zeros_margin_nats": _margin(headline, "replace:zeros"),
            "replace_constant_margin_nats": _margin(headline, "replace:constant"),
            "permute_margin_nats": _margin(headline, "permute"),
            "n_control_pairs": control_pairs,
            "n_same_recording_pairs": same_recording,
            "skipped": skipped,
            "note": (
                "Each margin is the arm's own predictive score less the matched full branch's, in "
                "nats per anchor, so a positive value means the fitted model predicts worse under "
                "the intervention. The silence arm verifies the equality invariant and nothing "
                "else: its margin equals the base-minus-full gap by construction, because every "
                "selector off reproduces the prior exactly."
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
        ``{column: value}``, empty when nothing was scored.
    """
    totals: Dict[str, float] = {}
    denominator = 0.0
    for record in records:
        anchors = record["n_anchors"].to(torch.float64)
        denominator += float(anchors.sum())
        for name, values in record["columns"].items():
            totals[name] = totals.get(name, 0.0) + float(
                (values.to(torch.float64) * anchors).sum()
            )
    if denominator <= 0.0:
        return {}
    return {name: total / denominator for name, total in totals.items()}


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
    table = results.pop(PER_RECORDING_KEY)
    write_per_recording_table(results_dir / PER_RECORDING_FILENAME, table)
    logger.info(f"wrote {results_dir / PER_RECORDING_FILENAME}")
    results["scored_split"] = {
        **scored_split_record(config, list(table)),
        "per_recording_table": PER_RECORDING_FILENAME,
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
#: what the run *measures* -- the seed, the bands, the bootstrap resamples -- belongs in the override
#: delta, which is dumped into the run directory and is therefore the durable record; a value
#: injected from Python appears in no artifact and cannot be recovered from the output afterwards.
#: ``num_mc_samples`` is the one borderline entry and it is here because a draw-count sweep is
#: exactly the iteration a flag exists for -- the resolved value reaches the summary either way.
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
