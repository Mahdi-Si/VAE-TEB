r"""Preservation gates, held-out metrics and the GUID bootstrap.

The preservation half is LP-07 and is implemented below; the held-out metrics, the bootstrap and the
controls are LP-11 and are still described here rather than written.

Preservation, measured through the model's own losses
-----------------------------------------------------

An evaluation that re-implements the objective it evaluates measures its re-implementation, so the
forecast is scored through the existing functions rather than a local copy:
``teb_vae.lag_attn_rws.nets.losses`` supplies ``raw_sample_score``, ``masked_raw_block_per_anchor``,
``masked_raw_likelihood``, ``masked_source_kl``, ``masked_prior_rate`` and ``kld_tensor``, and
``teb_vae.lag_attn_cfs.nets.causal_feature_target`` supplies the gathered forecast target
(``_build_forecast_target``, which applies the checkpoint's per-channel ``target_forecast_shift``)
and ``scored_weight``, the conservative pooled validity over that shift span. Masks come from
``raw_masks.forecast_mask`` / ``contributing_anchors`` at the checkpoint's own ``coverage_floor``.
The comparison is in the checkpoint's normalized coefficient space, with anchors, segments and
recordings averaged in the same order for both models.

The validation subset the gates are measured on is chosen deterministically, includes healthy-BG and
adverse-outcome recordings where both exist, and its identities and support are saved with the run,
so "before" and "after" are the same recordings on the same anchors.

Gates, all declared before training and all heuristic rather than clinically validated:

* full-branch forecast MSE at most $10\%$ above the frozen baseline's, with healthy-only drift
  reported separately and an explicit rule for a zero baseline MSE;
* posterior-delta saturation (``delta_mu_sat_frac``, the fraction of $|\Delta\mu|$ at the
  ``delta_mu_scale`` bound) no more than $5$ percentage points above baseline on valid support;
* all means and losses finite, and no global latent collapse.

Alongside the gates, and not as a substitute for them: matched-policy Monte Carlo full-branch NLL and
the source-conditioned KL diagnostics, through the existing routines, with **common random draws**
keyed per example so before/after see identical noise. Eight draws to start, with a larger-draw
comparison available when a conclusion turns on a small NLL difference. The training-path
``pred_gap`` uses asymmetric decoding policies and is not the preservation statistic. A fixed decoder
does not preserve a forecast when its latent input moves.

Held-out metrics
----------------

One observation per eligible held-out recording, computed for the frozen and the adapted model on the
**same GUIDs** and the same final-hour bags: AUROC, average precision beside the adverse-outcome
prevalence that sets its chance level, and balanced accuracy at a threshold selected on validation and
never on test. Each model is read through its own fitted linear head -- both have the same capacity.

Paired differences carry 95% intervals from 1000 outcome-stratified bootstrap resamples of GUIDs,
with the **same** resampled GUIDs used for both models; if patients contribute several GUIDs, patients
are resampled instead, and GUID-only grouping is disclosed when no stronger identifier exists.
Undefined strata -- a resample with one class absent -- are handled explicitly rather than dropped
silently. ``teb_vae.lag_attn.eval.stats.bootstrap_ci`` and the shared rank statistics are reused, so
"significant" means here what it means everywhere else in this repository.

Reported beside them: forecast MSE and NLL, standardized latent movement, covariance and effective
rank, and saturation, before and after.

Descriptive strata, never refitted per subgroup: healthy-BG-only controls as a prespecified
sensitivity analysis, CS/no-CS where counts permit, and acidosis and HIE contrasted separately
against the same healthy controls. Training is binary and the headline stays binary -- picking the
better subgroup afterwards would be selection on the test set. Tiny strata get counts and wide
intervals, or points, not significance claims.

Controls beyond the frozen baseline: the shuffled-label run from ``train.py``, the time/coverage and
subgroup comparison (a cloud separated mainly by ascertainment or signal quality is not an outcome
result), and a cheap frozen linear probe on pooled ``mu_prior`` under the same split and protocol.
That probe is required before any claim that the combined branch helps -- and ``mu_prior`` is
unchanged by this adaptation, so it is also an invariant check. Better ``mu_post`` discrimination
alone does not establish UP-specific information; that would need source-null and shuffle controls
with matched forecasts.

Test data is untouched until the baseline, the selected checkpoint, the scaler, the threshold, the
controls and the analysis settings are locked, and the lock is enforced in code rather than by
convention.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval.metrics import mc_predictive_block, model_inputs
from teb_vae.lag_attn_rws.nets.losses import (
    kld_tensor,
    masked_prior_rate,
    masked_raw_block_per_anchor,
    masked_source_kl,
)
from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
from teb_vae.lag_attn_rws.nets.raw_masks import (
    contributing_anchors,
    forecast_mask,
    kl_mask,
)
from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

# =============================================================================
# What one preservation pass measures
# =============================================================================
#: The per-anchor scalars a preservation pass produces, in a fixed column order.
#:
#: ``mse_full`` is the gated quantity: the **deterministic** forecast from $\mu^q$ -- not from a
#: draw of $z^q$ -- scored against the checkpoint's own gathered target, per coefficient of the
#: normalized block. ``mse_base`` is the same for $\mu^p$ and moves only if the freeze broke, so it
#: is an invariant beside the measurement rather than a second result. ``delta_mu_sat`` is the
#: fraction of an anchor's latent coordinates whose $|\Delta\mu|$ sits at the model's own bound.
#: The two Monte Carlo columns and the KL are diagnostics reported beside the gates and are filled
#: only when a draw count is requested; they are ``nan`` otherwise, never zero.
SCORE_COLUMNS: Tuple[str, ...] = (
    "mse_full",
    "mse_base",
    "delta_mu_sat",
    "nll_full",
    "nll_base",
    "kl",
)

#: On-disk names inside a run directory.
GATE_SUBSET_FILENAME = "gate_subset.json"
PRESERVATION_INDEX_FILENAME = "preservation_anchors.parquet"
PRESERVATION_RECORD_FILENAME = "preservation.json"

#: Stratum labels for the gate subset, in the order recordings are drawn round-robin from them.
#: Adverse first, because it is the scarce one on this cohort and a subset that ran out of places
#: should run out of healthy ones.
GATE_STRATA: Tuple[str, ...] = ("adverse", "healthy_bg", "healthy_no_bg")

#: Fewest resampling units below which no interval is reported. The repository's own floor, from
#: ``teb_vae.lag_attn.eval.stats``: below it a percentile bootstrap reproduces the sample rather
#: than estimating its spread, and the two bounds are decided by two order statistics.
MIN_UNITS = 3

#: How the gate reports a baseline that carries no error to compare against. A relative tolerance
#: is undefined at zero, so the run says which rule it fell back to rather than dividing.
ZERO_BASELINE_RULE = "zero_baseline_requires_zero_candidate"


class GateEvaluationError(PilotConfigError):
    """A gate could not be decided, as distinct from a candidate that failed it.

    Raised when the *baseline* is unusable -- a non-finite reference MSE, an empty subset, a
    candidate measured on different anchors. None of these says anything about the candidate, and
    reporting one as a failed gate would put "the adaptation degraded the forecast" in a run log
    for a reason that has nothing to do with the adaptation.
    """


@dataclass(frozen=True)
class PreservationReading:
    """One model's preservation measurement on the fixed gate subset.

    Attributes:
        frame: One row per decoded, in-window anchor of the subset: identity, its hour coordinate,
            its latest scored endpoint, its support flag, an exclusion reason (empty when retained)
            and ``row`` -- the index into ``values``.
        values: ``(N, len(SCORE_COLUMNS) + d_z)``: the per-anchor scalars followed by the raw
            ``mu_post`` coordinates, so the recording reduction that produces the scores also
            produces the latent summary the collapse check reads.
        recordings: One row per recording, with the scores reduced anchors -> segments ->
            recordings and the outcome joined on.
        record: The scalars the gate is decided from, the counts behind them, and the support
            digest that says which anchors they were measured over.
    """

    frame: pd.DataFrame
    values: np.ndarray
    recordings: pd.DataFrame
    record: Dict[str, Any]


# =============================================================================
# The fixed validation subset
# =============================================================================
def outcome_map(recordings: pd.DataFrame) -> Dict[str, Optional[int]]:
    """GUID -> binary outcome, for the recordings that have one.

    Args:
        recordings: The recording table.

    Returns:
        The mapping. A recording excluded at the labelling stage is absent rather than present
        with ``None``, so a lookup miss and an unknown outcome are the same thing.
    """
    return {
        str(row[data.GUID_COLUMN]): (
            None if pd.isna(row[data.OUTCOME_COLUMN]) else int(row[data.OUTCOME_COLUMN])
        )
        for _index, row in recordings.iterrows()
        if not str(row.get(data.EXCLUSION_COLUMN, ""))
    }


def _stratum_of(row: Mapping[str, Any]) -> Optional[str]:
    """Which gate stratum a recording belongs to, or ``None`` when it has no usable outcome."""
    outcome = row.get(data.OUTCOME_COLUMN)
    if outcome is None or pd.isna(outcome):
        return None
    if int(outcome) == 1:
        return "adverse"
    return "healthy_bg" if data._flag_is(row.get("bg_label"), True) else "healthy_no_bg"


def gate_subset(
    recordings: pd.DataFrame, *, size: int, seed: int, split: str = "val"
) -> Dict[str, Any]:
    """Choose, once, the recordings every preservation measurement in this run is made on.

    Deterministic in the strong sense: the same cohort and the same seed choose the same GUIDs,
    whatever order the shards were read in. Recordings are sorted by GUID inside each stratum,
    permuted by a generator seeded from the run's own seed, and then drawn **round-robin** across
    the strata -- so a subset that cannot hold everything still holds adverse-outcome and
    healthy-BG recordings side by side, which is what the gate needs to be a statement about both
    classes rather than about whichever one is larger.

    The gate must be measured on the *same* recordings before and after; a subset re-chosen per
    candidate would let a difference between two populations be reported as a difference between
    two models. That is why this returns a record to be saved rather than a list to be recomputed.

    Args:
        recordings: The recording table, with eligibility already attached.
        size: How many recordings to draw. The whole eligible split is used when it is smaller.
        seed: The run's seed.
        split: The split the subset is drawn from. Validation, always, in this pilot: the gate
            decides which candidate is eligible for selection, and a gate measured on training
            recordings would be measured on the data the candidate was fitted to.

    Returns:
        The record: the chosen GUIDs, the per-stratum counts, the strata that were unavailable and
        the settings behind the draw.

    Raises:
        GateEvaluationError: If no eligible recording of that split has a usable outcome.
    """
    frame = recordings[recordings[data.SPLIT_COLUMN].astype(str) == str(split)]
    if "eligible" in frame.columns:
        frame = frame[frame["eligible"].astype(bool)]
    else:
        frame = frame[frame[data.EXCLUSION_COLUMN].astype(str) == ""]

    pools: Dict[str, List[str]] = {name: [] for name in GATE_STRATA}
    for _index, row in frame.sort_values(data.GUID_COLUMN).iterrows():
        stratum = _stratum_of(row)
        if stratum is not None:
            pools[stratum].append(str(row[data.GUID_COLUMN]))

    if not any(pools.values()):
        raise GateEvaluationError(
            f"no eligible {split!r} recording carries a usable binary outcome, so the preservation "
            f"gate has nothing to measure. Every candidate would then be eligible for selection "
            f"without its forecast having been checked at all."
        )

    generator = np.random.default_rng(int(seed))
    order = {name: [pool[index] for index in generator.permutation(len(pool))]
             for name, pool in pools.items()}

    chosen: List[str] = []
    limit = min(int(size), sum(len(pool) for pool in order.values()))
    position = 0
    while len(chosen) < limit:
        drawn_this_round = False
        for name in GATE_STRATA:
            if len(chosen) >= limit:
                break
            if position < len(order[name]):
                chosen.append(order[name][position])
                drawn_this_round = True
        if not drawn_this_round:
            break
        position += 1

    counts = {
        name: sum(1 for guid in chosen if guid in set(pools[name])) for name in GATE_STRATA
    }
    missing = [name for name in GATE_STRATA if not pools[name]]
    record = {
        "split": str(split),
        "seed": int(seed),
        "requested_size": int(size),
        "guids": sorted(chosen),
        "n_recordings": len(chosen),
        "counts_by_stratum": counts,
        "available_by_stratum": {name: len(pool) for name, pool in pools.items()},
        "strata_unavailable": missing,
        "note": (
            "the gate subset is drawn once, before any candidate is fitted, and every preservation "
            "measurement in this run is made on exactly these recordings"
        ),
    }
    if missing:
        logger.warning(
            f"gate subset: no {', '.join(missing)} recording in the {split!r} split, so the "
            f"preservation gate is measured without that stratum"
        )
    logger.info(
        f"gate subset: {len(chosen)} {split!r} recording(s), {counts}, seed {int(seed)}"
    )
    return record


def save_gate_subset(record: Mapping[str, Any], directory: Any) -> Path:
    """Write the gate subset into a run directory.

    Args:
        record: The record from :func:`gate_subset`.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / GATE_SUBSET_FILENAME
    target.write_text(json.dumps(dict(record), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_gate_subset(directory: Any) -> Dict[str, Any]:
    """Read the gate subset back.

    Args:
        directory: The run directory.

    Returns:
        The record.

    Raises:
        FileNotFoundError: If it was never written. The subset is chosen once, and a stage that
            cannot find it must not choose another one.
    """
    target = Path(directory) / GATE_SUBSET_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} is missing. The preservation subset is drawn once, before any candidate is "
            f"fitted; a stage that re-drew it would compare two models on two populations."
        )
    return json.loads(target.read_text(encoding="utf-8"))


# =============================================================================
# One preservation pass
# =============================================================================
def _sample_seed(seed: int, guids: Sequence[Optional[str]], epochs: Sequence[float]) -> int:
    """A draw seed keyed to the examples in one batch.

    Keyed rather than counted, so the Monte Carlo noise a batch sees is a property of *which
    recordings and segments it holds* and not of how many batches came before it. That is what
    makes the before/after comparison a common-random-numbers comparison even if the two passes
    are run in different processes, on different days, from different stages.

    Args:
        seed: The run's Monte Carlo seed.
        guids: The batch's recording identifiers, one per selected sample.
        epochs: Their segment starts.

    Returns:
        A non-negative seed inside torch's accepted range.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(seed)).encode("utf-8"))
    for guid, epoch in zip(guids, epochs):
        digest.update(f"|{guid}|{float(epoch):.6f}".encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") % (2**63 - 1)


def _select_samples(
    inputs: Tuple[Any, ...], target_features: torch.Tensor, weight: torch.Tensor, keep: np.ndarray
) -> Tuple[Tuple[Any, ...], torch.Tensor, torch.Tensor]:
    """Restrict a batch to the samples the gate subset names.

    The forward is the expensive part of a gate that runs once per candidate epoch, so the batch is
    narrowed before it rather than the scores discarded after it. The anchor phase is sliced too
    where it is per-sample; the stride is scalar and is not.

    Args:
        inputs: The five forward arguments.
        target_features: The gathered target stream $(B, T, c_y)$.
        weight: The decimated validity signal $(B, T)$.
        keep: A boolean mask over the batch.

    Returns:
        The same three, restricted. The inputs are returned untouched when every sample is kept.
    """
    if bool(keep.all()):
        return inputs, target_features, weight
    y_st, y_ph, u_stream, phase, stride = inputs
    index = torch.as_tensor(np.flatnonzero(keep), dtype=torch.long, device=y_st.device)
    if torch.is_tensor(phase) and phase.dim() > 0:
        phase = phase.index_select(0, index)
    return (
        (
            y_st.index_select(0, index),
            y_ph.index_select(0, index),
            u_stream.index_select(0, index),
            phase,
            stride,
        ),
        target_features.index_select(0, index),
        weight.index_select(0, index),
    )


def preservation_pass(
    loaded: Any,
    loader: Any,
    *,
    guids: Sequence[str],
    outcomes: Mapping[str, Optional[int]],
    preservation_hours: float,
    mc_draws: Optional[int] = None,
    mc_seed: int = 0,
    max_batches: Optional[int] = None,
) -> PreservationReading:
    r"""Score one model's forecast, saturation and latent spread on the fixed gate subset.

    **Deterministic, and that is the point.** The gated forecast is decoded from $\mu^q$ rather
    than from a draw of $z^q$: two readings of one model must agree exactly, or a gate could be
    passed or failed by the noise between them. The decode is the model's own -- the shared
    decoder, at the anchors the forward returned, carrying the forward's own persistence tensor
    where the checkpoint was built with the residual -- and the target is the checkpoint's own
    gathered block, so the per-channel forecast clock is applied by the code that owns it.

    Support is the same rule the rest of this package uses and is applied in the same order: the
    objective's forecast-contributing mask first, then the preserved window, then the post-delivery
    endpoint check on the furthest scored coefficient, then both duplication rules. An anchor whose
    forecast reaches past delivery is excluded here for exactly the reason it is excluded from the
    latent tables, so the gate and the analysis describe one anchor set.

    The Monte Carlo columns are optional because they are not what the gate decides: eight draws
    per branch per batch is the expensive part of this function, and the gate runs once per
    candidate epoch while the marginal NLL is wanted twice in a whole run. Requested, the draws are
    keyed to the examples in each batch, so the frozen and the adapted model see identical noise.

    Args:
        loaded: The loaded checkpoint bundle, at the weights being measured.
        loader: The validation dataloader.
        guids: The gate subset's recordings.
        outcomes: GUID -> binary outcome, for the healthy-only drift check.
        preservation_hours: The preserved window's upper edge.
        mc_draws: Monte Carlo draws $K$ for the marginal predictive NLL, or ``None`` to skip it.
        mc_seed: The seed those draws are keyed from.
        max_batches: Stop after this many batches. Smoke runs only.

    Returns:
        The reading.

    Raises:
        PilotConfigError: If the forward does not resolve the dense anchor geometry.
        GateEvaluationError: If the subset contributes no retained anchor at all.
    """
    task, model = loaded.task, loaded.model
    geometry = dict(loaded.geometry)
    trim_minutes = geometry.get("trim_minutes")
    horizon = int(geometry["horizon"])
    shift = data.max_forecast_shift(model)
    likelihood = str(getattr(task, "hparams", {}).get("likelihood", "gaussian_nll"))
    wanted = {str(guid) for guid in guids}
    d_z = int(model.d_z)

    rows: List[Dict[str, Any]] = []
    collected: List[np.ndarray] = []
    n_batches = 0
    n_scored_batches = 0
    n_outside_window = 0
    # The pooled KL readouts, accumulated as (sum over anchors, anchor count) so the pass reports
    # one nats-per-anchor number over the whole subset rather than an average of per-batch
    # averages, which would weight a short final batch like a full one.
    pooled: Dict[str, float] = {"kl_raw": 0.0, "prior_rate": 0.0, "active_frac": 0.0}
    n_kl_anchors = 0.0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                if max_batches is not None and n_batches >= max_batches:
                    break
                n_batches += 1
                batch = pilot_model.to_device(task, batch)
                batch_size = data._batch_size(batch)
                batch_guids = data._strings_of(batch, data.GUID_COLUMN, batch_size)
                keep_samples = np.array(
                    [str(guid) in wanted for guid in batch_guids], dtype=bool
                )
                if not keep_samples.any():
                    continue
                n_scored_batches += 1

                _y_st, _y_ph, _u, target_features, weight = model_inputs(task, batch)
                inputs, target_features, weight = _select_samples(
                    pilot_model.forward_inputs(task, batch), target_features, weight, keep_samples
                )
                outputs = model(*inputs)

                anchor_index = outputs["anchor_index"]
                anchor_valid = outputs["anchor_valid"]
                # The checkpoint's own target gather and its own mask, both at the anchors the
                # forward returned rather than at a second derivation of them.
                target = model._build_forecast_target(target_features, anchor_index)
                mask, coverage = forecast_mask(
                    model.scored_weight(weight),
                    model.geometry,
                    coverage_floor=model.coverage_floor,
                    anchors=anchor_index,
                    anchor_valid=anchor_valid,
                )
                # The composition ``data.contributing_support`` performs, unrolled because the mask
                # itself is scored against here and calling that seam would build it twice. Same
                # two functions, same order, same arguments: one support policy, not a second one.
                support = contributing_anchors(mask).bool()

                gather_index = anchor_index[:, :, None].expand(-1, -1, d_z)
                persistence = outputs.get("persistence")
                block_width = float(horizon * int(model.decoder_out_channels))
                scores: Dict[str, torch.Tensor] = {}
                for name, key in (("mse_full", "mu_post"), ("mse_base", "mu_prior")):
                    forecast_mu, _forecast_logvar = model.decoder(
                        outputs[key].gather(1, gather_index), persistence=persistence
                    )
                    block, _contributing = masked_raw_block_per_anchor(
                        forecast_mu, target, mask, likelihood="mse"
                    )
                    # Per coefficient of the block, on the objective's own fixed divisor rather
                    # than on each anchor's surviving element count: a per-anchor divisor would
                    # make the number drift with mask density instead of with forecast error.
                    scores[name] = block / block_width

                delta = (outputs["mu_post"] - outputs["mu_prior"]).gather(1, gather_index)
                scores["delta_mu_sat"] = (
                    delta.abs() >= (SATURATION_FRAC * float(model.delta_mu_scale))
                ).to(delta.dtype).mean(dim=-1)

                kld_btd = kld_tensor(
                    mu_prior=outputs["mu_prior"],
                    logvar_prior=outputs["logvar_prior"],
                    mu_post=outputs["mu_post"],
                    logvar_post=outputs["logvar_post"],
                )
                scores["kl"] = kld_btd.sum(dim=-1).gather(1, anchor_index)

                # The pooled readouts, through the objective's own reduction and over the objective's
                # own KL support -- derived from the forecast mask rather than restated, so the KL
                # and the reconstruction are averaged over one anchor set. ``free_bits=0`` because
                # this is a diagnostic: the floored variant is a training quantity and reporting it
                # as a measurement would report the floor.
                kl_support = kl_mask(
                    mask, model.geometry, anchors=anchor_index, anchor_valid=anchor_valid
                )
                n_support = float(kl_support.sum().item())
                if n_support > 0.0:
                    source_kl = masked_source_kl(kld_btd, kl_support, free_bits=0.0)
                    pooled["kl_raw"] += n_support * float(
                        source_kl["source_conditioned_kl_raw"].item()
                    )
                    pooled["active_frac"] += n_support * float(
                        source_kl["kld_active_frac"].item()
                    )
                    pooled["prior_rate"] += n_support * float(
                        masked_prior_rate(outputs["logvar_prior"], kl_support).item()
                    )
                    n_kl_anchors += n_support

                epochs = data._to_numpy(data._field(batch, data.EPOCH_COLUMN))[keep_samples]
                kept_guids = [
                    guid for guid, keep in zip(batch_guids, keep_samples.tolist()) if keep
                ]

                if mc_draws is None:
                    scores["nll_full"] = torch.full_like(scores["mse_full"], float("nan"))
                    scores["nll_base"] = torch.full_like(scores["mse_full"], float("nan"))
                else:
                    generator = torch.Generator(device=target.device)
                    generator.manual_seed(_sample_seed(mc_seed, kept_guids, epochs.tolist()))
                    marginal, _contributing = mc_predictive_block(
                        model,
                        {
                            "nll_full": (outputs["mu_post"], outputs["logvar_post"]),
                            "nll_base": (outputs["mu_prior"], outputs["logvar_prior"]),
                        },
                        target,
                        mask,
                        anchors=anchor_index,
                        likelihood=likelihood,
                        num_samples=int(mc_draws),
                        generator=generator,
                        persistence=persistence,
                    )
                    scores.update(marginal)

                anchors = anchor_index.detach().cpu().numpy()
                valid = anchor_valid.detach().cpu().numpy().astype(bool)
                supported = support.detach().cpu().numpy().astype(bool)
                coverage_values = coverage.detach().cpu().numpy()

                hours = data.hours_before_delivery(
                    data.anchor_seconds(epochs[:, None], anchors, trim_minutes=trim_minutes)
                )
                inside = data.in_window(hours, 0.0, float(preservation_hours))
                keep = valid & inside
                n_outside_window += int((valid & ~inside).sum())
                if not keep.any():
                    continue

                sample_rows, anchor_rows = np.nonzero(keep)
                for sample, anchor_position in zip(sample_rows.tolist(), anchor_rows.tolist()):
                    rows.append({
                        data.GUID_COLUMN: kept_guids[sample],
                        data.EPOCH_COLUMN: float(epochs[sample]),
                        data.ANCHOR_COLUMN: int(anchors[sample, anchor_position]),
                        "coverage_frac": float(coverage_values[sample, anchor_position]),
                        "contributing": bool(supported[sample, anchor_position]),
                        data.ROW_COLUMN: len(rows),
                    })
                block_values = np.concatenate(
                    [
                        np.stack(
                            [
                                scores[name].detach().cpu().numpy()[sample_rows, anchor_rows]
                                for name in SCORE_COLUMNS
                            ],
                            axis=-1,
                        ),
                        outputs["mu_post"]
                        .gather(1, gather_index)
                        .detach()
                        .cpu()
                        .numpy()[sample_rows, anchor_rows],
                    ],
                    axis=-1,
                )
                collected.append(block_values.astype(np.float64))
    finally:
        if was_training:
            model.train()

    if not rows:
        raise GateEvaluationError(
            f"the gate subset contributed no anchor inside the last {preservation_hours} hour(s). "
            f"Either the loader never yielded one of its {len(wanted)} recording(s), or none of "
            f"them reaches the preserved window; the gate cannot be decided from an empty support."
        )

    frame = pd.DataFrame(rows)
    values = np.concatenate(collected, axis=0)
    frame = data.add_anchor_times(
        frame, trim_minutes=trim_minutes, horizon=horizon, forecast_shift=shift
    )
    frame[data.EXCLUSION_COLUMN] = np.where(
        frame["contributing"].to_numpy(dtype=bool), "", data.EXCLUDED_NOT_CONTRIBUTING
    )
    frame = data.mark_window_and_delivery(frame, preservation_hours=preservation_hours)
    frame = data.deduplicate_anchors(frame)

    retained = data.retained(frame)
    if retained.empty:
        raise GateEvaluationError(
            f"every one of the {len(frame)} in-window anchor(s) of the gate subset was excluded: "
            f"{data.anchor_exclusion_counts(frame)}. A gate decided on no anchor would pass every "
            f"candidate."
        )

    segments, segment_values = data.segment_means(retained, values)
    recordings, recording_values = data.recording_means(segments, segment_values)
    recordings = recordings.copy()
    for position, name in enumerate(SCORE_COLUMNS):
        recordings[name] = recording_values[:, position]
    recordings[data.OUTCOME_COLUMN] = [
        outcomes.get(str(guid)) for guid in recordings[data.GUID_COLUMN].tolist()
    ]

    record = _preservation_record(
        retained=retained,
        recordings=recordings,
        recording_values=recording_values,
        frame=frame,
        preservation_hours=preservation_hours,
        mc_draws=mc_draws,
        mc_seed=mc_seed,
        likelihood=likelihood,
        n_batches=n_batches,
        n_scored_batches=n_scored_batches,
        n_outside_window=n_outside_window,
        n_requested=len(wanted),
        pooled={
            name: (value / n_kl_anchors if n_kl_anchors > 0.0 else float("nan"))
            for name, value in pooled.items()
        },
        n_kl_anchors=n_kl_anchors,
    )
    logger.info(
        f"preservation: mse_full={record['mse_full']:.6g} on {record['n_recordings']} "
        f"recording(s) / {record['n_retained_anchors']} anchor(s); "
        f"saturation={record['delta_mu_sat_pp']:.3f} pp; "
        f"healthy-only mse_full={record['mse_full_healthy']}"
    )
    return PreservationReading(
        frame=frame, values=values, recordings=recordings, record=record
    )


def _support_digest(retained: pd.DataFrame) -> str:
    """A digest of the exact anchors a reading was measured over.

    Two readings compared as before and after must have been taken on the same anchors. Comparing
    the digests is how that is checked without carrying both frames to the comparison: a candidate
    whose support moved -- because its mask changed, or because a batch was skipped -- would
    otherwise report a difference in populations as a difference in forecast error.

    Args:
        retained: The retained anchor rows.

    Returns:
        The first sixteen hex characters of a digest over the sorted ``(guid, epoch, anchor)``
        keys.
    """
    digest = hashlib.sha256()
    keys = retained[[data.GUID_COLUMN, data.EPOCH_COLUMN, data.ANCHOR_COLUMN]].to_numpy()
    for guid, epoch, anchor in sorted(
        (str(row[0]), float(row[1]), int(row[2])) for row in keys
    ):
        digest.update(f"{guid}|{epoch:.6f}|{anchor}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def _preservation_record(
    *,
    retained: pd.DataFrame,
    recordings: pd.DataFrame,
    recording_values: np.ndarray,
    frame: pd.DataFrame,
    preservation_hours: float,
    mc_draws: Optional[int],
    mc_seed: int,
    likelihood: str,
    n_batches: int,
    n_scored_batches: int,
    n_outside_window: int,
    n_requested: int,
    pooled: Mapping[str, float],
    n_kl_anchors: float,
) -> Dict[str, Any]:
    """Reduce one pass to the scalars a gate is decided from.

    Every score is the mean over **recordings**, each of which is already the mean over its
    segments and then over their anchors, so a densely covered recording carries no more weight in
    the gate than a sparse one.

    Args:
        retained: The retained anchor rows.
        recordings: The per-recording table, scores attached.
        recording_values: The aligned per-recording matrix.
        frame: The full anchor frame, for the exclusion counts.
        preservation_hours: The preserved window's upper edge.
        mc_draws: The draw count, or ``None``.
        mc_seed: The seed the draws were keyed from.
        likelihood: The checkpoint's own likelihood, under which the NLL was scored.
        n_batches: Batches the loader yielded.
        n_scored_batches: How many of them held a subset recording.
        n_outside_window: Decoded anchors outside the preserved window.
        n_requested: How many recordings the subset named.
        pooled: The anchor-pooled KL readouts, already divided by their anchor count.
        n_kl_anchors: That count, so a later reader can say how much support they rest on.

    Returns:
        The record.
    """
    healthy = recordings[recordings[data.OUTCOME_COLUMN] == 0]
    latent = recording_values[:, len(SCORE_COLUMNS):]
    # Spread across recordings, not across anchors: a posterior that moved every recording to one
    # point is collapsed however much it varies within a recording.
    latent_std = latent.std(axis=0) if latent.shape[0] > 1 else np.zeros(latent.shape[1])

    record: Dict[str, Any] = {
        "n_requested_recordings": int(n_requested),
        "n_recordings": int(len(recordings)),
        "n_healthy_recordings": int(len(healthy)),
        "n_adverse_recordings": int((recordings[data.OUTCOME_COLUMN] == 1).sum()),
        "n_segments": int(retained[[data.GUID_COLUMN, data.EPOCH_COLUMN]].drop_duplicates().shape[0]),
        "n_retained_anchors": int(len(retained)),
        "n_in_window_anchors": int(len(frame)),
        "n_outside_window": int(n_outside_window),
        "n_batches": int(n_batches),
        "n_scored_batches": int(n_scored_batches),
        "exclusions": data.anchor_exclusion_counts(frame),
        "preservation_hours": float(preservation_hours),
        "support_digest": _support_digest(retained),
        "likelihood": likelihood,
        # Nats per anchor, over the objective's own KL support. A changed KL is a changed number
        # and not evidence of changed physiological coupling; it is reported because it moves when
        # mu_post moves, and a run that did not report it would be hiding that.
        "source_conditioned_kl": float(pooled["kl_raw"]),
        "prior_rate": float(pooled["prior_rate"]),
        "kld_active_frac": float(pooled["active_frac"]),
        "n_kl_anchors": float(n_kl_anchors),
        "mc_draws": None if mc_draws is None else int(mc_draws),
        "mc_seed": int(mc_seed),
        "n_latent_coordinates_varying": int(np.sum(latent_std > 0.0)),
        "d_z": int(latent.shape[1]),
        "note": (
            "mse_* are deterministic mean-decoded forecasts per coefficient of the normalized "
            "block; nll_* are marginal predictive block scores in nats per anchor and are nan when "
            "no draws were requested; the training path's pred_gap is a different quantity and is "
            "not this measurement"
        ),
    }
    for position, name in enumerate(SCORE_COLUMNS):
        column = recording_values[:, position]
        record[name] = float(np.mean(column)) if column.size else float("nan")
    record["delta_mu_sat_pp"] = 100.0 * float(record["delta_mu_sat"])
    record["mse_full_healthy"] = (
        float(healthy["mse_full"].mean()) if len(healthy) else None
    )
    record["finite"] = bool(
        np.isfinite(recording_values[:, : len(SCORE_COLUMNS)][
            :, [SCORE_COLUMNS.index(name) for name in ("mse_full", "mse_base", "delta_mu_sat")]
        ]).all()
    )
    return record


def save_preservation(reading: PreservationReading, directory: Any, *, name: str) -> Path:
    """Write one preservation reading into a run directory.

    Args:
        reading: The reading.
        directory: The run directory. Created if absent.
        name: A prefix distinguishing, for example, the frozen baseline's reading from a
            candidate's.

    Returns:
        The directory the two files were written into.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    reading.frame.to_parquet(path / f"{name}_{PRESERVATION_INDEX_FILENAME}", index=False)
    (path / f"{name}_{PRESERVATION_RECORD_FILENAME}").write_text(
        json.dumps(
            {
                "record": reading.record,
                "recordings": reading.recordings.to_dict(orient="records"),
            },
            indent=2, sort_keys=True, default=str,
        ),
        encoding="utf-8",
    )
    return path


def load_preservation_record(directory: Any, *, name: str) -> Dict[str, Any]:
    """Read one preservation record back.

    Args:
        directory: The run directory.
        name: The prefix used when it was written.

    Returns:
        The record.

    Raises:
        FileNotFoundError: If it was never written.
    """
    target = Path(directory) / f"{name}_{PRESERVATION_RECORD_FILENAME}"
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} is missing, so the {name!r} preservation measurement cannot be read back. "
            f"The frozen baseline's reading is taken before the first candidate is fitted."
        )
    return dict(json.loads(target.read_text(encoding="utf-8")).get("record") or {})


# =============================================================================
# The gates
# =============================================================================
@dataclass(frozen=True)
class GateResult:
    """Whether one candidate may enter selection at all, and why.

    A candidate that fails is not selected however good its validation AUROC is: the gates run
    first, so discrimination bought by wrecking the forecast never reaches the comparison.

    Attributes:
        passed: Whether every declared gate held.
        reasons: One line per failed gate, naming the measured values. Empty when it passed.
        warnings: Checks that are reported rather than gated -- healthy-only drift, and a
            fallback rule the run had to apply. A warning never blocks selection; it appears in
            the report beside the number it qualifies.
        record: Every measured and derived value the decision was made from.
    """

    passed: bool
    reasons: List[str]
    warnings: List[str]
    record: Dict[str, Any]


def _relative_increase(baseline: float, candidate: float) -> Optional[float]:
    """The candidate's fractional increase over the baseline, or ``None`` at a zero baseline.

    Args:
        baseline: The frozen model's value.
        candidate: The candidate's.

    Returns:
        $(c - b) / b$, or ``None`` when $b = 0$ -- where the ratio is undefined and the caller
        applies :data:`ZERO_BASELINE_RULE` instead of dividing.
    """
    if baseline == 0.0:
        return None
    return (candidate - baseline) / abs(baseline)


def gate_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    forecast_mse_max_increase: float,
    saturation_max_increase_pp: float,
) -> GateResult:
    r"""Decide whether a candidate passes the declared preservation gates.

    Three gates, all declared before training and all engineering tolerances rather than clinically
    validated criteria:

    1. **Forecast.** Deterministic full-branch MSE at most ``forecast_mse_max_increase`` above the
       frozen baseline's, on the same recordings and the same anchors. A fixed decoder does not
       preserve a forecast when its latent input moves, which is why this is measured rather than
       assumed.
    2. **Saturation.** Posterior-delta saturation on valid support at most
       ``saturation_max_increase_pp`` percentage points above the baseline's. A $\Delta\mu$ pinned
       at its bound is a latent that has stopped carrying a value and started carrying a clamp.
    3. **Finite and not collapsed.** Every reduced score finite, and at least one latent
       coordinate still varying across the subset's recordings.

    **Zero baseline.** A relative tolerance is undefined against a baseline MSE of zero, so the
    gate falls back to :data:`ZERO_BASELINE_RULE` -- the candidate must also be zero -- and records
    which rule it applied instead of dividing and reporting an infinity as a failure.

    Healthy-only drift is measured and **reported**, not gated: the protocol declares one forecast
    tolerance, and applying it a second time to a smaller, noisier stratum would reject candidates
    the declared protocol accepts. It is raised as a warning when it exceeds the same tolerance, so
    a candidate that preserved the adverse group's forecast by spending the healthy group's cannot
    pass quietly.

    Args:
        baseline: The frozen model's record from :func:`preservation_pass`.
        candidate: The candidate's, from the same subset and the same support.
        forecast_mse_max_increase: The declared forecast tolerance, as a fraction.
        saturation_max_increase_pp: The declared saturation tolerance, in percentage points.

    Returns:
        The decision.

    Raises:
        GateEvaluationError: If the baseline itself is unusable, or if the two readings were not
            taken over the same anchors. Neither says anything about the candidate, and reporting
            either as a failed gate would blame the adaptation for a broken measurement.
    """
    baseline_mse = float(baseline.get("mse_full", float("nan")))
    candidate_mse = float(candidate.get("mse_full", float("nan")))
    if not np.isfinite(baseline_mse):
        raise GateEvaluationError(
            f"the baseline forecast MSE is {baseline_mse}, so there is nothing to gate against. "
            f"The frozen model's reading is taken before any candidate is fitted; a non-finite one "
            f"is a broken measurement rather than a property of the adaptation."
        )
    if baseline.get("support_digest") != candidate.get("support_digest"):
        raise GateEvaluationError(
            f"the two readings were taken over different anchors "
            f"({baseline.get('support_digest')} against {candidate.get('support_digest')}): "
            f"{baseline.get('n_retained_anchors')} and {candidate.get('n_retained_anchors')} "
            f"retained anchor(s) over {baseline.get('n_recordings')} and "
            f"{candidate.get('n_recordings')} recording(s). A difference between two populations "
            f"would otherwise be reported as a difference between two models."
        )

    reasons: List[str] = []
    warnings: List[str] = []
    increase = _relative_increase(baseline_mse, candidate_mse)
    rule = ZERO_BASELINE_RULE if increase is None else "relative_increase"
    if increase is None:
        if candidate_mse != 0.0:
            reasons.append(
                f"forecast: baseline MSE is exactly 0, so the relative tolerance is undefined and "
                f"the fallback rule requires the candidate to be 0 too; it is {candidate_mse:.6g}"
            )
    elif not (increase <= float(forecast_mse_max_increase)):
        reasons.append(
            f"forecast: MSE rose {100.0 * increase:.2f}% "
            f"({baseline_mse:.6g} -> {candidate_mse:.6g}), above the declared "
            f"{100.0 * float(forecast_mse_max_increase):.2f}%"
        )

    baseline_sat = float(baseline.get("delta_mu_sat_pp", float("nan")))
    candidate_sat = float(candidate.get("delta_mu_sat_pp", float("nan")))
    saturation_change = candidate_sat - baseline_sat
    if not (saturation_change <= float(saturation_max_increase_pp)):
        reasons.append(
            f"saturation: posterior-delta saturation rose {saturation_change:.3f} percentage "
            f"points ({baseline_sat:.3f} -> {candidate_sat:.3f} pp), above the declared "
            f"{float(saturation_max_increase_pp):.3f} pp"
        )

    if not bool(candidate.get("finite", False)):
        reasons.append(
            "finiteness: at least one reduced forecast or saturation score is not finite, so the "
            "candidate's own measurement cannot be read"
        )
    n_varying = int(candidate.get("n_latent_coordinates_varying", 0))
    if n_varying == 0:
        reasons.append(
            f"collapse: no posterior-mean coordinate varies across the "
            f"{candidate.get('n_recordings')} recording(s) of the gate subset, so the adapted "
            f"latent carries no between-recording information at all"
        )

    healthy_baseline = baseline.get("mse_full_healthy")
    healthy_candidate = candidate.get("mse_full_healthy")
    healthy_increase: Optional[float] = None
    if healthy_baseline is None or healthy_candidate is None:
        warnings.append(
            "healthy-only drift was not measured: the gate subset holds no healthy recording, "
            "which the subset record names as an unavailable stratum"
        )
    else:
        healthy_increase = _relative_increase(
            float(healthy_baseline), float(healthy_candidate)
        )
        if healthy_increase is None:
            warnings.append(
                "healthy-only baseline MSE is exactly 0, so its drift is reported as an absolute "
                f"value ({float(healthy_candidate):.6g}) rather than a ratio"
            )
        elif healthy_increase > float(forecast_mse_max_increase):
            warnings.append(
                f"healthy-only forecast MSE rose {100.0 * healthy_increase:.2f}% "
                f"({float(healthy_baseline):.6g} -> {float(healthy_candidate):.6g}), past the "
                f"forecast tolerance the pooled gate applies. Reported, not gated: the declared "
                f"protocol carries one tolerance, and this stratum is the smaller of the two"
            )

    record = {
        "passed": not reasons,
        "rule": rule,
        "forecast_mse_max_increase": float(forecast_mse_max_increase),
        "saturation_max_increase_pp": float(saturation_max_increase_pp),
        "baseline_mse_full": baseline_mse,
        "candidate_mse_full": candidate_mse,
        "mse_full_increase": increase,
        "baseline_delta_mu_sat_pp": baseline_sat,
        "candidate_delta_mu_sat_pp": candidate_sat,
        "delta_mu_sat_increase_pp": saturation_change,
        "baseline_mse_full_healthy": healthy_baseline,
        "candidate_mse_full_healthy": healthy_candidate,
        "mse_full_healthy_increase": healthy_increase,
        "candidate_finite": bool(candidate.get("finite", False)),
        "n_latent_coordinates_varying": n_varying,
        "support_digest": baseline.get("support_digest"),
        "n_recordings": baseline.get("n_recordings"),
        "n_retained_anchors": baseline.get("n_retained_anchors"),
        "reasons": list(reasons),
        "warnings": list(warnings),
        "tolerances_note": (
            "both tolerances are declared engineering thresholds, fixed before training and not "
            "clinically validated"
        ),
    }
    if reasons:
        logger.warning(f"preservation gate failed: {'; '.join(reasons)}")
    else:
        logger.info(
            f"preservation gate passed: forecast {record['mse_full_increase']}, saturation "
            f"{saturation_change:+.3f} pp"
        )
    for line in warnings:
        logger.warning(f"preservation: {line}")
    return GateResult(
        passed=not reasons, reasons=reasons, warnings=warnings, record=record
    )


def nll_convergence(small: Mapping[str, Any], large: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare the marginal NLL at two draw counts.

    The optional check §6 asks for when a conclusion turns on a small NLL difference: the marginal
    predictive score is a log of an average likelihood, so it falls as draws are added, and a
    difference between two models smaller than the difference between two draw counts is not a
    difference the estimator can see.

    Args:
        small: A record measured at the smaller draw count.
        large: One measured at the larger, on the same subset.

    Returns:
        The two values, their movement, and whether the larger count changed the full branch by
        less than it changed the gap between branches -- which is the comparison that matters,
        since the gap is what any conclusion would rest on.

    Raises:
        GateEvaluationError: If the two were not measured over the same anchors.
    """
    if small.get("support_digest") != large.get("support_digest"):
        raise GateEvaluationError(
            f"the two draw counts were measured over different anchors "
            f"({small.get('support_digest')} against {large.get('support_digest')}), so their "
            f"difference is not a convergence statement."
        )
    small_full, large_full = float(small.get("nll_full", float("nan"))), float(
        large.get("nll_full", float("nan"))
    )
    small_gap = small_full - float(small.get("nll_base", float("nan")))
    large_gap = large_full - float(large.get("nll_base", float("nan")))
    return {
        "draws_small": small.get("mc_draws"),
        "draws_large": large.get("mc_draws"),
        "nll_full_small": small_full,
        "nll_full_large": large_full,
        "nll_full_movement": large_full - small_full,
        "nll_gap_small": small_gap,
        "nll_gap_large": large_gap,
        "nll_gap_movement": large_gap - small_gap,
        "support_digest": small.get("support_digest"),
        "note": (
            "the marginal predictive score falls as draws are added; a between-model difference "
            "smaller than the movement reported here is below what this estimator resolves"
        ),
    }



# =============================================================================
# Recording-level metrics
# =============================================================================
def auroc(labels: Any, scores: Any) -> float:
    """Area under the ROC curve over one observation per recording.

    ``sklearn.metrics.roc_auc_score`` rather than a rank formula written here: the two agree only
    if the local one handles ties by mid-rank, and a silently different tie rule would move every
    number this pilot reports by an amount nobody could trace.

    Args:
        labels: Binary outcomes, one per recording.
        scores: Any monotone score, typically the classifier's logit.

    Returns:
        The AUROC, or ``nan`` when the labels carry one class only -- where discrimination is not
        small, it is undefined, and a zero or a half would both read as a measurement.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64).reshape(-1)))


def binary_cross_entropy(labels: Any, logits: Any, *, balanced: bool = False) -> float:
    r"""Mean binary cross-entropy from logits.

    $$\operatorname{BCE}(\ell, y) = \log(1 + e^{\ell}) - y\,\ell,$$

    evaluated through ``logaddexp`` so a confident logit does not overflow before it is compared.

    Args:
        labels: Binary outcomes.
        logits: The classifier's logits, **not** probabilities.
        balanced: ``True`` averages the two class means, which is the fitting objective's own
            reduction. ``False`` -- the default -- is the plain mean at the split's natural
            prevalence, which is what validation and test are read at.

    Returns:
        The mean loss in nats, or ``nan`` on an empty input.
    """
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return float("nan")
    per_example = np.logaddexp(0.0, values) - y * values
    if not balanced:
        return float(per_example.mean())
    means = [per_example[y == label].mean() for label in (0.0, 1.0) if np.any(y == label)]
    return float(np.mean(means)) if means else float("nan")


def balanced_accuracy(labels: Any, logits: Any, *, threshold: float) -> float:
    """Mean of sensitivity and specificity at one threshold.

    Balanced rather than plain accuracy because the positive class is the scarce one: a rule that
    called every recording healthy would score well on accuracy and 0.5 here, which is what it is
    worth.

    Args:
        labels: Binary outcomes.
        logits: The classifier's logits.
        threshold: Predict positive where ``logit >= threshold``.

    Returns:
        The balanced accuracy, or ``nan`` when a class is absent and its rate is undefined.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    predicted = np.asarray(logits, dtype=np.float64).reshape(-1) >= float(threshold)
    rates = []
    for label in (0, 1):
        present = y == label
        if not present.any():
            return float("nan")
        rates.append(float((predicted[present] == bool(label)).mean()))
    return float(np.mean(rates))


def select_threshold(labels: Any, logits: Any) -> Dict[str, Any]:
    """Choose the decision threshold that maximises balanced accuracy, deterministically.

    **On validation, and only on validation.** A threshold chosen on the held-out split would make
    the reported balanced accuracy an in-sample number wearing a held-out label.

    The candidates are the ROC curve's own thresholds, so every distinct split of the scores is
    considered exactly once and ties among the scores are handled by the same code that computes
    the AUROC. ``roc_curve``'s leading infinite threshold -- the rule that calls nothing positive --
    is dropped: it is a legal split, but a threshold no finite score can ever meet is not a
    decision rule worth carrying into the report.

    Ties are broken by the **lower** threshold. Two thresholds with the same balanced accuracy
    differ in nothing this pilot measures, and the lower one predicts positive more often, so a
    tie resolves towards sensitivity rather than towards whichever candidate the curve happened to
    list first.

    Args:
        labels: Binary outcomes on the validation split.
        logits: The classifier's validation logits.

    Returns:
        The threshold, the balanced accuracy it achieves, how many candidates tied with it, and the
        rule that broke the tie.

    Raises:
        GateEvaluationError: If validation carries one class only, where balanced accuracy -- and
            therefore the choice -- is undefined.
    """
    from sklearn.metrics import roc_curve

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(set(y.tolist())) < 2:
        raise GateEvaluationError(
            f"the validation split carries one binary class ({int((y == 0).sum())} healthy, "
            f"{int((y == 1).sum())} adverse), so balanced accuracy and the threshold that "
            f"maximises it are both undefined."
        )
    false_positive, true_positive, thresholds = roc_curve(y, values)
    finite = np.isfinite(thresholds)
    scores = 0.5 * (true_positive + (1.0 - false_positive))
    scores, thresholds = scores[finite], thresholds[finite]

    best = float(scores.max())
    tied = thresholds[scores >= best - 1e-12]
    threshold = float(tied.min())
    return {
        "threshold": threshold,
        "balanced_accuracy": best,
        "n_candidates": int(thresholds.size),
        "n_tied": int(tied.size),
        "tie_rule": "lowest threshold among the maxima, which resolves towards sensitivity",
        "population": "validation only",
    }



# =============================================================================
# One observation per held-out recording
# =============================================================================
#: Metrics reported for every model on every population, in a fixed order.
METRIC_NAMES: Tuple[str, ...] = ("auroc", "average_precision", "balanced_accuracy")

#: Confidence level of every interval this package reports.
CONFIDENCE = 0.95


def average_precision(labels: Any, scores: Any) -> float:
    """Average precision, reported beside the prevalence that sets its chance level.

    Unlike AUROC, average precision has no fixed baseline: on a cohort with 12% adverse outcomes a
    useless ranker scores about 0.12, and quoting the number without the prevalence beside it says
    nothing. :func:`recording_metrics` always reports the pair.

    Args:
        labels: Binary outcomes, one per recording.
        scores: Any monotone score.

    Returns:
        The average precision, or ``nan`` when the labels carry one class only.
    """
    from sklearn.metrics import average_precision_score

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(scores, dtype=np.float64).reshape(-1)))


def recording_metrics(labels: Any, logits: Any, *, threshold: float) -> Dict[str, Any]:
    """The recording-level readout of one model on one population.

    One observation per recording, always: the bag reduction upstream is what makes that true, and
    every count here is a count of recordings rather than of segments or anchors.

    Args:
        labels: Binary outcomes.
        logits: The classifier's logits.
        threshold: The decision threshold, chosen on validation and never on this population when
            this population is the test split.

    Returns:
        AUROC, average precision, balanced accuracy at the threshold, the prevalence average
        precision must be read against, and the counts behind all of them.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    return {
        "auroc": auroc(y, values),
        "average_precision": average_precision(y, values),
        "balanced_accuracy": balanced_accuracy(y, values, threshold=float(threshold)),
        "bce": binary_cross_entropy(y, values),
        "threshold": float(threshold),
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "n_recordings": int(y.size),
        "n_healthy": int((y == 0).sum()),
        "n_adverse": int((y == 1).sum()),
        "chance_auroc": 0.5,
        "chance_average_precision": float(y.mean()) if y.size else float("nan"),
    }


# =============================================================================
# The paired, outcome-stratified cluster bootstrap
# =============================================================================
def bootstrap_units(
    guids: Sequence[str], patients: Optional[Mapping[str, str]] = None
) -> Tuple[List[str], Dict[str, List[int]], str]:
    """Group the rows into the units a resample draws.

    Patients where a mapping exists, GUIDs otherwise. It matters: repeated recordings of one
    patient are not independent observations, and resampling GUIDs when several belong to one
    delivery reports an interval narrower than the data supports. When no mapping exists the
    grouping is GUID-only and **the report says so** rather than implying independence that was
    never established.

    Args:
        guids: One GUID per row, in row order.
        patients: GUID -> patient/delivery identifier, or ``None``.

    Returns:
        ``(units, rows by unit, grouping)``, where ``grouping`` names what was resampled.
    """
    rows: Dict[str, List[int]] = {}
    for index, guid in enumerate(guids):
        unit = str(guid) if not patients else str(patients.get(str(guid), str(guid)))
        rows.setdefault(unit, []).append(index)
    grouping = "patient" if patients else "guid"
    return sorted(rows), rows, grouping


def _strata(
    units: Sequence[str], rows: Mapping[str, Sequence[int]], labels: np.ndarray
) -> Dict[Tuple[int, ...], List[str]]:
    """Partition the units by the set of outcomes they carry.

    A unit is a GUID or a patient, and a patient contributing both a healthy and an adverse
    recording forms a stratum of its own rather than being forced into one of the two: assigning it
    to a class by majority would resample it as though it were something it is not.
    """
    strata: Dict[Tuple[int, ...], List[str]] = {}
    for unit in units:
        signature = tuple(sorted({int(labels[index]) for index in rows[unit]}))
        strata.setdefault(signature, []).append(unit)
    return strata


def paired_bootstrap(
    labels: Any,
    columns: Mapping[str, Any],
    *,
    guids: Sequence[str],
    thresholds: Mapping[str, float],
    patients: Optional[Mapping[str, str]] = None,
    resamples: int = 1000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> Dict[str, Any]:
    r"""Outcome-stratified paired cluster bootstrap over recordings, or patients where known.

    **The same resampled units are used for every model.** That is the whole of what "paired"
    means here: the frozen model and the adapted one are re-scored on one draw of recordings, so
    the interval on their difference reflects the difference and not two independent samples of the
    cohort. A separate bootstrap per model would give two intervals whose overlap says nothing
    about the paired change.

    Stratified by the units' outcomes, so every draw holds both classes in the proportions the
    cohort has and a resample cannot silently become one on which AUROC is undefined. A draw that
    does turn out undefined -- possible only for a degenerate stratum -- is **counted and excluded**
    from the quantiles rather than dropped silently, and the count is reported beside the interval.

    Percentile intervals, as :func:`teb_vae.lag_attn.eval.stats.bootstrap_ci` reports them
    elsewhere in this repository, so "significant" means here what it means there. The point
    estimate is the metric on the **full** sample rather than the mean of the draws: the draws
    estimate the spread, not the value.

    Args:
        labels: Binary outcomes, one per recording.
        columns: ``{model name: logits}``, all aligned with ``labels`` and with each other.
        guids: One GUID per row, for the clustering.
        thresholds: Each model's own validation threshold.
        patients: GUID -> patient, or ``None`` for GUID-only grouping.
        resamples: Draws. The protocol's 1000.
        seed: Seeds the draws, so an interval is reproducible from the record alone.
        confidence: Interval coverage.

    Returns:
        Per model and per metric an interval; per **ordered pair** of models the interval on the
        paired difference; and the record of what was resampled.

    Raises:
        GateEvaluationError: If the columns are not aligned with the labels, or if fewer than
            :data:`MIN_UNITS` units survive -- below which a bootstrap reproduces the sample
            rather than estimating its spread.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = {
        name: np.asarray(values, dtype=np.float64).reshape(-1)
        for name, values in columns.items()
    }
    for name, values in scores.items():
        if values.size != y.size:
            raise GateEvaluationError(
                f"model {name!r} supplies {values.size} score(s) for {y.size} label(s). Every "
                f"model in a paired comparison is scored on the same recordings, in the same "
                f"order; a length mismatch means two populations are being compared as two models."
            )
    if len(guids) != y.size:
        raise GateEvaluationError(
            f"{len(guids)} GUID(s) for {y.size} label(s): the clustering must name the recording "
            f"every row came from."
        )

    units, rows, grouping = bootstrap_units(guids, patients)
    if len(units) < MIN_UNITS:
        raise GateEvaluationError(
            f"only {len(units)} {grouping}(s) survive to the held-out comparison; below "
            f"{MIN_UNITS} a bootstrap reproduces the sample rather than estimating its spread. "
            f"Report the counts and the point estimates without intervals."
        )
    strata = _strata(units, rows, y)

    def _metrics(index: np.ndarray) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                metric: value
                for metric, value in recording_metrics(
                    y[index], values[index], threshold=float(thresholds[name])
                ).items()
                if metric in METRIC_NAMES
            }
            for name, values in scores.items()
        }

    full = np.arange(y.size)
    point = _metrics(full)
    pairs = [
        (before, after)
        for position, before in enumerate(scores)
        for after in list(scores)[position + 1:]
    ]
    draws: Dict[str, Dict[str, List[float]]] = {
        name: {metric: [] for metric in METRIC_NAMES} for name in scores
    }
    deltas: Dict[str, Dict[str, List[float]]] = {
        f"{after} - {before}": {metric: [] for metric in METRIC_NAMES}
        for before, after in pairs
    }
    generator = np.random.default_rng(int(seed))
    n_undefined = 0

    for _draw in range(int(resamples)):
        drawn: List[int] = []
        for members in strata.values():
            picked = generator.integers(0, len(members), size=len(members))
            for position in picked.tolist():
                drawn.extend(rows[members[position]])
        index = np.asarray(drawn, dtype=np.int64)
        values = _metrics(index)
        if any(
            not np.isfinite(values[name][metric])
            for name in values for metric in METRIC_NAMES
        ):
            n_undefined += 1
            continue
        for name in scores:
            for metric in METRIC_NAMES:
                draws[name][metric].append(values[name][metric])
        for before, after in pairs:
            for metric in METRIC_NAMES:
                deltas[f"{after} - {before}"][metric].append(
                    values[after][metric] - values[before][metric]
                )

    alpha = 1.0 - float(confidence)

    def _interval(sample: Sequence[float], estimate: float) -> Dict[str, Any]:
        finite = np.asarray(sample, dtype=np.float64)
        if finite.size < MIN_UNITS:
            return {
                "point": float(estimate),
                "lo": float("nan"),
                "hi": float("nan"),
                "n_draws": int(finite.size),
                "note": (
                    f"only {finite.size} usable draw(s); no interval is reported rather than one "
                    f"decided by two order statistics of a tiny sample"
                ),
            }
        return {
            "point": float(estimate),
            "lo": float(np.quantile(finite, alpha / 2.0)),
            "hi": float(np.quantile(finite, 1.0 - alpha / 2.0)),
            "n_draws": int(finite.size),
        }

    record = {
        "models": {
            name: {
                metric: _interval(draws[name][metric], point[name][metric])
                for metric in METRIC_NAMES
            }
            for name in scores
        },
        "paired": {
            label: {
                metric: _interval(
                    deltas[label][metric],
                    point[label.split(" - ")[0]][metric] - point[label.split(" - ")[1]][metric],
                )
                for metric in METRIC_NAMES
            }
            for label in deltas
        },
        "grouping": grouping,
        "n_units": len(units),
        "n_recordings": int(y.size),
        "n_strata": len(strata),
        "strata": {
            "/".join(str(value) for value in signature): len(members)
            for signature, members in sorted(strata.items())
        },
        "resamples": int(resamples),
        "n_undefined_draws": n_undefined,
        "seed": int(seed),
        "confidence": float(confidence),
        "method": (
            "outcome-stratified paired percentile bootstrap over "
            f"{grouping}s; the same drawn units score every model"
        ),
        "note": (
            "the interval describes uncertainty conditional on this fold and this fitting seed, "
            "not variation across training runs"
        ),
    }
    if grouping == "guid":
        record["grouping_disclosure"] = (
            "no patient/delivery mapping was supplied, so recordings are resampled as if "
            "independent; repeated recordings of one patient would make this interval too narrow"
        )
    logger.info(
        f"paired bootstrap: {resamples} draw(s) over {len(units)} {grouping}(s) in "
        f"{len(strata)} stratum/strata; {n_undefined} undefined draw(s)"
    )
    return record


# =============================================================================
# Descriptive strata, never refitted
# =============================================================================
def subgroup_rows(frame: pd.DataFrame, subgroup: str) -> np.ndarray:
    """The rows one prespecified stratum selects.

    Every stratum here restricts the **same** fitted classifier's scores to a subset of recordings.
    None of them refits anything: a per-subgroup fit would be a model chosen after seeing which
    subgroup looked better, which is selection on the held-out split under another name.

    Args:
        frame: One row per recording, carrying the outcome and the descriptive flags.
        subgroup: One of ``'all'``, ``'healthy_bg'`` (healthy-BG controls against every adverse
            recording), ``'cs'``, ``'no_cs'``, ``'acidosis'`` or ``'hie'``. The last two contrast
            that class against the **same** healthy controls.

    Returns:
        A boolean mask over ``frame``'s rows.

    Raises:
        GateEvaluationError: On an unknown stratum name.
    """
    outcome = np.asarray(frame[data.OUTCOME_COLUMN], dtype=np.float64)
    adverse = outcome == 1
    healthy = outcome == 0
    if subgroup == "all":
        return np.ones(len(frame), dtype=bool)
    if subgroup == "healthy_bg":
        blood_gas = np.asarray(
            [data._flag_is(value, True) for value in frame.get("bg_label", [])], dtype=bool
        )
        return adverse | (healthy & blood_gas)
    if subgroup in ("cs", "no_cs"):
        wanted = subgroup == "cs"
        return np.asarray(
            [data._flag_is(value, wanted) for value in frame.get("cs_label", [])], dtype=bool
        )
    if subgroup in ("acidosis", "hie"):
        names = frame.get(labels.CLASS_COLUMN)
        if names is None:
            return np.zeros(len(frame), dtype=bool)
        matches = np.asarray([str(value) == subgroup for value in names], dtype=bool)
        return healthy | matches
    raise GateEvaluationError(
        f"unknown subgroup {subgroup!r}. The strata are prespecified: "
        f"{', '.join(SUBGROUPS)}."
    )


#: The descriptive strata, reported for every model and never used to choose one.
SUBGROUPS: Tuple[str, ...] = ("all", "healthy_bg", "cs", "no_cs", "acidosis", "hie")


def subgroup_table(
    frame: pd.DataFrame,
    columns: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
    subgroups: Sequence[str] = SUBGROUPS,
) -> pd.DataFrame:
    """Every model's readout on every prespecified stratum, with the counts beside it.

    Training is binary and the headline stays binary. These rows exist so a reader can see whether
    a result rests on one subgroup, not so the best one can be promoted afterwards -- which is why
    they are produced together, in a fixed order, for every model at once.

    A stratum too small to estimate discrimination reports its counts and ``nan`` metrics rather
    than a number, because a two-recording AUROC is not a small measurement but an undefined one.

    Args:
        frame: One row per recording: outcome, class name, CS and blood-gas flags.
        columns: ``{model name: logits}``, aligned with ``frame``.
        thresholds: Each model's validation threshold.
        subgroups: Which strata to report.

    Returns:
        One row per ``(subgroup, model)``.
    """
    outcome = np.asarray(frame[data.OUTCOME_COLUMN], dtype=np.int64)
    rows: List[Dict[str, Any]] = []
    for subgroup in subgroups:
        mask = subgroup_rows(frame, subgroup)
        for name, logits in columns.items():
            values = np.asarray(logits, dtype=np.float64).reshape(-1)[mask]
            measured = recording_metrics(
                outcome[mask], values, threshold=float(thresholds[name])
            )
            rows.append({
                "subgroup": subgroup,
                "model": name,
                "estimable": bool(
                    measured["n_healthy"] >= 1 and measured["n_adverse"] >= 1
                ),
                **measured,
            })
    return pd.DataFrame(rows)


def coverage_contrast(recordings: pd.DataFrame) -> pd.DataFrame:
    """Compare time coverage and signal availability between the two outcome groups.

    The control §7.3 asks for: a cloud separated mainly by ascertainment or by how much signal
    survived is not an outcome result, and the only way to see that is to put the two groups'
    coverage side by side before reading any separation.

    Args:
        recordings: The recording table, after eligibility.

    Returns:
        One row per binary outcome with the recording count, the median and quartiles of the last
        observed time before delivery, and the median segment and anchor counts inside the
        supervised window. Every entry is a description; none of them adjusts anything.
    """
    rows: List[Dict[str, Any]] = []
    for value in (0, 1):
        group = recordings[recordings[data.OUTCOME_COLUMN] == value]
        entry: Dict[str, Any] = {
            data.OUTCOME_COLUMN: value,
            "class": "healthy" if value == 0 else "adverse",
            "n_recordings": int(len(group)),
        }
        for column in ("last_anchor_hours", "n_late_segments", "n_late_anchors"):
            values = np.asarray(group.get(column, []), dtype=np.float64)
            values = values[np.isfinite(values)]
            entry[f"{column}_median"] = float(np.median(values)) if values.size else float("nan")
            entry[f"{column}_q1"] = (
                float(np.quantile(values, 0.25)) if values.size else float("nan")
            )
            entry[f"{column}_q3"] = (
                float(np.quantile(values, 0.75)) if values.size else float("nan")
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def control_disclosure(*, n_control_fits: int, prior_probe: bool) -> Dict[str, Any]:
    """State what the controls in this run do and do not establish.

    Two claims this package refuses to let a report make by omission:

    * **One shuffled-label fit is not a permutation p-value.** It is a leakage and overfitting
      sanity check: a single draw from the null, run through the same selection. A p-value would
      need many full refits *including* selection, and calling one fit by that name would put a
      significance claim on a sample of size one.
    * **Better ``mu_post`` discrimination alone does not establish UP-specific information.** The
      frozen ``mu_prior`` probe is what makes any combined-branch claim admissible at all, and when
      it is switched off the claim is excluded rather than left implied.

    Args:
        n_control_fits: How many shuffled-label fits this run performed.
        prior_probe: Whether the frozen prior probe ran.

    Returns:
        The disclosure record, carried into the report verbatim.
    """
    return {
        "n_shuffled_label_fits": int(n_control_fits),
        "permutation_p_value": False,
        "shuffled_label_note": (
            f"{int(n_control_fits)} shuffled-label fit(s): a leakage and overfitting sanity check, "
            f"not a permutation p-value, which would require many full refits including selection"
        ),
        "prior_probe_run": bool(prior_probe),
        "combined_branch_claim_supported": bool(prior_probe),
        "prior_probe_note": (
            "the frozen mu_prior probe ran, so a combined-branch comparison is reportable; better "
            "mu_post discrimination alone still does not establish UP-specific information, which "
            "would additionally need source-null and shuffle controls with matched forecasts"
            if prior_probe else
            "the frozen mu_prior probe was switched off, so this run makes no claim that the "
            "combined branch helps: there is nothing to compare the posterior's discrimination "
            "against"
        ),
    }


__all__ = [
    "GATE_STRATA",
    "GATE_SUBSET_FILENAME",
    "PRESERVATION_INDEX_FILENAME",
    "PRESERVATION_RECORD_FILENAME",
    "SCORE_COLUMNS",
    "ZERO_BASELINE_RULE",
    "GateEvaluationError",
    "GateResult",
    "PreservationReading",
    "CONFIDENCE",
    "METRIC_NAMES",
    "MIN_UNITS",
    "SUBGROUPS",
    "auroc",
    "average_precision",
    "balanced_accuracy",
    "bootstrap_units",
    "control_disclosure",
    "coverage_contrast",
    "binary_cross_entropy",
    "gate_decision",
    "gate_subset",
    "load_gate_subset",
    "load_preservation_record",
    "nll_convergence",
    "outcome_map",
    "paired_bootstrap",
    "preservation_pass",
    "recording_metrics",
    "save_gate_subset",
    "save_preservation",
    "select_threshold",
    "subgroup_rows",
    "subgroup_table",
]
