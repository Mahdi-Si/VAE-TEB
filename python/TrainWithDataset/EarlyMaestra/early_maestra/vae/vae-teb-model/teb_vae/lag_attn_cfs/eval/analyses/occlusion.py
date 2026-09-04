r"""When did the source matter? Removed a band of it and measure what the forecast lost.

Every other lag readout in this pipeline is **observational**: the attention weights say where the
model looked, the KL attribution says how much the belief moved, and both are properties of a
forward the source was fully present in. Neither answers the question a physiological delay is a
claim about -- *would the forecast have been worse without the source at that lag* -- and on this
geometry that gap is not academic. The pooled lag argmax sits on the window's near censoring edge
on every arm this family has measured, so the observational readouts are pinned by the geometry
before the model gets a vote.

This analysis intervenes instead. For each configured band of lags $[\ell_{\mathrm{lo}},
\ell_{\mathrm{hi}}]$ it zeroes the source stream on the steps
$[t - \ell_{\mathrm{hi}},\, t - \ell_{\mathrm{lo}}]$ of the scored anchor $t$, re-encodes, re-poses
the posterior, re-decodes, and reports the change in the block score **resolved by horizon step**:

$$\Delta^{\mathrm{occ}}_{\mathrm{band}}(\tau) \;=\;
D^{\mathrm{occ}}_{\mathrm{band}}(\tau) \;-\; D^{\mathrm{ref}}(\tau),$$

with $D(\tau)$ the masked block score summed over the kept channels at horizon step $\tau$. A
positive value is the forecast getting worse without that band, which is the band mattering.

**Why $\tau$ is the resolution axis.** The lag axis is censored at both ends by the alignment
geometry, and no model-side change reaches that. The horizon axis is not: a source band that
informs the first predicted step and not the last is a statement the window can carry whatever the
references are set to, and it is the axis on which a physiological delay is expressible here.

**Zero is the intervention, and it is the channel mean.** The loader's normalisation constants are
accumulated *excluding* the causal warm-up region, so zero is the mean of the region the model
actually reads. The occluded band is therefore uninformative rather than adversarial -- the same
argument the source-null control rests on, and the reason this analysis does not invent its own
fill value.

**The clock does not move, and that is the confound this exists to avoid.** The availability
announcement $m_t$ is built inside the input adapter from registered buffers -- a function of $t$
and the resolved warm-up vector -- and no value on the stream reaches it. So the intervention
changes what the source *says* and not when it *arrived*; an intervention that moved both would
measure the availability clock under the name of the source. It is checked rather than asserted:
every arm records the max-abs change in the adapter's own announcement (exactly zero, by
construction) and the fraction of occluded positions that actually held a value.

**The stream is occluded after the channel gate.** The gate shifts each channel onto the run's
common clock, so a band of gated steps is one lag range for every kept channel at once. The same
band applied before the gate would land at $\ell + d_c$ for channel $c$ and re-smear precisely the
axis the alignment exists to un-smear.

**One anchor per segment, and it is not an economy.** The source pathway has memory -- unbounded
under the deep encoder, tens of steps under the convolution stem -- so a band occluded relative to
anchor $a$ contaminates the state of every anchor after it. Scoring a second anchor in the same
forward would attribute one anchor's loss to another's band. The anchor is drawn once per segment
from a seeded generator, uniformly over the anchors the forward marked valid, and is held fixed
across every band **and** the reference arm: that is what makes the difference paired.

**Common random numbers.** The two arms differ by one band of source values and by nothing else,
including the reparameterisation draw: the generator is reseeded to the same value before each
arm, so the latent noise is identical and the difference carries the band alone. Without it the
per-anchor difference would be dominated by the draw at every band a model barely uses.

**How much of a band was live is reported beside its delta.** A band reaching into the warm-up
region occludes steps that the availability mechanism had already zeroed, so its delta is zero for
a reason that is about the geometry rather than about the source. The live fraction is what
separates "the source did not matter there" from "there was no source there to remove".
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry, per_recording_means
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    batch_field,
    batch_guids,
    batch_size_of,
    masked_raw_block_per_horizon_step,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets.controls import occluded_forward_outputs
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "occlusion"

#: What it writes. The per-recording frame's name and its per-band columns are read by the grouped
#: fan-out, so both are a contract rather than a filename and a label.
PER_RECORDING_FILENAME = "occlusion_per_recording.csv"
PER_HORIZON_FILENAME = "occlusion_per_horizon.csv"
SUMMARY_FILENAME = "occlusion_summary.csv"
HORIZON_FIGURE = "occlusion_horizon_delta"

#: The interventional deltas placed on the two clinical clocks. **Descriptive only** -- see
#: :data:`CLOCK_NOTE` for why there is no test on this page and why that is the honest choice at
#: this analysis's cap.
CLOCK_FILENAME = "occlusion_clocks.csv"
CLOCK_FIGURE = "occlusion_clock_delta"

#: Why the clock page carries no inference, in the record rather than left to be noticed.
CLOCK_NOTE = (
    "descriptive only: no Kruskal-Wallis, no Holm correction and no new family. This analysis "
    "scores one anchor per segment and is capped in segments, so a half-hour window holds tens of "
    "them at best and most (class, window) cells fall below the minimum group size a test needs. "
    "Reporting per-window means and quartiles is what the data supports; a p-value here would be "
    "a correction over cells that mostly could not be tested. Read it beside "
    "lag_kld_scaled's band trajectories, which are the observational answer to the same question"
)

#: The join key. ``guid`` alone does not identify a segment -- a recording contributes many -- so
#: the pair is what places a delta on a clock. Both sides read ``epoch`` through
#: ``metrics.batch_field`` and the same float64 cast, which is what makes an equality join exact.
JOIN_KEYS: Tuple[str, ...] = ("guid", "epoch")

#: ``eval_config.caps`` name bounding how many segments this analysis re-encodes.
#:
#: Absence means **no bound**, matching the oracle's cap and against the retention caps' opt-in
#: convention: a missing retention cap costs a figure, while a missing bound here would mean
#: "measure nothing", which is not a cheaper measurement.
CAP_NAME = "occlusion"

#: Offset applied to ``eval_config.seed`` for this analysis's two generators. Distinct from every
#: offset the runner uses, so the anchor draw and the reparameterisation noise here are their own
#: streams rather than a replay of the collection pass's.
_SEED_OFFSET_ANCHOR = 11
_SEED_OFFSET_NOISE = 12

#: The unit every delta here is in: the block score is a sum over the kept channels at one horizon
#: step, in nats, so a difference of two of them is nats per anchor per horizon step.
NATS_PER_ANCHOR_STEP = "nats per anchor per horizon step"

#: The statement that travels with every delta, because it bounds what the number can be read as.
OCCLUSION_CAVEAT = (
    "the occlusion zeroes the source's VALUES in the band and leaves the availability "
    "announcement untouched, so the delta is what the source's content at those lags was worth "
    "and not what its arrival time was worth. It is measured at one anchor per segment, because "
    "the source pathway has memory and a second anchor in the same forward would carry the first "
    "anchor's occlusion. A band whose live fraction is small was already empty before the "
    "intervention, and its delta says nothing about the source."
)


def band_mask(
    anchors: torch.Tensor, band: Tuple[int, int], sequence_length: int
) -> torch.Tensor:
    r"""Which source steps a band occludes, per sample, as a $(B, T)$ boolean.

    Step $s$ is in sample $b$'s band exactly when $a_b - \ell_{\mathrm{hi}} \le s \le
    a_b - \ell_{\mathrm{lo}}$, so the band is anchored to that sample's own scored anchor and the
    same lag range is removed from every sample whatever anchor it drew. Steps below zero fall off
    the front of the sequence and are simply absent, which is the honest state: the recording does
    not reach that far back and the live fraction reports it.

    Args:
        anchors: The scored anchor per sample, $(B,)$.
        band: The inclusive ``(lo, hi)`` lag pair.
        sequence_length: $T$, the source stream's step axis.

    Returns:
        A $(B, T)$ boolean, true where the source is to be zeroed.
    """
    low, high = int(band[0]), int(band[1])
    steps = torch.arange(int(sequence_length), device=anchors.device)[None, :]
    origin = anchors.to(torch.long)[:, None]
    return (steps >= origin - high) & (steps <= origin - low)


def choose_anchors(
    anchor_valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    r"""Draw one valid anchor column per sample, uniformly.

    Uniform over the columns the forward marked valid rather than a fixed position, because a fixed
    one would measure every band at one phase of every segment -- and the far bands of an early
    anchor reach into the warm-up region while the same bands of a late anchor do not, so a fixed
    phase would decide the live fraction rather than measure it.

    Args:
        anchor_valid: The forward's own $(B, A_{\max})$ validity flags.
        generator: The seeded generator, so the draw is reproducible from the run's own seed.

    Returns:
        One column index per sample, $(B,)$, into the anchor axis. A sample with no valid anchor
        at all yields column $0$; the caller drops it on the mask, which is already zero there.
    """
    weights = anchor_valid.to(torch.float32)
    # A row of zeros would make ``multinomial`` raise rather than return a column, and a sample
    # with no valid anchor is a real state on a short recording. Its column is arbitrary because
    # the forecast mask zeroes it anyway.
    empty = weights.sum(dim=1) <= 0.0
    weights = torch.where(empty[:, None], torch.ones_like(weights), weights)
    return torch.multinomial(weights, num_samples=1, generator=generator).squeeze(1)


def _horizon_scores(
    model: Any,
    outputs: Dict[str, torch.Tensor],
    *,
    target_features: torch.Tensor,
    weight: torch.Tensor,
    anchors: torch.Tensor,
    anchor_valid: torch.Tensor,
    likelihood: str,
) -> torch.Tensor:
    r"""One arm's masked block score at the scored anchor, resolved by horizon step.

    The target and the mask are rebuilt through the model's own
    ``_build_forecast_target`` and the anchored
    :func:`~teb_vae.lag_attn_rws.nets.raw_masks.forecast_mask` at the model's own ``coverage_floor``
    -- the same two functions the collection pass uses -- so the arms are scored over exactly the
    coefficients the rest of the run scored, at one anchor instead of all of them.

    Args:
        model: The rebuilt net.
        outputs: The arm's forward dict, read for ``mu_full`` and ``logvar_full``.
        target_features: The stored target stream $(B, T, c_y)$.
        weight: The decimated validity signal $(B, T)$.
        anchors: The scored anchor per sample as a $(B, 1)$ column.
        anchor_valid: Its validity, $(B, 1)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        $(B, H)$ in float64: the per-horizon-step block score at that anchor, zero wherever the
        mask dropped the step.
    """
    target = model._build_forecast_target(target_features, anchors)
    mask, _coverage = forecast_mask(
        weight,
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchors,
        anchor_valid=anchor_valid,
    )
    per_tau = masked_raw_block_per_horizon_step(
        outputs["mu_full"], target, mask, likelihood=likelihood, logvar=outputs["logvar_full"]
    )
    return per_tau[:, 0, :].to(torch.float64)


def _announcement(model: Any) -> Optional[torch.Tensor]:
    """Return the source adapter's availability announcement, or ``None`` where it builds none.

    Absent rather than zero on an unaligned arm whose delay vector is trivial: the adapter registers
    the buffer only when it has something to announce, and a zeros tensor standing in for it would
    make an invariance check pass on a mechanism that does not exist.

    Args:
        model: The rebuilt net.

    Returns:
        The ``(T, C_kept)`` buffer, or ``None``.
    """
    adapter = getattr(model, "source_adapter", None)
    availability = getattr(adapter, "availability", None)
    return availability if isinstance(availability, torch.Tensor) else None


@torch.no_grad()
def _epochs(batch: Any) -> np.ndarray:
    """The per-segment start time in seconds, as ``float64``, or all-``NaN`` when absent.

    The same read ``collect.py`` makes for its own tables, through the public accessor rather than
    that module's private one: the value must be *bit-identical* on both sides, because the join
    below is a float equality. Both go through ``batch_field`` and the same ``float64`` cast, so
    it is.

    All-``NaN`` and not zero when the batch carries no ``epoch``: zero would place every segment
    at the moment of delivery, which is a coordinate rather than an absence.

    Args:
        batch: One batch.

    Returns:
        A $(B,)$ ``float64`` array.
    """
    size = batch_size_of(batch)
    values = batch_field(batch, "epoch")
    if values is None:
        return np.full(size, np.nan, dtype=np.float64)
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().to(torch.float64).numpy().reshape(-1)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape[0] != size:
        # The batch is the authority on how many samples there are, always.
        raise ValueError(
            f"the batch carries {array.shape[0]} epoch(s) for {size} sample(s); a column of the "
            f"wrong length would silently misalign every segment's clock coordinate."
        )
    return array


def collect_batch(
    task: Any,
    batch: Any,
    *,
    bands: Dict[str, Tuple[int, int]],
    seed: int,
) -> Dict[str, Any]:
    r"""Score every band and the reference arm on one batch, at one anchor per sample.

    The reference arm runs through :func:`occluded_forward_outputs` with no occlusion rather than
    being read off the matched forward. That is deliberate: the matched forward decodes every
    anchor and draws its latent from the global stream, so a difference against it would carry the
    draw and the decode geometry as well as the band. Running both arms through one function with
    one reseeded generator leaves the band as the only thing that moved.

    Under ``no_grad`` at the outermost frame, which the two controls' own decorators do not cover:
    the matched forward here is this function's, and a graph over the $(B, T, M, L)$ attention it
    would build is both useless -- nothing backpropagates through an analysis -- and large enough
    that the arms below then fail to allocate a convolution workspace.

    Args:
        task: The Lightning task wrapping the loaded net.
        batch: One batch, already on the model's device.
        bands: The configured bands, ``{name: (lo, hi)}``.
        seed: This analysis's own seed, for the anchor draw and the latent noise.

    Returns:
        ``guids``, the $(B, H)$ reference score, one $(B, H)$ delta per band, the per-band live
        fraction, and the announcement invariance measured across the arms.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, batch)
    anchor_phase, anchor_stride = DENSE_ANCHOR_GEOMETRY
    outputs = model(
        y_st, y_ph, u_stream, anchor_phase=anchor_phase, anchor_stride=anchor_stride
    )

    device = y_st.device
    anchor_generator = torch.Generator(device=device)
    anchor_generator.manual_seed(int(seed) + _SEED_OFFSET_ANCHOR)
    columns = choose_anchors(outputs["anchor_valid"], anchor_generator)
    index = columns[:, None]
    anchors = outputs["anchor_index"].gather(1, index)
    anchor_valid = outputs["anchor_valid"].gather(1, index)
    # The forward's own persistence, narrowed to the scored anchor so it matches the one-column
    # anchor set every arm decodes at. Narrowed by the **column** rather than by the step index:
    # that tensor is one row per decoded anchor while ``anchors`` holds sequence positions, and the
    # two spaces differ by the anchor floor -- gathering with one on the other reads out of bounds
    # on the device rather than returning a wrong row.
    scored = dict(outputs)
    persistence = outputs.get("persistence")
    if persistence is not None:
        scored["persistence"] = persistence.gather(
            1, index[:, :, None].expand(-1, -1, persistence.shape[-1])
        )

    gated = u_stream if model.source_gate is None else model.source_gate(u_stream)
    announcement = _announcement(model)
    before = None if announcement is None else announcement.detach().clone()

    def _arm(occlusion: Optional[torch.Tensor]) -> torch.Tensor:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(seed) + _SEED_OFFSET_NOISE)
        arm = occluded_forward_outputs(
            model,
            scored,
            gated,
            occlusion=occlusion,
            anchors=anchors,
            generator=noise_generator,
        )
        return _horizon_scores(
            model,
            arm,
            target_features=target_features,
            weight=weight,
            anchors=anchors,
            anchor_valid=anchor_valid,
            likelihood=likelihood,
        )

    reference = _arm(None)
    deltas: Dict[str, torch.Tensor] = {}
    live: Dict[str, float] = {}
    for name, band in bands.items():
        mask = band_mask(anchors[:, 0], band, int(gated.shape[1]))
        # How much of the band actually held a value. Measured on the gated stream the intervention
        # edits, so a band inside the warm-up -- where the availability mechanism has already
        # zeroed every channel -- reports a live fraction near zero and its delta is read as the
        # absence of a source rather than as the source not mattering.
        occupied = (gated != 0).to(torch.float64) * mask[:, :, None].to(torch.float64)
        denominator = float(mask.sum().item()) * float(gated.shape[2])
        live[name] = (
            float("nan") if denominator == 0.0 else float(occupied.sum().item()) / denominator
        )
        deltas[name] = _arm(mask) - reference

    after = None if announcement is None else _announcement(model)
    invariance = (
        float("nan")
        if before is None or after is None
        else float((after - before).abs().max().item())
    )
    return {
        "guids": batch_guids(batch, batch_size_of(batch)),
        # The segment's own start, which is what makes a row here joinable. ``guid`` alone does
        # NOT identify a segment -- a recording contributes many, and the collection pass keys its
        # per-anchor table on ``(guid, epoch, anchor)`` for exactly that reason -- so without this
        # column the deltas below can be reduced per recording and placed on no clock at all.
        "epochs": _epochs(batch),
        "reference": reference.cpu().numpy(),
        "deltas": {name: value.cpu().numpy() for name, value in deltas.items()},
        "live_fraction": live,
        "announcement_max_abs_change": invariance,
        "anchors": anchors[:, 0].detach().cpu().numpy(),
    }


def _band_column(name: str) -> str:
    """The per-recording column one band's total delta is carried under."""
    return f"occlusion_delta_{name}_nats"


def build_frames(
    records: Sequence[Dict[str, Any]], bands: Dict[str, Tuple[int, int]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Reduce the batch records into the per-sample table and the per-horizon curve table.

    The per-sample table carries one row per scored segment with each band's delta **summed over
    the horizon** -- a total in nats per anchor, on the same scale as ``pred_gap`` -- so the
    per-recording fan-out and the cohort comparison have a scalar to work with. The per-horizon
    table keeps the axis the analysis exists for, one row per ``(band, horizon step)``.

    Args:
        records: What :func:`collect_batch` returned, in loader order.
        bands: The configured bands, for the column order.

    Returns:
        ``(per_sample, per_horizon)``.
    """
    if not records:
        return pd.DataFrame(), pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for record in records:
        for position, guid in enumerate(record["guids"]):
            row: Dict[str, Any] = {
                "guid": str(guid),
                # Beside the guid rather than instead of it: the pair is what identifies a
                # segment, and it is the key this frame is joined onto the collected table by.
                "epoch": float(record["epochs"][position]),
                "anchor_step": int(record["anchors"][position]),
                "reference_block_nats": float(record["reference"][position].sum()),
            }
            for name in bands:
                row[_band_column(name)] = float(record["deltas"][name][position].sum())
            rows.append(row)
    per_sample = pd.DataFrame(rows)

    horizon_rows: List[Dict[str, Any]] = []
    n_steps = int(records[0]["reference"].shape[1])
    for name, band in bands.items():
        stacked = np.concatenate([record["deltas"][name] for record in records], axis=0)
        # Weighted by batch size: an unweighted mean of per-batch fractions lets the short
        # last batch count as much as a full one.
        sizes = np.asarray([len(record["guids"]) for record in records], dtype=np.float64)
        fractions = np.asarray([record["live_fraction"][name] for record in records], dtype=np.float64)
        usable = np.isfinite(fractions) & (sizes > 0)
        live = (
            float(np.sum(fractions[usable] * sizes[usable]) / np.sum(sizes[usable]))
            if usable.any() else float("nan")
        )
        for step in range(n_steps):
            column = stacked[:, step]
            finite = column[np.isfinite(column)]
            horizon_rows.append(
                {
                    "band": name,
                    "lag_lo": int(band[0]),
                    "lag_hi": int(band[1]),
                    "horizon_step": step,
                    "n_anchors": int(finite.size),
                    "delta_nats": float(finite.mean()) if finite.size else float("nan"),
                    "delta_nats_sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                    "live_fraction": live,
                }
            )
    return per_sample, pd.DataFrame(horizon_rows)


def _precision(
    per_recording: Optional[pd.DataFrame],
    band: str,
    *,
    n_segments: Optional[int],
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    r"""One band's spread over recordings, and the two counts a cap is chosen from.

    The standard error is the plain $s/\sqrt{n}$ of the per-recording values, beside a
    percentile bootstrap interval of the same mean. Both, and not one: the SE is what the
    $n\prime = n\,(s/s\prime)^2$ cap arithmetic needs, while the interval is what a reader
    should quote, because these per-recording distributions are small-$n$ and skewed and a
    symmetric interval would report a symmetry the data does not have.

    Args:
        per_recording: The per-recording frame, or ``None`` when none was built.
        band: The band whose column to read.
        n_segments: Segments scored, carried through unchanged.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        The five precision fields. All ``NaN``/``None`` when the frame is absent or carries
        no column for this band -- absent rather than defaulted, so a run that measured no
        spread does not report one.
    """
    column = _band_column(band)
    absent = {
        "n_segments": None if n_segments is None else int(n_segments),
        "n_recordings": 0,
        "delta_total_recording_mean_nats": float("nan"),
        "delta_total_se": float("nan"),
        "delta_total_ci_lo": float("nan"),
        "delta_total_ci_hi": float("nan"),
    }
    if per_recording is None or column not in getattr(per_recording, "columns", []):
        return absent

    values = np.asarray(per_recording[column], dtype=np.float64)
    finite = values[np.isfinite(values)]
    interval = shared_stats.bootstrap_ci(finite, resamples=int(resamples), seed=int(seed))
    return {
        "n_segments": None if n_segments is None else int(n_segments),
        "n_recordings": int(finite.size),
        # The point estimate the SE and the interval below describe: the mean over RECORDINGS
        # of the horizon-summed delta. ``delta_total_nats`` beside it is the segment-pooled sum
        # of per-step means, a different reduction of the same deltas, so the interval is not
        # guaranteed to contain it -- read the interval against this column.
        "delta_total_recording_mean_nats": float(finite.mean()) if finite.size else float("nan"),
        # ``ddof=1``: the spread being estimated is the population's rather than this
        # sample's own, and at the handful of recordings this analysis is capped to the
        # difference between the two is not cosmetic.
        "delta_total_se": (
            float(finite.std(ddof=1) / np.sqrt(finite.size))
            if finite.size > 1
            else float("nan")
        ),
        "delta_total_ci_lo": float(interval["lo"]),
        "delta_total_ci_hi": float(interval["hi"]),
    }


def build_summary(
    per_horizon: pd.DataFrame,
    bands: Dict[str, Tuple[int, int]],
    *,
    per_recording: Optional[pd.DataFrame] = None,
    n_segments: Optional[int] = None,
    resamples: int = shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    r"""One row per band: its total delta, where over the horizon it peaks, and how precise
    it is.

    **The delta and its uncertainty are the same measurement and are emitted together.** A
    band delta quoted alone cannot be compared against another band: the identifiability
    record's four bands differ by $14.9$ nats against per-band standard deviations of
    $1.9$--$2.3$ at eight segments, and whether that separation survives is a question only a
    spread can answer. It is also the number that decides ``eval_config.caps.occlusion`` --
    the standard error falls as $1/\sqrt{n}$, so an operator wanting half the current one
    needs four times the cap.

    **The interval is over recordings, and the cap is in segments.** Both counts are
    therefore emitted. Resampling *segments* would report an interval narrower than the data
    supports, for the reason ``eval_config.bootstrap_resamples`` states for every other
    interval in this pipeline: segments of one recording are not independent draws. The cap
    is nevertheless a segment bound, because segments are what the loop consumes, so a reader
    converting one into the other needs the ratio -- and that ratio is a property of the
    split rather than of this analysis.

    The per-recording value is the mean over that recording's scored segments of the
    horizon-summed delta, which is the same quantity ``delta_total_nats`` is a pooled version
    of -- the two coincide exactly when every recording contributed the same number of
    segments and differ by the recording weighting otherwise.

    Args:
        per_horizon: The per-``(band, horizon step)`` table.
        bands: The configured bands.
        per_recording: One row per recording carrying each band's horizon-summed delta, as
            :func:`~teb_vae.lag_attn_cfs.eval.frames.per_recording_means` builds it. ``None``
            leaves every precision column ``NaN`` rather than defaulting them: a spread
            nobody measured must not be reported as one that was measured and found small.
        n_segments: Segments scored, for the cap arithmetic. ``None`` for the same reason.
        resamples: Bootstrap resamples behind the interval.
        seed: Seed, so the interval is reproducible from the summary alone.

    Returns:
        The summary table, in the configured band order.
    """
    rows: List[Dict[str, Any]] = []
    for name, band in bands.items():
        subset = per_horizon[per_horizon["band"] == name] if not per_horizon.empty else per_horizon
        values = (
            np.asarray(subset["delta_nats"], dtype=np.float64)
            if not subset.empty
            else np.asarray([], dtype=np.float64)
        )
        finite = values[np.isfinite(values)]
        # The horizon step the peak sits at, read off the row rather than off the position in
        # the finite-filtered vector, which is not a step once any per-step mean is NaN.
        peak = (
            int(np.asarray(subset["horizon_step"])[int(np.nanargmax(values))])
            if finite.size else None
        )
        rows.append(
            {
                "band": name,
                "lag_lo": int(band[0]),
                "lag_hi": int(band[1]),
                # The total is summed over the horizon, so it is in nats per anchor like
                # pred_gap; only the mean and the max beside it are per horizon step.
                "unit": "delta_total_nats: nats per anchor; delta_mean/max_nats: "
                        + NATS_PER_ANCHOR_STEP,
                # Summed over the horizon rather than averaged, so it is on the block score's own
                # scale and comparable with ``pred_gap`` directly.
                "delta_total_nats": float(finite.sum()) if finite.size else float("nan"),
                "delta_mean_nats": float(finite.mean()) if finite.size else float("nan"),
                "delta_max_nats": float(finite.max()) if finite.size else float("nan"),
                "peak_horizon_step": peak,
                "live_fraction": (
                    float(subset["live_fraction"].iloc[0]) if not subset.empty else float("nan")
                ),
                "n_horizon_steps": int(finite.size),
                **_precision(
                    per_recording, name, n_segments=n_segments, resamples=resamples, seed=seed
                ),
            }
        )
    return pd.DataFrame(rows)


def headline_record(summary: pd.DataFrame) -> Dict[str, Any]:
    """Flatten the most informative band into the block the headline registry resolves.

    A **scalar and deliberately not a verdict**: how much a forecast loses without a band of source
    is a measurement whose healthy range nothing has established, and a threshold guessed before
    the first production runs would decide a pass or a fail on exactly the run that was going to
    measure it. It reaches every arm table as a number instead, which is what lets a threshold be
    set from data later.

    Args:
        summary: The per-band summary.

    Returns:
        The winning band's name, its total delta, where over the horizon it peaks and how much of
        it was live -- or the same keys unmeasured when no band scored anything.
    """
    blank = {
        "band": None,
        "delta_total_nats": float("nan"),
        "peak_horizon_step": None,
        "live_fraction": float("nan"),
        "n_bands": 0 if summary.empty else int(len(summary)),
    }
    if summary.empty:
        return blank
    totals = np.asarray(summary["delta_total_nats"], dtype=np.float64)
    if not np.isfinite(totals).any():
        return blank
    winner = summary.iloc[int(np.nanargmax(totals))]
    return {
        "band": str(winner["band"]),
        "delta_total_nats": float(winner["delta_total_nats"]),
        "peak_horizon_step": (
            None if winner["peak_horizon_step"] is None else int(winner["peak_horizon_step"])
        ),
        "live_fraction": float(winner["live_fraction"]),
        "n_bands": int(len(summary)),
    }


def build_horizon_figure(per_horizon: pd.DataFrame, bands: Dict[str, Tuple[int, int]]) -> Any:
    """Draw one delta curve per band against the horizon step, with zero marked.

    Zero is drawn because the sign is the finding: a curve above it is a band the forecast needed,
    and one sitting on it is a band it did not -- or one that was already empty, which the legend's
    live fraction says.

    Args:
        per_horizon: The per-``(band, horizon step)`` table.
        bands: The configured bands, for the curve order.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(1)
    axis = axes[0, 0]
    if per_horizon.empty:
        figures.multi_line_panel(
            axis, np.asarray([]), np.asarray([]), [],
            title="source occlusion: no band scored",
            xlabel="horizon step", ylabel=NATS_PER_ANCHOR_STEP,
        )
        return figure

    steps = np.sort(per_horizon["horizon_step"].unique())
    curves, labels = [], []
    for name, band in bands.items():
        subset = per_horizon[per_horizon["band"] == name].sort_values("horizon_step")
        if subset.empty:
            continue
        curves.append(np.asarray(subset["delta_nats"], dtype=np.float64))
        labels.append(
            f"{name} [{int(band[0])}, {int(band[1])}] "
            f"(live {float(subset['live_fraction'].iloc[0]):.2f})"
        )
    figures.multi_line_panel(
        axis,
        steps,
        np.asarray(curves, dtype=np.float64) if curves else np.asarray([]),
        labels,
        title="forecast cost of removing a band of source, by horizon step",
        xlabel="horizon step",
        ylabel=NATS_PER_ANCHOR_STEP,
    )
    axis.axhline(0.0, linestyle="--", linewidth=1.0, color="0.4")
    return figure


def _skip(reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The protocol's keys with nothing measured, and why.

    Args:
        reason: What the run did not carry.
        extra: Anything else worth recording beside it.

    Returns:
        A result the runner records as a completed step that measured nothing.
    """
    return {
        "n_samples": 0,
        "composition": {"n_recordings": 0},
        "plan": {"capped": False, "skipped": True, "reason": reason},
        "skipped": True,
        "reason": reason,
        "headline": headline_record(pd.DataFrame()),
        "caveat": OCCLUSION_CAVEAT,
        "files": [],
        **(extra or {}),
    }


def cost_record(
    *,
    elapsed_s: float,
    n_batches: int,
    n_samples: int,
    n_arms: int,
    device: Any,
) -> Dict[str, Any]:
    r"""What this analysis cost, and the two rates a cap is chosen from.

    The key vocabulary is :meth:`~teb_vae.lag_attn_cfs.eval.collect.Collector._cost_record`'s,
    deliberately and to the letter, so that a reader holding both records compares two numbers
    rather than two conventions -- this pass and the collection pass are the two expensive things
    a run does, and "which of them dominates" is a question a differently-named key set cannot
    answer.

    **Two keys are this analysis's own, and they are what make the cost attributable.** The work
    here is not uniform per sample the way the collection pass's is: it is one encode and one
    single-anchor decode per *arm*, where an arm is the reference plus one per configured band, so

    $$\texttt{seconds\_per\_arm\_per\_segment} = \frac{\texttt{elapsed\_s}}{N \cdot A},
    \qquad A = 1 + |\mathrm{bands}|.$$

    A run that adds a fifth band pays $6/5$ of this pass, and only a per-arm rate says so. Quoting
    ``hours_per_1000_samples`` alone would make a band count look like a property of the dataset.

    Recorded rather than checked against a threshold, for
    :meth:`~teb_vae.lag_attn_cfs.eval.collect.Collector._cost_record`'s reason: a CI box's timing
    is not a production box's, so a bound here would either be met by accident or fail for the
    machine. What an operator needs before raising ``eval_config.caps.occlusion`` is the rate this
    pass ran at, taken on the same code path.

    Args:
        elapsed_s: Wall-clock seconds the scoring loop took, excluding the frames written after it.
        n_batches: Batches actually drawn from the loader before the cap stopped it.
        n_samples: Segments scored, which is one anchor each.
        n_arms: The reference arm plus one per configured band.
        device: The device the pass ran on, or ``None``.

    Returns:
        The measured rates, the settings they were measured at, and the extrapolation rule.
        ``peak_allocated_bytes`` is a **CUDA** figure and is ``None`` elsewhere rather than $0$:
        the allocator that reports it does not exist on CPU, and a zero there would read as
        "measured, and the pass used nothing".
    """
    elapsed = max(float(elapsed_s), 1e-9)
    samples = int(n_samples)
    batches = int(n_batches)
    arms = int(n_arms)
    resolved = None if device is None else torch.device(device)
    peak_bytes: Optional[int] = None
    if resolved is not None and resolved.type == "cuda":
        peak_bytes = int(torch.cuda.max_memory_allocated(resolved))
    rate = samples / elapsed
    return {
        "device": None if resolved is None else str(resolved),
        "n_batches": batches,
        "n_samples": samples,
        "n_arms": arms,
        # Measured rather than read off the loader: the cap stops the loop on a batch boundary, so
        # the last batch counted is a whole one but the loop's own mean need not be the loader's.
        "mean_batch_size": (samples / batches) if batches else None,
        "elapsed_s": float(elapsed),
        "seconds_per_batch": (float(elapsed / batches)) if batches else None,
        "samples_per_second": float(rate),
        "hours_per_1000_samples": (1000.0 / rate / 3600.0) if rate > 0.0 else None,
        # The band count divided out, which is what makes a cap chosen here survive a change to
        # the band partition.
        "seconds_per_arm_per_segment": (
            float(elapsed / (samples * arms)) if samples and arms else None
        ),
        "peak_allocated_bytes": peak_bytes,
        "note": (
            "measured on this pass at the batch size and band count recorded beside it. A longer "
            "pass extrapolates as hours = (n_samples / 1000) * hours_per_1000_samples at THIS "
            "band count; a pass with a different band count extrapolates as "
            "seconds = n_samples * n_arms * seconds_per_arm_per_segment. peak_allocated_bytes is "
            "the CUDA allocator's process peak up to the end of this pass and is null on any "
            "other device. See EVAL.md's occlusion section for how to choose caps.occlusion "
            "from this block and the per-band standard errors beside it."
        ),
    }


def clock_frames(
    per_sample: pd.DataFrame, collected: Any, bands: Dict[str, Tuple[int, int]]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Place each band's per-segment delta on both clinical clocks.

    **A join, not three more reads off the batch.** The clock coordinate needs ``epoch``, the
    second clock needs ``second_stage_onset`` and the cohort split needs ``clinical_class`` and
    ``subgroup`` -- and every one of them is already on the collected per-sample table, resolved
    once, by the pass every other analysis reads. Joining on ``(guid, epoch)`` picks them all up
    and additionally guarantees this analysis cannot disagree with any other about which class a
    segment belongs to, which reading them again here could not.

    **What does not join is counted rather than dropped.** A segment scored here that the collected
    table does not carry means the two passes disagree about the population, and a silently shorter
    table is exactly how that would go unnoticed.

    Args:
        per_sample: This analysis's per-segment frame, carrying :data:`JOIN_KEYS`.
        collected: The run's :class:`~teb_vae.lag_attn_cfs.eval.collect.Collection`.
        bands: The configured bands, for the delta columns.

    Returns:
        ``(frame, record)`` -- long-form rows per ``(clock, cohort axis, cohort, band, window)``,
        and the join's own census.
    """
    columns = [_band_column(name) for name in bands]
    keys = ["clock", "group_column", "group", "band", "time_bin", "bin_center_h",
            "n_recordings", "mean", "q25", "median", "q75"]
    collected_frame = getattr(collected, "per_sample", None)
    if per_sample.empty or collected_frame is None or collected_frame.empty:
        return pd.DataFrame(columns=keys), {
            "joined": False,
            "reason": "one of the two tables was empty, so there was nothing to place on a clock",
        }
    available = [name for name in JOIN_KEYS if name in collected_frame.columns]
    if list(available) != list(JOIN_KEYS):
        return pd.DataFrame(columns=keys), {
            "joined": False,
            "reason": (
                f"the collected table carries {available} of the join key {list(JOIN_KEYS)}, so a "
                f"segment here cannot be matched to its clinical coordinates"
            ),
        }

    carried = [
        name
        for name in (*labels.GROUP_COLUMNS, cohort.SECOND_STAGE_COLUMN)
        if name in collected_frame.columns
    ]
    merged = per_sample.merge(
        collected_frame[[*JOIN_KEYS, *carried]].drop_duplicates(subset=list(JOIN_KEYS)),
        on=list(JOIN_KEYS),
        how="left",
        validate="many_to_one",
    )
    # The tripwire. Zero on a healthy run; anything else means the two passes drew different
    # segments, and the number says how many rather than the table quietly being shorter.
    unjoined = int(merged[carried[0]].isna().sum()) if carried else int(len(merged))
    census: Dict[str, Any] = {
        "joined": True,
        "n_scored": int(len(per_sample)),
        "n_unjoined": unjoined,
        "carried_columns": carried,
        "note": CLOCK_NOTE,
    }

    rows: List[pd.DataFrame] = []
    for clock_name, binner, bin_column, center_column in (
        ("time_to_delivery", cohort.add_time_bins, cohort.BIN_COLUMN, cohort.BIN_CENTER_COLUMN),
        (
            "second_stage",
            cohort.add_second_stage_bins,
            cohort.SECOND_STAGE_BIN_COLUMN,
            cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
        ),
    ):
        binned = binner(merged)
        if binned.empty:
            continue
        for axis in labels.GROUP_COLUMNS:
            if axis not in binned.columns:
                continue
            wide = cohort.per_recording_in_bins(
                binned, columns, group_column=axis,
                bin_column=bin_column, center_column=center_column,
            )
            if wide.empty:
                continue
            for column in columns:
                if column not in wide.columns:
                    continue
                band = column[len("occlusion_delta_") : -len("_nats")]
                for row in cohort.trajectory_rows(
                    wide, column, metric=column,
                    bin_column=bin_column, center_column=center_column,
                ):
                    entry = {"clock": clock_name, "group_column": axis, "band": band, **row}
                    entry.pop("metric", None)
                    rows.append(pd.DataFrame([entry]))
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keys)
    census["n_rows"] = int(len(frame))
    return frame, census


def build_clock_figure(frame: pd.DataFrame, bands: Dict[str, Tuple[int, int]]) -> Any:
    """One panel per band: what removing it cost, window by window before delivery.

    The interventional counterpart of ``lag_kld_scaled``'s band trajectories, on the same
    partition and the same grid -- so the observational and the interventional answer to "did the
    informative past move" are two readings of one axis rather than two axes.

    Args:
        frame: The long-form clock table.
        bands: The partition, in panel order.

    Returns:
        The figure.
    """
    names = list(bands)
    figure, axes = figures.new_figure(max(len(names), 1), 1)
    subset = (
        frame[
            (frame["clock"] == "time_to_delivery")
            & (frame["group_column"] == labels.CLASS_COLUMN)
        ]
        if len(frame)
        else frame
    )
    groups = sorted({str(value) for value in subset["group"]}) if len(subset) else []
    colors = figures.group_colors(groups) if groups else {}
    for index, name in enumerate(names):
        panel = axes[index][0]
        band_rows = subset[subset["band"] == name] if len(subset) else subset
        if not len(band_rows):
            panel.text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=panel.transAxes,
                ha="center", va="center", color=figures.COLOR_GRAY, fontsize=figures.FONT_NOTE,
            )
        panel.axhline(0.0, linewidth=figures.LINE_HAIRLINE, color=figures.COLOR_GRAY)
        for group in groups:
            cell = band_rows[band_rows["group"] == group].sort_values("bin_center_h")
            if not len(cell):
                continue
            panel.plot(
                cell["bin_center_h"], cell["mean"],
                color=colors.get(group, figures.COLOR_GRAY),
                linewidth=figures.LINE_REGULAR, label=group,
            )
        span = bands[name]
        panel.set_title(
            f"{name}: lags {span[0]}-{span[1]} steps "
            f"({span[0] * SECONDS_PER_STEP:g}-{(span[1] + 1) * SECONDS_PER_STEP:g} s back)",
            fontsize=figures.FONT_SMALL,
        )
        panel.set_ylabel("nats per anchor")
        panel.invert_xaxis()
        figures.style_axes(panel)
    axes[-1][0].set_xlabel("Time before delivery (hours)")
    if groups:
        axes[0][0].legend(loc="best", fontsize=figures.FONT_TINY)
    figures.caveat_note(figure)
    return figure


def run_occlusion_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score every configured band's forecast cost, resolved by horizon step.

    One of the few analyses that reads ``context.task`` and ``context.loader``, and the reason is
    structural rather than a convenience: an intervention on the model's *input* cannot be served
    by any table, because the tables record a forward the source was fully present in. A pass with
    no checkpoint records a skip rather than assuming a model, which is what keeps the offline
    re-run working for everything else.

    Args:
        context: The analysis context, read for the task and the loader.
        eval_config: The validated block, for the bands, the cap and the seed.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the per-band summary, the headline block, the announcement-invariance
        measurement and the paths written.
    """
    configured = eval_config.get("occlusion_bands") or {}
    bands = {str(name): (int(span[0]), int(span[1])) for name, span in configured.items()}
    if not bands:
        return _skip(
            "eval_config.occlusion_bands names no band, so there is no intervention to score. "
            "This is a configuration state rather than a failure: the analysis measures what a "
            "named band of source was worth, and it has not been told which bands to remove."
        )

    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return _skip(
            "the occlusion arm re-encodes an edited source stream, so it needs a model and a "
            "loader rather than a table; this pass built neither, which is what an offline re-run "
            "against a finished directory is"
        )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    cap = (eval_config.get("caps") or {}).get(CAP_NAME)
    seed = int(eval_config.get("seed", 0))

    records: List[Dict[str, Any]] = []
    invariances: List[float] = []
    n_scored = 0
    n_batches = 0
    # Around the scoring loop alone, not around the frames and the figure below it: what an
    # operator raising the cap buys more of is re-encodes, and the table assembly does not scale
    # with the cap in any way a rate could extrapolate.
    started = time.perf_counter()
    for batch in loader:
        if cap is not None and n_scored >= int(cap):
            break
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        record = collect_batch(task, moved, bands=bands, seed=seed)
        records.append(record)
        invariances.append(record["announcement_max_abs_change"])
        n_scored += len(record["guids"])
        n_batches += 1
    elapsed_s = time.perf_counter() - started

    per_sample, per_horizon = build_frames(records, bands)
    # Before the summary rather than after it: the per-band interval is taken over RECORDINGS, so
    # the summary needs this frame. The two were the other way round while the summary reported a
    # delta and no spread.
    band_columns = [_band_column(name) for name in bands]
    per_guid = (
        per_recording_means(per_sample, band_columns)
        if not per_sample.empty
        else pd.DataFrame()
    )
    summary = build_summary(
        per_horizon,
        bands,
        per_recording=per_guid,
        n_segments=len(per_sample),
        resamples=int(
            eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES)
        ),
        seed=seed,
    )
    per_horizon.to_csv(directory / PER_HORIZON_FILENAME, index=False)
    summary.to_csv(directory / SUMMARY_FILENAME, index=False)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    figure_name = str(
        figures.render_figure(
            build_horizon_figure(per_horizon, bands), directory / HORIZON_FIGURE
        ).name
    )

    # The deltas placed on the two clinical clocks, by joining this frame's ``(guid, epoch)`` onto
    # the collected per-sample table -- which is where the class, the subgroup and the second-stage
    # offset already live, resolved once by the pass every other analysis reads.
    clocks, clock_census = clock_frames(per_sample, context.collection, bands)
    # Written with its header even when the join produced nothing: a table a reader can open and
    # find empty is a different statement from a table that was never written.
    clocks.to_csv(directory / CLOCK_FILENAME, index=False)
    clock_figure = str(
        figures.render_figure(
            build_clock_figure(clocks, bands), directory / CLOCK_FIGURE
        ).name
    )

    finite_invariance = [value for value in invariances if np.isfinite(value)]
    return {
        "n_samples": int(len(per_sample)),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {
            "capped": cap is not None and n_scored >= int(cap),
            "cap": None if cap is None else int(cap),
            "seed": seed,
            "bands": {name: [int(span[0]), int(span[1])] for name, span in bands.items()},
            "anchors_per_segment": 1,
        },
        # What the pass cost, so `eval_config.caps.occlusion` is set from a measurement rather
        # than guessed. Read it beside the per-band standard errors in ``bands``: time and
        # precision are the two things a cap trades between, and only one of them is a clock.
        "cost": cost_record(
            elapsed_s=elapsed_s,
            n_batches=n_batches,
            n_samples=n_scored,
            # The reference arm plus one per band -- the reference is scored through the same
            # function with no occlusion, so it costs an arm like any other.
            n_arms=1 + len(bands),
            device=getattr(task, "device", None),
        ),
        "unit": NATS_PER_ANCHOR_STEP,
        "bands": summary.to_dict(orient="records"),
        # The block the headline registry resolves; see ``headline_record`` for why it is a scalar
        # and not a verdict.
        "headline": headline_record(summary),
        # The confound this analysis exists to avoid, measured rather than asserted. Zero on every
        # batch is what "the intervention moved the values and not the clock" means; ``NaN`` is a
        # model whose adapter builds no announcement at all, which is the unaligned arm.
        "announcement_invariance": {
            "max_abs_change": (
                max(finite_invariance) if finite_invariance else float("nan")
            ),
            "n_batches_checked": len(finite_invariance),
            "meaning": (
                "the source adapter's availability announcement, compared before and after every "
                "occluded encode. It is a registered buffer over t and the resolved warm-up "
                "vector, so no value on the stream can reach it and this is exactly zero by "
                "construction; it is measured so a change to that construction fails here."
            ),
        },
        # Where the deltas fall on the two clinical clocks, and the census of the join that put
        # them there. Descriptive only -- see CLOCK_NOTE.
        "clocks": clock_census,
        "caveat": OCCLUSION_CAVEAT,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, tuple(band_columns))
        ],
        "files": [
            PER_RECORDING_FILENAME, PER_HORIZON_FILENAME, SUMMARY_FILENAME, CLOCK_FILENAME,
            figure_name, clock_figure,
        ],
    }
