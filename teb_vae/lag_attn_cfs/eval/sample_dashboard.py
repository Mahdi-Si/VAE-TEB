r"""The per-segment interactive dashboard: every input, output and readout of one forward, on one page.

The diagnostic page of :mod:`~teb_vae.lag_attn_cfs.eval.analyses.samples` is a fixed picture. This
module writes the same segment as a self-describing HTML page in which the reader chooses what to
compare. It draws from the **same forward dict** the printed page draws from, so the two cannot
describe different states of the model.

**The page has four linked figures.** The top two are exact copies of each other. Each one has a
trace list on the left, whose traces read against the left $y$ axis, and a trace list on the right,
whose traces read against the right $y$ axis, so that two signals of different scale share one
panel. The bottom two are exact copies of each other as well, and each one shows a heatmap with
optional traces over it. All four figures share one time axis at one pixel scale, so a column of
the page is one instant on every figure. Below them, the anchor inspector shows what the model
produced at one pinned anchor.

**Every quantity sits on one of three clocks**, and the page converts each clock to seconds from
the segment's first loaded step before it draws:

* the raw clock: sample $n$ sits at $n \Delta / R$ seconds, with $\Delta$ the stored step and $R$
  the raw samples per step;
* the step clock: step $t$ sits at $t \Delta$ seconds;
* the forecast clock: the prediction made at anchor $a$ for horizon step $\tau$ sits at
  $(a + 1 + \tau) \Delta$ seconds, which is the stored step it is scored against, so a forecast and
  its truth line up on the page.

**What is computed here and what is not.** The per-anchor scores are the objective's own elementwise
density, :func:`~teb_vae.lag_attn_rws.nets.losses.raw_sample_score`, under the model's own scored-cell
mask and AR(1) coefficient. The forward is stochastic, so these scores belong to this one forward
and differ from the Monte Carlo numbers in ``per_sample.csv`` by sampling noise. The page shows
both and names them differently.

**A lag axis here is stored-coefficient time**, as everywhere in this package. The stored timeline
is canonical, and the page applies no correction to it.

**Layout on disk.** A segment's page sits beside its two printed pages in the folder of every draw
that picked it, so a folder of pages is also a folder of dashboards. Each folder holds its own copy
of ``plotly.min.js``, ``dashboard_index.js`` and a gallery ``index.html``. A page therefore needs no
network, which is the normal case on the machines that hold these recordings, and a folder can be
moved as a whole. A second gallery in the samples directory lists every page of the run.
"""
from __future__ import annotations

import html
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import events, lag_axis, traces
from teb_vae.lag_attn_cfs.eval._reuse import labels
# The array encoding is the run dashboard's: both pages decode it with the same rule, so one
# function writes it. That module needs no ``torch``, which is why it owns the function.
from teb_vae.lag_attn_cfs.eval.run_dashboard_core import encode_array
from teb_vae.lag_attn_cfs.eval.metrics import (
    anchor_support,
    attention_entropy,
    batch_field,
    forecast_innovation_and_logvar,
    forecast_likelihood_terms,
)
from teb_vae.lag_attn_cfs.sample_page import _scored_clock_view
from teb_vae.lag_attn_cfs.ua_tolerance import tolerant_reward_validity
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS, raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors

#: What a dashboard file adds to the stem of the printed page of the same segment.
DASHBOARD_SUFFIX = "_dashboard.html"

#: The ``eval_config.caps`` name bounding how many segments get a dashboard. Absent means every
#: selected segment. The configuration schema refuses a cap of zero, as it does for every cap.
DASHBOARD_CAP = "dashboards"

#: The gallery page, the script that lists every page for the previous and next links, and the
#: shared copy of the plotting library.
GALLERY_FILENAME = "index.html"
INDEX_SCRIPT_FILENAME = "dashboard_index.js"
PLOTLY_FILENAME = "plotly.min.js"

_ASSET_DIR = Path(__file__).resolve().parent / "sample_dashboard_assets"

#: The readouts of the collection pass that the header and the gallery print.
GALLERY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("nll_base_block", "NLL base"),
    ("nll_full_block", "NLL full"),
    ("pred_gap", "Gap"),
    ("mc_pred_gap", "MC gap"),
    ("mean_pred_gap", "Mean gap"),
    ("source_conditioned_kl_raw", "KL"),
)


# =================================================================================================
# Encoding
# =================================================================================================
def _to_numpy(tensor: Any, sample: int = 0) -> np.ndarray:
    """One sample of a batched tensor as a host ``float64`` array."""
    value = tensor[sample]
    if hasattr(value, "detach"):
        value = value.detach().cpu().to(torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def _host_vector(values: Any) -> np.ndarray:
    """A tensor, list or tuple of numbers as a flat host ``float64`` array."""
    if hasattr(values, "detach"):
        return values.detach().cpu().to(torch.float64).numpy().reshape(-1)
    return np.asarray([float(value) for value in values], dtype=np.float64)


def _warmup_vector(values: Any, size: int) -> np.ndarray:
    r"""The per-channel warm-up $W'_c$ in steps, zeros for a model that builds no warm-up mask."""
    return np.zeros(size, dtype=np.float64) if values is None else _host_vector(values)


def _nan_reduce(reduction: Any, values: np.ndarray, axis: int) -> np.ndarray:
    """Apply a ``numpy`` nan-reduction, where a row with no value is expected and stays ``NaN``.

    Args:
        reduction: ``np.nanmean`` or ``np.nanmax``.
        values: The array. The steps outside the decoded anchors are all ``NaN`` by construction.
        axis: The axis reduced.

    Returns:
        The reduced array.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return reduction(values, axis=axis)


def _json_number(value: Any) -> Optional[float]:
    """A value as a JSON-safe float, or ``None`` when it is missing or not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


# =================================================================================================
# Channel names
# =================================================================================================
def _shard_channel_labels(path: Any, block: str) -> List[str]:
    """Name each stored channel of one block from the attributes of its HDF5 dataset.

    Args:
        path: The shard the segment came from.
        block: The dataset name, such as ``'fhr_ph'``.

    Returns:
        One label per stored channel. An unreadable shard gives an empty list, and the caller then
        falls back to bare channel numbers.
    """
    try:
        import h5py

        with h5py.File(str(path), "r") as handle:
            attrs = dict(handle[block].attrs)
            width = int(handle[block].shape[1])
    except Exception as error:  # noqa: BLE001 - a channel's name is not worth a page
        logger.debug(f"dashboard: no channel names for {block} in {path}: {error}")
        return []
    delays = np.asarray(attrs.get("causal_delay_s", np.full(width, np.nan)), dtype=float).ravel()
    band = np.asarray(attrs.get("sel_band_hz", np.full(width, np.nan)), dtype=float).ravel()
    harmonic = np.asarray(attrs.get("sel_harmonic", np.full(width, np.nan)), dtype=float).ravel()
    short = block.split("_")[1]
    names: List[str] = []
    for channel in range(width):
        parts = [f"{short} {channel}"]
        if channel < band.size and np.isfinite(band[channel]):
            parts.append(f"{band[channel]:.3g} Hz")
        if channel < harmonic.size and np.isfinite(harmonic[channel]):
            parts.append(f"h{int(harmonic[channel])}")
        if channel < delays.size and np.isfinite(delays[channel]):
            parts.append(f"{delays[channel]:.0f} s")
        names.append(", ".join(parts))
    return names


def _kept_channel_labels(path: Any, blocks: Sequence[str], keep: Sequence[int]) -> List[str]:
    """Name each kept channel of a concatenated stream, in the gate's order.

    Args:
        path: The shard the segment came from.
        blocks: The stored blocks in the stream's concatenation order.
        keep: The kept channels, positional into the concatenated stream.

    Returns:
        One label per kept channel.
    """
    declared: List[str] = []
    for block in blocks:
        declared.extend(_shard_channel_labels(path, block))
    return [
        declared[int(channel)] if int(channel) < len(declared) else f"channel {int(channel)}"
        for channel in keep
    ]


# =================================================================================================
# The catalogue a page offers
# =================================================================================================
def _trace(
    identifier: str,
    group: str,
    label: str,
    unit: str,
    clock: str,
    array: str,
    *,
    kind: str = "v",
    param: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """One entry of the page's trace list.

    Args:
        identifier: The stable name a preset and the saved view refer to.
        group: The heading the trace is listed under.
        label: What the list and the legend call it.
        unit: The unit printed on the axis.
        clock: ``'raw'``, ``'step'`` or ``'anchor'``.
        array: The key of the payload array holding the values.
        kind: How the page reads the array: ``'v'`` for a vector, ``'col'`` for one column of a
            matrix, ``'fc'`` for one forecast cell over the anchors.
        param: The parameter names the reader sets for this trace.
        extra: Further fields for the reader.

    Returns:
        The entry.
    """
    return {
        "id": identifier, "g": group, "l": label, "u": unit, "c": clock, "a": array, "k": kind,
        "p": list(param or []), **dict(extra or {}),
    }


def _heatmap(
    identifier: str,
    group: str,
    label: str,
    unit: str,
    array: str,
    *,
    kind: str = "m",
    y_title: str = "coordinate",
    y_values: Optional[str] = None,
    symmetric: bool = False,
    log: bool = False,
    param: Optional[Sequence[str]] = None,
    limit: Optional[str] = None,
    staircase: Optional[str] = None,
    ridge: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """One entry of the page's heatmap list.

    Args:
        identifier: The stable name a preset and the saved view refer to.
        group: The heading the map is listed under.
        label: What the list calls it.
        unit: The colour bar's unit.
        array: The key of the payload array holding the values.
        kind: ``'m'`` for a $(T, N)$ map, ``'head'`` for a $(T, M, L)$ map with a head selector,
            ``'fc_tau'`` for one horizon step over the channels, ``'fc_ch'`` for one channel over
            the horizon.
        y_title: The vertical axis title.
        y_values: The key of ``axes`` holding the vertical coordinate, or ``None`` for the index.
        symmetric: Whether the colour scale diverges about zero.
        log: Whether the logarithmic colour scale is the default.
        param: The parameter names the reader sets for this map.
        limit: The key of the run-wide colour limits for this map.
        staircase: The key of ``axes`` holding the warm-up of each row, drawn over the map.
        ridge: The key of a payload array holding a line drawn over the map.
        extra: Further fields for the reader.

    Returns:
        The entry.
    """
    return {
        "id": identifier, "g": group, "l": label, "u": unit, "a": array, "k": kind, "yt": y_title,
        "yv": y_values, "sym": bool(symmetric), "log": bool(log), "p": list(param or []),
        "lim": limit, "stair": staircase, "ridge": ridge, **dict(extra or {}),
    }


# =================================================================================================
# The payload of one page
# =================================================================================================
@dataclass
class SegmentInputs:
    """What the samples analysis already holds for one drawn segment.

    Attributes:
        batch: The batch on the model's device, of one sample.
        outs: The forward's dict.
        target_features: The feature target $(1, T, c_y)$.
        weight: The objective's validity $(1, T)$.
        u_stream: The source stream $(1, T, c_u)$.
        source_valid: The source-validity mask $(1, T, c_u)$, or ``None``.
        kld_per_dim: The per-coordinate divergence $(1, T, d_z)$.
    """

    batch: Any
    outs: Mapping[str, torch.Tensor]
    target_features: torch.Tensor
    weight: torch.Tensor
    u_stream: torch.Tensor
    source_valid: Optional[torch.Tensor]
    kld_per_dim: torch.Tensor


def _lag_centroid(profile: np.ndarray, lag_seconds: np.ndarray) -> np.ndarray:
    """The profile-weighted mean lag per step, ``NaN`` where the profile holds no mass."""
    total = np.nansum(profile, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.nansum(profile * lag_seconds, axis=1) / total
    return np.where(total > 0.0, centroid, np.nan)


def _profile_argmax(profile: np.ndarray, lag_seconds: np.ndarray) -> np.ndarray:
    """The lag of each step's largest bin, ``NaN`` where the profile holds no mass."""
    filled = np.nan_to_num(profile, nan=-np.inf)
    peaks = lag_seconds[np.argmax(filled, axis=1)]
    return np.where(np.nansum(profile, axis=1) > 0.0, peaks, np.nan)


@torch.no_grad()
def _forecast_readouts(
    model: Any,
    inputs: SegmentInputs,
    truth: np.ndarray,
    stored: np.ndarray,
    likelihood: str,
) -> Dict[str, np.ndarray]:
    r"""Score every decoded anchor of this one forward, under the objective's own density.

    Args:
        model: The rebuilt net.
        inputs: The segment.
        truth: The scored-clock truth $(T, C_{\mathrm{keep}})$.
        stored: The kept target stream as stored, $(T, C_{\mathrm{keep}})$. The persistence forecast
            reads it at step $t_a + \min(s_c, 0)$, the model's own persistence input.
        likelihood: The objective's likelihood name.

    Returns:
        ``(A,)`` arrays: the coverage, the contributing flag, the base and full block scores, their
        difference, and the mean squared error of the base branch, the full branch and the
        persistence forecast over the scored cells. Also ``z_full``, the full branch's standardised
        innovation $(A, H, C_{\mathrm{keep}})$, which is what the density scores when the residual
        is AR(1), and ``forecast_mask``, the objective's $(A, H)$ step mask.
    """
    outs = inputs.outs
    anchors = outs["anchor_index"][0].long()
    valid = outs["anchor_valid"][0].bool()
    mask, coverage, _support = anchor_support(model, inputs.weight, outs)
    contributing = contributing_anchors(mask)[0] > 0.0
    terms = forecast_likelihood_terms(model)
    cell_mask, ar_coef = terms.get("cell_mask"), terms.get("ar_coef")
    horizon = int(model.geometry.horizon)
    stream = torch.as_tensor(truth, dtype=outs["mu_full"].dtype, device=anchors.device)
    n_anchors = anchors.shape[0]
    columns: Dict[str, np.ndarray] = {
        name: np.full(n_anchors, np.nan)
        for name in ("nll_base", "nll_full", "mse_base", "mse_full", "mse_persist")
    }
    z_full = np.full((n_anchors, horizon, stored.shape[1]), np.nan)
    channel_index = np.arange(stored.shape[1])
    # The model's persistence clock: min(s_c, 0), so an advanced channel never reads future data.
    persistence_shift = (
        np.zeros(stored.shape[1], dtype=int) if model.target_forecast_shift is None
        else np.minimum(_host_vector(model.target_forecast_shift).astype(int), 0)
    )
    for position in range(n_anchors):
        if not bool(valid[position]):
            continue
        anchor = int(anchors[position])
        target = stream[anchor + 1: anchor + 1 + horizon]
        step_mask = mask[0, position]
        if target.shape[0] != horizon or not bool(step_mask.any()):
            continue
        innovation, logvar_effective = forecast_innovation_and_logvar(
            outs["mu_full"][0, position], target, outs["logvar_full"][0, position],
            ar_coef, step_mask,
        )
        z_full[position] = (
            (innovation * torch.exp(-0.5 * logvar_effective)).detach().cpu().to(torch.float64).numpy()
        )
        if not bool(contributing[position]):
            continue
        cells = step_mask[:, None].float()
        if cell_mask is not None:
            cells = cells * cell_mask.float()
        scored = torch.isfinite(target).float() * cells
        count = scored.sum().clamp_min(1.0)
        for branch in ("base", "full"):
            mean = outs[f"mu_{branch}"][0, position]
            score = raw_sample_score(
                mean, target, likelihood=likelihood,
                logvar=outs[f"logvar_{branch}"][0, position],
                cell_mask=cell_mask, ar_coef=ar_coef, step_mask=step_mask,
                # The model's own innovation density; absent (Gaussian) on most models.
                student_nu=terms.get("student_nu"),
            )
            columns[f"nll_{branch}"][position] = float((score * step_mask[:, None]).sum())
            columns[f"mse_{branch}"][position] = float(
                (torch.nan_to_num(target - mean) ** 2 * scored).sum() / count
            )
        persistence = torch.as_tensor(
            stored[anchor + persistence_shift, channel_index], dtype=target.dtype, device=target.device
        )
        columns["mse_persist"][position] = float(
            (torch.nan_to_num(target - persistence) ** 2 * scored).sum() / count
        )
    columns["coverage"] = coverage[0].detach().cpu().to(torch.float64).numpy()
    columns["contributing"] = contributing.detach().cpu().to(torch.float64).numpy()
    columns["pred_gap"] = columns["nll_base"] - columns["nll_full"]
    columns["z_full"] = z_full
    columns["forecast_mask"] = mask[0].detach().cpu().to(torch.float64).numpy()
    return columns


def build_payload(
    task: Any,
    inputs: SegmentInputs,
    row: pd.Series,
    *,
    index: int,
    delay_steps: int,
    scales: Mapping[str, Tuple[float, float]],
    row_limits: Optional[Mapping[str, float]] = None,
    run_label: str = "",
) -> Dict[str, Any]:
    r"""Reduce one forward to the arrays and the catalogue of one dashboard.

    Args:
        task: The loaded task.
        inputs: The segment's tensors, off the forward the printed pages draw.
        row: The resolved table row of the segment.
        index: The dataset index.
        delay_steps: The model's causal input delay, for the lag axis.
        scales: The map from loader units to physical units of the raw signals, from
            :func:`~teb_vae.lag_attn_cfs.eval.traces.raw_signal_scales`.
        row_limits: The run's shared colour limits, or ``None``.
        run_label: What the header calls this run.

    Returns:
        The JSON-ready payload.
    """
    model = task.orig_model
    geometry = model.geometry
    outs, batch = inputs.outs, inputs.batch
    n_steps, horizon = int(geometry.t), int(geometry.horizon)
    raw_per_step = int(geometry.decimation)
    step_seconds = float(SECONDS_PER_STEP)
    anchors = outs["anchor_index"][0].long().detach().cpu().numpy()
    anchor_valid = outs["anchor_valid"][0].bool().detach().cpu().numpy()
    decoded = anchors[anchor_valid]
    n_lags = int(outs["attn_weights"].shape[-1])
    lag_seconds = np.asarray(
        lag_axis.compensated_seconds_axis(n_lags, int(delay_steps)), dtype=np.float64
    )

    arrays: Dict[str, np.ndarray] = {}
    items: List[Dict[str, Any]] = []
    maps: List[Dict[str, Any]] = []

    def on_anchors(values: np.ndarray) -> np.ndarray:
        """Blank a step-clock model quantity outside the decoded anchors."""
        masked = np.full_like(values, np.nan, dtype=np.float64)
        masked[decoded] = values[decoded]
        return masked

    # ---- Raw signals, on the raw clock ----------------------------------------------------------
    raw_valid: Dict[str, Optional[np.ndarray]] = {}
    for name, word in (("fhr", "FHR"), ("up", "UA")):
        stored = batch_field(batch, name)
        if stored is None:
            continue
        values = np.asarray(_to_numpy(stored)).reshape(-1)
        valid_tensor = batch_field(batch, f"{name}_valid")
        valid = None if valid_tensor is None else np.asarray(_to_numpy(valid_tensor)).reshape(-1) > 0.5
        raw_valid[name] = valid
        drawn, unit = traces.physical_raw(values, name, scales, valid)
        whole, _unit = traces.physical_raw(values, name, scales, None)
        arrays[name] = drawn
        arrays[f"{name}_stored"] = whole
        limits = list(traces.RAW_SIGNAL_LIMITS[name]) if unit == traces.RAW_SIGNAL_UNITS[name] else None
        group = "Raw signals"
        items.append(_trace(name, group, f"{word}, valid samples", unit, "raw", name, extra={"ctg": limits}))
        items.append(_trace(f"{name}_stored", group, f"{word}, stored with repairs", unit, "raw", f"{name}_stored", extra={"ctg": limits}))
        if valid is not None:
            arrays[f"{name}_valid"] = valid.astype(np.float64)
            items.append(_trace(f"{name}_valid", "Data quality", f"{word} sample valid", "0 or 1", "raw", f"{name}_valid"))

    # ---- Contractions and decelerations, on the raw clock -----------------------------------------
    event_spans: Dict[str, List[List[float]]] = {"contractions": [], "decelerations": []}
    weight_batch = batch_field(batch, "weight")
    if "up" in arrays:
        raw_len = arrays["up"].size
        valid = None if weight_batch is None else traces.raw_validity_of(_to_numpy(weight_batch), raw_len)
        # The detector's validity as the collection pass builds it, so a contraction on the page is
        # the contraction the tables and the training weight name.
        if raw_valid.get("up") is not None and bool(task.hparams.get("event_weight_ua_valid", False)):
            # With the training detector's tolerance for short UA runs, when the run set one.
            up_ok = tolerant_reward_validity(
                torch.as_tensor(np.asarray(raw_valid["up"], dtype=bool))[None],
                task.hparams.get("event_weight_ua_tolerance_s"),
            )[0].numpy()
            valid = up_ok if valid is None else (np.asarray(valid, dtype=bool) & up_ok)
        found = events.detect_contractions(arrays["up_stored"], valid=valid)
        event_spans["contractions"] = [
            [float(a), float(b), float(c)]
            for a, b, c in zip(found["onset_raw"], found["peak_raw"], found["end_raw"])
        ]
    if "fhr" in arrays:
        raw_len = arrays["fhr"].size
        valid = None if weight_batch is None else traces.raw_validity_of(_to_numpy(weight_batch), raw_len)
        if raw_valid.get("fhr") is not None:
            valid = raw_valid["fhr"] if valid is None else (np.asarray(valid, dtype=bool) & raw_valid["fhr"])
        valid = np.isfinite(arrays["fhr_stored"]) if valid is None else (valid & np.isfinite(arrays["fhr_stored"]))
        found = events.detect_decelerations(arrays["fhr_stored"], valid=valid)
        event_spans["decelerations"] = [
            [float(a), float(b), float(c)]
            for a, b, c in zip(found["onset_raw"], found["nadir_raw"], found["end_raw"])
        ]

    # ---- The streams the model reads ----------------------------------------------------------------
    target_gate = model.target_gate
    target_keep = (
        np.arange(inputs.target_features.shape[-1]) if target_gate is None
        else _host_vector(target_gate.keep_index).astype(int)
    )
    source_gate = model.source_gate
    source_keep = (
        np.arange(inputs.u_stream.shape[-1]) if source_gate is None
        else _host_vector(source_gate.keep_index).astype(int)
    )
    block_split = int(model.TARGET_BLOCK_SPLIT)
    shard = (batch_field(batch, "source_file") or [None])[0]
    target_labels = _kept_channel_labels(shard, ("fhr_st", "fhr_ph"), target_keep)
    source_blocks = ("up_st", "up_ph") if bool(getattr(model, "use_up_st", False)) else ("up_ph",)
    source_labels = _kept_channel_labels(shard, source_blocks, source_keep)
    target_stream = _to_numpy(inputs.target_features)[:, target_keep]
    source_stream = _to_numpy(inputs.u_stream)[:, source_keep]
    truth = np.asarray(_scored_clock_view(target_stream, model.target_forecast_shift), dtype=np.float64)
    arrays["target"], arrays["source"] = target_stream, source_stream
    arrays["truth"] = truth
    is_st = target_keep < block_split
    target_warmup = _warmup_vector(getattr(model, "target_warmup_steps", None), target_keep.size)
    source_warmup = _warmup_vector(getattr(model, "source_warmup_steps", None), source_keep.size)
    group = "Inputs"
    items.append(_trace("target_ch", group, "FHR feature, channel", "z", "step", "target", kind="col", param=["tch"]))
    items.append(_trace("source_ch", group, "UA feature, channel", "z", "step", "source", kind="col", param=["sch"]))
    maps.append(_heatmap("target_st", group, "FHR scattering input", "z", "target", y_title="kept channel", symmetric=True,
                         staircase="target_warmup", extra={"rows": np.flatnonzero(is_st).tolist()}))
    maps.append(_heatmap("target_ph", group, "FHR phase-harmonic input", "z", "target", y_title="kept channel", symmetric=True,
                         staircase="target_warmup", extra={"rows": np.flatnonzero(~is_st).tolist()}))
    maps.append(_heatmap("target_all", group, "FHR input, all kept channels", "z", "target", y_title="kept channel", symmetric=True,
                         staircase="target_warmup"))
    maps.append(_heatmap("source_all", group, "UA input, all kept channels", "z", "source", y_title="kept channel", symmetric=True,
                         staircase="source_warmup"))
    if inputs.source_valid is not None:
        kept_valid = _to_numpy(inputs.source_valid)[:, source_keep]
        arrays["source_valid"] = kept_valid
        maps.append(_heatmap("source_valid", "Data quality", "UA channel usable at step", "0 or 1", "source_valid", y_title="kept channel"))
        arrays["ua_valid_frac"] = kept_valid.mean(axis=1)
        items.append(_trace("ua_valid_frac", "Data quality", "Share of UA channels usable", "fraction", "step", "ua_valid_frac"))
    if weight_batch is not None:
        arrays["weight"] = _to_numpy(inputs.weight).reshape(-1)
        items.append(_trace("weight", "Data quality", "Objective weight at step", "0 or 1", "step", "weight"))

    # ---- Latent state ------------------------------------------------------------------------------
    latent_names = {"mu_prior": "Prior mean", "mu_post": "Posterior mean", "logvar_prior": "Prior log-variance",
                    "logvar_post": "Posterior log-variance", "z_post": "Posterior sample"}
    latents = {name: on_anchors(_to_numpy(outs[name])) for name in latent_names if name in outs}
    for name, title in latent_names.items():
        if name not in latents:
            continue
        arrays[name] = latents[name]
        limit = "latent" if name.startswith("mu") or name == "z_post" else None
        maps.append(_heatmap(name, "Latent", title, "latent units" if limit else "log variance", name,
                             symmetric=limit is not None, limit=limit))
    delta_mu = latents["mu_post"] - latents["mu_prior"]
    arrays["delta_mu"] = delta_mu
    maps.append(_heatmap("delta_mu", "Latent", "Source shift, posterior minus prior mean", "latent units", "delta_mu",
                         symmetric=True, limit="latent_shift"))
    kld_per_dim = on_anchors(_to_numpy(inputs.kld_per_dim))
    arrays["kld_per_dim"] = kld_per_dim
    maps.append(_heatmap("kld_per_dim", "Latent", "KL per latent dimension", "nats", "kld_per_dim", log=True, limit="kld_dims"))

    group = "KL and gate"
    arrays["kld"] = on_anchors(_to_numpy(outs["kld_per_t"]).reshape(-1))
    items.append(_trace("kld", group, "KL per step, K_t", "nats", "step", "kld"))
    if "kld_per_t_per_head" in outs:
        arrays["kld_head"] = on_anchors(_to_numpy(outs["kld_per_t_per_head"]))
        items.append(_trace("kld_head", group, "KL by head", "nats", "step", "kld_head", kind="col", param=["head"]))
    arrays["n_active"] = np.where(np.isfinite(kld_per_dim[:, 0]), (np.nan_to_num(kld_per_dim) > KLD_ACTIVE_EPS).sum(axis=1), np.nan)
    items.append(_trace("n_active", group, "Active latent dimensions", "count", "step", "n_active"))
    for name, word in (("event_gate", "Applied event gate"), ("event_gate_learned", "Learned event gate"), ("source_open", "UA keys open")):
        if name in outs:
            arrays[name] = on_anchors(_to_numpy(outs[name]).reshape(-1))
            items.append(_trace(name, group, word, "0 to 1", "step", name))
    if "source_invalid_mass" in outs:
        arrays["invalid_mass"] = on_anchors(_to_numpy(outs["source_invalid_mass"]).mean(axis=-1))
        items.append(_trace("invalid_mass", group, "Attention mass on flagged UA keys", "fraction", "step", "invalid_mass"))

    group = "Latent"
    arrays["mu_prior_norm"] = np.linalg.norm(latents["mu_prior"], axis=1)
    arrays["delta_mu_norm"] = np.linalg.norm(delta_mu, axis=1)
    items.append(_trace("mu_prior_norm", group, "Prior mean norm", "latent units", "step", "mu_prior_norm"))
    items.append(_trace("delta_mu_norm", group, "Source shift norm", "latent units", "step", "delta_mu_norm"))
    for name, word in (("logvar_prior", "Mean prior log-variance"), ("logvar_post", "Mean posterior log-variance")):
        arrays[f"{name}_mean"] = _nan_reduce(np.nanmean, latents[name], axis=1)
        items.append(_trace(f"{name}_mean", group, word, "log variance", "step", f"{name}_mean"))
    for name, word, unit in (("mu_prior", "Prior mean, dimension", "latent units"), ("mu_post", "Posterior mean, dimension", "latent units"),
                             ("delta_mu", "Source shift, dimension", "latent units"), ("kld_per_dim", "KL, dimension", "nats")):
        items.append(_trace(f"{name}_d", group, word, unit, "step", name, kind="col", param=["z"]))

    # ---- Lag attention ------------------------------------------------------------------------------
    group = "Lag"
    attention = on_anchors(_to_numpy(outs["attn_weights"]))
    kl_lag = on_anchors(_to_numpy(outs["source_kl_lag_map"]))
    arrays["attn"], arrays["kl_lag"] = attention, kl_lag
    head_mean = _nan_reduce(np.nanmean, attention, axis=1)
    arrays["attn_mean"] = head_mean
    maps.append(_heatmap("attn", group, "Lag attention weight", "weight", "attn", kind="head", y_title=lag_axis.COEFFICIENT_LAG_AXIS_LABEL,
                         y_values="lag_s", log=True, param=["hm"], limit="lag_attn", ridge="attn_argmax"))
    maps.append(_heatmap("kl_lag", group, "KL attribution by lag", "nats", "kl_lag", y_title=lag_axis.COEFFICIENT_LAG_AXIS_LABEL,
                         y_values="lag_s", log=True, limit="kl_lag_map", ridge="kl_argmax"))
    if "source_key_mask" in outs:
        arrays["key_mask"] = on_anchors(_to_numpy(outs["source_key_mask"]))
        maps.append(_heatmap("key_mask", "Data quality", "Usable UA key by lag", "0 or 1", "key_mask", y_title=lag_axis.COEFFICIENT_LAG_AXIS_LABEL, y_values="lag_s"))
    arrays["attn_argmax"] = _profile_argmax(head_mean, lag_seconds)
    arrays["attn_centroid"] = _lag_centroid(head_mean, lag_seconds)
    arrays["attn_peak"] = _nan_reduce(np.nanmax, head_mean, axis=1)
    arrays["kl_argmax"] = _profile_argmax(kl_lag, lag_seconds)
    arrays["kl_centroid"] = _lag_centroid(kl_lag, lag_seconds)
    entropy = attention_entropy(outs["attn_weights"].nan_to_num(0.0)).mean(dim=-1)
    arrays["attn_entropy"] = on_anchors(_to_numpy(entropy).reshape(-1))
    unit = "s"
    items.append(_trace("attn_argmax", group, "Max lag of attention, head mean", unit, "step", "attn_argmax"))
    head_peaks = np.stack([_profile_argmax(attention[:, head, :], lag_seconds) for head in range(attention.shape[1])], axis=1)
    arrays["attn_argmax_head"] = head_peaks
    items.append(_trace("attn_argmax_head", group, "Max lag of attention by head", unit, "step", "attn_argmax_head", kind="col", param=["head"]))
    items.append(_trace("attn_centroid", group, "Attention lag centroid", unit, "step", "attn_centroid"))
    items.append(_trace("attn_peak", group, "Peak attention weight", "weight", "step", "attn_peak"))
    items.append(_trace("attn_entropy", group, "Attention entropy", "nats", "step", "attn_entropy"))
    items.append(_trace("attn_lag", group, "Attention by lag", "weight", "step", "attn_mean", kind="col", param=["lag"]))
    items.append(_trace("kl_argmax", group, "Max lag of KL attribution", unit, "step", "kl_argmax"))
    items.append(_trace("kl_centroid", group, "KL lag centroid", unit, "step", "kl_centroid"))
    items.append(_trace("kl_lag", group, "KL attribution by lag value", "nats", "step", "kl_lag", kind="col", param=["lag"]))

    # ---- Hidden representations ------------------------------------------------------------------
    group = "Representations"
    for name, title in (("target_state", "FHR encoder state"), ("source_state", "UA encoder state")):
        if name in outs:
            arrays[name] = _to_numpy(outs[name])
            maps.append(_heatmap(name, group, title, "activation", name, symmetric=True, y_title="unit"))

    # ---- Forecast ------------------------------------------------------------------------------------
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    scores = _forecast_readouts(model, inputs, truth, target_stream, likelihood)
    arrays["z_full"] = scores["z_full"]
    arrays["forecast_mask"] = scores["forecast_mask"]
    group = "Forecast score"
    for name, word, unit in (("nll_base", "NLL base, this forward", "nats"), ("nll_full", "NLL full, this forward", "nats"),
                             ("pred_gap", "Forecast gap, base minus full", "nats"),
                             ("mse_base", "MSE base, scored cells", "z squared"), ("mse_full", "MSE full, scored cells", "z squared"),
                             ("mse_persist", "MSE persistence, scored cells", "z squared"),
                             ("coverage", "Forecast window coverage", "fraction"), ("contributing", "Anchor is scored", "0 or 1")):
        arrays[f"score_{name}"] = scores[name]
        items.append(_trace(name, group, word, unit, "anchor", f"score_{name}"))
    for name in ("mu_base", "logvar_base", "mu_full", "logvar_full"):
        arrays[name] = _to_numpy(outs[name])
    group = "Forecast cell"
    fields = (("mu_full", "full mean"), ("mu_base", "base mean"), ("sigma_full", "full innovation sigma"), ("sigma_base", "base innovation sigma"),
              ("err_full", "full absolute error"), ("err_base", "base absolute error"), ("skill", "skill, base minus full error"),
              ("z_full", "full innovation z score"))
    items.append(_trace("truth_tau", group, "Truth at the forecast step", "z", "target", "truth", kind="fc", param=["tch", "tau"], extra={"f": "truth"}))
    for name, word in fields:
        unit = "z"
        items.append(_trace(f"fc_{name}", group, f"Forecast, {word}", unit, "target", "truth", kind="fc", param=["tch", "tau"], extra={"f": name}))
    group = "Forecast by channel"
    for name, word in (("truth", "truth"), *fields):
        symmetric = name in ("mu_full", "mu_base", "truth", "skill", "z_full")
        maps.append(_heatmap(f"tau_{name}", group, f"At one horizon step, {word}", "z", "truth", kind="fc_tau", y_title="kept channel",
                             symmetric=symmetric, param=["tau"], limit={"sigma_full": "pred_sigma", "sigma_base": "pred_sigma", "err_full": "pred_error", "err_base": "pred_error", "skill": "pred_skill"}.get(name),
                             extra={"f": name}))
    group = "Forecast by horizon"
    for name, word in (("truth", "truth"), *fields):
        symmetric = name in ("mu_full", "mu_base", "truth", "skill", "z_full")
        maps.append(_heatmap(f"ch_{name}", group, f"One channel over horizon, {word}", "z", "truth", kind="fc_ch", y_title="horizon (s ahead)",
                             y_values="tau_s", symmetric=symmetric, param=["tch"], extra={"f": name}))

    # ---- The events' age -------------------------------------------------------------------------------
    onsets = np.asarray([span[0] for span in event_spans["contractions"]], dtype=np.float64)
    ages = np.full(n_steps, np.nan)
    raw_seconds = step_seconds / raw_per_step
    for step in range(n_steps):
        # The step's causal endpoint on the raw grid, as the collection pass measures the age.
        endpoint = raw_per_step * (step + 1.0) - 1.0
        earlier = onsets[onsets <= endpoint]
        if earlier.size:
            ages[step] = (endpoint - earlier.max()) * raw_seconds
    arrays["contraction_age"] = ages
    items.append(_trace("contraction_age", "KL and gate", "Time since last contraction onset", "s", "step", "contraction_age"))

    terms = forecast_likelihood_terms(model)
    cell_mask = terms.get("cell_mask")
    ar_coef = terms.get("ar_coef")
    arrays["ar_coef"] = (
        np.zeros(target_keep.size) if ar_coef is None
        else ar_coef.detach().cpu().to(torch.float64).numpy()
    )
    arrays["cell_mask"] = (
        np.ones((horizon, target_keep.size)) if cell_mask is None
        else cell_mask.detach().cpu().to(torch.float64).numpy()
    )

    epoch = float(row["epoch"]) if np.isfinite(float(row["epoch"])) else 0.0
    t_zero = float(traces.absolute_seconds(epoch, 0.0))
    meta = {
        "index": int(index),
        "guid": str(row["guid"]),
        "class": None if labels.CLASS_COLUMN not in row.index or pd.isna(row[labels.CLASS_COLUMN]) else str(row[labels.CLASS_COLUMN]),
        "subgroup": None if labels.SUBGROUP_COLUMN not in row.index or pd.isna(row[labels.SUBGROUP_COLUMN]) else str(row[labels.SUBGROUP_COLUMN]),
        "epoch": epoch,
        "hours_before_delivery": -epoch / 3600.0,
        "labour_onset": _json_number(_scalar_field(batch, "time_from_labor_onset")),
        "second_stage": _json_number(_scalar_field(batch, "second_stage_onset")),
        "run": run_label,
        "metrics": {name: _json_number(row[name]) if name in row.index else None for name, _ in GALLERY_METRICS},
        "lag_note": lag_axis.GROUP_DELAY_NOTE,
    }
    parameters = {
        "tch": {"label": "FHR channel", "options": target_labels},
        "sch": {"label": "UA channel", "options": source_labels},
        "z": {"label": "Latent dim", "options": [f"z {d}" for d in range(arrays["mu_prior"].shape[1])]},
        "head": {"label": "Head", "options": [f"head {m}" for m in range(attention.shape[1])]},
        "hm": {"label": "Head", "options": ["mean of heads", *(f"head {m}" for m in range(attention.shape[1]))]},
        "lag": {"label": "Lag", "options": [f"{value:g} s" for value in lag_seconds]},
        "tau": {"label": "Horizon step", "options": [f"{(tau + 1) * step_seconds:g} s ahead" for tau in range(horizon)]},
    }
    return {
        "meta": meta,
        "geometry": {
            "T": n_steps, "H": horizon, "dt": step_seconds, "raw_dt": step_seconds / raw_per_step,
            "raw_n": n_steps * raw_per_step, "t0": t_zero, "warmup": int(geometry.warmup),
            "t_valid": int(geometry.t_valid),
        },
        "axes": {
            "anchors": anchors.tolist(),
            "anchor_valid": anchor_valid.astype(int).tolist(),
            "lag_s": lag_seconds.tolist(),
            "tau_s": [(tau + 1) * step_seconds for tau in range(horizon)],
            "target_warmup": target_warmup.tolist(),
            "source_warmup": source_warmup.tolist(),
        },
        "events": event_spans,
        "traces": items,
        "heatmaps": maps,
        "params": parameters,
        "limits": {name: float(value) for name, value in (row_limits or {}).items()},
        "arrays": {name: encode_array(value) for name, value in arrays.items()},
    }


def _scalar_field(batch: Any, name: str) -> Optional[float]:
    """A per-sample scalar field of a one-sample batch, or ``None`` when the batch has none."""
    value = batch_field(batch, name)
    if value is None:
        return None
    return float(np.asarray(_to_numpy(value)).reshape(-1)[0])


# =================================================================================================
# Writing the pages
# =================================================================================================
def _asset(name: str) -> str:
    """Read one of the page's template files."""
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


def render_page(payload: Mapping[str, Any]) -> str:
    """Fill the page template with one segment's payload.

    Args:
        payload: From :func:`build_payload`.

    Returns:
        The page's HTML text.
    """
    meta = payload["meta"]
    title = f"Sample {meta['index']:04d} dashboard"
    # A closing tag inside a JSON string would end the script block early.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (
        _asset("dashboard.html")
        .replace("__TITLE__", html.escape(title))
        .replace("__PLOTLY__", PLOTLY_FILENAME)
        .replace("__INDEX__", INDEX_SCRIPT_FILENAME)
        .replace("__GALLERY__", GALLERY_FILENAME)
        .replace("__CSS__", _asset("dashboard.css"))
        .replace("__PAYLOAD__", data)
        .replace("__JS__", _asset("dashboard.js"))
    )


@dataclass
class DashboardSet:
    """The dashboards of one run: one page beside the printed pages of each selected segment.

    Attributes:
        root: The samples directory. Each draw's folder sits under it.
        scales: The raw-signal scales of :func:`~teb_vae.lag_attn_cfs.eval.traces.raw_signal_scales`.
        row_limits: The run's shared colour limits.
        delay_steps: The model's causal input delay.
        run_label: What each page's header calls this run.
        cap: How many segments get a page, or ``None`` for every selected segment.
        entries: One record per segment with a page, keyed by dataset index.
        failures: One record per page that could not be written.
    """

    root: Path
    scales: Mapping[str, Tuple[float, float]]
    row_limits: Optional[Mapping[str, float]] = None
    delay_steps: int = 0
    run_label: str = ""
    cap: Optional[int] = None
    entries: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def file_name(self, stem: str) -> str:
        """The name of the page for the printed page called ``stem``."""
        return f"{stem}{DASHBOARD_SUFFIX}"

    def place_copy(self, index: int, directory: Path, selection: str) -> Optional[str]:
        """Copy a segment's page into another draw's folder.

        Args:
            index: The dataset index.
            directory: The folder of the draw.
            selection: The name of the draw.

        Returns:
            The file name, or ``None`` when the segment has no page.
        """
        entry = self.entries.get(int(index))
        if entry is None:
            return None
        target = Path(directory) / entry["file"]
        if selection not in entry["selections"]:
            shutil.copy2(entry["path"], target)
            entry["selections"].append(selection)
        return entry["file"]

    def has_room(self) -> bool:
        """Whether the cap leaves room for another segment."""
        return self.cap is None or len(self.entries) < int(self.cap)

    def write(
        self,
        task: Any,
        inputs: SegmentInputs,
        row: pd.Series,
        *,
        index: int,
        stem: str,
        directory: Path,
        selection: str,
    ) -> Optional[str]:
        """Write one segment's page beside its printed pages, recording a failure rather than raising.

        Args:
            task: The loaded task.
            inputs: The segment's tensors.
            row: The resolved table row.
            index: The dataset index.
            stem: The stem of the printed page, which this page's name extends.
            directory: The folder of the draw.
            selection: The name of the draw.

        Returns:
            The file name, or ``None`` when the cap or a failure left no page.
        """
        directory = Path(directory)
        if int(index) in self.entries:
            return self.place_copy(index, directory, selection)
        if not self.has_room():
            return None
        name = self.file_name(stem)
        try:
            payload = build_payload(
                task, inputs, row, index=index, delay_steps=self.delay_steps, scales=self.scales,
                row_limits=self.row_limits, run_label=self.run_label,
            )
            (directory / name).write_text(render_page(payload), encoding="utf-8")
        except Exception as error:  # noqa: BLE001 - one page is not worth the rest of the run
            logger.warning(f"dashboard for dataset index {index} failed: {error}")
            self.failures.append({"dataset_index": int(index), "guid": str(row["guid"]),
                                  "selection": selection,
                                  "error": f"{type(error).__name__}: {error}"})
            return None
        meta = payload["meta"]
        self.entries[int(index)] = {
            "file": name,
            "path": directory / name,
            "index": int(index),
            "guid": meta["guid"],
            "class": meta["class"],
            "subgroup": meta["subgroup"],
            "epoch": meta["epoch"],
            "hours": meta["hours_before_delivery"],
            "metrics": meta["metrics"],
            "contractions": len(payload["events"]["contractions"]),
            "decelerations": len(payload["events"]["decelerations"]),
            "selections": [selection],
        }
        return name

    def _write_gallery(self, folder: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write one folder's page index and gallery."""
        script = "window.DASHBOARD_INDEX=" + json.dumps(list(rows), separators=(",", ":")).replace("</", "<\\/") + ";"
        (folder / INDEX_SCRIPT_FILENAME).write_text(script, encoding="utf-8")
        gallery = (
            _asset("gallery.html")
            .replace("__INDEX__", INDEX_SCRIPT_FILENAME)
            .replace("__CSS__", _asset("dashboard.css"))
            .replace("__METRICS__", json.dumps(list(GALLERY_METRICS)))
        )
        (folder / GALLERY_FILENAME).write_text(gallery, encoding="utf-8")

    def finalise(self) -> Dict[str, Any]:
        """Write each folder's plotting library, page index and gallery, and the run's gallery.

        Returns:
            The record for ``summary.json``: the counts and any failures.
        """
        if not self.entries:
            return {"n_segments": 0, "n_files": 0, "failures": self.failures}
        from plotly.offline import get_plotlyjs

        library = get_plotlyjs()
        ordered = sorted(self.entries.values(), key=lambda entry: (str(entry["class"]), entry["hours"]))
        public = [{key: value for key, value in entry.items() if key != "path"} for entry in ordered]
        folders = sorted({selection for entry in public for selection in entry["selections"]})
        everything: List[Dict[str, Any]] = []
        for selection in folders:
            folder = self.root / selection
            (folder / PLOTLY_FILENAME).write_text(library, encoding="utf-8")
            rows = [entry for entry in public if selection in entry["selections"]]
            self._write_gallery(folder, rows)
            everything.extend({**entry, "file": f"{selection}/{entry['file']}", "selections": [selection]} for entry in rows)
        self._write_gallery(self.root, everything)
        logger.info(f"dashboards: wrote {len(everything)} file(s) for {len(public)} segment(s) under {self.root}")
        return {
            "n_segments": len(public),
            "n_files": len(everything),
            "gallery": GALLERY_FILENAME,
            "failures": self.failures,
        }


__all__ = [
    "DASHBOARD_CAP",
    "DashboardSet",
    "SegmentInputs",
    "build_payload",
    "encode_array",
    "render_page",
]
