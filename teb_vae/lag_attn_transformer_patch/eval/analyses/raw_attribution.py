r"""``raw_attribution`` -- where in raw UP history the belief change and the forecast gain come from.

Owner E1-C (``notes/EVAL_PLAN.md`` §2; proposals P1, P2, P5, P6, P7 of ``notes/EVAL_MAP_CAPTUM.md``).

**Question.** At the shared high-KL clean anchors, which raw UP samples -- at which offset behind the
anchor, at 0.25 s resolution over the lag window -- moved the divergence ``kld``, the mean-decoded
forecast gap ``pred_gap`` and the UP-driven forecast-level shift ``delta_level``
($\overline{\mu^{full}_{\tau,level} - \mu^{base}_{\tau,level}}$, standardized level units), and is
that where the model's own lag readout ``source_kl_lag_map`` (``K_t * alpha``) puts it?

**Method.** Integrated gradients from the source-null baseline (raw UP = 0, which patchifies to the
exact null the source-null control feeds) entered at ``core.BASELINE_ENTRY_FRACTION``, FHR held at its
own values, through :class:`~..raw.RawReadout`. One ``LayerIntegratedGradients`` call per readout on
the **input of** ``source_adapter`` with ``multiply_by_inputs=False`` returns the path-integrated
gradient ``G`` over the UP patch tokens; because ``patchify`` is affine in the raw samples with the
validity channel fixed, the raw straight path *is* the patch straight path, so

* patch IG ``= (p - p0) * G`` (value / delta / validity channels; validity is exactly 0), and
* raw IG ``= (x - x0) * A^T G`` (one vector-Jacobian product through ``patchify``),

both complete with the same sum. Also per anchor: a ``LayerIntegratedGradients`` split through
``lag_attn.W_k`` / ``W_v`` (does UP decide *where* to look, or carry *what* is read), and the IG of each
head's own divergence ``kld_per_t_per_head[m]`` against that head's attention ``alpha^(m)``.

**Reading.** A raw-lag ``|IG|`` profile that sits where ``source_kl_lag_map`` sits (high ``lag_corr``,
low ``lag_js``) says the attention's lag is where UP content causally moves the KL. ``|IG|`` near lag 0
while the attention peaks far says the attention mass is parked and recent UP drives the KL. The
lag here is the raw-sample offset from the anchor token's last sample: no transform group delay.
Token ``t`` reads samples ``[16t, 16t + 16)`` plus sample ``16t - 1`` (its first delta), so a readout at
anchor ``t_a`` reads raw lags ``0 .. 16 L`` exactly; ``outside_window_max_abs`` and
``after_anchor_max_abs`` check that support on every row.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from loguru import logger

from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import class_contrast, lag_hist, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_transformer_patch.eval import figures as patch_figures
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record
from teb_vae.lag_attn_transformer_patch.nets.patching import patchify

NAME = "raw_attribution"

#: ``(headline_name, key)`` pairs registered as ``("raw_attribution", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("raw_attr_kld_lag_corr", "kld_lag_corr"),
    ("raw_attr_kld_centroid_s", "kld_centroid_s"),
    ("raw_attr_completeness_max", "completeness_rel_max"),
)

#: Caps under ``eval_config.caps`` and their defaults. Segments: one per drawn recording.
CAP_SEGMENTS, DEFAULT_SEGMENTS = f"{NAME}_segments", 24
CAP_ANCHORS, DEFAULT_ANCHORS = f"{NAME}_anchors", 4
CAP_IG_STEPS, DEFAULT_IG_STEPS = f"{NAME}_ig_steps", 128
CAP_EXAMPLES, DEFAULT_EXAMPLES = f"{NAME}_examples_per_class", 1
#: Segments per forward batch: rows per Captum call stay <= 4 x anchors (x heads for the head rows),
#: so one interpolation chunk of ``IG_BATCH`` rows fits a 6 GB card at production width.
SEGMENTS_PER_BATCH = 4
IG_BATCH = 64
SEED_OFFSET = 29

READOUT_DELTA_LEVEL = "delta_level"
READOUT_KLD_HEAD = "kld_head"
#: The readouts mapped at raw resolution, and the ones split through the attention's keys/values.
READOUTS: Tuple[str, ...] = (core.READOUT_KLD, core.READOUT_PRED_GAP, READOUT_DELTA_LEVEL)
UNITS: Mapping[str, str] = {core.READOUT_KLD: "nats", core.READOUT_PRED_GAP: "nats",
                            READOUT_DELTA_LEVEL: "standardized level", READOUT_KLD_HEAD: "nats"}
SHORT: Mapping[str, str] = {core.READOUT_KLD: "Divergence $K_t$", core.READOUT_PRED_GAP: "Forecast gap",
                            READOUT_DELTA_LEVEL: "UP level shift", READOUT_KLD_HEAD: "Head $K_t^{(m)}$"}
COLOURS: Mapping[str, str] = {core.READOUT_KLD: figures.COLOR_BLUE, core.READOUT_PRED_GAP: figures.COLOR_VERMILLION,
                              READOUT_DELTA_LEVEL: figures.COLOR_GREEN}
NOTE = "Model sensitivity, not a causal effect. Lag: raw-sample offset (0.25 s) from the anchor token's end."
LAG_LABEL = "lag behind the anchor (s, raw samples)"


# =============================================================================
# The readouts this analysis adds to the shared registry
# =============================================================================
class _Readout(core.AnchorReadout):
    """The shared readout, plus ``delta_level`` and ``kld_head`` (head = the row's ``coordinate``)."""

    def __init__(self, model: Any, readout: str, likelihood: str) -> None:
        super().__init__(model, core.ATTENTION_CELL,
                         readout=readout if readout in core.READOUTS else core.READOUT_KLD, likelihood=likelihood)
        self.readout = readout

    def forward(self, y_st, y_ph, u_stream, target_features, weight, columns, coordinates):  # noqa: D102
        if self.readout in core.READOUTS:
            return super().forward(y_st, y_ph, u_stream, target_features, weight, columns, coordinates)
        anchors = self.anchor_steps(columns)
        with core.attributed_forward(self.model, anchors):
            out = self.model(y_st, y_ph, u_stream)
        tie = 0.0 * (y_st.sum() + y_ph.sum() + u_stream.sum() + out["mu_post"].sum() + out["logvar_post"].sum())
        if self.readout == READOUT_DELTA_LEVEL:
            return (out["mu_full"][:, 0, :, 0] - out["mu_base"][:, 0, :, 0]).mean(dim=-1) + tie
        rows = torch.arange(columns.shape[0], device=columns.device)
        return out["kld_per_t_per_head"][rows, anchors, coordinates] + tie


# =============================================================================
# Shared plumbing (hoist candidates for raw.py: the selection loop and the stacked page)
# =============================================================================


def _lag_bands(eval_config: Mapping[str, Any], n_lags: int) -> Dict[str, Tuple[int, int]]:
    """The configured token-lag bands, clipped to the window; a non-pair entry is ignored."""
    bands: Dict[str, Tuple[int, int]] = {}
    for name, span in dict(eval_config.get("occlusion_bands") or {}).items():
        if isinstance(span, (list, tuple)) and len(span) == 2 and int(span[0]) < n_lags:
            bands[str(name)] = (int(span[0]), min(int(span[1]), n_lags - 1))
    return bands


def _raw_signals_physical(batch: Any, rows: pd.DataFrame, scales: Mapping[str, Tuple[float, float]]) -> List[Any]:
    """Each sample's raw FHR and UP in physical units, through the traces' own path (gaps are NaN)."""
    holders = [traces.SegmentTrace(guid=str(r["guid"]), epoch=float(r["epoch"]), clinical_class=None, subgroup=None,
                                   anchor=np.zeros(0, dtype=np.int64), contributing=np.zeros(0, dtype=bool))
               for _, r in rows.iterrows()]
    traces.attach_raw_signals(holders, batch, scales)
    return holders




# =============================================================================
# The attributions
# =============================================================================
def _forward_values(wrapper: Any, inputs: Sequence[torch.Tensor], args: Sequence[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return wrapper(*inputs, *args)


def patch_and_raw_ig(wrapper: Any, inputs: Sequence[torch.Tensor], extra: Sequence[torch.Tensor],
                     columns: torch.Tensor, coordinates: torch.Tensor, *, n_steps: int,
                     valid: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, np.ndarray]:
    r"""Source-null IG of one readout over raw UP, at raw and at patch resolution from one Captum call.

    Args:
        wrapper: A :class:`~..raw.RawReadout`.
        inputs: Row-expanded ``(fhr (N,T,R), empty (N,T,0), up (N,T,R))``.
        extra: Row-expanded ``(summaries, weight)``.
        columns: Per-row anchor-axis positions.
        coordinates: Per-row coordinate (the head for ``kld_head``).
        n_steps: Integration steps.
        valid: Row-expanded ``(fhr_valid, up_valid)`` (``raw.RawReadout.prepare``), held fixed.

    Returns:
        ``raw (N,T,R)``, ``patch (N,T,2R+1)``, ``value_input``, ``value_null`` (UP = 0), ``value_entry``.
    """
    model = wrapper.model
    fhr3, empty, up3 = (x.detach() for x in inputs)
    start = core.BASELINE_ENTRY_FRACTION * up3          # x0 = b + a0 (x - b), b = 0
    args = (*extra, columns, coordinates, *valid)
    value_input = _forward_values(wrapper, (fhr3, empty, up3), args)
    value_null = _forward_values(wrapper, (fhr3, empty, torch.zeros_like(up3)), args)
    value_entry = _forward_values(wrapper, (fhr3, empty, start), args)
    gradients = LayerIntegratedGradients(wrapper, model.source_adapter, multiply_by_inputs=False).attribute(
        (fhr3, empty, up3), baselines=(fhr3, empty, start), additional_forward_args=args, n_steps=int(n_steps),
        internal_batch_size=max(IG_BATCH, int(columns.shape[0])), attribute_to_layer_input=True,
    )
    gradients = gradients[0] if isinstance(gradients, (tuple, list)) else gradients
    weight, r = extra[1], int(model.raw_per_step)
    # The forward's own patchify: the original invalid samples back to NaN, so they are masked here too.
    up_valid = valid[1].flatten(1)
    nan = up3.new_full((), float("nan"))
    with torch.enable_grad():
        x = up3.flatten(1).clone().requires_grad_(True)
        patches = patchify(torch.where(up_valid, x, nan), weight, raw_per_step=r, validity=model.source_validity)
        (pull,) = torch.autograd.grad((patches * gradients.detach()).sum(), x)
    patch0 = patchify(torch.where(up_valid, start.flatten(1), nan), weight, raw_per_step=r, validity=model.source_validity)
    return {
        "raw": raw.host(((up3 - start).flatten(1) * pull).view_as(up3)),
        "patch": raw.host((patches.detach() - patch0) * gradients),
        "value_input": raw.host(value_input), "value_null": raw.host(value_null), "value_entry": raw.host(value_entry),
    }


def key_value_split(wrapper: Any, inputs: Sequence[torch.Tensor], extra: Sequence[torch.Tensor],
                    columns: torch.Tensor, coordinates: torch.Tensor, *, n_steps: int,
                    valid: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """Layer IG through ``lag_attn.W_k`` and ``W_v`` on the source-null path: per-token sums ``(N, T)`` each.

    The query is target-only and fixed on this path, so the two sum to ``f(x) - f(x0)``.
    """
    model = wrapper.model
    fhr3, empty, up3 = (x.detach() for x in inputs)
    start = core.BASELINE_ENTRY_FRACTION * up3
    with warnings.catch_warnings():
        # Captum warns that a list of layers must not be a chain; W_k and W_v each read kv_norm(h_u).
        warnings.simplefilter("ignore", category=UserWarning)
        method = LayerIntegratedGradients(wrapper, [model.lag_attn.W_k, model.lag_attn.W_v])
    parts = method.attribute(
        (fhr3, empty, up3), baselines=(fhr3, empty, start), additional_forward_args=(*extra, columns, coordinates, *valid),
        n_steps=int(n_steps), internal_batch_size=max(IG_BATCH, int(columns.shape[0])),
    )
    parts = [p[0] if isinstance(p, (tuple, list)) else p for p in parts]
    return raw.host(parts[0].sum(dim=-1)), raw.host(parts[1].sum(dim=-1))


# =============================================================================
# Reductions (numpy)
# =============================================================================
def raw_lag_profile(maps: np.ndarray, anchors: Sequence[int], r: int, n_lags: int) -> np.ndarray:
    r"""Re-index ``(N, T, R)`` raw maps by raw lag ``s = R t_a + R - 1 - n``, ``s = 0 .. R L``; NaN before the record."""
    flat = np.asarray(maps, dtype=np.float64).reshape(len(anchors), -1)
    lags = np.arange(r * n_lags + 1)
    index = (np.asarray(anchors, dtype=np.int64)[:, None] * r + r - 1) - lags[None, :]
    out = np.take_along_axis(flat, np.clip(index, 0, None), axis=1)
    return np.where(index >= 0, out, np.nan)


def support_checks(maps: np.ndarray, anchors: Sequence[int], r: int, n_lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Largest |IG| after the anchor token's last sample, and before the window's first read sample."""
    flat = np.abs(np.asarray(maps, dtype=np.float64).reshape(len(anchors), -1))
    n = np.arange(flat.shape[1])[None, :]
    a = np.asarray(anchors, dtype=np.int64)[:, None]
    after = np.where(n > r * a + r - 1, flat, 0.0).max(axis=1)
    before = np.where(n < r * (a - n_lags + 1) - 1, flat, 0.0).max(axis=1)
    return after, before


def shares(profile: np.ndarray) -> np.ndarray:
    """|profile| normalised per row to a distribution over its finite bins (NaN rows stay NaN)."""
    return lag_hist.normalise(np.abs(np.asarray(profile, dtype=np.float64)))


def centroid_s(share: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    return np.nansum(share * seconds[None, :], axis=1) / np.where(np.isfinite(share).any(axis=1), 1.0, np.nan)


def token_seconds(n_lags: int, r: int) -> np.ndarray:
    """The centre of token lag ``l`` on the raw-lag axis: raw lags ``R l .. R l + R - 1``."""
    return (np.arange(n_lags) * r + 0.5 * (r - 1)) / float(raw.events.FS_RAW)


def band_shares(share: np.ndarray, bands: Mapping[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    return {name: np.nansum(share[:, lo:hi + 1], axis=1) for name, (lo, hi) in bands.items()}


def profile_agreement(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per row: Pearson of two |token profiles| and the lag shift (s) of right behind left at peak cross-correlation."""
    corr = np.full(left.shape[0], np.nan)
    shift = np.full(left.shape[0], np.nan)
    for row in range(left.shape[0]):
        a, b = np.abs(left[row]), np.abs(right[row])
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < 3 or a[keep].std() == 0.0 or b[keep].std() == 0.0:
            continue
        a, b = a[keep], b[keep]
        corr[row] = float(np.corrcoef(a, b)[0, 1])
        cross = np.correlate(b - b.mean(), a - a.mean(), mode="full")
        shift[row] = float((int(np.argmax(cross)) - (a.size - 1)) * raw_seconds_per_token())
    return corr, shift


def raw_seconds_per_token() -> float:
    return float(core.SECONDS_PER_STEP)


# =============================================================================
# The pass
# =============================================================================
class _Work:
    """Rows, row-aligned vectors, per-anchor KL-vs-gain pairs, the example pages and the cost."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.vectors: Dict[str, List[np.ndarray]] = {}
        self.pairs: List[Dict[str, Any]] = []
        self.examples: List[Dict[str, Any]] = []
        self.forward_equivalents = 0
        self.n_unclean = 0

    def add(self, record: Dict[str, Any], **vectors: np.ndarray) -> None:
        self.rows.append(record)
        for name, value in vectors.items():
            self.vectors.setdefault(name, []).append(np.asarray(value, dtype=np.float32))


def _attribute_batch(task: Any, batch: Any, rows: pd.DataFrame, work: _Work, *, anchors_per_segment: int,
                     n_steps: int, bands: Mapping[str, Tuple[int, int]], scales: Mapping[str, Tuple[float, float]],
                     want_examples: Mapping[str, bool]) -> None:
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    r, n_lags, n_heads = int(model.raw_per_step), int(model.lag_attn.L), int(model.lag_attn.num_heads)
    fhr, up, weight = raw.batch_signals(batch)
    steps = int(weight.shape[1])
    # The dense forward and the target off the ORIGINAL raw, exactly as the evaluation's own pass reads it;
    # only the IG path gets finite raw, with the original validity held fixed beside it.
    summaries = model.summary_target(fhr, weight)
    with torch.no_grad():
        outputs = raw.forward_raw(model, fhr, up, weight)
    columns_per_sample, unclean = raw.informative_anchor_columns(model, outputs, weight, rows, anchors_per_segment)
    work.n_unclean += unclean
    fhr3, up3, fhr_valid, up_valid = raw.RawReadout.prepare(model, fhr, up, weight)
    inputs, extra, columns, sample = raw.expand_rows((fhr3, fhr3[..., :0], up3), (summaries, weight, fhr_valid, up_valid),
                                                     columns_per_sample)
    extra, valid = extra[:2], extra[2:]
    n_rows = int(columns.shape[0])
    if n_rows == 0:
        return
    sample_t = torch.as_tensor(sample, device=columns.device)
    anchors_t = outputs["anchor_index"][sample_t, columns]
    anchors = anchors_t.cpu().numpy().astype(np.int64)
    with torch.no_grad():
        model_profile = raw.host(outputs["source_kl_lag_map"][sample_t, anchors_t])                    # (N, L)
        alpha = raw.host(outputs["attn_weights"][sample_t, anchors_t])                                 # (N, M, L)
    zero = torch.zeros(n_rows, dtype=torch.long, device=columns.device)
    identity = [{"guid": str(row["guid"]), "epoch": float(row["epoch"]),
                 labels.CLASS_COLUMN: row.get(labels.CLASS_COLUMN), labels.SUBGROUP_COLUMN: row.get(labels.SUBGROUP_COLUMN)}
                for _, row in rows.iterrows()]
    seconds_raw = np.arange(r * n_lags + 1) / float(raw.events.FS_RAW)
    seconds_tok = token_seconds(n_lags, r)
    window = [np.arange(max(0, a - n_lags + 1), a + 1) for a in anchors]

    token_profiles: Dict[str, np.ndarray] = {}
    raw_maps: Dict[str, np.ndarray] = {}
    for readout in READOUTS:
        wrapper = raw.RawReadout(_Readout(model, readout, likelihood).eval())
        result = patch_and_raw_ig(wrapper, inputs, extra, columns, zero, n_steps=n_steps, valid=valid)
        keys, values = key_value_split(wrapper, inputs, extra, columns, zero, n_steps=n_steps, valid=valid)
        work.forward_equivalents += 2 * n_rows * (n_steps + 3)
        raw_maps[readout] = result["raw"]
        _reduce(work, result, readout=readout, head=-1, identity=identity, sample=sample, anchors=anchors,
                columns=columns, model_profile=model_profile, bands=bands, r=r, n_lags=n_lags, window=window,
                seconds_raw=seconds_raw, seconds_tok=seconds_tok, keys=keys, values=values, up_valid=raw.host(valid[1]) > 0)
        token_profiles[readout] = core.lag_profile(core.time_profile(result["raw"]), anchors, n_lags)

    # Per head: the head's own divergence against the head's own attention.
    repeat = lambda x: x.repeat_interleave(n_heads, dim=0)  # noqa: E731
    head_inputs, head_extra, head_valid = (tuple(repeat(x) for x in group) for group in (inputs, extra, valid))
    heads = torch.arange(n_heads, device=columns.device).repeat(n_rows)
    wrapper = raw.RawReadout(_Readout(model, READOUT_KLD_HEAD, likelihood).eval())
    result = patch_and_raw_ig(wrapper, head_inputs, head_extra, repeat(columns), heads, n_steps=n_steps, valid=head_valid)
    work.forward_equivalents += n_rows * n_heads * (n_steps + 3)
    head_rows = np.repeat(np.arange(n_rows), n_heads)
    _reduce(work, result, readout=READOUT_KLD_HEAD, head=heads.cpu().numpy(), identity=identity,
            sample=sample[head_rows], anchors=anchors[head_rows], columns=repeat(columns),
            model_profile=alpha.reshape(n_rows * n_heads, n_lags), bands=bands, r=r, n_lags=n_lags,
            window=[window[i] for i in head_rows], seconds_raw=seconds_raw, seconds_tok=seconds_tok,
            up_valid=raw.host(head_valid[1]) > 0)

    # KL against gain, per anchor.
    for other in (core.READOUT_PRED_GAP, READOUT_DELTA_LEVEL):
        corr, shift = profile_agreement(token_profiles[core.READOUT_KLD], token_profiles[other])
        for row in range(n_rows):
            work.pairs.append({**identity[int(sample[row])], "anchor": int(anchors[row]), "pair": f"kld~{other}",
                               "profile_corr": float(corr[row]), "lag_shift_s": float(shift[row])})

    # The example page of a class that still wants one: the segment's first (highest-KL) anchor.
    if not any(want_examples.values()):
        return
    holders = None
    for element, (_, row) in enumerate(rows.iterrows()):
        key = str(row.get(labels.CLASS_COLUMN))
        ranked_out = "example_rank" in row.index and not np.isfinite(float(row["example_rank"]))
        if not want_examples.get(key) or ranked_out:
            continue
        offsets = np.flatnonzero(sample == element)
        if not offsets.size:
            continue
        holders = holders or _raw_signals_physical(batch, rows, scales)
        offset = int(offsets[0])
        heads_of = np.flatnonzero(head_rows == offset)
        head_profiles = core.lag_profile(core.time_profile(result["raw"][heads_of]), [anchors[offset]] * n_heads, n_lags)
        work.examples.append({
            **identity[element], "anchor": int(anchors[offset]), "horizon": int(model.horizon), "steps": steps,
            "n_lags": n_lags, "raw": holders[element].raw, "raw_units": holders[element].raw_units,
            "maps": {name: _blank_unread(raw_maps[name][offset], int(anchors[offset]), r, n_lags,
                                         raw.host(valid[1][offset]) > 0) for name in READOUTS},
            "lag_shares": {name: shares(raw_lag_profile(raw_maps[name][offset:offset + 1], [anchors[offset]], r, n_lags))[0]
                           for name in READOUTS},
            "token_shares": {name: shares(token_profiles[name][offset:offset + 1])[0] for name in READOUTS},
            "model_share": shares(model_profile[offset:offset + 1])[0],
            "head_shares": shares(head_profiles), "alpha_shares": shares(alpha[offset]),
        })
        want_examples[key] = False


def _blank_unread(field: np.ndarray, anchor: int, r: int, n_lags: int, valid: np.ndarray) -> np.ndarray:
    """A ``(T, R)`` raw map with every sample the readout never read set to NaN (drawn blank): outside the
    lag window, and invalid (``valid``, the original per-sample validity, ``(T, R)``)."""
    flat = np.array(field, dtype=np.float64).reshape(-1)
    n = np.arange(flat.size)
    flat[(n > r * anchor + r - 1) | (n < r * (anchor - n_lags + 1) - 1) | ~np.asarray(valid, dtype=bool).reshape(-1)] = np.nan
    return flat.reshape(np.asarray(field).shape)


def _reduce(work: _Work, result: Mapping[str, np.ndarray], *, readout: str, head: Any, identity: Sequence[Mapping[str, Any]],
            sample: np.ndarray, anchors: np.ndarray, columns: torch.Tensor, model_profile: np.ndarray,
            bands: Mapping[str, Tuple[int, int]], r: int, n_lags: int, window: Sequence[np.ndarray],
            seconds_raw: np.ndarray, seconds_tok: np.ndarray, up_valid: np.ndarray, keys: Optional[np.ndarray] = None,
            values: Optional[np.ndarray] = None) -> None:
    maps, patch = result["raw"], result["patch"]
    n_rows = maps.shape[0]
    heads = np.broadcast_to(np.asarray(head), (n_rows,))
    lag_raw = raw_lag_profile(maps, anchors, r, n_lags)
    lag_tok = core.lag_profile(core.time_profile(maps), anchors, n_lags)
    share_raw, share_tok, share_model = shares(lag_raw), shares(lag_tok), shares(model_profile)
    fit = core.agreement(lag_tok, model_profile)
    after, before = support_checks(maps, anchors, r, n_lags)
    ig_bands, model_bands = band_shares(share_tok, bands), band_shares(share_model, bands)
    key_tok = None if keys is None else core.lag_profile(keys, anchors, n_lags)
    value_tok = None if values is None else core.lag_profile(values, anchors, n_lags)
    for row in range(n_rows):
        tokens = window[row]
        position_raw = np.abs(maps[row][tokens]).sum(axis=0)
        position_value = np.abs(patch[row][tokens, :r]).sum(axis=0)
        position_delta = np.abs(patch[row][tokens, r:2 * r]).sum(axis=0)
        total_vd = position_value.sum() + position_delta.sum()
        move = float(result["value_input"][row] - result["value_entry"][row])
        attributed = float(maps[row].sum())
        record: Dict[str, Any] = {
            **identity[int(sample[row])], "anchor": int(anchors[row]), "column": int(columns[row].item()),
            "readout": readout, "head": int(heads[row]), "unit": UNITS[readout],
            "value_input": float(result["value_input"][row]), "value_null": float(result["value_null"][row]),
            "value_entry": float(result["value_entry"][row]),
            "entry_jump": float(result["value_entry"][row] - result["value_null"][row]),
            "attributed": attributed, "attributed_patch": float(patch[row].sum()),
            "completeness_rel": abs(attributed - move) / max(abs(move), abs(float(result["value_input"][row])), 1e-12),
            "completeness_strict": abs(attributed - move) / max(abs(move), 1e-12),
            "source_abs_total": float(np.abs(maps[row]).sum()),
            "after_anchor_max_abs": float(after[row]), "outside_window_max_abs": float(before[row]),
            "validity_channel_max_abs": float(np.abs(patch[row][:, -1]).max()),
            "invalid_sample_max_abs": float(np.abs(maps[row])[~up_valid[row]].max()) if (~up_valid[row]).any() else 0.0,
            "lag_corr": float(fit["lag_corr"][row]), "lag_js": float(fit["lag_js"][row]),
            "centroid_s": float(centroid_s(share_raw[row:row + 1], seconds_raw)[0]),
            "token_centroid_s": float(centroid_s(share_tok[row:row + 1], seconds_tok)[0]),
            "model_centroid_s": float(centroid_s(share_model[row:row + 1], seconds_tok)[0]),
            "argmax_s": float(seconds_raw[int(np.nanargmax(share_raw[row]))]) if np.isfinite(share_raw[row]).any() else np.nan,
            "model_argmax_s": float(seconds_tok[int(np.nanargmax(share_model[row]))]) if np.isfinite(share_model[row]).any() else np.nan,
            "value_share": float(position_value.sum() / total_vd) if total_vd > 0 else np.nan,
            "edge_share": float(position_delta[0] / position_delta.sum()) if position_delta.sum() > 0 else np.nan,
        }
        for name in bands:
            record[f"band_{name}"] = float(ig_bands[name][row])
            record[f"model_band_{name}"] = float(model_bands[name][row])
        if key_tok is not None:
            k, v = float(np.nansum(keys[row])), float(np.nansum(values[row]))
            record.update({"key_total": k, "value_total": v,
                           "key_share": abs(k) / max(abs(k) + abs(v), 1e-12),
                           "key_value_completeness": abs(k + v - move) / max(abs(move), 1e-12)})
        work.add(record, raw_lag_profile=lag_raw[row], token_lag_profile=lag_tok[row], model_profile=model_profile[row],
                 position_raw=position_raw, position_value=position_value, position_delta=position_delta,
                 key_lag_profile=key_tok[row] if key_tok is not None else np.full(n_lags, np.nan),
                 value_lag_profile=value_tok[row] if value_tok is not None else np.full(n_lags, np.nan))


# =============================================================================
# Figures
# =============================================================================
def _recording_shares(rows: pd.DataFrame, matrix: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Per-recording mean of row-normalised |profiles|: one recording is one unit."""
    _guids, classes, curves = class_contrast.per_recording(rows, shares(matrix))
    return classes, curves


def _band_shading(ax: Any, bands: Mapping[str, Tuple[int, int]], r: int) -> None:
    for index, (_name, (lo, hi)) in enumerate(bands.items()):
        ax.axvspan(lo * r / raw.events.FS_RAW, (hi + 1) * r / raw.events.FS_RAW,
                   color=core.BAND_COLOURS[index % len(core.BAND_COLOURS)], alpha=0.08, linewidth=0)


def _quiet_mean(values: Any) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def build_lag_figure(rows: pd.DataFrame, vectors: Mapping[str, np.ndarray], *, r: int, n_lags: int,
                     bands: Mapping[str, Tuple[int, int]], seed: int) -> Any:
    """Per readout: |IG| share per 4 s token (per raw sample, faint) against the model's ``source_kl_lag_map`` share."""
    seconds_raw = np.arange(r * n_lags + 1) / float(raw.events.FS_RAW)
    edges = np.arange(n_lags + 1) * r / float(raw.events.FS_RAW)
    centres = token_seconds(n_lags, r)
    columns = 3 if rows[labels.CLASS_COLUMN].astype(str).nunique() > 1 else 2
    figure, axes = plt.subplots(len(READOUTS), columns, figsize=(14.0, 2.6 * len(READOUTS)), squeeze=False)
    for index, readout in enumerate(READOUTS):
        keep = (rows["readout"] == readout).to_numpy()
        if not keep.any():
            for ax in axes[index]:
                core._empty(ax, SHORT[readout])
            continue
        part = rows[keep]
        _classes, raw_curves = _recording_shares(part, vectors["raw_lag_profile"][keep])
        classes, token_curves = _recording_shares(part, vectors["token_lag_profile"][keep])
        _classes, model_curves = _recording_shares(part, vectors["model_profile"][keep])
        raw_mean = class_contrast.mean_band(raw_curves, seed=seed)[0] * float(r)
        token_mean, lo, hi = class_contrast.mean_band(token_curves, seed=seed)
        model_mean = class_contrast.mean_band(model_curves, seed=seed)[0]
        ax = axes[index, 0]
        ax.plot(seconds_raw, raw_mean, color=COLOURS[readout], linewidth=figures.LINE_HAIRLINE, alpha=0.45,
                label=f"|IG| per raw sample (x{r})")
        if np.isfinite(lo).any():
            ax.stairs(hi, edges, baseline=lo, fill=True, color=COLOURS[readout], alpha=0.2, linewidth=0)
        ax.stairs(token_mean, edges, baseline=None, color=COLOURS[readout], linewidth=figures.LINE_EMPHASIS, label="|IG| per token")
        ax.stairs(model_mean, edges, baseline=None, color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="model $K_t\\alpha$")
        _band_shading(ax, bands, r)
        ax.set_title(f"{SHORT[readout]}: lag_corr {_quiet_mean(part['lag_corr']):.2f}, JS {_quiet_mean(part['lag_js']):.2f} "
                     f"({len(part)} anchors, {len(token_curves)} recordings)")
        ax.set_xlabel(LAG_LABEL)
        ax.set_ylabel("share per token")
        patch_figures.share_axis(ax, token_mean, model_mean, raw_mean, ncol=3)
        figures.style_axes(ax)
        groups = class_contrast.by_class(classes, token_curves)
        class_contrast._class_lines(axes[index, 1], centres, groups, seed=seed, title=f"{SHORT[readout]}: by class",
                                    xlabel=LAG_LABEL, ylabel="|IG| share per token", legend=True)
        patch_figures.share_axis(axes[index, 1], *[np.nanmean(c, axis=0) for c in groups.values() if len(c)], legend=False)
        if columns == 3:
            class_contrast._difference_lines(axes[index, 2], centres, groups, seed=seed,
                                             title=f"{SHORT[readout]}: class differences", xlabel=LAG_LABEL,
                                             ylabel="Δ share", legend=True)
    figures.caveat_note(figure, NOTE)
    return figure


def build_within_patch_figure(rows: pd.DataFrame, vectors: Mapping[str, np.ndarray], *, r: int) -> Any:
    """Which sample of a 4 s patch matters, and value against delta, per readout."""
    position_s = np.arange(r) / float(raw.events.FS_RAW)
    figure, axes = plt.subplots(len(READOUTS), 2, figsize=(14.0, 2.4 * len(READOUTS)), squeeze=False,
                                gridspec_kw={"width_ratios": [2.0, 1.0]})
    for index, readout in enumerate(READOUTS):
        keep = (rows["readout"] == readout).to_numpy()
        ax = axes[index, 0]
        if not keep.any():
            core._empty(ax, SHORT[readout])
            core._empty(axes[index, 1], "value share")
            continue
        value, delta = np.asarray(vectors["position_value"])[keep], np.asarray(vectors["position_delta"])[keep]
        patch_total = (value.sum(1) + delta.sum(1))[:, None]
        raw_matrix = np.asarray(vectors["position_raw"])[keep]
        series = []
        for matrix, total, colour, label in ((raw_matrix, raw_matrix.sum(1)[:, None], COLOURS[readout], "raw sample"),
                                             (value, patch_total, figures.COLOR_BLACK, "value channel"),
                                             (delta, patch_total, figures.COLOR_ORANGE, "delta channel")):
            share = np.nanmean(matrix / np.where(total > 0, total, np.nan), axis=0)
            series.append(share)
            ax.plot(position_s, share, marker="o", markersize=figures.MARKER_SMALL, color=colour,
                    linewidth=figures.LINE_REGULAR, label=label)
        ax.set_title(f"{SHORT[readout]}: |IG| share by position in the patch (delta at 0 s reads the previous patch)")
        ax.set_xlabel("sample time within the patch (s)")
        ax.set_ylabel("share")
        patch_figures.share_axis(ax, *series, ncol=3)
        figures.style_axes(ax)
        ax = axes[index, 1]
        part = rows[keep]
        order = labels.ordered_groups(sorted(part[labels.CLASS_COLUMN].astype(str).unique()), labels.CLASS_COLUMN)
        colours = figures.group_colors(order)
        for x, name in enumerate(order):
            values = part.loc[part[labels.CLASS_COLUMN].astype(str) == name, "value_share"].to_numpy(dtype=float)
            ax.bar(x, _quiet_mean(values), color=colours[name], edgecolor=figures.COLOR_BLACK,
                   linewidth=figures.HISTOGRAM_EDGE_WIDTH)
            ax.plot(np.full(values.size, x), values, ".", color=figures.COLOR_BLACK, markersize=figures.MARKER_SMALL)
        ax.axhline(0.5, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("share")
        ax.set_title("value share of |patch IG| (per anchor)")
        figures.style_axes(ax)
    figures.caveat_note(figure, NOTE)
    return figure


def build_mechanism_figure(rows: pd.DataFrame, vectors: Mapping[str, np.ndarray], pairs: pd.DataFrame, *, r: int,
                           n_lags: int, n_heads: int) -> Any:
    """Key vs value by lag, each head's IG against its own attention, and KL against gain per anchor."""
    edges = np.arange(n_lags + 1) * r / float(raw.events.FS_RAW)
    width = max(len(READOUTS), n_heads, 2)
    figure, axes = plt.subplots(3, width, figsize=(14.0, 7.8), squeeze=False)
    for index, readout in enumerate(READOUTS):
        ax = axes[0, index]
        keep = (rows["readout"] == readout).to_numpy()
        if not keep.any() or "key_share" not in rows:
            core._empty(ax, SHORT[readout])
            continue
        key, value = np.abs(np.asarray(vectors["key_lag_profile"])[keep]), np.abs(np.asarray(vectors["value_lag_profile"])[keep])
        total = np.nansum(key, axis=1, keepdims=True) + np.nansum(value, axis=1, keepdims=True)
        key_share = np.nanmean(key / np.where(total > 0, total, np.nan), axis=0)
        value_share = np.nanmean(value / np.where(total > 0, total, np.nan), axis=0)
        ax.stairs(key_share, edges, baseline=None, color=figures.COLOR_PURPLE, linewidth=figures.LINE_REGULAR, label="through $W_k$ (where)")
        ax.stairs(value_share, edges, baseline=None, color=figures.COLOR_GREEN, linewidth=figures.LINE_REGULAR, label="through $W_v$ (what)")
        ax.set_title(f"{SHORT[readout]}: key share {_quiet_mean(rows.loc[keep, 'key_share']):.2f}")
        ax.set_xlabel("lag behind the anchor (s)")
        ax.set_ylabel("|IG| share per token")
        patch_figures.share_axis(ax, key_share, value_share, ncol=1)
        figures.style_axes(ax)
    for ax in axes[0, len(READOUTS):]:
        ax.set_axis_off()
    head_keep = (rows["readout"] == READOUT_KLD_HEAD).to_numpy()
    for head in range(width):
        ax = axes[1, head]
        keep = head_keep & (rows["head"] == head).to_numpy()
        if head >= n_heads or not keep.any():
            ax.set_axis_off()
            continue
        ig = np.nanmean(shares(np.asarray(vectors["token_lag_profile"])[keep]), axis=0)
        attention = np.nanmean(shares(np.asarray(vectors["model_profile"])[keep]), axis=0)
        ax.stairs(ig, edges, baseline=None, color=figures.COLOR_BLUE, linewidth=figures.LINE_REGULAR, label="|IG| of $K_t^{(m)}$")
        ax.stairs(attention, edges, baseline=None, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, label="$\\alpha^{(m)}$")
        ax.set_title(f"head {head}: lag_corr {_quiet_mean(rows.loc[keep, 'lag_corr']):.2f}")
        ax.set_xlabel("lag behind the anchor (s)")
        ax.set_ylabel("share per token")
        patch_figures.share_axis(ax, ig, attention, ncol=2)
        figures.style_axes(ax)
    for column, (value_name, title) in enumerate((("profile_corr", "|IG| profile correlation, KL against gain"),
                                                  ("lag_shift_s", "lag shift of the gain profile behind the KL's (s)"))):
        ax = axes[2, column]
        names = sorted(pairs["pair"].unique()) if len(pairs) else []
        drawn = False
        for pair, colour in zip(names, (figures.COLOR_GREEN, figures.COLOR_VERMILLION)):
            values = pairs.loc[pairs["pair"] == pair, value_name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=20, color=colour, alpha=0.5, label=f"{pair} (median {np.median(values):.2f})",
                        edgecolor=figures.COLOR_BLACK, linewidth=figures.HISTOGRAM_EDGE_WIDTH)
                drawn = True
        if drawn:
            ax.set_title(title)
            ax.set_ylabel("anchors")
            ax.legend(fontsize=figures.FONT_TINY)
            figures.style_axes(ax)
        else:
            core._empty(ax, title)
    for ax in axes[2, 2:]:
        ax.set_axis_off()
    figures.caveat_note(figure, NOTE)
    return figure


def build_checks_figure(rows: pd.DataFrame) -> Any:
    """Completeness residual per row (log axis) against the shared tolerance, per readout."""
    figure, ax = plt.subplots(1, 1, figsize=(14.0, 2.4))
    names = [*READOUTS, READOUT_KLD_HEAD]
    for index, readout in enumerate(names):
        values = rows.loc[rows["readout"] == readout, "completeness_rel"].to_numpy(dtype=float)
        values = np.clip(values[np.isfinite(values)], 1e-12, None)
        ax.plot(values, np.full(values.size, index) + np.random.default_rng(index).uniform(-0.2, 0.2, values.size),
                ".", color=figures.COLOR_BLUE, markersize=figures.MARKER_SMALL)
        if values.size:
            ax.plot([np.median(values)] * 2, [index - 0.3, index + 0.3], color=figures.COLOR_BLACK, linewidth=figures.LINE_HEAVY)
    ax.axvline(1e-2, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_THIN)
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SHORT[name] for name in names])
    ax.set_xlabel("relative completeness residual |ΣIG − (f(x) − f(x0))| / max(|Δf|, |f(x)|)")
    ax.set_title("Integrated-gradient completeness (tolerance 1e-2)")
    figures.style_axes(ax)
    return figure


def build_example_page(item: Mapping[str, Any], norms: Mapping[str, Any], *, r: int) -> Any:
    n_lags = int(item["n_lags"])
    seconds_raw = np.arange(r * n_lags + 1) / float(raw.events.FS_RAW)
    edges = np.arange(n_lags + 1) * r / float(raw.events.FS_RAW)

    def lags(ax: Any, cax: Any) -> None:
        cax.set_axis_off()
        series = []
        for name in READOUTS:
            ax.plot(seconds_raw, item["lag_shares"][name] * float(r), color=COLOURS[name], linewidth=figures.LINE_HAIRLINE, alpha=0.4)
            ax.stairs(item["token_shares"][name], edges, baseline=None, color=COLOURS[name], linewidth=figures.LINE_EMPHASIS, label=SHORT[name])
            series += [item["token_shares"][name], item["lag_shares"][name] * float(r)]
        ax.stairs(item["model_share"], edges, baseline=None, color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR, label="model $K_t\\alpha$")
        series.append(item["model_share"])
        ax.set_xlabel(LAG_LABEL + f"; faint: per raw sample (x{r})")
        ax.set_ylabel("share per token")
        patch_figures.share_axis(ax, *series, ncol=4)
        figures.style_axes(ax)

    def heads(ax: Any, cax: Any) -> None:
        field = np.stack([row for head in range(item["head_shares"].shape[0])
                          for row in (item["head_shares"][head], item["alpha_shares"][head])])
        norm = core.unsigned_log_norm(field)
        image = ax.imshow(np.where(field > 0, field, np.nan), aspect="auto", origin="upper", interpolation="none",
                          cmap=plt.get_cmap("viridis").with_extremes(bad=core.UNREAD_COLOUR), norm=norm,
                          extent=(edges[0], edges[-1], field.shape[0] - 0.5, -0.5))
        ax.set_yticks(np.arange(field.shape[0]))
        ax.set_yticklabels([f"{kind} h{head}" for head in range(item["head_shares"].shape[0]) for kind in ("|IG|", "α")],
                           fontsize=figures.FONT_TINY)
        ax.set_xlabel("lag behind the anchor (s)")
        if norm is not None:
            core._attach_colorbar(ax.get_figure(), image, ax=ax, cax=cax, label="share (log)", norm=norm)
        figures.style_axes(ax, grid="none")

    map_rows = [(name, f"{SHORT[name]}: raw UP attribution", item["maps"][name], UNITS[name]) for name in READOUTS]
    return patch_figures.stacked_page(
        item, map_rows=map_rows, norms=norms, note=NOTE,
        lower=[("Lag profile against the model's lag readout", lags),
               ("Per head: |IG| of the head's divergence and the head's attention", heads)],
    )


# =============================================================================
# The analysis
# =============================================================================
def run_raw_attribution_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Raw-UP integrated gradients at the shared high-KL clean anchors; see the module docstring."""
    task, loader, collection = getattr(context, "task", None), getattr(context, "loader", None), getattr(context, "collection", None)
    per_sample = getattr(collection, "per_sample", None)
    if task is None or loader is None or per_sample is None or per_sample.empty:
        return skip_record(NAME, "an attribution is a gradient of a forward: this pass built no model or loader")
    model = task.orig_model
    if bool(getattr(model, "prior_availability_input", False)):
        return skip_record(NAME, "prior_availability_input re-runs the source adapter, so the layer split is not one call")
    directory = Path(str(output_dir)) / NAME
    (directory / "maps").mkdir(parents=True, exist_ok=True)
    cap, anchors_per_segment = raw.cap(eval_config, CAP_SEGMENTS, DEFAULT_SEGMENTS), raw.cap(eval_config, CAP_ANCHORS, DEFAULT_ANCHORS)
    n_steps, examples_per_class = raw.cap(eval_config, CAP_IG_STEPS, DEFAULT_IG_STEPS), raw.cap(eval_config, CAP_EXAMPLES, DEFAULT_EXAMPLES)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    r, n_lags, n_heads = int(model.raw_per_step), int(model.lag_attn.L), int(model.lag_attn.num_heads)
    bands = _lag_bands(eval_config, n_lags)
    scales = raw.raw_signal_scales(getattr(context, "config", None), loader)

    selected, accounting = raw.select_informative_segments(context, cap=cap, seed=seed, examples_per_class=examples_per_class)
    want = {str(name): True for name in selected[labels.CLASS_COLUMN].astype(str).unique()} if len(selected) else {}
    work = _Work()
    started = time.perf_counter()
    for rows, batch in raw.segment_batches(task, loader, selected, batch_size=SEGMENTS_PER_BATCH):
        _attribute_batch(task, batch, rows, work, anchors_per_segment=anchors_per_segment, n_steps=n_steps, bands=bands,
                         scales=scales, want_examples=want)
    elapsed = time.perf_counter() - started

    rows = pd.DataFrame(work.rows)
    vectors = {name: np.stack(values) for name, values in work.vectors.items()}
    pairs = pd.DataFrame(work.pairs)
    files: List[str] = []
    if rows.empty:
        return {**skip_record(NAME, "no scored anchor in the drawn segments"), "selection": accounting}
    rows.to_csv(directory / f"{NAME}_rows.csv", index=False)
    pairs.to_csv(directory / f"{NAME}_kl_gain.csv", index=False)
    identity = {f"row_{key}": rows[key].astype(str).to_numpy() for key in ("guid", "readout")}
    np.savez_compressed(directory / f"{NAME}_vectors.npz", raw_lag_seconds=np.arange(r * n_lags + 1) / float(raw.events.FS_RAW),
                        token_lag_seconds=token_seconds(n_lags, r), row_anchor=rows["anchor"].to_numpy(),
                        row_head=rows["head"].to_numpy(), **identity, **vectors)
    files += [f"{NAME}_rows.csv", f"{NAME}_kl_gain.csv", f"{NAME}_vectors.npz"]

    # One recording is one unit: per-recording means, then the shared class tests.
    main = rows[rows["readout"].isin(READOUTS)]
    metric_names = ["centroid_s", "lag_corr", "lag_js", "value_share", "edge_share", "key_share", *[f"band_{b}" for b in bands]]
    pieces, families = [], {}
    for readout in READOUTS:
        part = main[main["readout"] == readout]
        present = [m for m in metric_names if m in part]
        renamed = {m: f"{readout}_{m}" for m in present}
        families[f"raw:{readout}"] = list(renamed.values())
        pieces.append(part.groupby("guid").agg({**{m: "mean" for m in present}, labels.CLASS_COLUMN: "first"}).rename(columns=renamed))
    recordings = pd.concat(pieces, axis=1)
    recordings = recordings.loc[:, ~recordings.columns.duplicated()]
    recordings.to_csv(directory / f"{NAME}_recordings.csv")
    stats, pairwise = class_contrast.class_tests(recordings, families, seed=seed)
    stats.to_csv(directory / f"{NAME}_class_stats.csv", index=False)
    pairwise.to_csv(directory / f"{NAME}_class_pairwise.csv", index=False)
    files += [f"{NAME}_recordings.csv", f"{NAME}_class_stats.csv", f"{NAME}_class_pairwise.csv"]

    main_vectors = {k: v[main.index.to_numpy()] for k, v in vectors.items()}
    main = main.reset_index(drop=True)
    for stem, build in (
        (f"{NAME}_lag_profile", lambda: build_lag_figure(main, main_vectors, r=r, n_lags=n_lags, bands=bands, seed=seed)),
        (f"{NAME}_within_patch", lambda: build_within_patch_figure(main, main_vectors, r=r)),
        (f"{NAME}_mechanism", lambda: build_mechanism_figure(rows, vectors, pairs, r=r, n_lags=n_lags, n_heads=n_heads)),
        (f"{NAME}_checks", lambda: build_checks_figure(rows)),
    ):
        with warnings.catch_warnings():
            # Lags before the record are NaN in every row of a short-anchor recording: an empty column, not an error.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            files.append(str(Path(figures.render_figure(build(), directory / stem)).name))
    norms = {name: core.signed_log_norm(np.concatenate([np.ravel(e["maps"][name]) for e in work.examples]))
             for name in READOUTS} if work.examples else {}
    for item in work.examples:
        stem = f"{traces.class_dirname(item[labels.CLASS_COLUMN])}_{traces.recording_stem(item['guid'], item[labels.SUBGROUP_COLUMN])}_anchor{item['anchor']}_{NAME}"
        files.append("maps/" + str(Path(figures.render_figure(build_example_page(item, norms, r=r), directory / "maps" / stem, tight=False)).name))

    readouts_block = {}
    for readout in [*READOUTS, READOUT_KLD_HEAD]:
        part = rows[rows["readout"] == readout]
        block = {column: _quiet_mean(part[column]) for column in
                 ["lag_corr", "lag_js", "centroid_s", "token_centroid_s", "model_centroid_s", "argmax_s", "model_argmax_s",
                  "value_share", "edge_share", *[f"band_{b}" for b in bands], *[f"model_band_{b}" for b in bands],
                  "key_share"] if column in part and part[column].notna().any()}
        block["n_rows"] = int(len(part))
        if readout == READOUT_KLD_HEAD:
            for name, column in (("per_head_lag_corr", "lag_corr"), ("per_head_centroid_s", "token_centroid_s"),
                                 ("per_head_attention_centroid_s", "model_centroid_s")):
                block[name] = {int(h): _quiet_mean(g[column]) for h, g in part.groupby("head")}
        readouts_block[readout] = block
    checks = {
        "completeness_rel_median": float(np.nanmedian(rows["completeness_rel"])),
        "completeness_rel_max": float(np.nanmax(rows["completeness_rel"])),
        "completeness_strict_median": float(np.nanmedian(rows["completeness_strict"])),
        "n_rows_over_tolerance": int((rows["completeness_rel"] > 1e-2).sum()),
        "key_value_completeness_median": float(np.nanmedian(rows["key_value_completeness"])) if "key_value_completeness" in rows else None,
        "after_anchor_max_abs": float(rows["after_anchor_max_abs"].max()),
        "outside_window_max_abs": float(rows["outside_window_max_abs"].max()),
        "validity_channel_max_abs": float(rows["validity_channel_max_abs"].max()),
        "invalid_sample_max_abs": float(rows["invalid_sample_max_abs"].max()),
        "raw_vs_patch_sum_max_abs": float((rows["attributed"] - rows["attributed_patch"]).abs().max()),
        "meaning": ("completeness_rel as the shared attribution pass defines it (completeness_strict divides by "
                    "|f(x) - f(x0)| only); after_anchor / outside_window: largest |IG| on a raw sample the readout "
                    "cannot read, exactly 0; validity_channel: the patch IG on m_t - 1, exactly 0 (held fixed); invalid_sample: largest |IG| on a "
                    "UP sample the evaluation's own pass masks (non-finite, or a weight gap under fhr_weight), exactly 0"),
    }
    by_class = {str(k): int(v) for k, v in selected[labels.CLASS_COLUMN].value_counts().items()}
    kld = readouts_block.get(core.READOUT_KLD, {})
    logger.info(f"{NAME}: {len(selected)} segment(s), {len(rows)} row(s) in {elapsed:.1f} s; "
                f"kld lag_corr {kld.get('lag_corr', np.nan):.2f}, completeness max {checks['completeness_rel_max']:.1e}")
    return {
        "n_samples": int(len(selected)),
        "composition": {"n_recordings": int(selected["guid"].nunique()), "n_segments_by_class": by_class,
                        "n_rows": int(len(rows)), "n_anchors": int(len(main) // len(READOUTS))},
        "plan": {"capped": True, "cap": cap, "anchors_per_segment": anchors_per_segment, "ig_steps": n_steps,
                 "seed": seed, "baseline": "source_null (raw UP = 0), FHR held", "entry_fraction": core.BASELINE_ENTRY_FRACTION,
                 "readouts": list(READOUTS), "head_readout": READOUT_KLD_HEAD, "n_heads": n_heads, "n_lags": n_lags,
                 "raw_lag_axis": f"0 .. {r * n_lags} raw samples at 4 Hz behind the anchor token's last sample",
                 "lag_bands": {k: list(v) for k, v in bands.items()}, "n_unclean_anchors": int(work.n_unclean),
                 "delta_level_bpm_per_unit": float(raw.SummaryUnits.from_model(model, getattr(context, "config", None), loader).level_delta_bpm(1.0)),
                 "method": "LayerIntegratedGradients on source_adapter input (multiply_by_inputs=False) -> patch and raw IG; "
                           "LayerIntegratedGradients on [lag_attn.W_k, lag_attn.W_v] for the key/value split"},
        "selection": accounting,
        "cost": core.cost_record(elapsed_s=elapsed, n_segments=int(len(selected)), n_rows=int(len(rows)),
                                 n_forward_equivalents=int(work.forward_equivalents), device=getattr(task, "device", None)),
        "checks": checks,
        "readouts": readouts_block,
        "kl_gain": {pair: {"profile_corr_median": float(np.nanmedian(g["profile_corr"])),
                           "lag_shift_s_median": float(np.nanmedian(g["lag_shift_s"]))} for pair, g in pairs.groupby("pair")} if len(pairs) else {},
        "class_contrast": {"significant": stats.loc[stats["significant"].astype(bool), "metric"].tolist()
                           if "significant" in stats else [], "n_metrics_tested": int(len(stats))},
        "headline": {"kld_lag_corr": kld.get("lag_corr"), "kld_centroid_s": kld.get("centroid_s"),
                     "completeness_rel_max": checks["completeness_rel_max"]},
        "caveat": NOTE,
        "files": files,
    }
