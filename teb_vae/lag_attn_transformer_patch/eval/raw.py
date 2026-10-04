r"""The raw-signal substrate every new patch analysis shares (plan D6). Import it; do not re-implement it.

Raw signals here are the loader's own: ``fhr`` and ``up`` z-scored at 4 Hz, ``(B, L)`` with
``L = T * R`` (``R = model.raw_per_step = 16``), and ``weight`` the decimated FHR validity ``(B, T)``.
Token ``t`` reads samples ``[R t, R t + R)``. ``model`` is always the eval view
(:class:`~teb_vae.lag_attn_transformer_patch.eval.view.SeqVaeLagAttnTrfPatch`), whose forward is the
CFS five-argument form.

API (one line each; details in the docstrings):

* Inputs and forward: :func:`batch_signals` (``(fhr, up, weight)`` off a batch); :func:`patch_streams`
  (raw → ``(y_patch, u_patch)`` as the task builds them); :func:`forward_raw` (raw → patch → the dense
  forward, optional raw edits); :func:`latent_means` (context: decode both branches at the latent
  means, no ε); :func:`decode_at` (decode a latent at the decoded anchors or chosen columns);
  :func:`at_columns` (forward outputs narrowed to one column per row, rows optionally repeated).
* Targets: :func:`forecast_target` (the loss's ``(B, A, H, 2)`` gather); :func:`summaries_from_raw`
  (the model-free standardized target); :func:`block_nll` (per-anchor block NLL under the model's
  density); :func:`retained_block` (retained arrays + the rebuilt ``forecast_mask``);
  :func:`marginal_sigma` (AR(1) marginal σ along the horizon).
* Edits: :func:`replace_tokens` (overwrite chosen tokens' samples; **gaps stay gaps** by default);
  :func:`occlusion_fill` (the fill of the ``baseline``/``zero``/``missing`` arms); :func:`resting_tone`
  (the "no contraction" UP level); :func:`sample_validity` (per-sample validity, patchify's rule).
* Segments: :func:`draw_segments` (seeded capped draw of scored, locatable segments);
  :func:`select_informative_segments` (the shared class-balanced high-KL draw, unlabelled filled);
  :func:`informative_anchor_columns` (each segment's high-KL clean anchor columns);
  :func:`segment_batches` (rows → collated, identity-checked batches on the device);
  :func:`fill_unlabelled` / :func:`with_unlabelled_classes` (class-balanced draws on unlabelled data).
* Attribution: :class:`RawReadout` (Captum over raw ``(B, T, R)``; :meth:`RawReadout.prepare` gives finite
  raw plus the original per-sample validity, held fixed so a NaN stays masked along the path);
  :func:`raw_ig` (IG over raw FHR or UP from an explicit baseline tensor, with values and delta).
* Units and small utilities: :class:`SummaryUnits` (``from_model``/``from_config``/``from_context``);
  :func:`vae_config`, :func:`cap`, :func:`finite_or_none`, :func:`host`.
* Re-exports: ``informative_anchors``, ``select_segments``, ``informative_columns``,
  ``spread_columns``, ``AnchorReadout``, ``ATTENTION_CELL``, ``integrated_gradients``, ``expand_rows``,
  ``raw_signal_scales`` and the layer-0 ``events`` module (``detect_contractions``,
  ``detect_decelerations``).

Layering: torch and the shared eval layers only; no Lightning, no ``model/*``, no
``teb_vae.lag_attn_rws.eval``.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch import nn

from teb_vae.lag_attn_cfs.eval import cohort, events
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.attribution_pass import informative_anchors, labelled_segments, select_segments
from teb_vae.lag_attn_cfs.eval.attributions import (
    ATTENTION_CELL,
    BASELINE_ENTRY_FRACTION,
    IG_INTERNAL_BATCH_SIZE,
    REPARAMETERISATION_SEAMS,
    AnchorReadout,
    contributing_columns,
    expand_rows,
    informative_columns,
    integrated_gradients,
    spread_columns,
)
from teb_vae.lag_attn_cfs.eval.dataset_rows import check_batch_identity, dataset_index_map, resolve_rows, subset_loader
from teb_vae.lag_attn_cfs.eval.events import detect_contractions
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, forecast_likelihood_terms
from teb_vae.lag_attn_cfs.eval.traces import raw_signal_scales
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD, forecast_mask
from teb_vae.lag_attn_transformer_patch.nets.patching import patch_summaries, patchify

#: A raw edit: ``raw (B, L) -> raw (B, L)``, applied to a clone.
RawEdit = Callable[[torch.Tensor], torch.Tensor]

#: The class heading a recording with no clinical class is drawn under (the planted instrument).
UNLABELLED = "unlabelled"


def vae_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The run config's ``model_config.VAE_model`` block (``{}`` when absent)."""
    return dict(((config or {}).get("model_config") or {}).get("VAE_model") or {})


def cap(eval_config: Mapping[str, Any], name: str, default: int) -> int:
    """``eval_config.caps[name]`` as an int, ``default`` when unset."""
    return int((dict(eval_config.get("caps") or {})).get(name) or default)


def finite_or_none(value: Any) -> Optional[float]:
    """``value`` as a float, ``None`` when missing or not finite (JSON-safe headline values)."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def host(value: Any) -> np.ndarray:
    """A tensor (detached) or array-like as a float64 host array."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)



def batch_signals(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(fhr (B, L), up (B, L), weight (B, T))`` off a loader batch (attribute or mapping)."""
    get = batch.get if isinstance(batch, dict) else lambda name: getattr(batch, name)
    return get("fhr"), get("up"), get("weight")


def patch_streams(
    model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(y_patch, u_patch)`` ``(B, T, 2R + 1)``: FHR masked by ``weight``, UP by ``model.source_validity``."""
    r = int(model.raw_per_step)
    y_patch = patchify(fhr, weight, raw_per_step=r, validity="fhr_weight")
    u_patch = patchify(up, weight, raw_per_step=r, validity=model.source_validity)
    return y_patch, u_patch


def forward_raw(
    model: Any,
    fhr: torch.Tensor,
    up: torch.Tensor,
    weight: torch.Tensor,
    *,
    fhr_edit: Optional[RawEdit] = None,
    up_edit: Optional[RawEdit] = None,
    weight_edit: Optional[RawEdit] = None,
    anchor_phase: int = DENSE_ANCHOR_GEOMETRY[0],
    anchor_stride: int = DENSE_ANCHOR_GEOMETRY[1],
) -> Dict[str, torch.Tensor]:
    """Patchify (optionally edited) raw signals and run the forward, dense ``(0, 1)`` by default.

    Edits act on clones, so the caller's tensors are untouched. ``weight_edit`` changes the FHR
    validity (a gap injection); with ``source_validity: finite`` it does not reach UP. No
    ``no_grad`` here: Captum and gradient callers need the graph; wrap the call yourself otherwise.
    The latent noise is the model's own: for paired arms, compare mean-decoded outputs
    (``mu_base``/``mu_full``) or seed ``torch`` identically before each call.

    Returns:
        The forward dict (the CFS key set; ``anchor_index``/``anchor_valid`` on the dense axis).
    """
    fhr = fhr if fhr_edit is None else fhr_edit(fhr.clone())
    up = up if up_edit is None else up_edit(up.clone())
    weight = weight if weight_edit is None else weight_edit(weight.clone())
    y_patch, u_patch = patch_streams(model, fhr, up, weight)
    return model(y_patch, y_patch[..., :0], u_patch, anchor_phase=anchor_phase, anchor_stride=anchor_stride)


@contextmanager
def latent_means(model: Any) -> Iterator[None]:
    """Within the block the forward decodes both branches at their latent means (no ε drawn)."""
    def at_means(mu_prior, logvar_prior, mu_post, logvar_post):  # noqa: ANN001
        return mu_prior, mu_post

    names = [name for name in REPARAMETERISATION_SEAMS if hasattr(model, name)]
    for name in names:
        object.__setattr__(model, name, at_means)
    try:
        yield
    finally:
        for name in names:
            object.__delattr__(model, name)


def decode_at(
    model: Any, outputs: Mapping[str, torch.Tensor], latent: torch.Tensor, columns: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode a dense latent ``(B, T, d_z)`` at every decoded anchor, or at ``columns`` ``(B, K)`` of that axis.

    The decoder's persistence input (when the model has one) is narrowed to the same columns.
    Returns ``(mu, logvar)``, each ``(B, A or K, H, 2)``.
    """
    anchors = outputs["anchor_index"] if columns is None else outputs["anchor_index"].gather(1, columns)
    persistence = outputs.get("persistence")
    if persistence is not None and columns is not None:
        persistence = persistence.gather(1, columns[..., None].expand(-1, -1, persistence.shape[-1]))
    z = latent.gather(1, anchors.long()[..., None].expand(-1, -1, latent.shape[-1]))
    return model.decoder(z, persistence=persistence)


def at_columns(
    outputs: Mapping[str, torch.Tensor], columns: torch.Tensor, *, repeat: int = 1
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Forward outputs narrowed to one decoded column per row, for re-encoding arms at one anchor.

    Every batched tensor is first repeated ``repeat`` times per segment (``repeat_interleave``), so
    ``columns`` is ``(B * repeat, 1)``. Returns ``(outputs, anchors, anchor_valid)`` with the
    persistence input narrowed to the column, as ``controls.occluded_forward_outputs`` expects.
    """
    size, repeat = int(outputs["anchor_index"].shape[0]), int(repeat)

    def rows(value: Any) -> Any:
        batched = isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == size
        return value.repeat_interleave(repeat, dim=0) if batched and repeat != 1 else value

    scored = {name: rows(value) for name, value in outputs.items()}
    anchors = scored["anchor_index"].gather(1, columns)
    anchor_valid = scored["anchor_valid"].gather(1, columns)
    if scored.get("persistence") is not None:
        scored["persistence"] = scored["persistence"].gather(
            1, columns[:, :, None].expand(-1, -1, scored["persistence"].shape[-1])
        )
    return scored, anchors, anchor_valid


def forecast_target(
    model: Any, fhr: torch.Tensor, weight: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """The scored block ``(B, A, H, 2)`` at ``anchors`` (``(B, A)``): the training loss's own gather."""
    return model._build_forecast_target(model.summary_target(fhr, weight), anchors)


def summaries_from_raw(fhr: torch.Tensor, weight: torch.Tensor, units: "SummaryUnits", raw_per_step: int) -> torch.Tensor:
    """The standardized ``(N, T, 2)`` target from loader-z raw FHR without a model (``summary_target``'s arithmetic)."""
    patches = patchify(fhr, weight, raw_per_step=raw_per_step, validity="fhr_weight")
    s = patch_summaries(patches[..., :raw_per_step], patches[..., -1] + 1.0, eps=units.eps)
    return torch.stack([(s[..., c] - units.loc[c]) / units.scale[c] for c in range(2)], dim=-1)


def block_nll(
    model: Any, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, likelihood: str
) -> torch.Tensor:
    """Per-anchor block NLL ``(B, A)`` under the model's own density (cell mask, AR(1) φ)."""
    return masked_raw_block_per_anchor(
        mu, target, mask, likelihood=likelihood, logvar=logvar, **forecast_likelihood_terms(model)
    )[0]


def retained_block(context: Any, names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The retained waveform arrays plus the shared ``forecast_mask`` rebuilt at the run's geometry.

    ``None`` unless every name in ``names`` (and ``weight``, ``anchor_index``) was retained for at
    least one segment. Returns ``arrays`` (``names`` as float32 tensors), ``weight``, ``anchors``,
    ``mask`` ``(N, A, H)``, ``geometry``, ``r``, ``phi`` (per-channel AR(1) or ``None``), ``clamp``,
    ``fhr_raw`` (float64 or ``None``) and ``sample_index`` (per-sample-table rows).
    """
    collection = context.collection
    retained = dict(getattr(collection, "retained", None) or {})
    wanted = (*names, "weight", "anchor_index")
    if any(name not in retained for name in wanted) or len(retained["weight"]) == 0:
        return None
    record = dict(getattr(collection, "record", None) or {})
    geometry_record = dict(record.get("geometry") or {})
    vae = vae_config(getattr(context, "config", None))
    weight = torch.as_tensor(np.asarray(retained["weight"], dtype=np.float32))
    anchors = torch.as_tensor(np.asarray(retained["anchor_index"], dtype=np.int64))
    n_steps = int(weight.shape[1])
    fhr = retained.get("fhr_raw")
    r = int(vae.get("raw_per_step") or (np.asarray(fhr).shape[1] // n_steps if fhr is not None else 16))
    target = retained.get("target")
    horizon = int(geometry_record.get("horizon") or np.asarray(target).shape[2])
    warmup = int(geometry_record.get("anchor_floor") or vae.get("warmup_period") or 0)
    geometry = TrimmedRawGeometry(raw_len=n_steps * r, decimation=r, horizon=horizon, warmup=warmup)
    mask, _coverage = forecast_mask(
        weight, geometry, coverage_floor=float(vae.get("coverage_floor", 0.0) or 0.0), anchors=anchors
    )
    phi = dict(record.get("likelihood_structure") or {}).get("ar_coef_per_channel")
    clamp = dict(record.get("bounds") or {}).get("logvar_clamp") or vae.get("logvar_clamp") or (-5.0, 3.0)
    return {
        "arrays": {name: torch.as_tensor(np.asarray(retained[name], dtype=np.float32)) for name in names},
        "weight": weight, "anchors": anchors, "mask": mask, "geometry": geometry, "r": r,
        "phi": None if phi is None else [float(value) for value in phi],
        "clamp": (float(clamp[0]), float(clamp[1])),
        "fhr_raw": None if fhr is None else np.asarray(fhr, dtype=np.float64),
        "sample_index": np.asarray(retained.get("waveforms_sample_index", np.arange(len(weight))), dtype=np.int64),
    }


def marginal_sigma(logvar: np.ndarray, phi: Optional[float]) -> np.ndarray:
    r"""Predictive σ along the horizon of one channel: the AR(1) marginal ``v_τ = σ²_τ + φ² v_{τ-1}``."""
    variance = np.exp(np.asarray(logvar, dtype=np.float64))
    if phi is None:
        return np.sqrt(variance)
    out = np.empty_like(variance)
    carried = 0.0
    for step, value in enumerate(variance):
        carried = float(value) + float(phi) ** 2 * carried
        out[step] = carried
    return np.sqrt(out)


def replace_tokens(
    raw: torch.Tensor,
    tokens: torch.Tensor,
    value: Union[float, torch.Tensor],
    *,
    raw_per_step: int,
    keep_gaps: bool = True,
) -> torch.Tensor:
    """Return ``raw`` with every sample of the chosen tokens set to ``value``.

    **Gaps stay gaps by default**: a non-finite sample keeps its value (``keep_gaps=True``), so an
    edit never turns a recorded gap into valid signal. Pass ``keep_gaps=False`` only for an edit
    that means to fill gaps (and say so where you call it).

    Args:
        raw: ``(B, L)`` raw signal.
        tokens: ``(B, T)`` bool, the tokens to overwrite.
        value: A scalar, a ``(B,)`` per-segment level (e.g. :func:`resting_tone`), or a ``(B, L)``
            signal read at the overwritten samples (e.g. ``raw + bump``). ``nan`` makes the samples
            invalid, which ``patchify`` turns into the learned ``missing`` token.
        raw_per_step: ``R``.
        keep_gaps: Leave non-finite samples untouched.

    Returns:
        A new ``(B, L)`` tensor; re-``patchify`` it to get consistent value, delta and validity at
        the edit's edges.
    """
    mask = tokens.to(torch.bool).repeat_interleave(int(raw_per_step), dim=-1)
    fill = torch.as_tensor(value, dtype=raw.dtype, device=raw.device)
    if fill.dim() == 1:
        fill = fill[:, None]
    if keep_gaps:
        mask = mask & torch.isfinite(raw)
    return torch.where(mask, fill.expand_as(raw), raw)


def occlusion_fill(arm: str, up: torch.Tensor, tone: torch.Tensor) -> Union[float, torch.Tensor]:
    """The :func:`replace_tokens` value of an occlusion arm: ``baseline`` the resting ``tone`` ``(B,)``,
    ``zero`` the z-mean (0), ``missing`` NaN (validity −1). Gaps are kept by ``replace_tokens``."""
    if arm == "missing":
        return float("nan")
    return torch.zeros_like(up) if arm == "zero" else tone


def sample_validity(
    raw: torch.Tensor, weight: Optional[torch.Tensor], *, raw_per_step: int, validity: str = "finite"
) -> torch.Tensor:
    """Per-sample validity ``(B, L)`` bool, the rule ``patchify`` applies (``finite`` or ``fhr_weight``)."""
    valid = torch.isfinite(raw)
    if validity == "fhr_weight" and weight is not None:
        valid = valid & (weight >= VALID_THRESHOLD).repeat_interleave(int(raw_per_step), dim=-1)
    return valid


def resting_tone(
    up: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    *,
    raw_per_step: int = 16,
    validity: str = "finite",
    quantile: float = 0.1,
) -> torch.Tensor:
    """The segment's resting UP tone ``(B,)``: the ``quantile`` of its valid raw samples (loader units).

    The "no contraction" level for the ``baseline`` occlusion arm, the tone IG baseline and the
    injection background. Whole-segment (20 min) by design: the segment is the unit every arm is
    paired within. A segment with no valid sample returns 0.0 (the z-mean), which is the CFS ``zero``.
    """
    valid = sample_validity(up, weight, raw_per_step=raw_per_step, validity=validity)
    masked = torch.where(valid, up, torch.full_like(up, float("nan")))
    tone = torch.nanquantile(masked.float(), float(quantile), dim=-1).to(up.dtype)
    return torch.nan_to_num(tone, nan=0.0)


def draw_segments(context: Any, *, cap: int, seed: int, max_hours: Optional[float] = None) -> pd.DataFrame:
    """A seeded draw (``np.random.default_rng(seed)``) of at most ``cap`` scored, locatable segments.

    Scored = ``n_anchors > 0`` on the per-sample table; ``max_hours`` applies the shared
    time-before-delivery bound. Rows carry ``dataset_index`` and are in ascending dataset order.
    """
    per_sample = context.collection.per_sample
    scored = per_sample[per_sample["n_anchors"] > 0] if "n_anchors" in per_sample.columns else per_sample
    if max_hours is not None:
        scored = cohort.within_horizon(scored, max_hours)
    rows = resolve_rows(scored, dataset_index_map(context.loader))
    if len(rows) > cap:
        chosen = np.random.default_rng(seed).choice(len(rows), size=cap, replace=False)
        rows = rows.iloc[np.sort(chosen)].reset_index(drop=True)
    return rows


def fill_unlabelled(per_sample: pd.DataFrame) -> pd.DataFrame:
    """A copy whose missing clinical class reads :data:`UNLABELLED`, so a class-balanced draw keeps it."""
    frame = per_sample.copy()
    if labels.CLASS_COLUMN in frame.columns:
        frame[labels.CLASS_COLUMN] = frame[labels.CLASS_COLUMN].fillna(UNLABELLED)
    return frame


def with_unlabelled_classes(run: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap an analysis so it sees :func:`fill_unlabelled` per-sample classes (the shared selection
    never draws a recording with no class). The shared function itself is not edited."""
    def wrapped(context: Any, *, eval_config: Dict[str, Any], output_dir: Any, probe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        collection = getattr(context, "collection", None)
        per_sample = getattr(collection, "per_sample", None)
        if per_sample is not None:
            collection = dataclasses.replace(collection, per_sample=fill_unlabelled(per_sample))
            context = dataclasses.replace(context, collection=collection)
        return run(context, eval_config=eval_config, output_dir=output_dir, probe=probe)

    wrapped.__name__ = getattr(run, "__name__", "wrapped")
    wrapped.__doc__ = run.__doc__
    return wrapped


def select_informative_segments(
    context: Any, *, cap: int, seed: int, examples_per_class: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The shared high-KL clean, class-balanced draw (one segment per recording), unlabelled filled.

    Returns ``(selected rows with dataset_index, accounting)``; ``accounting["anchor_rule"]`` is the
    informative-anchor rule.
    """
    collection = context.collection
    candidates, rule = informative_anchors(getattr(collection, "per_anchor", None))
    segments = labelled_segments(collection.per_sample)
    segments[labels.CLASS_COLUMN] = segments[labels.CLASS_COLUMN].fillna(UNLABELLED)
    selected, accounting = select_segments(
        segments, dataset_index_map(context.loader), cap=cap, seed=seed,
        candidates=candidates, examples_per_class=examples_per_class,
    )
    accounting["anchor_rule"] = rule
    return selected, accounting


def informative_anchor_columns(
    model: Any, outputs: Mapping[str, torch.Tensor], weight: torch.Tensor, rows: pd.DataFrame, per_segment: int
) -> Tuple[List[np.ndarray], int]:
    """Each segment's anchor columns: its high-KL clean candidates (``rows["anchors"]``), else spread anchors.

    Returns ``(columns per segment, number of candidates that failed the history check)``.
    """
    contributing = contributing_columns(model, weight, outputs)
    if "anchors" not in rows.columns:
        return spread_columns(contributing, per_segment), 0
    axis, validity = outputs["anchor_index"].cpu().numpy(), weight.cpu().numpy()
    chosen, unclean = [], 0
    for element, wanted in enumerate(rows["anchors"]):
        columns, failed = informative_columns(axis[element], contributing[element], wanted, validity[element],
                                              n_lags=int(model.lag_attn.L), per_segment=per_segment,
                                              spacing=int(model.horizon))
        chosen.append(columns)
        unclean += failed
    return chosen, unclean


def segment_batches(
    task: Any, loader: Any, rows: pd.DataFrame, *, batch_size: Optional[int] = None
) -> Iterator[Tuple[pd.DataFrame, Any]]:
    """Yield ``(rows chunk, batch on the task's device)`` over ``rows`` (ascending ``dataset_index``).

    Batches are collated by the loader's own ``collate_fn`` (``batch_size`` defaults to the loader's)
    and identity-checked against the chunk before they are moved.
    """
    size = max(1, int(batch_size or getattr(loader, "batch_size", None) or 1))
    for start, batch in zip(range(0, len(rows), size),
                            subset_loader(loader, rows["dataset_index"].tolist(), batch_size=size)):
        chunk = rows.iloc[start:start + size]
        check_batch_identity(batch, chunk)
        yield chunk, task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)


@dataclass(frozen=True)
class SummaryUnits:
    r"""Standardized patch summaries ↔ clinical units.

    ``level`` (channel 0): the patch mean of loader-z FHR, standardized as ``(m - loc0) / scale0``;
    ``bpm = (ŷ·scale0 + loc0)·σ + μ``. ``variability`` (channel 1): ``log(rms Δ_z + eps)``
    standardized as ``(v - loc1) / scale1``; ``rms Δ (bpm per 0.25 s) = (exp(ŷ·scale1 + loc1) - eps)·σ``.
    A variability *difference* is a log-ratio: report ``exp(|Δŷ|·scale1)`` as a factor, never as bpm.
    Nats stay in standardized units (a bpm density adds ``log(scale0·σ)`` per level cell).

    Attributes:
        fhr_mean, fhr_std: Loader FHR statistics (``traces.raw_signal_scales``), bpm.
        up_mean, up_std: The same for UP (loader units → the monitor's own).
        loc, scale: The model's ``target_summary_loc/scale`` (2 entries each).
        eps: The model's ``variability_eps`` (loader-z units).
    """

    fhr_mean: float
    fhr_std: float
    up_mean: float
    up_std: float
    loc: Tuple[float, float]
    scale: Tuple[float, float]
    eps: float

    @classmethod
    def from_model(
        cls, model: Any, config: Optional[Mapping[str, Any]] = None, loader: Any = None
    ) -> "SummaryUnits":
        """Constants from the rebuilt model (the authority) and the run's stats (loader or config)."""
        return cls._build(
            raw_signal_scales(config, loader),
            model.target_summary_loc, model.target_summary_scale, model.variability_eps,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SummaryUnits":
        """Offline (no model): the summary constants from the dumped config's ``VAE_model`` block."""
        vae = dict((config.get("model_config") or {}).get("VAE_model") or {})
        return cls._build(
            raw_signal_scales(config),
            vae.get("target_summary_loc", (0.0, 0.0)), vae.get("target_summary_scale", (1.0, 1.0)),
            vae.get("variability_eps", 0.01),
        )

    @classmethod
    def from_context(cls, context: Any) -> "SummaryUnits":
        """The model's constants when the pass built one, the dumped config's otherwise."""
        task = getattr(context, "task", None)
        if task is not None:
            return cls.from_model(task.orig_model, getattr(context, "config", None), getattr(context, "loader", None))
        return cls.from_config(getattr(context, "config", None) or {})

    @classmethod
    def _build(cls, scales: Mapping[str, Tuple[float, float]], loc: Any, scale: Any, eps: Any) -> "SummaryUnits":
        fhr = scales.get("fhr", (0.0, 1.0))
        up = scales.get("up", (0.0, 1.0))
        return cls(
            fhr_mean=float(fhr[0]), fhr_std=float(fhr[1]), up_mean=float(up[0]), up_std=float(up[1]),
            loc=(float(loc[0]), float(loc[1])), scale=(float(scale[0]), float(scale[1])), eps=float(eps),
        )

    @property
    def is_identity(self) -> bool:
        """True when the summary constants are the shipped placeholders (climatology is then not a mean)."""
        return self.loc == (0.0, 0.0) and self.scale == (1.0, 1.0)

    def level_bpm(self, y: Any) -> Any:
        """Standardized level → bpm."""
        return (y * self.scale[0] + self.loc[0]) * self.fhr_std + self.fhr_mean

    def level_delta_bpm(self, dy: Any) -> Any:
        """A level difference, error or σ (standardized) → bpm."""
        return dy * (self.scale[0] * self.fhr_std)

    def variability_bpm(self, y: Any) -> Any:
        """Standardized variability → rms of the within-patch first difference, bpm per 0.25 s."""
        exp = torch.exp if isinstance(y, torch.Tensor) else np.exp
        return (exp(y * self.scale[1] + self.loc[1]) - self.eps) * self.fhr_std

    def variability_factor(self, dy: Any) -> Any:
        """A variability difference (standardized) → the multiplicative factor on rms Δ."""
        exp = torch.exp if isinstance(dy, torch.Tensor) else np.exp
        return exp(abs(dy) * self.scale[1])

    def up_physical(self, up: Any) -> Any:
        """Loader-unit UP → the monitor's unit (mmHg or relative units)."""
        return up * self.up_std + self.up_mean



class RawReadout(nn.Module):
    r"""Captum readout over raw ``(B, T, R)`` streams: ``patchify`` inside, then the shared readout.

    Wraps :class:`~teb_vae.lag_attn_cfs.eval.attributions.AnchorReadout` (built on the eval view,
    whose forward is the CFS form, so ``attributed_forward``'s shadows and Grad-CAM hooks apply to the
    real instance). Inputs are raw FHR and UP reshaped ``(B, T, R)``, plus an empty ``(B, T, 0)``
    placeholder in the ``y_ph`` slot so the call signature matches ``AnchorReadout.forward``. Raw lag of
    sample ``k`` of token ``t`` relative to anchor ``t_a``: ``16 t_a + 15 - (16 t + k)``.

    **Validity is the original raw's, held fixed.** An IG path needs finite inputs (``alpha * nan`` is
    nan), so :meth:`prepare` zeroes the non-finite samples *and* returns each stream's per-sample
    validity of the **original** raw (:func:`sample_validity`, the rule ``patchify`` applies). Passed
    back as the two trailing extras ``fhr_valid`` / ``up_valid``, :meth:`forward` restores NaN at every
    invalid sample before ``patchify``, so the forward masks exactly what the evaluation's own pass
    masks, at every point of the path, and an invalid sample's attribution is exactly zero. Without
    them, validity is re-derived from the raw given -- correct only for raw that never held a NaN.

    Usage::

        fhr3, up3, fhr_valid, up_valid = RawReadout.prepare(model, fhr, up, weight)
        wrapper = RawReadout(AnchorReadout(model, ATTENTION_CELL, readout="kld").eval())
        ig = raw_ig(wrapper, (fhr3, fhr3[..., :0], up3), (summaries, weight), columns, coordinates,
                    stream="up", baseline=torch.zeros_like(up3), n_steps=128, valid=(fhr_valid, up_valid))
    """

    def __init__(self, inner: AnchorReadout) -> None:
        super().__init__()
        self.inner = inner
        self.model, self.cell, self.readout = inner.model, inner.cell, inner.readout

    @staticmethod
    def prepare(
        model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(fhr3, up3, fhr_valid, up_valid)``, each ``(B, T, R)``: finite raw for the IG path, and each
        stream's original per-sample validity (FHR by ``weight``, UP by ``model.source_validity``)."""
        batch, steps, r = int(weight.shape[0]), int(weight.shape[1]), int(model.raw_per_step)
        shape = (batch, steps, r)
        fhr_valid = sample_validity(fhr, weight, raw_per_step=r, validity="fhr_weight").view(shape)
        up_valid = sample_validity(up, weight, raw_per_step=r, validity=model.source_validity).view(shape)
        finite = lambda x: torch.where(torch.isfinite(x), x, x.new_zeros(())).view(shape)  # noqa: E731
        return finite(fhr), finite(up), fhr_valid, up_valid

    def anchor_steps(self, columns: torch.Tensor) -> torch.Tensor:
        """Each row's anchor as a stored step (the inner readout's)."""
        return self.inner.anchor_steps(columns)

    def forward(
        self,
        fhr: torch.Tensor,
        empty: torch.Tensor,
        up: torch.Tensor,
        summaries: torch.Tensor,
        weight: torch.Tensor,
        columns: torch.Tensor,
        coordinates: torch.Tensor,
        fhr_valid: Optional[torch.Tensor] = None,
        up_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw ``(B, T, R)`` → (original invalid samples back to NaN) → patch streams → the inner readout ``(B,)``."""
        nan = fhr.new_full((), float("nan"))
        if fhr_valid is not None:
            fhr = torch.where(fhr_valid, fhr, nan)
        if up_valid is not None:
            up = torch.where(up_valid, up, nan)
        y_patch, u_patch = patch_streams(self.model, fhr.flatten(1), up.flatten(1), weight)
        out = self.inner(y_patch, y_patch[..., :0], u_patch, summaries, weight, columns, coordinates)
        return out + 0.0 * empty.sum()


def raw_ig(
    wrapper: RawReadout,
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    columns: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    stream: str,
    baseline: torch.Tensor,
    n_steps: int,
    valid: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Integrated gradients of a :class:`RawReadout` over one raw stream from an explicit baseline.

    Args:
        wrapper: The readout.
        inputs: Row-expanded ``(fhr (N, T, R), empty (N, T, 0), up (N, T, R))``, finite
            (:meth:`RawReadout.prepare`); the other stream is held.
        extra: ``(summaries, weight)``, row-expanded.
        columns: Per-row anchor-axis positions; ``coordinates``: per-row coordinate (head, dimension).
        stream: ``"fhr"`` or ``"up"``, the attributed input.
        baseline: ``(N, T, R)`` baseline of that stream (e.g. a flat tone, a flat median, zeros).
        n_steps: Integration steps. The path enters at ``baseline + BASELINE_ENTRY_FRACTION·(x − baseline)``.
        valid: Row-expanded ``(fhr_valid, up_valid)`` from :meth:`RawReadout.prepare`, held fixed along
            the path; required (``None`` only for raw that never held a non-finite sample).

    Returns:
        Detached tensors: ``map`` ``(N, T, R)`` (exactly 0 on invalid samples), ``delta`` (Captum's
        convergence delta), ``value_input``, ``value_baseline``, ``value_entry`` (each ``(N,)``).
    """
    fhr3, empty, up3 = (x.detach() for x in inputs)
    if stream == "fhr":
        x, held = fhr3, (empty, up3)
        forward = lambda fhr, *rest: wrapper(fhr, *rest)  # noqa: E731 - (fhr, empty, up, *extra, columns, coordinates)
    else:
        x, held = up3, (fhr3, empty)

        def forward(u, f, e, *rest):  # noqa: ANN001 - Captum passes the attributed input first
            return wrapper(f, e, u, *rest)
    start = baseline + BASELINE_ENTRY_FRACTION * (x - baseline)
    # The held stream and the validity travel as additional arguments, so Captum expands them with the path.
    args = (*held, *extra, columns, coordinates, *(valid or ()))
    with torch.no_grad():
        value_input, value_baseline, value_entry = (forward(value, *args) for value in (x, baseline, start))
    attribution, delta = IntegratedGradients(forward).attribute(
        x, baselines=start.detach(), additional_forward_args=args, n_steps=int(n_steps),
        internal_batch_size=max(IG_INTERNAL_BATCH_SIZE, int(columns.shape[0])), return_convergence_delta=True,
    )
    return {"map": attribution.detach(), "delta": delta.detach(), "value_input": value_input,
            "value_baseline": value_baseline, "value_entry": value_entry}


__all__ = [
    "ATTENTION_CELL", "AnchorReadout", "RawEdit", "RawReadout", "SummaryUnits", "UNLABELLED",
    "at_columns", "batch_signals", "block_nll", "cap", "decode_at", "detect_contractions", "draw_segments",
    "events", "expand_rows", "fill_unlabelled", "finite_or_none", "forecast_target", "forward_raw", "host",
    "informative_anchor_columns", "informative_anchors", "informative_columns", "integrated_gradients",
    "latent_means", "marginal_sigma", "occlusion_fill", "patch_streams", "raw_ig", "raw_signal_scales",
    "replace_tokens", "resting_tone", "retained_block", "sample_validity", "segment_batches",
    "select_informative_segments", "select_segments", "spread_columns", "summaries_from_raw", "vae_config",
    "with_unlabelled_classes",
]
