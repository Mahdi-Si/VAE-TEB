r"""``fhr_drivers`` -- what in the raw FHR history drives the prior, and the FHR context of the UP effect.

Owner E1-C (``notes/EVAL_PLAN.md`` §2; proposals P3 and P8 of ``notes/EVAL_MAP_CAPTUM.md``).

**Question.** At the shared high-KL clean anchors, how far back, and where, does the raw FHR history
drive the target-only branch: the base forecast's level and variability
(``base_level`` / ``base_var`` = the mean over the horizon of ``mu_base`` per channel, standardized
units) and the prior mean (``mu_prior``, the coordinate with the largest ``|mu^p|`` at the anchor)?
And which FHR history makes the UP matter: ``src_effect`` = ``K_t(fhr, up) - K_t(fhr, 0)``, the
divergence the UP adds over the source-null arm, attributed to raw FHR with UP held.

**Method.** Integrated gradients over raw FHR ``(B, T, R)`` through :class:`~..raw.RawReadout` (UP held
at its own values; each stream's validity is the original raw's, held fixed along the path), from two
baselines entered at ``core.BASELINE_ENTRY_FRACTION``:

* ``pop_mean`` -- raw FHR 0, the loader's population-mean FHR: the whole history is attributed,
  level included;
* ``seg_median`` -- flat at the segment's own median valid FHR up to the anchor (causal): the level
  is removed, so what is left is the pattern (decelerations, variability, trends).

``src_effect`` is read under ``pop_mean`` only. Every row carries its completeness residual and two
structural checks that are exactly zero on this model: attribution after the anchor token's last
sample, and attribution to a sample the evaluation's own pass masks (a gap or a non-finite sample).

**Reading.** ``lag50_s`` / ``lag90_s`` are the lags holding half and 90 % of the row's ``|IG|``.
A prior whose 90 % sits inside the encoder's conv-stem reach (``share_in_stem``) is near-persistence;
a long tail means it tracks baseline or variability trends. The ``seg_median`` rows against the
``pop_mean`` rows separate FHR level from FHR pattern. For ``src_effect``, mass on the most recent
FHR says the UP explains an ongoing FHR change; mass far back says the UP is weighted by FHR state.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn

from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import class_contrast, lag_hist, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.metrics import batch_field
from teb_vae.lag_attn_transformer_patch.eval import figures as patch_figures
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "fhr_drivers"

#: ``(headline_name, key)`` pairs registered as ``("fhr_drivers", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("fhr_drivers_base_level_lag90_s", "base_level_lag90_s"),
    ("fhr_drivers_src_effect_lag50_s", "src_effect_lag50_s"),
    ("fhr_drivers_completeness_max", "completeness_rel_max"),
)

CAP_SEGMENTS, DEFAULT_SEGMENTS = f"{NAME}_segments", 24
CAP_ANCHORS, DEFAULT_ANCHORS = f"{NAME}_anchors", 4
CAP_IG_STEPS, DEFAULT_IG_STEPS = f"{NAME}_ig_steps", 128
CAP_EXAMPLES, DEFAULT_EXAMPLES = f"{NAME}_examples_per_class", 1
SEGMENTS_PER_BATCH = 4
SEED_OFFSET = 31

BASE_LEVEL, BASE_VAR, MU_PRIOR, SRC_EFFECT = "base_level", "base_var", "mu_prior", "src_effect"
POP_MEAN, SEG_MEDIAN = "pop_mean", "seg_median"
#: ``(readout, baseline)`` in the order every table and figure lists them.
VARIANTS: Tuple[Tuple[str, str], ...] = (
    (BASE_LEVEL, POP_MEAN), (BASE_LEVEL, SEG_MEDIAN), (BASE_VAR, POP_MEAN), (BASE_VAR, SEG_MEDIAN),
    (MU_PRIOR, POP_MEAN), (MU_PRIOR, SEG_MEDIAN), (SRC_EFFECT, POP_MEAN),
)
READOUTS: Tuple[str, ...] = (BASE_LEVEL, BASE_VAR, MU_PRIOR, SRC_EFFECT)
UNITS: Mapping[str, str] = {BASE_LEVEL: "standardized level", BASE_VAR: "standardized variability",
                            MU_PRIOR: "latent units", SRC_EFFECT: "nats"}
SHORT: Mapping[str, str] = {BASE_LEVEL: "Base level", BASE_VAR: "Base variability", MU_PRIOR: "Prior mean $\\mu^p_d$",
                            SRC_EFFECT: "UP effect $K_t - K_t^{null}$"}
BASELINE_COLOURS: Mapping[str, str] = {POP_MEAN: figures.COLOR_BLUE, SEG_MEDIAN: figures.COLOR_ORANGE}
NOTE = "Model sensitivity, not a causal effect. Lag: raw FHR history behind the anchor token's end (4 s tokens)."
LAG_LABEL = "lag behind the anchor (s, log)"


# =============================================================================
# Readouts
# =============================================================================
class _Readout(core.AnchorReadout):
    """``base_level`` / ``base_var`` (mean over the horizon of ``mu_base``), ``mu_prior`` (the row's
    coordinate) and ``src_effect`` (divergence minus the source-null divergence)."""

    def __init__(self, model: Any, readout: str, likelihood: str) -> None:
        super().__init__(model, core.ATTENTION_CELL, readout=core.READOUT_MU_PRIOR_DIM if readout == MU_PRIOR else core.READOUT_KLD,
                         likelihood=likelihood)
        self.target_readout = readout

    def forward(self, y_st, y_ph, u_stream, target_features, weight, columns, coordinates):  # noqa: D102
        name = self.target_readout
        if name == MU_PRIOR:
            return super().forward(y_st, y_ph, u_stream, target_features, weight, columns, coordinates)
        if name == SRC_EFFECT:
            null = super().forward(y_st, y_ph, torch.zeros_like(u_stream), target_features, weight, columns, coordinates)
            return super().forward(y_st, y_ph, u_stream, target_features, weight, columns, coordinates) - null
        anchors = self.anchor_steps(columns)
        with core.attributed_forward(self.model, anchors):
            out = self.model(y_st, y_ph, u_stream)
        tie = 0.0 * (y_st.sum() + y_ph.sum() + u_stream.sum() + out["mu_post"].sum() + out["logvar_post"].sum())
        channel = 0 if name == BASE_LEVEL else 1
        return out["mu_base"][:, 0, :, channel].mean(dim=-1) + tie


# =============================================================================
# Shared plumbing (duplicated in raw_attribution; hoist candidates for raw.py)
# =============================================================================


def stem_reach_tokens(model: Any) -> int:
    """Tokens the target encoder's causal conv stem reads: ``1 + sum (k - 1) d`` over its convolutions."""
    reach = 1
    for module in getattr(model.target_encoder, "conv_blocks", nn.ModuleList()).modules():
        if isinstance(module, nn.Conv1d):
            reach += (int(module.kernel_size[0]) - 1) * int(module.dilation[0])
    return reach


# =============================================================================
# The attribution
# =============================================================================
def history_statistics(share: np.ndarray, seconds: np.ndarray, reach: int) -> Dict[str, np.ndarray]:
    """Per row: lags (s) holding 50 % and 90 % of ``|IG|``, and the share inside the conv-stem reach."""
    filled = np.nan_to_num(share, nan=0.0)
    cumulative = np.cumsum(filled, axis=1)
    has = np.isfinite(share).any(axis=1) & (filled.sum(axis=1) > 0)
    out = {}
    for name, level in (("lag50_s", 0.5), ("lag90_s", 0.9)):
        index = np.argmax(cumulative >= level - 1e-9, axis=1)
        out[name] = np.where(has, seconds[index], np.nan)
    out["share_in_stem"] = np.where(has, cumulative[:, min(reach, share.shape[1]) - 1], np.nan)
    return out


# =============================================================================
# The pass
# =============================================================================
def _attribute_batch(task: Any, batch: Any, rows: pd.DataFrame, *, anchors_per_segment: int, n_steps: int, reach: int,
                     scales: Mapping[str, Tuple[float, float]], want_examples: Dict[str, bool], out: Dict[str, Any]) -> None:
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    r = int(model.raw_per_step)
    fhr, up, weight = raw.batch_signals(batch)
    steps = int(weight.shape[1])
    # The dense forward and the target off the ORIGINAL raw, exactly as the evaluation's own pass reads it;
    # only the IG path gets finite raw, with the original validity held fixed beside it.
    summaries = model.summary_target(fhr, weight)
    with torch.no_grad():
        outputs = raw.forward_raw(model, fhr, up, weight)
    columns_per_sample, unclean = raw.informative_anchor_columns(model, outputs, weight, rows, anchors_per_segment)
    out["n_unclean"] += unclean
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
        top = outputs["mu_prior"][sample_t, anchors_t].abs().argmax(dim=-1)                       # (N,)
    # The segment-median baseline: flat at the median of the valid FHR samples up to the anchor's last one.
    flat_rows = inputs[0].flatten(1)
    valid_rows = valid[0].flatten(1)
    upto = torch.arange(flat_rows.shape[1], device=flat_rows.device)[None, :] <= (anchors_t[:, None] * r + r - 1)
    masked = torch.where(valid_rows & upto, flat_rows, torch.full_like(flat_rows, float("nan")))
    median = torch.nan_to_num(torch.nanmedian(masked, dim=1).values, nan=0.0)
    baselines = {POP_MEAN: torch.zeros_like(inputs[0]), SEG_MEDIAN: median[:, None, None].expand_as(inputs[0]).contiguous()}
    identity = [{"guid": str(row["guid"]), "epoch": float(row["epoch"]), labels.CLASS_COLUMN: row.get(labels.CLASS_COLUMN),
                 labels.SUBGROUP_COLUMN: row.get(labels.SUBGROUP_COLUMN)} for _, row in rows.iterrows()]
    seconds = (np.arange(steps) * r + 0.5 * (r - 1)) / float(raw.events.FS_RAW)
    invalid = ~(raw.host(valid[0]) > 0)                                  # (N, T, R): masked by the evaluation's own pass
    flat_index = np.arange(steps * r)[None, :]
    maps: Dict[Tuple[str, str], np.ndarray] = {}
    for readout, baseline in VARIANTS:
        wrapper = raw.RawReadout(_Readout(model, readout, likelihood).eval())
        coordinates = top if readout == MU_PRIOR else torch.zeros_like(columns)
        result = {key: raw.host(value) for key, value in raw.raw_ig(
            wrapper, inputs, extra, columns, coordinates, stream="fhr", baseline=baselines[baseline], n_steps=n_steps,
            valid=valid,
        ).items()}
        out["forward_equivalents"] += n_rows * (n_steps + 3) * (2 if readout == SRC_EFFECT else 1)
        maps[(readout, baseline)] = result["map"]
        lag = core.lag_profile(core.time_profile(result["map"]), anchors, steps)                  # (N, T)
        share = lag_hist.normalise(np.abs(lag))
        statistics = history_statistics(share, seconds, reach)
        flat = np.abs(result["map"]).reshape(n_rows, -1)
        after = np.where(flat_index > anchors[:, None] * r + r - 1, flat, 0.0).max(axis=1)
        gap = np.where(invalid.reshape(n_rows, -1), flat, 0.0).max(axis=1)
        for row in range(n_rows):
            move = float(result["value_input"][row] - result["value_entry"][row])
            out["rows"].append({
                **identity[int(sample[row])], "anchor": int(anchors[row]), "readout": readout, "baseline": baseline,
                "unit": UNITS[readout], "coordinate": int(coordinates[row].item()) if readout == MU_PRIOR else -1,
                "value_input": float(result["value_input"][row]), "value_baseline": float(result["value_baseline"][row]),
                "value_entry": float(result["value_entry"][row]),
                "entry_jump": float(result["value_entry"][row] - result["value_baseline"][row]),
                "attributed": float(result["map"][row].sum()), "abs_total": float(flat[row].sum()),
                "completeness_rel": abs(float(result["delta"][row])) / max(abs(move), abs(float(result["value_input"][row])), 1e-12),
                "completeness_strict": abs(float(result["delta"][row])) / max(abs(move), 1e-12),
                "after_anchor_max_abs": float(after[row]), "invalid_sample_max_abs": float(gap[row]),
                "lag50_s": float(statistics["lag50_s"][row]), "lag90_s": float(statistics["lag90_s"][row]),
                "share_in_stem": float(statistics["share_in_stem"][row]),
                "segment_median": float(median[row].item()),
            })
            out["vectors"].setdefault("lag_profile", []).append(lag[row].astype(np.float32))
    for element, (_, row) in enumerate(rows.iterrows()):
        key = str(row.get(labels.CLASS_COLUMN))
        ranked_out = "example_rank" in row.index and not np.isfinite(float(row["example_rank"]))
        offsets = np.flatnonzero(sample == element)
        if not want_examples.get(key) or ranked_out or not offsets.size:
            continue
        holder = traces.SegmentTrace(guid=str(row["guid"]), epoch=float(row["epoch"]), clinical_class=None, subgroup=None,
                                     anchor=np.zeros(0, dtype=np.int64), contributing=np.zeros(0, dtype=bool))
        traces.attach_raw_signals([holder], _one_sample(batch, element), scales)
        offset = int(offsets[0])
        anchor = int(anchors[offset])
        out["examples"].append({
            **identity[element], "anchor": anchor, "horizon": int(model.horizon), "steps": steps, "seconds": seconds,
            "raw": holder.raw, "raw_units": holder.raw_units,
            "maps": {variant: _blank_unread(maps[variant][offset], anchor, invalid[offset]) for variant in VARIANTS},
            "shares": {variant: lag_hist.normalise(np.abs(core.lag_profile(core.time_profile(maps[variant][offset:offset + 1]),
                                                                            [anchor], steps)))[0] for variant in VARIANTS},
        })
        want_examples[key] = False


def _one_sample(batch: Any, element: int) -> Dict[str, Any]:
    """The raw fields of one batch element, as a one-sample mapping for ``traces.attach_raw_signals``."""
    return {name: batch_field(batch, name)[element:element + 1] for name in ("fhr", "up", "weight")
            if batch_field(batch, name) is not None}


def _blank_unread(field: np.ndarray, anchor: int, invalid: np.ndarray) -> np.ndarray:
    """A ``(T, R)`` FHR map with the samples after the anchor token and the invalid samples set to NaN."""
    out = np.array(field, dtype=np.float64)
    out[anchor + 1:] = np.nan
    out[np.asarray(invalid, dtype=bool)] = np.nan
    return out


# =============================================================================
# Figures
# =============================================================================
def build_history_figure(rows: pd.DataFrame, profiles: np.ndarray, *, seconds: np.ndarray, reach: int, seed: int) -> Any:
    """Per readout: the |IG| share by lag under both baselines, its cumulative share, and the class means."""
    figure, axes = plt.subplots(len(READOUTS), 3, figsize=(14.0, 2.6 * len(READOUTS)), squeeze=False)
    stem_s = reach * float(core.SECONDS_PER_STEP)
    for index, readout in enumerate(READOUTS):
        ax_share, ax_cum, ax_class = axes[index]
        series = []
        drawn = False
        for baseline in (POP_MEAN, SEG_MEDIAN):
            keep = ((rows["readout"] == readout) & (rows["baseline"] == baseline)).to_numpy()
            if not keep.any():
                continue
            _guids, classes, curves = class_contrast.per_recording(rows[keep], lag_hist.normalise(np.abs(profiles[keep])))
            mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
            colour = BASELINE_COLOURS[baseline]
            ax_share.plot(seconds, mean, color=colour, linewidth=figures.LINE_REGULAR, label=baseline)
            ax_share.fill_between(seconds, lo, hi, color=colour, alpha=0.2, linewidth=0)
            series.append(mean)
            cumulative = np.cumsum(np.nan_to_num(curves, nan=0.0), axis=1)
            middle = np.nanmedian(cumulative, axis=0)
            ax_cum.plot(seconds, middle, color=colour, linewidth=figures.LINE_REGULAR,
                        label=f"{baseline}: lag90 {np.nanmedian(rows.loc[keep, 'lag90_s']):.0f} s")
            ax_cum.fill_between(seconds, np.nanquantile(cumulative, 0.25, axis=0), np.nanquantile(cumulative, 0.75, axis=0),
                                color=colour, alpha=0.15, linewidth=0)
            if baseline == POP_MEAN:
                class_contrast._class_lines(ax_class, seconds, class_contrast.by_class(classes, curves), seed=seed,
                                            title=f"{SHORT[readout]}: by class ({baseline})", xlabel=LAG_LABEL,
                                            ylabel="|IG| share", legend=True)
                ax_class.set_xscale("log")
                patch_figures.share_axis(ax_class, *[np.nanmean(c, axis=0) for c in class_contrast.by_class(classes, curves).values()], legend=False)
            drawn = True
        if not drawn:
            for ax in axes[index]:
                core._empty(ax, SHORT[readout])
            continue
        for ax in (ax_share, ax_cum):
            ax.axvline(stem_s, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN, linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel(LAG_LABEL)
            figures.style_axes(ax)
        ax_share.set_title(f"{SHORT[readout]}: |IG| share per 4 s token (dashed: conv-stem reach)")
        ax_share.set_ylabel("share")
        patch_figures.share_axis(ax_share, *series, ncol=2)
        ax_cum.set_title(f"{SHORT[readout]}: cumulative share (median, IQR over recordings)")
        ax_cum.set_ylabel("cumulative share")
        ax_cum.set_ylim(0.0, 1.02)
        ax_cum.legend(fontsize=figures.FONT_TINY, loc="lower right")
    figures.caveat_note(figure, NOTE)
    return figure


def build_checks_figure(rows: pd.DataFrame) -> Any:
    """Completeness residual per row (log axis) against the shared tolerance, per readout and baseline."""
    figure, ax = plt.subplots(1, 1, figsize=(14.0, 2.8))
    for index, (readout, baseline) in enumerate(VARIANTS):
        values = rows.loc[(rows["readout"] == readout) & (rows["baseline"] == baseline), "completeness_rel"].to_numpy(dtype=float)
        values = np.clip(values[np.isfinite(values)], 1e-12, None)
        ax.plot(values, np.full(values.size, index) + np.random.default_rng(index).uniform(-0.2, 0.2, values.size), ".",
                color=BASELINE_COLOURS[baseline], markersize=figures.MARKER_SMALL)
        if values.size:
            ax.plot([np.median(values)] * 2, [index - 0.3, index + 0.3], color=figures.COLOR_BLACK, linewidth=figures.LINE_HEAVY)
    ax.axvline(1e-2, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_THIN)
    ax.set_xscale("log")
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([f"{SHORT[r]} ({b})" for r, b in VARIANTS])
    ax.set_xlabel("relative completeness residual |ΣIG − (f(x) − f(x0))| / max(|Δf|, |f(x)|)")
    ax.set_title("Integrated-gradient completeness (tolerance 1e-2)")
    figures.style_axes(ax)
    return figure


def build_example_page(item: Mapping[str, Any], norms: Mapping[Tuple[str, str], Any], *, reach: int) -> Any:
    """One anchor on the samples-page layout: raw FHR and UP, one raw-FHR map per readout and baseline, the lag shares."""
    def history(ax: Any, cax: Any) -> None:
        cax.set_axis_off()
        series = []
        for (readout, baseline), share in item["shares"].items():
            series.append(share)
            ax.plot(item["seconds"], share, color=BASELINE_COLOURS[baseline], linewidth=figures.LINE_REGULAR,
                    linestyle="-" if readout in (BASE_LEVEL, SRC_EFFECT) else (":" if readout == BASE_VAR else "--"),
                    label=f"{SHORT[readout]} ({baseline})")
        ax.axvline(reach * float(core.SECONDS_PER_STEP), color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel(LAG_LABEL)
        ax.set_ylabel("|IG| share per token")
        patch_figures.share_axis(ax, *series, ncol=4)
        figures.style_axes(ax)

    map_rows = [((readout, baseline), f"{SHORT[readout]} ({baseline}): raw FHR attribution",
                 item["maps"][(readout, baseline)], UNITS[readout]) for readout, baseline in VARIANTS]
    return patch_figures.stacked_page(item, map_rows=map_rows, norms=norms, note=NOTE,
                                      lower=[("FHR history profile (dashed vertical: conv-stem reach)", history)])


# =============================================================================
# The analysis
# =============================================================================
def run_fhr_drivers_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Raw-FHR integrated gradients of the target-only branch and of the UP effect; see the module docstring."""
    task, loader, collection = getattr(context, "task", None), getattr(context, "loader", None), getattr(context, "collection", None)
    per_sample = getattr(collection, "per_sample", None)
    if task is None or loader is None or per_sample is None or per_sample.empty:
        return skip_record(NAME, "an attribution is a gradient of a forward: this pass built no model or loader")
    model = task.orig_model
    directory = Path(str(output_dir)) / NAME
    (directory / "maps").mkdir(parents=True, exist_ok=True)
    cap, anchors_per_segment = raw.cap(eval_config, CAP_SEGMENTS, DEFAULT_SEGMENTS), raw.cap(eval_config, CAP_ANCHORS, DEFAULT_ANCHORS)
    n_steps, examples_per_class = raw.cap(eval_config, CAP_IG_STEPS, DEFAULT_IG_STEPS), raw.cap(eval_config, CAP_EXAMPLES, DEFAULT_EXAMPLES)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    reach = stem_reach_tokens(model)
    scales = raw.raw_signal_scales(getattr(context, "config", None), loader)

    selected, accounting = raw.select_informative_segments(context, cap=cap, seed=seed, examples_per_class=examples_per_class)
    want = {str(name): True for name in selected[labels.CLASS_COLUMN].astype(str).unique()} if len(selected) else {}
    out: Dict[str, Any] = {"rows": [], "vectors": {}, "examples": [], "forward_equivalents": 0, "n_unclean": 0}
    started = time.perf_counter()
    for rows, batch in raw.segment_batches(task, loader, selected, batch_size=SEGMENTS_PER_BATCH):
        _attribute_batch(task, batch, rows, anchors_per_segment=anchors_per_segment, n_steps=n_steps, reach=reach,
                         scales=scales, want_examples=want, out=out)
    elapsed = time.perf_counter() - started
    rows = pd.DataFrame(out["rows"])
    if rows.empty:
        return {**skip_record(NAME, "no scored anchor in the drawn segments"), "selection": accounting}
    profiles = np.stack(out["vectors"]["lag_profile"])
    steps = int(profiles.shape[1])
    seconds = (np.arange(steps) * int(model.raw_per_step) + 0.5 * (int(model.raw_per_step) - 1)) / float(raw.events.FS_RAW)
    files: List[str] = [f"{NAME}_rows.csv", f"{NAME}_vectors.npz"]
    rows.to_csv(directory / files[0], index=False)
    np.savez_compressed(directory / files[1], lag_seconds=seconds, lag_profile=profiles,
                        row_guid=rows["guid"].astype(str).to_numpy(), row_readout=rows["readout"].astype(str).to_numpy(),
                        row_baseline=rows["baseline"].astype(str).to_numpy(), row_anchor=rows["anchor"].to_numpy())

    pieces, families = [], {}
    for readout, baseline in VARIANTS:
        part = rows[(rows["readout"] == readout) & (rows["baseline"] == baseline)]
        renamed = {m: f"{readout}_{baseline}_{m}" for m in ("lag50_s", "lag90_s", "share_in_stem")}
        families[f"fhr:{readout}:{baseline}"] = list(renamed.values())
        pieces.append(part.groupby("guid").agg({**{m: "mean" for m in renamed}, labels.CLASS_COLUMN: "first"}).rename(columns=renamed))
    recordings = pd.concat(pieces, axis=1)
    recordings = recordings.loc[:, ~recordings.columns.duplicated()]
    recordings.to_csv(directory / f"{NAME}_recordings.csv")
    stats, pairwise = class_contrast.class_tests(recordings, families, seed=seed)
    stats.to_csv(directory / f"{NAME}_class_stats.csv", index=False)
    pairwise.to_csv(directory / f"{NAME}_class_pairwise.csv", index=False)
    files += [f"{NAME}_recordings.csv", f"{NAME}_class_stats.csv", f"{NAME}_class_pairwise.csv"]

    with warnings.catch_warnings():
        # Lags before the record are NaN in every row of a short-anchor recording: an empty column, not an error.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        files.append(Path(figures.render_figure(build_history_figure(rows, profiles, seconds=seconds, reach=reach, seed=seed),
                                                directory / f"{NAME}_history")).name)
    files.append(Path(figures.render_figure(build_checks_figure(rows), directory / f"{NAME}_checks")).name)
    examples = out["examples"]
    norms = {variant: core.signed_log_norm(np.concatenate([np.ravel(e["maps"][variant]) for e in examples]))
             for variant in VARIANTS} if examples else {}
    for item in examples:
        stem = f"{traces.class_dirname(item[labels.CLASS_COLUMN])}_{traces.recording_stem(item['guid'], item[labels.SUBGROUP_COLUMN])}_anchor{item['anchor']}_{NAME}"
        files.append("maps/" + Path(figures.render_figure(build_example_page(item, norms, reach=reach), directory / "maps" / stem, tight=False)).name)

    variants_block: Dict[str, Any] = {}
    for (readout, baseline), part in rows.groupby(["readout", "baseline"]):
        variants_block[f"{readout}:{baseline}"] = {
            "n_rows": int(len(part)), "lag50_s_median": float(np.nanmedian(part["lag50_s"])),
            "lag90_s_median": float(np.nanmedian(part["lag90_s"])), "share_in_stem_mean": float(np.nanmean(part["share_in_stem"])),
            "value_input_mean": float(np.nanmean(part["value_input"])), "value_baseline_mean": float(np.nanmean(part["value_baseline"])),
            "entry_jump_mean": float(np.nanmean(part["entry_jump"])),
            "completeness_rel_median": float(np.nanmedian(part["completeness_rel"])),
        }
    checks = {
        "completeness_rel_median": float(np.nanmedian(rows["completeness_rel"])),
        "completeness_rel_max": float(np.nanmax(rows["completeness_rel"])),
        "completeness_strict_median": float(np.nanmedian(rows["completeness_strict"])),
        "n_rows_over_tolerance": int((rows["completeness_rel"] > 1e-2).sum()),
        "after_anchor_max_abs": float(rows["after_anchor_max_abs"].max()),
        "invalid_sample_max_abs": float(rows["invalid_sample_max_abs"].max()),
        "meaning": ("completeness_rel as the shared attribution pass defines it; after_anchor: largest |IG| on a raw FHR "
                    "sample after the anchor token, exactly 0; invalid_sample: largest |IG| on a sample the evaluation's own "
                    "pass masks (a weight gap or a non-finite sample, held at its original validity), exactly 0"),
    }
    by_class = {str(k): int(v) for k, v in selected[labels.CLASS_COLUMN].value_counts().items()}
    headline = {
        "base_level_lag90_s": variants_block.get(f"{BASE_LEVEL}:{POP_MEAN}", {}).get("lag90_s_median"),
        "src_effect_lag50_s": variants_block.get(f"{SRC_EFFECT}:{POP_MEAN}", {}).get("lag50_s_median"),
        "completeness_rel_max": checks["completeness_rel_max"],
    }
    logger.info(f"{NAME}: {len(selected)} segment(s), {len(rows)} row(s) in {elapsed:.1f} s; "
                f"base level lag90 {headline['base_level_lag90_s']} s, completeness max {checks['completeness_rel_max']:.1e}")
    return {
        "n_samples": int(len(selected)),
        "composition": {"n_recordings": int(selected["guid"].nunique()), "n_segments_by_class": by_class,
                        "n_rows": int(len(rows)), "n_anchors": int(len(rows) // len(VARIANTS))},
        "plan": {"capped": True, "cap": cap, "anchors_per_segment": anchors_per_segment, "ig_steps": n_steps, "seed": seed,
                 "variants": [f"{r}:{b}" for r, b in VARIANTS], "entry_fraction": core.BASELINE_ENTRY_FRACTION,
                 "baselines": {POP_MEAN: "raw FHR = 0 (population mean)", SEG_MEDIAN: "flat at the segment's median valid FHR up to the anchor"},
                 "stem_reach_tokens": reach, "n_unclean_anchors": int(out["n_unclean"]),
                 "mu_prior_coordinate": "argmax |mu_prior| at the anchor, per row"},
        "selection": accounting,
        "cost": core.cost_record(elapsed_s=elapsed, n_segments=int(len(selected)), n_rows=int(len(rows)),
                                 n_forward_equivalents=int(out["forward_equivalents"]), device=getattr(task, "device", None)),
        "checks": checks,
        "variants": variants_block,
        "class_contrast": {"significant": stats.loc[stats["significant"].astype(bool), "metric"].tolist()
                           if "significant" in stats else [], "n_metrics_tested": int(len(stats))},
        "headline": headline,
        "caveat": NOTE,
        "files": files,
    }
