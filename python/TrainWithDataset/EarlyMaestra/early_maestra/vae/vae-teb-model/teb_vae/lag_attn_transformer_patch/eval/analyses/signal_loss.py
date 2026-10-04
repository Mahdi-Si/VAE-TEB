r"""``signal_loss`` -- how FHR signal loss (``missing`` tokens) moves the latent and the forecast (E1-L).

**Question.** Observational gap fraction against KL, logvar and NLL, plus an interventional
gap-injection sweep: across a gap, does the model fall back to the prior (KL down) or inflate its
variance?

**Pass.** A seeded draw of scored segments (cap ``signal_loss_segments``, default
:data:`DEFAULT_SEGMENTS`) is re-forwarded densely through ``raw.forward_raw``.

* **Observational.** Per decoded anchor ``a``, over the history window = the lag window, tokens
  ``[a - max_lag, a]`` (clipped at 0): ``gap_frac_weight = 1 - mean(clip(weight, 0, 1))`` and
  ``missing_frac`` = the fraction of those tokens the model replaces with the learned ``missing``
  embedding (``weight < 1`` or a non-finite sample in the patch). Readouts: ``kld_per_t``, mean
  ``logvar_prior``/``logvar_post`` at ``a`` (this forward), and ``nll_full_block``/``pred_gap`` joined
  from ``per_anchor.parquet`` by ``(guid, epoch, anchor)``. Binned curves (median, IQR; a bin with
  fewer than :data:`MIN_BIN_ANCHORS` anchors is blank) and pooled Spearman ρ. ``missing_occurrence``
  says how often ``missing`` appears in the draw: the embedding is rarely trained (DESIGN.md §4).
* **Interventional.** On the drawn segments with no ``missing`` token at all (cap
  ``signal_loss_inject_segments``, default :data:`DEFAULT_INJECT_SEGMENTS`), at :data:`ANCHORS_PER_SEGMENT`
  anchors spread over the segment, ``weight`` is set to 0 over a contiguous FHR span of
  :data:`GAP_S` seconds ending :data:`DISTANCE_S` seconds before the anchor (0 = the span ends at,
  and includes, the anchor token). This is how a real gap reaches the model: the FHR tokens become
  ``missing`` (and, under ``source_validity: fhr_weight``, the UP tokens too; under ``finite`` UP stays
  visible). The target and the scored future are unchanged. Readouts at ``a``, edited minus clean:
  ``d_kld``, ``d_logvar_prior``, ``d_logvar_post``, ``prior_shift`` (``‖Δμ^p‖``), the mean-decoded level
  forecast shift ``level_shift_{base,full}_bpm`` (mean over τ of ``|Δμ|``, bpm), ``d_logvar_full_level``
  (the level forecast's log-variance, > 0 = wider), and the share of the KL-lag map and of the
  head-averaged attention on the lags that read the gap's span (lag ℓ = source token ``a - ℓ``).

**Outputs** (``signal_loss/``): ``signal_loss_anchors.csv``, ``signal_loss_binned.csv``,
``signal_loss_observational.pdf``, ``signal_loss_injection.csv``, ``signal_loss_injection_summary.csv``,
``signal_loss_injection.pdf``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval.attributions import symlog_legend
from teb_vae.lag_attn_cfs.eval.dataset_rows import epoch_stamp
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "signal_loss"
CAP_NAME = f"{NAME}_segments"
INJECT_CAP_NAME = f"{NAME}_inject_segments"
DEFAULT_SEGMENTS = 1024
DEFAULT_INJECT_SEGMENTS = 64
SEED_OFFSET = 43
ANCHORS_PER_SEGMENT = 4
GAP_S: Tuple[float, ...] = (4.0, 16.0, 60.0, 120.0)
DISTANCE_S: Tuple[float, ...] = (0.0, 60.0, 180.0)
#: Bin edges of the gap fractions; the first bin is exactly zero.
BIN_EDGES: Tuple[float, ...] = (0.0, 1e-9, 0.1, 0.25, 0.5, 1.0)
BIN_LABELS: Tuple[str, ...] = ("0", "(0, .1]", "(.1, .25]", "(.25, .5]", "(.5, 1]")
MIN_BIN_ANCHORS = 20
GAP_COLUMNS: Dict[str, str] = {"gap_frac_weight": "FHR gap fraction (weight)", "missing_frac": "missing-token fraction"}
READOUTS: Dict[str, str] = {
    "kld_per_t": "$K_t$ (nats)",
    "logvar_prior": "mean log-var prior",
    "logvar_post": "mean log-var posterior",
    "nll_full_block": "NLL full (nats)",
    "pred_gap": "pred gap (nats)",
}
EFFECTS: Dict[str, str] = {
    "d_kld": "Δ$K_t$ (nats)",
    "d_logvar_prior": "Δ log-var prior",
    "d_logvar_post": "Δ log-var posterior",
    "level_shift_full_bpm": "|Δ level forecast|, full (bpm)",
    "d_logvar_full_level": "Δ log-var level forecast",
    "d_attn_share_gap": "Δ attention share on gap lags",
}
DISTANCE_COLORS = (figures.COLOR_BLUE, figures.COLOR_ORANGE, figures.COLOR_GREEN)

#: ``(headline_name, key)`` pairs registered as ``("signal_loss", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("signal_loss_missing_token_frac", "missing_token_frac"),
    ("signal_loss_d_kld_120s_at_anchor", "d_kld_120s_d0"),
    ("signal_loss_d_logvar_prior_120s_at_anchor", "d_logvar_prior_120s_d0"),
    ("signal_loss_level_shift_bpm_120s_at_anchor", "level_shift_full_bpm_120s_d0"),
)


def _windowed_mean(values: np.ndarray, anchors: np.ndarray, width: int) -> np.ndarray:
    """Mean of ``values (T,)`` over tokens ``[a - width + 1, a]`` clipped at 0, per anchor."""
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    lo = np.maximum(anchors - width + 1, 0)
    return (csum[anchors + 1] - csum[lo]) / (anchors + 1 - lo)


# =============================================================================
# Observational
# =============================================================================
@torch.no_grad()
def observe(task: Any, loader: Any, rows: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """Dense forward of every drawn segment; per-anchor gap fractions and readouts, the census, and the clean segments' raw signals."""
    model = task.orig_model
    r, window = int(model.raw_per_step), int(model.max_lag) + 1
    frames, clean = [], []
    n_tokens = n_missing = n_segments_missing = 0
    for chunk, batch in raw.segment_batches(task, loader, rows):
        fhr, up, weight = raw.batch_signals(batch)
        outputs = raw.forward_raw(model, fhr, up, weight)
        token_valid = raw.sample_validity(fhr, weight, raw_per_step=r, validity="fhr_weight").view(fhr.shape[0], -1, r).all(-1)
        missing = (~token_valid).double().cpu().numpy()
        w = weight.double().clamp(0.0, 1.0).cpu().numpy()
        for b, (_, row) in enumerate(chunk.iterrows()):
            keep = outputs["anchor_valid"][b].bool()
            anchors = outputs["anchor_index"][b][keep].long()
            a = anchors.cpu().numpy()
            n_tokens += missing[b].size
            n_missing += int(missing[b].sum())
            n_segments_missing += int(missing[b].any())
            frames.append(pd.DataFrame({
                "guid": str(row["guid"]), "epoch": float(row["epoch"]), "anchor": a,
                "gap_frac_weight": 1.0 - _windowed_mean(w[b], a, window),
                "missing_frac": _windowed_mean(missing[b], a, window),
                "kld_per_t": outputs["kld_per_t"][b, anchors].double().cpu().numpy(),
                "logvar_prior": outputs["logvar_prior"][b, anchors].mean(-1).double().cpu().numpy(),
                "logvar_post": outputs["logvar_post"][b, anchors].mean(-1).double().cpu().numpy(),
            }))
            if not missing[b].any():
                clean.append({"guid": str(row["guid"]), "epoch": float(row["epoch"]),
                              "fhr": fhr[b].cpu(), "up": up[b].cpu(), "weight": weight[b].cpu(),
                              "anchors": a})
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    census = {
        "history_window_tokens": window,
        "missing_token_frac": n_missing / max(n_tokens, 1),
        "segments_with_missing_frac": n_segments_missing / max(len(rows), 1),
        "anchors_with_missing_in_window_frac": float((table["missing_frac"] > 0).mean()) if len(table) else None,
        "n_segments": int(len(rows)), "n_clean_segments": int(len(clean)),
    }
    return table, census, clean


def join_scores(table: pd.DataFrame, per_anchor: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach the collected ``nll_full_block`` and ``pred_gap`` by ``(guid, epoch, anchor)``; NaN where not scored."""
    wanted = ["nll_full_block", "pred_gap"]
    if per_anchor is None or per_anchor.empty or not set(wanted) <= set(per_anchor.columns):
        return table.assign(**{name: np.nan for name in wanted})
    right = per_anchor[["guid", "epoch", "anchor", *wanted]].copy()
    right["guid"] = right["guid"].astype(str)
    right["_stamp"] = [epoch_stamp(e) for e in right["epoch"]]
    right["anchor"] = right["anchor"].astype(np.int64)
    left = table.assign(_stamp=[epoch_stamp(e) for e in table["epoch"]])
    return left.merge(right.drop(columns=["epoch"]), on=["guid", "_stamp", "anchor"], how="left").drop(columns=["_stamp"])


def binned(table: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of every readout per gap-fraction bin, for both gap measures."""
    records = []
    for gap in GAP_COLUMNS:
        bins = pd.cut(table[gap], bins=list(BIN_EDGES), labels=list(BIN_LABELS), include_lowest=True)
        for label in BIN_LABELS:
            cell = table[bins == label]
            for readout in READOUTS:
                values = cell[readout].dropna().to_numpy(np.float64)
                q = np.quantile(values, [0.25, 0.5, 0.75]) if values.size else [np.nan] * 3
                records.append({"gap_measure": gap, "bin": label, "readout": readout, "n": int(values.size),
                                "q25": q[0], "median": q[1], "q75": q[2]})
    return pd.DataFrame(records)


def spearman(table: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Pooled Spearman ρ of each gap measure against each readout (``None`` when undefined)."""
    from scipy.stats import spearmanr

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for gap in GAP_COLUMNS:
        out[gap] = {}
        for readout in READOUTS:
            both = table[[gap, readout]].dropna()
            constant = len(both) < 3 or both[gap].nunique() < 2 or both[readout].nunique() < 2
            out[gap][readout] = None if constant else raw.finite_or_none(spearmanr(both[gap], both[readout])[0])
    return out


def observational_figure(bins: pd.DataFrame, census: Dict[str, Any]) -> Any:
    """Readout rows × gap-measure columns: median with IQR per bin; sparse bins blank."""
    fig, axes = figures.new_figure(len(READOUTS), len(GAP_COLUMNS), height_per_row=1.3)
    x = np.arange(len(BIN_LABELS))
    for j, gap in enumerate(GAP_COLUMNS):
        for i, readout in enumerate(READOUTS):
            ax = axes[i, j]
            cell = bins[(bins["gap_measure"] == gap) & (bins["readout"] == readout)].set_index("bin").reindex(BIN_LABELS)
            sparse = cell["n"].to_numpy() < MIN_BIN_ANCHORS
            med, lo, hi = (np.where(sparse, np.nan, cell[c].to_numpy(np.float64)) for c in ("median", "q25", "q75"))
            ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color=figures.COLOR_BLUE,
                        markersize=figures.MARKER_SMALL, linewidth=figures.LINE_REGULAR)
            ax.set_xticks(x, BIN_LABELS if i == len(READOUTS) - 1 else [""] * len(x))
            ax.set_xlim(-0.5, len(x) - 0.5)
            ax.set_ylabel(READOUTS[readout] if j == 0 else "")
            if i == 0:
                ax.set_title(GAP_COLUMNS[gap])
            if i == len(READOUTS) - 1:
                ax.set_xlabel(f"fraction over the last {census['history_window_tokens']} tokens")
            figures.style_axes(ax)
    figures.caveat_note(fig, (
        f"missing tokens: {100 * census['missing_token_frac']:.2f}% of tokens, "
        f"{100 * census['segments_with_missing_frac']:.1f}% of {census['n_segments']} segments; "
        f"bins with < {MIN_BIN_ANCHORS} anchors blank"
    ))
    return fig


# =============================================================================
# Interventional
# =============================================================================
def injection_anchors(anchors: np.ndarray, *, lowest: int) -> np.ndarray:
    """:data:`ANCHORS_PER_SEGMENT` anchors spread evenly from ``lowest`` to the last decoded anchor."""
    eligible = anchors[anchors >= lowest]
    if not eligible.size:
        return eligible
    picks = np.linspace(0, eligible.size - 1, min(ANCHORS_PER_SEGMENT, eligible.size)).round().astype(int)
    return np.unique(eligible[picks])


@torch.no_grad()
def inject(task: Any, clean: List[Dict[str, Any]], units: raw.SummaryUnits, batch_size: int) -> pd.DataFrame:
    """Every (segment, anchor, gap, distance) edit plus one clean row per (segment, anchor); edited minus clean."""
    model = task.orig_model
    tokens_per_s = 1.0 / (float(model.raw_per_step) / raw.events.FS_RAW)
    gaps = [max(1, int(round(g * tokens_per_s))) for g in GAP_S]
    dists = [int(round(d * tokens_per_s)) for d in DISTANCE_S]
    plan = []  # (segment, anchor, gap_s, distance_s, first_token, last_token); gap_s 0 = clean
    for i, seg in enumerate(clean):
        for a in injection_anchors(seg["anchors"], lowest=max(dists) + max(gaps) - 1):
            plan.append((i, int(a), 0.0, 0.0, 0, -1))
            for g_s, g in zip(GAP_S, gaps):
                for d_s, d in zip(DISTANCE_S, dists):
                    plan.append((i, int(a), g_s, d_s, int(a) - d - g + 1, int(a) - d))
    lags = int(model.max_lag) + 1
    records = []
    for start in range(0, len(plan), batch_size):
        part = plan[start:start + batch_size]
        weight = torch.stack([clean[p[0]]["weight"] for p in part]).clone()
        for row, p in enumerate(part):
            weight[row, p[4]:p[5] + 1] = 0.0
        fhr = torch.stack([clean[p[0]]["fhr"] for p in part]).to(task.device)
        up = torch.stack([clean[p[0]]["up"] for p in part]).to(task.device)
        outputs = raw.forward_raw(model, fhr, up, weight.to(task.device))
        rows = torch.arange(len(part), device=task.device)
        anchors = torch.tensor([p[1] for p in part], device=task.device)
        position = (outputs["anchor_index"] == anchors[:, None]).long().argmax(1)
        mu_base = raw.decode_at(model, outputs, outputs["mu_prior"], position[:, None])[0][:, 0]
        mu_full, logvar_full = (x[:, 0] for x in raw.decode_at(model, outputs, outputs["mu_post"], position[:, None]))
        host = {name: value.double().cpu().numpy() for name, value in {
            "kld": outputs["kld_per_t"][rows, anchors],
            "logvar_prior": outputs["logvar_prior"][rows, anchors].mean(-1),
            "logvar_post": outputs["logvar_post"][rows, anchors].mean(-1),
            "mu_prior": outputs["mu_prior"][rows, anchors],
            "mu_base": mu_base, "mu_full": mu_full,
            "logvar_full_level": logvar_full[..., 0].mean(-1),
            "kl_lag": outputs["source_kl_lag_map"][rows, anchors],
            "attention": outputs["attn_weights"].mean(dim=2)[rows, anchors],  # head-averaged
        }.items()}
        for row, p in enumerate(part):
            in_gap = np.zeros(lags, dtype=bool)
            if p[2] > 0:
                in_gap[max(p[1] - p[5], 0):min(p[1] - p[4] + 1, lags)] = True
            records.append({
                "segment": p[0], "guid": clean[p[0]]["guid"], "epoch": clean[p[0]]["epoch"], "anchor": p[1],
                "gap_s": p[2], "distance_s": p[3], "in_gap": in_gap,
                **{name: values[row] for name, values in host.items()},
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    base = frame[frame["gap_s"] == 0].set_index(["segment", "anchor"])
    edits = frame[frame["gap_s"] > 0].reset_index(drop=True)
    ref = base.loc[list(zip(edits["segment"], edits["anchor"]))].reset_index(drop=True)
    level = units.level_delta_bpm
    out = edits[["guid", "epoch", "anchor", "gap_s", "distance_s"]].copy()
    out["d_kld"] = edits["kld"] - ref["kld"]
    out["d_logvar_prior"] = edits["logvar_prior"] - ref["logvar_prior"]
    out["d_logvar_post"] = edits["logvar_post"] - ref["logvar_post"]
    out["prior_shift"] = [float(np.linalg.norm(e - c)) for e, c in zip(edits["mu_prior"], ref["mu_prior"])]
    for branch in ("base", "full"):
        out[f"level_shift_{branch}_bpm"] = [
            float(level(np.abs(e[:, 0] - c[:, 0]).mean())) for e, c in zip(edits[f"mu_{branch}"], ref[f"mu_{branch}"])
        ]
    out["d_logvar_full_level"] = edits["logvar_full_level"] - ref["logvar_full_level"]
    for name in ("kl_lag", "attention"):
        share = {
            side: np.array([v[m].sum() / v.sum() if v.sum() > 0 else np.nan for v, m in zip(side_frame[name], edits["in_gap"])])
            for side, side_frame in (("edit", edits), ("clean", ref))
        }
        stem = "kl_lag" if name == "kl_lag" else "attn"
        out[f"{stem}_share_gap"] = share["edit"]
        out[f"d_{stem}_share_gap"] = share["edit"] - share["clean"]
    return out


def injection_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of every effect per (gap length, distance), with row and recording counts."""
    records = []
    for (g, d), cell in effects.groupby(["gap_s", "distance_s"]):
        record = {"gap_s": g, "distance_s": d, "n_rows": int(len(cell)), "n_recordings": int(cell["guid"].nunique())}
        for name in [*EFFECTS, "prior_shift", "level_shift_base_bpm", "d_kl_lag_share_gap", "attn_share_gap"]:
            q = np.nanquantile(cell[name].to_numpy(np.float64), [0.25, 0.5, 0.75])
            record.update({f"{name}_q25": q[0], f"{name}_median": q[1], f"{name}_q75": q[2]})
        records.append(record)
    return pd.DataFrame(records)


def injection_figure(summary: pd.DataFrame) -> Any:
    """Effect rows against gap length (log x), one line per distance before the anchor; symlog y."""
    fig, axes = figures.new_figure(3, 2, height_per_row=2.4)
    for i, (ax, name) in enumerate(zip(axes.flat, EFFECTS)):
        drawn = []
        for color, d in zip(DISTANCE_COLORS, DISTANCE_S):
            cell = summary[summary["distance_s"] == d].sort_values("gap_s")
            med, lo, hi = (cell[f"{name}_{q}"].to_numpy(np.float64) for q in ("median", "q25", "q75"))
            ax.plot(cell["gap_s"], med, "o-", color=color, markersize=figures.MARKER_SMALL,
                    linewidth=figures.LINE_REGULAR, label=f"ends {d:.0f} s before the anchor")
            ax.fill_between(cell["gap_s"], lo, hi, color=color, alpha=0.2, linewidth=0)
            drawn += [med, lo, hi]
        ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
        ax.set_xscale("log")
        ax.set_xticks(list(GAP_S), [f"{g:.0f}" for g in GAP_S])
        ax.set_ylabel(EFFECTS[name])
        symlog_legend(ax, *drawn, ncol=1, legend=i == 0)
        if i >= len(EFFECTS) - 2:
            ax.set_xlabel("injected FHR gap length (s, weight = 0)")
        figures.style_axes(ax)
    return fig


# =============================================================================
# The registry entry point
# =============================================================================
def run_signal_loss_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Observe gap fractions against the readouts, then inject gaps into clean segments and read the response."""
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return skip_record(NAME, "the gap census and the injection need the model and the loader (offline re-run)")
    caps = eval_config.get("caps") or {}
    cap = int(caps.get(CAP_NAME) or DEFAULT_SEGMENTS)
    inject_cap = int(caps.get(INJECT_CAP_NAME) or DEFAULT_INJECT_SEGMENTS)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    rows = raw.draw_segments(context, cap=cap, seed=seed, max_hours=eval_config.get("max_hours_before_delivery"))
    if rows.empty:
        return skip_record(NAME, "no scored segment resolves to a dataset row")

    started = time.perf_counter()
    table, census, clean = observe(task, loader, rows)
    table = join_scores(table, getattr(context.collection, "per_anchor", None))
    bins, rho = binned(table), spearman(table)
    observed_s = time.perf_counter() - started

    clean = clean[:inject_cap]
    units = raw.SummaryUnits.from_model(task.orig_model, context.config, loader)
    effects = inject(task, clean, units, max(1, int(getattr(loader, "batch_size", None) or 1)))
    summary = injection_summary(effects) if len(effects) else pd.DataFrame()
    elapsed_s = time.perf_counter() - started

    directory = Path(output_dir) / NAME
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"{NAME}_anchors.csv", index=False)
    bins.to_csv(directory / f"{NAME}_binned.csv", index=False)
    files = [f"{NAME}_anchors.csv", f"{NAME}_binned.csv",
             figures.render_figure(observational_figure(bins, census), directory / f"{NAME}_observational").name]
    headline = {"missing_token_frac": raw.finite_or_none(census["missing_token_frac"])}
    injection: List[Dict[str, Any]] = []
    if len(summary):
        effects.to_csv(directory / f"{NAME}_injection.csv", index=False)
        summary.to_csv(directory / f"{NAME}_injection_summary.csv", index=False)
        files += [f"{NAME}_injection.csv", f"{NAME}_injection_summary.csv",
                  figures.render_figure(injection_figure(summary), directory / f"{NAME}_injection").name]
        injection = [{k: (raw.finite_or_none(v) if isinstance(v, float) else v) for k, v in rec.items()}
                     for rec in summary.to_dict(orient="records")]
        longest = summary[(summary["gap_s"] == max(GAP_S)) & (summary["distance_s"] == 0.0)]
        for name in ("d_kld", "d_logvar_prior", "level_shift_full_bpm"):
            key = f"{name}_{max(GAP_S):.0f}s_d0"
            headline[key] = raw.finite_or_none(longest[f"{name}_median"].iloc[0]) if len(longest) else None
    logger.info(
        f"{NAME}: {len(rows)} segment(s), missing tokens {100 * census['missing_token_frac']:.3f}%, "
        f"{len(clean)} clean segment(s) injected ({len(effects)} edit row(s)), {elapsed_s:.1f} s"
    )
    return {
        "n_samples": int(len(rows)),
        "composition": {"n_recordings": int(rows["guid"].nunique()), "n_clean_segments_injected": int(len(clean))},
        "plan": {"capped": True, "cap": cap, "inject_cap": inject_cap, "seed": seed,
                 "gap_s": list(GAP_S), "distance_s": list(DISTANCE_S), "anchors_per_segment": ANCHORS_PER_SEGMENT,
                 "edit": "weight = 0 over the span (FHR tokens -> missing)"},
        "cost": {"elapsed_s": float(elapsed_s), "observational_s": float(observed_s),
                 "seconds_per_segment": float(observed_s / max(1, len(rows)))},
        "missing_occurrence": census,
        "spearman": rho,
        "injection": injection,
        "headline": {key: headline.get(key) for _, key in HEADLINE},
        "files": files,
    }
