r"""$\beta$ rate-distortion sweep, HP probes & rank-at-$\beta$ harness (Phase 6).

The shipped ``kld_beta=0.001`` barely regularises the bottleneck, so the TE
surrogate $\bar K$ read off a single run proves nothing -- the **sweep** is the
experiment. TEB theory (``model_validation.md`` Section 8) says exact TE
estimation is expected only near an appropriate $\beta$. This module operates
that experiment with three run modes:

    * ``beta`` (tasks 6.1, 6.2) -- train the fixed $(a, M)$ benchmark dataset at
      every $\beta$ in ``beta_sweep.grid``, evaluate each checkpoint, build the
      rate-distortion curve and recommend a $\beta$.
    * ``beta_grid`` -- cross ``beta_sweep.grid`` with the full
      $(M, \mathrm{TE})$ sweep grid (``enumerate_sweep``) so $\bar K$ vs $\beta$
      can be read as one coloured line per $M$ (faceted by TE) and per TE
      (faceted by $M$), plus a per-$\beta$ calibration slice. The
      $(M, \mathrm{TE})$ grid is the active benchmark's ``sweep.m_grid`` /
      ``target_te_grid``, optionally narrowed by ``beta_sweep.beta_grid``.
    * ``hp`` (task 6.4) -- the same orchestration over a secondary
      hyper-parameter axis (``lambda_base`` / ``d_z`` / ``warmup_period`` from
      ``beta_sweep.hp_probes``), at the config's $\beta$.
    * ``rank_at_beta`` (task 6.3) -- re-run the Phase-4 A-sweep
      (``a_grid`` $\times$ ``m_grid``) at a chosen $\beta$ and report the
      Spearman rank correlation $\rho(\bar K, \mathrm{TE}_{\rm true})$, to
      confirm it improves over the default $\beta = 0.001$.

This module **reuses** :mod:`train_minimal` (the real training loop) and
:mod:`evaluate_te` (the real per-checkpoint scoring and Metric 1-4 machinery)
wholesale -- it reimplements no training, loss, KL or metric code. Per-cell
training is :func:`train_minimal.train`; per-cell scoring is
:func:`evaluate_te.evaluate_checkpoint`.

Run modes (project convention -- Decision D9 in
``synthetic_te_validation_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``, auto-detected from
whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.beta_sweep --mode beta [--build-missing]
            [--train-missing] [--config PATH] [--benchmark B] [--device DEV]
            [--seed S]
        python -m ...synthetic.beta_sweep --mode hp --axis lambda_base ...
        python -m ...synthetic.beta_sweep --mode rank_at_beta --beta 0.003 ...

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.beta_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    build_dataset as bd,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    evaluate_te as ev,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.*`` config values are resolved relative to ``model_experiment/``
# (identical convention to train_minimal.py / evaluate_te.py).
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Sweep axis -> (config section, field). Every axis -- including ``beta`` -- is
# written onto the per-cell ``cfg`` dict directly before :func:`train_minimal.train`.
# The HP axes are absent from ``train_minimal._OVERRIDE_MAP`` on purpose: D9
# sanctions editing the ``config`` dict directly for settings with no override
# key, so ``train_minimal.py`` needs no edit (comment C18).
_AXIS_FIELD: Dict[str, Tuple[str, str]] = {
    "beta": ("loss", "kld_beta"),
    "lambda_base": ("loss", "lambda_base"),
    "d_z": ("model", "d_z"),
    "warmup_period": ("model", "warmup_period"),
}
_HP_AXES = ("lambda_base", "d_z", "warmup_period")

# Per-cell summary CSV columns (one row per swept value). The d_z-length
# ``per_dim_kl`` vector goes to ``analysis.json`` instead of this flat CSV.
_SUMMARY_FIELDS = [
    "axis", "value", "beta", "run_tag", "data_tag", "te_true", "te_per_step",
    "k_bar", "k_bar_shuffled", "pred_gap", "feat_loss", "base_loss", "kld_loss",
    "mu_post_prior_gap", "attn_entropy", "epoch", "ckpt_path",
]

# ``beta_grid`` mode (--mode beta_grid) crosses beta x M x TE, so its per-cell
# CSV/JSON carries the grid coordinates (``M``, ``target_te``) and whichever
# generator knob the active benchmark solved per cell (``B_y`` for G1, ``c`` for
# G2, ``p_switch`` for G3) on top of the shared 1-D-sweep columns.
_BETA_GRID_FIELDS = _SUMMARY_FIELDS + ["M", "target_te", "B_y", "c", "p_switch"]


# =============================================================================
# Tag / path helpers
# =============================================================================

def _fmt_token(value: Any) -> str:
    """Filesystem-safe token for a numeric sweep value.

    Args:
        value: A numeric sweep value (``beta``, ``lambda_base``, ...).

    Returns:
        A compact token with ``.`` -> ``p`` and ``-`` -> ``m`` (reusing the
        :mod:`evaluate_te` convention), e.g. ``3e-05`` -> ``3em05``,
        ``0.25`` -> ``0p25``.
    """
    return f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")


def _cell_tags(axis: str, value: Any, sweep_dir_tag: str) -> Tuple[str, str]:
    """Filesystem-safe ``(cell, run_tag)`` for one sweep cell.

    Args:
        axis: The sweep axis (``beta`` or an HP axis).
        value: The swept value for this cell.
        sweep_dir_tag: The parent results subdirectory (e.g. ``beta_sweep``).

    Returns:
        Tuple ``(cell, run_tag)`` -- ``cell`` is the leaf name (``b3em05`` for
        the $\\beta$ axis, the bare value token otherwise, since the axis is
        already encoded in ``sweep_dir_tag``); ``run_tag`` is
        ``<sweep_dir_tag>/<cell>``.
    """
    token = _fmt_token(value)
    cell = f"b{token}" if axis == "beta" else token
    return cell, f"{sweep_dir_tag}/{cell}"


def _fixed_data_tag(config: Dict[str, Any]) -> str:
    """Cache tag of the dataset the ``beta`` / ``hp`` sweep trains on.

    Reads ``beta_sweep.data_tag`` (defaulting to ``experiment.tag``). Warns --
    but does not fail -- when ``beta_sweep.fixed_setting`` disagrees with the
    ``data`` block; the sweep trains on whatever is cached under this tag.

    Args:
        config: The parsed ``config_synth.yaml``.

    Returns:
        The cache tag string.
    """
    bs = config.get("beta_sweep", {}) or {}
    tag = str(bs.get("data_tag") or config["experiment"]["tag"])
    fixed = bs.get("fixed_setting", {}) or {}
    data = config.get("data", {}) or {}
    for cfg_key, fix_key in (("a", "a"), ("M", "M")):
        fv = fixed.get(fix_key)
        dv = data.get(cfg_key)
        if fv is not None and dv is not None:
            if abs(float(fv) - float(dv)) > 1e-6:
                print(
                    f"[warn] beta_sweep.fixed_setting.{fix_key}={fv} != "
                    f"data.{cfg_key}={dv}; the sweep trains on the cached "
                    f"dataset '{tag}' as-is."
                )
    return tag


def _ensure_fixed_dataset(
    config: Dict[str, Any], data_tag: str, *, build_missing: bool
) -> None:
    """Ensure the fixed $(a, M)$ sweep dataset is cached.

    Args:
        config: The parsed config.
        data_tag: Cache tag of the fixed dataset.
        build_missing: If True, generate the dataset when it is not cached;
            otherwise raise.

    Raises:
        FileNotFoundError: If the dataset is not cached and ``build_missing``
            is False.
    """
    benchmark = str(config["experiment"]["benchmark"])
    cache_dir = ev._data_root(config) / benchmark / data_tag
    needed = ("train.npz", "val.npz", "test.npz")
    if all((cache_dir / f).is_file() for f in needed):
        return
    if not build_missing:
        raise FileNotFoundError(
            f"fixed sweep dataset not cached: {cache_dir}\n"
            f"Build it first, e.g.:\n"
            f"  python -m model.vae_teb_prediction.model.model_experiment."
            f"synthetic.build_dataset --tag {data_tag}\n"
            f"or re-run this sweep with build_missing=True."
        )
    fixed = (config.get("beta_sweep", {}) or {}).get("fixed_setting", {}) or {}
    cfg = deepcopy(config)
    bd._apply_overrides(
        cfg, {"tag": data_tag, "a": fixed.get("a"), "m": fixed.get("M")}
    )
    print(f"  [build] fixed sweep dataset '{data_tag}'")
    bd.build_dataset(cfg, force=False)


# =============================================================================
# Single sweep cell (train if needed, then evaluate)
# =============================================================================

def _run_one(
    config: Dict[str, Any],
    *,
    axis: str,
    value: Any,
    fixed_data_tag: str,
    sweep_dir_tag: str,
    build_missing: bool,
    train_missing: bool,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    r"""Train (if needed) and evaluate one sweep cell.

    Builds a per-cell config -- the swept ``value`` written directly onto its
    ``_AXIS_FIELD`` location -- trains via :func:`train_minimal.train` when the
    checkpoint is missing, then scores it with
    :func:`evaluate_te.evaluate_checkpoint` on the fixed dataset's test split.

    Args:
        config: The parsed config.
        axis: The sweep axis.
        value: The swept value for this cell.
        fixed_data_tag: Cache tag of the fixed $(a, M)$ dataset.
        sweep_dir_tag: Parent results subdirectory (e.g. ``beta_sweep``).
        build_missing: Unused here (the fixed dataset is ensured by the caller);
            kept for signature symmetry.
        train_missing: If True, train the cell when its checkpoint is missing.
        device: Compute device.

    Returns:
        The :func:`evaluate_te.evaluate_checkpoint` row augmented with ``axis``,
        ``value`` and ``beta``; or ``None`` when the checkpoint is missing and
        ``train_missing`` is False.
    """
    benchmark = str(config["experiment"]["benchmark"])
    _cell, run_tag = _cell_tags(axis, value, sweep_dir_tag)
    ckpt_path = ev._results_root(config) / benchmark / run_tag / "final.ckpt"

    if not ckpt_path.is_file():
        if not train_missing:
            print(
                f"  [skip ] {run_tag}: checkpoint not found ({ckpt_path}) "
                f"(pass train_missing=True to train it)."
            )
            return None
        # Per-cell config: write the swept value onto its field, optionally
        # shorten the run via beta_sweep.epochs, point training at the fixed
        # dataset. ``beta`` flows through cfg["loss"]["kld_beta"] like every
        # other axis -- no train_minimal override key is needed (comment C18).
        cfg = deepcopy(config)
        section, field = _AXIS_FIELD[axis]
        cfg[section][field] = value
        epochs = (cfg.get("beta_sweep", {}) or {}).get("epochs")
        if epochs is not None:
            cfg["optim"]["epochs"] = int(epochs)
        print(f"  [train] {run_tag}  ({axis}={value:g})")
        tm.train(
            cfg, overrides={"data_tag": fixed_data_tag, "run_tag": run_tag}
        )

    row = ev.evaluate_checkpoint(
        ckpt_path, config, device=device, data_tag=fixed_data_tag
    )
    # ``beta`` for this cell: exact on the beta axis; the config default on an
    # HP axis (the HP probes hold beta fixed -- recorded for provenance).
    row["axis"] = axis
    row["value"] = float(value)
    row["beta"] = (
        float(value) if axis == "beta"
        else float(config["loss"]["kld_beta"])
    )
    return row


# =============================================================================
# Analysis (tasks 6.2, 6.4)
# =============================================================================

def _rate_distortion_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Rate-distortion analysis of the $\beta$ sweep (task 6.2).

    All $\beta$ cells share one $(a, M)$ -- hence one $\mathrm{TE}_{\rm true}$ --
    so this is a **closest-match heuristic**, not a rank test (the rank test is
    :func:`run_a_sweep_at_beta`, task 6.3 -- comment C19). It returns the
    $(\beta, \bar K, \mathrm{feat\_loss})$ curve and the $\beta$ whose $\bar K$
    lands nearest the analytic block / per-step TE, plus the lowest-distortion
    $\beta$.

    Args:
        rows: Per-cell rows (>= 1) from :func:`_run_one`.

    Returns:
        A nested analysis dict; ``selected_beta`` is the recommendation
        consumed by ``--mode rank_at_beta``.
    """
    betas = np.array([r["beta"] for r in rows], dtype=float)
    kbar = np.array([r["k_bar"] for r in rows], dtype=float)
    feat = np.array([r["feat_loss"] for r in rows], dtype=float)
    pred_gap = np.array([r["pred_gap"] for r in rows], dtype=float)
    te_true = float(rows[0]["te_true"])
    te_per_step = float(rows[0]["te_per_step"])

    order = np.argsort(betas)
    curve = [
        {
            "beta": float(betas[i]),
            "k_bar": float(kbar[i]),
            "feat_loss": float(feat[i]),
            "pred_gap": float(pred_gap[i]),
        }
        for i in order
    ]

    def _argmin_beta(values: np.ndarray) -> Optional[float]:
        """Beta minimising ``values`` over finite entries, or None."""
        finite = np.isfinite(values)
        if not finite.any():
            return None
        idx = int(np.argmin(np.where(finite, values, np.inf)))
        return float(betas[idx])

    beta_block = _argmin_beta(np.abs(kbar - te_true))
    beta_perstep = _argmin_beta(np.abs(kbar - te_per_step))
    beta_min_feat = _argmin_beta(feat)

    return {
        "axis": "beta",
        "n_cells": len(rows),
        "te_true": te_true,
        "te_per_step": te_per_step,
        "rate_distortion_curve": curve,
        "selected_beta": beta_block,
        "beta_closest_block_te": beta_block,
        "beta_closest_per_step_te": beta_perstep,
        "beta_min_feat_loss": beta_min_feat,
        "note": (
            "selected_beta minimises |K_bar - te_true| (block TE). All cells "
            "share one (a, M), so there is no rank correlation within this "
            "sweep -- the rank test is `rank_at_beta` (task 6.3)."
        ),
    }


def _hp_analysis(rows: List[Dict[str, Any]], axis: str) -> Dict[str, Any]:
    """Summarise a secondary hyper-parameter probe (task 6.4).

    Args:
        rows: Per-cell rows (>= 1) from :func:`_run_one`.
        axis: The HP axis (``lambda_base`` / ``d_z`` / ``warmup_period``).

    Returns:
        A nested analysis dict with the per-value curve and the value that
        maximises ``pred_gap`` (the source-residual predictive gain).
    """
    values = np.array([r["value"] for r in rows], dtype=float)
    kbar = np.array([r["k_bar"] for r in rows], dtype=float)
    pred_gap = np.array([r["pred_gap"] for r in rows], dtype=float)
    feat = np.array([r["feat_loss"] for r in rows], dtype=float)

    order = np.argsort(values)
    curve = [
        {
            "value": float(values[i]),
            "k_bar": float(kbar[i]),
            "pred_gap": float(pred_gap[i]),
            "feat_loss": float(feat[i]),
        }
        for i in order
    ]
    best_value: Optional[float] = None
    finite = np.isfinite(pred_gap)
    if finite.any():
        idx = int(np.argmax(np.where(finite, pred_gap, -np.inf)))
        best_value = float(values[idx])

    return {
        "axis": axis,
        "n_cells": len(rows),
        "beta": float(rows[0]["beta"]),
        "te_true": float(rows[0]["te_true"]),
        "curve": curve,
        "best_value_by_pred_gap": best_value,
        "note": (
            f"HP probe over '{axis}' at fixed beta={rows[0]['beta']:g}; "
            f"best_value_by_pred_gap maximises the source-residual gain."
        ),
    }


def _analyse(rows: List[Dict[str, Any]], axis: str) -> Dict[str, Any]:
    """Dispatch the per-axis analysis.

    Args:
        rows: Per-cell rows from :func:`_run_one`.
        axis: The sweep axis.

    Returns:
        The :func:`_rate_distortion_analysis` (beta axis) or
        :func:`_hp_analysis` (HP axis) output; a minimal stub when no cells
        were evaluated.
    """
    if not rows:
        return {"axis": axis, "n_cells": 0, "note": "no cells evaluated"}
    if axis == "beta":
        return _rate_distortion_analysis(rows)
    return _hp_analysis(rows, axis)


# =============================================================================
# Output: CSV / JSON / plots
# =============================================================================

def write_summary_csv(
    rows: List[Dict[str, Any]],
    path: Path,
    fields: Optional[List[str]] = None,
) -> None:
    """Write the per-cell summary CSV.

    Args:
        rows: Per-cell rows from :func:`_run_one`.
        path: Destination CSV path (overwritten).
        fields: Column list; defaults to :data:`_SUMMARY_FIELDS` (the 1-D sweep
            schema). ``beta_grid`` mode passes :data:`_BETA_GRID_FIELDS`.
    """
    fields = fields or _SUMMARY_FIELDS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_analysis_json(
    analysis: Dict[str, Any],
    rows: List[Dict[str, Any]],
    axis: str,
    path: Path,
    fields: Optional[List[str]] = None,
) -> None:
    """Write the rate-distortion / HP analysis plus per-cell rows as JSON.

    Args:
        analysis: The :func:`_analyse` output.
        rows: Per-cell rows (the ``per_dim_kl`` vectors live here).
        axis: The sweep axis.
        path: Destination JSON path (overwritten).
        fields: Per-row field list; defaults to :data:`_SUMMARY_FIELDS`.
            ``beta_grid`` mode passes :data:`_BETA_GRID_FIELDS`.
    """
    fields = fields or _SUMMARY_FIELDS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "axis": axis,
        "n_cells": len(rows),
        "analysis": analysis,
        "per_dim_kl": {r["run_tag"]: r.get("per_dim_kl", []) for r in rows},
        "rows": [{k: r.get(k) for k in fields} for r in rows],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _make_sweep_plots(
    rows: List[Dict[str, Any]],
    axis: str,
    analysis: Dict[str, Any],
    out_dir: Path,
) -> None:
    r"""Render the sweep plots (task 6.2 for ``beta``, 6.4 for an HP axis).

    ``beta`` axis: ``kbar_vs_beta.{pdf,png}`` (log-x $\beta$ with the
    analytic-TE reference line and the selected $\beta$),
    ``ratedist_feat_vs_kbar.{pdf,png}`` (the rate-distortion curve) and
    ``predgap_vs_beta.{pdf,png}``. HP axis: ``kbar_vs_<axis>.{pdf,png}`` and
    ``predgap_vs_<axis>.{pdf,png}``. All figures use the shared publication
    style in :mod:`plot_style`.

    Args:
        rows: Per-cell rows (>= 2 expected).
        axis: The sweep axis.
        analysis: The :func:`_analyse` output.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()

    value = np.array([r["value"] for r in rows], dtype=float)
    kbar = np.array([r["k_bar"] for r in rows], dtype=float)
    feat = np.array([r["feat_loss"] for r in rows], dtype=float)
    pred_gap = np.array([r["pred_gap"] for r in rows], dtype=float)
    order = np.argsort(value)
    value, kbar, feat, pred_gap = (
        value[order], kbar[order], feat[order], pred_gap[order]
    )

    if axis == "beta":
        te_true = float(rows[0]["te_true"])

        # --- Plot 1: K_bar vs beta (log-x) + analytic-TE / selected-beta ----
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        ax.plot(value, kbar, marker="o", color=ps.COLOR_BLUE,
                label=r"$\bar K$")
        ax.axhline(
            te_true, color=ps.COLOR_GRAY, ls="--", lw=1.0,
            label=fr"analytic block TE = {te_true:.3g}",
        )
        sel = analysis.get("selected_beta")
        if sel is not None and np.isfinite(sel):
            ax.axvline(
                sel, color=ps.COLOR_VERMILLION, ls=":", lw=1.4,
                label=fr"selected $\beta$ = {sel:g}",
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"bottleneck coefficient $\beta$")
        ax.set_ylabel(r"$\bar K$ (nats)")
        ax.set_title(r"$\bar K$ vs $\beta$")
        ax.legend()
        ps.style_axes(ax)
        fig.tight_layout()
        ps.save_figure(fig, out_dir / "kbar_vs_beta")

        # --- Plot 2: rate-distortion -- feat_loss vs K_bar ------------------
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        ax.plot(kbar, feat, marker="o", color=ps.COLOR_BLUE)
        for xi, yi, bi in zip(kbar, feat, value):
            ax.annotate(
                fr"$\beta$={bi:g}", (xi, yi), fontsize=6,
                color=ps.COLOR_GRAY,
                textcoords="offset points", xytext=(4, 4),
            )
        ax.set_xlabel(r"$\bar K$ (nats)")
        ax.set_ylabel("feat_loss (weighted MSE)")
        ax.set_title("rate-distortion curve")
        ps.style_axes(ax)
        fig.tight_layout()
        ps.save_figure(fig, out_dir / "ratedist_feat_vs_kbar")

        # --- Plot 3: predictive gain vs beta --------------------------------
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        ax.axhline(0.0, color=ps.COLOR_GRAY, ls="--", lw=0.9)
        ax.plot(value, pred_gap, marker="o", color=ps.COLOR_BLUE)
        ax.set_xscale("log")
        ax.set_xlabel(r"bottleneck coefficient $\beta$")
        ax.set_ylabel(
            r"pred_gap $= \mathcal{L}_{\rm base}-\mathcal{L}_{\rm feat}$"
        )
        ax.set_title(r"predictive gain vs $\beta$")
        ps.style_axes(ax)
        fig.tight_layout()
        ps.save_figure(fig, out_dir / "predgap_vs_beta")
        return

    # --- HP axis: K_bar vs value, pred_gap vs value -------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(value, kbar, marker="o", color=ps.COLOR_BLUE)
    ax.set_xlabel(axis)
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.set_title(fr"$\bar K$ vs {axis}")
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / f"kbar_vs_{axis}")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0.0, color=ps.COLOR_GRAY, ls="--", lw=0.9)
    ax.plot(value, pred_gap, marker="o", color=ps.COLOR_BLUE)
    ax.set_xlabel(axis)
    ax.set_ylabel(r"pred_gap $= \mathcal{L}_{\rm base}-\mathcal{L}_{\rm feat}$")
    ax.set_title(fr"predictive gain vs {axis}")
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / f"predgap_vs_{axis}")


def _print_analysis(analysis: Dict[str, Any], axis: str, n: int) -> None:
    """Print the sweep analysis block.

    Args:
        analysis: The :func:`_analyse` output.
        axis: The sweep axis.
        n: Number of evaluated cells.
    """
    if n == 0:
        print(f"\n[analysis] axis='{axis}': no cells evaluated.")
        return
    if axis == "beta":
        print(
            f"\n[analysis] beta sweep: {n} cell(s)\n"
            f"  te_true={analysis['te_true']:.4f} nats   "
            f"te_per_step={analysis['te_per_step']:.4f}\n"
            f"  selected beta (|K_bar - block TE| min) : "
            f"{analysis['selected_beta']}\n"
            f"  beta closest to per-step TE            : "
            f"{analysis['beta_closest_per_step_te']}\n"
            f"  beta with lowest feat_loss             : "
            f"{analysis['beta_min_feat_loss']}"
        )
    else:
        print(
            f"\n[analysis] {axis} probe: {n} cell(s)  "
            f"beta={analysis['beta']:g}\n"
            f"  best {axis} by pred_gap: {analysis['best_value_by_pred_gap']}"
        )


# =============================================================================
# Mode: beta sweep / HP probe (tasks 6.1, 6.2, 6.4)
# =============================================================================

def run_sweep(
    config: Dict[str, Any],
    *,
    axis: str = "beta",
    build_missing: bool = False,
    train_missing: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Run a 1-D hyper-parameter sweep and aggregate the results.

    ``axis="beta"`` runs the rate-distortion $\beta$ sweep (tasks 6.1, 6.2);
    ``axis`` in :data:`_HP_AXES` runs a secondary HP probe (task 6.4). Every
    cell trains the same fixed $(a, M)$ dataset, so the comparison is valid.
    Missing checkpoints are **skipped** unless ``train_missing`` is set.

    Args:
        config: The parsed ``config_synth.yaml``.
        axis: ``beta`` or one of :data:`_HP_AXES`.
        build_missing: If True, generate the fixed dataset when it is missing.
        train_missing: If True, train any missing cell checkpoint (multi-hour).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.

    Returns:
        A results dict: ``axis``, ``rows``, ``analysis``, ``out_dir``,
        ``skipped``.

    Raises:
        ValueError: On an unknown ``axis`` or an empty value grid.
        FileNotFoundError: If the fixed dataset is missing and ``build_missing``
            is False.
    """
    axis = str(axis)
    if axis not in _AXIS_FIELD:
        raise ValueError(
            f"unknown sweep axis: {axis!r} (expected 'beta' or one of "
            f"{_HP_AXES})."
        )
    benchmark = str(config["experiment"]["benchmark"])
    device = device or tm.resolve_device(config["runtime"])
    bs = config.get("beta_sweep", {}) or {}

    if axis == "beta":
        grid = [float(v) for v in (bs.get("grid") or [])]
    else:
        grid = list((bs.get("hp_probes", {}) or {}).get(axis, []) or [])
    if not grid:
        loc = "grid" if axis == "beta" else f"hp_probes.{axis}"
        raise ValueError(
            f"empty sweep grid for axis '{axis}' -- populate beta_sweep.{loc}."
        )

    fixed_data_tag = _fixed_data_tag(config)
    _ensure_fixed_dataset(config, fixed_data_tag, build_missing=build_missing)

    sweep_dir_tag = "beta_sweep" if axis == "beta" else f"hp_{axis}"
    out_dir = ev._results_root(config) / benchmark / sweep_dir_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[sweep] axis='{axis}'  benchmark={benchmark}  {len(grid)} value(s)  "
        f"data_tag='{fixed_data_tag}'  device={device}  "
        f"build_missing={build_missing}  train_missing={train_missing}"
    )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for value in grid:
        try:
            row = _run_one(
                config, axis=axis, value=value,
                fixed_data_tag=fixed_data_tag, sweep_dir_tag=sweep_dir_tag,
                build_missing=build_missing, train_missing=train_missing,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001 - one bad cell must not abort
            print(f"  [error] {axis}={value:g}: {type(exc).__name__}: {exc}")
            skipped.append(f"{axis}={value:g}")
            continue
        if row is None:
            _, run_tag = _cell_tags(axis, value, sweep_dir_tag)
            skipped.append(run_tag)
            continue
        rows.append(row)

    analysis = _analyse(rows, axis)
    write_summary_csv(rows, out_dir / "summary.csv")
    write_analysis_json(analysis, rows, axis, out_dir / "analysis.json")
    if len(rows) >= 2:
        _make_sweep_plots(rows, axis, analysis, out_dir)
    else:
        print(
            "[plots] fewer than 2 cells evaluated -- skipping plots "
            "(the sweep training run is deferred; see the plan)."
        )
    _print_analysis(analysis, axis, n=len(rows))
    print(
        f"[done] {axis} sweep: {len(rows)} evaluated, {len(skipped)} skipped\n"
        f"       artifacts -> {out_dir}"
    )
    return {
        "axis": axis,
        "rows": rows,
        "analysis": analysis,
        "out_dir": str(out_dir),
        "skipped": skipped,
    }


# =============================================================================
# Mode: rank correlation re-run at a chosen beta (task 6.3)
# =============================================================================

def run_a_sweep_at_beta(
    config: Dict[str, Any],
    *,
    beta: float,
    build_missing: bool = False,
    train_missing: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Re-run the Phase-4 A-sweep at a chosen $\beta$ (task 6.3).

    Trains the ``a_sweep`` cells with ``loss.kld_beta = beta``,
    $\beta$-namespacing every ``run_tag`` (``rank_at_beta/<beta>/sweep_a..._m...``)
    and the output directory (``results/<bench>/rank_at_beta/<beta>/``) so the
    Phase-4 default-$\beta$ A-sweep artifacts are never clobbered (comment C20).
    The cell **datasets** are $\beta$-independent and shared with Phase 4.
    Enumeration, the Metric 1-4 aggregation and the plots reuse
    :mod:`evaluate_te` verbatim.

    Args:
        config: The parsed ``config_synth.yaml``.
        beta: The $\beta$ to re-run the A-sweep at (typically the
            ``selected_beta`` from a prior :func:`run_sweep`).
        build_missing: If True, generate any missing sweep dataset.
        train_missing: If True, train any missing cell checkpoint (multi-hour).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.

    Returns:
        A results dict: ``beta``, ``rows``, ``metrics``, ``out_dir``,
        ``skipped``.
    """
    beta = float(beta)
    benchmark = str(config["experiment"]["benchmark"])
    device = device or tm.resolve_device(config["runtime"])
    beta_token = f"b{_fmt_token(beta)}"
    out_dir = ev._results_root(config) / benchmark / "rank_at_beta" / beta_token
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = (config.get("beta_sweep", {}) or {}).get("epochs")

    settings = ev.enumerate_sweep(config)
    print(
        f"[rank_at_beta] benchmark {benchmark}: {len(settings)} (a, M) cell(s) "
        f"at beta={beta:g}  device={device}  build_missing={build_missing}  "
        f"train_missing={train_missing}"
    )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for setting in settings:
        m = setting["M"]
        kind = str(setting.get("kind", ""))
        # Each v2 sweep cell carries a single named knob; mirror the
        # `run_sweep` dispatch in evaluate_te.py.
        if kind == "gaussian_state_space":
            value = float(setting["B_y"])
            data_tag, base_run_tag = ev._setting_tags_knob(
                benchmark, "By", value, m
            )
            label = f"B_y={value:g}, M={m}"
            ensure_dataset = lambda: ev._ensure_dataset_state_space(
                config, data_tag, value, m
            )
        elif kind == "arx":
            value = float(setting["c"])
            data_tag, base_run_tag = ev._setting_tags_knob(
                benchmark, "c", value, m
            )
            label = f"c={value:g}, M={m}"
            ensure_dataset = lambda: ev._ensure_dataset_arx(
                config, data_tag, value, m
            )
        elif kind == "regime_switch":
            value = float(setting["p_switch"])
            data_tag, base_run_tag = ev._setting_tags_knob(
                benchmark, "p", value, m
            )
            label = f"p={value:g}, M={m}"
            ensure_dataset = lambda: ev._ensure_dataset_regime_switch(
                config, data_tag, value, m
            )
        else:
            print(f"  [skip ] unknown sweep kind {kind!r}")
            skipped.append(f"unknown_{kind}")
            continue
        run_tag = f"rank_at_beta/{beta_token}/{base_run_tag}"
        cache_dir = ev._data_root(config) / benchmark / data_tag
        ckpt_path = ev._results_root(config) / benchmark / run_tag / "final.ckpt"

        if not (cache_dir / "test.npz").is_file():
            if build_missing:
                print(f"  [build] {data_tag}  ({label})")
                ensure_dataset()
            else:
                print(
                    f"  [skip ] {run_tag}: dataset '{data_tag}' not cached "
                    f"(pass build_missing=True to generate it)."
                )
                skipped.append(run_tag)
                continue

        if not ckpt_path.is_file():
            if train_missing:
                print(f"  [train] {run_tag}  ({label}, beta={beta:g})")
                cfg = deepcopy(config)
                cfg["loss"]["kld_beta"] = beta
                if epochs is not None:
                    cfg["optim"]["epochs"] = int(epochs)
                tm.train(
                    cfg, overrides={"data_tag": data_tag, "run_tag": run_tag}
                )
            else:
                print(
                    f"  [skip ] {run_tag}: checkpoint not found ({ckpt_path}) "
                    f"(pass train_missing=True to train it)."
                )
                skipped.append(run_tag)
                continue

        try:
            row = ev.evaluate_checkpoint(
                ckpt_path, config, device=device, data_tag=data_tag
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not abort
            print(f"  [error] {run_tag}: {type(exc).__name__}: {exc}")
            skipped.append(run_tag)
            continue

    metrics = ev._aggregate_metrics(rows)
    metrics["n_skipped"] = len(skipped)
    metrics["beta"] = beta
    ev.write_summary_csv(rows, out_dir / "summary.csv")
    ev.write_metrics_json(metrics, rows, out_dir / "metrics.json")
    if len(rows) >= 2:
        ev._make_plots(rows, metrics, out_dir)
    else:
        print(
            "[plots] fewer than 2 cells evaluated -- skipping plots "
            "(the A-sweep training run is deferred; see the plan)."
        )
    _print_metrics_safe(metrics, n=len(rows))
    print(
        f"[done] rank_at_beta (beta={beta:g}): {len(rows)} evaluated, "
        f"{len(skipped)} skipped  Spearman rho={metrics.get('metric2_spearman')}"
        f"\n       artifacts -> {out_dir}"
    )
    return {
        "beta": beta,
        "rows": rows,
        "metrics": metrics,
        "out_dir": str(out_dir),
        "skipped": skipped,
    }


def _print_metrics_safe(metrics: Dict[str, Any], n: int) -> None:
    """Print the Metric 1-4 block, tolerating an all-skipped sweep.

    Args:
        metrics: The :func:`evaluate_te._aggregate_metrics` output.
        n: Number of evaluated cells.
    """
    if n == 0:
        print("\n[metrics] 0 cells evaluated -- nothing to aggregate.")
        return
    ev._print_metrics(metrics, n=n)


# =============================================================================
# Mode: beta x M x TE grid (multi-line K_bar-vs-beta across channels / TE)
# =============================================================================

def _knob_for_setting(setting: Dict[str, Any]) -> Tuple[str, str, float]:
    """Resolve the generator knob a v2 sweep setting varies.

    Args:
        setting: One :func:`evaluate_te.enumerate_sweep` setting dict.

    Returns:
        Tuple ``(knob_token, knob_field, value)`` -- ``knob_token`` is the short
        tag label (``"By"`` / ``"c"`` / ``"p"``) consumed by
        :func:`evaluate_te._setting_tags_knob`; ``knob_field`` is the CSV column
        (``"B_y"`` / ``"c"`` / ``"p_switch"``); ``value`` is the solved knob.

    Raises:
        ValueError: If ``setting["kind"]`` is not a known v2 dispatch.
    """
    kind = str(setting.get("kind", ""))
    if kind == "gaussian_state_space":
        return "By", "B_y", float(setting["B_y"])
    if kind == "arx":
        return "c", "c", float(setting["c"])
    if kind == "regime_switch":
        return "p", "p_switch", float(setting["p_switch"])
    raise ValueError(f"unknown sweep kind {kind!r}")


def _ensure_dataset_for_setting(
    config: Dict[str, Any], kind: str, data_tag: str, value: float, m: int
) -> None:
    """Materialise the (beta-independent) dataset for one sweep setting.

    Dispatches to the kind-specific :mod:`evaluate_te` dataset builder so both
    :func:`run_beta_grid` and :func:`gpu_pool._cells_beta_grid` share one
    mapping from ``setting["kind"]`` to its idempotent ``_ensure_dataset_*``.

    Args:
        config: The parsed config (copied before override by the builder).
        kind: The sweep ``kind`` (``gaussian_state_space`` / ``arx`` /
            ``regime_switch``).
        data_tag: Cache tag for this cell.
        value: The solved knob ($B_y$ / $c$ / $p_{\\rm switch}$).
        m: Number of informative channels.

    Raises:
        ValueError: If ``kind`` is not a known v2 dispatch.
    """
    if kind == "gaussian_state_space":
        ev._ensure_dataset_state_space(config, data_tag, value, m)
    elif kind == "arx":
        ev._ensure_dataset_arx(config, data_tag, value, m)
    elif kind == "regime_switch":
        ev._ensure_dataset_regime_switch(config, data_tag, value, m)
    else:
        raise ValueError(f"unknown sweep kind {kind!r}")


def _enumerate_beta_grid_settings(
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    r"""Enumerate the $(M, \mathrm{TE})$ settings for the $\beta$ grid.

    Reuses :func:`evaluate_te.enumerate_sweep` on a config whose ``sweep``
    block is optionally narrowed by ``beta_sweep.beta_grid.{m_grid,
    target_te_grid}`` -- so the 270-cell default grid can be shrunk from the
    config without touching the active benchmark's ``sweep`` block.

    Args:
        config: The parsed, benchmark-resolved config.

    Returns:
        The :func:`evaluate_te.enumerate_sweep` settings list (one per
        $(M, \mathrm{TE})$ cell), each crossed with every $\beta$ downstream.
    """
    override = (config.get("beta_sweep", {}) or {}).get("beta_grid", {}) or {}
    cfg_enum = deepcopy(config)
    cfg_enum.setdefault("sweep", {})
    if override.get("m_grid") is not None:
        cfg_enum["sweep"]["m_grid"] = [int(m) for m in override["m_grid"]]
    if override.get("target_te_grid") is not None:
        cfg_enum["sweep"]["target_te_grid"] = [
            float(t) for t in override["target_te_grid"]
        ]
    return ev.enumerate_sweep(cfg_enum)


def run_beta_grid(
    config: Dict[str, Any],
    *,
    build_missing: bool = False,
    train_missing: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Sweep $\beta$ across the full $(M, \mathrm{TE})$ grid.

    Crosses every $(M, \mathrm{TE})$ cell of
    :func:`evaluate_te.enumerate_sweep` (the coupling solved per cell, with the
    analytic ``te_true``) with ``beta_sweep.grid``. The cell **datasets** are
    $\beta$-independent and identical to the Phase-4 A-sweep caches
    (``benchmark_<B>_sweep_<knob>_m<M>``), so each is built once and reused for
    every $\beta$; only the ``loss.kld_beta`` patch and the
    $\beta$-namespaced ``run_tag`` (``beta_grid/<base>/b<token>``) differ across
    the inner loop. Training, scoring and dataset materialisation reuse
    :mod:`train_minimal` / :mod:`evaluate_te` verbatim.

    Args:
        config: The parsed ``config_synth.yaml``.
        build_missing: If True, generate any missing $(M, \mathrm{TE})$ dataset.
        train_missing: If True, train any missing cell checkpoint (multi-hour;
            270 cells at the G1 default).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.

    Returns:
        A results dict: ``axis`` (``"beta_grid"``), ``rows``, ``analysis``,
        ``out_dir`` and ``skipped``.

    Raises:
        ValueError: If ``beta_sweep.grid`` is empty.
    """
    benchmark = str(config["experiment"]["benchmark"])
    device = device or tm.resolve_device(config["runtime"])
    bs = config.get("beta_sweep", {}) or {}
    beta_grid = [float(v) for v in (bs.get("grid") or [])]
    if not beta_grid:
        raise ValueError("beta_grid mode needs a populated beta_sweep.grid.")
    epochs = bs.get("epochs")

    settings = _enumerate_beta_grid_settings(config)
    out_dir = ev._results_root(config) / benchmark / "beta_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[beta_grid] benchmark={benchmark}  {len(settings)} (M, TE) cell(s) x "
        f"{len(beta_grid)} beta = {len(settings) * len(beta_grid)} run(s)  "
        f"device={device}  build_missing={build_missing}  "
        f"train_missing={train_missing}"
    )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for setting in settings:
        m = int(setting["M"])
        try:
            knob_token, knob_field, value = _knob_for_setting(setting)
        except ValueError as exc:
            print(f"  [skip ] {exc}")
            skipped.append(f"unknown_kind_m{m}")
            continue
        data_tag, base_run_tag = ev._setting_tags_knob(
            benchmark, knob_token, value, m
        )
        cache_dir = ev._data_root(config) / benchmark / data_tag

        # Ensure the (beta-independent) cell dataset once, then reuse it.
        if not (cache_dir / "test.npz").is_file():
            if build_missing:
                print(f"  [build] {data_tag}  ({knob_field}={value:g}, M={m})")
                _ensure_dataset_for_setting(
                    config, str(setting["kind"]), data_tag, value, m
                )
            else:
                for beta in beta_grid:
                    skipped.append(
                        f"beta_grid/{base_run_tag}/b{_fmt_token(beta)}"
                    )
                print(
                    f"  [skip ] dataset '{data_tag}' not cached "
                    f"(pass build_missing=True); skipped its {len(beta_grid)} "
                    f"beta cells."
                )
                continue

        target_te = setting.get("target_te")
        for beta in beta_grid:
            run_tag = f"beta_grid/{base_run_tag}/b{_fmt_token(beta)}"
            ckpt_path = (
                ev._results_root(config) / benchmark / run_tag / "final.ckpt"
            )
            if not ckpt_path.is_file():
                if not train_missing:
                    print(f"  [skip ] {run_tag}: checkpoint not found.")
                    skipped.append(run_tag)
                    continue
                cfg = deepcopy(config)
                cfg["loss"]["kld_beta"] = beta
                if epochs is not None:
                    cfg["optim"]["epochs"] = int(epochs)
                print(
                    f"  [train] {run_tag}  ({knob_field}={value:g}, M={m}, "
                    f"beta={beta:g})"
                )
                tm.train(
                    cfg, overrides={"data_tag": data_tag, "run_tag": run_tag}
                )

            try:
                row = ev.evaluate_checkpoint(
                    ckpt_path, config, device=device, data_tag=data_tag
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell mustn't abort
                print(f"  [error] {run_tag}: {type(exc).__name__}: {exc}")
                skipped.append(run_tag)
                continue

            # Stamp robust grid coordinates from the *setting* (the faceting
            # keys). ``te_true`` is left as evaluate_checkpoint set it -- the
            # realised, histogram-weighted ``meta.te_true`` of the cached
            # dataset, which is the correct calibration x-axis (v2 sec. 3.1.1),
            # not the nominal enumeration target (kept separately as
            # ``target_te`` for clean panel grouping).
            row["axis"] = "beta_grid"
            row["value"] = float(beta)
            row["beta"] = float(beta)
            row["M"] = m
            row["target_te"] = (
                float(target_te) if target_te is not None else None
            )
            row[knob_field] = value
            rows.append(row)

    analysis = _beta_grid_analysis(rows)
    write_summary_csv(rows, out_dir / "summary.csv", fields=_BETA_GRID_FIELDS)
    write_analysis_json(
        analysis, rows, "beta_grid", out_dir / "analysis.json",
        fields=_BETA_GRID_FIELDS,
    )
    if len(rows) >= 2:
        _make_beta_grid_plots(rows, out_dir)
    else:
        print(
            "[plots] fewer than 2 cells evaluated -- skipping plots "
            "(the grid training run is deferred; see the plan)."
        )
    print(
        f"[done] beta_grid: {len(rows)} evaluated, {len(skipped)} skipped\n"
        f"       artifacts -> {out_dir}"
    )
    return {
        "axis": "beta_grid",
        "rows": rows,
        "analysis": analysis,
        "out_dir": str(out_dir),
        "skipped": skipped,
    }


def _beta_grid_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Per-$(M, \mathrm{TE})$ rate-distortion summary of the $\beta$ grid.

    For each $(M, \mathrm{TE})$ group the $\bar K(\beta)$ curve is recorded and
    the $\beta$ whose $\bar K$ lands nearest the analytic ``te_true`` is picked
    (a closest-match heuristic per group, mirroring
    :func:`_rate_distortion_analysis`).

    Args:
        rows: Per-cell rows from :func:`run_beta_grid`.

    Returns:
        A nested analysis dict: ``groups`` is one entry per $(M, \mathrm{TE})$
        with its curve and ``selected_beta``.
    """
    if not rows:
        return {"axis": "beta_grid", "n_cells": 0, "note": "no cells evaluated"}

    groups: Dict[Tuple[int, Optional[float]], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (int(r["M"]), r.get("target_te"))
        groups.setdefault(key, []).append(r)

    def _sort_key(item: Tuple[Tuple[int, Optional[float]], Any]) -> Tuple[int, float]:
        (m, tt), _ = item
        return (m, float(tt) if tt is not None else float("inf"))

    cells: List[Dict[str, Any]] = []
    for (m, tt), grp in sorted(groups.items(), key=_sort_key):
        betas = np.array([float(g["beta"]) for g in grp], dtype=float)
        kbar = np.array([float(g["k_bar"]) for g in grp], dtype=float)
        te_true = float(grp[0]["te_true"])
        order = np.argsort(betas)
        curve = [
            {"beta": float(betas[i]), "k_bar": float(kbar[i])} for i in order
        ]
        diff = np.abs(kbar - te_true)
        finite = np.isfinite(diff)
        sel = (
            float(betas[int(np.argmin(np.where(finite, diff, np.inf)))])
            if finite.any() else None
        )
        cells.append({
            "M": m, "target_te": tt, "te_true": te_true,
            "selected_beta": sel, "curve": curve,
        })

    return {
        "axis": "beta_grid",
        "n_cells": len(rows),
        "n_groups": len(cells),
        "m_levels": sorted({int(r["M"]) for r in rows}),
        "te_levels": sorted(
            {float(r["target_te"]) for r in rows
             if r.get("target_te") is not None}
        ),
        "beta_grid": sorted({float(r["beta"]) for r in rows}),
        "groups": cells,
        "note": (
            "per-(M, target_te) beta curve; selected_beta minimises "
            "|K_bar - te_true| within each group (closest-match heuristic)."
        ),
    }


def _make_beta_grid_plots(
    rows: List[Dict[str, Any]], out_dir: Path
) -> None:
    r"""Render the three multi-line $\beta$-grid figures.

    * ``kbar_vs_beta__byTE.{pdf,png}`` -- one panel per target-TE level; within
      a panel $\bar K$ vs $\beta$ (log-x) with one coloured line per $M$ and a
      dashed line at the realised ``te_true``.
    * ``kbar_vs_beta__byM.{pdf,png}`` -- one panel per $M$; one coloured line
      per target-TE level.
    * ``kbar_vs_te__byBeta.{pdf,png}`` -- one panel per $\beta$; $\bar K$ vs the
      realised ``te_true`` scattered and coloured by $M$ (the per-$\beta$
      calibration slice). No identity line is drawn: under the default ``mse``
      likelihood $\bar K$ is scale-free, so only the monotone trend is meaningful.

    Args:
        rows: Per-cell rows (>= 2) from :func:`run_beta_grid`.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()

    def _color(idx: int) -> str:
        return ps.PALETTE_EXTENDED[idx % len(ps.PALETTE_EXTENDED)]

    def _te_key(r: Dict[str, Any]) -> float:
        """Faceting key: the nominal target TE, or the realised TE if absent."""
        return (
            float(r["target_te"]) if r.get("target_te") is not None
            else round(float(r["te_true"]), 4)
        )

    betas = np.array([float(r["beta"]) for r in rows], dtype=float)
    kbar = np.array([float(r["k_bar"]) for r in rows], dtype=float)
    te_true = np.array([float(r["te_true"]) for r in rows], dtype=float)
    m_vals = np.array([int(r["M"]) for r in rows], dtype=int)
    te_keys = np.array([_te_key(r) for r in rows], dtype=float)

    m_levels = sorted(set(m_vals.tolist()))
    te_levels = sorted(set(te_keys.tolist()))
    beta_levels = sorted(set(betas.tolist()))

    def _new_grid(n: int, *, sharey: bool):
        """A near-square constrained subplot grid for ``n`` panels.

        ``sharex`` is always on (every panel shares the $\\beta$ / TE axis);
        spare axes are switched off. ``layout="constrained"`` reserves room for
        the suptitle and an out-of-panel legend so neither overlaps the data.
        Returns ``(fig, flat_axes, ncols)``.
        """
        ncols = int(np.ceil(np.sqrt(max(n, 1))))
        nrows = int(np.ceil(max(n, 1) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.9 * ncols, 3.2 * nrows), squeeze=False,
            sharex=True, sharey=sharey, layout="constrained",
        )
        flat = axes.ravel()
        for j in range(n, len(flat)):
            flat[j].axis("off")
        return fig, flat, ncols

    def _dedupe_axis_labels(flat, n, ncols, xlabel, ylabel):
        """Keep the $x$ label only on each column's lowest panel, $y$ on column 0.

        Ragged-grid safe (a partly-filled last row): a panel is the lowest in
        its column iff no used panel sits ``ncols`` slots later. ``sharex`` hides
        x tick numbers on non-bottom *rows*, which would also strip them from a
        column's lowest panel when the slot below is an (empty) spare -- so tick
        numbers are forced back on there. Per-panel $y$ tick numbers are left as
        ``sharey`` set them (off-grid panels may differ in scale).
        """
        for i, ax in enumerate(flat[:n]):
            is_col_bottom = i + ncols >= n
            ax.set_xlabel(xlabel if is_col_bottom else "")
            ax.set_ylabel(ylabel if i % ncols == 0 else "")
            if is_col_bottom:
                ax.tick_params(axis="x", labelbottom=True)

    def _facet_kbar_vs_beta(
        fname, *, panel_vals, panel_levels, panel_title, series_vals,
        series_levels, series_label, suptitle, draw_ref,
    ):
        """K_bar-vs-beta faceted by ``panel_vals``, one line per ``series_vals``."""
        fig, flat, ncols = _new_grid(len(panel_levels), sharey=False)
        for pi, pv in enumerate(panel_levels):
            ax = flat[pi]
            pmask = panel_vals == pv
            for idx, sv in enumerate(series_levels):
                sel = pmask & (series_vals == sv)
                if not sel.any():
                    continue
                order = np.argsort(betas[sel])
                ax.plot(betas[sel][order], kbar[sel][order], marker="o",
                        color=_color(idx), label=series_label(sv))
            if draw_ref and pmask.any():
                ref = float(np.nanmean(te_true[pmask]))
                if np.isfinite(ref):
                    ax.axhline(ref, color=ps.COLOR_GRAY, ls="--", lw=1.0)
            ax.set_xscale("log")
            ax.set_title(panel_title(pv))
            ps.style_axes(ax)
        _dedupe_axis_labels(
            flat, len(panel_levels), ncols,
            r"bottleneck coefficient $\beta$", r"$\bar K$ (nats)",
        )
        handles = [
            Line2D([0], [0], color=_color(i), marker="o", label=series_label(sv))
            for i, sv in enumerate(series_levels)
        ]
        if draw_ref:
            handles.append(Line2D(
                [0], [0], color=ps.COLOR_GRAY, ls="--",
                label=r"realised $\mathrm{TE}_{\rm true}$",
            ))
        fig.legend(handles=handles, loc="outside right upper")
        fig.suptitle(suptitle)
        ps.save_figure(fig, out_dir / fname)

    # --- Figures 1 & 2: K_bar vs beta, faceted by TE / by M ---------------
    _facet_kbar_vs_beta(
        "kbar_vs_beta__byTE",
        panel_vals=te_keys, panel_levels=te_levels,
        panel_title=lambda te: fr"target TE = {te:g} nats",
        series_vals=m_vals, series_levels=m_levels,
        series_label=lambda m: f"M={m}",
        suptitle=r"$\bar K$ vs $\beta$  (panel: target TE, line: $M$)",
        draw_ref=True,
    )
    _facet_kbar_vs_beta(
        "kbar_vs_beta__byM",
        panel_vals=m_vals, panel_levels=m_levels,
        panel_title=lambda m: fr"$M$ = {m}",
        series_vals=te_keys, series_levels=te_levels,
        series_label=lambda te: fr"TE={te:g}",
        suptitle=r"$\bar K$ vs $\beta$  (panel: $M$, line: target TE)",
        draw_ref=False,
    )

    # --- Figure 3: calibration -- K_bar vs realised TE, panel per beta ----
    # x (true TE) and y (K_bar) share scales across beta panels, so sharey=True
    # makes the per-beta calibration directly comparable.
    fig, flat, ncols = _new_grid(len(beta_levels), sharey=True)
    m_handles = [
        Line2D([0], [0], color=_color(i), marker="o", linestyle="none", label=f"M={m}")
        for i, m in enumerate(m_levels)
    ]
    for pi, beta in enumerate(beta_levels):
        ax = flat[pi]
        panel = betas == beta
        for idx, m in enumerate(m_levels):
            sel = panel & (m_vals == m)
            if not sel.any():
                continue
            ax.scatter(te_true[sel], kbar[sel], s=40, color=_color(idx),
                       edgecolors=ps.COLOR_BLACK, linewidths=0.4, zorder=3)
        ax.set_title(fr"$\beta$ = {beta:g}")
        ps.style_axes(ax)
    _dedupe_axis_labels(
        flat, len(beta_levels), ncols,
        "analytic block TE (nats)", r"$\bar K$ (nats)",
    )
    fig.legend(handles=m_handles, loc="outside right upper")
    fig.suptitle(r"calibration: $\bar K$ vs true TE  (panel: $\beta$, colour: $M$)")
    ps.save_figure(fig, out_dir / "kbar_vs_te__byBeta")


# =============================================================================
# Overrides + dispatch
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the config-level overrides onto ``config`` in place.

    Only ``benchmark`` / ``device`` / ``seed`` are config fields; the other
    overrides (``mode``, ``axis``, ``beta``, ``build_missing``,
    ``train_missing``) are call arguments handled by :func:`_dispatch`.

    Args:
        config: The config dict (mutated in place).
        overrides: Flat ``{key: value}`` overrides; ``None`` values ignored.

    Returns:
        The same ``config`` dict.
    """
    if overrides.get("benchmark") is not None:
        config["experiment"]["benchmark"] = overrides["benchmark"]
        # Re-overlay data / sweep for the newly selected benchmark.
        tm.resolve_active_benchmark(config)
    if overrides.get("device") is not None:
        config["runtime"]["device"] = overrides["device"]
    if overrides.get("seed") is not None:
        config["experiment"]["seed"] = overrides["seed"]
    return config


def _dispatch(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve overrides, seed, device and run the requested mode.

    Args:
        config: The parsed ``config_synth.yaml``.
        overrides: Flat overrides (from ``vars(args)`` or ``RUN_CONFIG``).

    Returns:
        The mode-specific results dict (see :func:`run_sweep` /
        :func:`run_a_sweep_at_beta`).

    Raises:
        ValueError: On an unknown ``mode``, a missing ``axis`` in ``hp`` mode,
            or a missing ``beta`` in ``rank_at_beta`` mode.
    """
    config = deepcopy(config)
    _apply_overrides(config, overrides)
    tm.set_seed(int(config["experiment"].get("seed", 0)))
    device = tm.resolve_device(config["runtime"])
    mode = str(overrides.get("mode") or "beta").lower()
    build_missing = bool(overrides.get("build_missing"))
    train_missing = bool(overrides.get("train_missing"))

    if mode == "beta":
        return run_sweep(
            config, axis="beta", build_missing=build_missing,
            train_missing=train_missing, device=device,
        )
    if mode == "beta_grid":
        return run_beta_grid(
            config, build_missing=build_missing,
            train_missing=train_missing, device=device,
        )
    if mode == "hp":
        axis = overrides.get("axis")
        if axis not in _HP_AXES:
            raise ValueError(
                f"hp mode requires --axis one of {_HP_AXES}; got {axis!r}."
            )
        return run_sweep(
            config, axis=str(axis), build_missing=build_missing,
            train_missing=train_missing, device=device,
        )
    if mode == "rank_at_beta":
        beta = overrides.get("beta")
        if beta is None:
            beta = (config.get("beta_sweep", {}) or {}).get("selected_beta")
        if beta is None:
            raise ValueError(
                "rank_at_beta mode requires a beta -- pass --beta B (CLI) or "
                "RUN_CONFIG['beta'], or set beta_sweep.selected_beta in "
                "config_synth.yaml."
            )
        return run_a_sweep_at_beta(
            config, beta=float(beta), build_missing=build_missing,
            train_missing=train_missing, device=device,
        )
    raise ValueError(
        f"unknown mode: {mode!r} (expected 'beta', 'beta_grid', 'hp' or "
        f"'rank_at_beta')."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        The parsed :class:`argparse.Namespace`. Config-routed flags default to
        ``None`` (fall back to ``config_synth.yaml``); ``--build-missing`` /
        ``--train-missing`` default to ``False``.
    """
    p = argparse.ArgumentParser(
        description="beta rate-distortion sweep, HP probes and rank-at-beta "
                    "harness for SeqVaeLagAttnV1 on synthetic data (Phase 6)."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--mode", type=str, default=None,
        choices=["beta", "beta_grid", "hp", "rank_at_beta"],
        help="beta sweep (6.1/6.2) / beta x M x TE grid / HP probe (6.4) / "
             "A-sweep at a chosen beta (6.3)",
    )
    p.add_argument(
        "--axis", type=str, default=None, choices=list(_HP_AXES),
        help="hp mode: which hyper-parameter to sweep",
    )
    p.add_argument(
        "--beta", type=float, default=None,
        help="rank_at_beta mode: the beta to re-run the A-sweep at "
             "(defaults to beta_sweep.selected_beta)",
    )
    p.add_argument(
        "--build-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="build_missing",
        help="generate any missing benchmark dataset (opt-in)",
    )
    p.add_argument(
        "--train-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="train_missing",
        help="train any missing checkpoint (opt-in, multi-hour)",
    )
    p.add_argument(
        "--benchmark", type=str, default=None,
        help="override experiment.benchmark",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override experiment.seed",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """CLI entry point: parse args, load config, dispatch.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    config = tm.load_config(args.config)
    _dispatch(config, vars(args))


if __name__ == "__main__":
    # =========================================================================
    # How to run this script  (project convention -- Decision D9)
    # -------------------------------------------------------------------------
    # Two equivalent modes, auto-detected from the command line:
    #
    #   * CLI mode      -- launched with any --flag -> argparse `main()`.
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the `RUN_CONFIG` dict
    #                      below is used. Edit it and run the file directly;
    #                      no terminal flags required.
    #
    # Every key in RUN_CONFIG mirrors a CLI flag and is forwarded to
    # `_dispatch`; `None` means "fall back to config_synth.yaml".
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "mode": "beta",            # "beta" | "beta_grid" | "hp" | "rank_at_beta"
        "axis": "lambda_base",     # hp mode: lambda_base | d_z | warmup_period
        "beta": None,              # rank_at_beta mode: None -> selected_beta
        "build_missing": False,    # generate any missing dataset (opt-in)
        "train_missing": False,    # train any missing checkpoint (opt-in)
        "benchmark": None,         # None -> config experiment.benchmark
        "device": None,            # None -> config runtime.device
        "seed": None,              # None -> config experiment.seed
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["beta_sweep"]["epochs"] = 1
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
