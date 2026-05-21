r"""Calibration of the latent-KL TE surrogate (Sprint 5).

The TE surrogate the model reports is the latent-space KL
:math:`K_t = D_{\mathrm{KL}}(q_\phi(z_t\mid Y,U)\,\|\,p_\psi(z_t\mid Y))`. By
itself the **shape** of :math:`\bar K` vs the data-generating block transfer
entropy is meaningful (rank correlation, lag attribution), but the
**slope** is not unit-grounded until the reconstruction loss has a nat
scale. ``compute_loss(likelihood='gaussian_nll')`` (Sprint 5.1) gives the
feat / base losses units of *nats per element*, which makes :math:`\bar K`
directly comparable to :math:`\mathrm{TE}^{(H)}_{U\to Y}` measured by the
Gaussian determinant ratio. This module orchestrates that comparison.

Pipeline
--------
1. For each per-step TE target (default :math:`\{0.05, 0.15, 0.30\}`), call
   :func:`analytic_te.B_y_for_te_block_state_space` to invert the uniform
   :math:`B_y` magnitude that yields the requested **block** TE on the
   default G1 oscillator. Build the corresponding dataset cache.
2. For each :math:`(\beta, \mathrm{TE}_{\text{point}})` cell, train one
   model with :math:`\mathrm{likelihood} = \texttt{gaussian\_nll}` and
   evaluate :math:`\bar K` via :func:`evaluate_te.evaluate_checkpoint`.
3. Per :math:`\beta`, fit the linear regression
   :math:`\bar K = \alpha + \gamma\,\mathrm{TE}_{\text{true}}` across the
   three TE points (:func:`fit_calibration_slope`). Pick the :math:`\beta`
   that minimises :math:`|\gamma - 1| + \lambda_\alpha|\alpha|`
   (:func:`select_beta_by_calibration`), tie-breaking by :math:`R^2`.
4. Write ``calibration.json`` (full per-:math:`\beta` table) and
   ``calibration_curve.pdf`` (scatter + fitted line + :math:`y=x` reference
   + inset of :math:`\gamma` vs :math:`\beta`) under
   ``results/<benchmark>/calibration/``.

Strong-result thresholds (``model_validation_v2_plan §8``):
:math:`|\gamma - 1| \le 0.2`, :math:`|\alpha| \le 0.1\cdot\overline{\mathrm{TE}}`,
:math:`R^2 \ge 0.95`.

Run modes follow project Decision V2-D8 -- both a CLI and an edit-and-run
``__main__`` are exposed, auto-detected from ``len(sys.argv)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    analytic_te as ate,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    build_dataset as bd,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    evaluate_te as ev,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- ``paths``
# values resolve relative to ``model_experiment/`` (same convention as the
# other v2 runners).
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Columns of the per-cell summary CSV (one row per (beta, te_point) cell).
# ``knob_name`` / ``knob_value`` are the generic identifiers of the bisected
# coupling magnitude -- ``B_y`` for G1, ``c`` for G2 -- so the CSV schema does
# not change between benchmarks.
_SUMMARY_FIELDS = [
    "benchmark", "te_per_step_target", "data_tag",
    "knob_name", "knob_value", "te_true_block",
    "te_per_step", "beta", "run_tag",
    "k_bar", "k_bar_shuffled", "pred_gap",
    "feat_loss", "base_loss", "kld_loss",
    "mean_logvar_full", "likelihood", "sigma_obs", "free_bits",
    "epoch", "ckpt_path",
]


# =============================================================================
# 1. Cache-builder: invert the coupling for each target TE, build the cache
# =============================================================================

def _emit_calibration_cache_entry(
    config: Dict[str, Any],
    *,
    tau: float,
    tag: str,
    data_key: str,
    data_value: Any,
    knob_name: str,
    knob_value: float,
    te_block_target: float,
    te_block_achieved: float,
    te_per_step_achieved: float,
    force: bool,
) -> Dict[str, Any]:
    r"""Materialise one calibration cache and return its metadata record.

    Overrides ``config["data"][data_key]`` with ``data_value`` (a list for
    G1's :math:`B_y`, a scalar for G2's :math:`c`), stamps ``experiment.tag``
    with ``tag``, runs :func:`build_dataset.build_dataset`, and packages the
    result fields the orchestrator's per-cell loop expects.

    Args:
        config: The parent config (deep-copied locally; not mutated).
        tau: Per-step TE target for this entry (echoed into the result).
        tag: Cache tag, e.g. ``"G1_te015"``.
        data_key: Which field of ``config["data"]`` to overwrite
            (``"B_y"`` for G1, ``"c"`` for G2).
        data_value: Value to write under ``data_key`` (list-wrapped for G1
            so :func:`build_dataset` can auto-tile to ``M``; scalar for G2).
        knob_name: Generic knob identifier (``"B_y"`` or ``"c"``).
        knob_value: The bisected scalar magnitude.
        te_block_target, te_block_achieved, te_per_step_achieved: TE
            bookkeeping fields recorded for the per-cell calibration plot.
        force: Forwarded to :func:`build_dataset.build_dataset`.

    Returns:
        The per-target result dict the builders' callers consume.
    """
    cfg_i = deepcopy(config)
    cfg_i["data"][data_key] = data_value
    cfg_i["experiment"]["tag"] = tag
    cache_dir = bd.build_dataset(cfg_i, force=force)
    return {
        "te_per_step_target": tau,
        "data_tag": tag,
        "knob_name": knob_name,
        "knob_value": knob_value,
        "te_block_target": float(te_block_target),
        "te_block_achieved": float(te_block_achieved),
        "te_per_step_achieved": float(te_per_step_achieved),
        "cache_dir": str(cache_dir),
    }


def build_g1_calibration_caches(
    config: Dict[str, Any],
    *,
    te_per_step_targets: Sequence[float],
    tag_prefix: str = "G1_te",
    inverter_kwargs: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    r"""Build the per-TE-target G1 caches and return their metadata.

    For each target per-step TE :math:`\tau_i`, the helper:

    1. Computes the target block TE :math:`\tau_i \cdot H`.
    2. Calls :func:`analytic_te.B_y_for_te_block_state_space` to find the
       uniform :math:`B_y` magnitude that realises that block TE on the
       active G1 oscillator (``oscillators``, ``target_ar``, ``delays``,
       ``sigma2_y``, ``sigma2_eta`` from ``config["data"]``).
    3. Deep-copies ``config``, overrides ``data.B_y`` to the bisected
       magnitude (length-1 list — the build_dataset dispatch auto-tiles to
       :math:`M`) and ``experiment.tag`` to
       ``f"{tag_prefix}{round(tau_i*100):03d}"``, then runs
       :func:`build_dataset.build_dataset`.

    The achieved (not requested) block TE is stamped into ``meta.json`` via
    the generator's ``meta.te_true`` -- it is what calibration plots use on
    the x-axis. Slight mismatch is expected because the bisection stops at
    its relative tolerance and the Monte-Carlo TE estimator has its own
    noise; see ``model_validation_v2_plan §5.3``.

    Args:
        config: Parsed config_synth.yaml. Must carry the active G1
            benchmark block (``experiment.benchmark`` resolved by
            :func:`train_minimal.resolve_active_benchmark`).
        te_per_step_targets: Sequence of target per-step block TEs
            (in nats). Each entry maps to one cache.
        tag_prefix: Tag prefix; the cache name is
            ``f"{tag_prefix}{round(tau*100):03d}"`` (zero-padded to 3
            digits so ``[0.05, 0.15, 0.30]`` becomes
            ``[..._te005, _te015, _te030]``, sorting-friendly).
        inverter_kwargs: Extra kwargs forwarded to
            :func:`analytic_te.B_y_for_te_block_state_space`
            (``n_samples``, ``lo``, ``hi``, ``tol``, ``max_iter``, ...).
        force: If True, rebuild caches even when they already exist.

    Returns:
        A list of one dict per target with keys ``te_per_step_target``,
        ``data_tag``, ``B_y_scalar``, ``te_block_target``,
        ``te_block_achieved``, ``te_per_step_achieved``, and ``cache_dir``.

    Raises:
        ValueError: If the active benchmark is not G1 (the inverter assumes
            the state-space generator).
    """
    benchmark = str(config["experiment"]["benchmark"])
    if benchmark not in ("G1", "G1-rev"):
        raise ValueError(
            f"build_g1_calibration_caches: expected G1 benchmark, "
            f"got {benchmark!r}. Set experiment.benchmark: G1 in the "
            f"config before calling this helper."
        )
    data = config["data"]
    horizon = int(config["model"]["horizon"])
    inv_kwargs: Dict[str, Any] = dict(inverter_kwargs or {})
    # The inverter needs length-M aligned oscillators / delays. The G1 cache
    # builder auto-tiles length-1 lists to M, so do the same here so the
    # inverter's MC matches the actual cache's MC.
    M = int(data["M"])

    def _tile(seq: Sequence[Any], name: str) -> List[Any]:
        seq = list(seq)
        if len(seq) == 1:
            return [seq[0]] * M
        if len(seq) != M:
            raise ValueError(
                f"build_g1_calibration_caches: data.{name} must have length "
                f"1 or M={M}, got {len(seq)}."
            )
        return seq

    oscillators = [tuple(pair) for pair in _tile(data["oscillators"], "oscillators")]
    delays = [int(d) for d in _tile(data["delays"], "delays")]
    sigma2_y = float(data["sigma2_y"])
    sigma2_eta = data["sigma2_eta"]
    target_ar = float(data["target_ar"])

    results: List[Dict[str, Any]] = []
    for tau in te_per_step_targets:
        tau = float(tau)
        target_block = tau * horizon
        tag = f"{tag_prefix}{int(round(tau * 100)):03d}"
        print(
            f"[caches] target per-step TE={tau:.3f}  "
            f"-> target block={target_block:.3f} nats  tag='{tag}'"
        )
        sol = ate.B_y_for_te_block_state_space(
            target_te_block=target_block,
            oscillators=oscillators,
            target_ar=target_ar,
            delays=delays,
            sigma2_y=sigma2_y,
            sigma2_eta=sigma2_eta,
            H=horizon,
            **inv_kwargs,
        )
        print(
            f"          inverter solved B_y={sol['B_y_scalar']:.4f}  "
            f"(achieved block TE={sol['te_block']:.3f} nats, "
            f"per-step={sol['te_per_step']:.4f}, n_iter={sol['n_iter']})"
        )
        results.append(_emit_calibration_cache_entry(
            config, tau=tau, tag=tag,
            data_key="B_y",
            data_value=[float(sol["B_y_scalar"])],
            knob_name="B_y",
            knob_value=float(sol["B_y_scalar"]),
            te_block_target=target_block,
            te_block_achieved=float(sol["te_block"]),
            te_per_step_achieved=float(sol["te_per_step"]),
            force=force,
        ))
    return results


def build_g2_calibration_caches(
    config: Dict[str, Any],
    *,
    te_per_step_targets: Sequence[float],
    tag_prefix: str = "G2_te",
    inverter_kwargs: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    r"""Build the per-TE-target G2 caches and return their metadata.

    Mirrors :func:`build_g1_calibration_caches` but for the v2-G2 smooth-ARX
    benchmark. For each target per-step TE :math:`\tau_i` the helper:

    1. Computes the *per-channel* target block TE :math:`\tau_i \cdot H`.
    2. Calls :func:`analytic_te.c_for_te_block_arx` to bisect the ARX
       coupling :math:`c` that realises that per-channel block TE under the
       active G2 process (``rho_u``, ``rho_y``, ``sigma2_eta``,
       ``sigma2_eps``, ``delay`` from ``config["data"]``). The inverter
       uses the closed-form :func:`te_block_arx_gaussian`, so no Monte-Carlo
       budget is needed.
    3. Deep-copies ``config``, overrides ``data.c`` to the bisected
       magnitude and ``experiment.tag`` to
       ``f"{tag_prefix}{round(tau_i*100):03d}"`` (so ``[0.05, 0.15, 0.30]``
       becomes ``[..._te005, _te015, _te030]``, matching the G1 prefix
       convention), then runs :func:`build_dataset.build_dataset`.

    The achieved per-channel block TE is stamped into the cache's
    ``meta.json`` via ``gen_smooth_arx.meta.te_true / M``; the *total*
    block TE on the calibration plot's x-axis is :math:`M \cdot` that value.

    Args:
        config: Parsed config_synth.yaml. Must carry the active G2
            benchmark block (``experiment.benchmark`` resolved by
            :func:`train_minimal.resolve_active_benchmark`).
        te_per_step_targets: Sequence of target per-step block TEs
            (in nats). Each entry maps to one cache.
        tag_prefix: Tag prefix; the cache name is
            ``f"{tag_prefix}{round(tau*100):03d}"``.
        inverter_kwargs: Extra kwargs forwarded to
            :func:`analytic_te.c_for_te_block_arx` (``lo``, ``hi``, ``tol``,
            ``max_iter``, ``K_history``).
        force: If True, rebuild caches even when they already exist.

    Returns:
        A list of one dict per target with keys ``te_per_step_target``,
        ``data_tag``, ``knob_name`` (= ``"c"``), ``knob_value`` (= the
        bisected coupling), ``te_block_target``, ``te_block_achieved``,
        ``te_per_step_achieved``, and ``cache_dir``.

    Raises:
        ValueError: If the active benchmark is not G2 (the inverter assumes
            the smooth-ARX generator).
    """
    benchmark = str(config["experiment"]["benchmark"])
    if benchmark != "G2":
        raise ValueError(
            f"build_g2_calibration_caches: expected G2 benchmark, "
            f"got {benchmark!r}. Set experiment.benchmark: G2 in the "
            f"config before calling this helper."
        )
    data = config["data"]
    horizon = int(config["model"]["horizon"])
    inv_kwargs: Dict[str, Any] = dict(inverter_kwargs or {})

    rho_u = float(data["rho_u"])
    rho_y = float(data["rho_y"])
    sigma2_eta = float(data["sigma2_eta"])
    sigma2_eps = float(data["sigma2_eps"])
    delay = int(data["delay"])

    results: List[Dict[str, Any]] = []
    for tau in te_per_step_targets:
        tau = float(tau)
        target_block = tau * horizon
        tag = f"{tag_prefix}{int(round(tau * 100)):03d}"
        print(
            f"[caches] target per-step TE={tau:.3f}  "
            f"-> target per-channel block={target_block:.3f} nats  "
            f"tag='{tag}'"
        )
        sol = ate.c_for_te_block_arx(
            target_te_block=target_block,
            rho_u=rho_u, rho_y=rho_y,
            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
            H=horizon, D=delay,
            **inv_kwargs,
        )
        print(
            f"          inverter solved c={sol['c_scalar']:.4f}  "
            f"(achieved per-channel block TE={sol['te_block']:.3f} nats, "
            f"per-step={sol['te_per_step']:.4f}, n_iter={sol['n_iter']})"
        )
        results.append(_emit_calibration_cache_entry(
            config, tau=tau, tag=tag,
            data_key="c",
            data_value=float(sol["c_scalar"]),
            knob_name="c",
            knob_value=float(sol["c_scalar"]),
            te_block_target=target_block,
            te_block_achieved=float(sol["te_block"]),
            te_per_step_achieved=float(sol["te_per_step"]),
            force=force,
        ))
    return results


class CalibrationSpec(NamedTuple):
    r"""Per-benchmark calibration cache-builder + meta-key descriptor.

    Mirrors the :class:`null_controls.ControlSpec` convention. The
    no-build branch of :func:`run_calibration` reads ``meta.get(meta_key)``
    and accepts either a scalar (G2's ``c``) or a length-$M$ list (G1's
    ``B_y``); the list case is detected at runtime, matching the polymorphism
    in :func:`evaluate_te.evaluate_checkpoint`.
    """
    builder: Callable[..., List[Dict[str, Any]]]
    knob_name: str
    meta_key: str


_CALIBRATION_BUILDERS: Dict[str, CalibrationSpec] = {
    "G1": CalibrationSpec(build_g1_calibration_caches, "B_y", "B_y"),
    "G2": CalibrationSpec(build_g2_calibration_caches, "c", "c"),
}


def _get_calibration_spec(benchmark: str) -> CalibrationSpec:
    r"""Look up the :class:`CalibrationSpec` for ``benchmark``.

    Raises:
        ValueError: If ``benchmark`` has no registered calibration builder.
    """
    spec = _CALIBRATION_BUILDERS.get(benchmark)
    if spec is None:
        raise ValueError(
            f"calibration: unsupported benchmark {benchmark!r}; "
            f"valid: {sorted(_CALIBRATION_BUILDERS)}."
        )
    return spec


def _read_meta_knob(raw: Any) -> float:
    r"""Coerce a meta knob (scalar or length-$M$ list) to a single ``float``.

    Mirrors the polymorphism at :func:`evaluate_te.evaluate_checkpoint`: if
    the generator stored a list (G1's ``B_y``), take the first entry; if it
    stored a scalar (G2's ``c``), cast directly; otherwise return NaN.
    """
    if isinstance(raw, (list, tuple)) and raw:
        return float(raw[0])
    if isinstance(raw, (int, float)):
        return float(raw)
    return float("nan")


# =============================================================================
# 2. Cell runner: train + evaluate at one (beta, te_point)
# =============================================================================

def _beta_token(beta: float) -> str:
    """Filesystem-safe label for one beta (``1.0e-3`` -> ``beta_1.0e-3``)."""
    return f"beta_{beta:.1e}"


def _run_calibration_cell(
    config: Dict[str, Any],
    *,
    benchmark: str,
    data_tag: str,
    beta: float,
    likelihood: str,
    sigma_obs: "float | str",
    free_bits: float,
    train_missing: bool,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Train (opt-in) and evaluate one (β, TE point) cell.

    The checkpoint lands at
    ``results/<benchmark>/calibration/<data_tag>/<beta_token>/best.ckpt``,
    so re-runs are idempotent (existing ckpt re-evaluated without retraining).

    Args:
        config: Parsed config (benchmark already resolved).
        benchmark: The active benchmark (G1 in v2-D9).
        data_tag: Cache tag for this TE point.
        beta: KL weight for this cell.
        likelihood, sigma_obs, free_bits: Sprint-5 loss-switch settings,
            forwarded to ``train_minimal.train``.
        train_missing: If True, train the cell when no checkpoint exists.
        device: Compute device.

    Returns:
        The ``evaluate_te.evaluate_checkpoint`` row augmented with the
        knobs of this cell, or ``None`` if the cell is missing and
        ``train_missing`` is False.
    """
    run_tag = f"calibration/{data_tag}/{_beta_token(beta)}"
    ckpt_path = ev._results_root(config) / benchmark / run_tag / "best.ckpt"

    if not ckpt_path.is_file():
        if not train_missing:
            print(
                f"  [skip ] {data_tag} @ beta={beta:g}: ckpt missing "
                f"(pass --train-missing to train)"
            )
            return None
        print(f"  [train] {data_tag} @ beta={beta:g}  -> {run_tag}")
        tcfg = deepcopy(config)
        # Loss-block overrides for this cell. ``train`` reads the loss
        # block directly, so editing in place is enough.
        tcfg["loss"] = dict(tcfg.get("loss", {}))
        tcfg["loss"]["kld_beta"] = float(beta)
        tcfg["loss"]["likelihood"] = likelihood
        tcfg["loss"]["sigma_obs"] = sigma_obs
        tcfg["loss"]["free_bits"] = float(free_bits)
        tm.train(
            tcfg,
            overrides={"data_tag": data_tag, "run_tag": run_tag},
        )

    row = ev.evaluate_checkpoint(
        ckpt_path, config, device=device, data_tag=data_tag,
    )
    row["te_per_step_target"] = None  # filled by the caller
    row["data_tag"] = data_tag
    row["beta"] = float(beta)
    row["run_tag"] = run_tag
    return row


# =============================================================================
# 3. Calibration slope fit + beta selector
# =============================================================================

def fit_calibration_slope(
    table: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    r"""Ordinary least squares of :math:`\bar K = \alpha + \gamma\,\mathrm{TE}`.

    Given a sequence of :math:`(\mathrm{TE}_{\text{true}}, \bar K)` pairs
    at a *fixed* :math:`\beta`, fits the two-parameter linear model by
    closed-form OLS and returns :math:`\alpha`, :math:`\gamma`, the
    coefficient of determination :math:`R^2`, and the number of points.

    With only :math:`n=3` points (the v2 calibration plan), :math:`R^2`
    can be misleadingly close to 1 even if the fit is poor; the headline
    quality metric is :math:`|\gamma - 1|` itself (see Metric 3 in
    ``model_validation_v2_plan §8``).

    Args:
        table: Iterable of ``(te_true, k_bar)`` pairs, at least two of which
            have distinct ``te_true`` (otherwise OLS is singular).

    Returns:
        ``{'alpha': float, 'gamma': float, 'r2': float, 'n': int}``.

    Raises:
        ValueError: If ``table`` has fewer than two points or all
            ``te_true`` values coincide.
    """
    arr = np.asarray(list(table), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"fit_calibration_slope: expected a sequence of (te, k_bar) "
            f"pairs, got shape {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise ValueError(
            "fit_calibration_slope: need at least 2 points for a slope fit."
        )
    x = arr[:, 0]
    y = arr[:, 1]
    x_var = float(np.var(x))
    if x_var <= 1e-18:
        raise ValueError(
            "fit_calibration_slope: te_true values coincide; OLS is singular."
        )
    gamma = float(np.cov(x, y, bias=True)[0, 1] / x_var)
    alpha = float(np.mean(y) - gamma * np.mean(x))
    y_hat = alpha + gamma * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return {"alpha": alpha, "gamma": gamma, "r2": r2, "n": int(arr.shape[0])}


def select_beta_by_calibration(
    per_beta_table: Sequence[Dict[str, Any]],
    *,
    alpha_penalty: float = 0.05,
) -> Dict[str, Any]:
    r"""Pick the :math:`\beta` with the best calibration fit.

    Scoring: :math:`s(\beta) = |\gamma(\beta) - 1| + \lambda_\alpha\,
    |\alpha(\beta)|`. The selected :math:`\beta` minimises :math:`s`, with
    :math:`R^2` as a tie-breaker (higher is better). Cells with non-finite
    :math:`\gamma` or :math:`\alpha` are skipped.

    Args:
        per_beta_table: Sequence of dicts with at least keys ``beta``,
            ``alpha``, ``gamma``, ``r2`` -- exactly the records produced by
            mapping :func:`fit_calibration_slope` over the
            :math:`\beta` grid.
        alpha_penalty: :math:`\lambda_\alpha` in the scoring rule. Default
            0.05 gives a mild preference for :math:`\alpha \approx 0`
            without overpowering the slope objective.

    Returns:
        ``{'beta': float, 'alpha': float, 'gamma': float, 'r2': float,
        'score': float, 'rationale': str}`` for the selected
        :math:`\beta`, or ``None`` if no cell has finite ``alpha``/
        ``gamma``.
    """
    best: Optional[Dict[str, Any]] = None
    for cell in per_beta_table:
        alpha = float(cell.get("alpha", float("nan")))
        gamma = float(cell.get("gamma", float("nan")))
        r2 = float(cell.get("r2", float("nan")))
        if not (math.isfinite(alpha) and math.isfinite(gamma)):
            continue
        score = abs(gamma - 1.0) + float(alpha_penalty) * abs(alpha)
        if (
            best is None
            or score < best["score"] - 1e-12
            or (
                abs(score - best["score"]) < 1e-12
                and math.isfinite(r2)
                and (not math.isfinite(best["r2"]) or r2 > best["r2"])
            )
        ):
            best = {
                "beta": float(cell["beta"]),
                "alpha": alpha,
                "gamma": gamma,
                "r2": r2,
                "score": score,
                "rationale": (
                    f"argmin |gamma-1| + {alpha_penalty:g}*|alpha| over "
                    f"the calibration beta grid; tie-break by R^2."
                ),
            }
    return best or {}


# =============================================================================
# 4. Headline figure
# =============================================================================

def plot_calibration_curve(
    rows: Sequence[Dict[str, Any]],
    per_beta_table: Sequence[Dict[str, Any]],
    selected: Dict[str, Any],
    out_dir: Path,
) -> Path:
    r"""Write ``calibration_curve.pdf`` (+ ``.png``) under ``out_dir``.

    The figure shows :math:`\bar K` vs :math:`\mathrm{TE}_{\text{true}}`
    at the **selected** :math:`\beta` as a scatter, with the fitted line
    :math:`\bar K = \alpha + \gamma\,\mathrm{TE}` and the
    :math:`y = x` reference dashed. :math:`(\alpha, \gamma, R^2)` are
    annotated in the corner. An inset (axes-fraction coordinates) shows
    :math:`\gamma(\beta)` across the full :math:`\beta` grid so the choice
    of :math:`\beta` is visible.

    Args:
        rows: Per-cell rows; the scatter uses those whose ``beta`` matches
            ``selected["beta"]`` to within ``1e-12``.
        per_beta_table: One record per :math:`\beta` with ``beta``,
            ``gamma``, ``r2``; drives the inset.
        selected: The selected-:math:`\beta` record from
            :func:`select_beta_by_calibration`.
        out_dir: Destination directory; created if missing.

    Returns:
        The path of the written PDF.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beta_star = float(selected.get("beta", float("nan")))
    cell_rows = [
        r for r in rows
        if math.isfinite(beta_star)
        and math.isfinite(float(r.get("beta", float("nan"))))
        and abs(float(r["beta"]) - beta_star) < 1e-12
    ]
    cell_rows.sort(key=lambda r: float(r.get("te_true", 0.0)))
    xs = np.array([float(r["te_true"]) for r in cell_rows], dtype=float)
    ys = np.array([float(r["k_bar"]) for r in cell_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    if xs.size > 0:
        # y = x reference (dashed grey)
        lim_hi = float(max(xs.max(), ys.max()) * 1.1) if xs.size else 1.0
        lim_hi = max(lim_hi, 1e-3)
        ax.plot(
            [0.0, lim_hi], [0.0, lim_hi],
            linestyle="--", color=ps.COLOR_GRAY, linewidth=1.0,
            label=r"$y = x$",
        )
        ax.scatter(
            xs, ys, s=64, color=ps.COLOR_BLUE,
            edgecolor=ps.COLOR_BLACK, linewidth=0.7, zorder=3,
            label=r"calibration points",
        )
        # Fitted line
        alpha = float(selected.get("alpha", 0.0))
        gamma = float(selected.get("gamma", 1.0))
        xx = np.linspace(0.0, lim_hi, 64)
        ax.plot(
            xx, alpha + gamma * xx,
            color=ps.COLOR_VERMILLION, linewidth=1.4,
            label=rf"$\bar K = {alpha:.3f} + {gamma:.3f}\,\mathrm{{TE}}$",
        )
        ax.set_xlim(0.0, lim_hi)
        ax.set_ylim(0.0, lim_hi)
        ax.annotate(
            rf"$\beta^\star = {beta_star:.1e}$"
            "\n"
            rf"$\gamma = {gamma:.3f}$, $\alpha = {alpha:.3f}$"
            "\n"
            rf"$R^2 = {float(selected.get('r2', float('nan'))):.3f}$",
            xy=(0.04, 0.96), xycoords="axes fraction",
            ha="left", va="top",
            fontsize=ps.FONT_LEGEND, color=ps.COLOR_BLACK,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor=ps.COLOR_LIGHT_GRAY),
        )

    ax.set_xlabel(r"true block TE $\mathrm{TE}^{(H)}_{U\to Y}$ (nats)")
    ax.set_ylabel(r"latent KL $\bar K$ (nats)")
    ax.set_title(r"calibration: $\bar K$ vs $\mathrm{TE}_{\mathrm{true}}$ "
                 r"at selected $\beta$")
    ax.legend(loc="lower right", frameon=False)
    ps.style_axes(ax)

    # Inset: gamma vs beta across the grid.
    inset = ax.inset_axes([0.58, 0.10, 0.36, 0.32])
    valid = [
        (float(c["beta"]), float(c["gamma"]))
        for c in per_beta_table
        if math.isfinite(float(c.get("gamma", float("nan"))))
        and math.isfinite(float(c.get("beta", float("nan"))))
    ]
    if valid:
        bs, gs = zip(*sorted(valid))
        inset.plot(
            bs, gs, marker="o", color=ps.COLOR_BLUE,
            linewidth=1.0, markersize=3, markeredgecolor=ps.COLOR_BLACK,
            markeredgewidth=0.4,
        )
        inset.axhline(1.0, color=ps.COLOR_GRAY, linestyle="--", linewidth=0.8)
        if math.isfinite(beta_star):
            inset.axvline(
                beta_star, color=ps.COLOR_VERMILLION,
                linestyle=":", linewidth=0.8,
            )
        inset.set_xscale("log")
        inset.set_xlabel(r"$\beta$", fontsize=ps.FONT_TICK)
        inset.set_ylabel(r"$\gamma$", fontsize=ps.FONT_TICK)
        inset.tick_params(labelsize=ps.FONT_TICK)
        ps.style_axes(inset)

    fig.tight_layout()
    paths = ps.save_figure(fig, out_dir / "calibration_curve")
    return Path(paths[0]) if isinstance(paths, (list, tuple)) else Path(paths)


# =============================================================================
# 5. Top-level orchestrator
# =============================================================================

def _write_summary_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write the per-cell summary CSV using :data:`_SUMMARY_FIELDS` ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SUMMARY_FIELDS})


def run_calibration(
    config: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    build_missing: bool = False,
    train_missing: bool = False,
) -> Dict[str, Any]:
    r"""Run the 27-cell calibration matrix end-to-end.

    Pipeline (see module docstring): build the per-TE-target caches
    (opt-in), train one model per :math:`(\beta, \mathrm{TE}_{\text{point}})`
    cell (opt-in), evaluate every checkpoint, fit the per-:math:`\beta`
    slope, pick :math:`\beta^\star`, write the JSON + summary CSV + PDF.

    Args:
        config: Parsed ``config_synth.yaml`` (benchmark-resolved).
        device: Compute device. Defaults to
            :func:`train_minimal.resolve_device`.
        build_missing: If True, generate any missing G1 calibration cache.
        train_missing: If True, train any missing calibration cell.

    Returns:
        A results dict: ``caches``, ``rows`` (per-cell), ``per_beta_table``,
        ``selected``, ``out_dir``, ``skipped``.
    """
    device = device or tm.resolve_device(config["runtime"])
    cal_cfg = config.get("calibration", {}) or {}
    benchmark = str(cal_cfg.get("benchmark", config["experiment"]["benchmark"]))
    spec = _get_calibration_spec(benchmark)
    targets = list(cal_cfg.get("te_per_step_targets", [0.05, 0.15, 0.30]))
    default_prefix = f"{benchmark}_te"
    tag_prefix = str(cal_cfg.get("tag_prefix", default_prefix))
    likelihood = str(cal_cfg.get("likelihood", "gaussian_nll"))
    sigma_obs_raw = cal_cfg.get("sigma_obs", "learned")
    sigma_obs = (
        sigma_obs_raw if (isinstance(sigma_obs_raw, str)
                          and sigma_obs_raw == "learned")
        else float(sigma_obs_raw)
    )
    free_bits = float(cal_cfg.get("free_bits", 0.0))
    # `beta_grid: null` in the YAML means "fall back to the rate-distortion
    # grid from `beta_sweep:`" so the calibration matrix and the rate-
    # distortion plot can share a single source of truth.
    beta_grid_raw = cal_cfg.get("beta_grid")
    if beta_grid_raw is None:
        beta_grid_raw = config.get("beta_sweep", {}).get("grid", [1e-3])
    beta_grid = [float(b) for b in beta_grid_raw]
    alpha_penalty = float(cal_cfg.get("alpha_penalty", 0.05))
    inverter_kwargs: Dict[str, Any] = dict(cal_cfg.get("inverter", {}) or {})
    out_subdir = str(cal_cfg.get("out_dir", "calibration"))

    out_dir = ev._results_root(config) / benchmark / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the active benchmark block is the one this orchestrator
    # claims to be running -- the cache builder assumes the matching
    # generator (G1 state-space or G2 smooth-ARX).
    cfg = deepcopy(config)
    cfg["experiment"]["benchmark"] = benchmark
    tm.resolve_active_benchmark(cfg)

    print(
        f"[calibration] benchmark={benchmark}  device={device}  "
        f"likelihood={likelihood}  sigma_obs={sigma_obs}  "
        f"free_bits={free_bits:g}\n"
        f"              targets={targets}  betas={beta_grid}\n"
        f"              build_missing={build_missing}  "
        f"train_missing={train_missing}  out_dir={out_dir}"
    )

    if build_missing:
        caches = spec.builder(
            cfg,
            te_per_step_targets=targets,
            tag_prefix=tag_prefix,
            inverter_kwargs=inverter_kwargs,
            force=False,
        )
    else:
        # Re-read the achieved TE from existing caches' meta.json so the
        # calibration plot uses the *measured* x-axis rather than the
        # requested one. Missing caches are reported but do not abort.
        caches = []
        data_root = ev._data_root(cfg)
        for tau in targets:
            tag = f"{tag_prefix}{int(round(float(tau) * 100)):03d}"
            meta_path = data_root / benchmark / tag / "meta.json"
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                caches.append({
                    "te_per_step_target": float(tau),
                    "data_tag": tag,
                    "knob_name": spec.knob_name,
                    "knob_value": _read_meta_knob(meta.get(spec.meta_key)),
                    "te_block_target": float(tau) * int(cfg["model"]["horizon"]),
                    "te_block_achieved": float(meta.get("te_true", float("nan"))),
                    "te_per_step_achieved": float(
                        meta.get("te_per_step", float("nan"))
                    ),
                    "cache_dir": str(meta_path.parent),
                })
            else:
                print(
                    f"  [skip ] cache '{tag}' not found "
                    f"(pass --build-missing to create it)"
                )

    if not caches:
        print(
            "[calibration] no caches available -- pass --build-missing "
            "to generate them. Aborting before the train/eval phase."
        )
        return {
            "caches": [], "rows": [], "per_beta_table": [],
            "selected": {}, "out_dir": str(out_dir), "skipped": [],
        }

    # --- Step 2: train + evaluate each (beta, TE-point) cell -------------
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for beta in beta_grid:
        for entry in caches:
            tag = entry["data_tag"]
            try:
                row = _run_calibration_cell(
                    cfg, benchmark=benchmark, data_tag=tag, beta=beta,
                    likelihood=likelihood, sigma_obs=sigma_obs,
                    free_bits=free_bits,
                    train_missing=train_missing, device=device,
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell must not abort
                print(
                    f"  [error] {tag} @ beta={beta:g}: "
                    f"{type(exc).__name__}: {exc}"
                )
                skipped.append(f"{tag}@beta_{beta:g}")
                continue
            if row is None:
                skipped.append(f"{tag}@beta_{beta:g}")
                continue
            row["benchmark"] = benchmark
            row["te_per_step_target"] = float(entry["te_per_step_target"])
            row["knob_name"] = str(entry["knob_name"])
            row["knob_value"] = float(entry["knob_value"])
            row["te_true_block"] = float(entry["te_block_achieved"])
            row["te_per_step"] = float(entry["te_per_step_achieved"])
            rows.append(row)

    _write_summary_csv(rows, out_dir / "summary.csv")

    # --- Step 3: per-beta slope fit + beta selector ----------------------
    per_beta_table: List[Dict[str, Any]] = []
    for beta in beta_grid:
        pairs = [
            (float(r["te_true_block"]), float(r["k_bar"]))
            for r in rows
            if abs(float(r["beta"]) - float(beta)) < 1e-12
            and math.isfinite(float(r.get("k_bar", float("nan"))))
        ]
        if len(pairs) < 2 or len({p[0] for p in pairs}) < 2:
            # Underdetermined; record the cell but skip the fit.
            per_beta_table.append({
                "beta": float(beta),
                "alpha": float("nan"),
                "gamma": float("nan"),
                "r2": float("nan"),
                "n": len(pairs),
            })
            continue
        fit = fit_calibration_slope(pairs)
        per_beta_table.append({"beta": float(beta), **fit})

    selected = select_beta_by_calibration(
        per_beta_table, alpha_penalty=alpha_penalty
    )

    # --- Step 4: artifact emission ---------------------------------------
    if selected:
        plot_calibration_curve(rows, per_beta_table, selected, out_dir)
        print(
            f"[calibration] selected beta={selected['beta']:.1e}  "
            f"gamma={selected['gamma']:.3f}  alpha={selected['alpha']:.3f}  "
            f"R^2={selected['r2']:.3f}  (score={selected['score']:.3f})"
        )
    else:
        print("[calibration] no beta cell yielded a valid slope fit; "
              "skipping the curve plot.")

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "likelihood": likelihood,
        "sigma_obs": sigma_obs,
        "free_bits": free_bits,
        "alpha_penalty": alpha_penalty,
        "te_points": [
            {
                "te_per_step_target": float(e["te_per_step_target"]),
                "data_tag": e["data_tag"],
                "knob_name": str(e["knob_name"]),
                "knob_value": float(e["knob_value"]),
                "te_block_target": float(e["te_block_target"]),
                "te_block_achieved": float(e["te_block_achieved"]),
                "te_per_step_achieved": float(e["te_per_step_achieved"]),
            }
            for e in caches
        ],
        "betas": list(beta_grid),
        "table": per_beta_table,
        "selected": selected,
        "skipped": skipped,
        "n_rows": len(rows),
        # Slim per-cell records so the final-report headline figure can plot
        # K_bar vs TE_true at the selected beta without re-reading summary.csv.
        # Mirrors the columns plot_calibration_curve uses.
        "cells": [
            {
                "te_per_step_target": float(r.get("te_per_step_target",
                                                  float("nan"))),
                "data_tag": str(r.get("data_tag", "")),
                "knob_value": float(r.get("knob_value", float("nan"))),
                "te_true_block": float(r.get("te_true_block", float("nan"))),
                "beta": float(r.get("beta", float("nan"))),
                "k_bar": float(r.get("k_bar", float("nan"))),
            }
            for r in rows
        ],
    }
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(
        f"[done] {len(rows)} cell(s) evaluated, {len(skipped)} skipped\n"
        f"       artifacts -> {out_dir}"
    )
    return {
        "caches": caches,
        "rows": rows,
        "per_beta_table": per_beta_table,
        "selected": selected,
        "out_dir": str(out_dir),
        "skipped": skipped,
    }


# =============================================================================
# 6. CLI / RUN_CONFIG dispatch (Decision V2-D8)
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the calibration-level overrides onto ``config`` in place.

    The ``benchmark`` override updates **both** ``experiment.benchmark`` (so
    :func:`train_minimal.resolve_active_benchmark` overlays the right
    per-benchmark ``data`` / ``sweep`` block) and ``calibration.benchmark``
    (so :func:`run_calibration` picks the matching cache builder).
    """
    if overrides.get("device") is not None:
        config["runtime"]["device"] = overrides["device"]
    if overrides.get("seed") is not None:
        config["experiment"]["seed"] = overrides["seed"]
    if overrides.get("benchmark") is not None:
        bench = str(overrides["benchmark"])
        config["experiment"]["benchmark"] = bench
        config.setdefault("calibration", {})["benchmark"] = bench
        # Reset tag_prefix so e.g. switching to G2 doesn't reuse a G1_te... tag.
        config["calibration"].pop("tag_prefix", None)
    return config


def _dispatch(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve overrides, seed, device and run the calibration."""
    config = deepcopy(config)
    _apply_overrides(config, overrides)
    tm.set_seed(int(config["experiment"].get("seed", 0)))
    device = tm.resolve_device(config["runtime"])
    return run_calibration(
        config,
        device=device,
        build_missing=bool(overrides.get("build_missing")),
        train_missing=bool(overrides.get("train_missing")),
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Calibration of latent KL against block TE "
                    "for the v2 G1 / G2 benchmarks."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--benchmark", type=str, default=None,
        choices=sorted(_CALIBRATION_BUILDERS.keys()),
        help="override calibration.benchmark (e.g. G1, G2). Defaults to the "
             "config's calibration.benchmark.",
    )
    p.add_argument(
        "--build-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="build_missing",
        help="generate any missing calibration cache for the active benchmark "
             "(opt-in)",
    )
    p.add_argument(
        "--train-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="train_missing",
        help="train any missing calibration cell (opt-in, multi-hour)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="override runtime.device (auto / cpu / cuda / cuda:N)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="override experiment.seed",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    config = tm.load_config(args.config)
    _dispatch(config, vars(args))


if __name__ == "__main__":
    # =========================================================================
    # Two equivalent run modes (Decision V2-D8):
    #   * CLI mode      -- launched with any --flag -> argparse main().
    #   * EDIT-AND-RUN  -- launched with NO arguments -> the RUN_CONFIG dict
    #                      below is used.
    # =========================================================================

    CONFIG_PATH = _DEFAULT_CONFIG

    RUN_CONFIG = {
        "benchmark": None,        # None -> use config calibration.benchmark
        "build_missing": False,
        "train_missing": False,
        "device": None,
        "seed": None,
    }

    if len(sys.argv) > 1:
        main()
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g. config["calibration"]["te_per_step_targets"] = [0.10, 0.20]
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
