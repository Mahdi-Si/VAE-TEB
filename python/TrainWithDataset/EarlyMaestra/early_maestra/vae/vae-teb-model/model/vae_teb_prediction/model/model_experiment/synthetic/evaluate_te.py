r"""TE evaluation harness -- $\bar{K}$ versus ground-truth TE (Phase 4).

Loads a trained :class:`SeqVaeLagAttnV1` checkpoint, runs inference over the
cached **test** split, and computes the four validation metrics of
``model_validation.md`` Section 5:

    Metric 1 -- null error: $E_0 = |\bar{K}|$ for zero-transfer ($a=0$) data,
        plus a cheap shuffled-source control.
    Metric 2 -- monotonicity: Spearman $\rho(\bar{K}, \mathrm{TE}_{\rm true})$
        across the A-sweep.
    Metric 3 -- calibration slope: fit $\bar{K} = \alpha + \gamma\,
        \mathrm{TE}_{\rm true}$ and report $\alpha, \gamma, R^2$.
    Metric 4 -- predictive gain: ``pred_gap`` $= \mathcal{L}_{\rm base} -
        \mathcal{L}_{\rm feat}$ versus $\bar{K}$ and $\mathrm{TE}_{\rm true}$.

$\bar{K}$ is the per-step KL $K_t$ averaged over valid anchors
$t \in [\text{warmup}, T)$, obtained via
:meth:`SeqVaeLagAttnV1.measure_transfer_entropy`. Checkpoints are reloaded with
``train/graph_models_utils.py:load_checkpoint_strict``.

This module **reuses** the training-loop helpers in :mod:`train_minimal`
(:func:`evaluate`, :func:`compute_kbar`, ...) so the harness scores models with
the exact loss / KL code Phase 3 trained against -- it never re-implements them.

Three run modes:
    * ``single`` -- evaluate one checkpoint, write a one-row summary.
    * ``sweep``  -- enumerate the active benchmark's ``sweep`` grid
      (Gaussian ``a_grid`` x ``m_grid``, XOR ``q_grid`` x ``m_grid``, or the
      single two-lag cell), evaluate every cell, aggregate Metrics 1-4, render
      the headline plots.
    * ``rho_null`` -- the Benchmark-B headline diagnostic: $\bar K \approx 0$ at
      $a=0$ for every $\rho$ in ``rho_null.rho_grid`` (no target-self-info
      leakage).

    Missing datasets / checkpoints can be built / trained behind the opt-in
    ``build_missing`` / ``train_missing`` flags (default OFF -- the harness
    otherwise strictly evaluates what already exists on disk).

Run modes (project convention -- Decision D9 in
``synthetic_te_validation_plan.md``): like every ``synthetic/`` runner this file
supports **both** a CLI and an edit-and-run ``__main__``, auto-detected from
whether any command-line argument is present.

    * CLI mode (any ``--flag`` passed)::

        python -m ...synthetic.evaluate_te --mode single --checkpoint PATH
        python -m ...synthetic.evaluate_te --mode sweep [--build-missing]
            [--train-missing] [--config PATH] [--data-tag TAG] [--benchmark B]
            [--batch-size N] [--device DEV] [--seed S]

    * Edit-and-run mode (no arguments) -- edit the ``RUN_CONFIG`` dict in the
      ``__main__`` block, then run the file directly (IDE / notebook)::

        python -m ...synthetic.evaluate_te
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

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    build_dataset as bd,
)
from model.vae_teb_prediction.model.model_experiment.synthetic import (
    train_minimal as tm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    te_block_gaussian,
    te_block_xor,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    AttributeDict,
    SyntheticTEDataset,
    build_u_stream,
    make_dataloader,
)
from train.graph_models_utils import load_checkpoint_strict

# ``synthetic/`` package dir and its parent ``model_experiment/`` -- the
# ``paths.*`` config values are resolved relative to ``model_experiment/``
# (identical convention to train_minimal.py / build_dataset.py).
_PKG_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _PKG_DIR.parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"

# Columns of the per-setting summary CSV (task 4.8). One row per evaluated
# checkpoint; the d_z-length ``per_dim_kl`` vector goes to ``metrics.json``.
# ``a`` / ``q`` / ``rho`` are the benchmark-specific knobs (Gaussian / XOR / AR):
# whichever the active benchmark does not use is left blank.
_SUMMARY_FIELDS = [
    "run_tag", "data_tag", "benchmark", "a", "q", "rho", "M",
    "te_true", "te_per_step",
    "k_bar", "k_bar_shuffled", "pred_gap", "feat_loss", "base_loss", "kld_loss",
    "mu_post_prior_gap", "attn_entropy", "n_test", "warmup", "epoch",
    "latent_stats_fitted", "ckpt_path",
]


# =============================================================================
# Path helpers
# =============================================================================

def _data_root(config: Dict[str, Any]) -> Path:
    """Resolve the dataset cache root from ``paths.data_dir``.

    Args:
        config: The parsed ``config_synth.yaml``.

    Returns:
        Absolute path of ``<model_experiment>/<paths.data_dir>``.
    """
    return (_EXPERIMENT_DIR / str(config["paths"]["data_dir"])).resolve()


def _results_root(config: Dict[str, Any]) -> Path:
    """Resolve the results root from ``paths.results_dir``.

    Args:
        config: The parsed ``config_synth.yaml``.

    Returns:
        Absolute path of ``<model_experiment>/<paths.results_dir>``.
    """
    return (_EXPERIMENT_DIR / str(config["paths"]["results_dir"])).resolve()


def _eval_out_dir(config: Dict[str, Any], benchmark: str) -> Path:
    """Resolve (and create) the Phase-4 output directory.

    Args:
        config: The parsed config.
        benchmark: Benchmark identifier (e.g. ``"A"``).

    Returns:
        ``<results_root>/<benchmark>/eval_te`` -- created if absent. Keeps the
        evaluation artifacts separate from per-run training directories.
    """
    out_dir = _results_root(config) / str(benchmark) / "eval_te"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =============================================================================
# Checkpoint / data loading
# =============================================================================

def load_eval_checkpoint(
    ckpt_path: Any, device: torch.device
) -> Tuple[SeqVaeLagAttnV1, Dict[str, Any]]:
    """Reconstruct a :class:`SeqVaeLagAttnV1` and load its trained weights.

    The model architecture is rebuilt from the ``model_kwargs`` stored in the
    checkpoint (so the harness needs no config to match the trained shapes),
    and the weights are loaded with the project-standard
    :func:`load_checkpoint_strict` (``strict=True`` alignment).

    Args:
        ckpt_path: Path to a ``.ckpt`` written by :func:`train_minimal.train`.
        device: Device to move the loaded model onto.

    Returns:
        Tuple ``(model, ckpt)`` -- the model in ``eval`` mode on ``device``,
        and the raw checkpoint dict (carries ``data_meta``, ``config``,
        ``loss_settings``, ``epoch`` ...).

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
        RuntimeError: If :func:`load_checkpoint_strict` cannot align the weights
            (it returns ``None`` on a key/shape mismatch rather than raising).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # weights_only=False: the checkpoint also carries plain-Python config /
    # meta dicts; this is trusted local data written by train_minimal.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_kwargs = dict(ckpt["model_kwargs"])
    # YAML / checkpoint store ``logvar_clamp`` as a list; the constructor wants
    # a tuple (same coercion train_minimal.build_model performs).
    clamp = model_kwargs.get("logvar_clamp")
    if clamp is not None and not isinstance(clamp, tuple):
        model_kwargs["logvar_clamp"] = (float(clamp[0]), float(clamp[1]))

    model = SeqVaeLagAttnV1(**model_kwargs)
    loaded = load_checkpoint_strict(model, ckpt)
    if loaded is None:
        raise RuntimeError(
            f"load_checkpoint_strict could not align weights for {ckpt_path}. "
            f"The checkpoint's architecture does not match SeqVaeLagAttnV1("
            f"**model_kwargs)."
        )
    model.to(device).eval()
    return model, ckpt


def make_test_loader(
    config: Dict[str, Any], benchmark: str, data_tag: str, batch_size: int
) -> Tuple[Any, Dict[str, Any]]:
    """Build a deterministic dataloader over a cached benchmark test split.

    Args:
        config: The parsed config (used only for ``paths.data_dir``).
        benchmark: Benchmark identifier (e.g. ``"A"``).
        data_tag: Cache tag -- the ``<data_root>/<benchmark>/<data_tag>/``
            subdirectory.
        batch_size: Samples per batch.

    Returns:
        Tuple ``(loader, meta)`` -- a non-shuffled :class:`DataLoader` over
        ``test.npz`` and the dataset ``meta.json``.

    Raises:
        FileNotFoundError: If ``test.npz`` is not cached -- the message points
            at :mod:`build_dataset`.
    """
    cache_dir = _data_root(config) / str(benchmark) / str(data_tag)
    test_npz = cache_dir / "test.npz"
    if not test_npz.is_file():
        raise FileNotFoundError(
            f"cached test split not found: {test_npz}\n"
            f"Build it first, e.g.:\n"
            f"  python -m model.vae_teb_prediction.model.model_experiment."
            f"synthetic.build_dataset --tag {data_tag}"
        )
    dataset = SyntheticTEDataset(test_npz)
    loader = make_dataloader(dataset, batch_size, shuffle=False, drop_last=False)
    return loader, dataset.meta


# =============================================================================
# Diagnostics (task 4.2)
# =============================================================================

def shuffle_source_batch(batch: Any) -> AttributeDict:
    r"""Permute the source streams of a batch along the batch axis.

    Produces a shuffled-source null control: the source $U$ of sample $b$ is
    paired with the target $Y$ of a different sample, destroying any genuine
    $U \to Y$ transfer while leaving the marginals untouched. A faithful TE
    surrogate should then collapse $\bar K$ towards zero. No retraining is
    needed -- this is the cheap task-4.3 control.

    Args:
        batch: A batched :class:`AttributeDict` with ``up_st`` / ``up_ph``.

    Returns:
        A new :class:`AttributeDict` sharing ``fhr_st`` / ``fhr_ph`` /
        ``weight`` with ``batch`` but with ``up_st`` / ``up_ph`` permuted.
    """
    shuffled = AttributeDict(batch)
    bs = int(batch.up_st.size(0))
    perm = torch.randperm(bs, device=batch.up_st.device)
    if bs > 1:
        identity = torch.arange(bs, device=perm.device)
        if bool((perm == identity).all()):
            # Guard the identity permutation (likely on a tiny last batch).
            perm = torch.roll(perm, 1)
    shuffled["up_st"] = batch.up_st[perm]
    shuffled["up_ph"] = batch.up_ph[perm]
    return shuffled


@torch.no_grad()
def _collect_diagnostics(
    model: SeqVaeLagAttnV1,
    loader: Any,
    device: torch.device,
    *,
    warmup: int,
) -> Dict[str, Any]:
    r"""Collect the task-4.2 diagnostics not returned by the reused helpers.

    One ``eval``-mode encoder pass over ``loader`` accumulating, as
    size-weighted means over valid anchors $t \in [\text{warmup}, T)$:

    * ``per_dim_kl`` -- per-latent-dim KL, a length-$d_z$ vector. Reveals
      whether the bottleneck uses a few dimensions or none (collapse).
    * ``mu_post_prior_gap`` -- mean $|\mu^q - \mu^p|$, the posterior-delta
      magnitude. Near zero means the posterior never departs from the prior.
    * ``attn_entropy`` -- mean Shannon entropy (nats) of the 91-lag attention
      distribution per head. Near $\ln 91 \approx 4.51$ means diffuse,
      untrained attention; lower means sharper lag selection.
    * ``k_bar_shuffled`` -- $\bar K$ recomputed with :func:`shuffle_source_batch`
      (the shuffled-source null control).

    Args:
        model: The trained model (moved to ``device``).
        loader: A test :class:`DataLoader`.
        device: Compute device.
        warmup: Number of leading time steps excluded from every aggregate.

    Returns:
        Dict with keys ``per_dim_kl`` (list of float), ``mu_post_prior_gap``,
        ``attn_entropy`` and ``k_bar_shuffled`` (all float).
    """
    model.eval()
    acc_per_dim: Optional[torch.Tensor] = None
    acc_mu_gap = 0.0
    acc_attn_ent = 0.0
    acc_k_shuffled = 0.0
    n_samples = 0

    for batch in loader:
        batch = tm.move_batch(batch, device)
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        u_stream = build_u_stream(batch)
        bs = int(y_st.size(0))

        # Encoder + posterior only -- no decoders needed for these diagnostics.
        enc = model.encode_only(y_st, y_ph, u_stream, sample_z=True)

        # Per-dim closed-form KL; warmup steps are NaN-masked then nan-averaged.
        kld = model.kld_tensor(
            enc["mu_prior"], enc["logvar_prior"],
            enc["mu_post"], enc["logvar_post"],
            mask_warmup=True,
        )  # (B, T, d_z)
        per_dim = torch.nanmean(kld, dim=(0, 1))  # (d_z,)

        # Posterior-delta magnitude over valid anchors.
        delta = (enc["mu_post"] - enc["mu_prior"]).abs()
        if warmup > 0:
            delta = delta[:, warmup:, :]
        mu_gap = float(delta.mean())

        # Attention lag entropy: -sum_l p log p over the 91-lag axis, then mean
        # over heads and valid anchors. torch.special.entr handles p=0 cleanly.
        attn = enc["attn_weights"]  # (B, T, num_heads, L)
        if warmup > 0:
            attn = attn[:, warmup:, :, :]
        ent = torch.special.entr(attn.clamp_min(0.0)).sum(dim=-1)  # (B, T', H)
        attn_ent = float(ent.mean())

        # Shuffled-source K_bar control.
        shuffled = shuffle_source_batch(batch)
        u_shuffled = build_u_stream(shuffled)
        k_shuffled = float(
            model.measure_transfer_entropy(
                y_st, y_ph, u_shuffled, reduce_mean=True
            )
        )

        weighted = per_dim * bs
        acc_per_dim = weighted if acc_per_dim is None else acc_per_dim + weighted
        acc_mu_gap += mu_gap * bs
        acc_attn_ent += attn_ent * bs
        acc_k_shuffled += k_shuffled * bs
        n_samples += bs

    if n_samples == 0:
        return {
            "per_dim_kl": [],
            "mu_post_prior_gap": float("nan"),
            "attn_entropy": float("nan"),
            "k_bar_shuffled": float("nan"),
        }
    return {
        "per_dim_kl": (acc_per_dim / n_samples).cpu().tolist(),
        "mu_post_prior_gap": acc_mu_gap / n_samples,
        "attn_entropy": acc_attn_ent / n_samples,
        "k_bar_shuffled": acc_k_shuffled / n_samples,
    }


# =============================================================================
# Single-checkpoint evaluation (tasks 4.1, 4.2)
# =============================================================================

def evaluate_checkpoint(
    ckpt_path: Any,
    config: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    data_tag: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Evaluate one trained checkpoint on its cached test split.

    Computes the TE surrogate $\bar K$ (task 4.1), the predictive gain and
    losses, and the task-4.2 diagnostics, returning one flat metrics row.

    Args:
        ckpt_path: Path to a ``.ckpt`` written by :func:`train_minimal.train`.
        config: The parsed ``config_synth.yaml`` (used for ``paths`` and
            optimiser-loss fallbacks only).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        data_tag: Test-split tag to evaluate against. Defaults to the tag the
            checkpoint was trained on (``ckpt["data_meta"]["tag"]``).
        batch_size: Inference batch size. Defaults to the checkpoint's training
            batch size.

    Returns:
        A flat metrics dict with the :data:`_SUMMARY_FIELDS` keys plus
        ``benchmark`` and ``per_dim_kl``.
    """
    device = device or tm.resolve_device(config["runtime"])
    model, ckpt = load_eval_checkpoint(ckpt_path, device)

    data_meta: Dict[str, Any] = ckpt.get("data_meta", {}) or {}
    ckpt_config: Dict[str, Any] = ckpt.get("config", {}) or {}
    ckpt_exp = ckpt_config.get("experiment", {})

    benchmark = str(ckpt_exp.get("benchmark", config["experiment"]["benchmark"]))
    tag = data_tag or data_meta.get("tag") or ckpt_exp.get("tag")
    if tag is None:
        raise ValueError(
            f"cannot resolve a test-split tag for {ckpt_path}: the checkpoint "
            f"carries no data_meta['tag']; pass an explicit data_tag."
        )
    if batch_size is None:
        batch_size = int(
            ckpt_config.get("optim", {}).get(
                "batch_size", config["optim"]["batch_size"]
            )
        )

    test_loader, test_meta = make_test_loader(
        config, benchmark, str(tag), int(batch_size)
    )

    # Provenance check -- warn (do not fail) if the test split's analytic TE
    # disagrees with what the checkpoint recorded.
    te_ckpt = data_meta.get("te_true")
    te_test = test_meta.get("te_true")
    if te_ckpt is not None and te_test is not None:
        if abs(float(te_ckpt) - float(te_test)) > 1e-6:
            print(
                f"[warn] checkpoint te_true={te_ckpt} != test-split "
                f"te_true={te_test} for tag '{tag}'. Using the test split's "
                f"value; check that --data-tag points at the intended dataset."
            )

    # Loss settings: prefer what the checkpoint trained with.
    loss_settings = ckpt.get("loss_settings", {}) or {}
    beta = float(loss_settings.get("beta", config["loss"]["kld_beta"]))
    lambda_full = float(
        loss_settings.get("lambda_full", config["loss"]["lambda_full"])
    )
    lambda_base = float(
        loss_settings.get("lambda_base", config["loss"]["lambda_base"])
    )

    # Task 4.1 -- the TE surrogate, mean per-step KL over valid anchors.
    k_bar = tm.compute_kbar(model, test_loader, device)
    # Task 4.2 -- pred_gap and the forecast losses, via the reused helper.
    eval_m = tm.evaluate(
        model, test_loader, device,
        beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
    )
    # Task 4.2 -- the encoder-side diagnostics.
    warmup = int(getattr(model, "warmup_period", 0) or 0)
    diag = _collect_diagnostics(model, test_loader, device, warmup=warmup)

    te_true = float(
        test_meta.get("te_true", data_meta.get("te_true", float("nan")))
    )
    te_per_step = float(
        test_meta.get("te_per_step", data_meta.get("te_per_step", float("nan")))
    )
    run_tag = Path(ckpt_path).resolve().parent.name

    row: Dict[str, Any] = {
        "run_tag": run_tag,
        "data_tag": str(tag),
        "benchmark": benchmark,
        "a": data_meta.get("a", test_meta.get("a")),
        "q": data_meta.get("q", test_meta.get("q")),
        "rho": data_meta.get("rho", test_meta.get("rho")),
        "M": data_meta.get("M", test_meta.get("M")),
        "te_true": te_true,
        "te_per_step": te_per_step,
        "k_bar": float(k_bar),
        "k_bar_shuffled": float(diag["k_bar_shuffled"]),
        "pred_gap": float(eval_m["pred_gap"]),
        "feat_loss": float(eval_m["feat_loss"]),
        "base_loss": float(eval_m["base_loss"]),
        "kld_loss": float(eval_m["kld_loss"]),
        "mu_post_prior_gap": float(diag["mu_post_prior_gap"]),
        "attn_entropy": float(diag["attn_entropy"]),
        "per_dim_kl": diag["per_dim_kl"],
        "n_test": int(len(test_loader.dataset)),
        "warmup": warmup,
        "epoch": ckpt.get("epoch"),
        "latent_stats_fitted": bool(ckpt.get("latent_stats_fitted", False)),
        "ckpt_path": str(Path(ckpt_path).resolve()),
    }
    print(
        f"[eval] {run_tag}: K_bar={row['k_bar']:.5f}  "
        f"K_shuffled={row['k_bar_shuffled']:.5f}  "
        f"pred_gap={row['pred_gap']:+.5f}  "
        f"te_true={te_true:.4f} nats  "
        f"epoch={row['epoch']}  n_test={row['n_test']}"
    )
    return row


# =============================================================================
# Pure metric helpers (tasks 4.3-4.6)
# =============================================================================

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank transform with tie handling (1-based ranks).

    Args:
        x: A 1-D array.

    Returns:
        Array of the same length holding each element's average rank; tied
        values share the mean of the ranks they span.
    """
    x = np.asarray(x, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    avg_rank = start + (counts + 1.0) / 2.0
    return avg_rank[inverse]


def _spearman_rho(x: Any, y: Any) -> float:
    r"""Spearman rank correlation $\rho$ -- numpy only, no SciPy dependency.

    Implemented as the Pearson correlation of the average-rank transforms, so
    the synthetic package stays ``numpy`` / ``torch`` only.

    Args:
        x: First sample (array-like).
        y: Second sample (array-like, same length as ``x``).

    Returns:
        $\rho \in [-1, 1]$, or ``nan`` when fewer than two finite pairs remain
        or either rank vector has zero variance (e.g. a collapsed-latent sweep
        where every $\bar K \approx 0$).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _linear_calibration(te: Any, kbar: Any) -> Tuple[float, float, float]:
    r"""Least-squares fit of $\bar K = \alpha + \gamma\,\mathrm{TE}_{\rm true}$.

    Args:
        te: Ground-truth block TE values (array-like).
        kbar: Matching $\bar K$ values (array-like).

    Returns:
        Tuple ``(alpha, gamma, r2)`` -- intercept, slope and coefficient of
        determination. All ``nan`` when fewer than two finite pairs remain or
        the TE axis has zero variance. A well-calibrated model has
        $\alpha \approx 0,\ \gamma \approx 1$.
    """
    te = np.asarray(te, dtype=float)
    kbar = np.asarray(kbar, dtype=float)
    mask = np.isfinite(te) & np.isfinite(kbar)
    te, kbar = te[mask], kbar[mask]
    if te.size < 2 or np.ptp(te) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    gamma, alpha = np.polyfit(te, kbar, 1)
    pred = gamma * te + alpha
    ss_res = float(np.sum((kbar - pred) ** 2))
    ss_tot = float(np.sum((kbar - kbar.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(alpha), float(gamma), float(r2)


def _aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Aggregate per-setting rows into Metrics 1-4.

    Args:
        rows: Per-checkpoint metrics rows from :func:`evaluate_checkpoint`.

    Returns:
        A nested dict with ``metric1_null``, ``metric2_spearman``,
        ``metric3_calibration`` and ``metric4_pred_gain``. Components that need
        the full A-sweep degrade to ``nan`` when only a few rows are present.
    """
    n = len(rows)
    te = np.array([r["te_true"] for r in rows], dtype=float)
    kbar = np.array([r["k_bar"] for r in rows], dtype=float)
    pred_gap = np.array([r["pred_gap"] for r in rows], dtype=float)
    k_shuffled = np.array([r["k_bar_shuffled"] for r in rows], dtype=float)

    is_null = np.abs(te) < 1e-9
    is_signal = te > 1e-9

    # Metric 1 -- null error.
    null_kbar = kbar[is_null]
    e0 = float(np.mean(np.abs(null_kbar))) if null_kbar.size else float("nan")
    signal_kbar = kbar[is_signal]
    smallest_signal = (
        float(np.min(signal_kbar)) if signal_kbar.size else float("nan")
    )
    null_signal_ratio = (
        e0 / smallest_signal
        if (np.isfinite(e0) and np.isfinite(smallest_signal)
            and smallest_signal > 1e-12)
        else float("nan")
    )

    # Metric 2 -- monotonicity / rank.
    rho = _spearman_rho(te, kbar)

    # Metric 3 -- calibration slope.
    alpha, gamma, r2 = _linear_calibration(te, kbar)

    # Metric 4 -- predictive gain.
    pg_null = pred_gap[is_null]
    pg_signal = pred_gap[is_signal]
    mean_pg_null = float(np.mean(pg_null)) if pg_null.size else float("nan")
    mean_pg_signal = (
        float(np.mean(pg_signal)) if pg_signal.size else float("nan")
    )

    return {
        "n_settings": n,
        "metric1_null": {
            "E_0": e0,
            "smallest_signal_k_bar": smallest_signal,
            "null_signal_ratio": null_signal_ratio,
            "k_bar_shuffled_mean": (
                float(np.nanmean(k_shuffled)) if k_shuffled.size
                else float("nan")
            ),
        },
        "metric2_spearman": rho,
        "metric3_calibration": {"alpha": alpha, "gamma": gamma, "r2": r2},
        "metric4_pred_gain": {
            "mean_pred_gap_te0": mean_pg_null,
            "mean_pred_gap_te_pos": mean_pg_signal,
            "verdict_te0_near_zero": (
                bool(abs(mean_pg_null) < 1e-2)
                if np.isfinite(mean_pg_null) else None
            ),
            "verdict_te_pos_positive": (
                bool(mean_pg_signal > 1e-2)
                if np.isfinite(mean_pg_signal) else None
            ),
        },
    }


# =============================================================================
# Sweep enumeration & opt-in orchestration (task 4.4)
# =============================================================================

def _setting_tags(benchmark: str, a: float, m: int) -> Tuple[str, str]:
    """Filesystem-safe ``(data_tag, run_tag)`` for one Gaussian sweep cell.

    Used by the Gaussian (A / B) sweep path and by
    :func:`beta_sweep.run_a_sweep_at_beta`.

    Args:
        benchmark: Benchmark identifier (e.g. ``"A"``).
        a: Transfer coefficient.
        m: Number of informative channels.

    Returns:
        Tuple ``(data_tag, run_tag)``, e.g. ``a=0.2791, M=4`` ->
        ``("benchmark_A_sweep_a0p2791_m4", "sweep_a0p2791_m4")``.
    """
    a_token = f"a{a:g}".replace(".", "p").replace("-", "m")
    data_tag = f"benchmark_{benchmark}_sweep_{a_token}_m{int(m)}"
    run_tag = f"sweep_{a_token}_m{int(m)}"
    return data_tag, run_tag


def _setting_tags_xor(benchmark: str, q: float, m: int) -> Tuple[str, str]:
    """Filesystem-safe ``(data_tag, run_tag)`` for one XOR sweep cell.

    Args:
        benchmark: Benchmark identifier (e.g. ``"C"``).
        q: Bit-flip probability.
        m: Number of informative channels.

    Returns:
        Tuple ``(data_tag, run_tag)``, e.g. ``q=0.25, M=4`` ->
        ``("benchmark_C_sweep_q0p25_m4", "sweep_q0p25_m4")``.
    """
    q_token = f"q{q:g}".replace(".", "p").replace("-", "m")
    data_tag = f"benchmark_{benchmark}_sweep_{q_token}_m{int(m)}"
    run_tag = f"sweep_{q_token}_m{int(m)}"
    return data_tag, run_tag


def enumerate_sweep(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""Enumerate the active benchmark's sweep grid with analytic ground-truth TE.

    Dispatches on ``config["sweep"]["kind"]`` (Phase 7):

    * ``gaussian`` (A / B) -- crosses ``a_grid`` x ``m_grid`` with the
      closed-form block TE $\mathrm{TE}^{(H)} = \frac{H}{2} M
      \ln(1 + a^2/\sigma^2)$.
    * ``xor`` (C) -- crosses ``q_grid`` x ``m_grid`` with
      $\mathrm{TE}^{(H)} = M H (\ln 2 - h_b(q))$.
    * ``two_lag`` (E) -- a single setting whose TE is the additive sum of the
      two bands.

    Args:
        config: The parsed (benchmark-resolved) config -- reads ``sweep``,
            ``data`` and ``model.horizon``.

    Returns:
        A list of setting dicts, each carrying ``kind``, ``M``, ``te_true``,
        ``te_per_step`` and (for Gaussian / XOR) the swept knob ``a`` / ``q``.
    """
    sweep = config.get("sweep", {}) or {}
    kind = str(sweep.get("kind", "gaussian")).lower()
    horizon = int(config["model"]["horizon"])
    data = config.get("data", {}) or {}

    settings: List[Dict[str, Any]] = []
    if kind == "xor":
        for m in sweep.get("m_grid", []):
            for q in sweep.get("q_grid", []):
                te = te_block_xor(float(q), horizon, int(m))
                settings.append({
                    "kind": "xor", "q": float(q), "M": int(m),
                    "te_true": float(te), "te_per_step": float(te) / horizon,
                })
    elif kind == "two_lag":
        sigma2 = float(data["sigma2"])
        m1, m2 = int(data["M1"]), int(data["M2"])
        te1 = te_block_gaussian(float(data["a1"]), sigma2, horizon, m1)
        te2 = te_block_gaussian(float(data["a2"]), sigma2, horizon, m2)
        te = te1 + te2
        settings.append({
            "kind": "two_lag", "M": m1 + m2,
            "te_true": float(te), "te_per_step": float(te) / horizon,
        })
    else:  # gaussian (A / B) -- the default.
        sigma2 = float(data["sigma2"])
        for m in sweep.get("m_grid", []):
            for a in sweep.get("a_grid", []):
                te = te_block_gaussian(float(a), sigma2, horizon, int(m))
                settings.append({
                    "kind": "gaussian", "a": float(a), "M": int(m),
                    "te_true": float(te), "te_per_step": float(te) / horizon,
                })
    return settings


def _ensure_dataset(
    config: Dict[str, Any], data_tag: str, a: float, m: int
) -> None:
    """Build a Gaussian sweep dataset cell if not already cached (opt-in).

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag for this cell.
        a: Transfer coefficient.
        m: Number of informative channels.
    """
    cfg = deepcopy(config)
    bd._apply_overrides(cfg, {"tag": data_tag, "a": a, "m": m})
    bd.build_dataset(cfg, force=False)


def _ensure_dataset_xor(
    config: Dict[str, Any], data_tag: str, q: float, m: int
) -> None:
    """Build an XOR sweep dataset cell if not already cached (opt-in).

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag for this cell.
        q: Bit-flip probability.
        m: Number of informative channels.
    """
    cfg = deepcopy(config)
    bd._apply_overrides(cfg, {"tag": data_tag, "q": q, "m": m})
    bd.build_dataset(cfg, force=False)


def _ensure_dataset_simple(config: Dict[str, Any], data_tag: str) -> None:
    """Build a knob-free dataset (e.g. the two-lag E cell) if not cached.

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag for this cell -- the only override; the DGP
            parameters come straight from the active benchmark's ``data`` block.
    """
    cfg = deepcopy(config)
    bd._apply_overrides(cfg, {"tag": data_tag})
    bd.build_dataset(cfg, force=False)


def _ensure_checkpoint(
    config: Dict[str, Any], data_tag: str, run_tag: str
) -> None:
    """Train a sweep checkpoint via :func:`train_minimal.train` (opt-in).

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag of the dataset to train on.
        run_tag: Results subdirectory name for this run.
    """
    cfg = deepcopy(config)
    tm.train(cfg, overrides={"data_tag": data_tag, "run_tag": run_tag})


# =============================================================================
# Output: CSV / JSON / plots (tasks 4.7, 4.8)
# =============================================================================

def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the per-setting summary CSV (task 4.8).

    Args:
        rows: Per-checkpoint metrics rows.
        path: Destination CSV path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _SUMMARY_FIELDS})


def write_metrics_json(
    metrics: Dict[str, Any], rows: List[Dict[str, Any]], path: Path
) -> None:
    """Write the aggregated Metrics 1-4 plus per-dim KL vectors as JSON.

    Args:
        metrics: The :func:`_aggregate_metrics` output.
        rows: Per-checkpoint metrics rows (the ``per_dim_kl`` vectors live
            here, not in the flat CSV).
        path: Destination JSON path (overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "n_evaluated": len(rows),
        "metrics": metrics,
        "per_dim_kl": {r["run_tag"]: r.get("per_dim_kl", []) for r in rows},
        "rows": [{k: r.get(k) for k in _SUMMARY_FIELDS} for r in rows],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _make_plots(
    rows: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: Path
) -> None:
    r"""Render the three headline Phase-4 plots (task 4.7).

    Produces ``kbar_vs_te.{pdf,png}`` (with the calibration line),
    ``kbar_vs_<knob>.{pdf,png}`` (one series per $M$, the knob being $a$ for
    Gaussian benchmarks or $q$ for XOR) and ``predgap_vs_kbar.{pdf,png}``. All
    figures use the shared publication style in :mod:`plot_style`.

    Args:
        rows: Per-checkpoint metrics rows (>= 2 expected).
        metrics: The :func:`_aggregate_metrics` output.
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()

    def _m_color(idx: int) -> str:
        """Pick a stable palette colour for the ``idx``-th $M$ level."""
        return ps.PALETTE_EXTENDED[idx % len(ps.PALETTE_EXTENDED)]

    te = np.array([r["te_true"] for r in rows], dtype=float)
    kbar = np.array([r["k_bar"] for r in rows], dtype=float)
    pred_gap = np.array([r["pred_gap"] for r in rows], dtype=float)
    a_vals = np.array(
        [float(r["a"]) if r.get("a") is not None else np.nan for r in rows],
        dtype=float,
    )
    q_vals = np.array(
        [float(r["q"]) if r.get("q") is not None else np.nan for r in rows],
        dtype=float,
    )
    # Plot 2 sweeps whichever knob the active benchmark actually varied.
    if np.isfinite(a_vals).any():
        knob_vals, knob_name = a_vals, "a"
    elif np.isfinite(q_vals).any():
        knob_vals, knob_name = q_vals, "q"
    else:
        knob_vals, knob_name = a_vals, "a"
    m_vals = np.array(
        [int(r["M"]) if r["M"] is not None else -1 for r in rows], dtype=int
    )
    m_levels = sorted(set(m_vals.tolist()))

    # --- Plot 1: K_bar vs true TE + calibration line ----------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for idx, m in enumerate(m_levels):
        sel = m_vals == m
        ax.scatter(te[sel], kbar[sel], s=44, color=_m_color(idx),
                   edgecolors=ps.COLOR_BLACK, linewidths=0.4,
                   zorder=3, label=f"M={m}")
    cal = metrics.get("metric3_calibration", {})
    alpha, gamma = cal.get("alpha"), cal.get("gamma")
    if (alpha is not None and gamma is not None
            and np.isfinite(alpha) and np.isfinite(gamma)
            and np.isfinite(te).any()):
        xs = np.linspace(float(np.nanmin(te)), float(np.nanmax(te)), 100)
        ax.plot(
            xs, gamma * xs + alpha, ls="--", lw=1.1, color=ps.COLOR_BLACK,
            zorder=2, label=fr"fit: ${gamma:.3g}\,\mathrm{{TE}}+{alpha:.3g}$",
        )
    rho = metrics.get("metric2_spearman", float("nan"))
    title = r"$\bar K$ vs analytic block TE"
    if isinstance(rho, float) and np.isfinite(rho):
        title += fr"   (Spearman $\rho$={rho:.3f})"
    ax.set_title(title)
    ax.set_xlabel("analytic block TE (nats)")
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.legend()
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "kbar_vs_te")

    # --- Plot 2: K_bar vs the swept knob (one series per M) ---------------
    knob_label = (
        "transfer coefficient $a$" if knob_name == "a"
        else "bit-flip probability $q$"
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for idx, m in enumerate(m_levels):
        sel = m_vals == m
        order = np.argsort(knob_vals[sel])
        ax.plot(
            knob_vals[sel][order], kbar[sel][order],
            marker="o", color=_m_color(idx), label=f"M={m}",
        )
    ax.set_title(fr"$\bar K$ vs {knob_label}")
    ax.set_xlabel(knob_label)
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.legend()
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / f"kbar_vs_{knob_name}")

    # --- Plot 3: predictive gain vs K_bar ---------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0.0, color=ps.COLOR_GRAY, ls="--", lw=0.9, zorder=1)
    for idx, m in enumerate(m_levels):
        sel = m_vals == m
        ax.scatter(kbar[sel], pred_gap[sel], s=44, color=_m_color(idx),
                   edgecolors=ps.COLOR_BLACK, linewidths=0.4,
                   zorder=3, label=f"M={m}")
    ax.set_title(r"predictive gain vs $\bar K$")
    ax.set_xlabel(r"$\bar K$ (nats)")
    ax.set_ylabel(r"pred_gap $= \mathcal{L}_{\rm base}-\mathcal{L}_{\rm feat}$")
    ax.legend()
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "predgap_vs_kbar")


def _print_metrics(metrics: Dict[str, Any], n: int) -> None:
    """Print the aggregated Metrics 1-4 block.

    Args:
        metrics: The :func:`_aggregate_metrics` output.
        n: Number of evaluated settings.
    """
    m1 = metrics["metric1_null"]
    m3 = metrics["metric3_calibration"]
    m4 = metrics["metric4_pred_gain"]
    print(
        f"\n[metrics] {n} setting(s) evaluated\n"
        f"  Metric 1 (null)        : E_0={m1['E_0']:.5f}  "
        f"null/signal ratio={m1['null_signal_ratio']:.4f}  "
        f"K_shuffled(mean)={m1['k_bar_shuffled_mean']:.5f}\n"
        f"  Metric 2 (Spearman rho): {metrics['metric2_spearman']:.4f}\n"
        f"  Metric 3 (calibration) : alpha={m3['alpha']:.5f}  "
        f"gamma={m3['gamma']:.5f}  R^2={m3['r2']:.4f}\n"
        f"  Metric 4 (pred. gain)  : mean pred_gap  TE=0 -> "
        f"{m4['mean_pred_gap_te0']:+.5f}   TE>0 -> "
        f"{m4['mean_pred_gap_te_pos']:+.5f}"
    )


# =============================================================================
# Sweep aggregation entry point (task 4.4)
# =============================================================================

def run_sweep(
    config: Dict[str, Any],
    *,
    build_missing: bool = False,
    train_missing: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Evaluate the whole A-sweep and aggregate Metrics 1-4.

    Enumerates the ``a_sweep`` grid, evaluates every cell whose dataset and
    checkpoint exist, and writes ``summary.csv`` / ``metrics.json`` / the three
    plots under ``results/<benchmark>/eval_te/``.

    Missing datasets and checkpoints are **skipped** unless the corresponding
    opt-in flag is set -- ``build_missing`` generates the dataset (a fast
    :func:`build_dataset.build_dataset` call) and ``train_missing`` trains the
    checkpoint (a multi-hour :func:`train_minimal.train` call). Both default to
    ``False`` so the harness, by default, only evaluates what is on disk.

    Args:
        config: The parsed ``config_synth.yaml``.
        build_missing: If True, generate any missing sweep dataset.
        train_missing: If True, train any missing sweep checkpoint.
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.

    Returns:
        A results dict: ``rows``, ``metrics``, ``out_dir``, ``skipped``.
    """
    benchmark = str(config["experiment"]["benchmark"])
    device = device or tm.resolve_device(config["runtime"])
    out_dir = _eval_out_dir(config, benchmark)
    data_root = _data_root(config)
    results_root = _results_root(config)

    settings = enumerate_sweep(config)
    print(
        f"[sweep] benchmark {benchmark}: {len(settings)} sweep setting(s)  "
        f"build_missing={build_missing}  train_missing={train_missing}"
    )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for setting in settings:
        kind = str(setting.get("kind", "gaussian"))
        m = setting["M"]
        if kind == "xor":
            q = setting["q"]
            data_tag, run_tag = _setting_tags_xor(benchmark, q, m)
            label = f"q={q:g}, M={m}"
        elif kind == "two_lag":
            data_tag = str(config["experiment"]["tag"])
            run_tag = "sweep_two_lag"
            label = f"two-lag, M={m}"
        else:  # gaussian
            a = setting["a"]
            data_tag, run_tag = _setting_tags(benchmark, a, m)
            label = f"a={a:g}, M={m}"
        cache_dir = data_root / benchmark / data_tag
        ckpt_path = results_root / benchmark / run_tag / "final.ckpt"

        if not (cache_dir / "test.npz").is_file():
            if build_missing:
                print(f"  [build] {data_tag}  ({label})")
                if kind == "xor":
                    _ensure_dataset_xor(config, data_tag, q, m)
                elif kind == "two_lag":
                    _ensure_dataset_simple(config, data_tag)
                else:
                    _ensure_dataset(config, data_tag, a, m)
            else:
                print(
                    f"  [skip ] {run_tag}: dataset '{data_tag}' not cached "
                    f"(pass --build-missing to generate it)"
                )
                skipped.append(run_tag)
                continue

        if not ckpt_path.is_file():
            if train_missing:
                # ``label`` is kind-aware (a= / q= / two-lag); ``a`` is only
                # bound on the Gaussian branch, so a non-Gaussian sweep would
                # otherwise NameError here.
                print(f"  [train] {run_tag}  ({label})")
                _ensure_checkpoint(config, data_tag, run_tag)
            else:
                print(
                    f"  [skip ] {run_tag}: checkpoint not found "
                    f"({ckpt_path}) (pass --train-missing to train it)"
                )
                skipped.append(run_tag)
                continue

        try:
            row = evaluate_checkpoint(
                ckpt_path, config, device=device, data_tag=data_tag
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not abort
            print(f"  [error] {run_tag}: {type(exc).__name__}: {exc}")
            skipped.append(run_tag)
            continue

    metrics = _aggregate_metrics(rows)
    metrics["n_skipped"] = len(skipped)
    write_summary_csv(rows, out_dir / "summary.csv")
    write_metrics_json(metrics, rows, out_dir / "metrics.json")
    if len(rows) >= 2:
        _make_plots(rows, metrics, out_dir)
    else:
        print(
            "[plots] fewer than 2 settings evaluated -- skipping plots "
            "(the A-sweep training run is deferred; see the plan)."
        )

    _print_metrics(metrics, n=len(rows))
    print(
        f"[done] sweep eval: {len(rows)} evaluated, {len(skipped)} skipped\n"
        f"       artifacts -> {out_dir}"
    )
    return {
        "rows": rows,
        "metrics": metrics,
        "out_dir": str(out_dir),
        "skipped": skipped,
    }


# =============================================================================
# Benchmark-B rho-null diagnostic (task 7.1)
# =============================================================================

def run_rho_null_check(
    config: Dict[str, Any],
    *,
    build_missing: bool = False,
    train_missing: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Benchmark-B headline test: $\bar K \approx 0$ at $a=0$ for every $\rho$.

    Benchmark B's AR target carries strong self-predictability. The residual
    bottleneck should absorb that target self-information into the baseline
    branch, leaving $\bar K \approx 0$ when there is no source transfer
    ($a=0$) -- *even for a highly autocorrelated target*. This routine builds
    (opt-in) one zero-transfer B dataset per $\rho$ in ``rho_null.rho_grid``,
    trains (opt-in) a model on each, evaluates $\bar K$ and checks that it stays
    small across the whole $\rho$ range.

    Args:
        config: The parsed config (must have a ``rho_null`` block -- only
            Benchmark B defines one).
        build_missing: If True, generate any missing $\rho$-null dataset.
        train_missing: If True, train any missing $\rho$-null checkpoint.
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.

    Returns:
        A results dict: ``rows``, ``out_dir``, ``skipped``, ``verdict``.

    Raises:
        ValueError: If the config carries no ``rho_null.rho_grid``.
    """
    benchmark = str(config["experiment"]["benchmark"])
    device = device or tm.resolve_device(config["runtime"])
    rho_grid = list((config.get("rho_null", {}) or {}).get("rho_grid", []))
    if not rho_grid:
        raise ValueError(
            "run_rho_null_check: config has no rho_null.rho_grid -- only "
            "Benchmark B defines one (set experiment.benchmark: B)."
        )

    out_dir = _results_root(config) / benchmark / "rho_null"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = _data_root(config)
    results_root = _results_root(config)
    print(
        f"[rho_null] benchmark {benchmark}: {len(rho_grid)} rho value(s) at "
        f"a=0  build_missing={build_missing}  train_missing={train_missing}"
    )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for rho in rho_grid:
        token = f"r{float(rho):g}".replace(".", "p").replace("-", "m")
        data_tag = f"benchmark_{benchmark}_rho_null_{token}"
        run_tag = f"rho_null/{token}"
        cache_dir = data_root / benchmark / data_tag
        ckpt_path = results_root / benchmark / run_tag / "final.ckpt"

        if not (cache_dir / "test.npz").is_file():
            if build_missing:
                print(f"  [build] {data_tag}  (a=0, rho={rho:g})")
                cfg = deepcopy(config)
                bd._apply_overrides(
                    cfg, {"tag": data_tag, "a": 0.0, "rho": float(rho)}
                )
                bd.build_dataset(cfg, force=False)
            else:
                print(f"  [skip ] {run_tag}: dataset '{data_tag}' not cached "
                      f"(pass --build-missing to generate it)")
                skipped.append(run_tag)
                continue

        if not ckpt_path.is_file():
            if train_missing:
                print(f"  [train] {run_tag}  (a=0, rho={rho:g})")
                cfg = deepcopy(config)
                tm.train(cfg, overrides={
                    "data_tag": data_tag, "run_tag": run_tag, "a": 0.0,
                })
            else:
                print(f"  [skip ] {run_tag}: checkpoint not found ({ckpt_path}) "
                      f"(pass --train-missing to train it)")
                skipped.append(run_tag)
                continue

        try:
            row = evaluate_checkpoint(
                ckpt_path, config, device=device, data_tag=data_tag
            )
            row["rho"] = float(rho)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not abort
            print(f"  [error] {run_tag}: {type(exc).__name__}: {exc}")
            skipped.append(run_tag)

    kbars = np.array([r["k_bar"] for r in rows], dtype=float)
    max_abs_kbar = float(np.nanmax(np.abs(kbars))) if kbars.size else float("nan")
    verdict = {
        "n_rho": len(rows),
        "max_abs_k_bar": max_abs_kbar,
        # All rho cells have a=0 -> te_true=0; a faithful surrogate keeps K
        # small regardless of rho. 0.05 nats mirrors the model_validation null
        # threshold (a heuristic gate -- see plan Section 8 / Decision D8).
        "verdict_no_leakage": (
            bool(max_abs_kbar < 0.05) if np.isfinite(max_abs_kbar) else None
        ),
    }
    write_summary_csv(rows, out_dir / "summary.csv")
    with open(out_dir / "rho_null.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "created": datetime.now(timezone.utc).isoformat(),
                "verdict": verdict,
                "rows": [{k: r.get(k) for k in _SUMMARY_FIELDS} for r in rows],
            },
            fh, indent=2,
        )
    print(
        f"[done] rho_null: {len(rows)} evaluated, {len(skipped)} skipped  "
        f"max|K_bar|={max_abs_kbar:.5f}  "
        f"no-leakage verdict={verdict['verdict_no_leakage']}\n"
        f"       artifacts -> {out_dir}"
    )
    return {
        "rows": rows, "out_dir": str(out_dir),
        "skipped": skipped, "verdict": verdict,
    }


# =============================================================================
# Overrides + dispatch
# =============================================================================

def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply the config-level overrides onto ``config`` in place.

    Only ``benchmark`` / ``device`` / ``seed`` are config fields; the other
    overrides (``mode``, ``checkpoint``, ``data_tag``, ``batch_size``,
    ``build_missing``, ``train_missing``) are passed as call arguments by
    :func:`_dispatch` and are ignored here.

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
        The mode-specific results dict (see :func:`run_sweep`; ``single`` mode
        returns ``rows`` / ``metrics`` / ``out_dir``).

    Raises:
        ValueError: On an unknown ``mode`` or a missing ``checkpoint`` in
            ``single`` mode.
    """
    config = deepcopy(config)
    _apply_overrides(config, overrides)
    tm.set_seed(int(config["experiment"].get("seed", 0)))
    device = tm.resolve_device(config["runtime"])
    mode = str(overrides.get("mode") or "single").lower()

    if mode == "single":
        ckpt = overrides.get("checkpoint")
        if not ckpt:
            raise ValueError(
                "single mode requires a checkpoint -- pass --checkpoint PATH "
                "(CLI) or set RUN_CONFIG['checkpoint'] (edit-and-run)."
            )
        row = evaluate_checkpoint(
            ckpt, config, device=device,
            data_tag=overrides.get("data_tag"),
            batch_size=overrides.get("batch_size"),
        )
        out_dir = _eval_out_dir(config, str(row["benchmark"]))
        metrics = _aggregate_metrics([row])
        write_summary_csv([row], out_dir / "summary.csv")
        write_metrics_json(metrics, [row], out_dir / "metrics.json")
        _print_metrics(metrics, n=1)
        print(f"[done] single-checkpoint eval -> {out_dir}")
        return {"rows": [row], "metrics": metrics, "out_dir": str(out_dir)}

    if mode == "sweep":
        return run_sweep(
            config,
            build_missing=bool(overrides.get("build_missing")),
            train_missing=bool(overrides.get("train_missing")),
            device=device,
        )

    if mode == "rho_null":
        return run_rho_null_check(
            config,
            build_missing=bool(overrides.get("build_missing")),
            train_missing=bool(overrides.get("train_missing")),
            device=device,
        )

    raise ValueError(
        f"unknown mode: {mode!r} (expected 'single', 'sweep' or 'rho_null')."
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
        description="Transfer-entropy evaluation harness for SeqVaeLagAttnV1 "
                    "on synthetic benchmark data (Phase 4)."
    )
    p.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="path to config_synth.yaml",
    )
    p.add_argument(
        "--mode", type=str, default=None,
        choices=["single", "sweep", "rho_null"],
        help="single-checkpoint evaluation, full sweep aggregation, or the "
             "Benchmark-B rho-null diagnostic",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="path to the .ckpt to evaluate (required for --mode single)",
    )
    p.add_argument(
        "--data-tag", type=str, default=None, dest="data_tag",
        help="test-split tag to evaluate against (defaults to the tag the "
             "checkpoint was trained on)",
    )
    p.add_argument(
        "--benchmark", type=str, default=None,
        help="override experiment.benchmark (which benchmark to sweep)",
    )
    p.add_argument(
        "--build-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="build_missing",
        help="sweep mode: generate any missing benchmark dataset (opt-in)",
    )
    p.add_argument(
        "--train-missing", action=argparse.BooleanOptionalAction, default=False,
        dest="train_missing",
        help="sweep mode: train any missing checkpoint (opt-in, multi-hour)",
    )
    p.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="inference batch size (defaults to the checkpoint's training "
             "batch size)",
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
        "mode": "single",                   # "single" | "sweep" | "rho_null"
        # single mode: which checkpoint to evaluate.
        "checkpoint": str(
            _EXPERIMENT_DIR / "results" / "A" / "pol_easy_a1" / "final.ckpt"
        ),
        "data_tag": None,                   # None -> the checkpoint's own tag
        "benchmark": None,                  # None -> config experiment.benchmark
        "build_missing": False,             # sweep mode: build missing datasets
        "train_missing": False,             # sweep mode: train missing ckpts
        "batch_size": None,                 # None -> checkpoint's batch size
        "device": None,                     # None -> config runtime.device
        "seed": None,                       # None -> config experiment.seed
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["optim"]["batch_size"] = 16
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
