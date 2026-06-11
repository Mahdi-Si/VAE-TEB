r"""TE evaluation harness -- $\bar{K}$ versus ground-truth TE (v2 benchmarks).

Loads a trained :class:`SeqVaeLagAttnV1` checkpoint, runs inference over the
cached **test** split, and computes the four validation metrics of
``model_validation_v2.md`` Section 9:

    Metric 1 -- null error: $E_0 = |\bar{K}|$ for zero-transfer data,
        plus a cheap shuffled-source control.
    Metric 2 -- monotonicity: Spearman $\rho(\bar{K}, \mathrm{TE}_{\rm true})$
        across the active benchmark's sweep.
    Metric 3 -- calibration slope: fit $\bar{K} = \alpha + \gamma\,
        \mathrm{TE}_{\rm true}$ and report $\alpha, \gamma, R^2$ (the
        $\gamma \to 1$ headline becomes meaningful in Sprint 5 once the
        Gaussian-NLL likelihood is wired).
    Metric 4 -- predictive gain: ``pred_gap`` $= \mathcal{L}_{\rm base} -
        \mathcal{L}_{\rm feat}$ versus $\bar{K}$ and $\mathrm{TE}_{\rm true}$.

$\bar{K}$ is the per-step KL $K_t$ averaged over valid anchors
$t \in [\text{warmup}, T)$, obtained via
:meth:`SeqVaeLagAttnV1.measure_transfer_entropy`. Checkpoints are reloaded with
``train/graph_models_utils.py:load_checkpoint_strict``.

This module **reuses** the training-loop helpers in :mod:`train_minimal`
(:func:`evaluate`, :func:`compute_kbar`, ...) so the harness scores models with
the exact loss / KL code training used -- it never re-implements them.

Two run modes:
    * ``single`` -- evaluate one checkpoint, write a one-row summary.
    * ``sweep``  -- enumerate the active v2 benchmark's ``sweep`` grid
      (``gaussian_state_space`` $B_y$ x M for G1, ``arx`` $c$ x M for G2,
      ``regime_switch`` $p$ x M for G3), evaluate every cell, aggregate
      Metrics 1-4, render the headline plots.

    Missing datasets / checkpoints can be built / trained behind the opt-in
    ``build_missing`` / ``train_missing`` flags (default OFF -- the harness
    otherwise strictly evaluates what already exists on disk).

Run modes (project convention -- Decision V2-D8 in
``model_validation_v2_plan.md``): like every ``synthetic/`` runner this file
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
    B_y_for_mean_te_block_state_space,
    c_for_mean_te_block_arx,
    mean_te_block_arx_over_delays,
    mean_te_block_state_space_over_delays,
    te_block_arx_gaussian,
    te_block_state_space_gaussian,
    te_categorical_switch_block,
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

# Columns of the per-setting summary CSV. One row per evaluated checkpoint;
# the d_z-length ``per_dim_kl`` vector goes to ``metrics.json``. ``B_y`` / ``c``
# / ``p_switch`` are the v2 benchmark-specific sweep knobs (G1 / G2 / G3);
# whichever the active benchmark does not use is left blank.
_SUMMARY_FIELDS = [
    "run_tag", "data_tag", "benchmark", "B_y", "c", "p_switch", "M",
    "delay_min", "delay_max", "delay_walk", "target_te",
    "te_true", "te_per_step",
    # Null-control surrogates: ``k_bar_shuffled`` permutes $U$ along the batch
    # axis, ``k_bar_reversed`` flips $U$ along $T$. Both are computed in
    # :func:`_collect_diagnostics` during the same eval pass.
    "k_bar", "k_bar_shuffled", "k_bar_reversed",
    "pred_gap", "feat_loss", "base_loss", "kld_loss",
    "mu_post_prior_gap", "attn_entropy", "n_test", "warmup", "epoch",
    # Sprint 5 calibration provenance: the likelihood + sigma_obs the
    # checkpoint was trained under, plus the collapse diagnostic for the
    # observation-noise head.
    "likelihood", "sigma_obs", "free_bits", "mean_logvar_full",
    "latent_stats_fitted", "ckpt_path",
]


# =============================================================================
# Path helpers
# =============================================================================

def _data_root(config: Dict[str, Any]) -> Path:
    """Resolve the dataset cache root from ``paths.data_dir``.

    Accepts a relative path (joined with ``model_experiment/``), an
    absolute path on any drive (used as-is), or a path with ``~`` /
    ``$VAR`` references (expanded). Delegates to
    :func:`train_minimal.resolve_user_path`.

    Args:
        config: The parsed ``config_synth.yaml``.

    Returns:
        Absolute path of the resolved data root.
    """
    return tm.resolve_user_path(config["paths"]["data_dir"])


def _results_root(config: Dict[str, Any]) -> Path:
    """Resolve the results root from ``paths.results_dir``.

    Same resolution rules as :func:`_data_root` -- relative, absolute on
    any drive, or ``~`` / ``$VAR`` patterns are all honoured.

    Args:
        config: The parsed ``config_synth.yaml``.

    Returns:
        Absolute path of the resolved results root.
    """
    return tm.resolve_user_path(config["paths"]["results_dir"])


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


def reverse_source_batch(batch: Any) -> AttributeDict:
    r"""Flip the source streams of a batch along the time axis.

    Produces a time-reversed null control: each source channel is replayed
    backwards in time so the present source value at step $t$ becomes the
    original value at step $T-1-t$, destroying any genuine
    $U_{\le t} \to Y_{t+1:t+H}$ alignment while leaving the per-channel
    marginals and autocorrelation magnitude untouched. A faithful TE surrogate
    should collapse $\bar K$ towards zero (or at least move attention to the
    wrong lags). No retraining is needed.

    The returned dict **shares references** to ``fhr_st`` / ``fhr_ph`` /
    ``weight`` with the input (shallow copy); ``up_st`` / ``up_ph`` are fresh
    tensors from :func:`torch.flip`. Callers must not in-place mutate the
    target streams. Models that read inputs without mutating them (the
    contract :class:`SeqVaeLagAttnV1` honours) are safe.

    Args:
        batch: A batched :class:`AttributeDict` with ``up_st`` / ``up_ph``.

    Returns:
        A new :class:`AttributeDict` sharing ``fhr_st`` / ``fhr_ph`` /
        ``weight`` with ``batch`` but with ``up_st`` / ``up_ph`` flipped
        along ``dim=1`` (the time axis).
    """
    reversed_b = AttributeDict(batch)
    reversed_b["up_st"] = batch.up_st.flip(dims=[1])
    reversed_b["up_ph"] = batch.up_ph.flip(dims=[1])
    return reversed_b


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
    * ``k_bar_reversed`` -- $\bar K$ recomputed with :func:`reverse_source_batch`
      (the time-reversed source null control, Sprint 6.3).

    Args:
        model: The trained model (moved to ``device``).
        loader: A test :class:`DataLoader`.
        device: Compute device.
        warmup: Number of leading time steps excluded from every aggregate.

    Returns:
        Dict with keys ``per_dim_kl`` (list of float), ``mu_post_prior_gap``,
        ``attn_entropy``, ``k_bar_shuffled`` and ``k_bar_reversed`` (all
        float).
    """
    model.eval()
    acc_per_dim: Optional[torch.Tensor] = None
    acc_mu_gap = 0.0
    acc_attn_ent = 0.0
    acc_k_shuffled = 0.0
    acc_k_reversed = 0.0
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

        # Null-control K_bar values. The shuffle / flip transforms commute
        # with ``build_u_stream`` (concat along the channel axis), so we
        # operate directly on the already-built 101-channel ``u_stream`` and
        # avoid allocating two extra AttributeDict shallow copies plus two
        # extra channel-concat tensors per batch.
        bs_idx = torch.randperm(bs, device=u_stream.device)
        if bs > 1:
            identity = torch.arange(bs, device=bs_idx.device)
            if bool((bs_idx == identity).all()):
                bs_idx = torch.roll(bs_idx, 1)
        u_shuffled = u_stream[bs_idx]
        k_shuffled = float(
            model.measure_transfer_entropy(
                y_st, y_ph, u_shuffled, reduce_mean=True
            )
        )
        u_reversed = u_stream.flip(dims=[1])
        k_reversed = float(
            model.measure_transfer_entropy(
                y_st, y_ph, u_reversed, reduce_mean=True
            )
        )

        weighted = per_dim * bs
        acc_per_dim = weighted if acc_per_dim is None else acc_per_dim + weighted
        acc_mu_gap += mu_gap * bs
        acc_attn_ent += attn_ent * bs
        acc_k_shuffled += k_shuffled * bs
        acc_k_reversed += k_reversed * bs
        n_samples += bs

    if n_samples == 0:
        return {
            "per_dim_kl": [],
            "mu_post_prior_gap": float("nan"),
            "attn_entropy": float("nan"),
            "k_bar_shuffled": float("nan"),
            "k_bar_reversed": float("nan"),
        }
    return {
        "per_dim_kl": (acc_per_dim / n_samples).cpu().tolist(),
        "mu_post_prior_gap": acc_mu_gap / n_samples,
        "attn_entropy": acc_attn_ent / n_samples,
        "k_bar_shuffled": acc_k_shuffled / n_samples,
        "k_bar_reversed": acc_k_reversed / n_samples,
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
    benchmark_override: Optional[str] = None,
    model_ckpt: Optional[Tuple[Any, Dict[str, Any]]] = None,
    loader_meta: Optional[Tuple[Any, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    r"""Evaluate one trained checkpoint on its cached test split.

    Computes the TE surrogate $\bar K$ (task 4.1), the predictive gain and
    losses, and the task-4.2 diagnostics, returning one flat metrics row.

    Args:
        ckpt_path: Path to a ``.ckpt`` written by :func:`train_minimal.train`.
            Still used for the ``run_tag`` / ``ckpt_path`` provenance fields
            even when ``model_ckpt`` is injected, so pass the real path.
        config: The parsed ``config_synth.yaml`` (used for ``paths`` and
            optimiser-loss fallbacks only).
        device: Compute device. Defaults to :func:`train_minimal.resolve_device`.
        data_tag: Test-split tag to evaluate against. Defaults to the tag the
            checkpoint was trained on (``ckpt["data_meta"]["tag"]``). When
            ``loader_meta`` is injected this is used only as the row's
            ``data_tag`` label.
        batch_size: Inference batch size. Defaults to the checkpoint's training
            batch size. Ignored when ``loader_meta`` is injected.
        benchmark_override: Optional benchmark name to use when resolving the
            test cache directory; ``None`` (default) falls back to the
            checkpoint's own benchmark. :mod:`null_controls` sets this so a
            trained checkpoint can be re-evaluated on a sibling control cache
            (e.g. G2 -> ``G2_wrong_delay`` / ``G2_zero_coupling``) without
            retraining; the row's ``benchmark`` field then records the
            override. Empty string is treated as ``None`` (falsy fallback).
        model_ckpt: Optional pre-loaded ``(model, ckpt)`` pair. When given, the
            checkpoint is **not** re-read from disk -- :mod:`mixed_per_cell_diag`
            uses this to evaluate the same model on many per-cell loaders
            without reloading it each time.
        loader_meta: Optional pre-built ``(loader, meta)`` pair. When given,
            :func:`make_test_loader` is bypassed and the on-disk single-cell
            cache requirement is lifted; the caller supplies a per-cell loader
            and a single-cell-style ``meta`` (``te_true`` / ``M`` / ``delay_*``
            / ``B_y``). The checkpoint-vs-test TE provenance warning is skipped
            in this mode (a per-cell TE deliberately differs from the pooled
            training TE).

    Returns:
        A flat metrics dict with the :data:`_SUMMARY_FIELDS` keys plus
        ``benchmark`` and ``per_dim_kl``.
    """
    device = device or tm.resolve_device(config["runtime"])
    if model_ckpt is not None:
        model, ckpt = model_ckpt
    else:
        model, ckpt = load_eval_checkpoint(ckpt_path, device)

    data_meta: Dict[str, Any] = ckpt.get("data_meta", {}) or {}
    ckpt_config: Dict[str, Any] = ckpt.get("config", {}) or {}
    ckpt_exp = ckpt_config.get("experiment", {})

    benchmark = str(
        benchmark_override
        or ckpt_exp.get("benchmark", config["experiment"]["benchmark"])
    )
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

    if loader_meta is not None:
        test_loader, test_meta = loader_meta
    else:
        test_loader, test_meta = make_test_loader(
            config, benchmark, str(tag), int(batch_size)
        )

    # Provenance check -- warn (do not fail) if the test split's analytic TE
    # disagrees with what the checkpoint recorded. Skipped under
    # ``benchmark_override``: re-evaluating a checkpoint on a sibling control
    # cache is *meant* to change te_true, so the divergence is expected. Also
    # skipped when ``loader_meta`` is injected: a per-cell TE deliberately
    # differs from the pooled training TE.
    if benchmark_override is None and loader_meta is None:
        te_ckpt = data_meta.get("te_true")
        te_test = test_meta.get("te_true")
        if te_ckpt is not None and te_test is not None:
            if abs(float(te_ckpt) - float(te_test)) > 1e-6:
                print(
                    f"[warn] checkpoint te_true={te_ckpt} != test-split "
                    f"te_true={te_test} for tag '{tag}'. Using the test "
                    f"split's value; check that --data-tag points at the "
                    f"intended dataset."
                )

    # Loss settings: prefer what the checkpoint trained with. The legacy
    # trio (beta / lambda_full / lambda_base) has been mandatory in
    # ``loss:`` since v1 and is read strictly. Sprint-5 keys
    # (``likelihood`` / ``sigma_obs`` / ``free_bits``) are optional with
    # MSE / unit-sigma / no-floor defaults, so pre-Sprint-5 checkpoints
    # score identically.
    loss_settings = ckpt.get("loss_settings", {}) or {}
    cfg_loss = config["loss"]
    beta = float(loss_settings.get("beta", cfg_loss["kld_beta"]))
    lambda_full = float(
        loss_settings.get("lambda_full", cfg_loss["lambda_full"])
    )
    lambda_base = float(
        loss_settings.get("lambda_base", cfg_loss["lambda_base"])
    )
    likelihood = str(
        loss_settings.get("likelihood", cfg_loss.get("likelihood", "mse"))
    )
    sigma_obs_raw = loss_settings.get(
        "sigma_obs", cfg_loss.get("sigma_obs", 1.0)
    )
    if isinstance(sigma_obs_raw, str) and sigma_obs_raw != "learned":
        try:
            sigma_obs: "float | str" = float(sigma_obs_raw)
        except ValueError as exc:
            raise ValueError(
                f"loss.sigma_obs must be a positive float or 'learned', "
                f"got {sigma_obs_raw!r}"
            ) from exc
    else:
        sigma_obs = (
            sigma_obs_raw if isinstance(sigma_obs_raw, str)
            else float(sigma_obs_raw)
        )
    free_bits = float(
        loss_settings.get("free_bits", cfg_loss.get("free_bits", 0.0))
    )

    # Task 4.1 -- the TE surrogate, mean per-step KL over valid anchors.
    k_bar = tm.compute_kbar(model, test_loader, device)
    # Task 4.2 -- pred_gap and the forecast losses, via the reused helper.
    eval_m = tm.evaluate(
        model, test_loader, device,
        beta=beta, lambda_full=lambda_full, lambda_base=lambda_base,
        likelihood=likelihood, sigma_obs=sigma_obs, free_bits=free_bits,
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

    # G1 sweeps over B_y (single-scalar coupling, broadcast across the per-channel
    # B_y list); G2 sweeps over c; G3 sweeps over p_switch. The CSV captures
    # whichever knob this benchmark exposes; the others stay None.
    by_val = data_meta.get("B_y", test_meta.get("B_y"))
    if isinstance(by_val, (list, tuple)) and by_val:
        by_val = float(by_val[0])  # representative coupling for a multi-osc sweep
    row: Dict[str, Any] = {
        "run_tag": run_tag,
        "data_tag": str(tag),
        "benchmark": benchmark,
        "B_y": by_val,
        "c": data_meta.get("c", test_meta.get("c")),
        "p_switch": data_meta.get("p_switch", test_meta.get("p_switch")),
        "M": data_meta.get("M", test_meta.get("M")),
        "delay_min": data_meta.get("delay_min", test_meta.get("delay_min")),
        "delay_max": data_meta.get("delay_max", test_meta.get("delay_max")),
        "delay_walk": data_meta.get("delay_walk", test_meta.get("delay_walk")),
        "te_true": te_true,
        "te_per_step": te_per_step,
        "k_bar": float(k_bar),
        "k_bar_shuffled": float(diag["k_bar_shuffled"]),
        "k_bar_reversed": float(diag["k_bar_reversed"]),
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
        # Sprint-5 calibration provenance + collapse diagnostic.
        "likelihood": likelihood,
        "sigma_obs": sigma_obs,
        "free_bits": free_bits,
        "mean_logvar_full": float(eval_m["mean_logvar_full"]),
        "latent_stats_fitted": bool(ckpt.get("latent_stats_fitted", False)),
        "ckpt_path": str(Path(ckpt_path).resolve()),
    }
    print(
        f"[eval] {run_tag}: K_bar={row['k_bar']:.5f}  "
        f"K_shuffled={row['k_bar_shuffled']:.5f}  "
        f"K_reversed={row['k_bar_reversed']:.5f}  "
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
    # Time-reversed-source mean. NaN-safe so legacy rows that pre-date the
    # column do not crash the aggregate.
    k_reversed = np.array(
        [r.get("k_bar_reversed", float("nan")) for r in rows], dtype=float,
    )

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

    # Metric 3 -- calibration slope (pooled over all M).
    alpha, gamma, r2 = _linear_calibration(te, kbar)

    # Metric 3b -- per-M calibration. Because the per-channel TE is held fixed
    # as M varies (channel-dilution isolation, V2-D6 / Section 3.1.1), the
    # calibration slope is expected to differ across M; the pooled fit hides
    # this. Group rows by M and fit / correlate each channel count separately.
    m_vals_agg = np.array(
        [int(r["M"]) if r.get("M") is not None else -1 for r in rows],
        dtype=int,
    )
    calib_by_m: Dict[str, Any] = {}
    for m in sorted(set(m_vals_agg.tolist())):
        if m < 0:
            continue
        sel = m_vals_agg == m
        te_m, kbar_m = te[sel], kbar[sel]
        a_m, g_m, r2_m = _linear_calibration(te_m, kbar_m)
        rho_m = _spearman_rho(te_m, kbar_m)
        calib_by_m[str(int(m))] = {
            "alpha": a_m, "gamma": g_m, "r2": r2_m,
            "spearman": rho_m, "n": int(np.count_nonzero(sel)),
        }

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
            "k_bar_reversed_mean": (
                float(np.nanmean(k_reversed))
                if k_reversed.size and not np.all(np.isnan(k_reversed))
                else float("nan")
            ),
        },
        "metric2_spearman": rho,
        "metric3_calibration": {"alpha": alpha, "gamma": gamma, "r2": r2},
        "metric3b_calibration_by_M": calib_by_m,
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

def _knob_token(knob_name: str, value: float) -> str:
    """Filesystem-safe knob token, e.g. ``"By0p5"`` for ``B_y=0.5``.

    Args:
        knob_name: Short knob label (``"By"``, ``"c"``, ``"p"``).
        value: Numeric value.

    Returns:
        A token with ``.`` -> ``"p"`` and ``-`` -> ``"m"``.
    """
    return f"{knob_name}{value:g}".replace(".", "p").replace("-", "m")


def _setting_tags_knob(
    benchmark: str, knob_name: str, value: float, m: int
) -> Tuple[str, str]:
    """Filesystem-safe ``(data_tag, run_tag)`` for one v2 sweep cell.

    Args:
        benchmark: Benchmark identifier (e.g. ``"G1"``).
        knob_name: Short knob label (``"By"`` for G1, ``"c"`` for G2,
            ``"p"`` for G3).
        value: Knob value.
        m: Number of informative channels.

    Returns:
        Tuple ``(data_tag, run_tag)``, e.g. ``benchmark='G1', knob_name='By',
        value=0.5, m=4`` -> ``("benchmark_G1_sweep_By0p5_m4",
        "sweep_By0p5_m4")``.
    """
    token = _knob_token(knob_name, value)
    data_tag = f"benchmark_{benchmark}_sweep_{token}_m{int(m)}"
    run_tag = f"sweep_{token}_m{int(m)}"
    return data_tag, run_tag


def enumerate_sweep(
    config: Dict[str, Any],
    dropped_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    r"""Enumerate the active v2 benchmark's sweep grid with analytic TE.

    Dispatches on ``config["sweep"]["kind"]``:

    * ``gaussian_state_space`` (G1) -- with a ``target_te_grid`` (variable
      delays), the uniform coupling $B_y$ is **solved per (target, M)** via
      :func:`B_y_for_mean_te_block_state_space` so the cell lands on the
      requested *mean* block TE (real-data [0.1, 3] band), holding TE fixed as
      $M$ varies. Otherwise crosses the legacy ``B_y_grid`` x ``m_grid`` using
      the mean-over-delays TE (variable) or single-delay TE (fixed).
    * ``arx`` (G2) -- same, with ``c`` solved via
      :func:`c_for_mean_te_block_arx` (closed form) or the legacy ``c_grid``.
    * ``regime_switch`` (G3) -- crosses ``p_grid`` x ``m_grid``;
      $\mathrm{TE}^{(H)} = M\cdot$ :func:`te_categorical_switch_block`.

    Per V2-D6, the per-channel TE scales linearly in $M$ because each
    informative channel pair is independent under the v2 DGPs; with
    ``target_te_grid`` the per-channel target is therefore the cell target
    divided by $M$.

    For G1 (``gaussian_state_space``) the coupling is solved by bisecting a
    **Monte-Carlo** TE estimator, which has an upward finite-sample bias floor
    $F$. At high $M$ + low ``target_te`` the per-channel target $t/M$ can fall
    at or below $F$, where the bisection bracket check fails. Such cells are
    **trimmed** (not solved): a cell is kept only when $t/M > F\cdot\text{margin}$
    (``sweep.te_floor_margin``, default $1.5$); $F$ is probed once per benchmark
    at the solver's lower bracket $B_y = 10^{-4}$. Any residual solver failure
    is also caught and trimmed rather than raised. G2 (``arx``) uses a
    closed-form TE with no MC floor, so no cell is trimmed there.

    Args:
        config: The parsed (benchmark-resolved) config -- reads ``sweep``,
            ``data`` and ``model.horizon``.
        dropped_out: Optional list; when provided, every trimmed cell is
            appended as a dict with ``kind``, ``M``, ``target_te``,
            ``per_channel_target``, ``floor`` and ``reason``. Default ``None``
            keeps the call source-compatible with existing callers.

    Returns:
        A list of setting dicts, each carrying ``kind``, ``M``, ``te_true``,
        ``te_per_step`` and the swept knob (``B_y`` / ``c`` / ``p_switch``).

    Raises:
        ValueError: If ``sweep.kind`` is not a known v2 dispatch.
    """
    sweep = config.get("sweep", {}) or {}
    kind = str(sweep.get("kind", "")).lower()
    horizon = int(config["model"]["horizon"])
    data = config.get("data", {}) or {}

    # Variable per-sample delays (delay_min/delay_max) vs fixed delays/delay.
    variable_delay = (
        data.get("delay_min") is not None or data.get("delay_max") is not None
    )
    K_history = data.get("K_history")
    K_history = None if K_history is None else int(K_history)

    settings: List[Dict[str, Any]] = []
    if kind == "gaussian_state_space":
        # Per-channel TE (single oscillator spec); the cell TE is that value x M
        # since the M informative channels are independent. With a
        # ``target_te_grid`` the coupling B_y is *solved* per (target, M) so the
        # cell lands on the requested mean block TE (real-data [0.1, 3] band),
        # holding TE fixed as M varies (isolating channel dilution).
        oscillators = [tuple(pair) for pair in data["oscillators"]]
        target_ar = float(data["target_ar"])
        sigma2_y = float(data["sigma2_y"])
        sigma2_eta = data["sigma2_eta"]
        te_n_samples = int(sweep.get("te_n_samples", 12_000))
        target_te_grid = sweep.get("target_te_grid")
        if target_te_grid is not None:
            if not variable_delay:
                raise ValueError(
                    "enumerate_sweep: target_te_grid requires variable delays "
                    "(delay_min/delay_max) in the data block."
                )
            dmin, dmax = int(data["delay_min"]), int(data["delay_max"])
            # Per-channel MC bias floor of the state-space TE estimator. The
            # coupling solver bisects this Monte-Carlo estimator, so a cell
            # whose per-channel target (= t_target / M) lands at or below the
            # floor makes the bracket check fail in
            # ``analytic_te._bisect_for_te_target``. The floor is
            # M-independent (solving always uses the single oscillator spec),
            # so probe it once at the solver's lower bracket B_y = 1e-4 and
            # trim every unreachable (M, target_te) cell.
            floor_margin = float(sweep.get("te_floor_margin", 1.5))
            te_floor = float(mean_te_block_state_space_over_delays(
                delay_min=dmin, delay_max=dmax,
                oscillators=oscillators, target_ar=target_ar,
                B_y=1e-4, sigma2_y=sigma2_y, sigma2_eta=sigma2_eta,
                H=horizon, K_history=K_history, n_samples=te_n_samples,
            ))
            min_per_channel = te_floor * floor_margin

            def _drop(m: int, t_target: float, per_channel: float,
                      reason: str) -> None:
                """Record (and log) a G1 cell trimmed below the MC floor."""
                if dropped_out is not None:
                    dropped_out.append({
                        "kind": "gaussian_state_space", "M": int(m),
                        "target_te": float(t_target),
                        "per_channel_target": per_channel,
                        "floor": te_floor, "reason": reason,
                    })
                print(
                    f"[sweep] drop G1 cell M={m} target_te={t_target:g} "
                    f"(per-channel {per_channel:.4g}): {reason}"
                )

            for m in sweep.get("m_grid", []):
                for t_target in target_te_grid:
                    per_channel = float(t_target) / int(m)
                    if per_channel <= min_per_channel:
                        _drop(m, t_target, per_channel,
                              f"below MC floor*margin {min_per_channel:.4g}")
                        continue
                    try:
                        sol = B_y_for_mean_te_block_state_space(
                            target_te_block=per_channel,
                            delay_min=dmin, delay_max=dmax,
                            oscillators=oscillators, target_ar=target_ar,
                            sigma2_y=sigma2_y, sigma2_eta=sigma2_eta,
                            H=horizon, K_history=K_history,
                            n_samples=te_n_samples,
                        )
                    except ValueError as exc:  # bracket miss near the floor
                        _drop(m, t_target, per_channel,
                              f"solver bracket failed: {exc}")
                        continue
                    te = float(sol["te_block"]) * int(m)
                    settings.append({
                        "kind": "gaussian_state_space",
                        "B_y": float(sol["B_y_scalar"]), "M": int(m),
                        "te_true": te, "te_per_step": te / horizon,
                        "target_te": float(t_target),
                    })
        else:
            for m in sweep.get("m_grid", []):
                for b in sweep.get("B_y_grid", []):
                    if variable_delay:
                        te_per_channel = mean_te_block_state_space_over_delays(
                            delay_min=int(data["delay_min"]),
                            delay_max=int(data["delay_max"]),
                            oscillators=oscillators, target_ar=target_ar,
                            B_y=float(b), sigma2_y=sigma2_y,
                            sigma2_eta=sigma2_eta, H=horizon,
                            K_history=K_history, n_samples=te_n_samples,
                        )
                    else:
                        delays = [int(d) for d in data["delays"]]
                        te_per_channel = te_block_state_space_gaussian(
                            oscillators=oscillators, target_ar=target_ar,
                            delays=delays, B_y=[float(b)] * len(delays),
                            sigma2_y=sigma2_y, sigma2_eta=sigma2_eta,
                            H=horizon, n_samples=te_n_samples,
                        )
                    te = float(te_per_channel) * int(m)
                    settings.append({
                        "kind": "gaussian_state_space",
                        "B_y": float(b), "M": int(m),
                        "te_true": te, "te_per_step": te / horizon,
                    })
    elif kind == "arx":
        rho_u = float(data["rho_u"])
        rho_y = float(data["rho_y"])
        sigma2_eta = float(data["sigma2_eta"])
        sigma2_eps = float(data["sigma2_eps"])
        target_te_grid = sweep.get("target_te_grid")
        if target_te_grid is not None:
            if not variable_delay:
                raise ValueError(
                    "enumerate_sweep: target_te_grid requires variable delays "
                    "(delay_min/delay_max) in the data block."
                )
            dmin, dmax = int(data["delay_min"]), int(data["delay_max"])
            for m in sweep.get("m_grid", []):
                for t_target in target_te_grid:
                    sol = c_for_mean_te_block_arx(
                        target_te_block=float(t_target) / int(m),
                        delay_min=dmin, delay_max=dmax,
                        rho_u=rho_u, rho_y=rho_y,
                        sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
                        H=horizon, M=1, K_history=K_history,
                    )
                    te = float(sol["te_block"]) * int(m)
                    settings.append({
                        "kind": "arx",
                        "c": float(sol["c_scalar"]), "M": int(m),
                        "te_true": te, "te_per_step": te / horizon,
                        "target_te": float(t_target),
                    })
        else:
            for m in sweep.get("m_grid", []):
                for c in sweep.get("c_grid", []):
                    if variable_delay:
                        te_per_channel = mean_te_block_arx_over_delays(
                            delay_min=int(data["delay_min"]),
                            delay_max=int(data["delay_max"]),
                            rho_u=rho_u, rho_y=rho_y, c=float(c),
                            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
                            H=horizon, M=1, K_history=K_history,
                        )
                    else:
                        te_per_channel = te_block_arx_gaussian(
                            rho_u=rho_u, rho_y=rho_y, c=float(c),
                            sigma2_eta=sigma2_eta, sigma2_eps=sigma2_eps,
                            H=horizon, D=int(data["delay"]),
                        )
                    te = float(te_per_channel) * int(m)
                    settings.append({
                        "kind": "arx",
                        "c": float(c), "M": int(m),
                        "te_true": te, "te_per_step": te / horizon,
                    })
    elif kind == "regime_switch":
        K = int(data["K_classes"])
        for m in sweep.get("m_grid", []):
            for p in sweep.get("p_grid", []):
                te_per_channel = te_categorical_switch_block(float(p), K, horizon)
                te = float(te_per_channel) * int(m)
                settings.append({
                    "kind": "regime_switch",
                    "p_switch": float(p), "M": int(m),
                    "te_true": te, "te_per_step": te / horizon,
                })
    else:
        raise ValueError(
            f"enumerate_sweep: unknown sweep kind {kind!r} (expected one of "
            f"'gaussian_state_space', 'arx', 'regime_switch')."
        )
    return settings


def _ensure_dataset_state_space(
    config: Dict[str, Any], data_tag: str, B_y: float, m: int
) -> None:
    """Build a G1 sweep dataset cell if not already cached (opt-in).

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag for this cell.
        B_y: Single-scalar coupling broadcast over ``data["delays"]``.
        m: Number of informative channels.
    """
    cfg = deepcopy(config)
    bd._apply_overrides(cfg, {"tag": data_tag, "m": m})
    n_osc = len(cfg["data"].get("delays", [])) or 1
    cfg["data"]["B_y"] = [float(B_y)] * n_osc
    bd.build_dataset(cfg, force=False)


def _ensure_dataset_arx(
    config: Dict[str, Any], data_tag: str, c: float, m: int
) -> None:
    """Build a G2 sweep dataset cell if not already cached (opt-in).

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag for this cell.
        c: ARX coupling.
        m: Number of informative channels.
    """
    cfg = deepcopy(config)
    bd._apply_overrides(cfg, {"tag": data_tag, "m": m})
    cfg["data"]["c"] = float(c)
    bd.build_dataset(cfg, force=False)


def _ensure_dataset_regime_switch(
    config: Dict[str, Any], data_tag: str, p_switch: float, m: int
) -> None:
    """Build a G3 sweep dataset cell if not already cached (opt-in).

    Args:
        config: The parsed config (copied before override).
        data_tag: Cache tag for this cell.
        p_switch: Regime-change probability.
        m: Number of informative channels.
    """
    cfg = deepcopy(config)
    bd._apply_overrides(cfg, {"tag": data_tag, "m": m})
    cfg["data"]["p_switch"] = float(p_switch)
    bd.build_dataset(cfg, force=False)


def _ensure_dataset_simple(config: Dict[str, Any], data_tag: str) -> None:
    """Build a knob-free dataset (e.g. the G1-rev directionality cell) if not cached.

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
    # Plot 2 sweeps whichever v2 knob the active benchmark actually varied
    # (B_y for G1, c for G2, p_switch for G3). Each row carries exactly one
    # of these as a non-None value.
    knob_candidates = (
        ("B_y", "coupling $B_y$"),
        ("c", "ARX coupling $c$"),
        ("p_switch", "switch probability $p$"),
    )
    knob_name = "B_y"
    knob_label = "coupling $B_y$"
    knob_vals = np.full(len(rows), np.nan, dtype=float)
    for field, label in knob_candidates:
        vals = np.array(
            [float(r[field]) if r.get(field) is not None else np.nan
             for r in rows], dtype=float,
        )
        if np.isfinite(vals).any():
            knob_name = field
            knob_label = label
            knob_vals = vals
            break
    m_vals = np.array(
        [int(r["M"]) if r["M"] is not None else -1 for r in rows], dtype=int
    )
    m_levels = sorted(set(m_vals.tolist()))

    # --- Plot 1: K_bar vs true TE + calibration line ----------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for idx, m in enumerate(m_levels):
        sel = m_vals == m
        # Connect the per-M circles with a line in the same palette colour
        # (ordered along the TE axis), so each channel count reads as a trend.
        order_te = np.argsort(te[sel])
        ax.plot(te[sel][order_te], kbar[sel][order_te], color=_m_color(idx),
                lw=1.0, zorder=2, label="_nolegend_")
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
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for idx, m in enumerate(m_levels):
        sel = m_vals == m
        if not np.isfinite(knob_vals[sel]).any():
            continue
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
    # Filesystem-safe knob slug for the filename (B_y -> B_y, p_switch -> p_switch).
    ps.save_figure(fig, out_dir / f"kbar_vs_{knob_name}")

    # --- Plot 3: predictive gain vs K_bar ---------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0.0, color=ps.COLOR_GRAY, ls="--", lw=0.9, zorder=1)
    for idx, m in enumerate(m_levels):
        sel = m_vals == m
        # Connect the per-M circles with a same-colour line ordered along the
        # K_bar axis -- traces the (K_bar, pred_gap) trajectory as the knob grows.
        order_k = np.argsort(kbar[sel])
        ax.plot(kbar[sel][order_k], pred_gap[sel][order_k], color=_m_color(idx),
                lw=1.0, zorder=2, label="_nolegend_")
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


def _make_calibration_by_m(
    rows: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: Path
) -> None:
    r"""Render the per-$M$ calibration grid (``kbar_vs_te__byM``).

    One subplot per number of informative channels $M$, each plotting that
    channel count's $\bar K$ against the analytic block TE with its **own**
    least-squares fit $\bar K = \alpha_M + \gamma_M\,\mathrm{TE}$ and Spearman
    $\rho_M$. Because the per-channel TE is held fixed as $M$ varies (channel
    dilution, Section 3.1.1), the calibration slope is expected to differ
    across $M$; the pooled :func:`_make_plots` ``kbar_vs_te`` figure hides this,
    so this grid breaks it out per channel count. The per-$M$ fit / correlation
    values come straight from ``metrics['metric3b_calibration_by_M']`` (computed
    once in :func:`_aggregate_metrics`), so the figure never re-fits.

    Args:
        rows: Per-checkpoint metrics rows (>= 2 expected).
        metrics: The :func:`_aggregate_metrics` output (reads
            ``metric3b_calibration_by_M``).
        out_dir: Destination directory.
    """
    import math

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
    m_vals = np.array(
        [int(r["M"]) if r["M"] is not None else -1 for r in rows], dtype=int
    )
    m_levels = [m for m in sorted(set(m_vals.tolist())) if m >= 0]
    if not m_levels:
        return

    calib_by_m = metrics.get("metric3b_calibration_by_M", {}) or {}

    n = len(m_levels)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()

    for idx, m in enumerate(m_levels):
        ax = flat_axes[idx]
        sel = m_vals == m
        te_m, kbar_m = te[sel], kbar[sel]
        ax.scatter(te_m, kbar_m, s=44, color=_m_color(idx),
                   edgecolors=ps.COLOR_BLACK, linewidths=0.4, zorder=3)
        cal = calib_by_m.get(str(int(m)), {})
        alpha, gamma = cal.get("alpha"), cal.get("gamma")
        rho = cal.get("spearman", float("nan"))
        r2 = cal.get("r2", float("nan"))
        if (alpha is not None and gamma is not None
                and np.isfinite(alpha) and np.isfinite(gamma)
                and np.isfinite(te_m).any()):
            xs = np.linspace(float(np.nanmin(te_m)), float(np.nanmax(te_m)), 100)
            ax.plot(
                xs, gamma * xs + alpha, ls="--", lw=1.1, color=ps.COLOR_BLACK,
                zorder=2,
                label=fr"fit: ${gamma:.3g}\,\mathrm{{TE}}+{alpha:.3g}$",
            )
            ax.legend(loc="best")
        title = fr"M={m}"
        if isinstance(rho, float) and np.isfinite(rho):
            title += fr"   ($\rho$={rho:.3f}"
            if isinstance(r2, float) and np.isfinite(r2):
                title += fr", $R^2$={r2:.3f}"
            title += ")"
        ax.set_title(title)
        ax.set_xlabel("analytic block TE (nats)")
        ax.set_ylabel(r"$\bar K$ (nats)")
        ps.style_axes(ax)

    # Hide any unused trailing axes so a non-square grid stays clean.
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(
        r"$\bar K$ vs analytic block TE -- per informative-channel count $M$",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ps.save_figure(fig, out_dir / "kbar_vs_te__byM")


def _make_checkpoint_figure(row: Dict[str, Any], out_dir: Path) -> list:
    r"""Render a 2x2 single-checkpoint TE-diagnostics figure.

    Works for **one** evaluated checkpoint (the ``single`` mode and, per cell,
    the ``sweep`` mode). The four panels expose whether the latent bottleneck
    is actually being used:

        * per-latent-dim KL -- a flat-zero bar row is the collapse signature.
        * $\bar K$ vs the shuffled-source null control -- a faithful surrogate
          collapses $\bar K$ when the genuine $U\to Y$ pairing is destroyed.
        * baseline vs full forecast loss -- ``pred_gap`` is their difference.
        * a text panel with every scalar diagnostic and the analytic ``te_true``.

    Args:
        row: One :func:`evaluate_checkpoint` metrics row.
        out_dir: Destination directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_dim = np.asarray(row.get("per_dim_kl") or [], dtype=float)
    te_true = float(row.get("te_true", float("nan")))
    k_bar = float(row.get("k_bar", float("nan")))
    k_shuf = float(row.get("k_bar_shuffled", float("nan")))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.suptitle(
        f"Checkpoint TE diagnostics -- {row.get('run_tag','?')}  "
        f"(benchmark {row.get('benchmark','?')}, epoch {row.get('epoch')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel (0,0): per-latent-dim KL -----------------------------------
    ax = axes[0, 0]
    if per_dim.size:
        ax.bar(np.arange(per_dim.size), per_dim, color=ps.COLOR_BLUE, width=0.9)
        ax.axhline(float(np.nanmean(per_dim)), color=ps.COLOR_VERMILLION,
                   ls="--", lw=1.0, label=f"mean={np.nanmean(per_dim):.2e}")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no per-dim KL recorded", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(r"per-latent-dim KL (flat $\approx 0$ row = posterior collapse)")
    ax.set_xlabel("latent dimension $d$")
    ax.set_ylabel(r"$K_d$ (nats)")
    ps.style_axes(ax)

    # --- Panel (0,1): K_bar vs the shuffled-source null control -----------
    ax = axes[0, 1]
    ax.bar([0, 1], [k_bar, k_shuf], width=0.6,
           color=[ps.COLOR_BLUE, ps.COLOR_GRAY])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$\bar K$ (paired)", r"$\bar K$ (shuffled src)"])
    for x, v in ((0, k_bar), (1, k_shuf)):
        if np.isfinite(v):
            ax.text(x, v, f"  {v:.3e}", ha="center", va="bottom",
                    fontsize=ps.FONT_TICK)
    ax.set_title(f"TE surrogate vs null control   (analytic te_true={te_true:.3f} nats)")
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.margins(y=0.18)
    ps.style_axes(ax)

    # --- Panel (1,0): baseline vs full forecast loss ----------------------
    ax = axes[1, 0]
    base_l = float(row.get("base_loss", float("nan")))
    feat_l = float(row.get("feat_loss", float("nan")))
    ax.bar([0, 1], [base_l, feat_l], width=0.6,
           color=[ps.COLOR_ORANGE, ps.COLOR_BLUE])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["baseline $L_{base}$", "full $L_{feat}$"])
    for x, v in ((0, base_l), (1, feat_l)):
        if np.isfinite(v):
            ax.text(x, v, f"  {v:.4f}", ha="center", va="bottom",
                    fontsize=ps.FONT_TICK)
    ax.set_title(
        f"forecast loss   (pred_gap = $L_{{base}}-L_{{feat}}$ = "
        f"{float(row.get('pred_gap', float('nan'))):+.4f})"
    )
    ax.set_ylabel("weighted MSE")
    ax.margins(y=0.18)
    ps.style_axes(ax)

    # --- Panel (1,1): scalar-diagnostics text -----------------------------
    ax = axes[1, 1]
    ax.axis("off")
    ln_l = float(np.log(91.0))  # ln(max_lag + 1): diffuse-attention reference.
    lines = [
        "TE surrogate",
        f"  te_true (block)  : {te_true:.4f} nats",
        f"  te_per_step      : {float(row.get('te_per_step', float('nan'))):.4f} nats",
        f"  K_bar            : {k_bar:.5f} nats",
        f"  K_bar shuffled   : {k_shuf:.5f} nats",
        f"  kld_loss         : {float(row.get('kld_loss', float('nan'))):.5f}",
        "",
        "Latent / attention",
        f"  mu_post_prior_gap: {float(row.get('mu_post_prior_gap', float('nan'))):.5f}",
        f"  attn_entropy     : {float(row.get('attn_entropy', float('nan'))):.4f}"
        f"  (diffuse ref ln 91 = {ln_l:.3f})",
        "",
        "Run",
        f"  epoch            : {row.get('epoch')}",
        f"  n_test           : {row.get('n_test')}",
        f"  warmup           : {row.get('warmup')}",
        f"  latent_stats_fit : {row.get('latent_stats_fitted')}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=9.0, color=ps.COLOR_BLACK,
            transform=ax.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe_tag = str(row.get("run_tag", "ckpt")).replace("/", "_")
    return ps.save_figure(fig, out_dir / f"diagnostics_{safe_tag}")


def _make_sweep_extras(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    r"""Render the extra sweep-level analysis figures (task 4.7 extension).

    Adds two figures to the headline three:

        * ``per_dim_kl_heatmap`` -- a (setting x latent-dim) heatmap of the
          per-dimension KL, settings ordered by analytic TE. Reveals whether
          the bottleneck recruits more latent dimensions as TE grows.
        * ``null_control`` -- $\bar K$ against its shuffled-source control per
          setting, with the analytic TE overlaid; the gap between the paired
          and shuffled bars is the genuine transfer signal.

    Args:
        rows: Per-checkpoint metrics rows (>= 2 expected).
        out_dir: Destination directory.
    """
    import matplotlib.pyplot as plt

    from model.vae_teb_prediction.model.model_experiment.synthetic import (
        plot_style as ps,
    )

    ps.apply_style()
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["te_true"]))
    rows = [rows[i] for i in order]
    labels = [
        f"{r.get('run_tag','?')}\nTE={float(r['te_true']):.2f}" for r in rows
    ]
    te = np.array([float(r["te_true"]) for r in rows])
    kbar = np.array([float(r["k_bar"]) for r in rows])
    kshuf = np.array([float(r["k_bar_shuffled"]) for r in rows])

    # --- Figure 1: per-dim KL heatmap (setting x latent dim) --------------
    per_dim = [np.asarray(r.get("per_dim_kl") or [], dtype=float) for r in rows]
    width = max(len(v) for v in per_dim) if per_dim else 0
    if width > 0:
        grid = np.full((len(rows), width), np.nan)
        for i, v in enumerate(per_dim):
            grid[i, : v.size] = v
        fig, ax = plt.subplots(figsize=(8.8, 0.5 * len(rows) + 2.2))
        vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma",
                       vmin=0.0, vmax=max(vmax, 1e-9), interpolation="nearest")
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labels, fontsize=ps.FONT_TICK)
        ax.set_xlabel("latent dimension $d$")
        ax.set_title(r"per-dimension KL $K_d$ across sweep settings "
                     r"(rows ordered by analytic TE)")
        ps.add_colorbar(fig, im, ax, label=r"$K_d$ (nats)")
        ps.style_axes(ax, grid="none")
        fig.tight_layout()
        ps.save_figure(fig, out_dir / "per_dim_kl_heatmap")

    # --- Figure 2: K_bar vs shuffled-source null control ------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, kbar, width=0.4, color=ps.COLOR_BLUE, label=r"$\bar K$ (paired)")
    ax.bar(idx + 0.2, kshuf, width=0.4, color=ps.COLOR_GRAY,
           label=r"$\bar K$ (shuffled source)")
    ax2 = ax.twinx()
    ax2.plot(idx, te, marker="o", color=ps.COLOR_VERMILLION, lw=1.2,
             label="analytic TE")
    ax2.set_ylabel("analytic block TE (nats)", color=ps.COLOR_VERMILLION)
    ax2.tick_params(axis="y", colors=ps.COLOR_VERMILLION)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=ps.FONT_TICK)
    ax.set_ylabel(r"$\bar K$ (nats)")
    ax.set_title(r"TE surrogate vs shuffled-source null control")
    ax.legend(loc="upper left")
    ps.style_axes(ax)
    fig.tight_layout()
    ps.save_figure(fig, out_dir / "null_control")


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
        f"K_shuffled(mean)={m1['k_bar_shuffled_mean']:.5f}  "
        f"K_reversed(mean)={m1.get('k_bar_reversed_mean', float('nan')):.5f}\n"
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
        A results dict: ``rows``, ``metrics``, ``out_dir``, ``skipped`` and
        ``dropped`` (G1 cells trimmed below the MC floor; also stored under
        ``metrics["dropped_cells"]`` and written to ``metrics.json``).
    """
    benchmark = str(config["experiment"]["benchmark"])
    device = device or tm.resolve_device(config["runtime"])
    out_dir = _eval_out_dir(config, benchmark)
    data_root = _data_root(config)
    results_root = _results_root(config)

    dropped: List[Dict[str, Any]] = []
    settings = enumerate_sweep(config, dropped_out=dropped)
    print(
        f"[sweep] benchmark {benchmark}: {len(settings)} sweep setting(s)  "
        f"build_missing={build_missing}  train_missing={train_missing}"
    )
    if dropped:
        n_total = len(settings) + len(dropped)
        cells = ", ".join(
            f"(M={d['M']}, target_te={d['target_te']:g})" for d in dropped
        )
        print(
            f"[sweep] trimmed {len(dropped)}/{n_total} unreachable cell(s) "
            f"below the MC floor: {cells}"
        )

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for setting in settings:
        kind = str(setting.get("kind", ""))
        m = setting["M"]
        # Each v2 sweep cell carries a single named knob; the dispatch picks
        # the matching tag + ensure-dataset helper.
        if kind == "gaussian_state_space":
            value = float(setting["B_y"])
            data_tag, run_tag = _setting_tags_knob(benchmark, "By", value, m)
            label = f"B_y={value:g}, M={m}"
            ensure_dataset = lambda: _ensure_dataset_state_space(
                config, data_tag, value, m
            )
        elif kind == "arx":
            value = float(setting["c"])
            data_tag, run_tag = _setting_tags_knob(benchmark, "c", value, m)
            label = f"c={value:g}, M={m}"
            ensure_dataset = lambda: _ensure_dataset_arx(
                config, data_tag, value, m
            )
        elif kind == "regime_switch":
            value = float(setting["p_switch"])
            data_tag, run_tag = _setting_tags_knob(benchmark, "p", value, m)
            label = f"p={value:g}, M={m}"
            ensure_dataset = lambda: _ensure_dataset_regime_switch(
                config, data_tag, value, m
            )
        else:
            print(f"  [skip ] unknown sweep kind {kind!r}")
            skipped.append(f"unknown_{kind}")
            continue
        cache_dir = data_root / benchmark / data_tag
        ckpt_path = results_root / benchmark / run_tag / "final.ckpt"

        if not (cache_dir / "test.npz").is_file():
            if build_missing:
                print(f"  [build] {data_tag}  ({label})")
                ensure_dataset()
            else:
                print(
                    f"  [skip ] {run_tag}: dataset '{data_tag}' not cached "
                    f"(pass --build-missing to generate it)"
                )
                skipped.append(run_tag)
                continue

        if not ckpt_path.is_file():
            if train_missing:
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
    metrics["dropped_cells"] = dropped
    write_summary_csv(rows, out_dir / "summary.csv")
    write_metrics_json(metrics, rows, out_dir / "metrics.json")
    if len(rows) >= 2:
        _make_plots(rows, metrics, out_dir)
        _make_calibration_by_m(rows, metrics, out_dir)
        _make_sweep_extras(rows, out_dir)
    else:
        print(
            "[plots] fewer than 2 settings evaluated -- skipping sweep plots "
            "(the A-sweep training run is deferred; see the plan)."
        )
    if rows:
        pc_dir = out_dir / "per_checkpoint"
        for row in rows:
            _make_checkpoint_figure(row, pc_dir)
        print(f"[plots] {len(rows)} per-checkpoint diagnostics -> {pc_dir}")

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
        "dropped": dropped,
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
    # data_dir / results_dir overrides -> config["paths"] (None -> YAML default).
    tm.apply_path_overrides(config, overrides)
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
        fig_paths = _make_checkpoint_figure(row, out_dir)
        _print_metrics(metrics, n=1)
        print(
            f"[done] single-checkpoint eval -> {out_dir}\n"
            f"       diagnostics figure -> {len(fig_paths)} file(s)"
        )
        return {"rows": [row], "metrics": metrics, "out_dir": str(out_dir)}

    if mode == "sweep":
        return run_sweep(
            config,
            build_missing=bool(overrides.get("build_missing")),
            train_missing=bool(overrides.get("train_missing")),
            device=device,
        )

    raise ValueError(
        f"unknown mode: {mode!r} (expected 'single' or 'sweep')."
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
        choices=["single", "sweep"],
        help="single-checkpoint evaluation or full v2 sweep aggregation",
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
    p.add_argument(
        "--data-dir", type=str, default=None, dest="data_dir",
        help="override paths.data_dir (absolute/relative path, ~, or $VAR); "
             "None -> config paths.data_dir",
    )
    p.add_argument(
        "--results-dir", type=str, default=None, dest="results_dir",
        help="override paths.results_dir (same format as --data-dir); "
             "None -> config paths.results_dir",
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
        "mode": "single",                   # "single" | "sweep"
        # single mode: which checkpoint to evaluate.
        "checkpoint": str(
            _EXPERIMENT_DIR / "results" / "G1" / "G1_baseline" / "final.ckpt"
        ),
        "data_tag": None,                   # None -> the checkpoint's own tag
        "benchmark": None,                  # None -> config experiment.benchmark
        "build_missing": False,             # sweep mode: build missing datasets
        "train_missing": False,             # sweep mode: train missing ckpts
        "batch_size": None,                 # None -> checkpoint's batch size
        "device": None,                     # None -> config runtime.device
        "seed": None,                       # None -> config experiment.seed
        "data_dir": None,                   # None -> config paths.data_dir
        "results_dir": None,                # None -> config paths.results_dir
    }

    if len(sys.argv) > 1:
        main()                              # CLI mode -- argparse
    else:
        config = tm.load_config(CONFIG_PATH)
        # --- optional: tweak any config value not covered by RUN_CONFIG ------
        # e.g.  config["optim"]["batch_size"] = 16
        # ---------------------------------------------------------------------
        _dispatch(config, RUN_CONFIG)
