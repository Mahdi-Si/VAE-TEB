r"""Stage 0: a model-free ceiling on UP$\to$FHR predictive coupling.

The lag-attention VAE reports the source's contribution through the trained model only: every
number in ``eval/analyses`` flows through the encoder/posterior/decoder. So when the model shows a
near-zero ``pred_gap`` (the source barely improves the forecast) there is no way, from inside the
suite, to tell *"there is no extractable UP$\to$FHR coupling in the data"* apart from *"there is
coupling but the model fails to route it"*. This probe supplies the missing measurement.

It asks the model's own question with **no** neural network in the loop. For each anchor $t$ it
predicts the future target features $Y_{t+h}$ from

* a **baseline** design of the target's own lagged history $\{Y_{t-\ell}\}$, and
* a **full** design that adds the source's lagged history $\{U_{t-m}\}$,

with an ordinary ridge regression (a *linear, conservative* lower bound on extractable coupling)
fit on one set of recordings and scored on a disjoint set. The **uplift**
$\mathrm{MSE}_{\mathrm{base}} - \mathrm{MSE}_{\mathrm{full}}$ on held-out recordings is the
model-free analogue of the VAE's ``pred_gap``. A **shuffle control** (feed each held-out anchor a
*stranger's* source through the already-fit weights) is the analogue of ``shuffle_penalty``: it
must collapse the uplift to $\approx 0$ if the uplift is genuine pairing rather than overfitting.

Reading the result:

* **uplift $\gg 0$ and vanishes under shuffle** $\Rightarrow$ coupling is present and extractable
  even by a linear model, so the VAE's near-zero ``pred_gap`` is a *routing failure* -- proceed to
  the architecture change (close the posterior's direct-``h_y`` laundering door).
* **uplift $\approx 0$ for ridge *and* the optional nonlinear check** $\Rightarrow$ the source's
  incremental predictive value over a strong FHR self-baseline is at or below the noise floor at
  this feature set and horizon -- a *data ceiling*; an architecture change will not manufacture a
  signal that is not there, and the lever is the data/feature/horizon side (Stage 3).

The features are read through the *exact* training loader (same shards, ``stat_path``
normalisation, ``trim_minutes`` and ``use_up_st``), so the ceiling is measured on the same
representation the model consumes -- otherwise the two numbers would not be comparable.

Run (from the repo root):

.. code-block:: bash

    # The real measurement, against the config's held-out (test) shards:
    python -m teb_vae.lag_attn.eval.stage0_source_uplift_probe \
        --config teb_vae/lag_attn/configs/default.yaml

    # Correctness self-test -- fabricates data with a known injected coupling and a null, and
    # asserts the probe recovers the first and not the second. Needs no dataset:
    python -m teb_vae.lag_attn.eval.stage0_source_uplift_probe --self-test

From a PyCharm/IDE Run button (no command line): hit Run on this file. With no arguments it uses the
``RUN_*`` constants near the bottom -- edit ``RUN_CONFIG``, ``RUN_SPLIT``, or set ``RUN_SELF_TEST =
True`` there rather than adding a Run Configuration. Any ``--flag`` passed on the command line always
wins over those constants.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo root on sys.path when launched as a script from an IDE (mirrors trainer.py's guard).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# --- Defaults ---------------------------------------------------------------------------------
# Lag sets are in decimated steps (4 s each). The UP lags concentrate on the physiological
# contraction->deceleration band (20-120 s = lags 5-30), shifted by the pipeline's -20 s UP
# pre-shift, plus a couple of shorter/longer probes; the history lags span the FHR autocorrelation
# the baseline must exhaust before the source can add anything. Horizon offsets sample near / mid /
# far of the 30-step (2 min) forecast so uplift can be read per-offset, not just pooled.
_DEFAULT_HIST_LAGS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
_DEFAULT_UP_LAGS: Tuple[int, ...] = (0, 3, 5, 8, 10, 15, 20, 25, 30, 40)
_DEFAULT_HORIZON_OFFSETS: Tuple[int, ...] = (1, 5, 15, 29)
_DEFAULT_WARMUP = 30
_RIDGE_LAMBDAS: Tuple[float, ...] = (1e0, 1e1, 1e2, 1e3, 1e4)
# Aggregate held-out relative uplift below this reads as "no extractable coupling". A priori and
# deliberately generous: a genuine but small coupling should clear it, and it is worth recalibrating
# once a first real run exists (mirrors scalars.py's collapse thresholds).
_UPLIFT_REL_FLOOR = 0.01


@dataclass
class ProbeConfig:
    """Knobs for one probe run.

    Attributes:
        hist_lags: Target-history lags (decimated steps) forming the baseline design.
        up_lags: Source lags (decimated steps) the full design adds on top of the baseline.
        horizon_offsets: Future offsets $h$ (steps) whose $Y_{t+h}$ are predicted.
        warmup: Leading steps excluded from anchors (the encoders' warm-up, kept for parity).
        max_rows: Cap on total anchor rows fed to the solver; excess is randomly subsampled.
        seed: RNG seed for the guid split, the row subsample, and the shuffle control.
        eval_frac: Fraction of recordings held out for scoring.
        val_frac: Fraction of the fit recordings held out to select the ridge penalty.
    """

    hist_lags: Tuple[int, ...] = _DEFAULT_HIST_LAGS
    up_lags: Tuple[int, ...] = _DEFAULT_UP_LAGS
    horizon_offsets: Tuple[int, ...] = _DEFAULT_HORIZON_OFFSETS
    warmup: int = _DEFAULT_WARMUP
    max_rows: int = 60000
    seed: int = 0
    eval_frac: float = 0.3
    val_frac: float = 0.2


@dataclass
class Sample:
    """One recording's model-input features.

    Attributes:
        y: Target features $(T, c_y)$ = ``concat(fhr_st, fhr_ph)``.
        u: Source features $(T, c_u)$ = ``concat(up_st, up_ph)`` or ``up_ph`` alone.
        weight: Per-step validity $(T,)$; ``> 0`` marks a usable step.
        guid: Recording identifier, used to split fit/eval by recording (never by anchor).
    """

    y: np.ndarray
    u: np.ndarray
    weight: np.ndarray
    guid: str


# --- Data loading (real shards) ---------------------------------------------------------------
def load_samples_from_config(
    config_path: str, *, split: str, max_samples: Optional[int]
) -> List[Sample]:
    r"""Load samples through the training loader so features match what the model sees.

    Builds a :class:`CombinedHDF5Dataset` with the config's own ``stat_path`` normalisation,
    ``trim_minutes``, ``load_fields`` and filters, then reads the target/source blocks exactly as
    the task assembles them (``concat(fhr_st, fhr_ph)`` and, under ``use_up_st``,
    ``concat(up_st, up_ph)``).

    Args:
        config_path: Path to a leaf YAML config; its ``base:`` chain is resolved.
        split: ``'train'``, ``'test'`` or ``'both'`` -- which shard list to read.
        max_samples: Cap on recordings loaded (``None`` for all).

    Returns:
        The loaded samples.

    Raises:
        ValueError: If the requested split names no shards, or a required field is absent.
    """
    from teb_vae.lag_attn.config import load_config
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    config = load_config(config_path)
    dataset_config = config.get("dataset_config", {}) or {}
    dataloader_config = dataset_config.get("dataloader_config", {}) or {}
    vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    use_up_st = bool(vae_config.get("use_up_st", True))

    train_shards = list(dataset_config.get("vae_train_datasets") or [])
    test_shards = list(dataset_config.get("vae_test_datasets") or [])
    shards = {"train": train_shards, "test": test_shards, "both": train_shards + test_shards}.get(split)
    if not shards:
        raise ValueError(f"split={split!r} names no shards in {config_path}")

    # Only forward kwargs the dataset constructor accepts; cache is forced off (we read each index
    # once, so a cache is pure overhead here).
    accepted = {
        "load_fields", "allowed_guids", "cs_label", "bg_label", "epoch_min", "epoch_max",
        "label", "pin_memory", "trim_minutes",
    }
    raw_kwargs = dict(dataloader_config.get("dataset_kwargs") or {})
    dataset_kwargs = {name: value for name, value in raw_kwargs.items() if name in accepted}

    dataset = CombinedHDF5Dataset(
        paths=shards,
        stats_path=dataset_config.get("stat_path"),
        normalize_fields=dataloader_config.get("normalize_fields"),
        cache_size=0,
        **dataset_kwargs,
    )

    count = len(dataset)
    if max_samples is not None:
        count = min(count, int(max_samples))

    samples: List[Sample] = []
    for index in range(count):
        item = dataset[index]
        y = _cat_fields(item, ("fhr_st", "fhr_ph"))
        u = _cat_fields(item, ("up_st", "up_ph")) if use_up_st else _cat_fields(item, ("up_ph",))
        weight = getattr(item, "weight", None)
        if weight is None:
            weight_np = np.ones(y.shape[0], dtype=np.float32)
        else:
            raw = weight.numpy() if hasattr(weight, "numpy") else weight
            weight_np = np.asarray(raw, dtype=np.float32).reshape(-1)
        guid = str(getattr(item, "guid", f"_row{index}"))
        samples.append(Sample(y=y, u=u, weight=weight_np, guid=guid))
    return samples


def _cat_fields(item: object, names: Sequence[str]) -> np.ndarray:
    """Concatenate the named ``(T, C)`` tensor fields of a loader sample along the channel axis.

    Args:
        item: A loader sample (attribute-accessible tensor fields).
        names: Field names to concatenate, in order.

    Returns:
        The concatenated array $(T, \\sum C)$ as float32.

    Raises:
        ValueError: If any named field is missing from the sample.
    """
    blocks = []
    for name in names:
        value = getattr(item, name, None)
        if value is None:
            raise ValueError(
                f"sample has no field {name!r}; add it to dataloader_config.dataset_kwargs."
                f"load_fields and confirm the shard carries it"
            )
        blocks.append(np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=np.float32))
    return np.concatenate(blocks, axis=-1)


# --- Design-matrix construction ---------------------------------------------------------------
def build_design(
    samples: Sequence[Sample], config: ProbeConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Assemble anchor-wise regression matrices from a list of recordings.

    For every valid anchor $t$ in every recording it stacks the lagged target history, the lagged
    source history, and the future target block. An anchor is valid when it clears the warm-up and
    the longest lag, its whole forecast window fits inside the sequence, and the validity weight is
    positive at the anchor, every used lag, and every predicted future step -- so no padded or
    masked step ever enters the fit.

    Args:
        samples: The recordings to draw anchors from.
        config: The lag sets, horizon offsets and warm-up.

    Returns:
        ``(x_hist, x_up, y_future, group)``: the baseline design $(N, |hist|\,c_y)$, the source
        design $(N, |up|\,c_u)$, the target block $(N, |off|\,c_y)$, and a per-row integer group id
        $(N,)$ that ties each anchor to its recording (so the shuffle control can permute across
        recordings only). ``x_hist`` and ``y_future`` are non-finite-scrubbed to zero, which is the
        normalised mean.

    Raises:
        ValueError: If no valid anchor exists across all samples (e.g. lags longer than the
            sequence, or an all-zero weight field).
    """
    hist_lags = np.asarray(config.hist_lags, dtype=np.int64)
    up_lags = np.asarray(config.up_lags, dtype=np.int64)
    offsets = np.asarray(config.horizon_offsets, dtype=np.int64)
    max_back = int(max(hist_lags.max(), up_lags.max(), config.warmup))
    max_fwd = int(offsets.max())

    hist_rows: List[np.ndarray] = []
    up_rows: List[np.ndarray] = []
    fut_rows: List[np.ndarray] = []
    group_rows: List[np.ndarray] = []

    for group_id, sample in enumerate(samples):
        seq_len = sample.y.shape[0]
        y = np.nan_to_num(sample.y, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.nan_to_num(sample.u, nan=0.0, posinf=0.0, neginf=0.0)
        valid = sample.weight > 0.0
        if valid.shape[0] != seq_len:  # a defensive guard; the loader keeps these aligned
            valid = np.ones(seq_len, dtype=bool)

        first = max(max_back, config.warmup)
        last = seq_len - max_fwd  # exclusive
        anchors = np.arange(first, last, dtype=np.int64)
        if anchors.size == 0:
            continue

        # Validity: anchor, every used history/source lag, and every predicted future step.
        keep = valid[anchors].copy()
        for lag in np.unique(np.concatenate([hist_lags, up_lags])):
            keep &= valid[anchors - int(lag)]
        for off in offsets:
            keep &= valid[anchors + int(off)]
        anchors = anchors[keep]
        if anchors.size == 0:
            continue

        # Gather lag/offset windows with fancy indexing: (n_anchors, n_lags, C) -> flatten C-last.
        n = anchors.size
        hist = y[anchors[:, None] - hist_lags[None, :]].reshape(n, -1)
        src = u[anchors[:, None] - up_lags[None, :]].reshape(n, -1)
        fut = y[anchors[:, None] + offsets[None, :]].reshape(n, -1)

        hist_rows.append(hist)
        up_rows.append(src)
        fut_rows.append(fut)
        group_rows.append(np.full(n, group_id, dtype=np.int64))

    if not hist_rows:
        raise ValueError(
            "no valid anchors across all samples; check that lags fit inside the sequence and the "
            "weight field is not all-zero"
        )

    return (
        np.concatenate(hist_rows, axis=0),
        np.concatenate(up_rows, axis=0),
        np.concatenate(fut_rows, axis=0),
        np.concatenate(group_rows, axis=0),
    )


# --- Ridge regression (numpy closed form) -----------------------------------------------------
def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-column mean and a safe standard deviation (zero-variance columns -> 1)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def _ridge_fit(
    x: np.ndarray, y_centered: np.ndarray, lam: float
) -> np.ndarray:
    r"""Solve the multi-output ridge normal equations $(X^\top X + \lambda I)W = X^\top Y$.

    Args:
        x: Standardised design $(N, P)$.
        y_centered: Mean-centred targets $(N, K)$.
        lam: Ridge penalty $\lambda$.

    Returns:
        The weight matrix $W$ of shape $(P, K)$.
    """
    p = x.shape[1]
    gram = x.T @ x
    gram.flat[:: p + 1] += lam  # add lam to the diagonal in place
    return np.linalg.solve(gram, x.T @ y_centered)


def _fit_predict_ridge(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: np.ndarray,
    lambdas: Sequence[float],
) -> Tuple[np.ndarray, float]:
    r"""Fit ridge with a $\lambda$ chosen on a validation split; return eval-set predictions.

    Standardisation and the target mean come from the fit split only, so nothing about the
    validation or eval rows leaks into training.

    Args:
        x_fit: Fit design $(N_f, P)$.
        y_fit: Fit targets $(N_f, K)$.
        x_val: Validation design $(N_v, P)$.
        y_val: Validation targets $(N_v, K)$.
        x_eval: Eval design $(N_e, P)$.
        lambdas: Candidate ridge penalties; the one with the lowest validation MSE is used.

    Returns:
        ``(y_eval_pred, lam)``: eval-set predictions $(N_e, K)$ and the selected $\lambda$.
    """
    x_mean, x_std = _standardize_fit(x_fit)
    y_mean = y_fit.mean(axis=0)

    xs_fit = (x_fit - x_mean) / x_std
    xs_val = (x_val - x_mean) / x_std
    xs_eval = (x_eval - x_mean) / x_std
    yc_fit = y_fit - y_mean

    best_lam, best_mse, best_w = float(lambdas[0]), np.inf, None
    for lam in lambdas:
        weight = _ridge_fit(xs_fit, yc_fit, float(lam))
        val_pred = xs_val @ weight + y_mean
        mse = float(np.mean((val_pred - y_val) ** 2))
        if mse < best_mse:
            best_lam, best_mse, best_w = float(lam), mse, weight

    assert best_w is not None
    return xs_eval @ best_w + y_mean, best_lam


# --- Metrics ----------------------------------------------------------------------------------
def _pooled_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error pooled over all rows and target columns."""
    return float(np.mean((pred - target) ** 2))


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination pooled over all rows and target columns."""
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def _per_offset_mse(pred: np.ndarray, target: np.ndarray, n_offsets: int) -> np.ndarray:
    """MSE for each horizon offset, averaging over its channels and all rows."""
    n, width = target.shape
    per_channel = width // n_offsets
    p = pred.reshape(n, n_offsets, per_channel)
    t = target.reshape(n, n_offsets, per_channel)
    return np.mean((p - t) ** 2, axis=(0, 2))


# --- Orchestration ----------------------------------------------------------------------------
def run_probe(samples: Sequence[Sample], config: ProbeConfig) -> Dict[str, object]:
    r"""Fit the baseline and full regressors and return the uplift, per-offset and shuffle results.

    Args:
        samples: The recordings to draw anchors from.
        config: The probe configuration.

    Returns:
        A JSON-serialisable results dict: pooled baseline/full MSE and $R^2$, absolute and relative
        uplift, the per-offset uplift profile, the shuffle-control uplift, the selected ridge
        penalties, the row/recording counts, and a one-word ``verdict``.

    Raises:
        ValueError: If the fit or eval split ends up empty (too few recordings for ``eval_frac``).
    """
    rng = np.random.default_rng(config.seed)
    x_hist, x_up, y_future, group = build_design(samples, config)

    # Optional row cap: subsample anchors uniformly to bound the solve, keeping group ids aligned.
    if x_hist.shape[0] > config.max_rows:
        pick = rng.choice(x_hist.shape[0], size=config.max_rows, replace=False)
        x_hist, x_up, y_future, group = x_hist[pick], x_up[pick], y_future[pick], group[pick]

    # Split by recording (group), never by anchor: anchors from one recording are correlated, so an
    # anchor-level split would leak the eval distribution into the fit and inflate every number.
    groups = np.unique(group)
    rng.shuffle(groups)
    n_eval = max(1, int(round(config.eval_frac * groups.size)))
    if groups.size - n_eval < 1:
        raise ValueError(
            f"only {groups.size} recording(s) after the row cap; need >= 2 for a fit/eval split"
        )
    eval_groups = set(groups[:n_eval].tolist())
    fit_val_groups = groups[n_eval:]
    n_val = max(1, int(round(config.val_frac * fit_val_groups.size)))
    val_groups = set(fit_val_groups[:n_val].tolist())

    is_eval = np.isin(group, list(eval_groups))
    is_val = np.isin(group, list(val_groups))
    is_fit = ~is_eval & ~is_val

    def split(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return mat[is_fit], mat[is_val], mat[is_eval]

    hist_fit, hist_val, hist_eval = split(x_hist)
    up_fit, up_val, up_eval = split(x_up)
    y_fit, y_val, y_eval = split(y_future)

    # Baseline: target history only. Full: target history + source history.
    base_fit, base_val, base_eval = hist_fit, hist_val, hist_eval
    full_fit = np.concatenate([hist_fit, up_fit], axis=1)
    full_val = np.concatenate([hist_val, up_val], axis=1)
    full_eval = np.concatenate([hist_eval, up_eval], axis=1)

    base_pred, base_lam = _fit_predict_ridge(base_fit, y_fit, base_val, y_val, base_eval, _RIDGE_LAMBDAS)
    full_pred, full_lam = _fit_predict_ridge(full_fit, y_fit, full_val, y_val, full_eval, _RIDGE_LAMBDAS)

    # Shuffle control: feed each eval anchor a STRANGER's source through the SAME fitted full model.
    # Permute the source rows across recordings; a genuine coupling collapses this uplift to ~0.
    perm = _derange_across_groups(group[is_eval], rng)
    full_eval_shuffled = np.concatenate([hist_eval, up_eval[perm]], axis=1)
    # Re-standardise/predict with the full model's fit statistics by re-running the closed form on
    # the fit split (cheap) so the shuffled eval passes through identical weights.
    x_mean, x_std = _standardize_fit(full_fit)
    y_mean = y_fit.mean(axis=0)
    weight = _ridge_fit((full_fit - x_mean) / x_std, y_fit - y_mean, full_lam)
    shuffled_pred = ((full_eval_shuffled - x_mean) / x_std) @ weight + y_mean

    mse_base = _pooled_mse(base_pred, y_eval)
    mse_full = _pooled_mse(full_pred, y_eval)
    mse_shuffled = _pooled_mse(shuffled_pred, y_eval)
    uplift_abs = mse_base - mse_full
    uplift_rel = uplift_abs / mse_base if mse_base > 0 else 0.0
    uplift_shuffled_rel = (mse_base - mse_shuffled) / mse_base if mse_base > 0 else 0.0

    n_off = len(config.horizon_offsets)
    off_base = _per_offset_mse(base_pred, y_eval, n_off)
    off_full = _per_offset_mse(full_pred, y_eval, n_off)
    per_offset = [
        {
            "offset_steps": int(o),
            "offset_seconds": int(o) * 4,
            "mse_base": float(b),
            "mse_full": float(f),
            "uplift_rel": float((b - f) / b) if b > 0 else 0.0,
        }
        for o, b, f in zip(config.horizon_offsets, off_base, off_full)
    ]

    verdict = _verdict(uplift_rel, uplift_shuffled_rel)
    return {
        "verdict": verdict,
        "uplift_rel": float(uplift_rel),
        "uplift_abs": float(uplift_abs),
        "uplift_shuffled_rel": float(uplift_shuffled_rel),
        "mse_base": mse_base,
        "mse_full": mse_full,
        "mse_full_shuffled": mse_shuffled,
        "r2_base": _pooled_r2(base_pred, y_eval),
        "r2_full": _pooled_r2(full_pred, y_eval),
        "per_offset": per_offset,
        "ridge_lambda_base": base_lam,
        "ridge_lambda_full": full_lam,
        "n_rows_total": int(x_hist.shape[0]),
        "n_rows_fit": int(is_fit.sum()),
        "n_rows_eval": int(is_eval.sum()),
        "n_recordings": int(groups.size),
        "uplift_rel_floor": _UPLIFT_REL_FLOOR,
    }


def _derange_across_groups(eval_group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a row permutation that sends each eval anchor to a row from a *different* recording.

    Falls back to a plain shuffle when a strict derangement is not reachable (a single dominant
    recording), which only weakens the control, never corrupts the matched result.

    Args:
        eval_group: Per-eval-row recording id $(N_e,)$.
        rng: RNG for the permutation.

    Returns:
        An index array $(N_e,)$ into the eval rows.
    """
    n = eval_group.shape[0]
    order = rng.permutation(n)
    # Repair fixed points (same-recording pairings) by rotating them among themselves.
    same = eval_group[order] == eval_group
    if same.any() and (~same).any():
        bad = np.where(same)[0]
        rotated = np.roll(bad, 1)
        order[bad] = order[rotated]
    return order


def _verdict(uplift_rel: float, uplift_shuffled_rel: float) -> str:
    """Classify the run as ``coupling_present``, ``data_ceiling`` or ``ambiguous``.

    ``coupling_present`` requires the matched uplift to clear the floor *and* to survive the shuffle
    (the matched uplift must be at least twice the shuffled one), so an uplift explained by mere
    overfitting to source statistics is not mistaken for genuine pairing.

    Args:
        uplift_rel: Held-out relative uplift with the matched source.
        uplift_shuffled_rel: Held-out relative uplift with a stranger's source.

    Returns:
        The verdict slug.
    """
    if uplift_rel < _UPLIFT_REL_FLOOR:
        return "data_ceiling"
    if uplift_rel >= 2.0 * max(uplift_shuffled_rel, 0.0):
        return "coupling_present"
    return "ambiguous"


def nonlinear_uplift(samples: Sequence[Sample], config: ProbeConfig) -> Optional[Dict[str, float]]:
    r"""Optional nonlinear check: gradient-boosted uplift on a single scalar target.

    Ridge is a *linear* lower bound; if it finds nothing, a nonlinear model might still. This fits
    :class:`sklearn.ensemble.HistGradientBoostingRegressor` on one reduced scalar target -- the
    channel-mean of the far-horizon future block -- for the baseline and full designs, and reports
    the held-out MSE uplift. Returns ``None`` if scikit-learn is not installed.

    Args:
        samples: The recordings to draw anchors from.
        config: The probe configuration.

    Returns:
        ``{'uplift_rel', 'mse_base', 'mse_full'}`` on the scalar target, or ``None``.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:  # noqa: BLE001 - optional dependency; absence is a documented no-op
        return None

    rng = np.random.default_rng(config.seed)
    x_hist, x_up, y_future, group = build_design(samples, config)
    if x_hist.shape[0] > config.max_rows:
        pick = rng.choice(x_hist.shape[0], size=config.max_rows, replace=False)
        x_hist, x_up, y_future, group = x_hist[pick], x_up[pick], y_future[pick], group[pick]

    # Reduce the target to one scalar: the mean over channels of the last horizon offset's block.
    n_off = len(config.horizon_offsets)
    per_channel = y_future.shape[1] // n_off
    target = y_future.reshape(y_future.shape[0], n_off, per_channel)[:, -1, :].mean(axis=1)

    groups = np.unique(group)
    rng.shuffle(groups)
    n_eval = max(1, int(round(config.eval_frac * groups.size)))
    eval_groups = set(groups[:n_eval].tolist())
    is_eval = np.isin(group, list(eval_groups))
    is_fit = ~is_eval

    def fit_score(design: np.ndarray) -> float:
        model = HistGradientBoostingRegressor(max_depth=4, max_iter=200, learning_rate=0.05)
        model.fit(design[is_fit], target[is_fit])
        return float(np.mean((model.predict(design[is_eval]) - target[is_eval]) ** 2))

    mse_base = fit_score(x_hist)
    mse_full = fit_score(np.concatenate([x_hist, x_up], axis=1))
    return {
        "uplift_rel": float((mse_base - mse_full) / mse_base) if mse_base > 0 else 0.0,
        "mse_base": mse_base,
        "mse_full": mse_full,
    }


# --- Reporting --------------------------------------------------------------------------------
def format_report(results: Dict[str, object], nonlinear: Optional[Dict[str, float]]) -> str:
    """Render a human-readable report of a probe result.

    Args:
        results: The dict from :func:`run_probe`.
        nonlinear: The dict from :func:`nonlinear_uplift`, or ``None`` if it was skipped.

    Returns:
        The formatted multi-line report.
    """
    def num(key: str) -> float:
        """Pull a numeric result as a plain float (the dict is typed ``object`` for JSON)."""
        return float(results[key])  # type: ignore[arg-type]

    lines = ["", "=" * 74, "Stage 0 - model-free UP->FHR predictive-uplift probe", "=" * 74]
    lines.append(
        f"recordings={results['n_recordings']}  rows(total/fit/eval)="
        f"{results['n_rows_total']}/{results['n_rows_fit']}/{results['n_rows_eval']}"
    )
    lines.append(
        f"ridge lambda base/full = {num('ridge_lambda_base'):g} / {num('ridge_lambda_full'):g}"
    )
    lines.append("")
    lines.append(f"  MSE  baseline (FHR history only)        = {num('mse_base'):.6f}")
    lines.append(f"  MSE  full     (+ lagged UP source)      = {num('mse_full'):.6f}")
    lines.append(f"  MSE  shuffled (+ a stranger's UP)       = {num('mse_full_shuffled'):.6f}")
    lines.append(f"  R^2  baseline / full                    = {num('r2_base'):.4f} / {num('r2_full'):.4f}")
    lines.append("")
    lines.append(f"  UPLIFT  matched  (rel)                  = {num('uplift_rel')*100:+.3f} %")
    lines.append(f"  UPLIFT  shuffled (rel, control ~0)      = {num('uplift_shuffled_rel')*100:+.3f} %")
    lines.append(f"  floor for 'extractable'                 = {num('uplift_rel_floor')*100:.3f} %")
    lines.append("")
    lines.append("  per horizon offset:")
    lines.append("     offset      MSE_base   MSE_full   uplift_rel")
    for row in results["per_offset"]:  # type: ignore[index]
        lines.append(
            f"     {row['offset_steps']:>3d}st/{row['offset_seconds']:>3d}s   "
            f"{row['mse_base']:.5f}    {row['mse_full']:.5f}   {row['uplift_rel']*100:+.3f} %"
        )
    if nonlinear is not None:
        lines.append("")
        lines.append(
            f"  nonlinear (GBM, scalar far-horizon target) uplift_rel = "
            f"{nonlinear['uplift_rel']*100:+.3f} %"
        )
    lines.append("")
    lines.append(f"  VERDICT: {str(results['verdict']).upper()}")
    lines.append(_verdict_gloss(str(results["verdict"])))
    lines.append("=" * 74)
    return "\n".join(lines)


def _verdict_gloss(verdict: str) -> str:
    """One-line interpretation for each verdict slug."""
    return {
        "coupling_present": (
            "  -> A linear model extracts source uplift that vanishes under shuffle. Coupling is\n"
            "     present in the data; the VAE's ~0 pred_gap is a ROUTING failure. Proceed to the\n"
            "     architecture change (close the posterior's direct-h_y laundering door)."
        ),
        "data_ceiling": (
            "  -> No extractable source uplift, even linearly. The source's marginal value over a\n"
            "     strong FHR self-baseline is at/below the noise floor at this feature set and\n"
            "     horizon. Architecture will not manufacture it; the lever is the data/feature/\n"
            "     horizon side (Stage 3). If the nonlinear check also reads ~0, this is firm."
        ),
        "ambiguous": (
            "  -> Matched uplift clears the floor but does not clearly beat the shuffle control, so\n"
            "     it may be source marginal-statistics rather than pairing. Re-run with more\n"
            "     recordings / rows before deciding."
        ),
    }.get(verdict, "")


# --- Self-test (no dataset required) ----------------------------------------------------------
def _fabricate_samples(
    *, coupled: bool, n_records: int, seq_len: int, rng: np.random.Generator
) -> List[Sample]:
    r"""Fabricate recordings with or without a known injected UP$\to$FHR coupling.

    The target is a per-channel AR(1) process (strong self-predictability, like FHR). When
    ``coupled``, a fixed fraction of the *future* target is driven by the source at a known lag, so
    a correctly built probe must find held-out uplift that vanishes under the shuffle control. When
    not, the source is independent noise and the uplift must stay at ~0.

    Args:
        coupled: Whether to inject the source->target dependence.
        n_records: Number of recordings to fabricate.
        seq_len: Sequence length $T$.
        rng: RNG.

    Returns:
        The fabricated samples.
    """
    c_y, c_u = 6, 4
    inject_lag = 12          # source acts on the target this many steps later
    ar = 0.85                # target self-persistence
    coupling = 0.6           # weight of the injected source term

    samples: List[Sample] = []
    for record in range(n_records):
        u = rng.standard_normal((seq_len, c_u)).astype(np.float32)
        y = np.zeros((seq_len, c_y), dtype=np.float32)
        noise = 0.3 * rng.standard_normal((seq_len, c_y)).astype(np.float32)
        for t in range(1, seq_len):
            y[t] = ar * y[t - 1] + noise[t]
            if coupled and t - inject_lag >= 0:
                # Route the first source channel into the first two target channels at a fixed lag.
                y[t, 0] += coupling * u[t - inject_lag, 0]
                y[t, 1] += coupling * u[t - inject_lag, 1]
        weight = np.ones(seq_len, dtype=np.float32)
        samples.append(Sample(y=y, u=u, weight=weight, guid=f"rec{record}"))
    return samples


def self_test() -> int:
    """Validate the probe against a known coupling and a null; return a process exit code.

    Fabricates a coupled dataset and a null dataset, runs the probe on both, and asserts the probe
    reports ``coupling_present`` (with a collapsing shuffle control) on the first and
    ``data_ceiling`` on the second. This exercises the whole pipeline -- design construction, the
    by-recording split, ridge selection, uplift and the shuffle control -- with no HDF5 dataset.

    Returns:
        ``0`` if both assertions pass, ``1`` otherwise.
    """
    rng = np.random.default_rng(0)
    config = ProbeConfig(
        hist_lags=(0, 1, 2, 4, 8),
        up_lags=(0, 4, 8, 12, 16, 20),
        horizon_offsets=(1, 6, 12),
        warmup=8,
        seed=0,
    )

    coupled = run_probe(_fabricate_samples(coupled=True, n_records=40, seq_len=200, rng=rng), config)
    null = run_probe(_fabricate_samples(coupled=False, n_records=40, seq_len=200, rng=rng), config)

    print(format_report(coupled, None))
    print(format_report(null, None))

    ok = True
    if coupled["verdict"] != "coupling_present":
        print(f"FAIL: coupled dataset -> {coupled['verdict']} (expected coupling_present)")
        ok = False
    matched = float(coupled["uplift_rel"])  # type: ignore[arg-type]
    shuffled = float(coupled["uplift_shuffled_rel"])  # type: ignore[arg-type]
    if matched <= 3.0 * max(shuffled, 0.0):
        print("FAIL: coupled uplift did not clearly beat its shuffle control")
        ok = False
    if null["verdict"] != "data_ceiling":
        print(f"FAIL: null dataset -> {null['verdict']} (expected data_ceiling)")
        ok = False
    print("\nSELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --- CLI --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and run either the self-test or a real probe.

    Args:
        argv: Argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to the YAML config (resolves its base: chain).")
    parser.add_argument("--split", default="test", choices=("train", "test", "both"),
                        help="Which shard list to read (default: the held-out test shards).")
    parser.add_argument("--max-samples", type=int, default=1500, help="Cap on recordings loaded (default 1500).")
    parser.add_argument("--max-rows", type=int, default=60000, help="Cap on anchor rows fed to the solver.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the split, subsample and shuffle control.")
    parser.add_argument("--nonlinear", action="store_true", help="Also run the gradient-boosting nonlinear check.")
    parser.add_argument("--json-out", default=None, help="Optional path to write the results dict as JSON.")
    parser.add_argument("--self-test", action="store_true", help="Run the no-dataset correctness self-test and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.config is None:
        parser.error("--config is required unless --self-test is given")

    config = ProbeConfig(max_rows=args.max_rows, seed=args.seed)
    samples = load_samples_from_config(args.config, split=args.split, max_samples=args.max_samples)
    print(f"loaded {len(samples)} recordings from the {args.split} split of {args.config}")
    results = run_probe(samples, config)
    nonlinear = nonlinear_uplift(samples, config) if args.nonlinear else None
    print(format_report(results, nonlinear))

    if args.json_out:
        payload = dict(results)
        if nonlinear is not None:
            payload["nonlinear"] = nonlinear
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


# --- IDE Run-button configuration -------------------------------------------------------------
#: Used only when the module is launched with **no** command-line arguments -- i.e. a PyCharm/IDE
#: Run button. The moment any ``--flag`` is passed the CLI wins and every constant here is ignored,
#: so the two launch styles never fight. Edit these instead of adding a Run Configuration.
#:
#: The config path is repo-root-relative; a no-argument launch switches the working directory to
#: the repo root first, so it resolves regardless of what the IDE set the working directory to
#: (and so do the shard/stats paths *inside* the config when they are repo-root-relative, as in
#: ``tiny.yaml``). ``default.yaml``'s own shard paths are absolute, so they need no such help.
RUN_CONFIG: Optional[str] = "teb_vae/lag_attn/configs/default.yaml"
RUN_SELF_TEST: bool = False      # True -> run the no-dataset correctness self-test instead of a probe
RUN_SPLIT: str = "test"          # 'train' | 'test' | 'both' -- which shard list to read
RUN_MAX_SAMPLES: int = 1500      # cap on recordings loaded
RUN_MAX_ROWS: int = 60000        # cap on anchor rows fed to the solver
RUN_SEED: int = 0                # seed for the split, subsample and shuffle control
RUN_NONLINEAR: bool = True       # also run the gradient-boosting nonlinear check
RUN_JSON_OUT: Optional[str] = None  # path to also dump results as JSON, or None


def _run_button_argv() -> List[str]:
    """Build a CLI ``argv`` from the ``RUN_*`` constants for a no-argument (IDE Run button) launch.

    Returns:
        The argument vector, equivalent to what the user would type on the command line.

    Raises:
        SystemExit: If a probe run is requested but ``RUN_CONFIG`` is ``None``.
    """
    if RUN_SELF_TEST:
        return ["--self-test"]
    if RUN_CONFIG is None:
        raise SystemExit(
            "RUN_CONFIG is None: set it near the bottom of this file (or set RUN_SELF_TEST=True), "
            "or launch with --config on the command line."
        )
    argv = [
        "--config", RUN_CONFIG,
        "--split", RUN_SPLIT,
        "--max-samples", str(RUN_MAX_SAMPLES),
        "--max-rows", str(RUN_MAX_ROWS),
        "--seed", str(RUN_SEED),
    ]
    if RUN_NONLINEAR:
        argv.append("--nonlinear")
    if RUN_JSON_OUT:
        argv += ["--json-out", RUN_JSON_OUT]
    return argv


if __name__ == "__main__":
    # No CLI arguments means a PyCharm/IDE Run button: fall back to the RUN_* constants above and
    # switch to the repo root so repo-root-relative config/shard paths resolve. An explicit --flag
    # takes the normal argparse path and is left to resolve paths against the current directory,
    # exactly as a command-line launch expects.
    if len(sys.argv) == 1:
        _argv = _run_button_argv()
        if os.path.abspath(os.getcwd()) != _REPO_ROOT:
            print(f"no CLI args (IDE Run button); using RUN_* constants and chdir to {_REPO_ROOT}")
            os.chdir(_REPO_ROOT)
        raise SystemExit(main(_argv))
    raise SystemExit(main())
