r"""Neural + empirical CMI comparison for the v3 transfer-entropy surrogate (G11, Sprint 6).

The v3 model reports a transfer-entropy surrogate :math:`K_{\mathrm{raw}} =
\mathrm{KL}(q(z\mid Y,U)\,\|\,p(z\mid Y))` that is derived from the model's *own* latent KL.
Nothing internal to that quantity proves it tracks genuine source :math:`\to` target
information flow rather than an artefact of the two variance heads. This module corroborates
it against two *independent* estimates of the conditional mutual information
:math:`I(U_{\le t}; Y^+ \mid c_t)`:

1. A **neural CMI lower bound** (this file): a small critic trained per evaluation with two
   bounds -- **InfoNCE/CPC** (low-variance, bounded by :math:`\log N`) and **MINE**
   (Donsker--Varadhan, unbounded) -- reported side by side and cross-checked at rank level.
2. The **empirical transfer-entropy** tooling under ``TE_Calculated/`` (an IDTxl estimate),
   wired in as an optional baseline column when a precomputed CSV is supplied.

Conditioning. The exact conditioning set :math:`Y_{\le t}` is variable-length and
high-dimensional, so the estimator conditions on the model's own fixed-dimensional causal
target summary :math:`c_t = \texttt{target\_state}`. This couples the estimate to the model
and is why the deliverable is **rank-level** corroboration of :math:`K_{\mathrm{raw}}` -- not a
magnitude match. Per-sample summaries are anchor-means over the training support
:math:`[w_{\mathrm{warm}}, T-H)`, giving one :math:`(u, y, c)` triple per sample and therefore
a per-sample CMI density that aligns directly with the per-sample :math:`K_{\mathrm{raw}}`.

Both bounds are optimised by gradient ascent. MINE is trained in numerically-stable log-space
(the :math:`\log\,\mathrm{mean}\,\exp` form of the DV bound) rather than the exp-space
EMA-corrected form; the two share the same optimum, and the log-space objective avoids the
overflow that motivates the EMA trick -- appropriate here because MINE is the *cross-check*,
not the objective. The estimate is in-sample (fit and read on the same held-out subset), which
is standard for a variational MI lower bound.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import collect_cmi_features
from model.vae_teb_prediction.model.model_raw.testing.metrics import _rankdata_1d

try:  # pragma: no cover - the GUID bootstrap lives beside the causal-TE validation suite
    from model.vae_teb_prediction.model.model_raw.testing.causal_te_validation.statistics import (
        guid_bootstrap_ci,
    )
except Exception as exc:  # noqa: BLE001
    logger.warning(f"cmi_comparison: guid_bootstrap_ci unavailable ({exc}); CIs disabled")
    guid_bootstrap_ci = None  # type: ignore[assignment]

try:  # pragma: no cover - plotting is optional and must never fail the analysis
    from model.vae_teb_prediction.model.model_raw.testing.visualizers import plot_cmi_comparison

    _PLOTTING = True
except Exception as exc:  # noqa: BLE001
    logger.warning(f"cmi_comparison: plotting unavailable ({exc}); CSVs will still be written")
    _PLOTTING = False

_BOUNDS = ("infonce", "mine")


# =============================================================================
# Neural CMI estimator (S6-T01a)
# =============================================================================


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    """Build a GELU MLP with ``depth`` hidden layers.

    Args:
        in_dim: Input feature dimension.
        hidden: Hidden width.
        out_dim: Output embedding dimension.
        depth: Number of hidden layers (``>= 1``).

    Returns:
        The assembled :class:`torch.nn.Sequential`.
    """
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(max(0, depth - 1)):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class NeuralCMIEstimator(nn.Module):
    r"""Separable-critic lower-bound estimator of :math:`I(U; Y^+ \mid c)`.

    The critic factorises as a scaled dot product of two embeddings -- a *query* over the
    source-and-context pair :math:`[u, c]` and a *key* over the future target :math:`y` -- so
    the full :math:`N \times N` cross-pair score matrix is a single matrix product:

    .. math:: S_{ij} = \frac{\langle \phi([u_i, c_i]),\, \psi(y_j)\rangle}{\sqrt{k}}.

    Conditioning on :math:`c` is applied on the query side only and negatives are drawn from
    the batch marginal of :math:`y`; this is the pragmatic conditional-CPC approximation the
    spec pins (the estimate is model-coupled, hence rank-level).

    Both bounds return ``(bound, pointwise)`` where ``pointwise[i]`` is the per-sample MI
    density of triple :math:`i` -- averaged over a sample's rows it is that sample's CMI
    estimate, which aligns with the per-sample :math:`K_{\mathrm{raw}}`.

    Args:
        dim_u: Source-summary dimension.
        dim_y: Future-target-summary dimension.
        dim_c: Conditioning (target-state) summary dimension.
        hidden: Hidden width of both MLPs.
        embed: Output embedding dimension :math:`k`.
        depth: Hidden-layer count of both MLPs.
    """

    def __init__(
        self,
        dim_u: int,
        dim_y: int,
        dim_c: int,
        *,
        hidden: int = 256,
        embed: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.query = _mlp(dim_u + dim_c, hidden, embed, depth)
        self.key = _mlp(dim_y, hidden, embed, depth)
        self._embed = int(embed)

    def scores(self, u: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        r"""Return the ``(N, N)`` scaled dot-product score matrix :math:`S_{ij}`.

        Args:
            u: Source summaries ``(N, dim_u)``.
            y: Future-target summaries ``(N, dim_y)``.
            c: Conditioning summaries ``(N, dim_c)``.

        Returns:
            The ``(N, N)`` matrix with ``S[i, j]`` scoring query ``i`` against key ``j``.
        """
        phi = self.query(torch.cat([u, c], dim=-1))  # (N, k)
        psi = self.key(y)  # (N, k)
        return phi @ psi.t() / math.sqrt(self._embed)

    def infonce_bound(
        self, u: torch.Tensor, y: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""InfoNCE / CPC lower bound :math:`I \ge \log N - \mathcal{L}_{\mathrm{NCE}}`.

        Args:
            u: Source summaries ``(N, dim_u)``.
            y: Future-target summaries ``(N, dim_y)``.
            c: Conditioning summaries ``(N, dim_c)``.

        Returns:
            ``(bound, pointwise)``: the scalar bound (capped at :math:`\log N`) and the
            ``(N,)`` per-sample density :math:`S_{ii} - \log\sum_j e^{S_{ij}} + \log N`.
        """
        s = self.scores(u, y, c)
        n = s.shape[0]
        diag = s.diagonal()
        lse = torch.logsumexp(s, dim=1)
        pointwise = diag - lse + math.log(n)
        return pointwise.mean(), pointwise

    def mine_bound(
        self, u: torch.Tensor, y: torch.Tensor, c: torch.Tensor, *, clip: float = 5.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""MINE / Donsker--Varadhan lower bound, clipped (SMILE-style) for stability.

        .. math:: I \ge \frac{1}{N}\sum_i \mathrm{clip}(S_{ii},\,-\tau,\,\tau)
                  - \Big(\log\!\!\sum_{i\ne j} e^{\mathrm{clip}(S_{ij},\,-\tau,\,\tau)}
                  - \log[N(N-1)]\Big).

        The plain DV estimator is notoriously unstable: the partition
        :math:`\log\,\mathrm{mean}\,\exp` is dominated by the largest score, and -- because the
        critic here trains *on* this objective -- an unclipped joint term is maximised by
        driving the diagonal to infinity. Clipping the log-ratios to :math:`[-\tau, \tau]`
        (the SMILE estimator, Song & Ermon 2020) bounds the whole objective, so the critic
        cannot diverge and, by Jensen, an independent :math:`(U, Y)` sits at :math:`\le 0`.
        :math:`\tau \to \infty` recovers exact MINE. Computed in stable log-space.

        Args:
            u: Source summaries ``(N, dim_u)``.
            y: Future-target summaries ``(N, dim_y)``.
            c: Conditioning summaries ``(N, dim_c)``.
            clip: Symmetric clip :math:`\tau` on the log-ratios (both joint and marginal).

        Returns:
            ``(bound, pointwise)``: the scalar bound and the ``(N,)`` per-sample density
            :math:`\mathrm{clip}(S_{ii}) - \log\,\mathrm{mean}\,\exp(\mathrm{clip}(S_{
            \mathrm{marg}}))` (the log-partition is a shared constant, so it does not change
            the per-sample ranking).
        """
        s = self.scores(u, y, c).clamp(min=-float(clip), max=float(clip))
        n = s.shape[0]
        joint = s.diagonal()
        off = ~torch.eye(n, dtype=torch.bool, device=s.device)
        marg = s[off]
        log_mean_exp = torch.logsumexp(marg, dim=0) - math.log(marg.numel())
        pointwise = joint - log_mean_exp
        return joint.mean() - log_mean_exp, pointwise


# =============================================================================
# Fit loop + synthetic recovery (S6-T01b)
# =============================================================================


def _as_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """Coerce a numpy array / tensor to a float32 tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _bound_fn(est: NeuralCMIEstimator, bound: str):
    """Return the estimator's bound method for ``'infonce'`` / ``'mine'``."""
    if bound not in _BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; expected one of {_BOUNDS}")
    return est.infonce_bound if bound == "infonce" else est.mine_bound


def fit_cmi_estimator(
    u: Any,
    y: Any,
    c: Any,
    *,
    bound: str = "infonce",
    n_iters: int = 300,
    lr: float = 1e-3,
    hidden: int = 128,
    embed: int = 32,
    depth: int = 2,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[NeuralCMIEstimator, Dict[str, Any]]:
    r"""Fit a :class:`NeuralCMIEstimator` by gradient ascent on the chosen bound.

    This is the fit primitive; it reports the **in-sample** bound (which overfits with a
    flexible critic on finite data). Use :func:`estimate_cmi` for an honest, cross-fitted
    estimate suitable for comparison.

    Args:
        u: Source summaries ``(N, dim_u)`` (numpy or tensor).
        y: Future-target summaries ``(N, dim_y)``.
        c: Conditioning summaries ``(N, dim_c)``.
        bound: ``'infonce'`` or ``'mine'``.
        n_iters: Full-batch optimisation steps.
        lr: Adam learning rate.
        hidden: Critic hidden width.
        embed: Critic embedding dimension.
        depth: Critic hidden-layer count.
        weight_decay: Adam weight decay (a mild regulariser on the critic).
        seed: RNG seed for the deterministic critic initialisation.
        device: Compute device; defaults to CPU.

    Returns:
        ``(estimator, history)`` where ``history`` carries the final in-sample ``bound`` value,
        the ``(N,)`` per-sample ``pointwise`` density (numpy), the loss ``trajectory``, and the
        resolved capacity (``critic_hidden`` / ``critic_embed`` / ``n_iters``).

    Raises:
        ValueError: If ``bound`` is unknown or ``N < 2`` (no negatives available).
    """
    dev = device or torch.device("cpu")
    ut, yt, ct = (_as_tensor(v, dev) for v in (u, y, c))
    n = int(ut.shape[0])
    if n < 2:
        raise ValueError(f"need >= 2 samples for a lower bound, got {n}")

    torch.manual_seed(int(seed))
    est = NeuralCMIEstimator(
        dim_u=int(ut.shape[1]), dim_y=int(yt.shape[1]), dim_c=int(ct.shape[1]),
        hidden=hidden, embed=embed, depth=depth,
    ).to(dev)
    opt = torch.optim.Adam(est.parameters(), lr=lr, weight_decay=weight_decay)
    bound_fn = _bound_fn(est, bound)

    trajectory: list[float] = []
    est.train()
    for _ in range(int(n_iters)):
        opt.zero_grad(set_to_none=True)
        value, _ = bound_fn(ut, yt, ct)
        (-value).backward()
        opt.step()
        trajectory.append(float(value.detach()))

    est.eval()
    with torch.no_grad():
        final_bound, pointwise = bound_fn(ut, yt, ct)
    return est, {
        "bound": float(final_bound),
        "pointwise": pointwise.detach().cpu().numpy(),
        "trajectory": trajectory,
        "critic_hidden": int(hidden),
        "critic_embed": int(embed),
        "critic_depth": int(depth),
        "n_iters": int(n_iters),
        "n_samples": n,
    }


def estimate_cmi(
    u: Any,
    y: Any,
    c: Any,
    *,
    bound: str = "infonce",
    n_folds: int = 2,
    seed: int = 42,
    device: Optional[torch.device] = None,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    r"""Cross-fitted neural CMI lower bound with per-sample densities.

    A single in-sample bound overfits: a flexible critic memorises which :math:`y` pairs with
    which :math:`(u, c)` even when they are independent, inflating the estimate. Cross-fitting
    fixes it -- the critic is trained on the other folds and the bound is read on the held-out
    fold, where memorised associations do not transfer, so an independent :math:`(U, Y)`
    collapses to :math:`\approx 0` while a genuine dependence survives. Stitching the
    out-of-fold densities gives an honest per-sample CMI for **every** sample, aligned to the
    per-sample :math:`K_{\mathrm{raw}}`.

    Args:
        u: Source summaries ``(N, dim_u)``.
        y: Future-target summaries ``(N, dim_y)``.
        c: Conditioning summaries ``(N, dim_c)``.
        bound: ``'infonce'`` or ``'mine'``.
        n_folds: Number of cross-fitting folds (``>= 2``).
        seed: RNG seed for the fold shuffle and each critic's init.
        device: Compute device; defaults to CPU.
        **fit_kwargs: Forwarded to :func:`fit_cmi_estimator` (``n_iters``, ``hidden`` ...).

    Returns:
        A dict with the scalar out-of-fold ``bound`` (mean of the per-sample densities), the
        ``(N,)`` per-sample ``pointwise`` density (numpy), ``per_fold_bounds``, and the
        resolved ``capacity`` record.

    Raises:
        ValueError: If ``bound`` is unknown or ``N`` is too small for ``n_folds`` folds with a
            usable held-out negative set.
    """
    dev = device or torch.device("cpu")
    ut, yt, ct = (_as_tensor(v, dev) for v in (u, y, c))
    n = int(ut.shape[0])
    n_folds = max(2, int(n_folds))
    if n < 2 * n_folds:
        raise ValueError(f"need >= {2 * n_folds} samples for {n_folds}-fold cross-fit, got {n}")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    folds = [perm[i::n_folds] for i in range(n_folds)]

    pointwise = np.full(n, np.nan, dtype=np.float64)
    per_fold: list[float] = []
    capacity: Dict[str, Any] = {}
    for k, eval_idx in enumerate(folds):
        train_idx = torch.cat([folds[j] for j in range(n_folds) if j != k])
        if int(eval_idx.numel()) < 2:
            continue
        est, hist = fit_cmi_estimator(
            ut[train_idx], yt[train_idx], ct[train_idx],
            bound=bound, seed=seed + k, device=dev, **fit_kwargs,
        )
        capacity = {
            "critic_hidden": hist["critic_hidden"],
            "critic_embed": hist["critic_embed"],
            "critic_depth": hist["critic_depth"],
            "n_iters": hist["n_iters"],
            "n_folds": n_folds,
        }
        est.eval()
        with torch.no_grad():
            fold_bound, fold_pw = _bound_fn(est, bound)(
                ut[eval_idx], yt[eval_idx], ct[eval_idx]
            )
        pointwise[eval_idx.cpu().numpy()] = fold_pw.detach().cpu().numpy()
        per_fold.append(float(fold_bound))

    finite = pointwise[np.isfinite(pointwise)]
    return {
        "bound": float(finite.mean()) if finite.size else float("nan"),
        "pointwise": pointwise,
        "per_fold_bounds": per_fold,
        "capacity": capacity,
        "n_samples": n,
    }


# =============================================================================
# Rank correlation (signed Spearman; reuses metrics._rankdata_1d)
# =============================================================================


def signed_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Signed Spearman rank correlation with finite/constant guards.

    Mirrors :func:`metrics._safe_abs_spearman` but keeps the sign, which matters here: a
    genuine surrogate should be *positively* rank-correlated with an independent CMI estimate.

    Args:
        x: First sample vector.
        y: Second sample vector.

    Returns:
        Signed :math:`\\rho \\in [-1, 1]`, or ``0.0`` when fewer than three finite pairs exist
        or either ranking is constant.
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return 0.0
    xr = _rankdata_1d(xa[mask])
    yr = _rankdata_1d(ya[mask])
    if float(np.std(xr)) <= 0.0 or float(np.std(yr)) <= 0.0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


# =============================================================================
# Empirical-TE adapter (S6-T03)
# =============================================================================


def _attach_empirical_te(
    per_sample: pd.DataFrame, empirical_te_csv: Optional[Union[str, Path]]
) -> bool:
    """Attach a per-sample ``ite_valid`` column mapped from a precomputed empirical-TE CSV.

    Uses the guid-level mean from :func:`TE_Calculated.cmi_adapter.load_empirical_te_by_guid`
    (the per-sample :math:`K_{\\mathrm{raw}}` is an anchor summary, so empirical TE aligns at
    the patient level). Degrades to a logged skip -- no column added -- when the CSV is absent,
    malformed, or shares no GUID with the evaluated subset.

    Args:
        per_sample: The per-sample frame (mutated in place to add ``ite_valid``).
        empirical_te_csv: Optional path to the IDTxl empirical-TE CSV.

    Returns:
        ``True`` if an ``ite_valid`` column with any finite value was attached.
    """
    if empirical_te_csv is None:
        return False
    try:
        from model.vae_teb_prediction.model.model_raw.testing.TE_Calculated.cmi_adapter import (
            load_empirical_te_by_guid,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"cmi_comparison: empirical-TE adapter unavailable ({exc}); skipping")
        return False

    mapping = load_empirical_te_by_guid(empirical_te_csv)
    if not mapping:
        return False
    values = per_sample["guid"].map(lambda g: mapping.get(str(g), np.nan))
    if not np.isfinite(values.to_numpy(dtype=float)).any():
        logger.warning("cmi_comparison: empirical-TE CSV shares no GUID with the subset; skip")
        return False
    per_sample["ite_valid"] = values.to_numpy(dtype=float)
    return True


# =============================================================================
# CMI comparison analysis (S6-T02)
# =============================================================================


def _bootstrap_ci(values: np.ndarray, guids: np.ndarray, *, n_boot: int, seed: int):
    """Patient-level bootstrap CI via :func:`guid_bootstrap_ci`, or ``(nan, nan)`` if disabled."""
    if guid_bootstrap_ci is None:
        return (float("nan"), float("nan"))
    return guid_bootstrap_ci(values, guids, n_boot=n_boot, seed=seed)


def run_cmi_comparison(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = 1000,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    bounds: Sequence[str] = _BOUNDS,
    n_folds: int = 2,
    n_iters: int = 300,
    hidden: int = 128,
    embed: int = 32,
    depth: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
    n_boot: int = 1000,
    empirical_te_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    r"""Corroborate :math:`K_{\mathrm{raw}}` against neural CMI (and optional empirical TE).

    Collects per-sample :math:`(u, y, c, K_{\mathrm{raw}})`, fits a cross-fitted neural CMI
    lower bound per requested ``bound`` (InfoNCE and/or MINE), and reports how each aligns with
    :math:`K_{\mathrm{raw}}` at rank level, with patient-level bootstrap CIs. When an empirical-
    TE CSV is supplied its patient-mean is added as a third column.

    Args:
        runner: The configured :class:`TestRunner`.
        loader: Dataloader over the evaluation subset.
        max_samples: Cap on samples consumed; ``<= 0`` skips the analysis entirely.
        output_dir: Destination; defaults to ``runner.ensure_dir("cmi_comparison")``.
        bounds: Which neural bounds to fit (subset of ``('infonce', 'mine')``).
        n_folds: Cross-fitting folds for the honest held-out estimate.
        n_iters: Critic optimisation steps per fold.
        hidden: Critic hidden width.
        embed: Critic embedding dimension.
        depth: Critic hidden-layer count.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        seed: RNG seed.
        n_boot: Bootstrap iterations for the patient-level CIs.
        empirical_te_csv: Optional precomputed IDTxl empirical-TE CSV path.

    Returns:
        The scalar summary dict extended with artefact paths. On a model with no
        ``target_state``/``source_state``/``kld_per_t`` the dict carries an ``error`` entry
        instead of raising, so the pipeline's step harness records the skip and continues.
    """
    if max_samples is not None and max_samples <= 0:
        logger.info("cmi_comparison: skipped (max_samples <= 0)")
        return {}

    try:
        collected = collect_cmi_features(runner, loader, max_samples)
    except RuntimeError as exc:
        logger.warning(f"cmi_comparison: {exc}")
        return {"error": str(exc)}

    n = int(collected["n_samples"])
    if n < 2 * int(n_folds):
        msg = f"only {n} samples; need >= {2 * int(n_folds)} for a {n_folds}-fold estimate"
        logger.warning(f"cmi_comparison: {msg}")
        return {"error": msg, "n_samples": n}

    u, y, c = collected["u"], collected["y"], collected["c"]
    k_raw = collected["k_raw"]
    guids = collected["guids"]
    per_sample = pd.DataFrame({
        "guid": guids,
        "label": collected["labels"],
        "k_raw": k_raw,
    })

    summary: Dict[str, Any] = {"n_samples": n, "n_guids": int(len(set(map(str, guids))))}
    capacity: Dict[str, Any] = {}
    for bound in bounds:
        est = estimate_cmi(
            u, y, c, bound=bound, n_folds=n_folds, n_iters=n_iters, hidden=hidden,
            embed=embed, depth=depth, lr=lr, weight_decay=weight_decay, seed=seed,
        )
        col = f"cmi_{bound}"
        per_sample[col] = est["pointwise"]
        summary[f"{col}_bound"] = est["bound"]
        summary[f"spearman_kraw_{bound}"] = signed_spearman(k_raw, est["pointwise"])
        capacity = est["capacity"]
    if "infonce" in bounds and "mine" in bounds:
        summary["spearman_infonce_mine"] = signed_spearman(
            per_sample["cmi_infonce"].to_numpy(), per_sample["cmi_mine"].to_numpy()
        )
    summary["capacity"] = capacity

    has_empirical = _attach_empirical_te(per_sample, empirical_te_csv)
    if has_empirical:
        summary["spearman_kraw_empirical"] = signed_spearman(
            k_raw, per_sample["ite_valid"].to_numpy()
        )

    # Patient-level bootstrap CIs on every reported quantity.
    quantity_cols = ["k_raw", *[f"cmi_{b}" for b in bounds]]
    if has_empirical:
        quantity_cols.append("ite_valid")
    table_rows = []
    for col in quantity_cols:
        vals = per_sample[col].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(vals, guids, n_boot=n_boot, seed=seed)
        finite = vals[np.isfinite(vals)]
        row = {
            "quantity": col,
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
            "spearman_vs_kraw": (
                1.0 if col == "k_raw" else signed_spearman(k_raw, vals)
            ),
            "n": int(np.isfinite(vals).sum()),
        }
        table_rows.append(row)
        summary[f"{col}_mean"] = row["mean"]
        summary[f"{col}_ci"] = [lo, hi]
    comparison_table = pd.DataFrame(table_rows)

    out = Path(output_dir) if output_dir is not None else runner.ensure_dir("cmi_comparison")
    out.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(out / "per_sample.csv", index=False)
    comparison_table.to_csv(out / "comparison_table.csv", index=False)

    if _PLOTTING:
        try:
            plot_cmi_comparison(comparison_table, per_sample, out / "cmi_comparison.pdf")
        except Exception as exc:  # noqa: BLE001 - a bad figure must not lose the CSVs
            logger.warning(f"cmi_comparison: plotting failed ({exc})")

    summary["per_sample_csv"] = str(out / "per_sample.csv")
    summary["comparison_table_csv"] = str(out / "comparison_table.csv")
    summary["has_empirical_te"] = has_empirical
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    logger.info(
        "cmi_comparison: {} samples; rho(K_raw, InfoNCE)={:.3f}, rho(K_raw, MINE)={:.3f}",
        n, summary.get("spearman_kraw_infonce", float("nan")),
        summary.get("spearman_kraw_mine", float("nan")),
    )
    return summary
