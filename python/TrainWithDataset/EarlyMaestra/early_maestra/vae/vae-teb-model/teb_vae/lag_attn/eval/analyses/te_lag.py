r"""At what lag does the source information arrive, and which head carried it?

``te_lag_map`` redistributes the per-step surrogate transfer entropy $K_t$ across the lag axis:

$$\widetilde{TE}_{t,\ell} = \sum_m K^{(m)}_t\,\alpha^{(m)}_{t,\ell},
\qquad \sum_\ell \widetilde{TE}_{t,\ell} = K_t$$

The identity is a model contract, already pinned by ``test_kl_report.py``. What this module has to
establish is the *eval-side* property: that time-averaging and dead-anchor exclusion preserve it
in the aggregate. They are not free.

**Dead anchors break the identity, and silently.** ``_ablate_dead_anchors`` zeroes an attention
row rather than renormalising it, so at a dead anchor $\sum_\ell \widetilde{TE}_{t,\ell} = 0$
while $K_t$ stays positive. Averaging those anchors in does not merely add noise -- it subtracts
mass from the lag profile in proportion to how many anchors a band mask killed, which is largest
for exactly the long-lag bands an ablation most wants to compare. The support therefore excludes
them, and the identity is checked *at runtime* on every run rather than only in a test, because
the thing most likely to break it is a support that drifts, not a formula that changes.

**The per-head decomposition is a claim, and it has a precondition.** ``kld_per_t_per_head`` is
emitted whatever the configuration -- it is always the contiguous
``view(B, T, M, d_z // M).sum(-1)`` -- so its shares sum to $K_t$ as a property of the *view*, on
any model, including one where the flag is off and every latent dimension depends on every head.
That is why the sum-to-one identity is not what this module asserts and not what
``head_structured_latent`` gates: the guard is on the *attribution*, and under a flat latent the
per-head numbers are refused rather than reported with a footnote. Refused means the columns are
**absent from the per-sample table**, not zero-filled and not null-filled. A null share reads as
"computed, no data" and a set of shares summing to $1$ reads as a decomposition; under a flat
latent it is neither, so the only rendering that cannot be misread is no column at all. The guard
therefore sits at emission, in :func:`_per_batch_te_lag`, and not only at the per-head section,
which refuses the figure and the summary entry but runs *after* the table has been written.

**Absolute per-head KL is reported beside the share.** At initialisation, and on a genuinely
collapsed run, $K_t \equiv 0$ and every share is $0/0$. A share is undefined there -- zero is a
legitimate share, so it cannot double as "no data" -- but the absolute $K^{(m)}$ is a perfectly
good $0$, so the table stays informative and the count of undefined samples is recorded rather
than left for a reader to infer from a column of nulls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, masks, metrics, preflight, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner, get_field, to_numpy

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. $\bar{K}$ per cohort is the closest this pipeline comes to a per-cohort
#: transfer entropy, and the selected lag per cohort is what would make it a clinical statement.
GROUPED_METRICS = ("kld_mean", "argmax_lag")

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "te_lag"

#: ``eval_config.caps`` key naming this analysis's retention cap.
CAP_NAME = "te_lag"

#: Column prefixes for the flattened per-lag and per-head values.
_LAG_PREFIX = "te_l"
_HEAD_KL_PREFIX = "k_head"
_HEAD_SHARE_PREFIX = "share_head"

#: Relative deviation above which the runtime identity check is reported as violated. Loose
#: enough for fp32 summation over $L$ lags and $T$ anchors, far tighter than the drift a wrong
#: support produces -- excluding one dead anchor in thirty moves this by percent, not by $10^{-4}$.
IDENTITY_TOLERANCE = 1e-4


def support_mean(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    r"""Average over the anchor axis under a $(B, T)$ support, yielding ``NaN`` where it is empty.

    Args:
        values: $(B, T)$ or $(B, T, K)$.
        support: Anchor support, $(B, T)$.

    Returns:
        $(B,)$ or $(B, K)$.
    """
    counts = support.sum(dim=1)
    weights = support if values.dim() == 2 else support.unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    denominator = counts if values.dim() == 2 else counts.unsqueeze(-1)
    return torch.where(
        denominator > 0,
        total / denominator.clamp_min(1.0),
        torch.full_like(total, float("nan")),
    )


def _per_batch_te_lag(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    r"""Compute one batch's support-averaged lag attribution and per-head decomposition.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value, with the lag profile flattened one column per lag. The
        per-head columns are present only on a head-structured checkpoint; see below.
    """
    outputs = runner.forward(batch)
    alpha = outputs["attn_weights"]
    support = masks.lag_readout_support(runner.model, alpha, get_field(batch, "weight"))

    te_mean = support_mean(outputs["te_lag_map"], support)
    kld_mean = support_mean(outputs["kld_per_t"], support)

    # The runtime identity reading. Relative rather than absolute: K_t spans orders of magnitude
    # between a collapsed run and a healthy one, and an absolute tolerance would be vacuous on
    # the first and unachievable on the second.
    attributed = te_mean.sum(dim=1)
    deviation = (attributed - kld_mean).abs() / kld_mean.abs().clamp_min(
        torch.finfo(kld_mean.dtype).tiny
    )

    columns: Dict[str, Any] = {
        "kld_mean": kld_mean,
        "te_attributed_total": attributed,
        "identity_rel_deviation": deviation,
        "n_support_anchors": support.sum(dim=1),
        "argmax_lag": torch.where(
            support.sum(dim=1) > 0,
            torch.nan_to_num(te_mean, nan=-float("inf")).argmax(dim=1),
            torch.full((te_mean.shape[0],), -1, dtype=torch.long, device=te_mean.device),
        ),
    }
    for lag in range(int(te_mean.shape[1])):
        columns[f"{_LAG_PREFIX}{lag:03d}"] = te_mean[:, lag]

    # The per-head decomposition is a claim about attribution and it has a precondition, so the
    # columns exist only where the claim does. ``kld_per_t_per_head`` itself is emitted whatever
    # the flag -- heads.py takes the contiguous view(B, T, M, d_z // M).sum(-1) unconditionally --
    # so without this guard a flat-latent run ships a full set of finite shares summing to 1.000,
    # which looks exactly like a valid decomposition and is an arbitrary partition of a quantity
    # every head contributed to. That is an ablation-run hazard specifically: production ships
    # head_structured_latent=True, so the only tables carrying the bad numbers are the flat-latent
    # ones a reader would be comparing against. Gated on the same accessor that labels the map,
    # so the table can never carry per-head shares while calling itself a diagnostic.
    if preflight.te_lag_map_label(runner) != "attribution":
        return columns

    per_head_mean = support_mean(outputs["kld_per_t_per_head"], support)
    shares = per_head_mean / kld_mean.unsqueeze(-1).clamp_min(
        torch.finfo(kld_mean.dtype).tiny
    )
    # A share is undefined at K = 0, and zero is a legitimate share -- so NaN, not 0. The count
    # of samples in that state is recorded in the summary rather than left to be inferred.
    shares = torch.where(
        kld_mean.unsqueeze(-1) > 0, shares, torch.full_like(shares, float("nan"))
    )
    for head in range(int(per_head_mean.shape[1])):
        columns[f"{_HEAD_KL_PREFIX}{head}"] = per_head_mean[:, head]
        columns[f"{_HEAD_SHARE_PREFIX}{head}"] = shares[:, head]
    return columns


def head_lag_profile(
    runner: EvalRunner, loader: Any, max_samples: Optional[int]
) -> np.ndarray:
    r"""Accumulate the head-resolved lag profile $K^{(m)}_t \alpha^{(m)}_{t,\ell}$, $(M, L)$.

    The quantity head structuring actually claims: not "head $m$ carries this much KL" and not
    "head $m$ looks at this lag", but the product -- how much of the source information arrived
    through head $m$ *at lag $\ell$*. Summing it over $m$ recovers ``te_lag_map``, which is what
    makes it a decomposition rather than two independent readouts plotted side by side.

    Averaged across the split rather than retained per sample: the per-head, per-lag, per-sample
    tensor is $(N, M, L)$, and the figure that consumes this draws one $(M, L)$ panel.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        max_samples: Prefix cap on iteration.

    Returns:
        The support-weighted mean profile, $(M, L)$.

    Raises:
        TEPreconditionUnmet: If the checkpoint has ``head_structured_latent=False``.
    """
    preflight.require_head_structured_latent(runner, "the per-head lag profile")

    total: Optional[torch.Tensor] = None
    weight_total = 0.0
    for batch in runner.iter_batches(loader, max_samples=max_samples):
        outputs = runner.forward(batch)
        alpha = outputs["attn_weights"]
        support = masks.lag_readout_support(runner.model, alpha, get_field(batch, "weight"))

        # (B, T, M) x (B, T, M, L) -> (M, L), support-weighted.
        contribution = outputs["kld_per_t_per_head"].unsqueeze(-1) * alpha
        weighted = (contribution * support[:, :, None, None]).sum(dim=(0, 1))
        total = weighted if total is None else total + weighted
        weight_total += float(support.sum())

    if total is None or weight_total <= 0.0:
        return np.zeros((0, 0), dtype=np.float64)
    return to_numpy(total / weight_total).astype(np.float64)


def _lag_columns(frame: pd.DataFrame) -> list:
    """Return the per-lag attribution column names, in lag order."""
    return sorted(
        name
        for name in frame.columns
        if name.startswith(_LAG_PREFIX) and name[len(_LAG_PREFIX):].isdigit()
    )


def _write_te_figure(frame: pd.DataFrame, lag_columns: list, directory: Path) -> str:
    """Draw the lag attribution ribbon, the argmax histogram and the identity residual.

    Args:
        frame: The per-sample frame.
        lag_columns: The per-lag column names, in lag order.
        directory: The analysis directory.

    Returns:
        The path written.
    """
    profile = frame[lag_columns].to_numpy(dtype=np.float64) if lag_columns else np.zeros((0, 0))
    lags = np.arange(len(lag_columns), dtype=np.float64)

    figure, axes = figures.new_figure(3)
    try:
        figures.ribbon_plot(
            axes[0, 0], lags, profile,
            title="Lag-resolved transfer-entropy surrogate "
                  "$\\widetilde{TE}_{t,\\ell}$, averaged over the valid support",
            xlabel="Model lag $\\ell$",
            ylabel="$\\widetilde{TE}_\\ell$ (nats)",
            label="median over samples",
        )
        seconds = axes[0, 0].secondary_xaxis(
            "top",
            functions=(
                lambda lag: metrics.lag_to_seconds(lag),
                lambda sec: sec / metrics.STEP_SECONDS,
            ),
        )
        seconds.set_xlabel("Physical delay (s)", fontsize=8)

        # $-1$ is the "no supported anchor" sentinel written by ``_per_batch_te_lag``, not a lag.
        # ``run_te_lag_analysis`` strips it before ``median_argmax_lag``; stripping it here too is
        # what keeps the drawn median and the reported one the same number.
        argmax_lag = frame["argmax_lag"] if "argmax_lag" in frame else pd.Series(dtype=float)
        figures.histogram_panel(
            axes[1, 0], argmax_lag[argmax_lag >= 0],
            title="Per-sample argmax of the lag attribution",
            xlabel="Model lag $\\ell$", bins=max(len(lag_columns), 1),
            color=figures.COLOR_PURPLE,
        )
        figures.histogram_panel(
            axes[2, 0], frame.get("identity_rel_deviation", pd.Series(dtype=float)),
            title="Identity residual $|\\sum_\\ell \\widetilde{TE}_\\ell - K| / |K|$ "
                  "after eval's averaging",
            xlabel="relative deviation", color=figures.COLOR_GRAY,
            reference=IDENTITY_TOLERANCE, reference_label="tolerance",
        )
        figure.suptitle(
            f"Lag axis: seconds = {metrics.STEP_SECONDS:g}$\\ell$ on the stored timeline. "
            f"Dead anchors are excluded: their attention rows are zeroed, not renormalised, so "
            f"averaging them in would subtract mass from the profile.",
            fontsize=7, y=0.999,
        )
        return str(figures.render_figure(figure, directory / "te_lag"))
    finally:
        figures.plt.close(figure)


def _write_per_head_figure(profile: np.ndarray, frame: pd.DataFrame, directory: Path) -> str:
    r"""Draw the head-resolved lag profile and the per-head shares of $K_t$.

    Args:
        profile: The $(M, L)$ head-resolved lag profile.
        frame: The per-sample frame, for the share bars.
        directory: The analysis directory.

    Returns:
        The path written.
    """
    share_columns = sorted(
        name for name in frame.columns if name.startswith(_HEAD_SHARE_PREFIX)
    )
    figure, axes = figures.new_figure(2)
    try:
        figures.heatmap_with_colorbar(
            figure, axes[0, 0], profile,
            title="Head-resolved lag profile "
                  "$K^{(m)}_t\\,\\alpha^{(m)}_{t,\\ell}$ (support mean)",
            xlabel="Model lag $\\ell$", ylabel="Head $m$",
            cmap="viridis", symmetric=False, colorbar_label="nats",
        )
        top = axes[0, 0].secondary_xaxis(
            "top",
            functions=(
                lambda lag: metrics.lag_to_seconds(lag),
                lambda sec: sec / metrics.STEP_SECONDS,
            ),
        )
        top.set_xlabel("Physical delay (s)", fontsize=8)

        ax = axes[1, 0]
        means = [float(frame[name].mean()) for name in share_columns]
        ax.bar(
            range(len(share_columns)), means, color=figures.COLOR_BLUE,
            edgecolor=figures.COLOR_BLACK, linewidth=0.4,
        )
        ax.set_title("Mean per-head share of $K_t$")
        ax.set_xlabel("Head $m$")
        ax.set_ylabel("share")
        ax.set_xticks(range(len(share_columns)))
        figures.style_axes(ax)
        figure.suptitle(
            "Rows sum to the lag attribution; shares sum to 1. Both are decompositions only "
            "because head_structured_latent is on -- under a flat latent this panel is refused.",
            fontsize=7, y=0.999,
        )
        return str(figures.render_figure(figure, directory / "per_head_lag_profile"))
    finally:
        figures.plt.close(figure)


def run_te_lag_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the lag attribution, its runtime identity check, and the per-head decomposition.

    The ``causal_norm`` guard runs first and refuses the whole analysis: without it $K_t$ is not
    a transfer entropy and nothing downstream is worth computing. The ``head_structured_latent``
    guard is narrower -- it refuses the per-head panel *and* the per-head columns of the
    per-sample table, while the lag attribution is still emitted, labelled as a diagnostic rather
    than an attribution in both ``summary.json`` and the table's own ``te_lag_map_label`` column.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, for the sample count and per-file grouping.

    Returns:
        The headline summary for ``summary.json``.

    Raises:
        TEPreconditionUnmet: If the checkpoint was built with ``causal_norm=False``.
    """
    preflight.require_causal_norm(runner, "te_lag_map")

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    caps = eval_config.get("caps") or {}
    max_samples = eval_config.get("max_samples")
    n_total = int((probe or {}).get("n_samples") or 0)
    plan = (
        CollectionPlan.build(
            n_total, caps.get(CAP_NAME), int(eval_config.get("seed", 0)),
            groups=(probe or {}).get("source_files"),
        )
        if n_total
        else None
    )

    collected = collect_metrics(
        runner, loader, _per_batch_te_lag,
        max_samples=max_samples, plan=plan, progress_label="te_lag",
    )
    frame = collected.frame
    lag_columns = _lag_columns(frame)

    if "argmax_lag" in frame:
        frame = frame.assign(
            argmax_lag_seconds=metrics.lag_seconds_physical(
                frame["argmax_lag"].to_numpy()
            )
        )
    # The table carries its own interpretation marker rather than relying on the reader still
    # holding the summary.json it was written beside. A per-sample CSV outlives that pairing --
    # it gets copied into a notebook and diffed against another run's -- and the ``te_l*`` columns
    # of a flat-latent run mean something different from a head-structured run's while being
    # column-for-column identical. One constant column makes the two tables self-describing
    # instead of silently incomparable.
    label = preflight.te_lag_map_label(runner)
    frame = frame.assign(te_lag_map_label=label)
    frame.to_csv(directory / "te_lag_mean_per_sample.csv", index=False)

    deviations = (
        frame["identity_rel_deviation"].to_numpy(dtype=np.float64)
        if "identity_rel_deviation" in frame
        else np.zeros(0)
    )
    finite_deviations = deviations[np.isfinite(deviations)]
    max_deviation = float(finite_deviations.max()) if finite_deviations.size else float("nan")
    identity_holds = bool(finite_deviations.size and max_deviation <= IDENTITY_TOLERANCE)

    if finite_deviations.size and not identity_holds:
        logger.warning(
            f"the lag attribution identity does not survive eval's averaging: max relative "
            f"deviation {max_deviation:.3e} exceeds {IDENTITY_TOLERANCE:.0e}. The likeliest "
            f"cause is an anchor support that includes rows the model zeroed -- dead anchors "
            f"sum to zero against a nonzero K_t -- rather than a change to the attribution "
            f"itself, which test_kl_report.py pins as a model contract."
        )

    summary: Dict[str, Any] = {
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "num_lags": len(lag_columns),
        # The seconds convention of every lag column and axis here: the stored timeline.
        "step_seconds": float(metrics.STEP_SECONDS),
        "te_lag_map_label": label,
        "identity": {
            "holds": identity_holds,
            "tolerance": IDENTITY_TOLERANCE,
            "max_rel_deviation": max_deviation,
            "mean_rel_deviation": (
                float(finite_deviations.mean()) if finite_deviations.size else float("nan")
            ),
            "checked": "sum over lags of the support-averaged te_lag_map against the "
                       "support-averaged kld_per_t, per sample",
        },
        "mean_kld": float(frame["kld_mean"].mean()) if "kld_mean" in frame else float("nan"),
        "figures": [],
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS),
            stem="te_lag_mean_per_sample",
        ),
    }
    if "argmax_lag" in frame:
        argmax = frame["argmax_lag"].to_numpy()
        argmax = argmax[argmax >= 0]
        summary["median_argmax_lag"] = float(np.median(argmax)) if argmax.size else float("nan")
        summary["median_argmax_lag_seconds"] = (
            float(metrics.lag_seconds_physical(np.median(argmax)))
            if argmax.size
            else float("nan")
        )

    summary["figures"].append(_write_te_figure(frame, lag_columns, directory))
    summary["per_head"] = _per_head_section(
        runner, loader, frame, directory, max_samples, summary
    )

    logger.info(
        f"te_lag: identity holds={identity_holds} (max rel deviation {max_deviation:.3e}); "
        f"median argmax lag {summary.get('median_argmax_lag')}; label={label}"
    )
    return summary


def _per_head_section(
    runner: EvalRunner,
    loader: Any,
    frame: pd.DataFrame,
    directory: Path,
    max_samples: Optional[int],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Emit the per-head decomposition, or record why it was refused.

    Caught here rather than allowed to propagate: the ``head_structured_latent`` precondition
    invalidates the per-head attribution alone, and failing the whole step would throw away the
    lag attribution, which remains valid as a diagnostic on such a checkpoint.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        frame: The per-sample frame.
        directory: The analysis directory.
        max_samples: Prefix cap on iteration.
        summary: The analysis summary, extended with the figure path on success.

    Returns:
        Either the per-head record, or ``{'available': False, 'reason': ...}``.
    """
    try:
        profile = head_lag_profile(runner, loader, max_samples)
    except preflight.TEPreconditionUnmet as unmet:
        logger.warning(f"per-head decomposition refused: {unmet}")
        return {"available": False, "reason": str(unmet)}

    share_columns = sorted(
        name for name in frame.columns if name.startswith(_HEAD_SHARE_PREFIX)
    )
    kl_columns = sorted(name for name in frame.columns if name.startswith(_HEAD_KL_PREFIX))
    identity = ["sample_index", "guid", "source_file"]
    if share_columns:
        per_head = frame[identity + kl_columns + share_columns].melt(
            id_vars=identity, var_name="column", value_name="value"
        )
        per_head["head"] = per_head["column"].str.extract(r"(\d+)$").astype(int)
        per_head["quantity"] = np.where(
            per_head["column"].str.startswith(_HEAD_SHARE_PREFIX), "share", "kld"
        )
        per_head.drop(columns=["column"]).sort_values(
            ["sample_index", "head", "quantity"]
        ).to_csv(directory / "per_head.csv", index=False)

    summary["figures"].append(_write_per_head_figure(profile, frame, directory))

    undefined = int(frame["kld_mean"].le(0).sum()) if "kld_mean" in frame else 0
    if undefined:
        logger.warning(
            f"{undefined} sample(s) have K = 0 over the support, so their per-head shares are "
            f"undefined and reported as null. The absolute per-head KL is reported regardless."
        )
    return {
        "available": True,
        "n_heads": int(profile.shape[0]) if profile.size else 0,
        "mean_share": {
            name.replace(_HEAD_SHARE_PREFIX, "head_"): float(frame[name].mean())
            for name in share_columns
        },
        "n_samples_with_zero_kl": undefined,
    }
