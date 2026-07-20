r"""Is the source information *specific* to this recording's source, or merely source-shaped?

Every other analysis in this pipeline establishes that the source pathway carries information.
None of them establishes that the information is about *this* recording. A model that had learned
the marginal statistics of the UP stream -- and nothing about the pairing -- would show a healthy
$K_t$, a live residual and a real uplift. This is the control that separates the two: feed the
decoder a latent inferred from a **different** recording's source and re-score against the true
future.

**The criterion is the prediction-space ordering:**

$$L_{\mathrm{feat}} < L_{\mathrm{base}} < L_{\mathrm{feat,\ shuffled}}$$

A matched source beats the target-only baseline; a mismatched source is *worse* than no source at
all. That middle term is what makes it a specificity test rather than a sensitivity one.

**The KL-space reading is reported and is deliberately not the criterion.** $K_{\mathrm{shuffled}}$
measures whether the source moved the posterior, which a mismatched source does too -- often more
strongly, since the posterior only ever trained on matched pairs and a stranger's source is out
of distribution. So $K_{\mathrm{shuffled}} \gtrsim K_{\mathrm{true}}$ is routinely observed on
models that plainly do use the source, and reading it as a failure is the specific misreading
this module exists to make hard. :func:`source_specificity_verdict` therefore takes the three
losses and *nothing else*: the KL cannot flip the verdict because it is not an argument to it.

**Two RNG sources, and seeding one is not enough.** ``perm_forward_outputs`` samples $z$ through
``model.reparameterize``, which calls ``torch.randn_like`` -- a function that takes no generator
and draws from the global RNG. The ``generator=`` argument seeds only the derangement. A run that
passed a generator and left the global RNG alone would produce a different forecast on every
rerun while looking carefully seeded, so the global RNG is pinned per batch as well.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, masks, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner, get_field
from teb_vae.lag_attn.nets import controls

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. Source *specificity* holding on the healthy majority while failing on the
#: pathological cohorts would be invisible in the pooled ordering, and would matter more than
#: the pooled verdict does.
GROUPED_METRICS = ("shuffle_penalty", "kld_shuffled_ratio")

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "perm_control"

#: ``eval_config.caps`` key naming this analysis's retention cap.
CAP_NAME = "perm_control"

#: Column prefixes for the flattened per-step $K_t$ curves.
_TRUE_PREFIX = "kt"
_SHUFFLED_PREFIX = "ks"

#: Stated on the KL-space outputs so the reading cannot be mistaken for the criterion.
KL_READOUT_LABEL = (
    "influence, not specificity: K_shuffled >= K_true is expected on a healthy model, because a "
    "mismatched source is out of distribution and moves the posterior more, not less. The "
    "specificity criterion is the prediction-space ordering."
)


def source_specificity_verdict(
    l_feat: float, l_base: float, l_shuffled: float
) -> Dict[str, Any]:
    r"""Judge the ordering $L_{\mathrm{feat}} < L_{\mathrm{base}} < L_{\mathrm{shuffled}}$.

    Takes the three losses and nothing else. That is a deliberate structural choice rather than a
    minimal signature: the KL-space readout is the thing most likely to be reached for when this
    verdict disappoints, and a function that cannot see it cannot be swayed by it.

    Args:
        l_feat: Mean full-pathway feature loss, matched source.
        l_base: Mean baseline (target-only) feature loss.
        l_shuffled: Mean full-pathway feature loss under a deranged source.

    Returns:
        The verdict, both pairwise margins, and the three losses that produced it.
    """
    uplift = l_base - l_feat
    penalty = l_shuffled - l_base

    if not all(np.isfinite([l_feat, l_base, l_shuffled])):
        verdict = "undetermined"
        explanation = "one or more of the three losses is not finite."
    elif uplift <= 0.0:
        verdict = "no_uplift"
        explanation = (
            "the matched source does not beat the target-only baseline, so there is no "
            "information to test the specificity of. Check the residual and uplift analyses "
            "before reading anything into the shuffled term."
        )
    elif penalty <= 0.0:
        verdict = "influential_not_specific"
        explanation = (
            "the source helps, but a stranger's source is no worse than no source at all -- "
            "consistent with the model having learned the source's marginal statistics rather "
            "than the pairing. This is the outcome the control exists to detect."
        )
    else:
        verdict = "source_specific"
        explanation = (
            "the matched source beats the baseline and a mismatched source is worse than no "
            "source, which is the ordering that establishes specificity."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "l_feat": l_feat,
        "l_base": l_base,
        "l_shuffled": l_shuffled,
        "uplift_margin": uplift,
        "shuffle_penalty_margin": penalty,
        "criterion": "L_feat < L_base < L_feat_shuffled",
    }


def _make_per_batch(seed: int, state: Dict[str, int]):
    """Build the per-batch closure, carrying the batch counter the seeding needs.

    Args:
        seed: The run seed.
        state: Mutable counters -- ``index`` and ``n_skipped`` -- read back after collection.

    Returns:
        A ``(runner, batch) -> columns`` callable.
    """

    def _per_batch(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
        index = state["index"]
        state["index"] += 1

        model = runner.model
        outputs = runner.forward(batch)
        weight = get_field(batch, "weight")
        batch_size = int(outputs["mu_post"].shape[0])

        if batch_size < 2:
            # A derangement of one element does not exist. Skipped and counted rather than
            # raised on: the last batch of a split is routinely short, and a single sample is a
            # legitimate split size.
            state["n_skipped"] += batch_size
            return {}

        # Both RNG sources, per batch and derived from the run seed. The global seed is what
        # makes perm_forward_outputs' z draw reproducible; the generator only covers the
        # derangement. Seeding either alone leaves the run irreproducible.
        torch.manual_seed(seed + index)
        generator = torch.Generator().manual_seed(seed + index)

        kl_space = controls.perm_kl_from_forward(
            model, outputs, weight=weight, generator=generator
        )
        permuted = controls.perm_forward_outputs(
            model, outputs, perm_index=kl_space["perm_index"]
        )

        view = runner.forecast_view(batch, outputs)
        anchors = int(view.mu_full.shape[1])
        objective = runner.objective
        loss_kwargs = {
            "likelihood": objective.likelihood,
            "sigma_obs": objective.sigma_obs,
        }

        def _per_sample_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            return metrics.masked_per_sample_mean(
                metrics.per_element_loss(
                    (mu - view.y_plus) ** 2, logvar, **loss_kwargs
                ),
                view.mask,
            )

        l_feat = _per_sample_loss(view.mu_full, view.logvar_full)
        l_base = _per_sample_loss(view.mu_base, view.logvar_base)
        l_shuffled = _per_sample_loss(
            permuted["mu_full"][:, :anchors], permuted["logvar_full"][:, :anchors]
        )

        seq_len = int(outputs["kld_per_t"].shape[1])
        kld_mask = masks.kld_mask(
            model, weight, batch_size, seq_len, device=outputs["kld_per_t"].device
        )
        steps = kld_mask.sum(dim=1)

        def _support_mean(values: torch.Tensor) -> torch.Tensor:
            return torch.where(
                steps > 0,
                (values * kld_mask).sum(dim=1) / steps.clamp_min(1.0),
                torch.full_like(steps, float("nan")),
            )

        kld_true = _support_mean(outputs["kld_per_t"])
        kld_shuffled = _support_mean(kl_space["kld_shuffled_per_t"])

        columns: Dict[str, Any] = {
            "l_feat": l_feat,
            "l_base": l_base,
            "l_feat_shuffled": l_shuffled,
            "uplift_abs": l_base - l_feat,
            "shuffle_penalty": l_shuffled - l_base,
            # Named for their normalisation, not "kld_raw": this is the $d_z$-summed per-step KL
            # support-averaged over $t$, which is $d_z = 24$ times the ``kld_raw`` that
            # ``scalars`` writes into the same ``summary.json`` and $24\times$ ``latent``'s
            # ``kld_mean``. Three quantities differing by a factor of 24 under one name in one
            # file is a reader trap; the ratio and the verdict are unaffected either way.
            "kld_true_per_t": kld_true,
            "kld_shuffled_per_t": kld_shuffled,
            "kld_shuffled_ratio": kld_shuffled / kld_true.abs().clamp_min(
                torch.finfo(kld_true.dtype).tiny
            ),
            "perm_index": kl_space["perm_index"],
        }
        # The pooled shuffled loss, via the documented route: the permuted dict still carries the
        # *true* forward's kld_* keys by reference, so compute_kld_loss=False is mandatory rather
        # than an optimisation -- without it the reported KL would describe the matched pairing.
        pooled = runner.compute_loss(batch, permuted, compute_kld_loss=False)
        columns["pooled_feat_loss_shuffled"] = float(pooled["feat_loss"])

        for step in range(seq_len):
            columns[f"{_TRUE_PREFIX}{step:03d}"] = outputs["kld_per_t"][:, step]
            columns[f"{_SHUFFLED_PREFIX}{step:03d}"] = kl_space["kld_shuffled_per_t"][:, step]
        return columns

    return _per_batch


def _prefixed(frame: pd.DataFrame, prefix: str) -> list:
    """Return the flattened per-step columns carrying ``prefix``, in step order."""
    return sorted(
        name
        for name in frame.columns
        if name.startswith(prefix) and name[len(prefix):].isdigit()
    )


def _write_figures(
    frame: pd.DataFrame, verdict: Dict[str, Any], runner: EvalRunner, directory: Path
) -> list:
    """Draw the paired-loss plot and the per-step $K_t$ overlay.

    Args:
        frame: The per-sample frame.
        verdict: The specificity verdict, annotated onto the loss figure.
        runner: The loaded runner, for the warm-up shading.
        directory: The analysis directory.

    Returns:
        The two paths written.
    """
    paths = []
    names = ["l_feat", "l_base", "l_feat_shuffled"]
    labels = ["$L_{feat}$\n(matched)", "$L_{base}$\n(no source)", "$L_{feat}$\n(shuffled)"]

    figure, axes = figures.new_figure(2)
    try:
        ax = axes[0, 0]
        present = [name for name in names if name in frame]
        values = [frame[name].to_numpy(dtype=np.float64) for name in present]
        positions = np.arange(len(present))

        # One line per sample across the three positions: the *paired* structure is the point.
        # Three independent box plots would hide a model where the ordering holds on average but
        # is violated on most individual recordings.
        for row in range(len(frame)):
            ax.plot(
                positions, [column[row] for column in values],
                color=figures.COLOR_GRAY, alpha=0.25, linewidth=0.6, zorder=1,
            )
        for position, column in zip(positions, values):
            finite = column[np.isfinite(column)]
            if finite.size:
                ax.scatter(
                    [position], [finite.mean()], color=figures.COLOR_VERMILLION,
                    zorder=3, s=40, marker="_", linewidths=2.0,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[names.index(name)] for name in present], fontsize=7)
        ax.set_ylabel("per-sample loss")
        ax.set_title(
            f"Source-specificity control -- verdict: {verdict.get('verdict', 'n/a')}"
        )
        ax.text(
            0.5, 0.02,
            f"criterion {verdict.get('criterion', '')}; uplift margin "
            f"{verdict.get('uplift_margin', float('nan')):.4g}, shuffle penalty "
            f"{verdict.get('shuffle_penalty_margin', float('nan')):.4g}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6,
            color=figures.COLOR_GRAY,
        )
        figures.style_axes(ax)

        figures.histogram_panel(
            axes[1, 0], frame.get("shuffle_penalty", pd.Series(dtype=float)),
            title="Per-sample shuffle penalty $L_{feat,shuffled} - L_{base}$ "
                  "(positive supports specificity)",
            xlabel="nats", color=figures.COLOR_PURPLE,
            reference=0.0, reference_label="no penalty",
        )
        paths.append(str(figures.render_to_pdf(figure, directory / "losses.pdf")))
    finally:
        figures.plt.close(figure)

    true_columns = _prefixed(frame, _TRUE_PREFIX)
    shuffled_columns = _prefixed(frame, _SHUFFLED_PREFIX)
    figure, axes = figures.new_figure(1, height_per_row=3.2)
    try:
        ax = axes[0, 0]
        steps = figures.sequence_axis(len(true_columns))
        if true_columns:
            figures.ribbon_plot(
                ax, steps, frame[true_columns].to_numpy(dtype=np.float64),
                title="$K_t$ under the matched source and under a deranged one",
                xlabel="Step $t$ (decimated)", ylabel="$K_t$ (nats)",
                color=figures.COLOR_BLUE, label="$K_{true}$ median",
            )
        if shuffled_columns:
            shuffled = frame[shuffled_columns].to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                median = np.nanpercentile(shuffled, 50, axis=0)
            ax.plot(
                figures.sequence_axis(len(shuffled_columns)), median,
                color=figures.COLOR_ORANGE, linewidth=1.4, label="$K_{shuffled}$ median",
            )
        warmup = int(runner.model._warmup_steps(len(true_columns))) if true_columns else 0
        if warmup > 0:
            ax.axvspan(0, warmup, color=figures.COLOR_LIGHT_GRAY, alpha=0.6, zorder=0)
        ax.legend(fontsize=7, loc="best")
        # The caption is the mitigation for a high-likelihood misreading, so it is part of the
        # figure rather than of the surrounding prose.
        ax.text(
            0.5, 0.02,
            "K_shuffled >= K_true is EXPECTED and is NOT a failure: a mismatched source is out "
            "of distribution and moves the posterior more. Specificity is decided in prediction "
            "space, not here.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6,
            color=figures.COLOR_VERMILLION, wrap=True,
        )
        figures.style_axes(ax)
        paths.append(str(figures.render_to_pdf(figure, directory / "kl_overlay.pdf")))
    finally:
        figures.plt.close(figure)
    return paths


def run_perm_control_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run both permutation controls and emit the source-specificity verdict.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, for the sample count and per-file grouping.

    Returns:
        The headline summary for ``summary.json``, carrying the verdict at its top level.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    caps = eval_config.get("caps") or {}
    seed = int(eval_config.get("seed", 0))
    n_total = int((probe or {}).get("n_samples") or 0)
    plan = (
        CollectionPlan.build(
            n_total, caps.get(CAP_NAME), seed, groups=(probe or {}).get("source_files")
        )
        if n_total
        else None
    )

    state = {"index": 0, "n_skipped": 0}
    collected = collect_metrics(
        runner, loader, _make_per_batch(seed, state),
        max_samples=eval_config.get("max_samples"), plan=plan, progress_label="perm_control",
    )
    frame = collected.frame

    true_columns = _prefixed(frame, _TRUE_PREFIX)
    shuffled_columns = _prefixed(frame, _SHUFFLED_PREFIX)
    frame.drop(columns=true_columns + shuffled_columns).to_csv(
        directory / "per_sample.csv", index=False
    )

    def _mean(column: str) -> float:
        if column not in frame:
            return float("nan")
        values = frame[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    verdict = source_specificity_verdict(
        _mean("l_feat"), _mean("l_base"), _mean("l_feat_shuffled")
    )
    figure_paths = _write_figures(frame, verdict, runner, directory)

    if state["n_skipped"]:
        logger.warning(
            f"{state['n_skipped']} sample(s) were skipped because their batch held fewer than "
            f"two samples, which cannot be deranged. Their control keys are absent from "
            f"per_sample.csv rather than zero-filled: a zero there would scale the mean of "
            f"feat_loss_shuffled toward zero and invert the very ordering this control checks."
        )

    penalty = (
        frame["shuffle_penalty"].to_numpy(dtype=np.float64)
        if "shuffle_penalty" in frame
        else np.zeros(0)
    )
    penalty = penalty[np.isfinite(penalty)]

    summary: Dict[str, Any] = {
        "n_samples": int(len(frame)),
        "n_skipped_undersized_batches": state["n_skipped"],
        "composition": collected.composition,
        "plan": collected.plan,
        "seed": seed,
        "specificity": verdict,
        "kl_space": {
            "label": KL_READOUT_LABEL,
            "mean_kld_true_per_t": _mean("kld_true_per_t"),
            "mean_kld_shuffled_per_t": _mean("kld_shuffled_per_t"),
            "mean_kld_shuffled_ratio": _mean("kld_shuffled_ratio"),
            "normalisation": (
                "d_z-summed per-step KL, support-averaged over t. This is d_z times "
                "scalars.kld_raw and d_z times latent.kld_mean; compare ratios, not levels."
            ),
            "shuffled_exceeds_true": bool(
                np.isfinite(_mean("kld_shuffled_per_t"))
                and np.isfinite(_mean("kld_true_per_t"))
                and _mean("kld_shuffled_per_t") > _mean("kld_true_per_t")
            ),
        },
        "mean_pooled_feat_loss_shuffled": _mean("pooled_feat_loss_shuffled"),
        # Scored over the samples that *have* a penalty, not over every row. An undersized final
        # batch cannot be deranged, so ``_per_batch`` returns ``{}`` for it and those rows are
        # NaN here -- and ``np.nan > 0`` is ``False``, which silently counted every unscored
        # sample as evidence *against* a shuffle penalty. That contradicts this module's own
        # absent-is-not-zero policy (see the n_skipped warning above) and matches how
        # ``uplift`` already filters. ``n_scored`` is reported so the denominator is visible.
        "positive_shuffle_penalty_frac": (
            float((penalty > 0).mean()) if penalty.size else float("nan")
        ),
        "n_shuffle_penalty_scored": int(penalty.size),
        "figures": figure_paths,
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS),
            references={"shuffle_penalty": 0.0, "kld_shuffled_ratio": 1.0},
        ),
    }

    logger.info(
        f"perm_control: verdict={verdict['verdict']} (L_feat={verdict['l_feat']:.6g} "
        f"L_base={verdict['l_base']:.6g} L_shuffled={verdict['l_shuffled']:.6g}); "
        f"K_shuffled/K_true={summary['kl_space']['mean_kld_shuffled_ratio']:.4g} "
        f"-- {KL_READOUT_LABEL}"
    )
    return summary
