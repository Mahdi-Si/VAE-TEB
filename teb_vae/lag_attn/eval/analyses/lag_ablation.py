r"""Which lags actually *matter*? Attention says where the model looks; this says what it costs.

The attention diagnostics and ``te_lag_map`` are both descriptions of the model's internal state.
Neither is a causal statement: a head can place mass on a lag that contributes nothing to the
forecast. This analysis restricts the attention to one band of lags with ``lag_band_mask``,
re-runs the forward, and measures what the forecast lost -- which is the only readout here that
answers "does this lag matter" rather than "does the model attend to it".

**The mask keeps, it does not remove -- so this measures sufficiency, not necessity.**
``masks.lag_band_keep_mask`` sets ``mask[lo : hi + 1] = True`` and the model combines it as
``validity & band`` (``nets/model.py``, whose own argument documentation calls it a *keep-mask*),
so a band's forward runs with **only** that band available. The sign therefore reads opposite to
a removal ablation, and this is the single easiest thing to get backwards here:

- A **small** ``feat_mse_delta`` means that band *alone* nearly reproduced the unmasked forecast.
  That band is **sufficient** -- it carries the source information.
- A **large** ``feat_mse_delta`` means that band alone was not enough. It says the *rest* of the
  window carried what this band lacks; it does **not** say this band mattered more.

Read as a removal ablation the ranking inverts exactly, which on a model whose UP influence lives
at short lags would publish the longest lags as the important ones. The summary reports both ends
explicitly -- ``most_sufficient_band`` ($\min$) and ``least_sufficient_band`` ($\max$) -- rather
than a single "most damaging" band, because that phrase has no correct reading under a keep-mask.
Necessity, the complementary question, would need a keep-mask over the band's *complement*; no
analysis here measures it.

Two further rules make the per-band numbers comparable, and both are easy to get wrong in ways
that produce a plausible table.

**Every band is scored on one identical anchor support.** A band that excludes lag $0$ leaves
anchors $t < \min(\mathrm{band})$ with no causally valid lag at all; the model forces lag $0$ back
on to keep ``entmax15`` well-posed and then zeroes those rows, so those anchors ran with a source
the ablation did *not* remove. Scoring them dilutes the band's measured effect toward zero, most
severely for the long-lag bands -- which is precisely the comparison the ablation exists to make.
So the support starts at $\max(\mathrm{warmup}, \max_b \min(b))$, shared by every band **and by
the unmasked baseline**, and the anchors each band gives up are recorded rather than absorbed.

**The per-band KL is recomputed, never read from ``kld_raw``.** ``compute_loss`` reduces the KL
over the model's own band-unaware support. At a dead anchor the ablation drives the attended
source to zero, which under a head-structured posterior still produces a non-zero delta against
the prior -- so a long-lag band would fold thirty meaningless anchors into its reported KL and
read as though the ablation had *changed* something there. The KL is therefore rebuilt from
``model.kld_tensor`` restricted to the same common support the feature loss uses.

**Bands are compared under common random numbers.** ``forward`` samples $z$, so two bands scored
under independent draws differ by sampling noise as well as by their ablation -- and on a band
whose real effect is small, the noise is the larger of the two. Every band therefore re-runs from
the same RNG state, which makes the comparison paired rather than independent.

One consequence worth knowing: because the noise tensor's shape follows the batch,
``ablation_batch_size`` changes which draws are made and therefore moves every absolute number by
that same sampling noise. It bounds peak memory and does not change what is measured; it is not
a knob to tune for a result.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, masks, metrics
from teb_vae.lag_attn.eval.runner import (
    EvalRunner,
    TENSOR_FIELDS,
    batch_size_of,
    get_field,
)

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "lag_ablation"

#: Name under which the unmasked run appears in the table, beside the bands. Scored on the same
#: support as every band, or its deltas would compare two different anchor sets.
BASELINE_NAME = "unmasked"

#: Upper bound on the configured band count. The cost is one forward per band per batch on top of
#: the attention window's dense clone, so a config with fifty bands would turn a twenty-minute
#: analysis into a day's work with no warning. Raise it deliberately if a finer sweep is wanted.
MAX_BANDS = 12


def _slice_batch(batch: Any, start: int, stop: int) -> Any:
    """Return a view of ``batch`` holding samples ``[start, stop)``.

    Tensor fields are sliced on the batch axis and list-valued metadata -- ``guid``,
    ``source_file_basename`` -- is sliced as a list. Everything else is carried through, so a
    micro-batch is indistinguishable from a real one to the code that consumes it.

    Args:
        batch: A batch from the data module.
        start: First sample to keep.
        stop: One past the last sample to keep.

    Returns:
        A ``SimpleNamespace`` carrying the sliced fields.
    """
    fields: Dict[str, Any] = {}
    names = set(TENSOR_FIELDS) | {"guid", "source_file_basename", "cs_label", "bg_label"}
    for name in names:
        value = get_field(batch, name)
        if isinstance(value, torch.Tensor):
            fields[name] = value[start:stop]
        elif isinstance(value, (list, tuple)):
            fields[name] = list(value[start:stop])
        elif value is not None:
            fields[name] = value
    return SimpleNamespace(**fields)


def _micro_batches(batch: Any, size: Optional[int]) -> List[Any]:
    """Split a batch into micro-batches of at most ``size`` samples.

    Args:
        batch: A batch from the data module.
        size: Maximum micro-batch size, or ``None`` to leave the batch whole.

    Returns:
        The micro-batches, in order.
    """
    if size is None:
        return [batch]
    total = int(batch_size_of(batch))
    if total <= int(size):
        return [batch]
    return [
        _slice_batch(batch, start, min(start + int(size), total))
        for start in range(0, total, int(size))
    ]


class _Pooled:
    r"""A running mask-weighted sum and its denominator.

    A pooled mean over a split is not the mean of the per-batch means unless every batch has the
    same mask density, and the last batch of a split is routinely short. Carrying numerator and
    denominator separately is what makes the final number equal to what the whole split would
    have produced as one batch.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def add(self, value: float, weight: float) -> None:
        """Add one contribution, ignoring non-finite values and empty denominators."""
        if not np.isfinite(value) or weight <= 0.0:
            return
        self.total += float(value) * float(weight)
        self.weight += float(weight)

    def mean(self) -> float:
        """The pooled mean, or ``NaN`` when nothing contributed."""
        return self.total / self.weight if self.weight > 0.0 else float("nan")


def _score_one(
    runner: EvalRunner,
    batch: Any,
    band_mask: Optional[torch.Tensor],
    scoring_start: int,
) -> Dict[str, tuple]:
    r"""Score one micro-batch under one band, on the common support.

    Args:
        runner: The loaded runner.
        batch: A micro-batch already on the compute device.
        band_mask: The lag keep-mask, or ``None`` for the unmasked baseline.
        scoring_start: First anchor of the common support.

    Returns:
        ``{'feat_mse': (value, weight), 'kld': (value, weight)}``.
    """
    outputs = runner.forward(batch, lag_band_mask=band_mask)
    view = runner.forecast_view(batch, outputs)

    # Narrowed rather than sliced: the mask keeps its shape, so it still multiplies a full
    # error tensor and the two cannot fall out of alignment.
    mask = masks.anchor_slice_mask(view.mask, scoring_start)
    squared_error = (view.mu_full - view.y_plus) ** 2
    feat_mse = float(metrics.masked_pooled_mean(squared_error, mask))
    feat_weight = float(mask.sum()) * float(view.mu_full.shape[-1])

    seq_len = int(outputs["kld_per_t"].shape[1])
    batch_size = int(outputs["kld_per_t"].shape[0])
    kld_mask = masks.kld_mask(
        runner.model, get_field(batch, "weight"), batch_size, seq_len,
        device=outputs["kld_per_t"].device,
    )
    # Restricted to the same anchors the feature loss scored. Reading kld_raw instead would
    # reduce over the model's band-unaware support and fold dead anchors into a long-lag band.
    restricted = torch.zeros_like(kld_mask)
    stop = min(seq_len, int(view.mask.shape[1]))
    if scoring_start < stop:
        restricted[:, scoring_start:stop] = kld_mask[:, scoring_start:stop]

    kld_btd = metrics.kld_per_dim(outputs, runner.model)
    kld = float(metrics.kld_pooled(kld_btd, restricted, free_bits=0.0))
    kld_weight = float(restricted.sum()) * float(kld_btd.shape[-1])

    return {"feat_mse": (feat_mse, feat_weight), "kld": (kld, kld_weight)}


def run_lag_ablation_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ablate each configured lag band and score the cost on a common anchor support.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block. ``bands`` supplies the sweep and
            ``ablation_batch_size`` bounds the peak memory of the per-band forwards.
        output_dir: The run's results directory.
        probe: The loader probe's record. Unused here -- the ablation is a pooled measurement
            over whatever the loader yields, not a per-sample retention.

    Returns:
        The headline summary for ``summary.json``.

    Raises:
        ValueError: If no band is configured, if more than :data:`MAX_BANDS` are, or if the
            common support is empty.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    bands: Dict[str, Sequence[int]] = dict(eval_config.get("bands") or {})
    if not bands:
        raise ValueError(
            "no lag bands are configured, so there is nothing to ablate. Set eval_config.bands "
            "to a mapping of band name to an inclusive [lo, hi] lag pair."
        )
    if len(bands) > MAX_BANDS:
        raise ValueError(
            f"{len(bands)} lag bands are configured, above the bound of {MAX_BANDS}. The "
            f"ablation costs one forward per band per batch on top of the attention window's "
            f"dense clone, so a large sweep silently turns a short analysis into a very long "
            f"one. Narrow eval_config.bands, or raise MAX_BANDS deliberately."
        )

    seq_len = int(runner.model.sequence_length)
    num_lags = int(runner.num_lags)
    micro_batch = eval_config.get("ablation_batch_size")

    scoring_start = masks.common_scoring_start(runner.model, bands, seq_len)
    exclusions = masks.band_exclusion_counts(runner.model, bands, seq_len)
    _, anchor_stop = masks.valid_anchor_range(runner.model, seq_len)

    ordered = sorted(bands.items(), key=lambda item: (int(item[1][0]), item[0]))
    runs: Dict[str, Optional[torch.Tensor]] = {BASELINE_NAME: None}
    for name, band in ordered:
        runs[name] = masks.lag_band_keep_mask(band, num_lags, device=runner.device)

    accumulators = {
        name: {"feat_mse": _Pooled(), "kld": _Pooled()} for name in runs
    }
    n_samples = 0

    seed = int(eval_config.get("seed", 0))
    chunk_index = 0
    for batch in runner.iter_batches(loader, max_samples=eval_config.get("max_samples")):
        for micro in _micro_batches(batch, micro_batch):
            for name, band_mask in runs.items():
                # Common random numbers: every band re-runs from the *same* RNG state, so the
                # standard normal drawn inside `reparameterize` is identical across bands and
                # only the posterior it is scaled by differs. Without this each band would be
                # scored under its own independent z draw and the band-to-band difference --
                # which is the entire measurement -- would carry that sampling noise on top of
                # the ablation's actual effect, easily swamping a small band.
                torch.manual_seed(seed + chunk_index)
                scored = _score_one(runner, micro, band_mask, scoring_start)
                for metric, (value, weight) in scored.items():
                    accumulators[name][metric].add(value, weight)
            chunk_index += 1
        n_samples += int(batch_size_of(batch))

    baseline_mse = accumulators[BASELINE_NAME]["feat_mse"].mean()
    baseline_kld = accumulators[BASELINE_NAME]["kld"].mean()

    rows: List[Dict[str, Any]] = []
    for name in runs:
        band = bands.get(name)
        record = exclusions.get(name, {})
        feat_mse = accumulators[name]["feat_mse"].mean()
        kld = accumulators[name]["kld"].mean()
        rows.append(
            {
                "band": name,
                "lag_lo": int(band[0]) if band is not None else -1,
                "lag_hi": int(band[1]) if band is not None else -1,
                "seconds_lo": (
                    float(metrics.lag_seconds_physical(band[0]))
                    if band is not None
                    else float("nan")
                ),
                "seconds_hi": (
                    float(metrics.lag_seconds_physical(band[1]))
                    if band is not None
                    else float("nan")
                ),
                "n_kept_lags": int(band[1]) - int(band[0]) + 1 if band is not None else num_lags,
                "dead_before": record.get("dead_before", 0),
                "anchors_excluded": record.get("excluded_by_common_support", 0),
                "anchors_scored": anchor_stop - scoring_start,
                "feat_mse": feat_mse,
                "kld": kld,
                "feat_mse_delta": feat_mse - baseline_mse,
                "kld_delta": kld - baseline_kld,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "per_band.csv", index=False)

    figure_paths = _write_figures(frame, directory)

    ablated_rows = [row for row in rows if row["band"] != BASELINE_NAME]
    finite = [row for row in ablated_rows if np.isfinite(row["feat_mse_delta"])]
    # The mask keeps rather than removes, so the smallest degradation marks the band that alone
    # best reproduces the unmasked forecast. Both ends are reported: a single "most damaging"
    # band has no correct reading under a keep-mask, and naming one inverts the causal claim.
    most_sufficient = min(finite, key=lambda row: row["feat_mse_delta"])["band"] if finite else None
    least_sufficient = max(finite, key=lambda row: row["feat_mse_delta"])["band"] if finite else None

    summary: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_bands": len(ordered),
        "common_scoring_start": scoring_start,
        "anchors_scored": anchor_stop - scoring_start,
        "scoring_note": (
            "every band and the unmasked baseline are scored on the identical anchor support "
            "[common_scoring_start, T - H_d), and the per-band KL is recomputed from "
            "kld_tensor on that support rather than read from compute_loss's band-unaware "
            "kld_raw."
        ),
        "semantics": (
            "lag_band_mask KEEPS only the named band, so each row is that band run alone "
            "against the unmasked baseline. A SMALL feat_mse_delta means the band alone nearly "
            "reproduced the full forecast (sufficient); a LARGE one means it did not. This is "
            "sufficiency, not necessity -- it does not say the band is unimportant, only that "
            "it is not self-sufficient. Read as a removal ablation the ranking inverts exactly."
        ),
        "ablation_batch_size": micro_batch,
        "baseline_feat_mse": baseline_mse,
        "baseline_kld": baseline_kld,
        "per_band": {
            row["band"]: {
                "kept_lags": [row["lag_lo"], row["lag_hi"]],
                "seconds": [row["seconds_lo"], row["seconds_hi"]],
                "dead_before": row["dead_before"],
                "anchors_excluded": row["anchors_excluded"],
                "feat_mse": row["feat_mse"],
                "feat_mse_delta": row["feat_mse_delta"],
                "kld": row["kld"],
                "kld_delta": row["kld_delta"],
            }
            for row in rows
            if row["band"] != BASELINE_NAME
        },
        "most_sufficient_band": most_sufficient,
        "least_sufficient_band": least_sufficient,
        "figures": figure_paths,
    }

    logger.info(
        f"lag_ablation: {len(ordered)} band(s) scored on anchors "
        f"[{scoring_start}, {anchor_stop}); baseline feat_mse {baseline_mse:.6g}; "
        f"band that alone best reproduces the forecast: {summary['most_sufficient_band']} "
        f"(worst: {summary['least_sufficient_band']}) -- the mask keeps, so smaller is better"
    )
    return summary


def _write_figures(frame: pd.DataFrame, directory: Path) -> list:
    """Draw the per-band forecast-degradation and KL-change bar charts.

    Args:
        frame: The per-band table, including the unmasked baseline row.
        directory: The analysis directory.

    Returns:
        The two paths written.
    """
    ablated = frame[frame["band"] != BASELINE_NAME].sort_values("lag_lo").reset_index(drop=True)
    positions = np.arange(len(ablated))
    # Dual labelling: the model-lag band is what the tensor is indexed by, the physical seconds
    # are what a reader wants, and the excluded-anchor count is what makes the bars comparable.
    labels = [
        f"{row.band}\n$\\ell$ {int(row.lag_lo)}-{int(row.lag_hi)}\n"
        f"{row.seconds_lo:.0f}-{row.seconds_hi:.0f} s\n"
        f"({int(row.anchors_excluded)} anchors given up)"
        for row in ablated.itertuples()
    ]

    paths = []
    for filename, column, title, ylabel, color in (
        (
            "forecast_degradation", "feat_mse_delta",
            "Forecast degradation with ONLY this lag band kept (lower = this band alone suffices)",
            "$\\Delta$ feature MSE vs unmasked", figures.COLOR_VERMILLION,
        ),
        (
            "kl_change", "kld_delta",
            "Change in $K_t$ with ONLY this lag band kept, recomputed on the common support",
            "$\\Delta$ KL (nats)", figures.COLOR_PURPLE,
        ),
    ):
        figure, axes = figures.new_figure(1, height_per_row=3.4)
        try:
            ax = axes[0, 0]
            values = (
                ablated[column].to_numpy(dtype=np.float64)
                if column in ablated
                else np.zeros(0)
            )
            ax.bar(
                positions, values, color=color,
                edgecolor=figures.COLOR_BLACK, linewidth=0.4,
            )
            ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=0.8)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.text(
                0.5, 0.98,
                f"All bands and the unmasked baseline scored on the same "
                f"{int(ablated['anchors_scored'].iloc[0]) if len(ablated) else 0} anchors; "
                f"lag axis is the stored timeline, {metrics.STEP_SECONDS:g} s per lag.",
                transform=ax.transAxes, ha="center", va="top", fontsize=6,
                color=figures.COLOR_GRAY,
            )
            figures.style_axes(ax)
            paths.append(str(figures.render_figure(figure, directory / filename)))
        finally:
            figures.plt.close(figure)
    return paths
