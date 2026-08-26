r"""Which frequencies does the forecast actually get right?

The forecast analysis reports one number for the scattering block and one for the phase-harmonic
block. That split is a property of *how the features were computed*, not of what they describe: a
$0.5$ Hz scattering channel and a $0.5$ Hz phase-harmonic channel are both beat-to-beat structure,
and averaging each into its own block hides the far more clinically meaningful question of whether
the model predicts slow baseline drift better than beat-to-beat variability.

This analysis answers that, over two partitions of the same $c_y$ channels:

**Clinical.** The fetal-monitoring bands -- ``slow_baseline``, ``deceleration``, ``variability``,
``beat_to_beat`` -- plus ``unknown`` for the scattering channels whose centre frequency the shard's
provenance does not determine. The bands mean the same thing they mean in the predecessor tree.

**By kind.** The coefficient kinds: the order-0 lowpass, the order-1 scattering channels, and the
phase-harmonic channels grouped by their harmonic step $k$. This is the axis that says whether the
phase-harmonic block is earning its width, which the clinical partition cannot show because it
mixes both blocks into every band.

Below the bands, a **per-channel** pass: one MSE per channel, and one per (channel, horizon), so a
single badly-forecast channel is visible rather than diluted into the twenty around it.

**All three share one inference loop.** The band metrics and the channel accumulator are computed
from the same :meth:`~teb_vae.lag_attn.eval.runner.EvalRunner.forecast_view` inside a single pass
over the loader, because a second pass would double the most expensive part of the analysis to
recompute tensors that were already in hand. The channel accumulator is bounded by
$O(c_y \cdot H_d)$ and the band profiles by $O(\mathrm{bands} \cdot A)$, so neither grows with the
split.

**The partition comes from the run directory, not from a rebuild.** ``run.py`` emits
``band_partition.json`` from the shards' own ``sel_*`` provenance before any analysis starts. This
reads that file. A shard that predates ``_write_selection_attrs`` produces no partition, and then
this analysis records a **skip** rather than failing: the frequency resolution is unavailable, but
every other number in the run is unaffected, and there is nothing to fix in the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import band_partition, figures, metrics, report
from teb_vae.lag_attn.eval.band_partition import BandPartition
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "frequency_band"

#: Subdirectory holding the per-channel outputs, beside the per-partition ones.
CHANNEL_DIRNAME = "per_channel"

#: The partitions run, in output order. Both come from the same channel map; see the module
#: docstring for why one of them is not enough.
PARTITION_NAMES: Tuple[str, ...] = ("clinical", "by_kind")


class _ProfileAccumulator:
    r"""Running mean over samples of a per-sample profile, keyed by name.

    Each band contributes a $(B, N)$ profile per batch -- $N$ being the horizon steps or the
    anchors -- and what the CSVs and figures want is its mean over the split. Accumulating the
    running sum and the finite count is $O(N)$ per key regardless of how many samples pass
    through, where retaining the profiles themselves would be $O(\mathrm{samples} \cdot N)$: at
    the production geometry that is roughly $23$ MB per partition per $2000$ samples, and it grows
    without bound on an uncapped run over eight shards.

    ``NaN`` entries are *skipped*, not summed as zero. A warm-up anchor is ``NaN`` in every
    sample's profile by construction, and folding it in as zero would draw the warm-up prefix as a
    perfectly forecast region rather than as the gap it is.
    """

    def __init__(self) -> None:
        self._totals: Dict[str, np.ndarray] = {}
        self._counts: Dict[str, np.ndarray] = {}

    def add(self, key: str, values: Any) -> None:
        """Fold one batch's $(B, N)$ profile in under ``key``.

        Args:
            key: Accumulator name, typically ``f"{partition}/{label}"``.
            values: The profile, $(B, N)$, tensor or array.

        Raises:
            ValueError: If a later batch's profile length disagrees with the first's, which would
                otherwise broadcast one position's error onto another.
        """
        array = np.asarray(
            values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values,
            dtype=np.float64,
        )
        if array.ndim == 1:
            array = array[None, :]
        finite = np.isfinite(array)
        total = np.where(finite, array, 0.0).sum(axis=0)
        count = finite.sum(axis=0).astype(np.float64)

        if key not in self._totals:
            self._totals[key], self._counts[key] = total, count
            return
        if self._totals[key].shape != total.shape:
            raise ValueError(
                f"profile {key!r} arrived with length {total.shape[0]} after "
                f"{self._totals[key].shape[0]}; every batch must span the same positions."
            )
        self._totals[key] += total
        self._counts[key] += count

    def mean(self, key: str) -> np.ndarray:
        """Return the mean profile under ``key``, ``NaN`` at positions nothing contributed to."""
        if key not in self._totals:
            return np.zeros((0,))
        with np.errstate(invalid="ignore", divide="ignore"):
            counts = self._counts[key]
            return np.where(counts > 0.0, self._totals[key] / np.where(counts > 0.0, counts, 1.0),
                            np.nan)

    def n_contributing(self, key: str) -> np.ndarray:
        """Return the per-position count of samples that contributed a finite value."""
        return self._counts.get(key, np.zeros((0,)))


# ---------------------------------------------------------------------------
# Label presentation
# ---------------------------------------------------------------------------
def _label_frequencies(partition: BandPartition, channels: Sequence[int]) -> Tuple[float, float]:
    """Return the observed $(\\min, \\max)$ primary frequency over a label's channels.

    Args:
        partition: The channel map.
        channels: The label's channel indices.

    Returns:
        The bounds in Hz, or ``(NaN, NaN)`` when no channel carries a frequency.
    """
    values = np.asarray(
        [partition.channels[index].freq_hz_primary for index in channels], dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.min()), float(finite.max())


def hz_label(partition: BandPartition, name: str, label: str, channels: Sequence[int]) -> str:
    """Render a partition label with the frequency range it covers and its channel count.

    Explicit Hz on every axis tick, rather than a bare band name a reader has to look up. For a
    clinical band the *defining* range is used, since that is what the label means; for a
    harmonic kind, which is not a frequency range, the range its channels actually occupy is
    shown instead -- which is the honest answer to "what frequencies is this?".

    Args:
        partition: The channel map.
        name: Partition name, ``'clinical'`` or ``'by_kind'``.
        label: The label within it.
        channels: The label's channel indices.

    Returns:
        A single-line axis label.
    """
    count = len(channels)
    if name == "clinical" and label in partition.band_hz_ranges:
        low, high = partition.band_hz_ranges[label]
        span = f"{low:g}-{high:g} Hz" if np.isfinite(high) else f">{low:g} Hz"
        return f"{label} ({span}, {count} ch)"

    low, high = _label_frequencies(partition, channels)
    if not np.isfinite(low):
        return f"{label} (no centre frequency, {count} ch)"
    return f"{label} ({low:.3g}-{high:.3g} Hz, {count} ch)"


def ordered_labels(partition: BandPartition, name: str) -> List[str]:
    """Return a partition's labels ordered from the highest frequency to the lowest.

    One rule for both partitions: sort by the highest centre frequency any of the label's channels
    carries, descending. A label whose channels carry no frequency at all -- ``unknown``, and the
    order-0 lowpass -- sorts last rather than being treated as $0$ Hz, since "we could not tell"
    is not the same statement as "it is slow".

    Args:
        partition: The channel map.
        name: Partition name.

    Returns:
        The non-empty labels, high frequency first.
    """
    groups = partition.partition(name)
    ranked = []
    for label, channels in groups.items():
        if not channels:
            continue
        _, high = _label_frequencies(partition, channels)
        ranked.append((0 if np.isfinite(high) else 1, -(high if np.isfinite(high) else 0.0), label))
    return [label for _, _, label in sorted(ranked)]


# ---------------------------------------------------------------------------
# The single inference pass
# ---------------------------------------------------------------------------
def _make_per_batch(
    partitions: Mapping[str, Dict[str, List[int]]],
    profiles: _ProfileAccumulator,
    channel_accumulator: metrics.ChannelErrorAccumulator,
):
    """Build the per-batch callable that feeds every output from one forward.

    Args:
        partitions: Partition name to its label-to-channel-index mapping.
        profiles: Accumulator receiving each band's horizon and anchor profiles.
        channel_accumulator: Accumulator receiving the per-channel squared error.

    Returns:
        A ``(runner, batch) -> {column: per-sample value}`` callable, suitable for
        :func:`~teb_vae.lag_attn.eval.collectors.collect_metrics`.
    """

    def _per_batch(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
        view = runner.forecast_view(batch)
        channel_accumulator.update(view.mu_full, view.y_plus, view.mask)

        columns: Dict[str, Any] = {}
        for name, groups in partitions.items():
            band_metrics = metrics.band_forecast_metrics(
                view.mu_full, view.y_plus, view.mask, groups
            )
            for label, record in band_metrics.items():
                # Per sample in the CSV -- the distribution the violins draw -- while the two
                # profiles go to the accumulator, because one column per (band, anchor) would be
                # some fifteen hundred columns on a frame meant to have one row per recording.
                columns[f"{name}__{label}__mse"] = record["feat_mse"]
                columns[f"{name}__{label}__r2"] = record["feat_r2"]
                profiles.add(f"{name}/{label}/horizon", record["horizon"])
                profiles.add(f"{name}/{label}/anchor", record["anchor"])
        return columns

    return _per_batch


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def _write_partition_csvs(
    frame: pd.DataFrame,
    partition: BandPartition,
    name: str,
    labels: Sequence[str],
    groups: Mapping[str, Sequence[int]],
    profiles: _ProfileAccumulator,
    directory: Path,
) -> Dict[str, str]:
    """Write one partition's three CSVs.

    Args:
        frame: The collected per-sample frame, carrying every partition's columns.
        partition: The channel map.
        name: Partition name.
        labels: Its labels, high frequency first.
        groups: Label to channel indices.
        profiles: The profile accumulator.
        directory: The partition's output directory.

    Returns:
        CSV name to the path written.
    """
    directory.mkdir(parents=True, exist_ok=True)

    identity = [column for column in ("sample_index", "guid", "source_file") if column in frame]
    columns = identity + [
        f"{name}__{label}__{metric}" for label in labels for metric in ("mse", "r2")
    ]
    per_sample = frame[[column for column in columns if column in frame]].copy()
    # Renamed on the way out: the partition name is already the directory, so repeating it in
    # every column header makes a table that has to be read sideways.
    per_sample = per_sample.rename(
        columns={
            f"{name}__{label}__{metric}": f"{metric}_{label}"
            for label in labels
            for metric in ("mse", "r2")
        }
    )
    per_sample_path = directory / "per_sample.csv"
    per_sample.to_csv(per_sample_path, index=False)

    written = {"per_sample": str(per_sample_path)}
    for kind, filename in (("horizon", "horizon.csv"), ("anchor", "anchor.csv")):
        rows: List[Dict[str, Any]] = []
        for label in labels:
            profile = profiles.mean(f"{name}/{label}/{kind}")
            counts = profiles.n_contributing(f"{name}/{label}/{kind}")
            for position in range(int(profile.shape[0])):
                rows.append({
                    "band": label,
                    "band_label": hz_label(partition, name, label, groups[label]),
                    "n_channels": len(groups[label]),
                    "position": position,
                    "mse": float(profile[position]),
                    "n_samples": int(counts[position]) if position < counts.shape[0] else 0,
                })
        path = directory / filename
        pd.DataFrame(rows).to_csv(path, index=False)
        written[kind] = str(path)
    return written


def _write_channel_csvs(
    accumulator: metrics.ChannelErrorAccumulator, partition: BandPartition, directory: Path
) -> Dict[str, str]:
    """Write the per-channel and per-(channel, horizon) tables, joined to the channel map.

    Joined rather than emitted bare: a per-channel MSE indexed only by channel number cannot be
    replotted against frequency without also loading ``band_channel_map.csv`` and reproducing the
    join, and the whole reason the channel map exists is so a downstream plot does not have to.

    Args:
        accumulator: The filled channel accumulator.
        partition: The channel map.
        directory: The ``per_channel`` output directory.

    Returns:
        CSV name to the path written.
    """
    directory.mkdir(parents=True, exist_ok=True)

    per_channel = accumulator.per_channel_mse()
    rows = []
    for record in partition.channels:
        row = dict(record.as_row())
        row["mse"] = float(per_channel[record.channel])
        rows.append(row)
    channel_path = directory / "per_channel.csv"
    pd.DataFrame(rows).to_csv(channel_path, index=False)

    field = accumulator.per_channel_horizon_mse()
    horizon_rows = []
    for record in partition.channels:
        for step in range(int(field.shape[1])):
            horizon_rows.append({
                "channel": record.channel,
                "block": record.block,
                "kind": record.kind,
                "band": record.band,
                "freq_hz_primary": record.freq_hz_primary,
                "horizon": step,
                "mse": float(field[record.channel, step]),
            })
    horizon_path = directory / "per_channel_horizon.csv"
    pd.DataFrame(horizon_rows).to_csv(horizon_path, index=False)

    return {"per_channel": str(channel_path), "per_channel_horizon": str(horizon_path)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _write_partition_figures(
    frame: pd.DataFrame,
    partition: BandPartition,
    name: str,
    labels: Sequence[str],
    groups: Mapping[str, Sequence[int]],
    profiles: _ProfileAccumulator,
    directory: Path,
) -> Dict[str, str]:
    """Emit one partition's violin and band-by-horizon figures.

    Args:
        frame: The collected per-sample frame.
        partition: The channel map.
        name: Partition name.
        labels: Its labels, high frequency first.
        groups: Label to channel indices.
        profiles: The profile accumulator.
        directory: The partition's output directory.

    Returns:
        Figure name to the path written.
    """
    # Owned here rather than inherited from whichever writer happened to run first, so either
    # can be called on its own.
    directory.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    rendered = [hz_label(partition, name, label, groups[label]) for label in labels]

    figure, axes = figures.new_figure(2, height_per_row=3.2)
    try:
        for row, metric in enumerate(("mse", "r2")):
            samples = {
                rendered[index]: frame.get(f"{name}__{label}__{metric}", [])
                for index, label in enumerate(labels)
            }
            figures.violin_panel(
                axes[row, 0], samples,
                title=(
                    f"Per-sample masked MSE by {name} band" if metric == "mse"
                    else f"Per-sample $R^2$ by {name} band"
                ),
                ylabel="Masked MSE" if metric == "mse" else "$R^2$",
                reference=None if metric == "mse" else 0.0,
                reference_label="$R^2 = 0$ (predicting the channel mean)",
            )
        written["band_violins"] = str(
            figures.render_figure(figure, directory / "band_violins")
        )
    finally:
        figures.plt.close(figure)

    horizon = np.stack(
        [profiles.mean(f"{name}/{label}/horizon") for label in labels]
    ) if labels else np.zeros((0, 0))
    anchor = np.stack(
        [profiles.mean(f"{name}/{label}/anchor") for label in labels]
    ) if labels else np.zeros((0, 0))

    figure, axes = figures.new_figure(3, height_per_row=2.8)
    try:
        figures.multi_line_panel(
            axes[0, 0], figures.sequence_axis(horizon.shape[1] if horizon.size else 0),
            horizon, rendered,
            title=f"Forecast error by horizon step, per {name} band",
            xlabel="Horizon step $h$", ylabel="Mean masked MSE",
        )
        figures.heatmap_with_colorbar(
            figure, axes[1, 0], horizon,
            title=f"Band by horizon ({name})", xlabel="Horizon step $h$", ylabel="",
            symmetric=False, colorbar_label="Mean masked MSE",
        )
        figures.label_rows(axes[1, 0], rendered)
        figures.heatmap_with_colorbar(
            figure, axes[2, 0], anchor,
            title=f"Band by anchor ({name})",
            xlabel="Anchor $t$ (decimated steps)", ylabel="",
            symmetric=False, colorbar_label="Mean masked MSE",
        )
        figures.label_rows(axes[2, 0], rendered)
        written["band_horizon"] = str(
            figures.render_figure(figure, directory / "band_horizon")
        )
    finally:
        figures.plt.close(figure)

    return written


def _write_channel_figure(
    accumulator: metrics.ChannelErrorAccumulator, partition: BandPartition, directory: Path
) -> Dict[str, str]:
    r"""Emit the per-channel MSE-against-frequency scatter, split by feature block.

    Two panels, not one. A scattering channel is described by a single centre frequency; a phase
    harmonic channel is described by a *pair* $(\xi_i, \xi_j)$ and its ratio $p = \xi_j / \xi_i$,
    and plotting it at $\xi_j$ alone throws away the half of its identity that distinguishes it
    from the scattering channel at the same frequency. The phase panel therefore colours by $p$,
    so the dual-frequency identity survives into the figure.

    Args:
        accumulator: The filled channel accumulator.
        partition: The channel map.
        directory: The ``per_channel`` output directory.

    Returns:
        Figure name to the path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    per_channel = accumulator.per_channel_mse()
    figure, axes = figures.new_figure(2, height_per_row=3.0)
    try:
        for row, block in enumerate(("scattering", "phase")):
            records = [record for record in partition.channels if record.block == block]
            frequencies = np.asarray(
                [record.freq_hz_primary for record in records], dtype=np.float64
            )
            errors = np.asarray(
                [per_channel[record.channel] for record in records], dtype=np.float64
            )
            ratios = np.asarray(
                [record.harmonic_ratio for record in records], dtype=np.float64
            )
            figures.frequency_scatter(
                figure, axes[row, 0], frequencies, errors,
                colour_by=ratios if block == "phase" else None,
                colour_label="harmonic ratio $p = \\xi_j / \\xi_i$",
                title=(
                    "Per-channel MSE against centre frequency (scattering)" if block == "scattering"
                    else "Per-channel MSE against $\\xi_j$ (phase-harmonic)"
                ),
                xlabel="Centre frequency (Hz)" if block == "scattering" else "$\\xi_j$ (Hz)",
                ylabel="Pooled masked MSE",
            )
        return {
            "per_channel_frequency": str(
                figures.render_figure(figure, directory / "per_channel_frequency")
            )
        }
    finally:
        figures.plt.close(figure)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_run_partition(output_dir: Any) -> Optional[BandPartition]:
    """Reload the partition ``run.py`` emitted into this run directory, or ``None``.

    Args:
        output_dir: The run's results directory.

    Returns:
        The partition, or ``None`` when the run produced none -- which happens exactly when no
        configured shard carried ``sel_*`` provenance.
    """
    path = Path(output_dir) / band_partition.PARTITION_FILENAME
    if not path.is_file():
        return None
    return band_partition.load_partition(path)


def run_frequency_band_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the forecast by frequency band and by channel, in one pass over the loader.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, supplying the sample count and per-file grouping a
            capped draw stratifies over.

    Returns:
        The headline summary for ``summary.json``, or a ``skipped`` record when the run carried
        no channel provenance to resolve frequencies with.

    Raises:
        RuntimeError: If the partition describes a different number of channels from the one the
            checkpoint forecasts. Every band would then be built from the wrong channels, and the
            resulting numbers would look entirely ordinary.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    partition = load_run_partition(output_dir)
    if partition is None:
        # A skip, not a failure: the shards are an older vintage, there is nothing to fix in the
        # pipeline, and every other analysis in the run is unaffected.
        logger.warning(
            f"frequency_band: skipped -- no {band_partition.PARTITION_FILENAME} in the run "
            f"directory, so no shard carried the sel_* channel provenance a frequency-resolved "
            f"analysis needs."
        )
        return {
            "skipped": True,
            "reason": (
                f"no {band_partition.PARTITION_FILENAME} was written for this run; the "
                f"configured shards carry no sel_* channel provenance"
            ),
        }

    if int(partition.n_channels) != int(runner.model.c_y):
        raise RuntimeError(
            f"the band partition describes {partition.n_channels} channels but the checkpoint "
            f"forecasts c_y={int(runner.model.c_y)}. Every band would be assembled from the "
            f"wrong channels and the numbers would look ordinary. The partition is built from "
            f"dataset_config.vae_test_datasets -- point it at the shards this checkpoint was "
            f"trained on."
        )

    directory.mkdir(parents=True, exist_ok=True)
    partitions = {name: partition.partition(name) for name in PARTITION_NAMES}
    label_order = {name: ordered_labels(partition, name) for name in PARTITION_NAMES}

    profiles = _ProfileAccumulator()
    channel_accumulator = metrics.ChannelErrorAccumulator(
        n_channels=int(partition.n_channels), horizon=int(runner.model.horizon)
    )

    caps = eval_config.get("caps") or {}
    seed = int(eval_config.get("seed", 0))
    n_total = int((probe or {}).get("n_samples") or 0)
    plan = (
        CollectionPlan.build(n_total, caps.get("frequency_band"), seed,
                             groups=(probe or {}).get("source_files"))
        if n_total else None
    )

    collected = collect_metrics(
        runner, loader,
        _make_per_batch(partitions, profiles, channel_accumulator),
        max_samples=eval_config.get("max_samples"),
        plan=plan,
        progress_label="frequency_band",
    )
    frame = collected.frame

    written: Dict[str, Any] = {}
    summary_bands: Dict[str, Any] = {}
    for name in PARTITION_NAMES:
        labels = label_order[name]
        partition_dir = directory / name
        written[name] = _write_partition_csvs(
            frame, partition, name, labels, partitions[name], profiles, partition_dir
        )
        written[name].update(
            _write_partition_figures(
                frame, partition, name, labels, partitions[name], profiles, partition_dir
            )
        )
        # Only the band MSEs, and only for this partition's own directory: "is the deceleration
        # band worse for the acidosis cohort?" is the question the two axes together answer, and
        # it is the one a pooled band number and a pooled cohort number each half-answer.
        written[name]["by_group"] = report.emit_grouped_variants(
            frame, partition_dir,
            value_columns=[f"{name}__{label}__mse" for label in labels],
            stem="band_mse",
        )
        summary_bands[name] = {
            label: {
                "n_channels": len(partitions[name][label]),
                "label": hz_label(partition, name, label, partitions[name][label]),
                "mean_mse": _finite_mean(frame.get(f"{name}__{label}__mse")),
                "mean_r2": _finite_mean(frame.get(f"{name}__{label}__r2")),
            }
            for label in labels
        }

    channel_dir = directory / CHANNEL_DIRNAME
    written[CHANNEL_DIRNAME] = _write_channel_csvs(channel_accumulator, partition, channel_dir)
    written[CHANNEL_DIRNAME].update(
        _write_channel_figure(channel_accumulator, partition, channel_dir)
    )

    per_channel = channel_accumulator.per_channel_mse()
    finite = per_channel[np.isfinite(per_channel)]
    worst = int(np.nanargmax(per_channel)) if finite.size else None

    summary: Dict[str, Any] = {
        "skipped": False,
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "n_channels": int(partition.n_channels),
        "partitions": summary_bands,
        "pooled_feat_mse": channel_accumulator.total_mse(),
        "worst_channel": None if worst is None else {
            "channel": worst,
            "mse": float(per_channel[worst]),
            "band": partition.channels[worst].band,
            "kind": partition.channels[worst].kind,
            "freq_hz_primary": partition.channels[worst].freq_hz_primary,
        },
        "artifacts": written,
    }

    logger.info(
        f"frequency_band: {len(frame)} sample(s) over {partition.n_channels} channel(s); "
        f"pooled feat_mse={summary['pooled_feat_mse']:.6g}; worst channel "
        f"{summary['worst_channel']}"
    )
    return summary


def _finite_mean(values: Any) -> float:
    """Return the mean of the finite entries, or ``NaN`` when there are none.

    Args:
        values: A column, possibly absent.

    Returns:
        The mean as a float.
    """
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")
