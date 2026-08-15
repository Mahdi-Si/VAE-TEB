r"""What each of the model's input channels is, read off the shards rather than re-derived.

The model consumes two streams, and every later frequency-resolved statement about either of them
needs to know which channel is which: that a given ``fhr_ph`` coefficient pairs filters at $0.021$
and $0.05$ Hz and therefore sits in the deceleration band, or that seven of the thirty-six target
scattering channels have no recoverable centre frequency at all.

The shards already carry the answer. ``create_new_pipeline.py::_write_selection_attrs`` stamps
per-channel provenance -- ``sel_i``, ``sel_j``, ``sel_xi_i_hz``, ``sel_xi_j_hz``, ``sel_power``,
``sel_band_hz``, ``sel_k_steps`` -- onto the ``fhr_ph`` and ``up_ph`` datasets, ordered to match
the stored channel axis. The arithmetic that turns those attributes into a band map is the shared
one and is imported rather than restated: the harmonic grid, the *descending* filter bank, the
clinical band edges and the frequency map recovered from the selected pairs are properties of the
dataset pipeline, not of either model, and two copies of them would be two chances for a band
label to mean different things in two trees.

What is this model's, and therefore what is written here, is the **shape of its input**: two
streams rather than one target vector.

* the target stream, $c_y = 36\ (\texttt{fhr\_st}) + 66\ (\texttt{fhr\_ph}) = 102$;
* the source stream, $c_u = 36\ (\texttt{up\_st}) + 15\ (\texttt{up\_ph}) = 51$, or $15$ alone
  when ``use_up_st`` is false.

Both blocks are laid out exactly as the model receives them -- ``cat([st, ph])`` -- so a channel
index in this map is the channel index a later analysis will hold, and the two streams are
distinguished by a ``stream`` column rather than by a reader remembering an offset.

**Three columns exist here and in no other artifact, and they are the causal dataset's own.** A
one-sided filter reads only the past, so before its warm-up has passed its output is a function of
the assumed pre-recording history rather than of the recording; and its output is stale by a
composed group delay. Both are stamped per channel on every stored block, and both are read here:

* ``causal_warmup_steps`` -- $W'_c$, **rebased** into the coordinates of the window the loader
  serves. The stored vector counts from stored step $0$ while the loader starts reading at
  ``trim_steps``, and the un-rebased number would misplace every channel's validity boundary by
  exactly the trim.
* ``causal_delay_s`` -- the composed one-sided group delay, which is what makes the lag axis
  stored-coefficient time rather than physical time.
* ``kept`` -- whether the resolved warm-up budget retained the channel. The budget gates the
  **target** stream only; the source stream's keep-index is the identity by construction, for the
  reason ``causal_warmup.resolve_warmup_budget`` records: the source channels the budget would drop
  are the ones carrying the contraction envelope, against a lag search that exists to find the
  contraction-to-deceleration delay.

**A second map is emitted on the kept channel axis**, beside the declared-axis one. The per-channel
readouts the collection pass writes are positional against the $C_{\mathrm{keep}}$ *surviving*
target channels while everything above is over the $c_y$ *declared* ones, so joining the two
positionally would shift band membership across the axis -- and on the shipped dataset the dropped
channels happen to be the trailing four, which makes a positional join look right here and be wrong
on any dataset whose survivors are not a prefix. A join that is accidentally correct is worse than
one that is wrong, because no test catches it.

The kept axis is read from the collection record's ``target_keep_index`` rather than from the
model's own channel gate: this layer holds no model on the path that matters, an offline re-run
against a finished directory with no checkpoint and no GPU.

**Absent provenance is a recorded skip, not a failure.** Whether a production shard predates
``_write_selection_attrs`` is a property of that file. A run whose readouts all succeeded must not
be marked failed because its shards are an older vintage. A *block* whose provenance is missing is
narrower still and is the common case here rather than the exception: on this dataset the ``sel_*``
attributes are on the two phase blocks only, so the scattering channels are banded through the
filter-index-to-Hz map those attributes carry, and the ones no selected pair named are recorded as
``unknown`` -- never bucketed into a neighbour, whose skill they do not share.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from teb_vae.lag_attn_cfs.eval._reuse import band_partition as shared

#: Files written into the results directory. The JSON round-trips into an equivalent partition for
#: anything reasoning about the groups; the declared-axis CSV is one row per input channel so a plot
#: can be redrawn from disk with ``pandas`` and no import from this package at all; the kept-axis CSV
#: is what the band-resolved skill readout joins through.
PARTITION_FILENAME = "band_partition.json"
CHANNEL_MAP_FILENAME = "band_channel_map.csv"
KEPT_CHANNEL_MAP_FILENAME = "band_channel_map_kept.csv"

#: ``stream -> (scattering dataset, phase dataset)``, in the order the model concatenates them.
#: The stream names are the model's own vocabulary -- the target is what is forecast, the source
#: is what the posterior additionally conditions on -- rather than the signal names, because a
#: reader of this map is asking which of the two model inputs a channel belongs to.
STREAM_DATASETS: Dict[str, Tuple[str, str]] = {
    "target": ("fhr_st", "fhr_ph"),
    "source": ("up_st", "up_ph"),
}

#: The stream the warm-up budget gates. The other one's keep-index is the identity by construction,
#: which is a property of the resolver rather than of a configuration, so it is written out here
#: rather than derived from a keep-index this layer does not have.
GATED_STREAM = "target"

#: The per-channel causal attributes every stored block of a causal shard carries, and the column
#: each becomes. Read from the datasets' own attributes: a config literal would describe a boundary
#: the data may no longer have.
CAUSAL_ATTRIBUTES: Tuple[str, ...] = ("causal_warmup_steps", "causal_delay_s")


def _vae_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the model block of a merged run config, or an empty mapping."""
    return dict((config.get("model_config") or {}).get("VAE_model") or {})


def trim_steps_of(config: Dict[str, Any]) -> int:
    """Return the decimated steps the loader's trim discards from each end.

    The conversion is the dataset package's own rather than a second copy of it: a consumer that
    rebases a warm-up against a trimmed window must use the loader's trim *exactly*, and a copy
    that rounded differently would move every channel's validity boundary without moving anything
    that reports it.

    Args:
        config: The merged run config.

    Returns:
        Decimated steps discarded per end. Zero when the config declares no trim, which is what
        an untrimmed window means rather than a missing setting.
    """
    from hdf5_dataset.hdf5_dataset import decimated_trim_steps

    dataloader = (config.get("dataset_config") or {}).get("dataloader_config") or {}
    kwargs = dataloader.get("dataset_kwargs") or {}
    trim = kwargs.get("trim_minutes")
    _, steps = decimated_trim_steps(None if trim is None else float(trim))
    return int(steps)


def stored_width(shard_path: Any, dataset: str) -> Optional[int]:
    """Return a dataset's stored channel count, or ``None`` when the shard has no such dataset.

    Args:
        shard_path: Path to the shard.
        dataset: Dataset name.

    Returns:
        The middle axis of a ``(n_samples, n_channels, n_steps)`` dataset, or ``None``.
    """
    import h5py

    with h5py.File(str(shard_path), "r") as handle:
        if dataset not in handle:
            return None
        shape = handle[dataset].shape
    return int(shape[1]) if len(shape) >= 2 else None


def causal_columns(
    shard_path: Any, datasets: Sequence[str], *, trim_steps: int
) -> Dict[str, List[Optional[float]]]:
    r"""Read one stream's per-channel causal attributes, concatenated in model order.

    Args:
        shard_path: Path to the shard.
        datasets: The stored blocks making up the stream, in concatenation order. A block the
            stream does not carry -- ``up_st`` on a model built without it -- is passed as absent
            by the caller rather than filtered here.
        trim_steps: Decimated steps the loader discards from each end, for the rebase
            $W' = \max(W - \mathrm{trim},\ 0)$.

    Returns:
        ``{column: [value or None per declared channel]}``. A block whose attributes are missing
        contributes ``None`` for each of its channels rather than shortening the vector, so the
        columns stay positional against the declared channel axis whatever a shard omits.
    """
    import h5py

    columns: Dict[str, List[Optional[float]]] = {name: [] for name in CAUSAL_ATTRIBUTES}
    with h5py.File(str(shard_path), "r") as handle:
        for dataset in datasets:
            width = 0 if dataset not in handle else int(handle[dataset].shape[1])
            for name in CAUSAL_ATTRIBUTES:
                attrs = handle[dataset].attrs if dataset in handle else {}
                if name not in attrs:
                    columns[name].extend([None] * width)
                    continue
                values = np.asarray(attrs[name], dtype=np.float64).reshape(-1)
                if name == "causal_warmup_steps":
                    values = np.maximum(values - float(trim_steps), 0.0)
                columns[name].extend(float(value) for value in values[:width])
                # A block whose attribute is shorter than its channel axis describes a different
                # file; pad rather than let the columns slip out of register with the map.
                columns[name].extend([None] * max(width - int(values.size), 0))
    return columns


def build_stream_partitions(
    shard_path: Any, *, use_up_st: bool = True
) -> Dict[str, shared.BandPartition]:
    r"""Build one channel map per input stream from a single shard.

    Args:
        shard_path: Path to any shard of the split under evaluation. The selection is a property
            of the pipeline revision rather than of the recording, so one shard describes them
            all -- and a mismatch between two would already have failed preflight's width guard.
        use_up_st: Whether the model's source stream carries the ``up_st`` scattering block. When
            it does not, $c_u$ is the phase block alone and the scattering side of that stream is
            genuinely zero channels wide rather than merely unread.

    Returns:
        Stream name to its partition, ordered as :data:`STREAM_DATASETS`.

    Raises:
        RuntimeError: If a stream's phase provenance is missing or inconsistent. See
            :func:`~teb_vae.lag_attn.eval.band_partition.read_selection`.
    """
    partitions: Dict[str, shared.BandPartition] = {}
    for stream, (scattering_dataset, phase_dataset) in STREAM_DATASETS.items():
        width = stored_width(shard_path, scattering_dataset) or 0
        if stream == "source" and not use_up_st:
            # Not a missing dataset: the model is built without that block, so those channels are
            # not part of its input and must not appear in a map of what it consumes.
            width = 0
        partitions[stream] = shared.build_partition(
            shard_path, n_scattering=width, phase_dataset=phase_dataset
        )
    return partitions


def stream_datasets_for(stream: str, *, use_up_st: bool) -> Tuple[str, ...]:
    """Return the stored blocks a stream is assembled from, in concatenation order.

    Args:
        stream: ``'target'`` or ``'source'``.
        use_up_st: Whether the source stream carries its scattering block.

    Returns:
        The block names. The source stream drops ``up_st`` when the model is built without it, so
        the causal columns stay positional against the same declared axis the partition uses.
    """
    scattering, phase = STREAM_DATASETS[stream]
    if stream == "source" and not use_up_st:
        return (phase,)
    return (scattering, phase)


def channel_rows(
    partitions: Dict[str, shared.BandPartition],
    *,
    causal: Optional[Dict[str, Dict[str, List[Optional[float]]]]] = None,
    keep_index: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Flatten the per-stream partitions into one row per input channel.

    Args:
        partitions: Stream name to partition.
        causal: Stream name to its per-channel causal columns, positional against that stream's
            declared channel axis. A stream absent from the mapping gets ``NaN`` columns, which is
            what "the shard does not say" looks like in a CSV.
        keep_index: The declared indices of the gated stream's surviving channels, or ``None``
            when the checkpoint carries no gate -- which is the ungated model and means every
            declared channel survives, not that survival is unknown.

    Returns:
        One row per channel, each carrying its stream beside the shared channel record and the
        three causal columns. The channel index is the index *within its stream*, which is the
        index the model's own tensors use.
    """
    kept_set = None if keep_index is None else {int(value) for value in keep_index}
    rows: List[Dict[str, Any]] = []
    for stream, partition in partitions.items():
        columns = (causal or {}).get(stream, {})
        for position, record in enumerate(partition.channels):
            row: Dict[str, Any] = {"stream": stream, **record.as_row()}
            for name in CAUSAL_ATTRIBUTES:
                values = columns.get(name) or []
                value = values[position] if position < len(values) else None
                row[name] = float("nan") if value is None else float(value)
            # The source stream is never gated: its keep-index is the identity by construction.
            row["kept"] = (
                True
                if stream != GATED_STREAM or kept_set is None
                else int(record.channel) in kept_set
            )
            rows.append(row)
    return rows


def kept_channel_rows(
    rows: Sequence[Dict[str, Any]], keep_index: Optional[Sequence[int]]
) -> List[Dict[str, Any]]:
    """Project the gated stream's rows onto the kept channel axis, in gather order.

    Args:
        rows: The declared-axis rows.
        keep_index: The surviving declared indices, strictly ascending -- the order the model's
            own gather uses -- or ``None`` for the ungated model, whose kept axis is the declared
            one.

    Returns:
        One row per kept channel, carrying ``kept_channel`` (the position on the axis every
        per-channel readout is indexed on) beside the declared ``channel`` it came from, so the
        two axes can be reconciled by a reader rather than trusted.

    Raises:
        KeyError: If ``keep_index`` names a declared channel the map does not have. That is a
            keep-index and a shard describing different channel axes, and silently emitting a
            shorter map would give every band-resolved statement a silently wrong denominator.
    """
    declared = {
        int(row["channel"]): row for row in rows if row.get("stream") == GATED_STREAM
    }
    order = sorted(declared) if keep_index is None else [int(value) for value in keep_index]
    projected: List[Dict[str, Any]] = []
    for position, channel in enumerate(order):
        if channel not in declared:
            raise KeyError(
                f"the collection record's target_keep_index names declared channel {channel}, "
                f"which the shard's channel map does not have (it describes "
                f"{len(declared)} target channels). The keep-index and the shard describe "
                f"different channel axes, so no band-resolved statement over them is meaningful."
            )
        projected.append({"kept_channel": position, **declared[channel]})
    return projected


def coverage_counts(
    rows: Sequence[Dict[str, Any]], kept_rows: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    r"""Count what the band map does and does not cover, on both axes.

    Five counts rather than one ratio, because the declared and scored numerators can coincide by
    arithmetic accident -- on the shipped dataset both are $95$, since $102 - 7 = 98 - 3$ -- and
    quoting "95 of 102" would imply the analysis banded channels the decoder never emitted.

    Args:
        rows: The declared-axis rows.
        kept_rows: The kept-axis rows.

    Returns:
        The five counts, plus the band breakdown of the channels the budget dropped. The
        breakdown is what says *which* end of the frequency axis the budget removed, which is the
        same property that makes those channels' warm-up longest.
    """
    declared = [row for row in rows if row.get("stream") == GATED_STREAM]
    dropped = [row for row in declared if not bool(row.get("kept", True))]
    dropped_bands: Dict[str, int] = {}
    for row in dropped:
        band = str(row.get("band", shared.UNKNOWN_BAND))
        dropped_bands[band] = dropped_bands.get(band, 0) + 1
    unknown_kept = sum(
        1 for row in kept_rows if str(row.get("band")) == shared.UNKNOWN_BAND
    )
    return {
        "declared_total": len(declared),
        "dropped_declared": len(dropped),
        "kept_total": len(kept_rows),
        "known_kept": len(kept_rows) - unknown_kept,
        "unknown_kept": unknown_kept,
        "dropped_bands": dropped_bands,
    }


def declared_widths(config: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Return the stream widths the configuration declares, for the agreement check.

    Args:
        config: The merged run config.

    Returns:
        ``{'target': c_y, 'source': c_u}``, either value ``None`` when the config does not say.
    """
    vae = _vae_config(config)
    return {
        "target": None if vae.get("c_y") is None else int(vae["c_y"]),
        "source": None if vae.get("c_u") is None else int(vae["c_u"]),
    }


def write_partitions(
    partitions: Dict[str, shared.BandPartition],
    output_dir: Any,
    *,
    rows: Sequence[Dict[str, Any]],
    kept_rows: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    """Write the per-stream partition JSON and both channel-map CSVs.

    Args:
        partitions: Stream name to partition.
        output_dir: The results directory.
        rows: The declared-axis rows.
        kept_rows: The kept-axis rows.

    Returns:
        Artifact name to the path written, **relative to the results directory**. Relative rather
        than absolute because this record is part of ``results``, which is compared across runs
        and across sweep arms: an absolute path makes two identical findings differ, and it stops
        being true the moment the directory is copied anywhere.
    """
    import json

    import pandas as pd

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / PARTITION_FILENAME
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {stream: partition.as_dict() for stream, partition in partitions.items()},
            handle,
            indent=2,
        )

    csv_path = directory / CHANNEL_MAP_FILENAME
    pd.DataFrame(list(rows)).to_csv(csv_path, index=False)

    # Written even when every channel survives: the join that reads it must not have to branch on
    # whether the file exists, and "the axes coincide here" is itself a fact worth recording.
    kept_path = directory / KEPT_CHANNEL_MAP_FILENAME
    pd.DataFrame(list(kept_rows)).to_csv(kept_path, index=False)
    return {
        "partition": json_path.relative_to(directory).as_posix(),
        "channel_map": csv_path.relative_to(directory).as_posix(),
        "kept_channel_map": kept_path.relative_to(directory).as_posix(),
    }


def emit_partition(
    shard_paths: Sequence[Any],
    output_dir: Any,
    *,
    use_up_st: bool = True,
    declared: Optional[Dict[str, Optional[int]]] = None,
    trim_steps: int = 0,
    keep_index: Optional[Sequence[int]] = None,
    kept_width: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and write both channel maps, recording a clean skip when it cannot.

    Args:
        shard_paths: The configured test shards. The first readable one is used.
        output_dir: The results directory.
        use_up_st: Whether the source stream carries its scattering block.
        declared: The widths the configuration declares, compared against what the shard
            describes. A disagreement is **recorded, not raised**: preflight already refuses a
            width mismatch against the data the model is fed, and what this would add is a second
            refusal from a step that describes rather than decides.
        trim_steps: Decimated steps the loader discards per end, for the warm-up rebase.
        keep_index: The gated stream's surviving declared indices, from the collection record, or
            ``None`` for the ungated model.
        kept_width: The kept-axis width the collection record reports, cross-checked against the
            map this builds. A disagreement is recorded rather than raised for the same reason a
            width disagreement is, but it is the one that would make a band-resolved join wrong,
            so it is reported as its own field rather than folded into the width disagreements.

    Returns:
        A JSON-safe record: either the written paths, the per-stream counts, the coverage counts
        and the kept-axis record, or ``skipped`` with every attempt's reason.
    """
    reasons: List[str] = []
    for path in shard_paths or ():
        try:
            partitions = build_stream_partitions(path, use_up_st=use_up_st)
            causal = {
                stream: causal_columns(
                    path,
                    stream_datasets_for(stream, use_up_st=use_up_st),
                    trim_steps=trim_steps,
                )
                for stream in partitions
            }
            rows = channel_rows(partitions, causal=causal, keep_index=keep_index)
            kept_rows = kept_channel_rows(rows, keep_index)
        except Exception as exc:  # noqa: BLE001 - recorded and reported; see the module docstring
            reasons.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        written = write_partitions(
            partitions, output_dir, rows=rows, kept_rows=kept_rows
        )
        widths = {stream: partition.n_channels for stream, partition in partitions.items()}
        declared = dict(declared or {})
        disagreements: Dict[str, Dict[str, int]] = {}
        for stream, width in widths.items():
            expected = declared.get(stream)
            if expected is not None and int(expected) != int(width):
                disagreements[stream] = {
                    "declared": int(expected), "described_by_shard": int(width)
                }
        coverage = coverage_counts(rows, kept_rows)
        return {
            "skipped": False,
            "shard": str(path),
            "n_channels": int(sum(widths.values())),
            "trim_steps": int(trim_steps),
            "streams": {
                stream: {
                    "n_channels": partition.n_channels,
                    "n_scattering": partition.n_scattering,
                    "n_phase": partition.n_phase,
                    "kind_counts": partition.kind_counts(),
                    "band_counts": {
                        name: len(indices)
                        for name, indices in partition.partition("clinical").items()
                    },
                    # The channels the attributes could not place. Counted rather than dropped:
                    # a band-resolved statement about this model's inputs is incomplete by
                    # exactly this many channels, and that has to be visible.
                    "n_scattering_without_frequency": int(
                        partition.coverage.get("n_scattering_without_frequency", 0)
                    ),
                    "coverage": partition.coverage,
                }
                for stream, partition in partitions.items()
            },
            "declared_widths": dict(declared),
            "width_disagreements": disagreements,
            "kept_axis": {
                "gated_stream": GATED_STREAM,
                "from_keep_index": keep_index is not None,
                "n_channels": len(kept_rows),
                "declared_index": [int(row["channel"]) for row in kept_rows],
                "reported_kept_width": None if kept_width is None else int(kept_width),
                "width_agrees": (
                    None if kept_width is None else int(kept_width) == len(kept_rows)
                ),
                **coverage,
            },
            "files": written,
        }

    return {
        "skipped": True,
        "reason": (
            "no configured shard carried usable sel_* channel provenance"
            if reasons else "no test shards were configured"
        ),
        "attempts": reasons,
    }


def run_band_partition_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit both channel maps for this run.

    Args:
        context: The analysis context, read for the merged config's shard list and channel widths
            and for the collection record's kept target channel axis.
        eval_config: The validated ``eval_config`` block. Unused: this step describes the data
            rather than sampling it, so no cap applies to it.
        output_dir: The results directory.
        probe: The loader probe's record. Unused for the same reason.

    Returns:
        The protocol's four keys plus the partition record. ``n_samples`` is ``None`` rather than
        zero: this analysis scores no segments at all, and a zero would enter the coverage block's
        population comparison as a disagreement with every analysis that does.
    """
    config = dict(getattr(context, "config", None) or {})
    shards = list((config.get("dataset_config") or {}).get("vae_test_datasets") or [])
    geometry = dict(
        (getattr(getattr(context, "collection", None), "record", None) or {}).get("geometry")
        or {}
    )
    record = emit_partition(
        shards,
        output_dir,
        use_up_st=bool(_vae_config(config).get("use_up_st", True)),
        declared=declared_widths(config),
        trim_steps=trim_steps_of(config),
        keep_index=geometry.get("target_keep_index"),
        kept_width=geometry.get("target_kept_width"),
    )
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False, "shard": record.get("shard")},
        **record,
    }
