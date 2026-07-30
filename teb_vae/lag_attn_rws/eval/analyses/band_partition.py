r"""What each of the model's input channels is, read off the shards rather than re-derived.

The model consumes two streams, and every later frequency-resolved statement about either of them
needs to know which channel is which: that a given ``fhr_ph`` coefficient pairs filters at $0.021$
and $0.05$ Hz and therefore sits in the deceleration band, or that fourteen of the forty-three
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

* the target stream, $c_y = 43\ (\texttt{fhr\_st}) + 66\ (\texttt{fhr\_ph}) = 109$;
* the source stream, $c_u = 43\ (\texttt{up\_st}) + 15\ (\texttt{up\_ph}) = 58$, or $15$ alone
  when ``use_up_st`` is false.

Both blocks are laid out exactly as the model receives them -- ``cat([st, ph])`` -- so a channel
index in this map is the channel index a later analysis will hold, and the two streams are
distinguished by a ``stream`` column rather than by a reader remembering an offset.

**The scattering widths are read from the shard**, not passed in from a batch. This step runs
against a finished run directory with no model and no loader, which is what makes the channel map
recoverable after the fact; and the shard is the file the provenance is being read from anyway, so
taking the width from anywhere else would risk describing one file with another's layout.

**Absent provenance is a recorded skip, not a failure.** Whether a production shard predates
``_write_selection_attrs`` is a property of that file. A run whose readouts all succeeded must not
be marked failed because its shards are an older vintage, especially while nothing else in this
pipeline consumes the partition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from teb_vae.lag_attn_rws.eval._reuse import band_partition as shared

#: Files written into the results directory. The JSON round-trips into an equivalent partition for
#: anything reasoning about the groups; the CSV is one row per input channel so a plot can be
#: redrawn from disk with ``pandas`` and no import from this package at all.
PARTITION_FILENAME = "band_partition.json"
CHANNEL_MAP_FILENAME = "band_channel_map.csv"

#: ``stream -> (scattering dataset, phase dataset)``, in the order the model concatenates them.
#: The stream names are the model's own vocabulary -- the target is what is forecast, the source
#: is what the posterior additionally conditions on -- rather than the signal names, because a
#: reader of this map is asking which of the two model inputs a channel belongs to.
STREAM_DATASETS: Dict[str, Tuple[str, str]] = {
    "target": ("fhr_st", "fhr_ph"),
    "source": ("up_st", "up_ph"),
}


def _vae_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the model block of a merged run config, or an empty mapping."""
    return dict((config.get("model_config") or {}).get("VAE_model") or {})


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


def channel_rows(partitions: Dict[str, shared.BandPartition]) -> List[Dict[str, Any]]:
    """Flatten the per-stream partitions into one row per input channel.

    Args:
        partitions: Stream name to partition.

    Returns:
        One row per channel, each carrying its stream beside the shared channel record. The
        channel index is the index *within its stream*, which is the index the model's own
        tensors use.
    """
    return [
        {"stream": stream, **record.as_row()}
        for stream, partition in partitions.items()
        for record in partition.channels
    ]


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
    partitions: Dict[str, shared.BandPartition], output_dir: Any
) -> Dict[str, str]:
    """Write the per-stream partition JSON and the flat channel-map CSV.

    Args:
        partitions: Stream name to partition.
        output_dir: The results directory.

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
    pd.DataFrame(channel_rows(partitions)).to_csv(csv_path, index=False)
    return {
        "partition": json_path.relative_to(directory).as_posix(),
        "channel_map": csv_path.relative_to(directory).as_posix(),
    }


def emit_partition(
    shard_paths: Sequence[Any],
    output_dir: Any,
    *,
    use_up_st: bool = True,
    declared: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Build and write the input channel map, recording a clean skip when it cannot.

    Args:
        shard_paths: The configured test shards. The first readable one is used.
        output_dir: The results directory.
        use_up_st: Whether the source stream carries its scattering block.
        declared: The widths the configuration declares, compared against what the shard
            describes. A disagreement is **recorded, not raised**: preflight already refuses a
            width mismatch against the data the model is fed, and what this would add is a second
            refusal from a step that describes rather than decides.

    Returns:
        A JSON-safe record: either the written paths, the per-stream counts and the coverage, or
        ``skipped`` with every attempt's reason.
    """
    reasons: List[str] = []
    for path in shard_paths or ():
        try:
            partitions = build_stream_partitions(path, use_up_st=use_up_st)
        except Exception as exc:  # noqa: BLE001 - recorded and reported; see the module docstring
            reasons.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        written = write_partitions(partitions, output_dir)
        widths = {stream: partition.n_channels for stream, partition in partitions.items()}
        declared = dict(declared or {})
        disagreements: Dict[str, Dict[str, int]] = {}
        for stream, width in widths.items():
            expected = declared.get(stream)
            if expected is not None and int(expected) != int(width):
                disagreements[stream] = {
                    "declared": int(expected), "described_by_shard": int(width)
                }
        return {
            "skipped": False,
            "shard": str(path),
            "n_channels": int(sum(widths.values())),
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
    """Emit the input channel map for this run.

    Args:
        context: The analysis context, read for the merged config's shard list and channel widths.
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
    record = emit_partition(
        shards,
        output_dir,
        use_up_st=bool(_vae_config(config).get("use_up_st", True)),
        declared=declared_widths(config),
    )
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False, "shard": record.get("shard")},
        **record,
    }
