r"""Cut a large HDF5 shard down to a control shard matched to the acidosis evaluation shard.

The reference shard ``acidosis_no_cs.hdf5`` holds 3341 segments from 157 recordings, all with
``bg_label = 1`` and ``cs_label = 0``. Each recording holds 5 to 34 segments, and ``epoch`` places
each segment up to about 12.4 h before delivery. This tool writes a shard of the same size from a
larger source shard. It pairs each reference recording with one source recording and copies the
same number of segments at about the same times before delivery. The two shards then balance
each other: same recording count, same segment count per recording, same time profile.

**Candidates.** A source recording is a candidate when every segment passes the label filter
(by default ``bg_label = 0`` and ``cs_label = 0``) and its GUID is not in the reference shard.
A candidate needs at least as many segments as the reference recording it replaces.

**Recording pairing.** For reference recording $r$ with segment epochs $e_1, \dots, e_{n_r}$ and
candidate $c$ with epochs $E_c$, the cost is the mean distance to the nearest candidate epoch:
$$C_{rc} = \frac{1}{n_r} \sum_{i=1}^{n_r} \min_{t \in E_c} |e_i - t|,$$
and $C_{rc} = \infty$ when $|E_c| < n_r$. The Hungarian algorithm then picks the one-to-one pairing
with the smallest total cost. A small seeded jitter breaks ties, so file order does not decide.

**Segment selection.** Inside each pair, a second Hungarian assignment on $|e_i - t_j|$ picks
exactly $n_r$ distinct candidate segments.

**What the output keeps.** Every dataset of the source is copied with its dtype, chunk shape,
compression, ``maxshape`` and attributes. The root attributes are copied too; only
``source_guid_digest`` is recomputed for the new GUID set, with the pipeline's own
``guid_set_digest``. Rows keep the source order. No root attribute is added, because downstream
tools compare root attributes between shards. The selection record goes to a JSON file next to
the output instead: the pairing, the epoch error and the reference-against-output distributions.

Run from the command line::

    python teb_vae/neural_surgery_trf_cfs/data_tools/make_matched_shard.py SRC [--dst DST]
        [--reference REF] [--label-filter bg_label=0,cs_label=0] [--seed 0] [--dry-run]

or edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from hdf5_dataset.new_pipeline.create_new_pipeline import guid_set_digest  # noqa: E402
from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402

#: Run-button inputs. Edit ``src`` (and ``dst`` if needed) and run the file with no arguments.
RUN_ARGS = {
    "src": r"C:\path\to\large_shard.hdf5",
    "reference": common.DATA,
    "dst": None,                                  # None: <src dir>/<src stem>_matched_<ref stem>.hdf5
    "label_filter": {"bg_label": 0, "cs_label": 0},
    "exclude_reference_guids": True,
    "seed": 0,
    "rows_per_pass": 512,
    "dry_run": False,                             # True: select and report, write no HDF5
}

#: Cost for a candidate with too few segments; any finite cost is far below it.
INFEASIBLE = 1e12
#: Candidates per block in the cost computation, to bound memory.
COST_BLOCK = 2048


def _text(values: np.ndarray) -> np.ndarray:
    """Decode a GUID column to Python strings.

    Args:
        values: The raw ``guid`` column (bytes or str objects).

    Returns:
        An object array of ``str``.
    """
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def read_index(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read the one-dimensional columns that the selection needs.

    Args:
        path: An HDF5 shard.

    Returns:
        ``guid`` (str), ``epoch`` and every present label or validity column, one entry per row.
    """
    with h5py.File(path, "r") as f:
        out = {"guid": _text(f["guid"][:]), "epoch": f["epoch"][:].astype(np.float64)}
        for name in ("bg_label", "cs_label", "fhr_valid_frac", "up_valid_frac", "time_from_labor_onset"):
            if name in f:
                out[name] = f[name][:]
    return out


def group_rows(index: Dict[str, np.ndarray], rows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Group row numbers by recording, each group sorted by epoch.

    Args:
        index: Output of :func:`read_index`.
        rows: Optional subset of row numbers; ``None`` uses every row.

    Returns:
        ``{guid: row numbers sorted by epoch}``.
    """
    rows = np.arange(len(index["guid"])) if rows is None else rows
    groups: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        groups[index["guid"][r]].append(int(r))
    return {g: np.array(sorted(v, key=lambda r: index["epoch"][r])) for g, v in groups.items()}


def candidate_rows(index: Dict[str, np.ndarray], label_filter: Dict[str, int],
                   exclude: Sequence[str]) -> np.ndarray:
    """Find the rows of recordings that pass the label filter in every segment.

    Args:
        index: The source index.
        label_filter: ``{column: required value}``; every column must exist in the source.
        exclude: GUIDs to drop, normally those of the reference shard.

    Returns:
        The kept row numbers.

    Raises:
        KeyError: If a filter column is missing from the source.
    """
    ok = np.ones(len(index["guid"]), dtype=bool)
    for name, value in label_filter.items():
        if name not in index:
            raise KeyError(f"label filter column {name!r} is not in the source shard")
        ok &= index[name] == value
    bad = set(index["guid"][~ok]) | set(exclude)
    keep = np.array([g not in bad for g in index["guid"]]) & ok
    return np.flatnonzero(keep)


def cost_matrix(ref_epochs: List[np.ndarray], cand_epochs: List[np.ndarray]) -> np.ndarray:
    r"""Mean nearest-epoch distance $C_{rc}$ for every reference and candidate recording.

    Args:
        ref_epochs: Sorted epochs per reference recording.
        cand_epochs: Sorted epochs per candidate recording.

    Returns:
        The cost matrix $(R, N)$ in seconds, with :data:`INFEASIBLE` where the candidate has fewer
        segments than the reference recording.
    """
    counts = np.array([len(e) for e in cand_epochs])
    width = counts.max()
    padded = np.full((len(cand_epochs), width), np.inf)
    for i, e in enumerate(cand_epochs):
        padded[i, :len(e)] = e
    cost = np.empty((len(ref_epochs), len(cand_epochs)))
    for r, e in enumerate(ref_epochs):
        for s in range(0, len(cand_epochs), COST_BLOCK):
            block = padded[s:s + COST_BLOCK]
            nearest = np.abs(block[:, :, None] - e[None, None, :]).min(axis=1)
            cost[r, s:s + COST_BLOCK] = nearest.mean(axis=1)
        cost[r, counts < len(e)] = INFEASIBLE
    return cost


def select(ref_groups: Dict[str, np.ndarray], ref_index: Dict[str, np.ndarray],
           cand_groups: Dict[str, np.ndarray], src_index: Dict[str, np.ndarray],
           seed: int) -> Tuple[np.ndarray, List[dict]]:
    """Pair recordings, then pick segments inside each pair.

    Args:
        ref_groups: Reference rows per GUID.
        ref_index: Reference index.
        cand_groups: Candidate source rows per GUID.
        src_index: Source index.
        seed: Seed of the tie-breaking jitter.

    Returns:
        ``(rows, pairs)``: the selected source rows in ascending order, and one record per pair.

    Raises:
        ValueError: If there are fewer candidates than reference recordings, or no full pairing
            exists because too few candidates have enough segments.
    """
    ref_guids = sorted(ref_groups)
    cand_guids = sorted(cand_groups)
    if len(cand_guids) < len(ref_guids):
        raise ValueError(f"{len(cand_guids)} candidate recordings for {len(ref_guids)} reference recordings")
    ref_e = [ref_index["epoch"][ref_groups[g]] for g in ref_guids]
    cand_e = [src_index["epoch"][cand_groups[g]] for g in cand_guids]
    cost = cost_matrix(ref_e, cand_e)
    cost += np.random.default_rng(seed).uniform(0.0, 1e-3, size=cost.shape)
    r_idx, c_idx = linear_sum_assignment(cost)
    short = int((cost[r_idx, c_idx] >= INFEASIBLE).sum())
    if short:
        need = sorted(len(ref_e[r]) for r in r_idx[cost[r_idx, c_idx] >= INFEASIBLE])
        raise ValueError(f"{short} reference recordings have no candidate with enough segments "
                         f"(they need {need}); relax the label filter or use a larger source")
    rows, pairs = [], []
    for r, c in zip(r_idx, c_idx):
        e, t = ref_e[r], cand_e[c]
        _, pick = linear_sum_assignment(np.abs(e[:, None] - t[None, :]))
        chosen = cand_groups[cand_guids[c]][pick]
        rows.extend(chosen.tolist())
        err = np.abs(e - src_index["epoch"][chosen])
        pairs.append({"reference_guid": ref_guids[r], "control_guid": cand_guids[c], "segments": int(len(e)),
                      "mean_abs_epoch_error_s": float(err.mean()), "max_abs_epoch_error_s": float(err.max())})
    return np.array(sorted(rows)), pairs


def write_subset(src: Union[str, Path], dst: Union[str, Path], rows: np.ndarray, digest: str,
                 rows_per_pass: int) -> None:
    """Copy the selected rows of every dataset into a new shard with the source layout.

    Args:
        src: The source shard.
        dst: The output shard; refused if it exists.
        rows: Ascending source row numbers.
        digest: The new ``source_guid_digest``.
        rows_per_pass: Rows read per pass.

    Raises:
        FileExistsError: If ``dst`` exists.
    """
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite existing file {dst}; delete it or pick another path")
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    with h5py.File(src, "r") as source, h5py.File(dst, "w", libver="latest") as target:
        target.attrs.update(source.attrs)
        if "source_guid_digest" in source.attrs:
            target.attrs["source_guid_digest"] = digest
        for name, ds in source.items():
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"group {name!r} found; this tool copies flat shards only")
            maxshape = (None, *ds.shape[1:]) if ds.chunks is not None else None
            copy = target.create_dataset(
                name, shape=(n, *ds.shape[1:]), dtype=ds.dtype, maxshape=maxshape, chunks=ds.chunks,
                compression=ds.compression, compression_opts=ds.compression_opts,
                shuffle=ds.shuffle, fletcher32=ds.fletcher32)
            copy.attrs.update(ds.attrs)
            for start in range(0, n, rows_per_pass):
                block = rows[start:start + rows_per_pass]
                lo, hi = int(block[0]), int(block[-1]) + 1
                if hi - lo <= 4 * len(block):
                    copy[start:start + len(block)] = ds[lo:hi][block - lo]
                else:
                    copy[start:start + len(block)] = ds[block.tolist()]
            print(f"  copied {name} {copy.shape}", flush=True)


def describe(index: Dict[str, np.ndarray], rows: Optional[np.ndarray] = None) -> dict:
    """Summarise the size, labels and distributions of a set of rows.

    Args:
        index: A shard index.
        rows: Row numbers; ``None`` uses every row.

    Returns:
        Counts, label counts and quantiles of the epoch and validity columns.
    """
    rows = np.arange(len(index["guid"])) if rows is None else rows
    q = [0, 5, 25, 50, 75, 95, 100]
    groups = group_rows(index, rows)
    per_rec = np.array([len(v) for v in groups.values()])
    out = {"segments": int(len(rows)), "recordings": len(groups),
           "segments_per_recording_quantiles": np.percentile(per_rec, q).tolist()}
    for name in ("bg_label", "cs_label"):
        if name in index:
            u, c = np.unique(index[name][rows], return_counts=True)
            out[name] = {str(int(a)): int(b) for a, b in zip(u, c)}
    for name in ("epoch", "fhr_valid_frac", "up_valid_frac", "time_from_labor_onset"):
        if name in index:
            out[f"{name}_quantiles"] = np.nanpercentile(index[name][rows].astype(np.float64), q).tolist()
    return out


def default_dst(src: Union[str, Path], reference: Union[str, Path]) -> Path:
    """Name the output next to the source.

    Args:
        src: The source shard.
        reference: The reference shard.

    Returns:
        ``<src dir>/<src stem>_matched_<reference stem>.hdf5``.
    """
    src, reference = Path(src), Path(reference)
    return src.with_name(f"{src.stem}_matched_{reference.stem}.hdf5")


def make_matched_shard(src, reference=common.DATA, dst=None, label_filter=None, exclude_reference_guids=True,
                       seed=0, rows_per_pass=512, dry_run=False) -> dict:
    """Select a matched control set from ``src`` and write it as a new shard.

    Args:
        src: The large source shard.
        reference: The shard whose size and time profile the output matches.
        dst: Output path; ``None`` uses :func:`default_dst`.
        label_filter: ``{column: value}`` every segment of a candidate must satisfy; ``None``
            means no filter.
        exclude_reference_guids: Drop recordings that also appear in the reference.
        seed: Seed of the tie-breaking jitter.
        rows_per_pass: Rows read per pass when copying.
        dry_run: Select and report without writing the HDF5 file.

    Returns:
        The selection report, which is also written to ``<dst>.selection.json``.
    """
    t0 = time.time()
    dst = Path(dst) if dst is not None else default_dst(src, reference)
    if not dry_run and dst.exists():
        raise FileExistsError(f"refusing to overwrite existing file {dst}; delete it or pick another path")
    ref_index = read_index(reference)
    src_index = read_index(src)
    with h5py.File(src, "r") as s, h5py.File(reference, "r") as r:
        missing = sorted(set(r) - set(s))
        shapes = {k: (s[k].shape[1:], r[k].shape[1:]) for k in set(r) & set(s) if s[k].shape[1:] != r[k].shape[1:]}
    if missing or shapes:
        print(f"WARNING: layout differs from the reference; missing {missing}, shape differences {shapes}")
    ref_groups = group_rows(ref_index)
    exclude = set(ref_groups) if exclude_reference_guids else set()
    cand = candidate_rows(src_index, label_filter or {}, exclude)
    cand_groups = group_rows(src_index, cand)
    print(f"reference: {len(ref_index['guid'])} segments, {len(ref_groups)} recordings; "
          f"source candidates: {len(cand)} segments, {len(cand_groups)} recordings", flush=True)
    rows, pairs = select(ref_groups, ref_index, cand_groups, src_index, seed)
    err = np.array([p["mean_abs_epoch_error_s"] for p in pairs])
    report = {
        "source": str(src), "reference": str(reference), "output": None if dry_run else str(dst),
        "label_filter": label_filter, "exclude_reference_guids": exclude_reference_guids, "seed": seed,
        "layout_missing_datasets": missing, "layout_shape_differences": {k: [list(a), list(b)] for k, (a, b) in shapes.items()},
        "mean_abs_epoch_error_s": float(np.average(err, weights=[p["segments"] for p in pairs])),
        "reference_summary": describe(ref_index),
        "output_summary": describe(src_index, rows),
        "pairs": pairs,
    }
    selected = sorted({src_index["guid"][r] for r in rows})
    report["source_guid_digest"] = guid_set_digest(selected)
    print(f"selected {len(rows)} segments from {len(selected)} recordings; "
          f"mean |epoch error| {report['mean_abs_epoch_error_s']:.1f} s", flush=True)
    if not dry_run:
        write_subset(src, dst, rows, report["source_guid_digest"], rows_per_pass)
    side = dst.with_suffix(".selection.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"wrote {side if dry_run else dst} in {time.time() - t0:.0f}s")
    return report


def _parse_filter(text: str) -> Dict[str, int]:
    """Parse ``"bg_label=0,cs_label=0"``; an empty string means no filter.

    Args:
        text: The filter text.

    Returns:
        ``{column: value}``.
    """
    return {k.strip(): int(v) for k, v in (p.split("=") for p in text.split(",") if p.strip())}


def main(argv: Optional[list] = None) -> int:
    """Command-line entry; with no arguments, uses :data:`RUN_ARGS`.

    Args:
        argv: Arguments to parse, or ``None`` for ``sys.argv``.

    Returns:
        The process exit code.
    """
    a = RUN_ARGS
    p = argparse.ArgumentParser(description="Write a control shard matched to a reference shard.")
    p.add_argument("src", nargs="?", default=a["src"])
    p.add_argument("--reference", default=str(a["reference"]))
    p.add_argument("--dst", default=a["dst"])
    p.add_argument("--label-filter", default=",".join(f"{k}={v}" for k, v in a["label_filter"].items()))
    p.add_argument("--keep-reference-guids", action="store_true", default=not a["exclude_reference_guids"])
    p.add_argument("--seed", type=int, default=a["seed"])
    p.add_argument("--rows-per-pass", type=int, default=a["rows_per_pass"])
    p.add_argument("--dry-run", action="store_true", default=a["dry_run"])
    args = p.parse_args(argv)
    make_matched_shard(args.src, args.reference, args.dst, _parse_filter(args.label_filter),
                       not args.keep_reference_guids, args.seed, args.rows_per_pass, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
