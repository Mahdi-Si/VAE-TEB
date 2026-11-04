r"""Cut a large HDF5 shard down to a control shard matched to the acidosis evaluation shard.

The reference shard ``acidosis_no_cs.hdf5`` holds 3341 segments from 157 recordings, all with
``bg_label = 1`` and ``cs_label = 0``. Each recording holds 5 to 34 segments, and ``epoch`` places
each segment up to about 12.4 h before delivery. This tool writes a shard of the same size from a
larger source shard. It pairs each reference recording with one source recording and copies the
same number of segments at about the same times before delivery. The two shards then balance
each other: same recording count, same segment count per recording, same time profile.

**The reference can be a profile.** The matching needs only the segment epochs of each reference
recording. ``assets/acidosis_no_cs_profile.json`` holds exactly that, plus the label values, the
dataset layout and a summary, and no GUIDs. It is the default reference, so the tool runs on a
machine that does not have the reference shard. ``--save-profile`` writes such a profile from any
reference shard.

**Candidates.** A source recording is a candidate when every segment passes the label filter
(by default ``bg_label = 0`` and ``cs_label = 0``). With an HDF5 reference, recordings that also
appear in the reference are dropped. A profile has no GUIDs, so this exclusion then relies on the
label filter: when the filter requires a label value that no reference segment has, no reference
recording can be a candidate. Otherwise the tool prints a warning.

**Recording pairing.** For reference recording $r$ with segment epochs $e_1, \dots, e_{n_r}$ and
candidate $c$ with epochs $E_c$, the cost is the mean distance to the nearest candidate epoch:
$$C_{rc} = \frac{1}{n_r} \sum_{i=1}^{n_r} \min_{t \in E_c} |e_i - t|,$$
and $C_{rc} = \infty$ when $|E_c| < n_r$. The Hungarian algorithm then picks the one-to-one pairing
with the smallest total cost. A small seeded jitter breaks ties, so file order does not decide.

**Segment selection.** Inside each pair, a second Hungarian assignment on $|e_i - t_j|$ picks
exactly $n_r$ distinct candidate segments.

**What the output keeps.** Every dataset of the source is copied with its dtype, chunk shape,
compression, ``maxshape`` and attributes. The root attributes are copied too; only
``source_guid_digest`` is recomputed for the new GUID set. It is the SHA-256 of the sorted GUIDs
joined by newlines, the formula of ``guid_set_digest`` in
``hdf5_dataset/new_pipeline/create_new_pipeline.py``; that module is not imported because it loads
torch. Rows keep the source order. No root attribute is added, because downstream tools compare
root attributes between shards. The selection record goes to ``<output>.selection.json``.

Run from the command line::

    python teb_vae/neural_surgery_trf_cfs/data_tools/make_matched_shard.py SRC [--dst DST]
        [--reference REF.json|REF.hdf5] [--label-filter bg_label=0,cs_label=0] [--seed 0]
        [--dry-run]
    python teb_vae/neural_surgery_trf_cfs/data_tools/make_matched_shard.py --reference REF.hdf5
        --save-profile OUT.json

or edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

ASSETS = Path(__file__).resolve().parents[1] / "assets"

#: Run-button inputs. Edit ``src`` (and ``dst`` if needed) and run the file with no arguments.
RUN_ARGS = {
    "src": r"/path/to/large_shard.hdf5",
    "reference": ASSETS / "acidosis_no_cs_profile.json",   # a profile JSON or a reference HDF5 shard
    "dst": None,                                  # None: <src dir>/<src stem>_matched_<ref name>.hdf5
    "label_filter": {"bg_label": 0, "cs_label": 0},
    "exclude_reference_guids": True,              # used with an HDF5 reference only
    "seed": 0,
    "rows_per_pass": 512,
    "dry_run": False,                             # True: select and report, write no HDF5
    "save_profile": None,                         # a path: write a profile of ``reference`` and stop
}

#: Cost for a candidate with too few segments; any finite cost is far below it.
INFEASIBLE = 1e12
#: Candidates per block in the cost computation, to bound memory.
COST_BLOCK = 2048
#: Label columns recorded in a profile and checked against the label filter.
LABELS = ("bg_label", "cs_label")


def guid_set_digest(guids: Sequence[str]) -> str:
    """SHA-256 of the sorted GUID stems, as ``create_new_pipeline.guid_set_digest`` computes it.

    Args:
        guids: The GUIDs of a shard.

    Returns:
        The hex digest.
    """
    stems = sorted({os.path.splitext(os.path.basename(g))[0] for g in guids})
    return hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()


def _text(values: np.ndarray) -> np.ndarray:
    """Decode a GUID column to Python strings.

    Args:
        values: The raw ``guid`` column (bytes or str objects).

    Returns:
        An object array of ``str``.
    """
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def _require(path: Union[str, Path], what: str) -> Path:
    """Fail early, with the fix, when an input file is missing.

    Args:
        path: The input path.
        what: The run argument that names it.

    Returns:
        The path.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}. Set RUN_ARGS['{what}'] or pass it on the command line.")
    return path


def read_index(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read the one-dimensional columns that the selection needs.

    Args:
        path: An HDF5 shard.

    Returns:
        ``guid`` (str), ``epoch`` and every present label or validity column, one entry per row.
    """
    with h5py.File(path, "r") as f:
        out = {"guid": _text(f["guid"][:]), "epoch": f["epoch"][:].astype(np.float64)}
        for name in (*LABELS, "fhr_valid_frac", "up_valid_frac", "time_from_labor_onset"):
            if name in f:
                out[name] = f[name][:]
    return out


def read_layout(path: Union[str, Path]) -> Dict[str, List[int]]:
    """Read the per-row shape of every dataset of a shard.

    Args:
        path: An HDF5 shard.

    Returns:
        ``{dataset name: shape without the row axis}``.
    """
    with h5py.File(path, "r") as f:
        return {k: list(v.shape[1:]) for k, v in f.items() if isinstance(v, h5py.Dataset)}


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
    for name in LABELS:
        if name in index:
            u, c = np.unique(index[name][rows], return_counts=True)
            out[name] = {str(int(a)): int(b) for a, b in zip(u, c)}
    for name in ("epoch", "fhr_valid_frac", "up_valid_frac", "time_from_labor_onset"):
        if name in index:
            out[f"{name}_quantiles"] = np.nanpercentile(index[name][rows].astype(np.float64), q).tolist()
    return out


# ---------------------------------------------------------------------------------- reference
def profile_from_shard(path: Union[str, Path]) -> dict:
    """Build a GUID-free reference profile from a reference shard.

    Recordings are sorted by segment count and then by epochs, so the order carries no GUID
    information.

    Args:
        path: The reference shard.

    Returns:
        ``{"name", "recordings": [{"epochs"}], "labels", "layout", "summary"}``.
    """
    index = read_index(path)
    groups = group_rows(index)
    epochs = sorted((index["epoch"][rows].tolist() for rows in groups.values()), key=lambda e: (len(e), e))
    return {"name": Path(path).stem,
            "recordings": [{"epochs": e} for e in epochs],
            "labels": {k: sorted(int(v) for v in np.unique(index[k])) for k in LABELS if k in index},
            "layout": read_layout(path),
            "summary": describe(index)}


def load_reference(path: Union[str, Path]) -> dict:
    """Load a reference from a profile JSON or a reference HDF5 shard.

    Args:
        path: The reference.

    Returns:
        The profile dictionary; an HDF5 reference also carries ``guids``, one per recording, in
        the profile's recording order.
    """
    path = _require(path, "reference")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    index = read_index(path)
    groups = group_rows(index)
    guids = sorted(groups, key=lambda g: (len(groups[g]), index["epoch"][groups[g]].tolist()))
    return {"name": path.stem,
            "recordings": [{"epochs": index["epoch"][groups[g]].tolist()} for g in guids],
            "labels": {k: sorted(int(v) for v in np.unique(index[k])) for k in LABELS if k in index},
            "layout": read_layout(path),
            "summary": describe(index),
            "guids": guids}


def check_exclusion(reference: dict, label_filter: Dict[str, int]) -> Optional[str]:
    """Say whether the label filter alone keeps every reference recording out of the candidates.

    Args:
        reference: The loaded reference.
        label_filter: The label filter.

    Returns:
        ``None`` when the filter excludes the reference, else a warning text.
    """
    for name, value in label_filter.items():
        if name in reference.get("labels", {}) and value not in reference["labels"][name]:
            return None
    return ("the reference has no GUIDs and the label filter does not exclude its labels "
            f"{reference.get('labels')}; reference recordings could be selected as controls")


# ---------------------------------------------------------------------------------- selection
def candidate_rows(index: Dict[str, np.ndarray], label_filter: Dict[str, int],
                   exclude: Sequence[str]) -> np.ndarray:
    """Find the rows of recordings that pass the label filter in every segment.

    Args:
        index: The source index.
        label_filter: ``{column: required value}``; every column must exist in the source.
        exclude: GUIDs to drop.

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
    keep = np.array([g not in bad for g in index["guid"]], dtype=bool) & ok
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
    padded = np.full((len(cand_epochs), counts.max()), np.inf)
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


def select(ref_epochs: List[np.ndarray], ref_names: List[str], cand_groups: Dict[str, np.ndarray],
           src_index: Dict[str, np.ndarray], seed: int) -> Tuple[np.ndarray, List[dict]]:
    """Pair recordings, then pick segments inside each pair.

    Args:
        ref_epochs: Sorted epochs per reference recording.
        ref_names: A name per reference recording (its GUID, or its profile position).
        cand_groups: Candidate source rows per GUID.
        src_index: Source index.
        seed: Seed of the tie-breaking jitter.

    Returns:
        ``(rows, pairs)``: the selected source rows in ascending order, and one record per pair.

    Raises:
        ValueError: If there are fewer candidates than reference recordings, or no full pairing
            exists because too few candidates have enough segments.
    """
    cand_guids = sorted(cand_groups)
    if len(cand_guids) < len(ref_epochs):
        raise ValueError(f"{len(cand_guids)} candidate recordings for {len(ref_epochs)} reference recordings; "
                         "relax the label filter or use a larger source")
    cand_e = [src_index["epoch"][cand_groups[g]] for g in cand_guids]
    cost = cost_matrix(ref_epochs, cand_e)
    cost += np.random.default_rng(seed).uniform(0.0, 1e-3, size=cost.shape)
    r_idx, c_idx = linear_sum_assignment(cost)
    infeasible = cost[r_idx, c_idx] >= INFEASIBLE
    if infeasible.any():
        need = sorted(len(ref_epochs[r]) for r in r_idx[infeasible])
        raise ValueError(f"{int(infeasible.sum())} reference recordings have no candidate with enough segments "
                         f"(they need {need}); relax the label filter or use a larger source")
    rows, pairs = [], []
    for r, c in zip(r_idx, c_idx):
        e, t = ref_epochs[r], cand_e[c]
        _, pick = linear_sum_assignment(np.abs(e[:, None] - t[None, :]))
        chosen = cand_groups[cand_guids[c]][pick]
        rows.extend(chosen.tolist())
        err = np.abs(e - src_index["epoch"][chosen])
        pairs.append({"reference_recording": ref_names[r], "control_guid": cand_guids[c], "segments": int(len(e)),
                      "mean_abs_epoch_error_s": float(err.mean()), "max_abs_epoch_error_s": float(err.max())})
    return np.array(sorted(rows)), pairs


# ---------------------------------------------------------------------------------- writing
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
        ValueError: If the source holds a group instead of flat datasets.
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


def default_dst(src: Union[str, Path], ref_name: str) -> Path:
    """Name the output next to the source.

    Args:
        src: The source shard.
        ref_name: The reference name.

    Returns:
        ``<src dir>/<src stem>_matched_<ref name>.hdf5``.
    """
    src = Path(src)
    return src.with_name(f"{src.stem}_matched_{ref_name}.hdf5")


def make_matched_shard(src, reference=RUN_ARGS["reference"], dst=None, label_filter=None,
                       exclude_reference_guids=True, seed=0, rows_per_pass=512, dry_run=False) -> dict:
    """Select a matched control set from ``src`` and write it as a new shard.

    Args:
        src: The large source shard.
        reference: A profile JSON or a reference shard, whose size and time profile the output
            matches.
        dst: Output path; ``None`` uses :func:`default_dst`.
        label_filter: ``{column: value}`` every segment of a candidate must satisfy; ``None``
            means no filter.
        exclude_reference_guids: Drop recordings that also appear in an HDF5 reference.
        seed: Seed of the tie-breaking jitter.
        rows_per_pass: Rows read per pass when copying.
        dry_run: Select and report without writing the HDF5 file.

    Returns:
        The selection report, which is also written to ``<dst>.selection.json``.
    """
    t0 = time.time()
    label_filter = label_filter or {}
    src = _require(src, "src")
    ref = load_reference(reference)
    dst = Path(dst) if dst is not None else default_dst(src, ref["name"].replace("_profile", ""))
    if not dry_run and dst.exists():
        raise FileExistsError(f"refusing to overwrite existing file {dst}; delete it or pick another path")

    src_layout = read_layout(src)
    missing = sorted(set(ref["layout"]) - set(src_layout))
    shapes = {k: [src_layout[k], v] for k, v in ref["layout"].items() if k in src_layout and src_layout[k] != v}
    if missing or shapes:
        print(f"WARNING: layout differs from the reference; missing {missing}, shape differences {shapes}")

    guids = ref.get("guids")
    exclude = set(guids) if (guids and exclude_reference_guids) else set()
    warning = None if guids else check_exclusion(ref, label_filter)
    if warning:
        print("WARNING:", warning)

    src_index = read_index(src)
    cand = candidate_rows(src_index, label_filter, exclude)
    cand_groups = group_rows(src_index, cand)
    ref_epochs = [np.sort(np.asarray(r["epochs"], dtype=np.float64)) for r in ref["recordings"]]
    ref_names = guids or [f"reference_{i:03d}" for i in range(len(ref_epochs))]
    print(f"reference: {sum(len(e) for e in ref_epochs)} segments, {len(ref_epochs)} recordings; "
          f"source candidates: {len(cand)} segments, {len(cand_groups)} recordings", flush=True)
    rows, pairs = select(ref_epochs, ref_names, cand_groups, src_index, seed)

    selected = sorted({src_index["guid"][r] for r in rows})
    report = {
        "source": str(src), "reference": str(reference), "output": None if dry_run else str(dst),
        "label_filter": label_filter, "exclude_reference_guids": bool(exclude), "exclusion_warning": warning,
        "seed": seed, "layout_missing_datasets": missing, "layout_shape_differences": shapes,
        "mean_abs_epoch_error_s": float(np.average([p["mean_abs_epoch_error_s"] for p in pairs],
                                                   weights=[p["segments"] for p in pairs])),
        "source_guid_digest": guid_set_digest(selected),
        "reference_summary": ref["summary"],
        "output_summary": describe(src_index, rows),
        "pairs": pairs,
    }
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
    p = argparse.ArgumentParser(description="Write a control shard matched to a reference shard or profile.")
    p.add_argument("src", nargs="?", default=a["src"])
    p.add_argument("--reference", default=str(a["reference"]))
    p.add_argument("--dst", default=a["dst"])
    p.add_argument("--label-filter", default=",".join(f"{k}={v}" for k, v in a["label_filter"].items()))
    p.add_argument("--keep-reference-guids", action="store_true", default=not a["exclude_reference_guids"])
    p.add_argument("--seed", type=int, default=a["seed"])
    p.add_argument("--rows-per-pass", type=int, default=a["rows_per_pass"])
    p.add_argument("--dry-run", action="store_true", default=a["dry_run"])
    p.add_argument("--save-profile", default=a["save_profile"],
                   help="write a GUID-free profile of --reference (an HDF5 shard) to this path and stop")
    args = p.parse_args(argv)
    if args.save_profile:
        out = Path(args.save_profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profile_from_shard(_require(args.reference, "reference")), indent=1),
                       encoding="utf-8")
        print(f"wrote {out}")
        return 0
    make_matched_shard(args.src, args.reference, args.dst, _parse_filter(args.label_filter),
                       not args.keep_reference_guids, args.seed, args.rows_per_pass, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
