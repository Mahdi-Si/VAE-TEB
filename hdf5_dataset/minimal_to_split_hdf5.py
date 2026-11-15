r"""Build one compact causal train/test dataset from the files of ``create_minimal_hdf5.py``.

No folds. Each subgroup's GUIDs are split once into train and test, by GUID, so no recording is in
both. Only the segments the model is meant to see are kept, and the coefficient blocks are stored
in float16:

* **Window.** A segment is kept if it lies wholly in the last ``last_hours`` before delivery
  (``epoch >= -last_hours * 3600``; every stored segment already ends before delivery).
* **Signal quality.** A segment is kept if at most ``max_gap_fraction`` of its FHR and of its UP lie
  in straight runs of 10 s or longer. That is what the MIMO fill writes into a gap (a hold or a
  line), and the minimal files keep no other record of where the gaps were.
* **Transform.** The pipeline's own ``create_hdf5_dataset_from_records_list``, causal bank, with the
  stored segments standing in for the ``.mat`` records (``minimal_to_full_hdf5.MinimalAdaptor``).
  The masks are computed on the CPU, so every file carries bit-identical attributes.
* **Storage.** ``fhr_st``, ``fhr_ph``, ``up_st`` and ``up_ph`` are 95 % of a file and are written as
  float16 (``half_precision``), every dataset with gzip + shuffle, one sample per chunk. The loader
  casts float16 back to float32 on read (``CombinedHDF5Dataset._create_tensor``) and the statistics
  calculator accumulates in float64. After the model's own normalisation (log on ``*_st``, asinh on
  ``*_ph``, then z-score) the float16 error is about 1e-4 sigma (median) and below 0.03 sigma at
  worst, measured on real segments. ``mantissa_bits`` below 10 also rounds away the low mantissa
  bits, which then compress: measured, 8 bits = -15 % at a worst error of 0.1 sigma, 7 bits = -23 %
  at 0.2 sigma (the worst values are tiny scattering values, which ``log(x + 1e-6)`` stretches).
  Every field the pipeline writes is kept: the small ones
  compress to almost nothing, and the causal build has no ``fhr_up_ph``.

Output, in ``output_dir``::

    train/<subgroup>.hdf5, test/<subgroup>.hdf5
    stats.hdf5                  over train and test alike, as build_fold_datasets.py prescribes
    dataset_files.yaml          vae_train_datasets / vae_test_datasets / stat_path, ready to paste
    split_guids.csv             one row per GUID: subgroup, split, segments kept
    segments.csv                one row per segment in the window: gap fractions, kept or not

Not supported, as in ``minimal_to_full_hdf5.py``: the signal-quality step (it needs the signals
before the MIMO fill). Use a configuration with every quality switch off, as ``tiny.yaml`` does.

Run: edit :data:`RUN_ARGS` at the bottom and press Run, or override a key on the command line.
A rerun into the same directory skips every complete file.
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``hdf5_dataset/minimal_to_split_hdf5.py`` -> up two.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np
import pandas as pd
import torch

# Production: every file in one folder. Repository: the packages. Each falls back only when the
# module itself is absent, so an error inside it is not hidden.
try:
    import minimal_to_full_hdf5 as m2f
except ModuleNotFoundError as _exc:
    if _exc.name != "minimal_to_full_hdf5":
        raise
    from hdf5_dataset import minimal_to_full_hdf5 as m2f
try:
    from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args
except ModuleNotFoundError as _exc:
    if not str(_exc.name).startswith("teb_vae"):
        raise
    from launch import missing_required, resolve_launch_args

cnp, bfd = m2f.cnp, m2f.bfd
logger = logging.getLogger(__name__)

#: The blocks stored as float16 under ``half_precision``.
COEFFICIENT_BLOCKS = ("fhr_st", "fhr_ph", "up_st", "up_ph")
#: A straight run this long (samples, 4 Hz) is a filled gap: 10 s.
GAP_MIN_RUN = 40
#: Rows per read while compacting and while measuring the gaps.
BATCH_ROWS = 2048


def gap_fraction(x: np.ndarray, min_run: int = GAP_MIN_RUN, tol: float = 1e-3) -> np.ndarray:
    """Per row of ``(n, L)``: the share of samples inside straight runs of ``min_run`` or more.

    Straight means a second difference below ``tol``; a hold and a linear bridge both are.
    """
    lin = np.abs(np.diff(x.astype(np.float64), 2, axis=1)) < tol
    edges = np.diff(np.pad(lin.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    out = np.zeros(len(x))
    for i, row in enumerate(edges):
        length = np.flatnonzero(row == -1) - np.flatnonzero(row == 1)
        out[i] = length[length >= min_run].sum()
    return out / lin.shape[1]


def select_segments(paths: Sequence[str], last_hours: float,
                    max_gap_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure every segment in the window and keep the good ones.

    Returns:
        ``(guids, segments)``: :func:`minimal_to_full_hdf5.scan_minimal`'s GUID table with the
        kept rows (``rows``) and their count (``n_segments``) added, GUIDs without a kept segment
        dropped; and one row per segment in the window.
    """
    guids, _ = m2f.scan_minimal(paths)
    epoch_min = -last_hours * 3600.0
    tables = []
    for path in map(os.path.abspath, paths):
        with h5py.File(path, "r") as f:
            epoch = f["epoch"][:]
            rows = np.flatnonzero(epoch >= epoch_min)
            names = f["guid"].asstr()[:][rows]
            gaps = {"fhr": np.zeros(len(rows)), "up": np.zeros(len(rows))}
            fhr_zero = np.zeros(len(rows), dtype=bool)
            for a in range(0, len(rows), BATCH_ROWS):
                sel = rows[a:a + BATCH_ROWS]
                for name in gaps:
                    x = f[name][sel[0]:sel[-1] + 1][sel - sel[0]]
                    gaps[name][a:a + len(sel)] = gap_fraction(x)
                    if name == "fhr":
                        fhr_zero[a:a + len(sel)] = (x == 0).any(axis=1)
        tables.append(pd.DataFrame(dict(
            file=path, row=rows, tracing_guid=names, epoch=epoch[rows],
            fhr_gap=gaps["fhr"], up_gap=gaps["up"], fhr_has_zero=fhr_zero)))
    segments = pd.concat(tables, ignore_index=True)
    segments["kept"] = (segments["fhr_gap"] <= max_gap_fraction) & (
        segments["up_gap"] <= max_gap_fraction)
    kept = segments[segments["kept"]].groupby("tracing_guid")["row"].apply(np.sort)
    guids = guids[guids["tracing_guid"].isin(kept.index)].copy()
    guids["rows"] = guids["tracing_guid"].map(kept)
    guids["n_segments"] = guids["rows"].map(len)
    return guids.reset_index(drop=True), segments


def split_guids(guids: pd.DataFrame, test_fraction: float, seed: int,
                max_guids_per_subgroup: Optional[int]) -> pd.DataFrame:
    """Give every GUID ``split`` = train or test, per subgroup; optionally subsample first.

    Each subgroup is shuffled by its own generator, seeded ``f"{seed}:{subgroup}"`` (the rule of
    ``build_fold_datasets.split_and_shard``), so one subgroup's split does not move when another
    changes. The first ``max_guids_per_subgroup`` of that order are kept; of those,
    ``floor((1 - test_fraction) n)`` go to train.
    """
    frames = []
    for sg, grp in guids.groupby("subgroup", sort=True):
        order = sorted(grp["tracing_guid"])
        random.Random(f"{seed}:{sg}").shuffle(order)
        order = order[:max_guids_per_subgroup]
        n_train = int(len(order) * (1.0 - test_fraction))
        split = {g: ("train" if i < n_train else "test") for i, g in enumerate(order)}
        frames.append(grp[grp["tracing_guid"].isin(split)].assign(
            split=lambda d: d["tracing_guid"].map(split)))
    return pd.concat(frames, ignore_index=True).sort_values(["split", "subgroup", "tracing_guid"])


def round_mantissa(x: np.ndarray, bits: int) -> np.ndarray:
    """Round float16 values to *bits* of their 10 mantissa bits (nearest, ties up in magnitude)."""
    drop = 10 - bits
    if drop <= 0:
        return x
    u = x.view(np.uint16).astype(np.uint32)
    u = ((u + (1 << (drop - 1))) >> drop) << drop
    # A carry out of an exponent's top value would reach inf: clamp to the largest finite value.
    out = np.minimum(u, 0x7BFF | (u & 0x8000)).astype(np.uint16).view(np.float16)
    return np.where(np.isfinite(x), out, x)


def compact(src: str, dst: str, half_precision: bool, mantissa_bits: int = 10) -> None:
    """Copy *src* into *dst*: the same datasets, attributes and chunks, the coefficient blocks in
    float16 under *half_precision* (rounded to *mantissa_bits*), every dataset but ``guid`` with
    gzip + shuffle."""
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        for key, value in s.attrs.items():
            d.attrs[key] = value
        for name, ds in s.items():
            is_str = ds.dtype.kind == "O"
            dtype = np.float16 if half_precision and name in COEFFICIENT_BLOCKS else ds.dtype
            new = d.create_dataset(
                name, shape=ds.shape, maxshape=ds.maxshape, dtype=dtype, chunks=ds.chunks,
                compression=None if is_str else "gzip", compression_opts=None if is_str else 4,
                shuffle=not is_str)
            for key, value in ds.attrs.items():
                new.attrs[key] = value
            for a in range(0, ds.shape[0], BATCH_ROWS):
                if is_str:
                    new[a:a + BATCH_ROWS] = ds.asstr()[a:a + BATCH_ROWS]
                elif dtype == np.float16:
                    new[a:a + BATCH_ROWS] = round_mantissa(
                        ds[a:a + BATCH_ROWS].astype(np.float16), mantissa_bits)
                else:
                    new[a:a + BATCH_ROWS] = ds[a:a + BATCH_ROWS]
        if half_precision and mantissa_bits < 10:
            d.attrs["float16_mantissa_bits"] = mantissa_bits


def compute_statistics(files: Sequence[str], output_file: str, trim_minutes: float,
                       device: str) -> None:
    """Run ``calculate_dataset_stats.py`` over *files*, as a script.

    As a script, not an import: it imports ``from hdf5_dataset import ...`` and means the module
    ``hdf5_dataset.py`` beside it, which only its own folder at the front of ``sys.path`` gives;
    the repository root goes on ``PYTHONPATH`` for its other imports. This launch works in the
    repository and in the production folder alike.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calculate_dataset_stats.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_REPO_ROOT, env.get("PYTHONPATH")) if p)
    subprocess.run([sys.executable, script, "--input-files", *files, "--output-file", output_file,
                    "--trim-minutes", str(trim_minutes), "--device", device,
                    "--no-plot-histograms"], env=env, check=True)


def main(minimal_paths: Sequence[str], output_dir: str, last_hours: Optional[float] = None,
         max_gap_fraction: Optional[float] = None, test_fraction: Optional[float] = None,
         max_guids_per_subgroup: Optional[int] = None, seed: Optional[int] = None,
         half_precision: Optional[bool] = None, mantissa_bits: Optional[int] = None,
         leg_alignment: Optional[str] = None,
         phase_operator: Optional[str] = None, device: Optional[str] = None,
         scatter_batch_size: Optional[int] = None, trim_minutes: Optional[float] = None,
         compute_stats: Optional[bool] = None, verbose: Optional[bool] = None) -> int:
    """Apply the defaults, select, split, transform, compact and compute the statistics."""
    last_hours = 4.0 if last_hours is None else float(last_hours)
    max_gap_fraction = 0.10 if max_gap_fraction is None else float(max_gap_fraction)
    test_fraction = 0.2 if test_fraction is None else float(test_fraction)
    seed = cnp.RANDOM_STATE if seed is None else int(seed)
    half_precision = True if half_precision is None else bool(half_precision)
    mantissa_bits = 10 if mantissa_bits is None else int(mantissa_bits)
    if not 1 <= mantissa_bits <= 10:
        raise ValueError(f"mantissa_bits={mantissa_bits}; float16 has 1..10")
    leg_alignment = cnp.validate_leg_alignment(leg_alignment or "envelope")
    phase_operator = phase_operator or cnp.PHASE_OPERATOR_INTEGER
    cnp.validate_phase_operator(phase_operator)
    trim_minutes = 1.0 if trim_minutes is None else float(trim_minutes)
    paths = [os.path.abspath(p) for p in minimal_paths]
    os.makedirs(output_dir, exist_ok=True)

    guids, segments = select_segments(paths, last_hours, max_gap_fraction)
    segments.to_csv(os.path.join(output_dir, "segments.csv"), index=False)
    guids = split_guids(guids, test_fraction, seed, max_guids_per_subgroup)
    guids.drop(columns="rows").to_csv(os.path.join(output_dir, "split_guids.csv"), index=False)
    kept = segments[segments["kept"]]
    logger.info(f"window: {len(segments)} segments, kept {len(kept)} "
                f"({kept['fhr_has_zero'].mean():.1%} of them hold an FHR zero)")
    logger.info("GUIDs / segments per split and subgroup:\n" + guids.groupby(
        ["split", "subgroup"])["n_segments"].agg(["count", "sum"]).to_string())

    m2f.MinimalAdaptor.index = {g: (f, r) for g, f, r in zip(
        guids["tracing_guid"], guids["record_path"].map(os.path.dirname), guids["rows"])}
    cnp.EarlyMaestraMimoAdaptor = m2f.MinimalAdaptor
    keys = guids["tracing_guid"].map(cnp._normalize_guid)
    labor_map = dict(zip(keys, guids["tlo_hours"] * 3600.0))
    ss_map = dict(zip(keys, guids["ss_hours"] * 3600.0))
    masks = cnp.compute_scattering_masks(
        cnp.SIGNAL_LENGTH, scattering_T=16, device=torch.device("cpu"), transform=cnp.CAUSAL,
        leg_alignment=leg_alignment, phase_operator=phase_operator)
    source = os.path.join(os.path.abspath(output_dir), "split_guids.csv")
    torch_device = torch.device(device) if device else None

    files: Dict[str, List[str]] = {"train": [], "test": []}
    for (split, sg), grp in guids.groupby(["split", "subgroup"], sort=True):
        final = os.path.join(output_dir, split, f"{sg}.hdf5")
        files[split].append(os.path.abspath(final))
        records = grp["record_path"].tolist()
        if bfd.is_complete(final, cnp.guid_set_digest(records), source):
            logger.info(f"complete, skipped: {final}")
            continue
        os.makedirs(os.path.dirname(final), exist_ok=True)
        full = final + ".full.partial"
        cnp.create_hdf5_for_masks(full, masks, len_sequence=cnp.SEQUENCE_LENGTH,
                                  records_list=records, source_pickle_path=source)
        target, cs, bg = cnp.SUBGROUP_META[sg]
        errors = cnp.create_hdf5_dataset_from_records_list(
            full, records, cs, bg, target, masks, labor_map, ss_map, device=torch_device,
            scatter_batch_size=int(scatter_batch_size or 64), verbose=bool(verbose),
            transform=cnp.CAUSAL)
        if errors:
            raise RuntimeError(f"{final}: {len(errors)} records failed, e.g. {errors[:3]}")
        compact(full, final + ".partial", half_precision, mantissa_bits)
        os.replace(final + ".partial", final)
        os.remove(full)
        logger.info(f"wrote {final}: {os.path.getsize(final) / 1e6:.1f} MB")

    stat_path = os.path.abspath(os.path.join(output_dir, "stats.hdf5"))
    if compute_stats is None or compute_stats:
        # Over train and test alike, the rule build_fold_datasets.write_dataset_files states.
        compute_statistics(files["train"] + files["test"], stat_path, trim_minutes, device or "cpu")
    with open(os.path.join(output_dir, "dataset_files.yaml"), "w", encoding="utf-8") as f:
        f.write(f"# Built by hdf5_dataset/minimal_to_split_hdf5.py: last {last_hours:g} h, "
                f"gaps <= {max_gap_fraction:.0%}, trim_minutes {trim_minutes:g}.\n")
        for key, split in (("vae_train_datasets", "train"), ("vae_test_datasets", "test")):
            f.write(f"{key}:\n")
            f.writelines(f"  - {p}\n" for p in files[split])
        f.write(f"stat_path: {stat_path}\n")
    total = sum(os.path.getsize(p) for p in files["train"] + files["test"])
    logger.info(f"Done: {int(guids['n_segments'].sum())} segments, {total / 1e9:.2f} GB in "
                f"{len(files['train']) + len(files['test'])} files, {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser. No defaults here: :data:`RUN_ARGS` fills every flag not given."""
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/minimal_to_split_hdf5.py",
        description="One compact causal train/test dataset from the minimal raw-signal files.")
    p.add_argument("--minimal-paths", dest="minimal_paths", nargs="+")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--last-hours", dest="last_hours", type=float)
    p.add_argument("--max-gap-fraction", dest="max_gap_fraction", type=float)
    p.add_argument("--test-fraction", dest="test_fraction", type=float)
    p.add_argument("--max-guids-per-subgroup", dest="max_guids_per_subgroup", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--half-precision", dest="half_precision",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--mantissa-bits", dest="mantissa_bits", type=int)
    p.add_argument("--leg-alignment", dest="leg_alignment")
    p.add_argument("--phase-operator", dest="phase_operator")
    p.add_argument("--device", dest="device")
    p.add_argument("--scatter-batch-size", dest="scatter_batch_size", type=int)
    p.add_argument("--trim-minutes", dest="trim_minutes", type=float)
    p.add_argument("--compute-stats", dest="compute_stats",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--verbose", dest="verbose", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("minimal_paths", "output_dir"))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{k}={values[k]!r} (from {sources[k]})" for k in sorted(values)))
    return main(**values)


_MINIMAL_DIR = os.path.join(_REPO_ROOT, "tmp", "dataset")

#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # list[str] -- the files written by create_minimal_hdf5.py. Required.
    "minimal_paths": [os.path.join(_MINIMAL_DIR, f"{c}.hdf5")
                      for c in ("acidosis", "hie", "healthy")],
    # str -- the output directory. Required. A rerun into it skips complete files.
    "output_dir": os.path.join(_MINIMAL_DIR, "split_last4h_gap10"),

    # ================================ Selection ================================
    # float | None -- keep segments wholly inside the last N hours before delivery.  (None -> 4)
    "last_hours": 4.0,
    # float | None -- most of FHR, and of UP, in filled gaps (straight runs >= 10 s). (None -> 0.10)
    "max_gap_fraction": 0.10,
    # float | None -- test share of each subgroup's GUIDs.  (None -> 0.2)
    "test_fraction": 0.2,
    # int | None -- at most this many GUIDs per subgroup, drawn with the seed; for a quick
    #        subset.  (None -> all)
    "max_guids_per_subgroup": None,
    # int | None -- seed of the subsample and the split.  (None -> 42)
    "seed": None,

    # ========================= Transform and storage =========================
    # bool | None -- the four coefficient blocks in float16.  (None -> True)
    "half_precision": True,
    # int | None -- float16 mantissa bits kept, 1..10; fewer compress better and cost precision.
    #        Measured: 10 = 65 KB/segment (worst error 0.03 sigma), 8 = -15 % (0.1 sigma),
    #        7 = -23 % (0.2 sigma). Recorded as the root attribute float16_mantissa_bits.
    #        (None -> 10, plain float16)
    "mantissa_bits": None,
    # "none" | "envelope"                          (None -> "envelope", what the cfs configs expect)
    "leg_alignment": "envelope",
    # "ratio_power_v0" | "integer_harmonic_v1"     (None -> "integer_harmonic_v1")
    "phase_operator": "integer_harmonic_v1",
    # str | None -- torch device of the transform and the statistics, e.g. "cuda:0".
    #        (None -> the first CUDA device for the transform, the CPU for the statistics)
    "device": None,
    # int | None -- segments per forward pass of the transform.  (None -> 64)
    "scatter_batch_size": 64,
    # float | None -- trim of the statistics; must equal the config's trim_minutes.  (None -> 1.0)
    "trim_minutes": 1.0,
    # bool | None -- compute stats.hdf5 over every file.  (None -> True)
    "compute_stats": True,
    # bool | None -- per-record progress output.  (None -> False)
    "verbose": False,
}


if __name__ == "__main__":
    sys.exit(_cli())
