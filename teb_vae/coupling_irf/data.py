"""Per-step (4 s) cache of FHR and UA in physical units, built once from a split's shards.

Step grid: step t (0..299) covers the 16 untrimmed raw samples 240 + 16 t - 15 .. 240 + 16 t, the
window the existing loader gives trimmed step t (``hdf5_dataset.py``: ``_raw_window``, ``_step_masks``).

Validity, as ``CombinedHDF5Dataset._quality_masks``: a UA sample is valid where
``up_quality & invalid_bits == 0``; an FHR sample where ``fhr_quality & B == 0`` with B the
segment's ``invalid_bits_exact`` if ``fhr_repair_exact`` else ``invalid_bits_rules``. The bit sets
are read from the dataset attributes of each shard, as the loader reads them.
"""
import argparse
import hashlib
import time
from pathlib import Path

import h5py
import numpy as np

TRIM, D, T = 240, 16, 300  # raw samples trimmed per end, raw samples per step, steps per segment
LO = TRIM - D + 1          # first raw sample of step 0
META = {"epoch": np.float32, "cs_label": np.uint8, "bg_label": np.uint8, "second_stage_onset": np.float32}


def _steps(x, valid):
    """(n, 5280) signal and bool mask -> per-step mean, SD (ddof=1) and valid fraction, each (n, 300)."""
    x = x[:, LO:LO + D * T].reshape(-1, T, D).astype(np.float64)
    v = valid[:, LO:LO + D * T].reshape(-1, T, D)
    n = v.sum(-1)
    mean = np.where(n > 0, (x * v).sum(-1) / np.maximum(n, 1), x.mean(-1))  # no valid sample: gap-filled mean
    sd = np.sqrt((((x - mean[..., None]) ** 2) * v).sum(-1) / np.maximum(n - 1, 1)) * (n > 1)
    return mean.astype(np.float32), sd.astype(np.float16), (n / D).astype(np.float16)


def build_step_cache(split_dir, out_dir, chunk=512):
    """Write ``<out_dir>/steps_<split>.npz`` for every split folder of ``split_dir``; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        cols = {k: [] for k in ("fhr", "fhr_sd", "fhr_ok", "ua", "ua_ok", "guid", "row", "subgroup", *META)}
        for shard in sorted(split.glob("*.hdf5")):  # sorted subgroup names, rows in shard order
            with h5py.File(shard, "r") as f:
                n = f["fhr"].shape[0]
                up_bad = int(f["up_quality"].attrs["invalid_bits"])
                fa = f["fhr_quality"].attrs
                fhr_bad = np.where(f["fhr_repair_exact"][:].astype(bool),
                                   int(fa["invalid_bits_exact"]), int(fa["invalid_bits_rules"]))
                for a in range(0, n, chunk):
                    b = min(a + chunk, n)
                    m, sd, ok = _steps(f["fhr"][a:b], (f["fhr_quality"][a:b] & fhr_bad[a:b, None]) == 0)
                    cols["fhr"].append(m), cols["fhr_sd"].append(sd), cols["fhr_ok"].append(ok)
                    m, _, ok = _steps(f["up"][a:b], (f["up_quality"][a:b] & up_bad) == 0)
                    cols["ua"].append(m), cols["ua_ok"].append(ok)
                cols["guid"].append(f["guid"][:].astype(str))
                cols["row"].append(np.arange(n, dtype=np.int32))
                cols["subgroup"].append(np.full(n, shard.stem))
                for k, dtype in META.items():
                    cols[k].append(f[k][:].astype(dtype))
        out = {k: np.concatenate(v) for k, v in cols.items()}
        out["cs"], out["bg"] = out.pop("cs_label"), out.pop("bg_label")
        paths.append(out_dir / f"steps_{split.name}.npz")
        np.savez_compressed(paths[-1], **out)
    return paths


def load_steps(path):
    """One ``steps_<split>.npz`` as a dict of numpy arrays."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def guid_folds(guids, n_folds=5, seed=0):
    """Fold number per segment, a deterministic hash of (seed, GUID): the same for one GUID's segments."""
    return np.array([int.from_bytes(hashlib.blake2b(f"{seed}|{g}".encode(), digest_size=8).digest(), "big")
                     % n_folds for g in guids])


def stitch(d):
    """Join the segments of each recording on one timeline (segments tile at multiples of 120 s).

    Returns (rec, seg). rec: (G, L) arrays ``fhr, fhr_sd, fhr_ok, ua, ua_ok`` padded with invalid steps,
    ``length`` (steps up to the last segment's end), ``t0`` (time of step 0 in seconds, negative = before
    delivery) and per-recording ``guid, subgroup, cs, bg``. seg: per segment, its recording ``g`` and its
    first step ``pos`` on that timeline."""
    guid, first_seg, g = np.unique(d["guid"], return_index=True, return_inverse=True)
    start = np.full(len(guid), np.inf)
    np.minimum.at(start, g, d["epoch"])
    pos = np.round((d["epoch"] - start[g]) / 4).astype(np.int64)
    length = np.zeros(len(guid), np.int64)
    np.maximum.at(length, g, pos + T)
    rec = {k: np.zeros((len(guid), int(length.max())), np.float32) for k in ("fhr", "fhr_sd", "fhr_ok", "ua", "ua_ok")}
    cols = pos[:, None] + np.arange(T)[None]
    for k in rec:
        rec[k][g[:, None], cols] = d[k]
    rec.update(length=length, t0=(start + 60).astype(np.float32), guid=guid,
               **{k: d[k][first_seg] for k in ("subgroup", "cs", "bg")})
    return rec, {"g": g, "pos": pos}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-dir", default="tmp/dataset/split_last4h_gap10")
    ap.add_argument("--out", default="tmp/coupling_irf")
    args = ap.parse_args()
    t0 = time.time()
    for p in build_step_cache(args.split_dir, args.out):
        s = load_steps(p)
        print(f"{p}: N={len(s['guid'])} guids={len(set(s['guid']))} {p.stat().st_size / 1e6:.1f} MB")
    print(f"{time.time() - t0:.0f} s")
