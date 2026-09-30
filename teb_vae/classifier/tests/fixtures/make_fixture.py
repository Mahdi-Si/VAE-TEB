r"""Generate the classifier's tiny k-fold fixture tree (SPEC §15).

Layout, under ``--out``::

    fold_{1,2,3}/{train,val,test}/{healthy_bg_no_cs,healthy_no_bg_no_cs,acidosis_no_cs,hie_cs}.hdf5
    stats.hdf5            # normalisation statistics over fold_1's shards (real calculator)

**Geometry matches the trf_cfs test fixtures.** The shards are causal files written by the
production writer through ``scripts/make_tiny_shard.py`` (``create_causal_file`` +
``append_samples_batch``), from the real causal bank run once over the committed raw segments, with
the same leg alignment (``envelope``) and phase operator (``ratio_power_v0``) as the committed
``tiny_shard_causal.hdf5`` and the ``causal_cohort`` shards the trf_cfs conftest trains its tiny
checkpoint on: ``fhr_st`` 36, ``fhr_ph`` 66, ``up_st`` 36, ``up_ph`` 15 channels, 330 stored steps
(T = 300 after the 1-minute trim), and the real ``causal_warmup_steps`` / ``causal_delay_s`` attrs.

**Cohort.** 60 GUIDs (15 per subgroup), 3-6 segments each on a 660-s grid with gaps, the last one
ending within the final hour. Fold k: GUID ``i`` goes to test if ``i % 5 == k-1``, val if
``i % 5 == k % 5``, train otherwise -- except ``healthy_no_bg_no_cs-14``, which is in every fold's
test split (the shared test GUID). TLO is NaN for about 20% of GUIDs and second-stage onset for about
half; ``hie_cs-00`` carries the second-stage sentinel (onset at delivery) and ``healthy_bg_no_cs-01``
carries one duplicate epoch plus one segment per other exclusion reason (outside the window,
crossing delivery, low valid fraction).

**Planted signal.** ST channels (``fhr_st``, ``up_st``) of positive-GUID segments ending within
1 h of delivery are multiplied by ``exp(PLANT_LOG_SHIFT[class])``. Coefficients are the real ones,
coarse-grained to 30-step blocks so lzf keeps the tree small (~25 MB); none of it is clinical.

**Covariates** (§7.3, class-independent, their own RNG so the shards are unchanged).
``static_covariates.csv``: ``guid, parity`` (categorical ``0 | 1 | 2+``, ~15% empty, a few GUIDs absent; odd
GUIDs written in their ``guid_norm`` spelling). ``timed_covariates.csv``: long ``guid, time_s, variable, value`` of
``temp_c`` for ~75% of GUIDs, every 0.75-2.5 h from 3 h before the first segment (gaps beyond ``max_age_h``), some
series stopping 3 h early, and one :data:`FUTURE_TEMP_C` reading a few seconds after each series' last segment end,
which a causal join never uses.

    python -m teb_vae.classifier.tests.fixtures.make_fixture --out output/classifier_fixture
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_FOLDS = 3
SPLITS = ("train", "val", "test")
SEQ_LEN = 330
STRIDE_S = 660.0
GUIDS_PER_SUBGROUP = 15
#: (class code, cs, bg) per subgroup shard, as SPEC §2.3.
SUBGROUPS: Dict[str, tuple] = {
    "healthy_bg_no_cs": (1, 0, 1),
    "healthy_no_bg_no_cs": (1, 0, 0),
    "acidosis_no_cs": (2, 0, 1),
    "hie_cs": (3, 1, 1),
}
PLANT_LOG_SHIFT = {1: 0.0, 2: 0.75, 3: 1.25}
PLANT_WINDOW_S = 3600.0
SHARED_TEST_GUID = "healthy_no_bg_no_cs-14"
SENTINEL_GUID = "hie_cs-00"
SPECIAL_GUID = "healthy_bg_no_cs-01"
STATS_FILENAME = "stats.hdf5"
STATIC_CSV, TIMED_CSV = "static_covariates.csv", "timed_covariates.csv"
#: The reading after the last segment end: a causal as-of join never returns it.
FUTURE_TEMP_C = 45.0
DEFAULT_OUT = REPO_ROOT / "output" / "classifier_fixture"
_LEAD_END_S = 1260.0  # epoch -> end of the observed window at trim 1.0


def _split_of(guid: str, index: int, fold: int) -> str:
    """The split of GUID ``index`` in ``fold`` (see the module docstring)."""
    if guid == SHARED_TEST_GUID or index % 5 == fold - 1:
        return "test"
    return "val" if index % 5 == fold % 5 else "train"


def _segments(rng: np.random.Generator, guids_per_subgroup: int = GUIDS_PER_SUBGROUP,
              segments: Tuple[int, int] = (3, 7), slots: int = 11) -> List[Dict[str, Any]]:
    """One dict per stored segment, fold-independent: ``guids_per_subgroup`` GUIDs per subgroup, each with
    ``segments`` [lo, hi) segments on the last ``slots + 1`` slots of a 660-s grid."""
    rows: List[Dict[str, Any]] = []
    for subgroup, (code, cs, bg) in SUBGROUPS.items():
        for index in range(guids_per_subgroup):
            guid = f"{subgroup}-{index:02d}"
            n = int(rng.integers(*segments))
            grid = np.sort(np.r_[rng.choice(slots, n - 1, replace=False), slots])
            last = -STRIDE_S * (2 + int(rng.integers(0, 2)))  # ends 60 s or 720 s before delivery
            epochs = list(last - STRIDE_S * (slots - grid))
            weights = [None] * n
            if guid == SPECIAL_GUID:  # duplicate, outside_window, crosses_delivery, low_valid_frac
                epochs += [epochs[0], -69 * STRIDE_S, -STRIDE_S, last - (slots + 1) * STRIDE_S]
                weights += [None, None, None, 0.05]
            # 0.5-12 h: some recordings (about 2 h long) start before labour onset (context.tlo.pre_onset, T-L1)
            onset_before = rng.uniform(0.5, 12) * 3600 if rng.random() > 0.2 else np.nan
            ss_before = rng.uniform(0.2, 1.5) * 3600 if rng.random() > 0.5 else np.nan
            for epoch, weight in zip(epochs, weights):
                late = epoch + _LEAD_END_S >= -PLANT_WINDOW_S
                rows.append({
                    "guid": guid, "index": index, "subgroup": subgroup, "code": code, "cs": cs,
                    "bg": bg, "epoch": epoch, "weight": weight,
                    "tlo": epoch + onset_before,
                    "ss": epoch if guid == SENTINEL_GUID else epoch + ss_before,
                    "source": int(rng.integers(0, 8)),
                    "gain": float(np.exp(0.1 * rng.standard_normal())),
                    "shift": PLANT_LOG_SHIFT[code] if late else 0.0,
                })
    return rows


def _covariates(rows: List[Dict[str, Any]], out: Path, rng: np.random.Generator) -> Dict[str, str]:
    """Write the static and timed covariate CSVs (see the module docstring); return their paths."""
    import pandas as pd

    ends: Dict[str, List[float]] = {}
    for r in rows:
        if r["epoch"] + _LEAD_END_S <= 0:  # not the crosses-delivery segment
            ends.setdefault(r["guid"], []).append(r["epoch"] + _LEAD_END_S)
    static, timed = [], []
    for i, (guid, t) in enumerate(ends.items()):
        name = guid.upper().replace("-", "") if i % 2 else guid
        if rng.random() > 0.05:
            static.append({"guid": name, "parity": rng.choice(["0", "1", "2+"], p=[0.45, 0.35, 0.2])
                           if rng.random() > 0.15 else ""})
        if rng.random() < 0.25:
            continue
        clock, stop = min(t) - 3 * 3600.0, max(t) - (3 * 3600.0 if rng.random() < 0.15 else 0.0)
        while clock <= stop:
            timed.append({"guid": name, "time_s": round(clock), "variable": "temp_c",
                          "value": f"{37.0 + 0.3 * rng.standard_normal():.1f}"})
            clock += rng.uniform(0.75, 2.5) * 3600.0
        timed.append({"guid": name, "time_s": round(max(t) + rng.uniform(5, 50)), "variable": "temp_c",
                      "value": f"{FUTURE_TEMP_C:.1f}"})
    pd.DataFrame(static).to_csv(out / STATIC_CSV, index=False)
    pd.DataFrame(timed).to_csv(out / TIMED_CSV, index=False)
    return {"static_csv": str(out / STATIC_CSV), "timed_csv": str(out / TIMED_CSV)}


def generate(out: Any = DEFAULT_OUT, seed: int = 0, *, n_folds: int = N_FOLDS,
             guids_per_subgroup: int = GUIDS_PER_SUBGROUP, segments: Tuple[int, int] = (3, 7),
             slots: int = 11) -> Dict[str, Any]:
    """Write the fixture tree and its stats file into ``out``; return where they are. The defaults are the tests'
    tiny tree; larger values make a scale rehearsal tree (``n_folds`` <= 5: the split rule is ``index % 5``)."""
    if not 1 <= n_folds <= 5:
        raise ValueError(f"n_folds must be 1-5 (GUID index % 5 picks the test split), got {n_folds}")
    from hdf5_dataset.calculate_dataset_stats import calculate_and_save_dataset_stats
    from scripts.make_tiny_shard import (
        DECIMATION, TRIM_MINUTES, causal_transform, cohort_weight_profile, create_causal_file,
        read_causal_source, read_causal_source_count,
    )

    out = Path(out)
    n_source = read_causal_source_count()
    transformed = causal_transform(
        read_causal_source(n_source, SEQ_LEN * DECIMATION), SEQ_LEN, leg_alignment="envelope"
    )
    # Coarse-grained to 30-step blocks: real magnitudes and warm-up transients, lzf-compressible.
    base = {
        name: block.reshape(n_source, block.shape[1], -1, 30).mean(-1).repeat(30, -1)
        for name, block in transformed["blocks"].items()
    }
    raw = transformed["raw"]
    profile = cohort_weight_profile(SEQ_LEN)
    rows = _segments(np.random.default_rng(seed), guids_per_subgroup, tuple(segments), slots)

    written: Dict[int, List[str]] = {}
    for fold in range(1, n_folds + 1):
        for split in SPLITS:
            for subgroup in SUBGROUPS:
                part = [r for r in rows if r["subgroup"] == subgroup
                        and _split_of(r["guid"], r["index"], fold) == split]
                if not part:
                    continue
                path = out / f"fold_{fold}" / split / f"{subgroup}.hdf5"
                create_causal_file(str(path), transformed, SEQ_LEN)
                for at in range(0, len(part), 2000):  # bounded memory on a scale tree
                    chunk = part[at:at + 2000]
                    src = [r["source"] for r in chunk]
                    scale = np.array([r["gain"] for r in chunk])[:, None, None]
                    plant = np.exp([r["shift"] for r in chunk])[:, None, None]
                    blocks = {
                        name: (block[src] * scale * (plant if name.endswith("_st") else 1.0)).astype("f4")
                        for name, block in base.items()
                    }
                    weight = np.stack([profile if r["weight"] is None else np.full(SEQ_LEN, r["weight"])
                                       for r in chunk]).astype("f4")
                    transformed["pipeline"].append_samples_batch(
                        str(path),
                        fhr_batch=raw["fhr"][src], up_batch=raw["up"][src],
                        fhr_st_batch=blocks["fhr_st"], fhr_ph_batch=blocks["fhr_ph"],
                        up_st_batch=blocks["up_st"], up_ph_batch=blocks["up_ph"],
                        target_batch=(np.array([r["code"] for r in chunk])[:, None] * weight).astype("f4"),
                        weight_batch=weight,
                        guid_batch=[r["guid"] for r in chunk],
                        epoch_batch=np.array([r["epoch"] for r in chunk], dtype="f4"),
                        cs_label_batch=np.array([r["cs"] for r in chunk], dtype="u1"),
                        bg_label_batch=np.array([r["bg"] for r in chunk], dtype="u1"),
                        tlo_batch=np.array([r["tlo"] for r in chunk], dtype="f4"),
                        second_stage_batch=np.array([r["ss"] for r in chunk], dtype="f4"),
                    )
                guids = "\n".join(sorted({r["guid"] for r in part}))
                with h5py.File(path, "a") as handle:
                    handle.attrs["source_guid_digest"] = hashlib.sha256(guids.encode()).hexdigest()
                written.setdefault(fold, []).append(str(path))

    stats_path = out / STATS_FILENAME
    calculate_and_save_dataset_stats(
        written[1], str(stats_path), trim_minutes=TRIM_MINUTES, progress_bar=False,
        device="cpu", plot_histograms=False,
    )
    return {"root": str(out), "stats_path": str(stats_path), "shards": written,
            "n_guids": len({r["guid"] for r in rows}), "n_segments": len(rows),
            **_covariates(rows, out, np.random.default_rng([seed, 1]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="1-5")
    parser.add_argument("--guids-per-subgroup", type=int, default=GUIDS_PER_SUBGROUP)
    parser.add_argument("--segments", type=int, nargs=2, default=(3, 7), metavar=("LO", "HI"),
                        help="segments per GUID, [LO, HI)")
    parser.add_argument("--slots", type=int, default=11, help="660-s grid slots before the last segment")
    args = parser.parse_args()
    info = generate(args.out, args.seed, n_folds=args.folds, guids_per_subgroup=args.guids_per_subgroup,
                    segments=tuple(args.segments), slots=args.slots)
    print(f"wrote {info['n_guids']} GUIDs / {info['n_segments']} segments per fold to "
          f"{info['root']}; stats: {info['stats_path']}")
