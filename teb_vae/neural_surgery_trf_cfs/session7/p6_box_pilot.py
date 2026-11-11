r"""E8.3: pilot fine-tune of rung 7 on UA features in which the masked boxes are valid.

The box events of :mod:`p6_box_events` are masked in the shards, so the stored UA features hold a
bridge in their place. This pilot rebuilds the features of every box segment with the builder's
own chain, with the box runs taken out of the invalid mask, and asks whether rung 7 forecasts
better when it trains on them.

**Arms.** Both start from the rung-7 warm start and use the fixed budget of :mod:`s7_pilot`.

* ``s7ref``: the stored segments.
* ``s7box``: the fit and the scored segments that hold a box event are replaced by their rebuilt
  form. Five fields of such a segment change, and only around its box runs:

  - ``up_st`` and ``up_ph``: the transform of the UA with the raw box in place of the bridge;
  - ``up_st_step_valid`` and ``up_ph_step_valid``: the loader's strict rule on the changed
    invalid mask and its rebuilt contamination $\rho$;
  - ``up_valid``: valid inside the box runs;
  - ``up``: the UA trace that the contraction detector of the event weight reads.

  Every other invalid run of the segment keeps its stored values and flags.

A third table, ``s7ref_onbox``, scores the ``s7ref`` network on the rebuilt scored segments. It
separates the effect of the data at scoring time from the effect of training on it.

**Scoring.** :func:`s7_pilot.evaluate` on the select recordings (split code 2). The targets and
the anchors are those of the stored segments in every arm, because only UA fields change. An
anchor is affected when its 38-step key window holds at least one step of a box run; its timing
class refers to the box run that most recently started before it.

**Summary.** Paired differences per anchor, averaged over the seeds, with recording-level
bootstrap intervals (:func:`s7common.paired_bootstrap`).

Modes: ``features`` builds the cache ``out/phase6/box_features.npz``; ``run`` fine-tunes and
scores one arm with one seed; ``summary`` writes ``out/phase6/box_pilot.json``.

Edit :data:`RUN_ARGS` and run this file from the IDE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot as P  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import p1_common as C  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p1_t4_model import N_LAGS, TRIM, strict_valid_full  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7.p6_box_events import OUT6  # noqa: E402

#: Run-button inputs.
RUN_ARGS: Dict[str, Any] = {"mode": "summary", "arm": "s7box", "seed": 0, "device": "cuda:0", "steps": None,
                            "n_boot": 2000}

#: The rung-7 warm start.
WARM_START = S.RUN_DIR / "lag-attn-trf-cfs-epoch=424-s7init.ckpt"
FEATURES = OUT6 / "box_features.npz"
PILOTS = OUT6 / "pilots"
ARMS = ("s7ref", "s7box")
RAW_TRIM = TRIM * S.RAW_PER_STEP


def build_features(device: str) -> Dict[str, np.ndarray]:
    """Rebuild the UA fields of every box segment with the box runs valid, and cache them.

    Args:
        device: Torch device of the builder chain.

    Returns:
        ``{'cohort', 'row', 'up', 'up_st', 'up_ph', 'up_st_rho', 'up_ph_rho', 'invalid'}`` with
        one entry per box segment, on the untrimmed grid.
    """
    from teb_vae.neural_surgery_trf_cfs.session7.p1_rebuild import Rebuilder, builder_inputs

    rb = Rebuilder(device)
    events = pd.read_parquet(OUT6 / "box_events.parquet")
    out: Dict[str, List[Any]] = {k: [] for k in ("cohort", "row", "up", "up_st", "up_ph", "up_st_rho", "up_ph_rho",
                                                 "invalid")}
    worst = 0.0
    started = time.time()
    for i, ((cohort, row), part) in enumerate(events.groupby(["cohort", "row"], sort=True)):
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            row = int(row)
            x = handle["up_raw"][row].astype(np.float64)
            invalid = (handle["up_quality"][row].astype(np.int64) & S.UP_INVALID_BITS) != 0
            runs = [(int(a), int(b)) for a, b in zip(part["start"], part["stop"])]
            steps = builder_inputs(handle, row, rb.cfg)["steps"]
            if i < 5:
                stored = rb.rebuild(x[None], invalid[None], shifts=[steps])
                worst = max(worst, float(np.abs(stored["up_st"][0] - handle["up_st"][row]).max()),
                            float(np.abs(stored["up"][0] - handle["up"][row]).max()))
        reduced = invalid.copy()
        for a, b in runs:
            reduced[a:b] = False
        shifts = [p for p in steps if not any(a - S.FS <= p <= b + S.FS for a, b in runs)]
        built = rb.rebuild(x[None], reduced[None], shifts=[shifts])
        out["cohort"].append(cohort)
        out["row"].append(row)
        out["invalid"].append(reduced)
        for key in ("up", "up_st", "up_ph", "up_st_rho", "up_ph_rho"):
            out[key].append(built[key][0])
        if i % 50 == 0:
            print(f"rebuilt {i + 1} segments, {time.time() - started:.0f} s", flush=True)
    if worst > 1e-4:
        raise RuntimeError(f"the rebuild does not reproduce the stored fields (max error {worst:.3g})")
    arrays = {key: np.asarray(value) for key, value in out.items()}
    np.savez_compressed(FEATURES, **arrays)
    print(f"wrote {FEATURES} ({len(arrays['row'])} segments; stored fields reproduced to {worst:.2g})")
    return arrays


class BoxStore(P.SegmentStore):
    """A segment store that can serve the box segments in their rebuilt form.

    Attributes:
        rows: ``(cohort, shard row)`` of each item.
        patched: Number of items that were replaced.
        checks: Largest differences found by the self-checks of :meth:`apply`.
    """

    def __init__(self, cfg: dict, guids: Sequence[str], rebuilt: bool) -> None:
        """Load the segments and, when asked, replace the box segments.

        Args:
            cfg: The merged evaluation configuration.
            guids: The recordings to keep.
            rebuilt: Serve the box segments in their rebuilt form.
        """
        import copy

        from train.data_module import GraphDataModule

        cfg = copy.deepcopy(cfg)
        cfg["dataset_config"]["dataloader_config"]["dataset_kwargs"]["allowed_guids"] = list(guids)
        loader = GraphDataModule(cfg).test_dataloader()
        self.collate = loader.collate_fn
        dataset = loader.dataset
        self.items = [dataset[index] for index in range(len(dataset))]
        self.guids = [S.guid_text(self._guid(item)) for item in self.items]
        norm = lambda p: os.path.normcase(os.path.normpath(str(p)))  # noqa: E731
        cohort_of = {norm(path): cohort for cohort, path in S.SHARDS.items()}
        self.rows = [(cohort_of[norm(dataset.paths[fi])], int(si)) for fi, si in dataset.index_map]
        self.patched = 0
        self.checks: Dict[str, float] = {}
        if rebuilt:
            self.apply(dataset)

    def apply(self, dataset: Any) -> None:
        """Replace the UA fields of every box segment by the cached rebuilt ones.

        The first replaced segment is also rebuilt from the STORED fields with the same code
        path, and the result is compared with the loader's item; a difference stops the run.

        Args:
            dataset: The loader's dataset, for its normalisation.
        """
        cache = np.load(FEATURES, allow_pickle=False)
        index = {(str(c), int(r)): i for i, (c, r) in enumerate(zip(cache["cohort"], cache["row"]))}
        for item, key in zip(self.items, self.rows):
            if key not in index:
                continue
            i = index[key]
            if self.patched == 0:
                self._self_check(dataset, item, key)
            fields = self._fields(dataset, cache["up"][i], cache["up_st"][i], cache["up_ph"][i],
                                  cache["up_st_rho"][i], cache["up_ph_rho"][i], cache["invalid"][i])
            for name, value in fields.items():
                item[name] = value.to(item[name].dtype)
            self.patched += 1

    @staticmethod
    def _fields(dataset: Any, up: np.ndarray, up_st: np.ndarray, up_ph: np.ndarray, rho_st: np.ndarray,
                rho_ph: np.ndarray, invalid: np.ndarray) -> Dict[str, torch.Tensor]:
        """Turn untrimmed builder fields into the loader's trimmed, normalised item fields.

        Args:
            dataset: The loader's dataset.
            up: ``(N,)`` UA trace.
            up_st: ``(36, T)``.
            up_ph: ``(10, T)``.
            rho_st: ``(36, T)`` ``uint8``.
            rho_ph: ``(10, T)`` ``uint8``.
            invalid: ``(N,)`` invalid mask.

        Returns:
            The six item fields.
        """
        valid = strict_valid_full(invalid, rho_st, rho_ph)
        st = dataset._normalize_data("up_st", torch.from_numpy(np.ascontiguousarray(up_st[:, TRIM:-TRIM], dtype=np.float32)))
        ph = dataset._normalize_data("up_ph", torch.from_numpy(np.ascontiguousarray(up_ph[:, TRIM:-TRIM], dtype=np.float32)))
        trace = dataset._normalize_data("up", torch.from_numpy(np.ascontiguousarray(up[RAW_TRIM:-RAW_TRIM], dtype=np.float32)))
        return {"up_st": st.T, "up_ph": ph.T, "up": trace,
                "up_valid": torch.from_numpy(~invalid[RAW_TRIM:-RAW_TRIM]),
                "up_st_step_valid": torch.from_numpy(valid[:, :36]),
                "up_ph_step_valid": torch.from_numpy(valid[:, 36:])}

    def _self_check(self, dataset: Any, item: Any, key: Tuple[str, int]) -> None:
        """Compare this module's field construction with the loader's own item on stored data.

        Args:
            dataset: The loader's dataset.
            item: The loader's item of the segment, not yet replaced.
            key: ``(cohort, row)`` of the segment.

        Raises:
            RuntimeError: If a field differs.
        """
        cohort, row = key
        with h5py.File(S.SHARDS[cohort], "r") as handle:
            invalid = (handle["up_quality"][row].astype(np.int64) & S.UP_INVALID_BITS) != 0
            fields = self._fields(dataset, handle["up"][row], handle["up_st"][row], handle["up_ph"][row],
                                  handle["up_st_rho"][row], handle["up_ph_rho"][row], invalid)
        for name, value in fields.items():
            mine, theirs = value.to(torch.float32), item[name].to(torch.float32)
            self.checks[name] = float((mine - theirs).abs().max())
            if self.checks[name] > 1e-4:
                raise RuntimeError(f"field {name} built here differs from the loader's item by {self.checks[name]:.3g}")


def covariates(store: BoxStore) -> pd.DataFrame:
    """Box covariates of every anchor of the box segments of a store.

    Args:
        store: A segment store (rebuilt or not; only its row list is read).

    Returns:
        One row per ``(guid, epoch, anchor)`` with ``affected`` and ``timing``.
    """
    events = pd.read_parquet(OUT6 / "box_events.parquet")
    groups = {key: part for key, part in events.groupby(["cohort", "row"])}
    rows: List[Dict[str, Any]] = []
    steps_total = C.N_STEPS - 2 * TRIM
    for item, key in zip(store.items, store.rows):
        if key not in groups:
            continue
        part = groups[key].sort_values("start")
        ind = np.zeros(C.N_RAW, dtype=bool)
        for a, b in zip(part["start"], part["stop"]):
            ind[int(a):int(b)] = True
        padded = np.concatenate([np.zeros(S.RAW_PER_STEP - 1, dtype=bool), ind])
        hit = padded[: C.N_STEPS * S.RAW_PER_STEP].reshape(C.N_STEPS, S.RAW_PER_STEP).any(axis=1)[TRIM:C.N_STEPS - TRIM]
        cs = np.concatenate([[0], np.cumsum(hit)])
        starts, stops = part["start"].to_numpy(), part["stop"].to_numpy()
        for anchor in range(steps_total):
            n_a = RAW_TRIM + S.RAW_PER_STEP * anchor
            before = np.flatnonzero(starts <= n_a)
            if before.size == 0:
                timing = "before"
            elif n_a < stops[before[-1]]:
                timing = "inside"
            else:
                d = (n_a - stops[before[-1]]) / S.FS
                timing = "0-60" if d < 60 else ("60-152" if d < 152 else "later")
            rows.append({"guid": S.guid_text(store._guid(item)), "epoch": float(item["epoch"]), "anchor": anchor,
                         "affected": bool(cs[anchor + 1] - cs[max(anchor - (N_LAGS - 1), 0)] > 0), "timing": timing,
                         "cohort": key[0]})
    return pd.DataFrame(rows)


def run(arm: str, seed: int, device: str, steps: Optional[int]) -> Dict[str, Any]:
    """Fine-tune one arm with one seed from the rung-7 warm start and score it.

    Args:
        arm: ``'s7ref'`` or ``'s7box'``.
        seed: The seed.
        device: Torch device string.
        steps: Optimizer steps; ``None`` uses the fixed budget.

    Returns:
        The run record.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    PILOTS.mkdir(parents=True, exist_ok=True)
    stem = PILOTS / f"{arm}_s{seed}"
    if stem.with_suffix(".json").exists():
        print(f"{stem.name}: exists, skipped")
        return json.loads(stem.with_suffix(".json").read_text())
    if not FEATURES.exists():
        build_features(device)
    started = time.time()
    dev, cfg, task = S.load_model(checkpoint=WARM_START, device=device)
    model = task.orig_model
    rebuilt = arm == "s7box"
    fit = BoxStore(cfg, P.guids_with_code([S.FIT_CODE]), rebuilt)
    scored = BoxStore(cfg, P.guids_with_code([S.SELECT_CODE]), rebuilt)
    torch.manual_seed(seed)
    budget = dict(P.BUDGET)
    if steps is not None:
        budget["steps"] = int(steps)
    history = P.fine_tune(task, fit, dev, seed, budget,
                          on_step=lambda step, row: print(
                              f"step {step}: loss {row.get('total_loss', float('nan')):.1f} "
                              f"kl {row.get('source_conditioned_kl_raw', float('nan')):.3f} "
                              f"gap {row.get('pred_gap', float('nan')):.2f}", flush=True) if step % 25 == 0 else None)
    nu = getattr(model, "forecast_student_nu", None)
    table = P.evaluate(task, scored, dev, nu=nu)
    table.to_parquet(stem.with_suffix(".parquet"))
    if not rebuilt:
        other = BoxStore(cfg, P.guids_with_code([S.SELECT_CODE]), True)
        P.evaluate(task, other, dev, nu=nu).to_parquet(PILOTS / f"s7ref_onbox_s{seed}.parquet")
        covariates(other).to_parquet(PILOTS / "covariates.parquet")
    tail = history[-20:]
    record = {"arm": arm, "seed": seed, "budget": budget, "n_fit_segments": len(fit), "n_scored_segments": len(scored),
              "fit_segments_replaced": fit.patched, "scored_segments_replaced": scored.patched,
              "field_checks": {**fit.checks}, "seconds": time.time() - started, "nu": nu,
              "start_checkpoint": str(WARM_START),
              "train_tail": {key: float(np.mean([row[key] for row in tail])) for key in (tail[0] if tail else {})},
              "calibration": P.calibration(table),
              "means": {key: float(table[key].mean()) for key in ("nll_full", "nll_base", "gain", "kl", "kl_null", "g")}}
    stem.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(f"{stem.name}: {record['means']} | replaced fit {fit.patched}, scored {scored.patched} | "
          f"{record['seconds']:.0f} s")
    return record


def summary(n_boot: int) -> Dict[str, Any]:
    """Compare the arms on the select recordings and write the summary.

    Args:
        n_boot: Bootstrap resamples.

    Returns:
        The summary dictionary.
    """
    keys = ["guid", "epoch", "anchor"]
    metrics = ["nll_full", "nll_base", "gain", "kl", "g", "sse_st_full", "sse_ph_full"]

    def load(name: str) -> pd.DataFrame:
        files = sorted(PILOTS.glob(f"{name}_s[0-9]*.parquet"))
        if not files:
            raise FileNotFoundError(f"no table of arm {name} in {PILOTS}")
        frames = [pd.read_parquet(f)[keys + metrics] for f in files]
        mean = pd.concat(frames).groupby(keys, as_index=False)[metrics].mean()
        mean.attrs["seeds"] = len(files)
        return mean

    tables = {name: load(name) for name in ("s7ref", "s7box", "s7ref_onbox")}
    cov = pd.read_parquet(PILOTS / "covariates.parquet")
    base = tables["s7ref"].merge(cov, on=keys, how="left")
    base["affected"] = base["affected"].eq(True)
    base["timing"] = base["timing"].fillna("no_box")
    table = base
    for name in ("s7box", "s7ref_onbox"):
        table = table.merge(tables[name], on=keys, suffixes=("", f"@{name}"))
    scopes = {"all": np.ones(len(table), dtype=bool), "affected": table["affected"].to_numpy(),
              "not_affected": ~table["affected"].to_numpy()}
    for timing in ("inside", "0-60", "60-152", "later"):
        scopes[f"affected/{timing}"] = (table["affected"] & (table["timing"] == timing)).to_numpy()
    scopes["affected/after_0-152"] = (table["affected"] & table["timing"].isin(["0-60", "60-152"])).to_numpy()
    cohorts = S.cohort_of_guids()
    unseen = table["guid"].map(lambda g: S.MEMBERSHIP[cohorts[g]] == "unseen").to_numpy()
    scopes["affected/unseen_cohorts"] = table["affected"].to_numpy() & unseen
    out: Dict[str, Any] = {"seeds": {name: t.attrs["seeds"] for name, t in tables.items()}, "scopes": {}}
    for scope, keep in scopes.items():
        part = table[keep]
        if part.empty:
            continue
        entry: Dict[str, Any] = {"n_anchors": int(len(part)), "n_rec": int(part["guid"].nunique())}
        for metric in ("nll_full", "gain", "g", "kl"):
            entry[f"{metric}:s7ref"] = float(part[metric].mean())
            for name in ("s7box", "s7ref_onbox"):
                entry[f"{metric}:{name}"] = float(part[f"{metric}@{name}"].mean())
                entry[f"d_{metric}:{name}-s7ref"] = S.paired_bootstrap(
                    part["guid"].to_numpy(), (part[f"{metric}@{name}"] - part[metric]).to_numpy(), n_resamples=n_boot)
            entry[f"d_{metric}:s7box-s7ref_onbox"] = S.paired_bootstrap(
                part["guid"].to_numpy(), (part[f"{metric}@s7box"] - part[f"{metric}@s7ref_onbox"]).to_numpy(),
                n_resamples=n_boot)
        for block in ("st", "ph"):
            ref = float(part[f"sse_{block}_full"].sum())
            entry[f"mse_ratio_{block}:s7box/s7ref"] = float(part[f"sse_{block}_full@s7box"].sum() / max(ref, 1e-12))
        out["scopes"][scope] = entry
    (OUT6 / "box_pilot.json").write_text(json.dumps(out, indent=1))
    for scope, entry in out["scopes"].items():
        print(f"{scope}: n={entry['n_anchors']} rec={entry['n_rec']} | G ref {entry['gain:s7ref']:.2f} "
              f"box {entry['gain:s7box']:.2f} ref-on-box {entry['gain:s7ref_onbox']:.2f} | gate ref "
              f"{entry['g:s7ref']:.2f} box {entry['g:s7box']:.2f}")
        for key in ("d_nll_full:s7box-s7ref", "d_gain:s7box-s7ref", "d_nll_full:s7ref_onbox-s7ref",
                    "d_gain:s7ref_onbox-s7ref", "d_nll_full:s7box-s7ref_onbox"):
            print(f"   {key}: {S.format_ci(entry[key], 2)}")
        print(f"   MSE ratio box/ref: scattering {entry['mse_ratio_st:s7box/s7ref']:.4f}, "
              f"phase {entry['mse_ratio_ph:s7box/s7ref']:.4f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("features", "run", "summary"), default=RUN_ARGS["mode"])
    parser.add_argument("--arm", default=RUN_ARGS["arm"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    parser.add_argument("--device", default=RUN_ARGS["device"])
    parser.add_argument("--steps", type=int, default=RUN_ARGS["steps"])
    parser.add_argument("--n-boot", type=int, default=RUN_ARGS["n_boot"])
    args = parser.parse_args()
    if args.mode == "features":
        build_features(args.device)
    elif args.mode == "run":
        run(args.arm, args.seed, args.device, args.steps)
    else:
        summary(args.n_boot)
