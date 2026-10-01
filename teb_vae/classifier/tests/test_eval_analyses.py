"""P2 evaluation pipeline: the analysis registry end to end, T-E6 and T-E7 (SPEC §11.12-§11.14, §15).

:func:`make_run` fabricates a CONTRACT-shaped run directory (config, cohort tables, predictions,
thresholds, provenance) with a planted signal; ``test_report`` and ``test_verify`` reuse it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml

from teb_vae.classifier import metrics as M

CLASS = {0: ("healthy", 1, "healthy_bg_no_cs"), 1: ("acidosis", 2, "acidosis_no_cs"), 3: ("hie", 3, "hie_cs")}


def _split_of(i: int, fold: int) -> str:
    if i == 0:  # the shared test GUID: in every fold's test split
        return "test"
    return "test" if i % 5 == fold - 1 else "val" if i % 5 == fold % 5 else "train"


def make_run(root: Path, *, folds: Sequence[int] = (1, 2, 3), n_guids: int = 100, resamples: int = 20,
             overrides: Sequence[str] = (), seed: int = 0) -> Tuple[Path, Any]:
    """A run dir with ``config.resolved.yaml``, ``manifest.json``, ``cohort/*`` and ``predictions/*``.

    GUID ``i`` is healthy for even i, acidosis for i % 4 == 1, HIE otherwise; 8-12 segments on a
    660-s grid ending in the last 20 min. Positives' segment logits are shifted up (more so in the
    last hour), so AUROC is high; the shortcut baseline scores GUID length with a weak signal.
    """
    from teb_vae.classifier.config import digest, load
    from teb_vae.classifier.thresholds import select_thresholds

    cfg = load(overrides=[f"classifier.run.folds=[{','.join(map(str, folds))}]",
                          f"classifier.eval.bootstrap.resamples={resamples}", *overrides])
    run, rng = Path(root) / "run", np.random.default_rng(seed)
    (run / "predictions").mkdir(parents=True)
    (run / "cohort").mkdir()
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))

    segs = []
    for i in range(n_guids):
        name, code, sub = CLASS[i % 4 if i % 2 else 0]
        n, last = 8 + i % 5, -rng.uniform(60, 1200)
        t_end = last - 660.0 * np.arange(n)[::-1]
        y = int(code > 1)
        segs.append(pd.DataFrame({
            "guid": f"g{i:03d}", "seg_pos": np.arange(n), "slot": np.arange(n), "t_end_s": t_end,
            "epoch_s": t_end - 1260.0, "class_code": code, "clinical_class": name, "source_file": sub, "subgroup": sub,
            "y": y, "cs": sub.endswith("_cs") and "no_cs" not in sub, "bg": True, "has_tlo": i % 3 > 0, "has_ss": i % 2 == 0,
            "tlo_end_s": np.where(i % 3 > 0, t_end + 6 * 3600.0, np.nan), "ss_rel_s": np.nan, "stage": "unknown",
            "valid_frac": 1.0, "n_valid_steps": 300,
            "base": rng.normal(0, 1, n) + rng.normal(0, 0.7) + 1.5 * y * (1 + (t_end > -3600)) / 1.25,
        }))
    base = pd.concat(segs, ignore_index=True)
    base["hours_to_delivery"] = -base["t_end_s"] / 3600
    base["shared_test"] = base["guid"] == "g000"
    idx = base["guid"].str[1:].astype(int)

    cseg, cgd, pseg, pgd, thr = [], [], [], [], {}
    policies = [p.model_dump() for p in cfg.classifier.eval.thresholds]
    for fold in folds:
        split = idx.map(lambda i: _split_of(i, fold))
        s = base.assign(fold=fold, split=split, in_eval_window=True, label_weight=1.0)
        cseg.append(s)
        for m, sd in (("probe", "42"), ("shortcut", "na")):
            p = s[s["split"] != "train"].assign(model_id=m, seed=sd, run_id="run")
            if m == "probe":
                p["logit_seg"] = p["base"] + rng.normal(0, 0.3, len(p))
                p["logit_online"] = p.groupby(["split", "guid"])["logit_seg"].transform(lambda v: v.expanding().mean())
            else:
                p["logit_seg"] = p["logit_online"] = np.nan
            p["logit_seg_cal"], p["logit_online_cal"] = p["logit_seg"], p["logit_online"]
            p["tlo_end_h"], p["ss_rel_h"] = p["tlo_end_s"] / 3600, np.nan
            g = p.groupby(["split", "guid"], as_index=False).agg(
                **{c: (c, "first") for c in ("run_id", "model_id", "seed", "fold", "class_code", "y", "cs", "bg",
                                             "clinical_class", "subgroup", "has_tlo", "has_ss", "shared_test")},
                n_segments=("seg_pos", "size"), first_epoch_s=("epoch_s", "min"), last_t_end_s=("t_end_s", "max"),
                score_final=("logit_online", "last"))
            if m == "shortcut":
                g["score_final"] = 0.2 * g["n_segments"] + 0.8 * g["y"] + rng.normal(0, 1, len(g))
            g["score_final_cal"] = g["score_final"]
            pseg.append(p.drop(columns=["base", "source_file", "tlo_end_s", "ss_rel_s", "slot"]).assign(slot=p["slot"]))
            pgd.append(g)
            vs, vg = p[p["split"] == "val"], g[g["split"] == "val"]
            L = cfg.classifier.eval.exclude_last_min
            if m == "probe":
                thr[f"{m}|{sd}|{fold}"] = select_thresholds(
                    vs, vg, policies, bin_h=0.5, exclude_last_min=L,
                    staleness_h=cfg.classifier.eval.snapshot_max_staleness_h)
            else:
                end = [q | {"basis": "guid_final"} for q in policies if q["at"] == "end"]
                t = select_thresholds(vs, vg, end, bin_h=0.5, exclude_last_min=L)
                t["guid"].update({q["id"]: {"skipped": "guid-level model"} for q in policies if q["at"] != "end"})
                thr[f"{m}|{sd}|{fold}"] = t

    seg = pd.concat(cseg, ignore_index=True).drop(columns=["base", "subgroup", "has_tlo", "has_ss", "n_valid_steps", "shared_test"])
    seg["excluded"], seg["exclusion_reason"] = False, ""
    seg.loc[seg.index[::97], ["excluded", "exclusion_reason"]] = [True, "low_valid_frac"]
    seg["ds_index"], seg["guid_norm"] = seg.index, seg["guid"].str.upper()
    gd = (pd.concat(cseg).groupby(["fold", "split", "guid"], as_index=False)
          .agg(**{c: (c, "first") for c in ("source_file", "class_code", "clinical_class", "y", "cs", "bg", "has_tlo",
                                            "has_ss", "shared_test")},
               n_segments=("seg_pos", "size"), first_epoch_s=("epoch_s", "min"), last_t_end_s=("t_end_s", "max")))
    gd["excluded"], gd["exclusion_reason"], gd["guid_norm"] = False, "", gd["guid"].str.upper()
    seg.to_parquet(run / "cohort" / "segments.parquet", index=False)
    gd.to_parquet(run / "cohort" / "guids.parquet", index=False)
    pd.concat([f.assign(level=lvl, status=f["exclusion_reason"].replace("", "retained"))
               .groupby(["fold", "split", "class_code", "source_file", "level", "status"]).size().rename("n").reset_index()
               for f, lvl in ((gd, "guid"), (seg, "segment"))]).to_csv(run / "cohort" / "fold_summary.csv", index=False)
    (run / "cohort" / "exposure.json").write_text(json.dumps({"applicable": False, "note": "hdf5 source"}))
    (run / "cohort" / "confound.json").write_text(json.dumps(
        {str(f): {"has_tlo": {"missing_rate_by_class": {}, "delta": 0.01, "flagged": False}} for f in folds}))

    pd.concat(pseg, ignore_index=True).to_parquet(run / "predictions" / "segments.parquet", index=False)
    pd.concat(pgd, ignore_index=True).to_parquet(run / "predictions" / "guids.parquet", index=False)
    (run / "predictions" / "thresholds.json").write_text(json.dumps(thr))
    (run / "predictions" / "provenance.json").write_text(json.dumps(
        {"config_digest": digest(cfg), "run_id": "run", "written_at": "now", "units": [], "missing_units": []}))
    (run / "manifest.json").write_text(json.dumps(
        {"config_digest": digest(cfg), "software": {"revision": "deadbeef"}, "cohort": {"sentinel_guids": []}}))
    return run, cfg


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    run, cfg = make_run(tmp_path_factory.mktemp("eval"))
    code = M.evaluate(run, cfg)
    return run, cfg, code, json.loads((run / "evaluation" / "summary.json").read_text())


def test_fail_soft_step_is_recorded(tmp_path, monkeypatch):
    """T-E7: one analysis raising -> n_failed = 1, the others complete, exit 1, traceback kept."""
    run, cfg = make_run(tmp_path, folds=(1, 2), resamples=5)

    def boom(ctx, **_):
        raise RuntimeError("planted failure")

    monkeypatch.setitem(M.ANALYSES, "R9", boom)
    assert M.evaluate(run, cfg, skip=["C"]) == 1
    summary = _partial_summary(run)
    assert summary["n_failed"] == 1 and summary["failed"] == ["R9"] and summary["exit_code"] == 1
    failed = next(s for s in summary["steps"] if s["name"] == "R9")
    assert "planted failure" in failed["traceback"] and "Traceback" in failed["traceback"]
    assert {s["name"] for s in summary["steps"] if s["ok"]} == set(M.ANALYSES) - {"C", "R9"} | {"tables"}
    with pytest.raises(ValueError, match="unknown analyses"):
        M.evaluate(run, cfg, only=["nope"])


def _partial_summary(run: Path) -> dict:
    (part,) = (run / "evaluation").glob("partial_*")
    return json.loads((part / "summary.json").read_text())


def test_provenance_digest_mismatch_refuses(tmp_path):
    run, cfg = make_run(tmp_path, folds=(1, 2))
    prov = json.loads((run / "predictions" / "provenance.json").read_text())
    (run / "predictions" / "provenance.json").write_text(json.dumps(prov | {"config_digest": "0" * 64}))
    with pytest.raises(ValueError, match="config digest"):
        M.evaluate(run, cfg)
