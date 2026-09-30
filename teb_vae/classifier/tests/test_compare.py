"""Model comparison (SPEC §11.8, §11.12 block Q): the paired GUID bootstrap on known answers, the cohort-digest pairing
and refusal, the Q3 "only differing keys" table, and a ``compare`` CLI smoke on two synthetic finished runs (no
training)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from teb_vae.classifier import metrics as M
from teb_vae.classifier import report as R
from teb_vae.classifier.tests.test_eval_analyses import make_run

BOOT = dict(resamples=200, seed=0)


def _pop(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """``n`` pooled test GUIDs, alternately adverse and healthy, two per patient, a weakly informative score."""
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    return pd.DataFrame({"fold": 1, "guid": [f"g{i}" for i in range(n)], "patient": [f"p{i // 2}" for i in range(n)],
                         "y": y, "score": rng.normal(0, 1, n) + 0.5 * y})


def test_paired_bootstrap_known_answers():
    a = _pop()
    (same,) = M.paired_deltas(a, a.copy(), ["auroc"], mn=5, **BOOT)
    assert same["subgroup"] is None and same["metric"] == "delta_auroc"
    assert same["value"] == same["ci_lo"] == same["ci_hi"] == 0.0 and same["p_value"] == 1.0  # identical runs
    assert same["run_value"] == same["reference_value"] == pytest.approx(roc_auc_score(a["y"], a["score"]))
    (better,) = M.paired_deltas(a, a.assign(score=a["y"].astype(float)), ["auroc"], mn=5, **BOOT)
    assert better["run_value"] == 1.0 and better["value"] > 0 and better["ci_lo"] > 0 and better["p_value"] < 0.05
    assert better["value"] == pytest.approx(1.0 - roc_auc_score(a["y"], a["score"]))

    # rates: the run alarms on every adverse GUID and no healthy one, the reference never alarms
    sens, spec = M.paired_deltas(a.assign(fire=False), a.assign(fire=a["y"] == 1), ["sens", "spec"], mn=5, **BOOT)
    assert (sens["metric"], sens["value"], sens["ci_lo"], sens["ci_hi"]) == ("delta_sens", 1.0, 1.0, 1.0)
    assert (spec["metric"], spec["value"], spec["ci_lo"], spec["ci_hi"]) == ("delta_spec", 0.0, 0.0, 0.0)


def test_paired_deltas_per_cell_population_and_power_guard():
    a = _pop()
    cells = pd.DataFrame({"subgroup": ["half", "half", "class"], "subgroup_value": ["first", "second", "healthy"],
                          "population": ["mixed", "mixed", "healthy_only"]})

    def member(f: pd.DataFrame) -> np.ndarray:
        i = np.arange(len(f))
        return np.stack([i < 30, i >= 30, f["y"].to_numpy() == 0])

    b = a.assign(fire=a["score"] > 0.4)
    recs = M.paired_deltas(a.assign(fire=a["score"] > 0.0), b, ["sens", "spec"], mn=5, cells=cells, member=member,
                           **BOOT)
    got = [(r["subgroup"], r["subgroup_value"], r["metric"]) for r in recs]
    assert got == [(None, None, "delta_sens"), (None, None, "delta_spec"), ("half", "first", "delta_sens"),
                   ("half", "first", "delta_spec"), ("half", "second", "delta_sens"), ("half", "second", "delta_spec"),
                   ("class", "healthy", "delta_spec")]  # a healthy-only member reports specificity only (§11.6)
    (au,) = M.paired_deltas(a, b, ["auroc"], mn=100, **BOOT)  # 30 GUIDs per class < 100: underpowered
    assert au["underpowered"] and np.isnan(au["value"]) and np.isnan(au["p_value"]) and np.isfinite(au["value_raw"])


def _brute_delong(y: np.ndarray, S: np.ndarray):
    """AUROCs and covariance from explicit placement values (O(m n)): psi = 1 if x > y, 1/2 on ties."""
    X, Y = S[:, y == 1], S[:, y == 0]
    psi = (X[:, :, None] > Y[:, None, :]) + 0.5 * (X[:, :, None] == Y[:, None, :])  # (k, m, n)
    v10, v01 = psi.mean(2), psi.mean(1)
    return v10.mean(1), np.cov(v10) / X.shape[1] + np.cov(v01) / Y.shape[1]


def test_delong_known_answers_and_brute_force():
    # hand: X = (0.9, 0.4), Y = (0.5, 0.1): V10 = (1, 1/2), V01 = (1/2, 1), AUC 3/4, Var = 0.125/2 + 0.125/2
    auc, cov = M.delong([1, 1, 0, 0], [0.9, 0.4, 0.5, 0.1])
    assert auc[0] == 0.75 and cov[0, 0] == pytest.approx(0.125)
    rng = np.random.default_rng(3)
    y = (np.arange(80) % 3 == 0).astype(int)
    S = np.round(rng.normal(0, 1, (2, 80)) + y, 1)  # rounded: plenty of ties
    auc, cov = M.delong(y, *S)
    b_auc, b_cov = _brute_delong(y, S)
    np.testing.assert_allclose(auc, b_auc, atol=1e-12)
    np.testing.assert_allclose(cov, b_cov, atol=1e-12)
    assert auc[0] == pytest.approx(roc_auc_score(y, S[0]))
    z = (auc[1] - auc[0]) / np.sqrt(b_cov[0, 0] + b_cov[1, 1] - 2 * b_cov[0, 1])
    assert M.delong_p(y, *S) == pytest.approx(2 * norm.sf(abs(z)))
    assert M.delong_p(y, S[0], S[0]) == 1.0 and np.isnan(M.delong_p([1, 0, 0], S[0, :3], S[1, :3]))


def test_nadeau_bengio_known_answer():
    # d = (0.1, 0, 0.2): mean 0.1, var 0.01, k 3 -> t = 0.1 / sqrt((1/3 + 1/2) 0.01); df 2: sf(t) = (1 - t/sqrt(t^2+2))/2
    t = 0.1 / np.sqrt((1 / 3 + 1 / 2) * 0.01)
    assert M.nadeau_bengio_p([0.1, 0.0, 0.2]) == pytest.approx(1 - t / np.sqrt(t ** 2 + 2))
    assert M.nadeau_bengio_p([0.0, 0.0]) == 1.0 and M.nadeau_bengio_p([0.1, 0.1]) == 0.0
    assert np.isnan(M.nadeau_bengio_p([0.1, np.nan]))


def _cohort(root, name: str, g: pd.DataFrame):
    (root / name / "cohort").mkdir(parents=True)
    g.to_parquet(root / name / "cohort" / "guids.parquet", index=False)
    return root / name


def test_cohort_digest_pairs_ablations_and_compare_refuses_other_cohorts(tmp_path):
    g = pd.DataFrame({"fold": [1, 1, 1, 2, 2, 2], "split": ["test", "val", "train", "test", "val", "train"],
                      "guid": ["a", "b", "c", "b", "c", "a"], "y": [1, 0, 1, 0, 1, 1], "excluded": False,
                      "label_weight": 1.0})
    ref = _cohort(tmp_path, "ref", g.assign(patient=g["guid"]))
    same = [_cohort(tmp_path, "no_patient_map", g),  # no patient column: the GUID is the cluster
            _cohort(tmp_path, "three_class", g.assign(patient=g["guid"], y=g["y"] * np.array([1, 1, 2, 1, 2, 2]))),
            _cohort(tmp_path, "weights", g.assign(patient=g["guid"], label_weight=0.3, cs=True)),
            _cohort(tmp_path, "excluded_row", pd.concat([g, g.iloc[:1].assign(guid="z", excluded=True)])
                    .assign(patient=lambda f: f["guid"]))]
    assert {M.cohort_digest(p) for p in same} == {M.cohort_digest(ref)}
    other = _cohort(tmp_path, "other_folds", g.assign(patient=g["guid"], split=g["split"].iloc[::-1].to_numpy()))
    assert M.cohort_digest(other) != M.cohort_digest(ref)
    with pytest.raises(ValueError, match="cohort digests differ"):
        M.compare_runs([ref, other])
    with pytest.raises(ValueError, match="2 to 6 runs"):
        M.compare_runs([ref])


def test_comparison_md_lists_only_the_differing_keys(tmp_path):
    from teb_vae.classifier.config import load

    c = M.classifier_cfg(load())
    c2 = copy.deepcopy(c)
    c2["labels"]["strategy"] = "propagate"
    head = {"model|42|guid|threshold_free|auroc": {"test": 0.9, "test_ci": [0.8, 0.95], "fold_mean": 0.88,
                                                   "fold_sd": 0.02}}
    runs = [{"name": n, "path": tmp_path / n, "c": cfg, "pm": ("model", "42"), "head": head}
            for n, cfg in (("a", c), ("b", c2))]
    md = R.comparison_md(runs, pd.DataFrame(), tmp_path, pid="np30", digest="abc").read_text()
    lines = md.splitlines()
    assert next(x for x in lines if x.startswith("| run |")).startswith("| run | labels.strategy | model | AUROC fold")
    row = next(x for x in lines if x.startswith("| b |"))
    assert row.startswith("| b | propagate | model (seed 42) | 0.880 ± 0.020 | 0.900 [0.800, 0.950] |")
    ref = next(x for x in lines if x.startswith("| a (reference) |"))
    assert ref.startswith(f"| a (reference) | {c['labels']['strategy']} |")


def test_compare_cli_smoke(tmp_path):
    """Two finished synthetic runs on one cohort, differing in ``labels.strategy``: ``compare`` writes the table, every
    figure and ``comparison.md`` under ``--out`` and nothing in the runs; a run compared with itself has Δ = 0."""
    from teb_vae.classifier import run as RUN

    runs = []
    for name, seed, strategy in (("a", 0, "horizon_decay"), ("b", 1, "propagate")):
        run, cfg = make_run(tmp_path / name, folds=(1, 2), resamples=5, seed=seed, overrides=[
            "classifier.eval.figure_formats=[png]", f"classifier.labels.strategy={strategy}"])
        assert M.evaluate(run, cfg) == 0
        runs.append(run)
    before = {p: p.stat().st_mtime_ns for r in runs for p in r.rglob("*")}
    out = tmp_path / "compare"
    assert RUN._cli(["compare", "--runs", *map(str, runs), "--out", str(out)]) == 0
    assert {p: p.stat().st_mtime_ns for r in runs for p in r.rglob("*")} == before  # compare only reads the runs

    q = pd.read_parquet(out / "comparison.parquet")
    assert set(q["run"]) == {str(runs[1])} and set(q["reference"]) == {str(runs[0])}
    whole = q[q["subgroup"].isna()]
    assert {"delta_auroc", "delta_sens", "delta_spec"} <= set(whole["metric"]) and whole["value"].notna().any()
    assert set(whole["policy_id"].dropna()) == {p.id for p in cfg.classifier.eval.thresholds}
    assert q["subgroup"].notna().any() and q["p_holm"].notna().any()
    au = whole[whole["metric"] == "delta_auroc"].iloc[0]
    assert np.isfinite(au["p_delong"]) and np.isfinite(au["p_nb"]) and 0 <= au["p_delong"] <= 1 and 0 <= au["p_nb"] <= 1
    assert q.loc[q["metric"] != "delta_auroc", ["p_delong", "p_nb"]].isna().all(axis=None)
    stems = [R.COMPARE_ROC, R.COMPARE_FOREST,
             *(R.COMPARE_METRIC_TYPES.format(axis=a) for a in cfg.classifier.eval.time_axes)]
    assert [s for s in stems if not (out / "figures" / f"{s}.png").is_file()] == []
    md = (out / R.COMPARISON_MD).read_text()
    assert "| labels.strategy |" in md and "| propagate |" in md and "delta_auroc" in md and "| DeLong p | NB p |" in md

    self_q = M.compare_runs([runs[0], runs[0]], policy="np30")
    assert set(self_q["policy_id"].dropna()) == {"np30"}
    assert (self_q["value"].dropna() == 0).all() and (self_q["value_raw"].dropna() == 0).all()
    self_au = self_q[(self_q["metric"] == "delta_auroc") & self_q["subgroup"].isna()].iloc[0]
    assert (self_au["p_delong"], self_au["p_nb"]) == (1.0, 1.0)  # identical scores: zero difference, zero variance
    with pytest.raises(ValueError, match="not a GUID-level policy"):
        M.compare_runs(runs, policy="nope")
