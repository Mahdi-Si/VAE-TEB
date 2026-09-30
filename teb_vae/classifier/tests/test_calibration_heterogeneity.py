"""Blocks K and H, R10, R12, R13 (SPEC §11.7, §11.12): known answers for the prior-shift correction, I², the H3
shared-test dedupe, the K5 PPV re-weighting, K6 and K7, and every KH figure builder on empty tables."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss

from teb_vae.classifier import metrics as M
from teb_vae.classifier.tests.test_eval_analyses import make_run


@pytest.fixture(scope="module")
def kh(tmp_path_factory):
    """A fabricated run (``make_run``: g000 is in every fold's test split) with ``metrics``, R1 and the KH analyses run."""
    run, cfg = make_run(tmp_path_factory.mktemp("kh"), overrides=["classifier.eval.reference_prevalence=0.1"])
    ctx = M.load_context(run, cfg)
    out = run / "evaluation"
    (out / "tables").mkdir(parents=True)
    res = {}
    for name in ("metrics", "R1", "K", "H", "R10", "R12", "K7"):
        res[name] = M.ANALYSES[name](ctx, eval_config=ctx.cfg["eval"], out_dir=out)
        ctx.rows += res[name].pop("rows", [])
        if res[name].get("frame") is not None:
            ctx.frames.append(res[name].pop("frame"))
    return SimpleNamespace(run=run, cfg=cfg, ctx=ctx, out=out, res=res, df=M._rows_frame(ctx))


def test_prior_shift_known_answer():
    """logit p' = logit p - logit pi_val + logit pi_test with each fold's own GUID prevalences."""
    test = pd.DataFrame({"fold": [1, 1, 1, 1, 2, 2], "y": [1, 0, 0, 0, 1, 0],
                         "score_final_cal": [0.0, 1.0, -2.0, 3.0, 0.5, 0.5]})
    val = pd.DataFrame({"fold": [1, 1, 2, 2, 2, 2], "y": [1, 0, 1, 1, 1, 0]})
    got = M.prior_shifted(test, val)
    shift = np.array([logit(0.25) - logit(0.5)] * 4 + [logit(0.5) - logit(0.75)] * 2)
    np.testing.assert_allclose(got, test["score_final_cal"] + shift)
    assert np.isclose(shift[0], -np.log(3.0))


def test_i_squared_known_answer():
    r = M.i_squared([0.7, 0.8, 0.9], [0.05, 0.05, 0.05])  # w = 400: Q = 400 * (0.01 + 0 + 0.01) = 8, df 2
    assert np.isclose(r["q"], 8.0) and np.isclose(r["i2"], 0.75) and np.isclose(r["pooled"], 0.8)
    assert np.isclose(r["q_p"], chi2.sf(8.0, 2)) and np.isclose(r["q_p"], np.exp(-4.0))
    assert M.i_squared([0.8, 0.8], [0.1, 0.2])["i2"] == 0.0  # no dispersion
    assert M.i_squared([0.7, 0.9], [0.1, 1.0])["i2"] == 0.0  # Q < df
    assert np.isnan(M.i_squared([1.0, 0.9], [0.0, 0.1])["i2"])  # a zero-width fold CI
    assert np.isnan(M.i_squared([0.9], [0.1])["i2"]) and M.i_squared([0.9, np.nan], [0.1, 0.1])["k"] == 1


def test_ppv_reweighting_known_answer(kh):
    """Bayes at the population's own prevalence gives back the observed PPV/NPV; K5 re-weights each policy row to pi_ref."""
    c = M.confusion_rates(tp=30, fp=20, tn=80, fn=10)
    ppv, npv = M.adjust_ppv_npv(c["sens"], c["spec"], 40 / 140)
    assert np.isclose(ppv, c["ppv"]) and np.isclose(npv, c["npv"])
    assert np.allclose(M.adjust_ppv_npv(0.75, 0.8, 0.1), (0.075 / (0.075 + 0.18), 0.72 / (0.72 + 0.025)))
    d = kh.df[(kh.df["subgroup"] == "prevalence_shift") & (kh.df["subgroup_value"] == "pi_ref")]
    df = kh.df
    base = df[df["subgroup"].isna() & (df["level"] == "guid") & (df["split"] == "test") & (df["point"] == "n/a")]
    assert len(d) and set(d["metric"]) == {"ppv@pi_ref", "npv@pi_ref"}
    for r in d[d["metric"] == "ppv@pi_ref"].itertuples():
        v = base[(base["model_id"] == r.model_id) & (base["policy_id"] == r.policy_id) & (base["fold"] == r.fold)]
        v = v.drop_duplicates("metric").set_index("metric")["value"]
        with np.errstate(invalid="ignore"):  # a rule that never fires: 0/0
            assert np.isclose(r.value, M.adjust_ppv_npv(v["sens"], v["spec"], 0.1)[0], equal_nan=True)


def test_k1_k5_rows(kh):
    """K1 rows carry the uncalibrated calibration metrics; K5's prevalences are each population's GUID prevalence."""
    k1 = kh.df[(kh.df["subgroup"] == "calibration") & (kh.df["subgroup_value"] == "uncalibrated")]
    assert set(k1["metric"]) == set(M.CALIB_NAMES) and set(k1["split"]) == {"val", "test"}
    assert {"1", "2", "3", "pooled"} <= set(k1["fold"])
    prev = kh.df[(kh.df["subgroup_value"] == "prevalence") & (kh.df["model_id"] == "probe")]
    prev = prev.set_index(["fold", "metric"])["value"]
    g = kh.ctx.guids[kh.ctx.guids["model_id"] == "probe"]
    for f in (1, 2, 3):
        for sp, name in (("test", "pi_test"), ("val", "pi_val")):
            assert np.isclose(prev[(str(f), name)], g[(g["fold"] == f) & (g["split"] == sp)]["y"].mean())


def test_h3_dedupe_counts(kh):
    """g000 sits in all three test folds: first_fold keeps it once, exclude drops it; the difference is one GUID."""
    t = pd.read_parquet(kh.out / "tables" / "shared_test.parquet")
    au = t[(t["model_id"] == "probe") & (t["metric"] == "auroc")].iloc[0]
    g = kh.ctx.guids[(kh.ctx.guids["model_id"] == "probe") & (kh.ctx.guids["split"] == "test")]
    assert au["n_first_fold"] == g["guid"].nunique() and au["n_first_fold"] - au["n_exclude"] == 1
    assert np.isclose(au["delta"], au["exclude"] - au["first_fold"])
    other = kh.df[(kh.df["subgroup"] == "shared_test_policy") & (kh.df["metric"] == "auroc") & (kh.df["model_id"] == "probe")]
    assert other["subgroup_value"].tolist() == ["exclude"] and kh.res["H"]["H3"]["n_shared_test_guids"] == 1
    assert {"probe|42|test", "shortcut|na|test"} <= set(kh.res["H"]["H1"])


def test_k6_decision_curve(kh):
    d = pd.read_parquet(kh.out / "tables" / "decision_curve.parquet")
    p = d[(d["model_id"] == "probe") & (d["fold"] == "pooled")]
    assert np.allclose(p["pt"], M.DC_PTS)
    assert (p["nb_lo"] <= p["net_benefit"] + 1e-9).all() and (p["net_benefit"] <= p["nb_hi"] + 1e-9).all()
    f1 = d[(d["model_id"] == "probe") & (d["fold"] == "1")]
    g = kh.ctx.guids[(kh.ctx.guids["model_id"] == "probe") & (kh.ctx.guids["split"] == "test") & (kh.ctx.guids["fold"] == 1)]
    want = M.net_benefit(g["y"], expit(g["score_final_cal"]), M.DC_PTS)
    np.testing.assert_allclose(f1["net_benefit"], want["net_benefit"])


def test_k7_brier_at_end_is_the_final_score_brier(kh):
    """At ``end`` the snapshot is each GUID's last segment, whose online score is make_run's score_final."""
    df = kh.df
    d = df[(df["metric"] == "brier") & (df["point"] == "end") & (df["axis"] == "to_delivery") & (df["model_id"] == "probe")
           & (df["split"] == "test") & (df["fold"] == "1") & df["subgroup"].isna()]
    g = kh.ctx.guids[(kh.ctx.guids["model_id"] == "probe") & (kh.ctx.guids["split"] == "test") & (kh.ctx.guids["fold"] == 1)]
    assert len(d) == 1 and np.isclose(d["value"].iloc[0], brier_score_loss(g["y"], expit(g["score_final_cal"])))
    pooled = kh.df[(kh.df["metric"] == "brier") & (kh.df["level"] == "online") & (kh.df["fold"] == "pooled")
                   & kh.df["value"].notna()]
    assert len(pooled) and (pooled["ci_lo"] <= pooled["value"] + 1e-9).all()
    assert (pooled["value"] <= pooled["ci_hi"] + 1e-9).all()


def test_snapshots_and_stage_roc(kh):
    s = pd.read_parquet(kh.out / "tables" / "snapshots.parquet")
    assert set(s["model_id"]) == {"probe"}  # the GUID-level shortcut has no time-resolved score
    assert {"checkpoint", "bin"} == set(s["point"]) and not s.duplicated(["axis", "point", "t", "guid"]).any()
    roc = pd.read_parquet(kh.out / "tables" / "roc_points.parquet")
    assert roc["variant"].str.startswith("score_final").any()  # R10 appends, R1's rows stay
    assert kh.res["R10"]["n_rows"] >= 0  # make_run's stage is 'unknown' everywhere, so no stage rows here


def test_builders_render_empty_tables(tmp_path):
    """Every KH builder on a run with no tables at all draws the empty note and never raises."""
    import matplotlib.pyplot as plt

    from teb_vae.classifier import report as R
    from teb_vae.classifier.config import load

    c = M.classifier_cfg(load(overrides=["classifier.labels.task=three_class", "classifier.labels.head=multiclass",
                                         "classifier.labels.aux_3class_weight=0", "classifier.train.loss.name=ce",
                                         "classifier.run.seeds=[1,2]"]))
    stems = R.FIGURE_REGISTRY(c)
    assert {R.CALIBRATION_PER_CLASS, R.SEED_SPREAD, "val/" + R.CALIBRATION_GUID} <= set(stems)
    assert R.SEED_SPREAD not in R.FIGURE_REGISTRY(M.classifier_cfg(load()))
    T = R.load_tables(tmp_path)
    for f in (R._calibration_guid, R._calibration_per_class, R._calibration_folds, R._prevalence_shift,
              R._decision_curve, R._fold_forest, R._seed_spread, R._roc_stage, R._score_distributions):
        plt.close(f(T, c))
        plt.close(f(T, c, split="val", fold="1"))
    for f in (R._brier_vs_time, R._score_windows):
        plt.close(f(T, c, axis="to_delivery"))
    assert "Fold heterogeneity (H1)" in "\n".join(R._heterogeneity_md(T, c))
    assert "## 6. " in "\n".join(R._calibration_md(T, c))


def test_seed_ensemble_calibration_reaches_the_report(tmp_path):
    """A seed ensemble's ``folds/fold_<k>/ens[/<kind>]/calibration.json`` (``run.ens_dir``) is read as seed ``ens``, so
    §6, the per-unit table and K4 show the primary model's temperature; unit files keep their keys."""
    import json

    from teb_vae.classifier import report as R

    for rel, t in (("fold_1/seed_42", 1.5), ("fold_1/seed_42/frozen", 1.2), ("fold_1/ens", 2.0),
                   ("fold_1/ens/frozen", 3.0)):
        (tmp_path / "folds" / rel).mkdir(parents=True, exist_ok=True)
        (tmp_path / "folds" / rel / "calibration.json").write_text(json.dumps({"method": "temperature", "temperature": t}))
    T = R.load_tables(tmp_path)
    got = {(m, sd): R._unit_calibration(T, m, sd, 1).get("temperature")
           for m, sd in (("model", "42"), ("frozen", "42"), ("model", "ens"), ("frozen", "ens"), ("model_covoff", "ens"))}
    assert got == {("model", "42"): 1.5, ("frozen", "42"): 1.2, ("model", "ens"): 2.0, ("frozen", "ens"): 3.0,
                   ("model_covoff", "ens"): 2.0}
