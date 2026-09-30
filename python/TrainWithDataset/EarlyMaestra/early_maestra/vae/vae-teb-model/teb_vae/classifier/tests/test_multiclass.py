"""Block X (confusion and 3-class, SPEC §11.12 X1-X9): known answers for the pooling and snapshot rules, the vectorised
3-class statistics against sklearn, the CI rows, X9 on a synthetic multi-task run, and empty-table figures."""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from teb_vae.classifier import metrics as M
from teb_vae.classifier import report as R

BOOT = {"resamples": 100, "seed": 0}
CODE = {"healthy": 1, "acidosis": 2, "hie": 3}
MC = ["classifier.labels.task=three_class", "classifier.labels.head=multiclass", "classifier.labels.aux_3class_weight=0",
      "classifier.train.loss.name=weighted_ce", "classifier.train.loss.weighting=sqrt_inverse"]


def _probs(y, rng, signal=1.5):
    z = rng.normal(size=(y.size, 3)) + signal * np.eye(3)[y]
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def test_mc_stats_match_sklearn_and_multiclass():
    """X8: with unit weights the vectorised statistics are multiclass()'s (sklearn) and the collapses' ranking metrics
    threshold_free()'s; QWK, weighted F1, Hand-Till and the ordinal alarm-score AUROC against sklearn directly."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 80)
    P, o = _probs(y, rng), rng.normal(size=80) + y
    st = {k: float(v[0]) for k, v in M._mc_stats(y, P, np.ones((1, y.size)), alpha=0.3, ords=o).items()}
    for k, v in M.multiclass(y, P).items():
        assert st[k] == pytest.approx(v, nan_ok=True), k
    pred = P.argmax(1)
    assert st["qwk"] == pytest.approx(cohen_kappa_score(y, pred, weights="quadratic"))
    assert st["auroc_hand_till"] == pytest.approx(roc_auc_score(y, P, multi_class="ovo"))
    assert st["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert st["ordinal_auroc_adverse"] == pytest.approx(roc_auc_score(y > 0, o))
    assert st["ordinal_auroc_hie"] == pytest.approx(roc_auc_score(y == 2, o))
    tf = M.threshold_free(y > 0, np.log((P[:, 1] + P[:, 2]) / P[:, 0]), alpha=0.3)
    for m in ("auroc", "auprc", "pauc@0.3"):
        assert st[f"adverse_vs_healthy/{m}"] == pytest.approx(tf[m])


def test_x2_pooling_summed_vs_mean_of_row_normalised():
    """X2 known answer: the pooled matrix is the summed counts; the second view averages the per-fold row-normalised
    matrices, so a small fold weighs as much as a large one, and a fold without a class skips that row."""
    a = np.array([[8, 2, 0], [0, 1, 1], [0, 0, 2]])
    b = np.array([[1, 1, 0], [2, 6, 2], [0, 0, 0]])  # no HIE in this fold
    np.testing.assert_allclose(M.row_normalised(a + b)[1], [2 / 12, 7 / 12, 3 / 12])
    mean, sd = M.fold_mean_rownorm([a, b])
    np.testing.assert_allclose(mean[0], [(0.8 + 0.5) / 2, (0.2 + 0.5) / 2, 0.0])
    np.testing.assert_allclose(mean[1], [(0 + 0.2) / 2, (0.5 + 0.6) / 2, (0.5 + 0.2) / 2])
    np.testing.assert_allclose(mean[2], [0, 0, 1])  # fold b has no HIE row: fold a alone
    assert sd[0, 0] == pytest.approx(np.std([0.8, 0.5], ddof=1)) and np.isnan(sd[2]).all()


def _ctx(rng, n=30, folds=(1, 2)):
    """A 3-class context of ``n`` test GUIDs per fold, 3 in-window segments each (segment scores present)."""
    g = pd.DataFrame([{"model_id": "m", "seed": "1", "fold": f, "split": "test", "guid": f"g{f}{i:02d}",
                       "class_code": 1 + i % 3, "shared_test": False} for f in folds for i in range(n)])
    P = _probs(g["class_code"].to_numpy() - 1, rng)
    g = g.assign(y=(g["class_code"] > 1).astype(int), **{f"p_c{k}_cal": P[:, k] for k in range(3)})
    s = g.loc[g.index.repeat(3)].assign(seg_pos=np.tile(np.arange(3), len(g)), in_eval_window=True, t_end_s=-600.0)
    s["p_c0_cal"] = s["p_c0_cal"] * rng.uniform(0.8, 1.2, len(s))  # segments differ from the GUID output
    s = s.assign(logit_seg_cal=0.0, **{f"p_c{k}_cal": s[f"p_c{k}_cal"] / s[["p_c0_cal", "p_c1_cal", "p_c2_cal"]].sum(axis=1)
                                        for k in range(3)})
    lab = {"task": "three_class", "head": "multiclass", "strategy": "hard", "eval_window": "all"}
    return SimpleNamespace(cfg={"labels": lab, "eval": {"exclude_last_min": 0}}, prov={"run_id": "r"}, guids=g,
                           seg=s.reset_index(drop=True), inclusion=[])


def test_three_class_rows_carry_cis_and_pool_by_summing():
    ctx = _ctx(np.random.default_rng(1))
    df = pd.DataFrame(M._three_class_rows(ctx, "m", "1", "test", 0.3, "first_fold", BOOT), columns=M.METRIC_COLUMNS)
    assert set(df["level"]) == {"guid", "segment"}
    g = df[df["level"] == "guid"].set_index(["fold", "metric"])["value"]
    for name in M.CONFUSION_NAMES:  # pooled = summed over folds (no shared test GUID here)
        assert g[("pooled", name)] == g[("1", name)] + g[("2", name)]
    assert sum(g[("pooled", n)] for n in M.CONFUSION_NAMES) == 60
    ci = df[df["metric"].isin(["auroc_macro", "auroc_hand_till", "qwk", "rps", "recall_c1", "bal_acc", "macro_f1",
                               "auprc_ovr_c2", "adverse_vs_healthy/auroc", "hie_vs_rest/pauc@0.3"])]
    assert len(ci) == 10 * 3 * 2 and ci[["ci_lo", "ci_hi"]].notna().all(axis=None)
    assert ((ci["ci_lo"] <= ci["value"] + 1e-9) & (ci["value"] <= ci["ci_hi"] + 1e-9)).all()
    assert (ci["n_boot_undefined"] == 0).all()
    assert df[df["metric"].isin(M.CONFUSION_NAMES)]["ci_lo"].isna().all()
    seg = df[(df["level"] == "segment") & (df["fold"] == "pooled") & (df["metric"] == "auroc_macro")].iloc[0]
    assert seg["n_pos"] + seg["n_neg"] == 180 and seg["eval_window"] == "all"


def test_x3_confusion_is_one_snapshot_per_guid_per_bin():
    """Each GUID counts once per bin, with its LAST segment in the bin (the snapshot), whatever its segment count."""
    spec = {  # guid: (class, clocks in h to delivery, argmax per segment)
        "A": ("healthy", [-1.4, -1.2], [2, 0]),
        "B": ("hie", [-1.3, -0.2], [2, 1]),
        "C": ("acidosis", [-2.2, -1.1, -1.05], [1, 1, 0]),
    }
    seg = pd.DataFrame([{"guid": g, "seg_pos": i, "t_end_s": c * 3600.0, "epoch_s": c * 3600.0 - 1260.0,
                         "class_code": CODE[cl], "y": int(cl != "healthy"), "fold": 1, "stage": "first",
                         "logit_online_cal": 0.0, "logit_seg_cal": 0.0,
                         **{f"p_c{k}_cal": 0.8 if k == p else 0.1 for k in range(3)}}
                        for g, (cl, cs, ps) in spec.items() for i, (c, p) in enumerate(zip(cs, ps))])
    ev = {"checkpoints_h": [1.0], "snapshot_max_staleness_h": 1.0, "bin_h": 0.5}
    U = M._prep(seg, {1: {"p": {"threshold": 0.0, "basis": "committed_overall"}}}, ["p"], {}, [], {"kind": "latch"})
    pts = M._points("to_delivery", M._grids([("1", U)], ["to_delivery"], 0.5)["to_delivery"], ev)
    V = M._view(U, "to_delivery", pts)
    P = U.o[[f"p_c{k}_cal" for k in range(3)]].to_numpy()
    df = M._blocks_frame(M._argmax_block(U, V, pts, P, mn=0, boot=BOOT, fold="1", axis="to_delivery"))

    def at(point, t, metric):
        q = df[(df["point"] == point) & (df["metric"] == metric) & (np.isclose(df["t"], t) if t else df["t"].isna())]
        assert len(q) == 1
        return q["value"].iloc[0]

    conf = {n: at("bin", 1.25, n) for n in M.CONFUSION_NAMES}  # bin (-1.5, -1.0]: 5 segments of 3 GUIDs
    assert sum(conf.values()) == 3 and conf["confusion_t0_p0"] == conf["confusion_t1_p0"] == conf["confusion_t2_p2"] == 1
    assert at("bin", 1.25, "top1_acc") == pytest.approx(2 / 3) and at("bin", 1.25, "recall_c1") == 0.0
    assert at("bin", 0.25, "confusion_t2_p1") == 1 and at("bin", 2.25, "confusion_t1_p1") == 1
    assert at("checkpoint", 1.0, "confusion_t1_p0") == 1  # the checkpoint snapshot within the staleness window
    end = {n: at("end", None, n) for n in M.CONFUSION_NAMES}
    assert sum(end.values()) == 3 and end["confusion_t2_p1"] == 1 and end["confusion_t0_p0"] == 1
    under = M._blocks_frame(M._argmax_block(U, V, pts, P, mn=2, boot=BOOT, fold="1"))  # 1 GUID per class < 2
    assert under[under["metric"].isin(M.ARGMAX_TR)]["value"].isna().all()


def test_x9_collapse_consistency_on_a_synthetic_multi_task_run(tmp_path):
    """The aux collapse is a monotone map of the binary head: both AUROCs equal, Spearman 1, no disagreement (the aux
    threshold, re-selected on val, is the same order statistic); reversed, AUROC flips and Spearman is -1."""
    from teb_vae.classifier.tests.test_eval_analyses import make_run

    run, cfg = make_run(tmp_path)
    assert cfg.classifier.labels.aux_3class_weight > 0 and cfg.classifier.labels.task == "adverse_vs_healthy"
    pred = run / "predictions"
    g0, s0 = pd.read_parquet(pred / "guids.parquet"), pd.read_parquet(pred / "segments.parquet")
    out = {}
    for sign in (1.0, -1.0):
        for name, f, col in (("guids", g0, "score_final_cal"), ("segments", s0, "logit_online_cal")):
            q = expit(sign * f[col].to_numpy(np.float64))
            f.assign(p_c0=1 - q, p_c1=q / 2, p_c2=q / 2).to_parquet(pred / f"{name}.parquet", index=False)
        ctx = M.load_context(run, cfg)
        res = M.run_X9(ctx, eval_config=ctx.cfg["eval"], out_dir=run / "evaluation")
        out[sign] = pd.DataFrame(res["rows"], columns=M.METRIC_COLUMNS)
    same, flip = (x.set_index(["model_id", "split", "fold", "metric"])["value"].sort_index() for x in out.values())
    assert set(out[1.0]["model_id"]) == {"probe", "shortcut"}
    for key in same.index.droplevel(3).unique():
        v, w = same[key], flip[key]
        assert v["collapse/auroc_aux"] == pytest.approx(v["collapse/auroc_binary"])
        assert v["collapse/spearman"] == pytest.approx(1.0) and w["collapse/spearman"] == pytest.approx(-1.0)
        assert v["collapse/disagreement"] == 0 and v["collapse/binary_only"] == v["collapse/aux_only"] == 0
        assert w["collapse/auroc_aux"] == pytest.approx(1 - w["collapse/auroc_binary"])
    assert (flip.xs("collapse/disagreement", level=3) > 0).any()


@pytest.mark.parametrize("overrides", [MC, []], ids=["three_class", "multi_task_binary"])
def test_builders_render_empty_tables(tmp_path, overrides):
    """Every block X figure (and §7) on a run with no tables at all: the empty note, never an exception."""
    from teb_vae.classifier.config import load

    c = M.classifier_cfg(load(overrides=[*overrides, "classifier.eval.time_axes=[to_delivery,position]"]))
    T = R.load_tables(tmp_path)
    three = {R.CONFUSION_3CLASS, R.ROC_OVR, R.PR_OVR, *(
        t.format(axis=a) for t in (R.CONFUSION_EVOLUTION, R.PER_CLASS_VS_TIME, R.PER_CLASS_AUROC_VS_TIME, R.F1_VS_TIME)
        for a in ("to_delivery", "position"))}
    want = {R.CONFUSION_BINARY} | (three if overrides else {R.COLLAPSE_VS_BINARY})
    builders = R._figure_set(c)
    assert want <= set(builders) and not ((three | {R.COLLAPSE_VS_BINARY}) - want) & set(builders)
    for stem in want:
        plt.close(builders[stem](T, c))
    plt.close(R._collapse_vs_binary(T, c))  # builders also tolerate a config they are not expected for
    md = R._three_class_md(T, c)
    assert md[0].startswith("## 7. ") and ("Not applicable" in "\n".join(md)) != bool(overrides)


def test_non_causal_frames_drop_segment_class_probabilities():
    """Top-15 #5 at the source: a non-causal sequence model's per-position outputs (online logit, class probabilities,
    CORAL g) have seen later segments, so ``train._frames`` NaNs them on every segment row; the GUID row keeps the final
    position's, and the causal segment-local head's ``pseg_c<k>`` stay."""
    from teb_vae.classifier.train import _frames

    frame = pd.DataFrame({"guid": ["a", "a", "a", "b", "b"], "fold": 1, "split": "val", "y": [1, 1, 1, 0, 0],
                          "class_code": [2, 2, 2, 1, 1]})
    P, Q = np.random.default_rng(0).dirichlet([1, 1, 1], (2, 5))
    part = {"row": np.arange(5), "logit_seg": np.zeros(5), "logit_online": np.ones(5), "ord_score": np.arange(5.0),
            "p3": P, "p3_seg": Q}
    kw = dict(scope="sequence", aggregators=["max"], tau=1.0)
    seg, gd = _frames(frame, [part], causal=False, **kw)
    p = [f"p_c{k}" for k in range(3)]
    assert seg[["logit_online", "ord_score", *p]].isna().all(axis=None)
    assert np.allclose(seg[[f"pseg_c{k}" for k in range(3)]], Q) and np.allclose(seg["logit_seg"], 0.0)
    assert np.allclose(gd.set_index("guid").loc[["a", "b"], p], P[[2, 4]])
    assert gd.set_index("guid")["ord_score"].tolist() == [2.0, 4.0]
    causal, _ = _frames(frame, [part], causal=True, **kw)
    assert np.allclose(causal[p], P) and causal["logit_online"].notna().all()
