"""Block E (SPEC §11.12 E1-E5): the E1 top-errors ranking (known answer, top_k, per fold) and its agreement with block A,
the seeded E2 page selection, the pooling-attention scalars, the masking and log-scale helpers ported from the VAE
attribution module, and every block E builder (figures, page, summary lines) on empty tables."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from teb_vae.classifier import metrics as M
from teb_vae.classifier import report as R
from teb_vae.classifier.config import load
from teb_vae.classifier.tests.test_eval_analyses import make_run


def test_top_errors_known_answer():
    """Per fold: FPs are the healthy GUIDs by descending score, FNs the adverse by ascending, ``k`` capped at half a
    class (the ported extreme_rows' disjoint tails); NaN scores never rank."""
    g = pd.DataFrame({"fold": [1] * 9 + [2] * 4, "guid": list("abcdefghi") + list("wxyz"),
                      "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                      "max_score": [0.1, 0.9, 0.5, np.nan, 0.7, 2.0, -1.0, 0.3, 1.5, 3.0, 1.0, -2.0, 0.0]})
    got = pd.concat([M.top_errors(x, 2).assign(fold=f) for f, x in g.groupby("fold")], ignore_index=True)
    assert got[["fold", "kind", "rank", "guid"]].values.tolist() == [
        [1, "fp", 1, "b"], [1, "fp", 2, "e"], [1, "fn", 1, "g"], [1, "fn", 2, "h"],
        [2, "fp", 1, "w"], [2, "fn", 1, "y"]]  # fold 2 has two GUIDs per class: one per tail
    assert M.top_errors(g[g["fold"] == 1], 10)["kind"].value_counts().to_dict() == {"fp": 2, "fn": 2}
    assert M.top_errors(g.iloc[:0], 3).empty
    tails = M.extreme_rows(g, "max_score", per_tail=3)
    assert tails["low"]["guid"].tolist() == ["y", "g", "z"] and tails["high"]["guid"].tolist() == ["i", "f", "w"]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """``make_run`` with block A and block E run (the probe has online scores; the shortcut has none)."""
    run, cfg = make_run(tmp_path_factory.mktemp("e"))
    ctx = M.load_context(run, cfg)
    out = run / "evaluation"
    (out / "tables").mkdir(parents=True)
    M.run_A(ctx, eval_config=ctx.cfg["eval"], out_dir=out)
    res = M.run_E(ctx, eval_config=ctx.cfg["eval"], out_dir=out)
    return run, cfg, ctx, res, {n: pd.read_parquet(out / "tables" / f"{n}.parquet")
                                for n in ("errors", "error_segments", "alarms")}


def test_run_e_tables(ran):
    """E1 per model x split x fold, agreeing with block A's latched alarm and lead time of the primary policy; the
    shortcut is skipped; ``error_segments`` holds the primary model's pooled test GUIDs once each."""
    _, _, ctx, res, t = ran
    e, k = t["errors"], ctx.cfg["eval"]["error_analysis"]["top_k"]
    assert list(e.columns) == M.ERROR_COLUMNS and set(e["model_id"]) == {"probe"}
    assert sorted(res["skipped"]) == ["shortcut|na|test", "shortcut|na|val"]
    assert set(map(tuple, e[["split", "fold"]].drop_duplicates().values)) == {(s, f) for s in M.SPLITS for f in (1, 2, 3)}
    assert e.groupby(["split", "fold", "kind"])["rank"].max().le(k).all()
    fp = e[e["kind"] == "fp"]
    assert (fp["y"] == 0).all() and (e.loc[e["kind"] == "fn", "y"] == 1).all()
    assert fp.groupby(["split", "fold"])["max_score"].apply(lambda s: s.is_monotonic_decreasing).all()
    a = t["alarms"]
    a = a[(a["rule"] == "latch") & (a["policy_id"] == ctx.cfg["eval"]["primary_policy"])]
    j = e.merge(a, on=["model_id", "seed", "fold", "split", "guid"], suffixes=("", "_a"), validate="one_to_one")
    assert len(j) == len(e) and (j["alarmed"] == j["alarmed_a"]).all()
    assert np.allclose(j["first_alarm_h"].fillna(-1), j["lead_time_h"].fillna(-1))
    assert (j["alarmed"] == (j["max_score"] > j["threshold"])).all()
    s = t["error_segments"]
    assert set(s["model_id"]) == {"probe"} and set(s["split"]) == {"test"}
    assert s.groupby("guid")["fold"].nunique().eq(1).all()  # the shared test GUID once
    assert s[list(M.ATTN_SCALARS)].isna().all(axis=None)  # the probe has no pooling attention


def test_page_selection_is_seeded_per_class_and_includes_e1(ran):
    _, _, _, _, t = ran
    run = ran[0]
    g = pd.read_parquet(run / "predictions" / "guids.parquet")
    g = g[(g["model_id"] == "probe") & (g["split"] == "test")]
    e = t["errors"][(t["errors"]["split"] == "test")]
    kw = dict(per_class=2, top_errors=1)
    a, b = R.page_selection(g, e, seed=3, **kw), R.page_selection(g, e, seed=3, **kw)
    assert a.equals(b)
    samples = a[a["reason"].str.contains("sample")]
    assert samples.groupby(["fold", "clinical_class"]).size().le(2).all()
    assert set(samples["clinical_class"]) == {"healthy", "acidosis", "hie"}
    want = {(f, x) for f, x in e.loc[e["rank"] <= 1, ["fold", "guid"]].itertuples(index=False)}
    assert want <= set(a[["fold", "guid"]].itertuples(index=False, name=None))
    assert not a.duplicated(["fold", "guid"]).any()
    assert set(a["reason"].str.split("+").explode()) <= {"sample", "fp1", "fn1"}
    assert R.page_selection(g.iloc[:0], e.iloc[:0], seed=0, **kw).empty
    drawn = R.per_class_rows(g[g["fold"] == 1], per_class=1, seed=0)
    assert drawn["clinical_class"].value_counts().to_dict() == {"healthy": 1, "acidosis": 1, "hie": 1}


def test_attention_scalars_known_answer():
    import torch

    from teb_vae.classifier.train import _attention

    a = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0], [0.25, 0.25, 0.25, 0.25, 0.0]])
    valid = torch.tensor([[False, False, True, True, True], [True, True, True, True, False]])
    seq = torch.tensor([[[1.0, 0.0, 0.0], [0.3, 0.7, 0.0], [0.2, 0.3, 0.5]]])
    out = _attention(a, valid, seq, torch.tensor([[True, True, False]]))
    assert torch.allclose(out["attn_centroid"], torch.tensor([0.25, 0.5]))  # positions 0, .5, 1 | 0, 1/3, 2/3, 1
    assert torch.allclose(out["attn_late_mass"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out["seq_attn_final"], torch.tensor([0.3, 0.7]))  # row 1: the last real position
    assert out["attn_centroid"].shape == _attention(a, valid, None, torch.ones(2, dtype=torch.bool))[
        "seq_attn_final"].shape


@pytest.mark.parametrize("scope,pool,seq", [("segment", "query", None), ("segment", "conjunctive", None),
                                            ("sequence", "gated_attention", "causal_transformer"),
                                            ("sequence", "mean", "attention_mil")])
def test_prediction_rows_carry_the_attention_scalars(scope, pool, seq):
    """``train._rows`` on an untrained net: the step scalars lie in [0, 1] on every real segment; the final position's
    ``attention_mil`` weights over a GUID's segments sum to 1, and are NaN for every other aggregator."""
    import torch

    from teb_vae.classifier.tests.test_model import cfg, jitter, net, seg_batch, seq_batch
    from teb_vae.classifier.train import ATTN_COLUMNS, _rows

    over = {"model__scope": scope, "model__pooling__kind": pool, **({"model__sequence__kind": seq} if seq else {})}
    m = jitter(net(cfg(**over))).eval()
    batch = seq_batch() if scope == "sequence" else seg_batch()
    with torch.no_grad():
        r = _rows(m, m(batch), batch, torch.zeros(1))
    assert set(ATTN_COLUMNS) <= set(r) and all(len(r[k]) == len(r["row"]) for k in ATTN_COLUMNS)
    for k in ("attn_late_mass", "attn_centroid"):
        assert ((r[k] >= -1e-6) & (r[k] <= 1 + 1e-6)).all()
    if seq == "attention_mil":
        guid = batch["guid"][:, None].expand_as(batch["seg_mask"])[batch["seg_mask"]]
        per_guid = pd.Series(r["seq_attn_final"]).groupby(guid.numpy()).sum()
        assert np.allclose(per_guid, 1.0, atol=1e-5)
    else:
        assert np.isnan(r["seq_attn_final"]).all()


def test_ported_masking_and_log_helpers():
    field = np.arange(12, dtype=float).reshape(4, 3)
    m = R.masked_field(field, np.array([0, 2, 4]))
    assert np.isnan(m).tolist() == [[False, True, True], [False, True, True], [False, False, True], [False, False, True]]
    assert np.array_equal(R.masked_field(field, None), field)
    n = R.signed_log_norm(np.array([-2.0, 0.5, np.nan]))
    assert (n.vmin, n.vmax, n.linthresh) == (-2.0, 2.0, pytest.approx(2e-4))
    u = R.unsigned_log_norm(np.array([0.0, 3.0, -1.0]))
    assert (u.vmin, u.vmax) == (pytest.approx(3e-4), 3.0)
    assert R.signed_log_norm(np.zeros(3)) is None and R.unsigned_log_norm(np.array([-1.0, np.nan])) is None
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.plot([0, 1], [-50.0, 100.0])
    R.symlog_axis(a1, [-50.0, 100.0])
    R.symlog_axis(a2, [0.0, np.nan])
    assert a1.get_yscale() == "symlog" and a1.yaxis.get_transform().linthresh == pytest.approx(1e-2)
    assert a2.get_yscale() == "linear"
    plt.close(fig)


def test_registry_predicates():
    base = M.classifier_cfg(load())
    stems = R.FIGURE_REGISTRY(base)
    assert {R.SCORE_VS_QUALITY, R.SCORE_VS_LENGTH, R.ATTENTION_SUMMARY} <= set(stems)  # default: gated attention
    for over in (["classifier.model.pooling.kind=mean"], ["classifier.model.pooling.kind=mean_max"]):
        assert R.ATTENTION_SUMMARY not in R.FIGURE_REGISTRY(M.classifier_cfg(load(overrides=over)))
    mil = ["classifier.model.pooling.kind=mean", "classifier.model.sequence.kind=attention_mil"]
    assert R.ATTENTION_SUMMARY in R.FIGURE_REGISTRY(M.classifier_cfg(load(overrides=mil)))
    assert "errors" in M.P2_TABLES and {"errors", "error_segments"} <= set(R.EXTRA_TABLES)


def test_builders_render_empty_tables(tmp_path):
    """Every block E builder on a run with no tables draws the empty note and never raises; the page step writes an
    empty manifest; a page renders with and without cached features."""
    c = M.classifier_cfg(load())
    T = R.load_tables(tmp_path)
    for stem in (R.SCORE_VS_QUALITY, R.SCORE_VS_LENGTH, R.ATTENTION_SUMMARY):
        plt.close(R._BUILDERS[stem](T, c))
        plt.close(R._BUILDERS[stem](T, c, split="val", fold="1"))
    assert "_no top-error rows" in "\n".join(R._errors_md(T, c))
    assert R.render_pages(tmp_path, T, c) == {"n_pages": 0}
    assert pd.read_csv(tmp_path / "evaluation" / "pages" / R.PAGES_CSV).empty
    g = pd.DataFrame({"guid": "a", "seg_pos": [0, 1], "epoch_s": [-4000.0, -3340.0], "t_end_s": [-2740.0, -2080.0],
                      "logit_online_cal": [-1.0, 2.0], "logit_seg_cal": np.nan, "attn_late_mass": [0.4, 0.6]})
    mask = np.zeros((2, 6), bool)
    mask[:, 3:] = True
    X = np.ones((2, 6, 3))
    X[:, 3, 1] = 0.0  # a cold cell on a valid step: blank
    kw = dict(thresholds={"np30": 0.0, "emp30": -0.5}, geometry=(4.0, 60.0), header="h", caveat="x")
    for feats in ((X, mask, ("fhr_st[0]", "fhr_st[1]", "kld_per_t")), (X, np.zeros_like(mask), ("a", "b", "c")), None):
        plt.close(R._trajectory_page(g, feats, c, **kw))
