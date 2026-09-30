"""E6 feature attribution (SPEC §11.12): hand-written Integrated Gradients (captum is not a dependency), the
``predictions/attribution.parquet`` table, the E6 analysis and its figure (only when ``eval.attribution.enabled``)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.classifier.tests.conftest import SMOKE_CONFIG
from teb_vae.classifier.tests.test_model import cfg, jitter, net, seg_batch, seq_batch


@pytest.mark.parametrize("kind", ["causal_transformer", "gru", "attention_mil"])
def test_ig_completeness_sequence_scope(kind):
    """Per GUID, Σ IG = f(x) - f(0) up to the midpoint-rule error; padding and masked steps get no attribution."""
    from teb_vae.classifier.train import integrated_gradients

    m = jitter(net(cfg(model__scope="sequence", model__sequence__kind=kind))).eval()
    b = seq_batch((5, 3, 1))
    attr, f_x, f_0 = integrated_gradients(m, b, torch.zeros(1), n_steps=64, aggregator="max", tau=1.0)
    gap = f_x - f_0
    assert attr.shape == (3, 5, b["x"].shape[2], b["x"].shape[3] + b["attn"].shape[3])
    assert gap.abs().min() > 1e-3  # a non-trivial check
    torch.testing.assert_close(attr.sum((1, 2, 3)), gap, atol=1e-3, rtol=1e-2)
    assert (attr[~b["step_mask"]] == 0).all()


@pytest.mark.parametrize("agg", ["max", "mean", "lse", "last", "topk_mean"])
def test_ig_completeness_segment_scope_over_whole_guids(agg):
    """Segment scope: the GUID score is ``agg`` over its segments, so the IG of each GUID sums over its segments."""
    from teb_vae.classifier.train import guid_scores, integrated_gradients

    m = jitter(net(cfg(model__scope="segment"))).eval()
    b = seg_batch(6) | {"guid": torch.tensor([0, 0, 1, 1, 1, 2])}
    attr, f_x, f_0 = integrated_gradients(m, b, torch.zeros(1), n_steps=64, aggregator=agg, tau=1.0)
    per_guid = torch.zeros(3).index_add_(0, b["guid"], attr.sum((1, 2)))
    torch.testing.assert_close(per_guid, f_x - f_0, atol=5e-3, rtol=2e-2)
    with torch.no_grad():  # f_x is the GUID score itself
        torch.testing.assert_close(f_x, guid_scores(m, b, torch.zeros(1), aggregator=agg, tau=1.0))


def test_whole_guid_batches_never_split_a_guid():
    from teb_vae.classifier.train import _whole_guid_batches

    codes = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    batches = _whole_guid_batches(codes, 4)
    assert batches == [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9]]  # an over-long GUID gets its own batch
    owners = [{int(g) for g in codes[b]} for b in batches]
    assert all(not (x & y) for i, x in enumerate(owners) for y in owners[i + 1:])  # no GUID in two batches
    assert sorted(i for b in batches for i in b) == list(range(len(codes)))


def test_attribution_figure_is_expected_only_when_enabled(smoke_config):
    from teb_vae.classifier import report

    c = smoke_config.model_dump(mode="json")
    assert report.ATTRIBUTION_CHANNELS not in report.FIGURE_REGISTRY(c)
    c["eval"]["attribution"]["enabled"] = True
    assert report.ATTRIBUTION_CHANNELS in report.FIGURE_REGISTRY(c)


@pytest.mark.slow
def test_attribution_end_to_end(smoke_overrides, tmp_path):
    """``eval.attribution.enabled`` on the 3-fold smoke (sequence scope): predict writes the table, E6 summarises it
    with near-exact completeness, report draws the figure, verify passes (criterion 11 now expects the figure)."""
    from loguru import logger

    from teb_vae.classifier import run, verify
    from teb_vae.classifier.train import ATTRIBUTION_COLUMNS

    overrides = [*smoke_overrides, "classifier.run.folds=[1,2,3]", "classifier.eval.attribution.enabled=true"]
    try:
        run_dir = run.main(config=str(SMOKE_CONFIG), stage="all", overrides=overrides, run_dir=str(tmp_path / "run"))
    finally:
        logger.remove()
        logger.add(sys.stderr)
    a = pd.read_parquet(run_dir / "predictions" / "attribution.parquet")
    assert list(a.columns) == ATTRIBUTION_COLUMNS and set(a["fold"]) == {1, 2, 3} and set(a["split"]) == {"test"}
    assert set(a["channel_group"]) == {"fhr_st", "fhr_ph", "up_st", "up_ph"}  # smoke.yaml's hdf5 fields
    per_guid = a.groupby(["fold", "guid"]).agg(total=("signed", "sum"), f_x=("f_x", "first"), f_0=("f_0", "first"))
    np.testing.assert_allclose(per_guid["total"], per_guid["f_x"] - per_guid["f_0"], atol=0.05, rtol=0.05)
    summary = json.loads((run_dir / "evaluation" / "summary.json").read_text())
    e6 = summary["results"]["E6"]
    assert e6["status"] == "done" and e6["completeness"]["median_rel_error"] < 0.05
    t = pd.read_parquet(run_dir / "evaluation" / "tables" / "attribution.parquet")
    pooled = t[(t["fold"] == "pooled") & (t["metric"] == "share")]
    assert len(pooled) and pooled["value"].between(0, 1).all() and pooled["ci_lo"].le(pooled["ci_hi"]).all()
    assert list((run_dir / "evaluation" / "figures").glob("errors/attribution_channels.*"))
    assert verify.main([str(run_dir)]) == 0
