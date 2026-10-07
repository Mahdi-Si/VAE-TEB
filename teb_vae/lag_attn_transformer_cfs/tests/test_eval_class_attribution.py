r"""Grad-CAM on this model and the class comparison of the attribution cohort pass.

Grad-CAM is checked on the tiny conv-Transformer against facts the forward fixes exactly: every view
is a share on the model's own axis, a target-only readout leaves no evidence on the source side, and
the attention view of a lag-band readout is the forward's own attention, restricted to the band and
in lag order. The class statistics are checked on planted per-recording values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import class_contrast, gradcam
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, model_inputs

from .conftest import make_stub_batch, make_task


def _rows():
    """The tiny task, a few scored anchors per stub segment, and the dense forward."""
    task = make_task()
    task.eval()
    model = task.orig_model
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, make_stub_batch(batch=2, seed=3))
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=DENSE_ANCHOR_GEOMETRY[0], anchor_stride=DENSE_ANCHOR_GEOMETRY[1])
    columns_per_sample = core.spread_columns(core.contributing_columns(model, weight, outputs), 3)
    inputs, extra, columns, sample = core.expand_rows((y_st, y_ph, u_stream), (target_features, weight), columns_per_sample)
    return task, inputs, extra, columns, sample, outputs


def _wrapper(task, readout: str, **kwargs) -> core.AnchorReadout:
    return core.AnchorReadout(task.orig_model, core.ATTENTION_CELL, readout=readout, **kwargs).eval()


def test_every_view_is_a_share_on_the_models_own_axis() -> None:
    task, inputs, extra, columns, _sample, _outputs = _rows()
    model = task.orig_model
    assert gradcam.target_layer(model)[1] == "target_encoder.attention_blocks[-1] input"
    for readout in core.MAIN_READOUTS:
        wrapper = _wrapper(task, readout)
        cams = gradcam.gradcam(wrapper, inputs, extra, columns)
        with torch.no_grad():
            value = wrapper(*inputs, *extra, columns, torch.zeros_like(columns))
        assert np.allclose(cams["value"], value.numpy(), atol=1e-6)
        assert cams["target"].shape == (len(columns), int(model.sequence_length))
        # The warm-up steps are no part of the target map: every offset reaching before F is NaN.
        reach = wrapper.anchor_steps(columns).cpu().numpy() - int(model.warmup_period)
        offsets = np.arange(cams["target"].shape[1])
        assert np.isnan(cams["target"][offsets[None, :] > reach[:, None]]).all()
        assert cams["source"].shape == cams["attention"].shape == (len(columns), int(model.lag_attn.L))
        for view in gradcam.VIEWS:
            sums = np.nansum(cams[view], axis=1)
            populated = np.isfinite(cams[view]).any(axis=1)
            assert np.allclose(sums[populated], 1.0) and (cams[f"{view}_total"] >= 0.0).all()


def test_a_target_only_readout_leaves_no_evidence_on_the_source_side() -> None:
    task, inputs, extra, columns, _sample, _outputs = _rows()
    cams = gradcam.gradcam(_wrapper(task, core.READOUT_NLL_BASE), inputs, extra, columns)
    assert (cams["source_total"] == 0.0).all() and (cams["attention_total"] == 0.0).all()
    assert (cams["target_total"] > 0.0).any()


def test_the_attention_view_of_a_lag_band_readout_is_the_forwards_attention_on_that_band() -> None:
    task, inputs, extra, columns, sample, outputs = _rows()
    low, high = 1, 3
    cams = gradcam.gradcam(_wrapper(task, core.READOUT_LAG_BAND, lag_band=(low, high)), inputs, extra, columns)
    anchors = outputs["anchor_index"][torch.as_tensor(sample), columns]
    alpha = outputs["attn_weights"][torch.as_tensor(sample), anchors].mean(dim=1).numpy()          # (N, L)
    band = alpha[:, low:high + 1]
    expected = np.zeros_like(alpha)
    expected[:, low:high + 1] = band / band.sum(axis=1, keepdims=True)
    assert np.allclose(cams["attention"], expected, atol=1e-5)


def _planted(shift: float, n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name in ("healthy", "acidosis", "hie"):
        for index in range(n):
            rows.append({"guid": f"{name}{index}", labels.CLASS_COLUMN: name,
                         "planted": rng.normal() + (shift if name == "hie" else 0.0), "null": rng.normal()})
    return pd.DataFrame(rows).set_index("guid")


def test_the_class_tests_find_a_planted_difference_and_read_it_worst_first() -> None:
    stats, pairwise = class_contrast.class_tests(_planted(3.0), {"demo": ["planted", "null"]}, seed=0)
    by_metric = stats.set_index("metric")
    assert by_metric.loc["planted", "significant"] and not by_metric.loc["null", "significant"]
    pair = pairwise[(pairwise["metric"] == "planted") & (pairwise["left"] == "hie") & (pairwise["right"] == "healthy")].iloc[0]
    assert pair["cliffs_delta"] > 0.9 and pair["diff_ci_lo"] > 0.0
    assert list(pairwise[pairwise["metric"] == "planted"][["left", "right"]].itertuples(index=False, name=None)) == [
        ("hie", "acidosis"), ("hie", "healthy"), ("acidosis", "healthy"),
    ]


def test_the_difference_band_of_one_group_against_itself_straddles_zero() -> None:
    curves = np.random.default_rng(1).normal(size=(10, 5))
    difference, lo, hi = class_contrast.difference_band(curves, curves.copy(), seed=0)
    assert np.allclose(difference, 0.0) and (lo <= 0.0).all() and (hi >= 0.0).all()
