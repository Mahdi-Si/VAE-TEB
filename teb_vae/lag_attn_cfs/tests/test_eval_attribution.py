r"""The Captum attributions: the wrapper, the four structural properties, the reductions, the pass.

Four things are proved here on the tiny model rather than left to a look at a figure. **The
wrapper is the forward**: every readout it returns equals the quantity the forward, the mean
decode and the attention carry at that anchor, so an attribution of it is an attribution of the
number the tables report. **The four structural properties hold exactly**: no attribution to a
step after the anchor, none to a source step a channel had not warmed up at, none from the source
to a target-only readout, and the integrated-gradient sum reproduces the readout difference from
the entry point. **The reductions are the arithmetic they claim** -- the lag re-indexing, the band
sums, the agreement, the per-head split. **The selection is class-balanced, one segment per
recording, seeded and capped**, and the traced recording is the most complete one of its class.

The end-to-end test drives the analysis through the same stub dataset the traces' test uses, at a
reduced step count, so the whole path -- selection, the sequential subset loader, the identity
check, every Captum call, the tables, the figures and the trace -- runs on CPU in seconds.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import attribution_pass
from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import attribution as analysis
from teb_vae.lag_attn_cfs.eval.dataset_rows import dataset_index_map
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, anchor_support, mean_decoded_block, model_inputs
from teb_vae.lag_attn_cfs.eval.report_seam import json_safe
from teb_vae.lag_attn_rws.nets.controls import source_null_forward_outputs

from .conftest import TINY_STRIDE, make_stub_batch, make_task, tiny_warmup_kwargs

#: Seconds between consecutive stored segments in the stub batch.
STRIDE_S = 1200.0

#: How far a completeness residual may sit, as a share of the readout, on the tiny conv-LSTM
#: model at the shipped step count. Its per-step group norm makes its path the roughest of the
#: family's; the transformer cells sit two orders below this.
COMPLETENESS_MEDIAN = 1e-2
COMPLETENESS_MAX = 5e-2


def _module():
    """The tiny task at the tiling stride, **with the causal norm** the causality tests require,
    and a posterior head moved off its zero start so the divergence is not identically zero."""
    module = make_task(tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, causal_norm=True))
    module.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in module.orig_model.posterior_head.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return module


def _inputs(module, batch=None):
    """The three input streams and the two extras, plus the scored anchor columns per sample."""
    batch = make_stub_batch(batch=2, seed=3) if batch is None else batch
    y_st, y_ph, u_stream, target_features, weight = model_inputs(module, batch)
    model = module.orig_model
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=DENSE_ANCHOR_GEOMETRY[0], anchor_stride=DENSE_ANCHOR_GEOMETRY[1])
    contributing = core.contributing_columns(model, weight, outputs)
    return (y_st, y_ph, u_stream), (target_features, weight), outputs, contributing


def _rows(module, readout: str, baseline: str, columns_per_sample: List[np.ndarray], **kwargs):
    """One integrated-gradient call at the given columns of the stub batch."""
    inputs, extra, _outputs, _contributing = _inputs(module)
    rows_inputs, rows_extra, columns, _sample = core.expand_rows(inputs, extra, columns_per_sample)
    wrapper = core.AnchorReadout(module.orig_model, core.ATTENTION_CELL, readout=readout, **kwargs).eval()
    return core.integrated_gradients(wrapper, rows_inputs, rows_extra, columns, baseline=baseline), rows_inputs, rows_extra, columns


# =================================================================================================
# The wrapper is the forward
# =================================================================================================
def test_the_wrapper_returns_the_forwards_own_quantities_at_each_rows_anchor() -> None:
    module = _module()
    model = module.orig_model
    inputs, extra, outputs, contributing = _inputs(module)
    columns = core.spread_columns(contributing, 3)
    rows_inputs, rows_extra, cols, sample = core.expand_rows(inputs, extra, columns)
    rows = torch.arange(cols.shape[0])
    anchors = outputs["anchor_index"][torch.as_tensor(sample), cols]
    args = (rows_extra[0], rows_extra[1], cols, torch.ones_like(cols))

    with torch.no_grad():
        kld = core.AnchorReadout(model, core.ATTENTION_CELL, readout=core.READOUT_KLD)(*rows_inputs, *args)
        expected = outputs["kld_per_t"][torch.as_tensor(sample), anchors]
        torch.testing.assert_close(kld, expected)

        coordinate = core.AnchorReadout(model, core.ATTENTION_CELL, readout=core.READOUT_MU_POST_DIM)(*rows_inputs, *args)
        torch.testing.assert_close(coordinate, outputs["mu_post"][torch.as_tensor(sample), anchors, 1])

        band = core.AnchorReadout(model, core.ATTENTION_CELL, readout=core.READOUT_LAG_BAND, lag_band=(1, 3))(*rows_inputs, *args)
        alpha = outputs["attn_weights"][torch.as_tensor(sample), anchors].mean(dim=1)[:, 1:4].sum(dim=-1)
        torch.testing.assert_close(band, alpha)

        # The mean-decoded gap is what the collection pass reports as ``mean_pred_gap``.
        gap = core.AnchorReadout(model, core.ATTENTION_CELL, readout=core.READOUT_PRED_GAP)(*rows_inputs, *args)
        mask, _coverage, _support = anchor_support(model, extra[1], outputs)
        target = model._build_forecast_target(extra[0], outputs["anchor_index"])
        scores, _ = mean_decoded_block(
            model, {"base": (outputs["mu_prior"], None), "full": (outputs["mu_post"], None)}, target, mask,
            anchors=outputs["anchor_index"], likelihood="gaussian_nll", persistence=outputs.get("persistence"),
        )
        expected_gap = (scores["base"] - scores["full"])[torch.as_tensor(sample), cols]
        torch.testing.assert_close(gap, expected_gap, rtol=1e-4, atol=1e-4)
    assert rows.shape[0] == sum(len(c) for c in columns)


def test_an_unknown_readout_or_baseline_is_refused_by_name() -> None:
    module = _module()
    with pytest.raises(ValueError, match="readout must be one of"):
        core.AnchorReadout(module.orig_model, core.ATTENTION_CELL, readout="entropy")
    with pytest.raises(ValueError, match="baseline must be one of"):
        core.baselines_for("mean", tuple(torch.zeros(1, 2, 3) for _ in range(3)))


# =================================================================================================
# The four structural properties
# =================================================================================================
@pytest.mark.parametrize("readout", [core.READOUT_KLD, core.READOUT_PRED_GAP, core.READOUT_LAG_BAND])
@pytest.mark.parametrize("baseline", list(core.BASELINES))
def test_no_attribution_reaches_a_step_after_the_anchor_or_a_cold_source_channel(readout, baseline) -> None:
    """Causality and the input gate, exactly. The conv-LSTM cell needs ``causal_norm`` for the
    first, as its own causality tests state; the tiny warm-up staircase makes the second
    non-vacuous, since most source channels are cold for the first few steps."""
    module = _module()
    _inputs_, _extra, _outputs, contributing = _inputs(module)
    columns = core.spread_columns(contributing, 2)
    result, *_ = _rows(module, readout, baseline, columns, lag_band=(0, 2))
    live = core.warm_from_step(module.orig_model, core.STREAM_SOURCE)
    assert live is not None and (live > 0).any(), "the tiny source has no cold channel to test"

    steps = np.arange(result.target.shape[1])
    for row in range(result.anchor.shape[0]):
        after = steps > int(result.anchor[row])
        assert after.any()
        assert np.abs(result.target[row][after]).max() == 0.0
        assert np.abs(result.source[row][after]).max() == 0.0
        cold = steps[:, None] < live[None, :]
        assert np.abs(result.source[row][cold]).max() == 0.0
        # And the attribution is not vacuously zero everywhere.
        if readout in core.MAIN_READOUTS:
            assert np.abs(result.source[row]).sum() > 0.0


@pytest.mark.parametrize("readout", list(core.TARGET_ONLY_READOUTS))
def test_a_target_only_readout_takes_exactly_no_source_attribution(readout) -> None:
    """The prior mean and the base block score read no source; the zero tie in the wrapper is
    what turns an autograd error about an unused tensor into the exact zero this asserts."""
    module = _module()
    _inputs_, _extra, _outputs, contributing = _inputs(module)
    columns = core.spread_columns(contributing, 2)
    result, *_ = _rows(module, readout, core.BASELINE_ALL_ZERO, columns)

    assert np.abs(result.source).max() == 0.0
    assert np.abs(result.target).sum() > 0.0


def test_integrated_gradients_are_complete_from_the_entry_point_and_the_jump_is_reported() -> None:
    """Completeness against $f(x) - f(x_0)$, where $x_0$ is the entry point; and the readout at
    the exact baseline is the null arm's own divergence, so the entry jump is measurable."""
    module = _module()
    model = module.orig_model
    inputs, extra, outputs, contributing = _inputs(module)
    columns = core.spread_columns(contributing, 3)
    result, rows_inputs, rows_extra, cols = _rows(module, core.READOUT_KLD, core.BASELINE_SOURCE_NULL, columns)

    total = result.target.sum(axis=(1, 2)) + result.source.sum(axis=(1, 2))
    difference = result.value_input - result.value_entry
    residual = np.abs(total - difference) / np.maximum(np.abs(difference), np.abs(result.value_input))
    assert np.median(residual) < COMPLETENESS_MEDIAN
    assert residual.max() < COMPLETENESS_MAX
    np.testing.assert_allclose(result.delta, total - difference, atol=1e-5)

    # The exact baseline value is the null arm's divergence at that anchor.
    with torch.no_grad():
        nulled = source_null_forward_outputs(model, dict(outputs), inputs[2])
        kld_null = model.kld_tensor(
            mu_prior=outputs["mu_prior"], logvar_prior=outputs["logvar_prior"],
            mu_post=nulled["mu_post"], logvar_post=nulled["logvar_post"],
        ).sum(dim=-1)
    _rows_inputs, _rows_extra, _cols, sample = core.expand_rows(inputs, extra, columns)
    expected = kld_null[torch.as_tensor(sample), torch.as_tensor(result.anchor)].numpy()
    np.testing.assert_allclose(result.value_baseline, expected, rtol=1e-4, atol=1e-5)
    # Target attribution is zero under the source-null baseline: the targets do not move.
    assert np.abs(result.target).max() == 0.0


def test_several_anchors_of_one_segment_attributed_in_one_call_equal_one_at_a_time() -> None:
    module = _module()
    _inputs_, _extra, _outputs, contributing = _inputs(module)
    both = core.spread_columns(contributing, 2)
    batched, *_ = _rows(module, core.READOUT_KLD, core.BASELINE_SOURCE_NULL, both)
    for position, column in enumerate(both[0]):
        single, *_ = _rows(module, core.READOUT_KLD, core.BASELINE_SOURCE_NULL, [np.array([column]), np.zeros(0, dtype=np.int64)])
        np.testing.assert_allclose(batched.source[position], single.source[0], atol=1e-5)
        assert batched.anchor[position] == single.anchor[0]


# =================================================================================================
# The reductions
# =================================================================================================
def test_the_lag_profile_reads_the_source_time_profile_at_the_anchor_minus_the_lag() -> None:
    profile = np.arange(2 * 6, dtype=np.float64).reshape(2, 6)
    lagged = core.lag_profile(profile, [4, 1], n_lags=4)

    np.testing.assert_array_equal(lagged[0], [4.0, 3.0, 2.0, 1.0])
    assert lagged[1, 0] == 7.0 and lagged[1, 1] == 6.0
    assert np.isnan(lagged[1, 2:]).all()


def test_band_sums_and_lag_band_groups_partition_the_axis_and_skip_out_of_range_positions() -> None:
    profile = np.ones((3, 5))
    groups = core.lag_band_groups({"near": (0, 1), "far": (2, 9)}, n_lags=5)
    sums = core.band_sums(profile, groups)

    assert list(groups["far"]) == [2, 3, 4]
    np.testing.assert_array_equal(sums["near"], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sums["far"], [3.0, 3.0, 3.0])
    assert np.isnan(core.band_sums(profile, {"none": np.array([9])})["none"]).all()


def test_the_agreement_is_nan_on_a_flat_or_empty_profile_and_finite_otherwise() -> None:
    lags = np.array([[1.0, -2.0, 3.0, 0.5], [1.0, 1.0, 1.0, 1.0], [np.nan, 1.0, 2.0, np.nan]])
    model = np.array([[0.1, 0.4, 0.4, 0.1], [0.1, 0.4, 0.4, 0.1], [0.5, 0.5, 0.0, 0.0]])
    fit = core.agreement(lags, model)

    assert np.isfinite(fit["lag_corr"][0]) and 0.0 <= fit["lag_js"][0] <= 1.0
    assert np.isnan(fit["lag_corr"][1]) and np.isfinite(fit["lag_js"][1])
    assert np.isnan(fit["lag_corr"][2]) and np.isnan(fit["lag_js"][2])


def test_the_channel_groups_come_off_the_declared_axis_map_per_stream() -> None:
    frame = pd.DataFrame({"stream": ["target", "target", "source"], "channel": [0, 2, 1], "band": ["a", "b", "a"]})
    groups = core.channel_groups_from_map(frame)

    assert list(groups["target"]["a"]) == [0] and list(groups["target"]["b"]) == [2]
    assert list(groups["source"]["a"]) == [1]
    assert core.channel_groups_from_map(None) == {}


def test_the_per_head_split_sums_to_the_readout_difference_along_the_source_null_path() -> None:
    module = _module()
    model = module.orig_model
    inputs, extra, _outputs, contributing = _inputs(module)
    columns = core.spread_columns(contributing, 2)
    rows_inputs, rows_extra, cols, _sample = core.expand_rows(inputs, extra, columns)
    wrapper = core.AnchorReadout(model, core.ATTENTION_CELL, readout=core.READOUT_KLD).eval()
    layer = core.layer_attribution(wrapper, rows_inputs, rows_extra, cols, baseline=core.BASELINE_SOURCE_NULL)
    result = core.integrated_gradients(wrapper, rows_inputs, rows_extra, cols, baseline=core.BASELINE_SOURCE_NULL)

    assert layer["per_unit"].shape == (cols.shape[0], int(model.posterior_head.num_heads))
    difference = result.value_input - result.value_entry
    np.testing.assert_allclose(layer["total"], difference, rtol=COMPLETENESS_MAX, atol=1e-3)


def test_ablating_every_source_step_up_to_the_anchor_reproduces_the_null_arm_and_the_rest_does_nothing() -> None:
    module = _module()
    model = module.orig_model
    inputs, extra, _outputs, contributing = _inputs(module)
    columns = core.spread_columns(contributing, 2)
    rows_inputs, rows_extra, cols, _sample = core.expand_rows(inputs, extra, columns)
    wrapper = core.AnchorReadout(model, core.ATTENTION_CELL, readout=core.READOUT_KLD).eval()
    with torch.no_grad():
        anchors = model(*rows_inputs, anchor_phase=0, anchor_stride=1)["anchor_index"][torch.arange(cols.shape[0]), cols]
        f_input = wrapper(*rows_inputs, *rows_extra, cols, torch.zeros_like(cols))
        f_null = wrapper(*core.baselines_for(core.BASELINE_SOURCE_NULL, rows_inputs), *rows_extra, cols, torch.zeros_like(cols))
    whole = {"whole": (0, int(rows_inputs[0].shape[1]))}
    deltas = core.ablate_lag_bands(wrapper, rows_inputs, rows_extra, cols, anchors.tolist(), whole)

    np.testing.assert_allclose(deltas["whole"], (f_null - f_input).numpy(), rtol=1e-4, atol=1e-5)
    # Zero to the float noise of a kernel re-run, not the exact zero the gradient check has: the
    # value is recomputed on a batch whose later steps changed, and a library kernel may sum in
    # another order for it.
    np.testing.assert_allclose(deltas["rest"], 0.0, atol=1e-5)


# =================================================================================================
# The selection
# =================================================================================================
def _segments(counts, segments: int = 3) -> pd.DataFrame:
    rows = []
    for name, count in counts.items():
        for index in range(count):
            for segment in range(segments):
                rows.append({"guid": f"{name}{index:02d}", "epoch": -7200.0 + STRIDE_S * segment,
                             labels.CLASS_COLUMN: name, labels.SUBGROUP_COLUMN: f"{name}_cs"})
    return pd.DataFrame(rows)


def test_the_selection_draws_one_middle_segment_per_recording_class_balanced_and_capped() -> None:
    segments = _segments({"healthy": 6, "acidosis": 3, "hie": 2})
    index_map = {(row["guid"], int(row["epoch"])): index for index, (_, row) in enumerate(segments.iterrows())}

    chosen, accounting = attribution_pass.select_segments(segments, index_map, cap=6, seed=3)

    assert len(chosen) == 6 and chosen["guid"].is_unique
    assert dict(chosen[labels.CLASS_COLUMN].value_counts()) == {"healthy": 2, "acidosis": 2, "hie": 2}
    assert (chosen["epoch"] == -7200.0 + STRIDE_S).all()
    assert list(chosen["dataset_index"]) == sorted(chosen["dataset_index"])
    assert accounting["segment_cap"] == 6 and accounting["n_segments_selected"] == 6
    again, _ = attribution_pass.select_segments(segments, index_map, cap=6, seed=3)
    assert list(again["guid"]) == list(chosen["guid"])


def test_completeness_counts_the_segments_a_window_could_hold_against_the_ones_it_does() -> None:
    """Six-segment tiling over the last two hours at a twenty-minute stride is seven slots; a
    recording with one slot missing scores below the full one, and the window is the bound when
    the run sets it and the recording's own span otherwise."""
    full = [-7200.0 + STRIDE_S * k for k in range(7)]
    holed = [value for value in full if value != -3600.0]
    index_map = {("full", int(e)): i for i, e in enumerate(full)}
    index_map.update({("holed", int(e)): 100 + i for i, e in enumerate(holed)})
    index_map[("old", -20000)] = 200
    index_map[("old", -20000 + int(STRIDE_S))] = 201

    bounded = attribution_pass.recording_completeness(index_map, stride_s=STRIDE_S, window_hours=2.0).set_index("guid")
    unbounded = attribution_pass.recording_completeness(index_map, stride_s=STRIDE_S, window_hours=None).set_index("guid")

    assert bounded.loc["full", "coverage"] == 1.0 and bounded.loc["full", "n_expected"] == 7
    assert bounded.loc["holed", "coverage"] == pytest.approx(6 / 7)
    assert bounded.loc["old", "n_in_window"] == 0 and bounded.loc["old", "coverage"] == 0.0
    # Over its own span the old recording tiles perfectly, and the holed one still shows the hole.
    assert unbounded.loc["old", "coverage"] == 1.0
    assert unbounded.loc["holed", "coverage"] == pytest.approx(6 / 7)


def test_the_traced_recording_is_the_most_complete_of_its_class_and_ties_break_by_name() -> None:
    segments = _segments({"healthy": 3, "acidosis": 2, "hie": 2})
    index_map = {(row["guid"], int(row["epoch"])): index for index, (_, row) in enumerate(segments.iterrows())}
    # Knock the middle segment out of every recording but one per class; drop a healthy one to
    # a single segment, which makes it ineligible however complete it is.
    for guid in ("healthy00", "healthy02", "acidosis01", "hie00"):
        index_map.pop((guid, int(-7200.0 + STRIDE_S)))
    index_map.pop(("healthy02", int(-7200.0)))

    chosen, accounting = attribution_pass.select_trace_recordings(segments, index_map, stride_s=STRIDE_S, window_hours=None)

    assert list(chosen["guid"]) == ["hie01", "acidosis00", "healthy01"]
    assert (chosen["coverage"] == 1.0).all()
    assert accounting["classes"]["healthy"] == {"n_recordings": 3, "n_eligible": 2, "n_selected": 1, "coverage": [1.0]}
    assert accounting["window_hours"] is None and accounting["segment_stride_s"] == STRIDE_S
    # With every recording equally complete the identifier decides, so two runs agree.
    tied, _ = attribution_pass.select_trace_recordings(
        segments, {(row["guid"], int(row["epoch"])): index for index, (_, row) in enumerate(segments.iterrows())},
        stride_s=STRIDE_S, window_hours=None,
    )
    assert list(tied["guid"]) == ["hie00", "acidosis00", "healthy00"]


# =================================================================================================
# The analysis, end to end through a stub loader
# =================================================================================================
class _StubDataset:
    """A dataset of stub segments that lists its own recordings, as the real one does."""

    def __init__(self, guids: List[str], epochs: List[float]) -> None:
        self.guids, self.epochs = list(guids), list(epochs)

    def __len__(self) -> int:
        return len(self.guids)

    def __getitem__(self, index: int) -> int:
        return int(index)

    def get_the_lists(self):
        return self.guids, self.epochs, [None] * len(self.guids)


def _collate_factory(dataset: _StubDataset):
    def _collate(items: List[int]):
        batch = make_stub_batch(batch=len(items), seed=int(items[0]))
        batch.guid = [dataset.guids[index] for index in items]
        batch.epoch = torch.tensor([dataset.epochs[index] for index in items])
        return batch
    return _collate


def _stub_population():
    """Three classes, two recordings each, two segments per recording, plus a one-segment one."""
    guids, epochs, classes = [], [], []
    for name in ("healthy", "acidosis", "hie"):
        for recording in range(2):
            for segment in range(2):
                guids.append(f"{name}{recording}")
                epochs.append(-7200.0 + STRIDE_S * segment)
                classes.append(name)
    guids.append("lonely"); epochs.append(-5000.0); classes.append("healthy")
    return guids, epochs, classes


#: The tiny lag window's four contiguous bands, in the shipped names.
TINY_BANDS = {"anchor": [0, 1], "near": [2, 4], "mid": [5, 6], "far": [7, 8]}


def test_the_analysis_attributes_a_balanced_draw_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core, "IG_STEPS", 8)
    guids, epochs, classes = _stub_population()
    dataset = _StubDataset(guids, epochs)
    loader = types.SimpleNamespace(dataset=dataset, collate_fn=_collate_factory(dataset), batch_size=2)
    per_sample = pd.DataFrame(
        {"guid": guids, "epoch": epochs, labels.CLASS_COLUMN: classes,
         labels.SUBGROUP_COLUMN: [f"{name}_cs" for name in classes]}
    )
    module = _module()
    context = AnalysisContext(
        collection=types.SimpleNamespace(per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={}),
        config={}, task=module, loader=loader,
    )
    plt.close("all")

    result = analysis.run_attribution_analysis(
        context, eval_config={"seed": 0, "caps": {core.CAP_NAME: 3}, "occlusion_bands": TINY_BANDS}, output_dir=tmp_path
    )

    assert result["failures"] == [], result["failures"]
    assert result["n_samples"] == 3 and result["plan"]["capped"] is True and result["plan"]["cap"] == 3
    assert result["composition"]["n_segments_by_class"] == {"healthy": 1, "acidosis": 1, "hie": 1}
    checks = result["checks"]
    assert checks["after_anchor_max_abs"] == 0.0 and checks["gated_off_max_abs"] == 0.0
    assert checks["target_only"]["source_attr_max_abs"] == 0.0
    assert set(result["methods"]) >= {"IntegratedGradients", "DeepLift", "GradientShap"}
    assert result["joined"] == {"channel_map": False, "occlusion_summary": False, "spectral_skill_bands": False}
    assert len(result["traces"]) == 3 and result["cost"]["n_rows"] > 0
    json.dumps(json_safe(result), allow_nan=False)

    directory = tmp_path / core.ANALYSIS_DIRNAME
    for name in result["files"]:
        assert (directory / name).is_file(), name
    for stem in (core.MAP_FIGURE, core.LAG_PROFILE_FIGURE, core.BAND_FIGURE, core.LAYER_FIGURE, core.NULL_FIGURE):
        assert (directory / f"{stem}.pdf").is_file()
    rows = pd.read_csv(directory / core.ROWS_FILENAME)
    assert set(rows["readout"]) == {core.READOUT_KLD, core.READOUT_PRED_GAP, core.READOUT_LAG_BAND, core.READOUT_KLD_DIM}
    assert set(rows[rows["readout"] == core.READOUT_LAG_BAND]["band"]) == set(TINY_BANDS)
    assert rows["guid"].nunique() == 3
    for name in TINY_BANDS:
        assert f"lagband_{name}" in rows.columns and f"ablation_{name}" in rows.columns
    with np.load(directory / core.VECTORS_FILENAME) as handle:
        assert handle["lag_profile"].shape == (len(rows), int(module.orig_model.lag_attn.L))
        assert handle["layer_per_unit"].shape[1] == int(module.orig_model.posterior_head.num_heads)
    manifest = pd.read_csv(directory / attribution_pass.TRACE_MANIFEST_FILENAME)
    assert list(manifest.columns) == list(attribution_pass.TRACE_MANIFEST_COLUMNS)
    for _, row in manifest.iterrows():
        assert (directory / row["figure_file"]).is_file() and (directory / row["arrays_file"]).is_file()
        assert row[labels.SUBGROUP_COLUMN] in row["figure_file"]
        assert row["figure_file"].startswith(f"{core.TRACE_DIRNAME}/{row[labels.CLASS_COLUMN]}/")
    grouped = result["grouped_frames"][0]
    assert (tmp_path / grouped["path"]).is_file()
    assert plt.get_fignums() == []


def test_a_pass_with_no_model_records_a_skip(tmp_path) -> None:
    context = AnalysisContext(
        collection=types.SimpleNamespace(per_sample=pd.DataFrame({"guid": ["g0"], "epoch": [-1000.0]}),
                                         per_anchor=pd.DataFrame(), record={}, retained={}, results={}),
        config={},
    )

    result = analysis.run_attribution_analysis(context, eval_config={"seed": 0}, output_dir=tmp_path)

    assert result["skipped"] is True and result["n_samples"] is None
    assert "no model" in result["reason"]


def test_the_stub_loader_lists_the_recordings_the_selection_resolves() -> None:
    """Non-vacuity for the end-to-end test: the mapping the selection goes through is the one the
    dataset lists, so a segment the analysis attributes is one the loader can serve."""
    guids, epochs, _classes = _stub_population()
    dataset = _StubDataset(guids, epochs)
    loader = types.SimpleNamespace(dataset=dataset, collate_fn=_collate_factory(dataset), batch_size=2)
    index_map = dataset_index_map(loader)
    assert len(index_map) == len(guids)


# =================================================================================================
# Against the real run
# =================================================================================================
@pytest.mark.slow
def test_a_real_run_attributes_every_class_with_the_structural_checks_exactly_zero(collected_run) -> None:
    result = collected_run["summary"]["results"][core.ANALYSIS_DIRNAME]

    assert result["failures"] == []
    assert result["n_samples"] > 0
    assert result["checks"]["after_anchor_max_abs"] == 0.0
    assert result["checks"]["gated_off_max_abs"] == 0.0
    assert result["checks"]["target_only"]["source_attr_max_abs"] == 0.0
    assert result["joined"]["channel_map"] is True
    directory = Path(collected_run["results_dir"]) / core.ANALYSIS_DIRNAME
    for name in result["files"]:
        assert (directory / name).is_file(), name
    bands = pd.read_csv(directory / core.BANDS_FILENAME)
    assert not bands.empty and set(bands["stream"]) == set(core.STREAMS)
