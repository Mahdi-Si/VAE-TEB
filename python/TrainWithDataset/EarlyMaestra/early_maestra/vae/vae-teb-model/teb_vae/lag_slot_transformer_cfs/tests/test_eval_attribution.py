r"""The Captum attributions of this cell: the wrapper on the anchor axis, the four properties, the stage.

What is this cell's own, and therefore asserted here rather than left to the family's tests: the
wrapper selects the anchor axis rather than gathering a dense one, and its lag readout is the
proposal norm on a band; the four structural properties hold exactly on this architecture, whose
transformer target encoder needs no causal-norm switch; the per-lag split on the proposal head's
output is complete along the source-null path; and the stage records a skip by name when no
segment carries a class, writes the family's tables and figures otherwise, and emits no key the
acceptance gate refuses.
"""
from __future__ import annotations

import json
from typing import Any, List

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import attribution_pass
from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.report_seam import json_safe
from teb_vae.lag_slot_transformer_cfs.eval import attribution as stage
from teb_vae.lag_slot_transformer_cfs.eval.verify import FORBIDDEN_KEYS

from .conftest import TINY_SEQ_LEN, build_tiny_model
from .test_eval_recording_traces import _StubDataset, _Task, _loader, _population, stub_batch

#: Seconds between consecutive stored segments in the stub batches.
STRIDE_S = 1200.0

#: How far a completeness residual may sit, as a share of the readout, on this cell's tiny model.
COMPLETENESS_MAX = 1e-3


def _task(**overrides: Any) -> _Task:
    """The tiny model in evaluation mode, its proposal head moved off its zero start.

    Args:
        **overrides: Constructor keywords to replace, for the ablated arms.
    """
    model = build_tiny_model(**overrides)
    torch.manual_seed(2)
    with torch.no_grad():
        for parameter in model.proposal_head.output_proj.parameters():
            parameter.add_(0.2 * torch.randn_like(parameter))
    return _Task(model)


def _inputs(task: _Task):
    """The input streams, the extras, the retained forward and the scored anchor columns."""
    batch = stub_batch(["A", "B"], [-3600.0, -3600.0 + STRIDE_S], class_code=3, seed=4)
    y_st, y_ph = task._build_target_streams(batch)
    u_stream = task._build_source_stream(batch)
    target_features, weight = task._build_raw_target(batch)
    model = task.orig_model
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, return_proposals=True)
    contributing = core.contributing_columns(model, weight, outputs)
    return (y_st, y_ph, u_stream), (target_features, weight), outputs, contributing


def _attribute(task: _Task, readout: str, baseline: str, per_segment: int = 2, **kwargs):
    """One integrated-gradient call at evenly spread scored anchors of the stub batch."""
    inputs, extra, _outputs, contributing = _inputs(task)
    columns = core.spread_columns(contributing, per_segment)
    rows_inputs, rows_extra, cols, _sample = core.expand_rows(inputs, extra, columns)
    wrapper = core.AnchorReadout(task.orig_model, core.SLOT_CELL, readout=readout, **kwargs).eval()
    return core.integrated_gradients(wrapper, rows_inputs, rows_extra, cols, baseline=baseline), wrapper, rows_inputs, rows_extra, cols


# =================================================================================================
# The wrapper
# =================================================================================================
def test_the_wrapper_selects_the_anchor_axis_and_reads_the_proposal_norm_on_a_band() -> None:
    """Every latent tensor here is already on the anchor axis, so the wrapper selects a column
    rather than gathering a stored step; the band readout is the proposal norm summed over it."""
    task = _task()
    model = task.orig_model
    inputs, extra, outputs, contributing = _inputs(task)
    columns = core.spread_columns(contributing, 3)
    rows_inputs, rows_extra, cols, sample = core.expand_rows(inputs, extra, columns)
    args = (rows_extra[0], rows_extra[1], cols, torch.zeros_like(cols))
    who = torch.as_tensor(sample)

    with torch.no_grad():
        kld = core.AnchorReadout(model, core.SLOT_CELL, readout=core.READOUT_KLD)(*rows_inputs, *args)
        torch.testing.assert_close(kld, outputs["kld_per_anchor"][who, cols])
        band = core.AnchorReadout(model, core.SLOT_CELL, readout=core.READOUT_LAG_BAND, lag_band=(1, 2))(*rows_inputs, *args)
        expected = outputs["mean_proposals"][who, cols][:, 1:3].norm(dim=-1).sum(dim=-1)
        torch.testing.assert_close(band, expected)
        coordinate = core.AnchorReadout(model, core.SLOT_CELL, readout=core.READOUT_KLD_DIM)(*rows_inputs, *args)
        torch.testing.assert_close(coordinate, outputs["kld_per_anchor_dim"][who, cols, 0])


def test_the_model_lag_readout_is_the_masked_proposal_norm_per_row() -> None:
    """The profile the agreement statistic compares against is the traces' own: the proposal norm
    at the anchor, absent where the lag carried no available channel."""
    task = _task()
    model = task.orig_model
    inputs, extra, outputs, contributing = _inputs(task)
    columns = core.spread_columns(contributing, 2)
    _rows_inputs, _rows_extra, cols, sample = core.expand_rows(inputs, extra, columns)

    profile = core.model_lag_readout(model, core.SLOT_CELL, outputs, torch.as_tensor(sample), cols)

    assert profile.shape == (cols.shape[0], int(model.n_lags))
    valid = outputs["lag_valid"][torch.as_tensor(sample), cols].numpy()
    assert (np.isnan(profile) == ~valid).all()


# =================================================================================================
# The four structural properties
# =================================================================================================
@pytest.mark.parametrize("readout", [core.READOUT_KLD, core.READOUT_PRED_GAP, core.READOUT_LAG_BAND])
@pytest.mark.parametrize("baseline", list(core.BASELINES))
def test_no_attribution_reaches_a_step_after_the_anchor_or_a_cold_source_channel(readout, baseline) -> None:
    """Causality and the input gate, exactly, on an architecture whose encoders are step-wise causal
    by construction; the tiny source warm-up staircase makes the gate half non-vacuous."""
    task = _task()
    result, *_ = _attribute(task, readout, baseline, lag_band=(0, 1))
    live = core.warm_from_step(task.orig_model, core.STREAM_SOURCE)
    assert live is not None and (live > 0).any()

    steps = np.arange(result.target.shape[1])
    for row in range(result.anchor.shape[0]):
        after = steps > int(result.anchor[row])
        assert np.abs(result.target[row][after]).max() == 0.0
        assert np.abs(result.source[row][after]).max() == 0.0
        cold = steps[:, None] < live[None, :]
        assert np.abs(result.source[row][cold]).max() == 0.0


@pytest.mark.parametrize("readout", list(core.TARGET_ONLY_READOUTS))
def test_a_target_only_readout_takes_exactly_no_source_attribution(readout) -> None:
    """The prior reads the metadata clock and the target state and no source value."""
    result, *_ = _attribute(_task(), readout, core.BASELINE_ALL_ZERO)

    assert np.abs(result.source).max() == 0.0
    assert np.abs(result.target).sum() > 0.0


def test_integrated_gradients_are_complete_along_the_source_null_path() -> None:
    """The source pathway here is a pointwise encoder into a small multilayer perceptron, so the
    path from the null arm is smooth and the residual is far below the family's tolerance."""
    result, *_ = _attribute(_task(), core.READOUT_KLD, core.BASELINE_SOURCE_NULL, per_segment=3)

    total = result.target.sum(axis=(1, 2)) + result.source.sum(axis=(1, 2))
    difference = result.value_input - result.value_entry
    residual = np.abs(total - difference) / np.maximum(np.abs(difference), np.abs(result.value_input))
    assert residual.max() < COMPLETENESS_MAX
    assert np.abs(result.target).max() == 0.0
    assert np.abs(result.value_entry - result.value_baseline).max() < 1e-2


def test_the_per_lag_split_on_the_proposal_head_is_complete_and_has_one_unit_per_lag() -> None:
    """The layer attribution here is on the proposal head's output, through the summation and the
    limiter, and its per-lag sums recover the readout difference along the source-null path."""
    task = _task()
    result, wrapper, rows_inputs, rows_extra, cols = _attribute(task, core.READOUT_KLD, core.BASELINE_SOURCE_NULL)

    layer = core.layer_attribution(wrapper, rows_inputs, rows_extra, cols, baseline=core.BASELINE_SOURCE_NULL)

    assert layer["per_unit"].shape == (cols.shape[0], int(task.orig_model.n_lags))
    np.testing.assert_allclose(layer["total"], result.value_input - result.value_entry, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("readout", [core.READOUT_KLD, core.READOUT_PRED_GAP])
def test_an_ablated_coordinate_takes_exactly_no_attribution_on_either_stream(readout) -> None:
    """The ablation is the first step of the forward, so the integrated gradient with respect to
    the ablated coordinate is exactly zero, while its neighbours still carry attribution."""
    task = _task(zero_fhr_scattering_s0=True, zero_up_scattering_s0=True)
    result, *_ = _attribute(task, readout, core.BASELINE_ALL_ZERO)

    assert np.abs(result.target[:, :, 0]).max() == 0.0
    assert np.abs(result.source[:, :, 0]).max() == 0.0
    assert np.abs(result.target[:, :, 1:]).sum() > 0.0


def test_the_stage_marks_the_ablated_coordinates_and_reads_their_attribution_back(tmp_path, monkeypatch) -> None:
    """The marker table and the check travel in the block; the read-back is exactly zero."""
    monkeypatch.setattr(core, "IG_STEPS", 4)
    guids, epochs, codes = _population()
    identities = pd.DataFrame(
        {"guid": guids, "epoch": epochs, labels.CLASS_COLUMN: [labels.CLASS_NAMES[code] for code in codes],
         labels.SUBGROUP_COLUMN: ["hie_cs"] * len(guids)}
    )

    block = stage.run_attribution(
        _task(zero_fhr_scattering_s0=True), _loader(_StubDataset(guids, epochs, codes)), identities,
        config={}, eval_config={"seed": 0, "caps": {core.CAP_NAME: 3}, "occlusion_bands": {"near": [0, 2]}},
        results_dir=tmp_path, geometry_record={"t": TINY_SEQ_LEN},
    )

    assert block["status"] == stage.STATUS_ATTRIBUTED
    ablated = block["ablated_inputs"]
    assert [entry["field"] for entry in ablated["coordinates"]] == ["fhr_st"]
    assert ablated["coordinates"][0]["status"] == stage.STATUS_ABLATED
    assert ablated["max_abs_attribution"] == 0.0
    assert block["checks"]["ablated_input_max_abs"] == 0.0
    assert stage.ABLATED_INPUTS_FILENAME in block["files"]
    table = pd.read_csv(tmp_path / core.ANALYSIS_DIRNAME / stage.ABLATED_INPUTS_FILENAME)
    assert list(table["status"]) == [stage.STATUS_ABLATED]
    # And a model with no ablation carries an empty marker rather than none.
    plain = stage.ablated_input_check(build_tiny_model(), tmp_path / core.ANALYSIS_DIRNAME)
    assert plain["coordinates"] == [] and plain["max_abs_attribution"] is None


def test_the_target_only_arm_has_no_layer_to_split_and_no_lag_readout() -> None:
    """No source pathway means no proposal head and no proposal norm: both absent, never zeros."""
    model = build_tiny_model(source_disabled=True)
    task = _Task(model)
    inputs, extra, outputs, contributing = _inputs(task)
    columns = core.spread_columns(contributing, 1)
    rows_inputs, rows_extra, cols, sample = core.expand_rows(inputs, extra, columns)
    wrapper = core.AnchorReadout(model, core.SLOT_CELL, readout=core.READOUT_KLD).eval()

    layer = core.layer_attribution(wrapper, rows_inputs, rows_extra, cols, baseline=core.BASELINE_SOURCE_NULL)
    profile = core.model_lag_readout(model, core.SLOT_CELL, outputs, torch.as_tensor(sample), cols)

    assert layer["per_unit"].shape == (0, 0)
    assert np.isnan(profile).all()


# =================================================================================================
# The stage
# =================================================================================================
def _walk(block: Any, found: List[str], path: str = "") -> None:
    """Collect every key of a nested block that the acceptance gate refuses."""
    if isinstance(block, dict):
        for key, value in block.items():
            if key in FORBIDDEN_KEYS:
                found.append(f"{path}/{key}")
            _walk(value, found, f"{path}/{key}")
    elif isinstance(block, list):
        for value in block:
            _walk(value, found, path)


def test_the_stage_attributes_a_balanced_draw_end_to_end(tmp_path, monkeypatch) -> None:
    """Selection, the sequential subset loader, the identity check, every Captum call, the tables,
    the figures and the trace, on a stub population with every class present."""
    monkeypatch.setattr(core, "IG_STEPS", 8)
    guids, epochs, codes = _population()
    dataset = _StubDataset(guids, epochs, codes)
    identities = pd.DataFrame(
        {"guid": guids, "epoch": epochs, labels.CLASS_COLUMN: [labels.CLASS_NAMES[code] for code in codes],
         labels.SUBGROUP_COLUMN: ["hie_cs"] * len(guids)}
    )

    block = stage.run_attribution(
        _task(), _loader(dataset), identities, config={},
        eval_config={"seed": 0, "caps": {core.CAP_NAME: 3}, "occlusion_bands": {"near": [0, 2], "far": [3, 4]}},
        results_dir=tmp_path, geometry_record={"t": TINY_SEQ_LEN},
    )

    assert block["status"] == stage.STATUS_ATTRIBUTED, block
    assert block["failures"] == []
    assert block["n_samples"] == 3
    assert block["composition"]["n_segments_by_class"] == {"healthy": 1, "acidosis": 1, "hie": 1}
    assert block["checks"]["after_anchor_max_abs"] == 0.0
    assert block["checks"]["gated_off_max_abs"] == 0.0
    assert block["checks"]["target_only"]["source_attr_max_abs"] == 0.0
    assert block["channel_map"]["skipped"] is True
    assert block["lag_qualification"] == core.SLOT_CELL.lag_qualification
    found: List[str] = []
    _walk(json_safe(block), found)
    assert found == []
    json.dumps(json_safe(block), allow_nan=False)

    directory = tmp_path / core.ANALYSIS_DIRNAME
    for name in block["files"]:
        assert (directory / name).is_file(), name
    rows = pd.read_csv(directory / core.ROWS_FILENAME)
    assert set(rows[rows["readout"] == core.READOUT_LAG_BAND]["band"]) == {"near", "far"}
    # The per-step score rows carry their horizon step in the same column, first and last.
    horizons = core.horizon_steps(_task().orig_model)
    assert set(rows[rows["readout"] == core.READOUT_NLL_HORIZON]["band"]) == {f"h{step}" for step in horizons.values()}
    assert set(rows["readout"]) >= set(core.MAIN_READOUTS)
    blocks = pd.read_csv(directory / core.BLOCKS_FILENAME)
    assert set(blocks["block"]) == set(core.BLOCKS)
    for stem in (core.BLOCK_FIGURE, core.HORIZON_FIGURE):
        assert (directory / f"{stem}.pdf").is_file()
    with np.load(directory / core.VECTORS_FILENAME) as handle:
        assert handle["layer_per_unit"].shape[1] == int(_task().orig_model.n_lags)
    manifest = pd.read_csv(directory / attribution_pass.TRACE_MANIFEST_FILENAME)
    assert len(manifest) == 3
    for _, row in manifest.iterrows():
        assert (directory / row["figure_file"]).is_file()
        assert row[labels.SUBGROUP_COLUMN] in row["figure_file"]


def test_the_stage_records_a_skip_when_no_segment_carries_a_class(tmp_path) -> None:
    """What the pass produces when the delta does not name the target field."""
    identities = pd.DataFrame(
        {"guid": ["A", "A"], "epoch": [-3600.0, -2400.0],
         labels.CLASS_COLUMN: [None, None], labels.SUBGROUP_COLUMN: ["hie_cs", "hie_cs"]}
    )

    block = stage.run_attribution(
        _task(), _loader(_StubDataset(["A", "A"], [-3600.0, -2400.0], [1, 1])), identities,
        config={}, eval_config={"seed": 0}, results_dir=tmp_path,
    )

    assert block["status"] == stage.STATUS_SKIPPED
    assert "load_fields" in block["reason"]
    assert not (tmp_path / core.ANALYSIS_DIRNAME).exists()


def test_no_name_this_stage_writes_is_one_of_the_keys_this_package_refuses() -> None:
    """The summary walker refuses attention-shaped keys at any depth; the constants below are the
    keys the block, the plan and the tables carry."""
    names = {*core.READOUTS, *core.BASELINES, *core.STREAMS, core.CAP_NAME, core.ANALYSIS_DIRNAME}
    names |= {panel.vector for panel in core.TRACE_PANELS if hasattr(panel, "vector")}
    names |= set(core.TRACE_LAG_PROFILES)
    names |= set(attribution_pass.RECORDING_VALUE_COLUMNS)

    assert not names & set(FORBIDDEN_KEYS)
