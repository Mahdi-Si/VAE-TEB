r"""The per-recording traces of this cell: the gather on the anchor axis, the identities, the stage.

What is this cell's own, and therefore asserted here rather than left to the family's tests: the
gather is a **selection** of the valid anchor slots rather than a gather over a dense time axis,
because every latent tensor of this architecture already lives on the anchor axis; the lag family
is the proposal norm masked to the lags that carried a channel, present on the explicit-sum arm
with a source pathway and absent -- never zero-filled -- on the target-only arm; the per-anchor
scores are the single-draw block scores off the forward's own forecasts; and the stage records a
skip by name when no segment carries a class, which is what the pass produces when the override
delta does not name ``target``.

The end-to-end test drives the stage through a stub dataset that lists its own recordings and
collates into tiny stub batches, so the whole path -- identities, selection, the sequential subset
loader, the identity check, the forward, the files and the figures -- runs on CPU in seconds.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from teb_vae.lag_attn_cfs.eval import traces
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_slot_transformer_cfs.eval import recording_traces as stage

from .conftest import (
    DECLARED_SOURCE_PH,
    DECLARED_ST,
    DECLARED_TARGET_PH,
    TINY_FLOOR,
    TINY_MODEL_HORIZON,
    TINY_SEQ_LEN,
    build_tiny_model,
)

#: Seconds between consecutive stored segments in the stub batches.
STRIDE_S = 1200.0

#: The decimated step every stub batch leaves invalid, inside the decoded range.
GAP_STEP = TINY_FLOOR + 3


def stub_batch(guids: List[str], epochs: List[float], *, class_code: int, seed: int = 0):
    """A batch at the tiny declared widths with every field the task and the identities read."""
    generator = torch.Generator().manual_seed(seed)
    batch = len(guids)
    weight = torch.ones(batch, TINY_SEQ_LEN)
    weight[:, GAP_STEP] = 0.0
    return types.SimpleNamespace(
        fhr_st=torch.randn(batch, TINY_SEQ_LEN, DECLARED_ST, generator=generator),
        fhr_ph=torch.randn(batch, TINY_SEQ_LEN, DECLARED_TARGET_PH, generator=generator),
        up_st=torch.randn(batch, TINY_SEQ_LEN, DECLARED_ST, generator=generator),
        up_ph=torch.randn(batch, TINY_SEQ_LEN, DECLARED_SOURCE_PH, generator=generator),
        weight=weight,
        target=float(class_code) * weight,
        guid=list(guids),
        epoch=torch.tensor([float(value) for value in epochs]),
        source_file_basename=["hie_cs.hdf5"] * batch,
    )


class _Task:
    """The smallest task the stage drives: the net, the device, the objective and the builders."""

    device = "cpu"
    hparams: Dict[str, Any] = {"likelihood": "gaussian_nll"}

    def __init__(self, model) -> None:
        """Wrap the net in evaluation mode."""
        self.orig_model = model.eval()

    def transfer_batch_to_device(self, batch, *_args, **_kwargs):
        """Identity: the stub batches are already where they need to be."""
        return batch

    def _build_target_streams(self, batch):
        return batch.fhr_st, batch.fhr_ph

    def _build_source_stream(self, batch):
        return torch.cat([batch.up_st, batch.up_ph], dim=-1)

    def _build_raw_target(self, batch):
        return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1), batch.weight


def _forward(task: _Task, batch):
    """One dense forward with the proposals retained, plus what the gather needs beside it."""
    model = task.orig_model
    y_st, y_ph = task._build_target_streams(batch)
    u_stream = task._build_source_stream(batch)
    target_features, weight = task._build_raw_target(batch)
    outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, return_proposals=True)
    target = model._build_forecast_target(target_features, outputs["anchor_index"])
    return outputs, target, weight


def _rows(guid: str, epochs: List[float]) -> pd.DataFrame:
    return pd.DataFrame({"guid": [guid] * len(epochs), "epoch": [float(value) for value in epochs]})


# =================================================================================================
# The identities
# =================================================================================================
def test_the_identity_of_a_batch_recovers_the_class_from_the_scaled_target() -> None:
    """The class is the ratio against the weight, and the subgroup is the shard basename."""
    batch = stub_batch(["A", "B"], [-3600.0, -2400.0], class_code=2)

    identity = stage.batch_identity(batch, 2)

    assert identity["guid"] == ["A", "B"]
    assert identity["epoch"] == [-3600.0, -2400.0]
    assert identity[labels.CLASS_COLUMN] == ["acidosis", "acidosis"]
    assert identity[labels.SUBGROUP_COLUMN] == ["hie_cs", "hie_cs"]


def test_a_batch_without_a_target_is_unlabelled_rather_than_guessed() -> None:
    """A loader not asked for the target yields no class, which the stage then records as a skip."""
    batch = stub_batch(["A"], [-3600.0], class_code=3)
    del batch.target

    identity = stage.batch_identity(batch, 1)

    assert identity[labels.CLASS_COLUMN] == [None]
    frame = stage.identity_frame([{"identity": identity}, {"identity": stage.batch_identity(batch, 1)}])
    assert len(frame) == 2 and frame[labels.CLASS_COLUMN].isna().all()


# =================================================================================================
# The gather
# =================================================================================================
def test_the_gather_selects_the_valid_anchor_slots_and_masks_the_proposal_map() -> None:
    """Every array is on the anchor axis at the valid slots, and the lag map is masked to the lags
    that carried a channel."""
    task = _Task(build_tiny_model())
    batch = stub_batch(["REC", "REC"], [-3600.0, -3600.0 + STRIDE_S], class_code=3)
    with torch.no_grad():
        outputs, target, weight = _forward(task, batch)
        segments = stage.gather_segment_traces(
            task.orig_model, outputs, target, weight, _rows("REC", [-3600.0, -3600.0 + STRIDE_S]),
            likelihood="gaussian_nll", clinical_class="hie", subgroup="hie_cs",
        )

    n_anchors = TINY_SEQ_LEN - TINY_MODEL_HORIZON - TINY_FLOOR
    n_lags = int(task.orig_model.n_lags)
    assert len(segments) == 2
    for segment in segments:
        assert list(segment.anchor) == list(range(TINY_FLOOR, TINY_SEQ_LEN - TINY_MODEL_HORIZON))
        assert segment.vectors["mu_post"].shape == (n_anchors, task.orig_model.d_z)
        assert segment.vectors["update_mean"].shape == segment.vectors["mu_post"].shape
        assert segment.vectors["proposal_lag_map"].shape == (n_anchors, n_lags)
        lag_valid = outputs["lag_valid"][0].numpy()
        # Masked to the lags that carried a channel: NaN exactly where none did.
        assert (np.isnan(segment.vectors["proposal_lag_map"]) == ~lag_valid).all()
        assert (segment.vectors["proposal_lag_map"][lag_valid] >= 0.0).all()
        # The divergence is the sum of its per-coordinate split, at every anchor.
        np.testing.assert_allclose(
            segment.vectors["kld_per_dim"].sum(axis=-1), segment.scalars["kld_per_t"], rtol=1e-5, atol=1e-6
        )
        # The single-draw gap is the two block scores' difference, and the invalid step is not scored.
        np.testing.assert_allclose(
            segment.scalars["pred_gap"], segment.scalars["nll_base_block"] - segment.scalars["nll_full_block"]
        )
        assert not segment.contributing[segment.anchor == GAP_STEP].any()
        assert segment.contributing.any()
        assert "cancellation_ratio_mean" in segment.scalars
        assert np.isfinite(segment.scalars["proposal_argmax_lag"]).all()


def test_the_target_only_arm_traces_the_latent_without_a_lag_family() -> None:
    """No source pathway means no proposal map: absent from the trace, never a column of zeros."""
    task = _Task(build_tiny_model(source_disabled=True))
    batch = stub_batch(["REC"], [-3600.0], class_code=1)
    with torch.no_grad():
        outputs, target, weight = _forward(task, batch)
        segments = stage.gather_segment_traces(
            task.orig_model, outputs, target, weight, _rows("REC", [-3600.0]),
            likelihood="gaussian_nll", clinical_class="healthy", subgroup=None,
        )

    segment = segments[0]
    assert "proposal_lag_map" not in segment.vectors
    assert "proposal_argmax_lag" not in segment.scalars
    assert "mu_post" in segment.vectors and "kld_per_t" in segment.scalars
    # And the assembly carries no lag statistics rather than a column of zeros.
    recording = traces.assemble_recording(
        segments, lag_profiles=stage.LAG_PROFILES, lag_seconds=np.arange(task.orig_model.n_lags, dtype=float),
        break_after_s=1e9,
    )
    assert "proposal_lag_centroid_s" not in recording.summary.columns


# =================================================================================================
# The stage
# =================================================================================================
class _StubDataset:
    def __init__(self, guids: List[str], epochs: List[float], codes: List[int]) -> None:
        """Hold one identity and one class code per stub segment."""
        self.guids, self.epochs, self.codes = list(guids), list(epochs), list(codes)

    def __len__(self) -> int:
        """How many stub segments the dataset holds."""
        return len(self.guids)

    def __getitem__(self, index: int) -> int:
        """The item is its own index; the collation builds the batch from the indices."""
        return int(index)

    def get_the_lists(self):
        """Return ``(guids, epochs, targets)`` exactly as the real dataset does."""
        return self.guids, self.epochs, [None] * len(self.guids)


def _loader(dataset: _StubDataset, batch_size: int = 2):
    def _collate(items: List[int]):
        return stub_batch(
            [dataset.guids[index] for index in items],
            [dataset.epochs[index] for index in items],
            class_code=dataset.codes[items[0]], seed=int(items[0]),
        )
    return types.SimpleNamespace(dataset=dataset, collate_fn=_collate, batch_size=batch_size)


def _population():
    """Three classes, two recordings each, two segments per recording, plus a one-segment one."""
    guids, epochs, codes = [], [], []
    for code, name in labels.CLASS_NAMES.items():
        for recording in range(2):
            for segment in range(2):
                guids.append(f"{name}{recording}")
                epochs.append(-7200.0 + STRIDE_S * segment)
                codes.append(code)
    guids.append("lonely"); epochs.append(-5000.0); codes.append(1)
    return guids, epochs, codes


def test_the_stage_traces_a_balanced_draw_end_to_end(tmp_path) -> None:
    """Selection, the sequential subset loader, the identity check, the forward, the files and the
    figures, on a stub population with every class present."""
    guids, epochs, codes = _population()
    dataset = _StubDataset(guids, epochs, codes)
    identities = pd.DataFrame(
        {
            "guid": guids, "epoch": epochs,
            labels.CLASS_COLUMN: [labels.CLASS_NAMES[code] for code in codes],
            labels.SUBGROUP_COLUMN: ["hie_cs"] * len(guids),
        }
    )
    task = _Task(build_tiny_model())

    block = stage.run_recording_traces(
        task, _loader(dataset), identities,
        eval_config={"seed": 0, "caps": {traces.TRACES_CAP: 1}}, results_dir=tmp_path,
        geometry_record={"t": TINY_SEQ_LEN},
    )

    assert block["status"] == stage.STATUS_TRACED, block
    assert block["failures"] == []
    assert block["n_recordings_by_class"] == {"healthy": 1, "acidosis": 1, "hie": 1}
    assert block["n_segments"] == 6
    assert block["lag_family_present"] is True
    assert block["selection"]["classes"]["healthy"]["n_eligible"] == 2
    assert block["plan"]["lag_qualification"] == stage.PROPOSAL_QUALIFICATION

    directory = tmp_path / traces.ANALYSIS_DIRNAME
    manifest = pd.read_csv(directory / traces.MANIFEST_FILENAME)
    assert len(manifest) == 3 and (manifest["n_segments"] == 2).all()
    for _, row in manifest.iterrows():
        assert (directory / row["arrays_file"]).is_file()
        assert (directory / row["figure_file"]).is_file()
        assert row[labels.SUBGROUP_COLUMN] in row["figure_file"]
    summary = pd.read_csv(directory / traces.SEGMENT_SUMMARY_FILENAME)
    assert len(summary) == 6 and "proposal_lag_centroid_s" in summary.columns
    anchors = pd.read_parquet(directory / traces.ANCHOR_TRACE_FILENAME)
    assert len(anchors) == int(manifest["n_anchors"].sum())
    assert (directory / f"{traces.SUMMARY_FIGURE}.pdf").is_file()


def test_the_stage_traces_only_the_segments_inside_the_delivery_window(tmp_path) -> None:
    """With ``max_hours_before_delivery`` set, the segment recorded before the window is neither
    traced nor counted, and the plan records that the bound was applied."""
    guids, epochs, codes = _population()
    # A third, earlier segment on every recording: outside a window that keeps the other two.
    extra = [(guid, -7200.0 - STRIDE_S, code) for guid, epoch, code in zip(guids, epochs, codes)
             if epoch == -7200.0 and guid != "lonely"]
    guids += [g for g, _, _ in extra]; epochs += [e for _, e, _ in extra]; codes += [c for _, _, c in extra]
    dataset = _StubDataset(guids, epochs, codes)
    identities = pd.DataFrame(
        {
            "guid": guids, "epoch": epochs,
            labels.CLASS_COLUMN: [labels.CLASS_NAMES[code] for code in codes],
            labels.SUBGROUP_COLUMN: ["hie_cs"] * len(guids),
        }
    )
    window = 7200.0 / 3600.0

    block = stage.run_recording_traces(
        _Task(build_tiny_model()), _loader(dataset), identities,
        eval_config={"seed": 0, "caps": {traces.TRACES_CAP: 1}, "max_hours_before_delivery": window},
        results_dir=tmp_path, geometry_record={"t": TINY_SEQ_LEN},
    )

    assert block["status"] == stage.STATUS_TRACED, block
    assert block["plan"]["max_hours_before_delivery"] == window
    assert block["plan"]["max_hours_before_delivery_applied"] is True
    manifest = pd.read_csv(tmp_path / traces.ANALYSIS_DIRNAME / traces.MANIFEST_FILENAME)
    assert (manifest["n_segments"] == 2).all()
    summary = pd.read_csv(tmp_path / traces.ANALYSIS_DIRNAME / traces.SEGMENT_SUMMARY_FILENAME)
    assert (summary["epoch"] >= -window * 3600.0).all()


def test_the_stage_records_a_skip_when_no_segment_carries_a_class(tmp_path) -> None:
    """What the pass produces when the delta does not name the target field."""
    identities = pd.DataFrame(
        {"guid": ["A", "A"], "epoch": [-3600.0, -2400.0],
         labels.CLASS_COLUMN: [None, None], labels.SUBGROUP_COLUMN: ["hie_cs", "hie_cs"]}
    )

    block = stage.run_recording_traces(
        _Task(build_tiny_model()), _loader(_StubDataset(["A", "A"], [-3600.0, -2400.0], [1, 1])),
        identities, eval_config={"seed": 0}, results_dir=tmp_path,
    )

    assert block["status"] == stage.STATUS_SKIPPED
    assert "load_fields" in block["reason"]
    assert not (tmp_path / traces.ANALYSIS_DIRNAME).exists()


def test_the_stage_records_an_empty_draw_when_no_recording_has_two_segments(tmp_path) -> None:
    """One segment is no evolution: counted as ineligible, and the draw is empty rather than a trace."""
    identities = pd.DataFrame(
        {"guid": ["A"], "epoch": [-3600.0], labels.CLASS_COLUMN: ["hie"], labels.SUBGROUP_COLUMN: ["hie_cs"]}
    )

    block = stage.run_recording_traces(
        _Task(build_tiny_model()), _loader(_StubDataset(["A"], [-3600.0], [3])),
        identities, eval_config={"seed": 0}, results_dir=tmp_path,
    )

    assert block["status"] == stage.STATUS_EMPTY
    assert block["selection"]["classes"]["hie"]["n_eligible"] == 0


def test_no_trace_name_is_one_of_the_keys_this_package_refuses() -> None:
    """The summary walker refuses attention-shaped keys at any depth, and a trace column is a
    key in the block's manifest and plan."""
    from teb_vae.lag_slot_transformer_cfs.eval.verify import FORBIDDEN_KEYS

    names = set(stage.LAG_PROFILES) | {panel.vector for panel in stage.PANELS if isinstance(panel, traces.HeatmapPanel)}
    names |= {column for panel in stage.PANELS if isinstance(panel, traces.LinePanel) for column in panel.columns}
    names |= {metric.column for metric in stage.SUMMARY_METRICS}

    assert not names & set(FORBIDDEN_KEYS)
