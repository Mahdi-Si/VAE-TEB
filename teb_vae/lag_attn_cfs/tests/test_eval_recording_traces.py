r"""The per-recording traces: the reductions, the identity round trip, and what a run leaves behind.

Three things are asserted here rather than left to a look at a figure. **The gather is at the
decoded anchors**: every vector row is the forward's value at that anchor's own decimated step,
which is what an off-by-one on a gathered anchor set would silently break -- the per-anchor
divergence must equal the sum of its per-coordinate split, at every anchor of every segment.
**The reductions obey the three rules**: only scored anchors enter a mean, an unscored segment is
``NaN`` and never zero, and the summary's lag statistics are of the segment's mean profile rather
than a mean of per-anchor statistics. **The selection is balanced, seeded and eligibility-gated**,
so ten recordings per class is what the directory holds when the class has them, and a
one-segment recording is counted rather than drawn.

The end-to-end test drives the analysis through a stub dataset that lists its own recordings and
collates into the tiny stub batch, so the whole path -- selection, the sequential subset loader,
the identity check, the forward, the join, the files and the figures -- runs on CPU in seconds.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval import traces
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import recording_traces as analysis
from teb_vae.lag_attn_cfs.eval.collect import load_collection
from teb_vae.lag_attn_cfs.eval.lag_axis import compensated_seconds_axis
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, expected_anchors_per_sample, model_inputs

from .conftest import STUB_GAP_STEP, make_stub_batch, make_task

#: Seconds between consecutive stored segments in the stub batch, and a gap wide enough to be a
#: break under any tolerance the tiny geometry resolves to.
STRIDE_S = 1200.0
BREAK_S = 10 * STRIDE_S


def _forward(module, batch):
    """One dense forward of the stub batch through the tiny task, plus the validity signal."""
    model = module.orig_model
    y_st, y_ph, u_stream, _target, weight = model_inputs(module, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    return model, model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride), weight


def _rows(guid: str, epochs: List[float]) -> pd.DataFrame:
    """Rows of one recording: the identity the gather stamps on each sample of a batch."""
    return pd.DataFrame(
        {
            "guid": [guid] * len(epochs),
            "epoch": [float(value) for value in epochs],
            "time_from_labor_onset": [float("nan")] * len(epochs),
            "second_stage_onset": [100.0 + index for index in range(len(epochs))],
        }
    )


def _segments(seed: int = 0, guid: str = "REC", epochs=(-3600.0, -3600.0 + STRIDE_S)):
    """Trace the stub batch as two consecutive segments of one recording."""
    module = make_task()
    module.eval()
    batch = make_stub_batch(seed=seed)
    with torch.no_grad():
        model, outputs, weight = _forward(module, batch)
        gathered = analysis.gather_segment_traces(
            model, outputs, weight, _rows(guid, list(epochs)),
            clinical_class="hie", subgroup="hie_cs",
        )
    return module, gathered


def _lag_seconds(segments) -> np.ndarray:
    return compensated_seconds_axis(int(segments[0].vectors["kl_lag_map"].shape[-1]), 0)


# =================================================================================================
# The gather
# =================================================================================================
def test_a_forward_gathers_one_trace_per_sample_at_its_decoded_anchors() -> None:
    module, segments = _segments()
    model = module.orig_model

    assert len(segments) == 2
    for segment in segments:
        n_anchors = expected_anchors_per_sample(model)
        assert segment.anchor.shape == (n_anchors,)
        assert segment.contributing.shape == (n_anchors,)
        # The anchor is the decimated step, starting at the floor -- not a row position.
        assert int(segment.anchor.min()) == int(model.warmup_period)
        assert segment.vectors["mu_post"].shape == (n_anchors, int(model.d_z))
        assert segment.vectors["kld_per_dim"].shape == (n_anchors, int(model.d_z))
        assert segment.vectors["kl_lag_map"].shape[0] == n_anchors
        assert segment.vectors["attention_lag_map"].shape == segment.vectors["kl_lag_map"].shape
        assert set(segment.scalars) == {"kld_per_t", "coverage", "argmax_lag", "attention_entropy_nats"}
        # The stub batch's deliberate gap sits inside the decoded range, and an anchor whose own
        # step is invalid is not scored.
        assert STUB_GAP_STEP in segment.anchor
        assert not segment.contributing[segment.anchor == STUB_GAP_STEP].any()
        assert segment.contributing.any()
        assert segment.clocks["second_stage_onset"] >= 100.0


def test_the_gathered_divergence_is_the_sum_of_its_per_coordinate_split() -> None:
    """The identity that an off-by-one on the gathered axis would break: both come off the same
    anchor, or they do not agree."""
    _module, segments = _segments()

    for segment in segments:
        np.testing.assert_allclose(
            segment.vectors["kld_per_dim"].sum(axis=-1), segment.scalars["kld_per_t"],
            rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(
            segment.vectors["kl_lag_map"].sum(axis=-1), segment.scalars["kld_per_t"],
            rtol=1e-4, atol=1e-5,
        )


# =================================================================================================
# The reductions
# =================================================================================================
def test_segments_are_ordered_by_epoch_and_anchors_placed_on_the_absolute_axis() -> None:
    _module, segments = _segments(epochs=(-3600.0 + STRIDE_S, -3600.0))
    seconds = _lag_seconds(segments)

    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=seconds, break_after_s=BREAK_S
    )

    assert [segment.epoch for segment in recording.segments] == [-3600.0, -3600.0 + STRIDE_S]
    assert list(recording.summary["segment_order"]) == [0, 1]
    first = recording.anchors[recording.anchors["segment_order"] == 0]
    expected = -3600.0 + first["anchor"].to_numpy() * 4.0
    np.testing.assert_allclose(first["t_abs_sec"].to_numpy(), expected)
    np.testing.assert_allclose(first["hours_before_delivery"].to_numpy(), -expected / 3600.0)
    # The epoch gap between consecutive segments is the stride and is not a break.
    assert np.isnan(recording.summary["epoch_gap_s"].iloc[0])
    assert recording.summary["epoch_gap_s"].iloc[1] == STRIDE_S
    assert not recording.summary["is_break"].iloc[1]


def test_a_wide_epoch_gap_is_a_break() -> None:
    _module, segments = _segments(epochs=(-9000.0, -9000.0 + BREAK_S + 1.0))
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=_lag_seconds(segments),
        break_after_s=BREAK_S,
    )

    assert bool(recording.summary["is_break"].iloc[1])


def test_only_scored_anchors_enter_the_summary_and_an_unscored_segment_is_nan() -> None:
    _module, segments = _segments()
    # Unscore the second segment entirely.
    segments[1].contributing = np.zeros_like(segments[1].contributing)
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=_lag_seconds(segments),
        break_after_s=BREAK_S,
    )
    summary = recording.summary

    scored = recording.anchors[(recording.anchors["segment_order"] == 0) & recording.anchors["contributing"]]
    assert summary["n_contributing"].iloc[0] == len(scored) > 0
    assert summary["kld_per_t"].iloc[0] == pytest.approx(float(scored["kld_per_t"].mean()))
    assert summary["n_contributing"].iloc[1] == 0
    assert np.isnan(summary["kld_per_t"].iloc[1])
    assert np.isnan(summary["latent_dispersion"].iloc[1])
    assert np.isnan(summary["latent_step"].iloc[1])
    assert np.isnan(recording.segment_vectors["mu_post"][1]).all()
    # The full form still carries every decoded anchor of the unscored segment.
    assert (recording.anchors["segment_order"] == 1).sum() == len(segments[1].anchor)


def test_the_summary_lag_statistics_are_of_the_mean_profile_not_the_mean_of_statistics() -> None:
    _module, segments = _segments()
    seconds = _lag_seconds(segments)
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=seconds, break_after_s=BREAK_S
    )

    # The attention profile rather than the attribution: a freshly built model starts at zero
    # divergence, so its attribution carries no mass and its shape is NaN by the rule -- which
    # is asserted below -- while the attention sums to one at every anchor.
    keep = segments[0].contributing.astype(bool)
    mean_profile = segments[0].vectors["attention_lag_map"][keep].mean(axis=0)
    shares = mean_profile / mean_profile.sum()
    centroid = float(shares @ seconds)
    assert recording.summary["attn_lag_centroid_s"].iloc[0] == pytest.approx(centroid, rel=1e-6)
    per_anchor = recording.anchors[(recording.anchors["segment_order"] == 0) & recording.anchors["contributing"]]
    assert np.isfinite(per_anchor["attn_lag_centroid_s"]).all()
    # A profile with no mass reduces to NaN throughout, never to a plausible centroid.
    if float(segments[0].vectors["kl_lag_map"][keep].sum()) == 0.0:
        assert np.isnan(recording.summary["kl_lag_centroid_s"].iloc[0])


def test_the_derived_latent_scalars_agree_with_the_vectors() -> None:
    _module, segments = _segments()
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=_lag_seconds(segments),
        break_after_s=BREAK_S,
    )
    rows = recording.anchors[recording.anchors["segment_order"] == 0]
    segment = recording.segments[0]

    np.testing.assert_allclose(
        rows["delta_mu_norm"].to_numpy(),
        np.linalg.norm(segment.vectors["mu_post"] - segment.vectors["mu_prior"], axis=-1),
    )
    np.testing.assert_allclose(
        rows["mean_logvar_prior"].to_numpy(), segment.vectors["logvar_prior"].mean(axis=-1)
    )
    assert (rows["n_active_dims"] <= segment.vectors["kld_per_dim"].shape[1]).all()
    share = rows["kld_top_dim_share"]
    # NaN exactly where the anchor carries no divergence to share, a share otherwise.
    assert (share.isna() == (segment.vectors["kld_per_dim"].sum(axis=-1) <= 0.0)).all()
    assert ((share.dropna() > 0) & (share.dropna() <= 1)).all()


# =================================================================================================
# The selection
# =================================================================================================
def _recordings(counts: Dict[str, int], segments: int = 3) -> pd.DataFrame:
    rows = []
    for name, count in counts.items():
        for index in range(count):
            rows.append({"guid": f"{name}{index:02d}", labels.CLASS_COLUMN: name,
                         labels.SUBGROUP_COLUMN: f"{name}_cs", "n_segments": segments})
    return pd.DataFrame(rows)


def test_the_selection_draws_the_same_number_from_every_class_as_an_upper_bound() -> None:
    frame = _recordings({"healthy": 30, "acidosis": 12, "hie": 4})

    chosen, accounting = traces.select_recordings(frame, per_class=10, seed=3)

    assert dict(chosen[labels.CLASS_COLUMN].value_counts()) == {"healthy": 10, "acidosis": 10, "hie": 4}
    assert accounting["classes"]["hie"] == {"n_recordings": 4, "n_eligible": 4, "n_selected": 4}
    assert accounting["classes"]["healthy"]["n_selected"] == 10


def test_a_single_segment_recording_is_counted_and_never_drawn() -> None:
    frame = _recordings({"healthy": 5})
    frame.loc[frame["guid"] == "healthy00", "n_segments"] = 1

    chosen, accounting = traces.select_recordings(frame, per_class=10, seed=0)

    assert "healthy00" not in set(chosen["guid"])
    assert accounting["classes"]["healthy"] == {"n_recordings": 5, "n_eligible": 4, "n_selected": 4}
    assert accounting["min_segments"] == traces.MIN_SEGMENTS_PER_TRACE == 2


def test_the_selection_is_a_function_of_the_seed_and_skips_unlabelled_recordings() -> None:
    frame = _recordings({"healthy": 40, "hie": 40})
    frame.loc[frame.index[:3], labels.CLASS_COLUMN] = None

    once, accounting = traces.select_recordings(frame, per_class=5, seed=7)
    again, _ = traces.select_recordings(frame, per_class=5, seed=7)
    other, _ = traces.select_recordings(frame, per_class=5, seed=8)

    assert list(once["guid"]) == list(again["guid"])
    assert list(once["guid"]) != list(other["guid"])
    assert accounting["n_unlabelled_recordings"] == 3
    assert not once[labels.CLASS_COLUMN].isna().any()


# =================================================================================================
# The join
# =================================================================================================
def test_the_join_attaches_collected_scores_by_rounded_epoch_and_reports_the_agreement() -> None:
    _module, segments = _segments()
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=_lag_seconds(segments),
        break_after_s=BREAK_S,
    )
    first = recording.anchors[recording.anchors["segment_order"] == 0]
    # A table carrying the first segment only, with an epoch that differs by a float32 cast,
    # the collected divergence exactly, and one score column.
    per_anchor = pd.DataFrame(
        {
            "guid": first["guid"].to_numpy(),
            "epoch": np.float32(first["epoch"].to_numpy()).astype(np.float64) + 1e-4,
            "anchor": first["anchor"].to_numpy(),
            "kld_per_t": first["kld_per_t"].to_numpy(),
            "mean_pred_gap": np.arange(len(first), dtype=np.float64),
        }
    )

    joined = analysis.join_collected_anchors(recording.anchors, per_anchor)

    assert len(joined) == len(recording.anchors)
    matched = joined[joined["segment_order"] == 0]
    np.testing.assert_allclose(matched["mean_pred_gap"].to_numpy(), np.arange(len(first)))
    assert joined[joined["segment_order"] == 1]["mean_pred_gap"].isna().all()
    assert analysis.kl_agreement(joined) == pytest.approx(0.0)
    # And a table that disagrees is reported rather than hidden.
    per_anchor["kld_per_t"] += 0.5
    assert analysis.kl_agreement(analysis.join_collected_anchors(recording.anchors, per_anchor)) == pytest.approx(0.5)


# =================================================================================================
# On disk and on the page
# =================================================================================================
def test_the_arrays_file_and_both_figures_are_written(tmp_path) -> None:
    _module, segments = _segments()
    seconds = _lag_seconds(segments)
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=seconds, break_after_s=BREAK_S
    )

    path = traces.write_recording_arrays(tmp_path / "rec_full.npz", recording, lag_seconds=seconds)
    with np.load(path) as handle:
        keys = set(handle.files)
        assert {"anchor", "epoch", "t_abs_sec", "contributing", "segment_order", "lag_seconds",
                "segment_epoch", "mu_post", "kld_per_dim", "kl_lag_map", "segment_mu_post"} <= keys
        assert handle["mu_post"].shape[0] == len(recording.anchors)
        assert handle["segment_mu_post"].shape[0] == 2

    plt.close("all")
    page = figures.render_figure(
        traces.build_recording_figure(recording, panels=analysis.PANELS, lag_seconds=seconds, caveat="c"),
        tmp_path / "rec_trace",
    )
    summary = figures.render_figure(
        traces.build_summary_figure(recording.summary, metrics=analysis.SUMMARY_METRICS),
        tmp_path / "summary",
    )
    assert Path(page).is_file() and Path(summary).is_file()
    assert plt.get_fignums() == []


def test_an_empty_summary_still_draws_a_figure(tmp_path) -> None:
    figure = traces.build_summary_figure(pd.DataFrame(), metrics=analysis.SUMMARY_METRICS)
    try:
        # One metric row each, under the coverage row.
        assert len(figure.axes) == len(analysis.SUMMARY_METRICS) + 1
    finally:
        plt.close(figure)


def test_the_joined_scores_reach_the_summary_form_as_segment_means() -> None:
    """A column attached to the full form after assembly is averaged over each segment's scored
    anchors into the summary, so the summary figure can draw it; an unscored segment is NaN."""
    _module, segments = _segments()
    seconds = _lag_seconds(segments)
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=seconds, break_after_s=BREAK_S
    )
    anchors = recording.anchors
    anchors["mean_pred_gap"] = np.where(anchors["segment_order"] == 0, 2.0, 5.0)
    anchors["absent_everywhere"] = np.nan

    written = traces.add_segment_means(recording, ("mean_pred_gap", "absent_everywhere", "never_attached"))

    assert written == ["mean_pred_gap", "absent_everywhere"]
    summary = recording.summary.sort_values("segment_order")
    scored = anchors[anchors["contributing"]]
    expected = [2.0 if (scored["segment_order"] == 0).any() else np.nan, 5.0 if (scored["segment_order"] == 1).any() else np.nan]
    np.testing.assert_allclose(summary["mean_pred_gap"].to_numpy(), expected)
    assert summary["absent_everywhere"].isna().all()


def test_the_recording_figure_lays_itself_out_with_one_column_of_data_axes(tmp_path) -> None:
    """Every row's data axes share one x extent (the colour axis lives in its own column), the
    rows share the delivery axis with delivery on the right, and the figure is stamped as laid
    out so rendering does not re-layout it."""
    _module, segments = _segments()
    seconds = _lag_seconds(segments)
    recording = traces.assemble_recording(
        segments, lag_profiles=analysis.LAG_PROFILES, lag_seconds=seconds, break_after_s=BREAK_S
    )
    figure = traces.build_recording_figure(recording, panels=analysis.PANELS, lag_seconds=seconds, caveat="c")
    try:
        data_axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
        assert len(data_axes) == len(analysis.PANELS)
        lefts = {round(ax.get_position().x0, 6) for ax in data_axes}
        rights = {round(ax.get_position().x1, 6) for ax in data_axes}
        assert len(lefts) == 1 and len(rights) == 1
        low, high = data_axes[0].get_xlim()
        assert low > high, "hours before delivery decrease to the right"
        assert all(ax.get_shared_x_axes().joined(ax, data_axes[0]) for ax in data_axes[1:])
        assert getattr(figure, "_eval_layout_done", False) is True
    finally:
        plt.close(figure)


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


def test_the_analysis_traces_a_balanced_draw_end_to_end(tmp_path) -> None:
    guids, epochs, classes = _stub_population()
    dataset = _StubDataset(guids, epochs)
    loader = types.SimpleNamespace(dataset=dataset, collate_fn=_collate_factory(dataset), batch_size=2)
    per_sample = pd.DataFrame(
        {
            "guid": guids, "epoch": epochs, labels.CLASS_COLUMN: classes,
            labels.SUBGROUP_COLUMN: [f"{name}_cs" for name in classes],
            "time_from_labor_onset": [float("nan")] * len(guids),
            "second_stage_onset": [-1.0] * len(guids),
        }
    )
    module = make_task()
    module.eval()
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        ),
        config={}, task=module, loader=loader,
    )

    result = analysis.run_recording_traces_analysis(
        context, eval_config={"seed": 0, "caps": {traces.TRACES_CAP: 1}}, output_dir=tmp_path
    )

    assert result["failures"] == [], result["failures"]
    assert result["composition"]["n_recordings_by_class"] == {"healthy": 1, "acidosis": 1, "hie": 1}
    assert result["composition"]["n_segments"] == 6 == result["n_samples"]
    assert result["selection"]["classes"]["healthy"]["n_eligible"] == 2
    assert result["plan"]["capped"] is True and result["plan"]["traces_per_class"] == 1
    # No anchor was collected by the stub pass, so the agreement is unmeasured rather than zero.
    assert result["kl_agreement_max_abs"] is None

    directory = tmp_path / traces.ANALYSIS_DIRNAME
    manifest = pd.read_csv(directory / traces.MANIFEST_FILENAME)
    assert list(manifest.columns) == list(traces.MANIFEST_COLUMNS)
    assert len(manifest) == 3 and (manifest["n_segments"] == 2).all()
    for _, row in manifest.iterrows():
        assert (directory / row["arrays_file"]).is_file()
        assert (directory / row["figure_file"]).is_file()
        # The class directory and the subgroup in the name.
        assert row["arrays_file"].startswith(f"{row[labels.CLASS_COLUMN]}/")
        assert row[labels.SUBGROUP_COLUMN] in row["figure_file"]
    summary = pd.read_csv(directory / traces.SEGMENT_SUMMARY_FILENAME)
    assert len(summary) == 6 and set(summary["guid"]) == set(manifest["guid"])
    anchors = pd.read_parquet(directory / traces.ANCHOR_TRACE_FILENAME)
    assert len(anchors) == int(manifest["n_anchors"].sum())
    assert (directory / f"{traces.SUMMARY_FIGURE}.pdf").is_file()


def test_the_analysis_traces_only_the_segments_inside_the_delivery_window(tmp_path) -> None:
    """With ``max_hours_before_delivery`` set, the segment recorded before the window is neither
    traced nor counted, and the plan records that the bound was applied."""
    guids, epochs, classes = _stub_population()
    extra = [(guid, -7200.0 - STRIDE_S, name) for guid, epoch, name in zip(guids, epochs, classes)
             if epoch == -7200.0 and guid != "lonely"]
    guids += [g for g, _, _ in extra]; epochs += [e for _, e, _ in extra]; classes += [c for _, _, c in extra]
    dataset = _StubDataset(guids, epochs)
    loader = types.SimpleNamespace(dataset=dataset, collate_fn=_collate_factory(dataset), batch_size=2)
    per_sample = pd.DataFrame(
        {"guid": guids, "epoch": epochs, labels.CLASS_COLUMN: classes,
         labels.SUBGROUP_COLUMN: [f"{name}_cs" for name in classes]}
    )
    module = make_task()
    module.eval()
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        ),
        config={}, task=module, loader=loader,
    )
    window = 7200.0 / 3600.0

    result = analysis.run_recording_traces_analysis(
        context, eval_config={"seed": 0, "caps": {traces.TRACES_CAP: 1}, "max_hours_before_delivery": window},
        output_dir=tmp_path,
    )

    assert result["failures"] == [], result["failures"]
    assert result["plan"]["max_hours_before_delivery"] == window
    assert result["plan"]["max_hours_before_delivery_applied"] is True
    manifest = pd.read_csv(tmp_path / traces.ANALYSIS_DIRNAME / traces.MANIFEST_FILENAME)
    assert len(manifest) == 3 and (manifest["n_segments"] == 2).all()
    summary = pd.read_csv(tmp_path / traces.ANALYSIS_DIRNAME / traces.SEGMENT_SUMMARY_FILENAME)
    assert (summary["epoch"] >= -window * 3600.0).all()


def test_a_pass_with_no_model_records_a_skip(tmp_path) -> None:
    context = AnalysisContext(
        collection=types.SimpleNamespace(
            per_sample=pd.DataFrame({"guid": ["g0"], "epoch": [-1000.0]}),
            per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        ),
        config={},
    )

    result = analysis.run_recording_traces_analysis(context, eval_config={"seed": 0}, output_dir=tmp_path)

    assert result["skipped"] is True and result["n_samples"] is None
    assert "no model" in result["reason"]


# =================================================================================================
# Against the real run
# =================================================================================================
@pytest.mark.slow
def test_a_real_run_traces_every_class_and_agrees_with_the_collected_divergence(collected_run) -> None:
    """The cohort fixture holds two segments per recording, so every recording is eligible and
    every class present is traced; and the re-read forward reproduces the collected divergence."""
    result = collected_run["summary"]["results"][traces.ANALYSIS_DIRNAME]
    collection = load_collection(collected_run["results_dir"])
    present = set(collection.per_sample[labels.CLASS_COLUMN].dropna().unique())

    assert result["failures"] == []
    assert set(result["composition"]["n_recordings_by_class"]) == present
    assert result["kl_agreement_max_abs"] is not None
    assert result["kl_agreement_max_abs"] <= result["kl_agreement_tolerance"]
    directory = Path(collected_run["results_dir"]) / traces.ANALYSIS_DIRNAME
    manifest = pd.read_csv(directory / traces.MANIFEST_FILENAME)
    assert (manifest["n_segments_collected"] == manifest["n_segments"]).all()
    for _, row in manifest.iterrows():
        assert (directory / row["figure_file"]).is_file()
        assert (directory / row["arrays_file"]).is_file()
    anchors = pd.read_parquet(directory / traces.ANCHOR_TRACE_FILENAME)
    assert anchors["mean_pred_gap"].notna().any()
