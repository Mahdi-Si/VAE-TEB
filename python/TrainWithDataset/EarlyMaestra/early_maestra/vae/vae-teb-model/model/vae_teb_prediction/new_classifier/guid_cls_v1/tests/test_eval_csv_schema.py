"""Eval CSV schema test for ``run_inference_per_position``.

Confirms:
* Row count = ``Σ_g N_g`` (one row per observed segment).
* Column set = legacy schema minus ``aux_prob_*`` columns plus ``position``,
  with ``prefix_length`` retained as alias for back-compat.
* ``prefix_length`` column equals the new ``position`` column on every row
  so legacy aggregator code that filters on ``prefix_length`` keeps working.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from model.vae_teb_prediction.new_classifier.guid_cls_v1.collate import (  # noqa: E402
    guid_sequence_collate_fn,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.evaluate_guid_classifier import (  # noqa: E402
    run_inference_per_position,
    run_inference_prefix_sweep,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_classifier import (  # noqa: E402
    GuidClassifierConfig,
    GuidOutcomeClassifier,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (  # noqa: E402
    GuidSequenceDataset,
)
from model.vae_teb_prediction.new_classifier.guid_cls_v1.tests.synthetic_cache import (  # noqa: E402
    D_MODEL,
    D_Z,
    write_synthetic_cache,
)


ALWAYS_PRESENT_COLUMNS = {
    "guid",
    "epoch",
    "target",
    "position",
    "prefix_length",
    "cs_label",
    "bg_label",
    "tlo_hours",
    "sso_hours",
    "t_rel_sso_hours",
    "guid_binary_target",
    "guid_class_3_target",
}

BINARY_HEAD_COLUMNS = {
    "binary_target",
    "predicted_class",
    "prob_class_0",
    "prob_class_1",
}

THREE_CLASS_HEAD_COLUMNS = {
    "prob_healthy",
    "prob_acidosis",
    "prob_hie",
    "predicted_class_3",
}

REQUIRED_COLUMNS = (
    ALWAYS_PRESENT_COLUMNS | BINARY_HEAD_COLUMNS | THREE_CLASS_HEAD_COLUMNS
)

REMOVED_COLUMNS = {"aux_prob_bin_segment", "aux_prob_3_segment_max"}


@pytest.fixture()
def eval_loader_and_model(tmp_path: Path):
    cache = tmp_path / "fold_1" / "test.hdf5"
    cache.parent.mkdir(parents=True, exist_ok=True)
    n_guids, n_segs = 5, 4
    write_synthetic_cache(cache, num_guids=n_guids, segments_per_guid=n_segs)
    ds = GuidSequenceDataset(cache, min_samples_per_guid=3)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: guid_sequence_collate_fn(batch),
    )
    cfg = GuidClassifierConfig(d_model_vae=D_MODEL, d_z=D_Z, n_layers=1)
    model = GuidOutcomeClassifier(cfg).eval()
    return loader, model, ds, n_segs


def test_alias_points_to_per_position(eval_loader_and_model) -> None:
    """``run_inference_prefix_sweep`` must be a back-compat alias."""
    assert run_inference_prefix_sweep is run_inference_per_position


def test_csv_schema_and_row_count(eval_loader_and_model) -> None:
    """Schema matches the new design; row count equals Σ_g N_g."""
    loader, model, ds, n_segs = eval_loader_and_model
    df = run_inference_per_position(model, loader, device=torch.device("cpu"))

    # Row count: each GUID contributes exactly its number of valid segments.
    expected_rows = len(ds) * n_segs  # synthetic cache uses uniform N_g
    assert len(df) == expected_rows

    # Required columns present.
    assert REQUIRED_COLUMNS.issubset(df.columns), (
        f"missing columns: {REQUIRED_COLUMNS - set(df.columns)}"
    )
    # Removed columns absent.
    assert REMOVED_COLUMNS.isdisjoint(df.columns), (
        f"unexpected legacy columns present: {REMOVED_COLUMNS & set(df.columns)}"
    )

    # ``prefix_length`` alias equals ``position`` on every row.
    assert (df["prefix_length"] == df["position"]).all()

    # Per-GUID positions are 1-indexed and contiguous.
    for _, group in df.groupby("guid"):
        positions = group["position"].tolist()
        assert positions == list(range(1, len(positions) + 1))


@pytest.fixture()
def head_toggle_loader_and_model_factory(tmp_path: Path):
    """Build a per-call loader + classifier with the requested head flags.

    Tied to a single synthetic cache so per-toggle row counts are
    directly comparable.
    """
    cache = tmp_path / "fold_1" / "test.hdf5"
    cache.parent.mkdir(parents=True, exist_ok=True)
    n_guids, n_segs = 4, 4
    write_synthetic_cache(cache, num_guids=n_guids, segments_per_guid=n_segs)
    ds = GuidSequenceDataset(cache, min_samples_per_guid=3)
    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: guid_sequence_collate_fn(batch),
    )

    def _build(*, enable_three_class: bool, enable_binary: bool):
        cfg = GuidClassifierConfig(
            d_model_vae=D_MODEL,
            d_z=D_Z,
            n_layers=1,
            enable_three_class_head=enable_three_class,
            enable_binary_head=enable_binary,
        )
        model = GuidOutcomeClassifier(cfg).eval()
        return loader, model, ds, n_segs

    return _build


@pytest.mark.parametrize(
    "enable_three_class,enable_binary,expected_extra,unexpected",
    [
        (
            True,
            False,
            THREE_CLASS_HEAD_COLUMNS,
            BINARY_HEAD_COLUMNS,
        ),
        (
            False,
            True,
            BINARY_HEAD_COLUMNS,
            THREE_CLASS_HEAD_COLUMNS,
        ),
    ],
)
def test_csv_schema_under_head_toggle(
    head_toggle_loader_and_model_factory,
    enable_three_class: bool,
    enable_binary: bool,
    expected_extra: set,
    unexpected: set,
) -> None:
    """Disabled-head columns must be omitted from the inference CSV.

    Confirms the head-toggle plan's CSV-schema contract: the always-
    present columns are unchanged, the enabled head's columns are
    present, the disabled head's columns are absent.
    """
    loader, model, _ds, _n_segs = head_toggle_loader_and_model_factory(
        enable_three_class=enable_three_class, enable_binary=enable_binary
    )
    df = run_inference_per_position(model, loader, device=torch.device("cpu"))

    assert ALWAYS_PRESENT_COLUMNS.issubset(df.columns), (
        f"missing always-present columns: "
        f"{ALWAYS_PRESENT_COLUMNS - set(df.columns)}"
    )
    assert expected_extra.issubset(df.columns), (
        f"enabled-head columns missing: {expected_extra - set(df.columns)}"
    )
    assert unexpected.isdisjoint(df.columns), (
        f"disabled-head columns leaked: {unexpected & set(df.columns)}"
    )
