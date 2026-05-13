"""Unit tests for :mod:`sso_metrics_utils`.

Validates the four building blocks of the SSO-anchored evaluation
pipeline against hand-rolled DataFrames:

* :func:`ensure_t_rel_sso_hours` derives the signed axis column from the
  legacy ``sso_hours`` alias.
* :func:`filter_to_sso_eligible` drops only GUIDs that lack SSO end-to-end
  and reports the dropped count.
* :func:`compute_sso_time_bins` produces signed bin edges spanning the
  data with the segment-spacing bin width.
* :func:`recompute_t_rel_sso_after_fill` repairs the SSO column on rows
  synthesised by :func:`fill_missing_epochs`.
* :func:`compute_instantaneous_metrics_sso` has the documented output
  schema and correct numerator/denominator on a small fixture.

Heavy plotting paths are exercised end-to-end by the per-fold smoke run;
keep the unit tests fast (no matplotlib).
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _make_predictions_df() -> "pd.DataFrame":
    """Synthetic predictions: 4 GUIDs, 3 segments each, epoch step 1200 s.

    GUID ``g_nan`` carries NaN ``sso_hours`` to exercise the eligibility
    filter. The remaining three GUIDs span a range of SSO offsets so the
    bin computation produces multiple non-empty bins.
    """
    rows = []
    segment_step = 1200.0  # 20-minute segments.
    cases = [
        # (guid, sso_offset_seconds_relative_to_delivery, binary_target,
        #  per-segment clinical_pred sequence over 3 segments)
        ("g1", -3 * 3600.0, 1, [0, 1, 1]),  # detected at second segment
        ("g2", -2 * 3600.0, 1, [0, 0, 0]),  # missed
        ("g3", -1 * 3600.0, 0, [0, 0, 0]),  # true negative
        ("g_nan", None,       0, [0, 0, 0]),  # SSO missing
    ]
    for guid, sso_offset, btarget, preds in cases:
        for i, pred in enumerate(preds):
            # epoch = -delta from delivery; later segments closer to birth.
            epoch_seconds = -3 * 3600.0 + i * segment_step
            if sso_offset is None:
                sso_h = float("nan")
            else:
                sso_h = (epoch_seconds - sso_offset) / 3600.0
            rows.append(
                {
                    "guid": guid,
                    "epoch": epoch_seconds,
                    "binary_target": int(btarget),
                    "clinical_pred": int(pred),
                    "sso_hours": sso_h,
                }
            )
    return pd.DataFrame(rows)


def test_ensure_t_rel_sso_hours_copies_from_alias() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = _make_predictions_df()
    out = sso_metrics_utils.ensure_t_rel_sso_hours(df)
    assert "t_rel_sso_hours" in out.columns
    # Equal numerically (NaN-tolerant).
    finite_mask = out["sso_hours"].notna()
    assert np.allclose(
        out.loc[finite_mask, "t_rel_sso_hours"].to_numpy(),
        out.loc[finite_mask, "sso_hours"].to_numpy(),
    )


def test_ensure_t_rel_sso_hours_passthrough_when_present() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = _make_predictions_df()
    df = df.drop(columns=["sso_hours"])
    df["t_rel_sso_hours"] = 1.5
    out = sso_metrics_utils.ensure_t_rel_sso_hours(df)
    assert (out["t_rel_sso_hours"] == 1.5).all()


def test_ensure_t_rel_sso_hours_raises_without_source() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = pd.DataFrame({"guid": ["g1"], "epoch": [0.0]})
    with pytest.raises(KeyError):
        sso_metrics_utils.ensure_t_rel_sso_hours(df)


def test_filter_to_sso_eligible_drops_nan_guid() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    filtered, stats = sso_metrics_utils.filter_to_sso_eligible(df)
    assert stats["n_dropped_guids"] == 1
    assert stats["dropped_guids"] == ["g_nan"]
    assert stats["n_kept_guids"] == 3
    assert "g_nan" not in set(filtered["guid"])
    assert set(filtered["guid"]) == {"g1", "g2", "g3"}


def test_filter_to_sso_eligible_all_missing() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = pd.DataFrame(
        {
            "guid": ["g1", "g2"],
            "epoch": [-3600.0, -3600.0],
            "t_rel_sso_hours": [float("nan"), float("nan")],
        }
    )
    filtered, stats = sso_metrics_utils.filter_to_sso_eligible(df)
    assert stats["n_kept_guids"] == 0
    assert stats["n_dropped_guids"] == 2
    assert filtered.empty


def test_compute_sso_time_bins_signed_range() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    df, _ = sso_metrics_utils.filter_to_sso_eligible(df)
    bins = sso_metrics_utils.compute_sso_time_bins(df)
    assert bins.ndim == 1 and bins.size >= 2
    # Bins must be monotonically increasing.
    assert np.all(np.diff(bins) > 0)
    # Bins must span both negative and positive t_rel values present
    # in the fixture (g1's first segment starts -3h before SSO; the
    # latest segment is at -2h before SSO + 40min so still negative).
    # Confirm the signed range is preserved (no post-delivery filter).
    sso_vals = df["t_rel_sso_hours"].to_numpy()
    assert bins[0] <= sso_vals.min()
    assert bins[-1] >= sso_vals.max()


def test_compute_sso_time_bins_explicit_width() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    df, _ = sso_metrics_utils.filter_to_sso_eligible(df)
    bins = sso_metrics_utils.compute_sso_time_bins(df, bin_size_hours=0.5)
    deltas = np.diff(bins)
    assert np.allclose(deltas, 0.5, atol=1e-6)


def test_recompute_t_rel_sso_after_fill_restores_synthesised_rows() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    df, _ = sso_metrics_utils.filter_to_sso_eligible(df)
    # Simulate ``fill_missing_epochs`` by appending a synthesised row for
    # GUID ``g1`` at a new epoch — NaN in t_rel_sso_hours but real epoch.
    fake_filled_epoch = -3 * 3600.0 + 5 * 1200.0  # 1h40m past first segment
    expected_sso_offset = -3 * 3600.0  # g1's per-recording offset
    synthesised = pd.DataFrame(
        [
            {
                "guid": "g1",
                "epoch": fake_filled_epoch,
                "binary_target": 1,
                "clinical_pred": 1,
                "sso_hours": float("nan"),
                "t_rel_sso_hours": float("nan"),
            }
        ]
    )
    df_with_gap = pd.concat([df, synthesised], ignore_index=True)
    repaired = sso_metrics_utils.recompute_t_rel_sso_after_fill(df_with_gap)
    # Newly added row should now carry a finite t_rel_sso_hours equal
    # to (epoch - offset) / 3600.
    new_row = repaired[
        (repaired["guid"] == "g1")
        & np.isclose(repaired["epoch"], fake_filled_epoch)
    ]
    assert len(new_row) == 1
    expected = (fake_filled_epoch - expected_sso_offset) / 3600.0
    assert np.isclose(new_row["t_rel_sso_hours"].iloc[0], expected, atol=1e-6)


def test_compute_instantaneous_metrics_sso_schema() -> None:
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    df, _ = sso_metrics_utils.filter_to_sso_eligible(df)
    bins = sso_metrics_utils.compute_sso_time_bins(df, bin_size_hours=0.5)
    out = sso_metrics_utils.compute_instantaneous_metrics_sso(df, bins)
    required = {
        "bin_center", "sensitivity", "specificity", "fpr",
        "n_positive", "n_negative", "n_tp", "n_fp", "n_tn", "n_fn",
    }
    assert required.issubset(set(out.columns))
    # Bins with no observations should yield zero counts; bins with
    # observations should yield non-negative integer counts.
    for col in ("n_positive", "n_negative", "n_tp", "n_fp", "n_tn", "n_fn"):
        assert (out[col] >= 0).all()


def test_compute_instantaneous_metrics_sso_numerator() -> None:
    """The aggregate TP count across bins must match the raw clinical_pred."""
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    df, _ = sso_metrics_utils.filter_to_sso_eligible(df)
    bins = sso_metrics_utils.compute_sso_time_bins(df, bin_size_hours=0.5)
    out = sso_metrics_utils.compute_instantaneous_metrics_sso(df, bins)
    expected_tp = int(
        ((df["binary_target"] == 1) & (df["clinical_pred"] == 1)).sum()
    )
    expected_fp = int(
        ((df["binary_target"] == 0) & (df["clinical_pred"] == 1)).sum()
    )
    assert int(out["n_tp"].sum()) == expected_tp
    assert int(out["n_fp"].sum()) == expected_fp


def test_compute_committed_overall_metrics_sso_monotonic() -> None:
    """Committed-overall sensitivity must be non-decreasing in bin_center."""
    from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (
        sso_metrics_utils,
    )

    df = sso_metrics_utils.ensure_t_rel_sso_hours(_make_predictions_df())
    df, _ = sso_metrics_utils.filter_to_sso_eligible(df)
    bins = sso_metrics_utils.compute_sso_time_bins(df, bin_size_hours=0.5)
    out = sso_metrics_utils.compute_committed_overall_metrics_sso(df, bins)
    sens = out["sensitivity"].dropna().to_numpy()
    # When sorted by bin_center ascending (the function does this internally),
    # sensitivity must be non-decreasing.
    assert np.all(np.diff(sens) >= -1e-9), (
        f"committed_overall_sso sensitivity decreased: {sens}"
    )
