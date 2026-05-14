"""Unit tests for :func:`sso_metrics_utils.filter_to_sso_eligible_strict`.

Covers the strict NaN-or-zero-sentinel filter introduced to drop GUIDs
whose ``second_stage_onset`` was either absent (NaN) or stored as the
``0.0`` sentinel ("SSO == delivery / not recorded"). The latter case
otherwise produces the $-12\\,\\mathrm{h}$-before-SSO plot artefact in
the SSO-axis evaluation tree.

Fixture covers four GUIDs spanning the cross-product of NaN / zero /
mixed / normal SSO patterns; each policy combination is asserted to
drop the right subset.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (  # noqa: E402
    sso_metrics_utils,
)


SSO_COL = sso_metrics_utils.SSO_TIME_COL  # ``"t_rel_sso_hours"``


def _make_df():
    """Synthetic predictions: 5 GUIDs spanning the NaN/zero/mixed/normal axes.

    * ``g_normal``: three rows, real SSO offsets in $\\{-2, -1, 0.5\\}$h.
    * ``g_all_nan``: three rows, all NaN — dropped under ``drop_nan``.
    * ``g_all_zero``: three rows, all $0.0$ — the sentinel; dropped under
      ``drop_zero_sentinel``.
    * ``g_mixed_nan``: two non-NaN rows ($-1$h, $0$h) plus one NaN —
      dropped under ``drop_nan`` (NaN path takes precedence).
    * ``g_some_zero``: two rows $0.0$, one row $-2$h — *not* all-zero
      and not NaN, so kept under every policy.
    """
    rows = []
    for guid, sso_values in [
        ("g_normal", [-2.0, -1.0, 0.5]),
        ("g_all_nan", [float("nan"), float("nan"), float("nan")]),
        ("g_all_zero", [0.0, 0.0, 0.0]),
        ("g_mixed_nan", [-1.0, 0.0, float("nan")]),
        ("g_some_zero", [0.0, 0.0, -2.0]),
    ]:
        for sso_val in sso_values:
            rows.append({"guid": guid, SSO_COL: sso_val})
    return pd.DataFrame(rows)


def test_drops_only_nan_when_zero_sentinel_disabled() -> None:
    """``drop_nan=True, drop_zero_sentinel=False`` matches legacy behaviour."""
    df = _make_df()
    out, stats = sso_metrics_utils.filter_to_sso_eligible_strict(
        df, drop_nan=True, drop_zero_sentinel=False
    )
    kept_guids = set(out["guid"].unique())
    # ``g_all_nan`` and ``g_mixed_nan`` drop (NaN); the others stay.
    assert kept_guids == {"g_normal", "g_all_zero", "g_some_zero"}
    assert stats["n_dropped_nan"] == 2
    assert stats["n_dropped_zero_sentinel"] == 0
    assert stats["n_dropped_guids"] == 2
    assert stats["n_kept_guids"] == 3
    assert stats["n_total_guids"] == 5
    assert sorted(stats["dropped_guids_nan"]) == ["g_all_nan", "g_mixed_nan"]
    assert stats["dropped_guids_zero_sentinel"] == []
    assert stats["policy"] == {"drop_nan": True, "drop_zero_sentinel": False}


def test_drops_only_zero_when_nan_disabled() -> None:
    """``drop_nan=False, drop_zero_sentinel=True`` isolates the new behaviour."""
    df = _make_df()
    out, stats = sso_metrics_utils.filter_to_sso_eligible_strict(
        df, drop_nan=False, drop_zero_sentinel=True
    )
    kept_guids = set(out["guid"].unique())
    # Only ``g_all_zero`` drops; NaN GUIDs survive because ``drop_nan=False``.
    assert kept_guids == {"g_normal", "g_all_nan", "g_mixed_nan", "g_some_zero"}
    assert stats["n_dropped_nan"] == 0
    assert stats["n_dropped_zero_sentinel"] == 1
    assert stats["n_dropped_guids"] == 1
    assert stats["dropped_guids_zero_sentinel"] == ["g_all_zero"]


def test_default_policy_drops_nan_and_zero_sentinel() -> None:
    """Default policy (both predicates on) drops both classes of bad GUID."""
    df = _make_df()
    out, stats = sso_metrics_utils.filter_to_sso_eligible_strict(df)
    kept_guids = set(out["guid"].unique())
    assert kept_guids == {"g_normal", "g_some_zero"}
    assert stats["n_dropped_nan"] == 2  # g_all_nan + g_mixed_nan
    assert stats["n_dropped_zero_sentinel"] == 1  # g_all_zero
    assert stats["n_dropped_guids"] == 3
    assert stats["n_kept_guids"] == 2
    assert sorted(stats["dropped_guids_nan"]) == ["g_all_nan", "g_mixed_nan"]
    assert stats["dropped_guids_zero_sentinel"] == ["g_all_zero"]
    # NaN takes precedence: ``g_mixed_nan`` has a 0.0 row but is counted
    # under NaN only (the per-GUID predicates are mutually exclusive).
    assert "g_mixed_nan" not in stats["dropped_guids_zero_sentinel"]


def test_no_op_when_both_predicates_disabled() -> None:
    """``drop_nan=False, drop_zero_sentinel=False`` returns the input unchanged."""
    df = _make_df()
    out, stats = sso_metrics_utils.filter_to_sso_eligible_strict(
        df, drop_nan=False, drop_zero_sentinel=False
    )
    assert set(out["guid"].unique()) == set(df["guid"].unique())
    assert stats["n_dropped_guids"] == 0
    assert stats["n_kept_guids"] == 5
    assert stats["dropped_guids_nan"] == []
    assert stats["dropped_guids_zero_sentinel"] == []


def test_all_rows_kept_for_well_formed_guids() -> None:
    """Kept GUIDs retain *all* their rows after filtering (row-wise integrity)."""
    df = _make_df()
    out, _ = sso_metrics_utils.filter_to_sso_eligible_strict(df)
    # ``g_normal`` and ``g_some_zero`` had 3 rows each → 6 rows in output.
    assert len(out) == 6
    assert (out.groupby("guid").size() == 3).all()


def test_missing_column_raises_keyerror() -> None:
    """Helpers refuse a DataFrame missing the SSO column."""
    df = pd.DataFrame({"guid": ["g1", "g2"], "other": [1, 2]})
    with pytest.raises(KeyError, match=SSO_COL):
        sso_metrics_utils.filter_to_sso_eligible_strict(df)


def test_missing_guid_column_raises_keyerror() -> None:
    """Helpers refuse a DataFrame missing the ``guid`` column."""
    df = pd.DataFrame({SSO_COL: [0.0, 1.0]})
    with pytest.raises(KeyError, match="guid"):
        sso_metrics_utils.filter_to_sso_eligible_strict(df)


def test_empty_input_returns_empty_output() -> None:
    """Empty input → empty output with zero counts (no exceptions)."""
    df = pd.DataFrame({"guid": [], SSO_COL: []})
    out, stats = sso_metrics_utils.filter_to_sso_eligible_strict(df)
    assert out.empty
    assert stats["n_total_guids"] == 0
    assert stats["n_kept_guids"] == 0
    assert stats["n_dropped_guids"] == 0
