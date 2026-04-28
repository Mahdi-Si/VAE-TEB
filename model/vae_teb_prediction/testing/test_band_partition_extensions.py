"""Unit tests for the band-partition / per-channel forecast extensions.

Covers the four partitions exposed by
``model.vae_teb_prediction.testing.band_partition.BandPartition``
(clinical_4band, clinical_7band, by_kind, by_octave) and the
:func:`compute_per_channel_forecast_metrics` helper added alongside
``compute_band_forecast_metrics``.

The full ``build_band_partition`` round-trip needs the
``hdf5_dataset.kymatio_phase_scattering`` module (the kymatio wrapper);
those tests are skipped when kymatio isn't available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.vae_teb_prediction.testing.band_partition import (
    KIND_NAMES,
    OCTAVE_DC_LABEL,
    REFINED7_BAND_NAMES,
    REFINED7_HZ_RANGES,
    _band_for_hz,
    _build_octave_ranges,
    _octave_for_hz,
    _refined7_for_hz,
)
from model.vae_teb_prediction.testing.metrics import (
    compute_band_forecast_metrics,
    compute_forecast_metrics,
    compute_per_channel_forecast_metrics,
)


# ---------------------------------------------------------------------------
# Helper functions: pure, no dependency on kymatio.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq_hz,expected",
    [
        (float("nan"), "baseline"),
        (0.0, "baseline"),
        (0.005, "baseline"),
        (0.008, "early_decel"),
        (0.012, "early_decel"),
        (0.013, "late_decel"),
        (0.0399, "late_decel"),
        (0.04, "lf_var"),
        (0.149, "lf_var"),
        (0.15, "mf_var"),
        (0.249, "mf_var"),
        (0.25, "beat_to_beat"),
        (0.999, "beat_to_beat"),
        (1.0, "nyquist_edge"),
        (1.96, "nyquist_edge"),
    ],
)
def test_refined7_boundary_assignment(freq_hz, expected):
    assert _refined7_for_hz(freq_hz) == expected


def test_refined7_ranges_cover_zero_to_inf_without_gaps():
    """The refined-7 partition must cover the full positive real line."""
    sorted_ranges = [REFINED7_HZ_RANGES[b] for b in REFINED7_BAND_NAMES]
    # baseline starts at 0
    assert sorted_ranges[0][0] == 0.0
    # adjacent intervals are contiguous (lo of band k == hi of band k-1)
    for prev, nxt in zip(sorted_ranges, sorted_ranges[1:]):
        assert prev[1] == nxt[0]
    # final interval extends to infinity
    assert sorted_ranges[-1][1] == float("inf")


def test_octave_ranges_for_canonical_config():
    ranges = _build_octave_ranges(fs=4.0, J=11)
    assert len(ranges) == 11
    assert ranges["octave_0"] == (2.0, 4.0)
    assert ranges["octave_1"] == (1.0, 2.0)
    # Exponential decay: octave_k covers [fs*2^-(k+1), fs*2^-k).
    for k in range(11):
        lo, hi = ranges[f"octave_{k}"]
        assert hi == pytest.approx(4.0 * 2 ** (-k))
        assert lo == pytest.approx(4.0 * 2 ** (-(k + 1)))


def test_octave_for_hz_classifies_known_inputs():
    ranges = _build_octave_ranges(fs=4.0, J=11)
    assert _octave_for_hz(float("nan"), ranges) == OCTAVE_DC_LABEL
    assert _octave_for_hz(3.0, ranges) == "octave_0"
    assert _octave_for_hz(1.5, ranges) == "octave_1"
    assert _octave_for_hz(0.5, ranges) == "octave_2"
    # Below the lowest octave -> octave_dc
    assert _octave_for_hz(0.0001, ranges) == OCTAVE_DC_LABEL


def test_kind_names_exhaustive():
    """Every channel kind produced by the dataset selection must be listed."""
    assert set(KIND_NAMES) == {
        "st_S0", "st_S1", "ph_diag", "ph_h2", "ph_h3", "ph_other",
    }


def test_band_for_hz_keeps_clinical_4band_inclusive_at_0_25():
    """The legacy clinical_4band uses inclusive upper bound at 0.25 Hz."""
    assert _band_for_hz(0.25) == "variability"
    # 0.2501 is no longer in variability — but we don't pin a specific band
    # because the boundary just above 0.25 lives in the next band.
    assert _band_for_hz(0.26) == "beat_to_beat"


# ---------------------------------------------------------------------------
# BandPartition end-to-end (requires kymatio).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def partition():
    pytest.importorskip("kymatio")
    from model.vae_teb_prediction.testing.band_partition import (
        build_band_partition,
    )
    return build_band_partition()  # canonical defaults: J=11, Q=4, T=16, ...


def _is_partition_of_range_n(idx_dict, n_total):
    """Return True iff the union of int arrays in ``idx_dict`` equals range(n_total)."""
    if not idx_dict:
        return False
    flat = np.concatenate([np.asarray(v, dtype=int) for v in idx_dict.values()])
    return sorted(flat.tolist()) == list(range(n_total))


def test_clinical_4band_is_partition_of_87(partition):
    assert _is_partition_of_range_n(partition.combined_idx, partition.n_total)


def test_clinical_7band_is_partition_of_87(partition):
    assert _is_partition_of_range_n(partition.refined7_idx, partition.n_total)


def test_by_kind_is_partition_of_87(partition):
    assert _is_partition_of_range_n(partition.kind_idx, partition.n_total)


def test_by_octave_is_partition_of_87(partition):
    assert _is_partition_of_range_n(partition.octave_idx, partition.n_total)


def test_kind_partition_canonical_counts(partition):
    """For the v1 dataset config we expect 1 S0 channel and 42 S1 channels."""
    assert int(partition.kind_idx["st_S0"].size) == 1
    assert int(partition.kind_idx["st_S1"].size) == 42
    # The phase coefficients are split among ph_diag / ph_h2 / ph_h3 /
    # ph_other; their total must equal n_ph_channels (44).
    ph_total = sum(
        int(partition.kind_idx[k].size)
        for k in ("ph_diag", "ph_h2", "ph_h3", "ph_other")
    )
    assert ph_total == partition.n_ph_channels


def test_refined7_subbands_match_clinical_decel(partition):
    """early_decel ∪ late_decel == deceleration (clinical_4band)."""
    decel_clinical = set(partition.combined_idx["deceleration"].tolist())
    decel_refined = (
        set(partition.refined7_idx["early_decel"].tolist())
        | set(partition.refined7_idx["late_decel"].tolist())
    )
    assert decel_refined == decel_clinical


def test_refined7_var_subset_of_clinical_variability(partition):
    """lf_var ∪ mf_var ⊆ variability (clinical_4band).

    Subset, not equality: 0.25 Hz lands in variability under
    clinical_4band's inclusive convention but in beat_to_beat under
    refined7's half-open convention.
    """
    var_clinical = set(partition.combined_idx["variability"].tolist())
    var_refined = (
        set(partition.refined7_idx["lf_var"].tolist())
        | set(partition.refined7_idx["mf_var"].tolist())
    )
    assert var_refined.issubset(var_clinical)


def test_partition_idx_lookup(partition):
    """``partition_idx(name)`` returns the right index dict for each name."""
    assert partition.partition_idx("clinical_4band") is partition.combined_idx
    assert partition.partition_idx("clinical_7band") is partition.refined7_idx
    assert partition.partition_idx("by_kind") is partition.kind_idx
    assert partition.partition_idx("by_octave") is partition.octave_idx


def test_channel_metadata_has_new_columns(partition):
    """``channel_metadata`` must carry the new ``refined_band`` / ``octave`` columns."""
    md = partition.channel_metadata
    assert "refined_band" in md.columns
    assert "octave" in md.columns
    # S0 channel sits at row 0 (channel index 0) and must carry baseline / octave_dc.
    s0_row = md.loc[md["channel"] == 0].iloc[0]
    assert s0_row["kind"] == "st_S0"
    assert s0_row["refined_band"] == "baseline"
    assert s0_row["octave"] == OCTAVE_DC_LABEL


# ---------------------------------------------------------------------------
# compute_per_channel_forecast_metrics: numerical invariant.
# ---------------------------------------------------------------------------


def test_per_channel_mean_matches_feat_mse_total():
    """``mse_per_channel.mean(dim=1) == feat_mse_total`` (channel-mean identity)."""
    torch.manual_seed(0)
    B, T, H_d, C = 4, 60, 12, 8
    warmup = 5
    mu_full = torch.randn(B, T, H_d, C)
    y_plus = mu_full[:, : T - H_d, :, :] + 0.3 * torch.randn(B, T - H_d, H_d, C)

    pc = compute_per_channel_forecast_metrics(mu_full, y_plus, warmup, H_d)
    fm = compute_forecast_metrics(mu_full, y_plus, warmup, H_d)

    diff = (pc["mse_per_channel"].mean(dim=1) - fm["feat_mse_total"]).abs().max()
    assert diff.item() < 1e-5


def test_per_channel_shapes_and_finiteness():
    torch.manual_seed(0)
    B, T, H_d, C = 2, 40, 8, 5
    warmup = 4
    mu_full = torch.randn(B, T, H_d, C)
    y_plus = mu_full[:, : T - H_d, :, :] + 0.1 * torch.randn(B, T - H_d, H_d, C)
    pc = compute_per_channel_forecast_metrics(mu_full, y_plus, warmup, H_d)
    assert pc["mse_per_channel"].shape == (B, C)
    assert pc["r2_per_channel"].shape == (B, C)
    assert torch.isfinite(pc["mse_per_channel"]).all()
    assert torch.isfinite(pc["r2_per_channel"]).all()


# ---------------------------------------------------------------------------
# compute_band_forecast_metrics: backwards-compatible alias.
# ---------------------------------------------------------------------------


def test_band_forecast_metrics_partition_idx_and_legacy_alias_match():
    torch.manual_seed(1)
    B, T, H_d, C = 2, 30, 6, 10
    warmup = 3
    mu_full = torch.randn(B, T, H_d, C)
    y_plus = mu_full[:, : T - H_d, :, :] + 0.2 * torch.randn(B, T - H_d, H_d, C)
    partition_idx = {
        "low": np.array([0, 1, 2, 3], dtype=int),
        "high": np.array([4, 5, 6, 7, 8, 9], dtype=int),
    }
    new = compute_band_forecast_metrics(
        mu_full, y_plus, warmup, H_d, partition_idx=partition_idx,
    )
    legacy = compute_band_forecast_metrics(
        mu_full, y_plus, warmup, H_d, band_combined_idx=partition_idx,
    )
    for label in partition_idx:
        assert torch.allclose(new[label]["mse_total"], legacy[label]["mse_total"])
        assert torch.allclose(new[label]["mse_per_horizon"], legacy[label]["mse_per_horizon"])


def test_band_forecast_metrics_rejects_dual_supply():
    torch.manual_seed(2)
    B, T, H_d, C = 1, 20, 4, 4
    mu_full = torch.randn(B, T, H_d, C)
    y_plus = mu_full[:, : T - H_d, :, :]
    partition_idx = {"a": np.array([0, 1], dtype=int)}
    with pytest.raises(TypeError):
        compute_band_forecast_metrics(
            mu_full, y_plus, 2, H_d,
            partition_idx=partition_idx, band_combined_idx=partition_idx,
        )


def test_band_forecast_metrics_rejects_missing_partition():
    torch.manual_seed(3)
    B, T, H_d, C = 1, 20, 4, 4
    mu_full = torch.randn(B, T, H_d, C)
    y_plus = mu_full[:, : T - H_d, :, :]
    with pytest.raises(TypeError):
        compute_band_forecast_metrics(mu_full, y_plus, 2, H_d)
