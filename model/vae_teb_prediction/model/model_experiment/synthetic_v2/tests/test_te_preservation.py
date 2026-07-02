r"""Tests for the Sprint 3 three-TE de-risk probes (S3-T02…S3-T06).

Covers the coupled-channel slice (S3-T02), the ``TE_raw`` raw-domain probe (S3-T04,
pure NumPy), the pre-flight harness + fatal gate (S3-T05), and the recovery sweep
(S3-T06). The fs-correct ``frac_Phi`` / ``TE_scat`` strong/null check (S3-T03) uses the
render knobs recovered by the Sprint 3 sweep.

The real scattering transform is expensive, so the full-shape adapter is built once per
module and reused; the ``TE_raw`` and slice checks need no transform. Probe sample counts
are kept small but above the R0 ill-conditioning floor ($n_{\mathrm{test}} > H = 30$).
See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 3.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    build_dataset_v2 as bd,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import eval_v2
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    raw_generators as rg,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (
    scattering_adapter as sa,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"

# Probe sample count: n_test = 0.3 * N must exceed H * M = 30 (else R0 is
# ill-conditioned). N = 192 -> n_test ~= 58.
_N = 192
_STRONG_D = 8

# Render knobs recovered by the Sprint 3 sweep (S3-T06): raising the carrier to
# 0.06 Hz lifts frac_Phi from ~0.3 (default 0.02 Hz) to ~1.12 at n=384, with the
# physiological modulation depth (am_offset_ratio=4.0) and contraction rhythm
# (omega=0.10) unchanged. The S3-T03 strong-cell check runs at these knobs; the bar
# is the frac_threshold (0.7) with headroom for the smaller test sample count.
_RECOVERED = {"f_pulse": 0.06, "am_offset_ratio": 4.0, "omega": 0.10}
_STRONG_FRAC_MIN = 0.7


@pytest.fixture(scope="module")
def config() -> dict:
    r"""Load the parsed ``config_synth_v2.yaml``."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _fast_inverter(cfg: dict) -> dict:
    r"""Cheapen the inverter Monte-Carlo (the slow part of enumeration)."""
    cfg = copy.deepcopy(cfg)
    cfg["benchmarks"]["G1_raw"]["mix"]["inverter"]["n_samples"] = 4000
    return cfg


@pytest.fixture(scope="module")
def adapter(config) -> "sa.ScatteringAdapter":
    r"""The production-shape scattering adapter, built once and reused."""
    return sa.ScatteringAdapter(config)


# ---------------------------------------------------------------------------
# S3-T02: coupled-channel slicing wrapper
# ---------------------------------------------------------------------------


def test_slice_coupled_channels_shapes() -> None:
    r"""The slice returns single-channel (n, T, 1) arrays cut at the coupled index."""
    rng = np.random.default_rng(0)
    fields = {
        "fhr_st": rng.standard_normal((5, 300, 43)),
        "up_st": rng.standard_normal((5, 300, 43)),
    }
    coupled = {"fhr_st": 26, "up_st": 26, "hz": 0.0196, "xi": 0.0049}
    Y, U = eval_v2.slice_coupled_channels(fields, coupled)
    assert Y.shape == (5, 300, 1)
    assert U.shape == (5, 300, 1)
    assert np.array_equal(Y[:, :, 0], fields["fhr_st"][:, :, 26])
    assert np.array_equal(U[:, :, 0], fields["up_st"][:, :, 26])


# ---------------------------------------------------------------------------
# S3-T04: TE_raw probe (pure NumPy, no GPU)
# ---------------------------------------------------------------------------


def test_teraw_null_and_signal(config) -> None:
    r"""Demodulated ``TE_raw`` is positive on a strong cell and ~0 on a null cell."""
    strong = rg.generate_cell_raw(_N, B=3.0, D=_STRONG_D, config=config, seed=0, te_inj=3.0)
    null = rg.generate_cell_raw(_N, B=0.0, D=_STRONG_D, config=config, seed=1, te_inj=0.0)

    # Default demodulate=True extracts the carrier-band amplitude envelope where the AM
    # coupling lives (the raw-domain analog of the scattering modulus).
    te_strong = eval_v2.measure_te_raw(
        strong["fhr_raw"], strong["up_raw"], D=_STRONG_D, config=config
    )["te_raw"]
    te_null = eval_v2.measure_te_raw(
        null["fhr_raw"], null["up_raw"], D=_STRONG_D, config=config
    )["te_raw"]

    assert np.isfinite(te_strong) and np.isfinite(te_null)
    assert te_strong > 0.1         # the AM coupling is recovered in the raw envelope
    assert te_null < 0.1           # independent dressing alone -> ~0 (may be slightly < 0)
    assert te_strong > te_null

    # The non-demodulated control reads ~0 even for the strong cell: a linear probe cannot
    # extract amplitude coupling from the phase-carrying carrier.
    te_strong_nodemod = eval_v2.measure_te_raw(
        strong["fhr_raw"], strong["up_raw"], D=_STRONG_D, config=config, demodulate=False
    )["te_raw"]
    assert te_strong_nodemod < te_strong


# ---------------------------------------------------------------------------
# S3-T03: frac_Phi / TE_scat probe (real transform)
# ---------------------------------------------------------------------------


def _strong_cell_fields(config, adapter, knobs):
    r"""Solve + generate + transform one strong cell at the given render knobs."""
    cfg = eval_v2._override_render_config(
        _fast_inverter(config), "G1_raw",
        f_pulse=knobs["f_pulse"], am_offset_ratio=knobs["am_offset_ratio"],
        omega=knobs["omega"],
    )
    sol = bd.solve_cell_coupling(cfg, 3.0, _STRONG_D)
    cell = bd.CellV2(cell_id=0, target_te=3.0, D=_STRONG_D,
                     B_y_scalar=float(sol["B_y_scalar"]),
                     te_block_realised=float(sol["te_block"]))
    raw = bd.generate_pilot_samples(cell, _N, "train", cfg)
    fields, _ = adapter.transform_and_normalise(raw["fhr_raw"], raw["up_raw"])
    coupled = adapter.coupled_channel_indices(f_pulse=knobs["f_pulse"])
    return cfg, cell, fields, coupled


def test_fracphi_strong_and_null(config, adapter) -> None:
    r"""On the recovered strong cell ``frac_Phi`` clears the bar; a null gives ``TE_scat~=0``."""
    cfg, cell, fields, coupled = _strong_cell_fields(config, adapter, _RECOVERED)
    scat = eval_v2.measure_te_scat(fields, cell, coupled, config=cfg)
    assert scat["frac_phi"] is not None
    assert scat["frac_phi"] >= _STRONG_FRAC_MIN
    # The two-sided gate exposes the upper bound + overshoot flag (Fix 2 / §14.3).
    for key in ("frac_upper", "frac_over_one", "passes"):
        assert key in scat

    # Null cell at the same knobs -> the coupled channel carries no source info.
    null_cell = bd.CellV2(cell_id=1, target_te=0.0, D=_STRONG_D,
                          B_y_scalar=0.0, te_block_realised=0.0)
    null_raw = bd.generate_pilot_samples(null_cell, _N, "train", cfg)
    null_fields, _ = adapter.transform_and_normalise(null_raw["fhr_raw"], null_raw["up_raw"])
    null_scat = eval_v2.measure_te_scat(null_fields, null_cell, coupled, config=cfg)
    assert null_scat["frac_phi"] is None       # te_inj = 0 -> undefined ratio
    # ~0 realizable TE from dressing alone. The held-out R0 estimator is noisy at the small
    # pilot _N (=192) and returns a *negative* gain here (definitionally non-coupling; real
    # spurious coupling would be a sizeable positive value), so the tolerance covers the
    # pilot noise floor -- still far below the strong cell's TE~3.
    assert null_scat["te_scat"] < 0.4          # no spurious positive coupling from dressing


def test_two_sided_gate_summary() -> None:
    r"""``_summarise_preflight`` is two-sided: over-one cells fail and are flagged (Fix 2)."""
    per_cell = {
        0: {"cell_id": 0, "te_inj": 0.0, "frac_phi": None},   # null (ignored)
        1: {"cell_id": 1, "te_inj": 1.0, "frac_phi": 0.95},   # in band -> pass
        2: {"cell_id": 2, "te_inj": 1.0, "frac_phi": 0.40},   # below lo -> fail (low)
        3: {"cell_id": 3, "te_inj": 1.0, "frac_phi": 2.10},   # above hi -> fail (over-one)
    }
    s = eval_v2._summarise_preflight(per_cell, frac_threshold=0.7, frac_upper=1.3)
    assert s["failing_cell_ids"] == [2, 3]
    assert s["over_one_cell_ids"] == [3]
    assert s["n_frac_over_one"] == 1
    assert s["headline_pass"] is False
    assert s["n_signal_cells"] == 3 and s["n_null_cells"] == 1
    # A lower-bound-only gate (frac_upper=inf) accepts the over-one cell.
    s_lo = eval_v2._summarise_preflight(per_cell, frac_threshold=0.7, frac_upper=float("inf"))
    assert s_lo["failing_cell_ids"] == [2]
    assert s_lo["n_frac_over_one"] == 0


# ---------------------------------------------------------------------------
# S3-T05: pre-flight harness + fatal gate
# ---------------------------------------------------------------------------


def test_preflight_writes_json(config, adapter, tmp_path) -> None:
    r"""The pre-flight runs the three probes per cell and writes ``realizability.json``."""
    cfg = _fast_inverter(config)
    cfg["benchmarks"]["G1_raw"]["eval"]["realizability"]["pilot"] = {
        "target_te_grid": [0.0, 2.0], "lag_grid": [8], "n_per_cell": _N,
    }
    result = eval_v2.run_realizability_preflight(
        cfg, pilot=True, out_dir=tmp_path, adapter=adapter, print_table=False
    )
    assert (tmp_path / "realizability.json").exists()
    per_cell = result["per_cell"]
    assert len(per_cell) == 2
    for cell in per_cell.values():
        for key in ("te_inj", "te_raw", "te_scat", "frac_phi", "D"):
            assert key in cell
    # The null cell has frac_phi = None; the signal cell has a finite frac_phi.
    fracs = {c["target_te"]: c["frac_phi"] for c in per_cell.values()}
    assert fracs[0.0] is None
    assert fracs[2.0] is not None and np.isfinite(fracs[2.0])
    assert "headline_pass" in result["summary"]


def test_preflight_fatal_gate_raises(config, adapter) -> None:
    r"""With an impossible threshold and ``fatal=true`` the gate raises."""
    cfg = _fast_inverter(config)
    ev = cfg["benchmarks"]["G1_raw"]["eval"]["realizability"]
    ev["fatal"] = True
    ev["frac_threshold"] = 2.0   # unreachable -> the signal cell fails
    ev["pilot"] = {"target_te_grid": [0.0, 2.0], "lag_grid": [8], "n_per_cell": _N}
    with pytest.raises(RuntimeError, match="frac_Phi gate FAILED"):
        eval_v2.run_realizability_preflight(
            cfg, pilot=True, adapter=adapter, print_table=False
        )


# ---------------------------------------------------------------------------
# S3-T06: recovery sweep
# ---------------------------------------------------------------------------


def test_recovery_sweep(config, adapter, tmp_path) -> None:
    r"""The sweep scores each render setting and picks the best; writes ``recovery.json``."""
    cfg = _fast_inverter(config)
    cfg["benchmarks"]["G1_raw"]["eval"]["realizability"]["recovery"] = {
        "f_pulse_grid": [0.02, 0.06],
        "am_offset_ratio_grid": [1.5],
        "omega_grid": [0.06],
        "n_per_cell": _N,
        "target_te": 3.0,
        "D": _STRONG_D,
        "lag1_autocorr_floor": 0.9,
    }
    result = eval_v2.sweep_render_knobs(
        cfg, out_dir=tmp_path, adapter=adapter, print_table=False
    )
    assert (tmp_path / "recovery.json").exists()
    assert len(result["table"]) == 2
    for row in result["table"]:
        for key in ("f_pulse", "frac_phi", "margin_peak", "lag1_autocorr", "valid"):
            assert key in row
    chosen = result["chosen"]
    assert chosen is not None
    assert chosen["frac_phi"] is not None
    # The higher carrier should preserve at least as much as the lower one.
    by_fp = {row["f_pulse"]: row["frac_phi"] for row in result["table"]}
    assert by_fp[0.06] >= by_fp[0.02] - 0.05


# ---------------------------------------------------------------------------
# S7-T04: pulse_train render variant frac_Phi (measured vs am_carrier)
# ---------------------------------------------------------------------------


def test_pulse_train_fracphi_measured(config, adapter) -> None:
    r"""``render_mode='pulse_train'`` yields a finite, positive measured ``frac_Phi``.

    The pulse_train (raised-cosine event train) is the waveform-realistic variant; its
    preservation is generally lower than am_carrier and is *measured*, not assumed equal
    (§7.3). This runs the same recovered knobs and asserts the probe returns a finite
    positive frac_Phi so the two render modes can be reported side by side.
    """
    cfg = eval_v2._override_render_config(
        _fast_inverter(config), "G1_raw",
        f_pulse=_RECOVERED["f_pulse"], am_offset_ratio=_RECOVERED["am_offset_ratio"],
        omega=_RECOVERED["omega"],
    )
    cfg["benchmarks"]["G1_raw"]["raw"]["render_mode"] = "pulse_train"
    sol = bd.solve_cell_coupling(cfg, 3.0, _STRONG_D)
    cell = bd.CellV2(cell_id=0, target_te=3.0, D=_STRONG_D,
                     B_y_scalar=float(sol["B_y_scalar"]),
                     te_block_realised=float(sol["te_block"]))
    raw = bd.generate_pilot_samples(cell, _N, "train", cfg)
    assert raw["meta"]["render_mode"] == "pulse_train"
    fields, _ = adapter.transform_and_normalise(raw["fhr_raw"], raw["up_raw"])
    coupled = adapter.coupled_channel_indices(f_pulse=_RECOVERED["f_pulse"])
    scat = eval_v2.measure_te_scat(fields, cell, coupled, config=cfg)
    assert scat["frac_phi"] is not None and np.isfinite(scat["frac_phi"])
    assert scat["frac_phi"] > 0.0
