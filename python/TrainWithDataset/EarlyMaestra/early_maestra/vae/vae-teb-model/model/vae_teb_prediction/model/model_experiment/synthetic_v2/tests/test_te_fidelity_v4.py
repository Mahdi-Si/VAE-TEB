r"""S1-T04: direct-bipolar TE fidelity -- ``te_raw`` tracks ``te_inj``; one-sided distorts more.

Bipolar direct rendering (``raw.direct.one_sided: false``) carries the exact linear $c\to d$
coupling into the raw waveform, so the model-free $\mathrm{TE}_{\mathrm{raw}}$ is positive and
monotone across the ladder and stays closer to $\mathrm{TE}_{\mathrm{inj}}$ than the rectified
one-sided render, whose $\max(\cdot,0)$ clips half the coupling. This is the evidence that
$\mathrm{TE}_{\mathrm{inj}}$ can serve as the sole calibration axis for the bipolar cache.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    generate_cell_raw,
    solve_cell_coupling,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.te_raw_v4 import (
    measure_te_raw_v4,
)

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"
_D = 8
_N = 384
_SEED = 7


def _base_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["benchmarks"]["G1_raw_v4"]["mix"]["inverter"]["n_samples"] = 2000
    return cfg


def _te_raw_direct(cfg: dict, B: float) -> float:
    out = generate_cell_raw(_N, B=B, D=_D, config=cfg, benchmark="G1_raw_v4",
                            seed=_SEED, render_mode="direct")
    return measure_te_raw_v4(out["fhr_raw"], out["up_raw"], D=_D, render_mode="direct",
                             config=cfg, benchmark="G1_raw_v4")["te_raw"]


@pytest.fixture(scope="module")
def fidelity():
    r"""Solve the signal ladder once; measure bipolar te_raw per level and a one-sided pair."""
    base = _base_config()
    ladder = {}
    for te in (1.0, 2.0, 3.0):
        sol = solve_cell_coupling(base, te, _D, benchmark="G1_raw_v4")
        ladder[te] = {"B": float(sol["B_y_scalar"]), "te_inj": float(sol["te_block"])}
        ladder[te]["te_raw_bipolar"] = _te_raw_direct(base, ladder[te]["B"])

    # One-sided render of the strongest cell (same B, D, seed) for the distortion comparison.
    one_cfg = copy.deepcopy(base)
    one_cfg["benchmarks"]["G1_raw_v4"]["raw"]["direct"]["one_sided"] = True
    te_raw_one = _te_raw_direct(one_cfg, ladder[3.0]["B"])
    return {"ladder": ladder, "te_raw_one_sided": te_raw_one}


def test_bipolar_te_raw_is_positive(fidelity) -> None:
    r"""Direct bipolar $\mathrm{TE}_{\mathrm{raw}}$ is positive on every signal level."""
    for te, rec in fidelity["ladder"].items():
        assert rec["te_raw_bipolar"] > 0.0, f"te_inj={te} gave non-positive te_raw"


def test_bipolar_te_raw_is_monotone(fidelity) -> None:
    r"""Direct bipolar $\mathrm{TE}_{\mathrm{raw}}$ is non-decreasing across the TE ladder."""
    te_raws = [fidelity["ladder"][te]["te_raw_bipolar"] for te in (1.0, 2.0, 3.0)]
    assert te_raws == sorted(te_raws)


def test_one_sided_deviates_more_than_bipolar(fidelity) -> None:
    r"""Rectified one-sided rendering deviates from ``te_inj`` more than bipolar (same cell)."""
    rec = fidelity["ladder"][3.0]
    te_inj = rec["te_inj"]
    dev_bipolar = abs(rec["te_raw_bipolar"] - te_inj) / te_inj
    dev_one_sided = abs(fidelity["te_raw_one_sided"] - te_inj) / te_inj
    assert dev_one_sided > dev_bipolar
