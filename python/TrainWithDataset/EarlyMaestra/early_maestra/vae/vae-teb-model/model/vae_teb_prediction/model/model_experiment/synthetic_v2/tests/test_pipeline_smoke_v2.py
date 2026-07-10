r"""Sprint 6 (S6-T01): synthetic_v2 pipeline drop-in validation on v2.

With the ``SeqVaeLagAttn`` alias monkeypatched to :class:`SeqVaeLagAttnV2`
**in-process** (the committed default stays v1; see
``vae-teb-lag-attn-v2-spec-and-sprints.md`` decision 1), this exercises the
primary injected-TE pipeline end-to-end on a tiny CPU fixture:

* :func:`pl_module_v2.build_model` merges the nested ``model.v2`` overlay only
  under the v2 alias and ignores it under v1 (the config drop-in mechanism).
* :func:`pl_module_v2.train_v2` (pilot overrides) trains v2 and writes a
  ``final.ckpt`` stamped ``model_class == "SeqVaeLagAttnV2"``.
* :func:`eval_v2.run_eval` grades the trained v2 checkpoint and writes
  ``metrics.json`` (the eval half of the pipeline).
* The checkpoint reloads strict into a fresh v2 model (the drop-in round trip).

Everything runs on CPU against a tiny in-test fixture cache + a tiny v2 model
(small ``d_model`` / ``d_z`` / ``horizon`` with the production ``c_y=87`` /
``c_u=101`` channel counts), mirroring ``test_train_v2`` / ``test_eval_v2``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

# Force the repo root ahead of the sibling ``model/vae_teb_prediction`` on ``sys.path``
# (which lacks ``utils.custom_logger``) so the model imports resolve under pytest --
# the same guard ``pl_module_v2`` applies at import time. See test_train_v2.
_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    eval_v2,
    pl_module_v2 as plm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.build_dataset_v2 import (  # noqa: E402
    resolve_cache_dir,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v2.yaml"

# Channel counts fixed by the transform / model contract.
_C_FHR_ST, _C_FHR_PH, _C_UP_ST, _C_UP_PH = 43, 44, 43, 58

_T = 32
_HORIZON = 4
_WARMUP = 2
_MAX_LAG = 8
_DELAY = 4

# Flat v1-valid tiny keys (mirrors test_train_v2's tiny model). Shared by both
# alias arms; the v2-only tuning lives in the nested ``v2`` overlay below.
_TINY_FLAT: Dict[str, Any] = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": _HORIZON,
    "warmup_period": _WARMUP,
    "c_y": _C_FHR_ST + _C_FHR_PH,   # 87
    "c_u": _C_UP_ST + _C_UP_PH,     # 101
    "use_up_st": True,
    "max_lag": _MAX_LAG,
    "num_heads": 2,
    "d_head": 8,                    # num_heads * d_head == d_model
    "lstm_layers": 1,
    "dropout": 0.0,
    "decoder_hidden": 16,
    "logvar_clamp": [-5.0, 3.0],
    "mu_scale": 5.0,
    "delta_mu_scale": 3.0,
    "latent_stats_momentum": 0.01,
    "use_entmax": False,            # v1 default; the overlay flips it for v2
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
    "encoder_extra_dilations": [],
}

# v2-only overlay: small, entmax-on, so the v2 source path runs fast on CPU.
_TINY_V2_OVERLAY: Dict[str, Any] = {
    "use_entmax": True,
    "target_encoder_blocks": 2,
    "target_kernel": 3,
    "target_dilations": [1, 2],
    "source_scales": [3, 5],
    "d_u": 16,
    "d_k": 8,
    "d_e": 8,
    "active_lags": 4,
    "active_lags_warmup": 6,
    "kappa_z": 0.05,
    "lambda_tv": 1.0e-4,
    "lambda_ent": 0.0,
    "context_dim": 5,
    "step_seconds": 4.0,
    "delta_up_seconds": 20.0,
    "use_crossphase_bias": False,
    "use_outcome_head": False,
}


def _tiny_v2_model() -> Dict[str, Any]:
    r"""A flat + nested-``v2`` tiny model block for the drop-in smoke."""
    block = copy.deepcopy(_TINY_FLAT)
    block["v2"] = copy.deepcopy(_TINY_V2_OVERLAY)
    return block


@pytest.fixture
def use_v2_alias(monkeypatch) -> None:
    r"""Activate the v2 alias in-process (build_model reads this module global)."""
    monkeypatch.setattr(plm, "SeqVaeLagAttn", SeqVaeLagAttnV2)


@pytest.fixture
def force_cpu(monkeypatch) -> None:
    r"""Force the training driver / eval onto CPU (deterministic, no GPU contention)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# ---------------------------------------------------------------------------
# Config-overlay mechanism
# ---------------------------------------------------------------------------
def test_config_has_v2_overlay_and_curriculum() -> None:
    r"""The shipped config carries the ``model.v2`` overlay and a v2 curriculum."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    v2 = cfg["model"].get("v2")
    assert isinstance(v2, dict), "config_synth_v2.yaml is missing the model.v2 overlay"
    assert v2["use_entmax"] is True
    assert v2["source_scales"] == [3, 9, 21]
    assert v2["use_crossphase_bias"] is False and v2["use_outcome_head"] is False
    cur = cfg["curriculum"]
    assert cur["enabled"] is True and len(cur["stages"]) == 3


def test_v2_overlay_merged_under_v2_ignored_under_v1(monkeypatch) -> None:
    r"""``build_model`` applies the overlay under v2 and drops it under v1."""
    # v2 alias: overlay merged -> entmax on, v2 attributes present.
    monkeypatch.setattr(plm, "SeqVaeLagAttn", SeqVaeLagAttnV2)
    model_v2, kwargs_v2 = plm.build_model(_tiny_v2_model(), torch.device("cpu"))
    assert isinstance(model_v2, SeqVaeLagAttnV2)
    assert model_v2.use_entmax is True
    assert model_v2.active_lags == 4
    assert model_v2.source_scales == (3, 5)
    assert "v2" not in kwargs_v2                      # overlay stripped from stored kwargs
    assert kwargs_v2["use_entmax"] is True            # overlay value won

    # v1 alias: overlay dropped, v2-only keys filtered, use_entmax stays False.
    # (v1 does not expose ``use_entmax`` as an attribute, so assert via kwargs.)
    monkeypatch.setattr(plm, "SeqVaeLagAttn", SeqVaeLagAttnV1)
    model_v1, kwargs_v1 = plm.build_model(_tiny_v2_model(), torch.device("cpu"))
    assert isinstance(model_v1, SeqVaeLagAttnV1)
    assert kwargs_v1["use_entmax"] is False           # flat value, overlay ignored
    assert "v2" not in kwargs_v1 and "source_scales" not in kwargs_v1
    assert not hasattr(model_v1, "source_scales")     # v2-only attrs absent under v1
    assert not hasattr(model_v1, "active_lags")


# ---------------------------------------------------------------------------
# Fixture cache (native + provenance fields consumed by run_eval)
# ---------------------------------------------------------------------------
def _write_split(path: Path, cells: List[Dict[str, Any]], *, per_cell: int,
                 seed: int) -> None:
    r"""Write one split ``.npz`` split evenly across ``cells`` (native + provenance)."""
    rng = np.random.default_rng(seed)
    n = per_cell * len(cells)
    arrays: Dict[str, np.ndarray] = {
        "fhr_st": rng.standard_normal((n, _T, _C_FHR_ST)).astype(np.float32),
        "fhr_ph": rng.standard_normal((n, _T, _C_FHR_PH)).astype(np.float32),
        "up_st": rng.standard_normal((n, _T, _C_UP_ST)).astype(np.float32),
        "up_ph": rng.standard_normal((n, _T, _C_UP_PH)).astype(np.float32),
        "weight": np.ones((n, _T), dtype=np.float32),
        "true_lag_tt": np.zeros((n, _T), dtype=np.float32),
    }
    te_true = np.empty(n, np.float32)
    te_scat = np.empty(n, np.float32)
    frac = np.empty(n, np.float32)
    delay = np.empty(n, np.int16)
    cid = np.empty(n, np.int16)
    held = np.zeros(n, np.int8)
    for i, cell in enumerate(cells):
        sl = slice(i * per_cell, (i + 1) * per_cell)
        te_true[sl] = cell["te_inj"]
        te_scat[sl] = cell["te_scat"]
        frac[sl] = cell["frac_phi"]
        delay[sl] = cell["delay"]
        cid[sl] = cell["cell_id"]
        arrays["true_lag_tt"][sl] = cell["delay"]
    arrays.update({
        "sample_te_true": te_true, "sample_te_scat": te_scat,
        "sample_frac_phi": frac, "sample_delay": delay,
        "sample_cell_id": cid, "sample_held_out": held,
    })
    np.savez(path, **arrays)


@pytest.fixture
def tiny_v2_config(tmp_path) -> Dict[str, Any]:
    r"""A trimmed config with a tiny v2 model + a 2-cell fixture cache.

    The curriculum is disabled so a 1-epoch pilot exercises the full v2 source
    path (an enabled curriculum would keep a 1-epoch run in the baseline-only
    Stage 1).
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = copy.deepcopy(yaml.safe_load(handle))
    cfg["model"] = _tiny_v2_model()
    cfg["curriculum"] = {"enabled": False, "stages": []}
    cfg["experiment"]["tag"] = "test_smoke_v2"
    cfg["experiment"]["benchmark"] = "G1_raw"
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    cfg["dataset"] = {"num_workers": 0, "pin_memory": False,
                      "persistent_workers": False, "mmap": "auto"}
    cfg["optim"]["batch_size"] = 4
    cfg["optim"]["epochs"] = 1
    cfg["optim"]["lr_milestones"] = []

    cells = [
        {"cell_id": 0, "te_inj": 0.0, "te_scat": -0.1, "frac_phi": float("nan"),
         "delay": _DELAY},
        {"cell_id": 1, "te_inj": 2.0, "te_scat": 1.8, "frac_phi": 0.9,
         "delay": _DELAY},
    ]
    cache_dir = resolve_cache_dir(cfg, benchmark="G1_raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_split(cache_dir / "train.npz", cells, per_cell=6, seed=0)
    _write_split(cache_dir / "val.npz", cells, per_cell=3, seed=1)
    _write_split(cache_dir / "test.npz", cells, per_cell=4, seed=2)
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump({
            "te_true": 1.0, "tag": "test_smoke_v2", "benchmark": "G1_raw",
            "true_lag_band": list(range(max(0, _DELAY - _HORIZON), _DELAY)),
        }, handle)
    return cfg


# ---------------------------------------------------------------------------
# End-to-end drop-in: train (v2) -> checkpoint identity -> eval -> round trip
# ---------------------------------------------------------------------------
def test_pipeline_train_eval_v2_dropin(tiny_v2_config, use_v2_alias, force_cpu) -> None:
    r"""``train_v2`` + ``run_eval`` run clean on v2 and stamp ``model_class``."""
    overrides = {
        "epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 1,
        "batch_size": 4,
        "devices": 1,
        "progress_bar": False,
    }
    result = plm.train_v2(tiny_v2_config, overrides, benchmark="G1_raw")

    ckpt_path = Path(result["checkpoint"])
    assert ckpt_path.is_file() and ckpt_path.name == "final.ckpt"

    # The checkpoint self-identifies as v2 (the drop-in class tag).
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeLagAttnV2"
    assert "source_scales" in blob["model_kwargs"]      # v2 kwargs persisted

    # Eval half: grade the trained v2 checkpoint and write metrics.json.
    results_dir = ckpt_path.parent
    metrics = eval_v2.run_eval(
        tiny_v2_config, benchmark="G1_raw", split="test", out_dir=results_dir,
        batch_size=4, device="cpu",
    )
    assert (results_dir / "metrics.json").is_file()
    assert metrics["n_cells"] == 2

    # Drop-in round trip: the v2 checkpoint reloads strict into a fresh v2 model.
    from train.graph_models_utils import load_checkpoint_strict

    fresh, _ = plm.build_model(tiny_v2_config["model"], torch.device("cpu"))
    assert isinstance(fresh, SeqVaeLagAttnV2)
    loaded = load_checkpoint_strict(fresh, str(ckpt_path), map_location="cpu")
    assert loaded is not None


def test_checkpoint_guard_rejects_v2_under_v1(tiny_v2_config, use_v2_alias,
                                              force_cpu, monkeypatch) -> None:
    r"""``check_model_class`` refuses a v2 checkpoint when the alias is flipped to v1."""
    from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import check_model_class

    result = plm.train_v2(
        tiny_v2_config,
        {"epochs": 1, "limit_train_batches": 2, "limit_val_batches": 1,
         "batch_size": 4, "devices": 1, "progress_bar": False},
        benchmark="G1_raw",
    )
    blob = torch.load(str(result["checkpoint"]), map_location="cpu",
                      weights_only=False)
    # A v2 checkpoint loaded under the v1 alias name must raise before any rebuild.
    with pytest.raises(ValueError):
        check_model_class(blob, SeqVaeLagAttnV1.__name__)
    # Same blob under the correct v2 name passes.
    check_model_class(blob, SeqVaeLagAttnV2.__name__)
