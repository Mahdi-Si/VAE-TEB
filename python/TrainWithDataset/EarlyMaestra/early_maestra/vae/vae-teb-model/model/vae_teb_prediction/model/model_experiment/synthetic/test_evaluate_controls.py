"""Pytest checks for the Sprint 6.3 null-control helper.

Covers :func:`evaluate_te.reverse_source_batch` semantics and the new
``k_bar_reversed`` column emitted by :func:`evaluate_te.evaluate_checkpoint`
via :func:`evaluate_te._collect_diagnostics`.

Run from the repo root with ``python -m pytest``.
"""

from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    AttributeDict,
    SyntheticTEDataset,
    make_dataloader,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.build_dataset import (
    build_dataset,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te import (
    _SUMMARY_FIELDS,
    _collect_diagnostics,
    evaluate_checkpoint,
    reverse_source_batch,
    shuffle_source_batch,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    resolve_active_benchmark,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1

_CONFIG_PATH = Path(__file__).resolve().parent / "config_synth.yaml"
_N_TRAIN, _N_VAL, _N_TEST = 6, 4, 4


def _tiny_g1_config(tmp_path: Path) -> Dict[str, Any]:
    """Resolve a tiny G1 config rooted at ``tmp_path`` (per-test data dir)."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["experiment"]["benchmark"] = "G1"
    raw["experiment"]["tag"] = "test_eval_ctrl"
    config = resolve_active_benchmark(raw)
    config["paths"]["data_dir"] = str(tmp_path / "data")
    config["paths"]["results_dir"] = str(tmp_path / "results")
    config["data"]["n_train"] = _N_TRAIN
    config["data"]["n_val"] = _N_VAL
    config["data"]["n_test"] = _N_TEST
    config["data"]["te_n_samples"] = 2_000
    config["optim"]["batch_size"] = 2
    return config


# --- reverse_source_batch (pure helper) --------------------------------------

def test_reverse_source_batch_flips_up_along_time():
    """``up_st`` / ``up_ph`` get flipped on dim 1; target / weight pass through."""
    torch.manual_seed(0)
    batch = AttributeDict({
        "fhr_st": torch.randn(2, 8, 43),
        "fhr_ph": torch.randn(2, 8, 44),
        "up_st":  torch.randn(2, 8, 43),
        "up_ph":  torch.randn(2, 8, 58),
        "weight": torch.ones(2, 8),
    })
    rev = reverse_source_batch(batch)
    assert isinstance(rev, AttributeDict)
    assert torch.equal(rev["up_st"], batch.up_st.flip(dims=[1]))
    assert torch.equal(rev["up_ph"], batch.up_ph.flip(dims=[1]))
    # Target streams + weight are unchanged (identity object share is fine).
    assert rev["fhr_st"] is batch.fhr_st
    assert rev["fhr_ph"] is batch.fhr_ph
    assert rev["weight"] is batch.weight


def test_reverse_is_idempotent_under_double_apply():
    """Two reversals along the time axis restore the original source tensors."""
    torch.manual_seed(1)
    batch = AttributeDict({
        "fhr_st": torch.zeros(1, 4, 43),
        "fhr_ph": torch.zeros(1, 4, 44),
        "up_st":  torch.randn(1, 4, 43),
        "up_ph":  torch.randn(1, 4, 58),
        "weight": torch.ones(1, 4),
    })
    double_rev = reverse_source_batch(reverse_source_batch(batch))
    assert torch.equal(double_rev["up_st"], batch.up_st)
    assert torch.equal(double_rev["up_ph"], batch.up_ph)


def test_reverse_is_distinct_from_shuffle():
    """Reverse and shuffle change the source in different ways."""
    torch.manual_seed(2)
    batch = AttributeDict({
        "fhr_st": torch.zeros(4, 6, 43),
        "fhr_ph": torch.zeros(4, 6, 44),
        "up_st":  torch.randn(4, 6, 43),
        "up_ph":  torch.randn(4, 6, 58),
        "weight": torch.ones(4, 6),
    })
    rev = reverse_source_batch(batch)
    shuf = shuffle_source_batch(batch)
    # Reverse never permutes along the batch axis: row 0 of the reversed up_st
    # equals row 0 of the original up_st flipped on the time axis. The shuffle
    # permutes batches, so it does NOT equal that flip for at least one row
    # (a bs-of-4 cycle covers any non-identity permutation).
    assert torch.equal(rev["up_st"][0], batch.up_st[0].flip(dims=[0]))
    assert not torch.equal(shuf["up_st"], rev["up_st"])


# --- _collect_diagnostics emits k_bar_reversed -------------------------------

def test_diagnostics_emit_k_bar_reversed(tmp_path):
    r"""``_collect_diagnostics`` returns a finite ``k_bar_reversed`` on a G1 loader."""
    config = _tiny_g1_config(tmp_path)
    out_dir = build_dataset(config, force=True)
    ds = SyntheticTEDataset(out_dir / "test.npz")
    loader = make_dataloader(ds, batch_size=2, shuffle=False)

    torch.manual_seed(0)
    model = SeqVaeLagAttnV1()
    model.eval()
    diag = _collect_diagnostics(
        model, loader, torch.device("cpu"), warmup=model.warmup_period,
    )
    for key in ("per_dim_kl", "mu_post_prior_gap", "attn_entropy",
                "k_bar_shuffled", "k_bar_reversed"):
        assert key in diag, key
    # The new control is finite (an untrained model still produces a
    # well-defined per-batch KL; the value need not be small).
    assert isinstance(diag["k_bar_reversed"], float)
    assert torch.isfinite(torch.tensor(diag["k_bar_reversed"]))


# --- evaluate_checkpoint row carries k_bar_reversed --------------------------

def test_evaluate_checkpoint_row_has_k_bar_reversed(tmp_path):
    """The flat metrics row written by ``evaluate_checkpoint`` includes the column."""
    config = _tiny_g1_config(tmp_path)
    out_dir = build_dataset(config, force=True)

    # Build a stand-in checkpoint exactly like train_minimal.save_checkpoint:
    # only the model state + the metadata evaluate_checkpoint reads are needed.
    torch.manual_seed(0)
    model = SeqVaeLagAttnV1()
    ds = SyntheticTEDataset(out_dir / "test.npz")
    ckpt_path = tmp_path / "results" / "G1" / "test_eval_ctrl_run" / "final.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": {},  # SeqVaeLagAttnV1 defaults match V2-D1 shapes
            "data_meta": dict(ds.meta),
            "config": {
                "experiment": {"benchmark": "G1", "tag": "test_eval_ctrl"},
                "optim": {"batch_size": 2},
            },
            "loss_settings": {
                "beta": float(config["loss"]["kld_beta"]),
                "lambda_full": float(config["loss"]["lambda_full"]),
                "lambda_base": float(config["loss"]["lambda_base"]),
                "likelihood": str(config["loss"].get("likelihood", "mse")),
                "sigma_obs": config["loss"].get("sigma_obs", 1.0),
                "free_bits": float(config["loss"].get("free_bits", 0.0)),
            },
            "epoch": 0,
            "latent_stats_fitted": False,
        },
        ckpt_path,
    )

    row = evaluate_checkpoint(
        ckpt_path, config, device=torch.device("cpu"),
        data_tag="test_eval_ctrl", batch_size=2,
    )
    assert "k_bar_reversed" in row
    assert "k_bar_reversed" in _SUMMARY_FIELDS
    assert isinstance(row["k_bar_reversed"], float)
    assert torch.isfinite(torch.tensor(row["k_bar_reversed"]))
