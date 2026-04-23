"""Guard test: ``freeze_vae=False`` raises a clear NotImplementedError.

Phase 7 (live VAE + two-stage fine-tune) is scaffolded but not wired. The
trainer must fail fast with an informative message rather than producing
silently broken behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from model.vae_teb_prediction.new_classifier.guid_cls_v1.single_fold_trainer import (  # noqa: E402
    train_fold,
)


def test_live_vae_path_raises(tmp_path: Path) -> None:
    """A config with ``vae.freeze_vae=False`` must raise NotImplementedError."""
    cfg = {
        "general_config": {
            "tag": "live_vae_guard",
            "folders_config": {"out_dir_base": str(tmp_path)},
            "epochs": 1,
            "seed": 42,
            "cuda_devices": [0],
        },
        "vae": {
            "checkpoint": "/dev/null",
            "freeze_vae": False,
            "model_kwargs": {
                "sequence_length": 300,
                "d_model": 128,
                "d_z": 24,
                "horizon": 30,
                "warmup_period": 30,
                "c_y": 87,
                "c_u": 101,
                "use_up_st": True,
                "max_lag": 90,
                "num_heads": 4,
                "d_head": 32,
                "lstm_layers": 2,
                "dropout": 0.1,
                "decoder_hidden": 128,
                "use_entmax": False,
                "attention_grad_checkpoint": False,
            },
        },
        "model_config": {
            "classifier": {
                "warmup_left": 30,
                "warmup_right": 30,
                "cross_delivery_censoring": True,
            },
        },
        "dataset_config": {
            "kfold_base_path": str(tmp_path),
            "num_folds": 1,
            "min_samples_per_guid": 3,
            "min_valid_weight_fraction": 0.1,
        },
        "training": {},
    }
    with pytest.raises(NotImplementedError, match="freeze_vae=False"):
        train_fold(fold_id=1, config=cfg, gpu_id=0, output_dir_override=str(tmp_path))
