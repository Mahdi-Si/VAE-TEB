r"""Sprint 0 (S0-T04): checkpoint ``model_class`` tag + pre-instantiation guard.

Checkpoints now record ``model_class`` and a guard runs on the raw checkpoint dict
BEFORE any ``SeqVaeLagAttn(**model_kwargs)`` reconstruction, so a cross-version load
fails with a descriptive ``ValueError`` instead of a cryptic constructor
``TypeError``. Old checkpoints without the field load with a warning (back-compat).
See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` S0-T04.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    check_model_class,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402
    build_model,
    save_checkpoint_v2,
)

_TINY_CFG = {
    "sequence_length": 16,
    "d_model": 16,
    "d_z": 8,
    "horizon": 4,
    "warmup_period": 2,
    "c_y": 87,
    "c_u": 101,
    "use_up_st": True,
    "max_lag": 8,
    "num_heads": 4,
    "d_head": 4,
    "logvar_clamp": [-5.0, 3.0],
}


def test_roundtrip_has_model_class(tmp_path) -> None:
    """``save_checkpoint_v2`` writes ``model_class`` matching the built model."""
    model, kwargs = build_model(_TINY_CFG, torch.device("cpu"))
    ckpt_path = tmp_path / "tiny.ckpt"
    save_checkpoint_v2(
        ckpt_path,
        model=model,
        model_kwargs=kwargs,
        config={},
        data_meta={},
        epoch=0,
        val_loss=float("nan"),
        loss_settings={},
        latent_stats_fitted=False,
    )
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeLagAttnV1"
    # The guard accepts a matching checkpoint without raising.
    check_model_class(blob, "SeqVaeLagAttnV1")


def test_mismatch_raises_before_instantiation() -> None:
    """The guard raises ``ValueError`` before any (failing) construction runs."""
    # ``model_kwargs`` carries a v2-only key that v1's keyword-only ``__init__``
    # would reject with ``TypeError``. The guard must fire first (ValueError).
    blob = {
        "model_class": "SeqVaeLagAttnV2",
        "model_kwargs": {"d_e": 32, "source_scales": (3, 9, 21)},
    }
    with pytest.raises(ValueError):
        check_model_class(blob, "SeqVaeLagAttnV1")
        # Never reached; proves the guard is pre-instantiation.
        SeqVaeLagAttnV1(**blob["model_kwargs"])  # pragma: no cover


def test_missing_field_warns_and_passes() -> None:
    """A pre-guard checkpoint (no ``model_class``) warns but does not raise."""
    blob = {"model_kwargs": {}}
    with pytest.warns(RuntimeWarning):
        check_model_class(blob, "SeqVaeLagAttnV1")


def test_matching_class_is_silent() -> None:
    """A matching ``model_class`` neither raises nor warns."""
    import warnings

    blob = {"model_class": "SeqVaeLagAttnV1", "model_kwargs": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_model_class(blob, "SeqVaeLagAttnV1")


def test_non_dict_checkpoint_is_skipped() -> None:
    """A non-dict checkpoint object is skipped (no false positive)."""
    check_model_class(["not", "a", "dict"], "SeqVaeLagAttnV1")
