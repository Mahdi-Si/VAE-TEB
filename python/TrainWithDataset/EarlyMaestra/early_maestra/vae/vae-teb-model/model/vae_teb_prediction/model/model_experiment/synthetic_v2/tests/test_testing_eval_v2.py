r"""Sprint 6 (S6-T03): testing/ + classifier drop-in on v2.

With the ``SeqVaeLagAttn`` alias monkeypatched to v2 **in-process**, this checks:

* ``testing.base.TestRunner.from_checkpoint`` rebuilds a v2 model from the
  checkpoint's own ``model_kwargs`` (the version-agnostic path) and aligns it
  strict; the analysis-facing forward keys are all present on v2.
* The ``check_model_class`` guard rejects a v2 checkpoint under the v1 alias
  **before** construction, and the legacy (no-``model_kwargs``) path needs a
  config.
* ``precompute_latents.build_vae_from_config`` builds a frozen/eval v2 model
  behind the guard, and its consumer helpers (``encode_only`` / ``kld_tensor``)
  produce latents from the v2 checkpoint.

All CPU, tiny model, in-memory checkpoint. Verifying a real legacy ``_old``
checkpoint is operator-run (none is committed in-tree). See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` S6-T03.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

import model.vae_teb_prediction.testing.base as testing_base  # noqa: E402
from model.vae_teb_prediction.new_classifier.guid_cls_v1 import (  # noqa: E402
    precompute_latents as pcl,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_T = 32

_TINY_V2_KW: Dict[str, Any] = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": 4,
    "warmup_period": 2,
    "c_y": 87,
    "c_u": 101,
    "use_up_st": True,
    "max_lag": 8,
    "num_heads": 2,
    "d_head": 8,
    "dropout": 0.0,
    "decoder_hidden": 16,
    "logvar_clamp": (-5.0, 3.0),
    "mu_scale": 5.0,
    "delta_mu_scale": 3.0,
    "latent_stats_momentum": 0.01,
    "use_entmax": True,
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
    "target_encoder_blocks": 2,
    "target_kernel": 3,
    "target_dilations": (1, 2),
    "source_scales": (3, 5),
    "d_u": 16,
    "d_k": 8,
    "d_e": 8,
    "active_lags": 4,
    "active_lags_warmup": 6,
}

# Forward-dict keys the testing/ analyses (collectors.py) consume.
_ANALYSIS_KEYS = (
    "mu_full", "mu_base", "delta_mu_src", "attn_weights", "te_lag_map",
    "kld_per_t", "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z",
)


def _save_v2_ckpt(path: Path, *, with_kwargs: bool = True,
                  model_class: str = "SeqVaeLagAttnV2") -> None:
    r"""Save a tiny v2 checkpoint in the ``load_checkpoint_strict`` format."""
    model = SeqVaeLagAttnV2(**_TINY_V2_KW)
    blob: Dict[str, Any] = {
        "model_class": model_class,
        "model_state_dict": model.state_dict(),
    }
    if with_kwargs:
        blob["model_kwargs"] = dict(_TINY_V2_KW)
    torch.save(blob, path)


def _random_streams(n: int = 2):
    g = torch.Generator().manual_seed(0)
    y_st = torch.randn(n, _T, 43, generator=g)
    y_ph = torch.randn(n, _T, 44, generator=g)
    u_stream = torch.randn(n, _T, 101, generator=g)
    return y_st, y_ph, u_stream


# ---------------------------------------------------------------------------
# TestRunner.from_checkpoint version-agnostic path
# ---------------------------------------------------------------------------
def test_testrunner_builds_v2_from_model_kwargs(tmp_path, monkeypatch) -> None:
    r"""``from_checkpoint`` rebuilds v2 from ``model_kwargs`` and exposes v2 keys."""
    monkeypatch.setattr(testing_base, "SeqVaeLagAttn", SeqVaeLagAttnV2)
    ckpt = tmp_path / "final.ckpt"
    _save_v2_ckpt(ckpt)

    runner = testing_base.TestRunner.from_checkpoint(
        checkpoint_path=ckpt, output_dir=tmp_path / "out", device=torch.device("cpu"),
    )
    assert isinstance(runner.model, SeqVaeLagAttnV2)
    assert runner.warmup_steps == 2 and runner.horizon == 4 and runner.max_lag == 8

    # The analysis-facing forward keys are all present on the rebuilt v2 model.
    runner.model.eval()
    with torch.no_grad():
        out = runner.model(*_random_streams())
    for key in _ANALYSIS_KEYS:
        assert key in out, key


def test_testrunner_guard_rejects_v2_under_v1(tmp_path, monkeypatch) -> None:
    r"""A v2 checkpoint under the v1 alias raises before any construction."""
    monkeypatch.setattr(testing_base, "SeqVaeLagAttn", SeqVaeLagAttnV1)
    ckpt = tmp_path / "final.ckpt"
    _save_v2_ckpt(ckpt)
    with pytest.raises(ValueError):
        testing_base.TestRunner.from_checkpoint(
            checkpoint_path=ckpt, output_dir=tmp_path / "out",
            device=torch.device("cpu"),
        )


def test_testrunner_legacy_path_needs_config(tmp_path, monkeypatch) -> None:
    r"""A checkpoint without ``model_kwargs`` and no config raises a clear error."""
    monkeypatch.setattr(testing_base, "SeqVaeLagAttn", SeqVaeLagAttnV2)
    ckpt = tmp_path / "legacy.ckpt"
    _save_v2_ckpt(ckpt, with_kwargs=False)
    with pytest.raises(ValueError, match="model_kwargs"):
        testing_base.TestRunner.from_checkpoint(
            checkpoint_path=ckpt, output_dir=tmp_path / "out", config_path=None,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Classifier latent precompute
# ---------------------------------------------------------------------------
def test_precompute_build_vae_v2_frozen_and_guarded(tmp_path, monkeypatch) -> None:
    r"""``build_vae_from_config`` builds a frozen/eval v2 model behind the guard."""
    monkeypatch.setattr(pcl, "SeqVaeLagAttn", SeqVaeLagAttnV2)
    ckpt = tmp_path / "vae.ckpt"
    _save_v2_ckpt(ckpt)

    config = {"vae": {"checkpoint": str(ckpt), "model_kwargs": dict(_TINY_V2_KW)}}
    model = pcl.build_vae_from_config(config, torch.device("cpu"))
    assert isinstance(model, SeqVaeLagAttnV2)
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())

    # The classifier consumer helpers produce latents from the v2 model.
    y_st, y_ph, u_stream = _random_streams()
    enc = model.encode_only(y_st, y_ph, u_stream, sample_z=False)
    for key in ("mu_prior", "logvar_prior", "mu_post", "logvar_post"):
        assert key in enc
    kld = model.kld_tensor(
        enc["mu_prior"], enc["logvar_prior"], enc["mu_post"], enc["logvar_post"],
        mask_warmup=False,
    )
    assert kld.shape == (2, _T, _TINY_V2_KW["d_z"])
    assert torch.isfinite(kld).all()


def test_precompute_guard_rejects_v2_under_v1(tmp_path, monkeypatch) -> None:
    r"""``build_vae_from_config`` rejects a v2 checkpoint under the v1 alias."""
    monkeypatch.setattr(pcl, "SeqVaeLagAttn", SeqVaeLagAttnV1)
    ckpt = tmp_path / "vae.ckpt"
    _save_v2_ckpt(ckpt)
    config = {"vae": {"checkpoint": str(ckpt), "model_kwargs": dict(_TINY_V2_KW)}}
    with pytest.raises(ValueError):
        pcl.build_vae_from_config(config, torch.device("cpu"))
