r"""Sprint 5 (S5-T01): build SeqVaeLagAttnV2 from config_lag_attn_v2.yaml.

With the trainer's ``SeqVaeLagAttn`` alias monkeypatched to v2, checks that
``_build_model_kwargs`` forwards every v2 constructor arg present in the config,
that the dead ``encoder`` group is handled (no crash) while ``horizon_refine`` is
mapped, that non-constructor keys (curriculum, loss hparams) are NOT forwarded,
and that the model constructs. See ``vae-teb-lag-attn-v2-spec-and-sprints.md``
Sprint 5.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import yaml  # noqa: E402

import model.vae_teb_prediction.model.trainer_lag_attn_v1 as trainer_mod  # noqa: E402
from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_CONFIG = (
    Path(_REPO_ROOT)
    / "model" / "vae_teb_prediction" / "model" / "config_lag_attn_v2.yaml"
)

# v2-only constructor args that must survive the config -> kwargs round trip.
_V2_KEYS = [
    "target_encoder_blocks", "target_kernel", "target_dilations",
    "source_scales", "d_u", "d_k", "d_e", "active_lags", "active_lags_warmup",
    "kappa_z", "lambda_tv", "lambda_ent", "context_dim", "delta_up_seconds",
    "use_crossphase_bias", "use_outcome_head",
]


def _load_cfg():
    with open(_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_build_model_kwargs_forwards_v2_args(monkeypatch) -> None:
    """Every v2 arg in the config is forwarded; dead groups handled; model builds."""
    cfg = _load_cfg()
    vae_cfg = cfg["model_config"]["VAE_model"]
    # Activate the v2 alias so _build_model_kwargs introspects v2's signature.
    monkeypatch.setattr(trainer_mod, "SeqVaeLagAttn", SeqVaeLagAttnV2)

    fake = SimpleNamespace(config=cfg)
    kwargs = trainer_mod.GraphModelVaeTebLagAttnV1Trainer._build_model_kwargs(fake)

    v2_params = set(inspect.signature(SeqVaeLagAttnV2.__init__).parameters)
    for key in _V2_KEYS:
        assert key in vae_cfg, f"{key} missing from config"
        assert key in v2_params, f"{key} is not a v2 constructor param"
        assert key in kwargs, f"{key} was not forwarded by _build_model_kwargs"

    # Nested horizon_refine mapped to flat kwargs; dead 'encoder' group absent.
    assert kwargs["horizon_depth"] == 3
    assert kwargs["horizon_kernel"] == 3
    assert kwargs["horizon_film"] is True
    assert kwargs.get("encoder_extra_dilations", ()) == ()

    # Non-constructor keys must NOT be forwarded as model kwargs.
    for key in ("curriculum", "kld_beta", "free_bits", "likelihood",
                "lag_smoothness_lambda", "loss_spike_skip"):
        assert key not in kwargs, f"{key} must not be a model constructor kwarg"

    # The model constructs from the forwarded kwargs.
    model = SeqVaeLagAttnV2(**kwargs)
    assert model.M == 4 and model.d_z == 24 and model.active_lags == 8
    assert model.target_dilations == (1, 2, 4, 8, 16, 32)
    assert model.source_scales == (3, 9, 21)
    assert model.use_entmax is True


def test_curriculum_block_present_and_wellformed() -> None:
    """The config carries an enabled 3-stage curriculum that resolves cleanly."""
    cfg = _load_cfg()
    cur = cfg["model_config"]["VAE_model"]["curriculum"]
    assert cur["enabled"] is True
    stages = cur["stages"]
    assert len(stages) == 3
    # The pure resolver accepts the config stages at each boundary.
    s0 = SeqVaeLagAttnV2._resolve_stage(0, stages)
    assert s0["enable_source"] is False and s0["beta"] == 0.0
    s_last = SeqVaeLagAttnV2._resolve_stage(1000, stages)
    assert s_last["enable_source"] is True and abs(s_last["beta"] - 5.0e-2) < 1e-9
