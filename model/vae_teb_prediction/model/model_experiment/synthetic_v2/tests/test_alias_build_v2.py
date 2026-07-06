r"""Sprint 0 (S0-T03): canonical ``SeqVaeLagAttn`` alias seam.

Every consumer refactored in S0-T03 must reference the canonical alias
``SeqVaeLagAttn`` (currently resolving to :class:`SeqVaeLagAttnV1`) for construction,
type hints, and ``inspect.signature`` introspection, so a one-line comment toggle
swaps v1 and v2. These tests assert the alias resolves to v1 in each consumer, that
``inspect.signature(SeqVaeLagAttn.__init__)`` is byte-identical to v1's (so the
config builders forward the same kwargs), and that a real config builder
(:func:`pl_module_v2.build_model`) constructs through the alias. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` S0-T03.
"""

from __future__ import annotations

import importlib
import inspect
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

# Consumers refactored to the alias in S0-T03 (testing/base.py is deferred to S6).
_CONSUMER_MODULES = [
    "model.vae_teb_prediction.model.trainer_lag_attn_v1",
    "model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2",
    "model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal",
    "model.vae_teb_prediction.model.model_experiment.synthetic.train_ddp",
    "model.vae_teb_prediction.model.model_experiment.synthetic.evaluate_te",
    "model.vae_teb_prediction.model.model_experiment.synthetic.lag_recovery",
    "model.vae_teb_prediction.new_classifier.guid_cls_v1.precompute_latents",
]


@pytest.mark.parametrize("module_name", _CONSUMER_MODULES)
def test_alias_resolves_to_v1(module_name) -> None:
    """Each consumer exposes ``SeqVaeLagAttn`` bound to :class:`SeqVaeLagAttnV1`."""
    mod = importlib.import_module(module_name)
    alias = getattr(mod, "SeqVaeLagAttn", None)
    assert alias is SeqVaeLagAttnV1, (
        f"{module_name}.SeqVaeLagAttn should be the v1 class while v1 is active"
    )


@pytest.mark.parametrize("module_name", _CONSUMER_MODULES)
def test_no_bare_v1_name_outside_toggle_block(module_name) -> None:
    """The literal ``SeqVaeLagAttnV1`` appears only inside the alias toggle block."""
    mod = importlib.import_module(module_name)
    src = Path(mod.__file__).read_text(encoding="utf-8")
    bad = [
        ln
        for ln in src.splitlines()
        if "SeqVaeLagAttnV1" in ln and " as SeqVaeLagAttn" not in ln
    ]
    assert not bad, (
        f"{module_name} still names SeqVaeLagAttnV1 outside the toggle block:\n"
        + "\n".join(bad)
    )


def test_alias_signature_matches_v1() -> None:
    """``inspect.signature(SeqVaeLagAttn.__init__)`` is identical to v1's."""
    trainer = importlib.import_module(
        "model.vae_teb_prediction.model.trainer_lag_attn_v1"
    )
    alias_params = inspect.signature(trainer.SeqVaeLagAttn.__init__).parameters
    v1_params = inspect.signature(SeqVaeLagAttnV1.__init__).parameters
    assert list(alias_params) == list(v1_params)
    for name in v1_params:
        assert alias_params[name].default == v1_params[name].default


def test_build_model_constructs_via_alias() -> None:
    """``pl_module_v2.build_model`` builds a v1 instance through the alias."""
    plm = importlib.import_module(
        "model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2"
    )
    model_cfg = {
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
    model, kwargs = plm.build_model(model_cfg, torch.device("cpu"))
    assert isinstance(model, SeqVaeLagAttnV1)
    # ``model_kwargs`` (stored verbatim in checkpoints) echoes the config, with
    # ``logvar_clamp`` coerced to a tuple.
    assert kwargs["d_model"] == 16 and kwargs["num_heads"] == 4
    assert kwargs["logvar_clamp"] == (-5.0, 3.0)
