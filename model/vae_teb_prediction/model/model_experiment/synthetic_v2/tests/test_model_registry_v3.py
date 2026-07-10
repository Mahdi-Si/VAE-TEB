r"""S0-T02 / S0-T03: config-driven model-class registry and the ``model.v3`` overlay.

The ``synthetic_v3`` ablation ladder builds three :class:`SeqVaeLagAttnV3` arms while the
committed ``pl_module_v2`` toggle alias stays v1. :func:`pl_module_v2.build_model` must
therefore select the constructor from an optional ``model.class`` key and use the
*resolved* class for **both** the ``inspect.signature`` filter and the construction, so a
v3-only kwarg is never dropped just because the toggle still points at v1 (the single
most load-bearing failure mode in the effort). These tests pin:

* absent ``class`` falls back to the committed alias (v1);
* ``class: SeqVaeLagAttnV3`` + a ``v3`` overlay produces a real v3 with the overlay
  attributes set (``causal_norm``, ``posterior_logvar``, ``logvar_bound``,
  ``kld_support``), proving no v3 key was filtered into ``dropped``;
* the ``v2`` overlay does **not** leak into a v3 build, and *both* overlays are popped and
  ignored under the v1 alias so the resolved kwargs are byte-identical to today's;
* an unknown ``class`` raises ``ValueError`` naming the registry keys.

Companion guards ``test_alias_build_v2.py`` and ``test_rollback_v1.py`` must still pass
unmodified (run them alongside this module).
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

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    pl_module_v2 as plm,
)

_CPU = torch.device("cpu")


def _tiny_v1_model_cfg() -> dict:
    """A minimal, flat v1-compatible ``model`` block (no ``class``, no overlays)."""
    return {
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
        "lstm_layers": 1,
        "logvar_clamp": [-5.0, 3.0],
        "head_structured_latent": True,
    }


def _v2_overlay() -> dict:
    """A ``v2`` overlay whose sentinel ``source_scales`` is v2-only (never a v3 kwarg)."""
    return {"source_scales": [3, 9, 21], "d_u": 96, "active_lags": 4}


def _v3_overlay() -> dict:
    """The production ``v3`` overlay: every value differs from the v3 constructor default."""
    return {
        "causal_norm": True,
        "posterior_logvar": "residual",
        "logvar_bound": "smooth",
        "delta_logvar_scale": 2.0,
        "kld_support": "anchor",
        "lag_bias_init": "alibi_decay",
        "use_entmax": True,
        "freeze_unused_attn_proj": True,
    }


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------
def test_resolve_absent_class_is_alias() -> None:
    """``None`` (absent ``model.class``) resolves to the committed v1 alias."""
    assert plm._resolve_model_class(None) is plm.SeqVaeLagAttn
    assert plm.SeqVaeLagAttn.__name__ == "SeqVaeLagAttnV1"
    # The alias name round-trips through the registry to the same class.
    assert plm._resolve_model_class(plm.SeqVaeLagAttn.__name__) is plm.SeqVaeLagAttn


def test_resolve_v3_and_v2_by_name() -> None:
    """The v3 / v2 names resolve to the real (lazily-imported) classes."""
    v3 = plm._resolve_model_class("SeqVaeLagAttnV3")
    v2 = plm._resolve_model_class("SeqVaeLagAttnV2")
    assert v3.__name__ == "SeqVaeLagAttnV3"
    assert v2.__name__ == "SeqVaeLagAttnV2"


def test_resolve_unknown_raises_naming_keys() -> None:
    """An unknown class name raises ``ValueError`` naming the registry keys."""
    with pytest.raises(ValueError) as excinfo:
        plm._resolve_model_class("NotAModel")
    msg = str(excinfo.value)
    for key in plm._KNOWN_MODEL_CLASSES:
        assert key in msg


# ---------------------------------------------------------------------------
# build_model: v1 default (S0-T02 backward-compat)
# ---------------------------------------------------------------------------
def test_build_model_default_is_v1() -> None:
    """Absent ``class`` builds a v1 instance through the alias (2-tuple preserved)."""
    model, kwargs = plm.build_model(_tiny_v1_model_cfg(), _CPU)
    assert isinstance(model, plm.SeqVaeLagAttn)
    assert type(model).__name__ == "SeqVaeLagAttnV1"
    assert kwargs["logvar_clamp"] == (-5.0, 3.0)


# ---------------------------------------------------------------------------
# build_model: v3 via registry (S0-T02) + v3 overlay (S0-T03)
# ---------------------------------------------------------------------------
def test_build_model_v3_overlay_lands_and_v2_does_not() -> None:
    """``class: SeqVaeLagAttnV3`` + ``v3`` overlay builds a configured v3; ``v2`` is dropped."""
    cfg = {
        **_tiny_v1_model_cfg(),
        "class": "SeqVaeLagAttnV3",
        "v2": _v2_overlay(),
        "v3": _v3_overlay(),
    }
    model, kwargs = plm.build_model(cfg, _CPU)

    # Correct concrete class (v3 subclasses v1, so ``isinstance`` can't discriminate --
    # the exact ``__name__`` is what proves the registry selected v3, not the alias).
    assert type(model).__name__ == "SeqVaeLagAttnV3"

    # v3 overlay keys survived the signature filter (were NOT in ``dropped``) ...
    for key in ("causal_norm", "posterior_logvar", "logvar_bound", "kld_support"):
        assert key in kwargs, f"v3 key {key!r} was dropped"
    # ... and are reflected on the built model (the ablation actually takes effect).
    assert model.causal_norm is True
    assert model.posterior_logvar == "residual"
    assert model.logvar_bound == "smooth"
    assert model.kld_support == "anchor"

    # The v2 overlay's v2-only sentinel never reaches a v3 constructor.
    assert "source_scales" not in kwargs


def test_build_model_v3_defaults_when_no_overlay() -> None:
    """A v3 class without a ``v3`` overlay keeps the v3 constructor defaults."""
    cfg = {**_tiny_v1_model_cfg(), "class": "SeqVaeLagAttnV3"}
    model, _ = plm.build_model(cfg, _CPU)
    assert type(model).__name__ == "SeqVaeLagAttnV3"
    # Defaults per the v3 constructor.
    assert model.causal_norm is False
    assert model.posterior_logvar == "independent"


# ---------------------------------------------------------------------------
# build_model: overlays inert under the v1 alias (S0-T03)
# ---------------------------------------------------------------------------
def test_overlays_popped_and_ignored_under_v1() -> None:
    """Under the v1 alias both overlays are popped; resolved kwargs equal today's."""
    base = _tiny_v1_model_cfg()
    with_overlays = {**base, "v2": _v2_overlay(), "v3": _v3_overlay()}

    _, kwargs_base = plm.build_model(base, _CPU)
    model_ov, kwargs_ov = plm.build_model(with_overlays, _CPU)

    assert isinstance(model_ov, plm.SeqVaeLagAttn)  # still v1
    # v1 accepts neither ``source_scales`` (v2) nor ``posterior_logvar`` (v3), so both
    # overlays are stripped and the resolved kwargs are byte-identical to the no-overlay
    # build.
    assert kwargs_ov == kwargs_base
