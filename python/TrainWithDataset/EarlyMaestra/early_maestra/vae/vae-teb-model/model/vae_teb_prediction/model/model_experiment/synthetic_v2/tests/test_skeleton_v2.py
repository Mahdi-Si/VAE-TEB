r"""Sprint 0 (S0-T01): ``SeqVaeLagAttnV2`` skeleton contract.

Asserts the v2 constructor is a strict superset of the v1 constructor (every v1
keyword-only parameter name AND default is present in v2 with the same default),
so ``inspect.signature`` auto-forwarding and the ``model_kwargs`` round-trip cannot
silently drift, and that v2 constructs from a full v1 kwarg dict. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 0.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

# Force the repo root ahead of the sibling ``model/vae_teb_prediction`` on
# ``sys.path`` (which lacks ``utils.custom_logger``) so the model imports resolve
# under pytest -- the same guard ``pl_module_v2`` applies at import time.
_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_EMPTY = inspect.Parameter.empty


def _keyword_params(cls) -> dict:
    """Map ``{name: default}`` for every non-``self`` ``__init__`` parameter."""
    params = inspect.signature(cls.__init__).parameters
    return {
        name: p.default
        for name, p in params.items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }


def test_signature_is_v1_superset() -> None:
    """Every v1 ``__init__`` param name + default is present in v2 unchanged."""
    v1 = _keyword_params(SeqVaeLagAttnV1)
    v2 = _keyword_params(SeqVaeLagAttnV2)

    missing = [name for name in v1 if name not in v2]
    assert not missing, f"v2 is missing v1 constructor params: {missing}"

    mismatched = {
        name: (v1[name], v2[name])
        for name in v1
        if v2[name] != v1[name]
    }
    assert not mismatched, (
        f"v2 changed the default of shared v1 params (v1, v2): {mismatched}"
    )


def test_v2_adds_only_keyword_only_params() -> None:
    """v2 introduces only new keyword-only params (no new positionals)."""
    v1_names = set(_keyword_params(SeqVaeLagAttnV1))
    v2_params = inspect.signature(SeqVaeLagAttnV2.__init__).parameters
    new_names = [n for n in v2_params if n not in v1_names and n != "self"]
    assert new_names, "expected v2 to declare new parameters"
    for name in new_names:
        assert v2_params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"new v2 param {name!r} must be keyword-only"
        )
        assert v2_params[name].default is not _EMPTY, (
            f"new v2 param {name!r} must have a default"
        )


def test_construct_from_v1_kwargs() -> None:
    """v2 constructs from a full explicit v1 kwarg dict without error."""
    v1_kwargs = {
        name: default
        for name, default in _keyword_params(SeqVaeLagAttnV1).items()
        if default is not _EMPTY
    }
    model = SeqVaeLagAttnV2(**v1_kwargs)
    assert isinstance(model, torch.nn.Module)
    # Derived v2 attributes and buffers are wired.
    assert model.M == model.num_heads
    assert model.d_z_m == model.d_z // model.num_heads
    assert model.d_v == model.d_head
    assert model.L == model.max_lag + 1
    assert model.mu_post_running_mean.shape == (model.d_z,)
    assert model.mu_post_running_var.shape == (model.d_z,)
    assert int(model.mu_post_running_count.item()) == 0


def test_fallback_use_up_st_false() -> None:
    """The ``use_up_st=False`` ablation resolves ``c_u=58`` consistently."""
    model = SeqVaeLagAttnV2(use_up_st=False, c_u=58)
    assert model.c_u == 58


def test_inconsistent_c_u_raises() -> None:
    """An inconsistent ``(c_u, use_up_st)`` pair raises ``ValueError`` (v1 parity)."""
    with pytest.raises(ValueError):
        SeqVaeLagAttnV2(use_up_st=True, c_u=58)


def test_scale_guards() -> None:
    """Non-positive mu_scale / delta_mu_scale fail fast at construction (v1 parity)."""
    with pytest.raises(ValueError):
        SeqVaeLagAttnV2(mu_scale=0.0)
    with pytest.raises(ValueError):
        SeqVaeLagAttnV2(delta_mu_scale=0.0)


def test_sprint7_flags_now_wired() -> None:
    """The Sprint 7 optional features are wired (S7-T01/T02): enabling them builds
    the corresponding sub-module instead of raising ``NotImplementedError``."""
    m_cross = SeqVaeLagAttnV2(use_crossphase_bias=True)
    assert m_cross.crossphase_bias is not None
    m_out = SeqVaeLagAttnV2(use_outcome_head=True)
    assert m_out.outcome_head is not None
    # Default off: neither sub-module is constructed.
    m_off = SeqVaeLagAttnV2()
    assert m_off.crossphase_bias is None and m_off.outcome_head is None


def test_source_forward_runs() -> None:
    """The full source forward path (``enable_source=True``) landed in Sprint 3.

    The exhaustive contract is asserted in ``test_forward_contract_v2``; this is a
    minimal smoke that the default (source-enabled) forward now runs and returns a
    dict (it used to raise ``NotImplementedError`` in Sprints 0-2).
    """
    model = SeqVaeLagAttnV2().eval()  # enable_source=True by default
    y_st = torch.randn(1, 40, 43)
    y_ph = torch.randn(1, 40, 44)
    u = torch.randn(1, 40, 101)
    out = model(y_st, y_ph, u)
    assert isinstance(out, dict)
    assert out["mu_full"].shape == (1, 40, 30, 87)
