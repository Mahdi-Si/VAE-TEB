r"""S1-T01: numerical parity ``v3(parity) == v1`` on a real cached batch.

The ``parity`` arm is ``SeqVaeLagAttnV3`` under v1's latent machinery (``causal_norm=False``,
``posterior_logvar='independent'``, ``logvar_bound='clamp'``, ``lag_bias_init='normal'``,
``use_entmax=False``). It is the ablation ladder's baseline, so its equivalence to a literal
v1 must be *proved*, not asserted. This copies a v1 ``state_dict`` into a ``v3(parity)`` (the
parameter keys are byte-identical -- the three v3 head subclasses add only string attributes)
and asserts, on a real cached scattering batch under ``likelihood='mse'``, ``sigma_obs=1.0``,
``free_bits=0``:

* all 23 shared forward tensors agree to ``< 1e-5`` (with the reparameterisation RNG seeded
  identically before each forward, so the sampled ``z`` is comparable);
* every ``compute_loss`` term (``feat_loss`` / ``base_loss`` / ``kld_loss`` / ``total_loss``)
  agrees to ``< 1e-5``.

The default path is ``slow`` and **fails loudly** when no cache exists. A ``-k fallback``
variant runs on random tensors WITHOUT the cache; it proves code-path parity only, NOT parity
on real scattering features.
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

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.dataset_v2 import (  # noqa: E402,E501
    build_u_stream,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    build_model,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E402,E501
    resolve_arm,
)

_TOL = 1e-5
_SHARED_FLOAT_KEYS = [
    "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z", "target_state",
    "source_state", "decoder_state", "attended_source", "attended_source_heads",
    "attn_weights", "mu_base", "logvar_base", "delta_mu_src", "mu_full",
    "logvar_full", "kld_per_t", "kld_per_t_per_head", "te_lag_map",
    "mu_prior_sat_frac", "delta_mu_sat_frac",
]
_LOSS_TERMS = ["feat_loss", "base_loss", "kld_loss", "total_loss"]


def _v1_kwargs(model_cfg: dict) -> dict:
    return {k: v for k, v in model_cfg.items() if k not in ("class", "v2", "v3")}


def _build_parity_pair(model_cfg: dict):
    """Return ``(v1, v3_parity)`` sharing one weight set (v1's state_dict copied into v3)."""
    torch.manual_seed(0)
    v1, _ = build_model(_v1_kwargs(model_cfg), torch.device("cpu"))
    parity_cfg = resolve_arm({"model": model_cfg,
                              "arms": _PARITY_ARMS}, "parity")["model"]
    v3, _ = build_model(parity_cfg, torch.device("cpu"))
    # The parameter keys are identical; a strict copy makes the two models bit-equivalent.
    v3.load_state_dict(v1.state_dict(), strict=True)
    return v1.eval(), v3.eval()


# The parity arm deltas (as in config_synth_v3.yaml), inlined so the fallback path needs no
# config file.
_PARITY_ARMS = {
    "parity": {"model": {"v3": {"causal_norm": False, "posterior_logvar": "independent",
                                "logvar_bound": "clamp", "kld_support": "full",
                                "lag_bias_init": "normal", "use_entmax": False}}}
}


def _assert_parity(v1, v3, y_st, y_ph, u, weight) -> None:
    # Seed identically before each forward so the reparameterised z is comparable.
    torch.manual_seed(123)
    o1 = v1(y_st, y_ph, u)
    torch.manual_seed(123)
    o3 = v3(y_st, y_ph, u)

    for key in _SHARED_FLOAT_KEYS:
        assert key in o1 and key in o3, f"missing shared key {key}"
        diff = float((o1[key] - o3[key]).abs().max())
        assert diff < _TOL, f"forward tensor {key} differs by {diff:.3e}"

    kw = dict(compute_kld_loss=True, beta=1.0, lambda_full=1.0, lambda_base=0.5,
              likelihood="mse", sigma_obs=1.0, free_bits=0.0,
              detach_baseline_in_full=False)
    loss1 = v1.compute_loss(o1, y_st, y_ph, weight=weight, **kw)
    loss3 = v3.compute_loss(o3, y_st, y_ph, weight=weight, **kw)
    for term in _LOSS_TERMS:
        assert term in loss1 and term in loss3, f"missing loss term {term}"
        diff = float((loss1[term] - loss3[term]).abs().max())
        assert diff < _TOL, f"loss term {term} differs by {diff:.3e}"


@pytest.mark.slow
def test_v3_parity_equals_v1_on_cache(cache_batch) -> None:
    """v3(parity) reproduces v1 to <1e-5 on real scattering features."""
    assert cache_batch is not None, (
        "no cache found; build the shared cache (S0-T00) before this parity gate, or run "
        "the weaker `-k fallback` variant."
    )
    cfg = cache_batch["config"]
    batch = cache_batch["batch"]
    y_st, y_ph, u = batch.fhr_st, batch.fhr_ph, build_u_stream(batch)
    weight = getattr(batch, "weight", None)
    v1, v3 = _build_parity_pair(cfg["model"])
    _assert_parity(v1, v3, y_st, y_ph, u, weight)


def test_v3_parity_equals_v1_fallback() -> None:
    """Code-path parity on RANDOM tensors -- proves NOT parity on real scattering features.

    Runs without the cache (``-k fallback``) so the parity code path is exercised on a fresh
    checkout; it is a strictly weaker claim than :func:`test_v3_parity_equals_v1_on_cache`.
    """
    model_cfg = {
        "sequence_length": 48, "d_model": 32, "d_z": 8, "horizon": 6,
        "warmup_period": 4, "c_y": 87, "c_u": 101, "use_up_st": True,
        "max_lag": 12, "num_heads": 4, "d_head": 8, "lstm_layers": 1,
        "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
    }
    torch.manual_seed(1)
    B, T = 2, model_cfg["sequence_length"]
    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u = torch.randn(B, T, 101)
    weight = torch.ones(B, T)
    v1, v3 = _build_parity_pair(model_cfg)
    _assert_parity(v1, v3, y_st, y_ph, u, weight)
