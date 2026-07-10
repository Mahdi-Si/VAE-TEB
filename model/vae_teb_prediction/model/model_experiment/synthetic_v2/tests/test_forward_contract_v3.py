r"""S1-T02: SeqVaeLagAttnV3 forward-contract guard on real cached data.

On a real cached scattering batch and for all three arms, asserts:

* the forward emits exactly **25** keys -- v1's 23 plus the two v3-only additive keys
  ``kld_active_frac`` and ``raw_logvar_prior`` -- cross-checked against a live v1 forward
  so the "23" is measured, not hard-coded;
* the hard-required testing-collector subset (``testing/TESTING_PIPELINE_ARCHITECTURE.md``)
  is present (the doc's "21-key contract" is that consumed subset, not the full dict);
* ``te_lag_map.sum(dim=-1) == kld_per_t`` (what lets ``eval_v2`` run unchanged on v3);
* ``kld_per_t`` is finite everywhere -- including the warm-up prefix and the final-``H``
  tail -- unlike ``measure_transfer_entropy``, which NaN-masks out-of-support steps.

Marked ``slow`` (loads the real cache); fails loudly if no cache has been built.
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
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)

pytestmark = pytest.mark.slow

_ARMS = ["parity", "v3_noncausal", "v3_prod"]
_V3_ADDITIVE_KEYS = {"kld_active_frac", "raw_logvar_prior"}

# The testing collectors consume this subset (TESTING_PIPELINE_ARCHITECTURE.md:25-32); it
# must be present in every arm's forward dict for the shared testing/ pipeline to bridge.
_TESTING_SUBSET = {
    "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z",
    "target_state", "source_state", "decoder_state", "attended_source",
    "attn_weights", "mu_base", "mu_full", "delta_mu_src",
    "kld_per_t", "te_lag_map", "warmup_mask", "mu_prior_sat_frac",
    "delta_mu_sat_frac",
}


def _inputs(cache_batch):
    assert cache_batch is not None, (
        "no cache found; build the shared cache (S0-T00) before running the Sprint 1 "
        "cached-data guards."
    )
    batch = cache_batch["batch"]
    return batch.fhr_st, batch.fhr_ph, build_u_stream(batch)


def test_v1_forward_has_23_keys(cache_batch) -> None:
    """Measure v1's key count on the real batch (the baseline the '25' is 23 + 2 against)."""
    y_st, y_ph, u = _inputs(cache_batch)
    v1 = SeqVaeLagAttnV1().eval()
    with torch.no_grad():
        out = v1(y_st, y_ph, u)
    assert len(out) == 23, f"v1 forward emitted {len(out)} keys: {sorted(out)}"


@pytest.mark.parametrize("arm", _ARMS)
def test_v3_forward_contract(arm, cache_batch) -> None:
    y_st, y_ph, u = _inputs(cache_batch)
    cfg = cache_batch["config"]

    v1 = SeqVaeLagAttnV1().eval()
    model, _ = build_model(resolve_arm(cfg, arm)["model"], torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        v1_out = v1(y_st, y_ph, u)
        out = model(y_st, y_ph, u)

    # 25 keys = v1's 23 + the two v3 additive keys, and v3 is a strict superset of v1.
    assert len(out) == 25, f"[{arm}] forward emitted {len(out)} keys: {sorted(out)}"
    assert set(out) == set(v1_out) | _V3_ADDITIVE_KEYS
    assert set(v1_out) <= set(out)  # no v1 key dropped (additive contract)

    # The testing-collector subset is present.
    missing = _TESTING_SUBSET - set(out)
    assert not missing, f"[{arm}] missing testing-subset keys: {missing}"

    # Shapes on the load-bearing diagnostics.
    B, T = y_st.shape[0], y_st.shape[1]
    assert out["kld_per_t"].shape == (B, T)
    assert out["te_lag_map"].shape[:2] == (B, T)
    assert out["raw_logvar_prior"].shape[-1] == out["mu_post"].shape[-1]  # d_z
    assert out["kld_active_frac"].numel() == 1  # scalar

    # te_lag_map summed over lags equals kld_per_t (lets eval_v2 run unchanged on v3).
    assert torch.allclose(out["te_lag_map"].sum(dim=-1), out["kld_per_t"], atol=1e-5)

    # kld_per_t finite everywhere -- warm-up prefix AND the final-H tail.
    kp = out["kld_per_t"]
    assert torch.isfinite(kp).all(), f"[{arm}] kld_per_t has non-finite entries"
    warmup = int(model.warmup_period)
    horizon = int(model.horizon)
    assert torch.isfinite(kp[:, :warmup]).all()      # warm-up prefix
    assert torch.isfinite(kp[:, T - horizon:]).all()  # final-H tail
