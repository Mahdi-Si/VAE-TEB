r"""S1-T03: zero-KL-at-init on real cached data.

The whole v3 reading of $\bar K$ as a TE surrogate rests on $K \equiv 0$ at initialisation:
the residual posterior starts equal to the (causal) prior, so the calibration intercept no
longer absorbs a random log-variance head mismatch. On a real cached scattering batch this
asserts, at initialisation:

* ``max|kld_per_t| < 1e-6`` for ``v3_prod`` and ``v3_noncausal`` (residual posterior);
* ``max|kld_per_t| > 1e-3`` for ``parity`` (its INDEPENDENT log-variance head is random --
  that non-zero floor is exactly the point of the baseline arm);
* ``delta_mu_src.abs().max() < 1e-6`` for all three (the source residual is off at init).

``head_structured_latent: true`` throughout (from the config). Marked ``slow``.
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

pytestmark = pytest.mark.slow


def _forward_at_init(cfg, arm, y_st, y_ph, u):
    torch.manual_seed(0)
    model, _ = build_model(resolve_arm(cfg, arm)["model"], torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        return model(y_st, y_ph, u)


def _inputs(cache_batch):
    assert cache_batch is not None, (
        "no cache found; build the shared cache (S0-T00) before running the Sprint 1 "
        "cached-data guards."
    )
    b = cache_batch["batch"]
    return b.fhr_st, b.fhr_ph, build_u_stream(b)


def test_v3_arms_zero_kl_at_init(cache_batch) -> None:
    """The residual-posterior v3 arms have K ~ 0 at init on real scattering features."""
    y_st, y_ph, u = _inputs(cache_batch)
    cfg = cache_batch["config"]
    for arm in ("v3_prod", "v3_noncausal"):
        out = _forward_at_init(cfg, arm, y_st, y_ph, u)
        max_kld = float(out["kld_per_t"].abs().max())
        assert max_kld < 1e-6, f"[{arm}] max|kld_per_t|={max_kld:.3e} (expected < 1e-6)"


def test_parity_arm_nonzero_kl_at_init(cache_batch) -> None:
    """The independent-logvar parity arm has a non-zero KL floor at init (the baseline point)."""
    y_st, y_ph, u = _inputs(cache_batch)
    cfg = cache_batch["config"]
    out = _forward_at_init(cfg, "parity", y_st, y_ph, u)
    max_kld = float(out["kld_per_t"].abs().max())
    assert max_kld > 1e-3, f"[parity] max|kld_per_t|={max_kld:.3e} (expected > 1e-3)"


def test_all_arms_zero_delta_mu_src_at_init(cache_batch) -> None:
    """The source residual is off at init for every arm (warm-start invariant)."""
    y_st, y_ph, u = _inputs(cache_batch)
    cfg = cache_batch["config"]
    for arm in ("parity", "v3_noncausal", "v3_prod"):
        out = _forward_at_init(cfg, arm, y_st, y_ph, u)
        max_delta = float(out["delta_mu_src"].abs().max())
        assert max_delta < 1e-6, f"[{arm}] delta_mu_src max={max_delta:.3e} (expected < 1e-6)"
