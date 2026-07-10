r"""Sprint 0 (S0-T05): v1 golden regression / rollback gate.

Captures a deterministic :class:`SeqVaeLagAttnV1` reference (forward tensors,
``compute_loss`` scalars for ``mse`` and ``gaussian_nll``, a short SGD training
trace, and a plotting-callback fingerprint) into a committed ``.npz`` and asserts
that v1 -- reached **through the canonical alias** -- still reproduces it within
tolerance. This proves the S0-T03 alias refactor and any later shared-file edits
(trainer, plotting, config injection) do not change v1 behaviour; re-run it at the
end of Sprint 6 and Sprint 7 as the rollback gate.

Regenerate the reference (only when a v1 change is intended) with::

    .venv/Scripts/python model/vae_teb_prediction/model/model_experiment/synthetic_v2/tests/test_v1_golden_regression.py --regen

See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` S0-T05.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

# Reference anchor: the v1 class imported DIRECTLY (regeneration path).
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)

# Canonical alias -- comment-toggle mirrors the consumers (v1 active). The test
# runs the capture through this alias; flip it to v2 to exercise the gate.
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1 as SeqVaeLagAttn,
)
# from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (
#     SeqVaeLagAttnV2 as SeqVaeLagAttn,
# )

_REF_PATH = Path(__file__).resolve().parent / "v1_golden_ref.npz"

# Deterministic fixture: production channel counts, tiny geometry, no dropout so
# the only stochasticity is the (seeded) reparameterisation.
_MODEL_SEED = 1234
_FWD_SEED = 4321
_B, _T = 2, 16
_TINY_KWARGS = dict(
    sequence_length=16,
    d_model=32,
    d_z=8,
    horizon=4,
    warmup_period=2,
    c_y=87,
    c_u=101,
    use_up_st=True,
    max_lag=8,
    num_heads=4,
    d_head=8,
    dropout=0.0,
)

# Plotting-callback keys whose distribution fingerprint is checked explicitly.
_CALLBACK_KEYS = ("z", "attn_weights", "te_lag_map", "kld_per_t")
_LOSS_KEYS = (
    "feat_loss",
    "base_loss",
    "kld_loss",
    "total_loss",
    "mean_logvar_full",
    "mean_logvar_base",
    "lag_smoothness",
)


def _make_inputs():
    """Deterministic ``(y_st, y_ph, u_stream, weight)`` fixture."""
    g = torch.Generator().manual_seed(7)
    y_st = torch.randn(_B, _T, 43, generator=g)
    y_ph = torch.randn(_B, _T, 44, generator=g)
    u = torch.randn(_B, _T, 101, generator=g)
    weight = torch.rand(_B, _T, generator=g)
    return y_st, y_ph, u, weight


def _capture(model_cls) -> dict:
    """Run the deterministic forward / loss / trace capture for ``model_cls``."""
    y_st, y_ph, u, weight = _make_inputs()

    torch.manual_seed(_MODEL_SEED)
    model = model_cls(**_TINY_KWARGS)
    model.eval()
    torch.manual_seed(_FWD_SEED)
    with torch.no_grad():
        out = model(y_st, y_ph, u)

    result: dict = {}
    for key, val in out.items():
        if torch.is_tensor(val):
            result[f"fwd__{key}"] = val.detach().cpu().numpy().astype(np.float32)

    # Distribution fingerprint of the callback-relevant keys.
    for key in _CALLBACK_KEYS:
        arr = out[key].detach().cpu().numpy().astype(np.float64)
        result[f"fp__{key}"] = np.asarray(
            [arr.mean(), arr.std(), arr.min(), arr.max()], dtype=np.float64
        )

    for likelihood in ("mse", "gaussian_nll"):
        with torch.no_grad():
            losses = model.compute_loss(
                out, y_st, y_ph, weight=weight, beta=1.0,
                likelihood=likelihood, sigma_obs=1.0,
            )
        for key in _LOSS_KEYS:
            result[f"loss__{likelihood}__{key}"] = np.float64(float(losses[key]))

    # Short deterministic SGD trace from a freshly-seeded model.
    torch.manual_seed(_MODEL_SEED)
    trace_model = model_cls(**_TINY_KWARGS)
    trace_model.train()
    opt = torch.optim.SGD(trace_model.parameters(), lr=1e-2, momentum=0.0)
    trace = []
    for step in range(3):
        torch.manual_seed(_FWD_SEED + 1 + step)
        opt.zero_grad()
        o = trace_model(y_st, y_ph, u)
        losses = trace_model.compute_loss(
            o, y_st, y_ph, weight=weight, beta=1.0, likelihood="mse",
        )
        losses["total_loss"].backward()
        opt.step()
        trace.append(float(losses["total_loss"]))
    result["trace"] = np.asarray(trace, dtype=np.float64)
    return result


def _regenerate() -> None:
    """Capture the reference from the DIRECT v1 class and write the ``.npz``."""
    ref = _capture(SeqVaeLagAttnV1)
    np.savez(_REF_PATH, **ref)
    print(f"[golden] wrote reference with {len(ref)} arrays to {_REF_PATH}")


def _tol_for(key: str):
    """Return ``(atol, rtol)`` for a stored key (looser for deep-conv tensors)."""
    if key.startswith("loss__"):
        return 1e-5, 1e-5
    if key.startswith("fp__"):
        return 1e-4, 1e-4
    return 1e-4, 1e-4  # fwd__* tensors and the training trace


@pytest.mark.skipif(not _REF_PATH.exists(), reason="reference .npz not generated")
def test_v1_forward_loss_trace_matches_reference() -> None:
    """v1 (through the alias) reproduces the stored golden reference."""
    ref = np.load(_REF_PATH)
    got = _capture(SeqVaeLagAttn)

    assert set(ref.files) == set(got), (
        f"key set drift: only-ref={set(ref.files) - set(got)}, "
        f"only-got={set(got) - set(ref.files)}"
    )
    for key in ref.files:
        atol, rtol = _tol_for(key)
        np.testing.assert_allclose(
            np.asarray(got[key]), ref[key], atol=atol, rtol=rtol,
            err_msg=f"golden mismatch on {key!r}",
        )


if __name__ == "__main__":
    if "--regen" in sys.argv:
        _regenerate()
    else:
        print("pass --regen to (re)write the golden reference .npz")
