r"""Sprint 1 (S1-T04a/T04b): baseline-only forward + base-loss branch.

With ``enable_source=False`` the v2 forward must emit EXACTLY the v1 key set (checked
against a live :class:`SeqVaeLagAttnV1` forward) with v1 shapes, ``mu_full ==
mu_base``, and the source-dependent keys zero-filled. The ``compute_loss`` base term
must be finite for both likelihoods and reduce ``base_loss`` over a short optimisation
loop. See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 1.
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
    SeqVaeLagAttnV2,
)

_B, _T = 2, 16
_KWARGS = dict(
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


def _inputs():
    g = torch.Generator().manual_seed(0)
    y_st = torch.randn(_B, _T, 43, generator=g)
    y_ph = torch.randn(_B, _T, 44, generator=g)
    u = torch.randn(_B, _T, 101, generator=g)
    return y_st, y_ph, u


def test_keys_shapes() -> None:
    """Baseline-only forward emits the exact v1 key set with v1 shapes."""
    y_st, y_ph, u = _inputs()

    torch.manual_seed(1)
    v1 = SeqVaeLagAttnV1(**_KWARGS).eval()
    with torch.no_grad():
        out1 = v1(y_st, y_ph, u)

    torch.manual_seed(1)
    v2 = SeqVaeLagAttnV2(enable_source=False, **_KWARGS).eval()
    with torch.no_grad():
        out2 = v2(y_st, y_ph, u)

    assert set(out2) == set(out1), (
        f"key drift: only-v1={set(out1) - set(out2)}, only-v2={set(out2) - set(out1)}"
    )
    for key in out1:
        if torch.is_tensor(out1[key]):
            assert torch.is_tensor(out2[key]), f"{key} should be a tensor"
            assert out2[key].shape == out1[key].shape, (
                f"{key}: v2 {tuple(out2[key].shape)} != v1 {tuple(out1[key].shape)}"
            )
        else:
            assert out2[key] is None, f"{key} should be None (raw_future_pred)"

    # Warm-start / baseline contract.
    assert out2["mu_full"].shape[-1] == 87
    assert torch.equal(out2["mu_full"], out2["mu_base"])
    assert torch.equal(out2["mu_post"], out2["mu_prior"])
    assert float(out2["delta_mu_src"].abs().max()) == 0.0
    assert float(out2["kld_per_t"].abs().max()) == 0.0

    # Attributes external consumers read.
    assert v2.use_up_st is True
    assert v2.warmup_period == 2
    assert v2.horizon == 4
    assert v2._warmup_steps(_T) == 2
    assert v2._build_warmup_valid_mask(_T).shape == (_T,)


def test_running_stats_update_in_train_mode() -> None:
    """A training-mode baseline forward updates the latent running-stats buffers."""
    y_st, y_ph, u = _inputs()
    model = SeqVaeLagAttnV2(enable_source=False, **_KWARGS).train()
    assert int(model.mu_post_running_count.item()) == 0
    model(y_st, y_ph, u)
    assert int(model.mu_post_running_count.item()) > 0


@pytest.mark.parametrize("likelihood", ["mse", "gaussian_nll"])
def test_base_loss_finite(likelihood) -> None:
    """The ``base_loss`` branch is finite for both likelihoods."""
    y_st, y_ph, u = _inputs()
    model = SeqVaeLagAttnV2(enable_source=False, **_KWARGS).train()
    out = model(y_st, y_ph, u)
    losses = model.compute_loss(
        out, y_st, y_ph, beta=0.1, likelihood=likelihood, sigma_obs=1.0,
    )
    for key in ("feat_loss", "base_loss", "kld_loss", "total_loss"):
        assert torch.isfinite(losses[key]), f"{key} not finite"


def test_base_loss_decreases() -> None:
    """A 50-step optimisation loop reduces ``base_loss`` in trend."""
    y_st, y_ph, u = _inputs()
    weight = torch.rand(_B, _T, generator=torch.Generator().manual_seed(3))
    model = SeqVaeLagAttnV2(enable_source=False, **_KWARGS).train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first, last = None, None
    for step in range(50):
        opt.zero_grad()
        out = model(y_st, y_ph, u)
        losses = model.compute_loss(
            out, y_st, y_ph, weight=weight, beta=0.0,
            lambda_full=0.0, lambda_base=1.0, likelihood="mse",
        )
        losses["total_loss"].backward()
        opt.step()
        if step == 0:
            first = float(losses["base_loss"])
        last = float(losses["base_loss"])
    assert last < first, f"base_loss did not decrease: {first:.4f} -> {last:.4f}"
