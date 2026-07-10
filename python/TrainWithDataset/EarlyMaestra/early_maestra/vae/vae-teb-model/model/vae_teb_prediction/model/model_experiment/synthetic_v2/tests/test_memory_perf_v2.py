r"""Sprint 6 (S6-T04): real-scale memory / perf smoke for v2 (GPU-gated).

Runs a forward + backward on a **production-shaped** :class:`SeqVaeLagAttnV2`
(``d_model=128``, ``T=300``, ``M=4``, ``L=91``, ``K_a=8``) at a reduced batch on
the local GPU, records the peak allocation, and extrapolates it to the
``B=128, T=300`` per-GPU A6000 budget (section 7: the full lag-score tensor is
~56 MB and the active values ~157 MB, so the model is deliberately not
memory-heavy). Skips on CPU.

The full ``B=128`` per-GPU peak on the A6000 (48 GB) under DDP is operator-run
and confirmed by the user; this assistant gate is the reduced-batch measurement +
extrapolation. See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` S6-T04.
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

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

# Production widths (config_lag_attn_v2.yaml VAE_model), full T and lag grid.
_PROD_KW: Dict[str, Any] = {
    "sequence_length": 300,
    "d_model": 128,
    "d_z": 24,
    "horizon": 30,
    "warmup_period": 30,
    "c_y": 87,
    "c_u": 101,
    "use_up_st": True,
    "max_lag": 90,
    "num_heads": 4,
    "d_head": 32,
    "dropout": 0.1,
    "decoder_hidden": 128,
    "logvar_clamp": (-5.0, 3.0),
    "mu_scale": 5.0,
    "delta_mu_scale": 3.0,
    "use_entmax": True,
    "horizon_depth": 3,
    "horizon_kernel": 3,
    "horizon_film": True,
    "target_encoder_blocks": 6,
    "target_kernel": 5,
    "target_dilations": (1, 2, 4, 8, 16, 32),
    "source_scales": (3, 9, 21),
    "d_u": 96,
    "d_k": 16,
    "d_e": 32,
    "active_lags": 8,
    "active_lags_warmup": 16,
    "kappa_z": 0.05,
}

_T = 300
# A6000 per-GPU budget for the full B=128 forward+backward (generous headroom
# below the 48 GB card). The reduced-batch measurement is extrapolated to B=128
# and asserted under this bound; the real B=128 peak is operator-measured.
_A6000_BUDGET_GB = 40.0
_TARGET_BATCH = 128


def _measure_peak_gb(batch: int, device: torch.device) -> float:
    r"""Peak forward+backward allocation (GB) DELTA over the pre-model baseline.

    The delta over the baseline at model-construction time isolates this call's
    footprint from any leftover allocations, so a full-suite run reports this
    model's memory rather than the shared caching-allocator high-water mark.
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated(device)

    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(**_PROD_KW).to(device)
    model.train()
    g = torch.Generator(device="cpu").manual_seed(0)
    y_st = torch.randn(batch, _T, 43, generator=g).to(device)
    y_ph = torch.randn(batch, _T, 44, generator=g).to(device)
    u_stream = torch.randn(batch, _T, 101, generator=g).to(device)
    weight = torch.ones(batch, _T, device=device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    out = model(y_st, y_ph, u_stream)
    loss = model.compute_loss(
        forward_outputs=out, y_st=y_st, y_ph=y_ph, weight=weight,
        beta=5.0e-2, lambda_full=1.0, lambda_base=0.5,
        likelihood="gaussian_nll", sigma_obs=1.0,
        detach_baseline_in_full=True, lambda_lag=1.0e-3,
    )["total_loss"]
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    peak = (torch.cuda.max_memory_allocated(device) - baseline_bytes) / (1024 ** 3)
    # Drop references so the next measurement starts from a clean baseline.
    del model, out, loss, y_st, y_ph, u_stream, weight
    return float(peak)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_backward_memory_v2() -> None:
    r"""Two-point linear fit of the peak footprint, extrapolated to B=128.

    A single reduced-batch peak scaled by ``128 / B`` grossly over-estimates,
    because the fixed cost (parameters + optimizer-free autograd buffers + the
    cuDNN workspace) does not scale with the batch. Measuring at two batch sizes
    fits $\mathrm{peak}(B) = c_0 + p\,B$ (fixed cost $c_0$, per-sample cost $p$)
    and extrapolates $\mathrm{peak}(128) = c_0 + 128 p$ -- the honest per-GPU
    estimate. The full $B=128$ peak on the A6000 (48 GB) is operator-run.

    The two measurement points are $B=16$ and $B=32$ rather than a tighter pair
    like $4$ and $8$: dividing a peak-memory difference by a denominator of
    only $4$ amplifies any allocator/fragmentation noise (e.g. from a shared
    pytest process where earlier tests left cached blocks on the caching
    allocator) roughly $8\times$ into the fitted slope, which can push the
    $B=128$ extrapolation over budget on pure measurement noise even though the
    true per-sample cost is well within it. A denominator of $16$ (and a
    shorter $4\times$ extrapolation to $B=128$ instead of $16$--$32\times$)
    keeps the fit stable under full-suite contamination while both points still
    comfortably fit the local 16 GB dev GPU.
    """
    device = torch.device("cuda")
    b_lo, b_hi = 16, 32
    peak_lo = _measure_peak_gb(b_lo, device)
    peak_hi = _measure_peak_gb(b_hi, device)

    per_sample = max((peak_hi - peak_lo) / (b_hi - b_lo), 0.0)
    fixed = peak_hi - per_sample * b_hi
    extrapolated_gb = fixed + per_sample * _TARGET_BATCH

    print(
        f"\n[S6-T04] v2 peak(B={b_lo})={peak_lo:.3f} GB, peak(B={b_hi})="
        f"{peak_hi:.3f} GB -> fixed={fixed:.3f} GB + {per_sample * 1024:.1f} "
        f"MB/sample; fitted B={_TARGET_BATCH}: {extrapolated_gb:.2f} GB "
        f"(A6000 budget {_A6000_BUDGET_GB:.0f} GB, "
        f"margin {_A6000_BUDGET_GB - extrapolated_gb:.2f} GB)."
    )

    # The fitted per-GPU B=128 peak must sit under the A6000 budget with margin.
    assert extrapolated_gb < _A6000_BUDGET_GB, (
        f"fitted B={_TARGET_BATCH} peak {extrapolated_gb:.2f} GB exceeds the "
        f"{_A6000_BUDGET_GB:.0f} GB A6000 budget"
    )
