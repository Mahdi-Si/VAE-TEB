r"""The structural invariants at the production budget, and the one init policy the width change reaches.

Source purity, no decoder bypass, one shared decoder and the exact zero-KL start are properties of
the forward, which is the conv-Transformer parent's own code object here and is pinned, with its
negative controls, by ``lag_attn_transformer_rws/tests`` (``test_source_purity.py``,
``test_no_bypass.py``, ``test_zero_kl_init.py``, ``test_init_policies.py``). This file runs the
composition once at the real geometry -- the shipped gate gathers $C_{\mathrm{keep}}$ of $c_y$ target
channels and delays every survivor -- and measures ``head_init_calibration`` at a decoder width the
raw parent never produces.

``_TARGET_ONLY_KEYS`` and ``_closed_form_kl`` are imported from that suite, not copied: an invariant
whose definition lived in two places could hold in one and drift in the other.
"""
from __future__ import annotations

import math

import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import BATCH, shipped_gated_kwargs
from teb_vae.lag_attn_transformer_rws.tests.test_source_purity import _TARGET_ONLY_KEYS
from teb_vae.lag_attn_transformer_rws.tests.test_zero_kl_init import _closed_form_kl


def test_the_invariants_hold_at_the_production_geometry_and_budget():
    """Source purity and the exact zero-KL start, through the assembled model at the shipped budget:
    resampling the source moves the source state and nothing target-only."""
    kwargs = shipped_gated_kwargs()
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**dict(kwargs, dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(0)
    length = int(kwargs["sequence_length"])
    split = SeqVaeLagAttnTrfFs.TARGET_BLOCK_SPLIT
    y_st = torch.randn(BATCH, length, split, generator=generator)
    y_ph = torch.randn(BATCH, length, int(kwargs["c_y"]) - split, generator=generator)
    u_stream = torch.randn(BATCH, length, int(kwargs["c_u"]), generator=generator)

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    torch.manual_seed(0)
    with torch.no_grad():
        resampled = model(y_st, y_ph, torch.randn(u_stream.shape, generator=generator))

    for key in _TARGET_ONLY_KEYS:
        assert torch.equal(reference[key], resampled[key]), key
    assert not torch.equal(reference["source_state"], resampled["source_state"])
    assert float(_closed_form_kl(reference).abs().max()) == 0.0
    assert reference["mu_base"].shape[-1] == len(kwargs["target_keep_index"])


def test_the_calibration_reaches_the_gate_wide_head(tiny_gated):
    r"""The decoder is built from the width hook *before* the calibration pass, so the log-variance
    bias must sit at $\log(5/3)$ -- the exact pre-image of log-variance $0$ under
    ``smooth_bound(-5, 3)`` -- on **every** channel the gate kept. A head built after the pass, or at
    another width, would leave an uncalibrated tail whose initial NLL sits far above the trivial
    predictor's."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnTrfFs(**dict(tiny_gated, head_init_calibration=True))
    bias = model.decoder.logvar_head.bias
    lo, hi = model.logvar_clamp

    assert bias.numel() == len(tiny_gated["target_keep_index"]) != model.raw_per_step
    assert torch.allclose(bias, torch.full_like(bias, math.log(5.0 / 3.0)))
    assert torch.allclose(smooth_bound(bias, lo, hi), torch.zeros_like(bias), atol=1e-6)
