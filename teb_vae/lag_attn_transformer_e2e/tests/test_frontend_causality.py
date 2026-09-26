r"""The front end at production geometry: token $t$ reads raw sample $r(t+1) - 1$ and nothing later.

The assembled model's causality is measured at the tiny geometry in ``test_causality.py``. What that
cannot see is the production kernel schedule, which is the front end's own default and reaches far
further back; a stack can be causal at one width and not at another only through an arithmetic
error, which is what the probe here exists to catch.

The probe is :func:`assert_raw_causal`: bitwise in float64 rather than thresholded, and paired with
movement of the last token so a dead stack cannot pass. Its negative control is a future-reading
front end, built here and never in production code: a symmetrically padded FIR --
``padding=(k-1)//2``, what ``nn.Conv1d`` does by default -- which changes no shape, parameter count
or reach, so nothing but a causality probe could find it. A centred *offset* would not be a valid
control: it makes the token depend on raw samples $\le rt$, which is strictly more conservative.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from teb_vae.lag_attn_transformer_e2e.nets import frontend as frontend_module
from teb_vae.lag_attn_transformer_e2e.nets.frontend import CausalAntiAliasDecimate
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    assert_raw_causal,
    build_frontend,
    make_stub_batch,
)


class _SymmetricallyPaddedDecimate(CausalAntiAliasDecimate):
    """The planted defect: the anti-alias filter padded on both sides instead of only on the left.

    A subclass of the real thing with one method replaced, so it inherits the same taps, stride and
    reach -- and therefore passes the reach guard, exactly as the real accident would.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter with the padding split evenly, then subsample at the same right offset."""
        left = (self.taps - 1) // 2
        padded = F.pad(x, (left, (self.taps - 1) - left))
        filtered = F.conv1d(padded, self.fir.to(dtype=x.dtype), groups=self.channels)
        return filtered[..., self.stride - 1 :: self.stride]


def test_the_production_geometry_is_causal():
    """The production kernels at the production sequence length, cut at an interior token."""
    steps = int(SHIPPED_KWARGS["sequence_length"])
    stride = int(SHIPPED_KWARGS["raw_per_step"])
    token = (2 * steps) // 3
    net = build_frontend(SHIPPED_KWARGS).double()
    raw = torch.randn(1, steps * stride, dtype=torch.float64)
    weight = torch.ones(1, steps, dtype=torch.float64)

    with torch.no_grad():
        assert_raw_causal(
            lambda value: net(value, weight),
            raw,
            stride * (token + 1) - 1,
            stride,
            label=f"production front end @ t={token}",
        )


def test_the_probe_rejects_a_symmetrically_padded_front_end(monkeypatch):
    """Split the anti-alias padding evenly -- the single most likely real edit -- and the probe must
    reject it for reading its own future."""
    batch = make_stub_batch(BATCH, SEQ_LEN)
    raw, weight = batch.fhr.double(), batch.weight.double()
    monkeypatch.setattr(frontend_module, "CausalAntiAliasDecimate", _SymmetricallyPaddedDecimate)
    leaking = build_frontend(TINY_KWARGS).double()
    token = int(TINY_KWARGS["warmup_period"])

    with torch.no_grad():
        with pytest.raises(AssertionError, match="reads its own future"):
            assert_raw_causal(
                lambda value: leaking(value, weight),
                raw,
                leaking.total_stride * (token + 1) - 1,
                leaking.total_stride,
                label="symmetrically padded",
            )
