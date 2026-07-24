r"""The target history state must not depend on the target's future.

The prior $p(z_t \mid Y_{\le t})$ conditions on the past *by construction* only if the encoder
behind it is strictly causal. ``nn.GroupNorm`` on a ``(B, C, T)`` tensor pools its statistics
across time, so without ``causal_norm`` the state at $t$ carries a low-bandwidth image of
$Y_{>t}$ -- invisible in any loss curve, corrupting exactly the quantity the model exists to
measure. The probe perturbs the target stream strictly after $t_0$ and measures the *relative*
movement of ``target_state``$[t_0]$, through the model's own forward.

The perturbation is **random**, not a constant offset: the encoder opens with a per-timestep
``LayerNorm``, which removes a uniform channel shift, so a constant probe would report a false
pass on a leaky model.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

_T0 = 8
_LEAK_TOL = 1e-5   # float32 round-off on O(1) activations
_LEAK_FLOOR = 1e-3  # what a genuinely leaky normaliser measurably exceeds


def _relative_leak(tiny_kwargs, causal_norm: bool) -> float:
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**dict(tiny_kwargs, causal_norm=causal_norm)).eval()

    generator = torch.Generator().manual_seed(1)
    y_st = torch.randn(2, model.sequence_length, 43, generator=generator)
    y_ph = torch.randn(2, model.sequence_length, 66, generator=generator)
    u_stream = torch.randn(2, model.sequence_length, model.c_u, generator=generator)

    # Resample the strict future of both target blocks; the past and the source are untouched.
    resample = torch.Generator().manual_seed(11)
    y_st_pert = y_st.clone()
    y_ph_pert = y_ph.clone()
    y_st_pert[:, _T0 + 1 :] = torch.randn(y_st[:, _T0 + 1 :].shape, generator=resample)
    y_ph_pert[:, _T0 + 1 :] = torch.randn(y_ph[:, _T0 + 1 :].shape, generator=resample)

    torch.manual_seed(0)
    with torch.no_grad():
        base = model(y_st, y_ph, u_stream)["target_state"]
    torch.manual_seed(0)
    with torch.no_grad():
        perturbed = model(y_st_pert, y_ph_pert, u_stream)["target_state"]

    movement = (base[:, _T0] - perturbed[:, _T0]).abs().max().item()
    scale = base[:, _T0].abs().max().item()
    return movement / scale


def test_with_causal_norm_the_measured_future_leak_is_zero(tiny_kwargs):
    assert _relative_leak(tiny_kwargs, causal_norm=True) < _LEAK_TOL


def test_without_causal_norm_the_probe_detects_the_leak(tiny_kwargs):
    """The probe is only worth having if a leaky encoder fails it. Without this, a broken
    perturbation or a wrong index would make the causality test above pass on any model."""
    assert _relative_leak(tiny_kwargs, causal_norm=False) > _LEAK_FLOOR
