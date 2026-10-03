r"""Grad-CAM for the lag-attentive forecasters: where in time one anchor's readout looked.

Grad-CAM (Selvaraju et al., 2017) explains a scalar output $f$ through the activations
$A \in \mathbb R^{T \times D}$ of one layer. Each feature channel $d$ gets one weight, the mean
gradient of $f$ over the time steps the readout reads, and the map is the positive part of the
weighted channel sum:

$$
w_d = \frac{1}{|\mathcal R|} \sum_{t \in \mathcal R} \frac{\partial f}{\partial A_{t,d}},
\qquad
\mathrm{CAM}_t = \mathrm{ReLU}\Bigl(\sum_d w_d A_{t,d}\Bigr), \qquad t \in \mathcal R .
$$

$\mathcal R$ is the set of steps where the gradient is not zero. On these causal models that set
never reaches past the anchor $t_a$. The ReLU keeps only the evidence that *raises* $f$, so the
meaning of a peak follows the readout: a larger divergence $K_t$ for ``kld``, a larger advantage of
the full branch for ``pred_gap``, and a *worse* score for ``nll_full`` and ``mse_full``.

**Three views per anchor**, one per place where this architecture keeps a time axis:

* ``target`` -- the target stream at the **input of the target encoder's last attention block**.
  The prior and posterior heads are per-step, so the readout at $t_a$ reads the encoder's *output*
  only at $t_a$: a map there is a spike at lag $0$ by construction. The input of the last
  attention block is the deepest target layer whose earlier steps the readout still reads. An
  encoder without attention blocks (the conv-LSTM cell) falls back to the encoder's own input.
* ``source`` -- the source representation the lag attention reads as keys and values, the
  forward's ``source_state``.
* ``attention`` -- the lag-attention weights $\alpha^{(m)}_{t_a,\ell}$ at the anchor, read where
  they enter the attended summaries (the output of ``lag_attn.attn_dropout``), scored as
  gradient-weighted attention (Chefer et al., 2021):
  $\mathrm{CAM}_\ell = \frac1M \sum_m \mathrm{ReLU}\bigl(\alpha^{(m)}_{t_a,\ell}\,
  \partial f / \partial \alpha^{(m)}_{t_a,\ell}\bigr)$. It is the model's own lag axis, weighted
  by how much each lag's attention moved the readout.

Every map is re-indexed by offset from the anchor, $\ell = t_a - t$, and normalised to sum to one
per row, so that class means compare *where* the readout looked rather than how large it is. The
unnormalised sum travels beside it as ``total``.

Grad-CAM needs one forward and one backward per readout, against the $n$ forwards of an
integrated-gradient call. It has no baseline and no completeness property: it is a first-order,
local summary at the observed input. Read it beside the integrated gradients, not instead of them.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import attributions as core

#: The three views, in the order every table and figure lists them.
VIEWS: Tuple[str, ...] = ("target", "source", "attention")

#: One line per view for a figure title.
VIEW_TITLES: Dict[str, str] = {"target": "target encoder", "source": "source K/V", "attention": "lag attention"}


def target_layer(model: nn.Module) -> Tuple[nn.Module, str]:
    """The module whose **input** is the ``target`` view.

    Args:
        model: A lag-attentive model with a ``target_encoder``.

    Returns:
        ``(module, label)``: the last attention block of the target encoder, or the encoder itself
        when it has no attention block.
    """
    encoder = model.target_encoder
    blocks = getattr(encoder, "attention_blocks", None)
    if blocks is not None and len(blocks):
        return blocks[-1], "target_encoder.attention_blocks[-1] input"
    return encoder, "target_encoder input"


def temporal_cam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    r"""The Grad-CAM map of one time-major layer, one row per sample.

    Args:
        activations: $A$, $(N, T, D)$.
        gradients: $\partial f / \partial A$, $(N, T, D)$.

    Returns:
        $\mathrm{CAM}$, $(N, T)$, zero outside each row's read steps $\mathcal R$.
    """
    read = (gradients.abs().sum(dim=-1) > 0).to(gradients.dtype)                 # (N, T)
    weights = (gradients * read[..., None]).sum(dim=1) / read.sum(dim=1).clamp(min=1.0)[:, None]
    return torch.relu((activations * weights[:, None, :]).sum(dim=-1)) * read


def normalise(cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split per-row maps into a profile that sums to one and its total.

    Args:
        cam: $(N, W)$ non-negative maps, ``NaN`` where an offset falls before the record.

    Returns:
        ``(profile, total)``: $(N, W)$ with ``NaN`` rows where the total is zero, and $(N,)$.
    """
    total = np.nansum(cam, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(total[:, None] > 0.0, cam / total[:, None], np.nan)
    return profile, total


def centroid(profile: np.ndarray, axis_seconds: np.ndarray) -> np.ndarray:
    r"""The centre of mass $\sum_\ell p_\ell \tau_\ell$ of each normalised row, in seconds."""
    return np.nansum(profile * np.asarray(axis_seconds)[None, :], axis=1) / np.where(
        np.isfinite(profile).any(axis=1), 1.0, np.nan
    )


def gradcam(
    wrapper: core.AnchorReadout,
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    columns: torch.Tensor,
    *,
    coordinates: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    r"""Grad-CAM of one readout in the three views, one row per anchor.

    Rows are independent copies of their segments (:func:`~.attributions.expand_rows`), so one
    backward of $\sum_i f_i$ gives every row its own gradient.

    Args:
        wrapper: The readout module, on a dense-latent (lag-attentive) cell.
        inputs: The row-expanded ``(y_st, y_ph, u_stream)``.
        extra: The row-expanded ``(target_features, weight)``.
        columns: Per-row anchor-axis positions, $(N,)$.
        coordinates: Per-row latent coordinate, or ``None``.

    Returns:
        ``value`` $(N,)$, the readout at the input; per view ``<view>`` the normalised profile by
        offset from the anchor -- $(N, T)$ for ``target``, $(N, L)$ for ``source`` and
        ``attention`` -- and ``<view>_total`` $(N,)$.
    """
    model = wrapper.model
    n_rows = int(columns.shape[0])
    if coordinates is None:
        coordinates = torch.zeros(n_rows, dtype=torch.long, device=columns.device)
    captured: Dict[str, torch.Tensor] = {}

    def keep(view: str, tensor: torch.Tensor) -> None:
        """Store the first call's tensor. Returns ``None``: a hook's return value replaces the
        module's input or output."""
        captured.setdefault(view, tensor)

    layer, _label = target_layer(model)
    hooks = [
        layer.register_forward_pre_hook(lambda _module, args: keep("target", args[0])),
        model.register_forward_hook(lambda _module, _args, out: keep("source", out["source_state"])),
        # The weights *on the graph*: the forward's ``attn_weights`` is a lag-order flip of this
        # window-order tensor, a copy no readout depends on, so its gradient would be empty.
        model.lag_attn.attn_dropout.register_forward_hook(lambda _module, _args, out: keep("attention", out)),
    ]
    try:
        with torch.enable_grad():
            values = wrapper(*(x.detach().requires_grad_(True) for x in inputs), *extra, columns, coordinates)
            tensors = [captured[view] for view in VIEWS]
            grads = torch.autograd.grad(values.sum(), tensors, allow_unused=True)
    finally:
        for hook in hooks:
            hook.remove()
    grads = [torch.zeros_like(t) if g is None else g for t, g in zip(tensors, grads)]
    anchors = wrapper.anchor_steps(columns)
    rows = torch.arange(n_rows, device=columns.device)
    n_lags = int(captured["attention"].shape[-1])
    target_cam = temporal_cam(tensors[0].detach(), grads[0])
    source_cam = temporal_cam(tensors[1].detach(), grads[1])
    # Window position j is lag L - 1 - j, so a flip on the last axis puts both in lag order.
    alpha = tensors[2].detach()[rows, anchors].flip(-1)                               # (N, M, L)
    alpha_grad = grads[2][rows, anchors].flip(-1)
    attention_cam = torch.relu(alpha * alpha_grad).mean(dim=1)
    anchor_list = anchors.detach().cpu().numpy()
    host = lambda tensor: tensor.detach().cpu().to(torch.float64).numpy()  # noqa: E731
    out: Dict[str, np.ndarray] = {"value": host(values)}
    for view, cam in (
        ("target", core.lag_profile(host(target_cam), anchor_list, int(target_cam.shape[1]))),
        ("source", core.lag_profile(host(source_cam), anchor_list, n_lags)),
        ("attention", host(attention_cam)),
    ):
        out[view], out[f"{view}_total"] = normalise(cam)
    return out


def offset_seconds(width: int) -> np.ndarray:
    """The plain stored-step offset axis of the ``target`` view, in seconds."""
    return np.arange(int(width), dtype=np.float64) * float(SECONDS_PER_STEP)


__all__ = ["VIEWS", "VIEW_TITLES", "centroid", "gradcam", "normalise", "offset_seconds", "target_layer", "temporal_cam"]
