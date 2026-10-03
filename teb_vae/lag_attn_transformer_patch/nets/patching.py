"""Raw-to-patch tokens, the per-patch target summaries, and the patch embedding.

Token ``t`` reads raw samples ``[R t, R t + R)`` and nothing later. The featurization is the e2e
package's ``featurize``, imported, not copied.
"""
from __future__ import annotations

import torch
from torch import nn

from teb_vae.lag_attn.nets.encoders import START_EMBED_STD, AvailabilityInputAdapter
from teb_vae.lag_attn_transformer_e2e.nets.frontend import featurize


def patchify(
    raw: torch.Tensor, weight: torch.Tensor, *, raw_per_step: int, validity: str = "fhr_weight"
) -> torch.Tensor:
    """Cut a raw stream into tokens ``[value (R), delta (R), m_t - 1]``.

    ``m_t`` is the minimum of the per-sample mask over the patch. The validity channel is
    ``m_t - 1`` so that an all-zero stream means valid-and-flat (the null the source-null control
    feeds), and no invalid token reaches a norm layer as an exact zero vector.

    Args:
        raw: Loader-normalized signal ``(B, L)`` with ``L = T * raw_per_step``.
        weight: Decimated validity ``(B, T)``.
        raw_per_step: Samples per token ``R``.
        validity: ``"fhr_weight"`` masks with ``weight``; ``"finite"`` masks only non-finite samples.

    Returns:
        ``(B, T, 2R + 1)`` in ``raw``'s dtype, all finite.
    """
    if validity == "fhr_weight":
        w = weight
    elif validity == "finite":
        w = torch.ones_like(weight)
    else:
        raise ValueError(f"validity must be 'fhr_weight' or 'finite', got {validity!r}")
    steps = int(weight.shape[-1])
    if int(raw.shape[-1]) != steps * raw_per_step:
        raise ValueError(
            f"raw length {int(raw.shape[-1])} != T * raw_per_step = {steps} * {raw_per_step}"
        )
    shape = (int(raw.shape[0]), steps, raw_per_step)
    value, mask, delta = (c.reshape(shape) for c in featurize(raw, w).unbind(1))
    return torch.cat((value, delta, mask.amin(-1, keepdim=True) - 1.0), dim=-1)


def patch_summaries(values: torch.Tensor, valid: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Per-patch ``[level, variability]``, the one definition of the forecast target.

    ``level`` is the mean of the ``R`` values; ``variability`` is ``log(rms(d) + eps)`` with ``d``
    the ``R - 1`` first differences inside the patch (never the boundary-crossing ``delta[..., 0]``).
    Invalid cells (``valid < 0.5``) are 0.0 in both channels. Not differentiable at a flat patch;
    it is a target, not a model output.

    Args:
        values: Patch values ``(..., R)``.
        valid: Token validity ``(...)``, 1 valid, 0 invalid.
        eps: Variability floor, in the units of ``values``.

    Returns:
        ``(..., 2)``.
    """
    level = values.mean(-1)
    variability = torch.log(torch.diff(values, dim=-1).square().mean(-1).sqrt() + eps)
    out = torch.stack((level, variability), dim=-1)
    return torch.where(valid.unsqueeze(-1) >= 0.5, out, out.new_zeros(()))


class PatchEmbedding(nn.Module):
    """``AvailabilityInputAdapter`` plus one learned ``missing`` embedding for invalid tokens.

    Shapes:
        Input:  ``(B, T, in_dim)`` from :func:`patchify` (last channel ``m_t - 1``)
        Output: ``(B, T, d_model)``
    """

    def __init__(self, *, in_dim: int, d_model: int, sequence_length: int, dropout: float) -> None:
        super().__init__()
        self.adapter = AvailabilityInputAdapter(
            in_dim=in_dim,
            d_model=d_model,
            sequence_length=sequence_length,
            dropout=dropout,
            delays=None,
        )
        # A bare Parameter: the generic `initialization` pass only walks Linear/Conv/LSTM/LayerNorm.
        self.missing = nn.Parameter(torch.randn(d_model) * START_EMBED_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed every token, then swap in ``missing`` where ``m_t = 0``; no data-dependent branch."""
        e = self.adapter(x)
        return torch.where(x[..., -1:] + 1.0 > 0.5, e, self.missing.to(e.dtype))
