r"""Raw-domain loss terms for the raw-signal VAE-TEB v4 model (Sprint 3).

All terms operate on the raw future block $(B, T_{\mathrm{valid}}, H, R)$ and share a per-sample
forecast mask $m \in \{0,1\}^{B \times T_{\mathrm{valid}} \times H \times R}$ (from
``raw_masks.forecast_mask``). Every reduction is masked, ``nan_to_num``-sanitised (so a NaN/sentinel
gap at a masked position cannot poison the sum via $\text{NaN}\times 0 = \text{NaN}$), and uses a
``clamp_min(1)`` denominator.

Provided:
- :func:`raw_nll` -- masked heteroscedastic Gaussian NLL (learned variance) + the ``mean_logvar``
  variance-collapse diagnostic (G7 / §9).
- :func:`raw_mae` -- masked mean absolute forecast error, a scale-free reported diagnostic.
- :func:`kld_terms` -- a thin adapter over the inherited v3 KL machinery (anchor support G3, free bits,
  honest ``kld_raw``/``kld_train``/``kld_active_frac`` reporting G4).
- :func:`lowpass_loss` / :func:`smooth_loss` -- multi-scale block-average error and a first-difference
  $\ell_1$ term, so the model is scored on the deceleration *trend* and *slope* rather than punished for
  unpredictable per-sample jitter (§10).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch


def _sanitize(target: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries with $0$ so masked-out positions cannot poison a masked reduction."""
    return torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)


def _masked_mean(
    x: torch.Tensor, m: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    r"""Mask-weighted mean $\sum (x\,m) / \max(\sum m, 1)$ -- the shared empty-mask convention.

    Reduces over ``dim`` (all axes when ``None``). A fully-masked reduction returns $0$ (numerator
    $0$ over a denominator clamped to $1$). Centralises the ``clamp_min(1.0)`` policy so every raw
    reduction agrees on how an all-invalid block is scored.
    """
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(1.0)
    return num / den


def raw_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Masked heteroscedastic Gaussian NLL over the raw future block.

    Per element (in nats):

    $$
    \tfrac12 (x^+ - \mu)^2 e^{-\ell} + \tfrac12 \ell,
    $$

    reduced as a mask-weighted mean. Unlike the v3 feature NLL, the mask is the **full**
    $(B, T_{\mathrm{valid}}, H, R)$ per-sample forecast mask (future validity is per raw sample), so the
    denominator is $\sum m$ directly (no extra channel factor).

    Args:
        mu: Predicted mean $(B, T_{\mathrm{valid}}, H, R)$.
        logvar: Predicted (smooth-bounded) log-variance, same shape.
        target: Raw future target $X^+$, same shape (may contain NaN/sentinel at masked positions).
        mask: Per-sample forecast mask $(B, T_{\mathrm{valid}}, H, R)$, values in $\{0, 1\}$.

    Returns:
        ``(loss, mean_logvar)`` -- the masked NLL scalar and the masked-mean log-variance (the
        variance-collapse early-warning diagnostic).
    """
    target = _sanitize(target)
    diff2 = (mu - target) ** 2
    per_elem = 0.5 * diff2 * torch.exp(-logvar) + 0.5 * logvar
    # One shared denominator for both outputs (same ``clamp_min(1.0)`` masked-mean convention as
    # :func:`_masked_mean`); kept explicit here to avoid reducing the mask twice on the hot path.
    denom = mask.sum().clamp_min(1.0)
    loss = (per_elem * mask).sum() / denom
    mean_logvar = (logvar * mask).sum() / denom
    return loss, mean_logvar


def raw_mae(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    r"""Masked mean absolute error over the raw future block (a scale-free forecast diagnostic).

    A plain, likelihood-free companion to :func:`raw_nll`: the mask-weighted mean of
    $|\mu - x^+|$ over the $(B, T_{\mathrm{valid}}, H, R)$ forecast block. Reported (not optimised)
    so the trainer surfaces forecast error in the target's own (normalised) units, comparable across
    runs regardless of the learned variance.

    Args:
        mu: Predicted mean $(B, T_{\mathrm{valid}}, H, R)$.
        target: Raw future target $X^+$, same shape (may contain NaN/sentinel at masked positions).
        mask: Per-sample forecast mask $(B, T_{\mathrm{valid}}, H, R)$, values in $\{0, 1\}$.

    Returns:
        The masked mean absolute error scalar.
    """
    return _masked_mean((mu - _sanitize(target)).abs(), mask)


def kld_terms(
    model,
    forward_outputs: Dict[str, torch.Tensor],
    *,
    weight: Optional[torch.Tensor] = None,
    free_bits: float = 0.0,
    compute_kld_loss: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Adapter over the inherited v3 KL machinery (anchor support G3, honest reporting G4).

    Returns ``(kld_train, kld_raw, kld_active_frac)``:
    - ``kld_train`` -- the free-bit-floored, anchor-supported mean that enters the loss;
    - ``kld_raw`` -- the un-floored (same support) detached TE surrogate ($\le$ ``kld_train``);
    - ``kld_active_frac`` -- the forward's fraction of active latent dims.

    Args:
        model: The :class:`SeqVaeRawV4` (its inherited ``_kld_loss`` honours ``self.kld_support``).
        forward_outputs: The forward dict (needs ``mu_prior``/``logvar_prior``/``mu_post``/``logvar_post``).
        weight: Optional per-anchor validity $(B, T)$ (the low-rate KL mask).
        free_bits: Per-dim free-bit floor for ``kld_train``.
        compute_kld_loss: If False, both KL terms are $0$ (ablation / perm readout).

    Returns:
        ``(kld_train, kld_raw, kld_active_frac)`` scalars.
    """
    ref = forward_outputs["mu_prior"]
    if not compute_kld_loss:
        zero = ref.new_zeros(())
        return zero, zero, forward_outputs.get("kld_active_frac", zero)

    kld_train = model._kld_loss(
        mu_prior=forward_outputs["mu_prior"],
        logvar_prior=forward_outputs["logvar_prior"],
        mu_post=forward_outputs["mu_post"],
        logvar_post=forward_outputs["logvar_post"],
        reduce_mean=True,
        weight=weight,
        free_bits=free_bits,
    )
    with torch.no_grad():
        kld_raw = model._kld_loss(
            mu_prior=forward_outputs["mu_prior"],
            logvar_prior=forward_outputs["logvar_prior"],
            mu_post=forward_outputs["mu_post"],
            logvar_post=forward_outputs["logvar_post"],
            reduce_mean=True,
            weight=weight,
            free_bits=0.0,
        )
    kld_active_frac = forward_outputs.get("kld_active_frac", ref.new_zeros(()))
    return kld_train, kld_raw, kld_active_frac


def _masked_block_average(
    x: torch.Tensor, m: torch.Tensor, block: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked block-average of ``x`` (and its block-validity) over the last axis, block width ``block``.

    Drops the trailing remainder if the axis length is not a multiple of ``block``. A block's average
    uses only its valid samples; a block with no valid sample is marked invalid.
    """
    n = x.size(-1)
    block = max(1, min(block, n))
    nb = n // block
    x = x[..., : nb * block].reshape(*x.shape[:-1], nb, block)
    m = m[..., : nb * block].reshape(*m.shape[:-1], nb, block)
    avg = _masked_mean(x, m, dim=-1)
    block_valid = (m.sum(-1) > 0).to(x.dtype)
    return avg, block_valid


def lowpass_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scales_sec: Sequence[int] = (4, 16, 32, 60),
    fs: int = 4,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    r"""Multi-scale masked block-average squared error over the raw forecast axis.

    For each scale $q$ (seconds) the raw axis $H R$ is partitioned into blocks of width $q\,f_s$ samples
    and the masked block-averages of prediction and target are compared:

    $$
    \mathcal L_{\mathrm{lowpass}} = \sum_q \omega_q\, \big\| \mathcal A_q(\hat x) - \mathcal A_q(x^+) \big\|^2_{2, \mathrm{masked}}.
    $$

    Args:
        mu: Predicted mean $(B, T_{\mathrm{valid}}, H, R)$.
        target: Raw future target, same shape.
        mask: Per-sample forecast mask, same shape.
        scales_sec: Block-average scales in seconds.
        fs: Raw sampling rate (Hz), so block width $= q\,f_s$ samples.
        weights: Optional per-scale weights (default: uniform $1$).

    Returns:
        The scalar multi-scale low-pass loss.
    """
    b, tv, h, r = mu.shape
    mu_f = mu.reshape(b, tv, h * r)
    tg_f = _sanitize(target).reshape(b, tv, h * r)
    m_f = mask.reshape(b, tv, h * r)
    if weights is None:
        weights = [1.0] * len(scales_sec)

    total = mu.new_zeros(())
    for scale, w in zip(scales_sec, weights):
        block = int(round(scale * fs))
        mu_avg, bv = _masked_block_average(mu_f, m_f, block)
        tg_avg, _ = _masked_block_average(tg_f, m_f, block)
        err = (mu_avg - tg_avg) ** 2
        total = total + float(w) * _masked_mean(err, bv)
    return total


def smooth_loss(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    r"""Masked first-difference $\ell_1$ loss over the raw forecast axis.

    $$
    \mathcal L_{\Delta} = \big\| \Delta \hat x - \Delta x^+ \big\|_{1, \mathrm{masked}},
    $$

    where $\Delta$ is the first difference along the flattened $H R$ raw axis and a difference is
    counted only where **both** of its endpoints are valid.

    Args:
        mu: Predicted mean $(B, T_{\mathrm{valid}}, H, R)$.
        target: Raw future target, same shape.
        mask: Per-sample forecast mask, same shape.

    Returns:
        The scalar first-difference $\ell_1$ loss.
    """
    b, tv, h, r = mu.shape
    mu_f = mu.reshape(b, tv, h * r)
    tg_f = _sanitize(target).reshape(b, tv, h * r)
    m_f = mask.reshape(b, tv, h * r)
    dmu = mu_f[..., 1:] - mu_f[..., :-1]
    dtg = tg_f[..., 1:] - tg_f[..., :-1]
    dm = m_f[..., 1:] * m_f[..., :-1]
    return _masked_mean((dmu - dtg).abs(), dm)
