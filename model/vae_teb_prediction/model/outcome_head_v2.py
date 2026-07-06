r"""Supervised outcome head for ``SeqVaeLagAttnV2`` (arch spec section 22).

A default-off, separable Stage-4 classifier that pools the per-timestep
transfer-entropy representation of a frozen (or lightly fine-tuned) variational
model into a segment-level outcome prediction. Kept in its own module (it consumes
only a forward dict, so it imports nothing from the core model) to keep the main
``vae_teb_lag_attn_v2`` module manageable; the core model imports
:class:`OutcomeHead` and constructs it only when ``use_outcome_head=True``.

Per section 22, the per-step classification representation is
$$r_t^{\mathrm{cls}} = [\,h^y_t \mid z_t \mid K_t \mid \bar\ell_t \mid \sigma^2_{\ell,t}\,],$$
attention-pooled over the valid time steps into a segment vector $r^{\mathrm{seg}}$
that feeds a binary / multi-class classifier. Labels never enter the variational
``forward`` -- they are supervised targets only.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG_INF = -1e9


class OutcomeHead(nn.Module):
    r"""Attention-pooled segment classifier over the TE representation (section 22).

    Builds $r_t^{\mathrm{cls}} = [h^y_t \mid z_t \mid K_t \mid \bar\ell_t \mid
    \sigma^2_{\ell,t}]$ from the model forward dict, pools it over valid steps with
    a learned attention weight
    $$a^{\mathrm{cls}}_t = w_{\mathrm{cls}}^\top \tanh(W_{\mathrm{cls}} r_t^{\mathrm{cls}}),
    \qquad \omega_t = \operatorname{softmax}_t(a^{\mathrm{cls}}_t + \log w_t),$$
    and classifies $r^{\mathrm{seg}} = \sum_t \omega_t r_t^{\mathrm{cls}}$.

    The expected lag $\bar\ell_t = \sum_\ell \ell\,\bar\alpha_{t,\ell}$ and lag
    variance $\sigma^2_{\ell,t} = \sum_\ell (\ell-\bar\ell_t)^2 \bar\alpha_{t,\ell}$
    are computed here from the full head-averaged posterior
    $\bar\alpha_{t,\ell} = \frac{1}{M}\sum_m \alpha^{(m)}_{t,\ell}$ (``attn_weights``),
    so the head is faithful to section 22 and independent of the active-set
    truncation.
    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        d_z: int = 24,
        num_lags: int = 91,
        num_classes: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the outcome head.

        Args:
            d_model: Target-state width $d$ of ``target_state``.
            d_z: Latent width $d_z$ of ``z``.
            num_lags: Number of lag bins $L$ (for the expected-lag statistics).
            num_classes: Output classes. ``1`` uses a single-logit BCE head; a
                value $\ge 2$ uses a softmax / cross-entropy head.
            hidden_dim: Attention-scorer hidden width (defaults to ``d_model``).
            dropout: Dropout on the pooled segment representation.
        """
        super().__init__()
        self.num_lags = int(num_lags)
        self.num_classes = int(num_classes)
        # r_cls = [h^y | z | K_t | ell_bar | sigma2_ell]; the last three are scalars.
        self.feat_dim = int(d_model) + int(d_z) + 3
        hidden = int(hidden_dim) if hidden_dim is not None else int(d_model)
        # Attention pooling: a_t = w^T tanh(W r_cls).
        self.attn_proj = nn.Linear(self.feat_dim, hidden)
        self.attn_score = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)
        self.register_buffer(
            "_lag_index", torch.arange(self.num_lags, dtype=torch.float32)
        )

    def build_features(self, fo: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Assemble $r_t^{\mathrm{cls}}$ of shape ``(B, T, feat_dim)`` from ``fo``.

        Args:
            fo: A source-path forward dict with ``target_state (B,T,d)``,
                ``z (B,T,d_z)``, ``kld_per_t (B,T)``, and ``attn_weights (B,T,M,L)``.

        Returns:
            The per-step classification features ``(B, T, feat_dim)``.
        """
        h_y = fo["target_state"]                       # (B,T,d)
        z = fo["z"]                                     # (B,T,d_z)
        k_t = fo["kld_per_t"].unsqueeze(-1)             # (B,T,1)
        alpha = fo["attn_weights"]                      # (B,T,M,L)
        alpha_bar = alpha.mean(dim=2)                   # (B,T,L) head-averaged
        lag = self._lag_index.to(alpha_bar.dtype)       # (L,)
        ell_bar = (alpha_bar * lag).sum(-1)             # (B,T)
        var = (alpha_bar * (lag - ell_bar.unsqueeze(-1)) ** 2).sum(-1)   # (B,T)
        return torch.cat(
            [h_y, z, k_t, ell_bar.unsqueeze(-1), var.unsqueeze(-1)], dim=-1
        )

    def forward(
        self, fo: Dict[str, torch.Tensor], valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Return segment logits ``(B, num_classes)`` from the forward dict.

        Args:
            fo: The model forward dict (source path).
            valid_mask: Optional ``(B, T)`` weight in ``[0, 1]`` (warm-up
                exclusion times the dataset weight). Zero-weight steps are removed
                from the attention pooling.

        Returns:
            Class logits ``(B, num_classes)``.

        Raises:
            ValueError: If any row of ``valid_mask`` has no valid step. Silently
                falling back to a uniform average over every step (including
                un-warmed-up ones) would produce a segment logit built from
                semantically invalid representations with no indication of the
                problem.
        """
        r_cls = self.build_features(fo)                        # (B,T,F)
        scores = self.attn_score(torch.tanh(self.attn_proj(r_cls))).squeeze(-1)  # (B,T)
        if valid_mask is not None:
            m = valid_mask.to(scores.dtype)
            if bool((m > 0.0).sum(dim=1).eq(0).any()):
                raise ValueError(
                    "OutcomeHead.forward: at least one batch row has no valid "
                    "(warm-up-passed, nonzero-weight) time step; the segment "
                    "cannot be pooled. Check that the sequence length exceeds "
                    "warmup_period and the dataset weight is not all-zero."
                )
            scores = scores.masked_fill(m <= 0.0, _NEG_INF)
        omega = F.softmax(scores, dim=1)                       # (B,T)
        r_seg = (omega.unsqueeze(-1) * r_cls).sum(dim=1)       # (B,F)
        return self.classifier(self.drop(r_seg))               # (B,num_classes)


def outcome_loss(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    r"""Supervised classification loss (arch spec section 22).

    Cross-entropy for ``num_classes >= 2``; single-logit binary cross-entropy for
    ``num_classes == 1``.

    Args:
        logits: ``(B, num_classes)`` class logits.
        labels: ``(B,)`` integer class labels (CE) or ``(B,)`` / ``(B, 1)`` float
            targets in $\{0, 1\}$ (BCE).
        num_classes: Number of output classes.

    Returns:
        A scalar loss.
    """
    if int(num_classes) == 1:
        target = labels.to(logits.dtype).view(-1)
        return F.binary_cross_entropy_with_logits(logits.view(-1), target)
    return F.cross_entropy(logits, labels.long().view(-1))
